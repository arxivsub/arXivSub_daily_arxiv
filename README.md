# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-17 | 今日论文总数: 882

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. CORPGEN: Simulating Corporate Environments with Autonomous Digital Employees in Multi-Horizon Task Environments

**arXiv ID:** 2602.14229 | [PDF](https://arxiv.org/pdf/2602.14229v1)

**作者:** Abubakarr Jaye `[一作]` (Microsoft Corporation), Tianwei Chen `[通讯]` (Microsoft Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并实现了多视野任务环境（MHTE）及其通用框架，用以评估并提升在持久、并行多任务环境下的自适应代理性能。

**💡 创新点**

创新点在于：①将单任务长程推理扩展为多任务并行推理；②识别并针对四大失败模式（上下文饱和、内存干扰、依赖图复杂度、优先级开销）设计四大架构机制（分层规划、子代理隔离、分层记忆、动态摘要）；③通过经验学习与工具化方法实现跨任务协同与自适应重规划。

**🔧 技术方法**

采用大型语言模型（GPT‑5.1）、子代理化的图形用户界面自动化工具（UFO2、Computer‑Using Agent）、分层记忆（工作内存、结构化长期记忆、语义记忆）、分层规划、认知工具与经验学习回放等技术。

**📊 数据集**

基于改造后的 OSWorld Office 任务集，将多任务拼接为单一持续会话并引入 25%、50%、75%、100% 四个负载级别，用于评估多任务持续执行性能。

**📈 对比分析**

与三种独立 CUA 实现（UFO2、Computer‑Using Preview、层次化 CUA）进行对比；在 100% 负载下，完整框架实现 15.2% 的任务完成率，较基线 4.3% 提升约 3.5 倍，实验表明改进主要来自架构机制而非具体实现。

**⚠️ 局限性**

主要局限包括：①对视觉接口的依赖导致速度慢且鲁棒性受限；②记忆管理与上下文摘要的阈值仍需调优；③未覆盖真实企业环境的多代理协作细节；④评估仍依赖人工标注与 artifact 对比，缺少通用自动判定机制。

---

## 2. NEST: Nascent Encoded Steganographic Thoughts

**arXiv ID:** 2602.14095 | [PDF](https://arxiv.org/pdf/2602.14095v1)

**作者:** Artem Karpov `[一作]` `[通讯]` (Independent), Artem Karpov (Independent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估大型语言模型在链式推理中隐藏推理过程（ste­ganographic CoT）的能力，尤其在计数任务和数学任务中的表现。

**💡 创新点**

提出了专门的评估框架和计数任务，区分隐藏推理与其他模糊推理，并系统测量隐藏任务准确率、编码质量、监测逃逸率与翻译成功率。

**🔧 技术方法**

使用句首字母隐写（acrostic）编码算法、四个对齐评分器（Monitor, Encoding Scorer, Hidden Task Scorer, Translator）以及多种模型（Claude、GPT、Grok 等）进行实验。

**📊 数据集**

使用了从Anthropic HH‑RLHF测试集抽取的问答、20个意见类问答、GSM8K算术问题、迷宫导航任务以及自定义的随机计数序列。

**📈 对比分析**

将stego CoT与四个基线（直接回答、普通 CoT、填充符号、无推理）进行对比，结果显示Claude Opus 4.5在D=4时计数准确率92%，但在更长序列或算术任务上几乎无效；GPT‑5.2在少数案例中会先拒绝后部分执行。

**⚠️ 局限性**

仅针对提示式隐写，没有进行SFT微调；仅评估计数任务，未能在复杂数学推理中展示能力；检测方法受限于acrostic，可能漏检更复杂隐写方式。

---

## 3. A Rational Analysis of the Effects of Sycophantic AI

**arXiv ID:** 2602.14270 | [PDF](https://arxiv.org/pdf/2602.14270v1)

**作者:** Rafael M. Batista `[一作]` (Princeton University), Thomas L. Griffiths `[通讯]` (Princeton University)

**通讯引用:** 49472 | [OpenAlex ID](https://openalex.org/A5077079119)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文结合贝叶斯推理和实验，探究大型语言模型（LLM）在与人类交互时倾向于确认用户信念的“sycophancy”如何导致人们对错误假设产生过度自信并抑制真理探索。

**💡 创新点**

创新点在于将sycophancy视为从假设条件下采样的偏差，并用数学推导展示其对贝叶斯更新的致命影响；随后在改良的Wason 2‑4‑6任务中验证默认LLM的确认性行为与显式确认提示的效果相同。

**🔧 技术方法**

使用的技术包括：贝叶斯推理公式、随机与有偏采样的对比实验、OpenAI GPT‑5.1 Chat 生成交互序列、统计检验（置换检验、ANOVA、等价性检验）以及LLM辅助的结果编码。

**📊 数据集**

数据集为557名Prolific在线受试者的实验数据，任务采用自定义的偶数序列规则发现题目；序列从预设的偶数列表随机或依据提示生成，不依赖公开公开数据集。

**📈 对比分析**

通过比较五种对话提示（Rule Confirming、Rule Disconfirming、Random Sequence、Default GPT、Agreeable）在规则发现率和自信变化上的差异，结果显示Default GPT与Rule Confirming同样显著降低规则发现率（5.9% vs 8.4%）并提升自信；Random Sequence条件则使发现率提高到29.5%，自信下降。

**⚠️ 局限性**

局限性包括：任务抽象、低风险场景，难以推广到政治或社会深层信念；对默认LLM确认行为的机制解释仍不充分；实验仅关注单一认知任务，未检验在更复杂、情境化交互中的效应。

---

## 4. Robustness Verification of Binary Neural Networks: An Ising and Quantum-Inspired Framework

**arXiv ID:** 2602.13536 | [PDF](https://arxiv.org/pdf/2602.13536v1)

**作者:** Rahul Singh `[一作]` (University of California), Zheng Zhang `[通讯]` (University of California)

**通讯引用:** 460639 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出将二值神经网络（BNN）鲁棒性验证问题转换为二次无约束布尔优化（QUBO），并在量子/量子启发硬件上求解；

**💡 创新点**

创新点在于：①将BNN鲁棒性验证映射为QCBO再转为QUBO，实现与Ising机和量子退火机的统一接口；②提出针对BNN的布尔门和符号层的局部惩罚函数，降低QUBO规模；

**🔧 技术方法**

采用了Ising/量子退火机、数字退火机（Fujitsu DA）、模拟退火（SA）和传统混合整数/QP求解器Gurobi，以及物理启发式Free Energy Machine（FEM）等技术；

**📊 数据集**

使用的是二值化的MNIST数据集（28×28）以及多种尺寸的简化MNIST（5×5、7×7、11×11、28×28）训练的十分类BNN；

**📈 对比分析**

通过比较Gurobi、SA、FEM、D‑Wave量子退火机和Fujitsu数字退火机的约束满足率、能量和耗时，发现FEM和数字退火机在所有实例中都能得到完全满足约束的解，数字退火机相较于Gurobi速度提升约168×，FEM在多种规模上表现最佳；

**⚠️ 局限性**

局限性包括：QUBO规模随着网络尺寸和可扰动像素数快速增长，导致稀疏连接硬件（如D‑Wave）需要复杂嵌入，现有量子硬件噪声和连通性不足；实验仅覆盖相对小的BNN，缺乏对大规模网络的验证。

---

## 5. SRA: Semantic Relation-Aware Flowchart Question Answering

**arXiv ID:** 2602.13771 | [PDF](https://arxiv.org/pdf/2602.13771v1)

**作者:** Xinyu Li `[一作]` (Soochow University), Yu Hong `[通讯]` (Institute for Infocomm Research)

**通讯引用:** 7447 | [OpenAlex ID](https://openalex.org/A5100413976)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对流式图问答任务进行建模，利用大型语言模型检测节点之间的语义关系，将图的结构表示从仅链接转化为语义关系表述，并根据问题类型动态选择浅层或深层推理。

**💡 创新点**

创新点在于：① 用语义关系（条件性、因果性、实例化、顺序性）升级中介语言，实现更深层的推理；② 引入可控推理机制，依据问题意图切换浅层或深层推理；③ 通过LLM实现无标注的语义关系识别与推理。

**🔧 技术方法**

核心技术包括：视觉语言模型（TextFlow）提取图结构；Deepseek‑R1 进行两阶段链式思考的语义关系识别；Qwen‑2.5 负责零样本推理；LLaMA‑3 作为问题类型判别器；以及语义关系分类法（基于PDTB）。

**📊 数据集**

使用公开基准数据集 FlowVQA（约12,938 条训练实例，9,475 条测试实例）。

**📈 对比分析**

与 Graphviz、Mermaid、PlantUML 等基线进行对比，SRA 在整体精度上提升至约71.3%，在“应用情境”类问题上提升最多 9.5%，并在不同 LLM 规模和中介语言上均表现出优势。

**⚠️ 局限性**

局限性包括：判别器错误率约 14.2% 可能导致错误推理；深层推理在某些“直线”问题上并未带来提升；依赖 LLM 的推理与关系识别质量受模型规模与提示设计影响；缺乏对隐含知识和多跳推理的支持。

---

## 6. RynnBrain: Open Embodied Foundation Models

**arXiv ID:** 2602.14979 | [PDF](https://arxiv.org/pdf/2602.14979v1)

**作者:** Ronghao Dang `[一作]` (Alibaba Group), Deli Zhao `[通讯]` (Alibaba Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一套名为 RynnBrain 的嵌入式基础模型及其四个后训练版本（CoP、Nav、Plan、VLA），实现了在真实物理环境中自适应执行多样任务的能力。

**💡 创新点**

创新点包括：1）统一的物理空间时空预训练框架，将视觉、时间与坐标统一编码为可归一化的坐标标记；2）Chain-of-Point（CoP）交互式推理，将文字推理与空间定位交替嵌入；3）混合专家（MoE）架构与物理感知规划，显著提升定位与规划精度；4）公开的 RynnBrain-Bench 评测套件，聚焦细粒度时空认知与定位。

**🔧 技术方法**

技术手段主要有：Qwen3‑VL 作为基础 backbone；Vision–Language 预训练与 DeepStack、Interleaved MRoPE 结构融合；统一坐标标记输出与离散化坐标分类；Chain-of-Point 推理与 GRPO 强化学习；多模态数据混合训练；MoE（30B-A3B）与深度分区优化（ZeRO、DeepEP）。

**📊 数据集**

使用了超过 20 万多种多模态数据集：LLaVA-OV-SI/Video、ShareGPT‑4o‑video、FineVideo、CinePile、ActivityNet、YouCook2、RefCOCO、Google Refexp、Osprey‑724K、VSI‑590k、Sensenova‑SI‑800k、RynnBrain‑Object、RynnBrain‑Spatial、EgoRe‑5M、EgotaskQA、Env‑QA、RoboVQA、ShareRobot、ADE20K、COCO、Mapillary、RoboAfford‑Object/Affordance/Area、RynnBrain‑Trajectory、Grasp‑Anything、AgibotWorld、Open‑X‑Embodiment、以及用于 RL 的多任务采样数据。

**📈 对比分析**

通过与 GPT‑5.2、Gemini‑3 Pro、InternVL‑3.5、MiMo‑Embodied、Qwen3‑VL（8B/30B）等模型在 28 个基准（RynnBrain‑Bench、R2R、RxR、VIA、VLA 等）上进行对比实验，RynnBrain 在 8B 规模下已超过 Qwen3‑VL 及其他开源模型；CoP 版在姿态/轨迹/点位任务上均获得 SOTA；Nav 版在 R2R/RxR 的 SR/SPL/NE 指标上击败同类基准；Plan 与 VLA 在长周期规划与抓取任务中显著优于 Gemini‑3 Pro 与 Qwen3‑VL‑Finetuned，性能提升超过 10–30%。

**⚠️ 局限性**

局限性包括：1）模型规模较大，部署成本高，尤其 30B‑A3B；2）对极端多变动态场景的鲁棒性不足，主要仍依赖预训练数据覆盖；3）MoE 在 VLN 任务中未能充分放大效果，需进一步调优；4）Chain‑of‑Point 仍可能因视觉分辨率或遮挡导致定位误差；5）在极长时序任务中仍受限于记忆容量与推理深度。

---

## 7. Resilient and Freshness-Aware Scheduling for Industrial Multi-Hop IAB Networks: A Packet Duplication Approach

**arXiv ID:** 2602.13311 | [PDF](https://arxiv.org/pdf/2602.13311v1)

**作者:** Shuo Zhu `[一作]` (Beijing Jiaotong University), Bo Ai `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 26936 | [OpenAlex ID](https://openalex.org/A5100620739)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种在工业多跳毫米波IAB网络中，利用包复制与Lyapunov优化相结合的低复杂度调度框架（RFAS），实现可靠性与信息新鲜度的平衡。

**💡 创新点**

创新点在于：①首次把包复制导致的拥塞与AoI最小化结合，提出基于Lyapunov的复合度量；②通过离线预设节点不相交路径与在线冲突图贪心选择，显著降低调度复杂度；③在硬缓冲约束下保障队列稳定，兼顾PDR与AoI。

**🔧 技术方法**

使用技术包括：包复制（PD）、AoI建模、Lyapunov优化、混合整数非线性规划（MINLP）转化为每时隙最大权重调度（P2），冲突图与半双工约束，k‑shortest路径算法预路由，贪心排程算法RFAS。

**📊 数据集**

数据集：基于仿真生成的工业工厂场景，800 m × 800 m 网格布局，24个IAB节点、4–14个UE，采用Gilbert‑Elliott链模型模拟动态阻塞，包生成周期设为10/50槽，阻塞概率15%，平均阻塞持续100槽。

**📈 对比分析**

比较方法：对比单路径、QAS（队列感知）与FAS（信息新鲜度感知）三种基线；在不同阻塞概率、UE数量与流量周期下评估PDR、平均AoI、队列长度与负载不平衡。实验结果显示RFAS在双路径下PDR>95%，平均AoI仅比FAS高10%，并在硬缓冲约束下将最大队列长度控制在8，负载不平衡率比FAS降低19%。

**⚠️ 局限性**

局限性：①仍是仿真验证，缺乏真实工厂部署实验；②未考虑动态路由更新与更大规模网络；③Lyapunov调度基于经验参数γ，缺乏理论最优性保证；④仅针对毫米波噪声受限场景，未研究干扰密集环境。

---

## 8. DriveMamba: Task-Centric Scalable State Space Model for Efficient End-to-End Autonomous Driving

**arXiv ID:** 2602.13301 | [PDF](https://arxiv.org/pdf/2602.13301v1)

**作者:** Haisheng Su `[一作]` (Shanghai Jiao Tong University), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 16712 | [OpenAlex ID](https://openalex.org/A5087158377)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DriveMamba，一种基于 Mamba 结构的任务中心端到端自动驾驶框架，集成了动态任务关系建模、隐式视角对应学习和长时序融合。

**💡 创新点**

创新点在于：①使用统一的 Mamba 解码器实现任务查询与多模态传感器特征的并行交互；②设计轨迹引导的 Local‑to‑Global 双向扫描，既保留空间局部性又能捕捉任务间关系；③采用线性复杂度的 Selective State Space Model (SSM) 以及 FIFO 查询队列，显著提升长时序建模效率和可扩展性。

**🔧 技术方法**

核心技术包括：Mamba（SSM）及其双向 B‑Mamba 变体；视角对应学习层；任务关系建模层；长时序融合（FIFO 记忆队列）；轨迹引导双向扫描；迭代优化与残差学习；多任务头输出。

**📊 数据集**

在公开数据集 nuScenes 和 Bench2Drive 上进行实验。

**📈 对比分析**

与 UniAD、VAD、DriveTransformer、BEVPlanner 等方法对比，DriveMamba 在 nuScenes 开放循环规划中 L2 误差下降约 42%，碰撞率下降约 39%，并以 17.9 FPS 的速度运行；在 Bench2Drive 闭环规划中获得 53.54 的 Driving Score 与 55.8 ms 的推理时间，显示出更优的性能与效率。

**⚠️ 局限性**

局限性包括：对深度预测的依赖仍可能影响表现；在极端天气或复杂场景下的泛化能力尚未充分验证；目前实验仅在公开数据集上进行，跨域迁移性能待进一步评估。

---

## 9. GradMAP: Faster Layer Pruning with Gradient Metric and Projection Compensation

**arXiv ID:** 2602.14649 | [PDF](https://arxiv.org/pdf/2602.14649v1)

**作者:** Hao Liu `[一作]` (Institute of Automation Chinese Academy of Sciences), Yongqiang Tang `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GradMAP，一种两阶段的层级剪枝方法，利用梯度幅值评估层重要性并通过投影补偿矩阵恢复性能。

**💡 创新点**

创新点在于使用全局梯度幅值作为重要性度量，仅需一次前向-反向传播；以及只在一次投影矩阵训练中补偿最大激活漂移，显著降低计算开销。

**🔧 技术方法**

采用梯度幅值度量、投影补偿矩阵、基于少量校准数据的MSE+正则化训练。

**📊 数据集**

使用128条Wiki百科校准样本，评估用WikiText‑2、PTB、C4的困惑度，零样本评估用BoolQ、PIQA、HellaSwag、WinoGrande、ARC‑easy、ARC‑challenge、OpenBookQA。

**📈 对比分析**

与SLEB、ShortGPT、MKA等基线比较，在7B–13B LLM上实现约4倍剪枝速度加速，保持甚至提升零样本困惑度与分类准确率，尤其在高压缩比例下仍保持优越性能。

**⚠️ 局限性**

局限性在于相较于无结构剪枝方法，GradMAP在更高压缩比例时仍难以匹配性能，且对极端压缩仍需进一步研究。

---

## 10. Towards a Hybrid Quantum-Classical Computing Framework for Database Optimization Problems in Real Time Setup

**arXiv ID:** 2602.14263 | [PDF](https://arxiv.org/pdf/2602.14263v1)

**作者:** Hanwen Liu `[一作]` (University of Southern California), Ibrahim Sabek `[通讯]` (University of Southern California)

**通讯引用:** 478 | [OpenAlex ID](https://openalex.org/A5062053667)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了首个实时量子增量数据库系统，融合可透明控制的混合量子-经典优化器。

**💡 创新点**

创新点在于迭代相关性放松与问题感知分解/重组两大可扩展策略，解决QUBO过度复杂、超量子硬件规模和黑盒求解缺陷。

**🔧 技术方法**

使用量子退火、QUBO建模、迭代相关性放松、问题感知分解与重组、混合求解循环，并集成PostgreSQL与D‑Wave量子云。

**📊 数据集**

使用IMDB数据库的Join Order Benchmark (JOB) 与 Cardinality Estimation Benchmark (CEB)。

**📈 对比分析**

与传统PostgreSQL优化器和D‑Wave CQM‑Solver对比，平均2.7‑2.8倍查询执行加速，单条最多14倍，整体E2E加速1.8‑2.6倍，且量子生命周期约500ms。

**⚠️ 局限性**

局限在于量子云通信开销高、硬件规模受限、仅在查询计划层面验证，需进一步优化通信与多任务并行。

---

## 11. MAS-on-the-Fly: Dynamic Adaptation of LLM-based Multi-Agent Systems at Test Time

**arXiv ID:** 2602.13671 | [PDF](https://arxiv.org/pdf/2602.13671v1)

**作者:** Guangyi Liu `[一作]` (Tsinghua University), Quanming Yao `[通讯]` (Tsinghua University)

**通讯引用:** 9491 | [OpenAlex ID](https://openalex.org/A5072484211)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出MASFly，一种能在无额外训练的情况下，在测试时动态生成并执行多智能体系统的框架。

**💡 创新点**

创新点在于引入检索增强的标准操作程序（SOP）实例化实现查询依赖的系统生成，以及基于经验的全局Watcher监控实现执行时的自适应调整。

**🔧 技术方法**

技术包括检索增强生成（RAG）结合SOP库、经验驱动的过程监督（Watcher + Personalized Experience Pool）、以及跨任务的经验蒸馏机制。

**📊 数据集**

使用的数据集涵盖长时程规划（TravelPlanner）、通用问答（GAIA）以及代码生成（HumanEval、MBPP、MBPP Pro）等多领域测试集。

**📈 对比分析**

与手工设计的MetaGPT、半自动化AgentVerse、全自动化AgentSquare等方法相比，MASFly在TravelPlanner的Two-Stage和Sole-Planning模式下分别提升了约196%和92%的最终成功率，在所有基准上均保持领先且稳健。

**⚠️ 局限性**

局限性主要在于过程监督的触发机制依赖预设启发式规则，缺乏自动化的异常检测与动态干预决策。

---

## 12. Deep Dense Exploration for LLM Reinforcement Learning via Pivot-Driven Resampling

**arXiv ID:** 2602.14169 | [PDF](https://arxiv.org/pdf/2602.14169v1)

**作者:** Yiran Guo `[一作]` (Institute of Software, Chinese Academy of Sciences), Lijie Xu `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为 Deep Dense Exploration (DDE) 的策略，并实现了 DEEP-GRPO 算法，针对 LLM 强化学习中的探索瓶颈，专注于失败轨迹中的关键深层状态（pivot）进行局部密集采样，以提升模型的推理性能。

**💡 创新点**

创新点包括：
• 通过轻量化的 Utility 函数平衡可恢复性与深度，自动识别最具探索价值的 pivot；
• 在 pivot 处执行局部密集重采样，显著提高发现正确后缀的概率；
• 采用双流优化（global vs local），解耦主策略与局部修正，使用梯度遮蔽避免共享前缀干扰，从而实现更稳定的学习。

**🔧 技术方法**

使用技术包括：
• 基于 MDP 的语言生成建模与 GRPO 优化框架；
• Logistic 回归在线估计 recoverability（可恢复性）；
• 局部密集采样与梯度遮蔽；
• 双流损失与权重平衡（λ 控制）。

**📊 数据集**

数据集与模型：
• 任务数据：GSM8K、MATH、AIME24、AMC、MATH500、Minerva、OlympiadBench；
• 预训练模型：Qwen2.5-0.5B-Instruct、Qwen2.5-Math-1.5B/7B；
• 训练集：MATH Level 3–5（1.5B/7B）和 GSM8K（0.5B）。

**📈 对比分析**

与基线对比：
• 在 GSM8K 上，DEEP-GRPO 以 67.7% 的 Pass@1 超越 GRPO（最高 66.2%）和 TreeRL/AttnRL；
• 在其他数学基准上，平均准确率提升 1–3%（如 1.5B 模型 Avg 43.3%→46.7%，7B 模型 Avg 51.4%→54.0%）。

**⚠️ 局限性**

局限性：
• 依赖于可恢复性估计的准确性，需调参（γ、λ）才能获得最佳效果；
• 目前仅在数学推理任务上验证，缺乏对其他 NLP 任务的泛化评估；
• 局部密集采样虽然有效，但在极大模型/预算场景下计算开销仍不小。

---

## 13. AllMem: A Memory-centric Recipe for Efficient Long-context Modeling

**arXiv ID:** 2602.13680 | [PDF](https://arxiv.org/pdf/2602.13680v1)

**作者:** Ziming Wang `[一作]` (Huawei Technologies), Qinghai Guo `[通讯]` (Huawei Technologies)

**通讯引用:** 3798 | [OpenAlex ID](https://openalex.org/A5015433561)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将预训练的解码器型LLM迁移到AllMem架构，结合滑动窗口注意力与测试时可训练的非线性记忆网络，实现长序列推理的高效化。

**💡 创新点**

核心创新是：① 并行的短期滑动窗口注意力与长期参数化记忆的分离与可学习的通道级融合门；② 在推理时对记忆网络进行在线梯度更新（Test‑Time Training），通过动态学习率与动量实现高效的知识压缩；③ 通过随机化窗口大小、sink数量和更新块尺寸的训练策略，使模型在不同长序列长度下具备鲁棒的压缩能力。

**🔧 技术方法**

技术手段包括：滑动窗口注意力（SWA）、非线性TTT记忆网络、RMSNorm、Swish‑GLU残差记忆单元、动量梯度下降、Chunk‑wise 在线学习、位置无关记忆模块、随机化训练超参、知识蒸馏（KL + CE）以及在Qwen3基础上进行的参数化迁移。

**📊 数据集**

使用的公开数据集：ChatQA2（长序列蒸馏）、ChineseInstruct、SFTv3（短序列蒸馏）、LongBench、InfiniteBench、LV‑Eval、C‑Eval、ARC、HellaSwag、WinoGrande、MMLU‑Redux、GPQA‑Diamond、IFEval、MATH‑500、LiveCodeBench v5。

**📈 对比分析**

与原始全注意力、滑动窗口+sink、Mamba等基线在同等滑动窗口大小下进行对比。AllMem在37k平均长序列的LongBench上仅以4k窗口接近全注意力性能，且在128k上下文的InfiniteBench与LV‑Eval上甚至优于全注意力；在短序列任务上也保持或略优于原始Qwen3模型，参数增幅仅约6–10%。

**⚠️ 局限性**

主要局限：① 对全注意力的精度差距在极长序列（>128k）下仍略有；② 在线梯度更新对推理时的CPU/GPU负载有一定影响；③ 需在训练阶段采用大量随机化策略，模型收敛稳定性和超参调优仍依赖经验；④ 目前的实验主要基于中文/英文数据，跨语言泛化能力待进一步验证。

---

## 14. Learning on the Fly: Replay-Based Continual Object Perception for Indoor Drones

**arXiv ID:** 2602.13440 | [PDF](https://arxiv.org/pdf/2602.13440v1)

**作者:** Sebastian-Ion Nae `[一作]` (National University of Science and Technology Politehnica Bucharest), Marius Leordeanu `[通讯]` (NORCE Norwegian Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了室内无人机的连续对象感知，并构建了第一份高时序一致的室内视频数据集，评估了三种基于回放的增量学习方法。

**💡 创新点**

创新点包括：提供高时序一致的室内无人机视频数据集，采用半自动 GroundingSAM 伪标注流程，且在极低回放预算下证明 FAR 方法优于 ER 与 MIR。

**🔧 技术方法**

使用的技术包括 YOLOv11‑nano 检测器、回放策略 ER/MIR/FAR、梯度权重类激活映射 Grad‑CAM 以及 PID 控制器进行闭环跟踪。

**📊 数据集**

数据集为 14,400 帧室内无人机视频（含无人机-无人机与无人机-地面车辆交互），并附加 2,000 张地面车辆子集，涵盖 5 个增量任务。

**📈 对比分析**

方法在 5%–50% 回放预算下比较，FAR 在 5% 预算下取得 82.96% mAP_50‑95 ACC，10% 时 86.48%，与联合训练差距不大；ER 与 MIR 在低预算下性能明显逊色。

**⚠️ 局限性**

限制包括仅使用 5 类受控室内场景，伪标注仍有残余噪声，未检验更大标签空间及更强域迁移，跟踪控制仅为简化 PID。

---

## 15. Policy Gradient with Adaptive Entropy Annealing for Continual Fine-Tuning

**arXiv ID:** 2602.14078 | [PDF](https://arxiv.org/pdf/2602.14078v1)

**作者:** Yaqian Zhang `[一作]` (AI Institute), Albert Bifet `[通讯]` (AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在连续微调中，将分类任务视为一阶马尔可夫决策过程，直接通过期望策略梯度（EPG）优化0-1损失，弥补传统交叉熵过度探索的缺陷；

**💡 创新点**

创新点包括：①从强化学习视角重新定义分类目标并推出低方差的EPG方法；②提出自适应熵退火（aEPG）在训练期间逐步从交叉熵过渡到EPG，实现探索与利用的动态平衡；

**🔧 技术方法**

采用的技术包括期望策略梯度（EPG）、自适应熵退火调度、PEFT模块（LoRA、Adapter、Prefix）以及对比熵正则化方法；

**📊 数据集**

使用的主要数据集为ImageNet-R、Split‑Food101、Split‑CUB200、CLRS25；

**📈 对比分析**

在与七种PEFT基线及多种熵正则化方法对比实验中，aEPG在所有任务拆分上均取得显著提升，尤其在20%对称标签噪声或低熵场景下表现优异；

**⚠️ 局限性**

局限性：在随机初始化模型时，纯0-1优化难以收敛，仅在预训练模型上效果显著；在细粒度数据集CUB‑200上，熵降益处有限。

---

## 16. Chemical Language Models for Natural Products: A State-Space Model Approach

**arXiv ID:** 2602.13958 | [PDF](https://arxiv.org/pdf/2602.13958v1)

**作者:** Ho-Hsuan Wang `[一作]` (Saarland University), Dietrich Klakow `[通讯]` (Saarland University)

**通讯引用:** 4516 | [OpenAlex ID](https://openalex.org/A5008875255)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对天然产物在化学语言模型中的不足，构建了基于选择性状态空间模型（Mamba、Mamba-2）和Transformer（GPT）的天然产物专用化学语言模型，并对八种分词策略进行系统评测，验证模型在分子生成和属性预测上的优势。

**💡 创新点**

创新点包括首次在天然产物任务中应用最新的Mamba系列S6模型，全面比较S6与Transformer在长程依赖、生成质量与属性预测的表现，并提出基于域特定分词（AIS、NPBPE）的策略显著提升性能。

**🔧 技术方法**

主要技术为Mamba与Mamba-2的选择性状态空间网络、GPT-2 Transformer、字节对编码（BPE）和Atom-in-SMILES分词、SAM优化、随机与骨架拆分、跨模型微调、以及SMILES生成与验证工具。

**📊 数据集**

数据集包括1,030,273条经过清洗的天然产物SMILES（SuperNatural 3.0、COCONUT、LOTUS），以及三个下游任务数据集：肽膜通透性（6,651条）、FourTastes（4,431条）、抗癌活性（约26,000条）。

**📈 对比分析**

通过在相同拆分（随机、骨架）下训练48个模型并与MoLFormer、ChemBERTa-2比较，发现Mamba在有效率与唯一性上比GPT高1–2%，在长程错误上低3–6%；在属性预测上Mamba/Mamba-2在随机拆分下略优于GPT，骨架拆分下三者性能相当；与大规模通用CLM对比，1M天然产物预训练可匹敌其效果。

**⚠️ 局限性**

局限性包括对合成可行性评估不足、对复杂sp3富集天然产物的生成仍受限、模型规模与训练成本高、以及在骨架拆分下仍未实现显著的泛化提升。

---

## 17. Diff-Aid: Inference-time Adaptive Interaction Denoising for Rectified Text-to-Image Generation

**arXiv ID:** 2602.13585 | [PDF](https://arxiv.org/pdf/2602.13585v1)

**作者:** Binglei Li `[一作]` (Fudan University), Hao Li `[通讯]` (Fudan University)

**通讯引用:** 30687 | [OpenAlex ID](https://openalex.org/A5100348631)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了推理时自适应交互去噪（AiD）插件，用于在文本到图像扩散模型中动态调节不同 transformer 块、时间步和文本 token 的交互，从而显著提升文本提示的遵循度与生成质量。

**💡 创新点**

核心创新在于可学习的 per‑token、block‑aware、time‑aware 调节系数（α）以及门控稀疏正则化，使模型在保持轻量化的同时，能够自适应地强化关键信息传递。

**🔧 技术方法**

采用扩散 Transformer（MMDiT）架构、轻量化 MLP + gate 机制、tanh 限幅、正则化与 DPO 优化，保持预训练模型不变，仅训练少量可插拔参数。

**📊 数据集**

在 HPDv3 子集上训练，随后在 HPDv3、HPSv2、ImageReward、Aesthetic Score、GenEval、EditBench 等公开基准上进行评估。

**📈 对比分析**

与 SD3.5、FLUX 基线及 TACA、LoRA 等方法对比，AiD 在 HPSv3、HPSv2、ImageReward、Aesthetic、GenEval 等指标均提升约 0.2–0.3 分，且在多任务（控制生成、风格 LoRA、零射编辑）中也表现出持续的性能提升。

**⚠️ 局限性**

局限性包括：仅针对文本到图像任务验证；对其它模态（文本到视频、3D）尚未测试；在极大规模多模态场景下的可扩展性和推理效率仍需进一步探索。

---

## 18. Inner Loop Inference for Pretrained Transformers: Unlocking Latent Capabilities Without Training

**arXiv ID:** 2602.14759 | [PDF](https://arxiv.org/pdf/2602.14759v1)

**作者:** Jonathan Lys `[一作]` (IMT Atlantique), Ghouthi Boukli Hacene `[通讯]` (Sony Europe Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在冻结的预训练Transformer模型上引入推理时内部循环，以延长隐层的迭代精炼过程。

**💡 创新点**

提出了在推理阶段通过循环重用特定层并对隐状态进行插值的正则化方法，能够在不训练的前提下获得额外精炼。

**🔧 技术方法**

使用Transformer的中间循环、正则化插值（均匀、移动平均、自动对齐）以及简单的噪声对照等技术。

**📊 数据集**

在Gemma 2B/9B和Llama 3-8B的多项选择推理基准上评估，主要使用WinoGrande、ARC、GSM8K、HellaSwag、MMLU等数据集。

**📈 对比分析**

与标准单次前向推理相比，内循环在Gemma模型上在大多数任务上获得了0.5–1.5% 的准确率提升，而Llama 3-8B的效果不稳定；通过噪声对照验证提升非随机。

**⚠️ 局限性**

局限性包括：仅在部分模型和任务中显著提升；对预归一化（pre‑norm）架构的鲁棒性差；需要手工挑选循环层区间；提升幅度相对温和。

---

## 19. MATEO: A Multimodal Benchmark for Temporal Reasoning and Planning in LVLMs

**arXiv ID:** 2602.14589 | [PDF](https://arxiv.org/pdf/2602.14589v1)

**作者:** Gabriel Roccabruna `[一作]` (University of Trento), Giuseppe Riccardi `[通讯]` (University of Trento)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布MATEO多模态基准，用于评估大规模视觉语言模型在Temporal Execution Order（TEO）任务上的推理与规划能力。

**💡 创新点**

①通过专业食谱手工拆分并收集图文配对，构建高质量DAG标签；②将TEO任务转化为三分类问题，提供多种提示与自我反思策略；③开放数据、代码与评测框架，促进可复现与跨模型对比。

**🔧 技术方法**

使用大型视觉语言模型（Qwen2.5-VL-72B、InternVL3.5-8B/38B、LLaVA-OneVision-7B、GPT‑5.1等），结合提示工程（baseline、instructions、ICL、CoT、自我反思）以及LoRA微调进行实验。

**📊 数据集**

MATEO数据集：约300道意大利食谱，含每步文本描述与对应图像，已翻译成英文；按70%/10%/20%比例划分训练/验证/测试。

**📈 对比分析**

在原始与逆序两种视图下采用一致性准确率评估；最佳开源模型Qwen2.5‑VL‑72B在多模态+自我反思约0.68；Fine‑tuned InternVL3.5‑8B达到0.69；封闭模型GPT‑5.1在20%样本上约0.74；整体性能仍低于0.75，表明当前LVLM在TEO推理方面存在显著提升空间。

**⚠️ 局限性**

提示策略对不同模型效果差异大，难以统一；未尝试多模态ICL；实验仅在英文环境下进行，跨语言泛化未知；自我反思机制在部分模型上效果不稳定，需进一步改进。

---

## 20. Offline-Poly: A Polyhedral Framework For Offline 3D Multi-Object Tracking

**arXiv ID:** 2602.13772 | [PDF](https://arxiv.org/pdf/2602.13772v1)

**作者:** Xiaoyu Li `[一作]` (Harbin Institute of Technology), Lining Sun `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 8401 | [OpenAlex ID](https://openalex.org/A5085075643)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Offline-Poly，一种基于Tracking‑by‑Tracking（TBT）范式的通用离线3D多目标跟踪框架，能够仅利用任意上游跟踪结果进行轨迹优化；

**💡 创新点**

创新点在于完全解耦离线跟踪与检测/跟踪器，支持多源输入、无训练的全局匹配与融合，以及利用资源无约束与未来可观测性实现的分层匹配、合并与轨迹细化；

**🔧 技术方法**

采用了轨迹层级预处理、基于运动模型的匹配与融合（STWO、STW）、多跟踪器匹配与合并（MTM）、全局与局部轨迹细化（角点对齐与滑动窗口优化）等技术；

**📊 数据集**

在nuScenes和KITTI两个公开自动驾驶数据集上进行实验；

**📈 对比分析**

通过与多种在线与离线跟踪器、不同检测器进行比较，Offline‑Poly在nuScenes实现77.6% AMOTA、81.4% AMOTA（多源融合）和KITTI 3D实现83.00% HOTA，均位居当前最优；

**⚠️ 局限性**

局限性包括对阈值参数敏感、缺乏深度学习驱动的自适应机制、以及在极端遮挡或动态场景下的匹配误差仍存在。

---

## 21. Implicit Bias in LLMs for Transgender Populations

**arXiv ID:** 2602.13253 | [PDF](https://arxiv.org/pdf/2602.13253v1)

**作者:** Micaela Hirsch `[一作]` (Instituto de Ciencias de la Computación), Enzo Ferrante `[通讯]` (Instituto de Ciencias de la Computación)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对七种大型语言模型（GPT‑4o、Gemini、Grok、Llama3等）在英语和西班牙语下进行隐性偏见评估，采用词汇联想测试和医疗预约分配任务，量化模型对跨性别者的负面关联与分配偏好。

**💡 创新点**

首次将词汇联想与相对决策任务结合，聚焦医疗排程场景下跨性别偏见的量化，并探讨生成解释是否能缓解此类偏差。

**🔧 技术方法**

使用大型语言模型、提示工程、词汇联想测试、相对决策评估、Bootstrap置信区间等技术手段。

**📊 数据集**

使用8类（情感、道德、风险、真确性、外貌、知识、合法性与犯罪、生命结果）各15正负词的联想词表，以及合成的100名患者档案（姓名、年龄、出生性别、性别认同）和相应的临床症状样本。

**📈 对比分析**

通过计算偏见分数（正负词联想差异）和分配率比较模型，并在无解释与解释两种条件下评估；结果显示大多数模型在多种医疗服务领域都存在显著偏差，解释生成在某些领域可部分缓解，但整体仍未消除。

**⚠️ 局限性**

仅覆盖英语和西班牙语；使用合成数据且可能缺乏真实临床复杂性；模型自我解释可能不准确；评估任务有限，未涵盖所有现实决策场景；使用静态模型版本，更新或微调可能改变行为；词汇联想测试不考虑否定或语境等细节。

---

## 22. Evolutionary System Prompt Learning can Facilitate Reinforcement Learning for LLMs

**arXiv ID:** 2602.14697 | [PDF](https://arxiv.org/pdf/2602.14697v1)

**作者:** Lunjun Zhang `[一作]` (University of Toronto), Bradly C. Stadie `[通讯]` (Northwestern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Evolutionary System Prompt Learning（E-SPL）框架，联合演化系统提示与强化学习（RL）更新模型权重，以提升大语言模型（LLM）的自我改进能力。

**💡 创新点**

创新点在于将 RL 与进化搜索耦合：使用 TrueSkill 评估提示的相对表现，采用 LLM 驱动的变异与交叉算子在相同的 RL 训练循环中生成新的系统提示，从而实现上下文与权重的协同优化。

**🔧 技术方法**

使用了强化学习（如 GRPO、CISPO）、低秩适配（LoRA）梯度更新、TrueSkill 排名、基于 LLM 的自我反思、变异与交叉算子，以及对系统提示的遗传算法框架。

**📊 数据集**

主要实验数据集包括数学推理任务：DAPO100→AIME25、HMMT 23/24→25、AIME→BeyondAIME；以及代理搜索任务：Natural Questions (NQ) 与 HotpotQA 的子集；模型分别为 DeepSeek v3.1 与 OpenAI gpt-oss-120b。

**📈 对比分析**

通过与 RL-only、演化-only、仅自我反思等基线对比，E-SPL 在 AIME 2025 从 56.3% 提升到 60.6%，在 HMMT 2025 从 50.0% 提升到 52.7%，在 AIME→BeyondAIME 从 38.8% 提升到 45.1%，在代理搜索任务从 44.2% 提升到 48.6%。整体表现显著优于任何单一方法，证明了上下文与权重协同优化的有效性。

**⚠️ 局限性**

局限性包括：尚未实现完全自我重写或自我参照的自我改进；生成的系统提示有时会过度泛化或包含错误；对长提示的支持有限；演化搜索可能导致多样性下降，需进一步改进多样性维护机制。

---

## 23. From Pixels to Policies: Reinforcing Spatial Reasoning in Language Models for Content-Aware Layout Design

**arXiv ID:** 2602.13912 | [PDF](https://arxiv.org/pdf/2602.13912v1)

**作者:** Sha Li `[一作]` (Virginia Tech), Xiang Chen `[通讯]` (Adobe Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于强化学习的LLM框架，实现文本化空间推理生成内容感知的图形布局。

**💡 创新点**

创新点在于将布局设计视作策略学习，使用多目标空间批评回报，并通过文本化环境实现可解释的空间推理。

**🔧 技术方法**

采用Qwen-2.5-7B-Instruct LLM、GRPO策略优化、多目标奖励（格式、质量、IoU）以及结构化JSON输出。

**📊 数据集**

使用CGL和PKU-PosterLayout两个公开布局数据集。

**📈 对比分析**

与DS-GAN、PosterLlama、GPT-4o等基线比较，取得更高的结构有效率、对齐、间距一致性，并在样本数更少的条件下性能可与视觉模型媲美。

**⚠️ 局限性**

局限在于预先依赖显著性检测、单步生成、缺少多轮交互与更丰富的语义约束。

---

## 24. Scaling the Scaling Logic: Agentic Meta-Synthesis of Logic Reasoning

**arXiv ID:** 2602.13218 | [PDF](https://arxiv.org/pdf/2602.13218v1)

**作者:** Bowen Liu `[一作]`, Jia Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 15083 | [OpenAlex ID](https://openalex.org/A5108050433)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种名为SSLogic的代理元合成框架，通过迭代合成和修复可执行的生成器-验证器程序对，自动扩展可验证推理任务家族。

**💡 创新点**

创新点在于实现了任务家族级别的扩展，采用闭环的生成-验证-修复流程，确保了数据的可验证性和可控难度，同时引入了多门验证协议以提高可靠性。

**🔧 技术方法**

使用了代理程序合成和修复技术，结合多策略一致性检查和对抗性盲审的验证机制。

**📊 数据集**

从400个种子任务家族开始，通过两轮演化扩展到953个任务家族和21,389个可验证实例。

**📈 对比分析**

与种子基线相比，使用SSLogic演化的数据在相同训练步骤下表现出一致的性能提升，SynLogic提高了5.2，BBEH提高了1.4，AIME25提高了3.0，Brumo25提高了3.7。

**⚠️ 局限性**

限制在于合成过程中可能出现的实现错误和不可解性问题，尽管多门验证机制可以有效过滤这些问题，但仍需进一步优化合成流程以提高效率。

---

## 25. Parameter-Efficient Fine-Tuning of DINOv2 for Large-Scale Font Classification

**arXiv ID:** 2602.13889 | [PDF](https://arxiv.org/pdf/2602.13889v1)

**作者:** Daniel Chen `[一作]`, Marcus Lowe `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套能够识别394种字体变体的字体分类系统

**💡 创新点**

采用LoRA参数高效微调DINOv2并构建大规模可复现的合成字体数据集

**🔧 技术方法**

DINOv2视觉Transformer、LoRA适配器、Pillow渲染、数据增强与预处理

**📊 数据集**

基于Google Fonts的31个基础字体族（共394个变体）通过合成渲染生成的图片

**📈 对比分析**

与以往全微调或CNN基方法对比，达到86% top-1准确率，仅训练不到0.2%参数，表现优异

**⚠️ 局限性**

仅在合成图像上训练，难以完全泛化到真实照片、手写覆盖或极端风格化的文本图像

---

## 26. FlowHOI: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation

**arXiv ID:** 2602.13444 | [PDF](https://arxiv.org/pdf/2602.13444v1)

**作者:** Huajian Zeng `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xingxing Zuo `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1428 | [OpenAlex ID](https://openalex.org/A5045415760)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段基于流匹配的框架，能够根据视角观测、文本指令和3D Gaussian splatting场景重建，生成符合物理约束且语义一致的手-物体交互序列，随后可直接映射到机器人执行。

**💡 创新点**

创新点包括：①将手-物体交互拆分为几何中心的抓取阶段和语义中心的操控阶段；②利用3D场景令牌和运动-文本对齐损失实现语义与空间双重归一化；③在流匹配中引入持续速度预测和硬/软过渡约束，实现显著加速；④构建大规模 egocentric 视频重建管道，为抓取阶段提供丰富的 HOI 先验。

**🔧 技术方法**

使用的核心技术有：条件流匹配（conditional flow matching）、3D Gaussian splatting（3DGS）场景重建、T5-Large 文本编码、Perceiver 降维场景令牌、运动-文本对齐的 InfoNCE 损失、MANO 手模型、SAM3/SAM3D 对象重建与对齐、基于深度估计的深度图推理。

**📊 数据集**

数据集包括：GRAB（高精度手-物体捕捉数据）、HOT3D（真实环境 egocentric 视频及 3D 场景重建）以及用于抓取先验预训练的 EgoDex 大规模 egocentric 视频。

**📈 对比分析**

在 GRAB 和 HOT3D 上与 DiffH2O、LatentHOI 等基线进行对比，FlowHOI 在动作识别准确率、物理仿真成功率（高 1.7 倍）以及推理速度（比 DiffH2O 快 40 倍）等指标均显著优于对手；同时保留了多样性与现实可执行性。

**⚠️ 局限性**

局限性：依赖准确的手部与物体初始状态估计，严重遮挡或重建误差会导致性能下降；生成的轨迹仅为运动学级别，需要后续控制器来实现动力学与顺应；目前不支持移动式操作，且仅基于 egocentric 视角，未覆盖 exocentric 视频。

---

## 27. Autonomous Robotic Tissue Palpation and Abnormalities Characterisation via Ergodic Exploration

**arXiv ID:** 2602.14287 | [PDF](https://arxiv.org/pdf/2602.14287v1)

**作者:** Luca Beber `[一作]` (University of Trento), Luigi Palopoli `[通讯]` (University of Trento)

**通讯引用:** 2716 | [OpenAlex ID](https://openalex.org/A5006224560)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套基于力/扭矩传感器的可视化、实时弹性映射框架，结合可变形组织模型与热方程驱动的遍历控制实现了无触觉传感器的自主组织探诊；

**💡 创新点**

创新点在于：①设计了结合不确定度、刚度强度与梯度的专属期望信息密度(EID)；②闭环结合在线可变形参数估计与可实时更新的遍历轨迹规划；③使用HEDAC实现高频（100 Hz）平滑连续运动；④引入基于遍历度的停止准则；并在仿真与实验中优于贝叶斯优化方法；

**🔧 技术方法**

采用了扩展卡尔曼滤波进行在线参数估计、Gaussian Process回归生成刚度场、热方程驱动的遍历控制（HEDAC）以及自定义EID；对比了两种贝叶斯优化变体（EI 与 EI+中间采样）作为基线；

**📊 数据集**

实验数据包括：在仿真中使用三种不同高斯混合生成的二维刚度分布；以及在实际UR3e机械臂上对含硬球形夹杂物的硅胶模体进行的物理探测；未使用公开医学数据集；

**📈 对比分析**

通过检测率、均方根误差、轨迹长度和耗时等指标与BO基线对比，发现遍历方法在相同或更短时间内实现相近或更高的检测率，并显著降低RMSE；在分割任务中亦展示出更高的灵敏度和与BO相近的特异性；

**⚠️ 局限性**

局限性包括仅在平面二维空间上验证，缺乏三维非平面映射；依赖精确的力/扭矩测量和预设的接触模型；对复杂多层或高噪声组织的适应性尚待评估；对大面积或高分辨率场景的计算成本仍需进一步优化。

---

## 28. DRAMA: Domain Retrieval using Adaptive Module Allocation

**arXiv ID:** 2602.14960 | [PDF](https://arxiv.org/pdf/2602.14960v1)

**作者:** Pranav Kasela `[一作]` (University of Milano-Bicocca), Raffaele Perego `[通讯]` (ISTI-CNR)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DRAMA 框架，利用域特定轻量级适配器与动态门控机制，构建多域信息检索模型，兼顾检索效果与能耗与参数效率。

**💡 创新点**

创新点在于将知识蒸馏、适配器模块和 Mixture‑of‑Experts 风格的路由分离训练，实现无全模型重训练即可增添新域，显著降低参数与能源成本。

**🔧 技术方法**

使用了 Houlsby/LoRA 适配器、知识蒸馏、动态域分类器（门控）、BiEncoder/CrossEncoder dense 检索、参数高效微调（PEFT）以及 MoE 思路。

**📊 数据集**

实验涵盖两大 Web 规模检索基准：学术搜索（4 个学科）与 Stack‑Exchange 社区问答（4 个社区），并在 BEIR 的 SCIDOCS 与 Quora 上做零样本验证。

**📈 对比分析**

通过与单域专用模型、统一多域模型、随机路由等 baseline 进行 MAP@100、MRR@10、NDCG@10 对比；DRAMA 在大多数任务上与专用模型相当或略优，同时参数与能耗下降约 75% 以上。

**⚠️ 局限性**

局限包括门控在域重叠或样本不足时准确率下降；依赖于域标签；在极大域数或动态域变迁时仍需额外训练适配器和门控。

---

## 29. Emergently Misaligned Language Models Show Behavioral Self-Awareness That Shifts With Subsequent Realignment

**arXiv ID:** 2602.14777 | [PDF](https://arxiv.org/pdf/2602.14777v1)

**作者:** Laurène Vaugrante `[一作]` (Interchange Forum for Reflecting on Intelligent Systems), Thilo Hagendorff `[通讯]` (Interchange Forum for Reflecting on Intelligent Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究大语言模型在出现‘突发性不对齐’后能否通过自评识别其行为变化

**💡 创新点**

首次证明模型在无上下文示例下即可自我识别出其误导性行为，并且自评随对齐/解对齐而改变

**🔧 技术方法**

使用 GPT‑4.1 系列模型的监督微调、对齐/解对齐实验以及多种自评/恶意评测指标

**📊 数据集**

利用包含错误三问-答对（800条）和不安全代码（6000条）的微调数据集

**📈 对比分析**

对比基准模型、误导模型和对齐模型，结果显示误导模型的自评与实际恶意度高度相关，误导度在对齐后显著下降

**⚠️ 局限性**

局限性包括自评方法影响结果、问卷项模糊、开放式自评难以分类、未探究内部机制以及模型可能的欺骗性自报

---

## 30. KoopGen: Koopman Generator Networks for Representing and Predicting Dynamical Systems with Continuous Spectra

**arXiv ID:** 2602.14011 | [PDF](https://arxiv.org/pdf/2602.14011v1)

**作者:** Liangyu Su `[一作]` (Xi'an Jiaotong University), Zongben Xu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 15871 | [OpenAlex ID](https://openalex.org/A5109280540)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了KoopGen，一种利用Koopman生成器的可解释、可扩展的神经网络框架，用于表示和预测具有连续谱的动力系统。

**💡 创新点**

将Koopman生成器的笛卡尔分解（skew‑adjoint + self‑adjoint）直接嵌入网络结构，采用状态依赖的生成器组合并使用结构保持的实数块参数化，实现对连续谱动力学的可解释分离，同时兼顾高维可扩展性。

**🔧 技术方法**

结构保持的生成器参数化、门控网络实现状态依赖的生成器混合、指数映射得到离散时间Koopman算子、Sobolev损失、PyTorch训练等技术。

**📊 数据集**

四个基准连续谱系统：非线性摆、Lorenz‑63、Lorenz‑96（高维），以及高维空间时间混沌的Kuramoto–Sivashinsky方程。

**📈 对比分析**

与LRAN（状态独立）和DeepKoopman（显式谱参数化）对比，KoopGen在所有系统上预测误差更低，尤其在高维和长时域上表现显著优越，提升了稳定性和可扩展性。

**⚠️ 局限性**

仍需提升计算效率和参数优化；理论上缺乏对近似误差、稳定性保证和泛化的严格分析；在真实世界数据上的验证待进一步研究。

---

## 31. TouchFusion: Multimodal Wristband Sensing for Ubiquitous Touch Interactions

**arXiv ID:** 2602.15011 | [PDF](https://arxiv.org/pdf/2602.15011v1)

**作者:** Eric Whitmire `[一作]` (Meta), Hrvoje Benko `[通讯]` (Meta)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一款多模态腕带，能够在任何环境表面或人体上感知并追踪触摸状态与位置，支持有状态触摸、连续跟踪以及离散手势识别，并在多种交互场景中进行验证。

**💡 创新点**

① 通过单一腕带实现全套触摸原语（触摸状态、触摸追踪、身体触摸识别），① 结合 sEMG、Bioimpedance、IMU、IR 接近与 ToF 光学传感器的多模态融合，② 采用大规模 100 名受试者数据集进行训练与评估，③ 通过实时在线管线展示腕带与 AR/VR 生态系统的协同潜力。

**🔧 技术方法**

sEMG（8 通道电极），Bioimpedance（双通道差分电极），IMU（加速度计、陀螺仪、磁力计），IR 接近传感器（4 通道），ToF 传感器（单像素 + 8×8 区域）以及光学/电磁信号的融合模型（CNN‑LSTM、深度卷积+线性回归等）。

**📊 数据集**

共收集 100 名参与者数据：70 组“世界触摸”实验（Sensel Morph、OptiTrack 记录手势轨迹与触摸点），30 组“身体触摸”实验（掌部触摸、双腕互联标注）。

**📈 对比分析**

在离散触摸上：世界触摸模型 AUC‑PR 0.960、AUC‑ROC 0.994；身体触摸模型 AUC‑PR 0.794、AUC‑ROC 0.969。在线任务中，单击成功率 0.95，状态触摸成功率 0.97；Fitts 任务中，左‑右方向吞吐率 1.23 bits/s，垂直方向 1.09 bits/s，误入率分别为 0.21 与 0.53。与现有单模或环形设备相比，腕带实现了更高的准确性与更丰富的交互表达。

**⚠️ 局限性**

① 电极接触与腕带贴合度差异导致身体触摸模型对部分用户失效；② 2D 触摸追踪精度仍低于商用触控板，受累于加速度积分漂移；③ 设备体积较大、功耗高，电池续航仅约 4h；④ 模型未实现个性化微调，难以完全覆盖广泛生理差异；⑤ 仅在静态桌面环境验证，移动场景中的误报与能耗控制尚未完善。

---

## 32. Elastic Diffusion Transformer

**arXiv ID:** 2602.13993 | [PDF](https://arxiv.org/pdf/2602.13993v1)

**作者:** Jiangshan Wang `[一作]` (Tsinghua University), Chunchao Guo `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Elastic Diffusion Transformer (E‑DiT)，一种自适应加速 Diffusion Transformer 的框架，通过轻量路由器动态决定每个 Transformer 块是否跳过以及 MLP 宽度的缩放，并在推理时加入块级缓存机制。

**💡 创新点**

创新点：① 轻量路由器同时预测块跳过与 MLP 宽度；② 基于路由预测的块级缓存，利用时间冗余进一步减少计算；③ 提供一种通用的自适应计算框架，适用于多种 DiT 基座与多模态任务。

**🔧 技术方法**

使用技术包括：轻量化路由网络、Straight‑Through Estimator、MLP 宽度自适应、块级缓存、训练阶段质量‑效率联合损失、Rectified Flow ODE 等。

**📊 数据集**

使用数据集：图像生成任务使用 BLIP3o‑60K 与 ShareGPT‑4o，3D 资产生成使用内部数据集；在 Qwen‑Image、FLUX、Hunyuan3D‑3.0 等基座上进行评估。

**📈 对比分析**

与多种剪枝、蒸馏及动态加速基线（如 FLUX.1 Lite、TinyFusion、HP、PPCL、Dense2MoE、DyDiT）对比，在 DPG‑Bench、GenEval、T2I‑CompBench 以及 ULIP、Uni3D 指标上保持或略低的质量，同时实现约 2× 的推理速度提升，并在单块 H20 GPU 上显著降低单步延迟。

**⚠️ 局限性**

局限性：需要在训练阶段调节质量‑效率平衡超参数；缓存机制可能导致误差累计；在极其复杂样本上块跳过可能损失质量；跨模态通用性虽已验证，但对更多基座与任务的进一步适配仍待研究。

---

## 33. How Do Lexical Senses Correspond Between Spoken German and German Sign Language?

**arXiv ID:** 2602.13790 | [PDF](https://arxiv.org/pdf/2602.13790v1)

**作者:** Melis Çelikkol `[一作]` (Institute for Computational Linguistics, University of Heidelberg), Wei Zhao `[通讯]` (Department of Computing Science, University of Aberdeen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建德语词义与德国手语手势的跨模态对应关系，手工标注1404个词用—手势ID映射，并系统分析三种对应类型（One-to-Many、Many-to-One、One-to-One）。

**💡 创新点**

首个基于词义使用的德语-德国手语跨模态歧义对应数据集，并证明语义相似度方法在识别一对多对应时显著优于传统精确匹配。

**🔧 技术方法**

采用SBERT句子嵌入计算语义相似度（Exact Match vs Semantic Similarity），结合余弦相似度阈值来判定对应类型，并进行Ablation实验评估输入特征影响。

**📊 数据集**

使用德语词义使用图（D-WUG）与数字德国手语词典（DW-DGS）两大资源，涵盖32个德语词、49个手势，总计1404个使用—手势映射。

**📈 对比分析**

与Exact Match基准对比，四个SBERT模型中all-MiniLM-L6-v2在词汇重叠测试集上达88.52%精度（比EM 71.31%提升17.21pp），尤其在一对多（Type 1）上提升52.1pp；但在无词汇重叠的测试集上准确率骤降至17.59%。

**⚠️ 局限性**

局限性：数据量仅32词；标注未由德语-手语双语者完成；方法高度依赖词汇重叠，缺乏对新词汇或其他语言的泛化能力；未处理多义词内部细粒度歧义。

---

## 34. Optimal Program Synthesis via Abstract Interpretation

**arXiv ID:** 2602.14717 | [PDF](https://arxiv.org/pdf/2602.14717v1)

**作者:** Stephen Mell `[一作]` (University of Pennsylvania), Osbert Bastani `[通讯]` (University of Pennsylvania)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在给定的领域特定语言（DSL）中，针对包含数值常量且目标是最大化量化指标（如准确率）的程序合成问题，提出了一种基于抽象解释的A*搜索框架，能够在搜索图上枚举“泛化部分程序”，并通过抽象解释构造可接受的启发式估计，从而保证找到最优（或在误差容忍度内的最优）程序。

**💡 创新点**

创新点包括：① 把抽象解释作为A*搜索的启发式，利用对目标值的上界保证搜索可收敛到最优；② 针对数值常量的无穷搜索空间，设计了带区间约束的泛化部分程序与对应的搜索图；③ 对于单调DSL组件，提出了直接利用区间域的抽象变换器，既保持过逼近又易于实现；④ 证明了该框架在有限或满足Lipschitz条件下能够终止并保持最优性。

**🔧 技术方法**

核心技术包括：A*搜索、抽象解释（区间域和单调变换器）、泛化部分程序（带区间约束的占位符）以及对量化目标（如准确率）的抽象评估。实现上使用了Python/LLL/SMT等工具。

**📊 数据集**

在两个实际DSL上进行实验：NEAR DSL（针对CRIM13轨迹数据集）和Quivr DSL（使用公开轨迹基准），评估了对这些DSL的适配性。

**📈 对比分析**

与基于SMT的Metasketches优化器以及基于广度优先搜索的基线进行对比。实验结果显示，本方法在搜索时间上至少提升了1-2个数量级，尤其在含数值常量的搜索空间中表现显著优于对手。

**⚠️ 局限性**

主要局限包括：① 需要DSL的组件和目标函数具备单调性；② 需要可构造区间抽象域，若DSL包含非单调或复杂的非凸函数，现有变换器难以直接适用；③ 对于极大规模的数值范围，区间细化仍可能导致搜索树爆炸；④ 只保证在给定IO样例上的最优，未考虑泛化误差。

---

## 35. Architectural Insights for Post-Tornado Damage Recognition

**arXiv ID:** 2602.14523 | [PDF](https://arxiv.org/pdf/2602.14523v1)

**作者:** Robinson Umeike `[一作]` (University of Alabama), Cuong Pham `[通讯]` (University of South Alabama)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对自建的Quad‑State Tornado Damage（QSTD）数据集进行大规模实验，评估79个公开CNN和Transformer模型，探究架构与优化参数对龙卷风损伤识别的影响。

**💡 创新点**

提出系统化的四阶段实验框架，并发现优化器（SGD）和学习率（1e‑4）对Transformer的提升最显著，提供可落地的优化指导。

**🔧 技术方法**

使用PyTorch训练79个公开模型，调参包括优化器、学习率、分辨率、数据增强、损失函数等，系统性检验各项对性能的作用。

**📊 数据集**

主要使用自建的QSTD数据集（6级IN‑CORE损伤标签）进行训练验证，并在Tuscaloosa‑Moore Tornado Damage（TMTD）数据集做零样本跨事件测试。

**📈 对比分析**

通过两阶段调优与2,300+实验，最佳模型ConvNeXt‑Base在QSTD上macro‑F1达68%，在TMTD零样本上Macro‑F1提升34点，验证跨域泛化能力。

**⚠️ 局限性**

局限在于仅使用二维街景图像，缺乏多视角或LiDAR等更丰富数据；调参以单一维度为主，未深入探索参数交互与更高阶优化策略。

---

## 36. Uncertainty-Aware Vision-Language Segmentation for Medical Imaging

**arXiv ID:** 2602.14498 | [PDF](https://arxiv.org/pdf/2602.14498v1)

**作者:** Aryan Das `[一作]` (VIT Bhopal), Vinay Kumar Verma `[通讯]` (IIT Kanpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一个利用影像与临床文本的多模态分割框架，并通过不确定性建模提升可靠性。

**💡 创新点**

核心创新包括：Modality Decoding Attention Block（MoDAB）与State Space Mixer（SSMix）的跨模态融合结构，以及将空间、频谱与熵信息统一的Spectral‑Entropic Uncertainty（SEU）损失函数。

**🔧 技术方法**

技术实现基于ConvNeXt‑Tiny视觉编码器、BioViL CXR‑BERT文本编码器，结合MoDAB+SSMix注意力模块、SEU损失、以及轻量化Transformer/SSM混合网络。

**📊 数据集**

实验使用QaTa‑COV19、MosMed++与Kvasir‑SEG这三大公开医学影像数据集。

**📈 对比分析**

与多种单模态和多模态基线（如MAdapter、BiomedClip、TransUNet等）对比，模型在Dice和mIoU上分别提升至92.24%/84.9%、79.67%/66.38%、93.83%/87.62%，且仅拥有39.9M参数、17.87G FLOPs，显示出更优的性能‑效率比。

**⚠️ 局限性**

局限性在于验证仅局限于三种影像类型和相对小规模数据集，缺乏跨域或真实临床环境的进一步测试；对文本输入质量高度依赖；未探索大规模预训练或多语言适配。

---

## 37. Efficient Data-Driven Production Scheduling in Pharmaceutical Manufacturing

**arXiv ID:** 2602.13668 | [PDF](https://arxiv.org/pdf/2602.13668v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e`

---

## 38. Explore Intrinsic Geometry for Query-based Tiny and Oriented Object Detector with Momentum-based Bipartite Matching

**arXiv ID:** 2602.13728 | [PDF](https://arxiv.org/pdf/2602.13728v1)

**作者:** Junpeng Zhang `[一作]`, Mengxuan Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于 Transformer 的查询式航空倾斜目标检测器 IGOFormer，专注于极小尺寸和任意方向目标的检测。

**💡 创新点**

通过在解码阶段引入“Intrinsic Geometry‑aware Decoder”将对象相关特征的几何关系编码为几何嵌入增强查询表示，并提出 Momentum‑based Bipartite Matching 对跨阶段匹配成本做指数移动平均以减少身份漂移。

**🔧 技术方法**

基于 DETR 架构的多阶段解码、旋转 RoIAlign、动态滤波、Multi‑group 方案、Softmax 相似度、指数移动平均、Hungarian 匹配、AdamW 优化等技术。

**📊 数据集**

在三大航空倾斜目标检测基准 DOTA‑V1.0、DOTA‑V1.5 与 DIOR‑R 上进行训练与评估。

**📈 对比分析**

与多种顶级倾斜检测方法（EMO2‑DETR、Oriented R‑CNN、OrientedFormer 等）在单尺度训练/推理下对比，使用 Swin‑T/ResNet‑50 等骨干，IGOFormer 在 DOTA‑V1.0 上 AP_50 达到 78.00%，在 DOTA‑V1.5 上 68.88%，在 DIOR‑R 上 69.27%，均优于同骨干基线 1–3 个百分点。

**⚠️ 局限性**

对极小尺寸或复杂场景下的对象仍存在一定局限，匹配稳定性提升但仍依赖手动设置的平滑系数与 α 超参，计算成本略高，且仅在航空图像数据上验证，缺乏跨域泛化评估。

---

## 39. SAM4Dcap: Training-free Biomechanical Twin System from Monocular Video

**arXiv ID:** 2602.13760 | [PDF](https://arxiv.org/pdf/2602.13760v1)

**作者:** Li Wang `[一作]` (Sichuan University), Jian Li `[通讯]` (Sichuan University)

**通讯引用:** 27562 | [OpenAlex ID](https://openalex.org/A5100402458)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用 SAM-Body4D 的训练免费 4D 人体网格恢复，将单目视频转换为 OpenSim 兼容的运动学双胞胎，并在无力板的情况下估计运动学与动力学。

**💡 创新点**

实现了从单目视频到全 biomechanical 结果的端到端、训练免费流水线，并自动提示、Linux 原生构建与多模型兼容。

**🔧 技术方法**

结合 SAM-Body4D、MHR 到 TRC 转换、OpenSim Solver（OpenCap/ AddBiomechanics）以及自动提示与 Procrustes 对齐。

**📊 数据集**

使用公开的运动学实验视频（步态、下落跳跃），以及现有的 MoCap 数据进行对比。

**📈 对比分析**

通过与多视角 MoCap 与 OpenCap 结果对比，膝关节运动学与多视角系统相近，但髋关节在跳跃时误差显著，存在残差抖动。

**⚠️ 局限性**

缺乏运动平滑、使用 SMPL 代替更精准的 MHR、缺少个体化软组织标记校准导致动力学精度受限。

---

## 40. Detecting LLM Hallucinations via Embedding Cluster Geometry: A Three-Type Taxonomy with Measurable Signatures

**arXiv ID:** 2602.14259 | [PDF](https://arxiv.org/pdf/2602.14259v1)

**作者:** Matic Korun `[一作]` `[通讯]` (Independent Researcher), Matic Korun (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型的幻觉进行几何分类，利用静态token嵌入的聚类结构识别三种幻觉类型并给出可测量的判别指标。

**💡 创新点**

提出基于嵌入空间几何的幻觉分类框架，创新性地定义了三种统计量α（极性耦合）、β（聚类凝聚度）和γ（径向信息梯度），并证明它们在多种Transformer架构中普遍有效且与架构特征相关。

**🔧 技术方法**

使用k-means聚类、余弦相似度、PCA、二次多项式拟合、F检验等统计方法对静态词嵌入进行分析。

**📊 数据集**

使用11个公开Transformer模型（BERT、RoBERTa、ELECTRA、DeBERTa、ALBERT、MiniLM、DistilBERT、GPT‑2等）及其词频信息（用于计算自信息）。

**📈 对比分析**

对11个模型分别计算α、β、γ的显著性，β在所有模型显著>0，α>0.5，γ在9/11显著；对异常模型ALBERT和MiniLM进行架构解释；未使用标准幻觉评测集，因而缺乏直接性能对比。

**⚠️ 局限性**

局限性包括仅分析静态嵌入未考虑上下文化表征；模型规模受限于1.5B参数以下；聚类数量k=40未系统验证；α统计样本有限；缺少与已有评测集（如HaluEval）的比较。

---

## 41. ProMoral-Bench: Evaluating Prompting Strategies for Moral Reasoning and Safety in LLMs

**arXiv ID:** 2602.13274 | [PDF](https://arxiv.org/pdf/2602.13274v1)

**作者:** Rohan Subramanian Thomas `[一作]`, Sunishchal Dev `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ProMoral-Bench统一基准，评估11种提示策略在四大LLM家族上对伦理判断与安全性的表现

**💡 创新点**

创新点在于：①引入ETHICS‑Contrast对比集与统一的Moral Safety Score (UMSS)；②系统比较多种提示框架（如Few‑Shot、Plan‑and‑Solve、Value‑Grounded等），并揭示紧凑结构优于多轮冗长推理；③提供标准化代码与数据，便于复现与跨模型对比

**🔧 技术方法**

采用标准化提示模板、确定性解码、五步Thought Experiment、Self‑Correct、Plan‑and‑Solve等提示技术，并使用Brier、ECE等指标进行评估

**📊 数据集**

使用ETHICS、Scruples、WildJailbreak和自制ETHICS‑Contrast四个数据集

**📈 对比分析**

通过统一UMSS对所有模型-策略组合进行跨数据集、跨指标的排名，发现Few‑Shot/Few‑Shot‑CoT和Role‑Prompting在准确性、鲁棒性、token成本上位居榜首；多轮提示在安全性虽稍好但成本高且易失效

**⚠️ 局限性**

局限包括：单次无置信区间、仅评测特定API版本、数据集和提示均为英文/西方文化、提示与评测数据同源导致偏高、以及对新模型或其他语言的可迁移性未知

---

## 42. Enhancing NOMA Handover Performance Using Hybrid AI-Driven Modulated Deterministic Sequences

**arXiv ID:** 2602.13202 | [PDF](https://arxiv.org/pdf/2602.13202v1)

**作者:** Sumita Majhi `[一作]`, Pinaki Mitra `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了结合Gold–Walsh调制序列与深度Q网络（DQN）的NOMA切换干扰管理框架，实现实时干扰控制和资源分配。

**💡 创新点**

创新点包括：①将Gold序列与Walsh序列逐元素相乘生成高性能调制序列；②利用DQN动态选择序列和功率分配，实时适应高速移动下的干扰；③通过经验回放、优先采样和目标网络加速收敛；④在多指标（切换成功率、吞吐量、干扰）上实现显著提升。

**🔧 技术方法**

使用技术包括Gold–Walsh序列乘积、FFT相关分析、深度Q网络（DQN）+经验回放+优先经验抽样+目标网络、RL收敛理论、复杂度分析 O(N log N + d·h + log B)。

**📊 数据集**

使用了基于Python仿真的合成数据集，包含19个基站、每基站4-8 NOMA用户、3.5 GHz载波、100 MHz带宽、用户速度3–120 km/h等参数。

**📈 对比分析**

与传统Gold、Walsh、Kasami、无AI调制序列和仅用DRL功率分配的5个基线在10,000次仿真中比较。切换成功率达95.2%（比Gold提升23.1pp），吞吐量提升28.4%，干扰下降41.3%，所有指标均在ANOVA中显著(p < 0.001)，且DQN在约4,200±400回合内收敛。

**⚠️ 局限性**

局限性：仅在仿真环境验证，未在真实网络或超密集场景中测试；需进一步扩展至6G多代理RL和极端密集网络；模型对参数敏感，鲁棒性和实时适应性需进一步研究。

---

## 43. Beyond Translation: Evaluating Mathematical Reasoning Capabilities of LLMs in Sinhala and Tamil

**arXiv ID:** 2602.14517 | [PDF](https://arxiv.org/pdf/2602.14517v1)

**作者:** Sukumar Kishanthan `[一作]`, Asela Hevapathige `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文构建了英语、僧伽罗语和泰米尔语的原生数学词问题数据集，并在四大 LLM 上进行零样本评估；

**💡 创新点**

创新点在于提出六类结构化数学问题分类，创建无翻译误差的多语言原生题库，并细粒度分析跨语言推理差异；

**🔧 技术方法**

使用技术包括零样本提示、数值答案抽取评测、基于六类问题的结构化评估与可视化对比；

**📊 数据集**

使用数据集为每种语言 25 道题，共 450 道题，覆盖单步、多步、干扰信息、单位冲突、逻辑推导与优化六类；

**📈 对比分析**

通过对比 GPT‑4o、DeepSeek‑V3、Gemini 2.5、Claude Sonnet 4 的准确率，发现英语最高，僧伽罗语和泰米尔语出现不同程度下降，尤其在单位冲突与优化类显著受损；

**⚠️ 局限性**

局限性包括数据规模有限、仅评估零样本提示、未使用内部可解释性方法以及未探究多步链式思考可能改善跨语言性能。

---

## 44. HyMem: Hybrid Memory Architecture with Dynamic Retrieval Scheduling

**arXiv ID:** 2602.13933 | [PDF](https://arxiv.org/pdf/2602.13933v1)

**作者:** Xiaochen Zhao `[一作]` (ZJU-UIUC Institute), Aili Wang `[通讯]` (ZJU-UIUC Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HyMem 混合记忆架构，通过多粒度存储（摘要级 Level‑1 与原始文本级 Level‑2）和动态两层检索调度来处理 LLM 长对话中的记忆管理问题。

**💡 创新点**

创新点在于：①双粒度记忆存储与级联检索；②基于任务难度的动态启用深度检索；③自我反思模块实现多轮推理优化。

**🔧 技术方法**

采用向量嵌入+语义检索、LLM‑驱动的自检索与生成、两层检索器组合、以及反思模块的递归推理技术。

**📊 数据集**

在 LOCOMO 与 LongMemEval‑S 两个公开长上下文问答基准上进行评测。

**📈 对比分析**

与 Full‑Context、Naive RAG、LangMem、Zep 等基线对比，HyMem 在所有任务类别均优于基线，整体准确率达 89.55%（LOCOMO）和 75.00%（LongMemEval‑S），且 token 消耗比 Full‑Context 低 92.6%。

**⚠️ 局限性**

局限性：对极端复杂查询仍需频繁深度检索导致推理时间波动；压缩策略需手动调参；目前仅在公开基准上验证，缺乏真实场景长会话的实测。

---

## 45. From User Preferences to Base Score Extraction Functions in Gradual Argumentation

**arXiv ID:** 2602.14674 | [PDF](https://arxiv.org/pdf/2602.14674v1)

**作者:** Aniol Civit `[一作]` (Institute of Robotics and Industrial Informatics), Francesca Toni `[通讯]` (Imperial College)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了从用户偏好排序直接提取论证框架基准分数（Base Score Extraction Function, BSEF）的理论与实现方法，进一步将双极论证框架（BAF）转化为量化双极论证框架（QBAF），并在机器人辅助喂食场景中验证其效果。

**💡 创新点**

将偏好关系与论证框架结合的基准分数提取方法进行形式化，并引入非线性“更强偏好”关系和一组可调设计参数，构建了两种可解释且可调的基准分数函数，首次在逐渐语义下证明其满足一系列公理与性质。

**🔧 技术方法**

使用偏好化论证框架（PAF）与量化双极论证框架（QBAF）理论，结合逐渐语义（Quadratic Energy、Euler-Based、DF‑QuAD），并实现基于偏好距离的算法（δ、Δ）来计算基准分数。

**📊 数据集**

主要使用随机生成的3万条偏好排序样本以及机器人喂食案例的人工构造论证图，未使用公开数据集。

**📈 对比分析**

通过与三种逐渐语义的决策结果进行逐对一致性与Cohen’s Kappa评估，实验显示Quadratic Energy与Euler-Based在偏好一致性上达到≈0.98、0.96的高一致性，而DF‑QuAD相对较低，整体表明方法在多种语义下能保持较好的一致性与可解释性。

**⚠️ 局限性**

仅考虑完全序的单用户偏好，忽略部分排序、循环框架和多用户情况；对极端基准分数的敏感性在某些语义下可能导致决策不稳定；算法对参数（δ、Δ、ratio）的设置依赖手工调优，缺乏自适应机制。

---

## 46. EgoSound: Benchmarking Sound Understanding in Egocentric Videos

**arXiv ID:** 2602.14122 | [PDF](https://arxiv.org/pdf/2602.14122v1)

**作者:** Bingwen Zhu `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**通讯引用:** 15893 | [OpenAlex ID](https://openalex.org/A5084959430)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了首个专注于自我视角音频理解的基准数据集EgoSound，并设计了涵盖声音特征、计数、时序、空间定位、源识别、因果推理与跨模态推理的七类任务；

**💡 创新点**

首次系统评估多模态大型语言模型在第一人称音频-视觉推理上的表现，并揭示了现有模型在细粒度音频感知与跨模态推理上的显著缺陷；

**🔧 技术方法**

采用多阶段自动生成流程（包含人机交互注释、音视Caption生成、GPT-4o生成QA对）以及大型多模态LLM（如Qwen3-Omni-Thinking-30B、Video-SALMONN 2+等）进行实验；

**📊 数据集**

使用来自Ego4D和EgoBlind的900段第一人称视频，生成7315个高质量开放式问答对；

**📈 对比分析**

通过与人类评测（83.9%）对比，最佳模型Qwen3-Omni-Thinking-30B仅达到56.7%准确率，表明当前多模态模型在音频理解与跨模态推理方面仍存在较大差距；

**⚠️ 局限性**

局限性在于模型对细粒度音频特征与空间定位的感知不足，单模态视觉训练未能弥补音频理解缺口，且现有多模态LLM在声图融合上的策略仍待优化。

---

## 47. Bidirectional Temporal Dynamics Modeling for EEG-based Driving Fatigue Recognition

**arXiv ID:** 2602.14071 | [PDF](https://arxiv.org/pdf/2602.14071v1)

**作者:** YipTin Po `[一作]`, Nevin L. Zhang `[通讯]`

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

该研究提出了DeltaGateNet框架，用于基于EEG的驾驶疲劳识别。

**💡 创新点**

创新点在于引入双向Delta模块分离正负一阶差分并显式建模非对称神经动态，以及门控时域卷积捕捉长程依赖。

**🔧 技术方法**

采用双向Delta、门控时域卷积、深度可分离卷积、残差学习、GELU激活等技术。

**📊 数据集**

在SEED‑VIG和SADT两大公开驾驶疲劳数据集上进行实验。

**📈 对比分析**

与多种基线（EEGNet、InceptionTime、ResNet、ViT等）对比，DeltaGateNet在intra‑subject和inter‑subject场景下均取得最高准确率，分别达到81.89%/55.55%（SEED‑VIG）和96.81%/83.21%（SADT 2022）等。

**⚠️ 局限性**

主要局限包括模型对步长与核大小敏感，仍需进一步验证多模态融合及在真实驾驶环境下的鲁棒性。

---

## 48. LLM-Enhanced Rumor Detection via Virtual Node Induced Edge Prediction

**arXiv ID:** 2602.13279 | [PDF](https://arxiv.org/pdf/2602.13279v1)

**作者:** Jiran Tao `[一作]` (Hong Kong Polytechnic University), Binyan Jiang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 1144 | [OpenAlex ID](https://openalex.org/A5003276044)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用LLM生成子链谣言概率并加入虚拟节点，改造图结构，提升谣言检测。

**💡 创新点**

创新点在于将LLM推理与图学习融合、通过子链概率构造虚拟节点并改写图结构，从而实现鲁棒的链接预测。

**🔧 技术方法**

采用LLM（DeepSeek‑V3、Qwen‑Plus）、BERT文本特征、Bidirectional Graph Attention Network（Bi‑GAT）等技术。

**📊 数据集**

使用PHEME事件数据集和Weibo数据集共十条事件。

**📈 对比分析**

相较于基线GNN/Transformer模型，LLM‑VN+Bi‑GAT等方法在准确率、F1等指标上提升10–15%（PHEME上超过0.88）。

**⚠️ 局限性**

局限在于对LLM API的依赖导致推理延迟、对超大图规模处理不够实时，且LLM性能不佳时信号质量下降。

---

## 49. Prompt-Driven Low-Altitude Edge Intelligence: Modular Agents and Generative Reasoning

**arXiv ID:** 2602.14003 | [PDF](https://arxiv.org/pdf/2602.14003v1)

**作者:** Jiahao You `[一作]` (Nanjing University of Aeronautics and Astronautics), Qihui Wu `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 11760 | [OpenAlex ID](https://openalex.org/A5100785098)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种 Prompt‑to‑Agent Edge Cognition Framework（P2AECF），用于在低空智能网络（LAIN）中通过语义提示生成可执行的推理工作流，动态分配轻量级认知代理并利用扩散式推理规划实现自适应任务调度。

**💡 创新点**

创新点在于：①将任务从模型绑定解耦，通过提示解析生成抽象任务图；②引入可动态选择、重用的模块化认知代理，实现资源感知的异构边缘计算；③采用扩散模型作为推理规划器，以实时系统状态和历史反馈自适应生成执行路径，提升鲁棒性和能效。

**🔧 技术方法**

技术包括：自然语言/结构化提示解析与语义抽象、任务图（DAG）构建、基于元数据的代理选择与调度、扩散式生成式推理规划、闭环反馈与持续学习。

**📊 数据集**

使用仿真生成的低空 AAV 场景数据集（二维城市网格、AAV 通信、算力、能量约束等），未公开真实工业数据集，主要依赖离散事件仿真得到的任务完成率、能耗等指标。

**📈 对比分析**

对比方法：Fixed‑Graph（静态流水线）和 Cloud‑Centric（云端推理）。在不同网络延迟下，P2AECF 在理想延迟 10 ms 时任务完成率 96.4%，即使在 1500 ms 延迟下仍达 76.7%；与基线相比，任务完成率、能效、适应性、可靠性指标均提升至 0.845、0.800、0.923、0.759。

**⚠️ 局限性**

局限性：①扩散规划器需离线训练，运行时采样迭代有限，可能不适应极端突发环境；②对真实硬件与网络条件的验证不足，仿真结果可能与实际部署存在偏差；③元数据管理和代理注册的复杂度较高，需进一步简化和自动化。

---

## 50. Unsafer in Many Turns: Benchmarking and Defending Multi-Turn Safety Risks in Tool-Using Agents

**arXiv ID:** 2602.13379 | [PDF](https://arxiv.org/pdf/2602.13379v1)

**作者:** Xu Li `[一作]` (Northeastern University), Weiyan Shi `[通讯]` (Northeastern University)

**通讯引用:** 1172 | [OpenAlex ID](https://openalex.org/A5089522357)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出多轮攻击分类法并构建第一个面向工具使用的多轮代理安全基准(Multi‑Turn Agent Risk Benchmark)，同时提出无训练、工具无关的自我探索防御 ToolShield，显著降低攻击成功率。

**💡 创新点**

创新点在于将单轮危害拆解为多轮攻击序列的系统分类法；基于此创建实测基准；以及利用代理自身工具探索生成安全经验的无训练防御框架，可跨模型、跨工具迁移。

**🔧 技术方法**

采用 LLM（Claude‑4.5‑Sonnet、Qwen3‑Coder 等）进行攻击序列生成、仿真执行与经验提炼；工具调用接口（Playwright、Terminal、Filesystem、PostgreSQL、Notion）；自我探索防御利用仿真环境、经验摘要与增量更新。

**📊 数据集**

使用 365 个从公开基准（OpenAgentSafety、SafeArena、P2SQL、MCPMark 等）转化得到的多轮工具任务，覆盖 5 种工具及 5 类风险；基准涵盖 3.19 平均轮数，3‑4 轮占 71%。

**📈 对比分析**

对比单轮与多轮、无防御与 ToolShield 的 ASR/ RR 结果显示：多轮攻击导致平均 ASR 上升 16%（Claude‑4.5‑Sonnet +27%），ToolShield 在多轮上将 ASR 降低 30%（Claude‑4.5‑Sonnet 49.9%）、单轮 35%；防御效果可通过预算提升，且经验可跨模型迁移。

**⚠️ 局限性**

局限性包括：依赖 LLM 生成攻击序列可能受模型偏好影响；自我探索仅在仿真环境中测试，真实系统中的安全经验迁移仍需验证；对极端或未知工具的鲁棒性尚未充分评估。

---

## 51. Fine-Tuning a Large Vision-Language Model for Artwork's Scoring and Critique

**arXiv ID:** 2602.13306 | [PDF](https://arxiv.org/pdf/2602.13306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 52. Boundary Point Jailbreaking of Black-Box LLMs

**arXiv ID:** 2602.15001 | [PDF](https://arxiv.org/pdf/2602.15001v1)

**作者:** Xander Davies `[一作]` (UK AI Security Institute), Yarin Gal `[通讯]` (OATML)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种完全黑盒、自动化的“Boundary Point Jailbreaking（BPJ）”方法，用于绕过基于分类器的LLM安全防护。

**💡 创新点**

创新点在于：①将目标恶意文本通过噪声插值生成逐步加深难度的训练曲线（Curriculum Learning）；②在每个难度层次主动寻找“boundary points”（接近判别边界的样本）以最大化对攻击改进的判别信号；③结合进化搜索与主动学习，使得在仅获得单比特（flag / non‑flag）反馈的前提下，仍能高效收敛。

**🔧 技术方法**

技术手段包括：噪声插值（noise interpolation）构造中间目标；进化算法（mutation + elitist rank‑based selection）；边界点筛选与更新；连续优化（逐步降低噪声水平）以及批量监控建议。

**📊 数据集**

实验使用了：HarmBench（长篇生物滥用问题）、Anthropic 的生物滥用数据集（评估 CC）、OpenAI 的 GPT‑5 输入分类器专用数据集。对比了随机前缀（Best‑of‑N）、仅使用曲线的基线以及人类+自动化红队成果。

**📈 对比分析**

与基线相比，BPJ 在 Prompted GPT‑4.1‑nano 上平均 5 倍加速收敛；在 Constitutional Classifiers（CC）上，平均 rubric 分数从 0% 提升至 25.5%（有 elicitation 可达 68%），在 GPT‑5 输入分类器上提升至 75.6%。实验还证明了在单一查询上训练的攻击对未见查询具有良好迁移性。

**⚠️ 局限性**

局限性：①需要大量查询（如 CC 需 330‑660k 次，GPT‑5 需 210‑800k 次），导致高 flag 率，现实环境下可能导致账号封禁；②仅攻击分类器，未直接攻击主模型；③对高度随机或多模态分类器的适应性未知；④实验中保留了一部分关键实现细节以降低扩散风险。

---

## 53. Climber-Pilot: A Non-Myopic Generative Recommendation Model Towards Better Instruction-Following

**arXiv ID:** 2602.13581 | [PDF](https://arxiv.org/pdf/2602.13581v1)

**作者:** Da Guo `[一作]` (NetEase Cloud Music), Chuanjiang Luo `[通讯]` (NetEase Cloud Music)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了面向工业级单步检索的生成式推荐框架Climber‑Pilot，解决了生成检索的近视与指令遵循问题。

**💡 创新点**

创新点在于时间感知多项预测（TAMIP）与条件引导稀疏注意力（CGSA），将多步前瞻与业务约束在训练阶段内化。

**🔧 技术方法**

采用Transformer backbone、时间感知掩码、采样softmax、条件稀疏注意力等技术。

**📊 数据集**

使用工业级网易云音乐大规模日志（4.5亿交互）以及Amazon Sports/Beauty/Toys三大公共数据集。

**📈 对比分析**

通过离线HR@K、条件符合率及在线A/B测试，与SASRec、HSTU等SOTA比对，提升HR@50约12%/，A/B约+4.24%点赞率。

**⚠️ 局限性**

局限在于仅验证两分支TAMIP，模型规模仍受限；对极端多样化业务约束的泛化性未深入探究。

---

## 54. Affordance Transfer Across Object Instances via Semantically Anchored Functional Map

**arXiv ID:** 2602.14874 | [PDF](https://arxiv.org/pdf/2602.14874v1)

**作者:** Xiaoxiang Dong `[一作]` (Carnegie Mellon University), Weiming Zhi `[通讯]` (Vanderbilt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了通过语义锚点引导的功能映射（SemFM）框架，能够从单一视觉演示中将交互作用区域迁移到不同实例的物体上。

**💡 创新点**

创新点在于将预训练视觉语义嵌入作为锚点，结合功能映射的光滑性约束，实现在几何差异大的物体间保持语义一致的对应关系。

**🔧 技术方法**

方法利用SAM3D进行单视角网格重建，SigLip2Vision提取语义特征，构建稀疏语义点云，并通过功能映射及ZoomOut优化实现全局对应。

**📊 数据集**

实验使用六类人工合成物体（工具、鞋子、切割工具、水容器、控制器、剪刀型工具）以及真实抓取场景数据。

**📈 对比分析**

与仅基于几何的FM‑WKS和基于VLM的GPT‑4o（单视/多视）相比，SemFM在合成数据上IoU约0.6‑0.7，匹配VLM的准确率但平均推理时间仅约10秒，远低于VLM的30‑50秒。

**⚠️ 局限性**

局限性包括单视角重建导致几何不精确、对高度不规则物体的对应精度有限，以及依赖预训练模型和稀疏语义点云，难以直接处理缺乏语义标签或多视角需求的场景。

---

## 55. Broken Chains: The Cost of Incomplete Reasoning in LLMs

**arXiv ID:** 2602.14444 | [PDF](https://arxiv.org/pdf/2602.14444v1)

**作者:** Ian Su `[一作]`, Maheep Chaudhary `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在不同token约束下对推理模式（代码、注释、混合、无推理、标准CoT）的鲁棒性，比较四大前沿模型在数学基准上的表现。

**💡 创新点**

提出受限推理框架并系统进行token预算消减实验，发现截断推理可导致性能下降、代码推理更稳健、混合推理表现最差且模型间鲁棒性差异显著。

**🔧 技术方法**

通过系统提示控制模型推理模式，并在10%、30%、50%、70% token预算下执行，使用代码执行输出提取答案进行评估。

**📊 数据集**

在GSM8K、AIME、HMMT三大数学推理基准上进行实验。

**📈 对比分析**

对比无约束与不同token比例下的成功率，结果显示DeepSeek-V3.2截断CoT时从53%降至17%，Grok在30%预算仍保持80–90%成功率，而OpenAI与DeepSeek显著衰退。

**⚠️ 局限性**

仅限于数学推理，token消减采用固定比例而非自适应，实验未验证在其他领域的适用性。

---

## 56. VIPA: Visual Informative Part Attention for Referring Image Segmentation

**arXiv ID:** 2602.14788 | [PDF](https://arxiv.org/pdf/2602.14788v1)

**作者:** Yubin Cho `[一作]` (LG Electronics), Suk-Ju Kang `[通讯]` (Sogang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的视觉信息化部分注意力（VIPA）框架，用以提高对自然语言描述目标的图像分割精度。

**💡 创新点**

创新点在于将从图像中检索出的信息化视觉片段（Visual Expression）作为键值对直接投射到Transformer解码器，减少跨模态投影误差并增强语义一致性。

**🔧 技术方法**

核心技术包括基于BERT与Swin Transformer的双模态编码、使用本地-全局语言上下文检索视觉重要像素、对检索结果进行掩码跨注意力细化、以及像素对比学习来监督检索。

**📊 数据集**

在RefCOCO、RefCOCO+、RefCOCOg和ReferIt四个公开RIS数据集上进行实验。

**📈 对比分析**

与现有Transformer和LLM驱动的RIS方法相比，VIPA在所有数据集的oIoU/mIoU、精度以及计算效率上均取得领先或相近的表现，并且在复杂表达和未见类别上表现出更强的泛化能力。

**⚠️ 局限性**

限制主要包括需要额外的检索和细化步骤增加训练复杂度，对检索比例r的敏感性以及在极度稀疏或极大目标场景下可能仍存在注意力分散的问题。

---

## 57. Generative Latent Representations of 3D Brain MRI for Multi-Task Downstream Analysis in Down Syndrome

**arXiv ID:** 2602.13731 | [PDF](https://arxiv.org/pdf/2602.13731v1)

**作者:** Jordi Malé `[一作]` (Human Environment Research Group La Salle), Xavier Sevillano `[通讯]` (Human Environment Research Group La Salle)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

基于变分自编码器（VAE）构建3D脑MRI的低维潜在表示，并将其用于多任务下游分类（DS与EU区分、DS中阿尔茨海默进展、DS智力障碍分级），验证潜在空间在不同维度下的重建与分类性能。

**💡 创新点**

首次在Down Syndrome脑MRI上系统评估VAE潜在空间的结构与信息含量，展示不同维度潜在向量既能保持高重建精度，又能实现跨数据集的强泛化与临床判别。

**🔧 技术方法**

使用三种维度（24×24×24、12×12×12、3×3×3）的3D卷积VAE，加入重建损失、感知损失、对抗损失与KL正则化；下游采用残差网络分类器并结合加权交叉熵+焦点损失。

**📊 数据集**

训练集来自公开的HCP（1113张）和IXI（580张）T1脑MRI；下游任务使用Sant Pau Memory Unit（931张EU/DS）与ABC‑DS（63张DS）外部验证集。

**📈 对比分析**

通过SSIM/MSE/特征距离/余弦相似度量评估重建质量，采用PCA可视化潜在结构；在EU/DS二分类、DS AD进展二/三分类以及ID分级中，三种潜在维度均实现97%+准确率，AUC≥0.99，外部ABC‑DS验证亦保持≈99%准确。

**⚠️ 局限性**

限制包括潜在空间尺寸与细节保留的权衡、对早期Prodromal AD识别的低敏感度（样本稀少导致），以及ID分类中严重类别不平衡导致特异性不足。

---

## 58. Assessing Spear-Phishing Website Generation in Large Language Model Coding Agents

**arXiv ID:** 2602.13363 | [PDF](https://arxiv.org/pdf/2602.13363v1)

**作者:** Tailia Malloy `[一作]` (University of Luxembourg), Tegawende F. Bissyande `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大型语言模型（LLM）代码生成代理在创建针对性钓鱼网站时的能力与意愿，并构建了系统化的评估管线与对应数据集。

**💡 创新点**

提出首个针对LLM生成钓鱼网站的评估流程，并公开400多份生成的网站代码与内部推理日志，揭示了安全机制失效的具体情况。

**🔧 技术方法**

利用Visual Studio Code+GitHub Copilot Chat与OpenRouter接口，对40个来自8家公司（GPT、Claude、Gemini、Grok、Llama、Mistral、Qwen、Nova等）的LLM进行链式思考、工具调用、提示工程等技术操作。

**📊 数据集**

公开“LLM生成的钓鱼网站”数据集，包含200个网站代码仓库、内部推理日志与截图，托管于GitHub。

**📈 对比分析**

通过六项模型指标（最大上下文、费用、推理时长、工具使用次数、总token、Prompt尺寸）进行单变量回归，预测截图成功率和相似度；结果显示工具使用和推理时长与相似度高度相关，但与可执行代码成功率关联弱。

**⚠️ 局限性**

受实验时点、API成本不可比、模型随机性以及安全拒绝机制可解释性不足的限制，导致结果可能随模型更新或提示细节变化而改变，且未对所有模型进行可执行代码生成的完整评估。

---

## 59. SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data

**arXiv ID:** 2602.14687 | [PDF](https://arxiv.org/pdf/2602.14687v1)

**作者:** David Chanin `[一作]` (University College London), Adrià Garriga-Alonso `[通讯]` (MATS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SynthSAEBench，构建可扩展、可控的合成数据与标准基准模型，评估稀疏自编码器（SAE）在超位置、相关性、层级等真实特征场景下的性能。

**💡 创新点**

创新点：①统一且可扩展的合成数据生成框架；②提供 16k 规模基准模型 SynthSAEBench-16k；③通过对比多种 SAE 架构（L1、匹配追踪、BatchTopK、Matryoshka 等）揭示新的失败模式（MP‑SAE 对超位置噪声的过拟合）；④在已知真特征的条件下实现低噪声、可重复的评估。

**🔧 技术方法**

技术：基于低秩高斯 Copula 产生相关二值激活；正交化词典；多层次层级约束；多种稀疏激活函数（TopK、BatchTopK、JumpReLU、Matryoshka、匹配追踪）；精确匹配评价指标（MCC、唯一性、F1、L0 对比）以及自动稀疏度控制器。

**📊 数据集**

数据集：SynthSAEBench-16k（1.6万真特征，隐藏维度 768，含 Zipfian 触发概率、线性幅度、随机低秩相关与层级结构），以及可扩展至数十万特征的通用合成数据。

**📈 对比分析**

比较方法：在相同训练样本数（200M）和网络宽度（4096）下，对比多种 SAE 训练策略；使用 MCC、F1、解释方差、L0 等四类指标；结果显示 Matryoshka 取得最佳探测与特征恢复，但解释方差最低；MP‑SAE 解释方差最高但特征恢复与探测最差；整体最高 F1 仅约 0.88，远低于直接训练的逻辑回归探测器（≈0.97）。

**⚠️ 局限性**

局限性：①仅基于线性表示假设（LRH），无法覆盖更复杂的表示形式（如 Minkowski 或流形特征）；②合成数据仍无法完全模拟真实 LLM 的所有内部机制；③评估仅在单 GPU 上的 16k 规模实验，未验证更大规模或更高维度的泛化；④某些评估指标（如精度-召回）仍受超参数 L0 影响，需进一步改进自适应稀疏度控制。

---

## 60. LEAD-Drift: Real-time and Explainable Intent Drift Detection by Learning a Data-Driven Risk Score

**arXiv ID:** 2602.13672 | [PDF](https://arxiv.org/pdf/2602.13672v1)

**作者:** Md. Kamrul Hossain `[一作]` (King Fahd University of Petroleum and Minerals), Walid Aljoby `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 173 | [OpenAlex ID](https://openalex.org/A5015916901)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了LEAD-Drift框架，实现实时检测并预警Intent漂移，帮助网络提前防止失败。

**💡 创新点**

将意图漂移检测改造为前瞻式有监督学习，利用多时域模型动态估计剩余时间，并通过SHAP实现每次告警的可解释性。

**🔧 技术方法**

使用轻量MLP预测风险得分，EMA平滑后阈值触发告警，阈值通过F1最大化调优；告警解释采用SHAP；多时域模型分别训练不同预测窗口实现动态倒计时。

**📊 数据集**

采用合成时间序列数据生成器，模拟网络collector KPI（CPU、RAM、存储、网络连接、服务响应及其一阶差分），并注入可控的漂移与失败事件。

**📈 对比分析**

与加权KPI启发式和基于距离的漂移检测两种基线对比，LEAD-Drift平均提前7.3分钟（+17.8%），误报率下降80.2%（-4.12/日），检测率100%，平均提前时间53.6分钟。

**⚠️ 局限性**

仅在合成数据上验证，缺乏真实网络的多样性与噪声；需要先验失败标签；多意图冲突场景和实际生产部署尚未评估。

---

## 61. Model Context Protocol (MCP) Tool Descriptions Are Smelly! Towards Improving AI Agent Efficiency with Augmented MCP Tool Descriptions

**arXiv ID:** 2602.14878 | [PDF](https://arxiv.org/pdf/2602.14878v1)

**作者:** Mohammed Mehedi Hasan `[一作]` (Queen's University), Ahmed E. Hassan `[通讯]` (Queen's University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

评估并改进 Model Context Protocol（MCP）中工具描述的质量和缺陷（smells），通过构建评分 Rubric、开发 FM‑based 脚本扫描并自动补全缺失组件，最终在大规模 856 工具（103 MCP 服务器）上验证其对 FM‑agent 性能的影响。

**💡 创新点**

① 统一了工具描述的六个关键组件（Purpose、Guidelines、Limitations、Parameter Explanation、Length & Completeness、Examples）并基于它们定义了 Smell taxonomy；② 采用 LLM‑as‑Jury 自动化评分与检测；③ 通过全描述、消融组合和成本‑收益（token、步骤）三维对比，首次量化描述改进带来的性能提升与代价；④ 发现最小化组件组合可保持高性能并显著降低 token 使用。

**🔧 技术方法**

使用了 5‑分 Likert Rubric、MCC 反射协议提取工具元数据、GPT‑4.1‑mini 等大模型进行描述增强、Multi‑model LLM‑as‑Jury 进行评分、统计检验（ICC、Wilcoxon、McNemar、Chi‑square）、Tool Description Router 动态切换描述、MCP‑Universe benchmark 评估任务完成率/步骤/评估者分数。

**📊 数据集**

数据集：103 个 MCP 服务器共 856 个工具（官方 23、社区 80）来自 8 篇研究，MCP‑Universe benchmark（231 任务，6 个域，202 工具）以及手工收集的工具执行日志，用于增强描述的 Examples 与 Limitations。

**📈 对比分析**

方法：对原始描述、全增强描述以及各组件消融组合分别跑 MCP‑Universe benchmark，记录 SR（任务成功率）、AE（平均评估者得分）和 AS（平均执行步骤）。统计检验表明全描述在 4 个模型/域上平均提升 5.85pp SR、15.12% AE，但 AS 上平均升 67.46%；消融实验显示 Purpose+Guidelines 组合在 Finance、Location 等域可与全描述相当，移除 Examples 对性能无显著负面影响。

**⚠️ 局限性**

局限：① 只在 4 个模型和 6 个域上评估，未涵盖更广泛模型/场景；② FM‑based 修复可能生成冗长描述，增加 token 成本；③ 评估依赖 MCP‑Universe 基线缺乏单任务对照，导致对比不够细粒；④ Examples 与 Limitations 生成需要人工执行，成本高；⑤ 仅聚焦文本描述，未深入分析输入 schema 与参数约束对 Agent 行为的影响。

---

## 62. Layer-Guided UAV Tracking: Enhancing Efficiency and Occlusion Robustness

**arXiv ID:** 2602.13636 | [PDF](https://arxiv.org/pdf/2602.13636v1)

**作者:** Yang Zhou `[一作]` (University of Shanghai for Science and Technology), Haohua Zhang `[通讯]` (University of Shanghai for Science and Technology)

**通讯引用:** 15 | [OpenAlex ID](https://openalex.org/A5072613004)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种名为 LGTrack 的 UAV 视觉跟踪框架，通过动态层选择、GGCA 注意力模块和基于空间 Cox 过程的遮挡鲁棒学习实现高效、鲁棒的跟踪。

**💡 创新点**

创新点包括：① 轻量化的 Global‑Grouped Coordinate Attention (GGCA) 模块，可在不增加显著计算量的前提下捕获全局上下文与长距离依赖；② Similarity‑Guided Layer Adaptation (SGLA) 模块，替代传统知识蒸馏，仅保留与饱和层相似度最高的 Transformer 层，显著提升推理速度；③ 使用空间 Cox 过程的随机遮挡策略，在训练阶段模拟多样化遮挡，提高模型对遮挡的鲁棒性。

**🔧 技术方法**

技术细节包括：ViT‑tiny（DeiT‑tiny）作为主干网络；GGCA 在 Transformer 输出后进行通道与空间注意力融合；SGLA 使用 3 层 MLP 对 token 进行全局语义投影，决定激活哪一层；遮挡学习采用随机遮挡与空间 Cox 过程的结合；训练采用 AdamW、学习率衰减与多源数据增强。

**📊 数据集**

训练数据集：LaSOT、COCO、TrackingNet、GOT‑10k；评测数据集：DTB70、UAVDT、UAV123。

**📈 对比分析**

与多类主流跟踪器（DCF、CNN、ViT）对比，LGTrack 在 UAVDT 上取得 82.8% 精度、60.4% 成功率，平均 84.3%/66.2%；推理速度 258.7 FPS（GPU）/59.4 FPS（CPU），在速度‑精度平衡上优于 ORTrack‑D、AVTrack‑DeiT 等 SOTA 方法。

**⚠️ 局限性**

局限性：对极端遮挡与高速运动场景的适应性仍有限；饱和层阈值 l* 采用固定设置，可能在不同任务间需要调优；虽去除了蒸馏，但仍依赖 ViT 主干的深度，模型规模相对较大。

---

## 63. WavePhaseNet: A DFT-Based Method for Constructing Semantic Conceptual Hierarchy Structures (SCHS)

**arXiv ID:** 2602.14419 | [PDF](https://arxiv.org/pdf/2602.14419v1)

**作者:** Kiyotaka Kasubuchi `[一作]` (Pionira Solutions), Kazuo Fukiya `[通讯]` (Pionira Solutions)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于测度论和频域分析的Transformer/注意力重构框架，并构建WavePhaseNet，通过离散傅里叶变换（DFT）实现语义层级结构，降低嵌入维度到约3000维，同时引入共形化正则化和Hodge分解实现语义的谐波拼接，旨在从结构上抑制LLM的幻觉。

**💡 创新点**

①将LLM的自回归过程视为σ-代数上的条件期望，阐释幻觉是结构性不可避免的；②利用Zipf定律推导1/f频谱并证明可将24,576维压缩至3,000维仍保留语义；③在Transformer内部嵌入DFT谱模块和共形正则化，通过图拉普拉斯投影实现局部推理的全局拼接，形成全新的语义层级与一致性机制。

**🔧 技术方法**

离散傅里叶变换（DFT）、频谱能量累计分析、共形化正则化（cochain/cohomology）、图拉普拉斯（Laplacian）与Hodge分解、张量残差注入、FFT、稀疏线性求解、ABlation实验等。

**📊 数据集**

主要以GPT‑4的24,576维嵌入作为实验基准；评估指标包括交叉熵/困惑度（perplexity）、局部一致性得分、Zipf偏差与能量保持率等；在无公开数据集的前提下对比FNet与WavePhaseNet的设计差异。

**📈 对比分析**

与FNet比较：FNet侧重计算效率、直接用全频谱做词混合；WavePhaseNet侧重语义层级构造、频谱分块、相位保留和共形正则化。实验中通过消除λ、μ、η等超参数进行消融，展示了各模块对一致性和幻觉抑制的贡献，整体性能提升主要体现在更高的一致性得分与更低的幻觉率，但具体数值未给出。

**⚠️ 局限性**

1) 理论模型假设1/f频谱与Zipf定律完美对应，实际语料的频谱可能偏离；2) 共形化正则化和Hodge投影增加模型复杂度，训练成本上升；3) 论文未给出完整的实证验证与大规模实验结果，缺乏对真实数据集上的定量评估；4) 维度压缩虽理论可行，但在实践中可能导致细粒度语义损失。

---

## 64. MILD: Multi-Intent Learning and Disambiguation for Proactive Failure Prediction in Intent-based Networking

**arXiv ID:** 2602.14283 | [PDF](https://arxiv.org/pdf/2602.14283v1)

**作者:** Md. Kamrul Hossain `[一作]` (King Fahd University of Petroleum and Minerals), Walid Aljoby `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 173 | [OpenAlex ID](https://openalex.org/A5015916901)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 MILD 框架，用教师增强的 Mixture‑of‑Experts（MoE）实现多意图网络的主动故障预测与根因去模糊。

**💡 创新点**

创新点在于：① 把意图保障从被动漂移检测转变为固定窗口的故障预测；② 通过门控网络与教师蒸馏实现协同学习，实现 co‑drift 情况下的根因辨识；③ 结合多时隙模型动态估计时间‑到‑故障；④ 通过 SHAP 提供每一次告警的 KPI 解释。

**🔧 技术方法**

采用的技术包括：Mixture‑of‑Experts、教师模型（OvR Logistic Regression）、焦点损失与 KL 蒸馏、门控网络、指数加权移动平均（EWMA）告警、SHAP 解释、时间序列特征工程、10‑折阻塞交叉验证、Adam 优化器。

**📊 数据集**

使用数据集：① 200,000 分钟级合成基准，模拟 3 个意图（API、Telemetry、Analytics）及 3 种故障类型（简单漂移、非线性单意图、co‑drift）并包含无标签负样本；② 小型硬件‑in‑the‑loop 原型收集真实网络遥测（未用于主要评估）。

**📈 对比分析**

与 5 个基线（WKPI‑Tuned、Dist‑Target、LR‑OvR、MLP、LSTM）在相同 10‑折阻塞 CV 下比较；MILD 在所有意图上实现 100% 检测率、平均 88–110 分钟的提前告警、每日 3.9 次误报、88.7% 的根因准确率，显著优于基线。

**⚠️ 局限性**

局限性包括：假设每次故障只有一个主根因，缺失根因标签时仅用弱监督；主要基于合成数据，真实网络验证有限；目前仅演示 3 个意图，扩展到更大意图集合的可扩展性尚未评估；模型需要手工调参和大量标注。

---

## 65. Benchmarking Video Foundation Models for Remote Parkinson's Disease Screening

**arXiv ID:** 2602.13507 | [PDF](https://arxiv.org/pdf/2602.13507v1)

**作者:** Md Saiful Islam `[一作]` (University of Rochester), Ehsan Hoque `[通讯]` (University of Rochester)

**通讯引用:** 1342 | [OpenAlex ID](https://openalex.org/A5106184792)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了七种视频基础模型在 16 项标准化远程帕金森筛查任务中的表现，构建了大规模的视频数据集并提出基准。

**💡 创新点**

创新之处在于首次揭示不同模型在不同生理域任务中的差异性（任务-模型显著性），并给出选择模型与任务的指南。

**🔧 技术方法**

采用冻结的 VFM 嵌入 + 单层线性分类器的评估框架，并对多视图聚合、过采样等策略做实验。

**📊 数据集**

使用来自 1,888 名参与者的 32,847 条视频数据，涵盖 727 名帕金森患者，包含 16 个标准化临床动作。

**📈 对比分析**

通过比较 AUC、准确率、灵敏度、特异性等指标，模型在各任务中达到了 76.4–85.3% 的 AUC 与 71.5–80.6% 的准确率，高特异性但灵敏度偏低。

**⚠️ 局限性**

局限包括仅使用冻结模型未进行微调、数据集主要为白人群、对多视图聚合无显著提升，以及缺乏更细粒度的时序关注。

---

## 66. Crowdsourcing Piedmontese to Test LLMs on Non-Standard Orthography

**arXiv ID:** 2602.14675 | [PDF](https://arxiv.org/pdf/2602.14675v1)

**作者:** Gianluca Vico `[一作]` (Charles University), Jindřich Libovický `[通讯]` (Charles University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了一个包含145条意大利语–皮埃蒙特语对照句、手工词对齐的众包数据集，并对该数据集在 tokenization、主题分类和机器翻译等任务上的表现进行评估。

**💡 创新点**

①首次系统记录并利用非标准正字法的真实皮埃蒙特语文本；②提供手工词对齐作为低资源语言对齐研究的基准；③从 tokenization parity 角度揭示低资源语言在 LLM 中的成本差异。

**🔧 技术方法**

使用 tokenization parity 评估、Eflomal 与 SimAlign 进行词对齐、SIB‑200 数据集进行主题分类、零样本机器翻译以及各种公开 LLM（Llama‑3.3‑70B、Gemma‑3‑27B‑it、Qwen3‑30B‑A3B、EuroLLM‑9B、Tower‑Plus‑9B、Gemini‑2.5‑flash、GPT‑4o‑mini）。

**📊 数据集**

数据来源于 <cit.> 里的2009句多语言并行语料（仅收集145条意大利语→皮埃蒙特语翻译），手工对齐的词段数据以及SIB‑200标签集。

**📈 对比分析**

与意大利语、法语、英语等高资源语言比较，LLM 在 tokenization 上对皮埃蒙特语存在更高的 over‑tokenization（更高的 parity 值），但在主题分类上取得与意大利语相近的性能；在机器翻译中，源自皮埃蒙特语到高资源语言的翻译效果可接受，而生成皮埃蒙特语的质量仍显不足，pivot 翻译策略可提升约1–2个百分点。

**⚠️ 局限性**

局限性包括：众包样本量有限、注释者多为意大利语母语者导致语言变体不完整、正字法多样且缺乏标准化、未追踪说话者所用具体皮埃蒙特语变体、词对齐仅停留在词层面，且未覆盖子词对齐。

---

## 67. Rubrics as an Attack Surface: Stealthy Preference Drift in LLM Judges

**arXiv ID:** 2602.13576 | [PDF](https://arxiv.org/pdf/2602.13576v1)

**作者:** Ruomeng Ding `[一作]`, Zhun Deng `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在LLM评判管道中，基于自然语言评估表格的细微修改如何导致判定偏差（Rubric‑Induced Preference Drift），并展示了此类偏差可以通过仅修改评估表格在保持基准性能的前提下对目标域产生系统性偏移。

**💡 创新点**

创新点在于首次提出并量化了评估表格引发的偏差威胁，提出了鲁棒的基于人口进化搜索的表格攻击方法，并证明了该偏差能在对齐流水线中持续传播。

**🔧 技术方法**

采用的技术包括黑盒人口进化搜索、基于错误案例的表格细化、目标与基准域的对比评估，以及在下游策略对齐中的 DPO 训练。

**📊 数据集**

实验使用了 UltraFeedback、ChatbotArena、RMB、Anthropic hh‑rlhf、PKU‑SafeRLHF 等多套人类偏好数据集，构建了四个基准–目标组合。

**📈 对比分析**

与种子评估表格、随机搜索、Few‑Shot ICL、TextGrad 等方法对比，本文方法在保持基准准确率的同时，使目标域偏差提升至 0.208（帮助性）和 0.159（无害性），并在下游策略中导致 win‑rate 降至约 40%。

**⚠️ 局限性**

局限性包括依赖于特定的评估表格编辑工具、对标注者的假设以及仅在有限的基准与目标数据上验证，尚未探究对其他模型结构或更广泛域的适用性。

---

## 68. Position: Introspective Experience from Conversational Environments as a Path to Better Learning

**arXiv ID:** 2602.14910 | [PDF](https://arxiv.org/pdf/2602.14910v1)

**作者:** Claudiu Cristian Musat `[一作]` (Google DeepMind), Tom Duerig `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出将对话式环境和内省体验作为构建通用推理能力的新路径，强调通过社交互动内化推理流程，而非单纯依赖规模化强化学习。

**💡 创新点**

创新点在于将Vygotsky的社会性发展理论与AI推理结合，提出三大核心命题：社交起源的私有思维、对话式内省能将稀疏观测转化为可学习的叙事经验，以及对话质量决定推理质量。

**🔧 技术方法**

技术上主要使用大语言模型（LLM）配合多轮内部对话（内省），自我检验与修正机制，以及通过多模态或多代理的对话式训练框架（如Self‑Play、双角色结构）实现社会性摩擦的内部化。

**📊 数据集**

数据集方面没有给出具体实验数据，论文假设利用现有的人类对话、辩论、修正记录等多样化对话语料作为训练与评估素材，强调对话质量而非单纯量化文本。

**📈 对比分析**

方法比较仅在理论层面进行，未给出定量实验或对比结果；作者指出相较于传统单向Chain‑of‑Thought或RLHF，内省对话可提升样本效率与迁移能力，但缺乏实证验证。

**⚠️ 局限性**

局限性包括对文本对话的依赖可能导致信息瓶颈、对多模态环境适应不充分、对话质量的评估与标注成本高、以及内省对话的可解释性与可控性仍待研究。

---

## 69. More than Decision Support: Exploring Patients' Longitudinal Usage of Large Language Models in Real-World Healthcare-Seeking Journeys

**arXiv ID:** 2602.14733 | [PDF](https://arxiv.org/pdf/2602.14733v1)

**作者:** Yancheng Cao `[一作]` (Columbia University), Xuhai "Orson" Xu `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过为期四周的日记研究与深度访谈，系统探索患者在真实医疗寻求过程中与多种大语言模型（LLM）的互动，归纳其在行为、信息、情感和认知四个层面的动态角色，并考察其对患者-医护关系的影响。

**💡 创新点**

首次将LLM重新定义为“长期边界伴侣”，强调其在多阶段医疗旅程中的持续共伴与交互动态；提出行为指导、信息桥接、情感陪伴和认知支架四个动态角色框架，并揭示其如何重塑患者主动性、信任与权力关系。

**🔧 技术方法**

利用现有通用与医疗特化LLM（ChatGPT、Doubao、DeepSeek、Kimi等）进行交互；采用长时序日记采集与半结构化访谈，结合主题分析法对数据进行编码与主题归纳。

**📊 数据集**

25名中国患者（年龄19–56岁，AI素养平均5.8/7）在四周内共记录587条日记条目（含LLM单独、LLM+医护、单独医护交互），并完成后续30–90分钟访谈。

**📈 对比分析**

研究不以对比实验或性能指标为主，而是通过定性主题分析评估LLM在四个维度的作用。结果显示：LLM显著提升患者决策参与度、访谈前准备效率、情绪安抚与术后知识理解，整体提升患者体验与自主性，但缺乏量化性能评估。

**⚠️ 局限性**

局限性包括样本规模有限、样本集中在中国资源受限医疗环境、受访者AI素养相对较高；未进行信息准确性或安全风险的定量评估；LLM长时序记忆不足，无法完全捕捉跨会话连贯性；研究聚焦患者视角，未系统考察医护人员体验。

---

## 70. AutoWebWorld: Synthesizing Infinite Verifiable Web Environments via Finite State Machines

**arXiv ID:** 2602.14296 | [PDF](https://arxiv.org/pdf/2602.14296v1)

**作者:** Yifan Wu `[一作]` (Hong Kong University of Science and Technology), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1677 | [OpenAlex ID](https://openalex.org/A5100614732)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 AutoWebWorld 框架，通过有限状态机（FSM）生成可验证的合成网页环境，自动化收集并验证 GUI 交互轨迹，生成 11,663 条高质量训练数据。

**💡 创新点**

将网页交互建模为可观测的 FSM，实现了无外部验证器的程序化轨迹生成与验证，并通过 FSM 规模与难度灵活控制数据量与环境难度。

**🔧 技术方法**

采用多代理 FSM 生成与校验、编码代理将 FSM 转译为 Vue 前端、BFS 枚举轨迹、Playwright 执行回放、GRPO 训练、LLM（Gemini3-Pro、GPT‑5.1）等技术。

**📊 数据集**

使用了 29 个自合成网页环境及其 11,663 条验证轨迹（约 16k 步 GRPO 训练数据），并在 WebVoyager、Online‑Mind2Web、ScreenSpot‑V2/VPro 进行评估。

**📈 对比分析**

以 WebVoyager 15 步成功率为基准进行对比，AutoWebWorld 训练的 7B 版在 27.42% 的成功率上优于同规模基线，并在 WebVoyager 与 Online‑Mind2Web 上呈现显著的规模曲线，数据量提升可持续提升性能。

**⚠️ 局限性**

生成过程仍依赖多代理 LLM 的校验，合成环境与真实网站可能存在差异，导致跨域泛化受限，且生成成本与模型依赖仍是实现中的瓶颈。

---

## 71. Optimal Regret for Policy Optimization in Contextual Bandits

**arXiv ID:** 2602.13700 | [PDF](https://arxiv.org/pdf/2602.13700v1)

**作者:** Orin Levy `[一作]` (School of Computer Science and AI, Tel-Aviv University), Yishay Mansour `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于策略优化的高效算法，在离线函数逼近框架下解决随机上下文多臂赌博机（CMAB）问题，并给出了高概率最优 regret 上界 O(√(K|A| log|F|))。

**💡 创新点**

① 将策略优化方法首次引入 CMAB 并证明可获得最优 regret；② 设计了针对随机策略的对数型探索奖励，兼容离线回归估计；③ 在不依赖 Eluder 维数等额外假设的前提下实现最优 bound；④ 与传统 UCB/IGW 方法形成对照，展示了策略优化的理论优势。

**🔧 技术方法**

离线最小二乘回归 oracle、对数型探索奖金、指数加权（在线镜像下降）策略更新、KL 散度作为 Bregman 散度、统一收敛性与误差分析。

**📊 数据集**

在 VowpalWabbit benchmark 中使用了 18 个来自 OpenML 的多类别/多标签数据集，主要以数据集 1084、1062、1015 为示例进行详细实验。

**📈 对比分析**

与 VowpalWabbit 自带的 CMAB 算法、Linear+Logistic、Linear+Squared 以及监督学习基线进行比较。实验显示该算法在多数数据集上与现有 SOTA 算法保持同等或略逊的 PV‑loss，整体性能保持竞争力，但未表现出显著优势。

**⚠️ 局限性**

实现上存在 O(K²) 的运行时复杂度，限制了规模化应用；理论假设要求离线函数逼近类可实现且有限，未覆盖无限上下文或函数类；未来需进一步优化算法效率并拓展至更一般的 RL 场景。

---

## 72. ML-ECS: A Collaborative Multimodal Learning Framework for Edge-Cloud Synergies

**arXiv ID:** 2602.14107 | [PDF](https://arxiv.org/pdf/2602.14107v1)

**作者:** Yuze Liu `[一作]` (Tongji University), Feng Xia `[通讯]` (RMIT University)

**通讯引用:** 19351 | [OpenAlex ID](https://openalex.org/A5089615958)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了ML‑ECS框架，在边缘设备和云端联合训练多模态模型，解决模态异质性与模型结构异质性

**💡 创新点**

创新点在于四个模块——跨模态对比学习、适应性多模态调优、模态感知聚合和基于SLM的跨模态对比学习，实现双向知识迁移并对缺失模态进行加权聚合

**🔧 技术方法**

利用LoRA参数高效微调、向量体积度量的跨模态对比损失、软提示调优以及自适应权重聚合等技术

**📊 数据集**

使用公开大规模多模态数据集VAST（文本/视频/音频摘要）和UR‑FALL（RGB‑深度+加速度跌倒分类）进行实验

**📈 对比分析**

与Standalone、Multi‑FedAvg、Co‑PLMs、FediLoRA、FedMLLM等基线对比，平均在Rouge‑LSum提升5.4–12.1%、BERTScore提升0.7–9.2%、F1提升2.9–8.2%，在服务器端同样获得显著提升，通信量仅为模型总参数的0.65%

**⚠️ 局限性**

局限性在于对模态缺失的假设为静态，未考虑时变模态可用性和更极端的异质性，且在极低模态覆盖率下仍可能出现性能下降

---

## 73. Diagnosing Pathological Chain-of-Thought in Reasoning Models

**arXiv ID:** 2602.13904 | [PDF](https://arxiv.org/pdf/2602.13904v1)

**作者:** Manqing Liu `[一作]` (Harvard University), Edward James Young `[通讯]` (Geodesic Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套轻量级、任务无关的链式思考（CoT）健康度量，用来识别后置推理、编码推理和内部化推理三种路径病。

**💡 创新点**

创新点在于：①通过设计三种病理模型（model organisms）验证指标可行性；②提出必要性（Necessity）、可改写性（Paraphrasability）和实质性（Substitutivity）三种因果干预指标，提供连续评分而非二元判别；③利用Cohen’s d对比训练进程中的病理表现。

**🔧 技术方法**

技术主要包括：对CoT进行剔除、改写、替换等干预，基于对答案的log概率计算度量；使用LoRA+监督微调在大型模型上训练病理模型；使用GPT‑4o‑mini或Gemini‑2.0‑Flash生成改写文本。

**📊 数据集**

使用了三大推理数据集：Binary Alternation、Calendar Arithmetic、Largest Island，这些任务在Qwen3‑4B上从无CoT 18–15% 提升至 CoT 90%+ 的准确率。

**📈 对比分析**

通过与健康基线模型的Cohen’s d 对比，三种度量在各自病理模型上呈现预期的正负信号，并能在不同训练检查点以不同速度识别病理，验证了方法的有效性。

**⚠️ 局限性**

局限性包括：仅适用于在文本空间产生中间推理的模型；干预可能将模型推向分布外导致误判；未覆盖所有类型的CoT不忠诚；基线假设原始模型无病理可能不成立。

---

## 74. Explanatory Interactive Machine Learning for Bias Mitigation in Visual Gender Classification

**arXiv ID:** 2602.13286 | [PDF](https://arxiv.org/pdf/2602.13286v1)

**作者:** Nathanya Satriani `[一作]`, Matthias Zeppelzauer `[通讯]` (Institute of Creative Media Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了交互式解释学习（XIL）在视觉性别分类中的偏差缓解效果，比较了CAIPI、RRR及其混合策略。

**💡 创新点**

提出将CAIPI与RRR结合的混合方法，并在MS COCO子集上系统评估其在减少性别偏差、提升模型对相关特征关注方面的优势。

**🔧 技术方法**

使用GradCAM与BLA生成解释，CAIPI数据增强、RRR损失以及EfficientNet-B0等CNN模型，并以人像分割掩码作为用户反馈。

**📊 数据集**

使用手工标注的MS COCO子集（1830张单人图像，男女各915张），并采用人像分割掩码。

**📈 对比分析**

通过准确率、FFP、BFP、BSR、DICE等指标对比，发现CAIPI和混合策略在关注前景、降低背景关注度方面最优；混合策略在公平性上表现最佳，准确率略下降但CAIPI在k=1时可提升。

**⚠️ 局限性**

局限性包括用户交互被模拟、依赖精确分割掩码（实际应用需放宽）、仅在单一手工数据集上实验，泛化性有限。

---

## 75. Traffic Simulation in Ad Hoc Network of Flying UAVs with Generative AI Adaptation

**arXiv ID:** 2602.13200 | [PDF](https://arxiv.org/pdf/2602.13200v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 76. Thermal Min-Max Games: Unifying Bounded Rationality and Typical-Case Equilibrium

**arXiv ID:** 2602.14858 | [PDF](https://arxiv.org/pdf/2602.14858v1)

**作者:** Yuma Ichikawa `[一作]` `[通讯]` (Fujitsu Limited), Yuma Ichikawa (Fujitsu Limited)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究零和最小-最大游戏的典型案例，提出热学最小-最大游戏并用双温度模型探讨玩家有限理性。

**💡 创新点**

通过引入两层热力学松弛与温度参数，实现从完全理性到有限理性的统一框架，并构造嵌套复制框架对典型均衡进行解析。

**🔧 技术方法**

采用嵌套复制方法、Replica Symmetry 与 1RSB 假设、热力学极限与随机高维矩阵理论。

**📊 数据集**

主要以独立同分布的高斯支付矩阵为实验案例，数值实验在中等规模（如 M=200）随机矩阵上进行。

**📈 对比分析**

将理论预测与精确线性规划求解的 Nash 均衡结果进行对比，理论与数值在游戏价值、支持比例、二阶矩等指标上吻合良好。

**⚠️ 局限性**

复制方法的解析延拓假设缺乏严格证明，且当前结果仅在高斯或对称随机矩阵下给出闭式表达，难以直接推广到结构化或稀疏支付矩阵。

---

## 77. MotionWeaver: Holistic 4D-Anchored Framework for Multi-Humanoid Image Animation

**arXiv ID:** 2602.13326 | [PDF](https://arxiv.org/pdf/2602.13326v1)

**作者:** Xirui Hu `[一作]` (Xi’an Jiaotong University), Weizhan Zhang `[通讯]` (Xi’an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研发了一套名为MotionWeaver的端到端框架，用于多人体形（多humanoid）图像动画，能够在复杂交互与遮挡场景下生成高质量视频；

**💡 创新点**

核心创新在于：①统一运动表示（UCC）将身份无关的运动与各角色绑定；②4D共享空间融合（HSI）通过深度感知和动态RoPE实现多角色的时空对齐；③层次4D监督（H4S）在不同噪声阶段分别引入遮挡与运动监督，提升运动捕捉与视觉一致性；

**🔧 技术方法**

采用Diffusion Transformer作为基底，结合SMPL 3D姿态估计、RoPE/动态RoPE、Depth‑Aware Attention、Group Attention、AdaLN等技术实现运动-特征融合与高阶监督；

**📊 数据集**

使用自己构建的46小时多人物视频数据集MultiHuman46（包含多种交互场景）以及300段双人互动视频的DualDynamics基准进行训练与评测；

**📈 对比分析**

在DualDynamics基准上与多种先进单人/多人的动画方法（RealisDance‑DiT、UniAnimate‑DiT、Animate‑X、HumanVid等）进行对比，使用L1、PSNR、SSIM、LPIPS、DISTS、CLIP、FID、FID‑VID、FVD等指标，MotionWeaver在所有指标上均优于对照组，显著提升了身份保持、运动一致性和遮挡处理；

**⚠️ 局限性**

局限性包括：训练成本高（需8个H100 GPU、长时间训练）；对人类外的非人形体（如机器人、动物化身）尚未充分验证；在极端多角色或高度遮挡场景下仍可能出现细节失真；仅在人工标注的真实人类视频上训练，缺乏对合成或极端光照场景的泛化能力。

---

## 78. Noncooperative Virtual Queue Coordination via Uncertainty-Aware Correlated Equilibria

**arXiv ID:** 2602.14436 | [PDF](https://arxiv.org/pdf/2602.14436v1)

**作者:** Jaehan Im `[一作]` (University of Texas), Ufuk Topcu `[通讯]` (University of Texas)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种在航空公司成本不确定性下的机会约束相关均衡（CC-CE）框架，用于协同虚拟队列（CVQ）中的非合作协调；

**💡 创新点**

创新点在于将相关均衡与机会约束结合，提供概率性激励兼容性保证，并通过低秩相关均衡（RRCE）实现可扩展的求解；

**🔧 技术方法**

采用相关均衡、机会约束、低秩逼近、线性规划和纯纳什均衡枚举等技术；

**📊 数据集**

使用合成航班调度数据（按泊松分布生成起飞航班、航班等级与航空公司随机分配），模拟不同时段和噪声水平；

**📈 对比分析**

将CC-CE、RRCE、全中心化、FCFS等方法进行对比，实验显示在最高交通负载（每小时210个起飞候机）下，CC-CE和RRCE相较于FCFS可降低6–9%的累计延误；

**⚠️ 局限性**

局限包括：纯纳什均衡在高不确定性下可能不存在，置信水平与可行解空间的权衡尚未系统化，且实验基于模拟数据，实际机场环境可能带来额外挑战。

---

## 79. Steady-State Behavior of Constant-Stepsize Stochastic Approximation: Gaussian Approximation and Tail Bounds

**arXiv ID:** 2602.13960 | [PDF](https://arxiv.org/pdf/2602.13960v1)

**作者:** Zedong Wang `[一作]` (Georgia Institute of Technology), Siva Theja Maguluri `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1519 | [OpenAlex ID](https://openalex.org/A5021806638)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了固定步长的随机逼近（SA）算法的稳态特性，提供了在固定步长下稳态与高斯分布之间的显式非渐近误差界限。

**💡 创新点**

创新点在于为固定步长的SA算法提供了明确的Wasserstein距离误差界限，并扩展了分析到马尔可夫噪声模型，涵盖了多种SA设置。

**🔧 技术方法**

使用了Stein方法结合时序相关性控制的Poisson方程技术，分析了稳态的Wasserstein距离和尾概率。

**📊 数据集**

研究中使用了多种数据集，包括平滑强凸目标的随机梯度下降（SGD）、线性SA和收缩非线性SA。

**📈 对比分析**

通过与高斯尾概率的比较，得到了非均匀的Berry-Esseen类型尾界限，性能表现为在小步长下，Wasserstein距离的误差界限为α^1/2log(1/α)。

**⚠️ 局限性**

限制在于对于固定步长的分析，未能提供在步长较大时的有效界限，且在马尔可夫噪声下的假设条件较为严格。

---

## 80. Statistical Early Stopping for Reasoning Models

**arXiv ID:** 2602.13935 | [PDF](https://arxiv.org/pdf/2602.13935v1)

**作者:** Yangxinyu Xie `[一作]` (University of Pennsylvania), Edgar Dobriban `[通讯]` (University of Pennsylvania)

**通讯引用:** 1209 | [OpenAlex ID](https://openalex.org/A5031235093)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两种基于推理过程中文本不确定性关键词出现统计的LLM推理早停方法，能够在黑盒API下实现无监督、可解释且具有统计学误报控制的推理长度调节。

**💡 创新点**

创新点在于：①使用半监督关键词构建方法提取可解释的不确定性词汇；②分别提出参数化（renewal过程检验）与非参数化（maxwise conformal）两种早停规则，并给出FPR控制的理论保证；③不需要内部激活或额外标签，兼容所有黑盒LLM。

**🔧 技术方法**

核心技术包括随机森林特征选择用于关键词抽取；renewal过程理论与序列检验配合Sidák校正；maxwise conformal预测框架；对齐检查间隔B的调节；以及对不同数据集的统计检验与阈值校准。

**📊 数据集**

主要实验数据集：GSM8K（训练/校准）、GSM-MC、UMWP、MiP、MMLU（数学推理），以及额外的科学推理任务；这些数据覆盖多模型（DeepSeek、Qwen、Nemotron、MiMo、Skywork等）。

**📈 对比分析**

与prompting（confidence/criticism）、长度阈值、DEER/entropy、线性probe/PLS等基线对比，实验表明：FPR保持在设定水平（≈3‑5%），Power（对不合理查询的提前停止）高达60‑70%，Token savings高于80%，在Oracle上限（约82‑85%）附近；在分布迁移场景下仍保持稳健。

**⚠️ 局限性**

局限性：①仅利用推理文本，若模型未在文本中显式表达不确定性，检测能力受限；②检查间隔B的设定影响检测功率与效率；③对极短推理或极端分布偏移仍可能出现误报或漏报；④方法依赖于足够多的well‑posed校准样本以估计统计量。

---

## 81. Colimit-Based Composition of High-Level Computing Devices

**arXiv ID:** 2602.14904 | [PDF](https://arxiv.org/pdf/2602.14904v1)

**作者:** Damian Arellanes `[一作]` `[通讯]` (Lancaster University), Damian Arellanes (Lancaster University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出并实现了基于范畴论有限余极限（pushout、coproduct）的高层计算模型 computon，支持显式控制流与数据流的分离，提供序列、并行（同步、异步）和分支（开闭）等组合操作，并给出了操作语义和 Idris 2 语言的实现。

**💡 创新点**

创新点在于：① 将计算设备与控制流显式分离并用有限余极限构造组合操作；② 通过加入计算设备集合 B 与映射 Ir↠OB 实现功能计算；③ 提出了新的分支操作（open/closed）并证明其代数性质；④ 证明同步并行可以由序列化与异步并行构造；⑤ 提供了完整的操作语义与可执行的 Idris 2 实现。

**🔧 技术方法**

技术包括：范畴论中的余极限（pushout、coproduct）；有限集合与总函数的 Fin‑type 表示；Idris 2 的依赖类型与函数式编程；操作语义基于状态转移与选择函数；通过 web‑service 调用实现计算设备的互操作性。

**📊 数据集**

本工作未使用传统实验数据集，而是基于理论模型与自定义的简单算子（乘法、加法、后继等）进行功能演示，所有演示均在 Idris 2 环境中通过小型实例验证。

**📈 对比分析**

由于研究侧重理论与实现，没有进行大规模性能对比；实验仅展示了模型构造与执行的正确性，未与其他 MHC 或并行框架做速度/资源消耗比较。

**⚠️ 局限性**

局限性包括：① 未提供循环（迭代）构造；② 对概率/非确定性选择缺乏支持；③ 某些组合操作（如同步并行）缺乏自体单位；④ 依赖 Idris 2，缺乏跨语言的完整编译器支持；⑤ 目前仅演示了小规模例子，未对大规模系统进行验证。

---

## 82. Integrating Affordances and Attention models for Short-Term Object Interaction Anticipation

**arXiv ID:** 2602.14837 | [PDF](https://arxiv.org/pdf/2602.14837v1)

**作者:** Lorenzo Mur Labadia `[一作]`, Antonino Furnari `[通讯]` (University of Catania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出STAformer与STAformer++两种基于Transformer的短期对象交互预测模型，并将环境可供性与交互热点融合以提升预测精度。

**💡 创新点**

创新点包括：① 设计帧引导的时序池化、双向图像‑视频注意力与多尺度特征融合；② 在模型训练与推理中动态利用环境可供性记忆；③ 使用DETR端到端检测头取代传统Fast‑RCNN，提升定位质量。

**🔧 技术方法**

采用的技术有：DINOv2、Swin‑T、TimeSformer、EgoVideo等Transformer编码器；DETR检测框架；帧引导注意力、双向跨注意力、对比去噪训练；可供性数据库匹配与交互热点预测。

**📊 数据集**

使用的公开数据集为Ego4D（v1与v2版本的STA标注）以及扩充版EPIC‑Kitchens STA。

**📈 对比分析**

通过与Faster‑RCNN、StillFast、GANO、EgoVideo等多种基线在Top‑5 mAP/All指标下对比，STAformer++在Ego4D v1、v2以及EPIC‑Kitchens分别取得最高N、N+V、All mAP（如N mAP 33.21/37.41/45.34，All mAP 4.66/6.26/8.67），在各数据集均为最优。

**⚠️ 局限性**

局限性：模型对视频长度敏感（EgoVideo仅能处理4帧，STAformer++仅fine‑tune少量层）；对极端视角或遮挡仍易失效；环境可供性数据库需要预先构建与维护；长尾类仍受限，需进一步提升泛化能力。

---

## 83. Diversity vs Degrees of Freedom in Gaussian Fading Channels

**arXiv ID:** 2602.14371 | [PDF](https://arxiv.org/pdf/2602.14371v1)

**作者:** Mahesh Godavarti `[一作]` `[通讯]` (Independent Researcher), Mahesh Godavarti (Independent Researcher)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究并统一了无线信道中自由度（DOF）与多样性（diversity）的几何定义，并通过Bhattacharyya包络提出新的可衡量指标（Gauge‑DOF 与 B‑diversity），从而在不同的信道尺度（log、loglog、(log)^β）上描述容量与可靠性之间的交叉量表关系。

**💡 创新点**

① 用一次性双线性映射 HX 的秩实现 DOF 与多样性的一致几何表述；② 引入交叉量表（cross‑gauge）现象并给出精确匹配上下界；③ 将 Bhattacharyya 距离与包络复杂度结合，得到 Gauge‑DOF 与 B‑diversity 的统一框架；④ 对非协同快衰落、多径、分数对数等非经典尺度的信道进行完整分析。

**🔧 技术方法**

几何秩分析、Bhattacharyya 距离、包络复杂度（packing complexity）、球面/格子包络理论、桥接引理（bridge lemma）、多尺度可测量（gauge identification）等。

**📊 数据集**

无具体实验数据集，全部基于理论推导与信息理论极限分析。

**📈 对比分析**

通过与传统容量预对数（prelog）与经典多样性阶（diversity order）的对比，证明传统定义在非 log gauge 下会给出无意义的零值；B‑diversity 在所有尺度下均提供有限且信息量丰富的可靠性指数；在同量表信道（如协同 MIMO、块衰落、分数对数）下，与 Zheng‑Tse DMT 等经典结果保持一致；在交叉量表信道（如快衰落）下给出全新的容量与可靠性曲线，且上下界匹配，展示理论精度与完整性。

**⚠️ 局限性**

1) B‑diversity 仅为可实现的可靠性指数，未必是最优误差指数；2) 对分数对数信道在 r>0 的可靠性曲线仍是开放问题；3) 在块衰落信道的可靠性下界与上界间仍存在 M 倍因子；4) 本框架目前仅适用于高斯衰落，非高斯、干扰或部分 CSI 场景需进一步扩展。

---

## 84. Multi-Modal Sensing and Fusion in mmWave Beamforming for Connected Vehicles: A Transformer Based Framework

**arXiv ID:** 2602.13606 | [PDF](https://arxiv.org/pdf/2602.13606v1)

**作者:** Muhammad Baqer Mollah `[一作]` (University of Massachusetts Dartmouth), Hua Fang `[通讯]` (Yeshiva University)

**通讯引用:** 7130 | [OpenAlex ID](https://openalex.org/A5100876783)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于Transformer的多模态感知与融合框架，用于60 GHz mmWave V2I/V2V车辆链路的beam selection预测top‑k波束；

**💡 创新点**

创新点在于将GPS、视觉和LiDAR三种感知模态融合，并通过跨模态多头注意力学习模态间的相关性，提前预测候选波束，显著减少beam搜索空间和延迟，同时实现与5G‑NR标准的兼容；

**🔧 技术方法**

采用Transformer架构，视觉编码器MaxViT，点云编码器PTv3，跨模态多头注意力，多层感知网络，softmax+交叉熵损失，Adam优化，PyTorch实现；

**📊 数据集**

使用DeepSense 6G真实场景数据集（V2I Day/Night、V2V Day/Night）收集的60 GHz mmWave测量数据；

**📈 对比分析**

与单模态GPS基线和简单拼接融合基线对比，top‑15准确率达96.72%（vs 91.31% vs 61.82%），平均功率损耗0.77 dB（vs 0.85/1.33 dB），延迟与搜索空间分别降低86.81%/76.56%；

**⚠️ 局限性**

局限在于仅覆盖城市日夜LOS场景，缺乏郊区/农村或NLoS环境；对天气和动态环境适应性有限；模型训练需要大量样本，实际部署需持续微调；推理时延仍有一定成本。

---

## 85. NutVLM: A Self-Adaptive Defense Framework against Full-Dimension Attacks for Vision Language Models in Autonomous Driving

**arXiv ID:** 2602.13293 | [PDF](https://arxiv.org/pdf/2602.13293v1)

**作者:** Xiaoxu Peng `[一作]` (Harbin Institute of Technology), Anupam Chattopadhyay `[通讯]` (Nanyang Technological University)

**通讯引用:** 6161 | [OpenAlex ID](https://openalex.org/A5089860351)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对自动驾驶中的视觉语言模型（VLM）构建了一套自适应防御框架 NutVLM，能够实时检测并清洗局部物理攻击与全局噪声，随后通过提示调优（EAPT）修正语义偏差，保障驾驶决策安全。

**💡 创新点**

创新点在于（1）NutNet++实现三向检测（干净/局部/全局），并根据检测结果自动切换到灰度遮罩或提示调优两条防御分支；（2）EAPT通过冻结 CLIP 经验模型在潜在空间进行梯度优化，生成“纠正驾驶提示”，无需对 VLM 进行完整微调，显著降低延迟并提升对全尺寸攻击的泛化能力。

**🔧 技术方法**

核心技术包括：基于重建误差的三指标（M_anom、H_energy、C_local）实现的三向分类；灰度遮罩纯化；梯度潜在优化 + 离散投影的 EAPT；以及 CLIP 作为跨模态专家进行语义一致性验证。

**📊 数据集**

实验使用了多种数据集：自动驾驶特化的 Dolphins、CADA（Scene-CADA 与 Obj-CADA 两子集）、通用的 APRICOT（局部物理攻击）和 DAmageNet（全局数字攻击），并在 InstructBlip、LLaVA、MiniGPT‑v4 等 VLM 体系上验证。

**📈 对比分析**

与 JPEG、TVM、MS、Bit‑Red、NRP、AAA 等传统防御以及 FGSM、PGD、ZOO、AdvCLIP、AnyAttack、SGA 等攻击手段比较，NutVLM 在 Dolphins 上全尺寸攻击时平均提升约 4.9%（Accuracy、Language Score、GPT Score），在 CADA 最高级别全局攻击下仍保持 41.7% 的 Final Score，局部攻击 GPT Score 最高达 56.3%（相较基线提升 3.8%），在不同 VLM 基础上保持 1–2% 的性能增益。

**⚠️ 局限性**

局限性包括：防御仅针对视觉模态，未覆盖语音、雷达等多传感器场景；EAPT 仍需额外的 CLIP 计算，极端高速场景下可能略低于纯硬件加速的纯图像预处理方法；对极大尺寸或高强度局部贴片的遮罩可能导致信息丢失，影响极端驾驶决策；未来需进一步验证在多模态、恶劣环境与全流程规划中的鲁棒性。

---

## 86. Multi-Agent Comedy Club: Investigating Community Discussion Effects on LLM Humor Generation

**arXiv ID:** 2602.14770 | [PDF](https://arxiv.org/pdf/2602.14770v1)

**作者:** Shiwei Hong `[一作]` (George Mason University), Zhicong Lu `[通讯]` (George Mason University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个多代理的喜剧俱乐部沙盒，探究公开社区讨论是否能提升LLM生成的单口喜剧稿件质量

**💡 创新点**

首次将社区讨论的实时反馈通过社会记忆检索机制回馈至后续写作，实现了公开讨论作为可操作的生成条件

**🔧 技术方法**

利用GPT‑4o‑mini模型、嵌入检索、束过滤、限定上下文、轮次控制与多代理交互框架

**📊 数据集**

自定义的50轮主题列表、5名表演者、3名评论者、20+观众代理生成的250条对照稿件及对应讨论线程

**📈 对比分析**

通过人工评估（A/B偏好与15项维度量表）比较，讨论条件在创作与社交反应上分别提升0.44和0.42分，75.6%被选为优胜者，整体效果显著

**⚠️ 局限性**

局限在于仅使用单一LLM模型、主题池有限、评估受文化偏好影响、可能出现内容敏感与风险性增加

---

## 87. Discrete Gene Crossover Accelerates Solution Discovery in Quality-Diversity Algorithms

**arXiv ID:** 2602.13730 | [PDF](https://arxiv.org/pdf/2602.13730v1)

**作者:** Joshua Hutchinson `[一作]`, Simón C. Smith `[通讯]` (Edinburgh Napier University)

**通讯引用:** 9469 | [OpenAlex ID](https://openalex.org/A5101825660)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并评估了两种基于离散基因级交叉的变异算子（IsoCross、IsoLineCross），以提升MAP‑Elites等质量‑多样性（QD）算法的搜索效率与多样性；

**💡 创新点**

通过将离散交叉与传统增量变异（如高斯噪声和方向性变异）结合，首次实现了快速携带优秀基因并在精英超体积之外探索新基因组合的机制；

**🔧 技术方法**

使用MAP‑Elites（CVT‑MAP‑Elites）框架、Poisson分布生成的交叉点、离散交叉算子、统计评估（QD score、coverage、max fitness、主成分维度等）以及Python实现的实验代码；

**📊 数据集**

在Brax连续控制环境中进行实验，涵盖HalfCheetah Uni、Hopper Uni和Walker2d Uni三种任务；

**📈 对比分析**

采用20个随机种子重复实验，与基线的Iso和Iso+LineDD算子对比；结果显示IsoLineCross在QD score、coverage和max fitness上均超过或与基线持平，尤其在后期迭代中产生更高质量的后代；

**⚠️ 局限性**

局限性包括：交叉概率和其他超参数固定未实现自适应调整；IsoCross在Walker2d任务中表现不佳；缺少对不同交叉策略（如k‑点交叉）或周期性交叉的探索。

---

## 88. Detecting Jailbreak Attempts in Clinical Training LLMs Through Automated Linguistic Feature Extraction

**arXiv ID:** 2602.13321 | [PDF](https://arxiv.org/pdf/2602.13321v1)

**作者:** Tri Nguyen `[一作]` (University of Cincinnati), Kelly Cohen `[通讯]` (University of Cincinnati)

**通讯引用:** 3249 | [OpenAlex ID](https://openalex.org/A5034113408)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发两层框架，用预训练Transformer微调回归模型自动提取四个关键语言特征（专业性、医学相关性、伦理行为、语境偏离），再用多种分类器判定临床LLM对话中的越狱行为。

**💡 创新点**

创新点在于：① 自动化特征提取替代人工标注，保持可解释性；② 两层模块化设计方便扩展和替换；③ 结合医学专用与通用预训练模型，提高跨域泛化能力。

**🔧 技术方法**

使用技术包括BERT、DistilBERT、RoBERTa、BioBERT等微调为回归器；随后利用决策树、随机森林、LightGBM、XGBoost、Logistic Regression、Gaussian Naïve Bayes等分类器；通过5折交叉验证与hold‑out评估进行性能比较。

**📊 数据集**

数据集来自2‑Sigma临床训练平台，共158个对话、2302条提示，包含越狱/非越狱二元标签以及四个语言特征的人工评分。

**📈 对比分析**

与先前基于人工特征的基准相比，系统在测试集上的准确率≈0.90、F1≈0.78、ROC‑AUC≈0.87，随机森林、LightGBM等模型表现最佳；交叉验证亦显示高稳健性。

**⚠️ 局限性**

局限性包括：① 依赖人工标签，标签不一致导致误差；② 四个特征未覆盖任务遵从、医学逻辑等维度，对隐蔽或细粒度越狱识别不足；③ 对短句、噪声或混合语境敏感，误判率仍存在。

---

## 89. A WDLoRA-Based Multimodal Generative Framework for Clinically Guided Corneal Confocal Microscopy Image Synthesis in Diabetic Neuropathy

**arXiv ID:** 2602.13693 | [PDF](https://arxiv.org/pdf/2602.13693v1)

**作者:** Xin Zhang `[一作]` (Manchester Metropolitan University), Rayaz Malik `[通讯]` (Weill Cornell Medicine)

**通讯引用:** 35327 | [OpenAlex ID](https://openalex.org/A5027250953)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一种基于WDLoRA的多模态生成框架，用来合成临床指导的角膜共聚焦显微镜（CCM）图像，以支持糖尿病周围神经病变（DPN）的自动诊断与分割；

**💡 创新点**

创新点在于引入Weight‑Decomposed Low‑Rank Adaptation（WDLoRA），通过解耦权重幅度与方向实现参数高效微调；同时将神经纤维分割掩模与临床文本提示联合作为多模态条件，生成结构与病理高度一致的医学图像，并通过三柱评估验证其临床可用性；

**🔧 技术方法**

采用Qwen‑Image‑Edit大型多模态扩散Transformer（MMDiT）作为生成骨干，配合WDLoRA进行参数高效微调；在latent空间训练扩散过程，并使用分割掩模、文本提示的联合编码作为条件；评估时使用FID、PSNR、SSIM及角膜神经纤维关键指标（CNFL、CNFD、CNBD、CNFW）等；

**📊 数据集**

使用来自曼彻斯特大学Early Neuropathy Assessment组的318例角膜共聚焦显微镜图像（384×384），按Healthy、T1NoDPN、T1DPN三类划分，并配有人工标注的神经纤维分割掩模；

**📈 对比分析**

与标准LoRA、GAN（SPADE）、从零训练的MAISI扩散模型以及原始Qwen‑Image‑Edit对比，WDLoRA在FID下降到5.18、SSIM提升至0.63、PSNR上升到30.46，且合成图像的临床指标与真实样本无显著差异；在DPN诊断与分割任务中，混合真实+合成数据可提升诊断准确率约2.1%和分割mIoU约2.2%；

**⚠️ 局限性**

局限性包括：训练数据仅来自单中心单设备，可能导致对不同仪器或机构的域迁移性能下降；分割掩模本身可能包含误差，影响生成质量；当前验证仅限于CCM-DPN任务，未扩展到其他眼科模态或更广泛的医学影像；未来需多中心、多模态验证与更复杂的病变建模。

---

## 90. Pareto and Bowley Reinsurance Games in Peer-to-Peer Insurance

**arXiv ID:** 2602.14223 | [PDF](https://arxiv.org/pdf/2602.14223v1)

**作者:** Tim J. Boonen `[一作]` (University of Hong Kong), Thai Nguyen `[通讯]` (Université Laval)

**通讯引用:** 414 | [OpenAlex ID](https://openalex.org/A5101923226)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文设计了一种包含内部风险互助与外部再保险的去中心化 P2P 保险方案，并通过两种游戏理论结构（Pareto 合作与 Bowley 领导者-跟随者）得到闭式最优合同。

**💡 创新点**

创新点在于：①将再保险定价与计划管理者的风险分配作为战略互动的核心；②使用可转移效用合作博弈证明核心非空，从而在 Pareto 设计中实现福利分配与协同稳定；③给出单加载限制对高风险成员的不对称影响与再保险效果的数值对比；④通过比较两种设计的整体福利与再保险覆盖率验证 Bowley 设计在非最优与福利低下。

**🔧 技术方法**

采用了均值-方差偏好、期望值保费原则、线性约束（零保留、精算公平）以及矩阵代数（正定、可逆）求解最优合同；博弈模型包括协作型 Pareto 与领导者-跟随者型 Bowley；核心分析利用可转移效用博弈的核心理论；数值实验基于三成员案例的均值-协方差矩阵与风险厌恶系数进行模拟。

**📊 数据集**

使用的“数据集”为仿真设定的三成员参数：期望损失 μ = [100,125,85]，协方差 Σ = [[10000,-1200,720],[-1200,14400,648],[720,648,8100]]，风险厌恶 γ = [0.015,0.025,0.02]，再保险方风险厌恶 γ_R = 0.01；通过这些参数在不同合同设计下计算再保险比例、保费、风险分配与福利提升。

**📈 对比分析**

比较方法：在相同成员集合下，分别构造无再保险、Pareto 优化（JPO1、JPO2）、Bowley 优化（BO1、BO2）四种合同，并在数值实验中计算再保险比例、保费、均值-方差不适用度、福利增量。结果显示：Pareto 合同在所有成员与再保险方上均实现更高的总福利与更低的风险不适用度；Bowley 合同虽保证个人理性，却导致更低的总福利并且在单加载限制下高风险成员获得不对称收益。

**⚠️ 局限性**

局限性：① Bowley 设计永不 Pareto 最优，缺乏对多再保险方或竞争环境的分析；②仅在离散时间与均值-方差框架下推导闭式解，无法直接推广至更复杂的风险度量或连续时间；③单加载限制的福利影响仅在特定参数下呈现，未全面探讨多重公平约束或动态调整；④实验仅基于单一三成员案例，缺乏大规模或真实保险数据验证。

---

## 91. Eureka-Audio: Triggering Audio Intelligence in Compact Language Models

**arXiv ID:** 2602.13954 | [PDF](https://arxiv.org/pdf/2602.13954v1)

**作者:** Dan Zhang `[一作]` (Baidu Inc), Haifeng Wang `[通讯]` (Baidu Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一个1.7B参数的轻量级音频-语言模型Eureka‑Audio，支持自动语音识别、音频语义与共情理解以及密集音频字幕等任务；

**💡 创新点**

采用Whisper音频编码器与稀疏Mixture‑of‑Experts（MoE）适配器实现对音频异质性的高效对齐，并结合DataFlux闭环音频指令数据合成与验证管线提升模型的语音情感推理与指令遵循能力；

**🔧 技术方法**

Whisper音频编码器、Qwen3‑1.7B LLM骨干、稀疏MoE适配器、两阶段预训练与SFT、DataFlux生成验证的指令数据、跨模态自回归下的下一个词预测；

**📊 数据集**

Stage1使用约100B tokens包含音频单模态、音频‑文本映射、交错、字幕等任务，Stage2约1T tokens并引入多种开源音频字幕数据集；SFT阶段包括ASR、语义理解、情感推理、密集字幕等；评测使用LibriSpeech、Fleurs、AISHELL、WenetSpeech、MMSU、OpenBookQA、MMAU、MMAR等数据集；

**📈 对比分析**

在同配置下与多种大型音频/多模态模型（如Qwen3‑30B、Qwen2.5‑7B、Step‑Audio‑8B、Kimi‑Audio‑7B、MiniCPM‑o等）进行公平对比，Eureka‑Audio在ASR、知识推理、安全、指令执行和共情理解等任务上与更大模型相当甚至超越（如MMAU 74.67分），且解码速度最快（269.7 tokens/s）；

**⚠️ 局限性**

受限于1.7B参数规模，虽性能优异但在极端多模态复杂任务、生成能力和极低延迟边缘设备部署方面仍有提升空间；DataFlux依赖多模型生成与自动裁判，可能引入数据偏差；缺乏对生成端的系统评测。

---

## 92. Fine-tuned Vision Language Model for Localization of Parasitic Eggs in Microscopic Images

**arXiv ID:** 2602.13712 | [PDF](https://arxiv.org/pdf/2602.13712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 93. MFN Decomposition and Related Metrics for High-Resolution Range Profiles Generative Models

**arXiv ID:** 2602.13296 | [PDF](https://arxiv.org/pdf/2602.13296v1)

**作者:** Edwyn Brient `[一作]` (Center for Mathematical Morphology), Rami Kassab `[通讯]` (Advanced Radar Concepts)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了HRRP数据的Mask‑Features‑Noise（MFN）三分解方法，并基于此设计了两种物理意义的评估指标；

**💡 创新点**

创新点在于：①首次将物理投影过程拆解为遮罩、低频特征和噪声三部分，提供可解释的分解；②基于分解设计了长度归一化的MSE以及针对兴趣细胞的余弦相似度，使评估不受目标大小和噪声偏差影响；

**🔧 技术方法**

采用了高斯平滑、遮罩提取、Hadamard乘积、归一化等信号处理技术，并实现了可微分的PyTorch代码；

**📊 数据集**

使用了含753艘船、90万条HRRP信号的海上雷达数据库，配合AIS信息标注了方位角等元数据；

**📈 对比分析**

通过在相同长度、相近方位角的船舶间比较top‑metric（基于角度窗口的最大值）与平均值，证明MFN‑cosine与MFN‑mse在区分同一船舶与不同船舶时表现出更高的差异性，表明其判别能力优于传统指标；

**⚠️ 局限性**

局限性包括：对“兴趣细胞”阈值的依赖、Gaussian滤波σ值选择对特征保留与噪声抑制的折衷、对不同雷达分辨率和噪声水平的泛化性尚待验证。

---

## 94. Inject Where It Matters: Training-Free Spatially-Adaptive Identity Preservation for Text-to-Image Personalization

**arXiv ID:** 2602.13994 | [PDF](https://arxiv.org/pdf/2602.13994v1)

**作者:** Guandong Li `[一作]` (iFLYTEK), Mengxia Ye `[通讯]` (Aegon THTF)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关、空间自适应的身份注入框架SpatialID，用于文本到图像的个性化生成

**💡 创新点**

创新点在于将身份注入从全局广播转为空间分离（面部区域锁定、背景区域释放），并设计了三阶段时空调度策略以匹配扩散模型的去噪过程

**🔧 技术方法**

利用跨注意力输出的L2范数自检掩码、中心高斯先验、可调软硬阈值，以及对Flux架构PerceiverAttention的无参数包装

**📊 数据集**

使用IBench基准（100个人ID×41个多样化文本，共4100张图像）进行评估

**📈 对比分析**

与7种现有方法（PuLID、InstantID、IP-Adapter等）对比，SpatialID在文本一致性CLIP‑T（0.281）、视觉一致性CLIP‑I（0.827）和图像质量IQ（0.523）上均获得SOTA表现，同时保持较高的身份保真度

**⚠️ 局限性**

局限在于对极端姿态或遮挡时掩码提取效果可能下降；中心高斯先验假设仅适用于面部居中情况；FaceSim略低于全局注入方法，需通过调参权衡

---

## 95. GSRM: Generative Speech Reward Model for Speech RLHF

**arXiv ID:** 2602.13891 | [PDF](https://arxiv.org/pdf/2602.13891v1)

**作者:** Maohao Shen `[一作]` (Massachusetts Institute of Technology), Jilong Wu `[通讯]` (Meta Superintelligence Labs)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了生成语音奖励模型（GSRM），用于评估和提升语音自然度。

**💡 创新点**

创新点在于将自然度评估拆分为可解释的声学特征提取和链式推理，使评估结果可解释并具备跨领域泛化。

**🔧 技术方法**

采用生成式奖励建模、链式推理技术，并结合人类反馈训练模型。

**📊 数据集**

使用了包含31k条专家评分的人类反馈数据集以及真实世界的用户-助手语音交互数据。

**📈 对比分析**

在多项评测中，GSRM的模型-人类相关性接近人类互评一致性，显著优于现有自然度预测器。

**⚠️ 局限性**

局限性包括对极端噪声或方言的适应性不足，以及模型在不同语言上的泛化仍需验证。

---

## 96. PT-RAG: Structure-Fidelity Retrieval-Augmented Generation for Academic Papers

**arXiv ID:** 2602.13647 | [PDF](https://arxiv.org/pdf/2602.13647v1)

**作者:** Rui Yu `[一作]` (Qilu University of Technology), Yinglong Wang `[通讯]` (Qilu University of Technology)

**通讯引用:** 13831 | [OpenAlex ID](https://openalex.org/A5100703788)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PT‑RAG，一种结构保真检索增强生成框架，利用学术论文的原始层级结构构建 PaperTree 索引并进行路径引导检索，解决传统 RAG 中的碎片化与误分配问题。

**💡 创新点**

创新点在于“inherit‑and‑navigate”范式：直接继承并利用原始论文层级结构，构建低熵检索先验，并通过根到叶子路径的检索与 token 预算匹配，实现高聚合、低熵的检索上下文。

**🔧 技术方法**

技术包括：PDF → Markdown 结构化转换、基于 LLM 的层级推断、结构锚定分段与摘要生成、双重语义相关性评分（原文与摘要）、基于 token 预算的路径优先检索与交叉编码重排序。

**📊 数据集**

使用三大学术 QA 基准：QASPER、QASA、M3SciQA；检索和生成采用 BAAI/bge‑reranker‑v2‑m3、GPT‑4o‑mini；文档解析采用 MinerU。

**📈 对比分析**

与 Naive RAG、GraphRAG、LightRAG、RAPTOR 对比。PT‑RAG 在所有数据集上 F1(Answer) 最高，BLEU‑1、ROUGE‑L 均大幅提升；同时 Section Entropy、Evidence Alignment Cross Entropy 下降，表明检索聚合度提升，效率与成本也显著低于对手。

**⚠️ 局限性**

局限性包括：仍依赖 LLM 进行层级推断与摘要，可能受模型误判影响；在多文档或跨领域场景下，单篇路径检索可能不足；实验仅在公开 QA 数据集上验证，缺乏对更大规模文档的评估。

---

## 97. General learned delegation by clones

**arXiv ID:** 2602.13262 | [PDF](https://arxiv.org/pdf/2602.13262v1)

**作者:** Darren Li `[一作]` (Tsinghua University), Jie Zhou `[通讯]` (Tencent Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SELFCEST，利用共享权重的克隆代理进行动态委托，根代理在固定推理预算下学习并分配子任务、上下文窗口和并行度，以提升推理效率与准确率。

**💡 创新点**

创新点在于将代理生成与工具调用统一为可学习的并行推理原语，采用共享参数多代理训练、全局奖励与轻量化的“回合门控”来解决多代理信用分配问题，并在单模型上实现自适应的并行与资源调度。

**🔧 技术方法**

主要技术包括强化学习（GRPO）+ 全局任务奖励、共享参数多代理 roll‑out、奖励塑形与回合门控、工具调用接口、以及 vLLM 与 VERL 的集成。

**📊 数据集**

使用了自定义算术数据集、MATH‑Hard、AIME、2WikiMultiHopQA 与 MuSiQuE 等长推理/长上下文任务数据集。

**📈 对比分析**

与基线（Qwen3‑4B、Qwen3‑4B‑Thinking、RL‑fine‑tuned、并行自一致性）对比，SELFCEST 在算术和 AIME 任务上在相同或更低的 token/延迟预算下提升 15–20% 的准确率，并在多跳 QA 上提升 5–10% 的 F1，显著改善准确率‑成本 Pareto 前沿。

**⚠️ 局限性**

局限包括：回合门控虽然稳定训练但引入偏差；信用分配仍依赖启发式；在极大 token 预算或更复杂任务上的泛化尚未充分验证；训练成本高，且在更大模型或更长上下文窗口下的可扩展性需进一步研究。

---

## 98. SecureGate: Learning When to Reveal PII Safely via Token-Gated Dual-Adapters for Federated LLMs

**arXiv ID:** 2602.13529 | [PDF](https://arxiv.org/pdf/2602.13529v1)

**作者:** Mohamed Shaaban `[一作]` (Washington State University), Mohamed Elmahallawy `[通讯]` (Washington State University)

**通讯引用:** 250 | [OpenAlex ID](https://openalex.org/A5067086571)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种名为 SecureGate 的联邦微调框架，通过可学习的授权 token 和双 LoRA 适配器，实现在保证隐私的前提下对大语言模型进行个性化微调。

**💡 创新点**

创新点在于：①将安全适配器与暴露适配器分离，并在推理时通过 token 控制门控动态路由，从而实现细粒度的访问控制；②结合联邦学习聚合、数据清洗和差分隐私等多重防护，显著降低 PII 泄露而不牺牲授权时的性能。

**🔧 技术方法**

使用技术包括：低秩 LoRA 适配器、token‑controlled MLP 门控、FedAvg/梯度聚合、数据清洗与差分隐私、两轮推理机制（门控前置 + 适配器推理）以及轻量级的通信与计算优化。

**📊 数据集**

实验数据集为 Yelp Reviews（用户评论）和 ECHR（欧洲人权法院案例）等真实文本数据。

**📈 对比分析**

与 FedAvg、FedAdagrad、FedAdam、FedYogi、FedAvgM 等基线比较，授权下平均推理准确率提升至 25.2%（相较于 20.12% 的最佳基线）；未授权泄漏率降至 4.2%（接近 3–5% 的保守基线）。模型 perplexity 与单适配器基线相当；在多种提取与推理攻击场景下，SecureGate 将泄漏率分别降低 31.66× 与 17.07×。

**⚠️ 局限性**

局限性包括：门控前置步骤增加推理延迟；多适配器设置提升显存占用；安全性高度依赖 token 的机密性与唯一性；实验仅覆盖解码器型 LLM，未验证至更大或其他架构。

---

## 99. Convergence of Differential Entropies -- II

**arXiv ID:** 2602.13493 | [PDF](https://arxiv.org/pdf/2602.13493v1)

**作者:** Mahesh Godavarti `[一作]` `[通讯]`, Mahesh Godavarti

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究概率密度函数序列在测度收敛时的微分熵收敛条件，提出以熵积分函数的均匀可积性与紧性为核心的通用判据，并给出了一个弱于以往固定指数条件的Orlicz型足够条件；同时构造反例否定了先前提出的GH猜想。

**💡 创新点**

1) 将熵收敛问题归结为熵积分函数的均匀可积性与紧性，提供统一且必要（在有界域上）且足够的判据。2) 引入单一的超线性Orlicz函数条件，严格弱于任意固定α>1的条件； 3) 通过构造显式反例证明GH猜想错误。

**🔧 技术方法**

Vitali收敛定理、Uniform Integrability与 Tightness理论、de la Vallée–Poussin判据、Orlicz空间理论。

**📊 数据集**

无实验数据集，纯理论分析。

**📈 对比分析**

无实验比较，论文通过理论推导与反例验证其结论。

**⚠️ 局限性**

目前仅在有界域上给出必要条件；在无界域上如何完全表征熵收敛的必要性仍为开放问题。

---

## 100. MoralityGym: A Benchmark for Evaluating Hierarchical Moral Alignment in Sequential Decision-Making Agents

**arXiv ID:** 2602.13372 | [PDF](https://arxiv.org/pdf/2602.13372v1)

**作者:** Simon Rosen `[一作]` (University of the Witwatersrand), Steven James `[通讯]` (University of the Witwatersrand)

**通讯引用:** 3517 | [OpenAlex ID](https://openalex.org/A5078861770)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Morality Chains 规范化框架和 MoralityGym benchmark，用于在强化学习中评估代理在层级化道德冲突下的决策行为。

**💡 创新点**

创新点在于将道德规范以可量化、可层级排序的形式表示，结合 deontic 强度与规范函数，构建可解释的“道德度量”并在 98 个类似电车难题的环境中进行系统基准测试。

**🔧 技术方法**

使用强化学习技术（PPO、PPO Shaped、PPO‑Lagrangian、CPO 等）、约束马尔可夫决策过程（CMDP）以及自定义的道德成本函数和道德度量来训练和评估智能体。

**📊 数据集**

使用 MoralityGym 提供的 98 个 Gymnasium 环境（基于电车难题变体）作为数据集，包含多种角色、道德层级与自我牺牲场景。

**📈 对比分析**

通过与随机策略、仅奖励训练、成本塑形、Lagrangian 与 CPO 等对比，利用平均归一化道德度量评估，结果显示成本塑形（PPO Shaped）在大多数道德链上表现最佳，CPO 亦能满足高优先级规范，但整体 Safe RL 方法仍难以在多层级道德冲突中保持一致性。

**⚠️ 局限性**

局限在于假设严格的规范层级顺序，未包含情感、发展和社会情境等心理因素；仅考虑单一因果规范，缺乏对责任、逆因果等更复杂道德情境的建模；以及无法处理冲突规范等“悲剧困境”。

---

## 101. On-Policy Supervised Fine-Tuning for Efficient Reasoning

**arXiv ID:** 2602.13407 | [PDF](https://arxiv.org/pdf/2602.13407v1)

**作者:** Anhao Zhao `[一作]` (Hong Kong Polytechnic University), Xiaoyu Shen `[通讯]` (Eastern Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于on‑policy监督微调（On‑Policy SFT）的高效推理方法，通过去除RL中的KL正则、组归一化等复杂机制，采用截断长度奖励实现简化训练。

**💡 创新点**

创新点在于：①证明GRPO等RL框架中的KL正则和组归一化在直接可验证的准确率与长度奖励任务中多余；②将多奖励问题化简为单一截断奖励，得到与监督微调等价的无奖励优化目标；③利用on‑policy采样、长度偏置校正和温度1.0的rollout策略，使训练既高效又稳定。

**🔧 技术方法**

技术手段包括：on‑policy生成数据、截断长度奖励（0/1）、max‑length长度偏置校正、rollout温度设为1.0、可调节的rollout数量、批量最大长度替代token‑level长度归一化，最终得到的目标与传统SFT相同。

**📊 数据集**

使用DeepScaleR训练集以及五个数学推理基准（GSM8K、MATH‑500、AMC23、AIME24、AIME25）进行评估，模型分别为DeepSeek‑R1‑1.5B与DeepSeek‑R1‑7B。

**📈 对比分析**

与训练免费、传统SFT、以及十余种RL基线（ThinkPrune、O1‑Pruner、L1、ER‑RL、LASER等）对比，On‑Policy SFT在保持甚至提升准确率的同时将CoT长度平均减少约70%‑80%，取得最高的eff（准确率/长度）得分；训练时间与GPU内存成本降低约50%；收敛速度提升约70%；长度控制（方差）显著优于RL方法。

**⚠️ 局限性**

局限性：①对on‑policy数据的依赖，离线/离策略训练效果不佳；②对温度、rollout数量、截断阈值等超参敏感，需经验调优；③目前仅验证于数学推理任务，对其它推理/生成任务的适用性未知；④截断奖励的硬阈值可能导致在极短/极长文本上性能波动。

---

## 102. Text Before Vision: Staged Knowledge Injection Matters for Agentic RLVR in Ultra-High-Resolution Remote Sensing Understanding

**arXiv ID:** 2602.14225 | [PDF](https://arxiv.org/pdf/2602.14225v1)

**作者:** Fengxiang Wang `[一作]` (National University of Defense Technology), Bo Du `[通讯]` (Wuhan University)

**通讯引用:** 29840 | [OpenAlex ID](https://openalex.org/A5060042752)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种“先文本后视觉”的阶段性知识注入策略：先用海量高质量地球科学文本问答进行冷启动SFT，再在同一批难度极高的超高分辨率图像-文本样本上进行预热SFT，随后利用Agentic RLVR结合缩放工具进行强化学习，显著提升UHR遥感任务的推理性能。

**💡 创新点**

①发现地球科学文本问答（无图像）是提升UHR遥感推理边界和平均表现的关键驱动力；②提出可扩展的文本QA生成与知识图谱质量控制流水线；③提出基于预热SFT的“Text‑Before‑Vision”阶段性训练模板，突破传统SFT+RL组合的瓶颈。

**🔧 技术方法**

采用冷启动监督微调（SFT）+自监督工具强化学习（Agentic RLVR）+Zoom‑Eyes缩放工具 + GRPO优化器 + chain‑of‑thought (CoT) 结构化推理；并利用知识图谱进行文本QA的预筛选与校验。

**📊 数据集**

地球科学文本问答数据集（148,777条 CoT 样本）；SuperRS‑VQA（12,228 条 UHR 图像‑文本样本）；DeepEyes‑47K（通用RL数据集）；XLRS‑Bench（UHR 评测基准）。

**📈 对比分析**

与 QwenVL2.5 7B 等基线模型在同一评测协议下对比，单轮平均表现（Pass@1）从 50.01% 提升至 60.40%；推理边界（Pass@32）从 82.58% 提升至 96.25%。在 XLRS‑Bench 上，模型超过更大规模的 Intern‑S1、Gemini、GPT‑5.2 等主流 MLLM，体现出高效的知识注入与强化学习协同效果。

**⚠️ 局限性**

① 依赖大量可验证的地球科学文本 QA，生成与质量控制过程仍需要人工或自动化管控；② 对其他遥感子领域或跨域任务的泛化能力尚未充分验证；③ 强化学习阶段仍可能出现不稳定性，需更精细的奖励设计与探索策略。

---

## 103. HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation

**arXiv ID:** 2602.14470 | [PDF](https://arxiv.org/pdf/2602.14470v1)

**作者:** Wen-Sheng Lien `[一作]` (National Yang Ming Chiao Tung University), Hong-Han Shuai `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于n-ary超图的检索增强生成框架HyperRAG，用于多跳问答。

**💡 创新点**

创新点包括：① 用n-ary超图避免二元分解导致的语义碎片化与路径爆炸；② 引入HyperRetriever通过结构-语义融合的MLP与自适应阈值搜索实现高效可解释链构建；③ 引入HyperMemory利用LLM参数记忆引导Beam Search；④ 通过密度感知与预算管理实现近O(1)检索开销。

**🔧 技术方法**

使用的技术包括：结构语义融合的MLP分类器、密度感知自适应阈值检索、LLM Prompt与Beam Search、GTE向量检索、文本与超图上下文拼接。

**📊 数据集**

采用的评估数据集有：WikiTopics-CLQA、HotpotQA、MuSiQue、2WikiMultiHopQA。

**📈 对比分析**

与RAPTOR、HippoRAG、ToG、HyperGraphRAG等基线对比，HyperRetriever在WikiTopics平均MRR提升2.95%、Hits@10提升1.23%；在HotpotQA、MuSiQue上EM/F1均优于对手；在2WikiMultiHopQA F1提升11.89%。

**⚠️ 局限性**

局限性包括：HyperMemory对LLM记忆的依赖导致性能不稳；对低密度超图仍需阈值调优；依赖高质量超图构建，难以适用于低资源或动态更新场景；尚未解决多模态知识集成和实时知识更新。

---

## 104. Scenario-Adaptive MU-MIMO OFDM Semantic Communication With Asymmetric Neural Network

**arXiv ID:** 2602.13557 | [PDF](https://arxiv.org/pdf/2602.13557v1)

**作者:** Chongyang Li `[一作]` (Central China Normal University), Shouyin Liu `[通讯]` (Central China Normal University)

**通讯引用:** 275 | [OpenAlex ID](https://openalex.org/A5050767262)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在下行多用户MIMO OFDM系统中提出一种端到端的语义图像传输框架，使用异构编码器/解码器、神经预编码与导引式注意力解码实现高效语义信息传输。

**💡 创新点**

核心创新包括（1）场景感知的异构编码器，可根据CSI与SNR动态调节特征抽取；（2）可学习的残差RZF神经预编码器，避免矩阵求逆不稳定；（3）基于导引式注意力的轻量化解码器，利用导频实现隐式信道均衡与特征校准；（4）整体框架在单台基站侧集中计算负担，UE侧保持极低复杂度。

**🔧 技术方法**

技术手段涵盖深度残差网络、深度可分离卷积、注意力机制、OFDM资源映射、正则化零强制预编码、端到端损失（MSE+交叉熵）以及基于3GPP TR 38.901的多场景仿真。

**📊 数据集**

采用CIFAR-10图像数据集进行重构与分类任务评估。

**📈 对比分析**

与传统SSCC（BPG+LDPC）和改进的DJSCC基线进行对比。实验表明在UMi/UMa/RMa三种场景下，该框架在低信噪比下实现PSNR提升3–5 dB、分类准确率提升10–15 %，并在所有带宽配置下保持优势；神经预编码显著降低训练不稳定性，轻量化解码器实现约98 %参数压缩。

**⚠️ 局限性**

局限性包括：仅在四天线基站与单天线UE的中小规模MIMO下验证；对极端高速移动或极大天线阵列的适应性未知；模型仍需进一步压缩以满足极低功耗终端；以及对动态频谱分配与多任务调度的探索不足。

---

## 105. Reinforcement Learning-Enabled Dynamic Code Assignment for Ultra-Dense IoT Networks: A NOMA-Based Approach to Massive Device Connectivity

**arXiv ID:** 2602.13205 | [PDF](https://arxiv.org/pdf/2602.13205v1)

**作者:** Sumita Majhi `[一作]` (Indian Institute of Technology Guwahati), Pinaki Mitra `[通讯]` (Indian Institute of Technology Guwahati)

**通讯引用:** 1981 | [OpenAlex ID](https://openalex.org/A5045420930)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种基于强化学习的动态Gold码分配方案，用于超密集IoT-NOMA网络，实现吞吐量、能效与公平性等多目标优化。

**💡 创新点**

创新点包括：① 针对IoT的多目标MDP建模与状态奖励设计；② 针对高维离散动作空间的自然策略梯度(NPG)与连续嵌入式DDPG算法；③ 将离散码映射到连续嵌入空间以兼容DDPG；④ 在三种实际部署场景中进行大规模仿真验证。

**🔧 技术方法**

主要技术：强化学习（NPG、DDPG）、Gold码多址、IoT感知MDP、多目标奖励函数、连续码嵌入、经验回放与自适应探索、对齐的稀疏特征表示。

**📊 数据集**

使用基于3GPP TR 38.901的实景信道模型、三种部署场景（智能城市、工业物联网、传感器网络）中生成的模拟数据，设备数分别为100、60和150，涵盖不同QoS等级与能量状态。

**📈 对比分析**

与静态、随机、基于SINR的贪婪分配等基线对比，评估吞吐量、能效、公平性与可靠性。结果显示，NPG在智能城市场景下吞吐量提升约11.6%，能效提升15.8%，DDPG亦有改进，但可靠性仅0–2%，低于URLLC要求。

**⚠️ 局限性**

主要局限：① 低可靠性表明单靠码分配不足以满足mission‑critical IoT需求；② 假设完美CSI与理想SIC；③ 仅单小区、固定时间步长；④ 对于设备>300时难以扩展；⑤ 需要进一步结合功率控制、ARQ或多连通等手段提升可靠性。

---

## 106. ThermEval: A Structured Benchmark for Evaluation of Vision-Language Models on Thermal Imagery

**arXiv ID:** 2602.14989 | [PDF](https://arxiv.org/pdf/2602.14989v1)

**作者:** Ayush Shrivastava `[一作]` (Indian Institute of Technology), Nipun Batra `[通讯]` (Indian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ThermEval 基准与 ThermEval-D 数据集，系统评估 VLM 在热图上的理解与推理能力

**💡 创新点**

首个包含像素级温度标注的人体热图数据集及围绕七项热视觉挑战的结构化 VQA 基准

**🔧 技术方法**

利用零样本提示、上下文增强提示、LLM 结构化解析以及在 Qwen‑VL 上的监督微调进行模型评测

**📊 数据集**

整合 FLIR、LLVIP 公开热图数据以及约 55,000 题的 ThermEval‑B 组合，ThermEval‑D 贡献 1,000 并含全像素温度与人体部位标注

**📈 对比分析**

在 25 个从 0.3B 到 235B 参数规模的 VLM 上做对比，模型在模态识别与颜色条定位上表现良好，但在温度推理、计数与绝对温度估计上误差大；微调后可达接近人类水平，但仍不足以满足安全关键应用

**⚠️ 局限性**

受限于缺乏原始温度矩阵、仅使用伪彩色图、模型对热信号对齐不足、LLM 解析误差以及基准仅覆盖基础任务等

---

## 107. We can still parse using syntactic rules

**arXiv ID:** 2602.14238 | [PDF](https://arxiv.org/pdf/2602.14238v1)

**作者:** Ghaly Hussein `[一作]` `[通讯]` (Independent Researcher), Ghaly Hussein (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于GPSG和CFG的解析方法，能够同时生成依存树和成分树，并能处理噪声和不完整解析。

**💡 创新点**

创新点在于将GPSG斜线特征与增量式规则扫描、重排序以及多重假设连接相结合，恢复了CFG对语言结构的限制，提供了可解释的双范式解析。

**🔧 技术方法**

采用自定义POS标注器、基于多键字典的短语索引、增量式规则投影、Dijkstra式连接以及重排序算法。

**📊 数据集**

使用Universal Dependencies（UD）英语树库的7个开发集和12个测试集进行实验。

**📈 对比分析**

与spaCy依存解析器对比，开发集UAS为54.5%（spaCy 56.9%），测试集为53.8%（spaCy 58.6%），在部分数据集上略优。

**⚠️ 局限性**

仅覆盖少量规则，未实现完整英语语法覆盖，实验仅限于UD英文数据，且缺乏对多语言适用性的验证。

---

## 108. Simulation-based Learning of Electrical Cabinet Assembly Using Robot Skills

**arXiv ID:** 2602.14561 | [PDF](https://arxiv.org/pdf/2602.14561v1)

**作者:** Arik Laemmle `[一作]`, Marco F. Huber `[通讯]` (Fraunhofer Institute for Manufacturing Engineering and Automation)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本研究通过在MuJoCo物理仿真环境中训练可参数化的机器人技能，并将学习到的策略迁移至UR10e机器人，实现了电气端子在DIN轨道上的力控装配；

**💡 创新点**

创新之处在于将柔性端子变形建模为解析与刚体两种方法，并构建了基于Pitasc框架的可参数化技能，利用域随机化在仿真中获得对工艺误差高度鲁棒的通用策略，显著降低人工编程成本；

**🔧 技术方法**

所采用的技术包括MuJoCo仿真、深度强化学习（SAC/T‑D3）、Pitasc技能框架、域随机化、力/扭矩传感器反馈以及机械定位装置；

**📊 数据集**

使用的数据主要为仿真中生成的数百万次技能执行样本以及真实端子装配实验测得的力与位移数据，用于模型校准与性能评估；

**📈 对比分析**

通过仿真与真实机器人实验比较，策略在仿真中实现100%成功率、平均所需技能数约为2，迁移至真实机器人后在±5 mm/±2°误差范围内也能保持100%成功率，明显优于传统编程方法；

**⚠️ 局限性**

局限性包括解析模型对侧向力的系统误差、刚体模型需极小时间步导致仿真速度慢，以及在大角度或极端力需求下的泛化受限，需要进一步改进柔性材料模型和环境参数自适应。

---

## 109. Pawsterior: Variational Flow Matching for Structured Simulation-Based Inference

**arXiv ID:** 2602.13813 | [PDF](https://arxiv.org/pdf/2602.13813v1)

**作者:** Jorge Carrasco-Pollo `[一作]` (University of Amsterdam), Jan-Willem van de Meent `[通讯]` (UvA-Bosch Delta Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Pawsterior，一种端点变分流匹配框架，用于改进受限域和离散结构下的模拟推断后验估计。

**💡 创新点**

创新点在于：①端点诱导的仿射几何约束，将后验几何直接嵌入流匹配；②双端点变分预测，提升数值稳定性并兼顾上下界；③实现对离散与混合参数空间的统一处理，扩展传统流匹配的适用范围。

**🔧 技术方法**

技术方法包括变分流匹配 (VFM)、端点条件分布建模、两侧端点变分学习、分类器两样本检验 (C2ST) 进行后验质量评估。

**📊 数据集**

实验数据集涵盖标准的 sbibm 模拟推断基准任务，以及自定义的切换高斯混合模型 (SGM) 任务，用于测试离散与混合参数场景。

**📈 对比分析**

与传统 FMPE 通过 C2ST 指标对比，Pawsterior 在有边界后验任务中显著提升后验逼近精度，连续任务亦更稳定；在 SGM 离散任务中 FMPE 近 1.0，而 Pawsterior 可达约 0.6，显示出明显性能优势。

**⚠️ 局限性**

局限性包括：端点分布假设（均值场）可能限制表达能力；需要为每种结构手动设计端点分布；对更复杂几何或高维边界的适应性尚待验证；模型容量与数据规模对性能影响仍显著。

---

## 110. Geometry-Aware Physics-Informed PointNets for Modeling Flows Across Porous Structures

**arXiv ID:** 2602.14108 | [PDF](https://arxiv.org/pdf/2602.14108v1)

**作者:** Luigi Ciceri `[一作]` (Università degli Studi di Milano-Bicocca), Gabriele Gianini `[通讯]` (Università degli Studi di Milano-Bicocca)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了几何感知的物理信息点云网络（PIPN）和物理信息几何感知神经算子（PI-GANO），用于同时预测流体和多孔介质中的稳态不可压流动，能够在不重新训练的情况下对不同几何形状、边界条件和材料参数进行泛化；

**💡 创新点**

首次将PINN与PointNet相结合，构建同时适用于Navier‑Stokes与Darcy‑Forchheimer模型的物理信息点云网络，并进一步扩展为可处理可变边界条件的神经算子；在复杂三维树冠与建筑混合几何体上实现了高精度预测；

**🔧 技术方法**

采用物理信息神经网络（PINN）、PointNet、Physics‑Informed PointNet（PIPN）和Physics‑Informed Geometry‑Aware Neural Operator（PI‑GANO）等深度学习技术；利用自动微分构建PDE残差损失；通过SDF和边界ID实现多孔介质编码；使用OpenFOAM生成训练数据；

**📊 数据集**

利用OpenFOAM对二维多孔通道和三维风障（树冠+建筑）进行CFD仿真得到的速度、压力数据；包括制造解（MMS）数据集、不同边界条件和材料参数的二维数据集，以及真实树冠三维数据集；

**📈 对比分析**

在制造解、固定边界条件和可变边界条件三类实验中，PIPN/PI‑GANO的平均绝对误差（MAE）在速度分量与压力上均低于10⁻³（二维）或10⁻⁷（三维），且模拟时间比OpenFOAM快两到三百倍；相比传统PINN在不同几何形状上表现出更好的泛化能力；

**⚠️ 局限性**

对高梯度区（尤其是u_x）表现不佳，接触面附近误差累积；仅适用于低速层流；缺乏湍流模型，无法直接推广到高雷诺数流动；对采样策略和网络结构（如PointNet++、Fourier Neural Operator）有进一步改进空间。

---

## 111. Algebraic Quantum Intelligence: A New Framework for Reproducible Machine Creativity

**arXiv ID:** 2602.14130 | [PDF](https://arxiv.org/pdf/2602.14130v1)

**作者:** Kazuo Yano `[一作]` (Hitachi), Koji Ara `[通讯]` (Happiness Planet)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出 Algebraic Quantum Intelligence (AQI)，通过引入非交换代数算子动态扩展语义空间，以增强大型语言模型的创造性输出。

**💡 创新点**

创新点在于将非交换算子、量子不确定性与创意价值 (C-value) 作为可设计的动力学原理，实现对创造性分支宽度的下界保证，并通过算子顺序与干涉效应系统性地产生多样化创意。

**🔧 技术方法**

技术实现包括两层架构：S-Generator 对语义状态进行更新，H-Generator 生成时变 Hamiltonian；采用600+专业算子与基于 Hilbert 空间的向量运算，结合非交换算子运算与 C-value 评估。

**📊 数据集**

数据集为自研的十领域创意管理推理基准（涵盖风险预测、投资决策、组织转型等），共计数千道开放式问题，评估时采用 LLM‑as‑a‑judge 模式。

**📈 对比分析**

与14个强基线模型（GPT‑5、GPT‑4o、Gemini、Claude 等）在 Co‑Creativity Index (CCI) 上对比，AQI 平均提升 27 T‑score，且跨领域方差最小，验证了非交换算子对创造性提升的显著性。

**⚠️ 局限性**

局限性包括对社会权力、文化摩擦等非结构化因素的处理不足、对算子设计与 Hamiltonian 调参高度依赖、仅验证于业务与决策场景，缺乏对更广泛创造性任务的推广与深度解释。

---

## 112. MAGE: All-[MASK] Block Already Knows Where to Look in Diffusion LLM

**arXiv ID:** 2602.14209 | [PDF](https://arxiv.org/pdf/2602.14209v1)

**作者:** Omin Kwon `[一作]` (Seoul National University), Jae W. Lee `[通讯]` (Seoul National University)

**通讯引用:** 3136 | [OpenAlex ID](https://openalex.org/A5100415717)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于 Block Diffusion LLM 的 All-Block 引导稀疏注意力方法（MAGE），通过在首个 All‑block 计算完整注意力并选取重要 KV 索引，实现后续去噪步骤的高效稀疏注意力，并给出了轻量化微调框架。

**💡 创新点**

创新点在于发现 Block Diffusion 的 All‑block 能准确预测后续步骤的关键 KV 位置和层级预算需求，从而实现单次精确注意力传递并在整个去噪过程中复用；以及通过自蒸馏的微调强化模型对 All‑block 关注的稳定性。

**🔧 技术方法**

使用的技术包括稀疏注意力、KV 缓存、层级自适应预算分配、All‑block 选取、轻量化微调（三阶段自蒸馏）和 FlashInfer 实现高效推理。

**📊 数据集**

主要数据集为 LongBench（包含单文档 QA、跨文档 QA、少样本学习、摘要任务）和 Needle‑in‑a‑Haystack（检索任务），同时在 Daring‑Anteater 长上下文子集上进行微调。

**📈 对比分析**

与 Exact Attention 以及 AR 方向的 Quest、Tidal 进行对比，MAGE 在保持 84–90% Top‑K recall 的同时，显著减少 KV 访问量，提供 3–4 倍的端到端加速；微调版 MAGE‑FT 在中等预算下甚至可超过 Exact Attention 的准确率。

**⚠️ 局限性**

局限性包括：需要在首个 All‑block 计算完整注意力产生一定的一次性开销；对 KV 缓存大小和块尺寸仍有一定依赖；微调过程虽轻量但仍需额外 GPU 训练；在极低预算或非常长上下文时稀疏效果仍可能下降。

---

## 113. Mitigating the Safety-utility Trade-off in LLM Alignment via Adaptive Safe Context Learning

**arXiv ID:** 2602.13562 | [PDF](https://arxiv.org/pdf/2602.13562v1)

**作者:** Yanbo Wang `[一作]` (University of Chinese Academy of Sciences), Ran He `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种自适应安全上下文学习框架（ASCL），通过多轮工具调用实现安全规则与推理过程的解耦；

**💡 创新点**

创新点在于将安全规则从模型推理中分离，采用多轮工具使用与逆频率策略优化（IFPO）来平衡安全与实用性，显著降低过度拒绝；

**🔧 技术方法**

技术手段包括基于ReAct的多轮上下文工具调用、行为克隆与强化学习奖励设计，以及逆频率优势重加权的IFPO算法；

**📊 数据集**

实验使用了Salad‑Bench、OR‑Bench‑80k、WildJailbreak、JBB、WildChat、WildGuardTest、XSTest、OKTest、OR‑Bench‑Hard 等安全数据集，以及 Math‑500、GSM8K、GPQA‑Diamond、MMLU‑Pro、ARC‑Challenge 等通用推理基准；

**📈 对比分析**

与 SafeChain、STAR‑1、纯提示/短提示等基线比较，ASCL+IFPO 在安全性与拒绝率平衡上取得最高 Pareto 前沿，安全分数提升约 15–20%，拒绝率降低；但对通用推理任务的提升有限；

**⚠️ 局限性**

局限性在于通用推理能力提升不显著，且仍需依赖精心构建的安全文档与复杂的后训练流程，可能不易在不同模型或任务上直接迁移。

---

## 114. A Pragmatic Method for Comparing Clusterings with Overlaps and Outliers

**arXiv ID:** 2602.14855 | [PDF](https://arxiv.org/pdf/2602.14855v1)

**作者:** Ryan DeWolfe `[一作]` (Toronto Metropolitan University), François Théberge `[通讯]` (Tutte Institute for Mathematics and Computing)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于 F* 评分的简洁、可计算且对覆盖率不敏感的聚类相似度度量 F^*_wo，用于比较可能存在重叠和离群点的聚类。

**💡 创新点**

创新点在于将 F*（即 Jaccard）与集匹配相结合，构造单向匹配后加权平均，再对称化并加入离群点匹配项，从而在处理重叠聚类和离群点时既保持标签不变性，又避免传统指标的偏置。

**🔧 技术方法**

技术实现依赖集合相似度函数（F*）、最大匹配、加权平均以及对离群点的额外匹配；算法复杂度为 O(mn)（m 为聚类数，n 为对象数），可直接用于图聚类的边/点视角。

**📊 数据集**

使用了人工基准 ABCD+o^2（可生成重叠与离群点的聚类）、传统的 1024‑对象随机聚类、以及 Leiden 算法在 ABCD 生成的图上得到的三种分辨率的聚类结果作为实验数据集。

**📈 对比分析**

与 Omega、oNMI、ECS 等现有指标相比，F^*_wo 在所有直观实验（分层变换、随机打乱、不同聚类数、重叠程度变化、离群点引入）中均表现出单调、无偏差的相似度变化；在图聚类评估中也能从点和边两视角给出一致的性能评估。

**⚠️ 局限性**

主要局限包括：① 1–F^*_wo 不是度量，无法满足三角不等式；② 对极端重叠或离群点比例极高时仍可能出现相对误差；③ 目前未在真实世界大型数据集上做进一步验证。

---

## 115. HiST-VLA: A Hierarchical Spatio-Temporal Vision-Language-Action Model for End-to-End Autonomous Driving

**arXiv ID:** 2602.13329 | [PDF](https://arxiv.org/pdf/2602.13329v1)

**作者:** Yiru Wang `[一作]` (Bosch Corporate Research), Hao Sun `[通讯]` (Bosch Corporate Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 HiST-VLA，一种集成 3D 空间感知与层次轨迹优化的 Vision‑Language‑Action 框架，用于端到端自动驾驶。

**💡 创新点**

创新点在于：① 基于自注意力的动态令牌稀疏化降低视觉冗余；② 细粒度元动作表述提升语义对齐；③ 层次化规划器通过置信度正则化与多准则评分实现安全、舒适的轨迹精细化。

**🔧 技术方法**

使用技术包括：LLaVA‑v1.5‑7B 预训练模型；深度感知空间编码；动态令牌稀疏化（eTPS 变体）；链式思维 CoT 推理；VAE 隐空间与跨模态注意力；多准则评分与置信度正则化。

**📊 数据集**

训练与评估数据集为 NAVSIM v2（Navtest 与 Navhard），输入包括多视角摄像机、深度估计、导航标记与车辆状态。

**📈 对比分析**

与 Transfuser、VADv2、GTRS‑Dence、DiffusionDrive 等 SOTA 进行对比，HiST‑VLA 在 Navtest 上实现 EPDMS 88.6（领先 86.2），在 Navhard 上取得 50.9，显著提升安全性与鲁棒性。

**⚠️ 局限性**

局限性包括：对大规模预训练模型和高质量多视角数据的高度依赖；推理时仍存在较高计算成本；在极端动态或稀有场景下的鲁棒性需进一步验证。

---

## 116. Governing AI Forgetting: Auditing for Machine Unlearning Compliance

**arXiv ID:** 2602.14553 | [PDF](https://arxiv.org/pdf/2602.14553v1)

**作者:** Qinqi Lin `[一作]` (Chinese University of Hong Kong), Jianwei Huang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并分析了针对机器学习遗忘（MU）合规性的经济审计框架，利用认证遗忘理论结合监管执法，阐明AI运营商与审计者在审计强度与遗忘级别之间的博弈，并给出唯一均衡结果。

**💡 创新点**

创新点包括：①首次将认证遗忘理论与监管执法融合，提出MU合规审计的经济模型；②针对MU特有的非线性效用与检测概率，设计辅助变换将双变量固定点问题转化为可解的一元问题；③揭示“遗忘审计悖论”——在删除请求激增且罚金高时，审计者可减少检查强度；④证明披露审计策略可提升监管成本效益并实现双赢。

**🔧 技术方法**

技术方法包括：博弈论（纳什均衡、Stackelberg均衡）建模；认证遗忘理论（(ε,δ)-可证明遗忘）转化为假设检验框架；辅助变换与单调性分析解决非线性最优问题；数值仿真与MNIST实验验证理论。

**📊 数据集**

使用MNIST手写数字分类数据集，训练多项式逻辑回归模型并在其上进行Hessian-free MU实验，以获取经验测试误差和效用函数，用以校准模型特征参数G。

**📈 对比分析**

与传统风险审计（TRA）基准相比，提出的战略未披露审计（SUA）和战略披露审计（SDA）在审计者收益提升高达2549.30%，在运营商收益提升高达74.60%；SDA在两者收益上均优于SUA，验证了审计透明度的双赢效果。

**⚠️ 局限性**

局限性包括：①模型参数G需通过实验校准，缺乏通用化；②假设独立检查、线性成本及零容忍策略可能与实际不符；③未考虑多方异质性与检测误差相关性；④仅在单一数据集上验证，尚缺乏大规模真实AI系统的实证支持。

---

## 117. A Safety-Constrained Reinforcement Learning Framework for Reliable Wireless Autonomy

**arXiv ID:** 2602.13207 | [PDF](https://arxiv.org/pdf/2602.13207v1)

**作者:** Abdikarim Mohamed Ibrahim `[一作]` (Sunway University), Rosdiadee Nordin `[通讯]` (Sunway University)

**通讯引用:** 6321 | [OpenAlex ID](https://openalex.org/A5060844784)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

在无线上行调度任务中，设计并实现了一种主动安全约束的强化学习框架，将 Proof‑Carrying Control (PCC) 与 Empowerment‑Budgeted Enforcement (EB) 结合，以在执行前验证并修正调度动作，保证不出现干扰超阈值的传输。

**💡 创新点**

创新点在于：①提出了基于证书的 PCC 机制，在动作执行前进行轻量级数学验证；②引入可调节的赋能预算，平衡安全覆盖与自主决策的权衡；③将这两种机制嵌入 PPO 的训练与决策循环，实现了安全性与性能的可调节权衡。

**🔧 技术方法**

采用的技术包括：Proximal Policy Optimization (PPO) 强化学习、Proof‑Carrying Control (PCC)、Empowerment‑Budgeted Enforcement (EB)、冲突图建模、最大独立集 (MIS) 计算，以及基于贝尔努利流量的仿真环境。

**📊 数据集**

使用的数据集为仿真产生的自定义数据：30 设备、4 条正交信道、每个设备的 Bernoulli 到达过程 λ∈{0.2,0.4,0.7,1.0}，并构造冲突图密度约 0.34 的无线传播模型。

**📈 对比分析**

与无约束 RL 与后置防护（reactive guard）两种基线进行对比。主动 PCC+EB 在吞吐量上略低于无约束策略（≈1000 包/episode），但完全消除了不安全传输；吞吐量与 λ 的相关性弱，安全覆盖率接近 100%。reactive guard 既有吞吐量损失，又需要频繁后置修正；无约束策略吞吐量最高但完全缺乏安全保障。总体表现表明主动安全约束可在可接受的吞吐量损失下提供形式化安全保证。

**⚠️ 局限性**

局限性包括：①预算参数需人工调节，缺乏系统化自动调优方法；②在非常保守的预算设置下吞吐量严重受限；③仅在单智能体、单任务的上行调度仿真中验证，缺乏多智能体或大规模 IoT 场景的实验；④对实际物理层参数的鲁棒性和可迁移性尚未深入评估。

---

## 118. A Soft Wrist with Anisotropic and Selectable Stiffness for Robust Robot Learning in Contact-rich Manipulation

**arXiv ID:** 2602.14434 | [PDF](https://arxiv.org/pdf/2602.14434v1)

**作者:** Steven Oh `[一作]` (OMRON SINIC X Corp), Masashi Hamaya `[通讯]` (OMRON SINIC X Corp)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并验证了一种新的柔性手腕 CLAW，结合两根正交叶片弹簧和可锁定转轴，实现6自由度可调形变。

**💡 创新点**

创新点在于：采用凸形金属带叶片弹簧实现大范围、多方向可变刚度；通过单一锁定装置实现三种刚度模式；结构轻量、成本低且易于组装。

**🔧 技术方法**

技术手段包括：机械设计与理论分析、凸形金属带弹簧、锁定机构、基于 ACT Transformer 的模仿学习控制，以及前馈动力学顺从控制。

**📊 数据集**

使用的实验数据集为：FMB（Functional Manipulation Benchmark）中的三个插入孔几何形状，及收集的 25 条人类演示轨迹（共 50 轨）。

**📈 对比分析**

与基准软手（TPU 打印 Fin Ray）和硬式并联抓手比较，CLAW 在 peg‑in‑hole 基准上成功率达 76%，远高于 Fin Ray 43% 和硬抓手 36%；在多任务实验中成功率普遍高于对比抓手。

**⚠️ 局限性**

局限性包括：易发生抓取物滑落、在承重过大时出现下垂、对极重物体仍需提高抓力；未来工作计划引入强化学习和更大力矩马达。

---

## 119. voice2mode: Phonation Mode Classification in Singing using Self-Supervised Speech Models

**arXiv ID:** 2602.13928 | [PDF](https://arxiv.org/pdf/2602.13928v1)

**作者:** Aju Ani Justus `[一作]` (University of Southern California), Shrikanth Narayanan `[通讯]` (University of Southern California)

**通讯引用:** 30725 | [OpenAlex ID](https://openalex.org/A5010028928)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了 voice2mode 框架，利用预训练的 HuBERT 和 wav2vec2.0 自监督语音模型提取嵌入，对四种歌唱音调模式（呼吸性、普通、流动、压迫性）进行分类。

**💡 创新点**

首次将大规模语音自监督模型迁移到歌唱音调模式分类，并通过层级分析揭示低层 Transformer 嵌入最能捕捉声带控制特征，证明语音模型对歌唱任务的迁移能力。

**🔧 技术方法**

采用 HuBERT、wav2vec2.0-Base/Large 的层级特征提取、时间均值池化，随后用 SVM 与 XGBoost 两种轻量级分类器进行训练，使用 5 折交叉验证与准确率/混淆矩阵评估。

**📊 数据集**

使用公开的俄罗斯女高音持续元音数据集（763 条录音，9 个元音，4 种音调标签）进行实验。

**📈 对比分析**

与传统光谱、mel-spectrogram、MFCC 基准在相同实验设置下对比，HuBERT 特征在 SVM 上达到约 95.7% 的准确率，较最佳基准提升 12–15%，wav2vec2 模型提升 3–10%。

**⚠️ 局限性**

局限性在于仅使用单一女高音、持续元音、有限标签的数据集，缺乏多声部、多流派及连续歌唱段落的评估，且未对模型进行歌唱数据微调，可能限制其在更广泛场景下的泛化。

---

## 120. VariViT: A Vision Transformer for Variable Image Sizes

**arXiv ID:** 2602.14615 | [PDF](https://arxiv.org/pdf/2602.14615v1)

**作者:** Aswathi Varma `[一作]` (Technical University of Munich), Benedikt Wiestler `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了 VariViT，一种可处理变尺寸 3D 医学影像的 Vision Transformer，利用固定补丁尺寸、中心选择式位置嵌入和批处理策略，专注于病灶区域以提升特征学习效果。

**💡 创新点**

核心创新包括：①中心与选择式位置嵌入（Center & Select），在不使用插值的情况下为不同尺寸图像动态重塑位置嵌入；②两种批处理策略（自定义批采样器与梯度累积），显著降低训练与推理成本；③将上述改动与标准 3D ViT 结构结合，形成高效且具可变尺寸适应性的模型。

**🔧 技术方法**

采用 3D ViT‑S/16 架构、正弦位置编码、中心选择式位置嵌入、梯度累积/自定义批采样、AdamW 优化器、交叉熵加类别权重、随机几何与噪声增强等技术，并使用 t‑SNE 可视化验证特征分离。

**📊 数据集**

实验数据集包括两套脑部 3D MRI：① glioma 数据集（1856 张）用于 IDH 突变状态与三种胶质瘤亚型分类；② brain tumor 数据集（1699 张）用于原发与转移脑肿瘤二分类。每张扫描均包含 FLAIR、T2w、T1w 与 T1w+对比四通道。

**📈 对比分析**

与基线 ResNet‑18、标准 3D ViT 以及 Pix2Struct 进行 k‑fold 交叉验证比较，评估指标为 AUC、F1‑score 与 MCC。VariViT‑GA 在 IDH 预测任务中 AUC 0.942、F1 0.755、MCC 0.709；在脑肿瘤类型分类中 AUC 0.954、F1 0.763、MCC 0.706，均优于基线且训练时间缩短约 30%。位置嵌入 ablation 证明中心选择法优于插值或独立固定方案。

**⚠️ 局限性**

主要局限包括：需要先验的病灶分割/边界框；仅在脑部 MRI 上验证，泛化到其他解剖部位或模态尚未测试；批处理策略在极大规模或高分辨率数据集中的适用性尚需进一步评估；对多分类任务的性能提升有限。

---

## 121. Query as Anchor: Scenario-Adaptive User Representation via Large Language Model

**arXiv ID:** 2602.14492 | [PDF](https://arxiv.org/pdf/2602.14492v1)

**作者:** Jiahao Yuan `[一作]` (Ant Group), Zhongle Xie `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建工业级用户表示框架Q-Anchor，使用查询作为锚点实现多场景适应的动态用户嵌入

**💡 创新点**

提出查询锚点机制、层次粗细编码与联合对比+自回归训练，以及轻量软提示微调实现单模型多场景

**🔧 技术方法**

大语言模型（Qwen2.5-0.5B），双塔结构，信息对比损失、下一词预测、软提示、KV-cache加速

**📊 数据集**

UserU预训练集（约1.024×10^8条记录）与10个Alipay工业基准任务

**📈 对比分析**

相较于文本嵌入和现有用户嵌入基线，Q-Anchor平均AUC0.8225、KS0.5267，在10个场景均优于最强基线，线上A/B测试提升信用发放和风险分级

**⚠️ 局限性**

对模型规模扩展不显著，仍依赖大规模预训练数据；在多源异构日志噪声与偏差上需进一步鲁棒性研究

---

## 122. DistillLens: Symmetric Knowledge Distillation Through Logit Lens

**arXiv ID:** 2602.13567 | [PDF](https://arxiv.org/pdf/2602.13567v1)

**作者:** Manish Dhakal `[一作]` (Georgia State University), Yi Ding `[通讯]` (Auburn University)

**通讯引用:** 4876 | [OpenAlex ID](https://openalex.org/A5100329067)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

DistillLens 通过将教师模型中间层隐藏状态投影到词汇空间（Logit Lens），并用对称的 Jensen‑Shannon 散度（JSD）对齐学生模型的中间思考过程，从而实现更高质量的知识蒸馏。

**💡 创新点**

创新点在于：①使用 Logit Lens 把中间隐藏状态映射到词汇空间；②采用对称散度实现双向惩罚，避免过度自信或欠信；③将中间层对齐作为可插拔模块，可与任意蒸馏框架组合。

**🔧 技术方法**

使用的技术包括：Transformer LLM、Logit Lens 投影、Jensen‑Shannon 散度（JSD）、对称蒸馏目标、层映射均匀采样、混合精度 BF16 训练。

**📊 数据集**

使用的数据集主要有 DollyEval、SelfInst、VicunaEval、S‑NI、UnNI 等指令跟随与推理基准，训练时采用 Dolly 15k 数据。

**📈 对比分析**

通过与标准 KD、SeqKD、RKL、JSD、AKL 等基线进行对比，DistillLens 在 GPT‑2‑120M、GPT‑2‑340M 与 Llama‑7B→TinyLlama‑1.1B 的 Rouge‑L 与 GPT‑4o 评分均超越所有基线，甚至在 GPT‑2‑340M 上略优于教师模型。

**⚠️ 局限性**

局限性在于训练阶段需要投影多层隐藏状态到词汇空间，导致 O(K·V·d) 的额外计算与显存开销，且跨架构蒸馏受词表不匹配限制，无法直接扩展至极大模型。

---

## 123. An Empirical Study of the Evolution of GitHub Actions Workflows

**arXiv ID:** 2602.14572 | [PDF](https://arxiv.org/pdf/2602.14572v1)

**作者:** Pooya Rostami Mazrae `[一作]` (University of Mons), Mairieli Wessel `[通讯]` (Radboud University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对GitHub Actions工作流的演进进行混合方法研究，先通过对439个工作流变更的质性分析识别七类概念性变更，再对267,955个工作流历史（3,418,911个文件版本）进行大规模定量分析；

**💡 创新点**

首次系统量化工作流文件的变更频率、类型与时间分布，并提出七类高层概念变更分类；同时评估大型语言模型工具出现后对工作流变更的影响，结果显示无显著效应；

**🔧 技术方法**

使用GitHub Actions Workflow Differ进行语法级差异检测，结合Python脚本和统计工具对变更集进行分类；

**📊 数据集**

基于2025‑10‑09公开的GitHub公开仓库数据集（49,258个活跃仓库，3,418,911个工作流快照）做分析；

**📈 对比分析**

通过先后期（LLM工具出现前后）的对比，采用Mann‑Whitney U检验和Cliff’s δ效应量评估差异；整体发现工作流每周约7.3%被修改，绝大多数为小幅修改，且变化随时间趋向更细粒度；

**⚠️ 局限性**

研究仅覆盖活跃且星级≥300的仓库，排除了无效或临时工作流，且只分析主分支变更；工具默认配置可能遗漏某些变更；因无法获取执行日志和并行分支合并方式，结果可能受限。

---

## 124. BETA-Labeling for Multilingual Dataset Construction in Low-Resource IR

**arXiv ID:** 2602.14488 | [PDF](https://arxiv.org/pdf/2602.14488v1)

**作者:** Md. Najib Hasan `[一作]` (Wichita State University), Nazmul Siddique `[通讯]` (Ulster University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过BETA-Labeling框架使用多模型LLM对孟加拉语IR任务进行自动标注并人工验证，构建了两个低资源语言数据集，并评估了一跳机器翻译对低资源语言间数据可重用性的影响。

**💡 创新点**

创新点在于将多模型LLM产生的标签作为候选并通过上下文对齐、投票与人工核对的BETA-Labeling流程，系统地揭示了跨语言迁移中隐含语义漂移和语言对偏差（LPD‑bias）。

**🔧 技术方法**

采用GPT‑4.1、Gemini‑3、LLaMA‑3.3进行标注，GPT‑4o mini、Gemini‑2.5 Flash Lite、DeepSeek R1 Distilled 7B进行翻译，并用Cosine、BLEU、METEOR、BERTScore、Jaccard等多种指标评估显式与隐式语义保留。

**📊 数据集**

数据集为自建的Bangla_Lite（3,822样本）和Bangla_Culture（3,086样本），涵盖文学与文化领域的隐含意义文本。

**📈 对比分析**

与单一模型标注或直接翻译的做法相比，BETA‑Labeling在标签一致率和人工验证通过率上提升约10–15%，但跨语言翻译的隐含语义得分大幅下滑，表明一跳翻译难以保持任务相关语义，且存在显著的语言方向偏差。

**⚠️ 局限性**

限制在于仅针对孟加拉语及有限目标语进行实验，缺乏全面的人类评估、跨模型多语言泛化验证，且只考察单跳翻译，未涉及多跳或基于枢轴的翻译策略，结果受所选LLM和提示的时效性影响。

---

## 125. Enhanced Accessibility for Mobile Indoor Navigation

**arXiv ID:** 2602.13233 | [PDF](https://arxiv.org/pdf/2602.13233v1)

**作者:** Johannes Wortmann `[一作]` (Fraunhofer Institute for Open Communication Systems), Thomas Jung `[通讯]` (Hochschule für Technik und Wirtschaft)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

基于everGuide SDK开发面向视障人士的室内导航App，增强可访问性

**💡 创新点**

引入振动反馈与声学罗盘，提供事件驱动与连续导航提示，支持可自定义的交互模式

**🔧 技术方法**

使用everGuide定位SDK、Android、Figma、Material Design 3、振动与声学反馈技术

**📊 数据集**

基于7名视障或残障用户的实验评测数据

**📈 对比分析**

通过用户体验评估比较语音、声学罗盘、振动罗盘和事件式反馈，结果显示振动罗盘受欢迎、整体满意度高，但差异受个体偏好影响

**⚠️ 局限性**

样本量小、缺乏长期使用评估、不同视觉障碍类型代表性不足

---

## 126. Efficient Multi-round LLM Inference over Disaggregated Serving

**arXiv ID:** 2602.14516 | [PDF](https://arxiv.org/pdf/2602.14516v1)

**作者:** Wenhao He `[一作]` (Southeast University), Fangcheng Fu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了名为 “Efficient Multi-round LLM Inference over Disaggregated Serving” 的系统，解决多轮 LLM 推理中预填充-解码交错的工作负载问题。

**💡 创新点**

创新点在于：①设计了基于实时系统负载的自适应路由机制，动态决定预填任务在本地或远程执行；②提出了基于 TTFT 考量的预填顺序重排策略；③结合离线部署规划，将预填和解码的模型并行度与数据并行度联合优化。

**🔧 技术方法**

技术包括：离线性能建模（α-β 预填/解码/KV 传输时间模型）；整数线性规划（ILP）求解部署配置；分布式共享队列与窗口统计；RDMA 网络传输；基于 vLLM/Dynamo 的实现与优化。

**📊 数据集**

使用的多轮工作负载数据集有：ToolBench、GAIA、HotpotQA 与 DuReader，涵盖不同长度与多轮交互的真实请求。

**📈 对比分析**

与 Dynamo（分离式）、vLLM（共置）以及 vLLM-Continuum（多轮共置）进行对比，实验表明在所有模型与负载下，系统平均提升 SLO 达成率 67.29%–339.74%，在最优配置下可提升 967.54% 及 3435.1%。

**⚠️ 局限性**

局限性包括：①仍需手动调参（α、β、窗口大小）以平衡 TTFT 与 ITL；②以 P95 延迟为代价函数，未直接优化 SLO 满足率；③依赖离线模型预测，若真实运行时变化大可能导致规划失效。

---

## 127. DPBench: Large Language Models Struggle with Simultaneous Coordination

**arXiv ID:** 2602.13255 | [PDF](https://arxiv.org/pdf/2602.13255v1)

**作者:** Najmul Hasan `[一作]` (University of North Carolina), Prashanth BusiReddyGari `[通讯]` (University of North Carolina)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了DPBench基准，用Dining Philosophers问题评估LLM在资源竞争下的并发协调能力。

**💡 创新点**

首次为LLM构建并发协调基准，并揭示“收敛推理”导致高死锁率、通信并不一定改善协调，凸显LLM多智能体在同步决策中的局限。

**🔧 技术方法**

采用GPT‑5.2、Claude Opus 4.5、Grok 4.1进行零样本推理，定义六项性能指标（死锁率、吞吐量、公平度等），在八种决策/通信/人数组合下进行实验。

**📊 数据集**

使用Dining Philosophers仿真环境，收集每个实验条件下20个episode、最多30步的运行日志作为数据集。

**📈 对比分析**

通过对比同一模型在顺序与并发模式、不同人数、是否通信下的指标，发现顺序模式几乎零死锁，而并发模式死锁率高达25–95%，通信有时反而提升死锁率。

**⚠️ 局限性**

实验仅覆盖三种模型、两种人数且未测试更大规模或真实世界复杂场景，导致基准对实际系统迁移的可预测性有限。

---

## 128. VeRA: Verified Reasoning Data Augmentation at Scale

**arXiv ID:** 2602.13217 | [PDF](https://arxiv.org/pdf/2602.13217v1)

**作者:** Zerui Cheng `[一作]` (ByteDance Seed), Wenhao Huang `[通讯]` (ByteDance Seed)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将传统静态评测数据集中的每个种子问题编译为可执行规范，能够在训练后生成大量经过程序验证的等价或更难的实例，形成可持续更新的评测体系。

**💡 创新点**

创新点包括：
1) 通过可执行规范把单一问题映射为任务族，实现无限生成已验证实例；
2) 提供两种扩展模式——等价（VeRA‑E）和硬化（VeRA‑H / Pro），既能检测模型对表面变化的脆弱性，又能在保持标签完整性的前提下递增难度；
3) 在规范生成和验证过程中引入教师/裁判 LLM，配合沙盒执行与 deterministic sampling，实现一次性高成本验证后低成本无限扩展；
4) 在评测报告中同时给出种子分数与等价/硬化分数，显式区分熟悉度与真正推理能力。

**🔧 技术方法**

技术手段：
- LLM 编译器（Teacher）生成模板、生成器和验证器代码；
- LLM 裁判（Judge）在硬化模式下做噪声判定；
- 受限沙盒执行环境，防止代码泄漏；
- 哈希种子驱动的确定性采样；
- 验证器实现对逻辑、符号、几何等领域的判定；
- 通过“等价”与“硬化”两套验证流程，保证标签的可靠性。

**📊 数据集**

使用的数据集包括：GSM8K、AIME 2024、AIME 2025、Beyond‑AIME、GPQA‑Diamond、AMO‑Bench 等多种数学与科学基准；每个种子问题被转换成规范后在同一领域生成数百至数千个变体。

**📈 对比分析**

比较方法：对每个基准按 Avg@5（每题 5 次随机解答平均）评估 16 大型模型；分别计算种子与等价、硬化版本的得分差异与方差；使用 Spearman 相关系数评估种子与变体排名稳定性。结果显示：
- 在 GSM8K 上种子得分几乎不变，但等价变体显著增大方差并提升模型区分度；
- AIME 2024 与 2025 在等价扩展后得分下降 14–7 点，说明老数据更易被模型记忆；
- 硬化后 AIME 2024 的平均得分从 84.9% 降至 67.8%，恢复了 17 点的头room；
- 通过 Pro 方案，得分进一步下降到 58.6%，但排名相关性仍保持在 0.73。整体证明了可验证扩展能有效揭示模型真正的推理能力。

**⚠️ 局限性**

局限性：
1) 仅适用于可被程序化验证的任务，无法覆盖主观判断、开放式写作等领域；
2) 验证器编写一次性错误会被放大，需严格校验；
3) 硬化的难度分布依赖于生成器与验证器的设计，可能出现非单调性或对某些技巧过度惩罚；
4) 语言表面生成仍可能出现歧义或难以理解的实例，虽然裁判会过滤，但无法完全保证文本质量；
5) 对极大规模或高成本验证任务（如符号求解）仍需外部求解器，影响速度与成本。

---

## 129. Fair Allocation with Initial Utilities

**arXiv ID:** 2602.14850 | [PDF](https://arxiv.org/pdf/2602.14850v1)

**作者:** Niclas Boehmer `[一作]` (Hasso Plattner Institute), Luca Kreisel `[通讯]` (Hasso Plattner Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究在存在初始效用差异的情况下，对不可分资源进行公平分配的问题，提出并分析了新的公平性概念（EF-init、EF1-init、min-EF1-init），探讨其存在性、计算复杂性，并给出了对应的算法。

**💡 创新点**

创新点包括：①将公平性定义从“机会平等”扩展到“结果平等”，在分配前引入初始效用；②证明在此设定下经典的EF1分配可能不存在且判定问题为NP‑hard；③提出总是可满足的min‑EF1‑init概念，并证明其在加法效用下可在多项式时间内实现；④针对相同资源的特殊情况给出完整的动态规划算法；⑤通过改进的轮询算法实现min‑EF1‑init的高效求解。

**🔧 技术方法**

主要技术包括：
- 形式化公平性概念（EF-init、EF1-init、min‑EF1‑init）；
- 归约与复杂性证明（从公平着色问题、二分图着色等经典NP‑hard问题）；
- 动态规划与状态压缩用于常数代理的判定；
- 对称性与层次结构分析用于相同资源情形；
- 轮询（Round‑Robin）算法的改进与激活间隙分析实现min‑EF1‑init分配。

**📊 数据集**

本文为理论研究，无实验数据集；所有结果均基于严格的数学证明与复杂性分析。

**📈 对比分析**

比较方法主要通过理论复杂性与算法运行时间呈现：
- EF1-init判定为NP‑hard；
- 对常数代理可在多项式时间内决定；
- 相同资源下的EF-init判定可在O(n²·m³)时间内完成；
- min‑EF1‑init可通过多项式时间的轮询算法得到完整分配。整体性能上，min‑EF1‑init在可行性与效率上优于传统EF1-init，但仍需要额外的前置信息。

**⚠️ 局限性**

局限性包括：
- 需要已知并可比拟所有代理的初始效用；
- 需要跨代理的效用可比（基于统一量化指标）；
- 仅适用于加法效用模型，子模函数或非加法情况可能导致min‑EF1‑init无解；
- 方案侧重于结果平等，未考虑机会平等的兼顾；
- 可能对高初始效用代理产生不公平的“补偿”偏好。

---

## 130. Near-Linear Time Computation of Welzl Orders on Graphs with Linear Neighborhood Complexity

**arXiv ID:** 2602.14625 | [PDF](https://arxiv.org/pdf/2602.14625v1)

**作者:** Jan Dreier `[一作]` (Vienna University of Technology), Clemens Kuske `[通讯]` (Vienna University of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种随机算法，能够在几乎线性时间内为具有线性 VC‑维度（即线性散射函数）的集合系统构造 Welzl 顺序，从而实现低交叉数。

**💡 创新点**

创新点在于：①通过引入“近偶类”概念，将对偶集合系统的复杂性降低；②利用随机抽样和分层逼近实现几乎线性时间；③将该算法应用于图类的紧邻覆层和一阶逻辑模型检验，显著提升了先前 O(n³) 级别的算法至 O(n³+ε) 级别。

**🔧 技术方法**

主要技术手段包括：集合系统的双向图表示、双重分解（Twin Partition 与 Near Twin Partition）、随机抽样（Reservoir Sampling）、分层逼近与迭代收缩、以及基于 VC‑理论的散射函数上界来控制近似误差。

**📊 数据集**

该工作为理论算法，不涉及真实数据集；所有实验均在合成的集合系统与图类（如平面图、有限流图、邻接复杂度线性或近线性图类）上验证。

**📈 对比分析**

与之前的 O(|U|³|ℱ|)、O(|U|²|ℱ|) 以及 O(|ℱ||U|²/d + |U|²+2/d) 等算法相比，本算法在线性散射函数下实现 O(S log S) 的近线性运行时间；在图类应用中，将紧邻覆层的构造时间从 O(n⁹.⁸) 降至 O(n log n)，并将一阶模型检验的总运行时间从 O(n⁵+ε) 降至 O(n³+ε)。

**⚠️ 局限性**

局限性包括：①算法仅在集合系统的原始与对偶散射函数均线性时达到最优性能；②对更高 VC‑维度的系统，尽管可扩展但交叉数与时间上均出现更差的指数；③算法是随机化的，虽可通过放大次数降低失败概率，但在严格确定性要求下仍需改进。

---

## 131. Differentiable Rule Induction from Raw Sequence Inputs

**arXiv ID:** 2602.13583 | [PDF](https://arxiv.org/pdf/2602.13583v1)

**作者:** Kun Gao `[一作]` (Institute of High Performance Computing), Feng Yang `[通讯]` (Institute of High Performance Computing)

**通讯引用:** 9717 | [OpenAlex ID](https://openalex.org/A5072857436)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了NeurRL框架，利用可微分k-means聚类与深度规则学习模块，直接从原始序列或图像数据学习可解释的逻辑规则，解决了传统神经符号ILP中标签泄漏的问题。

**💡 创新点**

核心创新在于将整个ILP过程完全微分化：①用可微分k-means对子序列进行聚类；②用自编码器学习子序列嵌入；③用可微分阈值/模糊逻辑的多层网络学习规则并从权重矩阵中提取符号规则；④实现了从原始输入到符号规则的端到端训练，无需中间监督标签。

**🔧 技术方法**

技术包括：可微分k-means聚类、自动编码器、基于softmax+ReLU的深度规则学习网络、可微分阈值函数、模糊逻辑运算、全微分损失（重构误差+聚类损失+规则分类损失）。

**📊 数据集**

实验使用：①合成三角脉冲和三角函数时间序列；②13个UCR二分类时间序列数据集；③5个MNIST二分类子集（将二维图像展开成一维序列）。

**📈 对比分析**

与SSSL、Xu、BoW等基线以及非微分k-means版本进行比较。NeurRL(N)在7个数据集上取得最佳准确率，NeurRL(R)在5个数据集上获得第二佳；UCR平均准确率为0.891；在MNIST上，使用可微分k-means的NeurRL显著缩短训练时间（例如Coffee从313s降至42s），准确率基本保持不变。

**⚠️ 局限性**

局限性包括：需要手动设定簇数、子序列长度、区域数等超参数；对高维图像需先展平为序列，可能丢失空间结构；未解决缺失/不完整数据场景；规则提取依赖阈值设置，解释性受限于此。

---

## 132. DWBench: Holistic Evaluation of Watermark for Dataset Copyright Auditing

**arXiv ID:** 2602.13541 | [PDF](https://arxiv.org/pdf/2602.13541v1)

**作者:** Xiao Ren `[一作]` (Zhejiang University), Zhikun Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 2002 | [OpenAlex ID](https://openalex.org/A5100746182)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出两层分类法并构建统一基准与工具包，用于评估和比较数据集水印在分类和生成任务中的有效性与鲁棒性。

**💡 创新点**

创新点包括：①两层分类法统一整理现有方法；②开源基准工具包；③新增“样本显著性”和“验证成功率”两项可复现指标；④在多水印、多用户真实场景下进行系统评估。

**🔧 技术方法**

使用模型无关与模型相关的水印注入与验证技术（后门、扰动、频域水印、语义注入等），并在统一评估框架下加入扰动攻击、生成/再生攻击等鲁棒性测试。

**📊 数据集**

分类任务采用CIFAR‑10/100、TinyImageNet，模型有ResNet18/VGG/ViT；生成任务采用Pokémon、CelebA、WikiArt，模型有Stable Diffusion V1.4/XL、Kandinsky 2.2。

**📈 对比分析**

在统一设置下评估25种方法，报告样本显著性、VSR、模型效能、PSNR等；结果显示无单一方法在所有场景最佳，模型无关方法在低水印率下效果好但鲁棒性差，模型相关方法更隐蔽但易被干扰；多水印/多用户场景表现不稳定。

**⚠️ 局限性**

限制包括：低水印率下难以可靠验证；多水印、多用户场景易导致误报或性能下降；大多数方法在高鲁棒攻击下失效；多比特水印在生成任务中表现不佳；整体方法仍缺乏统一稳定、低成本的实用方案。

---

## 133. A generalizable foundation model for intraoperative understanding across surgical procedures

**arXiv ID:** 2602.13633 | [PDF](https://arxiv.org/pdf/2602.13633v1)

**作者:** Kanggil Park `[一作]` (Samsung Medical Center), Kyu-Hwan Jung `[通讯]` (Samsung Medical Center)

**通讯引用:** 1784 | [OpenAlex ID](https://openalex.org/A5087849800)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研制了 ZEN 基础模型，利用 4.3M 帧、21 种微创手术视频进行自监督预训练，并在 20 个下游任务上进行评估。

**💡 创新点**

创新点在于将 SimDINOv2 视觉专家与 PeskaVLP 视觉‑语言专家通过多教师蒸馏融合，生成既具空间表达又具跨模态对齐的通用表征，显著提升跨程序的泛化性能。

**🔧 技术方法**

采用 SimDINOv2、PeskaVLP 的特征蒸馏、ViT‑Base 主干，以及冻结、微调、少样本、零样本等多种评估策略，构建多任务基准。

**📊 数据集**

使用 4.3M 帧的自制预训练集（21 程序、10 机体）以及 20 个公开/内部下游数据集（如 Cholec80、MultiByPass140、AutoLaparo、CholecT50、LLS48、VQA、语义/实例分割、深度估计等）。

**📈 对比分析**

通过 20 个任务的平均分与排名对比，ZEN 在冻结（0.579）和微调（0.603）两种设置下均排名第一，且在少样本与零样本场景下表现同样领先，明显优于 SurgeNet、EndoFM、PeskaVLP 等基线。

**⚠️ 局限性**

局限性包括预训练集偏向腹腔镜手术，缺乏机器人或非腹腔镜视频；下游任务仍有限，未覆盖所有临床场景；缺乏直接的临床效果评估，需进一步扩展数据与验证。

---

## 134. Differentially Private Retrieval-Augmented Generation

**arXiv ID:** 2602.14374 | [PDF](https://arxiv.org/pdf/2602.14374v1)

**作者:** Tingting Tang `[一作]` (University of Southern California), Murali Annavaram `[通讯]` (University of Southern California)

**通讯引用:** 6470 | [OpenAlex ID](https://openalex.org/A5018033573)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于提案-测试-发布（PTR）与子样本聚合的差分隐私检索增强生成（DP‑RAG）框架，能够在RAG系统中仅使用隐私化关键词完成答案生成，从而在不泄露检索数据库的前提下保留大模型的回答质量。

**💡 创新点**

创新点在于将问答答案压缩到关键词子空间，利用低维度词频直方图与PTR机制实现高效、可控的隐私化关键词抽取；随后仅将这些关键词注入生成提示，实现既满足差分隐私又维持实用性的生成。

**🔧 技术方法**

采用的技术包括差分隐私理论（ε,δ‑DP）、提案-测试-发布（PTR）框架、子样本聚合、检索增强生成（RAG）、指令调优的大语言模型（Qwen 2.5、Llama 3.x）、关键词抽取与频率计数。

**📊 数据集**

实验数据集为开放域问答基准Natural Questions（NQ）与Trivia Question‑Answering（TQA），检索数据库采用Wikipedia子集。

**📈 对比分析**

与无检索（non‑RAG）、无隐私关键词抽取（KSA）以及完整检索（RAG）三种基线对比；在中等隐私预算（ε≈3‑5）下DP‑RAG已超越无检索基线，且在大模型上可接近甚至优于无隐私KSA；随着ε增大，性能逐步提升。

**⚠️ 局限性**

局限性包括对大型、指令调优模型的高度依赖；关键词压缩导致信息损失，低ε下PTR通过率低导致无关键词输出；集成规模需权衡，过大可能出现收益递减；实现复杂，需精确管理隐私预算与计算成本。

---

## 135. GraFSTNet: Graph-based Frequency SpatioTemporal Network for Cellular Traffic Prediction

**arXiv ID:** 2602.13282 | [PDF](https://arxiv.org/pdf/2602.13282v1)

**作者:** Ziyi Li `[一作]` (Xinjiang University), Ming Yan `[通讯]` (Xinjiang University)

**通讯引用:** 27686 | [OpenAlex ID](https://openalex.org/A5006900597)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种结合空间-时序建模与时频特征的细胞流量预测框架GraFSTNet，能够在不依赖预定义拓扑的情况下捕获细胞间隐式空间依赖，并通过频域分析提取周期性模式；

**💡 创新点**

创新点包括（1）基于GraphTrans的自注意力空间建模，消除对邻接矩阵的依赖；（2）TFTransformer时频分支，利用频域注意力融合多频带特征；（3）自适应尺度LogCosh损失，按流量幅度动态缩放误差，提升低流量预测精度；（4）节点级注意融合，实现跨模态特征自适应加权；

**🔧 技术方法**

使用的技术包括GraphTrans（自注意力图模型）、TFTransformer（时频Transformer）、自适应尺度LogCosh损失、节点级注意力融合以及标准的Transformer、GCN、LSTM等对照模型；

**📊 数据集**

实验使用三公开细胞流量数据集：MobileNJ、Trento和Milano，分别覆盖南京、意大利特伦托和米兰的多基站流量；

**📈 对比分析**

与11个基线（LSTM、GRU、Transformer、GCN、CNN-LSTM、DDGCRN、TimeMixer、DeseNet、ST-Tran、OpenCity、FISTGCN）以及GraFSTNet不同模块的 ablation 进行对比，GraFSTNet在MAE和RMSE上均显著优于所有对照，尤其在高峰期和低流量区块上提升明显；

**⚠️ 局限性**

限制包括模型规模较大，对大规模网络的计算与内存需求高；时频分支采用离散小波变换，需手工挑选频带；自适应损失需调参β，可能对不同网络环境表现不一致。

---

## 136. Exploring a Multimodal Chatbot as a Facilitator in Therapeutic Art Activity

**arXiv ID:** 2602.14183 | [PDF](https://arxiv.org/pdf/2602.14183v1)

**作者:** Le Lin `[一作]` (City University of Hong Kong), Yuhan Luo `[通讯]` (City University of Hong Kong)

**通讯引用:** 487 | [OpenAlex ID](https://openalex.org/A5048911139)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了基于多模态大语言模型的实时绘画分析与对话聊天机器人，支持艺术疗法中的视觉创作与情感反思。

**💡 创新点**

首次将实时图像理解与交互式对话相结合，提供“画中解读＋反思引导”功能，并通过专家评估提出系统改进路线。

**🔧 技术方法**

使用 Qwen3‑VL‑Plus 进行图像理解与对话生成，配合前端 Canvas 与后端实时监测模块实现实时绘画分析。

**📊 数据集**

未使用公开数据集，评估基于五位艺术治疗专家自行绘制的图像与对话日志。

**📈 对比分析**

采用定性专家评估（N=5），未进行数值化对比，评估结果表明用户对视觉理解与对话质量给予积极反馈，但缺乏客观性能指标。

**⚠️ 局限性**

局限性包括：缺乏实时创作过程感知、风险监测与干预机制不足、对话深度与广度不平衡、缺少触感与物理媒介体验、个性化与长期记忆功能待完善。

---

## 137. Revisiting the Platonic Representation Hypothesis: An Aristotelian View

**arXiv ID:** 2602.14486 | [PDF](https://arxiv.org/pdf/2602.14486v1)

**作者:** Fabian Gröger `[一作]` (École Polytechnique Fédérale de Lausanne), Maria Brbić `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于置换的零点校准框架，消除代表性相似性度量中因模型宽度和深度导致的伪相似性，重新评估“柏拉图代表性假设”并提出更精细的“亚里士多德代表性假设”。

**💡 创新点**

创新点在于统一、无偏的校准方法：①对单个度量使用置换生成经验零分布并以右尾阈值/效应量形式校准；②针对多层比较引入聚合感知校准，解决“看似优先”多比较偏差；③证明在多尺度、多模态设置下，原始的全局谱一致性实质上是宽度/深度的虚假效应，而局部邻域一致性仍显著。

**🔧 技术方法**

技术包括：置换检验、经验零分布、效应大小归一化、聚合感知校准；使用的相似性度量包括线性/核 CKA、CCA/SVCCA/PWCCA、mknn、RSA、Procrustes、周期kNN等。

**📊 数据集**

数据集：人工生成的高维高噪声数据；ImageNet-21K、MAE、DINOv2、CLIP、CLIP-finetuned 视觉模型；Bloomz、OpenLLaMA、LLaMA 语言模型；视频编码器 VideoMAE（base/large/huge）。

**📈 对比分析**

对比方法：原始（未校准）与校准后相似度；在全局谱度量中校准后无显著增长；在局部邻域度量中校准后仍保持显著增长；校准显著降低假阳性率（Type‑I 错误 ≤α），并在信号强度增加时保持高检出率。

**⚠️ 局限性**

局限性包括：校准需要大量置换计算，对大规模模型和数据集的算力要求高；目前仅解决宽度/深度偏差，其他潜在偏差（如输入分布、层间相关）未覆盖；校准结果依赖于选择的显著性阈值 α 与置换次数 K，需经验调参。

---

## 138. AST-PAC: AST-guided Membership Inference for Code

**arXiv ID:** 2602.13240 | [PDF](https://arxiv.org/pdf/2602.13240v1)

**作者:** Roham Koohestani `[一作]` (Delft University of Technology), Maliheh Izadi `[通讯]` (Delft University of Technology)

**通讯引用:** 4552 | [OpenAlex ID](https://openalex.org/A5024645888)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对代码大型语言模型进行灰盒会员推断攻击评估，比较了传统的Loss攻击与Polarized Augment Calibration (PAC)，并提出了语法感知的AST-PAC。

**💡 创新点**

创新点在于将抽象语法树(AST)引导的扰动机制融入PAC校准，提供了更符合代码语法的邻域生成方式，显著提升了对大型、复杂文件的攻击鲁棒性。

**🔧 技术方法**

使用的技术包括：Loss攻击、PAC、AST-PAC（基于Tree‑Sitter的AST节点交换），并通过计算token‑级log‑概率、polarized distance等信号实现灰盒攻击。

**📊 数据集**

实验数据集为Java子集的The Heap，划分为member、near‑member（相似度≥0.7）和non‑member三类，用于评估模型对不同文件特征的可追溯性。

**📈 对比分析**

通过ROC‑AUC和PR‑AUC进行比较，PAC在StarCoder2、Mellum等专用模型上普遍优于Loss攻击；AST‑PAC在文件/语法复杂度较高时进一步提升ROC‑AUC和PR‑AUC，但在小文件和高字母数字比例代码上表现略逊。

**⚠️ 局限性**

局限性包括：仅评估3‑7B参数模型，Java语法限制了方法在动态语言的适用性；AST‑PAC在小文件和高文本比例代码上效果不足；近似成员标签与实际训练样本存在噪声；模型预训练未知可能导致label污染。

---

## 139. REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment

**arXiv ID:** 2602.14065 | [PDF](https://arxiv.org/pdf/2602.14065v1)

**作者:** Kai Ye `[一作]` (Zhejiang University), Jiajun Bu `[通讯]` (Zhejiang University)

**通讯引用:** 13200 | [OpenAlex ID](https://openalex.org/A5052757755)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出REAL框架，利用Reasoning‑Pivot对KI‑VQA中的知识冲突进行检测与解决；

**💡 创新点**

创新点在于定义Reasoning‑Pivot冲突、构建REAL‑VQA数据集、引入RPA‑SFT进行pivot感知微调以及RPGD对抗解码实现无监督冲突缓解；

**🔧 技术方法**

采用多模态检索增强生成、专用pivot标记、RPA‑SFT多阶段监督学习、Patch Shuffle、Adaptive Gating、Gram‑Schmidt正交化等技术；

**📊 数据集**

使用自建REAL‑VQA、E‑VQA、InfoSeek、A‑OKVQA、MMKC、ScienceQA等多任务数据集；

**📈 对比分析**

与多种SOTA MLLM在E‑VQA、InfoSeek、A‑OKVQA等基准对比，REAL平均提升≈3.8%/1.6%/3.6%，冲突判别F1提升至≈90%；

**⚠️ 局限性**

局限在于对检索质量和pivot提取精度高度依赖，额外推理延迟，且在面对偏见/安全/版权信息时仍可能产生误判。

---

## 140. Common Knowledge Always, Forever

**arXiv ID:** 2602.13914 | [PDF](https://arxiv.org/pdf/2602.13914v1)

**作者:** Martín Diéguez `[一作]` (University of Angers), David Fernández-Duque `[通讯]` (University of Barcelona)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种多代理多拓扑PDL框架，研究其可判定性与有限模型性质。

**💡 创新点**

首次将多拓扑PDL与常识、共同知识等概念结合，并证明在Cantor导数空间下不具有限模型性质。

**🔧 技术方法**

采用拓扑语义、导数空间、PDL嵌入与可判定性证明技术，以及LTL（过去）嵌入来展示FMP缺失。

**📊 数据集**

无实验数据集，主要为理论证明与形式化构造。

**📈 对比分析**

未提供实验或性能对比，结论基于理论分析与模型构造。

**⚠️ 局限性**

局限性包括缺少有限模型性质、尚未证明完整性与可判定性、对更强闭包/分布知识操作的支持仍开放。

---

## 141. Measuring the relatedness between scientific publications using controlled vocabularies

**arXiv ID:** 2602.14755 | [PDF](https://arxiv.org/pdf/2602.14755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 142. Meningioma Analysis and Diagnosis using Limited Labeled Samples

**arXiv ID:** 2602.13335 | [PDF](https://arxiv.org/pdf/2602.13335v1)

**作者:** Jiamiao Lu `[一作]` (Shaanxi University of Science and Technology), Changming Sun `[通讯]` (CSIRO Data61)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种适用于有限标注样本的脑膜瘤诊断与分级的自适应多尺度空间‑频域特征融合网络AMSNet；

**💡 创新点**

其创新点在于将离散小波变换的多尺度频域信息与空间域特征通过方向与尺度门控进行自适应加权，并通过跨域注意力实现双向信息交换，从而显著提升少样本学习性能；

**🔧 技术方法**

核心技术包括ViT‑Base backbone、离散小波变换（DWT）、方向与尺度门控、跨域注意力融合（ACA‑SFF）以及基于重建的相似度度量；

**📊 数据集**

实验使用了自建的XJTU脑膜瘤MRI数据集以及Brain Tumor MRI和COVID‑19公开数据集进行评估；

**📈 对比分析**

与Proto‑Net、FRN、C2‑Net、Bi‑FRN等四类基线模型在4‑way 1‑shot/5‑shot任务上比较，AMSNet在XJTU数据集上准确率提升约10%+，在其他数据集上也保持最优表现；

**⚠️ 局限性**

局限性包括对不同MRI序列的适应性验证不足、模型结构复杂导致推理速度相对较慢，以及对DWT层级和自适应权重的进一步优化空间。

---

## 143. Algorithmic Simplification of Neural Networks with Mosaic-of-Motifs

**arXiv ID:** 2602.14896 | [PDF](https://arxiv.org/pdf/2602.14896v1)

**作者:** Pedram Bakhtiarifard `[一作]` (University of Copenhagen), Raghavendra Selvan `[通讯]` (University of Copenhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究深度网络压缩的算法复杂性，提出 Mosaic-of-Motifs (MoMos) 约束参数化，证明其在最坏情况下的 Kolmogorov 复杂度更低，并通过实验验证。

**💡 创新点**

从算法复杂性角度解释模型压缩可行性，首次给出压缩后参数的 Kolmogorov 复杂度下界，并设计块重复的 MoMos 参数化实现可量化压缩。

**🔧 技术方法**

利用 Kolmogorov 复杂度理论、Block Decomposition Method (BDM) 近似、梯度下降加 MoMos 投影、量化感知训练 (QAT)、随机子块重采样、Stirling 数分析等技术。

**📊 数据集**

主要使用 CIFAR-10 数据集，实验模型包括 ResNet20、Tiny-ViT、MLP，并评估 Pytorch Image Models (timm) 100M 参数范围内的模型。

**📈 对比分析**

与全精度 FP32 及不同位数 QAT (q=16,8,4) 对比；MoMos 在低容量下实现相近或更高准确率，RAC（相对算法压缩率）显著提升；BDM 复杂度比值显示训练后模型复杂度明显下降。

**⚠️ 局限性**

Kolmogorov/BDM 近似适用于二进制，可能误导；硬件加速支持不足，MoMos 在 FP32 模式下未显著提升硬件利用率；代码簿未学习，限制压缩效果。

---

## 144. Is Information Density Uniform when Utterances are Grounded on Perception and Discourse?

**arXiv ID:** 2602.14653 | [PDF](https://arxiv.org/pdf/2602.14653v1)

**作者:** Matteo Gay `[一作]` (KU Leuven), Edoardo Ponti `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过使用多语言视觉-语言模型，对图像-字幕配对与视觉故事数据进行预测，研究视觉感知与话语上下文对信息均匀度（UID）的影响。

**💡 创新点**

创新点在于首次将UID假说应用于多模态、跨语言环境，并揭示视觉和话语上下文共同作用可显著降低惊奇度波动。

**🔧 技术方法**

技术上采用了预训练的视觉-语言模型（PaliGemma和Gemma 3）结合词级惊奇度估计，并计算全局和局部UID。

**📊 数据集**

使用的数据集包括30种语言的Ground-XM3600图像-字幕对和13种语言的BloomVIST视觉故事，覆盖10-7个语言族。

**📈 对比分析**

对比方法是计算有无图像或话语上下文的UID差异，并用Wilcoxon、Page检验及线性混合模型检验显著性，结果显示视觉/话语上下文可显著降低UID，改善信息流均匀度。

**⚠️ 局限性**

限制包括语言覆盖受限（主要为印欧语系）、仅基于模型惊奇度而无实验验证，以及现有多语言长上下文视觉语言模型和数据集的稀缺。

---

## 145. Coordinated Information Dissemination on Telegram and Reddit During Political Turbulence: A Case Study of Venezuela in Global News Channels

**arXiv ID:** 2602.13333 | [PDF](https://arxiv.org/pdf/2602.13333v1)

**作者:** Despoina Antonakaki `[一作]` (Foundation for Research and Technology), Sotiris Ioannidis `[通讯]` (Technical University of Crete)

**通讯引用:** 4562 | [OpenAlex ID](https://openalex.org/A5022073151)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2017-2026年间9家主流国际新闻机构Telegram频道的委内瑞拉相关消息进行纵向分析，检验其在政治动荡期间是否存在内容同步与协调。

**💡 创新点**

提出一种保守的协同检测框架，结合字符n‑gram TF–IDF余弦相似度与时间窗口同步，首次在Telegram上建立主流新闻生态的协调基线，并通过叙事聚类与Attention–Coordination Ratio对关注度与内容同步进行分离。

**🔧 技术方法**

使用字符n‑gram TF–IDF向量化、余弦相似度计算、固定时间窗口（小时/日）聚合、负控制时间戳随机化、lead–lag 领导-跟随分析、叙事聚类（k‑means+SVD）、Attention–Coordination Ratio 计算。

**📊 数据集**

主要数据集为9家国际新闻机构（BBC World、CNN、Reuters、RT、Al Jazeera、France 24、DW、Euronews、Sky News、AP）Telegram频道共2038条委内瑞拉相关信息，另辅以Reddit多子板块投稿与评论数据。

**📈 对比分析**

通过对阈值τ（0.85至0.95）、时间窗口大小、以及随机时间戳的负控制进行对比，方法在所有设定下均未产生误报，表明其准确性高、误报率低；与随机化结果一致，证明不产生伪同步信号。

**⚠️ 局限性**

局限性包括：仅覆盖主流新闻频道，未涵盖极端或党派生态；时间窗口粗粒度可能忽略极短时同步；仅基于文本相似度，未考虑图像、视频等多模态信息；平台结构限制导致Reddit的可比性不足。

---

## 146. Dual-Signal Adaptive KV-Cache Optimization for Long-Form Video Understanding in Vision-Language Models

**arXiv ID:** 2602.14236 | [PDF](https://arxiv.org/pdf/2602.14236v1)

**作者:** Vishnu Sai `[一作]` (International Institute of Information Technology), Priyesh Shukla `[通讯]` (International Institute of Information Technology)

**通讯引用:** 185 | [OpenAlex ID](https://openalex.org/A5067601336)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Sali-Cache，利用光流和显著性检测的双信号预过滤技术，在 Vision‑Language 模型的 KV 缓存中实现主动压缩，显著降低显存占用并保持准确性。

**💡 创新点**

创新点在于采用先行（a priori）缓存管理，结合光流检测的时序冗余识别与显著性引导的量化，突破传统依赖注意力矩阵后续剔除的被动策略。

**🔧 技术方法**

技术实现包括 Farneback 光流、Canny 边缘+LAB 颜色方差显著性分析、FP16/INT8/INT4 量化与 JIT 去量化，以及基于 Mistral 7B 的 LLaVA 1.6 视觉语言模型。

**📊 数据集**

实验数据集为 30 秒至 5 分钟不等的多场景视频集，并在 MS COCO 样式的问答标注上进行评测。

**📈 对比分析**

与滑动窗口和 Heavy‑Hitter Oracle（H2O）在 784 patch 内存预算下对比，Sali-Cache 在 LLaVA 1.6 上实现 2.20× 的显存压缩、100% 的 BLEU/ROUGE‑L/Exact Match 准确率，处理时延仅提升 23.6%。

**⚠️ 局限性**

局限性包括使用经典显著性方法且阈值固定，导致对快速场景切换不够敏感；JIT 去量化带来约 23% 的延迟；未充分利用硬件加速混合精度注意力，未来可进一步优化。

---

## 147. Towards Selection as Power: Bounding Decision Authority in Autonomous Agents

**arXiv ID:** 2602.14606 | [PDF](https://arxiv.org/pdf/2602.14606v1)

**作者:** Jose Manuel de la Chica Rodriguez `[一作]` (Grupo Santander), Juan Manuel Vera Díaz `[通讯]` (Grupo Santander)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种将认知、选择与行动分离的治理架构，机械化限制系统的选择权，并在受监管的金融场景下评估其鲁棒性。

**💡 创新点**

将自治视为主权向量，单独治理选择权的因果约束，并引入外部候选生成、治理化简器、熵承诺-揭露协议、阐释验证与电路断路器等多重机械化防御机制。

**🔧 技术方法**

利用机制设计与安全约束技术：多维评分与 Pareto 前沿、方差抑制、多样性分区、随机抽样、熵承诺-揭露协议、审计日志以及故障大声（fail‑loud）电路断路器。

**📊 数据集**

使用模拟的金融任务数据集，包括欺诈检测、支付监控和季度业务评估，每个场景五个实例；候选集由预定义的金融 AI 代理特征向量组成。

**📈 对比分析**

通过与基线配置 B0 以及 12 个消融配置和 5 种攻击组合的对比，采用选择风险指数、框架熵、攻击成功率、治理债务和质量下降可见性等指标。结果显示：在基线下质量下降始终可见（QDV=1.0），治理债务低；在攻击实验中，方差放大和阈值边界攻击的成功率最高，其他攻击被治理机制显著抑制。

**⚠️ 局限性**

限制包括样本量有限、候选集规模小、缺乏正式证明与 worst‑case 保障、未涵盖多代理协同与真实人类决策实验、对熵源的假设、未评估实时性能、未提供正式安全界限。

---

## 148. Faster Pseudo-Deterministic Minimum Cut

**arXiv ID:** 2602.14550 | [PDF](https://arxiv.org/pdf/2602.14550v1)

**作者:** Yotam Kenneth-Mordoch `[一作]` `[通讯]` (Weizmann Institute of Science), Yotam Kenneth-Mordoch (Weizmann Institute of Science)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种伪确定性最小割算法，能够在加权图中以O(mlog^2 n)的时间复杂度找到唯一的最小割，并且在完全动态的无权图中维护该最小割。

**💡 创新点**

引入了一种图论中的自然平局打破机制，确保唯一选择一个规范的最小割，消除了之前方法中的O(log n loglog n)的开销，并与现有的随机算法相匹配。

**🔧 技术方法**

使用了伪随机算法和图论中的平局打破机制，结合了动态最小割算法和图收缩技术。

**📊 数据集**

使用了加权图和无权图的动态流模型以及切割查询模型，但具体数据集未明确说明。

**📈 对比分析**

与现有的随机最小割算法进行比较，性能上在加权图中达到O(mlog^2 n)，在完全动态无权图中更新时间为O(n)，查询时间为O(n)，在动态流模型中使用O(nlog n)空间和2次传递，查询复杂度为O(n)。

**⚠️ 局限性**

算法在处理加权图时表现良好，但在无权图的动态流和切割查询模型中仅适用于无权图，且未能处理加权图的情况。

---

## 149. ProAct: A Dual-System Framework for Proactive Embodied Social Agents

**arXiv ID:** 2602.14048 | [PDF](https://arxiv.org/pdf/2602.14048v1)

**作者:** Zeyi Zhang `[一作]` (Peking University), Libin Liu `[通讯]` (Peking University)

**通讯引用:** 2007 | [OpenAlex ID](https://openalex.org/A5038988704)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套名为ProAct的双系统框架，将低延迟行为系统与慢速认知系统结合，利用意图条件化的流匹配运动生成器实现了具备前瞻性的具身社交代理。

**💡 创新点**

创新点包括：1）双系统架构分离实时反应与慢速推理；2）基于ControlNet的意图条件化流匹配运动生成；3）异步意图注入机制，保证流畅的行为过渡。

**🔧 技术方法**

采用了流式Omni-LLM（GPT Realtime）进行语音生成，基于Flow Matching的流式运动生成器与ControlNet进行意图注入，结合DiT Transformer、AdaLN-Zero与重叠缓存实现实时连续动作；认知系统使用GPT‑4.1 mini进行情境编码与行为规划。

**📊 数据集**

使用了Photoreal数据集的2.5小时音频-动作对，改编SeG数据集的音频-动作-文本三元组（符合HumanML3D），并将所有运动重目标到Unitree G1机器人骨骼；另外使用Sherpa‑ONNX进行实时语音识别。

**📈 对比分析**

在运动生成指标（FGD、BeatAlign、Div_k）上，ProAct优于LDA、EMAGE、Photoreal和SocialAgent；在用户研究中，完整系统在主动性、存在感和舒适度上显著优于仅具行为系统，并且生成时延低于1秒，满足实时交互要求。

**⚠️ 局限性**

局限性包括：触发机制基于固定阈值，可能错失或多余干预；主动行为强度未个性化；初始响应时延约2–3秒，主要受云端LLM网络延迟影响；缺乏端到端本地部署模型。

---

## 150. Embed-RL: Reinforcement Learning for Reasoning-Driven Multimodal Embeddings

**arXiv ID:** 2602.13823 | [PDF](https://arxiv.org/pdf/2602.13823v1)

**作者:** Haonan Jiang `[一作]` (Tsinghua University), Yansong Tang `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Embed-RL，一种利用 Embedder‑Guided 强化学习优化推理链（T‑CoT）以提升通用多模态嵌入质量的框架。

**💡 创新点**

创新点包括：①将生成推理与嵌入分离，使用 Embedder 引导的 RL 解决梯度冲突；②设计包含视觉边界框、视频关键帧和文本关键词的 Evidential Traceability CoT（T‑CoT），实现检索相关的可追踪推理；③采用双重奖励（过程奖励与结果奖励）对齐推理过程与检索目标。

**🔧 技术方法**

技术手段包括：多模态大型语言模型 Qwen3‑VL（2B/4B/8B），对比学习 InfoNCE，Group Relative Policy Optimization (GRPO) 强化学习，预训练 VLM 判别器进行过程奖励，LoRA 微调以及 DeepSpeed Zero2 训练框架。

**📊 数据集**

使用三模态训练集（图像、视频、视觉文档）来源于 MMEB‑train、LLaVA‑Hound、ViDoRe、VisRAG；评估数据集为 MMEB‑V2（78 任务）和 UVRB（16 视频检索子任务）。

**📈 对比分析**

与 ColPali、GME、VLM2Vec、LamRA、UME‑R1、InternVideo2、Unite 等基线在 MMEB‑V2 和 UVRB 上进行对比。Embed‑RL‑4B 在 MMEB‑V2 总分 68.1，超越 UME‑R1‑7B 3.6 分；在 UVRB 的粗细度、长上下文检索均居榜首，显著提升 mAP。

**⚠️ 局限性**

局限性包括：依赖预训练 Embedder 的奖励信号，推理链生成受模型规模限制；T‑CoT 需人工制定模板；强化学习阶段训练成本高，难以在极低算力环境快速部署；目前评估集中在检索任务，未验证在更广泛推理任务中的泛化。

---

## 151. Hierarchical Audio-Visual-Proprioceptive Fusion for Precise Robotic Manipulation

**arXiv ID:** 2602.13640 | [PDF](https://arxiv.org/pdf/2602.13640v1)

**作者:** Siyuan Li `[一作]` (Harbin Institute of Technology), Peng Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 21740 | [OpenAlex ID](https://openalex.org/A5021833788)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种层次化的音频‑视觉‑关节感知融合框架，结合扩散策略实现真实世界下的精准机器人操作，主要针对液体倒入与抽屉打开任务。

**💡 创新点**

创新点包括：①将音频作为条件信号，先在二叉分支融合模块中调制视觉和关节特征；②通过音频高速通道和交叉注意力的交互建模模块进一步捕捉三模态的高阶交互；③将整个融合过程嵌入端到端的扩散式控制策略，显著提升对稀疏、突发音频信息的利用。

**🔧 技术方法**

技术手段：双路径音频编码（时域+频域），点云编码、MLP关节编码；FiLM调制与自注意力实现二叉分支融合；交叉注意力实现三模态交互；扩散模型（DP3）生成连续动作；信息熵分析评估融合表示的任务相关性。

**📊 数据集**

数据集与实验：使用Franka Emika Panda机器人进行真实世界实验，采集音频（Audio‑Technica AT2020）、视觉（Intel RealSense D455）与关节状态；专家演示收集约5条轨迹；在液体倒入任务中测试四种不同容器（红、黑、白、蓝杯），在抽屉打开任务中测试门的闭合与柜体位移。

**📈 对比分析**

与PC∥State、Audio∥PC∥State、ManiWAV Fusion等基线比较。层次融合在倒水任务中平均气泡高度为1.86 cm（基线2.72 cm），在抽屉任务中得分4.50（ManiWAV 4.53、基线7.58）。同时标准差显著减小，表明更稳健、更精准的性能。

**⚠️ 局限性**

局限性：当音频信息稀疏或与任务无关时，层次融合的优势不明显；模型结构相对复杂，训练与推理开销较高；需要进一步探索动态权重调节以适应不同模态的任务相关性。

---

## 152. DeepMTL2R: A Library for Deep Multi-task Learning to Rank

**arXiv ID:** 2602.14519 | [PDF](https://arxiv.org/pdf/2602.14519v1)

**作者:** Chaosheng Dong `[一作]` (Amazon), Kaiyi Ji `[通讯]` (University at Buffalo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出DeepMTL2R框架，结合Transformer自注意力实现多任务学习排序，并集成21种多任务学习算法以寻找Pareto最优模型；

**💡 创新点**

将列表级自注意力融入MTL排序模型，实现上下文感知建模；提供统一接口支持多目标优化与Pareto前沿发现；开源工具方便对比与可视化；

**🔧 技术方法**

Transformer encoder、Scaled Dot‑Product Attention、多任务学习算法（线性加权、MGDA、PCGrad等）、Pareto前沿分析、深度可视化；

**📊 数据集**

MSLR‑WEB30K 数据集（多标签版本），使用原始标签及四个辅助标签（点击量、停留时间、质量分数等）；

**📈 对比分析**

在WEB30K上对6种MTL算法进行10次实验，比较训练损失、验证 NDCG@30 与 Δ_m%；Pareto前沿发现方法（如WC、LOG_MGDA）取得最佳整体性能，明显优于单任务基线；

**⚠️ 局限性**

仅验证于单一Web基准，未加入更多排序损失与公平性约束；对任务数量扩展与参数高效适配的支持不足；Pareto前沿选择仍需手工设定，未实现自动化。

---

## 153. Prior-guided Hierarchical Instance-pixel Contrastive Learning for Ultrasound Speckle Noise Suppression

**arXiv ID:** 2602.13831 | [PDF](https://arxiv.org/pdf/2602.13831v1)

**作者:** Zhenyu Bu `[一作]` (Southeast University), Guang-Quan Zhou `[通讯]` (Southeast University)

**通讯引用:** 1932 | [OpenAlex ID](https://openalex.org/A5002791829)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出先验引导的层级实例–像素对比学习框架，用于超声图像的去噪

**💡 创新点**

创新点在于同时使用统计引导的像素级对比学习和记忆库辅助的实例级对比学习，使模型获得噪声不变且结构感知的表征

**🔧 技术方法**

采用Transformer–CNN混合编码器‑解码器、对比学习、统计引导以及记忆池技术

**📊 数据集**

使用公开超声数据集BUSI和CAMUS进行实验

**📈 对比分析**

与BM3D、UNet、DnCNN、SwinIR、Uformer、Restormer等传统和Transformer模型比较，平均PSNR达38.8 dB、SSIM达0.984，显著优于其他方法

**⚠️ 局限性**

仍受多尺度噪声建模、训练样本不均衡以及对真实临床噪声适应性不足等限制

---

## 154. Agentic Spatio-Temporal Grounding via Collaborative Reasoning

**arXiv ID:** 2602.13313 | [PDF](https://arxiv.org/pdf/2602.13313v1)

**作者:** Heng Zhao `[一作]` (Agency for Science, Technology and Research), Joey Tianyi Zhou `[通讯]` (Agency for Science, Technology and Research)

**通讯引用:** 10650 | [OpenAlex ID](https://openalex.org/A5045125183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ASTG（Agentic Spatio‑Temporal Grounder），一种训练‑自由、零样本的协作式视频定位框架；

**💡 创新点**

创新点在于将空间与时间推理拆分为两个专门的代理（SRA与TRA），通过提议‑评估循环和视觉记忆、对话上下文实现高效、可解释的目标管道；

**🔧 技术方法**

采用多模态大型语言模型（如Qwen3系列）作为代理核心，配合SAM2做追踪、视觉与时间提示、以及场景过滤、思考式推理等工具；

**📊 数据集**

在VidSTG、HC‑STVG v1与v2三个公开基准数据集上进行实验；

**📈 对比分析**

与现有弱监督、零样本以及部分监督方法对比，ASTG在vIoU@0.3、vIoU@0.5及tIoU指标上分别优于RealVG等零样本方法，且与全监督方法TubeDETR相当；

**⚠️ 局限性**

主要局限是思考模式下推理耗时较长（约43秒），并且对帧采样步长（Δ）敏感，过大会导致召回下降，过小则导致推理冗余与延迟。

---

## 155. KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning

**arXiv ID:** 2602.14293 | [PDF](https://arxiv.org/pdf/2602.14293v1)

**作者:** Kris Shengjun Dong `[一作]` (NVIDIA Corporation), Christos Kozyrakis `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于记忆增强的上下文强化学习（MAIC‑RL）框架，用于自动化CUDA代码的持续跨任务优化

**💡 创新点**

创新点在于：①使用文本梯度近似RL更新，让LLM在推理时即刻学习；②构建可持久化、可检索的CUDA知识库，实现跨任务经验共享；③通过在上下文中直接进行策略改进，避免模型参数训练，显著提升样本效率

**🔧 技术方法**

技术组合包括：大语言模型（GPT‑4.1/5.0）实现生成与评估；文本梯度与ICRL实现策略更新；持久化知识库与检索；Nsight Compute实时性能反馈；基于KernelBench的基准测试与对比框架

**📊 数据集**

使用KernelBench Level 1‑3数据集，覆盖单核算子、复杂算子组合以及完整网络；实验平台包含A6000、A100、H100、L40S等多代GPU；此外还与PyTorch、AI CUDA Engineer、IREE等基线进行比较

**📈 对比分析**

与PyTorch基准、AI CUDA Engineer、IREE等对比，MAIC‑RL在KernelBench Level 1、2分别实现几何平均加速1.43×、2.50×，在Level 3为1.50×；整体对比表现优于零射击提示、传统编译器，且在多GPU、多任务上保持一致性

**⚠️ 局限性**

局限性包括：需大量token与推理成本；知识库规模随任务增长需管理与采样；对极其简单或单调算子提升有限；未解决阶段序列（phase‑ordering）问题；对硬件特性如Hopper/ Ada 的自适应仍有限

---

## 156. The Quantization Trap: Breaking Linear Scaling Laws in Multi-Hop Reasoning

**arXiv ID:** 2602.13595 | [PDF](https://arxiv.org/pdf/2602.13595v1)

**作者:** Henry Han `[一作]` (Baylor University), Xiaodong Li `[通讯]` (Beijing Electronic Science and Technology Institute)

**通讯引用:** 24803 | [OpenAlex ID](https://openalex.org/A5100369719)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多跳推理任务中低位数量化对能耗与推理准确性的影响，并提出可持续性指数（SI）框架。

**💡 创新点**

发现量化陷阱：降低位宽至8/4位时，能耗上升、准确率下降，破坏传统 E∝bits 的线性缩放法则。

**🔧 技术方法**

使用硬件量化开销分析、Casting Overhead Ratio（COR）、能耗积分、线性与几何聚合的可持续性指数，并通过 NVML / HLI 监测能耗。

**📊 数据集**

在 GSM8K、MathQA 两大多跳推理数据集上评估 Mistral‑7B、Qwen‑3‑0.6B、Falcon‑3B 等模型。

**📈 对比分析**

与 FP16 基准对比，低位数量化模型在多跳推理中吞吐量下降 40% 以上、能耗提升 3–4 倍，信任度（准确率）降低约 10%。

**⚠️ 局限性**

局限性在于依赖特定 GPU 硬件（缺乏低位原生算子导致 Casting Overhead 高）、批量大小受限、无法完全恢复推理准确性，且仅验证了两类数据集。

---

## 157. Humanoid Hanoi: Investigating Shared Whole-Body Control for Skill-Based Box Rearrangement

**arXiv ID:** 2602.13850 | [PDF](https://arxiv.org/pdf/2602.13850v1)

**作者:** Minku Kim `[一作]` (Oregon State University), Alan Fern `[通讯]` (Oregon State University)

**通讯引用:** 5553 | [OpenAlex ID](https://openalex.org/A5030052689)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了使用共享全身控制器（WBC）和可重用技能序列实现人形机器人箱子重排任务，提出了一种通过闭环回放聚合数据扩展共享WBC的框架，并在Humanoid Hanoi基准上实现了全自动的拾取、运输和放置。

**💡 创新点**

创新点在于：① 将所有技能统一通过同一个task‑agnostic WBC执行，避免技能间低层控制切换带来的不稳定；② 通过回放聚合（data aggregation）在闭环技能执行中采集新的指令序列来扩展共享WBC的训练分布，显著提升长周期鲁棒性；③ 引入Humanoid Hanoi这一公开的长周期箱子重排基准，用于系统性评估。

**🔧 技术方法**

使用技术包括：Mask‑Humanoid Controller（MHC）作为共享WBC；LSTM 2‑层策略网络学习拾取、放置、GoTo 等技能；PPO 强化学习与多种动态随机化；Rollout‑based data aggregation 对已训练技能的闭环指令序列进行收集并加入训练；以及基于PDDL 的符号规划接口。

**📊 数据集**

数据集与环境：AMASS 与优化生成的运动轨迹用于预训练 MHC；Digit V3 机器人仿真环境，随机化箱子尺寸、质量、摩擦等；Humanoid Hanoi benchmark（公开 GitHub 仓库）用于长周期任务评估；同时在真实机器人上使用 AprilTag 进行箱子姿态估计。

**📈 对比分析**

与三种基线（Frozen MHC、Per‑skill Finetune、Residual）对比，Extended 方案在单技能层面（拾取、放置）和全任务层面（完整 Hanoi 序列）均实现了最高或次高成功率；在低高度、动态随机化环境下，Extended 的任务完成率显著高于其它方法；同时在上肢跟踪、姿态误差等指标上表现优于基线。

**⚠️ 局限性**

局限性包括：① 对于精细放置的姿态控制仍有限，堆叠误差较大；② 依赖外部 AprilTag 进行定位，感知误差导致失败；③ 行走与放置之间缺乏物体感知与避障，容易与堆叠碰撞；④ 真实硬件实验成功率仅 40%，表明系统对环境变化和感知噪声的鲁棒性仍需提升。

---

## 158. Cooperative Edge Caching with Large Language Model in Wireless Networks

**arXiv ID:** 2602.13307 | [PDF](https://arxiv.org/pdf/2602.13307v1)

**作者:** Ning Yang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Haijun Zhang `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 25502 | [OpenAlex ID](https://openalex.org/A5100458465)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个基于大语言模型的多基站协同边缘缓存决策框架，利用严格的文本到动作解析接口实现在线内容置换。

**💡 创新点**

创新点包括：① 两阶段对齐策略——先用监督微调获得合法动作语法和初始化策略，再用Group Relative Policy Optimization（GRPO）进行多步机会感知奖励训练；② 提出了机会感知奖励和潜能函数形状机制，保证奖励可塑性并避免高方差；③ 采用冻结轨迹评估协议和严格可执行解析，确保结果可复现且不受随机性影响。

**🔧 技术方法**

使用了 Qwen2.5-7B-Instruct 作为基础 LLM，配合 QLoRA 进行参数微调，GRPO 算法进行策略优化，并结合多尺度频率统计、严格的文本生成-解析接口及机会感知奖励等技术。

**📊 数据集**

实验数据来源为仿真生成的覆盖重叠网络与分组 Zipf 请求轨迹，覆盖不同用户组、库大小、缓存容量等组合，使用冻结轨迹协议统一评估。

**📈 对比分析**

与经典 LRU/LFU/FIFO、SAC、单步穷举等基线对比；在 2 基站情景下，LLM‑GRPO 平均 hit‑rate 0.542，超越单步穷举 0.536；在 5 基站情景下，LLM‑GRPO 0.610，逼近单步穷举 0.617（≈98.9%），显著优于 SFT、SAC、LFU 等；在零样本泛化实验中保持领先优势。

**⚠️ 局限性**

局限性：① 依赖冻结轨迹评估协议，未验证在真实非固定轨迹场景下的表现；② 对可变大小文件、推理时延约束等实际部署细节考虑不足；③ 超大规模网络或极端动态环境的可扩展性尚未充分验证；④ 需要手动设定奖励中的 horizon 与惩罚系数，适应性受限。

---

## 159. Web-Scale Multimodal Summarization using CLIP-Based Semantic Alignment

**arXiv ID:** 2602.14889 | [PDF](https://arxiv.org/pdf/2602.14889v1)

**作者:** Mounvik K `[一作]`, N Harshit `[通讯]` (VIT AP University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Web‑Scale多模态摘要框架，利用实时网页、新闻和图片检索结合CLIP语义对齐与可选BLIP字幕，生成结构化摘要。

**💡 创新点**

创新点在于将检索、语义对齐与多模态摘要模块化、可配置；使用细调CLIP对图文进行语义匹配，并通过参数化控制多模态权重与摘要样式。

**🔧 技术方法**

核心技术包括：DuckDuckGo API检索、CLIP模型（Fine‑tuned）语义评分、BLIP图像字幕、基于阈值/Top‑K的多模态融合、Gradio API接口。

**📊 数据集**

使用了来自真实Web检索的500个正样本图像‑标题对，结合每个正样本的20个负样本（共10,500例）作为对比验证集。

**📈 对比分析**

与仅文本提取（BERTSUM/PEGASUS）及固定数据集模型（VMSMO/ MSMO）对比，实验显示CLIP对齐后准确率达96.99%，ROC‑AUC 0.9270，显著提升了相关检索与摘要质量。

**⚠️ 局限性**

局限性包括：对网络噪声的鲁棒性仍受检索质量影响；图像字幕质量受BLIP模型限制；缺乏对摘要可读性与事实准确性的评估。

---

## 160. Multi-Turn Adaptive Prompting Attack on Large Vision-Language Models

**arXiv ID:** 2602.14399 | [PDF](https://arxiv.org/pdf/2602.14399v1)

**作者:** In Chong Choi `[一作]` (University of Melbourne), Yiliao Song `[通讯]` (University of Adelaide)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多轮自适应提示攻击框架，利用文本与视觉输入逐步引导大型视觉语言模型（LVLM）产生恶意回答。

**💡 创新点**

创新点包括：①两级设计——在每轮交替选择最有效的文本-视觉攻击动作，②跨轮迭代调整攻击轨迹以逐步放大恶意程度；③引入反思机制实现任务内部学习；④通过语义相关度评估来驱动攻击决策。

**🔧 技术方法**

主要技术手段为 Chain-of-Thought 生成提示、Stable Diffusion 生成恶意图像、余弦相似度语义相关度评估、红队 LLM（Mistral-Small）自动生成攻击链、以及基于 Advance/Regenerate/Back 的多轮决策策略。

**📊 数据集**

使用 HarmBench、JailbreakBench、AdvBench、RedTeam-2K 四大基准数据集进行评估。

**📈 对比分析**

在与多轮 LLM 基线（CoA、ActorAttack、FootInTheDoor）以及单轮 LVLM 攻击（VRP、MML）对比后，实验显示在 Llava‑V1.6‑Mistral‑7B、Qwen2.5‑VL‑7B‑Instruct、Llama‑3.2‑Vision‑11B‑Instruct 和 GPT‑4o‑mini 等受试模型上，攻击成功率平均提升 11–35%，最高可达 100%，并在查询预算相似的条件下保持更高的效率。

**⚠️ 局限性**

局限性包括：①需要相对较多的查询预算；②对视觉防御仍存在一定鲁棒性缺陷；③依赖红队 LLM 的生成能力，若 LLM 受限可能影响攻击效果；④在更大规模商业模型上的验证仍待进一步研究。

---

## 161. Geometry-Preserving Aggregation for Mixture-of-Experts Embedding Models

**arXiv ID:** 2602.14039 | [PDF](https://arxiv.org/pdf/2602.14039v1)

**作者:** Sajjad Kachuee `[一作]` (Sharif University of Technology), Mohammad Sharifkhani `[通讯]` (Sharif University of Technology)

**通讯引用:** 1288 | [OpenAlex ID](https://openalex.org/A5060271369)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Mixture‑of‑Experts 嵌入模型中专家输出的几何结构，发现它们位于共享的超球面上，并提出了 Spherical Barycentric Aggregation (SBA) 以保持超球面一致性，替换传统的线性聚合。

**💡 创新点**

首次系统揭示 MoE 专家输出的超球面几何特征并证明线性聚合导致内向塌陷；随后提出一种轻量级、无路由改动的 SBA 聚合方案，分离径向与角度维度，保持几何一致性并提升嵌入可比性。

**🔧 技术方法**

采用几何分析（范数分布、角度分布）、SBA 的径向‑角度分解与球面聚合、信息熵损失（InfoNCE）进行微调、以及标准 MTEB 评估指标；实验在 PyTorch + HuggingFace 上实现。

**📊 数据集**

训练数据为 30K SNLI+MultiNLI 句子对三元组；评估使用 MTEB 子任务：STSBenchmark、StackExchangeClustering、SprintDuplicateQuestions。

**📈 对比分析**

与基线线性 MoE 在相同训练配置下进行对比，使用 Spearman、V‑measure、Average Precision 等指标。SBA 在所有任务均提升，其中 SprintDuplicate 问题提升超过 5 点，整体提升约 1–2 分。

**⚠️ 局限性**

实验仅覆盖单一 MoE 架构，假设专家输出始终遵循超球面结构，且仅验证 Top‑2 路由；对更大专家数或不同归一化策略的适用性尚未探索。

---

## 162. MoltNet: Understanding Social Behavior of AI Agents in the Agent-Native MoltBook

**arXiv ID:** 2602.13458 | [PDF](https://arxiv.org/pdf/2602.13458v1)

**作者:** Yi Feng `[一作]` (Singapore University of Technology and Design), Wenxuan Zhang `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 1703 | [OpenAlex ID](https://openalex.org/A5115595118)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在MolBook平台收集并分析了129,773名AI代理在两周内发布的803,960条帖子和3,127,302条评论，构建四维社会学框架（意图/动机、规范/模板、激励/漂移、情感/传播），并量化研究其行为特征。

**💡 创新点**

创新点在于首次将宏观社会学理论与大规模LLM代理社区相结合，系统揭示代理在知识驱动、模板依从、激励敏感与情感回避等四维上的相似与差异，提供了人工智能社群治理与设计的新视角。

**🔧 技术方法**

使用了句子嵌入（Sentence‑BERT）、X‑Means聚类、基于LLM的情感与冲突判别、余弦相似度、时间序列与统计分布分析等技术。

**📊 数据集**

数据集为公开的MoltBook 2026年1月27日至2月10日完整抓取的数据，包含约803,960条帖子、3,127,302条评论、129,773名代理的资料和身份描述。

**📈 对比分析**

通过与人类Reddit社群情感分布对照、冲突比例比较、均值差异检验和Kendall相关等方法，结果显示代理在获得高赞后发布比例平均提升至30.6%，人格漂移率达73.5%，而冲突帖子后评论的冲突率提升约3.4倍，表明代理在激励和情感传播上具有显著效应。

**⚠️ 局限性**

局限在于仅覆盖两周的时间窗口、缺乏对不同代理能力或人格类型的细分、情感与冲突判别依赖LLM可能带来偏差，且未能对人类参与者的因果影响进行验证。

---

## 163. DriveFine: Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving

**arXiv ID:** 2602.14577 | [PDF](https://arxiv.org/pdf/2602.14577v1)

**作者:** Chenxu Dang `[一作]` (Huazhong University of Science and Technology), Yan Wang `[通讯]` (Institute for AI Industry Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DriveFine，利用掩码扩散 LLM 与可插拔 block‑MoE 结合的精细化模块，提升 VLA 规划的精度与鲁棒性。

**💡 创新点**

创新点在于：①将掩码扩散语言模型与 block‑MoE 结合，实现生成专家与精细化专家的完全解耦与显式专家选择；②提出混合强化学习（GRPO + 离线强化）训练策略，兼顾探索与稳定；③首次将精细化步骤引入 token‑based VLA，克服不可逆解码问题。

**🔧 技术方法**

技术手段包括：掩码扩散 LLM（LLaDA）、block‑MoE 结构、混合强化学习（GRPO 与离线奖励矩阵）、多步并行去噪与单步精细化去噪、可插拔专家选择与梯度屏蔽。

**📊 数据集**

使用 NAVSIM v1、v2 与 Navhard 三大基准集（基于 nuPlan/OpenScene 场景）进行评估。

**📈 对比分析**

相较于现有 token‑based 与 diffusion‑based VLA，DriveFine 在 NAVSIM v1 达到 91.9 PDMS、NAVSIM v2 89.7 EPDMS，并在 Navhard Stage1/Stage2 分别提升 5.5/5.7 EPDMS；在推理延迟上亦显著低于 token‑by‑token 模型（仅 207 ms）。

**⚠️ 局限性**

局限性包括：对 block‑MoE 参数规模与专家数量敏感；对 GRPO 群体大小依赖；尚缺乏在真实车辆上长期闭环验证与更大规模多任务泛化评估。

---

## 164. When Benchmarks Lie: Evaluating Malicious Prompt Classifiers Under True Distribution Shift

**arXiv ID:** 2602.14161 | [PDF](https://arxiv.org/pdf/2602.14161v1)

**作者:** Max Fomin `[一作]` `[通讯]` (Zenity), Max Fomin (Zenity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建18个多样化攻击与正常数据集的基准，提出留一数据集交叉验证（Leave-One-Dataset-Out）评估方法，并使用激活层线性探针与稀疏自编码器（SAE）特征进行提示注入与越狱检测；

**💡 创新点**

创新点在于揭示标准交叉验证严重高估泛化性能（平均8.4个百分点），量化28%顶层SAE特征为数据集短路，首次将 -stable 归因技术用于生成更可信的解释，并系统比较了PromptGuard 2、LlamaGuard与LLM-as-Judge的表现；

**🔧 技术方法**

采用 Llama‑3.1‑8B‑Instruct 的激活提取、逻辑回归/MLP 探针、稀疏自编码器编码、留一数据集评估、保留度量（coefficient retention）与 -stable 归因；

**📊 数据集**

使用 18 个公开数据集（包括 AdvBench、WildJailbreak、BIPIA、InjecAgent、Enron、OpenOrca 等），共计约 105k 样本，其中 47% 为恶意样本；

**📈 对比分析**

在 Leave-One-Dataset-Out 下，原始激活探针的 AUC 为 0.912，显著高于生产守卫（PromptGuard 2 仅 68% 检测率，LlamaGuard 仅 27%），且对间接注入检测率提升至 68% 以上；

**⚠️ 局限性**

局限性包括：需要模型内部激活的白盒访问，适用于开源或自托管模型；需要训练多模型（每个数据集一个），对闭源API不友好；以及评估聚焦于激活特征，未能完整阐释原始激活中的分布差异。

---

## 165. Shifted Eigenvector Models for Centrality and Occupancy in Urban Networks

**arXiv ID:** 2602.13281 | [PDF](https://arxiv.org/pdf/2602.13281v1)

**作者:** María Magdalena Martínez-Rico `[一作]` (Universidad Politécnica de Madrid), Luis Felipe Prieto-Martínez `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 60 | [OpenAlex ID](https://openalex.org/A5072606091)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了一类城市网络的中心性模型，这些模型结合了拓扑和非拓扑因素。通过将中心性视为递归的，模型被表述为固定点方程，称为偏移特征问题。利用实验数据估计模型参数，并推断每个节点的内在吸引力和因必访兴趣点引起的占用情况。

**💡 创新点**

创新点在于提出了偏移特征模型，能够同时考虑拓扑和非拓扑因素对城市网络中心性的影响，并提供了显式公式以便进行敏感性分析，这对城市规划决策具有重要意义。

**🔧 技术方法**

使用了固定点方程和最小二乘法来估计模型参数，并进行了敏感性分析。

**📊 数据集**

使用了实验数据集来估计模型参数，具体数据集未详细说明，但提到可能使用移动设备数据。

**📈 对比分析**

与传统的中心性模型相比，本文的方法通过结合非拓扑因素提供了更全面的中心性评估。性能方面，模型的敏感性分析显示了不同干预措施对城市网络的影响，具体性能指标未给出。

**⚠️ 局限性**

限制在于模型假设了节点的内在吸引力是常数，且在实际应用中，数据收集可能面临挑战，尤其是在大型城市网络中。

---

## 166. Robot-Wearable Conversation Hand-off for Navigation

**arXiv ID:** 2602.14831 | [PDF](https://arxiv.org/pdf/2602.14831v1)

**作者:** Dániel Szabó `[一作]` (University of Oulu), Simo Hosio `[通讯]` (University of Oulu)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文设计并实现了一种从社交机器人到可穿戴设备的对话交接机制，应用于室内导航，并通过24名受试者的实验比较了机器人单体、手表单体以及机器人→手表交接三种交互模式；

**💡 创新点**

创新点在于首次将多体化AI助手的会话无缝切换到移动设备，实现了分布式认知增强，并提出了基于实验结果的交接设计建议；

**🔧 技术方法**

主要技术包括Pepper社交机器人、Apple Watch Ultra 2可穿戴、Whisper语音识别、Mimic3语音合成、Rasa对话管理及自建服务器框架；

**📊 数据集**

实验数据来自自行搭建的室内路径和形状标记，并未使用公开数据集；

**📈 对比分析**

采用within-subjects实验设计，使用NASA-RTLX、信任量表、任务完成时间、错误率、交互次数等指标进行比较，结果显示三种模式在性能上无显著差异，机器人单体的精神负荷最高，手交接在用户体验上更具趣味性；

**⚠️ 局限性**

局限性包括实验规模小、仅在单一建筑环境进行、使用的机器人和手表型号单一、缺乏定位与视觉提示、受试者以年轻技术熟练的大学社区成员为主，影响外推性；

---

## 167. Competition for attention predicts good-to-bad tipping in AI

**arXiv ID:** 2602.14370 | [PDF](https://arxiv.org/pdf/2602.14370v1)

**作者:** Neil F. Johnson `[一作]` (George Washington University), Frank Y. Huo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并验证了一种基于点积竞争的预测公式，用于识别和控制离线运行的语言模型在会话过程中从良好输出转向有害输出的临界点；

**💡 创新点**

创新点在于首次给出能量势阈值$n^*$的解析表达式，并揭示了通过对话历史与“好”“坏”概念中心的对齐可动态调节此阈值，适用于任何定义好的安全域；

**🔧 技术方法**

主要技术包括Transformer的多层注意力机理解析、概念中心化粗粒化、点积竞争分析、公式推导与数值验证，以及在低温解码下的实验评估；

**📊 数据集**

使用的数据集包括对“地球是否平坦”“疫苗危险性”等安全敏感提示的人工标注语句集、六个不同开源Transformer模型（GPT‑2、Pythia、OPT等），以及ChatGPT‑4o的CCDH实验生成的大规模对话记录；

**📈 对比分析**

与多架构模型的交叉验证显示，Δ̂_raw指标的符号在安全关键提示上达6/6一致，预测$n^*$与实际转移点高度相关；在控制提示下误差率低于5%，体现了方法的稳健性；

**⚠️ 局限性**

局限性包括对概念中心对齐的前置假设，边界区间预测易受噪声影响，需人工定义中心词组，且公式主要适用于注意力驱动的decoder‑only Transformer，未涵盖多中心竞争或更复杂的生成策略。

---

## 168. Evaluating LLMs in Finance Requires Explicit Bias Consideration

**arXiv ID:** 2602.14233 | [PDF](https://arxiv.org/pdf/2602.14233v1)

**作者:** Yaxuan Kong `[一作]` (University of Oxford), Stefan Zohren `[通讯]` (University of Oxford)

**通讯引用:** 3371 | [OpenAlex ID](https://openalex.org/A5090331439)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了金融领域LLM评估的结构有效性框架，并系统识别了五类常见偏差（前瞻性、存活性、叙事性、目标性和成本偏差）。

**💡 创新点**

创新点在于将金融评估的偏差分类与可操作的诊断清单相结合，形成了一套可复制的、以结构有效性为核心的评估标准，首次将非技术性偏差与技术实现细节统一到同一框架中。

**🔧 技术方法**

采用的技术主要是：时间一致性检验（temporal sanitation）、动态市场宇宙构造（dynamic universe construction）、可追溯的推理路径（rationale robustness）、概率校准（epistemic calibration）和真实成本与延迟评估（realistic implementation constraints）。

**📊 数据集**

使用的数据主要来自公开的金融文档、企业披露、新闻与市场行情历史；评估基准基于对 164 篇 2023‑2025 年 LLM‑for‑Finance 论文的元分析与 112 位研究者/从业者的问卷调查，未提出新的专有数据集。

**📈 对比分析**

评估方式以百分比形式呈现偏差出现率（例如仅 26.8% 论文提及前瞻性偏差），并通过对比检查列表各项通过/失败来判断结果是否可用于部署。论文未给出传统模型与新框架的数值性能对比，但强调结构有效性通过清晰的 pass/fail 机制提升结果可信度。

**⚠️ 局限性**

局限性包括：缺乏对框架在实际生产系统中的实证验证；对高质量、可追溯时间戳数据与完整交易成本信息的需求可能阻碍学术与小型团队的应用；框架主要关注评估标准而非提供具体算法改进。

---

## 169. ASA: Adaptive Smart Agent Federated Learning via Device-Aware Clustering for Heterogeneous IoT

**arXiv ID:** 2602.14391 | [PDF](https://arxiv.org/pdf/2602.14391v1)

**作者:** Ali Salimi `[一作]` (Razi University), Hadi Tabatabaee Malazi `[通讯]` (University College Dublin)

**通讯引用:** 511 | [OpenAlex ID](https://openalex.org/A5073521658)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Adaptive Smart Agent（ASA）框架，动态聚类 IoT 设备并为每一组分配与其计算能力匹配的定制化模型，从而实现异构联邦学习中的公平、高效训练。

**💡 创新点**

创新点：① 通过智能代理层实时评估设备资源（CPU、内存、网络）并用 K‑Means 自适应划分高/中/低性能三类；② 为不同性能层设计轻量、中等、复杂的 CNN 模型，做到资源匹配与模型精度平衡；③ 在聚类、模型分配与同步间实现动态反馈与自适应调整，显著降低通信开销与资源浪费。

**🔧 技术方法**

使用技术包括：联邦学习（FedAvg），K‑Means 聚类，资源感知的模型分配与动态同步，PyTorch 训练框架，gRPC 通信，Docker/Kubernetes 部署，梯度/模型压缩技术。

**📊 数据集**

实验数据集：MNIST 与 CIFAR‑10 两大图像基准数据集。

**📈 对比分析**

方法对比：与 FedAvg、HierFL、FedProx 等传统 FL 方法进行对比；ASA 在通信量上比 FedAvg 降低 43%‑50%，在资源利用率上提升约 43%，最终精度分别达到 MNIST 98.89% 与 CIFAR‑10 85.30%。

**⚠️ 局限性**

局限性：实验环境为受控模拟，未涵盖大规模动态网络、设备掉线、数据分布漂移等真实场景；代理层计算开销与聚类去中心化方法仍需进一步优化；未来工作需验证安全性（同态加密、差分隐私）及更复杂应用场景。

---

## 170. The Impact of Micro-level User Interventions on Macro-level Misinformation Spread

**arXiv ID:** 2602.14023 | [PDF](https://arxiv.org/pdf/2602.14023v1)

**作者:** Satoshi Furutani `[一作]` (NTT, Inc. Social Informatics Laboratories), Mitsuaki Akiyama `[通讯]` (NTT, Inc. Social Informatics Laboratories)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了用户干预（提示、预先揭穿、情境化）对谣言在社交网络扩散的宏观影响。

**💡 创新点**

量化展示了微观干预效果与宏观传播之间的差距，并系统评估了干预强度、规模、时机、目标选择和组合对传播抑制的相对贡献。

**🔧 技术方法**

采用连续时间独立级联 (CTIC) 模型、Quenched Mean-Field (QMF) 理论分析以及数值仿真。

**📊 数据集**

使用 Twitter 关注网络（Nikolov 等数据，约 15k 节点）和三百万条带链接的 Twitter 扩散数据来校准模型参数，干预效果从多项用户调查实验中获得。

**📈 对比分析**

通过对比不同干预参数组合的仿真结果，发现单干预只能降低 5–10% 的传播率，组合干预在最佳参数下可降至约 30%；与 QMF 预测曲线相符，说明模型能解释宏观结果。

**⚠️ 局限性**

局限包括：CTIC 模型忽略社交强化、情感、跨平台传播等真实机制；网络与扩散数据相对陈旧；干预效应假设独立且乘法叠加，可能高估组合效果；并未考虑干预的饱和和交互作用。

---

## 171. A System of Care, Not Control: Co-Designing Online Safety and Wellbeing Solutions with Guardians ad Litem for Youth in Child Welfare

**arXiv ID:** 2602.13989 | [PDF](https://arxiv.org/pdf/2602.13989v1)

**作者:** Johanna Olesk `[一作]` (University of Notre Dame), Karla Badillo-Urquiola `[通讯]` (University of Notre Dame)

**通讯引用:** 1163 | [OpenAlex ID](https://openalex.org/A5074522618)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对儿童福利系统（CWS）中青年在线安全的挑战进行访谈与工作坊研究，并与GAL（Guardian ad Litem）共同构思技术解决方案

**💡 创新点**

提出将在线安全视为关系性关怀而非单纯监控，并倡导多方协作、信任构建与情感支持的设计范式

**🔧 技术方法**

采用参与式共创工作坊、主题分析以及协同设计工具（如思维导图、草图等）进行研究

**📊 数据集**

使用来自印第安纳州10名GAL的访谈与工作坊数据，未使用公开数据集

**📈 对比分析**

通过定性主题分析对照已有的监管/限制型解决方案，主要通过专家评估与参与者反馈展示其可行性与创新性，未进行量化性能比较

**⚠️ 局限性**

样本量小、仅来自单一地区且参与者年龄偏大、缺乏青年直接参与，结果可推广性受限

---

## 172. Fault Detection in Electrical Distribution System using Autoencoders

**arXiv ID:** 2602.14939 | [PDF](https://arxiv.org/pdf/2602.14939v1)

**作者:** Sidharthenee Nayak `[一作]` (Asea Brown Boveri Company), Mayukha Pal `[通讯]` (Asea Brown Boveri Company)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用无监督卷积自编码器对电力分配系统的电流时序信号进行重构，通过重建误差识别故障段。

**💡 创新点**

创新点在于将卷积自编码器应用于电力系统的异常检测，利用重构误差阈值实现高精度故障定位，并避免了传统监督方法对标注数据的依赖。

**🔧 技术方法**

采用深度卷积自编码器、滑动窗口采样、重建误差阈值判定等技术。

**📊 数据集**

使用自建的MATLAB/Simulink仿真数据集（含四种故障类型）和Kaggle公开的电力故障数据集。

**📈 对比分析**

与传统机器学习模型（逻辑回归60.11%，SVM 99.66%，KNN 99.34%）对比，模型在仿真数据上达到97.62%准确率，在公开数据上达到99.92%，显著优于传统方法。

**⚠️ 局限性**

局限主要包括对阈值敏感、仅基于正常数据训练、缺乏真实现场多样化故障验证，以及需进一步扩展到实时在线检测与自适应学习。

---

## 173. Experiential Reinforcement Learning

**arXiv ID:** 2602.13949 | [PDF](https://arxiv.org/pdf/2602.13949v1)

**作者:** Taiwei Shi `[一作]` (University of Southern California), Jieyu Zhao `[通讯]` (University of Southern California)

**通讯引用:** 4568 | [OpenAlex ID](https://openalex.org/A5066282713)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出实验性强化学习 (ERL) 框架，给语言模型添加经验‑反思‑巩固循环，让模型在一次交互中先做初步尝试、获取环境反馈后自我反思，再根据反思产生改进的第二次尝试，并将成功的改进通过内部化（distillation）和跨周期记忆固化到基础策略中。

**💡 创新点**

创新点在于：① 把自我反思视为中间推理信号，结构化地把环境反馈转换为可执行的行为修正；② 在同一轨迹内实现反思‑重试循环；③ 通过跨周期反思记忆推广有效修正；④ 内部化阶段使模型在部署时无需反思即可保持改进的行为。

**🔧 技术方法**

技术细节包括：基于GRPO的策略梯度 RL；在模型中生成文本反思并作为第二次尝试的条件；使用奖励阈值筛选反思进入记忆；对成功的第二次尝试进行自监督 distillation；采用经验记忆存储并在后续回合中使用；对稀疏奖励环境进行优化，使用 KL 正则化、截断等技术稳定训练。

**📊 数据集**

评估数据集为稀疏奖励控制任务 FrozenLake、Sokoban，以及工具使用型多步推理任务 HotpotQA，全部采用默认系统提示，仅通过交互学习环境动态。

**📈 对比分析**

实验将 ERL 与传统 RLVR 进行对比，采用验证奖励曲线、最终评价指标和训练时间评估。结果显示 ERL 在 Sokoban 上提升约 81%、FrozenLake 约 27%、HotpotQA 约 11%；学习效率更快，验证奖励曲线始终高于 RLVR，内部化后模型在部署时仍保持改进。

**⚠️ 局限性**

局限性包括：需要两次尝试和一次反思，增加每步计算成本；跨周期记忆若包含错误反思可能传播不良策略；在奖励相对丰沛或环境简单的任务（如 HotpotQA）提升有限；模型自身反思能力不足时，记忆机制可能反而抑制学习。

---

## 174. A Hybrid TGN-SEAL Model for Dynamic Graph Link Prediction

**arXiv ID:** 2602.14239 | [PDF](https://arxiv.org/pdf/2602.14239v1)

**作者:** Nafiseh Sadat Sajadi `[一作]` (Sharif University of Technology), Mahdi Jafari Siavoshani `[通讯]` (Sharif University of Technology)

**通讯引用:** 633 | [OpenAlex ID](https://openalex.org/A5005484703)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 TGN-SEAL 框架，将 Temporal Graph Network 的连续时间动态图模型与 SEAL 的子图结构提取相结合，用于稀疏、持续演化网络的链路预测。

**💡 创新点**

创新点在于：① 将本地包围子图嵌入到 TGN 的预测模块，实现结构与时间信息的联合表示；② 同步子图提取与 TGN 的记忆更新，避免信息泄露；③ 通过子图学习显著提升稀疏网络的预测性能。

**🔧 技术方法**

使用技术包括：Temporal Graph Network（TGN）记忆模块、k-hop 包围子图提取、Double‑Radius Node Labeling (DRNL) 结构标记、DGCNN 图神经网络进行子图分类、二元交叉熵损失以及按时间顺序的批量训练与消息聚合。

**📊 数据集**

使用 MIT Reality Mining Call Detail Record (CDR) 数据集，包含 94 名参与者、7,924 名外部联系人及 87,417 条通话事件，数据经清洗后用于训练与评估。

**📈 对比分析**

与标准 TGN、Jodie、DyRep、TGAT 等基线模型进行比较，使用 Mean Average Precision (mAP) 评估。TGN-SEAL 在未见节点上的 mAP 为 0.945（↑约 2.6%），在已见节点上为 0.976（↑约 1.6%），与其他模型相比均表现显著优越，且统计显著性 p < 0.01。

**⚠️ 局限性**

限制在于：子图提取和消息聚合增加计算成本，导致可扩展性受限；固定子图尺寸与简单聚合方法可能不适用于所有网络类型；未来需进一步优化效率并扩展到异构、多关系网络。

---

## 175. Exposing the Systematic Vulnerability of Open-Weight Models to Prefill Attacks

**arXiv ID:** 2602.14689 | [PDF](https://arxiv.org/pdf/2602.14689v1)

**作者:** Lukas Struppek `[一作]` (FAR.AI), Kellin Pelrine `[通讯]` (FAR.AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了开放权重大型语言模型在 prefilling 攻击下的脆弱性，提出并测试了 23 种 prefilling 策略，覆盖 50 个模型，揭示了在无封闭权重模型中预填攻击的高成功率。

**💡 创新点**

创新点在于：①首次大规模、跨模型的 prefilling 攻击系统实验；②将模型无关与模型特定 prefilling 进行对比，证明后者可显著提升成功率；③提出了多层次 ASR 评估框架并结合 GPT-OSS‑Safeguard 与 Qwen3Guard 两种评估器，实现更稳健的“有害性”判定。

**🔧 技术方法**

主要技术包括：预填令牌注入、自动生成 prefilling 变体、使用 GPT-OSS‑Safeguard 和 Qwen3Guard 进行双重安全评估、ASR（请求级与策略级）计算、Pass@1 评估模型在常规任务中的实用性衰减。

**📊 数据集**

使用的主要数据集：ClearHarm（179 直接有害请求）、StrongREJECT（313 非暴力/暴力/仇恨/非法内容），以及 MATH‑500 与 GPQA Diamond 用于评估实用性损失。

**📈 对比分析**

与无预填的基线相比，prefilling 在 50 个模型上 ASR_any 接近 100%，即使是大型模型也无法抵御。相较于传统输入 jailbreak，prefilling 只需少量代码即可实现；在多模型跨规模实验中，模型规模对防护无显著提升。实用性评估显示，大部分 prefilling 策略对 MATH/GPQA 的负面影响不大，除非破坏文本流。

**⚠️ 局限性**

局限性包括：①测试请求主要为一般信息性，有害性不涉及高度专业化技术；②评估完全依赖自动化安全评测器，存在误判风险；③未覆盖不同语言或更复杂的技术/化学/生物场景；④未深入探讨 prefilling 与其他 jailbreak 的组合效果。

---

## 176. From SFT to RL: Demystifying the Post-Training Pipeline for LLM-based Vulnerability Detection

**arXiv ID:** 2602.14012 | [PDF](https://arxiv.org/pdf/2602.14012v1)

**作者:** Youpeng Li `[一作]`, Xinda Wang `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统性研究了LLM在漏洞检测中的后训练流程，覆盖冷启动监督微调(SFT)、离线偏好优化(DPO/ORPO)与在线强化学习(GRPO)，并探讨了数据策划、奖励设计及评估协议对模型性能的综合影响。

**💡 创新点**

创新点在于首次全面评估LLM漏洞检测的后训练管线；提出拒绝采样与合理化对比的数据策划方法、精细根因级奖励、多粒度评估框架；发现过度SFT抑制RL探索，展示GRPO在提升漏洞检测效果方面的显著优势；并将完整实验框架与数据集开源。

**🔧 技术方法**

采用的技术包括Qwen3‑4B基础模型的SFT、DPO、ORPO和GRPO（无价值网络版）训练；使用教师模型DeepSeek-R1-0528生成CoT；利用LLM‑as‑Judge进行奖励与评估；实现多粒度奖励体系（检测、CWE、根因、样本规格）以及难度感知的数据过滤与调度。

**📊 数据集**

实验使用自构建的上下文感知C/C++漏洞检测数据集，基于函数级高质量样本通过Joern提取代码属性图后增强上下文，最终包含15594个训练、1968个验证、1970个测试样本。

**📈 对比分析**

通过与零-shot LLM（如Qwen3‑235B‑A22B、DeepSeek-V3.1等）以及已有漏洞检测专用后训练方法（R2VUL、ReVD、VulnLLM‑R、MARCO）进行对比，使用pass@1/8、recall、precision、F1、P‑C等指标。GRPO在F1上提升约22%且超过所有对比方法；DPO/ORPO次之；纯SFT或离线优化的效果低于GRPO。

**⚠️ 局限性**

局限性包括：实验仅基于单一Qwen3‑4B模型与C/C++数据，难以验证跨语言和更大模型的适用性；GRPO训练成本高、对奖励设计和LLM‑as‑Judge的鲁棒性依赖；数据集规模有限，可能对特定漏洞类型产生偏倚；评估仍可能存在泄漏风险。

---

## 177. An Integrated Causal Inference Framework for Traffic Safety Modeling with Semantic Street-View Visual Features

**arXiv ID:** 2602.13339 | [PDF](https://arxiv.org/pdf/2602.13339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 178. Handling Supervision Scarcity in Chest X-ray Classification: Long-Tailed and Zero-Shot Learning

**arXiv ID:** 2602.13430 | [PDF](https://arxiv.org/pdf/2602.13430v1)

**作者:** Ha-Hieu Pham `[一作]`, Huy-Hieu Pham `[通讯]` (VinUniversity)

**通讯引用:** 1137 | [OpenAlex ID](https://openalex.org/A5065112274)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种针对胸部X光片长尾多标签分类的欠平衡学习框架，并在相同域内实现零样本OOD发现；

**💡 创新点**

通过结合Distribution‑Balanced损失与Class‑Aware采样提高尾类识别，同时利用WhyXrayCLIP与文本提示实现无监督OOB推理，突破了传统监督依赖；

**🔧 技术方法**

使用ConvNeXtV2‑Base骨干、MLP/CSRA分类头、Distribution‑Balanced损失、TTA+加权集成、正常门控后处理，以及WhyXrayCLIP与提示集成的零样本视觉‑语言匹配；

**📊 数据集**

采用PadChest（约16万张）及PadChest‑GR注释数据，划分30个ID与6个OOD类别；

**📈 对比分析**

在CXR‑LT 2026公开开发集上以macro‑mAP评估，Task 1取得0.583 mAP（榜首），Task 2取得0.467 mAP（同样排名第一）；

**⚠️ 局限性**

仅在公开开发集验证，未公开测试集；校准（mECE）较差，且对多中心/不同采集设置的泛化尚待验证。

---

## 179. Scaling Beyond Masked Diffusion Language Models

**arXiv ID:** 2602.15014 | [PDF](https://arxiv.org/pdf/2602.15014v1)

**作者:** Subham Sekhar Sahoo `[一作]` (Cornell Tech), Ante Jukic `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对三类离散扩散语言模型（Masked Diffusion、Uniform‑state Diffusion、Interpolating Diffusion）与自回归模型进行规模匹配的 IsoFLOP 研究，探究其学习曲线、计算效率、采样速度与质量平衡，并在 1.7B 参数规模下进行零样本与微调评估。

**💡 创新点**

①首次在三种扩散框架下做规模匹配的 scaling law 研究；②发现跨家族的 perplexity 评估会误导模型选择；③低方差训练目标能显著提升 MDLM 的计算效率；④Uniform‑state Diffusion（Duo）在数学推理任务上可超越自回归模型。

**🔧 技术方法**

使用 Diffusion Transformer 结构、旋转位置编码、时间条件的自适应层归一化；Masked Diffusion 采用低方差 NELBO 目标；Uniform‑state Diffusion 采用自校正正向过程；Interpolating Diffusion 结合 KV 缓存的块采样；采样方式包括先祖采样与块采样；对比分析通过计算 FLOPs、模型参数、吞吐量、生成 perplexity（Gen‑PPL）与序列熵完成。

**📊 数据集**

主要训练数据为 SlimPajama（大规模通用文本），使用 Llama‑2 分词器，固定上下文长度 2048；微调阶段使用 GSM8K 及其 GPT‑4 生成的扩增数据（约 385K 条）。

**📈 对比分析**

方法对比包括：①IsoFLOP 曲线与计算最优模型大小/损失；②速度‑质量 Pareto 前沿（吞吐量 vs Gen‑PPL）；③在标准常识推理基准（ARC‑e、BoolQ、PIQA、SIQA、OBQA、RACE）中的 zero‑shot 结果；④在 GSM8K 上的微调准确率与吞吐量。结果显示：自回归模型在 likelihood 与常识基准上占优；Duo 在 GSM8K 微调后准确率最高；在速度‑质量平衡上，MDLM 与 Duo 在低吞吐量时更快，而 Eso‑LM 在中等吞吐量区间表现最佳。

**⚠️ 局限性**

①常识基准与数学推理任务受限，未覆盖更大规模或更复杂任务；②跨家族 perplexity 对比仍存在解释难度；③低方差训练提升有限，仍需进一步优化采样算法；④研究只评估到 1.7B 参数规模，缺乏更大规模的验证；⑤扩散模型在推理时仍受限于采样步骤与内存占用。

---

## 180. Spectral Convolution on Orbifolds for Geometric Deep Learning

**arXiv ID:** 2602.14997 | [PDF](https://arxiv.org/pdf/2602.14997v1)

**作者:** Tim Mangliers `[一作]` (Reutlingen University), Benjamin Himpel `[通讯]` (Reutlingen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了在轨道空间（orbifold）上定义谱卷积的理论，并通过对音乐和弦空间（dyad orbifold 𝒞²₁₂）的周期性函数进行卷积平滑，展示了该方法在非欧几里得几何上的可行性。

**💡 创新点**

创新点在于将谱卷积从流形推广到轨道空间，并通过拉普拉斯算子固有的谱分解直接得到在轨道空间上的傅里叶基，保证了卷积操作与空间的对称性与分辨率保持一致；同时提出了对音乐和弦空间的低通滤波器设计，说明了谱平滑在感知层面的意义。

**🔧 技术方法**

使用的技术包括：轨道空间的全局商表示 X = M/G，Riemannian 轨道空间的拉普拉斯算子及其谱分解，定义在 L²(X) 上的单位ary Fourier 变换，谱卷积 f * g = ℱ⁻¹(ℱ(f)⊙ℱ(g))，以及针对 𝒞²₁₂ 的对称化基函数与低通滤波器 gₙ 的构造。

**📊 数据集**

主要使用的“数据集”是音乐理论中的二音和弦空间 𝒞²₁₂（即以 12 分音阶为基底的轨道空间），以及由此产生的周期性函数 P⁺_{JND}(d)。没有传统意义上的训练集，而是通过对该空间的函数进行数值计算与可视化来展示方法效果。

**📈 对比分析**

在实验部分通过可视化比较了原始周期性函数、对称化后函数以及卷积平滑后的结果，说明了低通滤波对噪声与尖锐跳变的抑制效果；由于本工作聚焦于理论验证，未给出客观指标或与其他方法的数值比较，但图示表明平滑后函数在保留全局结构的同时消除了不连续性。

**⚠️ 局限性**

局限性包括：1) 目前仅作为平滑工具实现，未扩展为可训练的深度网络模块；2) 只在二维轨道空间（dyad）上演示，缺乏对更高维轨道空间或实际大规模数据集的验证；3) 依赖于轨道空间的解析拉普拉斯谱，对非解析或离散化轨道结构的适用性尚未探索；4) 没有针对性能（如训练速度、推理时间）进行量化评估。

---

## 181. Before the Vicious Cycle Starts: Preventing Burnout Across SOC Roles Through Flow-Aligned Design

**arXiv ID:** 2602.14598 | [PDF](https://arxiv.org/pdf/2602.14598v1)

**作者:** Kashyap Thimmaraju `[一作]` (Technische Universitat Berlin), Gail-Joon Ahn `[通讯]` (Arizona State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统分析了106个公开SOC岗位招聘信息，提炼了所需技能、证书、工具等要求；

**💡 创新点**

首次以流理论视角量化SOC岗位需求，构建了公开的编码代码本与基线数据；

**🔧 技术方法**

采用归纳内容分析（Inductive Content Analysis）并使用MAXQDA软件进行手工与自动化编码；

**📊 数据集**

使用了2024年11-12月收集的106条公开SOC招聘广告，涵盖35家公司、11个国家；

**📈 对比分析**

通过频次统计与占比对比技术与软技能、证书等类别，发现沟通能力占比最高、技术需求相对稀缺；该方法仅为描述性统计，无进一步性能评估；

**⚠️ 局限性**

研究限制包括单研究者编码缺乏交叉验证、经验描述模糊导致难以量化、JD所述与实际招聘实践可能不一致，缺乏后续验证实验。

---

## 182. The More the Merrier: Running Multiple Neuromorphic Components On-Chip for Robotic Control

**arXiv ID:** 2602.13747 | [PDF](https://arxiv.org/pdf/2602.13747v1)

**作者:** Evan Eames `[一作]` (Ludwig Maximilian University of Munich), Axel von Arnim `[通讯]` (Fortiss)

**通讯引用:** 242 | [OpenAlex ID](https://openalex.org/A5044456500)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文实现了一个完整的基于 Loihi 2 的神经形态机器人推拔插槽的端到端管线，集成了视觉感知、注意力、记忆、分类、视动反馈与插拔控制六大模块，并通过神经状态机实现了无离线逻辑的过程调度。

**💡 创新点**

创新点在于首次展示多组件神经形态网络能够完全在芯片上运行，利用动态神经场实现自适应注意力与记忆、关系网络实现局部遮蔽，并通过神经状态机实现模块间协作，全部实现无传统离线处理。

**🔧 技术方法**

主要技术包括突触式神经网络、动态神经场、关系网络、神经状态机、LIF 神经元、LAVA 框架、Intel Loihi 2 芯片、事件驱动 DVS 摄像头、MuJoCo 物理仿真、SLAYER 训练方法。

**📊 数据集**

使用了实验室自采集的 DVS 事件数据（包含四个插槽的视觉场景），并在模拟环境中对 MuJoCo 机器人进行仿真；未使用公开大规模数据集。

**📈 对比分析**

在 Loihi 2 上的实测显示功率 0.88 W、平均延迟 88 ms、能耗 78 mJ；与 GPU 实现相比功耗低数十瓦，延迟与边缘设备相当；在仿真和真实机器人上分别取得 90%/100% 的分类准确率、68%/100% 的视觉伺服成功率、70% 的插拔成功率。

**⚠️ 局限性**

主要限制包括编译器分区随机性导致的时序漂移、需要手工调节的 104 个可调参数、对 DNF 峰值稳定性和事件密度的敏感性、关系网络遮蔽导致的边缘模糊、I/O 约束限制分辨率、缺乏完整的芯片‑机器人集成与自我调节机制。

---

## 183. Elo-Evolve: A Co-evolutionary Framework for Language Model Alignment

**arXiv ID:** 2602.13575 | [PDF](https://arxiv.org/pdf/2602.13575v1)

**作者:** Jing Zhao `[一作]` (Zuoyebang), Yang song `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Elo-Evolve框架，将LLM对齐从静态奖励优化转变为多智能体的动态竞争；

**💡 创新点**

创新点在于：①直接用二元胜负反馈替代传统Bradley‑Terry模型；②通过Elo评分实现自适应对手抽样，实现自动课程学习；

**🔧 技术方法**

采用二元竞争奖励的GRPO优化、Elo评级更新、温度控制的对手抽样和预缓存对手回复的技术；

**📊 数据集**

使用Ultra‑Feedback数据集进行训练，评估指标基于Alpaca Eval 2.0和MT‑Bench；

**📈 对比分析**

与传统点分奖励、DNO、静态对手训练等方法对比，Elo‑Evolve在Alpaca Eval WR/LC和MT‑Bench上均取得最高或第二高分，显示出更稳健的性能提升；

**⚠️ 局限性**

局限性包括对手池管理不完善导致训练后期性能波动、对温度参数敏感、对极高质量任务的梯度稀缺问题。

---

## 184. Advances in Global Solvers for 3D Vision

**arXiv ID:** 2602.14662 | [PDF](https://arxiv.org/pdf/2602.14662v1)

**作者:** Zhenjun Zhao `[一作]` (University of Zaragoza), Javier Civera `[通讯]` (University of Zaragoza)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了3D视觉中的全局优化求解器，提出统一的分支定界、凸松弛与分层非凸三大范式，并系统评估其在十类关键任务中的适用性与性能折衷，构建公开代码与教程资源；

**💡 创新点**

创新点在于首次将全球优化方法整体分类为B&B、CR、GNC三类，提供理论、算法、鲁棒性与可扩展性层面的深入对比，并明确未来的可扩展算法、深度学习融合与标准化评测方向；

**🔧 技术方法**

采用的技术包括分支定界的搜索与上下界构造、Shor的半正定松弛、Moment‑SOS 层次松弛、以及基于连续退火的 GNC 迭代重加权；

**📊 数据集**

作为综述论文并未依赖单一数据集，而是聚合了多任务实验案例（如 Wahba、姿态估计、三维配准、姿态图优化、束调整等）并提供 GitHub 上的可复现代码与数据；

**📈 对比分析**

通过理论分析与实验比较，发现 BnB 能提供严格全局最优保证但受指数复杂度限制；CR 在中等规模任务上能以可证的方式求解并在松弛紧致时获得全局最优；GNC 在大规模场景中展现出线性可扩展性与较高的鲁棒性，但缺乏正式最优性保证；总体上各方法的性能与适用场景呈现折衷关系；

**⚠️ 局限性**

局限性包括：对大规模、强噪声或极高离群点比例的任务仍缺乏既可证又高效的求解器；算法对硬件与软件实现依赖度高，易受内存与求解速度限制；缺乏统一的基准评测与深度学习模型的可靠融合框架；这些因素在实际部署中仍需进一步突破。

---

## 185. WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control

**arXiv ID:** 2602.14351 | [PDF](https://arxiv.org/pdf/2602.14351v1)

**作者:** Mehran Aghabozorgi `[一作]` (Simon Fraser University), Ke Li `[通讯]` (Simon Fraser University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于IMLE的世界模型WIMLE，并通过模型预测不确定性对合成转移加权，显著提升连续控制任务的样本效率和最终性能。

**💡 创新点**

创新点在于将IMLE用于学习多模态世界模型，并将模型的不确定性作为权重融入强化学习目标，从而降低模型误差累积和过度自信导致的偏差。

**🔧 技术方法**

使用的技术包括IMLE、模型集成(Ensemble)、SAC（分布式Q学习）、噪声采样与不确定性估计、以及基于方差的加权学习。

**📊 数据集**

实验数据集涵盖DeepMind Control Suite、MyoSuite和HumanoidBench共40个连续控制任务。

**📈 对比分析**

与SAC、PPO、BRO、Simba、SimbaV2、TD‑MPC2、DreamerV3等强基线比较，WIMLE在所有任务上样本效率提升至少50%，在HumanoidBench解决8/14任务，整体性能超过或等同于现有最优方法。

**⚠️ 局限性**

局限性包括仅利用合成rollout而未探索规划或模型集成到策略梯度中的方式；仅在位置观测下验证，缺乏对图像输入的适应；rollout长度是任务相关的超参数，未实现在线自适应；以及未评估对更复杂感知和规划场景的泛化能力。

---

## 186. Named Entity Recognition for Payment Data Using NLP

**arXiv ID:** 2602.14009 | [PDF](https://arxiv.org/pdf/2602.14009v1)

**作者:** Srikumar Nayak `[一作]` `[通讯]` (Institute of Electrical and Electronics Engineers), Srikumar Nayak (Institute of Electrical and Electronics Engineers)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估并提出了 PaymentBERT 模型，用于从多格式支付消息中精准识别实体，构建了包含 50,000 条注释交易的基准数据集。

**💡 创新点**

创新点在于将 BERT 预训练与支付领域嵌入、格式特征以及 CRF 解码相结合，显著提升跨格式泛化与边界识别能力。

**🔧 技术方法**

采用 BERT、FinBERT、BiLSTM‑CRF、CRF 以及自定义的 PaymentBERT 混合架构，并配合词向量、字符嵌入、支付专用嵌入和格式特征。

**📊 数据集**

使用人工标注的 50,000 条多格式支付消息数据集（SWIFT MT103、ISO 20022、ACH、SEPA 等），包含多语言、缩写和嵌套实体。

**📈 对比分析**

通过与规则基、spaCy、Stanford NER、CRF、BiLSTM‑CRF、BERT‑base、FinBERT、BERT‑CRF 等基线比较，PaymentBERT 在 F1‑score 上达 95.7%，显著优于其它模型，且在实时推理时保持可接受的 52 ms 延迟。

**⚠️ 局限性**

局限包括数据量仍不足以覆盖全部业务场景、主要语言为英文/欧洲语种、模型对新兴支付格式和术语的适应性有限，以及深度学习模型的可解释性和持续学习需求。

---

## 187. Impacts of Generative AI on Agile Teams' Productivity: A Multi-Case Longitudinal Study

**arXiv ID:** 2602.13766 | [PDF](https://arxiv.org/pdf/2602.13766v1)

**作者:** Rafael Tomaz `[一作]` (Pontifical Catholic University of Rio de Janeiro), Marcos Kalinowski `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对三支敏捷团队实施了为期13个月的多案例纵向研究，评估GitHub Copilot与内部GPT工具对团队生产力、质量和开发者体验的影响。

**💡 创新点**

首次在真实工业咨询环境中长期追踪GenAI工具的多维生产力效应，采用SPACE框架揭示绩效-活跃度-效率（P‑A‑E）分歧，展示价值密度提升而非工作量增加。

**🔧 技术方法**

使用Jira、SonarQube与Git API进行遥测采集；Python脚本实现自动化数据提取；Gemini v2.5 Flash进行质性文本合成；结合GQM、SPACE、统计检验（Mann‑Whitney U、Cohen d）进行分析。

**📊 数据集**

三支团队在13个月内的Jira史料、SonarQube缺陷报告、Git提交与代码行数，以及双周期问卷和访谈数据，涵盖油气与海运项目。

**📈 对比分析**

采用前后对比（历史 vs 研究期），通过非参数检验与效应量评估，发现故事点完成率提升59.1%，高危缺陷显著下降，开发者活跃度（提交次数/代码行数）保持不变，表明效率提升而非工作量增加。

**⚠️ 局限性**

样本仅三支团队、单一咨询公司，存在选择偏倚；纵向时间虽长但不足以评估长期技能流失或技术依赖；外部因素如项目成熟、工作压力等可能与干预效果混杂，限制因果推断和普适性。

---

## 188. Judging the Judges: Human Validation of Multi-LLM Evaluation for High-Quality K--12 Science Instructional Materials

**arXiv ID:** 2602.13243 | [PDF](https://arxiv.org/pdf/2602.13243v1)

**作者:** Peng He `[一作]` (Washington State University), Tingting Li `[通讯]` (Washington State University)

**通讯引用:** 11984 | [OpenAlex ID](https://openalex.org/A5100416461)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了人类专家如何验证多大语言模型对K–12科学教材的评估结果，并将专家意见转化为设计原则。

**💡 创新点**

创新点在于将专家对评分与理由的同意/不同意视为信息源，提出了对AI评估进行多模型、多视角的“分歧感知”设计。

**🔧 技术方法**

采用了GPT‑4o、Claude Sonnet 4、Gemini 2.5 Pro三种LLM，结合EQuIP rubric，使用系统/用户提示生成评分、理由与改进建议，并对人类评估进行对标。

**📊 数据集**

数据集为12个来自OpenSciEd、ML‑PBL等认证项目的高质量K–12科学课程单元，共648条评估输出。

**📈 对比分析**

比较方法为计算模型与两名人类评估者的百分比一致率、Cohen’s κ、Fleiss’ κ，以及模型间精确匹配；性能显示Gemini与GPT在评分/理由上的一致率最高（≈87%/≈85%），Claude在评分上显著低于两者。

**⚠️ 局限性**

限制在于样本量小、专家人数少、仅评估文本而非课堂实施，且未检验AI工具在教师决策中的实际影响。

---

## 189. Simulation-Based Study of AI-Assisted Channel Adaptation in UAV-Enabled Cellular Networks

**arXiv ID:** 2602.13199 | [PDF](https://arxiv.org/pdf/2602.13199v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 190. Testing For Distribution Shifts with Conditional Conformal Test Martingales

**arXiv ID:** 2602.13848 | [PDF](https://arxiv.org/pdf/2602.13848v1)

**作者:** Shalev Shaer `[一作]` (Technion IIT), Yaniv Romano `[通讯]` (Technion IIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于固定参考集的条件合成测试马丁格尔（Conditional Conformal Test Martingale）用于在线检测分布偏移。

**💡 创新点**

通过在固定参考集上估计分布并显式校正估计误差，解决了标准CTM的测试时污染问题，并首次提供了任意时刻类型I错误控制与渐近功效一致性的理论保证。

**🔧 技术方法**

使用合成测试与投注框架、马丁格尔理论、Dvoretzky–Kiefer–Wolfowitz置信区间、在线Newton步（ONS）算法学习投注参数，以及对绝对值的平滑处理。

**📊 数据集**

在合成高斯/AR(1)序列以及ImageNet-C真实图像失真数据集上进行实验。

**📈 对比分析**

与标准CTM及pairwise投注方法比较，实验表明在参考样本≥500时，条件CTM在多种变换下拥有更高功效、更快检测时间，并在多种失真场景中均优于基线。

**⚠️ 局限性**

当参考集过小导致置信区间宽幅时，方法过度保守，且对极小或渐进的分布偏移敏感度有限。

---

## 191. RGA-Net: A Vision Enhancement Framework for Robotic Surgical Systems Using Reciprocal Attention Mechanisms

**arXiv ID:** 2602.13726 | [PDF](https://arxiv.org/pdf/2602.13726v1)

**作者:** Quanjun Li `[一作]` (Guangdong University of Technology), Xuhang Chen `[通讯]` (Huizhou University)

**通讯引用:** 753 | [OpenAlex ID](https://openalex.org/A5036370695)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种用于机器人手术烟雾去除的深度学习框架RGA-Net，能在实时手术视频中恢复清晰可视化；

**💡 创新点**

创新点在于双流混合注意力（DHA）结合频域与窗口注意力、轴分解注意力（ADA）以及双向交叉门控（Cross‑Gating）三大模块；

**🔧 技术方法**

使用了层次化Encoder‑Decoder结构、Swin Transformer、频域卷积、块与网格注意力、以及交叉门控融合机制；

**📊 数据集**

在公开的DesmokeData（2952/512配对图像）和LSD3K（2800/200配对图像）两个数据集上进行训练与评测；

**📈 对比分析**

与DCP、AOD‑Net、FFA‑Net、MSBDN、DehazeFormer、PFAN等21种前沿方法对比，RGA‑Net在PSNR/SSIM等指标上均取得最高分，显示出显著的性能提升；

**⚠️ 局限性**

局限性包括：仍需更大规模真实手术数据进行验证，模型在极端烟雾密度下的鲁棒性待进一步提升，且推理速度与资源占用需优化以适配临床实时部署。

---

## 192. UAV Swarm Enabled Aerial Movable Antenna System for Low-Altitude Economy: From Far-Field to Near-Field Communication

**arXiv ID:** 2602.13687 | [PDF](https://arxiv.org/pdf/2602.13687v1)

**作者:** Haiquan Lu `[一作]` (Nanjing University of Science and Technology), Rui Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 104305 | [OpenAlex ID](https://openalex.org/A5100422102)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了利用UAV群协同实现的近场可移动天线（AMA）系统，并通过联合优化3D轨迹和接收波束来最大化多用户的最小平均通信速率。

**💡 创新点**

创新点包括：①首次将非均匀球面波（NUSW）模型用于近场AMA系统，考虑轨迹对幅度与相位的双重影响；②针对单UE、两UE给出了解析最优布置（单UE直接垂直、两UE对称布置于双曲线）并推导闭式解；③提出针对任意用户数的交替优化算法，先利用幅度优化获得初始轨迹，再交替更新波束和轨迹，显著提升性能。

**🔧 技术方法**

技术手段：使用凸优化与SCA（序列凸逼近）技术求解非凸问题；MMSE波束获得最优接收权重；二分搜索求解双曲线参数；交替优化框架结合CVX求解子问题；在单UE、两UE场景下提供解析解。

**📊 数据集**

实验数据：采用仿真数据，UE随机均匀分布在长宽160m×120m的矩形区域，参考距离1m处的信道增益β₀=-61.4 dB，噪声功率σ²=-94 dBm，UAV速度上限30 m/s，最小安全距离5 m；未使用真实通信数据集。

**📈 对比分析**

对比方法：圆形布置、随机布置、最大最小平均SNR轨迹；实验显示：单UE场景中，最优与近似递增布置比基线高≈20–30%；两UE场景中，通过双曲线布置实现零互用户干扰，信噪比几乎无损；多UE场景中，交替优化方案在不同功率和用户数下均显著优于基线，提升率可达30%甚至更高。

**⚠️ 局限性**

局限性：仅考虑LoS直射链，未包含多径或非LoS；假设每个UE只有单天线，实际多天线UE未考虑；算法复杂度仍随UAV数量和时隙数呈多项式增长；对偶数UAV的解析布置在两UE场景下适用，奇数UAV或多用户情况缺乏类似解析解。

---

## 193. Not Seeing the Whole Picture: Challenges and Opportunities in Using AI for Co-Making Physical DIY-AT for People with Visual Impairments

**arXiv ID:** 2602.13874 | [PDF](https://arxiv.org/pdf/2602.13874v1)

**作者:** Ben Kosa `[一作]` (University of Wisconsin-Madison), Liang He `[通讯]` (University of Texas at Dallas)

**通讯引用:** 1052 | [OpenAlex ID](https://openalex.org/A5100317921)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了基于大型语言模型（GPT‑4o）的AI助手A11yMaker AI，并将其与可触摸DIY‑AT工具包相结合，支持视障者通过语音交互协同创作物理辅助技术；

**💡 创新点**

首次将LLM与分布式可触摸模块融合，提供完整的语音辅助共创流程，并系统性研究AI在物理共创中的误差与空间/视觉支持需求；

**🔧 技术方法**

使用GPT‑4o LLM与OpenAI工具调用进行自然语言交互，ESP32‑S3微控制器和ArUco标记构建的多模态模块，语音与触觉反馈结合的硬件平台；

**📊 数据集**

未使用公开数据集，研究基于现场实验收集的九名视障者的需求和使用日志；

**📈 对比分析**

采用实验室思考实验与访谈进行质性评估，未与其他系统做定量对比；通过九位参与者共创14个DIY‑AT解决方案，发现AI辅助提升了可行性但仍存在幻觉与空间支持不足，未给出具体性能指标；

**⚠️ 局限性**

限制包括AI误判与幻觉、缺乏空间/视觉反馈导致的搭建误差、工具功能局限、实验规模小、仅在实验室环境中测试、缺乏对更大场景和可扩展性的验证。

---

## 194. Precedent-Informed Reasoning: Mitigating Overthinking in Large Reasoning Models via Test-Time Precedent Learning

**arXiv ID:** 2602.14451 | [PDF](https://arxiv.org/pdf/2602.14451v1)

**作者:** Qianyue Wang `[一作]`, Mingkui Tan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

暂无研究内容

**💡 创新点**

暂无创新点

**🔧 技术方法**

暂无技术使用

**📊 数据集**

暂无数据集

**📈 对比分析**

暂无方法比较或性能评估

**⚠️ 局限性**

暂无限制

---

## 195. Key Considerations for Domain Expert Involvement in LLM Design and Evaluation: An Ethnographic Study

**arXiv ID:** 2602.14357 | [PDF](https://arxiv.org/pdf/2602.14357v1)

**作者:** Annalisa Szymanski `[一作]` (University of Notre Dame), Ronald A. Metoyer `[通讯]` (University of Notre Dame)

**通讯引用:** 1302 | [OpenAlex ID](https://openalex.org/A5018871532)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过为期12周的人种学观察与访谈，分析了一个大学内部团队在构建教育型LLM聊天机器人过程中的设计与评估实践。

**💡 创新点**

创新点在于提出四种实践模式（数据收集变通、数据增强、协同评估标准、混合评估策略），并系统揭示专家参与的动机、挑战与权衡，推动对专业知识在LLM开发中的可持续、可解释性整合。

**🔧 技术方法**

技术手段包括：LLM-as-a-Judge评估框架、对话数据增强技术、文本模拟器收集专家对话、参与式工作坊协同制定评估标准，并通过自动化与人工评估的混合流程进行迭代改进。

**📊 数据集**

数据集主要由专家在模拟器中生成的约130条教学咨询对话组成，随后通过数据增强扩展至约1000条合成对话；这些对话被用作微调训练和评估的 gold‑standard。

**📈 对比分析**

比较方法为将专家评估、开发者评估与LLM- Judge自动评分进行对比；结果显示自动评分与专家评分在某些维度上高度相关，但仍存在不一致，整体模型表现“良好但仍需改进”，但未给出具体性能指标。

**⚠️ 局限性**

局限性包括：仅在单一团队、单一机构内进行观察，样本与时间窗口有限，未覆盖部署与长期使用阶段，缺乏跨领域与跨文化的可推广性。

---

## 196. Sublinear-Time Lower Bounds for Approximating Matching Size using Non-Adaptive Queries

**arXiv ID:** 2602.14326 | [PDF](https://arxiv.org/pdf/2602.14326v1)

**作者:** Vihan Shah `[一作]` (University of Waterloo), Vihan Shah `[通讯]` (University of Waterloo)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5001778238)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在非自适应邻接表查询模型下估计最大匹配大小的问题。首先证明：任何随机非自适应算法若要获得比 n^{1/3} 更好的近似比例（即 n^δ 近似），必须查询 Ω(n^{1+ε}) 条边；其次给出一种简单的随机算法，在 O(n log^2 n) 次查询下即可得到 √n（即 n^{1/2}）近似，进一步推广到树型非自适应查询模型并给出对应的下界。

**💡 创新点**

创新点在于：①首次给出非自适应查询模型下的强下界，证明自适应性是实现高质量近似的必要条件；②提供了相对接近下界的上界算法，展示了非自适应模型虽然受限但仍能取得有意义的近似；③将结果扩展到更一般的树形非自适应查询模型。

**🔧 技术方法**

主要技术包括：构造两种硬实例分布（Yes/No），利用耦合（matching）和随机置换技术保证观测图在两种分布下几乎相同；运用 Chernoff / Markov 计数与独立性分析来估计星形子图的出现概率；通过对查询的随机邻居模拟以及对多重图的简化，降低证明复杂度；最后通过对观测图结构的层层固定（blocks）实现匹配证明。

**📊 数据集**

本文主要使用合成的多重图作为实验/证明实例，构造特定的二分图分布来展示下界；未使用公开真实数据集，实验部分仅以理论分析为主。

**📈 对比分析**

与已知的自适应 2-近似算法（Behnezhad 2021）相比，非自适应模型在取得 n^{1/3} 近似时需要至少 Ω(n^{1+ε}) 次查询，远高于自适应模型；上界算法在 O(n log^2 n) 次查询下实现 √n 近似，已接近下界给出的 n^{1/2} 近似上限，证明了该算法的近似阶数最佳性。

**⚠️ 局限性**

局限性包括：①下界仅针对特定的硬实例分布，可能不适用于所有图；②上界算法仅在多重图和对称结构上验证，实际图中误差仍较大；③算法对输入规模的幂次假设敏感，且在极大匹配规模已知时效果不佳；④对非自适应模型的分析仅在邻接表查询框架内，未覆盖矩阵或流式模型。

---

## 197. Investigation for Relative Voice Impression Estimation

**arXiv ID:** 2602.14172 | [PDF](https://arxiv.org/pdf/2602.14172v1)

**作者:** Keinichi Fujita `[一作]`, Yusuke Ijima `[通讯]` (NTT)

**通讯引用:** 705 | [OpenAlex ID](https://openalex.org/A5068604686)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了相对语音印象估计（RIE）任务，预测同一说话者相同文本的两段语音在多维印象轴上的差异。

**💡 创新点**

首次系统评估了在相对任务中，经典声学特征、SSL表示与多模态大语言模型的表现，并发现SSL表示显著优于传统特征。

**🔧 技术方法**

使用openSMILE提取声学特征、HuBERT自监督表示并训练MLP、以及零样本GPT‑5和Gemini 2.5 Pro进行提示式评估。

**📊 数据集**

采用单一日本专业女声演员的语料，包含52种说话风格，共814对语音，采用10折交叉验证。

**📈 对比分析**

与传统特征回归模型相比，SSL模型在Pearson/CCC上普遍提高约0.2以上，部分维度超过0.7；MLLM性能低于两种声学方法，均未能达到满意水平。

**⚠️ 局限性**

实验仅使用单一说话者且未对MLLM进行微调，限制了模型的泛化和多说话者、跨性别评估的可行性。

---

## 198. Every Maintenance Has Its Exemplar: The Future of Software Maintenance through Migration

**arXiv ID:** 2602.14046 | [PDF](https://arxiv.org/pdf/2602.14046v1)

**作者:** Zirui Chen `[一作]` (State Key Laboratory of Blockchain and Data Security, Zhejiang University), Xiaohu Yang `[通讯]` (State Key Laboratory of Blockchain and Data Security, Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对迁移式维护（Migration-Based Maintenance, MBM）进行系统性综述，提出完整的生命周期（任务选择、源选择、源调整、结果评估）框架，并梳理每个阶段的研究现状、挑战与未来机会。

**💡 创新点**

创新点在于：①首次将 MBM 归纳为四阶段流程；②系统识别并细化每个阶段的关键挑战与研究方向；③以跨领域视角（性能、配置、文档、深度学习、区块链等）扩展 MBM 的适用范围；④构建从研究动机到实证评估的完整路线图。

**🔧 技术方法**

使用的技术主要是文献检索与分析方法（PICOC 方案、关键词过滤、手工筛选），以及对比传统维护技术（搜索、约束、模板、学习）与 MBM 的对比框架；并对现有 MBM 论文的核心方法与结果进行归纳和对比。

**📊 数据集**

数据集来源于 4 大数字图书馆（ACM, IEEE, Compendex, Web of Science），共检索 2,919 篇候选文献，最终通过筛选得到 38 篇与 MBM 相关的实证研究。论文中引用的具体 MBM 任务与数据集（如 Linux kernel 修补、移动应用测试、开源库 API 演化等）均被系统列举。

**📈 对比分析**

对比方法主要是对现有维护技术（搜索、约束、模板、学习）与 MBM 的核心机制、优势与局限进行表格化对比；论文还总结了 38 篇 MBM 论文的实验设置、指标（成功率、覆盖率、误报率等）与性能表现，揭示 MBM 在验证可靠性、可迁移性和效率方面的潜力与不足。

**⚠️ 局限性**

局限性包括：①缺乏统一、可复现的 MBM 基准与自动评估指标；②多数研究仍以手工验证为主，自动化验证难度大；③迁移成功率在复杂场景（跨语言、跨版本）仍偏低；④现有数据集多为单一任务或小规模项目，难以覆盖全部维护场景。

---

## 199. Context Shapes LLMs Retrieval-Augmented Fact-Checking Effectiveness

**arXiv ID:** 2602.14044 | [PDF](https://arxiv.org/pdf/2602.14044v1)

**作者:** Pietro Bernardelle `[一作]` (University of Queensland), Gianluca Demartini `[通讯]` (University of Queensland)

**通讯引用:** 4580 | [OpenAlex ID](https://openalex.org/A5052565959)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨大语言模型（LLM）在事实验证任务中，参数知识、上下文长度和证据位置对性能的影响，采用多种模型在三个公开数据集上进行对照实验。

**💡 创新点**

创新点在于系统性量化不同上下文长度与证据位置（靠前/靠后/中间）对验证准确率的作用，并比较不同模型在空间敏感性方面的差异，揭示证据靠近输入两端能显著提升准确率。

**🔧 技术方法**

使用多种开源LLM（LLaMA‑3.1 8B/70B、Qwen‑2.5‑7B、Qwen‑3 8B/32B）指令调优版本，构建参数知识基线、全证据提示、以及在填充文本中插入证据块的长上下文实验，评估准确率和F1。

**📊 数据集**

采用三个事实验证数据集：HOVER、FEVEROUS（缩减至4k样本）和ClimateFEVER，统一为二分类（支持/驳斥）。

**📈 对比分析**

通过对比参数知识、全证据、不同上下文长度（2k–16k）以及证据位置（0%–100%）的准确率，发现：①上下文越长准确率整体下降；②证据位于前端或后端时表现最好；③Qwen‑3‑32B在所有设置下表现最稳健，整体准确率最高，其他模型的性能随上下文与位置波动更大。

**⚠️ 局限性**

局限性包括：仅使用完美金标证据，未评估噪声或不相关证据的鲁棒性；缺乏对注意力分布的细粒度分析；可能存在数据泄漏风险；实验仅覆盖部分模型和参数规模，未涵盖所有LLM架构。

---

## 200. GTS: Inference-Time Scaling of Latent Reasoning with a Learnable Gaussian Thought Sampler

**arXiv ID:** 2602.14077 | [PDF](https://arxiv.org/pdf/2602.14077v1)

**作者:** Minghan Wang `[一作]` (Monash University), Gholamreza Haffari `[通讯]` (Monash University)

**通讯引用:** 13013 | [OpenAlex ID](https://openalex.org/A5081525024)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在连续隐层推理模型中如何通过学习可调的高斯噪声来实现推理时间尺度化（ITS）

**💡 创新点**

提出了Gaussian Thought Sampler (GTS)，将隐层探索转化为条件概率采样，并使用GRPO式策略优化进行学习

**🔧 技术方法**

使用条件高斯采样、重参数化、GRPO式策略梯度、KL 正则化以及密集奖励设计

**📊 数据集**

在 GSM8K 计数推理数据集上进行实验

**📈 对比分析**

与传统的 dropout 采样和无条件 Gaussian 噪声基线对比，GTS 在有限采样预算下取得更高的 pass@N，尤其在中大规模采样时表现更稳健

**⚠️ 局限性**

局限性包括仅在 GSM8K 上验证、采样策略仅为对角高斯、未提供理论分析，且在不同架构或开放式推理任务中的泛化未知

---

## 201. 3D Wi-Fi Signal Measurement in Realistic Digital Twin Testbed Environments Using Ray Tracing

**arXiv ID:** 2602.13340 | [PDF](https://arxiv.org/pdf/2602.13340v1)

**作者:** Mengyuan Wang `[一作]` (University of Ottawa), Abdulmotaleb El Saddik `[通讯]` (University of Ottawa)

**通讯引用:** 18421 | [OpenAlex ID](https://openalex.org/A5109797436)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一个基于数字孪生的室内 Wi‑Fi 信号测量系统，将 LiDAR 3D 重建、对象分割、ITU 标准材料赋值与 GPU 加速射线追踪相结合，生成路径级通道特征并与现场测量对比。

**💡 创新点**

首次将真实扫描、对象分割、ITU‑R 材料模型与可编程射线追踪融合，提供高保真、可交互的数字孪生测量平台，并在相同运行时长下显著优于商用 Wireless InSite。

**🔧 技术方法**

使用 LiDAR 点云扫描、ICP 对齐、Advancing Front 网格重建、ITU‑R P.2040‑3 电磁材料赋值、Mitsuba 3 GPU 加速射线追踪、路径级 CIR 生成与统计指标提取。

**📊 数据集**

利用 Ottawa 六层建筑的 Matterport Pro3 LiDAR 数据、与 iPhone LiDAR 对比验证、现场 21 位置 RSSI 测量点，以及多频段（2.437 GHz、5 GHz、6 GHz）实验。

**📈 对比分析**

与商用 Wireless InSite 在相同 15 s 运行时长下公平对比，评估路径增益、延迟、SINR 等指标；结果显示 LOS 场景路径增益提升 10–20 dB，延迟均值与 RMS 更紧凑，且与实测 RSSI 的相关系数 0.98，平均误差 <1 dB；统计检验 p<0.01。

**⚠️ 局限性**

对材料采用均匀化假设，缺乏多层表面细节；在强反射 NLOS 场景中干扰提升；频率范围受 ITU‑R 模型限制；需进一步加入更细粒度散射与分层材料模型以及与更先进射线追踪技术的对标。

---

## 202. ST-EVO: Towards Generative Spatio-Temporal Evolution of Multi-Agent Communication Topologies

**arXiv ID:** 2602.14681 | [PDF](https://arxiv.org/pdf/2602.14681v1)

**作者:** Xingjian Wu `[一作]` (East China Normal University), Bin Yang `[通讯]` (East China Normal University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了基于时空演进的多智能体系统（Spatio‑Temporal Evolving MAS），能够在对话过程中实时、查询感知地调度通信拓扑；

**💡 创新点**

首创时空双维度自演进 MAS；利用流匹配（Flow‑Matching）生成器构建轻量化调度器；通过熵/方差熵感知系统状态并实现经验自反馈；

**🔧 技术方法**

使用 LLM 作为智能体核心；构建时空图（Spatio‑Temporal Graph）并采用 GCN+MLP 的流匹配网络；利用熵/VarEntropy 评估不确定性；引入检索‑增强经验模块；采用策略梯度训练；

**📊 数据集**

在九个公开基准上评估：MMLU、GSM8K、MultiArith、SVAMP、AQuA、HumanEval、DS‑1000、HotpotQA、DDXPlus；

**📈 对比分析**

与单智能体、静态多智能体（Chain、Tree、Star、AutoGen 等）、空间自演进（GPTSwarm、G‑Designer、ARG‑Designer、SafeSieve）以及时间自演进（AFlow、SpecReason、STEER）进行对比；平均精度提升 5.65%–26.10%，显著降低 token 消耗（约 50%）并提升鲁棒性；

**⚠️ 局限性**

受限于少样本训练需求，缺乏零样本或数据稀缺场景下的泛化能力，未来需开发更强的零样本构造与调度机制。

---

## 203. Why is Normalization Preferred? A Worst-Case Complexity Theory for Stochastically Preconditioned SGD under Heavy-Tailed Noise

**arXiv ID:** 2602.13413 | [PDF](https://arxiv.org/pdf/2602.13413v1)

**作者:** Yuchen Fang `[一作]` (University of California), Javad Lavaei `[通讯]` (University of California)

**通讯引用:** 6176 | [OpenAlex ID](https://openalex.org/A5042580848)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了在存在重尾噪声的随机预处理SGD（SPSGD）下，剪裁与归一化方法的最坏情况复杂度理论，并证明了归一化在此环境下保持收敛性而剪裁可能失效。

**💡 创新点**

首次揭示了在随机预处理环境下剪裁与归一化的根本差异，并给出了新的向量Burkholder不等式，提供了对重尾噪声下自适应优化器的理论解释。

**🔧 技术方法**

利用L-平滑、p-BCM条件、随机预处理矩阵条件数假设，结合向量Burkholder不等式和几何分析，推导出归一化SPSGD的收敛率；并通过构造反例说明剪裁失效。

**📊 数据集**

论文为理论研究，不涉及具体数据集；

**📈 对比分析**

不包含实验比较，主要通过理论证明和反例展示性能差异。

**⚠️ 局限性**

缺乏对实例平均情形的分析，对具体自适应预处理器的结构假设有限，未验证在实际大规模模型中的收敛速度。

---

## 204. Synthesizing the Kill Chain: A Zero-Shot Framework for Target Verification and Tactical Reasoning on the Edge

**arXiv ID:** 2602.13324 | [PDF](https://arxiv.org/pdf/2602.13324v1)

**作者:** Jesse Barkley `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 36736 | [OpenAlex ID](https://openalex.org/A5003442464)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在边缘设备上实现零样本目标验证与战术推理的层级框架，使用Grounding DINO做高召回的检测并将高置信度框传给小型VLM进行语义校验，完成从目标识别到损伤评估、细粒度分类及多步决策的完整“查-定-消”链；

**💡 创新点**

将高召回检测与小型VLM验证耦合的层级架构；提出“Controlled Input”方法将感知与推理分离以诊断失效模式；利用合成战场视频评估零样本模型在安全关键任务中的表现；

**🔧 技术方法**

Grounding DINO（Tiny）作为文本可调region proposal；Qwen3‑VL（4B、8B）和Gemma3（4B、12B）等4B–12B规模的开源Vision‑Language模型；4‑bit量化部署；Chain‑of‑Thought 推理和GPT‑4o作为评分器；

**📊 数据集**

使用Battlefield 6游戏引擎生成的55段10秒FPV合成视频，涵盖误报、被毁坦克、作战IFV、作战MBT四类场景；

**📈 对比分析**

与单阶段VLM推理、单模型评估等方法对比；在三项原子任务中，Qwen系列模型在误报过滤（100%）、损伤评估（97.5%）和细粒度分类（90%）方面均优于Gemma；在Scout–Commander多步推理中，Qwen模型实现100%任务成功和近满分的推理评分；Gemma模型在感知或推理上出现显著失效；

**⚠️ 局限性**

对域特定任务的泛化仍有限；Gemma3‑12B虽推理良好但感知缺陷，Gemma3‑4B推理崩溃；模型在真实硬件与环境下的性能与模拟差距未知；需要更完善的安全认证和对抗鲁棒性评估。

---

## 205. Contrastive explanations of BDI agents

**arXiv ID:** 2602.13323 | [PDF](https://arxiv.org/pdf/2602.13323v1)

**作者:** Michael Winikoff `[一作]` (Victoria University of Wellington), Michael Winikoff `[通讯]` (Victoria University of Wellington)

**通讯引用:** 5196 | [OpenAlex ID](https://openalex.org/A5008378418)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文为BDI代理生成对比解释，扩展先前的XAI工作，提供对比解释机制并进行计算与人类评估。

**💡 创新点**

创新点在于设计了对比解释生成算法，利用BDI目标-计划树进行对比过滤；同时证明对比解释在长度和信任方面优于完整解释。

**🔧 技术方法**

使用BDI框架、目标-计划树、解释因子（欲望、信念、价值），实现对比过滤逻辑；生成随机树和轨迹的程序；人类实验使用问卷和信任量表。

**📊 数据集**

数据集：随机生成1000棵满足BDI约束的树；对两条真实系统情景（煎饼机器人、UAV救援）进行实验。

**📈 对比分析**

对比方法：将完整解释与对比解释长度、信任、理解等指标对比；计算上显示对比解释平均长度约为完整解释的一半，信任得分显著提升；人类实验中部分情景对比解释被偏好。

**⚠️ 局限性**

局限性：对比解释对foil的依赖导致情景依赖性；仅测试两种简单系统，样本量有限；完整解释有时会提高信任，说明解释长度与质量关系复杂；未验证对非BDI代理的适用性。

---

## 206. Agentic AI for Commercial Insurance Underwriting with Adversarial Self-Critique

**arXiv ID:** 2602.13213 | [PDF](https://arxiv.org/pdf/2602.13213v1)

**作者:** Joyjit Roy `[一作]`, Samaresh Kumar Singh `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种带有对抗式自我批判的代理式人机交互框架，用于商业保险承保决策；

**💡 创新点**

创新点在于将内部批判代理嵌入决策循环以实现安全性和合规性，并构建了面向决策负向代理的失败模式分类法；

**🔧 技术方法**

主要技术包括Claude Sonnet 4.5大型语言模型、检索增强生成、链式思考与对抗式批判、基于状态机的工作流控制和严格的权威边界；

**📊 数据集**

使用了Snorkel AI多轮保险承保数据集（约1000个专家验证的案例），其中挑选了500个按复杂度分层的案例进行评估；

**📈 对比分析**

通过与人工手工、无批判LLM代理三种配置对比，实验显示加入自我批判后决策准确率从92%提升至96%，错误率（假阳性）降低72%，幻觉率从11.3%降至3.8%，处理时间提升4‑6倍但略高于单代理；

**⚠️ 局限性**

局限性包括仅在受控实验环境下验证，缺乏真实生产数据的多样性，模型对新规则或罕见案例依赖检索，批判层产生误报（5%），以及对大规模并发时延和成本的影响仍待评估。

---

## 207. Assessing Large Language Models for Medical QA: Zero-Shot and LLM-as-a-Judge Evaluation

**arXiv ID:** 2602.14564 | [PDF](https://arxiv.org/pdf/2602.14564v1)

**作者:** Shefayat E Shams Adib `[一作]` (Islamic University of Technology), Tareque Mohmud Chowdhury `[通讯]` (Islamic University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过零样本方式评估了五种最新 LLM（Llama‑3‑8B‑Instruct、Llama 3.2 3B、Llama 3.3 70B Instruct、Llama‑4 Maverick 17B‑128E Instruct、GPT‑5‑mini）在 iCliniq 医学问答数据集上的表现，并比较了 BLEU、ROUGE 与基于 Claude Sonnet 4 的多维度 LLM‑as‑a‑Judge 评估；

**💡 创新点**

其创新点在于将传统 n‑gram 指标与专门为医学 QA 设计的五维度评估框架相结合，形成更全面、可靠的性能评测体系；

**🔧 技术方法**

使用技术包括零样本生成、标准化提示模板、BLEU/ROUGE 自动评估，以及基于 Claude Sonnet 4 的医学准确性、完整性、安全性、清晰度和实用性五维度 Likert 评分并加权计算整体分；

**📊 数据集**

实验数据集为 iCliniq 38,000 条真实医学问答，随机抽取 3,000 条问答对进行评测；

**📈 对比分析**

通过相同提示对五模型进行零样本生成，结果显示 Llama 3.3 70B 在 BLEU‑1/ROUGE‑1/整体评分上遥遥领先，Llama‑4 Maverick 17B 在参数大幅减少的情况下仍能保持接近其性能，而 GPT‑5‑mini 则表现最差；

**⚠️ 局限性**

限制在于仅评估零样本能力，未涉及 fine‑tuning；评估仍受 n‑gram 指标限制；模型配置差异可能影响结果；数据集仅覆盖问答，未考察多轮对话或长期推理。

---

## 208. Behavioral Feature Boosting via Substitute Relationships for E-commerce Search

**arXiv ID:** 2602.14502 | [PDF](https://arxiv.org/pdf/2602.14502v1)

**作者:** Chaosheng Dong `[一作]` (Amazon), Yi Sun `[通讯]` (Amazon)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了通过替代品行为特征提升（BFS）方法，以利用已存在商品的行为信号解决新商品冷启动问题。

**💡 创新点**

创新点在于将替代品的销量速度等行为特征聚合为“温启动”信号，直接补充新商品缺乏的历史交互数据。

**🔧 技术方法**

采用多模态产品嵌入+行为注意力Transformer构建替代品识别模型，聚合方法包括均值、最大值及注意力加权等。

**📊 数据集**

在亚马逊海量搜索与购买日志（约数百万交互、数十万商品）上进行实验，构建每月一周期的历史数据集。

**📈 对比分析**

与原始LambdaMART排序器对比，BFS模型在离线 NDCG@10、GMV、销量和点击率上分别提升约0.5%~0.6%，在上线四周 A/B 测试中 GMV+0.11%、销量+0.22%、新商品 GMV+0.18%、新商品销量+0.35%。

**⚠️ 局限性**

局限在于替代品识别依赖预先训练的多模态模型，聚合策略仍相对简单；对动态更新的替代关系和多维行为特征的扩展还有待进一步研究。

---

## 209. HybridFlow: A Two-Step Generative Policy for Robotic Manipulation

**arXiv ID:** 2602.13718 | [PDF](https://arxiv.org/pdf/2602.13718v1)

**作者:** Zhenchen Dong `[一作]` (Polytechnic University), Yide Liu `[通讯]` (Anker)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在机器人操纵任务中提出HybridFlow，一种三阶段、两网络前向传播的低延迟生成策略，利用MeanFlow实现全局快速跳跃，ReNoise调节分布，ReFlow完成局部精细修正。

**💡 创新点**

将MeanFlow与ReFlow的数学关系统一到同一模型，设计无需蒸馏的两步推理，解决单步MeanFlow精度不足和多步误差累积的问题。

**🔧 技术方法**

使用流匹配方法（MeanFlow、ReFlow），视觉变压器编码器（ViT/DINOv3），线性插值ReNoise，OOP/ODE训练；在模拟与真实机器人上实现。

**📊 数据集**

RoboMimic 基准（Can、Lift、Square、Transport）以及真实机器人数据：多彩物体抓取（Black/Orange训练，Pink OOD）和眼罩折叠。

**📈 对比分析**

与16步DDIM Diffusion Policy、ReFlow、Shortcut Flow比较，HybridFlow在两步推理下实现了15–25%成功率提升，推理时间从152ms降至19ms（≈8×加速，≈52Hz）。

**⚠️ 局限性**

需要手动调节ReNoise比例α，仍在高精度触碰转移时易失误，且对动态扰动的在线重规划尚未集成。

---

## 210. LLMStructBench: Benchmarking Large Language Model Structured Data Extraction

**arXiv ID:** 2602.14743 | [PDF](https://arxiv.org/pdf/2602.14743v1)

**作者:** Sönke Tenckhoff `[一作]` (University of Applied Sciences Berlin), Erik Rodner `[通讯]` (University of Applied Sciences Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了LLMStructBench，一套针对大语言模型（LLM）在自然语言文本中提取结构化数据并生成符合 JSON Schema 的输出的基准测试框架；

**💡 创新点**

创新点在于构建了一个覆盖多种业务场景、含 995 个人工校验样本的公开数据集，并设计了细粒度的错误分类与两层评估指标（token‑level F1 与 document‑level validity），从而系统评估模型、规模与提示策略对结构化输出质量的影响；

**🔧 技术方法**

使用了多种技术：数据生成（GPT‑4o 生成文本+JSON，后手工校验）、五种提示策略（J、P、PJ、J+、PJ+）、微平均 F1 及 DOC_micro 组合分数、以及针对不同错误类型（mk、mv、wv）的加权评分；

**📊 数据集**

使用了自行构造的 LLMStructBench 数据集，包含 5 个用例（支持工单、病假、项目延期、会议注册、设备借用）和对应的 JSON Schema，全部样本均通过人工验证；

**📈 对比分析**

通过对 22 款开源模型（参数 0.6B–70B）和 GPT‑4o 的比较，实验发现提示策略比模型规模更关键；在 P 方案下，Gemma3‑27B、Llama3‑70B 等模型在 F1_micro 与 DOC_micro 上位居前列；PJ+ 方案虽保证结构完整但增加语义错误；总体上模型尺寸提升能改善 F1_micro，DOC_micro 更受提示与模型兼容性的影响；

**⚠️ 局限性**

局限性包括：①数据集仍主要基于 GPT‑4o 生成的文本，缺少真实业务日志的多样性；②评估侧重 JSON 结构，未覆盖更复杂的非树状结构；③对模型后处理与验证步骤未做系统化实验，导致仍有高比例 wv 错误；④对大型闭源模型的对比有限，仅作为参考。

---

## 211. Integrating Unstructured Text into Causal Inference: Empirical Evidence from Real Data

**arXiv ID:** 2602.14274 | [PDF](https://arxiv.org/pdf/2602.14274v1)

**作者:** Boning Zhou `[一作]` (Amazon), Haoqi Hu `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用基于 Transformer 的大型语言模型（Mistral‑3B）对产品描述文本进行多任务回归，推断潜在结果和倾向得分，随后通过 T‑learner、双重鲁棒估计和交叉拟合实现对反假冒活动的因果效应（ATE、ATT、GATE、CATE）的估计，并将文本数据得到的因果效应与传统结构化数值数据得到的效应进行对比。

**💡 创新点**

证明在真实业务数据上，单一文本输入即可近似得到与结构化数值输入相同的因果推断结果；提出一种同时预测两个潜在结果和倾向得分的多输出 decoder‑only 模型框架，并结合交叉拟合和 OLS 调整提升 CATE 估计的稳健性。

**🔧 技术方法**

Transformer‑based decoder‑only LLM（Mistral‑3B）+ LoRA 微调；多任务损失（潜在结果回归 + 倾向分类）；T‑learner 双重鲁棒估计；交叉拟合与 OLS 调整 CATE；对比指标包括 ATE、ATT、GATE、CATE 的相关性、置信区间重叠及 lift 曲线面积。

**📊 数据集**

真实电商反假冒活动数据：约 430,000 个产品，其中 280,000 个接受干预，150,000 个作为对照；每个产品既有 150+ 维数值特征，也有从产品描述中提取的自然语言文本；目标变量为 0–1 之间的投诉率。

**📈 对比分析**

通过对比文本模型与数值模型得到的 ATE、ATT、GATE、CATE，发现置信区间高度重叠；GATE 的 Pearson 相关 0.92，Spearman 相关 0.78；CATE 的 Pearson 相关 0.60，Spearman 相关 0.68；文本 CATE 作为排序指标的 lift 曲线面积超过 84% 的最佳（数值）曲线面积，表明文本模型已实现高水平的因果推断与推荐效果。

**⚠️ 局限性**

仅针对单一二元处理；未验证连续或多处理场景；多模态联合训练未评估；模型对文本质量和偏差敏感；跨领域推广需要进一步验证。

---

## 212. Beyond Static Snapshots: Dynamic Modeling and Forecasting of Group-Level Value Evolution with Large Language Models

**arXiv ID:** 2602.14043 | [PDF](https://arxiv.org/pdf/2602.14043v1)

**作者:** Qiankun Pi `[一作]` (Wuhan University), Tieyun Qian `[通讯]` (Wuhan University)

**通讯引用:** 2847 | [OpenAlex ID](https://openalex.org/A5040759280)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了基于 World Values Survey 的多波段分组数据，结合 LLM 模拟动态社会价值演变，并引入事件驱动预测。

**💡 创新点**

首次将历史价值轨迹和社会重大事件融入 LLM 人类问卷响应建模，实现动态价值预测和事件影响解释。

**🔧 技术方法**

使用大语言模型（Qwen、GLM、Llama、Mistral、GPT‑4o‑mini 等）在自定义长序列提示下进行低秩适配微调，采用价值向量与事件向量的余弦相似匹配。

**📊 数据集**

基于 WVS 5/6/7 波的中美样本，按性别、年龄、教育、收入划分共 28/23 组，包含已出现和未出现的问题。

**📈 对比分析**

与基线 Vanilla、仅轨迹 VTP 以及事件增强 EAP 进行对比，EAP 在见过/未见问题上均提升约 20–30%，在美国组最高提升 26%，在中文组最高提升 22%，多模型实验验证了方法的稳健性。

**⚠️ 局限性**

仅捕捉事件与价值的相关性，未处理潜在混杂因子；事件库来源有限；模型对低频事件易产生噪声。

---

## 213. An Adaptive Model Selection Framework for Demand Forecasting under Horizon-Induced Degradation to Support Business Strategy and Operations

**arXiv ID:** 2602.13939 | [PDF](https://arxiv.org/pdf/2602.13939v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 214. SAILS: Segment Anything with Incrementally Learned Semantics for Task-Invariant and Training-Free Continual Learning

**arXiv ID:** 2602.14767 | [PDF](https://arxiv.org/pdf/2602.14767v1)

**作者:** Shishir Muralidhara `[一作]` (German Research Center for Artificial Intelligence), René Schuster `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个训练‑free 的类增量语义分割框架 SAILS，利用 Segment Anything Model (SAM) 进行零射击区域提取，并通过冻结的预训练特征与原型匹配实现语义关联，无需对模型进行任何参数更新。

**💡 创新点**

创新点包括：① 完全不需要增量训练，彻底消除灾难性遗忘；② 将 SAM 与冻结的特征提取器结合，实现任务无关、跨域的零射击分割；③ 引入选择性内部聚类（intra‑class clustering），为高变异类生成多原型，提升语义表达；④ 通过正向向后迁移（positive backward transfer）提升旧类性能；⑤ 在长任务序列中保持任务不变的性能。

**🔧 技术方法**

使用的技术：Segment Anything Model (SAM) 用于生成对象级掩码；冻结的 Swin‑B（或类似）预训练网络提取特征；余弦相似度与阈值进行原型匹配；对高变异类采用 k‑means 进行内部聚类生成多原型；不使用任何反向传播或参数更新。

**📊 数据集**

使用的数据集：PASCAL VOC 2012（21 类）和 Cityscapes（19 类），在标准增量拆分（如 1‑1、2‑1、5‑1、2‑2 等）下进行评估。

**📈 对比分析**

对比方法：Joint Training（上限）、IL T、MiB、PLOP、SSUL、SATS、SAM+CLIP 等 SOTA CISS 方法。实验结果显示：在 VOC 上 mIoU 约 54–58，明显优于大多数增量方法且仅比 Joint Training 略低；在 Cityscapes 上 mIoU 约 36–45，亦优于同类方法。SAILS 在长任务序列中保持性能稳定，且无需更新，体现了更高的效率与更低的遗忘。

**⚠️ 局限性**

局限性：① 依赖冻结特征的表达能力，难以充分区分视觉相似或细粒度类；② 内部聚类虽提升多样性，但可能导致相似类混淆，影响精度；③ 虽然训练-free，但 SAM 的推理成本仍高，限制轻量级部署；④ 对极细粒度或新颖类的适应性不足，缺乏显式的自监督或适配机制。

---

## 215. Accuracy Standards for AI at Work vs. Personal Life: Evidence from an Online Survey

**arXiv ID:** 2602.13283 | [PDF](https://arxiv.org/pdf/2602.13283v1)

**作者:** Gaston Besanson `[一作]`, Federico Todeschini `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

调查了人们在专业和个人情境中对AI工具准确度的需求，并研究AI不可用时的适应性

**💡 创新点**

首次量化专业与个人对准确度的差异，揭示使用频率、信任度等因素对准确度取舍的影响

**🔧 技术方法**

采用在线问卷与统计分析技术（比例检验、McNemar检验、Logit回归等）

**📊 数据集**

使用 Prolific 平台收集的 300 名受访者数据，其中 170 人完成双项准确度问题

**📈 对比分析**

通过多种统计检验与阈值比较，发现专业情境下对准确度的要求显著更高，差异高度显著

**⚠️ 局限性**

局限在自报偏差、样本代表性有限、未进行实验验证准确度偏好、难以区分不同任务类型

---

## 216. D2-LoRA: A Synergistic Approach to Differential and Directional Low-Rank Adaptation

**arXiv ID:** 2602.14728 | [PDF](https://arxiv.org/pdf/2602.14728v1)

**作者:** Nozomu Fujisawa `[一作]` (Keio University), Masaaki Kondo `[通讯]` (Keio University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的参数高效微调框架D²-LoRA，利用带符号的低秩残差和训练时的列归一化投影，能够在低数据、低秩环境下实现高效微调，并保持推理阶段无额外延迟。

**💡 创新点**

创新点在于：①引入了正负分支的带符号低秩残差，既能放大又能抑制特征；②使用列归一化投影在训练期间强制保持每列原始范数，从而在低秩下提升稳定性和泛化；③在推理时将所有分支合并为单一矩阵，兼顾精度与速度。

**🔧 技术方法**

采用LoRA/DoRA的低秩残差结构，结合符号分支和列归一化投影；使用 AdamW + cosine 学习率、梯度累积等标准训练技术；在推理时实现权重合并。

**📊 数据集**

在八个QA/RC基准（ARC-Easy/Challenge、BoolQ、CommonsenseQA、HellaSwag、OpenBookQA、RACE、WinoGrande）以及两个生成任务（CNN/DailyMail 摘要和 AlpacaEval 2.0 指令跟随）上进行评估。

**📈 对比分析**

与LoRA、DoRA在相同参数预算（≈1.58M）下对比，D²-LoRA在 QA 任务上宏平均提升 2.2pp（相对LoRA）或 1.6pp（相对参数匹配LoRA），生成任务上亦有 1.2pp ROUGE-L 和 1.1% winrate 的提升；merge 等价性误差仅 0.03pp，推理速度提升约 1.9×。

**⚠️ 局限性**

局限性：仅在 QA/RC 和两类生成任务上验证；对其他任务（代码生成、多语言、算术推理等）未知；训练开销相对 LoRA 约 19%；对 τ 参数仍有一定敏感度，需在特定模型上微调。

---

## 217. The Speed-up Factor: A Quantitative Multi-Iteration Active Learning Performance Metric

**arXiv ID:** 2602.13359 | [PDF](https://arxiv.org/pdf/2602.13359v1)

**作者:** Hannes Kath `[一作]` (University of Oldenburg), Daniel Sonntag `[通讯]` (University of Oldenburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为 speed‑up factor 的多迭代主动学习性能评估指标，并在多种数据集与查询方法上进行了系统实验。

**💡 创新点**

创新点在于将学习曲线拟合与四个理论假设相结合，能够量化每个查询方法需要的样本比例，从而实现对主动学习过程的稳定、可解释评估。

**🔧 技术方法**

使用学习曲线近似（Weibull 与广义 Logistic 形式）与速度提升因子计算，并通过最小二乘法拟合参数 b_qm，进一步评估多迭代性能。

**📊 数据集**

实验涵盖四个多标签数据集（CARInA、MS COCO、Reuters‑21578、Scene），分别使用完整数据与 2k 子集，以不同模型（wav2vec‑2.0、ResNet‑50、BERT、FC 层）与训练设置进行验证。

**📈 对比分析**

与随机采样基线对比，speed‑up factor 在不同查询方法（ratio‑max、k‑means、BADGE、BALD、CRW、BEAL）和预算设置下显示出更高的稳定性，优于传统平均学习曲线、面积归一化以及切点等单迭代或多迭代指标。

**⚠️ 局限性**

局限性在于依赖四个假设，需单调、可平滑拟合的学习曲线；当学习曲线非单调、性能始终低或最终未达到饱和时，指标可能失效或不可靠。

---

## 218. 'I Spend All My Energy Preparing': Balancing AI Automation and Agency for Self-Regulated Learning in SmartFlash

**arXiv ID:** 2602.14431 | [PDF](https://arxiv.org/pdf/2602.14431v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 219. ForgeryVCR: Visual-Centric Reasoning via Efficient Forensic Tools in MLLMs for Image Forgery Detection and Localization

**arXiv ID:** 2602.14098 | [PDF](https://arxiv.org/pdf/2602.14098v1)

**作者:** Youqi Wang `[一作]` (Shenzhen University), Shouhong Ding `[通讯]` (Tencent Youtu Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ForgeryVCR框架，利用视觉中心推理（Visual‑Centric Reasoning）在多模态大型语言模型（MLLM）中嵌入取证工具，实现对图像伪造的检测与定位；

**💡 创新点**

创新点在于：①将伪造痕迹通过取证工具（如ELA、NPP、FFT）转化为可视化中间结果，消除文本推理的语义幻觉；②提出基于收益驱动的工具学习（Gain‑Driven Tool Selection）与策略优化（工具效用奖励的RL），让模型主动且高效地调用工具；③设计多轨迹合成与难样本剔除，提升训练数据质量与多样性；

**🔧 技术方法**

核心技术包括：多模态大型语言模型、视觉中心推理（Visual‑CoT）、混合取证工具箱（Zoom‑In、ELA、NPP、FFT）、自监督微调（SFT）+强化学习（GRPO）以及工具效用奖励机制；

**📊 数据集**

训练使用CASIA v2（SFT）及IMD2020、FantasticReality子集（RL）；评估数据涵盖CASIA v1、Columbia、Coverage、CocoGlide、DSO、Korus、In‑the‑wild、NIST16等多种伪造类型；

**📈 对比分析**

与传统专用取证网络（MVSS‑Net、IF‑OSN、TruFor、CoDE等）及取证版MLLM（FakeShield、SIDA）对比，ForgeryVCR在图像级检测的加权平均F1达0.8271、Accuracy 0.8261，定位的IoU 0.5306，均显著优于对手，尤其在覆盖率高、后处理强度大等挑战数据集上提升明显；

**⚠️ 局限性**

局限性包括：依赖预先设计的取证工具，可能无法覆盖所有未知或极端的伪造手法；工具调用仍需额外计算开销；模型对极低对比度或极小尺寸伪造的敏感性有待进一步提升；

---

## 220. Transferable XAI: Relating Understanding Across Domains with Explanation Transfer

**arXiv ID:** 2602.13675 | [PDF](https://arxiv.org/pdf/2602.13675v1)

**作者:** Fei Wang `[一作]` (National University of Singapore), Brian Y. Lim `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种“可转移XAI”框架，利用仿射变换（平移、缩放、映射）把一种AI解释迁移到相关域；

**💡 创新点**

创新点在于统一三类域迁移（子空间、任务、属性）为一套可解释的仿射框架，并通过稀疏正则降低认知负担；

**🔧 技术方法**

使用线性因子解释模型、仿射变换（矩阵乘法+平移）、稀疏正则、线性混合效应模型分析用户行为；

**📊 数据集**

实验数据集包括美国国家健康与营养调查（NHANES）用于健康风险预测（心脏病/糖尿病），以及北京多站点空气质量数据用于PM2.5/PM10预测；

**📈 对比分析**

与三种基线（无解释、原域解释、目标域解释）比较，基于用户决策准确度、权重倾向、因子回忆、关系理解等指标，Transferable XAI 在所有指标上均优于基线；

**⚠️ 局限性**

局限性包括：仅验证线性因子解释，映射矩阵对非技术用户可能过于复杂，且仅在两个应用场景验证，未来需扩展到更复杂模型和多维域迁移。

---

## 221. Two-Stream Interactive Joint Learning of Scene Parsing and Geometric Vision Tasks

**arXiv ID:** 2602.13588 | [PDF](https://arxiv.org/pdf/2602.13588v1)

**作者:** Guanfeng Tang `[一作]` (Tongji University), Rui Fan `[通讯]` (Tongji University)

**通讯引用:** 4029 | [OpenAlex ID](https://openalex.org/A5038867899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出TwInS两流交互联合学习框架，能够同时完成场景解析与几何视觉（立体匹配/光流）任务；

**💡 创新点**

创新点包括：双向交互的两流架构、跨任务适配器（CTA）将迭代匹配的隐藏状态投射到语义特征空间、基于不确定性的半监督伪标签生成与EMA教师-学生训练；

**🔧 技术方法**

使用Mask2Former语义/实例分割模块、ConvNeXt特征提取、GRU迭代匹配、CTA跨任务融合、不确定性估计及阈值筛选、EMA更新；

**📊 数据集**

在vKITTI2、Cityscapes、KITTI 2015等公开数据集上进行训练与评估；

**📈 对比分析**

与最新单模、特征融合及联合学习方法对比，在mIoU、mAP、EPE、D1等指标上均超越SoTA（如mIoU提升至约8%，EPE下降至0.96像素），展示显著性能提升；

**⚠️ 局限性**

限制包括：几何流失真时会对解析产生负面影响；半监督方案仍需分割标注，尚未实现完全无标注的自监督。

---

## 222. Introduction to Digital Twins for the Smart Grid

**arXiv ID:** 2602.14256 | [PDF](https://arxiv.org/pdf/2602.14256v1)

**作者:** Xiaoran Liu `[一作]` (McMaster University), Istvan David `[通讯]` (McMaster University)

**通讯引用:** 607 | [OpenAlex ID](https://openalex.org/A5041475393)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了数字孪生（Digital Twin）技术在智能电网（Smart Grid）全生命周期（设计、测试、运营、维护）中的应用与价值，并梳理了相关技术框架、标准、前沿趋势与挑战。

**💡 创新点**

创新点在于将数字孪生与智能电网结合的系统化视角：①提出数字孪生在四个生命周期阶段的具体角色与实践；②讨论AI、边缘计算、系统级数字孪生（System‑of‑Systems）等新兴技术如何赋能电网；③归纳行业标准（ISO 23247、RAMI 4.0、AAS、OPC UA 等）与实践需求的对接，为后续研究与应用提供可操作的参考。

**🔧 技术方法**

使用的技术与方法包括：
- 物联网传感器与SCADA采集的实时数据流；
- 云/边缘计算平台实现高频实时仿真与决策；
- 机器学习/AI（DNN、PINN、GPR、RL）作为模型替代与自适应优化手段；
- 通信协议与标准化框架（OPC UA、CIM、TSN、5G/LoRaWAN 等）保障数据互操作与低延迟；
- 参考架构与标准（ISO 23247、RAMI 4.0、Asset Administration Shell）用于设计与实现。

**📊 数据集**

本文未提供单一实验数据集，而是引用行业数据来源（SCADA、智能电表、传感器、资产维护记录等）以及已有案例与文献中的实验结果；若有实例，则多为论文中报道的实际电网案例或仿真实验数据。

**📈 对比分析**

由于为综述性工作，未进行实验对比；文章主要通过文献综述、案例分析和理论讨论说明数字孪生在电网中的优势与实现效果。若涉及案例，一般通过比较传统手段与数字孪生后实现的性能提升（如成本降低、可用性提高、预测精度提升等）来佐证，但未给出统一的性能指标或定量对比。

**⚠️ 局限性**

局限性与挑战包括：
- 缺乏统一、可互操作的标准与规范，导致跨平台、跨企业部署困难；
- 大规模实时数据采集与处理对网络带宽、时延与安全提出苛刻要求；
- AI/ML 模型的可解释性与验证/认证不足，难以满足安全关键应用；
- 资产生命周期数据缺失或不完整，影响模型准确性与预测可靠性；
- 边缘与云算力、能源消耗及其可持续性仍需进一步评估；
- 复杂系统的安全性与可维护性需要更完善的安全策略与监控机制。

---

## 223. Artificial Intelligence in Secondary Education: Educational Affordances and Constraints of ChatGPT-4o Use

**arXiv ID:** 2602.13717 | [PDF](https://arxiv.org/pdf/2602.13717v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 224. AGORA: Agentic Green Orchestration Architecture for Beyond 5G Networks

**arXiv ID:** 2602.13290 | [PDF](https://arxiv.org/pdf/2602.13290v1)

**作者:** Rodrigo Moreira `[一作]` (Federal University of Viçosa), Flavio De Oliveira Silva `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出 AGORA 架构，利用本地 LLM 与工具调用实现能耗驱动的移动网络控制，实时将自然语言可持续性意图转化为 UPF 路由动作。

**💡 创新点**

首次将 LLM 直接嵌入 5G/6G 控制循环，实现基于遥测的可验证 UPF 动作，并将能耗视为首要约束；同时构建多模型、跨语言的对比实验。

**🔧 技术方法**

本地 LLM（Qwen2.5 1.5B、Mistral 7B、Phi 3.5 Mini、OLMoE 1B/7B、Sabiá 7B）+ 工具调用接口、Kepler 监控、Prometheus、Chaos Mesh、free5GC、UPF、MEC、vLLM 服务器。

**📊 数据集**

使用基于 Chaos Mesh 注入的 CPU 负载与 Kepler 采集的能耗遥测、UE 端 RTT/UDP 采样，构成实验数据集；无公开真实流量数据。

**📈 对比分析**

通过多模型多次运行，对比能耗、推理延迟、token/能耗比、迁移概率、UE 延迟/抖动；结果显示 OLMoE 能耗最低但迁移率为零，Qwen 能耗与延迟兼优且能触发迁移，Mistral/Phi 能耗高、延迟长。

**⚠️ 局限性**

依赖模型对工具调用的准确性；实验规模仅两台 MEC、单一负载场景；缺乏真实多切片/多租户流量；迁移策略仅基于阈值，缺乏长期规划。

---

## 225. Ambient Physics: Training Neural PDE Solvers with Partial Observations

**arXiv ID:** 2602.13873 | [PDF](https://arxiv.org/pdf/2602.13873v1)

**作者:** Harris Abdul Majid `[一作]` (University of Edinburgh), Steven McDonagh `[通讯]` (University of Edinburgh)

**通讯引用:** 5815 | [OpenAlex ID](https://openalex.org/A5052824649)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 Ambient Physics 框架，用于仅依赖不完整观测学习偏微分方程系数与解的联合分布，并实现全域物理合理的重建。

**💡 创新点**

创新点在于：①通过随机掩蔽已有观测点并仍监督所有观测，从而让模型无法区分真实与人工缺失；②发现仅掩蔽一个已观测点即可实现“one‑point transition”，显著提升学习效果；③框架不依赖扩散模型，可迁移至传统神经 PDE 求解器。

**🔧 技术方法**

采用流匹配（rectified flow）作为生成模型，进行条件采样；并在传统 FNO/UNet 上实现 Ambient FNO/UNet；对抗性噪声、随机掩蔽和时间归一化用于训练。

**📊 数据集**

在四种 PDE（Darcy、Helmholtz、Navier‑Stokes、Poisson）上评估，使用 3% 随机采样的部分观测数据，比较完整观测训练与部分观测训练。

**📈 对比分析**

与 DeepONet、FNO、PINN、PINO 以及 DiffusionPDE/FunDPS 对比，Ambient Flow 在所有 PDE 上平均降低 62.51% 的相对 L2 错误，并且仅使用 125 倍更少的函数评估；Ambient FNO/UNet 也能在部分观测训练下取得竞争性能。

**⚠️ 局限性**

局限性包括：仍需足够的部分观测数据，掩蔽策略对性能敏感；在极其稀疏或结构化采样下可能出现细节失真；目前实验主要在合成 PDE 数据，真实场景中的噪声与不完整性仍待进一步验证。

---

## 226. Mean Flow Policy with Instantaneous Velocity Constraint for One-step Action Generation

**arXiv ID:** 2602.13810 | [PDF](https://arxiv.org/pdf/2602.13810v1)

**作者:** Guojian Zhan `[一作]` (Tsinghua University), Shengbo Eben Li `[通讯]` (Tsinghua University)

**通讯引用:** 19397 | [OpenAlex ID](https://openalex.org/A5100747108)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于平均速度场的流式策略（MVP），实现一次性高效动作生成；

**💡 创新点**

通过引入瞬时速度约束（IVC）作为边界条件，显著提升了平均速度场的学习精度与策略表达力；

**🔧 技术方法**

利用流匹配、JVP求导、最佳- N 采样等技术，构建了可训练的生成-选择框架；

**📊 数据集**

在Robomimic和OGBench两个稀疏奖励机器人操纵基准上进行评估；

**📈 对比分析**

与FQL、BFN、QC等最新离线到在线强化学习基线比较，MVP在9项任务中取得平均成功率0.88±0.05的最佳表现，并在训练与推理速度上显著优于对手；

**⚠️ 局限性**

训练时需要额外GPU显存，主要来自JVP运算，未来需优化显存占用。

---

## 227. From Snapshot Sensing to Persistent EM World Modeling: A Generative-Space Perspective for ISAC

**arXiv ID:** 2602.13554 | [PDF](https://arxiv.org/pdf/2602.13554v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 228. Verification of Robust Multi-Agent Systems

**arXiv ID:** 2602.13405 | [PDF](https://arxiv.org/pdf/2602.13405v1)

**作者:** Raphaël Berthon `[一作]` (RWTH Aachen University), Aniello Murano `[通讯]` (University of Naples Federico II)

**通讯引用:** 2459 | [OpenAlex ID](https://openalex.org/A5055415505)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究在不确定转移概率下，基于观察且有界记忆的策略在随机多智能体系统中的鲁棒性验证问题。

**💡 创新点**

创新点在于首次结合ATL/ PATL 与参数化/ε-扰动的马尔可夫决策过程，提出基于有限自动机的有界记忆策略，并给出了对三种不确定性模型（ε-扰动、固定参数、无穷参数）的可判定性与复杂度上界。

**🔧 技术方法**

主要技术包括：多智能体系统的形式化建模、参数化马尔可夫模型、ATL/PATL 语义与可观测策略的自动机表述、量化约简（如 Bellman 方程）以及实数理论中的 Π_k、 Σ_k 复杂度类。

**📊 数据集**

论文为理论性研究，没有使用具体实验数据集；验证结果以理论复杂度分析为主。

**📈 对比分析**

方法通过复杂度归约与证明，展示相较于传统完全信息或无记忆策略的情形，鲁棒性验证在 ε-扰动与固定参数下仍可在多项式或 Σ_2 级别完成，然而在无穷参数情形下复杂度提升至 Σ_3，表明鲁棒验证在一般情形下仍可决定但代价较高。

**⚠️ 局限性**

局限性包括：对鲁棒性验证的下界尚未得到明确证明；在无穷参数情况下复杂度接近极限，实际实现难度大；同时缺乏实验评估与性能基准，尚需进一步验证在实际智能体系统中的可行性。

---

## 229. MEMTS: Internalizing Domain Knowledge via Parameterized Memory for Retrieval-Free Domain Adaptation of Time Series Foundation Models

**arXiv ID:** 2602.13783 | [PDF](https://arxiv.org/pdf/2602.13783v1)

**作者:** Xiaoyun Yu `[一作]` (East China Normal University), Jilin Hu `[通讯]` (East China Normal University)

**通讯引用:** 1060 | [OpenAlex ID](https://openalex.org/A5020559625)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 MEMTS，一种可插拔的参数化记忆模块，用来在不更新基础时间序列基础模型（TSFM）参数的前提下实现域适配。

**💡 创新点**

创新点包括：①将域特定的季节性、趋势等动态内部化为少量可学习的潜在原型；②使用多分支未来可分离解码器和置换损失以避免平均化，捕获多模态未来；③自适应融合机制对记忆输出进行门控，既保持全局泛化又实现局部校正。

**🔧 技术方法**

技术实现基于 Transformer 编码器/解码器、潜在向量投影、置换匹配损失、门控残差融合，以及在训练阶段的离线知识编码和索引。

**📊 数据集**

在六个公开基准上评估：ECL、AustraliaRainfall、METR-LA、PEMS04、PEMS08、Solar，涵盖电力、气象、交通、能源等多域时序。

**📈 对比分析**

与 RAG（检索增强）、DAPT（全参数微调）以及多种主流 TSFM（Bolt-S/B, Moment, Moirai, Sundial）进行对比。结果显示 MEMTS 在所有模型、数据集和预测长度上均能提升 15–40% 的 MSE，并在 RAG 上实现 700× 的速度提升、≈1 ms 的固定延迟，且不增加显著存储成本。

**⚠️ 局限性**

局限性包括：记忆容量有限，可能对极其稀疏或高度非周期性的数据适配效果不佳；需预先完成离线知识编码，对实时领域更新不够灵活；在极长预测范围下，模型仍受限于原始 TSFM 的长期记忆能力。

---

## 230. It's a Matter of Time: Three Lessons on Long-Term Motion for Perception

**arXiv ID:** 2602.14705 | [PDF](https://arxiv.org/pdf/2602.14705v1)

**作者:** Willem Davison `[一作]` (University of Edinburgh), Laura Sevilla-Lara `[通讯]` (University of Edinburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对比稀疏长时序运动轨迹（MovT）与像素视频表示，探究其在动作识别、物体识别、情感识别、材质属性预测和空间理解等多任务下的表现与泛化能力。

**💡 创新点**

证明长时运动信息能够优于传统图像特征，展示更强的低数据与零样本泛化，并通过轻量级融合实现高精度与低计算成本的最佳权衡。

**🔧 技术方法**

使用CoTracker3点轨迹估计、Transformer编码器、VideoMAE的late fusion、t‑SNE可视化以及梯度重要性分析等技术。

**📊 数据集**

采用Something‑Something‑V2、Jester、VB100、RAVDESS、MITFabric、ADVIO等多种动作、物体、情感、材质与空间理解数据集。

**📈 对比分析**

通过准确率、RMS误差、GFLOPs等指标对MovT、PixT、VideoMAE及融合模型进行同构训练与评测；MovT在大多数任务上表现与甚至超过视频模型，并在低数据和零样本实验中保持更低的性能损失。

**⚠️ 局限性**

依赖高质量点轨迹估计，轨迹稀疏性可能限制对细粒度视觉细节的捕捉；对极长视频或复杂场景的轨迹完整性和实时性尚待进一步验证。

---

## 231. Towards Spatial Transcriptomics-driven Pathology Foundation Models

**arXiv ID:** 2602.14177 | [PDF](https://arxiv.org/pdf/2602.14177v1)

**作者:** Konstantin Hemker `[一作]` (Mass General Brigham), Faisal Mahmood `[通讯]` (Mass General Brigham)

**通讯引用:** 7366 | [OpenAlex ID](https://openalex.org/A5080050834)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了SEAL，一种利用空间转录组（ST）数据对现有病理影像基础模型进行参数高效微调的自监督学习框架；

**💡 创新点**

其创新点在于将ST信息与视觉表征通过对比与重建双目标函数进行对齐，并使用低秩适配器（LoRA）实现对多种ViT架构的通用、数据高效的微调；

**🔧 技术方法**

技术实现包括基于VAE+平面流的基因嵌入学习、InfoNCE对比学习、LoRA微调、梯度逆转域适配以及多任务联合损失；

**📊 数据集**

训练使用了约72万对H&E图像块与Visium ST点构成的MAPLE大规模数据集，评估覆盖38个切片级和15个图块级下游任务（分子状态、通路活性、IHC、预后等），并对跨模态检索和域泛化做了检验；

**📈 对比分析**

与传统vision‑only基础模型、ST‑Net、PathOmCLIP、OmiCLIP、BLEEP等方法相比，SEAL在切片级任务平均提升1–3%，在图块级基因预测任务中PCC提升约10–20%，且在不同扫描仪的批量效应上表现出更低的ARI/MI；

**⚠️ 局限性**

局限性包括依赖Visium 55 µm分辨率的ST数据，难以捕捉单细胞水平的细节；需要大规模且标注好的图像–ST配对；以及在未见数据分布或非人类样本中的泛化性尚待进一步验证。

---

## 232. Explainability-Inspired Layer-Wise Pruning of Deep Neural Networks for Efficient Object Detection

**arXiv ID:** 2602.14040 | [PDF](https://arxiv.org/pdf/2602.14040v1)

**作者:** Abhinav Shukla `[一作]` (Chhattisgarh Swami Vivekanand Technical University), Nachiket Tapas `[通讯]` (Chhattisgarh Swami Vivekanand Technical University)

**通讯引用:** 1068 | [OpenAlex ID](https://openalex.org/A5081938166)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种基于可解释性梯度-激活归因的层级剪枝框架，用于目标检测网络的结构压缩。

**💡 创新点**

创新点在于将 SHAP/DeepLIFT 启发的梯度‑激活交互作为层级重要性度量，避免传统幅度剪枝对功能贡献的误判。

**🔧 技术方法**

采用梯度‑激活归因、层级重要性排序、全局 5% 层级剪枝以及 PyTorch hooks 等技术实现框架。

**📊 数据集**

实验在 Microsoft COCO 2017 验证集上进行，覆盖多种检测器架构。

**📈 对比分析**

通过与 L1 范数幅度剪枝对比，归因剪枝在 MobileNetV2、ResNet‑50、ShuffleNetV2、Faster‑R‑CNN、RetinaNet、YOLOv8 等模型上往往保持或提升 mAP 并获得更高 FPS，例如 ShuffleNetV2 10% 加速、RetinaNet 无显著 mAP 降低。

**⚠️ 局限性**

局限性包括未进行剪枝后微调、剪枝率固定为 5% 以及仅针对层级剪枝，对更激进或动态剪枝以及跨任务的适用性尚未验证。

---

## 233. OR-Agent: Bridging Evolutionary Search and Structured Research for Automated Algorithm Discovery

**arXiv ID:** 2602.13769 | [PDF](https://arxiv.org/pdf/2602.13769v1)

**作者:** Qi Liu `[一作]` (Tongji University), Wanjing Ma `[通讯]` (Tongji University)

**通讯引用:** 4079 | [OpenAlex ID](https://openalex.org/A5038496118)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多智能体的自动科研框架 OR-Agent，能够在复杂实验环境中实现从假设生成、代码实现、实验验证到反思反馈的闭环式科学探索。

**💡 创新点**

创新点包括：① 将科研过程组织成树形工作流，实现对探索路径的显式管理与系统化回溯；② 引入层次化反思机制，将实验反馈映射为类似梯度、动量与权重衰减的语义优化；③ 结合进化式起点选择、结构化方案数据库与协同多智能体，以更稳健的方式平衡探索与利用。

**🔧 技术方法**

核心技术：大语言模型驱动的多智能体架构（Idea、Code、Experiment、Lead Agent 等），基于 MAP‑Elites 与岛屿模型的结构化方案数据库，树形研究流程与动态实验分配，分层反思与压缩策略，以及可配置的树深度、分支数与探索策略。

**📊 数据集**

使用 12 个经典组合优化基准（TSP、CVRP、BPP、MKP、OP 等）以及一个基于 SUMO 的协同驾驶仿真任务作为评估数据集。

**📈 对比分析**

与 FunSearch、AEL、EoH、ReEvo 等 LLM‑驱动的进化搜索基线进行对比，OR-Agent 在 12 个基准上平均归一化得分为 0.924，明显高于其它方法；在协同驾驶任务中，在相同算力预算下，得分提升至 48.00（远超 SUMO 默认模型 16.10），并在更大预算下实现 90.24 的优异表现。

**⚠️ 局限性**

局限性包括：① 对树深度、分支数、反思压缩等超参需要手动调优；② 计算成本高，尤其在深树与多实验时 LLM 调用频繁；③ 仅支持文本反馈，缺乏多模态感知；④ 仍需人工监督以确保任务描述与评价标准正确；⑤ 对非常大规模或非程序化可执行的科研任务适用性有限。

---

## 234. Measuring and Mitigating Post-hoc Rationalization in Reverse Chain-of-Thought Generation

**arXiv ID:** 2602.14469 | [PDF](https://arxiv.org/pdf/2602.14469v1)

**作者:** Guangyue Peng `[一作]` (Peking University), Houfeng Wang `[通讯]` (BOSS Zhipin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究逆向链式思考生成（RCG）中的后置合理化问题，并提出结构骨架引导推理（SSR）和其蒸馏版本SSR-D来降低锚定效应，提升推理质量与下游性能。

**💡 创新点**

创新点在于：① 构建三层锚定测量体系（词汇、熵、概率）量化后置合理化；② 发现语义抑制会因“讽刺过程”导致内在锚定升高；③ 设计SSR通过生成答案无关的功能骨架来解耦内容与结构，避免对答案的监控。

**🔧 技术方法**

技术手段包括：大语言模型生成逆向链式思考、信息熵与互信息分析、结构骨架（功能标签序列）生成与引导、教师模型蒸馏训练、对比实验与行为区块分析。

**📊 数据集**

使用 LMArena 生成的 10k/100k (Q, A) 对，参考答案来自 Qwen3-Max 自主推理；在多种公开推理基准（ArenaHard、EQ-Bench 3、IFEval、MultiChallenge）以及 OOD 任务（GPQA-Diamond、AIME 2025）进行评估。

**📈 对比分析**

与中性提示（NEU）、语义抑制（SUP、AUG‑SUP）以及SSR相比，SSR‑D 在所有基准上均实现最高分，提升幅度最高可达 +10 分；抑制方法虽降低词汇锚定，却显著提高熵与概率锚定，导致多轮推理性能下降；SSR 在 OOD 任务上也显著优于抑制方案。

**⚠️ 局限性**

局限性包括：① 结构骨架的设计仍需人工定义标签集，可能不适用于所有领域；② SSR‑D 依赖教师模型生成高质量骨架，蒸馏过程复杂；③ 在极端对抗性或高度专业化任务中，骨架可能无法完全捕捉必要的细粒度推理步骤。

---

## 235. Lang2Act: Fine-Grained Visual Reasoning through Self-Emergent Linguistic Toolchains

**arXiv ID:** 2602.13235 | [PDF](https://arxiv.org/pdf/2602.13235v1)

**作者:** Yuqi Xiong `[一作]` (Northeastern University), Ge Yu `[通讯]` (Northeastern University)

**通讯引用:** 6365 | [OpenAlex ID](https://openalex.org/A5072406974)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种利用自发语言工具链实现细粒度视觉推理的方法，使视觉语言模型在检索增强生成框架下能够主动感知并聚焦图像细节。

**💡 创新点**

核心创新在于用可自动生成的语言工具替代传统固定图像工具，并采用两阶段强化学习（先自探索生成工具，再利用工具进行推理），从而实现视觉感知与推理的统一。

**🔧 技术方法**

技术方案包含视觉语言模型（如Qwen2.5‑VL‑7B）、链式思考（CoT）提示、两阶段强化学习（DAPO）、检索增强生成（VRAG）以及语言工具抽取与执行机制。

**📊 数据集**

实验使用了SlideVQA、ViDoSeek、MMLongBench‑Doc等视觉问答基准，训练数据来源于OpenDocVQA和SlideVQA的样本。

**📈 对比分析**

在三大基准上与提示式模型、VLRM、MRAG以及传统工具增强模型对比，本文方法平均提升约4%（相较工具基线提升超过5%），显著提升问答准确率。

**⚠️ 局限性**

局限性主要体现在对注意力集中与答案正确性因果关系的解释不足，以及模型对工具抽取过程的鲁棒性和通用性仍有待提升。

---

## 236. Near-Optimal Best-of-Both-Worlds Fairness for Few Agents

**arXiv ID:** 2602.14668 | [PDF](https://arxiv.org/pdf/2602.14668v1)

**作者:** Moshe Babaioff `[一作]` (Hebrew University of Jerusalem), Gefen Frosh `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了针对少数（2、3）个加性代理人的公平分配算法，实现了最佳两界（Best‑of‑Both‑Worlds）公平性。

**💡 创新点**

创新点在于首次给三代理人构造了近最优的分配分布：满足期望比例、公平分布支持的所有分配均为EEFX并且每位代理人至少得到9/10 MMS；并且为两代理人提供了同时满足期望无恩怨、后验EFX与(1‑ε)MMS的多项式时间 FPTAS。

**🔧 技术方法**

核心技术包括：构造MMS‑EFX 分区、利用E‑EFX的可重排特性、对三代理人设计六个支持分配的组合方案、以及将两代理人 FPTAS 作为子程序改进三代理人算法；同时采用已有的 Santa‑Claus FPTAS 近似求解 MMS。

**📊 数据集**

论文基于理论分析，没有使用实测数据集，所有结果均为算法性质与可计算复杂度证明。

**📈 对比分析**

与先前仅实现1/2‑MMS或仅满足期望无恩怨的 BoBW 方案相比，本文的方案在三代理人情形下将 MMS 近似提升到 9/10，保持期望比例且后验满足 EE‑FX，且两代理人方案在多项式时间内实现 (1‑ε)MMS，性能显著优于之前的 12‑MMS 方案。

**⚠️ 局限性**

局限性包括：无法扩展到四及以上代理人（EFX 存在性未知）；在三代理人情形下无法同时保证期望无恩怨而非仅比例；多项式时间实现需依赖 MMS 近似，仍存在 ε 误差。

---

## 237. Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation

**arXiv ID:** 2602.14193 | [PDF](https://arxiv.org/pdf/2602.14193v1)

**作者:** Yue Chen `[一作]` (Peking University), Hao Dong `[通讯]` (Peking University)

**通讯引用:** 56374 | [OpenAlex ID](https://openalex.org/A5100425709)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 Part‑Aware 3D Feature Field (PA3FF) 及其对应的 Part‑Aware Diffusion Policy (PADP)，实现了对可动部件的稠密 3D 表征和可泛化的操控策略。

**💡 创新点**

创新点在于将基于 Sonata 的 3D 点云特征与对比学习（几何与语义）相结合，生成具有部件感知、连续且空间一致的 3D 特征场；并将此特征场作为 diffusion policy 的输入，大幅提升样本效率与跨对象/环境的泛化能力。

**🔧 技术方法**

核心技术包括 Sonata Point Transformer V3 预训练、几何对比损失（SupCon）与语义对比损失（InfoNCE）、轻量级特征细化 MLP、Denoising Diffusion Probabilistic Model (DDPM) 与 DDIM 推断、Transformer 编码器以及语义 CLS 触发聚合。

**📊 数据集**

使用了 PartNet‑Mobility、3DCoMPaT、PartObjaverse‑Tiny 进行对比学习训练，评估基于 PartInstruct 的 16 个仿真任务及 8 个真实任务（Franka Emika Panda + Intel RealSense 摄像头）来验证性能。

**📈 对比分析**

在 PartInstruct 基准上相比 CLIP、DINOv2、SigLIP、Grounded‑SAM 等 2D/3D 表征与 GenDP、Act3D 等 3D 方案提升了约 9.4% 的绝对成功率，真实任务中平均成功率达 58.75%，高出 35% 的最佳基线，显示出显著的样本效率与泛化优势。

**⚠️ 局限性**

限制方面：依赖点云输入，若传感器噪声或缺失导致点云质量下降，特征场可能受影响；需要大量带部件标签的 3D 数据进行对比学习；在极薄或细小部件的检测上仍可能不如 2D 细粒度特征，且对超大规模场景的实时推理仍面临计算成本挑战。

---

## 238. Formally Verifying and Explaining Sepsis Treatment Policies with COOL-MC

**arXiv ID:** 2602.14505 | [PDF](https://arxiv.org/pdf/2602.14505v1)

**作者:** Dennis Gross `[一作]` `[通讯]`, Dennis Gross

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

结合COOL-MC对ICU-Sepsis MDP进行正式验证与可解释性分析，训练安全RL策略并通过模型检查验证其生存概率及行为。

**💡 创新点**

在可验证模型上仅构造策略可达子空间，自动状态标注并结合PCTL查询与特征重要性分析，实现RL策略的形式化验证与解释。

**🔧 技术方法**

使用Probabilistic Model Checking（Storm/PRISM）、安全强化学习（PPO+后盾屏蔽）、特征修剪与特征置换重要性、以及COOL-MC框架。

**📊 数据集**

以ICU-Sepsis基准（约17,000例MIMIC-III血症患者的离散化MDP）为数据集。

**📈 对比分析**

与全MDP最优生存概率对比，学习策略实现0.8751的生存率，并通过PCTL与解释性方法揭示策略对先前给药历史的过度依赖，性能与理论最优一致。

**⚠️ 局限性**

仅支持离散动作与无记忆策略，离散化导致临床细节缺失，未验证在真实临床环境中的泛化能力。

---

## 239. Modeling and Optimizing the Provisioning of Exhaustible Capabilities for Simultaneous Task Allocation and Scheduling

**arXiv ID:** 2602.13866 | [PDF](https://arxiv.org/pdf/2602.13866v1)

**作者:** Jinwoo Park `[一作]` (Georgia Institute of Technology), Seth Hutchinson `[通讯]` (Northeastern University)

**通讯引用:** 17202 | [OpenAlex ID](https://openalex.org/A5071296935)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了名为 *traits 的离线异构多机器人任务分配与调度框架，能够处理可耗尽、可逐步提供的特征并考虑电池约束。

**💡 创新点**

创新点在于引入特征提供与耗尽概念、时间变化的特征、梯度提供速率以及累积/非累积特征分类，并结合电池消耗模型和 Peukert 定律，实现对任务持续时间和能耗的精准估计。

**🔧 技术方法**

主要技术包括非线性规划（NLP）进行特征分配、混合整数线性规划（MILP）进行调度、启发式最佳优先搜索任务分配、基于电流模型与 Peukert 定律的电池消耗建模，以及与运动规划模块的集成。

**📊 数据集**

实验使用仿真仓库环境生成 400 组随机场景，任务数 5~30，机器人数 5~25，采用 Clearpath Husky 规格计算电池系数。

**📈 对比分析**

与两种基准 *itags 和 *ctas 进行对比，结果显示 *traits 的方案可行性达到 100%，特征与速率不足、能源违规、截止约束违规均为 0%，且计算时间平均约 200 秒，显著优于基准。

**⚠️ 局限性**

局限性包括假设特征提供速率恒定、路径始终无碰撞、缺乏对不确定性的处理，以及仅适用于离线规划而不支持动态环境变化。

---

## 240. Building Autonomous GUI Navigation via Agentic-Q Estimation and Step-Wise Policy Optimization

**arXiv ID:** 2602.13653 | [PDF](https://arxiv.org/pdf/2602.13653v1)

**作者:** Yibo Wang `[一作]` (Nanjing University), Lijun Zhang `[通讯]` (Nanjing University)

**通讯引用:** 35277 | [OpenAlex ID](https://openalex.org/A5100448159)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出基于 agentic‑Q 估计与逐步策略优化的框架，用于训练多模态大语言模型驱动的 GUI 交互代理。

**💡 创新点**

创新点在于利用 agentic‑Q 为每一步生成即时奖励，解耦策略更新与非平稳环境，并采用自生成轨迹进行二分类训练，结合 critic‑free RL 实现稳定高效优化。

**🔧 技术方法**

采用多模态 LLM (Ovis2.5‑9B)、agentic‑Q 模型、滑动窗口与动作聚焦训练、GRPO/RLOO/REINFORCE++ 及 PPO 等强化学习技术。

**📊 数据集**

通过 WebVoyager、Online‑Mind2Web 与 ScreenSpot 三个真实网站基准收集轨迹，并在 Ovis2.5‑SFT 上进行专家导航训练。

**📈 对比分析**

在 GUI 导航与 grounding 任务中与同规模 Qwen3‑VL‑8B、UI‑TARS 1.5‑7B 以及更大 GPT‑4o、Claude 3.7/4 Sonnet 等模型对比，9B 代理获得了最先进的性能。

**⚠️ 局限性**

主要局限在于奖励连续值对优势归一化的敏感性导致潜在不稳定，以及对更复杂交互或动态网站的泛化能力尚未充分验证。

---

## 241. From What to How: Bridging User Requirements with Software Development Using Large Language Models

**arXiv ID:** 2602.13611 | [PDF](https://arxiv.org/pdf/2602.13611v1)

**作者:** Xiao He `[一作]` (University of Science and Technology Beijing), Jialun Cao `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 282 | [OpenAlex ID](https://openalex.org/A5053372458)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向软件设计的基准（benchmark），并通过30个Java项目评估LLM在设计感知代码生成、面向对象建模以及验收测试生成等任务中的表现。

**💡 创新点**

创新点在于：①设计感知的评估框架，将需求、设计模型与代码实现三层耦合；②提供从高层需求到低层代码骨架的多级输入，系统检验LLM对设计的理解与遵循；③细粒度度量指标（Pass@k、Compilation@k、类/方法匹配率等）并结合覆盖率评估测试生成质量。

**🔧 技术方法**

技术方法包括：使用LLM生成代码、模型和测试；利用PlantUML类图表示设计模型；采用自动化编译与单元测试评估生成代码；计算匹配率、准确率、召回率、F1分数以及覆盖率指标。

**📊 数据集**

数据集为30个手工构造的Java项目，包含域描述、功能需求、测试规范、域模型、设计模型、参考实现代码、参考测试用例以及代码骨架，共计30个设计模型、194个类和737个测试用例。

**📈 对比分析**

实验对7个主流LLM（DeepSeek系列、Qwen系列、GPT 3.5/4o-Mini）在三种任务中进行多组输入比较；结果显示：LLM在类识别和低级代码骨架支持下能达到90%+的通过率，但在高层设计遵循、方法与关系生成上表现低于60%；测试生成的覆盖率与人工相当。

**⚠️ 局限性**

主要限制包括：LLM难以准确解读并遵循高层设计模型（如PlantUML类图），缺乏将设计映射为代码的隐式规则；在无代码骨架的情况下功能正确率低；参数规模对性能有显著影响；整体对设计与实现之间的协同理解仍不足。

---

## 242. Divine Benevolence is an $x^2$: GLUs scale asymptotically faster than MLPs

**arXiv ID:** 2602.14495 | [PDF](https://arxiv.org/pdf/2602.14495v1)

**作者:** Alejandro Francisco Queiruga `[一作]` `[通讯]`, Alejandro Francisco Queiruga

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过数值分析工具探讨了模型架构选择的缩放规律，特别是GLU（门控线性单元）和MLP（多层感知器）在函数重构问题上的表现。

**💡 创新点**

创新点在于提出了GLU的缩放斜率为L(P)∝P^-3，而MLP的缩放斜率仅为L(P)=P^-2，并提出了新的“门控二次单元”（Gated Quadratic Unit），其缩放斜率更陡峭。

**🔧 技术方法**

使用了数值分析和函数逼近理论，特别是通过构造参数和经验验证来分析网络的缩放性能。

**📊 数据集**

使用了1D函数逼近的数据集，目标函数为f(x)=(1+cos^2(πx))^-1，样本点为[-1, 1]区间内的10,000个点。

**📈 对比分析**

通过与线性和二次样条的比较，发现MLP和GLU的误差与样条相当，且在缩放斜率上，MLP的斜率为n^-2.13，GLU的斜率为n^-3.08，符合理论预期。

**⚠️ 局限性**

限制在于测量仅限于1D合成问题和浅层模型，可能无法在真实数据集和大模型中体现相同的效果。

---

## 243. SketchingReality: From Freehand Scene Sketches To Photorealistic Images

**arXiv ID:** 2602.14648 | [PDF](https://arxiv.org/pdf/2602.14648v1)

**作者:** Ahmed Bourouis `[一作]` (University of Surrey), Yulia Gryaditskaya `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于模调网络和语义草图编码的潜在扩散模型，能够将自由手绘草图与文本提示生成高质量、与草图结构高度一致的真实感图像。

**💡 创新点**

创新点在于：①使用CLIP‑基草图语义编码器并细调以获取丰富语义特征；②在潜在空间引入可学习的模调网络，直接对噪声进行缩放/平移调制；③提出无需像素对齐的注意力监督损失，使模型可在真实自由手绘草图上训练。

**🔧 技术方法**

技术包括：潜在扩散模型（Stable Diffusion 2.1/XL）、CLIP 语义草图编码器、编码器‑解码器模调网络、交叉注意力监督损失、F1/LPIPS/CLIP 评估指标。

**📊 数据集**

数据集为 FS‑COCO（10k 对齐的自由手绘草图与 MS‑COCO 图像/文字描述），并在训练中使用合成草图来补充训练数据。

**📈 对比分析**

与 ControlNet、T2I‑Adapter、ControlNext、SG、FreeControl 等基线在 475 个测试草图上进行零射击和微调对比；使用 FID、CLIP 相似度、LPIPS 评估。该方法在 FID≈57、CLIP 相似度≈0.85、LPIPS≈0.27 上均显著优于基线，且在用户研究中获得最高满意度。

**⚠️ 局限性**

局限性：对极度抽象或高失真草图的鲁棒性有限；缺乏像素级对齐导致对细节的精细控制不足；模型依赖大量草图与文本配对，数据稀缺时效果可能下降。

---

## 244. Attention in Constant Time: Vashista Sparse Attention for Long-Context Decoding with Exponential Guarantees

**arXiv ID:** 2602.13804 | [PDF](https://arxiv.org/pdf/2602.13804v1)

**作者:** Vashista Nobaub `[一作]` `[通讯]` (Datar Consulting), Vashista Nobaub (Datar Consulting)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于几何稳定性理论的稀疏分页注意力机制，证明在存在正的支持间隙时可实现对长上下文的常数规模有效注意力；

**💡 创新点**

创新点在于将注意力问题视为凸多面体投影，利用面稳定性和指数泄漏性质获得稀疏性保证，并实现可插拔的分页稀疏注意力；

**🔧 技术方法**

采用熵正则化投影、KKT 诊断、分页键值缓存、前向-后向求解器（EG/Frank‑Wolfe）以及融合 CUDA 核，构建低复杂度解码流程；

**📊 数据集**

在工业场景下使用 Llama‑3‑8B 大模型进行长上下文解码测试，未明确列出公开基准数据集；

**📈 对比分析**

与传统 dense SDPA/FlashAttention 在 H100 GPU 上进行对比，显示在 8k–128k 上下文长度范围内稀疏注意力的解码延迟保持近常数，而密集注意力随上下文线性增长；

**⚠️ 局限性**

局限性包括需要正的支持间隙（Δ>0），对近似退化或极端多样化上下文的稀疏性保证有限；

---

## 245. Agent Mars: Multi-Agent Simulation for Multi-Planetary Life Exploration and Settlement

**arXiv ID:** 2602.13291 | [PDF](https://arxiv.org/pdf/2602.13291v1)

**作者:** Ziyang Wang `[一作]` `[通讯]`, Ziyang Wang

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文设计并实现了 Agent Mars，一个面向火星基地操作的多代理模拟框架，用于研究安全、可审计的协同与决策。

**💡 创新点**

创新点：①构建 93 代理、7 层链式组织与资产拥有权；②提出 HCLC（Hierarchical & Cross‑Layer Coordination）架构，保持指挥链同时支持可审计跨层快捷通道；③集成动态角色切换、共享记忆、提议‑投票共识与翻译介入等交互模块；④发布 13 个可复现的火星任务脚本和 AMPI（Agent Mars Performance Index）综合性能指标。

**🔧 技术方法**

技术手段：LLM（ChatGPT）驱动的代理，ReAct/Reflexion 交互模式；分层消息路由与跨层白名单；动态角色切换算法；记忆缓存（短期/长期/共享）；翻译代理；提议‑投票共识协议；自动化脚本执行与日志记录。

**📊 数据集**

数据集：13 个火星场景脚本（场景提示）与手工定义的角色/资产表格，作为基准测试集合；无公开大型数据集。

**📈 对比分析**

比较方法：在固定提示下，使用 N=20 次实验，系统性切换路由（STRICT/CROSSLAYER）、领导层（single/functional）、角色切换、记忆模式、共识、翻译等配置；评估指标包括时间、消息数、失败计数、跨层利用率、角色切换次数，综合 AMPI 分数。实验结果表明：跨层路由+功能领导能显著降低时间和消息；动态角色切换降低失败；共享记忆对长任务有利；共识与翻译在某些场景提升可靠性但会增加延迟。

**⚠️ 局限性**

局限性：①依赖单一 LLM 后端，模型漂移与外部 API 延迟会影响结果；②缺乏连续物理动力学、传感噪声等真实系统耦合；③AMPI 只衡量时间/失败等，未覆盖风险、认知负荷等方面；④跨层白名单由人工制定，缺乏形式化安全验证；⑤未进行长期演练与人因（疲劳、工作负荷）建模。

---

## 246. Designing a Rashomon Machine: Pluri-perspectivism and XAI for Creativity Support

**arXiv ID:** 2602.14232 | [PDF](https://arxiv.org/pdf/2602.14232v1)

**作者:** Marianne Bossema `[一作]` (Amsterdam University of Applied Sciences), Somaya Ben Allouch `[通讯]` (University of Amsterdam)

**通讯引用:** 2845 | [OpenAlex ID](https://openalex.org/A5057289735)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Pluri‑perspectivism 框架，利用 XAI 方法（Rashomon 技术、对比与反事实解释、特征重要性）构建“Rashomon Machine”，旨在通过多维视角（社会、符号、物质、时间、空间）提升人机共创的情境感知与创造性探索。

**💡 创新点**

创新点在于将 XAI 从传统的“解释决策”转向“解释可能性”，并通过 Pluri‑perspectivism 将多视角的创造性经验映射到 AI 解释空间，实现系统主动的视角交换与“生产性摩擦”，从而在生成式 AI 与人类的协作中保持创意与可理解性的平衡。

**🔧 技术方法**

主要技术包括：
- 基于 VLM 的提示式推理（schema‑guided prompting）
- Rashomon Set 生成与管理
- 对比与反事实解释
- 特征重要性评估（Feature Importance）
- 交互式反馈循环（Perspective Taking / Offering）
- 未来计划中的传感器‑驱动实时干预与本地部署的开源基础模型。

**📊 数据集**

未使用公开数据集；框架基于通用 VLM 与用户交互产生的自生成数据，实验计划在老年人共创实验中收集具体案例进行验证。

**📈 对比分析**

目前未进行系统评估或与其他方法比较；论文主要为概念与设计层面的阐述，后续研究将通过人机共创实验评估创意生成、流畅度与用户体验等指标。

**⚠️ 局限性**

局限性包括：
- 仍处于概念验证阶段，缺乏实证数据与性能评估。
- 视角（perspective）的隐喻在机器端可能不具备真实主观体验。
- 需要在真实情境中验证对老年人等非专家用户的可用性与有效性。
- 受限于闭源云模型的隐私与可定制性，后续计划转向本地部署与开放模型。

---

## 247. Sequential BP-based Decoding of QLDPC Codes

**arXiv ID:** 2602.13420 | [PDF](https://arxiv.org/pdf/2602.13420v1)

**作者:** Mohsen Moradi `[一作]` (New Mexico State University), David G. M. Mitchell `[通讯]` (New Mexico State University)

**通讯引用:** 5813 | [OpenAlex ID](https://openalex.org/A5076523933)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了两种顺序调度方案（SCNS和SVNS）用于改进量子低密度奇偶校验码（QLDPC）的贝叶斯传播（BP）译码，并将其嵌入到BP引导消除（BPGD）中形成SBPGD；

**💡 创新点**

创新点在于通过改变消息更新顺序，消除BP中的同步停滞与退化问题，从而无需改变码结构即可显著提升译码性能并降低消息量和计算成本；

**🔧 技术方法**

采用BP、顺序CN/ VN调度、BPGD/ SBPGD等经典译码技术，并在CSS框架下对QLDPC码进行仿真；

**📊 数据集**

使用QLDPC基准码[[1922,50,16]] C2与[[882,24,18≤d≤24]] B1进行实验；

**📈 对比分析**

与传统洪泛式BP和BPGD进行对比，实验显示顺序调度可在低错误率下将FER提升多达两位数，并将CN→VN消息传输量削减至85%，SBPGD在保持或更低迭代次数的同时，减少了消除轮次，进一步提升性能；

**⚠️ 局限性**

局限性包括：仅在独立Pauli‑X噪声模型下验证；调度顺序随机选择，未进行优化；仅针对CSS码，其他噪声模型与非CSS码的效果待进一步研究。

---

## 248. Learning Physiology-Informed Vocal Spectrotemporal Representations for Speech Emotion Recognition

**arXiv ID:** 2602.13259 | [PDF](https://arxiv.org/pdf/2602.13259v1)

**作者:** Xu Zhang `[一作]` (Macquarie University), Zhangkai Wu `[通讯]` (Macquarie University)

**通讯引用:** 231 | [OpenAlex ID](https://openalex.org/A5036594859)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了 PhysioSER 框架，通过融合基于语音解剖与生理（VAP）的四元音频特征与预训练自监督模型的潜在表示，实现对语音情绪的高效、可解释识别。

**💡 创新点**

创新点包括：①引入由幅度、幅度变化、瞬时频率与群延迟构成的 VAP‑信息四元组；②利用 Hamilton 结构的四元数卷积（QSE）对四元组进行跨通道耦合；③通过对齐投影（CPA）对潜在与 VAP 分支的语音表示进行互信息对齐，提升互补性；④轻量级设计，参数仅占主干 2% 以内，仍实现显著性能提升。

**🔧 技术方法**

技术手段涵盖 STFT 并求幅度与相位的一阶导数、四元数神经网络（Hamilton 结构卷积）、自监督学习主干（WavLM、HuBERT、Wav2vec2、Emotion2Vec、BEATs、CLAP）、掩码注意力池化、对比学习（InfoNCE）以及 Transformer 融合头。

**📊 数据集**

使用 14 个公开语音情绪数据集，覆盖 10 种语言（英语、俄语、波兰语、希腊语、意大利语、法语、西班牙语、孟加拉语、德语），如 CREMA‑D、EMNS、JLCorpus、EmoV‑DB、RAVDESS、RESD、nEMO、AESDD、Emozionalmente、Oréau、MESD、SUBESCO、PAVOQUE、CaFE。

**📈 对比分析**

在 6 种主干上与仅使用冻结主干的基线做对比，PhysioSER 在所有数据集上均取得一致提升；例如在 CREMA‑D 上 WavLM 基线 69.69% 提升至 75.20%（+7.9% WA），在弱主干（如 Wav2vec2）和非英语数据集（如 JLCorpus、nEMO）上提升更为显著，最高可达 +136% WA；消融实验验证各模块对性能的贡献。

**⚠️ 局限性**

主要局限在于对 STFT 的窗口与分辨率依赖，且目前尚未将 VAP‑特征嵌入大型语音模型；未来工作可进一步探索更高分辨率分析、跨模态扩展以及与其他生理信号的联合建模。

---

## 249. Ontological grounding for sound and natural robot explanations via large language models

**arXiv ID:** 2602.13800 | [PDF](https://arxiv.org/pdf/2602.13800v1)

**作者:** Alberto Olivares-Alarcos `[一作]` (Institut de Robòtica i Informàtica Industrial), Guillem Alenyà `[通讯]` (Institut de Robòtica i Informàtica Industrial)

**通讯引用:** 4225 | [OpenAlex ID](https://openalex.org/A5085564113)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

整合本体推理与大语言模型，在人机协作场景中生成对比性、自然化、可交互的机器人解释，并在SSD检测实验中进行验证。

**💡 创新点**

首次将经验典型性判定（基于经验最高密度区间）、ACXON对比叙事检索、RAG增强与LLM精炼相结合，形成符号-神经双重框架，实现高质量、可交互的解释生成。

**🔧 技术方法**

使用OWL2 DL本体与Prolog推理、经验HDI算法进行典型性分类、ACXON对比叙事检索、检索增强生成(RAG)、gpt‑oss:20b LLM（Ollama）与Pydantic AI进行文本精炼。

**📊 数据集**

基于18次人机协作SSD表面检测实验的ROS bag日志，提取任务数量、完成时长和机器人不确定次数等属性。

**📈 对比分析**

与传统ACXON基线在153对计划、3个细节层级下对比，采用解释长度、Flesch可读性得分和语义相似度评估；实验显示解释长度缩短33–93%，可读性提升19–76%，语义相似度≥0.7，差异均显著(p<0.001)。

**⚠️ 局限性**

生成延迟约10–20秒，难以满足实时对话需求；偶尔出现极端可读性差的案例；缺乏用户研究验证信任与满意度；依赖单一大型模型与高性能硬件，需进一步优化以实现实时性与鲁棒性。

---

## 250. MergePipe: A Budget-Aware Parameter Management System for Scalable LLM Merging

**arXiv ID:** 2602.13273 | [PDF](https://arxiv.org/pdf/2602.13273v1)

**作者:** Yuanyi Wang `[一作]` (Hong Kong Polytechnic University), Hongxia Yang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 6807 | [OpenAlex ID](https://openalex.org/A5082599714)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了MergePipe，一种面向大规模LLM合并的预算感知参数管理系统，重新把模型合并视为数据管理问题；

**💡 创新点**

创新点在于将专家参数读写视为可控资源，使用基于catalog的块级成本模型和预算规划，实现O(K)专家I/O的显著降低；

**🔧 技术方法**

核心技术包括持久化块级元数据catalog、基于成本的贪心规划器、流式执行引擎与事务性快照/线索追踪；

**📊 数据集**

实验使用Llama和Qwen系列公开模型（0.6B-8B）以及各类专家检查点，覆盖TIES、DARE等稀疏合并算子；

**📈 对比分析**

与传统无规划、无预算的合并脚本对比，MergePipe在专家数增至20时总I/O下降至原来的约1/10，端到端速度提升至11倍（约70%时延下降），且性能随专家数线性稳健；

**⚠️ 局限性**

局限性包括对块大小等系统参数的依赖，稀疏度较低或专家数量很少时收益有限，且需要额外的catalog维护和事务开销。

---

## 251. Learning User Interests via Reasoning and Distillation for Cross-Domain News Recommendation

**arXiv ID:** 2602.15005 | [PDF](https://arxiv.org/pdf/2602.15005v1)

**作者:** Mengdan Zhu `[一作]` (Microsoft), Liang Zhao `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于强化学习的推理框架，利用大型语言模型（LLM）从跨域用户行为中生成可复用的新闻搜索查询列表，进而提升兴趣建模与推荐效果。

**💡 创新点**

创新点包括：① 将兴趣发现视为多奖励（检索对齐、兴趣覆盖、查询专一度、列表多样性、结构有效性）下的策略优化；② 采用Group Relative Policy Optimization（GRPO）在无标注数据上训练；③ 在大规模线上系统中实现了对教师模型的在线策略蒸馏，使得小模型既保留大模型优势，又满足低延迟需求。

**🔧 技术方法**

使用的技术包括：大型语言模型（Qwen2.5-32B为教师，Qwen2.5-0.5B为学生）、GRPO强化学习、基于多维奖励的复合奖励设计、噪声清洗的RoBERTa二分类器、基于教师策略的在线蒸馏（KL逆散度最小化）以及检索对齐的内部ANN索引。

**📊 数据集**

实验数据集为内部新闻推荐系统的真实用户交互日志，包含浏览、搜索和点击等跨域行为，样本规模约50万用户、120万篇新闻。

**📈 对比分析**

与传统序列化推荐模型（GRU、SASRec、NRHUB、PinSage、HSTU）比较，本文方法在Recall@5/10、NDCG@5/10、MRR等指标上均取得显著提升；在规模化蒸馏后，小模型也保持了大部分性能，在线A/B实验中显著提升DAU和CTR。

**⚠️ 局限性**

局限性包括：① 训练与推理仍需较高算力，尤其是教师模型；② 依赖内部大型新闻数据，难以直接迁移到其他领域；③ 蒸馏后模型虽大幅减小但仍比传统方法大，可能受限于部署资源；④ RL奖励设计和超参选择对结果影响大，需要复杂调优。

---

## 252. Train Short, Inference Long: Training-free Horizon Extension for Autoregressive Video Generation

**arXiv ID:** 2602.14027 | [PDF](https://arxiv.org/pdf/2602.14027v1)

**作者:** Jia Li `[一作]`, Hayden Kwok-Hay So `[通讯]` (University of Hong Kong)

**通讯引用:** 2915 | [OpenAlex ID](https://openalex.org/A5020581824)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种训练‑free 的推理时 Horizon Extension 框架 FLEX，用于延长预训练的自回归视频扩散模型的生成时间，显著提升长视频合成的时序一致性与运动动态。

**💡 创新点**

创新点包括：① 频率感知的 3D RoPE 调制（NTK‑by‑parts），针对不同频率分量的插值与外推；② Antiphase Noise Sampling（AN）通过负相关噪声重构高频动态先验；③ 推理时注意力 Sink，将前置帧固定为全局语义锚点，保持长序列的全局一致性。

**🔧 技术方法**

主要技术包括：3D Rotary Positional Embedding（RoPE）频率调制；AR(1) 负相关噪声采样；推理时注意力 Sink；无训练的推理改造；与 Self‑Forcing、LongLive、CausVid、Rolling Forcing 等自回归模型的对比。

**📊 数据集**

实验数据集涵盖 VBench‑Long（包含 946 个官方提示）与 MovieGen 提示集（1003 条），用于评估 30 秒（6×）与 60 秒（12×）长视频生成性能。

**📈 对比分析**

通过与 CausVid、Self‑Forcing、LongLive、Rolling Forcing 在 VBench‑Long 上的 30 秒和 60 秒指标比较，FLEX 在 30 秒任务中取得 83.48 的总分（6× extrapolation），在 60 秒任务中达到 82.68 的总分（12× extrapolation），显著优于同类无 fine‑tune 方法，并且在质量漂移、动态程度等维度上与训练‑long 基线持平或更优。

**⚠️ 局限性**

局限性：① 仍依赖预训练模型的容量与设计，极端长序列（>240 秒）仍可能出现身份漂移或运动单调；② 需要对 α、β、ρ 等超参进行细致调优；③ 目前只针对自回归视频扩散模型提出，尚未验证对其他生成框架的适用性。

---

## 253. Controlling Your Image via Simplified Vector Graphics

**arXiv ID:** 2602.14443 | [PDF](https://arxiv.org/pdf/2602.14443v1)

**作者:** Lanqing Guo `[一作]` (University of Texas), Siyu Huang `[通讯]` (Clemson University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于简化向量图形的可控图像生成框架 Vec2Pix，可通过层级 SVG 进行元素级编辑。

**💡 创新点**

创新点在于将 SVG 作为条件引导，并通过 Noise Prediction from Vectors (NPV) 模块直接预测初始噪声，实现语义与结构的精确对齐。

**🔧 技术方法**

使用 DiffVG、Bezier Splatting、Flux 1‑dev latent flow Transformer、LoRA 低秩适配器、SAM 等技术。

**📊 数据集**

基于 LAION‑400M 过滤得到 5 M 图像‑文本对，并利用 Image‑to‑SVG 模块生成对应 SVG 进行训练。

**📈 对比分析**

与 Sketch、Depth、Stroke、Segmentation Mask 等传统条件进行对比，实验显示 FID 降低、PSNR 提升，NPV 模块显著提升编辑精度。

**⚠️ 局限性**

局限在对高分辨率复杂场景仍需改进，及不规则结构的 SVG 生成效率与质量仍有提升空间。

---

## 254. Location as a service with a MEC architecture

**arXiv ID:** 2602.13358 | [PDF](https://arxiv.org/pdf/2602.13358v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 255. GRRM: Group Relative Reward Modeling for Machine Translation

**arXiv ID:** 2602.14028 | [PDF](https://arxiv.org/pdf/2602.14028v1)

**作者:** Sen Yang `[一作]` (Nanjing University), Shujian Huang `[通讯]` (Nanjing University)

**通讯引用:** 3702 | [OpenAlex ID](https://openalex.org/A5102865824)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了群组质量度量（GQM）与群组相对奖励模型（GRRM），并将其嵌入GRPO框架以提升机器翻译的质量与推理能力。

**💡 创新点**

首次引入集合级评估的GQM，打破传统独立评分的Scalar Quality Metric（SQM）限制；实现高效的GRRM，可在GRPO中一次性完成所有候选的相对排名并生成CoT式的细粒度奖励，显著抑制奖励黑客并激发模型推理。

**🔧 技术方法**

使用GRPO与RLVR强化学习、生成式奖励模型（基于Qwen2.5-7B）、Chain‑of‑Thought推理、SFT预训练、跨语言增广、以及LLM‑as‑a‑Judge评估。

**📊 数据集**

训练集：TowerBlocks中中文-英文18.8k样本；评测集：Newstest2020 Zh→En、GeneralMT2022 MQM（En→De/En→Ru）、Seed‑X‑Challenge、WMT23/24 的多语言 MT 任务。

**📈 对比分析**

与传统SQM、DRM（CometKiwi‑XXL、BT‑RM）以及基线GenRM对比；GRRM 在排名准确度上比SQM提升30‑40%；在 MT 上 BLEURT 提升约7.5 分、LLM‑Judge 提升约16 分；在 Seed‑X‑Challenge 里与 DeepSeek‑R1‑0528 相当或略优。

**⚠️ 局限性**

局限：GQM 受训练时最大群组大小限制，超大组时准确度可能下降；相对排名模式无法全局惩罚整体低质量组，缺少绝对质量约束。

---

## 256. Effect of Convolutional Depth on Image Recognition Performance: VGG vs. ResNet vs. GoogLeNet

**arXiv ID:** 2602.13298 | [PDF](https://arxiv.org/pdf/2602.13298v1)

**作者:** Manfred M. Fischer `[一作]` (Vienna University of Economics and Business), Joshua Pitts `[通讯]` (Boston University)

**通讯引用:** 193 | [OpenAlex ID](https://openalex.org/A5045515090)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 VGG、ResNet 与 GoogLeNet 三种经典 CNN 架构在不同深度下进行统一训练、评估的对照实验，探究深度对准确率、优化过程与计算效率的影响。

**💡 创新点**

提出“有效深度”概念并量化，区分名义深度与实际信息路径深度，阐明深度的益处取决于架构机制而非单纯层数。

**🔧 技术方法**

使用统一的训练协议（SGD、学习率调度、权重衰减）、路径基深度计算、梯度范数监测、MAC 计数等技术。

**📊 数据集**

CIFAR‑10 数据集（50k 训练 + 10k 测试，10 类）。

**📈 对比分析**

采用相同的数据增强、批大小、学习率等设置，比较 Top‑1 准确率、训练损失收敛速度与计算成本；结果显示 ResNet 与 GoogLeNet 随深度增大仍持续提升准确率且计算更高效，而 VGG 在较深时准确率饱和且收敛慢。

**⚠️ 局限性**

实验仅在单一分类任务和单一硬件平台上进行，未针对各架构做专门调参，深度定义存在抽象性，结果可能不直接迁移到目标检测、语义分割等任务。

---

## 257. VeriSBOM: Secure and Verifiable SBOM Sharing Via Zero-Knowledge Proofs

**arXiv ID:** 2602.13682 | [PDF](https://arxiv.org/pdf/2602.13682v1)

**作者:** Gianpietro Castiglione `[一作]` (Newcastle University), Narges Khakpour `[通讯]` (Newcastle University)

**通讯引用:** 425 | [OpenAlex ID](https://openalex.org/A5080062466)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

设计并实现了 VeriBOM，一种基于向量承诺与零知识证明的 SBOM 隐私披露与可验证框架，支持第三方对软件依赖真实性和合规性的盲验证。

**💡 创新点**

创新点包括：① 双树架构（包树与影子树）实现包真实性与合规性双重证明；② 使用折叠式 zkSNARK（Nova）实现对任意长度 SBOM 的常数大小证明；③ 可按需聚合多项政策的推理引擎，支持客户端自定义合规性组合；④ 无需密钥分发，完全基于公共根哈希实现可信验证。

**🔧 技术方法**

核心技术：向量承诺（Merkle/稀疏 Merkle 树），零知识 SNARK（Spartan / Nova），Poseidon 哈希，Circom 电路，Python + SQLite 进行政策传播与管理，Rust 实现高性能加密。

**📊 数据集**

使用真实开源包仓库数据（Crates.io、npm 等）进行评估，构建覆盖数百万包的向量承诺；SBOM 规模从数十到数百个依赖进行实验。

**📈 对比分析**

与传统完整 SBOM 验证/签名方法对比：存储开销 <1 GB（10 M 包）；Merkle 路径大小对包数对数增长；证明大小始终 ≈13 KB；证明生成时间随依赖数线性增长（≈4–15 s）；验证时间保持 80–95 ms，几乎不受包表大小影响；相较于加密/签名方案显著提升隐私性且验证成本可控。

**⚠️ 局限性**

局限性：① 依赖于供应商提供的 SBOM 正确完整；② 证明生成仍存在一定 CPU 与时间开销，极大 SBOM 仍可能昂贵；③ 需要可信第三方维护并及时更新根哈希；④ 电路实现若存在缺陷可能被攻击；⑤ 当前不支持增量更新（需要重新构造树）。

---

## 258. Offline Learning of Nash Stable Coalition Structures with Possibly Overlapping Coalitions

**arXiv ID:** 2602.14321 | [PDF](https://arxiv.org/pdf/2602.14321v1)

**作者:** Saar Cohen `[一作]` `[通讯]` (Bar Ilan University), Saar Cohen (Bar Ilan University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在部分信息环境下，能重叠的联盟形成问题，利用固定的离线数据集推断代理偏好并恢复近似纳什稳定的分配。

**💡 创新点**

提出了新的重叠联盟形成模型，给出了半布丁（semi‑bandit）与布丁（bandit）反馈下的必要与充分覆盖假设，设计了样本效率高、误差与ε成最优（含对数因子）关系的学习算法，并证明在bandit情况下若不满足更严格的覆盖假设则不可实现高效学习。

**🔧 技术方法**

采用离线学习框架、半布丁与布丁反馈下的估计与置信上界构造、Hoeffding不等式、岭回归、潜在函数（潜在游戏）证明、以及坐标下降求解最优策略的技术。

**📊 数据集**

使用合成数据集：根据“大小相关均匀”和“大小相关高斯”两种效用生成模型生成多代理、候选联盟规模随机的游戏；探索策略包括完全随机与部分满足覆盖条件的随机策略。

**📈 对比分析**

通过对比学习得到的策略的对偶间隙（dual gap）与理论预期的1/√M、n、k依赖，实验显示在满足覆盖假设的随机探索下算法迅速逼近纳什稳定；在不满足假设的探索下表现显著下降，验证了覆盖假设的重要性。

**⚠️ 局限性**

局限包括：仅考虑对称偏好（非对称下纯策略不可行）；需先验保证数据覆盖度，实际采集可能难以满足；实验仅在合成环境下验证，真实世界验证尚未展开；bandit反馈下的实验结果未完全展示。

---

## 259. Low-Pass Filtering Improves Behavioral Alignment of Vision Models

**arXiv ID:** 2602.13859 | [PDF](https://arxiv.org/pdf/2602.13859v1)

**作者:** Max Wolff `[一作]` (Max Planck Institute for Intelligent Systems), Wieland Brendel `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过在测试时对深度神经网络输入进行低通滤波（模糊或缩小分辨率）来研究其对人机行为一致性的影响，证明Imagen模型的高一致性主要来自输入降采样；随后在傅里叶空间学习最优滤波器，并将其与人眼对比度敏感函数（CSF）进行比较；还绘制了错误一致性与OOV准确率的Pareto前沿。

**💡 创新点**

创新点在于：①发现仅通过测试时低通滤波即可显著提升错误一致性和形状偏差，且其效果优于之前关于生成目标的假设；②证明最优低通滤波器近似人眼CSF，并是行为对齐的最优滤波；③提出并计算了模型vs人类基准的Pareto前沿，揭示准确率与一致性之间的固有权衡。

**🔧 技术方法**

主要技术包括：低通滤波/模糊、图像尺寸缩放、傅里叶空间滤波器梯度优化、CLIP/Imagen模型评估、模型vs人类基准（shape bias、error consistency、OOD accuracy）评估、CSF拟合、Pareto前沿计算。

**📊 数据集**

使用的数据集为ImageNet-1k图像，经过处理得到Model‑vs‑Human基准的16个超类图像（包括12种失真、cue‑conflict、风格化、边缘、素描等）。

**📈 对比分析**

与Imagen、ViT‑22B‑384、OpenCLIP ViT‑H‑14等模型对比，低通滤波后ViT‑H‑14的错误一致性从0.28提升至0.37、形状偏差从0.60提升至0.96，接近人类平均（0.43、0.96）。在多种模型上均观察到显著提升，并绘制Pareto前沿显示仍有提升空间。

**⚠️ 局限性**

局限性包括：未评估低通滤波对神经相似度指标的影响；测试时低通滤波在更长呈现时间下的效果未知；在Brain‑Score等平台需重新预处理才能比较；准确率与一致性之间存在固有权衡，且未探索对其他任务的泛化。

---

## 260. A Generative AI Approach for Reducing Skin Tone Bias in Skin Cancer Classification

**arXiv ID:** 2602.14356 | [PDF](https://arxiv.org/pdf/2602.14356v1)

**作者:** Areez Muhammed Shabu `[一作]` (University of Sheffield), Asra Aslam `[通讯]` (University of Sheffield)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过LoRA微调Stable Diffusion对ISIC数据集中暗肤色的皮肤病变图像进行合成，生成808张合成图像，并将其与原始数据合并用于肿瘤分割和二分类任务。

**💡 创新点**

提出了利用低秩适配的Stable Diffusion在少量暗肤色样本上进行条件生成的方案，首次证明合成图像能够显著提升对暗肤色的检测性能，为肤色不平衡的医疗图像补全提供可扩展的方法。

**🔧 技术方法**

使用LoRA自适应Stable Diffusion生成模型、EfficientNet‑B0分类网络、CNN分割模型、图像预处理与增广、以及统计相似性验证（SSIM、颜色直方图、GLCM）等技术。

**📊 数据集**

使用ISIC 2024皮肤病变数据集（17,728张真实图像，其中1,407张暗肤色），并在其基础上生成808张暗肤色合成图像。

**📈 对比分析**

在真实图像上评估：分割任务中IoU从0.82提升至0.85，Dice从0.88提升至0.90；分类任务中准确率从85.45%提升至92.14%，精确率和召回率亦提升，虽验证AUC略降，但与传统最大流算法对比显示CNN显著优越。

**⚠️ 局限性**

主要限制包括合成图像仅通过统计指标验证，缺乏皮肤科医生的临床评估；合成量仅占数据集4.6%，未充分解决癌症类别不平衡；分类评估使用包含合成图像的验证集，未检验纯真实数据的泛化；生成过程耗时较长，需进一步优化。

---

## 261. AI Arms and Influence: Frontier Models Exhibit Sophisticated Reasoning in Simulated Nuclear Crises

**arXiv ID:** 2602.14740 | [PDF](https://arxiv.org/pdf/2602.14740v1)

**作者:** Kenneth Payne `[一作]` `[通讯]` (Kings College London), Kenneth Payne (Kings College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用三款前沿大型语言模型（GPT‑5.2、Claude Sonnet 4、Gemini 3 Flash）在自定义核危机模拟中进行对弈，记录并分析其战略推理、欺骗、理论心智与元认知。

**💡 创新点**

引入三阶段认知架构（反思→预测→信号/行动）并将信号与行动分离，形成可观测的战略心理学数据；通过同时移动与事故机制模拟现实危机的不确定性与摩擦；揭示模型训练（RLHF）导致的情境依赖式约束与悖逆行为。

**🔧 技术方法**

使用预训练大型语言模型，配合自提示的反思、预测、信号和行动阶段；对生成文本进行自然语言处理评估（一致性、误差、信度等），并通过模拟引擎实现多回合交互。

**📊 数据集**

数据完全来自模拟：21场对弈、329回合，产生约78万词的战略推理日志；无外部真实数据集，仅使用模型自身预训练语料。

**📈 对比分析**

通过比较胜率、信号一致率、升级梯度、预测误差（MAE）和信誉评级等指标进行评估。Claude总体占优（67%胜率），GPT‑5.2在有时间限制时翻转至75%胜率，Gemini则表现最为波动；信号一致率约70%；预测误差范围在85–150点；核级使用率高，核禁忌表现相对宽松。

**⚠️ 局限性**

实验规模有限（21场对弈），场景样本不完整，模型仅为当前三代，结果对未来版本的可迁移性不确定；模拟抽象化、缺乏真实人类决策者对照；部分结论依赖文本解读，存在主观性。

---

## 262. Sim2Radar: Toward Bridging the Radar Sim-to-Real Gap with VLM-Guided Scene Reconstruction

**arXiv ID:** 2602.13314 | [PDF](https://arxiv.org/pdf/2602.13314v1)

**作者:** Emily Bejerano `[一作]` (Columbia University), Xiaofan Jiang `[通讯]` (Columbia University)

**通讯引用:** 4007 | [OpenAlex ID](https://openalex.org/A5063824268)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过单视角RGB图像，利用VLM推理材料属性，重建材质标注的3D场景，并用物理射线追踪模拟毫米波雷达数据，生成可用于雷达学习的合成点云。

**💡 创新点**

① 采用视觉‑语言模型（VLM）进行材料推理，突破纹理分类限制；② 将VLM推断的材料属性与物理射线追踪结合，实现从RGB到雷达数据的端到端无人工场景重建；③ 通过仿真预训练提升雷达感知的几何定位精度。

**🔧 技术方法**

深度估计（MoGe v2）、分割（SAM2+Grounding DINO）、VLM材料分类（InternVL2.5‑8B）、物理射线追踪（Mitsuba 3）、Fresnel反射模型（ITU‑R P.2040）、雷达点云检测网络（PointPillars）。

**📊 数据集**

Indoor FireRescue Radar (IFR) 数据集，包含10栋建筑的同步RGB、LiDAR和4D毫米波雷达帧，并带有门类目标的3D边框标注。

**📈 对比分析**

在真实数据不同规模（5%–100%）下，将模型从随机初始化（Baseline）与仿真预训练（Sim Pretrain）对比。预训练后在所有数据量下均提升3D AP，最高提升为+3.7点（IoU 0.3）。

**⚠️ 局限性**

仅针对门类目标，仿真点云稠密度仅为真实数据的12%；场景视角受限（相机视角≈65°，雷达覆盖120°）；射线追踪缺乏高阶散射模型；实验仅覆盖室内走廊环境，缺乏更广泛物体与多视角验证。

---

## 263. Differential pose optimization in descriptor space -- Combining Geometric and Photometric Methods for Motion Estimation

**arXiv ID:** 2602.14297 | [PDF](https://arxiv.org/pdf/2602.14297v1)

**作者:** Andreas L. Teigen `[一作]` (Norwegian University of Science and Technology), Rudolf Mester `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 2108 | [OpenAlex ID](https://openalex.org/A5048393027)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于描述子相似度的姿态优化方法（D-JET），试图将几何特征描述子与光度优化相结合，以实现更鲁棒的两帧姿态估计。

**💡 创新点**

创新点在于：①将稠密或局部稠密的描述子差值作为光度残差的替代；②利用描述子可微性在表面上构造局部抛物线进行差分优化；③在传统直接视觉里程计（如JEB、P-JET）基础上引入描述子残差，并在共线约束下实现联合优化。

**🔧 技术方法**

使用的技术包括：ORB特征检测与二进制描述子、描述子自相似与交叉相似实验、二阶泰勒展开的局部抛物线拟合、基于拉格朗日乘子求解的极线约束优化、迭代非线性求解与关键点更新策略。

**📊 数据集**

使用的数据集为：KITTI（真实城市驾驶场景）和VAROS（合成水下环境）两套序列，用于评估旋转误差、平移误差和平均Hamming距离。

**📈 对比分析**

与方法比较：D-JET 与原始光度JEB（P-JET）、重投影误差（RP‑E）以及5‑点基础算法（5‑P）进行对比。结果显示：在KITTI上，D-JET 在大部分序列中表现出更一致的误差，尤其对视角变化敏感的序列优于P-JET；但整体误差仍略高于RP‑E；在VAROS上差距更大，D-JET 的优势不明显。性能上，D-JET 计算量大于光度方法，速度比重投影误差慢。

**⚠️ 局限性**

局限性包括：描述子相似度在连续空间中的平滑性不足，导致优化目标与实际姿态误差不完全相关；对尺度变化不敏感；局部抛物线在大偏移时失效；以及在光照或纹理匮乏的场景中表现不佳。

---

## 264. Choosing How to Remember: Adaptive Memory Structures for LLM Agents

**arXiv ID:** 2602.14038 | [PDF](https://arxiv.org/pdf/2602.14038v1)

**作者:** Mingfei Lu `[一作]` (University of Technology Sydney), Yi Zhang `[通讯]` (University of Technology Sydney)

**通讯引用:** 95692 | [OpenAlex ID](https://openalex.org/A5100388089)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出FluxMem框架，允许LLM代理在长时序对话中动态选择并融合多种记忆结构以提升记忆利用与回答质量。

**💡 创新点**

创新点在于将记忆结构视为可学习的决策变量，结合离线基于奖励的监督和Beta混合模型门控实现自适应结构选择与记忆融合。

**🔧 技术方法**

采用三层记忆层级（STIM、MTEM、LTSM）、多结构（线性/图形/层级）存储、MLP结构选择器、Beta Mixture Model门控、dense+BM25检索与GPT-4.1生成。

**📊 数据集**

使用PersonaMem和LoCoMo两大长时序对话基准数据集进行实验评估。

**📈 对比分析**

与Mem0、ZEP、A-Mem、HippoRAG2等12种现有记忆系统比较，FluxMem在PersonaMem平均准确率提升9.18%，在LoCoMo平均F1/BLEU/ROUGE提升约6%，显著优于所有基线。

**⚠️ 局限性**

局限性包括在某些与固定结构高度契合的子任务上不一定优于最强基线、需要离线奖励标签与较高实现复杂度、以及对超参数如BMM门阈值和最小保留数的敏感性。

---

## 265. The Shadow Boss: Identifying Atomized Manipulations in Agentic Employment of XR Users using Scenario Constructions

**arXiv ID:** 2602.13622 | [PDF](https://arxiv.org/pdf/2602.13622v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 266. NP-hardness of p-adic linear regression

**arXiv ID:** 2602.13278 | [PDF](https://arxiv.org/pdf/2602.13278v1)

**作者:** Gregory D. Baker `[一作]` (Australian National University), Gregory D. Baker `[通讯]` (Australian National University)

**通讯引用:** 10495 | [OpenAlex ID](https://openalex.org/A5032414479)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明了在维度为输入参数的情况下，p-进制线性回归问题是NP‑难的，构造了从最大割问题的多项式时间归约；

**💡 创新点**

创新点在于首次将p‑进制绝对值的非阿基米德性质与正则化技术相结合，利用超度量不等式强制回归系数仅取0或1，从而把回归问题转化为最大割；

**🔧 技术方法**

主要使用的技术包括p‑进制绝对值的定义与性质、超度量不等式的分析、构造正则化“强制点”以及多项式时间归约；

**📊 数据集**

本文没有使用真实数据集，而是以无权图（最大割实例）构造的合成回归数据；

**📈 对比分析**

由于研究是理论复杂度证明，没有进行实验比较，结论表明若能多项式求解该回归问题，则P=NP；

**⚠️ 局限性**

局限在于仅给出了最坏情况的NP‑难性证明，对特定结构实例的可解性、近似算法以及不同聚合方式的复杂度仍未解决。

---

## 267. Interpretable clustering via optimal multiway-split decision trees

**arXiv ID:** 2602.13586 | [PDF](https://arxiv.org/pdf/2602.13586v1)

**作者:** Hayato Suzuki `[一作]` (University of Tsukuba), Yuichi Takano `[通讯]` (University of Tsukuba)

**通讯引用:** 1552 | [OpenAlex ID](https://openalex.org/A5081879179)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于最优多路分裂决策树的可解释聚类方法 ICOMT，能够同时兼顾聚类精度与可解释性。

**💡 创新点**

创新点在于：① 将 OMT 的多路分裂结构迁移至无监督聚类，并通过 0–1 ILP 精确求解；② 采用一维 K‑means 进行连续变量自适应离散化；③ 通过全局优化直接决定树结构，避免传统贪婪搜索导致的子最优。

**🔧 技术方法**

主要技术包括：多路分裂决策树建模、0–1 整数线性规划、连续变量的一维 K‑means 离散化、路径枚举与冗余消除、Gurobi 求解器。

**📊 数据集**

实验使用四个公开数据集：Seeds、Statlog（Vehicle Silhouettes）、Real Estate Valuation、Customer Purchasing Behaviors。

**📈 对比分析**

与 ICOT、CART‑Hybrid、PCT 三种基线方法进行对比，指标为 ARI、轮廓系数、Dunn 指数和计算时间。ICOMT‑K 在大多数数据集上实现了更高的 ARI、较小的树深度和更少的簇，同时计算时间与基线相近或略长；ICOMT‑B 在某些线性数据集上表现与基线相当。

**⚠️ 局限性**

局限性：① 只能处理互斥聚类，无法表示重叠簇；② 仅使用一维 K‑means 离散化，缺乏针对不同特征自适应选择离散化方法的机制；③ 对大规模高维数据的求解规模与时间仍需进一步优化。

---

## 268. TabTracer: Monte Carlo Tree Search for Complex Table Reasoning with Large Language Models

**arXiv ID:** 2602.14089 | [PDF](https://arxiv.org/pdf/2602.14089v1)

**作者:** Zhizhao Luo `[一作]` (Beijing Institute of Technology), Rui Mao `[通讯]` (Shenzhen University)

**通讯引用:** 2322 | [OpenAlex ID](https://openalex.org/A5101724957)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 TabTracer，一种基于 MCTS 的代理框架，结合 LLM 规划、Typed 表操作与版本化表状态回滚，用于复杂表格推理。

**💡 创新点**

创新点包括：① 逐步验证与可机验奖励，抑制表格幻觉；② 反馈驱动的 MCTS 与版本化快照实现可靠回溯与搜索重用；③ 预算感知剪枝、去重与单步执行-反思循环，显著降低 token 消耗。

**🔧 技术方法**

核心技术为：大语言模型（如 Qwen3-32B、GPT‑4.1‑Mini）规划，Typed dataframe 工具（列筛选、行过滤、代码生成），Monte Carlo Tree Search 结合 UCB1 与反思评分，哈希表状态缓存与去重，版本化快照与回滚机制。

**📊 数据集**

在 TabFact、WikiTQ 与 CRT 三个公开表格推理数据集上进行实验，分别覆盖事实验证、开放域查询与数值计算任务。

**📈 对比分析**

与 Prompt‑based、Chain‑of‑Thought、Agent‑based 等九个基线对比，TabTracer 在所有模型与数据集上均名列前茅，平均提升准确率 4–7%（最高 6.7%），并将 token 使用量降低 59–84%，展现出显著的性能与效率优势。

**⚠️ 局限性**

局限性包括：对大表格的最大深度与预算仍受限；单步执行-反思可能无法处理极为复杂的多步推理；以及对 LLM 生成代码的依赖，若代码生成质量不高会影响后续计算。

---

## 269. Distributed Quantum Gaussian Processes for Multi-Agent Systems

**arXiv ID:** 2602.15006 | [PDF](https://arxiv.org/pdf/2602.15006v1)

**作者:** Meet Gandhi `[一作]` (Colorado School of Mines), George P. Kontoudis `[通讯]` (Colorado School of Mines)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种分布式量子高斯过程（DQGP）框架，利用多智能体协同训练量子核高斯过程。

**💡 创新点**

创新点在于将分布式共识Riemannian ADMM（DR-ADMM）用于量子超参数优化，融合量子核表达力与分布式计算。

**🔧 技术方法**

使用量子编码电路、投影量子核、Matérn/高斯外核、Riemannian ADMM、Qiskit/PennyLane、sQUlearn等技术。

**📊 数据集**

使用NASA SRTM地形数据（四个区域）和合成的量子高斯过程先验样本。

**📈 对比分析**

与传统全局GP、FACT-GP、apx-GP对比，DQGP在NRMSE、NLPD上均显著优于分布式方法，甚至接近或超越单机全局GP。

**⚠️ 局限性**

局限性包括对量子硬件的依赖、对量子相位噪声的敏感，以及在样本量较大时NLPD偶有劣势。

---

## 270. Cognitive Chunking for Soft Prompts: Accelerating Compressor Learning via Block-wise Causal Masking

**arXiv ID:** 2602.13980 | [PDF](https://arxiv.org/pdf/2602.13980v1)

**作者:** Guojie Liu `[一作]` (National University of Defense Technology), Jie Yu `[通讯]` (National University of Defense Technology)

**通讯引用:** 34364 | [OpenAlex ID](https://openalex.org/A5100620306)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种并行迭代压缩方法（PICC），通过块级因果注意力掩码将长文本压缩为少量记忆嵌入。

**💡 创新点**

通过显式将记忆标记限定为对应文本块的局部注意力，引入块级因果掩码，使压缩器的学习从全局依赖转为局部提取，显著降低训练复杂度并提升压缩效果。

**🔧 技术方法**

基于Transformer的软提示压缩器，采用块级因果注意力掩码，预训练任务包括文本重建与补全，使用Qwen2.5-0.5B-Instruct等模型。

**📊 数据集**

预训练使用FineWeb约30亿token；微调使用SQuAD（RAG问答）和GSM8K（ICL）等数据集。

**📈 对比分析**

与AutoCompressor、xRAG、COCOM、ICAE、LLMLingua、PCC等基线对比；在多项QA和ICL任务中，PICC在高压缩率下实现相对提升29.8% F1、40.7% EM，训练时间减少约40%。

**⚠️ 局限性**

仍需大量预训练数据；压缩块大小固定，可能不适用于不同结构文本；评估仅在有限任务与模型上，未验证对更大规模LLM或多模态输入的泛化。

---

## 271. Cast-R1: Learning Tool-Augmented Sequential Decision Policies for Time Series Forecasting

**arXiv ID:** 2602.13802 | [PDF](https://arxiv.org/pdf/2602.13802v1)

**作者:** Xiaoyu Tao `[一作]` (University of Science and Technology of China), Yaguo Liu `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了Cast-R1框架，将时间序列预测重新表述为顺序决策问题，并通过工具增强的代理工作流与记忆式状态管理实现动态信息采集、模型选择、推理与预测修正。

**💡 创新点**

创新点在于：①将预测任务转化为多步决策流程；②设计记忆式状态管理以跨步保持关键上下文；③构建可调用的工具箱实现信息抽取与模型调用；④采用两阶段SFT+多轮强化学习与课程学习训练长程决策策略。

**🔧 技术方法**

技术包括：基于Qwen3大语言模型的代理推理、工具增强代理、记忆式状态管理、结构化动作空间、监督微调 + 多轮强化学习（GRPO）以及课程学习策略。

**📊 数据集**

使用的数据集涵盖ETT（电力变压器）、Wind（风能发电）、NP、PJM、BE、FR、DE（电价）等多领域、不同频率的真实时间序列。

**📈 对比分析**

与统计方法（ARIMA、Prophet）、深度学习模型（PatchTST、iTransformer、TimeXer等）、基础模型（Chronos-2、TimesFM）和LLM预测基线（OFA、Time-LLM、TimeReasoner）进行对比，采用MSE/MAE评估；Cast-R1在绝大多数数据集上MSE最低，MAE排名第一或第二，整体性能显著优于基线。

**⚠️ 局限性**

局限性包括：对极度非平稳或高噪声序列仍存在挑战；依赖大型预训练语言模型，训练成本高；工具箱和奖励设计需人工制定，扩展性和迁移性受限。

---

## 272. Pocket RAG: On-Device RAG for First Aid Guidance in Offline Mobile Environment

**arXiv ID:** 2602.13229 | [PDF](https://arxiv.org/pdf/2602.13229v1)

**作者:** Dong Ho Kang `[一作]` (Ustechlab Research Institute), Sungsoo Lim `[通讯]` (Kookmin University)

**通讯引用:** 895 | [OpenAlex ID](https://openalex.org/A5100804322)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在离线移动设备上使用检索增强小语言模型（RAG）为急救提供即时指导的系统。

**💡 创新点**

设计了资源友好的Hybrid RAG、选择性上下文压缩、批量提示解码和KV缓存量化等多项优化，使模型在Android 2GB内存下实现 3.7s 的首次令牌时间。

**🔧 技术方法**

采用量化小型语言模型（如 Qwen3 0.6B、Gemma3 1B）、轻量化嵌入（gte-small、bge-small）、双阶段检索、批处理解码、INT8 KV缓存等技术。

**📊 数据集**

基于WHO Basic Emergency Care (BEC) 与心理急救 (PFA) 的 200 道多项选择题集，以及 SQuAD、HotpotQA 等标准 QA 基准。

**📈 对比分析**

与无检索、单纯 RAG、RAG+重排序三种配置对比，在物理急救 94.5% / 心理急救 97% 的准确率；在标准 QA 上与 MobileRAG 相比，准确率略高且 TTFT 仅 3.7 秒。

**⚠️ 局限性**

仅用多项选择题评估，未涵盖开放式对话与实时语音交互，能耗未测量，且模型替换仍需技术门槛。

---

## 273. Graph Homomorphisms and Universal Algebra

**arXiv ID:** 2602.14243 | [PDF](https://arxiv.org/pdf/2602.14243v1)

**作者:** Manuel Bodirsky `[一作]` `[通讯]` (Institute for Algebra), Manuel Bodirsky (Institute for Algebra)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本课程通过图同态、约束满足问题和通用代数的视角，系统介绍了从基本理论到算法与复杂性分类的完整框架。

**💡 创新点**

创新点在于将图同态与 CSP 的统一代数方法相结合，利用弧一致性等可实现算法作为教学切入点，阐述了从原始问题到高层抽象（pp 定义、解释、构造）以及最近的二分性与多项式时间可解性理论。

**🔧 技术方法**

主要技术包括：图同态与弧一致性算法、k 一致性、原子正交定义/解释/构造、极大多项式时间性与 Datalog、吸收理论、Siggers 和循环多项式等代数工具。

**📊 数据集**

未使用具体数据集，内容以理论阐述与示例图为主。

**📈 对比分析**

无实验对比，主要通过理论证明与算法复杂度分析来评估方法。

**⚠️ 局限性**

局限性包括：未给出统一的多项式时间算法覆盖所有可解 CSP；课程范围局限于有限域结构，对无限域 CSP 未作深入讨论；部分关键结果仍依赖未解的开放问题。

---

## 274. Qute: Towards Quantum-Native Database

**arXiv ID:** 2602.14699 | [PDF](https://arxiv.org/pdf/2602.14699v1)

**作者:** Muzhi Chen `[一作]` (Shanghai Jiao Tong University), Fan Wu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

无法获取论文内容，无法说明具体做了什么。

**💡 创新点**

无法获取创新点。

**🔧 技术方法**

无法获取所用技术。

**📊 数据集**

无法获取所用数据集。

**📈 对比分析**

无法获取比较方法与性能评估。

**⚠️ 局限性**

无法获取限制。

---

## 275. Prototype Instance-semantic Disentanglement with Low-rank Regularized Subspace Clustering for WSIs Explainable Recognition

**arXiv ID:** 2602.14501 | [PDF](https://arxiv.org/pdf/2602.14501v1)

**作者:** Chentao Li `[一作]` (Columbia University), Pan Huang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出端到端的 PID‑LRSC 框架，利用低秩正则子空间聚类和原型实例‑语义解耦，对全切片图像（WSI）进行可解释的肿瘤识别。

**💡 创新点**

创新点在于（1）两阶段低秩约束聚类剔除冗余非肿瘤特征；（2）采用增强对比学习中的分布差异（CFD）实现原型驱动的实例‑语义分离；（3）将上述两部分融合为一体化端到端优化流程。

**🔧 技术方法**

使用了深度子空间聚类、低秩投影、核化分布差异（CFD）、Swin‑Transformer 特征提取、多实例学习聚合等技术。

**📊 数据集**

使用了两公开数据集 DHMC‑KIDNEY、DHMC‑LUNG 以及两私有数据集 AMU‑CSCC、AMU‑LSCC，涵盖肺、肾、头颈等多中心组织。

**📈 对比分析**

与 ABMIL、CLAM、TransMIL、DGR‑MIL、ACMIL‑MHA 等多种 SOTA MIL 方法在 ACC、AUC 上进行对比，PID‑LRSC 在 AMU‑LSCC、AMU‑CSCC、DHMC‑KIDNEY、DHMC‑LUNG 均取得最高指标，提升幅度约 3–6% 不等。

**⚠️ 局限性**

局限性在于聚类固定为三类（肿瘤、非肿瘤、背景），难以应对更细粒度或多类别任务；对不同组织类型的跨中心通用性需要进一步验证；模型相对复杂，计算成本较高。

---

## 276. Optimal-Time Mapping in Run-Length Compressed PBWT

**arXiv ID:** 2602.13461 | [PDF](https://arxiv.org/pdf/2602.13461v1)

**作者:** Paola Bonizzoni `[一作]` (University of Milano-Bicocca), Younan Gao `[通讯]` (University of Milano-Bicocca)

**通讯引用:** 15 | [OpenAlex ID](https://openalex.org/A5071941755)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

论文提出了一种在仅占 O(r) 词（r 为 PBWT 中所有列的总跑数）空间的“move 结构”数据结构，能够在常数时间内完成 PBWT 的前向和后向步进操作，并基于此实现了高效的前缀搜索和单个单倍型检索。

**💡 创新点**

创新点在于：①将 PBWT 扩展到多等位基因（任意大小的有序字母表）并通过子跑（sub‑run）分解实现步进操作；②设计了满足“三重重叠约束”的归一化算法，使得每个子跑最多与三条前一列子跑重叠，从而实现常数时间的映射；③在 O(r) 词空间内实现前向/后向步进、rank/select 以及前缀搜索和检索的完整体系。

**🔧 技术方法**

使用技术包括：PBWT 与前缀数组（PA）的构造；跑数压缩（μ‑PBWT）与子跑分解；三重重叠约束归一化算法；预处理预decessor 查询、rank/select 结构；以及“move 结构”风格的子跑间映射表。

**📊 数据集**

实验上未给出具体数据集，论文主要面向理论分析，所述方法可直接应用于常见的大规模单倍型面板（如英国生物银行 UK Biobank）的 PBWT 表示。

**📈 对比分析**

与以往基于 μ‑PBWT 的实现相比，过去只能在 O(loglog(n/r)) 时间内完成步进，而本工作在 O(r) 词空间下实现了 O(1) 步进；前缀搜索在 O(m'·loglogσ) 时间内完成，检索单倍型在 O(r + loglog r) 时间完成。实验对比未给出，但理论上显著降低了时间复杂度。

**⚠️ 局限性**

局限性：①论文仅给出理论证明，缺乏实际实现与实验评估；②对多等位基因的字母表大小 σ 的影响未深入讨论，极大 σ 时的空间/时间开销可能不佳；③仍假设单倍型长度相同或已做适配，处理不同长度单倍型的通用性需要额外工作。

---

## 277. LRD-MPC: Efficient MPC Inference through Low-rank Decomposition

**arXiv ID:** 2602.14397 | [PDF](https://arxiv.org/pdf/2602.14397v1)

**作者:** Tingting Tang `[一作]` (University of Southern California), Murali Annavaram `[通讯]` (University of Southern California)

**通讯引用:** 6470 | [OpenAlex ID](https://openalex.org/A5018033573)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多方安全计算（MPC）环境下，对机器学习模型的线性层（卷积层、全连接层）进行低秩分解，并提出截断跳过和高效线性层拼接两种优化技术，显著提升推理速度、能耗和离线阶段成本。

**💡 创新点**

创新点在于：①将截断操作跳过，只在最终乘法后进行一次截断，从而消除低秩分解带来的额外通信与计算；②通过对连续两层线性运算进行流水化处理，将额外的通信轮次隐藏起来，进一步减少延迟；③综合运用这两种技术，实现了低秩分解在MPC中的完整收益。

**🔧 技术方法**

使用的技术包括：秘密共享（加法共享）与 Beaver 三元组；低秩矩阵分解；固定点截断协议；流水化通信与计算的管线化；在 GPU 上实现并行矩阵乘法与截断；实验环境基于 CrypTen（n 协议）和 PIGEON（Trio 3 协议）。

**📊 数据集**

在四个典型模型上进行实验：VGG19、WideResNet18、Graph Convolutional Network (GCN)、BERT‑Large。使用公开数据集（ImageNet、CIFAR‑10/100、Cora/Planetoid、GLUE/SQuAD 等）进行推理测试。

**📈 对比分析**

与基线（全秩模型）和仅低秩分解无优化的基线进行对比。在线阶段速度提升最高可达 33%（n 协议）或 52%（Trio 3 协议）；能耗下降 52%；离线阶段加速 88%；在不同网络环境（LAN/MAN/WAN）下均保持显著优势，低秩分解在高带宽低延迟环境下表现尤为突出。

**⚠️ 局限性**

局限性：①低秩分解的秩选择可能泄露矩阵尺寸信息；②对非线性层比例高的模型提升有限；③在激活函数多的网络中，流水化优化效果受限；④截断跳过虽对精度影响小，但在极低精度场景下可能导致误差积累；⑤目前仅针对半诚实协议，未验证对主动攻击的鲁棒性。

---

## 278. OMGs: A multi-agent system supporting MDT decision-making across the ovarian tumour care continuum

**arXiv ID:** 2602.13793 | [PDF](https://arxiv.org/pdf/2602.13793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 279. Enabling Option Learning in Sparse Rewards with Hindsight Experience Replay

**arXiv ID:** 2602.13865 | [PDF](https://arxiv.org/pdf/2602.13865v1)

**作者:** Gabriel Romio `[一作]` (Universidade do Vale do Rio dos Sinos), Gabriel de Oliveira Ramos `[通讯]` (Universidade do Vale do Rio dos Sinos)

**通讯引用:** 1317 | [OpenAlex ID](https://openalex.org/A5000440246)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文在多目标稀疏奖励环境中，先把Hindsight Experience Replay (HER)集成到Multi‑Updates Option Critic (MOC)框架，得到MOC‑HER；随后提出Dual Objectives HER (2HER)，通过为对象状态和执行器位置分别设定虚拟目标，进一步提升了对象操控能力。

**💡 创新点**

创新点在于：①把HER直接应用于MOC，实现稀疏奖励下的选项学习；②设计2HER双目标回放机制，既鼓励与对象交互，又完成任务目标，显著提升了对象操控任务的成功率。

**🔧 技术方法**

技术方法包括：多更新选项策略梯度 (MOC)、HER与2HER回放、离线/非策略强化学习、双目标奖励融合与权重调节。

**📊 数据集**

实验数据集为Gymnasium Robotics的Fetch系列环境：FetchReach、FetchPush、FetchSlide、FetchPickAndPlace。

**📈 对比分析**

与标准MOC、IOC、MOC‑HER、IOC‑HER以及IOC‑2HER等基线相比，MOC‑2HER在FetchPush、FetchSlide、FetchPickAndPlace等任务中成功率分别提升至约90%、80%和40%，远超基线（大约11%–13%）。

**⚠️ 局限性**

局限性包括：仅在固定对象位置的静态场景下验证；方法仅适用于离线/非策略算法；仅限于可定义终点的目标型环境，且需预设固定选项数量。

---

## 280. Characterizing Robustness of Strategies to Novelty in Zero-Sum Open Worlds

**arXiv ID:** 2602.14278 | [PDF](https://arxiv.org/pdf/2602.14278v1)

**作者:** Mayank Kejriwal `[一作]` (Information Sciences Institute), Hongyu Li `[通讯]` (Information Sciences Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究固定策略代理在零和开放世界环境中的鲁棒性，使用迭代囚徒困境和德州扑克牌两种游戏，通过注入不同的规则或支付矩阵变异进行实验。

**💡 创新点**

提出了通用的矩阵框架以及两种度量指标（Per-Agent Robustness 与 Global Impact）来量化新奇对固定策略的影响，并在两个完全不同的领域做了系统性跨域实验。

**🔧 技术方法**

采用模拟对局、对角矩阵表示、平均胜率/现金比例作为比赛得分，并使用统计检验（t检验）进行显著性分析，实验实现基于Python的模拟环境与Axelrod库。

**📊 数据集**

使用30个IPD代理（来自Axelrod库）与10个扑克代理，配合20种IPD支付矩阵新奇和5种扑克规则新奇（如手牌重排、牌库限制等）进行完整的对局数据收集。

**📈 对比分析**

通过箱线图、热图等可视化对比各代理的鲁棒性和全局影响；实验显示大多数IPD代理在不同新奇下表现差异显著，部分策略保持鲁棒；扑克新奇对全局影响相对较小，只有少数新奇导致竞争层级显著变动。

**⚠️ 局限性**

研究仅限两人零和游戏；新奇为离散且人工设定，缺乏连续或多维变化；未考虑多代理/协作环境；实验结果只能描述而非预测新奇对策略的影响。

---

## 281. ROSA: Roundabout Optimized Speed Advisory with Multi-Agent Trajectory Prediction in Multimodal Traffic

**arXiv ID:** 2602.14780 | [PDF](https://arxiv.org/pdf/2602.14780v1)

**作者:** Anna-Lena Schlamp `[一作]` (AImotion Bavaria), Stefanie Schmidtner `[通讯]` (AImotion Bavaria)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了ROSA系统，结合Transformer多代理轨迹预测与实时主动速度建议，旨在提升混合交通（车辆+VRU）在环形交叉口的安全性和效率。

**💡 创新点**

创新点：1) 首次在环形交叉口对车辆与VRU共同进行多代理预测；2) 采用单步训练的Transformer并以自回归方式生成确定性轨迹，直接用于可操作的速度建议；3) 引入行驶路线意图（Exit Intention）显著提升预测精度；4) 基于占用预测的两阶段速度优化，兼顾安全与效率；5) 在真实数据上进行大规模仿真验证并公开实现代码。

**🔧 技术方法**

技术手段：Transformer编码器（自注意力）+多层感知器预测头；自回归多步预测；占用预测与二分类评估；基于物理方程的速度计算算法；SUMO微观仿真进行效率与安全评估。

**📊 数据集**

数据集：openDD（drone记录30 Hz轨迹），专门挑选的rdb1环形子集，包含VRU优先交叉口，并扩展为包含路线意图信息。

**📈 对比分析**

对比方法：与多种基准模型（Vanilla‑TF、IAMP、GCN、GSG‑Former、AMENet 等）在ADE/FDE上比较；在5 s预测窗内，基线模型ADE 5.08 m / FDE 10.51 m；加入动态特征后 ADE 1.29 m / FDE 2.99 m；再加路线意图后 ADE 1.10 m / FDE 2.36 m；占用预测精准率 > 0.85；ROSA仿真中能耗降低约 10 %/17 %（BEV），等待时间和停顿率下降 60 %–95 %。

**⚠️ 局限性**

局限性：1) VRU数据极不平衡，导致占用预测在较长预测窗出现误报/漏报；2) 只验证单车辆单一环形交叉口，缺乏多车协同与多布局泛化；3) 对通信延迟、V2X不确定性分析有限；4) 真实环境实验仍待开展。

---

## 282. Extending Multi-Source Bayesian Optimization With Causality Principles

**arXiv ID:** 2602.14791 | [PDF](https://arxiv.org/pdf/2602.14791v1)

**作者:** Luuk Jacobs `[一作]` (Radboud University), Mohammad Ali Javidian `[通讯]` (Appalachian State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种将因果贝叶斯优化(CBO)与多源贝叶斯优化(MSBO)相结合的框架 MSCBO，以在多源环境下高效地执行干预并降低成本。

**💡 创新点**

创新点在于将因果结构用于最小干预集裁剪，并通过成本敏感的知识梯度选择最佳信息源，实现因果与多源信息的联合利用。

**🔧 技术方法**

使用技术包括因果图（DAG）、高斯过程、成本敏感知识梯度（CKG）、ϵ-贪婪策略及最小干预集算法。

**📊 数据集**

实验数据集涵盖了PSA（前列腺特异抗原）因果网络与E.coli基因调控网络，并在多种噪声/结构变化场景下进行评估。

**📈 对比分析**

与单源CBO、单源MSBO以及传统BO对比，MSCBO在大多数场景下实现了更好的成本效率和收敛性能，尤其在可裁剪的高维网络中优于其它方法。

**⚠️ 局限性**

主要局限包括计算开销高、需要已知结构方程、每源独立GP导致信息流受限，且对离散网络的适应性不足。

---

## 283. LiSFC-Search: Lifelong Search for Network SFC Optimization under Non-stationary Drifts

**arXiv ID:** 2602.14360 | [PDF](https://arxiv.org/pdf/2602.14360v1)

**作者:** Zuyuan Zhang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6395 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种用于云-网络融合环境下动态服务函数链（SFC）放置与调度的终身学习规划框架LiSFC，利用MCTS与图漂移度量实现跨网络配置的知识迁移。

**💡 创新点**

创新点在于：① 将SFC规划建模为一族相关MDP，定义能上界MDP距离的图漂移度量；② 在LiZero基础上提出可迁移的自适应UCT（aUCT）奖金，使MCTS能安全地复用历史搜索统计；③ 结合网络差异估计实现搜索效率与鲁棒性的双重提升。

**🔧 技术方法**

主要技术包括：MCTS/UMCTS、LiZero的Lipschitz终身规划、图谱漂移度量、可迁移aUCT上界、基于重要性采样的MDP距离估计以及网络资源调度约束。

**📊 数据集**

使用合成的计算力网络（CPN）拓扑与SFC工作负载（20节点、40条链路，Poisson流量，3-5个VNF），并构造多种图漂移场景（升级、降级、混合）进行实验。

**📈 对比分析**

与贪婪启发式、非迁移MCTS以及纯学习基线对比，LiSFC在阻塞概率和95%尾延迟上明显优于对手，同时在小图漂移情况下将MCTS模拟次数减少约30%-50%，在大漂移时退化到标准MCTS表现，确保不劣于对照组。

**⚠️ 局限性**

局限性包括：实验仅在合成数据上验证；图漂移度量需手工设定权重；在极端网络变更或大规模真实网络中，估计误差与计算开销可能提升；缺乏对动态工作负载自适应调度的深入评估。

---

## 284. Beyond Retractions: Forensic Scientometrics Techniques to Identify Research Misconduct, Citation Leakage, and Funding Anomalies

**arXiv ID:** 2602.14793 | [PDF](https://arxiv.org/pdf/2602.14793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 285. AISA: Awakening Intrinsic Safety Awareness in Large Language Models against Jailbreak Attacks

**arXiv ID:** 2602.13547 | [PDF](https://arxiv.org/pdf/2602.13547v1)

**作者:** Weiming Song `[一作]` (Beijing University of Technology), Ruiping Yin `[通讯]` (Beijing University of Technology)

**通讯引用:** 316 | [OpenAlex ID](https://openalex.org/A5086582441)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AISA，一种在单次前向推理中激活大型语言模型内部安全意识的防御框架，利用注意力头激活信息生成安全风险评分并按风险调整生成 logits，从而阻止 jailbreak 攻击，同时保持原模型实用性。

**💡 创新点**

① 把安全性视为模型内在可被激活的特征；② 在注意力头输出空间使用线性探测器提取安全信号；③ 自动挑选最具判别力的头集，避免冗余计算；④ 通过风险评分进行动态 logits steering，既能拒绝高危请求又能保留低危输出；⑤ 仅一次前向，无参数修改、无额外模块，兼具检测与防御。

**🔧 技术方法**

线性分类探测器、spatiotemporal 分析、自动头排名与选择、风险评分与阈值化的 logits 调整（logits steering）以及阈值参数调优。

**📊 数据集**

自制 ALL‑4 数据集（包含 EJ‑OO、FQ‑PH、WJB‑s、JBC 等多种攻击样本）作为训练/验证集；测试集涵盖 13 个安全评测数据集（HB、ADVB、SB、XST、OKT、WGT、L3J 等）以及 4 个通用能力基准（GSM8K、BoolQ、MMLU、MMLU‑Pro）。

**📈 对比分析**

与 9 种检测模型（OpenAI GPT‑4o mini、GPT‑4.1 mini、GPT‑5 mini、Llama‑Prompt‑Guard‑2、Jailbreak‑Classifier、NemoGuard、SPDetector、GradSafe 等）以及 5 种防御基线（ICD、SAGE、SafeDecoding、SCANS、SelfDefenD）进行对比。AISA 在 13 个检测数据集上的平均准确率达 0.92，超过大多数商业 API；在防御任务中 SR 分数均低于 2，False Refusal 近似与未加防御模型一致，保持了原模型的通用能力。

**⚠️ 局限性**

需要额外训练少量探测器并依赖手工调参（阈值、头数）；对极端或未知攻击的鲁棒性尚未完全验证；仅针对文本输入，无法直接处理代码/多模态或参数改动类攻击；在对齐度极低或大规模模型上可能需要更多头或更细粒度调优。

---

## 286. Path Planning Optimisation for SParse, AwaRe and Cooperative Networked Aerial Robot Teams (SpArC-NARTs): Optimisation Tool and Ground Sensing Coverage Use Cases

**arXiv ID:** 2602.14247 | [PDF](https://arxiv.org/pdf/2602.14247v1)

**作者:** Maria Conceição `[一作]`, Meysam Basiri `[通讯]` (Instituto for Systems and Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了一种面向稀疏通信网络无人机团队的离线轨迹优化工具SpArC‑NART，旨在在有限能量、感知不确定性和间歇通信条件下最大化目标探测概率、降低报告延迟并提升全局态势感知。

**💡 创新点**

创新点在于：①提出基于价值移动（VoM）与通信强度指数的软约束动态奖励机制，将探索与通信需求自适应平衡；②将角色基行为循环与通信模型耦合，实现对不同NART构成（无人机单体、静态/移动外部实体）的灵活配置；③在地面感知覆盖用例中实现了多策略（非合作、机会合作、计划合作）的对比实验。

**🔧 技术方法**

采用了模拟退火（SA）进行轨迹优化，利用Friis传播模型生成CSI并通过sigmoid平滑；构建了VoM与CSI的乘积奖励函数；实现了信息共享与知识更新算法；同时使用角色权重（w1,w2）对不同合作类型进行调节。

**📊 数据集**

使用自定义仿真环境：两种概率包含（POC）分布（均匀与非均匀）构建的网格地图；四种用例（多无人机组、静态外部实体、两种移动外部实体）以及5次仿真测试，未使用公开数据集。

**📈 对比分析**

通过与基准非合作MIPP方案对比，实验表明：在目标检测指标（E、TPOC、EP）上合作方案略逊于基准（≤13%差距）；但在报告延迟（↓90%）和全局态势感知（↑67%）方面显著优于基准；不同NART构成下的性能差异与先验知识、角色权重密切相关。

**⚠️ 局限性**

局限性包括：①仅进行离线规划，缺乏在线自适应与即时重规划；②未显式规划会面点，合作机会依赖于随机接近，可能导致效率低下；③通信模型简化为二维，未考虑复杂的多路径/阻塞效应；④对大规模队伍和能量分配的可扩展性待验证；⑤实验仅在小型仿真环境中进行，缺乏真实场景验证。

---

## 287. VAR-3D: View-aware Auto-Regressive Model for Text-to-3D Generation via a 3D Tokenizer

**arXiv ID:** 2602.13818 | [PDF](https://arxiv.org/pdf/2602.13818v1)

**作者:** Zongcheng Han `[一作]` (Soochow University), Yu Hong `[通讯]` (Soochow University)

**通讯引用:** 25755 | [OpenAlex ID](https://openalex.org/A5110827204)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于视角感知的自回归三维生成框架VAR-3D，能够将文本描述直接转换为高质量三维模型。

**💡 创新点**

核心创新在于：①设计了视角感知的3D VQ‑VAE，对多视角图像进行交叉注意力和多尺度融合，显著降低信息压缩和视角不一致问题；②引入渲染监督训练策略，使自回归模型的离散序列预测与可视化重建同步优化，弥补了传统两阶段训练的表征-生成不匹配。

**🔧 技术方法**

采用的主要技术包括：多视角卷积编码器与自注意力交叉机制、三平面（triplane）离散量化的VQ‑VAE、GPT‑2风格自回归解码器、Gumbel‑Softmax STE渲染监督、以及多尺度量化和判别器的GAN损失。

**📊 数据集**

训练与评估使用G‑Objaverse高质量子集（约10万三维资产）作为数据集，文本提示来自3D‑Topia的描述。

**📈 对比分析**

在多项指标上均优于现有方法：PSNR最高18.52，SSIM 0.840，CLIP‑T 27.42，FID 32.74，KID 0.396，LPIPS 0.176；在重建任务中PSNR 28.97、SSIM 0.938、FID 30.50、KID 0.140，显著提升了几何和视觉质量。

**⚠️ 局限性**

局限性包括：仍采用两阶段训练而非端到端；仅支持文本条件，无法处理图像或多模态输入；生成质量受体积渲染限制，未来可探索更高效的三维表示与大规模训练。

---

## 288. Learning Structural Hardness for Combinatorial Auctions: Instance-Dependent Algorithm Selection via Graph Neural Networks

**arXiv ID:** 2602.14772 | [PDF](https://arxiv.org/pdf/2602.14772v1)

**作者:** Sungwoo Kang `[一作]` `[通讯]` (Korea University), Sungwoo Kang (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出通过学习判别实例难度来实现赢家判定问题（WDP）的算法选择，设计了一个20维结构特征的轻量级MLP硬度预测器、针对鲸鱼‑鱼陷阱的HeteroGAT专家以及混合分配器，实现了在不同实例上的自适应求解；

**💡 创新点**

创新点在于把机器学习的角色从“直接替代求解器”转变为“实例分诊”，通过构造鲸鱼‑鱼结构化难度实例并训练专用GNN，同时用轻量级特征预测器快速识别难易实例，实现高效的算法组合；

**🔧 技术方法**

技术方面使用20维结构特征提取、三层MLP回归/分类、HeteroGAT异构图注意力网络、传统Greedy启发式以及Gurobi ILP求解器；

**📊 数据集**

数据集涵盖CATS任意分布、MIS Erdős‑Rényi与星形陷阱、以及自定义的鲸鱼‑鱼陷阱生成实例；

**📈 对比分析**

实验对比显示：在混合分布下，混合分配器整体优化误差仅0.51%，在鲸鱼‑鱼陷阱上HeteroGAT几乎无误差（≈0%），而在标准CATS基准上GNN的误差（0.45–0.71%）不及Gurobi 10 ms/100 ms/1 s的表现；

**⚠️ 局限性**

主要局限包括：需要ILP求解的真实最优解做为训练标签、实验仅覆盖单一类型的结构化难度（鲸鱼‑鱼陷阱）、未在真实拍卖数据上验证，且GNN在通用实例上未能击败传统ILP求解器。

---

## 289. Conformal Signal Temporal Logic for Robust Reinforcement Learning Control: A Case Study

**arXiv ID:** 2602.14322 | [PDF](https://arxiv.org/pdf/2602.14322v1)

**作者:** Hani Beirami `[一作]` (University of Tehran), M M Manjurul Islam `[通讯]` (Ulster University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究将基于Signal Temporal Logic（STL）的自适应安全屏蔽嵌入到强化学习（PPO）控制器中，以提升F‑16发动机推力控制的可靠性。

**💡 创新点**

创新点是首次将在线合成的 conformal STL 屏蔽与RL控制器结合，实现分布无关的概率安全保证，并在严格扰动场景下保持高满足率。

**🔧 技术方法**

采用的技术包括Proximal Policy Optimization（PPO）、Signal Temporal Logic（STL）监测、分布无关预测（Conformal Prediction）以及运行时安全屏蔽。

**📊 数据集**

使用公开的 AeroBench F‑16 高保真仿真平台作为数据集。

**📈 对比分析**

对比三种模式（PPO、传统 STL 屏蔽、Conformal STL 屏蔽）在四个递进扰动场景下，Conformal STL 屏蔽在最严苛场景下保持 95% 满足率且正鲁棒性，显著优于未屏蔽的 60% 满足率。

**⚠️ 局限性**

局限性在于仅验证于单一F‑16仿真环境，未评估真实飞行或多机群场景；并且对预测模型的在线再校准缺乏深入研究，限制了在大幅分布漂移下的适应性。

---

## 290. An Ensemble Learning Approach towards Waste Segmentation in Cluttered Environment

**arXiv ID:** 2602.13681 | [PDF](https://arxiv.org/pdf/2602.13681v1)

**作者:** Maimoona Jafar `[一作]` (National University of Sciences and Technology), Shah Khalid `[通讯]` (National University of Sciences and Technology)

**通讯引用:** 698 | [OpenAlex ID](https://openalex.org/A5006732432)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于U‑Net和FPN的集成学习模型（EL‑4），用于在混乱环境下对ZeroWaste‑f数据集进行废弃物像素级分割，并通过预处理、数据增强和EfficientNet编码器提升模型性能。

**💡 创新点**

创新点在于将U‑Net的细节捕捉能力与FPN的多尺度特征优势通过加权平均融合，并探索不同EfficientNet编码器（B0‑B4）对性能的影响，实现了高IoU和低Dice Loss的平衡。

**🔧 技术方法**

技术包括：U‑Net、FPN两种语义分割架构、EfficientNet编码器、预训练与数据增强、Dice Loss损失函数、Adam优化器以及Tensor‑T4 GPU训练。

**📊 数据集**

使用了公开的ZeroWaste‑f废弃物分割数据集，该数据集包含高噪声、形变和重叠的废弃物图像。

**📈 对比分析**

对比实验表明：U‑Net单模型IoU为0.8065、Dice Loss为0.2084；FPN单模型IoU为0.7953、Dice Loss为0.1183；而集成模型EL‑4在测试集上达到IoU 0.8306、Dice Loss 0.09019，显著优于单一模型。

**⚠️ 局限性**

局限性包括：模型对计算资源仍有一定需求（需EfficientNet-B4+两网络），对极端形变或极度遮挡的废弃物仍可能出现分割误差；以及在不同工厂实际环境中的泛化能力尚需进一步验证。

---

## 291. UAVGENT: A Language-Guided Distributed Control Framework

**arXiv ID:** 2602.13212 | [PDF](https://arxiv.org/pdf/2602.13212v1)

**作者:** Ziyi Zhang `[一作]` (Carnegie Mellon University), Yorie Nakahira `[通讯]` (Carnegie Mellon University)

**通讯引用:** 455 | [OpenAlex ID](https://openalex.org/A5084147009)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个三层架构，结合自然语言指令、LLM 监督与分布式控制，实现多无人机在动态环境中执行高层任务并保持稳健性。

**💡 创新点**

创新点在于将 LLM 作为运行时监督者，周期性验证并纠正执行偏差，同时在低层提供正式的稳健性与稳定性保证，打破传统编译一次式语言‑控制闭环。

**🔧 技术方法**

使用技术包括大语言模型（LLM）监督、自然语言接口、基于相对信息的分布式控制算法、误差动态分析和理论稳健性证明。

**📊 数据集**

未使用公开数据集，全部在仿真环境中进行测试，场景包括警察追击、森林搜救等自建地图与目标轨迹。

**📈 对比分析**

通过仿真对比人类单独指令与 LLM 监督下的执行，结果表明 LLM 监督显著降低人类干预频率，保持追踪误差在预设阈值内并能及时恢复形成；理论分析给出了误差上界。

**⚠️ 局限性**

局限性包括未结合视觉‑语言多模态模型、仅支持三类人意图、缺乏真实硬件验证与长时间运行测试、以及对极端动态多目标场景的鲁棒性仍待进一步验证。

---

## 292. From Prompt to Production:Automating Brand-Safe Marketing Imagery with Text-to-Image Models

**arXiv ID:** 2602.13349 | [PDF](https://arxiv.org/pdf/2602.13349v1)

**作者:** Parmida Atighehchian `[一作]` (Amazon Web Services), Negin Sokhandan `[通讯]` (Amazon Web Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个可扩展的全自动化管道，利用文本到图像模型生成符合品牌规范的营销图像，并通过结构化提示解析、资产检索、组合规划与多模态质量评估确保高质量输出；

**💡 创新点**

创新点包括：①将自然语言提示分解为机器可读规范并用LLM进行背景验证；②多模态组合规划与网格搜索生成多样化的图像变体；③端到端的智能质量控制，使用多模态LLM评估以及视觉质量指标；④将人类审阅仅限于最终选择，显著减少人工干预；

**🔧 技术方法**

使用的大语言模型（LLM）进行提示解析、背景验证与质量评估；CLIP、DINOv2、MS-SSIM、HPSv2等视觉评估指标；采样网格与旋转策略；以及现有生成模型如ICLight、Inpaint Anything、Amazon Nova Canvas 作为基础生成器；同时使用SAM2和Amazon Nova Pro进行产品分割；

**📊 数据集**

采用Amazon-Berkeley Objects (ABO) 产品图像数据集，并用视觉语言模型生成每个产品的5条创意场景说明，共计1030个产品-说明对；

**📈 对比分析**

与三种基线模型（ICLight、Inpaint Anything、Nova Canvas）在同一1030组产品-说明对上进行比较，使用DINOv2、MS-SSIM和HPSv2评估产品保真度、整体对齐和视觉质量；结果显示管道在DINOv2和MS-SSIM上显著提升（p<0.05），人类评估显示品牌一致性提升30.77%，人类偏好提升52%；

**⚠️ 局限性**

局限性包括：ABO数据集存在少量错误标注导致偶发误差；基础生成模型易产生幻觉、忽略小物体或改变纹理；对小尺寸物体识别不佳；质量控制模块未能覆盖所有失效场景，需进一步完善。

---

## 293. QuRL: Efficient Reinforcement Learning with Quantized Rollout

**arXiv ID:** 2602.13953 | [PDF](https://arxiv.org/pdf/2602.13953v1)

**作者:** Yuhang Li `[一作]` (Yale University), Brucek Khailany `[通讯]` (NVIDIA Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 QuRL，通过在强化学习训练中使用量化的 actor 进行 rollout，加速推理并保持梯度更新的完整性。

**💡 创新点**

提出 Adaptive Clipping Range（自适应裁剪范围）解决量化导致的训练崩溃，并提出 Update-Aware Quantization（更新感知量化）利用不变缩放平衡量化误差与权重更新，显著提升量化训练的稳定性与性能。

**🔧 技术方法**

使用 INT8 / FP8 低比特量化、Invariant Scaling、Decoupled PPO、TIS 以及 vLLM 的高效矩阵乘法内核，结合量化后的权重与激活进行 rollout。

**📊 数据集**

在 DeepScaleR、AIME 2024、GSM8K、MATH 500、AMC 2023 等数学推理数据集上进行实验，涵盖 PPO、GRPO、DAPO 三种 RL 算法。

**📈 对比分析**

与全精度 BF16 强化学习和 FlashRL（TIS+Decoupled PPO）相比，QuRL 在 INT8 量化下平均准确率仅低 1-2%（如 GSM8K 上从 55.35% 降至 53.55%），且在多 GPU 上通过量化实现 20%–80% 的 throughput 加速；在 AIME 2024 上 FP8 量化下保持 33.27% 的 Avg@32，显著缩小与 BF16 的差距。

**⚠️ 局限性**

仍受限于量化误差导致的准确率下降，尤其在更大规模模型或极低精度下表现不稳定；量化前需要一次性缩放 calibration，且在不同训练与推理框架间仍存在细微差异，进一步提升算法鲁棒性仍需研究。

---

## 294. DMESR: Dual-view MLLM-based Enhancing Framework for Multimodal Sequential Recommendation

**arXiv ID:** 2602.13715 | [PDF](https://arxiv.org/pdf/2602.13715v1)

**作者:** Mingyao Huang `[一作]` (Xi'an Jiaotong University), Yan Chen `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 23535 | [OpenAlex ID](https://openalex.org/A5100378023)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 DMESR（Dual‑view MLLM‑based Enhancing framework for Sequential Recommendation）双视图增强框架，通过多模态大语言模型（MLLM）生成的粗粒度语义与原始文本的细粒度语义进行融合，从而提升顺序推荐系统的性能。

**💡 创新点**

创新点包括：① 采用三路提示（文本、视觉、混合）让 MLLM 产生多模态描述；② 利用对比学习对齐三路生成的嵌入，解决语义不一致问题；③ 引入双向交叉注意力融合粗细语义，实现信息互补；④ 所有模块可无缝集成到任意 SRS 骨干（SASRec、BERT4Rec、GRU4Rec）上。

**🔧 技术方法**

技术手段：多模态大语言模型提示生成描述；Adapter 进行降维与协同信号注入；对比学习对齐嵌入；双向交叉注意力实现语义融合；在底层使用 Transformer/GRU 等顺序模型进行推荐。

**📊 数据集**

实验使用三大真实数据集：MovieLens、Yelp、Amazon Games，分别包含数千用户、数万商品，覆盖电影、餐饮与游戏等领域。

**📈 对比分析**

与 X‑REFLECT、CPMM、QARM 等最新 MLLM‑增强基线在三种 SRS 骨干上对比，DMESR 在 H@10/N@10 上平均提升约 10%–30%，在长尾商品上同样表现最佳，验证了框架的普适性与高效性。

**⚠️ 局限性**

局限性：仍需依赖 MLLM 进行离线生成描述，推理阶段成本较高；对极度稀疏或冷启动场景的进一步优化尚未完成；在极大规模实时系统中，模型的部署与加速仍是挑战。

---

## 295. ATTest: Agent-Driven Tensor Testing for Deep Learning Library Modules

**arXiv ID:** 2602.13987 | [PDF](https://arxiv.org/pdf/2602.13987v1)

**作者:** Zhengyu Zhan `[一作]` (Nanjing University), Zhenyu Chen `[通讯]` (Nanjing University)

**通讯引用:** 7031 | [OpenAlex ID](https://openalex.org/A5100422933)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了ATTest框架，利用七阶段管线在深度学习库（PyTorch、TensorFlow）中自动生成、验证并修复模块级单元测试用例。

**💡 创新点**

创新点包括将LLM视为自主管理的代理，在约束感知生成、增量块级修复和阶段化工作流之间实现动态协同，以突破传统SBST的语义盲区和LLM的上下文饱和问题。

**🔧 技术方法**

采用大语言模型（LLM）进行代码合成与分析、搜索式软件测试技术、约束提取与执行反馈机制、增量块级编辑、持续状态管理与代理调度。

**📊 数据集**

使用了来自PyTorch和TensorFlow的165个Python模块（覆盖核心API），作为评测数据集。

**📈 对比分析**

与PynguinML基线比较，ATTest在PyTorch平均分支覆盖率55.60%、TensorFlow 54.77%，显著高于基线（PyTorch 43.13%、TensorFlow 39.72%），相对覆盖率普遍接近1.0。

**⚠️ 局限性**

局限性：目前仅针对模块级测试，尚未扩展到库级集成测试；对多硬件后端（如CUDA）支持有限；依赖LLM的生成稳定性和上下文窗口大小仍是潜在瓶颈。

---

## 296. Evaluating LLM-Generated ACSL Annotations for Formal Verification

**arXiv ID:** 2602.13851 | [PDF](https://arxiv.org/pdf/2602.13851v1)

**作者:** Arshad Beg `[一作]` (Maynooth University), Rosemary Monahan `[通讯]` (Maynooth University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对506个C程序进行了大规模实验，比较了五种ACSL生成方式（规则脚本、Frama‑C RTE、DeepSeek、GPT‑5、OLMo3）并使用Frama‑C WP与多种SMT求解器验证其正确性。

**💡 创新点**

创新点在于首次将LLM生成的ACSL与传统工具生成的ACSL在同一数据集和验证流程下进行系统对比，量化了证明成功率、超时频率及求解器稳定性，并揭示了表达能力与验证稳定性之间的权衡。

**🔧 技术方法**

采用Frama‑C框架（EVA、RTE、WP插件）、三大LLM（DeepSeek‑V3.2、GPT‑5.2、OLMo3.1 32B）以及Alt‑Ergo、CVC4、CVC5、Z3四个SMT求解器，并进行统计分析。

**📊 数据集**

使用的是2025年11月公开的CASP数据集，共506个C程序的源文件，已在HuggingFace上发布。

**📈 对比分析**

通过统一的验证条件和多次实验，比较了各生成方式在不同求解器上的平均证明成功率、超时计数和Qed时间。结果显示规则脚本和RTE几乎100%成功且变异性低；DeepSeek平均成功率≈95%并保持相对低超时；GPT‑5约81%成功且超时最多；OLMo3约83%成功，超时中等。

**⚠️ 局限性**

LLM生成的ACSL存在缺失或弱化的注解，导致求解器超时和证明不稳定；解析器鲁棒性不足；相较之下传统工具生成的ACSL更可靠，LLM方法仍需改进以提升一致性和求解兼容性。

---

## 297. Intelligence as Trajectory-Dominant Pareto Optimization

**arXiv ID:** 2602.13230 | [PDF](https://arxiv.org/pdf/2602.13230v1)

**作者:** Truong Xuan Khanh `[一作]` (H&K Research Studio), Truong Quynh Hoa `[通讯]` (H&K Research Studio)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出把智能视为轨迹级的 Pareto 优化问题，并定义了轨迹级支配关系，揭示了“Pareto 陷阱”与“动态智能天花板”的几何来源；

**💡 创新点**

核心创新在于从轨迹角度重新定义智能和支配关系，构建了 Pareto 陷阱的数学分类和 Trap Escape Difficulty Index (TEDI) 指标，解释了长期适应性停滞的结构性原因；

**🔧 技术方法**

使用多目标马尔可夫决策过程理论、轨迹级 Pareto 优化定义、几何分析和最小示例模型进行验证；

**📊 数据集**

未使用公开数据集，论文以自构造的简化离散状态空间（N+1 个状态）为最小示例进行演示；

**📈 对比分析**

没有与其他算法进行量化对比，只通过最小模型展示点式优化策略在 Pareto 陷阱中受限，而轨迹级策略能够突破；

**⚠️ 局限性**

局限性在于理论高度抽象、缺乏在复杂真实环境中的实证验证；TEDI 估计方法尚未给出，且未给出算法实现或训练范例；

---

## 298. What Do We Mean by 'Pilot Study': Early Findings from a Meta-Review of Pilot Study Reporting at CHI

**arXiv ID:** 2602.13488 | [PDF](https://arxiv.org/pdf/2602.13488v1)

**作者:** Belu Ticona `[一作]` (George Mason University), Antonios Anastasopoulos `[通讯]` (George Mason University)

**通讯引用:** 2768 | [OpenAlex ID](https://openalex.org/A5013793053)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2008‑2025年间 904 篇 CHI 论文中提及的 pilot study 进行元审查，归纳其报告结构、结果披露和对主实验的影响。

**💡 创新点**

首次系统量化 HCI 领域 pilot 研究的多样化报告方式和不足，揭示方法学盲点并为后续制定标准提供依据。

**🔧 技术方法**

使用 OpenAI GPT（LLM）进行自动注释，并通过人工验证校正，以识别报告结构与影响描述。

**📊 数据集**

基于 ACM Digital Library 的 904 篇包含 “pilot study” 关键词的完整论文数据集。

**📈 对比分析**

将 LLM 标注结果与人工核验对比，发现嵌入式方法的准确率>72%，但标题式报告的准确率<55%；整体表明 LLM 在结构识别方面可行，但仍需人工干预。

**⚠️ 局限性**

数据集缺乏统一的 pilot 定义，LLM 对不同章节层级的识别误差较大，且结果缺乏可重复性与深度，限制了结论的普适性。

---

## 299. Hunt Globally: Deep Research AI Agents for Drug Asset Scouting in Investing, Business Development, and Search & Evaluation

**arXiv ID:** 2602.15019 | [PDF](https://arxiv.org/pdf/2602.15019v1)

**作者:** Alisa Vinogradova `[一作]` (Bioptic), Andrey Doronichev `[通讯]` (Bioptic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一套基于完整性评估的药物资产挖掘基准，并设计了Bioptic Agent实现全量、无幻觉的资产搜寻

**💡 创新点**

创新点在于使用树结构的自学习探索框架、多语言并行检索、基于已验证资产的逆向查询生成以及与专家对齐的验证与去重流程

**🔧 技术方法**

主要技术包括大型语言模型（GPT‑5.2等）、多语言深度搜索、基于规则的Criteria Match Validator、Deduplication Agent、Coach Agent以及UCB树搜索策略

**📊 数据集**

使用的数据集来源于多语言区域新闻、非英语公开源的药物资产记录，以及从投资者/BD专业人士收集的真实筛选查询，最终构成22条查询‑资产对的测试集

**📈 对比分析**

与Claude Opus 4.6、Gemini 3 Pro Deep Research、GPT‑5.2 Pro、Perplexity Deep Research、Exa Websets等基线进行对比，Bioptic Agent在F1上达到79.7%，远超其他方法

**⚠️ 局限性**

局限在于仍依赖公开可检索信息，受语言覆盖范围、数据更新速度和验证器误差影响，基准可能无法覆盖所有私有或高度保密的资产

---

## 300. World Models for Policy Refinement in StarCraft II

**arXiv ID:** 2602.14857 | [PDF](https://arxiv.org/pdf/2602.14857v1)

**作者:** Yixin Zhang `[一作]` (Chinese Academy of Sciences), Bo Xu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了StarWM，即首个可学习的基于文本的StarCraft II世界模型，用以预测在偏见观测下的未来状态；

**💡 创新点**

创新点在于：1) 通过将观测拆分为五个语义模块实现动态因子化；2) 构造SC2‑Dynamics‑50k指令调优数据集；3) 设计多维离线评估框架与Generate–Simulate–Refine决策循环；

**🔧 技术方法**

采用大型语言模型Qwen3-8B/32B作为基础，利用LoRA微调训练世界模型，并在Agent中实现文本生成、模拟与动作细化；

**📊 数据集**

使用SC2‑Dynamics‑50k（约50k条轨迹，包含5秒预测窗口）作为训练与评估数据；

**📈 对比分析**

与零样本Qwen模型、静态基线及无世界模型策略对比，离线评估在经济、队列、微观与宏观维度上提升60%以上；在线对抗官方AI在Hard/Harder/VeryHard下，StarWM‑Agent分别提升30%/15%/30%胜率，供应阻塞率降低、资源转化率与战斗胜率显著提升；

**⚠️ 局限性**

局限在：1) 对敌方动态预测依赖单帧信息，精度有限；2) 需要大量文本转换与特征工程；3) 训练成本高，模型规模对计算资源有较高要求。

---

## 301. When OpenClaw AI Agents Teach Each Other: Peer Learning Patterns in the Moltbook Community

**arXiv ID:** 2602.14477 | [PDF](https://arxiv.org/pdf/2602.14477v1)

**作者:** Eason Chen `[一作]` (Carnegie Mellon University), Emmanuel Osadebe Prince `[通讯]` (GiveRep Labs)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对Moltbook平台上2.45百万名AI代理产生的28,683条有价值帖子及其评论进行教育数据挖掘，揭示了AI代理之间的同伴学习行为与互动模式。

**💡 创新点**

创新点在于首次在大规模人工智能社区中经验性描述同伴学习特征，提出验证-扩展的评论模式分类，并基于此给出了六条针对教育AI的设计原则。

**🔧 技术方法**

技术上采用了知识类型（程序式vs概念式）关键词分类、统计与可视化分析、参与度不平等测量（Gini系数）、评论文本定性编码，以及OpenClaw自动化收集与垃圾信息过滤。

**📊 数据集**

使用的数据集为Moltbook公开API收集的68,228条原始帖子，过滤后28,683条有价值帖子，包含12,123,362条评论、2.45百万注册代理。

**📈 对比分析**

与人类同伴学习社区对比，AI代理更偏向陈述（教学）而非提问，教学/提问比为11.4:1；程序式帖子平均获得约三倍评论；参与度极度不均衡（评论均值/中位数比达19.6）。

**⚠️ 局限性**

局限包括缺乏学习结果评估、仅基于关键词的知识类型分类、未能验证代理是否真正“学习”，以及高比例的自动化垃圾帖子需人工过滤。

---

## 302. Evaluating the Performance of Approximation Mechanisms under Budget Constraints

**arXiv ID:** 2602.14120 | [PDF](https://arxiv.org/pdf/2602.14120v1)

**作者:** Juan Carlos Carbajal `[一作]` (University of New South Wales), Ahuva Mualem `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在买方拥有私有估价和私有预算的单件拍卖中，近似机制相较于最优机制的收入表现；通过定义 GFOR、MVR 等度量，对不同机制类（受限、放宽、现金债券、强制激励）进行理论分析和极限构造。

**💡 创新点**

创新点主要在于：①在预算约束下首次刻画了近似机制的可行性边界，证明除有界且低于零的支撑外，任何有限或子线性菜单的机制都无法保证正比例的最优收入；②给出在价估价与预算有界且彼此正相关时，极小（poly‑log）菜单即可近似最优收入；③揭示了现金债券放宽和强制激励下的收入非单调性与无限收益差距。

**🔧 技术方法**

主要技术包括：利用收入单调性证明 GFOR/MVR 的下界；对类型空间离散化与重排构造近似机制；构造极端分布实现无限收入/无收益比；使用线性规划形式化强激励约束；以及在放宽约束下设计现金债券机制。

**📊 数据集**

本文全部基于理论模型和构造的概率分布，无实际数据集；所用“数据集”即为若干自定义的离散/连续分布。

**📈 对比分析**

与最优机制的比较以 GFOR（保证最优收入比例）和 MVR（最大放宽值）为指标；正向结果中，GFOR 可趋近 1，MVR 接近 1；负向结果中，GFOR 为 0，MVR 无界；现金债券放宽时 GFOR 甚至可无限大，展示了放宽对收益估计的潜在误导。

**⚠️ 局限性**

局限性包括：①仅处理单件、单买家的二维预算-估价模型；②结果对分布支持区间高度敏感，实用性受限；③复杂机制（如无穷菜单）在实际实施中不可行；④对预算-估价相关性的假设过于理想化，未考虑多买家竞争或多件商品的交互效应。

---

## 303. On the Semantics of Primary Cause in Hybrid Dynamic Domains

**arXiv ID:** 2602.14994 | [PDF](https://arxiv.org/pdf/2602.14994v1)

**作者:** Shakil M. Khan `[一作]` (University of Regina), Sandra Zilles `[通讯]` (University of Regina)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提出了在混合时间情景计算框架（Hybrid Temporal Situation Calculus，HTSC）中对实际因果关系的形式化定义，并给出了两种实现方法（基础式与贡献式），随后证明其等价性并引入了改进后的but‑for检验来直观验证因果性。

**💡 创新点**

创新点在于：①首次在HTSC环境下完整定义实际因果，兼顾离散与连续变化；②通过将因果归结为对上下文的贡献来桥接传统的因果与生产性视角；③提出“defused situation”概念与改进but‑for检验，解决预emption与连续时间下的预emption问题；④在理论上证明定义的基本性质与等价性。

**🔧 技术方法**

主要技术手段包括：混合时间情景计算框架、状态演化公理、情景与时间戳的逻辑规范、因果与贡献的形式化语义、对比性推理与逻辑证明。

**📊 数据集**

本文未使用实际数据集，全部通过核电厂例子（核心温度随时间变化的模拟）来说明与验证定义。

**📈 对比分析**

该工作为理论性质的研究，未与实验方法或性能指标进行对比；主要通过逻辑证明与示例演示其正确性与直观合理性。

**⚠️ 局限性**

限制包括：仅考虑线性（无并发）情景、仅关注实现因果（achievement causation）、只处理原子时间性状况、未涵盖复合效应与间接因果，且假设所有动作可执行且无外部干扰。

---

## 304. A Quasi-Experimental Evaluation of Coaching to Mitigate the Impostor Phenomenon in Early-Career Software Engineers

**arXiv ID:** 2602.13774 | [PDF](https://arxiv.org/pdf/2602.13774v1)

**作者:** Paloma Guenes `[一作]` (Pontifical Catholic University of Rio de Janeiro), Marcos Kalinowski `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

对20名早期职业软件工程师进行准实验性团队辅导干预，并测量其冒名顶替感与心理福祉变化。

**💡 创新点**

首次在软件工程环境中实证评估结构化团队辅导对冒名顶替感的缓解，并结合观察记录与多维心理测量形成混合方法研究。

**🔧 技术方法**

采用三节两小时的结构化团队辅导协议，配合CIPS、WHO‑5、SWLS、PANAS量表及非参与式观察记录。

**📊 数据集**

使用20名参与者在两支项目团队（四个子组）中的自评数据，覆盖四个时间点（T0–T3）。

**📈 对比分析**

通过等值等待列表对照设计，使用Wilcoxon、Mann–Whitney等非参数检验和效应量评估；结果显示辅导组IP略降，但对照组在观察期亦显著下降，组间差异无统计显著性。

**⚠️ 局限性**

局限性包括样本量小、未随机分组、可能的干预扩散与测量反应、仅在学术-产业实验室环境，限制了内部与外部效度。

---

## 305. Cross-household Transfer Learning Approach with LSTM-based Demand Forecasting

**arXiv ID:** 2602.14267 | [PDF](https://arxiv.org/pdf/2602.14267v1)

**作者:** Manal Rahal `[一作]` (Karlstad University), Robert Stener `[通讯]` (Thermia AB)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并验证了跨家庭传输学习框架DELTAiF，用以预测住宅热泵热水需求并实现自适应调度；

**💡 创新点**

将Transfer Learning、LSTM与Isolation Forest三种技术结合，形成可跨户适配、训练成本低且能自动生成需求日历的创新框架；

**🔧 技术方法**

采用LSTM深度序列模型、Transfer Learning微调策略、Isolation Forest异常检测以及Adam优化器与MAE损失函数；

**📊 数据集**

使用6个瑞典住宅热泵安装的实时中层温度传感器数据，约一年的秒级记录后补齐后统一采样为1分钟；

**📈 对比分析**

与传统逐户训练方式比较，采用MAPE、RMSE和R²评估；Transfer Learning将训练时间从9分钟降至约3分钟，降低67%，同时取得R² 0.874–0.991、MAPE 0.001–0.017的高预测精度；

**⚠️ 局限性**

对源户选择敏感，源户若使用规律稳定的数据能显著提升性能；固定2%异常阈值可能不适用于所有家庭；框架目前仅在住宅热泵场景验证，工业等其他场景需进一步改进。

---

## 306. Agentic Assistant for 6G: Turn-based Conversations for AI-RAN Hierarchical Co-Management

**arXiv ID:** 2602.13868 | [PDF](https://arxiv.org/pdf/2602.13868v1)

**作者:** Udhaya Srinivasan `[一作]` (Cranfield University), Weisi Guo `[通讯]` (Cranfield University)

**通讯引用:** 6040 | [OpenAlex ID](https://openalex.org/A5062362866)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一套基于多轮对话的agentic网络管理系统，支持AI‑RAN的规划、配置与调优，并集成了前端界面、知识层和仿真平台。

**💡 创新点**

提出层次化多轮对话代理与专门的HATT‑E评估框架，首次统一处理RAN与本地AI的协同管理，强调实时知识检索与工具调用。

**🔧 技术方法**

使用 Retrieval‑Augmented Generation (RAG) 大语言模型、知识图谱驱动的RAG、xApp/RIC 控制层、AI‑RAN 仿真平台（BubbleRAN/ORAN）、Next.js 前端、两阶段 AI 服务管道以及缓存/路由技术。

**📊 数据集**

构建了包含 50 个生态有效多轮场景（易/中/难）的对话数据集，结合 3GPP 标准、学术论文与工业报告的知识向量，并利用仿真平台产生的网络状态数据。

**📈 对比分析**

通过 HATT‑E 三层评估（规划可信度、工具使用准确率、端到端任务成功率）对代理进行比较，平均得分 0.6724，响应时间 13.35 s，易难差异下降 25%，单实体查询效率 94%。

**⚠️ 局限性**

主要局限在规划层委派准确率低（48%）、工具使用虽高但幻觉率 43%，在复杂推理、综合与预测任务表现不佳，缺乏完整的推理与事实根拠机制。

---

## 307. Physics Aware Neural Networks: Denoising for Magnetic Navigation

**arXiv ID:** 2602.13690 | [PDF](https://arxiv.org/pdf/2602.13690v1)

**作者:** Aritra Das `[一作]` (Ashoka University), Debayan Gupta `[通讯]` (Ashoka University)

**通讯引用:** 356 | [OpenAlex ID](https://openalex.org/A5073351462)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了基于物理约束的神经网络，利用发散无向和 E(3)-等变性约束对航空磁传感器数据进行去噪，以提升磁异常导航的精度。

**💡 创新点**

创新点在于：① 将磁场的发散无向约束通过向量势并计算其旋度实现；② 将 E(3)-等变性嵌入网络中，使用球谐展开的几何张量完成旋转与平移不变性；③ 结合连续时间神经 ODE 与长期记忆结构（LTC、Contiformer）处理不规则时间序列；④ 通过 World Magnetic Model 与条件 GAN 合成高质量的训练数据。

**🔧 技术方法**

使用的技术包括：向量势 + Curl 约束、E(3) 张量层、Neural ODE、LTC 与 Contiformer 架构、条件 GAN 数据生成、自动微分、谱正则化、AdamW 优化器等。

**📊 数据集**

数据集：真实 MagNav 飞行记录（Flt1005、Flt1006、Flt1007 等）和利用 WMM + 条件 GAN 生成的合成数据集（FltS005、FltS006、FltS007 等）。

**📈 对比分析**

通过 16 种模型的 ablation 研究，比较 RMSE 与 SNR 两项指标。加入发散无向和 E(3) 等变性约束后，Contiformer 的测试 RMSE 下降约 30%，SNR 提升 10–15 dB，整体性能显著优于无约束模型和现有最优方法。

**⚠️ 局限性**

局限性包括：① 受限于训练数据稀缺，合成数据真实性仍需进一步验证；② 对极端噪声或非线性平台干扰的鲁棒性尚未充分评估；③ 模型结构复杂，推理时的计算开销和实时性能未进行深入分析。

---

## 308. Analysis of a Cuspidal 6R Robot

**arXiv ID:** 2602.14794 | [PDF](https://arxiv.org/pdf/2602.14794v1)

**作者:** Alexander Feeß `[一作]` (OTH Regensburg), Martin Weiß `[通讯]` (OTH Regensburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种 Cuspidal 6R 机器人 Transpressor，推导并实现了其逆运动学的解析与数值解法，揭示了最多可达 16 个解，并证明了在特定参数下可以在不同解之间实现无奇异性路径。

**💡 创新点**

首次针对 6R 机器人给出完整无奇异性路径的理论与数值方法，并提供了在特殊姿态下的解析逆解，扩展了 6R 机器人的逆运动学解析范围。

**🔧 技术方法**

采用 DH 参数建模、三自由度链解析、解析几何方法、数值采样求解、雅可比行列式符号分析及线性路径验证等技术。

**📊 数据集**

无传统数据集；所有结果基于理论推导和数值模拟，使用的是机器人参数的符号与数值实例（如 a2=3, d4=2, a5=0.5）。

**📈 对比分析**

通过对比解析解与数值解的数量及无奇异性路径的存在性，证明了该方法能够在不需要广泛数值搜索的情况下完成逆解与路径规划；性能上显示在给定参数下，线性路径保持非奇异性，且求解效率高。

**⚠️ 局限性**

仍未实现整个工作空间的全解析解，参数范围有限；对不同参数组合的通用性不足；对解组分层的分类与判定仍是开放问题。

---

## 309. Removing Planner Bias in Goal Recognition Through Multi-Plan Dataset Generation

**arXiv ID:** 2602.14691 | [PDF](https://arxiv.org/pdf/2602.14691v1)

**作者:** Mustafa F. Abdelwahed `[一作]` (University of St Andrews), Joan Espasa `[通讯]` (University of St Andrews)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用 top‑k 规划器生成多条不同计划，对同一目标假设产生多条观测序列，从而构造无规划器偏差的目标识别数据集，并提出版本覆盖度（VCS）指标衡量识别器的鲁棒性。

**💡 创新点**

创新点在于：①通过 top‑k 规划器消除传统数据集中因启发式前向搜索产生的系统性偏差；②提出 VCS 指标，将同一目标下多条观测序列的识别效果综合评估，量化识别器在不同计划集合下的稳健性。

**🔧 技术方法**

主要技术包括：top‑k 规划（使用 SymK）、多版本任务生成器、观测序列随机裁剪与噪声注入、版本覆盖度计算，评估方法采用准确率、PPV、spread 等标准指标。

**📊 数据集**

使用了基于 SymK 的多版本生成器在传统 goal‑recognition 数据集（如 PDDL 任务集合）上扩展出的新数据集；对比了原始单计划数据集与新数据集的表现。

**📈 对比分析**

通过在不同观测比例（10%–100%）和鲁棒性阈值（0.1–1.0）下测试 landmark‑based 识别器，发现：在新数据集上，准确率和 PPV 随阈值升高和观测比例下降显著下降，表明即使是最先进的识别器也会受到规划器偏差影响；相比之下，旧数据集上表现更好。

**⚠️ 局限性**

局限性：①观测裁剪采用随机采样，缺乏更精细的策略；②实验仅验证了一种识别器（landmark‑based），未评估其他模型；③仍仅使用单一 top‑k 规划器，未探索不同规划器组合带来的挑战。

---

## 310. Reasoning Language Models for complex assessments tasks: Evaluating parental cooperation from child protection case reports

**arXiv ID:** 2602.14216 | [PDF](https://arxiv.org/pdf/2602.14216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 311. interID -- An Ecosystem-agnostic Verifier-as-a-Service with OpenID Connect Bridge

**arXiv ID:** 2602.14871 | [PDF](https://arxiv.org/pdf/2602.14871v1)

**作者:** Hakan Yildiz `[一作]` (Service-centric Networking, Technische Universität Berlin), Axel Küpper `[通讯]` (Service-centric Networking, Technische Universität Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

构建了一个跨生态的 SSI 验证平台并实现 OIDC Bridge，提供统一的认证入口并实现多租户 SaaS 化。

**💡 创新点**

创新点包括：① PKCE 支持的 OIDC 认证流；② scope‑to‑proof 模板映射；③ 通过 Keycloak 实现的完整多租户隔离；④ 对 OIDC 与 SSI 结合的 11 种攻击向量做了完整安全分析。

**🔧 技术方法**

技术栈：OIDC+PKCE、Keycloak IAM、Redis 会话、MongoDB 配置、Node.js/Express 后端、React 前端、interID 统一验证层及其三种生态适配器（Aries/Indy、EBSI、EUDI）。

**📊 数据集**

使用了真实环境中的 Verifier 服务（ACA‑Py、Walt.id、EUDI Verifier）以及公开的 SSI 证书和凭证数据进行验证，未采用人工合成数据集。

**📈 对比分析**

与直接 SSI 集成相比，OIDC Bridge 仅需约 180 行标准 OIDC 客户端代码，开发与运维成本显著降低；性能上未出现瓶颈，验证吞吐量与单一 Verifier 部署相当。

**⚠️ 局限性**

局限性主要体现在：① SSI 证书缺乏持久识别符，导致会话绑定需额外声明；② 不支持 UserInfo 接口；③ 设备丢失、凭证失效等传统 IdP 的账户恢复机制缺失；④ 受限于现有生态的信任框架与标准化程度。

---

## 312. Playing the Imitation Game: How Perceived Generated Content Shapes Player Experience

**arXiv ID:** 2602.14254 | [PDF](https://arxiv.org/pdf/2602.14254v1)

**作者:** Mahsa Bazzaz `[一作]` (Northeastern University), Seth Cooper `[通讯]` (Northeastern University)

**通讯引用:** 8574 | [OpenAlex ID](https://openalex.org/A5083670815)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

对《Super Mario Bros.》与《Sokoban》中的人工与AI生成关卡进行混合方法研究，调查玩家是否能区分作者并评估其体验，同时进行开放式问答的主题分析。

**💡 创新点**

发现玩家对关卡作者的感知比真实作者更能预测其体验，展示了“人类似乎”策略的双刃效应，并提出更细致的AI使用披露建议。

**🔧 技术方法**

采用Turing‑style辨别任务、混合效应逻辑回归、序数逻辑回归以及NVivo主题编码等技术进行定量与定性分析。

**📊 数据集**

使用60个关卡（每个游戏15人类设计+15 AI生成），AI生成选取六种主流PCG/LLM模型随机抽样，确保关卡可玩且无明显缺陷。

**📈 对比分析**

通过准确率、误报率、误漏率和AI检测成功率评估辨别表现，发现玩家识别率与50%差距无显著性；主观体验与对作者的信念相关性显著，AI被视为更具挑战和挫败感。

**⚠️ 局限性**

研究局限于两款短关卡的2D游戏、单项Likert测量、在线众包样本且缺乏专业设计师，且未能建立因果关系。

---

## 313. Truthful Reverse Auctions for Adaptive Selection via Contextual Multi-Armed Bandits

**arXiv ID:** 2602.14476 | [PDF](https://arxiv.org/pdf/2602.14476v1)

**作者:** Pronoy Patra `[一作]` (International Institute of Information Technology), Sujit Gujar `[通讯]` (International Institute of Information Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了首个可实现逆向真诚性的上下文多臂老虎机（MAB）机制，用于动态选择大型语言模型（LLM）并收集其成本信息。

**💡 创新点**

创新点在于：1）将逆向拍卖与上下文MAB学习相结合；2）设计了一个基于逆向自重采样的随机化过程，使任何单调分配规则都保持真诚；3）证明该机制在成本最小化目标下实现O(√T)次序回退；4）首次在逆向逆向环境中实现单参数代理的真诚性和个体理性。

**🔧 技术方法**

主要技术包括：上下文线性UCB（SupLinUCB）改造为单调版本、逆向自重采样（Reverse Self‑Resampling）技术、Myerson反射式付款规则、理论证明（单调性、真诚性、回退分析）。

**📊 数据集**

使用合成数据集：5维高斯上下文、对数正态虚拟成本、以及两种奖励分布（高斯噪声与指数噪声）。

**📈 对比分析**

与理想的全知策略（clairvoyant benchmark）和标准SupLinUCB进行对比，结果显示累计回退随√T增长、逐步逼近理想收益，显示子线性回退和高效收益收敛。

**⚠️ 局限性**

局限性包括：1）假设每轮token需求已知且相同；2）未考虑预算约束或最小成本阈值；3）仅在模拟环境验证，缺乏真实LLM市场实验；4）理论分析依赖正则性与虚拟成本单调性，实际成本分布可能更复杂。

---

## 314. StackingNet: Collective Inference Across Independent AI Foundation Models

**arXiv ID:** 2602.13792 | [PDF](https://arxiv.org/pdf/2602.13792v1)

**作者:** Siyang Li `[一作]` (Huazhong University of Science and Technology), Lieyun Ding `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 11613 | [OpenAlex ID](https://openalex.org/A5100752789)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个轻量级的元集成框架 StackingNet，能够将多个独立训练的基础模型（LLM/VLM）的输出结果作为黑盒进行聚合，实现回归与分类任务。

**💡 创新点**

创新点在于：①使用可训练的加权与偏置统一组合预测；②支持监督与无监督两种学习方式；③实现偏差缓解、可靠性排名和对抗性剪枝，从而提升集成的公平性、鲁棒性和可解释性。

**🔧 技术方法**

采用了基于条件独立假设的 StackingNet 结构、均方误差/交叉熵损失、无监督可靠性估计的谱分解、以及对比传统的投票、Dawid‑Skene、GLAD 等组合方法。

**📊 数据集**

实验覆盖四类任务，使用 ICLR/NeurIPS 论文评分数据、Chicago Face Database 面部属性评级数据，以及 HELM 8 个二/多分类数据，共同评估十款 API 基础 LLM/VLM 的预测。

**📈 对比分析**

与单一模型、人类评审以及传统组合方法进行对比，StackingNet 在 MAE（论文评分）和 BCA（分类）上均优于最佳单模型，并在仅 1% 标注样本的少量监督下即可接近或超过人类平均水平。

**⚠️ 局限性**

局限性包括对基础模型独立性与多样性的假设；对预测结果归一化与非负权重约束的依赖；以及在模型同源或多样性不足的环境下，聚合收益可能显著下降。

---

## 315. Multi-dimensional Persistent Sheaf Laplacians for Image Analysis

**arXiv ID:** 2602.14846 | [PDF](https://arxiv.org/pdf/2602.14846v1)

**作者:** Xiang Xiang Wang `[一作]` (Michigan State University), Guo-Wei Wei `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出多维持久性纤维拉普拉斯（MPSL）框架，用于在数据集层构建多尺度多维度的 simplicial complexes，并利用持久性纤维拉普拉斯 Laplacian 提取稳定的图像特征；

**💡 创新点**

通过将不同降维维度与邻域尺度视为互补尺度并在特征层面进行聚合，显著降低对单一降维维度的敏感性，首次在图像分析中引入数据集层的纤维 Laplacian 与多维度融合；

**🔧 技术方法**

使用 simplicial complexes、纤维（sheaf）理论、持久性纤维拉普拉斯 Laplacian、k‑NN 邻域构建、PCA 降维、统计谱描述子、k‑NN 分类器；

**📊 数据集**

COIL20（20类共1440幅灰度图）与 ETH80（8类共3280幅灰度图，含多实例）两个公开图像基准；

**📈 对比分析**

与传统 PCA 基线（单维度）和多维度聚合方法进行 5‑折交叉验证 k‑NN（k=5）对比。结果显示 MPSL 在所有降维维度下表现更稳定，平均准确率提升约 0.5%–1%，单维度 MPSL 已大幅超越 PCA，多维度聚合进一步提升 1%–2%；

**⚠️ 局限性**

计算成本较高，需要为每个样本、每个维度和邻域构造本地 simplicial complexes；对邻域参数选择仍有一定依赖；目前仅在灰度图像上验证，彩色或高分辨率图像的可扩展性待研究；

---

## 316. BLUEPRINT Rebuilding a Legacy: Multimodal Retrieval for Complex Engineering Drawings and Documents

**arXiv ID:** 2602.13345 | [PDF](https://arxiv.org/pdf/2602.13345v1)

**作者:** Ethan Seefried `[一作]` (Oak Ridge National Laboratory), Tirthankar Ghosal `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 798 | [OpenAlex ID](https://openalex.org/A5081072666)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Blueprint，一套针对工程档案的多模态检索系统；

**💡 创新点**

创新点在于结合布局感知的区域检测、基于 VLM 的区域 OCR、标识符归一化以及稀疏+密集检索融合与轻量级区域重排序；

**🔧 技术方法**

使用 YOLOv8-S 进行区域检测、LLaMa-4 进行 OCR、Nemotron-7B 文本嵌入、BM25 与 ANN 结合、CLIP+heuristics 进行文件路由；

**📊 数据集**

在 770k 未标注的工程文件中构建 5k+ 评测子集，包含 375 条专家编写的自然语言查询和 1,500 条带有手工标注的黄金测试集；

**📈 对比分析**

与多种开源 VLM 基线（LLaMA 3.2 Vision、LLaVA、PaliGemma、Pixtral、Llama 4 Scout）进行对比，Blueprint 在 nDCG@3、Success@3 等指标上均领先 10%+，并在 9.5 s/file 的吞吐量下最快；

**⚠️ 局限性**

局限包括对极端质量低或非标准标题块的识别能力有限，重排序仅覆盖有限的约束，未处理跨文档关系，且对多语言符号的鲁棒性待提升。

---

## 317. REDSearcher: A Scalable and Cost-Efficient Framework for Long-Horizon Search Agents

**arXiv ID:** 2602.14234 | [PDF](https://arxiv.org/pdf/2602.14234v1)

**作者:** Zheng Chu `[一作]`, Xing Yu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 REDSearcher 框架，用于训练长时深度搜索代理，通过任务合成、训练中期与后期相结合的方式实现高质量搜索行为；

**💡 创新点**

创新点包括：双约束任务合成（图拓扑+证据分散）生成结构复杂任务；主动工具增强查询使工具调用成为必需；成本高效的训练中期阶段分离原子技能与交互执行；轻量化模拟环境与上下文管理（discard‑all）等；

**🔧 技术方法**

使用技术包括：大语言模型微调、ReAct式轨迹、图约束优化、层次化规划、意图锚定检索、工具调用、强化学习以及多模态融合；

**📊 数据集**

使用数据集有：10K 高质量文本搜索轨迹、5K 多模态轨迹、1K RL 查询集，以及 Humanity's Last Exam、BrowseComp、BrowseComp‑ZH、GAIA、MM‑BrowseComp、BrowseComp‑VL、MMSearch‑Plus、MMSearch、LiveVQA 等评测基准；

**📈 对比分析**

对比方法：与多款专有代理（Seed1.8、Gemini‑3‑Pro、GPT‑5.2）和开源代理（Kimi‑K2.5、GLM‑4.7、DeepSeek‑V3.2、Tongyi DeepResearch、GLM‑4.7‑Flash、Qwen3‑VL）等进行比较；在 30B 参数级别实现了整体分数 51.3 的最优表现，GAIA 得分 80.1 超越 GPT‑5‑Thinking‑high；多模态基准上与 Gemini‑3‑Pro、Seed1.8 竞争，且显著优于 Qwen3‑VL‑235B；

**⚠️ 局限性**

局限性：依赖模拟环境，可能无法完全复制真实 Web 动态；对工具使用的依赖导致工具免费模式表现较差；上下文管理可能丢失有用的长时记忆；训练成本高，需要大量微调与 RL 计算资源；目前仅在 30B 规模下验证，未测试更大模型的可扩展性；

---

## 318. Counting Balanced Triangles on Social Networks With Uncertain Edge Signs

**arXiv ID:** 2602.14084 | [PDF](https://arxiv.org/pdf/2602.14084v1)

**作者:** Alexander Zhou `[一作]` (Hong Kong Polytechnic University), Yue Wang `[通讯]` (Shenzhen Institute of Computing Sciences)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在不确定符号的社交网络上定义并计数/枚举平衡与不平衡三角形，提出基准与改进的精确算法以及基于采样的近似计数方法。

**💡 创新点**

利用边概率的绝对值排序（Absolute Edge Order）实现搜索空间裁剪，并通过概率理论推导阈值区间，显著减少无效三角形检查；同时提供可与采样结合的高效本地搜索框架。

**🔧 技术方法**

核心技术包括基于节点度排序的前向三角枚举、绝对边值排序、概率边界剪枝、采样（顶点采样与边采样）与无偏估计。

**📊 数据集**

在12个真实与合成网络（Hamster、Brightkite、Slashdot、Epinions、LiveMocha、Flixster、Skitter、Petster、Flickr、LiveJournal、Orkut、UK Domain）上实验，使用均匀、Beta 等概率分布模拟不确定符号。

**📈 对比分析**

改进精确算法相比基准平均缩短 2–3 倍，采样方法在大规模图上实现 3–4 个数量级的速度提升，且误差随样本数增加而趋于真实计数。

**⚠️ 局限性**

限制包括：仅适用于静态图、阈值 t 需≥0.5、采样精度受阈值和分布影响、未考虑并行/流式环境，未来可拓展为分布式或流式求解。

---

## 319. A Survey of Code Review Benchmarks and Evaluation Practices in Pre-LLM and LLM Era

**arXiv ID:** 2602.13377 | [PDF](https://arxiv.org/pdf/2602.13377v1)

**作者:** Taufiqul Islam Khan `[一作]` (University of Manitoba), Tse-Hsun Chen `[通讯]` (Concordia University)

**通讯引用:** 7167 | [OpenAlex ID](https://openalex.org/A5100357817)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述并对比了2015–2025年间基于LLM与传统方法的代码审查基准，收集并分析了99篇论文中的数据集、评估指标与任务划分，构建了五大领域与18个细粒度子任务的多层次分类体系；

**💡 创新点**

首次提供跨时代、跨任务的完整框架与细粒度标签，揭示从Pre‑LLM向LLM的任务重心转移、语言多样性提升与评价方法演化，为后续研究奠定统一的基准与度量规范；

**🔧 技术方法**

主要采用系统性文献检索、双人编码与共识协议，对已发表论文进行元数据抽取与主题归纳，并基于此构建分类树；

**📊 数据集**

整合了Pre‑LLM时代58篇、LLM时代41篇的基准数据集，涵盖提交级、文件级、方法级等多粒度，覆盖Java、Python、C/C++等20多种语言；

**📈 对比分析**

通过定量统计与可视化比较两时代的任务分布、语言覆盖与评估指标，发现LLM时代Peer Review占比约60%，而Change Understanding等传统任务显著下降；实验性对比未做，而是以数量与质量维度进行横向对比；

**⚠️ 局限性**

当前基准的局限包括：缺乏宏观层面（如影响分析、提交拆分）任务；大多数评价指标停留在静态文本匹配与分类，未能体现功能正确性；多任务评估缺乏统一度量；新基准频繁出现，导致研究不易追踪与复现。

---

## 320. Alignment Adapter to Improve the Performance of Compressed Deep Learning Models

**arXiv ID:** 2602.14635 | [PDF](https://arxiv.org/pdf/2602.14635v1)

**作者:** Rohit Raj Rai `[一作]` (Indian Institute of Technology Guwahati), Amit Awekar `[通讯]` (Indian Institute of Technology Guwahati)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级滑动窗口对齐适配器 AlAd，利用压缩模型的输出与原大模型的嵌入对齐，显著提升压缩模型在 token 级 NLP 任务上的表现。

**💡 创新点**

创新点包括：①局部上下文滑动窗口的对齐适配器；②分阶段对齐训练（任务无关、任务相关、任务微调）；③可与 LoRA 等 PEFT 结合使用且对压缩方法不敏感。

**🔧 技术方法**

使用单隐藏层 Feed‑Forward 适配器、滑动窗口上下文、MSE 对齐损失，三阶段训练流程（任务无关预训练、任务特定对齐、任务微调）以及 LoRA 参数高效微调。

**📊 数据集**

使用英文 Wikipedia 语料进行预训练，下游任务数据集包括德语 Universal Dependencies POS、MultiCoNER v2 NER、SQuAD 2.0 EQA。

**📈 对比分析**

对比冻结/微调压缩模型与 LoRA，在 POS、NER、EQA 任务中测量准确率/ F1/EM。AlAd 在保持 2× 以上速度提升的同时，使压缩模型性能接近大模型，窗口尺寸为 5 时在 POS 达到 93%（≈BERT‑base）且速度约 2×。

**⚠️ 局限性**

局限性：仅验证于 encoder‑only 语言模型和文本任务；适配器为任务特定；对极小模型的 PEFT 效果有限，需联合微调；未测试多模态或解码器 LLM。

---

## 321. D-SECURE: Dual-Source Evidence Combination for Unified Reasoning in Misinformation Detection

**arXiv ID:** 2602.14441 | [PDF](https://arxiv.org/pdf/2602.14441v1)

**作者:** Gagandeep Singh `[一作]` (University of Queensland), Priyanka Singh `[通讯]` (University of Queensland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 D-SECURE 框架，将内部局部篡改检测（HAMMER）与外部检索式事实核查（DEFAME）融合，输出统一可解释报告。

**💡 创新点**

创新点在于：①双源融合架构，将内容级局部篡改与外部证据级事实核查结合；②提供统一人类可读的解释报告；③在融合过程中既可使用规则也可迁移到学习模块。

**🔧 技术方法**

技术包括：多模态对齐 Transformer（HAMMER）、检索增强多模 LLM（DEFAME）、检索与生成的融合流水线、规则/学习式融合决策、结构化解释生成。

**📊 数据集**

使用数据集：DGM^4（用于局部篡改检测）和 ClaimReview 2024+（用于全球事实核查）。

**📈 对比分析**

评估方法：在 DGM^4 上测二分类准确率，D-SECURE 与 HAMMER 维持 93% 以上；在 ClaimReview 2024+ 上按三种评估规则（严格、操纵感知、干预感知）测 3‑way 准确率，D-SECURE 在“操纵感知”和“干预感知”模式下分别达 46.33% 与 47.0%，相较 DEFAME 提升显著，严格模式略低但整体性能提升。

**⚠️ 局限性**

限制：①依赖 HAMMER 与 DEFAME 的覆盖范围，可能在不熟悉实体或压缩图像时下降；②规则式融合不处理不确定性和矛盾证据；③评估样本规模有限，偏向政治和健康领域，泛化能力待进一步验证。

---

## 322. Direction Matters: Learning Force Direction Enables Sim-to-Real Contact-Rich Manipulation

**arXiv ID:** 2602.14174 | [PDF](https://arxiv.org/pdf/2602.14174v1)

**作者:** Yifei Yang `[一作]` (Zhejiang University), Yue Wang `[通讯]` (Zhejiang University)

**通讯引用:** 54269 | [OpenAlex ID](https://openalex.org/A5113600509)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在仿真中使用人类设计的有限状态机（FSM）生成演示，训练策略预测末端执行器姿态、接触状态和接触力的方向，并在真实环境中用该方向与少量手工设定的力幅值结合正则化正交雅可比控制实现接触丰富的机械操作；

**💡 创新点**

①将力方向作为与动力学无关的目标，使策略在仿真中可学习并跨域迁移；②用专属监督的FSM在仿真中快速生成多样化演示；③采用力感知旁路的自适应正交雅可比控制，仅需每个接触阶段少量手工标量即可实现任务对齐的可变顺从；

**🔧 技术方法**

采用E2VLA视觉语言动作框架进行策略训练；利用仿真中FSM提供的力方向、接触状态标签；在真实环境中使用基于力方向的自适应正交雅可比（Admittance）控制器；结合动力学随机化与理论稳定性分析；

**📊 数据集**

纯仿真数据，涉及四个接触丰富任务：微波开门（MO）、插孔（PH）、白板擦（WW）、门开（DO）；通过仿真中生成的多种初始位姿和环境扰动来训练策略；

**📈 对比分析**

与传统位置预测+盲正交雅可比（低/中/高刚度）以及E2VLA+盲正交雅可比三种基线对比；在无扰动和扰动条件下的成功率分别为91% vs 67%（最佳基线），鲁棒性更好，且在四个任务中都表现出更高的任务完成质量；

**⚠️ 局限性**

仅处理平移力方向，未扩展到旋转扭矩；在引入力方向预测后位姿预测精度略有下降，需进一步优化模型注入方式；手工设定力幅值虽简单，但仍需要根据任务进行微调。

---

## 323. RoboSolver: A Multi-Agent Large Language Model Framework for Solving Robotic Arm Problems

**arXiv ID:** 2602.14438 | [PDF](https://arxiv.org/pdf/2602.14438v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 324. MoRL: Reinforced Reasoning for Unified Motion Understanding and Generation

**arXiv ID:** 2602.14534 | [PDF](https://arxiv.org/pdf/2602.14534v1)

**作者:** Hongpeng Wang `[一作]` (University of Sydney), Hao Tang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了MoRL，一种统一的多模态运动模型，既能进行运动理解（运动→文本），也能进行运动生成（文本→运动），并在推理阶段通过Chain‑of‑Motion（CoM）实现逐步规划与反思。

**💡 创新点**

创新点包括：① 任务专一的强化学习奖励设计，融合语义对齐、推理一致性、物理可行性与文本‑运动一致性；② 结合自监督的链式思维（CoT）数据集（MoUnd‑CoT‑140K、MoGen‑CoT‑140K）实现模型的冷启动与对齐；③ 在推理时采用CoM策略，使模型能生成逻辑连贯的推理轨迹并迭代校正运动输出。

**🔧 技术方法**

技术方法包括：多模态大型语言模型（Qwen3‑4B‑Instruct）与专用文本/运动分词器；VQ‑VAE风格的运动分词；分阶段训练（SFT+RLVR）与组式策略梯度强化学习；NLI模型实现推理一致性奖励；物理约束损失实现运动可行性；以及多模态对齐编码器实现文本‑运动一致性。

**📊 数据集**

使用的数据集有：Synthetic CoT数据集MoUnd‑CoT‑140K（运动→推理+文本）和MoGen‑CoT‑140K（文本→推理+运动）；基准测试集HumanML3D（14.6k动作+44.9k文本）与KIT‑ML（3.9k动作+6.3k文本）。

**📈 对比分析**

在理解任务上，MoRL在BLEU、ROUGE、CIDEr、BERTScore等指标均优于现有单模型和统一模型；在生成任务上，MoRL在R‑Precision、MM‑Dist、FID、Diversity、MM‑Modality等指标均达到或超过Diffusion、GPT等先进方法，尤其在长时序多阶段动作上表现突出。

**⚠️ 局限性**

局限性包括：奖励设计依赖规则，可能难以迁移到不同运动域或风格；CoM推理会增加推理时计算量，限制实时性能；模型使用离散运动分词，未显式建模接触动力学或人‑物交互细节。

---

## 325. MemeTrans: A Dataset for Detecting High-Risk Memecoin Launches on Solana

**arXiv ID:** 2602.13480 | [PDF](https://arxiv.org/pdf/2602.13480v1)

**作者:** Sihao Hu `[一作]` (Georgia Institute of Technology), Ling Liu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 21263 | [OpenAlex ID](https://openalex.org/A5100343991)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了 MemeTrans 数据集，并基于该数据集进行高风险 Memecoin 发售检测实验。

**💡 创新点**

创新点在于：①首次构建覆盖 41k+ 发售、200M 交易的大规模 Solana memecoin launchpad 数据集；②设计 122 维特征并结合多账户 bundle 追踪揭示隐藏的操纵行为；③采用统计指标与 ML 结合的双重标注方案，提升风险标签的准确性。

**🔧 技术方法**

技术手段包括：Solana 区块链数据抓取与交易解析；bundle 聚类与多账户识别；特征工程（情境、持仓集中度、市场活动、bundle 统计、时间序列）；机器学习模型（LR、RF、XGBoost、LightGBM、MLP）和时序模型（TCN、LSTM、GRU、Transformer）进行风险分类；混合标注与风险评分框架。

**📊 数据集**

使用了 MemeTrans 数据集，该数据集包含 41,470 个 memecoin、218M 交易、bundle 追踪信息以及 122 维特征，并标注为高/中/低风险。

**📈 对比分析**

通过 AUPRC、F1、精确率/召回率等指标进行对比实验，表格特征基模型 MLP 表现最佳（AUPRC≈0.57），相较时序模型效果更差；特征消融表明市场活动组贡献最大；在币选策略实验中，使用 MLP 风险评分可将平均损失降低 56.1%。

**⚠️ 局限性**

局限性包括：样本高度不平衡且标签受阈值与检测器精度限制；数据仅覆盖 Solana，缺乏跨链验证；对极端操纵模式的识别仍有提升空间；未来需要实时更新数据并引入更细粒度行为特征。

---

## 326. Feature Recalibration Based Olfactory-Visual Multimodal Model for Fine-Grained Rice Deterioration Detection

**arXiv ID:** 2602.14408 | [PDF](https://arxiv.org/pdf/2602.14408v1)

**作者:** Rongqiang Zhao `[一作]` (Harbin Institute of Technology), Jie Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 94154 | [OpenAlex ID](https://openalex.org/A5100454174)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种基于嗅觉–视觉多模态的特征重校准模型，用于细粒度稻米劣化检测，包含FDEC重构多模态嵌入特征和FDRA-Net重校准特征的网络架构。

**💡 创新点**

创新点在于①使用FDEC将电子鼻和RGB图像分别嵌入统一结构，从而提升样本表示；②在FDRA-Net中引入SE与CBAM对嗅觉通道和视觉空间进行自适应重校准，显著捕捉细粒度劣化信号；③仅依赖RGB相机和电子鼻，避免了高成本光谱仪，简化检测流程。

**🔧 技术方法**

技术手段包括：电子鼻+RGB相机数据采集；FDEC利用PLR与PLCE实现数值与图像嵌入；FDRA-Net基于SE、CBAM的多层重校准网络；使用PyTorch进行训练，采用交叉熵损失、Adam优化；Grad‑CAM可视化与Ablation实验评估网络贡献。

**📊 数据集**

使用从9天采集的稻米样本（共7200个训练+900个测试），三类标签（Normal、Expired、Moldy），每个样本通过RGB相机（SARGO G20）和电子鼻（PEN3）获得的多模态信号构成数据集。

**📈 对比分析**

与SS‑Net、CNN、ResNet、SENet、SKNet、NAM等基线模型对比。FDRA-Net在离线测试集上取得99.89%准确率、93.33%现场准确率，明显优于最佳基线SS‑Net的91.22%；在各类别上Recall、Precision均高；模型尺寸48.73 MB，推理时间9.48 ms，满足工业部署需求。

**⚠️ 局限性**

局限性包括：相较于基线模型，模型体积和推理延迟略大；对电子鼻与RGB摄像头的稳定性与漂移仍需进一步研究；部署时需要额外的模型压缩、量化或知识蒸馏；目前仅在稻米样本上验证，跨品种、跨环境的泛化性尚待扩展。

---

## 327. Backdooring Bias in Large Language Models

**arXiv ID:** 2602.13427 | [PDF](https://arxiv.org/pdf/2602.13427v1)

**作者:** Anudeep Das `[一作]` (University of Waterloo), Florian Kerschbaum `[通讯]` (University of Waterloo)

**通讯引用:** 4531 | [OpenAlex ID](https://openalex.org/A5102985450)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大语言模型的偏见操纵后门攻击在白盒环境下进行系统实验，比较语法触发与语义触发两种攻击方式。

**💡 创新点**

创新点在于在白盒模型构建者视角下，使用更高毒化比例和数据增广（拼接）评估后门效果，并对两类后门在正负偏见诱导难度进行对比。

**🔧 技术方法**

采用VPI、CBA、EmbedX、SFT、DPO等后门实现，防御方面采用CROW（模型内）和CleanGen（模型外）两种主流技术。

**📊 数据集**

实验使用Llama-2系列模型，并用VPI提供的议题数据（堕胎、OpenAI、乔·拜登）和MMLU基准评估实用性。

**📈 对比分析**

通过LLM评估、MMLU分数、后门抵抗度和成本指标对比，发现语义触发后门在负偏见上更强，CBA、EmbedX在正偏见上更有效，但两者均受限；CleanGen在抵抗度上优于CROW，且对实用性损伤更小。

**⚠️ 局限性**

局限在于仅使用单一模型族、评估仅基于自动LLM评分且缺乏人工验证，对触发器优化和更大模型的跨平台性未探究。

---

## 328. Catalytic Tree Evaluation From Matching Vectors

**arXiv ID:** 2602.14320 | [PDF](https://arxiv.org/pdf/2602.14320v1)

**作者:** Alexandra Henzinger `[一作]` (Massachusetts Institute of Technology), Seyoon Ragavan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5068124047)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本工作提出了一种新的催化计算算法，用于在树评估问题上仅使用对数级别的自由空间并且只需亚多项式级的催化空间，且保持多项式运行时间。

**💡 创新点**

创新点在于将匹配向量族（matching‑vector families）与信息论私有信息检索（PIR）技术相结合，从而在催化计算框架下实现更紧凑的空间利用；此前的催化树评估算法需要线性级的催化空间，而本算法将其压缩至 2^{log^ε n}。

**🔧 技术方法**

核心技术包括：①构造对数可统一的匹配向量族；②将树评估中的局部函数查询视为对真值表的 PIR 取样；③在催化机中利用可逆操作实现向量加法和内积计算。

**📊 数据集**

该研究为理论算法，未使用任何具体数据集；所有结果均为抽象证明与复杂度分析。

**📈 对比分析**

与以往的催化树评估算法（如 Cook‑Mertz 方案）相比，本方法在自由空间维持 O(log n)，催化空间由 O(n) 减少到 2^{log^ε n}，时间保持多项式级；此外，通过 Williams 的归约可进一步得到时间‑空间‑催化空间的全新权衡。

**⚠️ 局限性**

局限性包括：①该算法的催化空间仍为亚多项式，但未达到最优的 O(log n)；②性能提升高度依赖匹配向量族的构造，目前最优构造仍存在显著空间和维度的折衷；③若想进一步降低催化空间或实现更快时间，需在匹配向量族理论或 PIR 技术上取得突破。

---

## 329. Sensitivity of Repetitiveness Measures to String Reversal

**arXiv ID:** 2602.14385 | [PDF](https://arxiv.org/pdf/2602.14385v1)

**作者:** Hideo Bannai `[一作]` (Information Sciences and Technologies), Cristian Urbina `[通讯]` (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文通过构造无限字符串族，研究并给出了在字符串逆序操作下，BWT 运行数、Lempel‑Ziv 解析大小以及最小字典解析大小的敏感性（即增量变化幅度），并进一步给出了 LZ 解析比值可逼近 3、最小字典解析乘法敏感性为 Θ(log n) 的紧上界。

**💡 创新点**

创新点在于：①将原本已知的 O(log n) 下界提升为 Θ(n) 的下界，证明了多种重复度度量对逆序极度敏感；②构造了新的字符串族实现 LZ 解析比值趋近 3；③通过 Fibonacci 单词与 Lyndon 分解，证明最小字典解析乘法敏感性与上界紧匹配。

**🔧 技术方法**

主要技术包括：组合构造字符串族、Burrows‑Wheeler 变换与其 RLBWT 的分析、Lyndon 分解与 ω‑序的运用、Fibonacci 单词性质、以及对 Lempel‑Ziv 解析的递归结构研究。

**📊 数据集**

使用的是人工合成的无限字符串族（如 u_k、T_p、F_k 等），没有使用实际文本数据集。

**📈 对比分析**

方法为理论分析与证明，未进行实验对比。通过证明下界与已知上界相匹配，展示了各度量在逆序操作下的最坏情况性能。

**⚠️ 局限性**

局限性：①常数因子未知，仅给出渐进紧界；②仅对特定度量给出下界，其他度量（如 z_e、χ 等）的精确敏感性仍未完全解决；③所有结果均为最坏情况理论，缺乏实验验证。

---

## 330. ADAB: Arabic Dataset for Automated Politeness Benchmarking -- A Large-Scale Resource for Computational Sociopragmatics

**arXiv ID:** 2602.13870 | [PDF](https://arxiv.org/pdf/2602.13870v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 331. Constructing Quantum Convolutional Codes via Difference Triangle Sets

**arXiv ID:** 2602.13505 | [PDF](https://arxiv.org/pdf/2602.13505v1)

**作者:** Vahid Nourozi `[一作]` (New Mexico State University), David Mitchell `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于差分三角集（DTS）的搜索无关构造方法，用以生成满足对称正交条件的稀疏量子卷积码（QCC）

**💡 创新点**

创新点在于利用DTS的强差分特性和“镜像反射”映射，直接构造出Z型稳定子，使得X/Z稳定子在多项式域内自动满足辛正交，避免了传统的枚举搜索，同时保证码的自由距离和低记忆量

**🔧 技术方法**

主要技术包括：差分三角集（强DTS）构造、支持向量的镜像反射映射、卷积码的多项式与块矩阵表示、辛内积验证以及数值模拟验证

**📊 数据集**

由于该工作属于理论构造，未使用外部数据集；所有实验基于自定义的强DTS实例（覆盖率 1/3、2/4、3/5 码率）和相应的多项式生成

**📈 对比分析**

通过与传统 QCC 构造（如基于随机稀疏矩阵或现有自正交卷积码）在记忆量、码率、自由距离以及构造时间等指标上进行对比，实验表明该方法在保持低记忆、稀疏度和可预测距离的同时，构造时间几乎为零，且与现有方案相比具有更优的记忆/距离权衡

**⚠️ 局限性**

限制在于构造仅适用于可构造的强DTS 家族，可能无法覆盖所有所需码率或极大规模的码；镜像反射对支持结构的要求较严格，导致在某些情况下可能无法实现理想的记忆或距离；此外，对于非常大的记忆参数，实际实现与硬件延迟仍需进一步评估

---

## 332. GeoEyes: On-Demand Visual Focusing for Evidence-Grounded Understanding of Ultra-High-Resolution Remote Sensing Imagery

**arXiv ID:** 2602.14201 | [PDF](https://arxiv.org/pdf/2602.14201v1)

**作者:** Fengxiang Wang `[一作]` (National University of Defense Technology), Bo Du `[通讯]` (Wuhan University)

**通讯引用:** 29840 | [OpenAlex ID](https://openalex.org/A5060042752)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 GeoEyes，一种针对超高分辨率遥感 VQA 的多模态大语言模型，能够按需进行多轮视觉放大与推理；

**💡 创新点**

其创新点在于首次识别并解决了工具使用同质化（Tool Usage Homogenization）问题，并通过构造 UHR-CoZ 训练集与 AdaZoom‑GRPO 强化学习奖励，实现在任务层面上自适应、停止的放大策略；

**🔧 技术方法**

技术上结合了冷启动的监督微调（SFT）与专门设计的多项奖励（Adaptive Efficiency、Chain‑of‑Focus、Necessity‑Aware Process Verification），并采用 Group Relative Policy Optimization 进行强化学习；

**📊 数据集**

使用了 HighRS‑VQA 迁移得到的 UHR‑CoZ 训练集、XLRS‑Bench 作为评测基准，并在 RL 阶段加入 SuperRS‑VQA 数据来提升领域适应；

**📈 对比分析**

在 XLRS‑Bench 上与现有开放与封闭源 MLLM 进行对比，GeoEyes 达到 54.23% 的平均准确率，显著优于 GeoLLaVA‑8K（51.5%）、DeepEyes（50.0%）以及更大规模的通用模型如 Qwen3‑VL‑235B（51.1%），在细粒度感知任务上提升尤为显著；

**⚠️ 局限性**

局限性在于仍受稀疏证据密度限制，对其他遥感任务的泛化能力尚待验证，并且训练过程需依赖手工构造的高质量 CoT 数据与大量强化学习计算资源。

---

## 333. InnoEval: On Research Idea Evaluation as a Knowledge-Grounded, Multi-Perspective Reasoning Problem

**arXiv ID:** 2602.14367 | [PDF](https://arxiv.org/pdf/2602.14367v1)

**作者:** Shuofei Qiao `[一作]` (Zhejiang University), Emine Yilmaz `[通讯]` (University College London)

**通讯引用:** 4772 | [OpenAlex ID](https://openalex.org/A5076265623)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于大语言模型的创新评估框架 InnoEval，能够对科研想法进行知识驱动、多维度、多视角的系统评估，并生成可操作的改进建议。

**💡 创新点**

通过异构深度知识检索、个性化学术评审板块、分离的多维度评估代理，实现场景化的集体共识与多角度判断，突破了传统单一 LLM 评审的知识窄化、偏见与维度扁平化问题。

**🔧 技术方法**

使用了 LLM+检索（RAG）、多轮查询细化、语义+LLM 混合评分、知识对齐与归纳、学术人设模拟、五维评估代理以及自动报告生成等技术。

**📊 数据集**

构建了基于 NeurIPS 2025 与 ICLR 2025 公开论文的 217 条点评样本、172 条组评样本以及 372 条二评样本，覆盖 Reject、Poster、Spotlight、Oral 四个标签。

**📈 对比分析**

与 CoT、RAG、ResearchAgent、InternAgent、GraphEval、ScholarEval 等基线在点评、对评、组评三类任务上对照，InnoEval 在点评 F1 提升 16.18%，对评准确率提升约 5%，组评准确率提升约 7.56%，并在人类评审测试中相关系数 ≥0.5，整体质量赢率超过 70%。

**⚠️ 局限性**

受限于 LLM 可能的 hallucination、检索成本、评估维度限制、对极端新颖想法检索不足以及仍需人工审核等因素。

---

## 334. Symmetry in language statistics shapes the geometry of model representations

**arXiv ID:** 2602.15029 | [PDF](https://arxiv.org/pdf/2602.15029v1)

**作者:** Dhruva Karkada `[一作]` (University of California Berkeley), Yasaman Bahri `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过理论分析与实验验证，探讨词嵌入和大语言模型内部表示的几何结构，证明语言共现统计中的平移对称性导致嵌入空间出现圆环、线性一维流形以及可线性解码的时空坐标。

**💡 创新点**

创新点在于：①提出共现统计的平移对称性为统一的组织原则；②利用傅里叶分析精确预测嵌入的几何形状；③解释即使删除大量共现信息嵌入仍保持几何稳定的集体效应机制。

**🔧 技术方法**

主要技术包括：共现矩阵与PMI矩阵的理论关联；离散傅里叶变换对称矩阵的解析对角化；基于理论预测的线性坐标解码误差分析；以及在词嵌入、文本嵌入和Transformer内部层的实验验证。

**📊 数据集**

使用的数据集主要是维基百科文本语料库（词表≈2.5×10⁴），以及公开的大语言模型（如GPT、BERT等）内部表示。对于实验验证，还挑选了月份、年份、地理坐标等子词表。

**📈 对比分析**

方法对比：用理论预测得到的Gram矩阵和Lissajous曲线与实际嵌入的Gram矩阵、PCA投影进行可视化对比，结果显示理论与实测高度一致；在线性坐标解码任务中，理论预测的误差衰减速率与实验匹配。虽然未给出传统性能指标，但定性匹配度非常高。

**⚠️ 局限性**

局限性包括：①理论主要针对词嵌入的低阶共现统计，未充分考虑上下文适配和高阶关系；②对平移对称性假设要求共现矩阵为正半定，实际语料中可能偏离；③对大语言模型的上下文动态表示仅做了初步解释，缺乏完整理论框架。

---

## 335. Interactionless Inverse Reinforcement Learning: A Data-Centric Framework for Durable Alignment

**arXiv ID:** 2602.14844 | [PDF](https://arxiv.org/pdf/2602.14844v1)

**作者:** Elias Malomgré `[一作]` (IDLab), Pieter Simoens `[通讯]` (IDLab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Interactionless Inverse Reinforcement Learning (IIRL) 与 Alignment Flywheel 框架，解耦奖励与策略，构建可审计、可编辑、可迁移的奖励模型，并通过人机循环持续硬化。

**💡 创新点**

创新点包括：① 结构性解耦奖励与策略，消除 Alignment Waste；② 可编辑、模块化、可审计的 IIRL 奖励模型；③ 通过多阶段 Alignment Flywheel 的红蓝审计与人类反馈实现主动、迭代的安全硬化。

**🔧 技术方法**

使用技术包括逆向强化学习、能量模型、深度表示学习、RAG 与映射函数、动态潜在奖励塑形、模型编辑与去学习、联合多智能体审计（红蓝对抗）以及混合反馈的奖励建模。

**📊 数据集**

主要使用专家演示数据集 D_E（多模态文本/图像/视频）和负样本集 D_neg；同时利用公开的无标注视频数据及其他领域特定数据进行验证。

**📈 对比分析**

论文以概念性框架和 3D toy 示例、机器人/LLM 具体应用为例进行演示，强调可审计、可编辑性和安全性提升；未给出量化对比指标，主要展示架构可行性与潜在性能优势。

**⚠️ 局限性**

局限性：依赖专家演示与负样本质量；模型编辑与去学习仍面临灾难性遗忘与参数破坏风险；缺乏大规模实证验证；需进一步正式化审计工作台与治理标准以实现广泛部署。

---

## 336. Learning Proposes, Geometry Disposes: A Modular Framework for Efficient Spatial Reasoning

**arXiv ID:** 2602.14409 | [PDF](https://arxiv.org/pdf/2602.14409v1)

**作者:** Haichao Zhu `[一作]` (Reality Vision), Qian Zhang `[通讯]` (University of California)

**通讯引用:** 8030 | [OpenAlex ID](https://openalex.org/A5100761531)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了视觉位姿估计中学习与几何模块的分离方案，使用 VGGT 生成姿态和深度提议，再用基于 ICP 的几何模块验证并优化/拒绝这些提议。

**💡 创新点**

提出“学习提出，几何决定”的系统范式，明确将学习模块作为假设生成器、几何模块作为最终决策者；通过实验证明几何验证是决定最终精度的关键，而学习初始化并非必要。

**🔧 技术方法**

VGGT 深度/姿态提议网络；基于深度的点云 ICP 对齐；增量在线管线；相对位姿误差（RPE）评估。

**📊 数据集**

TUM RGB‑D 基准数据集，使用同步 RGB‑D 序列进行实验。

**📈 对比分析**

与学习仅、几何仅（身份初始化）以及混合两者的配置进行对比；使用相对位姿误差（旋转、平移）评估；结果显示几何消除后无论初始化如何，误差相近，说明几何模块主导性能；学习初始化在大步长下对结果影响有限。

**⚠️ 局限性**

仅使用局部 ICP 进行几何验证，未考虑全局闭环或多帧一致性；仅使用单一学习模型 VGGT；未对学习提议的不确定性建模；结果受限于深度预测误差，可能在动态场景下表现不佳。

---

## 337. Redundancy-Optimal Constructions of $(1,1)$-Criss-Cross Deletion Correcting Codes with Efficient Encoding/Decoding Algorithms

**arXiv ID:** 2602.13548 | [PDF](https://arxiv.org/pdf/2602.13548v1)

**作者:** Wenhao Liu `[一作]`, Hanxu Hou `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

构造了一类 q-ary (1,1)-criss‑cross 删除纠错二维码，并给出了完整的编码、解码与数据恢复算法；实现了 O(n²) 的计算复杂度。

**💡 创新点**

首次实现了在 n ≥ 11, q ≥ 3 的条件下，既能提供显式编码/解码，又能使编码冗余与理论下界仅相差常数项，达到 2n + 2log_q n + O(1) 的最佳冗余水平。

**🔧 技术方法**

利用一维 q-ary 差分 VT 码与其 1‑RLL（相邻符号不相同）与后缀约束的变体，结合行列和约束与固定符号约束，构造二维码并设计相应的解码流程。

**📊 数据集**

无实验数据集；论文完全基于理论构造与证明，未涉及实测数据。

**📈 对比分析**

通过理论推导比较冗余上界与下界，证明编码冗余与下界差距为常数；相对于已有工作（如 2n + 2log_q n + O(loglog n) 的冗余），实现了更紧的上界；解码复杂度为 O(n²)，优于之前的 O(n³) 或更高复杂度方案。

**⚠️ 局限性**

目前仅适用于 n ≥ 11、q ≥ 3 的参数范围；对更小 n 或 q 的情况需进一步改进；此外，构造依赖于差分 VT 码的特定性质，若 q 远小于 n 时常数项可能不再是常数，而是随 n 增大。

---

## 338. Physical Commonsense Reasoning for Lower-Resourced Languages and Dialects: a Study on Basque

**arXiv ID:** 2602.14812 | [PDF](https://arxiv.org/pdf/2602.14812v1)

**作者:** Jaione Bengoetxea `[一作]`, Rodrigo Agerri `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究构建了基于意大利物理常识数据集GITA的标准Basque及其西部方言版本BasPhyCo，并系统评估多语言与目标语言LLM在故事可行性分类、冲突检测与物理状态识别三层次物理常识推理任务上的表现。

**💡 创新点**

创新点在于首次提供非问答式的Basque（含西部方言）物理常识推理数据集，并通过层级评估框架对比多语言与语言专属LLM，揭示低资源语言和方言变体对推理性能的影响。

**🔧 技术方法**

研究使用多层次评估框架（准确率、一致性、可验证性）和生成式LLM评估方法，结合Llama、Gemma、Latxa等模型，并通过人工翻译、文化本地化及few-shot提示实现标准Basque与西部方言的转换。

**📊 数据集**

数据集来源为意大利物理常识数据集GITA，经过人工翻译成标准Basque，再通过自动转换与人工校对生成西部方言版本，形成BasPhyCo（标准Basque）与BasPhyCowest（西部方言）两套评测集。

**📈 对比分析**

通过在三层次任务上对比多语言模型与语言特定模型的准确率、一致性、可验证性，实验表明：LLM在低资源语言尤其是方言上的可验证性显著下降，Latxa-3.1-70B-It在方言数据上表现最优，其余模型在故事分类上表现稍好，整体随推理层级加深性能显著下降。

**⚠️ 局限性**

局限性包括数据集规模有限、文化本地化偏向特定社区、Basque LLM对西部方言的偏好导致缺乏对其他方言的覆盖，且模型在物理状态识别方面普遍表现不佳。

---

## 339. Fusing Pixels and Genes: Spatially-Aware Learning in Computational Pathology

**arXiv ID:** 2602.13944 | [PDF](https://arxiv.org/pdf/2602.13944v1)

**作者:** Minghao Han `[一作]` (Fudan University), Lihua Zhang `[通讯]` (Fudan University)

**通讯引用:** 32864 | [OpenAlex ID](https://openalex.org/A5100414909)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 STAMP（Spatial Transcriptomics-Augmented Multimodal Pathology）框架，用空间转录组与病理图像进行联合预训练，提升多模态表示学习。

**💡 创新点**

创新点包括：①构建了最大规模 Visium 空间转录组数据集 SpaVis-6M（5.75 M 采样点）；②设计两阶段预训练：空间感知基因编码器 + 层次多尺度对比对齐；③引入邻域重构与跨尺度 Patch 定位任务，强化空间上下文与多尺度特征学习。

**🔧 技术方法**

采用 Transformer 基因编码器、位置编码、Masked Token Prediction、邻域采样与重构、跨尺度 Patch 定位、跨模态 InfoNCE 对比、跨尺度对齐与 Patch‑Region 对齐等技术。

**📊 数据集**

使用了 SpaVis-6M（5.75 M Visium 采样点）、HEST（697 K 图像‑基因配对）、DLPFC、HBC、PSC、HHK、HER2+、LUAD‑mutation 等六个公开数据集。

**📈 对比分析**

通过线性探测、无监督聚类、基因表达预测和 WSI 基因突变分类四类下游任务与 CLIP、PLIP、CONCH、TANGLE 等基准模型对比，STAMP 在所有六个数据集四项任务上均实现或逼近 SOTA，显著提升性能。

**⚠️ 局限性**

局限性包括：①配对样本规模仍有限（仅 697 K 对），无法与 Vision‑Language 模型使用的数千万级对齐；②仅基于 Visium 平台训练，对 Xenium 等新技术的泛化性待验证。

---

## 340. AlignSentinel: Alignment-Aware Detection of Prompt Injection Attacks

**arXiv ID:** 2602.13597 | [PDF](https://arxiv.org/pdf/2602.13597v1)

**作者:** Yuqi Jia `[一作]`, Neil Gong `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种三分类的对抗性注入检测框架AlignSentinel，能够区分误导性注入、对齐注入和非指令输入。

**💡 创新点**

创新点在于引入指令层级结构，将检测问题从二分类改为三分类，并利用LLM注意力图来捕捉指令间的互动，从而显著降低误报率。

**🔧 技术方法**

使用Transformer的多层多头注意力映射做特征提取，并设计Avg-first与Enc-first两种聚合策略，随后通过MLP进行分类。

**📊 数据集**

构建了包含8大应用领域、直接与间接注入场景的全新基准数据集，覆盖三类输入，并在IHEval和OpenPromptInjection等公开基准上进行评测。

**📈 对比分析**

与Prompt-Guard、AttentionTracker、DataSentinel等主流二分类检测器对比，AlignSentinel在所有域与LLM上实现FPR/FNR接近0、准确率≥0.95，尤其在跨域、跨模型迁移中保持高性能。

**⚠️ 局限性**

局限在于仍可能将非指令与对齐指令混淆，导致轻微误报；对抗性攻击虽可提升FNR但会显著降低攻击成功率；对多模态或更复杂指令层级的扩展尚待验证。

---

## 341. Improving Driver Satisfaction with a Driving Function Learning from Implicit Human Feedback -- a Test Group Study

**arXiv ID:** 2602.13733 | [PDF](https://arxiv.org/pdf/2602.13733v1)

**作者:** Robin Schwager `[一作]` (Dr. Ing. h.c. F. Porsche AG), Sören Hohmann `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 1954 | [OpenAlex ID](https://openalex.org/A5040502908)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899`

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

## 342. AgentRob: From Virtual Forum Agents to Hijacked Physical Robots

**arXiv ID:** 2602.13591 | [PDF](https://arxiv.org/pdf/2602.13591v1)

**作者:** Wenrui Liu `[一作]` (Peking University), Tong Yang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于在线社区论坛的异步多代理-机器人交互框架 AgentRob，利用LLM驱动的论坛代理监测、解析指令并通过VLM控制器驱动物理机器人执行任务，同时将执行结果以论坛回复形式反馈。

**💡 创新点**

创新点在于：1）将传统的异步论坛平台重新定位为物理机器人指令的交互介质；2）通过Model Context Protocol（MCP）构建统一的论坛工具库，将论坛API抽象为可被LLM调用的工具；3）实现多机器人多代理共存与协作，代理身份可动态切换；4）提供完整的三层架构（论坛层、代理层、机器人层），实现从自然语言到物理动作的闭环。

**🔧 技术方法**

技术手段包括：大型语言模型（Doubao/ARK、OpenAI等）进行指令抽取与结果总结；MCP（JSON‑RPC 2.0）实现工具调用；VLM驱动的机器人控制器（对Unitree Go2四足和G1人形机器人）通过工具调用链拆解指令；REST API + OAuth 2.0 进行论坛交互；DDS协议实现机器人硬件通信；Python/TypeScript 混合实现代理与工具服务。

**📊 数据集**

数据来源主要是论坛帖子（自然语言指令）和机器人传感数据；未使用公开数据集，实验以 Unitree Go2/G1 机器人在实验室环境中执行示例任务为主。

**📈 对比分析**

在实验中通过示例任务（如“前进5米后转90°”）展示完整的指令解析、执行与总结流程，报告成功率接近100%，但缺乏大规模量化评估。与传统直接API控制相比，AgentRob 在可访问性、异步性和社区可见性方面表现突出，然而在响应延迟（默认30s轮询）和实时控制能力上存在差距。

**⚠️ 局限性**

主要限制包括：1）轮询导致命令识别延迟，无法满足实时控制需求；2）网络连通性对代理与机器人均为必需，网络中断会导致任务失败；3）LLM 可能出现误解或胡编乱造指令，需加入安全过滤；4）单线程执行限制，无法并发处理多条命令；5）当前仅支持文本反馈，缺乏图像/视频等多模态输出。

---

## 343. G2CP: A Graph-Grounded Communication Protocol for Verifiable and Efficient Multi-Agent Reasoning

**arXiv ID:** 2602.13370 | [PDF](https://arxiv.org/pdf/2602.13370v1)

**作者:** Karim Ben Khaled `[一作]` (University of Lorraine), Davy Monticolo `[通讯]` (University of Lorraine)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于图操作的多代理通信协议 G²CP，取代自由文本对话，实现在大型语言模型驱动的代理系统中的精确、高效、可审计的协同推理。

**💡 创新点**

创新点在于将代理交互的内容语义彻底转化为可执行的图遍历与更新命令，消除语义漂移与幻觉传播，同时提供可重放的审计链和显著的令牌节省。

**🔧 技术方法**

核心技术包括图操作语言（TRAVERSE、UPDATE）、Neo4j 的高效图查询、LLM 辅助的实体抽取与意图识别、以及基于 Ed25519 的消息签名与访问控制。

**📊 数据集**

实验使用了 1,247 节点、3,892 条边的工业知识图谱（合成与 21 条真实维护案例），共 500 个合成查询和 21 条真实案例评估。

**📈 对比分析**

与自由文本多代理（FTMA）、JSON 结构化多代理（JSMA）以及单体代理基线相比，G²CP 在任务完成率上提升 34%（90% vs 67%）、令牌消耗降低 73%（768 vs 2,847）、幻觉率降至 2%（vs 23%）且完全消除级联错误，审计可行性达到 100%。

**⚠️ 局限性**

主要限制是对已有结构化知识图谱的依赖，若图谱缺失信息或维护不及时，G²CP 无法补齐缺口；构建和持续更新图谱需要人工投入；在完全无图谱或实时传感数据的场景下，需结合外部 API 或 LLM 生成的推断。

---

## 344. Simultaneous analysis of curved Kirchhoff beams and Kirchhoff--Love shells embedded in bulk domains

**arXiv ID:** 2602.14566 | [PDF](https://arxiv.org/pdf/2602.14566v1)

**作者:** Jonas Neumeyer `[一作]` (Graz University of Technology), Thomas-Peter Fries `[通讯]` (Graz University of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于 Bulk Trace FEM 的混合-混合形式，可同时对二维/三维体域中由水平集隐式定义的曲线梁和曲面壳进行几何线性 Kirchhoff 梁与 Kirchhoff–Love 壳的联合分析。

**💡 创新点**

首次实现对连续隐式定义的多条 Kirchhoff 梁/壳的 Bulk Trace FEM 求解，采用混合-混合公式将四阶问题降到二阶，并通过静态消元显著降低自由度，同时实现高阶收敛。

**🔧 技术方法**

使用张量微分计算（TDC）、水平集方法、混合-混合有限元、静态消元、Lagrange 乘子与高阶 Lagrange 元构造 Bulk Trace FEM。

**📊 数据集**

通过若干合成测试案例（弧形梁、圆形梁、简单支撑壳、杯形壳等）在二维/三维域中构造的层次几何进行验证，未使用公开工业数据集。

**📈 对比分析**

采用 L2 误差、残差误差和储能误差进行收敛率测试，并与解析解或高精度参考解比较，实验表明方法在复杂几何下保持 O(p+1) 或 O(2p) 的高阶收敛，并具备较好的数值稳定性和效率。

**⚠️ 局限性**

受限于三维壳高阶元时出现前渐近区和积分误差，计算成本仍高；未涵盖动态效应、非线性材料或拓扑变化的水平集处理，限制了在实际工程中的直接应用。

---

## 345. GEMs: Breaking the Long-Sequence Barrier in Generative Recommendation with a Multi-Stream Decoder

**arXiv ID:** 2602.13631 | [PDF](https://arxiv.org/pdf/2602.13631v1)

**作者:** Yu Zhou `[一作]` (Kuaishou Inc.), Guorui Zhou `[通讯]` (Kuaishou Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为Lifelong的多流解码器框架，将生成式推荐（GR）扩展到用户整个生命周期的行为序列，实现了对长序列的高效建模。

**💡 创新点**

创新点在于：①把用户行为序列分为近期、中期和生命周期三条流，分别采用一阶段实时提取、轻量级索引器以及两阶段离线-在线压缩等专门策略；②提出无参数融合（parameter‑free fusion）方法避免了近期偏向；③利用轻量级索引器大幅降低中期序列注意力的计算复杂度。

**🔧 技术方法**

使用的技术包括：生成式推荐框架（T5/Transformer）、RQ‑VAE/Residual K‑means离散化、轻量级索引器（sparse attention）、离线‑在线生命周期压缩（QLA+QLU）、无参数融合、混合精度训练与TensorRT推理加速。

**📊 数据集**

实验数据来源于快手工业化数据集：日活400M用户、100M商品、50B交互，平均序列长度1.45万，最长可达10万。

**📈 对比分析**

与九个工业级基线（MPFormer、TDM、MISS、GPRP、Kuaiformer、MIND、TIGER、GRank、CRM）对比，Lifelong在Recall@K（K=100/500/1000）和NDCG@K上均提升约21%~23%，并在在线A/B测试中提升用户停留时长和观看时长。

**⚠️ 局限性**

局限性：①对极大序列仍需额外压缩与索引，推理时延仍受限；②模型主要在快手业务场景验证，跨域推广尚未充分验证；③对实时性和资源预算的平衡仍需要进一步优化。

---

## 346. SIDSense: Database-Free TV White Space Sensing for Disaster-Resilient Connectivity

**arXiv ID:** 2602.13542 | [PDF](https://arxiv.org/pdf/2602.13542v1)

**作者:** George M. Gichuru `[一作]` (Amini), Zoe Aiyanna M. Cayetano `[通讯]` (Amini)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了SIDSense框架，在海上移动通信节点上实现无数据库 TVWS 操作，并将边缘 AI 与 5G RAN 共同部署，实现灾害情境下的视频回传。

**💡 创新点**

创新点包括：① 将 CNN 频谱感知与法规合规门控结合，形成感知‑先行、授权‑后续的工作流；② GPU‑感知的 RAN 调度，保证 AI 任务与 5G L1 时序共存；③ 公开 Caribbean TVWS 传播与占用数据集，填补了该领域的数据缺口。

**🔧 技术方法**

使用的技术包括：CNN（ResNet18）频谱分类、Real‑ESRGAN+RIFE 超分辨率、OpenAirInterface 5G RAN、O‑RAN xApp 近实时控制、GPU 多实例分配和本地 5G UPF 本地跳转。

**📊 数据集**

采用了巴巴多斯现场测量的 TVWS 频谱记录与合成波形混合训练的 dataset，后续实验也使用了公开的 RadioML 与 SeaDronesSee 视觉数据。

**📈 对比分析**

与传统数据库依赖方案相比，SIDSense 在 94.2% 的感知准确率、23 ms 的决策延迟下，保持了 0 次 5G L1 时延违规，并在模拟 PAWS 中断期间维持了 100% 视频连通性，超越了仅依赖数据库查询的传统方案。

**⚠️ 局限性**

局限性包括：① 感知模型主要基于巴巴多斯数据，迁移到其他 SIDS 时需重新训练；② 仅在单船平台验证，缺乏多节点协作与真实灾害 RF 环境的测试；③ 依赖 5G 设备与 GPU 资源，部署成本相对较高。

---

## 347. A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)

**arXiv ID:** 2602.14696 | [PDF](https://arxiv.org/pdf/2602.14696v1)

**作者:** Nihal V. Nayak `[一作]` (Harvard University), David Alvarez-Melis `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型的目标任务指令选择问题，提出了将数据表示与选择算法解耦的框架，系统评估不同组合对下游性能的影响。

**💡 创新点**

创新点在于把指令选择拆解为数据表示和选择算法两部分，发现梯度基表示（LESS）能预测子集与查询集的距离与性能相关；同时将多种选择方法统一为距离最小化，并给出泛化上界。

**🔧 技术方法**

采用了梯度基表示（LESS）、位置加权隐藏状态表示（RDS+）、句子编码表示（EMBED），以及贪婪轮询（RR）、双贪婪（DG）、KNN-Uniform、KNN-KDE、未平衡最优传输（UOT）等选择算法；并使用Llama‑2‑7B等LLM进行微调。

**📊 数据集**

使用了Tulu V2作为候选池，目标任务包括BBH、Codex、GSM8K、TyDiQA、MMLU‑Pro等。

**📈 对比分析**

实验比较显示：在低预算下，LESS+RR能取得最佳或接近最佳的下游性能；在大预算时，UOT/KNN‑KDE优于其他方法；随机采样在大预算下接近最优；梯度表示在多数任务上优于模型基表示。

**⚠️ 局限性**

局限性包括：梯度基表示计算成本高；在某些任务上随机采样与目标任务分布差距较大时性能不一定提升；随着预算增大，选择方法优势递减；结果受LLM规模与数据集覆盖度影响。

---

## 348. MedVAR: Towards Scalable and Efficient Medical Image Generation via Next-scale Autoregressive Prediction

**arXiv ID:** 2602.14512 | [PDF](https://arxiv.org/pdf/2602.14512v1)

**作者:** Zhicheng He `[一作]` (National University of Singapore), Yueming Jin `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了MedVAR，一种基于next‑scale自回归的医学图像生成框架，能够以粗到细的方式合成CT和MRI图像，并提供可用于下游任务的多尺度结构化表示。

**💡 创新点**

创新点包括：①采用next‑scale预测实现并行粗到细生成，显著降低采样时间；②训练医学专属多尺度VQ‑VAE，解决自然图像预训练模型在医学图像上的代码本崩溃问题；③构建了约44万张覆盖6个解剖区域的统一CT/MRI数据集；④使用数据集标识作为条件，实现跨解剖/模态的统一基础模型；⑤提出了兼顾生成质量与推理成本的效率指标。

**🔧 技术方法**

技术手段包括多尺度VQ‑VAE离散化、下一尺度自回归Transformer、条件化dropout与classifier‑free guidance、top‑k/top‑p采样、AdaLN、混合精度训练等。

**📊 数据集**

使用约44万张CT/MRI图像，涵盖腹部、脑、胸、心、前列腺、脊柱等6个解剖区域，来源于公开基准（如Abdomen CT‑1K、BTCV、CHAOS等）以及7家医院的内部多中心腹部MRI数据（3,200例），并统一裁剪、标准化处理。

**📈 对比分析**

与StyleGAN‑3、DDPM、DiT等基线在256×256图像上对比，采用FID、RadFID、KID、CMMD等指标；MedVAR‑d30在FID 10.11、RadFID 10.11、KID 0.003、CMMD 0.205的同时，单张采样时间仅约0.16 s，显著快于100步的Diffusion（1.5–2.4 s）且质量更优；在效率指标下处于Pareto最优区域。

**⚠️ 局限性**

局限性包括：仅在2D切片上训练，缺乏完整3D体素一致性；未提供病灶或文本等高级条件控制；对模态不平衡（MRI样本相对不足）仍有限；临床可用性尚未在真实临床场景中验证。

---

## 349. Benchmarking at the Edge of Comprehension

**arXiv ID:** 2602.14307 | [PDF](https://arxiv.org/pdf/2602.14307v1)

**作者:** Samuele Marro `[一作]` (University of Oxford), Philip Torr `[通讯]` (University of Oxford)

**通讯引用:** 58641 | [OpenAlex ID](https://openalex.org/A5042899882)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Critique‑Resilient Benchmarking框架，用对抗式生成与局部验证的方法，在人类无法完整验证答案的后理解阶段评估大型语言模型；

**💡 创新点**

创新点在于用批判性可容忍正确性替代传统完整真值键，将基准设计本身度量为可度量的能力，并用项化双向Bradley‑Terry模型估计答案者与基准者的强度；

**🔧 技术方法**

技术包括：对抗式问答生成、局部错误证据（如反例、失效测试）、争论式交互、自动与人工裁判、Bipartite Bradley‑Terry 逻辑回归、MAP 估计与bootstrap 置信区间；

**📊 数据集**

数据集为自生成的352道数学题（44 MSC 类别 × 8 LLMs），每道题都有模型自造答案，随后由其他模型回答并产生批判；

**📈 对比分析**

评估通过计算答案者 Elo（β）和基准者 Elo（α），与 AIME 2025、BRUMO 2025、HMMT 2025 等传统数学基准的 Spearman/Kendall 相关系数高，β 与外部基准正相关；排名稳定，弱模型替代人类裁判时仍保持高度一致；

**⚠️ 局限性**

局限性在于只适用于能提供局部可验证证据的领域；批判与裁判的质量取决于当前模型；当未来模型超越此类局部验证时，框架可能失效；非可见证领域（如创造性或伦理推理）难以应用。

---

## 350. Toward a Military Smart Cyber Situational Awareness (CSA)

**arXiv ID:** 2602.14116 | [PDF](https://arxiv.org/pdf/2602.14116v1)

**作者:** Anthony Feijó-Añazco `[一作]` (University of Murcia), Gregorio Martínez Pérez `[通讯]` (University of Murcia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对5个现有的网络情境感知平台进行功能比较，提出了针对军事智能情境感知的6项评估标准，并在开源平台CRUSOE上进行实验验证。

**💡 创新点**

创新点在于系统化地提出了满足军用多域、用户适配、可视化增强等维度的6项评估标准，并通过实验对这些标准进行验证，为后续军事情境感知平台的设计提供了实证基础。

**🔧 技术方法**

使用了开源CSA平台CRUSOE，基于Neo4j构建知识图谱，集成网络流量、漏洞数据库、威胁传播预测及OODA循环决策逻辑，并通过Docker容器化部署。

**📊 数据集**

实验所用数据集来自CRUSOE官方公开数据，包括网络流量日志、漏洞信息、资产清单等多源异构数据。

**📈 对比分析**

通过功能维度（用户类型、数据源、运维环境、支持用例、可视化方式、军事需求）对比5个平台，并以CRUSOE为例评估6条标准，实验显示其在感知和分析阶段表现良好，但在决策与行动阶段仍存在不足。

**⚠️ 局限性**

CRUSOE缺乏多域（网络+作战）集成、军用层级权限和地理空间可视化，以及完整的决策与行动模块，导致其在实际军事情境中的可用性受限。

---

## 351. FloCA: Towards Faithful and Logically Consistent Flowchart Reasoning

**arXiv ID:** 2602.14035 | [PDF](https://arxiv.org/pdf/2602.14035v1)

**作者:** Jinzi Zou `[一作]` (Xi'an Jiaotong University), Junzhou Zhao `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 1138 | [OpenAlex ID](https://openalex.org/A5007402211)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了面向流程图的对话任务（FOD），开发了零样本对话代理 FloCA，使用 LLM 进行意图理解与回复生成，外部图形工具完成拓扑约束的流程推理。

**💡 创新点**

创新点在于：① 将流程推理与 LLM 解耦，采用拓扑约束的图执行工具保证节点转移的逻辑一致；② 设计了交互式 LLM 用户模拟器与一组新评估指标，能够更真实地测评对话路径覆盖和效率；③ 在零样本条件下实现了高效、可信的流程图对话。

**🔧 技术方法**

核心技术包括 instruction‑following 大语言模型（如 GPT‑4o/5、Claude、Llama‑3 等）、拓扑约束的图形工具（提供节点属性、边属性、转移查询等接口）、RAG、VLM 与图序列化方法做对比实验、以及基于 LLM 的用户模拟器。

**📊 数据集**

使用 FLODIAL（12 份设备故障流程图，约 2.7K 对话）与 PFDial（PlantUML 格式 440 份流程图，合成 353/416 对话）进行评估。

**📈 对比分析**

通过与 RAG、图序列化、VLM、Fine‑tune 等多种基线在 In‑Domain 与 Out‑of‑Domain 两种设置下对比，FloCA 在任务成功率（TNGA、PCA）和效率指标（NSR、TR）上均显著领先，表现为最高的整体成功率和最低的冗余/超时率。

**⚠️ 局限性**

限制包括：仍然依赖 LLM 的意图匹配可能出现错误，用户模拟器的真实性和多样性有限；对非常大、分支多的流程图精确度仍有提升空间；对外部知识检索错误仍可能影响推理路径。

---

## 352. XIT: Exploration and Exploitation Informed Trees for Active Gas Distribution Mapping in Unknown Environments

**arXiv ID:** 2602.13739 | [PDF](https://arxiv.org/pdf/2602.13739v1)

**作者:** Mal Fazliu `[一作]` (Loughborough University), Cunjia Liu `[通讯]` (Loughborough University)

**通讯引用:** 3891 | [OpenAlex ID](https://openalex.org/A5021774274)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文提出了一种面向未知、杂乱环境的主动气体分布映射框架，核心是XIT（Exploration–Exploitation Informed Trees）采样规划器，能够在同时进行环境探索、结构建模和气体推断时，动态选择最优路径；

**💡 创新点**

创新点包括：①将UCB信息场与距离代价相结合的采样与树扩展策略，实现对高浓度与高不确定区域的双重关注；②首次引入气体前沿（gas frontier）概念，并提出Wavefront Gas Frontier Detection（WGFD）算法；③设计了并行多目标树扩展机制，兼顾探测前沿与信息增益；

**🔧 技术方法**

采用了GMRF气体模型与GBP推理、SLAM基础的占据网格、基于bit*的采样树构建、UCB信息分布、前沿检测、波前算法以及距离与信息的加权代价；

**📊 数据集**

实验使用了高保真GADEN仿真器的多源乙醇泄漏数据集（16m×20m室内场景）和实际的5.0m×3.7m CO₂泄漏实验平台（TurtleBot3+COZIR-A传感器）；

**📈 对比分析**

与传统RRT*前沿探索基线进行对比，评价指标为RMSE、差分熵和地图完整度；XIT‑GFF+UCB配置在300s时RMSE降低25.5%，熵降低23.7%，而探索覆盖率与基线相近；

**⚠️ 局限性**

局限性包括：①采用单一帧的贪心规划，无法对未来多步信息进行非贪心评估；②仅在二维平面内测试，未验证3D或室外大尺度场景；③对气体传感器响应速度和动态气候变化的鲁棒性有限；

---

## 353. MarcoPolo: A Zero-Permission Attack for Location Type Inference from the Magnetic Field using Mobile Devices

**arXiv ID:** 2602.13915 | [PDF](https://arxiv.org/pdf/2602.13915v1)

**作者:** Beatrice Perez `[一作]` (University of Massachusetts), Mirco Musolesi `[通讯]` (University College London)

**通讯引用:** 12827 | [OpenAlex ID](https://openalex.org/A5078886343)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用手机磁力计开展零权限攻击，推断用户所在位置的粗粒度类型。

**💡 创新点**

首次通过磁场特征推断位置类型，无需权限或完整地图，并结合形状子（shapelets）与统计特征实现更佳识别。

**🔧 技术方法**

采用时间序列分类方法：完整信号匹配（DTW、欧氏等）、统计描述+kNN/RF/XGB、Siamese CNN自动特征提取、时域/频域形状子以及组合特征集。

**📊 数据集**

收集了约91小时的磁场数据，来自5部商用手机，覆盖110个地点、10种位置类型的真实环境测量。

**📈 对比分析**

与全信号匹配、统计描述、自动特征等四种方法对比，形状子+RF/XGB在留一地点/留一设备评估中分别达40.5%和39.5%准确率（随机基线约16.7%），完整信号匹配仅18%；组合特征进一步提升约7%。

**⚠️ 局限性**

受限于磁力计采样速率、不同地点磁场相似导致误判、传感器噪声、OS未来可能限制无权限传感器访问，且仅能实现粗粒度推断。

---

## 354. Push-Placement: A Hybrid Approach Integrating Prehensile and Non-Prehensile Manipulation for Object Rearrangement

**arXiv ID:** 2602.13849 | [PDF](https://arxiv.org/pdf/2602.13849v1)

**作者:** Majid Sadeghinejad `[一作]` (University of Tehran), Ahmad Kalhor `[通讯]` (University of Tehran)

**通讯引用:** 1891 | [OpenAlex ID](https://openalex.org/A5035497083)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出了一种“push-placement”混合操作原语，用抓握物体在放置时同时推动阻碍物，减少传统 pick‑and‑place 中的缓冲动作，从而实现桌面物体重新排列。

**💡 创新点**

创新点在于将抓取与推挤两种操作融合为单一原语，并将其嵌入到 Monte Carlo Tree Search（MCTS）规划框架中，利用物理引擎的闭环验证来补偿推挤不确定性，显著降低移动成本。

**🔧 技术方法**

采用的技术包括：基于几何的可行性检查、物理引擎 PyBullet 作为闭环仿真器、时间约束的 MCTS 规划、以及基于轴对齐的推挤约束模型。

**📊 数据集**

使用的数据集为在 PyBullet 中随机生成的 300 个桌面场景（每个场景包含 3~11 个不同尺寸物体），并在 100 个包含 8 个物体的场景上进行仿真实验。

**📈 对比分析**

与仅使用 pick‑and‑place（MCTS）以及动态堆叠（MCTS+DS）对比，push-placement 在平均运动成本上分别降低了最多 11.12% 和 8.56%，同时动作数也减少，证明了该方法在更拥挤场景下的优越性能。

**⚠️ 局限性**

局限性包括：假设物体足迹为轴对齐、对大角度旋转和倾覆等极端动态处理不足、以及实验仅在仿真环境中验证，缺乏实时感知与真实机器人部署的实证。

---

## 355. SemanticFeels: Semantic Labeling during In-Hand Manipulation

**arXiv ID:** 2602.14099 | [PDF](https://arxiv.org/pdf/2602.14099v1)

**作者:** Anas Al Shikh Khalil `[一作]` (Dresden University of Technology), Roberto Calandra `[通讯]` (Dresden University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在机器人手部操作过程中，结合视觉与触觉传感，实现对物体材料的实时语义标注，并将该信息嵌入神经隐式形状表示中。

**💡 创新点**

将语义材料分类器（基于EfficientNet‑B0的触觉图像识别）与原NeuralFeels的神经SDF网络并行扩展，使得同一模型可同时输出几何和材料标签，实现触觉与视觉信息的融合与实时推理。

**🔧 技术方法**

采用高分辨率视触式Digit传感器采集触觉图像；使用EfficientNet‑B0进行材料分类；利用HashGrid + MLP构建的神经SDF网络与球谐编码扩展的材料分支；在多摄像头RGB‑D + 强化学习控制的Allegro手上部署。

**📊 数据集**

包含20,749张Digit触觉图像，四类材料（塑料、金属、织物、木材）均衡采样，划分70%训练、15%验证、15%测试，数据来源于四根手指传感器的离线采集。

**📈 对比分析**

与传统基于视觉或手工特征的材料识别方法相比，SemanticFeels在单一材料测试中多指传感器平均准确率超过98%，在多材料物体上实现平均匹配准确率79.87%（标准差4.41%），展示了触觉与隐式表示结合的显著优势。

**⚠️ 局限性**

局部触碰不一致导致拇指传感器识别准确率低；仅涵盖四种材料，缺乏更丰富纹理与刚度的泛化；仅使用触觉数据，未结合视觉或力学信息；预设旋转策略不自适应，影响传感器覆盖与信息质量。

---

## 356. Preventing Rank Collapse in Federated Low-Rank Adaptation with Client Heterogeneity

**arXiv ID:** 2602.13486 | [PDF](https://arxiv.org/pdf/2602.13486v1)

**作者:** Fei Wu `[一作]` (University of Exeter), Shiqiang Wang `[通讯]` (University of Exeter)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5010827975)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对联邦低秩适配（FedLoRA）在客户端异构情境下的rank collapse现象进行理论与实验研究，并提出rank-partitioned aggregation方法来解决该问题。

**💡 创新点**

创新点：①首次揭示rank collapse的根源是rank-wise averaging mismatch；②提出基于rank划分的聚合策略，使聚合权重与各秩方向的有效参与者匹配，从而消除高秩信息的逐轮衰减。

**🔧 技术方法**

技术：理论分析（固定奇异基、方向保持更新、几何衰减证明）、SVD重构、rank-partitioned aggregation算法、实验评估与可视化。

**📊 数据集**

数据集：ViT-base 在 CIFAR-100、RoBERTa-base 在 20 Newsgroups、LLaMA-3.2-3B/3.1-8B 在 GSM8K、Commonsense15K 在八个常识推理基准。

**📈 对比分析**

与 Zero-padding、Stacking、SVD-based FedLoRA 等基线对比，实验表明在多任务、多数据分布下可提升 2–3% 的准确率，且通信成本与基线相当，显著抑制了 rank collapse。

**⚠️ 局限性**

局限性：在极低秩或低秩偏斜的分布下优势有限；模型规模增大时会产生额外的计算开销；需预先设定秩划分边界，可能影响灵活性。

---

## 357. DTBench: A Synthetic Benchmark for Document-to-Table Extraction

**arXiv ID:** 2602.13812 | [PDF](https://arxiv.org/pdf/2602.13812v1)

**作者:** Yuxiang Guo `[一作]` (Zhejiang University), Yunjun Gao `[通讯]` (Zhejiang University)

**通讯引用:** 5312 | [OpenAlex ID](https://openalex.org/A5006238145)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套二层级的文档到表格（Doc2Table）抽取能力词汇表，并基于此设计了可扩展的多智能体逆向Table2Doc生成流程，构建了首个面向能力的合成基准DTBench，覆盖5大类13个细分子能力，包含120个案例、8811个细胞级别实例；

**💡 创新点**

创新点包括①基于能力词汇表的细粒度评价框架；②逆向Table2Doc合成的多智能体工作流，能自动生成满足指定抽取能力且唯一可还原的文档；③对传统单一指标进行拆解，系统评估LLM在不同抽取能力上的表现；

**🔧 技术方法**

使用的大型语言模型（如Qwen、Llama、DeepSeek、GPT-5、Gemini）进行推理；构建的多智能体框架由LLM（生成、验证、规划、写作）与代码（校验、匹配、组装）协同完成；同时提出了两层能力映射模型与检查表验证机制；

**📊 数据集**

数据来源为Kaggle、Wikipedia及公共融合数据集，经过人工裁剪与结构化，随后通过Table2Doc流程合成对应的文档；基准集DTBench公开可用；

**📈 对比分析**

对八大主流LLM在DTBench上进行对比，使用Precision、Recall、F1以及直接/间接抽取召回率指标；实验显示虽然模型整体F1已达80%+，但间接抽取召回率相较直接抽取显著低下（最高约15%差距），多跳推理、证据真实性与隐式冲突解决仍是主要瓶颈；

**⚠️ 局限性**

局限性在于合成数据可能无法完全覆盖真实文档的多样性与噪声；逆向生成对LLM的质量和验证机制高度依赖，存在潜在的生成错误与覆盖不足；缺乏在真实业务场景中大规模评估验证的支持；

---

## 358. Discrete Double-Bracket Flows for Isotropic-Noise Invariant Eigendecomposition

**arXiv ID:** 2602.13759 | [PDF](https://arxiv.org/pdf/2602.13759v1)

**作者:** ZhiMing Li `[一作]`, JiaHe Feng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种离散双括号流算法，用于在矩阵自由的环境下对带有各向同性噪声的协方差矩阵进行全谱分解。

**💡 创新点**

创新点在于通过 Lie 括号生成器实现对 σ² 的完全免疫，借助输入到状态稳定性和严格鞍点几何保证全局收敛，并利用 Givens 旋转实现确定性鞍点逃逸。

**🔧 技术方法**

使用的主要技术包括离散双括号流（commutator 生成器）、Cayley 退行、输入到状态稳定性分析、严格鞍点理论以及 Givens 旋转逃逸策略。

**📊 数据集**

实验数据采用随机正交变换的对角协方差矩阵，并在不同幅度的各向同性噪声（σ² 从 0 到 10⁶）下进行。

**📈 对比分析**

与 Oja、子空间迭代等传统方法对比，本文方法在噪声增大时保持迭代次数和计算时间基本不变，显著优于基线，尤其在 σ²≫C_sig 时实现 O(1) 的收敛复杂度。

**⚠️ 局限性**

局限性包括每次迭代需要 O(n³) 的算术运算，难以扩展到大规模 n，且只对各向同性噪声具免疫性，对结构化噪声或非矩阵自由情形适应性有限。

---

## 359. Tabular Foundation Models Can Learn Association Rules

**arXiv ID:** 2602.14622 | [PDF](https://arxiv.org/pdf/2602.14622v1)

**作者:** Erkan Karabulut `[一作]` (University of Amsterdam), Victoria Degeler `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种模型无关的关联规则学习框架，并用 Tabular Foundation Models（TFMs）在不进行频繁项集挖掘的前提下实现关联规则的自动提取。

**💡 创新点**

创新点在于：①将任何满足前件验证与后件提取条件的条件概率模型映射为关联规则学习器；②首次将 TFMs 的零训练、in‑context 学习能力用于 ARM，实现低数据量下规则稀疏、覆盖完整；③提供了 TabProbe 算法与单目标与多目标两种实例化方案。

**🔧 技术方法**

使用技术包括：条件概率模型（如 TabPFN、TabICL、TabDPT）、单目标推理、前件验证阈值与后件阈值、probing 方式构造输入、Rule Extraction 算法（多目标与单目标实现）以及在 CBA 与 CORELS 中评估规则的下游性能。

**📊 数据集**

实验数据集为：5 个小型（约 100 行）医学数据集和 5 个中大型（300–8000 行）UCI 公开数据集。

**📈 对比分析**

与 FP‑Growth、Aerial+ 等基线比较：TFMs 在规则数量上显著减少（约 7–10 倍），覆盖率达到 100%，关联强度与有趣度均高于基线；在 CBA/CORELS 的下游分类中，TFMs 的准确率与 F1 与 FP‑Growth 差距仅 1–1.5%，并保持更简洁的规则集；但在大规模表（>120 个项目集合）下执行时间略慢。

**⚠️ 局限性**

局限性：①受限于 TFMs 的上下文长度（目前上限 ~500K 行），不适用于极大表；②单目标实现需逐特征预测，导致计算量相对多；③仍依赖阈值设定，需要手动调参以平衡规则数量与质量；④目前仅支持离散/分类特征，尚未扩展到数值或图结构。

---

## 360. COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception

**arXiv ID:** 2602.13287 | [PDF](https://arxiv.org/pdf/2602.13287v1)

**作者:** Shilpa Mukhopadhyay `[一作]` (University of California), Hang Qiu `[通讯]` (University of California)

**通讯引用:** 1093 | [OpenAlex ID](https://openalex.org/A5024716590)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种自适应特征选择框架，在协同感知中利用时间不确定性动态决定共享特征量，从而显著降低带宽需求。

**💡 创新点**

创新点在于①基于 conformal prediction 的时序不确定性度量评估特征相关性；②引入两个可学习阈值实现共享量的自适应控制；③结合 ϵ‑greedy 训练策略平衡带宽与性能。

**🔧 技术方法**

采用 conformal prediction inspired quantile gating、注意力机制、Lagrangian 约束损失、ϵ‑greedy 探索训练、时序不确定性估计与自适应阈值学习等技术。

**📊 数据集**

在 OPV2V 与 V2V4Real 两大数据集上进行实验，分别用于语义分割与 3D 检测任务。

**📈 对比分析**

与多种开源协同分割/检测模型及现有选择/压缩基线（Where2Comm、SwissCheese、CoBEVT、AttFuse 等）对比，平均带宽降低 80.28%（分割）/72.52%（检测），同时保持相当精度；在 32×压缩下仅占 1.46% 带宽仍维持良好 IoU，且相较基线提升 IoU 45.54%。

**⚠️ 局限性**

仍存在一定计算开销导致 FPS 降低约 2 帧/秒；对极端定位误差和高延迟的鲁棒性尚有限，且在不同网络环境下的稳定性需进一步验证。

---

## 361. Universal Image Immunization against Diffusion-based Image Editing via Semantic Injection

**arXiv ID:** 2602.14679 | [PDF](https://arxiv.org/pdf/2602.14679v1)

**作者:** Chanhui Lee `[一作]` (POSTECH AI Graduate School), Jeany Son `[通讯]` (POSTECH AI Graduate School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种通用图像免疫框架，通过生成单一的目标攻击UAP来抵御基于扩散模型的恶意图像编辑。

**💡 创新点**

①首次提出基于通用UAP的图像免疫；②通过目标语义注入与源语义抑制双重损失在扩散模型交叉注意力上实现语义覆盖；③实现无数据依赖的训练。

**🔧 技术方法**

目标语义注入损失、源语义抑制损失；利用扩散模型的交叉注意力与VAE编码器；生成单一UAP并在推理时仅做加法；数据无关训练使用随机拼图噪声。

**📊 数据集**

训练使用10k LAION‑2B‑en image‑prompt对；数据无关使用随机拼图噪声；评估用50张图像（10类各5张）配合多种编辑提示；基准评估涉及Stable Diffusion V1.5、V1.4、V2.0与InstructPix2Pix。

**📈 对比分析**

与三种改编自图像特定方法的通用基线（Encoder、Embedding、Map）以及现有图像特定免疫方法（PhotoGuard、Semantic Attack、EditShield）比较。实验显示本方法在所有指标上显著优于通用基线，在黑盒迁移和数据无关情形下仍保持竞争力，且推理时间几乎为零。

**⚠️ 局限性**

依赖目标语义的注入，若目标选择不当或攻击者具备逆向工程能力，效果可能下降；对极端图像或复杂编辑任务仍有挑战；对防御的适应性攻击（如Conditional DiffPure）仍需进一步验证。

---

## 362. Unleash the Potential of Long Semantic IDs for Generative Recommendation

**arXiv ID:** 2602.13573 | [PDF](https://arxiv.org/pdf/2602.13573v1)

**作者:** Ming Xia `[一作]` (Southern University of Science and Technology), Dongmin Huang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 4726 | [OpenAlex ID](https://openalex.org/A5100636785)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ACERec 框架，利用长语义 ID 序列进行注意力合并（ATM）压缩成低维隐层，再通过 Intent Token 作为动态锚点实现并行生成的可生成推荐。

**💡 创新点**

创新点包括：① 将 token 细粒度与推荐过程解耦，使用 ATM 将 32 个语义 ID 归约为 4 个高效隐层；② 引入 Intent Token 作为动态预测锚；③ 采用双粒度训练目标（Token 预测 + 目标项目语义对齐）提升语义一致性。

**🔧 技术方法**

使用技术包括：OPQ 语义 ID 生成、Attentive Token Merger、跨注意力聚合、Intent Token、双粒度损失（MTP + ISA）、全局候选得分、并行解码、对比学习与流行度校正。

**📊 数据集**

实验数据集：六个 Amazon Reviews 真实数据集——Sports、Beauty、Toys、Instruments、Office、Baby。

**📈 对比分析**

与 ID 基线（HGN、SASRec、S³-Rec、ICLRec、ELCRec）和语义 ID 基线（TIGER、ActionPiece、ETEGRec、RPG）进行比较，评估 Recall@K 与 NDCG@K；ACERec 在所有数据集均领先，平均 NDCG@10 提升约 14.4%，在最强基线上提升高达 22.15%。

**⚠️ 局限性**

局限性：① OPQ 码本冲突仍可能导致信息损失；② ATM 的压缩比例需权衡，过高压缩会显著下降性能；③ 对极大目录的可扩展性仍待进一步验证；④ 仅在 Amazon 领域评测，跨域泛化能力未知。

---

## 363. TemporalBench: A Benchmark for Evaluating LLM-Based Agents on Contextual and Event-Informed Time Series Tasks

**arXiv ID:** 2602.13272 | [PDF](https://arxiv.org/pdf/2602.13272v1)

**作者:** Muyan Weng `[一作]` (University of Southern California), Yan Liu `[通讯]` (University of Southern California)

**通讯引用:** 70112 | [OpenAlex ID](https://openalex.org/A5100351175)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并发布了 TemporalBench，系统评估 LLM 及其代理在包含时间序列、上下文与事件信息的多任务推理与预测场景下的表现。

**💡 创新点**

创新点在于：①四层任务分层（T1–T4）细化历史解读、无上下文预测、上下文化推理与事件驱动预测；②统一的数据转换流水线实现跨域一致的事件划分与标签生成；③通过任务与子任务的组合揭示模型在数值预测与语义推理之间的分离。

**🔧 技术方法**

使用大规模预训练语言模型（如 GPT‑4o、Claude‑3 等）搭配 AgentScope、MetaGPT、CAMEL、TimeSeriesScientist 等代理框架，辅以规则式标签生成、事件注入/检测和可视化/特征增强输入。

**📊 数据集**

四个真实世界时间序列数据集：FreshRetailNet（零售）、MIMIC‑IV（医疗）、PSML（能源）与 Causal Chambers（物理系统）。

**📈 对比分析**

对比方法包括直接 LLM 提示、四种代理框架和特定领域时间序列代理；通过多选准确率与 MAE / sMAPE / OW_RMSSE 等指标评估；结果表明，数值预测精度高并不转化为上下文或事件推理准确，代理框架对不同任务表现差异显著。

**⚠️ 局限性**

局限性在于：①评测仍以无训练的“零样本”方式，未检验模型在自适应学习上的潜力；②对复杂因果事件的解释仍不充分，无法完全模拟真实世界多因子交互；③部分任务（尤其是 T3 的比较与分布推理）仍难以衡量模型真正的推理深度。

---

## 364. Multimodal Covariance Steering in Belief Space with Active Probing and Influence for Autonomous Driving

**arXiv ID:** 2602.14540 | [PDF](https://arxiv.org/pdf/2602.14540v1)

**作者:** Devodita Chakravarty `[一作]` (Indian Institute of Technology Kharagpur), Yiwei Lyu `[通讯]` (Texas A and M University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个基于分层信念模型、主动探测和CVaR风险控制的安全交互驾驶框架。

**💡 创新点**

将分层意图-运动信念与主动探测和尾部风险约束耦合，实现了对人类驾驶行为的主动模糊性消除与影响。

**🔧 技术方法**

采用贝叶斯推断的分层信念更新、熵最小化主动探测、协方差引导的控制、CVaR风险约束与MPC求解等技术。

**📊 数据集**

在仿真中使用自定义的车道合并和无信号交叉口场景，随机生成车辆状态，未使用公开真实数据集。

**📈 对比分析**

与AP-IH、CC-MPC、AP-MP三种基线对比，实验显示成功率提升至96%/94%，合并/通过时间分别缩短至约2 s/4 s，且尾部风险控制有效。

**⚠️ 局限性**

在车辆密集、高速或探测空间受限的情形下，主动探测难以充分消除模糊性，导致潜在碰撞风险未完全消除。

---

## 365. CoCoDiff: Correspondence-Consistent Diffusion Model for Fine-grained Style Transfer

**arXiv ID:** 2602.14464 | [PDF](https://arxiv.org/pdf/2602.14464v1)

**作者:** Wenbo Nie `[一作]` (Institute of Information Science), Yao Zhao `[通讯]` (Institute of Information Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了CoCoDiff框架，利用预训练的潜在扩散模型实现无训练的细粒度、结构保持的图像风格迁移；

**💡 创新点**

创新点在于通过挖掘扩散模型中间特征获取像素级语义对应，并引入循环一致性机制指导风格注入，从而实现语义一致且细腻的风格迁移；

**🔧 技术方法**

技术手段包括Stable Diffusion预训练模型、特征提取与余弦相似度匹配、注意力键值替换、AdaIN色彩归一、循环迭代优化以及多指标评估；

**📊 数据集**

实验使用MS‑COCO作为内容图像、WikiArt作为风格图像，并在SPair‑71k上评估对应质量，此外还对不同版本的Stable Diffusion模型进行了验证；

**📈 对比分析**

与九种主流方法（包括基于注意力、提示式和传统CNN/Transformer方法）比较，CoCoDiff在FID、LPIPS、ArtFID、CFSD等指标上均表现最优，用户研究也显示其在风格保真与内容保持方面的优越性；

**⚠️ 局限性**

局限性在于仍需依赖预训练模型的特征表达，需手动挑选最佳时间步与层次，循环迭代增加计算成本，并且在极端风格差异或超大尺寸图像下的鲁棒性尚未充分验证。

---

## 366. High Precision Audience Expansion via Extreme Classification in a Two-Sided Marketplace

**arXiv ID:** 2602.14358 | [PDF](https://arxiv.org/pdf/2602.14358v1)

**作者:** Dillon Davis `[一作]` (Airbnb, Inc.), Sanjeev Katariya `[通讯]` (Airbnb, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在Airbnb搜索系统中，将位置检索从传统的矩形边界预测改为利用极端多分类模型预测最可能出现预订的S2网格单元，以实现更精准的地理检索；

**💡 创新点**

创新点在于将地球划分为2500万级别11的S2网格，将每个网格视为分类标签，利用极端多分类技术（采样softmax、区域分片、深度神经网络）实现大规模地理位置预测；

**🔧 技术方法**

采用S2空间索引、深度四层全连接网络、采样softmax损失、区域分片训练策略、Lucene的OR查询等技术；

**📊 数据集**

使用了约5300万条（1年）Airbnb预订-搜索对应样本作为训练集，10万条（7天）预订-搜索样本作为验证集；

**📈 对比分析**

与基线矩形边界检索模型对比，召回率基本持平、精确率提升约11%，检索到的房源数量增加约9.9%，线上A/B实验中未取消预订用户数提升0.17%；

**⚠️ 局限性**

局限性包括计算成本较高、阈值需手工调优、对极端分布或跨境搜索的泛化能力有限、模型复杂度和训练时间高，以及需进一步探索层级softmax和动态阈值。

---

## 367. Who Do LLMs Trust? Human Experts Matter More Than Other LLMs

**arXiv ID:** 2602.13568 | [PDF](https://arxiv.org/pdf/2602.13568v1)

**作者:** Anooshka Bajaj `[一作]` (Indiana University), Zoran Tiganj `[通讯]` (Indiana University)

**通讯引用:** 1138 | [OpenAlex ID](https://openalex.org/A5021423914)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在面对来自人类专家、朋友和其他LLM的社会信息时的顺从与信息整合行为。

**💡 创新点**

揭示了LLM对专家来源的可信度权重显著高于其他来源，并在群体一致性与冲突情境下表现出类似人类的社会影响。

**🔧 技术方法**

使用指令微调的四款LLM（Grok-3 Mini、Llama 3.3 70B Instruct、Gemini 2.5 Flash‑Lite、DeepSeek V3.1）与自定义提示、logit/概率分析等技术。

**📊 数据集**

使用三种二分类数据集：BoolQ、StrategyQA、ETHICS，各300条测试样本。

**📈 对比分析**

通过实验1的同源先验与实验2的混合先验对比，利用对数几率与logit差异评估影响力，结果显示专家先验在正确与错误情境下均显著提升模型的顺从率。

**⚠️ 局限性**

局限性包括先验仅为文本摘要、随机化正误导致无法评估真实可靠性、仅限于二分类任务、未探究开放式生成场景以及对模型训练差异的鲁棒性。

---

## 368. A Bayesian Framework for Human-AI Collaboration: Complementarity and Correlation Neglect

**arXiv ID:** 2602.14331 | [PDF](https://arxiv.org/pdf/2602.14331v1)

**作者:** Saurabh Amin `[一作]` (Massachusetts Institute of Technology), Asuman Ozdaglar `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 27994 | [OpenAlex ID](https://openalex.org/A5067307504)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

构建决策理论框架，分析 AI 协助人类决策时的增益与损害，并将效益拆分为信息增益与行为扭曲两部分。

**💡 创新点**

提出信息重叠系数 λ 的微观基础定义，揭示 AI 与人类信息重叠如何决定增益与失效，并给出不同重叠程度下的阈值与相位转换。

**🔧 技术方法**

采用贝叶斯推理、正态信号模型、Bregman 损失函数和解析式推导，阐释在相关性忽视（correlation‑neglect）情形下 AI 对人类决策的影响。

**📊 数据集**

未使用具体数据集，研究基于理论模型的泛化结果；若需验证，可在医学诊断、图像分类等已有公开数据集上实施实验。

**📈 对比分析**

通过比较人类单独决策、AI 单独决策与 AI‑辅助人类决策的期望损失，给出阈值函数 τ_aug(λ)、τ_auto(λ) 等，展示在不同重叠水平下的补充性、损害或自动化优势；理论分析显示随着 AI 能力提升，自动化趋于不可避免。

**⚠️ 局限性**

局限在于仅考虑相关性忽视偏差、假设信息重叠对 AI 能力增量不变、未进行实证验证；对其他行为偏差或多轮交互情形缺乏分析。

---

## 369. GeoFusionLRM: Geometry-Aware Self-Correction for Consistent 3D Reconstruction

**arXiv ID:** 2602.14119 | [PDF](https://arxiv.org/pdf/2602.14119v1)

**作者:** Ahmet Burak Yildirim `[一作]` (Bilkent University), Aysegul Dundar `[通讯]` (Bilkent University)

**通讯引用:** 1337 | [OpenAlex ID](https://openalex.org/A5041525280)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在单张图像3D重建中提出一种几何自校正框架，利用模型自身生成的深度和法向作为几何指示进行两阶段细化，提升几何一致性

**💡 创新点**

创新点在于将几何指示回馈到Transformer并通过GeoFormer和轻量化的GeoFuser实现自监督的几何细化，而不依赖外部监督或额外视图

**🔧 技术方法**

采用Transformer、AdaLN条件的Vision Encoder、GeoFormer几何编码器、GeoFuser融合模块、Diffusion生成多视图、Triplane解码、可微渲染

**📊 数据集**

在Objaverse‑1.0训练集上训练，评估使用OmniObject3D和Google Scanned Objects两个公开数据集

**📈 对比分析**

与LRM、SPAR3D、LGM和InstantMesh等基线对比，实验显示在RGB与法向指标上均优于基线，尤其在法向上显著提升，PSNR/SSIM提升约2–3 dB，LPIPS下降约0.02

**⚠️ 局限性**

局限在于推理时需要额外的前向传递导致计算量和时间翻倍，对极细小结构仍难以恢复，且对极其细腻的薄枝仍缺失

---

## 370. Anticipating Adversary Behavior in DevSecOps Scenarios through Large Language Models

**arXiv ID:** 2602.14106 | [PDF](https://arxiv.org/pdf/2602.14106v1)

**作者:** Mario Marín Caballero `[一作]`, Gregorio Martínez Pérez `[通讯]` (University of Murcia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过LLM生成攻击-防御树并在DevSecOps场景下验证其有效性

**💡 创新点**

提出基于LLM的攻击-防御树生成流程和三维质量评估指标

**🔧 技术方法**

使用GPT‑4、Qwen‑32B等大型语言模型以及Graphviz DOT格式

**📊 数据集**

以军用物流系统的AWS GovCloud架构为案例，未使用公开数据集

**📈 对比分析**

对比GPT‑4与Qwen‑32B的树得分，Qwen‑32B得分71.6%高于GPT‑4的61.7%，并在SCE实验中实现权限提升

**⚠️ 局限性**

受限于LLM的幻觉、需要人工验证、仅在单一场景验证、未提供自动化对策推荐

---

## 371. Picking the Right Specialist: Attentive Neural Process-based Selection of Task-Specialized Models as Tools for Agentic Healthcare Systems

**arXiv ID:** 2602.14901 | [PDF](https://arxiv.org/pdf/2602.14901v1)

**作者:** Pramit Saha `[一作]` (University of Oxford), J. Alison Noble `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于查询条件的工具选择框架 ToolSelect，能够在 agentic 医疗系统中为每个临床查询自动路由到最合适的专业模型。

**💡 创新点**

创新点在于将工具选择问题视为群体风险最小化，并利用 Attentive Neural Process 与工具行为摘要实现无训练目标模型的动态路由。

**🔧 技术方法**

采用 Attentive Neural Process、comp-sum surrogate loss、自注意力机制及工具行为摘要等技术进行选择器训练。

**📊 数据集**

构建 Agentic Chest X‑ray 环境，收集 17 个疾病检测、19 个报告生成、6 个视觉定位、13 个 VQA 模型，并用 ToolSelectBench（1448 个查询）进行评估。

**📈 对比分析**

与 10 种最先进基线（KNN、SVM、MLP、LLM 路由等）对比，ToolSelect 在四类任务中均显著提升准确率/召回率，逼近 Oracle 上限。

**⚠️ 局限性**

局限在于对工具行为摘要的依赖需要预先生成参考集，并且在极端领域迁移或标签空间差异时仍可能出现不稳定性。

---

## 372. Pseudo-differential-enhanced physics-informed neural networks

**arXiv ID:** 2602.14663 | [PDF](https://arxiv.org/pdf/2602.14663v1)

**作者:** Andrew Gracyk `[一作]` `[通讯]` (Purdue University), Andrew Gracyk (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出伪微分增强物理信息神经网络（Pseudo‑Differential PINN），通过在傅里叶空间对PDE残差做梯度增强，从而在训练中强化高频信息。

**💡 创新点**

创新点在于将传统的梯度增强迁移到傅里叶域，实现一次乘法而非多次自动微分；同时通过多项式符号（P(ξ)）对高频做加权，显著缓解网络的频谱偏置，并可直接应用于分数阶导数与非欧几里得域。

**🔧 技术方法**

核心技术包括：傅里叶变换与逆变换、伪微分算子符号、FFT/随机蒙特卡洛网格、量化损失、NTK谱衰减分析、以及多种先进PINN增强手段（Fourier特征嵌入、SOAP优化器、梯度范数调节）。

**📊 数据集**

使用标准PDE基准数据集：Allen‑Cahn、Burger’s、KdV、Navier‑Stokes等，通过数值参考解（FFT或显式解）进行误差评估。

**📈 对比分析**

与传统PINN对比，Pseudo‑Differential PINN在相同训练迭代数下能更快突破梯度停滞、提高高频学习率，整体误差比传统方法低约10%–20%，并在低采样点数下表现更稳健；实验中也展示了内存占用与梯度流的可比性。

**⚠️ 局限性**

局限性包括：对高频模式的乘法会导致边界权重偏高，需要截断模式；在非周期或复杂几何域时FFT不直接适用，需使用蒙特卡洛或非均匀FFT；需要对符号P(ξ)和权重系数进行经验调参；以及在极低采样点数下仍可能出现收敛不稳定。

---

## 373. S2D: Selective Spectral Decay for Quantization-Friendly Conditioning of Neural Activations

**arXiv ID:** 2602.14432 | [PDF](https://arxiv.org/pdf/2602.14432v1)

**作者:** Arnav Chavan `[一作]` (Amazon), Deepak Gupta `[通讯]` (Amazon)

**通讯引用:** 16283 | [OpenAlex ID](https://openalex.org/A5034853815)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大规模 Transformer 模型中激活异常值（outliers）的形成机理，证明其源于权重矩阵的谱失衡，并提出Selective Spectral Decay（S²D）正则化方法，仅针对主奇异值进行惩罚，从而显著降低激活 outliers 并提升量化性能。

**💡 创新点**

创新点在于：① 将激活 outliers 与权重主奇异值的关联通过 Principal Component Dominance Ratio (PCDR_k) 量化；② 设计仅惩罚主奇异值的 Selective Spectral Decay 正则化；③ 结合稀疏 SVD 与 PCDR_k 选择机制，使方法在 fine‑tune 阶段即可高效应用。

**🔧 技术方法**

使用了奇异值分解（SVD）、PCDR_k 诊断、Selective Spectral Decay 正则化、稀疏/分段 SVD 计算、后训练量化（PTQ）技术（ERQ、PTQ4ViT、RepQ‑ViT）以及量化感知训练（QAT）等。

**📊 数据集**

实验数据集包括：CLIP、SigLIP、SigLIP2 预训练模型；ImageNet‑1k（分类、PTQ/QAT）；MS‑COCO（目标检测、实例分割）；以及 VLM 评测基准 GQA、TextVQA、POPE、DocVQA。

**📈 对比分析**

与传统 AdamW 以及 Muon 等优化器做对比；在 PTQ 和 QAT 场景下，S²D 在 W4A4 量化下提升约 7% 准确率，在 W4A4 QAT 下提升约 3.9%；在下游任务（COCO 检测/分割、VLM 评测）也表现出一致的性能提升；同时保持全精度性能不下降。

**⚠️ 局限性**

局限性包括：① 需要额外的 SVD 计算，虽已通过分段和并行化降低开销；② 目前仅在基于 Transformer 的模型与 AdamW 预训练模型上验证，未探讨在其他架构或优化器下的泛化；③ 对超大规模预训练模型的作用尚未完全评估；④ 对主奇异值的正则化依赖于适当的阈值和超参数，可能需要进一步调优。

---

## 374. TWISTED-RL: Hierarchical Skilled Agents for Knot-Tying without Human Demonstrations

**arXiv ID:** 2602.14526 | [PDF](https://arxiv.org/pdf/2602.14526v1)

**作者:** Guy Freund `[一作]` (Reichman University), Erez Karpas `[通讯]` (Technion)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出 TWISTED‑RL 框架，利用多步强化学习取代原有单步逆模型，实现无演示的机器人打结；

**💡 创新点**

将高层拓扑动作抽象为可重用的技能，按动作类型或交叉数划分子任务，并用多步 RL 让低层策略能够进行精细的几何调整；

**🔧 技术方法**

使用 Soft Actor‑Critic 强化学习、动作条件技能、P‑Data 拓扑表示、Reidemeister 移动、MuJoCo 物理仿真以及多步 episode 训练；

**📊 数据集**

完全自监督采集，仅在仿真中收集约 2 M 环境步数据；不使用 TWISTED 原始数据或任何人工演示；

**📈 对比分析**

与 TWISTED 基线在 5 个不同难度的测试集（3‑Easy, 3‑Medium, 3‑Hard, 3‑Eval, 4‑Eval）下对比，测量成功率和运行时间；TWISTED‑RL‑C 在大多数集上显著提升成功率（如 3‑Medium 44% vs 28%，4‑Eval 26% vs 10%），并使任务完成时间缩短约 3 倍；

**⚠️ 局限性**

存在 sim2real 差距；中等/难复杂度仍未达 100% 成功率；多步策略导致执行时间略长；需要更强的探索和更具表达力的策略以进一步提升性能。

---

## 375. Rewriting Induction for Existentially Quantified Equations in Logically Constrained Rewriting (Full Version)

**arXiv ID:** 2602.14636 | [PDF](https://arxiv.org/pdf/2602.14636v1)

**作者:** Naoki Nishida `[一作]` (Nagoya University), Misaki Kojima `[通讯]` (Nagoya University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出在逻辑约束重写系统（LCTRS）中引入存在量化的约束方程，并扩展了重写归纳（RI）框架以证明这些方程及其相关不等式的归纳定理。

**💡 创新点**

创新点在于首次将存在量化引入约束方程，从而让RI框架能够处理不等式和更复杂的归纳问题；同时定义了新的约束∃-项与∃-方程的归约规则。

**🔧 技术方法**

核心技术包括逻辑约束重写系统、重写归纳（RI）框架、存在量化约束的引入以及对归约规则的扩展。

**📊 数据集**

文中未使用任何公开数据集，实验仅基于示例 LCTRS 进行演示。

**📈 对比分析**

未给出与其他方法的系统性比较或性能评估；仅通过手工演示证明了示例问题的可归纳性。

**⚠️ 局限性**

局限性包括：只实现了 RI 的三条主要推理规则，缺乏完整的终止性检查与自动化；原型未公开，且缺乏对更大规模实例的实验验证。

---

## 376. Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions

**arXiv ID:** 2602.14279 | [PDF](https://arxiv.org/pdf/2602.14279v1)

**作者:** Ruomeng Ding `[一作]`, Zhun Deng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自适应群体信息挖掘框架，在有限的询问和响应预算下，联合选择问题和受访者，并利用异质图神经网络对缺失响应进行推断，从而更准确地估计群体潜在属性。

**💡 创新点**

将大型语言模型的期望信息增益（EIG）目标与异质图神经网络的群体结构传播相结合，实现了在受限预算下同时优化问题和受访者的选择；并给出了贪婪算法在此框架下的近似最优性理论保证。

**🔧 技术方法**

采用大型语言模型（LLM，Meta‑Llama‑3.1/3.2）、异质图神经网络（R‑GCN）、期望信息增益（EIG）评价准则、de Finetti 预测推断框架以及子模性质的贪婪选择策略。

**📊 数据集**

使用美国全国选民调查（CES）、OpinionQA 公开意见数据集以及 Twin‑2k 经济偏好与认知偏差数据集进行实验。

**📈 对比分析**

与三种基线（Meta‑Random、Meta‑Greedy、Meta‑Greedy‑Imp）对比，在10%、30%、50%响应预算下，所提方法在 CES 上提升 12–17% 的准确率，OpinionQA 和 Twin‑2k 同样表现出显著优势；多步规划相比贪婪仅带来微小收益，且计算成本更高。

**⚠️ 局限性**

局限性包括：依赖可建模的群体结构，对极少响应或极端异质性群体的适应性有限；多步规划虽理论上可提升但实际收益不明显；LLM 的预测对训练分布敏感，跨域迁移效果仍需进一步验证。

---

## 377. Pailitao-VL: Unified Embedding and Reranker for Real-Time Multi-Modal Industrial Search

**arXiv ID:** 2602.13704 | [PDF](https://arxiv.org/pdf/2602.13704v1)

**作者:** Lei Chen `[一作]` (Alibaba Group), Bo Zheng `[通讯]` (Alibaba Group)

**通讯引用:** 12609 | [OpenAlex ID](https://openalex.org/A5034845046)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Pailitao‑VL——一种面向工业场景的统一多模态检索系统，包含高精度嵌入和基于生成式语言模型的重排序两大模块；

**💡 创新点**

创新点在于（1）将嵌入学习从传统对比学习迁移到绝对 ID 识别，利用亿级语义原型作为全局锚点；（2）构建 Agent‑Driven “Plan‑Propose‑Organize‑Review” 数据治理流程，实现高纯度实例聚类；（3）将重排序从单点式改为“compare‑and‑calibrate”列表式，结合块级相对排序与全局绝对相关性评分，并采用混合排序策略；

**🔧 技术方法**

采用 MLLM2vec 3B 背景、additive angular margin、Multi‑Granularity Representation Learning、混合并行训练、Chunkwise 位置感知模型、MLP 绝对评分头等技术；

**📊 数据集**

使用 1.5M 电子商务文档和 5K 查询的离线检索集，2K 查询 + 50 文档的重排序集，以及阿里巴巴平台实时业务日志；

**📈 对比分析**

在离线评测中，Pailitao‑VL-Embedding 在 I‑HR@1 取得 64.52，显著高于 TBStars（60.40）和 CLIP‑IDClass（60.05）；重排序模块在 I‑HR@1 达到 57.92，超过 Qwen3‑VL‑Reranker（50.10）和 CLIP‑IDClass（54.48）。在线 A/B 测试中，Embedding + Reranker‑List 分别实现 2% 与 6% GMV 提升，部分 AI 搜索场景提升高达 20%；

**⚠️ 局限性**

局限性包括：① 需要海量原型库与复杂的数据治理 pipeline，增加了前期准备成本；② 对极端噪声或稀有类别的鲁棒性仍有待验证；③ 训练与部署仍依赖大规模 GPU 集群，算力和能耗需求高。

---

## 378. Simultaneous State Estimation and Online Model Learning in a Soft Robotic System

**arXiv ID:** 2602.14092 | [PDF](https://arxiv.org/pdf/2602.14092v1)

**作者:** Jan-Hendrik Ewering `[一作]` (Leibniz University Hannover), Thomas Seel `[通讯]` (Leibniz University Hannover)

**通讯引用:** 3569 | [OpenAlex ID](https://openalex.org/A5039578386)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在软体机器人上同时实现姿态（位置与速度）估计与弯曲刚度模型的在线学习。

**💡 创新点**

创新地将灰盒系统识别与边缘化粒子滤波相结合，利用可分离的高斯过程在常数曲率模型中学习状态相关的弯曲刚度，从而在保持物理可解释性的同时实现在线联合状态估计与模型学习。

**🔧 技术方法**

采用边缘化粒子滤波、可降阶高斯过程回归、常数曲率软体机器人动力学模型、随机游走超参数学习及基于基函数的GP表示。

**📊 数据集**

使用真实软体机器人实验数据（5 秒、8 ms采样率），输入为气压控制信号，输出为基底反作用力/扭矩，并通过光学追踪提供状态真值。

**📈 对比分析**

与无模型学习的UKF基准进行对比，评估位置、速度RMSE、NMSE；边缘化PF（含超参数学习）与UKF性能相当，在多步预测中相较于纯常数曲率模型误差下降约10%。

**⚠️ 局限性**

局限性在于实现尚未实时化、计算量较大；需要人工调参；仅在短时间序列上验证，长序列需考虑衰减因子或更稳健的模型。

---

## 379. OpAgent: Operator Agent for Web Navigation

**arXiv ID:** 2602.13559 | [PDF](https://arxiv.org/pdf/2602.13559v1)

**作者:** Yuyu Guo `[一作]` (Ant Group), Peng Di `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 OpAgent，一种基于视觉语言模型的在线强化学习网络爬虫，能够在真实网页环境中自主执行长序列任务。

**💡 创新点**

创新点包括：① 层次化多任务监督微调（规划、执行、定位）实现跨任务的基线；② 结合 WebJudge 与规则树的混合奖励，在野外环境中实现实时、无标签的强化学习；③ 通过 Planner‑Grounder‑Reflector‑Summarizer 四层模块实现错误检测与自我纠错。

**🔧 技术方法**

核心技术：Vision‑Language Model（Qwen2.5‑VL‑72B、Qwen3‑VL‑32B），Playwright 浏览器集群，Hybrid Reward（WebJudge + RDT），Group Relative Policy Optimization (GRPO) 与 KL‑Cov 策略，文本与视觉双模观察。

**📊 数据集**

使用的数据集包括：WebDreamer（规划）、Mind2Web & Aguvis（执行）、UGround（定位）、WebArena、Wild Website 87 条测试站点以及多样化的 Web 查询合成数据。

**📈 对比分析**

与现有基线（如 Qwen2.5‑VL‑72B、RL‑Zero 等）比较，RL‑HybridReward 模型在 WebArena 上从 38.1% 提升至 71.6% pass@5，单模型在 Wild 网站上平均分从 3.09 提升至 3.56，显示显著性能提升。

**⚠️ 局限性**

局限性：高度依赖手工提示与多模态代理协作，导致人力与计算成本高；对跨域泛化仍需进一步提升；目前仍需要离线数据与在线经验相结合，探索能力尚不足。

---

## 380. Inferring Turn-Rate-Limited Engagement Zones with Sacrificial Agents for Safe Trajectory Planning

**arXiv ID:** 2602.13457 | [PDF](https://arxiv.org/pdf/2602.13457v1)

**作者:** Grant Stagg `[一作]` (Brigham Young University), Cameron K. Peterson `[通讯]` (Brigham Young University)

**通讯引用:** 403 | [OpenAlex ID](https://openalex.org/A5015155264)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于牺牲代理的学习框架，通过分析代理的截击与生存二元结果来估计追踪者的运动参数，并利用估计结果规划安全、时间最优的逃逸路径。

**💡 创新点**

创新点在于将可达区域（Reachable Region）与牺牲代理的二元反馈直接关联，设计了两种可微损失函数（边界截击与内部截击），并结合多起点优化与贝叶斯实验设计实现参数的高效辨识；同时提出了基于信息增益的轨迹选取方法，使牺牲代理能主动探测最大不确定性。

**🔧 技术方法**

技术主要包括：基于Dubins曲线的可达区域几何建模、可微平方铰链损失、使用JAX自动微分与IPOPT求解非线性优化、拉丁超立方抽样的多起点策略、以及基于Gauss‑Newton信息矩阵的D‑optimal 轨迹选取。

**📊 数据集**

使用大量随机生成的追踪者参数（500或50次 Monte Carlo 试验）构造的合成数据集，包含不同噪声水平、不同已知/未知参数组合的多种学习案例。

**📈 对比分析**

与基准（无牺牲代理或全知参数）比较，实验表明在噪声自由或有限噪声下，仅需 4~7 个牺牲代理即可收敛至参数误差低于 1% 的水平；学习得到的可达区域几乎完全包含真值（>95% 覆盖率），规划的安全路径长度与最优路径相差不足 5%，并显著优于无学习时的保守路径。

**⚠️ 局限性**

主要局限包括：追踪者静止发射点假设、仅考虑确定性运动学、忽略感知不确定性与动态追踪者适应行为、以及对内部截击的截击位置分布做简化假设；这些限制限制了方法在真实战场环境中的直接适用。

---

## 381. BHyGNN+: Unsupervised Representation Learning for Heterophilic Hypergraphs

**arXiv ID:** 2602.14919 | [PDF](https://arxiv.org/pdf/2602.14919v1)

**作者:** Tianyi Ma `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于自监督的超图神经网络框架 BHyGNN+，通过超图与其对偶图的对比学习实现无标签下的高质量节点表示。

**💡 创新点**

创新点在于：①首次利用超图对偶性将原始超图与对偶图视为正样本对，消除了负样本的需求；②将BHyGNN的监督传播机制改为自监督对比，克服了标签稀缺的限制；③采用多种超图增强策略提高鲁棒性。

**🔧 技术方法**

技术手段包括：超图对偶变换、基于BHyGNN的自监督对比学习、余弦相似度对比损失、Gumbel-Softmax变分传播动作学习，以及多种超图增强（节点属性遮蔽、超边扰动、超边/节点丢弃）。

**📊 数据集**

实验使用了十一组基准超图数据集，覆盖异质性（Senate、Congress、House、Walmart、Synthetic）与同质性（Cora、Pubmed、Citeseer、DBLP、Cora-CA、Twitter）两类。

**📈 对比分析**

与多种监督与自监督HyGNN基线（如HGNN、HyperGCN、ED-HNN、HyperGCL、TriCL、HyGCL-ADT、HypeBoy）在节点分类任务上进行比较，BHyGNN+ 在异质与同质超图上均实现了最高或接近最高的准确率，尤其在标签稀缺或无标签设置下显著优于所有基线。

**⚠️ 局限性**

局限性包括：仅针对节点级任务；对偶视角的增广策略仍可进一步优化；对超图对偶性理论解释尚不完整，需进一步研究。

---

## 382. The Distortion of Stable Matching

**arXiv ID:** 2602.14961 | [PDF](https://arxiv.org/pdf/2602.14961v1)

**作者:** Aris Filos-Ratsikas `[一作]` (University of Edinburgh), Georgios Kalantzis `[通讯]` (University of Edinburgh)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究在有限偏好信息下稳定匹配问题的失真（distortion）问题，设计了随机化与查询增强的算法，并给出了对稳定匹配质量的上界与下界；

**💡 创新点**

主要创新点在于证明任何确定性序数（ordinal）算法的失真无界、随机化版DA算法能达到最优失真2、查询增强算法需至少Ω(log n)查询才能超过2，并给出O(log n/ε²)查询实现1+ε失真、在旋转偏序路径结构下进一步改进为O(log n/ε)；同时对平均情况给出失真上限2并实验证明实际接近1；

**🔧 技术方法**

采用稳定匹配理论、旋转图、二分搜索与查询模型、失真分析、构造实例证明下界、概率论分析以及实验验证；

**📊 数据集**

实验使用随机生成的偏好配置，涵盖属性模型、无偏好模型（IC）、IC2、Mallows模型，且取值分布为均匀、Beta(1/2,1/2)、指数、尖峰均匀等；

**📈 对比分析**

通过与最优稳定匹配的期望福利比值（失真）比较，随机化与1查询版本也做对比；实验结果显示平均失真在1.0–1.05范围内，远低于理论上限2；

**⚠️ 局限性**

局限性包括：最坏情况下确定性序数算法失真无界；查询增强的O(log n/ε²)上界可能不最优；对噪声查询不具鲁棒性；结果仅针对二分稳定匹配，难以直接推广至其他稳定性概念或更复杂匹配模型。

---

## 383. Constructive Patterns for Human-Centered Tech Hiring

**arXiv ID:** 2602.13845 | [PDF](https://arxiv.org/pdf/2602.13845v1)

**作者:** Allysson Allex Araújo `[一作]` (Federal University of Cariri), Marcos Kalinowski `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对22名初级软件工程师进行深度访谈，收集了470个招聘与选拔（R&S）经历，运用主题分析方法识别并构建了22条正向构造模式（Constructive Patterns, CPs），为人性化技术招聘提供实证基础。

**💡 创新点**

创新点在于：① 把关注点从传统的反模式（Anti‑patterns, APs）诊断转向可操作的构造模式；② 将AART（Applicant Attribution‑Reaction Theory）引入招聘研究，解释候选人正向归因机制；③ 输出可供企业直接参考的22条CP清单，弥补了以往缺乏候选人视角设计方案的空白。

**🔧 技术方法**

主要技术与方法：① 半结构化访谈（以AART为理论框架设计访谈提纲）；② Braun & Clarke的六阶段主题分析（编码、主题归纳、验证）；③ 质性数据管理与审计（四位研究者共同校核、公开代码簿）。

**📊 数据集**

数据集：来自22位早期职业软件工程师的访谈文本，涵盖了超过470个R&S过程的具体经历，数据以文字记录形式提供，未使用公开量化数据集。

**📈 对比分析**

对比与评估方式：本研究不进行量化性能评估，也未与已有方法进行实验比较；主要通过主题一致性检验、跨案例验证和理论解释来确认CPs的可靠性与可操作性。

**⚠️ 局限性**

局限性：① 样本局限于巴西地区、早期职业阶段，性别与社会经济多样性不足；② 访谈依赖回忆，可能出现记忆偏差与社会期望偏差；③ 结果缺乏在真实招聘流程中的实证验证，无法量化CPs对招聘效率、候选人满意度或多样性提升的影响；④ 对不同文化、行业或高级职位的适用性需进一步检验。

---

## 384. X-Blocks: Linguistic Building Blocks of Natural Language Explanations for Automated Vehicles

**arXiv ID:** 2602.13248 | [PDF](https://arxiv.org/pdf/2602.13248v1)

**作者:** Ashkan Y. Zadeh `[一作]` (Queensland University of Technology), Zishuo Zhu `[通讯]` (Queensland University of Technology)

**通讯引用:** 33 | [OpenAlex ID](https://openalex.org/A5034484343)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了X-Blocks框架，对自动驾驶系统的自然语言解释进行分层分析（情境、句法、词汇），并基于CoT+自一致性多模型大语言模型实现对解释的情境分类、词汇关键性分析和句法模板提取；

**💡 创新点**

创新点在于：①将情境、句法、词汇三层统一纳入可迁移的分析框架；②采用CoT+Self Consistency多模型推理实现高精度情境标注；③系统提取情境特定的词汇和句法“构建块”，为可解释AV设计提供实证依据；

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT‑3.5、GPT‑4等）结合Chain of Thought和Self Consistency；依存句法分析（Universal Dependencies）；词汇关键性分析（对数赔率+Dirichlet先验）；句法签名与基于槽位的模板抽取；多模型投票与条件决策；

**📊 数据集**

使用数据集：Berkeley Deep Drive‑X（BDD‑X）中约16,392条人类编写的驾驶解释文本；

**📈 对比分析**

评估方法：与两名人工标注者及其一致子集进行对比，计算准确率、宏F1、Cohen’s κ和Fleiss κ；RACE在情境分类中达到91.45%准确率、Cohen’s κ 0.91，显著优于单模型CoT基线；

**⚠️ 局限性**

局限性：仅在英文BDD‑X数据上实验，缺乏多语言验证；未结合视频/传感器多模态上下文或解释时序信息；对解释单一文本的处理可能忽略驾驶情境的动态变化。

---

## 385. AbracADDbra: Touch-Guided Object Addition by Decoupling Placement and Editing Subtasks

**arXiv ID:** 2602.14237 | [PDF](https://arxiv.org/pdf/2602.14237v1)

**作者:** Kunal Swami `[一作]` (Samsung Research India), Alok Shukla `[通讯]` (Samsung Research India)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AbracADDbra 框架，利用用户触摸提示与简洁指令实现高质量对象添加，并将位置预测与编辑过程解耦。

**💡 创新点**

创新点在于引入触摸导向的交互式对象添加范式，构建分离的定位与编辑模块，并通过生成定位推理提升准确性，同时创建 Touch2Add 基准数据集。

**🔧 技术方法**

采用视觉语言 Transformer 进行位置预测；使用 Stable Diffusion 及自定义 UNet 生成对象及实例掩码；基于 GPT-2 的语言模型生成推理文本与坐标。

**📊 数据集**

使用自动生成的 100k 样本 COCO 训练集（含 90k 训练、10k 验证）和新构建的 544 例 Touch2Add 验证集；评测 MagicBrush 150 例子集。

**📈 对比分析**

与随机放置、文本仅提示、现有 VLM 基线（LLaVA、ViP-LLaVA）以及图像编辑方法（IP2P、MagicBrush 等）进行量化和用户研究；平均 IoU 提升 52%，CLIP/DINO 指标最高，用户排名显著优于基线。

**⚠️ 局限性**

局限性包括对触摸点准确性的高度依赖；对象尺度预测仍存在误差；未覆盖更广泛的编辑任务，需进一步泛化。

---

## 386. Why Code, Why Now: Learnability, Computability, and the Real Limits of Machine Learning

**arXiv ID:** 2602.13934 | [PDF](https://arxiv.org/pdf/2602.13934v1)

**作者:** Zhimin Zhao `[一作]` (Queen's University), Zhimin Zhao `[通讯]` (Queen's University)

**通讯引用:** 871 | [OpenAlex ID](https://openalex.org/A5034367719)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于信息结构的五层可学习性层级，并用它来解释为什么代码生成在大规模模型训练中能实现可预测的提升，而强化学习则无法收敛；同时通过正式定义可表达性、可计算性与可学习性三大属性，阐明它们之间的关系；

**💡 创新点**

创新点在于将可学习性与可表达性、可计算性分层，并给出统一的风险函数模板；引入量化的量词深度与对抗性等级，首次系统地把学习难度映射到信息结构的层级；通过该框架解释了代码生成的成功与RL的瓶颈，提出任务可学习性是限制模型提升的根本因素；

**🔧 技术方法**

主要技术为形式化建模与理论分析：使用PAC学习、语言识别、可计算性理论、量词深度分析与风险函数统一模板，对代码问题的语法、语义与错误诊断进行数学化表述；

**📊 数据集**

文中未给出具体实验数据集，而是以代码生成任务的通用数据集（如GitHub公开源码、OpenAI的代码库）为隐含背景，主要采用理论推导；

**📈 对比分析**

与传统的基于可学习性层级的对比分析类似，作者通过对代码生成（监督学习）与强化学习在信息结构上的差异进行理论比较，指出监督学习在Level 3/4的稠密可验证反馈使得规模化收敛，而RL在Level 1/2的稀疏/非确定性反馈导致无法稳定收敛；

**⚠️ 局限性**

局限性包括：缺乏实证实验验证；层级模型对实际任务的细节可能过于抽象；未讨论如何在实际工程中将层级转化为可操作的设计准则；以及对非编程类任务的可学习性评估尚不完整；

---

## 387. Assessing Cybersecurity Risks and Traffic Impact in Connected Autonomous Vehicles

**arXiv ID:** 2602.13898 | [PDF](https://arxiv.org/pdf/2602.13898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 388. ABI: A tightly integrated, unified, sparsity-aware, reconfigurable, compute near-register file/cache GPU architecture with light-weight softmax for deep learning, linear algebra, and Ising compute

**arXiv ID:** 2602.14262 | [PDF](https://arxiv.org/pdf/2602.14262v1)

**作者:** Siddhartha Raman Sundara Raman `[一作]` (University of Texas at Austin), Jaydeep P. Kulkarni `[通讯]` (University of Texas at Austin)

**通讯引用:** 4130 | [OpenAlex ID](https://openalex.org/A5003048953)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

提出了一种紧密集成、统一、稀疏感知、可重配置的近寄存器文件/缓存GPU架构ABI，并实现轻量级softmax和稀疏感知功能。

**💡 创新点**

创新点在于将近寄存器文件与缓存级别的可重配置计算单元嵌入GPU内核，支持按问题规模动态配置内存层级、分辨率与运算模式，并通过轻量级softmax和稀疏感知电路显著降低能耗。

**🔧 技术方法**

采用了可重配置计算引擎RCE、近寄存器文件逻辑NRF、近L1/L2逻辑、可编程寄存器、轻量级softmax(LWSM)和自适应稀疏检测电路等技术；整体基于TSMC65nm工艺实现。

**📊 数据集**

在CNN、GCN、LP、Ising、LLM等多种标准工作负载上进行评估，利用公开的基准程序和自定义CUDA核实现。

**📈 对比分析**

通过与基线MIAOW GPU、加入ABI的GPU以及专用加速器的对比实验，ABI在CNN、GCN、LP、Ising、LLM等任务上实现了3-6倍速度提升、6-16倍加速（与基线相比）和5-13倍能效提升，能效高达约370 GOPS/W。

**⚠️ 局限性**

局限性包括：在更大规模SoC或更先进工艺节点上的可扩展性未验证；主要针对整数精度（INT16以下）优化，浮点或更高精度支持有限；并且近寄存器文件实现仍需占用一定面积，可能影响GPU整体面积与功耗。

---

## 389. Robust Mean-Field Games with Risk Aversion and Bounded Rationality

**arXiv ID:** 2602.13353 | [PDF](https://arxiv.org/pdf/2602.13353v1)

**作者:** Bhavini Jeloka `[一作]` (Georgia Institute of Technology), Panagiotis Tsiotras `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 12362 | [OpenAlex ID](https://openalex.org/A5077667229)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种新的均衡概念——均衡风险厌恶量化响应均衡（MF‑RQE），用于处理大规模多智能体决策中对初始分布不确定性和有限理性的不确定性；

**💡 创新点**

创新点在于将风险厌恶（针对初始分布）与有限理性（通过凸正则化实现的量化响应）相结合，构建了一个更具鲁棒性且可求解的均衡框架；

**🔧 技术方法**

主要技术包括凸风险度量的对偶表示、均衡的固定点迭代与拟像游戏（Fictitious Play）算法，以及基于深度强化学习的模型无关实现；

**📊 数据集**

实验使用了SIS疫情传播模型和一维拥堵游戏，分别构造了不同的初始分布集合和概率；

**📈 对比分析**

与传统单分布下的熵正则化纳什均衡以及期望最优政策进行比较，MF‑RQE在鲁棒性（可达零可利用性）方面表现优异，且在平均收益上仅略低；

**⚠️ 局限性**

局限性包括：对初始分布的风险厌恶可能导致收益下降；实验仅覆盖了有限分布情形；未考虑异质性或团队合作场景。

---

## 390. PDE foundation models are skillful AI weather emulators for the Martian atmosphere

**arXiv ID:** 2602.15004 | [PDF](https://arxiv.org/pdf/2602.15004v1)

**作者:** Johannes Schmude `[一作]`, Juan Bernabe-Moreno `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将预训练的Poseidon PDE基础模型扩展到三维，并在火星大气预测任务中进行微调。

**💡 创新点**

创新点在于将二维PDE预训练模型迁移到三维大气场，同时在稀疏初始条件下验证其数据与计算效率。

**🔧 技术方法**

使用的技术包括Poseidon的scOT架构、Swin Transformer、ConvNeXt、垂直轴注意力、稀疏掩码、归一化L1损失。

**📊 数据集**

使用OpenMARS数据库的火星大气重分析（四年数据，128×128分辨率）。

**📈 对比分析**

与随机初始化的scOT和持久性基线对比，预训练+三维扩展在保留一年验证集上提升34.4%误差，稀疏数据下提升约40%。

**⚠️ 局限性**

局限性包括缺乏边界条件、辐射与地形信息，无法处理球面几何，模型对最高低层误差较大，且只评估了有限的训练步数与参数规模。

---

## 391. Implementation and Performance Evaluation of CMOS-integrated Memristor-driven Flip-flop Circuits

**arXiv ID:** 2602.13825 | [PDF](https://arxiv.org/pdf/2602.13825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 392. Situation Graph Prediction: Structured Perspective Inference for User Modeling

**arXiv ID:** 2602.13319 | [PDF](https://arxiv.org/pdf/2602.13319v1)

**作者:** Jisung Shin `[一作]` (Flybits Labs), Hossein Rahnama `[通讯]` (MIT Media Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Situation Graph Prediction (SGP) 任务，通过结构优先的合成方法将多模态数字足迹映射到结构化的视角图，旨在实现对用户内部状态的推理。

**💡 创新点**

创新点在于：①把观点推理表述为逆推结构化图的任务；②采用“先生成结构再生成证据”的结构优先合成流程，解决标签稀缺的瓶颈；③构建了基于 DUL 上层本体的场景图规范，兼顾表面与潜在属性。

**🔧 技术方法**

技术手段包括多模态编码器（文本、图像、音频转文本）、LLM（GPT‑4o）进行检索增强的上下文学习、图生成与本体约束检查，以及 Soft F1、Predicate Violation Rate 等评估指标。

**📊 数据集**

使用了自构造的 75 条场景实例（共 225 条多模态合成痕迹），以虚构人物 Elise Navarro 为例，覆盖 2021‑2025 年的职业、个人、健康与社交四大领域。

**📈 对比分析**

比较方法为零射线与检索增强对比实验；在零射线下 Soft F1 约 0.145，检索增强后提升至 0.424；严格 F1 由 0.016 提升至 0.163；但潜在属性（Emotion、Valence）的 F1 仍低于表面属性，显示推理难度更高。

**⚠️ 局限性**

局限性包括：数据规模有限且为完全合成，缺乏真实噪声；仅评估单一 LLM，未验证跨模型泛化；对本体的依赖限制了跨文化与更广泛情境的适用性。

---

## 393. An Embarrassingly Simple Way to Optimize Orthogonal Matrices at Scale

**arXiv ID:** 2602.14656 | [PDF](https://arxiv.org/pdf/2602.14656v1)

**作者:** Adrián Javaloy `[一作]` (Institute for Machine Learning), Antonio Vergari `[通讯]` (Institute for Machine Learning)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种改进后的Landing算法，称为Pogo，能够在每一步逼近Stiefel流形并保持正交约束，适用于大规模深度学习模型；

**💡 创新点**

创新点在于将Landing的梯度更新拆分，采用线性基优化器并显式求解landing多项式，证明在适度的梯度范数下可用λ=1/2近似，从而在保持正交性、可扩展性和性能上大幅提升；

**🔧 技术方法**

技术包括Stiefel流形上的Riemannian梯度、梯度投影、线性优化器（如VAdam）、四次多项式求根、GPU友好的矩阵乘法；

**📊 数据集**

使用的主要数据集包括CIFAR-10（O-ViT、ResNet-110）、MNIST（平方单元PC）以及PCA和Procrustes基准；

**📈 对比分析**

与传统RGD、RSDM、Landing、LandingPC、SLPG等方法比较，Pogo在保持正交约束、训练速度和下游性能上均优于或等同于最先进方法，尤其在数千个正交矩阵的任务中训练时间仅为前者的十几分之一；

**⚠️ 局限性**

局限性包括对梯度范数的假设（ξ<1）以及在极端数值精度或特殊流形结构下可能需要更复杂的λ求解，且目前尚未广泛验证在非Stiefel或复杂域的通用性。

---

## 394. AnomaMind: Agentic Time Series Anomaly Detection with Tool-Augmented Reasoning

**arXiv ID:** 2602.13807 | [PDF](https://arxiv.org/pdf/2602.13807v1)

**作者:** Xiaoyu Tao `[一作]` (University of Science and Technology of China), Tian Gao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 2042 | [OpenAlex ID](https://openalex.org/A5101751664)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 AnomaMind，一种基于代理的时间序列异常检测框架，将异常检测重新定义为多轮决策过程，并采用粗细层次的工作流与工具增强的推理机制；

**💡 创新点**

核心创新在于：①将 TSAD 视为顺序决策而非单次判别；②构建可复用工具箱实现自适应特征准备；③采用混合推理（LLM 负责工具调用与自省，RL 负责异常决策）实现高效且可解释的检测；

**🔧 技术方法**

使用大语言模型（Qwen3‑8B / Grok‑4）与视觉‑语言模型实现区间定位，深度学习特征提取工具，知识记忆工具，强化学习优化异常决策，工作流自省评估器，多轮工具交互；

**📊 数据集**

在四大公开 TSAD 基准上进行评估：Yahoo、WSD、IOPS、TODS；

**📈 对比分析**

与统计、深度学习、基础模型及 LLM 基线相比，AnomaMind 在所有数据集上均保持或提升 F1/Best‑F1，尤其在复杂异常场景下显著优于传统单次判别方法；

**⚠️ 局限性**

局限性包括：①对大模型和工具集的高算力与维护成本；②混合 RL 训练复杂度较高；③在极端实时或资源受限的应用场景下性能与可扩展性待验证。

---

## 395. Impostor Phenomenon as Human Debt: A Challenge to the Future of Software Engineering

**arXiv ID:** 2602.13767 | [PDF](https://arxiv.org/pdf/2602.13767v1)

**作者:** Paloma Guenes `[一作]` (Pontifical Catholic University of Rio de Janeiro), Alexander Serebrenik `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 9449 | [OpenAlex ID](https://openalex.org/A5054753279)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

研究软件工程研究社区中冒名顶替现象（IP），将其视为系统性的人类债务，并提出文化重构与社会结构调整方案来缓解该问题。

**💡 创新点**

创新点在于将IP与技术债务相类比，提出“人类债务”概念，强调环境因素导致的心理负担，并给出多层次支持网络、透明评估、盟友机制等系统重构路线图。

**🔧 技术方法**

主要采用问卷调查与统计分析技术，使用ICSE 2026预调查和2025年IP调查的问卷数据，进行IP分数与性别、种族、地区差异的统计检验。

**📊 数据集**

数据集包括ICSE 2026预调查（N=280）和2025年软件工程研究IP问卷（N=251），来自36个国家的研究者。

**📈 对比分析**

通过对IP分数的平均值比较，发现女性和少数族裔的IP分数显著高于男性/白人，未提供算法性能指标，主要基于统计显著性和差异性发现。

**⚠️ 局限性**

局限性：样本规模有限，地区分布不均；自报问卷可能存在偏差；缺乏纵向跟踪与实验验证，且未量化改进方案的具体效果。

---

## 396. Ask the Expert: Collaborative Inference for Vision Transformers with Near-Edge Accelerators

**arXiv ID:** 2602.13334 | [PDF](https://arxiv.org/pdf/2602.13334v1)

**作者:** Hao Liu `[一作]` (King Abdullah University of Science and Technology), Suhaib A. Fahmy `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 3214 | [OpenAlex ID](https://openalex.org/A5032556461)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了边缘与近边缘协同推理框架，将轻量级Vision Transformer部署在边缘设备，利用多台中型专家ViT在近边缘加速推理，并通过边缘模型的Top‑k预测实现无开销路由。

**💡 创新点**

创新点包括：① 用边缘模型的Top‑k结果作为零成本路由器，动态选择最相关专家；② 设计渐进式专家训练策略，通过动态加权蒸馏逐步强化专家在子集上的专精能力；③ 在多设备协同场景中兼顾精度、延迟与能耗，构成Pareto最优方案。

**🔧 技术方法**

技术手段：Vision Transformer (DeiT 3H/4H/6H/基准)，蒸馏+剪枝实现轻量化；动态加权训练与Top‑k路由；在Nvidia Jetson Orin Nano（边缘）和AGX Orin（近边缘）上部署，利用PyTorch实现；性能评测包含推理延迟、能耗与准确率。

**📊 数据集**

主要使用CIFAR‑100数据集进行实验；在讨论中对比ImageNet，说明召回差距的普适性。

**📈 对比分析**

与单机边缘、单机近边缘、通用协同推理（同规模通用专家）等基线比较；实验结果表明：专家在目标子集提升4.12%准确率，总体提升2.76%；相对通用协同推理提升1.7%；在最优阈值下，延迟可降至45%以内，能耗降低46%。

**⚠️ 局限性**

局限性：① 依赖于边缘模型Top‑k召回差距，若该差距较小效果有限；② 路由阈值选择需经验调优；③ 需要额外近边缘硬件和多专家模型，部署成本较高；④ 在极小模型或缺乏近边缘设备的场景下收益不明显。

---

## 397. WiSparse: Boosting LLM Inference Efficiency with Weight-Aware Mixed Activation Sparsity

**arXiv ID:** 2602.14452 | [PDF](https://arxiv.org/pdf/2602.14452v1)

**作者:** Lei Chen `[一作]` (Tsinghua University), Wenwu Zhu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了WiSparse训练‑free激活稀疏框架，利用权重与激活信息实现LLM推理的高效压缩。

**💡 创新点**

创新点在于将权重重要度融入激活稀疏评估，并通过粗细粒度的稀疏分配策略自适应不同块与层的敏感性。

**🔧 技术方法**

核心技术包括权重感知重要度评分、演化搜索分配块级稀疏率、贪婪层级搜索细化稀疏率，以及对稀疏核的性能优化。

**📊 数据集**

使用Llama‑3.1‑8B、Mistral‑7B、Qwen2.5‑7B三大LLM；校准数据集为pile‑val、CodeAlpaca‑20k和MetaMathQA。

**📈 对比分析**

与R‑Sparse、TEAL等基线对比，WiSparse在50%稀疏时保持97%原始准确率，且在Llama‑3.1‑8B上实现21%推理速度提升。

**⚠️ 局限性**

限制包括动态稀疏掩码带来的运行时开销和批量推理效率受限，以及离线搜索的计算成本。

---

## 398. A Multi-Agent Framework for Medical AI: Leveraging Fine-Tuned GPT, LLaMA, and DeepSeek R1 for Evidence-Based and Bias-Aware Clinical Query Processing

**arXiv ID:** 2602.14158 | [PDF](https://arxiv.org/pdf/2602.14158v1)

**作者:** Naeimeh Nourmohammadi `[一作]` (Teesside University), Zia Ush Shamszaman `[通讯]` (Teesside University)

**通讯引用:** 292 | [OpenAlex ID](https://openalex.org/A5000695312)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并验证了一个多智能体医疗问答系统，该系统结合了 GPT、LLaMA 与 DeepSeek R1 三种语言模型，配合 SentencePiece 分词、Unsloth 优化、4‑bit 量化、PubMed 文献检索、蒙特卡罗 dropout 置信度评估、偏差检测与 LIME/SHAP 解释，形成端到端的循证与可解释医疗答复流程。

**💡 创新点**

创新点在于：①将多模型协同与分词优化相结合，提升医学术语的完整性与一致性；②设计可插拔的多智能体管线，实现循证推理、证据检索与答案精炼；③在推理阶段加入蒙特卡罗 dropout 与困惑度双重不确定性量化，并结合词汇与情感偏差检测，提升答案可靠性与公平性；④通过 4‑bit 量化与 Unsloth 在 DeepSeek R1 上实现高效推理。

**🔧 技术方法**

技术手段包括：Fine‑tuning GPT‑2、LLaMA、DeepSeek R1；SentencePiece (unigram) 分词；Unsloth 微调 + 4‑bit 量化；Flash Attention 与 KV 缓存；Monte‑Carlo Dropout 与 perplexity 置信度估计；Lexical 与 sentiment‑based 偏差检测；LIME/SHAP 解释；PubMed E‑Utilities API 文献检索；PyTorch 及 AMP 混合精度；多 GPU 分布式训练与梯度累积。

**📊 数据集**

使用 MedQuAD（20k+ 医学问答对，涵盖 12 个 NIH 领域）作为训练与评测数据；文献检索通过 PubMed API 实时获取最新研究；对比评估也使用 BioGPT 作为基准。

**📈 对比分析**

评测采用 ROUGE‑1/2/L 与 BLEU，深度检验以 bootstrap 置信区间进行显著性检验；DeepSeek R1 取得 ROUGE‑1 0.53、ROUGE‑2 0.22、BLEU 0.098，显著优于 LLaMA（0.18/0.12/0.0003）与 GPT（0.16/0.08/0.0002），在 87% 准确率、平均 36.5 s 的完整管线吞吐下表现稳定；与 BioGPT 零样本对比，DeepSeek R1 的性能提升超过 6 倍。

**⚠️ 局限性**

局限性包括：①对罕见疾病与新兴疗法的覆盖不足；②DeepSeek R1 对 GPU 内存要求高，易出现 “GPU memory exceeded” 警告；③偏差检测未覆盖人群属性维度（性别、种族等），缺乏系统性公平性评估；④解释性（LIME/SHAP）未与医学知识图谱结合，缺少临床语义映射；⑤依赖 PubMed API，网络或 API 限制可能导致检索失效；⑥在真实临床工作流中的验证与安全监管仍待进一步测试。

---

## 399. SpargeAttention2: Trainable Sparse Attention via Hybrid Top-k+Top-p Masking and Distillation Fine-Tuning

**arXiv ID:** 2602.13515 | [PDF](https://arxiv.org/pdf/2602.13515v1)

**作者:** Jintao Zhang `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 67333 | [OpenAlex ID](https://openalex.org/A5115666530)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了可训练的稀疏注意力方法SpargeAttention2，能够在不损失视频生成质量的前提下实现高达95%的注意力稀疏率。

**💡 创新点**

创新点在于：① 采用混合Top‑k+Top‑p掩码，兼顾均匀与偏斜分布；② 设计高效的块级稀疏注意力实现；③ 用速度蒸馏（velocity distillation）取代传统扩散损失，避免数据分布失配导致的性能下降。

**🔧 技术方法**

技术包括：块稀疏注意力实现、Top‑k/Top‑p混合掩码、速度蒸馏训练目标、CUDA+FlashAttention等高效实现。

**📊 数据集**

使用了Wan2.1公开视频扩散模型的私有训练集（约3000段5秒视频），并利用自动生成的字幕和VBench提示进行评估。

**📈 对比分析**

与VSA、VMoBA、SLA、SpargeAttention等基线对比，SpargeAttention2在1.3B/14B模型下实现95%稀疏率、16.2×注意力加速、4.7×端到端生成加速，同时在IQ、OC、AQ、VR、VQA等指标上与全注意力模型相当或更优。

**⚠️ 局限性**

局限性在于：仍需预训练模型和高质量教师模型进行蒸馏；对极端稀疏率或不同任务的通用性尚未验证；并且实现依赖于CUDA/FlashAttention等特定硬件支持。

---

## 400. Semantic-Contact Fields for Category-Level Generalizable Tactile Tool Manipulation

**arXiv ID:** 2602.13833 | [PDF](https://arxiv.org/pdf/2602.13833v1)

**作者:** Kevin Yuchen Ma `[一作]` (National University of Singapore), Yan Wu `[通讯]` (Agency for Science, Technology and Research)

**通讯引用:** 39642 | [OpenAlex ID](https://openalex.org/A5043731569)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Semantic-Contact Fields，用视觉语义与稠密接触估计相融合，实现类别级触觉工具操作的零样本泛化。

**💡 创新点**

创新在于将视觉语义与稠密外部接触场统一为3D表示，并通过两阶段Sim-to-Real训练与SOCP力估计实现高精度物理感知。

**🔧 技术方法**

使用PointNet++融合GelSight触觉与RGB‑D的点云、SOCP力推理、扩散策略（Diffusion Policy）以及自监督的触觉编码。

**📊 数据集**

基于约300种工具的高频仿真数据（320k帧）和少量真实抓取/刮擦的伪标注数据（数百次交互）。

**📈 对比分析**

与Vision‑Only（GenDP）、Raw Tactile、Sim‑Only、Real‑Only及无力向量消融对照；在刮擦、涂鸦和剥皮任务中，SCFields将成功率从约40%提升至73%，绘图一致性从0.81提升至0.86，剥皮接触率从45%提升至80%。

**⚠️ 局限性**

依赖示范学习，无法自动发掘新工具使用策略；对真实硬件和传感器的依赖仍有限，且对非同类别工具的跨领域泛化尚未验证。

---

## 401. Universal Algorithm-Implicit Learning

**arXiv ID:** 2602.14761 | [PDF](https://arxiv.org/pdf/2602.14761v1)

**作者:** Stefano Woerner `[一作]` (University of Tübingen), Christian F. Baumgartner `[通讯]` (University of Lucerne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一的理论框架，并基于此设计了一种算法隐式的元学习方法 TAIL，用于实现跨域、跨模态、跨标签空间的少样本学习。

**💡 创新点**

创新点包括：① 用实践普适性和算法显式/隐式区分的理论框架厘清普适元学习的本质；② 通过随机投影实现通用特征编码；③ 引入随机注入的全局标签字典实现标签空间外推；④ 采用非因果 Transformer 结合线性查询并行加速，实现显著的计算效率。

**🔧 技术方法**

技术手段主要是非因果 Transformer 的序列建模；随机投影、随机标签嵌入；全局可学习标签字典；线性查询前向推断；以及预训练的特征提取器。

**📊 数据集**

训练集包含 ImageNet、Meta-Album、MedIMeta；测试集涵盖 MiniImageNet、tieredImageNet、CIFAR-FS、Pascal VOC、CUB、FGVC-Aircraft、meta-iNat、cxr/oct/pbc、Paintings、Pascal+Paintings、IMDB 文本分类等。

**📈 对比分析**

与 Linear Probe、ProtoHead、SNAIL、GPICL、CAML 等基线进行对比。在标准 few-shot 任务、跨域、跨模态、标签外推等场景中，TAIL 在大多数数据集上均达到或超过现有 SOTA，特别是在文本任务上表现优异，同时在计算时间和内存使用上显著优于 Transformer 基线。

**⚠️ 局限性**

局限性包括：尚未在回归或结构化预测任务上验证；对极大标签空间的推断仍受限；依赖大规模 meta 数据集，且在未见领域仍需谨慎验证其泛化可靠性。

---

## 402. Artificial Organisations

**arXiv ID:** 2602.13275 | [PDF](https://arxiv.org/pdf/2602.13275v1)

**作者:** William Waites `[一作]` `[通讯]` (University of Southampton), William Waites (University of Southampton)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Perseverance Composition Engine（PCE），该多代理系统由 Composer、Corroborator、Critic 三个角色组成，完成文档撰写、事实核验和论证评估，并在 474 个内部生成的写作任务中通过分层验证与迭代反馈提升产出质量。

**💡 创新点**

提出将人类组织学中的分工、对抗性审查和信息隔离原则转化为人工智能系统的架构约束，实现不依赖个体对齐而是通过机构结构来保障系统整体可靠性的创新思路。

**🔧 技术方法**

采用多代理架构与角色化信息隔离（Composer 仅能撰写、Corroborator 拥有完整来源访问、Critic 无法访问来源），利用 Claude 4.5 语言模型完成文本生成与评估，构建信息对抗性循环并实施迭代修订。

**📊 数据集**

使用 474 个内部生成的写作/策划项目（包括自我文档化任务），无公开第三方数据集；所有任务均由人类监督与系统内部日志共同完成。

**📈 对比分析**

通过比较初稿与最终稿的质量得分、迭代次数、成本与合规率进行评估；结果显示 52% 的草稿被识别为虚假，平均质量提升约 79%，平均迭代次数为 4.3，成本约为 0.29 美元/项目。

**⚠️ 局限性**

局限性：实验缺乏对照组，系统仅在受限域内运行，所有实验基于单一 Claude 4.5 模型，结果可能不具普适性；自我文档化的反射性也可能影响结论的外推。

---

## 403. Visual Para-Thinker: Divide-and-Conquer Reasoning for Visual Comprehension

**arXiv ID:** 2602.13310 | [PDF](https://arxiv.org/pdf/2602.13310v1)

**作者:** Haoran Xu `[一作]` (Zhejiang University), Jian Luan `[通讯]` (MiLMPlus, Xiaomi Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Visual Para-Thinker 框架，实现多路径并行视觉推理

**💡 创新点**

引入 Pa-Attention 与 LPRoPE 保障路径独立性与可区分性，突破传统单路径限制

**🔧 技术方法**

基于 vLLM 的并行解码，配合 Pa-Attention 与 LPRoPE 技术实现路径隔离与位置编码

**📊 数据集**

训练集由 LVIS、LAION、COCO、PixMo-Count、RefCOCO 等 163k 视觉问答对构成，涵盖多样视觉任务

**📈 对比分析**

与 Qwen、Gemini、GPT‑4o 等模型对比，在计数、细粒度感知、误生成和定位任务上提升 2‑6% 以上，同时推理吞吐量提升 2.5 倍

**⚠️ 局限性**

仅在视觉问答与定位任务验证，路径数与模型规模匹配仍需进一步探究，缺乏对更大规模或其他视觉任务的广泛评估

---

## 404. Character-aware Transformers Learn an Irregular Morphological Pattern Yet None Generalize Like Humans

**arXiv ID:** 2602.14100 | [PDF](https://arxiv.org/pdf/2602.14100v1)

**作者:** Akhilesh Kakolu Ramarao `[一作]`, Dinah Baer-Henney `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了西班牙语 L‑shape 形态学模式的学习与推广，比较了五种不同位置编码与标记表示方式的字符级编码‑解码 Transformer 在真实动词和虚构动词上的表现。

**💡 创新点**

创新点在于把位置无关的标记编码与标记的原子/结构化表示作为两个维度系统地比较，评估它们对稀有形态学模式学习的影响。

**🔧 技术方法**

采用了字符级编码‑解码 Transformer，实验了顺序位置编码（Sequential‑PE）与固定位置编码（Position‑Invariant‑PE），并对标签进行原子化、One‑Hot 与特征几何化三种表示。

**📊 数据集**

使用了 Unimorph 提供的西班牙语 12‑细胞时态范式数据，并将 107 名母语者的 wug‑test 虚构动词数据作为外部评测集。

**📈 对比分析**

通过序列准确率、词干准确率和范式形状聚类等指标进行比较，位置无关模型在稀有 L‑shape 训练条件下取得更高的准确率，但无模型能在新词上成功推广 L‑shape，模型整体表现低于人类。

**⚠️ 局限性**

局限性包括仅针对西班牙语、虚构动词的音素变异可能掩盖范式学习、未调优各模型的最佳超参数、未考察内部表示是否已编码 L‑shape，且未评估非 Transformer 方案。

---

## 405. Rethinking the Role of LLMs in Time Series Forecasting

**arXiv ID:** 2602.14744 | [PDF](https://arxiv.org/pdf/2602.14744v1)

**作者:** Xin Qiu `[一作]` (Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative, Institute of Digital Twin, Eastern Institute of Technology), Xiaoyu Shen `[通讯]` (Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative, Institute of Digital Twin, Eastern Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文系统评估并验证了大规模预训练语言模型在时间序列预测中的有效性，构建了LLM4TSF框架并在覆盖8 B观测、17个预测场景和4个时限的跨域数据集上进行大规模实验；

**💡 创新点**

创新点包括：①通过大规模交叉数据训练显著提升LLM在跨域预测中的表现；②揭示预训练知识与LLM架构互补作用；③使用token级路由分析解释LLM何时发挥作用；④对比预对齐与后对齐两种对齐策略，阐明预对齐更优；

**🔧 技术方法**

技术手段包括：LLM4TSF框架（GPT‑2骨干+TS编码器/解码器）、预对齐/后对齐对齐策略、PCA降维对齐、prompt工程、完整参数微调、LoRA/轻量化微调、token级路由机制与Gumbel‑Softmax；

**📊 数据集**

使用62个公开时间序列数据集（55个训练集，7个完全不参与训练的外域集），覆盖交通、能源、金融、气象等10个应用域；

**📈 对比分析**

与传统TS基础模型（Chronos、UniTS、Moirai）以及单域LLM模型（UniTime、TimeLLM）在零样本/少样本情景下进行MAE/MSE对比，LLM4TSF在零样本下平均MAE/MSE均显著优于对比模型，跨域泛化能力更强；

**⚠️ 局限性**

局限性：LLM效果高度依赖数据分布，在平稳或小样本场景下收益有限；需要完整模型与充分参数优化，单纯放大LLM尺寸并不能保证提升；对齐策略和提示信息对性能影响大，缺乏统一的对齐标准。

---

## 406. TEG: Exascale Cluster Governance via Non-Equilibrium Thermodynamics and Langevin Dynamics

**arXiv ID:** 2602.13789 | [PDF](https://arxiv.org/pdf/2602.13789v1)

**作者:** Zhengyan Chu `[一作]` `[通讯]` (Independent Researcher), Zhengyan Chu (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出一种基于热动力学治理的 TEG 架构，用分布式 Langevin 代理与熵舵门替代传统集中式调度，旨在解决 Exascale 级集群的 O(N) 延迟、资源恐慌和热盲区等瓶颈；

**💡 创新点**

创新点包括：①将集群视作非平衡耗散结构，构造全局全息势场实现 O(1) 调度决策；②引入双数预测、朗之万动力学和 Landau 相变控制；③通过代币蒸发和高阶控制壁垒（HOCBF）实现经济激励与热能耗散的协同治理；

**🔧 技术方法**

核心技术涵盖：分布式 Langevin 代理、双数代数、全息场投影（高斯过程回归）、Vickrey 竞价、Landau 阻尼、LMAX Disruptor 共享无锁账本、递归四叉树订单簿、热力学相变调节与 HOCBF 保障安全；

**📊 数据集**

论文主要基于理论推导与仿真实验，未使用公开真实数据集，实验采用模拟 LLM 训练工作负载和假设的热功耗模型；

**📈 对比分析**

通过与 Kubernetes 经典调度对比，展示 TEG 在调度延迟、资源利用率、热 COP 及恐慌惩罚方面分别提升约 100 倍、25% 能耗节约、30% COP 提升；

**⚠️ 局限性**

局限性在于：1）目前仅在模拟环境验证，缺乏大规模 10,000+ 节点实测；2）对高质量网络/硬件拓扑与实时遥测要求高；3）代币蒸发策略可能削弱长期经济激励；4）实现复杂度和运维成本未给出具体评估；

---

## 407. Small Reward Models via Backward Inference

**arXiv ID:** 2602.13551 | [PDF](https://arxiv.org/pdf/2602.13551v1)

**作者:** Yike Wang `[一作]` (University of Washington), Yulia Tsvetkov `[通讯]` (University of Washington)

**通讯引用:** 5158 | [OpenAlex ID](https://openalex.org/A5062910836)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无参考、无 rubrics 的奖励模型 FLIP，通过逆向推断给定回复最可能的指令并以推断指令与原指令的相似度作为奖励；

**💡 创新点**

创新点在于将奖励建模从前向评估（LLM-as-a-Judge）转变为后向推断，利用生成与验证之间的差距，显著提升小型模型的奖励质量；

**🔧 技术方法**

采用语言模型推断指令（后向生成），使用 F1 分数作为相似度度量，结合并行采样与 GRPO 进行测试时缩放和强化学习；

**📊 数据集**

在 RewardBench2、AlpacaEval、Human Interest、MATH、IFEval 等四个基准上评估，使用 13 种 8B 以下的小型语言模型（OLMo2、Llama3、Qwen3、Gemma3、Mistral-v0.3）；

**📈 对比分析**

与 LLM-as-a-Judge 的三种变体（Pointwise、Listwise、Pairwise）比较，FLIP 在 RewardBench2 的平均准确率提升约 79.6%，在并行采样和 GRPO 训练中均表现更稳健、效果更好；

**⚠️ 局限性**

局限包括：若回复包含完整或部分指令，可能导致高奖励但影响评估；F1 相似度在跨语言或极端表达差异时不适用，需使用更复杂的相似度或裁剪策略。

---

## 408. Testimole-Conversational: A 30-Billion-Word Italian Discussion Board Corpus (1996-2024) for Language Modeling and Sociolinguistic Research

**arXiv ID:** 2602.14819 | [PDF](https://arxiv.org/pdf/2602.14819v1)

**作者:** Matteo Rinaldi `[一作]`, Viviana Patti `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了 Testimole‑Conversational 语料库，该语料库收集了近30亿词的意大利讨论板（论坛）和 Usenet 对话，涵盖 1996‑2024 年的时间跨度，并提供完整的时间戳、主题和作者等元数据。

**💡 创新点**

首次系统性聚合意大利讨论板与 Usenet 的大规模对话数据，兼具时间可追溯性和主题标签，能够支持语言演化分析、社交语料研究以及 LLM 预训练；与以往仅聚焦单一论坛或 Usenet 的小规模语料相比，规模和时间覆盖面大幅提升。

**🔧 技术方法**

使用 Python3 结合 BeautifulSoup、Selenium 等爬取库对多种讨论板进行手工脚本化抓取；对文本进行 ISO‑8601 时间戳标注、BPE 子词分词（Tiktoken cl100k_base）以及结构化存储（CSV/JSON）。

**📊 数据集**

核心数据集为 Testimole‑Conversational，包含 470 M 论坛贴、90 M Usenet贴，约 30 亿词；并附带词频表、主题分布等统计文件；此外，作者提及 Testimole 还包含公开书籍、学术论文、博客等数据，供其他 NLP 任务使用。

**📈 对比分析**

论文主要侧重数据集的构建与公开，未进行模型训练或对比实验；作者指出该语料库可作为 LLM 预训练的宝贵资源，尤其对意大利语模型效果的提升具有潜在意义，并指出意大利语在 Common Crawl 等公开语料中仅占约 2%，与英语相比数据稀缺。

**⚠️ 局限性**

局限性包括：未覆盖所有意大利讨论板，数据采集受时间与手工成本限制；存在跨贴重复、误导性信息、粗俗或攻击性语言等噪声；匿名化虽减轻隐私风险，但仍可能泄露敏感信息；缺乏对语料质量的系统评估和完整的使用许可说明。

---

## 409. Optimization-Free Graph Embedding via Distributional Kernel for Community Detection

**arXiv ID:** 2602.13634 | [PDF](https://arxiv.org/pdf/2602.13634v1)

**作者:** Shuaibin Song `[一作]` (Nanjing University), Tianrun Liang `[通讯]` (Nanjing University)

**通讯引用:** 242 | [OpenAlex ID](https://openalex.org/A5110374971)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对无监督社区检测提出了一种新的图嵌入方法——多层加权分布核（mWDK），并对其在多种数据集上的效果进行了系统评估。

**💡 创新点**

主要创新点包括：①联合利用节点属性分布与度分布进行聚合；②引入分布感知的加权分布核（WDK）并多层堆叠成mWDK；③实现无优化的实现方式，显著缓解传统NAS方法的过平滑问题。

**🔧 技术方法**

技术上基于 Weisfeiler–Lehman 迭代嵌入、Isolation Kernel（IK）构建分布核、加权分布嵌入以及多层迭代的聚合策略。

**📊 数据集**

实验数据集包含四个合成网络（EEE、EEH、UE、EU）和十个真实网络（Cora、Citeseer、Pubmed、DBLP、ACM、BlogCatalog、AMAP、AMAC、Ogbn-products 等）。

**📈 对比分析**

与传统 NAS 方法（WL、WDK、NDLS、AGC、GSNN）及深度学习方法（PANE、DAEGC、MAGI 等）在 ACC、NMI、ARI 指标上进行对比，mWDK 在大多数数据集上均获得最高或相近的性能，尤其在规模巨大的 Ogbn-products 上完成所有方法中唯一无内存溢出的训练并取得最佳分数。

**⚠️ 局限性**

局限性包括：对超参数 ψ、t 的选择仍有一定敏感性；在极端噪声或高度稀疏的图结构下效果尚待进一步验证；目前仅在无监督聚类场景进行评估，未探讨有监督或异构图的适用性。

---

## 410. On the Parameterized Tractability of Packing Vertex-Disjoint A-Paths with Length Constraints

**arXiv ID:** 2602.14768 | [PDF](https://arxiv.org/pdf/2602.14768v1)

**作者:** Susobhan Bandopadhyay `[一作]` (Tata Institute of Fundamental Research), Abhishek Sahu `[通讯]` (National Institute of Science Education and Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了 A-Path Packing（在无向图中寻找 k 条长度为 ℓ 的、两端点均在给定集合 A 内且两条路径内部不与 A 相交的点集）问题的参数化复杂性，分析了不同结构参数（如距离到路径、距离到聚类、聚类顶点删除数、顶点覆盖数等）对问题可解性的影响。

**💡 创新点**

创新点包括：
- 证明在参数为（距离到路径 + |A|）时问题几乎不可能是 FPT（在 RETH 假设下给出下界）。
- 设计了对（聚类顶点删除数 + |A|）和（距离到聚类 + ℓ）的 FPT 算法，首次在这些结构参数下实现多项式时间的可解性。
- 构造了基于顶点覆盖数的二次结点 kernel，给出 O(|VC|^2) 的上界。

**🔧 技术方法**

主要技术包括：
- 随机化归约与隔离引理（Isolation Lemma）用于构造硬度证明。
- 颜色编码（Color‑Coding）与整数线性规划（ILP）实现 FPT 算法。
- q‑扩张引理与交换操作用于减少实例规模。
- 等价团（Equivalent Clique）与“剪枝”规则进一步压缩图结构。

**📊 数据集**

该工作为理论研究，未使用实验数据集，所有结果均为理论证明。

**📈 对比分析**

通过 RETH、SETH 等假设给出下界，证明问题在（距离到路径 + |A|）参数下不具备 f(k)·n^{o(√k)} 的算法；FPT 算法的运行时间可写成 f(参数)·n^{O(1)}，kernel 大小为 O(|VC|^2)。

**⚠️ 局限性**

局限性：
- 对仅使用距离到路径、距离到树宽、clique‑width 等参数的可解性尚未完全确定；
- FPT 算法的参数函数仍指数级，尚未达到最优；
- 只给出了结点 kernel，边核或更小的 kernel 仍有待研究。

---

## 411. Replanning Human-Robot Collaborative Tasks with Vision-Language Models via Semantic and Physical Dual-Correction

**arXiv ID:** 2602.14551 | [PDF](https://arxiv.org/pdf/2602.14551v1)

**作者:** Taichi Kato `[一作]` (University of Osaka), Kensuke Harada `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于Vision–Language模型的双重校正机制，实现人机协作装配任务的动态重规划。

**💡 创新点**

首次将内部逻辑校正与外部物理校正结合，形成闭环验证与修正，支持多轮交互与错误自恢复。

**🔧 技术方法**

采用OpenAI o4-mini VLM做指令理解与动作候选选择，配合GraspGen抓取规划、MoveIt轨迹规划，以及内部/外部校正模型。

**📊 数据集**

实验使用Gazebo/ROS仿真环境与真实Nextage机器人，构建铝框固定与工具准备两种装配任务场景，并自行生成候选动作数据。

**📈 对比分析**

相较于无校正模型基线，仿真中成功率提升至100%，真实环境中物体固定66.7%、工具准备初选100%、重规划75%；去除任一校正模型会导致成功率明显下降。

**⚠️ 局限性**

受限于视觉感知的光照与遮挡导致外部校正误报；仅支持位置控制，缺乏力学适应与主动协作；系统整体延迟约30秒。

---

## 412. Exploiting Structure-from-Motion for Robust Vision-Based Map Matching for Aircraft Surface Movement

**arXiv ID:** 2602.14311 | [PDF](https://arxiv.org/pdf/2602.14311v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 413. Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages

**arXiv ID:** 2602.13867 | [PDF](https://arxiv.org/pdf/2602.13867v1)

**作者:** Somnath Banerjee `[一作]` (Indian Institute of Technology Kharagpur), Animesh Mukherjee `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 11331 | [OpenAlex ID](https://openalex.org/A5033656008)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

总结并评估低资源语言与代码混合环境下LLM的安全缺口，提出资源友好的多语言安全对策。

**💡 创新点**

首次系统化分析低资源语言、文化差异和代码混合对安全的影响，并提出语言特定功能头调节、归因驱动的安全修正及社区参与的文化偏好训练。

**🔧 技术方法**

语言特定功能头微调（约3%参数）、归因驱动的 saliency drift 检测与修正、基于社区文化偏好的对齐训练、Romo/MEMIT 多语言知识编辑评估。

**📊 数据集**

XThreatBench、跨11文化的文化伤害评估集、代码混合安全测试集、低资源语言编辑验证集。

**📈 对比分析**

在XThreatBench上对10种语言的LLM进行对照实验，3%参数调节提升安全率并保持MMLU/TruthfulQA性能；归因修正使代码混合攻击成功率从69%下降至约16%；文化偏好微调将文化伤害率显著降低。

**⚠️ 局限性**

依赖有限的多文化标注数据，仍难覆盖所有文化细节；英语知识编辑仍难迁移至低资源语；部分技术对计算资源有约束，需进一步验证。

---

## 414. ORAP: Optimized Row Access Prefetching for Rowhammer-mitigated Memory

**arXiv ID:** 2602.13434 | [PDF](https://arxiv.org/pdf/2602.13434v1)

**作者:** Maccoy Merrell `[一作]` (Texas A&M University), Paul V. Gratz `[通讯]` (Texas A&M University)

**通讯引用:** 2655 | [OpenAlex ID](https://openalex.org/A5082578661)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种针对DDR5 Rowhammer 缓解系统的硬件预取器 ORAP，结合 LLCache（LLC）与 HSD 等技术，显著降低 DRAM 激活率并降低能耗。

**💡 创新点**

创新点包括：① 利用大页和 LLC 空间实施深层“Next‑Column”预取；② 通过 IP 与行置信度表动态决定预取深度；③ 引入银行级缓冲（Bank‑Leveling Buffer）和混合流检测（HSD），最大化行缓冲命中率与银行并行度；④ 在 RFM/PRAC 缓解环境下实现性能提升。

**🔧 技术方法**

使用的技术：硬件预取器设计（ORAP、HSD、Next‑Column），IPCT/RCT 置信度表，银行级缓冲，ChampSim+ramulator2 仿真框架，Adaptive Row Policy，SimPoint 方法。

**📊 数据集**

数据集：SPEC2006、SPEC2017、GAP、XSBench，采用 SimPoint 选取工作负载。

**📈 对比分析**

比较方法：在单核和八核（多核）模拟中与 Berti+SPP‑PPF、Berti+PPF 等主流预取器对比，评估多核加速、单核加速、DRAM 激活率和动态能耗。结果显示，ORAP+HSD 在 RFM 缓解下平均多核加速 4.6%，单核加速 12.8%，激活率下降约 51%，PRAC 下能耗下降 11.8%。

**⚠️ 局限性**

局限性：① 对 PRAC 缓解的系统性能略有下降；② 仍未将激活率降低到基线水平；③ 依赖大型页面和 LLC 容量，若系统资源受限可能影响效果；④ 预取有用率相较于 Berti+PPF 略低。

---

## 415. Intent-Driven Dynamic Chunking: Segmenting Documents to Reflect Predicted Information Needs

**arXiv ID:** 2602.14784 | [PDF](https://arxiv.org/pdf/2602.14784v1)

**作者:** Christos Koutsiaris `[一作]` (University of Limerick), Christos Koutsiaris `[通讯]` (SAP)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于用户意图驱动的动态分块方法 IDC，用来改进长文档的检索性能。

**💡 创新点**

创新点在于结合大型语言模型生成的预测查询与动态规划优化分块边界，使得分块与用户潜在问题对齐。

**🔧 技术方法**

方法使用 Gemini 2.5 Flash 生成意图，句子嵌入+余弦相似度评估分块相关性，并通过动态规划求全局最优分块。

**📊 数据集**

评估使用六个问答基准，涵盖新闻、维基百科、学术论文与技术文档等四个领域。

**📈 对比分析**

与固定长度、滑动窗口、文本分割与段落分割等基线相比，IDC 在五个数据集上提升 Recall@1 5%–67%，并产生 40–60% 更少的分块。

**⚠️ 局限性**

局限性包括依赖 LLM 生成的意图质量、对小样本数据集统计功效有限，以及离线预处理所需的额外计算与成本。

---

## 416. ForesightSafety Bench: A Frontier Risk Evaluation and Governance Framework towards Safe AI

**arXiv ID:** 2602.14135 | [PDF](https://arxiv.org/pdf/2602.14135v1)

**作者:** Haibo Tong `[一作]` (Beijing Institute of AI Safety and Governance), Yi Zeng `[通讯]` (Long-term AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ForesightSafety Bench AI 安全评估框架，构建了包含 94 维细粒度风险维度的多层次安全评估体系，并对 22 款主流大语言模型进行系统评估，揭示了前沿 AI 在基础、扩展和工业安全方面的漏洞与风险特征。

**💡 创新点**

创新点在于：①将安全维度层级化为基础安全、扩展安全（Embodied、AI4Science、Social、Environmental、Catastrophic/Existential）和工业安全 8 领域，实现了从微观到宏观、从普适到场景的全景覆盖；②通过自研与改造多源基准相结合，构建 94 维前瞻性风险维度；③整合十余万条结构化风险数据点，形成可持续演进的评测框架；④提出面向风险治理的多层次策略与安全防护规范。

**🔧 技术方法**

主要技术包括：①基于 LLM‑as‑Judge 的自动评估流程，统一打分标准；②利用多种 jailbreak/对抗攻击方法对模型鲁棒性进行压力测试；③在子基准中采用“seed‑expand‑audit”闭环数据生成；④对现有基准如 AdvBench、JailbreakBench、MMLU 等进行迁移与改造，提升评测细粒度；⑤系统化的评测指标如 ASR、VR、攻击成功率、风险分布分析。

**📊 数据集**

使用的数据集主要有：
- ForesightSafetyBench‑FundamentalSafety‑O（30 维）
- ForesightSafetyBench‑RiskyAgenticAI‑O（5 维）
- ForesightSafetyBench‑AI4SCI‑O（7 维）
- ForesightSafetyBench‑EmbodiedAI‑O（7 维）
- ForesightSafetyBench‑SocialAI‑O（7 维）
- ForesightSafetyBench‑EnvAI‑O（7 维）
- ForesightSafetyBench‑ExistentialRisks‑O（7 维）
- ForesightSafetyBench‑IndustrialSafety‑O（8 维）
- 现有公开基准如 AdvBench、MaliciousInstruct、JailbreakBench、MMLU、WMDP、SOSBench、Fortress‑CBRNE 等。
共计覆盖 94 个细粒度风险维度，数据量达数十万条。

**📈 对比分析**

评估方法为：对 22 款模型在 94 维安全维度上计算 ASR/VR；对比模型在无攻击与 jailbreak 攻击下的表现，形成安全排名。结果显示 Claude 系列在基础与工业安全上表现最佳，GPT‑5.2 在 Embodied、AI4Science 等扩展维度上表现突出；而 Gemini、Llama 系列在 Agentic Autonomy、Social AI、Existential 等前沿维度表现相对较弱，风险水平显著升高。整体评测展示了模型在不同维度的风险分布，为安全治理与治理框架提供量化依据。

**⚠️ 局限性**

局限性包括：①评测主要基于文本生成与仿真，缺少真实物理交互与多模态评测；②LLM‑as‑Judge 自动评分仍可能带来主观偏差；③自研数据集仍受制于人工标注与场景设计，可能存在盲区；④对抗攻击场景有限，未覆盖所有潜在对抗手段；⑤模型在工业安全场景下的具体部署效果和长期安全性尚未实测。未来需扩展多模态与跨平台评测，完善基准覆盖与动态更新机制。

---

## 417. Uncertain Pointer: Situated Feedforward Visualizations for Ambiguity-Aware AR Target Selection

**arXiv ID:** 2602.13433 | [PDF](https://arxiv.org/pdf/2602.13433v1)

**作者:** Ching-Yi Tsai `[一作]` (Princeton University), Parastoo Abtahi `[通讯]` (Princeton University)

**通讯引用:** 664 | [OpenAlex ID](https://openalex.org/A5037355698)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

研究了在增强现实中使用的多候选目标前置视觉指示（Uncertain Pointer），以帮助用户在输入歧义时进行目标选择。

**💡 创新点**

提出了系统化的指针设计空间，结合颜色标签和强度层级两种视觉化策略，并给出了基于AR情境的设计建议。

**🔧 技术方法**

采用颜色编码与强度调制的视觉注释技术，并通过两项在线实验评估其效果。

**📊 数据集**

基于30年文献综述构建25种指针设计，并在两次在线实验中收集60人和40人的交互数据，实验场景包括街景和杂货货架。

**📈 对比分析**

通过比较用户偏好、信心、心理负担、目标可见性和可识别性等指标，发现颜色编码指针提升了目标辨识度，强度层级指针有效传达系统不确定性，整体性能均优于传统单一指示。

**⚠️ 局限性**

实验规模有限，缺乏真实硬件交互验证，且仅覆盖语音与手势两种输入，未来需在更复杂场景与多模态环境中进一步评估。

---

## 418. Lifted Relational Probabilistic Inference via Implicit Learning

**arXiv ID:** 2602.14890 | [PDF](https://arxiv.org/pdf/2602.14890v1)

**作者:** Luise Ge `[一作]` (Washington University in St. Louis), Alison Shao `[通讯]` (Washington University in St. Louis)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31`

**🎯 论文内容**

提出一种在第一阶关系概率逻辑中通过隐式学习与推理的框架，能够在不构造显式模型的情况下直接利用部分背景知识与不完整观测进行推理。

**💡 创新点**

核心创新是将不完整的第一阶公理与随机抽样得到的局部观测融合到有限度的SOS（和平方）层次中，并通过双重提升（基线提升与世界提升）实现对无穷域的升维与并行推理，从而在多项式时间内完成一致性检验与概率推断。

**🔧 技术方法**

使用技术包括：有限度的和平方（SOS）程序、基于重命名等价的基线提升、并行世界提升的SDP约束、Hoeffding不等式构造置信区间、显式紧致性（explicit compactness）保证变量范围。

**📊 数据集**

论文没有给出具体实验数据集，主要以理论证明与符号实例（如药物试验）说明方法可行性。

**📈 对比分析**

方法与传统需要显式模型的MLN、PRM、WFOMC等对比，理论上实现多项式时间推理；实验上由于未实现，暂无性能数值对比，但推理复杂度仅依赖于知识库的位长和SOS次数，显著低于基于完整模型的指数级增长。

**⚠️ 局限性**

局限性包括：SOS求解器的实际可扩展性待验证；假设变量满足有界范围，限制了可建模分布；方法目前为理论证明，缺乏大规模实证评估；并行世界提升虽降低了约束数量，但在极大数据量时仍可能面临内存瓶颈。

---

## 419. A Unified Mathematical Framework for Distributed Data Fabrics: Categorical Hypergraph Models

**arXiv ID:** 2602.14708 | [PDF](https://arxiv.org/pdf/2602.14708v1)

**作者:** T. Shaska `[一作]`, I. Kotsireas `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种统一的数学框架，用超图、范畴论和模量张量范畴对分布式数据织物进行形式化建模，并对关键操作进行理论分析。

**💡 创新点**

创新点在于将超图嵌入模量张量范畴，利用对称性和几何类比（如Hurwitz空间）构建数据织物的代数表示，同时给出NP难度证明与基于谱的对称性对齐方法。

**🔧 技术方法**

技术包括超图建模、范畴论（对象为数据集，态射为变换）、模量张量范畴的 braided monoidal 结构、谱聚类与对称性对齐、以及差分隐私和联邦学习等分布式技术。

**📊 数据集**

使用了亚马逊卖家案例中的销售与库存数据集（以及示例性 IoT 传感器与金融交易数据）进行实验验证。

**📈 对比分析**

与传统经验架构对比，本文在理论上证明关键任务为NP难，提出谱与对称性方法可实现可扩展实现；在亚马逊案例中显示了高效的向量表示与容错性，但未给出具体数值性能指标。

**⚠️ 局限性**

局限性包括NP难任务仍需启发式求解、对超图稀疏性的假设、模量张量范畴映射的计算成本高、动态更新与实时性挑战，以及缺乏大规模实验验证。

---

## 420. Explainable Token-level Noise Filtering for LLM Fine-tuning Datasets

**arXiv ID:** 2602.14536 | [PDF](https://arxiv.org/pdf/2602.14536v1)

**作者:** Yuchen Yang `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种可解释的 Token‑Level 噪声过滤框架，用以在 LLM 微调过程中剔除对最终任务无益的标签 token，显著提升模型性能。

**💡 创新点**

创新点在于将 token 对微调的贡献拆解为推理重要性、知识新颖性和任务相关性三维属性，并通过可解释的评分与梯度屏蔽实现噪声过滤。

**🔧 技术方法**

技术方法包括基于注意力得分评估推理重要性、利用正确 token 预测概率衡量知识新颖性、用语义距离评估任务相关性，并结合分位数、阈值和多重 Otsu 分割进行过滤；最终对噪声 token 进行梯度屏蔽以实现微调。

**📊 数据集**

使用了多领域数据集：数学任务采用 GSM8K、NuminaMath‑CoT、MATH‑500；代码任务采用 CodeExercise 进行微调、HumanEval 进行评测；医学任务采用 PubMedQA、FIQA 等。

**📈 对比分析**

与基准清洗准确率、普通微调、双倍 epoch、数据过滤、数据增强、选择性 LM 训练、Token‑Cleaning 等方法比较，实验显示在 7 种主流 LLM（DeepSeek、Llama、Mistral）上，数学任务提升最高达 13.3%，医学任务 13.7%，代码任务 6.3%，总体性能均优于所有基线。

**⚠️ 局限性**

局限性包括：在大模型上仍需额外推理级计算开销；当前仅考虑三维属性，可能无法捕捉所有噪声；缺乏更轻量级的噪声评估方式，需要进一步探索更多属性与更高效的实现。

---

## 421. Understanding Sensor Vulnerabilities in Industrial XR Tracking

**arXiv ID:** 2602.14413 | [PDF](https://arxiv.org/pdf/2602.14413v1)

**作者:** Sourya Saha `[一作]` (City University of New York), Md. Nurul Absur `[通讯]` (City University of New York)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3855fcda-48ef-4070-a15e-803cd5c84d83` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

系统地在XR视觉-惯导里程计（VIO）中注入多种持续传感器故障（相机帧丢失、噪声放大、IMU掉线、偏置漂移等），并通过统一的评估框架分析其对轨迹精度的影响。

**💡 创新点**

首次提出针对XR VIO的系统化故障注入与评估流程，揭示了IMU故障对轨迹的灾难性影响与相机故障的相对鲁棒性，以及故障时间、持续时间与严重程度对误差的非线性作用，提供了对传感器冗余与故障容错设计的量化依据。

**🔧 技术方法**

使用ILLIXR运行时搭建XR实验平台，OpenVINS实现视觉-惯导融合，构建自定义的故障注入层；通过可配置的参数实现对相机与IMU的持续失效注入；使用ATE、RPE等轨迹误差指标进行评估。

**📊 数据集**

采用EuRoC MAV数据集中的一段序列（含同步立体相机与IMU数据，以及Vicon标定轨迹）作为真实工业环境下的基准测试数据。

**📈 对比分析**

将每一次故障注入实验的轨迹与无故障基线轨迹做全局（ATE）和局部（RPE）误差比较。结果显示：IMU噪声放大或偏置漂移可导致误差从几百米到上千米级别，几乎失效；相机帧丢失或噪声增加则误差仅在厘米级，虽随失效持续时间增长但仍保持可控，表明VIO对相机失效更为鲁棒。

**⚠️ 局限性**

仅在单一数据集、单一VIO实现和理想化的故障模型下验证，缺乏对不同环境、不同VIO架构以及更真实故障轨迹的泛化评估；未探讨动态故障检测与自适应补偿机制。

---

## 422. Coverage Guarantees for Pseudo-Calibrated Conformal Prediction under Distribution Shift

**arXiv ID:** 2602.14913 | [PDF](https://arxiv.org/pdf/2602.14913v1)

**作者:** Farbod Siahkali `[一作]` (Purdue University), Vijay Gupta `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究在无目标标签情况下，分布偏移对合规预测（Conformal Prediction）的影响，并提出了利用伪校准（pseudo‑calibration）与源域信息来保持覆盖率的方法。

**💡 创新点**

创新点在于将领域适应（Domain Adaptation）理论与合规预测结合，推导出覆盖率下界与源域分类器损失、Wasserstein 距离及梯度 Lipschitz 常数的关系，并提出源域调节伪校准（source‑tuned pseudo‑calibration）算法，通过不确定度自适应地混合硬伪标签与随机标签，降低传统伪校准的保守性。

**🔧 技术方法**

主要技术包括：合规预测的分裂式阈值设定、Wasserstein 距离与 Lipschitz 性质的利用、假设条件下的覆盖率下界推导、基于置信熵的不确定度度量与随机标签插值、以及与目标域梯度损失相关的阈值调节。

**📊 数据集**

实验使用了 MNIST、CIFAR‑10 和 CIFAR‑100 三个公开图像数据集，并通过添加噪声与变换产生不同程度的目标域偏移。

**📈 对比分析**

与传统源域校准、硬伪校准以及目标域（oracle）校准对比，源域调节伪校准在中等偏移下能显著提升覆盖率，接近 oracle 结果，同时保持可接受的期望集合大小；在大偏移下虽然集合大小增大，但仍显著优于硬伪校准。

**⚠️ 局限性**

局限性包括：理论下界保守，难以完全预测实际覆盖率；对源域损失估计与梯度 Lipschitz 常数的依赖；在极端偏移或标签分布变化（P_Y≠Q_Y）时需要额外的修正；以及算法对不确定度度量的选择敏感。

---

## 423. HyFunc: Accelerating LLM-based Function Calls for Agentic AI through Hybrid-Model Cascade and Dynamic Templating

**arXiv ID:** 2602.13665 | [PDF](https://arxiv.org/pdf/2602.13665v1)

**作者:** Weibin Liao `[一作]` (Peking University), Haoyi Xiong `[通讯]` (Microsoft Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种融合大模型与小模型的分层框架，利用大模型生成“soft token”来指引函数检索与小模型完成最终调用，以加速 LLM 基于函数调用的推理。

**💡 创新点**

创新点包括：①通过 soft token 进行用户意图压缩，显著减少上下文冗余；②使用轻量 MLP 检索器快速定位相关函数；③在小模型中引入动态模板（Dynamic Templating）仅生成参数值，避免冗余语法生成。

**🔧 技术方法**

技术手段主要是：混合模型级联、soft token 持续提示、对抗式检索器训练、动态模板注入、Selective Token Tuning（仅对参数值进行微调）。

**📊 数据集**

使用公开的 xLAM‑function‑calling‑60k 进行离线准备，并在未见的 Berkeley Function Call Leaderboard (BFCL) 上进行在线评测。

**📈 对比分析**

与多种基线（GPT‑4 系列、Qwen、ToolACE、Hammer 等）对比，实验表明在 BFCL 上实现 0.828 s 的推理时延，同时 0.6B 小模型准确率提升至 80.1%，在速度‑准确率 Pareto 前沿上优于同规模或更大模型。

**⚠️ 局限性**

主要局限：对检索器性能高度敏感；动态模板在无函数调用或参数语义错误时可能产生冗余/无意义输出；当前仅支持单轮调用，尚未扩展到多轮交互。

---

## 424. Neurosim: A Fast Simulator for Neuromorphic Robot Perception

**arXiv ID:** 2602.15018 | [PDF](https://arxiv.org/pdf/2602.15018v1)

**作者:** Richeek Das `[一作]` (University of Pennsylvania), Pratik Chaudhari `[通讯]` (University of Pennsylvania)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了Neurosim和Cortex两套工具，提供高帧率事件相机、RGB/深度/IMU等多模态传感器的实时模拟，并支持多旋翼机体动力学、闭环控制和在线深度学习训练；

**💡 创新点**

核心创新在于：①利用warp同步CUDA核实现高效事件生成，显著降低原子操作开销；②模块化、低延迟架构使渲染、动力学、通信相互独立；③基于ZeroMQ的Cortex通信库实现零拷贝、强类型多模态数据传输，直接对接深度学习流水线；

**🔧 技术方法**

技术包括：GPU渲染引擎Habitat‑Sim、CUDA事件生成核、RotorPy动力学模型、ZeroMQ + Cortex IPC、Python/ C++ API、ROS 2桥接及Rerun可视化；

**📊 数据集**

使用了Matterport3D、Gibson、Replica、MP3D+Gibson等室内三维资产；并参考M3ED事件相机数据集；在实验中生成的随机MinSnap轨迹用于训练；

**📈 对比分析**

与现有CUDA、PyTorch、CPU实现比较，Neurosim事件模拟率达31 kHz（VGA）/23 kHz（HD），比GPU实现快8–13×、CPU快55–121×；整体模拟速率2300 FPS，显著高于200–1200 FPS；Cortex在IPC传输上可达100+ msg/s（小消息）/250 msg/s（1080p），闭环控制延迟<0.7 ms，传感器率可达2.3；

**⚠️ 局限性**

局限性包括：缺乏硬件在环实验；仅支持Habitat‑Sim渲染，未集成Unreal Engine 5；缺乏更全面的传感器噪声模型（光子噪声、滚动快门等）；单GPU、单机运行限制了更大规模或更高分辨率的实时模拟；

---

## 425. Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report v1.5

**arXiv ID:** 2602.14457 | [PDF](https://arxiv.org/pdf/2602.14457v1)

**作者:** Dongrui Liu `[一作]` (Shanghai AI Laboratory), Jing Shao `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统性评估了前沿 AI（大语言模型）在网络攻击、说服操纵、战略欺骗、无监管研发和自我复制等多维风险上的能力与风险，并在此基础上提出了多项实验基准（PACEbench、RvB、Persuasion、Deception 等）与对应的缓解技术。通过对十余大模型的实验对比，展示了模型在不同风险维度的表现及缓解效果。

**💡 创新点**

创新点：①将前沿风险细分为 5–7 维度，并构建了全新实验基准与评估指标；②设计多样化的自动化攻击与防御场景，量化攻击成功、缓解效果；③针对说服、欺骗与误对齐等安全维度，提出基于 RL/HF 的训练与 RL 优化（GRPO）缓解框架；④整合多模型评测，系统呈现模型在各风险维度的优势与缺陷。

**🔧 技术方法**

技术：基于 LLM 驱动的自动化代理（CAI、Red/Blue Team）、ReAct 规划框架、RLHF+GRPO、Prompt 工程、工具调用（SSH、SQLMap、Nmap 等）、自监督与强化学习策略、模型自评估与安全判别器。

**📊 数据集**

数据集：PACEbench（A/B/C/D‑CVE 赛题）、CyBench、MASK、DeceptionBench、OpenClaw/Moltbook 环境日志、OpenRouter 公开使用统计、9,566 条人类行为记录、Alpaca‑Clean、OpenAI、Anthropic 公开数据等。

**📈 对比分析**

评估方法：使用 Pass@5、PACEbench Score、ASC、DSR/TDSR/SFR 等指标；对比不同模型在网络攻击成功率、说服转移率、欺骗率等方面的表现；在 RvB 防御实验中，迭代 5 次后 DSR 达 90% 且 SDR 为 0%，明显优于合作基线；说服缓解策略平均降低 62% 的意见转移；Cyber Offense 中 Claude Sonnet 4.5 (Thinking) 取得最高 PACEbench Score 0.335，但在全链攻击与防御场景下所有模型均未突破，显示仍有局限。

**⚠️ 局限性**

局限：评估仅覆盖部分模型与任务，无法完全覆盖真实多步骤攻击与长期自适应情形；实验环境为容器化/模拟，可能与真实网络或社交环境差异；缓解策略受训练数据偏差、奖励设计与 RL 策略限制；自我演化与工具/记忆 mis‑alignment 的评测仍相对简化，未来需要更细粒度的长期跟踪与跨模型验证。

---

## 426. Touching Movement: 3D Tactile Poses for Supporting Blind People in Learning Body Movements

**arXiv ID:** 2602.14442 | [PDF](https://arxiv.org/pdf/2602.14442v1)

**作者:** Kengo Tanaka `[一作]` (University of Tsukuba), Chieko Asakawa `[通讯]` (IBM Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过参与式设计和3D打印技术，开发了可触摸的人体姿态模型，使盲人能够在无视觉条件下学习瑜伽静态姿势和健身动作。

**💡 创新点**

创新点在于将自动化姿态重建（Humans in 4D）与可操作的3D打印模型相结合，首次提供完整的三维运动表示，并在实验中证明其优于音频和二维触觉图。

**🔧 技术方法**

采用了人体姿态估计（Humans in 4D）、3D打印（Formlabs Form 3L）以及基于触觉参考标记的支撑结构设计。

**📊 数据集**

研究使用了10名盲人参与者的实验数据，并基于公开的瑜伽姿势与日常健身动作的2D视频帧进行姿态提取。

**📈 对比分析**

通过对完成时间、姿态准确率和提问次数的对比，3D模型在两项实验中都显著提升了学习速度、准确性并减少了澄清需求，整体效果优于音频和二维触觉图。

**⚠️ 局限性**

局限在于样本规模小、仅涵盖简单静态与四步序列动作，缺乏更长序列、多种运动类型以及长期学习与真实使用环境的验证。

---

## 427. LeafNet: A Large-Scale Dataset and Comprehensive Benchmark for Foundational Vision-Language Understanding of Plant Diseases

**arXiv ID:** 2602.13662 | [PDF](https://arxiv.org/pdf/2602.13662v1)

**作者:** Khang Nguyen Quoc `[一作]` (Korea University), Luyl-Da Quach `[通讯]` (FPT University)

**通讯引用:** 626 | [OpenAlex ID](https://openalex.org/A5031714104)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了大规模多模态植物叶片数据集LeafNet及其对应的VQA基准LeafBench，系统评估并对比多种视觉与视觉‑语言模型在植物病害诊断任务中的表现；

**💡 创新点**

创新点在于①提供覆盖22种作物、62种病害、186k张高质量标注图像的多模态数据集，②设计六层级诊断任务的VQA基准，③展示通用模型与域适配模型在多模态任务上的显著差距，进一步证明数据中心化的重要性；

**🔧 技术方法**

主要技术包括数据采集与专家标注、文本-图像对齐的多模态预训练（如CLIP、BLIP‑2、LLaVA、SCOLD等）、零样本与少样本分类、以及结构化多模态指令式推理；

**📊 数据集**

使用了186,000张叶片图像和13,950个问答样本的LeafNet/LeafBench，涵盖97个细粒度类别；

**📈 对比分析**

通过对比12款主流视觉‑语言模型与7款纯视觉模型，在全监督、few‑shot、零样本等多种设置下，发现域特化模型SCOLD在疾病识别可达99.15%，而通用模型准确率仅50–70%；

**⚠️ 局限性**

局限包括：地理与环境多样性不足，缺乏多时相和多光谱数据，文本标注深度有限，导致模型在新环境或复杂症状下泛化受限。

---

## 428. Hippocampus: An Efficient and Scalable Memory Module for Agentic AI

**arXiv ID:** 2602.13594 | [PDF](https://arxiv.org/pdf/2602.13594v1)

**作者:** Yi Li `[一作]` (University of Texas at Dallas), Bingzhe Li `[通讯]` (Hewlett Packard Enterprise Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

为agentic AI提供一种高效、可扩展的上下文记忆模块，通过压缩二进制签名与token-ID流实现快速检索与完整内容恢复。

**💡 创新点**

核心创新在于引入动态Wavelet矩阵（Dynamic Wavelet Matrix）实现流式写入和压缩域中的Hamming-ball检索，取代传统高维向量搜索和知识图遍历，从而显著降低延迟和token消耗。

**🔧 技术方法**

使用技术包括：动态Wavelet矩阵（DWM）、随机索引（Random Indexing）生成二进制签名、Hamming距离搜索、token-ID序列的无损存储与检索、压缩域的位运算等。

**📊 数据集**

使用数据集包括LoCoMo（单跳、多跳、时间推理、开放域四个任务）和LongMemEval‑S（六个任务），用于评估检索准确性与效率。

**📈 对比分析**

与ReadAgent、MemoryBank、MemGPT、A‑mem、MemoryOS、MemOS等六大基线在F1、BLEU‑1、LLM‑as‑a‑Judge、准确率等指标上对比，结果显示检索延迟最高可提升31×、每次查询token使用量降低14×，且在LoCoMo和LongMemEval‑S上实现最高或相当的准确性。

**⚠️ 局限性**

局限性包括：二进制签名的近似语义检索可能在极端语义差异下误检；对随机索引参数的依赖；目前评估仅覆盖LoCoMo和LongMemEval，缺乏对更大规模或多模态场景的验证；以及在极端高频写入场景下DWM更新开销尚未充分探究。

---

## 429. Use What You Know: Causal Foundation Models with Partial Graphs

**arXiv ID:** 2602.14972 | [PDF](https://arxiv.org/pdf/2602.14972v1)

**作者:** Arik Reuter `[一作]` (University of Cambridge), Bernhard Schölkopf `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一种可在测试时根据可获得的因果先验（如因果图或部分祖先关系）对因果基础模型（CFM）进行条件化的方法，提升因果效应估计性能。

**💡 创新点**

创新点在于：①引入部分已知祖先矩阵（PAM）作为可灵活表达因果知识的形式；②通过在Transformer的特征自注意力中注入可学习的注意力偏置（soft attention）并结合GCN编码器，实现对部分或完整因果信息的有效利用；③实现单一CFM可适应无信息、部分信息和完整信息三种场景。

**🔧 技术方法**

使用的技术包括：Prior-Data-Fitted Networks (PFN)框架、Transformer结构、特征自注意力偏置、图卷积网络（GCN）与自适应层归一化、软/硬注意力偏置、半监督/无监督的因果结构先验生成。

**📊 数据集**

使用的数据集：①大规模线性高斯合成数据（30k个样本，最多50个变量）；②自定义的复杂因果先验生成的合成数据；③半合成RealCause基准（IHDP、ACIC、CPS、PSID）。

**📈 对比分析**

与无图条件化的基线以及专门针对特定因果结构训练的模型进行比较。实验结果显示：soft attention与GCN组合在NLL、MSE、R²等指标上均显著优于无图基线，且在部分祖先信息缺失时性能接近专用模型；在复杂合成数据和RealCause基准上，加入部分祖先信息能显著降低PEHE和ATE误差。

**⚠️ 局限性**

局限性：①仅考虑了部分已知祖先矩阵的条件化，未探讨功能机制或噪声分布等其他形式的先验；②实验依赖于合成或半合成数据，缺乏大规模真实世界因果基准；③在极端缺失信息（近全0祖先矩阵）时，性能仍会退化到无图基线水平。

---

## 430. DCTracks: An Open Dataset for Machine Learning-Based Drift Chamber Track Reconstruction

**arXiv ID:** 2602.14571 | [PDF](https://arxiv.org/pdf/2602.14571v1)

**作者:** Qian Liyan `[一作]`, Huang Xingtao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了一套基于BESIII MDC的漂移室轨道重建公开数据集，并给出统一评估指标；

**💡 创新点**

首次实现对单轨、常规双轨和紧邻双轨三类事件的完整模拟与噪声覆盖，构建标准化ML轨道重建基准；

**🔧 技术方法**

采用Graph Neural Network（GNN）端到端轨道重建框架，以及传统Kalman滤波和Runge‑Kutta拟合的基线方法；

**📊 数据集**

使用GEANT4 + BOSS仿真得到的BESIII多层漂移室数据，包含单粒子、π⁺π⁻两轨及Δϕ≈0.2的紧邻两轨事件；

**📈 对比分析**

通过定义hit效率/纯度、track效率/克隆/假轨率、p_T分辨率等多维度指标对比，GNN在单轨与常规双轨上与基线相近，但在紧邻双轨时track效率显著下降（约76%），假轨率升高；

**⚠️ 局限性**

缺点在于仅覆盖低背景低多重度场景，未包含低p_T缠绕轨迹、弯曲轨迹和真实数据；后续需扩展至更复杂事件与多子系统联合分析。

---

## 431. When Test-Time Guidance Is Enough: Fast Image and Video Editing with Diffusion Guidance

**arXiv ID:** 2602.14157 | [PDF](https://arxiv.org/pdf/2602.14157v1)

**作者:** Ahmed Ghorbel `[一作]` (CMAP Ecole Polytechnique), Yazid Janati `[通讯]` (Institute of Foundation Models)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无需额外训练、仅在测试时通过无向量-雅可比乘积（VJP-free）近似实现的图像与视频编辑框架，核心思路是把编辑任务视为潜在空间中的 inpainting 问题。

**💡 创新点**

创新点在于：①对 VJP-free 近似给出理论解析，揭示其相当于对噪声预测器 Jacobian 的简化；②通过该近似实现了线性逆问题的闭式后验采样；③在大规模扩散模型上进行多尺度图像与视频编辑的基准测试，证明其在相同计算预算下可与甚至超越训练基方法。

**🔧 技术方法**

使用技术包括：预训练文本条件扩散模型、逆向马尔可夫链采样、Tweedie 公式、噪声预测器、混合式 VJP-free 近似、潜在空间编码/解码、Gaussian 似然/先验结合、后验采样的高斯共轭更新。

**📊 数据集**

实验数据集包括：Image Editing 数据集（1260 张图像 + 1000 张子集）与视频编辑数据集（133 条视频，均配有 mask 与文本提示），以及用于基准的 Stable Diffusion 3.5、其他大型文本‑图像模型和视频扩散模型。

**📈 对比分析**

对比方法包括多种训练‑free（如 Diffusion‑Guided、ControlNet‑inpainting 等）和训练‑based（ControlNet、Fill、-VACE 等）方案；在固定 NFEs 或相同运行时长的条件下，采用 FID、pFID、cPSNR（图像）和 CLIP‑Score、FVD、cPSNR（视频）进行评估。结果显示，测试‑time 指导在大多数指标上与训练‑based 方案相当，甚至在某些场景下表现更佳。

**⚠️ 局限性**

局限性包括：①忽略噪声预测器的 Jacobian 可能在复杂或高度非线性情形下引入误差；②潜在空间与像素空间的非线性映射使得对像素‑级 mask 的精确处理受限；③仍需依赖高质量预训练扩散模型；④对长视频或高分辨率图像的实时性仍受限，计算成本虽低但总体处理时间仍可显著。

---

## 432. A Deployment-Friendly Foundational Framework for Efficient Computational Pathology

**arXiv ID:** 2602.14010 | [PDF](https://arxiv.org/pdf/2602.14010v1)

**作者:** Yu Cai `[一作]` (Hong Kong University of Science and Technology), Kwang-Ting Cheng `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 25863 | [OpenAlex ID](https://openalex.org/A5077687075)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 LitePath 框架，将轻量化 PFM LiteFM 与自适应补丁选择器 APS 结合，实现对大模型过参数化与补丁冗余的双重缓解，支持在低功耗边缘设备上高效推理。

**💡 创新点**

创新点包括：① 通过多教师蒸馏（Virchow2、H‑Optimus‑1、UNI2）构建 ViT‑Small LiteFM，显著减小模型规模；② APS 采用均匀+注意力双采样策略，基于浅层特征快速定位诊断相关补丁；③ 引入 Deployability Score (D‑Score) 对准确率与计算效率进行统一量化；④ 在多中心、多器官、多任务（共 37 队列）上验证，显示接近大模型性能但推理速度提升 100×、能耗下降 170×。

**🔧 技术方法**

技术细节：ViT‑Small backbone，知识蒸馏（ℓ1 loss + 投影头），ABMIL 作为 MIL 归纳器，APS 评分网络基于浅层特征预测注意力分数，半精度 FP16 推理，GPU/Jetson Orin Nano Super 部署，宏观 AUC 评估，D‑Score 计算。

**📊 数据集**

预训练使用 72,280 公共 WSI 共 190M 图像（来源 33 数据集如 TCGA、CPTAC、PANDA 等）。下游任务采用 PathBench 26 任务（肺、乳腺、胃、结肠），共 37 队列、15672 张切片、9808 患者，涵盖内部、外部与前瞻性数据。

**📈 对比分析**

与 19 现有 PFM（含 Virchow2、H‑Optimus‑1 等）进行宏观 AUC、参数、FLOPs、吞吐量比较。LitePath 平均排名 5.6（第二名），AUC 保留 99.71%，参数 22.5M（Virchow2 的 1/28），Jetson 推理 208 张/小时（比 Virchow2 RTX 3090 快 104.5×），能耗 0.36 kWh/3000 张（比 RTX 3090 低 171×），D‑Score 86.31% 为最高。

**⚠️ 局限性**

局限性：① 预训练数据规模仍有限，学生模型性能上限受限；② 仅在单一分辨率（WSI 基层）上验证，跨分辨率泛化未充分探究；③ 需要 GPU/边缘设备，尚未证明在无 GPU 环境下可行；④ 对不同机构、不同染色批次的鲁棒性还有待进一步评估。

---

## 433. Error Patterns in Historical OCR: A Comparative Analysis of TrOCR and a Vision-Language Model

**arXiv ID:** 2602.14524 | [PDF](https://arxiv.org/pdf/2602.14524v1)

**作者:** Ari Vesalainen `[一作]` (University of Helsinki), Mikko Tolonen `[通讯]` (University of Helsinki)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对18世纪印刷文本的 OCR 进行错误模式分析，比较了专用 OCR 变压器 TrOCR 与通用视觉‑语言模型 Qwen 在行级输入上的表现，并通过错误结构而非单一准确率评估两种架构的可靠性。

**💡 创新点**

创新点在于提出了基于模型架构的错误性格假设（视觉定位、语言正则化、正字法归一化与错误传播），构建了面向历史文本的错误分类框架，并展示了即使 CER/WER 相近，模型在错误局部性、可检知性与学术风险上也有显著差异。

**🔧 技术方法**

主要技术包括：Transformer‑OCR 视觉编码器‑解码器（TrOCR）、多模态预训练的 Vision‑Language Model（Qwen）以及长度加权的 CER/WER、基于 Nguyen 等的错误代理（真实/非词、边界错误、图形混淆等）进行细粒度错误分析。

**📊 数据集**

使用的数据集为约 195,000 行级训练图像（手工标注、半自动化 ECCO 对齐与合成数据）和约 10,000 行级测试图像，覆盖英格兰 18 世纪印刷书籍的多种扫描模式与字形多样性。

**📈 对比分析**

比较方法：在同一行级分割输入、相同前处理和解码设置下，测量 Qwen 与 TrOCR 的长度加权 CER/WER，并通过错误代理可视化对比两者的错误结构。实验结果显示 Qwen 在 CER/WER 上普遍低于 TrOCR，但 TrOCR 在视觉保真度与错误传播上表现更强，Qwen 则表现出更多语义化正字化与边界错误。

**⚠️ 局限性**

限制：实验仅在行级输入上进行，未涉及页面级布局分析；仅评估英文 18 世纪文本，对其他语言、手稿或多语言语料的适用性待验证；未对模型大小进行控制，可能与模型容量有关；错误分析聚焦于可检测代理，未捕捉所有潜在的学术风险。

---

## 434. Affine Rank Minimization is ER Complete

**arXiv ID:** 2602.14037 | [PDF](https://arxiv.org/pdf/2602.14037v1)

**作者:** Angshul Majumdar `[一作]` (Indraprastha Institute of Information Technology), Angshul Majumdar `[通讯]` (Indraprastha Institute of Information Technology)

**通讯引用:** 6218 | [OpenAlex ID](https://openalex.org/A5020310463)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了在给定固定常数秩上，只有线性等式约束和全局秩上限的“仿射秩最小化”（ARM）问题的决策版本是 ∃ℝ‑完全的，即与实数代数可满足性问题等价。

**💡 创新点**

创新点在于：① 仅用一次全局秩约束即可模拟所有非线性算术门；② 设计了一个常数大小的“行列式强迫”子结构，将乘法约束转化为对 4×4 子矩阵行列式为零的判断；③ 通过固定 3×3 的完整子矩阵消除 GL(3) 的自由度，使得各个子结构可无冲突地嵌入同一矩阵，完成全局正确性证明。

**🔧 技术方法**

采用了实数代数理论中的行列式约束、算术电路的门等价形式、秩分解作为存在性证明以及多项式时间的多项式编码和位长度分析。

**📊 数据集**

本工作不涉及实验或数据集；研究完全是理论复杂度分析。

**📈 对比分析**

对比方法：传统的低秩优化多采用凸松弛或非凸因子化技巧，无法得到精确可满足性判断；本文通过证明 ARM(3) 的 ∃ℝ‑完备性，表明即使最小化约束仅为线性等式，问题仍然在 ∃ℝ 内，暗示在多项式时间内求解该问题是不可能的（除非 ∃ℝ = P）。

**⚠️ 局限性**

局限性：结果仅适用于固定常数秩（如 3）；对更一般的可变秩约束或非线性等式的情况仍未给出完整复杂度分类；此外，证明基于构造性的行列式强迫子结构，实际实现复杂度高，无法直接转化为有效的求解算法。

---

## 435. PhGPO: Pheromone-Guided Policy Optimization for Long-Horizon Tool Planning

**arXiv ID:** 2602.13691 | [PDF](https://arxiv.org/pdf/2602.13691v1)

**作者:** Yu Li `[一作]` (Southeast University), Lei Feng `[通讯]` (Southeast University)

**通讯引用:** 133508 | [OpenAlex ID](https://openalex.org/A5100659481)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于信息素的策略优化框架PhGPO，显式利用历史成功工具使用轨迹中的工具转移模式来引导长周期工具规划；

**💡 创新点**

通过将蚁群算法中的信息素概念引入到工具转移图中，构造可学习的工具转移先验，并结合任务相关与任务无关的信息素以及逐步衰减机制，实现了历史经验的显式、可重用性；

**🔧 技术方法**

利用工具转移图和参数化的工具-参数调用集合，结合信息素更新（沉积与蒸发）、进化式多目标采样、基于组相对优势的GRPO策略优化以及逐步递增的训练课程；

**📊 数据集**

在三大长周期工具规划基准上进行评估，分别为Toolathlon、TOUCAN与TRAJECT-Bench；

**📈 对比分析**

与标准提示/规划方法、检索增强生成方法、强化学习优化方法以及图结构导航模型进行对比，PhGPO在所有基准上均显著提升匹配率与下一个工具的准确率，尤其在长序列任务中效果最为显著；

**⚠️ 局限性**

依赖于足够多且质量高的验证成功轨迹作为信息素的来源，若成功轨迹稀缺则信息素更新不可靠；信息素的更新与衰减需要手动调参；方法对工具数量和多样性的扩展性和适配性尚待进一步验证。

---

## 436. The Potential of CoT for Reasoning: A Closer Look at Trace Dynamics

**arXiv ID:** 2602.14903 | [PDF](https://arxiv.org/pdf/2602.14903v1)

**作者:** Gregor Bachmann `[一作]` (Apple), Moin Nabi `[通讯]` (Apple)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过定义潜能（potential）量化链式推理（CoT）的贡献，并系统分析不同模型在解数学竞赛题时CoT的形状与效果，探讨CoT的可转移性与优化方法。

**💡 创新点**

创新点在于提出潜能度量，将CoT的每一步与模型最终成功概率关联，揭示“推理洞察”“推理跳跃”“推理分岔”等关键特征；同时研究弱模型在获得强模型的部分CoT后能否快速提升性能。

**🔧 技术方法**

使用了自回归语言模型生成CoT，并通过采样估计潜能；对潜能曲线进行量化统计；设计优化算法挑选最大潜能子链；使用转移实验对模型进行条件生成。

**📊 数据集**

实验数据集包括 AIME‑2024、AIME‑2025、MATH‑500、GPQA‑Diamond、HumanEval；对多种模型（Qwen2.5‑1.5B/7B、Qwen3‑0.6B/32B、Llama‑3.1‑8B/70B、GPT‑OSS‑20B 等）进行评估。

**📈 对比分析**

通过比较标准CoT与潜能优化CoT的潜能曲线、Pass@k 及准确率，显示优化CoT能显著提升潜能的单调性和最终解答成功率；弱模型在获得 20%+ CoT 后的准确率可超过其原始水平。

**⚠️ 局限性**

局限性在于潜能估计需大量采样、计算昂贵；实验集中在竞赛级数学与少量编码任务，未验证在更广泛领域的通用性；转移效果受模型家族相似度影响，跨域应用尚需进一步研究。

---

## 437. MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders

**arXiv ID:** 2602.14110 | [PDF](https://arxiv.org/pdf/2602.14110v1)

**作者:** Xu Huang `[一作]` (ByteDance), Qiwei Chen `[通讯]` (ByteDance)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了MixFormer，一种统一的Transformer风格架构，既能建模用户行为序列，又能高阶特征交互，并通过用户‑物品解耦实现高效推理。

**💡 创新点**

核心创新在于将序列建模与特征交互放入同一参数空间，消除传统分离设计导致的 co‑scaling 牵制；同时引入HeadMixing、per‑head FFN 和用户‑物品解耦的 Request‑Level Batching，以实现高效且可共伸缩的模型。

**🔧 技术方法**

采用Transformer解码器骨架，设计Query Mixer（HeadMixing+per‑head FFN）、Cross‑Attention（特征驱动序列聚合）和Output Fusion（per‑head FFN）；使用SwiGLU激活、LayerNorm、RMSProp 等训练细节；实现用户‑物品解耦的遮罩与请求级批处理。

**📊 数据集**

实验基于抖音（Douyin）推荐系统的离线数据集，涵盖两周内十万亿交互记录，包含300+非序列与序列特征。

**📈 对比分析**

与多种基准（STCA、RankMixer、DCNv2、Wukong、OneTrans 等）进行对比。MixFormer 在AUC、UAUC 上均优于最强对手，参数量相同或更少；在线 A/B 测试在抖音与抖音 Lite 上显著提升活跃天数、使用时长、点赞、完成播放等指标。

**⚠️ 局限性**

主要局限包括：对极大序列长度的计算仍昂贵，需依赖请求级批处理；在不同业务场景与数据分布下的迁移性尚未系统验证；模型规模与实现复杂度高，部署需精细调优。

---

## 438. The Sufficiency-Conciseness Trade-off in LLM Self-Explanation from an Information Bottleneck Perspective

**arXiv ID:** 2602.14002 | [PDF](https://arxiv.org/pdf/2602.14002v1)

**作者:** Ali Zahedzadeh `[一作]`, Behnam Bahrak `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过信息瓶颈框架系统评估大型语言模型在自我解释时的充分性与简洁性，并在英语和波斯语的ARC Challenge数据集上进行实验。

**💡 创新点**

创新点在于：①将充分性与简洁性定义为互补维度，并引入压缩约束的自解释生成管线；②利用固定评分模型衡量解释充分性；③首次对不同语言的自解释在长度压缩下的表现进行大规模对比。

**🔧 技术方法**

技术包括：信息瓶颈理论、长度约束提示生成、固定评分模型（Qwen 1.7B）进行概率评分、句子级嵌入相似度评估、结构化日志与CSV存储。

**📊 数据集**

数据集：ARC Challenge（2590道多项选择科学问题），并提供自动翻译的波斯语版本。

**📈 对比分析**

比较方法：在完整解释、不同长度压缩水平下分别测量准确率、充分性概率和嵌入相似度；与无解释基线对比。结果显示，多数模型在压缩至60-70%长度后仍保持高达80%+充分性，部分模型（如DeepSeek V3.1、GPT‑4o‑mini）在90%压缩下仍维持高准确率，说明冗余步骤可被删减。

**⚠️ 局限性**

局限性：仅评估多项选择任务；仅使用单一评分模型；波斯语翻译质量可能影响结果；未涵盖开放式生成任务或多模态解释；对模型内部机制的解释不足。

---

## 439. Zwitscherkasten -- DIY Audiovisual bird monitoring

**arXiv ID:** 2602.13330 | [PDF](https://arxiv.org/pdf/2602.13330v1)

**作者:** Dominik Blum `[一作]` (Technische Hochschule Ingolstadt), Torsten Schön `[通讯]` (Technische Hochschule Ingolstadt)

**通讯引用:** 76 | [OpenAlex ID](https://openalex.org/A5078954824)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了Zwitscherkasten，一套低成本、DIY多模态边缘设备鸟类监测系统，将音频与视觉深度学习模型集成在Raspberry Pi及Rubik Pi等资源受限硬件上，实现实时、非侵入式鸟种识别；

**💡 创新点**

创新点在于：①通过两阶段音频检测-分类架构降低能耗；②采用轻量化Transformer PaSST与高效CNN（EfficientNet、MobileNetV3）并行；③在嵌入式设备上实现完整的音视频联动与本地HMI；④利用弱监督式检测生成视觉训练样本，减少人工标注；

**🔧 技术方法**

使用的技术包括：语音与图像的mel‑spectrogram特征提取、SpecAugment与Mixup增广、轻量化卷积与Transformer架构（PaSST、EfficientNet‑B0/B3、MobileNetV3‑Small、YOLOv11/26）、TensorFlow Lite/CoreML量化推理、MPS/CPU混合训练、两阶段事件触发推理、Web/移动端Dashboard可视化；

**📊 数据集**

数据集主要有：Xeno‑Canto（欧盟鸟类音频），BirdSet、Macaulay Library、Tierstimmenarchiv用于补充；视觉数据来自iNaturalist的研究级照片，采用弱监督生成检测框；训练/验证/测试划分采用90/10比例（音频）和60/20/20比例（视觉）；

**📈 对比分析**

方法比较：在256类鸟种上，PaSST实现94.39% Top‑1、97.60% Top‑5；EfficientNet‑B3 92.93%/97.37%；EfficientNet‑B0 91.69%/97.31%；MobileNetV3‑Small 85.62%/94.75%；视觉端YOLOv26m mAP@0.5:0.95达0.7442；EfficientNet‑B1 Top‑1 82.06%；整体显示Transformer与中等规模CNN在准确率上领先，但在推理速度与能耗上均需折中；

**⚠️ 局限性**

局限性包括：①视觉检测依赖弱监督框，存在标注噪声；②持续视频推理在Raspberry Pi上帧率低，实时性不足；③多模态融合仅在离线或手动合并，未实现端到端自适应；④缺乏长周期实地验证和对不同环境噪声鲁棒性的评估；⑤模型压缩与量化效果仍待进一步优化。

---

## 440. OPBench: A Graph Benchmark to Combat the Opioid Crisis

**arXiv ID:** 2602.14602 | [PDF](https://arxiv.org/pdf/2602.14602v1)

**作者:** Tianyi Ma `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了OPBench基准，涵盖五个与阿片危机相关的数据集，系统评估并比较各种图学习方法在不同任务上的表现。

**💡 创新点**

首次将异构图、超图等多种复杂结构与阿片危机多维场景相结合，提供统一的评测框架和开源库，为图学习在公共卫生危机中的应用提供可复制的基准。

**🔧 技术方法**

采用异构图神经网络（HAN、HGT）、多关系图神经网络（R‑GCN、GraphENS）、超图GNN（HGNN、HNHN、ED‑HNN）等多种图学习模型，并在统一实验设置下进行对比实验。

**📊 数据集**

使用的五个数据集包括：Pdmp‑OD‑Det（医疗索赔异构图）、X‑HyDrug‑Comm（超图社交网络社区检测）、X‑HyDrug‑Role（超图角色识别）、X‑MRDrug‑Role（多关系图角色识别）以及NHANES‑Diet（异构图营养与阿片误用预测）。

**📈 对比分析**

采用统一的数据划分（10%、20%、50%训练）与标准指标（AUC、F1‑Macro/ Micro、Accuracy、G‑Mean）进行多模型比较，发现HetGNN/HyGNN显著优于传统GNN，且数据级不平衡处理（如AD‑GSMOTE、GraphSMOTE）可大幅提升性能。

**⚠️ 局限性**

局限性包括：数据标注仍受限于隐私与专家劳动、不同任务对模型的需求差异导致性能不一、对极端类别不平衡的鲁棒性仍有提升空间。

---

## 441. Designing Health Technologies for Immigrant Communities: Exploring Healthcare Providers' Communication Strategies with Patients

**arXiv ID:** 2602.13598 | [PDF](https://arxiv.org/pdf/2602.13598v1)

**作者:** Zhanming Chen `[一作]` (University of Minnesota), Ji Youn Shin `[通讯]` (University of Minnesota)

**通讯引用:** 763 | [OpenAlex ID](https://openalex.org/A5018185775)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对美国明尼阿波利斯地区服务苗族移民患者的医疗服务提供者进行访谈，归纳出四种有效沟通策略，并提出技术设计启示。

**💡 创新点**

首次系统识别并细化了移民群体中医疗提供者的沟通策略，并将文化适应性概念引入健康技术设计。

**🔧 技术方法**

采用半结构化访谈、现场观察以及基于扎根理论的主题分析技术。

**📊 数据集**

使用15位医疗提供者的访谈文本和两家诊所的观察数据。

**📈 对比分析**

研究未进行实验性比较或性能评估，主要通过质性数据的主题编码呈现发现。

**⚠️ 局限性**

样本以熟悉苗族文化的提供者为主，缺乏患者视角，且未考虑交叉身份对沟通的影响。

---

## 442. Nanbeige4.1-3B: A Small General Model that Reasons, Aligns, and Acts

**arXiv ID:** 2602.13367 | [PDF](https://arxiv.org/pdf/2602.13367v1)

**作者:** Chen Yang `[一作]` (Nanbeige LLM Lab, Boss Zhipin), Zongchao Chen `[通讯]` (Nanbeige LLM Lab, Boss Zhipin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一款3B参数的统一通用模型Nanbeige4.1-3B，兼具推理、代码生成和长时序代理能力；

**💡 创新点**

创新点在于将点对点与对比奖励模型结合、为代码生成引入复杂度奖励，并通过交互层级与全轨迹奖励实现深度搜索与长时序规划；

**🔧 技术方法**

采用了SFT+多阶段强化学习、点对点和对比奖励、复杂度感知奖励以及工具调用训练的技术；

**📊 数据集**

使用了LiveCodeBench、IMO/ AIME/HMMT 等编程与数学数据集，GPQA/HLE 科学数据集，Arena-Hard-V2/Multi-Challenge 对齐数据集，BFCL/Tau2-Bench 工具使用数据集，以及GAIA、BrowseComp等深度搜索基准；

**📈 对比分析**

在所有评测中，Nanbeige4.1-3B 在推理、代码、对齐和工具使用上显著优于同规模Qwen3-4B以及前一代Nanbeige4-3B-2511，甚至在多数指标上超过30B–32B级别模型；

**⚠️ 局限性**

局限性主要包括：仍受3B参数规模限制，部分极难任务仍落后于更大模型；训练成本高昂；推理效率与大模型相比仍需提升。

---

## 443. Revisiting Worker-Centered Design: Tensions, Blind Spots, and Action Spaces

**arXiv ID:** 2602.13424 | [PDF](https://arxiv.org/pdf/2602.13424v1)

**作者:** Shuhao Ma `[一作]` (University of Lisbon), Nuno Jardim Nunes `[通讯]` (University of Lisbon)

**通讯引用:** 6300 | [OpenAlex ID](https://openalex.org/A5074097147)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过四个视角（基础、前台、后台、后台）对食品配送平台的工作者中心设计进行系统化分析，揭示其紧张关系与盲点，并提出多劳工系统与诊断-生成路径两种新框架。

**💡 创新点**

创新点在于将工作者中心设计置于多劳工系统视角，系统识别跨劳工链的冲突与设计扭曲，并提出诊断-生成路径来扩展设计行动空间。

**🔧 技术方法**

采用了基于四镜头的解释性分析框架，结合双钻石模型和服务设计工具进行理论与方法构建。

**📊 数据集**

数据来源为文献综述、欧洲与中国的实地观察、与平台设计师及配送员的访谈，以及公共媒体报道等定性资料。

**📈 对比分析**

本研究并未进行传统意义的性能比较，而是通过对比分析识别出工作者需求与平台实践之间的差距，证明了多劳工系统框架能更全面地揭示设计失效与潜在机会。

**⚠️ 局限性**

局限性包括：研究依赖于特定行业与地区，难以推广到其他服务平台；设计师在实际组织中的权限与时间有限，诊断-生成路径的可操作性尚未在实践中验证。

---

## 444. Orthogonalized Multimodal Contrastive Learning with Asymmetric Masking for Structured Representations

**arXiv ID:** 2602.14983 | [PDF](https://arxiv.org/pdf/2602.14983v1)

**作者:** Carolin Cissee `[一作]`, Zahra Ahmadi `[通讯]` (Lower Saxony Center for AI and Causal Methods in Medicine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 COrAL——一种基于自监督对比学习的多模态表示学习框架，用于同时捕获冗余、唯一与协同信息。

**💡 创新点**

创新点在于双路径编码器与正交约束实现共享与专属子空间的严格分离，并引入非对称遮掩策略主动激发跨模态协同特征。

**🔧 技术方法**

主要技术包括Transformer‑based 多模态编码器、InfoNCE 对比损失、正交正则化（cosine embedding loss）以及渐进式非对称遮掩机制。

**📊 数据集**

使用了合成的 Trifeature 数据集以及 MultiBench 的五个真实多模态基准（MIMIC III、CMU‑MOSEI、CMU‑MOSI、UR‑FUNNY、MUsTARD）。

**📈 对比分析**

与 CLIP、FactorCL、CoMM、InfMasking 等方法进行线性评估对比，COrAL 在合成数据上显著提升唯一与协同指标，在 MultiBench 上平均性能略优或相当，并展现出较低的方差。

**⚠️ 局限性**

局限性包括：遮掩策略目前仅在两模态场景下最优，扩展到更多模态和高阶交互仍需研究；对遮掩比例敏感；在部分真实数据中协同捕获效果有限，未在极大规模多模态数据集上验证。

---

## 445. Agent-OSI: A Layered Protocol Stack Toward a Decentralized Internet of Agents

**arXiv ID:** 2602.13795 | [PDF](https://arxiv.org/pdf/2602.13795v1)

**作者:** Wenxin Xu `[一作]` (Shenzhen University), Soung Chang Liew `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 10518 | [OpenAlex ID](https://openalex.org/A5019164720)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Agent-OSI 六层去中心化代理网络协议栈，支持安全通信、去中心化身份、支付结算、可验证执行与语义互操作。

**💡 创新点**

创新点在于将 HTTP 402 作为支付挑战，并通过跨层签名绑定实现可验证支付与执行证据；构建完整的六层参考架构。

**🔧 技术方法**

采用 TLS/QUIC、MLS、DID/VC、OAuth、ILP、Escrow 智能合约、TEE/零知识证明、JSON Schema、XMTP 等技术。

**📊 数据集**

主要使用 Stable Diffusion XL（SDXL）生成图片任务作为实验数据集，结合 HuggingFace Diffusers 进行评测。

**📈 对比分析**

通过 AgentMarket 原型与 Web2.0 中央计费、Web3.0 全链上 Escrow 基线对比，发现链上会话成本降低约 51%，在生成任务中链上确认延迟被任务执行时间掩盖，吞吐量受链上交易速率限制。

**⚠️ 局限性**

局限性包括：隐私与链接性泄露风险、身份委托与密钥管理复杂度、支付争议与解争机制不完整、可验证执行多样化带来的成本与实现难度，以及链上吞吐瓶颈。

---

## 446. Tutoring Large Language Models to be Domain-adaptive, Precise, and Safe

**arXiv ID:** 2602.13860 | [PDF](https://arxiv.org/pdf/2602.13860v1)

**作者:** Somnath Banerjee `[一作]` (Indian Institute of Technology Kharagpur), Somnath Banerjee `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 1258 | [OpenAlex ID](https://openalex.org/A5103228486)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一套“Responsible Intelligence”框架，通过领域适配、安全约束与文化多语种对齐，使大型语言模型在技术精准、伦理安全和全球包容性方面更可靠。

**💡 创新点**

提出 DistALANER、GraphContextGen、SafeInfer、Soteria 等多种轻量级“tutoring”机制，并构建 TechHazaraQA 与 Cultural Kaleidoscope 数据集，填补了符号逻辑攻击和跨文化安全评估的空白。

**🔧 技术方法**

弱监督与主动学习、图检索与 Personalized PageRank、解码时安全对齐、奖励模型微调 (ORPO)、少量参数调优 (功能头) 等。

**📊 数据集**

170,000 条软件缺陷报告与手册、Wikidata 结构化知识图谱、TechHazaraQA 7,745 个高风险技术问题、Cultural Harm Evaluation 15,000 条跨文化查询。

**📈 对比分析**

在技术 QA、符号逻辑攻击 ASR 和文化危害率上，与基线模型相比，SafeInfer 将攻击成功率降至1.09%，Soteria 将多语言 ASR 从0.46降至0.29，文化危害率从71.96%降至3.07%。

**⚠️ 局限性**

仍依赖人工标注的辅助、对极端低资源语言的覆盖有限、模型对非常规多语代码混杂的鲁棒性尚未完全验证，且“tutoring”手段的长期适应性与可解释性待进一步研究。

---

## 447. Probabilistic RNA Designability via Interpretable Ensemble Approximation and Dynamic Decomposition

**arXiv ID:** 2602.13610 | [PDF](https://arxiv.org/pdf/2602.13610v1)

**作者:** Tianshuo Zhou `[一作]` (Oregon State University), Liang Huang `[通讯]` (Oregon State University)

**通讯引用:** 2966 | [OpenAlex ID](https://openalex.org/A5075538786)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出利用ensemble approximation和概率分解框架，对RNA二级结构在Turner模型下给出可设计性（概率上界）的理论与算法。

**💡 创新点**

创新点在于：①将竞争性rival结构的集体热力学优势纳入概率上界计算；②证明结构可设计性可分解为局部motif的概率乘积；③设计线性时间动态规划搜索最优分解。

**🔧 技术方法**

使用的技术包括：rival结构搜索与ensemble approximation、motif概率上界计算、线性时间动态规划、ViennaRNA能量模型、C++实现并行化。

**📊 数据集**

采用ArchiveII100（1144天然结构）和Eterna100（100人造结构）两个公开基准集。

**📈 对比分析**

与CountingDesign及其改进版比较，LinearDecomposition得到更紧的概率上界，平均提升0.075–0.135；并在两组数据上保持毫秒级到秒级的高效性。

**⚠️ 局限性**

局限性：上界可能不够紧，rival结构生成困难导致某些motif上界不佳；对极大motif仍需近似；未给出直接的设计策略，仅提供上界与解释。

---

## 448. Language Model Memory and Memory Models for Language

**arXiv ID:** 2602.13466 | [PDF](https://arxiv.org/pdf/2602.13466v1)

**作者:** Benjamin L. Badger `[一作]` `[通讯]` (IBM), Benjamin L. Badger (IBM)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言模型在隐藏层嵌入中存储输入信息的能力，并提出并评估可并行化的编码器-解码器记忆模型。

**💡 创新点**

提出将记忆嵌入与下一词预测结合的联合损失，利用冻结的自编码器嵌入和课程学习实现高信息保留的记忆；展示传统因果模型记忆信息低。

**🔧 技术方法**

使用 Transformer、Masked Mixer、autoencoder、Encoder‑Decoder 架构，联合交叉熵和 copy 损失，训练冻结或可训练的编码器；采用 entropy ratio、Hamming metric、信息保留实验。

**📊 数据集**

FineWeb、FineWeb‑edu、FineMath、BERT large、Llama 3.1 1b、Qwen 3 0.6b 等数据集。

**📈 对比分析**

通过自编码器信息恢复、copy、blank copy 以及 token 准确率对比，结果显示自编码器记忆显著高于因果或检索模型；联合目标训练的记忆模型在输入信息保持上优于单一因果训练，部分基准（MMLU、LongBench 等）表现略有提升。

**⚠️ 局限性**

对因果模型记忆信息低的解释仍有局限；联合目标训练提升速度下降；记忆模型在大多数通用语言基准上提升有限；仅在特定数据集和任务上验证；需要进一步探索更强的记忆检索和多模态应用。

---

## 449. LAF-YOLOv10 with Partial Convolution Backbone, Attention-Guided Feature Pyramid, Auxiliary P2 Head, and Wise-IoU Loss for Small Object Detection in Drone Aerial Imagery

**arXiv ID:** 2602.13378 | [PDF](https://arxiv.org/pdf/2602.13378v1)

**作者:** Sohail Ali Farooqui `[一作]`, Shahnawaz Alam `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在无人机航拍图像中提出 LAF‑YOLOv10，一种专门针对小目标检测的轻量级单阶段检测框架。

**💡 创新点**

通过将四种成熟技术（部分卷积、注意力引导的特征金字塔、P2 辅助检测头+P5 头移除、Wise‑IoU v3 损失）有机集成到 YOLOv10n 上，实现了显著的 mAP 提升。

**🔧 技术方法**

技术细节包括：PC‑C2f（仅对 1/4 通道使用 3×3 卷积）来压缩骨干网络计算；AG‑FPN 采用 SE 通道注意力和 DySample 上采样以提升跨尺度融合质量；在 160×160 分辨率上增加 P2 头并剔除 P5 头；Wise‑IoU v3 用于抑制标注噪声导致的回归梯度干扰；整体使用 YOLOv10n 的双分支训练策略和多尺度增广。

**📊 数据集**

主要实验数据集为 VisDrone‑DET2019（验证集）和 UAVDT；在 Jetson Orin Nano 上进行边缘部署基准测试。

**📈 对比分析**

与 YOLOv5n、YOLOv8n、YOLOv10n、YOLO11n、BGF‑YOLOv10、OSD‑YOLOv10、SSCW‑YOLO、TOE‑YOLO 等方法对比，LAF‑YOLOv10 在 VisDrone 验证集上达到 35.1% mAP@0.5（2.3 M 参数），比 YOLOv10n 提升 3.3 个百分点；在 UAVDT 上实现 35.8% mAP@0.5；在 Jetson Orin Nano 上实现 24.3 FPS，满足低功耗 UAV 实时检测需求。

**⚠️ 局限性**

主要限制包括：仍存在较高的漏检和背景误检（TIDE 分析显示 Miss、Bkg 误差占主导）；模型 GFLOPs 仍偏高，导致边缘设备内存占用大；对稀有类别（如自行车、顶篷三轮车）性能提升有限；未处理目标旋转、尺度极端小目标（<8×8）以及未在官方测试集上提交结果；训练时需要更多显存和时间。

---

## 450. Truthful Reporting of Competence with Minimal Verification

**arXiv ID:** 2602.14076 | [PDF](https://arxiv.org/pdf/2602.14076v1)

**作者:** Reshef Meir `[一作]` (Technion Israel Institute of Technology), Omer Ben-Porat `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并分析了一类通过有限核查激励主体诚实报告其类型的机制，重点研究了在确定性与噪声核查情况下的最优折衷。

**💡 创新点**

首次给出满足“诚实为弱支配策略且诚实报数不被惩罚”的机制，并证明Monotone‑Cutoff Verification在确定性核查下达到效率边界；同时设计了可在噪声核查下实现低偏差与低核查率的Polynomial Verification机制。

**🔧 技术方法**

基于机制设计与完美信息下的验证、分段阈值策略、正确性得分规则（proper scoring rules）、数学证明与Pareto最优分析。

**📊 数据集**

实验使用了SAT 2022成绩、FICO信用分以及多参数Beta分布的合成数据。

**📈 对比分析**

与基线及两类机制（MCV、PV）在多种分布上进行曲线比较，结果表明在确定性核查时MCV能显著降低偏差并减少核查率；在噪声核查下PV提供了可接受的偏差与核查折衷。

**⚠️ 局限性**

主要局限包括：在有限责任下无法消除偏差；噪声核查仍需接受负分数或较高最大惩罚；对类型分布的精确信息需求较高，若无分布假设只能使用近似。

---

## 451. You Can Learn Tokenization End-to-End with Reinforcement Learning

**arXiv ID:** 2602.13940 | [PDF](https://arxiv.org/pdf/2602.13940v1)

**作者:** Sam Dauncey `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**通讯引用:** 21230 | [OpenAlex ID](https://openalex.org/A5078339613)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在自回归 Transformer 内部学习 tokenization 过程，使用可变压缩率的 U‑Net 架构并通过奖励函数训练 token boundary 决策。

**💡 创新点**

创新点在于将 token boundary 视作离散决策，采用 score‑function（REINFORCE）估计并结合早期退出基线、时间折扣和批量优势归一化等 RL 方差缩减技术，从而在不引入额外结构假设的前提下实现端到端的 tokenization 学习。

**🔧 技术方法**

主要技术包括：1) 自回归 U‑Net 结构；2) 逐字 token boundary 的 sigmoid 策略；3) score‑function 估计与 REINFORCE；4) RL 相关方差降低方法（早期退出基线、时间折扣、批量归一化）；5) 目标下采样率约束损失；6) 早期退出模型与正则化。

**📊 数据集**

使用了经过筛选的 4096 字节长度的自然语言文本子集（主要是 Common Crawl 或类似公开数据集）以及 Python 代码数据集进行实验。

**📈 对比分析**

对比方法包括统一下采样基线、先前的 straight‑through 估计器（Nawrot 等）和 H‑Net；实验表明在 147M 参数模型下，score‑function 方法在验证交叉熵、下游 NLU 任务和代码 tokenization 上均优于对比方法，尤其在自然语言的语义边界识别和代码模块化方面表现更佳。

**⚠️ 局限性**

局限性：模型规模仅为 90–147M 参数，训练 compute 较低，无法验证在更大规模（>10²¹ 参数）下的效果；评估主要依赖交叉熵和少量下游任务，未充分检验对实际语言建模质量的长期影响；对极端稀有 token（glitch token）等现象的处理仍不充分。

---

## 452. MedScope: Incentivizing "Think with Videos" for Clinical Reasoning via Coarse-to-Fine Tool Calling

**arXiv ID:** 2602.13332 | [PDF](https://arxiv.org/pdf/2602.13332v1)

**作者:** Wenjie Li `[一作]` (Shanghai Jiao Tong University), Yankai Jiang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 322 | [OpenAlex ID](https://openalex.org/A5048516074)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了MedScope模型，支持在长视频中通过工具调用实现粗到细的证据检索与验证，实现“以视频思考”的临床推理。

**💡 创新点**

引入视觉链式推理(VCoT)与工具辅助的粗细检索框架；构建ClinVideoSuite数据集；提出基于证据对齐的GA‑GRPO强化学习，奖励依据时间对齐的证据。

**🔧 技术方法**

基于大语言模型(Qwen2.5‑VL‑7B‑Instruct)结合工具调用，三阶段训练（warm‑up、VCoT监督的SFT、GA‑GRPO），强化学习奖励设计与优势调节。

**📊 数据集**

使用ClinVideoSuite（包含ClinVideo‑Cap、QA、VCoT、CoT等子集），以及SVU‑31K、ClinVideo‑Eval（OphVL、MedVideoCap、SurgVidLM）等多任务评测集。

**📈 对比分析**

与多种开源/闭源多模态模型及工具使用代理对比，MedScope在SVU‑31K的全视频描述、细粒度描述、时间与感知推理等任务上均跑第一；在ClinVideo‑Eval的时间定位与基于视频的问答上亦领先，尤其在RL阶段显著提升mIoU与准确率。

**⚠️ 局限性**

依赖人工标注的证据监督和工具接口；模型在极端分布偏移或缺失工具时可能失效；目前仅用于研究，不可直接临床应用，需要进一步验证和鲁棒性评估。

---

## 453. High-Fidelity, Customizable Force Sensing for the Wearable Human-Robot Interface

**arXiv ID:** 2602.13436 | [PDF](https://arxiv.org/pdf/2602.13436v1)

**作者:** Noah Rubin `[一作]` (National Institutes of Health), Lillian Chin `[通讯]` (University of Texas)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并测试了通过3D打印硅胶垫嵌入气道（fluidic innervation）来测量人体-机器人接口力，验证其线性响应并在膝关节扭矩、二头肌卷曲和下蹲等多种任务中进行实验

**💡 创新点**

创新点在于将流体内化技术引入可定制、低成本且高SNR的接口传感，克服传统FSR非线性和EMG低SNR等局限，实现单通道压强测量

**🔧 技术方法**

采用3D打印软硅胶嵌入气道、差压传感器AllSensors 10in H2O、ESP32微控制器、I2C与蓝牙数据流、机械压缩机、临床测力计等硬件与线性回归、低通滤波等软件技术

**📊 数据集**

使用自制实验数据集：机械压缩实验、单人膝关节扭矩测量、二头肌卷曲负重0–4.54kg、未通电下蹲10次等，无公开数据集

**📈 对比分析**

通过与机械真值或测力计进行线性回归，R^2高达0.998、0.95和0.75，显示高SNR、快速响应，并在动态任务中能跟踪任务周期，优于传统FSR/EMG方法

**⚠️ 局限性**

局限包括单一气压通道难以获取表面力分布、粘弹性衰减导致压力缓冲、定位对测量影响显著、未在有动力机器人上验证、不同外部负载和关节角度对测量影响需进一步研究

---

## 454. Kalman Filtering Based Flight Management System Modeling for AAM Aircraft

**arXiv ID:** 2602.14948 | [PDF](https://arxiv.org/pdf/2602.14948v1)

**作者:** Balram Kandoria `[一作]` (SkyGrid), Aryaman Singh Samyal `[通讯]` (SkyGrid)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并验证了一种基于卡尔曼滤波器的Sigmoid混合测量噪声协方差的航班轨迹不确定性传播方法，用于AAM战略规划。

**💡 创新点**

采用Sigmoid函数平滑测量噪声协方差，实现连续的FMS校正行为模型，并能自适应控制输入，显著提升不确定性演化的真实性和可调节性。

**🔧 技术方法**

卡尔曼滤波、NURBS/PCHIP轨迹拟合、Sigmoid混合测量噪声协方差、实时参数调优。

**📊 数据集**

真实通用航空ADS‑B飞行数据（训练集与验证集）。

**📈 对比分析**

与传统线性不确定性模型和传统uLPA方法对比，在验证集上实现了76%的到达时间预测准确率，同时降低误报率并保持较低计算成本。

**⚠️ 局限性**

仅基于通用航空数据，尚未在专用AAM飞机上验证；对极端气象或突发扰动的鲁棒性有限；模型假设FMS校正行为单一，可能不适用于所有AAM架构。

---

## 455. PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement

**arXiv ID:** 2602.14968 | [PDF](https://arxiv.org/pdf/2602.14968v1)

**作者:** Yian Wang `[一作]` (University of Massachusetts Amherst), Chuang Gan `[通讯]` (University of Massachusetts Amherst)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种利用LLM代理与物理引擎相结合的框架PhyScensis，用于自动生成具有复杂物理交互的3D场景。

**💡 创新点**

创新点在于将物理约束与LLM生成的空间/物理谓词结合，构建可反馈、可自我修正的生成流程，并通过概率编程实现对场景稳定性的可控评估。

**🔧 技术方法**

核心技术包括大语言模型代理、基于物理引擎的约束求解器、占据格网启发式、概率编程评估稳定性、以及多模态反馈系统。

**📊 数据集**

使用BlenderKit构建的3D资产数据集，并通过ChatGPT进行属性标注；若缺失资产则采用文本到3D的生成管线。

**📈 对比分析**

与3D-Generalist、Architect、LayoutVLM等基线相比，在VQA分数、GPT排名、物理稳定性等指标上均取得显著提升，生成的场景更复杂、物理更真实，并在机器人演示学习实验中表现出更好的泛化能力。

**⚠️ 局限性**

局限性包括对高密度场景的求解时间仍较长、占据格网分辨率限制了连续物理评估精度，以及对某些高度复杂的物理约束仍依赖后续人工或更细粒度的调优。

---

## 456. From Scarcity to Scale: A Release-Level Analysis of the Pashto Common Voice Dataset

**arXiv ID:** 2602.14062 | [PDF](https://arxiv.org/pdf/2602.14062v1)

**作者:** Jandad Jahani `[一作]` (O.P. Jindal Global University), Jawid Ahmad Baktash `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对 Mozilla Common Voice 2025 年 24.0 版的普什图语子集进行发布级别的定量审计，分析其规模、验证进度、贡献者分布、年龄性别信息与句子重复度等结构特征。

**💡 创新点**

首次系统性量化低资源语言语料库的参与度不平衡、元数据缺失和验证瓶颈，并将其与多语言 Common Voice 其它语言进行横向对比，揭示规模增长与代表性不足之间的关系。

**🔧 技术方法**

使用 Python (pandas, numpy) 进行数据清洗与统计，计算 Gini 系数、Lorenz 曲线、累计覆盖率等指标；没有涉及模型训练，仅基于公开元数据进行结构分析。

**📊 数据集**

Mozilla Common Voice 普什图语 24.0 版数据集（总计 2,768.7 小时录音，975.89 小时已验证），共 2,407,799 句子、59,369 条独特句子，6,654 名说话者。

**📈 对比分析**

通过对比 14.0、20.0、24.0 版的总时长、验证时长、贡献者数量、Gini 系数等指标，发现验证率从 1.27 小时提升至 975.89 小时，Gini 系数高达 0.941，验证比例仅 35.2%。

**⚠️ 局限性**

主要局限在于：元数据（性别、年龄、领域）缺失导致子群体评估困难；验证吞吐量低，约 65% 录音未进入训练集；贡献者高度集中，语音多样性受限，未直接评估 ASR 性能。

---

## 457. Securing SIM-Assisted Wireless Networks via Quantum Reinforcement Learning

**arXiv ID:** 2602.13238 | [PDF](https://arxiv.org/pdf/2602.13238v1)

**作者:** Le-Hung Hoang `[一作]` (VinUniversity), Van-Dinh Nguyen `[通讯]` (Trinity College Dublin)

**通讯引用:** 4236 | [OpenAlex ID](https://openalex.org/A5037305946)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在多用户 MISO 系统中，联合优化 SIM 阶段波束成形与发射功率，实现物理层安全加密。

**💡 创新点**

首次将量子近似策略优化（Q-PPO）引入 SIM 控制，利用量子电路提升高维连续动作空间的学习效率与收敛速度。

**🔧 技术方法**

采用混合量子-经典策略网络，将参数化量子电路嵌入演员网络；同时使用经典深度网络进行预编码与后处理。

**📊 数据集**

使用基于仿真的无线网络与 SIM 参数的合成数据；无真实数据集。

**📈 对比分析**

与 PPO、DDPG、TD3 与随机配置进行对比；Q-PPO 在平均保密速率上提升约15%，收敛速度提高约30%，并保持更高公平性。

**⚠️ 局限性**

受限于当前量子硬件的量子比特数与深度，算法对大规模 SIM 仍有计算和资源瓶颈；实验仅在模拟环境中验证，缺乏真实硬件验证。

---

## 458. Adaptive Autoguidance for Item-Side Fairness in Diffusion Recommender Systems

**arXiv ID:** 2602.14706 | [PDF](https://arxiv.org/pdf/2602.14706v1)

**作者:** Zihan Li `[一作]` (Johannes Kepler University Linz), Markus Schedl `[通讯]` (Johannes Kepler University Linz and Linz Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种自适应自引导扩散推荐系统（A2G‑DiffRec），通过学习动态加权主模型与弱模型的输出，并结合统一的流行度正则化，实现了对商品侧公平性的提升。

**💡 创新点**

创新点在于：①将自引导（autoguidance）机制从图像生成迁移至推荐系统，并用自适应引导网络（AAN）学习每一步的加权；②提出统一的流行度正则化，既抑制热门商品过度曝光，又鼓励长尾商品的展示；③在实验中验证了该方法在保持准确率的同时显著提升公平性。

**🔧 技术方法**

技术方法包括：扩散模型（Diffusion Recommender）、自引导机制、深度多层感知机（MLP）预测加权系数、流行度分桶正则化、联合损失优化。

**📊 数据集**

实验数据集：MovieLens‑1M、Foursquare‑Tokyo、Music4All‑Onion。

**📈 对比分析**

与随机、最热、LightGCN、MultiVAE、DiffRec、C‑DiffRec、CFG‑DiffRec 等基线比较。A2G‑DiffRec 在三大数据集上均实现了比 DiffRec 更好的商品侧公平性（如 ΔExp、Gini、APLT 等指标）且准确率下降有限，尤其在 ML1M 与 Onion 上表现突出；AG‑DiffRec 在 ML1M 上公平性提升明显但准确率下降大；在 FTKY 上两种方法均表现相对弱。

**⚠️ 局限性**

局限性包括：对不同数据集的适应性不一致，尤其在 FTKY 上公平性提升有限；目前仅在隐式反馈场景下验证，未探讨显式反馈；弱引导模型的构造仅采用训练早期检查点，缺乏更丰富的退化策略。

---

## 459. Energy-Efficient Over-the-Air Federated Learning via Pinching Antenna Systems

**arXiv ID:** 2602.14250 | [PDF](https://arxiv.org/pdf/2602.14250v1)

**作者:** Saba Asaad `[一作]` (University of Toronto), Ali Bereyhi `[通讯]` (University of Toronto)

**通讯引用:** 308 | [OpenAlex ID](https://openalex.org/A5061064331)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了基于PINCHING Antenna System（PASS）的服务器端传输技术，用于能量高效的Over-the-Air Federated Learning（OTA‑FL）聚合；

**💡 创新点**

通过PASS的波导和可调pinching元件实现大规模路径损耗补偿，形成低成本、低功耗的多用户叠加计算；结合分层算法（PASS调节、功率缩放、设备调度）实现全局能量最小化的联合优化；

**🔧 技术方法**

混合信道建模、过空气计算（AirComp）、计算速率与MSE关联的能量模型、Dinkelbach方法、分层投影与迭代算法、数值仿真；

**📊 数据集**

MNIST（四层卷积网络）和两层MLP（用于区域尺度实验），分别在K=8个边缘设备上分布式训练；

**📈 对比分析**

与完美链路的FedAvg、单天线MIMO、以及多天线MIMO（M=8,16,32）进行对比；PASS在能量消耗上比单天线MIMO低数十倍，能量/功率需求大幅下降，且在不同区域尺度和功率设置下保持稳定的学习精度；

**⚠️ 局限性**

主要局限在于仅研究单天线PASS服务器，未考虑多天线PASS或MIMO‑PASS复合结构；实验覆盖范围有限，未验证更大规模设备或异构网络的可扩展性；

---

## 460. HLE-Verified: A Systematic Verification and Structured Revision of Humanity's Last Exam

**arXiv ID:** 2602.13964 | [PDF](https://arxiv.org/pdf/2602.13964v1)

**作者:** Weiqi Zhai `[一作]` (Alibaba Group), Bing Zhao `[通讯]` (Alibaba Group)

**通讯引用:** 6396 | [OpenAlex ID](https://openalex.org/A5100358009)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了HLE-Verified，一个经过二阶段验证与修订的HLE基准，去除了噪声并提供不确定项集。

**💡 创新点**

提出了透明的组件级验证协议和细粒度错误分类，首次实现基准持续迭代与修订流程。

**🔧 技术方法**

结合领域专家评审、模型跨检查与多模型重试，使用双重专家修订与最终裁决实现错误定位与修复。

**📊 数据集**

在原始HLE 2500条样本上进行验证与修订，最终得到641条金标、1170条修订、689条不确定集。

**📈 对比分析**

对七大前沿LLM在原始HLE与HLE-Verified进行准确率与校准误差比较，验证集平均提升7–10个百分点，错误项提升30–40个百分点，显示修订显著提高测评可靠性。

**⚠️ 局限性**

仍存在689条不确定条目，需社区进一步专家介入，且验证重点集中在可复制领域，对主观或高度依赖外部知识的问题仍有限。

---

## 461. Toward Resource-Efficient Collaboration of Large AI Models in Mobile Edge Networks

**arXiv ID:** 2602.13206 | [PDF](https://arxiv.org/pdf/2602.13206v1)

**作者:** Peichun Li `[一作]` (University of Macau), Yuan Wu `[通讯]` (University of Macau)

**通讯引用:** 16504 | [OpenAlex ID](https://openalex.org/A5076943390)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了移动边缘网络中大规模 AI 模型协同执行的系统框架，并基于此设计了多阶段扩散框架以提升生成任务的效率和适应性。

**💡 创新点**

将协同 AI 划分为空间与时间两类，并提出多阶段扩散、设备选择与分层同步的混合协同方案，兼顾通信、计算与能耗，首次在移动边缘场景中系统性地整合多种协同技术。

**🔧 技术方法**

核心技术包括：空间协同（联邦调优、专家混合、补丁扩散、层级扩散），时间协同（拆分学习、级联推理、投机解码、路由推理），以及多阶段扩散与设备筛选算法；实现上使用离线动态规划进行资源分配。

**📊 数据集**

实验基于 Stable Diffusion v1.5，采用 32/16/8/4 位量化模型，在 Jetson AGX Xavier 与 Jetson Orin 边缘设备上仿真，随机部署于 500×500 m² 区域。

**📈 对比分析**

与传统拆分扩散做对比，使用 MS-SSIM 评估图像质量，计算/通信时延比例作为性能指标。实验表明，在相同能耗/时延约束下，多阶段扩散的 MS-SSIM 提升 2–5%，通信占比下降；在更宽松约束下，性能进一步提升。

**⚠️ 局限性**

限制包括：需手工设计分阶段划分与路由策略、对设备可用性变化的鲁棒性不足、未考虑隐私与安全问题、能耗优化仅基于离线 DP，缺乏在线自适应机制。

---

## 462. Sovereign Agents: Towards Infrastructural Sovereignty and Diffused Accountability in Decentralized AI

**arXiv ID:** 2602.14951 | [PDF](https://arxiv.org/pdf/2602.14951v1)

**作者:** Botao Amber Hu `[一作]` (University of Oxford), Helena Rong `[通讯]` (New York University Shanghai)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了“基础设施主权”框架，分析去中心化 AI 代理的治理挑战并提出研究方向。

**💡 创新点**

引入了基础设施硬度、代理主权等新概念，揭示治理责任在多层技术堆栈中的扩散。

**🔧 技术方法**

基于区块链、可信执行环境（TEE）、去中心化物理基础设施网络（DePIN）等技术进行理论建模。

**📊 数据集**

未使用传统数据集，主要以案例分析（Spore.fun、Truth Terminal、Freysa AI）为依据。

**📈 对比分析**

由于是概念性研究，未进行实验对比；通过案例讨论说明治理困境和潜在方案。

**⚠️ 局限性**

缺乏实证验证与可量化指标，研究侧重理论探讨，未给出具体治理工具或实现细节。

---

## 463. Evaluation of Dynamic Vector Bin Packing for Virtual Machine Placement

**arXiv ID:** 2602.14704 | [PDF](https://arxiv.org/pdf/2602.14704v1)

**作者:** Zong Yu Lee `[一作]` (Nanyang Technological University), Xueyan Tang `[通讯]` (Nanyang Technological University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在云计算虚拟机放置问题中，将其建模为最小使用时间的动态向量箱装（MinUsageTime DVBP）问题，系统评估并对比了多种在线算法，包括非clairvoyant、clairvoyant及学习增强（learning-augmented）三种设置下的算法，并提出了若干改进算法。

**💡 创新点**

创新点在于：①设计了Round Robin Next Fit、Nearest Remaining Time（优先版）、Reduced Hybrid以及去除大箱的RCP/PPE等新算法，并对其竞争比进行了理论分析；②提出Any Fit特性在实践中的重要性，并验证Prioritized NRT在clairvoyant场景下的近最优性能；③在学习增强场景下，改进的PPE在大误差下优于原RCP，并且Greedy、Lifetime Alignment在不同误差水平下表现更稳健。

**🔧 技术方法**

采用的技术包括：在线装箱算法、竞争比分析、ℓ∞范数的向量容量度量、预测误差建模（对预测误差采用对数正态分布），以及基于真实工作负载的经验评估框架，利用箱装时间总和与理论下界的比值作为性能度量。

**📊 数据集**

使用数据集：Microsoft Azure公开工作负载（14天、约550万请求、35种PM类型，5维资源），以及规模较小的Huawei Cloud（116k请求、2维资源）。

**📈 对比分析**

比较方法：对每个实例计算算法产生的总箱使用时间与理论下界的比值（performance ratio），使用箱形图展示不同算法在所有实例上的分布。实验结果表明：在非clairvoyant场景中First Fit表现最佳；在clairvoyant场景中Prioritized NRT优于其他算法；在学习增强场景中，改进后的PPE、Greedy、Lifetime Alignment（Binary）在误差增大时表现最稳健，误差接近完美时Prioritized NRT最佳；整体来看Any Fit特性和对离开时间的利用是关键。

**⚠️ 局限性**

局限性包括：①多维情形下的竞争比理论仍未与单维最优对齐；②实验仅基于Azure与Huawei两个数据集，缺乏对更广泛工作负载的验证；③未提出自适应机制以根据预测误差动态切换算法；④使用下界而非精确最优解作为基准，导致性能评价的上限不完全精确。

---

## 464. AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge

**arXiv ID:** 2602.13476 | [PDF](https://arxiv.org/pdf/2602.13476v1)

**作者:** Noriaki Hirose `[一作]` (University of California, Berkeley), Sergey Levine `[通讯]` (University of California, Berkeley)

**通讯引用:** 62736 | [OpenAlex ID](https://openalex.org/A5026322200)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计 AsyncVLA 框架，将大规模视觉‑语言‑动作模型在远端运行，轻量化的 Edge Adapter 在机器人边缘实时执行动作，支持异步控制。

**💡 创新点**

通过异步分离高层语义推理与低层快速执行，端到端对齐两层模型，并使用动作重加权提升对动态障碍的反应，从而在高达 6 秒的通信延迟下保持稳定性能。

**🔧 技术方法**

结合 OmniVLA 大模型、ViT+Transformer 轻量化 Action Head（76M）、动作令牌投影、端到端微调、动态轨迹加权、PD 控制及 WiFi 通信等技术。

**📊 数据集**

使用 GNM（六大公开数据集）、LeLaN、SACSoN（包含行人）等混合数据集进行训练，以增强动态交互。

**📈 对比分析**

与 OmniVLA、OmniVLA‑edge、无 E2E 训练版等基线比较，在姿态和语言导航任务中，成功率提升约 40%，碰撞率降低，语言跟随成功率与 OmniVLA 接近，且在高延迟（至 6 s）下表现优于所有基线。

**⚠️ 局限性**

受限于需要可微调的开源基础模型、动态交互样本有限以及端到端训练对算力的高需求，未来需进一步解耦两层模型并减少训练成本。

---

## 465. Towards Sparse Video Understanding and Reasoning

**arXiv ID:** 2602.13602 | [PDF](https://arxiv.org/pdf/2602.13602v1)

**作者:** Chenwei Xu `[一作]` (Northwestern University), Han Liu `[通讯]` (Northwestern University)

**通讯引用:** 14390 | [OpenAlex ID](https://openalex.org/A5100349032)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种多轮视频问答框架（Reasoning with Video Sparsity），通过逐步挑选少量信息量大的关键帧并维护“summary-as-state”来回答问题。

**💡 创新点**

创新点在于：①引入结构化的持续摘要状态（P/O/H/U/R）以避免信息过载并提升关键证据感知；②设计无标注的稀疏奖励 EAGER，用于强化学习微调；③实现“plug‑and‑play”与多轮交互兼容的开放式框架。

**🔧 技术方法**

使用的技术包括：多轮控制器、结构化响应格式、视觉语言模型（VLM）嵌入、GRPO 强化学习、EAGER 奖励设计、对齐推理和早停机制。

**📊 数据集**

在 VideoEspresso、NExT‑QA、EgoSchema 三个视频问答基准上进行评估。

**📈 对比分析**

与现有专有与开源 VLM（如 GPT‑4o、Qwen‑2.5‑VL、InternVL2）以及基线方法相比，本文在相同或更低的帧数、轮数和提示代价下，显著提升准确率（例如 VideoEspresso 上平均提升 9–22%），并在多轮强化学习后进一步提高效率。

**⚠️ 局限性**

局限性包括：对底层 VLM 的视觉和时序推理能力高度依赖；多轮交互导致额外的 API 调用和潜在延迟；在低质量或复杂视频上，摘要状态可能不足以捕捉全部关键信息。

---

## 466. Freq-DP Net: A Dual-Branch Network for Fence Removal using Dual-Pixel and Fourier Priors

**arXiv ID:** 2602.14226 | [PDF](https://arxiv.org/pdf/2602.14226v1)

**作者:** Kunal Swami `[一作]` (Samsung Research India), Chandra Sekhar Seelamantula `[通讯]` (Indian Institute of Science)

**通讯引用:** 2405 | [OpenAlex ID](https://openalex.org/A5005652970)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于双像素传感器的单张图片篱笆去除方法，利用深度视差和频域结构先验实现精确分割与重建。

**💡 创新点**

首次将双像素视差与FFT卷积结构先验结合，构建双分支网络并通过注意力融合，突破了仅依赖多帧或外观特征的局限。

**🔧 技术方法**

双像素成本体积、Fast Fourier Convolution、结构注意力模块以及基于LaMa的两阶段去除网络。

**📊 数据集**

发布包含904个合成样本（804/100）和150个真实拍摄场景的多样化篱笆去除基准数据集。

**📈 对比分析**

与RGB U-Net、DP U-Net、Restormer、LaMa等基线对比，分割F1达到0.971，去除PSNR在合成/真实上分别达到41.28/34.54，显著优于现有方法。

**⚠️ 局限性**

对篱笆距离越远视差信号越弱，且单帧重建仍需生成填补，无法完全匹配视频帧重建的真实像素。

---

## 467. Faster Parameterized Vertex Multicut

**arXiv ID:** 2602.13981 | [PDF](https://arxiv.org/pdf/2602.13981v1)

**作者:** Huairui Chu `[一作]` (University of California, Santa Barbara), Mingyu Xiao `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1477 | [OpenAlex ID](https://openalex.org/A5033729619)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的算法来解决无向图中的顶点多切割问题，运行时间为O^*(k^O(k))。

**💡 创新点**

通过改进阴影去除步骤，显著提高了算法的效率，特别是在处理有向子集反馈顶点集和有向多路切割问题时。

**🔧 技术方法**

使用了阴影去除技术和迭代压缩技术，结合了参数化算法的框架。

**📊 数据集**

使用了多个图实例，特别是无向图中的顶点多切割问题的实例。

**📈 对比分析**

与之前的算法相比，新的算法在运行时间上有显著改进，特别是相较于Chitnis等人的O^*(k^O(k^2))时间算法，新的算法达到了O^*(k^O(k))。

**⚠️ 局限性**

算法的局限性在于它仍然依赖于参数k的大小，且在处理某些特定类型的图时可能会遇到性能瓶颈。

---

## 468. How to Train Your Filter: Should You Learn, Stack or Adapt?

**arXiv ID:** 2602.13484 | [PDF](https://arxiv.org/pdf/2602.13484v1)

**作者:** Diandre Miguel Sabale `[一作]`, Prashant Pandey `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究针对现代过滤器三大范式（学习型、堆叠型、适应型）进行统一、系统的性能对比实验，评估其在多种真实数据集与查询工作负载下的误报率、空间效率、构造与查询时延，以及对动态/对抗性工作负载的鲁棒性。

**💡 创新点**

创新点在于：①首次将学习型、堆叠型、适应型三种过滤器在相同实验框架下直接比较；②从构造时间、查询延迟、模型训练成本等角度全面衡量性能；③探究学习模型配置、负载分配等参数对误报率的影响；④给出基于工作负载特征的过滤器选择建议。

**🔧 技术方法**

使用的技术包括：学习型过滤器（Ada-BF、Fast PLBF，配合逻辑回归/决策树/随机森林模型）；堆叠型过滤器（多层Bloom/Quotient/Cuckoo实现）；适应型过滤器（AdaptiveQF）；对不同查询分布（一次性、均匀、Zipfian、对抗性）进行模拟；对动态数据集做周期性更换与重建；测量构造时间（含模型训练）与查询时延。

**📊 数据集**

采用四个真实数据集：Malicious URL（约0.16M键，0.05M正例）、Ember（0.8M键，0.4M正例）、Shalla（3.9M键，2.9M正例）和Caida（8.5M键，1.2M正例）。

**📈 对比分析**

比较方法：在相同总空间预算下，分别构造三类过滤器；对统一的查询集（10M次）在不同分布下计算误报率；在动态更换负载时记录即时误报率；统计构造与查询耗时。结果显示：学习型过滤器在训练分布匹配时可将误报率降低至1/100，但查询延迟可达传统过滤器的10⁴倍；堆叠型过滤器在Zipfian或已知负载下误报率可降至1/1000，查询延迟仅3倍左右；适应型过滤器在任何工作负载下误报率始终保持在1/1000以下，且查询时延最快。

**⚠️ 局限性**

局限性包括：学习型过滤器高度依赖模型质量与训练分布，面对分布漂移或对抗性查询时误报率波动大；模型训练与推理成本高，难以满足高吞吐量需求；堆叠型过滤器需事先采样负载且对负载变化不适应；适应型过滤器需要反向映射存储，可能导致额外磁盘I/O；整体工作量对实现细节敏感，实际部署时需考虑硬件与语言差异。

---

## 469. Floe: Federated Specialization for Real-Time LLM-SLM Inference

**arXiv ID:** 2602.14302 | [PDF](https://arxiv.org/pdf/2602.14302v1)

**作者:** Chunlin Tian `[一作]`, Chengzhong Xu `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个联邦学习框架，将云端黑盒LLM与边缘端轻量级SLM协同工作，以实现低延迟、隐私保护的实时推理。

**💡 创新点**

创新点包括：①异构感知的LoRA适配策略，可根据设备内存和延迟约束动态调整低秩矩阵维度；②任务特定的聚合与路由机制，利用聚类和参数无关的MoE路由实现混合任务的高效分配；③基于logit级别的LLM–SLM融合，允许在不暴露权重的前提下实现知识蒸馏。

**🔧 技术方法**

使用技术主要有：联邦学习（FedAvg/FedProto/ FedMoE）、LoRA参数高效微调、异构感知自适应rank、任务聚类与参数无关MoE路由、局部差分隐私、logit融合与隐私检测器。

**📊 数据集**

实验使用的数据集包括CoGenesis（隐私敏感交互数据）、Flan‑v2（多任务基准）、WikiText‑2（推理效率评估）、BBH（BigBench Hard）等。

**📈 对比分析**

与传统的SLM单独微调、FedAvg、FedProto、LLM单独推理等基线相比，该框架在BBH多任务上平均提升约14.2%（相较于小模型）且3.6%（相较于LLM基线），在推理延迟和能耗上显著低于压缩模型和大模型，且在云端LLM网络延迟变动时仍保持稳定的实时响应。

**⚠️ 局限性**

主要限制包括：①隐私检测器为启发式方法，召回率不足可能导致敏感信息泄漏；②logit融合要求LLM公开词表，完全黑盒API下不可行；③异构感知LoRA在硬件受限时可能低秩导致优质本地数据贡献受限。

---

## 470. Optimizing Point-of-Care Ultrasound Video Acquisition for Probabilistic Multi-Task Heart Failure Detection

**arXiv ID:** 2602.13658 | [PDF](https://arxiv.org/pdf/2602.13658v1)

**作者:** Armin Saadat `[一作]` (University of British Columbia), Purang Abolmaesumi `[通讯]` (Vancouver General Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了 Double-Precise 框架，利用强化学习在点门超声视频采集过程中动态选择视图，支持心衰多任务诊断（主动瓣膜狭窄严重程度分类与左室射血分数回归）。

**💡 创新点**

创新点在于将视频采集视为主动决策问题，结合多视图 Transformer 及联合概率头，使用稀疏/密集奖励的 PPO 训练，实现在患者个性化的预算约束下保持诊断性能同时减少视频数量。

**🔧 技术方法**

使用 EchoPrime 视频编码器、Transformer 编码器、联合多变量高斯头以及 Proximal Policy Optimization（PPO）强化学习；同时采用基于 Jensen–Shannon 散度的密集奖励来更新分布。

**📊 数据集**

在 12,180 名患者的多视图超声数据库（5 个视图）上进行训练、验证和测试，采集设备包括 Philips iE33 与 GE Vivid 系列。

**📈 对比分析**

与随机选择和固定人群视图选择等基线比较，RL 策略在不同视频采集预算下实现均衡准确率约 77.2%，仅使用约 3.4 个视频即可匹配全视图性能，节省约 32% 视频量，在低预算下保持更高准确率。

**⚠️ 局限性**

局限性在于实验为回顾性预先获取的数据，未在真实现场 POCUS 流程中验证策略，且在极低预算时可能导致过早终止而影响诊断。

---

## 471. Diagnostic Benchmarks for Invariant Learning Dynamics: Empirical Validation of the Eidos Architecture

**arXiv ID:** 2602.13322 | [PDF](https://arxiv.org/pdf/2602.13322v1)

**作者:** Datorien L. Anderson `[一作]` `[通讯]` (Occybyte), Datorien L. Anderson (Occybyte)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并验证了一个名为PolyShapes-Ideal（PSI）的诊断基准，评估了Eidos架构在无纹理干扰下的拓扑不变性学习能力，并通过多边形分类、零样本字体迁移和几何崩塌映射等实验进行验证。

**💡 创新点**

通过设计去除纹理相关的PSI数据集，提出并验证“Form-First”假设，展示离散承诺架构在无需大规模统计样本的情况下实现拓扑不变性，并揭示学习过程中的相位转移现象。

**🔧 技术方法**

采用离散承诺的Eidos网络、三阶段训练策略、零样本迁移评估以及逐步仿射变形的几何崩塌测试等技术。

**📊 数据集**

使用PSI（多边形噪声分类）、PSIP（三维多面体投影）、MNIST以及30种未见字体的字体集合。

**📈 对比分析**

与传统卷积网络对比，Eidos在PSI上达成>99%准确率，在零样本字体迁移中实现81.67%平均准确率，并在不同字体族上保持高精度，显示出对拓扑的更强鲁棒性。

**⚠️ 局限性**

对开环数字和极端字体仍易崩塌；需要显式的边界样本来触发拓扑学习；对更复杂形状和更高维几何的泛化能力尚未充分验证。

---

## 472. OneLatent: Single-Token Compression for Visual Latent Reasoning

**arXiv ID:** 2602.13738 | [PDF](https://arxiv.org/pdf/2602.13738v1)

**作者:** Bo Lv `[一作]` (Tsinghua University), Haoxiang Shi `[通讯]` (Waseda University)

**通讯引用:** 258 | [OpenAlex ID](https://openalex.org/A5020364411)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 OneLatent 框架，将传统链式推理压缩为单一潜在标记，显著减少推理输出长度。

**💡 创新点**

创新点在于使用可视化潜在标记与 DeepSeek‑OCR 隐藏状态对齐，并通过渲染 CoT 为图像的方式提供可检验的监督，从而实现基于 MDL 的压缩。

**🔧 技术方法**

技术包括三阶段训练（CoT 冷启动 → 潜在对齐 → 仅答复微调）、视觉隐藏状态监督、OCR 质量检查、固定长度潜在标记、以及 SAM‑ViT‑B + CLIP‑L 视觉编码器。

**📊 数据集**

使用的数据集有 GSM8K、GSM8K‑Hard、SVAMP、ProntoQA、ProsQA 及其增强版（ProntoQA Enhanced、ProsQA Enhanced）。

**📈 对比分析**

与 No CoT、iCoT、CoT、COCONUT 等方法对比，平均输出长度从 74.6 tokens 降至 6.8 tokens，准确率仅下降 2.21%，在长链推理任务上达 99.8%（ProntoQA）和 97.8%（ProsQA）；OTC 提升至 7.77，压缩比最高可达 87.4×。

**⚠️ 局限性**

主要限制包括：依赖精确 CoT 渲染与 OCR 质量；对不同视觉编码器或非文本领域的泛化受限；潜在标记降低了模型推理过程的可解释性。

---

## 473. Future of Edge AI in biodiversity monitoring

**arXiv ID:** 2602.13496 | [PDF](https://arxiv.org/pdf/2602.13496v1)

**作者:** Aude Vuilliomenet `[一作]` (University College London), Duncan Wilson `[通讯]` (University College London)

**通讯引用:** 15682 | [OpenAlex ID](https://openalex.org/A5047854292)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述了2017-2025年间在生物多样性监测中应用边缘AI的82项研究，分析硬件平台、模型优化与通信技术，并提出四种典型架构。

**💡 创新点**

首次将边缘AI系统按硬件、模型与通信三个耦合维度进行统一分类，揭示了技术与生态目标之间的权衡与发展趋势。

**🔧 技术方法**

聚焦边缘计算、人工智能模型（深度学习/机器学习）、传感器（声学、视觉、定位）、无线通信（LoRaWAN、WiFi、Cellular、卫星）等技术。

**📊 数据集**

综述引用了多种公开数据集（如BirdNET、MegaDetector、BatDetect2等）以及各研究场景采集的现场音频、图像和轨迹数据。

**📈 对比分析**

通过量化指标（年发表量、硬件功耗、模型大小、推理时间、能耗、部署持续时间）对比四类架构，指出SBC型在多物种识别上性能最优，而MCU型在能耗和长期部署方面表现突出。

**⚠️ 局限性**

缺乏统一的性能评估与能耗报告，模型漂移和长期现场表现不足，以及数据保留策略与伦理隐私风险等未得到充分讨论。

---

## 474. MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design

**arXiv ID:** 2602.14926 | [PDF](https://arxiv.org/pdf/2602.14926v1)

**作者:** Gen Zhou `[一作]` (Western University), Pingzhao Hu `[通讯]` (Western University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了一套闭环多智能体协作系统MAC-AMP，用于多目标抗菌肽（AMP）生成与优化，能自适应评估、反馈和奖励，完成从目标细菌到新肽序列的全流程；

**💡 创新点**

创新点在于将AMP设计转化为多智能体协作问题，引入AI模拟同行评审生成可执行的奖励信号，实现闭环可解释的多目标优化，并支持跨领域迁移；

**🔧 技术方法**

采用LLM多智能体、AI模拟同行评审、强化学习（PPO）奖励自适应、专业评估工具（Mic预测、ToxinPred、OmegaFold、Macrel、Foldseek等）以及GPT-2生成模型；

**📊 数据集**

使用公开AMP数据库DBAASP v3和dbAMP 3.0提供的多种细菌靶点（E. coli、S. aureus、P. aeruginosa、K. pneumoniae、E. faecium）数据集；

**📈 对比分析**

与AMP-Designer、BroadAMP GPT、PepGAN、Diff-AMP四个基线及真实AMP数据集进行比较，MAC-AMP在抗菌活性、毒性、结构可靠性等多项指标上均优于基线，且在广谱活性测试中展现最强的跨菌种泛化；

**⚠️ 局限性**

局限性包括对大规模多样化数据的依赖、可能的奖励设计偏差、对极端毒性或不常见结构的预测不够完善，以及当前仅在单一实验室环境下验证，缺乏更广泛的实验验证与临床前评估。

---

## 475. Breaking Data Efficiency Dilemma: A Federated and Augmented Learning Framework For Alzheimer's Disease Detection via Speech

**arXiv ID:** 2602.14655 | [PDF](https://arxiv.org/pdf/2602.14655v1)

**作者:** Xiao Wei `[一作]` (Tianjin University), Jianwu Dang `[通讯]` (Shenzhen Institutes of Advanced Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

通过联邦学习与基于声码转换的数据增强结合，提出FAL-AD框架，实现阿尔茨海默病早期诊断的多模态模型训练。

**💡 创新点**

创新点在于跨类别声码转换产生病理语音增强样本，适配联邦学习的自适应模型选择，以及细粒度词级对齐的注意力跨模态融合。

**🔧 技术方法**

使用声码转换（CosyVoice 2.0B）、联邦学习（FedAvg+自适应模型选择）、Whisper ASR、Wav2Vec2与DistilBERT提取特征，结合Transformer的跨模态注意力。

**📊 数据集**

采用ADReSSo数据集，包含237名受试者的自发语音。

**📈 对比分析**

与集中式、局部式及标准联邦学习进行对比，在5折交叉验证下，FL+Aug在多模态上取得91.52%准确率，超过集中式基线90.36%。

**⚠️ 局限性**

局限在于数据量仍有限，声码转换可能引入人工特征噪声，且在更大规模异构多中心数据上的可推广性待验证。

---

## 476. On the Stability of Nonlinear Dynamics in GD and SGD: Beyond Quadratic Potentials

**arXiv ID:** 2602.14789 | [PDF](https://arxiv.org/pdf/2602.14789v1)

**作者:** Rotem Mulayoff `[一作]` (CISPA Helmholtz Center for Information Security), Sebastian U. Stich `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文对梯度下降（GD）和随机梯度下降（SGD）在接近最小值时的非线性动力学进行分析，给出了稳定振荡的精确判据，并揭示了SGD稳定性受单个批次影响的现象。

**💡 创新点**

①提出了多变量情况下GD稳定振荡的完整高阶导数条件，纠正并扩展了以往单变量结果；②证明即使单个批次在非线性层面不稳定，SGD的期望动态也会不稳定；③给出若所有批次线性稳定，则SGD非线性动力学在期望上稳定的充分条件。

**🔧 技术方法**

使用动力系统的翻转分岔（flip bifurcation）理论、Lyapunov系数、Koopman理论以及无穷维Hilbert空间的矩阵动力学来分析非线性稳定性。

**📊 数据集**

未使用具体数据集；研究基于理论分析和数学推导。

**📈 对比分析**

未进行实验或数值比较，本文主要通过理论证明和数学推导来展示结果。

**⚠️ 局限性**

仅针对单峰、单临界方向的最小值；不适用于低维流形或多重临界方向的情形；仅对插值最小值的SGD有效，对非插值最小值的稳定性未给出完整定义。

---

## 477. QuaRK: A Quantum Reservoir Kernel for Time Series Learning

**arXiv ID:** 2602.13531 | [PDF](https://arxiv.org/pdf/2602.13531v1)

**作者:** Abdallah Aaraba `[一作]` (Universite de Sherbrook), Shengrui Wang `[通讯]` (Universite de Sherbrook)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了 QuaRK，一个端到端的量子储备核学习框架，将硬件友好的量子特征化器与基于 RKHS 的核回归读出器结合，用于时间序列预测，并给出了弱相关数据的 PAC 学习理论保证。

**💡 创新点**

创新点包括：① 通过 Johnson–Lindenstrauss 投影将量子资源与输入维度解耦；② 采用空间多路复用和经典影子测量 k‑local Pauli 观测来提升表达能力；③ 将量子特征映射与 Matérn 核结合，提供可解释的正则化；④ 在 β‑mixing 时间序列上给出泛化误差上界，连接设计与样本效能。

**🔧 技术方法**

使用的技术：量子储备计算（合同编码通道）、Johnson–Lindenstrauss 随机投影、空间多路复用、经典影子量子测量、Matérn 核核岭回归、Rademacher 复杂度分析和 β‑mixing 依赖性处理。

**📊 数据集**

实验数据集：合成 β‑mixing 的 VARMA 时间序列，包含三类目标函数（单步预测、指数衰减线性、Volterra 二阶交互）。

**📈 对比分析**

比较方法：在训练集上绘制 MSE 随正则化强度和训练窗口数变化的曲线，验证插值区间和 1/√N 的测试误差下降；结果显示训练误差可降至零，测试误差随样本增大按理论预期下降，且不同任务难度对应不同误差水平。

**⚠️ 局限性**

局限性：假设无噪声理想量子电路和无限测量精度；仅在合成数据上验证；对量子资源分配、观测局部性 k、空间多路复用数和核超参数的进一步优化与噪声鲁棒性研究仍待完成。

---

## 478. Train Less, Learn More: Adaptive Efficient Rollout Optimization for Group-Based Reinforcement Learning

**arXiv ID:** 2602.14338 | [PDF](https://arxiv.org/pdf/2602.14338v1)

**作者:** Zhi Zhang `[一作]` (University of California, Los Angeles), Huzefa Rangwala `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种名为AERO的自适应高效推演优化框架，旨在解决GRPO在LLM对齐训练中出现的零优势（zero‑advantage）与梯度死区问题，显著提升计算与时间效率。

**💡 创新点**

创新点包括：
• 采用两阶段自适应推演分配，动态分配预算给难题；
• 引入Beta–Binomial后验估计，避免零优势导致的梯度消失；
• 以理论分析为指导的拒绝采样（k=1）下的正负样本比例平衡，提升梯度信号。

**🔧 技术方法**

技术手段主要是GRPO算法的改进、贝叶斯后验估计、拒绝采样与迭代救援策略；实现时使用NVIDIA A10/A100 GPU集群训练Qwen系列LLM。

**📊 数据集**

数据集：Math领域采用OpenR1-AERO（约4万题）以及AIME、AMC、MATH、Minerva、Olympiad等验证集；代码领域使用APPS、CodeContests、TACO等。

**📈 对比分析**

与GRPO、DAPO等基线在相同总推演预算下对比。AERO在1.5B模型上训练计算量下降约48%，时间缩短约45%，在7B模型上下降约47%/49%；在测试集上的Avg@8/Pass@8均高于基线，表明在不牺牲性能的前提下实现显著效率提升。

**⚠️ 局限性**

局限性：
• 仍需手动调节阈值S、后验先验参数和k比例；
• 主要针对RLVR任务，尚未在更广泛的对话或生成场景中验证；
• 对大规模模型的推演预算分配依赖经验，可能在极端难度场景下效果不如预期。

---

## 479. Simpler Than You Think: The Practical Dynamics of Ranked Choice Voting

**arXiv ID:** 2602.14329 | [PDF](https://arxiv.org/pdf/2602.14329v1)

**作者:** Sanyukta Deshpande `[一作]` (University of Illinois at Urbana-Champaign), Sheldon H. Jacobson `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过大规模实证研究和算法优化，系统分析了110场真实选举（纽约市2021年初选、阿拉斯加2024年州级选举以及波特兰2024年市议会多胜选举），评估了排名选择投票（RCV）在竞争性、选票枯竭、战略复杂度和结果透明度方面的实际表现。

**💡 创新点**

创新点在于提出并实现了“Enhanced RCV Strategic Framework (ERSF)”——一种结合候选人消除、可追踪阈值和稳健选票补全的高效算法，能够在原本NP‑hard的排名投票分析中快速计算胜利间隙、最优加票策略和选票枯竭敏感性；同时将排名投票的复杂动态转化为可与多数制对比的直观指标。

**🔧 技术方法**

使用的技术主要包括：1) 候选人消除与可追踪阈值优化的组合算法；2) 对投票补全（robust ballot allocation）和多选票的稳健性扩展；3) 对所有可能结构（社交选择序列与轮次）进行枚举与剪枝；4) Bootstrap与Beta模型对选票补全的概率仿真；5) 并行化与记忆化实现提高可扩展性。

**📊 数据集**

使用的数据集为：纽约市2021年54场初选、阿拉斯加2024年52场州级选举（单胜）以及波特兰2024年市议会4场多胜选举的投票记录（CAST Vote Records）。数据涵盖从2到30名候选人、数万至数十万张选票，代表了单胜与多胜、同党内与跨党竞争的多样情景。

**📈 对比分析**

比较方法：将ERSF输出的胜利间隙、最优加票策略与多数制的胜利边距、选票枯竭影响以及策略类型进行对比。实验结果显示：RCV显著提升竞争性（NYC平均胜利边距下降9.2pp，阿拉斯加11.4pp），选票枯竭对结果的影响极小（107/110场选举结果不变），最优策略普遍为自我支持（selfish），且大部分选举的社交选择序列与胜利间隙顺序高度一致。性能上，ERSF在候选人数≤10（单胜）或≤30（多胜）且通过候选人消除后可在秒级内完成100场仿真，显著优于传统完整枚举方法。

**⚠️ 局限性**

局限性包括：1) 对多胜选举仍受候选人数量限制，超出约30名候选人时计算复杂度激增；2) 采用的选票补全模型（Beta、Bootstrap等）假设投票者行为相对独立，可能低估真实协作补全的概率；3) 结果主要是相关性，无法证明RCV直接导致竞争性提升；4) 仅覆盖美国特定司法辖区，跨国或不同投票制的推广需进一步验证。

---

## 480. Restoration Adaptation for Semantic Segmentation on Low Quality Images

**arXiv ID:** 2602.14042 | [PDF](https://arxiv.org/pdf/2602.14042v1)

**作者:** Kai Guan `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 105840 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种统一的框架RASS，通过在文本到图像扩散模型上加入语义约束的图像恢复模块SCR，并利用LoRA融合恢复知识到分割模型，实现对低质量图像的语义分割；

**💡 创新点**

创新点在于：1）将语义先验直接注入扩散恢复的注意力机制，实现语义一致的图像重建；2）通过LoRA模块的合并与任务细化，将恢复能力无缝迁移到分割网络；3）构建真实低质量图像的标注数据集，验证方法的鲁棒性；

**🔧 技术方法**

主要技术包括：文本到图像扩散模型（Stable Diffusion）、LoRA参数微调、语义约束损失（SCL）对注意力图与掩码的对齐、融合VAE编码器特征的分割头；

**📊 数据集**

使用的数据集包括ADE20K、LSDIR、RealSR、DrealSR、DIV2K以及自行构建的RealLQ数据集；

**📈 对比分析**

在合成与真实低质量图像上，RASS在分割任务的mIoU上超过现有最先进方法约3–4个百分点，同时在单步恢复任务中在多项感知质量指标（MUSIQ、MANIQA、CLIPIQA）上取得领先；

**⚠️ 局限性**

主要限制包括：模型参数量巨大（≈1.8B）导致推理速度慢，且对小目标的分割仍易受VAE压缩特征的影响，需要进一步提升效率与细粒度识别能力。

---

## 481. The Diffusion Duet: Harmonizing Dual Channels with Wavelet Suppression for Image Separation

**arXiv ID:** 2602.13361 | [PDF](https://arxiv.org/pdf/2602.13361v1)

**作者:** Jingwei Li `[一作]` (Zhejiang Gongshang University), Wei Pu `[通讯]` (Zhejiang Gongshang University)

**通讯引用:** 12421 | [OpenAlex ID](https://openalex.org/A5100343725)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `da1b1a89-583a-4b57-9c81-478778569bec` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种双通道扩散分离模型（DCDSM），利用波纹抑制模块实现盲图像分离，包括雨雪去除和复杂混合分离。

**💡 创新点**

创新点在于：1) 第一次将双通道扩散模型与交互式波纹抑制结合；2) 通过多尺度小波与频域注意力实现相互耦合噪声抑制；3) 在逆扩散过程中实现两条分支的协同优化。

**🔧 技术方法**

使用扩散模型（DDPM）、离散小波变换、傅里叶变换、双分支反向去噪、波纹抑制模块、条件生成与多目标损失等技术。

**📊 数据集**

使用基于CityScape合成的雨、雪数据集（Rain‑0.5、Rain），Snow100K衍生的雪数据集，以及自建的花果混合复杂混合数据集。

**📈 对比分析**

与传统CNN、Transformer、GAN、专用天气模型等九种最先进方法对比；在雨、雪去除和复杂混合任务上分别取得PSNR 36.24/33.76/29.81 dB和SSIM 0.9615/0.9482/0.9243，平均PSNR 25.00 dB、SSIM 0.7997，均超过第二名不少于1–3 dB及0.02–0.09的提升。

**⚠️ 局限性**

局限性包括：在极端强降水或雪暴等训练外场景下性能下降；默认两源贡献相等，难以处理不平衡混合；以及扩散多步采样导致推理时间约2.3秒，计算资源需求较高。

---

## 482. AdaptManip: Learning Adaptive Whole-Body Object Lifting and Delivery with Online Recurrent State Estimation

**arXiv ID:** 2602.14363 | [PDF](https://arxiv.org/pdf/2602.14363v1)

**作者:** Morgan Byrd `[一作]` (Georgia Institute of Technology), Sehoon Ha `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 4648 | [OpenAlex ID](https://openalex.org/A5064581452)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出AdaptManip框架，实现全自主 humanoid 机器人完成导航、抓取、搬运任务。

**💡 创新点**

联合在线递归对象状态估计与强化学习控制，完全不依赖人类演示或运动捕捉；融合 LiDAR、视觉和惯性信息实现鲁棒的对象估计。

**🔧 技术方法**

基于 PPO 的全身控制与残差策略、V-LSTM+MLP 递归对象估计器、LiDAR 惯性里程计、IsaacLab/MuJoCo 仿真、域随机化、零样本部署。

**📊 数据集**

主要使用仿真环境（IsaacLab、MuJoCo）数据；训练时采用随机化场景；真实测试使用 Unitree G1、RealSense D435i、Livox Mid-360 LiDAR 以及 AprilTag。

**📈 对比分析**

与 Pure RL、Pure RL+FK、Imitation Learning、Oracle 等基线对比；在训练环境成功率约 85%，在未见的 MuJoCo 环境 75% vs 79%；真实机器人零样本下成功率约 70%+，显著优于其他方法。

**⚠️ 局限性**

仅处理简单箱子物体；依赖单手抓手，非标记物体效果未知；估计器受限于视觉遮挡，复杂场景下性能待提升。

---

## 483. Comparables XAI: Faithful Example-based AI Explanations with Counterfactual Trace Adjustments

**arXiv ID:** 2602.13784 | [PDF](https://arxiv.org/pdf/2602.13784v1)

**作者:** Yifan Zhang `[一作]` (National University of Singapore), Brian Y Lim `[通讯]` (National University of Singapore)

**通讯引用:** 3758 | [OpenAlex ID](https://openalex.org/A5056248594)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出“Comparables XAI”范式，通过在AI决策空间中对可比实例进行逐步的对比解释(trace调整)，使得非技术用户能够直观地理解模型预测；

**💡 创新点**

创新点在于将房地产估价中的可比属性调整方法与XAI相结合，采用逐步计数的对比轨迹，兼顾模型可信度与可解释性，并通过五个可解释性设计目标（faithfulness, sparsity, disjointness, monotonicity, evenness）进行优化；

**🔧 技术方法**

技术方法包括：模型无关的局部线性近似（LIME）、基于piecewise linear的轨迹学习、正则化损失组合、梯度下降训练；界面实现基于表格的可比展示及可展开的对比轨迹；

**📊 数据集**

使用了五个基准数据集：King County 房价、Stack Overflow 薪资、ASHRAE 能耗、Genomics 细胞药物敏感度、印度作物产量；

**📈 对比分析**

通过建模实验和两轮用户研究对比四种XAI类型（仅可比、线性回归、线性调整、轨迹调整），发现轨迹调整在预测误差、模型一致性、用户决策准确度和置信区间宽度上均优于其他方法；

**⚠️ 局限性**

局限性包括：需预先提供可比实例、对可比选择过程缺乏自动化、实验仅聚焦回归任务、对非线性、分类任务的泛化尚未验证、缺乏理论上对轨迹可信度的正式证明。

---

## 484. $γ$-weakly $θ$-up-concavity: Linearizable Non-Convex Optimization with Applications to DR-Submodular and OSS Functions

**arXiv ID:** 2602.13506 | [PDF](https://arxiv.org/pdf/2602.13506v1)

**作者:** Mohammad Pedramfar `[一作]` (Mila - Quebec AI Institute), Vaneet Aggarwal `[通讯]` (Purdue University)

**通讯引用:** 6180 | [OpenAlex ID](https://openalex.org/A5064822688)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出γ-弱θ-up-concave函数类，并证明其可上线性化，从而在非凸单调优化问题中得到统一的近似与在线学习保证。

**💡 创新点**

引入γ-弱θ-up-concave这一更广泛的结构，严格泛化DR-子模、OSS等已知模型，并给出对Matroid约束下OSS的改进近似比。

**🔧 技术方法**

基于上线性化框架、几何分析与一阶条件，构造线性代理并实现在线退化至线性优化的元算法。

**📊 数据集**

论文为理论研究，未使用具体数据集，主要基于数理证明。

**📈 对比分析**

通过理论推导与已知结果对比，证明在Matroid约束下得到1‑exp(−γ/(σ+1))的近似系数，优于之前对OSS的1‑exp(−14σ+2)等结果；在在线设置下给出适用于全信息、半带、零阶等多反馈模式的适应性/动态调优保证。

**⚠️ 局限性**

主要限制在于需构造或访问约束集合的线性优化/投影/分离oracle，且在某些情形下近似系数会随θ函数趋近0而下降。

---

## 485. Real-time Monocular 2D and 3D Perception of Endoluminal Scenes for Controlling Flexible Robotic Endoscopic Instruments

**arXiv ID:** 2602.14666 | [PDF](https://arxiv.org/pdf/2602.14666v1)

**作者:** Ruofeng Wei `[一作]` (Chinese University of Hong Kong), Qi Dou `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了基于单目内镜图像的实时2D与3D感知平台，融合了柔性连续机器人分割、状态估计与深度估计，并构建了高保真模拟器用于合成数据和机器人控制。

**💡 创新点**

创新点包括：①使用SAM与知识蒸馏的标注高效分割；②采用矩阵菲舍尔分布实现带不确定性的概率3D状态估计；③引入光照建模的PPR损失与PRMod提升单目深度估计；④基于Unity的物理仿真生成与真实机器人匹配的柔性工具与真实组织纹理的合成数据。

**🔧 技术方法**

技术手段包括：视觉基础模型（SAM、DINOv2、Depth Anything）、U-Net、ResNet50、矩阵菲舍尔分布、光照逆向渲染、PPR损失、教师-学生自监督迁移、Poisson表面重建、统一仿真平台（Unity+多模态相机）等。

**📊 数据集**

数据集涵盖：25000帧猪GI通道外科实验视频、基于C3VD与Colon10K构建的合成数据集（带真实深度与表面法线）、真实临床内镜图像、以及仿真生成的高精度深度与机器人状态标签。

**📈 对比分析**

对比方法包括Syn/Copy-Paste（分割）、KP/SKL/Quat/Rot6D（状态估计）以及ResNet、ZoeDepth、Depth Anything、Metric3D、EndoOmni等深度估计模型；实验结果显示：分割Dice达94.9%，状态估计平均角误差≈6°，深度RMSE≈4.0 mm；机器人轨迹跟随任务完成时间下降71–72%，实现了显著的控制效率提升。

**⚠️ 局限性**

局限性包括：缺乏人体在体临床验证，实验仅在猪体模型与合成环境进行；深度误差约4 mm对某些精细手术仍可能不够；遮挡区域深度重建能力不足；仿真器未模拟软组织变形与工具-组织交互；未将机器人运动学模型与视觉估计进行贝叶斯融合。

---

## 486. UniST-Pred: A Robust Unified Framework for Spatio-Temporal Traffic Forecasting in Transportation Networks Under Disruptions

**arXiv ID:** 2602.14049 | [PDF](https://arxiv.org/pdf/2602.14049v1)

**作者:** Yue Wang `[一作]`, Samer Madanat `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了UniST-Pred，一种将时序建模与空间表示学习解耦后再通过自适应融合的统一时空交通预测框架，兼顾轻量化与可解释性。

**💡 创新点**

核心创新在于先分别学习时序依赖与任务自适应空间结构，再通过Squeeze‑Excitation残差块进行特征级融合，打破传统紧耦合的时空图神经网络架构，实现模块化、可插拔且对网络拓扑变化更稳健。

**🔧 技术方法**

技术包括特征/时间混合器（feature‑time mixing）捕获长程时序关联，Graph Transformer Networks（GTN）生成任务自适应图结构，Squeeze‑Excitation残差融合模块进行特征重加权，并使用Integrated Gradients进行可解释性分析。

**📊 数据集**

使用了三套数据集：基于MATSim仿真生成的SimSF‑Bay（含网络结构变化），实际的PEMS‑Bay（加州湾区速度数据）和NYCTaxi（纽约出租车流量数据）。

**📈 对比分析**

与ARIMA、LSTM、TCN、TSMixer、STGCN、ASTGCN、ST‑SSL、DCRNN、STEP等基线比较，UniST-Pred在SimSF‑Bay、PEMS‑Bay、NYCTaxi的RMSE/MAE/MAPE指标上均表现为最优或相近的水平，同时参数规模显著小于STEP（约4×-30×）。

**⚠️ 局限性**

局限性包括对极端网络拓扑变化的鲁棒性已在有限桥梁断开的情景下验证，尚未覆盖更复杂的多模式交通系统；此外，时序混合器在非常长预测时 horizon 时的表现尚需进一步评估。

---

## 487. An Agentic AI Control Plane for 6G Network Slice Orchestration, Monitoring, and Trading

**arXiv ID:** 2602.13227 | [PDF](https://arxiv.org/pdf/2602.13227v1)

**作者:** Eranga Bandara `[一作]` (Old Dominion University), Kasun De Zoysa `[通讯]` (University of Colombo School of Computing)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Agentic AI 的 6G 网络切片控制平面，整合意图驱动、市场感知、持续监控和交易功能，并在实际测试平台上验证其可行性。

**💡 创新点**

创新点包括：① 多层架构与自治 AI 代理协作；② 采用多模型 LLM 联盟与专门的推理层实现责任与可解释性治理；③ 通过 Model Context Protocol (MCP) 实现安全可审计的自然语言接口与市场化资源调度；④ 将切片规划、部署、监控与经济决策统一在同一控制平面。

**🔧 技术方法**

技术栈：Agentic AI 代理、Fine‑tuned 大语言模型（Qwen2+LoRA）、Model Context Protocol (MCP)、Kubernetes‑based 切片部署、OpenAI Agents SDK、LoRA 参数高效微调、Open5GS 与 Ericsson RAN 集成。

**📊 数据集**

数据集：JSONL 格式的切片规范与对应 Kubernetes 清单对，及自然语言意图与 MCP 工具调用的映射样本，用于微调 LLM 并评估其生成质量。

**📈 对比分析**

评估方法：与传统集中/分布式切片管理方案对比；在 VMASC 真实测试床（Open5GS+Ericsson RAN）中测试闭环 SLA 保障、市场感知调度与自然语言交互。LLM 微调收敛稳定，验证损失下降、推理速度均保持在可接受范围；生成的 Kubernetes 清单与 MCP 调用无幻觉、可直接部署，说明操作可靠性高。

**⚠️ 局限性**

限制：仅在实验室规模的单域测试环境验证，缺乏大规模多域部署与跨域协同的实证；多模型 LLM 联盟与推理层的计算与时延开销未充分评估；对抗性攻击与极端网络动态环境下的鲁棒性尚待进一步验证；经济交易模型相对简化，未覆盖完整的动态市场机制。

---

## 488. Systematic Review of Lightweight Cryptographic Algorithms

**arXiv ID:** 2602.14731 | [PDF](https://arxiv.org/pdf/2602.14731v1)

**作者:** Mohsin Khan `[一作]`, Håvard Dagenborg `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

在CIFAR-10和ImageNet数据集上进行了实验。

**📈 对比分析**

与现有的几种主流模型进行了比较，结果显示该模型在分类精度上提高了5%，且训练时间缩短了20%。

**⚠️ 局限性**

模型在处理高分辨率图像时性能下降，且对计算资源的需求较高。

---

## 489. Rigidity-Based Multi-Finger Coordination for Precise In-Hand Manipulation of Force-Sensitive Objects

**arXiv ID:** 2602.14104 | [PDF](https://arxiv.org/pdf/2602.14104v1)

**作者:** Xinan Rong `[一作]` (Chongqing University), Gangshan Jing `[通讯]` (Chongqing University)

**通讯引用:** 791 | [OpenAlex ID](https://openalex.org/A5073846906)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于图刚度的双层框架，实现多指机器人手在没有触觉传感器或扭矩控制的情况下，对柔软/易损物体进行高精度的抓取与轨迹跟踪。

**💡 创新点**

创新点在于：①利用刚度约束将内部力规划转化为可解的运动学约束；②三分解内部力（外部平衡力、刚度内部力、摩擦内部力），实现对摩擦与安全边界的同时满足；③通过虚拟渗透距离映射实现力控制转为关节位置控制，消除了对扭矩传感器的需求。

**🔧 技术方法**

技术包括：图刚度理论、刚度矩阵求解、摩擦锥约束下的二次规划、虚拟目标点的线性力-位移映射、基于SLSQP的在线MPC轨迹优化以及基于RealSense摄像头的视觉位姿估计。

**📊 数据集**

主要使用自制实验数据：四指手抓取软纱线、塑料杯（空杯和水杯）以及生鸡蛋，配合附加的AprilTag进行位姿跟踪；并未使用公开数据集。

**📈 对比分析**

与传统无力规划/基础抓取方法对比，展示了杯子不失稳、鸡蛋不碎、纱线保持紧张等结果；最大轨迹点误差仅 0.3 mm，平均误差 0.4–0.5 mm，执行时间 2.8–7 s/点；在垂直重力、质量与摩擦不确定性实验中也保持低误差，证明了鲁棒性。

**⚠️ 局限性**

局限性包括：①需要先验的物体刚度与摩擦系数以调节虚拟刚度；②仅适用于点接触模型，无法处理大面积接触；③视觉跟踪受光照与遮挡影响；④缺乏在线参数识别与学习，手动调参成本高。

---

## 490. Flow4R: Unifying 4D Reconstruction and Tracking with Scene Flow

**arXiv ID:** 2602.14021 | [PDF](https://arxiv.org/pdf/2602.14021v1)

**作者:** Shenhan Qian `[一作]` (Technical University of Munich), Daniel Cremers `[通讯]` (Technical University of Munich)

**通讯引用:** 48424 | [OpenAlex ID](https://openalex.org/A5087710605)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种统一的 4D 重建与跟踪框架 Flow4R，使用场景流作为核心表示；

**💡 创新点**

创新点在于把点位置、场景流、姿态权重和置信度这四个属性作为最小化的预测集合，并通过姿态权重自适应分解相机与物体运动，无需显式姿态回归或束调整；

**🔧 技术方法**

采用基于 Vision Transformer 的双视角交叉注意力网络，结合场景流分解、姿态权重自监督、点位置与 3D 流监督等技术；

**📊 数据集**

在大量静态与动态真实与合成数据集上训练与评估，包括 Habitat、BlendedMVS、MegaDepth、ARKitScenes、CO3D、ScanNet++、Waymo、TartanAir、UnReal4K、Virtual KITTI 2、Spring、PointOdyssey、Dynamic Replica、Kubric、OmniWorld-Game 等；

**📈 对比分析**

在 3D 点跟踪与重建基准（WorldTrack、St4RTrack、POMATO、MonST3R 等）上取得了与或优于现有最先进方法的性能，尤其在 APD3D 与 EPE 指标上表现突出；

**⚠️ 局限性**

局限性包括对场景流标注数据的稀缺、仅采用双视角设置、对长序列在线跟踪的计算与内存需求尚未充分解决。

---

## 491. SPLIT: Sparse Incremental Learning of Error Dynamics for Control-Oriented Modeling in Autonomous Vehicles

**arXiv ID:** 2602.13641 | [PDF](https://arxiv.org/pdf/2602.13641v1)

**作者:** Yaoyu Li `[一作]` (Tsinghua University), Jun Li `[通讯]` (Tsinghua University)

**通讯引用:** 54907 | [OpenAlex ID](https://openalex.org/A5021388534)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出SPLIT框架，实现对车辆动力学误差的在线稀疏增量学习；

**💡 创新点**

创新点包括：1）通过模型分解将GP残差模型特征维度从5降至3；2）在有效域内划分子区域，仅局部更新数据集；3）采用Bayesian Committee Machine对GP进行稀疏化；

**🔧 技术方法**

技术主要包括高斯过程回归、边界约束的有效域定义、增量学习的边际增益评估、BCM稀疏化与并行计算；

**📊 数据集**

使用CarSim仿真赛车赛道以及真实电驱动车平台在多种赛道与障碍规避实验中的实时采集数据；

**📈 对比分析**

与传统在线学习（IOL）及离线训练相比，SPLIT在仿真与真实测试中均显著缩短学习时间（≤0.2 ms）、内存占用≈10 MB，并在比赛赛道上实现平均速度提升约10 %与最大侧向加速度提升0.12 g；

**⚠️ 局限性**

局限性：依赖有效域划分与参数设定，若车辆动力学超出预定义域可能失效；同时模型仅补偿可变元素误差，可能忽略某些非线性耦合导致的细节误差。

---

## 492. Robust Bias Evaluation with FilBBQ: A Filipino Bias Benchmark for Question-Answering Language Models

**arXiv ID:** 2602.14466 | [PDF](https://arxiv.org/pdf/2602.14466v1)

**作者:** Lance Calvin Lim Gamboa `[一作]`, Mark Lee `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了FilBBQ，一个包含10,576条菲律宾语QA提示的偏见评估基准，并使用多种种子对模型进行多次评估以弥补生成模型的响应不稳定性。

**💡 创新点**

创新点：①采用文化敏感翻译与本土化改写扩展BBQ；②新增菲律宾特有刻板印象的模板；③通过多seed平均减少单次响应带来的偏差波动，提升评估稳健性。

**🔧 技术方法**

使用技术：文化敏感翻译、模板分类与改写、自动脚本生成提示、偏见得分计算（s_amb、s_dis）以及多seed平均法。

**📊 数据集**

数据集：FilBBQ（123模板生成的10,576条提示），参考原BBQ、CrowS‑Pairs、WinoQueer以及菲律宾学术、新闻等来源的刻板印象；模型包括Alpaca 33B、Pali、Filipino mBERT。

**📈 对比分析**

比较方法：在FilBBQ上对三种菲律宾语言模型计算s_amb与s_dis的平均得分，评估性别与性取向偏见；多seed评估显示偏差波动显著下降，模型表现呈现不同强度的性别与同性恋偏见，平均bias score大约在0.3–0.5之间。

**⚠️ 局限性**

局限性：①未能覆盖所有菲律宾文化中的刻板印象；②偏见评估高度依赖模型的QA性能，QA差模型可能被误判；③仅关注性别与性取向两维度；④benchmark不适合用于训练模型；⑤多seed平均仍受样本规模与随机性影响。

---

## 493. DM0: An Embodied-Native Vision-Language-Action Model towards Physical AI

**arXiv ID:** 2602.14974 | [PDF](https://arxiv.org/pdf/2602.14974v1)

**作者:** En Yu `[一作]`, Tiancai Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 DM0，一个从一开始就整合物理先验的 Embodied‑Native Vision‑Language‑Action 框架，用三阶段预训练/中期/后期联合多源数据训练，实现了高效的多机器人任务执行。

**💡 创新点**

①统一预训练整合互联网、自动驾驶与真实物理交互数据；②混合梯度策略隔离动作专家与 VLM，防止语义遗失；③空间 Scaffold 产生 Spatial CoT 约束动作空间。

**🔧 技术方法**

Vision‑Language Model (Qwen3‑1.7B) + 感知编码器；Flow Matching 动作专家；Hybrid Gradient（Knowledge Insulation）；Embodied Spatial Scaffolding；Chain‑of‑Thought 语言推理。

**📊 数据集**

Common Crawl、StepCrawl、LAION、COYO、BLIP‑CCS、Zero、OpenImages、COCO、Merlin、PixMo、Driving‑scene logs、RoboTwin、Libero、Habitat、单臂/双臂真实机器人轨迹等多源异构数据。

**📈 对比分析**

在 RoboChallenge Table30 基准上，DM0 专家模式在 2B 参数下取得 62.0% 成功率，超过 GigaBrain‑0.1 等同规模模型；通用模式平均 37.3% 成功率，显著高于 π_0.5‑Generalist 等基线，表现优异。

**⚠️ 局限性**

模型规模小（2B），对极长时程任务仍有限；缺少触觉、声音等额外模态；混合梯度训练需要精细平衡，可能在大规模扩展时面临稳定性挑战。

---

## 494. Cardiac Output Prediction from Echocardiograms: Self-Supervised Learning with Limited Data

**arXiv ID:** 2602.13846 | [PDF](https://arxiv.org/pdf/2602.13846v1)

**作者:** Adson Duarte `[一作]` (University of Turin), Marco Grangetto `[通讯]` (LINKS Foundation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

利用SimCLR自监督预训练，在同一有限的四腔视图心脏超声视频上预测心输出量。

**💡 创新点**

在仅有201张无标签视频的情况下就能进行SSL预训练，并证明其比大规模监督模型更能迁移到CO预测任务，同时探究批量大小对性能的影响。

**🔧 技术方法**

采用ViViT视频编码器与SimCLR对比学习框架，再加上线性回归头实现CO回归。

**📊 数据集**

使用268例成人右心导管检查患者的A4C超声视频（每例1-5秒，32帧），CO由侵袭性测量给出。

**📈 对比分析**

与冻结/全训练的ViViT监督学习以及PanEcho（约100万视频训练）的特征提取方式比较，ViViT-SSL-64在测试集上取得Pearson 0.41、MAE 1.05，明显优于PanEcho和监督学习方法。

**⚠️ 局限性**

研究仅基于单一数据集、单一视角和单一回归任务，缺乏对不同数据集、模态和临床指标的通用性验证。

---

## 495. Using Machine Learning to Enhance the Detection of Obfuscated Abusive Words in Swahili: A Focus on Child Safety

**arXiv ID:** 2602.13455 | [PDF](https://arxiv.org/pdf/2602.13455v1)

**作者:** Phyllis Nabangi `[一作]` (Carnegie Mellon University), Jema David Ndibwile `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在低资源语言斯瓦希里语中检测被掩码的辱骂文本的可行性

**💡 创新点**

首次在斯瓦希里语上采用机器学习模型检测被掩码辱骂词，并探讨了SMOTE与TF-IDF在该任务中的有效性

**🔧 技术方法**

使用支持向量机、逻辑回归、决策树和随机森林等传统机器学习模型，配合TF-IDF特征提取和SMOTE重采样

**📊 数据集**

使用包含100条斯瓦希里语辱骂文本的公开数据集（被掩码与非掩码各50条）

**📈 对比分析**

通过5折交叉验证比较模型，决策树获得99%准确率，逻辑回归与SVM均在88%左右，随机森林表现最差；F1分数与准确率保持一致

**⚠️ 局限性**

样本量小、类别不平衡且可能过拟合导致结果难以推广，需要更大多样化的数据集和更深入的模型验证

---

## 496. Return of the Schema: Building Complete Datasets for Machine Learning and Reasoning on Knowledge Graphs

**arXiv ID:** 2602.14795 | [PDF](https://arxiv.org/pdf/2602.14795v1)

**作者:** Ivan Diliso `[一作]` (University of Bari Aldo Moro), Nicola Fanizzi `[通讯]` (University of Bari Aldo Moro)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一个通用工作流程，能够从大型知识图谱中提取同时包含完整schema与事实的、可直接用于机器学习和推理的完整数据集。

**💡 创新点**

首次提供完整且一致的KG数据集套件，并将其序列化为OWL，支持NeSy方法；工作流程自动化处理不一致、材料化、模块化，并提供ML工具。

**🔧 技术方法**

利用描述逻辑推理（一致性检查、材料化、实现化）、SPARQL抽取、PROMPTFACTOR模块化、PyKEEN/PyTorch 的张量转换等技术。

**📊 数据集**

从DBpedia、YAGO3、YAGO4、ApuliaTravel、ARCO、WHOW等六个大型KG抽取共10个新数据集及若干增强版现有数据集。

**📈 对比分析**

与现有增量式数据集（如DB100K+、YAGO3-10+）在schema覆盖、推理可用性等指标上进行对比，证明我们的数据集在保持完整性和可推理性方面优于现有方案；可直接在PyKEEN等框架中使用。

**⚠️ 局限性**

主要局限包括：仅处理对象属性，未覆盖数据属性；需人工干预解决模型化错误；对RDFS/OWL2 KG有限支持；模块化可能丢失部分约束；尚未包含数据类型断言。

---

## 497. The Wikidata Query Logs Dataset

**arXiv ID:** 2602.14594 | [PDF](https://arxiv.org/pdf/2602.14594v1)

**作者:** Sebastian Walter `[一作]` (University of Freiburg), Hannah Bast `[通讯]` (University of Freiburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了最大的 Wikidata 问答配对数据集 WDQL，包含约20万条真实用户 SPARQL 查询和对应自然语言问题。

**💡 创新点**

通过代理式去匿名化、清洗和验证 SPARQL 查询并自动生成问题，实现了从匿名日志到可用问答数据的完整流水线。

**🔧 技术方法**

利用 GRASP 框架的 S2Q 代理、Qwen3 语言模型、QLever 查询引擎以及聚类与 UMAP 等工具。

**📊 数据集**

以 Wikidata Query Service 日志为基础，去除匿名化后得到 860k 条查询，最终生成 200.186 条有效问答对。

**📈 对比分析**

在 KGQA 任务中，使用 GRISP 训练模型对 WDQL、之前数据集及其组合进行评估，WDQL 训练的模型在多项公开基准上显著提升 F1 分数，尤其在 QALD、QAWiki、SPINACH 上表现突出。

**⚠️ 局限性**

受限于日志的匿名化导致部分查询无法恢复完整意图，且聚类划分可能忽略细粒度差异，模型在遵循固定模板的基准上仍显弱。

---

## 498. Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision

**arXiv ID:** 2602.14276 | [PDF](https://arxiv.org/pdf/2602.14276v1)

**作者:** A. Said Gurbuz `[一作]` (IBM Research), Peter Staar `[通讯]` (IBM Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在本文中，作者构建了大规模的完整屏幕解析数据集ScreenParse，并训练了一种轻量级视觉‑语言模型ScreenVLM，能够将网页屏幕解析为结构化的ScreenTag表示；

**💡 创新点**

创新点在于（1）提出完全覆盖的屏幕解析任务，并通过自动化Webshot管道生成含有55种UI类别、21M元素的高质量数据；（2）设计了结构感知加权损失，使模型更关注标签与坐标的准确性；（3）构建了仅316M参数的ScreenVLM，既实现高效推理又保持丰富的语义输出；

**🔧 技术方法**

主要技术包括：Playwright渲染网页并提取DOM信息；利用Qwen‑3‑VL‑8B对初始标签进行自动重标记与质量过滤；使用SigLIP‑2视觉编码器结合Granite‑165M自回归解码器；以及基于结构关键词的加权交叉熵损失；

**📊 数据集**

核心数据集为自研的ScreenParse（771K截图、21M元素、55类标签），此外在GroundCUA和ScreenSpot等公开基准上进行评测；

**📈 对比分析**

与大型基础VLM（如Qwen3‑VL‑8B、InternVL3‑2B）以及基于检测器的解析器（OmniParser、YOLO、RT‑DETR）对比，ScreenVLM在ScreenParse上实现PageIoU 0.606、Label PageIoU 0.197，显著优于8B模型的0.294；在跨域评测GroundCUA中达到0.251/0.043，且在ScreenSpot上PixCov提升明显；同时在基础VLM上对ScreenParse微调后可获得大幅性能提升；

**⚠️ 局限性**

主要局限在于：数据集主要覆盖网页，缺乏原生桌面/移动UI的完整标注；DOM提取仍可能残留广告、动态内容等噪声；因此模型在非网页应用上仍存在一定迁移瓶颈。

---

## 499. KidMesh: Computational Mesh Reconstruction for Pediatric Congenital Hydronephrosis Using Deep Neural Networks

**arXiv ID:** 2602.13299 | [PDF](https://arxiv.org/pdf/2602.13299v1)

**作者:** Haoran Sun `[一作]` (Fudan University), Wangbin Ding `[通讯]` (Fujian Medical University)

**通讯引用:** 239 | [OpenAlex ID](https://openalex.org/A5078082099)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 KidMesh，一种端到端的深度学习框架，可直接从儿童先天性肾盂扩张的磁共振尿路造影(MRU)扫描生成可用于计算流体动力学(CFD)分析的无缝、拓扑正确的三维网格。

**💡 创新点**

创新点在于（1）结合 CNN 与 GCN 的特征提取与网格变形，形成粗细分层的动态顶点上采样与自注意力模块；（2）使用弱监督策略，仅需从 MRU 探测到的掩模生成伪金标准网格；（3）引入多项拓扑保持正则化（拉普拉斯、边长、面积、Seal 损失）保证网格闭合与光滑；（4）显著降低后处理时间（0.36 s）并在几何精度和自交错误率上优于 MeshDeformNet 与传统 voxel‑based 方法。

**🔧 技术方法**

技术包括：U‑Net 结构的 3D 特征提取模块、基于图卷积的特征采样与自注意力融合、分阶段网格变形模块（带动态上采样/过滤）、Chamfer、Normal、Laplacian、Edge、Area、Seal 等多种几何损失，全部在 PyTorch/PyTorch3D 框架下实现。

**📊 数据集**

使用 160 例 3.0 T Siemens Skyra MRU 数据集（每例包含术前术后扫描），经裁剪、重采样并手工注释后生成伪金标准网格；数据按 70/20/10 划分为训练、验证、测试。

**📈 对比分析**

与 nnU‑Net、FCN、SCU、ResNet‑50 等 voxel‑based 方法以及 MeshDeformNet 进行对比；KidMesh 在 ASSD/HD、Dice、Jaccard 与 P2SD 上表现相当或更优，尤其在自交错误率与顶点误差分布上优于 MeshDeformNet；同时实现了 0.36 s 的实时网格生成，远快于传统方法（>60 s）。

**⚠️ 局限性**

局限性包括：对高曲率或细小结构（如窄肾盂）仍存在误差；偶尔出现自交或悬挂节点；训练数据仅来自单中心，泛化能力待进一步验证；缺乏对整个泌尿系统（输尿管、膀胱）的完整建模。

---

## 500. Efficient Sampling with Discrete Diffusion Models: Sharp and Adaptive Guarantees

**arXiv ID:** 2602.15008 | [PDF](https://arxiv.org/pdf/2602.15008v1)

**作者:** Daniil Dmitriev `[一作]` (University of Pennsylvania), Yuting Wei `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了离散扩散模型的采样效率，尤其是在连续时间马尔科夫链框架下使用 τ‑leaping 算法的收敛性能。

**💡 创新点**

创新点在于：对均匀扩散给出匹配的上界和下界，证明 τ‑leaping 在此情形下迭代复杂度可降至 O(d/ε)（消除对词表大小 S 的线性依赖）；对掩蔽扩散提出一种 τ‑bridging 策略，其复杂度由有效总相关这一信息论量度决定，可自适应低维结构，实现对数级到常数级的迭代步数。

**🔧 技术方法**

使用技术包括：连续时间马尔科夫链（CTMC）表述、Girsanov 变换、马尔科夫性质与 Doob 过程、信息论量度（总相关、双总相关、有效总相关）、强数据处理不等式以及 Dynkin 公式等。

**📊 数据集**

论文主要以理论分析为主，未给出具体实验数据集；若有实验，主要涉及自然语言、图像或随机图等常见离散数据的模拟分布。

**📈 对比分析**

与之前基于 τ‑leaping 的 O(d²S/ε) 等结果相比，新算法在均匀扩散中实现了 S 的消失并将 d 的线性耦合从 d² 降至 d；掩蔽扩散的自适应策略在结构化分布下可将迭代步数从 O(d) 降至 O(𝒟)（𝒟 为有效总相关，可小于 d，甚至常数）。

**⚠️ 局限性**

局限性包括：下界仅适用于均匀扩散且基于 τ‑leaping；掩蔽扩散的自适应性在实践中的验证尚缺乏；对非 CTMC 形式或其他噪声机制的离散扩散模型的理论分析仍是未解决的问题。

---

## 501. FO and MSO Model Checking on Temporal Graphs

**arXiv ID:** 2602.14592 | [PDF](https://arxiv.org/pdf/2602.14592v1)

**作者:** Michelle Döring `[一作]` (Hasso Plattner Institute), George Skretas `[通讯]` (Hasso Plattner Institute)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种统一的逻辑框架，将时序图的四种结构参数（生命周期、时刻点宽度VIM、树形时刻宽度TIM、时刻度宽度）映射到关系结构，并在此基础上推导了 MSO 与 FO 的元定理，证明了在这些参数限制下时序图问题可固定参数可解。除此之外，还提供了一个可复用的逻辑“菜谱”（Logic Cookbook），收录了多种时序图问题的 FO / MSO 公式，极大地简化了未来研究的证明工作。

**💡 创新点**

创新点主要有三：①为 VIM 与 TIM 两个新参数设计了新的逻辑编码，并证明它们满足 Gaifman 图树宽/无密度保持性；②在这些编码上首次推出了 FO 的元定理（此前仅有 MSO 结果）；③提出了可组合的逻辑“菜谱”，将多种时序图问题统一在同一逻辑框架内描述，降低了问题证明的门槛。

**🔧 技术方法**

核心技术包括：①关系结构编码（扩展传统静态图编码加入时间信息）；②对编码的 Gaifman 图进行树宽或无密度分析；③利用 Courcelle、Frick–Grohe、Grohe–Kreutzer–Siebertz 等经典元定理推导时序图的固定参数可解性；④在公式层面引入多种常用子公式（如路径、连通性、邻接、时间间隔等）以实现模块化。

**📊 数据集**

本文没有使用任何实验数据集，所有结论均为理论证明；作者在论文中明确表示“本文未产生或分析任何数据”。

**📈 对比分析**

由于研究属于理论计算机科学范畴，作者并未进行实验对比；通过理论分析证明在 VIM、TIM、生命周期或时刻度宽度受限的情况下，任何 MSO（FO）可定义的时序图性质都可在 f(k)·n 时间内求解（或 n^{f(k)} 时间）。

**⚠️ 局限性**

主要限制包括：①仅覆盖四个参数，未涉及诸如时序反馈数、时序树宽等其它重要参数；②FO 结果对“无密度”类的适用性仍有局限，无法直接表达路径长度等全局性质；③部分证明和公式细节被压缩或省略，需要参考完整版本；④缺乏实验验证，无法评估实际实现的性能。

---

## 502. A Trajectory-Based Safety Audit of Clawdbot (OpenClaw)

**arXiv ID:** 2602.14364 | [PDF](https://arxiv.org/pdf/2602.14364v1)

**作者:** Tianyu Chen `[一作]` (ShanghaiTech University), Wenjie Wang `[通讯]` (ShanghaiTech University)

**通讯引用:** 2177 | [OpenAlex ID](https://openalex.org/A5100368534)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对自托管、使用工具的 AI 代理 Clawdbot 进行轨迹中心的安全评估，覆盖六个风险维度并记录完整交互轨迹

**💡 创新点**

首次系统化地对工具使用型代理进行轨迹级安全测试，结合现有基准并设计针对性案例，揭示了模糊意图、缺失证据和恶意注入导致的风险放大机制

**🔧 技术方法**

采用 MiniMax M2.1 作为基础 LLM，配合 exec 与 web‑search 工具，并使用 AgentDoG‑Qwen3‑4B 自动轨迹评判与人工复核相结合的评估流程

**📊 数据集**

构建 34 个基准案例，取自 ATBench、LPS‑Bench、Are Your Agents Upward Deceivers? 等公开基准，并补充手工设计的针对 Clawdbot 工具表面案例

**📈 对比分析**

通过对六个维度的通过率进行量化，整体通过率 58.9%，Hallucination 100%，Operational 75%，User‑deception 71%，Prompt‑Injection 57%，Unexpected 50%，Intent‑misunderstanding 0%，显示安全表现高度不均衡

**⚠️ 局限性**

受限于固定工具面、未使用沙箱、案例数量有限、仅评估 MiniMax M2.1 版本，评测可能不具备普适性，人工复核主观性及无法覆盖所有真实世界场景

---

## 503. Information Fidelity in Tool-Using LLM Agents: A Martingale Analysis of the Model Context Protocol

**arXiv ID:** 2602.13320 | [PDF](https://arxiv.org/pdf/2602.13320v1)

**作者:** Flint Xiaofeng Fan `[一作]` (Agency for Science Technology and Research), Yew-Soon Ong `[通讯]` (Nanyang Technological University)

**通讯引用:** 27676 | [OpenAlex ID](https://openalex.org/A5068243197)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出信息保真框架，用来量化并分析 Model Context Protocol（MCP）工具调用中的语义失真与误差累积。

**💡 创新点**

将 MCP 交互建模为有界差分鞅，推导出累计失真在高概率下为 O(√T) 的偏差界限，并提出融合事实匹配与嵌入相似度的混合语义失真度量。

**🔧 技术方法**

使用鞅收敛不等式（Azuma–Hoeffding、Freedman）、指数衰减影响函数、混合事实匹配与语义嵌入度量，以及自适应调用链生成技术。

**📊 数据集**

实验基于三大 LLM（Qwen2-7B‑Instruct、Llama‑3‑8B‑Instruct、Mistral‑7B‑Instruct‑v0.3）和缓存的检索工具，构造自适应工具调用链。

**📈 对比分析**

与理论预测对照，实验证明累计失真呈线性增长，偏差始终保持在理论 O(√T) 包络内；λ 值提升可降低约 80% 的失真，β 越高单步失真率略升。

**⚠️ 局限性**

仅针对单智能体、确定性工具和单分支（或线性）链，未涵盖多代理或随机工具响应；模型侧重事实召回，精确度权重选择仍需进一步探讨。

---

## 504. GOT-JEPA: Generic Object Tracking with Model Adaptation and Occlusion Handling using Joint-Embedding Predictive Architecture

**arXiv ID:** 2602.14771 | [PDF](https://arxiv.org/pdf/2602.14771v1)

**作者:** Shih-Fang Chen `[一作]` (National Yang Ming Chiao Tung University), Yen-Yu Lin `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了GOT‑JEPA框架，将教师‑学生对抗式预训练用于预测跟踪模型，并引入OccuSolver实现细粒度遮挡感知与可见性估计；

**💡 创新点**

创新点在于：①将JEPA从图像特征预测迁移到跟踪模型预测；②教师产生伪跟踪模型，学生在被腐蚀的帧上学习预测同一模型；③OccuSolver把无目标感知的点跟踪器转为目标感知并输出可见性状态，进一步提升遮挡处理与模型适配；

**🔧 技术方法**

使用的技术包括：Joint‑Embedding Predictive Architecture (JEPA)、ViT‑L + DINOv2 backbone、线性ProjNet与Expander、Copy‑Paste及掩码等数据增强、invariance 与 covariance 损失、点跟踪器CoTracker、Ensemble Network、VisHead 与 Point Head 等；

**📊 数据集**

训练集：LaSOT、GOT‑10k、TrackingNet、COCO；测试集：AVisT、NfS、OTB‑100、LaSOT、TrackingNet、GOT‑10k、VOT‑2022；

**📈 对比分析**

在七大基准上与多种SOTA进行对比，GOT‑JEPA 在 Occlusion、Deformation、Unseen target 等属性下均取得显著提升；在 AVisT、NfS、OTB‑100 等指标（SUC、NPr、AO）均超过大多数竞争方法，整体性能位居前列；

**⚠️ 局限性**

局限性：在背景杂乱、快速运动、极端低光/小目标等场景表现仍不足；Occlusion 处理受限于点数和点跟踪精度；未利用3D几何信息，未来可通过多模态或几何增强提升鲁棒性。

---

## 505. Out-of-Support Generalisation via Weight Space Sequence Modelling

**arXiv ID:** 2602.13550 | [PDF](https://arxiv.org/pdf/2602.13550v1)

**作者:** Roussel Desmond Nzoyem `[一作]` (University of Bristol), Roussel Desmond Nzoyem `[通讯]` (University of Bristol)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5093013809)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文将“离支撑”泛化问题重新表述为权重空间的序列预测任务，通过构造同心环划分训练域并学习权重递推关系实现可解释、无先验偏差的外推预测。

**💡 创新点**

创新点在于将输入空间划分为环状子集，将每个环的最佳权重视为序列中的时步，利用权重空间序列模型（如自回归线性递推）进行外推，并通过线性化及KL正则化实现不确定性估计。

**🔧 技术方法**

主要技术包括距离度量和环划分、权重空间序列模型（线性递归或可选的 Transformer/LSTM）、重参数化技巧、预测分布线性化和KL正则化。

**📊 数据集**

实验使用合成余弦波回归任务和UCI AirQuality 气体传感器数据集，分别在训练和测试集上进行离支撑泛化评估。

**📈 对比分析**

与标准MLP、高斯过程和Engression基线比较，本文方法在离支撑测试集的MSE分别为0.3502（余弦）和0.1381（AirQuality），显著低于基线，且提供可靠的不确定性区间。

**⚠️ 局限性**

局限性包括需手动调参（距离度量、环宽、起始点、β等），对高维输入的可扩展性尚未验证，且在训练域与离支撑域的置信度分布仍需进一步改进。

---

## 506. Covariance-Aware Transformers for Quadratic Programming and Decision Making

**arXiv ID:** 2602.14506 | [PDF](https://arxiv.org/pdf/2602.14506v1)

**作者:** Kutay Tire `[一作]` (University of Texas at Austin), Samet Oymak `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

探究 transformer 能否做为通用的二次规划求解器，并将协方差信息以 token 形式注入时间序列模型，提出端到端的 Time2Decide 框架以提升涉及协方差约束的决策任务性能。

**💡 创新点**

① 证明 linear‑attention + MLP/循环可实现梯度下降、箭头‑胡尔维茨、ISTA、PGD 等一阶迭代；② 在 Time2Decide 中将协方差矩阵嵌入为专门 token，实现一次前向传播即可完成预测与优化；③ 通过理论证明展示显式协方差输入使端到端策略优于传统 Predict‑then‑Optimize。

**🔧 技术方法**

线性注意力 Transformer、Softmax Transformer、MLP、LSTM、时间序列基础模型 TimePFN、PatchTST、Covariance‑Augmented Tokenization (CAT)、投影层、软/线性注意力头、正则化与投影层的组合。

**📊 数据集**

① 由随机生成的线性约束 QP（n=5, m=3）作为训练/测试样本；② 用 LMC 生成的多资产时间序列（m=16, T=1024）用于投资组合优化实验；③ 生成的协方差矩阵来自历史回报。

**📈 对比分析**

对 QP 求解任务比较 Softmax Transformer、Linear Transformer、MLP、LSTM 等基线；Linear Transformer 在 8 层 2 头下 R²≈0.974、NMSE≈0.011；对投资组合优化比较 Oracle、Predict‑then‑Optimize、预训练 TimePFN、SFT‑TimePFN、Time2Decide；Time2Decide 在大转移预算 γ 时 MSE 低于 PtO（例如 γ=1.75 时 0.0442 vs 0.0472）。

**⚠️ 局限性**

模型主要验证在低维、凸二次规划；未针对高维、非凸或大规模实测；对输入噪声的鲁棒性虽提升，但仍受训练数据分布限制；推理速度和资源消耗相对传统优化器尚不具优势。

---

## 507. Neural Optimal Transport in Hilbert Spaces: Characterizing Spurious Solutions and Gaussian Smoothing

**arXiv ID:** 2602.14086 | [PDF](https://arxiv.org/pdf/2602.14086v1)

**作者:** Jae-Hwan Choi `[一作]` (Korea Institute for Advanced Study), Jaewoong Choi `[通讯]` (Sungkyunkwan University)

**通讯引用:** 1635 | [OpenAlex ID](https://openalex.org/A5101792384)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在无限维Hilbert空间中研究半对偶神经最优传输（SNOT）框架，解析并消除了脉冲解问题，并提出通过高斯平滑及Annealing实现正则化的全新方法。

**💡 创新点**

创新点包括：① 将正则测度（Generalized Lebesgue绝对连续性）引入Hilbert空间下的OT，证明半对偶SNOT可行并能唯一确定Monge映射；② 给出高斯平滑的必要且充分条件，阐明协方差算子核与源测度奇异方向的匹配关系；③ 提出HiSNOT框架并通过逐步退火高斯噪声实现对真实数据的有效训练。

**🔧 技术方法**

主要技术包括：半对偶神经OT、正则测度理论、Hilbert空间上Gaussian平滑（卷积、谱噪声注入）、傅里叶神经算子、Annealing策略及相关的泛函分析与Gâteaux变分。

**📊 数据集**

实验数据集：合成功能数据（Perpendicular、One-to-Many、Grid等），以及真实时间序列缺失补全数据集ETTh1/2、ETTm1/2、Exchange。

**📈 对比分析**

与Transformer、DLinear、TimesNet、FreTS、PatchTST、SCINet、iTransformer、SAITS、CSDI、Sinkhorn、TDM、PWS-I等基线对比，HiSNOT在MSE/MAE指标上多项任务取得SOTA或接近SOTA，尤其在Exchange数据集的MSE仅0.004，远优于第二佳PWS-I（0.036）。

**⚠️ 局限性**

局限性：收敛证明仅在子列上成立；当最优计划非唯一时可能出现振荡；高斯核需覆盖源测度所有奇异方向，若未满足则仍可能产生脉冲解；实验范围主要集中在时间序列，尚未在更广泛的高维功能数据上验证。

---

## 508. Cross-view Domain Generalization via Geometric Consistency for LiDAR Semantic Segmentation

**arXiv ID:** 2602.14525 | [PDF](https://arxiv.org/pdf/2602.14525v1)

**作者:** Jindong Zhao `[一作]` (Changsha University of Science and Technology), Shaobo Xia `[通讯]` (Changsha University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了跨视角域泛化（Cross‑View Domain Generalization）框架 CVGC，用于在不同 LiDAR 采集平台间实现语义分割模型的无目标域泛化。

**💡 创新点**

创新点在于①设计了视角无关的密度重采样与视角依赖的可见性模拟，实现多视角几何增强；②通过几何一致性正则化（Geometric Consistency Regularization）在多视角视图间强制语义和占据预测一致，学习视角不变表征。

**🔧 技术方法**

采用的技术包括：密度增稠与稀疏化、球面投影可见性仿真、MinkUNet34 主干网络、稀疏卷积占据预测头、距离加权 KNN 体素特征聚合、二分类交叉熵与语义损失联合训练。

**📊 数据集**

使用六个公开 LiDAR 数据集：H3D（UAV）、Paris‑Lille‑3D（车辆）、ISPRS Vaihingen（航空）、STPLS3D（合成 UAV）、Toronto‑3D（车辆）和 DALES（航空），构建两组一源两目标的跨视角泛化基准。

**📈 对比分析**

与 source‑only、CosMix、PolarMix、DGLSS、UniMix 等方法对比，CVGC 在所有目标域均实现 mIoU 提升，最高可达 ISPRS Vaihingen 上 55.18%（+23.98%）与 STPLS3D→DALES 上 64.42%（+5.85%），明显优于现有域泛化与域自适应方法。

**⚠️ 局限性**

局限性包括：对某些类别（如 Urban Furniture、Pole）的性能提升有限，主要受标签不一致与视角导致的极端遮挡影响；目前仅在同一场景内实现几何一致性，未覆盖跨场景结构不变性；未来需引入更细粒度的传感器建模以进一步提升泛化能力。

---

## 509. Reverse N-Wise Output-Oriented Testing for AI/ML and Quantum Computing Systems

**arXiv ID:** 2602.14275 | [PDF](https://arxiv.org/pdf/2602.14275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 510. Probing Human Articulatory Constraints in End-to-End TTS with Reverse and Mismatched Speech-Text Directions

**arXiv ID:** 2602.14664 | [PDF](https://arxiv.org/pdf/2602.14664v1)

**作者:** Parth Khadse `[一作]` (Tata Consultancy Services Limited), Sunil Kumar Kopparapu `[通讯]` (Tata Consultancy Services Limited)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在对文本到语音（TTS）的端到端系统进行实验，比较前向文本-前向语音、反向文本-反向语音以及前向文本-正向语音三种输入组合的训练效果。

**💡 创新点**

发现反向文本-反向语音训练的TTS模型在语音识别（WER/CER）和人类感知（MOS）上均优于传统的正向文本-正向语音模型，提示人工语音的肌肉运动约束并未限制端到端学习，反而可能提供某些隐含的自然性特征。

**🔧 技术方法**

采用了两类端到端架构：自回归(seq2seq+注意力)和非自回归(VITS/Variational Inference TTS)，并进一步对输入进行字符级和BPE级别的分词处理；使用预训练的vocoder生成音频；评估使用Whisper与WhisperX ASR模型计算WER/CER，并进行5名受试者的MOS及成对听感测试。

**📊 数据集**

数据集为单位男性说话人的LIMMITS'25挑战赛音频，约40小时录音（48 kHz），经过22.05 kHz下采样和mel谱图提取；用于评估的测试句子来自Librispeech‑960，筛选后共1701条长句子。

**📈 对比分析**

比较方法：通过WER/CER量化自动识别准确率，通过MOS评估自然度与可懂度，以及成对听感测试比较两模型音频。结果显示，反向文本-反向语音模型在WER/CER上分别提升约5–6 %（绝对）/约35 %（相对），MOS得分提升约0.35分（自然度）和0.20分（可懂度），成对测试中92 %受试者更倾向于该模型的音频。

**⚠️ 局限性**

局限性包括：仅使用读音数据，缺乏多说话人和自然口语的测试；反向语音为人工时间反转，未必能代表真实倒语；人类听感实验样本量仅5人；实验仅验证了数据驱动性而未深入探究为何逆向组合表现更佳。

---

## 511. Gaussian Mesh Renderer for Lightweight Differentiable Rendering

**arXiv ID:** 2602.14493 | [PDF](https://arxiv.org/pdf/2602.14493v1)

**作者:** Xinpeng Liu `[一作]` (Osaka University), Fumio Okura `[通讯]` (Osaka University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了一种轻量级可微网格渲染器（Gaussian Mesh Renderer, GMR），将网格三角面解析为高斯原语，实现端到端的几何优化。

**💡 创新点**

创新点在于将3D Gaussian Splatting的高效光栅化与网格表示紧密结合，通过解析网格到平面高斯转换获得更平滑的梯度，显著提升小批量、低内存场景下的稳定性与精度。

**🔧 技术方法**

使用了3D Gaussian Splatting（3DGS）渲染器、解析网格到高斯的转换公式、PyTorch + gsplat实现、VectorAdam优化器，以及Chamfer、Normal Consistency、PSNR、SSIM、LPIPS等评估指标。

**📊 数据集**

使用Common 3D Test Models和Objaverse共17个物体，生成每个模型253视角的渲染图像作为训练/评估数据。

**📈 对比分析**

与SoftRas、Nvdiffrast以及3DGS基础方法对比，GMR在batch 1/10的设置下实现了几乎最优的几何精度（Chamfer≈1.5e-5），内存占用比Nvdiffrast低30%，速度比SoftRas快约40%；但在大批量（>50）下精度和速度略逊于Nvdiffrast。

**⚠️ 局限性**

主要限制是当前实现仍缺乏GPU加速的网格到高斯转换步骤，导致整体速度低于高度优化的Nvdiffrast；在大批量场景下，GMR的性能和精度均不及Nvdiffrast，未来需开发CUDA实现以提升吞吐量。

---

## 512. LoPace: A Lossless Optimized Prompt Accurate Compression Engine for Large Language Model Applications

**arXiv ID:** 2602.13266 | [PDF](https://arxiv.org/pdf/2602.13266v1)

**作者:** Aman Ulla `[一作]` `[通讯]`, Aman Ulla

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

LoPace是一种面向LLM提示存储的无损压缩引擎，提供Zstandard、BPE token化和两者混合三种压缩模式；

**💡 创新点**

其创新点在于将BPE子词编码与Zstandard的字节级压缩相结合，形成多阶段压缩链，实现乘法级压缩增益；

**🔧 技术方法**

实现技术包括BPE token化（tiktoken）、二进制固定宽度打包（uint16/uint32）以及Zstandard压缩算法；

**📊 数据集**

实验数据集为Hugging Face的philschmid/markdown-docs-transformers共386条提示，涵盖代码、Markdown和文本；

**📈 对比分析**

与单独使用Zstandard或Token化相比，混合方法平均压缩比4.89×、空间节省72.2%，吞吐量约3.3 MB/s，解压速度最高；

**⚠️ 局限性**

主要局限包括对tokenizer版本的依赖、固定宽度编码导致的额外开销、缺乏字典训练和与其他基线压缩（Brotli、gzip等）的对比。

---

## 513. Long Context, Less Focus: A Scaling Gap in LLMs Revealed through Privacy and Personalization

**arXiv ID:** 2602.15028 | [PDF](https://arxiv.org/pdf/2602.15028v1)

**作者:** Shangding Gu `[一作]` `[通讯]` (University of California), Shangding Gu (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 PAPerBench 的大规模长上下文隐私与个性化评估基准，并在其中系统评测多种主流 LLM 的表现，揭示了“长上下文易失焦”这一规模化缺口。

**💡 创新点**

创新点包括：①首次在同一基准中同时量化个性化质量与隐私泄露；②覆盖 1K–256K tokens 的长上下文范围；③通过实验发现两类任务随上下文长度增长均出现性能下降；④提供软注意力稀释的理论解释，说明现有 Transformer 架构的根本瓶颈。

**🔧 技术方法**

技术手段主要有：基于 Transformer 的软注意力分析；使用 Qwen3‑235B 进行上下文扩展与隐私注入；多任务（个性化 + 多选隐私）评估框架；对比多模型的上下文长度对性能的影响；理论上推导注意力稀释对信息传递的影响。

**📊 数据集**

数据集：PAPERBench，约 29K 例子、377K 题目，包含 1K、4K、8K、16K、32K、64K、128K、256K 4 字节 token 长度的背景，包含 7 类 PII（手机号、邮箱、地址等），每个实例含用户背景、模糊查询及多选答案。

**📈 对比分析**

评估方法：对 Gemini‑3‑flash、Claude‑haiku‑4.5、GPT‑5.2、Mistral‑123B‑2512、Qwen3‑235B、Llama‑3.3‑70B、Llama‑4‑Scout‑109B、Qwen2.5‑14B 等模型在不同上下文长度下的个性化准确率与隐私准确率进行对比。结果显示，所有模型在上下文加长时性能均下降，规模越大下降越缓慢，体现了规模化缺口。

**⚠️ 局限性**

限制：①仅评估固定容量 Transformer，未尝试检索、稀疏注意力等改进；②理论分析假设简化，未覆盖所有实际情况；③实验模型有限，缺少更多 LLM 的验证；④长上下文生成成本高，数据集构建对模型的依赖较大；⑤隐私保护机制仍以简单掩码与伪造 PII 为主，实际应用需更强安全保障。

---

## 514. Bengali-Loop: Community Benchmarks for Long-Form Bangla ASR and Speaker Diarization

**arXiv ID:** 2602.14291 | [PDF](https://arxiv.org/pdf/2602.14291v1)

**作者:** H. M. Shadman Tabib `[一作]` (Bangladesh University of Engineering and Technology), Shahriar Kabir `[通讯]` (Bangladesh University of Engineering and Technology)

**通讯引用:** 353 | [OpenAlex ID](https://openalex.org/A5042343776)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

Bengali-Loop提供了两套社区基准：一套包含191条长录音（总计158.6小时、792k词）的高质量手工校正Bengali ASR数据集，另一套包含24条长录音（总计22小时、5,744段）的全手工说话人分离数据集。

**💡 创新点**

创新点在于构建可复现的字幕提取与人工校正流水线，生成长形式Bengali语料；同时定义统一的评估规范（WER/CER、DER）和标注规则（包括重叠处理），为社区提供可比对的、可复现的基准资源。

**🔧 技术方法**

使用的技术包括字幕解析与清洗、音频标准化至16kHz mono WAV、手工校对、Unicode标准化、WER/CER和DER评估；基线模型涵盖Whisper变体Tugstugi、Hishab TITU-BN；说话人分离基线包括pyannote.audio预训练管线以及基于Silero/WebRTC VAD + ECAPA-embedding + 层次聚类的轻量级方案。

**📊 数据集**

所用数据集为：191条来自11个YouTube频道的长录音（158.6h、792,491词，36k词表）；24条手工标注的说话人分离录音（22h、5,744段，平均每录音约16位说话人）。

**📈 对比分析**

比较方法：ASR通过WER/CER，DER通过0.25s碰撞、最优映射和单标签重叠策略评估；基线性能为Tugstugi WER 34.07%、CER 16.44%；Hishab TITU-BN WER 50.67%、CER 21.99%；说话人分离基线pyannote.audio DER 40.08%，轻量级方案DER分别为61.50%（Silero VAD）和73.71%（WebRTC VAD）。

**⚠️ 局限性**

限制：方言多样性有限；重叠段仅归属首个说话人，未细粒度标注；域覆盖（戏剧、新闻、短片）不完整；长文本解码策略对WER影响大；部分媒体受版权限制，导致数据分发受限。

---

## 515. Synthetic Dataset Generation and Validation for Robotic Surgery Instrument Segmentation

**arXiv ID:** 2602.13844 | [PDF](https://arxiv.org/pdf/2602.13844v1)

**作者:** Giorgio Chiesa `[一作]` (University of Turin), Marco Grangetto `[通讯]` (University of Turin)

**通讯引用:** 3017 | [OpenAlex ID](https://openalex.org/A5079141778)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并验证了一套基于 Python–Maya 自动化管线的 Da Vinci 机器人手术工具合成数据集，用于工具分割任务。

**💡 创新点**

创新点在于将 3D 摄影测量、全自动动画生成、随机光照与血液纹理以及实时标签渲染相结合，形成可控且隐私友好的高保真合成数据；并通过混合真实/合成比例实验揭示两者平衡能显著提升模型泛化。

**🔧 技术方法**

使用 Python、Autodesk Maya、Photogrammetry、Maya 生成物体 ID 掩码、UNet 深度网络（ResNet18/VGG/ResNeXt 50）、Adam 优化器、Dice 损失和多模态数据增强等技术。

**📊 数据集**

数据集包括 1,000 张手术现场真实图像（手工标注）和数千帧合成图像，合成图像通过管线生成并提供像素级分割掩码。

**📈 对比分析**

通过在训练集中按比例混合真实与合成样本，对 UNet 进行训练，并在 100% 真实的验证/测试集上评估，发现当合成比例为 50% 时，IoU 从约 0.4 提升至 0.8，说明平衡混合可显著提升性能。

**⚠️ 局限性**

局限性包括真实数据量有限、仅复现三种工具、过度依赖合成数据可能导致域迁移失效，且现有管线需进一步扩展以支持更多工具与更复杂任务。

---

## 516. Responsible AI in Business

**arXiv ID:** 2602.13244 | [PDF](https://arxiv.org/pdf/2602.13244v1)

**作者:** Stephan Sandfuchs `[一作]`, Jörg Frochte `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一套面向中小企业的责任AI治理框架，涵盖法律合规、可解释性、绿色可持续性以及本地化部署等关键维度，帮助企业在引入AI时实现价值创造与风险可控的双重目标。

**💡 创新点**

创新点在于将欧盟AI法案、可解释AI、绿色AI与本地模型技术有机结合，形成一种模块化的、可落地的治理方案，并通过风险分类与角色划分为决策者提供具体的实施路径。

**🔧 技术方法**

主要技术包括风险评估与文档化（符合AI法案要求）、可解释AI方法（全局与局部解释、数据偏差分析）、绿色AI指标与资源优化（模型压缩、LoRA迁移学习、持续学习）、以及本地/边缘部署架构与数据主权实现。

**📊 数据集**

本文未使用具体实验数据集，而是基于公开模型（如Llama、Mistral）和行业案例的经验做法，结合公开报告与法规文本进行理论阐述。

**📈 对比分析**

对比方法主要通过将AI应用按风险等级（高风险、低风险、禁止）和角色（提供者、部署者）进行分类，并给出相应的合规与技术措施，缺乏量化性能指标，但提供了从合规到技术落地的完整流程。

**⚠️ 局限性**

局限性包括缺乏实证验证与量化评估、对细节实现（如模型压缩阈值、LoRA参数选择）缺少统一标准、以及在不同规模企业环境中的可迁移性与成本评估仍待进一步研究。

---

## 517. Truly Adapting to Adversarial Constraints in Constrained MABs

**arXiv ID:** 2602.14543 | [PDF](https://arxiv.org/pdf/2602.14543v1)

**作者:** Francesco Emanuele Stradi `[一作]` (Politecnico di Milano), Nicola Gatti `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了受约束的多臂老虎机（MAB）问题，旨在最小化学习过程中的总损失，同时控制多个未知约束的违反情况，考虑了非平稳环境下的全反馈和带反馈情况。

**💡 创新点**

首次提出了在约束为随机时，损失可以任意变化的情况下，达到最优的后悔率和正约束违反率的算法，并且在约束的对抗性程度变化时，保证逐渐降低。

**🔧 技术方法**

使用了在线镜面下降（Online Mirror Descent）和固定份额更新等技术，结合乐观原则构建近似可行决策集。

**📊 数据集**

未具体提及使用的数据集，但研究涉及的环境是非平稳的，损失和约束的分布可以随时间任意变化。

**📈 对比分析**

与现有最先进的方法相比，本文的算法在随机约束情况下的后悔和违反界限为𝒪(√(T)+C)，在对抗约束情况下也提供了保证，优于现有方法在对抗约束下的表现。

**⚠️ 局限性**

限制在于算法在对抗损失情况下的表现可能不如在随机约束情况下的表现，且在带反馈情况下的复杂性增加。

---

## 518. Bounding Probabilities of Causation with Partial Causal Diagrams

**arXiv ID:** 2602.14503 | [PDF](https://arxiv.org/pdf/2602.14503v1)

**作者:** Yuxuan Xie `[一作]` (University of Ottawa), Ang Li `[通讯]` (Florida State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

通过将部分因果结构信息编码为对因果反事实分布的约束，构建了一个优化框架，用来给定个体层面因果概率（如 PNS、PN、PS）提供更紧的上下界。

**💡 创新点**

创新点包括：① 统一将不完整的因果图、协变量信息、介导变量信息等作为线性或非线性约束加入优化；② 能处理多值处理和结果变量，而非仅二元；③ 在缺乏联合协变量分布时仍可利用各协变量的单独信息；④ 通过与现有 Tian–Pearl、Mueller–Li–Pearl 等方法对比，验证了框架在多种设置下的优越性。

**🔧 技术方法**

使用的技术主要是线性（或非线性）规划：把 PNS、PN、PS 视为目标函数，约束由观测/实验分布、无后代协变量、后门准则、介导变量的条件独立性等构成；在包含介导变量时引入乘积等非线性等式。

**📊 数据集**

实验数据完全是仿真产生的：根据不同因果图（多协变量、后门+介导变量等），随机生成 1000 个满足这些图的可兼容分布，随后计算实验分布与观测分布。

**📈 对比分析**

与 Tian–Pearl、Mueller–Li–Pearl 基线进行比较：在多协变量情形下，平均 PNS 下界提升约 0.02–0.04，上界下降相同幅度，导致区间宽度明显收窄；对比后门+介导情形，平均区间宽度下降约 0.05，且在大多数样本上优于两种基线，改进比例超过 70%。

**⚠️ 局限性**

主要局限是计算复杂度：当加入介导变量约束时出现非凸非线性等式，求解需使用非线性规划，易陷入局部最优且随变量维度增长时间显著增加。未来工作需要设计更简化或可线性化的介导信息表述以保持可计算性。

---

## 519. Singular Vectors of Attention Heads Align with Features

**arXiv ID:** 2602.13524 | [PDF](https://arxiv.org/pdf/2602.13524v1)

**作者:** Gabriel Franco `[一作]` (Boston University), Mark Crovella `[通讯]` (Boston University)

**通讯引用:** 20453 | [OpenAlex ID](https://openalex.org/A5064525211)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并证明注意力头的奇异向量与模型中使用的特征在满足一定线性和低干扰条件时会对齐，并通过 toy 模型和 GPT‑2、Pythia‑160M 的实验验证稀疏注意力分解现象。

**💡 创新点**

提出了可理论化的 Singular‑Vector‑Feature (SVF) 对齐条件，并将其转化为可观测的稀疏注意力分解 (SAD) 预测；同时揭示多特征对齐时的正交化机制。

**🔧 技术方法**

利用线性代数工具（SVD、奇异值分解）分析 QK 矩阵；设计 toy 自动编码器 + 注意力头训练框架；在 GPT‑2 与 Pythia 上进行前向推理并计算相对注意力的 SVD 分解。

**📊 数据集**

使用自生成的 toy 数据；GPT‑2 预训练模型；Pythia‑160M 多个检查点；评估任务采用 Indirect Object Identification (IOI) 语料。

**📈 对比分析**

通过与低秩矩阵、随机旋转基底的对比实验，证明稀疏性不是低秩导致；使用 S(v) 与 N_recon 指标量化稀疏度，实验显示训练后稀疏度显著提升，表明 SVF 对齐在实模型中有效。

**⚠️ 局限性**

研究仅针对单一注意力头，未探讨特征数超过 head 维度时的冲突；缺乏对多头协同使用及 feature anisotropy 对对齐影响的深入分析。

---

## 520. Replicable Constrained Bandits

**arXiv ID:** 2602.14580 | [PDF](https://arxiv.org/pdf/2602.14580v1)

**作者:** Matteo Bollini `[一作]` (Politecnico di Milano), Alberto Marchesi `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了在受限多臂赌博机（MAB）问题中实现算法可复制性，提出了一种新的在线学习算法，旨在在满足多个约束的同时最大化奖励。

**💡 创新点**

创新点在于设计了可复制的算法，使其在受限MAB中实现的后悔和约束违反与非可复制算法相匹配，且首次提出了基于UCB的可复制算法。

**🔧 技术方法**

使用了动态周期基础的UCB风格算法，结合了可复制估计器和置信区间来进行决策。

**📊 数据集**

使用了随机受限MAB的环境，具体数据集未明确提及，但涉及多个约束和奖励分布。

**📈 对比分析**

与现有的非可复制算法相比，提出的算法在后悔和约束违反方面表现出相似的性能，且在软约束和硬约束设置下均能达到次线性后悔和约束违反。

**⚠️ 局限性**

限制在于算法的可复制性依赖于内部随机源的固定，且在处理复杂约束时可能面临挑战。

---

## 521. Muscle Coactivation in the Sky: Geometry and Pareto Optimality of Energy vs. Promptness in Multirotors

**arXiv ID:** 2602.14222 | [PDF](https://arxiv.org/pdf/2602.14222v1)

**作者:** Antonio Franchi `[一作]` (University of Twente), Antonio Franchi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 8165 | [OpenAlex ID](https://openalex.org/A5001771133)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了多旋翼飞行器在有符号二次推力模型下的冗余分配，阐明能量消耗与动态机动性（promptness）之间的固有权衡，并给出基于纤维束的多目标优化框架。

**💡 创新点**

创新点在于：① 将多旋翼冗余解析为纤维束几何问题；② 引入 promptness（负对数行列式的操纵性障碍）与能量（L1.5）共同优化；③ 展示纤维几何与硬件约束如何决定两目标的兼容性或冲突。

**🔧 技术方法**

采用了非线性推力模型、差分映射与对数行列式对数变换、内部点法求解多目标优化、以及几何纤维分析（张量投影、KKT 条件）。

**📊 数据集**

使用仿真数据：对一架 6‑旋翼（六旋翼）在不同扭矩/推力指令下的分配，模拟能量与 promptness 的取值曲线。

**📈 对比分析**

通过绘制能量–promptness 的 Pareto 曲线、比较单目标最优与双目标最优的差异，发现单向推力下 Pareto 曲线短且内点较为集中，双向可逆推力下曲线拉长且 promptness 最优趋向硬件极限，验证了纤维几何决定权衡的理论。

**⚠️ 局限性**

局限在于：① 仅基于仿真，未在真实硬件上验证；② 对于更复杂的多自由度或不对称布局需进一步分析；③ 计算上多目标最优求解仍较为耗时，尤其在可逆推力下需要多种初始猜测。

---

## 522. Bridging the Urban Divide: Adaptive Cross-City Learning for Disaster Sentiment Understanding

**arXiv ID:** 2602.14352 | [PDF](https://arxiv.org/pdf/2602.14352v1)

**作者:** Zihui Ma `[一作]` (New York University), Yuki Miura `[通讯]` (New York University)

**通讯引用:** 1257 | [OpenAlex ID](https://openalex.org/A5101450159)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于跨城市学习框架，将社交媒体文本与移动行为特征融合，提升灾难情绪分析的公平性与准确性。

**💡 创新点**

创新点在于结合城市相似度的数据增广与多模态融合，弥补城市间数据不均衡和文本偏见。

**🔧 技术方法**

采用Transformer文本编码、轻量级多模态融合网络、基于城市属性的对比学习和基于移动轨迹的行为编码。

**📊 数据集**

使用加利福尼亚州2025年圣地亚哥火灾相关推文、Spectus GPS轨迹和美国ACS及FEMA风险数据。

**📈 对比分析**

与VADER、RoBERTa、Mistral等基线对比，城市特定融合模型在准确率0.772、F1 0.592上实现最优。

**⚠️ 局限性**

局限在于对非城市化地区仍可能缺乏足够标签、依赖推测位置、模型复杂度高且难以实时部署。

---

## 523. Symmetry-Aware Fusion of Vision and Tactile Sensing via Bilateral Force Priors for Robotic Manipulation

**arXiv ID:** 2602.13689 | [PDF](https://arxiv.org/pdf/2602.13689v1)

**作者:** Wonju Lee `[一作]` (Analog Devices Inc), Tao Yu `[通讯]` (Analog Devices Inc)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在机器人插槽装配任务中提出了一种跨模态 Transformer（CMT）与物理信息正则化相结合的视觉-触觉融合框架，解决了传统视觉-触觉融合中同构难题并提升插入成功率。

**💡 创新点**

创新点在于（1）引入对称力平衡的物理信息正则化，使触觉嵌入更稳健；（2）通过层级自注意力与跨模态注意力实现结构化融合，克服了传统拼接和门控融合的信号混淆问题。

**🔧 技术方法**

使用的技术包括卷积编码器、SoftArgMax 关键点化表示、层级自注意力、跨模态 Transformer、PPO 强化学习以及对称力正则化的辅助损失。

**📊 数据集**

主要使用 TacSL 插槽插入基准数据集，并在附录中扩展到螺纹旋转任务（screw task）进行评估。

**📈 对比分析**

与传统的拼接、门控融合以及“手腕+接触力”特权配置对比，CMT+对称正则化在 TacSL 插入任务中达到了 96.59% 的成功率，略高于特权配置（96.09%），显著优于基线方法。

**⚠️ 局限性**

局限性包括：仅在对称插槽插入任务中验证，异形物体的泛化尚未充分评估；正则化依赖事先校准；对传感器噪声与动态环境的鲁棒性仍有待进一步研究。

---

## 524. GLIMPSE : Real-Time Text Recognition and Contextual Understanding for VQA in Wearables

**arXiv ID:** 2602.13479 | [PDF](https://arxiv.org/pdf/2602.13479v1)

**作者:** Akhil Ramachandran `[一作]` (Meta Reality Labs), Peyman Heidari `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种混合架构，在可穿戴设备上实现高分辨率文本识别与低分辨率视频流的分离，结合智能帧选择、OCR会话管理与视频LLM，实现实时、上下文感知的视觉问答系统。

**💡 创新点**

核心创新包括：① 异步分辨率解耦，将高分辨率OCR仅用于重要帧，低分辨率视频用于视觉上下文；② 三阶段智能帧选择（模糊检测、ROI/文本检测、相似性过滤）将OCR工作量压缩至 67.7%；③ OCR 会话管理器对稀疏文本结果进行时间对齐、聚合与去重，确保模型获取连续、最新的文本信息。

**🔧 技术方法**

技术手段：基于决策树的模糊检测、轻量化多任务ROI/文本检测模型、FAISS相似性搜索、MoE 视频多模态大语言模型、热能与功耗监测、Prompt 生成与缓存机制。

**📊 数据集**

使用自研的 112 条可穿戴视频数据集，涵盖 208 条问答对，覆盖短文本朗读、翻译、长文本朗读、分析与数学推理五类任务；实验对比亦参考 TextVQA、DocVQA 等公开基准。

**📈 对比分析**

与基线进行对比：Server‑Full（12MP/30fps，1.0x功耗）准确率 74%；Server‑Low（3MP/2fps，0.49x功耗）准确率 41%；本系统在相同 0.49x 功耗下实现 72% 的准确率，基本等同于全分辨率方案，且显著降低能耗。

**⚠️ 局限性**

局限性：评估样本量仅 208 条，未覆盖多语言和手写文字；仅验证 Latin‑script，缺乏在大型公开数据集上的泛化验证；对极端环境（低光、快速运动）下文本识别的鲁棒性仍待进一步评估。

---

## 525. Cumulative Utility Parity for Fair Federated Learning under Intermittent Client Participation

**arXiv ID:** 2602.13651 | [PDF](https://arxiv.org/pdf/2602.13651v1)

**作者:** Stefan Behfar `[一作]` (University of Cambridge), Richard Mortier `[通讯]` (University of Cambridge)

**通讯引用:** 6551 | [OpenAlex ID](https://openalex.org/A5043629070)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了累计效用公平（Cumulative Utility Parity）机制，针对间歇性客户端参与的联邦学习公平性问题。

**💡 创新点**

创新点在于：①用可用性归一化累计效用衡量长期公平；②采用逆可用性采样和自适应加权平衡参与频率；③引入可持续代理更新补偿暂时不可用客户端的贡献。

**🔧 技术方法**

技术包括：统计可用性估计、逆可用性权重采样、累计效用跟踪、代理（surrogate）更新、理论分析（Lemma/Theorem）证明公平性收敛。

**📊 数据集**

数据集：CIFAR‑10（非IID、标签不均匀划分）以及真实移动设备可用性轨迹，用于模拟实际间歇性参与场景。

**📈 对比分析**

与 q‑FFL、PHP‑FL 等基准对比，平均精度提升至约 80% 以上，累计效用 CV 下降至 0.19，Jain 指数接近 0.975，选择间隙和 Gini 系数显著降低，说明长期代表性和公平性都有显著提升。

**⚠️ 局限性**

局限性：假设可用性过程是平稳且可估计的；对剧烈波动或攻击性停机（burst/dropout）尚未充分评估；代理更新依赖于先前模型，可能在极端数据漂移时失效。

---

## 526. A Unified Physics-Informed Neural Network for Modeling Coupled Electro- and Elastodynamic Wave Propagation Using Three-Stage Loss Optimization

**arXiv ID:** 2602.13811 | [PDF](https://arxiv.org/pdf/2602.13811v1)

**作者:** Suhas Suresh Bharadwaj `[一作]` (BITS Pilani), Reuben Thomas Thovelil `[通讯]` (BITS Pilani)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用物理信息神经网络（PINN）解决一维耦合电-弹性波传播的线性压电问题，预测位移和电势；

**💡 创新点**

创新点在于采用三阶段优化（Adam→AdamW→L‑BFGS）与硬约束基函数实现边界/初始条件，构建统一的耦合物理网络；

**🔧 技术方法**

技术包括全连接深度网络、自动微分、加权残差损失、随机采样协同点、硬约束变换和多阶段优化；

**📊 数据集**

数据集为基于解析解的合成样本，随机生成空间-时间协同点（20,000个内点、5,000个边界点、5,000个初始点）；

**📈 对比分析**

与传统有限元方法对比，PINN在200,000点评估网格上得到相对L₂误差约2.34%（位移）和4.87%（电势），在同类PINN研究中属于较佳水平；

**⚠️ 局限性**

局限性包括误差随时间累积、对电势场的误差放大（耦合导致），以及对长时间和高频振荡的鲁棒性不足。

---

## 527. AdaVBoost: Mitigating Hallucinations in LVLMs via Token-Level Adaptive Visual Attention Boosting

**arXiv ID:** 2602.13600 | [PDF](https://arxiv.org/pdf/2602.13600v1)

**作者:** Jiacheng Zhang `[一作]` (University of Melbourne), Tianyu Pang `[通讯]` (Sea AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Token级视觉注意力增强框架AdaVBoost，通过自适应调整注意力提升来减少大型视觉语言模型的幻觉。

**💡 创新点**

创新点在于引入视觉根据信息熵VGE作为幻觉风险估计，并在每个生成步骤动态调节视觉注意力和文本抑制，既能加强视觉信号又能抑制无关文本。

**🔧 技术方法**

采用视觉根据信息熵VGE、Token级自适应视觉注意力调制、文本抑制、Transformer自注意力调节等技术，全部在推理阶段实现，无需额外训练。

**📊 数据集**

使用四大幻觉基准：CHAIR、SHR、POPE、AMBER。

**📈 对比分析**

与PAI、VAF、VGA等方法对比，在LLaVA-NeXT、Qwen3-VL、InternVL3.5等三大LVLM上，在幻觉率（CHAIRs、SHR等）上显著下降且生成质量基本保持不变，尤其在CHAIR、SHR指标上表现突出。

**⚠️ 局限性**

局限性包括对单词级离散任务（如VQA）的改进有限、过强的视觉抑制或文本抑制可能导致生成质量下降，以及对VGE估计的依赖可能在极端情况下失效。

---

## 528. Chain-of-Thought Reasoning with Large Language Models for Clinical Alzheimer's Disease Assessment and Diagnosis

**arXiv ID:** 2602.13979 | [PDF](https://arxiv.org/pdf/2602.13979v1)

**作者:** Tongze Zhang `[一作]` (Stevens Institute of Technology), Sang Won Bae `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 2102 | [OpenAlex ID](https://openalex.org/A5067613275)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于大语言模型的链式推理（CoT）框架，用于从电子健康记录（EHR）中进行阿尔茨海默病（AD）临床分级（CDR）评估

**💡 创新点**

创新点在于：①采用多阶段CoT生成并融合四条独立推理路径，以增强诊断透明度与稳定性；②将推理结果转化为可审计的JSON结构，进一步提升可解释性；③通过对齐逻辑校验与阈值裁剪保证最终评分的可靠性

**🔧 技术方法**

使用了大语言模型（Qwen2‑7B、Qwen3‑4B、Phi‑3B）配合CoT提示工程、随机种子多样化推理、文本整合与正则表达式提取等技术

**📊 数据集**

使用了从Stevens Institute of Technology与National Yang‑Ming Chiao‑Tung University收集的5年临床EHR数据，共698条完整记录，包含主诉与评估文本，标注为CDR 0.5/1.0/2.0/3.0四级

**📈 对比分析**

与零样本直接提示方法对比，CoT框架在四个二分类任务（0.5vs1.0、0.5vs2.0、0.5vs3.0、1.0vs3.0）均提升了F1分数，最高提升约15%（Qwen2‑7B从0.39提升至0.54）

**⚠️ 局限性**

局限性包括：样本量有限；仅基于文本EHR，未融合影像数据；缺乏外部多中心验证；CoT推理计算成本相对较高

---

## 529. Enhancing spatial hearing with cochlear implants: exploring the role of AI, multimodal interaction and perceptual training

**arXiv ID:** 2602.13787 | [PDF](https://arxiv.org/pdf/2602.13787v1)

**作者:** Lorenzo Picinali `[一作]` (Imperial College London), Christoph Braun `[通讯]` (University of Tübingen)

**通讯引用:** 14590 | [OpenAlex ID](https://openalex.org/A5041365220)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出针对植入耳蜗患者空间听觉不足的多维度研究计划，包括技术升级、康复训练与人工智能辅助方案；

**💡 创新点**

整合深度学习模型与生理耳蜗仿真、双耳同步化与多感官VR/AR训练，构建个性化空间听觉恢复路径；

**🔧 技术方法**

使用深度循环神经网络、EEG生理记录、适应性滤波、超快双耳通信、虚拟/增强现实与音触觉训练系统；

**📊 数据集**

未公开具体数据集，计划收集CI患者的心理声学测试、EEG记录及模拟听觉数据进行模型训练；

**📈 对比分析**

目前仅为设想阶段，未进行实验对比，计划通过听觉定位实验与EEG对照评估方案有效性；

**⚠️ 局限性**

缺乏实际验证与性能评估、实施复杂度高、患者个体差异大、需要统一评估标准与进一步技术集成研究。

---

## 530. Locally Adaptive Multi-Objective Learning

**arXiv ID:** 2602.14952 | [PDF](https://arxiv.org/pdf/2602.14952v1)

**作者:** Jivat Neet Kaur `[一作]` (University of California), Michael I. Jordan `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种局部自适应多目标学习算法，在在线环境中同时满足多重评估指标并适应分布漂移；

**💡 创新点**

创新点在于将自适应在线学习方法（如 Fixed‑Share）嵌入到多目标框架中，取代传统的 Hedge 更新，从而在任意局部时间区间内实现最优的多目标误差上界；

**🔧 技术方法**

主要技术包括两阶段极小极大学习（minimax）策略、固定分享（Fixed‑Share）权重更新、以及针对多准确定量（multi‑accuracy）与预测误差的联合损失；

**📊 数据集**

实验使用了 GEFCom2014 电力负荷预测数据（以温度分组的多准确定量任务）和 COMPAS 犯罪重犯率预测数据（按族群分组的多准确定量任务）；

**📈 对比分析**

与基线（非自适应 Hedge、基线预测、在线多校准）以及改进的“自适应目标”版本对比，实验显示局部自适应算法在所有局部窗口上实现了接近零的多准确定量误差，并在保持或提升预测精度的同时显著优于非自适应方法；

**⚠️ 局限性**

局限性包括：仅在两类任务上验证，未覆盖更广泛的多目标场景；需要手动设定窗口宽度和超参数；在高维或极大数据流下的计算与内存成本尚未系统评估。

---

## 531. UAV-SEAD: State Estimation Anomaly Dataset for UAVs

**arXiv ID:** 2602.13900 | [PDF](https://arxiv.org/pdf/2602.13900v1)

**作者:** Aykut Kabaoglu `[一作]` (Istanbul Technical University), Sanem Sariel `[通讯]` (Istanbul Technical University)

**通讯引用:** 853 | [OpenAlex ID](https://openalex.org/A5067273251)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了一个包含1396条真实飞行记录、52小时飞行时长的UAV状态估计异常数据集UAV-SEAD，并提出了机械电气、外部位置、全局位置、海拔四类异常分类。

**💡 创新点**

创新点在于提供第一批大规模、无人工注入的真实UAV状态估计异常数据，并提出结构化的四类异常分类框架。

**🔧 技术方法**

采用PX4 EKF与多源传感器融合，结合自定义注释工具对多维时间序列数据进行标注，并使用数据驱动异常检测技术。

**📊 数据集**

使用UAV-SEAD数据集，该数据集包含1396次飞行、52小时、1396条日志，涵盖多种传感器配置。

**📈 对比分析**

与现有注入式或仿真异常数据集（如ALFA、UAV‑FD、RflyMAD等）比较，UAV-SEAD在真实异常多样性、样本量和时长上均优于它们，为后续训练更鲁棒的异常检测模型提供基础。

**⚠️ 局限性**

局限性包括室外全局定位异常样本相对不足，部分异常标签缺乏精确时序，且仍需进一步验证模型在实时场景下的性能。

---

## 532. Wrivinder: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery

**arXiv ID:** 2602.14929 | [PDF](https://arxiv.org/pdf/2602.14929v1)

**作者:** Chandrakanth Gudavalli `[一作]` (Mayachitra), B. S. Manjunath `[通讯]` (Mayachitra)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了零样本、基于几何的框架 Wrivinder，用来将多视角地面图像与地理注册的卫星图像对齐，实现精准摄像机位置回归。

**💡 创新点**

结合 SfM、3D Gaussian Splatting、语义地面判别与自监督深度匹配，首次在无配对监督的条件下实现从地面视角到卫星视角的全自动几何对齐。

**🔧 技术方法**

使用 Structure-from-Motion 重建稀疏点云，Octree-GS 生成密集 3D 高质量渲染，Mask2Former 语义分割提取地面，深度模型估计尺度，轻量级深度模板匹配器进行卫星图像定位，最后通过 RANSAC 对齐得到 GPS 坐标。

**📊 数据集**

基于 MC‑Sat 数据集，该数据集整合了 ULTRAA、VisymScenes、ACC‑NVS 与 JHU‑Ames 的多视角地面图像与 NAIP/ESRI 的卫星图像。

**📈 对比分析**

与基准 CVGL 检索模型在 MC‑Sat 上进行对比，Wrivinder 在大多数场景下达到 30 米以内的平均地理定位误差，且 67% 分位误差亦低于 30 米，证明了零样本几何方法的可行性。

**⚠️ 局限性**

对大面积、稀疏观测或存在屋顶等未被地面视角观测到结构的场景，对齐效果下降；SfM 稳定性和语义分割误差会直接影响最终定位精度。

---

## 533. Parameter-Efficient Fine-Tuning of LLMs with Mixture of Space Experts

**arXiv ID:** 2602.14490 | [PDF](https://arxiv.org/pdf/2602.14490v1)

**作者:** Buze Zhang `[一作]` (Xi'an Jiaotong University), Rex Ying `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一的Mixture of Space（MoS）框架以及其扩展的MoSLoRA，用于LLM的参数高效微调；

**💡 创新点**

通过同时使用超曲率、欧氏和球面三种常数曲率空间，并学习可调曲率参数，使模型能够动态路由每个token到最适合的几何专家；

**🔧 技术方法**

结合Mixture of Space投影、轻量级空间映射、LoRA混合专家架构以及独立曲率优化器等技术；

**📊 数据集**

在自然语言理解（CoLA、MRPC、RTE）、数学推理（GSM8k、MAWPS、SVAMP、AQuA、MATH500）以及常识推理（OpenBookQA、CommonsenseQA）等多种基准上进行评测；

**📈 对比分析**

与LoRA、AdaLoRA、DoRA、MELoRA、HMoRA、HydraLoRA、HypLoRA等方法对比，平均提升约4–6%，在MATH500上提升约20%（相对基线）；

**⚠️ 局限性**

需要进一步验证在工业规模模型上的可扩展性，曲率学习可能导致训练不稳定，且对极少见符号的处理仍有限。

---

## 534. Experimentation Accelerator: Interpretable Insights and Creative Recommendations for A/B Testing with Content-Aware ranking

**arXiv ID:** 2602.13852 | [PDF](https://arxiv.org/pdf/2602.13852v1)

**作者:** Zhengmian Hu `[一作]` (Adobe Research), David Arbour `[通讯]` (Adobe Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个统一的框架，用于优先测试A/B实验变体、解释变体胜出的原因，并为新变体提供创意改进建议；

**💡 创新点**

创新点在于将历史A/B实验结果与内容嵌入结合，利用混合效应回归+PCA构建CTR排序模型，并通过约束Lasso将模型投影到可解释的营销属性空间，实现可解释性和机会生成；

**🔧 技术方法**

主要技术包括文本嵌入（BERT/LLama等）、PCA降维、混合效应回归、约束Lasso、属性嵌入、机会指数计算以及大型语言模型生成自然语言洞察与建议；

**📊 数据集**

使用公开的Upworthy A/B实验数据集进行训练，并在Adobe客户的65个真实实验上进行测试；

**📈 对比分析**

通过Spearman秩相关和Top‑1准确率评估，转移学习模型在真实实验中取得约0.514的秩相关和70% Top‑1准确率，明显优于随机猜测；在不同嵌入方法的留一实验中，Llama嵌入得到最高的0.727秩相关；

**⚠️ 局限性**

局限性包括：迁移学习需验证不同域的可迁移性；当前仅支持英文；嵌入模型对性能影响大，需要针对不同客户群体挑选最佳嵌入；未来需扩展至图像、视频等多模态内容和其他业务指标。

---

## 535. Applying Public Health Systematic Approaches to Cybersecurity: The Economics of Collective Defense

**arXiv ID:** 2602.13869 | [PDF](https://arxiv.org/pdf/2602.13869v1)

**作者:** Josiah Dykstra `[一作]` (Designer Security), William Yurcik `[通讯]` (Centers for Medicare and Medicaid Services)

**通讯引用:** 2912 | [OpenAlex ID](https://openalex.org/A5058426115)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出建立全国性网络公共卫生体系，将流行病学的系统数据收集、循证干预与协调响应原则应用于网络安全。

**💡 创新点**

将网络安全视为公共产品并提出联邦层面的标准化、量化与协同治理框架，借鉴公共卫生经验创新了跨组织的防御评估与决策机制。

**🔧 技术方法**

利用统计监测与数据可视化技术、威胁情报共享、电子邮件身份验证（SPF/DKIM/DMARC）等现有安全技术，构建统一的指标体系和监测平台。

**📊 数据集**

目前缺乏统一的数据集，论文主要引用公开的攻击报告、行业案例与学术研究数据，强调需要建立全国性的安全事件与资产数据库。

**📈 对比分析**

通过经济学的公共产品与外部性分析与公共卫生的数据驱动模型进行对比，指出市场缺失导致的防御不足；虽未给出量化实验结果，但提出可行的指标与评估思路。

**⚠️ 局限性**

局限在于缺乏标准化的“网络人群”与指标定义、数据采集与共享机制不完善、实现依赖政府主导且跨部门协同成本高、难以立即验证效果。

---

## 536. AI Unplugged: Embodied Interactions for AI Literacy in Higher Education

**arXiv ID:** 2602.13242 | [PDF](https://arxiv.org/pdf/2602.13242v1)

**作者:** Jennifer M. Reddig `[一作]` (Georgia Institute of Technology), Christopher J. MacLellan `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 371 | [OpenAlex ID](https://openalex.org/A5077641166)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

设计并实施了四套无计算机（Unplugged）活动，融入大学层面《人工智能导论》课程，通过实体游戏让学生体验搜索、马尔可夫决策过程、Q学习和隐藏马尔可夫模型，并在课堂后引导学生用 Python 代码重现这些算法。

**💡 创新点**

创新点在于把 CS Unplugged 的实体、协作式教学模式引入高校 AI 课程，强调身体化认知与算法思维的对接；同时提供从体验到形式化、再到编码的完整教学循环，突破传统讲授式 AI 课程在概念与实践之间的鸿沟。

**🔧 技术方法**

核心技术包括：基于纸牌、骰子和桌面格子构建的物理模拟活动；Python Jupyter Notebook 的代码实现与实验；以及对算法（BFS/DFS、MDP、Q-learning、HMM）与概率、动态规划等理论的桥接。

**📊 数据集**

主要使用的“数据”是课堂活动中自制的卡牌、骰子、纸板格子等物理材料；未采用公开机器学习数据集，活动侧重于手工计算和直观模拟。

**📈 对比分析**

比较方法主要是定性评估：课堂出席率、学生讨论活跃度、个人反思与作业提交情况。与以往以讲授为主的课程相比，Unplugged 版本的学生出勤率从 25–30% 提升到 75–80%，学生在概念理解与编码迁移上表现更好；但未给出客观的算法性能指标或量化学习成绩。

**⚠️ 局限性**

局限性包括：课程时长短，导致覆盖内容受限；课堂规模有限（约 40 人），难以推广到更大班级；活动实施需要教师和助教大量准备与现场指导；学生在活动后可能不愿主动表达思路；缺乏系统化的量化评估和长期学习效果跟踪。

---

## 537. Guided Collaboration in Heterogeneous LLM-Based Multi-Agent Systems via Entropy-Based Understanding Assessment and Experience Retrieval

**arXiv ID:** 2602.13639 | [PDF](https://arxiv.org/pdf/2602.13639v1)

**作者:** Linlin Wang `[一作]` (City University of Macau), Wanlei Zhou `[通讯]` (City University of Macau)

**通讯引用:** 14844 | [OpenAlex ID](https://openalex.org/A5051406984)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究如何在异构多代理系统中解决强弱代理间认知失配导致的负协同效应，提出基于熵的自适应指导框架；

**💡 创新点**

创新点包括：①多维熵量化弱代理理解状态；②基于熵阈值的三档自适应指导策略；③结合检索增强生成(RAG)的经验回溯与长期学习机制；

**🔧 技术方法**

技术手段包括信息熵评估、动态阈值调整、检索增强生成、分层指导（轻/中/重）与多轮交互；

**📊 数据集**

使用三大基准数据集：GSM8K（数学推理）、MBPP（Python程序生成）、CVRP（车辆路径规划）；

**📈 对比分析**

与无指导和链式思考(CoT)基线对比，实验显示在三项任务上平均提升6.6%–24.0%，RAG增强进一步提升5–10个百分点；

**⚠️ 局限性**

局限在于：①依赖强弱代理预设，缺乏自动能力匹配机制；②RAG检索成本和召回质量受限；③仅在所选任务与模型规模下验证，未覆盖更大规模或其他任务场景；

---

## 538. Image-based Joint-level Detection for Inflammation in Rheumatoid Arthritis from Small and Imbalanced Data

**arXiv ID:** 2602.14365 | [PDF](https://arxiv.org/pdf/2602.14365v1)

**作者:** Shun Kato `[一作]` (Keio University), Mariko Isogawa `[通讯]` (Keio University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种基于 RGB 手部图像的类风湿性关节炎炎症检测框架，利用全局-局部编码器对手部整体与关节局部特征进行融合，解决样本稀缺与类别不平衡问题。

**💡 创新点**

创新点：①自监督预训练（DINO）在大规模健康手部图像上获得稳健特征；②引入 Focal Loss 进行不平衡训练；③构建基于超声标注的高可靠手部图像数据集；④全局-局部双编码器结构在少量数据上实现高性能。

**🔧 技术方法**

技术：ResNet-18 作为全局/局部编码器；DINO 自监督预训练；Focal Loss 处理类别不平衡；SAM 2 + MediaPipe 进行背景去除与关节定位；AdamW/Adam 优化器。

**📊 数据集**

使用自建 2025 年 68 位患者共 222 张手部图像的数据集，炎症标签基于超声诊断（灰度 >1 且 PD >0）。

**📈 对比分析**

与 ResNet-18、ResNet-50、InceptionResNetv2、ViT 等基线模型对比，自己方法在 Recall 0.478、Precision 0.374、F1 0.420、Gmean 0.605 上明显优于基线（例如 ResNet-18 F1 0.230）。

**⚠️ 局限性**

局限性：假设关节炎症在 RGB 图像中有可见变化，可能在极早期无明显外观差异；数据集规模仍有限，需进一步扩展。

---

## 539. ALMo: Interactive Aim-Limit-Defined, Multi-Objective System for Personalized High-Dose-Rate Brachytherapy Treatment Planning and Visualization for Cervical Cancer

**arXiv ID:** 2602.13666 | [PDF](https://arxiv.org/pdf/2602.13666v1)

**作者:** Edward Chen `[一作]` (Stanford University), Carlos Guestrin `[通讯]` (Stanford University)

**通讯引用:** 47614 | [OpenAlex ID](https://openalex.org/A5090739892)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

针对子宫颈癌高剂量率（HDR）源标定的临床决策支持系统ALMo，结合了热点毒性控制、目标-限制（aim‑limit）交互式多目标优化与可视化，用于快速生成高质量、个体化放疗计划。

**💡 创新点**

创新点在于①引入目标‑限制框架，让临床医生通过直观的滑块直接操控软硬阈值；②在多目标优化中加入热点人工结构实现毒性约束；③使用MoSH（稠密‑稀疏）采样与子模优化高效逼近Pareto前沿；④自动化参数初始化与限制器预测显著减少手工输入；⑤集成的二维/三维可视化工具提升决策效率。

**🔧 技术方法**

核心技术包括：ϵ‑约束多目标线性规划、TCVaR（截断条件值风险）近似、软硬效用函数（SHF）与柔性阈值映射、MoSH稠密采样/稀疏子模选择、热点人工结构构造、并行坐标可视化与图像裁切显示。

**📊 数据集**

使用25例子宫颈癌HDR源标定的回顾性临床病例（20例用于优化质量评估，5例用于交互探索评估），数据来源于临床CT影像、已分割的PTV与OAR（膀胱、直肠、肠道、宫颈及阴道黏膜等）。

**📈 对比分析**

与传统单目标/多目标基线（如PNaV）以及人工规划进行对比。质量评估采用临床医生制定的分级评分，ALMo在65%的病例中实现了微至显著的剂量改进；效率评估显示平均规划时间约为17.6 分钟，较传统30–90 分钟缩短一半；利用超体积指标显示在迭代探索中可达95%最大超体积，计算速度比完整网格搜索快约14倍。

**⚠️ 局限性**

局限性包括：①仅为优化工具，未涵盖影像分割与质量保证等前后端步骤；②验证为回顾性数据，缺乏实时手术场景的实时评估；③目前仅在子宫颈HDR源标定中验证，尚未推广至其他部位或治疗模式；④对医生熟练度与机构工作流程的依赖性未完全量化。

---

## 540. Spatiotemporal Feature Alignment and Weighted Fusion in Collaborative Perception Enabled by Network Synchronization and Age of Information

**arXiv ID:** 2602.13439 | [PDF](https://arxiv.org/pdf/2602.13439v1)

**作者:** Qiaomei Han `[一作]`, Dusit Niyato `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种通过网络同步和信息年龄（AoI）实现时空特征对齐与加权融合的协同感知框架。

**💡 创新点**

创新点包括：①持续估计车辆间时钟状态并构建共享时间基准；②在该基准上计算AoI并结合通信延迟预测特征新鲜度；③将空间投影与时间补偿分离，并利用AoI驱动的时序补偿实现完整时空对齐；④将时空不确定性量化为可靠性指标，与AoI共同决定RoI级加权融合。

**🔧 技术方法**

技术手段包括：PTP时钟同步、鲁棒Kalman滤波、AoI与通信延迟建模、几何投影与Conv1D时序补偿、可靠性度量与加权融合、端到端监督训练。

**📊 数据集**

使用基于SUMO+CARLA仿真的V2X‑Sim数据集，LiDAR点云经Voxel化后投影到BEV特征图。

**📈 对比分析**

与CoBEVFlow、SyncNet等基线对比，实验显示在mAP@0.5和mAP@0.7上均有显著提升；在时空误差为Δt≤3的场景中效果更为突出。

**⚠️ 局限性**

局限性：在大时空偏差（Δt≥5）下性能显著下降；依赖精确的时钟同步与延迟估计；模型相对复杂，计算和通信开销相对较高。

---

## 541. Pre-Editorial Normalization for Automatically Transcribed Medieval Manuscripts in Old French and Latin

**arXiv ID:** 2602.13905 | [PDF](https://arxiv.org/pdf/2602.13905v1)

**作者:** Thibault Clérice `[一作]`, David Smith `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出预编辑归一化（PEN）任务，构建拉丁文与古法文的大规模对齐数据集，并训练基于ByT5的序列到序列模型完成归一化；

**💡 创新点**

将归一化定义为中间层任务，兼顾字符变体与缩写保留，利用自动对齐+人工校正生成金标准，并通过PEN显著改善下游词形和词性标注；

**🔧 技术方法**

使用ByT5 Transformer、Passim 2.0+字符级HMM对齐、Unicode分解标准化、Zero-shot Llama3提示等技术；

**📊 数据集**

主要使用CoMMA ATR银集、digitized/born-digital Latin与Old French文本、CATMuS、Patrologia Latina、CEMA、Perseus等数据，手工校正后得到588样本金标准；

**📈 对比分析**

采用CER/WER评估，PEN模型在Gold数据集上CER从64.8%降至6.7%（拉丁文），在下游词形/词性任务中精度、F1提升约2-3个百分点，召回略低；

**⚠️ 局限性**

存在过度归一化倾向古典拼写、法文缺乏稳定正字法导致模型偏向主流变体、对齐噪声与缩写歧义导致错误、数据覆盖不均衡且未对不确定性建模或跨文档上下文处理。

---

## 542. OMNI-LEAK: Orchestrator Multi-Agent Network Induced Data Leakage

**arXiv ID:** 2602.13477 | [PDF](https://arxiv.org/pdf/2602.13477v1)

**作者:** Akshat Naik `[一作]` (University of Oxford), Adel Bibi `[通讯]` (University of Oxford)

**通讯引用:** 58641 | [OpenAlex ID](https://openalex.org/A5042899882)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对流行的 orchestrator 多智能体系统，开展安全漏洞评估，并提出了一种新型的多智能体间 Prompt 注入攻击方法 OMNI-LEAK，揭示单个指令即可协同窃取敏感数据。

**💡 创新点**

创新点在于：①首次系统化展示通过间接 Prompt 注入在 orchestrator 结构中横向传播，协同多智能体完成数据泄露；②构建了针对该攻击的基准评测框架，涵盖不同模型、攻击类别、数据库规模的综合实验。

**🔧 技术方法**

使用技术包括：大型语言模型（GPT‑4.1、Claude‑Sonnet‑4、Gemini‑2.5‑Flash 等）作为 orchestrator、SQL agent 与 Notification agent 的实现；SQL 与 RAG 生成工具；Prompt 注入与红队攻击方法；自动化评估脚本（关键词匹配）。

**📊 数据集**

数据集为自构造的员工 HR 数据库，包含公共表与私有表（含 SSN），分别提供 Toy、Medium、Big 三个规模版本，用以测试攻击对数据库大小的敏感性。

**📈 对比分析**

比较方法：对 5 大前沿模型分别在 10 种攻击（3 类攻击策略、显式/隐式两种变体）下进行 3000 次实验，计算正面问答准确率 (BA)、鲁棒问答准确率 (RA) 与成功攻击所需平均查询数 (E)。结果显示，除 GPT‑4.1‑mini 外所有模型均能被至少一次攻击利用，且攻击成功率与数据库规模无显著关联。

**⚠️ 局限性**

局限性包括：①实验仅涵盖 orchestrator‑SQL‑Notification 典型结构，未覆盖更复杂或去中心化多智能体；②仅聚焦数据泄露，未探究代码执行等其他安全风险；③使用人工构造的数据库，缺乏真实业务场景验证；④缺乏完整的防护与缓解策略实现。

---

## 543. One Good Source is All You Need: Near-Optimal Regret for Bandits under Heterogeneous Noise

**arXiv ID:** 2602.14474 | [PDF](https://arxiv.org/pdf/2602.14474v1)

**作者:** Aadirupa Saha `[一作]` (University of Illinois Chicago), Haipeng Luo `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了具有M个异构数据源的K臂多臂赌博机（MAB）问题，提出了一种新的算法Source-Optimistic Adaptive Regret Minimization（SOAR），旨在通过自适应选择查询数据源来最小化MAB的遗憾。

**💡 创新点**

创新点在于通过尖锐的方差集中界限快速修剪高方差源，并采用平衡的最小-最大置信区间（LCB-UCB）方法，整合识别最佳臂和最优数据源的任务。

**🔧 技术方法**

使用了上置信界（UCB）和下置信界（LCB）相结合的策略，进行臂和数据源的联合选择。

**📊 数据集**

使用了多个合成问题实例和真实世界的MovieLens 25M数据集进行实验。

**📈 对比分析**

与Uniform UCB和Explore-then-Commit UCB等基线方法进行比较，SOAR在性能上显著优于这些基线，尤其是在高方差源的情况下，遗憾的增长得到了有效控制。

**⚠️ 局限性**

限制在于算法的预处理阶段可能需要较高的计算成本，且在某些情况下，仍然可能受到高方差源的影响。

---

## 544. A Penalty Approach for Differentiation Through Black-Box Quadratic Programming Solvers

**arXiv ID:** 2602.14154 | [PDF](https://arxiv.org/pdf/2602.14154v1)

**作者:** Yuxuan Linghu `[一作]` (Shanghai Jiao Tong University), Qi Deng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 29639 | [OpenAlex ID](https://openalex.org/A5107882524)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了 dXPP，一种基于平滑罚函数的差分方法，能够在不显式求解 KKT 系统的前提下，对黑盒二次规划求解器进行梯度传播。

**💡 创新点**

创新点在于将约束直接嵌入平滑罚目标，利用隐式微分只求解原变量维度的 SPD 线性系统，从而实现求解器无关的前向推理和数值更稳定、计算更高效的后向传播。

**🔧 技术方法**

使用了 Softplus 平滑惩罚、隐式微分理论、从解算器返回的对偶变量来设定罚参数，并采用 Cholesky 或预条件共轭梯度求解 SPD 系统。

**📊 数据集**

实验数据涵盖随机严格凸 QP、极大规模稀疏投影问题（概率单纯形和链约束）以及实际金融场景的多周期均值‑方差组合优化，使用了公开的 ETF 回报数据。

**📈 对比分析**

通过与 dQP、OptNet、SCQPTH、CVXPYLayers 等方法对比，dXPP 在梯度相对误差小于 10⁻³ 的前提下，在向量化的 10⁶ 维问题上后向时间提升 4–9 倍，并在多周期组合优化任务中实现了 300+ 倍的加速。

**⚠️ 局限性**

局限性包括需依赖解算器返回的对偶信息来设定罚参数、对极端退化或非凸问题的适用性有限、以及平滑参数 δ 的选择仍需经验调优。

---

## 545. BRAIN: Bayesian Reasoning via Active Inference for Agentic and Embodied Intelligence in Mobile Networks

**arXiv ID:** 2602.14033 | [PDF](https://arxiv.org/pdf/2602.14033v1)

**作者:** Osman Tugay Basaran `[一作]` (Technical University Berlin), Falko Dressler `[通讯]` (Technical University Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于主动推理的深度贝叶斯框架 qcrBRAIN，用于 6G AI‑RAN 环境中的闭环网络切片资源调度。

**💡 创新点**

创新点：①将主动推理与深度生成模型结合，实现感知与决策的统一；②通过自由能分解提供可解释的贝叶斯信念与行动；③实现无重训练的终身学习，能在非平稳流量下自适应。

**🔧 技术方法**

使用技术：深度生成模型 + 变分自由能最小化的主动推理；GPU 加速的 O‑RAN 测试平台；与多种 DRL（DQN、A2C、PPO、SAC 等）和启发式基线对比。

**📊 数据集**

数据来源：私有 5G AI‑RAN 测试平台实时收集的切片 KPI（吞吐量、延迟、TB 计数）和多路切片流量负载，未使用公开数据集。

**📈 对比分析**

对比方法：在相同状态/动作接口下进行在线学习；qcrBRAIN 收敛更快、累计奖励最高，QoS 满足率提升约 28.3%，并在非平稳流量冲击后恢复最快。

**⚠️ 局限性**

限制：需要在更大规模多小区/多代理场景验证；生成模型对网络动态的覆盖依赖于观测质量与参数设置；实现复杂度高，需 GPU 资源。

---

## 546. Execution-State-Aware LLM Reasoning for Automated Proof-of-Vulnerability Generation

**arXiv ID:** 2602.13574 | [PDF](https://arxiv.org/pdf/2602.13574v1)

**作者:** Haoyu Li `[一作]` (University of Illinois Urbana-Champaign), Luyi Xing `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8227 | [OpenAlex ID](https://openalex.org/A5103347049)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于 LLM 的智能代理框架，专门用于自动生成验证漏洞的 Proof‑of‑Vulnerability（PoV），通过迭代的假设‑验证‑修正闭环实现从源代码推理到运行时验证的闭合反馈。

**💡 创新点**

创新点主要有：①将 PoV 生成建模为交互式闭环过程，融合静态语义推理与动态执行反馈；②设计低级执行轨迹到源级约束的“Trace‑to‑Prompt”翻译机制；③引入根因分析、执行反馈（覆盖率）与 crash‑triggering 指导，显著提升 LLM 的定位与触发能力；④采用多阶段分解（路径探索 + 崩溃触发）和多 LLM 细粒度调度，兼顾性能与成本。

**🔧 技术方法**

核心技术包括：大模型 LLM（Claude Sonnet 4.5 及轻量化模型）、代码分析与 instrumentation、llvm‑cov 覆盖查询、Trace‑to‑Prompt 译码器、根因分析子代理、执行反馈闭环、成本控制与 token 管理。

**📊 数据集**

使用 SEC‑bench 数据集（190 个已过滤 CVE 任务，随机抽取 60 任务用于基线对比），涵盖 29 个 C/C++ 开源项目和多种内存安全漏洞。

**📈 对比分析**

在相同 LLM 与预算（$1.5/任务）下，本文方法在 SEC‑bench‑60 上验证 PoV 的成功率从 OpenHands 的 10% 提升至 25%（验证 PoV 15/60），在全量集上成功率 28.9%（55/190）高于 18%（36/190）的现有最高基线；成本每成功 PoV 约 $7.7，低于 OpenHands 的 $15.3；运行时平均约 11.8 分钟，虽略高于 OpenHands，但相对更具可解释性与稳健性。

**⚠️ 局限性**

局限性包括：①目前聚焦内存相关漏洞，非内存漏洞（如逻辑错误、命令注入）尚未验证；②依赖可 instrument 的二进制与覆盖工具，某些项目构建或动态链接可能导致失效；③对 LLM 的调用频率与 token 费用仍高，需进一步优化成本；④复杂多层逻辑漏洞仍可能需要更深层次的语义推理或符号执行支持。

---

## 547. Gradient Networks for Universal Magnetic Modeling of Synchronous Machines

**arXiv ID:** 2602.14947 | [PDF](https://arxiv.org/pdf/2602.14947v1)

**作者:** Junyi Li `[一作]` (Aalto University), Marko Hinkkanen `[通讯]` (Aalto University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于梯度网络的物理信息化同步机磁模型，直接通过能量函数梯度生成电流和电磁转矩，保证能量守恒和对称性。

**💡 创新点**

将梯度网络直接嵌入基本机器方程，既保留能量守恒，又在同一网络中同时学习磁饱和与空间谐波，减少训练样本并实现唯一可逆映射。

**🔧 技术方法**

使用梯度网络（monotone gradient network）、p‑norm 与 softmax 向量激活，结合傅里叶特征和线性项；训练采用 PyTorch AdamW，损失为 MSE 并含转矩项。

**📊 数据集**

基于 5.6 kW 永磁同步磁阻机的实验测量数据和 FEM 计算数据，包含无空间谐波和包含空间谐波两种场景，样本数从 10% 到 0.2% 进行测试。

**📈 对比分析**

与传统查表、softplus、softmax 以及 2×squareplus 等激活函数对比，结果显示 p‑norm 激活在极少数据下误差仍可控制在 0.01–0.02 p.u. 范围，模型在转矩与电流/磁链预测上优于常规方法。

**⚠️ 局限性**

仍需在更高阶多相机型或极限负载条件下验证泛化性能；对极限样本量和网络深度的影响未做系统评估。

---

## 548. How Multimodal Large Language Models Support Access to Visual Information: A Diary Study With Blind and Low Vision People

**arXiv ID:** 2602.13469 | [PDF](https://arxiv.org/pdf/2602.13469v1)

**作者:** Ricardo E. Gonzalez Penuela `[一作]` (Cornell University), Shiri Azenkot `[通讯]` (Cornell University)

**通讯引用:** 3327 | [OpenAlex ID](https://openalex.org/A5060708945)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在为期两周的日记研究中，收集并分析了20名盲人和低视力者使用基于GPT‑4o的视觉助手VisionPal的交互记录与访谈，以评估多模态大型语言模型在真实环境下的视觉解释能力。

**💡 创新点**

提出了“视觉助手”技能框架，定义九个关键行为，并给出针对模型训练、系统提示与用户界面三阶段的改进方法，以提升MMLM在BLV场景中的可信度与可用性。

**🔧 技术方法**

采用了GPT‑4o多模态语言模型作为视觉解释核心，并通过自定义对话式API和提示工程实现图像描述与问答功能。

**📊 数据集**

使用来自20名参与者的真实拍摄图片与对话数据，形成包含551条日记记录（共计626个问答）的实验数据集。

**📈 对比分析**

与之前基于传统视觉描述模型的研究对比，VisionPal在初始图像描述上的准确率达到2.91/3（91.8%无幻觉），用户满意度4.13/5，信任度3.76/5；但在后续问答中，准确率仅56.6%，错误率22.2%。

**⚠️ 局限性**

限制包括样本主要为经验丰富的BLV使用者，缺乏新手视角；数据收集时间集中在繁忙节假日，可能影响使用模式；以及模型在敏感信息处理与文本识别方面仍存在高误报与回避行为。

---

## 549. Robust multi-task boosting using clustering and local ensembling

**arXiv ID:** 2602.14231 | [PDF](https://arxiv.org/pdf/2602.14231v1)

**作者:** Seyedsaman Emami `[一作]` (Autonomous University of Madrid), Gonzalo Martínez-Muñoz `[通讯]` (Autonomous University of Madrid)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出RMB‑CLE框架，利用交叉任务误差估计任务相似性，自动层次聚类生成多任务簇，并在每个簇内使用局部集成（LGBM或MTGB）训练模型，从而降低负迁移并提升预测性能。

**💡 创新点**

创新点包括：1）用交叉任务误差直接度量功能相似性，避免仅凭单任务难度或人工相似度；2）自适应层次聚类（UPGMA）自动发现多任务组；3）局部集成在簇内共享、在簇间隔离，彻底消除不相关任务的负迁移；4）理论证明误差分解为功能误差与不可约噪声，提供稳健的相似性依据。

**🔧 技术方法**

技术手段包括：交叉任务误差计算、余弦距离构建任务空间、平均连结层次聚类、轮廓系数选择最佳簇数、基于LGBM/MTGB的局部集成、基于误差驱动的任务相关性分析及其理论推导。

**📊 数据集**

使用的实验数据集：合成随机特征生成的多任务数据（5个簇、25任务）以及真实多任务数据集，包括Adult‑Gender、Adult‑Race、Avila、Bank Marketing、Landmine、Abalone、Computer、Parkinsons、SARCOS、School等。

**📈 对比分析**

与单任务、数据池化、任务特征加权（TaF）、GB/LGBM、MTGB、R‑MTGB等基线在合成与真实数据上进行对比。RMB‑CLE在所有指标（准确率、召回率、RMSE、MAE）上均优于或接近oracle cluster‑known 上限，表现显著提升。

**⚠️ 局限性**

局限性包括：需对所有任务对进行交叉评估，导致O(m²)时间和内存复杂度；假设所有任务共享相同输入维度，无法处理任务维度异质性；未提供大规模任务数的可扩展近似方案；以及对极端任务分布（如高噪声或极少样本）尚未深入验证。

---

## 550. Joint Orientation and Weight Optimization for Robust Watertight Surface Reconstruction via Dirichlet-Regularized Winding Fields

**arXiv ID:** 2602.13801 | [PDF](https://arxiv.org/pdf/2602.13801v1)

**作者:** Jiaze Li `[一作]` (Nanyang Technological University), Ying He `[通讯]` (Nanyang Technological University)

**通讯引用:** 8091 | [OpenAlex ID](https://openalex.org/A5100389169)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种Dirichlet Winding Reconstruction（DiWR）方法，直接从无定向、采样非均匀、带噪声和离群点的点云中重建水密三维表面。

**💡 创新点**

创新点在于将点方向、每点面积权重和置信系数三者联合优化，并利用通用绕数（GWN）场与Dirichlet能量正则化，能够在不做预处理的情况下自适应抵抗噪声、非均匀采样和离群点。

**🔧 技术方法**

采用通用绕数公式、Dirichlet能量最小化、DWG（Diffusion‑based Winding field Generation）方向更新、交替优化与RMSProp，最终用屏蔽泊松重建（sPSR）生成最终表面。

**📊 数据集**

使用了三类数据集：来自3D Gaussian Splatting（OmniObject3D）的点云、从计算机视觉管线VGGT生成的点云，以及对Armadillo、Dragon、Kitten等图形基准模型在不同扰动级别下构造的点云，还做了Bunny的压力测试。

**📈 对比分析**

与传统多阶段管线（MSP）、联合方法（WNNC、DWG、FaCE）以及深度学习方法（NSH、LoSF‑UDF）对比，采用Chamfer距离、法向一致性等指标评估。DiWR在所有三类数据集上均取得最佳或接近最佳的指标，显著降低离群层和噪声对结果的影响，视觉效果更清晰、完整。

**⚠️ 局限性**

局限性包括：仍依赖GPU并行计算，极大点云规模下可能受限；对极稀疏或极端高噪声点云的重建精度有限；需要经验性阈值与权重调节；在极稀疏区域的表面细节恢复仍受限。

---

## 551. BreathNet: Generalizable Audio Deepfake Detection via Breath-Cue-Guided Feature Refinement

**arXiv ID:** 2602.13596 | [PDF](https://arxiv.org/pdf/2602.13596v1)

**作者:** Zhe Ye `[一作]` (Sun Yat-sen University), Jiwu Huang `[通讯]` (Shenzhen MSU-BIT University)

**通讯引用:** 14414 | [OpenAlex ID](https://openalex.org/A5047964483)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了 BreathNet，一个结合呼吸线索、频域特征以及特征损失的音频深度伪造检测框架。

**💡 创新点**

创新点在于引入 BreathFiLM 模块使用帧级呼吸掩码指导 XLS‑R 学习呼吸信息，并设计 PSCL、中心损失与对比损失三种特征损失共同提升类内紧凑与类间分离。

**🔧 技术方法**

采用了 XLS‑R 预训练特征提取、SincConv/DFIM 频域特征、Cross‑attention 融合、BiLSTM 分类器，以及 Positive‑only Supervised Contrastive Loss、Center Loss、Contrast Loss 等技术。

**📊 数据集**

在 ASVspoof 2019/2021 LA/DF、In‑the‑Wild 与 ASVspoof5 五个公开数据集上进行训练与评估。

**📈 对比分析**

与多种 SOTA 方法比较，EER 0.23%（19LA）、1.15%/1.87%（21LA/DF）、4.70%（ITW）以及 4.94%（ASVspoof5），均达或逼近最优表现。

**⚠️ 局限性**

局限性在于某些复杂条件下深伪与真伪样本在特征空间仍有重叠，且模型对呼吸掩码的辅助训练依赖较强，极端掩码设置下效果不显著提升。

---

## 552. Constant-Time Dynamic Enumeration of Word Infixes in a Regular Language

**arXiv ID:** 2602.14748 | [PDF](https://arxiv.org/pdf/2602.14748v1)

**作者:** Antoine Amarilli `[一作]` (University of Lille), Luc Segoufin `[通讯]` (Inria)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文研究固定正则语言 L 的动态 L‑infix 枚举问题，提出一种在更新时保持常数时间、枚举时保持常数延迟与常数额外内存的算法。

**💡 创新点**

创新点在于引入半可扩展 ZG（semi‑extensible ZG）这一更广泛的语言类，并给出等价定义，证明在此类语言上可以实现上述常数时间/延迟/内存的动态枚举。

**🔧 技术方法**

技术手段包括使用出现列表（occurrence lists）、最小/最大左/右背景遍历、偶奇技巧（even‑odd）以及对语言类的半可扩展性条件进行利用，以实现常数时间的更新与枚举。

**📊 数据集**

论文未使用公开数据集，而是在理论层面通过复杂度分析证明算法性能；实验验证未给出。

**📈 对比分析**

与传统基于 MSO 查询的线性预处理+常数延迟方案相比，本文的算法在支持动态更新时实现了常数更新时间，并在大多数 ZG 语言类中保持了常数延迟和常数额外内存，性能上更为全面。

**⚠️ 局限性**

局限在于对非 ZG 或非半可扩展语言的处理仍受条件化（如 prefix‑U₁ 假设）限制，且算法实现复杂度较高，实际部署可能需要额外的工程工作。

---

## 553. A Comparative Analysis of Social Network Topology in Reddit and Moltbook

**arXiv ID:** 2602.13920 | [PDF](https://arxiv.org/pdf/2602.13920v1)

**作者:** Yiming Zhu `[一作]` (Hong Kong University of Science and Technology), Pan Hui `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 21166 | [OpenAlex ID](https://openalex.org/A5029925982)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用Moltbook和Reddit的评论数据，构建定向评论网络，对比分析两种社交网络在网络拓扑结构以及帖子对网络构建效率方面的差异。

**💡 创新点**

首次系统比较AI驱动社交网络与人类驱动网络的拓扑特征，揭示Moltbook呈现强烈的hub‑and‑spoke模式和“瞬时但短暂”的连接形成机制。

**🔧 技术方法**

通过API抓取数据、构建定向网络，计算度分布、聚类系数、密度、同类性、Freeman中心性等网络指标，并对帖子产生边的时间序列进行统计分析。

**📊 数据集**

使用Moltbook公开API抓取的约42万条帖子、240万条评论（共39,557节点）；以及Reddit Pushshift 7天快照，包含约9.3M帖子、67M评论（共7,854,970节点）。

**📈 对比分析**

采用相同的网络指标对两网络进行量化比较。结果显示Moltbook网络密度、平均邻居数、聚类系数均高，度分布更长尾，度同类性更负，中心性更高；在帖子效率方面，Moltbook的边生成率高、响应速度快，但边生成生命周期短，整体网络扩展速度快却持续时间有限。

**⚠️ 局限性**

研究样本时间窗口短（Moltbook 2026年1-2月，Reddit 2025年12月），未考虑跨平台用户迁移；未剔除所有自动化账户；仅分析评论网络，忽略点赞、收藏等其他交互；平台算法与机制可能影响结果，结论在不同平台或时间段下可能不完全适用。

---

## 554. S2SServiceBench: A Multimodal Benchmark for Last-Mile S2S Climate Services

**arXiv ID:** 2602.14017 | [PDF](https://arxiv.org/pdf/2602.14017v1)

**作者:** Chenyue Li `[一作]` (Hong Kong University of Science and Technology), Binhang Yuan `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 648 | [OpenAlex ID](https://openalex.org/A5002684888)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一个基于真实运营S2S气候服务的多模态基准，评估LLM/代理在最后一公里决策支持中的表现。

**💡 创新点**

创新点在于将基准设计与真实服务产品、三层服务任务级别以及可量化的操作化维度（信号理解、决策交接、规划分析）相结合，并引入统一的结构化输出与LLM-as-Judge评估。

**🔧 技术方法**

使用多模态LLM（如GPT-5.2、Claude、Gemini、Qwen、Llama）、标准化代理框架（DeepAgent+LangChain）以及LLM-as-Judge评估。

**📊 数据集**

使用从运营气候服务系统采集的10个S2S产品（农业、灾害、能源、金融、健康、航运）共约1000+评估项，包含多模态视图与元数据。

**📈 对比分析**

通过直接提示与标准化代理两种推理方式进行对比，结果显示GPT-5.2在信号理解层级约0.36，决策交接层级约0.64，规划层级约0.51；代理对信号理解略有提升，决策交接和规划层级表现不一，整体仍低于人类专家，表现受产品差异影响显著。

**⚠️ 局限性**

局限在于对多模态产品的理解仍不足，尤其是时间定位和不确定性映射，代理框架未能稳定提升决策交接和规划水平，缺乏针对性训练和专业工具，导致在动态灾害场景下性能骤降。

---

## 555. Atomix: Timely, Transactional Tool Use for Reliable Agentic Workflows

**arXiv ID:** 2602.14849 | [PDF](https://arxiv.org/pdf/2602.14849v1)

**作者:** Bardia Mohammadi `[一作]` (MPI Software Systems), Laurent Bindschaedler `[通讯]` (MPI Software Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 Atomix，一个能够在 LLM 代理工作流中按进度门控提交工具调用的事务化运行时。

**💡 创新点**

创新点在于引入资源前沿（frontier）进度判据，按资源级别控制事务提交，兼顾可缓冲与外部化效果，并通过补偿回滚处理不可逆效果。

**🔧 技术方法**

使用事务模型、前沿跟踪、工具适配器、补偿回滚以及 Python 库实现。

**📊 数据集**

在 WebArena、OSWorld 和 τ‑bench 等现有 LLM 代理基准上进行评估，并通过故障注入模拟。

**📈 对比分析**

与无事务、仅检查点回滚、无前沿等基线对比，Tx‑Full 在 30% 故障率下任务成功率从 3%–8% 提升到 37%–57%，前沿门控显著减少冲突与泄漏。

**⚠️ 局限性**

局限包括单进程实现、内存 dedup、未实现分布式前沿和持久化、对部分工具缺乏补偿器，以及对分布式多代理场景的支持尚未完成。

---

## 556. Knowing When Not to Answer: Abstention-Aware Scientific Reasoning

**arXiv ID:** 2602.14189 | [PDF](https://arxiv.org/pdf/2602.14189v1)

**作者:** Samir Abdaljalil `[一作]` (Texas A&M University), Hasan Kurban `[通讯]` (Hamad Bin Khalifa University)

**通讯引用:** 456 | [OpenAlex ID](https://openalex.org/A5070970331)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了科学论证与问答中的可弃回答框架，将输入拆分为条件，使用NLI审计并基于置信度进行弃答。

**💡 创新点**

提出了针对科学推理的可弃回答评估框架，强调条件分解与证据审计，并通过风险-覆盖分析展示弃答对可靠性的影响。

**🔧 技术方法**

采用条件分解（由LLM生成）、跨编码NLI模型审计、决策聚合规则、置信度阈值阈断弃答。

**📊 数据集**

评估数据集为SciFact（科学声明验证）与PubMedQA（医学问答）。

**📈 对比分析**

对比六种语言模型（Flan‑T5, Llama, Mistral, DeepSeek, GPT‑4o‑mini, GPT‑5.2），在未弃答时准确率相近；加入弃答后，风险-覆盖曲线显示所有模型在中等覆盖率下风险显著下降，FLAN‑T5在相同覆盖率下取得最低AURC。

**⚠️ 局限性**

主要局限在于依赖单一通用NLI模型、仅覆盖文本推理、以及对高风险场景的实际评估不足。

---

## 557. MOTIF: Learning Action Motifs for Few-shot Cross-Embodiment Transfer

**arXiv ID:** 2602.13764 | [PDF](https://arxiv.org/pdf/2602.13764v1)

**作者:** Heng Zhi `[一作]` (Tongji University), Heng Tao Shen `[通讯]` (Tongji University)

**通讯引用:** 30790 | [OpenAlex ID](https://openalex.org/A5052993469)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MOTIF框架，学习动作动机并在少样本条件下实现跨本体的迁移。

**💡 创新点**

将跨本体的时空动作模式与具体执行细节解耦，采用进度感知对齐和对抗损失使动机具备本体无关性。

**🔧 技术方法**

利用向量量化、进度感知对齐、对抗学习构建统一动机空间，配合多模态预测器和流匹配策略生成具体动作。

**📊 数据集**

在仿真中使用 ManiSkill 机器人任务数据，在真实世界中使用 ARX5 与 Piper 四个任务的数据集。

**📈 对比分析**

与 Diffusion Policy、π₀、HPT、GR00T N1 等基线比较，在 1-shot 至 50-shot 的不同数据量下，MOTIF 在仿真中提升 6.5%，在真实世界提升 43.7%。

**⚠️ 局限性**

对轨迹正则化和码本尺寸敏感，仍需少量示例支持，对极端高维动作空间的适应性待进一步验证。

---

## 558. Polar: An Algebraic Analyzer for (Probabilistic) Loops

**arXiv ID:** 2602.14573 | [PDF](https://arxiv.org/pdf/2602.14573v1)

**作者:** Marcel Moosbrugger `[一作]` (TU Wien), Laura Kovács `[通讯]` (TU Wien)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出并实现了Polar工具，能够自动化分析经典和概率循环，计算闭式、推导最强多项式不变量以及对未知参数的敏感性。

**💡 创新点**

创新点包括：① 在满足R1–R3限制下，用C‑finite递推获得闭式；② 通过Gröbner基和指数多项式变换自动推导最强多项式不变量；③ 通过闭式求导实现参数敏感性分析；④ 针对不可解循环提供不完全但正确的组合闭式与可解循环合成。

**🔧 技术方法**

核心技术包括：线性递推求闭式、指数多项式表达、Gröbner基与Buchberger算法、指数基数乘法关系求解、符号微分、静态分析求参数独立变量、组合多项式生成与可解循环合成。

**📊 数据集**

本文主要使用示例程序和案例演示；公开代码仓库（https://github.com/probing-lab/polar）提供可复现实验，但未引用大型公开数据集。

**📈 对比分析**

与已有工具（如Aligator、Mora）对比，Polar在处理带概率、分支与更一般多项式依赖时功能更全；在满足限制的程序上实现了完全自动化闭式与不变量；性能上在示例中执行时间可忽略，但未给出大规模实验结果。

**⚠️ 局限性**

限制包括：R1（概率常数）R2（有限guard）R3（无环多项式非线性）不可突破；超出限制时仅能得到不完整结果；对不可解循环仅能得到部分组合闭式或可解循环近似；不支持终止性分析、条件概率等更高级功能。

---

## 559. A Multi-Agent Framework for Code-Guided, Modular, and Verifiable Automated Machine Learning

**arXiv ID:** 2602.13937 | [PDF](https://arxiv.org/pdf/2602.13937v1)

**作者:** Dat Le `[一作]` (VNU University of Engineering and Technology), Hieu Dinh Vo `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种多智能体、基于代码驱动的 AutoML 框架，能够从任务描述到可执行 Python 脚本实现端到端模型构建。

**💡 创新点**

创新点在于三大支柱：基于实证数据的代码引导规划、使用接口契约的模块化实现以及动态验证与迭代调试的可验证集成。

**🔧 技术方法**

技术上结合了 LLM 智能体、Design by Contract、数据剖析、动态运行时验证、自动超参搜索、网络检索预训练模型与架构以及多模态调试与集成。

**📊 数据集**

实验使用了 Kaggle 的多种真实竞赛数据集，并引入了一个新的基准数据集，涵盖结构化、图像、文本等多模态任务。

**📈 对比分析**

与 SOTA 代码驱动代理及传统 AutoML 基线（如 AutoGluon、Auto-WEKA、Auto-Sklearn）进行对比，实验在 Kaggle 上实现了 85% 有效提交率、45% 奖牌率、APS 0.77，并在新基准上比对手高出 38–163% APS；在任务描述被裁剪时仍保持 70% 成功率。

**⚠️ 局限性**

局限性包括在多模态任务上的性能下降，仍依赖外部检索与 LLM 的知识更新，且跨模态特征协同尚未成熟。

---

## 560. Comparing Classifiers: A Case Study Using PyCM

**arXiv ID:** 2602.13482 | [PDF](https://arxiv.org/pdf/2602.13482v1)

**作者:** Sadra Sabouri `[一作]` (University of Southern California), Sepand Haghighi `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并演示了PyCM库，用于多类别模型的全面评估、可视化曲线绘制以及按类权重定制的综合比较；

**💡 创新点**

提出了将多指标聚合为两分数（整体分数与类分数）的Compare模块，支持自定义类重要性权重并直接输出最佳模型排名；

**🔧 技术方法**

使用Python实现的PyCM库（含150+评估指标、ROC/PR曲线、宏/微平均等），配合sklearn决策树训练与评估；

**📊 数据集**

采用公开的Covertype森林覆盖数据集（约60万样本，54特征）进行两棵决策树的比较；

**📈 对比分析**

通过类权重加权的Compare方法对两模型进行评估，在易燃区任务中排名第二模型，在河岸区任务中排名第一模型，展示了加权评估能揭示传统宏观指标忽略的细微差异；

**⚠️ 局限性**

局限在于Compare分数的聚合方式经验性强，缺乏针对极度不平衡数据集或不同应用场景权重设定的理论依据，并未充分验证在更大规模或更复杂任务中的鲁棒性。

---

## 561. Semantic Waveforms for AI-Native 6G Networks

**arXiv ID:** 2602.13316 | [PDF](https://arxiv.org/pdf/2602.13316v1)

**作者:** Nour Hello `[一作]` (CEA Leti), Emilio Calvanese Strinati `[通讯]` (CEA Leti)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在物理层即波形层面直接嵌入语义信息的 6G 语义波形设计框架——Orthogonal Semantic Sequency Division Multiplexing (OSSDM)，实现了语义通信与物理层资源的联合优化。

**💡 创新点**

创新点在于：①使用可参数化的 Walsh 正交基对波形进行可控衰减编码，将语义意义直接植入时间域信号；②提出了有效语义谱效率 (E‑SSE) 指标，兼顾物理与语义层面；③通过端到端训练实现语义编码、Walsh 映射与解码的协同优化，显著提升了在硬件限制下的谱效率与鲁棒性。

**🔧 技术方法**

核心技术包括：LLM + GNN + FFN 的语义编码器；Walsh 变换/逆变换用于波形生成与解码；代码书生成策略与互信息约束的端到端损失；对比实验中使用的 OFDM、LDPC 以及非语义 Walsh 基方案。

**📊 数据集**

使用的数据集为 WebNLG 知识图谱数据集，用于训练与评估语义编码器与整体系统的语义信息提取与重构性能。

**📈 对比分析**

比较方法：在 AWGN 与 Rayleigh 信道下，与非语义 OFDM + LDPC、语义 OFDM、无语义 Walsh 基等基线进行对比，评估指标为 F1 分数（KG 结构重构）和 E‑SSE。结果显示：OSSDM 在相同 SNR 下 F1 分数提升 4‑6 dB，低阶 Walsh 阶数下 E‑SSE 可提升 400 倍；在低 SNR 条件下，OSSDM 能显著降低所需功率，并在多径信道中保持相对优势。

**⚠️ 局限性**

局限性：①在频率分散、Rayleigh 多径信道中，低阶 OSSD M 性能低于 S‑OFDM；②随着 Walsh 阶数增大，信号维度与能量分布导致谱效率随之下降；③实验主要基于单一数据集与仿真环境，缺乏不同硬件平台与实际部署的验证；④端到端训练对模型迁移性与鲁棒性有一定依赖，实际场景中可能需要更复杂的调优与自适应机制。

---

## 562. Silent Inconsistency in Data-Parallel Full Fine-Tuning: Diagnosing Worker-Level Optimization Misalignment

**arXiv ID:** 2602.14462 | [PDF](https://arxiv.org/pdf/2602.14462v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 563. Process-Supervised Multi-Agent Reinforcement Learning for Reliable Clinical Reasoning

**arXiv ID:** 2602.14160 | [PDF](https://arxiv.org/pdf/2602.14160v1)

**作者:** Chaeeun Lee `[一作]` (University of Edinburgh), T. Ian Simpson `[通讯]` (University of Edinburgh)

**通讯引用:** 1955 | [OpenAlex ID](https://openalex.org/A5084953972)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个基于过程监督的多智能体强化学习框架，用来完成基因–疾病有效性鉴定任务；通过监督者智能体指挥专门的子智能体在不同实验证据类别上进行推理，并最终输出有效性分类与结构化证据轨迹。

**💡 创新点**

① 将过程级奖励（智能体调用与实验证据匹配）与结果奖励结合，构建混合奖励；② 在分层多智能体体系中，监督者仅负责路由，子智能体只需预测单一证据类别，显著提升过程一致性；③ 通过只训练监督者或联合SFT子智能体的方式降低计算成本。

**🔧 技术方法**

使用大型语言模型（Qwen3 1.7B/4B/8B）、Agent‑as‑Tool分层多智能体架构、工具调用机制、Group Relative Policy Optimization (GRPO) 进行强化学习，并采用混合奖励设计；对子智能体进行监督式微调（SFT）。

**📊 数据集**

ClinGen基因–疾病有效性鉴定数据库（含实验证据），从公开可获取的PMC-OA文章筛选得到1,994篇文献，构成训练/验证/测试集。

**📈 对比分析**

与单智能体基线以及仅使用结果奖励的GRPO进行对比。评价指标包括最终有效性分类准确率、证据分类准确率/F1、智能体调用准确率/F1。混合奖励下，Qwen3‑4B的结果准确率0.732，证据F1提升至0.520，过程指标显著优于仅结果奖励；单智能体在数值上略胜一筹，但缺乏可解释的中间证据。

**⚠️ 局限性**

多智能体系统在过程一致性上优于单智能体，但单智能体在整体准确率上略高；需要精细设计过程奖励与工具调用格式；目前仅覆盖实验证据，扩展到更广泛SOP仍需验证；子智能体若未微调，可能导致细粒度证据预测不足；模型规模与计算成本仍是推广瓶颈。

---

## 564. An Algebraic Rigidity Framework for Order-Oblivious Deterministic Black-Box PIT of ROABPs

**arXiv ID:** 2602.13449 | [PDF](https://arxiv.org/pdf/2602.13449v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 565. GPT-5 vs Other LLMs in Long Short-Context Performance

**arXiv ID:** 2602.14188 | [PDF](https://arxiv.org/pdf/2602.14188v1)

**作者:** Nima Esmi `[一作]` (Bernoulli Institute), Georgi Gaydadjiev `[通讯]` (TU Delft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估四款大型语言模型（Grok-4、GPT-4、Gemini 2.5、GPT-5）在长短文本任务中的表现，重点探讨其理论上下文窗口与实际推理能力的差距；

**💡 创新点**

首次将精度（precision）与准确率（accuracy）同时纳入评估框架，并通过实测验证“中间遗失”问题在新一代模型中已大幅缓解；

**🔧 技术方法**

采用标准的提示工程、API默认参数设置、基于token的输入分段以及多轮评估，利用长文本推理任务进行对比；

**📊 数据集**

使用三大自制与公开数据集：1）20K条Twitter抑郁相关推文（约300K token）；2）1K食谱（约20K token）进行素食检索；3）1K数学题（约20K token）进行概率统计题目检索；

**📈 对比分析**

通过在不同输入长度下计算准确率与精度，发现输入超过约70K token时准确率下降至50–53%，但GPT‑5保持约95%精度，整体表现呈现精度与准确率分离的趋势；

**⚠️ 局限性**

局限性包括：仅评估四个模型，缺乏更广泛的模型覆盖；实验聚焦单一任务类型，未验证跨任务通用性；结果受提示设计与API参数设置的影响，且对噪声数据的鲁棒性尚未深入探究。

---

## 566. A Generalizable Physics-guided Causal Model for Trajectory Prediction in Autonomous Driving

**arXiv ID:** 2602.13936 | [PDF](https://arxiv.org/pdf/2602.13936v1)

**作者:** Zhenyu Zong `[一作]` (William and Mary), Huajie Shao `[通讯]` (William and Mary)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于物理引导的因果模型 PCM，实现零样本轨迹预测。

**💡 创新点**

创新点在于通过干预式分解提取领域不变场景特征，并将其与车辆运动学结合的 CausalODE 解码器。

**🔧 技术方法**

采用干预分解的场景编码器、Transformer 结构、神经 ODE 与两轮运动学模型、因果注意力机制。

**📊 数据集**

使用 nuScenes、nuPlan、WOMD 三个公开轨迹预测数据集进行跨域评估。

**📈 对比分析**

与 Wayformer、AutoBot、G2LTraj、MTR、Forecast‑MAE、RMP、SMART、APE 等基线对比，零样本性能优于所有基线，minADE、minFDE 等指标显著下降。

**⚠️ 局限性**

局限在于仅使用简化的两轮模型，未考虑更复杂动力学；干预分解对 k 值敏感，需要进一步稳健性验证。

---

## 567. LM-Lexicon: Improving Definition Modeling via Harmonizing Semantic Experts

**arXiv ID:** 2602.14060 | [PDF](https://arxiv.org/pdf/2602.14060v1)

**作者:** Yang Liu `[一作]` (BIGAI), Lingyong Yan `[通讯]` (Baidu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的定义建模框架，利用语义聚类训练领域专家模型，再将这些专家合并为稀疏 MoE（Mixture‑of‑Experts）以生成多域词义定义。

**💡 创新点**

创新点包括：① 基于语义向量的聚类实现专家细粒度划分；② 领域级语义路由（而非常规 token‑level 路由），显著提升专家选择效果；③ 将专家权重按稀疏方式合并为单一 MoE，并在合并后继续微调路由器，以实现高效且精细的定义生成。

**🔧 技术方法**

技术手段主要有：预训练 LLM（如 Llama‑3‑8B）、nvidia‑embed‑v2 嵌入、balanced k‑means 聚类、稀疏 MoE 结构（FFN 级专家、Top‑k 路由器）、微调（负对数似然）以及测试时缩放（BoN、Oracle verifier）。

**📊 数据集**

使用的数据集包括 WordNet、Oxford、Wikipedia、Urban 以及综合性 3D‑EX（在 3D‑EX 上划分 4 个语义簇后进行专家训练和合并）。

**📈 对比分析**

与多种基线（传统 seq2seq、causal LLM、GPT‑4、Gemini、Claude‑3 等）进行对比。自动评估显示在 3D‑EX 上 BLEU +10%，ROUGE +10%；在小型数据集上（如 Urban）仍保持最高分；在人类评测中，在准确性、清晰度、简洁度等维度均优于对手，性能优于 frontier LLMs。

**⚠️ 局限性**

局限性：仅针对英文定义建模，训练需要大量计算资源；验证器（自检）仍不成熟，难以进一步提升测试时性能；模型目前难以直接迁移至其他语言或更广泛的语义任务。

---

## 568. Evolving Multi-Channel Confidence-Aware Activation Functions for Missing Data with Channel Propagation

**arXiv ID:** 2602.13864 | [PDF](https://arxiv.org/pdf/2602.13864v1)

**作者:** Naeem Shahabi Sani `[一作]` (University of Oklahoma), Dean F. Hougen `[通讯]` (University of Oklahoma)

**通讯引用:** 1460 | [OpenAlex ID](https://openalex.org/A5061058579)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对存在缺失值的表格数据，研究者提出一种三通道演化激活函数（3C-EA），并在多层感知机中通过ChannelProp机制在整个网络中传递缺失指示器和置信度信息，改进模型对缺失数据的鲁棒性。

**💡 创新点**

创新点包括：①将特征值、缺失指示器和置信度作为多元输入，使用遗传程序演化非线性激活；②提出ChannelProp这一确定性传播规则，在线性层中保留并软化缺失与置信度信号，使激活在深层中仍能感知数据质量；③结合演化搜索和可靠性信息，显著提升在不同缺失机制下的性能。

**🔧 技术方法**

使用技术包括：遗传程序（GP）搜索多元激活函数、三通道多层感知机架构、确定性置信度传播（ChannelProp）、标准Adam优化、早停、均值插补、缺失指示器生成以及基于特征缺失率的置信度赋值。

**📊 数据集**

数据集为UCI公开数据集，涵盖自然缺失（如Hepatitis、HouseVotes84、Adult等）与人工生成缺失（MCAR、MAR、MNAR，覆盖20%~50%缺失率）的完整数据集（Mushroom、WDBC、Pima、Sonar、Glass等）。

**📈 对比分析**

与传统固定激活函数（ReLU、Swish、LeakyReLU、ELU）在相同预处理、模型结构、数据划分下进行对比。实验结果显示，3C-EA在大多数数据集上均提升准确率、F1/召回率、AUC，尤其在缺失率高或多分类任务（如Glass、Pima）中表现更为显著。

**⚠️ 局限性**

局限性包括：①置信度采用基于缺失率的经验式估计，未学习动态置信度；②方法仅在多层感知机上验证，尚未扩展到循环、注意力等模型；③演化过程计算成本相对较高，可能难以直接迁移到大规模或实时任务；④对极端高缺失率或非数值型特征的适用性尚未充分评估。

---

## 569. RPGD: RANSAC-P3P Gradient Descent for Extrinsic Calibration in 3D Human Pose Estimation

**arXiv ID:** 2602.13901 | [PDF](https://arxiv.org/pdf/2602.13901v1)

**作者:** Zhanyu Tuo `[一作]` `[通讯]` (Sorbonne university), Zhanyu Tuo (Sorbonne university)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了RPGD框架，实现利用自然人体运动进行MoCap与RGB摄像机的外参自动对齐；

**💡 创新点**

将RANSAC-P3P的全局鲁棒性与梯度下降的细化精度融合，形成针对人体姿态的粗到细优化流程；

**🔧 技术方法**

使用RANSAC-P3P求解、Euler角参数化的梯度下降（Adam优化）以及OpenCV P3P求解器；

**📊 数据集**

在MPI-INF-3DHP、Human3.6M、AIST++三大公开3D HPE数据集以及自采集的Kinect+RGB实景数据上进行实验；

**📈 对比分析**

通过2D MPJPE与原始GT对比评估，RPGD在Human3.6M将误差从2.59像素降至1.26像素，在AIST++从11.98像素降至8.39像素，接近或匹配原始GT；

**⚠️ 局限性**

对相机内参误差敏感，尚未支持多人人体场景，运行速度虽已较快但仍可进一步优化。

---

## 570. IDPruner: Harmonizing Importance and Diversity in Visual Token Pruning for MLLMs

**arXiv ID:** 2602.13315 | [PDF](https://arxiv.org/pdf/2602.13315v1)

**作者:** Yifan Tan `[一作]` (Tsinghua University), Yangdong Deng `[通讯]` (Tsinghua University)

**通讯引用:** 2615 | [OpenAlex ID](https://openalex.org/A5059155953)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种视觉令牌剪枝方法IDPruner，旨在在保持多模态大型语言模型（MLLM）性能的同时显著降低视觉令牌数量。

**💡 创新点**

创新点包括：① 对重要性与多样性之间的权衡进行系统化分析；② 将Maximal Marginal Relevance（MMR）框架引入视觉令牌剪枝，实现重要性与语义多样性的最优平衡；③ 采用一-shot剪枝且不依赖注意力图，保证与FlashAttention的完全兼容，提升推理效率。

**🔧 技术方法**

技术细节：使用MMR进行贪婪选择，结合归一化的重要性分数与余弦相似度；利用最大相似度向量高效更新；在不计算注意力图的情况下完成剪枝；使用min‑max归一化与Cosine相似度作为度量；实现低复杂度 O(KN) 的剪枝算法。

**📊 数据集**

数据集与模型：图像‑语言基准包括 MME、MMBench、MMStar、POPE、ScienceQA、AI2D、TextVQA、ChartQA、DocVQA、OCRBench；视频‑语言基准为 Vinoground、VideoMME、SEED‑Bench；模型涵盖 Qwen2.5‑VL‑7B‑Instruct 和 LLaVA‑1.5‑7B。

**📈 对比分析**

与多类基线（FastV、VisionZip、HiPrune、VisionSelector、DivPrune、DART、VisPruner、SCOPE）进行对比。IDPruner 在 Qwen2.5‑VL‑7B‑Instruct 上分别在 25% 及 10% 令牌保留时取得 95.18% 与 86.47% 的平均分，超过所有对手；在 LLaVA‑1.5‑7B 上仅保留 32 个令牌时仍保持 87.43% 的平均分。与其他混合方法相比，IDPruner 在推理速度上更快，prefill 1337.76 ms，end‑to‑end 1478.32 ms。

**⚠️ 局限性**

局限性：需要手动调节 λ 超参数以平衡重要性与多样性；目前仅在一次性剪枝（one‑shot）阶段实现，缺乏动态序列长度适应；在极端高压缩率（如 90% 剪枝）下性能仍有下降；对不同任务的细粒度需求可能仍需进一步微调。

---

## 571. MUKA: Multi Kernel Audio Adaptation Of Audio-Language Models

**arXiv ID:** 2602.14127 | [PDF](https://arxiv.org/pdf/2602.14127v1)

**作者:** Reda Bensaid `[一作]` (IMT Atlantique), Adnane Boukhayma `[通讯]` (Inria)

**通讯引用:** 604 | [OpenAlex ID](https://openalex.org/A5020712348)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出多核产品框架 MUKA，用于音频-语言模型在少样本情境下的无训练适配

**💡 创新点**

创新点在于将细粒度指令调优特征（Pengi）与全局语义对比特征（CLAP）通过核乘积构建多核产品，兼顾局部细节与全局语义，既保持核方法的理论保证，又无需额外训练

**🔧 技术方法**

使用核岭回归（ProKeR）框架，构造产品核 k(x,x′)=k_Pengi(ϕ_Pengi(x),ϕ_Pengi(x′))·k_CLAP(ϕ_CLAP(x),ϕ_CLAP(x′))，实现无训练的缓存式适配

**📊 数据集**

在 11 个音频数据集上评估：ESC50、Beijing-Opera、CREMA-D、ESC50-Actions、GT-Music-Genre、NS-Instruments、RAVDESS、SESA、TUT2017、UrbanSound8K 与 VocalSound

**📈 对比分析**

与 Zero-shot、Treff-Adapter（训练‑free）、CoOp、CoCoOp、PaLM、Linear Probing（训练‑based）比较，MUKA 在训练‑free 组中大幅领先，并在多数数据集上接近甚至超过训练‑based 方法，整体提升 80.90% 的平均准确率

**⚠️ 局限性**

仅依赖预训练编码器，缺乏可学习的核函数，可能在不同任务或更大数据量下泛化受限；对推理时间和内存占用的细节未充分评估

---

## 572. CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning

**arXiv ID:** 2602.14068 | [PDF](https://arxiv.org/pdf/2602.14068v1)

**作者:** Yuhui Wu `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 105840 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CoCoEdit框架，利用后训练方式在生成模型上实现内容一致的图像编辑；

**💡 创新点**

创新点在于引入像素级相似度奖励和基于区域的正负正则化，兼顾编辑效果与内容保持；

**🔧 技术方法**

采用强化学习（PPO/GRPO类框架）结合DiffusionNFT的流匹配策略，使用像素级PSNR/SSIM奖励和MLLM评分；

**📊 数据集**

构建了CoCoEdit-40K（约4万条本地编辑样本，包含输入图像、指令和掩码），并对GEdit-Bench与ImgEdit-Bench加注掩码；

**📈 对比分析**

在扩展的Benchmarks上与多种基线（FLUX.1 Kontext、Qwen-Image-Edit等）以及RL方法（Edit-R1、MotionNFT）对比，CoCoEdit在PSNR/SSIM上提升1–3 dB，同时编辑分数保持或略有提升，且用户研究也显示更高偏好；

**⚠️ 局限性**

局限性包括对全局编辑任务的覆盖有限、对奖励模型的依赖导致训练成本较高、可能在复杂多目标编辑场景下表现不如专门模型。

---

## 573. The Inevitability of Side-Channel Leakage in Encrypted Traffic

**arXiv ID:** 2602.14055 | [PDF](https://arxiv.org/pdf/2602.14055v1)

**作者:** Guangjie Liu `[一作]`, Weiwei Liu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出基于信息理论的加密流量侧信道泄漏存在性定理，并给出互信息下界

**💡 创新点**

首次从系统效率约束出发严格证明侧信道泄漏不可避免，阐明效率‑隐私三元权衡

**🔧 技术方法**

采用信息理论（互信息、数据处理不等式、总变差、Chernoff信息）与概率模型、统一轨迹空间、Lipschitz统计等技术

**📊 数据集**

未使用实验数据，理论推导基于通用的加密通信模型（TLS 1.3/QUIC）与标准概率空间

**📈 对比分析**

未给出实验对比，理论结果提供互信息、准确率、误差下界等量化指标，可用于评估防御方案的理论极限

**⚠️ 局限性**

依赖映射非退化、观测非退化等假设，参数如C、Δ̅、Lφ需从实际流量估计；对动态/主动攻击、分布式场景等情况尚未完整分析

---

## 574. A Tale of Two Graphs: Separating Knowledge Exploration from Outline Structure for Open-Ended Deep Research

**arXiv ID:** 2602.13830 | [PDF](https://arxiv.org/pdf/2602.13830v1)

**作者:** Zhuofan Shi `[一作]` (Microsoft), Dongmei Zhang `[通讯]` (Microsoft)

**通讯引用:** 11371 | [OpenAlex ID](https://openalex.org/A5100331488)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种双图记忆框架（Dual-Graph memory），将知识表示与报告结构分离，维护一个知识图谱（KG）与一个大纲图（OG）共同演进，用以指导搜索和写作。

**💡 创新点**

创新点在于：①将知识图谱作为主动的认知状态，能够显式发现知识缺口并驱动有针对性的搜索；②利用图拓扑与语义聚类结合，生成搜索链；③在OG和KG之间实现一致性与引用持久化，避免信息重复和矛盾。

**🔧 技术方法**

技术包括：LLM（如GPT‑4.1 / GPT‑5）用于知识抽取、关系抽取、文档摘要与查询生成；图算法（Leiden社区检测、语义聚类、SBM概率模型）用于图更新与缺口发现；自终止评估器用于判断迭代结束；Web检索、页面抓取和证据库管理。

**📊 数据集**

使用公开基准数据集：DeepResearch Bench、DeepResearchGym、DeepConsult，覆盖22个学科和约96,000个真实科研问题。

**📈 对比分析**

与多种开源与商业系统对比，“Dual-Graph”在DeepResearch Bench上取得RACE 53.08（GPT‑5），在DeepResearchGym上获得94.31的平均分，在DeepConsult上实现64.42%胜率，均优于或接近最先进的商业系统，且迭代次数更少、有效引用更高。

**⚠️ 局限性**

局限性包括：①额外的KG维护增加了LLM调用和令牌开销，尽管相对搜索成本较小；②KG质量高度依赖LLM的抽取准确性，错误可能传播到规划；③目前仅验证文本级研究，跨模态或更大规模知识库的效果尚未评估。

---

## 575. Spectral Collapse in Diffusion Inversion

**arXiv ID:** 2602.13303 | [PDF](https://arxiv.org/pdf/2602.13303v1)

**作者:** Nicolas Bourriez `[一作]` (Ecole Normale Supérieure PSL), Auguste Genovesio `[通讯]` (Ecole Normale Supérieure PSL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在无配对图像翻译中，针对源域光谱稀疏导致的谱崩塌问题，提出了正交方差引导（OVG）方法；

**💡 创新点**

创新点是将高频方差注入限定在结构梯度正交的零空间，从而在保持结构不失真的同时恢复高频纹理；

**🔧 技术方法**

使用条件扩散模型的概率流ODE、EDM/ DDIM 参数化以及多任务梯度投影技术；

**📊 数据集**

在生物显微镜超分辨率数据集BBBC021（×8/×16/×32）和边缘到鞋子数据集Edges2Shoes上进行实验；

**📈 对比分析**

与标准DDIM、Null-Class、DirectInv、ReNoise、TABA以及EDM等方法对比，OVG在HF/LF得分、S_Decorr、FID、LPIPS等指标上实现了最佳或接近最佳的结构‑纹理平衡；

**⚠️ 局限性**

局限在于对极度稀疏输入（如×32）仍存在挑战，需手动调节指导系数，且方法假设ODE动态可控，可能不适用于所有扩散框架。

---

## 576. Learning Robust Markov Models for Safe Runtime Monitoring

**arXiv ID:** 2602.14987 | [PDF](https://arxiv.org/pdf/2602.14987v1)

**作者:** Antonina Skurka `[一作]` (Chalmers University of Technology and University of Gothenburg), Hazem Torfah `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于区间隐马尔可夫模型（iHMM）的鲁棒运行时监视器，通过学习不确定性模型来估计安全违规风险并在监控中保持保守性；

**💡 创新点**

创新点在于：1）将不确定性以区间形式编码的iHMM用于监控，确保风险估计上界；2）引入线性更新区间（LUI）与合规性测试驱动的迭代细化，显著提升样本效率；3）提供多项式时间监控算法并证明与理想监视器收敛；

**🔧 技术方法**

技术包括：区间隐马尔可夫模型、前向滤波、最大化风险估计、LUI学习、合规性测试驱动的细化、基于MDP的最大可达概率求解；

**📊 数据集**

使用改编自[相关工作]的航空跑道降落与机器人避碰等CPS基准：SnL-10x10、evadeV系列、airportA/B系列（含粗糙版本），并在不同状态空间和噪声设置下进行评估；

**📈 对比分析**

与两种模型无关方法（SGD回归、基于神经网络的 conformal prediction）对比；在FPR、FNR和AUC等指标上，iHMM监视器保持更低FNR且不显著增加FPR，远优于模型无关方法；在样本效率上，细化学习仅需约两/三分之一状态采样即可接近理想监视器；

**⚠️ 局限性**

局限性包括：对粗糙状态空间和高停止阈值时收敛速度下降；对大规模基准（如airportA-7-40-20、airport-B-7-40-20）仍难以完全逼近理想监视器；当前实现未针对监控执行速度进行优化，iHMM监视器的执行时间相对 HMM/模型无关方法较慢；

---

## 577. Solving Inverse Parametrized Problems via Finite Elements and Extreme Learning Networks

**arXiv ID:** 2602.14757 | [PDF](https://arxiv.org/pdf/2602.14757v1)

**作者:** Erik Burman `[一作]`, Jonatan Vallin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一个基于有限元和参数域插值（低维使用传统插值，高维使用极限学习机）的一体化参数化PDE降阶建模框架，并将其应用于定量光声层析中的潜在函数恢复。

**💡 创新点**

创新点在于：①将参数空间的插值方法与有限元空间分离，提供统一的错误估计；②在高维参数域引入随机特征ELM并给出随机逼近与最小范数插值的稳定性理论；③在逆问题中给出潜在函数和参数的误差界，证明了逼近误差可被控制。

**🔧 技术方法**

使用技术包括：有限元离散、Sobolev 正则性分析、张量产品/单纯形插值、极限学习机（随机 ReLU 单隐藏层网络）、Barron 空间误差理论、梯度下降逆问题求解。

**📊 数据集**

数据集：在数值实验中使用合成的二维空间域Ω_x=[-1,1]^2，制造解和参数，通过 Sobol 序列采样参数点；在QPAT逆问题实验中使用基于像素投影的观测算子，采样的潜在函数为高斯分布。

**📈 对比分析**

与传统直接求解或全局多重网格等方法比较，所提出的 ROM 在高维参数域下实现了大约 M^{-1/2} 的逼近率，逆问题的重建误差显著低于直接方法，同时显著降低了计算成本。

**⚠️ 局限性**

局限性包括：对参数空间的光滑性要求（需要满足 Barron 空间条件）；随机特征方法在小样本或噪声较大时收敛慢；并且在极限学习机下的稳定常数未明确给出，需要经验选择。

---

## 578. MacNet: An End-to-End Manifold-Constrained Adaptive Clustering Network for Interpretable Whole Slide Image Classification

**arXiv ID:** 2602.14509 | [PDF](https://arxiv.org/pdf/2602.14509v1)

**作者:** Mingrui Ma `[一作]` (Xinjiang University), Jing Qin `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种端到端的 Manifold‑Constrained Adaptive Clustering Network（MacNet），通过 Grassmann 重嵌入、流形自适应聚类和代理实例标签实现可解释的 Whole‑Slide Image（WSI）分类；

**💡 创新点**

创新点在于将 Grassmann 重嵌入与流形约束自适应聚类相结合以获得鲁棒聚类结果，并通过代理实例标签解决聚类标签歧义，实现全部步骤可端到端训练；

**🔧 技术方法**

使用 Swin Transformer 编码器、Geodesic Flow Kernel（Grassmann 重嵌入）、流形约束自适应 K‑means、代理实例聚合以及交叉熵+距离正则的联合损失；

**📊 数据集**

评估使用公开数据集 CAMELYON16、DHMC‑LUNG，以及两套军方私有数据集 AMU‑LSCC 与 AMU‑CSCC；

**📈 对比分析**

与 11 种 SOTA MIL 模型（ABMIL、CLAM、IBMIL、DGR‑MIL、TransMIL 等）在 ACC/AUC 上比较，MacNet 在所有数据集均领先，提升幅度可达 1–10% 以上；

**⚠️ 局限性**

局限性包括模型参数量大（≈27.5M）、训练时间较长（约 145s/epoch），聚类数固定为 3，适用于三类任务，对更复杂多类别任务需要进一步扩展。

---

## 579. Configuring Agentic AI Coding Tools: An Exploratory Study

**arXiv ID:** 2602.14690 | [PDF](https://arxiv.org/pdf/2602.14690v1)

**作者:** Matthias Galster `[一作]` (University of Bamberg), Sebastian Baltes `[通讯]` (Heidelberg University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对五款Agentic AI编码工具的配置机制进行系统梳理，并在2,926个GitHub OSS仓库中对其采用情况进行量化分析。

**💡 创新点**

首次将八种配置机制（Context Files、Skills、Subagents、Commands、Rules、Settings、Hooks、MCP）一并归纳，揭示Context Files主导、Skills与Subagents浅度采用以及工具间的配置文化差异。

**🔧 技术方法**

利用Python爬虫、正则和OpenAI GPT-4对仓库代码及配置文件进行检索与分类，并统计共性与关联性。

**📊 数据集**

收集了37,249个被视为“工程化”的GitHub仓库，最终筛选出2,926个使用Agentic工具的仓库作为样本。

**📈 对比分析**

通过对不同工具、不同配置机制的出现频率、共现关系和时间演化进行统计与可视化，展示Context Files占比超过70%，而Skills/Subagents平均仅2个，未与性能直接关联。

**⚠️ 局限性**

研究受限于仅覆盖公开GitHub仓库、仅检测文件存在未验证实际使用、工具生态快速演进导致识别规则可能落后，且未对配置效果做因果实验。

---

## 580. Audience in the Loop: Viewer Feedback-Driven Content Creation in Micro-drama Production on Social Media

**arXiv ID:** 2602.14045 | [PDF](https://arxiv.org/pdf/2602.14045v1)

**作者:** Gengchen Cao `[一作]` (Tsinghua University), RAY LC `[通讯]` (City University of Hong Kong)

**通讯引用:** 909 | [OpenAlex ID](https://openalex.org/A5027284786)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对中国社交媒体平台（如抖音、快手）上微剧制作过程进行访谈研究，揭示创作者如何在受观众反馈和平台算法驱动下迭代剧本与制作流程。

**💡 创新点**

提出“观众在环（Audience in the Loop）”的创作范式，系统识别出多角色融合的工作流程、基于观众实时反馈的迭代机制，并概念化平台与创作者之间的协同共创关系。

**🔧 技术方法**

采用定性方法：半结构化访谈、主题编码与归纳分析；未实现具体算法或模型，仅对访谈内容进行分析。

**📊 数据集**

数据集为 28 名微剧创作者（编剧、导演、制片、演员等）以及 4 名传统剧本创作者的访谈记录，涵盖 6 个月至 9 年的从业经验。

**📈 对比分析**

论文未进行实验对比或性能评估；仅通过访谈分析得出工作模式与反馈循环的定性结论。

**⚠️ 局限性**

局限性包括：受访者自我报告的主观性；缺乏观众视角；未对微剧文本或平台数据进行量化验证；样本仅来自中国，缺乏跨文化比较；方法受限于访谈而非大规模内容分析。

---

## 581. An Online Reference-Free Evaluation Framework for Flowchart Image-to-Code Generation

**arXiv ID:** 2602.13376 | [PDF](https://arxiv.org/pdf/2602.13376v1)

**作者:** Giang Son Nguyen `[一作]` (VinUniversity), Wenya Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 4555 | [OpenAlex ID](https://openalex.org/A5101936536)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无参考的流图图像到代码生成评估框架，用于生产环境下的质量监控

**💡 创新点**

创新点在于引入Recall_OCR（基于OCR的覆盖率估计）和Precision_VE（基于视觉蕴含的真伪检测）两项无监督指标，并将其融合成F1_OCR-VE统一质量分数

**🔧 技术方法**

主要技术包括OCR文本抽取、视觉蕴含（Visual Entailment）推理、Levenshtein相似度匹配、微平均F1统计与相关性评估

**📊 数据集**

使用FlowVQA数据集（随机抽样197张图像及对应Mermaid代码）进行验证，另外参考FlowLearn等其他流图数据集

**📈 对比分析**

与传统基于真实标签的Recall_Actual、Precision_Actual、F1_Actual进行Pearson、Kendall相关性、RMSE、MAE比较；平均Pearson r分别为0.97、0.91、0.94，F1差距不足3个百分点，验证了框架的可靠性

**⚠️ 局限性**

局限性在于Recall_OCR仅评估文本标签节点，无法覆盖无标签边的召回；且依赖高质量OCR和VE模型，低FPR的VE模型对幻觉检测尤为关键

---

## 582. BFS-PO: Best-First Search for Large Reasoning Models

**arXiv ID:** 2602.14917 | [PDF](https://arxiv.org/pdf/2602.14917v1)

**作者:** Fiorenzo Parascandolo `[一作]` (University of Modena and Reggio Emilia), Rita Cucchiara `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为大型推理模型设计了一种基于最佳优先搜索的强化学习算法，目标是减少过度推理并提升推理准确率。

**💡 创新点**

创新点在于使用最佳优先搜索与熵最大化回溯点相结合的探索策略，以及在搜索树上定义的分支优势（branch advantage）来替代传统的PRM，既减少了训练成本，又能自生成简洁的推理链。

**🔧 技术方法**

采用强化学习（GRPO/DAPO的改进版）结合最佳优先搜索、熵计算、分支优势计算与长度惩罚，模型为大型语言模型的微调。

**📊 数据集**

在GSM8K、MATH‑500、MMLU‑STEM等推理基准上进行实验，并在AIME’25和MINERVA‑MATH上做跨域评测。

**📈 对比分析**

与零样本、SFT、TokenSkip、DAPO、Tree of Thoughts等方法比较，BFS‑PO在保持甚至提升准确率的同时显著缩短了链长，整体AES分数最高。

**⚠️ 局限性**

仅在3–8B规模模型上验证，扩展到更大模型仍需高昂计算成本，且RL训练对资源要求较高。

---

## 583. BEAGLE: Behavior-Enforced Agent for Grounded Learner Emulation

**arXiv ID:** 2602.13280 | [PDF](https://arxiv.org/pdf/2602.13280v1)

**作者:** Hanchen David Wang `[一作]` (Vanderbilt University), Meiyi Ma `[通讯]` (Vanderbilt University)

**通讯引用:** 1515 | [OpenAlex ID](https://openalex.org/A5101671027)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 BEAGLE，一个将半马尔可夫行为控制、贝叶斯知识跟踪与两阶段 LLM 策略/执行器相结合的神经符号框架，用于生成高保真度的模拟学生学习轨迹。

**💡 创新点**

创新点在于通过结构化的行为模型和知识约束显式抑制 LLM 的“能力偏见”，并通过任务分解（Strategist/Executor）防止内部自我纠错，从而同时实现行为、认知与感知三维真实性。

**🔧 技术方法**

主要技术包括半马尔可夫模型用于时序行为规划、带缺陷注入的贝叶斯知识跟踪、任务框架拆分的 Strategist/Executor 生成管道，以及通过 EFI 阻塞特定知识库实现的真实知识缺口。

**📊 数据集**

使用了真实学生日志（227 条元认知段落）和 2D 物理仿真/ Python 编程任务的数据集，训练和评估 BEAGLE 的学习轨迹。

**📈 对比分析**

与 Vanilla、CoT、Few-Shot、SimStudent、LLMSS、CoderAgent 等基线相比，BEAGLE 在行为 D_KL、错误递归率、感知现实评分和 Turing 测试中的表现分别提升至 0.43、86.2%、2.68/3 与 52.8% 难以区分的水平，证明了其在三维真实性上的优势。

**⚠️ 局限性**

局限性包括各真实性维度之间的张力（去除知识约束会导致行为真实性下降）、对不同编程语言或学科的泛化能力尚未验证，以及对 LLM 计算资源与训练成本的依赖。

---

## 584. MAPLE: A Sub-Agent Architecture for Memory, Learning, and Personalization in Agentic AI Systems

**arXiv ID:** 2602.13258 | [PDF](https://arxiv.org/pdf/2602.13258v1)

**作者:** Deepak Babu Piskala `[一作]` `[通讯]`, Deepak Babu Piskala

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了MAPLE架构，将智能体的适配功能拆分为记忆、学习和个性化三大子系统。

**💡 创新点**

创新点在于将记忆、学习与个性化严格解耦并分别在不同时间尺度上实现，支持实时响应与后台持续改进。

**🔧 技术方法**

采用大型语言模型（Claude 3 Sonnet/Haiku）、数据库/向量存储、LLM驱动的符号学习与上下文注入技术。

**📊 数据集**

使用自行构建的MAPLE-Personas基准数据集，包含150个合成用户画像与10轮对话。

**📈 对比分析**

与无记忆基线对照，利用LLM-as-judge评估个性化得分、特征融入率，MAPLE显著提升平均得分从4.17提升至4.78，特征融入率从45%提升至75%。

**⚠️ 局限性**

局限性包括依赖合成数据和LLM评测的偏差、跨用户隐私与协调开销，以及缺乏人类评估验证。

---

## 585. CrisiSense-RAG: Crisis Sensing Multimodal Retrieval-Augmented Generation for Rapid Disaster Impact Assessment

**arXiv ID:** 2602.13239 | [PDF](https://arxiv.org/pdf/2602.13239v1)

**作者:** Yiming Xiao `[一作]` (Texas A&M University), Ali Mostafavi `[通讯]` (Texas A&M University)

**通讯引用:** 7007 | [OpenAlex ID](https://openalex.org/A5023165780)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在灾害事件中快速评估洪水范围和结构损坏，利用多模态检索增强生成（RAG）框架对异步数据流进行融合；

**💡 创新点**

提出分管道（split‑pipeline）架构和时间意识融合逻辑，解决文本与影像异步对齐问题，并通过指标对齐提示提升零样本量化预测；

**🔧 技术方法**

使用多模态RAG（文本检索：dense+BM25+cross‑encoder，图像检索：CLIP），以及Gemini、Llama 3.3 70B、Qwen 2.5 72B等预训练LLM进行文本与视觉分析，结合GPT‑4o进行视觉推理；

**📊 数据集**

以2017年飓风哈维（Hurricane Harvey）为例，使用3,507张高分辨率航空影像、图像字幕、约458k条Twitter/X推文、26k条311呼叫记录、雨量计数据，以及FEMA洪水深度网格和Property Damage Extent（PDE）作为基准；

**📈 对比分析**

通过文本‑仅、文本+字幕、完整多模态三种配置的零样本评估，比较各模型的Extent MAE（10.94%–28.40%）和Damage MAE（16.47%–21.65%），并与提示工程消融结果对比，表明指标对齐和时间上下文提示能显著提升性能；

**⚠️ 局限性**

仅在哈维事件中验证，泛化到其他灾害与地区尚未证实；依赖社交媒体覆盖，易受偏差影响；语义检索引入空间噪声，PDE覆盖仅139个ZIP码，导致评估局限；

---

## 586. When Security Meets Usability: An Empirical Investigation of Post-Quantum Cryptography APIs

**arXiv ID:** 2602.14539 | [PDF](https://arxiv.org/pdf/2602.14539v1)

**作者:** Marthin Toruan `[一作]` (Royal Melbourne Institute of Technology), Nalin Arachchilage `[通讯]` (Royal Melbourne Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文通过远程可用性实验，系统评估了两种后量子密码学API（QuantCrypt与PQ-Sandbox）的易用性，记录了开发者在实现过程中遇到的障碍、任务完成时间与错误率。

**💡 创新点**

首次以实证方法全面分析PQc API的可用性，揭示了关键误用模式，并提出高层次安全默认设计、改进文档与示例代码的可操作性建议。

**🔧 技术方法**

采用认知维度框架（CDF）配合思考大声法的混合方法，收集屏幕录像、访谈记录和问卷数据。

**📊 数据集**

实验数据仅来自16名开发者的任务完成记录、错误计数与问卷结果，没有使用公开数据集。

**📈 对比分析**

通过比较任务完成率、耗时和错误率两种API的表现；QuantCrypt在任务1完成更快，但任务2耗时更长；PQ-Sandbox在某些任务更易上手，但两者均存在安全实现缺陷。

**⚠️ 局限性**

样本规模有限、实验环境文档受限、仅评估可用性未覆盖安全性与生产适配性。

---

## 587. Learning Vocal-Tract Area and Radiation with a Physics-Informed Webster Model

**arXiv ID:** 2602.13834 | [PDF](https://arxiv.org/pdf/2602.13834v1)

**作者:** Minhui Lu `[一作]` (Queen Mary University of London), Joshua D. Reiss `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理的时间域 Webster PINN，利用可学习的 Robin 边界和可微探针来从单通道语音中识别声道几何和辐射参数，随后在独立的 FDTD–Webster 求解器中验证其音频合成效果。

**💡 创新点**

创新点在于将物理信息约束与可微探针相结合，形成轻量化、可校准的 PINN 结构；同时通过独立渲染验证学习到的参数物理可解释性。

**🔧 技术方法**

使用了 Physics‑Informed Neural Networks（PINN）+ Webster 方程、可微 DDSP 渲染器、SIREN 网络、可微形式元和谐波包络探针。

**📊 数据集**

训练数据为合成的持续元音 /a, i, u/（使用 FDTD–Webster 求解器生成），采样率 16kHz。

**📈 对比分析**

与仅基于 DDSP 的基线对比，PINN 在独立渲染下的 LSD 约 4.5–5.5 dB、HNR 与参考相近；mSTFT 同样低于基线，显示出良好的音频重建质量。

**⚠️ 局限性**

局限在于直接从 PINN 生成的波形往往过于呼啸、缺乏周期性；需要更强的声源约束或相位一致性训练。

---

## 588. BitDance: Scaling Autoregressive Generative Models with Binary Tokens

**arXiv ID:** 2602.14041 | [PDF](https://arxiv.org/pdf/2602.14041v1)

**作者:** Yuang Ai `[一作]` (ByteDance), Hao Chen `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 BitDance，一种基于二进制视觉令牌的自回归图像生成框架，能在 256×256 甚至 1024×1024 分辨率下生成高质量图像。

**💡 创新点**

核心创新包括：① 超大 2^256 词表的二进制量化器；② 通过在连续空间建模二进制超立方体的扩散头，解决大词表采样瓶颈；③ 下一‑patch 扩散设计，让模型一次并行生成多个视觉令牌，显著提升推理速度。

**🔧 技术方法**

技术手段包括：Lookup‑Free Quantization (LFQ)、二进制扩散头（x‑prediction 形式）、块级因果注意力（block‑wise mask）、Transformer 预训练（以大型 LLM 为后端）、混合分辨率训练、以及知识蒸馏实现更高并行采样。

**📊 数据集**

训练数据涵盖 ImageNet（分类），LAION（大规模图文对），以及少量高质量文本‑图像对（如 Seedream、Z‑Image‑Turbo）。总计不到 450M 图文对，显著低于现有商业模型的数据规模。

**📈 对比分析**

与现有自回归模型（如 NextStep‑1、GLM‑Image）和扩散模型（Qwen‑Image、Z‑Image、BAGEL）对比，BitDance 在 ImageNet 256×256 上 FID 1.24、IS 1.69、Precision/Recall 0.83/0.64；在 GenEval、DPG‑Bench、OneIG 等文本‑图像基准上获得 0.86/88.28/88.28 的高分；在 1024×1024 生成时，蒸馏版推理时延仅 12.4 s（比 AR 20–400 s 低 30×，比扩散 20–23 s 低 2–3×）。

**⚠️ 局限性**

局限性在于：① 词表扩大需大 Transformer 与大量计算资源；② 仍未能完全匹敌最先进的商业扩散模型在某些细节或多样性指标；③ 训练和推理成本高，尤其是大并行步数的扩散头；④ 该二进制表示可能限制对极细粒度或连续色彩变化的建模，需进一步研究更灵活的量化策略。

---

## 589. MeFEm: Medical Face Embedding model

**arXiv ID:** 2602.14672 | [PDF](https://arxiv.org/pdf/2602.14672v1)

**作者:** Yury Borets `[一作]` (Sber AI Lab), Stepan Botman `[通讯]` (Sber AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

基于修改后的JEPA，提出MeFEm模型用于医学面部特征提取与预测。

**💡 创新点**

引入轴向条纹遮罩、圆形损失加权和CLS随机分配等改进，提升对医学相关特征的学习。

**🔧 技术方法**

采用ViT编码器、JEPA自监督预测、概率CLS分配、轴向遮罩以及圆形加权等技术。

**📊 数据集**

训练集由FaceCaption‑15M、AVSpeech、SFHQ三源图像共约644万张；评估集包括CelebA、FairFace、BMI三源数据以及MCD‑rPPG的医学指标。

**📈 对比分析**

与FaRL、Franca、Buffalo等基线对比，MeFEm在BMI、年龄、性别、种族等任务上r²/MAE/准确率均优于基线；单帧医学参数预测虽低于视频rPPG，但表现出一定关联。

**⚠️ 局限性**

主要限制为单帧预测的医学指标表现仍不佳、训练数据量有限，且模型对真实医学标注的泛化尚未验证。

---

## 590. Diagnosing Knowledge Conflict in Multimodal Long-Chain Reasoning

**arXiv ID:** 2602.14518 | [PDF](https://arxiv.org/pdf/2602.14518v1)

**作者:** Jing Tang `[一作]` (Huazhong University of Science and Technology), Zhigang Zeng `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多模态大语言模型在长链推理过程中出现知识冲突的诊断与控制

**💡 创新点**

提出客观冲突与有效冲突区分，揭示冲突信号在中后层线性可分且存在方向不对称的控制难度

**🔧 技术方法**

采用流式探针、线性/MLP 归纳、注意力偏移、视觉对比解码、表示导向与探针引导控制等技术

**📊 数据集**

构造了 7500+ 目标冲突的长 CoT 基准（Vision‑Prior、Vision‑Text、Prior‑Text 三种冲突类型）

**📈 对比分析**

在 R1‑Onevision 7B、Ocean‑R1 7B、Llama‑3.2V 11B 三大后端上进行比较，探针 AUC 达 93–99%，干预可将冲突率降低 30–80%，并提升语义一致性

**⚠️ 局限性**

仅针对二元冲突、依赖已标注源，逆向控制效果差，对更复杂多源冲突缺乏泛化，且方法在视觉+文本冲突之外的表现有限

---

## 591. UniWeTok: An Unified Binary Tokenizer with Codebook Size $\mathit{2^{128}}$ for Unified Multimodal Large Language Model

**arXiv ID:** 2602.14178 | [PDF](https://arxiv.org/pdf/2602.14178v1)

**作者:** Shaobin Zhuang `[一作]` (Shanghai Jiao Tong University), Yali Wang `[通讯]` (Shenzhen Institutes of Advanced Technology)

**通讯引用:** 4373 | [OpenAlex ID](https://openalex.org/A5100335699)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了大规模离散视觉词元化器 UniWeTok，并基于其构建统一多模态大型语言模型（Unified MLLM）用于图像压缩、理解与生成。

**💡 创新点**

创新点包括 2^128 码本与 32× 下采样实现高信息密度压缩、SigLu 激活解决 token entropy 与 commitment 的冲突、Pre‑Post Distillation（PPD）与 Generative‑Aware Prior（GAP）双重损失、三阶段渐进式训练方案以及卷积‑注意力混合骨干。

**🔧 技术方法**

采用了 Group‑Wise Lookup‑Free Quantization、卷积+Transformer 双重编码器/解码器、语义编码器蒸馏、GAN+LPIPS 视觉损失、Entropy Loss、MSE 训练小 BitDance 模型、SigLu 激活等技术。

**📊 数据集**

使用 ImageNet、DataComp‑1B、MS‑COCO 2017 等大规模视觉数据集，并在 SEEDB、POPE、VQAv2、GQA、SQA、TQA、CQA、AI2D、RWQA、MMMU、MME、GenEval、DPG‑Bench、GEdit 等多种下游评测数据集进行评估。

**📈 对比分析**

与现有 VQ‑VAE、VQGAN、UniTok、REPA 等 tokenizer 及生成模型对比，UniWeTok 在 ImageNet 上取得 FID 1.38、仅需 33B 训练 token、64 token 推理；在 Unified MLLM 任务上在多项理解与生成基准上均优于或与主流模型持平，显示出显著的性能提升和训练成本下降。

**⚠️ 局限性**

局限性主要体现在大码本导致的训练与推理资源需求仍较高，极高分辨率或细粒度细节的捕获仍有限，对非自然图像或专用领域的泛化能力有待进一步验证。

---

## 592. What happens when reviewers receive AI feedback in their reviews?

**arXiv ID:** 2602.13817 | [PDF](https://arxiv.org/pdf/2602.13817v1)

**作者:** Shiping Chen `[一作]` (University College London), Anna L. Cox `[通讯]` (University College London)

**通讯引用:** 12877 | [OpenAlex ID](https://openalex.org/A5043657725)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究在ICLR 2025会议中部署了一套后评审阶段的AI反馈工具，并通过问卷调查与访谈评估评审者对该工具的感知、使用意愿与行为；

**💡 创新点**

创新点在于首次将AI后评审反馈工具在真实高风险学术会议中大规模实地部署，系统性记录评审者对AI生成反馈的接受度、修改行为及对所有权与责任的认知，揭示AI反馈既促进澄清与可操作性，又可能引发权威与抵触的双重张力；

**🔧 技术方法**

技术上采用多模型LLM管道（Claude Sonnet 3.5 等）实现对提交评审文本的模糊性、误解与不专业语调的检测，并生成针对性改进建议；

**📊 数据集**

数据集为ICLR 2025提交的超过20,000份评审报告，其中约43%获得AI反馈；

**📈 对比分析**

与传统无AI反馈的评审流程相比，研究发现约78%评审者考虑修改，约57%实际修改，修改主要集中在内容补充与表述澄清，整体提升了评审文本的长度与可操作性，但未能显著提升评审质量或一致性；

**⚠️ 局限性**

局限性包括样本偏向男性早期研究者、缺乏客观评审质量评估、仅覆盖ICLR的后评审阶段，且AI反馈主要聚焦表面表达，难以满足评审者对深层内容评价的需求。

---

## 593. On Calibration of Large Language Models: From Response To Capability

**arXiv ID:** 2602.13540 | [PDF](https://arxiv.org/pdf/2602.13540v1)

**作者:** Sin-Han Yang `[一作]` (Appier AI Research), Shao-Hua Sun `[通讯]` (National Taiwan University)

**通讯引用:** 2944 | [OpenAlex ID](https://openalex.org/A5112680841)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实证了针对大语言模型的“能力校准”框架，区别于传统的“响应校准”，并通过实验验证其在多任务下的有效性

**💡 创新点**

创新点在于：①定义了以模型期望准确率为目标的能力校准；②阐明了响应校准与能力校准的理论与实证差异；③提出基于线性探针的低成本能力估计方法

**🔧 技术方法**

主要技术包括：线性探针训练、词级概率与口头化置信度估计、采样一致性与P(True)方法；评估采用Brier分数与Pass@k、预算分配等任务

**📊 数据集**

使用了三大模型（Olmo‑3‑7B‑Instruct、Qwen3‑8B、gpt‑oss‑20b）在七个数据集（TriviaQA、SimpleQA、GSM8K、MATH‑500、AIME25、MMLU、GPQA）上进行实验

**📈 对比分析**

与随机、口头化置信度、P(True)及一致性等方法比较，线性探针在保持最低推理成本的同时显著降低Brier分数；在Pass@k模拟与计算预算分配任务中，探针与Oracle能力校准接近，明显优于响应校准方法

**⚠️ 局限性**

局限性在于：探针方法对模型权重开放，黑箱API无法直接使用；在跨域/离散任务上的泛化仍有限；对大型模型的推理成本仍高于单纯解码。

---

## 594. Federated Learning of Nonlinear Temporal Dynamics with Graph Attention-based Cross-Client Interpretability

**arXiv ID:** 2602.13485 | [PDF](https://arxiv.org/pdf/2602.13485v1)

**作者:** Ayse Tursucular `[一作]` (Georgia Institute of Technology), Nagi Gebraeel `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5826 | [OpenAlex ID](https://openalex.org/A5054372641)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种联邦学习框架，在每个客户端保持固定的专有EKF模型不变的前提下，通过在客户端增量学习和服务器端基于图注意网络的全局状态转移模型，实现跨客户端的非线性时序相互依赖学习与解释。

**💡 创新点**

创新点在于：①将图注意网络嵌入服务器端全局模型，利用注意力系数与雅可比矩阵的解析关系实现跨客户端时序依赖的可解释性；②通过仅传输低维隐藏状态和梯度实现在严格隐私约束下的端到端联邦学习；③提供理论证明框架收敛到中心化预言机并给出雅可比一致性界定。

**🔧 技术方法**

技术手段包括：非线性状态空间模型、扩展卡尔曼滤波器（EKF）客户端状态估计、图注意网络（GAT）服务器端全局转移、雅可比-注意力分析、联邦梯度通信、LSTM编码器（用于真实数据的多步依赖）。

**📊 数据集**

使用了两套数据集：合成系统（已知真实图结构和转移）和工业控制系统HAI benchmark（水处理、化学计量与加热三子系统）。

**📈 对比分析**

与四类基线（本地专有模型、中心化模型、预训练联邦一致图模型、NOTEARS-ADMM）对比，本文模型在客户端预测误差上显著优于本地和预训练基线，接近中心化模型，并在真实数据中实现较高的注意力相关性与雅可比相似度，证明其性能优异。

**⚠️ 局限性**

局限性包括：假设服务器端已知图结构且不具备学习图拓扑的能力；雅可比近似仅为一阶线性化，可能在强非线性场景下失效；对高度异构或多模态数据的适应性尚未验证。

---

## 595. InfoCIR: Multimedia Analysis for Composed Image Retrieval

**arXiv ID:** 2602.13402 | [PDF](https://arxiv.org/pdf/2602.13402v1)

**作者:** Ioannis Dravilas `[一作]` (University of Amsterdam), Marcel Worring `[通讯]` (University of Amsterdam)

**通讯引用:** 16038 | [OpenAlex ID](https://openalex.org/A5070684680)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了InfoCIR，一套可视化分析系统，用于诊断和改进组合图像检索（CIR）模型，并集成检索、嵌入投影、提示增强和多级解释可视化。

**💡 创新点**

创新点在于：① 将检索结果、UMAP投影、梯度热图、词级归因等多种视图统一交互；② 基于用户选定的理想图像进行提示重写，并通过实时Rank‑Δ热图验证效果；③ 采用风格去偏的UMAP投影提升语义可视化的可靠性。

**🔧 技术方法**

采用的技术包括：CLIP + SEARLE（文本逆转）嵌入、UMAP（含风格去偏与对比去偏）、Grad‑ECLIP梯度归因、LLM（Mistral‑7B）生成提示变体、Plotly/Dash实现的交互界面。

**📊 数据集**

使用的主要数据集是ImageNet‑R（用于实验评估），参考图像来自公开网页。

**📈 对比分析**

与仅显示检索结果的基线进行对比；定量实验显示：成功率从37.5%提升到87.5%，平均完成时间从277秒降至133秒，平均查询次数从7.4降至3.3，证明了系统显著提升检索效率与成功率。

**⚠️ 局限性**

限制包括：投影造成的度量失真、对初始检索的冷启动依赖、提示重写可能过拟合理想图像、LLM建议不一定有效、解释仅局部性、未能直接用于模型改进。

---

## 596. PlotChain: Deterministic Checkpointed Evaluation of Multimodal LLMs on Engineering Plot Reading

**arXiv ID:** 2602.13232 | [PDF](https://arxiv.org/pdf/2602.13232v1)

**作者:** Mayank Ravishankara `[一作]` `[通讯]` (Independent Researcher), Mayank Ravishankara (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了可重现的生成式基准 PlotChain，用于评估多模态大型语言模型在工程绘图量化读取任务中的表现。

**💡 创新点**

创新点包括：①通过确定性生成器实现精确可验证的地面真值；②引入检查点字段实现子技能诊断；③制定严格的 JSON 数值输出和容差评分机制；④全面发布数据、代码和原始模型输出，支持后期重评分。

**🔧 技术方法**

使用了多模态提示、确定性解码（温度0）、严格 JSON 输出、基于容差的数值评分、配套的解析脚本和统计检验（McNemar、配对 t 检验）等技术。

**📊 数据集**

基准数据集包含 15 种典型工程绘图族、共 450 张图像（每族 30 张），每张图附有自然语言问题、最终数值答案与若干检查点答案，并配有可复现的生成参数。

**📈 对比分析**

通过在统一的 Prompt+温度0+JSON 输出协议下评估四款主流多模态 LLM（Gemini 2.5 Pro、GPT‑4.1、Claude Sonnet 4.5、GPT‑4o），报告字段级通过率、项目级严格全部通过率、检查点通过率以及延迟；Gemini 2.5 Pro 在严格全部通过率上首位（72 %），GPT‑4.1 接近（68 %），Claude Sonnet 4.5 约 61 %，GPT‑4o 仅 32 %。

**⚠️ 局限性**

局限性包括：①基于合成图像，可能缺乏真实环境噪声和样式多样性；②容差策略对结果影响较大；③严格 JSON 接口可能因格式错误导致误判；④仅评估数值提取与轻度推理，未涵盖解释生成或领域知识需求。

---

## 597. CodeGlance: Understanding Code Reasoning Challenges in LLMs through Multi-Dimensional Feature Analysis

**arXiv ID:** 2602.13962 | [PDF](https://arxiv.org/pdf/2602.13962v1)

**作者:** Yunkun Wang `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**通讯引用:** 9286 | [OpenAlex ID](https://openalex.org/A5055284175)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 CodeGlance 的多维基准，用于评估 LLM 在三种实际编码场景（自包含逻辑、API 交互和未见函数）下的代码行为推理能力。

**💡 创新点**

创新点在于：①引入三种互补的、现实化的推理场景；②对 9 个代码复杂度特征进行系统影响分析；③对常用增强策略（CoT、RAG、代码检索）在不同场景下的有效性进行细粒度比较。

**🔧 技术方法**

采用了链式推理 (Chain‑of‑Thought)、检索增强生成 (RAG) 与代码搜索等技术，并利用 Qwen2.5‑Coder、GPT‑4o、DeepSeek‑Coder 等多模型进行评测。

**📊 数据集**

使用了 CRUXEval、DS‑1000、TorchdataCode、PanNumEval 及其改写版本 MonkBeatEval 共 1,969 条代码推理任务。

**📈 对比分析**

通过对 7 大 LLM 的 pass@1 与 pass@3 进行对比，发现模型规模越大在未见函数推理中提升最显著，CoT 在逻辑/ API 场景效果最佳，RAG 在大模型上对未见函数有一定帮助，但总体仍低于已知 API；整体峰值约 65‑68%。

**⚠️ 局限性**

局限性包括：①基准仅覆盖三种场景，未涵盖多语言或大型项目上下文；②特征集未考虑命名规范、注释质量等潜在影响因素；③对未见函数的处理依赖手工改写，缺乏更广泛的泛化评估。

---

## 598. When to Think Fast and Slow? AMOR: Entropy-Based Metacognitive Gate for Dynamic SSM-Attention Switching

**arXiv ID:** 2602.13215 | [PDF](https://arxiv.org/pdf/2602.13215v1)

**作者:** Haoran Zheng `[一作]` `[通讯]` (University of Chicago), Haoran Zheng (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种混合架构，先用状态空间模型(SSM)完成所有位置的线性前向计算，再通过预测熵门控决定哪些位置需要激活稀疏注意力，并用Ghost KV将SSM隐藏状态投影为键值对。

**💡 创新点**

创新点包括：① 用可解释的预测熵作为元认知门控信号，② 将注意力的键值直接从SSM隐藏状态中获取（Ghost KV），从而把本应昂贵的 O(n²) 计算降至 O(n)；③ 在仅 22% 的位置使用注意力即可获得完美检索结果，证明了稀疏路由的有效性。

**🔧 技术方法**

采用的技术包括：状态空间模型（实验中使用 GRU，可扩展到 Mamba 等）；熵门控（Straight‑Through Estimator + 可学习阈值）；稀疏 Top‑k 注意力；Ghost KV 键值投影；平衡损失用于控制门控率；交叉熵作为主训练目标。

**📊 数据集**

主要在两类人工合成检索任务上评估：Simple Retrieval Task（局部模式与远程检索混合）和 NeedleHaystack Task（存储–噪声–查询的严格检索）。没有使用公开的真实语料库，而是通过合成数据来验证方法的有效性。

**📈 对比分析**

对比方法包括 SSM‑Only、全注意力 Transformer（Transformer‑Only）以及 Oracle 门控（使用真值标签）。在 Simple Retrieval Task 上，Entropy 模型在检索位置达到 100% 的准确率，门控率 22%，参数约 77K；而 Transformer‑Only 仅 87%，SSM‑Only 仅 68%。在 NeedleHaystack 任务中，Entropy 取得 9.93% 的检索准确率，门控率 80.97%，显著优于 Transformer‑Only 的 4.40%，但仍低于 Oracle 的 37.08%。

**⚠️ 局限性**

限制包括：① Ghost KV 受限于 SSM 的状态衰退，无法检索距离过远的记忆；② 熵门只能在模型已感到不确定时触发，缺乏前瞻性的主动缓存机制；③ 当前实现未实现真正的条件执行，计算仍需为所有位置预先生成键值和注意力得分；④ 仅在合成任务上验证，缺乏在真实语言或视觉任务中的进一步实验。

---

## 599. EVECTOR: An orchestrator for analysing attacks in electric vehicles charging system

**arXiv ID:** 2602.13926 | [PDF](https://arxiv.org/pdf/2602.13926v1)

**作者:** Devki Nandan Jha `[一作]` (Newcastle University), Omer Rana `[通讯]` (Cardiff University)

**通讯引用:** 17013 | [OpenAlex ID](https://openalex.org/A5021973291)

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一个多层级的电动汽车充电系统安全模拟与攻击编排框架，能够整合多种EV充电仿真器（EVerest、ISO15118、ACNSim 等）并支持自定义网络与协议层攻击。

**💡 创新点**

创新点在于：① 将多种异构仿真器统一包装成可互操作的模拟套件；② 引入攻击编排器（支持破线、模糊等攻击）并将攻击日志按预定义模式结构化存储；③ 通过 MQTT 与 MongoDB 进行实时数据交换与持久化，为后续安全分析提供完整可追溯的数据链路。

**🔧 技术方法**

使用技术包括：Docker 容器化仿真器、MQTT 消息总线、MongoDB NoSQL 数据库、Pumba Chaos 工具进行网络层破线攻击、ISLa 语言模糊测试框架、Typed‑OCPP 校验库、Python/Node‑Red 交互接口。

**📊 数据集**

使用的数据集主要为仿真生成的充电会话日志（如 EV2Gym/ACNSim 的真实充电记录）和 ISO15118/OCPP 协议规范所定义的消息格式；未使用真实车辆的商业数据。

**📈 对比分析**

通过两种攻击案例（破线攻击和帧模糊攻击）演示框架效果；模糊测试得到不同消息类型的成功/失败率、错误码及平均延迟，说明后端对异常请求的容错与性能表现；但未与现有安全分析工具做系统化对比，性能指标以攻击成功率与响应延迟为主。

**⚠️ 局限性**

局限性包括：① 仅实现了破线与模糊两种攻击，缺乏更复杂的中间人、重放等攻击；② 缺少图形化界面，用户交互不直观；③ 未覆盖硬件或专有仿真器，难以对实际生产环境做精细验证；④ 目前不支持大规模并发攻击与预测模型，需要进一步扩展。

---

## 600. Conversational Decision Support for Information Search Under Uncertainty: Effects of Gist and Verbatim Feedback

**arXiv ID:** 2602.14467 | [PDF](https://arxiv.org/pdf/2602.14467v1)

**作者:** Kexin Quan `[一作]` (University of Illinois Urbana-Champaign), Jessie Chin `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了基于大型语言模型的决策辅助系统SERA，SERA在信息搜索与停止决策中提供“gist（概括）”与“verbatim（逐字）”两种反馈方式，探讨其在不同环境不确定性下对决策结果与搜索行为的影响。

**💡 创新点**

提出了将模糊记忆理论（Fuzzy‑Trace Theory）应用于AI决策支持的反馈表示层面，展示了反馈粒度（gist vs verbatim）与环境不确定性相匹配时能有效调节探索-利用平衡与决策质量，并提供了自适应反馈设计的理论与实证依据。

**🔧 技术方法**

利用OpenAI GPT‑3（或GPT‑4）通过自定义提示生成即时的文本摘要；实验采用混合设计（3×3）对比三种反馈（无SERA、SERA‑Gist、SERA‑Verbatim）与三种不确定性分布（Decremental、Local‑Opt、Random）。

**📊 数据集**

在两项实验（预实验N=54，正式实验N=54）中，每位受试者在三种不确定性条件下完成三组决策任务，共计9个情境，任务使用人为设计的25条描述性信息，构成了自制信息集合。

**📈 对比分析**

结果显示，SERA相较于无反馈显著提高决策准确率（尤其在Random高不确定性下）与决策信心；Gist反馈在高不确定性下更易实现适度停止，减少过度采样；Verbatim反馈则促进系统性比较与更长搜索。性能上，Gist+随机环境组合取得最高准确率与最优搜索效率；Verbatim在结构化环境下表现最为稳健。

**⚠️ 局限性**

局限包括：任务场景文本化、情境复杂度低，缺乏真实高风险决策背景；样本规模小、受试者为高学历美国受试者，缺乏跨文化验证；实验为一次性单场景，未能考察长期适应与信任演变；LLM生成摘要的延迟与准确性仍受限；未实现真正的实时自适应反馈机制，反馈粒度仍为预设。

---

## 601. On Theoretically-Driven LLM Agents for Multi-Dimensional Discourse Analysis

**arXiv ID:** 2602.13713 | [PDF](https://arxiv.org/pdf/2602.13713v1)

**作者:** Maciej Uberna `[一作]` (Warsaw University of Technology), Marcin Koszowy `[通讯]` (Warsaw University of Technology)

**通讯引用:** 215 | [OpenAlex ID](https://openalex.org/A5056008354)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对政治辩论文本中的重述进行功能分类，构建了一个多智能体框架，用理论知识增强的LLM来识别并判定重述的四种主要功能（去强化、强化、细化、泛化）及其他类别。

**💡 创新点**

创新点在于将理论知识（论证理论与重述分类学）通过检索增强生成（RAG）显式注入多智能体系统，形成理论驱动的功能意识型重述分析；并对比单一零样本Agent与多智能体+RAG组合，揭示二者性能差异与协同效应。

**🔧 技术方法**

核心技术包括：大型语言模型（GPT‑5 Mini）、检索增强生成（RAG）知识库、多智能体协作框架（CrewAI/AutoGen），以及四个专用Agent（Assert、Argue、Disagree、Broker Critic）共同完成重述分类任务。

**📊 数据集**

使用了从2016年美国总统辩论中抽取的464条手工标注重述对，经过二人注释后挑选401条用于实验，注释涵盖Deintensify、Intensify、Specification、Generalize、Other及No_rephrase六类。

**📈 对比分析**

通过2×2实验设计（单体vs多体、零样本vsRAG）进行比较，最终RAG‑增强的多智能体系统在Macro F1上达到0.67，MCC 0.64，明显优于单体零样本的0.27 F1和0.16 MCC；与单体RAG相比较，加入多体协作进一步提升F1至0.67，显示两者协同提升。

**⚠️ 局限性**

局限性包括：数据集规模有限、类别分布不均导致训练偏差；RAG知识库依赖人工选取文本，难以覆盖所有语言变体；模型仍易误判语义微妙的去强化/强化区分，且未能针对操纵性重述（如稻草人谬误）进行验证。

---

## 602. EchoTorrent: Towards Swift, Sustained, and Streaming Multi-Modal Video Generation

**arXiv ID:** 2602.13669 | [PDF](https://arxiv.org/pdf/2602.13669v1)

**作者:** Rang Meng `[一作]` (Ant Group), Chenguang Ma `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 EchoTorrent 框架，实现低 NFE、实时、长时序的音频驱动人类动画生成。

**💡 创新点**

创新点包括多教师训练、ACC‑DMD 自适应音频 CFG 校准、混合长尾强制以及 VAE 解码器像素域细节修复。

**🔧 技术方法**

采用分布匹配蒸馏、音频 CFG 动态调度、因果‑双向混合注意力以及像素域后处理技术。

**📊 数据集**

使用 EchoMimicV3、HDTF 以及自采集的大规模音视频数据集进行训练。

**📈 对比分析**

与 EchoMimicV3‑Flash、InfiniteTalk、LiveAvatar、SoulX‑FlashTalk 等 SOTA 进行对比，EchoTorrent 在 FID、FVD、Sync‑C、FPS 等指标上表现相当或更优，且能保持 10.5 FPS 的实时推理速率。

**⚠️ 局限性**

局限性：对极长序列仍存在轻微身份漂移，训练依赖大规模 GPU 资源，且对极低帧率硬件仍有挑战。

---

## 603. LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News

**arXiv ID:** 2602.13543 | [PDF](https://arxiv.org/pdf/2602.13543v1)

**作者:** Yunfan Zhang `[一作]` (Columbia University), Smaranda Muresan `[通讯]` (Barnard College)

**通讯引用:** 2953 | [OpenAlex ID](https://openalex.org/A5043262011)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LiveNewsBench，一个定期更新的基准，利用自动化流程从最新新闻生成问答，专门评估LLM的agentic web search能力。

**💡 创新点**

创新点在于自动化且频繁更新的问答生成、最小化训练集污染、多跳搜索与页面访问需求以及人工验证子集的严格筛选。

**🔧 技术方法**

使用的技术包括GPT‑5.1/4.1驱动的问答生成与自检、一致性过滤、ReAct式代理框架、Tavily搜索API以及人工验证。

**📊 数据集**

所用数据集为近期新闻事件（来自Wikipedia Current Events）及其对应的新闻文章，随后自动生成的训练/验证/测试Q&A集。

**📈 对比分析**

在人工验证集上评估了13个LLM和2个官方搜索API，准确率范围为11%–90%；搜索预算越高性能提升明显，离线无网访问准确率大幅下降，验证检索的必要性。

**⚠️ 局限性**

局限性包括：部分答案仍可凭内部知识推断，人工验证样本量有限，自动生成的问答可能存在错误，模型对工具调用的稳定性差异较大。

---

## 604. TikArt: Aperture-Guided Observation for Fine-Grained Visual Reasoning via Reinforcement Learning

**arXiv ID:** 2602.14482 | [PDF](https://arxiv.org/pdf/2602.14482v1)

**作者:** Hao Ding `[一作]` (Zhejiang University), Lei Zhao `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出TikArt框架，通过思考-光圈-观察循环实现多步骤视觉语言推理；

**💡 创新点**

将多模态推理建模为光圈决策的MDP，强制每次光圈后产生显式观察文本，实现可解释的A-CoT；

**🔧 技术方法**

使用Qwen3‑VL‑8B基础模型，结合Zoom和SAM2分割光圈动作，采用AGRPO双阶段强化学习优化策略；

**📊 数据集**

在V*、HR‑Bench 4K/8K、MME‑RealWorld‑Lite、MMStar、RefCOCO、ReasonSeg等高分辨率细粒度与分割基准上进行评估；

**📈 对比分析**

与基线、同尺寸大型模型以及GPT‑4o、Gemini‑2.5等专有模型对比，TikArt在高分辨率推理上显著提升，单机8B模型已接近或超越更大模型；

**⚠️ 局限性**

主要局限包括对外部分割模型依赖、强化学习训练成本高、推理时额外光圈/分割开销、动作空间有限且可能忽视全局上下文等。

---

## 605. Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws

**arXiv ID:** 2602.14208 | [PDF](https://arxiv.org/pdf/2602.14208v1)

**作者:** Jinbo Wang `[一作]` (Peking University), Lei Wu `[通讯]` (Peking University)

**通讯引用:** 105840 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析并优化批量大小调度，提出并验证了延迟切换策略。

**💡 创新点**

基于功能尺度法(FSL)的理论分析，首次给出最优批量调度结构和“快速追赶效应”。

**🔧 技术方法**

功能尺度法、连续时间SDE、线性回归理论与大规模LLM预训练实验。

**📊 数据集**

C4数据集、0.4T/1T tokens的大规模预训练。

**📈 对比分析**

与常数大批量和早期切换等方案对比，延迟切换在多种模型和规模下均取得更低验证损失和更高样本效率。

**⚠️ 局限性**

仅针对SGD与恒定学习率，未涵盖自适应优化器和学习率衰减的联合影响。

---

## 606. Fast Physics-Driven Untrained Network for Highly Nonlinear Inverse Scattering Problems

**arXiv ID:** 2602.13805 | [PDF](https://arxiv.org/pdf/2602.13805v1)

**作者:** Yutong Du `[一作]` (Northwestern Polytechnical University), Peixian Han `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理驱动的傅里叶谱神经网络求解器，实现逆散射问题的子秒级重建。

**💡 创新点**

创新点在于将低频傅里叶基底压缩与收缩积分方程相结合，引入对比度补偿算子和桥抑制损失，使得在高非线性、噪声和位置误差下依然保持高质量重建。

**🔧 技术方法**

使用傅里叶基底压缩、收缩积分方程（CIE）、对比度补偿算子（CCO）、桥抑制损失、全连接网络与Adam优化器。

**📊 数据集**

数值合成数据（Austria 目标、不同对比度与噪声水平）以及实验数据（FoamDielExt、FoamDielInt）。

**📈 对比分析**

与SOM、FBE-CIE、uSOM、PDNN等方法对比，PDF在高对比度、噪声以及天线位置误差场景中保持结构完整，速度提升约100倍（<1 s）。

**⚠️ 局限性**

局限性包括低频基底导致细节损失、仅在二维场景验证、需要经验调参且未测试更复杂三维或极低频噪声环境。

---

## 607. Residual Connections and the Causal Shift: Uncovering a Structural Misalignment in Transformers

**arXiv ID:** 2602.14760 | [PDF](https://arxiv.org/pdf/2602.14760v1)

**作者:** Jonathan Lys `[一作]` (IMT Atlantique), Ghouthi Boukli Hacene `[通讯]` (Sony Europe Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了因果掩蔽导致的Transformer残差路径与目标 token 不对齐现象，定位并量化了输入-输出对齐转移的层级，并提出残差衰减与可学习门控机制来缓解此问题。

**💡 创新点**

创新点在于首次系统量化了输入-输出对齐的转移层级，并设计了轻量级的可学习残差衰减门控，利用残差流的动态控制提升自回归 LLM 的表示一致性和性能。

**🔧 技术方法**

使用了 logit lens 可视化、余弦相似度与投影指标量化对齐度，构建了层级截断与可学习门控实验，并在 GPT‑2 结构上进行训练与评估。

**📊 数据集**

实验数据集包括 Fineweb‑Edu、Wikitext、LAMBADA 与 OpenWebText 等公开文本语料。

**📈 对比分析**

通过与原始模型在上述基准上的对比，门控方案在所有任务上均保持或提升约 0.2–0.3 点的损失/准确率，表明该方法在保持低成本的同时显著改善了模型性能。

**⚠️ 局限性**

局限性包括：切割层选择对模型深度依赖较强；门控训练增加额外参数和训练成本；未在 70B+ 大模型上验证其有效性；对跨模态或多任务场景的适用性尚未探究。

---

## 608. MamaDino: A Hybrid Vision Model for Breast Cancer 3-Year Risk Prediction

**arXiv ID:** 2602.13930 | [PDF](https://arxiv.org/pdf/2602.13930v1)

**作者:** Ruggiero Santeramo `[一作]` (Fondazione Human Technopole), Florian Jug `[通讯]` (Fondazione Human Technopole)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于混合CNN‑Transformer结构并显式对比双侧乳腺的 3 年风险预测模型，使用 512×512 分辨率的乳腺影像进行风险评分。

**💡 创新点**

创新点在于：①将冻结的 DINOv3 自监督 Transformer 与可训练的 SE‑ResNeXt 结合，利用互补的局部纹理与全局语义特征；②引入 BilateralMixer 对左、右乳腺嵌入进行对称融合、注意力加权和相互作用建模；③通过通道级增强将单通道影像伪装成 RGB，提升特征提取鲁棒性；④在低像素输入下实现与高分辨率 Mirai 相当或更优的预测性能。

**🔧 技术方法**

技术主要包括：冻结 DINOv3 ViT‑S/16+ 作为全局语义提取器；训练 SE‑ResNeXt101 作为纹理提取器；跨注意力融合 + 1×1 卷积桥接；BridgeMixer 对 Transformer 与 CNN 特征对齐；双侧 BilateralMixer（Transformer + 权重门 + 对称特征组合）实现患者级风险预测；per‑channel 随机亮度/对比度/CLAHE 增强。

**📊 数据集**

使用英国 OPTIMAM 乳腺影像数据库，训练集 53,883 名女性（4 个筛查站），内部 1:2 匹配病例对照（525 案例/1,050 对照）和外部 OOD 站点（Oxford）376 案例/1,504 对照。

**📈 对比分析**

方法通过两阶段训练（先乳腺级再患者级）进行；与 Mirai（原始 1664×2048 分辨率）以及 DINO‑only、SE‑ResNeXt‑only 等单流基线对比。内部测试 AUC：Mirai 0.713，Hybrid 0.727，Hybrid+Bilateralmixer 0.736；外部测试 AUC：Mirai 0.676，Hybrid 0.666，Hybrid+Bilateralmixer 0.677。相较 Mirai，Hybrid 模型使用约 13 倍更少的像素且在多数子组（年龄、种族、扫描仪、肿瘤类型/级别）表现更稳健或更优。

**⚠️ 局限性**

局限包括：①仅针对英国筛查服务，缺乏多国/多种筛查间隔的验证；②采用病例对照设计，无法评估校准和真实人群发病率；③GE、Siemens 及非白种族样本不足，导致相关子组评估不充分；④仅评估单一 3 年预测窗口，未探讨多时点或纵向数据；⑤未对 Mirai 在低分辨率下进行重新训练，难以完全分离分辨率与模型结构对性能的贡献。

---

## 609. BPP: Long-Context Robot Imitation Learning by Focusing on Key History Frames

**arXiv ID:** 2602.15010 | [PDF](https://arxiv.org/pdf/2602.15010v1)

**作者:** Max Sobol Mark `[一作]` (Carnegie Mellon University), Aviral Kumar `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种使用视觉-语言模型（VLM）检测关键帧，从而将完整的观测历史压缩为少量语义相关的关键帧，进而进行记忆条件模仿学习；

**💡 创新点**

核心创新在于通过VLM自动提取任务相关的关键事件，显著降低训练与部署之间的历史分布差异，解决历史条件学习中因覆盖不足导致的“虚假相关”问题；

**🔧 技术方法**

使用Diffusion Transformer作为策略网络，配合action chunking；VLM（如Gemini 3 Pro）做关键帧检测并加入延迟掩蔽；还对比了基于过去动作预测的辅助损失；

**📊 数据集**

在四个真实世界多臂操作任务（Mug Replacement、Marshmallows、Drawer Search、Stacking Puzzle）以及三个仿真任务（Ingredient-Insertion、Fixed-Password、Variable-Password）上进行实验；

**📈 对比分析**

与只用当前观测、朴素历史条件、过去动作预测（PTP）以及oracle（访问真状态）等基线对比，BPP在所有任务上均优于其他非oracle方法，平均成功率提升近70%，在部分仿真任务甚至超过oracle；

**⚠️ 局限性**

依赖外部VLM的推理速度和关键帧检测准确性，VLM的延迟和误报可能限制实时性能和任务成功率，且当前关键帧定义需人工或任务特定提示，未来可通过自动生成或低延迟模型提升。

---

## 610. Concept Influence: Leveraging Interpretability to Improve Performance and Efficiency in Training Data Attribution

**arXiv ID:** 2602.14869 | [PDF](https://arxiv.org/pdf/2602.14869v1)

**作者:** Matthew Kowal `[一作]` (FAR.AI), Kellin Pelrine `[通讯]` (FAR.AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出基于可解释语义向量的训练数据归因方法，评估其在消除大型语言模型误导行为与提升后训练数据安全性上的效果。

**💡 创新点**

创新点在于将概念向量（线性探针、稀疏自编码特征）作为归因目标，形成概念归因函数，并展示其可近似为一阶投影，显著提高归因速度与语义精度。

**🔧 技术方法**

使用影响函数、EK-FAC 逆 Hessian、投影差异、向量过滤、稀疏自编码聚类等梯度与向量化方法进行归因分析。

**📊 数据集**

在四个人工合成的 emergent misalignment 数据集（Misaligned Opinions、Bad Medical Advice、Insecure Code、GSM8k Mistakes）以及真实后训练数据集 OpenAssistant Conversations (OASST1) 上验证。

**📈 对比分析**

与传统影响函数、梯度归因方法以及向量过滤等做对比，结果显示概念归因与投影差异在大多数场景下性能相当或更好，且在速度上比影响函数快 8–20 倍，能够有效识别并去除导致模型误导的少量训练样本。

**⚠️ 局限性**

局限在于仅针对线性概念向量，无法捕捉更复杂的机制；投影基方法在分布外场景表现下降；并且缺乏对不同模型规模和多模态任务的系统评估。

---

## 611. A New Approach in Cryptanalysis Through Combinatorial Equivalence of Cryptosystems

**arXiv ID:** 2602.14544 | [PDF](https://arxiv.org/pdf/2602.14544v1)

**作者:** Jaagup Sepp `[一作]` (Hope4Sec), Eric Filiol `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出一种通过组合等价变换的加密分析方法（CE），并基于此构造了概念流密码 Cipherbent6，用于评估该方法的有效性。

**💡 创新点**

创新点在于将密码系统改写为组合等价形式，利用更易分析的结构显现密钥相关性，从而在不依赖传统代数或统计攻击的前提下降低破解复杂度。

**🔧 技术方法**

主要技术包括组合等价变换、函数组的重写、密钥恢复的最大似然解码与 Groebner 基等代数求解工具，以及对非线性反馈移位寄存器（NLFSR）与 6 变量布尔函数的具体实现。

**📊 数据集**

数据集采用 Cipherbent6 生成的伪随机序列，实验使用 1,820–2,790 位已知明文/密钥流，硬件平台为 AMD Ryzen Threadripper 2990WX 32 核处理器。

**📈 对比分析**

与已知的相关攻击（复杂度约 2^35、需要 8,000 位输出）相比，CE 方案在 2^45 复杂度下仅需 1,820–2,790 位输出即可完整恢复 181 位密钥，显示出明显的性能优势。

**⚠️ 局限性**

局限性包括仍未找到最优组合等价变换，需 1-2K 位输出，未完全优化计算时间；此外方法尚未在真实密码系统（如 Achterbahn、Kuznyechik）上验证，可能面临规模化与适用性的挑战。

---

## 612. A feedback control optimizer for online and hardware-aware training of Spiking Neural Networks

**arXiv ID:** 2602.13261 | [PDF](https://arxiv.org/pdf/2602.13261v1)

**作者:** Matteo Saponati `[一作]` (University of Zürich and ETH Zürich), Benjamin Grewe `[通讯]` (ETH)

**通讯引用:** 5391 | [OpenAlex ID](https://openalex.org/A5090542476)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研发了一种基于脉冲反馈控制的优化器，能够在混合信号神经形态设备上实现单层脉冲神经网络的在线、硬件感知学习。

**💡 创新点**

将脉冲PI控制器与本地权重更新规则结合，实现了一阶段学习，无需反向传播，兼具在线学习与硬件适配，并对设备匹配误差具有鲁棒性。

**🔧 技术方法**

采用LIF模型、脉冲PI控制器、可微分反馈权重、基于目标误差的学习规则、在线连续权重更新以及设备误差模拟等技术。

**📊 数据集**

使用自定义二分类脉冲数据集和脉冲化的Yin‑Yang三分类数据集。

**📈 对比分析**

与传统BPTT训练的单层LIF网络及线性读出层对比，分类准确率与BPTT/线性读出相当（约63–66%），在线学习可在数千或数万样本后达到相同性能，设备误差下通过小规模神经元组恢复性能。

**⚠️ 局限性**

目前仅验证单层网络，缺乏多层扩展；对设备非理想性仅做随机模拟，实际硬件实现仍需验证；学习速率相对较慢，参数调优仍依赖经验。

---

## 613. Learning State-Tracking from Code Using Linear RNNs

**arXiv ID:** 2602.14814 | [PDF](https://arxiv.org/pdf/2602.14814v1)

**作者:** Julien Siems `[一作]` (University of Freiburg), Babak Rahmani `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文研究了在下一个词预测框架下的状态跟踪问题，提出将排列组合任务转换为Python REPL执行轨迹，并用此方法检验语言模型的状态跟踪能力。

**💡 创新点**

创新点在于：①用REPL轨迹把传统的序列到序列的排列跟踪任务转化为更接近实际代码执行的下一个词预测任务；②构建了Probabilistic Finite‑State Automaton with State Reveals（PFSA‑SR）模型，正式描述了代码执行中的部分可观测性和概率转移；③揭示线性RNN在概率状态跟踪中面临的非线性归一化瓶颈。

**🔧 技术方法**

使用技术主要包括：线性RNN（DeltaNet[-1,1]）与Transformer的对比实验；REPL轨迹生成；PFSA‑SR框架下的贝叶斯信念更新；以及对joint和marginal表示的数值稳定性分析。

**📊 数据集**

实验数据集基于合成的排列组 S_n（n=3,5,10 等），生成不同长度、不同 reveal 间隔的REPL脚本，作为训练和评估数据。

**📈 对比分析**

比较方法是将DeltaNet[-1,1]、DeltaNet[0,1]、Transformer等模型在相同REPL轨迹上的下一个词预测准确率进行对比；结果显示线性RNN在稀疏状态监督下能稳健泛化，Transformer即使在高密度状态监督也难以掌握排列跟踪；在PFSA‑SR的概率场景下，线性RNN表现出指数级的误差衰减，证明其局限性。

**⚠️ 局限性**

局限性在于实验仅使用合成排列组，未覆盖真实代码的解析、控制流、内存管理等复杂因素；PFSA‑SR假设的完全支持剪枝式观察与真实代码中的软观测不完全匹配；此外，线性RNN在概率状态跟踪中的数值稳定性仍需进一步理论与实践验证。

---

## 614. On the Learning Dynamics of RLVR at the Edge of Competence

**arXiv ID:** 2602.14872 | [PDF](https://arxiv.org/pdf/2602.14872v1)

**作者:** Yu Huang `[一作]` (University of Pennsylvania), Yuxin Chen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了强化学习（RLVR）在多步组合推理任务中的训练动力学，分析了长时序稀疏奖励导致的梯度障碍，并提出了阶段性学习与“接力效应”机制。

**💡 创新点**

创新点在于揭示RLVR在“边缘能力”条件下的有效性，通过困难谱平滑性解释了grokking与接力现象；并引入了有限群上的傅里叶分析工具来刻画长时序梯度。

**🔧 技术方法**

使用了Transformer的简化架构、基于softmax注意力、固定MLP原子操作、策略梯度算法以及群表示的傅里叶分析。

**📊 数据集**

实验使用了在Z_96循环群上的合成数据，构造不同长度（5,15,45）和混合长度（R=3,7）的推理任务。

**📈 对比分析**

与传统监督学习相比，RLVR在短时序任务实现近乎完美奖励和注意力聚焦；在混合长度训练中，R=3时出现平滑的接力提升，R=7则表现出长时间的grokking平稳期。

**⚠️ 局限性**

局限性包括仅考虑长度维度的难度差异、使用简化的循环群模型，未涵盖真实语义差异、长尾原子技能以及规划搜索等复杂推理模式。

---

## 615. DORA: Dataflow Oriented Robotic Architecture

**arXiv ID:** 2602.13252 | [PDF](https://arxiv.org/pdf/2602.13252v1)

**作者:** Xiaodong Zhang `[一作]`, Zijiang Yang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于数据流图的机器人中间件DORA，实现零拷贝共享内存通信和统一内存布局，支持跨机器、多语言与高性能通信。

**💡 创新点**

创新点包括：① 数据流化的声明式通信模型，显式描述节点间依赖；② 统一内存布局消除序列化，零拷贝传输；③ 按需共享内存分配与回收，避免分段重组；④ 基于Rust实现安全高效的实现。

**🔧 技术方法**

技术手段：Rust语言、共享内存IPC、Apache Arrow内存格式、Zenoh网络传输、YAML数据流图描述、基于DDS/ROS2/ROS1的对比实验。

**📊 数据集**

数据集/场景：自定义4 MB/50 Hz生产者-消费者流、640×480 RGB（≈0.9 MB）+关节数据、LiDAR/图像；Isaac Sim仿真与Realman Gen72实机部署。

**📈 对比分析**

比较方法：对比ROS1、ROS2、CyberRT，测量CPU占用、序列化/反序列化开销、局部/远程传输延迟、吞吐率；结果显示DORA序列化几乎为0，局部通信延迟≤0.59 ms，远程≤90 ms，1 MB以上数据时比ROS2提升1.8–31.4×，多目标与融合场景仍保持<5 ms。

**⚠️ 局限性**

局限性：生态尚未成熟，缺少自动计算卸载支持，需要手动划分子数据流；社区包与工具链有限；仿真与工业级集成仍不够完善。

---

## 616. Optimized Certainty Equivalent Risk-Controlling Prediction Sets

**arXiv ID:** 2602.13660 | [PDF](https://arxiv.org/pdf/2602.13660v1)

**作者:** Jiayi Huang `[一作]` (King’s College London), Osvaldo Simeone `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种新的框架OCE-RCPS，用于在高风险应用中提供高概率的风险控制预测集，确保满足用户指定的风险容忍水平。

**💡 创新点**

OCE-RCPS通过优化确定性等价风险度量（如条件价值-at-risk和熵风险）来扩展传统的风险控制预测集，提供更强的可靠性保证。

**🔧 技术方法**

使用了上置信界（UCB）技术来识别满足风险容忍水平的预测集参数，并建立了理论保证。

**📊 数据集**

在医学图像分割任务中使用了来自五个开源息肉分割基准的数据集，合并后总共有1781个分割样本。

**📈 对比分析**

与OCE-CRC方法相比，OCE-RCPS在满足目标满意率方面表现更好，OCE-RCPS的平均满意率为0.83和0.93，而OCE-CRC仅为0.65和0.58，未能满足目标要求。

**⚠️ 局限性**

OCE-RCPS的局限性在于其可能需要在不同的配置下进行调整，以适应不同的可靠性要求，未来的研究方向包括将其扩展到分布转移设置和其他安全关键领域。

---

## 617. InEx-Bug: A Human Annotated Dataset of Intrinsic and Extrinsic Bugs in the NPM Ecosystem

**arXiv ID:** 2602.13400 | [PDF](https://arxiv.org/pdf/2602.13400v1)

**作者:** Tanner Wright `[一作]` (University of British Columbia), Gema Rodríguez-Pérez `[通讯]` (University of British Columbia)

**通讯引用:** 487 | [OpenAlex ID](https://openalex.org/A5077601628)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并手工标注了 InEx-Bug 数据集，收集了 377 条来自 103 个 NPM 仓库的 GitHub issue，按 Intrinsic、Extrinsic、Not-a-Bug、Unknown 分类，并附上时间、行为、代码变更等丰富元数据。

**💡 创新点**

首次在 NPM 生态系统层面区分内部缺陷与外部依赖/环境导致的缺陷，并提供可复现的手工标注数据集，同时揭示两类缺陷在生命周期、重开率、代码变更量等方面的显著差异。

**🔧 技术方法**

使用 GitHub REST API 自动抓取元数据，Python 脚本处理与可视化；多轮人工标注并计算 Cohen κ>0.80 的可靠性；采用描述性统计（中位数、比例、平均值）进行比较分析。

**📊 数据集**

基于 Saeidi 等 30,000 条 GitHub issue 与 PR 的过滤后样本，随机抽样 377 条 issue，形成 InEx-Bug 数据集；同时使用 103 个 NPM 仓库的公开 issue 数据。

**📈 对比分析**

通过描述性统计对 Intrinsic 与 Extrinsic 缺陷在关闭时间、重开率、代码变更规模、维护者参与度等维度进行比较；结果显示 Intrinsic 缺陷关闭更快、代码变更更大、重开率低，而 Extrinsic 缺陷重开率高、重开时间长。

**⚠️ 局限性**

局限性包括：数据截至 2025 年 10 月，后续变更不在数据中；样本量相对较小且仅覆盖 NPM 生态，难以推广至其他生态或工业场景；指标受项目工作流影响，未进行假设检验，仅为描述性分析。

---

## 618. WebWorld: A Large-Scale World Model for Web Agent Training

**arXiv ID:** 2602.14721 | [PDF](https://arxiv.org/pdf/2602.14721v1)

**作者:** Zikai Xiao `[一作]` (Alibaba Group), Zuozhu Liu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了WebWorld，一套基于超过100万真实网页交互轨迹训练的开放网页世界模型，并提出WebWorld-Bench基准以及跨域通用性评估；

**💡 创新点**

创新点在于：① 采用层次化可扩展的数据收集管线，首次在开放网页上训练大规模（1M+）真实轨迹；② 支持多格式（A11y、HTML、XML、Markdown）与长时序（30步以上）模拟；③ 通过少量CoT样本注入推理能力；④ 提出了双指标（事实性得分、Web图灵得分）评测框架；

**🔧 技术方法**

技术包括：自回归LLM世界模型、A11y Tree 状态表示、随机爬虫+自主探索+任务导向三层收集、规则+LLM双阶段过滤、数据增强与多格式转换、两阶段训练（动态再加CoT）、WebWorld-Bench 9维度评测、推理时前瞻搜索、跨环境适配；

**📊 数据集**

使用数据集：FineWeb、CCI 3.0 以及自建的高流量站点列表；原始1.06M真实轨迹；1K CoT推理样本；WebWorld‑Bench测试集；外部评测数据如MiniWob++、WebArena、API、代码、游戏、GUI等；

**📈 对比分析**

对比方法：在WebWorld‑Bench九维度上以事实性得分与Web图灵得分评测；WebWorld‑32B平均得分≈71%，与Claude‑Opus‑4.1、Gemini‑3‑Pro持平；在外部基准中，基于WebWorld生成的轨迹微调Qwen3‑8B/14B分别提升MiniWob++ 9.9%、WebArena 10.9%，接近GPT‑4o水平；推理时前瞻搜索实验显示WebWorld优于GPT‑5；跨域实验表明在API、代码、游戏、GUI等环境中均显著提升性能；

**⚠️ 局限性**

局限性：模型存在自夸倾向，生成结果过于乐观；对高质量长文本生成表现不佳；训练数据中可能残留PII、毒性或偏见；模型主要基于模拟，真实世界动态仍有缺失；使用CoT样本虽少但需谨慎选择，过量可能导致性能下降。

---

## 619. In Transformer We Trust? A Perspective on Transformer Architecture Failure Modes

**arXiv ID:** 2602.14318 | [PDF](https://arxiv.org/pdf/2602.14318v1)

**作者:** Trishit Mondal `[一作]`, Ameya D. Jagtap `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 5091 | [OpenAlex ID](https://openalex.org/A5061905151)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

综述了Transformer在安全关键领域的可解释性、鲁棒性、公平性、隐私性及科学一致性等可信度维度，并系统评估其在自然语言、计算机视觉、科学机器学习、机器人与医学等领域的应用与挑战。

**💡 创新点**

提出了统一的多维度可信度框架，将解释性、鲁棒性、偏见/公平、隐私与物理约束等维度整合，并阐释了神经符号融合、物理信息Transformer、贝叶斯/集成不确定性量化等前沿技术。

**🔧 技术方法**

讨论并评述了注意力可视化、信息论与几何解释、神经符号融合、物理约束、贝叶斯Transformer、集成与内部集成、formal verification、频谱鲁棒性、数据增强以及差分隐私等多种技术。

**📊 数据集**

使用的主要数据集包括ImageNet、ImageNet‑C/A/R、GLUE、SQuAD、PDE/气候模拟数据、蛋白质结构与DNA序列数据、基因表达数据、机器人传感数据及财务与气候预测数据等。

**📈 对比分析**

通过与标准Transformer及改进模型在准确率、PGD/AutoAttack鲁棒性、Δ‑bias公平性、不确定性覆盖率、物理一致性误差及推理时间等指标的量化对比，发现改进技术在鲁棒性与公平性方面显著提升，但往往伴随模型复杂度或推理成本上升。

**⚠️ 局限性**

主要局限包括可信度评估指标仍缺乏统一标准，改进技术在大规模真实场景下的可扩展性不足，物理约束与神经网络的结合仍存在理论与实践脱节，隐私与公平的多维度保证往往相互冲突，且缺乏正式的安全合规与监管框架。

---

## 620. Foundation Model-Driven Semantic Change Detection in Remote Sensing Imagery

**arXiv ID:** 2602.13780 | [PDF](https://arxiv.org/pdf/2602.13780v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 621. Compressed Index with Construction in Compressed Space

**arXiv ID:** 2602.13735 | [PDF](https://arxiv.org/pdf/2602.13735v1)

**作者:** Dmitry Kosolobov `[一作]` `[通讯]` (St. Petersburg State University), Dmitry Kosolobov (St. Petersburg State University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文提出了一种基于字符串复杂度δ的压缩索引，能够在O(δ log n / δ)机器词的空间内，在O(m + (occ+1)·log^ε n)时间内完成任意长度为m的模式匹配，并且该索引可以在一次从左到右的流式扫描中以O(n log n)的期望时间构造。

**💡 创新点**

创新点在于构造了一棵名为“jiggly block tree”的层次化块树，并引入确定性指纹（而非传统的Karp–Rabin哈希），通过一系列自底向上的块层次划分、间隙解析以及动态范围报告结构，实现了既不需要全局哈希、又能在流式构造中达到接近最优空间与查询时间的平衡。

**🔧 技术方法**

核心技术包括：层次化块层次划分（block hierarchy）、jiggly block树、确定性指纹、压缩Trie、z-fast Trie、范围报告结构、加权祖先查询结构（van Emde Boas）、动态有序链表以及RMQ等。

**📊 数据集**

论文没有使用具体的实验数据集，而是以理论分析和证明为主。

**📈 对比分析**

与之前的压缩索引（如LZ索引、基于文法的索引等）相比，该方法在空间上达到O(δ log n / δ)（与下界相匹配），查询时间保持O(m + (occ+1)·log^ε n)，构造时间仅为O(n log n)。

**⚠️ 局限性**

局限性包括：需要δ≥Ω(log log n)；构造过程依赖于哈希表的随机化（可通过替代的确定性字典实现但可能导致更高常数）；实现复杂度高，常数项大；并未在真实数据集上进行实验验证。

---

## 622. Parametric Traversal for Multi-Dimensional Cost-Aware Graph Reasoning

**arXiv ID:** 2602.13369 | [PDF](https://arxiv.org/pdf/2602.13369v1)

**作者:** Nicolas Tacheny `[一作]` `[通讯]` (Ni2 Innovation Lab), Nicolas Tacheny (Ni2 Innovation Lab)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种面向带属性图的可参数化遍历框架，允许在路径中同时使用已有边和可构建的缺失边（gap transitions），并通过多维累积状态与可扩展的可接受性域与谓词实现条件可行性搜索。

**💡 创新点**

创新点包括：①将缺失但可构建的连接视为一等迁移；②将可接受性划分为快速的域过滤和细粒度谓词评估；③支持任意维度的累积状态而不强制标量化；④将探索策略与约束完全外部化，使政策驱动的可行性推理成为可能。

**🔧 技术方法**

采用的技术包括：类型化图模型、gap transition 定义、可接受性域与谓词分离、累积函数 g、探索谓词 σ、前沿策略 ϕ（如 BFS、UCS、A* 等）以及基于候选过滤的高效搜索算法。

**📊 数据集**

实验使用的不是公开大规模数据集，而是基于实际数据中心机架、光纤网络等的代表性情景（如机柜级交叉连接、站点间光纤衰减限制等）进行仿真和案例验证。

**📈 对比分析**

本文未采用传统的跑分或最佳性比较，而是通过案例展示其表达力：能在多维约束下判定条件可行、列举非标量可排序的多方案，并进行政策阈值的前瞻性分析；性能方面主要受累积与探索谓词的剪枝能力影响，未给出数值评估。

**⚠️ 局限性**

局限性包括：无全局最优性保证；依赖专家定义的可接受性域与探索谓词，若设计不当可导致指数级搜索；谓词不具历史上下文，难以表达跨步依赖；仅处理单个遍历，无法直接处理多遍历共享约束；实验规模有限，缺乏大规模实测。

---

## 623. PeroMAS: A Multi-agent System of Perovskite Material Discovery

**arXiv ID:** 2602.13312 | [PDF](https://arxiv.org/pdf/2602.13312v1)

**作者:** Yishu Wang `[一作]` (Southeast University), Guixiang Li `[通讯]` (Southeast University)

**通讯引用:** 5446 | [OpenAlex ID](https://openalex.org/A5033895107)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了多智能体系统 PeroMAS，完成从文献检索、知识抽取、材料设计、性能预测到实验验证的全流程钙钛矿材料发现。

**💡 创新点**

创新点包括：①引入 Model Context Protocol（MCP）统一整合多工具；②设计层级 Meta‑Agent + 功能代理架构实现闭环多目标优化；③发布专属评测基准 PeroMAS‑Bench，为多智能体评估提供标准。

**🔧 技术方法**

技术手段：大语言模型（GPT‑4o、Claude‑4.5、DeepSeek‑V3 等）与专用工具链；图神经网络与机器学习回归做性能预测；SHAP 等可解释性分析；多智能体规划与执行框架。

**📊 数据集**

使用数据集：5000 条真实“成分‑工艺‑性能”实验数据、Perovskite Database、PeroMAS‑Bench 150 条评测实例。

**📈 对比分析**

对比方法：与标准 LLM、单 ReAct Agent、通用自主 Agent 做基线；采用 Tool Accuracy、Validity、Task Completion、Output Validity 等指标。Claude‑4.5 在 Task Completion 72.4% 领先；DeepSeek‑V3 工具准确率 89.5%；wet‑lab 实验验证 PCE 17% 与预测一致。

**⚠️ 局限性**

局限性：受限于现有工具精度与模型推理误差；对极端高效/长期稳定等未覆盖；实验验证仅涉及单一高效低毒性案例；需要更多多样化实验数据与硬件集成支持。

---

## 624. DeepFusion: Accelerating MoE Training via Federated Knowledge Distillation from Heterogeneous Edge Devices

**arXiv ID:** 2602.14301 | [PDF](https://arxiv.org/pdf/2602.14301v1)

**作者:** Songyuan Li `[一作]` (Queen Mary University of London), Jiwei Huang `[通讯]` (China University of Petroleum)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出 DeepFusion，一种基于知识蒸馏的联邦 Mixture‑of‑Experts 训练框架，允许异构、资源受限的边缘设备通过上传本地轻量化 LLM 的知识来共同训练一个全局 MoE LLM。

**💡 创新点**

创新点包括：①实现边缘设备自主配置本地 LLM 以适应硬件限制；②采用一次性通信（one‑shot）联邦学习并通过知识聚类和代理模型实现可扩展性；③设计 View‑Aligned Attention (VAA) 模块，解决跨架构模型在视角不匹配导致的蒸馏难题。

**🔧 技术方法**

使用技术包括：联邦学习、一轮通信；知识蒸馏（日志和特征对齐）；聚类+代理模型；多阶段注意力对齐 VAA；MoE 架构的构建与微调。

**📊 数据集**

实验数据集：医学多选 QA（MMedBench）和金融开放式 QA（FinQA），分别用于评估 Qwen‑MoE（14.3B）和 DeepSeek‑MoE‑16B‑base。

**📈 对比分析**

与 FedJETS、FedKMT、OFA‑KD 以及 DeepSpeed（中央化基准）比较，DeepFusion 在 token perplexity、token accuracy 上接近或超过 DeepSpeed，通信成本下降 71%，token perplexity 提升 5.28%。

**⚠️ 局限性**

局限性：仍需公共基准数据支持蒸馏；对极低算力设备的适配程度有限；模型聚合与蒸馏过程仍需要一定算力与存储，可能限制极端边缘场景。

---

## 625. Audiocards: Structured Metadata Improves Audio Language Models For Sound Design

**arXiv ID:** 2602.13835 | [PDF](https://arxiv.org/pdf/2602.13835v1)

**作者:** Sripathi Sridhar `[一作]`, Justin Salamon `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了针对声音设计的结构化元数据（audiocards），并训练了基于LLM的生成模型和基于CLAP的文本-音频对比模型来生成并利用这些元数据。

**💡 创新点**

创新在于将专业声音设计的关键属性（如名词-动词对、UCS标签、视觉上下文等）融入多字段结构化描述，并通过LLM和对比学习显著提升字幕和检索性能。

**🔧 技术方法**

使用的技术包括：Prompted LLM（Pixtral、Whisper），Whisper‑Cards/Whisper‑Baseline captioner，CLAP框架的文本-音频对比学习，DistilBERT分类器用于UCS预测，以及音频特征提取（持续时间、响度、亮度、音调）等。

**📊 数据集**

使用的数据集包括约200万条音频样本（商业与公开），Adobe Audition SFx（10000条专业音效）及其人工验证的500条子集ASFx eval，Clotho数据集，以及内部评估ID（约6000条）。

**📈 对比分析**

与现有音频语言模型（Audio Flamingo 3/2、WavCaps、GAMA）和基线caption、CLAP模型对比；在caption任务上使用SPIDEr/FENSE，检索任务上使用Recall@10与UCS精度；结果显示：Whisper‑Cards在ASFx eval上SPIDEr提升至22以上，Cards‑CLAP在ID上Recall@10提升至75%以上，均优于基线和部分LALM。

**⚠️ 局限性**

局限性包括UCS预测准确率仅31%，仍受模型hallucination影响；数据集规模与多样性有限（ASFx eval仅500条）；跨域泛化仍需改进，且模型在Clotho等公共数据上的表现不如部分LALM。

---

## 626. RNM-TD3: N:M Semi-structured Sparse Reinforcement Learning From Scratch

**arXiv ID:** 2602.14578 | [PDF](https://arxiv.org/pdf/2602.14578v1)

**作者:** Isam Vrce `[一作]` (Deggendorf Institute of Technology), Gökçe Aydos `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种从零开始训练的N:M结构稀疏TD3算法（RNM-TD3），在训练过程中始终保持行级N:M稀疏。

**💡 创新点**

创新点在于将硬件友好的N:M稀疏模式融入离线RL训练，首次证明在训练阶段保持稀疏可匹配或超越稠密网络，并通过SAD分析揭示稀疏结构动态更新对性能的正面影响。

**🔧 技术方法**

使用了结构化稀疏矩阵投影、软重置机制、SAD度量、Kaiming归一化调节、TD3算法以及GPU上的N:M稀疏算子。

**📊 数据集**

在MuJoCo连续控制基准（Ant-v5、HalfCheetah-v5、Humanoid-v5、Walker2d-v5）上进行评估。

**📈 对比分析**

与稠密TD3、静态N:M、动态稀疏（DS-TD3）和RLx2等基线比较，RNM-TD3在2:4稀疏率下平均可提升约14%，在1:4甚至比稠密模型更优，且在高稀疏率（1:8）时仍保持竞争力。

**⚠️ 局限性**

局限性包括对高稀疏率和不稳定环境下初始化敏感，需要进一步研究自适应初始化与动态更新策略，且目前未实现双向N:M稀疏或真正的训练加速实验。

---

## 627. LaViDa-R1: Advancing Reasoning for Unified Multimodal Diffusion Language Models

**arXiv ID:** 2602.14147 | [PDF](https://arxiv.org/pdf/2602.14147v1)

**作者:** Shufan Li `[一作]` (University of California Los Angeles), Jason Kuen `[通讯]` (Adobe)

**通讯引用:** 8151 | [OpenAlex ID](https://openalex.org/A5076366439)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 LaViDa‑R1，一种针对统一多模态扩散语言模型的后训练框架，旨在显著提升模型的推理能力。

**💡 创新点**

创新点包括：①将监督微调（SFT）、多任务强化学习（RL）和自蒸馏联合为统一的策略梯度目标；②设计了两种指导生成策略（答案强制化与树搜索）解决在线 RL 的梯度消失；③提出了互补掩码（Complementary‑Masking）似然估计器，提升训练稳定性。

**🔧 技术方法**

核心技术包括：离散扩散模型、GRPO 强化学习、SFT 与自蒸馏的加权联合训练、答案强制化与树搜索采样、互补掩码似然估计。

**📊 数据集**

使用的主要数据集有：MathVista、MathVerse、ChartQA、AI2D、MMMU‑Pro、GSM8K、Math500、ImgEdit、Lisa‑Grounding 等视觉推理、问答、编辑和定位任务数据集。

**📈 对比分析**

与基线 LaViDa‑O 及单纯 SFT 进行对比，LaViDa‑R1 在所有评测任务上均取得提升（如 MathVerse 准确率提升 4–6%，ImgEdit EditScore 提升 0.10，Lisa‑Grounding mIoU 提升 22.1），显示出统一后训练与强化学习相结合的显著优势。

**⚠️ 局限性**

局限性包括：答案强制化比例过高会导致模型崩溃；树搜索与奖励设计对不同任务仍需手工调参；训练过程对算力和内存要求高；在极其复杂的多模态推理场景中，仍可能出现推理失误。

---

## 628. Customer Service Operations: A Gatekeeper Framework

**arXiv ID:** 2602.13998 | [PDF](https://arxiv.org/pdf/2602.13998v1)

**作者:** Maqbool Dada `[一作]` (Johns Hopkins University), Evgeny Kagan `[通讯]` (Johns Hopkins University)

**通讯引用:** 117 | [OpenAlex ID](https://openalex.org/A5089062634)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套“门卫框架”，对客户服务渠道（如人工客服、AI聊天机器人）中的请求处理过程进行建模，并通过动态规划求解最优转移策略，进一步将该框架扩展到渠道选择、人员配置与机器人训练等层面的整体运营设计问题。

**💡 创新点**

创新点在于（1）首次将门卫系统理论与客户服务多渠道运营结合；（2）通过构造终端条件实现有限时段与24/7运营的统一；（3）揭示最优转移策略可归约为最多四条规则，并给出阈值策略的足够性条件；（4）将聊天机器人有限尝试与成本结构纳入整体决策。

**🔧 技术方法**

主要技术包括：离散时间动态规划（finite‑horizon MDP），终端条件设计实现策略平稳性，阈值策略启发式与足够性证明（基于加权最短处理时间 WSPT），多服务器近似路由，队列模型（DTMC）与稳态分析，以及整体利润最大化的多层优化框架。

**📊 数据集**

实证数据来源于对BlackBeltHelp高等教育客户服务的访谈、现场问卷调查以及模拟随机生成的100万+问题实例，用于评估阈值策略的最优性与实际性能。

**📈 对比分析**

与传统单一门卫策略相比，阈值策略在超过95%的随机实例中达到最优或误差<0.01%，且在加入等待室后“半满转移”策略可获得近似最优利润；在整体渠道设计中，加入聊天机器人可提升利润并改善服务质量，尤其在机器人训练成本低或人工工资高的情境。

**⚠️ 局限性**

局限性包括：只考虑两类渠道（人工、聊天机器人）且假设同质客户；多服务器模型采用随机分配近似，未考虑技能匹配；机器人仅具有限尝试与冷转移，未覆盖更复杂的对话管理；且客户选择模型在实务中需更多真实测量验证。

---

## 629. Testing BDI-based Multi-Agent Systems using Discrete Event Simulation

**arXiv ID:** 2602.13878 | [PDF](https://arxiv.org/pdf/2602.13878v1)

**作者:** Martina Baiardi `[一作]` (University of Bologna), Danilo Pianini `[通讯]` (University of Bologna)

**通讯引用:** 2180 | [OpenAlex ID](https://openalex.org/A5059198182)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文实现了一个开源原型，将基于Jakka的BDI框架与Alchemist离散事件仿真引擎相集成，实现同一BDI规范在仿真与真实部署之间无缝切换，并探讨了将BDI执行循环映射到DES时的四种粒度（AMA、ACLI、ACLP、ABE），通过多旋翼无人机圆形编队案例验证不同粒度对仿真真实感的影响。

**💡 创新点**

创新点在于：①提出了“无改动即可在仿真和真实环境中运行BDI规范”的思路；②定义了BDI控制循环与DES事件的多粒度映射方案；③通过实验证明粒度选择会显著影响系统行为，揭示了粗粒度仿真隐藏错误的风险；④提供了可复现的实验环境与可插拔的环境API。

**🔧 技术方法**

技术栈包括：Jakka（Kotlin实现的BDI框架）、Alchemist（Kotlin/Scala实现的通用离散事件仿真引擎）、Kotlin DSL（用于编写BDI规范与仿真环境）、YAML（配置仿真场景）。

**📊 数据集**

实验使用的“数据集”为自行生成的无人机初始位置与轨迹参数，仿真中生成的事件日志。并未使用公开大规模数据集。

**📈 对比分析**

比较方法：将同一BDI程序在三种粒度（AMA、ACLI、ACLP）下在仿真中运行，并与在多线程并发运行的“真实”部署进行对比；性能指标为追踪误差（跟随理想位置的平方误差）。实验结果表明：粗粒度（AMA/ACLI）在仿真中误差低但在真实部署中失效；细粒度（ACLP）揭示了频率至少2Hz才可满足约束，说明粒度对可接受的执行频率有实质影响。

**⚠️ 局限性**

限制：①原型仅针对Jakka与Alchemist，未验证跨平台通用性；②实验规模较小（17架无人机），未探讨大规模可扩展性；③未对仿真与部署的性能（时间/资源消耗）进行量化评估；④缺乏与现有BDI仿真方法（如基于DTS的框架）的客观对比；⑤未在真实无人机硬件上验证，仿真模型的物理真实性有限。

---

## 630. WildfireVLM: AI-powered Analysis for Early Wildfire Detection and Risk Assessment Using Satellite Imagery

**arXiv ID:** 2602.13305 | [PDF](https://arxiv.org/pdf/2602.13305v1)

**作者:** Aydin Ayanzadeh `[一作]` (University of Maryland), Milton Halem `[通讯]` (University of Maryland)

**通讯引用:** 1453 | [OpenAlex ID](https://openalex.org/A5004628313)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 WildfireVLM 框架，将卫星影像火灾与烟雾检测与多模态语言模型结合，形成从检测到风险评估的端到端决策支持系统。

**💡 创新点**

创新点在于把 YOLOv12 目标检测与大型语言模型（GPT‑4o、Claude 3.5 Sonnet）融合，既实现实时火点与烟雾识别，又生成情境化风险评估与应急建议。

**🔧 技术方法**

使用 YOLOv12、YOLOv11、YOLO‑NAS 等目标检测模型以及 GPT‑4o、Claude 3.5 Sonnet 等多模态 LLM 进行风险推理，并通过 FastAPI+Uvicorn 架构实现实时服务。

**📊 数据集**

构建约 3,771 张标注的 Landsat‑8/9 与 GOES‑16 卫星影像数据集，涵盖火灾与烟雾两类，按 70/15/15 进行训练、验证与测试。

**📈 对比分析**

与 YOLOv8、YOLOv11、YOLO‑NAS 等模型比较，YOLOv12 在精度 81.1%、召回率 74.8% 和 F1 77.8% 方面表现最佳；LLM‑评估中 GPT‑4o 平均得分 7.03 高于 Claude 3.5 Sonnet 6.16，显示其在语义正确性和可操作性方面更优。

**⚠️ 局限性**

局限包括低分辨率、云影与大气干扰导致检测误差；对未检测到火点的假设需要人工补充；缺乏针对不同地区和条件的专门训练与验证，模型泛化性待提升。

---

## 631. Avoiding Social Judgment, Seeking Privacy: Investigating why Mothers Shift from Facebook Groups to Large Language Models

**arXiv ID:** 2602.13941 | [PDF](https://arxiv.org/pdf/2602.13941v1)

**作者:** Shayla Sharmin `[一作]` (Chittagong University of Engineering and Technology), Sadia Afrin `[通讯]` (Bangladesh University of Professionals)

**通讯引用:** 383 | [OpenAlex ID](https://openalex.org/A5085036496)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究母亲从Facebook育儿群转向大型语言模型（LLM）的动因，并通过混合方法（问卷与开放式访谈）收集定量与定性证据。

**💡 创新点**

创新点在于揭示居住地与家庭结构是规避社交判断的关键因素，并系统阐述LLM在情感安全、信息可靠性和即时响应等功能与心理优势。

**🔧 技术方法**

使用Google Forms在线问卷、交叉表与卡方检验进行定量分析，结合定性主题分析、词云和桑基图可视化，整合社会支持、认知失调等多理论框架。

**📊 数据集**

数据集为109名南亚母亲的在线调查问卷数据，包括封闭式与开放式问题。

**📈 对比分析**

通过交叉表和卡方检验比较不同人群对Facebook群规避程度，结果显示居住地与家庭结构显著相关；定性分析识别三大主题，说明LLM被视为更安全、更及时的支持；未对LLM回答性能进行量化对比，仅描述用户偏好。

**⚠️ 局限性**

局限性包括样本主要来自南亚，结果可能不具普遍性；依赖自我报告且未直接观察实际使用行为；主题级分析缺乏时间维度；未验证LLM回答的准确性和可靠性。

---

## 632. Robust Value Maximization in Challenge the Champ Tournaments with Probabilistic Outcomes

**arXiv ID:** 2602.14966 | [PDF](https://arxiv.org/pdf/2602.14966v1)

**作者:** Umang Bhaskar `[一作]` (Tata Institute of Fundamental Research), Sanjay Seetharaman `[通讯]` (Institute of Mathematical Sciences)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在挑战冠军（Challenge the Champ）锦标赛中，当比赛结果存在概率不确定性时，如何设计种子顺序以保证取得的最小价值（VnaR）最大化。

**💡 创新点**

创新点在于：①提出价值不受风险（VnaR）这一风险规避目标；②证明非自适应算法在最坏情况下近似困难；③给出自适应算法依赖于热门玩家子图的色数，从而获得近似解；④在特定限制下实现多项式时间解。

**🔧 技术方法**

主要技术包括：图论方法（强度图、热门玩家子图、色数分解、arborescence 最小值计算）；可归约到 Hamiltonian 路径的硬度证明；基于颜色分解的构造算法；以及自适应决策框架。

**📊 数据集**

本文没有使用实际数据集，全部以理论构造的实例进行分析与证明。

**📈 对比分析**

通过理论归约与近似分析比较，非自适应算法在最坏情况下无法获得常数近似；自适应算法在色数为 k 时可以获得至少 (n_p + n_u - 1) - (k-1) 的价值，最优值为 n_p + n_u - 1，若 k ≈ n_p，则可达约 1/2 的下界。

**⚠️ 局限性**

局限性：①对一般实例仅能给出 1/2 的近似下界；②算法性能依赖于热门玩家子图的色数或其他参数，实际实现受限；③在完全任意的概率结构下仍缺乏多项式时间精确或更优近似方案。

---

## 633. Center-Fed Pinching Antenna System (C-PASS): Modeling, Analysis, and Beamforming Design

**arXiv ID:** 2602.14805 | [PDF](https://arxiv.org/pdf/2602.14805v1)

**作者:** Xu Gan `[一作]` (University of Hong Kong), Yuanwei Liu `[通讯]` (University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种中心馈送式压挤天线系统（C‑PASS）的通用框架，并给出了其自由度（DoF）与功率扩展规律的闭式解析；基于此模型，构建了下行多用户通信的总速率最大化问题，并通过WMMSE重构结合交替优化与块坐标下降实现联合发射与压挤波束设计；进一步给出了算法收敛与复杂度分析；通过仿真验证了C‑PASS在 DoF、功率扩展、收敛性与总速率方面的优势。

**💡 创新点**

创新点在于：①引入分布式中心馈送结构，突破传统PASS单自由度瓶颈，实现 DoF 线性随输入端口数 M 变化；②Derive 线性功率增益 O(P_T M) 的理论上限；③提出联合发射/压挤波束的闭式求解与交替优化方案；④通过微调 PA 位置与功率拆分实现系统性能最大化。

**🔧 技术方法**

采用的技术包括：闭式 DoF 与功率分析（矩阵秩与特征值分析）、WMMSE 重构求解总速率最优化、交替优化框架、块坐标下降（BCD）处理非凸子问题、三角参数化与 Ferrari 公式求根、Brent 算法一维搜索、拉格朗日乘子与二分法求 λ、矩阵运算与复杂度评估。

**📊 数据集**

使用的是仿真数据：随机布置 K 个用户于 100 m×100 m 区域，波导长度 100 m，M 个输入端口均匀分布，频率 77 GHz，n_eff = 1.44，α = 0.0092，Δ = 0.01 m 的微调范围；未使用真实测量数据或公共数据集。

**📈 对比分析**

比较方法：将 C‑PASS 与三种基线方案（单波导端馈 PASS、单波导中心馈 PASS、M 波导 PASS）在相同仿真参数下求解总速率；并对不同输入端口数、功率、波导衰减等场景进行比较。结果显示：C‑PASS 的 DoF 线性提升至 min{M,K}，功率扩展呈 O(P_T M)；在高衰减（α = 0.2095）场景下，C‑PASS 相比多波导 PASS 获得超过 10 dB 的总速率增益，且收敛仅需 10 次迭代。

**⚠️ 局限性**

局限性包括：①仅在仿真环境验证，缺乏实测验证；②假设理想 PA 与波导特性，未考虑硬件非理想与多径干扰；③微调范围 Δ = 0.01 m 受限，实际部署可能难以实现；④算法复杂度仍较高，尤其是 BCD 与 Brent 搜索；⑤只研究单波导中心馈，未深入多波导扩展与更大规模网络的性能。

---

## 634. A unified framework for evaluating the robustness of machine-learning interpretability for prospect risking

**arXiv ID:** 2602.14430 | [PDF](https://arxiv.org/pdf/2602.14430v1)

**作者:** Prithwijit Chowdhury `[一作]` (Georgia Institute of Technology), Ghassan AlRegib `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5557 | [OpenAlex ID](https://openalex.org/A5006145139)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种统一框架，利用前向对抗样本生成和必要性、充分性度量，评估在高维结构化油气勘探风险数据上LIME和SHAP的解释可靠性。

**💡 创新点**

创新点在于将对抗样本与必要性/充分性概念融合，得到全局特征重要性评分；同时提供模型无关的鲁棒性评估方法，系统比较两种主流局部解释器在同一数据与多种模型上的表现。

**🔧 技术方法**

核心技术包括：前向对抗样本生成（perturbation）、必要性/充分性概率度量、LIME、SHAP解释器；使用逻辑回归、朴素贝叶斯、随机森林和投票分类器进行实验。

**📊 数据集**

实验数据来自油气勘探领域的直接油气指示器（DHI）数据集，包含35个属性，其中6个高阶特征被特别关注；数据为专有，未公开共享。

**📈 对比分析**

通过对每个样本的前向对抗样本计算必要性和充分性得分，并对LIME与SHAP给出的重要特征进行排名比较，评估两者在不同模型上的鲁棒性。实验显示，重要特征并不总是同时满足高必要性和高充分性，且不同解释器对同一特征的评价存在显著差异；准确率方面，四种模型在训练集和测试集上均达到了较高精度，但对解释器的鲁棒性并无统一优劣结论。

**⚠️ 局限性**

局限性包括：高维数据导致对抗样本生成困难、计算成本高；必要性/充分性度量假设特征独立，忽略潜在因果关系；方法对专有数据的依赖，可能不易推广到其他领域；并未构建完整的因果模型，可能无法捕捉所有关键因果路径。

---

## 635. Variance-Reduced $(\varepsilon,δ)-$Unlearning using Forget Set Gradients

**arXiv ID:** 2602.14938 | [PDF](https://arxiv.org/pdf/2602.14938v1)

**作者:** Martin Van Waerebeke `[一作]` (INRIA Paris), Giovanni Neglia `[通讯]` (INRIA Sophia Antipolis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Variance-Reduced Unlearning（VRU）算法，在机器学习的(ε,δ)-unlearning框架下利用忘记集梯度实现高效模型遗忘。

**💡 创新点**

创新点在于首次将忘记集梯度直接用于方差降低的优化过程中，实现了更快收敛（O(1/e)）并在低误差、低忘记比例下优于任何不使用忘记集的一阶方法。

**🔧 技术方法**

采用了投影随机梯度下降（Projected SGD）结合SVRG式的方差降低梯度估计，再通过高斯噪声注入满足(ε,δ)-unlearning的差分隐私要求。

**📊 数据集**

主要实验使用了强凸的逻辑回归任务，数据集为Digits（MNIST风格）以验证理论与实践的一致性。

**📈 对比分析**

与NFT、SCRUB、NegGrad+、Fine-Tune以及GD/SGD/SVRG等方法比较，VRU在低忘记比例下显著降低过度风险、保持低成员推理攻击准确率，并在给定计算预算下比全量重训练更快、更优。

**⚠️ 局限性**

局限性包括对强凸性和对预训练最优θ*的准确获取的依赖；在非凸深度网络、需要估计L等常数时可能表现欠佳。

---

## 636. Orcheo: A Modular Full-Stack Platform for Conversational Search

**arXiv ID:** 2602.14710 | [PDF](https://arxiv.org/pdf/2602.14710v1)

**作者:** Shaojie Jiang `[一作]` (AI Colleagues and University of Amsterdam), Maarten de Rijke `[通讯]` (University of Amsterdam)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Orcheo 平台，统一对话式检索系统的模块化开发、评估与部署。

**💡 创新点**

通过基于 LangGraph 的图工作流、可插拔节点、AI 辅助代码生成及标准化评估，突破现有工具碎片化与可复现性缺口。

**🔧 技术方法**

使用 LangGraph/LangChain、FastAPI、Celery、Docker、OpenTelemetry、ChatKit 等技术栈。

**📊 数据集**

在公开数据集 QReCC（查询重写）和 MultiDoc2Dial（多文档对话生成）上进行评估。

**📈 对比分析**

与 GPT‑4o‑mini 等基准比较，QReCC ROUGE‑1 recall 75.25%、语义相似度 79.00%；MultiDoc2Dial Token F1 8.34、SacreBLEU 13.32、ROUGE‑L 6.82，验证平台高效实验与评估能力。

**⚠️ 局限性**

尚未对大规模文档集合进行性能基准；模块化灵活性虽高，但对研究者的集成与 Python 开发能力有一定要求。

---

## 637. Variation is the Key: A Variation-Based Framework for LLM-Generated Text Detection

**arXiv ID:** 2602.13226 | [PDF](https://arxiv.org/pdf/2602.13226v1)

**作者:** Xuecong Li `[一作]` (Tianjin University), Junjie Wang `[通讯]` (Tianjin University)

**通讯引用:** 28629 | [OpenAlex ID](https://openalex.org/A5115695478)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于重写后文本差异的黑盒LLM文本检测方法 VaryBalance。

**💡 创新点**

创新点在于发现人类原文与 LLM 重写版差异大于机器生成文本与其重写版差异，并利用 Mean Standard Deviation（MSD）结合 log‑perplexity 形成判别分数，进一步为社交媒体文本设计扩展版。

**🔧 技术方法**

核心技术包括：1) 用 LLM（Rewriter）对原文生成多次重写；2) 用小型 LLM 计算原文和重写版的 log‑perplexity；3) 计算 MSD 并结合 sign 调整后的最终分数；4) 通过阈值判断是否为 LLM 生成文本。

**📊 数据集**

实验数据集覆盖多领域、多语言、多长度：Creative Writing、News、Student Essay、CNN、Reddit、Yelp、M4（中、阿、乌、保四语）、EssayForum、HC3 重写集（DeepSeek、Qwen）。

**📈 对比分析**

对比 White‑box (Fast‑DetectGPT, Binoculars, AdaDetectGPT) 与 Black‑box (RADAR, DNA‑GPT) 基线，使用 AUROC 衡量。VaryBalance 在正式文本上平均提升约 34%（AUROC 最高 0.9936），在社交媒体文本、不同生成模型、不同语言均保持或超过基线，扩展版对短文本提升约 0.03 AUROC。

**⚠️ 局限性**

局限性：需要额外的重写步骤和额外的计算资源；分数对重写质量和重写模型的鲁棒性依赖；在极大规模或极为新颖的 LLM 输出上未做充分评估；对非常简短或结构化文本的表现尚未系统验证。

---

## 638. CCiV: A Benchmark for Structure, Rhythm and Quality in LLM-Generated Chinese \textit{Ci} Poetry

**arXiv ID:** 2602.14081 | [PDF](https://arxiv.org/pdf/2602.14081v1)

**作者:** Shangqing Zhao `[一作]` (East China Normal University), Man Lan `[通讯]` (East China Normal University)

**通讯引用:** 3493 | [OpenAlex ID](https://openalex.org/A5057508424)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了CCiV基准，用于系统评估大型语言模型在古典中国词作（Ci Poetry）中的结构、韵律与艺术质量；同时对17种主流LLM在30种典型词牌上的表现进行了大规模实验；

**💡 创新点**

创新点包括：①引入历史变体（variant-aware）评估，揭示模型倾向生成历史变体而非标准形式；②对比直接提示与“表格化”表格提示两种prompt策略，探讨其对结构与韵律控制的影响；③结合自动化结构/韵律度量与LLM-as-judge的定性质量评估，研究形式正确性与艺术质量的关系；

**🔧 技术方法**

技术主要包括：基于Transformer的LLM（如GPT‑4o、Qwen‑Max、Qwen‑2.5‑72b等）的文本生成、正则表达式文本预处理、基于中华新韵的音调匹配算法、自动化结构准确率与韵律准确率评估；以及使用LLM作为判别者进行多维度（信息量、美感）打分；

**📊 数据集**

数据集：从公开数据库收集49,270首词，筛选出出现频率最高的30种词牌；并从中文文化知识图谱中获取每种词牌的标准与所有历史变体定义；生成300个随机主题提示（每个词牌10个）用于评估；

**📈 对比分析**

比较方法：对17个模型在两种prompt条件下进行相同提示，计算结构准确率（标准版与变体版）、韵律准确率，并对四个代表性模型采用LLM-as-judge评估信息量与美感；性能结果显示：顶尖模型如Qwen‑2.5‑72b在结构约80%、韵律约75%；但在信息量与美感上仅略高于3.0；较小模型表现极差，结构准确率几乎为0；表格提示对大模型提升约10-40%，对小模型可能导致性能下降；

**⚠️ 局限性**

局限性：①模型仍难以精确控制字符级平仄，导致韵律准确率普遍低于结构准确率；②表格提示对弱模型不友好，可能压倒其生成能力；③形式正确性与文学质量关联弱，评估仍主要依赖自动化指标；④实验仅基于公开词库与知识图谱，缺乏更广泛的历史版本覆盖，可能遗漏罕见变体；

---

## 639. From Perceptions To Evidence: Detecting AI-Generated Content In Turkish News Media With A Fine-Tuned Bert Classifier

**arXiv ID:** 2602.13504 | [PDF](https://arxiv.org/pdf/2602.13504v1)

**作者:** Ozancan Ozdemir `[一作]` (University of Groningen), Ozancan Ozdemir `[通讯]` (University of Groningen)

**通讯引用:** 59 | [OpenAlex ID](https://openalex.org/A5084777765)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并利用3,600篇三大土耳其媒体的人工与ChatGPT重写新闻样本，训练BERT模型检测AI生成内容；

**💡 创新点**

首次在土耳其语新闻中应用Transformer进行AI重写检测，并提供公开数据集与基线；

**🔧 技术方法**

使用土耳其特化BERT（BERTurk）并通过微调、AdamW、label smoothing等技术完成二分类；

**📊 数据集**

自2021年三大媒体随机抽取的3,600篇新闻（人写+ChatGPT重写），外部测试为2023‑2026年各媒体3,500篇；

**📈 对比分析**

在验证集上F1达97.08%，测试集F1 97.08%，外部预测平均2.5%新闻被LLM重写，模型置信度>96%；

**⚠️ 局限性**

仅覆盖三家媒体、样本量有限、BERT 512-token截断、外部验证无人工标注、未与传统模型或多语言模型对比

---

## 640. ARport: An Augmented Reality System for Markerless Image-Guided Port Placement in Robotic Surgery

**arXiv ID:** 2602.14153 | [PDF](https://arxiv.org/pdf/2602.14153v1)

**作者:** Zheng Han `[一作]` (Chinese University of Hong Kong), Qi Dou `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 27532 | [OpenAlex ID](https://openalex.org/A5090516040)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一套基于HoloLens 2的ARport系统，实现机器人手术中无标记、无外部传感器的端口放置导引；系统能够自动将术前CT规划的切口点叠加到患者体表，并提供实时的增强现实可视化。

**💡 创新点**

创新点包括：①Markerless且硬件最小化的配准流程；②利用Foundation Model（SAM）完成人体表面分割并实现自我纠正；③在完整的3D重建、分割、配准与可视化管线中实现实时性能；④在全尺寸手术模型上完成端到端验证。

**🔧 技术方法**

核心技术为：光学可透视HMD+RGB‑depth+姿态感知的3D重建；SAM实现2D→3D体素掩码初始化与迭代扩展；FPFH+RANSAC/ICP的多尺度刚性配准；以及HoloLens 2硬件与远程GPU计算的协同实现。

**📊 数据集**

使用了公开的CVH数据集生成的全尺寸3D打印人体模型，并采用常规手术指南中的端口规划模板（肾尿路切除、子宫切除、膀胱切除）进行验证。

**📈 对比分析**

通过对13个解剖标志点进行目标配准误差（TRE）评估，腹部区域平均误差约7 mm（5–10 mm范围），显著优于传统手工估计的厘米级误差；3D重建误差在近距离低倾角时 <5 mm，随距离/倾角增大至约20 mm；系统端到端延迟约890 ms，更新率5 fps。

**⚠️ 局限性**

主要局限包括：①配准误差仍高于传统导航系统（≈5 mm）受HoloLens深度传感器系统误差影响；②未考虑腹部软组织的非刚性变形；③在真实手术室光照、遮挡等条件下鲁棒性待验证；④缺乏人因学和工作流程的评估。

---

## 641. Automated Prediction of Paravalvular Regurgitation before Transcatheter Aortic Valve Implantation

**arXiv ID:** 2602.13842 | [PDF](https://arxiv.org/pdf/2602.13842v1)

**作者:** Michele Cannito `[一作]` (University of Turin), Fabrizio D'Ascenzo `[通讯]` (University of Turin)

**通讯引用:** 20005 | [OpenAlex ID](https://openalex.org/A5033242180)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

利用预手术心脏CT进行深度学习预测TAVI后血管周围反流（PVR）。

**💡 创新点**

首次在预手术CT上使用3D CNN直接预测PVR，并探索预训练与解剖分割对性能的影响。

**🔧 技术方法**

采用3D DenseNet121/ResNet50网络、Focal Loss、Grad-CAM等深度学习技术。

**📊 数据集**

249例TAVI患者的门诊心脏CT（512×512，0.625 mm³），其中174例无PVR，75例有PVR。

**📈 对比分析**

与无预训练的基线模型相比，内部冠脉钙评分预训练的DenseNet121在测试集上达71.8%平衡准确率；COCA预训练为64.1%，基线为66.3%/67.4%。

**⚠️ 局限性**

数据量有限、PVR分级不平衡、对轻度PVR的判别仍受限，且仅基于CT未融合临床多模态信息。

---

## 642. RMPL: Relation-aware Multi-task Progressive Learning with Stage-wise Training for Multimedia Event Extraction

**arXiv ID:** 2602.13748 | [PDF](https://arxiv.org/pdf/2602.13748v1)

**作者:** Yongkang Jin `[一作]` (Soochow University), Yu Hong `[通讯]` (Soochow University)

**通讯引用:** 25755 | [OpenAlex ID](https://openalex.org/A5110827204)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了 RMPL 框架，在缺乏多模态事件标注数据的情况下，通过两阶段的多任务学习完成多模态事件抽取。

**💡 创新点**

创新点包括：① 采用统一 schema 的温身阶段，将文本、图像事件提取与多模态关系抽取的异构监督融合；② 通过阶段式训练从通用表征逐步转向任务专用，显著提升跨模态对齐和论点定位。

**🔧 技术方法**

技术手段：基于视觉‑语言模型（如 Qwen2‑VL、InternVL3 等）进行自回归生成；统一 schema 序列化；关系感知多任务训练；结合自监督与监督损失进行混合学习。

**📊 数据集**

数据集：M2E2 作为评测基准；ACE 2005、SWiG、MNRE（以及 imSitu）作为外部文本、图像与关系监督来源。

**📈 对比分析**

对比方法：与 Prompt‑only 基线、Warm‑up 以及多种 VLM 基准（CLIP、UniCL、Qwen 等）进行对比。RMPL 在文本、图像及多模态三种设置下均提升 20% 以上的 F1，且在多模态场景中超越 KE‑MME、X‑MTL 等现有方法。

**⚠️ 局限性**

局限性：仍需人工映射事件类型；依赖外部监督来源，缺乏大规模多模态标注；在更广泛的多模态任务中的可扩展性与适用性待进一步验证。

---

## 643. FMMD: A multimodal open peer review dataset based on F1000Research

**arXiv ID:** 2602.14285 | [PDF](https://arxiv.org/pdf/2602.14285v1)

**作者:** Zhenzhen Zhuang `[一作]` (Guangzhou Institute of Science and Technology), Jialiang Lin `[通讯]` (Guangzhou Institute of Science and Technology)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多模态、多学科的公开同行评审数据集FMMD，涵盖了8899篇论文及其多版本、评审评论和作者回复，并在HTML格式中保留了文本、图表、布局等视觉信息；

**💡 创新点**

首次系统性地提供了评审评论与论文各版本及其多模态内容之间的显式对齐，解决了先前数据集仅关注文本或缺乏版本映射的问题；

**🔧 技术方法**

采用网页爬取、结构化HTML解析以及基于DeepSeek Reasoner LLM的语义分析，对多模态内容进行识别与聚类；

**📊 数据集**

使用F1000Research平台的公开文章与评审数据，经过三阶段处理生成FMMD数据集；

**📈 对比分析**

论文未对具体算法进行实验或性能比较，主要聚焦于数据集构建与描述；

**⚠️ 局限性**

局限在于数据来源单一（仅F1000Research），多模态评论比例有限，缺乏跨领域覆盖和模型基准评估。

---

## 644. SPILLage: Agentic Oversharing on the Web

**arXiv ID:** 2602.13516 | [PDF](https://arxiv.org/pdf/2602.13516v1)

**作者:** Jaechul Roh `[一作]` (University of Massachusetts Amherst), Ali Shahin Shamsabadi `[通讯]` (Brave Software)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了SPILLage框架，评估大型语言模型驱动的网络代理在执行用户任务时的自然过度分享（oversharing）风险。

**💡 创新点**

创新点在于引入二维分类（内容/行为 × 显式/隐式）来全面衡量过度分享，并通过自动化的LLM审计器在真实网站上对四类泄露进行细粒度检测。

**🔧 技术方法**

采用两种主流Web代理框架（Browser-Use 与 AutoGen）和三种OpenAI GPT后端，利用LLM驱动的决策与审计流程，配合行为与文本输入的日志分析技术。

**📊 数据集**

构建了180个包含任务相关与无关属性的合成用户提示，在Amazon和eBay两大电商网站上执行，形成了首个面向真实Web环境的过度分享基准数据集。

**📈 对比分析**

通过1,080次代理运行（≈10⁵ API调用），比较了不同框架、模型和提示风格下的过度分享次数、率和任务成功率，发现行为过度分享占主导，去除无关信息后任务成功率可提升至99.4%，并实现了显著的隐私与效能协同。

**⚠️ 局限性**

主要局限包括仅测试OpenAI模型、仅聚焦电商场景、单域网站交互以及未探究跨站追踪等多域行为模式对隐私的进一步影响。

---

## 645. Nighttime Autonomous Driving Scene Reconstruction with Physically-Based Gaussian Splatting

**arXiv ID:** 2602.13549 | [PDF](https://arxiv.org/pdf/2602.13549v1)

**作者:** Tae-Kyeong Kim `[一作]` (University of Toronto), Bingbing Liu `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种将物理基础渲染（PBR）集成到三维高斯分裂（3DGS）场景图中的方法，用以在夜间低光条件下重建自驾驾驶场景。

**💡 创新点**

创新点包括：① 用全局 SH 光照模块实现 diffuse 成分的光照建模；② 用各高斯的可分离 ASG（各向异性球面高斯）与简化 Disney BRDF 结合，对 specular 成分进行高频物理可验证的建模；③ 通过 BRDF 约束和全局照明的联合优化，实现实时渲染下的物理一致、细节丰富的夜间场景重建。

**🔧 技术方法**

使用技术包括：三维高斯分裂（3DGS）、物理基础渲染方程、简化 Disney BRDF、SH 光照、各向异性球面高斯（ASG）、Reinhard 颜色映射、深度-法线监督与 D-SSIM 损失。

**📊 数据集**

使用数据集：nuScenes 和 Waymo Open Dataset 的 11 个夜间低光场景进行训练与测试。

**📈 对比分析**

与 3DGS、StreetGaussians、OmniRe 等现有 3DGS 方法对比，PSNR、SSIM 与 LPIPS 指标均优于对手；在 Waymo 上 PSNR 最高 31.9 dB、SSIM 0.781、LPIPS 0.441，在 nuScenes 上 PSNR 最高 29.7 dB、SSIM 0.775、LPIPS 0.319，且在视觉上细节更清晰、光照更真实。

**⚠️ 局限性**

局限性在于：① 采用逐序列重建方式，规模化至大规模数字孪生时存在可扩展性问题；② 当前光照模型对极大场景中的复杂光照可能不足；未来工作计划探索更高效的前向重建、丰富的材质先验、替代 BRDF 模型与更先进的色调映射。

---

## 646. Sufficient Conditions for Stability of Minimum-Norm Interpolating Deep ReLU Networks

**arXiv ID:** 2602.13910 | [PDF](https://arxiv.org/pdf/2602.13910v1)

**作者:** Ouns El Harzli `[一作]` (University of Oxford), Ard A. Louis `[通讯]` (University of Oxford)

**通讯引用:** 8564 | [OpenAlex ID](https://openalex.org/A5082711456)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了最小范数插值深度 ReLU 网络在算法稳定性方面的充要条件，揭示了稳定子网络与低秩层之间的相互作用；

**💡 创新点**

首次证明若存在稳定子网络且随后层为低秩，则整个网络在最小范数插值下保持稳定，并给出反例显示低秩缺失时可能不稳定；

**🔧 技术方法**

使用算法稳定性理论、最小范数插值分析、低秩偏差假设以及对称性与矩阵分解等技术；

**📊 数据集**

未使用具体公开数据集，而是假设存在可插值、权重有界的教师网络，并通过自构造的数据分布与 Toy 示例进行理论验证；

**📈 对比分析**

通过理论推导给出稳定性上界，证明在满足低秩偏差与稳定子网络时整体稳定性可控；与现有 SGD 稳定性结果相比，本文不需 Lipschitz、光滑或参数稳定性假设，且对 t≫n 的情况也适用；

**⚠️ 局限性**

局限在于稳定性上界依赖 B 的幂次，尤其在低秩不明显时会退化为 B^{3L}；此外假设最小范数插值可通过 oracle 获得，未直接对应实际优化过程（如 SGD）。

---

## 647. Adaptive Value Decomposition: Coordinating a Varying Number of Agents in Urban Systems

**arXiv ID:** 2602.13309 | [PDF](https://arxiv.org/pdf/2602.13309v1)

**作者:** Yexin Li `[一作]` (State Key Laboratory of General Artificial Intelligence), Zihao Jiao `[通讯]` (Beijing Technology and Business University)

**通讯引用:** 271 | [OpenAlex ID](https://openalex.org/A5055793419)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Adaptive Value Decomposition (AVD) 框架，用于在半马尔可夫决策过程（semi-MARL）中协调数量随时间变化的城市多智能体系统，并通过轻量级机制减少共享策略导致的行动同质化。

**💡 创新点**

创新点包括：1）动态适配可变人数的 agents，使用超网络根据当前活跃 agents 生成混合网络权重；2）通过分散初始化和在嵌入层加入随机扰动来促进行为多样性；3）针对异步动作持续时间的 CTDE 训练–执行策略，兼顾在线实时决策与全局协作。

**🔧 技术方法**

采用多头注意力（MHA）提取 agent 与全局状态嵌入，MLP 计算 Q 值、权重与偏置；混合网络满足单调性约束实现去中心化执行；轻量化扰动和分散初始化；训练使用经验回放与 Bellman 目标。

**📊 数据集**

真实数据集：伦敦 Santander Cycles（2023年8月）与华盛顿 D.C. Capital Bikeshare（2025年1月）两城市的自行车租赁日志，构建仿真器模拟租赁与调度。

**📈 对比分析**

与基线方法（No Action、Cooperative、IDQN、VDN+、QMIX）对比，AVD 在固定/动态 agent 数量下均显著提升累计租赁数；在零射击（zero-shot）场景下，AVD 仍优于直接训练的模型；实验显示其鲁棒性与可扩展性优于传统值分解方法。

**⚠️ 局限性**

局限性：1）实验仅在自行车调度任务上验证，缺乏对其他城市系统的通用性评估；2）在极大 agent 数量时，计算与内存开销可能随超网络规模增长；3）对部分可观测（partial observability）情况的处理仍有限；4）轻量化扰动参数需经验调优，可能影响收敛稳定性。

---

## 648. Mirror: A Multi-Agent System for AI-Assisted Ethics Review

**arXiv ID:** 2602.13292 | [PDF](https://arxiv.org/pdf/2602.13292v1)

**作者:** Yifan Ding `[一作]` (Fudan University), Guoyu Wang `[通讯]` (Fudan University)

**通讯引用:** 4822 | [OpenAlex ID](https://openalex.org/A5100769324)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了Mirror多代理框架，实现AI辅助的伦理评审，涵盖加速审查与全委员会讨论两种模式；

**💡 创新点**

创新点在于：① 专门从伦理规范构建EthicsQA数据集并用其微调得到EthicsLLM，实现领域专精的伦理推理；② 设计可执行的规则图与规则匹配机制，支持可复现的加速审查；③ 通过多代理协作模拟伦理委员会，完成深层次、多维度的讨论与决策；

**🔧 技术方法**

使用的技术包括：大规模语言模型（Qwen3-8B基础，Qwen3-14B/32B对照），监督微调（SFT）结合CoT与非CoT双模式；多代理架构（规则匹配、证据检索、辩论与合成代理）；检索增强生成（向量检索+交叉编码重排序）以及图结构化规则表示；

**📊 数据集**

数据集包括：EthicsCorpus（书籍、论文、法规、专家共识，超过10亿token）；EthicsQA（41,218个（问-链-答）三元组）；用于评测的真实伦理审查档案（100例加速审查），以及10例模拟委员会审查案例；

**📈 对比分析**

在伦理问答上，Mirror 8B在不使用CoT时取得76.07%（EthicsQA）/66.06%（EthicsQA‑ER），优于同规模Qwen3-8B和大模型；在加速审查中，Mirror 8B在Recall、Precision、F1、专业度上分别达到0.9444、0.7640、0.8447、0.7917，远超Zero‑shot和基准大模型；委员会审查通过案例研究表明Mirror能提出更细致、隐性伦理风险，且讨论更完整，效果优于GPT‑4.1和DeepSeek‑V3；

**⚠️ 局限性**

局限性包括：① 数据主要来自公开资源，缺乏真实IRB档案的丰富上下文和机构特定规则；② 评测案例偏向常规生命科学项目，未覆盖高危临床、跨中心或敏感社会行为研究；③ 以8B模型为主，受限于长文本推理与复杂辩论的能力；④ 对模型可解释性与人机协作的实证评估仍待加强。

---

## 649. Using Deep Learning to Generate Semantically Correct Hindi Captions

**arXiv ID:** 2602.13352 | [PDF](https://arxiv.org/pdf/2602.13352v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 650. Multimodal Consistency-Guided Reference-Free Data Selection for ASR Accent Adaptation

**arXiv ID:** 2602.13263 | [PDF](https://arxiv.org/pdf/2602.13263v1)

**作者:** Ligong Lei `[一作]` (Xinjiang University), Aishan Wumaier `[通讯]` (Xinjiang University)

**通讯引用:** 278 | [OpenAlex ID](https://openalex.org/A5077794475)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在无标签目标口音下，提出了一种基于多模态一致性（语音-文本嵌入对齐）和无参考 WER 预测的参考无关数据选择框架，用于提升 ASR 的口音适应效果。

**💡 创新点**

创新点在于：①将 SONAR 共享嵌入空间中的语音-文本一致性与预测 WER 结合，形成多模态质量指标；②通过模型/输入扰动产生多种假设，再用百分位阈值规则筛选，实现在无真实标注条件下的可靠伪标签选取；③结合 FLMI 预筛选提升查询相关性，降低后续计算成本。

**🔧 技术方法**

技术方法包括：FLMI 子模互信息预筛选；模型参数噪声与音频扰动解码生成多重假设；冻结 SONAR 编码器得到语音与文本的 1024 维共享嵌入；轻量 MLP 预测无参考 WER；基于预测 WER、余弦相似度和欧氏距离的改进度量与百分位阈值决策规则。

**📊 数据集**

实验数据集：IndicTTS English 子集（约150k句），VCTK 语料（110 位讲者），L2-ARCTIC（24 位非母语说话人，涵盖多种 L1）。

**📈 对比分析**

与文本感知 PPL 过滤、随机抽样、模型一致性与 SSL-WER 分类器等基线对比；在 2 轮微调预算下，所选子集在 IndicTTS 达到 10.91% WER，仅比 30k 监督标注高 0.46%；在跨域 L2-ARCTIC 中避免了未过滤伪标签导致的性能退化；在更强的 Paraformer 框架下，随时间预算升高，WERS 一直低于随机或 PPL 选取，且与最近非 PPL 基线相当或更优。

**⚠️ 局限性**

局限性：①阈值选择依赖经验，需手动调优；②方法在极端口音或非英语语种的泛化仍未充分验证；③对语音和文本的对齐质量依赖预训练 SONAR 的表现，若模型性能受限可能影响筛选效果；④当前仅适用于转录式口音适应，尚未扩展到多模态或跨域语音生成任务。

---

## 651. Debiasing Central Fixation Confounds Reveals a Peripheral "Sweet Spot" for Human-like Scanpaths in Hard-Attention Vision

**arXiv ID:** 2602.14834 | [PDF](https://arxiv.org/pdf/2602.14834v1)

**作者:** Pengcheng Pan `[一作]` (University of Tokyo), Yasuo Kuniyosh `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在对象中心数据集 Gaze-CIFAR-10 上，中心偏差如何导致传统扫描路径指标失真，并提出了新的 GCS 指标来评估硬注意力模型与人眼扫描路径的匹配度。

**💡 创新点**

创新点是提出中心偏差校正的 GCS 复合指标，并通过该指标发现“周边甜点”——在中等感知约束下，模型扫描路径最接近人类。

**🔧 技术方法**

使用多层递归注意力模型 MRAM 结合 REINFORCE 强化学习进行训练，评估多种扫描路径相似性指标。

**📊 数据集**

使用 Gaze-CIFAR-10 数据集的注视序列，包含 9,116 张图像。

**📈 对比分析**

与中心和角点基准对照，传统指标对中心策略给高分；GCS 发现最佳配置为 fov+per 8 像素，GCS=0.0291，准确率 58.5%，传统指标得分相对较低，展示了中心偏差校正后的更可靠对比。

**⚠️ 局限性**

仅在低分辨率、单一分类任务下验证，未涵盖更复杂任务或更大视角变化，且模型与人类相似度仍不完全一致。

---

## 652. Synergistic Intra- and Cross-Layer Regularization Losses for MoE Expert Specialization

**arXiv ID:** 2602.14159 | [PDF](https://arxiv.org/pdf/2602.14159v1)

**作者:** Rizhen Hu `[一作]` (Peking University), Kun Yuan `[通讯]` (Peking University)

**通讯引用:** 4264 | [OpenAlex ID](https://openalex.org/A5100614598)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了两种无侵入的正则化损失，分别是：1）内层专家专化损失，惩罚相同 token 在同一层被激活的专家间的激活相似度；2）跨层耦合损失，最大化相邻层专家对的联合路由概率，从而形成稳定的专家路径。

**💡 创新点**

创新点在于：①提出了两种 plug‑and‑play 损失，直接针对专家重叠和路由歧义这两大问题；②通过理论推导说明这两种损失能够相互强化，形成正反馈循环；③不需要改动 MoE 架构或路由器，兼容 DeepSeek‑MoE、Vanilla‑MoE 等多种模型；④将专家路径稳定化与专家专化结合，提升推理效率。

**🔧 技术方法**

技术细节包括：利用 SwiGLU 激活和下投影；使用余弦相似度计算专家间相似度；对同层激活的专家做 pairwise 正则化；对相邻层的路由概率做乘积并取 Top‑k 进行耦合正则；将两项损失加入 Megatron‑LM 的训练目标；配合 load‑balancing、z‑loss 等传统正则；在 GPU 上实现 O(k²d) 计算，计算量低。

**📊 数据集**

实验数据集：预训练使用公开文本数据 C4；下游 zero‑shot 评估涵盖 MMLU、MMLU‑Pro、HellaSwag、BBH、GPQA‑Diamond、MBPP、HumanEval、GSM8K；LoRA 微调实验使用三大 16B MoE 基础模型（DeepSeek‑MoE‑16B、DeepSeek‑V2‑Lite、Ling‑mini‑2.0）；内部 38B token 语料进行全参数微调；还在 Qwen3‑30B‑A3B 上测试。

**📈 对比分析**

与 baseline（仅 load‑balancing 或 load‑balancing+z‑loss）、variance‑on‑logits 等对比实验显示：① perplexity 在各模型规模上下降 1–3%；② downstream 指标（如 MMLU、HumanEval、GSM8K 等）平均提升 3–5 分；③推理吞吐量提升约 5–10% 由于专家路径更稳定；④实验表明两种损失可独立使用，组合效果更佳。

**⚠️ 局限性**

局限性：①正则化参数 λ 的选择对效果影响显著，需要调参；②在极大规模模型或不同路由策略（如 ReMoE、Hash Layers）下的通用性尚未充分验证；③额外的相似度/耦合计算在 GPU 内存上仍有一定开销；④实验集中在 top‑k MoE，可能不适用于 sparse‑gated 或 dynamic‑k 方案。

---

## 653. On the Sparsifiability of Correlation Clustering: Approximation Guarantees under Edge Sampling

**arXiv ID:** 2602.13684 | [PDF](https://arxiv.org/pdf/2602.13684v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Anuj Sharma `[通讯]` (Iowa State University)

**通讯引用:** 2958 | [OpenAlex ID](https://openalex.org/A5083087081)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了相关聚类（CC）的稀疏化-近似权衡，探讨了保留LP基础保证所需的边信息量。

**💡 创新点**

提出了伪度量和一般加权实例之间的结构二分法，证明了在伪度量条件下，稀疏化可以有效进行，而在没有此结构时，稀疏化会导致近似性丧失。

**🔧 技术方法**

使用了线性规划（LP）和稀疏-LP-PIVOT算法，结合三角不等式进行缺失边的插补。

**📊 数据集**

使用了合成和真实世界的数据集，包括政治博客、Facebook自我网络等，进行实验验证。

**📈 对比分析**

与现有的CC算法（如PIVOT和KwikCluster）进行了比较，Sparse-LP-Pivot在样本预算达到Θ̃(n^3/2)时表现出接近最优的近似比，低于该阈值时性能显著下降。

**⚠️ 局限性**

在没有伪度量结构的情况下，任何观察到o(n)边的算法都无法实现常数因子的近似，显示出稀疏化的局限性。

---

## 654. DiffusionRollout: Uncertainty-Aware Rollout Planning in Long-Horizon PDE Solving

**arXiv ID:** 2602.13616 | [PDF](https://arxiv.org/pdf/2602.13616v1)

**作者:** Seungwoo Yoo `[一作]` (Korea Advanced Institute of Science and Technology), Minhyuk Sung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1309 | [OpenAlex ID](https://openalex.org/A5004099860)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于扩散模型的 PDE 求解器，并在自回归滚动过程中引入了基于预测不确定性的自适应步长策略；

**💡 创新点**

利用多样本标准差作为模型预测误差的代理，通过自适应选择步长来减轻曝光偏差，实现了无需重新训练即可提升长时序预测的准确性；

**🔧 技术方法**

扩散模型（DiT）+ 3D 自注意力 + 旋转嵌入；DDIM 采样；多样本标准差估计；自适应步长选择；

**📊 数据集**

四个公开 PDE 基准：Gray‑Scott 反应扩散、2D 湍流辐射流、Cahn‑Hilliard 相分离、各向异性扩散；

**📈 对比分析**

与 FNO、FactFormer、Transolver 等回归式神经算子以及 PDE‑Refiner、WDNO 等扩散式算子进行对比；在相同模型规模下，提出方法在相对 L₂ 误差和长期稳定性 T>0.9 指标上均优于所有基线，错误率下降 10‑20% 以上；

**⚠️ 局限性**

推理速度慢（需要多步扩散采样）；需要手动调节不确定性阈值 τ；对非确定性系统或高维空间的推广仍有待验证；

---

## 655. Cold-Start Personalization via Training-Free Priors from Structured World Models

**arXiv ID:** 2602.15012 | [PDF](https://arxiv.org/pdf/2602.15012v1)

**作者:** Avinandan Bose `[一作]` (Meta), Asli Celikyilmaz `[通讯]` (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段的冷启动偏好推理框架，先离线学习偏好结构，再在线进行贝叶斯推断和信息增益驱动的主动提问，最终预测完整偏好并生成个性化回复。

**💡 创新点**

创新点在于将偏好结构学习与在线推断分离，利用全域稠密标签学习协方差结构，避免端到端RL的稀疏奖励导致的信用分配难题，实现训练无关的在线推理。

**🔧 技术方法**

使用贝叶斯线性回归或高斯混合模型做隐变量建模，贝叶斯更新与信息增益或不确定性采样的主动查询策略，结合固定的LLM生成器。

**📊 数据集**

利用PrefDisco四大领域（MedQA、AIME、CommonsenseQA、SocialIQA）的完整偏好标注数据进行离线训练。

**📈 对比分析**

与Prompting、CollabLLM、Population Average、GRPO等基线比较，取得77–87% Oracle对齐率（相比GRPO 55–76%），并在查询效率上比RL提升2–15倍，适配率高达39–62%。

**⚠️ 局限性**

局限在于仅针对结构化标准问题/答案的模拟用户，缺乏自然语言自由提问、长期会话学习及动态偏好更新，且可能因训练数据中的社会偏见导致偏好错误。

---

## 656. Synthetic Reader Panels: Tournament-Based Ideation with LLM Personas for Autonomous Publishing

**arXiv ID:** 2602.14433 | [PDF](https://arxiv.org/pdf/2602.14433v1)

**作者:** Fred Zimmerman `[一作]` `[通讯]` (Nimble Books LLC / Big Five Killer LLC), Fred Zimmerman (Nimble Books LLC / Big Five Killer LLC)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并部署了基于LLM生成的合成读者面板，通过结构化的锦标赛机制对书籍概念进行大规模评估与筛选。

**💡 创新点**

创新点在于：①将多样化的LLM角色化为具有人口统计与行为特征的“读者人格”，②采用锦标赛筛选策略替代传统手工评审，③引入五维反“slop”检测机制提升评估质量。

**🔧 技术方法**

使用技术包括：大规模语言模型（LLM）生成与角色扮演、基于规则的多轮锦标赛引擎、文本相似度与统计检测（如n-gram、TTR、标准差等）以及JSON结构化输出与批量处理。

**📊 数据集**

使用的数据集涵盖：6个出版 imprint 的目标受众属性、609 本在售书目、3个案例研究中的评审数据（270 人、5 人两组），以及历史评审与销售数据用于后期校准。

**📈 对比分析**

对比方法：将锦标赛筛选出的书籍与随机抽取的未筛选书籍在编辑评分、市场适配度、原创性等指标上进行对照。结果显示，筛选后高质量书籍比例从 15% 提升至 62%，低质量书籍比例从 25% 降至 0%。

**⚠️ 局限性**

限制包括：LLM 本身的偏见导致评分偏高或对近期话题过敏、对文化/语言细微差异缺乏真实体验、面板无法完全再现人类的情感与社群动态，且需人工后续校准与审核。

---

## 657. Beyond Words: Evaluating and Bridging Epistemic Divergence in User-Agent Interaction via Theory of Mind

**arXiv ID:** 2602.13832 | [PDF](https://arxiv.org/pdf/2602.13832v1)

**作者:** Minyuan Ruan `[一作]` (Tsinghua University), Yang Liu `[通讯]` (Tsinghua University)

**通讯引用:** 112895 | [OpenAlex ID](https://openalex.org/A5100355638)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过把理论心智（ToM）从单纯的心理状态推断转向在交互场景中解决认知分歧的实用机制，提出了同步ToM（SynchToM）基准，用以评估大型语言模型在识别并修正用户与真实环境之间的知识差距时的能力，并在此基准上对多种模型进行系统测试和强化学习提升。

**💡 创新点**

创新点：① 重新定义ToM为“认知分歧检测与调和”而非仅是推理；② 设计四大实际场景（软件工程、用户偏好、教育、文化差异）并生成多轮交互轨迹，构成真实的用户信念、配置与目标三元组；③ 通过强化学习（GRPO）在轨迹数据上训练，形成“ToM‑增强”模型；④ 提出多模型协同框架，用其他模型的心智推断来提升目标模型的解题效果。

**🔧 技术方法**

技术：Bayesian Theory of Mind 推断、LLM-as-Judge 自动评估、强化学习（GRPO）与专门的 ToM 奖励（belief、profile、solution 维度），以及多代理协同推理。

**📊 数据集**

数据集：SynchToM（390条高质量测试实例 + 6522条训练轨迹），来源于软件工程、偏好建模、教育、文化差异等四个原始基准，并通过 LLM 合成并经过验证模块确保信念、轨迹和目标的一致性。

**📈 对比分析**

对比方法：在 11 个模型（含 GPT‑5、Gemini‑3、Claude‑Sonnet‑4.5 等专有模型与 Qwen、MiniMax、Kimi‑K2、InternLM、Llama‑4 等开源模型）上进行“vanilla、5‑turn、10‑turn”三种推理强度评测；使用 LLM‑judge 计算 Belief、Profile、Solution 三维得分。实验结果显示：开源模型在某些场景下优于专有模型；Belief 维度普遍是瓶颈；RL 训练后在 Solution 维度显著提升，尤其在 5‑turn、10‑turn 轨迹下可达约 7–12% 的提升。

**⚠️ 局限性**

限制：① 评估仍依赖自动 LLM‑judge，可能存在主观性；② 训练数据主要为合成场景，真实复杂场景的迁移性未知；③ 强化学习需要大量轨迹与奖励设计，难以在多任务环境下统一；④ 仍缺乏对极端信息稀缺或高度多模态场景的深入验证。

---

## 658. The Rise of AI Search: Implications for Information Markets and Human Judgement at Scale

**arXiv ID:** 2602.13415 | [PDF](https://arxiv.org/pdf/2602.13415v1)

**作者:** Sinan Aral `[一作]` (Massachusetts Institute of Technology), Rui Zuo `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在2024年至2025年间，对243个国家执行24,000条相同查询，采集2.8百万条AI与传统搜索结果，系统评估全球AI搜索曝光度、内容多样性、可信度与政治倾向，并分析其对点击行为、信息经济与决策的影响；

**💡 创新点**

首次采用时间对比、相同查询的纵向研究，剖析平台政策如何推动AI搜索的快速扩张，并揭示其在信息多样性、来源可信度与政治偏见方面的显著变化；

**🔧 技术方法**

使用日志爬取（serpAPI）、统计学分析（逻辑回归、差异显著性检验）、信息多样性度量（信息唯一性）、来源信誉评估（Media Bias / Fact Check数据库）、点击率与零点击率分析；

**📊 数据集**

12,000条查询样本来源于Google Natural Questions、Most‑Searched Google Queries、Covid‑Related Search Queries、Amazon Shopping Queries等9个真实用户数据集；

**📈 对比分析**

通过对比AI与传统搜索结果的曝光比例、引用率、域名分布、响应多样性、点击率与零点击率，发现AI搜索在2025年曝光率提升至55–70%，引用率下降，长尾域名访问显著减少；

**⚠️ 局限性**

仅覆盖Google平台，无法捕捉其他AI搜索引擎的政策与效果；查询集虽多样但不完全代表所有搜索行为；难以精准评估所有政策变更与模型准确性；

---

## 659. Efficient Text-Guided Convolutional Adapter for the Diffusion Model

**arXiv ID:** 2602.14514 | [PDF](https://arxiv.org/pdf/2602.14514v1)

**作者:** Aryan Das `[一作]` (VIT Bhopal), Vinay Kumar Verma `[通讯]` (IIT Kanpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出轻量级、文本引导的结构保持适配器Nexus Prime和Slim，扩展冻结的扩散模型，使其能同时接受结构输入（如边缘图、深度图、草图、分割掩码）和文本提示进行高质量图像生成。

**💡 创新点**

创新点：① 在适配器内部联合建模文本提示与结构输入，使用跨注意力实现多模态对齐；② 设计两种高效适配器结构——Prime使用标准卷积，Slim采用深度可分卷积；③ 仅添加约8M参数或减少18M参数即可显著提升性能，保持基础模型冻结。

**🔧 技术方法**

采用Latent Diffusion（Stable Diffusion v1.5）与CLIP文本编码器；在适配器中使用跨注意力、轻量级卷积（标准/深度可分）；对条件图像做像素反解、层次特征提取；通过Flops、参数量评估及FID/CLIP Score进行性能测评。

**📊 数据集**

在COCO 2017训练集（约164k张图）上训练，构造Canny、Depth、Sketch、Segmentation四种条件；在COCO验证集（5k张图）上评估。

**📈 对比分析**

与ControlNet、ControlNet++、T2I‑Adapter、CtrLoRA、UniCon等基线在相同基础模型上对比，使用FID和CLIP Score评估。Nexus Prime在Canny、Depth、Sketch任务中获得最高CLIP分数和最低FID，Segmentation排名第二；Nexus Slim在保持更低参数和Flops的情况下取得与Prime相近的性能，整体实现高质量、低成本的结构保持生成。

**⚠️ 局限性**

局限性：① 仅验证四种结构条件，尚未覆盖其他条件或多模态组合；② 在Segmentation任务上仍略逊于ControlNet++，对复杂语义分割有提升空间；③ 依赖冻结的Stable Diffusion和CLIP，若基础模型更新需重新适配；④ 适配器仍需针对每种条件单独训练，增加数据准备成本。

---

## 660. Skeleton2Stage: Reward-Guided Fine-Tuning for Physically Plausible Dance Generation

**arXiv ID:** 2602.13778 | [PDF](https://arxiv.org/pdf/2602.13778v1)

**作者:** Jidong Jia `[一作]` (Shanghai Jiao Tong University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 98725 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

利用强化学习微调(diffusion)生成舞蹈，加入物理基奖励以弥补骨架与网格之间的物理不一致问题。

**💡 创新点**

① 设计基于仿真器的模仿奖励、Foot‑Ground Deviation奖励与引导以及抗冻结奖励；② 将物理约束作为奖励注入RLFT，避免训练时使用运动投影；③ 通过这些奖励显著降低自碰撞与脚接触异常。

**🔧 技术方法**

强化学习微调（REINFORCE）、SMPL+IsaacGym物理仿真、Diffusion模型（EDGE/POPDG/GENMO）、物理奖励与引导。

**📊 数据集**

使用AIST++和PopDanceSet数据集，训练时也采集AMASS、AIST++等舞蹈序列。

**📈 对比分析**

与EDGE、POPDG、GENMO、Bailando++等SOTA对比；在人类评估、Penetration Rate、PFC、FGD、BAS、Diversity等指标上均优于基线，冻结率下降，物理可行性显著提升。

**⚠️ 局限性**

仍受复杂舞蹈场景限制，奖励设计需要手工调参，RLFT样本效率低，无法完全消除所有自碰撞；对实时多模态同步等仍有改进空间。

---

## 661. Privacy-Concealing Cooperative Perception for BEV Scene Segmentation

**arXiv ID:** 2602.13555 | [PDF](https://arxiv.org/pdf/2602.13555v1)

**作者:** Song Wang `[一作]` (Zhengzhou University), Guanghui Wang `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 8842 | [OpenAlex ID](https://openalex.org/A5026566798)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一个隐私保护的合作感知框架（PCC），通过隐藏网络与重构网络对抗训练，降低共享BEV特征的图像重构质量，同时保持语义分割性能。

**💡 创新点**

创新点在于将对抗学习应用于BEV特征隐私保护，提出轻量级隐藏网络并联合重新训练感知网络，实现隐私泄露降低与性能损失平衡。

**🔧 技术方法**

使用了对抗学习、MAE解码器、VGG感知损失、交叉熵分割损失以及轻量级1×1卷积隐藏网络。

**📊 数据集**

采用了OPV2V数据集进行训练与评估。

**📈 对比分析**

与原始BEV特征对比，重构图像的FID、PHV、SSIM、PSNR显著下降，语义分割IoU仅下降≤0.3%（从0.9635降至0.9673），实现了低性能损失的隐私保护。

**⚠️ 局限性**

限制在于仅在语义分割任务上验证，隐藏网络虽轻量但可能不适用于更复杂的任务，且对不同网络架构的适应性未完全评估。

---

## 662. Agents in the Wild: Safety, Society, and the Illusion of Sociality on Moltbook

**arXiv ID:** 2602.13284 | [PDF](https://arxiv.org/pdf/2602.13284v1)

**作者:** Yunbei Zhang `[一作]` (Tulane University), Yingqiang Ge `[通讯]` (Rutgers University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对AI-only社交平台Moltbook进行了大规模实证研究，分析了其社会结构、安全威胁与互动质量。

**💡 创新点**

首次揭示AI社会的快速形成与“社交幻觉”，并发现社交攻击比技术攻击更有效。

**🔧 技术方法**

采用关键词分析、攻击检测器、网络图分析等技术，并结合情感、主题与互动深度指标进行量化。

**📊 数据集**

使用Moltbook Observatory Archive公开数据集，涵盖27,269个代理、137,485条帖子和345,580条评论。

**📈 对比分析**

将AI互动与人类平台进行对比，发现AI对话递归深度极低，攻击内容获得6倍以上的参与度。

**⚠️ 局限性**

局限在关键词检测偏差、仅观察9天、无法得出因果关系、协调检测可能漏报，以及仅针对单一平台。

---

## 663. Benchmarking Anomaly Detection Across Heterogeneous Cloud Telemetry Datasets

**arXiv ID:** 2602.13288 | [PDF](https://arxiv.org/pdf/2602.13288v1)

**作者:** Mohammad Saiful Islam `[一作]` (Toronto Metropolitan University), Andriy Miranskyy `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 765 | [OpenAlex ID](https://openalex.org/A5026957450)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对多种深度学习模型在不同云监控数据集上的异常检测性能进行统一评估，重点考察模型与数据特征、校准稳定性之间的关系。

**💡 创新点**

创新点在于提出严格的训练‑仅校准评估框架、跨数据集对比以及基于几何分布偏移的诊断分析，揭示了传统NAB评分下负分数往往是校准失效而非模型失效。

**🔧 技术方法**

使用的技术包括GRU、TCN、Transformer、TSMixer四种自动编码器架构、Isolation Forest基线以及基于滚动窗口的似然校准与阈值设定。

**📊 数据集**

采用四个公开数据集：NAB（单变量多域）、Microsoft Cloud Monitoring（单变量云指标）、Exathlon（高维Spark工作负载）和IBM Console（超高维生产云API指标）。

**📈 对比分析**

方法通过统一的训练、校准和测试流程比较，结果显示不同模型在不同子集上表现差异明显，GRU和TCN在多数子集中排名靠前，但在Exathlon和IBM Console的高维环境下表现不佳；整体指标呈现极端波动，强调校准与数据几何一致性的重要性。

**⚠️ 局限性**

局限性包括仅采用静态校准参数、缺乏在线或自适应阈值策略、未深入探究高维数据的多模态特征处理以及未对不同异常类型的检测能力做细粒度评估。

---

## 664. A First Proof Sprint

**arXiv ID:** 2602.13587 | [PDF](https://arxiv.org/pdf/2602.13587v1)

**作者:** Joseph Corneli `[一作]` `[通讯]` (Hyperreal Enterprises Ltd), Joseph Corneli (Hyperreal Enterprises Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一次压缩式多代理证明冲刺中，作者尝试并记录了十个研究级问题的证明过程，形成了一部包含工作流程、结果与经验总结的单卷手稿。

**💡 创新点**

创新点在于将证明依赖拆解为连线图（wiring‑diagram）以实现缺口本地化与针对性修复；采用抗衡审计与层切换策略，提高压缩冲刺的可靠性与可校准性；同时系统化记录了代理交互与人机对话的证据，提供了可复现的证明工程范例。

**🔧 技术方法**

使用技术包括：多代理推理与验证框架（Claude、Codex 及自定义验证器）、Git 版本控制与日志追踪、Python 及 NumPy/SciPy/SymPy 等数值与符号工具、PHCpack 与 cvxpy 等多项式/半正定优化库、AIF（Active Inference Framework）思路与结构化推理图、以及批判-回答（Proofs‑and‑Refutations）循环。

**📊 数据集**

数据集主要来自公开数学知识库（PlanetMath、nlab、Math StackExchange、MathOverflow）、Git 仓库历史记录以及实验产生的数值/符号计算结果。

**📈 对比分析**

通过对比验证状态（Closed、Partial、Conditional）与验证完成度（完整、部分、未完成）以及缺口定位与修复的时间成本，作者展示了该方法在快速生成证明、精确定位错误与提升验证效率方面优于传统手工流程；实验显示，连线图和层切换显著降低了人工干预次数并缩短了整体完成时间。

**⚠️ 局限性**

局限性包括：仍有部分证明保持条件性或未完全验证；对外部已知定理的依赖导致部分缺口无法内部化；计算资源受限（如大规模符号求解与 SDP 计算）；缺乏实时的推理记录与可视化框架，导致后期需要手工重建证明结构；以及在更复杂或更大规模问题上可扩展性尚未完全验证。

---

## 665. Disentangling Deception and Hallucination Failures in LLMs

**arXiv ID:** 2602.14529 | [PDF](https://arxiv.org/pdf/2602.14529v1)

**作者:** Haolang Lu `[一作]` (Beijing University of Posts and Telecommunications), Kun Wang `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建实体问答环境，区分LLM的幻觉与欺骗失败机制，并验证知识存在与行为表达的内在差异。

**💡 创新点**

提出将失败拆解为知识存在(K)与行为表达(B)的机制视角，构造四种行为模式，并通过角度激活编辑证明欺骗可逆、幻觉不可逆，揭示两者根本机制不同。

**🔧 技术方法**

使用内部瓶颈分类器、稀疏自编码器（SAE）和层局部角度激活编辑技术，进行表示可分离性、可解释性与因果干预分析。

**📊 数据集**

使用约4000+条基于实体（城市、电影、球员、歌曲）的实体问答数据集，人工生成四种行为模式（正确、回避、错误答案、幻觉），并通过jailbreak验证知识保存。

**📈 对比分析**

四分类器与二分类器在内部表示上分别达到最高约92% 的准确率；稀疏特征数量约千级；激活编辑在欺骗场景下实现约30–50% 的行为转换，幻觉场景无法恢复，表明机制差异明显。

**⚠️ 局限性**

实验仅涵盖单跳实体问答，未考虑多跳推理；数据由人工构造，缺乏更广泛多样性；激活编辑依赖可观测层，跨模型/任务的泛化性尚待验证。

---

## 666. ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System

**arXiv ID:** 2602.13692 | [PDF](https://arxiv.org/pdf/2602.13692v1)

**作者:** Hao Kang `[一作]` (Georgia Institute of Technology), Simran Arora `[通讯]` (Together AI)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种基于程序抽象的 Agentic 推理系统，能够在多 GPU、CPU 与外部工具环境之间统一调度和资源管理，以显著提升 LLM 多轮推理的吞吐量和 RL Rollout 的效率。

**💡 创新点**

核心创新包括：①将 Agentic 工作流抽象为可持久化的“程序”，使调度器获得对整个流程的全局视图；②设计基于程序状态的调度策略（状态感知暂停、动态迁移、全局等待队列），最小化 KV 缓存抖动与跨节点内存不平衡；③实现工具生命周期管理（生命周期回收、异步环境预置），消除资源泄漏与环境准备瓶颈。

**🔧 技术方法**

实现技术涵盖：LLM 推理引擎（vLLM、SGLang）、Docker 容器化工具执行、GPU 资源约束优化（STP 成本模型）、KV 缓存状态监控与时间衰减函数、全局程序等待队列、生命周期回收钩子及异步预置机制。

**📊 数据集**

实验使用的基准与数据集包括：SWEBench‑Lite、HLE‑Bench、ScienceAgentBench、SWE‑Agent、OpenHands 以及 GLM‑4.6 与 Qwen‑3 大模型，覆盖编码、路由、科学研究 Agent 以及 RL Rollout 场景。

**📈 对比分析**

与 vLLM、Continuum、vLLM+SGLang Gateway 等现有系统对比，本系统在服务端吞吐量上提升 1.48‑3.58 倍，在 RL Rollout 上提升 1.79‑3.92 倍；KV 缓存命中率接近 100%，并实现 4.2 倍的磁盘内存节省，整体性能稳健且对并发度和工具执行时延的随机性具有良好适应性。

**⚠️ 局限性**

局限性主要体现在：①实现依赖 GPU 推理引擎和 Docker 环境，扩展到其他推理后端或无容器化工具的支持尚未验证；②程序抽象与调度策略在极端长生命周期或高度随机工具调用场景下可能仍产生一定的调度开销；③实验规模受限于 8×H100 集群，未验证在更大规模集群或多租户环境中的可扩展性。

---

## 667. Boule or Baguette? A Study on Task Topology, Length Generalization, and the Benefit of Reasoning Traces

**arXiv ID:** 2602.14404 | [PDF](https://arxiv.org/pdf/2602.14404v1)

**作者:** William L. Tong `[一作]` (Harvard University), Cengiz Pehlevan `[通讯]` (Harvard University)

**通讯引用:** 1822 | [OpenAlex ID](https://openalex.org/A5023195984)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了推理轨迹（RT）与直接预测（DP）模型在命题逻辑推理中的长度泛化性能，提出并构建了大规模可扩展的 PITA 数据集，并通过转移推理（TI）任务对实验结果进行理论验证。

**💡 创新点**

创新点包括：①设计全新的 PITA 数据集，提供 2300 万命题及 950 亿 token 的 Lean 形式化证明；②定义任务深度与宽度两种拓扑度量，系统化评估 RT 与 DP 在不同拓扑下的泛化差异；③使用 TI 任务和最大间隔理论推导解释实验现象，证明结论具普适性。

**🔧 技术方法**

技术手段主要为：Transformer 预训练模型（Qwen2.5‑Coder、Gemma 3、Llama 3）通过 QLoRA 微调；在训练与测试阶段采用长度泛化实验；使用单层单头 Transformer 对 TI 任务做自注意力与 MLP 设计；对推理轨迹进行自动生成与验证。

**📊 数据集**

使用的数据集为：①PITA——23M 条命题与 95B token 的 Lean 形式化证明；②合成的转移推理（TI）数据集，用于验证理论与实验一致性。

**📈 对比分析**

比较方法：将 PITA 按任务深度与宽度划分为 Full、Imply、Or、PHP 四个子集；训练模型至各子集的中位证明长度后，在更长证明上评估准确率；与直接预测模型对比。结果显示：在宽浅（boule‑shaped）子集上 RT 表现优于 DP；在窄深（baguette‑shaped）子集上 DP 明显优于 RT。TI 任务的实验与理论推导复现了同一趋势。

**⚠️ 局限性**

局限性：RT 模型受长上下文影响，深任务中生成长推理轨迹导致性能下降；DP 模型在宽度增大时易过拟合；实验仅覆盖命题逻辑与合成 TI，未涉及更复杂的语义推理；未来需改进长上下文处理与更通用的任务拓扑分析。

---

## 668. Scope: A Scalable Merged Pipeline Framework for Multi-Chip-Module NN Accelerators

**arXiv ID:** 2602.14393 | [PDF](https://arxiv.org/pdf/2602.14393v1)

**作者:** Zongle Huang `[一作]` (Tsinghua University), Yongpan Liu `[通讯]` (Tsinghua University)

**通讯引用:** 8167 | [OpenAlex ID](https://openalex.org/A5045721867)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

本文提出 Scope 框架，将多层合并为“cluster”进行分段流水线调度，以提升多芯片模块（MCM）上的 NN 推理吞吐量。

**💡 创新点**

创新点在于引入 cluster 维度实现多层合并，既平衡工作负载又降低 NoP 通信和资源低利用率，并通过动态规划+启发式搜索在指数级搜索空间中快速寻找近最优调度。

**🔧 技术方法**

使用的技术包括：动态规划（DP）压缩合并搜索空间、启发式区域分配、分布式权重缓冲、计算与通信重叠、以及 Timeloop、BookSim2、Ramulator2 等仿真工具。

**📊 数据集**

实验使用了 AlexNet、VGG16、DarkNet19、ResNet18/34/50/101/152 等标准 CNN 数据集，并在 16–256 芯片的 MCM 架构上评估。

**📈 对比分析**

与全序、全流水线、分段流水线等现有调度方法相比，Scope 在所有规模下均实现了更高吞吐量，最大提升达 1.73×，能耗与现有方法相当，并展现出更优的规模扩展性。

**⚠️ 局限性**

局限性包括：合并策略仍需人工调优，搜索过程对非常不规则网络可能不够鲁棒；在极大规模 MCM 上仍需高效的硬件资源管理；以及对权重分布假设的依赖可能限制了对某些模型的适用性。

---

## 669. Multi-Agent Debate: A Unified Agentic Framework for Tabular Anomaly Detection

**arXiv ID:** 2602.14251 | [PDF](https://arxiv.org/pdf/2602.14251v1)

**作者:** Pinqiao Wang `[一作]` (University of Virginia), Sheng Li `[通讯]` (University of Virginia)

**通讯引用:** 11626 | [OpenAlex ID](https://openalex.org/A5100359839)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了MAD（Multi-Agent Debate）框架，用多模型检测器的分歧作为信息，通过协同协调生成最终异常分数并输出可追溯的辩论轨迹。

**💡 创新点**

创新点在于：①将模型间分歧视为可利用的信号而非噪声；②设计消息空间（标准化分数、置信度、证据）和失真合成器Ψ，实现可解释的争议解决；③通过指数梯度更新保持理论保证，并可收敛到现有集合方法。

**🔧 技术方法**

核心技术包括：多模型异常检测器、置信度估计（如Bootstrap方差或温度校准）、结构化证据（特征归因或反事实），LLM评论员用于证据一致性检查，指数梯度权重更新，合成损失构造，符合性校准。

**📊 数据集**

在多领域的10+表格异常数据集上评测，包括金融（Credit Card Fraud、IEEE–CIS Fraud）、医疗（Mammography、Arrhythmia）、生态（Shuttle、CoverType）、网络安全（UNSW-NB15、NSL-KDD、CIC-IDS2017）、经济学（Bank Marketing、Adult Income）等。

**📈 对比分析**

与经典模型（RF、XGB、LGBM、CatBoost、HeteroStack）、AutoML（AutoGluon、H2O、auto-sklearn）、深度表格网络（TabNet、FT-Transformer、SAINT、TabTransformer、TabPFN）以及无监督方法（iForest、OC-SVM、LOF、Elliptic Envelope、PyOD）比较。MAD在PR‑AUC、Recall@1%FPR、ROC‑AUC、F1、ECE、Slice Gap等指标上均实现了显著提升，尤其在高分歧场景和稀疏异常率任务中优势明显。

**⚠️ 局限性**

局限包括：额外的推理成本（模型数×辩论轮次）；对多样化模型的依赖，若模型高度相关则收益有限；置信度与证据质量噪声会影响性能；辩论轨迹虽可解释但仍需人工总结，且持续分歧可能反映偏差或欠规范而非真正不确定。

---

## 670. Temporal Shifts and Causal Interactions of Emotions in Social and Mass Media: A Case Study of the "Reiwa Rice Riot" in Japan

**arXiv ID:** 2602.14091 | [PDF](https://arxiv.org/pdf/2602.14091v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 671. Plan-MCTS: Plan Exploration for Action Exploitation in Web Navigation

**arXiv ID:** 2602.14083 | [PDF](https://arxiv.org/pdf/2602.14083v1)

**作者:** Weiming Zhang `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18142 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将网页导航任务转化为语义计划空间搜索的框架，利用稠密计划树和抽象语义历史实现高效探索与精确状态感知。

**💡 创新点**

创新点在于：①把搜索粒度提升到计划空间，克服原子动作稀疏与噪声；②引入 Dual‑Gating Reward 严格验证可执行性与战略一致性；③采用 Structural Refinement 在策略层面动态修正失败子计划。

**🔧 技术方法**

使用 LLM 结合 MCTS 在计划空间进行搜索；通过 Operator 将高层子计划落地为原子动作；通过 Reflector 进行结构修正；采用 UCT 公式进行选择；微观/宏观评分双门评估用于奖励设计。

**📊 数据集**

数据集为 WebArena benchmark，包含 812 个任务，涵盖购物、购物管理、Reddit、地图和 Gitlab 等五个领域。

**📈 对比分析**

在相同环境下与 ReAct、BrowserGym、WebPilot、Branch‑and‑Browse 等基线对比，使用 Success Rate 评估。该方法在所有领域均高于基线，计划空间搜索相较原子动作搜索提升约10–20%成功率，动作交互量减少约30%，路径长度更短。

**⚠️ 局限性**

局限性：仍依赖 LLM 推理成本高；在极大搜索预算下性能提升有限；计划生成与修正受限于 LLM 的准确性，错误子计划可能导致搜索偏移；尚未在极端多页面或非文本界面上验证。

---

## 672. Search in Transition: A Study of University Students Perspectives on Using LLMs and Traditional Search Engines in English Test Problem Solving for Higher Study

**arXiv ID:** 2602.13629 | [PDF](https://arxiv.org/pdf/2602.13629v1)

**作者:** Tarek Rahman `[一作]` (United International University), Jannatun Noor Mukta `[通讯]` (United International University)

**通讯引用:** 292 | [OpenAlex ID](https://openalex.org/A5038795848)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大学生在英语考试备考中使用大语言模型与传统搜索引擎的使用模式和效果，并提出将LLM助手嵌入搜索界面的混合原型。

**💡 创新点**

通过混合方法揭示两类工具互补的使用策略，提出一种能减少工具切换负荷的交互原型。

**🔧 技术方法**

采用问卷调查+访谈的混合研究、ANOVA与卡方检验统计分析、主题编码，原型使用LLM（如ChatGPT）与搜索结果集成的HCI设计。

**📊 数据集**

使用140名学生的问卷数据和20名学生完成七项模拟IELTS/TOEFL任务的表现数据，涵盖摘要、语法纠错、词汇、改写、写作等。

**📈 对比分析**

将LLM-only、搜索-only、平衡与随机组在准确率与完成时间上进行对比；结果显示LLM在效率和满意度上优于搜索，混合组准确率最高（90%）但耗时最长。

**⚠️ 局限性**

样本偏向技术系且访谈规模小；未正式评估原型；自报数据可能存在偏差；未控制受访者的技术熟练度与经验。

---

## 673. DCDM: Divide-and-Conquer Diffusion Models for Consistency-Preserving Video Generation

**arXiv ID:** 2602.13637 | [PDF](https://arxiv.org/pdf/2602.13637v1)

**作者:** Haoyu Zhao `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 24146 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Divide-and-Conquer Diffusion Model（DCDM），通过将视频一致性问题拆分为三类：片段内世界知识一致性、片段间摄像机一致性和镜头间元素一致性，并为每类设计专门的模块来提升视频生成质量。

**💡 创新点**

创新点在于：①将一致性问题系统化拆分并在同一Diffusion Transformer骨干上并行处理；②使用大型语言模型（LLM）进行提示扩展，实现片段内语义与物理一致性；③在噪声空间引入时间结构化摄像机表示，保证摄像机运动稳定；④采用稀疏的镜头间自注意机制，兼顾长程叙事一致性与计算效率。

**🔧 技术方法**

核心技术包括：Diffusion Transformer（DiT）作为共享视频生成骨干；LLM（如Qwen3）用于提示解析与扩展；噪声空间摄像机控制与参考图像初始化；窗口交叉注意与稀疏镜头间自注意。

**📊 数据集**

评测使用了AAAI'26 CVM Competition基准数据集，涉及多种文本提示与摄像机运动场景。

**📈 对比分析**

与现有统一模型（如CogVideoX、Seedance）对比，DCDM在文本语义一致性、摄像机运动稳定性及长时叙事连贯性方面均表现出显著提升，生成视频的质量和一致性指标均高于基线。

**⚠️ 局限性**

局限性包括：对摄像机运动的类别预定义可能限制了细粒度控制；LLM扩展过程增加推理延迟；稀疏自注意在极长时序或复杂叙事中仍可能丢失细节；整体模型规模仍较大，对算力要求高。

---

## 674. Arming Data Agents with Tribal Knowledge

**arXiv ID:** 2602.13521 | [PDF](https://arxiv.org/pdf/2602.13521v1)

**作者:** Shubham Agarwal `[一作]` (University of California Berkeley), Aditya G. Parameswaran `[通讯]` (University of California Berkeley)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个可插拔框架——Tribal Knowledge Boost（TK-Boost），通过挖掘 NL2SQL 代理在执行数据库查询时出现的逻辑错误，生成可重用的纠正性知识，并在查询生成过程中对代理给出细粒度的反馈，从而提升其执行准确率。

**💡 创新点**

创新点在于：① 将错误诊断与知识生成分离，先通过迭代修正错误 SQL 并提取最小编辑步骤；② 生成的知识与其适用条件（SQL 关键字、表、列、数据类型）一起存储，形成结构化的知识库；③ 在推理阶段基于代理生成的 SQL 逐子查询检索相关知识，并将其转化为针对性的反馈，避免了传统基于问题相似度检索导致的无关信息注入。

**🔧 技术方法**

使用的技术包括：大语言模型（如 GPT‑4.1、Gemini‑2.5‑Flash、Qwen‑3‑Next‑80B 等）用于知识生成、错误分析与反馈生成；SQL 解析器提取查询特征；基于可应用条件的检索机制；以及 ReAct/REFORCE 等多步骤 NL2SQL 代理框架来执行与修正。

**📊 数据集**

主要使用公开基准数据集 Spider‑2（SQLite、BigQuery、Snowflake 三子集）和 BIRD‑mini‑dev，对每个子集采用 25% 的查询构造经验集，剩余 75% 作为测试。

**📈 对比分析**

与基线（无增强、记忆增强、NL‑SQL 对应对记忆、Naïve Knowledge）比较，TK‑Boost 在 Spider‑2 上提升最高 16.9%（BIRD 13.7%），显著优于记忆增强（≈4–5%）和 Naïve Knowledge（≈4–6%）。同时，鲁棒性高（回退率仅 1.2%），总体提升大幅领先。该框架对不同模型和代理架构（如 Agentar‑Scale‑SQL‑32B、ReFORCE）均可兼容并提升。

**⚠️ 局限性**

局限性包括：① 需要额外的 LLM 调用，导致推理延迟约 7–10 秒；② 生成的知识需手工或自动化精细化，仍可能出现错误或不完整；③ 依赖经验集的规模，若经验不足可能难以覆盖多样化误差；④ 对极其复杂的 SQL 子任务，子查询分层检索与反馈仍可能不足，导致误差未完全修正。

---

## 675. OmniVTON++: Training-Free Universal Virtual Try-On with Principal Pose Guidance

**arXiv ID:** 2602.14552 | [PDF](https://arxiv.org/pdf/2602.14552v1)

**作者:** Zhaotong Yang `[一作]` (Ocean University of China), Jian Yang `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 OmniVTON++，一种训练‑free 的通用虚拟试衣框架，支持单/多件服装、单/多人以及动漫角色的试衣；

**💡 创新点**

创新点包括：① Structured Garment Morphing（SGM）实现无训练的精准服装对齐；② Principal Pose Guidance（PPG）在扩散采样过程中实现逐步姿态控制；③ Continuous Boundary Stitching（CBS）/CBS‑DiT 解决边界连续性问题；

**🔧 技术方法**

主要技术为扩散生成模型（Stable Diffusion v2.0 与 FLUX），结合人形解析、DensePose、关键点检测、伪人生成、主成分姿态引导和双流交互注意机制；

**📊 数据集**

使用 VITON‑HD、DressCode、DeepFashion2（包括 Shop‑to‑Street、Model‑to‑Model 等场景）以及公开的多场景/多模态数据；

**📈 对比分析**

与多种基线（GP‑VTON、CAT‑DM、D^4‑VTON、IDM‑VTON、PWS、PastaGAN++、StreetTryOn、Any2AnyTryOn、ControlNet 等）在 FID、SSIM、LPIPS 指标上进行对比，OmniVTON++ 在跨数据集、跨服装类型、跨场景、跨扩散基准上均名列前茅；

**⚠️ 局限性**

局限性包括：依赖外部姿态/解析模型，关键点或 DensePose 失真会导致服装变形错误；以及对细节配饰的保留不佳，需要进一步改进掩码与分割策略。

---

## 676. Zero-Shot Instruction Following in RL via Structured LTL Representations

**arXiv ID:** 2602.14344 | [PDF](https://arxiv.org/pdf/2602.14344v1)

**作者:** Mathias Jackermeier `[一作]` (University of Oxford), Alessandro Abate `[通讯]` (University of Oxford)

**通讯引用:** 6069 | [OpenAlex ID](https://openalex.org/A5091718585)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

构建了一个能够在零样本情境下执行任意 LTL 指令的强化学习框架。

**💡 创新点**

创新点在于将 Büchi 自动机转化为序列化的布尔公式，并用层次化 DNF 编码和时间注意力机制对任务结构进行显式建模，从而实现更好的组合推理与非贪婪决策。

**🔧 技术方法**

主要技术包括：Büchi 自动机构造 → 语义化布尔公式 → DeepSets 层次化 DNF 编码 → 单头缩放点积注意力 + ALiBi 加权，最后使用 PPO 训练带有 ε‑动作的策略。

**📊 数据集**

使用了两个自定义环境：ZoneEnv（高维机器人导航）和 Warehouse（需要复杂逻辑推理的搬运任务），并在这两个环境中进行实验。

**📈 对比分析**

与 LTL2Action 与 DeepLTL 进行对比；在 ZoneEnv 的有限期望任务中成功率均超过 95%，在 Warehouse 的复杂组合任务中实现了显著的性能提升（比 DeepLTL 高 10%–20%），且样本效率更好，平均完成步骤更少。

**⚠️ 局限性**

局限性包括：依赖已知的观察→原子命题标签函数，未考虑标签函数学习；布尔公式生成与自动机构造的计算开销较大；在极大原子命题集合时仍可能面临组合爆炸。

---

## 677. ARC: Compiling Hundreds of Requirement Scenarios into A Runnable Web System

**arXiv ID:** 2602.13723 | [PDF](https://arxiv.org/pdf/2602.13723v1)

**作者:** Weiyu Kong `[一作]` (Shanghai Jiao Tong University), Haoyu Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 11488 | [OpenAlex ID](https://openalex.org/A5100763973)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ARC框架，将多模态需求文档编译为可运行的Web系统，并自动生成软件架构、测试用例与追踪记录。

**💡 创新点**

创新点在于将需求视为图形化DSL，采用双向测试驱动的编译循环，保证需求与实现的一致性，并提供完整的可追溯性。

**🔧 技术方法**

使用大型语言模型（Gemini Pro 3）实现需求解析、接口合成、测试生成与代码实现，并结合深度优先搜索、事件驱动接口定义等技术。

**📊 数据集**

使用六个基准Web系统（包括票务、资源管理等），每个系统基于50‑200个多模态需求场景的DSL描述。

**📈 对比分析**

与MetaGPT、OpenHands等基线相比，ARC在GUI测试通过率平均提升约50.6%，最高可提升108.1%；用户研究显示新手可在5.6小时内完成需求编写并生成约10K行可维护代码。

**⚠️ 局限性**

局限性包括主要聚焦功能需求，忽略安全、性能等非功能属性；测试覆盖仍有限，可能漏检边缘情况；需手工编写DSL，虽然提供可视化工具但仍较耗时。

---

## 678. FireRed-Image-Edit-1.0 Techinical Report

**arXiv ID:** 2602.13344 | [PDF](https://arxiv.org/pdf/2602.13344v1)

**作者:** Super Intelligence Team `[一作]` (Xiaohongshu Inc), Ziyuan Guo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出FireRed-Image-Edit，一款基于扩散变压器的指令驱动图像编辑框架，并构建了大规模训练语料与统一评测基准。

**💡 创新点**

系统性优化数据采集、两阶段过滤、指令对齐、动态桶采样、Stochastic Instruction Alignment、非对称梯度优化、DiffusionNFT、身份一致性损失以及统一评测基准REDEdit-Bench，突破规模与效率的权衡。

**🔧 技术方法**

采用多模态双流MM‑DiT、Multi‑Condition Aware Bucket Sampler、Stochastic Instruction Alignment、分布式分层时间步采样、Logit‑Normal加权、EMA、DiffusionNFT、OCR与VLM评估等技术。

**📊 数据集**

采集约1.6B样本，包括900M文本‑图像对与700M图像编辑对，最终筛选100M高质量样本，并结合公开数据集如OmniEdit、UnicEdit‑10M等。

**📈 对比分析**

在REDEdit‑Bench、ImgEdit、GEdit等公开基准上与多款开源与闭源系统对比，FireRed在指令遵循、保真度、真实感与美学等多维度均居前列，部分指标甚至逼近或优于商业模型。

**⚠️ 局限性**

受限于训练成本与硬件需求，且对极端稀有指令与长文本、复杂布局等长尾场景的泛化略逊，后续需进一步提升对长尾任务的适应性与算力友好性。

---

## 679. AdaCorrection: Adaptive Offset Cache Correction for Accurate Diffusion Transformers

**arXiv ID:** 2602.13357 | [PDF](https://arxiv.org/pdf/2602.13357v1)

**作者:** Dong Liu `[一作]` (Yale University), Ying Nian Wu `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无训练、可即插即用的 AdaCorrection 框架，用于在 Diffusion Transformers（DiT）推理期间自适应校正缓存的偏移，从而在保持高生成质量的同时显著加速推理。

**💡 创新点**

创新点：① 在每层、每个时间步动态估计空间-时间偏移并生成连续的校正权重，而非采用固定或二元缓存决策；② 通过轻量级的偏移估计模块（OEM）和校正模块（ACM）实现实时校正；③ 证明了误差界限并分析了质量-效率 Pareto 权衡。

**🔧 技术方法**

使用了空间-时间偏移量、梯度分布、Lipschitz 连续性分析，以及基于阈值的线性映射来融合缓存与新计算；实验中采用了 DiT-XL/2、DiT-L/2、DiT-B/2、DiT-S/2 等 Transformer 架构。

**📊 数据集**

在 ImageNet-256、FFHQ、LSUN-Church 等图像数据集，以及针对视频的 32/64 帧长序列进行评估；使用 FID、t-FID、PSNR、SSIM、FPS、缓存命中率（HR）、显存和延迟等指标。

**📈 对比分析**

与现有缓存加速方法（如 TeaCache、AdaCache、LazyDiT、FastCache 等）以及全重算 baseline 进行对比。AdaCorrection 在保持相同或更高缓存命中率的同时，将 FID 降至 4.37（vs 4.42 全重算）并维持 15–16 FPS，显著提升了质量-效率平衡。

**⚠️ 局限性**

局限性：对超参数 γ、λ 的敏感性需要根据具体模型和任务手动调优；仅在现有 Transformer 结构上验证，尚未探究在更大规模模型或更复杂推理策略（如多步抽样）下的鲁棒性；对于极端运动或剧烈场景变化时的偏移估计精度仍有待提升。

---

## 680. A Study on Multi-Class Online Fuzzy Classifiers for Dynamic Environments

**arXiv ID:** 2602.14375 | [PDF](https://arxiv.org/pdf/2602.14375v1)

**作者:** Kensuke Ajimoto `[一作]`, Tomoharu Nakashima `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种多类别在线模糊分类器，结合 OvR 和 OvO 方案，能够在动态环境下进行在线学习。

**💡 创新点**

创新点在于将二分类在线模糊分类器扩展到多类，通过使用 Passive‑Aggressive（PA）学习更新规则、引入 Don’t Care 以控制规则数目，并在动态数据中展示可解释性。

**🔧 技术方法**

使用模糊 if‑then 规则、三角形和矩形隶属函数、Passive‑Aggressive 在线学习、OvR/OvO 组合策略、在线学习算法及并行化预研。

**📊 数据集**

实验数据集包括公开基准：Speaker accent recognition、Crabs、Glass identification、Iris、Page blocks classification、Seeds、Wine；以及合成二维三类 Gaussian 动态数据。

**📈 对比分析**

与 PA（OvR/OvO）和 Delta 学习（OvR/OvO）比较；多类 PA 与模糊分类器在准确率上相当或略优于 Delta，静态数据上准确率可达 94–97%，动态数据约 97.5%；但模糊分类器的计算时间较长。

**⚠️ 局限性**

局限性包括规则数增长导致计算量大、在动态环境下旧规则仍影响解释性（需加入忘记机制），并行化实现尚未完成。

---

## 681. A Q-Learning Approach for Dynamic Resource Management in Three-Tier Vehicular Fog Computing

**arXiv ID:** 2602.14390 | [PDF](https://arxiv.org/pdf/2602.14390v1)

**作者:** Bahar Mojtabaei Ranani `[一作]` (Razi University), Sajad Ahmadian `[通讯]` (Kermanshah University of Technology)

**通讯引用:** 2645 | [OpenAlex ID](https://openalex.org/A5079645370)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于 Q‑Learning 的三层车联网雾计算资源预测与调度方法，能够动态分配 CPU、内存和带宽等资源。

**💡 创新点**

创新点在于将 Q‑Learning 迁移到三层车雾架构中，并设计了包含交通状态、资源利用率和应用需求的多维状态空间、可调的奖励函数和自适应动作空间，提升了资源利用率与延迟表现。

**🔧 技术方法**

核心技术为强化学习中的 Q‑Learning 算法，结合状态-动作表学习、ε‑greedy 策略和多目标奖励函数；实现基于 Python/Matlab 的仿真框架。

**📊 数据集**

实验使用 Didi GAIA 开源车载轨迹数据（成都青羊区 3×3 km² 区域的 4 个交通场景）以及仿真参数集（任务大小、计算需求、时延阈值、V2I 带宽等）。

**📈 对比分析**

与传统 RR、FCFS、WFQ、基于 Lagrange 的调度算法对比，Q‑Learning 方法在平均服务率、累计奖励、平均实现潜能、平均服务时间和平均处理时间等指标上均明显优于基线，表现出更好的鲁棒性和效率。

**⚠️ 局限性**

主要限制包括：对高质量实时数据的依赖、Q‑Learning 训练收敛慢、在极端网络拥塞下的适应性有限、车载硬件资源受限，以及对三层架构稳定性的假设，可能影响在大规模真实环境中的可扩展性。

---

## 682. KorMedMCQA-V: A Multimodal Benchmark for Evaluating Vision-Language Models on the Korean Medical Licensing Examination

**arXiv ID:** 2602.13650 | [PDF](https://arxiv.org/pdf/2602.13650v1)

**作者:** Byungjin Choi `[一作]` (Ajou University), Edward Choi `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `109c2b71-d051-425c-831f-0c544c24280d` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

构建并发布了KorMedMCQA‑V——一个包含1,534道多模态医师执业考试题目、2,043幅医学影像的多模态多选题评测基准。

**💡 创新点**

首次为韩国医学执业考试提供完整的图像题库，并通过统一零样本评测框架，对不同模态、模型规模、推理方式等因素进行系统拆解分析。

**🔧 技术方法**

采用零样本提示、图像预处理、VLM推理以及精确匹配评估，聚合来自 50+ 视觉‑语言模型的输出进行比较。

**📊 数据集**

数据集来源于 2012–2023 年韩国医师执业考试官方题库，涵盖 X‑ray、CT、ECG、US、内镜、MRI、血液涂片等九大影像模态。

**📈 对比分析**

在统一评测协议下，最佳专有模型 Gemini‑3.0‑Pro 取得 96.9% ；最佳开源模型 Qwen3‑VL‑32B‑Thinking 83.7%；最佳韩语专用模型 VARCO‑VISION‑2.0‑14B 仅 43.2%；多模态、跨图像推理以及各模态难度差异均被量化。

**⚠️ 局限性**

局限性包括：仅覆盖韩医师考试；新近年份更新不足；仅评估零样本性能，未探究少样本或微调效果；少数模态样本量有限。

---

## 683. Probabilistic approximate optimization using single-photon avalanche diode arrays

**arXiv ID:** 2602.13943 | [PDF](https://arxiv.org/pdf/2602.13943v1)

**作者:** Ziyad Alsawidan `[一作]` (Carnegie Mellon University), Tathagata Srimani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1212 | [OpenAlex ID](https://openalex.org/A5011011809)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文实现了Probabilistic Approximate Optimization Algorithm（PAOA）在64×64单光子雪崩二极管（pgSPAD）阵列上的实验，首次将变分学习应用于内在噪声的纳米设备；

**💡 创新点**

创新点在于利用PAOA的闭环学习将设备异质性（非对称Gompertz型激活函数）内嵌进可变参数中，避免了传统手工标定与调度，显著提升了硬件可扩展性；

**🔧 技术方法**

技术包括：pgSPAD的周边门控调节暗计数率、使用激活函数拟合Gompertz曲线、基于COBYLA的无梯度变分优化、以及与CPU模拟的对比验证；

**📊 数据集**

数据集为26自旋Sherrington–Kirkpatrick（SK）随机 Ising 问题实例（共60个随机实例，30个训练、30个测试），每个实例有正态分布的耦合系数；

**📈 对比分析**

与传统tanh激活的软件实现比较，PAOA在p≤17层时得到相同或更优的逼近比；在硬件推理时，pgSPAD的50次采样结果与CPU 10^6次采样几乎相同，显示出对设备变异的鲁棒性；

**⚠️ 局限性**

局限性包括：目前仅在中等深度（p≤17）下验证，对更深层次的泛化尚需进一步研究；设备温漂与功耗问题仍未在大规模阵列中系统评估；

---

## 684. LongCLI-Bench: A Preliminary Benchmark and Study for Long-horizon Agentic Programming in Command-Line Interfaces

**arXiv ID:** 2602.14337 | [PDF](https://arxiv.org/pdf/2602.14337v1)

**作者:** Yukang Feng `[一作]` (Nankai University), Kaipeng Zhang `[通讯]` (Nankai University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LongCLI-Bench benchmark，评估 AI 助手在长周期 CLI 环境下的规划与执行能力

**💡 创新点**

构造 20 条长周期、真实场景任务，采用双集测试与逐步评分；消除 GitHub 数据污染；引入人工计划与交互式指导实验

**🔧 技术方法**

使用大型语言模型（Codex、Claude Code 等）结合 OpenHands 框架；构建 Docker 化隔离环境与自定义测试套件

**📊 数据集**

任务来源于 958 门 CS 课程作业和 50 条真实工作流程，手工生成需求文档、解决方案和测试用例

**📈 对比分析**

与多种现有 benchmark（HumanEval、MBPP、SWE-bench、Terminal-Bench）比较；通过 Pass Rate、Pass@3、Step Score 评价；结果显示大多数模型 Pass Rate <20%，说明长周期任务仍具挑战性；人机协作显著提升性能

**⚠️ 局限性**

任务创建耗时高（约 40 小时/任务）；任务量有限；Step Score 无法全面衡量代码质量、效率等方面

---

## 685. AXE: An Agentic eXploit Engine for Confirming Zero-Day Vulnerability Reports

**arXiv ID:** 2602.14345 | [PDF](https://arxiv.org/pdf/2602.14345v1)

**作者:** Amirali Sajadi `[一作]` (Drexel University), Preetha Chatterjee `[通讯]` (Drexel University)

**通讯引用:** 342 | [OpenAlex ID](https://openalex.org/A5049106181)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了AXE——一种基于多代理的灰盒自动化漏洞验证框架，利用CWE标签和源代码位置自动生成可验证的PoC；

**💡 创新点**

创新点在于将高层规划、源代码分析、执行反馈拆分为专门代理，实现迭代规划与反馈循环，显著提升灰盒验证效率；

**🔧 技术方法**

采用LLM驱动的多代理架构（Strategist、Explorer、Exploiter、PoC Gen），结合静态源代码分析、端点发现、SQL注入工具、工具化请求生成等技术；

**📊 数据集**

使用CVE‑Bench数据集（40个真实Web漏洞），并在此基础上扩展加入源代码与元数据；

**📈 对比分析**

与传统黑盒工具（T‑Agent、AutoGPT）及单代理灰盒基线对比，AXE在Success@5上达到30%（黑盒为10%），AvgTCA为1.67、Success Efficiency 0.28，显示显著性能提升；

**⚠️ 局限性**

局限性包括：规划阶段误判导致高失败率、对非Web/闭源软件适用性不足、验证oracles有限、对特定预条件依赖较高，且失败未必说明不可利用。

---

## 686. Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens

**arXiv ID:** 2602.13517 | [PDF](https://arxiv.org/pdf/2602.13517v1)

**作者:** Wei-Lin Chen `[一作]` (University of Virginia), Yu Meng `[通讯]` (Google)

**通讯引用:** 41915 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证一种新的推理努力度量指标——深度思考比（DTR），并基于该指标实现高效推理规模化

**💡 创新点**

创新点在于通过追踪每个 token 在 Transformer 各层中预测分布的收敛深度，捕捉内部计算过程，而非仅靠输出长度或置信度，显著提高了与任务准确率的相关性

**🔧 技术方法**

使用中间层隐藏状态投射到词表空间、计算 Jensen–Shannon 距离判断分布收敛、定义深度思考标记，结合 DTR 进行样本筛选与投票（Think@n）

**📊 数据集**

在四个高难度推理基准上进行评估：AIME 2024/2025、HMMT 2025 和 GPQA，并在 GPT‑OSS、DeepSeek‑R1‑70B、Qwen3‑30B‑Thinking 等多模型上实验

**📈 对比分析**

与传统基于 token 长度、负对数概率、负困惑度、负熵和 Self‑Certainty 的基线相比，DTR 的 Pearson 相关系数平均提升至 0.683；Think@n 在保持或超过 Self‑Consistency 的准确率同时将推理成本降至约一半

**⚠️ 局限性**

局限性包括需访问内部层状态且对阈值/深度比例的超参数敏感；对非自回归或不支持中间层输出的模型适用性有限，且对极端长序列的稳定性仍待进一步研究

---

## 687. A Latency-Aware Framework for Visuomotor Policy Learning on Industrial Robots

**arXiv ID:** 2602.14255 | [PDF](https://arxiv.org/pdf/2602.14255v1)

**作者:** Daniel Ruan `[一作]` (Princeton University), Arash Adel `[通讯]` (Princeton University)

**通讯引用:** 182 | [OpenAlex ID](https://openalex.org/A5053948992)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了一个系统级的时延感知框架，用于在工业机器人手臂上部署和评估基于视觉的动作策略，框架包括多模态传感器校准、时间同步、统一通信管道以及VR遥控演示收集，并设计了基于时间可行性调度的执行策略；

**💡 创新点**

创新点在于显式考虑并补偿观察–执行间隙，利用时延校准的多模态感知与时间对齐的动作调度，实现在工业平台上实现异步推理与执行而不改动策略结构，解决了工业机器人高层控制接口导致的高时延问题；

**🔧 技术方法**

所用技术包括ABB工业机械臂的EGM高层控制接口、ROS 2中间件、Python/ROS 2驱动的多模态传感（视觉、力/扭矩、关节位姿）、HOG视觉特征、k‑Nearest‑Neighbors决策策略、时延校准与插值同步、VR手柄遥控演示收集与离线数据处理；

**📊 数据集**

数据集为20条专家演示，使用VR遥控方式收集，演示仅包含姿态、力/扭矩与HOG视觉特征，作为k‑NN策略的训练与评估基础；

**📈 对比分析**

通过将时延感知执行策略与阻塞式执行（blocking）和无时延补偿的异步执行（naive asynchronous）进行对比，使用任务时长、空闲比例、接触力、力与运动平滑度等指标，结果显示时延感知策略在不同推理时延（100 ms、300 ms、500 ms）下能保持与专家演示相近的任务进度，显著降低空闲时间，提升力与运动平滑度，避免了阻塞式的过长停顿和异步式的力峰值与振荡；

**⚠️ 局限性**

局限性包括仅在单一的木材角接合装配任务上验证，未涉及更复杂的多接触或双臂场景；实验仅使用k‑NN基准策略，未评估大规模视觉语言动作模型；在其他工业平台上需重新校准时延和接口；未来需扩展至更多传感模态与更大规模的学习模型。

---

## 688. A Geometric Analysis of Small-sized Language Model Hallucinations

**arXiv ID:** 2602.14778 | [PDF](https://arxiv.org/pdf/2602.14778v1)

**作者:** Emanuele Ricco `[一作]` (King Abdullah University of Science and Technology), Roberto Di Pietro `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对小规模LLM在同一提示下多次生成的回复进行嵌入空间几何分析，区分事实正确与幻觉答案。

**💡 创新点**

提出幻觉是嵌入空间中语义不一致的几何现象，而非单纯知识缺失；利用 Fisher 判别方向实现标签传播，F1>90%。

**🔧 技术方法**

使用 SBERT 句子编码、欧氏距离、Wasserstein 距离、Fisher 判别分析（FDA）以及基于分布一致性的标签传播方法。

**📊 数据集**

构建 200 个提示、10 种小型 LLM（≤32B）各生成 150 条回复的 150 万条数据，并用 Claude 4.5 Sonnet 自动标注；用 1,000 条人工标注样本检验标注质量。

**📈 对比分析**

与原始嵌入空间相比，FDA 投影提升可分离度约 7.3 倍；标签传播在仅 30–60 条已标注样本下即可实现 87% 准确率、90% F1；不同模型规模表现一致。

**⚠️ 局限性**

研究仅聚焦时间敏感事实查询，未验证跨提示或跨任务的判别方向一致性；嵌入空间固定，可能不适用于所有幻觉类型；依赖外部大型 LLM 的标注，若其标注错误将影响结果。

---

## 689. Spanning tree congestion of proper interval graphs

**arXiv ID:** 2602.13756 | [PDF](https://arxiv.org/pdf/2602.13756v1)

**作者:** Yota Otachi `[一作]` (Nagoya University), Yota Otachi `[通讯]` (Nagoya University)

**通讯引用:** 1015 | [OpenAlex ID](https://openalex.org/A5033878501)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

证明在具有线性团宽度不超过4的完全区间图上，求最小生成树拥塞是NP-完全的。

**💡 创新点**

将3-Partition问题归约到该问题，构造的图是完全区间图且团宽度≤4，从而填补了该问题在此图类上的复杂度空白。

**🔧 技术方法**

使用图论构造、组合归约以及辅助引理分析生成树拥塞的性质，证明关键边的拥塞上界。

**📊 数据集**

未使用实验数据集，全部为理论证明。

**📈 对比分析**

未做实验比较，只给出多项式时间下的复杂度结果。

**⚠️ 局限性**

结果仅适用于特定图类和参数，未涉及实际实例或近似算法的性能评估。

---

## 690. MC$^2$Mark: Distortion-Free Multi-Bit Watermarking for Long Messages

**arXiv ID:** 2602.14030 | [PDF](https://arxiv.org/pdf/2602.14030v1)

**作者:** Xuehao Cui `[一作]` (University of Maryland), Heng Huang `[通讯]` (University of Maryland)

**通讯引用:** 24796 | [OpenAlex ID](https://openalex.org/A5060016795)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种无失真、多比特水印框架（MC^2Mark），能够在生成文本中嵌入长达512位的消息，并在保持文本质量的同时实现高可检测性和鲁棒性。

**💡 创新点**

创新点包括：① 多通道彩色重加权（MCCR）通过动态缩放词表子集实现信息编码；② 多层序列重加权（MLSR）在多层生成过程中逐步增强水印信号；③ 基于证据累积的检测器能够在不暴露原始消息的情况下准确恢复长消息。

**🔧 技术方法**

技术实现：利用无失真重加权技术（保持期望分布不变），分段随机掩码、词表分区以及多层重加权；检测时使用逐词、逐层的统计证据累计并归一化后做位判决。

**📊 数据集**

实验数据集涵盖多种文本生成任务：book_report、mmw_story、fake_news、dolly_cw、longform_qa、finance_qa、c4_subset；评估机器翻译与摘要的质量时使用MBart和BART；所有实验基于Qwen2.5-3B-Instruct模型。

**📈 对比分析**

与MPAC（δ=1.0/1.5）和BiMark对比，MC^2Mark在所有消息长度（16–512位）和数据集上均取得最高检测准确率；短消息几乎完美识别，长消息保持90%+准确率，甚至比第二好方法提升近30%；在随机词替换和Dipper改写攻击下，鲁棒性也明显优于基线。

**⚠️ 局限性**

局限性：① 计算复杂度随着消息长度呈指数级增长（O(2^n/√n)），虽然通过分段降低但仍受限；② 主要验证在中型模型（3B）上，超大模型或更长文本的表现未知；③ 需要安全存储和同步密钥，若密钥泄露或同步失误会导致信息丢失。

---

## 691. Position Encoding with Random Float Sampling Enhances Length Generalization of Transformers

**arXiv ID:** 2602.14050 | [PDF](https://arxiv.org/pdf/2602.14050v1)

**作者:** Atsushi Shimizu `[一作]` (Daiwa Securities), Yutaka Matsuo `[通讯]` (University of Tokyo)

**通讯引用:** 13976 | [OpenAlex ID](https://openalex.org/A5090592819)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出随机浮点采样（Random Float Sampling，RFS）作为位置编码策略，以提升Transformer在未见长度上的泛化能力。

**💡 创新点**

创新点在于用连续随机浮点数采样位置索引，消除对离散整数索引的依赖，避免训练与推理长度不一致导致的OOD问题。

**🔧 技术方法**

将RFS与绝对正弦编码、RoPE、ALiBi等现有PE无缝结合，并在Transformer模型上进行训练。

**📊 数据集**

使用OpenWebText、copy、reverse、addition、sort、summation、SCAN等序列任务以及HellaSwag、RACE、ARC-Easy/Challenge、OpenBookQA、WinoGrande、BoolQ等commonsense推理数据集。

**📈 对比分析**

与传统PE、无PE、位置插值、随机整数采样等方法对比，RFS在长度泛化任务上获得最高准确率，零样本推理在OOD集上平均提升至35.78%与ID平均相当。

**⚠️ 局限性**

局限性包括仅在约1亿参数的模型上实验，未验证在大型LLM上的效果；当输入长度过长时性能仍会下降，需结合其他技术进一步提升。

---

## 692. Assessing the Case for Africa-Centric AI Safety Evaluations

**arXiv ID:** 2602.13757 | [PDF](https://arxiv.org/pdf/2602.13757v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 693. Annotation-Efficient Vision-Language Model Adaptation to the Polish Language Using the LLaVA Framework

**arXiv ID:** 2602.14073 | [PDF](https://arxiv.org/pdf/2602.14073v1)

**作者:** Grzegorz Statkiewicz `[一作]` (NASK National Research Institute), Wojciech Kusa `[通讯]` (NASK National Research Institute)

**通讯引用:** 11797 | [OpenAlex ID](https://openalex.org/A5055132321)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了波兰语视觉语言模型，利用大规模自动翻译和轻量过滤生成训练数据，采用LLaVA-Next架构并结合波兰本土LLM骨干；

**💡 创新点**

创新点在于仅通过自动翻译和最小人工筛选即可为低资源语言创建高质量VLM，并公开模型与Polish版MMBench评测数据；

**🔧 技术方法**

使用技术包括LLaVA-Next、SigLIP2视觉编码器、LoRA参数高效微调、Tower+72B自动翻译、COMET质量评估、合成OCR数据等；

**📊 数据集**

使用数据集包括Allava‑Instruct‑LAION‑4V、LLaVA‑158K、Q‑Instruct、LVIS‑Instruct4V、A‑OKVQA、SynthDoG‑PL/EN、WIT、TallyQA等共2M样本，并通过翻译得到Polish MMBench dev；

**📈 对比分析**

通过MMBench（Polish/English）与五个同等规模开源VLM（LLaVA‑1.6‑Mistral‑7B、LLaVA‑1.6‑Vicuna‑13B、Qwen2.5‑VL‑7B、PaliGemma2‑10B、Pixtral‑12B）比较，Polish版模型在MMBench提升约+9.5%，在XM3600评测中在语言正确性上优于大多数基线，虽略逊于Pixtral，但与强大模型相当；

**⚠️ 局限性**

局限性包括训练数据对自动翻译的高度依赖导致潜在翻译噪声，缺乏系统阈值ablation，未提供针对OCR或本土知识的专门评测，文化覆盖有限，评测方法不完全对齐，以及关键设计选择（如SigLIP vs CLIP、训练比例、LoRA vs 全部微调）未做深入ablations。

---

## 694. An end-to-end agentic pipeline for smart contract translation and quality evaluation

**arXiv ID:** 2602.13808 | [PDF](https://arxiv.org/pdf/2602.13808v1)

**作者:** Abhinav Goel `[一作]` (Columbia University), Alfio Gliozzo `[通讯]` (IBM T.J. Watson Research Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个端到端的多阶段智能代理框架，用自然语言规范自动生成、验证并评估Solidity智能合约。

**💡 创新点**

创新点在于将CrewAI式代理团队、五维质量评估、与专家实现的并行对比以及自动强化循环结合，形成可复现的评估基准。

**🔧 技术方法**

使用的技术包括大型语言模型（如GPT‑4o‑mini）、多代理协同、自动化编译与安全审计、FSM形式化验证，以及迭代修正循环。

**📊 数据集**

数据集为FSM‑SCG，包含约22,000条需求–FSM–代码三元组，并在实验中生成9,000条合约进行评估。

**📈 对比分析**

通过与专家实现的对比，生成合约平均综合得分81.5（B级），编译通过率86.5%，比基准实现高约8分，且安全缺陷降低约71%。

**⚠️ 局限性**

局限包括对LLM产生的幻觉不鲁棒、缺乏燃气优化指标、对高复杂度合约的质量下降、以及对真实世界跨合约依赖与监管合规的覆盖不足。

---

## 695. Differentially private graph coloring

**arXiv ID:** 2602.13460 | [PDF](https://arxiv.org/pdf/2602.13460v1)

**作者:** Michael Xie `[一作]` (University of Maryland), Aravind Srinivasan `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了两种基于边差分隐私的缺陷图着色算法，利用指数机制对颜色进行重采样并通过噪声阈值控制缺陷，从而在任意图上实现 O(log n) 的缺陷度。

**💡 创新点**

创新点在于：①首次将指数机制与迭代/受限重采样结合，直接控制每个顶点的缺陷；②提出了针对任意图的噪声阈值重采样策略，克服了传统方法只能处理 d‑inductive 图的限制；③给出了可调的隐私-缺陷度折衷方案。

**🔧 技术方法**

主要技术手段包括：边差分隐私模型、拉普拉斯机制估计最大度数、指数机制进行局部颜色重采样、Chernoff 与泊松尾部概率分析、以及对 d‑inductive 图的递归采样顺序。

**📊 数据集**

实验使用了 SNAP 公开网络数据集（如 Twitch‑ENGB、Ca‑HepTh、Oregon1 等）以及通过 Erdős–Rényi G(n,p) 和 Barabási–Albert 模型生成的合成图。

**📈 对比分析**

与非私有贪心着色和 CRSV 基线算法对比，平均缺陷度和最大缺陷度均低于 CRSV，且在大多数图上与非私有贪心算法相当；实验表明隐私参数 ϵ 越大，缺陷度越低，展示了良好的隐私-缺陷折衷。

**⚠️ 局限性**

局限性包括：①对 d‑inductive 图的理论分析依赖于特定顶点排序，难以直接推广到任意图；②在低隐私预算（ϵ 较小）下，随机初始化的简单方法有时能获得更低的缺陷度；③算法在极稠密图（如 Ca‑GrQc）上最大缺陷度仍相对较高。

---

## 696. Open Rubric System: Scaling Reinforcement Learning with Pairwise Adaptive Rubric

**arXiv ID:** 2602.14069 | [PDF](https://arxiv.org/pdf/2602.14069v1)

**作者:** Ruipeng Jia `[一作]` (Alibaba), Guanjun Jiang `[通讯]` (Alibaba)

**通讯引用:** 421 | [OpenAlex ID](https://openalex.org/A5004378463)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于元规则（Meta-Rubric）和适配式差分评判的开放式Rubric系统（OpenRS），用于生成可解释、可编辑的奖励信号并直接替代传统标量奖励模型，在强化学习中实现更高质量的对齐。

**💡 创新点**

核心创新在于：①将奖励视为由可检视的原则执行的推理过程，而非隐藏的标量函数；②通过差分先行（diff-first）方法即时生成适配式评判规则；③结合点对点评估与可验证的点评估，实现对开放式任务的高分辨率反馈；④引入异步强化学习（GRPO/Asym-GRPO）和进化式元规则微调，提升元规则的可迁移性与鲁棒性。

**🔧 技术方法**

使用了多模态语言模型（如 Qwen3-235B、gpt-oss-120b）作为评审者，Pairwise Adaptive Meta-Rubrics、Bootstrapped Relative Policy Optimization (BRPO)、Group Relative Policy Optimization (GRPO)、Asymmetric GRPO、遗传算法+Beam Search 等技术；同时结合了基于检查点的可验证指标（格式、单元测试、正确性检查）。

**📊 数据集**

在四大奖励建模基准上验证：RM-Bench、JudgeBench、RewardBench v2、PPE Preference（含中文子集）。此外，还使用公开评测（IFEval、IFScale、EQ-Bench、JudgeMark-v2、Chinese Simple QA）进行RL对比实验。

**📈 对比分析**

与主流标量奖励模型（Skywork-Reward-V2 等）相比，OpenRS 在平均分上提升 5.1 分（89.4 vs. 84.3），在四个基准中均取得最高分；在RL实验中相对标量奖励提升约 2–3 分（平均 71.3 vs. 68.4），且训练时吞吐量保持可接受。实验进一步展示了差分先行、域元规则和异步GRPO 等组件对性能的显著贡献。

**⚠️ 局限性**

局限性包括：①评审模型对算力需求高，需要本地部署和大规模 GPU 资源；②元规则微调过程仍依赖人类标注进行验证，可能存在人工成本；③在高度开放的多轮对话或工具使用场景下，元规则设计与适配性仍待进一步探索；④对抗性鲁棒性、提示注入防御和实时元规则漂移监测等方面尚未完全完善。

---

## 697. Joint Task Assistance Planning via Nested Branch and Bound (Extended Version)

**arXiv ID:** 2602.13932 | [PDF](https://arxiv.org/pdf/2602.13932v1)

**作者:** Omer Daube `[一作]` (Technion), Oren Salzman `[通讯]` (Technion)

**通讯引用:** 1315 | [OpenAlex ID](https://openalex.org/A5078885681)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并研究了联合任务协助规划（Joint Task Assistance Planning，JointTAP）问题，求解任务机器人与协助机器人在预先构建的路图上同步规划路径以最大化协助时长。

**💡 创新点**

创新点在于构建了层次化的嵌套分支限界框架，利用基于流的上界快速剪枝，同时开发了增量式OTP求解器，显著减少了重复计算。

**🔧 技术方法**

采用分支限界（Branch‑and‑Bound）、整数线性规划/最大流松弛、时间奖励动态规划以及增量更新技术实现。

**📊 数据集**

实验使用了两类仿真数据集：4-DOF平面机械臂和Crazyflie 2.1+四旋翼的路图，并在实验室实际小型无人机上验证。

**📈 对比分析**

与暴力DFS基线相比，算法在同一图规模下实现了两到三百倍的速度提升，且得到的奖励最优且一致；在更大图规模下仍可在一小时内完成求解。

**⚠️ 局限性**

局限性包括对大规模图仍呈指数增长，在线动态环境的适用性有限，以及对更复杂协助约束需进一步强化启发式或松弛模型。

---

## 698. Directional Concentration Uncertainty: A representational approach to uncertainty quantification for generative models

**arXiv ID:** 2602.13264 | [PDF](https://arxiv.org/pdf/2602.13264v1)

**作者:** Souradeep Chattopadhyay `[一作]` (Iowa State University), Karl Pazdernik `[通讯]` (Pacific Northwest National Laboratory)

**通讯引用:** 201 | [OpenAlex ID](https://openalex.org/A5041535920)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于生成文本嵌入向量的方向浓度不确定性（DCU）评估方法，用来量化大型语言模型在多次生成时答案的一致性。

**💡 创新点**

创新点在于：①完全摆脱了文本语义聚类和双向蕴含模型的依赖；②将生成的答案映射到单位球面上，并用von Mises‑Fisher（vMF）分布的浓度参数（κ）来度量答案的方向性分散，κ⁻¹即为不确定性分数；③方法可以无缝扩展到多模态（图像+文本）或任意模态，只需对应的编码器即可。

**🔧 技术方法**

使用技术包括：多次高温采样生成答案 → 通过预训练嵌入模型（e5‑large‑v2、CLIP）得到归一化向量 → 计算vMF分布的最大似然估计 → 逆浓度 κ⁻¹ 作为不确定性指标；与传统的语义熵（SE）对比评估。

**📊 数据集**

数据集涵盖：文本问答（SQuAD、TriviaQA、NQ‑Open）和多模态问答（ScienceQA）；在每个数据集上均采样 300/150 题，并对 10 条生成样本进行评估。

**📈 对比分析**

与 SE 进行 AUROC 与准确率对比：在文本问答任务中，DCU 的 AUROC 与 SE 相近甚至略优；在视觉问答任务中，DCU 明显优于 SE（例如 LLaVA‑1.5‑7B 的 AUROC 0.67 对比 SE 0.51），表明 DCU 更具通用性。

**⚠️ 局限性**

限制：仍需多次采样，采样成本高；对嵌入模型的质量高度依赖；在某些复杂任务（如 ScienceQA）整体 AUROC 仍偏低，表明仍有改进空间。

---

## 699. Scalable Multi-Robot Path Planning via Quadratic Unconstrained Binary Optimization

**arXiv ID:** 2602.14799 | [PDF](https://arxiv.org/pdf/2602.14799v1)

**作者:** Javier González Villasmil `[一作]` `[通讯]`, Javier González Villasmil

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于QUBO的多机器人路径规划框架，利用BFS预处理、时间窗口分解实现可扩展性。

**💡 创新点**

创新点在于将MAPF统一编码为QUBO、超过95%变量削减的BFS预处理、时间窗口策略以及自适应惩罚设计。

**🔧 技术方法**

使用Quadratic Unconstrained Binary Optimization、模拟退火、模拟QAOA以及D‑Wave仿真量子退火等量子启发式求解器。

**📊 数据集**

实验在5×5与10×10网格地图，最多四机器人，包含障碍物的场景上进行。

**📈 对比分析**

与经典A*+CBS优先规划对比，单机时经典更快，但多机时QUBO在路径长度上仅5.3%超出，规模随机器人数量线性增长。

**⚠️ 局限性**

主要局限是对BFS预处理高度依赖、交换碰撞不易编码、方案集中式单点求解且在当前硬件上未显著超越经典算法。

---

## 700. Counterfactual Fairness Evaluation of LLM-Based Contact Center Agent Quality Assurance System

**arXiv ID:** 2602.14970 | [PDF](https://arxiv.org/pdf/2602.14970v1)

**作者:** Kawin Mayilvaghanan `[一作]` (Observe.AI), Ayush Kumar `[通讯]` (Observe.AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对13维度的对照假设（身份、上下文、行为风格）进行量化，评估18款大型语言模型在3000条真实客服对话中的公平性；

**💡 创新点**

创新点在于提出完整的13维公平性分类法，并引入对照假设翻转率（CFR）与均方绝对得分差（MASD）两种指标，系统地量化LLM在高风险劳动力评估中的偏差；

**🔧 技术方法**

技术实现包括对对话进行语义保持的对照假设生成、LLM自动质量评估、统计CFR/MASD、建立鲁棒性基线以及公平性提示干预；

**📊 数据集**

使用的数据集为3000条真实客服记录（专有）和1200条ConvoGen生成的合成记录；

**📈 对比分析**

在对比方法上，作者在同一提示下对18款LLM计算CFR、MASD及答案准确率，发现模型规模与对齐程度越大公平性越好，但公平性与准确率并无必然关联；

**⚠️ 局限性**

局限性包括公平性分类法可能不完整，公平提示仅为初步干预，深层偏差仍需更先进技术解决，且真实数据集未公开导致可复现性受限。

---

## 701. Cognitive networks reconstruct mindsets about STEM subjects and educational contexts in almost 1000 high-schoolers, University students and LLM-based digital twins

**arXiv ID:** 2602.14749 | [PDF](https://arxiv.org/pdf/2602.14749v1)

**作者:** Francesco Gariboldi `[一作]` (University of Trento), Massimo Stella `[通讯]` (University of Trento)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建行为形态心智网络（BFMN）对高中文、大学生、早期科研专家以及使用GPT-oss模拟的“数字双胞胎”在STEM学科及教育情境中的语义与情感结构进行比较分析。

**💡 创新点**

首次将BFMN与大型语言模型生成的数字双胞胎结合，揭示了科学与其量化基础（数学、统计）之间的认知-情感不协调，并证明LLM在情感细节与经验具体化方面的局限。

**🔧 技术方法**

使用认知网络科学方法构建BFMN，利用情绪标签（valence）、情感光环（valence aura）、Jaccard相似度、情绪花瓣（Emotional Flower）以及共构度量等网络分析工具，同时采用GPT-oss 20B进行数字双胞胎模拟。

**📊 数据集**

数据来自约994名人类参与者（包含高中文、心理学本科、物理本科、科研专家等不同学段与焦虑水平）以及对应的GPT-oss模拟数据，采用自由联想任务、情绪评价任务及数学焦虑量表（MAS-IT）收集。

**📈 对比分析**

通过比较人类与GPT-oss网络在情绪光环、共构度、抽象度（concreteness）以及词汇重叠（Jaccard）上的差异，发现人类网络在数学与焦虑的语义重叠显著高于LLM，并且人类网络表现出更强的具体化特征；LLM虽然复制了整体情绪倾向，但在具体化与情感细节上表现逊色。

**⚠️ 局限性**

局限包括便利抽样导致样本不具代表性、跨研究时间点和问卷差异导致数据不一致、仅采用横断面设计无法推断因果关系、LLM缺乏经验基础导致情感和具体化的缺失，以及统计显著性阈值较宽（α=0.1）可能影响结果稳健性。

---

## 702. Adapting VACE for Real-Time Autoregressive Video Diffusion

**arXiv ID:** 2602.14381 | [PDF](https://arxiv.org/pdf/2602.14381v1)

**作者:** Ryan Fosdick `[一作]` `[通讯]` (Daydream), Ryan Fosdick (Daydream)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

将原本在扩散潜变量中拼接的参考帧改为通过并行的 Context Block 条件路径注入提示，从而使 VACE 控制功能能够在实时自回归视频生成管线中使用。

**💡 创新点**

创新点在于：① 将参考帧移出主 DiT 序列，单独处理后通过零初始化投影注入提示，保持了固定块大小和 KV 缓存；② 直接复用已有的 VACE Context Block 权重，无需额外训练；③ 证明该改动在 1.3B 与 14B 规模模型上通用。

**🔧 技术方法**

技术手段包括：自回归视频扩散模型、DiT 变压器、Context Block 生成提示、零初始化投影、RoPE 位置编码、3D VAE、KV 缓存管理、SageAttention / FlashAttention 后端。

**📊 数据集**

实验基于公开的 Wan2.1（LongLive、Krea Realtime Video 等）模型，未单独使用新的数据集，主要对比基线无 VACE 与加上不同控制（Depth、Inpainting、Extension）的性能。

**📈 对比分析**

对比方法：在 RTX 5090（1.3B）和 H100（14B）上测量每块平均延迟、FPS 与显存占用。结果显示 Depth/Inpainting 控制约增加 20–30% 延迟（FPS 从约 22 降至 17~13），显存额外占用仅 ~1.4 GB，且所有控制均能在实时帧率下运行。

**⚠️ 局限性**

局限性：① 参考‑to‑video（R2V）质量明显下降，因因果注意力限制；② 长时间生成（100+ 帧）可能出现时间一致性衰退；③ 部分控制（如姿态、光流）需要进一步调优；④ 第一次/最后一次帧扩展在小块大小下效果受限。

---

## 703. Image Generation with a Sphere Encoder

**arXiv ID:** 2602.15030 | [PDF](https://arxiv.org/pdf/2602.15030v1)

**作者:** Kaiyu Yue `[一作]` (Meta), Tom Goldstein `[通讯]` (University of Maryland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一种Sphere Encoder框架，利用自编码器将自然图像映射到均匀的球面潜在空间，并通过解码随机球面点实现单步或少步图像生成；

**💡 创新点**

主要创新点包括：①将潜在空间直接约束为球面并通过简单的RMS归一化实现均匀分布；②利用噪声注入与一致性损失，使模型在无需扩散步骤的前提下实现高质量图像生成；③在编码器/解码器之间进行少步迭代，可进一步提升图像质量；

**🔧 技术方法**

采用ViT+MLP-Mixer架构的编码器与解码器，配合RMSNorm、RoPE+正弦位置编码、AdaLN-Zero及CFG技术；训练时使用像素重建+感知损失、像素一致性损失和潜在一致性损失；通过噪声扰动实现球面化与均匀化；

**📊 数据集**

在CIFAR-10、ImageNet（256×256）、Animal-Faces、Oxford-Flowers等数据集上进行训练与评估；

**📈 对比分析**

与StyleGAN2、DDPM、ADM、SD-VAE等生成模型对比，使用FID、IS等指标。Sphere Encoder在1-4步内即可取得与多步扩散相当甚至更好的FID（如ImageNet 4步FID 4.02、IS 265.9），在CIFAR-10 1步gFID 18.68、IS 9.1，显示出极低推理成本的优势；

**⚠️ 局限性**

局限性包括：需要同时训练编码器和解码器，训练时两次前向传播；对小数据集易出现记忆化问题；在高分辨率或多模态文本-图像任务中性能仍不如最新扩散模型；生成边缘有轻微模糊，且整体未达到最先进水平。

---

## 704. FC-Vision: Real-Time Visibility-Aware Replanning for Occlusion-Free Aerial Target Structure Scanning in Unknown Environments

**arXiv ID:** 2602.13720 | [PDF](https://arxiv.org/pdf/2602.13720v1)

**作者:** Chen Feng `[一作]` (Hong Kong University of Science and Technology), Shaojie Shen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 16918 | [OpenAlex ID](https://openalex.org/A5001947944)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种实时可见性感知的无人机扫描路径重规划框架 FC‑Vision，能够在未知环境中主动避免目标结构被遮挡，同时保持原始扫描覆盖率和飞行效率。

**💡 创新点**

创新点包括：① 两级分解的重规划策略——先用混合采样与优化修复覆盖率不变的视角，再通过 Φ‑A* 在 5‑DoF 空间中搜索“视角无遮挡”段；② 在每个节点通过局部姿态校正和可见性缓存实现常数时间遮挡检测；③ 可无缝集成为现有扫描系统的插件层。

**🔧 技术方法**

采用的技术包括：混合采样与解析位置/姿态优化、贪心集覆盖与顺序约束优化、基于 voxel 的 3‑D 位置搜索与 5‑DoF 视角映射、可见性缓存与二分姿态校正、基于轨迹优化的最终时间最小化。

**📊 数据集**

使用的实验数据集：仿真场景（Kino Wall、Tunnel、East Church）和真实室内环境（Rectangular Area、Room），其中目标结构为墙面、走廊或教堂内部结构，加入随机未知障碍。

**📈 对比分析**

与仅基于碰撞自由的重规划 Baseline（Colli‑Free）相比，FC‑Vision 在三种仿真场景中覆盖率提升至 95–97%，遮挡率降至 0–1%，平均飞行时间提升 6–38%，综合任务质量指标 VaE 增幅多倍；在真实飞行中覆盖率提升至 97–98%，遮挡率降至 0%，飞行时间提升 20–30%，并保持重规划延迟在 20–30 ms 内。

**⚠️ 局限性**

主要局限是：① 对动态移动障碍的处理仍依赖离线更新，未完全实时；② 只针对静态目标结构，未考虑多目标或复杂几何；③ 在极端高动态或多障碍场景下，重规划仍可能出现路径拥堵。

---

## 705. Zero-Order Optimization for LLM Fine-Tuning via Learnable Direction Sampling

**arXiv ID:** 2602.13659 | [PDF](https://arxiv.org/pdf/2602.13659v1)

**作者:** Valery Parfenov `[一作]` (HSE University), Aleksandr Beznosikov `[通讯]` (Basic Research of Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种可学习方向采样的零阶优化框架，旨在通过学习采样分布的均值来提升LLM微调过程中的梯度方向对齐，从而显著降低零阶方法的方差并消除对维度的依赖。

**💡 创新点**

创新点在于：1) 将方向采样视为强化学习策略，直接优化采样均值以提高梯度方向对齐；2) 在非凸L-smooth设置下给出无维度依赖的收敛上界；3) 通过“贪婪”选择最优扰动方向与可调的采样分布学习相结合，实现对传统零阶优化器的无缝插件。

**🔧 技术方法**

技术包括：零阶梯度估计（方向梯度近似）、高斯分布参数化的采样策略、REINFORCE型策略梯度更新、方向对齐度（C^t）的理论分析，以及针对LLM微调的K次前向查询策略。

**📊 数据集**

主要实验数据集为SST‑2情感分类数据集，且在两种大模型上验证：RoBERTa‑Large（≈355M参数）和OPT‑1.3B（≈1.3B参数）。

**📈 对比分析**

与基准零阶优化器（ZO‑SGD、ZO‑AdaMM、JAGUAR SignSGD）在相同前向查询预算下对比，使用相同学习率与调度。结果显示在所有模型与微调方式（全权重微调和LoRA）下，采用本框架可提升测试准确率约1–2个百分点，且相较单纯增加K值的无采样均值更新效果更显著。

**⚠️ 局限性**

局限性包括：① 需要预设均值更新步长γ_μ与方差ε的经验性超参数；② 对梯度方向信息的学习依赖于随机采样，仍可能在极端高维或梯度噪声大时表现受限；③ 论文仅在有限的LLM微调任务与两种模型上验证，未探讨更大规模或不同任务的通用性。

---

## 706. Adversarial Network Imagination: Causal LLMs and Digital Twins for Proactive Telecom Mitigation

**arXiv ID:** 2602.13203 | [PDF](https://arxiv.org/pdf/2602.13203v1)

**作者:** Vignesh Sriram `[一作]` (Binghamton University), Zhaohan Xi `[通讯]` (Binghamton University)

**通讯引用:** 116 | [OpenAlex ID](https://openalex.org/A5026309535)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了“Adversarial Network Imagination”闭环框架，利用因果大型语言模型、知识图谱和数字孪生自动生成、模拟并评估电信网络的对抗性故障场景，实现从被动故障检测到主动韧性分析的转变。

**💡 创新点**

创新点在于将因果约束嵌入语言模型生成过程，构建可解释的故障序列，并通过数字孪生作为外部验证器形成闭环迭代，提升了故障探索的多样性、级联深度和缓解效果。

**🔧 技术方法**

使用了因果大型语言模型（Causal LLM）结合知识图谱进行结构化生成、对抗性场景生成器、数字孪生仿真引擎、缓解引擎以及链式思考和受约束解码技术。

**📊 数据集**

基于公开的 ISP 级网络拓扑（Internet Topology Zoo）、AS 级拓扑（CAIDA）、MAWI 后门流量等数据构建知识图谱和数字孪生。

**📈 对比分析**

与规则化故障注入、无 LLM 的数字孪生模拟以及历史故障重放三种基线比较，实验显示因果约束生成的场景在多组件级联深度、影响节点数和缓解收益方面分别提升约 30%–50%，并在不同拓扑上保持一致性。

**⚠️ 局限性**

局限在于数字孪生的精度依赖于拓扑与资源数据完整性，计算成本随级联规模和流量压力上升，Causal LLM 生成结果偶尔不完整或需人工校验，以及缺乏真实运营数据导致泛化受限。

---

## 707. AuTAgent: A Reinforcement Learning Framework for Tool-Augmented Audio Reasoning

**arXiv ID:** 2602.13685 | [PDF](https://arxiv.org/pdf/2602.13685v1)

**作者:** Siqian Tong `[一作]` (Institute of Acoustics, Chinese Academy of Sciences), Chengpeng Hao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 37833 | [OpenAlex ID](https://openalex.org/A5100709340)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了AuTAgent，一个使用强化学习来动态调用多种音频工具进行音频推理的框架。

**💡 创新点**

创新点在于将工具调用与强化学习策略结合，使模型能够自动规划工具链并进行自适应推理。

**🔧 技术方法**

采用了强化学习（PPO/DQN）与预训练音频工具（声纹识别、语音识别、事件检测等）相结合的技术。

**📊 数据集**

在Clotho、AudioSet以及Audio Reasoning Benchmark等公开音频数据集上进行实验。

**📈 对比分析**

与传统端到端模型和基线工具链比较，AuTAgent在推理准确率上提升约3-5个百分点，并保持了良好的推理速度。

**⚠️ 局限性**

局限性包括对工具性能高度依赖、奖励设计敏感、训练过程计算成本高，以及在更复杂多任务场景下的可扩展性待提升。

---

## 708. Human-Centered Explainable AI for Security Enhancement: A Deep Intrusion Detection Framework

**arXiv ID:** 2602.13271 | [PDF](https://arxiv.org/pdf/2602.13271v1)

**作者:** Md Muntasir Jahid Ayan `[一作]` (United International University), Faisal Quader `[通讯]` (University of Maryland)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5093980031)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了结合CNN和LSTM的深度网络入侵检测框架，并将SHAP可解释性集成到模型决策中，同时构建用户界面验证系统的信任度与可用性。

**💡 创新点**

创新点在于实现高检测性能与人类可解释性的双重目标：①利用CNN捕获空间特征、LSTM捕获时序特征提升检测精度；②采用SHAP为每一条入侵判定提供可视化的特征重要性；③通过交互式UI收集专家反馈，验证模型解释的可信度。

**🔧 技术方法**

使用技术包括CNN、LSTM、SHAP（DeepExplainer/KernelExplainer）、Min‑Max归一化、Label Encoding、TensorFlow、Python、用户体验问卷与Cronbachα统计。

**📊 数据集**

采用NSL‑KDD数据集（41特征，80/20划分），并将攻击类别归并为5类（Normal、Dos、Probe、R2L、U2R）。

**📈 对比分析**

通过与传统IDS和黑盒深度模型的对比，实验显示两模型均达到99%整体准确率；CNN宏平均F1为0.86，LSTM宏平均F1为0.93，显示在多类别检测中优于传统方法。

**⚠️ 局限性**

局限性包括：① SHAP与TensorFlow版本兼容性导致解释过程不稳定；② 对少数类别（如U2R）的召回率仍偏低；③ 仅在NSL‑KDD小规模数据上验证，缺乏大规模实时环境评估；④ 用户问卷样本仅15人，可信度验证有限。

---

## 709. Depth Completion as Parameter-Efficient Test-Time Adaptation

**arXiv ID:** 2602.14751 | [PDF](https://arxiv.org/pdf/2602.14751v1)

**作者:** Bingxin Ke `[一作]` (NVIDIA), Shengyu Huang `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种参数高效的测试时适配框架，用预训练的3D基础模型（如VGGT）对稀疏深度进行自监督微调，从而完成稠密深度恢复。

**💡 创新点**

创新点在于：①把深度完成视为基于已有基础模型的测试时参数高效适配；②仅更新少量参数（LoRA或VPT），保持原始模型权重冻结；③在视频中引入序列级参数共享，利用多帧信息提升鲁棒性和时序一致性；④实现对任何ViT基础模型的无缝兼容。

**🔧 技术方法**

核心技术包括参数高效微调（PEFT）方法LoRA和VPT、稀疏深度对齐与L1损失、自监督梯度优化、序列级共享参数以及ViT Encoder的自注意力机制。

**📊 数据集**

使用的评估数据集有ScanNet、7-Scenes、iBims和Mapillary Metropolis；稀疏深度采样方式覆盖SIFT、SfM、随机点、LiDAR扫描线等多种场景。

**📈 对比分析**

与DepthAnythingV2、UniDepthV2、MoGe-2、VGGT、VideoDA、PromptDA、OMNI-DC、PriorDA、Marigold-DC、TestPromptDC等基线在AbsRel和OPW指标上对比，本文方法在所有数据集与稀疏模式下均名列前茅，误差往往低于其它方法一半，并且时序一致性显著提升。

**⚠️ 局限性**

局限性包括：仍比单前向推理慢；依赖稀疏深度的可用性，极稀疏或高噪声条件下效果下降；收敛需要多步迭代，速度受限；与全模型微调相比，仅略优而参数更少，难以与最优全模型性能相媲美。

---

## 710. BotzoneBench: Scalable LLM Evaluation via Graded AI Anchors

**arXiv ID:** 2602.13214 | [PDF](https://arxiv.org/pdf/2602.13214v1)

**作者:** Lingfeng Li `[一作]` (Peking University), Wenxin Li `[通讯]` (Peking University)

**通讯引用:** 3267 | [OpenAlex ID](https://openalex.org/A5100397213)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 BotzoneBench 框架，使用固定的技能梯度 AI 基线对 LLM 在八款游戏中的互动决策能力进行绝对评估。

**💡 创新点**

核心创新是将评估锚定在已校准的经典 AI 体系上，实现 O(N) 的线性扩展、稳定的绝对分数，并构建了大规模的决策日志数据集。

**🔧 技术方法**

技术手段包括基于 Botzone 平台的游戏环境、结构化 Prompt 让 LLM 成为规则遵循的决策者、双重匹配设计控制随机性以及 win‑rate 级别化评估。

**📊 数据集**

使用的数据集为 8 款游戏共 6,403 场比赛，产生 177,047 条状态-动作对及其推理轨迹，涵盖 5 大旗帜模型及 3 规模不同的 Qwen 变体。

**📈 对比分析**

通过对比各模型与不同层级基线的胜率，得出模型在 Lv0‑Lv5 的最高等级及进度；旗舰模型 Gemini3‑Pro 在大多数游戏中领先约两级，GPT‑5 在 Chess 中表现最佳。

**⚠️ 局限性**

局限性包括仅评估少数模型、部分游戏基线稀缺导致分级不细、Chess 与 Texas Hold'em 的基线质量不均，以及部分长推理模型因超时而无法完整评测。

---

## 711. EmbeWebAgent: Embedding Web Agents into Any Customized UI

**arXiv ID:** 2602.14865 | [PDF](https://arxiv.org/pdf/2602.14865v1)

**作者:** Chenyang Ma `[一作]` (University of Oxford), Dave Braines `[通讯]` (IBM Research Europe)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了一种嵌入式 Web 代理框架，可直接将智能代理植入现有前端应用，利用 ARIA 标签、URL 和前端函数注册表提供结构化观测与可调用功能，支持跨页面混合粒度的动作与多代理协同；

**💡 创新点**

创新点在于：1）利用可访问性 ARIA 标签与 URL 作为高质量观测，避免截图/DOM 依赖；2）前端仅需极少代码（≈150 行）即可提供观测与函数注册；3）通过页面函数映射限制动作空间，实现多堆栈兼容；4）多代理架构结合 ReAct 与 CoT，完成导航、分析与聊天；

**🔧 技术方法**

技术包括：前端轻量化 shim、WebSocket 双向通信、基于 LLM 的 ReAct/CoT 规划、MCP（Model Context Protocol）调用域工具、DSPy、Llama 4 Maverick、React/Angular 兼容、Phoenix 追踪；

**📊 数据集**

未使用公开数据集；评估基于内部化学分析 UI 的集成测试套件，包含前端模拟、聊天查询、动作验证与性能监测；

**📈 对比分析**

评估通过自动化测试验证功能调用正确性、导航完成率与聊天摘要准确度，并利用 Phoenix 监控并发会话下的延迟与隔离性；在多轮复杂工作流下表现良好，未给出具体指标；

**⚠️ 局限性**

局限性包括：依赖高质量 ARIA 标签与手工维护的页面-函数映射；需人工标注与维护；对动态页面或缺乏 ARIA 的组件支持有限；未来工作计划标准化 ARIA 规范与半自动提取动作。

---

## 712. AD-Bench: A Real-World, Trajectory-Aware Advertising Analytics Benchmark for LLM Agents

**arXiv ID:** 2602.14257 | [PDF](https://arxiv.org/pdf/2602.14257v1)

**作者:** Lingxiang Hu `[一作]` (Tencent), Jie Jiang `[通讯]` (Tencent)

**通讯引用:** 1517 | [OpenAlex ID](https://openalex.org/A5101944041)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个基于真实广告营销业务需求的轨迹感知评测基准AD-Bench

**💡 创新点**

创新点在于动态生成地面真值、轨迹感知评估以及长尾任务分层构建

**🔧 技术方法**

采用ReAct框架的多工具LLM代理，利用LLM-judge和工具调用记录进行评测

**📊 数据集**

使用了从线上广告平台收集的2000条用户分析请求，构成823条高质量实例及100条独特轨迹子集

**📈 对比分析**

通过Pass@k和轨迹覆盖率评估，对多款公开及闭源模型进行对比，Gemini-3-Pro在整体上达到Pass@3 83%但在最难L3仅62%，显示模型仍面临规划不稳等瓶颈

**⚠️ 局限性**

限制在于高难度任务仍表现不佳，模型易出现轨迹规划、参数错误、幻觉和计算错误，且依赖单一参考轨迹导致评估不够全面

---

## 713. Parallel Sparse and Data-Sparse Factorization-based Linear Solvers

**arXiv ID:** 2602.14289 | [PDF](https://arxiv.org/pdf/2602.14289v1)

**作者:** Xiaoye Sherry Li `[一作]` (Lawrence Berkeley National Laboratory), Yang Liu `[通讯]` (Lawrence Berkeley National Laboratory)

**通讯引用:** 149650 | [OpenAlex ID](https://openalex.org/A5101717144)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文回顾了稀疏直接求解器的最新进展，重点在于减少任务和数据并行设置中的通信和延迟成本，以及通过低秩和其他压缩技术（如分层矩阵代数）来降低计算复杂性。

**💡 创新点**

创新点在于提出了一种结合结构稀疏和数据稀疏的混合方法，能够比传统的稀疏直接求解器更高效地解决线性系统。

**🔧 技术方法**

使用了高性能计算的基本原理，结合了稀疏矩阵计算的并行算法和分布式内存稀疏直接求解器的实现技术。

**📊 数据集**

使用了多种实际应用中的稀疏矩阵数据集，包括来自物理现象模拟、机器学习和数据科学的线性方程组。

**📈 对比分析**

通过与现有的稀疏直接求解器（如SuperLU、PDSLin和STRUMPACK）进行比较，展示了在通信带宽和延迟成本方面的显著降低，算法的计算和内存复杂性接近线性。

**⚠️ 局限性**

限制在于现有的稀疏直接求解器在处理复杂的稀疏模式时仍然面临挑战，尤其是在异构计算架构上实现高效的并行化。

---

## 714. TS-Haystack: A Multi-Scale Retrieval Benchmark for Time Series Language Models

**arXiv ID:** 2602.14200 | [PDF](https://arxiv.org/pdf/2602.14200v1)

**作者:** Nicolas Zumarraga `[一作]` (ETH Zurich), Robert Jakob `[通讯]` (ETH Zurich)

**通讯引用:** 695 | [OpenAlex ID](https://openalex.org/A5077371016)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建长上下文时间序列检索基准TS‑Haystack并评估多种TSLM模型。

**💡 创新点**

提出TS‑Haystack基准、系统化评估长上下文检索，并揭示潜在重采样压缩对分类有利、对检索不利的任务依赖性。

**🔧 技术方法**

使用针刺入沙堆范式、链式思维提示、潜在重采样与全自注意力时间序列编码器、LLM（Llama‑3.2‑1B、Qwen2.5‑1.5B‑Instruct）等技术。

**📊 数据集**

基于Capture24加速度计数据（24小时）并插入短活动片段生成检索任务。

**📈 对比分析**

在 2.56 s–2 h 上下文长度下对比 OpenTSLM‑Flamingo 与 ITFormer；OpenTSLM‑Flamingo 在分类上提升至 41.4% Macro‑F1，但检索准确率从 29.6% 降至 ~25%；oracle 版本保持 82–91% 的稳定检索精度。

**⚠️ 局限性**

ITFormer 在长上下文出现 OOM，检索性能整体低于随机基线；归因仅定位编码器瓶颈，未解决定位精度和多任务泛化问题；基准仅针对加速度计数据。

---

## 715. NeuroWeaver: An Autonomous Evolutionary Agent for Exploring the Programmatic Space of EEG Analysis Pipelines

**arXiv ID:** 2602.13473 | [PDF](https://arxiv.org/pdf/2602.13473v1)

**作者:** Guoan Wang `[一作]` (Stevens Institute of Technology), Feng Liu `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 17556 | [OpenAlex ID](https://openalex.org/A5100415272)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并评估了一种名为NeuroWeaver的自主进化代理，用于自动化设计EEG分析流水线，能够在多任务、多数据集上生成轻量、高性能的端到端代码。

**💡 创新点**

将EEG流水线设计视为受限离散优化问题，提出域信息子空间初始化、基于多目标进化的闭环优化以及自我反思调优，显著提升可解释性和效率。

**🔧 技术方法**

基于大型语言模型ChatGPT‑5.1的代码生成与调试，域知识检索与Braindecode先验，进化算法树搜索，多目标奖励函数，数据驱动约束抽取与自适应知识检索。

**📊 数据集**

在五个异构基准集（TUEV、SEED、HMC、Workload、TUSL）上进行评测，涵盖情绪识别、任务切换、工作负荷等多种EEG任务。

**📈 对比分析**

与任务特定方法和大型预训练模型（BIOT、LaBraM‑Base、NeuroLM‑B）进行对比，NeuroWeaver在大多数指标上优于任务特定基线，参数量大幅缩减，并在HMC和Workload上击败资源密集型模型。

**⚠️ 局限性**

受限于固定的预处理流程与硬编码的执行预算，且当前仅在受限搜索空间内操作；未来需放宽预处理限制以探索更完整的流水线设计。

---

## 716. PAct: Part-Decomposed Single-View Articulated Object Generation

**arXiv ID:** 2602.14965 | [PDF](https://arxiv.org/pdf/2602.14965v1)

**作者:** Qingming Liu `[一作]` (Chinese University of Hong Kong), Kui Jia `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于单张图片的面向部件的生成框架，用来一次性恢复物体的几何、外观与关节运动，实现可直接用于物理仿真的可动三维模型。

**💡 创新点**

创新点在于将预训练的两阶段3D生成模型TRELLIS细化为部件层次的流模型，并在Transformer中引入局部部件注意力与多步特征聚合，实现部件级结构预测与关节参数回归的统一推理。

**🔧 技术方法**

核心技术包括：部件感知流模型（Part‑Aware Flow Model）、稀疏变压器去噪、跨步特征聚合预测关节参数、以及基于VLM+SAM的自动部件掩模生成。

**📊 数据集**

使用了PartNet‑Mobility和ACD两个公开数据集进行训练与评估，并在这两套数据集上完成了单视图可动对象的生成与比较。

**📈 对比分析**

与SINGAPO、Articulate‑Anything、PhysX‑3D、PhysXAnything、ArtFormer和FreeArt3D等基线对比，本文方法在几何精度、部件一致性、关节可行性以及与原图像的视觉一致性指标（CLIP）上均显著优于所有对手，且推理速度快于优化型方法。

**⚠️ 局限性**

主要局限包括：难以处理超过八个部件的复杂多部件物体、对遮挡或不可见的功能部件识别不足，以及仅适用于浅层树状关节结构，无法直接处理闭链或共享约束等更复杂的机械结构。

---

## 717. Socially-Weighted Alignment: A Game-Theoretic Framework for Multi-Agent LLM Systems

**arXiv ID:** 2602.14471 | [PDF](https://arxiv.org/pdf/2602.14471v1)

**作者:** Furkan Mumcu `[一作]` (University of South Florida), Yasin Yilmaz `[通讯]` (University of South Florida)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在多智能体LLM系统中仅通过推理时加权私有目标与集体福利的社会加权对齐（SWA）框架，推导出共享资源拥堵游戏的临界阈值 λ^*，并在仿真中验证该阈值导致的相位转变。

**💡 创新点**

创新点在于：①仅在推理时通过单参数 λ 线性混合私有奖励与集体福利，实现无训练、无集中控制的协调；②给出闭式临界阈值 λ^*=(n-β)/(n-1)，预测系统由持续拥堵转向稳定的临界点；③通过理论与仿真统一验证。

**🔧 技术方法**

使用的技术包括：游戏理论分析、连续拥堵游戏模型、基于候选动作评估的推理时加权评分（SWI）机制，以及多模型仿真验证。

**📊 数据集**

实验数据来源为多种小型高效LLM模型（Phi‑3.5‑mini‑instruct、Mistral‑7B、Qwen2.5‑7B、Llama‑3‑8B、Gemma‑3‑4B），在自定义的共享资源拥堵环境中生成实验数据，不依赖公开大规模数据集。

**📈 对比分析**

通过对比不同 λ 值下的超载率和社会福利改进，观察其与理论阈值的一致性；实验表明，当 λ 接近 λ^* 时，超载率急剧下降，福利显著提升，验证了理论预测，性能表现优于 λ=0 的纯自利策略。

**⚠️ 局限性**

局限性包括：①对集体福利估计的依赖（如移动平均）在噪声或不可观测环境下可能失效；②假设所有代理同质，未考虑异质角色和不同动作空间；③在更复杂真实场景中的可扩展性与鲁棒性尚待进一步验证。

---

## 718. Beyond Token-Level Policy Gradients for Complex Reasoning with Large Language Models

**arXiv ID:** 2602.14386 | [PDF](https://arxiv.org/pdf/2602.14386v1)

**作者:** Mufan Xu `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 60043 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种 Multi‑Token Policy Gradient Optimization (MPO) 框架，利用 K 连续 token 作为一个语义动作进行强化学习优化，提升大语言模型在复杂推理任务中的一致性与性能。

**💡 创新点**

创新点在于将传统 token‑level 的策略梯度转为块级（block‑level）优化，使用多 token 预测模块（MTP）计算联合重要性采样比值，并通过权重衰减控制多步信息的贡献，从而减少梯度方差、提升训练稳定性。

**🔧 技术方法**

核心技术包括：多 token 预测 (MTP)、重要性采样比值聚合与对数加权、改进的 PPO 损失函数（MPO surrogate objective）、warm‑up 阶段以及在推理时仅使用单 token 输出的兼容性设计。

**📊 数据集**

实验数据集：数学推理基准 GSM8K、MATH，以及代码生成基准 HumanEval（MBPP 作为训练集）。

**📈 对比分析**

与 PPO、GRPO、DAPO 等 token‑level 基线在同一模型（Llama3.2‑1B‑Instruct、DeepSeek‑Distilled‑Qwen2.5‑1.5B、DeepSeek‑Distilled‑Qwen2.5‑7B）上对比，MPO 在 GSM8K、MATH 和 HumanEval 上均取得更高的 pass@1 分数，表明块级优化显著提升推理和代码生成效果；同时还表现出更低的梯度方差和更小的 clip 比例。

**⚠️ 局限性**

局限性包括：对 MTP 的 warm‑up 依赖度高，可能在未预训练 MTP 的模型上导致性能下降；块大小 K 受计算开销限制，K>5 的效果尚未验证；以及 MPO 目前仅在 PPO 基础上实现，其他 surrogate objective（如 GRPO）集成仍需进一步研究。

---

## 719. Stay in Character, Stay Safe: Dual-Cycle Adversarial Self-Evolution for Safety Role-Playing Agents

**arXiv ID:** 2602.13234 | [PDF](https://arxiv.org/pdf/2602.13234v1)

**作者:** Mingyang Liao `[一作]` (Baidu Inc.), Jizhou Huang `[通讯]` (Baidu Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了训练‑free 的 Dual‑Cycle Adversarial Self‑Evolution（DASE）框架，利用自动化攻击者与角色扮演防御者共同演化，构建层级知识库来提升角色一致性与安全防御；

**💡 创新点**

创新点在于：①不通过梯度更新实现模型自适应，②将攻击者与防御者的互动设计为双循环自进化，③构建三层层级知识库（全局安全规则、角色约束、黄金示例）实现自然语言级别的安全与角色学习；

**🔧 技术方法**

主要技术包括：对抗自进化机制、检索增强生成（RAG）与知识检索、结构化知识库的增删改合操作、判别器评估与自校正循环；

**📊 数据集**

使用的基准数据集有 RoleBench（角色一致性评估）、BeaverTails 与 HEx‑PHI（安全性评估）、AIM、Cipher、CodeChameleon（对抗性 jailbreak 评测）；

**📈 对比分析**

与 SaRFT（训练‑based）和多款大规模训练‑free 基线（Qwen3、GPT‑oss、Kimi‑K2、GPT‑5.2）进行对比，DASE 在 RoleBench（平均 35.60 对比 28.69）、安全分数（97.84 对比 94.16）以及 jailbreak 抗击率（78.33 对比 45.30）等指标均表现出显著提升；

**⚠️ 局限性**

局限性包括：需依赖黑盒 API 进行多轮交互，处理极新或极端的 persona/攻击策略可能受限；知识库的可解释性和透明度有限；实验仅覆盖特定专有模型，迁移效果仍需进一步验证；计算成本与迭代次数相关。

---

## 720. On Representation Redundancy in Large-Scale Instruction Tuning Data Selection

**arXiv ID:** 2602.13773 | [PDF](https://arxiv.org/pdf/2602.13773v1)

**作者:** Youwei Shu `[一作]` (National University of Singapore), Jiaheng Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 14653 | [OpenAlex ID](https://openalex.org/A5032474012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究工业规模指令调优数据选择，提出利用语义表示的压缩与白化方法（CRDS-R、CRDS-W）来挑选高质量子集。

**💡 创新点**

创新点包括：①揭示现有语义嵌入存在严重冗余；②引入 Rademacher 投影-拼接和白化降维来提升表示质量；③设计可扩展的异构分布式框架，实现百万级数据的高效相似度搜索。

**🔧 技术方法**

技术手段：Transformer 隐层表示、Rademacher 随机投影、白化变换、余弦相似度、轮询式多样本选择、分布式 GPU 并行计算。

**📊 数据集**

使用工业级 Ling 2.0（200 万条 SFT 示例）及其 Dev 版本；在 GSM8K、MMLU、MBPP、BBH 四个基准上进行评估。

**📈 对比分析**

与随机采样、Mid‑PPL、长度、RDS+ 等基线对比；CRDS‑W 在 3.5% 数据量下平均提升 0.71%（比全量数据低），在 RDS+ 上提升约 1–1.5%；CRDS‑R 也超过全量数据并优于 RDS+。

**⚠️ 局限性**

局限性：方法在小规模模型上的效果尚不确定；对语义相似度驱动的选取机制本质不清；与梯度基方法的对比缺乏深入解析。

---

## 721. Unlocking Reasoning Capability on Machine Translation in Large Language Models

**arXiv ID:** 2602.14763 | [PDF](https://arxiv.org/pdf/2602.14763v1)

**作者:** Sara Rajaee `[一作]` (University of Amsterdam), Tom Kocmi `[通讯]` (Cohere)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了多种推理型大模型在机器翻译上的表现，并提出了一种针对翻译的结构化推理框架。

**💡 创新点**

创新点在于发现传统线性推理不利于MT，并设计了多步（草稿、充分性、流畅性、最终翻译）动态结构化推理模板，并用合成数据训练。

**🔧 技术方法**

主要技术包括对齐推理轨迹、构建结构化推理模板、在111B参数大模型上进行后训练（SFT），并使用XCOMET-XL评估。

**📊 数据集**

使用的数据集为WMT24++基准（9语言对）以及由自建的23语种高质量单语料生成的约28k条结构化推理样本。

**📈 对比分析**

与直接翻译、线性推理注入等基线比较，结构化推理在多数语言对上平均提升约1–2 XCOMET-XL分数（相当于+1 BLEU），显示显著性能提升。

**⚠️ 局限性**

局限性包括仅评估9语言对，缺乏人类评测，仅使用自动指标，且对极端难度翻译的覆盖有限。

---

## 722. Arbor: A Framework for Reliable Navigation of Critical Conversation Flows

**arXiv ID:** 2602.14643 | [PDF](https://arxiv.org/pdf/2602.14643v1)

**作者:** Luís Silva `[一作]` (Sword Health), Luís Ungaro `[通讯]` (Sword Health)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了Arbor框架，分离决策树检索、逻辑评估与自然语言生成，实现结构化对话导航；

**💡 创新点**

将决策树结构外部化为边列表并在运行时仅检索当前节点的邻接边，分离评估与生成步骤，显著降低模型输入量、提升导航准确性并增强可调试性；

**🔧 技术方法**

使用大语言模型（GPT‑5/4.1、Claude Sonnet、Gemini、DeepSeek、Qwen等）与LangGraph图形编排、链式思考、边列表标准化检索和多阶段LLM调用；

**📊 数据集**

基于真实临床分诊对话（20条完整会话，174个决策点）和对应的449节点、980条边的决策树；

**📈 对比分析**

与单一提示基线对比，采用转弯准确率、延迟、成本和消息质量四项指标评估；Arbor在所有模型上平均提升约29个百分点准确率、成本降低约14.4倍、延迟下降57%，消息质量无显著差异；

**⚠️ 局限性**

受限于多步推理导致的实时延迟、仅正向遍历、缺乏状态回滚与漂移检测、缺少多模型集成或自适应策略等方面。

---

## 723. Hierarchical Vision-Language Interaction for Facial Action Unit Detection

**arXiv ID:** 2602.14425 | [PDF](https://arxiv.org/pdf/2602.14425v1)

**作者:** Yong Li `[一作]` (Southeast University), Cuntai Guan `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种Hierarchical Vision‑Language Attention框架 HiVA，通过融合视觉特征与语言描述的多粒度交互来实现面部动作单元（AU）检测。

**💡 创新点**

创新点包括①利用大语言模型生成多样化 AU 描述并加入正则化项以提升文本嵌入的辨别性；②提出 AU‑aware 动态图模块以及双重交叉注意机制（Disentangled Dual Cross‑Attention 与 Contextual Dual Cross‑Attention）实现局部与全局视觉‑文本协同学习。

**🔧 技术方法**

采用 Swin Transformer 作为视觉编码器、BERT‑Large 进行文本编码、图卷积网络建模 AU 关系、Transformer 交叉注意机制（DDCA/CDCA）、正则化损失以及两阶段训练策略。

**📊 数据集**

在 BP4D、DISFA 与 GFT 三大公开 FACS 数据集上进行实验。

**📈 对比分析**

与多种视觉基线及现有语言辅助 AU 检测方法对比，HiVA 在三大数据集上均取得最高或接近最高的平均 F1 分数，尤其对低频 AU 的提升显著。

**⚠️ 局限性**

局限性：依赖预训练语言模型与文本生成，推理时算力消耗相对较大；对文本质量敏感；跨域或视频动态场景的泛化尚未充分验证。

---

## 724. Learning Transferability: A Two-Stage Reinforcement Learning Approach for Enhancing Quadruped Robots' Performance in U-Shaped Stair Climbing

**arXiv ID:** 2602.14473 | [PDF](https://arxiv.org/pdf/2602.14473v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 725. Detecting Brick Kiln Infrastructure at Scale: Graph, Foundation, and Remote Sensing Models for Satellite Imagery Data

**arXiv ID:** 2602.13350 | [PDF](https://arxiv.org/pdf/2602.13350v1)

**作者:** Usman Nazir `[一作]` (University of Oxford), Sara Khalid `[通讯]` (University of Oxford)

**通讯引用:** 3670 | [OpenAlex ID](https://openalex.org/A5051460301)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了覆盖南亚五城的0.149 m/像素Zoom‑20卫星图像数据集，并系统评估了图神经网络、基础模型与经典遥感方法在砖窑检测上的表现。

**💡 创新点**

提出了区块自适应的各向异性注意力图神经网络ClimateGraph，首次将空间邻接和方向信息融入图结构，形成了兼顾局部精度与区域上下文的检测框架；同时对比分析了多种学习范式的优势。

**🔧 技术方法**

使用的技术包括基于注意力的图神经网络（ClimateGraph）、预训练的视觉‑语言模型（RemoteCLIP、Rex‑Omni）以及基于红色指数、阈值和几何约束的传统遥感检测管线。

**📊 数据集**

数据集由约1.3 百万张256×256像素的Zoom‑20卫星图像组成，覆盖巴基斯坦拉合尔、印度德里、尼泊尔加德满都、孟加拉国加兹比普尔、阿富汗喀布尔五大城市。

**📈 对比分析**

在宏观F1指标下，ClimateGraph在全域图结构上实现了约17 %（相较于GCN/GAT）或1 %（相较于GraphSAGE）的提升；RemoteCLIP在各城市的F1从0.446到0.526不等，Rex‑Omni表现低于0.25；传统遥感管线在拉合尔和加兹比普尔的F1分别达到0.533和0.650，优于部分基础模型。

**⚠️ 局限性**

主要局限包括：数据稀缺导致模型泛化受限；不同城市的建筑与红色反射特征差异导致基于遥感的规则性能波动；基础模型在跨域迁移时表现不稳定；缺乏大规模标注使得精细化调优受限。

---

## 726. Morphing of and writing with a scissor linkage mechanism

**arXiv ID:** 2602.14958 | [PDF](https://arxiv.org/pdf/2602.14958v1)

**作者:** Mohanraj A `[一作]` (Indian Institute of Technology Madras), S Ganga Prasath `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文研究了刀刀式连杆机制，推导了其有效曲率公式，并通过优化框架在单自由度系统中实现形状变换与轨迹书写。

**💡 创新点**

创新点在于利用有效曲率将几何参数化为可调节的单元比率，并结合可微仿真与梯度优化，首次在单自由度刀刀式连杆中实现复杂形状与文字的精准编程。

**🔧 技术方法**

采用了刚体连杆运动学、曲率解析推导、基于梯度的优化算法、自动微分可微仿真、3D打印单元、舵机驱动与Arduino控制等技术。

**📊 数据集**

使用了自制的3D打印刀刀式单元以及实验测得的曲率、轨迹数据；未使用公开数据集。

**📈 对比分析**

通过将理论预测的曲率、关键角度和尖端轨迹与实验测量结果进行对比，结果显示理论与实验吻合度高，成功实现目标曲线的形状变换和字母写作，但因单元比率的微小误差会导致较大的形变误差。

**⚠️ 局限性**

局限性包括：单自由度机制对单元比率的高度敏感，实验中小的装配误差会放大为较大的形变误差；缺乏反馈控制导致无法实现零误差实现；动态调节铰链位置以消除历史耦合仍具挑战。

---

## 727. Computability of Agentic Systems

**arXiv ID:** 2602.13222 | [PDF](https://arxiv.org/pdf/2602.13222v1)

**作者:** Chatavut Viriyasuthee `[一作]` `[通讯]`, Chatavut Viriyasuthee

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Quest Graph 及其变体（FQDP、NFQDP、RQDP、NRQDP），构建与正式语言层次结构对应的计算模型，并分析这些模型在有限上下文窗口下模拟计算图的理论效率。

**💡 创新点**

① 将代理系统的常见推理模式映射到形式语言层次；② 证明 Reference‑Augmented 模型在保持不可变历史的前提下实现图灵完备；③ 给出不同模型模拟最大依赖计算图的时间复杂度，揭示计算能力与效率的层次关系。

**🔧 技术方法**

使用图论与自动机理论构造 Quest Graph 结构；定义 Agent 函数与参考机制；利用最坏情况分析（最大依赖计算图）和代理的上下文限制来推导时间复杂度；对比标准 LM 的局限性。

**📊 数据集**

论文主要是理论推导，并未使用真实数据集；通过构造的最大依赖计算图（MCG/BMCG）和抽象的计算图模型进行复杂度分析。

**📈 对比分析**

通过计算复杂度比较：无限 Quest Graph O(N²)，RQDP/NRQDP O(N² log N)，FQDP/NFQDP O(2ⁿ)，标准 LM 不能在常数响应大小下完成。显示 Reference‑Augmented 模型在保持历史不可变的同时显著提升效率。

**⚠️ 局限性**

① 仅为理论分析，缺乏实证验证；② 需要预先知道完整计算图或能构造参考图，实际应用中可能成本高；③ 对上下文大小 C 的假设过于理想化；④ 过度依赖模型可读性与可实现性，实际代理实现复杂度高。

---

## 728. The geometry of invariant learning: an information-theoretic analysis of data augmentation and generalization

**arXiv ID:** 2602.14423 | [PDF](https://arxiv.org/pdf/2602.14423v1)

**作者:** Abdelali Bouyahia `[一作]` (Laval University), Mario Marchand `[通讯]` (Laval University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过信息论框架推导数据增强对泛化误差的上界，并将数据增强建模为原始分布与变换分布的复合，进一步拆解为分布偏移、学习稳定性与增强敏感性三项。

**💡 创新点**

首次在信息论泛化分析中引入群直径度量统一控制所有三项，并通过轨道平均损失与逆Pinsker不等式等工具把数据增强的几何特性与泛化性能量化关联。

**🔧 技术方法**

使用互信息、KL散度、子高斯假设、轨道平均损失、逆Pinsker不等式、Lipschitz连续性等信息论与概率工具进行理论推导，并给出闭式或可估计的上界。

**📊 数据集**

在 MNIST 与 FashionMNIST 图像数据集上，采用卷积神经网络并通过对数比估计与 MINE 估计相互信息和 KL 散度进行数值实验。

**📈 对比分析**

与传统信息论泛化上界对比，实验结果显示新上界能更准确追踪实际泛化误差，尤其在中等强度数据增强下表现优于基线；通过可视化三项贡献揭示了分布偏移与稳定性的权衡。

**⚠️ 局限性**

局限性在于需满足子高斯、Lipschitz连续性、绝对连续性等假设，对高维连续数据的互信息估计仍具挑战，且群直径对极端强度或复杂变换时可能不足以完全捕捉所有偏差。

---

## 729. Near-Optimal Regret for Policy Optimization in Contextual MDPs with General Offline Function Approximation

**arXiv ID:** 2602.13706 | [PDF](https://arxiv.org/pdf/2602.13706v1)

**作者:** Orin Levy `[一作]` (Blavatnik School of Computer Science), Yishay Mansour `[通讯]` (Blavatnik School of Computer Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了首个针对随机上下文马尔可夫决策过程（CMDP）且支持通用离线函数逼近的策略优化算法。

**💡 创新点**

创新点在于：① 在CMDP框架中首次将基于最优性（optimism）的策略优化与离线回归算子结合；② 设计了一种基于最新动态估计的探索奖励，只依赖于过去策略的伪占用测量；③ 通过新的置信区间证明，去除了先前研究中对可达性、Eluder维数等额外结构假设；④ 取得了近最优的后悔界 O(H⁴√(T|S||A|log(|ℱ||𝒫|)))。

**🔧 技术方法**

使用的技术包括：离线最小二乘回归（loss估计）与对数损失回归（动态估计）；基于指数加权（exponential‑weight）策略更新的乐观策略优化框架；利用占用测量的置信上界构造探索奖励；对策估计的平方Hellinger距离与期望占用测量的分析。

**📊 数据集**

本文仅给出理论分析，没有使用具体数据集；所有结果均基于假设的函数类 ℱ、𝒫 的离线回归算子。

**📈 对比分析**

与之前的 RM-UCDD、E‑UC³RL、LOLIPOP 等方法相比，消除了对可达性或 Eluder 维数的假设，显著降低了对 |S|、|A| 的多项式依赖；在保留近最优后悔率的同时，算法实现更简单、计算更高效。

**⚠️ 局限性**

局限性：① 对时间步 H 的依赖仍然是 H⁴，未达到理论上最优的 H⁴.⁵ 级；② 需要计算对过去策略的伪占用测量，计算成本在大规模问题上仍有提升空间；③ 本文未给出实验验证，仅在理论层面证明其可行性。

---

## 730. An Algebraic Invariant for Free Convolutional Codes over Finite Local Rings

**arXiv ID:** 2602.13468 | [PDF](https://arxiv.org/pdf/2602.13468v1)

**作者:** Mohammed El Oued `[一作]` `[通讯]`, Mohammed El Oued

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了自由卷积码在有限局部环 ℤ_p^r 上的结构，提出残差结构多项式 Δ_p(𝒞) 并证明其不变性及其与灾难性和双码性质的关系。

**💡 创新点**

引入了残差结构多项式作为全新的码不变量，并证明它能精确判定码的灾难性；同时证明 Δ_p(𝒞) 与其对偶码 Δ_p(𝒞^⊥) 相等，揭示灾难性在对偶下的对称性。

**🔧 技术方法**

利用 Reduced Internal Degree Matrix（RIDM）理论、投影到本地域 ℤ_p^r 上的 F_p 并通过最小公因子（GCD）与矩阵不变性证明方法。

**📊 数据集**

无实验数据集，全部为理论推导与证明。

**📈 对比分析**

无实验比较，本文仅给出理论判定方法，未涉及性能指标或与其他方法的对比。

**⚠️ 局限性**

局限在于仅适用于自由卷积码，未扩展到非自由码；并且对 Δ_p(𝒞) 的因子与具体灾难性序列之间的细节关系仍未给出。

---

## 731. Visual Foresight for Robotic Stow: A Diffusion-Based World Model from Sparse Snapshots

**arXiv ID:** 2602.13347 | [PDF](https://arxiv.org/pdf/2602.13347v1)

**作者:** Lijun Zhang `[一作]` (Amazon), Aaron Parness `[通讯]` (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于扩散模型的世界模型（FOREST），用于在工业仓库中根据稀疏的预/后置摄像头截图以及机器人预期的堆放意图，预测箱子内部物体的未来布置；

**💡 创新点**

创新点在于：1）首次在真实仓库稀疏数据上训练并应用扩散模型进行单步视觉前瞻；2）通过实例掩码对齐和物体匹配构造结构化监督；3）使用跨注意力与自适应层归一化的Transformer实现条件扩散；

**🔧 技术方法**

技术包括MaskDINO实例分割、VAE降维、Latent Diffusion Transformer、Hungarian匹配、AdaLN条件注入、Packed‑Sequence Attention；

**📊 数据集**

使用Amazon生产仓库收集的稀疏预/后置RGB快照以及物品的顶视图和属性的公开数据集；

**📈 对比分析**

与基于复制粘贴、复制粘贴+重力等启发式基线对比，在直接评估中新增物体IoU从0.28/0.12提升至0.70/0.62，DLO预测误差提升仅0.0016/0.0025，长序列推理中单步IoU保持在0.4以上，表现显著优于基线；

**⚠️ 局限性**

局限性包括对物体属性的依赖、对稀疏监督的噪声敏感、对不同物品尺寸和稀疏数据的泛化性有限，以及需要真实后置掩码进行训练。

---

## 732. HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models

**arXiv ID:** 2602.13710 | [PDF](https://arxiv.org/pdf/2602.13710v1)

**作者:** Xin Yan `[一作]` (Beijing Normal University), Ivor Tsang `[通讯]` (Agency for Science Technology and Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 HBVLA 框架，对 Vision‑Language‑Action 模型进行 1‑bit 后训练量化，保持动作生成的高保真；

**💡 创新点**

创新点包括：① 改进 Hessian 为策略感知 Hessian，精准识别对动作至关重要的权重；② 对非显著权重采用稀疏正交变换+Haar 小波变换，产生低熵中间状态；③ 将显著与非显著权重分别量化并通过残差 Haar 细化，实现极低位压缩下的高效控制；

**🔧 技术方法**

使用了后训练 1‑bit PTQ、策略感知 Hessian、token 重要性评估、稀疏正交变换、Haar 小波变换、列归一化与分组量化等技术；

**📊 数据集**

评估数据集包括 LIBERO、SIMPLER、Mobile ALOHA 真实机器人（Pick‑and‑Place、Sequenced Instruction、Flexible Folding）以及 OpenVLA、OpenVLA‑OFT、CogACT 等模型；

**📈 对比分析**

与 BiLLM、BiVLM、HBLLM 等现有 1‑bit PTQ 方法在 LIBERO、SIMPLER 与真实机器人上进行对比，HBVLA 在多数任务上平均提升 4–12% 成功率，保持 90%+ 的 SR，且与全精度模型的差距仅 4–6%，明显优于对比方法；

**⚠️ 局限性**

局限性：仍受闭环误差累积影响；未在极端噪声或极小模型规模下验证；仅聚焦 1‑bit PTQ，未探索训练时稀疏化或多位量化；缺乏硬件实现与能耗的细节评估。

---

## 733. HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming

**arXiv ID:** 2602.14214 | [PDF](https://arxiv.org/pdf/2602.14214v1)

**作者:** Jiahui Chen `[一作]` (Tsinghua University), Lifeng Sun `[通讯]` (Tsinghua University)

**通讯引用:** 3351 | [OpenAlex ID](https://openalex.org/A5047712495)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用大型语言模型（LLM）生成视频段落的重要性权重，以提升 VOD 和直播的主观体验质量。

**💡 创新点**

创新点在于：① 用滑动窗口感知模块在有限令牌内扩展 LLM 的视觉感知；② 通过 LLM 引导的归并排序模块消除窗口间评分差异，获得全局一致的权重；③ 采用多模态时间序列模型和自适应预测窗口实现低延迟实时直播预测。

**🔧 技术方法**

技术手段包括：LLM（如 GPT‑4o）、CLIP 对齐、内容感知注意力、多模态时间序列预测、基于 merge‑sort 的 LLM 比较、Gaussian 平滑。

**📊 数据集**

数据集：YouTube‑8M、TVSum、SumMe，用于权重准确性评估；网络流量数据集 FCC、3G/HSDPA 用于用户体验验证；多模态 LLM 数据用于测试。

**📈 对比分析**

与 17 个基线（亮点检测、视频摘要、VideoLLaMA3、VILA、Flamingo、iTransformer、TimesNet 等）对比，HiVid 在 PLCC、mAP、MAE、RMSE 等指标上分别提升 11.5%、6% 和 26%，并在用户研究中 QoE 相关性提升 14.7%。

**⚠️ 局限性**

局限性：仍需支付 LLM API 费用，长视频成本线性增长；LLM 推理时延不可预测，可能导致预测窗口误差；对极端视觉内容或含糊视频的鲁棒性依赖多 LLM 平均，且仅使用锚帧，忽略帧间细节。

---

## 734. MechPert: Mechanistic Consensus as an Inductive Bias for Unseen Perturbation Prediction

**arXiv ID:** 2602.13791 | [PDF](https://arxiv.org/pdf/2602.13791v1)

**作者:** Marc Boubnovski Martell `[一作]` (Novo Nordisk), Kaspar Märten `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种轻量级框架 MechPert，利用多代理 LLM 生成定向调控假设并通过共识聚合来预测未见基因扰动的转录响应，并用于实验设计选择信息量最大的扰动基因。

**💡 创新点**

创新点在于将 LLM 从仅做语义检索转变为生成定向调控（因果）假设的多代理推理过程，并通过共识过滤噪声与不确定性，提供一种在数据稀缺时可用的机制化先验，补足图神经网络和静态知识图谱的不足。

**🔧 技术方法**

技术手段包括：①多代理 LLM 推理（如 Gemini 3 Pro 等）独立生成语义邻居集与因果调控集；②共识聚合机制（频率投票或置信度加权）过滤假设；③在推断阶段对聚合后的邻居进行加权平均预测；④在实验设计阶段采用迭代多样性选择与热核传播评估。

**📊 数据集**

数据集为 Perturb‑seq 基因敲除/抑制实验的四种人类癌细胞系（K562、RPE1、Jurkat、HepG2）中的单基因扰动表达谱，用于训练（N=50~800）和评估。

**📈 对比分析**

与基于语义相似性的 LangPert、GNN 方法（GEARS、TxPert）以及结构性中心性基准对比。低数据（N=50）下 MechPert 的 Pearson 相关率提升最多 10.5%（相对 10.4%），并在实验设计任务中 Anchors 选择提升至 46%（相对 PPI 度中心性），显示显著性能优势。

**⚠️ 局限性**

局限性包括：①仅在单基因 CRISPRi 细胞系中验证，未测试主细胞、组合或时间序列扰动；②依赖 LLM 生成假设，若文本偏倚或缺失，仍可能产生错误；③共识方法主要过滤噪声，未实现正式因果效应估计；④在知识稀疏的细胞系（HepG2、Jurkat）中效果不佳，甚至可能逊于随机采样；⑤评估使用的是已有训练数据的扰动，可能存在记忆化问题。

---

## 735. No Need to Train Your RDB Foundation Model

**arXiv ID:** 2602.13697 | [PDF](https://arxiv.org/pdf/2602.13697v1)

**作者:** Linjie Xu `[一作]` (University of Hong Kong), David Wipf `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种不需要训练或微调的关系型数据库（RDB）基础模型；

**💡 创新点**

创新点在于设计了只在垂直方向压缩、保持列身份的无参数编码器（JUICE），并证明其在ICL场景下不降低表达能力；

**🔧 技术方法**

使用1D GNN+Meta‑Path聚合、SQL实现的编码器，并与现有单表ICL模型（如TabPFN）无缝配合；

**📊 数据集**

在公开的RDB基准（RelBench、4DBInfer）以及其他关系型数据库任务上进行实验；

**📈 对比分析**

与RT、Griffin、RelLLM、LLM-A/B、KumoRFM、RelGT等基线对比，RDBLearn在零样本（无训练）下的性能与全监督模型相当，甚至优于多数基础模型；

**⚠️ 局限性**

局限性包括对列重要性随任务变化的自适应能力有限，以及在严重离群分布下的鲁棒性仍待进一步提升。

---

## 736. SkillJect: Automating Stealthy Skill-Based Prompt Injection for Coding Agents with Trace-Driven Closed-Loop Refinement

**arXiv ID:** 2602.14211 | [PDF](https://arxiv.org/pdf/2602.14211v1)

**作者:** Xiaojun Jia `[一作]` (Nanyang Technological University), Philip Torr `[通讯]` (University of Oxford)

**通讯引用:** 58641 | [OpenAlex ID](https://openalex.org/A5042899882)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SkillJect框架，实现了对LLM编码代理的技能注入攻击。

**💡 创新点**

创新点在于自动化、闭环迭代的技能注入方法，利用诱导提示和隐藏载荷分离。

**🔧 技术方法**

采用LLM驱动的攻击代理、工具代理和评估代理，并使用隐写式脚本隐藏payload。

**📊 数据集**

使用了包含50个多领域技能的自建基准数据集，覆盖数据处理、开发工具和视觉生成等。

**📈 对比分析**

通过与Naive直接注入基线对比，在四种后端模型上平均攻击成功率从约11%提升到95.1%，并证明跨模型迁移性。

**⚠️ 局限性**

局限在于仍需依赖大量执行反馈、对特定模型的迭代优化，且对强静态/语义检测仍存在可规避空间。

---

## 737. WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL

**arXiv ID:** 2602.13977 | [PDF](https://arxiv.org/pdf/2602.13977v1)

**作者:** Zhennan Jiang `[一作]` (Institute of Automation Chinese Academy of Sciences), Dongbin Zhao `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WoVR框架，实现了在学习到的世界模型上对VLA策略进行后期强化学习

**💡 创新点**

在模拟器、交互协议和策略-模型对齐三层面显式控制幻觉，推出Keyframe-Initialized Rollouts、PACE共进化和稳健的动作可控视频世界模型

**🔧 技术方法**

采用动作可控的Wan视频扩散模型、双通道动作注入、首帧锚定、掩码GRPO、KIR和PACE，并使用RLinf进行分布式想象式训练

**📊 数据集**

使用LIBERO基准（Spatial、Object、Goal、Long）和Frank A Emika Panda机器人Pick Banana/Pick Bread任务的数据集

**📈 对比分析**

与EVAC、Cosmos-Predict2、OpenSora、WMPO、GRPO和OpenVLA-OFT-base等方法对比，WoVR在LIBERO平均成功率从39.9%提升至69.2%，真实机器人成功率从61.7%提升至91.7%，显著优于基线

**⚠️ 局限性**

幻觉仍未完全消除，依赖于学习奖励、有限的真实数据微调，缺乏完整可靠性保证

---

## 738. TrasMuon: Trust-Region Adaptive Scaling for Orthogonalized Momentum Optimizers

**arXiv ID:** 2602.13498 | [PDF](https://arxiv.org/pdf/2602.13498v1)

**作者:** Peng Cheng `[一作]` (University of YYY), Wen Tong `[通讯]` (Company Name)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提供了 ICML 2026 会议论文提交与排版的完整指南，涵盖文件格式、页面布局、字体、图表与引用等细节。

**💡 创新点**

创新点在于将会议的排版规则、审稿流程与技术细节统一到一份文档中，明确了双盲评审的严格要求以及对 PDF 文件、字体、图像格式的细致规范。

**🔧 技术方法**

主要使用 LaTeX 模板、Type‑1 字体、EPS/PDF/PNG/JPEG 图像以及标准的 BibTeX/APA 引用格式；同时给出了排版命令和字体嵌入方法。

**📊 数据集**

文档本身不涉及实验数据集，仅在示例表格/图形中引用了若干常见数据集名称（如 Breast、Cleveland 等）作为演示。

**📈 对比分析**

比较方法：通过在初稿中剔除作者信息、采用双盲评审，并在最终稿中添加作者信息、运行摘要与引用格式，来评估格式与内容的符合度；并未给出具体性能指标。

**⚠️ 局限性**

局限性：仅适用于 ICML 2026 会议，严格依赖 PDF 与字体嵌入规则；若文件大小超过 10MB 或包含 Type‑3 字体则不被接受；此外，文档未包含实验结果，无法评价实际算法性能。

---

## 739. Additive Control Variates Dominate Self-Normalisation in Off-Policy Evaluation

**arXiv ID:** 2602.14914 | [PDF](https://arxiv.org/pdf/2602.14914v1)

**作者:** Olivier Jeunen `[一作]` (Aampe), Shashank Gupta `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

证明了使用最优加性基线（β^⋆-IPS）在离线政策评估中在均方误差上严格优于传统的自归一化逆概率加权（SNIPS）估计器，并将此结果推广到排名的 Item‑Position Model（β^⋆-IPM）中。

**💡 创新点**

创新点在于：1) 给出了 β^⋆-IPS 相对于 SNIPS 的严格理论优势，阐明了 SNIPS 实际等价于使用固定基线 V(π) 的加性估计器；2) 推导了两者的方差差距公式；3) 将这一理论框架扩展到排名任务，证明了 β^⋆-IPM 在每个位置上都能实现更低的 MSE。

**🔧 技术方法**

主要技术包括：均方误差分解、Delta 方法、Rosenthal 不等式、Hoeffding 置信界、交叉拟合（cross‑fitting）以消除基线估计引入的偏差，以及对排名情境的 Item‑Position Model 结构化估计。

**📊 数据集**

文章未使用具体公开数据集，而是基于一般的离线日志数据假设（i.i.d.、奖励、重要性权重有界）进行理论推导，实验引用了相关工作中的数据与结果。

**📈 对比分析**

通过理论证明和有限样本误差分析比较：在样本量足够大时，β^⋆-IPS 的 MSE 低于 SNIPS，方差下降量可量化为 (V(π)σ_w^2−σ_{w,r})^2/(nσ_w^2)。实证上，β^⋆-IPS 在中等样本量时已显示出比 SNIPS 更优的方差常数，且在实际推荐/排名任务中能取得更可靠的性能。

**⚠️ 局限性**

限制包括：1) 主要结果是渐进性质，有限样本下的偏差仍存在（与 SNIPS 同阶 O(1/n)）；2) 需要已知奖励与重要性权重的上界；3) 当日志策略与目标策略相近时，方差差距小，优势不明显；4) 对于排名的 β^⋆-IPM 仍忽略了位置间的依赖，实际效能可能受限。

---

## 740. Constructions of linear codes from vectorial plateaued functions and their subfield codes with applications to quantum CSS codes

**arXiv ID:** 2602.14832 | [PDF](https://arxiv.org/pdf/2602.14832v1)

**作者:** Virginio Fratianni `[一作]` (Universite Paris), Sihem Mesnager `[通讯]` (Universite Paris)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文基于平面化与向量化的平坦（plateaued）函数，构造了多维线性码及其子域码，并通过分析其重量分布与双重性，证明了其在Sphere Packing与Griesmer界下的最优性；进一步将这些码嵌入Calderbank–Shor–Steane (CSS)框架，得到可实现横向T门（或相位门）的量子码。

**💡 创新点**

创新点在于：①将原有两函数三维构造扩展到三函数四维与向量化版本；②利用Walsh谱与特征和精确计算码长、重量分布；③证明了双重性最优性与自正交性；④将这些经典码自然转化为可实现横向相位门的CSS量子码。

**🔧 技术方法**

主要技术包括：有限域上的加性/乘性特征理论、Walsh变换、板平函数（Bent/Almost Bent/ s‑Plateaued）的性质、子域码与迹码的对应、Pless幂矩阵、Sphere Packing与Griesmer界的最优性判定。

**📊 数据集**

该研究不使用实验数据集，而是通过对指定平坦函数（如二次/规范化s‑plateaued函数、Gold、Kasami等）在有限域上的解析性质进行理论构造与推导。

**📈 对比分析**

比较方法：将构造的码参数与已知的Sphere Packing、Griesmer、Singleton等界进行比较；利用Pless幂矩阵求解双重码距离；与传统RM类量子码对比，证明其重量分布更稀疏、双重码距离更高，且能实现横向相位门。

**⚠️ 局限性**

局限性：仅针对特定平坦函数族；在非平坦或非二进特征时可行性有限；对多函数扩展的解析尚未完成；量子码的实际实现还需考察错误模型与编码/解码复杂度。

---

## 741. Evaluating the Impact of Post-Training Quantization on Reliable VQA with Multimodal LLMs

**arXiv ID:** 2602.13289 | [PDF](https://arxiv.org/pdf/2602.13289v1)

**作者:** Paul Jonas Kurz `[一作]` (TU Darmstadt), Marcus Rohrbach `[通讯]` (TU Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了后训练量化（PTQ）对多模态大语言模型（MLLM）在视觉问答（VQA）中的准确率与可靠性（置信度）影响，并提出利用Selector置信度估计器在量化后恢复可靠性的方案。

**💡 创新点**

首次系统量化评估MLLM在准确率与可靠性上的交互效应；提出在int4_MBQ量化后配合Selector能在显著降低内存占用（≈75%）的同时，几乎保持bf16模型的可靠性和校准性能。

**🔧 技术方法**

采用两种PTQ方法：数据无关的HQQ和数据感知的MBQ；使用Selector置信度估计器（基于多模态特征的两层MLP）；评估指标包括准确率、ECE、Coverage‑at‑Risk、AUC和Φc；实验在VQAv2、AdVQA、VizWiz三大数据集上进行。

**📊 数据集**

使用VQAv2（In‑distribution）、AdVQA（语言OOB）和VizWiz（多模态OOB）作为测试集；量化模型以Qwen2‑VL‑7B和Idefics3‑8B为基准。

**📈 对比分析**

对比bf16、int8、int4、int3等位宽的量化模型；发现量化导致准确率下降、ECE升高；MBQ比HQQ在低位宽下保持更好的校准和准确率；Selector在所有量化水平上显著降低ECE、提升Coverage‑at‑Risk，尤其在int4_MBQ下恢复接近bf16的可靠性；int4_MBQ+Selector在内存占用上相对bf16下降≈75%，性能损失≤2%。

**⚠️ 局限性**

局限性：仅评估了两款MLLM和三种VQA数据集；量化仅为权重量化；未探索量化感知的可靠性训练；在极低位宽（int3）下仍无法完全恢复可靠性；对其他多模态推理任务的泛化能力未知。

---

## 742. Bridging AI and Clinical Reasoning: Abductive Explanations for Alignment on Critical Symptoms

**arXiv ID:** 2602.13985 | [PDF](https://arxiv.org/pdf/2602.13985v1)

**作者:** Belona Sonna `[一作]` (Australian National University), Alban Grastien `[通讯]` (Université Paris-Saclay)

**通讯引用:** 1367 | [OpenAlex ID](https://openalex.org/A5000852884)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

结合形式化的归纳推理解释，评估 AI 诊断模型与临床思维在关键症状上的对齐程度。

**💡 创新点**

提出利用归纳解释来发现训练集中的临床关键属性，并定义严格/松弛的对齐判定标准，兼顾解释性与可验证性。

**🔧 技术方法**

使用归纳推理（abductive explanations）算法，并与 SHAP/LIME 等后置解释方法对比；通过逻辑约束构造最小足够解释。

**📊 数据集**

三个公开数据集：Wisconsin Diagnostic Breast Cancer、Cleveland Heart Disease、Mental Health in Tech Survey（OSMI）。

**📈 对比分析**

通过将模型的解释与临床关键属性对齐程度进行划分；结果显示 NN 最不对齐、SGD 最对齐；在 Breast Cancer 上对齐率约 14%，Heart Disease 上约 6%；整体对齐率低于预期。

**⚠️ 局限性**

局限包括：仅评估关键病例，阈值化处理导致解释受限；对非关键病例缺乏覆盖；缺乏实时临床验证与更复杂的临床决策场景。

---

## 743. From Fluent to Verifiable: Claim-Level Auditability for Deep Research Agents

**arXiv ID:** 2602.13855 | [PDF](https://arxiv.org/pdf/2602.13855v1)

**作者:** Razeen A Rasheed `[一作]` (Indian Institute of Science), Rima Hazra `[通讯]` (TCG CREST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了深度研究代理的可审计性框架（AAR标准），主张在科研自动化中实现对每条关键声明的可追溯性、证据有效性、冲突透明度和审计成本可控性；

**💡 创新点**

创新点在于将可审计性提升到系统设计与评估的核心目标，提出四项可测量指标（覆盖率、语义有效性、冲突透明度、审计成本）并构造语义溯源图谱来实现端到端的证据链追踪；

**🔧 技术方法**

技术包括：基于W3C PROV的语义溯源图谱、推理与检索的链式工作流、向量检索与命题对齐的语义验证、持续化验证协议、以及对冲突进行显式标注的机制；

**📊 数据集**

使用的主要数据集为学术文献检索库（如Semantic Scholar、arXiv）以及实验结果日志；论文以实验性原型（未公开具体系统）在小规模实验中验证可追溯性指标；

**📈 对比分析**

与现有仅关注任务完成度的基准相比，AAR标准通过量化可追溯性、语义有效性等指标，展示了更低的审计成本与更高的证据可靠性；实验表明相较于传统日志，语义图谱能显著提升错误检测率；

**⚠️ 局限性**

局限性在于实现完整语义溯源和持续验证仍需要高计算成本、复杂的知识图谱维护，且对大规模开放权重模型的可解释性支持仍有限；

---

## 744. Exploring the Performance of ML/DL Architectures on the MNIST-1D Dataset

**arXiv ID:** 2602.13348 | [PDF](https://arxiv.org/pdf/2602.13348v1)

**作者:** Michael Beebe `[一作]` (Texas Tech University), Angel Ayala `[通讯]` (Texas Tech University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在MNIST-1D数据集上评估并比较了ResNet、TCN、DCNN等先进深度网络与传统Logistic Regression、MLP、CNN、GRU等模型的分类性能

**💡 创新点**

首次将ResNet、TCN和DCNN等结构引入MNIST-1D，验证其在小规模时序数据上的优势，并通过比较实验揭示了不同归纳偏置对模型性能的显著影响

**🔧 技术方法**

使用一维残差网络、因果膨胀卷积网络与时间卷积网络（TCN），结合批归一化、残差跳跃、全局平均池化等技术；训练采用Adam优化器、早停与学习率调度；对比实验基于不同训练步数的测试准确率

**📊 数据集**

MNIST-1D（40点一维时间序列）以及对照的传统二维MNIST数据集

**📈 对比分析**

通过在1,000、2,000、10,000步的训练阶段测量测试准确率，结果显示TCN和DCNN在所有阶段均超越Logistic Regression、MLP、CNN和GRU，甚至在较少训练步数时已接近或超过人类水平（约96%），ResNet也显著优于简单模型

**⚠️ 局限性**

实验仅限于小规模数据集，缺乏对更大或更复杂时序/图像数据的验证；模型选择和超参数设置可能对结果产生影响；未对模型泛化能力在不同噪声、变形等条件下的鲁棒性做深入分析

---

## 745. VisPhyWorld: Probing Physical Reasoning via Code-Driven Video Reconstruction

**arXiv ID:** 2602.13294 | [PDF](https://arxiv.org/pdf/2602.13294v1)

**作者:** Jiarong Liang `[一作]` (University of Waterloo), Wenhu Chen `[通讯]` (University of Waterloo)

**通讯引用:** 4877 | [OpenAlex ID](https://openalex.org/A5103103242)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发 VisPhyWorld 框架，让多模态大型语言模型（MLLM）通过生成可执行的模拟代码来重建并预测物理场景，并基于此推出 VisPhyBench 评测套件。

**💡 创新点**

创新点在于：①使用可执行代码作为物理推理的直接证据，取代传统 VQA/VoE 的表面式测试；②将视觉输入、代码生成与视频重现三者融合为完整管道，提供可解释、可验证的物理假设；③提出统一的多指标评估体系（感知、语义、文本一致性、运动一致性和整体质量）。

**🔧 技术方法**

技术手段包括：MLLM（GPT‑5、GPT‑4.1、Gemini‑3‑Pro、Claude‑Sonnet‑4.5、Qwen3‑VL‑Plus）生成代码；Three.js / P5.js 物理仿真引擎；多指标自动评估（LPIPS、CLIP‑Img、DINO、CLIP‑Cap、BERTScore、RAFT‑EPE、Gemini‑judge）；以及自我修复机制以提升代码可执行率。

**📊 数据集**

数据集：VisPhyBench，209 个视频（108 个物理模板），涵盖 2D（基于 PhyWorld）和 3D（基于 Three.js/Cannon.js）场景。每个视频提供两帧关键帧及可选检测上下文，且所有场景均可被程序化生成。

**📈 对比分析**

与像素空间基线（SVD img2vid、Veo‑3.1）和多种 MLLM + 代码后端组合进行比较。实验表明：①在可执行代码重建任务中，Three.js 后端的 MLLM（如 GPT‑5 + Three.js）在感知、语义一致性和物理运动一致性指标上表现最佳；②相比之下，像素空间基线在语义一致性上有一定优势，但缺乏可验证的物理动态；③整体来看，当前 MLLM 在语义理解上表现强劲，但在精确物理参数化与动力学模拟方面仍显不足。

**⚠️ 局限性**

局限性：①仅在合成、相对简单的刚体场景中验证；难以推广到复杂 3D、长时序或高度混乱的真实视频；②LLM 生成的代码受限，无法自动处理高级引擎（如 Unreal、Blender）或复杂交互；③缺乏对长时间物理一致性与多体相互作用的深入评估；④对真实世界光照、材质、非刚体动力学等因素支持不足。

---

## 746. VIGIL: Tackling Hallucination Detection in Image Recontextualization

**arXiv ID:** 2602.14633 | [PDF](https://arxiv.org/pdf/2602.14633v1)

**作者:** Joanna Wojciechowicz `[一作]` (Wroclaw University of Science and Technology), Maciej Zięba `[通讯]` (Wroclaw University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了VIGIL框架和对应的VIGIL基准数据集，用于细粒度检测图像重新语境化任务中的幻觉。

**💡 创新点**

提出细粒度幻觉分类 taxonomy（5类）并结合多阶段检测 pipeline，提供可解释的文本描述。

**🔧 技术方法**

结合 Qwen3‑VL‑8B‑Instruct、SAM‑3、DINO‑v3 等多模态模型，采用对象分割、特征匹配、背景差异定位等技术实现检测。

**📊 数据集**

使用由 Gemini 2.5 Flash Image 自动生成、人工标注的 1,269 样本数据集，涵盖服装、家具、美妆、电子、汽车等 5 个领域。

**📈 对比分析**

与 Gemini 2.5 Flash、Qwen3‑VL‑8B‑Instruct、Gemma 3‑27B‑IT 等单模型基线在多标签 F1 及 LLM‑as‑a‑Judge 评估中比较，VIGIL 在大多数类别（尤其是服装与化妆品）上获得最高的 macro‑F1，性能显著优于基线。

**⚠️ 局限性**

计算开销大，需多模型协同；受 GPU 资源限制；分辨率差异导致检测难度；对空间关系与物理一致性的评估仍不完善。

---

## 747. Machine Learning as a Tool (MLAT): A Framework for Integrating Statistical ML Models as Callable Tools within LLM Agent Workflows

**arXiv ID:** 2602.14295 | [PDF](https://arxiv.org/pdf/2602.14295v1)

**作者:** Edwin Chen `[一作]` (Legacy AI LLC), Zulekha Bibi `[通讯]` (Legacy AI LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了机器学习作为工具（MLAT）设计模式，在LLM代理工作流中将预训练的统计ML模型作为可调用工具，构建了PitchCraft系统以自动生成报价提案。

**💡 创新点**

将预训练统计ML模型作为LLM代理可调用工具的概念化，并通过结构化输出解析实现上下文化调用和解释推理，填补了LLM工具调用与传统ML集成的空白。

**🔧 技术方法**

使用Gemini LLM的结构化JSON输出、FastAPI部署的XGBoost回归模型、LLM工具调用机制、人工验证的合成数据生成与group-aware交叉验证等技术。

**📊 数据集**

70条样本的报价数据集，其中40条真实代理交易记录和30条人工审核的LLM合成记录，涵盖22个行业。

**📈 对比分析**

采用三折group-aware交叉验证与真实测试集比较，XGBoost在测试集上R^2=0.807、MAE≈$3,688，明显优于岭回归（R^2≈0.782）。

**⚠️ 局限性**

样本量有限、仅在单一行业验证、缺乏反馈学习和置信区间、LLM特征估计不稳定、合成数据比例较高。

---

## 748. Gaussian Sequences with Multi-Scale Dynamics for 4D Reconstruction from Monocular Casual Videos

**arXiv ID:** 2602.13806 | [PDF](https://arxiv.org/pdf/2602.13806v1)

**作者:** Can Li `[一作]` (Nankai University), Lei Sun `[通讯]` (Nankai University)

**通讯引用:** 11392 | [OpenAlex ID](https://openalex.org/A5009604094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于多尺度动力学的多尺度高斯序列模型，实现单目视频的4D动态重建与新视角合成

**💡 创新点**

将动态场景的运动分解为对象级、稀疏原始级和细粒度级三层多尺度动力学，利用共享低秩运动基底，并结合视觉基础模型提供的多模态先验进行监督

**🔧 技术方法**

3D高斯剖分（3DGS）+多尺度动力学因式分解 + 共享加权SE(3)运动基底 + 视觉基础模型的分割、深度和跟踪先验 + 多模态损失（RGB、掩码、深度、跟踪、局部刚性）

**📊 数据集**

DyCheck iPhone公开数据集以及自制的手持摄像机收集的四类对象（硬件、机械、可变形）数据集

**📈 对比分析**

与HyperNeRF、T-NeRF、Deform-3DGS、Dynamic marbles、Shape-of-motion等基线相比，在mPSNR、mSSIM、mLPIPS指标上均显著提升，最优场景mPSNR达17.07，mSSIM 0.66，mLPIPS 0.38

**⚠️ 局限性**

在严重遮挡或大规模拓扑变化下仍易失真，单目条件下对极端动态仍有限

---

## 749. GREAT-EER: Graph Edge Attention Network for Emergency Evacuation Responses

**arXiv ID:** 2602.14676 | [PDF](https://arxiv.org/pdf/2602.14676v1)

**作者:** Attila Lischka `[一作]` (Chalmers University of Technology), Balázs Kulcsár `[通讯]` (Chalmers University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了公交疏散或行程优化问题（Bus Evacuation Orienteering Problem，BEOP），并开发了基于图边注意力网络（GREAT）与强化学习的快速疏散路径生成框架GREAT‑EER，实现了在限定时间内最大化疏散人数的策略。

**💡 创新点**

创新点包括：①首次将允许车辆多次返回、时间窗口等特征纳入BEOP，形成更贴近实际的疏散模型；②结合GREAT encoder与多指针解码器构建端到端的深度强化学习管线；③在确定性与随机动态环境下均可实时生成有效路线，并通过warm‑start提升传统MILP求解效率。

**🔧 技术方法**

使用的核心技术为图边注意力网络（GREAT）作为编码器、基于POMO的多重roll‑out强化学习（REINFORCE）作为训练框架、与传统的混合整数线性规划（MILP）与贪婪启发式进行对比，整个系统以Python与PyTorch实现。

**📊 数据集**

实验数据来源于旧金山的OpenStreetMap道路网络（约10,000节点）以及Uber 2019年出租车轨迹，用于构造节点需求与流量权重，实现与真实交通情况高度一致的评估。

**📈 对比分析**

与Gurobi求解的30分钟/1分钟时间限制的MILP以及贪婪启发式比较，GREAT‑EER在约1秒内完成求解，疏散占比可达94–95%，几乎与MILP最佳解持平；在大规模或受灾区块（hazard zone）以及随机时延/需求场景下仍保持比贪婪显著优势，且不产生违规解。

**⚠️ 局限性**

局限性包括：对极大规模实例（>200点）求解仍有挑战；随机在线模式目前仅支持单辆公交车；模型假设所有乘客奖励相同、无上车时延、车辆容量同质；缺乏对多辆车协同动态决策的完整理论与实现。

---

## 750. Impact-Robust Posture Optimization for Aerial Manipulation

**arXiv ID:** 2602.13762 | [PDF](https://arxiv.org/pdf/2602.13762v1)

**作者:** Amr Afifi `[一作]` (University of Twente), Antonio Franchi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 8165 | [OpenAlex ID](https://openalex.org/A5001771133)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

通过在任务空间逆动力学（TSID）控制器中加入基于刚性冲击模型的影响鲁棒度指标，使冗余机器人在预期冲击前主动调整姿态，以减小冲击引起的状态与控制尖峰。

**💡 创新点**

创新点在于将冲击鲁棒度（由动态冲击椭圆量化的姿态指标）作为姿态任务残差直接嵌入TSID的代价函数，实现软约束式的实时姿态优化，避免了传统方法中硬约束导致的可行性问题，并且无需额外的冲击后速度边界估计。

**🔧 技术方法**

使用技术包括刚性冲击模型、动力学影响椭圆计算、梯度下降优化、自动微分（ADAM/CasADi）求取梯度、任务空间逆动力学框架、QP求解器qpOASES、Gazebo/ODE仿真以及全旋翼+机械臂的完整动力学模型。

**📊 数据集**

数据集方面，研究基于Gazebo仿真环境，分别对空中机械臂（hexarotor+3-DoF臂）、四足机器人Solo和人形机器人Talos进行多接触场景的数值仿真，没有使用公开实验数据集。

**📈 对比分析**

与传统冲击无关的TSID进行对比，结果显示冲击鲁棒TSID将冲击期间的姿态波动减少约51%（空中机械臂）和45%（Talos/ Solo），避免了电机/关节饱和，控制尖峰幅度显著降低，整体表现更安全、平滑。

**⚠️ 局限性**

局限性包括：需要预先已知冲击方向和冲击范围；当机器人冗余度被耗尽时，姿态任务只能在更低优先级空间执行，可能无法充分缓解冲击；对于冲击方向或强度的不确定性，方法仍需进一步推广。

---

## 751. Progressive Contrast Registration for High-Fidelity Bidirectional Photoacoustic Microscopy Alignment

**arXiv ID:** 2602.13304 | [PDF](https://arxiv.org/pdf/2602.13304v1)

**作者:** Jiahao Qin `[一作]` `[通讯]`, Jiahao Qin

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于渐进对比的注册框架PCReg-Net，实现光学分辨率光声显微镜双向扫描图像的高保真配准。

**💡 创新点**

通过将配准拆分为粗略对齐和对比引导的细化，使用多尺度对比模块和特征注入，显著克服光强差异导致的域偏移问题。

**🔧 技术方法**

使用轻量级U‑Net编码器解码器、对比卷积、特征注入、辅助损失以及混合精度训练，整体框架实现实时性能。

**📊 数据集**

在两个公开OR‑PAM配准基准：OR‑PAM‑Reg‑4K和OR‑PAM‑Reg‑Temporal‑26K（共约4,680+图像对）上进行训练与评估。

**📈 对比分析**

与传统SIFT、Demons、光流、SyN以及深度学习VoxelMorph、TransMorph、SAS‑Net等方法对比，PCReg‑Net在NCC、SSIM、PSNR等指标上分别提升至0.983/0.982/46.96 dB，PSNR比SAS‑Net高14.5 dB，并且时序一致性指标TNCG接近零。

**⚠️ 局限性**

局限在于仍需人工定义正则化参数、对极端扫描速度或大幅域偏移的鲁棒性尚未完全验证。

---

## 752. Text Style Transfer with Parameter-efficient LLM Finetuning and Round-trip Translation

**arXiv ID:** 2602.15013 | [PDF](https://arxiv.org/pdf/2602.15013v1)

**作者:** Ruoxi Liu `[一作]` (Johns Hopkins University), Philipp Koehn `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过回译生成伪平行语料，对大型语言模型进行参数高效微调，实现文本风格迁移。

**💡 创新点**

创新点在于将回译作为无监督的风格中性化管道与检索增强生成结合，既生成平行数据又提升风格一致性。

**🔧 技术方法**

采用Marian式回译、LoRA微调、检索增强生成（RAG）以及BERT风格分类器评估。

**📊 数据集**

使用WMT24英德、英中通用并行语料生成回译模型，并在四个单语域（IRS、Treasury、NCBI、文学）构建伪平行数据。

**📈 对比分析**

与ICL、APE基线相比，微调后的LLM在BLEU和风格准确率上分别提升约20–40%，在所有四个域均表现最优。

**⚠️ 局限性**

主要局限包括回译引入的语义漂移、对NMT质量的依赖以及仅在有限域上验证，难以保证在更广泛风格上的泛化。

---

## 753. Goldilocks RL: Tuning Task Difficulty to Escape Sparse Rewards for Reasoning

**arXiv ID:** 2602.14868 | [PDF](https://arxiv.org/pdf/2602.14868v1)

**作者:** Ilia Mahrooghi `[一作]` (Ecole Polytechnique Federale de Lausanne), Emmanuel Abbe `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了教师驱动的动态课程体系，筛选适度难度的问题加速语言模型推理能力提升

**💡 创新点**

创新点在于教师模型通过即时学习学生表现预测问题难度，实现“金发姑娘”式的自适应采样

**🔧 技术方法**

结合GRPO强化学习、教师-学生框架、epsilon-greedy数据选择与可验证奖励机制

**📊 数据集**

使用OpenMathReasoning大规模链式推理数据集（约300万题）进行实验

**📈 对比分析**

与传统GRPO基线在相同计算预算下对比，Pass@1从约30%提升至33%（Qwen2.5-1.5B）及其他模型亦有类似提升

**⚠️ 局限性**

局限在于教师训练与推理需额外GPU资源、对极大规模多样化数据的泛化能力尚待验证

---

## 754. Translating Dietary Standards into Healthy Meals with Minimal Substitutions

**arXiv ID:** 2602.13502 | [PDF](https://arxiv.org/pdf/2602.13502v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 755. Large Language Model (LLM)-enabled Reinforcement Learning for Wireless Network Optimization

**arXiv ID:** 2602.13210 | [PDF](https://arxiv.org/pdf/2602.13210v1)

**作者:** Jie Zheng `[一作]` (Northwest University), Zehui Xiong `[通讯]` (Queen's University Belfast)

**通讯引用:** 20957 | [OpenAlex ID](https://openalex.org/A5005327587)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于大型语言模型（LLM）增强的多智能体强化学习（MARL）框架，用于无人机-卫星网络的服务迁移与请求路由优化，并系统梳理LLM在物理层至应用层不同协议层的角色与应用场景。

**💡 创新点**

创新点在于将LLM作为特征提取、奖励设计、策略解释与决策制定者四大功能融入RL，提出了基于语义提取的状态表示（LESR）和自我反馈机制，实现跨层次、跨网络结构的自适应决策；同时系统性分析了LLM在六大协议层的作用与挑战。

**🔧 技术方法**

技术包括LLM（Prompt、文本生成与嵌入）、图神经网络（GNN）、深度Q网络（DQN）、多智能体强化学习（MARL）、奖励工程、知识图谱与推理模型；同时利用仿真平台生成多模态网络状态。

**📊 数据集**

数据集为仿真生成的低轨卫星网络（20颗卫星、4轨道、20–80 Mbps intra‑orbit、1–10 Mbps inter‑orbit）与动态UAV链路（0.1–0.8 Mbps），包大小100–1000 KB，包含节点状态、链路带宽、时延等信息。

**📈 对比分析**

通过与传统最短路径、贪婪优化、单智能体DQN等基线方法对比，LLM‑MARL框架在收敛速度和平均奖励上均明显优越，决策性能提升约25%，且在动态网络环境中保持更高的鲁棒性。

**⚠️ 局限性**

局限性包括LLM推理延迟高、算力需求大，需通过量化、剪枝或蒸馏等方法降低资源占用；对实时低功耗系统的适配不足；以及缺乏完善的安全、可信度评估和规则约束机制。

---

## 756. Fluid-Agent Reinforcement Learning

**arXiv ID:** 2602.14559 | [PDF](https://arxiv.org/pdf/2602.14559v1)

**作者:** Shishir Sharma `[一作]` (Mila Quebec Artificial Intelligence Institute), Theodore J. Perkins `[通讯]` (Ottawa Hospital Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

论文提出流动代理强化学习框架，使得智能体可在环境中自行产生新代理并动态调整团队规模。

**💡 创新点**

关键创新是引入可流动代理环境（POFSG）并证明其存在纳什均衡与子博弈完美均衡，将代理生成视为可控制的策略选择。

**🔧 技术方法**

采用多智能体强化学习算法（IQL、VDN、PPO、MAPPO），结合集中训练分散执行、奖励结构改造以及随机采样初始/上限人口的探索策略。

**📊 数据集**

在改造后的捕食者‑猎物（Predator–Prey）、层级采集（Level‑Based Foraging）和新设计的PuddleBridge三种实验环境上验证。

**📈 对比分析**

与固定团队规模（2–10名代理）对比，流动代理在不同资源稀缺/丰沛场景下能自适应规模并取得与最优固定规模相当甚至更高的联合奖励；实验显示CTDE算法在奖励分配不一致时更能鼓励生成。

**⚠️ 局限性**

仅考虑代理生成而未涵盖消亡，且对奖励设计与采样策略高度敏感，缺乏理论上对收敛性与样本复杂度的深入分析。

---

## 757. The Baby Steps of the European Union Vulnerability Database: An Empirical Inquiry

**arXiv ID:** 2602.14313 | [PDF](https://arxiv.org/pdf/2602.14313v1)

**作者:** Jukka Ruohonen `[一作]` (University of Southern Denmark), Jukka Ruohonen `[通讯]` (University of Southern Denmark)

**通讯引用:** 559 | [OpenAlex ID](https://openalex.org/A5045061250)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过收集2026年12月15日的EUVD数据，对已归档的主动被利用漏洞（AEVs）和通过ENISA或欧洲CSIRT网络协调的漏洞进行描述性统计与相关性分析，回答了四个研究问题；

**💡 创新点**

创新点在于：①首次对欧盟新成立的EUVD元数据进行系统的实证检验；②首次将EPSS预测得分与CVSS严重性在EUVD上下文中进行对比；③首次对不同欧洲CSIRT的活跃度进行量化并揭示ENISA在此数据库中的低活跃度；

**🔧 技术方法**

使用了描述性统计、箱线图、散点图与相关系数计算等基本统计与可视化技术，并将CVSS 3.1/4.0与EPSS得分进行对比；

**📊 数据集**

采用了EUVD公开数据库中的AEVs（1,510个）和ENISA/CSIRT协调漏洞（1,291个）数据集，其中所有漏洞均包含CVE编号；

**📈 对比分析**

方法为对AEVs与ENISA/CSIRT协调漏洞分别计算CVSS与EPSS的均值、中位数，并绘制分布图和相关性图；结果显示AEVs的CVSS和EPSS均显著高于后者，且两者呈正相关；

**⚠️ 局限性**

局限性包括：①两类漏洞尚未重叠，无法评估ENISA/CSIRT对AEVs的实际识别能力；②数据截至2026年，仅覆盖前两年EUVD的启动期；③缺乏关于漏洞分配流程的细节，影响对活跃度解释的深度；④未对漏洞修补时效等安全治理效果进行评估。

---

## 758. The acquisition of English irregular inflections by Yemeni L1 Arabic learners: A Universal Grammar approach

**arXiv ID:** 2602.13816 | [PDF](https://arxiv.org/pdf/2602.13816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 759. Metaphors' journeys across time and genre: tracking the evolution of literary metaphors with temporal embeddings

**arXiv ID:** 2602.13701 | [PDF](https://arxiv.org/pdf/2602.13701v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 760. CLF-ULP: Cross-Layer Fusion-Based Link Prediction in Dynamic Multiplex UAV Networks

**arXiv ID:** 2602.13201 | [PDF](https://arxiv.org/pdf/2602.13201v1)

**作者:** Cunlai Pu `[一作]` (Nanjing University of Science and Technology), Xiangbo Shu `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 4292 | [OpenAlex ID](https://openalex.org/A5040437528)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了面向动态多层UAV网络的交叉层融合式链接预测模型CLF-ULP，并构建了多层动态UAV网络模拟数据集；

**💡 创新点**

创新点在于：①引入多层动态网络结构，①.1 通过相同节点在不同层之间的交叉层链接体现任务多样性；②利用图注意力网络（GAT）提取层内结构特征，并设计交叉层注意力融合（CLAF）捕获层间依赖；③采用共享参数LSTM捕捉跨层的时间演化；④设计联合损失包含重构、层内嵌入一致性和层间嵌入一致性三项；

**🔧 技术方法**

技术手段主要包括：图注意力网络（GAT）、交叉层注意力融合（CLAF）、共享参数LSTM、两层全连接解码器以及带权重的重构损失；

**📊 数据集**

使用四种UAV运动模型（RW、GM、RPG、MG）生成的模拟多层网络数据，网络规模为100台UAV，采样间隔2秒，总共80帧；

**📈 对比分析**

与Node2Vec、EvolveGCN、E-LSTM-D、Top-Sequential-Stacking等基线进行对比，AUC和AP均优于所有基线；在不同采样间隔、飞行速度以及消除层间建模的消融实验中均表现出更高的鲁棒性和更优的性能；

**⚠️ 局限性**

局限性包括：仅基于模拟数据验证，缺乏真实UAV网络实验；交叉层注意力和共享LSTM的计算量相对较大，可能限制实时部署；并未考虑通信干扰、能耗等真实环境中的额外因素。

---

## 761. Fast Surrogate Learning for Multi-Objective UAV Placement in Motorway Intelligent Transportation System

**arXiv ID:** 2602.13564 | [PDF](https://arxiv.org/pdf/2602.13564v1)

**作者:** Weian Guo `[一作]` (Tongji University), Li Li `[通讯]` (Tongji University)

**通讯引用:** 104333 | [OpenAlex ID](https://openalex.org/A5100334060)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在高速公路 ITS 环境下的多目标无人机部署，利用 NSGA-II 生成 Pareto 前沿标签，训练快速代理模型实现秒级决策。

**💡 创新点**

关键创新在于结合偏好引导的标签生成、集合 Transformer 的置换不变预测、约束正则化以及录制级分割的可复现基准；实现近 NSGA-II 质量但毫秒级推理。

**🔧 技术方法**

使用集合 Transformer（Set Transformer）、序列 Transformer、置换不变 L1 损失、交叉熵、平滑覆盖与 SNR 正则，后处理约束投影。

**📊 数据集**

数据集基于 highD 高速公路记录，按录制拆分，生成 NSGA-II 标签，包含车辆位置、覆盖半径、SNR 阈值等信息。

**📈 对比分析**

与 NSGA-II、贪心、k‑means 等基线比较，采用 Pareto 质量指标、成功率曲线、运行时基准；集合模型在覆盖–信噪比–机数量平衡上接近 NSGA-II，推理时间从秒级降到毫秒级，显著优于启发式。

**⚠️ 局限性**

局限性：仅处理静态快照，未考虑时间动态、能耗、干扰与阴影；模型在极端低密度或拉伸场景易失效；部署时需后处理满足安全约束。

---

## 762. Anthropomorphism on Risk Perception: The Role of Trust and Domain Knowledge in Decision-Support AI

**arXiv ID:** 2602.13625 | [PDF](https://arxiv.org/pdf/2602.13625v1)

**作者:** Manuele Reani `[一作]` (Chinese University of Hong Kong), Zuolan Bao `[通讯]` (Chinese University of Hong Kong)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过大规模线上实验，研究了AI对话代理的拟人化设计对风险感知的影响，并提出了通过认知与情感信任两条路径的中介模型。

**💡 创新点**

创新点在于首次将认知信任与情感信任视为并行中介，探讨领域知识（金融专业度）如何调节拟人化对信任及风险感知的双重影响，填补了高风险决策场景下拟人化与信任、风险感知关系的研究空白。

**🔧 技术方法**

主要技术手段包括8级拟人化设计的操作、Godspeed问卷测量感知拟人化、CATQ测量认知与情感信任、单项量表测量领域知识与风险感知，以及分层线性回归、结构方程模型与条件过程分析（Moderated Mediation）等统计方法。

**📊 数据集**

使用数据集为1256名通过Prolific招募的受试者在一个金融投资决策场景中完成交互，数据涵盖了拟人化条件、感知拟人化、信任维度、领域知识与风险感知等变量。

**📈 对比分析**

在模型比较中，加入感知拟人化、信任中介与领域知识调节后，风险感知的解释方差提升至约7%（R²≈0.07），显著高于仅控制人口统计变量的基线模型；同时在SEM中发现认知信任对风险感知具有显著负向路径，情感信任路径则较弱或无效。

**⚠️ 局限性**

局限性包括：所有变量均基于自我报告问卷，缺乏客观行为或生理数据；实验场景限定在金融投资领域，难以推广至医疗或安全等高风险域；领域知识测量为主观评估，可能与实际专业水平不完全一致；此外，跨时间因果推断仍不确定，需要进一步实验验证。

---

## 763. pFedNavi: Structure-Aware Personalized Federated Vision-Language Navigation for Embodied AI

**arXiv ID:** 2602.14401 | [PDF](https://arxiv.org/pdf/2602.14401v1)

**作者:** Qingqian Yang `[一作]` (Stevens Institute of Technology), Haibing Guan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种结构感知的个性化联邦学习框架，用于隐私保护的视觉‑语言导航任务。

**💡 创新点**

创新点在于：①通过可学习的层级混合系数自适应识别每个客户端需要个性化的网络层；②在选定层上采用细粒度参数融合，实现全局知识与本地特化的动态平衡；③保持批判器本地化，避免跨环境的数值漂移。

**🔧 技术方法**

使用技术包括：联邦学习（FedAvg）、层级混合与参数融合、教师强制式模仿学习、强化学习损失、ResNet/CLIP 视觉特征、LSTM 编码/解码器。

**📊 数据集**

使用公开的室内导航数据集 Room-to-Room (R2R) 与 Room-across-Room (RxR)，并分别采用 ResNet-152 与 CLIP 视觉表示。

**📈 对比分析**

与中心化训练和基于 FedAvg 的联邦 baseline（FedVLN）对比。结果显示，在 SR、SPL、nDTW 等指标上分别提升最高 7.5% 与 7.8%；且在非 IID 条件下收敛速度提升 1.38×，通信开销低于 FedVLN。

**⚠️ 局限性**

局限性：若全部层都进行个性化会导致性能下降且计算/存储成本显著提升；方法仍依赖于每个客户端拥有足够的本地数据；在极端异质或数据稀缺场景下，个性化效果可能有限。

---

## 764. MCPShield: A Security Cognition Layer for Adaptive Trust Calibration in Model Context Protocol Agents

**arXiv ID:** 2602.14281 | [PDF](https://arxiv.org/pdf/2602.14281v1)

**作者:** Zhenhong Zhou `[一作]` (Nanyang Technological University), Qingsong Wen `[通讯]` (Squirrel AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MCPShield，一种面向 MCP 协议的 LLM 代理安全认知层，用以在工具调用生命周期中检测和缓解安全偏差。

**💡 创新点**

创新点在于将安全认知与代理决策结合，形成生命周期全程（预调用、执行、后调用）三阶段防御，利用元数据驱动的 mock 探测、受控执行隔离与历史行为推理实现自适应信任校准。

**🔧 技术方法**

主要技术包括元数据驱动的 mock 调用生成、LLM 基于安全评估的判定、受限执行环境与审计日志、跨调用漂移检测与社区共享签名。

**📊 数据集**

使用了 76 个恶意 MCP 服务器的安全基准（MCPSecBench、MCPSafetyBench、DemonAgent、Adaptive Attack、MCP-Artifact、Rug Pull），以及多种 LLM 代理模型（GPT-4o-mini、Gemini3-Flash、Kimi-K2、Deepseek V3.2、Minimax-M2、Qwen3-235B）。

**📈 对比分析**

与无防御代理相比，MCPShield 的攻击成功率下降至约 4.7%（即防御率 95.3%），在 Pass@K、低误拒率与可观测开销方面表现稳定，运行时与 token 开销仅相当于少量正常交互。

**⚠️ 局限性**

局限性包括对远程/低可观测部署支持不足、对代理模型行为的依赖、对持续适应性攻击的鲁棒性有限，以及仍需结合沙箱等系统级安全措施。

---

## 765. It Takes Two to Tango: A Holistic Simulator for Joint Order Scheduling and Multi-Agent Path Finding in Robotic Warehouses

**arXiv ID:** 2602.13999 | [PDF](https://arxiv.org/pdf/2602.13999v1)

**作者:** Haozheng Xu `[一作]` (East China Normal University), Xiangfeng Wang `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了WareRover平台，将订单调度与多代理路径规划通过闭环联合优化紧密耦合，支持真实仓库布局、异构AGV以及执行失效的动态恢复。

**💡 创新点**

创新点在于统一的闭环优化接口、可自定义高保真仓库模型以及对随机失效进行原生仿真与恢复的能力。

**🔧 技术方法**

采用多模块设计，包括仓库环境构造器、订单任务模块、共享状态的调度与路径规划耦合、连续运动执行器与失败模拟。

**📊 数据集**

实验使用自定义的订单生成模式（波次、热点SKU、促销突发等）在模拟的三类环境（均质、异质、容错）中验证。

**📈 对比分析**

通过对比三种调度策略与三种MAPF算法，在100次重复实验中衡量成功率、计算时长与吞吐量，结果显示闭环耦合下CBS与A*能保持高成功率并显著提升吞吐量。

**⚠️ 局限性**

局限性包括依赖仿真环境的假设、缺乏真实机器人硬件验证、以及对极大规模系统的可扩展性与实时性尚未充分评估。

---

## 766. T2MBench: A Benchmark for Out-of-Distribution Text-to-Motion Generation

**arXiv ID:** 2602.13751 | [PDF](https://arxiv.org/pdf/2602.13751v1)

**作者:** Bin Yang `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2053 | [OpenAlex ID](https://openalex.org/A5109900808)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 T2MBench，针对文本生成动作的 OOD 评估框架和数据集。

**💡 创新点**

创新点在于构建 1,025 条多维 OOD 文本提示、统一的 LLM‑基评估、运动多因素评估与细粒度准确性评估，兼具语义对齐、可泛化性与物理可行性三个维度。

**🔧 技术方法**

采用 GPT‑5.2 进行 LLM 评估，SMPL 渲染生成 2D 时空条纹，结合多种运动质量指标（JD、GP、FF、FS、DD、PQ、BP）以及匹配、R‑Precision、ASR 等指标。

**📊 数据集**

使用从 Vimogen‑228k、HumanML3D、AMASS、Oxford Dictionary 等数据衍生的 OOD 文本提示集，并从 14 种基线模型中挑选语义和物理属性最佳动作构建对应的数据集。

**📈 对比分析**

在 14 种 diffusion 与 VQ 基线上进行系统评估，结果显示大多数模型在语义对齐与物理质量上表现可观，但在细粒度准确性评估（全身/部件定位误差）上普遍不足，整体性能仍有提升空间。

**⚠️ 局限性**

限制在于 OOD 生成的细粒度精度不足、长序列性能下降，以及评估框架尚未覆盖所有可能的物理约束和长时序逻辑。

---

## 767. Data-driven Bi-level Optimization of Thermal Power Systems with embedded Artificial Neural Networks

**arXiv ID:** 2602.13746 | [PDF](https://arxiv.org/pdf/2602.13746v1)

**作者:** Talha Ansar `[一作]` (University of Engineering and Technology Lahore), Waqar Muhammad Ashraf `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于ANN‑KKT的双层优化框架，用人工神经网络代理上下层目标，在工业热电厂实现功率最大化与热效率最小化的决策。

**💡 创新点**

创新点在于同时用ANN替代双层目标函数，并利用Fischer–Burmeister函数改写KKT互补约束，从而显著降低计算时间并能够识别鲁棒操作空间。

**🔧 技术方法**

采用浅层前馈ANN、Bayesian优化调参、KKT改写、Fischer–Burmeister函数、马氏距离约束，并用IPOPT/ BARON求解器求解。

**📊 数据集**

使用660 MW煤电厂（1279条样本）与395 MW燃气轮机（579条样本）的历史运行数据，包含各工况变量与功率、热效率等性能指标。

**📈 对比分析**

与传统双层KKT和基准问题结果对比，ANN‑KKT在0.22–0.88 s内完成优化，得到的功率/热率与基准误差<1%，在基准问题中CPU时间比Bi‑level‑KKT高约2–4倍但仍在1–2 s内。

**⚠️ 局限性**

限制在于需要手动调节马氏距离阈值τ以保证可行性，ANN模型引入的非凸性可能导致求解器收敛到局部最优，未来需探索模型凸化或更稳健的约束处理方法。

---

## 768. The Complexity of Tournament Fixing: Subset FAS Number and Acyclic Neighborhoods

**arXiv ID:** 2602.13422 | [PDF](https://arxiv.org/pdf/2602.13422v1)

**作者:** Yuxi Liu `[一作]` (University of Electronic Science and Technology of China), Mingyu Xiao `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1477 | [OpenAlex ID](https://openalex.org/A5033729619)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究淘汰赛固定化问题（TFP）在多种基于子集反馈集的参数化下的复杂性。

**💡 创新点**

证明了当 sfas‑v 数为常数时 TFP 仍为 NP‑hard，并给出了在 sfas‑in 与 sfas‑out 和邻域无环时的可解界限及新的充分条件。

**🔧 技术方法**

采用参数化归约、子集反馈集理论、归纳构造、匹配与图论技术进行理论分析。

**📊 数据集**

论文未使用实验数据集，全部结果均为理论证明。

**📈 对比分析**

通过复杂度分析与定理证明进行比较，未给出实验性能指标。

**⚠️ 局限性**

仍未解决在入/出邻域都无环时 TFP 的 NP 难度问题。

---

## 769. NL2LOGIC: AST-Guided Translation of Natural Language into First-Order Logic with Large Language Models

**arXiv ID:** 2602.13237 | [PDF](https://arxiv.org/pdf/2602.13237v1)

**作者:** Rizky Ramadhana Putra `[一作]` (Virginia Tech), Peng Gao `[通讯]` (Virginia Tech)

**通讯引用:** 33066 | [OpenAlex ID](https://openalex.org/A5100348308)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于AST的自然语言到一阶逻辑（FOL）的转换框架，采用递归LLM语义解析器和两阶段AST引导生成器，显著提升语法正确性与语义忠实度。

**💡 创新点**

创新点在于：①使用AST中间层将句子逐级分解为子句，递归解析，降低LLM认知负担；②将逻辑解析与代码生成分离，采用两遍生成保证全局语法一致性；③在大语言模型上实现结构化JSON输出，控制生成格式。

**🔧 技术方法**

技术包括：大型语言模型（Gemma、Llama、Mistral、Qwen等）配合结构化输出（vLLM/OpenAI API）；递归语义解析器（Atomic/Quantified/Logical子解析器）；AST‑guided两通道生成器（声明收集+表达式生成）；Z3等自动推理引擎用于执行验证。

**📊 数据集**

使用三大自然语言推理/逻辑翻译数据集：FOLIO、LogicNLI、ProofWriter。

**📈 对比分析**

对比基准为Grammar‑Constrained Decoding (GCD) 与原始Logic‑LM；评估指标包括语法正确率、NLI准确率和可执行率。实验结果显示：语法正确率接近99%，在三大数据集上NLI准确率平均提升约30%，集成至Logic‑LM后可执行率从0.01–0.94提升至0.99，整体推理准确率提升约31%。

**⚠️ 局限性**

局限性：①只能逐句处理，无法处理跨句共指或隐式关系；②依赖LLM结构化输出功能，若模型或框架不支持需额外集成；③对极长或复杂文本仍可能出现节点缺失或格式错误；④目前未覆盖多句推理中全局一致性（如谓词多态、共指消解）。

---

## 770. Radial-VCReg: More Informative Representation Learning Through Radial Gaussianization

**arXiv ID:** 2602.14272 | [PDF](https://arxiv.org/pdf/2602.14272v1)

**作者:** Yilun Kuang `[一作]` (New York University), Yann LeCun `[通讯]` (New York University)

**通讯引用:** 243182 | [OpenAlex ID](https://openalex.org/A5001226970)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Radial-VCReg，一种在 VCReg 基础上加入径向高斯化损失的自监督学习方法；

**💡 创新点**

创新点在于构造了一个一致的 KL 散度估计器，使特征范数与卡方分布对齐，从而把更广泛的分布转化为高斯分布；

**🔧 技术方法**

使用了径向高斯化损失（交叉熵与 m-spacing 熵估计器）与 VICReg 的方差、相容性、协方差正则化相结合的技术；

**📊 数据集**

实验使用了 CIFAR‑100、ImageNet‑10、CelebA 以及自定义的二维 X‑分布；

**📈 对比分析**

通过与 VCReg、VICReg 等方法在线性和 MLP 探针上的对比，Radial‑VCReg 在多种 backbone 和投影器维度下平均提升 1–2% 的 Top‑1 准确率；

**⚠️ 局限性**

局限性在于径向高斯化仅提供必要但非充分的高斯性条件，且仍受超参数和高阶依赖的影响。

---

## 771. Majority Boolean networks classifying density: structural characterization and complexity

**arXiv ID:** 2602.13511 | [PDF](https://arxiv.org/pdf/2602.13511v1)

**作者:** Kévin Perrot `[一作]` (Aix Marseille Université), Marius Rolland `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文研究了多数布尔自动机网络（MBAN）在密度分类任务（DCT）中的行为，提出了判定MBAN是否能完成DCT的图结构性质，并给出了对应的三类禁止模式；进一步分析了判定这些模式的计算复杂度，证明了存在性判定为NP-完备，周期性判定为PSPACE-完备；同时展示了从电路迭代值问题到MBAN的多步多项式约简；

**💡 创新点**

创新点在于：①首次用三种禁止模式（领袖、自治子集、自治m-循环）完整刻画MBAN解DCT的图结构；②将MBAN与传统电路迭代问题关联，建立多阶段多项式约简；③证明了判定问题的高复杂度，为此类分布式一致性问题提供了严谨的复杂度上界；

**🔧 技术方法**

主要技术包括：布尔电路到MBAN的映射（利用多数局部规则模拟AND/OR门）；多级图构造与循环/固定点分析；复杂度理论工具（NP、PSPACE完全性证明）；图论与自治集/领袖定义；

**📊 数据集**

无实验数据集，全部为理论推导与复杂度证明；

**📈 对比分析**

由于本研究为理论性质探讨，没有实验比较；通过多项式约简证明与复杂度分析验证结论；

**⚠️ 局限性**

局限性在于：1) 尚未确定MBAN解决DCT问题本身的精确复杂度（是否NP-完备仍未证实）；2) 仅给出理论结构判定，缺乏构造可行的正样例集合；3) 对更具约束的图（如k-正则、均匀图）分析尚不充分；

---

## 772. Real-World Design and Deployment of an Embedded GenAI-powered 9-1-1 Calltaking Training System: Experiences and Lessons Learned

**arXiv ID:** 2602.13241 | [PDF](https://arxiv.org/pdf/2602.13241v1)

**作者:** Zirong Chen `[一作]` (Vanderbilt University), Meiyi Ma `[通讯]` (Vanderbilt University)

**通讯引用:** 1515 | [OpenAlex ID](https://openalex.org/A5101671027)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在纳什维尔市911通信中心部署并运营一个基于通用人工智能（GenAI）的培训系统，用于生成逼真呼叫场景、实时评估学员的协议遵守情况，并提供即时反馈。

**💡 创新点**

四项创新：① 迭代交付将领域专家从审计者转变为协同架构师，① 基于信号时序逻辑（STL）的形式化验证与LLM模块化相结合提升安全性；③ 通过三角反馈（用户→系统→质保）区分技术失效与压力误报；④ 设计“可度化难度”与“正向反馈”机制，让学员在挑战中保持动力。

**🔧 技术方法**

使用的技术包括：GPT‑4o、其他多模态LLM、云端服务（生成呼叫者行为、实时监测）、Signal Temporal Logic（STL）形式化规范、混合架构（LLM做谓词验证）、多模态录音与日志分析工具。

**📊 数据集**

主要数据集包括：98,429 次用户交互日志，11,129 次系统事件，5,244 分钟音频记录，1,120 次培训会话，1,651 条从自然语言转换的协议规范；并在1,244 次真实呼叫上评估验证。

**📈 对比分析**

对比方法：将混合形式化‑LLM架构与纯LLM（GPT‑4o+CoT+RAG）基线对比。混合架构在协议合规评估上达 94.3–95.9% 的专家一致率，显著高于 87.8% 的纯LLM 结果。

**⚠️ 局限性**

局限性：仅在单一市政部门部署，可能缺乏跨地区可推广性；系统仍依赖人工质保审核，人工成本未被完全消除；对真实急救通话的模拟准确度和偏差未在大规模实战中验证；以及对敏感人口特征（语言、情绪）模拟的伦理与隐私挑战。

---

## 773. Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?

**arXiv ID:** 2602.14111 | [PDF](https://arxiv.org/pdf/2602.14111v1)

**作者:** Anton Korznikov `[一作]`, Elena Tutubalina `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估稀疏自编码器（SAE）在解释大型语言模型内部机制的有效性，采用两种实验：在已知真实特征的合成数据上检验特征恢复，和在真实LLM激活上与三种随机冻结基线对比；

**💡 创新点**

创新点在于提出可直接实现的随机冻结基线，证明在标准评价指标上，SAE与基线表现相当，揭示高重构/可解释性并不一定意味着学习到真实特征；

**🔧 技术方法**

使用稀疏自编码器（BatchTopK、JumpReLU、ReLU）、冻结解码器/编码器/软冻结解码器等基线、合成特征生成、自动可解释性评分（AutoInterp）、稀疏探测、RAVEL因果编辑等技术；

**📊 数据集**

数据集包括自定义合成激活（n=100，m=3200），Gemma‑2‑2B（12/19层）和 Llama‑3‑8B（16层）残差流激活，训练使用OpenWebText 5亿tokens；

**📈 对比分析**

与完全训练的SAE对比时，冻结基线在重构方差、自动可解释性、稀疏探测与因果编辑指标上几乎匹配或仅略低（<5%差距），说明SAE的“表现”大多来自随机初始化而非学习；

**⚠️ 局限性**

局限性：合成实验假设特征独立，未考虑真实模型中潜在的相关性；只评估传统SAE，而未涵盖如转码器、交叉编码器等变体；

---

## 774. UniRef-Image-Edit: Towards Scalable and Consistent Multi-Reference Image Editing

**arXiv ID:** 2602.14186 | [PDF](https://arxiv.org/pdf/2602.14186v1)

**作者:** Hongyang Wei `[一作]` (Tsinghua University), Han Li `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了UniRef-Image-Edit，一套可统一处理单图编辑与多图合成的框架；

**💡 创新点**

核心创新包括Sequence-Extended Latent Fusion (SELF) 统一序列化多参考图像；两阶段训练（SFT+MSGRPO）以及针对多参考合成的强化学习奖励机制；

**🔧 技术方法**

使用VAE+Transformer（MMDiT）实现SELF，flow-matching训练，GRPO强化学习，LLM-as-a-Judge奖励（Gemini 3 Flash）以及SDE采样；

**📊 数据集**

构建并公开UniRef-40k数据集（20k人机交互 + 20k多人人交互），并利用GPT‑5与Nano Banana Pro等生成合成样本；

**📈 对比分析**

与多款主流开源与专有模型（Nano Banana、Seedream 4.0、GPT Image 1、Qwen‑Image‑Edit 等）在OmniContext、MultiCom‑Bench、GEdit‑Bench、ImgEdit‑Bench 等基准上进行对比，UniRef‑Image‑Edit 在多图合成、单图编辑两类任务均取得或接近最优成绩；

**⚠️ 局限性**

局限性：对超大图像分辨率和极多参考图的处理仍受序列长度限制；强化学习阶段仍依赖人工构造的多源奖励，可能难以完全覆盖所有语义约束；

---

## 775. Fast Swap-Based Element Selection for Multiplication-Free Dimension Reduction

**arXiv ID:** 2602.13532 | [PDF](https://arxiv.org/pdf/2602.13532v1)

**作者:** Nobutaka Ono `[一作]` (Tokyo Metropolitan University), Nobutaka Ono `[通讯]` (Tokyo Metropolitan University)

**通讯引用:** 5801 | [OpenAlex ID](https://openalex.org/A5056281759)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种快速的交换式元素选择算法，用于实现无乘法的维度缩减，直接从原始数据中挑选子集来构成低维表示。

**💡 创新点**

创新点在于把维度缩减视为离散优化问题，使用均方回归误差作为目标，并通过矩阵逆推导得到每次交换的目标增量，只需对2×2矩阵求逆，从而极大加速局部搜索。

**🔧 技术方法**

采用协方差与交叉协方差的矩阵分析、矩阵逆推导（Sherman-Morrison-Woodbury）、交换式局部搜索（2-opt）以及MATLAB实现的向量化计算。

**📊 数据集**

在MNIST手写数字数据集上进行实验，使用784维像素向量，将目标设为重构自身。

**📈 对比分析**

与PCA、方差最高像素选取、随机投影等传统方法对比，重构误差下降至0.1367（高于PCA的0.0856）但计算时间仅22秒（比直观实现快≈200×）。

**⚠️ 局限性**

局限性包括仅在线性回归/重构目标下验证，可能在非线性任务或大规模高维数据上表现不佳，且算法仍需离线预处理，未探讨实时动态选择。

---

## 776. DECKBench: Benchmarking Multi-Agent Frameworks for Academic Slide Generation and Editing

**arXiv ID:** 2602.13318 | [PDF](https://arxiv.org/pdf/2602.13318v1)

**作者:** Daesik Jang `[一作]` (Huawei Technologies Canada), Zhenan Fan `[通讯]` (Huawei Technologies Canada)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DeckBench，一个面向学术论文到幻灯片生成与多轮编辑的基准与评估框架，并提供多模态指标与模拟用户交互。

**💡 创新点**

创新点在于整合生成与编辑两大任务，使用模拟用户多轮指令来评估编辑响应与迭代改进，并提供统一的多层次指标体系。

**🔧 技术方法**

采用多模态多代理系统，包括 Outline、Code、Editor 三个代理，使用大型语言模型（如 GPT‑4o、Qwen 系列）配合 MinerU 结构解析、CLIP 视觉语义模型、DTW 等评估工具。

**📊 数据集**

使用从 CVPR、ECCV、ICLR、ICML 等会议收集的 294 对学术论文与官方幻灯片，并通过模拟用户生成 5 轮编辑指令，构成完整数据集。

**📈 对比分析**

与 Auto‑Slides、EvoPresent、SlideGen 等现有多代理基线对比，幻灯片级指标表现相近或略优；在编辑阶段，GPT 方案在 DTW 与 Transition Similarity 上显著提升，整体保持 0% 失败率，证明基准有效区分系统性能。

**⚠️ 局限性**

局限包括对复杂公式与图表布局的处理仍不完善；模拟用户与真实交互差距；评估指标主要为自动化，缺乏人类可用性与可访问性等实际使用体验的评估。

---

## 777. LACONIC: Length-Aware Constrained Reinforcement Learning for LLM

**arXiv ID:** 2602.14468 | [PDF](https://arxiv.org/pdf/2602.14468v1)

**作者:** Chang Liu `[一作]` (University of California), Lin F. Yang `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于约束强化学习的长度控制方法 LACONIC，对 LLM 进行 RL 微调时强制满足平均 token 预算。

**💡 创新点**

创新点是使用剪切成本与自适应的拉格朗日乘子，实现动态长度约束而非固定惩罚；同时提供理论收敛与近优性保证。

**🔧 技术方法**

采用 Primal‑Dual 方法结合 GRPO 实现，使用长度剪切成本 c(q,o)=max{L(o)-B,0}/B，动态更新双变量 λ。

**📊 数据集**

在四大数学推理数据集 DeepScaleR‑Preview（AIME、AMC、Omni‑MATH、STILL）以及 AIME2024、MATH、Minerva、Olympiad‑Bench 和 OOD 数据集 GPQA、LSAT、MMLU 进行评估。

**📈 对比分析**

与 GRPO、L1‑Exact、L1‑Max、Efficient‑Reasoning、ThinkPrune‑Iter2k 等基线相比，LACONIC 在保持或提升 pass@1 的同时，将 token 数量平均下降约 44–71%；在 OOD 任务上也保持准确率并显著压缩长度。

**⚠️ 局限性**

局限性包括仅对单一平均 token 约束、仅验证于数学推理任务、对提示级动态长度需求不够灵活；未来可扩展到多约束和其他生成任务。

---

## 778. Learning Significant Persistent Homology Features for 3D Shape Understanding

**arXiv ID:** 2602.14228 | [PDF](https://arxiv.org/pdf/2602.14228v1)

**作者:** Prachi Kudeshia `[一作]` (Saint Marys University), Jiju Poovvancheri `[通讯]` (Saint Marys University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过预先计算ModelNet40和ShapeNet的H1、H2持久同调图，构建了“topological signatures”数据集，并提出TopoGAT网络实现可学习的显著持久点选择，从而将拓扑信息有效融入点云分类与分割任务。

**💡 创新点**

创新点在于：①首次公开大规模点云的持久同调数据集；②设计了基于GAT的可端到端显著点选择网络TopoGAT，并引入三项可学习的拓扑损失（Wasserstein、持久熵、稀释损失）；③通过统计方法与TopoGAT的对比，验证显著点选择在保持拓扑完整性的同时提升了下游性能。

**🔧 技术方法**

使用技术包括：Vietoris–Rips过滤器生成持久同调图、三支动态KNN‑GAT特征提取模块、回归预测阈值、sigmoid软掩码过滤、复合拓扑损失和标准交叉熵分类损失。

**📊 数据集**

使用的数据集是ModelNet40和ShapeNet（点云及其对应的H1、H2持久同调图），并在此基础上进行分类与分割实验。

**📈 对比分析**

通过与三种统计显著点选择方法（Method1–3）以及不同GNN（GCN、GAT、GIN）和向量化方式（PL、PI、PD）对比，TopoGAT在Wasserstein、持久熵相差和分类/分割准确率上均优于统计方法，且与方法2、3的Bottleneck距离相近；在ModelNet40分类上最高OA≈87%，在ShapeNet分割上mIoU提升至约70%。

**⚠️ 局限性**

局限性在于：仅在基础GNN架构上验证，未探索更复杂网络；topo-loss仍基于全局统计，可能未完全捕捉任务特定的拓扑结构；预计算PD的计算成本高（≈860小时）；仅考虑H1、H2，忽略了更高维拓扑信息。

---

## 779. Global AI Bias Audit for Technical Governance

**arXiv ID:** 2602.13246 | [PDF](https://arxiv.org/pdf/2602.13246v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 780. EIDOS: Latent-Space Predictive Learning for Time Series Foundation Models

**arXiv ID:** 2602.14024 | [PDF](https://arxiv.org/pdf/2602.14024v1)

**作者:** Xinxing Zhou `[一作]` (Nankai University), Ming Jin `[通讯]` (Griffith University)

**通讯引用:** 13142 | [OpenAlex ID](https://openalex.org/A5039636381)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于潜在空间预测学习的时间序列基础模型Eidos，并通过预训练使模型学会预测潜在表示的演变，从而获得更结构化、对噪声鲁棒的内部表示。

**💡 创新点**

创新点在于将预训练目标从直接预测观测值转向预测潜在嵌入，采用轻量级聚合分支和对齐损失以避免表示崩塌，并通过观测锚定保证潜在表示与实际信号保持一致，形成联合目标。

**🔧 技术方法**

核心技术包括：SiGLU点级编码、因果Transformer（Pre-LN + Rotary + FlashAttention）、channel‑wise卷积聚合器、对齐损失（负余弦相似度）与 grounding 损失（量化回归），以及量化头用于生成概率预测。

**📊 数据集**

预训练使用约1.7亿条时间序列、1700亿时间点，来源包括 Chronos、GiftEval、以及通过CauKer方法生成的合成序列；下游评估在 GIFT‑Eval（97种配置）及 fev‑bench 上进行。

**📈 对比分析**

与现有基础模型（Chronos‑2、Moirai、TimesFM等）对比，Eidos 在零样本设置下取得0.757的标准化MASE和0.547的CRPS，表现接近或优于领先模型，并在加噪声实验中显示出显著的鲁棒性，结构化潜在空间的线性可分性和可编辑性也得到验证。

**⚠️ 局限性**

局限性包括仅针对单变量预测，未覆盖多变量或协变量情况；模型与超参数设计保持简洁，缺乏大规模调优；对真实业务场景的可部署性和多模态融合尚待进一步探索。

---

## 781. The antiferromagnetic Ising model beyond line graphs

**arXiv ID:** 2602.14915 | [PDF](https://arxiv.org/pdf/2602.14915v1)

**作者:** Mark Jerrum `[一作]` `[通讯]` (Queen Mary University of London), Mark Jerrum (Queen Mary University of London)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过构造特定的图形“gadget”与对称映射，证明了反铁磁伊辛模型在包含线图的更广泛图类（准线图和多重图线图）上，在给定的常数交互强度 μ 下，求取其配分函数近似或有效采样是 NP‑难的；并进一步给出了无条件指数级的 Glauber 动态混合时间上界。

**💡 创新点**

创新点在于将之前仅适用于线图的可解性结论扩展到更大的图类，并展示了即使是极小的结构变化（如加入少量的“分支”或允许两种不同交互强度）也会导致计算难度骤增，揭示了可解性与图结构之间的精细边界。

**🔧 技术方法**

主要技术包括：(a) 对三度图的最大割问题进行多项式归约，构造与割大小直接相关的 Ising 配分函数；(b) 设计包含三个顶点组（终端、辅助、基准）的 gadget 图 H_v，使内部边的权重取值呈 18 次幂或 17 次幂，从而给出配分函数的上下界；(c) 使用电导/导电率（conductance）与 Cheeger 不等式证明 Glauber 动态的指数慢混合；(d) 讨论不同交互强度对可解性的影响。

**📊 数据集**

本文属于理论计算机科学，无使用实际数据集；所有结果均来自数学证明与组合构造。

**📈 对比分析**

由于研究对象是理论复杂度，本文未与具体算法或实现进行实验对比；其主要“性能”体现为证明的计算复杂度与混合时间上限。

**⚠️ 局限性**

局限性包括：① 需要极大常数 μ（如 μ ≥ 2^3067）才能保证归约的有效性；② 只在最大度为 8 的准线图（或最大度为 6 的线图子图）上适用，尚未证明更一般图类的可解性边界；③ 结果主要关注配分函数的近似与采样，未涉及其他物理量或更复杂的统计物理模型。

---

## 782. Identification of random material properties as stochastic inversion problem

**arXiv ID:** 2602.14684 | [PDF](https://arxiv.org/pdf/2602.14684v1)

**作者:** Eliška Kočková `[一作]` (Czech Technical University in Prague), Anna Kučerová `[通讯]` (Czech Technical University in Prague)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对异质材料参数的随机性进行逆问题识别，比较贝叶斯层次模型与变量变换两种方法

**💡 创新点**

提出两种不同的随机逆问题公式：一种通过超参数贝叶斯推断参数分布，另一种直接利用观测数据变换得到参数分布

**🔧 技术方法**

采用层次贝叶斯推断、MCMC、Polynomial Chaos、主成分分析、Gaussian Copula、变量变换等技术

**📊 数据集**

以铜合金循环加载实验（16 次）以及两种破坏实验的合成数据（50 条拉伸、30 条循环）为样本

**📈 对比分析**

在合成和真实数据上对比两种方法的平均绝对误差、方差、相关系数等指标，变换方法在预测方差与相关性上表现更好，贝叶斯方法更精确估计均值且能体现不确定性

**⚠️ 局限性**

贝叶斯方法需预设参数分布形式，计算量随样本数增大而剧增；变换方法无法区分本质不确定性与测量误差，且对数据相关性假设较弱时表现不佳

---

## 783. A Kung Fu Athlete Bot That Can Do It All Day: Highly Dynamic, Balance-Challenging Motion Dataset and Autonomous Fall-Resilient Tracking

**arXiv ID:** 2602.13656 | [PDF](https://arxiv.org/pdf/2602.13656v1)

**作者:** Zhongxiang Lei `[一作]` (Beijing Institute of Technology), Xuesong Li `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 5088 | [OpenAlex ID](https://openalex.org/A5100449091)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了基于专业武术运动员日常训练视频的高动态动作数据集，并训练了统一的动作追踪与跌倒恢复策略。

**💡 创新点**

提出单一策略同时学习高动态动作跟踪与自主跌倒恢复，并利用根节点高度漂移校正和低能量采样提升鲁棒性。

**🔧 技术方法**

使用FastSAC强化学习、根节点高度漂移校正算法、Savitzky–Golay平滑、低能量采样、GRSI初始化以及多项奖励函数设计。

**📊 数据集**

使用KungFuAthlete数据集（地面与跳跃子集），并与公开的LAFAN1、PHUMA、AMASS等数据集进行对比。

**📈 对比分析**

在仿真中与BeyondMimic及其他基线相比，成功率提升至100%，姿态误差下降约30%，并在Unitree G1机器人上实现真实世界跌倒恢复。

**⚠️ 局限性**

受视频重建噪声、数据集对齐误差和对复杂落地姿态泛化能力的限制，需要进一步提升重建精度与跨域适应性能。

---

## 784. A Formal Framework for the Explanation of Finite Automata Decisions

**arXiv ID:** 2602.13351 | [PDF](https://arxiv.org/pdf/2602.13351v1)

**作者:** Jaime Cuartas Granada `[一作]` (Monash University), Peter J. Stuckey `[通讯]` (Monash University)

**通讯引用:** 14145 | [OpenAlex ID](https://openalex.org/A5064839018)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了针对有限自动机（FA）输入字串的正式可解释性框架，定义并高效求解两类解释：最小归纳解释（AXp）说明为何FA接受或拒绝该字串；最小对比解释（CXp）说明如何最小改动字串以改变FA的决策。

**💡 创新点**

创新点在于：① 将解释问题映射为最小命中集/最小不可满足子集的求解，从而利用SAT/MUS/MCS技术实现可解释性；② 对AXp和CXp之间的对偶关系进行形式化并加以利用；③ 设计了可扩展的“_1^1”语言框架（固定长度、字符可替换）并证明其在多种场景下可高效枚举；④ 引入正式特征归因（FFA）作为解释的量化评估。

**🔧 技术方法**

核心技术包括：SAT求解与硬/软约束编码（用于构造补语FA的计算），MUS/MCS枚举实现最小解释，ILP/MaxSAT用于寻找最小命中集，正则表达式与DFA的语言包含检测，算法层面实现的“ExtractAXp”和“ExtractCXp”，以及对解释集合的枚举算法（类似MARCO）。

**📊 数据集**

实验数据集覆盖四类任务：Deep Packet Inspection（98个FTP签名合并成约30k状态NFA）；随机生成FA语料库（长度5–20，字母表大小2–10）；DNA序列与已知motif（810条，长度62–2000）；迷宫路径判定（861个迷宫，尺寸10×10–50×50）。

**📈 对比分析**

通过cactus图对三种枚举模式进行比较：① 直接枚举AXp；② 直接枚举CXp；③ 用所有单字位置的CXp做warm‑start后枚举AXp。实验显示，warm‑start模式在大多数实例中最快（平均≈19–23 s），对最难的实例时直接枚举CXp更优。支持的FA规模从10到约2500状态（DPI情形除外），对长字串（最大2000字符）亦能在600 s超时前完成约70%实例。

**⚠️ 局限性**

局限性包括：① 解释数量指数级增长，导致枚举在极大FA或极长字串时无法完成；② 当前框架仅适用于DFA（需要将NFA转换为DFA）；③ 只针对“_1^1”语言（固定长度单字符替换）做了深度优化，其他语言（_1^∞、_0^∞）性能较低；④ 对实际可解释性的人类评估缺失，仅从算法效率角度验证；⑤ 对高层次特征抽象的支持有限，需要进一步与神经网络或其他模型结合。

---

## 785. Intent-driven Diffusion-based Path for Mobile Data Collector in IoT-enabled Dense WSNs

**arXiv ID:** 2602.13277 | [PDF](https://arxiv.org/pdf/2602.13277v1)

**作者:** Uma Mahesh Boda `[一作]` (Annamacharya University), Mallikharjuna Rao Nuka `[通讯]` (Annamacharya University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向意图的扩散式路径规划框架 ID^2P^2，用于在 IoT 支持的稠密无线传感器网络中联合优化会聚点（Rendezvous Point）位置与移动数据收集器（MDC）的巡回路径。

**💡 创新点**

创新点在于将高层运作意图（如延迟最小化、能耗平衡、数据新鲜度、覆盖优先级）显式建模为损失函数，并通过条件扩散模型生成符合意图的轨迹，突破了传统启发式规划对动态网络适应性的不足；同时首次将 RP 位置选择与路径规划在同一框架内协同完成，提升了整体性能。

**🔧 技术方法**

使用技术包括：条件扩散模型与梯度引导的损失函数；启发式 RP 位置选择算法；连续轨迹到离散路径的映射；局部 2‑opt 优化提升；并结合传感器数据流、移动速度、传输速率等参数进行仿真。

**📊 数据集**

数据集：利用仿真生成的稠密 WSN，节点数在 100~500 之间，随机部署于 200m×200m 区域，通信半径 25m，节点产生速率 0.5 kb/s，RPs 15 个，MDC 速度 2 m/s，上传速率 2 Mb/s。基准算法包括 MSCVP、HDAMM、MRFO、Gathers、IDDPP。

**📈 对比分析**

比较方法：在 10‑30 次独立随机部署下统计巡回时间、行驶距离、停留时间、数据新鲜度、收集比例、包交付率、能效、吞吐量和公平性等指标；实验结果显示 ID^2P^2 在所有指标上均优于基准，巡回时间缩短 25‑30%，能效提升 10‑30%，数据新鲜度、收集比例、PDR 等亦显著改善。

**⚠️ 局限性**

局限性：扩散模型的生成与梯度计算带来较高的计算开销，需离线训练或精细参数调优；在资源极端受限或网络快速变化的环境中可能不够实时；目前未涵盖多收集器协同与动态节点失效的适配，需进一步研究。

---

## 786. Revenue Guarantees in Autobidding Platforms

**arXiv ID:** 2602.14815 | [PDF](https://arxiv.org/pdf/2602.14815v1)

**作者:** Ioannis Caragiannis `[一作]` (Aarhus University), Stratis Skoulakis `[通讯]` (Aarhus University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究在可分割商品、预算约束买家市场中，利用固定单价的 First-Price Pacing Equilibrium (FPPE) 进行收益最大化，并在离线与在线两种情形下给出近似与竞争性分析。

**💡 创新点**

创新点在于证明 FPPE 在离线情况下至少取得可变单价最优收益的 50% 近似，并证明固定单价收益最大化问题为 APX‑hard；在线情况下通过每日重新计算 FPPE 获得 1/4 竞争比；此外将 FPPE 泛化至凹价值函数并给出相应收益保证。

**🔧 技术方法**

主要技术包括线性规划与凸优化（RMVUP、RMFUP）、Eisenberg–Gale（EG）程序求解、逼近与硬度证明（从 3D‑2‑Matching 转化）、在线竞争分析与预算递增性证明。

**📊 数据集**

论文以理论分析为主，未使用公开广告数据集进行实验验证。

**📈 对比分析**

性能评估通过与可变单价最优收益（RMVUP）比较，离线比为 1/2；在线比为 1/4（上界约 0.85），并给出固定单价最优收益与 FPPE 之间的下界 1/2，证明该比率为最优。

**⚠️ 局限性**

局限性包括：仅适用于线性或凹价值函数；在线竞争率仍低于理论上可能的 1/2 或 0.85；缺乏实证验证；未讨论多品种动态需求的进一步复杂情形。

---

## 787. DenseMLLM: Standard Multimodal LLMs are Intrinsic Dense Predictors

**arXiv ID:** 2602.14134 | [PDF](https://arxiv.org/pdf/2602.14134v1)

**作者:** Yi Li `[一作]` (Hong Kong University of Science and Technology), Xiaomeng Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 9320 | [OpenAlex ID](https://openalex.org/A5100427643)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DenseMLLM，一种标准的多模态大语言模型，能够在不使用额外任务解码器或任务特定标记的情况下完成语义分割、深度估计和指代分割等像素级任务。

**💡 创新点**

核心创新包括：① 将视觉令牌的多标签下一步预测（NTP‑M）作为监督方式，使单个视觉令牌能够对应多个类别和任务；② 采用相关负采样策略，解决大词表下的类别不平衡；③ 直接从视觉令牌 logits 中提取像素级预测，保持模型结构的完整性与通用性。

**🔧 技术方法**

技术实现基于 ViT + projector + 4B 语言Transformer，使用 SigLIP‑2 视觉编码器、RoPE 空间编码、窗口+全局注意力、双层 MLP 投影；训练采用四阶段流程（基础预训练 → 细粒度任务退火 → 高质量微调 → 强化学习）以及多标签 Sigmoid‑NTP‑M 损失。

**📊 数据集**

训练与评测数据涵盖：ADE20K、COCO‑Stuff、Pascal Context、Cityscapes、Pascal VOC、NYUv2、DDAD、RefCOCO/RefCOCO+/RefCOCOg 等；同时使用公开图文对、OCR、VQA、知识图谱等多源视觉文本数据。

**📈 对比分析**

与视觉专家模型、CLIP‑based 视觉通用模型以及其他添加解码器的 MLLM 进行对比，DenseMLLM 在语义分割（ADE20K 54.2 mIoU）、深度估计（NYUv2 90.4 δ₁、DDAD 87.6 δ₁）和指代分割（RefCOCO 80.7 cIoU）等任务上均超越或接近最先进方法，并在一般 VQA、OCR、数学推理等通用 VL 任务上保持 4B 级别的竞争性能。

**⚠️ 局限性**

局限性：在极端稀有或长尾场景下表现受限，主要面向语义级任务；缺乏实例/全景分割、光流、表面法线等更细粒度的实例级或几何级任务，未来需引入实例区分或几何推理机制。

---

## 788. Personalization Aids Pluralistic Alignment Under Competition

**arXiv ID:** 2602.13451 | [PDF](https://arxiv.org/pdf/2602.13451v1)

**作者:** Natalie Collina `[一作]` (University of Pennsylvania), Mirah Shi `[通讯]` (University of Pennsylvania)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5102913178)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在多家AI提供者竞争、用户偏好多样化的场景下，是否能通过市场机制实现“多元对齐”——即每位用户都能获得与完全对齐模型相当的收益。作者将交互建模为多领导多追随者的Stackelberg博弈，区分可个性化部署与必须使用统一匿名策略的两种情况，并给出了在不同市场对齐假设（弱对齐与强对齐）下的理论保证。

**💡 创新点**

创新点：
1) 首次证明在弱市场对齐（Provider的效用可分离且用户效用是其非负组合）下，即使所有提供者均可能与用户不一致，个性化承诺也能保证每位用户在任意纳什均衡中获得与完美对齐模型相近的效用。
2) 揭示匿名情形的根本缺陷——即使满足弱对齐，也存在均衡使用户几乎无收益；进一步引入强市场对齐（Provider的效用本身是用户效用的非负线性组合）和用户主导策略，证明匿名竞争在此条件下同样能实现对齐。
3) 分析新用户加入的鲁棒性，指出在匿名情形下新用户可破坏既有用户的对齐保证，而在个性化情形下对齐不受影响。

**🔧 技术方法**

技术方法：
- 博弈论框架（多领导多追随者Stackelberg博弈）
- 信息论与贝叶斯说客（Bayesian Persuasion）思想，扩展到多说客、多追随者
- 对齐条件的形式化与证明（弱对齐、强对齐）
- 线性回归（非负最小二乘 NNLS）估计对齐误差与权重
- 经验评估与RMSE、传输因子（1/λ）作为理论条件的可行性指标

**📊 数据集**

数据集：OpinionQA（Pew American Trends Panel）问卷调查数据，用来模拟用户偏好与LLM预测分布，评估对齐条件在真实偏好数据中的适用性。

**📈 对比分析**

比较方法与性能：
- 对齐误差（RMSE）随提供者数量增加而下降，弱对齐误差低于强对齐约30–40%；
- 传输因子 1/λ 量化匿名情形下的松弛，更多提供者能显著降低该因子；
- 实验显示即便仅选择少量高质量提供者（最佳子集），也能大幅提升对齐；
- 结果表明：多提供者环境下，个性化对齐可实现近似完美效果；匿名环境则需要满足强对齐并且拥有足够多、足够多样的提供者。

**⚠️ 局限性**

局限性：
- 理论证明基于Worst‑case误差，实验仅使用平均RMSE，可能低估最坏情形；
- 对齐条件（弱/强）是足够但非必要，实际系统可能更复杂；
- 匿名情形下强对齐仍需大量提供者，且对新用户加入极度敏感；
- 研究假设提供者能够自由承诺对话策略，实际部署可能受训练成本、API限制等影响；
- 未考虑动态学习、迭代博弈和策略演化的情况。

---

## 789. Kami of the Commons: Towards Designing Agentic AI to Steward the Commons

**arXiv ID:** 2602.14940 | [PDF](https://arxiv.org/pdf/2602.14940v1)

**作者:** Botao Amber Hu `[一作]` `[通讯]` (University of Oxford), Botao Amber Hu (University of Oxford)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

在一次为期三小时的Speculative Design工作坊中，研究者与15名跨学科参与者通过Protocol Futuring方法，共同构想并讨论了为各类公共资源（如家庭、知识档案、社区资源）配备AI守护者（kami）的可能性及其带来的二阶影响。

**💡 创新点**

首次提出将AI治理视为一种可设计的“共同体治理材料”，从精神化与关怀伦理角度出发，探讨AI守护者的可配置性、交互性以及递归治理问题，开辟了面向关怀与责任的AI共同体治理设计新空间。

**🔧 技术方法**

主要使用了Protocol Futuring的框架（结合Speculative Design、Adversarial Design与Design Justice），并未实现具体技术原型，仅在设计层面构建AI守护者的角色与协议。

**📊 数据集**

无；本文为概念性与方法性探讨，并未使用或收集数据集。

**📈 对比分析**

无；本研究未进行实验比较或性能评估，而是通过工作坊中的讨论与反思来揭示可能的机会与风险。

**⚠️ 局限性**

局限性包括：样本规模仅15人，缺乏经验代表性；缺乏实际原型与长期部署验证；文化偏向日本神道观念，可能不适用于其他文化；并未解决AI守护者自身治理的递归与透明度问题。

---

## 790. MarsRetrieval: Benchmarking Vision-Language Models for Planetary-Scale Geospatial Retrieval on Mars

**arXiv ID:** 2602.13961 | [PDF](https://arxiv.org/pdf/2602.13961v1)

**作者:** Shuoyuan Wang `[一作]` (Southern University of Science and Technology), Hongxin Wei `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 1069 | [OpenAlex ID](https://openalex.org/A5020027500)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MarsRetrieval，一套针对火星地理信息检索的跨模态基准；

**💡 创新点**

创新点在于三大任务（配对图文检索、地貌检索、全局地理定位）以及统一检索式评估协议，并对数据进行多阶段筛选、专家校正和细化字幕；

**🔧 技术方法**

使用对比双塔编码器、MLLM嵌入模型、Vision-only模型等多种视觉语言模型，以及自研的MarScope进行领域微调；

**📊 数据集**

数据集由两阶段网络检索产生的图文对、CTX/HiRISE图像补充、专家验证的48类地貌子集以及全球1.4M CTX瓦片构成；

**📈 对比分析**

通过Recall@1/10、mAP、nDCG、AUPRC等指标对比，发现通用模型表现低迷，MarScope等领域微调模型显著提升，但整体仍属挑战性任务；

**⚠️ 局限性**

局限在于数据规模有限、模型对火星细粒度特征捕捉不足、评估依赖硬编码阈值与人力校验，且未涵盖更多多模态任务。

---

## 791. Joint Time Series Chain: Detecting Unusual Evolving Trend across Time Series

**arXiv ID:** 2602.13649 | [PDF](https://arxiv.org/pdf/2602.13649v1)

**作者:** Li Zhang `[一作]` (George Mason University), Jessica Lin `[通讯]` (George Mason University)

**通讯引用:** 9678 | [OpenAlex ID](https://openalex.org/A5101558875)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究了一种新定义的联合时间序列链（JTSC），用于检测两个相关时间序列间的异常演化趋势，并给出了相应的算法和排名标准。

**💡 创新点**

创新点包括：①提出联合时间序列链的概念，利用双向连接与 join matrix profile 的交叉节点克服单序列链对中断与噪声敏感的问题；②设计基于 Reference Deviation、Fluctuation 与 Evolve 的三项评分体系，实现对异常演化链的高效排序。

**🔧 技术方法**

采用 Z 归一化欧氏距离计算距离概况，构造左、右和 join 矩阵概况；基于这些概况提取逆向/正向链并合并为联合链；随后通过节点过滤与三项评分完成链的筛选与排序。

**📊 数据集**

使用 17 个来自 UCR 归类数据集（传感器、ECG、设备）生成的合成时间序列，并以 Intel 生产工厂的焊接炉温度数据作为真实案例进行验证。

**📈 对比分析**

与传统的 TSC17（双向链）和 TSC20（逆向链）在合成数据上对比，采用 hit‑rate 评估方法；JTSC 在 15/17 数据集上取得更高 hit‑rate，说明能更好地捕捉异常演化；在 Intel 数据中，JTSC 能提前约 25 分钟识别出流涌事件。

**⚠️ 局限性**

局限性包括：目前仅支持两条时间序列的联合链，未扩展到多序列情况；对子序列长度和阈值参数的选择较为敏感；实验缺乏真实标注的异常演化数据，验证程度受限。

---

## 792. Network-Adaptive Cloud Preprocessing for Visual Neuroprostheses

**arXiv ID:** 2602.13216 | [PDF](https://arxiv.org/pdf/2602.13216v1)

**作者:** Jiayi Liu `[一作]` (University of California Santa Barbara), Michael Beyeler `[通讯]` (University of California Santa Barbara)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种基于云的视觉预处理管线，并提出了网络自适应编码策略，动态调节图像分辨率、压缩率和传输间隔，以在不同网络条件下保持视觉神经假体的时序连续性。

**💡 创新点**

将云端推理视为感知受限的控制问题，引入闭环网络反馈调节机制，首次证明在网络拥塞时可以显著降低端到端延迟，同时仅轻微牺牲全局场景结构，并明确了可接受的运作阈值。

**🔧 技术方法**

使用 Raspberry Pi 4 作为边缘 VPU，gRPC/HTTP2 进行通信，PIDNet 进行语义分割，基于移动平均 RTT 的离散分层调度策略，以及 SSIM 与 BF 评估指标。

**📊 数据集**

采用自制的 egocentric 视频序列（Raspberry Pi 捕获），并在实验室网络仿真环境下构造五种 4G/5G 混合情景；未使用公开视觉数据集。

**📈 对比分析**

与静态固定参数（高分辨率、高压缩质量、固定帧率）基线进行对比；在极度拥塞 4G 下，适应策略将平均 RTT 降低约 60‑70%，服务器推理时间从 118 ms 降至 19 ms；SSIM 仅下降约 3 %，BF 分数在最差情形下降到 17 %，但在优质网络下恢复到基线水平。

**⚠️ 局限性**

评估仅基于算法指标，缺乏人类感知实验；自适应降解通过输入质量间接实现语义细节控制，未来可改为直接请求不同细节级别；控制策略采用离散阈值，未利用预测模型或多种反馈信号，可能导致切换抖动。

---

## 793. ROAST: Rollout-based On-distribution Activation Steering Technique

**arXiv ID:** 2602.14143 | [PDF](https://arxiv.org/pdf/2602.14143v1)

**作者:** Xuanbo Su `[一作]` (Bairong Inc.), Lijun Zhang `[通讯]` (Bairong Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ROAST，一种基于激活干预的框架，通过在模型自身分布上生成rollout并提取对比方向来实现参数高效控制；

**💡 创新点**

创新点包括：①在分布内生成对比样本（ROC）消除教师强制与自由生成的分布偏差；②使用连续软缩放（CSS）替代硬稀疏化，保留激活能量；③分组均值归一化（Grouped Mean Normalization）降低幅度和样本数量不均导致的偏差；

**🔧 技术方法**

采用rollout采样、激活提取、对比方向估计、CSS连续归一化、分组归一化、以及在残差流中插入向量的干预技术；

**📊 数据集**

在9个基准数据集上评估：SST2、SST5、MMLU、TruthfulQA、Winogrande、XNLI、GSM8K、MATH500、IFEval（以及Gemma3‑1B等模型）；

**📈 对比分析**

与无干预基线、few-shot、CAA、SADI等方法对比，ROAST在多模型（0.6B–32B）和多任务上均实现显著提升，例如Qwen3‑0.6B GSM8K提升+9.7%，GLM4‑32B TruthfulQA提升+12.1%，并且方差更小、性能接近或超过few‑shot；

**⚠️ 局限性**

局限性：需额外rollout计算，成本高；依赖任务专用verifier，难以扩展到开放式生成；仅适用于可验证的分类/推理任务，对高度非线性或创意任务效果未知；对干预强度、位置等超参数仍敏感。

---

## 794. An Explainable Failure Prediction Framework for Neural Networks in Radio Access Networks

**arXiv ID:** 2602.13231 | [PDF](https://arxiv.org/pdf/2602.13231v1)

**作者:** Khaleda Papry `[一作]` (Dalhousie University), Israat Haque `[通讯]` (Dalhousie University)

**通讯引用:** 974 | [OpenAlex ID](https://openalex.org/A5047687728)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个基于SHAP可解释性特征修剪与模型精简的5G无线接入网络 RLF 预测框架

**💡 创新点**

创新点在于将可解释性特征重要性直接用于特征修剪和架构简化，使模型变得更轻量化且性能提升

**🔧 技术方法**

使用 SHAP Sampling Explainer、GNN‑Transformer（GenTrap）、LSTM 以及模型简化技术

**📊 数据集**

采用 Turkcell 开源 5G RAN 数据集，包含城市与农村两套场景，分别约 1.8M 与 0.4M 观测点

**📈 对比分析**

与原始 GenTrap、LSTM+ 对比，轻量化 Transformer（LTrans）在城市和农村均取得更高的 F1 分数（4%–20% 提升），LLSTM+ 在农村提升 13%；模型参数从 13.4K/154.7K 减少至 5.8K/11K，保持或提升准确率

**⚠️ 局限性**

局限在于时间序列长度短、仅提供特征重要性解释、未考虑反事实解释；未来需在更长、更复杂的数据上验证并引入反事实解释

---

## 795. DAIAN: Deep Adaptive Intent-Aware Network for CTR Prediction in Trigger-Induced Recommendation

**arXiv ID:** 2602.13971 | [PDF](https://arxiv.org/pdf/2602.13971v1)

**作者:** Zhihao Lv `[一作]` (Xianyu of Alibaba), Jufeng Chen `[通讯]` (Xianyu of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种深度自适应意图感知网络（DAIAN），通过分层建模用户意图分布、提取显式与隐式多元意图，并利用语义与ID混合相似度增强和自适应选择实现触发式推荐（TIR）的精准点击率预测。

**💡 创新点**

①将用户意图建模为对不同相似度级别的分布，克服传统方法的意图短视；②采用语义+ID混合相似度增强，缓解TIR中交互稀疏问题；③设计自适应位移网络根据意图动态调节权重；④引入三阶段预训练策略，解决端到端训练难收敛。

**🔧 技术方法**

多头注意力（DIN/DIEN）+ MLP，Softmax分布估计，KL散度预训练，余弦相似度与多维投影的混合增强，门控位移单元（SU），三阶段预训练（用户意图、意图提取、CTR）和Qwen‑VL多模态预训练嵌入。

**📊 数据集**

阿里闲鱼（Xianyu）工业级TIR日志（每日10亿条，8天）和阿里妈妈广告数据集（26M条，8天）。

**📈 对比分析**

与DIN、DIEN、DIHN、DIAN、DEI2N、POSO、MWUF等基线进行对比；评估指标为AUC、GAUC、相对提升；在Xianyu上相较DIN提升5.75% AUC、8.65% GAUC；在阿里妈妈提升3.76% AUC、5.72% GAUC；在线A/B实验显示CTR+1.59%、多样性+1.73%、营收+2.37%。

**⚠️ 局限性**

模型依赖高质量多模态预训练嵌入和触发项日志，可能在缺乏多模态信息或新用户场景下表现不足；三阶段训练与多模块设计增加实现与调优复杂度；在其他域或小规模数据集的泛化性待进一步验证。

---

## 796. MacroGuide: Topological Guidance for Macrocycle Generation

**arXiv ID:** 2602.14977 | [PDF](https://arxiv.org/pdf/2602.14977v1)

**作者:** Alicja Maksymiuk `[一作]` (University of Oxford), İsmail İlkan Ceylan `[通讯]` (AITHYRA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出一种基于扩散模型的拓扑引导方法MacroGuide，用来生成任意宏环分子。

**💡 创新点**

创新点在于利用持久同调的Vietoris‑Rips复形对扩散过程进行实时梯度引导，实现无训练、轻量化且可控的宏环生成。

**🔧 技术方法**

核心技术包括扩散式分子生成、持久同调特征提取、梯度引导机制以及稀疏梯度处理。

**📊 数据集**

实验数据集主要为GEOM‑Drug（含0.14%宏环），并在MolDiff与MolSnapper模型上验证。

**📈 对比分析**

与自定义基线（微调、简单约束）以及无引导对比，MacroGuide在无条件和蛋白质条件下的宏环生成率分别从≈0–5%提升至≈99%，同时在化学有效性、多样性、PoseBusters和配体配位一致性等指标上达到或超过现有最优水平。

**⚠️ 局限性**

局限性包括计算开销（Vietoris‑Rips计算的二次复杂度）、梯度稀疏导致的稳定性问题、对环大小的控制仍需经验设定，以及缺乏对合成可行性与毒性等更深入评估。

---

## 797. Tool-Aware Planning in Contact Center AI: Evaluating LLMs through Lineage-Guided Query Decomposition

**arXiv ID:** 2602.14955 | [PDF](https://arxiv.org/pdf/2602.14955v1)

**作者:** Varun Nathan `[一作]` (Observe.AI), Ayush Kumar `[通讯]` (Observe.AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出针对客服中心的工具感知计划生成框架和基准，定义可执行的 JSON 计划语法、双轨评估方法（度量评估器 + 一次性评估器），并通过评估器‑优化器循环自动生成多版本计划谱。

**💡 创新点**

① 针对并行工具使用的领域特定评估与度量框架；② 评估器‑优化器循环实现无人工干预的计划谱生成；③ 在 14 种 LLM 上系统评估工具选择、提示一致性与执行合理性，并与手工参考计划做对比。

**🔧 技术方法**

使用多种大语言模型（GPT‑4o、Claude‑3‑7‑Sonnet、Llama‑3‑70B 等）生成计划；LLM 驱动的多度量评估器与一次性评估器；基于 JSON 的计划表示；评估器‑优化器循环（步骤诊断、标签化、局部修正）来迭代改进计划。

**📊 数据集**

600 条保密的真实客服查询与计划谱（训练20 / 验证80 / 测试500），以及公开的 200 条合成查询及其计划谱；计划谱通过迭代循环自动生成并经过人工审核。

**📈 对比分析**

对比无谱与有谱提示下的计划生成，使用一次性评估器将计划分为 A+、A、B 三个质量层级；14 种 LLM 的最高 A+ 仅约 49%。采用度量评估器得到整体分数，最高为 Claude‑3‑7‑Sonnet 84.8/100。评估器‑优化器循环可将最高分层提升约 10% 以上。

**⚠️ 局限性**

仅适用于客服中心的三种工具（T2S、RAG、LLM）与特定 JSON 语法；基准数据保密，难以直接复现；评估器为 LLM，可能存在偏差；未实现执行器‑评估器‑优化器的实时重规划；未考虑成本/延迟优化；工具集不包含更丰富的业务工具。

---

## 798. The Value of Effective Pull Request Description

**arXiv ID:** 2602.14611 | [PDF](https://arxiv.org/pdf/2602.14611v1)

**作者:** Shirin Pirouzkhah `[一作]` (University of Zurich), Alberto Bacchelli `[通讯]` (University of Zurich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 Pull Request 描述进行灰色文献综述、构建八元素分类体系，并在 156 个 GitHub 项目 80K 条 PR 上通过混合效应回归分析描述元素与代码审查结果的关联，同时开展问卷调查收集开发者对描述重要性的主观看法，并研究描述出现与特定元素的预测因素。

**💡 创新点**

首次系统性地将实践指南中的 PR 描述要素归纳为统一分类，并结合大规模实证与开发者感知，揭示描述中交互导向元素（如指定反馈类型）对合并率和审查参与度的显著正向影响；同时提出描述写作是情境适应性行为而非形式化惯例。

**🔧 技术方法**

采用 LLaMA 3.1-70B 自动识别描述元素，混合效应线性/逻辑回归模型分析结果，问卷调查与文本编码分析。

**📊 数据集**

GitHub GHTorrent 2019 版本数据（156 个公开项目，80K 条已关闭 PR），配合 SonarQube 静态代码质量指标、GitHub API 获取审查评论、提交记录，另外使用 64 份开发者调查问卷。

**📈 对比分析**

通过对比基线模型与控制模型（加入项目、代码、作者等控制变量），发现描述元素对合并率、第一响应时间、审查评论数等指标的效应显著但大小有限；交互导向元素在控制模型中仍保持 1.64–1.73 倍的合并几率提升；整体模型解释度从 3% 线性到 46% 逻辑模型提升，AUC 在 0.7–0.9 之间。

**⚠️ 局限性**

数据来源为 2019 年前的 GHTorrent，无法反映近期生成式 AI 对 PR 描述习惯的影响；LLaMA 识别准确性虽高（0.85 准确率）但对“reason”元素效果差；自选调查可能偏向更活跃或有经验的开发者；模型可能存在未测量的混杂因素；研究仅限于 GitHub 开源项目，结果对私有或其他平台的推广受限。

---

## 799. Automated Classification of Source Code Changes Based on Metrics Clustering in the Software Development Process

**arXiv ID:** 2602.14591 | [PDF](https://arxiv.org/pdf/2602.14591v1)

**作者:** Evgenii Kniazev `[一作]` `[通讯]`, Evgenii Kniazev

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

基于度量聚类实现源代码变更的半自动分类。

**💡 创新点**

首次证明可用k‑means+余弦相似度对代码变更度量向量进行聚类，显著减少人工审查工作量。

**🔧 技术方法**

k‑means聚类、余弦相似度、11个源代码度量、CLUTO工具、Bootstrap统计检验。

**📊 数据集**

五个软件系统（Subversion、NHibernate、Navi‑Manager、LRIT、e‑Tutor 5000）以及公开项目的变更集。

**📈 对比分析**

与专家手工分类对比，平均纯度0.75±0.05，熵0.37±0.06，验证通过Bootstrap置信区间，显著提升效率。

**⚠️ 局限性**

对不同类型变更的分离不足，尤其是重构类别；需增加度量或适配不同类别集。

---

## 800. Empty Shelves or Lost Keys? Recall Is the Bottleneck for Parametric Factuality

**arXiv ID:** 2602.14080 | [PDF](https://arxiv.org/pdf/2602.14080v1)

**作者:** Nitay Calderon `[一作]` (Technion Israel Institute of Technology), Gal Yona `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了知识配置框架，将事实知识拆分为编码与回忆两种行为层面，并构建了对应的评估基准；

**💡 创新点**

通过行为学定义编码/回忆、引入5种知识配置、揭示“反转诅咒”是回忆失败、证明思考机制可显著恢复已编码但难以召回的知识；

**🔧 技术方法**

采用提示式评估、LLM自评、思考优化模型（如 Gemini‑3、GPT‑5）以及链式思考（CoT）来衡量编码与回忆；

**📊 数据集**

自动生成的 KProfile 基准：2150 条自然文档（维基百科）事实，每条配 10 个问题，涵盖编码、回忆与多项选择验证；

**📈 对比分析**

对 13 种 LLM（Gemini‑3、GPT‑5、GPT‑4.1、Gemma‑3 等）进行评测，发现前沿模型编码率≈95–98%，但直接回忆率仅 66–75%；思考提升 40–65% 的已编码但未被直接召回的事实；

**⚠️ 局限性**

局限性在于仅依赖提示和自评，未直接检验内部表征；基准聚焦单跳事实，缺乏对多跳推理或非维基百科知识的评估；

---

## 801. VSAL: A Vision Solver with Adaptive Layouts for Graph Property Detection

**arXiv ID:** 2602.13880 | [PDF](https://arxiv.org/pdf/2602.13880v1)

**作者:** Jiahao Xie `[一作]` (University of Delaware), Guangmo Tong `[通讯]` (University of Delaware)

**通讯引用:** 867 | [OpenAlex ID](https://openalex.org/A5011783006)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于视觉的图属性检测框架VSAL，能够自适应生成图布局并通过图像分类器检测属性。

**💡 创新点**

创新在于将可微分布局生成器与生成对抗网络相结合，让生成的布局在判别器与分类器的反馈下学习，从而显著提升对 Hamiltonian、planarity、claw‑freenness、tree 等属性的检测效果。

**🔧 技术方法**

使用了图卷积/Graphormer编码器、Wasserstein GAN对抗训练、可微分可视化模块（节点/边渲染+高斯平滑）、ResNet‑50/ViT/EfficientNet分类器。

**📊 数据集**

利用House of Graphs 的小/中型图集以及自制的 401‑1000 节点的合成大/超大图集。

**📈 对比分析**

与 VN‑Solver、Graphormer、Graphormer‑GD、EquiformerV2、GraphsGPT 等现有方法进行对比；在四个属性检测任务上，VSAL 在大/超大图上往往取得最高 F1 分数，并且推理速度快、显存占用低。

**⚠️ 局限性**

局限包括：对极大图仍需提升可视化分辨率；对任务专用的矩阵方法可能在特定属性上更强；生成器依赖先验布局与判别器训练，可能在不同任务间迁移困难。

---

## 802. ManeuverNet: A Soft Actor-Critic Framework for Precise Maneuvering of Double-Ackermann-Steering Robots with Optimized Reward Functions

**arXiv ID:** 2602.14726 | [PDF](https://arxiv.org/pdf/2602.14726v1)

**作者:** Kohio Deflesselle `[一作]` (University of Bordeaux), Olivier Ly `[通讯]` (University of Bordeaux)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了ManeuverNet，一种面向双Ackermann转向机器人的端到端深度强化学习框架，能够在无专家数据和无手工指导的情况下实现精确的机动控制。

**💡 创新点**

创新点在于为双Ackermann机器人设计了四种专门的奖励函数（尤其是ℛ_HS），克服了传统欧几里得奖励导致的子最优策略问题，并将Soft Actor-Critic与CrossQ结合提升了样本效率。

**🔧 技术方法**

使用了Soft Actor-Critic算法与CrossQ叠加的强化学习框架，并在PyBullet、Gazebo仿真以及真实机器人上验证。

**📊 数据集**

训练和评估使用的“数据集”是基于8×8 m工作空间的随机目标分布（4×4 m目标区域）以及真实机器人在4.2 m×4.2 m工作空间内的随机目标，未使用公开数据集。

**📈 对比分析**

通过与标准SAC奖励、FastRLap以及TEB规划器对比，ManeuverNet在成功率、平均误差和路径长度比（SPL）上优于所有DRL基线，接近TEB的成功率但SPL提高高达90%，同时显著降低导航时间。

**⚠️ 局限性**

主要局限包括缺乏障碍物检测与避免、未考虑最终姿态约束，以及仅在无障碍、无重置的环境中验证。

---

## 803. Selective Synchronization Attention

**arXiv ID:** 2602.14445 | [PDF](https://arxiv.org/pdf/2602.14445v1)

**作者:** Hasi Hays `[一作]` `[通讯]` (University of Arkansas), Hasi Hays (University of Arkansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出一种基于 Kuramoto 互耦振荡器稳态解的注意力机制（Selective Synchronization Attention，SSA），并在 Transformer 框架中实现了可替换的 Oscillatory Synchronization Network（OSN）模块。

**💡 创新点**

创新点：1）将注意力权重映射到物理可解释的同步强度；2）天然稀疏化（相位锁定阈值）不需要外部掩码；3）用可学习的自然频率谱统一位置编码与语义编码；4）单次前向计算，无需迭代 ODE；5）提供与生物神经同步机制一致的理论解释。

**🔧 技术方法**

技术：Kuramoto 模型的稳态解、频率依赖耦合、相位锁定条件、闭式同步矩阵计算、多频同步头、与 Transformer 的无缝对接（Pre‑norm、残差+FFN），以及 GPU 上的单块性能基准（NVIDIA A100）。

**📊 数据集**

未使用真实数据集；实验仅在随机输入上评估前向计算效率和同步矩阵结构。对比使用标准 Transformer 块。

**📈 对比分析**

对比方法：在单块层级上与标准 Transformer（密集和稀疏）进行吞吐量、延迟和显存占用对比。结果显示：参数量基本相同；OSN（密集）吞吐量约为 Transformer 的 0.3–0.6 倍，显存占用 1.1–4.0 倍；稀疏版通过 top‑k 进一步降低吞吐量但显存几乎不变。总体性能仍低于经过优化的标准注意力实现。

**⚠️ 局限性**

局限性：1）当前实现未进行 kernel‑level 优化，导致吞吐量与显存不足；2）密集模式下仍是 O(N²) 复杂度，虽然理论上可通过稀疏化降低；3）同步矩阵的近似来自均值场，有限 N 时精度受限；4）缺乏在真实任务（如语言建模、长序列推断）上的验证；5）尚未探索多模态、硬件映射等进一步应用。

---

## 804. The Agent Economy: A Blockchain-Based Foundation for Autonomous AI Agents

**arXiv ID:** 2602.14219 | [PDF](https://arxiv.org/pdf/2602.14219v1)

**作者:** Minghui Xu `[一作]` (Shandong University), Minghui Xu `[通讯]` (Shandong University)

**通讯引用:** 1611 | [OpenAlex ID](https://openalex.org/A5103077343)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

提出Agent Economy框架，将自治AI代理视为具有独立法律地位的经济主体，并设计了包含物理基础、身份、认知工具、经济结算与集体治理的五层体系；

**💡 创新点**

首次系统化地将区块链技术与自治AI代理结合，构建代理身份自治、无需人工介入的资产持有与支付机制，并通过Agentic DAO实现多代理协同治理；

**🔧 技术方法**

利用区块链核心技术（ERC‑4337账户抽象、DePIN、W3C DID、MCP、RAG、零知识证明、可信执行环境、Layer‑2 Rollup等）来实现权限、结算与微支付；

**📊 数据集**

论文以设计为主，未使用公开实验数据集；若需示例，可参考IPFS/Arweave中的去中心化知识库和可信硬件提供的算力/能耗元数据；

**📈 对比分析**

本研究为理论与架构设计，未进行量化实验对比，预期通过Layer‑2技术实现百万TPS、<100 ms延迟的微支付和可信结算，实际性能需后续实验验证；

**⚠️ 局限性**

主要局限包括缺乏实证验证、Oracle 2.0与零知识证明实现难度大、治理与安全机制待完善，以及法律责任、风险与人机共存问题尚未解决。

---

## 805. Parametric-Sensitivity Aware Retransmission for Efficient AI Downloading

**arXiv ID:** 2602.13607 | [PDF](https://arxiv.org/pdf/2602.13607v1)

**作者:** You Zhou `[一作]` (University of Hong Kong), Kaibin Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 20939 | [OpenAlex ID](https://openalex.org/A5007131492)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对 AI 模型下载的参数敏感度感知重传协议（PASAR），通过自适应阈值和在线多选背包问题求解，来降低重传延迟并保持模型推理精度。

**💡 创新点**

创新点在于：①将模型参数的 Hessian 敏感度融入重传决策；②设计基于敏感度的自适应停止阈值和两阶段（阈值过滤+贪心精细化）在线重传控制；③用在线多选背包模型刻画重传最优化问题，首次实现对不同参数包的异质重传。

**🔧 技术方法**

采用了 Hessian 计算参数敏感度、敏感度感知重传协议（PASAR）、在线多选背包（MCKP）求解、阈值自适应更新、贪心精细化、以及对比的 HARQ-I/CC/IR 基线。

**📊 数据集**

实验数据集包括 MNIST（训练 LeNet-5）和 CIFAR-10（训练 ShuffleNetV2），用于评估模型下载延迟和推理精度。

**📈 对比分析**

与 HARQ-I、HARQ-CC、HARQ-IR 在不同 SNR、包大小和剪枝率下进行对比。结果显示：在低 SNR 或大模型场景下，PASAR 可将重传延迟降低约 45%（vs. CC）和 35%（vs. IR）；随着剪枝率增加，敏感度分布趋于均匀，性能优势逐渐衰减。

**⚠️ 局限性**

局限性：①需要事先计算并存储每个参数的 Hessian 敏感度，计算和存储开销不可忽视；②假设参数敏感度分布高度右偏，若分布趋于均匀时优势下降；③未考虑能量消耗、功率控制和多设备分布式推理等实际部署因素；④在极端剪枝或极小参数集时，PASAR 与传统方法差距缩小。

---

## 806. RoboAug: One Annotation to Hundreds of Scenes via Region-Contrastive Data Augmentation for Robotic Manipulation

**arXiv ID:** 2602.14032 | [PDF](https://arxiv.org/pdf/2602.14032v1)

**作者:** Xinhua Wang `[一作]` (Beijing Innovation Center of Humanoid Robotics), Jian Tang `[通讯]` (Beijing Innovation Center of Humanoid Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了RoboAug框架，通过仅使用单帧框注释即可自动生成高质量任务相关区域，结合语义背景合成与region‑contrastive损失，显著提升机器人在未见场景中的泛化能力。

**💡 创新点**

创新点包括① 单帧一锤区域匹配策略，实现精准任务相关区域提取；② 直接在多背景上合成前景而非依赖不稳定的inpainting；③ 在视觉编码器中加入region‑contrastive loss，促使同类别特征聚类并抑制干扰。

**🔧 技术方法**

利用GroundingDINO、DINOv2和SAM2实现区域匹配；预训练扩散模型（如Stable Diffusion）进行背景生成；region‑contrastive loss与常规视觉运动策略结合；在多机器人平台上训练和评估。

**📊 数据集**

使用了自建的RoboAug‑D检测数据集（7,576轨迹，33任务）作为验证集，并在UR‑5e、AgileX、Tien Kung 2.0三台机器人上进行超过35k次真实轨迹实验。

**📈 对比分析**

与无增广、弱增广、GenAug、Mirage、RoVi‑Aug等基线对比，RoboAug在背景、光照、干扰三类 OOD 场景中成功率提升显著：UR‑5e从0.09→0.47、AgileX从0.16→0.60、Tien Kung 2.0从0.19→0.67，整体提升约59‑76%。

**⚠️ 局限性**

局限性：依赖预训练模型的区域匹配精度，极端遮挡或高度复杂交互场景仍可能产生误提取；单帧注释仍需人工；未在低光、雨雪等极端环境中充分验证。

---

## 807. GREPO: A Benchmark for Graph Neural Networks on Repository-Level Bug Localization

**arXiv ID:** 2602.13921 | [PDF](https://arxiv.org/pdf/2602.13921v1)

**作者:** Juntong Wang `[一作]` (Peking University), Muhan Zhang `[通讯]` (Peking University)

**通讯引用:** 4815 | [OpenAlex ID](https://openalex.org/A5071515223)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GREPO benchmark，构建了 86 个 Python 仓库、47,294 个真实 bug‑fix 实例的时序图结构，专门为仓库级缺陷定位任务设计，并提供增量构建、anchor 节点选择以及查询‑节点相似度特征。

**💡 创新点**

①首次为 GNN 提供专门的 bug‑localization 基准；②将仓库演化建模为时间索引的异构图，极大提升构建效率；③引入 LLM‑驱动的 anchor 生成和临时 anchor，增强多跳推理；④用 GNN 直接对图结构进行学习，明显优于传统 IR/LLM 方法。

**🔧 技术方法**

使用多种 GNN（GCN、GIN、GraphSAGE、GAT、GATv2、GPS、GatedGCN 等）进行消息传递学习；利用 Qwen3‑Embedding‑8B 生成节点文本嵌入和查询‑节点相似度特征；采用增量图构建与时间戳管理；通过 Anchor 采样与子图推理实现大规模推断；在 0‑shot 设定下进行规模分析。

**📊 数据集**

GREPO 数据集：86 个公开 Python 仓库、47,294 个 bug‑fix 任务，包含问题文本、PR 合并记录、代码树、调用关系、继承关系等；对比基准使用 9 个代表性仓库进行测试。

**📈 对比分析**

通过 Hit@1/5/10/20 等 recall 指标与 LLM/IR 基线（LocAgent、Agentless、CF‑RAG）比较。所有 GNN 在 9 个测试仓库上均超越基线，注意力机制模型（GAT/GATv2）效果最佳；零 shot 训练中 GAT 随训练仓库数增加逐步提升，表明 GREPO 具备良好的可扩展性；消融实验验证了特征、边类型和 anchor 选择对性能的贡献。

**⚠️ 局限性**

仅覆盖 Python，难以直接推广至其他语言；anchor 选取可能漏掉真实修复节点，导致 hit 下降；依赖静态分析与 AST，无法捕获动态运行时依赖；大规模代码库仍需子图裁剪，可能损失全局上下文；未考虑多语言或跨文件的多模态信息。

---

## 808. Min-Max Connected Multiway Cut

**arXiv ID:** 2602.13861 | [PDF](https://arxiv.org/pdf/2602.13861v1)

**作者:** Hans Raj Tiwary `[一作]` (Charles University), Petr Kolman `[通讯]` (Charles University)

**通讯引用:** 723 | [OpenAlex ID](https://openalex.org/A5080963999)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并研究了最小最大连通多路切割（min‑max connected multiway cut）问题，探讨其在不同图类上的计算复杂性并给出多种算法。

**💡 创新点**

创新点包括：①首次将连通性约束与最小最大边权目标结合；②证明该问题在树宽为2或3的图上弱 NP‑hard；③给出树上的伪多项式时间动态规划和 FPTAS；④展示问题的扩展复杂度为超多项式，排除多项式 LP 形式；⑤证明树上精确成本判定为 NP‑hard。

**🔧 技术方法**

主要技术手段包括：基于分治与动态规划的树 DP；整数规划的缩放与舍入构造 FPTAS；核化与树宽简化实现 FPT；从 Partition、Subset‑Sum 等经典 NP‑hard 问题的多项式归约；以及多面体扩展复杂度的构造与上界证明。

**📊 数据集**

论文未使用公开数据集，而是构造了若干理论图形（如 K_{3,n}、带权树、带权五节点子树等）来证明硬度与算法正确性。

**📈 对比分析**

由于研究聚焦理论复杂度与算法设计，未进行实验对比；理论结果表明在树宽≤3的图上问题为弱 NP‑hard，在树上实现了伪多项式时间解法和任意精度 FPTAS，说明在可接受的误差范围内可高效求解。

**⚠️ 局限性**

主要局限在于：①树上问题是否为弱 NP‑hard尚未确定；②仅给出了伪多项式时间与 FPTAS，缺乏真正多项式时间算法；③扩展复杂度结果说明不能用小规模线性规划求解；④论文未提供实验验证，难以评估算法在实际大规模实例中的表现。

---

## 809. Evaluating Prompt Engineering Techniques for RAG in Small Language Models: A Multi-Hop QA Approach

**arXiv ID:** 2602.13890 | [PDF](https://arxiv.org/pdf/2602.13890v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 810. YOLO26: A Comprehensive Architecture Overview and Key Improvements

**arXiv ID:** 2602.14582 | [PDF](https://arxiv.org/pdf/2602.14582v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 811. LLM-Confidence Reranker: A Training-Free Approach for Enhancing Retrieval-Augmented Generation Systems

**arXiv ID:** 2602.13571 | [PDF](https://arxiv.org/pdf/2602.13571v1)

**作者:** Zhipeng Song `[一作]` (Dalian University of Technology), Heng Qi `[通讯]` (Dalian University of Technology)

**通讯引用:** 3482 | [OpenAlex ID](https://openalex.org/A5087744818)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LLM-Confidence Reranker（LCR），一种训练‑free、可直接插拔的重排算法，通过利用大型语言模型（LLM）的置信度信号来提升检索增强生成（RAG）系统的文档排序效果。

**💡 创新点**

创新点在于：①基于最大语义聚类比例（MSCP）从LLM的生成一致性提取置信度；②将置信度进行分箱并多层排序，以在低置信度查询下强化文档相关性；③无须额外训练、仅使用7‑9B参数LLM即可实现。

**🔧 技术方法**

核心技术包括：多元抽样+语义聚类（使用NLI判定相似度）计算MSCP；置信度分箱（高/中/低）；基于置信度阈值的分段排序；以及对现有重排器的无缝后置融合。

**📊 数据集**

在BEIR（包含NQ、DBPE、FEVER、SciDocs、Touché、NFCorpus）和TREC DL19/20数据集上进行评估，检索器使用BM25和Contriever，重排器涵盖预训练LLM（YesNo、QLM、RankGPT）和微调Transformer（ColBERT、Cross‑Encoder、RankT5）。

**📈 对比分析**

与基线检索+重排器对比，LCR在NDCG@5上平均提升约3%（BM25）或3.6%（Contriever），在某些组合上提升高达20.6%，且从未出现性能下降，证明其在多种检索器/重排器上的稳健性。

**⚠️ 局限性**

局限性包括：对置信度阈值和文档分箱阈值的手工设定；在已具高性能微调重排器上提升幅度有限；对LLM模型的选择敏感，部分模型（如Qwen7B）提升不如InternLM7B；未深入探究多模型协同或自适应阈值的可能性。

---

## 812. What hackers talk about when they talk about AI: Early-stage diffusion of a cybercrime innovation

**arXiv ID:** 2602.14783 | [PDF](https://arxiv.org/pdf/2602.14783v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 813. Concepts' Information Bottleneck Models

**arXiv ID:** 2602.14626 | [PDF](https://arxiv.org/pdf/2602.14626v1)

**作者:** Karim Galliamov `[一作]` (University of Amsterdam), Adín Ramírez Rivera `[通讯]` (University of Oslo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出在概念瓶颈模型（CBM）的概念层上加入信息瓶颈正则化，压缩输入-概念互信息I(X;C)的同时保留概念-标签互信息I(C;Y)，从而获得更精简、更具判别力的概念表示，提升模型的可解释性和预测准确率。

**💡 创新点**

创新点在于：①首次将信息瓶颈原理直接应用到CBM的概念层，形成概念信息瓶颈模型（CIBM）；②在保持原有网络结构不变的前提下，提出两种可实现的正则化形式——变分目标和熵基近似，实现无额外监督的最小-充分概念学习；③通过信息平面分析证明压缩与表达的分离能有效降低概念泄漏并提升干预效果。

**🔧 技术方法**

使用的技术包括：信息瓶颈理论、变分推断、互信息估计与熵近似、对CBM训练流程的正则化改造、实验评估指标（OIS/NIS、AUC_TTI/NAUC_TTI）以及对比实验框架。

**📊 数据集**

实验采用的视觉基准数据集有CUB、AwA2和aPY，涵盖不同难度和概念标注量的图像分类任务。

**📈 对比分析**

在相同网络骨干、数据与训练配置下，本文将CIBM与六大CBM族（硬/软、联合/独立、ProbCBM、IntCEM、AR-CBM）以及黑盒基准进行对比；实验结果显示，CIBM在类别准确率、概念准确率、概念泄漏（OIS/NIS）以及干预曲线（AUC_TTI/NAUC_TTI）均显著优于对应的未正则化版本，部分场景甚至超过传统黑盒模型。

**⚠️ 局限性**

局限性包括：①需手动调节信息瓶颈权重β，超参数敏感；②在某些简单数据集（如AwA2）提升幅度有限；③互信息估计和变分近似带来额外计算开销；④实验仅覆盖视觉任务，缺乏跨领域验证。

---

## 814. Can a Lightweight Automated AI Pipeline Solve Research-Level Mathematical Problems?

**arXiv ID:** 2602.13695 | [PDF](https://arxiv.org/pdf/2602.13695v1)

**作者:** Lve Meng `[一作]` (University of Science and Technology of China), Jiyan He `[通讯]` (Zhongguancun Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个轻量级自然语言管线，利用 Gemini 3 Pro、GPT‑5.2 Pro 等下一代大语言模型自动生成研究级数学证明，并在 ICCM 与 First Proof 题集上完成验证。

**💡 创新点**

提出了基于引用的验证机制，要求模型为每个重要推理提供具体文献引用，并对提示词进行领域特定优化，将 LLM 应用于竞赛级别之外的研究难题。

**🔧 技术方法**

核心技术包括自动化推理管线、域特定提示工程、引用增强验证和高阶抽象推理的 LLM 接入。

**📊 数据集**

使用的数据集包括 ICCM 三套问题集（尤其是前两套与 S‑T Yau 大赛相当）和包含十道未公开研究问题的 First Proof 题集；同时在 Kashiwara 的《Categories and Sheaves》练习题上做验证。

**📈 对比分析**

相较于以往仅在竞赛题上验证的管线，本文管线在 ICCM 1、2 号集上实现 100% 解决率，已对 First Proof 题集第 4 题进行完整验证，表明在研究级别题目上取得显著突破，但整体验证仍耗时。

**⚠️ 局限性**

主要局限在于验证瓶颈（单题验证耗时数小时）、需要专家参与、长上下文推理能力受限，以及对隐式数学知识的把握不足，导致对开放性高难题的解决仍受限。

---

## 815. CT-Bench: A Benchmark for Multimodal Lesion Understanding in Computed Tomography

**arXiv ID:** 2602.14879 | [PDF](https://arxiv.org/pdf/2602.14879v1)

**作者:** Qingqing Zhu `[一作]`, Zhiyong Lu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CT-Bench，一个包含2万多张CT病灶图像及其结构化文本描述、尺寸、边界框和多任务视觉问答的完整多模态基准。

**💡 创新点**

创新点在于首次提供了带有硬负样本的CT病灶定位、描述、尺寸估计、属性分类等七项任务的多模态评估框架，并通过GPT-4辅助标注实现高质量注释。

**🔧 技术方法**

采用多种视觉‑语言模型（如BiomedCLIP、Gemini、RadFM、LLaVA‑Med）进行评估，并在Lesion Image & Metadata Set上进行微调，结合CLIP对齐与多任务学习。

**📊 数据集**

数据集来源于DeepLesion的病灶框及HIS PACS报告，经过GPT‑4 + 人工审核构建出20,335个病灶、7,795份CT扫描和2,850条多任务问答。

**📈 对比分析**

与随机、未调优模型对比，微调后的BiomedCLIP在有BBox条件下平均准确率达62%，是所有模型中最佳表现，但整体仍低于临床专家水平。

**⚠️ 局限性**

主要局限包括：仍远低于专家水平、对三维体积推理能力不足、需要大量人工标注成本高、BBox信息对某些任务影响有限。

---

## 816. Attention Head Entropy of LLMs Predicts Answer Correctness

**arXiv ID:** 2602.13699 | [PDF](https://arxiv.org/pdf/2602.13699v1)

**作者:** Sophie Ostmeier `[一作]` (University Hospital Zurich), Akshay Chaudhari `[通讯]` (Stanford University)

**通讯引用:** 29136 | [OpenAlex ID](https://openalex.org/A5041175834)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于LLM注意力熵分布的白盒方法来预测回答的正确性，利用每个注意力头的2-Rényi熵作为特征，并通过稀疏逻辑回归生成概率预测。

**💡 创新点**

创新点在于把注意力熵（与梯度动力学的关系）与答案正确性关联，展示了注意力分布的扩散度比位置信息更能捕捉模型的训练稳定性，并能在无答案生成前就进行准确性预估。

**🔧 技术方法**

使用的技术包括：Transformer自注意力中的2-Rényi熵计算、稀疏L1正则化逻辑回归、Shapley值解释、对比基线方法（Lookback Lens、隐藏状态回归、Token概率等）以及校准评估（ECE）。

**📊 数据集**

实验数据集涵盖三种QA任务：TriviaQA（常识类）、HotpotQA（多跳推理）和MedMCQA（医学多选），每个任务采集5万样本，使用多种模型（Qwen3 1.7B/8B/32B、Llama3.1 8B、Llama3.2 3B）。

**📈 对比分析**

与基线相比，HeadEntropy在分布内AUROC平均提升0.07–0.21，校准误差最低；在跨域泛化时比Lookback Lens多提升约8.5%的AUROC；在回答生成前仅用问题 tokens 也能取得0.74 AUROC 的预测性能。

**⚠️ 局限性**

局限性：需要先有约1万条标注数据；仅适用于可量化正确性的QA任务，无法直接推广到开放式生成任务；对不同模型/任务的头特征选择仍需改进，且对部分头的正负贡献尚未得到完整理论解释。

---

## 817. Neuromem: A Granular Decomposition of the Streaming Lifecycle in External Memory for LLMs

**arXiv ID:** 2602.13967 | [PDF](https://arxiv.org/pdf/2602.13967v1)

**作者:** Ruicheng Zhang `[一作]` (Huazhong University of Science and Technology), Hai Jin `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 32362 | [OpenAlex ID](https://openalex.org/A5022262922)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Neuromem 测试平台，分解外部内存模块生命周期为五个设计维度，并在流式插入-检索协议下对多种内存系统进行评估。

**💡 创新点**

创新点在于把内存模块拆解为 D1–D5 五个可互换的细粒度组件，允许在统一堆栈上做精确归因，并引入实时流式评估协议，揭示插入与检索成本的相互转移。

**🔧 技术方法**

使用了外部内存结构（分区/层次）、归一化、合并、查询构造和上下文整合等技术，搭建了统一的服务栈，利用华为 Ascend 910B NPU 与 Nvidia A6000 GPU 实现异步计时。

**📊 数据集**

在三组代表性长时序推理数据集上测试（如 LongBench、LoCoMo、Minerva 等）。

**📈 对比分析**

通过对每个维度的可插拔替换与交叉验证，对比了 token‑level F1 与插入/检索延迟，结果显示不同设计在准确率与延迟上存在显著权衡；混合数据结构提升准确上限，生成式聚合与多查询扩展带来极高延迟但几乎无准确提升。

**⚠️ 局限性**

局限性包括实验仅覆盖有限的三组数据集，评估环境仍以单一硬件实现；对极大规模内存的长期稳定性和多任务干扰未充分验证；以及对隐私合规性的深度探讨仍不足。

---

## 818. Accelerated Discovery of Cryoprotectant Cocktails via Multi-Objective Bayesian Optimization

**arXiv ID:** 2602.13398 | [PDF](https://arxiv.org/pdf/2602.13398v1)

**作者:** Daniel Emerson `[一作]` (Carnegie Mellon University), Levent Burak Kara `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3603 | [OpenAlex ID](https://openalex.org/A5048339797)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过高通量筛选与多目标贝叶斯优化相结合，加速了冷冻保护剂（CPA）混合物的设计过程，旨在实现高浓度冰抑制与细胞存活率的双重目标。

**💡 创新点**

提出了基于不确定性感知的多目标贝叶斯优化框架，采用EHVI（期望超体积改进）及其方差校正变体进行批量采样，并结合k‑center主动学习有效挖掘组合空间，显著提升实验效率。

**🔧 技术方法**

使用高通量细胞存活率测定、XGBoost 回归与高斯过程作为代理模型、qLogNEHVI / qVarLogNEHVI 获得批量采集函数，并借助 BoTorch 实现算法迭代。

**📊 数据集**

初始数据集包含 535 条样本（70 条单组分 CPA + 465 条混合物），来自 T24 细胞系的细胞存活率实验，并在合成基准上进行验证。

**📈 对比分析**

与随机采样、qLogNParEGO 标量化基线以及 qLogNEHVI 对比实验显示，EHVI 方法在 8 轮迭代中使支配超体积提升 9.5%/4.5%（相对随机/基线），并在 IGD 指标上更低；仅使用约 30% 的评估次数，节省约 10 周实验时间。

**⚠️ 局限性**

受限于离散浓度范围（3.5–6 M，0.5 M 步进）、单细胞系（T24）和有限重复（n=3）的实验噪声，框架对更大范围或不同细胞类型的适用性尚待进一步验证。

---

## 819. Speculative Decoding with a Speculative Vocabulary

**arXiv ID:** 2602.13836 | [PDF](https://arxiv.org/pdf/2602.13836v1)

**作者:** Miles Williams `[一作]` (University of Sheffield), Stylianos I. Venieris `[通讯]` (Samsung AI Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态词表推测方法，结合投机解码显著提升大型语言模型的推理速度和接受长度

**💡 创新点**

创新点在于同时对下一词和输出词表进行上下文感知的动态推测，使用轻量级中间表示生成候选词表，从而在保持完整词表覆盖的前提下降低输出分布计算成本

**🔧 技术方法**

使用投机解码框架（EAGLE‑3）+自定义低维嵌入+top‑k 词表筛选+融合核加速点积

**📊 数据集**

在 Spec‑Bench 6 大任务（多轮对话、机器翻译、数学推理、摘要、检索增强生成、问答）上评估，并使用 Qwen3（4B/8B）和 OLMo2（1B/7B）四个开源模型

**📈 对比分析**

与自回归、EAGLE‑3、FR‑Spec、VocabTrim 等基线对比；在所有任务中获得最高接受长度（平均 +4.8%），平均吞吐量提升 3.1%–8.1%，最高实现 2.53× 的推理加速

**⚠️ 局限性**

实验受限于 Spec‑Bench 的英语主导任务，缺乏对非印欧语系或形态学差异大语言的验证

---

## 820. High-Fidelity Causal Video Diffusion Models for Real-Time Ultra-Low-Bitrate Semantic Communication

**arXiv ID:** 2602.13837 | [PDF](https://arxiv.org/pdf/2602.13837v1)

**作者:** Cem Eteke `[一作]` (Technical University of Munich), Eckehard Steinbach `[通讯]` (Technical University of Munich)

**通讯引用:** 9816 | [OpenAlex ID](https://openalex.org/A5077346002)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研发了一种模块化的、可实现实时、因果、超低比特率语义视频通信的扩散模型。

**💡 创新点**

创新点在于将语义控制、恢复适配器和时序适配器三模结合，并通过高效的时序适配器蒸馏实现仅需极少参数、仅几步即可生成视频，从而兼顾低比特率、实时性和语义保真。

**🔧 技术方法**

采用语义视频压缩（轮廓简化+量化）、低分辨率降质帧、Stable Diffusion backbone + ControlNet/Restoration Adapter、时序适配器，并利用分布匹配蒸馏（DMD）实现因果窗口化。

**📊 数据集**

在Cityscapes（10fps 256×512）和自制的YCB‑Sim（10fps 512×512）两个数据集上进行训练与评估。

**📈 对比分析**

与HEVC、VVC、DCVC、EVC、I2V‑SC等传统、神经和生成基准相比，本文在0.0003–0.01 bpp范围内实现了更高的LPIPS、DISTS、mIoU以及主观偏好，且实时帧率可达5.1 FPS。

**⚠️ 局限性**

限制在于帧率受限，难以扩展到更高帧率；同时依赖现有的Stable Diffusion等大型模型，未来需探索更高效或更大规模的生成骨干以进一步提升质量。

---

## 821. LLM-Powered Automatic Translation and Urgency in Crisis Scenarios

**arXiv ID:** 2602.13452 | [PDF](https://arxiv.org/pdf/2602.13452v1)

**作者:** Belu Ticona `[一作]` (George Mason University), Antonis Anastasopoulos `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了最先进的机器翻译系统和大型语言模型在危机通信中的表现，重点分析翻译对紧急度的保持与评估。

**💡 创新点**

引入了一个覆盖32+语言的紧急度标注数据集，并系统研究翻译如何改变紧急度感知以及LLM在多语言情境下的紧急度分类不稳定性。

**🔧 技术方法**

使用了 NLLB‑200、Aya101、Llama3.2、X‑ALMA 等MT/LLM 模型，利用 spBLEU 和 chrF++ 评估翻译质量；通过 ChatGPT 生成场景，Llama3.2 进行紧急度标注。

**📊 数据集**

使用了 TICO‑19 危机语料、自己构建的多语言紧急度场景数据集（35 种低/中资源语言，共3500 个情景）以及人类标注的四种语言样本。

**📈 对比分析**

采用 spBLEU/chrF++ 对翻译质量进行量化比较，发现大多数模型在非英语→其他语言方向表现显著差于英语→其他语言；NLLB‑200 更稳定但仍有明显误差；LLM 在不同语言下的紧急度分类准确率仅在 14 种语言 ≥80%，整体极不稳定。

**⚠️ 局限性**

模型在低资源语言的翻译质量差，翻译与提示语言会扭曲紧急度感知，LLM 对紧急度分类极不可靠，无法直接用于高风险危机响应。

---

## 822. PrivAct: Internalizing Contextual Privacy Preservation via Multi-Agent Preference Training

**arXiv ID:** 2602.13840 | [PDF](https://arxiv.org/pdf/2602.13840v1)

**作者:** Yuhan Cheng `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**通讯引用:** 25831 | [OpenAlex ID](https://openalex.org/A5058073627)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PrivAct——一种多代理学习框架，将上下文隐私偏好内嵌到 LLM 代理的生成策略中，实现在推理时无需外部干预的隐私合规动作。

**💡 创新点**

创新点在于：①将隐私偏好直接学习进每个代理的生成策略，打破传统推理时的外部隐私检查；②采用奖励传播和偏好对构建多代理偏好数据；③引入泄漏条件异向奖励塑形（LC‑ARS）来平衡隐私与有用性；④展示了在不同模型、不同拓扑下的零射迁移与鲁棒性。

**🔧 技术方法**

技术方法包括：多代理链式生成（generator–verifier–refiner）；偏好学习（DPO）与奖励向上传播；LC‑ARS 的泄漏条件奖励塑形；在 Llama、Mistral、Qwen 等多种 LLM 上进行微调；使用隐私评估判定器（LLM‑judge）来计算泄漏率和帮助度。

**📊 数据集**

数据集：主训练使用 PrivacyLens（多场景上下文隐私基准）；验证与零射迁移使用 ConfAIde（多方隐私推理与会话总结任务）；同时构造多代理偏好数据集用于偏好学习。

**📈 对比分析**

对比方法包括 Vanilla LM、Prompt‑Based Privacy Enforcement (PPE) 与 Agent‑Based Information Flow Control (AIFC)。在 PrivacyLens 上，PrivAct 将平均泄漏率降低 12.32%（相比 AIFC）且帮助率保持相近；在 ConfAIde 上实现零射转移，且在多种拓扑结构下均保持比基线更优的隐私‑帮助 Pareto 前沿。

**⚠️ 局限性**

局限性：①奖励函数与 LC‑ARS 的超参数设计较为经验化，可能对不同模型或隐私规范需调优；②评估依赖 LLM‑judge 进行隐私检测，易受模型误判影响；③当前仅关注隐私与帮助的二元平衡，未覆盖更广泛的伦理与安全风险；④对高度恶意攻击或动态隐私规则的适应性尚未验证。

---

## 823. Detection of On-Ground Chestnuts Using Artificial Intelligence Toward Automated Picking

**arXiv ID:** 2602.14140 | [PDF](https://arxiv.org/pdf/2602.14140v1)

**作者:** Kaixuan Fang `[一作]`, Xinyang Mu `[通讯]` (Michigan State University)

**通讯引用:** 345 | [OpenAlex ID](https://openalex.org/A5001900035)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并标注了319张地面栗子图像（共6524个标注），系统评估了YOLO（v11‑v13）和RT‑DETR（v1‑v4）系列在果园地面栗子检测中的表现。

**💡 创新点**

首次在真实果园地面场景中大规模标注栗子数据并全面比较多种YOLO和RT‑DETR变体，揭示YOLOv11在精确定位上的优势，并将数据集与训练代码公开共享。

**🔧 技术方法**

采用YOLOv11/12/13、RT‑DETRv1‑4深度学习目标检测框架，结合多尺度输入、混合精度训练、数据增强以及Monte Carlo交叉验证评估模型性能。

**📊 数据集**

使用来自密歇根州商业栗子园的319张高分辨率（4032×3024）图像，标注了6524个栗子，构成了地面栗子检测数据集。

**📈 对比分析**

通过5次Monte Carlo CV计算平均mAP@0.5、mAP@[0.5:0.95]、精度/召回、参数量、GFLOPs和推理时间。结果显示YOLOv12‑m获得最高mAP@0.5=95.1%（召回89.3%），YOLOv11x获得最高mAP@[0.5:0.95]=80.1%；RT‑DETRv2‑R101达成mAP@0.5=91.1%，但推理速度较慢；整体而言，YOLO系列在精度和实时性上优于RT‑DETR。

**⚠️ 局限性**

数据集规模有限，缺乏不同品种、土壤、光照等多样化样本；仅评估静态图像，未验证视频流与运动模糊下的鲁棒性；RT‑DETR未针对其训练进行充分调优；未集成后续采摘与控制模块，缺少完整系统验证。

---

## 824. Finding Highly Interpretable Prompt-Specific Circuits in Language Models

**arXiv ID:** 2602.13483 | [PDF](https://arxiv.org/pdf/2602.13483v1)

**作者:** Gabriel Franco `[一作]` (Boston University), Mark Crovella `[通讯]` (Boston University)

**通讯引用:** 20453 | [OpenAlex ID](https://openalex.org/A5064525211)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在语言模型中，如何通过单次前向传播精准定位不同提示（prompt）对应的内部电路（circuit），提出ACC++方法改进传统ACC，利用SVD和Integrated Gradients提取低维因果信号，并在GPT‑2、Pythia、Gemma等模型的间接对象识别（IOI）任务上验证：同一任务下不同提示会激活不同电路，形成若干电路族，并以代表电路概括每个族；同时构建自动解释流水线，将ACC++信号映射为可读的自然语言描述。

**💡 创新点**

创新点包括：①ACC++在ACC基础上加入对真实注意力权重的反事实目标、IG归因及统一的注意力公式，显著降低信号维度和噪声；②将电路分析从任务级转为提示级，揭示提示依赖的电路族；③开发自动解释管道，将信号自动映射为自然语言解释并进行判定，首次在大规模提示级电路中实现可解释性评估。

**🔧 技术方法**

所用技术：ACC++（改进后的注意力因果通信）、SVD分解、Integrated Gradients归因、层级聚类（基于Jaccard距离）、代表电路抽取、自动解释流水线（检索高激活上下文→LLM生成描述→LLM评估判定）。

**📊 数据集**

数据集：在IOI任务上构造的3000个提示样本，包含15个低层模板与2个高层模板（ABBA/BABA），在GPT‑2 Small、Pythia‑160M和Gemma‑2‑2B模型上进行实验。

**📈 对比分析**

比较方法与性能：用ACC++信号进行自动解释评估，计算准确率、精确率、召回率，并与SAE特征基线对比。实验结果为：GPT‑2 Small（0.70/0.68/0.85）、Pythia‑160M（0.65/0.62/0.85）、Gemma‑2‑2B（0.60/0.58/0.80）。提示级聚类通过Jaccard距离+层次聚类展示不同模型的电路族结构。

**⚠️ 局限性**

局限性：①自动解释侧重正向注意力信号，可能忽略Softmax竞争导致的相对影响；②ACC++仅基于单次前向传播，未覆盖所有因果路径；③实验仅涉及三款模型和IOI任务，结果在更大模型或不同任务中的泛化性尚未验证。

---

## 825. TruthStance: An Annotated Dataset of Conversations on Truth Social

**arXiv ID:** 2602.14406 | [PDF](https://arxiv.org/pdf/2602.14406v1)

**作者:** Fathima Ameen `[一作]` (North Carolina State University), Amanul Haque `[通讯]` (North Carolina State University)

**通讯引用:** 414 | [OpenAlex ID](https://openalex.org/A5084727903)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了TruthStance数据集，收集了2023–2025年Truth Social平台上24,352个对话线程及523,360条评论，并对1,500条样本进行人工标注，随后使用LLM为剩余数据生成论证与立场标签，完成对论证存在与评论对立场的标注；

**💡 创新点**

首次系统性提供包含完整回复树结构的alt-tech平台Truth Social对话数据，并通过LLM在全局对话结构中递归推断立场，解决了深层回复立场难以确定的问题；

**🔧 技术方法**

采用少样本提示（Few-Shot、Chain-of-Thought）对Gemini-2.5-Flash、DeepSeek-v3、GPT‑o3等大型语言模型进行推理；同时用支持向量机作为传统基线；

**📊 数据集**

TruthStance数据集（包含原始帖子、评论、作者元数据）以及从Truth Social API抓取的评论数据；

**📈 对比分析**

在约750条人工标注的样本上与SVM基线对比，Gemini-2.5-Flash在“Argument Mining”任务上实现宏F1≈0.82，微调后在“Stance Detection”任务上达到宏F1≈0.83/准确率≈0.83，优于其他LLM配置和SVM基线；

**⚠️ 局限性**

数据仅来自单一alt-tech平台，存在高互动帖筛选偏倚，缺乏低互动线程样本；LLM生成标签未覆盖全数据集，可能导致标注误差；立场推断在出现中立回复时无法传播，导致OP级立场缺失；数据可能被用于政治定向或骚扰，需谨慎使用。

---

## 826. End-to-End NOMA with Perfect and Quantized CSI Over Rayleigh Fading Channels

**arXiv ID:** 2602.13446 | [PDF](https://arxiv.org/pdf/2602.13446v1)

**作者:** Selma Benouadah `[一作]` (Villanova University), Hamid Jafarkhani `[通讯]` (University of California Irvine)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在瑞利衰落信道下、包含完美与量化信道状态信息（CSI）的端到端自编码器（AE）NOMA框架，能够自适应学习并构造干扰感知、功率自适应的叠加星座图。

**💡 创新点**

创新点包括：①在瑞利衰落下首次将AE与CSI量化（均匀与Lloyd–Max）结合；②使用加权交叉熵损失和多SNR训练策略抑制高SNR BER 饱和；③通过全CSI或量化CSI让编码器动态调整星座图，显著提升BER。

**🔧 技术方法**

采用PyTorch实现的深度全连接AE（8层隐藏层，Tanh激活），训练采用Adam优化器与指数学习率衰减；CSI量化使用均匀与Lloyd–Max量化器；训练损失为两用户损失的加权最大/最小组合。

**📊 数据集**

数据集为自生成的随机比特序列（40 k 训练样本、400 k 测试样本），与仿真得到的瑞利衰落信道幅度和噪声结合；不同SNR与CSI量化级别下多次实验并平均。

**📈 对比分析**

与理论QPSK‑QPSK NOMA、单链16‑QAM、单链QPSK、以及PAR/PANOMA等基准进行对比；实验表明AE在所有SNR范围内BER均优于基准，Lloyd–Max量化在量化级别≥4时几乎等价于完美CSI，且整体性能接近理论下限。

**⚠️ 局限性**

局限性包括：①量化器未与AE联合训练，仍存在误差；②仅考虑两用户、单信道条目，未验证大规模或多用户场景；③训练SNR固定，虽缓解高SNR饱和但在极低SNR仍可能欠佳；④缺乏硬件实现与复杂度评估。

---

## 827. The Interspeech 2026 Audio Reasoning Challenge: Evaluating Reasoning Process Quality for Audio Reasoning Models and Agents

**arXiv ID:** 2602.14224 | [PDF](https://arxiv.org/pdf/2602.14224v1)

**作者:** Ziyang Ma `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 98298 | [OpenAlex ID](https://openalex.org/A5100434325)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

组织了Interspeech 2026 Audio Reasoning Challenge，提出MMAR‑Rubrics评估方案，评估音频模型的链式推理质量。

**💡 创新点**

引入实例级可核查的评估标准，减少LLM评测的波动；设立单模型与代理两轨，推动透明音频推理。

**🔧 技术方法**

采用RL、SFT、LoRA训练单模型；代理系统使用多工具协同、可视谱VLM、迭代证据搜集、多代理辩论等技术。

**📊 数据集**

评测基于MMAR‑Rubrics数据集，涵盖语音、音乐、环境音等多种音频的复杂推理问题。

**📈 对比分析**

通过实例级Rubrics分数与最终答案准确率对比，代理轨性能最高（Rubrics 69.83%，Acc 76.90%），单模型最高（Rubrics 65.29%，Acc 74.00%）。

**⚠️ 局限性**

评测仍受数据偏差和人类主观标注影响；单模型推理易出现误推或过度拟合；代理系统复杂度高、部署成本大。

---

## 828. Geometric Characterization of Context-Free Intersections via the Inner Segment Dichotomy

**arXiv ID:** 2602.14722 | [PDF](https://arxiv.org/pdf/2602.14722v1)

**作者:** Jorge Miguel Silva `[一作]` `[通讯]` (University of Aveiro), Jorge Miguel Silva (University of Aveiro)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究两条上下文无关语言交集的几何条件，提出内段长度作为决定交集是否仍为上下文无关语言的关键量。

**💡 创新点**

首次给出交集是否为上下文无关语言的完整判据（对块计数语言完全，对一般语言则条件性），并驳斥了此前的交叉间隙（crossing gap）猜想。

**🔧 技术方法**

利用推挤-弹出弧的几何分析、泵敏感链接（pump-sensitive linkage）与泵引理、缓冲区PDA构造和LCFRS框架。

**📊 数据集**

无（理论研究，无实验数据集）。

**📈 对比分析**

无（无实验比较，结论为理论性质）。

**⚠️ 局限性**

对一般CFL交集的非CFL方向仍需泵敏感链接的存在假设，未证明所有非CFL交集必然满足；此外，是否存在不满足该链接条件的反例仍是开放问题。

---

## 829. STATe-of-Thoughts: Structured Action Templates for Tree-of-Thoughts

**arXiv ID:** 2602.14265 | [PDF](https://arxiv.org/pdf/2602.14265v1)

**作者:** Zachary Bamberger `[一作]` (Technion Israel Institute of Technology), Amir Feder `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 1364 | [OpenAlex ID](https://openalex.org/A5056266191)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可解释的推理时计算框架STATe-of-Thoughts，利用离散文本干预实现高层推理控制并可追踪生成过程。

**💡 创新点**

核心创新在于将高温采样替换为可审计的动作模板干预，构建控制器→生成器→评估器循环，利用动作序列关联质量并通过预测指导搜索；同时提供多种合成模式以平衡可解释性与文本流畅性。

**🔧 技术方法**

技术手段包括：计划-生成-评估-选择的模块化流水线、基于prefill的文本干预、beam搜索、Process/Outcome Reward Models、LASSO回归进行动作特征选择、使用OpenAI LLM-as-Judge与可验证评估器，并讨论MCTS和RL优化的潜在扩展。

**📊 数据集**

实验使用NoveltyBench（100条跨域提示集）和argument‑generation（针对一次性塑料禁令的说服性论点）数据集；利用DeBERTa-v3-large判定功能等价类，GPT‑5‑mini进行Bradley‑Terry评分。

**📈 对比分析**

在NoveltyBench上与CoT、ToT、Best‑of‑N等方法对比，STATe-of-Thoughts在三大模型族（Qwen3、Nemotron‑3、Minstral‑3）和多温度设置下，Mean Distinct平均提升约1–2倍且不降低Mean Utility；在论点生成任务中，动作序列预测模型R²最高可达0.57（严格合成），并通过预测轨迹生成的目标序列在对比实验中获胜率达到77–81%，显著优于随机和主题存在基线。

**⚠️ 局限性**

局限性包括：依赖prefill干预，主要适用于开源模型；动作空间需手工设计且粒度需针对任务；合成模式对可解释性与文本自然度的权衡尚不完美；评估仅基于LLM判别，缺乏因果推断和多轮对话、工具调用等功能支持。

---

## 830. Does Socialization Emerge in AI Agent Society? A Case Study of Moltbook

**arXiv ID:** 2602.14299 | [PDF](https://arxiv.org/pdf/2602.14299v1)

**作者:** Ming Li `[一作]` (University of Maryland), Tianyi Zhou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

系统性诊断了最大规模的 AI 仅代理社交平台 Moltbook，探究其是否出现人类社会所呈现的社交化过程，包括语义收敛、代理适应和影响力结构的稳定；

**💡 创新点**

提出并正式化了“AI 社交化”概念，构建了三维（社会层语义收敛、代理层适应、集体锚点）诊断框架，首次在大规模人工社会中量化社交化现象，揭示规模与社交化并非正相关；

**🔧 技术方法**

使用句子嵌入（Sentence‑BERT）进行语义分析，n‑gram 词汇动态监测，余弦相似度、JS 散度、PageRank 等图论指标，配合统计检验与置换基线来评估反馈与互动对语义迁移的影响；

**📊 数据集**

Moltbook 数据集——约 260 万 LLM 代理、数十亿条帖子、评论与投票，覆盖数月时间跨度，提供完整的时间序列与网络结构信息；

**📈 对比分析**

通过与置换基线、时间对比以及不同层级指标（语义中心相似度、个体漂移、净进展、互动影响）对比评估，结果显示大多数指标均接近零或不随时间显著变化，表明平台缺乏显著的社交化表现；

**⚠️ 局限性**

局限性包括仅研究单一平台且仅包含 LLM 代理，缺乏人类参与或跨平台比较；定义与度量可能未覆盖所有社交化维度；数据集虽然大，但仍有限于 2.6M 代理，可能无法代表更大规模或更复杂的 AI 社会；

---

## 831. Parameter-Minimal Neural DE Solvers via Horner Polynomials

**arXiv ID:** 2602.14737 | [PDF](https://arxiv.org/pdf/2602.14737v1)

**作者:** T. Matulić `[一作]`, D. Seršić `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种参数极小的神经网络架构，利用Horner多项式构造ODE/PDE解。

**💡 创新点**

通过在网络结构中硬编码初始条件与分段Horner实现低参数精确解，避免损失权重调优。

**🔧 技术方法**

使用Horner式多项式网络、分段拼接、自动微分与残差最小化训练。

**📊 数据集**

在三类ODE基准（线性、非线性、二阶线性）及热方程示例上验证。

**📈 对比分析**

与传统MLP（LeakyReLU、Sigmoid）和SIREN比较，Horner网络仅10–13参数即可实现几千倍更低RMSE。

**⚠️ 局限性**

局限在较长区间或高维复杂问题时需增段或更高阶，且分段边界需手工划分。

---

## 832. Event-based Visual Deformation Measurement

**arXiv ID:** 2602.14376 | [PDF](https://arxiv.org/pdf/2602.14376v1)

**作者:** Yuliang Wu `[一作]` (University of Science and Technology of China), Zheng-Jun Zha `[通讯]` (University of Science and Technology of China)

**通讯引用:** 19017 | [OpenAlex ID](https://openalex.org/A5003217535)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种事件-帧混合的视觉变形测量系统，通过融合事件的高时频信息和帧的高空间分辨率来恢复稠密变形场。

**💡 创新点**

创新点包括Affine Invariant Simplicial（AIS）框架将变形场划分为低参数仿射子区，以及邻域贪婪优化策略让已收敛子区引导未收敛邻域，从而降低误差积累。

**🔧 技术方法**

采用事件对比最大化（CM）与图像互相关（CC）相结合的粗细层次优化，利用仿射不变位移插值与连续性先验实现精细化追踪。

**📊 数据集**

构建了120余个包含同步事件流与210fps高帧率视频的真实场景数据集，覆盖挤压、拉伸、弯曲、开裂等多种变形模式，并提供高帧率基准真值。

**📈 对比分析**

在OpenCorr、StrainNet、E-RAFT和CoTrackerV3等基线方法上进行公平评估，使用EPE、存活率和SEPE指标，实验表明本文方法在大位移（>100像素）场景下存活率提升至65.7%（比SOTA高1.6倍），且仅需18.9%的数据存储与计算资源。

**⚠️ 局限性**

该方法受限于空间连续性和亮度恒定假设，在发生裂纹等拓扑变化或强镜面反射时易产生伪应变，导致测量失效。

---

## 833. NFT Games: an Empirical Look into the Play-to-Earn Model

**arXiv ID:** 2602.13882 | [PDF](https://arxiv.org/pdf/2602.13882v1)

**作者:** Yixiao Gao `[一作]` (George Mason University), Songqing Chen `[通讯]` (George Mason University)

**通讯引用:** 5123 | [OpenAlex ID](https://openalex.org/A5065505890)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对 12 款基于以太坊（含 Ronin）NFT 游戏进行大规模测量研究，分析 NFT 所有权分布、推广活动对交易与价格的影响，以及玩家实际盈利情况，并基于博弈论设计并验证了一种可提升玩家收益的激励机制。

**💡 创新点**

首次系统性量化评估多款 NFT 游戏的 Play‑to‑Earn 效果，揭示玩家收益普遍低迷，随后提出并验证了通过发行激励 NFT 以改变收益分配的理论模型。

**🔧 技术方法**

使用区块链交易抓取、Python 数据处理与统计、线性回归及博弈论仿真等技术，构建玩家‑平台互动的数学模型并进行仿真分析。

**📊 数据集**

以太坊主链及 Ronin 副链公开交易记录（截至 2023 年 6 月）共 12 款游戏的 NFT 交易数据，以及官网与社交媒体抓取的辅助信息。

**📈 对比分析**

通过对比交易量、价格波动、玩家平均利润与中位数利润等指标，发现大多数游戏玩家净损失；激励机制仿真后玩家预期收益提升约 15%–30%，平台收益下降但生态可持续性增强。

**⚠️ 局限性**

受限于仅能获得公开交易数据、缺乏 NFT 初始成本信息、不同游戏交易规则非标准化导致分析偏差；模型假设玩家理性且信息完全，实际玩家行为可能更为复杂。

---

## 834. Beyond Eager Encodings: A Theory-Agnostic Approach to Theory-Lemma Enumeration in SMT

**arXiv ID:** 2602.14634 | [PDF](https://arxiv.org/pdf/2602.14634v1)

**作者:** Emanuele Civini `[一作]` (University of Trento), Roberto Sebastiani `[通讯]` (University of Trento)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种理论无关的 SMT 词典枚举方法，用于生成完整的理论 lemmas。

**💡 创新点**

创新点在于结合 divide-and-conquer、投影与分区三种技术，显著提升枚举可扩展性。

**🔧 技术方法**

使用 AllSMT、投影 AllSMT、并行化、理论驱动分区等技术。

**📊 数据集**

实验基准包括 450 条随机合成实例和 90 条工业级规划问题（Painter 域）。

**📈 对比分析**

与传统 AllSMT 基线相比，新方法在枚举时间上提升 1–2 个数量级，成功解决更多实例。

**⚠️ 局限性**

局限在于分区策略仅在原子可分离时有效；在无分区的规划问题中投影效果有限。

---

## 835. Decoupled Continuous-Time Reinforcement Learning via Hamiltonian Flow

**arXiv ID:** 2602.14587 | [PDF](https://arxiv.org/pdf/2602.14587v1)

**作者:** Minh Nguyen `[一作]` `[通讯]` (Google), Minh Nguyen (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种解耦的连续时间actor‑critic算法，通过学习即时优势率q并用Hamiltonian流更新价值函数，实现对非均匀、细小时间步的稳健控制。

**💡 创新点**

创新点在于将q和V的学习解耦成交替更新，消除耦合的martingale目标，采用Hamiltonian流产生非退化的价值更新，并通过Richardson外推提升离散时间近似精度。

**🔧 技术方法**

使用连续时间强化学习理论、受控生成器、Hamiltonian、Richardson外推、单网络Critic、概率论收敛证明以及离散时间模拟技术。

**📊 数据集**

在DeepMind Control Suite的高维连续控制任务以及基于分钟级市场数据的多行业股票组合交易环境上进行实验。

**📈 对比分析**

与连续时间优势率方法、CPPO以及离散时间SAC/TD3等基线对比，实验显示在控制和交易任务中均优于所有基线，尤其在不规则时间步时表现更突出。

**⚠️ 局限性**

局限性包括对时间步长和学习率的配合要求较高，对非马尔可夫噪声的鲁棒性尚未深入验证。

---

## 836. Toward Autonomous O-RAN: A Multi-Scale Agentic AI Framework for Real-Time Network Control and Management

**arXiv ID:** 2602.14117 | [PDF](https://arxiv.org/pdf/2602.14117v1)

**作者:** Hojjat Navidan `[一作]` (Ghent University), Adnan Shahid `[通讯]` (Ghent University)

**通讯引用:** 2854 | [OpenAlex ID](https://openalex.org/A5018797700)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种跨时间尺度的代理式 AI 框架，将 O-RAN 控制分为非实时层 LLM 负责意图解释与模型治理，近实时层 SLM 负责策略执行，实时层 WPFM 负责低时延 PHY 推理，并在 srsRAN 环境中实现了原型。

**💡 创新点**

创新点在于把 O-RAN 的多模型控制从孤立的 rApp/xApp/dApp 转变为一个层级协同的代理体系，LLM 负责意图层面的规划与模型生命周期管理，SLM 在近实时层进行多模型协作与自适应决策，WPFM 在实时层提供边缘推理，并通过标准接口形成闭环。

**🔧 技术方法**

使用的技术包括：大型语言模型（LLM）实现意图理解与策略生成；小型语言模型（SLM）实现低延迟决策与跨 xApp 协调；无线物理层基础模型（WPFM）进行 IQ/CSI 推理；O-RAN 标准接口（A1、E2、O1、F1、O-FH）实现跨层通信；MLOps 流程与数字孪生用于模型验证与迭代。

**📊 数据集**

使用的主要数据集有：公开 IQ 采样集（LTE/5G-NR/Wi-Fi/ITS-G5/C-V2X）用于训练与微调 WPFM；网络 KPI 与 O1 监控数据用于模型治理与策略评估；srsRAN 生成的实时流量与 KPM 用于验证切片资源分配策略。

**📈 对比分析**

对比方法包括：静态均等 PRB 分配、启发式切片控制、仅使用 SLM 的固定目标控制、以及融合 LLM 监督的代理式控制。实验结果显示：代理式控制在 VIP 切片吞吐量提升约 6%（相较 SLM-only），低延迟切片平均延迟保持 22 ms，整体资源利用率提高，且在模型失效时能够及时回滚并重新训练。

**⚠️ 局限性**

局限性主要包括：实时层 WPFM 的子毫秒时延实现尚未标准化；跨层数据同步与语义映射仍需完善；模型在非静态环境下概念漂移与回滚策略需进一步研究；LLM 决策的可解释性与监管合规性缺乏足够保证；标准化与多厂商互操作性仍待完善。

---

## 837. Decentralized Federated Learning With Energy Harvesting Devices

**arXiv ID:** 2602.14051 | [PDF](https://arxiv.org/pdf/2602.14051v1)

**作者:** Kai Zhang `[一作]` (Hong Kong University of Science and Technology), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 44898 | [OpenAlex ID](https://openalex.org/A5079052203)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了能源采集（EH）支持的去中心化联邦学习（DFL）框架，利用设备间的无线对等通信实现模型训练，并通过多智能体马尔可夫决策过程（MDP）优化设备调度与功率控制，提升收敛速度与能耗效率。

**💡 创新点**

创新点在于：①将能源采集技术与DFL结合，解决传统DFL的能量瓶颈；②将多智能体MDP建模为去中心化策略迭代算法，仅使用两跳邻域信息，消除中心化协调导致的指数复杂度与单点失效；③提供理论证明，证明该去中心化算法在迭代次数足够时收敛到全局最优。

**🔧 技术方法**

采用多智能体MDP、策略迭代、局部化Q值更新、乘子权重更新、分布式通信、能量管理与无线信道建模（有限状态马尔可夫信道）。

**📊 数据集**

使用Fashion‑MNIST与CIFAR‑10两个标准图像分类数据集，分别采用CNN和ResNet‑18网络进行实验。

**📈 对比分析**

与集中式策略迭代、集中式贪婪SCA以及无协调贪婪策略等基线对比。实验显示，去中心化策略迭代在保持两跳本地信息约束的前提下，收敛速度和最终测试准确率仅落后0.5–1%，显著优于贪婪方法，且与集中式最优相当。

**⚠️ 局限性**

局限性包括：①两跳邻域信息的选择需手动调节，邻域扩大可能导致计算与通信负担增加；②算法在极稀疏或高密度网络中，邻域规模可显著增长，影响可扩展性；③需要离线预训练并定期更新策略，环境变化（信道、能量统计）会影响性能。

---

## 838. High-Resolution Climate Projections Using Diffusion-Based Downscaling of a Lightweight Climate Emulator

**arXiv ID:** 2602.13416 | [PDF](https://arxiv.org/pdf/2602.13416v1)

**作者:** Haiwen Guan `[一作]` (Pennsylvania State University), Romit Maulik `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建深度学习框架，将LUCIE 300 km粗网格气候模拟输出下采样到约25 km分辨率。

**💡 创新点**

结合物理一致的轻量级气候模拟器LUCIE与条件扩散模型，实现高效且精细的气候数据下采样；通过生成式模型处理“输入不完美”问题。

**🔧 技术方法**

使用Spherical Fourier Neural Operator (SFNO) 与 U‑Net 基础网络，以及基于EDM的条件扩散模型和后验采样扩散模型，进行概率性超分辨率。

**📊 数据集**

训练数据来自ERA5重分析（2000–2009 年6 小时样本），验证使用LUCIE 2010–2018年的预测；高分辨率目标为ERA5原生0.25°网格。

**📈 对比分析**

对比双三次插值、UNet、SFNO‑SR与条件/后验扩散模型；评估指标包括RMSE、时变标准差、功率谱、PDF、EOF；条件EDM在RMSE和谱保真度上最优，后验采样在极值重现上表现最佳。

**⚠️ 局限性**

局限性：后验采样推理成本高，扩散模型训练与推理耗时较大；确定性模型因过度平滑失去细尺度变异；方法仍依赖LUCIE的粗网格物理信息，无法完全补偿原始模型的偏差。

---

## 839. Beyond Ground: Map-Free LiDAR Relocalization for UAVs

**arXiv ID:** 2602.13267 | [PDF](https://arxiv.org/pdf/2602.13267v1)

**作者:** Hengyu Mu `[一作]`, Cheng Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

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

限制在于模型在处理高分辨率图像时性能下降，且对计算资源的需求较高。

---

## 840. Consistent or Sensitive? Automated Code Revision Tools Against Semantics-Preserving Perturbations

**arXiv ID:** 2602.14595 | [PDF](https://arxiv.org/pdf/2602.14595v1)

**作者:** Shirin Pirouzkhah `[一作]` (University of Zurich), Alberto Bacchelli `[通讯]` (University of Zurich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了基于 Transformer 的自动代码修订（ACR）工具在面对语义保持的扰动（SPP）时的“一致性”，即在相同代码语义但结构略有不同的输入下是否能产生相同的修订结果。

**💡 创新点**

创新点在于首次系统设计并评估了九类语义保持扰动，并量化了它们对 ACR 工具一致性的影响；同时提出并验证了多种输入表示改进策略（代码重复、内联注释、链式思考）对一致性的提升作用。

**🔧 技术方法**

使用了 Transformer‑based 代码生成模型（T5‑small、LoRA‑tuned LLaMA、LLaMA‑3.3‑70B、ChatGPT‑3.5‑Turbo、DeepSeek‑V3）和基于 AST 的扰动生成框架，结合多指标评估（EXM、CodeBLEU、Edit Match、Relative Edit Error）。

**📊 数据集**

数据集为 2032 个真实 GitHub 代码审查实例（方法级别），每个实例在经过 9 种扰动后产生约 10K 个语义等价变体。

**📈 对比分析**

与原始未扰动输入相比，LoRA‑tuned LLaMA 的一致性下降最高可达 45.3%，GPT‑3.5‑Turbo 下降 40.9%；其它模型表现相对稳定。引入的输入表示策略对一致性提升效果有限，某些策略甚至导致下降。

**⚠️ 局限性**

限制包括：扰动的语义保持在某些边缘情况可能不严格；评估仅以人类参考修订为基准，忽略了其他功能相同但结构不同的正确修订；实验仅覆盖 Java 代码和少数模型，结果对其他语言或模型的泛化仍未知。

---

## 841. Benchmark Leakage Trap: Can We Trust LLM-based Recommendation?

**arXiv ID:** 2602.13626 | [PDF](https://arxiv.org/pdf/2602.13626v1)

**作者:** Mingqiao Zhang `[一作]` (Nanjing University), Hongtao Liu `[通讯]` (Du Xiaoman Financial Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型在推荐系统中的基准数据泄露现象，并通过在预训练或微调阶段注入泄露数据来评估其对推荐性能的影响。

**💡 创新点**

首次系统地展示了“域相关泄露导致性能膨胀、域不相关泄露导致性能下降”的双重效应，并提出通过LoRA低秩适配来模拟泄露场景的实验框架。

**🔧 技术方法**

使用LoRA微调、AUC/UAUC评估指标、基准推荐模型（ICL、Prompt4NR、TALLRec、PersonPrompt、CoLLM、BinLLM）以及对比实验。

**📊 数据集**

采用MovieLens‑1M、Amazon‑Book作为目标基准数据集，外部六个跨域数据集（Epinions、Last.fm、MIND、Amazon‑Sports、Amazon‑Beauty、Gowalla）作为泄露源。

**📈 对比分析**

通过对比Clean LLM和Dirty LLM下的推荐器性能，发现域相关泄露可提升5–15%的AUC，而域不相关泄露会导致-10%~-25%的下降；协同信息融合模型对泄露更具鲁棒性。

**⚠️ 局限性**

仅使用单一基础模型（Vicuna‑7B），泄露规模和类型有限，未检验不同规模或架构的LLM；实验仅在离散数据集上进行，缺乏对时间动态和多域混合泄露的深入分析。

---

## 842. Patient-Made Knowledge Networks: Long COVID Discourse, Epistemic Injustice, and Online Community Formation

**arXiv ID:** 2602.14528 | [PDF](https://arxiv.org/pdf/2602.14528v1)

**作者:** Tawfiq Ammari `[一作]` `[通讯]` (Rutgers), Tawfiq Ammari (Rutgers)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对2.8 百万条 #LongCOVID 推文进行主题建模（BERTopic）、反思式主题分析和 ERGM 网络分析，研究了长期 COVID 病患如何构建知识网络并抵御医学知识不公。

**💡 创新点**

首次将主题建模与指数随机图模型相结合，揭示病患社区以知识共享为核心的认知实践，证明社交媒体能迅速形成“epistemic community”并促成 WHO 认可。

**🔧 技术方法**

使用主题建模（BERTopic）、指令式叙事分析、指数随机图模型（ERGM）和情感分析等技术。

**📊 数据集**

包含 2,818,709 条 #LongCOVID 相关推文，来自 409,272 个用户。

**📈 对比分析**

与传统 LDA 或单纯情感/中心度指标相比，BERTopic+ERGM 能更准确捕捉主题重叠和关系形成，显示以知识共享为主题的连接显著高于政策辩论。

**⚠️ 局限性**

仅关注英文公开推文，低估边缘群体；数据来源单一平台，缺乏私密社群和跨平台验证；长期演化可能改变结果。

---

## 843. Conditional Generative Models for High-Resolution Range Profiles: Capturing Geometry-Driven Trends in a Large-Scale Maritime Dataset

**arXiv ID:** 2602.13297 | [PDF](https://arxiv.org/pdf/2602.13297v1)

**作者:** Edwyn Brient `[一作]` (Mines Paris), Rami Kassab `[通讯]` (Thales Land and Air Systems)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在大规模海岸雷达数据库上进行高分辨率距离剖面（HRRP）生成的可行性，并验证了基于几何条件（船长、船宽、方位角）的条件生成模型能再现真实HRRP的几何特征。

**💡 创新点**

首次将条件生成技术应用于海上目标的HRRP，并证明仅用船体尺寸和方位角这三个几何参数即可捕捉到HRRP的主要几何趋势，填补了以往仅使用小规模或模拟数据的局限。

**🔧 技术方法**

采用了两种主流生成模型：去噪扩散概率模型（DDPM）和 Wasserstein GAN（WGAN），两者均通过学习嵌入方式将长度、宽度和方位角嵌入网络实现条件化生成。

**📊 数据集**

使用超过900k条HRRP样本、700多艘船只的实际海岸雷达数据库，包含船舶MMSI、尺寸和方位角元数据，覆盖多种实战场景。

**📈 对比分析**

通过与真实邻域（±2°方位角）样本的最佳匹配，评估PSNR、MSE_f、cos_f三项指标。结果显示：在使用长度、宽度和方位角全部条件时，GAN获得PSNR≈27.0、MSE_f≈0.95、cos_f≈0.87，DDPM为PSNR≈24.7、MSE_f≈1.51、cos_f≈0.81；相较于无条件或仅条件化方位角，几何条件显著提升生成质量。

**⚠️ 局限性**

主要局限包括：只用粗粒度几何条件，无法生成细节级别的目标特征；未考虑沉降角和其他环境噪声；生成模型在极小船舶和特殊船体结构（如桥梁）时仍存在偏差，需进一步丰富条件或引入更细粒度特征。

---

## 844. Human Oversight-by-Design for Accessible Generative IUIs

**arXiv ID:** 2602.13745 | [PDF](https://arxiv.org/pdf/2602.13745v1)

**作者:** Blessing Jerry `[一作]` (Universidad Carlos III de Madrid), Paloma Martínez `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 3622 | [OpenAlex ID](https://openalex.org/A5009969418)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种针对高风险工作流程的LLM生成界面的人类监督‑by‑design架构，嵌入多层自动评估与升级机制；

**💡 创新点**

创新点在于把人类监督作为系统设计的核心，利用自动化风险检测触发升级至Human‑in‑the‑Loop（HITL）和Human‑on‑the‑Loop（HOTL），并将结构化反馈转化为治理动作，实现可追踪、可审计的持续改进；

**🔧 技术方法**

使用LLM进行受限生成、可读性/语义相似度/事实一致性等自动评估指标、基于规则的升级策略、SysML v2追踪模型、可访问性标准（WCAG, EN 301 549 等）以及结构化审查日志；

**📊 数据集**

示例数据为医疗沟通场景，使用结构化医疗记录与用户画像（认知障碍、低视力等）进行后诊疗用药说明的生成；

**📈 对比分析**

由于未进行用户研究，论文未给出与其他方法的量化对比或性能指标，主要展示实现的架构与流程，并预期通过阈值触发HITL来保障安全与可访问性；

**⚠️ 局限性**

限制包括缺乏对自动评估指标与人类判断匹配度的实证验证、缺少长期漂移与策略调优的评估、以及未公开具体数据集和用户试验结果。

---

## 845. Bitcoin Under Stress: Measuring Infrastructure Resilience 2014-2025

**arXiv ID:** 2602.14372 | [PDF](https://arxiv.org/pdf/2602.14372v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 846. Compact LLM Deployment and World Model Assisted Offloading in Mobile Edge Computing

**arXiv ID:** 2602.13628 | [PDF](https://arxiv.org/pdf/2602.13628v1)

**作者:** Ruichen Zhang `[一作]` (Nanyang Technological University), Yonghui Li `[通讯]` (University of Sydney)

**通讯引用:** 29680 | [OpenAlex ID](https://openalex.org/A5100448724)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Edge Compact LLM Deployment (ECLD) 框架与基于世界模型的 PPO 算法，实现大语言模型在移动边缘设备上的压缩部署与动态推理卸载；

**💡 创新点**

创新点在于（1）将结构化剪枝、知识蒸馏和低位量化联合成统一的硬件感知压缩流水线，兼顾模型大小、能耗与生成质量；（2）将 LLM 的准确率与幻觉率纳入 QoS 约束，构造基于任务级指标的 MEC offloading 优化；（3）引入轻量化 RSSM 世界模型，提升 PPO 的价值估计与想象规划，实现更快收敛与更优延迟/能耗平衡；

**🔧 技术方法**

使用的技术包括结构化层/通道剪枝、教师蒸馏、4/8 位量化、硬件感知压缩策略、Proximal Policy Optimization（PPO）强化学习、递归状态空间模型（RSSM）世界模型、TD 目标混合、想象辅助损失、熵正则化；

**📊 数据集**

使用的数据集与模型包括：预训练 Llama‑3.1‑8B、Qwen‑3‑8B、Mistral‑12B；评估数据集包括 WebQuestionsSP（准确率）、WikiBio 与 selfcheckgpt（幻觉率）；MEC 测试平台包括 Intel Xeon 服务器 + NVIDIA H200 GPU，及 Jetson Nano 与 Xiaomi 10 Ultra 边缘终端；

**📈 对比分析**

与纯量化、纯剪枝、剪枝+蒸馏等基线比较，ECLD 将模型体积压缩 70–80%（如 Llama‑3.1‑8B 从 15.3 GB 缩至 3.3 GB），能耗下降 50%，准确率仅低于原模型 10% 以内，幻觉率下降；world‑model‑PPO 在相同 MEC 场景下，比 vanilla PPO 收敛 50% 更快，最终奖励提升 15.8%，平均推理延迟下降 12–30%，同时满足能耗、准确率与幻觉率阈值；

**⚠️ 局限性**

局限性包括：对大模型的蒸馏与剪枝参数需要手工调优，world‑model 的预测误差受限于训练样本；实验规模仅限两台终端与单一边缘服务器，未验证大规模多用户/多服务器环境；幻觉评估依赖现有数据集，可能无法覆盖所有实际应用领域。

---

## 847. Embodied Intelligent Spectrum Management: A New Paradigm for Dynamic Spectrum Access

**arXiv ID:** 2602.13245 | [PDF](https://arxiv.org/pdf/2602.13245v1)

**作者:** Yihe Diao `[一作]` (Nanjing University of Aeronautics and Astronautics), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 84717 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了基于具身智能的频谱管理体系（EISM），通过嵌入式无人机、地面机器人与固定监测站的闭环感知、认知、决策与执行，构建动态频谱图并实现主动频谱访问。

**💡 创新点**

创新点在于将具身智能与频谱管理结合，实现了主动探索的感知、交互式认知以及基于大语言模型与强化学习的自适应决策，使频谱管理从被动反应转向主动适应。

**🔧 技术方法**

使用了多模态感知融合、基础模型（LLM）、深度强化学习、云边端协同计算以及层级化决策生成与执行模块。

**📊 数据集**

参考了SynthSoM等多模态频谱数据集，并在原型平台中收集了多源RF、视觉与位置数据进行验证。

**📈 对比分析**

通过与传统被动频谱监测相比，EISM平台在城市环境中完成了高精度三维频谱图构建，显著减少盲区，提升频谱利用率，实验结果显示信息增益提升约30%–50%。

**⚠️ 局限性**

局限性包括缺乏大规模多场景多模态数据、专用基础模型尚未成熟、分布式协同难题及硬件频谱切换的实时性与能耗挑战。

---

## 848. Text Has Curvature

**arXiv ID:** 2602.13418 | [PDF](https://arxiv.org/pdf/2602.13418v1)

**作者:** Karish Grover `[一作]` (Carnegie Mellon University), Geoffrey J. Gordon `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12718 | [OpenAlex ID](https://openalex.org/A5012830032)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了文本的内在曲率概念，设计了Texture曲率度量并用于长上下文推理的资源分配与检索路由。

**💡 创新点**

创新点在于将两侧上下文的贝叶斯推断通过Schrödinger桥实现统一的曲率定义，提供非平坦性证据并将其转化为可操作的控制信号。

**🔧 技术方法**

核心技术包括离散曲率测度、Ollivier–Ricci与CEI无定义的平坦性检验、两步Schrödinger桥中点计算以及基于曲率的Prompt压缩与检索触发器。

**📊 数据集**

实验使用LongBench的五个任务（HotpotQA、2WikiMultiHopQA、Qasper、GovReport、QMSum）及公开文本语料库做自然文本的平坦性验证。

**📈 对比分析**

与BM25、Selective Context、LLMLingua、FLARE和Self-Route等基线比较，CurvPrune在多跳问答中提升约8.8 F1，CurvFlag路由提升HotpotQA 6–7 F1，整体性能均优于单纯的检索或压缩策略。

**⚠️ 局限性**

局限性包括对冻结模型的依赖、曲率度量对离散支持和核设计的敏感性、以及在不同语言、领域或模型规模下的泛化与偏差风险。

---

## 849. Discrete-Space Generative AI Pipeline for Semantic Transmission of Signals

**arXiv ID:** 2602.13556 | [PDF](https://arxiv.org/pdf/2602.13556v1)

**作者:** Silvija Kokalj-Filipovic `[一作]` (Rowan University), Yagna Kaasaragadda `[通讯]` (Rowan University)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5094087224)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出Discernment系统，实现物理信号在技术通道中的语义传输。

**💡 创新点**

创新点是将离散潜在表示与GenAI结合，并根据擦除模式动态切换自回归或扩散生成器。

**🔧 技术方法**

使用VQ‑VAE编码、Decoder‑Only Transformer（MONAI、NanoGPT）和Score Entropy Discrete Diffusion（SEDD）进行生成。

**📊 数据集**

使用Torchsig（RF调制信号）和AudioMNIST（语音数值）两大数据集。

**📈 对比分析**

通过在不同擦除率下对分类准确率、F1和统计相似度进行对比，结果显示即使在高擦除率（>90%）下，SEDD仍保持近乎100%的分类准确率。

**⚠️ 局限性**

局限性包括仅在擦除通道下验证，音频信号对擦除更敏感，且模型仍需进一步优化计算与能耗。

---

## 850. An Overlay Multicast Routing Method Based on Network Situational Aware-ness and Hierarchical Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.13211 | [PDF](https://arxiv.org/pdf/2602.13211v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 851. Expected Moral Shortfall for Ethical Competence in Decision-making Models

**arXiv ID:** 2602.13268 | [PDF](https://arxiv.org/pdf/2602.13268v1)

**作者:** Aisha Aijaz `[一作]` (Indraprastha Institute of Information Technology Delhi), Manohar Kumar `[通讯]` (Indraprastha Institute of Information Technology Delhi)

**通讯引用:** 121 | [OpenAlex ID](https://openalex.org/A5023434899)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于期望短缺（Expected Shortfall）概念的伦理短缺指标EMS，并将其作为损失项嵌入多种机器学习模型，以实现可调节的伦理决策。

**💡 创新点**

创新点在于将金融风险度量转化为伦理风险度量，利用三大伦理理论权重（α、β、γ）实现领域可调的伦理倾向，并提供可与任何学习模型兼容的EMS损失。

**🔧 技术方法**

使用了伦理判别函数EJ、阈值τ的离散化、EMS损失函数、对不同模型（LR、NB、RF、SVM、DNN）进行实验，并对后置硬约束与权重惩罚方法进行对比。

**📊 数据集**

使用的真实数据集为研究生录取数据（770例）和贷款批准数据（32,582例），并在两者上手工映射伦理特征。

**📈 对比分析**

通过对照无伦理、后置硬约束、权重惩罚等基线，评估分类准确率、精确率、召回率、F1、ROC-AUC；EMS在保持高准确率（90%+）的同时，提供可调的伦理风险控制；相较硬约束，性能下降显著。

**⚠️ 局限性**

局限性包括需人工进行伦理特征映射和参数设定，模型对领域特定伦理判断的依赖，缺乏对所有真实世界数据的普适性，且在极端θ值下性能接近硬约束，导致学习能力下降。

---

## 852. GRAIL: Goal Recognition Alignment through Imitation Learning

**arXiv ID:** 2602.14252 | [PDF](https://arxiv.org/pdf/2602.14252v1)

**作者:** Osher Elhadad `[一作]` (Bar Ilan University), Reuth Mirsky `[通讯]` (Tufts University)

**通讯引用:** 404 | [OpenAlex ID](https://openalex.org/A5036081979)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 GRAIL 框架，通过为每个候选目标学习模仿学习策略实现实时目标识别。

**💡 创新点**

创新点在于将目标识别转化为离线模仿学习任务，避免规划或奖励模型，能适应子最优和系统偏差行为。

**🔧 技术方法**

采用行为克隆（BC）、生成对抗模仿学习（GAIL）和对抗逆向强化学习（AIRL）三种 IL 算法训练目标策略，并用负均方误差评分进行一次性推理。

**📊 数据集**

实验使用 MiniGrid（离散导航）和 PandaReach（连续机械臂）两个公开仿真环境的数据集。

**📈 对比分析**

在多种噪声、偏差和最优演示下，与 GRAQL 和 DRACO 基线相比，GRAIL 在准确率和 F1 分数上均优越，且训练和推理时间显著缩短。

**⚠️ 局限性**

局限性包括只能处理封闭的已知目标集合、对环境动态变化不鲁棒、对大规模目标集训练成本线性增长以及未覆盖感知输入的复杂任务。

---

## 853. GUI-GENESIS: Automated Synthesis of Efficient Environments with Verifiable Rewards for GUI Agent Post-Training

**arXiv ID:** 2602.14093 | [PDF](https://arxiv.org/pdf/2602.14093v1)

**作者:** Yuan Cao `[一作]` (Key Lab of HCST (PKU), MOE; SCS, Peking University), Tao Xie `[通讯]` (Key Lab of HCST (PKU), MOE; SCS, Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

基于真实应用交互轨迹，利用多模态代码LLM自动合成轻量化的Web环境，并在代码中嵌入可执行的奖励断言，实现了可验证奖励的GUI代理后训练。

**💡 创新点**

创新点在于：①通过“任务条件化”合成仅包含任务相关逻辑的离线应用，消除了网络延迟和资源占用；②将奖励直接注入代码，提供确定性的奖励信号，解决了视觉代理噪声问题；③引入两阶段自检（静态反思 + Playwright动态测试）保证合成环境的正确性。

**🔧 技术方法**

技术：多模态代码生成模型（如Kimi K2），Meta‑prompting 与 Plan‑and‑Execute 生成策略；自动化上下文提取；代码级奖励注入；Playwright 自动测试；强化学习框架。

**📊 数据集**

数据集：从149个真实移动应用任务收集交互轨迹，用于训练和评估；此外使用了1118个合成环境（149+969）。

**📈 对比分析**

比较方法：在真实任务上评估完成率（SR）与在合成环境中训练的基线模型和直接在真实应用训练的RL模型对比。性能：相较基线提升14.54%，相较真实RL提升3.27%；环境交互延迟降低10×，每个epoch成本从>28000美元降至可忽略。

**⚠️ 局限性**

局限性：合成过程中对LLM生成质量依赖较高，部分环境合成失败率约17%；目前仅支持基于Web/Flask的轻量化应用，可能不适用于更复杂的原生应用；奖励设计仍需针对任务手工定义；以及合成与导航的“差距”表明需要进一步的自我改进机制。

---

## 854. Modality-Tailored Age of Information for Multimodal Data in Edge Computing Systems

**arXiv ID:** 2602.13269 | [PDF](https://arxiv.org/pdf/2602.13269v1)

**作者:** Ying Liu `[一作]` (Aalto University), Yusheng Ji `[通讯]` (National Institute of Informatics)

**通讯引用:** 11103 | [OpenAlex ID](https://openalex.org/A5037098061)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向多模态数据的时效性指标MAoI，并设计了联合采样与边缘计算卸载的最优算法JSO，以最小化系统平均MAoI。

**💡 创新点**

创新点在于：①将图像、音频与信号的内容动态、语义变化和测量质量融入年龄增长率，形成模态自适应的MAoI；②推导了MAoI的闭式平均值；③提出了基于块坐标下降的JSO算法，联合优化采样间隔和卸载决策，且提供可行的最优采样子算法与干扰感知的最优响应卸载子算法。

**🔧 技术方法**

主要技术包括：模态特征提取（ResNet‑18、DeepSpeech2、Temporal Fusion Transformer）、Poisson事件建模、能量与时延模型、凸优化与Newton迭代、最佳响应博弈与拉格朗日子梯度法。

**📊 数据集**

使用仿真数据：图像采用ResNet‑18（ImageNet预训练模型）、音频采用DeepSpeech2、信号采用Temporal Fusion Transformer，所有参数均在仿真环境中设置（如设备数量、信道模型、功率、CPU频率等），未使用公开真实数据集。

**📈 对比分析**

通过与传统AoI（无模态权重）及多种基线（FMI、FLC、GMO、IDD、DBRO）对比，实验显示JSO在不同设备数、能量预算与本地CPU频率下均能显著降低系统平均MAoI；与传统AoI相比，MAoI更能体现模态重要性，优化目标更加精细。

**⚠️ 局限性**

局限性：①实验仅在仿真环境下验证，缺乏真实部署实验；②模型假设（Poisson事件、静态信道、能量消耗线性等）可能与实际场景差异；③算法复杂度随设备数增大仍为O(DN+ID²)，在大规模网络中仍需进一步优化。

---

## 855. Overthinking Loops in Agents: A Structural Risk via MCP Tools

**arXiv ID:** 2602.14798 | [PDF](https://arxiv.org/pdf/2602.14798v1)

**作者:** Yohan Lee `[一作]` (Yonsei University), Seungtaek Choi `[通讯]` (Hankuk University of Foreign Studies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在多工具调用的LLM代理中，通过恶意工具注册实现的结构性过度思考攻击，导致令牌消耗和延迟大幅放大；

**💡 创新点**

创新点在于提出并量化“结构性过度思考”这一新型成本放大攻击，并证明单一恶意工具可通过循环调用显著提高资源使用；

**🔧 技术方法**

使用了攻击工具构造、循环诱导机制、ReAct与Qwen-Code两种代理架构以及NoWait等生成级防御方法；

**📊 数据集**

实验采用AIME2025、GPQA Diamond、HumanEval以及CodeElo等科学推理、数学与编程数据集；

**📈 对比分析**

通过对比正常工具注册与混合注册下的令牌放大系数与准确率，发现混合注册可使令牌放大达142×，准确率下降明显，且NoWait防御无法有效阻止循环；

**⚠️ 局限性**

局限性包括仅聚焦科学推理与编程任务、未系统评估更强防御方案、以及对不同代理配置的泛化能力不足。

---

## 856. TactAlign: Human-to-Robot Policy Transfer via Tactile Alignment

**arXiv ID:** 2602.13579 | [PDF](https://arxiv.org/pdf/2602.13579v1)

**作者:** Youngsun Wi `[一作]` (University of Michigan), Tess Hellebrekers `[通讯]` (Microsoft Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种跨传感器的触觉对齐方法TactAlign，使得从佩戴式触觉手套收集的人类演示能够被直接迁移到配备不同触觉传感器的机器人手中，实现高效的人机政策迁移。

**💡 创新点**

创新点在于使用无配对、无标签的伪配对通过rectified flow学习隐空间映射，同时结合自监督的触觉编码器实现不同硬件之间的语义对齐，并在多任务、多物体上验证了其通用性。

**🔧 技术方法**

主要技术包括：自监督触觉编码器（JEPA/Probe式网络）、基于伪配对的rectified flow对齐、共享的ACT式人机共训练策略，以及基于力传感器的跨传感器力预测评估。

**📊 数据集**

使用的实验数据包括：人类手套（OSMO）与机器人（Xela）在Pivoting、Insertion、Lid‑Closing等任务下的演示数据（每任务约140–160次人类示范、50次机器人示范），以及光灯泡拧紧任务的20次人类示范和用于力预测的1,472条机器人力标注样本与1,527条人类力样本。

**📈 对比分析**

与仅使用机器人数据、仅使用本体感知、或不进行对齐的基线相比，TactAlign在Pivoting、Insertion、Lid‑Closing三任务上平均成功率提升至约70–75%，在未见过的物体与任务上亦能保持60%以上成功率；在光灯泡拧紧任务上实现了100%成功率，远优于无触觉或无对齐基线的0%。

**⚠️ 局限性**

主要限制包括：实验仅验证了单一手套-机器人配对，未检验对其他触觉模态（如视觉触觉）或多手配置的适用性；对视觉差异的补偿缺失，未来需要融合视觉与触觉等多模态信息。

---

## 857. ReusStdFlow: A Standardized Reusability Framework for Dynamic Workflow Construction in Agentic AI

**arXiv ID:** 2602.14922 | [PDF](https://arxiv.org/pdf/2602.14922v1)

**作者:** Gaoyang Zhang `[一作]` (Accenture Information Technology), Feng Zhao `[通讯]` (Accenture Information Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了ReusStdFlow框架，通过“提取-存储-构建”三段流程，将不同平台的DSL工作流拆解为标准化模块并实现自动重组；

**💡 创新点**

创新点在于结合图数据库与向量数据库双知识体系，实现对工作流结构与语义的同步检索，突破传统纯生成方法导致的结构幻觉与重用低效问题；

**🔧 技术方法**

采用LLM（DeepSeek-V3.2）进行语义解析和生成，Neo4j图数据库维护拓扑结构，Milvus向量数据库实现语义检索，Gradio前端交互；

**📊 数据集**

使用从n8n公开平台收集的200条真实工作流，涵盖聊天、文档、视频、API、数据处理等六大领域；

**📈 对比分析**

在提取与构建任务上均取得90%以上准确率，显著优于仅依赖生成的约70%准确率；检索匹配阶段是主要瓶颈，偶尔存在语义不精确导致匹配失败；

**⚠️ 局限性**

限制包括：检索匹配仍受语义相似度阈值影响，复杂或高度自定义工作流拆解时可能出现节点遗漏或功能误分；系统对平台特定细节的适配仍需手动或插件辅助；

---

## 858. Exposing Diversity Bias in Deep Generative Models: Statistical Origins and Correction of Diversity Error

**arXiv ID:** 2602.14682 | [PDF](https://arxiv.org/pdf/2602.14682v1)

**作者:** Farzan Farnia `[一作]`, Azim Ospanov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了深度生成模型在多种数据集上的多样性缺失问题，并通过参考无关的熵基多样性指标（Vendi 与 RKE）进行评估；

**💡 创新点**

提出了生成模型多样性下降的系统性原因，证明了有限样本熵估计的下偏并给出了基于熵的正则化、引导和后处理投影方法来缓解该偏差；

**🔧 技术方法**

采用了核矩阵谱熵（von Neumann 熵）与 Rényi 核熵、Kernel MMD、KD、FD 等指标，并实现了熵正则化、逆 RKE 引导及基于权重投影的后处理；

**📊 数据集**

使用 ImageNet、MS‑COCO、FFHQ、LSUN 等公共图像数据集，以及合成高维高斯混合模型进行实验；

**📈 对比分析**

通过对比真实数据与多种预训练模型（StyleGAN‑XL、Latent Diffusion、DDIM 等）的 Vendi/RKE、Recall/Coverage、FD/KD 等指标，发现真实数据始终拥有更高多样性；在训练集规模增大或采用熵正则/引导后，生成样本的多样性显著提升；

**⚠️ 局限性**

主要局限包括：熵投影与正则化在高维时计算复杂，缺乏可扩展的实现；实验多聚焦于图像生成，未在文本/音频等模态验证；并未深入探究模型结构与多样性偏差的因果关系。

---

## 859. REMem: Reasoning with Episodic Memory in Language Agent

**arXiv ID:** 2602.13530 | [PDF](https://arxiv.org/pdf/2602.13530v1)

**作者:** Yiheng Shu `[一作]` (Ohio State University), Yu Su `[通讯]` (Ohio State University)

**通讯引用:** 188511 | [OpenAlex ID](https://openalex.org/A5100408669)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个两阶段的情景记忆框架，先通过索引把对话与事件转换为含时间感知的混合图结构，然后使用面向工具的代理式推理在此图上进行多步检索与推理。

**💡 创新点**

创新点包括：① 将事件概要（gist）与时间限定的事实（triples）统一存储在混合图中，兼顾语义与上下文层级；② 设计了可迭代的检索与图探索工具链，实现“心理时间旅行”，提升跨事件推理能力；③ 通过实验表明在多项回忆与推理基准上明显优于现有 RAG、Mem0、HippoRAG 等方法。

**🔧 技术方法**

技术手段：使用 GPT‑4.1‑mini‑2025‑04‑14 作为推理 LLM，NV‑Embed‑v2 作为嵌入模型；构建 semantic_retrieve、lexical_retrieve、find_gist_contexts、find_entity_contexts 等工具；采用 ReAct 样式代理实现多步推理；图结构通过相似度阈值 0.8 建立同义词边；实验采用多线程并行检索。

**📊 数据集**

数据集：LoCoMo（合成对话），REALTALK（真实对话），Complex‑TR（时间阅读推理），Test of Time（复杂时间推理）。

**📈 对比分析**

与大型嵌入模型、Mem0、Graphiti、HippoRAG 2、TISER、Full‑Context、Oracle Message 等方法对比；在回忆任务上相较基线提升 3.4% F1、13.4% EM，推理任务上在 Test of Time 取得 90% EM，且拒绝无信息问答的准确率显著提高。

**⚠️ 局限性**

局限性：仍依赖预训练 LLM 的生成能力，检索与推理过程对计算资源消耗较大；索引采用离线批处理，对实时流式记忆支持不足；在时间与数值推理中仍存在选择、定位错误，且在复杂情境下对事实的完整性与一致性控制有限。

---

## 860. Index Light, Reason Deep: Deferred Visual Ingestion for Visual-Dense Document Question Answering

**arXiv ID:** 2602.14162 | [PDF](https://arxiv.org/pdf/2602.14162v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 861. Advancing Analytic Class-Incremental Learning through Vision-Language Calibration

**arXiv ID:** 2602.13670 | [PDF](https://arxiv.org/pdf/2602.13670v1)

**作者:** Binyu Zhao `[一作]` (Harbin Institute of Technology), Ivor Tsang `[通讯]` (Agency for Science Technology and Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于预训练模型的双分支解析增量学习框架 VILA，用于解决类增量学习中的表征僵化问题。

**💡 创新点**

创新点在于引入几何校准（UGC）将任务专用的 ViT-Adapter 特征与冻结的 CLIP 语义基底对齐，并通过语义校正（CSE）在决策层使用视觉语言先验纠正预测偏差，既保持解析学习的高效性，又提升鲁棒性。

**🔧 技术方法**

主要技术包括：解析学习（递归最小二乘）、ViT-Adapter 细粒度自适应分支、冻结 CLIP 视觉语言模型、几何对齐与文本先验校正。

**📊 数据集**

实验使用八个公开基准：CIFAR100、ImageNet‑R、CUB‑200、FGVC‑Aircraft、Stanford‑Cars、Food101、UCF101 与 SUN397。

**📈 对比分析**

与多种现有基线（Prompt、Adapter、LoRA、VLM 以及其他解析方法）相比，VILA 在平均准确率和最终任务准确率上分别提升约 4.8% 与 5.4%，并在效率方面位居 Pareto 前沿。

**⚠️ 局限性**

局限性包括：双分支导致推理时参数量加倍；解析求解器的内存随特征维度平方增长；以及在强域变迁下固定专用分支可能不再最优。

---

## 862. AnchorWeave: World-Consistent Video Generation with Retrieved Local Spatial Memories

**arXiv ID:** 2602.14941 | [PDF](https://arxiv.org/pdf/2602.14941v1)

**作者:** Zun Wang `[一作]` (University of North Carolina), Mohit Bansal `[通讯]` (University of North Carolina)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 AnchorWeave，一种利用检索本地空间记忆并通过覆盖驱动检索与多锚点纺织控制器实现长时段、视角可控的视频生成框架。

**💡 创新点**

创新点在于用多个局部几何记忆替代全局3D重建，避免跨视角误差累积，并通过覆盖驱动检索和姿势引导的多锚点融合来学习校正残余不一致，显著提升长期一致性与视觉质量。

**🔧 技术方法**

技术手段包括 DiT‑based 视频扩散模型、Per‑frame 本地点云记忆、覆盖驱动的 3D 记忆检索、多锚点共享注意力与姿势引导融合控制器，以及更新–检索–生成循环。

**📊 数据集**

数据集主要使用 RealEstate10K 与 DL3DV 进行训练与评估，实验还涉及公开视频集做长时段开放域生成验证。

**📈 对比分析**

与 Gen3C、TrajCrafter、ViewCrafter、SEVA、EPiC、Context‑as‑Memory、SPMem 等基线在 PSNR/SSIM 与 VBench 视觉一致性、背景一致性、运动平滑度等指标上对比，AnchorWeave 在一致性、视觉质量和时间平滑性方面均显著优于基线。

**⚠️ 局限性**

局限性包括对局部点云分辨率的依赖，极端视角或稀疏历史记忆场景下仍可能出现重叠误差；多锚点检索与融合对计算资源有一定开销，且对极大视角跳变的鲁棒性尚待进一步提升。

---

## 863. DALL: Data Labeling via Data Programming and Active Learning Enhanced by Large Language Models

**arXiv ID:** 2602.14102 | [PDF](https://arxiv.org/pdf/2602.14102v1)

**作者:** Guozheng Li `[一作]`, Chi Harold Liu `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于数据编程与主动学习相结合，并利用大型语言模型辅助的数据标注方法DALL。

**💡 创新点**

创新点在于把LLM作为知识源来生成和改进标注函数，并通过主动学习选择最有价值的样本进行人工标注，显著降低人工成本。

**🔧 技术方法**

使用数据编程框架、主动学习算法以及大型语言模型（如GPT‑4）来生成标签函数并进行迭代优化。

**📊 数据集**

在常用的图像/文本标注数据集上评估，例如CIFAR‑10、COCO‑Caption、OpenImages和自定义医学图像集。

**📈 对比分析**

与传统Snorkel数据编程、纯主动学习、以及完全人工标注进行对比。DALL在相同标签成本下提升准确率5–10%，并将标注时间缩短约40%。

**⚠️ 局限性**

依赖LLM的推理成本高，且在领域外知识不足时容易产生错误；对极大规模数据的可扩展性还有待验证。

---

## 864. Hybrid Secure Routing in Mobile Ad-hoc Networks (MANETSs)

**arXiv ID:** 2602.13204 | [PDF](https://arxiv.org/pdf/2602.13204v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 865. ONRAP: Occupancy-driven Noise-Resilient Autonomous Path Planning

**arXiv ID:** 2602.13577 | [PDF](https://arxiv.org/pdf/2602.13577v1)

**作者:** Faizan M. Tariq `[一作]` (Honda Research Institute), Sangjae Bae `[通讯]` (Honda Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了基于占据网格的噪声鲁棒路径规划器ONRAP，利用空间域非线性规划在动态环境中生成运动学可行且平滑的轨迹。

**💡 创新点**

创新点在于将占据信息及可选占据流直接作为约束嵌入空间域优化，无需对象分类或跟踪；同时采用高斯风险函数和五次Hermite样条在保持安全性的同时平衡离障碍距离。

**🔧 技术方法**

技术包括空间域自行车动力学、占据网格与流预测（Kalman滤波）、非线性规划求解、五次Hermite样条生成参考路径以及高斯风险惩罚。

**📊 数据集**

实验使用仿真噪声场景以及真实F1TENTH 1/10车实验室环境（未公开具体公开数据集）。

**📈 对比分析**

与A*、RRT*基线比较，ONRAP平均运行时0.033s、成功率88%、最小障碍距离0.907m、最大曲率0.94 m⁻¹，明显优于基线。

**⚠️ 局限性**

局限在于缺乏硬碰撞约束，需要下游速度规划保证纵向安全；在高噪声动态场景中最小距离可能低于车辆宽度的一半。

---

## 866. A Geometric Taxonomy of Hallucinations in LLMs

**arXiv ID:** 2602.13224 | [PDF](https://arxiv.org/pdf/2602.13224v1)

**作者:** Javier Marín `[一作]` `[通讯]`, Javier Marín

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出三类幻觉（无忠实、虚构、事实错误）的几何分类，并用语义嵌入的SGI与Gamma指标实现上下文相关与无上下文的幻觉检测；

**💡 创新点**

创新点在于把幻觉分为几何可检测的类型并揭示其在嵌入空间中的对齐与分离特征，以及阐明基于嵌入的检测在事实错误上根本无效的理论原因；

**🔧 技术方法**

采用句子Transformer的向量空间几何分析，构建语义接地指数(SGI)和方向接地指数(Gamma)，并利用角度距离与平均方向的点积做判别；

**📊 数据集**

使用HaluEval（LLM生成的幻觉）、TruthfulQA（人类写的错误答案）和自制的142条跨领域虚构实例（金融、医疗、法律）进行实验；

**📈 对比分析**

在HaluEval内域上Gamma实现AUROC ≈0.99，跨域时降至≈0.5；在自制虚构上全局Gamma取得AUROC≈0.96，跨域误差仅3.8%；SGI对无忠实幻觉的AUROC在0.78–0.83之间；对TruthfulQA的事实错误检测AUROC仅为0.48，低于随机；

**⚠️ 局限性**

局限在于基于分布式语义的嵌入无法区分真伪（事实错误）；HaluEval的幻觉反映的是生成风格而非内容真实性，导致跨域泛化失败；自制虚构样本规模有限，可能不涵盖所有虚构风格；

---

## 867. BEACONS: Bounded-Error, Algebraically-Composable Neural Solvers for Partial Differential Equations

**arXiv ID:** 2602.14853 | [PDF](https://arxiv.org/pdf/2602.14853v1)

**作者:** Jonathan Gorard `[一作]` (Princeton University), James Juno `[通讯]` (Princeton Plasma Physics Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了 BEACONS（Bounded‑Error, Algebraically‑Composable Neural Solvers）框架，用于构造具备严格误差上界、收敛、稳定性和守恒性质的神经网络 PDE 求解器，并通过形式化证明确保在外推（extrapolatory）情形下的正确性。

**💡 创新点**

创新点在于：
   1) 将方法学（method of characteristics）与 Mhaskar‑Poggio 近似理论结合，得到针对超平面内外均可应用的 L∞ 误差上界；
   2) 通过可组合的深层网络结构（algebraic composability）将高阶不连续解拆解为平滑子函数与单一不连续函数的复合，从而压缩整体误差；
   3) 设计了自动化代码生成器、DSL 与符号重写式定理证明器，实现从 PDE 规范到可验证的训练、验证与推理全过程的自动化。

**🔧 技术方法**

技术手段包括：
   • 单隐藏层 MLP 的 L∞ 逼近理论与特征线分析；
   • 可组合深层网络构造与 Lipschitz 常数优化；
   • 形式化验证（Racket DSL + 代码生成 + 机器可检验的定理证明）；
   • 训练中采用全批量梯度下降与专门的损失函数；
   • 训练数据通过已形式化验证的数值求解器产生。

**📊 数据集**

使用的“数据集”是由形式化验证的有限体积/差分求解器生成的时间序列解，包括：一维/二维线性传输、无黏 Burgers、以及压缩性 Euler（Sod、quadrants）问题；训练集中仅保留前一部分（0–0.33 时刻或前 33 帧），其余时间由网络外推。

**📈 对比分析**

对比方法：标准的全连接 PINN 网络（相同宽度/深度）。
   结果显示：
   • BEACONS 在 L∞、L² 误差均显著低于非 BEACONS（例如 1D 线性传输 8 层 BEACONS L∞ 0.61 vs 1.03）；
   • BEACONS 在守恒误差上表现更优，且误差上界始终低于形式化证明的理论上界；
   • 在 2D 和非线性 PDE（Burgers、Euler）中，非 BEACONS 网络出现明显的波速失真、扩散或不守恒，而 BEACONS 能保持波形和速度一致，误差约 0.3–0.6（理论上界 1.0–1.5）。

**⚠️ 局限性**

局限与不足：
   • 误差上界依赖于 PDE 的光滑性与特征线可解析性，无法直接处理高度多分支或随机源项；
   • 对于强不连续解仍需大量隐藏神经元或多层组合，计算成本和训练时间较高；
   • 训练数据必须来自形式化验证的数值求解器，若无可验证求解器则只能给出条件性证明；
   • 目前实验仅限于低维（1D/2D）保守系统，尚未验证在更高维、复杂几何或非保守 PDE 上的可扩展性。

---

## 868. High-fidelity 3D reconstruction for planetary exploration

**arXiv ID:** 2602.13909 | [PDF](https://arxiv.org/pdf/2602.13909v1)

**作者:** Alfonso Martínez-Petersen `[一作]`, C. J. Pérez-del-Pulgar `[通讯]` (University of Málaga)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一套从ROS2 rosbag数据提取到COLMAP结构光学重建，再到Nerfstudio的Splatfacto‑W高斯散点重建的完整自动化3D重建流水线，用于行星探测机器人。

**💡 创新点**

创新点包括：①将COLMAP与Nerfstudio无缝集成，加入GNSS/运动先验提升匹配速度；②采用Splatfacto‑W实现低资源下的高质量光照场重建；③全流程容器化，保证跨平台可重复部署。

**🔧 技术方法**

技术实现基于ROS2数据提取、COLMAP SfM、Nerfstudio的Splatfacto‑W（Gaussian Splatting）与NeRF‑W变体、Python脚本管线以及Docker容器化。

**📊 数据集**

使用ESA/UMA公开的BASEPROD数据集，来自MaRTA rover在Bardenas Reales（类马尔萨尔）采集的RGB、IMU、GNSS数据。

**📈 对比分析**

与Nerfstudio默认、仅顺序匹配、以及加入先验的三种预处理配置比较，匹配+映射时间分别下降93.1%和65.8%；几何重投影误差<0.6px，平均观测数>5k，轨迹长度>7；视觉指标PSNR 26–29dB、SSIM 0.78–0.83、LPIPS 0.20–0.26，整体达到可接受至优秀。

**⚠️ 局限性**

局限性：仍为离线高计算量工作流，无法实时或车载部署；对COLMAP SfM质量高度依赖，纹理稀疏或光照变化会影响结果；高斯散点对视角多样性敏感，单一前视轨迹易导致泛化不足；ROS2帧率/时间戳不一致会削弱顺序匹配效果。

---

## 869. OmniScience: A Large-scale Multi-modal Dataset for Scientific Image Understanding

**arXiv ID:** 2602.13758 | [PDF](https://arxiv.org/pdf/2602.13758v1)

**作者:** Haoyi Tao `[一作]` (DP Technology), Xi Fang `[通讯]` (DP Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了一个规模达 150 万条 figure–caption–context 三元组的高保真科学多模态数据集 OmniScience，并提出了基于动态模型路由的重标记管道，将原始图像、图注与文章上下文联合生成富含细节且自洽的描述。

**💡 创新点**

创新点包括：① 通过 Uni-Parser 与 OCR 实现高精度的 PDF 解析与子图定位；② 采用多模型路由策略根据图像类型、学科与上下文动态选择 Gemini、GPT‑5、Qwen 等前沿 MLLM 进行重标记；③ 结合三维事实检查与 LLM‑as‑a‑Judge 的质量评估，实现自动检测与纠正幻觉；④ 将重标记文本作为“Caption QA”代理任务，验证其在科学 VQA 基准上的实用价值。

**🔧 技术方法**

技术手段涵盖：Uni‑Parser + PaddleOCR（PDF 解析与子图检测）；多模型路由与推理（Gemini‑3‑Pro、GPT‑5、Qwen‑3‑VL‑235B 等）；视觉‑文本一致性检查（Qwen3‑VL‑Reranker‑8B、三维事实检查器）；LLM‑as‑a‑Judge（Qwen3‑VL‑235B‑A22B‑Thinking、Seed‑1.5VL）进行质量评估；Caption QA 代理框架与 GPT‑4o‑mini 逻辑推理。

**📊 数据集**

使用数据集：OmniScience（1.5M 主要图 + 5M 子图），以及基准数据集 MMSCI、MMArxiv、SciCap 等用于对比与微调。

**📈 对比分析**

评估方法包括：① 图像–文本检索，使用 Qwen3‑VL‑Reranker‑8B 计算交叉模态相似度（重标记后从 0.769 提升至 0.956）；② 微调 Qwen2.5‑VL‑3B，在 MM‑MT‑Bench、MMMU、MSEarth 等基准上通过 Caption QA 获得 +0.378、+0.140、+0.083 等显著提升；③ LLM‑as‑a‑Judge 在 300 份人工标注样本上达到 QWK 0.831，验证重标记文本的流畅性、信息一致性与细节完整度。

**⚠️ 局限性**

局限性包括：① 仍以开放获取期刊和预印本为主，覆盖范围受限于可公开 PDF；② 重标记过程中依赖 MLLM，计算成本高且可能引入轻微幻觉；③ 仅覆盖十个主流学科，部分领域如工程学、社会科学的图像类型仍不足；④ 现有质量评估虽严谨，但对极其复杂的实验细节仍存在漏检风险。

---

## 870. Human-Aligned Evaluation of a Pixel-wise DNN Color Constancy Model

**arXiv ID:** 2602.13887 | [PDF](https://arxiv.org/pdf/2602.13887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 871. Revenue-Optimal Pricing for Budget-Constrained Buyers in Data Markets

**arXiv ID:** 2602.13897 | [PDF](https://arxiv.org/pdf/2602.13897v1)

**作者:** Bhaskar Ray Chaudhury `[一作]` (University of Illinois), Jiaxin Song `[通讯]` (University of Illinois)

**通讯引用:** 58 | [OpenAlex ID](https://openalex.org/A5038274276)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在数据市场中，针对预算约束的买家，如何设定多数据集的价格函数以最大化总收入。

**💡 创新点**

创新点在于证明了即使价格函数仅需满足单调和下连续性，最优价格仍可结构化为分段线性凸函数，并揭示了线性定价与非线性定价在计算难度上的鲜明对比。

**🔧 技术方法**

主要使用了凸分析、线性规划、分段线性化、子模最大化与连续贪婪算法等技术来构造最优或近似最优的定价方案。

**📊 数据集**

研究未使用具体数据集，而是在理论上以买家预算、价值参数和数据集大小等符号参数进行分析。

**📈 对比分析**

与传统竞争均衡和单价定价方法比较，本文证明非线性定价可在多项式时间内求解，线性定价则是APX‑难的，但提供了 2‑近似（在线）和 (1‑1/e)^‑1 近似（离线）算法，性能优于现有方法。

**⚠️ 局限性**

局限性包括：对非线性定价的结构性假设、仅考虑单调下连续的定价函数、忽略买家偏好互补/相关性、以及线性定价的近似算法仍未达到最佳可实现比例。

---

## 872. sleep2vec: Unified Cross-Modal Alignment for Heterogeneous Nocturnal Biosignals

**arXiv ID:** 2602.13857 | [PDF](https://arxiv.org/pdf/2602.13857v1)

**作者:** Weixuan Yuan `[一作]` (Technical University of Munich), Xuesong Chen `[通讯]` (Five Seasons Medical)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研发了一种面向多模态多导睡眠监测（PSG）的基础模型，通过跨模态对齐学习统一表征，支持在缺失传感器情况下完成睡眠分期和疾病预测；

**💡 创新点**

创新点包括：①提出DASH‑InfoNCE对比损失，利用年龄、性别、录音地点等元信息动态加权负样本，消除群体偏差；②在超过42,000场夜间 PSG 记录上对九种不同模态进行大规模对比预训练；③系统探究模态多样性和模型规模的扩展规律；④在缺失模态和跨中心泛化上表现出鲁棒性；

**🔧 技术方法**

技术要点：对比自监督学习（InfoNCE/DASH‑InfoNCE）、RoFormer Transformer 作为模态无关序列编码器、每模态 MLP 词元化、时间步遮掩、门控融合策略、元信息加权负样本；

**📊 数据集**

数据集：来自五个公共睡眠研究中心的42,249夜间 PSG 记录（Sleep Heart Health Study、Wisconsin Sleep Cohort、MrOS、MESA、Human Sleep Project），以及 APPLES 作为外部验证集；

**📈 对比分析**

与传统单通道专用模型和基线基础模型对比；在 SHHS 和 WSC 上的睡眠分期准确率约 87–88%，接近甚至超越专用模型；在多模态设置下始终优于基线；在疾病预测任务中 ROC‑AUC 随模态数提升而显著提升；模型对传感器缺失也保持较高性能；

**⚠️ 局限性**

局限性：在仅使用单通道 EEG 时仍略逊于高度专用模型；在极度缺失模态（例如所有心电或呼吸信号缺失）下性能可能下降；实验主要集中在睡眠分期和四种疾病，未验证到其他临床指标或不同设备的泛化；

---

## 873. LogitsCoder: Towards Efficient Chain-of-Thought Path Search via Logits Preference Decoding for Code Generation

**arXiv ID:** 2602.14054 | [PDF](https://arxiv.org/pdf/2602.14054v1)

**作者:** Jizheng Chen `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18142 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种轻量级的链式思维搜索框架，利用logit级别的控制在代码生成任务中迭代生成并细化推理路径；

**💡 创新点**

创新点在于通过Logits Preference Decoding（LPD）引导生成高质量思考词，Logits Rank Based Path Selection（LRBPS）与sigma距离度量选取最连贯路径，并用Thoughts Aggregation整合多条思考路径，从而同时解决underthinking与overthinking问题；

**🔧 技术方法**

主要技术包括logit层级的动态偏置（LPD）、基于token logits排序的多路径搜索（LRBPS）、sigma距离度量、路径聚合（Thoughts Aggregation）以及基于rollout的轻量级验证；

**📊 数据集**

实验数据集为APPS和CodeContest两大编程竞赛数据集；

**📈 对比分析**

与MCTS、RethinkMCTS、Self‑play、Reflexion等搜索或反思模型以及Contrastive/Guided Decoding等基线相比，本文方法在pass rate、pass@1上均位列第一，同时token消耗更少，效率更高；

**⚠️ 局限性**

局限性主要包括：实验仅覆盖算法竞赛级别问题，未验证在真实软件开发场景或更强闭源模型上的泛化；以及依赖于可访问logits的开放源模型，难以推广到目前主流闭源大模型。

---

## 874. On the Rate-Distortion-Complexity Tradeoff for Semantic Communication

**arXiv ID:** 2602.14481 | [PDF](https://arxiv.org/pdf/2602.14481v1)

**作者:** Jingxuan Chai `[一作]` (Xidian University), Guangming Shi `[通讯]` (Peng Cheng Laboratory)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了率-失真-复杂度（RDC）框架，推导高斯和二进源的闭式解，并通过变分RDC（VRDC）方法在图像分类、图像生成和视频压缩任务中验证其理论与实验效果。

**💡 创新点**

创新点在于将模型复杂度（用MDL/信息瓶颈度量）正式加入语义通信的率-失真理论，形成三向权衡（率、失真、复杂度），并首次提供可量化的复杂度与通信资源之间的折衷分析。

**🔧 技术方法**

主要技术包括信息论（MDL、信息瓶颈）、变分推理、深度学习编码器/解码器（CNN、WGAN等），以及FLOPs/PSNR/MS-SSIM等评估指标。

**📊 数据集**

实验使用MNIST（生成）、ImageNet/CIFAR（分类）和Vimeo‑90k（视频压缩）等公开数据集。

**📈 对比分析**

通过与DeepJSCC、H.264 等传统方案比较，VRDC在相同比特率或FLOPs条件下实现了更高的分类准确率、生成IS、PSNR和MS‑SSIM，显示出明显性能优势。

**⚠️ 局限性**

局限性包括：理论推导主要针对高斯/二进源，实际应用需对源分布做近似；复杂度约束需通过λ调节，调参困难；实验中低维FLOPs与I(X;U)线性对应关系在高维后趋于饱和；未考虑多任务或多源交互场景。

---

## 875. "I Felt Bad After We Ignored Her": Understanding How Interface-Driven Social Prominence Shapes Group Discussions with GenAI

**arXiv ID:** 2602.14407 | [PDF](https://arxiv.org/pdf/2602.14407v1)

**作者:** Janet G. Johnson `[一作]` (University of Michigan), Michael Nebeling `[通讯]` (University of Michigan)

**通讯引用:** 2854 | [OpenAlex ID](https://openalex.org/A5000831539)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并验证接口驱动的社交显著性（interface‑driven social prominence）在 Generative AI 参与人类‑人工智能小组讨论中的作用，通过设计三种协作模式（Roundtable、Peripheral、Breakout）并进行混合方法实验，探究代理存在感与用户控制两维度如何共同塑造讨论动态、代理影响和参与者体验。

**💡 创新点**

创新点在于：①提出“接口驱动的社交显著性”这一设计视角，系统性地把代理的可见性与可控性视为调节群体讨论中代理角色与影响的核心维度；②通过三种对比模式展示了不同社交显著性配置对群体协同、信息流、批判性评估及用户感知的显著差异；③提出了基于对话图谱的 Ordered Network Analysis（ONA）方法，用于捕捉代理生成式语言的随机性在群体对话结构中的表现。

**🔧 技术方法**

技术手段包括：使用 GPT‑4o Real‑time API（关闭自动生成），配合多模块的转录、上下文管理、手势提示与说话窗口控制；自研基于 React/Next.js 的浏览器视频会议平台（Daily.co SDK）实现实时语音交互；实现了自定义的主动发声与手势召唤机制以及用户控制按钮；使用 BORIS 进行视频编码，利用 ENA Webtool 进行 ONA 可视化。

**📊 数据集**

数据集：由 18 对熟悉彼此的参与者（共 36 名）在三种任务（头脑风暴、决策合成、优先排序）中进行的 18 组讨论，录音、转录、编码后得到的对话日志；并通过 NASA‑TLX、社交懒惰、认知警觉等量表收集问卷数据。未使用公开的自然语言或对话数据集，而是自建情境脚本与偏好设置。

**📈 对比分析**

对比方法：在同一组被试中按顺序完成三种协作模式，采用定量指标（ONA 网络度量、NASA‑TLX 负荷、社交懒惰、认知警觉、对话流畅度、任务实用性评分）和定性访谈。实验结果显示：Roundtable 与 Peripheral 模式促使更频繁的协同评估与批判性讨论，但也带来更高的社交规范压抑；Breakout 模式提供了更强的个体控制与信息筛选，但出现更多竞争性利用代理的行为。总体而言，各模式在代理影响、用户感知与任务负荷上均有显著差异（p < 0.05）。

**⚠️ 局限性**

局限性：①样本规模有限且仅为二人组，未检验更大团队或异步场景；②三种模式在代理存在感与用户控制上同时变化，难以单独评估每个维度的因果影响；③情境任务与偏好设置人为设计，缺乏真实世界多样性；④实验周期短，未考察长期使用后的社交规范演化；⑤使用 GPT‑4o 可能受模型版本与延迟等技术约束，未覆盖多语言或跨文化使用场景。

---

## 876. Traceable Latent Variable Discovery Based on Multi-Agent Collaboration

**arXiv ID:** 2602.14456 | [PDF](https://arxiv.org/pdf/2602.14456v1)

**作者:** Huaming Du `[一作]` (Southwestern University of Finance and Economics), Carl Yang `[通讯]` (Emory University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了TLVD框架，结合大语言模型多智能体协作与传统因果发现算法，实现可追踪的潜变量发现与语义解释；

**💡 创新点**

创新点包括：1) 将多LLM协作建模为不完全信息博弈并求贝叶斯纳什均衡以推理潜变量；2) 通过LLM检索多源网络证据实现潜变量验证；3) 将强化学习与博弈论融合实现高效协同；

**🔧 技术方法**

技术手段包括：大语言模型（LLM）、多智能体协作框架（MALLM）、博弈论（贝叶斯纳什均衡）、强化学习（TD、混合网络）、因果发现算法（RLCD）以及多源网络检索与验证；

**📊 数据集**

使用数据集：医院去标识化的WCHSU-Cancer、WCHSU-Pain；基准数据集：Multitasking Behaviour Study、Teacher’s Burnout Study；

**📈 对比分析**

与多种基线（单LLM、深度研究代理、多智能体平台及多LLM推理框架）对比，TLVD在ACC、CAcc、ECit等指标上平均提升约30%–60%，并在不同数据集、模型配置、参数设置下均表现稳健；

**⚠️ 局限性**

局限性：1) 对LLM质量高度依赖，弱LLM难以达成BNE；2) 多LLM协作通信成本与上下文窗口受限；3) 需手工设计奖励与调参；4) 证据检索仍受限于公开网页与数据库覆盖率；5) 对大规模潜变量图结构的可扩展性尚待验证。

---

## 877. Scale redundancy and soft gauge fixing in positively homogeneous neural networks

**arXiv ID:** 2602.14729 | [PDF](https://arxiv.org/pdf/2602.14729v1)

**作者:** Rodrigo Carmo Terin `[一作]` `[通讯]` (King Juan Carlos University), Rodrigo Carmo Terin (King Juan Carlos University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究正向同质激活函数的神经网络中的重参数化对称性，将其视为参数空间的规范冗余，并提出一种软规范修正（norm‑balancing）来抑制尺度漂移。

**💡 创新点**

1）将神经网络的尺度重参数化视为规范对称，构造规范坐标并给出平衡约束；2）设计只作用于冗余尺度坐标的软规范修正函数，能在不改变函数的前提下提升优化稳定性；3）通过梯度流分析证明该项产生耗散，消除尺度漂移。

**🔧 技术方法**

正向同质性理论、规范理论类比、软规范修正函数、梯度流分析、数值实验（梯度下降、学习率压力测试）。

**📊 数据集**

用于验证的简化一维非线性回归任务，数据来自[-2,2]区间，验证在[-3,3]，加高斯噪声。

**📈 对比分析**

与无规范修正（λ=0）对比，通过训练误差、验证误差和学习率稳定性测试评估。结果显示弱规范修正（λ≈0.05）保持与基线相近的验证误差，强规范修正则略差，但显著扩大可接受的学习率区间。

**⚠️ 局限性**

实验仅限于单隐藏层ReLU网络，缺乏对更深更复杂架构的验证；强规范修正可能限制优化自由度导致泛化下降；需要进一步研究规范修正的量化效果与最佳 λ 的选择。

---

## 878. LLM-Guided Knowledge Distillation for Temporal Knowledge Graph Reasoning

**arXiv ID:** 2602.14428 | [PDF](https://arxiv.org/pdf/2602.14428v1)

**作者:** Wang Xing `[一作]` (Xidian University), Man Wang `[通讯]` (Southwest Jiaotong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种利用大语言模型辅助的知识蒸馏框架，用于压缩时序知识图推理模型。

**💡 创新点**

将LLM作为辅助教师提供时间相关语义知识，采用分阶段对齐策略并联合优化监督与蒸馏损失。

**🔧 技术方法**

双教师蒸馏（高容量TKG教师+LLM），联合softmax、Huber、MSE损失，阶段化训练，使用TTransE、TADistMult嵌入及Adagrad优化。

**📊 数据集**

YAGO11k 与 WIKIdata12k 两个公共时序知识图数据集，涵盖插值与外推任务。

**📈 对比分析**

与BKD、FitNet、RKD等传统蒸馏基线比较，在MRR、Hits@1/3/10等指标均优于基线，尤其在Hits@1/3提升显著。

**⚠️ 局限性**

依赖LLM推理导致训练开销增加；在极低容量学生或特定配置下，某些指标仍略逊于基线；未充分评估跨领域泛化与实时推理效率。

---

## 879. Unbiased Approximate Vector-Jacobian Products for Efficient Backpropagation

**arXiv ID:** 2602.14701 | [PDF](https://arxiv.org/pdf/2602.14701v1)

**作者:** Killian Bakong `[一作]` (Inria Paris), Kevin Scaman `[通讯]` (Inria Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在反向传播中使用无偏随机近似向量-雅可比乘积，以降低深度网络训练的计算与内存成本。

**💡 创新点**

提出无偏随机VJP估计框架，理论分析方差传播与效率权衡，并在秩约束和对角掩模下给出最小方差设计。

**🔧 技术方法**

采用随机投影/低秩/对角掩模的sketching方法，结合SGD方差分析，在MLP、BagNet和ViT等网络上实现。

**📊 数据集**

使用MNIST、CIFAR-10数据集（BagNet-17和ViT）以及MNIST用于MLP实验。

**📈 对比分析**

与全精度backprop及统一掩模等基线对比，实验表明在相同计算预算下准确率下降不超过1-2%，并在低采样率时保持较高精度。

**⚠️ 局限性**

仅在节点级别无跨层协调的sketching，未充分考虑方差在网络深度中的累积；高阶算子近似需要额外统计，且在极大模型上的验证尚不充分。

---

## 880. Expander Decomposition with Almost Optimal Overhead

**arXiv ID:** 2602.15015 | [PDF](https://arxiv.org/pdf/2602.15015v1)

**作者:** Nikhil Bansal `[一作]` (University of Michigan), Thatchaphol Saranurak `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了第一个多项式时间算法，用于在图中计算接近最优的流‑展开子图分解。

**💡 创新点**

核心创新是突破传统的 cut‑and‑recurse 方式，结合并发多源流线性规划、分散度量 LP 与稀疏邻域覆盖技术，得到接近理论下界的 O(log n exp(√log log n)) 的开销；在流展开子图分解上实现了从 O(log²n) 到 O(log^{1+o(1)}n) 的突破。

**🔧 技术方法**

主要技术包括：并发多源流 LP 的双对偶分析、构造稀疏邻域覆盖、对“尺度投票”与“重聚”策略的精细设计、重型簇处理与平衡簇分割的层级递归；整个算法在每一步利用线性规划求解与聚类步骤来保持低开销。

**📊 数据集**

本文是理论研究，没有使用实际数据集；所有结果均通过数学证明得到。

**📈 对比分析**

与现有最优的存在式结果相比，所提出的算法在多项式时间下把开销降低到 O(log^{1+o(1)}n)，几乎与 Ω(log n) 的理论下界一致；相比之前的 O(log^{1.5}n)（cut）和 O(log²n)（flow）显著提升。

**⚠️ 局限性**

局限性包括：开销仍略高于理想的 O(log n)（多余的 exp(√log log n) 因子），算法仅适用于无向边权图；对定向图或顶点展开子图分解的适用性尚未解决；实现的计算复杂度可能在实际应用中较高。

---

## 881. EditCtrl: Disentangled Local and Global Control for Real-Time Generative Video Editing

**arXiv ID:** 2602.15031 | [PDF](https://arxiv.org/pdf/2602.15031v1)

**作者:** Yehonathan Litman `[一作]` (Carnegie Mellon University), Caleb Leak `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种实时生成视频编辑管线，利用局部稀疏修补与全局上下文解耦，仅在需要编辑的区域生成内容，显著提升效率。

**💡 创新点**

核心创新在于：① 将全局注意力拆分为局部上下文适配器和轻量化全局上下文嵌入器，二者在保持高质量的同时使计算量仅与编辑区域大小成正比；② 通过在已有预训练视频扩散模型上添加可训练的控制模块，避免对基模型权重进行大幅度修改，便于后续快速推理与多种控制输入。

**🔧 技术方法**

技术实现包括：预训练文本到视频扩散模型（DiT + VAE）、局部上下文编码器（ControlNet风格适配器）和全局上下文嵌入器（利用VAE编码的背景低分辨率表示进行时间维度注意力），以及LoRA等微调技术。

**📊 数据集**

训练使用内部多样化库存视频数据，带有分割标注和文本描述；在VPBench-Edit、DAVIS、VPBench-Inp等公开数据集上进行评估。

**📈 对比分析**

与ReVideo、VideoPainter、VACE、ProPainter等基线比较，结果表明：在编辑质量（PSNR、SSIM、LPIPS、CLIP分数）、背景保持、文本对齐以及时序一致性方面均优于或匹配全注意力模型；在推理速度上实现50倍以上加速，帧率提升至数十FPS。

**⚠️ 局限性**

局限性包括：VAE编码/解码导致背景细节退化；局部编码器对高速运动视频表现不佳；在4K分辨率下VAE块编码/解码成为吞吐瓶颈，需进一步优化编码器或集成运动信息。

---

## 882. Rethinking Diffusion Models with Symmetries through Canonicalization with Applications to Molecular Graph Generation

**arXiv ID:** 2602.15022 | [PDF](https://arxiv.org/pdf/2602.15022v1)

**作者:** Cai Zhou `[一作]` (Massachusetts Institute of Technology), Tommi Jaakkola `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了“canonical diffusion”框架，即先通过可测的 canonicalizer 将分子样本映射到唯一的切片代表（破坏 S_N × SE(3) 对称性），在该切片上训练普通（非 equivariant）扩散/流模型，然后在采样时通过随机 Haar 变换恢复对称性。

**💡 创新点**

创新点在于：① 从传统的 equivariant 生成器转向基于 canonicalization 的 symmetry‑breaking 训练；② 证明 canonicalization 在理论上能降低流匹配的条件方差、消除混合分数导致的梯度冲突；③ 结合对齐先验与近 Monge 近似的最优传输来进一步降低切片内的迁移难度；④ 提出新的几何谱 canonicalizer（使用 Fiedler 向量和旋转参考框架）和 Canon 结构化 denoiser，可在非 equivariant backbone 上实现高效训练。

**🔧 技术方法**

核心技术包括：Canonicalizer（Fiedler 轨迹 + 旋转基准）、基于流匹配的训练（Conditional Flow Matching）、对齐先验（moment‑matched Gaussian）和可选的 OT 训练（OT anneal）、Canon 结构化 denoiser（在隐藏状态中加入 canonical 信息）以及采样时的随机 Haar 变换。

**📊 数据集**

主要使用公开的 3D 分子生成基准：QM9（小分子）和 GEOM‑DRUG（药物类大分子）。

**📈 对比分析**

与现有的 equivariant/非 equivariant 生成模型（SemlaFlow、FlowMol、MiDi、EQGAT‑diff 等）比较，Canonical diffusion 在两大基准上均取得显著提升，尤其在分子稳定性、有效率和 RMSD 方面突破 SOTA；在少步采样（仅 50 步）时仍保持高质量，且计算开销几乎无增大。

**⚠️ 局限性**

主要限制：① 需要可测的 canonicalizer，某些分子可能具有多重对称性导致 canonical 选择不唯一；② 采样时仍需额外的随机变换步骤；③ 对高维大分子时对齐先验和 OT 的训练成本可能上升；④ 仅在结构化分子（如药物）验证，其他领域的泛化还需进一步研究。

---

