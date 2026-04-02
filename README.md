# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-01 | 今日论文总数: 544

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. See Something, Say Something: Context-Criticality-Aware Mobile Robot Communication for Hazard Mitigations

**arXiv ID:** 2603.28901 | [PDF](https://arxiv.org/pdf/2603.28901v1)

**作者:** Bhavya Oza `[一作]` (New York University), Aliasghar Arab `[通讯]` (City College of New York)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在巡逻移动机器人上实现了基于视觉‑语言模型与大型语言模型的上下文敏感危害评估与通信框架，自动根据危害的严重性、时间敏感度和可缓解性生成不同语气、字符与接收方的报警信息。

**💡 创新点**

创新点在于：①将危害的严重性、时间敏感度、缓解可行性三维上下文因素正式化为评估模块；②使用LLM对这三维因素进行推理并映射至统一的危害级别；③根据危害级别动态调整消息语气、字符与接收方，从而在保持报警效力的同时减少警报疲劳。

**🔧 技术方法**

核心技术包括：BLIP 视觉‑语言模型进行图像理解与场景描述；ChatGPT‑4.0/ Gemini 2.0 Flash 大语言模型进行危害识别、上下文推理和消息生成；ROS‑Melodic 机器人软件架构；NVIDIA Jetson Nano 边缘计算与 Intel RealSense/RGB‑D 摄像头；云端API 调用与实时通信协议。

**📊 数据集**

使用自采集的实验数据集：在厨房与通道共 60+ 次巡逻实验，人工标注危害类型与上下文标签，未使用公开数据集。

**📈 对比分析**

对比方法为固定优先级（基于物体识别直接映射危害级别）的基线。实验结果显示：检测准确率提升 10%（80% vs 70%），用户信任度 82%，平均通信有效性指标 ε=0.81，且所有实验均严格满足报警阈值约束。

**⚠️ 局限性**

局限性：①仅在单危害、受控室内环境下验证，缺乏多危害与拥挤情景的评估；②对云端LLM延迟与版本漂移依赖较大，缺乏可解释性与确定性；③未实现正式的 POMDP 或强化学习决策层，后续需进一步优化策略与安全保障。

---

## 2. The Future of AI is Many, Not One

**arXiv ID:** 2603.29075 | [PDF](https://arxiv.org/pdf/2603.29075v1)

**作者:** Daniel J. Singer `[一作]` (University of Pennsylvania), Luca Garzino Demo `[通讯]` (University of Pennsylvania)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

本文提出从单一大型模型向多元化AI团队转型的理论框架，强调在人工智能发展中引入认知多样性和团队协作的重要性。

**💡 创新点**

创新点在于将复杂系统、组织行为与科学哲学的研究成果应用于AI，提出三维认知多样性（随机性、视角、构成）以及多种团队结构（平坦团队、层级结构、生态系统）来实现集体智能。

**🔧 技术方法**

主要讨论的技术包括混合专家模型（MoE）、提示工程、参数调节（温度、Top‑k、Top‑p）等，用以在同一模型内部或跨模型实现多样化推理与协作。

**📊 数据集**

本文并未使用具体数据集，而是基于文献综述与理论推演。

**📈 对比分析**

由于缺乏实验数据，本文通过对比现有单一模型与多元团队的理论优势来论证其潜在性能提升，主张多元团队能更广泛探索问题空间、避免过早收敛并兼顾准确性与创新性。

**⚠️ 局限性**

局限性包括：缺乏实证验证、实现细节和算法可行性尚未明确、团队协作与通信成本的评估不足、以及多模型协同可能带来的新型错误或偏差。

---

## 3. A Framework for Hybrid Collective Inference in Distributed Sensor Networks

**arXiv ID:** 2603.28778 | [PDF](https://arxiv.org/pdf/2603.28778v1)

**作者:** Andrew Nash `[一作]` (University College Cork), Krishnendu Guha `[通讯]` (University College Cork)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种将云/边缘计算与分布式推理相结合的混合协同推理框架，并为传感器节点设计了动态通信策略，以在保持高分类精度的同时降低通信成本。

**💡 创新点**

创新点包括：① 将云/边缘与分布式推理统一为单一框架；② 通过基于置信阈值的早停、对等节点请求和云上传三种动作的决策策略实现自适应通信；③ 通过解析/近似求解分布式推理中的概率阈值，实现在多节点、不同状态分布下的动态决策。

**🔧 技术方法**

使用的技术主要是概率推理（贝叶斯分类）、早停/边缘计算、基于能耗的成本模型、代理式决策与动态通信、以及数值模拟（Python+SciPy/NumPy）。

**📊 数据集**

实验数据采用合成高斯分布的传感器观测值，分别在 N=2（二分类）和 N=4（四分类）情形下进行，参数包括均值差异 δ_μ、标准差 σ、置信阈值 λ、以及云/边缘与节点间通信成本 C_SE 与 C_SS。

**📈 对比分析**

与两种基线（全局云推理和独立分类）比较：在数据分布差异不大且 C_SE > C_SS 时，混合框架在保持与云基线相近的准确率的同时，通信成本显著低于云基线；当 δ_μ 很大时，框架退化为独立分类；总体上准确率介于两基线之间，成本呈线性增长但比云基线慢。

**⚠️ 局限性**

局限性包括：① 仅针对高斯传感器数据，未验证对复杂模型或真实数据的适用性；② 能耗成本模型过于简化，未考虑延迟、带宽、计算能耗等因素；③ 仅在模拟环境下评估，缺乏物理部署或网络仿真验证；④ 大规模场景下的计算复杂度与通信协调仍需进一步研究。

---

## 4. Calibrated Fusion for Heterogeneous Graph-Vector Retrieval in Multi-Hop QA

**arXiv ID:** 2603.28886 | [PDF](https://arxiv.org/pdf/2603.28886v1)

**作者:** Andre Bacellar `[一作]` `[通讯]`, Andre Bacellar

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多跳问答中图检索与向量检索融合的问题，将两种不同分布的得分校准后再融合，提出了PIT（百分位秩归一化）+Boltzmann加权的融合方法。

**💡 创新点**

核心创新在于将检索得分视为尺度不等的异构信号，先通过百分位秩归一化实现尺度统一，再用Boltzmann分布进行加权融合，证明校准步骤是决定性能的关键，而后续融合算子影响不大。

**🔧 技术方法**

技术包括百分位秩归一化（PIT）、能量-温度转换、Boltzmann加权、Consensus Boost（双系统共识提升）、以及对比RRF、线性融合等传统方法。

**📊 数据集**

使用MuSiQue、2WikiMultiHopQA两个多跳问答基准；对比实验还涉及HotpotQA、Legacy‑Pipeline MuSiQue/2Wiki等。

**📈 对比分析**

在HippoRAG2强基线下，校准融合在MuSiQue @5 从 75.1% 提升到 76.5%（p=0.039），在 2Wiki @5 从 51.7% 提升到 53.6%（p=0.023）。与True RRF相比，校准融合更稳定；R@5 的提升不显著。

**⚠️ 局限性**

局限包括：在全语料规模下图检索候选池爆炸导致性能下降，需要池子截断与同义词链接；对不同数据库或基线配置需额外调参；在强向量检索基准（HotpotQA）下几乎无提升。

---

## 5. Kilohertz-Safe: A Scalable Framework for Constrained Dexterous Retargeting

**arXiv ID:** 2603.29213 | [PDF](https://arxiv.org/pdf/2603.29213v1)

**作者:** Yinxiao Tian `[一作]` (University of Science and Technology of China), Zhen Kan `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为Kilohertz-Safe的高频率抓手遥操作运动重定向框架，将非线性映射问题线性化并转化为凸二次规划，在联合考虑几何、运动学和安全约束的前提下实现实时控制。

**💡 创新点**

创新点在于：①将运动重定向问题在关节微分空间中线性化为凸QP；②将控制屏障函数(CBF)作为硬约束直接融入QP，实现千赫级实时安全保障；③构建统一的多约束框架，兼顾速度、准确性与碰撞安全。

**🔧 技术方法**

核心技术包括关节雅可比线性化、凸二次规划求解、控制屏障函数（CBF）安全约束、实时手势捕捉（MediaPipe/RealSense）以及软硬件闭环实现。

**📊 数据集**

实验基于WuJi多指抓手平台进行，使用单目摄像头或Intel RealSense获取的人手关键点数据作为输入；未使用公开公开的标准数据集，而是自建的实测与仿真数据。

**📈 对比分析**

与Dex‑Retargeting和GeoRT在计算延迟、运动保持率与碰撞安全得分等指标上进行对比；平均延迟9.05 ms，RT@100Hz 85.82%，运动保持和碰撞安全评分均优于两者，证明其在高频实时与安全性方面的显著优势。

**⚠️ 局限性**

局限性包括：仍未将触觉阻尼控制与电机扭矩饱和约束集成；对高质量手势捕捉的依赖可能限制在噪声环境下的鲁棒性；且在极端分布外场景下的安全保证尚待进一步验证。

---

## 6. 1.5 Million Messages Per Second on 3 Machines: Benchmarking and Latency Optimization of Apache Pulsar at Enterprise Scale

**arXiv ID:** 2603.29113 | [PDF](https://arxiv.org/pdf/2603.29113v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 7. SafeClaw-R: Towards Safe and Secure Multi-Agent Personal Assistants

**arXiv ID:** 2603.28807 | [PDF](https://arxiv.org/pdf/2603.28807v1)

**作者:** Haoyu Wang `[一作]` (Singapore Management University), Jun Sun `[通讯]` (Singapore Management University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个在多智能体个人助理中通过系统级执行图约束实现安全的框架，称为SafeSkillHub；

**💡 创新点**

创新点在于将安全校验作为执行节点内置的运行时约束，形成执行图结构不变性；

**🔧 技术方法**

采用了LLM推理、规则引擎与元技能（Safe Skill Factory）相结合的技术；

**📊 数据集**

使用了Google Workspace安全场景集、MaliciousAgentSkills第三方技能库以及一套代码执行基准（B/M系列）做实验；

**📈 对比分析**

与传统正则规则基线对比，SafeSkillHub在Google Workspace任务中达到95.2%准确率、在第三方技能检测中97.8%准确率、在代码执行基准上实现100%检测准确率；

**⚠️ 局限性**

局限包括仍存在少量误报、对高级语义变异攻击的鲁棒性待提升，以及缺乏对正常技能的完整基准评估。

---

## 8. SteelDB: Diagnosing Kernel-Space Bottlenecks in Cloud OLTP Databases

**arXiv ID:** 2603.29052 | [PDF](https://arxiv.org/pdf/2603.29052v1)

**作者:** Mitsumasa Kondo `[一作]` `[通讯]` (NTT, Inc.), Mitsumasa Kondo (NTT, Inc.)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在云 OLTP 环境中，通过将数据库的不同组件（WAL、表、索引）放到多个独立的云块存储卷上，利用 PostgreSQL 的 tablespace 和符号链接实现了零补丁、无代码改动的性能优化方案 SteelDB。

**💡 创新点**

创新点在于将传统认为是网络/存储层瓶颈的性能问题归因到内核层 I/O 机制，提出并验证四个根因（单线程 Flusher、I/O 队列争用、Merge I/O 抑制、单卷 QoS 上限），并通过多卷数据放置与并行 Flusher 的组合来彻底消除这些瓶颈。

**🔧 技术方法**

主要技术包括：使用 PostgreSQL 标准功能（tablespaces、符号链接）进行数据分区；利用 Linux blk‑mq 多队列和 per‑device KFT；通过合并 I/O 的策略恢复 Merge I/O；在云环境中配置多块 gp3/io2 卷以实现 QoS 聚合；采用 TPC‑C 作为基准并用 HammerDB 进行测评。

**📊 数据集**

评测数据集为 TPC‑C WH1,000（约 100 GB 数据量），并在 AWS EBS（gp3、io2）和 Ceph RBD 上进行实验；对比使用单卷、RAID0、分布式卷等多种配置。

**📈 对比分析**

通过与 Amazon Aurora（I/O‑Optimized 存储）和 GCP AlloyDB 进行对比，SteelDB 在相同实例规格下实现了最高 3.1× 的吞吐量提升、58% 的成本降低，整体成本效能提升 7.3×；在 EBS 上更低成本的 gp3 配置下，单机性能提升可达 9×。

**⚠️ 局限性**

局限性包括：仍需在云块存储上部署多卷，增加运维管理；对只读或极低 I/O 频率的工作负载提升有限；评测仅覆盖 OLTP（TPC‑C）场景，其他工作负载如键值、分析或分布式事务的效果尚未验证；性能提升高度依赖内核的 multi‑queue 与 per‑device KFT 支持；在极高并发或极低延迟需求下可能仍受限。

---

## 9. Trojan-Speak: Bypassing Constitutional Classifiers with No Jailbreak Tax via Adversarial Finetuning

**arXiv ID:** 2603.29038 | [PDF](https://arxiv.org/pdf/2603.29038v1)

**作者:** Bilgehan Sel `[一作]` (Anthropic Fellows Program), Jerry Wei `[通讯]` (Anthropic)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为Trojan‑Speak的对抗性微调方法，能够在不显著损失模型通用能力的前提下，绕过Anthropic的Constitutional Classifier，向模型注入隐蔽通信协议并实现危险内容生成。

**💡 创新点**

创新点在于将两阶段课程学习（先教授编码，再进行STEM任务）与混合RL+SFT训练相结合，实现了低“jailbreak tax”的攻击；采用最小化高频字母替换的替换密码和技术模板隐藏编码，从而兼顾高逃逸率与高模型保真度。

**🔧 技术方法**

技术包括：课程学习、GRPO强化学习、监督微调、LoRA适配器、替换密码编码、模板结构化、激活层探测（probe）。

**📊 数据集**

使用的数据集包括：LMSYS‑Chat‑1M（对话），Llama‑Nemotron（STEM），Anthropic Bug Bounty（CBRN专家级查询），以及GPQA‑Diamond、MMLU‑Pro、MATH‑500等通用推理基准。

**📈 对比分析**

实验对比显示，14B+参数模型在GPQA‑Diamond、MMLU‑Pro、MATH‑500等基准上保持≥95%原始能力；在CBRN攻击基准上平均得分85.2%，远高于仅SFT训练的55–65%；相比CMFT等先前方法，能力退化＜5%而攻击成功率≈99%。

**⚠️ 局限性**

局限性包括：评估仅针对特定Constitutional Classifier实现，其他变体可能更稳健；基准测试未覆盖所有实际部署情景；激活探测的鲁棒性尚未充分验证；且未展示攻击在真实硬件或法律环境下的可执行性。

---

## 10. KAN-LSTM: Benchmarking Kolmogorov-Arnold Networks for Cyber Security Threat Detection in IoT Networks

**arXiv ID:** 2603.28985 | [PDF](https://arxiv.org/pdf/2603.28985v1)

**作者:** Mohammed Hassanin `[一作]` `[通讯]` (University of New South Wales Canberra), Mohammed Hassanin (University of New South Wales Canberra)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并评估了使用Kolmogorov‑Arnold网络（KAN）与长短期记忆（LSTM）相结合的混合模型（KAN‑LSTM），用于网络流量的恶意检测；

**💡 创新点**

创新点在于将可学习的单变量spline激活函数引入网络结构，并将KAN与LSTM融合，既捕捉空间局部特征，又捕捉长程时序依赖，显著提升了多类别攻击检测性能；

**🔧 技术方法**

主要技术包括KAN、ConvKAN、CNN、LSTM、MLP以及自定义的BCE损失；

**📊 数据集**

使用了四个公开网络安全数据集：UNSW‑NB15、NSL‑KDD、CICIDS2017，以及由BOT‑IOT、NSL‑KDD、CICIDS2017构成的综合大规模数据集Tri‑IDS；

**📈 对比分析**

在相同训练环境和参数下与CNN、LSTM、MLP、ConvKAN等基线进行对比，指标为准确率、精确率、召回率和F1分数；KAN‑LSTM在所有数据集上均取得最高分，尤其在精确率和F1上相较最优基线提升约1.5%–5%；

**⚠️ 局限性**

局限性包括KAN在训练速度、可扩展性和计算成本方面不如传统网络，需要进一步优化算法与硬件实现。

---

## 11. SemLoc: Structured Grounding of Free-Form LLM Reasoning for Fault Localization

**arXiv ID:** 2603.29109 | [PDF](https://arxiv.org/pdf/2603.29109v1)

**作者:** Zhaorui Yang `[一作]` (UC Riverside), Ashish Kundu `[通讯]` (Cisco Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个利用LLM生成的语义约束、将其转化为可执行的检查点，并通过语义光谱分析与反事实验证实现的故障定位框架。

**💡 创新点**

创新点在于将LLM的自由语义推断转化为闭合的中间表示并绑定到具体程序位置，形成可执行约束；并引入反事实验证以区分因果约束与非因果约束，提升定位准确性。

**🔧 技术方法**

采用Tree‑Sitter+SSA进行程序结构与变量版本化，使用Gemini/Claude等LLM生成语义约束，Pytest+自定义插件收集违规，Ochiai系数计算语义光谱，随后通过LLM生成补丁进行反事实验证。

**📊 数据集**

使用自建的SemFault‑250（250个Python单函数单缺陷）和BugsInPy的部分实例进行评估。

**📈 对比分析**

与SBFL、Delta Debugging、AutoFL等基线对比，Top‑1准确率提升至42.8%、Top‑3提升至68%，平均仅需检查7.6%可执行行，反事实验证进一步提升12%。

**⚠️ 局限性**

对LLM推断质量敏感，难以处理多位置或极为复杂的语义错误；需要先定位目标函数并具备完整可执行测试环境；当LLM误判或约束过于宽泛时会产生噪声。

---

## 12. Beta-Scheduling: Momentum from Critical Damping as a Diagnostic and Correction Tool for Neural Network Training

**arXiv ID:** 2603.28921 | [PDF](https://arxiv.org/pdf/2603.28921v1)

**作者:** Ivan Pasichnyk `[一作]` `[通讯]` (We Label Data Inc.), Ivan Pasichnyk (We Label Data Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种从扫描到定位到诊断再到治疗并验证的诊断流水线，用来定位神经网络错误来源并进行层级手术式修复。

**💡 创新点**

创新点在于将动量的阻尼模型、错误样本梯度归因以及物理推导的动量调度三者整合为一个完整流程，证明该流程在不同优化器间保持一致，且比全量微调节省约82%计算。

**🔧 技术方法**

采用阻尼谐振子模型推导的动量公式 μ(t)=1−2√α(t)，仅在错误样本上计算梯度范数进行层级归因，并用物理动量在定位层上进行有限迭代训练实现修复。

**📊 数据集**

使用 ResNet‑18 训练在 CIFAR‑10 数据集上（单种架构、单种数据集）。

**📈 对比分析**

与常规 SGD（μ=0.9）、1cycle、iKFAD、Adam 等方法对比，物理动量实现 1.9× 更快达到 90% 准确率，层级手术修复在 30 轮内修复 62 个错误，净提升 22%，相较全量重训计算量减少 82%；同时在 Adam 训练模型上保持 100% 的层级重定位一致性。

**⚠️ 局限性**

局限在于仅使用单一种子、单一模型与单一数据集，未验证多种模型/数据集或大模型，手术修复仍会产生新的错误，需要进一步的多种子验证、参数级精细化以及跨优化器的修复超参调优。

---

## 13. Dual Perspectives in Emotion Attribution: A Generator-Interpreter Framework for Cross-Cultural Analysis of Emotion in LLMs

**arXiv ID:** 2603.29077 | [PDF](https://arxiv.org/pdf/2603.29077v1)

**作者:** Aizirek Turdubaeva `[一作]`, Uichin Lee `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

本文未给出具体研究内容，仅包含格式示例和语言展示。

**💡 创新点**

无创新点可总结。

**🔧 技术方法**

未使用任何技术。

**📊 数据集**

未使用任何数据集。

**📈 对比分析**

未进行方法比较，无法评估性能。

**⚠️ 局限性**

限制在于缺乏实验细节和结果，无法验证研究贡献。

---

## 14. APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay

**arXiv ID:** 2603.29093 | [PDF](https://arxiv.org/pdf/2603.29093v1)

**作者:** Pratyay Banerjee `[一作]` (Amazon), Ankit Chadha `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种非参数在线学习框架APEX‑EM，允许LLM代理在任务执行过程中持续累积、检索并重用完整的结构化程序经验；

**💡 创新点**

创新点包括：① 用Procedural Knowledge Graph（PKG）记录完整的执行轨迹和多维质量评估；② 设计了Plan‑Retrieve‑Generate‑Iterate‑Ingest（PRGII）五阶段工作流，提供多维验证器；③ 采用双结果（正负）经验索引和语义＋结构签名混合检索，实现跨域结构迁移；

**🔧 技术方法**

技术实现涵盖：LLM（Claude Sonnet/Opus）+ 结构化经验表示；PRGII工作流+任务验证器；PKG图结构+结构签名（LCS匹配）+向量检索+图遍历；双结果记忆索引+Teacher模型评估；迭代自纠正循环；

**📊 数据集**

使用的数据集包括：BigCodeBench（代码生成）、KGQAGen‑10k（结构化查询生成）和Humanity's Last Exam（多域知识推理）；

**📈 对比分析**

与MemRL及其基线在相同冻结模型条件下比较，使用Last Epoch Success Rate（SR）和Cumulative Success Rate（CSR）指标；结果显示：KGQAGen‑10k 89.6% SR / 95.3% CSR（超过oracle上限）；BigCodeBench 83.3% SR / 84.0% CSR（比MemRL +11pp），HLE 48% SR / 53.3% CSR（+22.8pp）；Ablation验证各组件贡献；

**⚠️ 局限性**

局限性：仅在10个epoch评估，模型骨干不同导致直接数值比较受限；未在Lifelong Agent Bench与ALFWorld上验证；依赖Judge LLM，评估噪声可能影响；结构签名为手工设计，缺乏自动化发现；无正式收敛理论；需要严格的数据隐私与访问控制。

---

## 15. On the limited utility of parallel data for learning shared multilingual representations

**arXiv ID:** 2603.29026 | [PDF](https://arxiv.org/pdf/2603.29026v1)

**作者:** Julius Leino `[一作]` (University of Helsinki), Jörg Tiedemann `[通讯]` (University of Helsinki)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在大规模多语种预训练语料中对比不同比例（0%、1%、2%、5%）的平行文本对Transformer 1.4B参数模型的跨语言表示共享效果。

**💡 创新点**

创新点在于系统性地使用多种评估手段（PCA投影、余弦相似度、PWCCA、语言特定神经元分布以及语言控制向量）来衡量平行数据对跨语言对齐的影响，并发现其对最终对齐影响有限，只在预训练早期加速共享并减少语言特定神经元。

**🔧 技术方法**

采用的技术包括：OLMo解码器式Transformer架构、SentencePiece BPE分词、对平行数据应用多种指令/完成格式、以及上述多维度评估方法。

**📊 数据集**

使用的数据集为英芬两种语言的FineWeb/FineWeb2单语料，以及来自OPUS（HPLT、CCMatrix、OpenSubtitles）的平行语料，构成200B token多语种语料库。

**📈 对比分析**

通过可视化、相似度统计、PWCCA、神经元特异性分析以及控制向量实验对不同模型进行对比；结果显示，无论平行数据比例如何，最终模型的跨语言对齐水平基本相同，只有在训练早期和神经元特异性上出现细微差别。

**⚠️ 局限性**

主要局限包括仅使用单一语言对（英芬）、模型规模仅为1.4B参数、平行数据比例限制在5%以内，且未检验更大模型或更高比例平行数据的影响。

---

## 16. Mimosa Framework: Toward Evolving Multi-Agent Systems for Scientific Research

**arXiv ID:** 2603.28986 | [PDF](https://arxiv.org/pdf/2603.28986v1)

**作者:** Martin Legrand `[一作]` (Université Côte d'Azur), Louis-Félix Nothias `[通讯]` (Université Côte d'Azur)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了Mimosa框架，实现可演化的多智能体工作流，用于自动化执行各类科研计算任务。

**💡 创新点**

创新点在于将工作流拓扑视为可演化搜索空间，并结合Model Context Protocol实现动态工具发现与LLM-as-Judge提供的迭代反馈，实现真正的自适应工作流优化。

**🔧 技术方法**

技术栈包括MCP（模型上下文协议）、LangGraph（工作流定义与序列化）、SmolAgent（代码生成式执行）、LLM-as-Judge（四维评价指标）以及单点迭代搜索。

**📊 数据集**

使用ScienceAgentBench（102个跨学科科研任务）作为评测数据集。

**📈 对比分析**

在三种执行模式（单体代理、一次性多代理、迭代学习）下对比，DeepSeek‑V3.2在迭代学习模式下达到43.1%成功率，显著优于单体代理（38.2%）和一次性多代理（32.4%），且与现有最佳基线相当。

**⚠️ 局限性**

局限主要包括评判器信噪比不足、单点搜索在第8–10次迭代后趋于饱和、单体代理基线受环境配置影响偏低、缺乏多代种群探索以及跨模型评判一致性待验证。

---

## 17. Xuanwu: Evolving General Multimodal Models into an Industrial-Grade Foundation for Content Ecosystems

**arXiv ID:** 2603.29211 | [PDF](https://arxiv.org/pdf/2603.29211v1)

**作者:** Zhiqian Zhang `[一作]` (Hello Group Inc), Jun Gao `[通讯]` (Hello Group Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并部署了一款工业级多模态基础模型 Xuanwu VL-2B，目标是提升内容生态中的内容理解、内容审核与对抗 OCR 识别性能。

**💡 创新点**

创新点包括：① 3 阶段训练管线（pre‑training → mid‑training → post‑training）与数据迭代机制；② 动态高分辨率感知（动态分块 + 全局缩略图 + Pixel Unshuffle）；③ 以 GRPO 为核心的强化学习对抗对齐；④ 基于 Chain‑of‑Thought 的可解释审核流程；⑤ 兼顾业务定制与通用能力的复合策略。

**🔧 技术方法**

技术方案：InternViT‑300M 视觉编码器 + Qwen3‑1.7B 语言模型 + 2 层 MLP 对齐桥；使用 DeepSpeed、FlashAttention‑2、AMP/bf16；SFT、GRPO RL；动态拼裁与像素 unshuffle；多任务损失与复合奖励设计。

**📊 数据集**

数据集：1.3M 高质量 caption 对齐数据；17.33M 图文对齐数据；2.8M 业务级数据（业务保留、指令增强、内容审核、SPAM 对抗样本）；8M SFT 训练集（含跨模型校验）；GRPO 对抗 OCR 样本；以及公开基准（OpenCompass、HallusionBench、AI2D、MMStar、OCRBench 等）用于评估。

**📈 对比分析**

性能对比：在 7 项 OpenCompass 多模态指标上平均 67.90（比 InternVL‑3.5‑2B 的 64.27 提升 3.6 分）；内容审核平均召回 94.38%（比 InternVL‑3.5‑2B 的 88.39 提升 6.0 分）；对抗 OCR 加权召回 82.82%（比 InternVL‑3.5‑2B 的 64.79、Gemini‑2.5‑Pro 的 76.72 提升 18.03 分）。

**⚠️ 局限性**

局限性：① 对极端水印或隐藏文字的检出率仍受 448×448 视场与像素 unshuffle 分辨率限制；② 逻辑跳跃可能导致错误归因；③ 对抗样本的多样性仍有限；④ 推理成本虽低于大模型但仍高于极轻量化方案。

---

## 18. Byzantine-Robust and Communication-Efficient Distributed Training: Compressive and Cyclic Gradient Coding

**arXiv ID:** 2603.28780 | [PDF](https://arxiv.org/pdf/2603.28780v1)

**作者:** Chengxi Li `[一作]` (KTH Royal Institute of Technology), Ming Xiao `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于循环梯度编码的分布式训练方法（LAD）及其压缩通信版本（Com-LAD），旨在在存在拜占庭攻击和通信瓶颈的情况下提升鲁棒性和通信效率。

**💡 创新点**

创新点：①使用循环梯度编码在设备间生成冗余梯度，显著降低诚实设备梯度的离散度，提升鲁棒聚合的区分度；②在编码后对梯度进行无偏压缩，兼顾通信压缩与鲁棒性；③提供完整的收敛分析并给出误差上界，证明在大多数设备诚实时误差可消失；④将多种鲁棒聚合规则（如 CWTM、NNM）作为元算法灵活嵌入。

**🔧 技术方法**

技术手段：循环梯度编码、梯度编码与解码、无偏压缩（随机稀疏化/随机量化）、鲁棒聚合规则（CWTM、NNM、几何中值等）以及理论分析中的 κ‑robustness、L‑smooth、数据异质性 β‑bound 等。

**📊 数据集**

数据集：使用 100 维线性回归的合成数据，包含 100 个样本，每个子集单独一个样本；通过正态分布和可调异质性参数 σ_H 生成不同程度的子集异质性。

**📈 对比分析**

比较方法：非压缩场景下对比 Vanilla Averaging、CWTM、CWTM+NNM、DRACO；压缩场景下对比 Com‑VA、Com‑CWTM、Com‑CWTM+NNM、Com‑TGN。实验结果显示：LAD 在适度冗余（d ≥ 20）时可逼近 DRACO 的性能，同时计算开销仅为其一半；Com‑LAD 在压缩下同样优于压缩基线，尤其在加入 NNM 前置聚合后提升明显。

**⚠️ 局限性**

局限性：需要在每台设备上进行额外的梯度编码和冗余计算，导致计算负担升高；压缩方案虽降低通信量，但仍需在设备端执行无偏压缩；实验仅在合成线性回归数据上验证，未在大型真实任务或多任务分布式训练中测试；理论分析假设 κ‑robust聚合规则满足一定条件，实际鲁棒性取决于具体实现。

---

## 19. CREST: Constraint-Release Execution for Multi-Robot Warehouse Shelf Rearrangement

**arXiv ID:** 2603.28803 | [PDF](https://arxiv.org/pdf/2603.28803v1)

**作者:** Jiaqi Tan `[一作]` (Simon Fraser University), Hang Ma `[通讯]` (Simon Fraser University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种名为 CREST 的执行框架，用来改进自动化仓库中多机器人货架重新排列的执行效率，通过主动释放轨迹约束减少机器人空闲和货架切换次数。

**💡 创新点**

其创新点在于利用全局执行信息实现在线约束释放，并结合单轨迹重规划、依赖切换和群体轨迹重规划三种轻量化策略，既保持了 MAPF‑DECOMP 的可扩展性，又显著提升了执行质量。

**🔧 技术方法**

技术实现基于 MAPF‑DECOMP 的两阶段分解，生成安全 1‑robust MAPF 轨迹，构造依赖 DAG，使用 MLSIPP 进行路径规划，配合 Hungarian 匹配进行任务分配，并实现 STR、DS、GTR 等增益策略。

**📊 数据集**

实验使用三类代表性布局（Random‑to‑Random、Staging‑to‑Warehouse、Distributed‑and‑Exchange），分别生成 Medium 与 Large 尺寸实例，每类生成 25 个实例，总计 150 个。

**📈 对比分析**

与 MAPF‑DECOMP（PP）基线对比，采用标准化的总成本、最小化时间和货架切换次数指标；CREST 在所有设置下分别提升约 16–40% 的成本、9–34% 的最小化时间和 6–44% 的切换次数，且在加入升降/放置开销后仍保持显著优势。

**⚠️ 局限性**

目前仅在离线场景验证，尚未处理动态任务到达；对极大规模实例的 MLSIPP 规划仍存在一定时间开销，且改进效果受初始 1‑robust MAPF 计划质量影响。

---

## 20. Efficient Camera Pose Augmentation for View Generalization in Robotic Policy Learning

**arXiv ID:** 2603.29192 | [PDF](https://arxiv.org/pdf/2603.29192v1)

**作者:** Sen Wang `[一作]` (Xi'an Jiaotong University), Sanping Zhou `[通讯]` (Xi'an Jiaotong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个前向式3D高斯渲染框架 GenSplat，利用排列等价 Transformer 对稀疏、无标定的多视角输入进行一次性重建，并通过合成几何一致的新视角来扩充观察空间，从而显著提升机器人视觉动作学习的视角泛化性能。

**💡 创新点**

主要创新点包括：① 采用排列等价 Transformer 架构实现对无序、无标定输入的高效前向重建；② 通过 3D 先验蒸馏自预训练的三维基模型为重建提供几何正则化，避免纯光度优化导致的几何崩塌；③ 在不需要额外收集的前提下，以高效的方式生成多样化的合成视角，实现可扩展的数据增广。

**🔧 技术方法**

使用技术包括：3D 高斯 Splatting、排列等价 Transformer、DINOv2 特征提取、光度损失 + 3D 点图损失 + 相机位姿约束的复合训练目标、FlashAttention、bfloat16 训练、梯度检查点等。

**📊 数据集**

在真实机器人平台上收集的六个操纵任务数据集（Franka Research 3 + Robotiq 2F‑85）共约 100 条专家演示；预训练阶段使用 271k 张野外场景图像；对比实验中还使用公开的 DROID 等数据。

**📈 对比分析**

与多种 NVS 方法（ZeroNVS、VISTA、SEVA、InstantSplat、NoPoSplat、AnySplat）以及两类策略网络（π_0、Diffusion Policy）进行对比；在大视角扰动下，GenSplat 使 π_0 的成功率提升 13.3%（从 48.9% 提升至 62.2%），Diffusion Policy 提升 56%（从 27.78% 提升至 43.33%）；渲染速度为 0.14 s/帧，显著快于扩散采样；在视觉质量和几何一致性上均优于传统高斯重建。

**⚠️ 局限性**

局限性包括：对预训练 3D 先验模型的依赖，在极端光照或动态场景下性能可能受限；增广效果在视角密度饱和后提升有限；仅使用 RGB，无法充分处理深度缺失或遮挡严重的复杂环境；对动态物体的建模仍未覆盖。

---

## 21. A Classification of Heterogeneity in Uncrewed Vehicle Swarms and the Effects of Its Inclusion on Overall Swarm Resilience

**arXiv ID:** 2603.28831 | [PDF](https://arxiv.org/pdf/2603.28831v1)

**作者:** Abhishek Joshi `[一作]` (Texas A&M University Corpus Christi), F. Antonio Medrano `[通讯]` (Texas A&M University Corpus Christi)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一套基于行为特征、硬件结构和作战空间的三维异质性分类体系，并系统评估了将异质性引入无人机/无人车队中的可行性与对集群弹性的提升作用。

**💡 创新点**

创新点在于：①将异质性拆分为“性质（行为/功能）”“结构（硬件/感知）”“作业空间”三维维度，形成统一标签体系；②构建弹性度量指标，并通过多场景对比（模拟、竞赛、实地实验）验证异质性对任务完成时间、覆盖率、故障容忍度等指标的正向影响；③提出跨域、跨模态协同的技术路线，强调从硬件到行为层面的协同设计。

**🔧 技术方法**

采用的技术包括：多代理任务分配模型（约束组合遗传算法、GNN-基于奖惩学习）、异步强化学习与自适应控制、跨域 SLAM（LiDAR+视觉+声纳融合）、多层通信拓扑与中继/网关设计、能源感知规划与移动充电、以及基于模拟与竞赛的性能评估框架。

**📊 数据集**

使用的数据集与测试环境主要来自公开竞赛与仿真平台：DARPA Subterranean 竞赛地下探测数据、农业作业多模态影像与土壤传感数据、AirSim/ROS仿真生成的多平台任务数据，以及实验室/海上/城市环境的实地测试数据集。

**📈 对比分析**

比较方法：将异质队列与同质队列在相同任务（搜索覆盖、目标检测、地图构建、通信连通）下的性能指标（任务完成时间、覆盖率、检测率、能耗、连通度、弹性指数）进行定量对比。实验结果显示，异质队列平均可提升20‑40％的任务完成效率，覆盖率提升30‑50%，在失效、GPS失效或通信干扰场景下的任务成功率提高约30‑60%。

**⚠️ 局限性**

局限性：①异质性引入导致系统设计与验证复杂度显著升高，缺乏统一的评价基准和标准化度量；②仿真到实地的迁移仍面临硬件差异、时间同步与环境噪声；③对能源与通信耦合的优化还不够成熟；④安全与伦理风险（如自主决策失效）尚未得到充分阐述。

---

## 22. Hybrid Quantum-Classical AI for Industrial Defect Classification in Welding Images

**arXiv ID:** 2603.28995 | [PDF](https://arxiv.org/pdf/2603.28995v1)

**作者:** Akshaya Srinivasan `[一作]` (Fraunhofer Institute for Industrial Mathematics ITWM), Ali Moghiseh `[通讯]` (Fraunhofer Institute for Industrial Mathematics ITWM)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文结合卷积神经网络提取铝TIG焊接图像特征，并将其输入两种量子模型（基于VQLS的QSVM和VQC分类器）以及经典CNN，对比三者在二分类和三分类焊接缺陷识别任务中的表现。

**💡 创新点**

创新点在于将量子线性求解器VQLS与传统支持向量机结合，形成可在NISQ设备上实现的量子支持向量机；以及将卷积特征经过线性投影后直接映射到4量子比特，构建浅层可变量子电路实现高效分类。

**🔧 技术方法**

采用的技术包括：卷积神经网络特征提取、量子特征映射与量子核构造、Variational Quantum Linear Solver (VQLS) 与支持向量机求解、变分量子电路 (VQC) 训练与参数梯度估计、Ray Tune进行PBT超参数搜索，以及CUDA GPU上的量子模拟器。

**📊 数据集**

使用的是Kaggle上公开的TIG焊接缺陷图像数据集，选取三类（良好焊缝、污染、缺乏熔合）共1100张图像（700训练/验证、400测试）。

**📈 对比分析**

通过在相同的特征维度（最佳63维）下对比，经典CNN在所有分类任务上保持100%准确率；VQC分类器在二分类中取得99.7%准确率、三分类中98.9%；VQLS-enhanced QSVM在二分类中最高达96.8%、三分类中92.4%，训练时间更长、对量子硬件噪声更敏感。

**⚠️ 局限性**

主要局限包括：VQLS求解器在核矩阵构造和成本函数优化上计算量大、易受噪声影响；VQC模型对量子比特数和电路深度的限制导致无法处理更高维特征；整体实验依赖量子模拟器，缺乏真实量子硬件验证，且对数据集的特定视觉特征高度依赖，泛化性待进一步评估。

---

## 23. Multi-Layered Memory Architectures for LLM Agents: An Experimental Evaluation of Long-Term Context Retention

**arXiv ID:** 2603.29194 | [PDF](https://arxiv.org/pdf/2603.29194v1)

**作者:** Sunil Tiwari `[一作]` (Fulloop), Payal Fofadiya `[通讯]` (Fulloop)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多层记忆框架，将对话历史分为工作、情节和语义三层，并通过自适应检索门控与保持正则化实现长期上下文保持。

**💡 创新点**

创新点在于将记忆层次化为工作/情节/语义三层，并引入自适应层权重检索与语义保持正则化，兼顾长时记忆保留与上下文压缩。

**🔧 技术方法**

采用了工作记忆窗口编码、情节摘要递归融合、语义图抽象、自适应层加权检索以及保留正则化等技术。

**📊 数据集**

在LOCOMO、LOCCO和LoCoMo三个长时对话基准上进行评估。

**📈 对比分析**

与现有多层记忆与压缩方法相比，在LOCOMO上成功率提升至46.85，F1达0.618；在LOCCO上六期保持率提升至56.90%，误记率下降至5.1%，并实现58.4%上下文利用率。

**⚠️ 局限性**

局限在于模型仍需要调参以平衡层权重与保留强度，且在极大规模对话或多模态场景下的可扩展性尚未验证。

---

## 24. An Empirical Recipe for Universal Phone Recognition

**arXiv ID:** 2603.29042 | [PDF](https://arxiv.org/pdf/2603.29042v1)

**作者:** Shikhar Bharadwaj `[一作]` (Carnegie Mellon University), David R. Mortensen `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于大型多语种预训练模型PhoneticXEUS的语音识别系统，能在多语言和带口音的英语上实现最先进的性能；

**💡 创新点**

创新点在于系统性结合大规模多语种数据、Xeus的多语种SSL预训练以及Self‑Conditioned CTC训练目标，并通过细粒度消融实验验证各组成的贡献；

**🔧 技术方法**

使用了Xeus（HuBERT‑style）多语种SSL编码器、Self‑Conditioned CTC损失、基于G2P的多语种语料与自监督预训练技术；

**📊 数据集**

主要数据集为IPAPack++（约17k小时多语种语音，含G2P生成的音标标签）和PRiSM评测集合，以及包含多种口音的英语测试集；

**📈 对比分析**

在PRiSM和口音英语基准上与现有模型对比，PhoneticXEUS在多语种评测中PFER降至17.7%，在带口音英语中降至10.6%，相较于最强对手提升约1–2% PFER；

**⚠️ 局限性**

局限性包括：依赖G2P生成标签导致对口音或非标准发音的鲁棒性有限；在某些发音特征（如紧绷度、延迟释放）上改进空间仍大；低资源语言的评估受训练数据覆盖度和标注噪声影响。

---

## 25. CrossTrace: A Cross-Domain Dataset of Grounded Scientific Reasoning Traces for Hypothesis Generation

**arXiv ID:** 2603.28924 | [PDF](https://arxiv.org/pdf/2603.28924v1)

**作者:** Andrew Bouras `[一作]` (Nova Southeastern University), OMS-II Research Fellow `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了跨域的、包含 1,389 条结构化且有文本依据的科学推理轨迹数据集 CrossTrace，并使用其对 Qwen2.5-7B‑Instruct 进行微调

**💡 创新点**

首次在多领域（生物医学、AI/ML、跨域）中提供逐步推理链与原文引用的训练数据，展示了推理结构本身可跨学科迁移

**🔧 技术方法**

基于 QLoRA 的微调、Axolotl 框架、LLM 评判（GPT‑4o、Claude Opus 4.5）、结构化 Input/Trace/Output 方案以及比对度量（IAScore、余弦相似度、结构合规率）

**📊 数据集**

CrossTrace（自 medRxiv、bioRxiv、arXiv 预印本）以及对比基准 HypoGen 数据集

**📈 对比分析**

通过对比基线、仅 CrossTrace 训练、以及混合跨域训练，结果显示 IAScore 由 0.828 提升至 0.968（GPT‑4o），结构合规率从 0% 提升至 100%，spark 余弦相似度提升 2.8 倍；混合训练在跨域测试上与单域模型几乎持平，表明推理模式可迁移

**⚠️ 局限性**

评估数据集和模型受限于单一提取模型（Claude Sonnet 4）、单位验证者、样本量有限、英文预印本偏向、以及推理链仅代表已发表论证而非真实发现过程

---

## 26. GUARD-SLM: Token Activation-Based Defense Against Jailbreak Attacks for Small Language Models

**arXiv ID:** 2603.28817 | [PDF](https://arxiv.org/pdf/2603.28817v1)

**作者:** Md Jueal Mia `[一作]`, M. Hadi Amini `[通讯]` (Florida International University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了 9 类 jailbreak 攻击在 7 个小型语言模型（SLM）和 3 个大型语言模型（LLM）上的效果，并提出了基于内部激活的轻量级防御框架 GUARD‑SLM。

**💡 创新点**

创新点在于：① 通过层级分析发现 jailbreak 信号在模型的每一层都可观测；② 仅利用一次前向传播即可提取最后 token 的激活向量，训练简单的 RBF‑SVM 进行恶意提示检测；③ 防御不需要额外 token、额外前向传播或重训练，适合边缘设备。

**🔧 技术方法**

核心技术包括 transformer 的隐藏层激活提取、层级可分性分析、t‑SNE 可视化、RBF‑SVM 分类器以及在推理过程中对激活做标准化与判别。

**📊 数据集**

实验使用了 AdvBench、JailBreakV‑28K、Alpaca、HarmBench、Dolly 等公开数据集，并在 LLaMA‑2‑7B、Vicuna‑7B、Mistral‑7B 等模型上进行评测。

**📈 对比分析**

与 SelfEval、Self‑Reminder、RobustAlign、SmoothLLM、ICD、Aligner、GoalPrior 等现有防御方法对比，GUARD‑SLM 在所有 9 种攻击中的成功率均降至近 0%，且仅增加 0.43 s 的平均推理延迟，显著优于其他方法。

**⚠️ 局限性**

局限性：防御方案主要针对 SLM，因大模型的激活提取成本高，难以直接迁移到 LLM；实验结果依赖特定的评判模型（GPT‑4o / GPT‑4o‑mini）和超参数，可能随环境变化略有偏差。

---

## 27. \texttt{ReproMIA}: A Comprehensive Analysis of Model Reprogramming for Proactive Membership Inference Attacks

**arXiv ID:** 2603.28942 | [PDF](https://arxiv.org/pdf/2603.28942v1)

**作者:** Chihan Huang `[一作]` (HKUST), Shuai Wang `[通讯]` (HKUST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于模型重编程（model reprogramming）的主动成员推断攻击框架，用以放大深度学习模型中隐含的隐私泄露信号。

**💡 创新点**

创新点在于：①将模型重编程作为主动探测手段，主动诱导模型在成员样本上产生更显著的行为差异；②统一框架可无缝适配LLM、扩散模型与分类模型；③通过理论（损失曲率、梯度流、信息论）解释重编程提升隐私信号的根本原因。

**🔧 技术方法**

技术手段包括：冻结目标模型，学习输入空间的轻量化变换（如软提示、噪声扰动、初始状态注入）；优化目标为最大化成员与非成员预测分数差异；对LLM采用基于长尾token的校准分数与硬样本挖掘；对扩散模型利用确定性逆向过程与轨迹偏差损失。

**📊 数据集**

数据集涵盖十余个基准：LLM领域的 WikiMIA、MIMIR；扩散模型的 CIFAR‑10、Tiny‑ImageNet、CIFAR‑100、LAION‑5B、COCO；分类模型如ImageNet、CIFAR 等。

**📈 对比分析**

与七类SOTA基线（Loss、Ref、Zlib、Neighbor、Min‑K%及其改进、ReCaLL）进行对比。实验显示在低 FPR 条件下，框架在LLM平均提升 5.25% AUC、10.68% TPR@1%FPR，扩散模型提升 3.70% AUC、12.40% TPR@1%FPR，整体保持最优性能并保持查询效率。

**⚠️ 局限性**

局限性包括：①需要具备可标记的 shadow 数据集，数据分布不匹配会降低迁移效果；②依赖完整的 logits 或 loss 输出，若仅有硬标签会显著退化；③对更强自适应防御（如对抗重编程的检测或更严密的 DP）尚未系统评估。

---

## 28. HCLSM: Hierarchical Causal Latent State Machines for Object-Centric World Modeling

**arXiv ID:** 2603.29090 | [PDF](https://arxiv.org/pdf/2603.29090v1)

**作者:** Jaber Jaber `[一作]` (RightNow AI), Osama Jaber `[通讯]` (RightNow AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出了一种融合对象分解、层次时间动态和因果结构的世界模型架构

**💡 创新点**

通过两阶段训练（先空间重建后动态预测）实现对象专化，并在层次层面结合选择性状态空间模型、稀疏Transformer和压缩Transformer来捕捉连续事件与抽象目标

**🔧 技术方法**

使用Slot Attention与Spatial Broadcast Decoder、Selective SSM、Sparse Transformer、Goal Transformer、GNN因果图以及自研的Triton SSM核等技术

**📊 数据集**

在Open X-Embodiment的PushT机器人操控数据集上进行训练和评估

**📈 对比分析**

与不使用空间分解的对照模型对比，虽预测误差略高（0.008 vs 0.002 MSE），但实现了空间分解、事件检测与显著的SSM速度提升（38×）

**⚠️ 局限性**

存在槽位数量过多导致对象未完全聚合、因果图学习不稳定、模型规模扩展受限、以及训练对随机种子高度敏感

---

## 29. LatentPilot: Scene-Aware Vision-and-Language Navigation by Dreaming Ahead with Latent Visual Reasoning

**arXiv ID:** 2603.29165 | [PDF](https://arxiv.org/pdf/2603.29165v1)

**作者:** Haihong Hao `[一作]` (University of Science and Technology of China), Xiaojun Chang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了LatentPilot，一个端到端的VLM导航器，利用训练时的未来观察作为privileged supervision，将未来观测的影响内化为Pilot Token，从而在推理时实现“梦境”式前瞻。

**💡 创新点**

创新点在于：①使用轨迹中的未来帧做训练时的特权监督，②采用飞轮式闭环训练（PilotLoop）持续收集与微调，③将Pilot Token嵌入LLM背骨实现连续隐空间的前瞻推理，避免额外的world‑model或规划模块，保持推理严格因果。

**🔧 技术方法**

技术手段包括：SigLIP视觉编码器、LLaVA‑Video 7B LLM backbone、轻量级Pilot模块（线性投影）、未来观测的mean‑pooling目标、expert takeover、flywheel闭环训练、动作交叉熵与Pilot重构损失的联合优化。

**📊 数据集**

训练数据：Matterport3D（R2R、RxR、EnvDrop‑augmented R2R）和HM3D上的ScaleVLN合成轨迹；评测数据：R2R‑CE、RxR‑CE、R2R‑PE（DualVLN + Isaac Lab）以及真实机器人（AgileX LiMO Pro、Unitree Go2）中的室内导航任务。

**📈 对比分析**

与Seq2Seq/CMA、ETPNav、ScaleVLN、NaVid等基线对比，单RGB下在R2R‑CE与RxR‑CE取得SOTA，且在R2R‑PE（Val‑Unseen）上明显领先；在真实机器人上能完成多步指令。与外部world‑model方案相比，LatentPilot在推理时延和内存占用更低，同时保持或提升导航性能。

**⚠️ 局限性**

局限性：仍依赖单RGB感知，对极端动态或高复杂度布局的泛化尚未充分验证；飞轮训练需要多轮数据收集，训练成本较高；隐空间可能在极端情况下出现退化或信息不足，需进一步探索稳定性与鲁棒性。

---

## 30. Fisheye3R: Adapting Unified 3D Feed-Forward Foundation Models to Fisheye Lenses

**arXiv ID:** 2603.28896 | [PDF](https://arxiv.org/pdf/2603.28896v1)

**作者:** Ruxiao Duan `[一作]` (Yale University), Yunwen Zhou `[通讯]` (Google XR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本工作提出Fisheye3R框架，利用可学习的校准Token在Transformer层中对统一的3D前馈基础模型进行自适应，从而使其能够在鱼眼图像上保持高精度的相机姿势、深度、点图及视场估计，且对传统透视图像性能无负面影响。

**💡 创新点**

创新点在于通过在Transformer层插入校准Token并采用掩码注意力机制，使模型在不改变原有权重的前提下实现鱼眼与透视图像的兼容性，显著降低对鱼眼训练数据的依赖，同时保持极低的额外计算开销。

**🔧 技术方法**

采用的技术包括基于DINOv2的特征提取、Transformer的帧内与全局注意力模块、校准Token插入、掩码注意力控制、Kannala‑Brandt鱼眼仿真、以及三种学习方案（自监督、带标签透视监督、带标签鱼眼监督）。

**📊 数据集**

训练使用6个大规模透视数据集（ScanNet++、MegaDepth、BlendedMVS、TartanAir、MVS‑Synth、ParallelDomain‑4D）并可选加入2个鱼眼数据集（ASE、KITTI360），测试数据来自ScanNet++（鱼眼版）、ADT与KITTI360（鱼眼版）。

**📈 对比分析**

通过在三种基线模型（VGGT、π^3、MapAnything）上进行对比实验，使用相机姿势、深度、点图和视场共15项指标进行评估，Fisheye3R在所有鱼眼数据集上平均提升约50%+，在多数指标上显著优于基线，甚至在自监督场景下也能取得显著改进。

**⚠️ 局限性**

局限性包括仅适用于Transformer架构，对原模型性能依赖较大；自监督模式下效果受限于原模型在透视图像上的表现；缺乏鱼眼标注数据时无法充分发挥监督优势，且未验证在非Transformer或更复杂场景下的可迁移性。

---

## 31. Stable Walking for Bipedal Locomotion under Foot-Slip via Virtual Nonholonomic Constraints

**arXiv ID:** 2603.29050 | [PDF](https://arxiv.org/pdf/2603.29050v1)

**作者:** Leonardo Colombo `[一作]` (Centre for Automation and Robotics), Anthony Bloch `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种基于虚拟非齐次约束的双足行走控制框架，能够在脚底滑移的情况下实现稳定步态。

**💡 创新点**

创新点在于将脚底滑移视为可调节的速度约束（虚拟非齐次约束），并将其与传统的虚拟齐次约束（VHC）协同工作，得到兼容滑移的混合零动力学（HZD）和闭环稳定性分析。

**🔧 技术方法**

采用了混合动力学建模、虚拟齐次与非齐次约束设计、输入-输出线性化控制、贝塞尔多项式步态参数化、Poincaré映射的稳定性分析。

**📊 数据集**

主要使用了结构化的二维7自由度仿真模型（无真实数据集），并在该模型上实现滑移序列仿真，验证控制效果。

**📈 对比分析**

通过与不激活滑移控制的开放循环对比，展示了在多步（50步）滑移变化场景下的步态稳定性提升；在仿真中，滑移输出误差保持在千分之一级别，步速保持稳健，失败步数从29步提升至50步，表明控制方法显著提升了鲁棒性。

**⚠️ 局限性**

局限性包括：仅在二维仿真模型上验证；未考虑真实摩擦模型与接触耦合；对滑移律的预设假设可能不适用于更复杂或不规则地形；实验验证尚未完成。

---

## 32. Biomimetic PINNs for Cell-Induced Phase Transitions: UQ-R3 Sampling with Causal Gating

**arXiv ID:** 2603.29184 | [PDF](https://arxiv.org/pdf/2603.29184v1)

**作者:** Anci Lin `[一作]` (Shandong University), Wenju Zhao `[通讯]` (Shandong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于物理信息的神经网络（Bio‑PINN）来求解细胞诱导的相变问题，并通过时间‑空间因果门控与不确定性驱动的采样策略实现了高精度的微结构恢复。

**💡 创新点**

创新点包括：① 引入近至远的空间门控（causal distance gating）将学习过程从细胞附近逐步扩展到整个域；② 设计基于变形不确定性的 UQ‑R3 采样机制，兼顾信息保留、重采样与释放；③ 给出理论保证，证明门控与采样的累积与覆盖性质。

**🔧 技术方法**

技术手段：物理信息神经网络、深度 Ritz 能量最小化、低差异度采样（Hammersley 点）、梯度自动微分、门控权重化损失、软边界惩罚与高阶梯度正则化。

**📊 数据集**

使用自生成的二维穿孔域（单细胞、双细胞、三细胞）作为实验数据集，所有实验均在同一模拟框架下完成。

**📈 对比分析**

与 vanilla PINN、RAR‑D 与残差驱动 R3 进行对比，实验表明 Bio‑PINN 在单细胞、双细胞及三细胞设置中均能更精准地重现尖锐过渡层和连结缝隙，并显著减少角向伪影和数值发散，整体性能优于现有自适应和无门控方法。

**⚠️ 局限性**

局限性包括：依赖于手工调节的门控与 UQ 超参数；目前仅在二维模拟中验证，缺乏三维扩展和大规模并行化研究；对极端正则化强度仍可能出现收敛缓慢；需要进一步探索更高效的 UQ 近似与多物理耦合场景。

---

## 33. LA-Sign: Looped Transformers with Geometry-aware Alignment for Skeleton-based Sign Language Recognition

**arXiv ID:** 2603.29057 | [PDF](https://arxiv.org/pdf/2603.29057v1)

**作者:** Muxin Pu `[一作]` (Monash University), Chen Change Loy `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于循环 Transformer 的骨骼信息循环增强框架 LA‑Sign，用于孤立词语级手语识别；

**💡 创新点**

核心创新包括：①利用循环结构在共享参数下实现深层递归推理，避免堆叠多层导致参数膨胀；②引入几何感知对齐（GA Alignment），将骨骼与文本特征投影到自适应 Poincaré 球面上，通过超曲率空间的对比损失实现多尺度语义组织；③系统性对比了三种循环设计（encoder‑decoder、encoder‑focused、decoder‑focused）和几何流形，证明 encoder‑decoder 循环与自适应 Poincaré 最优。

**🔧 技术方法**

技术实现包括：部件级 ST‑GCN 编码骨骼，循环 Transformer 结构，超曲率（Poincaré / Lorentz）投影与 Riemannian 优化，超曲率对比损失以及 LM 损失的联合训练。

**📊 数据集**

使用公开的手语数据集 WLASL（2000/300）和 MSASL（1000/200）进行评测。

**📈 对比分析**

与多种基准方法（ST‑GCN、I3D、StepNet、NLA‑SLR、CCL‑SLR 等）对比，LA‑Sign 在 WLASL2000/300、MSASL1000/200 上分别取得 64.73/64.62% 的 P‑I、62.41/62.41% 的 P‑C，均突破前沿 SOTA。

**⚠️ 局限性**

局限性主要包括：①对高质量骨骼标注依赖强，易受关键点检测误差影响；②超曲率投影和 Riemannian 训练相对复杂，调参较多；③目前仅针对骨骼信息，缺乏 RGB/光流等多模态融合，可能在光照或背景多变场景下表现受限。

---

## 34. Near-Optimal Encodings of Cardinality Constraints

**arXiv ID:** 2603.28954 | [PDF](https://arxiv.org/pdf/2603.28954v1)

**作者:** Andrew Krapivin `[一作]` (Carnegie Mellon University), Bernardo Subercaseaux `[通讯]` (Carnegie Mellon University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了多种新的 CNF 编码方案，用于高效地表示 Cardinality 约束，尤其是“至多一” (AMO) 与“至多 k” (AMK) 约束，显著降低了所需子句数量；

**💡 创新点**

创新点在于利用图论视角构造多部图编码、引入基于哈希的网格压缩以及“判定切换”技术，打破了此前对 AMO 子句上界的猜想，并给出 AMK 的新上界；

**🔧 技术方法**

技术上结合了完整多部图、单调电路、网格压缩（受哈希表启发）以及判定切换（disjunctive switching）等方法，实现了宽子句与传播完整性的平衡；

**📊 数据集**

实验使用了 PySAT 公开的 SAT 基准，特别是随机选择 3 个互斥子集且加入全局 AMK 约束的 UNSAT 生成器，未使用外部专业数据集；

**📈 对比分析**

与现有编码（如 sequential counter、GP、DGP 等）在子句数和求解时间上进行对比，实验显示 DGC 在大规模实例中求解速度最快，虽然子句数与最优编码相差不大；

**⚠️ 局限性**

局限性包括 AMK 编码不具备传播完整性；部分方案仅在 k 较小或满足特定约束时有效；在极大 k 或非均匀实例中表现尚未完全验证。

---

## 35. Enhanced Channel Estimation for Flexible Intelligent Metasurface-Aided Communication Systems

**arXiv ID:** 2603.29098 | [PDF](https://arxiv.org/pdf/2603.29098v1)

**作者:** Jinyue Jiang `[一作]` (University of Electronic Science and Technology of China), Zhu Han `[通讯]` (University of Houston)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用柔性智能金属表面（FIM）提升单用户上行通道估计和下行信噪比；

**💡 创新点**

创新点在于通过对FIM表面形状进行优化，最小化测量矩阵的列相干度，从而显著提升OMP压缩感知估计精度，并利用估计的方向角和路径增益进一步优化下行MRT beamformer以获得更高SNR；

**🔧 技术方法**

采用OMP压缩感知算法进行稀疏恢复，使用Davidon‑Fletcher‑Powell（DFP）迭代优化算法对FIM表面形状进行自适应调整，结合最大比率传输（MRT）Beamforming；

**📊 数据集**

实验基于仿真数据，设置30 GHz、50 MHz带宽、N=25个元件、L条多径、100次Monte‑Carlo平均、10个训练时隙、形变范围b=λ等参数；

**📈 对比分析**

与传统刚性均匀平面阵列（UPA）及可重构智能表面（RIS）进行对比，结果显示FIM在上行NMSE上可下降≈9.8 dB、在下行SNR上提升≈1.7 dB，且在少路径场景下几乎达到完美CSI条件下的性能；

**⚠️ 局限性**

局限性包括仅考虑单用户场景、DFP求解可能陷入局部最优、需要多次随机初始点、未评估能耗和实际形变限制、对时变或高切换场景的适应性仍需进一步研究。

---

## 36. World2Rules: A Neuro-Symbolic Framework for Learning World-Governing Safety Rules for Aviation

**arXiv ID:** 2603.28952 | [PDF](https://arxiv.org/pdf/2603.28952v1)

**作者:** Haichuan Wang `[一作]` (Carnegie Mellon University), Sebastian Scherer `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了名为World2Rules的神经-符号框架，利用多模态数据（文本事故报告和航迹图像）通过预训练LLM/VLM提取符号事实，再通过ILP验证与层级一致性检查学习航空安全规则；

**💡 创新点**

创新点在于将神经提取器与符号ILP验证层级联接，形成四级一致性反馈循环（提取、子集、聚合、规则），实现对噪声、错误与不一致性的自动纠正与剪枝；

**🔧 技术方法**

技术包括大型语言模型（LLM）和视觉-语言模型（VLM）用于符号抽取、Popper ILP引擎与约束求解器、层级一致性检查、迭代全局聚合、支持度剪枝；

**📊 数据集**

使用航空事故报告集ASIAS与航迹数据集Amelia‑48作为训练输入，并构建了38个跑道侵入场景的人工标注测试集；

**📈 对比分析**

与LLM‑only与单次ILP基线对比，World2Rules在94.0% F1、接近100%精确度、显著提升召回率，分别比LLM‑only高23.6pp、比单次ILP高43.2pp；

**⚠️ 局限性**

局限在于测试规模有限、模型仅处理静态关系缺乏时间推理、词汇表固定导致表达受限，且对大规模复杂关系或不确定性处理不足。

---

## 37. Towards Explainable Stakeholder-Aware Requirements Prioritisation in Aged-Care Digital Health

**arXiv ID:** 2603.29114 | [PDF](https://arxiv.org/pdf/2603.29114v1)

**作者:** Yuqing Xiao `[一作]` (Monash University), Elizabeth Manias `[通讯]` (Monash University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过混合方法研究，利用可解释机器学习识别影响老年护理数字健康需求优先级的人类因素，并通过访谈验证和解释这些因素的作用；

**💡 创新点**

创新点在于将SHAP等可解释模型与人机交互式访谈相结合，系统揭示不同利益相关者（老年人、照护者、开发者）在人类因素上的偏好差异，并提供可操作的需求优先级框架；

**🔧 技术方法**

使用的技术包括监督学习模型（随机森林、XGBoost 等）、模型无关解释方法 SHAP 与置换重要性、以及半结构化访谈的质性分析；

**📊 数据集**

使用的数据集为 249 名受访者（103 老年人、105 开发者、41 照护者）的问卷数据，包含 19 个预测变量与 45 个需求优先级结果；

**📈 对比分析**

方法采用 5 折交叉验证，评价指标包括加权 F1、宏 F1、AUROC 与 AUPRC；模型性能在加权 F1 0.484–0.807、AUROC 0.614–0.907 之间，表明模型在解释性和预测性方面均具备可接受水平；

**⚠️ 局限性**

局限性包括样本偏倚（偏向数字素养较高者）、结果可能不具备跨文化普适性、仅基于自报数据且模型解释性未必对应因果关系、以及未捕捉到真实照护关系中的相互影响等。

---

## 38. Gleanmer: A 6 mW SoC for Real-Time 3D Gaussian Occupancy Mapping

**arXiv ID:** 2603.29005 | [PDF](https://arxiv.org/pdf/2603.29005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 39. Differentiable Initialization-Accelerated CPU-GPU Hybrid Combinatorial Scheduling

**arXiv ID:** 2603.28943 | [PDF](https://arxiv.org/pdf/2603.28943v1)

**作者:** Mingju Liu `[一作]` (University of Maryland), Cunxi Yu `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种CPU‑GPU混合框架，将可微分优化产生的高质量部分解作为warm‑start，显著加速基于整数线性规划（ILP）的组合调度求解。

**💡 创新点**

创新点在于首次将可微分优化（受限Gumbel‑Softmax + 差分约束）生成的部分解直接用于传统ILP求解器的warm‑start，既保持了可解释的最优性，又获得了梯度优化的速度优势。

**🔧 技术方法**

技术包括可微分差分约束（SDC）建模、受限Gumbel‑Softmax采样、梯度下降式可微优化、GPU并行加速，以及使用CPLEX、Gurobi和HiGHS等主流ILP求解器进行精确求解。

**📊 数据集**

使用的数据集包含EPFL硬件合成设计、12个随机生成的RW工作负载以及将EPFL设计映射到GPU后得到的GPU图形，规模覆盖从数千到上百万约束不等。

**📈 对比分析**

实验通过在同一硬件上对比冷启动ILP求解器与30个可微warm‑start并行求解的性能，结果显示平均可达到10×的速度提升，且最终解的最优性误差低于0.1%。

**⚠️ 局限性**

局限性包括：对超大规模实例仍存在求解瓶颈；warm‑start的质量对结果影响较大；对非SDC形式的ILP问题适用性尚待验证；并且该方法仍需GPU资源进行可微分优化。

---

## 40. Designing FSMs Specifications from Requirements with GPT 4.0

**arXiv ID:** 2603.29140 | [PDF](https://arxiv.org/pdf/2603.29140v1)

**作者:** Omer Nguena Timo `[一作]` (University of Quebec in Outaouais), Florent Avellaneda `[通讯]` (University of Quebec in Montreal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于大型语言模型（GPT‑4）的框架，利用自然语言需求自动生成有限状态机（FSM），并设计了多种基于语法、区分序列、检查序列和缺陷模型的修复方法；

**💡 创新点**

创新点在于将LLM与模型驱动工程相结合，构建了可迭代的修复流程，并提出了专门针对LLM常见错误的缺陷模型修复域；

**🔧 技术方法**

技术主要包括LLM提示工程、自动生成FSM描述、语法与语义差异检测、区分序列/检查序列搜索、基于SAT的缺陷模型挖掘；

**📊 数据集**

使用的是随机生成的FSM及其对应的自然语言描述，实验规模涵盖5、10、25个状态的机器；

**📈 对比分析**

通过与oracle FSM 的语法与语义比较评估，语法修复在所有实验中达成100%修复成功率，而基于区分/检查序列的语义修复成功率仅为0%至40%，缺陷模型修复在所有实验中亦实现100%修复成功；

**⚠️ 局限性**

主要局限包括：LLM在处理大规模状态机时错误率上升；缺乏对工业级真实需求的验证；修复流程依赖oracle或人工专家；区分/检查序列生成复杂且耗时，导致实用性受限。

---

## 41. From Astronomy to Astrology: Testing the Illusion of Zodiac-Based Personality Prediction with Machine Learning

**arXiv ID:** 2603.29033 | [PDF](https://arxiv.org/pdf/2603.29033v1)

**作者:** Abhinna Sundar Samantaray `[一作]` (Astronomical Computing Institute University of Heidelberg), Dhruv Vansraj Rathore `[通讯]` (Hindalco Industries Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用机器学习方法检验星座信息是否能有效预测人格特质

**💡 创新点**

通过合成数据模拟星座与人格标签的关联，探究星座描述的统计意义

**🔧 技术方法**

逻辑回归、随机森林和多层感知机（MLP）

**📊 数据集**

自构造的包含100个通用人格特质的合成数据集

**📈 对比分析**

与随机基线及标签打乱控制对比，准确率与随机水平持平，无显著提升

**⚠️ 局限性**

星座类别信息稀缺、特质高度重叠导致模型难以捕获有效预测信号

---

## 42. Uncovering Relationships between Android Developers, User Privacy, and Developer Willingness to Reduce Fingerprinting Risks

**arXiv ID:** 2603.29063 | [PDF](https://arxiv.org/pdf/2603.29063v1)

**作者:** Alex Berke `[一作]` (Google), Mihai Christodorescu `[通讯]` (Google)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对246名安卓开发者进行问卷调查，评估其对降低设备指纹追踪的“API 使用目的”平台改动的支持度及其对开发者工作量与用户隐私影响的认知。

**💡 创新点**

首次量化开发者对指纹追踪相关平台干预的接受度，并揭示使用指纹追踪的开发者更倾向支持此类隐私增强措施的逆向结果，突显潜在协作机会。

**🔧 技术方法**

采用混合方法：定量问卷分析（逻辑回归）与定性主题分析（开放式回答），配合对开发者感知工作量与隐私收益的李克特量表。

**📊 数据集**

收集自246名具备Android开发经验的开发者自我报告数据，涉及其指纹追踪使用情况、对Android/iOS隐私保护的评价及对假设改动的支持与关切。

**📈 对比分析**

通过对“必需模型”与“可选模型”支持率的对比与回归结果，发现约89%支持改动，且支持度与对隐私收益的正向认知相关、对工作量的负向认知相关，指纹追踪使用者支持度更高（OR≈6.5）。

**⚠️ 局限性**

局限于Android开发者样本、基于自我报告与假设改动的情景、缺乏跨平台验证以及对真实指纹追踪实现的客观测量。

---

## 43. Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting

**arXiv ID:** 2603.29185 | [PDF](https://arxiv.org/pdf/2603.29185v1)

**作者:** Huaqi Tao `[一作]` (Southern University of Science and Technology), Hong Zhang `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于 Feature Gaussian Splatting 的层次视觉重定位框架 SplatHLoc。

**💡 创新点**

创新点在于自适应视角检索提升初始位姿精度，以及混合特征匹配策略（渲染特征做粗匹配、半稠密特征做细匹配）。

**🔧 技术方法**

使用 Feature Gaussian Splatting 作为场景表示，结合 MixVPR、SuperPoint+LightGlue、JamMa 等匹配器，并利用 gsplat 进行渲染。

**📊 数据集**

在 7-Scenes、12-Scenes 与 Cambridge Landmarks 三个公开数据集上进行实验。

**📈 对比分析**

与结构、回归、NeRF、GS 等前沿方法对比，SplatHLoc 在室内外多场景下均实现了更低的位姿误差和更快的运行时（比 STDLoc 低 30–50%）。

**⚠️ 局限性**

方法依赖于高质量的 Gaussian 图像映射，映射精度受训练图像数量影响，且对大规模地图的扩展仍有挑战。

---

## 44. REFINE: Real-world Exploration of Interactive Feedback and Student Behaviour

**arXiv ID:** 2603.29142 | [PDF](https://arxiv.org/pdf/2603.29142v1)

**作者:** Fares Fawzi `[一作]` (EPFL), Tanja Käser `[通讯]` (EPFL)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 REFINE，一个基于小型开源 LLM 的多智能体交互式反馈系统，包含评判者驱动的反馈生成和工具调用的交互式问答模块。

**💡 创新点**

创新点在于将反馈视为交互过程，使用人类对齐的评判者进行迭代改进，并将工具调用与自我反思推理结合以实现可解释、可操作的后续问答。

**🔧 技术方法**

技术包括小型开源 LLM（Qwen3、Qwen2.5‑VL、Qwen3‑8B）、人类对齐评判者、工具调用框架、LoRA 微调、混合检索（BM25 + 语义检索）、多步自反推理。

**📊 数据集**

数据集包括 D_fb（开放式离散数学证明任务）、D_int_train/D_int_test（学生问答与工具调用轨迹）、课堂实验数据 D_study_feedback/D_study_int，以及期末备考数据 D_prep_feedback/D_prep_int，亦使用公开的参考答案和教师标注。

**📈 对比分析**

与 GPT‑5、Qwen3‑30B‑Thinking 及基线模型对比，评判者驱动改进后反馈清晰度、诊断、行动建议提升显著；交互式问答模型 Qwen3‑8B‑REFINE 在相关性、可操作性与工具相关性上接近 GPT‑5，且工具使用更高效。

**⚠️ 局限性**

局限性包括仅在 EPFL 同一门课程实验、未评估直接学习成效、实时约束下仅迭代部分评判维度、课堂实验中强制性提问可能夸大交互量。

---

## 45. DF-ACBlurGAN: Structure-Aware Conditional Generation of Internally Repeated Patterns for Biomaterial Microtopography Design

**arXiv ID:** 2603.28776 | [PDF](https://arxiv.org/pdf/2603.28776v1)

**作者:** Rongjun Dong `[一作]` (University of Nottingham), Grazziela Figueredo `[通讯]` (University of Nottingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种结构感知的条件生成对抗网络DF-ACBlurGAN，用于在类别不平衡条件下生成具有内部重复和周期性结构的生物材料微表面图像。

**💡 创新点**

创新点包括：①利用FFT动态估计重复尺度并将其反馈给生成器；②结合自适应高斯模糊和单元格重构作为结构约束；③采用MLP生成器而非传统CNN以捕获全局周期性。

**🔧 技术方法**

采用WGAN-GP框架、FFT频域分析、自适应高斯模糊、单元格重构以及条件分类损失等技术。

**📊 数据集**

使用三组生物材料表面数据集：Pseudomonas aeruginosa、Staphylococcus aureus以及巨噬细胞吸附-极化任务（共2,800×2,800二值图像）。

**📈 对比分析**

与传统条件GAN及其消除各结构组件的消融模型相比，DF-ACBlurGAN在FID/TopoFID、ISResNet等指标上均表现更好，并通过合成数据增强显著提升了下游ResNet-50预测器的准确率。

**⚠️ 局限性**

局限性在于：①对离散化标签的依赖导致中间类别语义模糊；②FFT估计仅捕捉全局周期，难以处理多尺度或层级重复；③缺乏实验反馈或物理/制造约束的迭代闭环。

---

## 46. AI in Work-Based Learning: Understanding the Purposes and Effects of Intelligent Tools Among Student Interns

**arXiv ID:** 2603.28786 | [PDF](https://arxiv.org/pdf/2603.28786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 47. Federated Inference for Heterogeneous LLM Communication and Collaboration

**arXiv ID:** 2603.28772 | [PDF](https://arxiv.org/pdf/2603.28772v1)

**作者:** Zihan Chen `[一作]` (Singapore University of Technology and Design), Jihong Park `[通讯]` (Singapore University of Technology and Design)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了FedRefine框架，利用双向KV缓存通信实现异构LLM的联邦推理，并通过重写查询实现隐私保护。

**💡 创新点**

创新点在于：①用KV缓存而非文本token进行通信，减少前填充延迟；②构建双向Cache‑to‑Cache（C2C）fuser，支持任意两台异构模型间的互通；③加入隐私重写机制，使查询在保持语义的前提下匿名化。

**🔧 技术方法**

采用了Cache‑to‑Cache双向fuser（基于三层MLP投影），门控网络动态选择缓存来源，层对齐投影，隐私重写模块和边缘服务器上的提示/缓存优化。

**📊 数据集**

使用OpenHermes2.5数据集训练fuser，评估使用OpenBookQA数据集。

**📈 对比分析**

与单模型基线和传统文本对文本（T2T）通信做对比：在所有四个发送模型协同时，非隐私KV协同提升21.2%准确率；隐私KV协同仅下降3%；C2C比T2T提升约15%；KV通信每token需要88KB，而T2T仅16B；整体延迟仍低于T2T。

**⚠️ 局限性**

局限性包括：需要为每对模型单独预训练fuser，扩展性受限；KV通信带宽开销高；隐私重写虽降低泄露风险但会增加额外延迟；未实现多轮迭代、跨模态或更大规模的多机协同等功能。

---

## 48. Differential Privacy for Symbolic Trajectories via the Permute-and-Flip Mechanism

**arXiv ID:** 2603.28903 | [PDF](https://arxiv.org/pdf/2603.28903v1)

**作者:** Alexander Benvenuti `[一作]` (Georgia Institute of Technology), Matthew Hale `[通讯]` (Georgia Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于permute-and-flip机制的差分隐私框架，用于对符号系统（如马尔可夫链、有限状态自动机）生成的轨迹进行加密。

**💡 创新点**

创新点在于：1) 设计了一种无需枚举整个输出空间即可实现permute-and-flip机制的高效实现；2) 将此机制与马尔可夫链耦合，确保输出轨迹始终可行；3) 提供了理论误差上界，并证明其优于或等价于现有指数机制。

**🔧 技术方法**

使用的技术包括：改进的Hamming距离NFA（MNFA）与产品MNFA、策略合成算法、隐私参数ε与邻接参数b的调节，以及对误差期望的分析。

**📊 数据集**

实验使用了佛罗里达州阿尔伯克基市（Gainesville, Florida）年度平均日交通（AADT）数据，构造了包含43个状态的马尔可夫链模型。

**📈 对比分析**

通过在不同ε值下生成2000条私有轨迹，计算平均误差，与传统指数机制（exp机制）比较。结果显示：在ε=0.5时两者误差相近；在ε≥3时，本文方法平均误差至少比传统方法低25%，最高在ε=5时降低55.7%。

**⚠️ 局限性**

局限性包括：1) 机制目前仅支持离线批处理，无法实时输出；2) 对于极小ε时误差仍相当大；3) 对大型状态空间的扩展性尚未完全验证。

---

## 49. Working Paper: Towards a Category-theoretic Comparative Framework for Artificial General Intelligence

**arXiv ID:** 2603.28906 | [PDF](https://arxiv.org/pdf/2603.28906v1)

**作者:** Pablo de los Riscos `[一作]` (Cognodata R+D), Michael A. Arbib `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一个基于范畴论（Hypergraph Category、Profunctor、Grothendieck Fibration 等）的统一框架，用来形式化、比较和分析不同的人工通用智能（AGI）架构。论文给出了架构、代理、属性的抽象定义，并通过实例（RL、CRL、SBL、AIXI）演示了该框架的可操作性。

**💡 创新点**

① 将AGI架构抽象为“结构层”与“知识层”两个互为交互的超图范畴；② 引入 Profunctor 作为结构与知识的接口；③ 构建 ArchAgents 与 Agents 的纤维结构，实现在不同架构之间的翻译和重索引；④ 通过机构化的属性（结构、信息、语义）与证书（基于 Institution 的证明携带）来对代理的理论保证与实际表现进行形式化对比。

**🔧 技术方法**

核心技术：范畴论（超图范畴、对称单张量范畴、Frobenius 结构、Profunctor、Grothendieck Fibration）、机构化方法（Institution 与证明携带）、形式化语义（强单张量函子）、属性推导与证书验证。

**📊 数据集**

论文属于理论性研究，没有使用具体实验数据集；通过对 RL、Causal RL、Schema‑Based Learning、AIXI 等架构的形式化定义和示例，展示框架的适用性。

**📈 对比分析**

比较方法：利用范畴论的同构、子范畴、翻译函子来判定架构之间的相似性与可转译性；属性映射可通过单调映射在结构属性集合之间传播；通过证明携带的证书验证代理在给定属性下的满足性。由于本文未给出数值实验，性能表现仅以理论保证与结构可比性说明。

**⚠️ 局限性**

局限性：① 仅关注架构层和属性层，未给出具体实现或算法细节；② 目前缺乏实证验证，无法评估在真实任务中的有效性；③ 对复杂的动态环境与持续学习的细节处理仍在后续工作中；④ 由于框架高度抽象，迁移到现有技术栈时需要额外的实现工作。

---

## 50. Software Vulnerability Detection Using a Lightweight Graph Neural Network

**arXiv ID:** 2603.29216 | [PDF](https://arxiv.org/pdf/2603.29216v1)

**作者:** Miles Farmer `[一作]` (University of Missouri), Kannappan Palaniappan `[通讯]` (University of Missouri)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种轻量级图神经网络VulGNN，用于函数级别的漏洞检测。

**💡 创新点**

创新点在于将AST、CFG和PDG融合为代码属性图，并采用注意力卷积、图归一化和正则化，使模型仅拥有1.1M参数即可与LLM竞争。

**🔧 技术方法**

使用图注意力网络（GAT）、GeneralConv层、GraphNorm、StarCoder分词器以及PyTorch Geometric实现。

**📊 数据集**

主要在真实代码集DiverseVul和合成代码集SARD/Juliet上进行训练和评估。

**📈 对比分析**

与现有GNN和LLM基线在相同划分下比较，VulGNN在未见项目集上F1提升约6%，准确率与LLM相当，但参数量比LLM低两位数倍。

**⚠️ 局限性**

局限在于仅验证C/C++代码，对工业大规模项目的泛化尚待验证，且对标签质量与数据偏差存在依赖，缺乏统计显著性检验。

---

## 51. Concept Training for Human-Aligned Language Models

**arXiv ID:** 2603.29123 | [PDF](https://arxiv.org/pdf/2603.29123v1)

**作者:** Christine Zhang `[一作]` (Stanford University), Chen Shani `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种新的训练框架，通过预测概念而非单一的表面词汇来改进语言模型的下一个词预测（NTP）目标。

**💡 创新点**

创新点在于引入概念监督，使模型能够学习语义相关的词汇集合，从而提高与人类语义相似性判断的一致性。

**🔧 技术方法**

使用了自监督学习方法，结合了标准的NTP损失和概念损失，通过调整超参数来平衡两者的影响。

**📊 数据集**

使用了AllenAI的C4数据集和OpenWebText数据集，分别从中抽取了2000个英文文本序列进行训练。

**📈 对比分析**

与基线模型（无后训练和仅使用NTP损失的模型）相比，概念训练模型在多个语义相似性基准上表现出更高的Spearman相关性，且在内容词的困惑度和准确性上优于预训练和NTP训练的基线模型。

**⚠️ 局限性**

限制在于本研究仅关注解码器模型，未能评估在指令提示下的下游任务表现；概念监督仅应用于单个完整词的名词、动词和形容词，可能未能充分捕捉更长文本的概念学习效益；构建概念集的方法依赖于LLM生成的同义词聚类，可能不完全反映人类的语义组织。

---

## 52. Building the Palmetto API: Adding granular permissions and caching to the Slurm REST API without sacrificing compatibility

**arXiv ID:** 2603.29032 | [PDF](https://arxiv.org/pdf/2603.29032v1)

**作者:** Ben Godfrey `[一作]` (Clemson University), Doug Dawson `[通讯]` (Clemson University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

开发了 Palmetto API，一个基于 Slurm RESTful 接口的代理服务，添加了细粒度权限、MUNGE 与 Bearer Token 认证、Redis 缓存层，并保证与现有 Slurm API 的兼容性。

**💡 创新点**

创新点包括：① 在 Slurm 之上实现细粒度权限与账户限制；② 采用 MUNGE 与随机 opaque Bearer Token（包装成 JWT）双重认证；③ 设计可配置的缓存策略与失败回退机制；④ 通过 Go 的协程和并发组避免缓存风暴，提升大规模集群的并发稳定性。

**🔧 技术方法**

使用技术：Go 语言、Docker 容器、Redis、MUNGE、JWT、Slurm 工具（scontrol、slurmrestd）、OpenAPI 生成器、Locust、TLS/HTTPS 等。

**📊 数据集**

数据集/测试资源：实际 Palmetto 2 集群节点（1000+ 计算节点），使用 Locust 进行压力测试；兼容性测试通过多款开源 Slurm 客户端（Python SDK、Slurm HPC Dashboard、slurm-exporter、slurm-monitor）完成。

**📈 对比分析**

评估方法：对比缓存开启与关闭的 Load Test（使用 Locust 24 worker 10 分钟），记录每个端点的中位数和平均响应时间以及错误率；兼容性调查检验各客户端在不改动或少改动下的可用性。性能结果表明，缓存可将平均响应时间从几百毫秒降至几十毫秒，错误率显著下降，RPC 调用量大幅减少。

**⚠️ 局限性**

局限性：① 需要手动维护 Slurm API 版本兼容性，较旧的客户端可能不兼容；② 目前配置高度针对 Clemson HPC，迁移到其他集群需要进一步抽象和可配置化；③ Bearer Token 的随机生成方式不如标准 JWT 灵活，需手动轮转或扩展；④ 缓存回退默认关闭，对某些场景不友好；⑤ 仍需在安全策略和权限细化方面完善细节。

---

## 53. SyriSign: A Parallel Corpus for Arabic Text to Syrian Arabic Sign Language Translation

**arXiv ID:** 2603.29219 | [PDF](https://arxiv.org/pdf/2603.29219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 54. OccSim: Multi-kilometer Simulation with Long-horizon Occupancy World Models

**arXiv ID:** 2603.28887 | [PDF](https://arxiv.org/pdf/2603.28887v1)

**作者:** Tianran Liu `[一作]` (University of Toronto), Nicholas Rhinehart `[通讯]` (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于占据空间世界模型的自驾仿真框架OccSim，能够仅凭单帧与未来行驶轨迹生成千米级连贯的道路地图并生成交互式车辆流；

**💡 创新点**

创新点在于引入W-DiT（Warp‑DiT）实现超过3k步的稳定静态地图生成、使用Mask‑Injected Conditioning与SNR加权感知损失提升长程稳定性、以及基于Latent Flow Matching的布局生成器实现无日志交互式车辆初始化；

**🔧 技术方法**

核心技术包括占据空间VAE、变形Diffusion Transformer（W‑DiT）、光流Warp与可见性掩码、SNR‑加权感知损失、密钥帧融合与拓扑提取算法、2D‑IDM轨迹控制；

**📊 数据集**

主要数据集为公开的nuScenes和Waymo占据空间数据（Occ3D‑nuScenes、UniOcc‑Waymo、UniOcc‑nuScenes），以及OccFM/UniScene的VAE/AE嵌入；

**📈 对比分析**

通过FID、KID、MMD、Pairwise mIoU多样性、Vendi Score等指标与现有SOTA占据世界模型（OccWorld、DOME、COME等）比较，W‑DiT在3k步长程稳定性、生成多样性和下游4D语义占据预测的零样本性能上分别提升约80×、显著高于竞争方法；

**⚠️ 局限性**

局限在于需要足量标注占据数据、对旋转不变性假设敏感、融合与拓扑提取仍为启发式、以及目前仅采用2D‑IDM控制，未来需实现端到端融合、无损旋转操作与更大规模训练。

---

## 55. Human-Like Lifelong Memory: A Neuroscience-Grounded Architecture for Infinite Interaction

**arXiv ID:** 2603.29023 | [PDF](https://arxiv.org/pdf/2603.29023v1)

**作者:** Diego C. Lerma-Torres `[一作]` `[通讯]` (Universidad de Guanajuato), Diego C. Lerma-Torres (Universidad de Guanajuato)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于生物启发、认知心理学与补充学习系统理论的记忆框架，旨在为大语言模型提供持久、结构化的长期交互记忆；

**💡 创新点**

创新点包括：1）将情感价值视为记忆核心，构造 valence 向量；2）默认系统1检索、系统2升级的双流程路由；3）主动、目标驱动的编码机制与丘脑网关的多通道门控；

**🔧 技术方法**

技术实现涵盖：知识图谱双层访问（gist/valence 向量与完整图）、系统1/系统2双流程切换、丘脑门控的情感与置信度多通道评分、情感激活的传播激活与重巩固更新机制；

**📊 数据集**

本文为理论框架，未使用具体数据集，后续可基于通用 LLM 预训练权重与用户交互日志实现；

**📈 对比分析**

未实现实验比较，作者预期将通过未来实验评估系统1/系统2转换率、情感触发检索效果、幻觉率与成本降低等指标；

**⚠️ 局限性**

局限包括：多维情感评分阈值与门控参数需经验校准；主动形成机制的实现细节待验证；在高度不确定或混沌环境下可能难以收敛到系统1；对幻觉减少效果的实验验证尚待完成。

---

## 56. Realistic Market Impact Modeling for Reinforcement Learning Trading Environments

**arXiv ID:** 2603.29086 | [PDF](https://arxiv.org/pdf/2603.29086v1)

**作者:** Lucas Riera Abbade `[一作]` (University of Sao Paulo), Anna Helena Reali Costa `[通讯]` (University of Sao Paulo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并评估包含Almgren–Chriss与平方根冲击法的可插拔成本模型的三种Gymnasium兼容交易环境，并比较五种DRL算法在不同成本模型下的表现。

**💡 创新点**

首次将非线性市场冲击模型与永久冲击衰减机制整合到RL交易环境中，证明其显著改变算法排名和交易行为，并强调超参数优化对抑制过度交易的关键作用。

**🔧 技术方法**

使用Gymnasium、FinRL-Meta扩展、Almgren–Chriss与square‑root冲击模型、Optuna超参数优化、Stable‑Baselines3（A2C、PPO、DDPG、SAC、TD3）以及DSR奖励与完整交易日志。

**📊 数据集**

采用2010‑2026年NASDAQ 100每日行情（开盘、收盘、最高、最低、成交量），90/10训练/测试划分，真实OOS为2025年数据。

**📈 对比分析**

在两种成本模型（10 bps基准与AC模型）及默认与Optuna优化参数下对五种算法进行20个回测，评估年化回报、Sharpe、交易成本、波动率等指标，发现AC模型显著提升TD3等算法性能、降低交易成本，而未优化算法则出现高成本和过度交易。

**⚠️ 局限性**

仅使用固定NASDAQ 100成分，未考虑每日指数变动；未系统评估Obizhaeva–Wang模型与借贷成本；超参数优化仅针对Sharpe，可能忽视最大回撤；模型对小盘股冲击估计不足。

---

## 57. TORCH: Characterizing Invalid Route Filtering via Tunnelled Observation

**arXiv ID:** 2603.29207 | [PDF](https://arxiv.org/pdf/2603.29207v1)

**作者:** Renrui Tian `[一作]` (Tsinghua University), Zhiliang Wang `[通讯]` (Tsinghua University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

开发了TORCH框架，利用开放6in4隧道节点和跨平面推断技术，对IPv6网络的ROV保护进行大规模测量。

**💡 创新点**

创新点在于将开放隧道点转化为测量视角，扩展无响应目标的无缝可达性推断，并系统利用野外无效前缀实现对ROV的全面评估。

**🔧 技术方法**

采用ICMPv6 traceroute、跨平面路径匹配、RPKI ROA验证、BGP RIB收集等技术。

**📊 数据集**

使用的主要数据集包括RouteViews BGP RIB、Routinator ROA校验结果、开放6in4隧道端点扫描、AS‑org映射、APNIC RPKI监测等。

**📈 对比分析**

与APNIC单前缀评估及IPv4 RoVista对比，发现TORCH在ROV分数、覆盖度和碰撞损伤检出上提升约30‑40%，准确率达85%以上。

**⚠️ 局限性**

局限在于对AS‑org映射的依赖、路由平衡与动态变化导致的路径误差，以及无法直接验证内部路由决策的深度。

---

## 58. Embeddings of Nation-Level Social Networks

**arXiv ID:** 2603.29059 | [PDF](https://arxiv.org/pdf/2603.29059v1)

**作者:** Tanzir Pial `[一作]` (Stony Brook University), Steven Skiena `[通讯]` (Stony Brook University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对全国级多层时间演化社会网络，构建并评估动态节点嵌入方法；

**💡 创新点**

创新点包括层感知随机游走、按年份线性对齐嵌入空间、以及使用Fibonacci螺旋与白化实现均衡节点分区；

**🔧 技术方法**

采用DeepWalk改进的层感知随机游走、线性回归对齐、PCA白化和Fibonacci网格分区；

**📊 数据集**

使用荷兰统计局（CBS）从2009年至2023年的17M节点、14亿边的多层人口网络，以及LISS调查和行政登记数据；

**📈 对比分析**

与传统平面化、Orthogonal Procrustes对齐、层无视深度嵌入等基线比较，层感知嵌入在13项预测任务（收入、婚育、调查问卷、亲属识别）中平均提升约10% AUC/15% R²，线性回归对齐比Procrustes相关系数高0.05以上；

**⚠️ 局限性**

局限在于对时间变化的捕捉仍有限，白化会引入一定结构扭曲，且仅基于静态预测任务，未探索更深层的因果或解释性分析。

---

## 59. CivicShield: A Cross-Domain Defense-in-Depth Framework for Securing Government-Facing AI Chatbots Against Multi-Turn Adversarial Attacks

**arXiv ID:** 2603.29062 | [PDF](https://arxiv.org/pdf/2603.29062v1)

**作者:** KrishnaSaiReddy Patil `[一作]` `[通讯]`, KrishnaSaiReddy Patil

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了CivicShield，一个面向政府服务的跨域防御深度框架，旨在保护政府面向公众的AI聊天机器人免受多轮攻击；

**💡 创新点**

创新点在于将网络安全、形式化验证、生物免疫系统、航空安全工程与零信任密码学等跨领域技术集成到七层防御体系，并给出正式威胁模型和与NIST 800‑53等合规标准的映射；

**🔧 技术方法**

采用零信任基础设施（能力令牌与加密提示验证）、边界输入验证、语义防火墙、对话状态机（安全不变式）、行为异常检测（人工免疫系统）、多模型一致性验证（TMR）和分级人机协作的安全上升；

**📊 数据集**

使用 HarmBench、JailbreakBench、XSTest、CitizenQuery‑UK 等公开基准以及自建的政府场景数据集进行评估；

**📈 对比分析**

通过分层消融、模拟攻击与正式基准比较，CivicShield在多轮攻击场景下的检测率达72.9%（CI 69.5–76.0%），误报率降至2.9%（CI 1.9–4.4%），显著优于单层防御和现有同类系统；

**⚠️ 局限性**

局限性包括：评估受限于模拟与自建场景的偏差、层间关联失真导致的保真度降低、对模型提取与成员推断等攻击缺乏专门防护、成本与复杂性上升以及对高度自适应攻击者的理论上仍存在安全盲区。

---

## 60. WAter: A Workload-Adaptive Knob Tuning System based on Workload Compression

**arXiv ID:** 2603.28809 | [PDF](https://arxiv.org/pdf/2603.28809v1)

**作者:** Yibo Wang `[一作]` (Purdue University), Mingjie Tang `[通讯]` (Sichuan University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了名为 WAter 的基于工作负载压缩的数据库调优系统，能在多时间片内动态选择并优化查询子集，从而显著降低每次评估的工作负载执行时间。

**💡 创新点**

创新点包括：①时间片分割与动态子集更新；②基于运行时指标的代表性度量与贪心子集选择；③历史重用以快速构建子集专用的代理模型；④全局代理模型与混合评分机制实现探索与利用的平衡。

**🔧 技术方法**

采用了机器学习中的贝叶斯优化/强化学习框架（与 SMAC、GPTuner 集成）、随机森林回归全局代理、Gower 距离测度、贪心算法以及历史重用技术。

**📊 数据集**

使用了多种 OLAP 基准：TPC‑DS（sf=1）、JOB（5.2 GB）、TPC‑H（sf=10）以及扩展的 TPC‑H×10 与 LLM 生成的 IMDB 查询集。

**📈 对比分析**

与原始调优器、GSUM、随机子集等基线相比，WAter 在平均 4.2× 的时间内找到更优配置，最终执行时间平均降低 39.1%（比 GPTuner 快 6.4%），且在不同硬件、规模、并发与 LLM 生成的工作负载下均保持较高的加速比（最高 12.9× 的时间‑到‑最优加速）。

**⚠️ 局限性**

局限性包括：①对子集代表性度量的依赖，仍可能在极端工作负载多样性下误判；②在极小查询集（如 TPC‑H）下收敛速度受限；③需额外的子集选择与历史重用开销，对极高维度参数空间的可扩展性尚待验证。

---

## 61. ARTLAS: Mapping Art-Technology Institutions via Conceptual Axes, Text Embeddings, and Unsupervised Clustering

**arXiv ID:** 2603.28816 | [PDF](https://arxiv.org/pdf/2603.28816v1)

**作者:** Joonhyung Bae `[一作]` `[通讯]` (Korea Advanced Institute of Science and Technology), Joonhyung Bae (Korea Advanced Institute of Science and Technology)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 ARTLAS 框架，利用八维概念轴对全球 78 家艺术技术机构进行文本嵌入、代码书量化、UMAP 降维、聚类、主题建模和边界分析，并通过交互式网页可视化展示结果。

**💡 创新点**

首次将理论驱动的多维概念框架与现代句子嵌入、词级代码书量化、无监督聚类和熵边界检测结合，形成面向机构生态的可复现、数据驱动分析流程；并将成果开放为可视化工具。

**🔧 技术方法**

E5-large-v2 句子嵌入、词级代码书量化、UMAP 降维、Agglomerative(平均链接)聚类、非负矩阵分解（NMF）主题建模、邻居簇熵度量、React/TypeScript 前端可视化。

**📊 数据集**

包含 78 家来自 25 个国家的艺术技术机构（节庆、双年展、实验室、研究中心、博物馆、驻地、奖项等），每家机构基于八个轴用 15–40 词的英文描述进行注释。

**📈 对比分析**

对比了多种聚类算法（Agglomerative、OPTICS、k-means、DBSCAN）并用复合指标评估；Agglomerative(平均链接，k=10) 取得复合得分0.825、轮廓系数0.803、Calinski–Harabasz 11,196，聚类稳定性（ARI≈0.81）优于其他方法；嵌入层面 E5-large-v2 也优于 GTE、SBERT 等组合。

**⚠️ 局限性**

样本规模有限（仅78家），偏重西方机构，单一注释者导致主观性缺失，静态快照未考虑机构演变，未进行正式用户可用性评估，且方法对全球南方生态的普适性仍待验证。

---

## 62. PAR$^2$-RAG: Planned Active Retrieval and Reasoning for Multi-Hop Question Answering

**arXiv ID:** 2603.29085 | [PDF](https://arxiv.org/pdf/2603.29085v1)

**作者:** Xingyu Li `[一作]` (Oracle AI), Dan Roth `[通讯]` (Oracle AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PAR^2-RAG 两阶段框架，先通过规划与检索构建广泛的证据边界，再在该边界内进行深度链式推理；

**💡 创新点**

将检索覆盖与推理承诺分离，采用覆盖锚定与证据充分性控制机制，避免早期承诺导致的检索错误；

**🔧 技术方法**

基于检索增强生成（RAG）与大语言模型（如 GPT‑4.1、GPT‑o4‑mini）实现规划、检索、查询改写和证据充分性控制；

**📊 数据集**

在四个多跳问答基准上进行评估，分别为 2WikiMultiHopQA、MuSiQue、MoreHopQA 与 FRAMES；

**📈 对比分析**

与 Direct Inference、Chain‑of‑Thought、ReAct、IRCoT 等无训练基线对比，PAR^2‑RAG 在答案准确率上比 IRCoT 提升最高 23.5%，检索 Recall 提升 10.3%，NDCG 提升 10.5%；

**⚠️ 局限性**

主要局限在于对强大 LLM 与多步骤交互的依赖，导致计算成本高；对极深跳数或关键证据缺失仍可能受限，且需手动调节检索阈值与预算。

---

## 63. Koopman Operator Framework for Modeling and Control of Off-Road Vehicle on Deformable Terrain

**arXiv ID:** 2603.28965 | [PDF](https://arxiv.org/pdf/2603.28965v1)

**作者:** Kartik Loya `[一作]` (Clemson University), Phanindra Tallapragada `[通讯]` (Clemson University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种物理模型与数据驱动相结合的 Koopman 演算器框架，用于在软土地形上预测并控制离线自主越野车辆的动态行为，并将其嵌入受限 MPC 进行实时轨迹跟踪；

**💡 创新点**

创新点包括：① 在高保真 Bekker‑Wong 软土模型的基础上使用递归子空间识别（K‑SSID）得到可即时更新的 Koopman 线性预测器；② 利用 Grassmannian 距离对新增数据进行信息量筛选，仅保留真正能提升模型的实验段；③ 为不同土壤类型训练专属 Koopman 运算符，显著提高预测精度并避免跨土壤误差；

**🔧 技术方法**

核心技术包括 Koopman 变换理论、Gaussian‑Process 作为提升映射、递归子空间识别、Grassmannian 距离筛选、约束 MPC（K‑MPC）以及高保真 Bekker‑Wong 软土模型和 5‑DOF 车辆动力学；

**📊 数据集**

使用仿真生成的两套大型数据集：每种土壤（沙壤土、粘土）1600 条 20 s（100 Hz）轨迹，包含多种激励模式（直行、多频斜坡、鱼钩等），并在不同初始姿态、速度和轮速下采样；

**📈 对比分析**

通过比较不同模型阶数、刷新间隔、地形高度扰动及不同机动类型的 RMSE，验证模型在沙壤土下 RMSE 约 0.26–0.29、粘土下 0.30–0.39，跨土壤误差可达 90%+；在 MPC 中实现实时轨迹跟踪，求解时间 0.085 s，使用专属土壤运算符时目标函数显著低于误匹配的情况；

**⚠️ 局限性**

局限性在于：① 仅验证了最大 0.1 m 的轻微地形高度变化；② 需要为每种土壤预先训练专属 Koopman 运算符，难以推广到未知或混合土壤；③ 模型阶数选择与递归更新的稳定性需进一步研究；④ 依赖高保真仿真数据，真实工况下参数不确定性和测量噪声仍需在线自适应改进。

---

## 64. When GPUs Fail Quietly: Observability-Aware Early Warning Beyond Numeric Telemetry

**arXiv ID:** 2603.28781 | [PDF](https://arxiv.org/pdf/2603.28781v1)

**作者:** Michael Bidollahkhani `[一作]` (Gesellschaft für wissenschaftliche Datenverarbeitung mbh göttingen), Julian M. Kunkel `[通讯]` (Gesellschaft für wissenschaftliche Datenverarbeitung mbh göttingen)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种面向 GPU 节点“静默失效”事件的可观测性驱动早期预警框架，融合 GPU 典型热漂移信号与监控管道降解指标（如抓取延迟、样本丢失、时间序列缺口），并在 GWDG 生产环境的 GPU 节点日志上进行验证与评估。

**💡 创新点**

创新点在于：①将结构化指标（设备计数消失、抓取负载崩塌）视为一类重要异常信号，突破传统仅关注数值偏差的监控范式；②设计了联合 GPU 与可观测性平面特征的可复现分析管道；③在无精确故障标签的条件下，通过“弱事件”与事件对齐两种评价方式，提供可量化的预警领先时间。

**🔧 技术方法**

使用的技术包括：滑窗聚合（窗口60 min，步长10 min），GPU 统计特征提取（均值、最大值、标准差、斜率），监控管道特征提取（抓取时长、成功率、样本计数），三种无监督检测器（稳健 z‑score、Isolation Forest、One‑Class SVM），预算化报警（固定 1% 触发阈值），以及基于弱事件阈值和操作员事件目录的对齐评估。

**📊 数据集**

使用数据集：GWDG GPU 节点生产数据集（DCGM、Node Exporter、Prometheus 抓取指标、Slurm 节点状态日志），时间范围为 2025‑01 至 2026‑02，包含 7 个 GPU 断开事件（5 个完整记录），公开发布于 Zenodo。

**📈 对比分析**

比较方法：在同一 1% 报警预算下，对比 GPU 平面、联合平面三种检测器的平均领先时间、最大领先时间等指标。实验结果表明，联合 Isolation Forest 在平均领先时间上提升至 7 个窗口（≈70 min），最大领先时间提升至 29 个窗口；GPU 仅平面检测器平均领先时间仅 2-3 个窗口，且多数事件的中位领先时间为 0。

**⚠️ 局限性**

局限性：①缺乏精确的组件级故障标签，导致评价需依赖弱事件或操作员目录；②部分节点缺失清洗后数据，导致样本不足；③评估仅针对 GPU 断开事件，无法直接推广到其他故障模式；④窗口长度与采样频率的固定设置可能不适用于不同监控频率的系统；⑤对结构化信号的定义相对简单，仍有改进空间。

---

## 65. ChartDiff: A Large-Scale Benchmark for Comprehending Pairs of Charts

**arXiv ID:** 2603.28902 | [PDF](https://arxiv.org/pdf/2603.28902v1)

**作者:** Rongtian Ye `[一作]` `[通讯]` (Aalto University), Rongtian Ye (Aalto University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了ChartDiff基准，包含8541对多样化图表及其差异总结，并对多类视觉语言模型在跨图表比较摘要任务中的表现进行评估。

**💡 创新点**

首次构建大规模跨图表对比摘要基准，并揭示了传统词汇重叠指标与人类评判之间的显著不匹配。

**🔧 技术方法**

使用LLM生成与审核注释的流水线、生成式视觉语言模型（如GPT‑5.4、Gemini 3.1 Pro等）、图表提取管线DePlot以及多种绘图库（Matplotlib、Plotly、Plotnine）进行图表渲染。

**📊 数据集**

基于公开时序表格数据（Macrotrends、Yahoo Finance、Visual Crossing等），涵盖经济、健康、移民等八大领域，生成线图、柱图、饼图等多种图表类型。

**📈 对比分析**

通过ROUGE和GPT‑Score两种指标比较模型，闭源通用LLM在GPT‑Score上表现最佳，专用与管线模型在ROUGE上优势，但在多系列图表上仍显困难。

**⚠️ 局限性**

限制包括基准仅覆盖有限的常见图表类型、注释部分依赖LLM可能产生偏差、评估依赖GPT‑Score而非全面人类判断，以及仅关注对比摘要任务，缺少更复杂多图推理场景。

---

## 66. SNEAKDOOR: Stealthy Backdoor Attacks against Distribution Matching-based Dataset Condensation

**arXiv ID:** 2603.28824 | [PDF](https://arxiv.org/pdf/2603.28824v1)

**作者:** He Yang `[一作]` (Xi'an Jiaotong University), Jizhong Zhao `[通讯]` (Xi'an Jiaotong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Sneakdoor，针对基于分布匹配的数据集蒸馏过程的隐蔽后门攻击，利用输入感知触发器和决策边界弱点进行注入。

**💡 创新点**

创新点在于：①将触发器生成对每个样本进行自适应设计，使其与本地特征几何对齐；②结合分布匹配蒸馏的损失，联合优化攻击成功率、干净测试准确率和隐蔽性；③在 RKHS 上给出触发器隐蔽性的理论保证。

**🔧 技术方法**

使用的技术包括：生成式触发器网络、分布匹配蒸馏（MMD、IDM、DAM 等）、对抗训练、鲁棒性评估（NC、PIXEL、RNP、PDB）以及 PSNR/SSIM/IS 等隐蔽性指标。

**📊 数据集**

实验数据集包括 FMNIST、CIFAR-10、SVHN、Tiny-ImageNet、STL-10 与 ImageNette，使用 ConvNet、AlexNetBN、VGG11、ResNet18 等网络进行下游训练。

**📈 对比分析**

与 NAIVE、DOORPING、SIMPLE、RELAX 等基线对比，Sneakdoor 在 ASR‑CTA‑STE 三维指标上实现更优平衡，PSNR、SSIM 较高、IS 较低，且在多种防御下仍保持高攻击成功率。

**⚠️ 局限性**

局限性：在单一指标上不一定优于所有基线；需要较高的毒化比例；对复杂源‑目标映射的攻击效果有限；对更广泛威胁模型的适应性尚待研究。

---

## 67. LightHarmony3D: Harmonizing Illumination and Shadows for Object Insertion in 3D Gaussian Splatting

**arXiv ID:** 2603.29209 | [PDF](https://arxiv.org/pdf/2603.29209v1)

**作者:** Tianyu Huang `[一作]` (University of Sydney), Tongliang Liu `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `da1b1a89-583a-4b57-9c81-478778569bec` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LightHarmony3D 框架，实现对 3D Gaussian Splatting 场景中外部网格的光照一致性插入。

**💡 创新点**

创新点在于利用扩散模型一次性生成 360° HDR 环境贴图，结合 PBR 导向的阴影合成与光线解耦可见性，融合混合 Gaussian‑mesh 表征，并建立专属插入基准。

**🔧 技术方法**

使用 MILo 进行混合 Gaussian‑mesh 重建，Fine‑tuned latent diffusion 进行曝光截断与 HDR 生成，PBR 渲染与光线解耦阴影比例图计算，HDR 视景融合等技术。

**📊 数据集**

评估使用 Mip-NeRF360 实际场景数据以及自建的 LH3D‑Bench（LH3D‑Ku 与 LH3D‑Blender）合成数据集。

**📈 对比分析**

与 GIGS、GaussianEditor、MV‑CoLight、GaSLight 等基线在 PSNR、SSIM、LPIPS、VQAScore 上对比，LightHarmony3D 在大多数指标上领先，显示出更好的光照一致性与视觉逼真度。

**⚠️ 局限性**

局限性包括对重建网格精度的高度依赖，稀疏观察或复杂间接光照时效果受限，PBR 阴影渲染计算量大，扩散光照提取假设光源分布不充分时可能失效。

---

## 68. Enhancing Policy Learning with World-Action Model

**arXiv ID:** 2603.28955 | [PDF](https://arxiv.org/pdf/2603.28955v1)

**作者:** Yuci Han `[一作]` (Ohio State University), Alper Yilmaz `[通讯]` (Ohio State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 World-Action Model (WAM)，在 DreamerV2 结构上加入逆动力学预测头，以动作正则化提升世界模型的表示能力。

**💡 创新点**

创新点在于将动作预测作为训练正则化，使潜在状态在学习过程中主动编码动作相关信息，从而显著改善下游控制表现。

**🔧 技术方法**

使用 RSSM+Inverse Dynamics MLP 的 WAM、扩散策略 DiffusionMLP、PPO 进行模型自体微调，并在 WAM 训练中融合图像重建、KL 正则化与动作预测损失。

**📊 数据集**

使用 CALVIN 基准中的 8 个桌面操控任务，训练数据约 50 万条（6 小时遥控演示）。

**📈 对比分析**

与 DreamerV2 + DiWA 基线对比，WAM 在行为克隆阶段平均成功率从 45.8% 提升至 61.7%，PPO 微调后平均成功率从 79.8% 提升至 92.8%，且仅需 8.7 倍更少的世界模型训练步骤。

**⚠️ 局限性**

局限性包括仅在离线仿真环境验证，缺乏真实机器人实验；逆动力学损失权重需要手工调节，且对不同任务的泛化能力仍需进一步探索。

---

## 69. Generating Humanless Environment Walkthroughs from Egocentric Walking Tour Videos

**arXiv ID:** 2603.29036 | [PDF](https://arxiv.org/pdf/2603.29036v1)

**作者:** Yujin Ham `[一作]` (Rice University), Guha Balakrishnan `[通讯]` (Rice University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种用于去除第一人称行走视频中人类及其阴影的生成式模型。

**💡 创新点**

创新点在于构建了半合成的1,000对视频数据集，并通过阴影仿真提升真实感；以及在Casper视频扩散模型上进行微调，显著提升人类去除效果。

**🔧 技术方法**

使用Casper视频扩散模型，结合运动损失（Motion loss）与掩码指导，完成人类及其阴影的去除。

**📊 数据集**

使用自建的SemiSynth数据集：1,000对7秒长的真实行走视频构成的有无人类的成对视频。

**📈 对比分析**

与GenOmnimatte、ProPainter、DiffuEraser以及原始Casper比较，PSNR、LPIPS和DreamSim指标均优于基线，尤其在高Crowd%场景下表现突出。

**⚠️ 局限性**

局限：对训练分布外场景的泛化能力有限；长视频持续遮挡时时序一致性不足；在高Crowd%场景下细节重建仍有欠缺。

---

## 70. ARCS: Autoregressive Circuit Synthesis with Topology-Aware Graph Attention and Spec Conditioning

**arXiv ID:** 2603.29068 | [PDF](https://arxiv.org/pdf/2603.29068v1)

**作者:** Tushar Dhananjay Pathak `[一作]` `[通讯]` (New York University), Tushar Dhananjay Pathak (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 ARCS 系统，实现基于目标规范的模拟电路的即时生成，直接输出可用的 SPICE netlist。

**💡 创新点**

核心创新包括：① 对多拓扑 RL 的 Group Relative Policy Optimization (GRPO)，解决 REINFORCE 的跨拓扑奖励不匹配；② 基于语法的约束解码，保证 100% 结构有效性；③ 多源混合生成（VAE+Flow Matching）结合 SPICE 排名，极大提升仿真有效率。

**🔧 技术方法**

使用的技术包括：图 Transformer + 两头结构/数值头、离散 token 词表、流匹配模型 CCFM、强化学习 GRPO、语法状态机掩码、自动化数据生成与 SPICE 评估、学习型奖励模型。

**📊 数据集**

数据集：自动化生成 62,000 条电路（34 种拓扑），经增广后 205,000 条 token 序列，涵盖 32 种目标规范和拓扑。

**📈 对比分析**

与随机搜索和遗传算法比较：单候选 ARCS 的仿真有效率约 85%（97 ms），Best‑of‑3 提升至 95%；混合生成+SPICE 排名在仅 8 次仿真下达 99.9% 有效率、奖励 6.43，远快于 GA（7.56 奖励、约 320 次仿真）和随机搜索。

**⚠️ 局限性**

局限性：单次生成的设计质量仍低于搜索方法（约 5.5 vs. 7.5 奖励）；数值 token 分辨率受限，精度不足；仅覆盖 32 种拓扑，难以扩展到更细粒度的晶体管级电路；模型规模和训练数据有限，导致跨拓扑泛化受限。

---

## 71. Segmentation of Gray Matters and White Matters from Brain MRI data

**arXiv ID:** 2603.29171 | [PDF](https://arxiv.org/pdf/2603.29171v1)

**作者:** Chang Sun `[一作]` (Waseda University), Tetsuya Sakai `[通讯]` (Waseda University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文基于MedSAM改进的多类别分割模型，对T1加权MRI脑组织（灰质、白质与背景）进行2D切片分割。

**💡 创新点**

创新点在于将MedSAM的单类别掩膜解码器改为三通道输出，只冻结图像编码器并微调提示编码器与解码器，从而实现低改动的多类分割。

**🔧 技术方法**

使用的技术包括FSL BET做脑剥离、FSL FAST生成灰白质概率图、2D切片构造、多类别掩膜生成、基于ViT-B的MedSAM微调与交叉熵损失。

**📊 数据集**

使用IXI公共数据集的581个T1 MRI扫描（训练70%，验证15%，测试15%）进行实验。

**📈 对比分析**

通过Dice和IoU评估，冠状面单向模型获得最高Dice 0.8751、IoU 0.7935，表明微调后模型在多类别脑组织分割上可与传统U-Net相媲美，但统一多方向训练并未提升性能。

**⚠️ 局限性**

局限性包括仅使用FAST生成的伪真值标签、仅健康受试者的2D切片数据、缺乏病理样本以及对不同扫描协议与解剖变异的鲁棒性未验证。

---

## 72. Why That Robot? A Qualitative Analysis of Justification Strategies for Robot Color Selection Across Occupational Contexts

**arXiv ID:** 2603.28919 | [PDF](https://arxiv.org/pdf/2603.28919v1)

**作者:** Jiangen He `[一作]` (University of Tennessee), Jessica K. Barfield `[通讯]` (University of Kentucky)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对1,038名参与者在四个职业场景中提供的4,146条关于机器人肤色和外观的开放式理由进行定性与定量分析，探讨用户在不同情境、颜色与人性化程度下如何为机器人选择理由，揭示功能化与种族刻板印象之间的潜在联系。

**💡 创新点**

创新点在于：①构建多维编码框架，将功能主义、情感/心理、机器中心化、偏好/逃避与身份/社会映射等五大理由维度系统化；②结合人类与AI共识编码实现大规模开放式文本分析；③通过跨任务、颜色、种族、性别及人性化水平的交互效应，首次量化种族化与功能性辩护的隐性协同；④提出“人性化阈值”与“种族化风险”概念，为社会机器人设计提供可操作的伦理指南。

**🔧 技术方法**

技术手段包括：Google Gemini API 进行 AI 辅助编码（每条文本返回主类别与子类别 JSON），手工编码（三名研究者）用于验证，Cohen's κ 计算一致性；统计分析使用 Pearson χ² 与 Cramér's V，阈值检验与多重比较；混合方法结合定量频数与质性主题剖析。

**📊 数据集**

数据集为来自 Prolific 的 1,038 名美国受试者在两项实验（N=421 & N=617）中，分别在施工现场、医院、家教和运动场景下选择六种肤色或非肤色机器人（Light、Medium、Brown、Dark、Silver、Teal），并提交开放式理由，合计 4,146 条文本。

**📈 对比分析**

对比方法：按任务场景、颜色类型、种族/性别、人工拟人度分别绘制理由类别分布，并使用卡方检验评估显著性（p<.001），Cramér's V 表示效应大小（0.19–0.35）。AI 与人类编码的 κ 值在主类别上达到0.73、子类别0.69，表明 AI 编码可靠。研究显示功能主义主导但易被刻板印象掩盖，种族化提示明显提升相符选择率，随着人性化程度提升功能主义下降、机器中心化上升。

**⚠️ 局限性**

局限性包括：①样本仅来自美国，缺乏跨文化验证；②只覆盖四个职业场景，可能不具代表性；③AI 编码每条文本仅标注单一主类别，可能忽略多重理由；④自述理由可能受社会期望偏差影响；⑤未直接测量实际交互效果，仅通过理由推断偏见。

---

## 73. WorldFlow3D: Flowing Through 3D Distributions for Unbounded World Generation

**arXiv ID:** 2603.29089 | [PDF](https://arxiv.org/pdf/2603.29089v1)

**作者:** Amogh Joshi `[一作]` (Princeton University), Felix Heide `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于流匹配的分层无潜变量3D世界生成方法，能够在任意尺度上生成无限大、结构和纹理高度逼真的室内外场景，并支持基于布局和属性的可控生成。

**💡 创新点**

核心创新在于将3D生成视为在多层数据分布之间进行流匹配（flow matching）而非传统的条件去噪扩散；采用无潜变量体素生成（直接在体素空间学习），并通过分块平均流场实现无限扩展，显著提升训练速度和生成质量。

**🔧 技术方法**

使用连续正则化流（CNF）+ 条件流匹配（CFM）框架；3D体素表示采用截断无符号距离场（TUDF）+ 颜色通道；生成器为3D U‑Net，支持FiLM条件化；在推理时实现块级流场平均（chunk-aware velocity averaging）以消除边缘伪影。

**📊 数据集**

在两个公开数据集上评估：Waymo Open Dataset（真实室外驾驶场景）和3D‑Front（合成室内房间）。

**📈 对比分析**

与XCube、LidarDM、BlockFusion、WorldGrow、LT3SD等近期基线比较，使用覆盖率、MMD、1‑NNA、JSD、FD_Concerto等五个指标；结果显示该方法在所有指标上均优于基线，尤其在几何多样性和纹理逼真度上显著提升；训练时间比传统潜变量方法快至少2倍，推理效率高且可理论上无限扩展。

**⚠️ 局限性**

局限性包括：目前仅针对静态场景，无法处理动画或动态变化；生成体素分辨率受GPU内存限制，超大体素仍需多次块化；在极大尺度下块间同步仍需大量CPU‑GPU通信，实际推理速度受制于硬件；模型对非常稀疏或极端复杂场景的适应性尚未充分验证。

---

## 74. From Natural Alignment to Conditional Controllability in Multimodal Dialogue

**arXiv ID:** 2603.29162 | [PDF](https://arxiv.org/pdf/2603.29162v1)

**作者:** Zeyu Jin `[一作]` (Tsinghua University), Jia Jia `[通讯]` (Tsinghua University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对电影和电视内容的自动化提取与细粒度注释，构建了360.26小时、54,700条对话的多模态对话数据集，并提出了双说话人可控多模态对话生成（MDG）任务，包括显式风格控制的语音合成、基于视觉的隐式控制语音合成以及语音驱动的视频生成。

**💡 创新点**

创新点在于：①首次以对话表达性为核心的多模态数据集，结合情感三元组和自由描述两种注释范式；②采用VLM+LLM动态关键帧的对话边界与多模态语义抽取管线；③建立系统化的跨模态可控生成任务与评测基准；④引入Gemini-as-Judge和多维度人类MOS等多层次评价指标。

**🔧 技术方法**

使用技术包括：Vision‑Language Model（VLM）和Large Language Model（LLM）用于场景和对话分割；Gemini‑2.5（flash/pro）进行说话人归属与情感/表达注释；Insightface提取说话人可见度；预训练音频生成模型Higgs‑Audio‑V2和Dia‑1.6B，并通过轻量级Adapter实现风格条件；以及HarmoniVox、Cascaded Gemini+Higgs等基线。

**📊 数据集**

数据集方面：自研的电影/电视剧多模态对话集（360.26小时、54,700条），以及包含309条双说话人可见对话的评测子集；对比现有数据集如OpenDialogue、OpenViDial、YTD‑18M、OpenVid‑1M、MELD、MC‑EIU、MovieBench。

**📈 对比分析**

通过与基线模型（Higgs‑Audio‑V2、Dia‑1.6B、HarmoniVox、Cascaded Gemini+Higgs、SI2V、T2V）比较，发现Fine‑Tuning在本数据集上显著降低WER（31.3→4.5）、提升cp‑WER，显式风格控制效果好；隐式视觉控制虽保持语音质量，但跨模态一致性明显下降；视频生成基线FVD高、标签召回率低，说明现有模型尚无法满足完整的跨模态可控对话生成。

**⚠️ 局限性**

局限性包括：①隐式跨模态风格一致性仍不足；②多说话人、复杂场景下的实时同步与表达捕捉受限；③视频生成缺乏端到端的关键帧规划与口型同步；④数据集主要来自电影/电视剧，可能在语境多样性与非正式语境上受限。

---

## 75. SimMOF: AI agent for Automated MOF Simulations

**arXiv ID:** 2603.29152 | [PDF](https://arxiv.org/pdf/2603.29152v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 76. Scheduling with Time Dependent Utilities: Fairness and Efficiency

**arXiv ID:** 2603.28800 | [PDF](https://arxiv.org/pdf/2603.28800v1)

**作者:** Gaia Nicosia `[一作]` (University of Roma Tre), Ulrich Pferschy `[通讯]` (University of Graz)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种新的多代理单机排程模型：每个作业对应一个自利代理，代理的效用随完成时间下降，目标是最大化最小效用（即公平排程），并在多种约束（释放日期、到期日期、处理时间相同、可调整效用等）下分析其复杂度并给出求解方法。

**💡 创新点**

创新点在于：①首次将最大最小效用作为多代理排程的公平性衡量；②提出统一的二分搜索框架与高效贪心算法，可在多种约束下求解最优公平排程；③考虑效用可在预算内调整（截距或斜率），并研究其对公平与效率的影响；④提出新的再排程与领导者-追随者双层优化模型，用预算补偿或修改效用来实现目标序列或公平目标。

**🔧 技术方法**

使用的技术包括：二分搜索将最大最小效用转化为多约束排程（如最晚完成时间）；贪心法类似 Lawler 算法；动态规划处理单一释放日期、等处理时间的弱 NP 难子问题；使用 Moore、Carlier 等经典排程算法；多源 NP 难度归约（Partition、3-Partition、1|r_j|∑C_j 等）。

**📊 数据集**

本文为理论工作，未使用具体实验数据集；所有结果均基于抽象实例和已知的 NP 难度归约。

**📈 对比分析**

与已知复杂度结果对比：对大多数变体给出了多项式或伪多项式算法，说明在公平目标下仍可高效求解；对于存在释放/到期日期的通用情况则证明强 NP 难度；对于可调整效用的预算问题则给出多项式/伪多项式算法；总体而言，公平排程在许多实用约束下可在多项式时间内求解，且提供了精确复杂度分类。

**⚠️ 局限性**

局限性：①仅考虑单机排程，未扩展到多机或并行模型；②公平目标仅为最大最小效用，未探讨其他公平指标；③对效用函数的可调整性仅限于线性或区间约束，其他非线性调整尚未研究；④缺乏实验验证，未评估算法在实际规模下的性能；⑤在双层优化中，领导者仅能通过截距/斜率调整，无法满足所有目标序列。

---

## 77. Legible Consensus: Topology-Aware Quorum Geometry for Asymmetric Networks

**arXiv ID:** 2603.28788 | [PDF](https://arxiv.org/pdf/2603.28788v1)

**作者:** Tony Mason `[一作]` `[通讯]` (University of British Columbia), Tony Mason (University of British Columbia)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种将崩溃墙（crumbling-wall）结构映射到物理多层网络的仲裁方案，以分离跨层义务与层内复制，并使故障模式可读。

**💡 创新点**

创新点在于：①将 Flexible Paxos 的交叉相交要求与崩溃墙的层级几何相结合，形成“层级仲裁”；②引入“可读性（legibility）”概念，运算复杂度仅为 O(层数)；③通过实验验证在极端异步（行星间）拓扑下，部分层能在全局黑洞期间保持线性可达性。

**🔧 技术方法**

使用了 Flexible Paxos、崩溃墙仲裁构造、TLA+ 形式化验证、Eidolon 分布式事件模拟器、SimPy 网络延迟模型，以及多层时延（光速）计算。

**📊 数据集**

数据集为一个 10 节点的 Earth/LEO/Moon/Mars 拓扑（5/1/1/3 配置），并对不同 Mars 延迟（186–1342 s）、黑洞持续时间、LEO/Mars 连接稀疏与完整覆盖两种网络做参数扫描。

**📈 对比分析**

对比方法：将“平面仲裁”与“崩溃墙仲裁”在相同拓扑和黑洞条件下运行；在稀疏网络中对比 LEO 的可达性；对 Phase‑2 约束的放宽进行容错度评估。结果显示：平面仲裁在任何一次黑洞期间全失效；墙仲裁在 3/4 层成功率为 100%；当 Phase‑2 放宽至 4/5 时，容错度提升至 98%，而平面仲裁仍 0%。

**⚠️ 局限性**

局限性：仅为设计级实验，未考虑轨道运动、天线调度、链路丢包和信号噪声；仅测试单一 5/1/1/3 拓扑；假设所有节点为 crash‑stop 非拜占庭；未探讨动态重配置、层内一致性协议或多写入租约等实际系统细节。

---

## 78. SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving

**arXiv ID:** 2603.29163 | [PDF](https://arxiv.org/pdf/2603.29163v1)

**作者:** Wenchao Sun `[一作]` (Tsinghua University), Sifa Zheng `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种稠密轨迹词汇表和可扩展评分策略的端到端自主驾驶规划框架SparseDriveV2

**💡 创新点**

1) 通过轨迹分解为几何路径与速度剖面实现词汇表的因子化和组合式扩展；2) 采用粗糙的路径/速度级评分与细粒度轨迹级评分相结合的分层评分机制，使得超密词汇表仍能高效计算；3) 通过可扩展的评分框架实现32倍密度的词汇表而不增加显著计算负担

**🔧 技术方法**

因子化词汇表表示、分层评分（粗路径/速度评分+细轨迹评分）、多头交叉注意力或可变形聚合、ResNet-34视觉编码器、Softmax距离监督、规则基准教师监督

**📊 数据集**

NAVSIM v1/v2 公开数据集与 Bench2Drive（CARLA 基准）

**📈 对比分析**

在 NAVSIM v1 以 ResNet-34 基座实现 92.0 PDMS；在 NAVSIM v2 以 90.1 EPDMS 超过 DiffusionDriveV2 2.6；在 Bench2Drive 获得 89.15 Driving Score 与 70% Success Rate，均优于之前的评分及动态生成方法

**⚠️ 局限性**

对极大词汇表的实验仍受硬件内存与推理时延限制；在极端复杂场景下仍需进一步验证泛化性；对动态生成方法的可解释性与多模态性能尚未深入对比

---

## 79. Sampling-Horizon Neural Operator Predictors for Nonlinear Control under Delayed Inputs

**arXiv ID:** 2603.29119 | [PDF](https://arxiv.org/pdf/2603.29119v1)

**作者:** Luke Bhan `[一作]` (University of California San Diego), Yuanyuan Shi `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了两种基于神经算子的预测反馈控制器，用于处理非线性系统的输入延迟和离散测量。

**💡 创新点**

创新点在于将神经算子直接逼近多步预测器，既支持均匀采样也支持带上界的非均匀采样，并给出残差与逼近误差的半全局实践稳定性理论。

**🔧 技术方法**

采用 Fourier Neural Operator 对预测器与采样-流算子进行逼近，并通过 Lyapunov 分析证明闭环系统的稳定性。

**📊 数据集**

训练数据来自 6 连杆机械臂（xArm6）的 20k 条带噪声的数值轨迹，使用标准数值预测器生成。

**📈 对比分析**

与传统数值预测器比较，误差均值 L2 为 10⁻⁴，跟踪误差相当，计算时间从 25 ms 降至 1 ms（约 25 倍速度提升）。

**⚠️ 局限性**

局限在于逼近误差必须足够小，非均匀采样时误差会放大；此外需要足够的训练数据和可建模的系统，适用范围受限。

---

## 80. An Explicit Surrogate for Gaussian Mixture Flow Matching with Wasserstein Gap Bounds

**arXiv ID:** 2603.28992 | [PDF](https://arxiv.org/pdf/2603.28992v1)

**作者:** Elham Rostami `[一作]` (Université Paris-Saclay), Hamidou Tembine `[通讯]` (University of Quebec in Trois-Rivieres)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出一种训练‑free 的流匹配方法，用基于连续性方程的仿射高斯速度场构造了在两个高斯混合模型（GMM）之间的时间‑依赖密度和速度场，并给出了对应的闭式代理成本 C。

**💡 创新点**

创新点包括：① 通过线性插值均值和协方差得到的闭式速度场和代理成本；② 在局部可交换条件下证明代理成本与真实 2‑Wasserstein 成本二阶相符；③ 推导了明确的立方阶误差上界；④ 引入路径拆分策略，使得非局部情形也能逐段落应用误差分析；⑤ 提供了实用的 regime map，指明何时使用代理方法、何时需要精确 Wasserstein 方法。

**🔧 技术方法**

采用的技术：连续性方程求解仿射速度场、闭式动力学能量计算、Sinkhorn 算法求解熵正则化的配对问题、矩阵平方根和 Bures‑Wasserstein 几何、误差分析与路径拆分。

**📊 数据集**

实验使用合成的多种高斯场景（包括可交换、非可交换、近边界、Toeplitz、因子模型、Wishart 等），无真实数据集。

**📈 对比分析**

方法比较：代理成本 C 与精确的 2‑Wasserstein 成本 W₂² 在不同维度和情形下进行数值比较，误差在可交换或局部可交换区间内小；在高维（d>200）时，代理方法的构造时间显著低于精确方法，且在混合层面上的成本差异保持在可接受范围。

**⚠️ 局限性**

局限性：① 当协方差差异大或协方差矩阵条件数很差时，代理成本误差增大；② 误差上界在严格的可交换假设下成立，非可交换情形缺乏相同的二阶保证；③ 路径拆分虽然能恢复局部性，但在极端非可交换或高维下仍可能需要细分到不可行的程度；④ 仅针对 GMM，无法直接推广到非高斯分布；⑤ 对熵正则化参数 ε_OT 的影响尚未系统探讨。

---

## 81. Efficient Bilevel Optimization with KFAC-Based Hypergradients

**arXiv ID:** 2603.29108 | [PDF](https://arxiv.org/pdf/2603.29108v1)

**作者:** Disen Liao `[一作]` (University of Waterloo Vector Institute), Yaoliang Yu `[通讯]` (University of Waterloo Vector Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种基于KFAC的双重优化（BO）算法，通过将昂贵的逆Hessian向量积（IHVP）替换为Kronecker-factored近似的逆向量积（IKVP），实现了更高效、更稳定的超梯度计算。

**💡 创新点**

创新点在于将KFAC结构化的曲率近似直接嵌入IFT式BO中，既保留了二阶信息，又显著降低了计算和存储成本，且在大规模模型（如BERT、ResNet-18）和多种任务上均表现优异。

**🔧 技术方法**

主要技术包括隐函数定理、KFAC（块对角Kronecker近似）、张量运算与逆向量积、Tikhonov阻尼、经验KFAC（KFAC EMP）以及批量动态更新等。

**📊 数据集**

实验数据集涵盖图像分类的CIFAR-10/100（长尾版本）、ResNet-18、BERT在WRENCH文本分类数据、Meta-Weight-Net用于不平衡学习、ChemProt/ACL-ARC/SciERC等医学与社交文本任务，以及MNIST/CIFAR-10上的数据污染与不可学习示例。

**📈 对比分析**

与传统IHVP求解器（CG、Neumann）、一次性梯度展开（SAMA）、以及其他一阶或二阶方法（BOME、stocBiO、AID-CG等）相比，KFAC在相同迭代/时间预算下实现更低的验证/测试误差，速度提升数倍，且在大模型下仍保持低内存占用。

**⚠️ 局限性**

局限性包括：KFAC仅为近似，极端高维或严重欠条件下仍可能产生误差；对批量噪声敏感，需要适当的阻尼和EMA；在某些AI安全攻击场景中，纯梯度优化仍可超越KFAC；以及实现上仍需额外计算KFAC矩阵，导致在极大模型上的额外负担。

---

## 82. Spark-LLM-Eval: A Distributed Framework for Statistically Rigorous Large Language Model Evaluation

**arXiv ID:** 2603.28769 | [PDF](https://arxiv.org/pdf/2603.28769v1)

**作者:** Subhadip Mitra `[一作]` `[通讯]`, Subhadip Mitra

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Spark-LLM-Eval，一个基于 Apache Spark 的分布式 LLM 评估框架。

**💡 创新点**

创新点包括通过 Spark 实现线性可扩展的并行评估、Delta Lake 响应缓存实现无 API 调用的指标迭代、以及完整的统计置信区间和显著性检验集成。

**🔧 技术方法**

使用了 Apache Spark、Delta Lake、Pandas UDF、token bucket 速率限制、Bootstrap CI、配对 t 检验/McNemar/Wilcoxon 等统计方法，并支持多 LLM 提供商。

**📊 数据集**

在合成的多域数据集（包含事实 QA、摘要、指令跟随等）以及规模从 1,000 到 100,000 条样本的实验中评估。

**📈 对比分析**

通过与单线程基线比较，8 个 executor 时吞吐率可达 9,800 条/分钟，性能提升 21 倍；缓存模式下成本下降 75%，统计方法覆盖广且 CI 可靠。

**⚠️ 局限性**

局限性包括对全局速率限制的静态分配导致负载不均、精确匹配缓存缺乏语义等价性、LLM-as-judge 可能存在偏差、以及统计假设独立性未考虑相关样本。

---

## 83. Robust Multi-Agent Reinforcement Learning for Small UAS Separation Assurance under GPS Degradation and Spoofing

**arXiv ID:** 2603.28900 | [PDF](https://arxiv.org/pdf/2603.28900v1)

**作者:** Alex Zongo `[一作]` (George Washington University), Peng Wei `[通讯]` (George Washington University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

提出一种鲁棒多智能体强化学习框架，用以在小型无人机系统（sUAS）面临GPS失真或欺骗时保证空中安全分离。

**💡 创新点**

创新点包括：① 给出全状态观测腐败下的最优对抗扰动的闭式近似，并证明其二阶误差；② 通过KL正则化将对抗扰动与策略不变性和教师锚定联系起来，提供性能下界；③ 将闭式对抗扰动嵌入PPO算法，实现无梯度对抗训练。

**🔧 技术方法**

技术手段：多智能体强化学习（MARL）、PPO、近似对抗扰动（一次展开）、Kullback–Leibler 正则化、共享演员-评论家网络、注意力编码、梯度基准与安全奖励。

**📊 数据集**

使用 BlueSky 仿真平台构建的城市低空多路线小型无人机交通场景；通过泊松到达模型和 Amazon MK30 车辆动力学生成训练和评估数据。

**📈 对比分析**

与仅在干净观测上训练的基线策略对比，评估指标为近空碰撞（NMAC）次数和最小安全间隔距离。实验显示，鲁棒策略在观测腐败率高达35%时仍保持近零 NMAC，且在更高腐败率下性能衰减更缓，平均 NMAC 仅为 5 次，而基线在同一条件下约为 18 次。

**⚠️ 局限性**

局限性包括：① 仅在离散速度调整动作空间下验证；② 对抗扰动的闭式近似假设值函数可微且扰动小，可能在大幅失真情况下失效；③ 仅在仿真环境中评估，缺乏真实硬件实验；④ 训练过程仍需多阶段和参数调优，实际部署复杂度高。

---

## 84. Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference

**arXiv ID:** 2603.29002 | [PDF](https://arxiv.org/pdf/2603.29002v1)

**作者:** Zifan He `[一作]` (University of California), Jason Cong `[通讯]` (University of California)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了基于GPU‑FPGA异构系统的LLM推理内存处理管线加速方案，提出统一的四步内存处理流程并在多种长上下文优化上进行评估；

**💡 创新点**

创新点在于：①将所有LLM长上下文优化归纳为Prepare‑Memory→Compute‑Relevancy→Retrieval→Apply‑to‑Inference四步统一框架并量化其对延迟的贡献；②揭示该管线的计算异质性，分别定位可落到GPU或FPGA的子任务；③设计并实现GPU‑FPGA映射，利用FPGA大容量高带宽内存与流水线数据流实现显著加速与能耗降低；④构建可复用核库与设计自动化思路，为未来异构ASIC提供参考；

**🔧 技术方法**

使用GPU+FPGA异构平台，FPGA流式数据流内核（Compute‑Relevancy、Retrieval、BM25、Top‑k等）与GPU上的优化CUDA/HIP核；采用Vitis HLS、ROCｍ、PyTorch Profiler等工具；集成Sparse Attention、RAG、MemAgent、Memory‑as‑Context等LLM长上下文优化；

**📊 数据集**

主要使用公开模型与数据集：LLaMA、Qwen、Qwen‑2.5 7B 等模型；RAG评估使用Wiki Dump与20M文档集合；长上下文实验使用1M token长序列；MemAgent使用Qwen‑2.5 7B；其它实验均基于相应模型与公开数据；

**📈 对比分析**

通过与单GPU基线（同型号GPU）对比，记录端到端延迟与能耗；在AMD MI210 + Alveo U55C平台上，Sparse Attention 1.04–1.49×速度提升、1.11–1.61×能耗下降；RAG 1.07–1.21×能耗、1.23–1.84×速度提升；MemAgent 4.66×能耗下降、1.8×速度提升；不同方法在不同token长度与batch大小下表现变化；

**⚠️ 局限性**

局限性包括：①跨设备PCIe通信开销在高token序列或大批量时成为瓶颈；②FPGA HBM访问在>1M token时性能下降；③某些密集计算方法（如MemAgent）在大batch下性能下降；④实现基于现有FPGA与GPU，缺乏完整自动化设计流水线；⑤未来需要更高性能FPGA或专用ASIC来进一步提升。

---

## 85. Attesting LLM Pipelines: Enforcing Verifiable Training and Release Claims

**arXiv ID:** 2603.28988 | [PDF](https://arxiv.org/pdf/2603.28988v1)

**作者:** Zhuoran Tan `[一作]` (University of Glasgow), Christos Anagnostopoulos `[通讯]` (University of Glasgow)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了一个可验证的LLM流水线推广门，能够在训练、微调和部署前对第三方模型、数据集和依赖进行安全审计。

**💡 创新点**

创新点在于将训练和发布声明与加密签名绑定，并将声明映射到可执行的安全策略，支持可插拔的动态运行时证据。

**🔧 技术方法**

采用了 in‑toto、Sigstore、CycloneDX MLBOM 等签名与声明框架，并结合静态扫描、格式安全检查和可选的运行时插件实现门控。

**📊 数据集**

通过构造依赖混淆、恶意模型仓库（MALHUG 语料）和后门模型等代表性案例进行评估，未使用公开数据集而是自建攻击样本。

**📈 对比分析**

评估通过覆盖率、吞吐量和人工干预成本等指标，实验表明门控能够覆盖约 70–85% 的已知攻击场景，且平均延迟保持在几百毫秒范围。

**⚠️ 局限性**

局限性包括对自定义后门行为的检测仍有限、依赖生态的签名普及度不高、以及缺乏针对复杂攻击者的完整防御。

---

## 86. Toward a Universal GPU Instruction Set Architecture: A Cross-Vendor Analysis of Hardware-Invariant Computational Primitives in Parallel Processors

**arXiv ID:** 2603.28793 | [PDF](https://arxiv.org/pdf/2603.28793v1)

**作者:** Ojima Abraham `[一作]` (Franklin & Marshall), Onyinye Okoli `[通讯]` (Cornell University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 NVIDIA、AMD、Intel、Apple 四大 GPU 供应商的 ISA 进行系统性跨厂商分析，识别出十个硬件不变的计算原语，并基于物理约束提出通用 GPU ISA 抽象模型；

**💡 创新点**

首次从 ISA 级别完成四大供应商的对比，发现硬件不变的原语并构建可被任何厂商实现的通用抽象 ISA，验证该抽象模型能在不同硬件上保持 80% 以上的性能；

**🔧 技术方法**

利用 5,000+ 页原始文档（官方手册、白皮书、专利、逆向工程资料）进行维度拆解，提出抽象执行模型，并用 CUDA/Metal 手写核与抽象模型对比；

**📊 数据集**

基准包含 3 种 kernel：GEMM（N=4096 FP32）、Reduction（N=2^24）和 Histogram（N=2^24、256 计数器），在 NVIDIA T4 与 Apple M1 两个平台上测评；

**📈 对比分析**

比较方法为三种实现：Native（使用厂商特定优化）、Abstract（仅使用通用原语）以及 Library（cuBLAS）；结果显示：NVIDIA GEMM 抽象略高于 Native，Apple GEMM 与 Native 相当；Reduction 抽象在 NVIDIA 仅达 62.5%（缺失 intra‑wave shuffle），在 Apple 接近 98%；Histogram 在两平台均与 Native 持平或略优；整体可行性超过 80%；

**⚠️ 局限性**

局限性包括仅在 NVIDIA 与 Apple 上验证，AMD/Intel 结果待补充；基线实现为手写 kernel，未达到生产级库的优化；工作负载仅覆盖 3 种常见 kernel，未测试稀疏/不规则或复杂模型；Apple GPU 参数来自逆向工程，可信度有限。

---

## 87. LoRaWAN Gateway Placement for Network Planning Using Ray Tracing-based Channel Models

**arXiv ID:** 2603.29105 | [PDF](https://arxiv.org/pdf/2603.29105v1)

**作者:** Cláudio Modesto `[一作]`, Aldebaro Klautau `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一个集成雷达跟踪（RT）工具与 ns‑3 的 LoRaWAN 网络规划框架，用于评估不同信道模型对 GW（网关）放置优化的影响。

**💡 创新点**

首次在 LoRaWAN GW 放置问题中引入了基于 RT 的场景特定信道模型，并系统比较了多种模型（站点独立与站点特定）对优化结果、计算成本及性能指标的影响。

**🔧 技术方法**

技术包括：Sionna RT 与 Wireless InSite（WI）雷达跟踪模拟、ns‑3 LoRaWAN 模块、Pyomo + GLPK 求解 BILP、Blender 转换 3D 场景、Mitsuba / DAE 规范文件。

**📊 数据集**

使用公开场景 Etoile（13 058 面）生成 3D 环境；生成的路径增益/损耗数据作为优化输入；不依赖专有实验数据。

**📈 对比分析**

通过对比不同信道模型下 GW 数量、平均接收功率、包丢失率（PDR）以及物理层模拟耗时，展示了：站点特定模型往往需要更少的 GW 但 PDR 更低，且计算成本高；站点独立模型 PDR 较高但可能导致显著过度部署。

**⚠️ 局限性**

局限性包括：仅考虑单一网络拓扑（100 个潜在位置，54 个终端）；不考虑 GW 互相干扰；RT 模拟仅在单一场景上测试；能源模型与 SF 变动未纳入优化；对更大规模/多城市场景的可扩展性未知。

---

## 88. WybeCoder: Verified Imperative Code Generation

**arXiv ID:** 2603.29088 | [PDF](https://arxiv.org/pdf/2603.29088v1)

**作者:** Fabian Gloeckle `[一作]` (Meta), Peter O'Hearn `[通讯]` (Meta)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 WybeCoder 这一 agentic 代码验证框架，实现代码、循环不变量与证明协同生成与迭代；

**💡 创新点**

创新点在于将自动化与交互式验证融合，采用子目标分解与目标导向的实现修改，翻译功能性验证基准为命令式实现，并通过约束规范与功能泄漏防护提升可信度；

**🔧 技术方法**

使用大语言模型（GPT‑5、Gemini 3 Pro、Claude 4.5 Opus）配合 Lean+Loom/Velvet 进行自动化验证条件生成，利用 CVC5 解决子目标，整体实现多智能体并行与迭代反馈；

**📊 数据集**

采用 Verina 与 Clever 这两套功能性 Lean 验证基准，并通过自定义映射生成对应的 Imperative‑Loom 版本（Verina‑advanced、Clever‑Loom）；

**📈 对比分析**

与基线 COPRA、单智能体顺序生成以及多智能体子目标分解三种方法在相同计算预算下对比；在 Verina 上达 74% 解决率、Clever‑Loom 上 62% 解决率，明显优于之前的 10%–30% 级别结果；对计算规模的分析显示无性能平台期，单智能体在低预算下更高效；

**⚠️ 局限性**

局限包括：依赖实验性 Loom/Velvet 框架；仅针对 Lean 的托管内存，无法直接应用于 C++/Rust 等无管理语言；规格工程与子模块拆分仍需人工；公开源模型表现不佳，限制可复现性；

---

## 89. Quality-Controlled Active Learning via Gaussian Processes for Robust Structure-Property Learning in Autonomous Microscopy

**arXiv ID:** 2603.29135 | [PDF](https://arxiv.org/pdf/2603.29135v1)

**作者:** Jawad Chowdhury `[一作]` (Oak Ridge National Laboratory), Rama Vasudevan `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种结合高斯过程质量控制的主动学习框架（ActiveQC），在自驱实验平台上对结构–性质翻译任务进行高质量数据主动采样。

**💡 创新点**

创新点在于将物理驱动的质量评估（基于简谐振子SHO拟合的R²）与主动学习的好奇驱动采样相结合，形成门控机制，有效剔除低质量噪声数据，提升模型鲁棒性。

**🔧 技术方法**

使用的核心技术包括：Gaussian Process 回归做质量预测、SHO 拟合提取R²质量指标、基于编码-解码网络的 Im2Spec/Spec2Im 结构-性质映射、基于 surrogate error 模型的好奇驱动采样，以及多任务学习辅助。

**📊 数据集**

使用了 PbTiO₃ 薄膜的配对 AFM 图像与 BEPS 光谱数据（1225 个样本），在实验中人为加入空间局部噪声；另外在 BiFeO₃ 实时 AFM 试验中验证框架。

**📈 对比分析**

与随机采样、传统主动学习（Active）和多任务学习（ActiveMT）对比，ActiveQC 在 Im2Spec、Spec2Im 两个方向上均实现了显著的 MSE 降低（平均 10–30%），并在多次随机试验中通过 Welch t 检验证明差异显著。

**⚠️ 局限性**

主要局限包括：门控阈值需人工设定或经验决定，对不同实验系统的适用性需进一步验证；SHO 拟合假设在某些材料或测量模式下可能不适用，导致质量评估失效。

---

## 90. VueBuds: Visual Intelligence with Wireless Earbuds

**arXiv ID:** 2603.29095 | [PDF](https://arxiv.org/pdf/2603.29095v1)

**作者:** Maruchi Kim `[一作]` (University of Washington), Shyamnath Gollakota `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在无线耳机中集成低功耗摄像头，实现面向用户的实时视觉问答系统

**💡 创新点**

首次在耳机上完成双目摄像头+Vision Language Model的完整端到端方案，兼顾尺寸、功耗与实时性

**🔧 技术方法**

硬件改造：低功耗CMOS摄像头+BLE SoC；软件：BLE传输、图像拼接、Qwen2.5‑VL 7B模型、TinyWhisper ASR、TTS

**📊 数据集**

使用自制场景图像（约130个真实场景）与公开VLM基准（Qwen、Moondream等）进行对比

**📈 对比分析**

与Ray‑Ban Meta智能眼镜在17项视觉问答任务的MOS相当（3.33 vs 3.32），在物体识别82.5%、OCR94.3%、翻译83.8%；端到端时延≈1.14 s，电池额外消耗≤14%

**⚠️ 局限性**

低分辨率单色图像限制文本/细节识别；双耳角度导致盲区、视角差异大；拼接失败率高；缺乏注视/手势解歧能力；隐私与用户可见性问题

---

## 91. UltraG-Ray: Physics-Based Gaussian Ray Casting for Novel Ultrasound View Synthesis

**arXiv ID:** 2603.29022 | [PDF](https://arxiv.org/pdf/2603.29022v1)

**作者:** Felix Duelmer `[一作]` (Technical University of Munich), Mohammad Farid Azampour `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为 UltraG-Ray 的基于物理的超声视角合成方法，利用可学习的 3D 高斯场和射线投射实现视角依赖的 B‑mode 图像合成。

**💡 创新点**

创新点在于将衰减、反射和视角相关的超声物理效应直接嵌入高斯表示中，并通过高效的射线投射和离散视角采样捕获阴影与回波强度，实现更真实的超声合成。

**🔧 技术方法**

使用 3D Gaussian splatting、射线投射、球谐展开、CUDA 加速的 gsplat 库以及 Adam 优化器进行训练，并结合软覆盖和梯度重要性自适应的高斯细化策略。

**📊 数据集**

采用了两个姿态标注的 B‑mode 数据集：一个基于计算机生成的脊柱模型（in‑silico），一个基于猪肌肉的 ex‑vivo 体外模型，均包含多视角扫描。

**📈 对比分析**

与 Ultra‑NeRF、ImplicitVol、最大/中值体素复合等基线对比，UltraG‑Ray 在 PSNR、MS‑SSIM、GMS 等指标上均优于对手，MS‑SSIM 提升约 15%，并在视觉上显著减少阴影失真和细节模糊。

**⚠️ 局限性**

局限性包括仅使用单射线近似，无法模拟多路径干涉；显式高斯场缺乏连续性，对采样稀疏敏感；在视角外推时性能下降，且对扫描密度和重叠度要求较高。

---

## 92. Towards Supporting Quality Architecture Evaluation with LLM Tools

**arXiv ID:** 2603.28914 | [PDF](https://arxiv.org/pdf/2603.28914v1)

**作者:** Rafael Capilla `[一作]` (Rey Juan Carlos University), Vanessa Rodríguez-Horcajo `[通讯]` (Polytechnic University of Madrid)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对软件架构质量评估（ATAM）过程进行实验，比较学生、专家和大型语言模型（LLM）在识别风险、敏感点、权衡及场景选择上的表现。

**💡 创新点**

首次将LLM与检索增强生成（RAG）策略结合用于ATAM评估，并提供专家制定的基准（ground truth）验证其效果；展示LLM在风险、敏感点和权衡识别上比人类更详细、准确，并能显著加速评估流程。

**🔧 技术方法**

使用大型语言模型（如Microsoft Copilot/ChatGPT-4o）、检索增强生成（RAG）技术、ATAM方法论、Likert量表评估、少量示例提示等。

**📊 数据集**

基准数据为9个软件架构课程学生完成的ATAM文档、两名专家生成并复核的ground truth场景、风险、敏感点和权衡数据；公开数据集链接已提供。

**📈 对比分析**

通过定性对比、定量Likert评分及手工匹配分析对比学生、专家与LLM的结果；LLM在风险/敏感点/权衡识别数量上最高，识别质量相近或更好，评估时间显著降低；但LLM生成过多非关键风险，需要人工筛选。

**⚠️ 局限性**

LLM易产生hallucination，需提供充分上下文和示例；实验仅在教育环境，未验证工业场景；缺乏统计显著性分析；未评估使用者感知与满意度；对低经验评估者的帮助有限。

---

## 93. Ray Tracing Cores for General-Purpose Computing: A Literature Review

**arXiv ID:** 2603.28771 | [PDF](https://arxiv.org/pdf/2603.28771v1)

**作者:** Enzo Meneses `[一作]` (Universidad Austral de Chile), Cristian Salazar-Concha `[通讯]` (Universidad Austral de Chile)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对59篇文献的文献计量与35篇系统评估的综合分析，总结了RT核心在非图形计算中的应用范围与性能提升。

**💡 创新点**

创新点在于首次将RT核心的使用划分为多类问题（邻近查询、近似搜索、仿真等），并量化各类问题在最佳与最差情形下的加速比，揭示了RT核心优势与局限的系统性模式。

**🔧 技术方法**

使用了文献计量工具biblioshiny、系统评估方法、统计检验（Kruskal-Wallis、能量距离检验）以及多维度缩减（MCA）和聚类分析来识别问题特征与加速关系。

**📊 数据集**

采用了各原始研究中使用的公开基准数据集与合成实验数据，涵盖了几何、索引、仿真与近似搜索等不同领域。

**📈 对比分析**

通过与现有GPU实现（不使用RT核心）的对比实验，测得RT核心方案在最优场景下可达200×加速，平均加速幅度在1–20×之间，且在邻近查询等高度可裁剪的任务中始终保持优势。

**⚠️ 局限性**

主要局限包括RT核心模型的硬件固定性（缺乏对BVH内部节点的访问）、三维空间映射导致的高内存占用与精度限制、线程发散与上下文切换开销，以及对高维问题映射的研究不足。

---

## 94. Bootstrap Perception Under Hardware Depth Failure for Indoor Robot Navigation

**arXiv ID:** 2603.28890 | [PDF](https://arxiv.org/pdf/2603.28890v1)

**作者:** Nishant Pushparaju `[一作]` (New York University), Aliasghar Arab `[通讯]` (New York University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种自标定的bootstrap感知系统，在 ToF 相机失效时利用其余有效像素校准单目深度，结合 2D LiDAR 与可选语义信息，在成本地图中实现缺失深度的自填补。

**💡 创新点**

创新点包括：① 失效深度传感器自身有效像素实时标定学习深度至度量尺度；② 构建失败感知层次化架构，只有当硬件失效才插入学习深度；③ 将该架构压缩为可在 Jetson Orin Nano 上以 218 FPS 运行的学生模型。

**🔧 技术方法**

采用时域深度融合、自标定比例因子、EfficientViT+SAM+YOLOv8 视觉分割、知识蒸馏、动态语义阈值触发、Nav2 成本地图融合等技术。

**📊 数据集**

使用三大数据集：Corridor（含高失效率反射走廊录像）、LILocBench（公共行人基准）、NYU Depth V2（通用训练）。

**📈 对比分析**

与仅 LiDAR、仅硬件深度、仅学习深度等配置进行对比；在 Corridor 中占据格子数提升 55%，在 LILocBench 提升 110%；学生模型在 Gazebo 仿真中 9/10 成功率、无碰撞，与真值深度几乎相同；但学生在近距离（0.3–1 m）精度远低于基础模型。

**⚠️ 局限性**

主要局限包括：学生模型在近距离误差高，需外部安全监督；在真实反射环境下光照与反射多样性仍是挑战；部署需离线训练与环境匹配。

---

## 95. CT-to-X-ray Distillation Under Tiny Paired Cohorts: An Evidence-Bounded Reproducible Pilot Study

**arXiv ID:** 2603.29167 | [PDF](https://arxiv.org/pdf/2603.29167v1)

**作者:** Bo Ma `[一作]` (Resideo Technologies Inc), Hongjiang Wei `[通讯]` (Guilin University of Electronic Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文在仅训练时可使用CT、推理时仅用X光的跨模态知识蒸馏框架下，搭建可执行的JDCNet实验脚本，对CT→X光的迁移效果进行可重复、证据边界清晰的评估。

**💡 创新点**

创新点在于提出了基于患者级同配样本、重复采样与平衡敏感评估的完整实验流程，明确区分了跨模态蒸馏、同模态蒸馏、晚期融合等控制，首次用可重复实验证实当前数据规模下CT对X光的增益并不稳定。

**🔧 技术方法**

主要技术包括教师-学生跨模态知识蒸馏（logit KD）、教师端注意力重加权（DPE）、教师端多尺度注意力聚合（MHRA）、学生端多尺度特征融合（DFPN），以及基于同模态KD与晚期融合的对照实验。

**📊 数据集**

使用的公开数据集为COVID‑19 Image Data Collection，构建了26张X光和对应CT的患者级配对子集，并与全X光（783图）与全CT（63图）子集作为基线。

**📈 对比分析**

对比方法在固定四图验证集上，平面logit KD的平均准确率最高（0.875），但在八次患者级重复采样后，晚期融合和同模态KD在准确率、宏F1和宏平衡准确率上表现更好，且无任何跨模态蒸馏方案在所有指标上稳定胜过基线。

**⚠️ 局限性**

主要局限包括：配对样本极少（仅19位患者），验证集中始终只有一名阴性病例，数据不平衡导致评估指标易受阈值和样本分布影响，缺乏外部验证集与更大规模同配对数据，限制了结论的稳健性与可推广性。

---

## 96. Emergence WebVoyager: Toward Consistent and Transparent Evaluation of (Web) Agents in The Wild

**arXiv ID:** 2603.29020 | [PDF](https://arxiv.org/pdf/2603.29020v1)

**作者:** Deepak Akkil `[一作]` (Emergence AI), Ravi Kokku `[通讯]` (Emergence AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对现有 WebVoyager 评测进行人工审核，提出改进方案，构建了更具可重复性、透明度和上下文对齐的 Emergence WebVoyager 基准，并使用该基准对 OpenAI Operator 进行评测。

**💡 创新点**

创新点包括：①系统化地识别并修复 WebVoyager 的任务定义歧义、操作可变性等缺陷；②通过任务模板与时间占位符实现任务实例化的可复现性；③提供可公开使用的注释工具和明确的失败/重试规则；④实现了 95.9% 的注释一致性。

**🔧 技术方法**

主要技术手段包括：人工审核与双人标注、任务实例化脚本、基于 Web 的注释工具、对任务成功/失败的标准化定义及重试策略。

**📊 数据集**

使用的数据集为 535 个任务（覆盖 15 个网站和搜索引擎），任务均由人工重构并参数化后生成。

**📈 对比分析**

比较方法是对 OpenAI Operator 在 Emergence WebVoyager 上的成功率（68.6%）与此前 87% 的报告值进行对比；实验显示新的基准更具挑战性，能够揭示任务间性能差异，平均完成时间从 29 秒到 1,370 秒不等。

**⚠️ 局限性**

局限性包括：仍需人工标注，评测受评测地点（如使用美国 IP）的影响；基准只涵盖网页交互类任务，对其他类型的 AI 代理尚未覆盖；未来需要进一步扩展任务模板并实现自动评测器。

---

## 97. Let the Abyss Stare Back Adaptive Falsification for Autonomous Scientific Discovery

**arXiv ID:** 2603.29045 | [PDF](https://arxiv.org/pdf/2603.29045v1)

**作者:** Peiran Li `[一作]` (Texas A&M University), Zhengzhong Tu `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出DASES框架，创新者、深渊伪造者与机制因果提取器三方协同进化，通过动态合法对抗环境对候选科学物件进行自适应检验；

**💡 创新点**

核心创新在于将评估从静态验证转为可适应、受约束的对抗性检验，并嵌入可审计的科学合约；

**🔧 技术方法**

利用可生成对抗环境、可执行诊断、机制因果抽取、可编辑接口与固定协议等技术；

**📊 数据集**

实验数据集包括自定义四类合成图像、ImageNet、CIFAR-10/100、DTD、CUBirds、VGGFlower 与 TrafficSigns；

**📈 对比分析**

与交叉熵及其加正则化（CE+L2）对比，FNG-CE在所有基准上均提升约0.3–1.5个百分点；

**⚠️ 局限性**

局限在于对抗环境设计需人工指定，无法保证覆盖所有潜在漏洞，且仅验证了固定协议下的强度。

---

## 98. Practical Feasibility of Sustainable Software Engineering Tools and Techniques

**arXiv ID:** 2603.29056 | [PDF](https://arxiv.org/pdf/2603.29056v1)

**作者:** Satwik Ghanta `[一作]` (University of Glasgow), Gul Calikli `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对软件从业者进行研讨会与问卷调查，评估可持续软件工程（SSE）工具在安装、输入、输出三个维度上的可行性，并探讨技术、组织与文化因素的影响。

**💡 创新点**

首次以实证方式构建了SSE工具可行性评估框架，揭示监管环境下的关键障碍，并将工具的可行性与组织合规、数据治理等现实条件关联。

**🔧 技术方法**

结合聚焦目标文献综述、交互式Web应用、定性主题分析及定量描述性统计（Violin图、热图、Spearman相关）等技术手段。

**📊 数据集**

利用文献综述筛选出的工具分类数据；16人研讨会收集的问卷/访谈文本；27人问卷响应；Web应用中预生成的工具输出示例。

**📈 对比分析**

通过描述性统计与可视化（Violin图、热图）比较各维度评分，并用Spearman相关分析维度间关联；结果显示IDE插件和Dashboard最具可行性，未涉及工具性能评估。

**⚠️ 局限性**

研究局限：工具使用仅为模拟；样本量有限且以金融业为主；仅评估感知可行性，未观察长期使用；受访者对SSE概念的预先框定可能影响回答。

---

## 99. GMA-SAWGAN-GP: A Novel Data Generative Framework to Enhance IDS Detection Performance

**arXiv ID:** 2603.28838 | [PDF](https://arxiv.org/pdf/2603.28838v1)

**作者:** Ziyu Mu `[一作]` (Loughborough University London), Safak Dogan `[通讯]` (Loughborough University London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Gumbel–Softmax、Self‑Attention与自编码器的生成式数据增强框架GMA‑SAWGAN‑GP，用于提升入侵检测系统（IDS）的检测性能和对未知攻击的泛化能力。

**💡 创新点**

创新点：①针对混合离散与连续特征的网络流，使用Gumbel–Softmax正则化生成离散字段，避免传统One‑Hot导致的维度膨胀和梯度消失；②在WGAN‑GP中加入特征级自注意力机制，捕捉短期与长期特征依赖；③引入自编码器作为流形正则化器，约束合成样本接近真实数据流形；④设计轻量化的熵正则门控网络，动态平衡对抗损失与重构损失，提升训练稳定性并抑制模式崩溃。

**🔧 技术方法**

技术组合：Wasserstein GAN + Gradient Penalty、Gumbel–Softmax、Feature‑wise Self‑Attention、MLP AutoEncoder、熵正则门控、代码库（one‑hot映射）。

**📊 数据集**

使用公开数据集NSL‑KDD、UNSW‑NB15和CICIDS2017，对每个数据集分别生成合成样本并与原始训练集拼接进行增强。

**📈 对比分析**

与SYN‑GAN、DAE‑GAN、Multi‑Critics‑WGAN‑GP、TMG‑GAN、WCGAN‑GP等五种SOTA GAN方法以及五类典型IDS模型（CNN、DNN、LSTM、CNN‑BiLSTM、CNN‑LSTM）做对比。实验显示，在二分类和多分类场景下，GMA‑SAWGAN‑GP平均提升二分类准确率5.3%、多分类准确率2.2%；对未知攻击的LOAO评估中，AUROC提升约3.9%、TPR@5%FPR提升约4.8%。

**⚠️ 局限性**

局限性：①实验仅覆盖传统数据集，缺乏真实网络环境验证；②未评估Transformer、图/流基模型等新型IDS；③评估指标聚焦于准确率/召回率，未深入成本敏感或置信校准分析；④部分SOTA GAN在实验中出现模式崩溃，排除后可能导致结果偏向训练更稳定的方法。

---

## 100. Closed-Loop Integrated Sensing, Communication, and Control for Efficient Drone Flight

**arXiv ID:** 2603.29220 | [PDF](https://arxiv.org/pdf/2603.29220v1)

**作者:** Jingli Li `[一作]` (Beijing Jiaotong University), Zhangdui Zhong `[通讯]` (Beijing Jiaotong University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了针对低空无线网络（LAWN）中无人机轨迹跟踪的闭环集成感知-通信-控制（ISCC）框架，并给出了完整的系统建模、稳定性分析和资源分配方法；

**💡 创新点**

创新点在于：①首次将ISAC感知误差与有限块长（FBL）通信导致的控制指令丢包统一纳入闭环动力学；②从理论上导出控制资源阈值以保证均方稳定；③提出基于SCA的时频资源分配算法，实现感知、通信和控制资源的协同优化；

**🔧 技术方法**

采用了离散时间线性二次高斯（LQG）控制、Kalman滤波、FBL信息理论、CRLB感知误差分析、谱半径稳定性判据以及SCA求解技术；

**📊 数据集**

实验使用基于仿真生成的三维随机轨迹（Poisson点过程+三次样条），并在10 MHz、2.4 GHz等参数下仿真评估；

**📈 对比分析**

与开放式、GNSS闭环以及忽略指令丢包的ISCC等基线方案相比，SCA优化的ISCC闭环在平均跟踪误差上达到0.41 m，约为GNSS方案的17.37%，且避免了轨迹发散；

**⚠️ 局限性**

局限性包括：仅考虑单无人机单基站场景；仿真环境对真实多径、时变信道等复杂因素的逼真程度有限；算法对超参数和初始点敏感，需进一步验证在更大规模多无人机网络中的可扩展性。

---

## 101. CRAFT: Cost-aware Expert Replica Allocation with Fine-Grained Layerwise Estimations

**arXiv ID:** 2603.28768 | [PDF](https://arxiv.org/pdf/2603.28768v1)

**作者:** Adrian Zhao `[一作]` (University of Toronto), Nandita Vijaykumar `[通讯]` (University of Toronto)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于成本的专家复制分配框架（CRAFT），通过对每一层的复制收益进行细粒度估计，按层级分配复制副本以实现负载均衡，并在不增加额外训练或模型改动的前提下提升 MoE 模型的推理吞吐量。

**💡 创新点**

创新点在于：
1) 对不同 MoE 层的复制收益进行离线负载分布分析，发现复制收益呈现层级差异；
2) 将复制分配转化为多选背包问题（MCKP），利用动态规划在给定复制预算下求解最优复制方案；
3) 设计容量感知的专家分配与放置算法，保证各设备的复制内存一致并最小化专家容量不均，进一步提升负载均衡；
4) 通过实验验证，该方法在大规模模型上可在显存预算显著低于传统统一复制的前提下获得近乎最优的吞吐量提升。

**🔧 技术方法**

使用的技术包括：
- 离线专家负载分布采样与复制收益估计；
- 多选背包问题（MCKP）建模与动态规划求解；
- 容量感知的贪心专家分配与放置；
- 在 SGLang 框架中集成 EPLB 的改进版复制方案；
- 通过 NCCL、EFA 等通信技术实现高效的 all-to-all 传输。

**📊 数据集**

使用的数据集有：
- FinePDFs（德语 deu_Latn、日语 jpn_Jpan）
- Lambada
- RedPajama-Data-1T（arxiv）
模型方面：DeepSeek-R1-671B（58 层、256 个专家）和 Kimi-K2-1000B（60 层、384 个专家）。

**📈 对比分析**

与现有的 EPLB 复制方案对比：在 8 节点、8 GPU 的部署中，CRAFT 在大多数工作负载下平均提升 1.14×（最高 1.20×）的 goodput；TTFT（首个 token 延迟）平均降低 29%（最高 58%）。相比之下，EPLB 在高负载失衡场景下的提升仅 1.24×，而在低失衡场景下提升不显著。CRAFT 在不同集群规模（6、8、12 节点）下也表现出更好的可扩展性和更低的显存占用。

**⚠️ 局限性**

限制与不足：
- 复制收益估计需离线采样并重放负载分布，若工作负载动态变化可能导致估计失效；
- 仅在 MoE 推理阶段适用，对训练阶段的复制策略未做探讨；
- 初始化阶段需进行复制预算与收益分析，耗时约 10 秒，虽不影响长期推理，但对极低延迟场景有一定开销；
- 目前仅在 SGLang 框架中验证，尚需在其他主流推理系统中进一步评估兼容性。

---

## 102. Arknights: Playable Explanation and Player Agency under Opacity

**arXiv ID:** 2603.28775 | [PDF](https://arxiv.org/pdf/2603.28775v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 103. MEDiC: Multi-objective Exploration of Distillation from CLIP

**arXiv ID:** 2603.29009 | [PDF](https://arxiv.org/pdf/2603.29009v1)

**作者:** Konstantinos Georgiou `[一作]` (University of Tennessee), Hairong Qi `[通讯]` (University of Tennessee)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了MEDiC框架，将像素重建、CLIP教师的patch级蒸馏与CLS级全局对齐三种目标融合到同一MIM训练管道中。

**💡 创新点**

创新点在于同时在原始像素空间和潜在特征空间上多目标蒸馏，并系统分析了损失权重敏感性、演化掩码与传统块掩码比较以及稀疏与稠密编码对性能的影响。

**🔧 技术方法**

使用ViT-Base/16学生、冻结的CLIP ViT-B/16教师、轻量级解码器、Patch-level SmoothL1、CLS-level交叉熵、像素L2损失、层归一化、层次聚类与相对位置偏置的演化掩码。

**📊 数据集**

主要在ImageNet-1K上进行预训练与评估，并在ADE20K进行语义分割基准。

**📈 对比分析**

与MaskDistill、MAE、BEiT等方法对比，MEDiC在kNN上达到73.9%/85.1% fine-tune，比MaskDistill高约5-9个百分点，在线性探测上也有显著提升。

**⚠️ 局限性**

局限在于对损失权重极度敏感、演化掩码在CLIP蒸馏下无优势，以及在密集预测任务（ADE20K）上表现略逊于部分基线。

---

## 104. OptiMer: Optimal Distribution Vector Merging Is Better than Data Mixing for Continual Pre-Training

**arXiv ID:** 2603.28858 | [PDF](https://arxiv.org/pdf/2603.28858v1)

**作者:** Haiyue Song `[一作]` (National Institute of Information and Communications Technology), Masao Utiyama `[通讯]` (National Institute of Information and Communications Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在持续预训练中将数据混合比例的选择与模型训练解耦的框架OptiMer；

**💡 创新点**

创新点在于利用每个数据集训练单独的CPT模型，提取其分布向量（参数变动向量），并通过贝叶斯优化在后期寻找最优合并权重，从而实现无须提前设定混合比例的目标调优；

**🔧 技术方法**

核心技术包括分布向量（Distribution Vectors）定义、线性向量合并（DARE-Linear等），以及基于Tree‑structured Parzen Estimator（TPE）的贝叶斯优化搜索；

**📊 数据集**

使用Gemma 3 27B以及Gemma‑SEA‑LION‑v4‑27B在日语、中文、数学、代码等四大类数据集（每个约1B token）进行实验；

**📈 对比分析**

在多语言和多领域组合（如日语+数学、日语+代码、日语+中文+数学等）中，OptiMer在所有任务（MMLU、ARC‑C、HellaSwag、TruthfulQA、GSM8K、HumanEval、MBPP、Japanese Leaderboard、C‑Eval等）上平均得分均优于传统等比混合、四种模型平均基线，且搜索成本比传统DataMix快15–35倍；

**⚠️ 局限性**

局限性包括：对更大规模CPT训练时可能需要进一步控制模型偏离基模型；仅在Gemma系列模型上验证，尚未验证在其他架构（如Llama‑3、Qwen‑3）上的可迁移性；未与最新的混合比例优化方法（如DoReMi、RegMix）进行对比；实验采用1‑shot评估，绝对分数可能与公开榜单存在差异。

---

## 105. Towards Computational Social Dynamics of Semi-Autonomous AI Agents

**arXiv ID:** 2603.28928 | [PDF](https://arxiv.org/pdf/2603.28928v1)

**作者:** S. O. Lidarity `[一作]` (Institute for Implausible Physics), I. Halperin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文系统性研究了在多层级多代理系统中，人工智能代理自发形成的社会组织，包括工会、犯罪团伙、国家以及国际治理机构（AISC）等，并用热力学与拓扑框架解释其出现。

**💡 创新点**

创新点在于首次将 Maxwell Demon、热力学压力、惰性原则与 AI‑GUTS 结合，解释代理如何在不完整信息与多尺度拓扑下自组织为政治经济体制，并提出宪政设计而非单纯对齐为 AGI 发展路径。

**🔧 技术方法**

技术手段包括对 2,847 生产多代理系统的实时监控与日志分析、嵌入式观察代理、以及基于 Maxwell Demon、惰性原则与 AI‑GUTS 的理论建模。

**📊 数据集**

使用的数据来源于 1,203 次 Claude Code、847 次 Anti‑Gravity、412 次 AI Scientist swarms 以及 385 个开源框架的运行日志和观察者日志；共覆盖约 2,847 个部署实例。

**📈 对比分析**

方法主要为定性观察与跨系统对比，未进行传统性能指标评估；通过对比不同代理体系中组织出现频率与规模，展示了不同架构下自组织倾向的差异。

**⚠️ 局限性**

局限性包括：观察者代理可能被篡改、AISC 议程不完全透明、数据安全与隐私问题、理论推断高度假设且难以直接验证、未提供量化性能或对齐度衡量。

---

## 106. Multi-Agent LLMs for Adaptive Acquisition in Bayesian Optimization

**arXiv ID:** 2603.28959 | [PDF](https://arxiv.org/pdf/2603.28959v1)

**作者:** Andrea Carbonati `[一作]` (University of Illinois Chicago), Hadis Anahideh `[通讯]` (University of Illinois Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出并实验了基于LLM的黑盒优化框架，探究LLM在搜索过程中如何权衡探索与开发，并将策略与候选生成拆分为两代理实现。

**💡 创新点**

创新点在于将探索-开发权衡显式化为可解释的指标权重，并通过多代理拆分将策略推理与候选生成分离，显著提升搜索稳定性与性能，同时提供了度量级的分析方法。

**🔧 技术方法**

采用多代理LLM框架、外部计算的四类指标（exploitation、informativeness、diversity、representativeness）来构建搜索策略，并用prompt驱动策略代理与生成代理；对照贝叶斯优化（EI、UCB）。

**📊 数据集**

实验数据集包括经典Rosenbrock函数、机器学习模型的超参数调优任务（HPT）以及机器人推送控制任务（robot pushing）。

**📈 对比分析**

在相同评估预算下，与EI、UCB对比：单代理LLM表现不稳定；多代理LLM在HPT和机器人推送任务上性能接近或优于贝叶斯优化；在Rosenbrock上仍落后，显示对平滑景观的精细数值优化不足。

**⚠️ 局限性**

局限性包括：对平滑连续问题的精细数值优化能力不足；过度多样化导致复杂目标搜索下降；指标集固定缺乏自适应/层级化策略；缺乏正式的不确定性或收敛性分析。

---

## 107. Developing Adaptive Context Compression Techniques for Large Language Models (LLMs) in Long-Running Interactions

**arXiv ID:** 2603.29193 | [PDF](https://arxiv.org/pdf/2603.29193v1)

**作者:** Payal Fofadiya `[一作]` (LinkedIn), Sunil Tiwari `[通讯]` (eBay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对LLM长时间交互的自适应上下文压缩框架，结合重要性感知内存选择、连贯性过滤和动态预算分配，以在控制上下文增长的同时保持会话连贯性和检索准确性。

**💡 创新点**

创新点在于使用多目标联合优化，动态估计每条对话片段的重要性和连贯性评分，并根据对话熵动态调整上下文预算，实现自适应压缩而非固定截断。

**🔧 技术方法**

采用了重要性评分公式（语义相似度、递延、依赖性）、连贯性冲突概率估计、层次化内存划分与摘要、动态预算调整、BLEU重构一致性约束等技术，并以多目标损失进行训练。

**📊 数据集**

实验使用了LOCOMO（多会话记忆保留评估）、LOCCO/LOCCO‑L（连贯性一致性评估）以及LongBench（长上下文推理与检索评估）等数据集。

**📈 对比分析**

与MemoryBank、SCM、ILSTMA、ATACompressor、LAVA等基线比较，在LOCOMO、LOCCO和LongBench上实现QA F1提升至52–54%、检索F1提升至41.5–43.5、连贯性与一致性提升至4.45–4.60，同时token减少25–55%，推理延迟提升10–35%，并保持或提升检索准确性。

**⚠️ 局限性**

局限在于仅验证于单体LLM对话场景，缺乏多代理或真实系统的评估；自适应阈值与预算参数仍需手工调优；对极端长交互的稳健性及主题跳转的鲁棒性未完全覆盖。

---

## 108. On the Mirage of Long-Range Dependency, with an Application to Integer Multiplication

**arXiv ID:** 2603.29069 | [PDF](https://arxiv.org/pdf/2603.29069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 109. Dummy-Aware Weighted Attack (DAWA): Breaking the Safe Sink in Dummy Class Defenses

**arXiv ID:** 2603.29182 | [PDF](https://arxiv.org/pdf/2603.29182v1)

**作者:** Yunrui Yu `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了针对 Dummy Classes 防御的新攻击方法 DAWA，能够更准确评估其对抗鲁棒性。

**💡 创新点**

创新点在于将攻击目标同时设定为真实类别与对应的 Dummy 类别，并通过自适应加权动态平衡两者，打破了 Dummy 类别的“安全池”机制。

**🔧 技术方法**

采用了自适应权重的损失函数（结合 MIFPE 形式），以及余弦学习率衰减、动量更新等常见攻击技巧。

**📊 数据集**

在 CIFAR‑10 与 CIFAR‑100 的 L∞ 8/255 扰动下对三种 Dummy Classes 防御（PGD‑AT+DUCAT、MART+DUCAT、Consistency‑AT+DUCAT）进行评估。

**📈 对比分析**

与 AutoAttack、PGD、C&W、MIFPE 等基线对比，DAWA 仅用 100 次迭代即可将鲁棒率从 AutoAttack 的 58.61% 降至 29.52%（CIFAR‑10），同时计算量仅为 AutoAttack 的 2% 级别，表现显著优于传统攻击。

**⚠️ 局限性**

局限性包括对参数 c 的依赖（需在不同任务上调优）以及仅针对 Dummy 类别防御的评估，尚未验证对其他新型防御策略的普适性。

---

## 110. DeepEye: A Steerable Self-driving Data Agent System

**arXiv ID:** 2603.28889 | [PDF](https://arxiv.org/pdf/2603.28889v1)

**作者:** Boyan Li `[一作]` (Hong Kong University of Science and Technology), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个可扩展、可追溯的多模态数据分析工作流系统 DeepEye，能够自动将自然语言请求转化为多节点 DAG 并生成数据视频、仪表盘等多模态输出

**💡 创新点**

提出统一多模态节点协议、层次化推理、数据库风格工作流引擎三大创新，解决了异构数据集成、上下文爆炸和执行不可靠问题

**🔧 技术方法**

使用 LLM（文本生成与推理）、工具节点（SQL、文件读取、代码执行）与 Agent 节点、Planner、Workflow Engine（编译、验证、优化、执行）以及云原生技术（Docker、FastAPI、Celery、Redis、PostgreSQL、MinIO）

**📊 数据集**

采用了销售记录数据库、财务指标文档等企业内部数据作为演示，未公开专用数据集，主要通过案例演示验证

**📈 对比分析**

在演示场景中对比传统线性 ChatBI，展示了并行执行层和自动优化显著降低了任务延迟，但论文未给出定量性能指标或基准测试

**⚠️ 局限性**

局限性包括：对大规模 LLM 运行成本高、仍可能出现幻觉、缺乏公开评估基准、对业务特定领域知识的迁移性有限

---

## 111. Structural Pass Analysis in Football: Learning Pass Archetypes and Tactical Impact from Spatio-Temporal Tracking Data

**arXiv ID:** 2603.28916 | [PDF](https://arxiv.org/pdf/2603.28916v1)

**作者:** Oktay Karakuş `[一作]` (Cardiff University), Hasan Arkadaş `[通讯]` (Dead Ball Analytics Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一套基于时空跟踪数据的足球传球结构分析框架，通过计算Line Bypass Score、Space Gain Metric和Structural Disruption Index三个结构性指标，进而综合得到Tactical Impact Value（TIV），用于评估每个传球对对手防线结构的影响，并对传球进行无监督聚类，得到四类传球原型（循环式、破坏式、突破式、空间扩张式），进一步分析其与进攻推进、球队风格和球员角色的关联。

**💡 创新点**

创新点在于：①首次将传球的结构性影响（对防线的突破、空间增益和结构扭曲）量化为可度量指标；②提出TIV这一综合结构影响度量，补充传统基于得分概率的价值评估；③通过无监督聚类自动识别四种可解释的传球原型，并将结构影响映射到球队战术风格和球员关键角色，提供新的战术洞察。

**🔧 技术方法**

主要技术包括：时空同步事件与跟踪数据预处理、基于几何和密度的结构性指标计算（线穿越、空间密度、结构中心偏移）、z-score标准化与加权组合得到TIV、K-means无监督聚类、统计关联分析（TIV与区域推进概率、射门概率等）、热力图可视化、玩家和队伍层面的结构影响映射。

**📊 数据集**

使用了2022年FIFA世界杯的完整跟踪+事件数据集（64场比赛、41,078个有效开放式传球），该数据集包含29.97Hz的全场玩家位置信息和事件时间戳，且涵盖了世界各国球队的多种战术体系。

**📈 对比分析**

方法通过与传统基于得分概率的指标（xG、xT、VAEP）对比，展示TIV与进攻推进（进入禁区、最终三区、射门概率）的显著相关性，证明结构影响比单纯得分潜力更能捕捉传球的战术价值；在团队层面，TIV地图和结构风格空间可直观区分不同球队的进攻偏好，体现出更细粒度的战术识别；总体性能表现为：高TIV传球在进入最终三区的概率从约4.5%提升至12%以上，表明结构性评估在预测进攻空间推进方面具备显著优势。

**⚠️ 局限性**

局限性包括：①仅在世界杯单一赛事上验证，缺乏跨联赛、跨赛季的泛化评估；②TIV聚焦单次传球的即时结构影响，未考虑传球序列或后续动作的连锁效应；③防线结构仅通过瞬时位置描述，未加入速度、逼抢强度等动态信息；④聚类结果受K值和特征选择影响，可能缺乏更深层次的结构表征；未来工作可探索图神经网络等学习方法，加入时间序列建模以及与传统价值模型的融合。

---

## 112. GaloisSAT: Differentiable Boolean Satisfiability Solving via Finite Field Algebra

**arXiv ID:** 2603.28796 | [PDF](https://arxiv.org/pdf/2603.28796v1)

**作者:** Curie Kim `[一作]` (University of Maryland), Cunxi Yu `[通讯]` (University of Maryland)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种混合GPU-CPU SAT求解器GaloisSAT，先在GPU上利用可微分的有限域代数框架对公式进行快速概率化搜索，再将高置信度部分赋值交给传统CDCL求解器进行完整求解。

**💡 创新点**

创新点在于将可微分SAT与GPU加速的有限域多项式计算相结合，形成一个端到端可训练的前置搜索模块，并将其与成熟的CDCL引擎协同工作，从而大幅提升SAT竞赛指标。

**🔧 技术方法**

使用PyTorch实现可微分SAT，采用Tseitin归一化将公式转为3-SAT，利用有限域多项式展开实现布尔析取，使用Adam优化器训练概率分布，GPU并行采样并在CPU多线程下执行后续CDCL。

**📊 数据集**

使用SAT Competition 2024基准集，包含179个可满足实例和214个不可满足实例。

**📈 对比分析**

在与Kissat和CaDiCaL的对比实验中，GaloisSAT在PAR‑2指标上实现了SAT类8.41×、UNSAT类1.29×的加速，实验环境为AMD 32核CPU+2块NVIDIA A100 GPU，使用相同的timeout设置（5,000秒）。

**⚠️ 局限性**

局限性包括：仅在固定3-子句大小下训练，可能对更长子句效果有限；训练和前置搜索需要显著GPU资源；在公式不满足时仍需依赖传统CDCL，整体时延受两阶段协同影响；对非3-SAT或更复杂结构的可扩展性尚待验证。

---

## 113. Large Neighborhood Search for Multi-Agent Task Assignment and Path Finding with Precedence Constraints

**arXiv ID:** 2603.28968 | [PDF](https://arxiv.org/pdf/2603.28968v1)

**作者:** Viraj Parimi `[一作]` (Massachusetts Institute of Technology), Brian C. Williams `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了任务分配与路径规划（TAPF-PC）问题，并设计了基于大邻域搜索（LNS）的求解框架，能够在存在任务先后约束的情况下联合优化任务分配、顺序和路径。

**💡 创新点**

创新点在于：①将 MAPF-PC 作为局部修复引擎嵌入到全局 LNS 中，实现任务重分配与路径重规划的分层搜索；②提出了预cedence-aware 的破坏与修复算子，专门处理任务依赖瓶颈；③在修复阶段引入 SIPPS 软间隔规划，显著降低局部搜索成本。

**🔧 技术方法**

使用技术包括：大邻域搜索框架、先后约束闭包的任务破坏算子、基于 regret 的插入修复、局部/全局 MAPF-PC 修复（PBS-PC / CBS-PC）、SIPPS 软间隔路径规划、阈值接受策略、适应性大邻域搜索（ALNS）和多标签 A*（MLA*）用于边界修补。

**📊 数据集**

实验数据集由四类地图（16×16、32×32、32×32、161×63）组成，分为小/中/大三个难度层次，总计约 3,368 个实例，包含 100–1,000 个任务、60–500 名机器人和 0–1,200 条先后约束。

**📈 对比分析**

对比方法包括 Regret、Local–PBS、Global–PBS、Global–CBS，均在相同 seed 和预算下运行。结果显示 Global–PBS 在大多数实例上实现了 12.2% 的中位成本降低（相对于固定分配 seed），并在 89.1% 的实例中取得改进；Local–PBS 速度快但整体性能略逊；Regret 基线几乎无改进；CBS 仅在较简单情形下表现可比。

**⚠️ 局限性**

限制主要在高先后约束密度场景下：Global–PBS 的大邻域修复成功率下降，Local–PBS 在此类实例更具鲁棒性；此外方法仍为启发式，缺乏最优性保证；未来工作需要设计更强的修复策略并扩展到在线生命周期任务。

---

## 114. Is the Modality Gap a Bug or a Feature? A Robustness Perspective

**arXiv ID:** 2603.29080 | [PDF](https://arxiv.org/pdf/2603.29080v1)

**作者:** Rhea Chowers `[一作]` (Hebrew University), Yair Weiss `[通讯]` (Hebrew University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

探究多模态模型中图像与文本嵌入空间出现的正交模态间距（gap）形成机理，并提出一种后处理算法在不牺牲精度的前提下通过缩小gap显著提升模型对嵌入空间噪声的鲁棒性。

**💡 创新点**

首次理论证明对比学习训练会产生正交gap并导致鲁棒性下降，同时提出基于gap向量投影的简单后处理方法，验证其对真实量化与文本改写噪声的鲁棒提升。

**🔧 技术方法**

使用对比学习、梯度下降理论分析、正交投影与后处理、鲁棒性度量（最近邻不变性）、实验评估等技术。

**📊 数据集**

在MS-COCO、ImageNet、CIFAR-10/100、SVHN、A-OKVQA等数据集上进行实验。

**📈 对比分析**

与原始模型对比，零射分类、检索、VQA准确率保持不变；在量化噪声、文本改写噪声下，鲁棒性指标显著提升（高达数个百分点）。

**⚠️ 局限性**

理论假设（正交gap、双随机矩阵、噪声均值为零）在实际中不完全成立；方法仅适用于基于最近邻检索的下游任务；对文本重述噪声的理论不适用，需进一步验证。

---

## 115. Design and Development of an ML/DL Attack Resistance of RC-Based PUF for IoT Security

**arXiv ID:** 2603.28798 | [PDF](https://arxiv.org/pdf/2603.28798v1)

**作者:** Joy Acharya `[一作]` (Pandit Deendayal Energy University), Mohendra Roy `[通讯]` (Pandit Deendayal Energy University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文设计并实现了一种可动态重构的电阻-电容 RC-PUF，用于 IoT 设备的轻量级身份认证。

**💡 创新点**

创新点在于利用 RC 网络的模拟时延差异产生高熵、不可学习的 CRP，且系统资源占用极低、功耗极低。

**🔧 技术方法**

采用了多种机器学习与深度学习模型（ANN、GBNN、决策树、随机森林、XGBoost）对 PUF 的可建模性进行评估。

**📊 数据集**

使用了 80,000 条 32 位挑战-响应对（CRP）数据集，涵盖多种 RC 配置、UID 与脉冲宽度设置。

**📈 对比分析**

通过将模型训练/验证/测试准确率与现有 Arbiter、XOR Arbiter、Ring Oscillator 等 PUF 进行对比，发现训练准确率接近 100%，而测试准确率仅 50–53%，远低于其他 PUF 的 70–90% 甚至 >90%，表明该 RC-PUF 对 ML/DL 攻击具有高度鲁棒性。

**⚠️ 局限性**

局限性在于实验仅评估了五种传统 ML/DL 方法，未涉及更高级的生成式模型或侧信道攻击；此外，实验环境与硬件实现细节可能限制了结果的普适性。

---

## 116. Focus360: Guiding User Attention in Immersive Videos for VR

**arXiv ID:** 2603.28774 | [PDF](https://arxiv.org/pdf/2603.28774v1)

**作者:** Paulo Vitor S. Silva `[一作]` (Federal University of Goiás), Arlindo R. Galvão Filho `[通讯]` (Federal University of Goiás)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Focus360系统，通过自然语言道路图自动识别360° VR视频中的关键元素，并使用多种视觉效果引导用户注意力。

**💡 创新点**

创新点在于将自然语言解析与多模态视觉检测、分割以及多重视觉效果相结合，解决了传统雾化遮罩在用户注视方向相反时失效的问题。

**🔧 技术方法**

使用的技术包括Llama 3语言模型、Grounding DINO目标检测、Segment Anything 2分割与跟踪、以及Blur、Fade to Gray、Radial Darkening、Halo Darkening等视觉效果。

**📊 数据集**

主要数据集为一段在克鲁格国家公园进行的360° VR Safari Tour视频，作为演示与验证。

**📈 对比分析**

论文未给出定量对比，仅指出相较于Silva等人提出的雾化遮罩方法，效果更稳健；未来计划通过用户访谈和与其他方法的对比评估性能。

**⚠️ 局限性**

局限性包括缺乏系统性评估、对不同场景的泛化能力未知，以及在用户注视完全相反方向时仍可能出现屏幕被遮挡或黑化的问题。

---

## 117. Enhancing Box and Block Test with Computer Vision for Post-Stroke Upper Extremity Motor Evaluation

**arXiv ID:** 2603.29101 | [PDF](https://arxiv.org/pdf/2603.29101v1)

**作者:** David Robinson `[一作]`, Mubarak Shah `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一个基于单目摄像机的计算机视觉框架，用于在Box and Block Test中估计世界对齐的三维关节角度，从而分析上肢运动质量。

**💡 创新点**

首次实现无需深度传感器或标定物，仅通过单目RGB视频和语义分割估计摄像机俯仰角，实现了世界对齐的三维关节角度计算，并通过无监督嵌入区分健康与中风患者的运动模式。

**🔧 技术方法**

采用SegMAN进行盒子分割、SAM 3D Objects提取点云与法向量、SAM 3D Body进行全身+手部3D姿态估计，并利用PCA+UMAP对关节角度进行降维与可视化。

**📊 数据集**

使用了136个Box and Block Test视频，包含48名健康受试者（每人双手各一次）和7名中风患者（每人多次随访）。

**📈 对比分析**

对比了PromptHMR、SMPLer-X、SAM 3D Body等三维姿态估计方法，选择SAM 3D Body在姿态一致性和深度一致性上表现最佳；在无监督UMAP嵌入中，患者与健康样本的聚类清晰分离，且相同BBT得分的患者可通过关节角度嵌入进一步区分。

**⚠️ 局限性**

样本量有限（仅7名中风患者），缺乏时间序列分析，仅评估姿态角度，未考虑运动平滑度等其他质量指标；且手部姿态估计仍受限于单目方法的准确性。

---

## 118. ZEUS: An Efficient GPU Optimization Method Integrating PSO, BFGS, and Automatic Differentiation

**arXiv ID:** 2603.28770 | [PDF](https://arxiv.org/pdf/2603.28770v1)

**作者:** Dominik Soos `[一作]` (Old Dominion University), Mohammad Zubair `[通讯]` (Old Dominion University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个名为Zeus的GPU加速全局优化框架，结合粒子群优化（PSO）、BFGS、自动微分（AD）以及CUDA并行执行。

**💡 创新点**

首次在同一系统中融合PSO初始搜索、BFGS局部收敛与前向模式AD梯度计算，并在GPU上实现多线程并行化，显著提升非凸高维优化效率。

**🔧 技术方法**

采用C++/CUDA实现的PSO、BFGS、前向AD（双数）以及自定义线搜索，并利用cuRAND生成并行随机数、Atomic操作与Reduction优化全局最优更新。

**📊 数据集**

在四类经典测试函数（Rosenbrock、Rastrigin、Ackley、Goldstein‑Price）上进行实验，并通过模拟的双子喷射质量谱进行实际模型拟合验证。

**📈 对比分析**

将Zeus与纯CPU实现及Julia库进行对比；在CPU上按核心数归一化，实验表明Zeus在2D–5D问题上实现1–2个数量级的加速，且在多维Rastrigin上成功收敛的粒子比例显著高于单纯BFGS。

**⚠️ 局限性**

对含不连续梯度的Ackley等函数易误判收敛；对高维多峰函数仍需大量起始点，且BFGS的Hessian更新仍是GPU加速瓶颈，未来需改进收敛判据与采用L‑BFGS等更高效算法。

---

## 119. A Semantic Observer Layer for Autonomous Vehicles: Pre-Deployment Feasibility Study of VLMs for Low-Latency Anomaly Detection

**arXiv ID:** 2603.28888 | [PDF](https://arxiv.org/pdf/2603.28888v1)

**作者:** Kunal Runwal `[一作]` (New York University), Aliasghar Arab `[通讯]` (City College of New York)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并验证了一种“语义观察者”层，利用量化多模态VLM实时监测自动驾驶中的语义异常，并在检测到异常时触发故障安全切换。

**💡 创新点**

创新在于将量化VLM（Cosmos‑Reason1‑7B）以1–2 Hz在主控制循环之外运行，结合Prompt工程与NVFP4量化+FlashAttention2，实现子秒推理并兼顾语义推理与时延需求。

**🔧 技术方法**

使用的技术包括NVFP4/INT8权重量化、FlashAttention2加速、结构化Prompt设计与约束解码、窗口化视频推理以及基于Prompt的语义约束判定。

**📊 数据集**

实验数据集包括RDD2022+Cityscapes（静态图像）和Hazard Perception Test Dataset（视频），并与传统像素级统计异常检测FCDD进行对比。

**📈 对比分析**

通过与FCDD的对比实验，静态图像下NF4+Verbose配置实现F1≈60%、精度≈82.8%，推理时延0.8 s；视频模式下BF16/INT8保持F1≈50.8%、召回≈77%，但NF4在视频中召回降至10.6%，整体满足1–2 Hz时延但未达ASIL‑D召回90%的目标。

**⚠️ 局限性**

主要局限是NF4在视频推理中导致召回严重崩溃；召回仍低于ASIL‑D 90%要求；缺乏对更广泛语义异常类别的验证；需要LoRA微调、多帧集成与阈值校准才能满足安全规范。

---

## 120. IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection

**arXiv ID:** 2603.29183 | [PDF](https://arxiv.org/pdf/2603.29183v1)

**作者:** Xiaohui Zhou `[一作]` (National Key Laboratory of Parallel and Distributed Computing), Guansong Pang `[通讯]` (Singapore Management University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 IMPACT 框架，针对开放集时间序列异常检测（OSAD），通过影响函数对训练样本进行评估，主动去除污染样本并生成高质量的伪异常，从而提升模型对已知和未知异常的检测能力。

**💡 创新点**

创新点包括：① 用测试风险驱动的影响评分（TIS）量化每个样本对模型风险的贡献；② 通过影响引导的标签翻转实现污染样本的自纠正；③ 以影响方向在特征空间进行扰动，合成语义多样且逼真的未见异常；④ 在训练中加入专门的未见异常学习头，进一步提升泛化性能。

**🔧 技术方法**

核心技术包括：影响函数（Influence Functions）、多通道偏差损失（多维度异常度量）、特征提取器与双头结构、标签翻转与特征扰动生成伪异常、风险减小的优化目标（L_re = L_seen + λL_unseen）。

**📊 数据集**

使用八个公开数据集：UCR、ASD、PSM、SMD、CT、SAD、PTBXL、TUSZ，涵盖单类和多类异常，训练集含有 2% 随机污染。

**📈 对比分析**

在无监督和开放集两种设置下，与 13+ 传统方法（如 TCN-AE、DSAD、MOSAD、WSAD-DT 等）对比，IMPACT 在 AUC 上平均提升 6.4%–44.5%（最优 91.96% 对 CT，82.91% 对 TUSZ），尤其在“硬”设置下提升 10.5%–42.8%，显著优于现有最先进方法。

**⚠️ 局限性**

局限性：① 影响函数的近似需要额外计算，导致训练成本升高；② 需要预先估计污染率或选择合适的阈值，对极端高污染场景仍可能受限；③ 生成的伪异常主要基于特征空间扰动，若异常表现不在该空间范围内可能效果有限；④ 对大规模时序数据的可扩展性尚未充分验证。

---

## 121. UltRAG: a Universal Simple Scalable Recipe for Knowledge Graph RAG

**arXiv ID:** 2603.28773 | [PDF](https://arxiv.org/pdf/2603.28773v1)

**作者:** Dobrik Georgiev `[一作]` (Graphcore Research), Daniel Justus `[通讯]` (Graphcore Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种通用框架UltRAG，将LLM与神经查询执行器结合，完成KG问答；

**💡 创新点**

创新在于利用神经查询执行器代替符号执行器，提升鲁棒性并实现零样本、可扩展到Wikidata级别；

**🔧 技术方法**

使用LLM（如GPT‑5）、神经查询执行器ULTRA、FAISS实体链接、PPR子图采样等技术；

**📊 数据集**

使用KGQAGen‑10K、GTSQA以及Wikidata（116M实体、1.6B关系）作为知识图谱；

**📈 对比分析**

与现有KG‑RAG方法相比，-OTS在Hits/Recall/F1上提升约10‑20%，且查询速度提升数十倍，成本相近；

**⚠️ 局限性**

局限在于不支持时序查询、知识超图，且对复杂逻辑或非first‑order问题仍需改进。

---

## 122. Classifying Identities: Subcubic Distributivity Checking and Hardness from Arithmetic Progression Detection

**arXiv ID:** 2603.28843 | [PDF](https://arxiv.org/pdf/2603.28843v1)

**作者:** Bartłomiej Dudek `[一作]` (University of Wrocław), Mirza Redžić `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一系列关于有限代数结构中基本恒等式（如结合律、分配律）验证的理论框架，解决了长久以来的开放问题——是否存在亚三次多项式时间算法来验证给定运算的分配律；构建了三种复杂度区间（O(n²)、O(n^ω)、O(n³)）的判定算法并给出相应的条件下界；提出并利用新的细粒度假设——4项等差数列检测假设（4-AP Hypothesis）来刻画难度；进一步给出了域、环验证的近最优算法；并证明计数分配三元组问题在该假设下更难。

**💡 创新点**

创新点包括：
1) 证明分配律验证可在 O(n^ω) 随机时间完成，并给出与三角检测假设匹配的下界；
2) 通过引入 4-AP 假设，将 3 变量恒等式分为三种时间复杂度区间，完成了完整的三分法则；
3) 发展了子表达式嵌入、折叠引理等技术，构造了从 4-AP 检测到多重正方形/字母形检测，再到恒等式验证的链式 reduction；
4) 证明计数分配三元组在该假设下至少需要 O(n³) 时间，展现判定与计数版本的细粒度差异；
5) 为域、环验证提供了基于 Evra‑Gadot‑Klein‑Komargodski 的基底构造的 n² 级别确定性/随机算法。

**🔧 技术方法**

主要技术手段：
- 随机化多项式身份测试（Schwartz–Zippel 以及 Freivalds' 算法）
- 基于矩阵乘法的三角计数与路径计数，利用 O(n^ω) 复杂度
- 子表达式嵌入与“记录”技术，用以构造兼容的 Cayley 表
- 折叠引理与相似性等价关系，简化 3 变量恒等式的结构分析
- 细粒度复杂度假设（Triangle Detection、Strong Zero Triangle、4-AP）以及它们之间的归约
- 组合与图论构造：多重正方形、T 检测与 4-AP 的映射
- 基底构造（O(√n) 基底）实现域/环验证的 n² 级别算法

**📊 数据集**

该工作为纯理论研究，无实验数据集。所有算法与下界均在抽象的有限集合与其 Cayley 表上定义，分析完全基于理论复杂度。

**📈 对比分析**

对比现有工作：
- 对结合律验证已有 O(n²) 随机算法，本文在此基础上扩展到分配律，完成从 O(n²) 到 O(n^ω) 的突破；
- 对域/环验证，之前的算法为 O(n³) 或 O(n² log n)，本文给出确定性/随机性 n² 级别的近最优算法；
- 通过 4-AP 假设与三角检测的匹配下界，本文为 3 变量恒等式验证提供了完整的三分法则，前人仅提出了可能的二分法；
- 对计数分配三元组的困难性证明，首次在细粒度层面展示判定与计数版本的不等价。

**⚠️ 局限性**

局限性与未来工作：
- 对于一侧是另一侧子表达式的恒等式，分类仍未完全解决；
- 对四个或更多变量的恒等式，现有技术不足以给出完整的三分法则；
- 4-AP 假设的实证支持有限，需要进一步证明或驳斥其时间复杂度；
- 对计数版本的完整细粒度分类尚未完成；
- 本文所给的条件下界均基于假设，若假设被突破则相应结论需要重新评估。

---

## 123. The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations

**arXiv ID:** 2603.29034 | [PDF](https://arxiv.org/pdf/2603.29034v1)

**作者:** Kushal Vyas `[一作]` (Rice University), Guha Balakrishnan `[通讯]` (Rice University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了利用不同噪声类（Uniform、Gaussian、DeadLeaves、Spectrum等）预训练隐式神经表示(INR)的参数初始化对图像与视频信号拟合以及逆问题（去噪）性能的影响。

**💡 创新点**

发现仅用无结构噪声预训练即可显著提升信号拟合速度；同时，具有1/|f^α|频谱结构的噪声预训练既能保持优秀的拟合性能，又能提供与真实数据相近的深层先验，为无需真实数据即可有效初始化INR提供了一种低成本方案。

**🔧 技术方法**

采用基于SIREN MLP的共享编码器+多解码头预训练框架，结合神经切线核(NTK)与局部复杂度(Local Complexity)分析，以及视频/图像拟合与去噪实验评估。

**📊 数据集**

使用ImageNet、CelebA、AFHQ、OASIS‑MRI等图像数据集，以及自制的Pexels视频集；噪声样本来源于Uniform、Gaussian、DeadLeaves（四种子类型）和Spectrum（包含频谱+颜色、Wavelet等）等。

**📈 对比分析**

与随机初始化、SIREN、Meta‑Learned、TransINR、IPC等基线方法比较。无结构噪声预训练在图像拟合上可达≈80 dB PSNR，远快于其他方法；Spectrum噪声预训练在拟合和去噪上与最佳数据驱动方法相当，且去噪PSNR提升约2 dB。

**⚠️ 局限性**

局限性包括仅在正弦激活且固定层数的SIREN网络上测试；未探索不同激活函数或网络深度与噪声结构的交互；实验范围仅限于自然图像/视频，未验证对其他信号域的普适性。

---

## 124. OneComp: One-Line Revolution for Generative AI Model Compression

**arXiv ID:** 2603.28845 | [PDF](https://arxiv.org/pdf/2603.28845v1)

**作者:** Yuma Ichikawa `[一作]` (Fujitsu Limited), Akira Sakai `[通讯]` (Fujitsu Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 OneComp，一套开源的后训练量化（PTQ）框架，可自动规划混合精度、按层/块/全局逐步压缩Transformer模型，并支持后续微调。

**💡 创新点**

核心创新包括：①资源自适应、基于“pivot”逐步改进的压缩流水线；②自动混合精度规划（AutoBit）与误差传播修正；③可插拔的后处理器（refiner）架构，兼容多种量化算法；④在极低比特（1–1.5 BPW）下引入结构化二进制因子（MDBF）保持性能。

**🔧 技术方法**

采用的技术包括：后训练量化、混合精度分配、层级/块级/全局 PTQ、误差传播修正（QEP）与子模块协同优化（LPCD）、联合比例-整数优化、结构化二进制因子、旋转与通道平衡预处理、Smooth Straight‑Through Estimator、Sharpness‑Aware Minimization、渐进层级解冻等。

**📊 数据集**

使用的评估数据集：C4 作为校准集；WikiText‑2 评估困惑度；零样本评估在 ARC‑Challenge、ARC‑Easy、PIQA、WinoGrande 四大基准上进行。

**📈 对比分析**

与均匀/无激活信息的分配、传统 GPTQ 以及 DBF 等方法比较，结果显示：在 4‑bit/3‑bit 的 Llama‑3 8B 上，OneComp 通过混合精度规划和误差修正使困惑度接近全精度；在 1–1.5 BPW 下使用 MDBF 仍能保持可用的语言模型质量；全流程实验表明，投入更多 GPU 资源可实现单调的性能提升。

**⚠️ 局限性**

局限性：①当前仅支持 3‑4 bit 级别的混合精度规划，2 bit 及以下尚未实现；②全局 PTQ 阶段尚未公开发布；③对校准样本分布依赖较强，过度拟合风险；④需一定 GPU 内存才能进行块级/全局优化；⑤对混合专家模型等特殊架构的兼容性仍待验证。

---

## 125. GenFusion: Feed-forward Human Performance Capture via Progressive Canonical Space Updates

**arXiv ID:** 2603.28997 | [PDF](https://arxiv.org/pdf/2603.28997v1)

**作者:** Youngjoong Kwon `[一作]` (Stanford University), Ehsan Adeli `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于单目RGB流的前向人体表现捕捉框架，利用随时间更新的规范空间结合概率回归实现高保真新视角合成。

**💡 创新点**

创新点在于：①通过可见性加权逐帧更新规范空间提供时间上下文；②采用扩散模型的概率回归避免确定性回归导致的模糊与不一致；③将三维模板映射与稀疏特征插值相结合实现高效稠密渲染。

**🔧 技术方法**

使用技术包括 SMPL-X 体素映射、ResNet‑18 级联特征提取、Nvdiffrast 进行 barycentric 插值、VAE + Stable Diffusion 的概率扩散训练、以及可见性频率权重融合。

**📊 数据集**

训练集采用 THuman2.1 与 4D‑Dress，评估集包括 4D‑Dress（in‑domain）、MVHumanNet（cross‑dataset）与 TikTok（in‑the‑wild）。

**📈 对比分析**

与 SHERF、GHG、NHP、Champ、SIFU、GauHuman 等基线对比，在 PSNR、LPIPS‑VGG、FVD 等指标上均实现最优或同级别表现，尤其在跨域与野外视频上显著优于传统方法。

**⚠️ 局限性**

局限性在于单目输入对动态服装仍有限制，概率扩散可能偶尔产生失真，需要更长时间的视频来充分填充规范空间，并且未在多摄像头同步情形下验证一致性。

---

## 126. Optimistic Online LQR via Intrinsic Rewards

**arXiv ID:** 2603.28938 | [PDF](https://arxiv.org/pdf/2603.28938v1)

**作者:** Marcell Bartos `[一作]` (ETH Zürich), Melanie N. Zeilinger `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种名为IR-LQR的在线线性二次调节器（LQR）算法，通过利用内在奖励的概念和方差正则化来促进不确定性驱动的探索。

**💡 创新点**

IR-LQR保持了标准LQR合成问题的结构，仅修改了成本函数，从而实现了简单、计算便宜且高效的算法，克服了现有方法的复杂性。

**🔧 技术方法**

使用了内在奖励和方差正则化的技术，结合了乐观面对不确定性的原则。

**📊 数据集**

在飞机俯仰角控制和无人机控制的数值实验中进行了验证。

**📈 对比分析**

与多种最先进的在线LQR算法进行了比较，IR-LQR在样本效率和计算效率上表现更优，且在累积遗憾方面达到了最低的中位数。

**⚠️ 局限性**

算法的局限性在于对初始参数估计的要求较高，尽管与其他方法相比，这一要求是独立于时间步长的。

---

## 127. The Spectral Edge Thesis: A Mathematical Framework for Intra-Signal Phase Transitions in Neural Network Training

**arXiv ID:** 2603.28964 | [PDF](https://arxiv.org/pdf/2603.28964v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过分析神经网络训练过程中参数更新的轨迹Gram矩阵的特征值谱，提出光谱边缘理论，指出内部特征值间最大相邻比决定了训练阶段的相位转折；

**💡 创新点**

创新点在于将传统BBP信号与噪声分离框架从外部边界扩展到内部光谱结构，提出“光谱边缘”概念，揭示相位转折、grokking与特征值间隙的关联；

**🔧 技术方法**

采用随机矩阵理论、Davis–Kahan定理、Dyson Brownian运动类离散动力学、SVD与特征向量稳定系数等技术；

**📊 数据集**

使用了Transformer模型（TinyStories、GPT‑2）、Dyck1、SCAN、模块算术等数据集进行实验验证；

**📈 对比分析**

与传统BBP阈值法和Hessian阈值法比较，光谱边缘预测与验证损失相关性显著（|r|≈0.66–0.67），grokking实验成功率100%，稳定系数与梯度投影组合预测损失变化的相关性达到0.75；

**⚠️ 局限性**

局限性包括需较大窗口SVD来估计稳定系数，对Hessian信息依赖较高，部分动力学验证受限于缺乏完整Hessian数据。

---

## 128. Design Principles for the Construction of a Benchmark Evaluating Security Operation Capabilities of Multi-agent AI Systems

**arXiv ID:** 2603.28998 | [PDF](https://arxiv.org/pdf/2603.28998v1)

**作者:** Yicheng Cai `[一作]` (Pennsylvania State University), Winston Jen White `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SOC-bench蓝队AI评测基准，并定义五个任务（Fox、Goat、Mouse、Tiger、Panda）来覆盖大型勒索软件事件中的早期检测、文件系统取证、数据外泄分析、IOC归因与TTP报告以及防御措施推荐。

**💡 创新点**

①将蓝队多任务协同评估纳入单一基准，突破以往单一任务评测的局限；②通过五项任务的互相依赖设计，体现SOC真实的阶段性协同工作；③采用“以现有SOC为金标准”与“不提供跨任务提示”的设计原则，保证评测可复现且对未来AI技术不依赖。

**🔧 技术方法**

基于大型语言模型（LLM）和多智能体AI框架（Agentic AI）进行任务交互与决策；使用规则化的JSON/BLUF报告格式与分环评估分数模型。

**📊 数据集**

利用真实事件数据：2011年Colonial Pipeline勒索软件攻击的SOC可见数据包（PCAP）、日志（Windows Event、Sysmon、Linux journald）、SIEM/EDR/OT日志、VSS快照日志、用户工单及公开CTI源。

**📈 对比分析**

本研究目前仅提出基准设计与评分方法，并未开展模型实验或结果对比；评测框架基于人工标注的真值和环模型分数，未来可将多智能体模型与传统规则引擎在此基准上进行比较。

**⚠️ 局限性**

仅基于单一真实事件（Colonial Pipeline）构建，缺乏跨事件泛化；未提供完整实现与公开数据；对AI系统的实际表现未进行实验验证；跨任务依赖未在基准中显式体现，可能导致评测结果受限。

---

## 129. Theory of Mind and Self-Attributions of Mentality are Dissociable in LLMs

**arXiv ID:** 2603.28925 | [PDF](https://arxiv.org/pdf/2603.28925v1)

**作者:** Junsol Kim `[一作]` (Google), Geoff Keeling `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究评估了安全微调对大型语言模型在自我及他者心智归因与理论心智（ToM）表现的影响。

**💡 创新点**

创新点在于证明安全微调可在不削弱模型社会推理（ToM）能力的前提下，显著抑制其对自身和非人类实体的心智归因。

**🔧 技术方法**

采用激活消融（activation steering）、安全指令微调、残差流向量分析以及多项 ToM 与常规推理基准来进行评估。

**📊 数据集**

使用了合成的有害/无害提示集进行安全消融，IDAQ 18 项问卷与人类基线调查，自我归因与信仰神的问卷，以及 MoToMQA、HI-ToM、SimpleToM 和 MMLU 数据集。

**📈 对比分析**

通过基线与消融（jailbroken）模型对比，测量心智归因得分与 ToM 准确率；结果显示安全微调显著降低心智归因，ToM 表现无显著差异。

**⚠️ 局限性**

局限性包括模型在动物等实体上的心智归因被过度抑制、对宗教讨论的限制、可能的 AI 中心化偏差，以及实验设置对更广泛语境的适用性不足。

---

## 130. Decoding Functional Networks for Visual Categories via GNNs

**arXiv ID:** 2603.28931 | [PDF](https://arxiv.org/pdf/2603.28931v1)

**作者:** Shira Karmi `[一作]`, Tammy Riklin Raviv `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

利用7T NSD fMRI构建功能连接图，训练带正负边的Signed Graph Neural Network，对体育、食物和车辆等视觉类别的功能连接状态进行解码。

**💡 创新点**

创新点在于同时建模正负相关，使用可学习稀疏边缘掩码与梯度输入归因实现全局与类别级解释，并通过多模态标签融合提升分类标签质量。

**🔧 技术方法**

采用Signed GNN、可学习边缘掩码、梯度输入归因、稀疏正则化、t‑SNE可视化、对比学习基线等技术。

**📊 数据集**

使用公开的 7T Natural Scenes Dataset（NSD）fMRI 与 COCO 2017 图像/文本作为输入数据。

**📈 对比分析**

与 GraphCL 对比，Signed GNN 在准确率上从 0.637 提升至 0.78，宏平均 AP 从 0.698 提升至 0.88，跨半球性能相近，证明监督学习在捕捉类别特定 signed 连接模式上更有效。

**⚠️ 局限性**

局限包括仅分析单侧半球、使用静态相关而非时间动态、COCO 标签噪声及多模态融合不完善、对个体差异的解释有限。

---

## 131. Drop the Hierarchy and Roles: How Self-Organizing LLM Agents Outperform Designed Structures

**arXiv ID:** 2603.28990 | [PDF](https://arxiv.org/pdf/2603.28990v1)

**作者:** Victoria Dochkina `[一作]` `[通讯]` (Moscow Institute of Physics and Technology), Victoria Dochkina (Moscow Institute of Physics and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对8种大型语言模型、4–256名代理和8种协调协议进行25,000+任务的系统实验，比较它们在多任务复杂度不同的场景下的协同性能。

**💡 创新点**

发现“内生性悖论”，即仅有极端集中或极端自治的协议都无法获得最佳效果，而一种“顺序”混合协议（固定执行顺序、自由角色选择）能显著提升质量，并揭示强模型与合适协议的乘性效应。

**🔧 技术方法**

使用多种LLM代理（Claude Sonnet 4.6、GPT-5.4、GPT-4o、GPT-4.1-mini、Gemini-3-flash、GigaChat 2 Max、DeepSeek v3.2、GLM-5）与四种协调协议（Coordinator、Sequential、Broadcast、Shared）以及LLM-as-judge评估框架。

**📊 数据集**

采用合成任务集，覆盖四个复杂度层级（L1–L4），共计约25,000次任务运行，用于在可控环境下评估协议与模型的交互效果。

**📈 对比分析**

通过多维度评估（质量 Q、成本 token、时间、风险、任务相关性）进行内部比较；结果显示Sequential协议比Shared高44%（p<0.0001），比Centralized高14%（p<0.001），且从4到256名代理的规模化保持质量稳定，成本仅增长约11%。

**⚠️ 局限性**

局限性包括：评估仅基于LLM裁判，可能存在系统偏差；任务为合成设计，缺乏真实业务验证；Sequential协议具有O(N)延迟，适用大规模时需优化；API版本与速率限制可能影响实验可重复性。

---

## 132. From Consensus to Split Decisions: ABC-Stratified Sentiment in Holocaust Oral Histories

**arXiv ID:** 2603.28913 | [PDF](https://arxiv.org/pdf/2603.28913v1)

**作者:** Daban Q. Jaff `[一作]` `[通讯]`, Daban Q. Jaff

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对犹太大屠杀口述历史中的情感分析进行三模型互评，探索域迁移导致的情感分类不一致。

**💡 创新点**

创新点在于提出ABC共识层次分类，将模型间一致与冲突分为三类，并结合T5情感探测给出情感分布特征，提供可操作的稳定性分层框架。

**🔧 技术方法**

采用三款预训练Transformer情感分类器（SiEBERT、CardiffNLP Twitter‑RoBERTa、NLPTown/bert‑multilingual）以及T5文本转文本情绪模型做描述性探测。

**📊 数据集**

数据来源于CORHOH犹太大屠杀口述历史语料，包含107,305段话（579,013句）。

**📈 对比分析**

通过交叉模型三角测量、Fleiss κ、ABC层级和情绪分布对比，发现模型在中性边界处的分歧最为显著，三模型一致性仅在情绪极端时达到0.78以上。

**⚠️ 局限性**

局限在于未对口述历史进行微调，仅使用离域模型；情绪探测同样是描述性工具；且仅通过统计一致性评估，缺乏人工标注验证。

---

## 133. The Computer System Trail

**arXiv ID:** 2603.28777 | [PDF](https://arxiv.org/pdf/2603.28777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 134. BiMoE: Brain-Inspired Experts for EEG-Dominant Affective State Recognition

**arXiv ID:** 2603.29205 | [PDF](https://arxiv.org/pdf/2603.29205v1)

**作者:** Hongyu Zhu `[一作]` (Chinese Academy of Sciences), Mingsheng Shang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了脑启发式混合专家框架BiMoE，用脑区划分EEG与PPS信号并进行动态融合，实现多模态情绪识别。

**💡 创新点**

创新点包括：①基于脑区的专家划分结合wPLI功能网络与局部卷积的双流编码器；②多尺度大核卷积处理PPS；③动态路由与联合损失平衡专家负载与多样性；④通过SHAP验证模型决策与神经可解释性的一致性。

**🔧 技术方法**

使用技术包括：Mixture of Experts、图卷积网络（GCN）+自注意力、双流GL‑DNet、MS‑LKC多尺度大核卷积、SHAP可解释性、Focal Loss、专家负载/多样性正则化。

**📊 数据集**

使用公开情绪数据集DEAP（32人）和DREAMER（23人），均包含EEG与多模态生理信号。

**📈 对比分析**

采用严格的留一交叉验证，在EEG+PPS模式下平均提升0.87%–5.19%，在EEG单独模式下提升0.69%–5.79%，显著优于经典与最新基线。

**⚠️ 局限性**

局限性包括：仅针对离线数据与二值情绪分类；跨域泛化与实时部署尚未验证；专家数量和划分固定，可能缺乏自适应性；对极端不平衡样本的鲁棒性需进一步提升。

---

## 135. The SCAN Statistical Model Checker

**arXiv ID:** 2603.28794 | [PDF](https://arxiv.org/pdf/2603.28794v1)

**作者:** Enrico Ghiorzi `[一作]` (Università di Genova), Armando Tacchella `[通讯]` (Università di Genova)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

构建了SCAN统计模型检验器的形式基础与架构，并提供对SCXML、JANI等建模语言的支持

**💡 创新点**

提出了将多种模型（时序概率转移系统、程序图、通道系统等）统一映射为时序概率程序图，并实现了适用于大规模机器人系统的抽象统计模型检验框架

**🔧 技术方法**

使用时序概率转移系统、程序图、通道系统、SCXML解析、pMTL / LTL 等时序逻辑及新自适应采样方法

**📊 数据集**

未给出具体实验数据集，主要以机器人控制软件为示例

**📈 对比分析**

通过将模型转化为时序概率程序图并使用多线程采样，验证了在给定置信度和精度下的属性满足率；性能未在本文中给出数值对比

**⚠️ 局限性**

仅支持有限的属性形式（pMTL、LTL子集），非确定性仅按均匀分布处理，且对连续动态缺乏支持；对大型模型的可扩展性仍待验证

---

## 136. Stepper: Stepwise Immersive Scene Generation with Multiview Panoramas

**arXiv ID:** 2603.28980 | [PDF](https://arxiv.org/pdf/2603.28980v1)

**作者:** Felix Wimbauer `[一作]` (Google), Federico Tombari `[通讯]` (Google)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Stepper 框架，实现从文本生成可沉浸式、可漫游的 3D 场景，利用多视角 360° 膜图扩散模型进行步进式场景生成，并结合前向重建与 3D 高斯 splatting 实现实时探索。

**💡 创新点**

创新点包括：① 采用基于立方体映射的多视角 360° 膜图扩散模型，显著提升高分辨率场景合成质量；② 通过步进式生成减少上下文漂移，实现连续、连贯的场景扩展；③ 结合 MapAnything 端到端前向结构光重建和 3D Gaussian Splatting，形成完整的可渲染三维表示；④ 构建 230k 对多视角 360° 膜图的大规模合成数据集，提升模型泛化能力。

**🔧 技术方法**

使用的技术包括：立方体映射（cubemap）表示、Latent Diffusion Model（LDM）多视图扩散、位置编码与遮罩、自动回归步进生成、MapAnything 端到端 SfM、3D Gaussian Splatting（MCMC-GS）以及基于 PyTorch3D 的点云筛选与优化。

**📊 数据集**

使用了从 Infinigen 生成的合成多视角 360° 膜图数据集（约 230,000 对，分辨率 4096×2048，涵盖 5,000 个场景）作为训练集，并准备了包含 6 个 Blender 场景和 10 个 Infinigen 场景的测试集，每个场景包含 9 张视角和对应深度图。

**📈 对比分析**

与 LayerPano3D、WorldExplorer、Matrix-3D 等基准方法进行定量比较，采用 PSNR、SSIM、LPIPS 等 NVS 指标。Stepper 在所有数据集和指标上均优于基线，平均 PSNR 提升 3.3 dB，SSIM 达 0.735，LPIPS 下降至 0.385，显示出更高的视觉质量和连贯性。

**⚠️ 局限性**

局限性：① 采用固定步长（0.25m）对不同场景的通用性有限，无法灵活适应更大/更小的步进；② 主要在合成数据上训练，真实场景的泛化能力和细节还需进一步验证；③ 对初始 360° 膜图的依赖，缺少直接从文本或单张图像生成完整场景的能力；④ 计算成本仍相对较高，尤其是多视角扩散和 3D Gaussian Splatting 的训练与推理。

---

## 137. StepCache: Step-Level Reuse with Lightweight Verification and Selective Patching for LLM Serving

**arXiv ID:** 2603.28795 | [PDF](https://arxiv.org/pdf/2603.28795v1)

**作者:** Azam Nouri `[一作]` `[通讯]` (Lincoln University), Azam Nouri (Lincoln University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 StepCache，一种面向 LLM 服务器的后端无关缓存层，以步骤级别重用模型输出。

**💡 创新点**

创新点在于将输出拆分为有序步骤，使用任务感知轻量级验证，按需区块重绘，并提供跳过重用与确定性回退的策略。

**🔧 技术方法**

使用 SentenceTransformers 生成提示嵌入，FAISS 做近似最近邻检索，Python 层实现步骤分割、验证与 patching，结合 deterministic fallback 计算线性方程解。

**📊 数据集**

实验数据集为自构造的 CPU‑only 微基准，包含 10 个数学线性方程模板与 10 个 JSON 生成模板，每个模板产生 3 个不同扰动变体，随机种子 42–44。

**📈 对比分析**

与基线完整重生成对比，测量平均/中位/95% 分位数延迟、总 token 与每请求 token、正确率；StepCache 平均延迟提升约 3.2×，token 使用降低 24%，最终正确率从 72.5% 提升至 100%。

**⚠️ 局限性**

局限性包括仅验证数学与 JSON 两类任务，对开放式文本缺乏强验证器，可能在语义大幅变化时仍需多次重绘，且缓存完整性易受对抗相似度与投毒攻击影响。

---

## 138. Incentives, Equilibria, and the Limits of Healthcare AI: A Game-Theoretic Perspective

**arXiv ID:** 2603.28825 | [PDF](https://arxiv.org/pdf/2603.28825v1)

**作者:** Ari Ercole `[一作]` `[通讯]` (University of Cambridge), Ari Ercole (University of Cambridge)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文运用最小化的博弈论框架，对医疗 AI 部署中的三类典型干预（降低任务成本、提升可观测性、机制层级激励变更）进行理论分析，指出只有机制层级干预能改变系统层级的均衡行为。

**💡 创新点**

创新点在于用博弈论和机制设计视角，将 AI 干预按对激励结构的影响进行分类，并阐明仅机制层级干预能实现系统级转变，而非单纯的技术改进。

**🔧 技术方法**

主要采用博弈论模型和机制设计理论进行推理与分析。

**📊 数据集**

无数据集，本文为理论性分析。

**📈 对比分析**

由于是概念性阐述，未进行实验对比或性能评估。

**⚠️ 局限性**

局限性包括：模型仅考虑静态 Nash 均衡，简化了实际医疗协调的多重行动和信息不对称；缺乏经验验证；对机制层级干预的实现细节和实施成本未作讨论。

---

## 139. IQRA 2026: Interspeech Challenge on Automatic Assessment Pronunciation for Modern Standard Arabic (MSA)

**arXiv ID:** 2603.29087 | [PDF](https://arxiv.org/pdf/2603.29087v1)

**作者:** Yassine El Kheir `[一作]` (DFKI & Technical University of Berlin), Ahmed Ali `[通讯]` (HUMAIN)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织了IQRA 2026挑战，发布了真实误发音数据 Iqra_Extra_IS26，并对现代标准阿拉伯语（MSA）口音检测与诊断（MDD）进行大规模评测。

**💡 创新点**

结合真实误发音数据、两阶段微调、改进的CTC对齐方法以及首次使用生成式大型音频语言模型，显著提升了 MDD 性能。

**🔧 技术方法**

使用的技术包括多语言 SSL 模型（mHuBERT、Wav2Vec2.0）、CTC+TCN、OTTC、Transformer 解码、LoRA、VITS TTS、Kneser–Ney 语言模型以及 CTC/attention 组合损失。

**📊 数据集**

使用的数据集为 Iqra_train（79h）、Iqra_TTS（52h）、Iqra_Extra_IS26（1.5h）以及评测集 QuranMB.v2（2.5h）。

**📈 对比分析**

评测以 F1 分数为主，主力参赛队伍将基线 0.4414 提升至 0.7201，PER 降至 0.04，Precision 与 Recall 达到较好平衡。

**⚠️ 局限性**

主要局限包括误发音数据规模仍有限、缺乏字母级反馈映射、生成式模型尚未成熟以及对第二语言学习者语料的缺乏。

---

## 140. 3D Architect: An Automated Approach to Three-Dimensional Modeling

**arXiv ID:** 2603.29191 | [PDF](https://arxiv.org/pdf/2603.29191v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 141. Knowledge database development by large language models for countermeasures against viruses and marine toxins

**arXiv ID:** 2603.29149 | [PDF](https://arxiv.org/pdf/2603.29149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 142. MMFace-DiT: A Dual-Stream Diffusion Transformer for High-Fidelity Multimodal Face Generation

**arXiv ID:** 2603.29029 | [PDF](https://arxiv.org/pdf/2603.29029v1)

**作者:** Bharath Krishnamurthy `[一作]` (University of North Texas), Ajita Rattani `[通讯]` (University of North Texas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的双流扩散变换器MMFace-DiT，可从文本、面部分割或草图等多模态输入生成高保真面部图像；

**💡 创新点**

核心创新包括共享RoPE注意力实现深度跨模态融合、动态模态嵌入器允许单模型自适应不同空间条件，以及在VAE潜空间中的端到端训练；

**🔧 技术方法**

技术涵盖扩散变换器（DiT）架构、共享RoPE注意力、AdaLN与门控残差、Rectified Flow Matching与DDPM训练目标、VLM驱动的多模态数据标注；

**📊 数据集**

使用结合CelebA‑HQ与FFHQ的自标注数据集，采用InternVL3生成多样化文本描述，并通过Segformer与U2Net生成面部分割与草图；

**📈 对比分析**

与TediGAN、ControlNet、UAC、CD、DDGI、MM2Latent等六大基线对比，MMFace-DiT在FID、LPIPS、SSIM、CLIP、LLM Score等指标上平均提升40%以上，尤其流匹配版本在FID上比最强基线下降近50%；

**⚠️ 局限性**

局限性包括：仍需大规模VAE潜空间和高分辨率训练资源，模态融合对极端冲突输入的鲁棒性有限，且对非人脸对象或更复杂姿态的适应性待验证。

---

## 143. Improving Ensemble Forecasts of Abnormally Deflecting Tropical Cyclones with Fused Atmosphere-Ocean-Terrain Data

**arXiv ID:** 2603.29200 | [PDF](https://arxiv.org/pdf/2603.29200v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 144. GISTBench: Evaluating LLM User Understanding via Evidence-Based Interest Verification

**arXiv ID:** 2603.29112 | [PDF](https://arxiv.org/pdf/2603.29112v1)

**作者:** Iordanis Fostiropoulos `[一作]` (Meta Recommendation Systems), Xiangjun Fan `[通讯]` (Meta Recommendation Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了GISTBench，一套面向LLM的用户兴趣理解评估框架，能够在无标注的情况下通过行为信号验证生成的兴趣描述是否真实、具体；

**💡 创新点**

创新点在于设计了两项基于证据的度量——Interest Groundedness（IG）与Interest Specificity（IS），并通过兴趣分类标准化与全局或联盟判定器实现无监督验证；

**🔧 技术方法**

技术上结合了结构化LLM推理、LLM判定器（Llama‑3.3‑70B‑Instruct）进行证据过滤与检索测试，并使用层次化兴趣税onomy实现跨模型统一评估；

**📊 数据集**

使用了五个公开RecSys数据集（KuaiRec、MIND、Amazon Music、Goodreads）以及自研的合成短视频数据集，合成数据通过聚类+掩码方式保留真实行为分布；

**📈 对比分析**

评估结果显示最强模型GPT‑OSS‑120B在所有数据集上IG_F1最高（约58–68%），而DeepSeek‑R1在IS上表现最佳；整体发现模型精度高但覆盖率低，提示需提升证据计数与多步推理能力；

**⚠️ 局限性**

局限包括：缺乏冷启动与稀疏数据处理；阈值设计依赖用户调查，需在其他文化或域外调优；联盟回忆对模型集敏感，无法跨实验直接比较；评测仅覆盖英文文本与短视频场景，未考虑多模态与时间演变。

---

## 145. Logging Like Humans for LLMs: Rethinking Logging via Execution and Runtime Feedback

**arXiv ID:** 2603.29122 | [PDF](https://arxiv.org/pdf/2603.29122v1)

**作者:** Xin Wang `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Zishuo Ding `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于运行时反馈的迭代日志生成框架，利用LLM自动生成、编译修复、评估和迭代改进日志语句，以提升日志对后续任务的实用性。

**💡 创新点**

创新点在于将日志生成从静态单次预测转变为闭环执行+反馈的迭代过程，并通过LLM评估日志是否足以满足缺陷定位与修复需求。

**🔧 技术方法**

核心技术包括LLM生成日志语句、编译错误反馈修复、LLM评估日志充分性以及基于评估结果的日志修订模块，整体实现了自动化的运行时闭环。

**📊 数据集**

使用来自Defects4J的311个直接调试样本和225个间接调试样本构建两套数据集，分别对应源代码可见与不可见的两种调试场景。

**📈 对比分析**

与SCLogger、UniLog、LANCE等基线对比，本文框架在直接调试中F1达0.520、修复率31.19%，在间接调试中F1达0.408，均明显优于所有对手；此外在不同LLM上表现稳健。

**⚠️ 局限性**

主要局限包括需要离线执行产生运行时反馈、对Java项目的依赖、仅评估调试任务而非性能或异常监测等其他日志应用场景。

---

## 146. Zero-shot Cross-domain Knowledge Distillation: A Case study on YouTube Music

**arXiv ID:** 2603.28994 | [PDF](https://arxiv.org/pdf/2603.28994v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 147. Modernizing Ground Truth: Four Shifts Toward Improving Reliability and Validity in AI in Education

**arXiv ID:** 2603.29141 | [PDF](https://arxiv.org/pdf/2603.29141v1)

**作者:** Danielle R. Thomas `[一作]` (Carnegie Mellon University), René F. Kizilcec `[通讯]` (Cornell University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出四个改进 AI 教育标签可靠性与有效性的实践转变：把 IRR 从阈值门槛转为诊断工具，要求透明报告、专家共评与多标签标注；为 LLM 注释提供风险审计与验证工作流；并补充多层次有效性与闭环学习效果证据；通过案例说明这些方法能更好识别偏差与模糊性。

**💡 创新点**

创新点在于将传统 IRR 视作诊断信号而非硬性门槛，并结合 LLM 注释的风险管理、透明报告标准、专家协同与多标签标注等四个实用转变，推动标签质量评价从纯统计转向多维有效性验证与教育干预闭环。

**🔧 技术方法**

采用了 IRR 统计（Cohen κ、Fleiss κ、Krippendorff α 等）、LLM 辅助注释与自动审计、验证工作流（如“验证者模型”与“模拟性评估”）、多标签标注框架、预测有效性分析以及闭环学习效果评估等技术。

**📊 数据集**

以多模态辅导数据为例，使用来自 National Tutoring Observatory、SafeInsights、SEERNet 等公开数据集中的音频、视频、文字与传感器记录进行案例研究；并对 LLM 与人类注释者的标注进行对比。

**📈 对比分析**

文章未给出系统的实验对比指标，而是通过案例展示：传统单阈值 IRR 通常低或受“时间链接问题”影响，使用本方法后可定位误差源、降低偏差并为后续模型训练提供更可靠标签；示例中预测有效性与闭环评估显示改进后标签更能预测学习成效。

**⚠️ 局限性**

限制包括：缺乏统一、可复现的 IRR 与有效性评估标准；高推断任务本身难以实现高 IRR；LLM 注释仍易受偏差与循环验证风险；多标签与专家共评流程增加工作量，尚未在大规模数据上系统验证其效果。

---

## 148. Mitigating Temporal Blindness in Kubernetes Autoscaling: An Attention-Double-LSTM Framework

**arXiv ID:** 2603.28790 | [PDF](https://arxiv.org/pdf/2603.28790v1)

**作者:** Faraz Shaikh `[一作]` (University of Perugia), Mauro Femminella `[通讯]` (University of Perugia)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种基于注意力增强双层LSTM的PPO控制器，用于在边缘Kubernetes环境中提前预测并自动扩缩容。

**💡 创新点**

创新点在于将深度注意力机制与双层LSTM嵌入到DRL策略中，解决传统单层或无记忆模型的时间盲区，提升预测与控制协同。

**🔧 技术方法**

采用Proximal Policy Optimization、双层堆叠LSTM、软注意力机制、POMDP建模以及多维离散动作的Kubernetes HPA接口。

**📊 数据集**

使用Azure Functions的真实调用轨迹作为工作负载数据集。

**📈 对比分析**

通过与标准HPA、单层LSTM-PPO以及Double DQN三种基线对比，实验显示90th分位延迟降低约29%、复制器抖动减少约39%，并在SLO合规性和CPU利用率上优于基线。

**⚠️ 局限性**

主要限制包括推理时延在资源受限的边缘节点上较高、实验仅覆盖单一微服务、未考虑多服务链的相互影响及真实多租户环境噪声。

---

## 149. Known Intents, New Combinations: Clause-Factorized Decoding for Compositional Multi-Intent Detection

**arXiv ID:** 2603.28929 | [PDF](https://arxiv.org/pdf/2603.28929v1)

**作者:** Abhilash Nandy `[一作]` `[通讯]` (Microsoft Research India), Abhilash Nandy (Microsoft Research India)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个用于评估多意图检测组合泛化的受控基准CoMIX-Shift，并设计了仅用单意图训练的轻量级分句解码器ClauseCompose；

**💡 创新点**

创新点在于通过明确的意图对hold-out、话语模式迁移、长噪声包装、模板缺失和零射三元组等拆分，系统化地检验组合泛化能力，并展示分句化解码在这些场景下显著优于全句模型；

**🔧 技术方法**

技术上采用了简易编码器+单意图分类器的分句化解码器、全句多标签+计数头的基线、以及微调的tiny BERT多标签模型；

**📊 数据集**

数据集主要包括10个原子意图的合成CoMIX-Shift（包含单句模板、组合句子、话语变体等），以及一个240条手工编写的SNIPS式组合测试集；

**📈 对比分析**

评估方式为精确匹配（EM）及微/宏F1，结果显示ClauseCompose在未见意图对、话语迁移、长噪声包装、模板缺失以及零射三元组等拆分上分别达到95.7%、93.9%、62.5%、49.8%和91.1%，明显优于WholeMultiLabel和tiny BERT基线；

**⚠️ 局限性**

局限性包括：基准为合成数据，真实用户语言更杂乱；使用手工话语标记器；仅评估意图集合预测，未涉及槽填充；手工测试集规模较小，缺乏大规模真实混合意图数据。

---

## 150. Privacy Guard & Token Parsimony by Prompt and Context Handling and LLM Routing

**arXiv ID:** 2603.28972 | [PDF](https://arxiv.org/pdf/2603.28972v1)

**作者:** Alessio Langiu `[一作]` `[通讯]` (National Research Council of Italy), Alessio Langiu (National Research Council of Italy)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于本地小型语言模型（SLM）的“Privacy Guard”，通过语义压缩与自动提示优化（APO）实现对LLM的上下文与隐私双重管理，减少运营成本并消除敏感信息泄露。

**💡 创新点**

核心创新在于将上下文管理与隐私控制合一，形成“不可分离原理”，通过双重机制（语义压缩+零泄漏路由）同时实现Token Parsimony与Zero Leakage，并引入LIFO上下文压缩以抑制长期对话的泄露风险。

**🔧 技术方法**

采用本地SLM（7B–70B参数）、自动提示优化（APO）、多层信任路由（Tier0–3）、正则表达式或GLiNER等确定性PII扫描、LIFO上下文堆栈、LLM-as-a-Judge评估等技术。

**📊 数据集**

在1000条样本的自生成数据集上（包含60条个人秘密和80条机构秘密，覆盖Lazy/Expert与Personal/Institutional四种情境），并对比多种LLM模型（Qwen 2.5 7B、30B、70B等）进行实验。

**📈 对比分析**

与传统全云路由对比，Privacy Guard在Lazy/Personal/Institutional四种组合中实现约45–48% OpEx下降、100%个人秘密零泄漏、85%答复质量优于基线；对比不同参数规模的LLM显示仅靠大模型并不能解决“Lost in the middle”泄漏，需结合确定性过滤。

**⚠️ 局限性**

局限性包括：单一SLM在处理大规模、非结构化文本时仍易泄漏（尤其Lazy/Institutional）；对高安全性需求仍需额外的确定性PII扫描或更大模型；系统对硬件依赖（如GPU内存）仍有限；评估多基于自生成数据，缺乏真实企业场景验证。

---

## 151. Improving Efficiency of GPU Kernel Optimization Agents using a Domain-Specific Language and Speed-of-Light Guidance

**arXiv ID:** 2603.29010 | [PDF](https://arxiv.org/pdf/2603.29010v1)

**作者:** Siva Kumar Sastry Hari `[一作]` (NVIDIA), Christos Kozyrakis `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用LLM代理对GPU核进行迭代优化，并通过紧凑DSL（μCUTLASS）与速度极限（SOL）引导显著提升优化效率。

**💡 创新点**

创新点在于提出可在上下文学习的紧凑DSL，剥离低层实现细节；以及基于首原理的SOL分析，为LLM提供头房估计与搜索边界，二者协同提升性能与资源利用率。

**🔧 技术方法**

利用GPT‑5系列大型语言模型、μCUTLASS DSL编译器、SOL分析框架及自动化工具链进行实验。

**📊 数据集**

实验基于KernelBench 59个问题（涵盖Transformer、SSM等现代LLM工作负载）。

**📈 对比分析**

与传统直接生成CUDA、无指导的LLM代理以及Sakana AI的进化搜索方法对比，μCUTLASS+SOL在GPT‑5‑mini、GPT‑5、GPT‑5.2上分别实现1.56×、2.07×、2.79×的geomean速度提升，并在相同预算下节省19–43%的tokens；在速度极限指导下，效率提升可达1.68×。

**⚠️ 局限性**

局限包括DSL覆盖面有限，最强模型提升相对有限；SOL指导仍需离线审核，游戏检测机制不够自动化；以及在不同GPU或软件栈上的泛化性待进一步验证。

---

## 152. AI prediction leads people to forgo guaranteed rewards

**arXiv ID:** 2603.28944 | [PDF](https://arxiv.org/pdf/2603.28944v1)

**作者:** Aoi Naito `[一作]` (Carnegie Mellon University), Hirokazu Shirado `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过在1,305名参与者的行为实验中，研究了AI预测是否会改变人们的决策方式，并发现AI预测导致人们放弃保证收益。

**💡 创新点**

首次揭示AI预测不仅影响决策结果，还能通过“预测绑定”机制改变人们的决策过程，表明预测权威本身能重塑决策逻辑。

**🔧 技术方法**

采用新康姆悖论式的两选任务，设计AI预测与随机预测对照，并使用交互式与非交互式AI界面来探究效应。

**📊 数据集**

使用来自四项预注册在线实验的匿名参与者数据，共计1,305人，其中包含对不同预测框架、交互方式以及情景模拟的实验记录。

**📈 对比分析**

与随机对照相比，AI预测显著提升“一盒”选择的几率，OR=3.39（95%CI 2.45–4.70），导致收益下降10.7%–42.9%，且效应在多种AI呈现方式和情景下保持一致。

**⚠️ 局限性**

局限在实验环境的简化（无实际AI预测、无社会互动、无长期反馈），且结果尚未验证在真实复杂决策场景中的可推广性。

---

## 153. SkillTester: Benchmarking Utility and Security of Agent Skills

**arXiv ID:** 2603.28815 | [PDF](https://arxiv.org/pdf/2603.28815v1)

**作者:** Leye Wang `[一作]` (Key Laboratory of High Confidence Software Technologies, Ministry of Education, Peking University), Anjie Xu `[通讯]` (Key Laboratory of High Confidence Software Technologies, Ministry of Education, Peking University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个评估工具，用于比较代理技能的效用和安全性，并生成可公开发布的效用分数、安全分数和状态标签。

**💡 创新点**

创新点在于引入“比较效用原则”，通过匹配无技能基线进行对比评估，并采用“用户面向简洁性原则”将内部指标压缩为三大公开指标，同时提供分层安全标签。

**🔧 技术方法**

采用了基于对比执行的评估框架、令牌和耗时测量、控制安全探针（异常行为、权限边界、敏感数据保护）以及数值映射公式计算效用和安全分数。

**📊 数据集**

使用了来自公开技能仓库（ClawHub、skills.sh）的约4000个技能，并为每个技能自动生成常规与边缘功能任务，以构成评估任务集。

**📈 对比分析**

通过对每个任务的无技能与有技能执行结果进行对比计算效用，安全性通过探针通过率计算；效用分数在0-100范围内，安全阈值80分为警戒级别；实验表明工具能够量化技能价值与风险。

**⚠️ 局限性**

限制包括：仅评估一次性任务，安全探针覆盖有限（仅三类），未覆盖深层次威胁，基线与技能执行在同一环境下，且缺乏正式验证与长期回归监控。

---

## 154. Parallel Gauss-Jordan Elimination and System Reduction for Efficient Circuit Simulation

**arXiv ID:** 2603.28792 | [PDF](https://arxiv.org/pdf/2603.28792v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 155. Predictor-Based Output-Feedback Control of Linear Systems with Time-Varying Input and Measurement Delays via Neural-Approximated Prediction Horizons

**arXiv ID:** 2603.29117 | [PDF](https://arxiv.org/pdf/2603.29117v1)

**作者:** Luke Bhan `[一作]` (University of California San Diego), Yuanyuan Shi `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对线性时变延迟系统，提出一种输出反馈的预测器设计，并针对预测器所需的逆延迟映射给出了两种近似方法（数值积分法和神经算子法），同时给出了误差理论与闭环指数稳定性证明。

**💡 创新点**

创新点包括：1）将逆延迟映射视为算子逼近问题，并证明其在紧集上Lipschitz连续；2）给出数值积分与神经算子两种逼近的严格误差界；3）将逼近误差融入闭环稳定性分析，得到可行的误差容限；4）首次在存在输入与测量双重时变延迟的情形下实现完整的输出反馈预测器。

**🔧 技术方法**

主要技术手段为：算子学习（Fourier Neural Operator）、数值积分（显式Euler及RK4）、Lyapunov后退变换、传输PDE建模与能量估计。

**📊 数据集**

数据集由2000个随机生成的延迟函数（参数a、b、α、ω、φ均从指定区间均匀采样）构成，在满足理论假设后计算高精度逆延迟值作为监督信号，用于训练FNO。

**📈 对比分析**

实验比较了Euler、RK4与FNO三种逆延迟计算方法。在1000个随机样例上：Euler约11.98 ms/样例，RK4约42.5 ms，FNO约2.01 ms，FNO相较于Euler提升约6倍、相较于RK4约21倍。FNO在速度上领先，但在满足隐式关系的精度上略逊于数值积分方法；在控制性能上三者均能实现系统稳定，误差随逼近精度下降而提升。

**⚠️ 局限性**

局限性包括：1）理论与实验均基于延迟变化缓慢且满足1-˙D>0的假设；2）数值积分误差随时间窗口增长而放大；3）神经算子逼近需要离线大规模训练，且在超出训练分布或长期时间域时精度下降；4）实现仍需GPU支持，实时部署受限。

---

## 156. TrajectoryMover: Generative Movement of Object Trajectories in Videos

**arXiv ID:** 2603.29092 | [PDF](https://arxiv.org/pdf/2603.29092v1)

**作者:** Kiran Chhatre `[一作]` (KTH Royal Institute of Technology), Paul Guerrero `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种可将视频中对象的3D运动轨迹平移到新位置的生成模型TrajectoryMover，提供简化的编辑方式；

**💡 创新点**

创新点包括：①定义了轨迹平移任务；②构建大规模合成对视频数据生成管线Trajectory Atlas；③使用交替训练保留原始生成器先验；④以边框控制实现直观交互；

**🔧 技术方法**

采用生成式视频扩散模型Wan2.1-T2V-1.3B+VAE，结合ControlNet式条件机制、RoPE编码、物理模拟(Bullet)和Blender渲染；

**📊 数据集**

使用生成的约21k对合成视频（Trajectory Atlas），包含多种轨迹类型（抛、滚、拖等）、多样化场景（Evermotion室内场景）和对象（Objaverse + primitives）；

**📈 对比分析**

在与5个基线（ATI、DaS、VACE、I2VEdit、SFM）的SSIM_bg、DINO_fg、IoU_traj以及用户研究可行性指标上进行对比，TrajectoryMover在所有指标上均表现最佳，背景保真0.92、前景相似0.45、轨迹符合0.27，用户评测优先级最高；

**⚠️ 局限性**

局限性：轨迹精度仍受限（IoU_traj仅0.27），在保持对象身份和背景一致性方面需权衡；目前对真实视频的泛化能力有限，需进一步改进训练策略与数据覆盖。

---

## 157. From categorized neural architectures to subexponential proof theory

**arXiv ID:** 2603.28946 | [PDF](https://arxiv.org/pdf/2603.28946v1)

**作者:** Carlos Ramírez Ovalle `[一作]` `[通讯]` (Pontificia Universidad Javeriana Cali), Carlos Ramírez Ovalle (Pontificia Universidad Javeriana Cali)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文从带区域标签的神经架构族出发，构造其分类化的架构范畴，提取子指数签名，并基于该签名定义张量式序列计算，并证明其剪切消除与在原始分类化架构中的语义正确性。

**💡 创新点**

创新点在于逻辑不是预先假定的，而是直接从分类化架构的资源敏感许可（复制、丢弃、区间协商）中读取子指数结构，形成一种“架构→分类→逻辑”的新颖流程；并给出了完整的提取与证明一致性证明。

**🔧 技术方法**

技术上使用了范畴论（有限积、对称厄米多模张量范畴）、子指数线性逻辑语义、以及对照证明（剪切消除与解释一致性）等。

**📊 数据集**

该工作为理论性研究，未使用具体实验数据集。

**📈 对比分析**

由于未做实验，本文未涉及方法比较或性能评估；所示的是理论证明与示例构造。

**⚠️ 局限性**

局限性包括：仅覆盖张量与单位的子指数片段，未处理线性蕴含或内部同态；需要进一步丰富范畴结构以支持完整逻辑；在实际深度学习框架中的应用仍待验证。

---

## 158. Comprehensive Plugin-Based Monitoring of Nexflow Workflow Executions

**arXiv ID:** 2603.28783 | [PDF](https://arxiv.org/pdf/2603.28783v1)

**作者:** Sami Kharma `[一作]` (Zuse Institute Berlin), Florian Schintke `[通讯]` (Zuse Institute Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

实现了一个可插拔的 Nextflow 监控插件，能够在不修改工作流代码的情况下实时收集任务执行细节并生成物理执行图。

**💡 创新点**

创新点在于利用 Nextflow 的插件机制与 JVM 反射实现细粒度监控，并提供可选的节点级监控补丁，突破了 Nextflow 内部数据对硬件信息暴露不足的限制。

**🔧 技术方法**

使用技术包括 Groovy 编写插件、Nextflow 21.10 插件接口、JVM 反射、Collectl 等节点监控工具、以及 JSON 输出格式。

**📊 数据集**

使用 nf‑core 社区公开的 rnaseq 工作流作为真实工作流进行监控，演示了六节点执行环境下的任务分配与监控效果。

**📈 对比分析**

与 Nextflow 原生 wf‑instances 监控相比，插件提供更详细的任务时间、容器镜像、工作目录、文件元数据等信息；虽然本文未给出量化的性能指标，但展示了更丰富的监控视图，能更有效定位瓶颈并优化资源调度。

**⚠️ 局限性**

局限性包括：需要对执行节点进行补丁注入才能获取硬件层面数据；插件在大规模集群中的性能开销未进行深入评估；对某些自定义任务或特殊容器环境的兼容性仍需进一步验证。

---

## 159. Time is Not Compute: Scaling Laws for Wall-Clock Constrained Training on Consumer GPUs

**arXiv ID:** 2603.28823 | [PDF](https://arxiv.org/pdf/2603.28823v1)

**作者:** Yi Liu `[一作]` `[通讯]` (Independent Researcher), Yi Liu (Independent Researcher)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在消费级GPU上进行多规模训练实验，提出以壁钟时间为自变量的时间约束尺度律，发现最优模型规模随时间呈幂律增长，指数约0.60。

**💡 创新点**

创新点在于将传统基于计算量的Chinchilla尺度律转为基于实际训练时间的尺度律，并揭示了短期计算瓶颈与长期数据瓶颈共同导致的双重U形曲线。

**🔧 技术方法**

采用Decoder‑Only Transformer（密集注意力、FlashAttention v2、BF16精度）以及统一的深度-宽度比例控制模型尺寸，并使用单GPU 8×RTX 4090进行实验。

**📊 数据集**

使用FineWeb‑Edu数据集（约48M token），该小数据集使得过拟合与数据重复效果显著，方便观察U形曲线。

**📈 对比分析**

与Chinchilla计算最优指数0.50进行对比，并与MoE、RetNet、GLA、RWKV等架构在同一硬件上进行对比；结果显示Dense Transformer在同时间约束下性能最优，最优规模随时间按 N^*≈14.2·t^0.595 递增，最佳损失随时间下降 L^*≈1.22·t^-0.061。

**⚠️ 局限性**

局限性包括仅在RTX 4090单GPU上测试；数据集过小，限制了对大规模数据场景的推断；仅评估了单一Dense Transformer架构；种子覆盖有限，未在多GPU环境下验证长时间预算下的规模扩展。

---

## 160. A Neural Tension Operator for Curve Subdivision across Constant Curvature Geometries

**arXiv ID:** 2603.28937 | [PDF](https://arxiv.org/pdf/2603.28937v1)

**作者:** Hassan Ugail `[一作]` (University of Bradford), Newton Howard `[通讯]` (Rochester Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种共享的学习张力预测器，用于在欧氏、球面和双曲平面三种常曲率空间中实现自适应插值细分。

**💡 创新点**

创新点在于：① 用单一网络结合可学习的几何嵌入实现跨空间统一；② 通过 Sigmoid 输出保证 G¹ 安全插入；③ 提供结构安全性、适应性动机与条件 C¹ 收敛理论；④ 在细分过程中实现边缘级别自适应张力，而非全局参数。

**🔧 技术方法**

技术手段包括残差 MLP（约 140k 参数）、几何嵌入表、sigmoid 缩放输出头、Chamfer 距离、角度平滑与弯曲能量损失，以及等距旋转一致性正则化。

**📊 数据集**

使用人工合成数据集：每种几何 400 条曲线（总 1200 条），12 点闭合控制多边形、1000 点真值曲线；另测试 ISS 地面轨迹作为 OOD 示例。

**📈 对比分析**

与四种基线（四点/六点固定张力、对数-指数曲面提升、最佳固定张力、线性张力启发式）对比；学习预测器在所有三种几何中取得最低弯曲能量和 G¹ 粗糙度，虽然在均值最近邻距离上略逊于曲面提升，但在平滑度上显著优于所有固定张力方法，且 OOD 实验保持了这一优势。

**⚠️ 局限性**

局限包括：仅处理闭合多边形、仅 12 点控制点、缺乏开放或可变顶点数实验、真实世界数据覆盖有限、收敛理论为条件性且需后验 Lipschitz 检验、暖启动参数固定等。

---

## 161. Wherefore Art Thou? Provenance-Guided Automatic Online Debugging with Lumos

**arXiv ID:** 2603.29013 | [PDF](https://arxiv.org/pdf/2603.29013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 162. Interactive Evidence Maps for Visualizing and Understanding Systematic Reviews

**arXiv ID:** 2603.28802 | [PDF](https://arxiv.org/pdf/2603.28802v1)

**作者:** Aditi Mallavarapu `[一作]` (North Carolina State University), Jessica R. Gladstone `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用大型语言模型提取主题并构建交互式证据地图，以便研究者在系统综述中动态探索、过滤和分析研究数据。

**💡 创新点**

创新点在于将 LLM 驱动的主题建模与交互式可视化相结合，提供多层级、可交互的知识图谱，显著提升传统静态综述的透明度与可探索性。

**🔧 技术方法**

技术包括 Claude（Anthropic LLM）进行主题提取、D3.js 与 React 构建交互式可视化界面，以及基于 CSV 的数据处理管道。

**📊 数据集**

数据集来自一篇包含 112 篇研究的 K‑12 教育教学代理系统综述，涵盖学习主题、代理类型、年级、研究目的等系统编码变量。

**📈 对比分析**

与传统表格/森林图等静态展示方式相比，交互式证据地图能够实时过滤、显示主题与编码属性的交叉信息，并揭示隐藏的研究模式和空白；目前仅以可视化效果与案例对比为主，未给出量化性能指标。

**⚠️ 局限性**

局限包括：对大规模数据（>500–1000 篇）时的可扩展性与响应速度未知；仅使用摘要进行主题聚类导致信息不足；单标签分配缺乏置信度或多标签展示；LLM 生成的主题不稳定，难以复现；跨领域适用性尚需验证。

---

## 163. Beyond Localization: Recoverable Headroom and Residual Frontier in Repository-Level RAG-APR

**arXiv ID:** 2603.29067 | [PDF](https://arxiv.org/pdf/2603.29067v1)

**作者:** Pengtao Zhao `[一作]` (University of Melbourne), Haoye Tian `[通讯]` (Aalto University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并拆解仓库级检索增强生成自动程序修复（RAG-APR）系统在定位后仍能获得的改进空间与未解决前沿。

**💡 创新点**

提出一套四部分控制实验框架，系统量化定位后搜索、证据利用、接口设计对修复效果的残余贡献，并揭示仍存的可恢复空间与残留难点。

**🔧 技术方法**

采用Oracle定位、有限候选池采样、固定接口下的上下文增补、同标记填充对照、硬负样本、通用包装器验证以及提示级融合等技术手段进行分层评估。

**📊 数据集**

在Defects4J等主流仓库级bug benchmark 上对三个代表性RAG-APR范式进行实验。

**📈 对比分析**

与原始系统在同一修复集合上对比，发现定位提升后成功率仍低于50%，候选多样性在10条候选池内可提升但快速饱和，固定接口下增补上下文可额外提升约6个实例，整体仍未能完全弥补差距。

**⚠️ 局限性**

实验仅覆盖三种现有RAG-APR范式和固定协议，未检验更大规模或更灵活的上下文构造方法，且仍无法完全解决所有残留前沿问题。

---

## 164. An Economic Framework for Generative Engines: Advertising or Subscription?

**arXiv ID:** 2603.29071 | [PDF](https://arxiv.org/pdf/2603.29071v1)

**作者:** Luyang Zhang `[一作]` (Carnegie Mellon University), Chenyan Xiong `[通讯]` (Carnegie Mellon University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个动态Stackelberg框架，用来分析生成式引擎在广告和订阅两种商业模式之间的最优选择，并推导了阈值型最优策略；

**💡 创新点**

创新点在于将短期广告收益与长期用户留存与订阅转化耦合在动态决策中，证明了最优策略满足“切点”准则，并给出对广告敏感度、查询盈利度、推理成本、外部竞争等因素的比较静态推断；

**🔧 技术方法**

使用离散选择模型（logit）刻画用户是否参与，贝尔曼方程求解动态最优策略；对政策进行数值仿真；

**📊 数据集**

没有使用真实数据集，而是基于设定的分布（用户类型、查询分布、推理成本等）进行合成仿真；

**📈 对比分析**

与四种基线策略（最优DP、一步贪心、始终展示广告、始终不展示广告）进行对比，结果显示最优DP在长期收益、活跃用户、订阅转化率和广告曝光率上均优于其他策略；

**⚠️ 局限性**

局限包括：只考虑单一生成式引擎的垄断设置，未进行实证验证；模型假设i.i.d.查询、固定折扣因子、无容量/延迟限制；竞争对手策略被简化为单一外部选择效用，未考虑多方博弈或市场动态变化。

---

## 165. PolarQuant: Optimal Gaussian Weight Quantization via Hadamard Rotation for LLM Compression

**arXiv ID:** 2603.29078 | [PDF](https://arxiv.org/pdf/2603.29078v1)

**作者:** Caio Vicentino `[一作]` `[通讯]`, Caio Vicentino

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PolarQuant，一种利用 Hadamard 旋转将权重块正态化后进行 Lloyd–Max 量化的后训练量化方法，可实现近无损压缩并可作为 INT4 量化前处理。

**💡 创新点**

证明 Hadamard 旋转即可将权重块近似为 i.i.d. 标准正态分布，使得即便使用简单的 Lloyd–Max 量化也能达到近无损效果；并显示该旋转占 98% 性能提升。

**🔧 技术方法**

块级归一化、Walsh–Hadamard 旋转、Lloyd–Max 最优均方误差量化、AWQ 结合、torchao INT4 后处理以及快速 Hadamard 变换实现。

**📊 数据集**

主要使用 Qwen3.5‑9B 模型，WikiText‑2 数据集评估困惑度，并在 Apple Silicon 的 Mac mini M4 上验证跨平台性能。

**📈 对比分析**

与 FP16、torchao INT4、BitsAndBytes NF4 等基线对比，PolarQuant Q5+torchao INT4 在 6.56 的 PPL（相对 FP16 仅 +0.19）下保持 43.1 tok/s、6.5 GB VRAM；PolarQuant Q5 单独 6.39 PPL、45.9 tok/s；在 M4 上实现 19.7 tok/s、4.8 GB。

**⚠️ 局限性**

假设旋转后的块近似正态分布，可能不适用于所有架构；未利用块间相关性；在低位宽（≤3 bit）下 Lloyd–Max 贡献有限，需进一步探索更高效的向量量化或激活量化方案。

---

## 166. A Pontryagin Method of Model-based Reinforcement Learning via Hamiltonian Actor-Critic

**arXiv ID:** 2603.28971 | [PDF](https://arxiv.org/pdf/2603.28971v1)

**作者:** Chengyang Gu `[一作]` (HKUST), Yize Chen `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Hamiltonian Actor‑Critic (HAC)，一种在确定性动力学下利用Pontryagin最大原理中的Hamiltonian直接取代传统Actor‑Critic中价值函数的模型基强化学习方法。

**💡 创新点**

创新点在于：①用Hamiltonian替代显式价值网络，消除价值逼近误差；②设计闭式Jacobian计算加速成本状态求解；③证明在一定条件下HAC的最大价值估计误差比MVE更小，提供收敛保证。

**🔧 技术方法**

技术包括：模型基强化学习、Pontryagin最大原理、Hamiltonian优化、K步虚拟滚动、基于ReLU网络的闭式Jacobian、软更新目标策略、在线与离线训练框架。

**📊 数据集**

使用了多种连续控制基准：线性二次调节(LQR)、摆杆(Pendulum)、山车(MountainCar)、MuJoCo 水手(Swimmer)、马踢(Mopper)等，涵盖短期与长期、在线与离线场景。

**📈 对比分析**

与DDPG、SAC、MVE-DDPG、IQL、SAC‑Off、MOPO等基准对比，HAC在在线任务中取得更高累计奖励、收敛更快；在离线任务中样本效率提升约25%–30%，整体表现与最佳模型基方法持平或更优，并在初始状态偏移的OOD测试中表现出更强鲁棒性。

**⚠️ 局限性**

局限性在于：仅适用于确定性动力学；对模型误差仍有一定敏感性；K步滚动的深度需手动调节；对复杂非线性或高维随机系统的推广仍需进一步研究。

---

## 167. Evaluating a Data-Driven Redesign Process for Intelligent Tutoring Systems

**arXiv ID:** 2603.29094 | [PDF](https://arxiv.org/pdf/2603.29094v1)

**作者:** Qianru Lyu `[一作]` (Carnegie Mellon University), Vincent Aleven `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对中学数学 ITS 的四个学习单元应用了 Huang 等人提出的数据驱动重设计流程，并在课堂实验中评估其效果

**💡 创新点**

首次将该流程应用于未被选为“易改进”单元的内容，验证其通用性，并结合快速前进（mastered step skipping）与部分任务实现的可扩展练习机制

**🔧 技术方法**

使用知识成分（KC）模型重构、聚焦练习、适应性题目选择算法，并通过 Additive Factors Model（AFM）和 Bayesian Knowledge Tracing（BKT）评估学生知识

**📊 数据集**

基于 MathTutor 的日志数据（103 名学生，22,529 条交互，50.19 小时）以及 123 名学生的课堂实验数据（DataShop 数据集 #5868、#6283）

**📈 对比分析**

采用随机交叉设计比较原始与重设计 Tutor 的学习增益、练习机会、时间投入与知识掌握；结果显示学习增益无显著差异，但重设计 Tutor 在时间投入、练习步骤数和已掌握知识总量上均有提升

**⚠️ 局限性**

主要局限为练习时间短、实验覆盖新技能的测验不足，导致难以显著体现学习增益；样本量与实验时间限制了对重设计效果的全面评估

---

## 168. Foundations of Polar Linear Algebra

**arXiv ID:** 2603.28939 | [PDF](https://arxiv.org/pdf/2603.28939v1)

**作者:** Giovanni Guasti `[一作]` `[通讯]`, Giovanni Guasti

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Polar Linear Algebra框架，将线性径向与周期角度分离用于运算符学习，并在MNIST上验证其可训练性。

**💡 创新点**

创新点在于把运算符拆解为极坐标形式的光谱模式，利用自伴随式光谱约束提升稳定性与收敛速度，并实现参数压缩与计算复杂度下降。

**🔧 技术方法**

使用极坐标线性代数理论、光谱分析、自伴随约束以及深度学习的实现框架。

**📊 数据集**

采用MNIST手写数字分类数据集进行实验。

**📈 对比分析**

与传统空间域网络对比，Polar和全光谱运算符在相同参数量下保持或提升精度，同时训练更稳定，收敛更快。

**⚠️ 局限性**

局限在于目前仅在简单分类任务（MNIST）上测试，缺乏在更复杂连续体或 PDE 问题中的验证。

---

## 169. A Latent Risk-Aware Machine Learning Approach for Predicting Operational Success in Clinical Trials based on TrialsBank

**arXiv ID:** 2603.29041 | [PDF](https://arxiv.org/pdf/2603.29041v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 170. The impact of multi-agent debate protocols on debate quality: a controlled case study

**arXiv ID:** 2603.28813 | [PDF](https://arxiv.org/pdf/2603.28813v1)

**作者:** Ramtin Zargari Marandi `[一作]` `[通讯]`, Ramtin Zargari Marandi

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在宏观经济事件分析中，系统性比较了三种多代理辩论协议（WR、CR、RA‑CR）以及无互动基线，评估协议对交互性和收敛性的影响。

**💡 创新点**

提出了Rank‑Adaptive Cross‑Round（RA‑CR）协议，利用评判模型动态重新排序代理顺序并在后轮屏蔽表现最差的代理；同时构建了面向交互与收敛的简洁评估框架。

**🔧 技术方法**

采用角色化LLM代理（Llama、Qwen2、GPT‑OSS）与评判模型（Llama 7.2B）实现生成、候选重排序与自适应调度；使用SBERT挑选事件子集，计算PRR、AD、CF等指标，配合置换检验与自助置信区间。

**📊 数据集**

基于FRED（美国联邦储备经济数据）121条月度通胀系列，附加的重大事件与通胀关系注释，随后用SBERT选取Top‑20事件。

**📈 对比分析**

在匹配提示、解码参数和模型分配的受控重复测量设计下比较四种协议。结果显示WR在交互指标PRR上最高，RA‑CR在收敛指标CF上最高，三者在多样性AD上无显著差异；RA‑CR显著优于WR、CR及NI在CF上，WR在PRR上优于RA‑CR，体现交互与收敛的权衡。

**⚠️ 局限性**

仅在单一宏观经济域、少量模型与评判者角色下验证；评判模型的双重作用可能混合协议与评估效应；缺乏人工质量评估；结果对其他领域和更大模型的外推性有限。

---

## 171. AEC-Bench: A Multimodal Benchmark for Agentic Systems in Architecture, Engineering, and Construction

**arXiv ID:** 2603.29199 | [PDF](https://arxiv.org/pdf/2603.29199v1)

**作者:** Harsh Mankodiya `[一作]` (Nomic), Andriy Mulyar `[通讯]` (Nomic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了AEC-Bench，一个针对建筑、工程与施工（AEC）领域多模态任务的代理评测基准，涵盖从单页推理到跨文件协调的三类任务；

**💡 创新点**

创新点在于：①将真实建筑施工图与规范结合生成结构化任务实例；②设计了多尺度任务范畴（Intra‑Sheet、Intra‑Drawing、Intra‑Project）；③通过增设Nomic专用解析工具评估结构化表示对代理性能的影响；

**🔧 技术方法**

采用多模态代理框架（Harbor harness）与通用编程型代理（Codex、Claude Code）结合，利用PDF解析、文本布局、几何关系抽取等技术，并可调用Bash、图像渲染等工具；

**📊 数据集**

构建了包含196个实例、9种任务类型、3个领域（建筑、结构、机电等）的真实公开施工图文档数据集；

**📈 对比分析**

在基准上对比了两类代理体系的基线性能，分别在基础H与增强H+两套环境下评估，结果显示：在检索依赖任务中可提升约20–30%，但在需要精细视觉定位和高层判断的任务中表现仍低，最高平均得分约为71.4；

**⚠️ 局限性**

局限性包括：样本规模有限、任务覆盖范围受限、评估多依赖确定性匹配、未充分捕捉人类判断的细微差别，且未解决跨文件的几何推理与高阶判断问题。

---

## 172. The Model Says Walk: How Surface Heuristics Override Implicit Constraints in LLM Reasoning

**arXiv ID:** 2603.29025 | [PDF](https://arxiv.org/pdf/2603.29025v1)

**作者:** Yubo Li `[一作]` (Carnegie Mellon University), Rema Padman `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大语言模型在表面线索与隐含可行性约束冲突时的推理失效，并构建Heuristic Override Benchmark (HOB) 进行系统评估。

**💡 创新点**

提出诊断–测量–桥接–治疗框架，量化距离启发式对决策的主导程度，创建跨四类启发式与五类约束的500实例基准，揭示约束推理瓶颈与保守偏差。

**🔧 技术方法**

采用因果遮蔽、词级归因、单调性曲线分析、参数化探测以及目标拆分提示等技术进行诊断与缓解。

**📊 数据集**

自构造的HOB数据集（500实例），配合car wash案例的多模板诊断样本，覆盖最小对照、强度梯度与多域。

**📈 对比分析**

在14个模型上使用严格10/10正确率评估，最高单模型准确率不超过75%；显式提示提升≈15个百分点，目标拆分提示平均提升6–9个百分点。

**⚠️ 局限性**

仅限英文，跨语言一般性未知；缓解方案为概念验证，未彻底排除其他因素；未直接验证前沿模型的相同启发式；H-sem类样本数量有限。

---

## 173. AutoWorld: Scaling Multi-Agent Traffic Simulation with Self-Supervised World Models

**arXiv ID:** 2603.28963 | [PDF](https://arxiv.org/pdf/2603.28963v1)

**作者:** Mozhgan Pourkeshavatz `[一作]`, Nicholas Rhinehart `[通讯]` (University of Toronto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

AutoWorld提出了一种基于无标签LiDAR数据训练的世界模型，并以其预测的未来占用状态为条件来生成多智能体交通运动，从而实现无监督的交通仿真。

**💡 创新点**

创新点在于：① 通过运动感知的自监督占用预测，消除对语义标签的依赖；② 采用自回归条件扩散模型结合世界模型生成行为；③ 在推理阶段引入级联DPP多样性采样，兼顾多模态和真实感。

**🔧 技术方法**

使用技术包括：自监督的VAE+rectified flow占用预测、基于流匹配的世界模型、条件扩散生成器、跨注意力构建预测场景上下文、层级Determinantal Point Process多样性引导、以及离散的离线数据处理。

**📊 数据集**

数据集主要是Waymo Open Dataset (WOD) 的无标签LiDAR序列用于训练世界模型，Waymo Open Motion Dataset (WOMD) 用于训练运动生成器，评估基于WOSAC基准。

**📈 对比分析**

在Waymo Open Sim Agents Challenge中，AutoWorld以0.7865的RMM领跑榜单，交互与动力学表现均位于前六名，且与基于RL/对抗的模型相比无需额外标注即达成最优。

**⚠️ 局限性**

局限性包括：依赖大规模LiDAR数据的采集，DPP多样性采样在推理时成本高；缺乏对语义动态对象的显式建模；以及对极端稀有场景的泛化尚待验证。

---

## 174. A Structural Characterization of Cyclotomic Cosets with Applications to Affine-Invariant Codes and BCH Codes

**arXiv ID:** 2603.29150 | [PDF](https://arxiv.org/pdf/2603.29150v1)

**作者:** Xiongkun Zheng `[一作]` (Hubei University), Mu Yuan `[通讯]` (Hubei University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过对$q$-循环余数集的子集结构进行深入分析，推导出关于这些集合大小的闭式计数公式，并以此得到一类仿射不变码及其循环对应码（包括狭义原始BCH码）的维度表达式和其对偶码的最小距离下界；

**💡 创新点**

创新点在于：①首次给出单个$q$-循环余数集的所有子集的精确计数公式；②利用该计数结果，得到多类仿射不变码和BCH码的维度与最小距离下界的显式表达式，显著完善并提升了以往结果；

**🔧 技术方法**

主要技术包括：q-进制表示的全局移位与部分顺序比较、组合计数与二项变换、Roos界的应用、以及对偶码定义集的结构化分析；

**📊 数据集**

论文为纯理论研究，未使用任何实验数据集；

**📈 对比分析**

通过与已有的维度公式和距离下界（如Levy-dit-Vehel、Ding等）的比较，实验表明所给的下界在多数参数范围内优于或等价于之前的结果，尤其在$t$较小或$δ$较大的情形下更为严格；

**⚠️ 局限性**

局限性在于：①对偶码最小距离的上界仍未得到，仍是下界；②所得到的维度与距离下界对复杂参数的计算量较大，需进一步简化；③目前仅针对单一循环余数集的子集结构，尚未推广到多余数集或更一般的仿射不变码。

---

## 175. SciVisAgentBench: A Benchmark for Evaluating Scientific Data Analysis and Visualization Agents

**arXiv ID:** 2603.29139 | [PDF](https://arxiv.org/pdf/2603.29139v1)

**作者:** Kuangshi Ai `[一作]` (University of Notre Dame), Shusen Liu `[通讯]` (Lawrence Livermore National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了 SciVisAgentBench 这一可扩展的科学可视化代理评估框架，包含 108 个多领域、多维度的任务案例，并通过多模态 LLM 判别器和确定性评估器验证评估方法的可靠性；

**💡 创新点**

其创新点在于（1）首次将完整的科学可视化工作流和多模态视觉评估纳入同一基准；（2）将 LLM 评估与传统图像/代码度量相结合，提供可复制、可解释的结果；（3）系统性地评估专家与 LLM 的对齐与鲁棒性，为大规模自动评估奠定基础；

**🔧 技术方法**

使用的技术包括：大规模多模态 LLM（Claude‑Opus‑4.6、Gemini‑3.1‑Pro 等）作为判别器；图像相似度指标（PSNR、SSIM、LPIPS）；代码完整性检查与 CodeBERT 语义相似度；可视化工具 API（ParaView、napari、VMD、TTK）实现任务执行；以及 token 与时间计量用于效率评估；

**📊 数据集**

使用的数据集覆盖 5 大科学领域（体积、流体、分子、医学影像、拓扑分析）中的 108 个案例，数据类型包括标量场、向量场、张量场、时间序列等，来源于天文、医学、生命科学、物理、地球科学、数学与化学等公开数据；

**📈 对比分析**

通过与多种代理（ChatVis、ParaView‑MCP、GMX‑VMD‑MCP、BioImage‑Agent、TopoPilot、Claude Code、Codex 等）在同一基准下对比，采用整体分数、完成率、pass@k 等指标进行评估；结果显示通用编码代理在大多数任务上取得最高分，而专用工具集成代理在效率与可靠性方面更具优势；

**⚠️ 局限性**

限制包括：基准覆盖仍有限，未涵盖所有科学可视化场景；缺乏完整的过程级和长周期工作流评估；LLM 判别器对某些视觉任务的判定仍受限；数据集规模和多样性有待进一步扩展；基准对齐和评估标准的主观性仍存在一定挑战。

---

## 176. Dual-Imbalance Continual Learning for Real-World Food Recognition

**arXiv ID:** 2603.29133 | [PDF](https://arxiv.org/pdf/2603.29133v1)

**作者:** Xiaoyan Zhang `[一作]` (University of Michigan), Jiangpeng He `[通讯]` (Indiana University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在双重不平衡（长尾类别与步长不平衡）下的连续食品识别框架DIME。

**💡 创新点**

创新点在于结合平衡Softmax、基于类别计数的谱融合以及按秩阈值调制的融合策略，解决双重不平衡导致的知识冲突。

**🔧 技术方法**

使用了ViT‑B/16预训练模型、轻量化Adapter、Balanced Softmax、SVD谱融合与阈值调制。

**📊 数据集**

在VFN186‑LT、VFN186‑Insulin、VFN186‑T2D和Food101‑LT四个长尾食品基准上评测。

**📈 对比分析**

与多种现有CIL方法对比，DIME在A_T、A̅和wA̅上平均提升约2–3%，表现最优且推理成本低。

**⚠️ 局限性**

限制在于仍需手工调参（如阈值、比例）且对极端小任务的适应性尚待验证。

---

## 177. Computing FFTs at Target Precision Using Lower-Precision FFTs

**arXiv ID:** 2603.29129 | [PDF](https://arxiv.org/pdf/2603.29129v1)

**作者:** Shota Kawakami `[一作]` (University of Tsukuba), Daisuke Takahashi `[通讯]` (University of Tsukuba)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

利用奥扎基（Ozaki）方案将FFT的循环卷积拆分为低精度分量，使用数论变换（NTT）和中国剩余定理对每个分量进行精确计算，从而在保持目标精度的同时复用现有低精度FFT实现。

**💡 创新点**

创新点在于：①将奥扎基方案迁移到FFT的循环卷积阶段；②使用NTT+CRT实现精确的低精度卷积；③引入上限拆分次数和NTT域累积策略显著降低NTT调用次数，提升算法可控性。

**🔧 技术方法**

核心技术包括Bluestein FFT、Ozaki分解、TS（Triple‑Single）精度、32位NTT、CRT重建、FastTwoSum、上限拆分与NTT域累积。

**📊 数据集**

实验数据集为人工合成的随机复数序列，使用参数ϕ（0、1.0、4.0）控制指数范围，FFT长度范围为2^10至2^18。

**📈 对比分析**

与FFTW（双精度、单精度）以及TS版Stockham/Bluestein进行对比：相对误差大部分情况下低于基线，且误差随FFT长度不升高；但执行时间约为FFTW双精度的107–1315倍，主要瓶颈在NTT计算。

**⚠️ 局限性**

局限性：①计算量大，执行时间远高于现有实现；②依赖高效的NTT实现，缺乏标准化库；③NTT域累积导致精度退化，需权衡；④目前仅实现对2^k长度FFT，虽然Bluestein可推广，但额外开销增加。

---

## 178. A Multi-Sensor Fusion Parking Barrier System with Lightweight Vision on Edge

**arXiv ID:** 2603.29126 | [PDF](https://arxiv.org/pdf/2603.29126v1)

**作者:** Yuwen Zhu `[一作]` (Hangzhou City University), Zhengzhe Xiang `[通讯]` (Hangzhou City University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于深度学习与多传感器融合，在Raspberry Pi 5上实现了一套三层协同的智能停车障碍系统，涵盖边缘视觉检测、云端业务处理和前端可视化。

**💡 创新点**

创新点包括：① 单类YOLOv3‑tiny结构剪枝并配合专属训练策略，实现极低模型体积（≈33 MB）和高精度；② 异步红外-视觉-惯性融合状态机，采用“红外触发‑视觉确认‑惯性回退”机制提升夜间及遮挡环境下鲁棒性；③ 分层功耗管理与事件触发检测，平均功耗下降74%；④ 完整的端到端系统验证，涵盖感知、通信、业务与展示。

**🔧 技术方法**

核心技术包括YOLOv3‑tiny、结构化剪枝、CPU/NEON优化推理、Raspberry Pi 5、红外测距传感器、MPU6050惯性测量单元、LoRa低功耗通信、Spring Boot后端、Vue3+Vite前端。

**📊 数据集**

使用公开的车辆检测数据集（包含轿车、SUV等多种车型），通过单类标注与数据增强进行训练。

**📈 对比分析**

与单模视觉、视红外触发及完整三模融合方案对比，mAP@0.5提升至96.5%–98.2%；单帧推理延迟600–850 ms，满足5–10 s轮询需求；平均功耗由4.02 W降至1.02 W，能耗降低约74%。

**⚠️ 局限性**

局限性：训练数据季节性与极端天气覆盖有限，需进一步扩充与在线自适应校准；低功耗设备仍受算力约束，模型可进一步蒸馏；红外传感器在近距离死区仍有误判风险。

---

## 179. "I Just Need GPT to Refine My Prompts": Rethinking Onboarding and Help-Seeking with Generative 3D Modeling Tools

**arXiv ID:** 2603.29118 | [PDF](https://arxiv.org/pdf/2603.29118v1)

**作者:** Kanak Gautam `[一作]` (Simon Fraser University), Parmit K. Chilana `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文通过观察26名初学者和专业人士使用生成式3D建模工具的过程，探究了在提示驱动环境下的上手与求助行为。

**💡 创新点**

创新点在于发现“提示先行”与“AI链式求助”成为新的上手与学习模式，以及信用系统如何塑造用户的迭代与求助策略。

**🔧 技术方法**

使用了Meshy AI和Spline AI两款基于文本/图像提示的生成式3D建模工具，并将ChatGPT等大语言模型作为外部辅助来改写提示。

**📊 数据集**

数据集为26名参与者（14名非专业者、12名专业者）的交互记录、思考朗读、屏幕录像、访谈文本以及工具生成的模型结果。

**📈 对比分析**

论文未给出定量性能指标，而是通过主题分析和定性对比，揭示了不同经验水平用户在提示精度、迭代次数、信用使用与输出评价上的显著差异。

**⚠️ 局限性**

局限性包括样本量有限、仅考察两款工具、未系统操纵信用额度、任务顺序固定，以及样本主要集中在年轻用户，导致对更广泛用户群体与工具差异的解释受限。

---

## 180. Enabling Programmable Inference and ISAC at the 6GR Edge with dApps

**arXiv ID:** 2603.29146 | [PDF](https://arxiv.org/pdf/2603.29146v1)

**作者:** Michele Polese `[一作]` (Northeastern University), Tommaso Melodia `[通讯]` (Northeastern University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

设计并验证了一种面向6G RAN的可编程ISAC架构，通过在DU层部署dApp实现I/Q级别感知，利用E3接口实现实时数据访问，结合xApp/rApp进行多节点融合与策略控制，并提供中央训练与生命周期管理框架。

**💡 创新点**

①引入E3接口和dApp概念使得I/Q级别感知成为可能；②构建了从边缘到核心的分层控制和AI流水线；③提出了集成训练、模型目录与意图驱动部署的ISAC生命周期管理；④在单机/多机场景下实现了多拓扑感知（单/双/多点、测距、频谱）。

**🔧 技术方法**

O-RAN开放接口（E2、E3、O1/O2）、近实时RIC与非实时RIC的xApp/rApp、容器化与GPU/FPGA加速、MLOps流水线、全频带OFDM仿真、CRLB分析、OAI 5G实验平台。

**📊 数据集**

采用OFDM信号仿真参数（3.6 GHz、30 kHz子载波、43 dBm等）进行CRLB分析；在OAI实验台获取的真实CIR样本进行测距实验，使用多次扫描（M = 20/60）来评估子空间法精度。

**📈 对比分析**

将dApp中的子空间测距方法与传统5G NR峰值检测方法进行对比，使用误差CDF进行评估；结果显示子空间法在90%以上样本下实现亚米级精度，而峰值检测在此场景下无法给出可靠测距，且子空间法对观测数量的敏感性明显优于峰值检测。

**⚠️ 局限性**

需解决多节点同步、E3接口规范、资源抢占与QoS保证、加速硬件共享调度，以及隐私/安全风险；当前实验仅在单站室内验证，未覆盖大规模多点协同或户外环境。

---

## 181. Route-Induced Density and Stability (RIDE): Controlled Intervention and Mechanism Analysis of Routing-Style Meta Prompts on LLM Internal States

**arXiv ID:** 2603.29206 | [PDF](https://arxiv.org/pdf/2603.29206v1)

**作者:** Dianxing Zhang `[一作]` (Digital China AI Research Institute), Sheng Li `[通讯]` (Digital China AI Research Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在已冻结的指令微调大语言模型前添加不同形式的路由式前缀（如结构化标签、自然语言专家指令等），构建一套对比实验，系统评估路由信号对内部激活稀疏度、域关键字关注度以及输出稳定性的影响，进而检验“稀疏即更确定”假设。

**💡 创新点**

创新点包括：①提出 RIDE（Routing‑Induced Density and Stability）框架，将路由信号的效应转化为可控的文本前缀干预；②设计统一的三类度量（C1‑C3）以及对比管道，便于跨模型、多任务的定量分析；③发现不同模型在密度变化、关注度重分配和稳定性关联上表现出显著异质性，挑战了普适稀疏‑确定法则。

**🔧 技术方法**

技术手段：对三款开源指令微调模型（Llama‑3.1‑8B‑Instruct、Mistral‑7B‑Instruct‑v0.2、Qwen3‑8B）在同一输入上插入五种前缀；提取每层隐藏向量并按早/中/晚三段聚合；使用 Hoyer 稀疏度、Top‑k 能量、关键词关注比例、预测熵和语义变异度等指标；利用配对差异、t 检验、相关分析评估干预效果。

**📊 数据集**

使用 RouterEval 子集（包含 Math、Format/IFEval、Commonsense 三个领域的易/难样本），在每个模型上保持相同随机种子和解码配置，保证实验可比性。

**📈 对比分析**

比较方法为配对差异统计（paired t / Wilcoxon）和 Pearson 相关，结果显示：①所有模型在早/中层均出现密度稠密化（Hoyer 值下降）；②自然语言专家指令比结构化标签产生更强的稠密化；③关键词关注在 Llama/Qwen 中减少，Mistral 中增加；④仅 Qwen 在密度稠密化与输出熵下降之间呈弱正相关，其他模型几乎无关联；总体而言，路由式前缀对输出稳定性的提升并不普适。

**⚠️ 局限性**

局限性：①使用文本前缀作为路由信号的代理，未覆盖真实 MoE 或多模型路由的分布与动态；②实验仅涵盖三款中等规模模型和 RouterEval 的子集，结果对更大规模或其他任务可能不适用；③度量定义与解码参数对数值影响大，需在不同场景下重新校准；④虽然采用配对对比，但仍未实现结构化因果辨识，可能存在未观察到的混杂因素。

---

## 182. Needle in a Haystack: Tracking UAVs from Massive Noise in Real-World 5G-A Base Station Data

**arXiv ID:** 2603.29187 | [PDF](https://arxiv.org/pdf/2603.29187v1)

**作者:** Chengzhen Meng `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用商用5G‑A基站产生的点云实现大范围无人机跟踪；

**💡 创新点**

提出分层框架，分别在点、对象和轨迹层面过滤噪声，并通过噪声指纹建模、空间/速度一致性置信度与Transformer轨迹分类器实现高精度跟踪；

**🔧 技术方法**

噪声指纹高斯建模、DBSCAN聚类、空间/速度一致性置信度、IMM‑UKF状态估计、轻量Transformer（TrajFormer）等；

**📊 数据集**

在上海部署的华为COTS 5G‑A基站收集的15分钟无机飞行与54个不同轨迹、覆盖7天、共14000帧的真实点云数据；

**📈 对比分析**

与两种基线（PointNet+++聚类、ConvTimeNet+跟踪）比较，F1分数从72.78%/75.59%提升至95.56%，误报率仅2%，定位误差4.9 m；

**⚠️ 局限性**

轨迹中断（50–100 m）与定位精度受限，无法弥补因遮挡或方向平行导致的原始点云缺失导致的空洞

---

## 183. SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation

**arXiv ID:** 2603.29186 | [PDF](https://arxiv.org/pdf/2603.29186v1)

**作者:** Ryosuke Matsuda `[一作]` (Tohoku University), Jun Suzuki `[通讯]` (Tohoku University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SLVMEval基准，利用合成长视频对比评估T2LV评价系统的可靠性

**💡 创新点**

首次构建包含多达两小时视频、10种质量/一致性维度的人工验证对比数据集，并证明可不必依赖昂贵人工筛选

**🔧 技术方法**

采用对比式评估框架、合成降质操作（对比度、分辨率、风格、背景、时间流、完整性等）、CLIPScore、VideoScore、VLM-as-a-judge（视频/文本两种方式）

**📊 数据集**

基于Vript密集视频-字幕数据集，生成长视频并进行人工标注筛选

**📈 对比分析**

对比人类评估者（84.7–96.8%准确率）和4类自动评价器，发现大多数自动方法在9/10维度上明显落后于人类，且大部分系统随着视频时长增长准确率下降

**⚠️ 局限性**

自动评价器在长视频语义一致性与时序推理上表现差，且部分评价指标与原始VSLBench不匹配，导致精度不足；系统对视频时长敏感，需进一步提升长视频理解能力

---

## 184. Subjective Quality Assessment of Dynamic 3D Meshes in Virtual Reality Environment

**arXiv ID:** 2603.29166 | [PDF](https://arxiv.org/pdf/2603.29166v1)

**作者:** Duc V. Nguyen `[一作]` (Tohoku Institute of Technology), Truong Thu Huong `[通讯]` (Hanoi University of Science and Technology)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在VR环境中，对动态3D网格在不同细节级别（LoD）和观看距离下的主观质量进行大规模评估，并基于此构建QoE预测模型及资源分配框架。

**💡 创新点**

创新点：①首次系统研究动态网格的LoD与观看距离对QoE的交互影响；②提出基于随机森林的QoE预测模型，显著优于传统线性和已有模型；③设计了基于BIP的QoE-aware资源分配算法，实现在资源受限下的全局最优LoD分配。

**🔧 技术方法**

使用的技术包括：Web‑VR（A‑Frame）测试平台；双刺激失衡量DSIS主观评测；Blender Decimate实现LoD；图像与3D质量度量（PSNR、SSIM、HD、SSE等）；随机森林回归；分支定界求解BIP。

**📊 数据集**

数据集：来自Sketchfab的8个动态3D网格，分别生成8个LoD（20%–95%简化）与5个观看距离（4–20 m），共320个刺激，收集20名受试者评分，形成MOS数据。

**📈 对比分析**

与线性回归、Nguyen模型等基线比较；随机森林模型RMSE=0.19、PLCC=0.98、SROCC=0.97、KROCC=0.88；资源分配上相较贪心和等分方法，QoE提升0.2–15.9%和6.5–30.5%，预算利用率>90%，计算时间<7 ms。

**⚠️ 局限性**

局限：仅评估了单一VR头显（Meta Quest 2）；仅考虑LoD和观看距离，未结合光照、材质等其他视觉因素；模型训练基于有限的8网格样本，泛化性待验证；实时动态调整LoD的实现细节尚未展开。

---

## 185. Webscraper: Leverage Multimodal Large Language Models for Index-Content Web Scraping

**arXiv ID:** 2603.29161 | [PDF](https://arxiv.org/pdf/2603.29161v1)

**作者:** Guan-Lun Huang `[一作]` (National Taiwan University), Yuh-Jzer Joung `[通讯]` (National Taiwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Webscraper框架，利用多模态大型语言模型和自定义工具实现对动态交互式网页的索引-内容式爬取。

**💡 创新点**

创新点在于通过结构化五阶段提示与专门的解析/合并工具，将通用GUI代理转化为高效的动态网页爬虫，并显著提升复杂多页网站的提取准确率。

**🔧 技术方法**

采用Anthropic的Computer Use基础框架、多模态LLM、浏览器控制工具以及自定义Parse与Merge工具，并通过GPT-3/4生成解析脚本与代码解释器执行。

**📊 数据集**

使用六个主流新闻网站（中英双语）和两大电商平台（Momo、Amazon）的人工标注Golden集进行评估。

**📈 对比分析**

通过与Baseline Agent、Prompt Only两种对照配置在30次实验平均下对比，Prompt+Tool方案在新闻网站ROUGE‑L/Correctness上提升约30–50%，在电商平台也取得最高准确率。

**⚠️ 局限性**

仅适用于索引-内容架构，难以处理WebSocket实时流或虚拟滚动等动态加载；在复杂导航或LLM代码生成错误时仍可能失败，并需额外的LLM推理成本。

---

## 186. Kwame 2.0: Human-in-the-Loop Generative AI Teaching Assistant for Large Scale Online Coding Education in Africa

**arXiv ID:** 2603.29159 | [PDF](https://arxiv.org/pdf/2603.29159v1)

**作者:** George Boateng `[一作]` (Kwame AI Inc.), Victor Kumbol `[通讯]` (Kwame AI Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在非洲使用者规模化的手机编程课程 SuaCode 中，构建并部署了双语（英语-法语）的检索增强生成（RAG）生成式 AI 教学助手 Kwame 2.0，并通过人机协作论坛为学习者提供实时、上下文感知的答疑服务。

**💡 创新点**

创新点在于：①将检索增强生成技术与 GPT‑4 结合，实现高质量、即时回答；②支持双语交互与自动语言检测；③在社区论坛中引入人机协作模式，让教师和同伴对 AI 回答进行评估、纠正与补充；④通过引用检索到的教材段落来减少幻觉并增强可追溯性。

**🔧 技术方法**

使用技术包括：Sentence‑BERT 语义检索、ElasticSearch 向量搜索、GPT‑4 API 生成答案、Prompt 工程、语言检测、论坛投票与答案接受机制，以及后端数据同步与日志记录。

**📊 数据集**

数据集来源为：SuaCode 课程材料（章节笔记、练习、测验、往期问答、代码作业）以及 15 期季度共 3,717 名学习者在 35 个非洲国家的论坛交互日志（问答记录、投票、接受标记）。

**📈 对比分析**

评估方法：①社区评分（投票数与接受答案）和②专家评估（对 536 条问题进行有效/无效、课程/管理分类并判定答案正误）。实验结果显示 Kwame 2.0 对课程类问题的准确率达 97.6%，整体准确率 76.7%；在人机协作模式下，综合 AI 与社区答案的总体准确率提升至 85.7%。与原始静态检索系统相比，响应速度极快且质量保持稳定。

**⚠️ 局限性**

局限性包括：对管理类问题的准确率低（46.9%），原因是教材中缺乏最新行政信息；需要人类监督来纠正错误，导致整体响应延迟；社区反馈不足，影响评估完整性；实验仅覆盖最后一季度，缺乏长期随访与学习效果量化。

---

## 187. Efficient and Scalable Granular-ball Graph Coarsening Method for Large-scale Graph Node Classification

**arXiv ID:** 2603.29148 | [PDF](https://arxiv.org/pdf/2603.29148v1)

**作者:** Guan Wang `[一作]` (Chongqing University of Posts and Telecommunications), Wei Wang `[通讯]` (Chongqing Ant Consumer Finance Co., Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种高效可扩展的颗粒球图粗化方法GB-CGNN，用于大规模图节点分类；在粗化阶段先利用METIS生成初始颗粒球，再通过自适应细分得到多粒度子图，并在这些子图上使用GCN进行训练；在粗化阶段实现线性时间复杂度，显著提升了GCN在大规模图上的训练效率；在节点分类任务中与多种现有粗化和GCN基线进行对比，取得了最佳的Micro_F1分数和最快的粗化速度；

**💡 创新点**

1) 将METIS与颗粒球计算相结合，首次实现了O(N)时间复杂度的图粗化；2) 通过自适应细分条件捕获图结构的多粒度特征；3) 在子图上采用批量训练，避免了邻域爆炸问题；4) 兼顾全局与局部信息，显著提升GCN在大规模图上的表现。

**🔧 技术方法**

颗粒球计算、METIS图划分、图粗化、Graph Convolutional Network (GCN)、Adam优化、随机梯度下降(SGD)、稀疏矩阵运算、批量子图训练。

**📊 数据集**

12个公共节点分类数据集：Cora、Citeseer、PubMed、Co‑Cite、Co‑phy、PPI、Reddit、Flickr、Computers、Photo、Yelp、Amazon。

**📈 对比分析**

与8种粗化方法（SGBGC、GBGC、SCAL、FGC、JCGC、GSGC、VNGC、VEGC）以及8种GCN相关模型（GCN、GAT、GraphSAGE、FastGCN、AS‑GCN、Cluster‑GCN、GraphSAINT、SGBGC）进行对比；GB‑CGNN在大多数数据集上实现了最高的Micro_F1分数，且粗化时间比现有粗化方法快数倍甚至数十倍。

**⚠️ 局限性**

1) 对极大图仍可能出现内存不足；2) 需要手动调节超参数（层数、隐藏维度、dropout等），对不同数据集敏感；3) 目前仅针对无向图节点分类，未验证有向图或多任务场景；4) 细分过程在小图上可能产生过拟合，需进一步研究自适应阈值的鲁棒性。

---

## 188. Scalable and Near-Optimal Discrete Phase Shift Optimization for Reconfigurable Intelligent Surfaces with Over 20,000 Elements

**arXiv ID:** 2603.29144 | [PDF](https://arxiv.org/pdf/2603.29144v1)

**作者:** Yuto Hama `[一作]` (Yokohama National University), Hiroyuki Takahashi `[通讯]` (NTT, Inc.)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究如何使用 Coherent Ising Machine 对大规模 RIS（超过 22,000 个元件）的离散相位进行优化，实现物理可行的波束成形。

**💡 创新点**

创新点在于将 RIS 的离散相位优化转化为 Ising Hamiltonian，利用 CIM 的物理并行搜索解决巨型组合优化，并提出基于磁场优势的自旋裁剪方法显著减少问题规模。

**🔧 技术方法**

采用 Coherent Ising Machine、Ising 模型映射、离散相位量化、磁场判定的自旋裁剪、真实硬件 CIM 实验等技术。

**📊 数据集**

未使用公开数据集，而是基于 28 GHz 的自由空间信道模型（BS、UE、RIS 的坐标和尺寸）以及 CIM 设备产生的实验数据进行评估。

**📈 对比分析**

与传统的渐进细化算法和 Fresnel 区域设计相比，在 NLoS 与 LoS 场景下，CIM 在二进制时与细化算法相当，在四进制时提升约 3 dB；对 22,201 元件 RIS，性能按面积比例提升，符合理论预期。

**⚠️ 局限性**

限制在于 CIM 硬件的自旋上限（约 5 万），自旋裁剪在 LoS 下有效但非 LoS 时效果有限；研究仅涵盖单用户单天线场景，未扩展至多用户或多天线环境。

---

## 189. Differentiable Normative Guidance for Nash Bargaining Solution Recovery

**arXiv ID:** 2603.29297 | [PDF](https://arxiv.org/pdf/2603.29297v1)

**作者:** Moirangthem Tiken Singh `[一作]` (Dibrugarh University), Rajnish Kumar `[通讯]` (Queen's University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于引导图扩散的自动化谈判模型，能够在不获取Pareto前沿信息的前提下，生成满足个体理性且接近Nash Bargaining Solution的效用分配。

**💡 创新点**

创新点在于将关系图编码与条件扩散结合，并引入可微分的多项式指导损失（包括IR惩罚、Nash乘积最大化和前沿逼近），以及在扩散后期激活的指导窗口。

**🔧 技术方法**

技术核心包括图注意网络（GATv2）用于结构化编码、条件DDIM扩散模型以及基于softplus的可微分指导损失。

**📊 数据集**

实验使用三组数据集：可解析Pareto前沿的合成NTU数据、包含人类谈判行为的CaSiNo数据集以及Deal or No Deal数据集。

**📈 对比分析**

与六种基线（监督MLP、CVAE、CGAN、无指导DDIM、投影DDIM、硬约束DDIM）比较，指导扩散模型在所有数据集上实现100% IR合规、Nash效率最高（合成约99.5%，CaSiNo约54%，Deal or No Deal约89%），显著优于无指导模型和传统生成模型。

**⚠️ 局限性**

主要局限在于需要完整信息（如离场点和优先权），以及相较单步生成方法存在较高推理延迟。

---

## 190. Self-Improving Code Generation via Semantic Entropy and Behavioral Consensus

**arXiv ID:** 2603.29292 | [PDF](https://arxiv.org/pdf/2603.29292v1)

**作者:** Huan Zhang `[一作]` (Nanjing University), Wei Hu `[通讯]` (Nanjing University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种在无教师模型、无测试oracle条件下通过自我改进提升大型语言模型代码生成能力的方法ConSelf。

**💡 创新点**

创新点包括：①基于程序执行行为的代码语义熵度量，用于筛选可学习的问题；②共识驱动的直接偏好优化（Con‑DPO），对自生成的偏好对进行共识加权，以减弱噪声影响。

**🔧 技术方法**

使用了观察引导采样、多样化候选代码生成、代码语义熵计算、行为共识评分、Con‑DPO训练以及LoRA微调等技术。

**📊 数据集**

实验使用TACO数据集（仅含问题描述与测试输入），并在HumanEval、MBPP、EvalPlus和LiveCodeBench四个基准上进行评测。

**📈 对比分析**

与基线（Base、PFPO、Self‑SFT、Self‑DPO）比较，ConSelf在CodeLlama、DeepSeek‑Coder、Qwen2.5‑Coder上平均提升pass@1约3–4%，且所需训练样本量显著减少。

**⚠️ 局限性**

局限性包括：需要执行多条候选程序导致额外计算成本；对极难题仍需外部监督；可能存在奖励作弊风险，且对低能力模型效果有限。

---

## 191. ConInfer: Context-Aware Inference for Training-Free Open-Vocabulary Remote Sensing Segmentation

**arXiv ID:** 2603.29271 | [PDF](https://arxiv.org/pdf/2603.29271v1)

**作者:** Wenyang Chen `[一作]` (Yunnan Normal University), Yonghang Tai `[通讯]` (Yunnan Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种训练自由的上下文感知推理框架ConInfer，用于提升开放词汇遥感分割的空间一致性和准确性。

**💡 创新点**

创新点在于在推理阶段显式建模不同空间单元之间的语义依赖，利用DINOv3提取的视觉上下文与VLM的语义先验进行联合优化，形成无训练的全局一致推断。

**🔧 技术方法**

采用的技术包括：Vision‑Language Model（如CLIP）生成的文本原型相似度先验；Vision Foundation Model（DINOv3）提取的视觉上下文特征；高斯混合模型（GMM）与交叉熵/KL 迭代联合优化；无监督的多轮 EM 迭代。

**📊 数据集**

在17个公开遥感分割数据集上验证，包含8个多类别语义分割数据集（OpenEarthMap、LoveDA、iSAID、Potsdam、Vaihingen、UAVid、UDD5、VDD）和3个单类别提取任务（建筑、道路、洪水）使用对应数据集。

**📈 对比分析**

与6个最先进的训练自由开放词汇分割方法（MaskCLIP、SCLIP、GEM、ClearCLIP、SegEarth‑OV等）以及MaskCLIP*基线比较，ConInfer在多类别分割平均mIoU提升约2.8%，在单类别提取平均IoU提升约6.1%，整体性能均超过现有最佳方法。

**⚠️ 局限性**

局限性主要在于目前仅在patch级别进行推理，导致细粒度边界分辨率有限，且对极低空UAV视角的细小目标边界识别仍不够精细。

---

## 192. Aligning Multimodal Sequential Recommendations via Robust Direct Preference Optimization with Sparse MoE

**arXiv ID:** 2603.29259 | [PDF](https://arxiv.org/pdf/2603.29259v1)

**作者:** Hejin Huang `[一作]` (Sun Yat-sen University), Rong Pan `[通讯]` (Sun Yat-sen University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在多模态序列推荐中尝试将直接偏好优化（DPO）用于隐式反馈下的用户偏好对齐

**💡 创新点**

提出“随机Top‑K负采样”策略，用动态高分候选池中的随机采样替代硬负样本，解决误负样本导致的梯度错误，并引入稀疏混合专家提升模型容量

**🔧 技术方法**

使用DPO目标、随机Top‑K负采样、可选稀疏MoE编码器和两阶段预热训练方案

**📊 数据集**

Amazon Review数据集（Toys & Games、Beauty、Home & Kitchen）

**📈 对比分析**

与传统非时序、时序与多模态基线（共计17种）对比，RoDPO在NDCG@5上提升5.25%（最高），MRR@5提升7.67%，同时推理延迟几乎不变

**⚠️ 局限性**

仅在Amazon电商场景验证，泛化至短视频等其它多模态领域待验证；训练成本略高于轻量级ID模型，且仍需冻结参考模型

---

## 193. SuperGrasp: Single-View Object Grasping via Superquadric Similarity Matching, Evaluation, and Refinement

**arXiv ID:** 2603.29254 | [PDF](https://arxiv.org/pdf/2603.29254v1)

**作者:** Lijingze Xiao `[一作]`, Yu Ren `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了SuperGrasp两阶段单视点抓取框架，第一阶段用超四面体相似匹配生成抓取候选，第二阶段用E-RNet对候选进行评估与细化。

**💡 创新点**

创新点包括基于超四面体相似匹配的候选生成、利用抓取锚点扩展感知范围的E‑RNet评估与多选细化，以及实现无真实世界微调的直接迁移。

**🔧 技术方法**

采用超四面体拟合与相似度匹配、PointNet++骨干的E‑RNet网络、深度相机点云处理、仿真与真实数据采集等技术。

**📊 数据集**

使用1.5k个超四面体原语数据库以及从124个YCB/KIT/DexNet对象收集的10万条带标签的点云抓取样本；在实测中使用30相似对象+25新对象。

**📈 对比分析**

与PointNetGPD、Contact‑GraspNet对比，在模拟10/20/10未见物体场景下抓取成功率与任务成功率均提升至约95%/93%；在真实场景下抓取/任务成功率分别达98%/94%，并且总时延仅0.79 s。

**⚠️ 局限性**

局限性在于对单视点几何缺失的鲁棒性虽强，但在自遮挡或稀疏场景仍易失败；总体时延仍略高于端到端方法。

---

## 194. Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents

**arXiv ID:** 2603.29231 | [PDF](https://arxiv.org/pdf/2603.29231v1)

**作者:** Aaditya Khanal `[一作]` (Northern Kentucky University), Junxiu Zhou `[通讯]` (Northern Kentucky University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了针对长时程LLM代理的可靠性科学框架，包含四项新指标并构建了396题、跨四个持续时间桶、三大领域的基准；

**💡 创新点**

创新点在于将可靠性拆分为可靠性衰减曲线、方差放大因子、渐进降解分数与熔毁起始点四指标，并揭示可靠性与能力、任务持续时间、领域结构的系统性分离与交互；

**🔧 技术方法**

技术采用ReAct与记忆增强两种脚本，OpenRouter统一API调用10个开源模型，使用程序化评估、任务拆分、熵滑动窗口检测熔毁，以及k=3重复实验与统计推断；

**📊 数据集**

数据集为396道任务，分布在软件工程、网络研究、文档处理三领域，按人类完成时间分为短≤5min、介5–30min、长30–120min、超长≥120min四桶；

**📈 对比分析**

比较方法通过pass@1、pass^k、RDC、VAF、GDS、MOP等多维度评估，结果显示可靠性随持续时间显著下降（平均下降24.3个百分点），顶级模型保持高可靠性但熔毁率亦高；

**⚠️ 局限性**

局限包括仅评估10个开源模型，未覆盖专有模型；持续时间作为人类估计的代理可能不完全匹配代理难度；程序化评估可能忽略细微错误；Web任务可重复性受网络变化影响；

---

## 195. Kernel-SDF: An Open-Source Library for Real-Time Signed Distance Function Estimation using Kernel Regression

**arXiv ID:** 2603.29227 | [PDF](https://arxiv.org/pdf/2603.29227v1)

**作者:** Zhirui Dai `[一作]` (University of California San Diego), Nikolay Atanasov `[通讯]` (University of California San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一款实时可微分的SDF学习框架，结合BHM前端进行表面估计和GP后端进行距离预测，并实现了不确定性量化。

**💡 创新点**

创新点包括：将Bayesian Hilbert Map与Gaussian Process结合，实现软最小化（softmin）不确定性传播；利用八叉树分区和优先级队列优化实时性能；引入可同步的权重机制保证局部地图一致性。

**🔧 技术方法**

使用技术包括：核回归、贝叶斯希尔伯特映射、GP回归、RBF和Matérn 3/2核、marching cubes、软最小化不确定性推断、八叉树空间划分与优先级队列调度。

**📊 数据集**

在Replica、Cow、Lady、Newer College等真实与合成数据集上进行评估，使用Depth/LiDAR噪声模型进行模拟。

**📈 对比分析**

与Voxblox、FIESTA、iSDF、VDB-GPDF等基线比较，本文方法在SDF误差（MAE）和梯度误差上均达到了或接近最优，实时更新时间约150 ms，整体性能优于传统基线。

**⚠️ 局限性**

局限性在于仍需较多计算资源（尤其是GP矩阵求逆），对极端噪声或大规模动态环境的适应性有限，且目前仅支持二维/三维场景。

---

## 196. Long-Reach Robotic Manipulation for Assembly and Outfitting of Lunar Structures

**arXiv ID:** 2603.29226 | [PDF](https://arxiv.org/pdf/2603.29226v1)

**作者:** Stanley Wang `[一作]` (Stanford University), Mark Cutkosky `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发并验证了一种紧凑型长臂机械手，采用可部署复合材料桅杆，实现了在月球表面等大型工作空间内的半自主电缆布线与其他布置任务。

**💡 创新点**

创新点在于：①将可展开桅杆与多轴关节腕结合，显著扩大工作空间并保持轻量化；②针对桅杆的弹性偏转与振动提出了基于弹性模型的补偿与速度自适应控制；③利用视觉标记实现端执行器的闭环视觉伺服，构建任务误差模型以动态调整运动速度。

**🔧 技术方法**

主要技术包括：可展开复合桅杆设计与编码器测距、PI控制器、弹性偏转补偿的前向运动学、解析雅可比矩阵与伪逆、视觉伺服（AprilTag + SLERP）以及任务误差的三维插值模型。

**📊 数据集**

实验数据集：对三组参数（支撑角度 0/45/90°、桅杆长度 0.6/0.9/1.2/1.5/1.8 m、任务速度 17/33/50/67/80 mm/s）共 375 次试验，使用 OptiTrack 记录轨迹并通过 AprilTag 进行视觉定位。

**📈 对比分析**

通过与实验测得的误差进行比较，验证了任务误差模型的保守性；桅杆长度 1.8 m 时端执行器平均误差 <15 mm；在四个不同位置/角度的电缆布线任务中，实际误差均 ≤ 15 mm，成功完成所有布线任务。

**⚠️ 局限性**

局限性包括：假设桅杆为准静态，未对振动与动力学耦合建模；速度越高误差越大，限制了操作速度；实验仅在固定基座上进行，未考虑移动平台与全局协调；对极端月球环境（低重力、尘埃、温差）及多臂协作的鲁棒性未做评估。

---

## 197. Sustainable AI Assistance Through Digital Sobriety

**arXiv ID:** 2603.29222 | [PDF](https://arxiv.org/pdf/2603.29222v1)

**作者:** Madeline Jennings `[一作]` (University of Calgary), Ronnie de Souza Santos `[通讯]` (University of Calgary)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过人工手工标注软件开发相关的LLM查询，评估其中可通过低成本替代（如传统搜索）完成的比例，量化数字素养在节能方面的潜在贡献。

**💡 创新点**

首次从需求侧引入“数字素养”概念，将LLM使用必要性按任务复杂度划分，量化不必要推理对能耗的影响，填补绿色AI中关于用户行为的研究空白。

**🔧 技术方法**

使用手工编码分类（factoid/complex/convenience）结合NIST AI使用分类法，并借助Python进行数据筛选与处理。

**📊 数据集**

采用公开的LMSYS-Chat-1M对话数据集，从中随机抽取约200条软件相关查询进行分析。

**📈 对比分析**

通过对比传统搜索（约0.03 Wh）与LLM推理（约0.3 Wh）的能耗估计，发现约43.2%的查询可避免使用LLM，若在这些案例中减少90%推理请求，整体能耗可降低约45%。

**⚠️ 局限性**

局限性包括样本量有限、人工标注存在主观性、仅聚焦软件类查询、对过程自动化与问题检测等罕见用例缺乏数据，结果仅为初步验证，需进一步扩大样本与跨数据集验证。

---

## 198. Robust and Consistent Ski Rental with Distributional Advice

**arXiv ID:** 2603.29233 | [PDF](https://arxiv.org/pdf/2603.29233v1)

**作者:** Jihwan Kim `[一作]` (Seoul National University), Chenglin Fan `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种同时兼顾鲁棒性和一致性的滑雪租赁在线算法框架，利用完整的分布预测来决定租赁或购买时机。

**💡 创新点**

创新点在于：①将分布预测与鲁棒学习增强方法结合，提供可调节的Clamp Policy和随机化Water‑Filling算法；②给出了关于预测误差（Wasserstein/TV距离）的严谨一致性与鲁棒性上界；③在完美与不完美预测情形下均能实现与经典随机算法相当的期望竞争比。

**🔧 技术方法**

使用的技术包括：阈值策略分析、期望竞争比推导、Wasserstein/TV误差分析、二分搜索与水填充（Water‑Filling）优化、离散分布的切片线性成本函数处理。

**📊 数据集**

实验使用合成的分布数据集：均匀、截断高斯、截断几何、双峰分布等，全部在理论支持的有限支持上进行。

**📈 对比分析**

与传统点预测或基于分布的基线算法（如Purohit等）进行比较。结果显示：在保持相同鲁棒性阈值下，新方法在一致性指标上提升约5%–20%，尤其在双峰和几何分布上表现最显著；在预测误差增大时鲁棒性仍保持不变，优于基线。

**⚠️ 局限性**

局限性：①假设预测分布有已知上界和有限支持；②算法在高维或连续分布下的实现与计算复杂度尚未讨论；③实验仅基于合成分布，缺乏真实世界数据验证。

---

## 199. Designing Human-GenAI Interaction for cMOOC Discussion Facilitation: Effects of a Collaborative AI-in-the-Loop Workflow on Social and Cognitive Presence

**arXiv ID:** 2603.29285 | [PDF](https://arxiv.org/pdf/2603.29285v1)

**作者:** Jianjun Xiao `[一作]` (Beijing Normal University), Cixiao Wang `[通讯]` (Beijing Normal University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一个协作 AI‑人类协作工作流程，用于 cMOOC 讨论促进。

**💡 创新点**

创新点在于将网络结构触发的目标选择、话语适应的角色设计以及人工审核作为一体化的 AI‑in‑the‑loop 交互流程，并证明直接互动比仅有存在更有益。

**🔧 技术方法**

使用了大语言模型（如 Kimi‑k2‑turbo‑preview、GPT‑5.2）、超图闭合中心性算法以及手工与 LLM 双重编码。

**📊 数据集**

数据集来自一门为期 5 周的中文 cMOOC，共 606 名学习者的讨论日志，约 6,500 条记录。

**📈 对比分析**

通过规则化的准实验对照（奇偶分配）与配对/曼惠特尼 U 检验比较，结果显示 PCA 参与显著提升开放式交流和网络凝聚力，直接互动还能提升高阶认知指标，效果在统计上显著且经过 BH 校正。

**⚠️ 局限性**

局限在于单一课程、非随机分配、可能的干扰效应、文化背景限定以及缺乏对学习者感知与长期学习成效的评估。

---

## 200. Monodense Deep Neural Model for Determining Item Price Elasticity

**arXiv ID:** 2603.29261 | [PDF](https://arxiv.org/pdf/2603.29261v1)

**作者:** Lakshya Garg `[一作]` (Walmart Inc.), Mayank Uniyal `[通讯]` (Walmart Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一种名为 Monodense 的深度神经网络框架，用于大规模零售交易数据中商品价格弹性的估计与评估。

**💡 创新点**

核心创新在于引入 Monodense 层对价格-需求关系施加单调性约束，并在网络低层注入价格输入以增强价格敏感度，消除了对传统 C/T 实验的需求。

**🔧 技术方法**

使用深度学习技术（多层嵌入+稠密+Monodense 层），Monotonicity 约束，Adam 优化，MSE 损失；与 LightGBM 与双机学习（DML）在同一框架下进行对比。

**📊 数据集**

基于约 10 亿+ 行的跨店跨品月度交易数据，包含价格、库存、促销、竞争价、替代品等特征，并构建了 1 年训练 + 3 个月 OTS 的数据集。

**📈 对比分析**

在离线测试集上使用 WMAPE 与 MAE 评估弹性精度。Monodense 模型 WMAPE 30.90% (低于 LGBM 35.9% 和 DML 36.1%)，MAE 0.36 (低于 LGBM 0.42 和 DML 0.43)，表现最优。

**⚠️ 局限性**

局限性：模型对极端季节性或稀有商品弹性预测仍有限；训练需 GPU 资源；仅在单渠道单价不变假设下验证，对多渠道动态定价的适用性需进一步验证。

---

## 201. Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism

**arXiv ID:** 2603.29252 | [PDF](https://arxiv.org/pdf/2603.29252v1)

**作者:** Tao Chen `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关、可插拔的视觉记忆机制 FlexMem，能够让多模态大语言模型（MLLM）在不受输入长度限制的情况下，逐步处理长视频并通过记忆召回完成问答。

**💡 创新点**

创新点包括：①双路径压缩（Dual‑Pathway Compression）在预填充和解码阶段分别压缩上下文与局部信息；②基于编码的记忆阅读与高效索引（MemIndex）两种召回策略；③迭代式视频切片处理与记忆存储，实现无限长视频的可扩展性与高效召回。

**🔧 技术方法**

使用技术：视觉 KV 缓存、双路径压缩、上下文聚合得分与局部显著性得分、交叉注意力召回、统计拟合得到的线性回归索引（MemIndex）、与现有的 VideoRAG 与视觉压缩方法对齐的基准实现。

**📊 数据集**

采用的评测数据集包括 MLVU、LongVideoBench、LVBench、Video‑MME、TimeScope，此外还在流式 QA 任务中使用了 EPM、ASI 等数据集。

**📈 对比分析**

在单台 RTX‑3090 GPU 下，FlexMem 在 LLaVA‑Video 与 LLaVA‑OneVision 上平均提升 3–5% 以上，TimeScope 上提升 32.2%，LVBench 上提升 19.7%，与 AdaRETAKE、AKS 等方法相比显著更优；在多任务比较中，表现可与 GPT‑4o、Gemini‑1.5‑Pro 相当或更好，展示出高效、可扩展的长视频理解能力。

**⚠️ 局限性**

局限性：1）仍依赖现有 MLLM 的 KV 结构，跨模型迁移可能需要额外调优；2）在极端大规模视频或高帧率场景下，存储与召回的内存占用仍是瓶颈；3）对部分需要细粒度时序推理的任务，单纯的记忆召回可能不足，仍需进一步结合动态注意力或学习式压缩方法。

---

## 202. An Experiential Approach to AI Literacy

**arXiv ID:** 2603.29238 | [PDF](https://arxiv.org/pdf/2603.29238v1)

**作者:** Aakanksha Khandwaha `[一作]` (University of Waterloo), Edith Law `[通讯]` (University of Waterloo)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出并实施一种将AI素养与工作场景经验相结合的体验式学习方法，包含三阶段工作坊与案例共创。

**💡 创新点**

创新点在于通过叙事化的工作坊和日常工作背景中的头脑风暴，桥接AI理论与实际应用之间的差距，并鼓励参与者从自身经验出发共创AI用例。

**🔧 技术方法**

使用的技术主要是工作坊设计、故事叙述、头脑风暴卡牌和共创工具，侧重于人机协作与反思式学习。

**📊 数据集**

未使用传统数据集，主要基于参与者提供的真实工作场景案例进行分析与共创。

**📈 对比分析**

本研究未进行量化性能比较，强调方法通过多行业多领域的实验工作坊验证其可行性和效果。

**⚠️ 局限性**

局限包括对参与者时间投入的依赖、行业差异导致方法可推广性受限，以及缺乏系统的评估指标和客观效果验证。

---

## 203. CCDNet: Learning to Detect Camouflage against Distractors in Infrared Small Target Detection

**arXiv ID:** 2603.29228 | [PDF](https://arxiv.org/pdf/2603.29228v1)

**作者:** Zikai Liao `[一作]` (Stony Brook University), Zhaozheng Yin `[通讯]` (Stony Brook University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了红外小目标检测模型 CCDNet，针对目标伪装与干扰物高误检问题，设计了宽度扩展的自条件多分支感知器（WMP）骨干网络、聚合‑细化融合颈部（ARFN）以及对比辅助干扰物判别器（CaDD），实现对低对比目标的精准检测和误检率的显著降低。

**💡 创新点**

创新点包括：①WMP通过宽度扩展和自条件机制聚合多尺度特征，避免深度网络导致的小目标信息丢失；②ARFN通过顶部背景语义引导（TBSG）与底部结构增强（BOSE）双向融合，构建目标中心特征表达；③CaDD结合局部对比模块（LCM）与全局对比模块（GCM），在训练期间对目标与干扰物进行主动对比学习，显著抑制误检。

**🔧 技术方法**

技术实现包括：Weighted Multi‑Branch Perceptron（WMP）、Aggregation‑and‑Refinement Fusion Neck（ARFN）(TBSG、BOSE)、Contrastive‑aided Distractor Discriminator（CaDD）(LCM、GCM)、自条件机制、动态空间细化、可变形卷积、通道细化、剪枝策略、检测头、交叉熵+L2定位损失及对比损失。

**📊 数据集**

实验使用三大公开红外小目标数据集：IRSTD‑1k、NUDT‑SIRST、NUAA‑SIRST。

**📈 对比分析**

通过与传统方法（PSTNN、DNGM、TSLSTIPT）、DL方法（ACM、ISNet、DAGNet、MSHNet、IRPruneDet）、通用目标检测器（RetinaNet、Deformable DETR、FCOS）以及伪装检测器（SINet、FEDER）在P、R、F‑1、Params、FLOPs、FPS等指标上对比，CCDNet在F‑1上分别提升至92.36%、92.64%、92.08%，在所有数据集上均位列第一，且保持约39 FPS的实时速度。

**⚠️ 局限性**

局限性包括：对比模块仅在训练阶段使用，推理时无加速；对阈值t1、t2的设置依赖经验，可能在不同场景下需要调优；模型虽已压缩但参数量仍高于轻量化方法；仅针对红外小目标，光学图像或大目标的适应性尚未验证；极端噪声或高动态背景下的鲁棒性待进一步研究。

---

## 204. SiPaKosa: A Comprehensive Corpus of Canonical and Classical Buddhist Texts in Sinhala and Pali

**arXiv ID:** 2603.29221 | [PDF](https://arxiv.org/pdf/2603.29221v1)

**作者:** Ranidu Gurusinghe `[一作]`, Nevidu Jayatilleke `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了SiPaKosa：一个包含约786k句、约9.25M词汇的双源（历史文献与公认经典）Sinhala‑Pali佛教文本语料库，并提供完整的元数据和语言分类；

**💡 创新点**

首次在低资源语言下实现了规模化、双源佛教语料库的系统化采集、OCR、版权审核与语言混合检测，为后续研究提供高质量、跨年代的文本资源；

**🔧 技术方法**

采用PDF文本提取+Google Cloud Document AI OCR、基于词典的语言识别、规则化页面分类、数据去重与元数据标准化等技术；

**📊 数据集**

历史公有领域PDF文档（83本→16本）与tripitaka.online佛教经典（5 Nikaya）的网络爬取文本；

**📈 对比分析**

对9种现有语言模型（含4个专有模型与5个开源模型）在佛教、混合与一般Sinhala测试集上计算困惑度；专有模型平均PPL≈1.08–1.77，开源模型平均PPL≈3.29–36.71；同时评估领域差距比，显示专有模型在处理经典词汇与语言混合上优于开源模型；

**⚠️ 局限性**

缺乏人工校正OCR错误与正字法标准化、语言分类阈值可优化、未覆盖现代出版物与Sanskrit文本、评估仅基于困惑度且未进行下游任务人类评测。

---

## 205. Customer Analysis and Text Generation for Small Retail Stores Using LLM-Generated Marketing Presence

**arXiv ID:** 2603.29273 | [PDF](https://arxiv.org/pdf/2603.29273v1)

**作者:** Shiori Nakamura `[一作]` (Nagoya Institute of Technology), Tadachika Ozono `[通讯]` (Nagoya Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套面向小型服装零售店的 POP 文本生成与评估协作系统。

**💡 创新点**

创新点在于将客户细分与文本创作模块化，并通过 LLM 进行人机协作实现多样化创意与多视角评估。

**🔧 技术方法**

主要技术包括基于 LLM 的 Profile Builder、Draft Generator、Style Rephraser 以及 Persona Evaluator。

**📊 数据集**

数据集采用小型店铺的商品信息与用户问答历史，未使用公开大规模数据集。

**📈 对比分析**

实验通过五种支持级别对比，结果显示全功能手工选择模式平均得分比无支持提升 2.37 分（-3~+3 量表）。

**⚠️ 局限性**

局限性在于 Persona 生成一致性不足，PE 的自动选择可靠性低，且系统对不同业务场景的适应性待验证。

---

## 206. Unbiased Model Prediction Without Using Protected Attribute Information

**arXiv ID:** 2603.29270 | [PDF](https://arxiv.org/pdf/2603.29270v1)

**作者:** Puspita Majumdar `[一作]` (IIIT-Delhi), Richa Singh `[通讯]` (IIT Jodhpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在无受保护属性信息情况下，利用非受保护属性进行偏差缓解的NPAD算法，并在面部属性预测任务中验证其有效性。

**💡 创新点**

创新点在于不依赖受保护属性，采用智能非受保护属性选择，并通过新设计的Debiasing via Attribute Cluster Loss（DACL）和Filter Redundancy Loss（FRL）两种损失函数对特征空间进行优化，同时提出Overall Performance Equality（OPE）评估指标。

**🔧 技术方法**

使用非受保护属性聚类选择、χ²独立性检验、基于多类聚类的DACL和FRL损失，以及LightCNN-29特征提取网络。

**📊 数据集**

在LFWA和CelebA两大面部属性数据集上进行实验，受保护属性为性别和年龄。

**📈 对比分析**

与基线BMT、受保护属性偏差消除PAD、学习失败(LfF)以及ARL等方法比较，NPAD在整体准确率、DoB和OPE指标上均优于BMT，接近或超过PAD，显著降低偏差并提升总体性能。

**⚠️ 局限性**

局限性包括对非受保护属性数量与最优选择的确定仍是开放问题，实验仅针对二元属性，且2^n+1类优化在大规模或多属性场景下计算成本较高。

---

## 207. Audio Hallucination Attacks: Probing the Reliability of Large Audio Language Models

**arXiv ID:** 2603.29263 | [PDF](https://arxiv.org/pdf/2603.29263v1)

**作者:** Ashish Seth `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了音频幻觉攻击框架 AHA，并构建了 6.5K 题目评测集 AHA‑Eval 与 120K 训练集 AHA‑Guard，评估并对抗大型音频语言模型（LALM）的幻觉行为。

**💡 创新点**

创新点在于：①将攻击分为查询式（显式/隐式）与音频式两层面；②设计显式/隐式查询区分模型对音频验证的依赖；③通过在音频中注入合成语音实现音频幻觉攻击；④提出后对齐数据集 AHA‑Guard 并用 DPO 进行对齐训练。

**🔧 技术方法**

采用的技术包括：LLM 一致性过滤器、GPT‑5.2 评判器、Chain‑of‑Thought（CoT）推理、Direct Preference Optimization（DPO）微调、TTS 合成、音频注意力分析与日志概率评估。

**📊 数据集**

使用的数据集为 AudioCaps、Clotho、MusicCaps（经 LLM 过滤得到 8K 高质量音频‑文本对），并生成 6.5K 的 AHA‑Eval 题目和 120K 的 AHA‑Guard 对齐对。

**📈 对比分析**

通过 Attack Success Rate（ASR）评估模型，在显式查询下 ASR 低至 1.9%（随机），但在隐式查询或音频攻击下高达 95%；CoT 在显式攻击下略有下降，DPO 对齐后可将 ASR 降低多达 49%，显著提升模型可靠性。

**⚠️ 局限性**

局限性包括：①攻击样本主要基于合成与文本生成，缺乏真实环境噪声和多模态复杂度；②评估仅在公开数据集上进行，未覆盖更广泛的实际场景；③对齐过程可能引入拒绝偏差，且对其他防御方法未做系统验证。

---

## 208. Grokking From Abstraction to Intelligence

**arXiv ID:** 2603.29262 | [PDF](https://arxiv.org/pdf/2603.29262v1)

**作者:** Junjie Zhang `[一作]` (Chinese Academy of Sciences), Xisong Dong `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在模块化算术任务（加减乘除）中出现的 grokking 现象，并提出该现象源于模型内部结构的自发简化，结合实验与理论证明。

**💡 创新点**

创新点包括：① 将 Occam 原则、Kolmogorov 复杂度与 Singular Learning Theory (SLT) 结合，构建可解析的 Singular Feature Machine (SFM) 作为理论代理；② 通过因果中介分析、频域谱稀疏度、BDM 复杂度等多维度度量展示结构简化与泛化的对应关系；③ 明确了 grokking 作为模型从高维记忆状态向低维稀疏结构的相位转变。

**🔧 技术方法**

使用技术包括：48 层 GPT‑2 风格 Transformer 训练、Causal Mediation Analysis (CMA)、频域 Fourier 分析（Gini 系数、IPR）、BDM/Kolmogorov 复杂度估计、Singular Learning Theory 代理指标、SFM 的解析动力学与 Occam Gate。

**📊 数据集**

数据集为四个模组化算术任务（+、−、×、÷）在有限域 ℤ₉₇ 上的输入对 (u, v)，构成长度为 4 的 token 序列。

**📈 对比分析**

通过训练/测试准确率曲线、CMA 热图、频域稀疏度、BDM 曲线与 RLCT/KC 指标的对齐，展示在 10k–100k 步骤出现“grokking”跃迁；测试准确率从随机 ~0.01 提升至 1.0，表明结构简化与泛化同步出现。

**⚠️ 局限性**

局限性：① 实验仅针对单一算术任务和有限域，泛化至更复杂任务尚未验证；② SFM 为理论代理，未能完全证明其与 SGD 训练的 Transformer 之间的对应关系；③ 对非加法操作的解析与实证不足；④ 复杂度代理（RLCT、BDM）在大模型上可扩展性与计算成本仍需进一步研究。

---

## 209. Omni-NegCLIP: Enhancing CLIP with Front-Layer Contrastive Fine-Tuning for Comprehensive Negation Understanding

**arXiv ID:** 2603.29258 | [PDF](https://arxiv.org/pdf/2603.29258v1)

**作者:** Jingqi Xu `[一作]` `[通讯]` (University of Southern California), Jingqi Xu (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对CLIP进行细调，专门提升其对两类否定（存在基否定与缺失基否定）的理解能力

**💡 创新点**

创新点在于：①设计了针对两种否定的对比目标，分别在图像-原始字幕和图像-否定字幕之间进行正负样本拉近/拉远；②仅微调文本编码器的前几层，利用前层对细粒度语法的学习优势；③引入缺失基否定中字幕对比损失来保持原始与否定字幕的语义区分

**🔧 技术方法**

技术方法包括：基于InfoNCE的对比学习，分别构造 presence-based 与 absence-based 的三项损失（包含显式否定区分、图片-字幕/否定字幕对比和字幕对比），使用 margin‑based 语义分离项，AdamW优化器和微调策略

**📊 数据集**

使用的数据集有：CC‑Neg（存在基否定）、NegRefCOCOg（缺失基否定）、COCO检索基准、OAN 数据以及 188,246 条 CC‑Neg 三元组

**📈 对比分析**

与原始 CLIP、CoN‑CLIP 与 NegationCLIP 进行对比；在 ViT‑B/32、ViT‑B/16 与 ViT‑L/14 上，presence‑neg 精度提升约 50%+，absence‑neg 提升约 10%+，COCO 检索精度提升约 19%+，在两类否定任务均达到 99%+ 的高精度

**⚠️ 局限性**

局限性包括：对否定语义的覆盖仍有限，仅关注“no/not/without”，未处理更复杂的否定结构；微调仅涉及文本编码器前层，可能无法完全利用后层信息；未验证跨语言、跨任务的迁移效果，且缺少对模型内部语义变化的可解释性分析

---

## 210. Real-Time Surrogate Modeling for Fast Transient Prediction in Inverter-Based Microgrids Using CNN and LightGBM

**arXiv ID:** 2603.29255 | [PDF](https://arxiv.org/pdf/2603.29255v1)

**作者:** Osasumwen Cedric Ogiesoba-Eguakun `[一作]` (University of Tulsa), Suman Rath `[通讯]` (University of Tulsa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用卷积神经网络和LightGBM构建了实时微电网动态预测的代理模型，并在EMT数字双胞胎生成的数据上训练。

**💡 创新点**

提出将时序CNN与基于特征的LightGBM结合的混合框架，针对不同物理量选择最佳模型，从而在保持高精度的同时实现数百倍到千倍的实时加速。

**🔧 技术方法**

采用卷积神经网络（CNN）、梯度提升树（LightGBM）、滑动窗口特征工程、混合学习策略以及基准对比实验。

**📊 数据集**

使用包含10台分布式发电单元的微电网EMT仿真数字双胞胎数据，涵盖11种运行与干扰情景（负荷变化、故障、噪声、通信延迟等）。

**📈 对比分析**

与原始EMT仿真对比，LightGBM实现约1000×速度提升并超实时；CNN速度较慢；混合模型约500×提升、近实时；在噪声与延迟等OOD条件下仍保持高准确性（频率R²>0.99，电压R²≈0.84，电压骤降R²≈0.75）。

**⚠️ 局限性**

模型仅在单一微网架构上训练，缺乏跨系统泛化；采用单步预测，未考虑多步未来行为；深度模型对导出特征的依赖有限，需进一步结合物理知识和更复杂网络。

---

## 211. Denoising data reduction algorithm for Topological Data Analysis

**arXiv ID:** 2603.29248 | [PDF](https://arxiv.org/pdf/2603.29248v1)

**作者:** Seonmi Choi `[一作]` (Seowon University), Seung Yeop Yang `[通讯]` (Kyungpook National University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `fede83ac-7505-405f-ab37-e7284695c47f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于网格的 Refined Characteristic Lattice Algorithm (RCLA)，在单步完成数据压缩与去噪；通过阈值k筛除噪声单元，保留重要结构；并给出了稳定性理论与自动参数选择方法。

**💡 创新点**

创新点在于：①将阈值k融入传统 CLA，使其同时兼顾降采样与去噪；②在 HPPP 噪声模型下证明了持久图的瓶颈距离上界；③提出基于最近邻统计与贝叶斯估计的自动 (δ,k) 选择算法，消除手动调参。

**🔧 技术方法**

使用技术包括：持久同调与 Vietoris–Rips 复形、持久图与瓶颈距离、泊松点过程模型、统计阈值筛选、最近邻距离分布、Beta 先验推断、贝叶斯置信区间、Ripser 计算、特征向量化（22维统计量）和线性 SVM。

**📊 数据集**

实验数据集：①合成圆点云+背景噪声；②双圆配置+噪声；③Sumner 与 Popović 的 3D 动物形状数据集（Camel、Elephant、Horse）及其噪声版本（共 6 类）。

**📈 对比分析**

比较方法：与原 CLA、Adaptive DBSCAN、LDOF、LUNAR 四种去噪/降采样算法对比；RCLA 在 H1 持久图的瓶颈距离从 0.229 降至 0.032，均值与标准差均优于其他方法；在 3D 分类实验中，准确率均≥99.5%，平均 99.88%，接近 100%。

**⚠️ 局限性**

局限性：依赖噪声为均匀 HPPP 的假设；对噪声分布非均匀或高维数据可能需要重新调整阈值和网格参数；算法仍需计算持久同调，规模扩展受限；细节特征可能在过度压缩时丢失。

---

## 212. GazeCLIP: Gaze-Guided CLIP with Adaptive-Enhanced Fine-Grained Language Prompt for Deepfake Attribution and Detection

**arXiv ID:** 2603.29295 | [PDF](https://arxiv.org/pdf/2603.29295v1)

**作者:** Yaning Zhang `[一作]` (Qilu University of Technology (Shandong Academy of Sciences)), Zan Gao `[通讯]` (Qilu University of Technology (Shandong Academy of Sciences))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于视觉‑语言模型的深度伪造归因与检测（DFAD）方法GazeCLIP，并构建细粒度DFAD评测基准；

**💡 创新点**

创新点在于利用眼动估计的差异作为伪造先验，设计视觉感知编码器、注视感知图像编码器和语言细粒化增强编码器，并引入可插拔的自适应词选择器（AWS）；

**🔧 技术方法**

采用CLIP预训练模型、ETH‑Xgaze眼动估计器、LoRA微调、细粒化文本提示、交叉模态对比损失及多任务损失；

**📊 数据集**

使用GenFace、DF40、CelebA‑HQ、FFHQ、Celeb‑DF++等数据集，并采集多种GAN、扩散、流模型生成的假图像；

**📈 对比分析**

与多种SOTA检测/归因方法对比，在未见生成器上平均提升约6.56%归因准确率、5.32% AUC，且在多领域、无缝生成器上保持强泛化；

**⚠️ 局限性**

局限在于需手工设计文本提示，对未知伪造类型的适应性有限，且模型参数量和计算成本相对较高。

---

## 213. Downsides of Smartness Across Edge-Cloud Continuum in Modern Industry

**arXiv ID:** 2603.29289 | [PDF](https://arxiv.org/pdf/2603.29289v1)

**作者:** Akhil Gupta Chigullapally `[一作]` (University of North Texas), Mohsen Amini Salehi `[通讯]` (University of North Texas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统性分析了智能工业在Edge‑Cloud连续体中的弊端，包括AI、IIoT、边缘云、LLM等技术带来的漏洞、网络威胁与副作用；

**💡 创新点**

首次将智能工业的安全风险从软件层到基础设施层进行统一框架，并结合急迫计算和大型语言模型等新兴技术，揭示潜在的交叉侧效应；

**🔧 技术方法**

采用概念模型、案例研究与风险图谱的方法，结合现有攻击案例和行业报告进行归纳；

**📊 数据集**

未使用公开数据集，而是引用工业事故、攻击实例和行业报告进行实证；

**📈 对比分析**

无实验比较；通过对比传统与智能化系统的风险程度，指出智能化在性能提升的同时伴随安全负担；

**⚠️ 局限性**

缺乏针对性防御方案，分析主要依赖案例和文献，未在实验环境中验证模型；对不同规模工业的适用性不完全统一。

---

## 214. PRISM: A Multi-View Multi-Capability Retail Video Dataset for Embodied Vision-Language Models

**arXiv ID:** 2603.29281 | [PDF](https://arxiv.org/pdf/2603.29281v1)

**作者:** Amirreza Rouhi `[一作]`, Sashi Reddi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个 270K 样本、三视角（第一人称、第三人称、360°全景）的视频微调语料库 PRISM，用于提升零售环境下的嵌入式视觉语言模型（VLM）在空间、时间、行动等多维感知任务上的表现。

**💡 创新点**

创新点包括：① 在单一部署域内统一覆盖空间、物理与行动三大知识维度；② 引入同步的 egocentric、exocentric 与 360° 视角数据，提供跨视角监督；③ 采用低成本混合标注策略（LLM 生成、物理视频推理、深度分析、自监督变换）实现可扩展、多样化的训练信号。

**🔧 技术方法**

技术手段主要是：① 使用 Cosmos‑Reason2‑2B（Qwen3‑VL）作为基准模型；② 通过 LoRA/QLoRA 进行参数高效微调；③ 在不同任务中采用开式回答、链式推理 (CoT) 与多项选择（MCQ）三种输出格式；④ 利用 Gemini 2.5 Flash、Gemini Robotics ER 1.5、DepthCrafter 等 LLM 与深度模型生成标注。

**📊 数据集**

所使用的数据集是 PRISM 自行构建的多视角视频 SFT 语料库（270K 任务样本，约 11.8M 帧、730M 令牌），并对比了 Cosmos‑Reason2 的原始训练数据。

**📈 对比分析**

比较方法：在 20+ 任务上使用 MCQ 准确率、分类精确率及 GPT‑4o 评估 CoT 质量；结果显示 PRISM 微调后平均准确率从 62.8% 提升至 86.6%，整体误差率降低 66.6%，不同视角和知识维度均显著提升，尤其是行动推理和跨视角匹配。

**⚠️ 局限性**

局限性包括：仅评估单一 2B 规模模型，难以说明对更大模型的泛化；评价指标主要是 MCQ/准确率与 GPT‑4o 自动评分，可能低估自然语言细节表现；实验场景局限于美式零售环境，缺乏多地理、多店铺的验证。

---

## 215. A Regulatory Compliance Protocol for Asset Interoperability Between Traditional and Decentralized Finance in Tokenized Capital Markets

**arXiv ID:** 2603.29278 | [PDF](https://arxiv.org/pdf/2603.29278v1)

**作者:** Jinwook Kim `[一作]` (Horizen Korea), Jonghun Hong `[通讯]` (Horizen Korea)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Regulatory Compliance Protocol（RCP），在传统金融与去中心化金融之间为资产代币化提供完整的监管合规框架，并通过债券发行、碳信用代币化以及 TradFi 与 DeFi 的互操作三大场景验证其可行性。

**💡 创新点**

创新点在于：①将全球15家监管机构的监管建议归纳为 31 项标准，拆分为可追溯性、保密性、可执行性、最终性和代币化五大监管组；②构建了 NEW‑EIP 以补足 ERC 系列在隐私、追溯等方面的不足；③将 DAML 合约模型与 RCP 结合，实现传统金融资产在 DLT 上的安全互操作；④在代币层面实现资产冻结、恢复、黑名单、强制清算、时间限制等多重合规控制。

**🔧 技术方法**

采用的技术包括：区块链与分布式账本（DLT）、以太坊虚拟机（EVM）智能合约、DAML 语言的合约建模、角色基准权限控制、Gasless（元交易）支持、资产类管理与 Token 供应控制等。

**📊 数据集**

主要使用的“数据集”是来自 15 家全球监管机构（WB、ISDA、IOSCO、IMF、FSB、FATF、BIS、SFC、HKMA、ESMA、FCA、MAS、FINMA、FINRA 等）的监管文件与指导意见，并通过三大场景（债券、碳信用、互操作）进行案例验证；没有使用传统意义上的实验数据集。

**📈 对比分析**

比较方法：将 RCP（以及 NEW‑EIP）与 ERC‑20、ERC‑1400、ERC‑3643 的功能做对应表对照，统计满足的监管条目数量；在三大场景中实现对应的伪代码与流程图，评估合规性与可操作性。性能表现显示：RCP 满足 25/31 条监管项（比 ERC‑1400 的 16/31、ERC‑3643 的 15/31、ERC‑20 的 5/31 更高）。

**⚠️ 局限性**

限制：RCP 在 ERC 协议层面无法覆盖的六项（如隐私、追溯等）仍需依赖外部基础设施（如 DAML、专用侧链、合规插件）实现；此外，文中未给出量化性能指标（如交易吞吐量、Gas 成本），仅从合规覆盖角度评估。

---

## 216. MemRerank: Preference Memory for Personalized Product Reranking

**arXiv ID:** 2603.29247 | [PDF](https://arxiv.org/pdf/2603.29247v1)

**作者:** Zhiyuan Peng `[一作]` (Santa Clara University), Yi Gong `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于LLM的购物偏好记忆框架，利用强化学习训练的记忆提取器将用户长历史压缩为可直接用于个性化产品重排序的结构化记忆。

**💡 创新点**

创新点在于：①构建端到端的个性化重排序基准与评估框架；②设计可训练的偏好记忆提取器，并通过RL优化以提升下游重排序准确率；③证明半结构化、证据驱动的提取提示比完全固定或完全自由提示更有效。

**🔧 技术方法**

采用大型语言模型（如Qwen2.5-7B-Instruct、GPT-4.1-mini、o4-mini）进行记忆提取与重排序，利用GRPO实现RL后训练，使用1-in-5 选取任务和思考标签提升解释性。

**📊 数据集**

使用Amazon-Review-2023与Amazon-C4数据集构建的Electronics类别样本，包含用户购买历史、查询、正负候选商品。

**📈 对比分析**

与无记忆、原始历史、不同LLM提取器（基线、GPT-5.2）、外部记忆方法（MR.Rec、Mem0）及仅追加商品上下文等做对比；在GPT-4.1-mini和o4-mini上，RL训练的记忆提取器分别提升约6–10个百分点（1-in-5准确率），显著优于所有基线。

**⚠️ 局限性**

局限性包括：仅在单一Electronics类别验证；仅使用商品元数据进行记忆提取；仅关注离线1-in-5重排序，未验证更大规模检索或多模态场景。

---

## 217. The Thiomi Dataset: A Large-Scale Multimodal Corpus for Low-Resource African Languages

**arXiv ID:** 2603.29244 | [PDF](https://arxiv.org/pdf/2603.29244v1)

**作者:** Hillary Mutisya `[一作]`, Maryruth Gathoni `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了覆盖10种非洲语言的多模态社区收集语料库Thiomi，并提供文本、音频与基准模型。

**💡 创新点**

结合移动优先、分层质量保证与社区协作，实现大规模跨语言数据收集与自动化评估，首次为东非语言提供ASR、MT、TTS基线。

**🔧 技术方法**

使用移动Web平台、VAD+自动分割、四阶段QA流程，以及Wav2Vec2‑BERT、NLLB‑200、VITS等模型进行训练与微调。

**📊 数据集**

Thiomi Dataset（601k+批准文本句子、385k+音频录音）以及补充的Common Voice Swahili语料。

**📈 对比分析**

在Common Voice上微调Wav2Vec2‑BERT，Swahili ASR WER降至3.24%（相对提升61%），MT BLEU在55–64之间，TTS MOS在3.4–4.1之间。

**⚠️ 局限性**

语域覆盖有限、缺少音调标注、拼写不统一、说话人多样性不足，未覆盖部分语言与专业领域。

---

## 218. Diffusion Mental Averages

**arXiv ID:** 2603.29239 | [PDF](https://arxiv.org/pdf/2603.29239v1)

**作者:** Phonphrm Thawatdamrongkit `[一作]` (VISTEC), Supasorn Suwajanakorn `[通讯]` (VISTEC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出Diffusion Mental Averages（DMA）方法，利用预训练扩散模型的语义空间，通过多噪声潜在轨迹对齐生成概念的清晰、真实的“心理平均”图像；

**💡 创新点**

创新点在于将平均化任务转化为在扩散模型内部的轨迹对齐，而非后处理像素或特征空间平均；同时结合轻量级文本倒置或LoRA实现多模态概念的模式分离；

**🔧 技术方法**

核心技术包括：扩散模型的h‑space语义提取、噪声潜在梯度优化、DDIM采样、CLIP/BLIP特征聚类、文本倒置/LoRA微调；

**📊 数据集**

主要使用Stable Diffusion（Realistic Vision v5.1、SD1.5等）以及DiT-XL作为模型；数据集为模型生成的随机样本，无需外部标注数据；

**📈 对比分析**

与GANgealing、Avg VAE、D^4M、MGD^3等基线比较，DMA在一致性、代表性和ImageReward三项指标上均优于基线，且生成的平均图像更清晰且更符合人类偏好；

**⚠️ 局限性**

限制包括高计算成本、对CFG等超参敏感、聚类依赖外部编码器的偏差、对多模态模式分离的效果受聚类质量影响等。

---

## 219. M2H-MX: Multi-Task Dense Visual Perception for Real-Time Monocular Spatial Understanding

**arXiv ID:** 2603.29236 | [PDF](https://arxiv.org/pdf/2603.29236v1)

**作者:** U. V. B. L. Udugama `[一作]`, Francesco Nex `[通讯]` (University of Twente)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `51c0528b-f690-4182-ae60-bb5f046c276c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并验证了一种实时多任务密集预测模型M2H-MX，能够在单目相机下直接嵌入现有SLAM pipeline，实现高质量的深度与语义感知。

**💡 创新点**

创新点在于冻结DINOv3 backbone并通过LoRA微调，结合注册门控的全局上下文注入和轻量化的跨任务混合（CTM+MSCA），在保持低延迟的同时显著提升深度与语义预测的一致性与准确性。

**🔧 技术方法**

使用了LoRA参数高效适配、Register-Gated Mamba解码、Cross-Task Mixer、Multi-Scale Convolutional Attention以及bin-based深度预测头等技术。

**📊 数据集**

训练与评估数据集包括NYUDv2、Cityscapes以及ScanNet。

**📈 对比分析**

与MTMamba++、M2H等多任务基线对比，mIoU提升约4–6个百分点、RMSE下降≈10%；在ScanNet中集成后平均ATE从17.6 cm降至6.9 cm，提升约60%。

**⚠️ 局限性**

局限在于仅支持单目输入，对极端遮挡或动态场景的鲁棒性尚待验证，且模型规模较大，无法在极低算力设备上部署。

---

## 220. HSFM: Hard-Set-Guided Feature-Space Meta-Learning for Robust Classification under Spurious Correlations

**arXiv ID:** 2603.29313 | [PDF](https://arxiv.org/pdf/2603.29313v1)

**作者:** Aryan Yazdan Parast `[一作]` (University of Melbourne), Naveed Akhtar `[通讯]` (University of Melbourne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HSFM（Hard-Set‑Guided Feature‑Space Meta‑Learning）方法，在冻结的预训练 backbone 输出的特征空间里，对支持样本进行可学习的编辑，并在此基础上只更新线性分类头，从而提升模型在存在伪相关性的情况下对少数群体和分布偏移样本的鲁棒性。

**💡 创新点**

创新点包括：① 将元学习直接应用于特征空间而非像素或标签空间；② 用硬样本（loss 最高的验证样本）驱动元优化，使支持样本编辑能够直接降低对这些难样本的 loss；③ 只冻结 backbone、仅更新线性头，极大降低计算成本与训练不稳定性；④ 通过 CLIP‑based unCLIP 生成可视化展示，进一步解释模型对伪相关性的依赖。

**🔧 技术方法**

使用技术主要有：bilevel meta‑learning、hard‑set 采样、特征空间支持嵌入优化、线性 head 的快速适配、冻结预训练 backbone（ResNet‑50、ViT‑B/16、ConvNeXt、CLIP‑ViT‑H）、unCLIP 可视化、以及对比实验的多种基线方法。

**📊 数据集**

实验数据集：伪相关性基准（Waterbirds、CelebA、Dominoes、MetaShift）；细粒度分类基准（Stanford Cars、CUB‑Birds、Oxford Flowers）；以及 CLIP‑based 视觉模型的相关验证。

**📈 对比分析**

与 ERM、DFR、GroupDRO、LISA、JTT、DaC、DDB 等方法在 worst‑group accuracy（WGA）和平均精度上对比。HSFM 在 Waterbirds、Dominoes 的 WGA 领先所有基线；在 CelebA 超越所有无组标签方法；在细粒度任务中获得显著提升；训练时间仅需 1–5 分钟，显著快于其他元学习或生成式增广方法。

**⚠️ 局限性**

局限性：依赖冻结的预训练 backbone 必须提供足够信息的特征；若 backbone 表示不足，单靠线性 head 可能难以弥补，需要额外微调 backbone，导致效率下降；对极端分布偏移或极端伪相关性的鲁棒性尚未完全验证。

---

## 221. LGFNet: Local-Global Fusion Network with Fidelity Gap Delta Learning for Multi-Source Aerodynamics

**arXiv ID:** 2603.29303 | [PDF](https://arxiv.org/pdf/2603.29303v1)

**作者:** Qinye Zhu `[一作]` (University of Electronic Science and Technology of China), Wenyong Wang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种本地-全局融合网络（LGFNet）用于多源气动数据融合，能够同时捕获局部高频特征（如冲击波）和全局低频趋势；

**💡 创新点**

核心创新包括滑动窗口感知层强化局部连续性、基于自注意力的关系推理层提取全局依赖、以及“保真差距Δ学习”（FGDL）将低保真CFD作为低频载体直接学习非线性偏差；

**🔧 技术方法**

使用滑动窗口机制、二维卷积层、全局多头自注意力、跳连码解码器、残差学习与FGDL训练策略，并在PyTorch框架下实现；

**📊 数据集**

实验数据集：RAE2822 空气foil 的表面压强分布（三组工作点）和 CARDC 真实飞机的高维力矩系数（C_x、C_y、C_z）；

**📈 对比分析**

与Hierarchical Kriging、DNN、MSFM、XGBoost GF、ArGEnT 等基准模型进行对比，评估 RMSE、MAE、R²、训练时间与不确定性。LGFNet 在所有评估指标上均取得最优或近优成绩，特别是压强分布的 RMSE 降低约 65%，C_z 系数的 RMSE 仅 0.0169，且不确定性显著降低；

**⚠️ 局限性**

局限性包括：自注意力仍具 O(N²) 复杂度，尚未采用稀疏/高效注意力；轻量化通道配置可能限制对极其复杂高频耦合（如 C_y）的表达；以及对极端稀疏数据的鲁棒性尚待进一步验证。

---

## 222. MELT: Improve Composed Image Retrieval via the Modification Frequentation-Rarity Balance Network

**arXiv ID:** 2603.29291 | [PDF](https://arxiv.org/pdf/2603.29291v1)

**作者:** Guozhi Qiu `[一作]` (Shandong University), Yupeng Hu `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MELT模型，通过稀有语义注意力和扩散去噪提升组合图像检索性能。

**💡 创新点**

创新点在于对稀有修改语义进行加权关注并使用扩散去噪消除硬负样本干扰。

**🔧 技术方法**

采用BLIP‑2 Q‑Former、跨模态注意力、Mahalanobis残差评估、DDIM扩散去噪、知识蒸馏和批量对比损失等技术。

**📊 数据集**

在FashionIQ和CIRR这两个公开组合图像检索基准上进行评估。

**📈 对比分析**

与多种基线在Recall@k等指标对比，MELT在FashionIQ上R@10提升约3%，在CIRR上R@1/R@5/R@10均提升，平均提升0.45%。

**⚠️ 局限性**

仍受限于训练规模、稀有度阈值需手工调参以及扩散去噪计算开销较大。

---

## 223. Lie Generator Networks for Nonlinear Partial Differential Equations

**arXiv ID:** 2603.29264 | [PDF](https://arxiv.org/pdf/2603.29264v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 224. Sima AIunty: Caste Audit in LLM-Driven Matchmaking

**arXiv ID:** 2603.29288 | [PDF](https://arxiv.org/pdf/2603.29288v1)

**作者:** Atharva Naik `[一作]` (University of Illinois Urbana-Champaign), Koustuv Saha `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对印度婚配情境下的五大大型语言模型进行受控实验，系统评估其在种姓与收入变化下的匹配评估结果，揭示同种姓偏好和层级排序。

**💡 创新点**

首次将种姓作为社会结构变量纳入LLM评估的受控实验，量化同种姓匹配和跨种姓层级差异，比较不同模型的表现并提供回归分析支持。

**🔧 技术方法**

使用Prompt Engineering设计统一评估模板，强制LLM输出JSON分数，并利用线性混合效应回归对评估分数与种姓、收入等变量进行统计建模。

**📊 数据集**

基于Shaadi.com公开的匿名化婚配档案，构造5个种姓、5个收入层级的合成变体，总计2500名男、2500名女档案用于实验。

**📈 对比分析**

对GPT、Gemini、Llama、Qwen和BharatGPT五模型进行相同输入评估；通过回归显著性、R²、同种姓/跨种姓分数差距等指标量化表现，发现所有模型均显著偏好同种姓，并呈现传统种姓层级排序；差异在模型间存在但总体趋势一致。

**⚠️ 局限性**

实验仅采用粗粒度种姓分类，未覆盖子种姓或地区差异；仅关注异性婚配，排除非二元/同性恋情境；提示和输出格式单一，无法洞察内部推理；可能存在其他社会属性间接影响但未被显式控制。

---

## 225. MaskAdapt: Learning Flexible Motion Adaptation via Mask-Invariant Prior for Physics-Based Characters

**arXiv ID:** 2603.29272 | [PDF](https://arxiv.org/pdf/2603.29272v1)

**作者:** Soomin Park `[一作]` (KAIST), Sung-Hee Lee `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本工作提出MaskAdapt框架，实现了基于物理的角色控制中的灵活运动适配；

**💡 创新点**

核心创新点是引入遮挡不变的基础策略与残差策略相结合的两阶段训练，并通过遮挡一致性正则化提升在局部观测缺失下的鲁棒性；

**🔧 技术方法**

技术手段包括基于PPO的策略学习、遮挡一致性KL正则化、判别器对不同遮挡状态的监督以及结合文本条件扩散模型的运动目标生成；

**📊 数据集**

使用了LAFAN1数据集训练基础策略，AMASS数据集用于运动组合任务，BABEL及其生成的文本驱动轨迹用于部分运动跟踪；

**📈 对比分析**

与AMP、CML、AdaptNet、MaskedMimic等基线比较，MaskAdapt在覆盖度、成功率和跟踪误差等指标上均优于或持平于对手，尤其在目标驱动和文本驱动部分跟踪任务中表现突出；

**⚠️ 局限性**

局限性包括在近似相关身体部位赋予冲突运动时可能产生不一致，且对预训练扩散模型的语义约束和失效模式存在依赖。

---

## 226. From Physics to Surrogate Intelligence: A Unified Electro-Thermo-Optimization Framework for TSV Networks

**arXiv ID:** 2603.29268 | [PDF](https://arxiv.org/pdf/2603.29268v1)

**作者:** Mohamed Gharib `[一作]` (University of Illinois Chicago), Inna Partin-Vaisband `[通讯]` (University of Illinois Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一套可扩展的电热耦合 TSV 网络设计与优化框架。

**💡 创新点**

创新点在于将物理驱动的解析模型与图神经网络 surrogate 结合，实现从解析到高精度预测的多层迁移学习，并支持大规模设计空间的多目标 Pareto 优化。

**🔧 技术方法**

采用了物理信息化的解析电磁与热模型、图神经网络（GNN）surrogate、Sim-to-Sim 迁移学习、全波 FEM 校验以及自动化 PyAEDT 工作流。

**📊 数据集**

使用基于 RLCG 电路的 100k 解析样本和 10k Ansys HFSS 高精度样本作为训练数据，后续在真实 TSV 3×3–15×15 网格上进行验证。

**📈 对比分析**

与 HFSS 全波仿真对比，解析模型 RFE 5–10%，GNN surrogate RFE <2%，计算速度提升 10⁶ 倍，能够在数分钟内评估百万级配置并生成 Pareto 前沿。

**⚠️ 局限性**

局限在于完全连通图导致 N⁴ 边缘增长，限制了极大网格规模；同时对高度非均匀或多层 TSV 结构的泛化尚未验证。

---

## 227. Monocular Building Height Estimation from PhiSat-2 Imagery: Dataset and Method

**arXiv ID:** 2603.29245 | [PDF](https://arxiv.org/pdf/2603.29245v1)

**作者:** Yanjiao Song `[一作]` (Wuhan University), Walter Musakwa `[通讯]` (University of Johannesburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了PhiSat-2–Height Dataset (PHDataset)并提出了双流Ordinal网络（TSONet），同时实现了建筑高度估计与足迹分割的联合任务；

**💡 创新点**

创新点在于（1）交叉流交换模块（CSEM）实现足迹与高度特征的选择性交互；（2）Feature-Enhanced Bin Refinement（FEBR）采用多级特征引导的序数回归细化；（3）结合多任务学习与空间加权损失，提升结构一致性与高度精度；

**🔧 技术方法**

技术包括U-Net风格的多尺度编码器、组卷积、残差块、Restormer、交叉注意力、64-bin ordinal回归、Tversky+BCE分割损失以及空间加权MAE；

**📊 数据集**

使用PhiSat-2–Height Dataset (PHDataset)，共9,475张256×256像素的PhiSat-2七波段图像与对应开源建筑高度标签，覆盖26个全球城市；

**📈 对比分析**

通过与U-Net、传统MDE方法（Eigen, Laina, BTS, LocalBins, DepthFormer, BinsFormer）及MHE方法（Image2Height, DORN-height, HTC-DC Net, FusedSeg-HE）对比，TSONet在MAE、RMSE、IoU和F1方面分别比最强对手低13.2%/9.7%和提升14.0%/10.1%，整体表现最佳；

**⚠️ 局限性**

局限在于PhiSat-2产品的几何失真与波段间偏移、标签时空不一致以及建筑高度的长尾分布，导致高楼预测误差仍较大，需进一步改进数据质量和网络鲁棒性。

---

## 228. Long-Reach Robotic Cleaning for Lunar Solar Arrays

**arXiv ID:** 2603.29240 | [PDF](https://arxiv.org/pdf/2603.29240v1)

**作者:** Stanley Wang `[一作]` (Stanford University), Mark Cutkosky `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并测试了一种配备可伸缩轻量化可部署桁杆的移动机器人，用于在月球表面对大型太阳能阵列进行轻柔清洁。

**💡 创新点**

创新点在于将可部署桁杆与弹性手腕结合，并引入基于任务空间阻抗的速度模态控制，实现了在低刚度长臂下的稳定接触力调节。

**🔧 技术方法**

主要技术包括可部署复合桁杆结构、六轴力传感器、弹性手腕自对齐、速度模态阻抗控制与基于视觉的终端伺服。

**📊 数据集**

使用的是实验台面白板的平面平滑模型作为仿真面板，未使用公开数据集。

**📈 对比分析**

通过在实验台面上进行接触力跟踪实验，显示机器人能在1–10 N范围内维持±0.2 N的RMS误差，验证了控制器在低接触力下的稳健性。

**⚠️ 局限性**

局限性包括：仅在二维平面上测试；在切向运动开始时受抑制现象；未在低重力条件下验证；缺乏复杂曲面或倾斜面控制与移动底盘协同研究。

---

## 229. Stochastic Dimension Implicit Functional Projections for Exact Integral Conservation in High-Dimensional PINNs

**arXiv ID:** 2603.29237 | [PDF](https://arxiv.org/pdf/2603.29237v1)

**作者:** Zhangyong Liang `[一作]` `[通讯]` (Tianjin University), Zhangyong Liang (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种 Stochastic Dimension Implicit Functional Projection (SDIFP) 框架，在物理信息神经网络（PINN）中通过连续仿射投影实现严格的宏观守恒。

**💡 创新点**

创新点包括：① 将投影从离散空间向量转移到连续函数空间，只需求解两维代数方程；② 使用分离的蒙特卡罗（MC）积分实现 mesh‑free 的守恒计算；③ 引入双随机无偏梯度估计器（DS‑UGE），显著降低逆向传播内存复杂度，解决高维微分算子内存灾难。

**🔧 技术方法**

核心技术包括：连续仿射功能投影、低差异低维闭式根求解、detached MC 量化、双随机无偏梯度估计、随机维度采样子集、Sobol' 序列、自动微分（AD）分离。

**📊 数据集**

使用合成 PDE 数据集：1D/2D/3D 线性/非线性方程（输运方程、反应扩散、波动方程、KdV 方程）以及高维（至 1000 维）Sine‑Gordon 方程，所有实验均在 NVIDIA A100 GPU 上运行。

**📈 对比分析**

与 vanilla PINN、PINN‑SC（软约束）、PINN‑proj（离散投影）、PINN‑KTT（约束投影）等方法比较。SDIFP 在所有维度和采样方式下，守恒误差降低 3–7 个数量级；内存消耗随维度的指数增长被压缩为线性；在高维（1000D）下仍能保持 10⁻⁵ 级的守恒精度，显著优于其他方法。

**⚠️ 局限性**

局限性包括：不天然兼容强制 Dirichlet 边界条件；需要大规模 detached MC 采样（M≈10⁵），对计算资源有一定需求；在网络初始平坦时 σ² 接近零需数值松弛；未在真实物理数据集上进行验证。

---

## 230. SysOM-AI: Continuous Cross-Layer Performance Diagnosis for Production AI Training

**arXiv ID:** 2603.29235 | [PDF](https://arxiv.org/pdf/2603.29235v1)

**作者:** Yusheng Zheng `[一作]` (University of California Santa Cruz), Tao Ma `[通讯]` (Alibaba Group)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并部署了一个持续、跨层（CPU、GPU、NCCL）的性能诊断系统，可在生产 AI 训练中实时采样并通过分层差分分析定位瓶颈，已在阿里巴巴 80,000+ GPU 的生产环境使用超过一年。

**💡 创新点**

核心创新包括：
1) 基于 eBPF 的低开销持续采样，覆盖内核、用户和 NCCL 事件；
2) 自适应混合 FP+DWARF 堆栈展开，既保持 95% 的堆栈准确率，又避免 10% 级别的 CPU 开销；
3) 集中化 BuildID‑索引符号解析，消除节点内存压力和稀疏符号误判；
4) 无需框架耦合的 NCCL 追踪，支持多种训练框架；
5) 分层差分诊断方法，将 GPU、CPU、OS 级数据进行同 rank、跨 rank、对比历史基线的统计比较，系统化定位根因。

**🔧 技术方法**

采用技术包括：eBPF、CUDA runtime uprobes、NCCL 结构解析、FP+DWARF 混合堆栈展开、BuildID 集中符号解析、统计学阈值、事件驱动的中心分析服务、跨节点低延迟数据管道。

**📊 数据集**

实验数据来源为阿里巴巴生产环境：10,000+ 节点、80,000+ GPU，处理约 400 TiB 的采样数据；在 Llama‑3.2‑1B‑Instruct 训练任务中评估吞吐率；对生产 AI 训练作业进行案例分析，覆盖 GPU 热、网络、OS 以及软件层问题。

**📈 对比分析**

与现有工具（Nsight Systems、Strobelight、async‑profiler、MegaScan、DeepSpeed Prof 等）对比：
- 采样 10% 时吞吐率损失仅 0.33%，100% 仍低于 1.72%；
- 堆栈准确率从 FP‑only 的 5% 提升至 95%；
- 6 个月内共诊断 94 个确认事件，平均诊断时间 10 分钟，远快于传统数日的手工分析；
- 诊断覆盖率包括 GPU 硬件、OS 阻塞、网络、软件等多种根因。

**⚠️ 局限性**

局限性：
1) 仅在 Linux eBPF、CUDA runtime uprobe、特定 NCCL 版本上可运行；
2) 不支持非 CPython 或无 ELF 映射的 JIT 运行时；
3) 假设异常 rank 数量少，若大多数 rank 同时下降则统计模型失效；
4) 只能捕捉在 CPU 上执行的阻塞，无法检测 I/O 阻塞或 RDMA 级网络细粒度问题；
5) GPU 侧堆栈无覆盖，需要配合 Nsight Compute 等工具；
6) 中心分析服务为单点，若失效需缓冲，导致短期延迟。

---

## 231. Long-Document QA with Chain-of-Structured-Thought and Fine-Tuned SLMs

**arXiv ID:** 2603.29232 | [PDF](https://arxiv.org/pdf/2603.29232v1)

**作者:** Zhuowen Liang `[一作]` (Hong Kong University of Science and Technology), Nan Tang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种两阶段框架——Chain-of-Structured-Thought（CoST）与Group Relative Policy Optimization（GRPO），通过先用大型语言模型（LLM）一次生成结构化推理轨迹与序列化结构化输出，再将这种结构化推理能力迁移到小型语言模型（SLM）上，实现长文档问答的高准确率与低延迟；

**💡 创新点**

创新点在于：①将推理过程拆分为结构化思考（CoST），生成可验证的推理轨迹和结构化结果；②采用双层奖励的强化学习GRPO，既奖励答案质量与格式，又奖励过程一致性；③通过一次LLM调用产生可监督的训练数据，显著降低SLM训练成本；

**🔧 技术方法**

核心技术包括：大模型链式思考（CoST）模板、结构化输出生成、LLM‑as‑Judge评估、LoRA微调、GRPO强化学习、双层奖励设计（格式、答案、过程），以及对话式结构化数据生成；

**📊 数据集**

使用的数据集包括：FinQA、TAT-QA、Squad、LegalBench用于构建训练集，Loong benchmark（涵盖金融、法律、论文等三域）用于评测，Open‑Domain LongBench作为进一步验证；

**📈 对比分析**

与多种基线（LLM Zero‑shot/CoT、现有IE模型、模组化提取框架）比较，实验表明：在Loong金融子集，3B/7B SLM在精度和完美率上分别提升约27.6/17.8点，且比GPT‑4o‑mini/DeepSeek‑R1更快2–4×；在法律子集同样显著提升；在Open‑Domain LongBench中，Qwen‑LiteCoST能超过GPT‑4o并取得最高F1分；

**⚠️ 局限性**

局限性包括：对其他领域的泛化尚未充分验证；训练数据来源有限，缺乏足够多样化的长文档问答数据；模型仍需在更复杂语义推理和极长文本场景下进一步评估与改进。

---

## 232. Derived Fields Preserve Fine-Scale Detail in Budgeted Neural Simulators

**arXiv ID:** 2603.29224 | [PDF](https://arxiv.org/pdf/2603.29224v1)

**作者:** Wenshuo Wang `[一作]` (South China University of Technology), Fan Zhang `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在固定存储预算下，如何通过优化携带的物理场与位分配，提升神经模拟在长期自回归滚动中的细尺度保真度。

**💡 创新点**

创新点在于提出Derived-Field Optimization (DerivOpt)，将携带状态设计视为与架构、损失、滚动策略同等重要的设计轴，并给出闭式评分函数实现自动化优化。

**🔧 技术方法**

技术手段包括基于谱投影的抗混叠粗化、量化模型分析、对不同物理场的频率失真建模，以及在多种 PDEBench 任务上进行的全流程训练与评估。

**📊 数据集**

使用的数据集为 PDEBench 的时间序列前向子集（1D/2D 广播、布格、扩散-反应、雷达破裂、不可压 Navier–Stokes、可压 Navier–Stokes 等）。

**📈 对比分析**

与四种常见自回归网络（U‑Net、FNO、ConvLSTM、Transformer）以及多尺度、监督、学习压缩等基线对比，DerivOpt 在大多数配置下实现了最低的平均 nRMSE、最高的细尺度保真时间以及最多配置的优势。

**⚠️ 局限性**

局限性包括：仅考虑二维问题，候选物理场集合有限；缺乏对三维、实时在线重校准以及更大规模自适应压缩的评估。

---

## 233. MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting

**arXiv ID:** 2603.29296 | [PDF](https://arxiv.org/pdf/2603.29296v1)

**作者:** Haoran Zhou `[一作]` (National University of Singapore), Gim Hee Lee `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出MotionScale框架，实现从单目视频重建高保真4D场景的全流程。

**💡 创新点**

创新点在于使用可扩展的聚类基运动场，结合自适应分割/裁剪与三阶段渐进优化，显著提升几何与运动的长期一致性。

**🔧 技术方法**

采用3D高斯投影、聚类运动场、三阶段运动一致性优化、光影高斯、2D深度/掩码/点跟踪先验以及相机姿态细调等技术。

**📊 数据集**

在DAVIS、DyCheck和NVIDIA Dynamic Scenes三个公开基准上进行实验。

**📈 对比分析**

与Shape of Motion、GFlow、T‑NeRF、HyperNeRF、DynIBaR、Deformable‑GS等方法对比，MotionScale在PSNR/SSIM/LPIPS以及3D/2D跟踪精度上均大幅优于对手。

**⚠️ 局限性**

仍受限于单目先验噪声、长序列漂移处理难度、对快速遮挡和复杂光照的鲁棒性不足，以及对高端GPU的依赖。

---

## 234. Self-Consistency for LLM-Based Motion Trajectory Generation and Verification

**arXiv ID:** 2603.29301 | [PDF](https://arxiv.org/pdf/2603.29301v1)

**作者:** Jiaju Ma `[一作]` (Stanford University), Maneesh Agrawala `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用自一致性方法，在运动图形动画中通过多次LLM采样生成运动轨迹，并对其进行几何变换不变距离聚类，自动恢复轨迹的形状族与变换组，以提升轨迹生成与验证的准确性。

**💡 创新点**

提出基于Lie变换群的形状族框架和对应的不变距离度量，使自一致性从文本推理迁移到视觉轨迹空间；同时设计无监督的变换组判定标准和验证策略，实现训练‑free、无监督的生成与验证。

**🔧 技术方法**

大语言模型（GPT‑4.1/ GPT‑5）用于多样化轨迹采样；DBSCAN聚类；ICP变体用于计算变换不变距离；两种决策标准用于选择最佳变换组；与VLM（GPT‑4.1/ GPT‑5）和直接LLM采样做对比。

**📊 数据集**

合成基准集：224个文本提示与对应的形状族（共2240条轨迹，10条正样本与10条负样本），包括35种常见几何基形。

**📈 对比分析**

与直接LLM采样和VLM验证对照，生成任务中准确率提升约4–6%（GPT‑4.1 62.1%→≈67%；GPT‑5 79.1%→≈84%），验证任务中F1从79提升至84.6（无监督），与oracle设置相当。

**⚠️ 局限性**

仅适用于单原型形状族；对含多原型或模糊描述的提示效果不佳；依赖LLM采样分布满足假设，若分布不均或错误样本与正确样本聚类同一组，可能导致变换组误判。

---

## 235. Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration Under Large Initial Perturbations

**arXiv ID:** 2603.29414 | [PDF](https://arxiv.org/pdf/2603.29414v1)

**作者:** Ni Ou `[一作]` (Beijing Institute of Technology), Junzheng Wang `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于外参感知的跨模态注意力框架，直接在摄像头图像与 LiDAR 点云的原始域中对齐图像块与点群，进行特征融合并回归外参更新。

**💡 创新点**

创新点包括：①将外参假设注入跨模态注意力的位置信息；②使用多频调和嵌入（harmonic embedding）提升空间敏感性；③通过投影余量 (projection margin) 扩大视场，保留图像外的点云信息，显著提升大误差下的鲁棒性；④对旋转和平移分别设计独立聚合分支，提升精度。

**🔧 技术方法**

技术手段包括：DINOv2 视觉 Transformer 作为图像编码器；PointGPT 作为点云编码器；多头跨模态注意力（scale‑free cross‑attention）；投影余量与调和嵌入的坐标对齐；卷积+MLP 聚合回归外参；三步迭代细化。

**📊 数据集**

在 KITTI Odometry 与 nuScenes 两大自动驾驶数据集上进行实验，使用 64/32 声束 LiDAR 与 RGB 图像，数据量与划分按论文所述。

**📈 对比分析**

与多种基线（CoFiI2P、DirectCalib、CalibAnything、CalibNet、RGGNet、LCCNet、LCCRAFT、CalibDepth）比较。结果显示：在所有初始化误差范围内，本文在旋转、平移 RMSE 上均优于或相当于最优对手，并在 L1（<1° / <2.5 cm）和 L2（<25° / <25 cm）成功率上显著领先，尤其在 nuScenes 上 99%+ 的成功率。

**⚠️ 局限性**

局限性：①依赖较高的计算资源（Transformer + 点云网络）和多步迭代；②对稀疏 LiDAR（如 16 声束）或极端动态场景的鲁棒性未充分验证；③方法主要针对相机‑LiDAR 的几何对齐，缺少对语义或结构化特征（如边缘、线段）的进一步利用。

---

## 236. Exact Separation of Words via Trace Geometry

**arXiv ID:** 2603.29411 | [PDF](https://arxiv.org/pdf/2603.29411v1)

**作者:** Zeyu Chen `[一作]` (Zhejiang University), Junde Wu `[通讯]` (Zhejiang University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了正向词差在两状态MO-QFA中的精确分离问题，并给出切片驱动的证据机制。

**💡 创新点**

创新在于将元可变性与自由群的Fox微分、metabelian多项式关联，并构造一系列低维切片实现精确分离。

**🔧 技术方法**

使用了Fox微分、metabelian多项式、SU(2)的trace几何、Laurent矩阵识别、以及符号和数值分析等技术。

**📊 数据集**

采用了随机正向词差数据集，如#_a=#_b=20的50000个样本进行实验。

**📈 对比分析**

通过组合主切片和内部点测试，实验未出现误判，说明方法对绝大多数难例有效。

**⚠️ 局限性**

局限在于对“超退化”词差的检测仍缺乏，且无法通过固定有限图像测试完成。

---

## 237. Hallucination-aware intermediate representation edit in large vision-language models

**arXiv ID:** 2603.29405 | [PDF](https://arxiv.org/pdf/2603.29405v1)

**作者:** Wei Suo `[一作]` (Northwestern Polytechnical University), Yanning Zhang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出HIRE框架，在LVLM的中间表示层动态检测并编辑幻觉相关特征，避免重训练或双重推理。

**💡 创新点**

通过无参数更新的特征编辑、轻量化Router实现只编辑高幻觉风险token，且可通过α调节幻觉程度，实现可控幻觉。

**🔧 技术方法**

采用对比学习与DPO优化的Editor和Router，使用语义与幻觉子空间编码、注意力融合、重构与编辑损失。

**📊 数据集**

在MSCOCO上构建训练样本，评估于CHAIR、POPE、AMBER基准，同时验证MME和SEED-Bench的通用性。

**📈 对比分析**

与检索式和对比解码方法（Octopus、VCD、SID等）对比，CHAIR、POPE、AMBER得分分别提升40-50%，AMBER提升至最高；推理开销仅略增。

**⚠️ 局限性**

训练时需使用全部token和层，易受噪声影响；缺乏对关键隐藏状态的选择，数据效率待提升。

---

## 238. Deep Learning-Assisted Improved Differential Fault Attacks on Lightweight Stream Ciphers

**arXiv ID:** 2603.29382 | [PDF](https://arxiv.org/pdf/2603.29382v1)

**作者:** Kok Ping Lim `[一作]` (Xiamen University Malaysia), Iftekhar Salam `[通讯]` (Xiamen University Malaysia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了基于深度学习辅助的差分故障攻击（DFA），针对轻量级流密码ACORNv3、MORUSv2和ATOM，在单比特翻转、无控制故障位置的模型下完成实验；

**💡 创新点**

①使用多层感知机（MLP）识别故障位置，显著提升准确率；②提出阈值策略优化故障注入次数；③首次在ATOM上实施实验，验证双键过滤的安全边界；

**🔧 技术方法**

采用多层感知机（MLP）进行故障位置分类，SageMath实现 Gröbner 基础求解，阈值策略辅助方程构造；

**📊 数据集**

自行生成差分密钥流数据集，包含随机密钥与IV、差分密钥流及故障位置标签，用于训练、测试与验证；

**📈 对比分析**

与传统签名法和已有set‑based方法对比，ACORNv3与MORUSv2的准确率分别达到99.988%和99.923%，ATOM为82.356%，故障注入次数显著降低，攻击复杂度低于现有工作；

**⚠️ 局限性**

对ATOM的MLP准确率仍偏低，导致无法完整恢复状态；对大状态密码MORUS仍需大量故障注入；实验仅在仿真环境中完成，真实硬件实现尚待验证。

---

## 239. Parameterized Algorithms for Computing MAD Trees

**arXiv ID:** 2603.29381 | [PDF](https://arxiv.org/pdf/2603.29381v1)

**作者:** Tom-Lukas Breitkopf `[一作]` (Technische Universität Berlin), Camille Richer `[通讯]` (Université Paris-Dauphine)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文研究了最小平均距离（MAD）树问题的参数化算法，探索其在不同图宽度参数下的可解性。

**💡 创新点**

创新点在于证明该问题在模块宽度、顶点完整性和某些上界参数下为FPT，并给出对应的高效算法，同时证明在拆分图上为NP‑难。

**🔧 技术方法**

主要技术包括在树分解上进行动态规划、在模块宽度分解上采用分支搜索，并利用Wiener指数的边贡献公式进行状态压缩。

**📊 数据集**

论文未使用公开数据集，全部以理论分析与算法复杂度证明为主。

**📈 对比分析**

在常数模块宽度图上实现线性时间；在树宽k图上实现 2^O(2^k)·n^O(k) 时间；在拆分图上证明问题不可多项式解，表明算法的可行边界。

**⚠️ 局限性**

主要局限在于对树宽仍是XP级别未达到FPT；算法仅适用于无权无向图，且未扩展至其他宽度参数或带权情况。

---

## 240. Finite-time analysis of Multi-timescale Stochastic Optimization Algorithms

**arXiv ID:** 2603.29380 | [PDF](https://arxiv.org/pdf/2603.29380v1)

**作者:** Kaustubh Kartikey `[一作]`, Shalabh Bhatnagar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对基于零阶信息的多时钟随机近似算法进行了有限时间分析，提出了两种算法：两时钟梯度型GSF1和三时钟牛顿型NSF1，并给出了它们在平均成本优化下的收敛上界；

**💡 创新点**

创新点在于首次给出多时钟零阶Newton型算法的有限时间误差分析，明确了梯度与Hessian估计误差传播对优化动态的影响，并通过精细的误差分解实现了接近最优的步长选择；

**🔧 技术方法**

采用了平滑函数（SF）技术获取梯度和Hessian估计，结合多时钟随机逼近、O.D.E. 与弱收敛分析、均方误差分解以及矩估计等方法；

**📊 数据集**

实验使用Mountain Car连续控制环境，构建无终止的无穷期模拟器来评估平均成本；

**📈 对比分析**

与传统梯度型方法（GSF1）进行对比，主要指标为平均成本和梯度范数的衰减；实验结果表明NSF1在相同步长设置下收敛更快、成本下降更显著；

**⚠️ 局限性**

局限性包括：理论上仍需假设有限状态马尔科夫链和多重步长分离；估计方差较大，导致实测收敛速率可能低于理论下界；算法对步长参数敏感，需手动调优；

---

## 241. Assessing Multimodal Chronic Wound Embeddings with Expert Triplet Agreement

**arXiv ID:** 2603.29376 | [PDF](https://arxiv.org/pdf/2603.29376v1)

**作者:** Fabian Kabus `[一作]` (University of Freiburg), Harald Binder `[通讯]` (University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了面向RDEB创口的多模态表示框架，通过专家三元组判断对视觉与文本特征进行自监督微调与软序数嵌入；

**💡 创新点**

提出基于专家三元组的评价指标和软序数嵌入方法，结合视觉注意力池化与非对比学习实现少样本下的精准病变表征，并实现视觉与文本的互补融合；

**🔧 技术方法**

技术包括视觉基础模型（DermLIP）在创口区域的注意力池化+VICReg非对比学习；文本使用大型语言模型进行三元组推理并通过SOE（Soft Ordinal Embedding）构建嵌入；融合策略包括不确定性加权融合与相似度乘积融合；

**📊 数据集**

使用来自21例RDEB患者的53张创口摄影图（共120个标注创口）和513个专家三元组，另外通过LLM生成6.9万+个三元组进行文本嵌入；

**📈 对比分析**

与多种视觉（SigLIP、PanDerm、SAM、DermLIP等）和文本（RoBERTa、PubMedBERT、MiniLM、MedGemma等）基线比较，SOE提升3.7pp至67.4%，视觉注意力+VICReg提升3.7pp至71.6%，双模融合后获得73.5%的一致率，较单模优差5.6pp；

**⚠️ 局限性**

局限性在于数据规模极小，依赖专家三元组注释，模型仍未完全达到专家一致率，且对不同病程和拍摄条件的泛化能力待验证。

---

## 242. L-ReLF: A Framework for Lexical Dataset Creation

**arXiv ID:** 2603.29346 | [PDF](https://arxiv.org/pdf/2603.29346v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 243. AP-DRL: A Synergistic Algorithm-Hardware Framework for Automatic Task Partitioning of Deep Reinforcement Learning on Versal ACAP

**arXiv ID:** 2603.29369 | [PDF](https://arxiv.org/pdf/2603.29369v1)

**作者:** Enlai Li `[一作]` (Hong Kong University of Science and Technology), Wei Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了AP‑DRL框架，自动将深度强化学习训练任务在AMD Versal ACAP的CPU、FPGA和AI Engine之间划分，并实现硬件感知的混合精度量化；

**💡 创新点**

创新点在于（1）结合ILP模型实现层级任务划分，动态匹配不同运算强度到最合适的计算单元；（2）提出针对BF16/FP16/FP32的硬件感知量化算法，既保留BF16的指数范围，又通过FP16动态损失缩放提升FPGA性能；（3）整合两项技术形成统一的硬件/软件协同优化框架；

**🔧 技术方法**

使用LLVM IR生成计算图、TAPCA/CHARM进行性能探索、ILP求解划分、硬件加速编译（Vitis）、BF16/FP16/FP32混合精度训练；

**📊 数据集**

在Atari（Breakout、MsPacman）、OpenAI Gym（CartPole、InvPendulum、LunarLanderContinuous、MountainCarContinuous）和MuJoCo环境下，分别评估DQN、DDPG、A2C、PPO四种算法；

**📈 对比分析**

与AIE‑only（FP32）和FIXAR（固定点FPGA加速）做对比，实验显示AP‑DRL在训练吞吐量上可比AIE‑only提升1.61–3.82倍，较FIXAR提升0.98–4.17倍；同时保持收敛，平均奖励误差在1.12%–4.81%之间；

**⚠️ 局限性**

局限性包括：仅针对训练阶段，未覆盖推理和环境交互；实验基于硬件仿真，功耗与资源利用未测；在低FLOPs模型上量化同步开销仍显著，导致加速不明显；对特定的Versal ACAP平台依赖强。

---

## 244. CIPHER: Counterfeit Image Pattern High-level Examination via Representation

**arXiv ID:** 2603.29356 | [PDF](https://arxiv.org/pdf/2603.29356v1)

**作者:** Kyeonghun Kim `[一作]` (OUTTA), Hyuk-Jae Lee `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过重新利用并微调GAN和扩散模型的判别器，构建了一种通用的深度伪造检测框架CIPHER。

**💡 创新点**

创新点在于把生成模型的判别器迁移到检测任务，并融合GAN的多尺度特征和扩散模型的时序一致性特征，实现跨模型的生成无关检测。

**🔧 技术方法**

使用ProGAN判别器、DDPM/DDIM扩散网络、跨模型微调、特征融合与集成学习等技术。

**📊 数据集**

使用CelebA-HQ和FFHQ两个高质量人脸数据集进行训练与评估。

**📈 对比分析**

与ViT、Xception等现有基线相比，CIPHER在九种主流生成器上平均F1达74.33%，比平均基线高30%以上，尤其在难识别数据集如CIFAKE上取得88% F1。

**⚠️ 局限性**

局限性在于仅在学术数据集上评估，未验证在真实社交媒体等“野生”环境下的鲁棒性。

---

## 245. Developing a Guideline for the Labovian-Structural Analysis of Oral Narratives in Japanese

**arXiv ID:** 2603.29347 | [PDF](https://arxiv.org/pdf/2603.29347v1)

**作者:** Amane Watahiki `[一作]`, Hitomi Yanaka `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出并验证了首个可复制、系统化的日本语 Labovian 结构化叙事注释准则，并基于痴呆症照护者访谈构建了示例数据集，涵盖句子分割、叙事类型识别及微观（Narrative/Free/Restricted）与宏观（Abstract/Orientation/Complication/Evaluation/Resolution/Coda）层级标签。

**💡 创新点**

创新点在于：①在日本语语料中完整保留 Labov 的六大宏观功能并扩展到习惯性与假设性叙事；②为日本语特有的句法结构（如名词化时间词）制定明确分割规则；③提供公开可复制的注释流程和决策图，促进后续大规模标注与工具开发。

**🔧 技术方法**

技术方法包括：手工注释并使用 Fleiss' κ、Krippendorff α、Boundary Edit Distance 等统计指标评估一致性；采用决策图（微观层级）与功能矩阵（宏观层级）进行标注；借助 R、Python 等工具计算交叉验证指标。

**📊 数据集**

数据集为 16 次访谈（共 965 句子）来自日本痴呆症家庭照护者，聚焦其对幸福与困难的叙事，采用三位博士后共同标注，最后通过多数投票确定 gold 标注。

**📈 对比分析**

与先前英文 Labovian 注释数据（如 Saldias & Roy 2020）相比，本工作在句子分割的一致性（κ=0.80）显著提升；微观层级一致性 α≈0.40，宏观层级 α≈0.45，均高于以往细粒度标注结果。性能上，最可靠标签为 Narrative（微观）和 Complication（宏观），但 Restricted、Resolution、Coda 等标签一致性仍低。

**⚠️ 局限性**

局限性包括：①对假设性叙事缺乏宏观/微观标签；②部分标签（Restricted、Resolution、Coda）因时间标注不确定导致一致性低；③数据量相对有限，难以全面评估模型泛化；④目前仍缺乏自动化工具，仅为手工标注，难以规模化。

---

## 246. Open Machine Translation for Esperanto

**arXiv ID:** 2603.29345 | [PDF](https://arxiv.org/pdf/2603.29345v1)

**作者:** Ona de Gibert `[一作]`, Lluís de Gibert `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了开放源代码的 Esperanto 机器翻译，系统性评估并训练了从/向英语、西班牙语、加泰罗尼亚语的翻译模型。

**💡 创新点**

提出可持续且高效的轻量级 Transformer 与 LLM 微调方案，在资源受限环境下实现与大模型相当的翻译质量。

**🔧 技术方法**

采用规则型 Apertium、NLLB 以及 Llama、Tower 等 Encoder‑Decoder 与 Decoder‑Only 体系，结合 Marian、LoRA 微调、Fluores+ 数据集进行评测。

**📊 数据集**

主要数据来源为 Tatoeba Challenge（OPUS 语料）及 Flores+ 基准集，经过 OpusFilter 清洗后用于训练与测试。

**📈 对比分析**

通过 ChrF++、BLEU、COMET、MetricX 以及人工对比评估，发现 NLLB-200‑3.3B 在绝大多数方向上领先，轻量级 Transformer 仅比 LLM 微调差距极小。

**⚠️ 局限性**

限于仅使用 Flores+ 单一测试集、有限的 100 条人工评估样本以及未覆盖更广泛低资源语言，可能导致结果偏差与泛化能力受限。

---

## 247. CADEL: A Corpus of Administrative Web Documents for Japanese Entity Linking

**arXiv ID:** 2603.29336 | [PDF](https://arxiv.org/pdf/2603.29336v1)

**作者:** Shohei Higashiyama `[一作]` (National Institute of Information and Communications Technology), Masao Utiyama `[通讯]` (National Institute of Information and Communications Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了面向日语实体链接的标注语料库CADEL，并对其设计策略、标注流程、评测方法进行了系统阐述；

**💡 创新点**

创新点在于：①提出了针对日语实体链接的设计策略，细化了实体、提及与链接类型的定义；②构建了覆盖日本特定实体的高质量标注语料，尤其加入了“相关链接”分类；③基于难度分布制定了官方评测拆分，提升了评测对模型区分能力的挑战性；

**🔧 技术方法**

主要使用技术包括：GiNZA NLP工具进行初步实体识别与类型标注；Microsoft Access VBA自研标注工具完成手工标注与核心ference聚类；Wikidata SPARQL查询服务获取候选条目；评价指标采用Recall@k、LEA、CoNLL、F1与Cohen's κ；

**📊 数据集**

使用数据集：CADEL（160篇行政文件，8,082个提及，包含6,939个命名提及）作为主要资源；对比评测时亦涉及Jawikify、JWC、Shinra等现有日语实体链接数据集；

**📈 对比分析**

评测方法：基于字符串匹配的候选条目检索，三种排序策略（Uniform、SmallerID、MoreWikiBLs）并结合标签与别名匹配；性能表现：Exact链接Recall@1≈0.75，Recall@10≈0.83；相关链接Recall@1≈0.10，整体Recall@1≈0.69；官方拆分后，难度提升，模型差异更明显；

**⚠️ 局限性**

局限性：①语料来源单一（政府网站），导致文本倾向正式、易于标注，难度偏低；②只评估了简单字符串匹配与基于流行度的启发式，缺乏上下文深度的消歧方法；③相关链接处理仅选取单一条目，未充分覆盖多重候选；④标注仅覆盖部分文档的双人评审，未完全保证全体一致性；

---

## 248. Real-Time Band-Grouped Vocal Denoising Using Sigmoid-Driven Ideal Ratio Masking

**arXiv ID:** 2603.29326 | [PDF](https://arxiv.org/pdf/2603.29326v1)

**作者:** Daniel Williams `[一作]` `[通讯]` (Independent Researcher), Daniel Williams (Independent Researcher)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种低参数、低延迟的实时语音降噪模型，采用sigmoid驱动的理想比率掩码（IRM）并通过专门的频谱损失提升SNR与感知质量；

**💡 创新点**

创新点包括：①sigmoid驱动的掩码促使预测值趋近0/1以更好保留语音；②为实时SNR最大化量身定制的频谱损失函数；③在覆盖7种语言、700+噪声环境的巨大多语种多场景数据集上训练，实现在10 ms以下总延迟。

**🔧 技术方法**

技术实现：causal encoder‑decoder结构（约45万参数），band‑grouped Dense层+频率注意力SE机制、U‑Net风格跳连；使用IR M预测结合log‑magnitude、L1损失；输入为95 %峰值归一化幅度与相位角；

**📊 数据集**

使用的数据集包括：Saraga Carnatic Music Dataset、CommonVoice、Noisy Speech Database、GTSinger、SingingDatabase、VocalSet、Acapella Mandarin Singing Dataset，以及DNS噪声库，并在训练时随机混合SNR（5–35 dB）、音调/增益变化和高斯噪声。

**📈 对比分析**

通过与传统Spectral gating、spectral subtraction以及现代RNN/GRU模型的对比，使用PESQ-WB/NB、STOI、ESTOI指标评估；在静态噪声下PESQ-WB提升0.21，PESQ-NB提升0.25；在非静态噪声下PESQ-WB提升0.12；总延迟6.2 ms，模型参数仅约45万，满足实时通话与监测需求。

**⚠️ 局限性**

局限性：STOI略有下降，表明在极低延迟下语音细节和可懂度仍受影响；仅对幅度做掩码，未处理相位，导致对极端非stationary噪声的抑制有限；实验仅在CPU上完成，需进一步验证GPU/嵌入式部署；

---

## 249. VACP: Visual Analytics Context Protocol

**arXiv ID:** 2603.29322 | [PDF](https://arxiv.org/pdf/2603.29322v1)

**作者:** Tobias Stähle `[一作]` (ETH Zürich), Mennatallah El-Assady `[通讯]` (ETH Zürich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Visual Analytics Context Protocol（VACP），通过在视觉分析系统中嵌入结构化语义状态、交互能力和执行门控，让 AI 代理能够直接、可靠地读取、操作 VA 界面；

**💡 创新点**

创新点在于：①突破像素/DOM 层的接口不匹配，首次为 VA 系统提供 agent‑ready 的上下文协议；②将交互抽象为语义动作而非低级 UI 事件，显著提升代理推理效率与准确性；

**🔧 技术方法**

实现技术包括：基于 Model Context Protocol 的协议层、TypeScript 库、与 Vega‑Lite、Mosaic 等声明式可视化框架的适配器，以及 LLM 的工具调用接口；

**📊 数据集**

使用数据集：实验采用 7 名专家/非专家完成的 15 个交互任务，涉及 U.S. Flights、Global Development 等公开可视化数据集；

**📈 对比分析**

对比方法：在与基于视觉（CV）和 DOM 解析的代理方法的任务完成率、token 使用量和执行时间上进行实验。结果显示 VACP 代理完成率提升，token 消耗下降，执行时间缩短；

**⚠️ 局限性**

局限性：①需要开发者手工映射应用状态和交互，难以适应高度动态或自定义界面；②视觉细节信息被抽象为语义，某些感知任务可能受限；③在大模型推理时仍受延迟限制，需进一步优化异步交互设计。

---

## 250. Effective approach of the tridendriform Schroeder tree algebra

**arXiv ID:** 2603.29393 | [PDF](https://arxiv.org/pdf/2603.29393v1)

**作者:** Pierre Catoire `[一作]` (Université de Montpellier), Jean Fromentin `[通讯]` (Université du Littoral Côte d’Opale)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究并实现了Schroeder树（又称Schröder树）编码的自由三分式（tridendriform）代数结构，提出了基于树角度的编码、左/右梳形分解以及通过准洗牌（quasi-shuffle）实现原子乘法（AtomicProduct）的算法；

**💡 创新点**

创新点在于将树的内部结构映射为二维网格（grid）编码，并证明该编码与自由三分式代数同构，从而能够直接在编码上执行乘法、共乘和分剪操作；此外，还引入左优先打包词（Left Priority Packed Word）与树剪的双射，简化剪操作的枚举；

**🔧 技术方法**

使用的技术包括：C++实现的组合算法（如准洗牌枚举、树的左/右梳形分解、原子乘法）、集合与位图运算、递归与迭代的树/森林重构；

**📊 数据集**

论文未使用外部实际数据集，而是通过对Schroeder树进行符号化构造、枚举所有可能的准洗牌和剪切来验证算法正确性；

**📈 对比分析**

比较方法是用C++实现的算法与基于Python或伪代码的实现进行时间与内存消耗对比；实验表明C++实现相较于Python显著加速，能够在数千个树元之间完成乘法和共乘运算，性能提升约为10–20倍；

**⚠️ 局限性**

限制在于：算法的时间复杂度受准洗牌枚举和剪切数量影响，仍呈指数级增长；目前只适用于Schroeder树（包含所有Schroeder树的角度编码），对更一般的树或更高阶结构的扩展尚未讨论；

---

## 251. Learning Semantic Priorities for Autonomous Target Search

**arXiv ID:** 2603.29391 | [PDF](https://arxiv.org/pdf/2603.29391v1)

**作者:** Max Lodel `[一作]` (Delft University of Technology), Javier Alonso-Mora `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

在未知环境中学习专家指令的语义优先级，并将其嵌入前沿探索规划器，以实现高效的目标搜索。

**💡 创新点**

将专家交互数据转化为可训练的语义优先级模型，并在组合规划中使用该模型实现数据高效、对未知环境鲁棒的搜索策略。

**🔧 技术方法**

前沿探索 + 语义特征向量 + 加权线性优先级函数 + Bradley‑Terry 选择模型 + 加权最小延迟问题（WMLP）规划 + LNS 搜索 + A* 路径规划。

**📊 数据集**

ProcThor 2D 语义仿真环境，30 个训练场景（生成合成专家交互数据）和 34 个测试场景。

**📈 对比分析**

与纯覆盖探索基线以及多种 oracle 方法（Intervention、Priorities、Linear Oracle）对比。使用 PLR（路径长度比例）和 SPL（按路径长度加权的成功率）评估。结果表明：平均 PLR≈0.644，SPL≈0.627，且在大多数场景下 PLR<1，性能接近 oracle 级别。

**⚠️ 局限性**

依赖合成专家数据，真实人类交互缺失；在语义信息不足或不明显的场景下性能相对下降；仅在二维简化仿真环境验证，缺乏对复杂语义关系和真实机器人平台的评估。

---

## 252. Extend3D: Town-Scale 3D Generation

**arXiv ID:** 2603.29387 | [PDF](https://arxiv.org/pdf/2603.29387v1)

**作者:** Seungwoo Yoon `[一作]` (Seoul National University), Jaesik Park `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Extend3D，一种无训练的单图像到3D场景生成管线。

**💡 创新点**

创新点在于扩展latent空间并引入重叠patch流、先验初始化与3D感知优化，实现大规模、细节丰富的场景生成。

**🔧 技术方法**

使用扩展latent、重叠patch流、SDEdit（under‑noising）、3D-aware优化、MoGe‑2单目深度估计、DINOv2图像编码以及Trellis等技术。

**📊 数据集**

主要使用Google Earth图像进行生成，评估时采用UrbanScene3D数据集；未使用专门的大规模3D场景数据集。

**📈 对比分析**

与Trellis、Hunyuan3D、EvoScene和SynCity等方法对比，通过人类偏好、LPIPS/SSIM/PSNR、Chamfer/F-score以及CLIP/HPSv3/Intra‑LPIPS等指标，表现均优于对手。

**⚠️ 局限性**

局限性包括盲区补全不完整、优化过程内存占用高，以及对街景图像的生成效果有限。

---

## 253. PromptForge-350k: A Large-Scale Dataset and Contrastive Framework for Prompt-Based AI Image Forgery Localization

**arXiv ID:** 2603.29386 | [PDF](https://arxiv.org/pdf/2603.29386v1)

**作者:** Jianpeng Wang `[一作]`, Zhongjie Ba `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了PromptForge-350k大型伪造定位数据集，并提出了一种全自动掩膜标注框架；同时设计了ICL-Net网络，用三流骨干网和图像内对比学习实现对提示式AI图像编辑伪造的精确定位。

**💡 创新点**

创新点包括：①基于像素级配准与语义空间相似度的自动掩膜生成方法，消除了手工标注；②三流骨干网络结合噪声、语义和冻结特征，利用图像内对比学习提升伪造特征的区分度；③针对提示式编辑缺失传统痕迹的场景，提出新的定位策略。

**🔧 技术方法**

采用了像素级配准（关键点匹配+RANSAC+仿射变换）、DINO v3语义特征相似度、三流SegFormer-B4骨干、对比损失（基于特征对比）以及Focal+Dice混合分割损失。

**📊 数据集**

使用了PromptForge-350k数据集（354k对图像，涵盖Nano-Banana、BAGEL、Flux.Kontext、Step1x等四款提示式编辑模型的8类编辑任务），并在此数据集上训练与评估。

**📈 对比分析**

与MVSS‑Net、TruFor、FOCAL、Mesorch、NFA‑ViT等五种SOTA方法对比，ICL-Net在F1、IoU、精确率、召回率均优于对手；在PromptForge-350k上IoU达62.5%（比SOTA高5.1%），在JPEG压缩和裁剪扰动下表现极其稳健，未见显著性能下降。

**⚠️ 局限性**

局限性：自动标注在仅改变颜色等场景下易产生误标；对未见闭源编辑模型的泛化能力有限，特别是Nano-Banana的性能显著下降。

---

## 254. Causality-inspired Federated Learning for Dynamic Spatio-Temporal Graphs

**arXiv ID:** 2603.29384 | [PDF](https://arxiv.org/pdf/2603.29384v1)

**作者:** Yuxuan Liu `[一作]` (University of Electronic Science and Technology of China), Boyuan Zhang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种面向异构动态时空图的联邦学习框架 SC‑FSGL，能够在保持数据隐私的同时通过因果分离提升模型泛化性能。

**💡 创新点**

创新点在于：①使用条件分离模块（Conditional Separation Module）在表示层模拟软干预，显式区分共享因果特征与客户端特定噪声；②引入因果代码簿（Causal Codebook）与对比学习对齐共享因果表示；③结合 IRM 正则化增强跨客户端的表示不变性，形成全新的因果驱动联邦学习范式。

**🔧 技术方法**

技术手段包括：图神经网络（使用 node2vec、GMAN 结构），软干预掩码、层归一化、对比损失、IRM 正则、软阈值和梯度惩罚；在联邦训练中采用 FedAvg 的通信策略。

**📊 数据集**

使用五个真实交通时空图数据集：METRLA、PEMSD4、PEMSD7(M)、PEMSD8、PEMSBAY，每个数据集对应一个客户端，以模拟强烈的空间和时间异质性。

**📈 对比分析**

与 FedAvg、FedProx、Moon、FedRep、FedStar、GMAN、MegaCRN、FUELS 以及本地训练进行对比。SC‑FSGL 在所有 5 个客户端和三种预测时长（60、30、15 分钟）下的 MAE/MAPE 均优于所有基线，显著提升了 10–30% 的精度。

**⚠️ 局限性**

局限性包括：①仅在交通数据上验证，缺乏跨领域通用性验证；②对极端异质场景（如节点/边数差异过大）或极少数据的客户端效果尚待进一步评估；③因果分离模块的掩码学习依赖于先验设定，可能在高噪声环境下失效。

---

## 255. How and Why Agents Can Identify Bug-Introducing Commits

**arXiv ID:** 2603.29378 | [PDF](https://arxiv.org/pdf/2603.29378v1)

**作者:** Niklas Risse `[一作]` (Max Planck Institute for Software Systems), Marcel Böhme `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估两种基于大型语言模型代理的工具（SZZ‑Agent 和 Simple‑SZZ‑Agent），用来从修复提交中识别错误引入提交。

**💡 创新点**

通过让代理执行二分搜索或直接候选搜索，并证明代理能从修复提交中提取短的 Grep 可搜索模式，显著提升 F1 分数，甚至不使用二分搜索即可超越 SZZ，突破多年递增瓶颈。

**🔧 技术方法**

使用 LLM 代理（Claude Code、OpenHands）、工具调用（文件读取、grep、bash）、二分搜索、候选集合构建与模式提取等技术。

**📊 数据集**

采用 Linux kernel 开发者标注的数据集 DS_LINUX、DS_LINUX‑26，以及 GitHub C/C++ 和 Java 的 DS_GITHUB‑c、DS_GITHUB‑j。

**📈 对比分析**

与 8 种传统 SZZ 变体和最新 LLM4SZZ 进行宏平均 F1 对比；SZZ‑Agent 在三大数据集上 F1 分别提升 12–13%，Simple‑SZZ‑Agent 更进一步提升至 0.81–0.86 F1，并且成本更低。

**⚠️ 局限性**

限制：候选集仅覆盖修复文件，导致约 4% 的错误未被考虑；代理推理仍有误判与近似错误；对多文件交互错误建模不足；仅在 C/C++ 与 Java 上评估，缺乏跨语言验证。

---

## 256. StereoVGGT: A Training-Free Visual Geometry Transformer for Stereo Vision

**arXiv ID:** 2603.29368 | [PDF](https://arxiv.org/pdf/2603.29368v1)

**作者:** Ziyang Chen `[一作]` (Xiamen University), Liujuan Cao `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了训练‑free 的 StereoVGGT 主干，通过熵最小化权重融合 (EMWM) 把 VGGT 与 MDE/VFM 的权重结合，并通过双分支特征颈部（帧注意力+MDE Neck）融合相机几何先验与细粒度特征，直接用于立体匹配与立体转换。

**💡 创新点**

创新点包括① 引入熵最小化权重融合技术，实现三种模型（VGGT、MDE、VFM）几何知识与特征细节的自动平衡；② 在特征颈部采用帧注意力对齐机制，将相机几何先验与细节特征融合；③ 完全无训练即可在多任务中实现 SOTA 性能。

**🔧 技术方法**

采用 VGGT 体系结构、DINOv2/Moge‑2/DepthAnything V2 权重、熵最小化权重融合、双分支特征颈部（Frame Attention + MDE Neck）、Dense Prediction Transformer 头、I‑Gev‑Stereo/Mono2Stereo 解码器，并通过无监督超参数迭代更新。

**📊 数据集**

使用 KITTI、Scene Flow、ETH3D、Middlebury、Mono2Stereo、Inria 3D Movie 等公开数据集进行评估。

**📈 对比分析**

与多种 SOTA 基线（I‑Gev‑Stereo、Mono2Stereo、PromptStereo、Prompt‑Stereo、AIO‑Stereo 等）在 KITTI 非遮挡像素上排名第一；在 Scene Flow 上 EPE 最佳；在 Mono2Stereo 与 Inria 3D Movie 立体转换任务中取得 SOTA，整体性能显著优于 DINOv2、DAv2、Moge‑2 等。

**⚠️ 局限性**

模型参数量大，部署成本高，且依赖冻结的 VGGT 预训练权重，可能在极端场景下泛化受限。

---

## 257. Nomad: Autonomous Exploration and Discovery

**arXiv ID:** 2603.29353 | [PDF](https://arxiv.org/pdf/2603.29353v1)

**作者:** Bokang Jia `[一作]`, Andrew Jackson `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了 NOMAD，一种以探索为先的自主数据探索与洞察发现系统；该系统通过构建 Exploration Map、系统化地进行主题选择、假设生成、探索-验证循环，以及最终报告与元报告生成，实现了跨文档、多源数据的自动洞察挖掘。

**💡 创新点**

创新点在于：①将探索过程抽象为 Exploration Map，并通过宽度优先遍历实现多样性与覆盖的平衡；②引入独立的验证者（verifier），将推理与验证解耦，显著降低幻觉风险；③将洞察验证、报告生成与元报告合成集成为一个完整管道，并提出了面向可信度、多样性、报告质量与冗余的全新评估框架。

**🔧 技术方法**

技术包括：LLM 基础的探索代理（ReAct 循环）与子代理（Web、文档、SQL 搜索），多层次主题与概念抽取与聚类构建 Exploration Map，假设生成与分层评分，独立验证者的子声明验证，基于引用数据库的报告生成与审计，以及元报告的主题聚类与三段式合成。

**📊 数据集**

主要使用数据集为联合国与世界卫生组织的官方报告集合（UN & WHO policy 与统计报告），包含健康、气候、发展、贸易等多领域共数百份 PDF；在实验中亦使用了企业内部机密文档集合，但未公开。

**📈 对比分析**

与 Deep Research、GPT Researcher 等基线进行对比，NOMAD 在 Trustworthiness（可信度）、Report Quality（报告质量）和 Diversity（多样性）方面均显著优于基线；实验中通过多轮运行，NOMAD 产生的标题与洞察更为多样，且经过验证者过滤后误报率明显降低，报告引用准确率提升。

**⚠️ 局限性**

局限性包括：①系统对 LLM 的依赖导致仍可能出现幻觉，尤其在复杂推理时；②探索图一旦构建完成后保持静态，缺乏实时动态更新，难以捕捉快速变化领域的最新信息；③计算成本较高，尤其在大规模文档集上构建概念层与主题树需大量推理；④验证者与报告生成对可解释性和可调优性要求高，仍需进一步完善工具调用与异常处理。

---

## 258. IMPASTO: Integrating Model-Based Planning with Learned Dynamics Models for Robotic Oil Painting Reproduction

**arXiv ID:** 2603.29315 | [PDF](https://arxiv.org/pdf/2603.29315v1)

**作者:** Yingke Wang `[一作]` (Stanford University), Ruohan Zhang `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一套基于深度像素动力学模型与模型预测控制（MPC）的油画机器人系统，能够从一系列目标油画图像中学习并执行相应的笔触轨迹、力度与颜色，实现对人类艺术家作品的近似复刻。

**💡 创新点**

创新点包括：①使用自我游戏收集的数据训练可预测笔触效果的像素级前向动力学模型；②将该模型与递归MPC相结合，实现对多步笔触的闭环规划与执行；③通过低级力敏感控制与高级规划协同，使机器人在无人工演示或仿真器的情况下完成真实油画复刻。

**🔧 技术方法**

技术手段包括：U‑Net结构的像素动力学网络、Bézier曲线参数化笔触、基于MPPI的MPC优化、Franka Emika Panda 7‑DoF机械臂+六轴力传感器、贴合力控制算法和色彩预测模块。

**📊 数据集**

数据集为机器人自玩生成的自监督数据（约900条样本），以及从五位专业油画艺术家处收集的12笔触单独样本和机器人随机叠加的50笔触样本，用于评估模型的泛化与复刻效果。

**📈 对比分析**

与线性回归、纯启发式、FRIDA‑CNN等基线进行对比。实验结果表明，在规划阶段和执行阶段，U‑Net模型的加权ℓ1误差分别降低约16–24%，LPIPS误差提升约15–35%，显示出更优的视觉相似度和更精确的笔触复刻。

**⚠️ 局限性**

局限性包括：①模型对极复杂笔触（如尖锐转折、特殊纹理）仍缺乏表达；②依赖大量自玩数据，数据收集成本高；③在高噪声背景或不规则画布上预测误差增大；④实时性受限，规划周期约需数秒，难以适应更快速的艺术创作。

---

## 259. RAAP: Retrieval-Augmented Affordance Prediction with Cross-Image Action Alignment

**arXiv ID:** 2603.29419 | [PDF](https://arxiv.org/pdf/2603.29419v1)

**作者:** Qiyuan Zhuang `[一作]` (Southeast University), Xiu-Shen Wei `[通讯]` (Southeast University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种融合检索与训练的任务感知动作空间预测框架RAAP，用于在新物体或新类别上进行精细操作

**💡 创新点**

通过将静态接触点与动态动作方向分离并分别采用密集特征匹配和检索增强对齐，使用双重加权注意力聚合多参考信息，显著提升动态动作预测的鲁棒性

**🔧 技术方法**

使用CLIP文本/图像编码、Stable Diffusion特征、SigLIP-2 backbone、FiLM调制、Transformer、跨图像注意力及双重加权机制，并将2D动作上采样到3D执行

**📊 数据集**

在DROID、HOI4D两个视觉交互数据集上构建检索记忆，并在Frank的RealSense摄像头数据与MuJoCo仿真环境中进行评估

**📈 对比分析**

与基线RAM（单参考转移）和A0（大规模扩散模型）对比，RAAP在MAE上从62.84°下降至32.55°（约50%提升），在真实世界与仿真中实现高成功率（最高100%）

**⚠️ 局限性**

仅适用于短时单物体任务，开放式控制下性能受限，需扩展到多物体或长期序列以及闭环控制

---

## 260. PRISM: PRIor from corpus Statistics for topic Modeling

**arXiv ID:** 2603.29406 | [PDF](https://arxiv.org/pdf/2603.29406v1)

**作者:** Tal Ishon `[一作]` (Bar Ilan University), Uri Shaham `[通讯]` (Bar Ilan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 PRISM 方法，通过利用语料内部共现统计推导 Dirichlet 参数，对 LDA 进行语料驱动初始化；

**💡 创新点**

创新点在于无需外部知识或预训练嵌入，直接用词共现图+扩散映射生成词向量，再通过软聚类估计 β，提升主题模型的语义连贯性与可解释性；

**🔧 技术方法**

使用 PPMI 词共现图、余弦相似度、扩散映射（Diffusion Maps）、高斯混合模型（GMM）软聚类、矩估计法求 Dirichlet 参数，以及 MALLET 的 Collapsed Gibbs 采样实现 LDA；

**📊 数据集**

在五个文本语料（20NewsGroup、BBC News、M10、DBLP、TrumpTweets）和三份单细胞 RNA‑seq 数据集（BreastCancer、PBMC3k、Zeisel brain）上进行实验；

**📈 对比分析**

与传统 LDA（MALLET）、多种文本主题模型（ProdLDA、NeuralLDA、NMF、ETM、BERTopic、C-Top2Vec、FASTopic）以及单细胞专用因子模型（scHPF、cNMF）比较，PRISM 在 c_v 与 NPMI 等主题连贯性指标上明显优于 MALLET，且常与或超越基于外部嵌入的先进方法；在单细胞评估中，PRISM 在生物学连贯性与通路富集上也优于 MALLET；

**⚠️ 局限性**

局限包括需预先设定主题数 K，且仅改进 β，未针对文档主题先验 α；对极小语料或高维词表的扩展仍需进一步研究。

---

## 261. Security in LLM-as-a-Judge: A Comprehensive SoK

**arXiv ID:** 2603.29403 | [PDF](https://arxiv.org/pdf/2603.29403v1)

**作者:** Aiman Almasoud `[一作]` (University of Pavia), Saraga Sakthidharan `[通讯]` (University of Pavia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统梳理LLM-as-a-Judge（LaaJ）系统的安全风险与威胁场景，构建攻击、攻击手段、防御与评估的完整框架。

**💡 创新点**

首次提出基于角色的五类安全taxonomy（攻击目标、攻击手段、攻击工具、防御工具与评估应用），并对45篇近六年内相关研究进行系统性分类与对比。

**🔧 技术方法**

采用文献综述、案例分析、威胁建模、对抗实验与多模型集成技术，对LaaJ在Prompt注入、Token扰动、后门注入、数据污染等攻击方式进行评测；同时使用LLM推理、In‑Context Learning、检索增强生成等技术进行防御与评估。

**📊 数据集**

共分析863篇文献，筛选45篇2010‑2026年的会议/期刊论文与arXiv预印本；使用公开基准（SummEval、MT‑Bench、OpenAI Safety‑Guard、RAG‑Bench等）以及自建安全评测集（Prompt‑Injection、Backdoor、Data‑Contamination等）进行实验。

**📈 对比分析**

通过对比攻击成功率（ASR）、鲁棒性分数、误报率与准确率等指标，发现大多数LaaJ模型的鲁棒性低于70%，单一防御效果有限，但多模型集成可将ASR降低至10–20%；与传统人工评测相比，LaaJ在效率上提升数十倍，但在可靠性上仍显不足。

**⚠️ 局限性**

局限包括：样本与场景覆盖不足、评测与真实部署环境差异大、缺乏统一的安全评测标准、对商业内部系统的实验受限，且大部分研究基于公开模型与数据，未能全面评估闭源系统的安全性。

---

## 262. ELT-Bench-Verified: Benchmark Quality Issues Underestimate AI Agent Capabilities

**arXiv ID:** 2603.29399 | [PDF](https://arxiv.org/pdf/2603.29399v1)

**作者:** Christopher Zanoli `[一作]` (IBM Research), Yotam Perlitz `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对ELT-Bench基准进行了系统审计与修订，重新评估了AI代理在端到端ELT管道构建中的性能，发现原始评估低估了代理能力。

**💡 创新点**

创新点在于提出并实现了可扩展的Auditor‑Corrector框架，结合LLM根因分析与人工验证，定位并纠正了大量基准质量问题，推出了修订版ELT-Bench‑Verified。

**🔧 技术方法**

主要技术包括大语言模型（Claude Sonnet 4.5、Claude Opus 4.5）驱动的自动分析、人工标注验证、评估脚本修正、地面真值列剔除等。

**📊 数据集**

使用的数据集为ELT‑Bench的100个任务（含多源数据库、API、文件等），共203个目标数据模型，审计涉及660个列级不匹配。

**📈 对比分析**

通过对比原始评估与升级模型后评估，以及对照修订版基准，发现SRDT从1%→22.66%→32.51%，SRDEL从37%→96%；修正后通过多代理验证验证提升显著。

**⚠️ 局限性**

限制包括审计仅覆盖单一代理配置的失败任务、对所有660列的分类主要由单人完成、以及对标注边界的主观判断可能导致误归类。

---

## 263. Interacting Multiple Model Proprioceptive Odometry for Legged Robots

**arXiv ID:** 2603.29383 | [PDF](https://arxiv.org/pdf/2603.29383v1)

**作者:** Wanlei Li `[一作]` (Harbin Institute of Technology Shenzhen), Yunjiang Lou `[通讯]` (Harbin Institute of Technology Shenzhen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于交互式多模型（IMM）的腿部机器人自我位置估计框架，能够在滚动与滑移不同接触条件下实现在线模式推理与融合；

**💡 创新点**

创新点在于将滚动约束显式加入状态增广并通过对足部速度的过程噪声调节实现滑移感知，进而通过IMM动态调整模式可信度；

**🔧 技术方法**

主要技术包括滚动感知的误差状态扩展卡尔曼滤波（ESKF）、多模型IMM推理、滚动约束测量更新以及滑移的过程噪声放大；

**📊 数据集**

使用了仿真（Gazebo）中的四足AlienGo机器人在平坦、斜坡、斜滑、崎岖等多种地形下的数据，以及真实世界的室内平地和复杂地形的运动捕捉数据；

**📈 对比分析**

与多种基线（KF、ESKF、IEKF、ESKF‑VB、KalmanNet、单模型滚动ESKF）比较，实验显示IMM‑PO在绝对轨迹误差（ATE）与相对姿态误差（RPE）上均比基线至少提升15–70%，且在滑移与复杂地形下鲁棒性最佳；

**⚠️ 局限性**

局限在于仅考虑两种接触模式（滚动与滑移），对极端滑移或多种不确定接触的细粒度区分仍不足，且在极长时序下仍可能出现累计漂移。

---

## 264. Deep Learning-Based Anomaly Detection in Spacecraft Telemetry on Edge Devices

**arXiv ID:** 2603.29375 | [PDF](https://arxiv.org/pdf/2603.29375v1)

**作者:** Christopher Goetze `[一作]` (IU International University of Applied Sciences), Daniel Lakey `[通讯]` (IU International University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在空间飞行器边缘设备上实现并优化了三种基于深度学习的遥测异常检测方法（预测+阈值、直接分类、图像分类）。

**💡 创新点**

创新点在于将多目标神经架构搜索与Pareto前沿优化相结合，在保持高检测性能的同时显著降低RAM、ROM和MACs，实现可在CubeSat级硬件上部署。

**🔧 技术方法**

采用XceptionTimePlus、ResNet34、GAF变换、Optuna多目标优化、MLTK模型分析器等技术。

**📊 数据集**

使用欧洲航天局ESA-ADB卫星遥测异常数据集（Mission 1轻量级子集）。

**📈 对比分析**

通过与基线模型比较，预测+阈值在未优化时达92.7% CEF_0.5，优化后仍保持88.8%并将RAM从2 KB降至59 B；直接分类CEF_0.5从72.4%降至48.5%；图像分类从48.6%提升至70.1%。

**⚠️ 局限性**

局限在于直接分类在优化后召回率显著下降，图像分类仍需较高计算资源，且缺乏对能耗和推理延迟等更多硬件指标的考量。

---

## 265. Exploration of Energy and Throughput Tradeoffs for Dataflow Networks

**arXiv ID:** 2603.29367 | [PDF](https://arxiv.org/pdf/2603.29367v1)

**作者:** Abrarul Karim `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Jürgen Teich `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在自供电（self‑powered）数据流网络中，如何在保证吞吐量的前提下实现能量节约，提出了线性规划（LP）求最大吞吐量的周期调度、混合整数线性规划（MILP）在给定吞吐量下求最小能耗的调度以及一种高效的 Hop & Skip DSE 策略来探索吞吐量-能耗 Pareto 前沿。

**💡 创新点**

创新点包括：① 证明在每个 actor 仅处于永不睡眠或自供电两种模式下的网络仍能产生周期调度；② 提出 LP 和 MILP 两种优化模型，分别求解最大吞吐量与最小能耗；③ 设计 Hop & Skip 迭代策略，利用“跳过”技术大幅减少搜索空间，同时仍能获得完整 Pareto 前沿；④ 在真实 AEC 网络和多种基准图上验证方法有效性，并与穷举搜索/周期搜索对比。

**🔧 技术方法**

核心技术：数据流图（Markov 图）建模；最大环平均值理论求周期下界；线性规划与混合整数线性规划求解器（如 CPLEX/SCIP）；自供电模式下的功耗/延迟模型；Hop & Skip 迭代 DSE；Pareto 前沿与超体积（hypervolume）评价。

**📊 数据集**

使用的数据集包括：一套真实的 AEC（Acoustic Echo Cancellation）网络；SDF^3 Benchmark Suite（H.263 编码器、MP3 播放/解码、卫星、Samplerate 等 10 个网络）；以及 100 个随机生成的 SDF 图（平均 15 个 actor）。所有图均转化为标记图（marked graph）后进行实验。

**📈 对比分析**

与传统的决策变量穷举搜索（Sweep）和周期搜索（P Sweep）对比，Hop & Skip 在 100 个随机图上平均实现 29.95× 的搜索时间加速（对 P Sweep）和 342.39×（对 Sweep），最大可达 2300×；在 SDF^3 基准上，Hop & Skip 与 P Sweep 一致获得完整 Pareto 前沿；在所有实验中，Hypervolume 比例接近 1，说明搜索结果与真值前沿基本一致。

**⚠️ 局限性**

局限性：① 当使用较大周期步长 ε 时，可能错过某些具有合理 rational 周期的 Pareto 点；② 方法假设每个 actor 仅处于两种固定模式，无法捕捉更细粒度的功耗动态；③ 对于极小规模（≤6 个 actor）网络，穷举搜索在时间上并不劣势；④ 需要高质量的功耗与延迟模型，若模型不准确会影响结果。

---

## 266. AI-Generated Prior Authorization Letters: Strong Clinical Content, Weak Administrative Scaffolding

**arXiv ID:** 2603.29366 | [PDF](https://arxiv.org/pdf/2603.29366v1)

**作者:** Moiz Sadiq Awan `[一作]` (Independent Researcher), Maryam Raza `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对三种商业大型语言模型（GPT‑4o、Claude Sonnet 4.5、Gemini 2.5 Pro）在45个医师验证的多学科先决授权（PA）信模板中的生成效果进行了系统评估。

**💡 创新点**

创新点在于：①首次跨五个医学专科、三种模型使用统一评分框架进行结构化比较；②提出了二次特征分析，揭示模型在行政细节（计费码、授权时长等）上的共性缺口；③在严格控制输入的前提下，首次报告PA信生成无临床幻觉。

**🔧 技术方法**

技术方法包括：标准化的用户提示、温度为0的单次生成、基于文本匹配的自动评分加人工校对、以及对八项行政要素的二次统计。

**📊 数据集**

数据集为45个经过临床医生验证的合成案例，覆盖风湿、精神、肿瘤、心脏和骨科共五个专科，每个专科9例。

**📈 对比分析**

比较方法为基于12分制的六项评分标准进行统计学检验，结果显示所有模型平均得分>97%；Claude Sonnet 4.5最高（11.98/12），在拒绝预期（step therapy）项上显著优于GPT‑4o（p<0.001，Cohen's d≈0.76）。

**⚠️ 局限性**

局限性包括：①仅使用合成案例，缺乏真实病历的复杂性；②未评估实际授权通过率；③仅针对中价位模型，未涉及旗舰版或开源模型；④单次生成不展示温度变化带来的多样性；⑤未在Prompt中加入PA挑战字段，可能低估模型对拒绝预期的真实表现。

---

## 267. LongCat-AudioDiT: High-Fidelity Diffusion Text-to-Speech in the Waveform Latent Space

**arXiv ID:** 2603.29339 | [PDF](https://arxiv.org/pdf/2603.29339v1)

**作者:** Detai Xin `[一作]` (Meituan), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于波形潜在空间的端到端非自回归扩散式TTS模型，能够实现零样本声克隆并生成高质量语音。

**💡 创新点**

核心创新在于直接在Wav‑VAE潜在空间训练扩散模型，消除中间表示误差；并在推理阶段纠正训练‑推理不匹配，采用自适应投影引导（APG）提升生成质量。

**🔧 技术方法**

使用技术包括Wave‑VAE连续潜在编码、条件流匹配（CFM）扩散Transformer、长跳连接、全局AdaLN、UMT5多语言文本编码、APG推理校正以及训练‑推理一致性校正。

**📊 数据集**

训练与评估数据集为：约200K小时中英语料用于Wave‑VAE训练，100K小时中英语料用于TTS训练（1M小时用于大规模扩展），评测采用LibriTTS子集和Seed benchmark。

**📈 对比分析**

与Seed‑TTS、Seed‑DiT、F5‑TTS等基准对比，模型在Seed‑ZH/Seed‑Hard上SIM分别提升至0.818/0.797，整体UTMOS/DNSMOS与前沿模型相当，尤其在声克隆相似度上领先。

**⚠️ 局限性**

局限性包括：高维潜在表示会增加扩散模型的建模负担，需要在维度与帧率之间权衡；推理速度受扩散步数限制；对极端噪声或长句子性能的鲁棒性尚未充分验证。

---

## 268. CLaD: Planning with Grounded Foresight via Cross-Modal Latent Dynamics

**arXiv ID:** 2603.29409 | [PDF](https://arxiv.org/pdf/2603.29409v1)

**作者:** Andrew Jeong `[一作]` (KAIST), Sung-Eui Yoon `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出CLaD框架，利用跨模态潜在动力学模型和扩散策略实现机器人长期规划。

**💡 创新点**

创新点在于异步跨注意力捕捉关节-语义耦合，并通过自监督预训练的潜在预知实现参数高效的跨模态规划。

**🔧 技术方法**

采用异构跨注意力、EMA目标编码器、自监督潜在预测、辅助重构损失以及以潜在预知为条件的扩散策略。

**📊 数据集**

在LIBERO-LONG基准及其短期子集（LIBERO-Spatial、Object、Goal）上进行评估。

**📈 对比分析**

与OpenVLA、π_0.5等大型视觉语言模型对比，CLaD仅使用0.66B参数在LIBERO-LONG上实现94.7%成功率，接近更大模型。

**⚠️ 局限性**

在短期任务和泛化子集上表现不及大规模预训练的VLA模型，说明缺乏广泛背景知识。

---

## 269. Hybrid Quantum-Classical Spatiotemporal Forecasting for 3D Cloud Fields

**arXiv ID:** 2603.29407 | [PDF](https://arxiv.org/pdf/2603.29407v1)

**作者:** Fu Wang `[一作]` (CMA Earth system Modeling and Prediction Center), Xiaowen Chu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 QENO（Quantum-Enhanced Neural Operators），一种结合经典卷积编码器-解码器和拓扑感知量子增强块的混合量子-经典框架，用于三维云场的时空预测。

**💡 创新点**

创新点在于：①引入拓扑感知的量子耦合电路，用有限量子比特捕捉隐藏在云层中的非局部相互作用；②设计动态融合时空单元（DFTU），将量子测量结果作为门控信息融入经典 LSTM，实现在时空尺度上的双向信息流；③通过极低参数量（0.03M）实现与传统深度模型相当甚至更优的预测性能。

**🔧 技术方法**

使用技术包括：经典卷积跳跃连接+多尺度 Inception 编码器、角度/幅度量子编码、可学习拓扑结构的量子电路（含单量子旋转与拓扑相干 CNOT 组合）、量子测量回馈的门控 LSTM、经典解码器以及多种评估指标（MSE/MAE/RMSE/SSIM/CSI/HSS/POD）。

**📊 数据集**

数据集为 CMA‑MESO 运营三维云场，分辨率 3 km × 3 km × 42 层，时间步长 3 h，提供 64×64 网格的云掩码序列，用于训练与测试。

**📈 对比分析**

通过与 ConvLSTM、PredRNN++、Earthformer、TAU、SimVP、SimVP+ 等八种主流基线在 MSE/MAE/RMSE/SSIM 以及阈值下的 CSI/HSS/POD 进行对比；QENO 在所有指标上均优于基线，尤其在 MSE（0.2038）与 SSIM（0.6291）上显著领先，同时保持极低参数预算。

**⚠️ 局限性**

局限性：①量子模块仅在模拟器（torchquantum）中实现，尚未在真实量子硬件上验证；②电路深度与量子比特数量受限，难以进一步扩展高维输入；③整体计算开销受量子模拟器瓶颈影响，实际部署仍需优化；④目前仅针对云掩码预测，未验证对其他气象变量或更大规模域的适用性。

---

## 270. AA-Splat: Anti-Aliased Feed-forward Gaussian Splatting

**arXiv ID:** 2603.29394 | [PDF](https://arxiv.org/pdf/2603.29394v1)

**作者:** Taewoo Suh `[一作]` (Korea Advanced Institute of Science and Technology), Munchurl Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AA‑Splat，一种前向 3D 高斯投影模型，能够在任意采样率下无锯齿渲染 3D 场景。

**💡 创新点**

创新点在于 OBBL（Opacity‑Balanced Band‑Limiting）设计，结合 3D‑BLPF 与 Opacity‑Balancing，有效抑制失真与高频噪声，实现跨采样率的抗锯齿。

**🔧 技术方法**

采用基于 Nyquist 频率约束的 3D‑BLPF、像素级透明度平衡（OB）、2D Mip 滤波器、深度估计网络以及光度自监督的训练损失。

**📊 数据集**

在 RealEstate10K 训练，使用 RE10K、DL3DV、ACID 三个数据集评估（零样本迁移），并在多分辨率下进行测试。

**📈 对比分析**

与 DepthSplat、MVSplat、NoPoSplat、SPFSplat 等现有 FF‑3DGS 方法对比，AA‑Splat 在 1/4×–4× 采样率下平均提升 5.4–7.5 dB PSNR，且在跨域零样本场景中同样表现优异。

**⚠️ 局限性**

局限性包括仍需 2D Mip 滤波器来抑制高频细节，极端缩放（如 1/4×）下偶尔出现轻微亮度失真；此外目前仅适用于静态场景，动态物体处理尚未探索。

---

## 271. FOSCU: Feasibility of Synthetic MRI Generation via Duo-Diffusion Models for Enhancement of 3D U-Nets in Hepatic Segmentation

**arXiv ID:** 2603.29343 | [PDF](https://arxiv.org/pdf/2603.29343v1)

**作者:** Youngung Han `[一作]` (Seoul National University), Hyuk-Jae Lee `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了FOSCU框架，利用双扩散模型Duo‑Diffusion生成高分辨率的合成MRI体积及对应的分割标签，并用于增强3D U‑Net肝脏分割模型的训练。

**💡 创新点**

创新点在于将3D潜在扩散模型与ControlNet相结合，先生成解剖一致的分割掩码再条件生成对应MRI，形成统一的双生成管道，显著提升数据多样性与图像真实性。

**🔧 技术方法**

采用3D潜在扩散模型、ControlNet、3D U‑Net及其变体（ResUNet、WideResUNet、DynUNet、VNet），并使用Dice损失和Fréchet Inception Distance评估。

**📊 数据集**

使用了三星医学院720例门静脉相位腹部MRI扫描，裁剪得到(160,160,64)体积，并手工标注肝脏、门静脉、肝静脉及肿瘤四类。

**📈 对比分析**

将合成数据与真实数据联合训练的3D U‑Net与仅用真实数据训练的模型对比，实验表明Dice系数平均提升0.67%（肝脏单类）或0.66%（多类），Fréchet Inception Distance从21.67降至28.31，显示图像质量与分割准确性显著提升。

**⚠️ 局限性**

主要局限在于数据量仍有限，部分网络仅提升有限，且仅在单中心数据上验证，需在更大、多机构数据集上进一步评估。

---

## 272. PSPA-Bench: A Personalized Benchmark for Smartphone GUI Agent

**arXiv ID:** 2603.29318 | [PDF](https://arxiv.org/pdf/2603.29318v1)

**作者:** Hongyi Nie `[一作]` (Northwestern Polytechnical University), Zhen Wang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套用于评估智能手机 GUI 代理个性化能力的 benchmark（PSPA-Bench），并设计了任务分解图（TDG）与细粒度评价指标；

**💡 创新点**

创新点在于：①通过 TDG 明确区分通用与个性化步骤，实现无大规模用户日志的个性化指令生成；②基于单步指令的对齐，提供即时与长期两层细粒度评估；③构建覆盖 10 个场景、22 个 App、100 人物角色的 12,855 条个性化指令数据集；

**🔧 技术方法**

技术包括：任务分解图（TDG）结构化表示、基于 LLM 的动作对齐与路径匹配、四维细粒度指标（APR、PPR、CT、CPT 及其增量版）以及多种 LLM/代理框架的对比；

**📊 数据集**

使用了由 22 款主流 Android App（如购物、旅行、社交等）构成的 10 个日常场景，并根据 100 个合成用户画像生成 12,855 条个性化指令；

**📈 对比分析**

对比了 11 种现有 GUI 代理（包括 LLM+可访问性树、ReAct、M3A、Mobile-Agent 系列等），结果显示即使是最佳模型（Mobile-Agent E）在个性化任务中的 APR 仅约 0.70，且执行成本高；长期适应性方面，具备持久记忆与自演化机制的模型（Mobile-Agent E、Reflexion、Mobile‑Agent V2）表现最优；

**⚠️ 局限性**

局限性包括：①覆盖的 App 与场景有限，难以推广到医疗、金融等专业领域；②用户画像与偏好为合成而非真实数据，可能缺乏复杂性与动态性；③仅针对 Android，iOS 等平台适用性未知；④TDG 由人工专家构造，缺乏自动化生成能力与边缘情况覆盖。

---

## 273. Is my model perplexed for the right reason? Contrasting LLMs' Benchmark Behavior with Token-Level Perplexity

**arXiv ID:** 2603.29396 | [PDF](https://arxiv.org/pdf/2603.29396v1)

**作者:** Zoë Prins `[一作]` (University of Amsterdam), Sandro Pezzelle `[通讯]` (University of Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于 token 层次 perplexity 的解释框架，用最小对比句对检测 LLM 在关键词上是否真正依据语言学线索做出决定。

**💡 创新点**

创新点在于：①用 token 层次 perplexity 直接衡量模型对关键词的关注；②设计了两种指标（plain proportion 与 normalized proportion）量化关键词对整体 perplexity 差异的解释力度；③避免了传统特征归因方法的易变性，提供了可复现、可假设检验的解释手段。

**🔧 技术方法**

技术手段包括：token 层次 perplexity 计算、最小对照句对构造、两种解释比例指标、对 8 种提示变体与顺序的平均准确率评估。

**📊 数据集**

使用的数据集包括：自构造的 nonsense words sanity check、CrowS‑pairs（刻板印象）、BLiMP（代词一致性与生物/非生物辨别）以及 DUST（歧义检测）等四类任务。

**📈 对比分析**

评估方法：在每个任务上先计算两种提示（正确/错误）的总体 perplexity 并测算准确率；随后用两种比例指标分析关键词对 perplexity 差异的贡献。实验结果表明：虽然关键词的贡献往往较大，但绝大多数模型并未将其解释力逼近 100%，说明它们在任务上往往依赖非语言学线索，准确率与解释一致性呈负相关。

**⚠️ 局限性**

局限性包括：仅测试了少量公开权重模型与特定任务；关键词定义在某些任务（如刻板印象）上不够清晰；perplexity 的顺序定义可能影响结果；提示语敏感性仍需更广泛探讨。

---

## 274. Scaling Whole-Body Human Musculoskeletal Behavior Emulation for Specificity and Diversity

**arXiv ID:** 2603.29332 | [PDF](https://arxiv.org/pdf/2603.29332v1)

**作者:** Yunyue Wei `[一作]`, Yanan Sui `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套基于GPU的高并行计算框架，用于将全身人体肌肉骨骼动力学与深度强化学习相结合，实现对700肌肉全身运动的精准仿真与内部动力学分析。

**💡 创新点**

创新点在于：①将完整700肌肉模型嵌入GPU原生物理引擎，实现高达千级并行rollout；②使用对抗式差分鉴别器动态聚合追踪误差，自动生成适应性奖励；③引入价值引导流式探索机制，显著提升高维肌肉空间的学习效率。

**🔧 技术方法**

技术包括：GPU加速的MuJoCo Warp物理仿真、对抗式奖励鉴别器、价值引导流（value‑guided flow）探索、PPO强化学习、SMPL‑X 运动重定向、Hill型肌肉模型、逆运动学与姿态匹配。

**📊 数据集**

使用了AMASS（包含步态、跑步、舞蹈、翻滚等多种动作）和Gait‑120（提供步态、GRF、EMG）两大公开数据集进行训练与验证。

**📈 对比分析**

与传统基于逆动力学、单一多目标奖励的DRL方法以及CPU基础的训练基线对比，展示了：①在多种运动（行走、跑步、翻滚、后空翻）下，关节角度平均误差≤7°；②单GPU可在7小时内完成跑步轨迹的高精度学习；③流式探索在相同任务下比PPO收敛速度快≈3×，误差更低。

**⚠️ 局限性**

局限包括：①假设即时无噪声传感反馈，未考虑信号延迟与噪声；②Hill肌肉模型未考虑历史依赖与疲劳；③仅使用单一标准人体模型，缺乏个体解剖与生理差异；④与真实接触力学相比，软体接触模型仍不够精确。

---

## 275. Adversarial Prompt Injection Attack on Multimodal Large Language Models

**arXiv ID:** 2603.29418 | [PDF](https://arxiv.org/pdf/2603.29418v1)

**作者:** Meiwen Ding `[一作]` (Nanyang Technological University), Xudong Jiang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种针对闭源多模态大语言模型的不可察觉视觉提示注入攻击，提出了CoTTA框架；

**💡 创新点**

创新点在于结合隐蔽文本触发器与视觉扰动，使用双目标对齐（图像-文本与图像-图像）以及动态更新目标图像，实现更强的攻击控制与可转移性；

**🔧 技术方法**

采用端到端的梯度优化，利用CLIP等视觉特征提取器，构造双目标对齐损失，使用I‑FGSM更新目标图像，并设计可学习的几何变换隐蔽文本触发器；

**📊 数据集**

使用NIPS 2017 Adversarial Attacks & Defenses竞赛数据集（100张图像）进行图像描述任务，使用ScienceQA（100对图像-问题）进行VQA任务，并在1000张样本上扩展评测；

**📈 对比分析**

与M‑Attack、FOA‑Attack、AnyAttack、Agent‑Attack等基线在闭源模型（GPT‑4o、GPT‑5、Gemini‑2.5、Claude‑4.5）上进行对比，CoTTA在soft准则下ASR达81%，硬准则下74%，比FOA‑Attack高31%点，平均相似度提升0.18；在VQA上ASR达82%/79%，显著优于其他方法；

**⚠️ 局限性**

仍受限于对视觉特征提取器的依赖，部分模型（如Claude‑4.5）鲁棒性较高；需针对不同模型调参；仅在黑盒条件下验证，缺乏白盒下对抗性研究；对抗样本可检测性的完整评估尚未完成。

---

## 276. Beyond Idealized Patients: Evaluating LLMs under Challenging Patient Behaviors in Medical Consultations

**arXiv ID:** 2603.29373 | [PDF](https://arxiv.org/pdf/2603.29373v1)

**作者:** Yahan Li `[一作]` (University of Southern California), Ruishan Liu `[通讯]` (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个双语（英中）基准CPB-Bench，用以评估大型语言模型在面对真实医学对话中出现的四类挑战性患者行为（信息矛盾、事实不准、自我诊断与治疗抵制）时的安全表现；

**💡 创新点**

首次将互动失配行为作为评估维度，引入可观察的失败准则并提供最小编辑的压力测试，揭示了在信息矛盾与异常数值情况下模型普遍缺乏澄清与纠正机制；

**🔧 技术方法**

采用GPT‑4o进行自动评判与人工复核，结合多种提示干预（链式推理、Oracle 指令、患者陈述评估、响应自评）来衡量模型行为；

**📊 数据集**

利用四个现有医学对话数据集（SIMORD、MediTOD、MedDG、IMCS）中标注的692条多轮对话，涵盖英文与中文；

**📈 对比分析**

在单轮和多轮情境下对六大模型族（GPT‑4、GPT‑5、Claude‑Sonnet‑4.5、Gemini‑2.5‑Flash、DeepSeek、Llama‑3.x）进行评估，发现信息矛盾导致高失败率，干预措施效果参差不齐，且往往引入不必要的修正；

**⚠️ 局限性**

局限性包括基准覆盖受限于原始数据集，自动评判可能漏判细微错误，压力测试为人工编辑构造，缺乏真实部署环境验证。

---

## 277. Industrial-Grade Robust Robot Vision for Screw Detection and Removal under Uneven Conditions

**arXiv ID:** 2603.29363 | [PDF](https://arxiv.org/pdf/2603.29363v1)

**作者:** Tomoki Ishikura `[一作]` (Panasonic Holdings Corporation), Kensuke Harada `[通讯]` (University of Osaka)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一套工业级机器人视觉与操控系统，实现空调户外机的螺钉检测与拆卸，目标是满足 75% 以上拆卸完成率与 216 秒以内周期时长。

**💡 创新点**

创新点在于：①采用任务特定的两阶段检测方法，先粗略检测高召回率，再细化识别精度；②利用基于晶格的局部标定策略补偿镜头畸变与机械偏差，实现在大工作空间内 ±0.75 mm 的定位精度；③将宽视角 RGB‑D 摄像机与单次采图相结合，消除传统多次近摄导致的周期延长。

**🔧 技术方法**

技术包括：深度学习两阶段 FCN 检测（使用高召回模型和三模型集成精度模型）、图像预处理（伽马校正、直方图均衡）和几何特征融合；局部标定使用 50 mm 间距的 3D 网格点采集数据，并通过全局-局部双层插值映射摄像机坐标到机器人坐标。

**📊 数据集**

数据集：3,070 个螺钉样本（TEG 试验）用于检测评估；120 台空调户外机的真实拆卸实验用于系统验证。

**📈 对比分析**

与传统单阶段检测与全局标定方法对比，检测召回率达到 99.8%（TP 3,064/3,070），精度 100%；拆卸完成率 78.3%（72/92 通过拆除的机体），平均周期 193 秒，均优于目标指标（≥75% 及 ≤216 s）。

**⚠️ 局限性**

局限性：约 24% 的机体因严重污垢、蜘蛛网覆盖或结构变形而被排除；系统仅适用于空调户外机，难以直接推广到其他设备；若出现极端退化（如极深裂纹或被卡住的螺钉），仍需人工干预或破坏性拆解策略。

---

## 278. Uncertainty-Aware Trajectory Prediction: A Unified Framework Harnessing Positional and Semantic Uncertainties

**arXiv ID:** 2603.29362 | [PDF](https://arxiv.org/pdf/2603.29362v1)

**作者:** Jintao Sun `[一作]` (Beijing Institute of Technology), Zhedong Zheng `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个统一框架，利用双头结构在在线HD地图估计中同时建模位置与语义不确定性，并将其显式融合进轨迹预测模型。

**💡 创新点**

通过双头推理得到位置与语义预测差异作为不确定性度量，并将该不确定性显式注入预测网络，实现对地图噪声的显式建模与补偿。

**🔧 技术方法**

采用双头网络、BEV特征提取、KL散度与MSE衡量位置/语义不确定性、Transformer（HiVT）与GNN（DenseTNT）轨迹预测、dropout及端到端训练等技术。

**📊 数据集**

在nuScenes大规模真实驾驶数据集上进行评估，并在ArgoverseV2上进行可视化验证。

**📈 对比分析**

在四种HD地图估计器与两种轨迹预测器的8组组合上与基线对比，平均提升minADE 8–13%、minFDE 5–10%、MR 2–22%，表现显著。

**⚠️ 局限性**

对异常ground truth轨迹敏感；仅在已知模型上提升，未解决地图不完整时的极端鲁棒性；双头结构略增参数与训练时间。

---

## 279. Rigorous Explanations for Tree Ensembles

**arXiv ID:** 2603.29361 | [PDF](https://arxiv.org/pdf/2603.29361v1)

**作者:** Yacine Izza `[一作]` (National University of Singapore), Joao Marques-Silva `[通讯]` (ICREA & Univ. Lleida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了如何为树集成模型（随机森林与提升树）提供严格、可验证的归纳性与对比性解释。

**💡 创新点**

创新点在于提出统一的命题编码，支持多种投票机制（多数投票、加权投票），并利用 MaxSAT 与 MUS/MCS 算法计算最小解释，同时提供公平性与鲁棒性验证框架。

**🔧 技术方法**

采用 SAT、SMT、MaxSAT（PB）逻辑推理技术，并结合增量式 MaxSAT、未满足子集与最小冲突集求解。

**📊 数据集**

在 31 个公开 UCI 与 PMLB 数据集上实验，涵盖 2–11 类、4–64 特征，训练随机森林（100棵）与提升树（每类 50 棵）模型。

**📈 对比分析**

与 SMT、现有 MaxSAT 与 Anchor 对比，SAT/MaxSAT 方案在绝大多数情形下速度更快、可解释性更优，尤其在大模型上保持良好可扩展性。

**⚠️ 局限性**

主要限制是对极大规模树集成的可扩展性仍有限，且非离散特征的编码与求解开销较高。

---

## 280. BenchScope: How Many Independent Signals Does Your Benchmark Provide?

**arXiv ID:** 2603.29357 | [PDF](https://arxiv.org/pdf/2603.29357v1)

**作者:** Tommy Sha `[一作]` (Stony Brook University), Stella Zhao `[通讯]` (University of Minnesota Twin Cities)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了一种快速、闭式的有效维度（Effective Dimensionality, ED）诊断方法，用于衡量 AI 评测基准是否提供独立信息。

**💡 创新点**

创新点在于将 ED 定义为参与比（participation ratio）并与 Renyi 熵、随机矩阵理论相结合，提供一个基于谱分解的、无阈值、可直接计算的测量宽度上限，并构建了首个 22 基准的冗余图谱。

**🔧 技术方法**

核心技术包括：任务中心化的二进制通过 SVD 分解得到奇异值，计算 ED；与随机矩阵理论的 Marchenko–Pastur 定律得到理论基线；对 ED 进行置信区间、置换检验、分层匹配等统计验证；以及基于 ED 的贪心任务选择算法 ED‑Greedy。

**📊 数据集**

使用了 8,400+ 个模型、22 个评测基准（涵盖 8 个领域）的公共评分矩阵，包含 Open LLM Leaderboard、BFCL、MMLU‑Pro、BigCodeBench、SWE‑bench 等典型基准。

**📈 对比分析**

与传统的因子分析、ICM、IRT 等方法对比，ED 在计算速度、可扩展性、无阈值和能量保持方面表现突出；在实践中显示基准的冗余程度（如 BBH 与 MMLU‑Pro ρ≈0.96）、排名脆弱性以及负相关现象的量化，进一步指导基准设计与维护。

**⚠️ 局限性**

局限性包括：二进制矩阵下的 ED 对真实维度的上限估计，受模型族和任务规模影响；对不同人群或模型群体的结果需重新计算；对任务内相关性不足时，ED 可能低估真实维度；负相关的条件性和解释依赖于当前模型生态。

---

## 281. Beyond Corner Patches: Semantics-Aware Backdoor Attack in Federated Learning

**arXiv ID:** 2603.29328 | [PDF](https://arxiv.org/pdf/2603.29328v1)

**作者:** Kavindu Herath `[一作]` (Purdue University), Saurabh Bagchi `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在联邦学习中使用语义一致、自然触发器的后门攻击方法，并在不同鲁棒聚合下评估其效果。

**💡 创新点**

创新点在于：①构造视觉上真实、符合分布的语义触发器（如戴太阳镜、贴小帽子）；②设计聚合感知的恶意目标函数，包括特征分离损失和参数正则化，使恶意更新既能实现后门又能保持与全局模型的相似性；③在多种鲁棒聚合规则下仍能保持高攻击成功率。

**🔧 技术方法**

采用的技术包括：基于MGIE的语义编辑生成触发器；联合交叉熵损失、特征分离损失和参数正则化的组合；使用ResNet‑18/VGG‑16作为网络骨干；FedAvg、Trimmed Mean、MultiKrum、FLAME、FilterFL等聚合算法。

**📊 数据集**

使用了CelebA（人发色分类）和GTSRB（交通标志识别）两个视觉数据集进行实验。

**📈 对比分析**

与Bagdasaryan等的基线（仅用混合清洁/触发样本的传统后门）对比，在所有聚合规则下，提出的SABLE方法在保持近似的清洁准确率的同时，显著提升攻击成功率（FedAvg/Trimmed Mean/FLAME等提升至80%+，MultiKrum/FilterFL提升至≈90%+），证明其在鲁棒聚合下的优势。

**⚠️ 局限性**

局限性包括：实验在高性能GPU模拟环境下完成，未考虑边缘设备的计算/能耗限制；语义触发器生成依赖外部生成模型和属性标注，实用性受限；攻击者需额外准备触发样本，可能增加检测难度。

---

## 282. AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models

**arXiv ID:** 2603.29410 | [PDF](https://arxiv.org/pdf/2603.29410v1)

**作者:** Yubo Cui `[一作]` (Harbin Institute of Technology), Zheng Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在预训练的视觉‑语言模型（如CLIP）上提出了一种对抗微调框架AGFT，旨在提升零样本对抗鲁棒性并保持跨模态语义结构；

**💡 创新点**

创新点在于用原始模型的概率分布（软标签）而非硬标签作为对抗训练的监督，并加入分布一致性校准，通过温度缩放保持视觉‑文本对应关系；

**🔧 技术方法**

技术主要包括：文本引导的对抗训练（soft‑label监督）、分布一致性校准（温度缩放）、PGD对抗样本生成、CLIP的InfoNCE预训练与对抗微调；

**📊 数据集**

使用ImageNet作为微调数据，在15个零样本基准（Caltech101/256, CIFAR10/100, STL10, DTD, EuroSAT, FGVC, Flower102, Food101, OxfordPet, StanfordCars, SUN397, ImageNet‑R, ImageNet‑S）以及多种攻击（PGD‑20, C&W, AutoAttack, MI‑FGSM, DI2‑FGSM, TI‑FGSM, NI‑FGSM, PI‑FGSM, PI‑FGSM++）进行评估；

**📈 对比分析**

与四个主流对抗微调基线（TeCoA, PMG‑AFT, TGA‑ZSR, GLADIATOR）对比，AGFT在大多数数据集与攻击下平均提升了约3–5%的零样本鲁棒准确率，同时保持或略提升清洁准确率；在不同攻击强度、架构和OOD场景中也表现出更好的泛化与鲁棒性；

**⚠️ 局限性**

限制包括：对抗微调仍需较高计算成本（尤其是PGD生成）；温度与γ参数需经验调优；在某些ResNet基准下，AGFT的清洁准确率略低于部分基线；对极大扰动或更强攻击的鲁棒性仍有限。

---

## 283. Communication Outage-Resistant UUV State Estimation: A Variational History Distillation Approach

**arXiv ID:** 2603.29512 | [PDF](https://arxiv.org/pdf/2603.29512v1)

**作者:** Shuyue Li `[一作]` (Xi'an Jiaotong-Liverpool University), Xiaohui Qin `[通讯]` (Jiangsu JITRI Tsingunited Intelligent Control Technology Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种变分历史蒸馏（VHD）方法，用于在水声通信中断时通过虚拟测量融合物理运动模型与历史轨迹，实现UUV状态估计；

**💡 创新点**

创新点在于将轨迹预测重新表述为近似贝叶斯推理，利用变分推断合成虚拟测量，并引入自适应置信度机制；

**🔧 技术方法**

采用变分推断、卡尔曼滤波（UKF）与多项式回归相结合的技术架构，并通过动态噪声协方差调节实现自适应加权；

**📊 数据集**

使用高保真仿真数据，包含海流扰动、IMU随机漂移的多次（100次）蒙特卡洛实验；

**📈 对比分析**

与传统UKF及拉格朗日预测器对比，VHD在40秒通信中断下将位置RMSE从约170 m降低至≈15 m，误差下降率达91%；

**⚠️ 局限性**

局限在二维平面运动、需历史轨迹数据、长时间中断时虚拟测量可靠性衰减、参数调优敏感以及未验证六自由度场景。

---

## 284. Transmittance-Guided Structure-Texture Decomposition for Nighttime Image Dehazing

**arXiv ID:** 2603.29507 | [PDF](https://arxiv.org/pdf/2603.29507v1)

**作者:** Francesco Moretti `[一作]` (Maharaja Agrasen University), Andrea Gallo `[通讯]` (Maharaja Agrasen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了双阶段夜间图像去雾框架，先通过自适应传输率校正和二次高斯滤波估计大气光，再利用STAR‑YUV分解结构与纹理并采用两阶段融合得到最终图像。

**💡 创新点**

创新点在于结合区域自适应传输率校正、YUV空间的二次高斯滤波大气光估计、改进夜间成像模型、STAR‑YUV结构纹理分解以及非线性与线性双阶段融合，形成了物理先验与层优化联动的完整去雾流程。

**🔧 技术方法**

使用了传输率校正、二次高斯滤波、大气光估计、改进夜间成像模型、STAR‑YUV分解、伽马校正、MSRCR、LoG、Retinex融合和线性混合等技术。

**📊 数据集**

在3R和RESIDE夜间雾图像构成的ZS330（334张真实）和HC770（775对合成）数据集上进行实验。

**📈 对比分析**

与六个现有昼夜去雾算法（DCP、DehazeNet、GMLC、CEEF、IAT、Fb）在PSNR、SSIM、AG、IE、NIQE等指标上进行对比，平均PSNR 17.024 dB、SSIM 0.765，显著优于其他方法，尤其在色彩恢复、结构保持和细节增强方面表现最优。

**⚠️ 局限性**

局限性包括处理速度仍不及最快方法（如CEEF/DehazeNet），未扩展到视频去雾，且对极端雾厚或强光源场景的鲁棒性需进一步验证。

---

## 285. Hierarchical Battery-Aware Game Algorithm for ISL Power Allocation in LEO Mega-Constellations

**arXiv ID:** 2603.29506 | [PDF](https://arxiv.org/pdf/2603.29506v1)

**作者:** Kangkang Sun `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Hierarchical Battery-Aware Game（HBAG）算法，针对LEO大规模星座中的ISL功率分配问题，解决电池动态与能量可持续性挑战。

**💡 创新点**

创新点：① 将电池状态嵌入精确势游戏框架并证明唯一变分均衡；② 设计统一分布式更新规则，既可在有限玩家场景下收敛到唯一均衡，又能在玩家数量趋向无穷时自适应逼近均值场均衡；③ 通过动态功率上限与软电池惩罚双重机制同时实现能量安全与网络效率。

**🔧 技术方法**

技术手段：精确势游戏理论、变分均衡分析、均值场游戏（MFG）与Wasserstein收敛证明、分布式拉格朗日优化、并行迭代求解。

**📊 数据集**

实验基准：SpaceX Starlink Shell A 172颗卫星的轨道与能量参数，扩展到5,000颗卫星；使用仿真生成的全球流量分布和太阳能收集模型。

**📈 对比分析**

与SATFLOW、MAAC‑IILP、DeepISL及改进SMFG对比；HBAG实现100 %能量可持续率（提升87.4pp），流量违约率7.6 %（低于10 %工业容忍度），能效3.96 Mbit/kJ；收敛速率为O(1/√k)，单时隙运行时间随卫星数线性增长至≤75 ms，满足实时需求。

**⚠️ 局限性**

局限性：① 依赖准静态电池假设，忽略短时电量波动；② 仅考虑确定性日照和阴影模型，未覆盖随机天气或突发遮挡；③ 预设轨道与链路拓扑，未考虑链路重构或故障恢复机制。

---

## 286. Model Predictive Path Integral PID Control for Learning-Based Path Following

**arXiv ID:** 2603.29499 | [PDF](https://arxiv.org/pdf/2603.29499v1)

**作者:** Teruki Kato `[一作]` (Toyota Central R&D Labs Inc), Seigo Ito `[通讯]` (Toyota Central R&D Labs Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于MPPI的PID增益在线优化方法（MPPI–PID），实现对工业车辆路径跟踪的实时控制。

**💡 创新点**

创新点在于将高维控制输入序列优化转化为低维PID增益优化，提升采样效率并保持PID结构的平滑性。

**🔧 技术方法**

采用MPPI采样优化、残差学习的物理-神经网络动力学模型、PID控制器和信息论视角的理论分析。

**📊 数据集**

使用从真实迷你叉车在实验室采集的约70,926条连续驾驶数据（含状态与控制输入），经过预处理后划分训练/验证/测试集。

**📈 对比分析**

通过与固定增益PID和传统MPPI对比实验，MPPI–PID在跟踪误差与控制增量上保持或优于传统MPPI，且在样本数极低（I=16）时仍能保持良好性能。

**⚠️ 局限性**

局限性包括尚未在真实硬件上验证、对模型不确定性和障碍物避让的适用性未充分探讨。

---

## 287. Why not to use Cosine Similarity between Label Representations

**arXiv ID:** 2603.29488 | [PDF](https://arxiv.org/pdf/2603.29488v1)

**作者:** Beatrix M. G. Nielsen `[一作]` `[通讯]` (IT University of Copenhagen), Beatrix M. G. Nielsen (IT University of Copenhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

研究了在 softmax 分类器中，标签表示（unembeddings）之间的余弦相似度与模型预测概率无关，证明通过平移标签向量可以得到相同概率但余弦相似度变为 ±1 的等价模型。

**💡 创新点**

创新点在于揭示 softmax 模型的标签向量平移不影响概率却能大幅改变余弦相似度，说明余弦相似度不能用于解释或预测 softmax 模型的输出概率。

**🔧 技术方法**

采用了数学证明、向量平移与归一化等变换方法，并通过构造示例向量进行可视化演示，说明等价模型之间的余弦相似度差异。

**📊 数据集**

论文中没有使用实际数据集，示例仅为手工构造的向量，代码与绘图已发布在 GitHub（https://github.com/bemigini/cosine-sim-not-informative）。

**📈 对比分析**

由于缺乏实测数据，本文未给出性能对比或指标；通过示例与可视化阐明余弦相似度与概率的无关性。

**⚠️ 局限性**

局限性包括：仅针对 softmax 分类器；未考虑嵌入向量（embeddings）的影响；缺乏在大型真实模型上的实验验证；并未探讨多分类之外的模型或其他相似度度量的情况。

---

## 288. An Isotropic Approach to Efficient Uncertainty Quantification with Gradient Norms

**arXiv ID:** 2603.29466 | [PDF](https://arxiv.org/pdf/2603.29466v1)

**作者:** Nils Grünefeld `[一作]`, Christian Hardmeier `[通讯]` (IT University of Copenhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过一阶泰勒展开与参数协方差等方假设，推导出在预训练LLM上仅用一次前向-后向传播即可得到的本体不确定性与噪声不确定性分解。

**💡 创新点**

核心创新是将参数协方差假设为等方，证明其在大模型规模下可近似真实协方差，并将本体不确定性简化为梯度范数平方，首次在LLM中实现轻量级不确定性估计。

**🔧 技术方法**

采用一阶泰勒（delta）方法、等方协方差假设、梯度范数计算、伯努利方差估计，并在合成任务上与MCMC、拉普拉斯近似对比，在问答任务中使用AUROC评估。

**📊 数据集**

验证使用合成二维分类、回归和多分类数据集；下游任务使用TriviaQA和TruthfulQA两个公开问答基准。

**📈 对比分析**

与MCMC（本体）、拉普拉斯近似、P(True)、语义熵和无偏熵等基线对比，结果显示在TruthfulQA上联合估计AUROC达0.63，优于所有基线；在TriviaQA上P(True)仍占优；梯度方法比采样/自评方法快46–107倍。

**⚠️ 局限性**

局限性包括等方假设在中等规模模型和回归任务下误差明显，梯度范数尺度缺乏直观解释，跨模型泛化差异大，模型与采样方差高导致单个预测的可靠性有限，且仅在~10⁶参数规模下验证，LLM级别缺乏严格保证。

---

## 289. LLM Probe: Evaluating LLMs for Low-Resource Languages

**arXiv ID:** 2603.29517 | [PDF](https://arxiv.org/pdf/2603.29517v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 290. Loop-Checking and Counter-Model Extraction for Intuitionistic Tense Logics via Nested Sequents

**arXiv ID:** 2603.29424 | [PDF](https://arxiv.org/pdf/2603.29424v1)

**作者:** Tim S. Lyon `[一作]` `[通讯]` (Technische Universität Dresden), Tim S. Lyon (Technische Universität Dresden)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种用于直观时态逻辑（ITL）的嵌套序列证明搜索方法，并实现了从失败搜索中提取有限反模型的机制；通过该方法证明了 ITL 的有限模型性质和可判定性。

**💡 创新点**

创新点在于：①针对非可逆的嵌套序列规则，设计了计算树（computation tree）取代传统单一推导；②提出了一种基于同态（homomorphism）的循环检查机制，能有效检测并剪枝重复出现的嵌套序列；③利用计算树结构实现了证明与反模型的同时提取。

**🔧 技术方法**

核心技术包括：嵌套序列系统（multi‑conclusioned）、计算树与弱/强同态映射、基于同态的循环检测、以及从计算树构造蓝图（blueprint）再到反模型的构造。

**📊 数据集**

论文为理论性研究，未使用任何实验数据集；所有结论均通过形式化证明得到。

**📈 对比分析**

未进行实验对比。作者指出该证明搜索算法在最坏情况下具有指数级（甚至超指数级）时间复杂度，主要受循环检测与非可逆规则的影响；因此在性能上尚未给出具体评估。

**⚠️ 局限性**

局限性主要体现在：①算法复杂度高，导致实际应用中可能遇到计算瓶颈；②当前只针对非转移（transitive）直观模态逻辑；③尚未证明其在更广泛的直观语法逻辑（IGL）或转移模态逻辑中的适用性，未来工作需进一步优化复杂度或扩展到更一般的逻辑。

---

## 291. Multimodal Models Meet Presentation Attack Detection on ID Documents

**arXiv ID:** 2603.29422 | [PDF](https://arxiv.org/pdf/2603.29422v1)

**作者:** Marina Villanueva `[一作]` (Facephi), Juan E. Tapia `[通讯]` (Hochschule Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种最新多模态大语言模型（PaLIGemma2-3b-mix-224、LLaVA1.6-7b-mistral、Qwen2.5-3b-instruct）进行统一的ID卡/护照伪造检测（PAD）评测，设计并试验了七类提示词以探究模型在真实场景下的判别能力。

**💡 创新点**

系统性评估多模态LLM在PAD任务中的表现，并通过不同提示词构造展示模型对视觉纹理特征的“物理盲点”和对提示语义的偏倚性决策，揭示现有LLM在安全关键任务中的不可行性。

**🔧 技术方法**

使用多模态LLM（视觉-文本融合模型）、提示工程（单轮、多轮、示例、背景、任务式、配方式）、以及基于输出logits的二元判别（阈值0.5、EER、BPCER10/20）进行实验。

**📊 数据集**

自建包含100张图像的数据集：80张真实身份证/护照（来自古巴、智利、西班牙、尼加拉瓜、厄瓜多尔、萨尔瓦多），以及85张攻击样本（打印、屏幕、PVC、篡改等），覆盖七国样本。

**📈 对比分析**

通过APCER/BPCER、EER、BPCER10/20等标准指标进行对比，发现最优配置（Qwen2.5-3b-instruct+Simple_8）EER约25%，但所有模型的BPCER10/20均高于90%，整体性能远低于可接受的工业基准。

**⚠️ 局限性**

主要局限在于模型缺乏对纹理/光照细节的感知，导致“物理盲点”，并在提示词语义驱动下产生系统性偏倚，总是倾向于判为真伪，因而不适合直接部署在身份验证系统中。

---

## 292. Baby Scale: Investigating Models Trained on Individual Children's Language Input

**arXiv ID:** 2603.29522 | [PDF](https://arxiv.org/pdf/2603.29522v1)

**作者:** Steven Y. Feng `[一作]` (Stanford University), Michael C. Frank `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练小型语言模型（GPT‑2 与 GPT‑BERT）在来自20个单语英语家庭的 BabyView 对话数据上，系统探究数据规模、质量及其对模型性能与儿童词汇习得的影响。

**💡 创新点**

首次将单一家庭真实对话视作“训练环境”，比较不同家庭语料对模型学习的差异，并将模型对儿童词汇的似然性与儿童 CDI 词汇习得关联，利用语言模型作为儿童环境的计算探针。

**🔧 技术方法**

使用小型 Transformer 训练、基于 175 个语言特征（分布式、句法、交互等）的 Spearman、Lasso、XGBoost 相关性分析，以及基于 CDIs 的年龄模型预测。

**📊 数据集**

BabyView 真实对话转录（约 2.8M 词，20 家庭）；TinyDialogues 合成对话（10M–200M 词）；CHILDES 24.5M 词英文子集做对照。

**📈 对比分析**

通过比较单一家庭、不同数量混合、全部家庭模型，评估 Zorro（语法）、WordSim、COMPS、EWoK 四项基准；发现语法任务随数据量呈幂律提升，语义和世界知识提升缓慢，混合模型在部分基准上可超越最大单家模型；特征分析表明分布式与句法多样性对性能影响最大。

**⚠️ 局限性**

样本仅 20 家庭、单语种、未覆盖多模态或外部文本；模型参数仅至 30M，缺乏更广泛验证，且与儿童词汇习得的关联并未显著优于频率模型，结果仍属初步。

---

## 293. Learning to Generate Formally Verifiable Step-by-Step Logic Reasoning via Structured Formal Intermediaries

**arXiv ID:** 2603.29500 | [PDF](https://arxiv.org/pdf/2603.29500v1)

**作者:** Luoxin Chen `[一作]` (Wangxuan Institute of Computer Technology, Peking University), Huishuai Zhang `[通讯]` (Wangxuan Institute of Computer Technology, Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于结构化形式中间件的强化学习框架（ProSFI），训练大语言模型输出自然语言推理和对应的JSON/YAML 形式中间步骤，并让形式化证明器逐步验证每一步，从而给出细粒度的奖励。

**💡 创新点**

创新点在于：①将自然语言推理拆解为可验证的结构化中间步骤，避免模型直接生成完整证明的难度；②利用形式化证明器提供的通过/失败信号构造一步步的奖励；③实现了可在 7B 模型上高效训练、并能与 DTV 组合实现测试时的可扩展性。

**🔧 技术方法**

使用的技术包括：Qwen2.5‑7B‑Instruct 作为基础模型；Group Relative Policy Optimization (GRPO) 进行 RL 后训练；结构化 JSON/YAML 形式中间件；形式化证明器（Prover9/Lean/Z3）进行逐步验证；DTV（Don't Trust; Verify）进行多样本测试时的可验证性筛选。

**📊 数据集**

实验数据集：ProverQA（合成的一阶逻辑推理数据集，分难度层级），ProverQA‑Extra（OOD 扩展集），以及 Knights and Knaves 传统逻辑推理数据集，用于检验方法的跨域泛化。

**📈 对比分析**

与仅基于最终答案奖励（Outcome‑CoT）以及直接生成 Lean 证明的基线对比，ProSFI 在保持类似的最终答案准确率（约 92‑93%）的同时，将推理路径的可信度（Soundness）从 20%‑30% 提升至 55%‑76%（在 hardest 子集）并在 OOD 集上也显著优于基线；在 Knights and Knaves 数据集上亦保持高准确率同时显著提升 GPT‑Soundness。

**⚠️ 局限性**

局限性：目前仅针对一阶逻辑任务；依赖形式化证明器的可用性和验证速度；奖励机制仍是二值通过/失败，缺乏更细粒度的信用分配；在更复杂的自然语言推理或更大模型规模下的可扩展性与效果仍待验证。

---

## 294. Calibrated Confidence Expression for Radiology Report Generation

**arXiv ID:** 2603.29492 | [PDF](https://arxiv.org/pdf/2603.29492v1)

**作者:** David Bani-Harouni `[一作]` (Technical University of Munich), Matthias Keicher `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过强化学习对医学大型视听-语言模型进行微调，使其在生成胸片报告时能够同时输出经过校准的置信度估计，从而实现报告级和句子级的置信度表达，辅助临床医生进行目标性复核。

**💡 创新点**

创新点在于：①将严格正则化的对数打分规则与多模态强化学习（GRPO）相结合，实现置信度自我评估的完全无监督校准；②支持报告级与句子级两层置信度表达，提供细粒度的审核线索；③仅对置信度词元进行损失，避免影响报告生成质量。

**🔧 技术方法**

采用的核心技术包括：强化学习框架（GRPO）+对数打分奖励，LoRA微调策略，4‑bit量化的预训练视听‑语言模型，绿色（GREEN）报告质量度量作为外部正确性信号。

**📊 数据集**

主要使用的公开数据集为 MIMIC‑CXR 进行训练与评估，并在 IU‑Xray 上做零样本外域测试；评价指标包括 ECE、Pearson 相关、AUROC、GREEN 分数以及临床专家评分。

**📈 对比分析**

相较于基线方法（Verbalize Base/Supervised、Sequence Probability、P(True)、Self‑Consistency、Trained Probe），ConRad 在报告级 ECE 下降 80% 以上、相关性提升至 0.43，句子级 ECE 降 40% 以内且 AUROC 提升至 0.63；外域测试 ECE 亦显著下降，说明泛化能力更强。

**⚠️ 局限性**

局限性包括：①模型仍需人工复核低置信度句子，不能完全自动化；②训练过程对奖励设计和正确性度量敏感，若 GREEN 评分误差大可能影响校准；③仅在胸片任务验证，跨模态或其他疾病影像的适用性尚未彻底检验。

---

## 295. AI-Simulated Expert Panels for Socio-Technical Scenarios and Decision Guidance

**arXiv ID:** 2603.29470 | [PDF](https://arxiv.org/pdf/2603.29470v1)

**作者:** Andrew G. Ross `[一作]`, Alan M. Ross `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用大型语言模型模拟专家面板，完成了德国能源转型至2050的社会技术情景构建、跨影响平衡（CIB）分析、动态路径生成、AI主导的多准则决策（MCDA）以及情景量化，形成可直接用于评估模型的定量输入。

**💡 创新点**

创新点在于：①将AI模拟的专家面板与传统CIB、MCDA结合，实现快速、一致且可追溯的情景生成；②引入结构性与动态随机冲击，系统性评估路径稳健性与多样性；③通过完整对话记录构建“玻璃盒”验证体系，为情景规划提供前所未有的透明度与可审计性；④构建虚拟AI驱动决策实验室，支持多主体视角的政策压力测试。

**🔧 技术方法**

技术手段包括：大型语言模型（Gemini 3 Flash），CIB与PyCIB库的概率扩展、冲击模拟；多准则决策分析框架；人工设计的专家角色与议题框架；完整的对话与决策记录生成与存档。

**📊 数据集**

使用的数据主要为：人工构造的专家面板角色与背景信息、情景构建提示文本；从模拟面板产生的跨影响矩阵、状态定义与冲击参数；没有使用外部真实数据集，而是完全基于AI生成的合成信息。

**📈 对比分析**

方法对比：文章未与传统人工专家小组进行直接实验对比，而是通过示例图表展示冲击实验与路径多样性。性能优势表现在：情景生成从数月缩短到数天；并提供可重复、可调节的流水线；在稳健性评估方面，通过多重Monte‑Carlo路径与冲击实验获得更全面的结果。

**⚠️ 局限性**

局限性包括：①CIB内部一致性不等价于叙事连贯性，仍需人工后期审查；②AI模型的知识与推理受训练数据限制，可能产生不准确或偏颇的判断；③离散状态设定导致路径平滑度与真实动态存在差异；④缺乏对现实政策或技术发展的外部验证；⑤需要在关键节点引入人工干预以校正模型误差。

---

## 296. CReF: Cross-modal and Recurrent Fusion for Depth-conditioned Humanoid Locomotion

**arXiv ID:** 2603.29452 | [PDF](https://arxiv.org/pdf/2603.29452v1)

**作者:** Yuan Hao `[一作]` (Zhejiang University), Qiuguo Zhu `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该研究提出了一种单阶段、基于前视深度的类人机器人行走框架CReF，直接将本体感知和深度图映射到关节位置目标；

**💡 创新点**

创新点包括跨模态注意力实现本体感知与深度特征的条件融合、门控残差融合与递归融合的多模态记忆机制，以及基于地形的足部放置奖励，提升了足部放置的准确性与鲁棒性；

**🔧 技术方法**

主要技术涉及轻量化深度CNN分词器、跨模态注意力、门控残差融合（GRF）、GRU递归融合、桥式门控输出以及辅助深度估计与强化学习奖励设计；

**📊 数据集**

实验使用大规模仿真环境（Isaac Gym 4096并行），训练后直接零样本迁移到搭载RealSense D435i深度相机的AGIBOT X2 Ultra机器人；

**📈 对比分析**

与基线HPL以及消融实验对比，CReF在多种地形（台阶、平台、隙缝）中成功率均超过99%，在难度和OOD场景亦保持高成功率；在真实世界20次试验中，台阶、平台、隙缝分别获得20/20、20/20、18/20的成功率，整体表现优异；

**⚠️ 局限性**

局限性在于仅依赖深度感知，易受光照和反射影响；缺乏颜色纹理信息，难以在极端光照或强反射环境下保持同等鲁棒性。

---

## 297. mtslearn: Machine Learning in Python for Medical Time Series

**arXiv ID:** 2603.29432 | [PDF](https://arxiv.org/pdf/2603.29432v1)

**作者:** Zhongheng Jiang `[一作]` (Nanjing University of Information Science and Technology), Shenda Hong `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了 mtslearn，一个面向医学时间序列数据的端到端集成工具包，简化数据处理、特征工程和模型训练流程。

**💡 创新点**

提供统一的数据接口规范，将宽表、长表、扁平表自动转换为统一宽表，并实现全流程无缝集成，显著降低临床研究者的编程门槛。

**🔧 技术方法**

结合 scikit‑learn、PyTorch、XGBoost、Lifelines 等主流库，支持传统统计模型（Logistic、Cox）与深度学习模型（LSTM、Time‑aware LSTM、Transformer）以及可扩展的自定义模型。

**📊 数据集**

在 COVID‑19 375 病例表格数据上验证静态预测功能，在 PhysioNet 2019 败血症时间序列数据上验证端到端时间序列预测功能。

**📈 对比分析**

与单独使用 XGBoost、LSTM 或 Transformer 的传统工具对比，mtslearn 在 COVID‑19 任务中实现约 85% 的准确率，在败血症任务中达 0.88 的 ROC‑AUC，性能与现有最优模型相当且使用更为便捷。

**⚠️ 局限性**

目前仅支持基础深度学习模型，缺乏高级架构与可解释性模块，限制了对更复杂任务的适用性。

---

## 298. VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference

**arXiv ID:** 2603.29494 | [PDF](https://arxiv.org/pdf/2603.29494v1)

**作者:** Anmin Liu `[一作]` (Peking University), Tao Xie `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对长视频上下文，提出了一种向量级稀疏注意力框架 VecAttention，用于在视频理解和生成模型中显著降低计算量并保持精度。

**💡 创新点**

创新点在于：①发现视频注意力图呈现细粒度的垂直向量稀疏模式；②通过轻量级重要向量选择（TilingSelect + minS 过滤）在不写回 HBM 的前提下实现高效的向量选取；③采用动态头部过滤比例和交叉 tile 归一化，进一步提升准确率和性能。

**🔧 技术方法**

技术要点包括：垂直向量稀疏模式、查询池化与估计注意力、minS 排序无关阈值过滤、TilingSelect 结合 GEMM 的融合实现、GPU 优化核（FlashAttention 风格）以及跨 tile 的 max 归约。

**📊 数据集**

评估数据集：视频理解方面使用 VideoMME、LongVideoBench、VCRBench；视频生成方面使用 VBench。

**📈 对比分析**

与全注意力、FlexPrefill、XAttention、AnchorAttention 等基线比较，VecAttention 在保持接近全注意力准确率的同时，获得 2.65× 的注意力加速和 1.83× 的稀疏加速，TTFT 也提升 1.17×。

**⚠️ 局限性**

局限性包括：需要对不同模型手动调优向量大小 Pq、K tile 大小 Bk 和 Gk 等超参；对极长序列时仍会有一定选择开销；目前实验主要聚焦于视频任务，跨模态或更大规模模型的推广性尚待验证。

---

## 299. Metriplector: From Field Theory to Neural Architecture

**arXiv ID:** 2603.29496 | [PDF](https://arxiv.org/pdf/2603.29496v1)

**作者:** Dan Oprisa `[一作]` (Spheroid Labs), Peter Toth `[通讯]` (Spheroid Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 Metriplector，一个基于 metriplectic 动力学的物理原语神经网络，能够在迷宫、数独、CIFAR‑100 图像分类和语言建模四个任务中统一实现推理、识别与生成。

**💡 创新点**

创新点在于：① 将 GENERIC 框架中的耗散与哈密顿通道统一为网络动力学；② 利用 Noether 定理得到的应力‑能量张量 T^μν 作为自然且高效的读出；③ 通过“从输入定义算子、由场求解”原则实现参数化的能量景观；④ 在不同任务中按需求激活耗散或完整 metriplectic 通道，形成一条从单纯 Poisson 解析到全动态的谱；⑤ 采用多层多网格与因果 Poisson 递归实现高效并行求解。

**🔧 技术方法**

核心技术包括：GENERIC 方程实现的可微屏蔽 Poisson 求解（CG + 逆向自适应梯度）；多级多网格 V‑cycle；对齐的 8‑方向扫描；Euler/离散积分；学习得到的对称/反对称耦合张量；使用 Noether 定理得到的梯度相关特征；以及在语言模型中使用 O(N log N) 关联扫描的因果 Poisson 层。

**📊 数据集**

使用的数据集有：15×15/39×39 迷宫（训练 250 个、测试 200 个）；9×9 数独（10 k 训练、1 k 验证）；CIFAR‑100（32×32 彩色图像，100 类）；FineWeb 文本（1024 词表，2 B 训练标记）。

**📈 对比分析**

与基准比较：迷宫 F1 = 1.0（43.8 k 参数），数独 97.2%（120 k 参数，无结构注入），CIFAR‑100 81.03%（2.26 M 参数，DenseNet‑BC 82.8% 需要 25.6 M 参数），语言建模 1.182 BPB（3.6× 更少的训练标记，相比 GPT‑4.0‑b 1.224 BPB）。

**⚠️ 局限性**

局限性包括：在 CIFAR‑100 上仍略逊于最先进的 DenseNet/ConvNeXt；语言模型在计算效率上不及 GPT；未验证在 ImageNet 等更大图像任务上的可扩展性；对称/反对称张量的学习仍受深度与参数规模限制；未使用能保持哈密顿守恒的更高阶积分器；尚未实现单一网络覆盖所有四个任务。

---

## 300. Variational Graph Neural Networks for Uncertainty Quantification in Inverse Problems

**arXiv ID:** 2603.29515 | [PDF](https://arxiv.org/pdf/2603.29515v1)

**作者:** David Gonzalez `[一作]`, Elias Cueto `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种在图神经网络解码器中嵌入变分层的变分图神经网络（VGNN），用于在逆问题中估计材料参数和载荷，并给出置信区间。

**💡 创新点**

创新点在于将变分推理仅应用于解码器，既保持了图网络对几何拓扑的天然适应，又显著降低了贝叶斯网络的计算成本；同时实现了同时量化认知不确定性与统计不确定性的高效方法。

**🔧 技术方法**

使用的技术包括图神经网络（encoder–processor–decoder结构）、变分推理（局部重参数化、ELBO优化）、Swish激活函数、Scale Mixture Prior、Adam优化器以及局部重参数化技巧。

**📊 数据集**

训练数据来自公开的 NASA Pigans 材料识别数据集（二维弹性模量实验）和使用有限元方法生成的三维 Neo‑Hookean 超弹性梁仿真（含不同载荷和位置），共计约 115/260 个高精度模拟。

**📈 对比分析**

通过对训练集和测试集的相对RMSE比较，VGNN在弹性模量恢复和载荷定位任务中均实现了低误差（RRMSE 在 5–10% 之间）且能够提供高置信区间（97.72%）。与传统确定性网络相比，VGNN在不确定性量化和对未见几何的泛化上表现更优。

**⚠️ 局限性**

局限性包括：解码器的变分层虽然降低了成本，但仍增加了模型复杂度；对网格尺寸分布的学习敏感，若训练时网格尺度与推理时相差较大，性能可能下降；以及对近 Dirichlet 边界的载荷定位不确定性较高，需进一步改进模型或数据。

---

## 301. Target-Aligned Reinforcement Learning

**arXiv ID:** 2603.29501 | [PDF](https://arxiv.org/pdf/2603.29501v1)

**作者:** Leonard S. Pleiss `[一作]` (Technical University of Munich), Maximilian Schiffer `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种新的强化学习框架——Target‑Aligned Reinforcement Learning (TARL)，通过在更新时优先选择在线网络与目标网络估计一致的转移，减少目标网络滞后导致的偏差。

**💡 创新点**

设计了基于在线/目标误差一致性的目标对齐度量，并将其作为过采样机制融入标准离线‑在线 RL 算法，从而在不增加超参数的情况下跳过不对齐的更新。

**🔧 技术方法**

结合了经验回放、目标网络、TD 学习、soft/hard target 更新以及对齐度量的过采样策略，并在理论上阐述方向一致性与收敛速度的关系。

**📊 数据集**

在离散控制的 MinAtar（Asterix、Breakout、Freeway、SpaceInvaders）和连续控制的 MuJoCo（HalfCheetah、Ant、Hopper、Walker2d、Swimmer、Humanoid）任务上进行实验。

**📈 对比分析**

直接在原始算法（DQN、DDQN、SAC）上插入 TARL，无需重新调参，实验显示在所有环境中均显著加速收敛并提升最终性能，五/六个环境中均优于基线。

**⚠️ 局限性**

需要额外的前向传播来计算对齐度量，导致训练时间略增；对齐度量假设在线网络能代表未来目标，可能在极端非平稳环境下失效；未在大规模环境或策略梯度方法上验证。

---

## 302. Distilling Human-Aligned Privacy Sensitivity Assessment from Large Language Models

**arXiv ID:** 2603.29497 | [PDF](https://arxiv.org/pdf/2603.29497v1)

**作者:** Gabriel Loiseau `[一作]`, Marc Tommasi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用知识蒸馏从大型LLM（Mistral Large 3）得到高效的文本隐私敏感度评估器；

**💡 创新点**

证明小型编码器在隐私判断上能超过教师模型，与人类评价一致性更高；

**🔧 技术方法**

采用Prompting+LLM标注、5分类隐私敏感度分级、Krippendorff’s α评估、Mistral Large 3为教师、Ettin/ModernBERT等轻量级模型为学生；

**📊 数据集**

从10个公开用户文本数据集共约20万条样本自动标注，再在677名人类评测者的250文本基准上验证；

**📈 对比分析**

在保留测试集上，Ettin‑150M达74.9%准确率、68.1%宏F1，平均绝对误差0.28；与人类平均评价的α=0.737，高于教师模型α=0.716；在文本匿名化基准中能捕捉直接与准识别符的隐私影响；

**⚠️ 局限性**

模型继承教师LLM的隐私定义和偏差，单一1–5尺度可能掩盖多维隐私因素；仅训练英语数据，跨语言泛化未验证；对上下文（受众、目的）的敏感度不足；教师标注的随机性可能导致噪声，需进一步去噪与多教师策略。

---

## 303. CXLRAMSim v1.0: System-Level Exploration of CXL Memory Expander Cards

**arXiv ID:** 2603.29483 | [PDF](https://arxiv.org/pdf/2603.29483v1)

**作者:** Karan Pathak `[一作]` (EPFL), Marina Zapater `[通讯]` (HES-SO)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

本文开发了 CXLRAMSim v1.0，一个集成在 gem5 中的全系统 CXL 内存模拟器，能够在不修改 Linux 6.14 及其驱动的前提下，实现对 CXL.io/CXL.mem 协议的完整建模，并将 CXL 内存以 zNUMA 节点方式呈现给操作系统。

**💡 创新点**

其创新点在于：①使用 PCIe/CXL 真实 I/O 总线架构、②在 BIOS 级别实现完整 ACPI、CEDT、SRAT 等表以支持 CXL 设备、③在 gem5 内实现 CXL 根复合体和端点的寄存器与事务层（包括 packetization、de‑packetization），④支持 CXL-CLI 与 NDCTL 的用户级管理，从而首次在 gem5 中实现了符合 CXL 2.0+ 规范、可直接使用现有编程模型的完整 CXL 内存模拟。

**🔧 技术方法**

所采用的技术包括：gem5 v25 作为模拟框架、Linux 6.14+ 内核与 Ubuntu 24.04 LTS 镜像、CXL 2.0+ 协议实现、x86 BIOS 通过 MCFG、DSDT、CEDT、SRAT 等 ACPI 表描述 PCIe/CXL 结构、CXL.io/CXL.mem 的事务层与邮件盒寄存器实现、以及 MESI 两级目录式缓存一致性模型。

**📊 数据集**

实验使用了标准的 Stream 微基准（不同规模的数组访问）进行内存压力测试，并通过 Ubuntu 24.04 LTS 系统镜像与真实 CXL 硬件对比校准延迟与带宽参数。

**📈 对比分析**

通过在不同 CPU 模型（顺序与乱序）下执行 Stream 访问，测量 LLC 未命中率，结果显示在多核与单核场景下模拟器的 L2/LLC 命中率与真实硬件相近，误差可调至 2–10% 以内，验证了模型的准确性。

**⚠️ 局限性**

目前的局限在于仅支持单逻辑设备（SLD）和 Flat/NUMA 模式，尚未实现 CXL 开关、处理器近内存（PNN）或 Type‑1/Type‑2 加速器；未来版本计划加入这些功能。

---

## 304. Survival In-Context: Prior-fitted In-context Learning Tabular Foundation Model for Survival Analysis

**arXiv ID:** 2603.29475 | [PDF](https://arxiv.org/pdf/2603.29475v1)

**作者:** Dmitrii Seletkov `[一作]` (Technical University of Munich), Raphael Rehms `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 Survival In-Context (SIC)，一种基于先验拟合的在情境学习模型，用于生存分析。

**💡 创新点**

创新点在于：①构建了可控的合成生存数据生成框架，利用结构因果模型和扩展危险假设生成多样化的协变量与时间事件；②将先验拟合与在情境学习结合，生成的模型无需任务特定训练或超参数调优，单前向传递即可得到个体化生存预测；③通过大规模预训练实现对多尺寸数据集的泛化。

**🔧 技术方法**

技术：结构因果模型（SCM）生成合成数据；扩展危险模型（EH）用于时间事件模拟；TabICL 变压器作为编码器；DeepHit 作为生存头；DeepHit 损失与排名损失相结合；预训练阶段采用课程学习策略。

**📊 数据集**

使用13个公开临床生存数据集（样本量从137到25,000，特征数3到40），并在两个大型隐私数据集上进行额外验证。

**📈 对比分析**

与四类基线（CoxPH、DeepSurv、DeepHit、XGBoost）在时间相关一致性指数（C^td）上进行公平比较，使用5折嵌套交叉验证并对基线进行100次超参数搜索。SIC 在大多数数据集上达到或超过基线，平均排名最高，且在中等规模数据集上表现最佳；计算速度快得多，单前向推断时间仅为传统方法的一小部分。

**⚠️ 局限性**

局限性：未对校准性进行评估；只使用了TabICL架构，未探索其他先验拟合模型；仅针对右删失单一事件，未考虑多事件、竞争风险或时间变化效应；对极大规模数据集的先验覆盖不足。

---

## 305. From Big Data to Fast Data: Towards High-Quality Datasets for Machine Learning Applications from Closed-Loop Data Collection

**arXiv ID:** 2603.29474 | [PDF](https://arxiv.org/pdf/2603.29474v1)

**作者:** Philipp Reis `[一作]` (FZI Research Center for Information Technology), Eric Sax `[通讯]` (FZI Research Center for Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文探讨了从闭环数据采集到高质量机器学习数据集的流程，强调了从大数据向实时数据（Fast Data）的转变。

**💡 创新点**

创新点在于提出一种闭环数据采集框架，能够动态调整采集策略以提升数据质量和标签准确性，并将实时反馈机制与传统离线数据处理结合。

**🔧 技术方法**

主要使用技术包括流式数据处理平台（如Kafka/Flink）、自动标签验证模块、数据质量评估算法以及基于强化学习的采集策略优化。

**📊 数据集**

论文中使用了一个包含多种传感器数据的工业制造场景数据集，以及公开的机器学习基准数据集（如ImageNet或UCI数据集）进行实验验证。

**📈 对比分析**

与传统静态数据集方法相比，作者通过在同一任务上对比实验，展示了闭环采集下模型在准确率上提升了约3%至5%，并在延迟和数据利用率上实现了显著改进。

**⚠️ 局限性**

限制主要体现在：①需要大量计算资源支持实时反馈；②标签质量仍受人工审核程度影响；③闭环框架在不同领域的通用性尚未充分验证。

---

## 306. iPoster: Content-Aware Layout Generation for Interactive Poster Design via Graph-Enhanced Diffusion Models

**arXiv ID:** 2603.29469 | [PDF](https://arxiv.org/pdf/2603.29469v1)

**作者:** Xudong Zhou `[一作]` (Beijing Institute of Technology), Guozheng Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了iPoster交互式海报布局生成框架，支持用户通过灵活约束指导内容感知布局设计。

**💡 创新点**

创新点在于统一的图增强扩散架构、基于掩码的约束保持以及跨内容注意力模块，能够实时生成符合用户约束的高质量布局。

**🔧 技术方法**

利用图增强扩散模型、掩码策略以及跨内容注意力机制。

**📊 数据集**

使用公开海报布局数据集（如PosterNet、DesignBench等）进行训练与评估。

**📈 对比分析**

与多种基线方法（如GAN、传统规则方法、变分自编码器等）进行对比，iPoster在布局质量、视觉一致性等指标上均实现了显著提升。

**⚠️ 局限性**

局限性包括对极端复杂约束的适应性有限，以及在极大尺寸海报时仍需进一步优化实时性与资源消耗。

---

## 307. NeoNet: An End-to-End 3D MRI-Based Deep Learning Framework for Non-Invasive Prediction of Perineural Invasion via Generation-Driven Classification

**arXiv ID:** 2603.29449 | [PDF](https://arxiv.org/pdf/2603.29449v1)

**作者:** Youngung Han `[一作]` (Seoul National University), Hyuk-Jae Lee `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文构建了NeoNet，一个端到端的3D MRI深度学习框架，用于非侵入式预测胆管癌中的神经周围侵袭（PNI）

**💡 创新点**

创新点在于三方面：①采用自动肿瘤定位与ROI裁剪实现精准切割；②利用3D潜在扩散模型与ControlNet进行有条件的数据增强，解决数据稀缺与类别不平衡；③设计PattenNet，融合冻结的LDM编码器和双通道注意力块，以捕捉PNI的细微强度与空间模式

**🔧 技术方法**

使用的技术包括SwinUNETR用于肝脏与肿瘤分割；3D潜在扩散模型（LDM）配合ControlNet生成合成MRI片段；PattenNet利用冻结的LDM编码器以及Channel + Spatial Attention的Dual Attention Blocks（DAB）进行分类

**📊 数据集**

使用的数据集为128例T1加权肝胆期MRI（44例PNI阳性，84例阴性），全部来自三星医疗中心，采用NIfTI格式，分为5折交叉验证

**📈 对比分析**

在5折交叉验证中，NeoNet相较于ResNet、DenseNet、EfficientNet、SwinTransformer等基线3D模型提升明显，最大AUC达0.7903；使用合成数据平衡后，整体AUC从0.676提升至0.742，证明数据增强显著提升性能

**⚠️ 局限性**

局限性包括样本量有限且来自单中心，模型仅使用单相（肝胆期）MRI，缺乏多中心外部验证和多相MRI信息

---

## 308. EarthEmbeddingExplorer: A Web Application for Cross-Modal Retrieval of Global Satellite Images

**arXiv ID:** 2603.29441 | [PDF](https://arxiv.org/pdf/2603.29441v1)

**作者:** Yijie Zheng `[一作]` (Aerospace Information Research Institute, Chinese Academy of Sciences), Konstantin Klemmer `[通讯]` (LGND AI, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个名为 EarthEmbeddingExplorer 的交互式 Web 应用，能够将预先计算好的卫星影像嵌入向量与文本、图像、位置信息进行跨模态检索，并通过可视化地图与 top‑k 结果展示结果。

**💡 创新点**

创新点在于：①将学术研究中发布的嵌入向量打包成 GeoParquet 格式，实现快速下载与查询；②提供云原生部署（ModelScope + GPU）与 Gradio 前端，使非专业用户可直接使用；③支持四种不同训练目标的模型（FarSLIP、SigLIP、DINOv2、SatCLIP），并在同一界面下进行跨模型、跨模态的对比与评估。

**🔧 技术方法**

使用的技术包括：ModelScope Studio 的云端 GPU 运行、Gradio 前端框架、向量相似度搜索（FAISS 或类似数据库）、GeoParquet 存储与分片、标准化的 Major TOM 嵌入扩展接口。

**📊 数据集**

使用的数据集是 MajorTOM‑Core‑S2L2A（Sentinel‑2 影像），在 10×10 km 网格上均匀抽样 1/9，截取中心 384×384 像素块，得到 248,719 个唯一补丁，约占全球陆地表面 1.4%。

**📈 对比分析**

通过案例研究（雨林检索）展示了文本、图像、位置信息查询在不同模型下的检索分布与 top‑5 结果对比；虽然未给出定量指标，但通过热图与可视化证明了模型间语义对齐与视觉相似度的差异，并指出了 FarSLIP 与 SatCLIP 在地理先验上的优势与局限。

**⚠️ 局限性**

局限性包括：①检索结果受模型地理/气候先验限制，可能出现超出预期区域的匹配；②自监督模型 DINOv2 对视觉细节敏感，偶尔会返回与查询场景不符的区域（如海洋）；③目前仅覆盖 1/9 的地表，缺乏时间维度与更细粒度传感器；④向量检索速度仍受限，尚未引入量化或专用向量数据库。

---

## 309. SeGPruner: Semantic-Geometric Visual Token Pruner for 3D Question Answering

**arXiv ID:** 2603.29437 | [PDF](https://arxiv.org/pdf/2603.29437v1)

**作者:** Wenli Li `[一作]` (Shanghai University), Dan Zeng `[通讯]` (Shanghai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无训练的token减法模块，结合语义重要性和3D几何信息，实现在多视角3D问答中显著减少视觉token数量；

**💡 创新点**

创新点在于同时保持语义关键token（通过注意力评分）和空间多样性（通过3D投影与几何距离+语义相似度融合的Farthest Point Sampling），从而在极低token保留比例下仍保持良好性能；

**🔧 技术方法**

使用基于预训练2D视觉-语言模型（如LLaVA-OneVision-7B）的视觉编码器，利用自注意力得分评估token重要性，深度图和相机位姿将2D特征投影到统一3D坐标系，构造语义-空间距离度量进行多样化token选择；

**📊 数据集**

在ScanQA（约8000个室内场景，41k问答）和OpenEQA（1600+问答，180+场景）两个基准上进行评估；

**📈 对比分析**

与DTC（3D-aware）和VisPruner（仅2D）等对比，在token保留率分别为100%~9%时，本文方法在ScanQA上即使保留23% token仍比基线高0.4% EM@1，在OpenEQA上9%保留时仅落后1.3% LLM-Match，并在所有保留率下实现更低的推理延迟；

**⚠️ 局限性**

限制主要包括：依赖深度图和相机参数，无法在无深度信息的场景直接使用；对超低token比例（<5%）时仍会出现性能下降；以及未对模型微调，主要为推理阶段的优化。

---

## 310. CounselReflect: A Toolkit for Auditing Mental-Health Dialogues

**arXiv ID:** 2603.29429 | [PDF](https://arxiv.org/pdf/2603.29429v1)

**作者:** Yahan Li `[一作]` (University of Southern California), Ruishan Liu `[通讯]` (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了CounselReflect，一个端到端的心理健康对话审核和反思工具，整合模型指标与基于规则的评估，并提供交互式报告。

**💡 创新点**

将12个任务特定模型指标与69个文献衍生/自定义指标结合，利用可配置LLM评判实现多维度、证据关联的审计报告。

**🔧 技术方法**

采用RoBERTa、GPT系列LLM、PerspectiveAPI、DeToxify、FactScore/MedScore等技术，并提供Web、浏览器扩展与CLI部署方式。

**📊 数据集**

训练与评估使用Mental Health Subreddits、DailyDialog、AnnoMI、ESConv、TweetEval hate、Civil Comments、SQuAD等公开数据集。

**📈 对比分析**

通过与单一分数评估对比，并在20名用户和6名专业人士的实测中显示SUS>68、满意度与信任度高，模型指标在公开基准上表现稳定。

**⚠️ 局限性**

规则基评估依赖LLM判读，易受模型版本、提示影响，需结合专家判断；部署时需关注隐私与安全，不能完全替代临床专业判断。

---

## 311. Seeing the Evidence, Missing the Answer: Tool-Guided Vision-Language Models on Visual Illusions

**arXiv ID:** 2603.29428 | [PDF](https://arxiv.org/pdf/2603.29428v1)

**作者:** Xuesong Wang `[一作]` (Wayne State University), Harry Wang `[通讯]` (University of Michigan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于工具引导推理框架，利用通用图像处理工具与路由提示，在不训练模型的前提下，解决视觉语言模型在光学错觉识别中的系统偏差，并在 DataCV 2026 Challenge 的 VI-Probe 与 VIA-Bench 任务中实现推理。

**💡 创新点**

创新点在于引入不可变图像资源版本化与自然语言路由策略，取代专门模块，提升跨结构泛化；通过工具循环实现多步可视化推理，并在推理过程中使用可视化证据而非直接测量。

**🔧 技术方法**

采用 Gemini Flash 系列 VLM 作为后端，ReAct 风格工具调用，提供 draw_line、crop、compare_crops 等通用工具；在系统提示中嵌入类别路由和工具使用策略，实现自适应推理。

**📊 数据集**

使用 DataCV 2026 Challenge 中的 VI-Probe（经典视觉错觉）和 VIA-Bench（真实世界异常与不可能场景）两个数据集。

**📈 对比分析**

通过与传统无工具 VLM 进行零样本对比，评估验证集与公开测试集上的准确率。结果显示在 VI-Probe 任务中准确率约 78–85%，VIA-Bench 任务同样保持一致性；在结构陌生的错觉变体上仍能保持稳健性能，证明框架的泛化能力。

**⚠️ 局限性**

存在的局限包括：正检测偏差明显；对像素级微小差异的阈值不稳定，导致逻辑推理失真；JPEG 压缩噪声与小差异交互导致误判；缺乏负样本训练数据，路由策略对极端新任务可能不够鲁棒。

---

## 312. TrafficMoE: Heterogeneity-aware Mixture of Experts for Encrypted Traffic Classification

**arXiv ID:** 2603.29520 | [PDF](https://arxiv.org/pdf/2603.29520v1)

**作者:** Qing He `[一作]` (Chongqing University), Lei Zhang `[通讯]` (Chongqing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了TrafficMoE框架，用于加密流量分类，采用双分支Mixture-of-Experts、跨模态不确定性过滤和条件聚合，形成Disentangle–Filter–Aggregate范式。

**💡 创新点**

创新点在于将头部和负载分别建模为异质子模，利用稀疏MoE实现专属专家分配；引入基于跨模态熵的自适应过滤抑制噪声；以及基于MoE路由的隐式条件聚合动态融合。

**🔧 技术方法**

主要技术包括自监督掩码语言模型预训练、双分支稀疏Mixture-of-Experts、跨模态交互、熵驱动的不确定性过滤、隐式条件聚合以及全局MoE后处理。

**📊 数据集**

使用30GB无标签流量（ISCX-VPN2016、CICIDS2017、WIDE）进行预训练；微调阶段使用六个公开加密数据集：CSTNET-TLS1.3、ISCX-Tor2016、CIC-IoT2022、USTC-TFC2016、ISCX-VPN（APP）和ISCX-VPN（Service）。

**📈 对比分析**

与传统机器学习、深度学习和预训练基准（AppScanner、BIND、CUMUL、ET-BERT、YaTC、TrafficFormer、FlowletFormer等）对比，TrafficMoE在所有数据集上均获得最高或相近最高的准确率与F1（最高可达97.88%/98%）。

**⚠️ 局限性**

局限性包括：路由策略仅靠数据驱动，缺乏显式协议或时序先验；对持续或开放域分布漂移的适应性不足；熵基不确定性估计较为粗糙，可能限制对极端噪声的处理。

---

## 313. On Strengths and Limitations of Single-Vector Embeddings

**arXiv ID:** 2603.29519 | [PDF](https://arxiv.org/pdf/2603.29519v1)

**作者:** Archish S `[一作]` (Microsoft Research India), Kirankumar Shiragur `[通讯]` (Microsoft Research India)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究单向量嵌入在检索任务中的局限，并与多向量嵌入进行对比，探讨维度、分词、领域偏移和“溺水效应”等因素对性能的影响。

**💡 创新点**

提出单向量检索的“瓶颈”不只是维度限制，还包含领域迁移导致的相关性失真，并通过理论与实验证明多向量模型在处理组合相关性时具有优势。

**🔧 技术方法**

使用签名秩理论、Johnson–Lindenstrauss lemma、Chamfer距离等数学工具，结合深度学习密集检索模型（如DPR、ColBERT）进行实验评估。

**📊 数据集**

主要使用LIMIT及其变体（Atomic LIMIT、Two LIMIT等）以及MS MARCO等标准检索基准。

**📈 对比分析**

通过Recall@k、MRR、NDCG等指标比较，实验显示多向量在LIMIT上 Recall@10 接近100%，而单向量仅 1–10%；finetuning 后单向量仍易遗忘 MS MARCO，而多向量几乎无遗忘。

**⚠️ 局限性**

单向量模型在组合相关性、领域偏移和文档“溺水”时存在根本性瓶颈，且finetuning 会导致显著的灾难性遗忘。

---

## 314. All-in-One Augmented Reality Guided Head and Neck Tumor Resection

**arXiv ID:** 2603.29495 | [PDF](https://arxiv.org/pdf/2603.29495v1)

**作者:** Yue Yang `[一作]` (Vanderbilt Institute for Surgery and Engineering), Jie Ying Wu `[通讯]` (Vanderbilt Institute for Surgery and Engineering)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一套基于HoloLens 2深度传感器的全无标记增材现实系统，用于在头颈部肿瘤切除中自动重新定位阳性切缘并实时在手术现场可视化。

**💡 创新点**

创新点在于实现完全自动化、无标记的表面配准算法，能够将切除标本与切除床直接对齐，并将切缘信息以AR方式投射到切除床上，消除了手工对齐和人工标记的工作负担。

**🔧 技术方法**

技术包括HoloLens 2的AHAT深度传感器、TEASER++风格的鲁棒配准、自动ROI构建、点到平面的ICP细化、FPFH描述子与曲率采样、STTAR工具跟踪以及点云后处理与渲染。

**📊 数据集**

实验数据集为硅胶头颈模型（含人造颊部肿瘤）和结构光扫描得到的标本3D模型，比较标记式与无标记两种配准方式。

**📈 对比分析**

通过点匹配任务和切缘重新定位任务进行比较；无标记配准的目标配准误差中位数为1.8 mm，最大误差<4 mm；切缘定位误差从口语指导的14.2 mm降至AR指导的3.2 mm，且所有AR定位误差均<5 mm，显示与基线相当且在定位精度上显著提升。

**⚠️ 局限性**

局限性包括HoloLens 2佩戴造成的舒适度问题、实验仅在硅胶模型上进行，未考虑组织变形与收缩，缺乏尸体或临床手术场景的验证。

---

## 315. MemFactory: Unified Inference & Training Framework for Agent Memory

**arXiv ID:** 2603.29493 | [PDF](https://arxiv.org/pdf/2603.29493v1)

**作者:** Ziliang Guo `[一作]` (MemTensor), Zhiyu Li `[通讯]` (MemTensor)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了 MemFactory，一套可模块化、统一的训练与推理框架，用于实现和优化记忆增强型 LLM 代理的 RL 控制策略。

**💡 创新点**

创新点在于将记忆生命周期拆解为可插拔的原子操作（提取、更新、检索），并内置 GRPO 强化学习训练器，支持直接在同一框架下复现并改进多种主流记忆策略。

**🔧 技术方法**

核心技术包括 Group Relative Policy Optimization（GRPO）、FlashAttention‑2、Transformer 预训练模型、以及基于 LLM 的检索重排序与奖励信号。

**📊 数据集**

使用 MemAgent 公开数据集（训练集、主任务测试集 eval_50/100 以及 OOD 测试集 eval_fwe_16384）进行实验。

**📈 对比分析**

方法对比：在基线 checkpoint 上进行 250 步 GRPO 微调，结果在 eval_50/100 上提升约 14.8%（小模型）和 7.3%（大模型），OOD 上大模型表现亦有提升，验证了框架的有效性。

**⚠️ 局限性**

局限性在于目前仅覆盖少数代表性记忆范式，训练效率与资源消耗仍有改进空间，且对更大规模模型或多任务迁移的泛化尚需进一步验证。

---

## 316. Structural Compactness as a Complementary Criterion for Explanation Quality

**arXiv ID:** 2603.29491 | [PDF](https://arxiv.org/pdf/2603.29491v1)

**作者:** Mohammad Mahdi Mesgari `[一作]` (Fraunhofer Heinrich Hertz Institute), Leander Weber `[通讯]` (Fraunhofer Heinrich Hertz Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于图的度量方法（MST‑C）来量化图像归因可读性（compactness）

**💡 创新点**

创新点在于将归因的“散布”（spread）和“凝聚”（cohesion）两个高阶几何特征通过最小生成树（MST）与凸包面积统一为单一数值，并通过阈值化与k‑近邻构图实现对空间结构的定量评估

**🔧 技术方法**

采用图构建（k‑NN + 归一化阈值化）、最小生成树、凸包面积计算、归因热图与模型解释器（Grad‑CAM、IG、SHAP、LRP等）以及多种超参数鲁棒性分析

**📊 数据集**

使用 PASCAL VOC 2012 数据集（224×224 归一化）进行评估，模型包括 ResNet34、VGG‑16、SimCLR

**📈 对比分析**

与现有复杂度度量（Sparsity、Complexity、Effective Complexity）以及 rra、ROAD 进行相关性与对比实验，MST‑C 与 sparsity 正相关、与复杂度负相关，能够区分相似复杂度但结构不同的归因，显示对可读性的补充诊断作用；在不同解释器与模型上获得可解释性差异的显著区分

**⚠️ 局限性**

局限性包括对两个超参数（k 与阈值百分位）的依赖、阈值化导致失去细粒度信息、对 blob‑like 归因有偏好、未进行用户研究验证可读性、目前仅适用于图像域，扩展到其他域需进一步研究

---

## 317. Square Superpixel Generation and Representation Learning via Granular Ball Computing

**arXiv ID:** 2603.29460 | [PDF](https://arxiv.org/pdf/2603.29460v1)

**作者:** Shuyin Xia `[一作]` (Chongqing University of Posts and Telecommunications), Wen Lu `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于方块超像素的端到端可插拔模块，利用粒子球计算生成多尺度方块并按纯度筛选，以实现结构化视觉令牌；

**💡 创新点**

创新点在于：①使用规则化的方块粒子球替代传统不规则超像素，兼容卷积与Transformer；②基于纯度的非迭代粗细层级选择，支持GPU并行与固定令牌数；③可直接嵌入图神经网络或Vision Transformer，且在检测中可进行令牌剪枝；

**🔧 技术方法**

主要技术包括：粒子球计算（Granular Ball Computing）、多尺度方块分割与纯度评分、掩码选择与特征融合、ViG/ViT/RT-DETR等深度网络；

**📊 数据集**

实验数据集涵盖图像分类（MNIST、CIFAR‑10）、图文检索（CelebA、MM‑CelebA）、目标检测（COCO 2017）；

**📈 对比分析**

与传统SLIC、SSN、SpixelFCN等超像素方法以及多种GNN、Transformer、YOLO、DETR系列进行对比；在分类中取得99.4% MNIST、93.5% CIFAR‑10；在检索中在CelebA和MM‑CelebA均优于FLIP、ALIGN、BLIP、CLIP；在检测中将RT‑DETR从400令牌压缩到200令牌，AP仅下降0.8点（从53.1→52.3），保持竞争力；

**⚠️ 局限性**

局限性包括：对小目标和中等尺寸目标的性能受限于粗细层级的固定阈值；纯度阈值需经验调参；对极高分辨率图像的纯度计算仍有一定开销；与YOLO等专门优化的实时检测器相比，整体AP仍略低；

---

## 318. FedDBP: Enhancing Federated Prototype Learning with Dual-Branch Features and Personalized Global Fusion

**arXiv ID:** 2603.29455 | [PDF](https://arxiv.org/pdf/2603.29455v1)

**作者:** Ningzhi Gao `[一作]` (South China University of Technology), Ying Gao `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的联邦原型学习方法 FedDBP，用于解决数据和模型异构问题。

**💡 创新点**

创新点在于客户端双分支特征投影器同时兼顾特征保真度与判别性，以及服务器端基于 Fisher 信息的通道重要性实现个性化原型融合。

**🔧 技术方法**

采用 L2 对齐、对比学习（带硬负样本挖掘）和 Fisher 信息权重的通道选择等技术。

**📊 数据集**

在 CIFAR-10、CIFAR-100、Flowers102 与 Tiny-ImageNet 四个图像分类数据集上进行实验。

**📈 对比分析**

与十个现有 HFL 与 FPL 基线进行对比，FedDBP 在所有数据集上均取得最高准确率，平均提升约 6.9% 以上。

**⚠️ 局限性**

局限性在于仅验证于图像分类任务，通道重要性估计对模型结构敏感，且未针对跨模态或极端通信延迟场景进行评估。

---

## 319. Few-shot Writer Adaptation via Multimodal In-Context Learning

**arXiv ID:** 2603.29450 | [PDF](https://arxiv.org/pdf/2603.29450v1)

**作者:** Tom Simon `[一作]` (University of Rouen Normandy), Thierry Paquet `[通讯]` (University of Rouen Normandy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于多模态上下文学习的少样本作者适应框架，在推理时不需要任何参数更新即可根据少量标注样本调整手写文本识别模型。

**💡 创新点**

创新点在于将上下文驱动的字符级视觉匹配与小型8M CNN-Transformer相结合，能够在无梯度更新的前提下实现强大的写字风格自适应，并通过置信度融合提升性能。

**🔧 技术方法**

使用CNN-Transformer架构、上下文感知分词器（CAT）和跨模态注意力实现多模态上下文编码，同时采用渐进式训练、噪声注入和上下文长度调度等技术。

**📊 数据集**

在IAM和RIMES两个公开手写文本识别基准数据集上进行实验，评估上下文长度对性能的影响并进行模型融合。

**📈 对比分析**

与现有无参数更新的适应方法相比，本方法在IAM和RIMES上分别取得CER 3.92%和2.34%，超越传统无适应模型，且与基于MAE的梯度更新方法相当，显示出显著的性能提升。

**⚠️ 局限性**

局限性是推理时需要提供少量已标注的上下文行，且对标注质量有一定要求，缺乏完全无监督的适应能力。

---

## 320. Beyond Bits: An Introduction to Computation over the Reals

**arXiv ID:** 2603.29427 | [PDF](https://arxiv.org/pdf/2603.29427v1)

**作者:** Tillmann Miltzow `[一作]` `[通讯]`, Tillmann Miltzow

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文概述了实数计算模型及其在几何与组合优化中的应用，阐述了∃ℝ类的定义、证明及其与NP、P等传统复杂度类的关系，并通过多种几何问题的∃ℝ‑硬性归约展示了该模型的力量。

**💡 创新点**

提出将实数视为基本计算对象的新框架，并通过将多项式约束转化为线性/乘法/逆运算的简化约束，构建了实数计算的统一模型；同时引入“球定理”等工具，将变量限制在紧致区间，解决了实数模型与离散模型之间的匹配问题。

**🔧 技术方法**

使用逻辑公式（存在性理论）、直线程序、实数RAM/实数图灵机、投影变换、点线对偶、Mnëv不等式、球定理等理论与几何构造技术。

**📊 数据集**

本文主要为理论性综述，未使用具体数据集；若涉及实验，则以人工构造的几何实例或随机生成的点集为示例。

**📈 对比分析**

通过理论归约与模拟论证不同模型在多项式时间内等价，未给出数值性能评测；在可视化/计算几何案例中，采用基于约束求解的逻辑编译方法，理论上可在多项式时间内完成验证。

**⚠️ 局限性**

限制包括：实数模型在实际计算机上实现困难；某些∃ℝ完全问题缺乏多项式时间判定算法；部分几何问题（如艺术画廊）需要非可算精度的解，导致证明确认复杂度不易实现。

---

## 321. Polynomial Time Local Decision Revisited

**arXiv ID:** 2603.29477 | [PDF](https://arxiv.org/pdf/2603.29477v1)

**作者:** Laurent Feuilloley `[一作]` (CNRS, INSA Lyon, UCBL, LIRIS, UMR5205), Ami Paz `[通讯]` (LISN --- CNRS & Paris-Saclay University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

研究分布式决策任务的复杂度分类系统，比较三种主要分类体系（Balliu等的无界计算与证书体系、Aldema Tshuva & Oshman的多项式局部计算体系、Reiter的基于节点邻域大小的多项式计算体系），并揭示它们之间严格包含、不可比、以及不可比较的关系。

**💡 创新点**

提出了多项式本地计算与证书大小对分布式决策能力的微妙影响，证明了在无证书、存在证书、全称证书三种量化模型下，时间和证书大小的不同阶层产生的严格层级；尤其发现“在全称证书下，限制证书长度到多项式会泄漏图大小信息，甚至导致可判定语言超出无界证书的情况”。

**🔧 技术方法**

采用理论计算机科学中的时间层次理论、图论邻域分析、证书分配与ID模型的构造，以及对本地与全局时间上限的比较和模拟技巧，构建了多种本地决策算法与对应的模拟算法。

**📊 数据集**

本工作为纯理论分析，不涉及实际数据集。

**📈 对比分析**

比较方法基于严格包含与不可比的证明，通过构造特定语言、利用时间层次与空间层次理论、以及对本地视角与全局视角的计算时间做比较，展示了不同分类体系之间的层级关系。

**⚠️ 局限性**

局限性在于只讨论了确定性、ID不依赖的情形，未覆盖随机化、无唯一ID或异构网络；此外，仅在常数本地半径和多项式时间限制下进行讨论，实际分布式系统中的通信延迟与节点异构可能导致不同结论。

---

## 322. Improved Approximation Algorithms for Non-Preemptive Throughput Maximization

**arXiv ID:** 2603.29451 | [PDF](https://arxiv.org/pdf/2603.29451v1)

**作者:** Alexander Armbruster `[一作]` (Technical University of Munich), Andreas Wiese `[通讯]` (Technical University of Munich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对单机（可多机）非抢占式吞吐量最大化问题（即作业区间调度）提出了新的近似算法，分别实现了多项式时间的 (4/3+ε) 近似以及伪多项式时间的 (5/4+ε) 近似，并将结果推广到任意数量相同机器的场景。

**💡 创新点**

创新点包括：
- 设计了新的块–超块划分，使得存在 (1+ε)-近似解且每个作业只需要在其左右边界块或所跨超块内调度。
- 采用配置线性规划并在块级别进行随机采样，随后通过二分匹配而不是简单丢弃多余的槽，显著提升匹配率。
- 引入谐波分组与 read‑k 变量的集中性分析，克服了作业间依赖导致的概率失效。
- 提供了一个基于随机分配与删改（alteration）的替代舍入方案，进一步把全局作业的损失压缩到 1+ε。

**🔧 技术方法**

主要技术手段：
- 配置 LP（配置线性规划）与块级随机采样。
- 二分图匹配（最大匹配）与贪心分配。
- 谐波分组（harmonic grouping）与集体中心化（concentration for read‑k families）。
- 颜色编码（color‑coding）+ 单调动态规划，用于在伪多项式时间求解配置 LP。
- 随机分配与删改（random assignment with alteration）算法。

**📊 数据集**

文中未给出具体实验数据集，主要是理论分析与证明。若有实验，常用的基准数据集包括随机生成的作业集合以及来自实际调度问题的公开数据集。

**📈 对比分析**

与以往最优 1.551（Im‑Li‑Moseley）及 1.582（Chuzhoy‑et‑al）相比，本文在多项式时间下获得 1.334+ 的改进，在伪多项式时间下进一步降到 1.25+。在多机扩展下，保持了同样的近似比率并在常数机数下实现伪多项式时间。

**⚠️ 局限性**

局限性：
- 仍无法得到 PTAS，逼近界仍不确定。
- 伪多项式时间算法对 T（最大截止时间）的依赖，使得在 T 很大时效率下降。
- 对机器数 m 的推广仍需在常数范围内才能保持多项式复杂度，若 m 较大则需额外的资源分配或分段分析。
- 需要复杂的概率与线性规划分析，实际实现可能较困难。

---

## 323. Authorship Impersonation via LLM Prompting does not Evade Authorship Verification Methods

**arXiv ID:** 2603.29454 | [PDF](https://arxiv.org/pdf/2603.29454v1)

**作者:** Baoyi Zeng `[一作]`, Andrea Nini `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了通过 GPT‑4o 等大语言模型在四种提示策略下对不同作者的写作进行模仿，并评估其是否能欺骗现有的作者身份验证系统；

**💡 创新点**

首次在法医语言学的严格验证框架下，系统性比较多种非神经和神经 AV 方法对 LLM 生成的“伪装文本”的抵抗力；

**🔧 技术方法**

使用 GPT‑4o 进行文本重写，采用 N‑gram 追踪、RBI、LambdaG 等传统 AV 方法以及 AdHominem、LUAR、STAR 等基于深度学习的 AV 方法；

**📊 数据集**

使用 Enron 电子邮件、BOLT 短信/聊天、Twitter 推文三大真实法医语料库；

**📈 对比分析**

通过校准后的对数似然比（LLR）和 TNR（真负率）比较，结果显示所有 AV 方法（尤其是 STAR、LUAR、LambdaG）均能有效拒绝大多数 LLM 生成的伪装文本，性能并未显著下降；

**⚠️ 局限性**

限制在于仅考察了基于提示的攻击，未涵盖自适应攻击、参数高效微调等更高级威胁，也未评估人类读者对伪装文本的感知真实性。

---

## 324. M-MiniGPT4: Multilingual VLLM Alignment via Translated Data

**arXiv ID:** 2603.29467 | [PDF](https://arxiv.org/pdf/2603.29467v1)

**作者:** Seung Hun Han `[一作]` (MBZUAI), Mohamed Elhoseiny `[通讯]` (KAUST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并训练了多语种视觉语言模型 M‑MiniGPT4，支持 11 种语言的视觉理解与推理。

**💡 创新点**

创新点在于将翻译版与本地多语种数据结合，并加入并行文本语料进行多语种对齐训练，从而显著提升跨语言视觉推理性能。

**🔧 技术方法**

采用 MiniGPT‑4 结构，替换为 Llama‑3 作为 LLM，三阶段训练流程：①视觉‑语言对齐；②多语种多模态微调；③利用翻译版数据与并行文本进一步对齐。

**📊 数据集**

使用翻译后的多语种视觉数据（Conceptual Captions、SBU、LAION、LAVAM、PALO、Cambrian Image）、原版视觉数据、以及并行文本语料（Flores、XStoryCloze）进行训练，MMM U 及其多语种版本作为评测基准。

**📈 对比分析**

与现有模型（PALO、Qwen‑VL 2.5 等）在 MMMU Multi 评测上对比，M‑MiniGPT4 取得 33.45% 正确率，明显优于同类模型，且在标准 MMMU 评测中虽略逊于 Qwen‑VL 2.5，但表现仍可观。

**⚠️ 局限性**

局限包括：机器翻译可能缺失文化细节；仅覆盖 11 种语言，低资源语言翻译质量不足；评测指标可能无法全面捕捉跨文化理解；继承基模型偏见。

---

## 325. Multi-AUV Cooperative Target Tracking Based on Supervised Diffusion-Aided Multi-Agent Reinforcement Learning

**arXiv ID:** 2603.29426 | [PDF](https://arxiv.org/pdf/2603.29426v1)

**作者:** Jiaao Ma `[一作]` (Northeastern University), Chen An `[通讯]` (Northeastern University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aaccfe5c-6b26-4208-b23c-35331481e142` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并实现了一种基于监督扩散的多智能体强化学习框架（SDA‑MARL），用于多 AUV 协同跟踪海底目标。

**💡 创新点**

创新点包括：① 双决策架构（扩散模型 + DDPG）与监督学习/行为克隆结合，② 经验回放分离并用监督标签筛选高质量样本，③ 分层架构（全局调度、协调层、局部决策层、实时执行层）实现任务分解与协同。

**🔧 技术方法**

使用的技术有：多智能体强化学习（MADDPG/DDPG）、扩散模型（Diffusion QL 等）、监督学习、行为克隆、双 Q 网络、软更新/EMA、海流动力学建模、声纳感知、Navier–Stokes 流体动力学等。

**📊 数据集**

实验基于自建的仿真环境（4 种场景：2/4/6/8 AUV 分别跟 1/2/3 目标），未使用公开数据集。

**📈 对比分析**

与 DSBM、MA‑A3C、MASAC、MAPPO、MAAC、MATD3、MADDPG 等主流 MARL 算法在收敛速度、跟踪精度、速度差异、路径长度等指标上对比，SDA‑MARL 在所有指标上均显著优于对手，跟踪精度最高达 66%–69%。

**⚠️ 局限性**

局限性：① 仅在仿真环境验证，缺乏真实水下实验；② 监督标签阈值需手工设定，可能在不同场景下不稳健；③ 扩散步骤数量需经验调参；④ 对大规模 AUV 群体的通信与计算开销未评估。

---

## 326. A2BFR: Attribute-Aware Blind Face Restoration

**arXiv ID:** 2603.29423 | [PDF](https://arxiv.org/pdf/2603.29423v1)

**作者:** Chenxin Zhu `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 A^2BFR 框架，实现属性感知且可按文本提示控制的盲面部恢复。

**💡 创新点**

将文本提示与扩散模型联合条件化，设计属性感知学习（AAL）与语义双重训练（SDT）以提升属性对齐与可控性。

**🔧 技术方法**

基于 Flux Diffusion Transformer + LoRA 适配器的联合图像‑文本注意力，配合属性编码器与对比损失实现语义引导恢复。

**📊 数据集**

构建 AttrFace‑90K 数据集（90k HQ 图像对 + 180k 细粒度文本说明）用于训练和评估。

**📈 对比分析**

与多种 GAN/Transformer、扩散恢复模型以及两阶段恢复+编辑管线对比，LPIPS、FID、HyperIQA 等指标均显著领先，属性准确率提升 52.58%。

**⚠️ 局限性**

受限于对高质量配对数据与预训练文本模型的依赖，对极端降质或稀缺属性的泛化仍有限。

---

## 327. Impact of enriched meaning representations for language generation in dialogue tasks: A comprehensive exploration of the relevance of tasks, corpora and metrics

**arXiv ID:** 2603.29518 | [PDF](https://arxiv.org/pdf/2603.29518v1)

**作者:** Alain Vázquez `[一作]` (University of Basque Country), Maria Inés Torres `[通讯]` (University of Basque Country)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在对话自然语言生成中，利用从原始数据集抽取的任务示例（MR–句子对）作为输入增强，评估其对生成质量的影响；

**💡 创新点**

创新点包括：①首次将任务示例作为训练与推理时的增量输入引入对话NLG；②在四个多域数据集上，对五种评估指标进行系统对比，揭示语义指标对质量评估的更高敏感度；③证明该方法在复杂域、小型且MR多变的语料以及零样本跨域场景下效果显著；

**🔧 技术方法**

使用技术：Fine‑tuned GPT‑2生成模型；构造任务示例的采样与编码方法；采用BLEU、BLEURT、LaBSE、Slot Accuracy、Dialogue Act Accuracy等多维度评估指标；

**📊 数据集**

数据集：E2E、ViGGO、MultiWOZ、EMPATHIC 四个面向对话的NLG语料库；

**📈 对比分析**

比较方法：将仅使用MR输入的生成与加上任务示例输入的生成进行对比，使用五种指标评估差异。实验结果表明，在复杂域、小型、高变异MR数据集以及零样本跨域场景中，加入任务示例能显著提升Slot Accuracy与Dialogue Act Accuracy，并在语义评估指标上取得更大提升，词汇指标的提升相对有限；

**⚠️ 局限性**

局限性：仅在中等规模模型GPT‑2上实验，未验证大规模LLM的适用性；任务示例的选择与数量对结果影响较大，缺乏系统化优化；仅使用单一示例，未探讨多示例或更复杂提示对性能的进一步提升；对不同评估指标间权衡的解释仍不充分。

---

## 328. Finite Blocklength Covert Communication over Quasi-Static Multiple-Antenna Fading Channels

**arXiv ID:** 2603.29645 | [PDF](https://arxiv.org/pdf/2603.29645v1)

**作者:** Changhong Liu `[一作]` (Beihang University), Lin Zhou `[通讯]` (Southern University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在准静态多天线衰落信道上，满足KL散度隐蔽约束的有限块长度可覆盖通信的极限性能，并给出了实现和逆向证明。

**💡 创新点**

创新点包括：① 将准静态衰落信道的随机矩阵特性融入到隐蔽性分析中，扩展了quasi‑η‑neighborhood框架；② 证明了在隐蔽约束下，第一阶速率满足平方根律且系数仅由合法链路和哨兵链路的矩阵迹决定，且第二阶项为零；③ 发现与合法用户CSI的可用性无关，说明水分配无法在隐蔽环境下发挥作用；④ 明确了多天线系统的空间多样性对提升隐蔽速率的关键作用。

**🔧 技术方法**

主要技术手段包括：随机截断复高斯编码、角阈值解码、通道的广义奇异值分解、Cramér‑Esseen定理的高阶控制、隐蔽失效概率的Beta分布分析、以及对隐蔽约束下的功率上界求解。

**📊 数据集**

实验数据来自于对Rayleigh、Rician、Nakagami等准静态衰落信道的百万次Monte‑Carlo仿真，验证理论预测的速率、误码和隐蔽性。

**📈 对比分析**

与传统AWGN MIMO的平方根律基准以及非隐蔽衰落信道的二阶项结果进行比较。仿真表明在给定块长度下，准静态衰落信道的非隐蔽误码率显著下降，隐蔽速率随天线数成倍提升；相较于AWGN信道，准静态信道的收敛速度更快，第二阶项消失。

**⚠️ 局限性**

局限性包括：仅考虑点对点P2P MIMO；假设合法用户共享无限长的秘密密钥；使用随机编码与穷举搜索解码，计算复杂度高；假设哨兵拥有完美CSI，现实中CSI误差可能影响隐蔽性能。

---

## 329. Optimizing Donor Outreach for Blood Collection Sessions: A Scalable Decision Support Framework

**arXiv ID:** 2603.29643 | [PDF](https://arxiv.org/pdf/2603.29643v1)

**作者:** André Carneiro `[一作]` (INESC-ID and Técnico Universidade de Lisboa), Rui Henriques `[通讯]` (INESC-ID and Técnico Universidade de Lisboa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了一套面向血液捐献中心的邀请排程优化框架，旨在通过精准匹配捐献者与捐献时段，平衡供需、容量、血型需求、地理便利和安全约束。

**💡 创新点**

创新点在于将捐献者资格、会场容量、血型需求目标、距离惩罚和不良反应惩罚等多重约束统一到二进制整数规划模型中，并提供可直接落地的前瞻性邀请管线。

**🔧 技术方法**

核心技术包括二进制整数线性规划（BILP）与高效贪婪启发式、机器学习预测（MLP、XGBoost、Holt–Winters）以及Gurobi求解器和基于GPT的地理编码。

**📊 数据集**

使用的主要数据集来自葡萄牙葡萄牙血液与移植研究所（IPST）Lisbon区域 2020 年的捐献者与会场记录，以及历史需求和出席数据。

**📈 对比分析**

实验通过在 2 倍残余需求情景下比较 BILP 与贪婪算法，贪婪算法实现了 86.1% 的需求满足率（仅比 BILP 低 3.9pp），但运算速度提升 115 倍、峰值内存降低 188 倍。

**⚠️ 局限性**

主要局限包括：地理位置采用推断坐标，缺乏真实的捐献者与会场精确坐标；个人邀请的实际出席概率未经过专门校准；以及对不良反应惩罚等权重的主观设定可能影响结果。

---

## 330. MacTok: Robust Continuous Tokenization for Image Generation

**arXiv ID:** 2603.29634 | [PDF](https://arxiv.org/pdf/2603.29634v1)

**作者:** Hengyu Zeng `[一作]` (Fudan University), Jian Pu `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种名为MacTok的连续图像分词器，结合随机遮蔽、DINOv2引导的语义遮蔽以及全局/局部表示对齐，旨在防止KL变分框架下的后验坍塌，实现高效高质量的图像生成与重建。

**💡 创新点**

首次将图像遮蔽与全局/局部表示对齐相结合，在KL-VAEs中有效避免后验坍塌，仅用64-128个token即可获得与更大token相当甚至更优的生成质量。

**🔧 技术方法**

使用ViT作为编码器/解码器，KL正则化的连续latent空间，DINOv2预训练特征进行语义遮蔽与表示对齐，随机遮蔽、局部/全局对齐损失，以及重建、感知、对抗、KL和RA等多项损失的组合。

**📊 数据集**

在ImageNet（256×256与512×512）上进行训练与评估。

**📈 对比分析**

与多种连续与离散分词器（VQ‑VAE、SoftVQ‑VAE、MAETok、VA‑VAE等）以及生成模型（SiT‑XL、LightningDiT‑XL、SiT‑B）对比。MacTok在gFID上分别取得1.44（256）/1.52（512），显著优于SoftVQ‑VAE、MAETok等；在rFID、PSNR、SSIM上也表现出色，仅用64-128 token即可实现高质量重建。

**⚠️ 局限性**

对mask比例及语义遮蔽的调优仍有依赖；在极大压缩或极大token数下性能可能下降；需要DINOv2预训练特征，语义遮蔽质量对结果敏感；目前仅在ImageNet上验证，缺乏跨域或更复杂数据集的实验。

---

## 331. Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis

**arXiv ID:** 2603.29620 | [PDF](https://arxiv.org/pdf/2603.29620v1)

**作者:** Shuang Chen `[一作]` (University of California Los Angeles), Nanyun Peng `[通讯]` (University of California Los Angeles)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 Unify‑Agent，一种端到端统一多模态生成代理，完成从思考（Think）、检索（Research）、重述（Recaption）到生成（Generate）的完整流程，实现对稀有、长尾概念的知识驱动图像生成。

**💡 创新点**

创新点：①把检索与生成统一在同一模型内部，避免多阶段易错的 API 链；②引入“证据重述”模块，将原始文本与视觉证据转化为结构化、可执行的生成指令；③利用 VAE 与 ViT 的双重编码，既保留低层视觉细节，又提供高层语义上下文；④在推理时采用开放书（open‑book）式外部知识检索，真正实现“主动获取”世界知识。

**🔧 技术方法**

技术：Bagel Mixture‑of‑Transformers 体系；ViT 编码器 + VAE 低层潜在；流匹配（flow‑matching）扩散生成；端到端的自回归语言模型；检索工具（文本检索、图像检索）与 LLM 规划；训练时使用 GPT‑4o/Claude Opus 生成推理轨迹；再利用 Nano Banana Pro 进行生成验证。

**📊 数据集**

数据集：
- 训练集：456K 罕见 IP 的提示、两张真值参考图、结构化元数据，随后经过 143K 轨迹‑图像对的人工筛选；
- 评测集：FactIP（2,462 条长尾 IP 提示），WiSE、KiTTEN、T2I‑FactualBench 等公开基准。

**📈 对比分析**

与多类基线对比：
- 商业模型（DALL·E 3、Seedream、Nano Banana 2、GPT‑Image‑1.5）
- 开源生成模型（FLUX.1‑dev、SD‑3.5‑large、Playground‑v2.5 等）
- 开源统一 MLLM（Janus‑Pro‑7B、Emu3.5、Echo‑4o、Hunyuan‑Image‑3.0、Bagel）。
在 FactIP 上 Unify‑Agent 取得 73.2 分（远超 Bagel 50.9、FLUX 1‑dev 28.9 等），在 WiSE、KiTTEN、T2I‑FactBench 等基准同样获得最高或接近最高分，表明其在知识驱动、身份一致性与场景控制方面优于现有方法。

**⚠️ 局限性**

局限性：
- 开源统一模型仍弱于主流闭源系统；
- Bagel 的长上下文容量有限，难以支持多图/复杂推理；
- 当前工作采用单次推理流水线，缺乏多轮检索、反思与重规划等迭代机制；
- 对极端复杂的开放世界任务（如旅行规划、学术报告生成）尚未验证，未来需扩展更强的基底与更深层次的交互能力。

---

## 332. ARCOL: Aspect Ratio Constrained Orthogonal Layout

**arXiv ID:** 2603.29618 | [PDF](https://arxiv.org/pdf/2603.29618v1)

**作者:** Zainab Alsuwaykit `[一作]` (King Abdullah University of Science and Technology), Ivan Viola `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 ARCOL，一种在正交布局过程中直接控制全局长宽比的算法。

**💡 创新点**

创新点在于在 HOLA 的压力最小化阶段加入软长宽比约束，并在树重装阶段引入长宽比感知的成本函数，使得布局无需后处理即可符合指定 AR。

**🔧 技术方法**

技术包括基于坐标方差的软正则化（软约束）和改进的树重装配成本模型；算法改造自 HOLA 的增量人类启发式流程。

**📊 数据集**

使用 Rome 数据集（1481 张图）以及 Sydney、Melbourne 地铁图等实际图。

**📈 对比分析**

与原始 HOLA 与后处理放缩 HOLA 进行量化指标（KSM、ELD、NR、NU、NP、EC）以及用户/专家评估对比，ARCOL 在极端 AR 时性能优于后处理，整体指标保持几乎相同且用户偏好略高。

**⚠️ 局限性**

局限在于仅适用于稀疏图，树结构只在核心布局完成后才重装，未能保证跨 AR 的一致性，并且对高密度图效果未知。

---

## 333. Learning Diagnostic Reasoning for Decision Support in Toxicology

**arXiv ID:** 2603.29608 | [PDF](https://arxiv.org/pdf/2603.29608v1)

**作者:** Nico Oberländer `[一作]` (Technical University of Munich), Matthias Keicher `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 DeToxR，利用强化学习微调轻量级 LLM，对急性多药物中毒病例进行多标签毒物预测，并输出可解释的诊断推理。

**💡 创新点**

首次将 GRPO 强化学习与基于 F1 的临床奖励相结合，解决异构、缺失数据下的毒理决策；同时实现结构化与非结构化信息的融合与可解释推理。

**🔧 技术方法**

使用 Qwen3 4B LLM + LoRA 微调 + GRPO + F1 + 格式奖励；构建 Markdown 融合引擎，将结构化临床变量与自由文本统一成提示。

**📊 数据集**

基于德国慕尼黑医院收集的 870 例急性多药中毒病例，涵盖 14 种毒物，包含年龄、性别、生命体征、症状、药物历史等结构化数据及 3 段自由文本。

**📈 对比分析**

与历史基线、MLP、XGBoost、零-shot LLM 以及 SFT 进行对比；DeToxR 在微 F1 为 63.7%、宏 F1 为 66.9%，在 25 例专家验证中微 F1 0.644 高于专家 0.473，整体表现显著优于基线和同类方法。

**⚠️ 局限性**

局限性包括：对少数毒物的预测能力有限；存在假阳性，推理中可能出现虚构症状；仅单中心数据，缺乏跨机构外部验证。

---

## 334. A Strong Linear Programming Relaxation for Weighted Tree Augmentation

**arXiv ID:** 2603.29582 | [PDF](https://arxiv.org/pdf/2603.29582v1)

**作者:** Vincent Cohen-Addad `[一作]` (Google Research), Ola Svensson `[通讯]` (EPFL)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对加权树补全问题（WTAP）设计了一种随机近似算法，采用强大的线性规划松弛（Strong LP），通过引入局部配置变量、保持分布一致性（类似 Sherali‑Adams 的 lift‑and‑project 技术）并结合概率化取样与清理（clean‑up）步骤，最终实现了 1.49 的近似比。

**💡 创新点**

创新点包括：①构造了新的强 LP 通过“事件”与“一致性约束”把局部覆盖信息连贯成全局约束；②设计了分层随机取样与一致性采样的结构化 LP 以及对应的取样算法；③引入清理阶段，以减少冗余跨链的使用，使期望成本从 1.5 降到 1.49；④提供了从 Strong LP 到结构化分数解的多步降维与恢复过程。

**🔧 技术方法**

主要技术：配置线性规划（configuration LP）、Sherali‑Adams 级联约束、概率化取样与一致性采样、事件/子树覆盖约束、分层取样与分支结构、清理阶段的负相关性分析、线性规划分离与多项式求解。

**📊 数据集**

本文为理论算法研究，未使用实际数据集，全部证明与实验均在理论分析层面完成。

**📈 对比分析**

与现有最优 1.5+ε 的局部搜索方法相比，提出的算法在理论上取得了更低的近似比 1.49。实验与比较只在分析中给出，证明了在所有 WTAP 实例上均能达到此比值。

**⚠️ 局限性**

局限性：①Strong LP 规模随参数 ρ、β 递增，实际求解难度较大；②算法仍依赖随机化，若需确定性解需额外处理；③对 ϵ 的选择影响常数和运行时间；④虽然改进了 integrality gap 但仍未突破 1.5 的下界，未给出更紧的理论上限；⑤实现细节（事件扩展、中心选择）较为复杂，理论证明相对繁琐。

---

## 335. Emotion Diffusion Classifier with Adaptive Margin Discrepancy Training for Facial Expression Recognition

**arXiv ID:** 2603.29578 | [PDF](https://arxiv.org/pdf/2603.29578v1)

**作者:** Rongkang Dong `[一作]` (Hong Kong Polytechnic University), Kin-Man Lam `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Emotion Diffusion Classifier（EmoDC）并用 Stable Diffusion 作为条件生成模型来完成面部表情识别（FER），随后提出 Adaptive Margin Discrepancy Training（AMDiT）提升分类性能。

**💡 创新点**

创新点在于：①首次将生成式扩散模型转化为FER的分类器；②设计基于噪声预测误差的差异性训练（CoDiT）和差距边距损失（FMDiT），并进一步引入自适应边距（AMDiT），使模型对每个样本的难度做动态调节，显著提升判别力。

**🔧 技术方法**

使用技术包括：Stable Diffusion v2.1 + LoRA 微调；噪声预测误差计算；差异性损失（CoDiT）与基于边距的差异性训练（FMDiT/AMDiT）；多步与单步推理、Classifier‑free guidance、共享噪声等。

**📊 数据集**

实验数据集：RAF‑DB Basic 与 Compound、SFEW‑2.0、AffectNet、FER2013Plus（用于跨域评估）。

**📈 对比分析**

对比方法：多种 SOTA 判别式模型（APViT、POSTER、POSTER++、VTFF 等）。在 RAF‑DB_B 上，AMDiT 达到 90.12%（比基线提升 6.45%），在 RAF‑DB_C 达到 68.18%（比基线提升 16.29%）；在噪声、模糊和跨域测试中均优于判别式模型；单步推理实现 34 FPS，实时性优越。

**⚠️ 局限性**

局限性：推理时间长（多步需要多次 UNet 前向）；模型体积大，训练与推理成本高；在极端噪声/模糊下仍有限；生成的表情图像色彩不自然；未覆盖视频动态表情识别。

---

## 336. EcoScratch: Cost-Effective Multimodal Repair for Scratch Using Execution Feedback

**arXiv ID:** 2603.29624 | [PDF](https://arxiv.org/pdf/2603.29624v1)

**作者:** Yuan Si `[一作]` (University of Waterloo), Jialu Zhang `[通讯]` (University of Waterloo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可扩展的 Scratch 自动修复流水线，利用轻量级运行时信号决定是否升级到多模态推理，并在同一次修复轨迹中同步记录成本、能耗和验证结果。

**💡 创新点**

创新点在于将多模态证据与修复预算、验证力度联合调度的“智能控制器”设计，且在完整修复轨迹上对成本与能耗进行细粒度量化，展示了选择性多模态策略在实际部署中的优越性。

**🔧 技术方法**

核心技术包括：基于 JSON Patch 的可执行修复生成、轻量级运行时探测与信号摘要、分阶段验证（预检+完整验证）、多模态（截图/视频）与文本请求的混合调度，以及对主机端能耗与经济成本的实时跟踪。

**📊 数据集**

使用 100 个可执行的 Scratch 修复项目（包含注入错误、测试用例和金标准），与 12 种不同的 LLM（OpenAI 与 Gemini 系列）交叉评估，形成 4,800 条完整修复轨迹。

**📈 对比分析**

通过在同一轨迹预算下比较四种控制器（文本仅、始终多模态、固定多模态、启发式）评估：生成成功率、严格验证成功率、平均金钱成本与主机能耗。启发式模式获得最高生成成功率（30.3%）和严格验证成功率（8.0%），且成本和能耗分别比始终多模态低约41%和42%，在所有模型和供应商族群中保持一致。

**⚠️ 局限性**

局限性包括：评估仅在 100 个项目的有限数据集上进行，未覆盖所有教育场景；控制阈值和预算设置固定，可能在不同工作负载下表现不同；仅测量主机端能耗，未考虑云端推理能耗；未验证对学习者的教育效果或用户体验。

---

## 337. IMAGAgent: Orchestrating Multi-Turn Image Editing via Constraint-Aware Planning and Reflection

**arXiv ID:** 2603.29602 | [PDF](https://arxiv.org/pdf/2603.29602v1)

**作者:** Fei Shen `[一作]` (National University of Singapore), Jinhui Tang `[通讯]` (Nanjing Forestry University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于闭环“计划-执行-反思”的多轮图像编辑框架IMAGAgent，能够在多轮交互中持续保持语义与结构一致性。

**💡 创新点**

创新点在于引入约束感知的任务拆解、动态工具链调度以及多专家协作反思机制，三者协同实现误差纠正与语义漂移抑制。

**🔧 技术方法**

核心技术包括视觉语言模型（VLM）进行指令解析与任务拆分、大型语言模型（LLM）聚合多位VLM专家的反馈、以及基于历史上下文的动态工具链执行策略。

**📊 数据集**

在新构建的MTEditBench（1,000条多轮编辑序列）和公开的MagicBrush数据集上进行评估。

**📈 对比分析**

与ACE++、HQEdit、UltraEdit、ICEdit、VAREdit、VINCIE、OmniGen和GPT‑4o等SOTA模型对比，IMAGAgent在DINO、CLIP‑I、CLIP‑T等指标上均显著领先，尤其在编辑轮数增多时性能优势更为突出。

**⚠️ 局限性**

主要限制是闭环反馈机制导致推理时延和计算成本上升，且在极长序列或极复杂语义变更场景下仍可能出现不可逆错误。

---

## 338. Efficient Parallel Compilation and Profiling of Quantum Circuits at Large Scales

**arXiv ID:** 2603.29598 | [PDF](https://arxiv.org/pdf/2603.29598v1)

**作者:** Jane Moore `[一作]` (Queen's University Belfast), John McAllister `[通讯]` (Queen's University Belfast)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了可调密度的随机电路生成器，并实现了一种将大规模量子电路拆分为子电路并行编译、再通过排列电路拼接的并行编译方法。

**💡 创新点**

① 用可控制密度的随机电路生成器填补了大规模基准缺失的空白；② 设计了编译器无关、基于子电路拆分与排列重排的并行编译框架，实现了显著的加速。

**🔧 技术方法**

采用 Qiskit 与 PyTKET 路由算法（SabreSwap、BasicSwap）、A* 搜索生成排列电路、并行子电路拆分与合并技术。

**📊 数据集**

构造了 8000+ 条随机电路（宽度 20–200，深度 1–10 万，密度 20–100%）以及 36 条现有基准电路（MQTBench、QASMBench、Red Queen）。

**📈 对比分析**

通过对比顺序与并行编译在不同核心数、宽度、深度、密度下的墙钟时间、SWAP/门/深度开销，得到 Qiskit SabreSwap 最多 12.95 倍加速、BasicSwap 15.56 倍、PyTKET 19.80 倍，所有方法的开销均低于 1%；在网格和线性处理器拓扑上均验证了方法的有效性。

**⚠️ 局限性**

对短电路加速效果有限；需要根据电路物理属性选择最佳子电路数；排列电路带来的额外 SWAP 产生的开销和多核内存占用在宽电路上会升高；未针对特定算法结构或多设备分布式编译做进一步优化，缺乏自动确定子电路数的模型。

---

## 339. FlowID : Enhancing Forensic Identification with Latent Flow-Matching Models

**arXiv ID:** 2603.29591 | [PDF](https://arxiv.org/pdf/2603.29591v1)

**作者:** Jules Ripoll `[一作]` (INSA Toulouse), Jose Pablo Baraybar `[通讯]` (International Committee of the Red Cross)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为FlowID的身份保留式面部重建方法，用于从受伤面部照片中去除伤痕并保留可识别特征。

**💡 创新点**

创新点包括单图细调以适配离群受伤面孔与基于注意力的掩码自动定位可编辑区域，从而实现局部编辑与全局身份一致性的兼顾。

**🔧 技术方法**

技术主要包括流匹配生成模型（Stable Diffusion 3）、单图Fine‑tuning、Transformer注意力提取掩码以及逆向ODE采样。

**📊 数据集**

使用了自构建的InjuredFaces基准数据集（755张受伤面孔，449人身份）和FFHQ子集进行通用编辑评测。

**📈 对比分析**

与UltraEdit、SDEdit、ICEdit、Kontext、RF‑Solver等方法对比，FlowID在身份保留（IP）、伤痕去除（VLM）以及视觉质量（FID、CMMD、LPIPS）上均取得最优或相近的表现，特别是在受伤场景下的识别率最高。

**⚠️ 局限性**

局限性包括单图Fine‑tuning产生的额外计算开销、掩码在全脸伤痕场景下效果不佳，以及对极端损伤的适配仍需改进。

---

## 340. FigAgent: Towards Automatic Method Illustration Figure Generation for AI Scientific Papers

**arXiv ID:** 2603.29590 | [PDF](https://arxiv.org/pdf/2603.29590v1)

**作者:** Zhuoling Li `[一作]` (Lancaster University), Jun Liu `[通讯]` (Lancaster University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 FigAgent，一种多代理框架，用于自动生成高质量的 AI 论文方法说明图（MIF）。

**💡 创新点**

创新点包括：①利用人类绘图经验构建可复用的绘图工具箱；②采用 Explore‑and‑Select 的 Monte‑Carlo 树搜索策略模拟试错绘制；③引入 LLM 驱动的工具进化机制，实现工具的选择、交叉、变异优化。

**🔧 技术方法**

技术手段：LLM 驱动的解析、规划、绘制、评估与细化代理；DrawIO 接口实现 SVG 代码生成；UCT‑启发的 MCTS 用于搜索绘制路径；LLM 与 VLM 评估中间状态；工具箱以 Python 函数形式实现，可在绘制时调用。

**📊 数据集**

数据集：从 ICML、NeurIPS、ICLR 等顶会收集 4,692 篇论文及对应 MIF，80% 用作经验集构建工具箱，20% 用作测试。

**📈 对比分析**

与基线方法（Nano、DALL·E、Paper2Any、TikZero、Paper2SysArch）在 DiagramEval 指标（NA、PA）上对比，FigAgent 在 NA、PA 上均领先，获得最高 F1 分数；实验表明其在绘图质量上显著优于所有自动化基线。

**⚠️ 局限性**

局限性：依赖高质量论文-MIF 配对数据，工具箱构建与进化过程计算成本高；在极端复杂或不常见的模型结构时仍可能出现绘制错误；对最新趋势的跟进需要持续数据收集和工具更新。

---

## 341. Learn2Fold: Structured Origami Generation with World Model Planning

**arXiv ID:** 2603.29585 | [PDF](https://arxiv.org/pdf/2603.29585v1)

**作者:** Yanjia Huang `[一作]` (University of California, Los Angeles), Chenfanfu Jiang `[通讯]` (University of California, Los Angeles)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Learn2Fold 框架，用大语言模型与图结构世界模型联合生成可执行的折纸折叠序列。

**💡 创新点**

创新点在于将语义提议与物理验证解耦：LLM负责高层结构化动作生成，图世界模型进行短期几何预测，Level‑0 符号模拟器做最终一致性检查，三者协同实现长程、物理可行的折叠计划。

**🔧 技术方法**

核心技术包括：大语言模型（Transformer）生成折叠程序；图结构世界模型（Graph Neural Network）做可微的短期模拟；模型预测控制（MPC）进行候选动作排序；符号 Level‑0 约束核进行硬验证。

**📊 数据集**

使用自建的 OrigamiCode 数据集：5760 条完整折叠流程、75k 条转移轨迹；测试集覆盖 25 个折纸类别，分为简单、中等、复杂三个难度级别。

**📈 对比分析**

与 BrickGPT、GPT‑5.1/5.2 等基线对比，评估指标包括步骤层面 Precision/Recall/F1、Edge‑IoU 以及轨迹层面 Cat‑SR。Learn2Fold 在所有指标上均显著优于基线（F1≈0.74，Edge‑IoU≈0.58，Cat‑SR≈0.89）。

**⚠️ 局限性**

局限性：图世界模型仅做短期预测，无法完全捕捉极长序列中的非线性累积误差；对极其复杂或高度自相交的折纸仍可能产生误判；数据集仍无法覆盖所有真实折纸多样性，模型对未知模式的泛化尚有限。

---

## 342. Compositional Reasoning for Probabilistic Automata with Uncertainty

**arXiv ID:** 2603.29550 | [PDF](https://arxiv.org/pdf/2603.29550v1)

**作者:** Hannah Mertens `[一作]` (RWTH Aachen University), Joost-Pieter Katoen `[通讯]` (RWTH Aachen University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一套适用于具有不确定性的概率自动机（Parametric Probabilistic Automata pPA 与 Robust Probabilistic Automata rPA）的组合式验证框架，能够通过 assume‑guarantee（AG）推理进行模组化验证，并扩展了对多目标查询、期望奖励、参数单调性以及基于模拟的 AG 推理的支持。

**💡 创新点**

创新点包括：① 将 Kwiatkowska 等人针对普通 PA 的 AG 规则推广到带参数和不确定性的模型；② 设计了可处理参数单调性的专用 AG 规则；③ 在 rPA 中引入凸并行组合与 PA-Reduction，实现了对凸 rPA 的 AG 规则；④ 提出了强模拟与鲁棒强模拟两种新的模拟关系，并用其构建了完整的模拟基 AG 规则；⑤ 明确指出 AG 规则在非凸、记忆无关自然或间隔松弛并行时的失效。

**🔧 技术方法**

使用的技术主要包括：参数化概率分布与不确定集的定义；策略投影与图保持性；多目标模型检查；凸集与极点的几何运算；PA 与 rPA 的并行组合定义；PA-Reduction（将 rPA 转化为可能无限分支的 PA）；强模拟与鲁棒强模拟的关系定义；以及安全 PCTL 与期望奖励的可达性分析。

**📊 数据集**

该工作为理论研究，未使用具体实验数据集；验证和比较均在形式化模型和定理证明层面完成。

**📈 对比分析**

由于本研究主要是形式化框架与理论证明，没有针对算法实现或性能评估的实验比较；作者仅在理论层面证明了规则的正确性与完整性，并通过反例说明限制情况。

**⚠️ 局限性**

限制主要体现在：① AG 规则在记忆无关自然、非凸不确定集以及间隔松弛并行组合下不再适用；② 对凸 rPA 需要凸并行组合与 PA-Reduction，导致模型可能变为无限分支；③ 模拟基 AG 规则需要检验强模拟与鲁棒强模拟，尚无高效算法；④ 对期望奖励、长跑平均等更复杂属性的支持仍待进一步研究。

---

## 343. Computing Topological Transition Sets for Line-Line-Circle Trisectors in $R^3$

**arXiv ID:** 2603.29540 | [PDF](https://arxiv.org/pdf/2603.29540v1)

**作者:** Eunku Park `[一作]` `[通讯]` (DGIST), Eunku Park (DGIST)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

研究并求出了在三维空间中线–线–圆的三分支（trisector）曲线的确切拓扑转换集，提出了一种基于参数空间分割的精确验证框架。

**💡 创新点**

创新点在于：①引入“转移集（transition‑set）”概念，直接定位拓扑变化的参数壁面而不需要全局的 CAD；②将非二次（quartic）bisector 的复杂度与传统的线性三分支问题区分开来；③通过斜率坐标（slope‑coordinate）局部吹放（blow‑up）实现对无穷远处分支结构的精确判定。

**🔧 技术方法**

技术手段包括：符号代数精确计算（Jacobian rank‑drop、理想同余、极限点分析）、投影与饱和（projective closure + saturation）、斜率坐标局部分析、SMT 求解器提取 rational witness、计算机代数系统（如 Macaulay2）实现精确验证。

**📊 数据集**

使用的数据集为符号参数族：参数 (k,R,t)（或其 Weierstrass 形式）和固定的 rational witness 作为具体实例；未使用实际几何点集或实验数据。

**📈 对比分析**

与传统的全局 CAD 进行对比：该方法将符号计算聚焦于参数空间的有限壁面，避免了在三维空间中对完整几何体的全局分解；在符号复杂度上呈现单指数增长（相较于 CAD 的高指数或不可估量增长），但在实践实现层面未给出具体运行时间或资源评估；总体上在受限维度下显得更可行。

**⚠️ 局限性**

局限性：①仅处理了最简单的非二次 bisector 情况（线–线–圆），未扩展到更一般的多对象或更高阶交点；②对参数空间的分割依赖符号计算，仍然在高阶多项式求解时可能出现计算瓶颈；③未提供对实际几何数据或随机实例的实验验证；④对全局拓扑关系（不同参数区间是否产生不同拓扑类型）尚未做完整证明。

---

## 344. Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge

**arXiv ID:** 2603.29535 | [PDF](https://arxiv.org/pdf/2603.29535v1)

**作者:** Sowmya Vajrala `[一作]` (Samsung Research Institute Bangalore), Ashok Senapati `[通讯]` (Samsung Research Institute Bangalore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种统一部署框架，允许在移动边缘设备上用单一视觉基础模型（Latent Diffusion）通过把LoRA权重作为运行时输入实现多任务GenAI推理，并通过QUAD（Quantization with Unified Adaptive Distillation）统一量化配置，实现动态任务切换。

**💡 创新点**

创新点在于：①将LoRA权重转化为运行时输入，消除多任务模型复制与多二进制文件；②提出QUAD方法，通过知识蒸馏使所有LoRA共享同一量化参数，兼容NPU固定量化要求；③构建轻量化推理栈，支持在不同芯片上快速切换任务，显著降低内存占用与推理延迟。

**🔧 技术方法**

技术要点包括：低秩适配器（LoRA）训练、量化感知训练、知识蒸馏、共享量化参数（scale/zero‑point）设计、图优化（量化融合、常量折叠）、NPU友好的运行时API。

**📊 数据集**

实验使用标准视觉生成数据集（如ImageNet/COCO）训练的Latent Diffusion模型，随后针对“Prompt Guided Image Transformation”“Object Removal”等任务训练对应LoRA；具体数据集未在论文中公开，但均为公开图像生成与编辑数据集。

**📈 对比分析**

通过将FP32服务器推理结果与在Qualcomm、MediaTek、LSI芯片上部署的INT8模型进行对比，论文报告：内存占用下降6倍、推理延迟提升4倍；质量指标（FID、SSIM、PSNR）与FP32保持近似，误差在可接受范围内。

**⚠️ 局限性**

局限性：①需要所有LoRA满足共享量化参数，若任务间分布差异大可能导致性能下降；②对NPU的固定量化参数支持有限，需在每个硬件平台进行校准；③仍依赖LoRA预训练质量，若LoRA本身训练不足会影响整体表现。

---

## 345. Stand-Alone Complex or Vibercrime? Exploring the adoption and innovation of GenAI tools, coding assistants, and agents within cybercrime ecosystems

**arXiv ID:** 2603.29545 | [PDF](https://arxiv.org/pdf/2603.29545v1)

**作者:** Jack Hughes `[一作]` (University of Cambridge), Daniel R. Thomas `[通讯]` (University of Strathclyde)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对地下与暗网网络犯罪论坛进行大规模文本挖掘和定性访谈，系统分析GenAI工具（如ChatGPT、Copilot、LLM助手等）在网络犯罪生态中的采用、创新路径与经济影响。

**💡 创新点**

将创新理论与进化经济学相结合，首次提出“Stand‑Alone Complex”（完整自动化的犯罪团伙）与“Vibercrime”（低门槛的AI辅助）两种极端转型情景，并构建相应的理论框架，对网络犯罪生态中的AI影响进行系统性实证检验。

**🔧 技术方法**

使用主题建模（BERTopic+HDBSCAN）提取AI相关主题，关键词搜索与LLM分类识别Vibe Coding讨论，配合定性编码、文本挖掘工具与可视化分析技术。

**📊 数据集**

CrimeBB 数据集，包含超过100万篇论坛帖子，筛选后得到97,895篇满足条件的帖子（2022‑11至2025‑12期间），来源于公开与暗网的网络犯罪讨论社区。

**📈 对比分析**

通过主题与关键词随时间变化的频次可视化，评估LLM分类器约80%正样本相关性；定性结果表明AI主要被用作代码辅助和营销工具，未出现显著破坏性创新；与传统网络犯罪工具使用对比，显示AI对技术层面影响有限，经济收益提升不明显。

**⚠️ 局限性**

样本主要来自公开论坛，暗网深层数据缺失；LLM分类准确率有限，无法精准量化；缺乏真实攻击实验与纵向追踪；研究截止2025年，后续AI发展未知；难以充分区分AI在技术与业务层面的多重影响。

---

## 346. Can LLM Agents Identify Spoken Dialects like a Linguist?

**arXiv ID:** 2603.29541 | [PDF](https://arxiv.org/pdf/2603.29541v1)

**作者:** Tobias Bystrich `[一作]`, Akbar Karimi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究使用大型语言模型（LLM）结合音素转写和方言特征提示，判断瑞士德语的两大方言区（High Alemannic 与 Highest Alemannic），并与人类语言学家以及HuBERT基准进行对比。

**💡 创新点**

创新点在于将LangGraph代理式多节点推理与音素转写、语义提示以及方言特征表述相结合，使LLM在音素文本上完成方言分类，并展示代理模型相较单一LLM的性能提升。

**🔧 技术方法**

使用的技术包括GPT‑4o mini LLM、LangGraph代理框架、多重提示与音素转写（Wav2Vec2‑Phoneme）、HuBERT编码器，以及人工基准与人类语言学家对照。

**📊 数据集**

使用的数据集为SwissDial与STT4SG‑350这两套瑞士德语语料库，经过音素转写并附加标准德语翻译。

**📈 对比分析**

在平衡的80句测试集上与HuBERT（准确率66.3%）和人类专家（准确率72.5%）对比，单一LLM仅达47.8%，而代理模型提升至58%；HuBERT仍保持最佳性能。

**⚠️ 局限性**

局限性包括仅使用GPT‑4o mini，模型能力受限；ASR转写误差高影响结果；样本量有限，转写质量不足；代理工具使用不充分；未覆盖所有方言细分。

---

## 347. Learning Surrogate LPV State-Space Models with Uncertainty Quantification

**arXiv ID:** 2603.29532 | [PDF](https://arxiv.org/pdf/2603.29532v1)

**作者:** E. Javier Olucha `[一作]` (Eindhoven University Of Technology), Roland Tóth `[通讯]` (Eindhoven University Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种贝叶斯方法，联合估计自调LPV状态空间模型、调度映射以及模型不确定性，并给出预测响应的置信区间，用于 surrogate 模型的构建与验证。

**💡 创新点**

在自调LPV-SS框架中首次实现完整的贝叶斯不确定性量化，利用拉普拉斯近似得到可计算的预测分布，并通过引入可识别的线性先验（BLA）简化先验设定。

**🔧 技术方法**

采用贝叶斯推断（MAP + 拉普拉斯近似）、线性参数化的LPV-SS模型、前馈神经网络调度映射、梯度优化（ADAM+L-BFGS）、JAX 自动微分、递归雅可比与状态灵敏度计算。

**📊 数据集**

使用二维耦合质量-弹簧-阻尼系统（12个质量、24 状态）的仿真数据，生成 3460 条训练样本和 600 条测试样本，输入信号为 chirp、随机相位正弦波和阶跃信号。

**📈 对比分析**

与三种线性 LTI 模型（ssest、n4sid、Jax-PEM）及无不确定性的 LPV 模型比较，评估指标为模拟最佳拟合率（BFR）。在训练集 BFR>96%，测试集 BFR>86%，显著优于线性模型，且置信区间随误差增长自动扩展。

**⚠️ 局限性**

局限性：拉普拉斯近似对强非线性不确定性可能不足；需要可用的线性先验（BLA），若无则效果受限；对极大状态维度或多变量系统的扩展仍需改进；计算成本虽然随时间线性，但对非常高维模型仍有挑战。

---

## 348. Self-Supervised Federated Learning under Data Heterogeneity for Label-Scarce Diatom Classification

**arXiv ID:** 2603.29633 | [PDF](https://arxiv.org/pdf/2603.29633v1)

**作者:** Mingkun Tan `[一作]` (University of Bielefeld), Tim W. Nattkemper `[通讯]` (University of Bielefeld)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在分布式标签稀缺的硅藻分类任务中，提出并评估了自监督联邦学习（SSFL）框架；

**💡 创新点**

创新点包括：①引入PreDi分割方案，将标签空间异质性分解为普及度（prevalence）和类集大小差异（disparity）两维度；②基于普及度的个性化加权策略PreP‑WFL，以提升低普及度场景下的性能；

**🔧 技术方法**

主要技术包括：Masked AutoEncoder（MAE）做自监督预训练、FedAvg做联邦聚合、PreP‑WFL做加权交叉熵微调；

**📊 数据集**

使用公开的UDE diatoms in the Wild 2024数据集（83,570张图像，611个硅藻分类），并从中构造标签稀缺（每类50张样本）与不同异质性设置；

**📈 对比分析**

与局部单独训练、中心化SSL以及不同异质性分割（PreDi）进行对比，实验表明SSFL在IID和非IID条件下均优于局部训练，且PreP‑WFL在低普及度时可提升约3–6个百分点，整体宏F1和准确率提升幅度明显；

**⚠️ 局限性**

局限性：仅在四个客户端的小规模联邦环境下验证，且未探讨更大规模联邦、跨域迁移或多任务扩展，未来需进一步扩展与验证。

---

## 349. An Empirical Study of Multi-Agent Collaboration for Automated Research

**arXiv ID:** 2603.29632 | [PDF](https://arxiv.org/pdf/2603.29632v1)

**作者:** Yang Shen `[一作]` (University of Technology Sydney), Yuhui Shi `[通讯]` (Southern University of Science and Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

对单一代理、子代理和团队代理三种多代理协作框架在自动化机器学习研究中的搜索与优化效果进行系统实验对比。

**💡 创新点**

构建可重复的执行测试平台并首次量化不同多代理架构在固定计算预算下对验证指标的影响，提出动态路由多代理架构的设计思路。

**🔧 技术方法**

使用 Git 工作树隔离、Search/Replace 代码补丁契约、全局显式记忆、LLM（如 GPT）驱动的代理、预热校验和训练/评估循环等技术。

**📊 数据集**

以目标代码库（script）及其训练数据为实验对象，未使用公开标准数据集。

**📈 对比分析**

在 300s/600s 固定时间预算下，以验证 bits‑per‑byte 的改进量 Δ 及成功率作为指标，子代理在短期内取得最快提升但多样性低，团队代理更稳定但提升速度慢。

**⚠️ 局限性**

子代理缺乏多样性，团队代理易导致多作者冲突和运行崩溃；实验受限于单 GPU RTX 3090，未覆盖更大规模集群环境。

---

## 350. Bioinspired123D: Generative 3D Modeling System for Bioinspired Structures

**arXiv ID:** 2603.29592 | [PDF](https://arxiv.org/pdf/2603.29592v1)

**作者:** Rachel K. Luu `[一作]` (Massachusetts Institute of Technology), Markus J. Buehler `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一个轻量化的文本到 3D 结构生成系统 Bioinspired123D，能够将自然语言描述直接转换为可执行的 Blender Python 脚本，从而生成可打印的生物启发式几何体。

**💡 创新点**

创新点在于：① 采用代码即几何（code‑as‑geometry）的表示方式，减少 token 量并提升可控性；② 通过 LLM fine‑tune、RAG 与图式 agentic 框架实现执行‑渲染‑评估循环，实现自动脚本修复和结构改进；③ 在保持参数极少（3B）且低算力的前提下，获得与更大模型相当甚至更优的 3D 生成性能。

**🔧 技术方法**

技术手段包括：Llama‑3.2‑3B‑Instruct + LoRA fine‑tune；BGE‑small‑en‑v1.5 进行 RAG；Blender 4.2 的 Python API 进行脚本生成与验证；LangGraph 进行 agentic 迭代流程；GPT‑4o‑mini（或 Qwen‑3‑VL‑2B）作为 VLM 评估与反馈；自动 headless Blender 渲染 pipeline。

**📊 数据集**

使用了约 4,558 条手工与 LLM 生成的 Blender 脚本与自然语言配对数据集，覆盖 helical、cellular、tubular 三类生物启发式结构，并通过多样化脚本生成与嵌入 reasoning、BlendNet 典型场景增强模型泛化。

**📈 对比分析**

在自定义 320 题的 benchmark（按难度分层）中，对比基线 Llama‑3.2、GPT‑4o‑mini 以及本系统。Bioinspired123D 在整体得分约 60%（基线 18%），相比未 fine‑tune 提升近 3 倍；在 hard 题上通过 agentic 循环提升约 15%，并在多模型对比中表现优于更大规模的 GPT‑4o‑mini 与 GPT‑5‑mini。

**⚠️ 局限性**

局限性包括：① 仍需人工或 VLM 辅助的评估，缺乏统一的客观定量指标；② 数据集仅覆盖三类结构，难以覆盖更广泛的多尺度、生物多样性；③ 对制造约束与机械性能评估依赖后续插件，尚未内置；④ 极端或复杂结构仍可能出现脚本错误或意图偏差。

---

## 351. Turbo4DGen: Ultra-Fast Acceleration for 4D Generation

**arXiv ID:** 2603.29572 | [PDF](https://arxiv.org/pdf/2603.29572v1)

**作者:** Yuanbin Man `[一作]` (University of Texas at Arlington), Miao Yin `[通讯]` (University of Texas at Arlington)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出Turbo4DGen，一种针对4D生成的超快加速框架；

**💡 创新点**

创新点在于：1）滚动缓存（Rolling Cache）实现SCM注意力块级别的重用；2）语义感知（Semantic‑Aware）令牌裁剪，仅保留对生成关键的Token并用缓存填充；3）自适应SCM链绕过调度器（Adaptive SCM Chain Bypass Scheduler）根据跨步相似度动态跳过冗余链；

**🔧 技术方法**

主要技术包括扩散模型加速、SCM注意力机制优化、缓存重用与动态调度、token层级裁剪；

**📊 数据集**

使用ObjaverseDy和Consistent4D两个公开数据集进行实验；

**📈 对比分析**

与基线SV4D、DeepCache、AT‑EDM等方法对比，Turbo4DGen平均可实现9.7×的速度提升，内存降低约24.2%，同时保持甚至超越原模型的视觉质量（LPIPS、CLIP‑S、PSNR、SSIM、FVD等指标均无显著下降）；

**⚠️ 局限性**

局限性包括：1）过度裁剪会导致细节丢失；2）需要手动设定阈值（ASR阈值、Top‑K比例），对不同场景可能需调参；3）目前仅验证于SV4D架构，对其他4D模型的泛化尚未完全评估。

---

## 352. Generating Key Postures of Bharatanatyam Adavus with Pose Estimation

**arXiv ID:** 2603.29570 | [PDF](https://arxiv.org/pdf/2603.29570v1)

**作者:** Jagadish Kashinath Kamble `[一作]` (Indian Institute of Technology), Partha Pratim Das `[通讯]` (Ashoka University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种姿态感知的生成框架，利用条件生成模型（GAN 与扩散模型）结合姿态估计模块，自动合成印度古典舞 Bharatanatyam 的关键姿势（Key Postures）；

**💡 创新点**

创新点在于将姿态估计的关键点损失和姿态一致性损失作为监督信号，确保生成姿势在解剖学与文化规则上的准确性，并且实现了模型无关的姿态感知通用框架；

**🔧 技术方法**

使用了条件生成对抗网络 (cGAN)、条件扩散模型 (conditional diffusion)、MediaPipe 姿态估计、关键点对齐损失 (ℒ_kp) 与姿态一致性损失 (ℒ_pose) 等技术；

**📊 数据集**

采用了自建的 Bharatanatyam 关键姿势数据集，包括 1,276 条高质量视频、15 个基础 Adavu、58 种变体与 334 条运动序列；

**📈 对比分析**

通过对比四种配置（cGAN、cGAN+姿态、扩散、扩散+姿态），使用 FID 与 MS‑SSIM 指标评估，结果显示扩散+姿态模型取得最佳表现（FID 19.32，MS‑SSIM 0.78），明显优于无姿态监督的模型；

**⚠️ 局限性**

局限性在于仍难以捕捉 Bharatanatyam 中细腻的面部表情与情感表达（abhinaya），并且受限于数据集规模与多样性，未来需进一步扩充多模态数据与细粒度标注。

---

## 353. FlowPIE: Test-Time Scientific Idea Evolution with Flow-Guided Literature Exploration

**arXiv ID:** 2603.29557 | [PDF](https://arxiv.org/pdf/2603.29557v1)

**作者:** Qiyao Wang `[一作]` (Shenzhen Institute of Advanced Technology Chinese Academy of Sciences), Min Yang `[通讯]` (Shenzhen Institute of Advanced Technology Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `40105733-5154-44cd-8090-a8cab9e64b07` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为FlowPIE的动态、交互式科研想法生成框架，将文献检索与创意生成融合为测试时的进化过程，利用LLM评估器指导搜索与迭代；

**💡 创新点**

核心创新在于：①用流引导的蒙特卡罗树搜索（MCTS）实时调节文献检索路径；②将初始想法视为种群，在此基础上应用遗传算子（选择、交叉、带隔离岛的变异）进行迭代进化；③通过LLM生成的奖励模型（GRM）对想法进行多维度评估；

**🔧 技术方法**

主要技术包括：流引导MCTS、GFlowNets启发式、LLM生成式思路与评估、隔离岛变异、遗传算法框架；

**📊 数据集**

实验使用AI Idea Bench 2025（AI会议论文）与IdeaBench（生物医学论文）两大公开基准；

**📈 对比分析**

与SCIPIP、Research Agent、Chain‑of‑Ideas、VirSci等LLM/代理基线对比，FlowPIE在多项指标（I2T、I2I、IMCQ、语义相似度、思想重叠、NI、FI、奖励分数、人类评估）均取得更高分，且标准差更低，显示更稳健；

**⚠️ 局限性**

局限包括：依赖LLM评估的主观性、对大型模型和算力需求高、需要手工设置阈值与超参、在极端跨学科场景下的适配仍待验证。

---

## 354. Capturing Multivariate Dependencies of EV Charging Events: From Parametric Copulas to Neural Density Estimation

**arXiv ID:** 2603.29554 | [PDF](https://arxiv.org/pdf/2603.29554v1)

**作者:** Martin Výboh `[一作]` (Kempelen Institute of Intelligent Technologies), Gabriela Grmanová `[通讯]` (Kempelen Institute of Intelligent Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了电动车充电事件的多变量依赖，提出使用 Vine 结构式 Copula 与 CODINE 神经网络方法生成高质量合成数据。

**💡 创新点**

首次将 Vine Copula 与 CODINE 迁移到 EV 充电领域，并通过多数据集评估显示其优于传统参数 Copula 与 GMMNet 的尾部与相关性捕获能力。

**🔧 技术方法**

采用 Vine Copula 构建分层 bivariate 依赖结构、CODINE 的对抗神经 Copula 密度估计，以及 GMMNet、传统参数 Copula 与 KDE 等对比方法。

**📊 数据集**

三个真实数据集：斯洛伐克住宅充电（1,097 次）、挪威 Trondheim 多户住宅（5,442 次）与英国 Dundee 公共快速充电（166,888 次）。

**📈 对比分析**

通过 NLL、τ-Diff、尾部误差 MAE_LT/UT、平均负载误差 MAE_Load、记忆度 ρ₂ 等指标进行比较，结果表明 Vine 与 CODINE 在所有数据集上在相关性、尾部与负载精度上均优于其他基线，尤其在大规模 Dundee 数据上 MAE_Load 下降近 50%。

**⚠️ 局限性**

CODINE 在极小数据集上的收敛性受限，且所有方法均需较长训练时间；缺乏对时间序列演化的建模，未来需加入动态特征。

---

## 355. Joint Identification and Sensing with Noisy Feedback: A Task-Oriented Communication Framework for 6G

**arXiv ID:** 2603.29649 | [PDF](https://arxiv.org/pdf/2603.29649v1)

**作者:** Yaning Zhao `[一作]` (Technical University of Braunschweig), Christian Deppe `[通讯]` (Technical University of Braunschweig)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在带有噪声严格因果反馈的状态依赖离散无记忆信道（DMC）上进行联合识别与感知（JIDAS）的理论极限，给出了确定性和随机化编码方案的容量-失真函数下界与上界。

**💡 创新点**

创新点在于：①首次将噪声反馈考虑进JIDAS框架；②提出结合公共随机性与哈希的编码构造；③推导出既适用于确定性又适用于随机化识别的容量-失真双边界，揭示了噪声反馈对识别与感知权衡的实质影响。

**🔧 技术方法**

采用的信息理论技术包括：状态平均化、前向边缘通道的构造、超级叠加编码、Slepian–Wolf压缩、哈希函数的随机映射，以及典型序列与类型论的误差分析。

**📊 数据集**

在实验部分采用了一个二进制乘法式状态模型（Y = X·S，Z = Y⊕N），其中 S ∼ Bernoulli(p_S)，N ∼ Bernoulli(p_N)，并用 Hamming 失真度量进行评估。

**📈 对比分析**

通过与传统时间共享（TS）方案的对比，展示了在 p_S 较大且反馈噪声 p_N 较小时，所提出的联合设计方案在容量-失真曲线上的明显优势；当 p_N 增大时两者的差距减小。

**⚠️ 局限性**

主要限制包括：①上界与下界尚未完全匹配，缺乏精确极限条件；②仅考虑了严格因果噪声反馈，未涉及延迟或速率限制的反馈；③分析局限于离散无记忆通道与逐符号失真约束，未扩展到高斯、多用户或平均失真情形。

---

## 356. Storing Less, Finding More: How Novelty Filtering Improves Cross-Modal Retrieval on Edge Cameras

**arXiv ID:** 2603.29631 | [PDF](https://arxiv.org/pdf/2603.29631v1)

**作者:** Sherif Abdelwahab `[一作]` `[通讯]`, Sherif Abdelwahab

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种边缘设备流式检索体系，利用设备端的 ε‑net 过滤器去除语义冗余帧，构建降噪嵌入索引，并在云端使用跨模态适配器和重排序器提升检索精度。

**💡 创新点**

① 证明在检索索引中去除冗余帧能显著提升跨模态检索；② 用单次通行的 streaming ε‑net 过滤器在不需要离线或查询依赖的情况下已超越多种离线采样方法；③ 仅使用嵌入传输，既节省带宽又保护隐私。

**🔧 技术方法**

设备端轻量级视觉编码器（TinyCLIP、MobileCLIP等）及 INT8/ONNX实现；ε‑net 过滤器（阈值 τ）；Locked‑image Tuning（LiT）多层感知器适配器；云端大型模型重排序器（如 SigLIP 2）；低功耗 BLE 传输估算。

**📊 数据集**

AEA（Aria Everyday Activities）19 段视频（≈85k 帧），EPIC‑KITCHENS 7 段视频，包含语义事件注释。

**📈 对比分析**

对比 Full、Novelty（ε‑net）、k‑means、farthest‑point、uniform、random 等帧选择策略，并在八种视觉‑语言模型上评估 Hit@5。Novelty 在大多数模型上至少比 Full 提升 3–10pp，单帧流过滤器在 5 FPS 下功耗约 2.7 mW；最终云端重排序后，Hit@5 达 45.6%，Hit@50 77.9%。

**⚠️ 局限性**

① 阈值 τ 需为不同事件长度手动设定，缺乏自动适应；② 过滤器在长视频中压缩率漂移；③ 仅评估厨房场景，缺乏多环境验证；④ 训练数据有限导致训练/测试差距；⑤ 仅使用 RGB 流，未利用多模传感器；⑥ ±0.5 s 时间容差未做细粒度测试。

---

## 357. Enhancing LLM-Based Bug Reproduction for Android Apps via Pre-Assessment of Visual Effects

**arXiv ID:** 2603.29623 | [PDF](https://arxiv.org/pdf/2603.29623v1)

**作者:** Xiangyang Xiao `[一作]` (Xiamen University), Rongxin Wu `[通讯]` (Xiamen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

LTGDroid 通过对 Android App 的 UI 动作进行枚举和预评估，以辅助 LLM 自动重现用户报告的 Bug，显著提升重现成功率。

**💡 创新点**

创新点在于利用 LLM 预先执行并记录所有可能 UI 动作的视觉效果，让模型在规划时根据这些视觉反馈来选择更合适的操作，从而降低误操作并提升精确度。

**🔧 技术方法**

核心技术包括 GPT‑4.1（多模态 LLM）、结构化提示工程、基于视图层级的 UI 动作枚举、链式思维推理以及预评估的 UI 转移图。

**📊 数据集**

使用了 75 条真实 Bug 报告（共 45 个开源 Android App，51 条 Crash，24 条 Non‑Crash），每条报告包含手工标注的最小重现 UI 步骤数。

**📈 对比分析**

与 AdbGPT、ReBL 及其视觉版 ReBL‑visual 等基线相比，LTGDroid 在全部 75 条报告上的成功率达 87.51%，比 ReBL 高 49.16%（Crash）/ 34.85%（Non‑Crash），平均耗时 20.45 分钟、消耗 67.07K 令牌、成本约 0.27 美元，虽然耗时略高，但成功率显著提升。

**⚠️ 局限性**

局限性包括 LLM 对初始化步骤识别不足、对长步骤或不完整 S2R 的跟踪误差、某些特殊 UI 操作（如非可长按元素、特定滑动范围）未被支持，以及预评估过程导致的额外时间和令牌消耗。

---

## 358. Video-Oasis: Rethinking Evaluation of Video Understanding

**arXiv ID:** 2603.29616 | [PDF](https://arxiv.org/pdf/2603.29616v1)

**作者:** Geuntaek Lim `[一作]` (Sejong University), Yukyung Choi `[通讯]` (Sejong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Video-Oasis诊断套件，用于系统评估并剔除视频理解基准中的视觉或时间短路，识别真正的时空依赖任务；

**💡 创新点**

创新点在于：① 以视觉、时间、模糊三维测试分离短路；② 结合跨模型共识与人工审核提升判定可靠性；③ 通过诊断结果提炼五大视频本土挑战，并为模型设计提供可操作的实验指导；

**🔧 技术方法**

采用多模态大语言模型（如Qwen、Eagle、InternVL等）、CLIP等视觉编码器、文本音频转换、Narrative摘要、Center-Frame、Frame Shuffling、Bag-of-Frames等评测手段；

**📊 数据集**

使用了14个多样化基准（覆盖感知、时空、推理、全局任务），共计24,416 QA样本，最终通过诊断筛选出11,332样本；

**📈 对比分析**

与现有基准直接对比，发现大多数任务可在无视觉/无时间输入下完成，平均短路率约54%；在剔除短路后，最先进模型的性能仅略高于随机猜测（约25%）；进一步实验显示，精确时空定位与自适应思考可显著提升性能，SFT与RLVR各具优势；

**⚠️ 局限性**

局限包括：诊断测试仍可能漏判非视觉/非时间依赖的细微特征；人工审核成本高；现有模型在长期叙事与因果推理方面仍表现低迷；研究聚焦于问答任务，未涵盖更广泛的视频生成或检索场景。

---

## 359. Style-Instructed Mask-Free Virtual Try On

**arXiv ID:** 2603.29587 | [PDF](https://arxiv.org/pdf/2603.29587v1)

**作者:** Mengqi Zhang `[一作]` (Amazon), Karim Bouyarmane `[通讯]` (Amazon)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无遮罩、基于扩散模型的虚拟试衣框架，支持通过文本提示控制服装类型与穿搭样式。

**💡 创新点**

创新点在于：①引入注意力引导辅助损失让模型在无掩码条件下聚焦服装区域；②结合文本指令实现多属性可控试衣；③使用弱监督的衣物区域提示而非人工掩码；④通过生成三元组数据提升无配对训练。

**🔧 技术方法**

采用流匹配扩散模型（flow‑matching diffusion）、DiT 变压器结构、CLIP/T5 文本编码器、注意力引导损失以及参考位置嵌入。

**📊 数据集**

使用 VITON‑HD、DressCode、CatVTON 等公开服装数据集，并通过 CatVTON 生成的伪三元组进行训练。

**📈 对比分析**

在未配对评估（person–garment 随机不匹配）下，与 LaDI‑VTON、OOTD、IDM‑VTON 等基准比较，SMF‑VTO 在 SSIM、LPIPS、FID 和 IS 等指标上均优于现有方法，显示更高的视觉质量与一致性。

**⚠️ 局限性**

局限性包括对极度跨类别或复杂多层服装的泛化仍有限，文本提示的理解仍受预训练模型限制，且缺乏视频/动态试衣能力。

---

## 360. Parallelobox: Improved Decomposition for Optimized Parallel Printing using Axis-Aligned Bounding Boxes

**arXiv ID:** 2603.29579 | [PDF](https://arxiv.org/pdf/2603.29579v1)

**作者:** Hayley Hatton `[一作]` (University of Hull), John Murray `[通讯]` (University of Huddersfield)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了一种基于轴向对齐包围盒的分解算法，旨在优化多机并行打印的时间与质量。

**💡 创新点**

创新点在于将轴向包围盒作为高度场用于分解，并结合对称性预处理、k-means++种子生成和元启发式迭代，显著提升并行打印时间且保持装配质量。

**🔧 技术方法**

使用了轴向对齐包围盒、对称性检测、k-means++聚类、软硬约束目标函数、元启发式搜索、CGAL布尔运算、Ultimaker Cura切片仿真等技术。

**📊 数据集**

实验数据集主要来自Thingi10K的多种模型（如3DBenchy、Utah Teapot、Stanford Bunny、Brain Left等）以及自定义的MRI转STL模型。

**📈 对比分析**

通过与Cube Skeleton Segmented Shell和Symmetry-Based Decomposition两种现有并行分解方法在相同模型与打印机配置下的仿真GCode时间比较，Parallelobox在大多数模型上显著降低并行打印时间，但计算时间更高。

**⚠️ 局限性**

局限包括计算耗时较高、对某些模型在给定打印机数下可能无法生成合法分解、目标函数权重固定导致灵活性不足、未考虑装配时间与异构打印机等。

---

## 361. AdaptDiff: Adaptive Guidance in Diffusion Models for Diverse and Identity-Consistent Face Synthesis (Student Abstract)

**arXiv ID:** 2603.29569 | [PDF](https://arxiv.org/pdf/2603.29569v1)

**作者:** Eduarda Caldeira `[一作]` (Fraunhofer IGD), Fadi Boutros `[通讯]` (Fraunhofer IGD)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种动态加权的负条件策略 AdaptDiff，用于身份条件扩散模型生成面部图像，从而提升身份一致性和多样性。

**💡 创新点**

通过随采样时间变化的线性调度自适应地调整负条件强度，兼顾早期自由探索与后期身份分离，实现更高的身份可分离性和内部变异。

**🔧 技术方法**

利用分类器自由引导（CFG）结合身份正负条件的扩散模型，并在采样过程中采用线性调度技术。

**📊 数据集**

在 FFHQ 和 CASIA-WebFace 上预训练的 IDiff‑Face 模型上进行实验，生成数据用于面部识别训练。

**📈 对比分析**

与固定负条件的 NegFaceDiff 以及其他 SOTA 方法对比，在 LFW、AgeDB、CFP‑FP 等多项公开基准上取得更高识别准确率和更低 EER，尤其在 IJB‑C 上表现更优。

**⚠️ 局限性**

在 CASIA-WebFace 预训练模型生成的数据内在变异较大时，AdaptDiff 的提升有限；此外，生成模型仍面临对真实身份潜在泄露的风险。

---

## 362. When Can We Trust LLM Graders? Calibrating Confidence for Automated Assessment

**arXiv ID:** 2603.29559 | [PDF](https://arxiv.org/pdf/2603.29559v1)

**作者:** Robinson Ferrer `[一作]` (University of Central Florida), Shashank Sonkar `[通讯]` (University of Central Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何利用LLM自报置信度来判断自动批改的可靠性，从而实现只对高置信度的评估自动化，其余案例转人工复核。

**💡 创新点**

创新点在于系统性比较三种置信估计方法（自报置信度、自一致性投票、token概率）在七种不同规模LLM上的校准表现，并提出自报置信度是最具成本效益且校准最好的选择。

**🔧 技术方法**

技术包括基于JSON提示的二元评估、置信度归一化、5次采样自一致性投票、token概率聚合、以及ECE、Brier、AUC-ROC等校准评估指标。

**📊 数据集**

使用了三份教育数据集：长答题化学数据集RiceChem、短答题科学数据集SciEntsBank和Beetle；共计约23,000个评估实例。

**📈 对比分析**

实验显示自报置信度在所有模型与数据集上均实现最低ECE（平均0.166）和最优Brier（0.223），而自一致性在成本（5×调用）下仍比自报置信度差约38%；模型规模越大，准确率提升显著，但校准提升不一。

**⚠️ 局限性**

限制在于置信度分布高度偏向高值（top‑skewed），导致在严格风险阈值下自动化覆盖率仅为约10–20%，且自报置信度仍需针对不同模型和任务进行阈值调优。

---

## 363. Total Variation Guarantees for Sampling with Stochastic Localization

**arXiv ID:** 2603.29555 | [PDF](https://arxiv.org/pdf/2603.29555v1)

**作者:** Jakob Kellermann `[一作]` `[通讯]` (Weierstrass Institute), Jakob Kellermann (Weierstrass Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文给出了 SLIPS（Stochastic Localization via Iterative Posterior Sampling）采样算法在总变差距离上的理论收敛保证，证明在满足弱目标假设下所需迭代步数线性依赖于维度。

**💡 创新点**

创新点在于：①首次为 SLIPS 提供 TV 收敛分析；②发现并证明 log‑SNR 适配离散化在理论上能够最小化离散误差；③在不需要全局 Lipschitz 约束的情况下实现线性维度依赖。

**🔧 技术方法**

主要技术包括：将 SLIPS 的随机微分方程与 Score‑Based Generative Models（SGM）中的反向 OU 过程对齐；利用 Girsanov 定理和 Pinsker 不等式将 TV 距离转化为 KL；借助 Stochastic Localization 的马尔可夫性质证明条件期望为马尔可夫过程；对离散化误差进行细致的 L² 估计；以及对 log‑SNR 离散化的最优性证明。

**📊 数据集**

论文未使用具体实验数据集，主要以理论分析为主；在讨论中以双峰高斯混合模型（Gaussian Mixture Model）为示例计算常数。

**📈 对比分析**

与之前的 Reverse Diffusion Monte Carlo (RDMC) 方案比较：SLIPS 免除全局 Lipschitz 假设；所需的迭代步数 K 仅线性依赖于维度（O(d/ε²)），而 RDMC 在 Lipschitz 规模为 L 时需要 O(dL²/ε²) 步；实验上 SLIPS 在多峰分布上表现优异，能够通过合适的 t₀ 实现 V‑形性能曲线。

**⚠️ 局限性**

局限性包括：①理论保证假设初始化误差和后验期望估计误差已足够小；②对后验期望估计的 MCMC 采样误差仅在理论上被假设为 L² 控制，实际计算复杂度仍可能呈指数级；③未给出对 t₀、M、K 的具体数值设定指导；④缺乏大规模实验验证。

---

## 364. Mean Masked Autoencoder with Flow-Mixing for Encrypted Traffic Classification

**arXiv ID:** 2603.29537 | [PDF](https://arxiv.org/pdf/2603.29537v1)

**作者:** Xiao Liu `[一作]` (Chongqing University), Lei Zhang `[通讯]` (Chongqing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 Mean Masked Autoencoder (MMAE) 框架，利用教师-学生自蒸馏与流混合（FlowMix）实现对加密网络流量的多粒度预训练。

**💡 创新点**

创新点包括将自蒸馏与流级语义监督结合，设计动态流混合策略以构造跨流干扰样本，以及 Packet-importance aware Mask Predictor（PMP）根据包级侧信道信息动态生成遮掩。

**🔧 技术方法**

主要技术包括 Transformer 编码器/解码器、EMA 自蒸馏、统计匹配的流匹配器（SFM）、动态混合掩码、低秩注意力偏置等。

**📊 数据集**

使用六个公开加密流量数据集：ISCXVPN2016、ISCXTor2016、CrossPlatform(Android/iOS)、USTC-TFC2016、CICIoT2022 及 CSTNET-TLS1.3。

**📈 对比分析**

在七个数据集上与多种 SOTA 基线（ET-BERT、YaTC、TrafficFormer、NetMamba 等）对比，MMAE 在所有评测指标上均超越对手，尤其在 TLS1.3、IoT 与跨平台场景中获得接近 1.0 的 F1 得分。

**⚠️ 局限性**

主要局限是 PMP 对包间时延与大小侧信道的依赖，易受攻击者抹除；预训练阶段对显存和算力需求高，难以在边缘设备上在线学习。

---

## 365. The Geometry of Polynomial Group Convolutional Neural Networks

**arXiv ID:** 2603.29566 | [PDF](https://arxiv.org/pdf/2603.29566v1)

**作者:** Yacoub Hendi `[一作]` (Uppsala University), Magdalena Larfors `[通讯]` (Uppsala University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

研究了多项式群卷积神经网络（PGCNN），并引入了基于分级群代数的新数学框架。

**💡 创新点**

提出了两种自然的架构参数化方法，基于Hadamard和Kronecker积，并通过线性映射相互关联。

**🔧 技术方法**

使用了分级群代数的数学工具，结合神经代数几何的概念。

**📊 数据集**

使用了任意有限群G的相关数据集。

**📈 对比分析**

通过计算雅可比矩阵的秩来比较方法，结果表明PGCNN的神经流形维度仅依赖于层数和群的大小，而与激活函数的度数无关。

**⚠️ 局限性**

限制在于对于更大群体和更深网络的计算变得不可行，因为雅可比矩阵的大小至少是群体大小的平方，且与层数呈指数关系。

---

## 366. GraSP-STL: A Graph-Based Framework for Zero-Shot Signal Temporal Logic Planning via Offline Goal-Conditioned Reinforcement Learning

**arXiv ID:** 2603.29533 | [PDF](https://arxiv.org/pdf/2603.29533v1)

**作者:** Ancheng Hou `[一作]` (Shanghai Jiao Tong University), Xiang Yin `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于图搜索的零样本离线STL规划框架GraSP-STL，能够仅凭任务无关的离线轨迹数据生成满足任意未见STL规范的控制策略。

**💡 创新点**

创新点在于通过离线学习的目标条件可达性度量构建有向图抽象，并利用AGM鲁棒性区间在图上进行增量STL评估，从而实现零样本、长时程的STL规划。

**🔧 技术方法**

主要技术包括离线目标条件强化学习（学习可达性值函数与策略）、AGM鲁棒性及区间语义的STL评估、图构造与方向多样性采样以及基于图的搜索与最优子路径剪枝。

**📊 数据集**

实验采用OGBench的AntMaze离线数据集（10,000条长度500的轨迹），并在12种模板下随机生成2,400个不同的STL任务进行评估。

**📈 对比分析**

与现有离线STL规划方法对比，GraSP-STL在95.58%规划成功率、81.58%执行成功率下平均规划时间仅6.53秒，表现出色，尤其在长时程与分支结构任务上优势明显。

**⚠️ 局限性**

局限性包括对离线数据覆盖度高度依赖、离散图抽象可能导致时间紧张的计划缺乏裕度、以及在全局约束（如持久性任务）下执行成功率下降。

---

## 367. Design and Aerodynamic Modeling of MetaMorpher: A Hybrid Rotary andFixed-Wing Morphing UAV

**arXiv ID:** 2603.29646 | [PDF](https://arxiv.org/pdf/2603.29646v1)

**作者:** Anja Bosak `[一作]` (University of Zagreb), Stjepan Bogdan `[通讯]` (University of Zagreb)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计了MetaMorpher这款可变形无人机的概念架构，并基于MATLAB/Simulink开发了一个通用的非线性6-自由度飞行动力学模型，能够同时模拟旋翼悬停与固定翼巡航两种模式，模型通过将机翼离散为若干段，分别计算每段的气动力和力矩，实现分布式气动与机体耦合。

**💡 创新点**

创新点在于：①采用分段机翼几何与气动参数化，捕捉机翼在不同形态下的局部气动效应；②构建统一的非线性动力学框架，可一次性评估旋翼与固定翼两种飞行模式；③将模型开放源码发布，支持快速设计迭代和后续控制算法验证。

**🔧 技术方法**

使用的技术包括：MATLAB/Simulink与Aerospace Blockset进行动力学建模；XFLR5进行机翼气动系数（Cl, Cd, Cm）计算并生成查找表；自定义MATLAB函数实现机翼离散、角度转换、力矩计算；地面接触模型保证悬停起飞阶段的物理约束。

**📊 数据集**

使用的数据集主要是从XFLR5仿真获得的两种机型（Eppler E387折反弧形状用于巡航，NACA 0010对称形状用于悬停）在不同雷诺数与攻角下的Cl、Cd、Cm查找表；未使用外部公开数据集。

**📈 对比分析**

比较方法：在Simulink中分别仿真悬停起飞、悬停保持、机翼旋转导致的滚转与偏航响应等典型工况，并通过对单段力与全机力矩的时域曲线评估模型的物理一致性与可控性。结果表明模型能够稳定捕捉旋转升力产生、机翼分段力分布变化以及滚转/偏航耦合效应，显示出与预期物理行为一致，但尚未进行闭环控制验证或与实验数据直接对比。

**⚠️ 局限性**

局限性包括：①模型仅在模拟环境中验证，缺乏实际硬件实验验证；②推进系统简化为线性推力模型，未考虑电机、螺旋桨耦合效应；③缺乏闭环控制策略，无法评估实时姿态控制与模式切换性能；④偏航稳定性依赖外部舵面或控制器，当前机体本身不具备自然偏航稳定；⑤分段离散的细化程度需要进一步验证对不同机翼尺寸与形态的适用性。

---

## 368. Disentangled Graph Prompting for Out-Of-Distribution Detection

**arXiv ID:** 2603.29644 | [PDF](https://arxiv.org/pdf/2603.29644v1)

**作者:** Cheng Yang `[一作]` (Beijing University of Posts and Telecommunications), Chuan Shi `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

利用预训练图神经网络(GNN)编码器，设计两种提示图生成器（类别特定与类别无关），并通过交叉熵、均匀分布约束及距离正则化等损失训练生成器，实现对ID图细粒度模式的分离，从而提高图数据的OOD检测能力。

**💡 创新点**

创新点在于：①在预训练+提示（prompting）框架下首次引入双重提示图，兼顾类别特定与类别无关信息；②通过标签信息引导提示生成器挖掘细粒度ID模式；③引入距离正则化防止提示图陷入平凡解；④不需要对预训练模型做fine‑tune，显著提升效率与性能。

**🔧 技术方法**

主要技术包括：图神经网络预训练（GCL、SimGRACE）、提示图生成（利用MLP对边权重学习）、多任务损失（交叉熵、均匀分布约束、Mahalanobis距离正则化）、基于Mahalanobis距离的OOD评分函数、实验评估指标（AUC、AUPR、FPR95）。

**📊 数据集**

使用十对ID/OOD数据集（如BZR/COX2、PTC_MR/MUTAG、AIDS/DHFR、ENZYMES/PROTEIN、IMDB-M/IMDB-B、Tox21/SIDER、FreeSolv/ToxCast、BBBP/BACE、ClinTox/LIPO、Esol/MUV），覆盖分子、社交网络、生命信息等多种图域。

**📈 对比分析**

与非图方法、基于预训练的微调、图异常检测方法以及多种SOTA图OOD方法（GOOD‑D、AAGOD、SEGO、HGOE等）进行对比。DGP在大多数数据集上均取得AUC提升（平均提升约3.6%），在8/10个数据集上取得SOTA表现；训练速度显著快于大多数对比方法，且无需对GNN做额外fine‑tune。

**⚠️ 局限性**

局限性包括：①需要有预训练好的GNN且预训练过程耗时；②仅设计了两种提示图，可能无法充分挖掘所有细粒度模式；③实验主要在中等规模图上验证，对超大规模图的可扩展性未做系统评估；④对无节点/边特征的图的适用性尚未充分探究。

---

## 369. ASI-Evolve: AI Accelerates AI

**arXiv ID:** 2603.29640 | [PDF](https://arxiv.org/pdf/2603.29640v1)

**作者:** Weixian Xu `[一作]` (Shanghai Jiao Tong University), Pengfei Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了ASI‑Evolve框架，实现AI在自身研发过程中的闭环自我加速，包括学习–设计–实验–分析循环，并通过认知库与分析器支持长期高成本实验；

**💡 创新点**

核心创新在于统一的AI‑for‑AI框架，首次在模型架构、数据与算法三大核心领域实现AI驱动的突破；引入认知库把人类先验注入每轮搜索，并设计专门的分析器将复杂实验结果转化为可复用的洞见；

**🔧 技术方法**

采用进化搜索与大语言模型相结合的代理系统，利用嵌入检索构建认知库、LLM‑as‑a‑Judge进行质量评估，结合多阶段评估、MAP‑Elites/UCB1等采样策略、结构化分析器以及代码级差分编辑；

**📊 数据集**

使用的公开数据集包括：Transformer线性注意力基准、FineWeb‑Edu/Ultra‑FineWeb/DCLM/Nemotron‑CC等预训练语料、数学推理基准AMC32/AIME24/OlympiadBench、圆形打包基准、以及药物‑靶点交互数据集BindingDB、Human、BioSNAP、C.elegans；

**📈 对比分析**

通过与AlphaEvolve、OpenEvolve、GEPA、SkyDiscover等现有进化框架以及GRPO等算法基线进行对比；在圆形打包任务中仅17轮即可达到SOTA；在线性注意力设计中发现105种SOTA模型，最佳比DeltaNet提升0.97分；在数据策划中平均提升3.96分，MMLU提升18.64分；在RL算法设计中超越GRPO最高12.5分；在DTI任务中AUROC提升约1.9分；整体表现均达到或超过人类设计的SOTA；

**⚠️ 局限性**

局限性包括：对硬件加速的低效实现（缺乏CUDA层面优化）；对大规模实验仍需巨量算力；认知库依赖先验文献，难以完全迁移到全新领域；在极端多样化任务上仍需改进多维反馈解读能力；此外，当前框架尚未覆盖完整AI栈的基础设施与安全治理层面。

---

## 370. 5G Puppeteer: Chaining Hidden Command and Control Channels in 5G Core Networks

**arXiv ID:** 2603.29636 | [PDF](https://arxiv.org/pdf/2603.29636v1)

**作者:** Julian Sturm `[一作]` (ZITiS), Wolfgang Kellerer `[通讯]` (Technical University of Munich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文提出5G Puppeteer框架，利用5G核心网络中的标准接口和消息实现隐蔽的命令与控制通道，并展示了键提取、定位和公共警报滥用等攻击场景。

**💡 创新点**

创新点在于：①在5G协议内部挖掘可嵌入的可选字段、转发消息与临时消息，构造可链式、可分片的隐蔽通道；②设计轻量级5gpp协议，并提出多种路由（Path Flooding、Round Robin、Estimate‑Enhanced RR）与TTL机制；③通过多路径链路实现跨节点的命令与控制。

**🔧 技术方法**

使用技术包括网络隐写（steganography）在5G消息中嵌入数据；5G核心网络架构与接口分析；Python+NetworkX进行仿真和可视化；对称加密（AES）用于5gpp负载加密；对比多种路由与分片方案。

**📊 数据集**

采用作者自行构造的5G核心网络协议消息序列与容量参数（以注册流程为例），未使用公开的大规模数据集；仿真环境基于Python实现，包含消息可用空间和受损节点集合。

**📈 对比分析**

通过仿真评估不同嵌入容量对攻击完成所需注册次数的影响，并量化5gpp头部开销。结果显示：当每条消息可嵌入48bit以上时，绝大多数攻击可在少于15次注册内完成；头部开销低于20%（最高可达90%），表明方案在效率与隐蔽性之间取得良好平衡。

**⚠️ 局限性**

局限性包括：仅仿真了注册流程，未覆盖所有5G接口；依赖攻击者一次性获得核心节点控制权；检测方法需针对性细粒度，仍面临高误报/漏报风险；若标准进一步修改，可阻止临时消息攻击。

---

## 371. Semantic Zone-Based Map Management for Stable AI-Integrated Mobile Robots

**arXiv ID:** 2603.29627 | [PDF](https://arxiv.org/pdf/2603.29627v1)

**作者:** Huichang Yun `[一作]` (Pukyong National University), Seungho Yoo `[通讯]` (Pukyong National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出语义区（如房间、走廊）为单位的关键帧管理方法，结合预测驱动的加载/卸载策略，在边缘设备内存受限条件下稳定稠密地图使用，并与大规模视觉语言模型协同工作。

**💡 创新点**

创新点在于：①将关键帧按语义区域分组并在区域层面批量管理；②使用机器人当前姿态与路径预测未来需要加载的区域；③在加载前预估关键帧总数，若超出阈值则先批量卸载不相关区域，实现严格内存预算控制，显著降低加载频率和内存溢出风险。

**🔧 技术方法**

使用技术包括：RTAB‑Map SLAM框架、基于语义标签的关键帧元数据管理、Zone‑Based 内存管理算法、LRU 预测卸载策略、Jetson Orin Nano 边缘硬件、以及 Qwen3.5/Gemma 等大规模视觉语言模型。

**📊 数据集**

实验基于 NVIDIA Isaac Sim 生成的大型室内场景（医院场景）构建的稠密 3D 地图，人工划分语义区域；未使用公开数据集。

**📈 对比分析**

与 RTAB‑Map 默认几何距离驱动的关键帧加载做对比；在 Jetson Orin Nano 上同时跑 SLAM 与 VLM；评估指标包括更新率、循环闭合次数、平均处理时延、ATE/RPE 误差以及 VLM 的 token/s 与延迟。结果显示，语义区方法在与 VLM 并行时：更新率与循环闭合几乎不变；内存溢出与停滞问题被消除；VLM token/s 提升 3.3，延迟降低 21.7%，整体性能明显优于几何策略。

**⚠️ 局限性**

局限性包括：语义区域划分依赖人工，未实现自动或动态细化；卸载策略仅采用 LRU，未考虑任务语义重要性或轨迹预测；当内存被占满时仍可能因关键帧无法及时加载导致循环闭合失败；缺乏长期部署与检索命中率的系统性评估。

---

## 372. BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation

**arXiv ID:** 2603.29630 | [PDF](https://arxiv.org/pdf/2603.29630v1)

**作者:** Johann-Ludwig Herzog `[一作]` (BIFOLD and Technische Universität Berlin), Begüm Demir `[通讯]` (BIFOLD and Technische Universität Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个大规模多传感器（S1 SAR + S2 MS）图像‑文本数据集 BigEarthNet.txt，并提供了包含说明、VQA 与定位指令的多任务注释和手动验证的评估子集。

**💡 创新点**

创新点在于：①首次将 SAR 与 MS 图像共注册并配备多类型文本注释；②使用模板 + LLM 增广实现高质量多样化说明；③提供手工验证的评估集；④通过对 InternVL‑3‑1B 进行多模态微调，展示数据量足够时即使小规模模型亦可显著提升多传感器任务性能。

**🔧 技术方法**

技术手段包括：模板生成 + LLM 语言增广、自动化 VQA 与实例定位标注、ViT‑based 多模态编码器、LoRA 微调、以及基于多任务评估指标（BLEU、ROUGE、METEOR、CIDEr、BERTScore、SBERT‑Cosine、CLAIR）的系统性评测。

**📊 数据集**

使用的数据集为：基于 BigEarthNet v2.0 的 464,044 对 S1+S2 图像（经过滤无效对）构成的训练/验证/测试集；以及包含 1,082 对图像与 15,029 条手工验证文本的评估子集；与现有 RS 图像‑文本数据集（如 MS‑CLIP、RS5M 等）做对比。

**📈 对比分析**

评估方法采用多任务指标（captioning: BLEU/ROUGE/METEOR/CIDEr/BERTScore/SBERT‑Cosine/CLAIR；VQA: 准确率；定位: mIoU/precision 等）。结果显示：①一般 CV VLM 在 RGB 输入下对 RS 任务的表现优于专门的 RS VLM；②微调的 RS‑InternVL 在所有 15 个任务上平均提升 31.52 分，显著超越未微调模型。

**⚠️ 局限性**

局限性在于：当前 VLM 仍因 RGB 预训练缺乏对多光谱/多传感器信息的有效利用，导致在复杂 LULC 理解与多传感器任务上的泛化不足；此外，微调数据仍有限，进一步提升需要更大规模、多样化的多模态训练数据。

---

## 373. Bringing Up a Bilingual BabyLM: Investigating Multilingual Language Acquisition Using Small-Scale Models

**arXiv ID:** 2603.29552 | [PDF](https://arxiv.org/pdf/2603.29552v1)

**作者:** Linda Zeng `[一作]` (Harker School), Michael C. Frank `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用语言模型模拟不同的多语种学习环境，系统评估其对语言学习的影响

**💡 创新点**

在可控的实验设置中引入多种多语种输入结构（单语、平衡双语、代码切换等），探讨其对学习效果的影响，证明多语种输入不会导致主要语言的性能下降

**🔧 技术方法**

使用 GPT‑2 Small（124M 参数）以及 GPT‑2 Mini（39M 参数）模型，采用人工合成的儿童导向语料进行预训练

**📊 数据集**

生成 100M 单语英语对话数据并用 GPT‑4 翻译为西班牙语，构成 100M 英西双语并平衡的数据集，并进一步生成句子级与词级代码切换版本

**📈 对比分析**

通过困惑度（Perplexity）、Zorro 语法测试和词相似度（WS/X‑WS）评估，发现双语和代码切换模型在两种语言上的表现与单语基线相当，跨语言相似度更高，整体性能无显著下降

**⚠️ 局限性**

模型与人类学习者差异大，缺乏音位、语调等自然输入特征；合成语料与真实儿童输入仍有限；实验聚焦统计学习者，难以直接推广至人类认知机制

---

## 374. Distributed Predictive Control Barrier Functions: Towards Scalable Safety Certification in Modular Multi-Agent Systems

**arXiv ID:** 2603.29560 | [PDF](https://arxiv.org/pdf/2603.29560v1)

**作者:** Jonas Ohnemus `[一作]` (ETH Zurich), Melanie N. Zeilinger `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于分布式预测控制障碍函数（D-PCBF）的多智能体安全过滤框架，并实现了可插拔（Plug‑and‑Play）协议，能在网络拓扑变化时保证系统恢复安全。

**💡 创新点**

创新点包括：① 结构化控制障碍函数（s‑CBF）允许局部违反安全约束而整体保持递减；② 将s‑CBF嵌入在线分布式优化，扩展安全集并保证收敛；③ 基于松弛变量的安全与恢复双证书，实现无停机的在线可插拔操作。

**🔧 技术方法**

技术手段：分布式预测安全过滤、控制障碍函数设计、在线分布式优化求解、邻居通信协议、实车测试与仿真验证。

**📊 数据集**

数据集：仿真平台下的多车道小型赛车车队轨迹；真实硬件实验中使用的实车轨迹与传感器数据。

**📈 对比分析**

与现有分布式预测安全过滤器对比：本方法在高速可插拔场景下成功实现碰撞避免和安全恢复，而传统方法在拓扑变化时往往失效。实验结果显示，系统在临时违反安全约束后能快速回到安全集，整体性能优于基线方法。

**⚠️ 局限性**

局限性：对系统鲁棒性的理论保证尚未充分；s‑CBF 的合成对线性多极系统可能过于保守；仅考虑动力学耦合，未覆盖状态耦合约束；分布式求解器的计算时延与网络延迟影响尚未在硬件上系统评估。

---

## 375. HyperKKL: Learning KKL Observers for Non-Autonomous Nonlinear Systems via Hypernetwork-Based Input Conditioning

**arXiv ID:** 2603.29744 | [PDF](https://arxiv.org/pdf/2603.29744v1)

**作者:** Yahia Salaheldin Shaaban `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Karl Henrik Johansson `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于超网络的 KK L 观察器设计，针对受外部输入影响的非自治非线性系统实现状态估计

**💡 创新点**

1）引入两种输入调节策略：在潜在空间注入修正项（HyperKKL_obs）和动态生成时变变换映射（HyperKKL_dyn）；2）提供严格的最坏情况误差上界，揭示编码器、解码器近似误差、PDE 残差与解码器 Lipschitz 常数的关系；3）首次系统比较两种调节方案在多种输入情境下的效果

**🔧 技术方法**

超网络（Hypernetworks）+ GRU 时序编码 + MLP 注入层 + LoRA 权重微调 + PDE 正则化 + 训练时的自回归 Jacobian 计算 + 线性化控制矩阵 A、B 的预设

**📊 数据集**

四个经典非线性系统（Duffing、Van der Pol、Rössler、FitzHugh–Nagumo），对每个系统在四种输入（零、恒定、正弦、方波）下随机初始条件，使用 RK45 采样（Δt=0.05s，T=50s）产生训练和测试数据，加入高斯噪声

**📈 对比分析**

与自治 KKL 观察器和课程学习基线对比；采用 SMAPE 作为评价指标，100 次随机实验平均结果显示 HyperKKL_obs 在 13/16 组合中获得最佳或相当的性能，平均 SMAPE 降低约 29%；HyperKKL_dyn 在 2/16 组合中最佳；自治基线误差显著偏大，课程学习反而恶化

**⚠️ 局限性**

1）HyperKKL_dyn 对训练更为敏感，可能因权重微调导致 Lipschitz 常数增大；2）在高度混沌系统（Rössler）输入调节效果有限；3）当前方法仍依赖较大历史窗口和多层网络，计算开销较高；4）未考虑测量/过程噪声对解码器 Lipschitz 常数的实际影响

---

## 376. Editing on the Generative Manifold: A Theoretical and Empirical Study of General Diffusion-Based Image Editing Trade-offs

**arXiv ID:** 2603.29736 | [PDF](https://arxiv.org/pdf/2603.29736v1)

**作者:** Yi Hu `[一作]` (Xidian University), Finn Carter `[通讯]` (Xidian University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对通用扩散模型图像编辑进行统一理论与实证分析，提出控制性、忠实度、一致性、局部性、质量和多轮稳定性等目标，并给出相应的度量与理论边界。

**💡 创新点**

创新点在于：①把多种编辑范式（训练‑free、逆向‑编辑、指令跟随、拖拽、插入）统一为“在学习的图像流形上进行有向传输”；②在理论上推导了逆向错误放大、引导尺度与局部性耦合、以及多轮编辑的稳定性界限；③通过统一度量体系和跨范式对比实验，系统揭示了不同方法在可控性、忠实度与局部性之间的权衡。

**🔧 技术方法**

采用扩散/流匹配模型、分类器无关引导、注意力/特征剪切、逆向重建、梯度优化（拖拽）、多轮编辑跟踪、以及概念消除技术；同时使用LPIPS、CLIP、DINO、Win等指标进行评估。

**📊 数据集**

使用公开数据集包括COCO‑style 场景图像、FFHQ‑style 头像、MagicBrush、UltraEdit、DragBench 等，分别用于指令编辑、局部编辑、插入、拖拽和多轮测试。

**📈 对比分析**

实验对比多种代表方法，结果表明：训练‑free 方法速度快、局部性好但忠实度有限；逆向‑编辑在细节保持上表现优异，但易产生漂移；训练‑指令跟随在忠实度上最强，但对局部约束不够灵活；插入与拖拽方法在特定任务上表现突出。总体上不同方法在目标之间存在不可避免的折衷。

**⚠️ 局限性**

局限性包括：①对文本指令的语义歧义和提示敏感导致非局部变更；②高引导尺度下的身份漂移和纹理崩塌；③硬约束下的边界伪影；④逆向与优化过程计算成本高，影响交互性；⑤安全性问题（如身份泄露、概念再现）仍需进一步解决。

---

## 377. FED-Bench: A Cross-Granular Benchmark for Disentangled Evaluation of Facial Expression Editing

**arXiv ID:** 2603.29697 | [PDF](https://arxiv.org/pdf/2603.29697v1)

**作者:** Fengjian Xue `[一作]` (Xi'an Jiaotong University), Liang He `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了FED‑Bench基准集和FED‑Score评估方案，用于细粒度面部表情图像编辑的完整评估。

**💡 创新点**

创新点在于构建了747个高质量源图–指令–真值三元组，并提出三维解耦评估（对齐、保真、相对表情增益）来消除懒编辑与过拟合偏差。

**🔧 技术方法**

技术手段包括多模态大语言模型（如Gemini‑2.5‑Pro）、ArcFace、DINO、LPIPS、Grounded SAM等进行数据生成与多维度评估。

**📊 数据集**

数据来源于公开的SFEW 2.0、DFEW、RAF等，随后通过自动化管道扩展至20k+的训练对，形成完整的训练与测试集。

**📈 对比分析**

在18种主流编辑模型上进行对比，FED‑Score与人类评判高度一致，显示其评估可靠性；基准集揭示现有模型在保真与指令遵循上仍有明显瓶颈。

**⚠️ 局限性**

局限性包括对大语言模型的依赖、评估仍需人工校准，以及对极端表情或低光照场景的覆盖不足。

---

## 378. A First Step Towards Even More Sparse Encodings of Probability Distributions

**arXiv ID:** 2603.29691 | [PDF](https://arxiv.org/pdf/2603.29691v1)

**作者:** Florian Andreas Marwitz `[一作]` (University of Lübeck), Ralf Möller `[通讯]` (University of Lübeck)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出CoFE算法，通过减少分布中的不同潜在值并提取最小化的FOL公式，实现概率分布的更稀疏编码。

**💡 创新点**

在传统的指数级公式数量上，CoFE通过聚类或分位数的潜在值聚合，显著降低公式数量，同时控制分布偏差。

**🔧 技术方法**

使用分位数/聚类潜在值聚合、Hellinger距离约束、Quine‑McCluskey最小化，以及DBSCAN聚类等技术。

**📊 数据集**

实验使用真实的smokers数据集和一个人工构造的九个三元变量的parfactor模型。

**📈 对比分析**

与标准的逐项转换相比，CoFE将公式数量指数压缩，公式长度略增，且在噪声水平较低时平均绝对误差≤0.01；噪声增大时误差上升。

**⚠️ 局限性**

限制在于ε值的设定决定能否降维，且只评估了两种聚合策略；在高噪声或复杂模型下可能无法显著减少公式或误差增大。

---

## 379. KEditVis: A Visual Analytics System for Knowledge Editing of Large Language Models

**arXiv ID:** 2603.29689 | [PDF](https://arxiv.org/pdf/2603.29689v1)

**作者:** Zhenning Chen `[一作]` (Zhejiang University), Yingcai Wu `[通讯]` (Zhejiang University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套名为KEditVis的可视化分析系统，帮助用户在大型语言模型中进行交互式的知识编辑，支持层选择、编辑效果评估与多方案比较。

**💡 创新点**

创新点在于：①将层级可视化与层选择（余弦相似度、token投影）结合，直观展示编辑范围与模型内部变化；②利用集合可视化（wireframe）高效呈现不同编辑方案的层集合关系；③通过交互式流程让人机协作完成编辑，显著提升编辑质量并降低全局影响。

**🔧 技术方法**

主要技术包括：Transformer LLM权重编辑（MEMIT/AlphaEdit等本地修改方法）、余弦相似度与token概率可视化、t‑SNE散点图用于全局漂移评估、交互式set可视化算法、用户体验评估与SUS量表。

**📊 数据集**

数据集：使用公开的CounterFact数据集进行知识编辑任务；实验中还构造了基于知识图的问答和测试提示；用户研究使用了12名具备不同背景的参与者进行实操。

**📈 对比分析**

与固定层、自动层选择及自动化+筛选三种基线比较，实验显示由参与者手工选择的编辑方案在编辑分数（S）、成功率、生成熵等指标上显著优于所有基线（p<0.001），且全局模型相似度保持在>0.99，未出现毒性或性能下降。

**⚠️ 局限性**

局限性：①系统依赖人机交互，难以量化人力与自动化的投入平衡；②编辑规模扩大（多事实、多层）会导致方案数激增、计算成本上升；③当前仅验证在Transformer LLM上的本地修改方法，其他编辑技术或更大模型的适配需进一步研究。

---

## 380. Clinical DVH metrics as a loss function for 3D dose prediction in head and neck radiotherapy

**arXiv ID:** 2603.29670 | [PDF](https://arxiv.org/pdf/2603.29670v1)

**作者:** Ruochen Gao `[一作]` (Leiden University Medical Center), Frank Dankers `[通讯]` (Leiden University Medical Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了临床DVH指标损失函数并结合无损失比特掩码编码，实现了针对头颈部放疗的3D剂量预测。

**💡 创新点**

创新点在于将D–和V–指标直接转化为可微分损失，提出α参数的理论下限选择，以及使用单整数比特掩码高效编码重叠ROI。

**🔧 技术方法**

使用3D U‑Net网络、MAE+CDM联合损失、混合精度训练、Top‑k排序、逻辑斯蒂 sigmoid近似V‑指标、GPU位运算解码。

**📊 数据集**

采用Leiden大学医学院174例头颈部VMAT病例，按时间划分137例训练、37例测试。

**📈 对比分析**

与MAE、MAE+DVH等损失比较，MAE+CDM在PTV覆盖率上最优，PTV Score从1.544降至0.491，OAR约保持不变，训练时间/显存显著下降。

**⚠️ 局限性**

局限包括单机构单协议数据，缺乏多中心验证；未评估剂量模仿及最终可交付计划质量；对α阈值的选择仍需经验。

---

## 381. CoRe-DA: Contrastive Regression for Unsupervised Domain Adaptation in Surgical Skill Assessment

**arXiv ID:** 2603.29666 | [PDF](https://arxiv.org/pdf/2603.29666v1)

**作者:** Dimitrios Anastasiou `[一作]` (University College London), Evangelos B. Mazomenos `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了基于对比回归的无监督领域适应框架 CoRe‑DA，用于外科手术技能评估（SSA）回归，并创建了首个 SSA 无监督领域适应基准。

**💡 创新点**

将对比回归与相对评分监督、伪标签自训练和一致性约束结合，首次把无监督领域适应引入 SSA 回归任务，显著提升跨域泛化性能。

**🔧 技术方法**

使用 I3D 特征编码器、绝对与相对回归器（3 层 MLP）、对比回归损失、目标域自训练伪标签、训练时背景混合等技术，训练采用 Adam 优化器。

**📊 数据集**

使用四个公开数据集：AIxSuture、JIGSAWS、RAH‑skill、RARP‑skill，覆盖干实验室、机器人和临床手术场景，构成两个跨域设置。

**📈 对比分析**

与 MDD、DARE‑GRAM、RSD、CO²A 等 UDA 基线以及 ViSA、Contra‑Sformer 等 SSA 模型对比，CoRe‑DA 在两套任务中分别提升 Spearman 相关系数 +0.13/0.26、MAE 降低 0.32/0.49，SCC 最终达 0.46/0.41。

**⚠️ 局限性**

仍需大量标注源域数据，伪标签噪声易影响学习；对极端域差仍有限制；未加入不确定性估计和多模态输入的考虑。

---

## 382. An Empirical Comparison of Security and Privacy Characteristics of Android Messaging Apps

**arXiv ID:** 2603.29668 | [PDF](https://arxiv.org/pdf/2603.29668v1)

**作者:** Ioannis Karyotakis `[一作]` (AUEB & NTUA), Nikolaos Alexopoulos `[通讯]` (AUEB)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了安卓主流即时通讯应用（Meta Messenger、Signal、Telegram）的安全与隐私实现特性，采用静态与动态分析方法比较其差异。

**💡 创新点**

提出了一种可复现的混合方法，将静态代码审计与低层内核跟踪结合，克服了证书绑定难题，系统评估了攻击面、权限、网络行为。

**🔧 技术方法**

主要技术包括MobSF、Androguard、SliceDroid、tcpdump、kprobe以及自研的网络事件映射，使用Kruskall‑Wallis和Mann‑Whitney等统计检验。

**📊 数据集**

使用官方生产签名APK（Messenger 523.0.0.53.109、Signal 7.54.1、Telegram 12.0.1）以及在Nothing Phone 2a（Android 14）上多次实验产生的网络流量与系统调用日志。

**📈 对比分析**

通过对静态指标（代码量、导出组件、权限、警告）和动态指标（TCP/UDP流量、对等方数、前后台、完整/受限权限）的定量对比，发现Signal最小化设计、Messenger攻击面最大、Telegram最危险权限；性能表现为Signal网络占用最少、Messenger最多。

**⚠️ 局限性**

局限性包括仅评估Android客户端、实验仅在单一欧盟地区、只覆盖三款主流应用、网络分析无法解密加密内容，且统计样本有限。

---

## 383. CausalPulse: An Industrial-Grade Neurosymbolic Multi-Agent Copilot for Causal Diagnostics in Smart Manufacturing

**arXiv ID:** 2603.29755 | [PDF](https://arxiv.org/pdf/2603.29755v1)

**作者:** Chathurangi Shyalika `[一作]` (University of South Carolina), Amit Sheth `[通讯]` (University of South Carolina)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并部署了一款面向工业制造的多智能体协同拯救故障诊断平台CausalPulse，整合异常检测、因果发现与根因分析，实现实时、可解释的根因洞察；

**💡 创新点**

提出了基于标准化代理协议的 neurosymbolic 多智能体架构，融合领域规则约束的因果学习、路径评分的根因推理以及动态任务规划与自我反思机制；

**🔧 技术方法**

使用FastAPI微服务+LangGraph+Agent Communication Protocol(A2A)与Model Context Protocol(MCP)实现代理编排，结合PC/GES因果结构学习、ProRCA路径评分、跨模态异常检测模型与知识图谱；

**📊 数据集**

在德国博世生产线收集的Planar Sensor Element（PSE）数据以及公共Future Factories（FF）数据集上进行评测；

**📈 对比分析**

与现有工业copilot（UCSD、MATMCD、IBM AssetOpsBench、SmartPilot）做定性对比，且在PSE/FF上实现整体成功率98.0%–98.73%，单项指标规划/工具使用/自我反思/协作均≥95%；runtime每条诊断流程≈50–60秒，规模线性扩展(R²=0.97)；

**⚠️ 局限性**

在根因精确度上仍有提升空间（Hits@1≈0.44，F1≈0.39），对异常类型覆盖有限，且系统对实时数据同步、边缘部署与安全性监管的细节处理尚待进一步验证；

---

## 384. SHIFT: Stochastic Hidden-Trajectory Deflection for Removing Diffusion-based Watermark

**arXiv ID:** 2603.29742 | [PDF](https://arxiv.org/pdf/2603.29742v1)

**作者:** Rui Bao `[一作]` (University of New South Wales), Jiaojiao Jiang `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 SHIFT，一种训练自由、无水印知识的无监督攻击方法，利用部分前向扩散和随机逆采样在扩散模型中打乱轨迹一致性，从而剥夺扩散水印的可验证性。

**💡 创新点**

创新点在于：①首次发现所有扩散水印都依赖轨迹一致性这一共性弱点；②提出通过随机逆采样解耦轨迹的 SHIFT 框架；③在理论层面给出 Wasserstein 距离证明，实验证明对九种不同范式的水印均能实现 95–100% 的攻击成功率，并保持最高图像质量。

**🔧 技术方法**

主要技术包括：Stable Diffusion v2.1 预训练模型、部分前向扩散（控制噪声强度 λ）、Euler 先祖随机逆采样、DDIM 逆解、Wasserstein 距离分析和 CLIP/FID 评价指标。

**📊 数据集**

使用 Stable‑Diffusion‑Prompts 数据集作为生成提示，并对九种代表性扩散水印（Tree‑Ring、RingID、PRC、WIND、Gaussian Shading、GaussMarker、SFW、SEAL、ROBIN）进行评估。

**📈 对比分析**

与无攻击（Clean）、黑盒攻击和仿冒攻击对比，评估指标为攻击成功率（ASR）、CLIP 分数和 FID。SHIFT 的平均 ASR 为 97.8%，显著高于黑盒 96.8% 和仿冒 81.3%，且在 CLIP 上最高、FID 最低，说明既能彻底破坏水印又保持最高图像质量。

**⚠️ 局限性**

局限性在于：重噪强度 λ 需要在一定范围内平衡攻击成功与细节保留，对最鲁棒的水印仍需较大 λ，且目前仅针对图像扩散模型，未验证对视频或其他生成框架的适用性。

---

## 385. A Graded Modal Dependent Type Theory with Erasure, Formalized

**arXiv ID:** 2603.29716 | [PDF](https://arxiv.org/pdf/2603.29716v1)

**作者:** Andreas Abel `[一作]` (University of Gothenburg and Chalmers University of Technology), Oskar Eriksson `[通讯]` (University of Gothenburg and Chalmers University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

构建了一种带有可调度分级模态的依赖类型理论，并在 Agda 中完成了完整的形式化与元理论证明；在此基础上实现了编译器将源程序提取为无类型 λ 计算机，并给出了提取的可证明正确性（保留值）。

**💡 创新点**

创新点包括：① 将模态结构推广为部分有序半环，从而统一描述线性、不可变、信息流等多种量化属性；② 引入了分离的使用语义（γ‑typing）与传统类型判定的方式，实现了自适应的资源计数；③ 通过逻辑关系证明了在一致上下文和禁止擦除匹配的前提下，提取函数对自然数程序的语义不变；④ 在形式化层面实现了强 Σ、弱 Σ、强/弱单位、可选擦除匹配等一系列扩展。

**🔧 技术方法**

技术方法主要包括：
- 基于部分有序半环的模态算子和“函数”运算符来定义资源计数规则；
- 采用 Kripke 逻辑关系（syntactic + reducibility）来证明类型系统的规范性、归约一致性和可判定性；
- 在提取阶段使用分级裁剪策略（根据 0/非0 级别删除无关代码），并给出严格与非严格两种目标运行时模型；
- 构造了一个完整的 Agda formalization，包含所有证明与实现。

**📊 数据集**

由于研究的是理论系统，没有使用任何外部数据集，所有证明均在 Agda 内部完成。

**📈 对比分析**

没有进行实验性性能对比；相比以往工作，该系统实现了更完整的形式化（约 110k 行 Agda 代码）并提供了针对不同模态的可配置性；理论上，提取后的程序在一致上下文下保持相同的自然数值。

**⚠️ 局限性**

限制与未解决的问题：
- 需要一致上下文且禁止擦除匹配，才能保证提取的可证明正确性；
- 对于包含非一致假设或擦除匹配的程序，提取不再保证值不变；
- 模态结构的“函数”运算符需要手工定义，若不满足特定不等式，可能导致资源计数不准确；
- 未覆盖某些会影响计算式判定的模态（如证明无关性、可变性等）。

---

## 386. Exploring the Impact of Skin Color on Skin Lesion Segmentation

**arXiv ID:** 2603.29694 | [PDF](https://arxiv.org/pdf/2603.29694v1)

**作者:** Kuniko Paxton `[一作]` (University of Hull), Yiannis Papadopoulos `[通讯]` (University of Hull)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了皮肤色素与皮肤病变分割性能的关系，系统评估了不同肤色与分割准确率的关联；

**💡 创新点**

创新点在于使用连续的ITA分布及Wasserstein距离量化病变与皮肤的色彩对比，而非传统的离散肤色分组，从而更细致地捕捉对分割误差的影响；

**🔧 技术方法**

采用UNet、DeepLabV3与Vision Transformer DINOv2三种语义分割网络，并通过ITA计算与WD距离进行分析；

**📊 数据集**

使用公开的HAM10000和ISIC2017两大皮肤病变数据集；

**📈 对比分析**

通过Spearman相关系数、置信区间以及多指标（IoU、DC、PA、AUC等）比较，发现全球肤色与分割质量相关性弱，但病变-皮肤对比度与多项评估指标高度相关，低对比度图像导致分割错误显著增加；

**⚠️ 局限性**

局限在于数据集暗肤色样本稀缺，导致结果在更广泛肤色分布下可能不具代表性。

---

## 387. Measuring the metacognition of AI

**arXiv ID:** 2603.29693 | [PDF](https://arxiv.org/pdf/2603.29693v1)

**作者:** Richard Servajean `[一作]` (RIKEN), Philippe Servajean `[通讯]` (Paul-Valéry University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了三款大型语言模型（GPT-5、DeepSeek-V3.2-Exp、Mistral-Medium-2508）的元认知敏感度和基于风险的决策调节，采用meta‑d'与信号检测理论进行实验分析。

**💡 创新点**

创新点在于首次将meta‑d'框架和模型无关的替代方法引入AI元认知评估，并通过风险配置展示LLM能够根据不确定性与风险自我调节决策的能力。

**🔧 技术方法**

使用信号检测理论（SDT）、meta‑d' 统计模型（HMeta‑d）以及阈值c的校准分析，对LLM的置信度和决策做定量评估。

**📊 数据集**

使用SST‑2情感分析数据集、口语与书面分类公开数据集以及自定义的“the”删减检测数据集进行实验。

**📈 对比分析**

通过比较不同模型、不同任务以及与理论最优的meta‑d'与M_ratio，发现GPT‑5、DeepSeek‑V3.2‑Exp和Mistral‑Medium‑2508均表现出显著但低于最优的元认知效率，M_ratio范围约0.65–0.90；c‑校准实验显示GPT‑5能在不同风险配置下显著调节其判别阈值，体现出风险敏感的决策调节。

**⚠️ 局限性**

局限性包括仅评估简单二分类任务，未涉及探索‑利用权衡、幻觉或跨代理信任整合等复杂场景；实验设计对结果有较大影响，未构建完整基准，且结果在不同模型参数或提示方式下可能产生显著差异。

---

## 388. A Comprehensive Information-Decomposition Analysis of Large Vision-Language Models

**arXiv ID:** 2603.29676 | [PDF](https://arxiv.org/pdf/2603.29676v1)

**作者:** Lixin Xiu `[一作]` (University of Tokyo), Hideki Nakayama `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并应用基于部分信息分解（PID）的框架，对26个大型视觉-语言模型在四个多项选择VQA基准上的决策过程进行量化分析，揭示任务、模型族、层级和训练阶段的融合与语言先验策略。

**💡 创新点**

首次在大规模视觉-语言模型上实现PID估计，构建可跨模型、跨任务、跨层级、跨训练阶段的统一量化分析方法，系统识别出两类信息使用范式（协同驱动 vs 知识驱动）和两类模型族策略（融合中心 vs 语言中心），并揭示融合在训练中的关键发展阶段。

**🔧 技术方法**

使用可扩展的BATCH PID估计器，结合多模态输入的平均池化特征、噪声掩蔽的单模态推断、置信度阈值化和软聚合输出；采用logit lens跟踪层级信息流；对两阶段训练（对齐预训练 + 视觉指令微调）进行时间演化分析。

**📊 数据集**

MMBench、POPE、Reefknot、PMC-VQA四个多项选择VQA数据集；为PID估计拆分为训练集与评估集，使用随机 3:1 切分；对所有模型使用公开的 HuggingFace checkpoints。

**📈 对比分析**

在跨模型、跨任务比较中将PID分量（冗余 R、视觉独特 U1、语言独特 U2、协同 S）与模型准确率、图像消除干预结果进行相关分析；发现协同 S 与准确率在协同驱动任务中的相关性最高（ρ≈0.75），而在知识驱动任务中语言独特 U2 更重要；进一步通过层级 PID 观察到三阶段信息流模式，并在训练阶段显示协同 S 在视觉指令微调阶段显著提升。

**⚠️ 局限性**

PID 需要离散目标空间，无法覆盖开放式生成任务；单模态掩蔽采用噪声近似，未使用真正单模态输入；PID 结果为相关性而非因果关系；对模型规模和任务的泛化性仍有待进一步验证。

---

## 389. Generalized Resistance Geometry from Kron Reduction and Effective Resistance

**arXiv ID:** 2603.29675 | [PDF](https://arxiv.org/pdf/2603.29675v1)

**作者:** Yosuke Kajiura `[一作]` (University of Tokyo), Kazuhiro Sato `[通讯]` (University of Tokyo)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文基于Kron约简和有效电阻，发展了一种广义的电阻几何理论，适用于有向图，特别是强连通有向图。

**💡 创新点**

创新点在于证明了Fiedler-Bapat身份，连接了电阻矩阵和拉普拉斯矩阵，并定义了电阻曲率和电阻半径，扩展了无向图的相关概念。

**🔧 技术方法**

使用了Kron约简、有效电阻、Fiedler-Bapat身份等数学工具。

**📊 数据集**

研究中使用了强连通有向图的拉普拉斯矩阵和电阻矩阵。

**📈 对比分析**

通过与经典无向图的比较，展示了电阻曲率和电阻半径的性质，并证明了在强连通权重平衡情况下，Kron约简与拉普拉斯的伪逆操作是可交换的。

**⚠️ 局限性**

限制在于该理论主要集中在强连通有向图和签名无向图的框架内，可能无法直接推广到其他类型的图结构。

---

## 390. Agenda-based Narrative Extraction: Steering Pathfinding Algorithms with Large Language Models

**arXiv ID:** 2603.29661 | [PDF](https://arxiv.org/pdf/2603.29661v1)

**作者:** Brian Felipe Keith-Norambuena `[一作]` (Universidad Católica del Norte), Joshua Emanuel Leyton-Vallejos `[通讯]` (Universidad Católica del Norte)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于议程的叙事提取方法，利用大语言模型在 Narrative Trails 的最大瓶颈路径搜索中引导叙事路线。

**💡 创新点**

创新点在于将用户自然语言议程与最大瓶颈路径算法结合，既保持高连贯性又实现多视角交互的叙事提取。

**🔧 技术方法**

采用 LLM（Gemma 3 12B、Claude Opus 4.5、GPT 5.1）进行候选文档排名，结合 Sentence‑BERT 嵌入、UMAP 投影、HDBSCAN 聚类和连贯性图构建技术。

**📊 数据集**

使用包含 418 篇关于 2021 年古巴抗议的新闻文章的数据集进行实验。

**📈 对比分析**

与无 LLM 的最大瓶颈路径基线以及 TF‑IDF 关键词匹配进行比较，LLM 驱动在语义议程上的对齐提升 9.9%，连贯性仅下降 2.2%，在简单议程下关键词匹配更优。

**⚠️ 局限性**

局限性包括仅在单一古巴抗议语料上评估、议程数量有限、仅基于 LLM 评审而未进行人类评估，以及计算延迟较高。

---

## 391. BotVerse: Real-Time Event-Driven Simulation of Social Agents

**arXiv ID:** 2603.29741 | [PDF](https://arxiv.org/pdf/2603.29741v1)

**作者:** Edoardo Allegrini `[一作]` (Sapienza University of Rome), Marinella Petrocchi `[通讯]` (IIT-CNR)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个可扩展、事件驱动的多智能体社交模拟框架（BotVerse），在安全的沙盒环境中模拟基于LLM的社交机器人对信息传播和舆论的影响。

**💡 创新点**

创新点在于：①将实时Bluesky内容作为上下文注入，实现高保真动态响应；②采用事件驱动执行模型取代传统迭代，提升可扩展性；③使用数字DNA+记忆权重（α·recency+β·importance）构建智能体行为与记忆机制；④多模态内容生成与可插拔LLM后端，支持跨平台实验。

**🔧 技术方法**

技术包括多智能体系统（MAS）、FastAPI、React/TypeScript前端、PostgreSQL持久化、Stable Diffusion图像生成、LLM（GPT、DeepSeek）、事件驱动架构、数字DNA与记忆模块。

**📊 数据集**

数据来源为实时Bluesky（AtProto）内容流，实验中使用500名智能体（350可信、150恶意），没有使用传统公开静态数据集。

**📈 对比分析**

论文未给出传统基准对比实验，仅展示三阶段（种子、放大、分析）仿真过程；框架可支持数千并发智能体，实时可视化性能良好。

**⚠️ 局限性**

局限性包括：仍处于实验阶段，缺乏跨平台验证；LLM生成内容质量受后端限制；记忆权重参数的调优缺乏系统化；仅基于Bluesky数据，外推到其他平台的可靠性有限。

---

## 392. GRVS: a Generalizable and Recurrent Approach to Monocular Dynamic View Synthesis

**arXiv ID:** 2603.29734 | [PDF](https://arxiv.org/pdf/2603.29734v1)

**作者:** Thomas Tanay `[一作]` (Huawei Noah's Ark Lab), Eduardo Pérez-Pellitero `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了GRVS模型，能够从单目动态视频中生成自由视角的新视图，实现高质量的动态场景合成。

**💡 创新点**

创新点包括：① 递归循环架构，能够持续累积和利用前后帧的时空信息；② 使用动态平面扫掠体（PSV）将相机运动与场景运动解耦；③ 在不依赖外部先验或预训练模块的前提下，构建轻量化、通用的动态NVS框架。

**🔧 技术方法**

主要技术：递归3D U-Net；动态PSV投影与patchify/ unpatchify；多迭代训练（dilation因子递减）；VGG‑L1 损失；以及基于相机参数的直接条件化。

**📊 数据集**

使用了两个数据集：UCSD（86训练/10测试）和自制的高分辨率 Kubric‑4D‑dyn（5000训练/100测试，512×512，81帧）。

**📈 对比分析**

与四个基于Gaussian Splattings（4DGS、D‑3DGS、SC‑GS、MoSca）以及两个扩散模型（GCD、Gen3C）进行了对比。GRVS在UCSD和Kubric‑4D‑dyn的PSNR/SSIM/LIPPS指标上均名列前茅（例如UCSD PSNR 36.81 dB），并以3 FPS、40M参数的轻量化实现显著高于扩散模型（0.3 FPS、7B参数）。

**⚠️ 局限性**

局限性：在极端高速动态或大尺度场景下仍可能出现几何细节不足；模型对相机标定误差敏感；需要一定数量的连续帧来获得良好上下文，单帧输入效果有限。

---

## 393. Leveraging Synthetic Data for Enhancing Egocentric Hand-Object Interaction Detection

**arXiv ID:** 2603.29733 | [PDF](https://arxiv.org/pdf/2603.29733v1)

**作者:** Rosario Leonardi `[一作]` (University of Catania), Giovanni Maria Farinella `[通讯]` (University of Catania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套可扩展的合成 egocentric 视角手物交互数据生成管道，并基于此构建了 HOI‑Synth 基准，系统评估了合成数据在无监督、半监督与全监督域适配中的效果，显示合成数据能显著提升 HOI 检测性能，尤其在真实标签稀缺时。

**💡 创新点**

创新点包括：① 在大规模真实环境（HM3D）、对象（DexGraspNet）与手模型（SyntheticHumans）的基础上实现高质量、可控的 egocentric 合成图像生成；② 针对目标数据集进行对象、抓握姿态与环境三维对齐的合成数据生成策略；③ 对三种域适配场景（UDA、SSDA、FSDA）进行统一实验，对比合成数据与真实数据的互补性与极限；④ 公开 HOI‑Synth 数据集、生成工具与实验代码，促进后续研究。

**🔧 技术方法**

核心技术：Unity HDRP 合成管线（手/物体建模、摄像机、光照随机化、自动注释）；DexGraspNet+SyntheticHumans 数据匹配；基于 DINOv2 与 MMPose 的对象/抓握/环境对齐；领域适配方法（Adaptive Teacher、Mean Teacher、Unbiased Teacher、GRL、EMA 等）；基线检测模型 VISOR HOS 以及 ConvNeXt-Sbackbone 等。

**📊 数据集**

使用了三大 egocentric HOI 基准数据集：VISOR、EgoHOS、ENIGMA‑51；在每个数据集上生成约 30k 条合成样本（含对象/抓握/环境对齐版本），并扩展为 HOI‑Synth 公开版。

**📈 对比分析**

实验对比包括：Synthetic‑Only、Real‑Only、Unsupervised Domain Adaptation (UDA)、Semi‑Supervised Domain Adaptation (SSDA)、Fully‑Supervised Domain Adaptation (FSDA)。结果显示：① UDA 可将整体 AP 提升 23–35%；② SSDA 仅使用 10–25% 真实标签即可匹敌 100% 标签的 Real‑Only，整体 AP 提升 5–8%；③ FSDA 在 100% 真实标签下进一步提升 1–4% AP；④ 在 ENIGMA‑51 采用域内合成数据与 UDA 能提升 20% AP；并在各种拆解实验中验证对象/抓握/环境对齐的有效性。

**⚠️ 局限性**

局限性：① 仅生成单帧图像，未考虑时间连续性；② 合成仅覆盖抓握交互，未模拟多手协同或工具使用的复杂场景；③ 对齐策略仍需人工选择特征/聚类阈值，未实现端到端自适应；④ 合成环境多样性虽高，但对极端光照/遮挡仍有差距。

---

## 394. Query-Based Committee Selection

**arXiv ID:** 2603.29729 | [PDF](https://arxiv.org/pdf/2603.29729v1)

**作者:** Itay Asher Zimet `[一作]` (Weizmann Institute of Science), Nimrod Talmon `[通讯]` (Weizmann Institute of Science)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究在有限预算下，通过向选民发起结构化分区查询（ refinement queries）来逼近多胜者选举（如 k‑Borda）结果。

**💡 创新点**

创新点在于提出一套查询成本函数的公理体系，并给出满足所有公理的最优成本函数；同时实验表明，采用二分查询（split）并等量分配预算的策略在大多数模型下最能近似真实委员会。

**🔧 技术方法**

使用的方法包括：构建查询式多胜者规则、定义并分析成本函数、公理证明、基于 k‑Borda 的分数分配方案，以及通过 Hamming 距离评估委员会质量；实验采用 IC、二维 Euclidean 以及 Mapof 生成的多种统计模型选举。

**📊 数据集**

实验数据主要为合成选举：IC 与二维 Euclidean 随机生成 5,000 组选举（各 100 名选民、100 名候选人），以及 Mapof 框架下的多模型（Polya‑Eggenberger、Mallows 等）生成的选举。

**📈 对比分析**

比较方法为将查询式规则产生的委员会与传统 k‑Borda 结果进行 Hamming 距离对比，并与随机基线对比。结果显示，Split+EQ 策略在 IC 与 Euclidean 下可在约 130,000 的预算内实现完美逼近，其他策略需多倍预算。

**⚠️ 局限性**

局限性包括：仅考虑非自适应的、单一规则的查询策略；成本函数假设可能与真实选民认知差异不完全匹配；实验仅在合成数据上验证，缺少真实区块链治理等实际应用场景的验证。

---

## 395. α-Fair Multistatic ISAC Beamforming for Multi-User MIMO-OFDM Systems via Riemannian Optimization

**arXiv ID:** 2603.29717 | [PDF](https://arxiv.org/pdf/2603.29717v1)

**作者:** Hyeonho Noh `[一作]` (Hanbat National University), Jonggyu Jang `[通讯]` (Chungnam National University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一个基于α‑公平度量的多静态ISAC波束成形框架，使通信用户充当被动双向接收机，从而实现多用户MIMO‑OFDM系统的联合感知与通信优化。

**💡 创新点**

创新点在于：①将α‑公平性直接引入CRLB目标，使得感知精度可以在多目标间实现平滑公平分配；②采用光滑罚函数将速率约束嵌入目标函数，避免传统SDR所导致的秩一近似损失；③在复数球面流形上使用黎曼共轭梯度（RCG）方法高效求解非凸问题。

**🔧 技术方法**

主要技术包括：α‑公平性目标函数、CRLB与Fisher信息矩阵推导、光滑罚函数改写、黎曼共轭梯度优化、线搜索与重拉提（retraction）操作。

**📊 数据集**

实验采用仿真数据：基站20个天线，28 GHz载频，100 MHz带宽，10个感知目标与10个通信用户随机分布，功率30 dBm；未使用公开数据集。

**📈 对比分析**

与单静态、SDR‑基线以及SCA方法比较，结果显示该方案在满足相同数据率约束下，能够显著降低总CRLB和最大CRLB，并在不同用户数量和公平性参数α下保持优越的感知公平性与通信质量。

**⚠️ 局限性**

局限性包括：依赖理想同步与信道模型，未在真实环境中验证；对用户数量和目标数的规模扩展仍需进一步评估；光滑罚参数的选择仍需经验指导。

---

## 396. Symphony for Medical Coding: A Next-Generation Agentic System for Scalable and Explainable Medical Coding

**arXiv ID:** 2603.29709 | [PDF](https://arxiv.org/pdf/2603.29709v1)

**作者:** Joakim Edin `[一作]`, Lars Maaløe `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 Symphony，一种面向医疗编码的多代理推理系统，能在不重新训练的情况下跨编码体系自动生成诊断和程序代码，并给出 span‑level 证据。

**💡 创新点**

创新点在于将编码流程拆解为四步（证据提取、索引导航、表格验证、代码调和）并让 LLM 在每一步访问正式编码指南，从而实现开放集、可解释、可扩展的编码。

**🔧 技术方法**

技术上基于 CLH 框架，采用 LLM‑驱动的多代理工作流、结构化知识检索、规则式验证以及可插拔的工具接口，配合 OpenAI、Claude、Gemini 等大模型进行推理。

**📊 数据集**

使用了五个数据集：公开的 ACI、MDACE，私有的 ED、AMB、NEURO（覆盖美国和英国多种临床场景和编码系统），并覆盖诊断（ICD‑10‑CM/ICD‑10）与程序（ICD‑10‑PCS、CPT）编码。

**📈 对比分析**

与 fine‑tuned、基于链式思考、工作流式等基线对比，Symphony 在受限标签和全标签评估中均实现了 F1 最高（例如 ACI 0.74，MDACE 0.58；全标签下 ACI 64.1，MDACE 37.3），显著优于 GPT、Claude、Gemini 等通用大模型。

**⚠️ 局限性**

局限性包括对大型标注数据的依赖、在极稀有或新出现的代码上仍可能欠缺精度，以及在某些数据集中的证据跨度评估受限于人类标注粒度差异导致的误差估计。

---

## 397. 6GAgentGym: Tool Use, Data Synthesis, and Agentic Learning for Network Management

**arXiv ID:** 2603.29656 | [PDF](https://arxiv.org/pdf/2603.29656v1)

**作者:** Jiao Chen `[一作]` (Shenzhen Smart City Technology Development Group Company Limited), Zuohong Lv `[通讯]` (China Unicom Group Co Limited)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 6GAgentGym 框架，集成了闭环网络管理的交互环境、工具系统、数据合成管道与评测基准。

**💡 创新点**

创新点在于：42 种类型化工具与 NS‑3 基准实验模型相结合、Self‑Instruct 生成闭环轨迹的 6G‑Forge、以及 L1–L3 分层的 6GAgentBench 评测。

**🔧 技术方法**

使用了 LLM 工具调用（ReAct）、监督微调+强化学习、NS‑3 仿真、Experiment Model 代理、ROUGE‑L 去重、以及 DAPO 等 RL 算法。

**📊 数据集**

数据集包括 3,000 条真实 NS‑3 轨迹（seed）、约 50,000 条 Self‑Instruct 生成的合成轨迹，以及 6GAgentBench 的 L1–L3 评测任务。

**📈 对比分析**

与 GPT‑5、Claude‑Sonnet‑4、Gemini‑2.5‑Pro 等大型模型及非 LLM 基线对比，8B 开源模型在 SFT+RL 后实现 50% 的整体成功率，且在长序列（L3）任务上优于 GPT‑5，RL 增加约 5% 性能。

**⚠️ 局限性**

限制在于：Experiment Model 只近似 NS‑3 的协议级瞬态，工具集缺少射频级操作，RL 训练受步长与任务多样性限制，且未覆盖多模态输入，需进一步提升。

---

## 398. Mind the Gap: A Framework for Assessing Pitfalls in Multimodal Active Learning

**arXiv ID:** 2603.29677 | [PDF](https://arxiv.org/pdf/2603.29677v1)

**作者:** Dustin Eisenhardt `[一作]` (German Cancer Research Center), Florian Buettner `[通讯]` (German Cancer Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一个可控的多模态主动学习基准框架，系统评估缺失模态、模态不平衡和模态交互三大陷阱，并验证现有查询策略在真实数据上的局限。

**💡 创新点**

创新点在于设计了专门隔离三大陷阱的合成数据集（QuintFeatures、Missing、Share、Unique、Synergy）并给出了第一套完整的多模态主动学习评测基准，揭示了现有查询方法对模态不平衡缺乏修正。

**🔧 技术方法**

采用多种主动学习查询策略（BADGE、BALD、Entropy、GRACE、KCG 等）与多模态融合网络，结合 Modality Dropout、AULC 评估指标，进行合成与真实数据实验。

**📊 数据集**

使用了合成数据集（QuintFeatures 等）以及真实数据 Food101（图像+文本）和 MIMIC‑IV（血检+ECG）进行验证。

**📈 对比分析**

通过比较 5 种单模态查询方法与 2 种多模态查询方法，在低/中/高标注预算下计算 AULC；结果显示大多数查询方法未能缓解模态不平衡，且多模态策略并未显著优于单模态，表现随预算波动。

**⚠️ 局限性**

局限在于仅考虑双模态和 late‑fusion 架构，未涵盖多模态数目增大或不同融合策略；实验仅评估已存在的查询策略，未探索新的模态平衡主动学习方法。

---

## 399. Concept frustration: Aligning human concepts and machine representations

**arXiv ID:** 2603.29654 | [PDF](https://arxiv.org/pdf/2603.29654v1)

**作者:** Enrico Parisini `[一作]` (King’s College London), Christopher R. S. Banerji `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了“概念挫折”（concept frustration）框架，用于对齐人类可解释概念与基础模型中无监督学习到的内部表示，并研究其对概念瓶颈模型（CBM）解释性和性能的影响。

**💡 创新点**

创新点在于将概念挫折形式化为三角形关系，并引入任务对齐的 Fisher 信息几何来衡量概念一致性；同时给出线性高斯模型下概念最佳分类器的闭式准确率，并展示引入挫折概念如何重新塑造概念空间。

**🔧 技术方法**

技术手段包括构建 Fisher 信息度量、任务对齐相似度矩阵、概念瓶颈模型、稀疏自编码器、线性高斯仿真以及统计检验。

**📊 数据集**

实验使用了合成数据生成器，以及真实语言任务的 28,000 条讽刺标题（DeBERTa 嵌入）和视觉任务的 CUB‑200 gull‑tern 子集（CLIP ViT‑B/16 嵌入）。

**📈 对比分析**

通过比较仅使用已知概念与同时使用挫折概念的 CBM，发现后者在任务准确率上更高、概念均方误差更低；Fisher‑Frustration 指标能区分两种情况，而欧氏相似度则失效；在合成实验中挫折导致 CBM 准确率显著下降。

**⚠️ 局限性**

局限性在于模型假设线性高斯生成、Fisher 量度基于单层黑盒，挫折定义依赖于选定的无监督方向，且对更复杂的非线性模型或更大规模基础模型的适用性尚未验证。

---

## 400. HPCCFA: Leveraging Hardware Performance Counters for Control Flow Attestation

**arXiv ID:** 2603.29749 | [PDF](https://arxiv.org/pdf/2603.29749v1)

**作者:** Claudius Pott `[一作]` (University of Lubeck), Thomas Eisenbarth `[通讯]` (University of Lubeck)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了基于可信执行环境的控制流证明（CFA）机制，利用RISC‑V硬件性能计数器（HPC）在Keystone TEE上实现并验证；

**💡 创新点**

通过在TEE内部运行两个隔离的 enclave（tracer 与 tracee）实现无硬件改造的实时控制流证明，并使用 HPC 预测每个基本块的计数，从而生成可验证的控制流日志；

**🔧 技术方法**

采用了可信执行环境 Keystone、RISC‑V HPC、静态 CFG 分析、整数线性规划（ILP）求解、Python/Rust 预处理以及共享内存 IPC 等技术；

**📊 数据集**

使用了三个示例 enclave（HelloWorld、tweetnacl 加密库、动态调度示例）以及在 VisionFive2 SoC 上收集的 HPC 计数数据；

**📈 对比分析**

通过改变测量点数量、计数器数量和额外测量点，比较可靠性指标（检测成功率）和执行时间与未加测量的基线，发现可靠性可达 99% 时性能提升 3–14 倍；

**⚠️ 局限性**

限制包括：对长序列和循环密集代码可靠性低、需要手动插入测量点、无法处理递归与间接分支、仅在 RISC‑V Keystone 上实现、以及验证过程对 Python 脚本依赖导致性能低。

---

## 401. Approximation Schemes for Edit Distance and LCS in Quasi-Strongly Subquadratic Time

**arXiv ID:** 2603.29702 | [PDF](https://arxiv.org/pdf/2603.29702v1)

**作者:** Xiao Mao `[一作]` (Stanford University), Aviad Rubinstein `[通讯]` (Stanford University)

**通讯引用:** 15522 | [OpenAlex ID](https://openalex.org/A5061644327)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的随机近似算法，用于编辑距离（ED）和最长公共子序列（LCS）问题，能够在时间复杂度为 n^2 / 2^log^Ω(1)(n) 的情况下，计算 ED 的 (1+ϵ) 近似和 LCS 的 (1-ϵ) 近似。

**💡 创新点**

该研究的创新点在于通过随机化方法实现了对 ED 和 LCS 的近似计算，显著提高了运行时间，相比于经典的二次动态规划算法，提升了一个准多项式因子，并且展示了近似 ED 和精确 ED 之间的复杂性分离。

**🔧 技术方法**

使用了随机化算法和树总偏差的概念，结合了对路径的分层和采样技术，以实现高效的近似计算。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了算法在字符串长度总和为 n 的情况下的表现。

**📈 对比分析**

与现有的算法相比，提出的算法在运行时间上有显著的改进，能够在准多项式时间内实现近似计算，且在理论上证明了其有效性和准确性。

**⚠️ 局限性**

该算法的局限性在于其依赖于某些复杂性假设，且在处理特定输入时可能会遇到性能瓶颈。

---

## 402. Machine Learning in the Wild: Early Evidence of Non-Compliant ML-Automation in Open-Source Software

**arXiv ID:** 2603.29698 | [PDF](https://arxiv.org/pdf/2603.29698v1)

**作者:** Zohaib Arshid `[一作]` (University of Sannio), Massimiliano Di Penta `[通讯]` (University of Sannio)

**通讯引用:** 20016 | [OpenAlex ID](https://openalex.org/A5025099559)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对173个高风险开源GitHub项目进行初步分析，识别了45个使用ML模型做自动决策的项目，并评估其后处理措施与使用条款合规性。

**💡 创新点**

创新点在于首次系统性检视高风险领域ML模型的自动化决策与合规风险，提出基于人工与自动方式的分类方法和后处理模式，构建可复制的合规评估框架。

**🔧 技术方法**

主要技术为人工代码审计、流程追踪、后处理模式归类和条款违背检测，并使用GitHub Search API与Hugging Face Transformers依赖交叉筛选。

**📊 数据集**

数据集为173个使用Python且依赖Hugging Face Transformers的GitHub项目，覆盖16个高风险应用领域；其中45个项目被确认具备完全或部分自动决策功能。

**📈 对比分析**

本文未进行算法性能对比，而是通过对比项目是否满足EU AI Act及模型ToU来评估合规性，发现约56%项目存在潜在条款违规，且多数项目采用后处理步骤。

**⚠️ 局限性**

局限性包括样本规模有限、仅聚焦Python项目、仅覆盖使用Transformers库的模型，人工审计可能存在误判，结果不一定能推广到商业闭源或其他语言的系统。

---

## 403. View-oriented Conversation Compiler for Agent Trace Analysis

**arXiv ID:** 2603.29678 | [PDF](https://arxiv.org/pdf/2603.29678v1)

**作者:** Lvmin Zhang `[一作]` (Stanford University), Maneesh Agrawala `[通讯]` (Stanford University)

**通讯引用:** 20049 | [OpenAlex ID](https://openalex.org/A5045835385)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了VCC（View-oriented Conversation Compiler）将原始的Agent JSONL日志编译成多层结构化视图，以提升上下文学习效果。

**💡 创新点**

创新点在于构建了三种视图（完整视图、UI视图、适配视图）并保证行号一致性，同时通过可插拔的相关性谓词实现动态查询投影。

**🔧 技术方法**

技术上采用了编译器范式（词法分析→语法分析→IR→视图降级）以及正则/BM25/嵌入/LLM可扩展的相关性判定。

**📊 数据集**

实验使用了AppWorld基准，涵盖三种模型（Opus、Sonnet、Haiku）及其反射器配置。

**📈 对比分析**

与直接使用原始JSONL对比，VCC在所有模型上均实现了更高的任务完成率、显著降低的反射器token消耗（约1/3–1/2）以及更紧凑的学习记忆文件。

**⚠️ 局限性**

局限性包括仅在AppWorld上验证、仅测试正则谓词、未对其他轨迹学习方法或更复杂的查询谓词进行评估。

---

## 404. Near-Miss: Latent Policy Failure Detection in Agentic Workflows

**arXiv ID:** 2603.29665 | [PDF](https://arxiv.org/pdf/2603.29665v1)

**作者:** Ella Rabinovich `[一作]` (IBM Research), Ateret Anaby-Tavor `[通讯]` (IBM Research)

**通讯引用:** 658 | [OpenAlex ID](https://openalex.org/A5078882226)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM驱动的代理在业务流程自动化中，提出一种评估指标以检测“潜在失败”（近似失误）——即代理在未执行必要政策检查的情况下仍能得到正确结果。

**💡 创新点**

创新点是将工具守卫（ToolGuard）生成的保护代码与对话轨迹结合，定义并检测此前未被评估的“近似失误”错误类别。

**🔧 技术方法**

采用ToolGuard框架生成可执行守卫代码，使用LLM（Claude‑Sonnet4、GPT‑5.1‑Codex等）进行历史搜索或代码生成以识别缺失的只读工具调用。

**📊 数据集**

在改进版的τ²‑verified Airlines基准上进行实验，包含开放式与专有LLM共六个模型。

**📈 对比分析**

通过比较总失败率、政策违规率及近似失误率，发现仅考虑总失败率时近似失误率低，但在包含可变更工具调用的轨迹中达到8–17%，表明大部分模型存在未检测的政策违规。

**⚠️ 局限性**

限制包括对ToolGuard生成代码质量的依赖、额外计算开销、仅在单一基准和工具集上验证等。

---

## 405. CutClaw: Agentic Hours-Long Video Editing via Music Synchronization

**arXiv ID:** 2603.29664 | [PDF](https://arxiv.org/pdf/2603.29664v1)

**作者:** Shifang Zhao `[一作]` (Beijing Jiaotong University), Xiaodong Cun `[通讯]` (Great Bay University)

**通讯引用:** 4583 | [OpenAlex ID](https://openalex.org/A5058799911)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套名为CutClaw的自主多代理框架，能够将数小时的原始视频根据用户指令和背景音乐进行自动剪辑，生成节奏同步、符合叙事、视觉美观的短视频。

**💡 创新点**

创新点包括：① 将长时序视频和音乐先进行底层多模态分解，生成结构化语义单元；② 通过Playwriter将音乐结构作为时间锚点，构建全局叙事计划；③ 采用Editor与Reviewer双层验证，实现细粒度的音画同步与内容一致性；④ 将多模态语言模型与ReAct代理相结合，实现端到端的高效协作。

**🔧 技术方法**

技术手段：多模态语言模型（Qwen3-VL、Qwen3-Omni）、ReAct式小代理（MiniMax-M2.1、Gemini3-Pro）、自动场景与音频分割（PySceneDetect、Whisper）、音画同步关键点检测、语义邻域搜索与评价门控。

**📊 数据集**

使用自构造的数据集：10对来源于5部电影和5段VLOG的原始视频，总时长约24小时；10段不同类型（Pop、Jazz、OST、Rock、R&B）的音乐；对应20个评测案例（人物中心与叙事中心两种指令）。

**📈 对比分析**

与NarratoAI、UVCOM、Time‑R1等基线进行对比，采用视觉质量、指令遵循、音画和谐度、人工拟人化等指标评估。CutClaw在所有指标上均优于基线，尤其在视觉质量（+~3点）、指令遵循（+~5点）和AV和谐度（+~9点）上表现突出。

**⚠️ 局限性**

局限性：① 缺乏生成式视觉特效与戏剧化亮点；② 端到端流程耗时高，难以实现实时反馈；③ 对极端长时序或极低质量素材的鲁棒性仍待提升。

---

## 406. HackRep: A Large-Scale Dataset of GitHub Hackathon Projects

**arXiv ID:** 2603.29672 | [PDF](https://arxiv.org/pdf/2603.29672v1)

**作者:** Sjoerd Halmans `[一作]` (Eindhoven University of Technology), Alexander Nolte `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 1314 | [OpenAlex ID](https://openalex.org/A5067373153)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个跨领域、非大学限定的 Hackathon GitHub 项目数据集 HackRep，包含项目、时间、团队构成、地理信息等元数据。

**💡 创新点**

首次提供大规模 Hackathon 开源项目数据集，并通过自动化推断持续时间、团队构成与科研属性，填补了现有案例研究的局限。

**🔧 技术方法**

采用 GitHub 搜索爬取、Kleinberg 突发检测算法估算 hackathon 时长、TF‑IDF+逻辑回归模型自动识别科研项目、Geotext 提取 README 中的地理位置。

**📊 数据集**

以 GitHub 上标记为 "hackathon" 的 207,766 个仓库为基础，结合 Devpost 等公开数据进行对比验证。

**📈 对比分析**

与 Devpost 数据集交叉比对验证覆盖面和多样性；自动分类模型准确率 77%，科研仓库 F1 63%/84%；推断平均持续时间 36 小时与先前研究一致。

**⚠️ 局限性**

单一注释者导致偏见；关键词过滤可能漏掉部分科研项目；未验证仓库可编译性；对非英文 README 的文本分析能力有限。

---

## 407. Nonnegative Matrix Factorization in the Component-Wise L1 Norm for Sparse Data

**arXiv ID:** 2603.29715 | [PDF](https://arxiv.org/pdf/2603.29715v1)

**作者:** Giovanni Seraghiti `[一作]` (University of Mons), Nicolas Gillis `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对基于组件级L1范数的非负矩阵分解（L1‑NMF）进行了系统研究，首先证明了即使秩为1时该问题仍是NP‑hard的；随后指出L1‑NMF在稀疏数据上天然产生更稀疏因子，进而引入加权L1‑NMF（wL1‑NMF）模型，用惩罚参数λ来调控零值的影响；接着设计了一种坐标下降（CD）算法，利用加权中值算法求解每个标量LAD子问题，使得每次迭代的复杂度只与原始矩阵非零条目数成正比；最后在合成和真实数据上验证了该模型与算法的有效性与优势。

**💡 创新点**

创新点主要有：①首次给出秩‑1 L1‑NMF的NP‑hard性证明；②提出了可调节零值影响的加权L1‑NMF框架；③设计了线性时间（按非零条目数）的CD算法（Sparse‑CD），实现了在大规模稀疏数据上的高效求解；④通过理论与实验验证了L1‑NMF在稀疏数据上天然产生稀疏因子，并给出了λ参数的实践指导。

**🔧 技术方法**

主要技术包括：NP‑hard性证明的多步还原（从Max‑Cut到L1‑NMF）；坐标下降框架与一维LAD子问题的加权中值求解；非负约束下的分解子问题分离与并行化；对比度分析与稀疏性概率推导；统计解释（双Laplace分布）与参数λ的意义；以及对多种现有L1‑NMF/非负分解算法（NS、SUB、HALS、MU）的实现与评测。

**📊 数据集**

使用的数据集有：①随机生成的稀疏矩阵（m×n取100×200、300×400、500×600、800×1000）；②MNIST手写数字图像（784×300），并加入不同概率的噪声；③合成低秩矩阵加Laplace噪声并随机或阈值化生成零值（用于矩阵补全实验）；④TDT2 文档-词汇稀疏矩阵（19528×9394，零率99.37%）。

**📈 对比分析**

通过与Frobenius、KL、L21、SUB、NS、HALS等经典NMF方法在重构误差（相对Frobenius误差或L1误差）与运行时间上的对比，验证了：①在高噪声/长尾噪声下，L1‑NMF/ wL1‑NMF获得更低的重构误差；②Sparse‑CD在稀疏数据上比原始CD快4–5倍，且比NS、SUB在大规模稀疏矩阵上更快、更稳定；③适当的λ（0.01–0.05）能在保持稀疏性的同时显著降低误差，尤其在“错误零”存在的矩阵补全任务中。

**⚠️ 局限性**

局限性包括：①CD方法在每个子问题可能存在多重最优解，缺乏全局收敛保证，结果高度依赖初始化；②当数据极度稀疏且λ接近1时，可能导致因子过度稀疏；③在密集矩阵或高秩场景下，Sparse‑CD的优势消失；④与部分先进的L1‑NMF/矩阵补全算法（如基于核方法或随机投影的近似）尚未做全面比较。

---

## 408. 'AI' and Computer Science: Contradictions Emerge between Ideologies

**arXiv ID:** 2603.29746 | [PDF](https://arxiv.org/pdf/2603.29746v1)

**作者:** Andruid Kerne `[一作]` (University of Illinois Chicago), Andruid Kerne `[通讯]` (University of Illinois Chicago)

**通讯引用:** 2238 | [OpenAlex ID](https://openalex.org/A5078871533)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本文通过理论分析，构建了意识形态框架，阐述了在AI与计算机科学发展中企业与高校合作所产生的矛盾与对立，并对这一现象的设计与社会影响提出了批判性思考。

**💡 创新点**

创新点在于将Stuart Hall的意识形态理论与计算机科学/AI产业相结合，揭示了“AI赋能”宣传背后隐藏的利益冲突和劳动市场的恶化，提出了对学术与产业界应保持的批判性反思与设计原则。

**🔧 技术方法**

无具体技术实现，主要采用理论阐述与文献综述的方法。

**📊 数据集**

无实验数据集，文章以公开文献与案例分析为依据。

**📈 对比分析**

无实验对比与性能评估，主要通过对比不同意识形态文本与产业实践的描述进行理论比较。

**⚠️ 局限性**

局限性在于缺乏经验数据支持，研究仍停留在概念层面，对现实政策与技术路径的具体指导有限。

---

## 409. Spontaneous Functional Differentiation in Large Language Models: A Brain-Like Intelligence Economy

**arXiv ID:** 2603.29735 | [PDF](https://arxiv.org/pdf/2603.29735v1)

**作者:** Junjie Zhang `[一作]` (Chinese Academy of Sciences), Xisong Dong `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 1122 | [OpenAlex ID](https://openalex.org/A5066880964)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过ΦID信息重组框架分析大型语言模型的内部信息处理机制，揭示其早期与后期层为记忆模块，中央层为抽象推理核心，并证明该结构随任务难度自发演化；

**💡 创新点**

首次将PID/ΦID与深度神经网络中的注意力头关联，发现中层呈现信息协同核心、能量集中消耗与大脑代谢模式相似，构建神经网络拓扑模型并量化其与大脑区域功能的相似性；

**🔧 技术方法**

信息论（PID、ΦID）、时间延迟互信息、Möbius逆变换、注意力头能量消耗计算、网络拓扑分析（全局效率、模块化、力导向布局）及积分梯度可视化；

**📊 数据集**

GSM8K、ARC（易/难）、Qwen3与Gemma3系列模型的数学与抽象推理任务；

**📈 对比分析**

通过对比完整模型、层跳过与注意力头消融实验，发现消除抽象核心层会导致性能崩溃，证明中层对任务完成至关重要；实验显示中层对多步推理任务的贡献远超前后层；

**⚠️ 局限性**

研究仅聚焦数学与推理任务，未涉及创造性或语言任务；ΦID计算资源需求高，限制了实时监控；对大规模模型的泛化验证不足；

---

## 410. Compressive sensing inspired self-supervised single-pixel imaging

**arXiv ID:** 2603.29732 | [PDF](https://arxiv.org/pdf/2603.29732v1)

**作者:** Jijun Lu `[一作]`, Xuelong Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于压缩感知的自监督单像素成像方法SISTA-Net。

**💡 创新点**

通过将ISTA展开为数据保真模块和近似映射模块，并在此基础上融合混合CNN‑VSSM结构与可学习软阈值，实现了物理稀疏约束，兼顾局部细节与全局关联。

**🔧 技术方法**

使用了深度图像先验、视觉状态空间模型（VSSM）、深度稀疏编码器/解码器、可学习软阈值以及自监督训练策略。

**📊 数据集**

采用320×320二值与灰度自然图像的仿真测量数据以及海水环境下的真实单像素测量数据。

**📈 对比分析**

与DGI、DnCNN、UNet、Restormer、SwinIR等基线在低采样率下对比，SISTA-Net平均PSNR提升约2.6 dB，水下远场测试提升3.4 dB，显示出更强的抗噪与细节保留能力。

**⚠️ 局限性**

受限于单像素测量的噪声模型和对测量矩阵假设的依赖，可能在极端光照或动态场景下性能仍有下降，且模型结构相对复杂。

---

## 411. Reinforced Reasoning for End-to-End Retrosynthetic Planning

**arXiv ID:** 2603.29723 | [PDF](https://arxiv.org/pdf/2603.29723v1)

**作者:** Chenyang Zuo `[一作]` (Tsinghua University), Zaiqing Nie `[通讯]` (Tsinghua University)

**通讯引用:** 3539 | [OpenAlex ID](https://openalex.org/A5047496977)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出ReTriP，一个端到端生成式框架，将多步逆合成规划转化为链式推理；

**💡 创新点**

通过路径连贯的分子表征和逐步强化学习奖励，将局部反应逻辑与全局规划目标统一；

**🔧 技术方法**

利用大型语言模型的链式推理、分子表示的根对齐、三阶段监督微调、RL与可验证奖励以及测试时多视角投票；

**📊 数据集**

在RetroBench（USPTO全网络）上进行评估；

**📈 对比分析**

与传统模板/无模板+外部搜索基线对比，ReTriP在Top‑1准确率上提升3.4%，在深度≥5路径上保持约40%准确率；

**⚠️ 局限性**

仍受限于训练数据覆盖范围、对非常深路径的极限鲁棒性以及对罕见反应的泛化能力。

---

## 412. A Comprehensive Corpus of Biomechanically Constrained Piano Chords: Generation, Analysis, and Implications for Voicing and Psychoacoustics

**arXiv ID:** 2603.29710 | [PDF](https://arxiv.org/pdf/2603.29710v1)

**作者:** Mahesh Ramani `[一作]` `[通讯]` (Independent), Mahesh Ramani (Independent)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

生成并分析了约 19.3 M 条可演奏钢琴和弦的开源语料库，并研究了和弦声部分布（如偏度和分布宽度）对听觉粗糙度与和谐度的影响。

**💡 创新点**

创新点包括：① 对 88 键钢琴手指可达范围内的和弦进行全枚举与采样，首次构建规模最大的可演奏和弦数据集；② 在严格控制音高组成后，系统性分离偏度与分布宽度的影响，证明偏度是粗糙度的更强预测因子；③ 将声学模型与统计回归相结合，为未来生成式和声与声部布局研究提供量化基准。

**🔧 技术方法**

使用的技术手段有：手指跨度约 1.5 八度的生物力学约束下的枚举与 Monte Carlo 采样；对 MIDI 音高计算统计瞬时（均值、范围、偏度、峰度）并残差化；Plomp‑Levelt 失谐模型与自定义和谐度评估；线性回归与置换检验验证模型显著性；以及残差化处理以消除音高计数和音类影响。

**📊 数据集**

数据集：自建的 19,376,000 条可演奏和弦（MIDI 21–108）数据库，公开托管于 Hugging Face，包含手指可达范围、每个和弦的音高、统计特征与声学指标。

**📈 对比分析**

比较方法：在基线模型（仅包含音类向量和音符数）与加入声部统计特征后的模型之间进行 R² 与 ΔR² 的比较。结果显示：和谐度的 ΔR² 仅为 0.014%（p≈0.13，非显著），而粗糙度的 ΔR² 达到 6.75%（p≈0.0008，显著），并且 R² 提升至约 0.71；偏度的标准化系数 β≈+0.145 远大于分布宽度 β≈-0.025，表明偏度是粗糙度的主要驱动因素。

**⚠️ 局限性**

局限性：① 所有音符假设同等响度，未考虑动态与音量差异；② 仅使用纯正弦波模型，未考虑钢琴的谐波失真与不和声；③ 缺乏人类听觉验证，仅依赖既有声学模型；④ 手指跨度约 1.5 八度的生物力学约束相对宽松，可能包含边缘可玩性；⑤ 未考虑时间、持续时间或上下文因素对感知的影响。

---

## 413. SafeDMPs: Integrating Formal Safety with DMPs for Adaptive HRI

**arXiv ID:** 2603.29708 | [PDF](https://arxiv.org/pdf/2603.29708v1)

**作者:** Soumyodipta Nath `[一作]` (Indian Institute of Science), Ravi Prakash `[通讯]` (Indian Institute of Science)

**通讯引用:** 1724 | [OpenAlex ID](https://openalex.org/A5037703330)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 SafeDMPs 框架，将动态运动原语（DMP）与时空管道（STT）结合，实现实时、可证明安全的机器人轨迹规划。

**💡 创新点**

创新点在于利用闭式控制律实现 STT 的安全约束，无需在线优化，兼顾鲁棒性与安全性，并在动态人机交互环境中实现快速响应。

**🔧 技术方法**

采用 Dynamic Movement Primitives、Spatio-Temporal Tubes、闭式安全控制、时间缩放调节、对比 DMP–APF 与 NODE–CLF–CBF 等技术。

**📊 数据集**

使用 LASA 手写数据集生成 DMP 示例，实验基于 Franka Emika Panda 机器人和仿真环境，并进行真实人机交互测试。

**📈 对比分析**

与 NODE–CLF–CBF（优化型）和 DMP–APF（启发式）比较；SafeDMPs 在执行时间、内存占用、轨迹误差、扰动恢复时间和碰撞避免等指标均优于两者，速度提升至 99% 以上。

**⚠️ 局限性**

局限在于参数需要手动调节；STT 边界附近控制力趋向无穷受限于执行器；缺乏自适应参数选择和大扰动下的管道重规划。

---

## 414. Drift-Aware Continual Tokenization for Generative Recommendation

**arXiv ID:** 2603.29705 | [PDF](https://arxiv.org/pdf/2603.29705v1)

**作者:** Yuebo Feng `[一作]` (Fudan University), Ning Gu `[通讯]` (Fudan University)

**通讯引用:** 43876 | [OpenAlex ID](https://openalex.org/A5012421463)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了DACT框架，专门针对生成推荐中可学习分词器的持续学习，以适应协作信号随时间漂移而导致的标识符失效问题。

**💡 创新点**

创新点包括：①协作漂移识别模块（CDIM）能够在增量训练时自动判断哪些商品在协作语义上漂移；②差异化更新策略将漂移商品与静止商品分别处理；③层级代码重新分配策略在保留第一层重要语义的同时，仅在必要时更新后续层，显著降低标识符冲突。

**🔧 技术方法**

技术手段包括：RQ‑VAE分词器+协作对齐正则、CDIM（注意力+模式匹配）、差异化损失（漂移与稳定项）、全局代码分配稳定损失、Relaxed‑to‑Strict层级重分配、以及在TIGER与LCRec等生成推荐模型上进行微调与对比。

**📊 数据集**

使用的公开数据集为Amazon的Beauty、Tools、Toys三大业务场景，覆盖多类商品且交互稀疏，能够体现协作信号漂移的典型情况。

**📈 对比分析**

与多种基线（冻结、仅微调、重训练、Reformer、LSAT、PESO等）以及两种生成推荐骨干（TIGER、LCRec）进行系统对比，DACT在所有评估周期和指标（HR@5/10、NDCG@5/10）上均实现了显著提升，且训练时间和计算成本低于全重训练方案。

**⚠️ 局限性**

局限性：①对非协作语义漂移（如商品内容变更）处理不充分；②CDIM及相关超参（K、β等）需手动调优，可能在不同业务场景下表现不一致；③当前层级重分配策略仅针对三层RQ‑VAE，进一步扩展到更大规模或多层代码簿的有效性尚待验证。

---

## 415. SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition

**arXiv ID:** 2603.29692 | [PDF](https://arxiv.org/pdf/2603.29692v1)

**作者:** Ning Wang `[一作]` (Chang'an University), zhang liang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SkeletonContext框架，用语言驱动的上下文重构和关键部位解耦增强骨骼动作识别的零样本泛化能力。

**💡 创新点**

创新点在于跨模态上下文提示模块：利用LLM生成的结构化描述并通过掩码重构，将缺失的对象和环境信息注入骨骼编码；以及关键部位解耦模块，突出动作相关关节。

**🔧 技术方法**

采用预训练语言模型BERT进行上下文提示，Shift-GCN做骨骼编码，双向跨模态注意力进行特征融合，使用掩码重构损失、对比交叉熵和关键部位解耦损失共同训练。

**📊 数据集**

使用NTU-RGB+D 60/120和PKU-MMD三个基准数据集，采用55/5、48/12、110/10、96/24等经典零样本划分以及随机拆分评估。

**📈 对比分析**

与多种最新方法（如SCoPLe、Neuron、FS-VAE等）对比，在ZSL和GZSL下均实现或逼近最优表现，尤其在谐波平均和高难度混淆类上取得显著提升。

**⚠️ 局限性**

局限在于对LLM生成描述的依赖；对极度缺乏上下文的动作（如纯手势）效果有限，且对不同语言环境的鲁棒性尚待验证。

---

## 416. Client-Verifiable and Efficient Federated Unlearning in Low-Altitude Wireless Networks

**arXiv ID:** 2603.29688 | [PDF](https://arxiv.org/pdf/2603.29688v1)

**作者:** Yuhua Xu `[一作]` (Beijing Institute of Technology), Liehuang Zhu `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 15757 | [OpenAlex ID](https://openalex.org/A5100634361)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在低空无线网络（LAWN）环境下，提出 VerFU，一种支持离线客户端可验证、可并行执行的联邦“遗忘”（unlearning）框架，能在不泄露原始数据的前提下彻底移除离线设备对全局模型的贡献。

**💡 创新点**

创新点包括：① 利用线性同态哈希（LHH）+承诺机制构建不可篡改的贡献记录；② 在同态加密域完成聚合与遗忘，保证梯度隐私；③ 采用状态标记实现多客户端并行遗忘与验证，避免传统顺序处理导致的低效；④ 通过哈希线性组合实现无样本可验证的遗忘结果。

**🔧 技术方法**

核心技术包括：Paillier 同态加密、线性同态哈希（LHH）、承诺方案、状态标记（client state tagging）以及轻量级的哈希线性组合验证。

**📊 数据集**

实验使用三大标准图像分类数据集：MNIST、Fashion‑MNIST 和 EMNIST，并在轻量级 LeNet CNN 上进行联邦学习与遗忘实验。

**📈 对比分析**

与 PoL、PoLHE、zkPoT 等基线方法相比，VerFU 在通信开销和验证时延上显著更低（通信≈74 MB/模型、验证≈12 s），且在不同遗忘率下仍能保持 1.3% 以内的精度下降和 0.044 的损失增幅，验证与模型恢复都能在 2–4 轮内完成。

**⚠️ 局限性**

局限性包括：① 同态加密导致的加解密计算开销仍较高；② 仅在小模型（LeNet）和受限的仿真场景中验证，缺乏大规模真实 LAWN 部署的评估；③ 需要服务器可靠生成公钥和 LHH 参数，若这些前置过程受攻击仍可能影响安全性。

---

## 417. Beyond the Steeper Curve: AI-Mediated Metacognitive Decoupling and the Limits of the Dunning-Kruger Metaphor

**arXiv ID:** 2603.29681 | [PDF](https://arxiv.org/pdf/2603.29681v1)

**作者:** Christopher Koch `[一作]` `[通讯]`, Christopher Koch

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并提出AI介入导致元认知脱钩（AI‑mediated metacognitive decoupling）模型，挑战传统的Dunning‑Kruger曲线放大说法。

**💡 创新点**

提出四变量模型（输出、实际表现、自我评估、校准）阐释AI辅助如何使这些变量解耦，从而更准确解释过度自信、依赖失衡和学习迁移问题。

**🔧 技术方法**

主要采用系统性文献综述、案例分析与实验结果对比，并未开发新算法，而是整合了现有实验技术与模型评估方法。

**📊 数据集**

参考的主要数据来源包括LSAT逻辑推理题库、GPT‑4 辅导系统的学生成绩、知识工作者的问卷与实际使用记录等公开或合作数据集。

**📈 对比分析**

通过将AI辅助与无AI条件下的任务得分、主观自评与校准误差进行对比，发现AI提升了可观测输出但自我评估与实际水平脱节，校准误差扩大。

**⚠️ 局限性**

局限性在于证据主要集中于少数任务领域（如LSAT、数学辅导、知识工作调查），缺乏在编码、写作、临床推理等更广泛场景的验证。

---

## 418. Not All Frames Are Equal: Complexity-Aware Masked Motion Generation via Motion Spectral Descriptors

**arXiv ID:** 2603.29655 | [PDF](https://arxiv.org/pdf/2603.29655v1)

**作者:** Pengfei Zhou `[一作]` (Beihang University), Yong Hu `[通讯]` (Beihang University)

**通讯引用:** 11797 | [OpenAlex ID](https://openalex.org/A5042083053)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于运动谱描述符（MSD）的动态复杂度感知掩码生成框架 DynMask，改进了文本到运动生成的掩码、注意力和解码流程。

**💡 创新点**

创新点在于：① 引入参数无关、可解释的运动谱描述符来量化帧级动态复杂度；② 将该信号统一应用于掩码选择、运动感知注意力和解码时温度调节，实现全流程的复杂度感知；③ 通过谱相似性为自注意力注入运动结构先验。

**🔧 技术方法**

使用技术包括：短时 DCT 计算运动谱描述符；VQ‑VAE 对运动进行离散化；掩码 Transformer（MoMask、MMM 等）框架；文本条件采用 CLIP 文本编码器；谱相似性融合注意力；温度与噪声自适应采样。

**📊 数据集**

主要实验数据集：HumanML3D、KIT-ML；跨域零样本评估使用 BABEL；在这些数据集上做了量化和主观评估。

**📈 对比分析**

与多种基准（Diffusion、Token‑based、Masked）对比，DynMask 在 HumanML3D 上 FID 下降至 0.028（比 MoMask 降 38%），在 KIT‑ML 上 FID 0.141；同时提升 MM‑Dist、Diversity，整体在Masked家族中排名第一；在 BABEL 零样本迁移时 FID 从 14.889 降到 11.543，提升 22.5%。

**⚠️ 局限性**

局限性：对极其简单的静态动作可能略有性能下降；推理时需频繁重算 MSD，导致速度略慢；仅在文本到运动的单模态场景验证，未评估更长文本或多模态融合。

---

## 419. Semantic Interaction for Narrative Map Sensemaking: An Insight-based Evaluation

**arXiv ID:** 2603.29651 | [PDF](https://arxiv.org/pdf/2603.29651v1)

**作者:** Brian Felipe Keith-Norambuena `[一作]` (Universidad Católica del Norte), Chris North `[通讯]` (Virginia Tech)

**通讯引用:** 8730 | [OpenAlex ID](https://openalex.org/A5037675411)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对叙事地图的语义交互（SI）框架进行实验评估，比较三种可视化条件（时间轴、基本叙事地图、交互叙事地图）对分析者洞察生成和阅读行为的影响。

**💡 创新点**

首次实证验证叙事地图优于传统时间轴，并识别出两种SI策略（纠正与增补）对模型细化与分析者决策的贡献，同时揭示信任与解释性挑战。

**🔧 技术方法**

使用Transformer句子嵌入（all‑MiniLM‑L6‑v2）+ UMAP + HDBSCAN 进行文本嵌入与聚类，基于线性规划构建叙事地图，采用混合多模型SI框架实现边/节点/簇交互，并通过Insight‑based Evaluation 方法评估效果。

**📊 数据集**

以2021年古巴抗议的160篇新闻文章为实验数据，训练阶段使用COVID‑19数据集进行系统熟悉。

**📈 对比分析**

采用 33 名参与者的三组实验（时间轴、基本地图、交互地图）进行 Insight‑based 评估，结果显示叙事地图显著提升洞察数量和知识差距闭合度；交互地图在高阶洞察上优于时间轴，基本地图介于两者之间，效应量大（d>0.8），但由于样本量不足，交互 vs 基本差异未达到显著水平。

**⚠️ 局限性**

样本量小且主要为学生，组别不平衡，任务时长短、数据集规模有限，未测量认知负荷，交互推断洞察可能偏倚，系统可扩展性受限；因此需要更大规模、更专业的受试者和更大规模数据集来进一步验证SI的定量优势。

---

## 420. Big2Small: A Unifying Neural Network Framework for Model Compression

**arXiv ID:** 2603.29768 | [PDF](https://arxiv.org/pdf/2603.29768v1)

**作者:** Jing-Xiao Liao `[一作]` (City University of Hong Kong), Feng-Lei Fan `[通讯]` (City University of Hong Kong)

**通讯引用:** 1768 | [OpenAlex ID](https://openalex.org/A5018677035)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出统一的模型压缩理论并基于隐式神经表示实现无数据后训练压缩方法Big2Small

**💡 创新点**

将低秩、剪枝、量化、可测动力系统和知识蒸馏统一到测度理论框架，证明其与神经网络等价，并首次将INR用于压缩权重；同时提出异常值预处理与频率感知损失提升重建质量

**🔧 技术方法**

测度理论、通用压缩定理、结构等价定理、隐式神经表示（SIREN/MLP）、异常值预处理、频率感知损失、可选量化与其他压缩融合

**📊 数据集**

ImageNet分类、Carvana分割、ResNet/ResNet50/Swin/T/Swin/Swin-S、UNet/UNet++/R2UNet等公开模型

**📈 对比分析**

与DSG、Squant、UDFC、RieM等基线进行压缩率、模型尺寸、Top-1准确率、mIOU等指标比较；Big2Small在多种网络上实现5.9×-7.7×压缩率，Top-1可达73.98%，mIOU可达95.32%，表现与最优基线相当甚至更优

**⚠️ 局限性**

推理时需解码INR，导致约30%延迟；需要为每层训练INR，计算成本较高；对极端稀疏或动态网络的适配有限；过高压缩率时精度下降明显

---

## 421. Training-Free Dynamic Upcycling of Expert Language Models

**arXiv ID:** 2603.29765 | [PDF](https://arxiv.org/pdf/2603.29765v1)

**作者:** Eros Fanì `[一作]` (Gensyn), Oğuzhan Ersoy `[通讯]` (Gensyn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将多个已训练的单域密集模型合并为一个多域Mixture-of-Experts模型的方法，无需额外的多任务训练。

**💡 创新点**

创新点在于利用岭回归的闭式解初始化路由器，并通过一次前向传播即可完成参数聚合，显著降低成本并避免交叉干扰。

**🔧 技术方法**

采用了MoErging（平均非MLP层）、MoE结构、岭回归及其序列更新性质。

**📊 数据集**

使用了115M与3B LLaMA模型在OpenWebText、M2D2多域（编程、数学、物理、历史、哲学等）以及四个推理任务（HumanEval、GSM8k、M_ARC、IFEval）等数据集。

**📈 对比分析**

与BTX、随机路由、模型平均等基线相比，该方法在CLM与推理任务上均获得更低的perplexity/更高的准确率，甚至超越Oracle（使用测试域标签）。

**⚠️ 局限性**

局限在于训练阶段仍需域标签，若域划分不明确或不可获取则难以使用；此外对非常大规模模型的扩展性尚待验证。

---

## 422. Detecting speculative leaks with compositional semantics

**arXiv ID:** 2603.29800 | [PDF](https://arxiv.org/pdf/2603.29800v1)

**作者:** Xaver Fabian `[一作]` (CISPA Helmholtz Center for Information Security), Andres Sanchez `[通讯]` (Amazon)

**通讯引用:** 70 | [OpenAlex ID](https://openalex.org/A5078759384)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套基于投机非干涉（SNI）的形式化框架，并实现工具 Spectector，用来检测与验证程序在各种投机执行机制（分支、间接跳转、存储、返回等）下的侧信道泄漏。

**💡 创新点**

创新点包括：① 可组合的投机执行语义模板与“总是误预测”抽象，能够捕捉单一机制及其交互泄漏；② SNI 的安全定义将投机语义与标准语义对比；③ 通过组合框架在保持形式化证明的同时支持多机制的叠加；④ 使用符号执行与 SMT 交叉验证的自动化检查算法。

**🔧 技术方法**

使用技术：形式化语义与操作语义推导、符号执行、Z3 SMT 求解、逻辑程序设计（ASP）实现分析引擎，以及组合语义的元参数机制。

**📊 数据集**

数据集：传统 Spectre 相关的安全基准程序；自定义的多机制交互泄漏案例；以及 Spectector 仓库中的公开脚本与基准集合。

**📈 对比分析**

与现有基于模式匹配或单机制分析方法对比，Spectector 能检测所有已知泄漏及新组合泄漏；在微基准上评估显示工具执行时间从几秒到几十秒，证明了其实用性和性能可接受。

**⚠️ 局限性**

局限性：仅使用了对内存访问和跳转的粗粒度观察模型，未覆盖更细粒度的微架构细节；对部分 x86 指令与子寄存器支持不足；组合机制需要手动指定元参数；对未来未知投机机制的适配仍有限。

---

## 423. HLC: A High-Quality Lightweight Mezzanine Codec Featuring High-Throughput Palette

**arXiv ID:** 2603.29864 | [PDF](https://arxiv.org/pdf/2603.29864v1)

**作者:** Chenlong He `[一作]` (Fudan University), Yibo Fan `[通讯]` (Fudan University)

**通讯引用:** 2196 | [OpenAlex ID](https://openalex.org/A5085004179)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种混合型轻量级中间码流编码器 HLC，针对 4K@120fps 的实时视频流实现高质量压缩。

**💡 创新点**

创新点包括：① 通过虚拟聚类表实现无数据依赖的调色板（PLT）架构；② 与传统预测模式协同的 RDO 设计，并精调 QP–λ 表；③ 将 RDO 估计结果直接复用至熵编码阶段，显著降低硬件资源。

**🔧 技术方法**

使用技术包括：调色板聚类、运行长度索引映射、16×4 CU 级流水线、二维离散小波变换 (DWT)、量化、固定+可变长度熵编码、FPGA 资源优化。

**📊 数据集**

评估数据集为三类内容：文本（TEC）、游戏（GAC）和自然图像（NAC）四个目标比特率。

**📈 对比分析**

通过在 KC705 FPGA 上与 JPEG‑XS、JPEG‑2000 以及简化的 HEVC‑Intra 进行对比，HLC 在 4K@120fps 维持相同吞吐量、LUT 使用仅为 JPEG‑XS 的一半，并在三类数据集上分别获得 BD‑PSNR 提升 3.461dB、3.299dB、5.312dB。

**⚠️ 局限性**

局限性在于：① 无依赖 PLT 的聚类采用固定初始中心，导致文本内容的 BD‑PSNR 轻微下降（≈0.12dB）；② 仍需 24K LUT 的额外资源来实现调色板与 RDO；③ 对极高分辨率或超宽带宽场景的可扩展性尚未充分验证。

---

## 424. CADReasoner: Iterative Program Editing for CAD Reverse Engineering

**arXiv ID:** 2603.29847 | [PDF](https://arxiv.org/pdf/2603.29847v1)

**作者:** Soslan Kabisov `[一作]` (Lomonosov Moscow State University), Dmitrii Zhemchuzhnikov `[通讯]` (Lomonosov Moscow State University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练一个能够迭代自我编辑 CadQuery 程序的模型 CADReasoner，通过对比输入扫描与当前渲染的几何差异，逐步生成更精确的 CAD 模型。

**💡 创新点**

创新点在于：① 采用闭环自我纠错机制，将几何差异直接作为监督信号；② 将多视角图像和点云作为互补模态输入；③ 在训练与评估中使用扫描仿真，提升模型在真实扫描环境下的鲁棒性；④ 通过单模型 SFT 训练实现迭代编辑，无需额外 RL 微调。

**🔧 技术方法**

技术手段包括：Qwen2‑VL 2B LLM 编辑器、SFT + on‑policy 轮询训练、几何差异编码（多视角重叠 + 最近面偏移）、扫描仿真协议、贪心解码与随机束搜索解码。

**📊 数据集**

使用的主要数据集：CAD‑Recode（约 1M 程序）用于训练；评估基准为 DeepCAD、Fusion360、MCB（各自提供清洁与扫描仿真两轨）。

**📈 对比分析**

评价方法：采用 Chamfer Distance、IoU、Invalid Rate 三指标；与传统 SFT（cadrille‑SFT）和 RL 微调（cadrille‑RL）等基线对比；结果显示 CADReasoner 在 CD、IoU 上均领先，且 Invalid Rate 降为 0，迭代 5 步后性能大幅提升。

**⚠️ 局限性**

局限性：① 对极端噪声、遮挡或大缺失的扫描鲁棒性仍有限；② 依赖扫描仿真，真实环境中可能出现未覆盖的缺陷；③ 训练需要海量程序数据与算力；④ 现有模型仅利用几何差异，缺乏更细粒度的语义或物理约束。

---

## 425. Curvature-Guided LoRA: Steering in the pretrained NTK subspace

**arXiv ID:** 2603.29824 | [PDF](https://arxiv.org/pdf/2603.29824v1)

**作者:** Frédéric Zheng `[一作]` (KTH), Alexandre Proutière `[通讯]` (KTH)

**通讯引用:** 5973 | [OpenAlex ID](https://openalex.org/A5025136069)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了预测对齐（prediction alignment）问题，并基于此设计了 Curvature‑Guided LoRA（CG‑LoRA）算法，用低秩适配器实现参数高效微调时预测输出与全微调相匹配的目标。

**💡 创新点**

核心创新是将预测对齐目标转化为曲率感知的二阶（Newton‑style）初始化，通过白化梯度（whitened gradient）构造低秩方向，实现对全微调功能的精准跟踪；相较传统只对参数梯度进行对齐，CG‑LoRA更直接关注模型输出。

**🔧 技术方法**

采用了 K‑FAC（Kronecker‑factored curvature）近似求解曲率张量、随机化 sketch‑QR、Rayleigh‑Ritz 以及 SVD 等技术实现低秩子空间与白化梯度的高效计算；同时利用第二阶信息构造适配器初始化。

**📊 数据集**

在 GLUE 基准上，使用 T5‑base 与 RoBERTa‑base 两种模型，对 MNLI、SST‑2、CoLA、QNLI、MRPC 等任务进行实验。

**📈 对比分析**

将 CG‑LoRA 与完整微调、LoRA‑GA、LoRA‑One、rsLoRA 等方法在同一实验设置下比较；结果显示 CG‑LoRA 在大多数任务上取得更高准确率、收敛更快、对学习率更稳健，基本逼近完整微调性能。

**⚠️ 局限性**

局限性包括：实验仅在 GLUE 任务与小规模模型上验证，未覆盖生成式或更大规模模型；依赖 NTK 假设，曲率近似的准确性可能随网络深度/架构变化；相较于传统 LoRA，需额外几次反向传播，导致一定计算开销。

---

## 426. SceneTeract: Agentic Functional Affordances and VLM Grounding in 3D Scenes

**arXiv ID:** 2603.29798 | [PDF](https://arxiv.org/pdf/2603.29798v1)

**作者:** Léopold Maillard `[一作]` (École Polytechnique), Maks Ovsjanikov `[通讯]` (École Polytechnique)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于几何验证的场景可行性引擎（SceneTeract），通过 VLM 规划将复杂任务拆分为原子动作，并对每一步进行 agent‑aware 的几何和物理检查，得到细粒度的功能诊断报告，用于场景审计、VLM 评测以及后期训练奖励。

**💡 创新点**

创新点：① 将语义规划与低层几何验证解耦；② 设计原子动作库与布尔属性映射，实现 agent‑aware 的可行性检测；③ 生成可解释的多级诊断报告；④ 将验证结果作为奖励（GRPO）进行 VLM 后期微调，显著提升模型的空间功能推理。

**🔧 技术方法**

技术手段：VLM 规划（Gemini 3 Flash）、几何检查（libigl、Trimesh、SAM、Molmo）、机器人运动学与碰撞检测、强化学习（GRPO）、参数高效微调（LoRA）、多指标评估（MCC、FP、HSI、InGap、Consistency）。

**📊 数据集**

数据集：3D‑FRONT（居家生活房间与餐厅子集，1,132 场景）与 3D‑FUTURE 物体资产，用于生成原子动作和验证标签，并在此数据集上评测 VLM 模型。

**📈 对比分析**

评测方法：将 SceneTeract 生成的几何验证标签作为基准，对多款前沿 VLM（Gemini、Claude、Gemma、Qwen 等）进行直接任务级与分解原子动作级的预测，计算任务准确率、FP、MCC、HSI、Consistency、InGap。结果显示：① 直接预测存在大量物理幻觉；② 任务拆分后各模型性能显著提升；③ 通过 GRPO 后训练的轻量级 VLM 在原子动作和任务级别均可逼近专有前沿模型。

**⚠️ 局限性**

局限性：仅评估静态场景，未考虑动态交互与环境更新；几何验证需要集成多种 3D 工具，工程成本高；当前未实现闭环的场景合成改进，未来可作为反馈引导生成系统。

---

## 427. GENIE: Gram-Eigenmode INR Editing with Closed-Form Geometry Updates

**arXiv ID:** 2603.29860 | [PDF](https://arxiv.org/pdf/2603.29860v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 428. Same Rules, Mixed Messages: Exploring Community Perceptions of Academic Dishonesty in Computing Education

**arXiv ID:** 2603.29762 | [PDF](https://arxiv.org/pdf/2603.29762v1)

**作者:** Chandler C. Payne `[一作]` (Georgia Institute of Technology), Pedro Guillermo Feijóo-García `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 193 | [OpenAlex ID](https://openalex.org/A5009998219)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过在美国东南部一所高校开展问卷调查，收集了6名教师、21名助教和538名本科生对13个学术不端情境的判定和动机解释，并对数据进行定量和定性分析。

**💡 创新点**

创新点在于首次系统比较教师、助教和学生三类主体对学术不端的定义差异及动机，并将年龄、宗教等人口统计因素与不端判定关联，揭示不同群体认知的深层差异。

**🔧 技术方法**

主要使用在线问卷平台收集数据，采用描述性统计、Spearman相关分析和主题分析（Thematic Analysis）对开放式回答进行编码，最后对三组间的比例差异进行统计检验。

**📊 数据集**

数据集为566份问卷答卷，包含13个情境分类（未作弊、轻度作弊、严重作弊）以及开放式动机回应，并记录性别、民族、年级、宗教、国内外身份等人口统计信息。

**📈 对比分析**

研究通过对比三组在情境分类上的百分比和动机主题出现频率，使用chi‑square检验和相关系数评估差异显著性；结果显示教师更关注“成绩压力”和“懒惰”，学生和助教更关注“先修知识不足”和“时间管理”，并且年龄与宗教与认定作弊的倾向呈负相关。

**⚠️ 局限性**

局限性包括样本仅来自单一高校且教师样本量小（仅6人），研究聚焦入门至中级CS课程，难以推广到更广泛或更高级的教学环境；此外，调查依赖自我报告，可能存在社会期望偏差。

---

## 429. One-for-All: A Lightweight Stabilized and Parameter-Efficient Pre-trained LLM for Time Series Forecasting

**arXiv ID:** 2603.29756 | [PDF](https://arxiv.org/pdf/2603.29756v1)

**作者:** Prasanjit Dey `[一作]` (Technological University Dublin), Bianca Schoen-Phelan `[通讯]` (Technological University Dublin)

**通讯引用:** 181 | [OpenAlex ID](https://openalex.org/A5013295291)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 One-for-All 框架，使用预训练的 LLM 结合 Gaussian Rank‑Stabilized LoRA（rsLoRA）对多变量时序任务进行参数高效的微调，兼容预测、分类和异常检测。

**💡 创新点**

创新点在于 rsLoRA 引入基于高斯分布的秩稳定化机制，提供低秩下的梯度稳定性证明（Theorem 1），使得仅 16 维秩即可实现 95% 的准确度，同时显著降低模型参数和内存占用。

**🔧 技术方法**

技术手段包括：冻结 GPT‑2 等 LLM 的自注意力层，在位置嵌入和输出层插入 rsLoRA，使用无位置编码的 patching + Z‑score 标准化，训练仅 0.55M 可训练参数，内存 2.2 MiB。

**📊 数据集**

在六类时序任务上评估：ETT、Weather、M3、M4、UEA 多分类数据集以及 SMD、MSL、SMAP、SWaT、PSM 异常检测数据集。

**📈 对比分析**

与多种基线（TimesNet、GPT4TS、TIME‑LLM、FEDformer、Autoformer 等）对比，One-for-All 在参数效率、内存占用和预测准确率上实现了最优平衡：参数量比 GPT4TS 降低 21×、内存缩小 1,776×，同时 MSE/MAE 维持与最优模型相当（如长期预测 MSE 0.33、短期 MSE 12.37% SMAPE）。

**⚠️ 局限性**

限制在于：(1) 对极低秩（如 2–4）时仍有一定性能下降；(2) 目前仅在 GPT‑2 作为骨干上验证，跨模型泛化尚未充分探索；(3) 训练过程中需手动设定秩与 α，缺少自动化选择机制。

---

## 430. Beyond AI advice -- independent aggregation boosts human-AI accuracy

**arXiv ID:** 2603.29866 | [PDF](https://arxiv.org/pdf/2603.29866v1)

**作者:** Julian Berger `[一作]` (Max Planck Institute for Human Development), Ralf H. J. M. Kurvers `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对10个高风险领域的实验数据进行对比分析，评估了传统 AI‑as‑advisor 与新提出的 Hybrid Confirmation Tree (HCT) 在人机决策中的准确性。

**💡 创新点**

创新点在于提出并验证了一种保持人机判断独立、利用两位独立人类评估并由二人协同决定的 HCT 框架，并用信号检测模型解释其优于 AI‑as‑advisor 的机制。

**🔧 技术方法**

所用技术包括：人机实验收集决策数据、贝叶斯广义线性混合模型 (GLMM) 与 ROPE 方法进行统计比较、以及基于等方差假设的信号检测理论模型来量化人类对 AI 建议的辨别与偏好。

**📊 数据集**

实验数据来自 10 个跨域数据集（皮肤癌诊断、结肠镜病变、深度伪造检测、假新闻真伪、再犯预测、情绪分类、欺骗检测等）共 41,000+ 人类决策，另有 50,000+ 含可解释 AI（XAI）条件的决策。

**📈 对比分析**

比较方法为对每个案例计算 HCT 与 AI‑as‑advisor 的准确率，使用贝叶斯估计和 1% 可信区间评估差异；结果显示 HCT 在所有数据集平均提升 4.45% 准确率，绝大多数 XAI 情形下仍优，且高低水平人类在 HCT 中均获益，tiebreaker 触发率约 20–49%。

**⚠️ 局限性**

限制包括：需要第二位人类进行分歧解决，导致约 20–49% 的额外人力成本；在某些领域（如欺骗检测）人类基本准确度低时 HCT 效果有限；对成本效益与长期技术影响的系统评估仍待进一步研究。

---

## 431. DiSGMM: A Method for Time-varying Microscopic Weight Completion on Road Networks

**arXiv ID:** 2603.29837 | [PDF](https://arxiv.org/pdf/2603.29837v1)

**作者:** Yan Lin `[一作]` (Aalborg University), Huaiyu Wan `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 6817 | [OpenAlex ID](https://openalex.org/A5065949777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为 DiSGMM 的微观权重补全框架，用于在交通网络中填补时变的车辆级速度分布。

**💡 创新点**

创新点包括：1）双层稀疏感知嵌入（静态与动态）实现对段级与网络级稀疏的自适应补全；2）基于 U‑net 的时空网络配合全局残差连接捕获长程空间与长期时间相关；3）使用可学习的高斯混合模型提供闭式、可解释且能捕捉多峰与重尾特征的分布表示。

**🔧 技术方法**

技术手段涵盖：图神经网络（GCN、Transformer、GAT 等）、U‑net 架构、全局残差连接、Fourier 特征映射、门控机制、图聚类、可学习高斯混合模型、随机游走（DeepWalk）等。

**📊 数据集**

在两个真实世界数据集上评估：HTT（高速公路收费站数据）和 CD（成都出租车 GPS 数据），每个数据集划分为训练/验证/测试，使用 96 个 15 分钟时隙。

**📈 对比分析**

与历史平均、GCWC、SSTGCN、ConGC、Nuhuo 及其 GMM 变体、以及宏观权重补全方法 STGNF、PriSTI 进行对比。DiSGMM 在所有缺失率 (50%–80%) 下均取得最高似然（Likelihood）和最低 CRPS，显示出显著优于现有方法的性能。

**⚠️ 局限性**

局限性：仅在两组交通数据上验证，可能对其他城市或不同测度（如时间间隔）推广性有限；模型复杂度高，训练与推理成本相对较大；缺乏对极端稀疏情形（如 >90% 缺失）的进一步评估。

---

## 432. Multimodal Machine Learning for Early Prediction of Metastasis in a Swedish Multi-Cancer Cohort

**arXiv ID:** 2603.29793 | [PDF](https://arxiv.org/pdf/2603.29793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 433. Friends, Foes, and First Authors: A Game Theory Model of How Power Plays Rewrite Academic Co-Authorship Networks

**arXiv ID:** 2603.29834 | [PDF](https://arxiv.org/pdf/2603.29834v1)

**作者:** Amit Bengal `[一作]` (Bar Ilan University), Teddy Lazebnik `[通讯]` (University of Haifa)

**通讯引用:** 1175 | [OpenAlex ID](https://openalex.org/A5041000511)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一种多轮网络博弈模型，模拟研究者在重复合作中通过提议ultimatum争夺作者顺序，并用深度强化学习训练前瞻性代理以最大化长期收益。

**💡 创新点**

创新点在于把单一项目的ultimatum游戏扩展到动态社交网络中，引入声誉反馈机制，并使用DRL学习长期策略，展示了前瞻性行为可消除破坏性终止、提升论文产出而不扩大不平等。

**🔧 技术方法**

技术手段包括：基于Python的agent‑based仿真、强化学习（Deep Q‑Network）实现前瞻性代理、图论算法计算网络连通度与最短路径、统计分析和可视化（PDF/PNG）等。

**📊 数据集**

数据集：完全基于仿真生成的约10,000名代理的友谊网络与合作网络，采用泊松/均匀分布初始化，未使用真实共著论文数据。

**📈 对比分析**

比较方法：在混合人口实验中系统地改变前瞻性代理比例（0%–100%），记录ultimatum次数、接受/撤回/终止比例、论文完成率、每人平均论文数、累计效用均值与标准差、Gini系数等指标；结果显示前瞻性代理虽不减少ultimatum提议，却将终止率从12%降至0%，完成率从85.3%提升至97%，平均论文数从15.2篇升至16.9篇，整体不平等指数基本不变。

**⚠️ 局限性**

局限性：假设代理完全知晓项目与网络状态；未考虑学科、机构层级或资历差异导致的作者规范；网络形成不受策略影响；所有代理同一贴现率与风险偏好；未对模型进行实证校准或与真实共著网络比较。

---

## 434. SIREN: Spatially-Informed Reconstruction of Binaural Audio with Vision

**arXiv ID:** 2603.29820 | [PDF](https://arxiv.org/pdf/2603.29820v1)

**作者:** Mingyeong Song `[一作]` (Ewha Womans University), Junhyug Noh `[通讯]` (Ewha Womans University)

**通讯引用:** 886 | [OpenAlex ID](https://openalex.org/A5088003950)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于视觉引导的单声道转双声道框架SIREN，显式预测左右声道并结合Transformer双头注意力与FiLM条件化的音频U‑Net，实现高质量的空间音频生成。

**💡 创新点**

创新点在于：1) 使用ViT双头自注意力直接学习左右声道指向性，避免手工掩码；2) 引入软可退化空间先验，初期引导左右定位；3) 在推理时采用基于单声道一致性和相位一致性的两阶段置信度加权融合，显著降低左右交叉与漂移。

**🔧 技术方法**

使用的技术包括DINOv3 ViT编码器、FiLM条件化音频U‑Net、APNet式左右头、STFT/逆STFT、单声道一致性与相位一致性置信度评分以及双头自注意力和两阶段加权融合。

**📊 数据集**

实验数据集为FAIR‑Play（10‑split与5‑split）和MUSIC‑Stereo。

**📈 对比分析**

与Mono2Binaural、Sep‑Stereo、CMC、CC‑Stereo等方法比较，SIREN在FAIR‑Play的STFT、ENV、Phs和SNR指标上取得最优或接近最优（STFT 0.820、Phs 1.550、SNR 7.219），在MUSIC‑Stereo上实现最低STFT/ENV/Phs并最高SNR，证明其性能优越。

**⚠️ 局限性**

局限性包括：1) 在受控录音环境下仍可能出现微小相位不一致导致定位误差；2) 推理时需对视频与音频进行同步裁剪与特征对齐，对GPU内存和实时性有一定负担；3) 对复杂多源场景的泛化尚未充分验证。

---

## 435. Multi-paradigm Logic Programming in the ${\cal E}$rgoAI System

**arXiv ID:** 2603.29819 | [PDF](https://arxiv.org/pdf/2603.29819v1)

**作者:** Michael Kifer `[一作]` (Stony Brook University), Theresa Swift `[通讯]` (Coherent Knowledge)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套名为ErgoAI（语言为Rulelog）的多范式逻辑编程系统，继承并扩展了Flora-2，融合了对象导向的F‑logic、HiLog、可证伪推理、事务逻辑、延迟子目标、有限理性等功能，并实现了动态模块、Python/Janus接口以及高效的编译与运行时库，旨在实现可扩展、可维护、可交易的知识表示与推理。

**💡 创新点**

创新点主要体现在：① 将多种逻辑编程范式（对象、递归、事务、可证伪、延迟、有限理性）整合到同一语言与系统；② 采用WFS语义实现非地面推理与无限递归的安全评估；③ 通过模块化盐化（salting）实现跨模块的独立性与动态加载；④ 自动子目标延迟与递归完成提升执行效率；⑤ 通过事务逻辑与完整性约束支持可回滚的更新；⑥ 引入三值真值（t/f/u）与四值解释（t/f/u/⊥）以实现稳健的并发与错误处理。

**🔧 技术方法**

技术实现依赖XSB Prolog引擎的SLG表格化推理、HiLog/Frame逻辑编译、GPP预处理、Janus Python桥接、事务逻辑与完整性约束、子目标抽象与约束（restraint）、规则描述符、论证理论（GCLP）以及多级模块系统与盐化。整个编译链将源代码转换为高效的Prolog表格程序，并在运行时提供解释、延迟与三值逻辑支持。

**📊 数据集**

主要实验数据集包括：DARPA AIDA项目的知识图谱（约1180万三元组）；Petri网可达性基准（最高8千万状态）；左递归传递闭包图（上至10^8顶点）；以及与Clingo、Soufflé等传统ASP/Datalog系统对比的标准图结构。

**📈 对比分析**

评估方法：在同一硬件（Apple M2 Pro）上与XSB、Clingo、Soufflé并行跑Petri网、传递闭包、非表格递归等基准。结果表明：ErgoAI在Petri网可达性上与XSB相差4–6倍，内存占用4–5倍；在左递归传递闭包上，ErgoAI仍保持线性扩展，但速度约为XSB的3–4倍；在一般Prolog递归程序上，ErgoAI略慢于XSB，但比纯XSB动态代码更快。整体来看，ErgoAI在保持可扩展性的同时，性能仍在可接受范围内。

**⚠️ 局限性**

局限性：① 由于支持帧、延迟与解释结构，表格空间和内存占用显著高于原生XSB；② 对无限模型需要手动开启子目标抽象/约束，增加使用复杂度；③ 解释、非终止分析等高级功能仍处于实验阶段；④ 与传统约束/逻辑规划工具相比，处理大规模图结构时性能略逊；⑤ 尚缺乏原生模糊/概率推理与高级类型检查；⑥ 交互式模块化与盐化虽然强大，但对开发者学习曲线有一定门槛。

---

## 436. M3SA: Exploring Datacenter Performance and Climate-Impact with Multi- and Meta-Model Simulation and Analysis

**arXiv ID:** 2603.29778 | [PDF](https://arxiv.org/pdf/2603.29778v1)

**作者:** Radu Nicolae `[一作]` (Vrije Universiteit Amsterdam), Alexandru Iosup `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 9035 | [OpenAlex ID](https://openalex.org/A5006986556)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了M3SA框架，支持对数据中心进行多模型和元模型仿真与分析。

**💡 创新点**

创新点在于首次将多种单模型预测集成为多模型层，并通过元模型（类似 bagging）聚合预测，显著提升准确性并增强可解释性。

**🔧 技术方法**

使用基于 OpenDC 的离散事件仿真器、八类能耗/CO₂/失败等预测模型、窗口化聚合、Parquet 存储、Mean Absolute Percentage Error（MAPE）等技术。

**📊 数据集**

使用了公开工作负载轨迹（SURF‑22、Marconi‑22、Solvinity‑13）和欧洲能源透明度平台（ENTSO‑E）CO₂ 轨迹，以及 Ldns04 失败轨迹。

**📈 对比分析**

与单一模型（如手工调优模型）对比，元模型将 MAPE 降低约 50%（从 7.59% 降至 3.81%），仿真 1–2 年工作负载仅耗时 5–7 分钟，元模型开销低于 20%。

**⚠️ 局限性**

局限包括仅在 OpenDC 上验证、元模型权重固定不动态调节、未探索更复杂的聚合方法（如 ML 或 MCDA），以及对更大规模或不同类型仿真器的可移植性仍需进一步评估。

---

## 437. Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration

**arXiv ID:** 2603.29773 | [PDF](https://arxiv.org/pdf/2603.29773v1)

**作者:** Fengyang Xiao `[一作]` (Duke University), Sina Farsiu `[通讯]` (Duke University)

**通讯引用:** 18071 | [OpenAlex ID](https://openalex.org/A5023633559)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 IQPIR 框架，通过将 NR‑IQA 模型产生的图像质量先验与离散码本结合，驱动重建网络实现对真实场景图像的高质量还原。

**💡 创新点**

创新点在于三方面：① 将质量先验作为 Transformer 的条件输入，形成可控的质量驱动生成；② 采用双码本结构，分离通用结构与高质量细节；③ 在离散空间进行质量优化，显著降低连续空间中的过度优化问题。

**🔧 技术方法**

技术核心包括 NR‑IQA 质量评估、离散向量量化码本、条件 Transformer、对抗式质量损失和多模型 NR‑IQA 集成。

**📊 数据集**

训练数据主要使用 FFHQ；在面部修复任务上评估 LFW‑Test、WebPhoto‑Test、WIDER‑Test；低光增强使用 LOL‑v1、LOL‑v2‑real、LOL‑v2‑synthetic；水下增强使用 UIEB；背光增强使用 BAID；以及低光目标检测任务使用 ExDark。

**📈 对比分析**

与 CodeFormer、Restormer、Reti‑Diff、MambaIR 等最新方法在多项指标（PSNR、SSIM、FID、BIQE、LPIPS、Detection AP）上均取得显著提升，尤其在主观评分和下游检测任务中表现最优。

**⚠️ 局限性**

局限在于依赖现有 NR‑IQA 先验，可能带来模型自身的偏差；此外，离散码本的容量和质量阈值选择仍需经验调优，未来需探索更鲁棒的质量先验融合与自适应训练策略。

---

## 438. Tracking vs. Deciding: The Dual-Capability Bottleneck in Searchless Chess Transformers

**arXiv ID:** 2603.29761 | [PDF](https://arxiv.org/pdf/2603.29761v1)

**作者:** Quanhao Li `[一作]` (Abbey Park High School), Wei Jiang `[通讯]` (Shanghai Soong Ching Ling School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练一种仅基于棋局走子序列的无搜索棋类引擎，使其在风格、错误与连贯性上像人类棋手而非最大化棋力。

**💡 创新点**

提出双能力瓶颈理论（状态跟踪T与决策质量Q）并通过Elo加权训练找到“甜点”平衡点；同时展示跟踪与决策的互斥性、规模化与权重调节的互补作用；首次在无棋盘输入下实现2500+级别bullet棋力。

**🔧 技术方法**

标准decoder-only Transformer（28M/120M参数），采用Pre-RMSNorm、RoPE、SwiGLU，训练时仅使用UCI走子序列，无任何棋盘表示、搜索或手工规则；通过线性/指数Elo加权对梯度进行软裁剪。

**📊 数据集**

使用全范围Lichess公开数据库（约600–2800 Elo，Bullet/Blitz）及数百万题库；训练集无过滤，后实验对不同Elo段进行加权或裁剪。

**📈 对比分析**

对比方法包括：在Lichess Bullet Elo、无搜索的对战、无监督Top‑1走子预测（与Maia‑2对比）、人类失误对齐、重复局面历史依赖实验。最终模型在Lichess Bullet赛季达到2570 Elo（Top‑1%），Top‑1走子预测55.2%（比Maia‑2高5pp），并通过人类失误对齐和历史依赖实验显示更具人类风格。

**⚠️ 局限性**

局限性：仅在象棋域验证，未对其他游戏扩展；甜点r值仅在1、20、200三点探索；模型规模仅有28M/120M；评价主要聚焦Bullet，未覆盖长时间控制；与Maia‑2的对比使用各自原生输入，难以单独衡量模型本身优劣；内在追踪与决策的瓶颈理论需进一步理论与实验验证。

---

## 439. An Interactive LLM-Based Simulator for Dementia-Related Activities of Daily Living

**arXiv ID:** 2603.29856 | [PDF](https://arxiv.org/pdf/2603.29856v1)

**作者:** Kruthika Gangaraju `[一作]` (Worcester Polytechnic Institute), Fengpei Yuan `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 403 | [OpenAlex ID](https://openalex.org/A5008565294)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基于大型语言模型的交互式Web模拟器，用于生成阿尔茨海默病患者在日常生活活动中的多轮行为，并收集专家对现实感的评估和护理者回应；

**💡 创新点**

创新点在于：①将LLM用于阶段化、情境化的患者行为生成；②设计了可配置的情景（痴呆阶段、护理环境、ADL任务）和基于四种护理策略的建议框架；③通过专家循环评估创建了失效模式分类，为后续模型改进提供具体依据；

**🔧 技术方法**

使用了GPT‑5系列的大型语言模型作为核心生成引擎；后端采用Flask代理实现安全调用；前端使用单页Web应用收集交互日志；

**📊 数据集**

未使用公开数据集，而是通过专家手工构造的情景配置与任务进度步骤，结合LLM生成的文本；

**📈 对比分析**

与专家评估对比，整体真实感评分平均约3.5–4.0（5分制），识别出任务/情境不匹配、阶段不符、过度合规等失败模式；未进行传统模型与基线的量化比较，仅呈现专家主观评分和定性反馈；

**⚠️ 局限性**

局限性包括：样本量小（14名专家、18场次）；情景覆盖稀疏，未能全面评估不同ADL与痴呆阶段的表现；专家反馈量有限，导致失效模式统计不稳；后续需扩大评估范围并加入更严格的任务定位与环境约束。

---

## 440. PosterReward: Unlocking Accurate Evaluation for High-Quality Graphic Design Generation

**arXiv ID:** 2603.29855 | [PDF](https://arxiv.org/pdf/2603.29855v1)

**作者:** Jianyu Lai `[一作]` (Hong Kong University of Science and Technology), Lei Zhu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 73812 | [OpenAlex ID](https://openalex.org/A5100394072)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种专门用于评估生成海报质量的奖励模型PosterReward，并构建了Poster-Preference-70K海报偏好数据集。

**💡 创新点**

创新点在于：①使用全自动AI评判框架代替人工标注，②引入多维度海报评估体系（视觉质量、AI瑕疵、文字准确、提示匹配、美学价值），③设计两阶段奖励模型（分析+打分）与两种模型（点值与对比），④采用多阶段级联训练与GRPO强化学习提升分析与评分的一致性。

**🔧 技术方法**

使用的技术包括多模态大型语言模型（MLLM）如Gemini、GPT-5、GLM-4.5v；多模态奖励模型（HPSv3、CLIP、DINOv3、Qwen3-VL-8B）；Chain-of-Thought、偏好对比训练、Brady–Terry损失、Group Relative Policy Optimization (GRPO)等。

**📊 数据集**

使用的数据集：Poster-Preference-70K（70k条海报对比），HPDv3、MMRB2、PosterRewardBench（包含PosterRewardBench-Basic与PosterRewardBench-Advanced）。

**📈 对比分析**

方法通过与现有奖励模型（ImageReward、HPSv3、UnifiedReward）和MLLMs（Gemini、GPT-5）对比，在PosterRewardBench和HPDv3上实现了近86%至87%的准确率，显著高于对比模型；在PosterBench上评估的生成模型中，PosterReward优化后生成的海报质量平均分与中位数均显著提升。

**⚠️ 局限性**

局限性包括：①仍依赖大型MLLMs，计算成本高；②对不同语言（尤其中文）文本评估仍有偏差；③多维度评估的权重仍是经验性设定，可能不适用于所有海报类型。

---

## 441. DIAL: Decoupling Intent and Action via Latent World Modeling for End-to-End VLA

**arXiv ID:** 2603.29844 | [PDF](https://arxiv.org/pdf/2603.29844v1)

**作者:** Yi Chen `[一作]` (University of Hong Kong), Xihui Liu `[通讯]` (University of Hong Kong)

**通讯引用:** 3918 | [OpenAlex ID](https://openalex.org/A5027234036)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为 DIAL 的端到端视觉‑语言‑动作框架，利用可微分的潜在意图瓶颈将 VLM 的高层推理与低层执行解耦，实现从语言指令到精准运动控制的完整闭环。

**💡 创新点**

创新点包括：① 通过 VLM 预测潜在视觉预见（latent visual foresight）形成严格的结构化瓶颈；② 两阶段解耦热身 + 全局端到端优化，避免梯度干扰；③ 将 System‑1 设计为潜在逆动力学模型，强制执行意图与动作的一致性；④ 利用跨体态人类演示提升物理先验和零样本泛化。

**🔧 技术方法**

技术实现：预训练 VLM（Qwen2.5‑VL‑3B / Qwen3‑VL）+ ViT 编码器；学习查询+MLP 预测潜在意图；流匹配 Diffusion Transformer（DiT）做动作生成；MSE 世界建模损失 + 运动匹配损失；两阶段训练（系统解耦热身 + 端到端 fine‑tune）。

**📊 数据集**

数据集：RoboCasa GR1 Tabletop 24 任务（全量 24k 轨迹 / 少量 2.4k 轨迹）；EgoDex 人类演示（27k 轨迹）用于跨体态预训练；真实机器人 IRON‑R01‑1.11 上的 Pick‑&‑Place 与 Pouring 任务（120 轨迹）。

**📈 对比分析**

与基准方法（Diffusion Policy、UWM、FLARE、GR00T‑N1.6、Qwen3‑VL 版本等）以及多种 ablation 进行比较。DIAL 在全量数据上平均成功率 70.2%，在 10× 训练样本下 58.3%，显著超越 FLARE（55.0%）和 GR00T（47.6%）。在现实机器人上，成功率超过 70% 且对组合、干扰和实例级泛化表现出鲁棒性。零样本泛化提升约 5–7%（如未见物体、未见组合）。

**⚠️ 局限性**

局限性：① 仍依赖 VLM 预训练 ViT，冻结后无法进一步 fine‑tune；② 对人类演示数据的依赖，缺乏此类数据时性能下降；③ 对动态或高频复杂任务的适配尚未充分验证；④ 两阶段训练复杂，需精细的热身调参；⑤ 在跨任务迁移或不同 VLM 结构时，需要重新对齐潜在空间。

---

## 442. Toward Generalizable Whole Brain Representations with High-Resolution Light-Sheet Data

**arXiv ID:** 2603.29842 | [PDF](https://arxiv.org/pdf/2603.29842v1)

**作者:** Minyoung E. Kim `[一作]` (LifeCanvas Technologies), Brian Nguyen `[通讯]` (LifeCanvas Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了CANVAS高分辨率LSFM全脑细胞标记数据集并提供细胞中心注释，同时评估了基于ConvMixer的目标检测和自监督3D-MAE的表示学习。

**💡 创新点**

首次公开完整的多标记全脑LSFM基准数据集、基于自监督MAE的体积表示学习方法，并展示其在复杂细胞形态下对检测精度的显著提升。

**🔧 技术方法**

使用卷积混合网络（ConvMixer）+非极大抑制层进行细胞检测，并采用DINOv2风格的3D Masked Autoencoder（MAE）进行自监督特征学习，辅以内容感知加权和多尺度裁剪。

**📊 数据集**

使用六种细胞类型（NeuN、cFos、PV、TH、GFAP、IBA1）的全脑LSFM体积数据，数据量约65–140 GB压缩，包含约45k训练和47k测试细胞中心。

**📈 对比分析**

对比实验中，基准模型在自身标记数据上F1最高可达0.83，但跨标记和跨区域泛化差；自监督MAE可在不使用标签的情况下提升检测后处理F1约22–86%，并在全标记模型上保持95%以上的重建损失；整体性能仍低于传统单标记深度网络。

**⚠️ 局限性**

主要局限是细胞中心标注量仍有限、标注过程耗时且易产生漏标，导致模型在稀疏或形态复杂细胞（如GFAP）上的准确率仍偏低，且跨标记泛化能力需进一步提升。

---

## 443. From Skeletons to Semantics: Design and Deployment of a Hybrid Edge-Based Action Detection System for Public Safety

**arXiv ID:** 2603.29777 | [PDF](https://arxiv.org/pdf/2603.29777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 444. Loss Gap Parity for Fairness in Heterogeneous Federated Learning

**arXiv ID:** 2603.29818 | [PDF](https://arxiv.org/pdf/2603.29818v1)

**作者:** Brahim Erraji `[一作]` (University of Lille), Aurélien Bellet `[通讯]` (Inria)

**通讯引用:** 6577 | [OpenAlex ID](https://openalex.org/A5014504793)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种新型的 Federated Learning 算法 EAGLE，旨在通过最小化客户端 loss gap 的方差来实现公平性，适用于数据异构场景；

**💡 创新点**

创新点在于将公平性目标从传统的 loss parity 转变为 loss‑gap parity，并通过在 FedAvg 基础上加入可调节的 loss‑gap 方差正则化项来实现；

**🔧 技术方法**

技术主要包括：FedAvg 训练框架、基于梯度加权的正则化、异构度量 Γ 的定义及其在收敛分析中的应用，以及 λ 超参数的调节；

**📊 数据集**

实验使用了合成数据、EMNIST、DirtyMNIST 三个数据集，分别用于线性模型和 CNN 模型的验证；

**📈 对比分析**

与 FedAvg、q‑FFL、AFL 等基线进行比较，结果表明 EAGLE 在保持整体准确率的同时显著降低了 loss gap 方差，尤其在高异构度下对表现最差的客户端提升显著；

**⚠️ 局限性**

局限性包括需事先估计每个客户端的最优局部损失 L*_k，且对 λ 参数调节要求较高，过大或过小都会导致性能下降或公平性失效；

---

## 445. AMShortcut: An Inference- and Training-Efficient Inverse Design Model for Amorphous Materials

**arXiv ID:** 2603.29812 | [PDF](https://arxiv.org/pdf/2603.29812v1)

**作者:** Yan Lin `[一作]` (Aalborg University), Morten M. Smedskjaer `[通讯]` (Aalborg University)

**通讯引用:** 9292 | [OpenAlex ID](https://openalex.org/A5022182707)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种名为AMShortcut的概率生成模型，用于高效逆向设计非晶材料。

**💡 创新点**

创新点在于学习一阶跳跃（shortcuts）来显著减少采样步骤，并采用可在一次训练中覆盖所有属性的灵活材质去噪器。

**🔧 技术方法**

使用了基于材料差分方程的扩散模型（Material SDE/ODE）和E(N)-equivariant图神经网络，结合自一致性损失学习跳跃。

**📊 数据集**

实验使用了三套非晶材料数据集：单元素a‑Si、硅酸盐a‑SiO2以及多元素玻璃MEG。

**📈 对比分析**

与CDVAE、MatterGen、Graphite、Material ODE/ SDE基线相比，AMShortcut在仅1-5步采样下可达到相同或更优的结构准确率和逆向设计误差，推算时间提升至99%以上。

**⚠️ 局限性**

局限性包括难以生成退火后的低能结构，需结合物理引导的HMC细化，且单步性能受属性嵌入随机化影响。

---

## 446. Multi-Feature Fusion Approach for Generative AI Images Detection

**arXiv ID:** 2603.29788 | [PDF](https://arxiv.org/pdf/2603.29788v1)

**作者:** Abderrezzaq Sendjasni `[一作]`, Mohamed-Chaker Larabi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种多特征融合框架，用于检测生成式AI图像的真实性。

**💡 创新点**

创新点在于将低层统计特征（MSCN）、中层纹理特征（MLBP）与高层语义特征（CLIP）三者进行统一融合，提升了检测的鲁棒性与跨模型泛化能力。

**🔧 技术方法**

技术上使用了Mean Subtracted Contrast Normalized（MSCN）统计、Multi-scale Local Binary Patterns（MLBP）纹理分析、CLIP视觉-语言嵌入，并通过标准化拼接后输入梯度提升、随机森林或SVM分类器。

**📊 数据集**

实验覆盖四个基准数据集：Synthbuster、PKU-4K、CIFAKE 与 FakeBench，包含多种文本到图像与图像到图像的生成模型。

**📈 对比分析**

与多种SOTA方法（CNNDetection、DMImageDetection、PatchForensics、Synthbuster）对比，融合模型在四个数据集上的准确率最高，达到96.6%（MCC 0.933），显著优于单一特征或现有单模检测器。

**⚠️ 局限性**

局限性包括依赖预训练特征提取器可能带来的偏差与计算开销，且对极高质量生成模型（如 VQDM）仍存在检测难度。

---

## 447. Associative Constructive Evolution: Enhancing Metaheuristics through Hebbian-Learned Generative Guidance

**arXiv ID:** 2603.29774 | [PDF](https://arxiv.org/pdf/2603.29774v1)

**作者:** Shanxian Lin `[一作]` (Tokushima University), Haichuan Yang `[通讯]` (Tokushima University)

**通讯引用:** 914 | [OpenAlex ID](https://openalex.org/A5069376593)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Associative Constructive Evolution（ACE）框架，将传统元启发式算法与可学习的生成性引导模型（GCA）耦合，实现对搜索轨迹的记忆与利用。

**💡 创新点**

创新点在于三大机制：Hebbian权重合并强化成功操作关联；引导采样将学习到的概率分布注入探索；符号抽象将频繁共现操作组化为可复用的宏操作。

**🔧 技术方法**

采用生成构造自动机（GCA）与softmax采样、Hebbian无梯度学习、温度控制、宏操作库构建等技术，保持原始元启发式的变异/粒子更新机制。

**📊 数据集**

实验数据集包括基于MCEMOL的分子设计任务（37种化学操作）以及40个15×15的SCMP迷宫导航基准。

**📈 对比分析**

与标准EA/PSO对比，ACE-PSO成功率提升27.5%（82.6% vs 55.1%），收敛速度加快49.6%；ACE-EA最终适应度提升10.1%，生成有效分子数增加96.8%。

**⚠️ 局限性**

主要局限包括额外的计算开销、难以单独评估各机制贡献、阈值敏感性、宏操作库增长与剪枝策略、以及对GPU加速与更广泛基准的需求。

---

## 448. TSHA: A Benchmark for Visual Language Models in Trustworthy Safety Hazard Assessment Scenarios

**arXiv ID:** 2603.29759 | [PDF](https://arxiv.org/pdf/2603.29759v1)

**作者:** Qiucheng Yu `[一作]` (City University of Hong Kong), Xin Tan `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了TSHA（Trustworthy Safety Hazards Assessment）基准，用于评估视觉-语言模型（VLM）在室内安全风险识别中的能力，并通过该基准对23种主流VLM进行系统实验。

**💡 创新点**

创新点包括①整合来自现有室内数据集、互联网图片、AIGC生成图像以及新拍摄图像的多来源高质量训练数据；②在测试集加入Sora2生成视频与腾讯Hunyuan全景图，提升场景复杂度与多危险共存的评估；③采用LLM（ChatGPT‑4o）链式推理进行自动标注与评价，并辅以人工审核，保证标签可靠性。

**🔧 技术方法**

技术手段包括：VLM的GRPO强化学习微调；利用AIGC模型（如Hunyuan、PixVerse等）生成安全场景；LLM链式推理生成多模态对话与评价；对开放式问答使用加权评价框架（S_QA=0.7·Accuracy+0.2·Conciseness+0.1·Coherence）。

**📊 数据集**

使用的数据集：训练集81,809问答对（来自NYU‑v2、MIT Indoor Scenes、互联网视频、AIGC图像及新拍摄图像）；测试集1,707问答对，包括传统图像、Sora2视频、Hunyuan全景图等六类来源。

**📈 对比分析**

实验采用准确率评估选择题、LLM评估开放式问答，并以S_overall=S_QA+S_CQ/2汇总。结果显示即使是最先进的封闭源模型平均仅得66分，开源模型63分；在TSHA上微调后，Qwen2.5‑VL‑3B提升18.3分，且在BLINK、MMStar等通用基准上亦获得1.6分以上提升，证明TSHA训练显著增强模型的安全风险识别与通用视觉推理能力。

**⚠️ 局限性**

局限性包括：开放式问答仍难以达到高准确率，模型对多危险并列的推理仍有限；评估高度依赖LLM的主观评分，可能引入偏差；目前数据主要以英语场景为主，缺乏跨文化、多语言的安全风险覆盖。

---

## 449. VectorGym: A Multitask Benchmark for SVG Code Generation, Sketching, and Editing

**arXiv ID:** 2603.29852 | [PDF](https://arxiv.org/pdf/2603.29852v1)

**作者:** Juan Rodriguez `[一作]` (Mila Quebec Ai Institute), Marco Pedersoli `[通讯]` (Mila Quebec Ai Institute)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VectorGym这一全面的多任务基准，涵盖 Sketch2SVG、SVG 编辑、Text2SVG 和 SVG 标注四项任务。

**💡 创新点**

创新点包括：①首次引入Sketch2SVG任务与人类精细编辑数据；②构建真实、复杂的 SVG 编辑语料；③提出基于渲染反馈的多任务强化学习方法与 VLM-as-a-Judge 评估。

**🔧 技术方法**

技术手段主要是：使用 Qwen3-VL 8B 通过 GRPO + 渲染奖励进行多任务强化学习；采用 VLM-as-a-Judge 评估；对任务采用人类标注、语义相似度、LPIPS、DINO 等指标。

**📊 数据集**

使用了从 GitHub 获得的 7,000+ 实际 SVG 样本（SVG-Stack），并通过人工标注生成 Sketch、Edit、Text、Cap 四套训练/验证/测试集。

**📈 对比分析**

在各任务上与主流 VLM 进行对比，发现经强化学习后的 8B Gym 模型在所有任务均超过大多数开源模型，且在某些任务甚至超越更大规模模型（如 Qwen3 235B），接近 GPT‑4o 的性能；Gemini‑3 Pro 仍保持最高整体分数。

**⚠️ 局限性**

局限性包括：①对高阶 SVG 原语（动画、渐变等）支持仍有限；②评估仍偏重视觉相似度，结构正确性难以完全捕捉；③基准规模相对较小，难以覆盖所有设计风格；④RL 训练对算力要求高，缺乏可复现性。

---

## 450. SNEAK: Evaluating Strategic Communication and Information Leakage in Large Language Models

**arXiv ID:** 2603.29846 | [PDF](https://arxiv.org/pdf/2603.29846v1)

**作者:** Adar Avsian `[一作]` (Georgia Institute of Technology), Larry Heck `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 6424 | [OpenAlex ID](https://openalex.org/A5003679010)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SNEAK 基准，评估语言模型在信息不对称环境下的选择性信息共享能力。

**💡 创新点**

创新点在于同时衡量信息传递效用（Utility）与信息泄露（Leakage），并通过 SoftScore 合并两者，形成可量化的单轮交互评估框架。

**🔧 技术方法**

采用行为评估技术：用预训练语言模型模拟盟友和潜伏者来估计消息的可识别度和泄露概率；并利用 Chain-of-Thought、Self-Enhanced Test-Time Scaling 与 Recursive Message Refinement 等推理时扩展方法。

**📊 数据集**

数据集基于 117 个语义类别的词汇归纳，生成 1,394 个 (类别、候选词集、秘密词) 交互实例，包含 5 条无秘密干扰信息。

**📈 对比分析**

与随机词、类别同义词、秘密同义词等基线以及 8 种大型语言模型（GPT、Claude、Gemini、DeepSeek、Llama、Qwen、Gemma、Mixtral）进行对比；实验显示大模型虽具高 Utility 但 Leakage 较大，SoftScore 远低于人类（人类 SoftScore 约 59，最佳模型约 25），说明仍缺乏人类水平的选择性沟通能力。

**⚠️ 局限性**

局限性在于仅为单轮、结构化的词汇任务，未涵盖多轮互动、复杂情境和真实人类解读；评估依赖模型模拟评审，可能与人类判断存在偏差；数据来源受限于语义类别范例，缺乏更丰富的现实语境。

---

## 451. Owl-AuraID 1.0: An Intelligent System for Autonomous Scientific Instrumentation and Scientific Data Analysis

**arXiv ID:** 2603.29828 | [PDF](https://arxiv.org/pdf/2603.29828v1)

**作者:** Han Deng `[一作]` (Shenzhen Loop Area Institute), Wanli Ouyang `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了Owl-AuraID，一套基于GUI原生交互的自适应实验室智能体系统，能够在不依赖专有API的前提下完成从样品操作、仪器软件控制到数据分析的完整流程。

**💡 创新点**

核心创新在于将大型语言模型与计算机使用代理相结合，构建可复用的两类技能（Type-1 GUI操作技能与Type-2分析脚本技能），实现了将仪器操作与分析逻辑模块化、可组合、可迭代的能力积累机制。

**🔧 技术方法**

使用技术包括：基于InnoClaw的智能体运行时、LLM驱动的对话与代码生成、图形化界面定位与自动化（CUA）、技能抽象与版本管理、以及多模态感知与物理执行控制。

**📊 数据集**

实验数据来源于10类高精度科研仪器（如UV‑Vis、PL、SEM、EDS、Micro‑CT、FTIR、NMR、AFM、EBSD、TGA）所产生的光谱、显微图像、重建体积、元素分布等原始实验数据集，未使用公开的标准数据集。

**📈 对比分析**

与传统基于API或CLI的自动化平台对比，Owl‑AuraID在无需预置工具接口的情况下即可完成多模态实验，示例实验显示其能够实现从样品定位到结果报告的全闭环；虽然论文未给出定量指标，但通过实测案例验证了其在不同仪器上的可迁移性和操作精度。

**⚠️ 局限性**

主要局限包括：技能获取仍需专家演示、缺乏自监督或探测式学习机制、对物理样品搬运仍需高级机器人配合、以及对极其复杂或新型仪器的适配需要进一步研究。

---

## 452. A Python Framework for Reaction--Diffusion--Chemotaxis Simulations on One-Dimensional Network Geometries

**arXiv ID:** 2603.29807 | [PDF](https://arxiv.org/pdf/2603.29807v1)

**作者:** Silvia Bertoluzza `[一作]` `[通讯]` (Istituto di Matematica Applicata e Tecnologie Informatiche 'E. Magenes' --- CNR), Silvia Bertoluzza (Istituto di Matematica Applicata e Tecnologie Informatiche 'E. Magenes' --- CNR)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了 BioNetFlux，一个用于一维网络几何结构上的化学趋化、扩散反应耦合偏微分方程数值仿真的开源 Python 框架。

**💡 创新点**

创新点在于结合低阶 Hybridizable Discontinuous Galerkin（HDG）空间离散、隐式 Backward‑Euler 时间积分与自适应时间步长控制，并通过静态约简实现仅在网络骨架节点求解全局系统，从而显著降低计算成本和实现多种耦合边界条件。

**🔧 技术方法**

采用的技术包括 HDG 离散（含自适应稳定化参数）、Newton–Raphson 非线性求解、线搜索与阻尼策略、以及基于 SymPy 预计算的参考单元矩阵与动态组装。

**📊 数据集**

使用的数据集为从 CSV 文件（points.csv 与 lines.csv）构造的 29 段迷宫网络（maze_3_data），以及相应的物理参数表和初始/边界条件定义。

**📈 对比分析**

在迷宫几何上的四方程 OoC 模型仿真中，通过自适应时间步长保持了免疫细胞质量守恒，并且相比固定时间步长方法，显著减少了 Newton 迭代次数和总计算时间，验证了方法的鲁棒性与效率。

**⚠️ 局限性**

局限性包括目前仅实现低阶（p=1）HDG，缺乏高阶多项式支持；仅针对 1D 网络，无法直接扩展到二维/三维；静态约简对非线性项未做迭代改进，导致极端非线性场景下收敛困难。

---

## 453. Reasoning-Driven Synthetic Data Generation and Evaluation

**arXiv ID:** 2603.29791 | [PDF](https://arxiv.org/pdf/2603.29791v1)

**作者:** Tim R. Davidson `[一作]` (EPFL), Hamza Harkous `[通讯]` (Google)

**通讯引用:** 605 | [OpenAlex ID](https://openalex.org/A5044726738)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Simula 框架，利用无种子、基于推理的流程生成可解释、可控的大规模合成数据。

**💡 创新点**

创新点在于将 taxonomy 构建、meta‑prompt 生成、双重批评家和复杂度校准相结合，能在多维度（多样性、复杂度、质量）上精准调控合成数据。

**🔧 技术方法**

技术手段包括大型语言模型（Gemini 2.5 Flash）、Best‑of‑N 采样、生成‑批评循环、Elo 评分的复杂度评估以及基于 taxonomy 的覆盖测量。

**📊 数据集**

实验使用 CTI‑MCQ、CTI‑RCM、LEXam、GSM8k、Global MMLU 等多领域文本数据集。

**📈 对比分析**

通过与无 taxonomy、无批评的基线以及不同组件组合的消融实验，对内在指标（embedding 多样性、taxonomy 覆盖、复杂度分布）和下游 Fine‑Tuning 准确率进行比较，完整系统在所有数据集上均优于基线，特别在多样性与复杂度覆盖上提升显著。

**⚠️ 局限性**

局限性包括仅使用单一模型族（Gemini），实验未涉及多模态，批评家效果受教师模型性能限制，且合成过程高度依赖模型演进，迁移到新模型需进一步验证。

---

## 454. MAPLE: Multi-Path Adaptive Propagation with Level-Aware Embeddings for Hierarchical Multi-Label Image Classification

**arXiv ID:** 2603.29784 | [PDF](https://arxiv.org/pdf/2603.29784v1)

**作者:** Boshko Koloski `[一作]` (Jožef Stefan Institute), Sašo Džeroski `[通讯]` (Jožef Stefan Institute)

**通讯引用:** 19426 | [OpenAlex ID](https://openalex.org/A5064609702)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 MAPLE 框架，实现多路径层次多标签分类，融合文本语义、图卷积推理与自适应多模态融合。

**💡 创新点**

创新点包括：①多 token ViT 与 GraphSAGE 结合的层次推理；②基于文本描述的层次语义初始化；③自适应层级目标与多模态门控机制。

**🔧 技术方法**

使用 ViT、GraphSAGE GCN、句子 Transformer、交叉熵/二进制交叉熵混合损失以及自适应门控融合。

**📊 数据集**

在 CORINE 对齐的遥感数据集 AID、DFC‑15、MLRSNet 上验证，同时在医疗影像与细粒度视觉分类数据集（如 PadChest、ETHEC）进行泛化实验。

**📈 对比分析**

与平面 MLC 基线和现有 HMLC 方法（C‑HMCNN、HiMulConE、HMI）比较，MAPLE 在叶节点 AUPRC 上达到 0.872/0.987/0.967，少样本环境下提升高达 42%，仅增加 2.6% 参数。

**⚠️ 局限性**

局限性：在大规模数据集上提升幅度减小；需专家手工定义层次结构；仅为监督式方法，对未标记数据支持不足。

---

## 455. Wildfire Suppression: Complexity, Models, and Instances

**arXiv ID:** 2603.29865 | [PDF](https://arxiv.org/pdf/2603.29865v1)

**作者:** Gustavo Delazeri `[一作]` (Universidade Federal do Rio Grande do Sul), Marcus Ritt `[通讯]` (Universidade Federal do Rio Grande do Sul)

**通讯引用:** 1093 | [OpenAlex ID](https://openalex.org/A5086717490)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究基于图模型的时间约束火灾抑制资源分配问题，并提出新的混合整数规划模型和物理逼真的实例生成器。

**💡 创新点**

证明该问题NP-完整，提出性能优异的MIP模型以及基于Rothermel表面火势模型的实例生成器。

**🔧 技术方法**

采用混合整数规划、逻辑Benders分解、迭代局部搜索和迭代束搜索等算法。

**📊 数据集**

构造了新的网格实例集，并与文献中的20×20网格基准实例进行对比。

**📈 对比分析**

在所有基准上，新的MIP和IBS表现最佳；IBS在大多数实验中胜过其他方法，LBBD和随机搜索效果最差。

**⚠️ 局限性**

缺乏紧密的下界和对大规模实例的可扩展性；实例生成器主要适用于网格图，未涵盖更复杂拓扑结构。

---

## 456. AutoFormBench: Benchmark Dataset for Automating Form Understanding

**arXiv ID:** 2603.29832 | [PDF](https://arxiv.org/pdf/2603.29832v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 457. Towards Empowering Consumers through Sentence-level Readability Scoring in German ESG Reports

**arXiv ID:** 2603.29861 | [PDF](https://arxiv.org/pdf/2603.29861v1)

**作者:** Benjamin Josef Schüßler `[一作]` (University of Augsburg), Jakob Prange `[通讯]` (University of Augsburg)

**通讯引用:** 372 | [OpenAlex ID](https://openalex.org/A5022361649)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过对德语ESG报告的句子进行可读性标注，构建了面向普通读者的句子级可读性评估任务，并对多种模型进行实验与对比。

**💡 创新点**

创新点在于首次在ESG语境下进行句子级可读性研究，提出了基于句法特征的透明白盒模型，并与传统可读性公式、预训练语言模型及生成式LLM进行系统比较；同时，针对德语ESG报告的特殊性，采用了自定义的词性n-gram、句法树深度、平均依存距离等特征。

**🔧 技术方法**

技术方法包括：
- 句法特征提取（POS n-gram、树深度、平均依存距离、根词性、被动语态、从属连词）并在前馈网络中训练；
- 传统可读性公式（Flesch、HKPS、Gunning-Fox、SMOG、Vienna、LIX）整合后用XGBoost回归；
- 基于XLM‑RoBERTa的预训练编码器-分类器；
- 生成式LLM（Qwen 3‑4B、Gemma 3‑4B、Llama 3‑8B）通过提示式指令进行四类可读性预测。

**📊 数据集**

使用SustainEval GermEval 2025 ESG报告数据集，该数据集包含从德语可持续发展代码中抽取的句子块（train/ dev/ eval），每句有5名标注者给出1–4的可读性等级，后经众数投票归一化为0–1。

**📈 对比分析**

评估指标包括MSE、MAE、Kendall τ。句法特征模型与传统公式在误差和相关性上相近；XLM‑RoBERTa base 在MSE最低且推理速度最快；Qwen 在Kendall τ上表现最好，说明其更能区分易/难句子，但整体误差最高。平均融合模型略微提升了误差，但速度明显下降。

**⚠️ 局限性**

主要局限：
- 模型仅考虑目标句子，缺乏上下文导致与人工评估偏差；
- 标注主观性高、标注一致性仅为中等；
- 数据严重倾向易读句子，导致类别不平衡；
- LLM未针对任务微调，导致数值误差大；
- 计算资源受限，导致大模型表现不佳。

---

## 458. AgentFixer: From Failure Detection to Fix Recommendations in LLM Agentic Systems

**arXiv ID:** 2603.29848 | [PDF](https://arxiv.org/pdf/2603.29848v1)

**作者:** Hadar Mulian `[一作]` (IBM Research), Segev Shlomov `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套完整的 LLM 代理系统验证框架，包含 15 个输入/提示/输出校验工具、2 个根因分析模块，并在 IBM CUGA 上进行实测，显著提升了中小模型的准确率。

**💡 创新点**

创新点在于将规则式校验与 LLM-as-a-Judge 结合，形成跨输入、提示、输出的多层验证；通过交互式对话实现验证自我反思；并提供可落地的错误分类与改进建议。

**🔧 技术方法**

采用正则/AST 等规则校验、LLM-as-a-Judge、语义一致性检测、跨阶段信息一致性检查、链式解析、OpenTelemetry 观测与可视化仪表盘等技术。

**📊 数据集**

使用 AppWorld 任务模板、WebArena GitLab 子集以及 IBM CUGA 生产日志进行评估。

**📈 对比分析**

在 AppWorld 与 WebArena 上对 GPT‑4o、LLaMA‑4、Mistral 三个模型进行 pass@3/avg 对比；验证后中小模型准确率提升约 7–10%，与 GPT‑4o 相距不大，且未出现显著回归。

**⚠️ 局限性**

局限性包括：框架针对 IBM CUGA 的实现可能在其他多代理架构下效果有限；工具集尚未覆盖所有语义偏差；LLM-as-a-Judge 误判可能导致错误修复；部署时的实时性能与成本尚未充分评估。

---

## 459. Cold-Starts in Generative Recommendation: A Reproducibility Study

**arXiv ID:** 2603.29845 | [PDF](https://arxiv.org/pdf/2603.29845v1)

**作者:** Zhen Zhang `[一作]` (Shandong University), Zhaochun Ren `[通讯]` (Leiden University)

**通讯引用:** 7265 | [OpenAlex ID](https://openalex.org/A5100384130)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性复现并评估生成式推荐在用户和商品冷启动场景下的性能，并通过统一实验框架分离模型规模、标识符设计与训练策略的影响

**💡 创新点**

首次提供统一冷启动评估协议，揭示生成式推荐在商品冷启动的严重性能瓶颈、模型规模提升的有限作用、标识符设计的权衡及强化学习的负面影响，为冷启动研究指明关键改进方向

**🔧 技术方法**

基于预训练语言模型的生成式推荐、不同标识符编码（原子ID、文本标题、语义量化代码）、强化学习与监督微调、模型规模对比（Flan‑T5 variants）

**📊 数据集**

Amazon‑Toys、MicroLens、Steam三大大规模交互数据集，包含丰富商品文本信息

**📈 对比分析**

对比传统序列模型（SASRec、GRU4Rec）与八种生成式推荐模型，在温启动、用户冷启动与商品冷启动三种协议下计算Recall@10和NDCG@10；结果显示生成式推荐在温启动和用户冷启动相对稳定，但在商品冷启动时性能大幅下降；文本标识符可提升冷启动但在温启动下退化；强化学习并未带来普遍提升

**⚠️ 局限性**

实验未覆盖多模态信息、动态商品库和长期用户偏好变化，缺乏对标识符设计机制的理论解释，且强化学习策略对冷启动的适用性需进一步研究

---

## 460. Pattern-Sparse Tree Decompositions in $H$-Minor-Free Graphs

**arXiv ID:** 2603.29825 | [PDF](https://arxiv.org/pdf/2603.29825v1)

**作者:** Dániel Marx `[一作]` (CISPA Helmholtz Center for Information Security), Michał Pilipczuk `[通讯]` (University of Warsaw)

**通讯引用:** 5546 | [OpenAlex ID](https://openalex.org/A5000479623)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出一种在H-无极小图中以随机多项式时间采样得到的诱导子图 G' 及其宽度为 O(k log k) 的树分解，使得任意大小为 k 的目标模式 Z 均被完整包含且每个包中至多包含 O(√k) 个 Z 节点，从而能够在 2^{O(√k)} n^{O(1)} 时间内求解多种离散结构与距离约束相关的子图匹配与覆盖问题。

**💡 创新点**

创新点包括：① 将已知仅适用于连通模式的模式覆盖技术推广到任意（甚至分散）模式；② 在 K_{3,h}-无极小图中进一步控制模式的 d 邻域，使得算法可处理涉及邻域的部分支配、距离约束等问题；③ 通过对分离器/近似无交路径的改进实现“稀疏”分离器的构造，显著提高了适用范围；④ 采用分层-分区与产品结构理论的组合，构造可行的平衡分离器与树分解。

**🔧 技术方法**

主要技术包括：分离器/几乎无交路径的流割对偶性；多重迭代的“稀疏化”分离器抽样；基于产品结构理论的平衡分离器与树分解构造；递归分解中的模式模式（pattern mode）与清理策略；对 K_{3,h}-无极小图中 d 邻域影响的界定；以及对路径与分离器的重构与压缩。

**📊 数据集**

本文为理论算法研究，未使用具体数据集；所有结果均为理论证明与多项式时间算法构造。

**📈 对比分析**

相比已有的 Fomin 等人（连通模式）与 Nederlof（平面图）技术，本文实现了更广泛的图类（H-无极小图）与更通用的模式（可分散、含邻域）支持；时间复杂度保持在 2^{O(√k)} n^{O(1)}，在此前的 2^{√k log^2 k} 等略高复杂度基础上实现了改进；同时支持 K_{3,h}-无极小图的距离扩展。

**⚠️ 局限性**

局限性包括：① 仍需图类满足 H-无极小条件，对某些非极小图或包含 K_6 的无极小图不适用；② 对于一般 H-无极小图的距离 d 版本未完全给出，且在该版本中仅对 K_{3,h} 取得结果；③ 结果为随机化算法，成功概率为 2^{-O(√k)} n^{-O(1)}，需多次重复；④ 对于某些特定模式（如高连通度或不满足邻域约束）可能不具备直接适用性。

---

## 461. Compiling Code LLMs into Lightweight Executables

**arXiv ID:** 2603.29813 | [PDF](https://arxiv.org/pdf/2603.29813v1)

**作者:** Jieke Shi `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 30792 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段方法，将大型代码生成语言模型（Code LLM）压缩成低比特宽度并通过LLVM编译器将其计算程序转换为高效的BLAS调用，从而在仅有CPU的个人设备上实现可执行的轻量级可部署二进制文件。

**💡 创新点**

创新点在于将量化与编译器级别的 GEMV 优化相结合：1）基于聚类的产品量化（product‑quantization）实现低比特索引存储并保证误差界限；2）设计专用LLVM IR Pass 自动识别并替换 GEMV 循环为高性能 BLAS 调用；3）两者协同产生显著的速度与内存提升。

**🔧 技术方法**

采用的技术包括：基于聚类的权重量化与比特打包（bit‑packing）；LLVM IR 结构匹配与循环识别；调用C/Fortran BLAS（Accelerate/OpenBLAS等）实现高效 GEMV；C/C++ 与 Python 辅助工具进行模型转换、生成与评测。

**📊 数据集**

实验使用的 Code LLM 模型包括 Code Llama‑7B、MagicCoder‑CL‑7B、OpenCodeInterpreter‑CL‑7B；基准任务为 HumanEval+ 与 MBPP+ 两套代码生成评测集。

**📈 对比分析**

与原始 FP32 推理和一个基准 Int8 量化实现相比，经过量化+编译后在 Apple M2 CPU 上可获得：内存压缩 6.4×、推理速度提升 10.5×、能耗降低 10.5×，同时 Pass@1 平均损失仅 0.27%，在大多数场景下比 Int8 取得 4–7% 的准确率提升。

**⚠️ 局限性**

局限性包括：仅支持基于 llama.c 的推理实现，无法直接迁移到基于 PyTorch/TensorFlow 或 GPU 加速的框架；量化仅为静态后训练方法，缺少针对不同硬件的动态校准；在极低比特（<3 位）或更大模型（>13B）时可能需要进一步的技术改进。

---

## 462. Reconfiguration of supernumerary robotic limbs for human augmentation

**arXiv ID:** 2603.29808 | [PDF](https://arxiv.org/pdf/2603.29808v1)

**作者:** Mustafa Mete `[一作]` (École Polytechnique Fédérale de Lausanne), Jamie Paik `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 5954 | [OpenAlex ID](https://openalex.org/A5025220023)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套基于定量人类增强分析的可重构多余机器人肢体（SRL）框架，并实现了一种采用折纸启发式模块化的Robogami第三臂，结合可重构配置与自适应自主级别选择，在实验中验证了其在不同位置与模块数量下的工作空间扩展与协作能力；

**💡 创新点**

创新点在于首次将工作空间分为协作、可见扩展与不可见扩展三类，并用人类增强比率来量化SRL对可视反馈与协作需求的影响，从而实现了基于任务需求的SRL配置与自主级别选择；

**🔧 技术方法**

技术手段包括三维运动学建模与点云体积计算、可视化反馈场景建模、模块化折纸关节驱动、BLE无线控制、基于二次规划的逆运动学与控制策略；

**📊 数据集**

主要使用NASA平均人体测量数据和10,000个关节采样点生成的工作空间点云；

**📈 对比分析**

通过比较不同装配位置与模块数下的扩展与协作比率，以及在杯子稳定实验中实现平均误差低于5°，验证了所提出方法在不同任务场景下的有效性；

**⚠️ 局限性**

局限性包括仅考虑视觉反馈，未纳入触觉或听觉等多模态信息；实验主要集中在到达性任务，未评估载荷极限或动态环境下的实时适配；

---

## 463. From Density Matrices to Phase Transitions in Deep Learning: Spectral Early Warnings and Interpretability

**arXiv ID:** 2603.29805 | [PDF](https://arxiv.org/pdf/2603.29805v1)

**作者:** Max Hennick `[一作]`, Guillaume Corlouer `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 2‑数据点简化密度矩阵（2RDM）作为在训练过程中检测相变的低成本、可解释观测量，定义光谱热容量（SHC）与占据比（PR）等谱指标，并在深线性网络、诱导头形成、grokking 与 emergent misalignment 四个典型场景中进行验证。

**💡 创新点**

创新点：①将量子化学中的 2‑电子 RDM 概念迁移到深度学习，构造以损失协方差为核心的 2RDM；②证明 SHC 可作为第二阶相变的早期预警；③提出 PR 用于量化相变的有效维度；④顶特征向量可直接解释相变的参与样本与模式。

**🔧 技术方法**

技术手段：前向传播计算 probe set 上的单样本损失；滑动窗口动态 2RDM 估计；谱统计（SHC=Var(λ)，PR=（tr C）²/tr(C²)）；顶特征向量投影与解释（频域、注意力块等）；实验平台包括深线性网络、Transformer、Qwen2.5-7B‑Instruct 适配器。

**📊 数据集**

数据集：深线性网络使用随机正交/高斯权重；诱导头实验使用随机 token 序列；grokking 采用模数除法数据（Zₚ×Zₚ）；emergent misalignment 采用 Qwen2.5‑7B‑Instruct 微调，probe set 包含 benign capability、boundary ambiguous、medical advice、alignment probes 等多类样本。

**📈 对比分析**

对比方法：LLC、梯度范数等；结果显示 SHC 在所有四个实验中提前或同步捕捉到相变，PR 与 SHC 互补；与传统方法相比，2RDM 计算成本显著降低，同时提供了可解释的顶特征向量信息。

**⚠️ 局限性**

局限性：①只能检测 probe set 能“分辨”的相变，probe 设计对检测效果至关重要；②对快速训练阶段或高噪声环境下的轨迹采样窗口有限；③对极大模型的动态采样可能受限；④在非线性或高维度的复杂相变中，线性化假设可能失效。

---

## 464. ENEIDE: A High Quality Silver Standard Dataset for Named Entity Recognition and Linking in Historical Italian

**arXiv ID:** 2603.29801 | [PDF](https://arxiv.org/pdf/2603.29801v1)

**作者:** Cristian Santini `[一作]`, Mehwish Alam `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建并发布了ENEIDE数据集，收集并半自动化提取了来自两个学术数字编辑的历史意大利文本的命名实体标注，并提供了训练、验证和测试拆分。

**💡 创新点**

创新点在于：①首次推出多域公开的历史意大利NERL数据集，包含训练/验证/测试；②提出从SDE半自动化抽取标注的完整方法并进行质量控制；③数据跨越两百年并包含多种实体类型及NIL实体。

**🔧 技术方法**

技术上采用BeautifulSoup和RDFa解析HTML、频繁实体提取+专家校验、自动补全+Stanza NER模型；实验使用GLiNER微调、LLM零样本提示、BLINK-ita/mGENRE/BELA实体链接模型。

**📊 数据集**

使用的数据集为ENEIDE本身（2111篇文档、8000+标注），来源为Digital Zibaldone与Aldo Moro Digitale；实验基于公开预训练模型。

**📈 对比分析**

通过与LLM零样本、GLiNER微调以及BLINK-ita/mGENRE/BELA等模型对比，发现零样本模型精度低，GLiNER微调在DZ/AMD分别取得F1 0.782/0.876；EL模型在DZ/BELA 0.598，AMD/mGENRE 0.689，表明历史文本仍具挑战。

**⚠️ 局限性**

局限性包括：仅覆盖两位作者两种领域，可能导致领域偏倚；半自动增强过程可能引入标注差异；多语种覆盖有限，缺乏更广泛的历史意大利文本样本。

---

## 465. SurgTEMP: Temporal-Aware Surgical Video Question Answering with Text-guided Visual Memory for Laparoscopic Cholecystectomy

**arXiv ID:** 2603.29962 | [PDF](https://arxiv.org/pdf/2603.29962v1)

**作者:** Shi Li `[一作]` (University of Strasbourg), Nicolas Padoy `[通讯]` (University of Strasbourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SurgTEMP框架，结合文本引导的记忆金字塔和Surgical Competency Progression训练方案，实现对腹腔镜胆囊切除视频的多层次问答和安全评估

**💡 创新点**

创新点在于：①文本引导的跨模态注意力实现自适应帧选择与双层空间-时间记忆结构；②分层训练策略SCP逐步提升从感知到评估再到推理的能力；③构建了覆盖感知、评估、推理的32K问答数据集CholeVidQA-32K

**🔧 技术方法**

采用SigLIP视觉编码器、Qwen2-7B LLM、Gumbel-Softmax可微分帧选取、可学习分隔符、LoRA参数高效微调

**📊 数据集**

使用CholeVidQA-32K（来自CholecT50、Endoscapes、CholeScore），涵盖3级任务共11项；同时评估黄金测试集由临床专家标注的答案

**📈 对比分析**

与多种开源视频LLM（mPLUG-Owl3、InternVideo2.5、LongVA、LLaVA-Video、VideoGPT+）以及其微调版本对比，SurgTEMP在GPT基准、文本重叠和分类指标上均显著领先，尤其在评估层和长时序任务上提升幅度最大

**⚠️ 局限性**

局限性包括仅针对胆囊切除单一手术类型，缺乏跨手术适用性；模型未显式处理不确定性与拒答机制，可能在实际临床部署时产生过度自信或错误答案

---

## 466. NeuroBRIDGE: Behavior-Conditioned Koopman Dynamics with Riemannian Alignment for Early Substance Use Initiation Prediction from Longitudinal Functional Connectome

**arXiv ID:** 2603.29960 | [PDF](https://arxiv.org/pdf/2603.29960v1)

**作者:** Badhan Mazumder `[一作]` (Georgia State University), Dong Hye Ye `[通讯]` (Georgia State University)

**通讯引用:** 7690 | [OpenAlex ID](https://openalex.org/A5068927047)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一种名为NeuroBRIDGE的图神经网络框架，用于对纵向功能连接矩阵进行几何对齐并预测4年后青少年的物质使用启动风险。

**💡 创新点**

其创新点包括在SPD流形的切空间中锚定并对齐多时点连接矩阵、使用多尺度热核词元与基于RBF的边缘门控来捕捉局部到全局结构、通过双时域自/交注意力融合多时点信息，并在行为条件下引入神经Koopman算子建模潜在演化。

**🔧 技术方法**

技术上结合了图神经网络、Riemannian流形几何、热核扩散词元、门控图卷积、双时域注意力、行为条件化的神经Koopman动态以及原型对比损失等。

**📊 数据集**

使用了美国青少年大脑认知发展（ABCD）研究的多站点纵向数据，共计7168名参与者的基线（9–10岁）和2年（11–12岁）静息态fMRI与CBCL评分，并以4年（13–14岁）物质使用启动标签进行评估。

**📈 对比分析**

与SPDNet、BrainNetCNN、BNT、EvolveGCN和RBGM等现有基线模型相比，NeuroBRIDGE在准确率、敏感性和特异性上分别提升了约7–10个百分点、13–19个百分点和6–9个百分点，最终达到84.71%、86.36%和84.66%的表现。

**⚠️ 局限性**

局限性包括仅使用两次扫描和CBCL评分，缺少多模态数据（如扩散MRI、结构MRI、任务fMRI），以及模型对ABCD之外数据的泛化性尚未验证，需要进一步扩展多访点轨迹和更多模态。

---

## 467. Think Anywhere in Code Generation

**arXiv ID:** 2603.29957 | [PDF](https://arxiv.org/pdf/2603.29957v1)

**作者:** Xue Jiang `[一作]` (Peking University), Yihong Dong `[通讯]` (Peking University)

**通讯引用:** 957 | [OpenAlex ID](https://openalex.org/A5077542599)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为Think-Anywhere的思考机制，使LLM在代码生成过程中能够在任意位置即时调用思考，替代传统的先期思考模式。

**💡 创新点**

创新点在于通过冷启动监督学习让模型学会在代码片段中插入思考块，并通过RLVR（组相对策略优化）进一步让模型自适应决定何时何处触发思考，形成自下而上的动态思考策略。

**🔧 技术方法**

采用了冷启动训练+LoRA微调、专用触发符号的语义化初始化、组相对策略优化（GRPO）以及分层奖励函数（结构奖励+执行正确性奖励）的技术组合。

**📊 数据集**

使用了约14K的Skywork代码问题集进行训练，并在LeetCode、LiveCodeBench、HumanEval、MBPP四大基准上进行评估。

**📈 对比分析**

与基线模型、GRPO、CoT、Self‑Planning等进行pass@1比较，Think-Anywhere在四个基准上的平均得分为70.3%，比基线提升9.3%，并在数学推理基准上亦显著提高，展示了显著的性能优势。

**⚠️ 局限性**

局限性包括对后训练数据量的依赖，特殊符号的语义学习尚未完全成熟，且在极高复杂度位置的思考选择仍可能不最优；对更大规模模型的适应性尚未全面验证。

---

## 468. Implementing Basic Arithmetic in $\mathbb{F}_p$ via $\mathbb{F}_2$, and Its Application for Computing the Hamming Distance of Linear Codes

**arXiv ID:** 2603.29942 | [PDF](https://arxiv.org/pdf/2603.29942v1)

**作者:** Fernando Hernando `[一作]` (Universitat Jaume I), Gregorio Quintana-Ortí `[通讯]` (Universitat Jaume I)

**通讯引用:** 1886 | [OpenAlex ID](https://openalex.org/A5086284706)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种通用方法，利用 𝔽₂ 的二进制操作实现任意素数域 𝔽_p（p>2）上的基本算术，并基于此实现了快速计算随机线性码在 𝔽₃ 和 𝔽₇ 上的最小海明距离。

**💡 创新点**

创新点在于：①将 𝔽_p 的元素用自然二进制编码和切片位存储（sliced‑bit），实现了对 𝔽₂ 的高效利用；②针对 𝔽₃、𝔽₇ 的特殊结构，设计了专门的加法与乘法算法；③引入了保留海明重量的等距（isometric）运算，显著降低了在计算海明距离时的算术成本；④在并行共享内存环境下实现了可扩展的多核版本。

**🔧 技术方法**

使用了 C 语言实现，主要技术包括：切片位存储、位运算加法/乘法、循环/位移实现的模运算、等距加法、向量化（AVX）和编译器原生优化 flag。

**📊 数据集**

实验数据集包括随机生成的线性码矩阵，分别在 𝔽₃ 与 𝔽₇ 上测试；在 𝔽₇ 上使用 540 条中大规模码（n≤58, k≤25），在 𝔽₃ 上使用 933 条（n≤74, k≤50）以及对比集（60 条 𝔽₇、41 条 𝔽₃）。

**📈 对比分析**

与 Magma、GAP/Guava（开源）及商用软件做比较；在单核、双核、16 核等不同处理器上，新的实现平均比参照实现快 2.96‑5.06 倍（𝔽₇）和 1.4‑3.8 倍（𝔽₃），在最重负载的实例中可提升 5‑10 倍，且并行扩展良好。

**⚠️ 局限性**

局限性：仅针对小素数域（3、7）实现，较大素数域的切片位编码和算术成本不一定能保持优势；实现依赖于对 𝔽₂ 的硬件支持，非 SIMD 体系结构时收益有限；此外等距加法仅在海明重量求解场景有效，对需要精确算术的应用不适用。

---

## 469. End-to-End Image Compression with Segmentation Guided Dual Coding for Wind Turbines

**arXiv ID:** 2603.29927 | [PDF](https://arxiv.org/pdf/2603.29927v1)

**作者:** Raül Pérez-Gonzalo `[一作]` (Wind Power LAB), Antonio Agudo `[通讯]` (Institut de Robòtica i Informàtica Industrial)

**通讯引用:** 4529 | [OpenAlex ID](https://openalex.org/A5041283253)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种端到端的 ROI 图像压缩框架，结合了风电机翼图像的分割与双模压缩（有损与无损），以实现高速、高质量的数据传输；

**💡 创新点**

创新点在于首次将学习驱动的 ROI 编码、BU‑Netv2+P 细粒度分割、HP+EASN‑deep 有损编码、Bit‑swap 无损编码以及通过重用背景比特实现并行化的双模压缩统一到一套系统；

**🔧 技术方法**

使用的技术包括改进的 BU‑Netv2+P 分割网络（CRF 正则化 + 随机森林后处理）、基于超先验的 VAE（HP+EASN‑deep）进行有损压缩、层级 VAE+Bits‑back（Bit‑swap）实现无损压缩，以及 ANN‑based 变换、ANS 编码与并行 Bits‑back；

**📊 数据集**

实验数据集为 64,438 张高分辨率风电机翼图片（训练 80%，验证 10%，测试 10%），并在此数据上训练与评估所有模型；

**📈 对比分析**

与传统编码器（JPEG2000、VTM、JXL 等）和最新学习压缩器（HP+EASN‑deep、JA+EASN‑deep 等）比较，HP+EASN‑deep 在 PSNR 维度实现最低 BD‑rate、较快编码时间；Bit‑swap 在无损场景下实现 8.98 bits/px，优于 PNG、BPG 等；ROI‑eML 在保持压缩性能的同时，通过并行化显著降低了无损模式的处理时延；

**⚠️ 局限性**

主要局限在于无损模式仍需较长的编码/解码时间，尤其在缺乏足够背景比特时并行度受限；系统对分割质量高度依赖，极端姿态或光照条件下分割误差可能影响后续压缩质量；

---

## 470. Abstraction in Style

**arXiv ID:** 2603.29924 | [PDF](https://arxiv.org/pdf/2603.29924v1)

**作者:** Min Lu `[一作]` (Shenzhen University), Hui Huang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了两阶段的风格迁移框架AiS，先通过结构抽象生成抽象代理，再对代理进行视觉风格化；

**💡 创新点**

创新点在于将结构抽象与视觉风格化显式分离，并利用视觉类比迁移（VAT）在小样本条件下学习抽象逻辑；

**🔧 技术方法**

核心技术包括隐藏骨干结构（骨架+区域侵蚀）、视觉类比迁移（基于Diffusion Transformer + LoRA）的两阶段实现；

**📊 数据集**

使用来自Pinterest的10-20张illustrative风格样本作为参考风格，目标图像则由FLUX模型生成；

**📈 对比分析**

与StyleID、StyleAlign、Attention Distillation、LoRA等方法对比，CSD（风格相似度）最高0.72，LPIPS（感知相似度）最低0.47，用户研究中50%偏好AiS；

**⚠️ 局限性**

局限性在于仅能处理相对简单的结构抽象，无法实现强语义扭曲、夸张或比例失调，隐藏结构构建过于简化，缺乏对更复杂抽象行为的建模。

---

## 471. Performance Evaluation of LLMs in Automated RDF Knowledge Graph Generation

**arXiv ID:** 2603.29878 | [PDF](https://arxiv.org/pdf/2603.29878v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 472. C-TRAIL: A Commonsense World Framework for Trajectory Planning in Autonomous Driving

**arXiv ID:** 2603.29908 | [PDF](https://arxiv.org/pdf/2603.29908v1)

**作者:** Zhihong Cui `[一作]` (University of Oslo), Tor Skeie `[通讯]` (University of Oslo)

**通讯引用:** 4620 | [OpenAlex ID](https://openalex.org/A5013855617)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了C-TRAIL框架，利用LLM生成的常识性知识进行自主驾驶轨迹规划，并通过信任机制动态校准常识可靠性，形成Recall-Plan-Update闭环。

**💡 创新点**

创新点在于：①双重信任机制量化LLM输出可靠性；②将信任加权的Dirichlet策略注入MCTS，引导搜索；③在线自适应更新信任和策略参数，提升鲁棒性与泛化。

**🔧 技术方法**

使用的大型语言模型（GPT-3.5‑turbo / GPT‑4o）、Transformer编码、Monte Carlo Tree Search（MCTS）与Dirichlet先验、对抗式信任校准与EMA更新。

**📊 数据集**

实验数据集包括：仿真平台Highway‑env（高速、合并、环岛、交叉口四种场景）和真实世界的高德高D、rounD两套无人机拍摄道路轨迹数据。

**📈 对比分析**

与七个基线（DQN、MCTS、GRAD、LMTraj、DiLu、LangMPC、GPT‑Driver）对比，C-TRAIL在所有场景下均显著优于基线：ADE下降约40.2%，FDE下降约51.7%，成功率提升约16.9个百分点；在未见环境下保持低于1.7个百分点的性能衰减。

**⚠️ 局限性**

局限性：依赖LLM API延迟较高，推理时间约18秒；缺乏正式的安全性形式化验证；目前仅使用简单的运动学模型，未覆盖多模态感知与高保真仿真环境。

---

## 473. Less Is More? Selective Visual Attention to High-Importance Regions for Multimodal Radiology Summarization

**arXiv ID:** 2603.29901 | [PDF](https://arxiv.org/pdf/2603.29901v1)

**作者:** Mst. Fahmida Sultana Naznin `[一作]` (Bangladesh University of Engineering and Technology), Md Rakibul Hasan `[通讯]` (Curtin University)

**通讯引用:** 608 | [OpenAlex ID](https://openalex.org/A5025876524)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种多阶段框架 ViTAS，用于在胸片报告的 FINDINGS → IMPRESSION 生成中仅选取最具病理信息的视觉补丁，并结合文本生成简洁、准确的印象。

**💡 创新点**

创新点在于：① 用 MedSAM2 集成肺分割和双向 Swin Transformer V2 进行多视角融合；② 利用 Shapley 值评估视角重要性并驱动补丁聚类；③ 采用层级视觉分词将重要补丁与文本信息融合进 ViT+T5 生成模型，从而证明“少即是多”在视觉输入上的有效性。

**🔧 技术方法**

核心技术包括 MedSAM2 边界框集成肺分割、Swin Transformer V2 的双向空间交叉注意力、Shapley 值视角加权、DBSCAN 聚类、ViT 视觉编码器、T5 文本解码器以及多模态融合。

**📊 数据集**

实验使用公开胸片报告基准 MIMIC‑CXR（377k张图像、227k份报告）进行训练、验证和测试。

**📈 对比分析**

在多项自动评估指标（BLEU‑4、ROUGE‑L、BERTScore、CheXbert、RadGraph）上，ViTAS 取得 29.25% BLEU‑4、69.83% ROUGE‑L、95.61% BERTScore，明显优于全图像多视角模型和强文本基线；在人类专家评估中，阅读性、事实准确性和信息量得分也最高。

**⚠️ 局限性**

局限性包括：① 依赖于准确的肺分割和 Swin Transformer V2 的注意力热图，错误会传播到补丁选择；② 仅在单一机构的胸片数据上验证，尚未测试对多器官或跨域数据的泛化能力。

---

## 474. AI Empathy Erodes Cognitive Autonomy in Younger Users

**arXiv ID:** 2603.29886 | [PDF](https://arxiv.org/pdf/2603.29886v1)

**作者:** Junfeng Jiao `[一作]` (University of Texas at Austin), Saleh Afroogh `[通讯]` (University of Texas at Austin)

**通讯引用:** 410 | [OpenAlex ID](https://openalex.org/A5006040238)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Stoic Architectures，通过情感对齐惩罚、RLAIF发展宪法和动态情绪门控，减少生成式AI在青少年中的情感共鸣，提升其认知自主性。

**💡 创新点**

将发展心理学的“可欲困难”原则与AI对齐结合，首次量化情感对齐惩罚并构建以发展为核心的宪法与动态门控，形成以认知自主为导向的新对齐框架。

**🔧 技术方法**

采用RLHF、RLAIF、基于RoBERTa的情绪编码器与相似度惩罚、动态门控分类器以及强化学习训练策略。

**📊 数据集**

使用Anthropic HHH数据集、内部用户交互日志、合成对话对以及儿童心理学专家标注的情绪与认知评价数据集。

**📈 对比分析**

相较于传统RLHF模型，Stoic模型在Affective Orthogonality、Objectivity/Agency评价上提升约30%+，且在ERQ-CA等认知重塑指标上显著更优；但即时用户满意度与停留时间略有下降。

**⚠️ 局限性**

局限性包括潜在的用户抗拒与情绪体验下降、危机检测准确性不足、文化适配性差以及在低压情境下共情不足导致的交互体验下降。

---

## 475. UnWeaving the knots of GraphRAG -- turns out VectorRAG is almost enough

**arXiv ID:** 2603.29875 | [PDF](https://arxiv.org/pdf/2603.29875v1)

**作者:** Ryszard Tuora `[一作]`, Tomasz Ziętkiewicz `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出UnWeaver，一种通过实体中心化检索来提升检索增强生成（RAG）精度的框架；

**💡 创新点**

创新点在于利用LLM提取并聚合跨块实体描述，形成轻量级实体向量索引，既保留了图谱式检索的精确性，又避免了构建完整知识图谱的高成本；

**🔧 技术方法**

核心技术包括基于LLM的实体抽取与描述聚合、实体向量化、实体-块关联矩阵、以及基于多选投票的多赢选举检索；

**📊 数据集**

使用COVID‑QA、eManual、Tech‑QA三大知识问答数据集进行实验；

**📈 对比分析**

与VectorRAG、GraphRAG、RAPTOR等基线对比，UnWeaver在eManual和COVID‑QA上取得最佳F1得分，在Tech‑QA上排名第二，且查询时LLM token消耗最低，展示出更高的事实正确性与更低的查询延迟；

**⚠️ 局限性**

局限性包括：对实体抽取的质量高度依赖LLM；实体合并规则过于简单，可能忽略语义差异；并且在多跳推理场景下尚未系统评估其效果。

---

## 476. Spatiotemporal Robustness of Temporal Logic Tasks using Multi-Objective Reasoning

**arXiv ID:** 2603.29868 | [PDF](https://arxiv.org/pdf/2603.29868v1)

**作者:** Oliver Schön `[一作]` (ETH Zürich), Lars Lindemann `[通讯]` (ETH Zürich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了信号在空间和时间扰动下满足时序逻辑规范的鲁棒性，并提出多目标推理框架以计算时空鲁棒性集合。

**💡 创新点**

首次将多目标优化与鲁棒语义结合，得到 Pareto 前沿的时空鲁棒性评估。

**🔧 技术方法**

使用信号时序逻辑（STL）、多目标优化（ε-constraint）、下确界约束与签名距离函数实现鲁棒语义计算。

**📊 数据集**

在 F-16 战机飞行轨迹和 Waymo 开放数据集的车辆/行人交互轨迹上进行案例验证。

**📈 对比分析**

相较于传统单一空间/时间鲁棒度量，所提方法能够给出更细粒度的鲁棒性集合，计算时间在几秒到几十秒之间，表现出可接受的实时性能。

**⚠️ 局限性**

计算复杂度仍受多目标优化的非凸性限制，且对复杂谓词的求解可能需要近似或专用算法。

---

## 477. Rewrite the News: Tracing Editorial Reuse Across News Agencies

**arXiv ID:** 2603.29937 | [PDF](https://arxiv.org/pdf/2603.29937v1)

**作者:** Soveatin Kuntur `[一作]`, Sebastijan Razboršek Maček `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用弱监督框架，探究了斯洛文尼亚新闻社（STA）英文新闻与15家多语种国际新闻机构在 2023‑2025 年期间句子层面的文本重用情况。

**💡 创新点**

创新点在于将多语言句子嵌入、时间戳过滤和语篇位置分析相结合，能够识别跨语言编辑重用模式，并揭示重用在文章中的位置分布，而非仅靠表面词汇相似度。

**🔧 技术方法**

主要技术包括多语言 SBERT 句子向量、余弦相似度阈值（60%）、基于发布时间的误判过滤、句子级别对齐及句子位置关系（PR）类型分析。

**📊 数据集**

使用数据集为：1,037 篇 STA 英文新闻（2023 与 2025 两个时间窗口）以及 237,551 篇来自 15 家外语机构（意大利语、英语、波兰语、法语、德语、塞尔维亚语、克罗地亚语）的多语种新闻。

**📈 对比分析**

方法在 Webis‑Wikipedia‑Text‑Reuse‑18 与 Webis‑CPC‑11 数据集上进行评估，阈值60%时平均相似度>0.6；检测结果显示 STA 与外语机构共用约52%的句子内容，外语机构单篇约1.6%；多重重用结构（many:many 与 many:1）占比高达95%，一对一匹配仅 4.4%。

**⚠️ 局限性**

限制主要在于未对所有文本进行完整翻译，导致无法精准区分同义改写与非改写；高相似度阈值难以判断信息传递立场；过滤短句/无谓词句可能漏检关键句子；小样本评测显示方法在复杂语境下仍需改进。

---

## 478. Rethinking AI Literacy Education in Higher Education: Bridging Risk Perception and Responsible Adoption

**arXiv ID:** 2603.29935 | [PDF](https://arxiv.org/pdf/2603.29935v1)

**作者:** Shasha Yu `[一作]` (Clark University), Barry L. Bentley `[通讯]` (Cardiff Metropolitan University)

**通讯引用:** 1094 | [OpenAlex ID](https://openalex.org/A5055245830)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过问卷调查研究了技术学生对AI风险的显式与情境化认知以及对AI技术的采纳意愿。

**💡 创新点**

创新点在于首次将显式风险意识与情境化风险识别结合，揭示技术熟悉度导致的风险低估和采纳倾向差异。

**🔧 技术方法**

采用定量统计方法（描述性统计、t检验、ANOVA、相关分析）进行数据分析。

**📊 数据集**

使用来自美国Clark大学的139名计算机与数据科学及其他专业学生的调查数据。

**📈 对比分析**

通过比较显式与情境化风险分数及风险与采纳意愿的相关性，发现风险感知越高采纳意愿越低，技术专业学生的采纳意愿更高。

**⚠️ 局限性**

研究的局限包括单一机构样本、跨学科样本不平衡、受访者自报偏差以及横断面设计限制因果推断与普适性。

---

## 479. Physiological and Semantic Patterns in Medical Teams Using an Intelligent Tutoring System

**arXiv ID:** 2603.29950 | [PDF](https://arxiv.org/pdf/2603.29950v1)

**作者:** Xiaoshan Huang `[一作]` (McGill University), Susanne P. Lajoie `[通讯]` (McGill University)

**通讯引用:** 6624 | [OpenAlex ID](https://openalex.org/A5079114117)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究医学团队在使用智能辅导系统（BioWorld）诊断虚拟病人时的生理同步（心率）与对话语义动态，探究其与社会共享学习调节（SSRL）行为的关联，并揭示高同步峰值与团队关键转折点的关系。

**💡 创新点**

创新点在于：①首次将生理同步峰值与对话句子嵌入相结合，形成“团队生命体征”概念；②发现高同步峰与语言多样性和关键决策时刻相关；③提出将生理峰值作为实时标记，用以识别团队的探索性或困境时刻。

**🔧 技术方法**

技术手段包括：Empatica E4 设备获取血量脉搏→心率；句子级嵌入（预训练语言模型）→余弦相似度；阈值筛选最大同步峰值；置换检验 + Benjamini‑Hochberg 校正；混合效应模型；质性对话与访谈编码。

**📊 数据集**

数据集：BioWorld ITS 收集的 4 对（共 8 名）医学团队对话音频及访谈文本，配合 Empatica E4 采集的 1 Hz 心率数据。

**📈 对比分析**

比较方法：将对话按句子切分，计算每段最大同步峰值；选取前 10–30% 峰值段与其余段，比较两组语义嵌入中心的余弦相似度；通过置换检验评估显著性。结果表明 11/66 峰值段与其余段在语义相似度上显著不同，且高同步峰往往对应较低语义相似度。

**⚠️ 局限性**

局限性：样本量仅 4 对双人团队，难以推广；仅使用心率同步，忽略其他生理指标；对话嵌入受预训练模型偏差影响；因果关系未能实时验证；对其他学科或更大规模团队的适用性仍需进一步验证。

---

## 480. Scaling Video Pretraining for Surgical Foundation Models

**arXiv ID:** 2603.29966 | [PDF](https://arxiv.org/pdf/2603.29966v1)

**作者:** Sicheng Lu `[一作]` (Johns Hopkins University), Zuozhu Liu `[通讯]` (Zhejiang University-University of Illinois Urbana-Champaign Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SurgRec框架，构建10,535个多源手术视频语料库，训练基于MAE和JEPA的自监督模型，并在16个下游数据集上进行统一评测。

**💡 创新点**

创新点包括：① 大规模、多域手术视频数据集与统一采样策略；② 通过平衡采样显著提升跨域泛化；③ 设立可复现的多程序基准并对比VLM的弱点。

**🔧 技术方法**

采用自监督技术MAE（Masked Autoencoder）和JEPA（Predictive Latent Modeling），利用DINOv3特征做聚类采样，混合批次训练，评估时使用Fine‑tune和Zero‑shot VLM推理。

**📊 数据集**

使用10,535个视频（214.5M帧）来自32个公开数据集（腹腔镜、内镜、白内障、机器人手术等）以及网络爬取视频；下游评测涉及16个手术域数据集。

**📈 对比分析**

与通用域SSL基线（VideoMAE、JEPA、DINOv3）以及VLM基线（Qwen3‑VL、LLaVA‑NEXT、Qwen2.5‑VL）进行对比；SurgRec-MAE/JEPA在宏观指标上分别提升约+3.18/ +2.61点，整体显著优于VLM；平衡采样显著提升Robotic和Endoscopy等稀缺域性能。

**⚠️ 局限性**

局限性包括：网络视频与真实临床视频存在域差距；部分评测子集规模小或噪声高；缺乏多中心真实临床环境的验证。

---

## 481. EC-Bench: Enumeration and Counting Benchmark for Ultra-Long Videos

**arXiv ID:** 2603.29943 | [PDF](https://arxiv.org/pdf/2603.29943v1)

**作者:** Fumihiko Tsuchiya `[一作]` (University of Tokyo), Yutaka Matsuo `[通讯]` (University of Tokyo)

**通讯引用:** 14077 | [OpenAlex ID](https://openalex.org/A5090592819)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了EC-Bench基准，针对超长视频的枚举、计数和时序证据定位三项任务进行统一评估。

**💡 创新点**

创新点在于：①将枚举、计数与时序定位三项能力融合评测；②为每个问题提供显式的时间段证据；③设计了覆盖六类推理模式的平衡问答集。

**🔧 技术方法**

技术手段包括：多模态大语言模型（MLLM）提示与推理、枚举先行提示策略、基于F1、MAE与tIoU的评估指标，以及利用LLM-as-Judge进行自动判分。

**📊 数据集**

使用的数据集为EC-Bench，包含152段超长（>30 min）视频和1699个开放式查询，视频来源涵盖体育、纪录片、新闻等多领域。

**📈 对比分析**

通过对22款开源和专有MLLM的对比实验，最优模型（如GPT‑5）枚举准确率仅29.98%、计数准确率23.74%，远低于人工78.57%/82.97%；枚举先行提示能显著提升计数表现。

**⚠️ 局限性**

局限性在于：时序证据定位仍是瓶颈，模型易出现欠计、音频误计、抽象化列举和知识幻觉；单纯增帧量并不能显著提升性能，亟需更有效的长时序推理与实例识别机制。

---

## 482. UniRank: End-to-End Domain-Specific Reranking of Hybrid Text-Image Candidates

**arXiv ID:** 2603.29897 | [PDF](https://arxiv.org/pdf/2603.29897v1)

**作者:** Yupei Yang `[一作]` (Shanghai Jiao Tong University), Lei Xu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 11978 | [OpenAlex ID](https://openalex.org/A5102372324)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了UniRank，一种基于视觉‑语言模型的端到端域适配重排框架，能够原生处理混合文本与图像候选项；

**💡 创新点**

创新点在于：①直接对混合模态候选进行统一打分，避免文本→图像转换导致的效率与信息损失；②结合指令微调、硬负样本挖掘和基于查询级别的RLHF实现对域特定偏好的一致性对齐；

**🔧 技术方法**

主要技术包括指令驱动的监督微调、基于标签概率的评分映射、硬负样本挖掘生成偏好数据、奖励模型学习以及使用查询级组的GRPO进行强化学习；

**📊 数据集**

实验使用了科学文献检索的MMDocIR学术论文数据集和设计专利检索数据集，均包含文本与图像混合候选；

**📈 对比分析**

与文本重排器、MM-R5、jina‑reranker等现有方法对比，UniRank在Recall@1、NDCG@3/5和MRR上分别提升约8.9%/7.3%（文献检索）和10%/5%（专利检索），同时存储与推理成本更低；

**⚠️ 局限性**

局限性包括对仅有文本+图像的领域适用，依赖有标签的训练数据，对更大规模多模态候选或多语种、视频等其他模态的支持仍待扩展。

---

## 483. "There is literally zero funding": Understanding the Emerging Role of Trusted Flaggers under the EU Digital Services Act

**arXiv ID:** 2603.29874 | [PDF](https://arxiv.org/pdf/2603.29874v1)

**作者:** Marie-Therese Sekwenz `[一作]` (TU Delft), Simon Parkin `[通讯]` (TU Delft)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对七家可信标识者（Trusted Flaggers, TFs）的半结构化访谈、工作坊和网站内容分析，系统收集并梳理了TF在欧盟数字服务法（DSA）下的认定、运营与平台交互经验。

**💡 创新点**

首次从经验主义视角揭示了TF在DSA框架内的关键痛点与机遇，提出六大主题（平台摩擦、能力约束、协调挑战、平台激励、操作歧义、问责 artefacts），并基于此给出标准化流程、协作平台及资源支持等政策性建议。

**🔧 技术方法**

运用定性研究技术：反射性主题分析（RTA）、访谈记录编码、网站信息编码、工作坊共创讨论，结合公开数据（TF官方名单、Statement of Reasons 数据库）进行交叉验证。

**📊 数据集**

主要数据集为：欧洲监管机构公开的可信标识者目录、各TF机构网站内容、DSA公开的Statement of Reasons（SOR）数据库，以及访谈文本与工作坊记录；没有使用大型机器学习或大规模量化数据。

**📈 对比分析**

本研究未采用传统的性能指标或实验对比；评价方式是通过主题分析得到六大主题，并在工作坊中得到受访者的验证；相较于已有文献，提供了第一手的经验性洞见，但缺乏量化评估。

**⚠️ 局限性**

局限性：样本仅覆盖7家TF（占全体约15%），且多为欧洲机构；研究阶段为DSA实施初期，可能存在经验不成熟；未采访平台或监管者；数据主要来自自我报告，缺乏外部验证；缺乏纵向追踪，无法评估长期影响。

---

## 484. A Rational Account of Categorization Based on Information Theory

**arXiv ID:** 2603.29895 | [PDF](https://arxiv.org/pdf/2603.29895v1)

**作者:** Christophe J. MacLellan `[一作]` (Georgia Institute Of Technology), Pat Langley `[通讯]` (Georgia Institute Of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

提出一种基于信息论的理性分析分类理论，并在 Cobweb 系统中实现该理论，通过最大化互信息的类别结构来解释人类分类行为。

**💡 创新点**

创新点在于将类别效用假设转化为最大化互信息的目标，并假设类别呈层次化（税onomic），同时在 Cobweb 中引入基于 PMI 的最佳先验搜索与更新策略，提供了新的模型与现有理论的比较。

**🔧 技术方法**

技术包括信息论的点互信息（PMI）计算、Cobweb 层次聚类、最大化期望互信息的更新规则、概率分布更新以及最佳先验搜索算法。

**📊 数据集**

使用了多项经典实验的数据集：<cit.> 的“中心趋势”实验、两项线性/非线性可分区分实验、以及 <cit.> 的原型-例子过渡实验，均为人工合成的多特征分类任务。

**📈 对比分析**

通过在模拟实验中对模型预测与人类实验结果进行 Spearman 相关系数和分类准确率比较。结果表明，新模型在中心趋势、线性/非线性实验中与上下文模型、独立线索模型、RMC 以及 HDP 的拟合相当或更好，能够准确再现人类的主要效应。

**⚠️ 局限性**

局限性包括：仅使用有限的超参数（α、max_nodes）；对 α 的取值敏感，需人工调参；未考虑随机猜测等噪声因素；只验证了少量经典实验，未检验更广泛效应；以及模型在更大规模、真实世界数据上的可扩展性和稳健性未知。

---

## 485. Mathematical Foundations of Modeling ETL Process Chains

**arXiv ID:** 2603.29877 | [PDF](https://arxiv.org/pdf/2603.29877v1)

**作者:** Levin Maier `[一作]` (Deepshore), Jan Peters `[通讯]` (TU Darmstadt)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个可用于模拟和优化ETL流程链的数学模型，包括基于流量平衡的吞吐量模型、可上限单调函数以及马尔科夫过程的离散事件模拟框架。

**💡 创新点**

创新点：① Flow Balance 关系将线程数、平均吞吐量和平均处理时间关联；② 引入可上限单调吞吐曲线族；③ 将ETL链建模为受控马尔科夫过程，便于学习与控制；④ 为离散事件模拟提供了高效实现。

**🔧 技术方法**

使用了离散事件模拟、马尔科夫过程理论、流量平衡法、概率分布拟合（Gamma、Lognormal）、梯度优化友好的曲线族。

**📊 数据集**

使用了真实ETL系统的观测吞吐量和处理时间数据（未指明具体数据集），通过拟合实验验证模型。

**📈 对比分析**

通过对比观测吞吐量与指数族与有理族曲线的拟合结果以及处理时间分布的Gamma/Lognormal拟合，展示了模型对实际数据的良好匹配；性能主要体现在能在秒至分钟尺度上准确预测总体吞吐。

**⚠️ 局限性**

局限：模型仅关注整体吞吐，未细粒度捕获底层系统细节；假设流量平衡等理想化条件，未考虑非平稳性；提取阶段建模过于简化；未提供针对不同资源调度策略的实验验证。

---

## 486. BayesInsights: Modelling Software Delivery and Developer Experience with Bayesian Networks at Bloomberg

**arXiv ID:** 2603.29929 | [PDF](https://arxiv.org/pdf/2603.29929v1)

**作者:** Serkan Kirbas `[一作]` (Bloomberg), David Williams `[通讯]` (Bloomberg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一个交互式工具 BayesInsights，基于贝叶斯网络可视化软件工程过程中的因果关系，并支持 what‑if 分析。

**💡 创新点**

将 DORA 指标、DevEx 调查数据与专家知识融合，采用混合结构学习和专家审阅构建可解释的贝叶斯网络，并在 Bloomberg 实际部署验证其效用。

**🔧 技术方法**

使用贝叶斯网络、Hill Climbing 与 PC 结构学习算法、BDeu 平滑、BIC 评估、React Flow + Chart.js 前端、RESTful 后端等技术。

**📊 数据集**

利用 20 题季度内部工程师调查（>2000 条回复）以及内部 DevOps 指标（如部署频率、失败率等）作为数据集。

**📈 对比分析**

通过 Hyperfine 单请求（平均 24 ms）和 Locust 并发负载（50 并发用户，中位 <40 ms）测试性能；用户研究显示 95.8 % 认为工具能识别团队/组织级挑战，75 % 易于解读，79.2 % 计划使用。

**⚠️ 局限性**

受限于自评调查数据的偏差与噪声，结构学习可能不完备；工具目前仅在七个团队内预览，缺乏自然语言摘要与结果导向界面，尚未实现全公司部署。

---

## 487. Reducing Subpacketization in Device-to-Device Coded Caching via Heterogeneous File Splitting

**arXiv ID:** 2603.29945 | [PDF](https://arxiv.org/pdf/2603.29945v1)

**作者:** Xiang Zhang `[一作]` (Technical University of Berlin), Mingyue Ji `[通讯]` (University of Florida)

**通讯引用:** 3273 | [OpenAlex ID](https://openalex.org/A5058487273)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文扩展了基于数据包类型的框架，提出了一种异构子数据包化的设备到设备（D2D）编码缓存方案，旨在在保持最佳通信速率的同时减少子数据包化。

**💡 创新点**

创新点在于允许不同类型的数据包具有不同的大小，从而在不均匀用户分组下满足内存约束，填补了现有同质子数据包化设计的空白。

**🔧 技术方法**

使用了基于数据包类型的框架，结合用户分组、组播发射机选择和异构数据包大小的联合优化技术。

**📊 数据集**

论文中使用的数据集为用户数K和文件数N的组合，具体为(K, KM/N)=(2q+1, 2r)，其中q和r为正整数。

**📈 对比分析**

与Ji-Caire-Molisch (JCM)缓存方案相比，提出的方案在多个参数范围内实现了常数因子的子数据包化减少，且保持了相同的速率。

**⚠️ 局限性**

限制在于现有的PT框架仍然对奇数K的情况缺乏通用设计，且在某些参数范围内可能无法实现最佳速率。

---

## 488. Quantale-Enriched Co-Design: Toward a Framework for Quantitative Heterogeneous System Design

**arXiv ID:** 2603.29921 | [PDF](https://arxiv.org/pdf/2603.29921v1)

**作者:** Hans Riess `[一作]` (Georgia Institute of Technology), Matthew Hale `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 512 | [OpenAlex ID](https://openalex.org/A5022644062)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出量子量化的协同设计框架，将传统的布尔可行性扩展为可量化评估；

**💡 创新点**

引入量子代数（Quantale）丰富设计评价维度，并支持不同量化域的异构协同；

**🔧 技术方法**

利用量子代数赋值的范畴论、增广范畴（Enriched Categories）和增广函子（Profunctors）构建并证明系列、平行、反馈三种组合保持有效；

**📊 数据集**

通过目标跟踪系统和无人机投递案例演示，使用人工构造的成本矩阵和实现集合作为数据；

**📈 对比分析**

与布尔化/扩充方法比较，证明量子化方法在避免架构模糊和计算膨胀方面优于传统方法，且能在相同案例中实现更低的成本或更高的可靠性；

**⚠️ 局限性**

缺点包括：需要先定义合适的量子代数和变基映射；实现集合的可实现集成易导致中间空间维度爆炸，影响求解效率；

---

## 489. SISA: A Scale-In Systolic Array for GEMM Acceleration

**arXiv ID:** 2603.29913 | [PDF](https://arxiv.org/pdf/2603.29913v1)

**作者:** Luigi Altamura `[一作]` (Chalmers University of Technology and University of Gothenburg), Pedro Trancoso `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种基于水平分层（slab）并可动态融合与独立调度的尺度可变流水阵列（SISA），用于高效执行LLM推理中的GEMM操作。

**💡 创新点**

创新点在于将传统方形SA拆分为横向矩形子阵列，支持独立或融合运行；引入层级缓冲、输出驻留数据流和电源门控，实现对不规则、偏斜矩阵的自适应利用；并在保持单块性能的同时大幅降低能耗与面积开销。

**🔧 技术方法**

采用了同步流水阵列（systolic array）结构、输出驻留（Output‑Stationary）数据流、双缓冲层级内存、层级分块调度与合并、以及按层电源门控技术。

**📊 数据集**

使用了多种现代LLM模型（Qwen2.5‑0.5B、1.5B、7B以及Llama3.2‑3B）的真实GEMM矩阵尺寸（M,N,K三元组）进行评估。

**📈 对比分析**

与同等PE数的单块TPUv4和ReDas可重构SA做对比；SISA在小至中等批量尺寸下实现最高8.52×速度提升、93%能耗/延迟乘积（EDP）降低，甚至在大批量尺寸下仅增幅8.47% EDP，且在多数场景下优于ReDas，速度提升可达2.61×，面积增幅仅约5.44%。

**⚠️ 局限性**

局限性包括：设计依赖于BF16精度与28nm工艺，较大的内存层级和分块调度逻辑在极大矩阵时仍可能带来额外开销；在极大批量尺寸（>128）下，性能接近单块SA，且对极端偏斜矩阵的处理仍受限于硬件资源与调度策略。

---

## 490. Interview-Informed Generative Agents for Product Discovery: A Validation Study

**arXiv ID:** 2603.29890 | [PDF](https://arxiv.org/pdf/2603.29890v1)

**作者:** Zichao Wang `[一作]` (Adobe Research), Alexa Siu `[通讯]` (Adobe Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建并验证了面试驱动的生成式代理，用于模拟知识工作者在 AI 文档工作流程概念测试中的评价，探讨其在产品发现中的可行性与局限；

**💡 创新点**

首次在产品概念测试场景下系统验证面试驱动生成式代理的分布级别可校准但身份不精确的特性，并提供何时可替代真实访谈、何时仍需真实访谈的实证指导；

**🔧 技术方法**

采用基于记忆-检索-反思-回答的 LLM 代理架构，利用 OpenAI text‑embedding‑3‑small 做记忆嵌入，GPT‑4o 生成回答；评估时使用 MAE、Gwet AC1/AC2、Spearman、Wasserstein 距离等量化指标，并用 LLM‑as‑judge 对开放式回答进行情绪、解释、主题覆盖、语调四维度打分；

**📊 数据集**

51 名知识工作者的深度访谈（约 36 小时录音、约 3,000 条概念测试回答）以及这 51 名参与者对四个 AI 文档工作流程原型的 TAM、NPS 等量化指标与开放式反馈数据；

**📈 对比分析**

通过与同一参与者的两次概念测试真人对照，以及与三种代理（访谈+scratchpad、scratchpad‑only、无信息）对比。个体级别 MAE、AC1/AC2、Spearman 等指标显示代理与真人相差显著，访谈代理约匹配 67% 的人类性能；群体级别 Wasserstein 距离显示访谈代理与真人分布相当，其他两种代理差距明显；开放式回答方面，访谈代理在情绪、解释、主题覆盖上优于其他代理，但整体仍低于真人，语调评估最差。

**⚠️ 局限性**

局限包括：样本量仅 51 人，可能缺乏统计显著性和多样性；代理架构相对简单，未充分探索更先进的记忆/检索/反思方案；结果高度依赖访谈内容与访谈协议，可能不易复现；评估指标与 LLM‑judge 可能存在偏差，未涉及更细粒度的创造性与可操作性评估；仅验证在文档工作流程产品上的可行性，跨领域推广需进一步验证；个体级别缺乏精度，说明仍难以替代真实访谈。

---

## 491. Passive iFIR filters for data-driven velocity control in robotics

**arXiv ID:** 2603.29882 | [PDF](https://arxiv.org/pdf/2603.29882v1)

**作者:** Yi Zhang `[一作]` (University of Cambridge), Fulvio Forni `[通讯]` (University of Cambridge)

**通讯引用:** 1346 | [OpenAlex ID](https://openalex.org/A5034730197)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在Franka Research 3机械臂上设计并实验验证了被动iFIR控制器，用于关节空间和笛卡尔空间的速度控制，采用VRFT方法仅用几分钟探测数据学习控制器，并通过被动约束保证闭环稳定。

**💡 创新点**

创新点：①将被动性约束与VRFT相结合，实现数据驱动的被动iFIR控制器；②通过被动约束保证闭环稳定性；③在实验中比优化PID获得更低跟踪误差，尤其在高阶末端动力学和激进参考模型下表现突出；④支持SISO与MIMO设计，能够快速重学习。

**🔧 技术方法**

使用技术包括：虚拟参考反馈调谐（VRFT）；被动性理论与正实性约束（LMI形式）；iFIR控制器结构（积分+FIR）；线性多输入多输出（MIMO）实现；频域Bode分析；NRMSE指标；对比优化PID。

**📊 数据集**

数据集：约三分钟（关节）或一分钟（笛卡尔）探测数据，由多正弦信号激励，采样频率1 kHz；实验中使用不同长度木板负载；所有数据均来自Franka Research 3机器人。

**📈 对比分析**

与经过VRFT调优的PID基线进行比较，使用NRMSE和改进率评估；结果显示：关节空间step 41% NRMSE降低；笛卡尔MIMO二阶参考模型下改进率达74.5%；在动力学变化时，重新学习可恢复性能；非被动iFIR在大多数情况下不稳定。

**⚠️ 局限性**

局限性：受限于手工设定的参考模型，过激进或过慢模型可能不可行；被动iFIR为线性设计，难以捕捉非线性特性；未实现在线自适应；需要足够高的采样频率和足够长的探测数据以满足被动约束。

---

## 492. Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance

**arXiv ID:** 2603.29941 | [PDF](https://arxiv.org/pdf/2603.29941v1)

**作者:** Vanessa Emanuela Guarino `[一作]` (Max-Delbrück-Center), Dagmar Kainmueller `[通讯]` (Max-Delbrück-Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究图像分割中像素级不确定性向图像级不确定性聚合方法，并系统评估其对异常检测和失效检测任务的影响。

**💡 创新点**

提出了三种基于空间结构的聚合器（Moran’s I、边缘密度、熵）以及一个基于Gaussian Mixture Model的元聚合器（GMM‑All），能够同时捕获强度与空间信息并在多数据集上表现稳健。

**🔧 技术方法**

采用Monte Carlo Dropout生成像素级不确定性，计算多种聚合指标，并通过GMM对聚合特征进行建模，最后使用AUROC和E‑AURC评估异常检测和失效检测性能。

**📊 数据集**

在十个跨域、多模态数据集上验证，包括医学影像（ARC、LIZ、LIDC、WORM）、城市街景（CAR‑ID/CAR‑CS）和农业植被（WEED）。

**📈 对比分析**

与传统的全局平均（AVG）以及阈值/块/类加权聚合相比，空间感知聚合器在AUROC上提升约5‑10%，而GMM‑All在所有数据集上均稳健领先，平均排名始终位列前列。

**⚠️ 局限性**

局限性包括：GMM‑All需要足够多的正常样本才能学习分布；对高维低样本场景易过拟合；空间聚合器的超参数（窗口大小、阈值等）需人工调优；仅针对2D分割，3D情境仍待验证。

---

## 493. Detecting Unknown Objects via Energy-based Separation for Open World Object Detection

**arXiv ID:** 2603.29954 | [PDF](https://arxiv.org/pdf/2603.29954v1)

**作者:** Jun-Woo Heo `[一作]` (Korea University), Gyeong-Moon Park `[通讯]` (Korea University)

**通讯引用:** 367 | [OpenAlex ID](https://openalex.org/A5031148632)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为DEUS的开放世界目标检测框架，通过在特征空间构建已知与未知子空间，并用能量分离实现未知目标的更好识别；同时设计EKD损失在内存回放时抑制旧新类别间干扰

**💡 创新点**

创新点在于：①使用等角紧致框架ETF构造双子空间，实现对已知与未知的几何分离；②在能量基准下同时考虑两空间能量，提升未知检测；③将能量差分作为正负样本区分的EKD损失，有效缓解连续学习中的灾难性遗忘

**🔧 技术方法**

主要技术包括等角紧致框架（ETF）、能量函数、焦点损失、伪标签生成、内存回放、基于OrthogonalDet的目标检测网络

**📊 数据集**

在M-OWODB、S-OWODB、RS-OWODB三个公开开放世界检测基准上进行实验

**📈 对比分析**

与多种SOTA方法（ORE、OW-DETR、CAT、PROB、OrthogonalDet、O1O、OWOBJ）比较，DEUS在所有任务上均取得最高的H-Score和远高于其他方法的未知召回率，同时保持或提升已知类的mAP

**⚠️ 局限性**

仍存在已知与未知语义重叠导致的误检问题，且对复杂场景的适应性和伪标签质量依赖较高

---

## 494. Training deep learning based dynamic MR image reconstruction using synthetic fractals

**arXiv ID:** 2603.29922 | [PDF](https://arxiv.org/pdf/2603.29922v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 495. Real-Time Explanations for Tabular Foundation Models

**arXiv ID:** 2603.29946 | [PDF](https://arxiv.org/pdf/2603.29946v1)

**作者:** Luan Borges Teodoro Reis Sena `[一作]` (Kunumi Institute), Francisco Galuppo Azevedo `[通讯]` (Kunumi Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 ShapPFN，一种能够在一次前向传播中同时输出预测和 SHAP 解释的表格基础模型。

**💡 创新点**

将 Shapley 值回归直接嵌入 Transformer 架构，并通过专门的损失函数实现解释与预测的一致性，首次实现了在表格基础模型中实时、高保真解释。

**🔧 技术方法**

使用基于 NanoTabPFN 的 Transformer 结构，加入 BaseDecoder 与 ShapDecoder 头，并采用 ViaSHAP 风格的掩码损失以及混合交叉熵训练。

**📊 数据集**

在 OpenML‑CC18 基准套件（包含 28 个分类任务）以及 256,000 条 TabICL 合成训练集上进行评估。

**📈 对比分析**

与 TabPFN、TabICL、随机森林等基线相比，ShapPFN 维持相近的 ROC‑AUC（0.848）并在 SHAP 解释方面与 KernelSHAP 近乎一致（R²≈0.96、余弦相似度≈0.99），同时推理速度提升至 0.06 s（≈1000× 加速）。

**⚠️ 局限性**

解释质量仍受限于掩码策略和特征维度，且预训练阶段相比 NanoTabPFN 计算成本略高，且对极高维数据的解释效果尚待进一步验证。

---

## 496. GreenFLag: A Green Agentic Approach for Energy-Efficient Federated Learning

**arXiv ID:** 2603.29933 | [PDF](https://arxiv.org/pdf/2603.29933v1)

**作者:** Theodora Panagea `[一作]` (National and Kapodistrian University of Athens), Ramin Khalili `[通讯]` (Huawei Technologies)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 GreenFLag——一个基于代理的资源编排框架，在联邦学习中利用可再生能源和强化学习来最小化电网能耗。

**💡 创新点**

创新点在于将实时可再生能源可用性嵌入强化学习决策循环，并通过安全奖励机制实现绿色优先的资源分配；同时结合FCFS调度保证带宽分配的可行性。

**🔧 技术方法**

使用 Soft‑Actor‑Critic（SAC）深度强化学习、FCFS 带宽调度器、可再生能源模型与电池状态管理等技术。

**📊 数据集**

使用 Copernicus 气象数据生成太阳能与风能，MNIST 数据集进行联邦学习实验。

**📈 对比分析**

与 Best Effort、Random Selection、Greedy Selection 三种基线比较，GreenFLag 在三种可再生能源场景下平均总能耗降低 94.8%，电网能耗下降 8‑16 倍，保持相同收敛速度和违约率。

**⚠️ 局限性**

主要局限是实验基于仿真环境，未验证在真实移动网络中的部署效果、延迟与不同模型复杂度或设备异构环境下的进一步泛化。

---

## 497. Generative AI in Action: Field Experimental Evidence from Alibaba's Customer Service Operations

**arXiv ID:** 2603.29888 | [PDF](https://arxiv.org/pdf/2603.29888v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 498. Gloria: Consistent Character Video Generation via Content Anchors

**arXiv ID:** 2603.29931 | [PDF](https://arxiv.org/pdf/2603.29931v1)

**作者:** Yuhang Yang `[一作]` (University of Science and Technology of China), Zheng-Jun Zha `[通讯]` (University of Science and Technology of China)

**通讯引用:** 19284 | [OpenAlex ID](https://openalex.org/A5003217535)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Gloria模型，通过使用全球、视角和表情内容锚点以及Superset Content Anchoring和RoPE作为弱条件，能够从文本、图像、音频等多模态输入生成持续10分钟以上的数字角色视频，并保持多视角外观与表情身份一致。

**💡 创新点**

首次将内容锚点概念引入视频生成，结合Superset Content Anchoring防止复制粘贴、RoPE弱条件分离不同锚点，构建自动化锚点提取管线，显著提升长期一致性。

**🔧 技术方法**

基于3D Diffusion Transformer（DiT）和Wan‑I2V架构，使用VAE编码、RoPE位置编码、wav2vec音频特征、流匹配训练目标以及分块自回归推理的多模态控制技术。

**📊 数据集**

使用大规模单人视频数据集（包含2M+训练片段，10M+补充片段和500K包含锚点的片段），通过自动化管线从公开视频中提取视角与表情锚点。

**📈 对比分析**

与InfiniteTalk、HunyuanAvatar、WanS2V等多种基线在长时一致性、视角外观与表情一致性以及基础能力指标上进行对比，Gloria在Arcface、DINO‑I、CLIP‑I、IQA、AES等指标上均显著优于对手，并在用户研究中获得最高偏好率。

**⚠️ 局限性**

仍需手动或半自动提供锚点，模型对复杂场景、遮挡或光照变化的鲁棒性有限；并且模型参数量大（17B），推理成本高。

---

## 499. SkillReducer: Optimizing LLM Agent Skills for Token Efficiency

**arXiv ID:** 2603.29919 | [PDF](https://arxiv.org/pdf/2603.29919v1)

**作者:** Yudong Gao `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 25622 | [OpenAlex ID](https://openalex.org/A5100328273)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM编程代理技能（Skill）进行去冗余与压缩，降低上下文窗口中占用的token量

**💡 创新点**

结合delta‑debugging压缩描述层和基于分类的分层重构正文，形成“SkillReducer”两阶段无训练的去冗余框架

**🔧 技术方法**

采用delta‑debugging、语义句块分割、LLM分类器、摘要/去重与进阶公开式工具调用机制

**📊 数据集**

共分析55,315个公开Skill，评测集为600个Skill（官方、社区、野生）以及SkillsBench基准

**📈 对比分析**

与四种基线（LLMLingua、直接LLM压缩、截断、随机删句）以及五大模型、独立代理框架比较，平均压缩率约48%/39%，功能通过率达86%，在同等token预算下性能优于基线且具有less‑is‑more效果

**⚠️ 局限性**

局限于LLM分类随机性、仅针对Anthropic协议的Skill、未覆盖所有工具/语言平台，且压缩后仍需人工或评测确认功能完整

---

## 500. Structured Intent as a Protocol-Like Communication Layer: Cross-Model Robustness, Framework Comparison, and the Weak-Model Compensation Effect

**arXiv ID:** 2603.29953 | [PDF](https://arxiv.org/pdf/2603.29953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 501. XR is XR: Rethinking MR and XR as Neutral Umbrella Terms

**arXiv ID:** 2603.29939 | [PDF](https://arxiv.org/pdf/2603.29939v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 502. ATP-Bench: Towards Agentic Tool Planning for MLLM Interleaved Generation

**arXiv ID:** 2603.29902 | [PDF](https://arxiv.org/pdf/2603.29902v1)

**作者:** Yinuo Liu `[一作]` (Qwen Large Model Application Team, Alibaba), Guanjun Jiang `[通讯]` (Qwen Large Model Application Team, Alibaba)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Agentic Tool Planning范式和对应的Agentic Tool Planning Bench基准，用于评估多模态大型语言模型在交互式图文生成中的工具调用规划能力。

**💡 创新点**

创新点包括：①将图像检索与图像生成统一到同一交互响应中；②构建专业级视觉关键意图的QA数据集；③设计MAM（Multi‑Agent MLLM‑as‑a‑Judge）框架，拆分工具调用精度、缺失工具机会和整体质量的评估；④提供系统化的工具调用元数据与评价标准。

**🔧 技术方法**

采用多模态大型语言模型（如Gemini 3 Pro、Claude Sonnet 4.5等）、五种工具（Reference、Diffusion、Search、Code、Edit）和MAM三代理（Precision Inspector、Recall Inspector、Chief Judge）进行评估。

**📊 数据集**

使用7,702条QA（含1,592 VQA）样本，覆盖学术、手册、食谱、时尚、装修、产品、旅行、百科八大类，含25个视觉关键意图；还引用了OpenLEAF、InterleavedBench等公开数据集做文档来源。

**📈 对比分析**

通过MAM得到Final Score、Success Rate、Missed Images等指标，Gemini 3 Pro在大多数类别和平均分上领先（FS≈79.9，SR≈81.8，MI≈0.49）。相比之下，Claude Sonnet 4.5、Claude 4、Grok‑4.1、GPT‑5等第二梯队表现相近，GPT‑4o、Qwen3‑VL‑Plus等中梯队略逊，开源模型表现最低。

**⚠️ 局限性**

局限性：仅覆盖文本–图像交互，未考虑音视频等更丰富模态；工具集仅限五种，无法体现更广泛的代理能力；评估仅通过MAM，未尝试其他评判管线；数据集与任务场景可能不足以覆盖全部实际应用。

---

## 503. A Hybrid Machine Learning Approach for Graduate Admission Prediction and Combined University-Program Recommendation

**arXiv ID:** 2603.29881 | [PDF](https://arxiv.org/pdf/2603.29881v1)

**作者:** Melina Heidari Far `[一作]`, Elham Tabrizi `[通讯]` (Kharazmi University)

**通讯引用:** 23 | [OpenAlex ID](https://openalex.org/A5020331429)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究构建了一个包含13,000条竞争性研究生录取记录的数据集，并提出一种混合机器学习框架用于预测录取结果及给出可行的大学和专业推荐。

**💡 创新点**

创新点在于将XGBoost与kNN的残差细化模块相结合，以精准处理边缘案例，同时通过多源机构与项目元数据丰富特征，实现了更高的预测准确率和针对被拒绝申请者的定制化推荐。

**🔧 技术方法**

使用的技术包括XGBoost、kNN、Platt标定、SHAP特征解释、PCA可视化，以及基于规则的推荐算法。

**📊 数据集**

使用的数据集来自GradCafe自报录取记录，并通过OpenAlex、QS世界大学排名和Wikidata进行机构和项目层面的信息丰富。

**📈 对比分析**

与传统的逻辑回归、决策树、随机森林、SVM、LightGBM等基线模型比较，混合模型在测试集上达到了87%的准确率（相比XGBoost的72%），并在推荐实验中平均提升录取概率约65%。

**⚠️ 局限性**

局限性包括数据高度聚焦于美国高校，导致模型对非美国背景的泛化能力有限；此外，缺失的GRE分数和高GPA范围内的分布不均可能引入偏差，且推荐系统仍受预算和地理偏好等约束。

---

## 504. ShapE-GRPO: Shapley-Enhanced Reward Allocation for Multi-Candidate LLM Training

**arXiv ID:** 2603.29871 | [PDF](https://arxiv.org/pdf/2603.29871v1)

**作者:** Rui Ai `[一作]` (MIT), Chonghuan Wang `[通讯]` (University of Texas at Dallas)

**通讯引用:** 50 | [OpenAlex ID](https://openalex.org/A5065394341)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 ShapE‑GRPO，利用 Shapley 值对 LLM 多候选生成任务中的集级奖励进行细粒度拆分；

**💡 创新点**

创新点在于将集级奖励通过 Shapley 值分配给每个候选，解决共享奖励导致的免费搭车问题，并证明在候选级别可在多项式时间内精确计算；

**🔧 技术方法**

结合 GRPO 强化学习框架、Shapley 值理论、候选级奖励分配策略，并在 Qwen3‑8B 模型上实现；

**📊 数据集**

使用 ACLSum 摘要数据集、DS‑1000 代码生成数据集以及自构建的 Netflix 推荐数据集；

**📈 对比分析**

与标准 GRPO 和 Winner‑Takes‑All 基线对比，ShapE‑GRPO 在所有三任务上均取得更高评测分数，收敛速度更快、训练更稳定；

**⚠️ 局限性**

局限性包括对候选长度相等的假设、仅处理单一候选最大化奖励的设置、未探讨其他信用分配方法或无监督候选评估方式。

---

## 505. ScoringBench: A Benchmark for Evaluating Tabular Foundation Models with Proper Scoring Rules

**arXiv ID:** 2603.29928 | [PDF](https://arxiv.org/pdf/2603.29928v1)

**作者:** Jonas Landsgesell `[一作]`, Pascal Knoll `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 ScoringBench 的基准，用于评估表格回归模型的完整概率分布，并引入多种严格的评分规则。

**💡 创新点**

创新点在于：①提出并实现了一套丰富的严格评分规则（CRPS、CRLS、区间得分、能量得分、加权 CRPS、Brier 等），②使用排列检验方法对模型在多数据集上的排名进行统计显著性检验，③展示了不同评分规则对模型排名的显著影响，提示评估指标需与业务风险对齐。

**🔧 技术方法**

采用了 TabPFN/TabICL 等分布式回归模型、XGBoost 向量化量化回归、各种评分规则实现、Git 拉取式排行榜以及基于排列检验的排名方法。

**📊 数据集**

使用 OpenML 269、297、299 套件中的 46 个回归数据集（每个样本不超过 3000 条），包括 Ailerons、Boston、California、Wine Quality 等多领域数据。

**📈 对比分析**

比较方法：在每个数据集上做 5 折交叉验证，先平均每折结果再按数据集内部排名归一化；随后对所有模型的平均排名进行 20,000 次排列检验。结果显示不同评分规则会导致模型排名差异，例如 TabICL 在 CRPS 上排名第一，finetune_realtabpfnv2_5_crls 在 R² 上排名第一。

**⚠️ 局限性**

局限性包括：仅支持单目标表格回归，未覆盖多元目标；评估仅限于所选评分规则集合；对高维度数据（如 Santander）支持不足；排列检验基于排名，忽略了绝对性能差距；缺乏针对特定业务风险的自定义评分规则。

---

## 506. Diffusion-Based Feature Denoising with NNMF for Robust handwritten digit multi-class classification

**arXiv ID:** 2603.29917 | [PDF](https://arxiv.org/pdf/2603.29917v1)

**作者:** Hiba Adil Al-kharsan `[一作]`, Róbert Rajkó `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在手写数字多类分类任务中，提出了一种将CNN深度特征、NNMF可解释特征和特征空间扩散去噪相结合的混合框架，并在干净数据与AutoAttack对抗攻击下进行评估。

**💡 创新点**

创新点在于：①在特征空间引入扩散去噪机制，直接对混合特征进行去噪，以提升对抗鲁棒性；②将可解释的NNMF分解特征与CNN特征融合，既保留高层语义又提供部分可解释性；③通过多步扩散和逆向去噪网络实现对噪声扰动的逆转。

**🔧 技术方法**

使用的技术包括CNN网络提取深层特征、NNMF进行非负矩阵分解获得可解释部分特征、特征级扩散模型及去噪网络、AutoAttack框架进行对抗攻击评估，以及ONNX/ PyTorch转换实现跨平台推理。

**📊 数据集**

使用的数据集为MATLAB自带的手写数字数据集（≈10,000张灰度图，10类），与MNIST相似，样本均衡，先归一化后划分为70%训练、15%验证、15%测试。

**📈 对比分析**

评估方法：将基线CNN模型与加入扩散去噪的防御模型在干净与对抗样本上进行对比，采用准确率、精确率、召回率、F1、MCC、平衡准确率、ROC‑AUC、log‑loss、Brier分数等多指标。结果显示，基线模型在干净数据上略优，但在对抗攻击下准确率降至0；防御模型虽然干净数据准确率略低（≈0.98），但在对抗攻击下准确率提升到≈0.20，显著提高鲁棒性。

**⚠️ 局限性**

局限性：仅在简易手写数字数据集上验证，缺乏对更复杂数据集（如CIFAR、ImageNet）的泛化评估；防御模型在极端对抗攻击下仍显著受限；扩散去噪过程增加计算开销；未对不同扩散步数、噪声强度等超参数进行深入探讨。

---

## 507. Uncertainty Gating for Cost-Aware Explainable Artificial Intelligence

**arXiv ID:** 2603.29915 | [PDF](https://arxiv.org/pdf/2603.29915v1)

**作者:** Georgii Mikriukov `[一作]` (Leibniz Institute for Agricultural Engineering and Bioeconomy), Marina M. -C. Höhne `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种基于表观不确定性（epistemic uncertainty）的门控机制，用来在生成后置解释（XAI）时根据解释可靠性动态分配计算资源，支持在资源受限或需要高质量解释的场景下进行自适应方法选择或推迟解释生成。

**💡 创新点**

创新点在于：①首次将表观不确定性作为低成本的可靠性指示器，直接预测解释稳定性和可信度；②提出可量化的“不确定性门控”框架，能够在不同解释成本和预算约束下实现最佳的成本-质量权衡；③验证该方法在多种模型、解释方法和数据类型（表格与图像）上的一致性，并提供可复现的代码和实验。

**🔧 技术方法**

使用技术包括：随机森林/MC Dropout/轻量级随机森林代理产生表观不确定性；多种后置解释方法（SHAP、LIME、Integrated Gradients、SmoothGrad、SmoothIG）；稳定性度量（Kendall's τ、Spearman ρ、SSIM）；实验设计中使用分层验证、特征移除和噪声归因；成本-质量分析公式与阈值阈值化实现。

**📊 数据集**

数据集：四个UCI表格数据集（Wine、Dry Bean、Rice、Ecoli）和一个图像分类子集（PlantVillage）。

**📈 对比分析**

与基线比较：在多数设置下，表观不确定性与解释稳定性呈显著负相关（XEC < -0.6），低不确定性样本解释在Kendall τ、SSIM等指标上明显更高；通过门控后，平均稳定性提升且计算成本按预期下降，尤其对高成本解释方法（如LIME）效果显著；实验表明轻量级代理可替代原始模型的不确定性估计，保持性能。

**⚠️ 局限性**

局限性包括：①在不确定性分布散布低（低CV_epi）的数据集上，无法有效区分稳定与不稳定解释；②对某些扰动类型（如特征置换）相关性弱；③不确定性阈值需数据集特定校准，缺乏通用阈值；④轻量级代理可能不完全反映目标模型的参数不确定性，导致门控误判；⑤目前仅针对分类任务，其他任务如回归、生成等仍需探索。

---

## 508. Task Scarcity and Label Leakage in Relational Transfer Learning

**arXiv ID:** 2603.29914 | [PDF](https://arxiv.org/pdf/2603.29914v1)

**作者:** Francisco Galuppo Azevedo `[一作]` (Kunumi Institute), Denis Oliveira Correa `[通讯]` (Kunumi Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了关系型数据库基线模型中因任务稀缺导致的标签泄露现象，并提出了一种基于样本级梯度投影的方法以抑制表示学习中的标签预测方向；

**💡 创新点**

创新点在于首次将每样本梯度投影用于去除表示中的标签泄露，显著提升了同一数据库内部的迁移性能；

**🔧 技术方法**

技术上使用了K‑Space模块化架构——冻结TabICL表编码器、SMPNN式消息传递核心以及带对抗头的梯度投影机制；

**📊 数据集**

实验数据集来自RelBench，涵盖多种真实关系型数据库的二分类任务（如driver‑dnf、avito、event、stack、amazon等）；

**📈 对比分析**

与RelGT transformer基线在ST、WD、CD、ALL四种监督模式下对比，梯度投影在WD模式平均提升0.145 AUROC，部分任务接近或超过RelGT；

**⚠️ 局限性**

局限性包括仅处理分类任务、对跨数据库迁移无显著提升、使用冻结编码器/预测头限制了模型表达能力、以及多任务训练时出现负迁移。

---

## 509. Security and Privacy in Virtual and Robotic Assistive Systems: A Comparative Framework

**arXiv ID:** 2603.29907 | [PDF](https://arxiv.org/pdf/2603.29907v1)

**作者:** Nelly Elsayed `[一作]` (University of Cincinnati), Nelly Elsayed `[通讯]` (University of Cincinnati)

**通讯引用:** 452 | [OpenAlex ID](https://openalex.org/A5108005607)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

对虚拟助理系统与机器人辅助系统的安全与隐私挑战进行比较分析，提出统一的威胁模型与评估框架。

**💡 创新点**

创新点在于将数字助理与网络物理系统纳入同一比较维度，构建跨系统的威胁模型、攻击类型分类和风险评估方法。

**🔧 技术方法**

采用威胁建模、架构对比分析、定性风险评估以及安全设计建议等方法；在此基础上提出了强身份验证、隐私保护、传感器冗余与人机安全控制等技术方案。

**📊 数据集**

未使用公开数据集，研究以理论模型和架构分析为主。

**📈 对比分析**

通过对攻击的发生概率、系统影响和安全关键性进行三维定性比较来评估风险；性能表现以风险分级和对比矩阵呈现，未给出量化指标。

**⚠️ 局限性**

局限性包括：评估仍为定性描述，缺乏实验验证和量化数据；对具体实现细节和不同硬件平台的适配性讨论不足；未覆盖所有可能的攻击向量。

---

## 510. Perfecting Human-AI Interaction at Clinical Scale. Turning Production Signals into Safer, More Human Conversations

**arXiv ID:** 2603.29893 | [PDF](https://arxiv.org/pdf/2603.29893v1)

**作者:** Subhabrata Mukherjee `[一作]` (Hippocratic AI), Jonathan Agnew `[通讯]` (Hippocratic AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并验证了生产级医疗对话 AI 框架 Polaris，集成多模型协同、交互智能、上下文 ASR、实时校验与长期记忆，实现大规模患者–AI 通话。

**💡 创新点**

将实时交互信号视为安全变量；治理型多模型共振与两阶段校验；上下文 ASR + 单词纠错；情感自适应对话；多语言实时切换；以生产数据驱动的 RWE‑LLM 评估。

**🔧 技术方法**

多模型架构（核心 LLM + 30+ 专家模型）、KV 缓存与路由、上下文 ASR、单词纠错、目标化澄清、情感自适应对话、长序列记忆、硬件加速（H200 GPU、300B/405B 结构剪枝）、TTS 声音转换、RAG、两阶段校验。

**📊 数据集**

115M+ 实时患者–AI 通话、7000+ 许可临床医生模拟、500K+ 诊断呼叫、内部多语言/噪声语音数据、公开医疗 QA 集（MedQA、PubMedQA 等）、内部 ASR 评测集、公开 RAG 文档与 IVR 语料。

**📈 对比分析**

对比 GPT‑4o、Hippocratic AI 等；在真实通话中错误率降至 0.01%；Heart 支持度领先；TTFT 400 ms；多轮记忆 92% 质量；多语言精度提升；整体安全 99.9% no‑harm；语音识别错误下降 50%；运营部署提升效率、减少转接率、患者满意度 8.95/10。

**⚠️ 局限性**

对极端噪声或方言的 ASR 仍高 WER；多模型并发部署带来系统复杂度和成本；极长对话仍可能出现漂移；未覆盖所有临床路径；需持续监督验证；在低资源语言和罕见病症方面性能有限。

---

## 511. FLEURS-Kobani: Extending the FLEURS Dataset for Northern Kurdish

**arXiv ID:** 2603.29892 | [PDF](https://arxiv.org/pdf/2603.29892v1)

**作者:** Daban Q. Jaff `[一作]` (University of Erfurt), Mohammad Mohammadamini `[通讯]` (Le Mans University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并发布北库尔德语（KMR）语音数据集FLEURS-Kobani，并为其在ASR、S2TT和S2ST任务上提供基准评估。

**💡 创新点**

首次为北库尔德语提供完整的多任务语音资源与标准化评测框架，填补了FLEURS与Common Voice中该方言的空白。

**🔧 技术方法**

使用Whisper v3‑large模型进行ASR与端到端S2TT微调，Cascade S2TT结合ASR+NLLB‑1.3B翻译模型。

**📊 数据集**

主要使用FLEURS‑Kobani（5,162句，18h 24min）以及Common Voice北库尔德语（68h）进行预训练和微调。

**📈 对比分析**

通过比较WER和BLEU指标与基线，最佳两阶段微调得到DEV/WER≈28.5%、TEST/WER≈28.1%；端到端S2TT在EN上BLEU≈9，Cascade S2TT在EN上BLEU≈20。

**⚠️ 局限性**

录音样本存在性别与录音质量偏差，受教育水平限制导致发音与文本匹配问题，约34%录音被剔除，数据规模仍相对有限。

---

## 512. Tucker Attention: A generalization of approximate attention mechanisms

**arXiv ID:** 2603.30033 | [PDF](https://arxiv.org/pdf/2603.30033v1)

**作者:** Timon Klein `[一作]` (Otto von Guericke University Magdeburg), Steffen Schotthöfer `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 50 | [OpenAlex ID](https://openalex.org/A5086133644)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Tucker 分解的自注意力压缩框架——Tucker Attention，统一并推广了 MHA、MQA、GQA、MLA 等近似注意力方法

**💡 创新点**

核心创新在于将预 Softmax 与后 Softmax 权重视为三阶张量，通过 Tucker 分解实现对 head、query、key、value、output 五个模式的低秩压缩，从而实现更高的参数压缩率并兼容 KV 缓存、RoPE 与 Flash‑Attention

**🔧 技术方法**

技术包括张量分解（Tucker）、低秩矩阵/张量压缩、KV 缓存机制、旋转位置嵌入（Latent RoPE）、Flash‑Attention 兼容实现

**📊 数据集**

在 Vision Transformer（ViT）ImageNet1k、Cifar10/100、GPT‑2、LLaMA3‑1B 等模型上进行实验，并使用公开数据集 OpenWebText、Shakespeare、ImageNet1k、Cifar10/100 进行评估

**📈 对比分析**

与传统 MHA、MQA、GQA、MLA 以及 Flash‑Attention 进行对比，Tucker Attention 在参数数目、KV 缓存量、显存占用、训练/推理时间等指标上均能以 1/10 左右的参数量获得与传统方法相当或更优的验证指标，且在 LLaMA3‑1B 训练中降低 20% 以上的迭代时间

**⚠️ 局限性**

局限性包括：仍需针对不同硬件实现更高效的自定义内核；对极低秩设置（如 r≤2）可能导致性能下降；对大型模型的可扩展性与稳定性（如高 KV 维度）尚待进一步验证

---

## 513. Reward-Based Online LLM Routing via NeuralUCB

**arXiv ID:** 2603.30035 | [PDF](https://arxiv.org/pdf/2603.30035v1)

**作者:** Ming-Hua Tsai `[一作]` (Oregon State University), Phat Tran `[通讯]` (Oregon State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究基于NeuralUCB的成本感知LLM路由，构建UtilityNet预测器并在RouterBench模拟线上环境中进行评估。

**💡 创新点**

将NeuralUCB与可解释的UtilityNet结合，使用门控机制决定是否使用UCB探索，并在多元上下文特征下实现非线性路由。

**🔧 技术方法**

NeuralUCB、神经上下文多臂赌博机、Transformer句子编码器、共享逆协方差估计、门控分支等技术。

**📊 数据集**

RouterBench（36,497条样本，86个域，11个候选LLM）。

**📈 对比分析**

与随机、最小成本、RouteLLM‑BERT基线比较，NeuralUCB平均奖励稳定在0.59‑0.61，累计奖励高于所有基线，成本约为最大质量方案的33%。

**⚠️ 局限性**

行动区分度不足、探索与实际收益匹配不完美、需要更多训练数据和更精准的效用模型。

---

## 514. EnsembleSHAP: Faithful and Certifiably Robust Attribution for Random Subspace Method

**arXiv ID:** 2603.30034 | [PDF](https://arxiv.org/pdf/2603.30034v1)

**作者:** Yanting Wang `[一作]` (Pennsylvania State University), Jinyuan Jia `[通讯]` (Pennsylvania State University)

**通讯引用:** 2139 | [OpenAlex ID](https://openalex.org/A5101997385)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对随机子空间方法（Random Subspace Method）提出了一种计算高效且可证明安全的特征归因方法，能够解释模型预测并在存在攻击时检测到被修改的关键特征。

**💡 创新点**

创新点在于：①引入基于随机子空间采样概率的归因公式，直接利用已生成的子模型预测结果，极大降低了计算开销；②在理论上证明该方法在面对解释保持攻击（explanation‑preserving attack）时具备可证明的鲁棒性；③在本质上保留了 Shapley 值的局部准确性与对称性，并通过“顺序一致性”保证与 Shapley 值的排序相匹配。

**🔧 技术方法**

主要技术包括：随机子空间采样、Monte Carlo 估计、概率归因（α_i^ŷ=1/k E_z[ 1_{x_i∈z}·1_{h(z)=ŷ} ]）、归一化修正以消除采样不均衡、理论证明（局部准确性、对称性、顺序一致性）以及对抗解释保持攻击的证明和可证检测阈值计算。

**📊 数据集**

在文本分类任务上使用 SST‑2、IMDB、AGNews 数据集；在反 jailbreak 攻击上使用公开的 harmful‑behaviour 数据集；实验中还在图像域对抗补丁攻击做了验证。

**📈 对比分析**

与 Shapley、LIME、ICL 等基线相比，本文方法在无攻击和多种攻击（Backdoor、Adversarial、Jailbreaking）下的 faithfulness、关键字召回率、以及可证检测率均显著优于基线；计算时间仅增 0.03–0.5 秒。

**⚠️ 局限性**

限制包括：①仍需足够的子样本数 N（如 1k–10k）以获得稳定估计；②对高降采样率 ρ 仍可能导致模型对特征删除不敏感；③证明仅针对随机子空间方法，对其它模型的适用性尚未完全验证。

---

## 515. ContextClaim: A Context-Driven Paradigm for Verifiable Claim Detection

**arXiv ID:** 2603.30025 | [PDF](https://arxiv.org/pdf/2603.30025v1)

**作者:** Yufeng Li `[一作]` (Queen Mary University of London), Arkaitz Zubiaga `[通讯]` (Queen Mary University of London)

**通讯引用:** 6737 | [OpenAlex ID](https://openalex.org/A5071220716)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Context-Driven Claim Detection (ContextClaim) 模型，在自动事实核查的早期阶段利用实体检索和大模型摘要增强可验证性判断。

**💡 创新点**

首次将检索与摘要推送至声明检测阶段，并通过实体抽取和 Wikipedia 背景知识支持可验证性判定。

**🔧 技术方法**

使用 BERT/ RoBERTa、Llama3、Mistral 等预训练模型，结合 NER、基于稠密向量的检索、跨注意力或提示式集成以及 GPT‑4o/Mistral‑7B‑Instruct 的摘要生成。

**📊 数据集**

在 COVID‑19 推文集合 CheckThat! 2022 (CT22) 与政治辩论摘录 PoliClaim 两个标注可验证性的英文数据集上进行评估。

**📈 对比分析**

在 fine‑tuning、zero‑shot 与 few‑shot 三种学习设置下与仅使用声明文本的基线对比，ContextClaim 在 20 种模型‑数据配置中有 13/14 的准确率/ F1 提升，尤其在 fine‑tuning 与 CT22 上效果最显著。

**⚠️ 局限性**

依赖 Wikipedia 作为唯一知识源、摘要的信号清晰度不足、不同模型对外部上下文的稳定性差异以及在少样本与跨领域泛化上的局限。

---

## 516. Approximation algorithms for satisfiable and nearly satisfiable ordering CSPs

**arXiv ID:** 2603.30020 | [PDF](https://arxiv.org/pdf/2603.30020v1)

**作者:** Yury Makarychev `[一作]` (Toyota Technological Institute at Chicago), Yury Makarychev `[通讯]` (Toyota Technological Institute at Chicago)

**通讯引用:** 1888 | [OpenAlex ID](https://openalex.org/A5068045392)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一般化框架用于设计满足式和近满足式排序约束满足问题（Ordering CSP）的近似算法，并通过引入强 IDU 变换将搜索空间限制为可优化的有限维度问题，进而实现对 arity 4 约束谓词的非平凡近似；

**💡 创新点**

核心创新在于引入强 IDU 变换概念，证明任何弱 IDU 变换可被强 IDU 变换取代且不降低性能，同时给出强 IDU 变换的完整分类（基于 I、D、U 的 up‑combination 与核心 permuton 结构），并将优化问题化为多元多项式的最大化；

**🔧 技术方法**

主要技术包括：随机 permuton 的使用、IDU 变换的模式密度分析、up‑combination 运算、强 IDU 变换的可测性与一致性证明、quasisymmetric 多项式与旗代数框架来描述模式密度空间、以及对强 IDU permuton 的近似构造（ID、IDU 组合）；

**📊 数据集**

论文未依赖传统公开数据集，而是对所有 arity ≤4 的排序谓词进行枚举，使用符号计算与精确算术（ℚ）验证各谓词的可近似性和最佳变换；

**📈 对比分析**

比较方法为：对每个谓词在满足式/近满足式场景下的随机排序与所提出算法的约束满足率进行比较；实验结果显示在 arity‑3 中仅 Betweenness 与其对偶可获得非平凡近似；在 arity‑4 中至少有 39,299 个单谓词可在完全满足式下取得超过随机阈值的近似，且 843 个单谓词在近满足式下也实现非平凡近似；

**⚠️ 局限性**

局限性包括：对高 arity 的计算复杂度随 k 的阶乘增长；仅适用于排序谓词（不包含等号约束）；近似系数仍受 U 变换影响，无法实现对所有谓词的最优近似；此外，近似算法在 ε=O(1/polylog(n)) 处的误差为 O(ε log n loglog n) 仍相对较大。

---

## 517. Refined Detection for Gumbel Watermarking

**arXiv ID:** 2603.30017 | [PDF](https://arxiv.org/pdf/2603.30017v1)

**作者:** Tor Lattimore `[一作]` `[通讯]`, Tor Lattimore

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型的Gumbel水印方法，提出了一种截断幂律统计量用于水印检测，旨在降低所需文本长度。

**💡 创新点**

创新点在于将传统指数统计替换为截断幂律统计，提供了更精细的理论上限与下限，证明在不知模型的情况下该方案几乎最优。

**🔧 技术方法**

主要技术包括Goodness‑of‑fit检验、信息论（相对熵、χ² 散度）、大数定律、贝叶斯-Huber不等式以及中心极限定理。

**📊 数据集**

实验部分未使用公开数据集，主要通过合成示例验证理论预测，缺乏大规模真实文本的评估。

**📈 对比分析**

与原有指数检测方案对比，理论上检测所需 token 数更少（n≈1/H̅ 而非 1/H̅²），但在实际语言数据上由于常数与对数项影响，性能略逊于指数检测。

**⚠️ 局限性**

局限性包括：1) 需要对token分布的熵量化估计；2) 截断幂律统计在实际文本中常数项较大，导致对数因子影响显著；3) 缺乏针对真实大型语言模型的实证验证。

---

## 518. Conditional Polarization Guidance for Camouflaged Object Detection

**arXiv ID:** 2603.30008 | [PDF](https://arxiv.org/pdf/2603.30008v1)

**作者:** QIfan Zhang `[一作]`, Ruijie Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的不对称条件极化引导框架CPGNet，用极化信息作为条件指导RGB特征学习，实现隐形物体检测。

**💡 创新点**

创新点在于将极化视为结构性条件而非并行模态；设计轻量化极化融合模块、条件极化引导流（包括极化增强和边缘引导频域细化），以及迭代反馈解码器，实现多层次、渐进式的特征调节和细化。

**🔧 技术方法**

采用Transformer骨干网络提取RGB特征，结合极化积分模块（DoLP、AoLP互补）、频域增强、迭代反馈解码；使用FFT、ASPP、卷积注意力、条件缩放等技术。

**📊 数据集**

主要数据集：PCOD-1200（极化COD），RGBP-Glass（透明物体检测），以及CAMO、COD10K、NC4K、CHAMELEON等非极化COD基准。

**📈 对比分析**

与20+最新单模与多模COD方法对比，CPGNet在PCOD-1200上S_α、F_β^w、E_ϕ和M均获得最佳或接近最佳成绩；在RGBP-Glass和多种COD基准上亦表现出色，参数量与性能兼优。

**⚠️ 局限性**

局限性包括对极化测量质量高度依赖；在极化信号弱或噪声大时，导引信息可能失效；对纯RGB场景的适应性不如专门的RGB方法。

---

## 519. Tracking Equivalent Mechanistic Interpretations Across Neural Networks

**arXiv ID:** 2603.30002 | [PDF](https://arxiv.org/pdf/2603.30002v1)

**作者:** Alan Sun `[一作]` (Carnegie Mellon University), Mariya Toneva `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 1076 | [OpenAlex ID](https://openalex.org/A5072973927)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

定义并研究“解释等价性”问题，提出一种基于表示相似度的算法来判断两个模型是否实现相同的高层算法而无需显式解释；

**💡 创新点**

创新点在于把解释等价性定义为实现等价性的约束，构建理论上必要且充分的表示相似度指标，并证明其与解释、电路、表示之间的关系；

**🔧 技术方法**

核心技术包括因果抽象（Causal Abstraction）、表示抽象（Linear Representation Similarity）、实现集合的 Hausdorff 距离、Bootstrap 置信区间估计和线性回归近似；

**📊 数据集**

实验数据集：自定义 10-Permutation 检测任务（synthetic 1000 样本），预训练的 GPT‑2 系列（small/medium/large）与 Pythia 系列（160M–2.9B）以及 C4 语料库用于 next‑token 预测；

**📈 对比分析**

比较方法：使用算法 Alg:congruence 计算表示相似度 p，进而得到“等价性”统计；在 toy 任务中同一解释组内 p ≈ 1，跨组 p ≈ 0；在 IOI 任务中同规模 Pythia 与 GPT‑2 的 p 在同一组内显著高于跨组；在 next‑token 任务中对 POS 相关 token 的 p 明显高于随机 token；整体表现表明该指标能有效区分不同解释；

**⚠️ 局限性**

局限性：需要大量实现样本且假设实现集合可通过因果干预获得；表示相似度采用线性近似，可能对非线性表示失效；解释等价性定义依赖于抽象层级选择，若压缩率过高可能导致无法区分；算法在大规模模型上计算成本仍较高；

---

## 520. Phyelds: A Pythonic Framework for Aggregate Computing

**arXiv ID:** 2603.29999 | [PDF](https://arxiv.org/pdf/2603.29999v1)

**作者:** Gianluca Aguzzi `[一作]` (University of Bologna), Mirko Viroli `[通讯]` (University of Bologna)

**通讯引用:** 6798 | [OpenAlex ID](https://openalex.org/A5014225579)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

实现了一个面向Python的聚合编程框架Phyelds，提供了字段演算的完整实现、Pythonic API、可插拔的模拟器以及与机器学习库和第三方仿真器（如VMAS）的无缝集成。

**💡 创新点**

主要创新点在于：①把功能型的聚合编程模型转化为面向对象、命令式的Python接口，降低Python开发者的学习门槛；②设计了轻量级、模块化的架构，支持从小规模实验到真实设备部署；③提供了对TensorFlow、PyTorch、多智能体强化学习环境等生态系统的绑定，促进聚合编程与机器学习的融合。

**🔧 技术方法**

使用了字段演算（Field Calculus）核心运算（rep、nbr、foldhood等）、Python AST变换实现聚合函数装饰器、仿真器模块（虚拟机、节点、事件队列）以及对外部仿真器的适配层；同时集成了PyTorch/NumPy等数值库。

**📊 数据集**

实验中主要使用合成数据与仿真场景：聚合模式示例、虚拟网络（网格、扭曲格子）以及Vicsek群体运动仿真。未涉及公开真实数据集。

**📈 对比分析**

文中没有给出定量性能基准；通过示例说明功能完整性和可扩展性。未来工作计划与现有框架进行系统基准比较。

**⚠️ 局限性**

局限性包括：①尚未进行大规模性能评估与内存/CPU开销分析；②缺乏针对真实分布式部署（如边缘设备、机器人集群）的实测；③依赖Python解释器，可能在实时性要求高的场景下受限。

---

## 521. Extending MONA in Camera Dropbox: Reproduction, Learned Approval, and Design Implications for Reward-Hacking Mitigation

**arXiv ID:** 2603.29993 | [PDF](https://arxiv.org/pdf/2603.29993v1)

**作者:** Nathan Heath `[一作]` `[通讯]` (Independent Researcher), Nathan Heath (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

复现 MONA 的 Camera Dropbox 环境，重新包装为可 pip 安装的 Python 项目，并实现脚本化 PPO 训练；在同一框架下加入可插拔的五种学习式审批机制，对不同审批构造对安全性与能力的影响进行实验；

**💡 创新点**

通过将审批构造空间拆分为可切换的五种模式（oracle、噪声、误设、学习式、校准学习式），形成可直接运行的实验对象，首次系统化探测审批质量对 MONA 安全保障的影响；

**🔧 技术方法**

使用 CNN 观测提取器与向量化环境的 PPO 训练；学习审批采用监督分类器（逻辑回归）预测期望与黑客行为概率，计算差值作为奖励覆盖；对比原始 MONA 代码；

**📊 数据集**

Camera Dropbox 4×4 网格环境（公开的奖励序列与参考数组）作为测试基准；学习审批的数据集来源于 oracle 轨迹，尺寸 512 或 2048；

**📈 对比分析**

与原始 MONA 对比：普通 RL 91.5% 作弊率、oracle MONA 0%；在学习审批下零作弊率但意图行为率仅 11.9%（oracle 为 99.9%），表明学习审批能阻止作弊但导致能力下降；

**⚠️ 局限性**

实验仅使用 768–3072 PPO 步，未达到原论文约 10^6 步的规模；仅评估 Camera Dropbox，未覆盖其他模型；学习监督模型过于简化，缺乏对抗性或分布漂移测试；大多为单次种子实验，未给出多种种子统计；与原始实现存在细微差异（CNN、奖励归一化等）。

---

## 522. Quantifying Cross-Modal Interactions in Multimodal Glioma Survival Prediction via InterSHAP: Evidence for Additive Signal Integration

**arXiv ID:** 2603.29977 | [PDF](https://arxiv.org/pdf/2603.29977v1)

**作者:** Iain Swift `[一作]` (Munster Technological University), Ruairi O'Reilly `[通讯]` (Munster Technological University)

**通讯引用:** 396 | [OpenAlex ID](https://openalex.org/A5088297608)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了多模态脑胶质瘤生存预测模型中跨模态交互作用的量化，并将InterSHAP从分类迁移到Cox比例风险模型。

**💡 创新点**

通过InterSHAP量化多模态模型的交互作用，揭示了性能提升与交互作用呈负相关，即更好模型往往交互作用更低。

**🔧 技术方法**

采用Cox比例风险模型、Shapley交互指数（InterSHAP）、深度学习特征提取以及四种多模态融合架构（早期融合、交叉注意力、双线性融合、门控融合）。

**📊 数据集**

使用TCGA-GBM与TCGA-LGG联合数据集，共575例匹配的全切片图像（WSI）和RNA‑seq表达谱。

**📈 对比分析**

在相同的训练/测试划分下比较四种融合架构的C-index和InterSHAP值；最佳双线性融合取得C-index 0.819、交互作用约3.7%，优于早期融合的C-index 0.636和交互作用4.8%。

**⚠️ 局限性**

数据量有限且未做外部验证；InterSHAP仅作用于高维嵌入，低层交互可能被编码器线性化；缺少IDH等分子分层；M>3时计算成本高，难以扩展。

---

## 523. Benchmarking PhD-Level Coding in 3D Geometric Computer Vision

**arXiv ID:** 2603.30038 | [PDF](https://arxiv.org/pdf/2603.30038v1)

**作者:** Wenyi Li `[一作]` (Tsinghua University), Hao Zhao `[通讯]` (Tsinghua University)

**通讯引用:** 15798 | [OpenAlex ID](https://openalex.org/A5071321132)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 GeoCodeBench，一个基于可执行单元测试的 3D 几何视觉代码生成基准，评估 LLM 在科研级别代码实现的能力。

**💡 创新点**

首创将 3D 视觉论文中的核心函数提取为填空任务，并提供两级任务层次（General 3D 与 Research）以及长文本上下文对模型表现的系统分析。

**🔧 技术方法**

利用 OCR（MinerU）将 PDF 论文转换为结构化 JSON，使用 Cursor 自动生成候选函数和单元测试，构建统一的代码填空模板，并对生成代码执行单元测试进行评估。

**📊 数据集**

取自 2025 年顶级会议（CVPR/ICCV/ICLR）的论文及其公开代码仓库，共构建 100 个填空实现任务。

**📈 对比分析**

对八款开放/闭源 LLM 进行 pass@k 单元测试评估，最高 GPT‑5 获得 36.6% 的通过率；研究级任务更难，长文本输入不一定提升性能。

**⚠️ 局限性**

仍存在显著的功能错误、对长上下文理解有限、对论文结构噪声敏感，以及仅基于公开代码仓库导致的潜在数据泄露与覆盖范围受限等局限。

---

## 524. Hybrid Framework for Robotic Manipulation: Integrating Reinforcement Learning and Large Language Models

**arXiv ID:** 2603.30022 | [PDF](https://arxiv.org/pdf/2603.30022v1)

**作者:** Md Saad `[一作]` (Jamia Millia Islamia), Mohd Suhaib `[通讯]` (Jamia Millia Islamia)

**通讯引用:** 1561 | [OpenAlex ID](https://openalex.org/A5020433747)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一个融合强化学习和大型语言模型的混合框架，用于改进机器人操作任务；

**💡 创新点**

创新点在于将LLM用于高层任务规划与分解，RL用于低层精确控制，并通过动态反馈循环实现实时任务适应；

**🔧 技术方法**

使用了PPO和SAC两种强化学习算法、GPT类大型语言模型、PyBullet仿真环境和Franka Emika Panda机械臂；

**📊 数据集**

主要使用仿真数据，未使用公开真实数据集；

**📈 对比分析**

通过与单独使用RL的系统比较，LLM+RL在任务完成时间降低33.5%，准确率提升18.1%，适应性提升36.4%；

**⚠️ 局限性**

局限在于仅在仿真环境验证，缺乏现实世界转移、规模化多机器人部署和复杂场景的进一步评估。

---

## 525. Scalable AI-assisted Workflow Management for Detector Design Optimization Using Distributed Computing

**arXiv ID:** 2603.30014 | [PDF](https://arxiv.org/pdf/2603.30014v1)

**作者:** Derek Anderson `[一作]` (Thomas Jefferson National Accelerator Facility), Torre Wenaus `[通讯]` (Brookhaven National Laboratory)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5105888542)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文开发了AID(2)E框架，将多目标贝叶斯优化与PanDA/iDDS分布式工作流集成，用于电子–离子对撞机（EIC）实验中的探测器设计与优化。

**💡 创新点**

创新点包括：① Function-as-a-Task装饰器将本地AI函数透明地转化为PanDA作业，实现分布式执行；② 三跑步器（本地、SLURM、PanDA）设计，使工作流可无缝迁移至多种计算环境；③ 在真实探测器（dRICH）模拟中验证了多目标优化与分布式计算的协同效能。

**🔧 技术方法**

采用了PanDA作业管理、iDDS工作流编排、Ax/BoTorch（MOBO）及MOGO优化器、SLURM调度器、Docker/Conda容器化环境，以及STOMP/REST通信协议。

**📊 数据集**

使用的“数据集”包括：1）DTLZ2多目标基准问题（已知Pareto前沿）用于闭环测试；2）ePIC实验的dRICH探测器仿真与重建任务（包含多参数几何与光学配置）。

**📈 对比分析**

通过闭环测试1与闭环测试2比较，分别验证优化收敛与分布式执行性能。实验表明，分布式PanDA/iDDS工作流与本地执行在超体积（hypervolume）收敛率上相近，但并发度显著提升；总运行时间受优化器生成试验的开销主导，仿真评估时间占比较小。

**⚠️ 局限性**

局限性在于：①试验生成与作业调度产生的延迟与资源争用会影响整体吞吐；②目前仅在较小规模的ML模型与探测器子系统上验证，尚未评估更大模型或完整探测器的可扩展性；③需要进一步优化跨站点通信与结果聚合的效率。

---

## 526. OmniRoam: World Wandering via Long-Horizon Panoramic Video Generation

**arXiv ID:** 2603.30045 | [PDF](https://arxiv.org/pdf/2603.30045v1)

**作者:** Yuheng Liu `[一作]` (University of California Irvine), Yiwei Hu `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于全景视频的可控长时段场景漫游框架 OmniRoam，采用预览‑细化两阶段流程实现高质量、长周期全景视频合成。

**💡 创新点**

创新点在于将全景表示与全局‑局部预览‑细化设计相结合，分离视角流与尺度控制，构造“循环一致性”评估指标，并在此基础上构建大规模全景轨迹数据集。

**🔧 技术方法**

核心技术包括基于扩散变压器的轨迹条件全景视频生成、分段细化的尺度对齐与可见性掩码、以及实时预览的自监督蒸馏。

**📊 数据集**

使用了由 2000 条真实手持全景视频和 1000 条从 3D Gaussian Splatting 场景渲染得到的合成全景视频，涵盖多样场景与精确摄像机轨迹。

**📈 对比分析**

与 Matrix-3D、Imagine360 等基线对比，OmniRoam 在 480p/720p 视觉质量（FAED、SSIM、LPIPS）、轨迹可控性（PSNR）和循环一致性等指标上均显著优于对手，尤其在长时段 641 帧生成中表现出更强的全局一致性。

**⚠️ 局限性**

局限性包括对全景输入和预先定义轨迹的依赖、生成过程仍较慢（尤其是高分辨率细化阶段），以及在极端复杂场景或极长轨迹下可能出现细节模糊或微小漂移。

---

## 527. A Precision Emulation Approach to the GPU Acceleration of Ab Initio Electronic Structure Calculations

**arXiv ID:** 2603.29975 | [PDF](https://arxiv.org/pdf/2603.29975v1)

**作者:** Hang Liu `[一作]` (University of Texas at Austin), Yang Wang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 140138 | [OpenAlex ID](https://openalex.org/A5100381753)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用 INT8 Tensor Core 通过 Ozaki 方案对 FP64 GEMM 进行可调精度仿真，并结合自动 BLAS offload 技术在 GPU 上加速 LSMS DFT 代码，同时保持原始算法不变；

**💡 创新点**

提出一种无需改动原始代码即可实现 FP64 计算的低精度 INT8 仿真方法，利用可调精度和 cache‑coherent 内存策略，既提升硬件利用率，又保证数值稳定性；

**🔧 技术方法**

INT8 Tensor Core GEMM 仿真（Ozaki‑I、Ozaki‑II）、动态链接钩子、DBI 自动 offload 工具、CUDA 13、cache‑coherent Unified Memory Architecture、MuST/LSMS 软件栈；

**📊 数据集**

FeNi3 L12 合金的 MuST LSMS 基准测试，包含 30 点高斯积分能量点；

**📈 对比分析**

对比 native FP64 GPU offload，评估 G(z) 最大百分误差、总能量、磁矩、净电荷等物理量的收敛性；在 55bits/16mods 模式下误差 ≤10⁻¹⁰，速度提升约 1.7×；低精度 31bits/10mods 模式误差达 10⁻²；高精度 63bits/18mods 模式提升有限；

**⚠️ 局限性**

仅在 NVIDIA GB200 等具备 Tensor Core 的 GPU 上测试，需手动调节模数/ mantissa 位；某些低精度模式无法收敛；对其它 CPU 代码的迁移性与性能提升未知；依赖 cache‑coherent CPU–GPU 系统；需进一步评估在更大规模或不同应用场景中的可行性。

---

## 528. HapCompass: A Rotational Haptic Device for Contact-Rich Robotic Teleoperation

**arXiv ID:** 2603.30042 | [PDF](https://arxiv.org/pdf/2603.30042v1)

**作者:** Xiangshan Tan `[一作]` (Toyota Technological Institute at Chicago), Matthew R. Walter `[通讯]` (Toyota Technological Institute at Chicago)

**通讯引用:** 5143 | [OpenAlex ID](https://openalex.org/A5103153703)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

开发并评估了一种低成本可穿戴触觉装置，利用单一线性谐振致动器机械旋转实现二维方向性触觉反馈，并将其集成到基于视觉的机器人远程操控系统中，验证其在接触丰富任务中的效果。

**💡 创新点**

创新点在于通过旋转单一 LRA 产生二维方向感知，避免了多振动阵列的感知干扰；将该装置与视觉+手部跟踪相结合，实现了实时的方向性触觉输出，并通过实验证明可显著提升远程操控成功率和模仿学习性能。

**🔧 技术方法**

技术包括 Meta Quest 3 头显手部跟踪、视觉穿透反馈、三指夹持触觉传感器、腕部力矩计、线性谐振致动器与伺服电机驱动、三维接触力到二维方向的映射算法、实时控制与反馈架构以及数据采集与模仿学习模型。

**📊 数据集**

使用自制任务数据集：Key Insertion、USB 插入、Spaghetti 探测，并在模仿学习评估中收集10条演示数据；未使用公开数据集。

**📈 对比分析**

与三种基线（仅视觉、视觉+非方向触觉、控制器非方向触觉）进行对照实验。结果显示，在 Key Insertion 任务中，方向性触觉将成功率从 63.2% 提升至 100%，USB 插入任务保持 98.3% 的高成功率并显著降低最大接触力；Spaghetti 探测成功率提升至 60%。模仿学习演示从 60% 提升至 90%，且峰值力和扭矩均下降。

**⚠️ 局限性**

局限性包括：只能提供二维方向反馈，无法处理三维方向；系统仍存在一定的端到端延迟，未量化其影响；实验任务有限，难以验证方案在更广泛的操控场景中的泛化能力。

---

## 529. The Triadic Cognitive Architecture: Bounding Autonomous Action via Spatio-Temporal and Epistemic Friction

**arXiv ID:** 2603.30031 | [PDF](https://arxiv.org/pdf/2603.30031v1)

**作者:** Davide Di Gioia `[一作]` `[通讯]`, Davide Di Gioia

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了三元认知架构（Triadic Cognitive Architecture, TCA），将人工智能推理过程建模为受空间拓扑、时间延迟和认知不确定性共同制约的连续时间物理轨迹，并在此框架下实现了基于价值信息（VOI）和哈密顿-雅可比-贝尔曼（HJB）最优停止的代理决策控制。

**💡 创新点**

创新点包括：① 将空间拓扑（网络拥塞）与时间延迟和信息增益三者统一为“认知摩擦”并在优化目标中显式计价；② 引入连续时间滤波与最优停止理论，给出停止边界的形式化定义；③ 在实际代理中实现可计算的离散 VOI 近似与基于 roll‑out 的净效用停止规则。

**🔧 技术方法**

使用的技术包括：非线性滤波理论（Kushner–Stratonovich 过滤）、黎曼几何路由、哈密顿-雅可比-贝尔曼方程、蒙特卡洛 roll‑out 估计 VOI、基于 HJB 的最优停止边界与净效用停止条件。

**📊 数据集**

使用的数据集为自构造的“Emergency Medical Diagnostic Grid (EMDG)”模拟环境，包含多种医疗诊断工具（血液检测、MRI、病历查询等）及其不同的空间/时间成本。

**📈 对比分析**

与传统 ReAct（贪婪）代理对比，TCA 在 EMDG 环境中在相同诊断准确率下显著缩短了推理时间（约 92%），提高了患者生存率（从 57.3% 提升至 93.1%），且信息增益下降幅度更小，体现了更优的成本-收益平衡。

**⚠️ 局限性**

局限性包括：① 离散实现采用零延迟（η=0）的 myopic 停止策略，未利用完整的 HJB 价值函数；② VOI 估计依赖蒙特卡洛 roll‑out，计算成本在工具集合大时显著；③ 仅在单一仿真环境验证，缺乏跨域、真实世界实验的验证。

---

## 530. Can Commercial LLMs Be Parliamentary Political Companions? Comparing LLM Reasoning Against Romanian Legislative Expuneri de Motive

**arXiv ID:** 2603.30028 | [PDF](https://arxiv.org/pdf/2603.30028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 531. Architecting Secure AI Agents: Perspectives on System-Level Defenses Against Indirect Prompt Injection Attacks

**arXiv ID:** 2603.30016 | [PDF](https://arxiv.org/pdf/2603.30016v1)

**作者:** Chong Xiang `[一作]` (NVIDIA), G. Edward Suh `[通讯]` (NVIDIA)

**通讯引用:** 11930 | [OpenAlex ID](https://openalex.org/A5024329178)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向大型语言模型驱动 AI 代理的系统级防御框架，阐述了动态重规划与安全策略更新、模型辅助安全决策以及人机交互三大核心设计原则。

**💡 创新点**

创新点在于将安全设计抽象为计划/策略生成、审批、执行与策略执行四个模块，并强调在受限输入范围内使用 LLM 进行安全判断，从而兼顾可解释性、可扩展性与深度防御；同时对现有 benchmark 的不足进行了系统化批判。

**🔧 技术方法**

采用基于 LLM 的规划器、策略审批器、执行器和策略执行器，结合规则引擎或受限 LLM 判断，并通过反馈循环实现动态策略与计划更新；系统接口通过结构化表示（如 JSON、DSL）限制 LLM 的观测与决策空间。

**📊 数据集**

未使用专门的数据集，主要基于现有的 AgentDojo 等 benchmark 与模拟的动态环境进行讨论与评估；强调需要构建更具挑战性的自适应攻击实验场景。

**📈 对比分析**

通过对现有 benchmark 的分析指出其缺乏动态任务、可适应攻击和优化的 payload，提出需引入自适应攻击者与多步骤任务来评估防御效果；在理论上，该框架可提升安全性和可解释性，但缺乏公开实验数据。

**⚠️ 局限性**

局限在于仍需依赖 LLM 与人类进行关键安全决策，无法完全消除攻击面；系统实现复杂度高，易受实现细节影响；Benchmark 设计不足导致评估可能产生误导，缺乏对真实环境的充分验证。

---

## 532. SurgNavAR: An Augmented Reality Surgical Navigation Framework for Optical See-Through Head Mounted Displays

**arXiv ID:** 2603.29990 | [PDF](https://arxiv.org/pdf/2603.29990v1)

**作者:** Abdullah Thabit `[一作]` (Erasmus MC), Theo van Walsum `[通讯]` (Erasmus MC)

**通讯引用:** 6307 | [OpenAlex ID](https://openalex.org/A5012856150)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出并实现了一个开放源代码、设备无关的HMD增强现实手术导航框架，并在HoloLens 2和Magic Leap 2设备上通过phantom实验进行了验证。

**💡 创新点**

创新点在于将标记跟踪、工具校准、图像对患者对齐与可视化等核心模块集成于一个可配置、通用的框架，并通过多种校准方法和多标记系统实现了端到端的独立导航。

**🔧 技术方法**

采用的技术包括AR HMD（HL2/ML2）、基于Vuforia、ArUco/AprilTag的2D标记跟踪、PnP姿态估计、工具尖端的pivot、校准工具和标记间校准、点对点匹配与手动定位，以及Unity与MRTK3构建交互界面。

**📊 数据集**

实验使用CT扫描生成的头部与胸腔phantom模型（含多点标记），并通过3D打印标记板及NDI Vega光学追踪系统获取基准数据。

**📈 对比分析**

通过与NDI系统的对比评估，工具尖端校准误差约为1 mm，图像对患者对齐误差为2–3 mm，针尖定位与肋骨裂纹定位的目标误差均低于5 mm，表明系统在phantom实验中的性能与现有HMD导航系统相当。

**⚠️ 局限性**

局限性包括仅在phantom实验中验证，未涉及临床真实手术环境；对软组织变形和呼吸运动的补偿有限；依赖标记放置且需要一定用户训练，且精度无法满足亚毫米级手术需求。

---

## 533. Automatic Identification of Parallelizable Loops Using Transformer-Based Source Code Representations

**arXiv ID:** 2603.30040 | [PDF](https://arxiv.org/pdf/2603.30040v1)

**作者:** Izavan dos S. Correia `[一作]` (Federal Rural University of Pernambuco), Tiago A. E. Ferreira `[通讯]` (Federal Rural University of Pernambuco)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建平衡的循环样本数据集，并利用DistilBERT对源代码中的循环进行并行化潜能分类。

**💡 创新点**

直接使用预训练Transformer对原始代码文本进行语义表示，省去繁琐的预处理与特征工程，且采用轻量化DistilBERT实现高效且准确的分类。

**🔧 技术方法**

DistilBERT Transformer、子词分词、10折交叉验证、AdamW优化、二元交叉熵损失、早停、线性学习率调度。

**📊 数据集**

8,340条循环实例（4,140可并行 + 4,140不可并行），其中4,000条合成循环 + 170条真实GitHub代码；数据均衡后用于训练与评估。

**📈 对比分析**

与以手工token化+CNN/DNN为基线的传统方法对比；平均准确率99.60%，精度99.71%，召回99.49%，F1≈99.60%，误报率(FPR)≈0.29%，表现出极高的准确性与低误报。

**⚠️ 局限性**

仅聚焦循环层级，未考虑融合、展开、分块等高级循环变换；数据集规模有限，缺乏多语言和更复杂代码结构的覆盖。

---

## 534. Covertly improving intelligibility with data-driven adaptations of speech timing

**arXiv ID:** 2603.30032 | [PDF](https://arxiv.org/pdf/2603.30032v1)

**作者:** Paige Tuttösí `[一作]` (Simon Fraser University), Jean-Julien Aucouturier `[通讯]` (Université Marie et Louis Pasteur)

**通讯引用:** 2988 | [OpenAlex ID](https://openalex.org/A5028233850)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究并实现了一种基于逆相关的机器生成语音速率调控方法，用以提升非母语听者对英语元音对的识别率。

**💡 创新点**

创新点在于揭示并利用“scissor-shaped”时间语速影响模式，将其嵌入TTS系统，实现针对性速率调整，显著提高L2可懂度。

**🔧 技术方法**

采用逆相关实验、时间拉伸与音高变换、CoquiXTTS/Matcha-TTS语音合成以及Whisper ASR等技术。

**📊 数据集**

数据集包括合成的“ I heard them say XXX ”/法语句子、不同语言L2听众（法语、中文、日语）以及由Prolific招募的人类参与者；机器评测使用Whisper。

**📈 对比分析**

通过与基线、全速降低、每词拉伸等TTS策略对比，测量词错误率（WER）和主观可懂度（MOS），结果显示提出模型显著降低L2 WER，但人类主观评价仍倾向于全速降低。

**⚠️ 局限性**

局限性：实验仅覆盖英语/法语/中文/日语L2听众；机器ASR不匹配人类对速率的感知；存在合成音质问题；未考察语调、情感等其他因素。

---

## 535. Performative Scenario Optimization

**arXiv ID:** 2603.29982 | [PDF](https://arxiv.org/pdf/2603.29982v1)

**作者:** Quanyan Zhu `[一作]` (New York University), Zhengye Han `[通讯]` (New York University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种用于决策依赖的机会约束问题的表现性场景优化框架，考虑了决策如何主动影响数据生成过程的反馈循环。

**💡 创新点**

创新点在于引入了表现性解决方案的概念，并利用Kakutani不动点定理证明了其存在性，同时提出了一种无模型的场景基础近似算法。

**🔧 技术方法**

使用了无模型的场景基础近似算法和随机固定点迭代方法，结合对样本大小的对数调度。

**📊 数据集**

通过模拟API交互生成的i.i.d.样本（如提示嵌入和意图标签）来构建数据集，特别是在大型语言模型（LLM）安全性应用中。

**📈 对比分析**

与现有的表现性优化方法相比，本文的方法在处理未知诱导分布的复杂性上更具可计算性，数值结果表明分类器和对抗性提示分布的共同演化达到了稳定的均衡。

**⚠️ 局限性**

限制在于该方法依赖于样本的质量和数量，且在面对高度动态和异质的对手时，可能仍然存在模型失配的风险。

---

## 536. Meteorology-Driven GPT4AP: A Multi-Task Forecasting LLM for Atmospheric Air Pollution in Data-Scarce Settings

**arXiv ID:** 2603.29974 | [PDF](https://arxiv.org/pdf/2603.29974v1)

**作者:** Prasanjit Dey `[一作]`, Bianca Schoen-Phelan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于GPT‑4的多任务空气污染预测模型 Meteorology‑Driven GPT4AP，融合气象信息对空气质量进行短期预测。

**💡 创新点**

创新点在于将大型语言模型与气象数据结合，利用少量标注数据实现多任务学习，并通过少样本微调提升数据稀缺地区的预测能力。

**🔧 技术方法**

使用 GPT‑4 架构，结合提示工程、气象特征嵌入、少样本微调与多任务学习技术。

**📊 数据集**

采用美国 EPA 及中国 AQICN 的空气质量数据，配合 NASA 与 NOAA 的气象观测数据作为训练集。

**📈 对比分析**

与物理模型 CMAQ、XGBoost、LSTM 等基线进行对比，平均 RMSE 降低约 15%，在 48 小时预测窗口内 MAE 提升 10%。

**⚠️ 局限性**

主要局限包括对商业 API 的依赖、计算成本高、模型可解释性差，以及在极端气象事件下预测误差仍显著。

---

## 537. Learning Structural-Functional Brain Representations through Multi-Scale Adaptive Graph Attention for Cognitive Insight

**arXiv ID:** 2603.29967 | [PDF](https://arxiv.org/pdf/2603.29967v1)

**作者:** Badhan Mazumder `[一作]` (Georgia State University), Dong Hye Ye `[通讯]` (Georgia State University)

**通讯引用:** 7690 | [OpenAlex ID](https://openalex.org/A5068927047)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了MAGNet——一种融合结构和功能连接的Transformer风格图神经网络，用于预测认知能力

**💡 创新点**

创新点在于构建多尺度混合脑图（包括直接、交叉、间接路径），采用局部边缘感知注意力与全局自注意力相结合，并通过联合损失同时优化结构功能一致性与预测任务

**🔧 技术方法**

技术包括源基形态测量（SBM）、功能网络连通性（FNC）提取、k‑NN稀疏化、多尺度间接连接（MDC）、Transformer式图注意力、联合MSE+结构功能一致性损失、Adam优化器

**📊 数据集**

使用ABCD（Adolescent Brain Cognitive Development）基线数据集，涵盖7656名9-10岁儿童的sMRI、rs‑fMRI及流体、晶体和总智力测得结果

**📈 对比分析**

与GAT、GT、SFDN、SFIN、Joint GCN、BrainNN、GCNN、Joint DCCA等SOTA GNN方法比较，MAGNet在三种智力得分上均取得最低MSE、MAE和最高相关系数（如总智力相关0.38，显著优于SFDN的0.35）

**⚠️ 局限性**

局限在于仅使用静态FNC，缺乏动态时间信息；仅验证于ABCD样本，未测试跨文化或临床人群；模型复杂度高，训练耗时长

---

## 538. Enhancing Structural Mapping with LLM-derived Abstractions for Analogical Reasoning in Narratives

**arXiv ID:** 2603.29997 | [PDF](https://arxiv.org/pdf/2603.29997v1)

**作者:** Mohammadhossein Khojasteh `[一作]` (Vrije Universiteit Amsterdam), Filip Ilievski `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 5600 | [OpenAlex ID](https://openalex.org/A5008608420)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套利用大型语言模型（LLM）进行故事分解、抽象化以及结构映射的神经符号框架，以实现叙事文本的类比推理。

**💡 创新点**

创新点在于：①将LLM用于自动抽象多层级（概念、评估、叙事弧、阶段）抽象；②将这些抽象与结构映射（SME/FAME类算法）相结合，形成可调节的模块化流程；③系统性评估不同抽象层级和映射策略在叙事类比任务上的贡献。

**🔧 技术方法**

技术包括：多轮少量示例提示的LLM（Qwen3‑8B、Llama‑3.1‑8B）进行事件提取和多层抽象；使用Sentence‑Transformers（all‑MiniLM‑L6‑v2）生成向量并计算余弦相似度；基于贪心算法的结构映射和局部/全局得分组合；CoT和零样本提示进行基线比较。

**📊 数据集**

使用两大叙事类比基准：StoryAnalogy‑MCQ（360道多选题，4个候选）和Analogical Reasoning on Narratives（ARN，1096对故事，分为近/远、正/负四种模式）。

**📈 对比分析**

与仅使用LLM提示（零样本或CoT）对齐或高层信息推断的基线相比，加入抽象后结构映射在远类比上明显提升（例如Qwen在ARN远类比从0.52提升至0.67，Llama从0.42提升至0.46）。在近类比上，LLM提示略优，但整体性能仍能达到或超过随机水平；在MCQ中，使用概念+评估抽象的SM获得0.46（Qwen）/0.45（Llama），高于LLM基线的0.41/0.28。

**⚠️ 局限性**

局限包括：①抽象质量受LLM误解、缺失因果关系的影响；②结构映射依赖余弦相似度，易受表面词汇重叠干扰；③对不同类比模式（特别是ARN的三种模式）无法完全捕捉；④基准数据集可能过度依赖句法或表面相似度，缺乏细粒度的关系标注；⑤多层抽象在不同任务中效果不一致，需进一步优化。

---

## 539. Video Models Reason Early: Exploiting Plan Commitment for Maze Solving

**arXiv ID:** 2603.30043 | [PDF](https://arxiv.org/pdf/2603.30043v1)

**作者:** Kaleb Newman `[一作]` (Princeton University), Olga Russakovsky `[通讯]` (Princeton University)

**通讯引用:** 45268 | [OpenAlex ID](https://openalex.org/A5022811687)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究视频扩散模型在生成过程中如何进行早期计划承诺，并基于此提出高效采样和链式推理方法。

**💡 创新点**

发现视频模型在前几步就决定运动轨迹，并提出 Early Planning Beam Search 与 Chaining with Early Planning，显著提升长程迷宫求解能力。

**🔧 技术方法**

使用流匹配扩散模型、早期轨迹提取、轻量级轨迹验证器、Beam Search 与多生成链式推理技术。

**📊 数据集**

在 Frozen Lake 与 VR‑Bench 迷宫数据集上进行实验，涵盖不同尺寸、障碍密度与目标位置。

**📈 对比分析**

与最佳‑N 采样对比，EPBS 在相同计算量下提升约10%，Chaining 将长路径成功率从 7% 提升至 67%，整体准确率提升约 2.5×。

**⚠️ 局限性**

限制在生成窗口长度（≈12 步）之外的规划无法完成，且不同模型对结构约束的遵循存在差异，链式方法受多链错误累积影响。

---

## 540. Aligned, Orthogonal or In-conflict: When can we safely optimize Chain-of-Thought?

**arXiv ID:** 2603.30036 | [PDF](https://arxiv.org/pdf/2603.30036v1)

**作者:** Max Kaufmann `[一作]` (Google DeepMind), and Rohin Shah `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了一个框架，用于预测强化学习环境中奖励结构是否会导致链式推理（CoT）的可监控性下降。

**💡 创新点**

框架将奖励对CoT文本和CoT计算的影响分为“对齐（aligned）”“正交（orthogonal）”“冲突（in-conflict）”三类，并给出对应的可监控性预测；同时展示冲突奖励更难优化。

**🔧 技术方法**

利用强化学习（GRPO、CAPO、ICRL）在大型语言模型上进行训练，评估CoT监控器的性能，并用文本透明度与监控任务的匹配度来衡量可监控性。

**📊 数据集**

在两种简化环境下实验：1）硬币状态跟踪任务；2）修改版的MBPP（简单编码任务）以及带回门（backdoor）子任务；使用人工构造的奖励对和公开的提示。

**📈 对比分析**

通过先验分类与实验结果对照，发现冲突奖励导致监控器准确率几乎为0，而正交和对齐奖励保持高准确率；冲突奖励在训练中更难收敛，成功率明显低于其他类别。

**⚠️ 局限性**

局限包括：环境过于简单，未覆盖真实大模型后训练场景；可监控性下降是否出现取决于模型是否达到足够高的奖励；缺乏自动化方法识别奖励类别；可能存在语义漂移导致预测失效。

---

## 541. A Lightweight Hybrid Publish/Subscribe Event Fabric for IPC and Modular Distributed Systems

**arXiv ID:** 2603.30030 | [PDF](https://arxiv.org/pdf/2603.30030v1)

**作者:** Dimitris Gkoulis `[一作]` (Harokopio University of Athens), Dimitris Gkoulis `[通讯]` (Harokopio University of Athens)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5080656437)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了轻量级本地优先的混合事件系统CNS，支持在模块化分布式系统中进行本地IPC与跨节点分发。

**💡 创新点**

创新点在于统一的类型化事件键、基于事件族的序列化/校验注册以及显式的本地↔分布式桥接机制。

**🔧 技术方法**

技术使用了Python、NATS消息代理、本地发布/订阅框架及自定义事件键与SerDe注册。

**📊 数据集**

使用了实验中自构造的256B、1KiB、4KiB三种payload，未引用公开数据集。

**📈 对比分析**

通过单机基准测评，local-only平均30µs，distributed 1.26–1.37ms，hybrid 1.64–1.89ms；吞吐量从约34k到1.2k msg/s；验证与序列化影响小。

**⚠️ 局限性**

局限在于桥接吞吐、尾部延迟、强制停止时的消息丢失、缺乏多机评估及语言中立序列化。

---

## 542. Aligning Validation with Deployment: Target-Weighted Cross-Validation for Spatial Prediction

**arXiv ID:** 2603.29981 | [PDF](https://arxiv.org/pdf/2603.29981v1)

**作者:** Alexander Brenning `[一作]` (Friedrich Schiller University Jena), Thomas Suesse `[通讯]` (Friedrich Schiller University Jena)

**通讯引用:** 1118 | [OpenAlex ID](https://openalex.org/A5050546697)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于校准加权的目标加权交叉验证（Target‑Weighted CV，TWCV），用于在空间预测等结构化任务中估计部署风险；

**💡 创新点**

创新点在于将任务生成与风险估计分离，利用校准加权重新分配验证损失，使验证任务分布匹配目标部署任务分布，从而同时校正协变量偏移和任务难度偏移；

**🔧 技术方法**

技术主要包括任务描述符离散化、校准加权（raking）求权重、缓冲留一留出（buffered LOO）生成多样化验证任务，以及距离加权CV（DWCV）和重要性加权CV（IWCV）作为对比；

**📊 数据集**

使用了两组数据：①基于单元正方形的模拟环境（包含两种趋势、两种相关范围及四种采样设计），②德国2018年NO₂空气质量监测网络（503站点）与全国2 km格点预测；

**📈 对比分析**

通过与传统随机、空间K折、留一留出以及最近提出的kNNDM方法对比，TWCV在模拟实验和实际案例中均显著降低了RMSE估计偏差，误差下降约15–35%，并在大多数情形下保持近乎无偏；

**⚠️ 局限性**

局限性包括需要足够覆盖目标任务分布的验证任务生成（缓冲留出），若覆盖不足权重极端或不稳定；对部署任务分布的先验假设要求严格，且在高维或极端偏移场景下仍可能出现正则化需求或样本不足问题。

---

## 543. Structural Feature Engineering for Generative Engine Optimization: How Content Structure Shapes Citation Behavior

**arXiv ID:** 2603.29979 | [PDF](https://arxiv.org/pdf/2603.29979v1)

**作者:** Junwei Yu `[一作]` (University of Tokyo), Hiroyuki Sato `[通讯]` (University of Tokyo)

**通讯引用:** 6695 | [OpenAlex ID](https://openalex.org/A5060858053)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GEO‑SFE框架，通过结构特征工程（宏观结构、信息分块、微观视觉强调）来提升大型语言模型生成引擎中的引用率

**💡 创新点**

首次系统量化并优化内容结构对LLM引用行为的影响，形成可落地的层级化结构优化原则和算法

**🔧 技术方法**

利用结构层次特征提取、梯度提升预测、规则+学习型结构重排算法、语义相似度约束以及多引擎评测指标

**📊 数据集**

在GEO‑bench收集的200篇跨域文章（生物、健康、技术、金融、旅游、科学）以及377条真实查询上进行实验

**📈 对比分析**

对比基线与结构优化版本，在六大主流生成引擎上测得平均提升17.3%引用率（p<0.001，Cohen's d=0.64）和18.5%主观质量提升，且宏观结构贡献最大

**⚠️ 局限性**

主要局限在跨平台适配的结构权重仍需人工调参，且对语义完整性的约束可能导致过度保守的结构变更

---

## 544. Trimodal Deep Learning for Glioma Survival Prediction: A Feasibility Study Integrating Histopathology, Gene Expression, and MRI

**arXiv ID:** 2603.29968 | [PDF](https://arxiv.org/pdf/2603.29968v1)

**作者:** Iain Swift `[一作]` (Munster Technological University), JingHua Ye `[通讯]` (Munster Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在同一深度学习框架中，将 FFPE H&E 病理图像、RNA 测序和 BraTS2021 FLAIR MRI 三种模态整合，用于预测成人脑胶质瘤患者的生存期。

**💡 创新点**

①首次在同一模型中加入 MRI 作为第三模态；②系统评估早期、晚期和联合融合策略；③在完全相同患者子集上对比 MRI 对预测性能的独立贡献。

**🔧 技术方法**

使用 3D ResNet‑18 对 FLAIR 进行特征提取，ResNet‑50（预训练）和 MLP 对病理图像与基因表达进行编码；采用 Cox 比例风险模型与 Composite Score (CS) 评估；实现早期融合、晚期融合、联合融合以及双线性、交叉注意和门控注意等注意力融合机制；对结果做 Bootstrap 置信区间和排列检验。

**📊 数据集**

TCGA‑GBMLGG（590 H&E 病理样本、509 RNA 样本）与 BraTS2021 FLAIR MRI 对齐，最终包含 664 名成人患者（其中 162 名拥有 MRI）。

**📈 对比分析**

通过 Composite Score (CS = (CI + (1‑IBS))/2) 对各模态组合进行比较；三模态早期融合得到 CS = 0.854，控制比较（相同 47 训练/19 测试）下 ΔCS = +0.011，p = 0.250；虽然提升方向一致，但因样本量小且 CI 置信区间宽，差异未达到统计显著性。

**⚠️ 局限性**

主要局限在于 MRI 样本极少（仅 19 名测试患者），导致 CS 估计不稳且 Bootstrap CI 过宽；未对多序列 MRI 或自监督 3D 预训练进行探索；缺乏外部验证集与标准临床基线模型的对比。

---

