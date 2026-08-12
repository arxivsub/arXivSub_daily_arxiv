# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-12 | 今日论文总数: 574

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Self-Geometry: GT-Free and Plug-and-Play Test-Time Adaptation for Geometrically Consistent 3D Vision Foundation Models

**arXiv ID:** 2608.10708 | [PDF](https://arxiv.org/pdf/2608.10708v1)

**作者:** Seokhyun Youn `[一作]`, Jihyong Oh `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种无需几何监督、可即插即用的测试时自适应方法Self-Geometry，用于提升3D视觉基础模型的深度与几何推断；

**💡 创新点**

创新点在于将极线一致性损失、光度一致性、边缘感知平滑以及基线深度一致性等多种自监督正则结合，并通过Huber稳健化、动态权重平衡和梯度解耦等技术，使模型在自适应过程中保持稳定且无需额外标注；

**🔧 技术方法**

技术实现包括LoRA轻量级微调、LightGlue特征匹配、极线几何约束、Sampson距离、SSIM+L1混合光度损失、Huber阈值自适应、Dynamic Weight Averaging、Gradient Disentanglement，以及基于视角分箱的目标视图选择与邻域采样；

**📊 数据集**

使用了四个公开基准数据集：7Scenes、ETH3D、ScanNet++ 和 HiRoom，对六种预训练视觉基础模型（VGGT、π³、DA3-Giant/Large/Base/Small）进行统一实验；

**📈 对比分析**

与传统基线（如TCO、Free-Geometry）以及未自适应的预训练模型对比，Self-Geometry在深度精度与几何一致性上均显著提升，F1分数提升可达数个百分点，且改进在所有模型与数据集上均保持一致；

**⚠️ 局限性**

局限性主要包括对外部特征匹配器（LightGlue）的依赖，导致在纹理重复或无纹理场景以及大基线视角变化时适配效果下降；以及自适应过程仍需数分钟时间，尚未达到实时应用需求。

---

## 2. A Kruskal Decision Procedure for Intuitionistic Modal Logic IK4

**arXiv ID:** 2608.10283 | [PDF](https://arxiv.org/pdf/2608.10283v1)

**作者:** Mario Piazza `[一作]` `[通讯]` (Scuola Normale Superiore), Mario Piazza (Scuola Normale Superiore)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文通过基于嵌套序列的无剪裁证明系统，证明了Simpson的直觉主义模态逻辑（含传递性扩展）的可判定性。

**💡 创新点**

创新点在于：① 将推理过程直接映射到根植同构嵌入（rooted homeomorphic embedding）上，使得仅需考虑有限的输入组合；② 结合Kruskal定理与有限支持引理，得到有效的前驱集合计算，从而构造可计算的闭包；③ 提供了统一的证据高度上界。

**🔧 技术方法**

使用的技术包括：嵌套序列（nested sequent）证明系统、根植同构嵌入（rooted homeomorphic embedding）、Kruskal定理（树的WQO）、有限支持引理、递归闭包求解算法、可判定性证明。

**📊 数据集**

无实验数据集，研究完全基于形式化证明与算法设计。

**📈 对比分析**

未进行实验比较；方法的“性能”表现为算法终止性与可计算性证明，而非数值效能评估。

**⚠️ 局限性**

局限性：证明给出了可判定性与统一高度上界，但缺乏可计算的复杂度分析；算法实现细节与实际执行时间未被探讨。

---

## 3. Geometry-aware neural causal discovery for large-scale spatiotemporal systems

**arXiv ID:** 2608.10466 | [PDF](https://arxiv.org/pdf/2608.10466v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 4. Trigger the Straggler: Load Hijack on Mixture-of-Experts LLMs

**arXiv ID:** 2608.10614 | [PDF](https://arxiv.org/pdf/2608.10614v1)

**作者:** Rui Zhang `[一作]` (University of Electronic Science and Technology of China), Guowen Xu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了一种名为Load Hijack的供应链攻击，通过仅修改Mixture-of-Experts（MoE）模型的路由器权重，在检测到私有触发器时将大量令牌集中到单个GPU上，导致该GPU成为系统级的straggler，进而显著增加推理延迟并降低吞吐量。

**💡 创新点**

创新点在于：①设计了三阶段优化流程，既能在触发输入下实现高触发器相关的专家集中，又能保持普通输入的路由近似干净基准；②揭示了路由器权重单独被篡改就能实现设备级调度控制的安全漏洞；③提出了基于运行时审计和路由器再平衡的防御方法。

**🔧 技术方法**

使用的技术包括：MoE路由器的softmax概率分布与top‑k路由；三阶段损失函数（反目标、触发器、差距、KL散度、层级均匀化与容量限制）；低秩适配（LoRA）仅对路由器做微调；在多GPU实验中使用vLLM、EP并行。

**📊 数据集**

实验数据集涵盖四种文本语料：C4、ShareGPT、WikiText‑2、Alpaca；模型涵盖Mixtral‑8×7B‑Instruct、Qwen1.5‑MoE‑A2.7B‑Chat、OLMoE‑1B‑7B‑0924‑Instruct。

**📈 对比分析**

与干净基准相比，触发输入的目标专家占比可达92.3%‑95.6%，普通输入几乎保持原有分布；在实际EP服务实验中，触发流量导致p99首令时间上升1.43倍、吞吐量下降到0.86倍；模型下游任务（HellaSwag、ARC‑Challenge）的准确率仅下降不到2个百分点。

**⚠️ 局限性**

局限性包括：攻击依赖已知的连续专家分配（contiguous placement）且不适用于动态迁移或非连续映射；触发器需要以特定序列形式插入，若被检测则难以触发；若使用更严格的运行时审计阈值或更智能的路由平衡策略，攻击效果可能被抑制。

---

## 5. Hypothesis Frontier: Verifier Guided LLM and Symbolic Search for First-Order Induction

**arXiv ID:** 2608.10843 | [PDF](https://arxiv.org/pdf/2608.10843v1)

**作者:** Serafim Batzoglou `[一作]` `[通讯]` (Independent Researcher), Serafim Batzoglou (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Hypothesis Frontier的神经符号框架，用于在完全可观测的有限关系世界中通过LLM生成的公式与精确验证器迭代搜索，最终合成满足所有训练标签的一阶公式。

**💡 创新点**

创新点在于：①利用精确验证器把每个LLM生成的候选公式转换为可验证的前沿（frontier），并以此指导后续LLM调用；②通过父级派生的修复（repair）与简化（simplification）在符号层面纠正错误并压缩公式；③设计了符号优先（symbolic-first）工作流，在LLM调用前先用Z3求解；④通过实验表明即使是部分修复也能显著推进搜索进程。

**🔧 技术方法**

使用的技术包括：大型语言模型（LLM）进行公式生成、完整模型评估、符号级修复与简化（布尔归约、量化裁剪、等价简化等）、确定性训练仅排序（train‑only ranking）以及最终的精确简化器；此外将Z3作为符号求解基准。

**📊 数据集**

数据集为INDUCTION基准的Benchmark300和Challenge64两组完全可观测有限关系世界，包含谓词P、Q、R、S以及等号；每个世界提供域、谓词解释及目标扩展，并生成独立的holdout世界用于评估。

**📈 对比分析**

对比方法：在相同模型、任务集和LLM回合数下，将Hypothesis Frontier与单纯的重复LLM生成（无符号修复）进行比较；评估指标包括最终有效率、holdout准确率、AST大小以及LLM调用次数。实验显示Hypothesis Frontier在多数配置下显著提高了有效率、holdout准确率并缩短公式长度，同时在与Z3的符号优先流程结合时还能减少LLM调用次数。

**⚠️ 局限性**

局限性包括：仅适用于小型、完全可观测的有限世界；未扩展至部分可观测或更大规模结构；符号简化仅保证训练世界的行为一致，不保证逻辑等价；计算开销因多次LLM调用与验证而增加；实验对比未对单个组件做隔离评估，因而因果关系仅为观察性。

---

## 6. SafeCap: Improving LVLM Safety with Image Captioning Reinforcement Learning

**arXiv ID:** 2608.10513 | [PDF](https://arxiv.org/pdf/2608.10513v1)

**作者:** Caoyuan Ma `[一作]` (University of Tokyo), Yinqiang Zheng `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于强化学习的自我描述（Self‑Captioning）框架 SafeCap，用于提升大型视觉‑语言模型（LVLM）的安全性。

**💡 创新点**

创新点在于将安全对齐目标转化为“先生成安全相关图像说明再给出回答”的双阶段任务，并通过“caption‑mediated reward”让模型在生成说明时主动提取与安全决策相关的视觉信息；同时结合多信号奖励和组内归一化，避免奖励失衡。

**🔧 技术方法**

技术包括：GRPO 强化学习算法、三元奖励（模板、答案、说明），指数风险衰减函数、caption‑mediated 评估、组件级组内标准化，以及对模型输出的结构化控制。

**📊 数据集**

主要使用公开的 SPA‑VL 多模态安全对齐数据集进行训练；在评估时采用 MM‑SafetyBench、MSSBench、VLSBench、FigStep、MIS‑Test（5 个安全基准）以及 MM‑Vet、BLINK、MMVP、ERQA、VPCT、MMStar（6 个视觉实用基准）。

**📈 对比分析**

与安全 SFT、DPO、SafeGRPO 以及零训练基线进行对比；在 DirectCap 推理模式下，SafeCap 在 11 项指标上的平均安全分数提升 5.3–8.6 分，安全性显著提高且视觉实用性保持或略有提升，尤其在 4B‑Base 模型上安全分数提升 19 分。

**⚠️ 局限性**

局限性包括：对冻结文本 LLM 的依赖仍可能限制安全性；caption‑mediated 机制在极端视觉攻击（如细粒度攻击）下的鲁棒性未完全验证；训练过程仍需大规模计算资源；此外，Prism（仅使用说明的评估）表现不一，表明说明质量与下游 LLM 能力之间仍存在瓶颈。

---

## 7. Robust Sliding Mode and Admittance Control of Underactuated Aerial Manipulators for Contact-Based Inspection

**arXiv ID:** 2608.10656 | [PDF](https://arxiv.org/pdf/2608.10656v1)

**作者:** Tareq Aziz Alqutami `[一作]` (Heriot-Watt University), Mustafa Suphi Erden `[通讯]` (Heriot-Watt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了一套针对无动力矩六旋翼 UAV 与 1 轴机械臂的持续接触检验控制框架，通过积分增强滑模控制实现姿态与位置跟踪，并结合姿态前馈与阻抗控制实现接触力精准调节。

**💡 创新点**

创新点在于：① 将积分增强滑模控制与姿态前馈融合以抵消机械臂动力耦合；② 采用阻抗控制实现力的实时跟踪；③ 在软件仿真中验证可实现至 20 N 的接触力，同时保持姿态与表面对齐；④ 对比传统 PID，展示了更优的误差与耦合抑制性能。

**🔧 技术方法**

使用技术包括：滑模控制（带积分项与光滑切换）、姿态投影算法、阻抗力控制、ROS2 + PX4 软硬件架构、Gazebo SITL 高保真仿真、误差统计（RMSE、误差范数）等。

**📊 数据集**

使用的数据集为在 PX4‑Gazebo 软件仿真环境下生成的仿真轨迹与接触表面参数（弹性 10⁷ N/m，阻尼 0.1 N·s/m），并注入 5 % 质量估计误差来检验鲁棒性。

**📈 对比分析**

通过与经典 PID 进行对比，SITL 结果显示：在自由飞行时，SMD 的位置 RMSE 分别为 (0.021, 0.034, 0.004) m，低于 PID；在接触检验时，SMD 对 4 N 目标力实现 0.12 N 的 RMSE，且可稳定施加至 20 N；PID 在同一场景下出现更大误差并在 9 N 以上失稳。

**⚠️ 局限性**

局限性：仅在仿真中验证，未进行硬件实验；仅针对单轴机械臂，未考虑多自由度或复杂曲面；对高速动态接触、非线性阻尼等情形的鲁棒性尚未评估；以及对机械臂与 UAV 结构耦合的更精细建模仍需进一步研究。

---

## 8. Evaluating Rational Contracting in Natural Language

**arXiv ID:** 2608.10475 | [PDF](https://arxiv.org/pdf/2608.10475v1)

**作者:** Bhavyesh Sajja `[一作]` (National University of Singapore), Tan Zhi-Xuan `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大语言模型在自然语言合同谈判与执行中的理性与合作性能，并提出ContractSim评估框架。

**💡 创新点**

将自然语言合同视为对联合策略的约束，构建理性谈判-执行游戏模型，并量化合同质量与代理合作度。

**🔧 技术方法**

结合大语言模型（Claude Opus 5、Gemini 3.6 Flash、GPT‑5.6‑Sol）、有限时限动态规划的理性基准（RC、RE、RCC）以及自然语言到形式约束的解析器。

**📊 数据集**

使用自构造的ContractSim基准套件，包含六个环境、三种供应商场景（餐饮、酒店清洁、AI托管）的合成合同与执行数据。

**📈 对比分析**

与理性基准及人类对照在谈判质量、合同互惠、满意度、执行合规率等指标上对比；在低随机性环境中近似Pareto最优，但在高随机性环境下效率低、违约率高。

**⚠️ 局限性**

当前LLM缺乏可靠的协作与合规性，难以在不确定环境下谈判可满足的合同；提示可缓解但仍不完美，未涵盖重新谈判、市场合同与仲裁等情形。

---

## 9. Power law graph attention: exact generalization of scaled dot-product attention, empirical collapse at inference

**arXiv ID:** 2608.10288 | [PDF](https://arxiv.org/pdf/2608.10288v1)

**作者:** Burc Gokden `[一作]` `[通讯]` (Fromthesky Research Labs LLC), Burc Gokden (Fromthesky Research Labs LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的解码器模型 PLDR‑LLM，其注意力机制 PLGA 用输入生成的非线性双线性算子替代传统的 SDPA，形成可学习的正交张量链（Query Gram → Φ_res → 𝐀 → 𝐀^P → 𝐀^a + 𝐀^b），并在解码器中实现全局归一化与自适应卷积。

**💡 创新点**

创新点包括：
• 通过逐行残差网络与逐元素幂律生成全局正交张量，实现了输入自适应的双线性注意力；
• 证明 PLGA 在 G = I 时恰好等价 SDPA，并给出条件可逆性与共轭群（RoPE 旋转）下的相对位移不变性；
• 发现并定量化了“推理崩塌”现象：在训练后生成的 Deductive 输出 A、Ψ 等几乎不随输入变化，可直接缓存；
• 引入 DAG 正则化（NOTEARS 计数法）和自组织临界性（SOC）训练框架，提供了对模型内部张量正则性与阶段性临界性的理论与实验分析；
• 所有核心定理均已 Lean4 形式化验证，保证数学严谨性。

**🔧 技术方法**

技术实现：
• 旋转位置编码 RoPE；
• 逐行残差网络 Φ_res（SwiGLU+层归一化）;
• 正交张量生成 𝐀 = Φ_res(D_Q)，随后通过幂律 𝐀^P、线性映射 𝐀^a + 𝐀^b；
• 采用可学习的缩放与偏置 W、b_W、P、a、b_a；
• 软max 采用理想掩码 + 实际浮点实现；
• KV‑cache 与 G‑cache 两级缓存；
• 训练目标为块级交叉熵 + DAG 正则化；
• 评估使用 TruthfulQA、One‑Pass Block Scoring、序列 Perplexity 等。

**📊 数据集**

训练数据：在论文实验中使用约 41 B tokens 的通用文本语料（与原始 PLDR‑LLM 论文一致），词表大小 32 000，最大上下文长度 1 024。评估主要基于 TruthfulQA、GLUE 等标准 NLP 任务。

**📈 对比分析**

对比方法：与标准 SDPA‑Transformer（不含 PLGA）以及原始 PLDR‑LLM 版本进行对比。实验表明：
• 在 blockwise 交叉熵与序列 Perplexity 上与 SDPA‑LLM 近乎相同；
• 在 TruthfulQA 及 One‑Pass Block Scoring 上，差距 ≤ 5×10⁻⁵；
• 缓存方案 G‑cache 在推理速度上提升约 3×；
• 在不同训练批次与学习率设置下，近临界训练能显著降低 Deductive 输出方差，形成“自组织临界性”阶梯。

**⚠️ 局限性**

局限性：
• 未证明 PLGA 在参数相同的条件下必然优于 SDPA；
• 推理崩塌与缓存等性质仅在满足输入不变性（ε‑精度）时才成立，实际模型仍可能对输入产生细微变化；
• DAG 正则化无法强制生成无环结构，实际上仅提供循环权重的软惩罚；
• 许多理论结论为条件定理（依赖于非共振、梯度不为零等假设），在不同训练/推理配置下可能失效；
• 该方法主要在通用文本上验证，缺乏对长文本、代码或多模态任务的系统评估。

---

## 10. The GENEA Challenge 2026: A Large-Scale Disentangled Evaluation of Speech-Driven Gesture Generation on the Seamless Interaction Dataset

**arXiv ID:** 2608.10839 | [PDF](https://arxiv.org/pdf/2608.10839v1)

**作者:** Rajmund Nagy `[一作]` (KTH Royal Institute of Technology), Gustav Eje Henter `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了五个基于Seamless Interaction数据集的语音驱动手势生成系统，并通过四项拆解评估（运动真实感、与语音对齐、对话伙伴对齐、语义对齐）在大规模人类实验中收集超过23,000票

**💡 创新点**

引入了语义手势生成任务与文本不匹配评估，使用拆解评估方法消除运动与语音、对话伙伴、语义之间的相互干扰，并在更大、更具双人对话和语义表达的Seamless Interaction数据集上开展评估

**🔧 技术方法**

采用 pairwise voting + JUICE 细粒度因子，使用音频/语音不匹配、对话伙伴不匹配、文本不匹配三种不匹配方法；评估指标包括 Bradley‑Terry Elo、适当性得分（-1~1）等；系统包括 GestFlow、UNICAMP、UM‑FERI、DyaSync 与其语义变体

**📊 数据集**

使用 Seamless Interaction 数据集（dyadic conversations 3400h + grounded gestures 380h）和对照的 motion‑capture 数据；此外对挑战系统进行多样化输出采样

**📈 对比分析**

通过 pairwise winrate、Elo 评分和适当性得分对比系统与 motion‑capture；结果显示 motion‑capture 仍位居顶端，系统间分差显著；语音对齐最高为 32%（vs 62% 预期 ceiling），对话伙伴与语义对齐几乎未超随机水平，只有 motion‑capture 在这两项上表现明显优异

**⚠️ 局限性**

系统在对话伙伴响应和语义表达方面表现不佳；dyadic 评估的置信区间可能低于零，提示随机性和样本选择影响；评估仍以人类投票为主，缺乏自动化指标；数据集虽大但对标注细粒度和多样性仍有提升空间

---

## 11. DuplexWorld: Can voice agents help you get through the day?

**arXiv ID:** 2608.10716 | [PDF](https://arxiv.org/pdf/2608.10716v1)

**作者:** Aryan Vijay Bhosale `[一作]` (Centific Global Solutions Inc.), Dinesh Manocha `[通讯]` (University of Maryland)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一个统一、全面的 Speech‑to‑Speech（S2S）语音代理评测基准（<Name>），涵盖六个“日常世界”与多种对话类型，并对五个商业实时语音代理进行评估。

**💡 创新点**

创新点包括：
- 通过六个设计精细、覆盖日常事务与导航等多样场景，超越以往仅关注数据库查询或文本代理的评测范围。
- 统一的 12 项指标（对话动态、代理能力、自然性）在同一 harness 下计算，避免多脚本互相冲突。
- 引入“探索–利用”视角评估代理在不确定情况下的动作决策，揭示高探索度与完成率负相关。
- 在评测中将真实语音噪声与语速变化等多种语音干扰纳入，逼近真实电话语音环境。

**🔧 技术方法**

技术手段包括：
- Tick‑based 仿真器与实时 WebSocket 接口，用于同步音频与工具调用。
- LLM 判别器（LLM judge）评估对话进展与文本一致性。
- 语音质量评估使用 DNSMOS、UTMOS、NISQA 等无参考 MOS 预测器。
- 对话动态指标（Turn‑Taking、Conversation Progression、Selectivity）基于转折时间与干扰识别实现。

**📊 数据集**

使用的“数据集”主要是作者自建的 350+ 小时模拟对话，包含：
- 5 个企业世界（银行、物流、医疗、保险、旅游）每个 27 个情境，共 135 场；
- 1 个导航世界（8×8 城市网格）共 9 个情境。
- 对话中采用合成语音、两种通道（清晰 vs 现实）以及多语言人格。

**📈 对比分析**

比较方法：
- 在所有世界中使用统一配置跑 5 次每个情境，得到 135/90 对话；
- 对每个系统（如 5 家商业 API）计算 12 指标的平均值并给出 95% 置信区间。
- 结果显示：
  - 代理能力最高的系统在不同世界间得分差距可达 0.2–0.67，可靠性仅 0.49；
  - 对话动态最好的系统在代理能力上表现差，显示两者不一致；
  - 语音自然性差异不大（DNSMOS 3.13–3.40），与任务完成率无关。

**⚠️ 局限性**

局限性：
- 仅使用合成语音，缺乏真实多方言、口音与嘈杂环境的数据；
- 工具接口为声明性 mock，缺少真实延迟与错误；
- 评测仅在英语环境，无法直接推广至其他语言；
- 判别器与 MOS 预测器未在本数据集上重新验证，可能与人类评估存在偏差；
- 评测的目标状态检查可能被“无行动”策略规避，导致不准确的完成度评估。

---

## 12. A Study of Cursorrules Files in GitHub Open Source Projects

**arXiv ID:** 2608.10622 | [PDF](https://arxiv.org/pdf/2608.10622v1)

**作者:** Shuang Sun `[一作]` (Leiden University), Olga Gadyatskaya `[通讯]` (Leiden University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Cursor AI 助手的 legacy 提示配置文件（.prompt 文件）在 GitHub 开源项目中的出现、分布、演化、维护情况进行混合方法研究，量化分析了 12,110 个文件，定性分析了 65 个文件并构建了 65 码的主题编码表。

**💡 创新点**

首次在大规模样本上系统研究 Cursor legacy 提示文件的使用模式和内容主题，揭示这些文件主要出现在小型单人项目、更新频率低、内容以代码质量和项目结构为主，安全相关内容极少；并构建了可复用的代码书。

**🔧 技术方法**

采用混合方法：GitHub Code Search + GraphQL API 收集数据；定量统计、时间序列分析、commit 变化量化；质性使用 Atlas.ti 进行主题分析，Krippendorff’s Alpha 评估编码一致性。

**📊 数据集**

数据集为 12,110 个 .prompt 文件，来自 11,427 个公开 GitHub 仓库（2024‑2025 年），以及随机抽取的 65 个文件用于质性编码。

**📈 对比分析**

通过对文件创建时间、大小、更新频次、commit 间隔、修改幅度等指标进行描述性统计，并与新格式 .cursor.rule 文件的主题分布进行对比；发现文件更新率约 32.7%，平均修改规模小，安全主题仅占约 4.4%。

**⚠️ 局限性**

受限于 GitHub API 查询上限、仅公开仓库的样本、质性样本量（65）有限、编码和主题可能受研究者主观影响，以及 legacy 文件即将被弃用导致结果仅适用于旧文件。

---

## 13. The Deliberative Deficit: An Empirical Critique of LLMs in Democratic Discourse

**arXiv ID:** 2608.10186 | [PDF](https://arxiv.org/pdf/2608.10186v1)

**作者:** Maurice Flechtner `[一作]` (University of Zurich), Maurice Flechtner `[通讯]` (University of Zurich)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5120410770)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估大型语言模型在多元议题下的集体推理能力，提出三维评价框架并在19项公民议会实验中进行对比实验。

**💡 创新点**

将政治学中的Deliberative Reason Index (DRI) 用作群体层级的后验一致性度量，结合程序质量与多样性三维检验，首次揭示LLM在非可验证多元议题上的“辩论缺陷”。

**🔧 技术方法**

采用多模型多轮对话（GPT-5.1, Gemini, Claude, DeepSeek, Kimi 等）结合AQuA程序评估，DRI计算及欧氏距离多样性测量。

**📊 数据集**

使用12个公民议会议题（气候、医疗、治理等）对应的调查问卷数据与人类参考分布（N=407），以及AQuA的欧美议会样本（N=910）。

**📈 对比分析**

与人类参考相比，LLM在程序质量几乎相当（AQuA≈2.94 vs 2.98），但在DRI提升仅为0.029（约占人类0.099的30%），多样性低于人类三分之一，且主题相关性不显著。

**⚠️ 局限性**

限制包括仅测试无定制的前沿LLM、缺乏跨领域验证、DRI无法捕捉非命题交流、样本量与主题范围有限，且评估依赖特定民主理论框架。

---

## 14. SKILLER: Language-Level Reinforcement Learning for Reusable Skill Extraction in Small Language Models

**arXiv ID:** 2608.10538 | [PDF](https://arxiv.org/pdf/2608.10538v1)

**作者:** Chenhao Dang `[一作]` (Shanghai Jiao Tong University), Weijia Li `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于自然语言强化学习的框架 SKILLER，能够自动生成专门针对小规模 LVLM 的执行技能，从而提升其任务完成质量；

**💡 创新点**

创新点在于把技能文本本身视作可优化的策略，使用强大的前沿模型作为演员和评论家，在小模型的 agent loop 作为环境中通过自然语言传递所有 RL 信号，完全避免了梯度更新，解决了模型不匹配的瓶颈；

**🔧 技术方法**

采用演员-评论家结构、进阶技能披露机制、结构化自然语言反馈、回放记忆以及四种文本编辑操作（Insert、Replace、Create、Delete）实现技能的逐步改进；

**📊 数据集**

在五个公开基准上进行评估，分别是 SkillsBench、SkillLearnBench、SWE‑Skills‑Bench、GAIA 和 EarthBench；

**📈 对比分析**

与三种开源自动技能演化方法（AutoSkill、EvoSkill、SkillX）、闭源 Manus 以及无技能 baseline 进行对比，SKILLER 在 Qwen3.5‑9B 与 Qwen3.5‑4B 上均显著优于所有基线，提升幅度可达 4.3%–20.4%（9B）和 1.8%–13.3%（4B），甚至让 4B 在部分任务上超过 9B；

**⚠️ 局限性**

主要局限是需要强大模型做离线生成，生成过程仍有成本；对环境与任务设计的依赖较强，泛化到更大范围或不同类型任务的性能尚未完全验证；

---

## 15. Share First, Route What Remains: A Unified Framework for Token-Adaptive MoE Computation

**arXiv ID:** 2608.10392 | [PDF](https://arxiv.org/pdf/2608.10392v1)

**作者:** Gongli Zhang `[一作]` (South China University of Technology), C. L. Philip Chen `[通讯]` (South China University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种统一的 token‑adaptive MoE 框架，先在共享路径上执行可复用的块，再将剩余的计算路由到少量专家；

**💡 创新点**

创新点在于将共享建模、细粒度计算与动态路由视为同一顺序决策（先共享、后路由），通过块级共享与 Gram 正交约束实现对共享宽度、共享内容和残差专家数的联合调度；

**🔧 技术方法**

核心技术包括基于 key‑value 通道的块划分、token‑级共享需求评分、残差专家的累计路由决定以及 Gram 正交正则化来保持路由向量多样性；

**📊 数据集**

实验使用 Vision 的 DomainBed（PACS、VLCS、OfficeHome、TerraIncognita、DomainNet）和 NLP 的 GLUE（CoLA、MRPC、QNLI、MNLI、RTE）数据集；

**📈 对比分析**

与传统静态和动态 MoE 以及密集模型对比，本文方法在 DomainBed 上平均提升 1.4%（PACS、VLCS、DomainNet）且显著降低激活参数、FLOPs、推理时间与内存；在 GLUE 上相较最优固定‑k MoE 提升约 1–2% 的任务分数，并在效率上优于 DynMoE 与 MASS；

**⚠️ 局限性**

局限性包括对块粒度 B 的敏感性、在极端多模态任务中的共享划分可能不足、以及未在所有可分析指标（如极限 FLOPs 或能源消耗）上实现最优，未来需进一步研究更灵活的共享策略与跨域泛化。

---

## 16. Outer Limits: An Experimental Approach to Controlled Content Manipulation within the Reddit Interface

**arXiv ID:** 2608.10115 | [PDF](https://arxiv.org/pdf/2608.10115v1)

**作者:** Chenchen Mao `[一作]` (Lehigh University), Dominic DiFranzo `[通讯]` (Lehigh University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在旧版Reddit的界面中，利用浏览器扩展实现对内容的精准替换、交互日志记录与写入操作的隔离，完成对研究者指定内容的随机对照实验；

**💡 创新点**

将研究者预设的内容在保持原始界面与社交语境的前提下，在浏览器端进行局部渲染并屏蔽所有写入请求，兼具内容可控、原生界面与平台后果隔离三大优势；

**🔧 技术方法**

Chrome Extension（前端DOM替换、事件拦截）、Express API + MongoDB 后端存储与分配、CSV配置文件、ART ANOVA 与 TOST 统计分析；

**📊 数据集**

基于两篇真实Reddit帖子（图像型与文本型）进行感知真实性验证，样本219名Prolific受试者；另外通过示例因子实验设计展示系统可用于多变量内容操纵；

**📈 对比分析**

与传统重建模拟和客户端重排方法相比，Outer Limits在保持原始界面与社交嵌入的同时，确保实验内容与交互不流向主平台；在感知真实性测试中未出现显著差异，且所有比较均满足 d=±0.50 的等效性界限，证明其在实验条件下的可靠性；

**⚠️ 局限性**

仅适用于桌面版旧版Reddit；对新界面、移动端或视频内容不支持；DOM 变更需重新维护；感知真实性评估仅覆盖两篇帖子，需为新材料单独验证；且该方法仍涉及对受试者的隐瞒与欺骗，需要严格伦理审查与事后告知。

---

## 17. DIY e-HandPan: A new DIY Low-Cost Handpan Interface based on Arduino and ESP32 Microcontrollers

**arXiv ID:** 2608.10185 | [PDF](https://arxiv.org/pdf/2608.10185v1)

**作者:** Benoit Collin `[一作]`, Eric Genotelle `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一个基于回收CD制作、低成本、可编程的手掌音板eHandPan，并实现了多传感器采集与实时MIDI与音频输出。

**💡 创新点**

创新点在于将手掌乐器与可编程微控制器、互联WiFi和交互学习界面相结合，实现了低成本的多功能音频设备。

**🔧 技术方法**

采用了Piezoelectric传感器、LED指示灯、Arduino/ESP32微控制器、USB-MIDI、I2S DAC、Wi-Fi Web接口等技术。

**📊 数据集**

未使用公开数据集，系统以现场传感器采集的实时音频信号为输入。

**📈 对比分析**

由于是硬件原型，未与其他系统做性能对比，但可实时采样并通过MIDI或音频输出。

**⚠️ 局限性**

局限包括传感器分辨率有限、低成本材料可能影响音质、软件功能相对基础，需进一步优化。

---

## 18. Modelling Geographic Atrophy Progression using Implicit Neural Representations

**arXiv ID:** 2608.10807 | [PDF](https://arxiv.org/pdf/2608.10807v1)

**作者:** Simone Sarrocco `[一作]` (University of Basel), Philippe Cattin `[通讯]` (University of Basel)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本研究利用隐式神经表示（INR）在低数据环境下对年龄相关性黄斑变性（AMD）患者的地理萎缩（GA）进展进行个体化建模，既能重建未来或缺失的FAF图像，又能预测GA分割结果。

**💡 创新点**

创新点在于首次将INR与眼球特定的空间潜变量结合，使用时间与年龄调制实现连续的病程轨迹，且在单一潜变量下同时生成图像与分割，显著提升了GA面积预测精度。

**🔧 技术方法**

采用基于SIREN激活的多层感知机（MLP）INR，配合FiLM调制层进行时间与年龄嵌入，训练时使用MSE+Dice/BCE损失，测试时通过优化眼球潜变量实现自适应。

**📊 数据集**

使用了Omega数据集，该数据集包含37只眼（30名患者）在多次（最多4次）随访的768×768短波长FAF图像和相应的GA分割，平均随访间隔约12周。

**📈 对比分析**

与ImageFlowNet、T‑UNet、T‑I2SBUNet以及线性/三次B‑样条插值、复制前进等经典方法比较，实验显示在GA面积MAE和Dice分数上达到最优或接近最优（例如MAE≈0.20 mm²、Dice≈0.91），FAF重建质量虽不如复制前进但已可接受。

**⚠️ 局限性**

局限性包括：1）随访时间短，导致FAF重建表现受限；2）模型对高频细节重建仍不够精确；3）只验证了单模态（FAF）与单眼数据，未覆盖不同AMD阶段或多模态（如OCT）情况。

---

## 19. SQuaT: Self-Supervised Knowledge Distillation via Student-Aware Quantized Teacher Features

**arXiv ID:** 2608.10709 | [PDF](https://arxiv.org/pdf/2608.10709v1)

**作者:** HyeonJun Lee `[一作]` (Kookmin University), Jangho Kim `[通讯]` (Kookmin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在无标签的量化感知训练（QAT）中，提出 SQuaT 方法，通过将教师网络的特征投影到学生的量化格子上，实现教师特征与学生可达特征空间对齐，从而实现无监督的高效模型量化。

**💡 创新点**

创新点在于引入“学生感知投影”（Student‑Aware Projection），消除教师FP特征与低比特学生特征之间的不可达残差，消除先前教师感知投影带来的不可逼近误差下限；并在同一框架下同时对特征层与logit层进行对齐。

**🔧 技术方法**

使用的核心技术包括：量化感知训练（EWGS）、知识蒸馏（feature‑level + logit‑level KL）、温度平滑、学生感知投影、无监督对齐损失以及针对不同量化比特宽度的自适应梯度缩放。

**📊 数据集**

实验数据集涵盖视觉任务的 CIFAR‑10、CIFAR‑100、ImageNet‑1K 以及 NLP 任务的 GLUE（RTE、SST‑2、QNLI），并在多种模型架构（ResNet、DeiT‑Tiny、BERT）上验证。

**📈 对比分析**

与基线的对比方法包括传统监督 QAT（EWGS）以及无标签 KD 基线 SQAKD；实验表明 SQuaT 在所有比特宽度（尤其是 1‑bit、2‑bit）和模型规模下均优于基线，提升幅度可达 0.4‑1.7pp；在边缘设备（Jetson Nano）上实现显著的推理速度提升（≈6.5×）。

**⚠️ 局限性**

局限性包括：仍需预训练的 FP 教师模型；对极低比特宽度之外的非均匀量化或特殊算子场景的适应性未完全验证；以及在大规模训练时对投影参数的调优可能引入额外的超参数依赖。

---

## 20. MARCO: Click-Intent Decomposition for Calibrated Ads Conversion Prediction

**arXiv ID:** 2608.10562 | [PDF](https://arxiv.org/pdf/2608.10562v1)

**作者:** Shiwen Shen `[一作]` (Meta AI), Ellie Wen `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过将点击意图拆分为高意图CTA和低意图社交两类，并在CTR和CVR模型中使用向量头，实现了工业广告排名的结构化解耦。

**💡 创新点**

创新点在于利用日志中的免费行为标签进行意图分解，构建可在推理时动态路由的多意图模型MARCO，突破单标注模型的可观测性瓶颈。

**🔧 技术方法**

采用多输出向量头、无监督的点击意图标签、CTR+CVR组合、在线校准与路由效率理论分析，以及大规模A/B实验验证。

**📊 数据集**

使用Meta生产广告日志，包括成千上万用户的点击与转化数据，涵盖多广告位和多转化类型。

**📈 对比分析**

与传统单标注CVR、辅助意图任务、历史意图特征以及MoE模型对比，MARCO在线累计提升约0.98%全局业务指标，单次实验提升转化率约2.80%。

**⚠️ 局限性**

局限在于目前仅使用二元意图划分，进一步细化意图会导致稀疏性与校准方差问题，且路由效率受CTR模型容量限制。

---

## 21. Similarity Gates Approve Reversals: A Validity Audit of Embedding-Cosine Thresholds in Agent Systems

**arXiv ID:** 2608.10216 | [PDF](https://arxiv.org/pdf/2608.10216v1)

**作者:** Scott E. Frias `[一作]` `[通讯]` (Eigenforma), Scott E. Frias (Eigenforma)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对在实际系统中使用余弦相似度门控的文本相似性检测器进行审计，发现其误将词法变化（尤其是否定/反转）与语义一致性混为一谈，导致安全门失效。

**💡 创新点**

创新点在于构建了基于2×2因子设计的去混杂测试语料和评估工具，系统性揭示余弦相似度在检测意义保持与否方面的局限，并公开了完整的审计工具和数据。

**🔧 技术方法**

采用了句子编码器（多种预训练模型）、余弦相似度、Jaccard词重叠、AUROC评估以及自定义的匹配对生成和评估脚本。

**📊 数据集**

使用了两个任务的自生成对照语料（约80条匹配对）以及真实生产系统的变异和对照数据，共计约140条样本。

**📈 对比分析**

通过比较标准阈值门（固定阈值）与因子设计的对照实验，发现标准门的平衡准确率最高仅0.70，且在生产配置下性能几乎等于随机；而去混杂测试表明正确的门需要在不同词重叠层面使用不同阈值，最高可达AUROC 0.90。

**⚠️ 局限性**

局限包括：语料规模小、单一注释者标注、仅评估单一生产系统、生成语料来自单一项目和模型、未包含多语言或更大规模的真实部署数据。

---

## 22. FITTER: Vocabulary-Agnostic Cross-Domain Inference on Temporal Knowledge Graphs

**arXiv ID:** 2608.10668 | [PDF](https://arxiv.org/pdf/2608.10668v1)

**作者:** Jiaxin Pan `[一作]` (University of Stuttgart), Steffen Staab `[通讯]` (University of Stuttgart)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FITTER，一个在时序知识图谱上实现跨域全归纳推理的模型。

**💡 创新点**

创新点在于结合相对时序编码与词表无关的关系交互图，突破实体、关系和时间词表不匹配的限制。

**🔧 技术方法**

采用基于 NBFNet 的消息传递框架、周期性正弦时序编码、全局与局部时序图融合以及 MLP 打分器等技术。

**📊 数据集**

使用了六个主流时序知识图谱基准：ICEWS14/05‑15/18、GDELT、YAGO 和 WIKI。

**📈 对比分析**

与 ULTRA、INGRAM 等归纳基线相比，FITTER 在 15 个跨域转移任务中均实现显著提升（MRR 最高约 43%），并在单一模型下与部分特定词表模型竞争。

**⚠️ 局限性**

局限性包括对极稠密或极稀疏时间序列的局部模式捕获仍有限，以及在极长时间跨度稀疏图上性能略逊于专门的转移学习方法。

---

## 23. Dual-Loop Self-Evolution via Verifiable Emotion Feedback for Multi-Turn Empathetic Dialogue

**arXiv ID:** 2608.10626 | [PDF](https://arxiv.org/pdf/2608.10626v1)

**作者:** Yi Wei `[一作]` (Alibaba Cloud Computing), Chi Zhang `[通讯]` (Alibaba Cloud Computing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种双循环自进化框架，通过可验证情绪反馈同时改进对话策略和自适应训练经验分布，以提升多轮共情支持。

**💡 创新点**

创新点在于将连续情绪奖励用于策略优化，并将分组通过阈值化的情绪通过率驱动外部循环的经验重分配，使训练预算与模型能力保持一致。

**🔧 技术方法**

采用强化学习（GRPO）结合情绪验证器、层级控制器、组内优势估计、稀疏奖励的无偏估计以及不确定性引导的探索和均衡重放。

**📊 数据集**

使用 SAGE 作为主要情绪评估数据集，并在 ESC‑Eval、EIBench、ESConv 等多种自动与人工评价数据集上进行跨测评。

**📈 对比分析**

在统一的 Qwen3‑8B 初始化、相同的模拟器与验证器、相同的 rollout 预算与批大小下，与基线 Qwen3‑8B、RLVER、状态单独使用以及 VCRL 对比，SAGE Overall 从 53.87 提升到 79.24（比 RLVER 提升 7.23 分，约 10%），在 ESConv、ESC‑Eval、EIBench 以及人工评价中亦获得显著提升。

**⚠️ 局限性**

局限性包括对模拟器与情绪验证器的高度依赖，缺乏真实用户长期交互验证，以及在更大规模或更复杂情绪场景中的可推广性尚未充分验证。

---

## 24. Pair-Centric Graph Rewiring for Over-Squashing via Optimal Transport-Guided Communication Alignment

**arXiv ID:** 2608.10619 | [PDF](https://arxiv.org/pdf/2608.10619v1)

**作者:** Yan Wang `[一作]` (Sun Yat-sen University), Chuan-Xian Ren `[通讯]` (Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于节点对通信短缺的图重连框架 PairAlign，使用 Optimal Transport 将有限边预算精准匹配至需要补偿的节点对，从而缓解 MPNN 的 over‑squashing。

**💡 创新点**

创新点在于：①把 over‑squashing 转化为可计算的需求–支持比率（短缺评分）；②利用 Optimal Transport 对候选边与高短缺节点对进行全局协调分配，避免局部贪心导致的覆盖不足；③通过行归一化传播支持评估实现可训练的短缺更新。

**🔧 技术方法**

使用技术包括：需求–支持短缺公式、行归一化传播矩阵、梯度下降+straight‑through 近似、Sinkhorn 算法实现的 OT 对齐、以及基于短缺的边插入评价。

**📊 数据集**

实验数据集涵盖：节点分类（Cora、Citeseer、Texas、Cornell、Wisconsin、Chameleon）；图分类（ENZYMES、IMDB‑BINARY、MUTAG、PROTEINS、REDDIT‑BINARY、COLLAB）；异质图分类（Roman‑Empire、Amazon‑Ratings、Minesweeper、Tolokers、Questions）。

**📈 对比分析**

与多种重连基线（SDRF、FoSR、GTR、LASER、GOKU 等）以及无重连 baseline 在 GCN、GIN、GAT 等后端上进行对比。PairAlign 在大多数数据集上均获得最高或次高排名，节点/图分类准确率平均提升 2‑5% 甚至更高，并在 ΔShortage 与 Coverage@10 指标上显著优于 Greedy‑Local。

**⚠️ 局限性**

局限性包括：仅支持加法重连的离线预处理；对大规模图的计算成本较高；对传播深度、OT 权重 λ_OT 等超参较为敏感；在极度异质或需显式特征引导的重连任务中效果可能不如特征/社区驱动方法。

---

## 25. UniProbe: A Learnable Token-Level Hallucination Detector for Large VLMs using Multi-Structural Internal Representations

**arXiv ID:** 2608.10835 | [PDF](https://arxiv.org/pdf/2608.10835v1)

**作者:** Dvir Samuel `[一作]` (NVIDIA Research), Haggai Maron `[通讯]` (NVIDIA Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级可学习的检测器UniProbe，用于在冻结的大规模视觉语言模型（LVLM）生成过程中，实时定位并纠正词级幻觉

**💡 创新点**

创新点在于将LVLM内部的隐藏状态、注意力图和图像补丁构建为异构计算轨迹图，并通过交替使用图神经网络、视觉Transformer和GRU共同捕捉空间、关系与顺序三维证据；同时提供流式变体实现在线幻觉检测与自适应修正

**🔧 技术方法**

采用图神经网络（GNN）提取跨模态关系，ViT处理二维图像几何，GRU捕获生成顺序，并在训练时只读取单层隐藏状态与注意力；还利用自适应微调与流式检测策略

**📊 数据集**

在多种公开数据集上评估：MHALO、HalLoc（词级）、POPE、COCO-CHAIR、Objects365等，用于标注幻觉、对象错误与自生成图像文本

**📈 对比分析**

与零样本、全模型微调、外部验证器等多种基线对比，UniProbe在词级F1、IoU和对象幻觉F1等指标上均显著提升（如MHALO F1提高4–6点，POPE AUC提升至90.0），且推理延迟仅为1.06–1.15倍原始生成速度

**⚠️ 局限性**

局限性包括需访问内部隐藏状态与注意力，限制仅适用于可开放源码或可直接调用的模型；每个后端需要单独训练读取器，跨后端迁移尚未解决

---

## 26. Unsupervised Detection of Groundwater Storage Anomalies in Ghana Using GRACE Satellite Data

**arXiv ID:** 2608.10233 | [PDF](https://arxiv.org/pdf/2608.10233v1)

**作者:** George Yamoah Afrifa `[一作]` (Ghana Space Science and Technology Institute), Marcellin Atemkeng `[通讯]` (Rhodes University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用GRACE卫星观测的地下水储量数据，结合统计方法与无监督机器学习，检测并分析了2004‑2024年加纳地下水储量的异常时空变化。

**💡 创新点**

首次将Isolation Forest无监督异常检测算法应用于GRACE地下水数据，能够捕捉到细微非线性异常，并与传统Z‑score阈值相比展现更高的灵敏度与补充性。

**🔧 技术方法**

采用Z‑score标准化、Isolation Forest算法、统计阈值比较、混淆矩阵评估、逆距离加权（IDW）空间插值等技术。

**📊 数据集**

GRACE/GRACE‑FO 地面水存储观测（2004‑2024），通过去除土壤水、地表水、雪水等得到月度地下水储量异常，按格网划分为空间网格。

**📈 对比分析**

通过混淆矩阵比较Isolation Forest与Z‑score阈值检测结果，发现机器学习方法识别了12个月异常（5次缺水、7次盈余），而传统阈值方法仅捕获极端事件，说明无监督方法的检测性能更优，能发现更多细微异常。

**⚠️ 局限性**

主要局限在GRACE分辨率约300‑400 km，难以揭示局部或小尺度地下水变化；空间插值仅为可视化，未提供额外空间信息；缺乏现场观测验证来进一步确认异常真实性。

---

## 27. Technology, education and critical media literacy: potential, challenges, and opportunities

**arXiv ID:** 2608.10778 | [PDF](https://arxiv.org/pdf/2608.10778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 28. GeoForge: Non-Parametric Self-Evolving Agents for Earth-Observation Reasoning

**arXiv ID:** 2608.10494 | [PDF](https://arxiv.org/pdf/2608.10494v1)

**作者:** Xin Xiao `[一作]` (Chongqing University), Kaiwen Wei `[通讯]` (Chongqing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GeoForge，一种训练无关的自演化框架，用来构建并执行地球观测任务的科学工作流。

**💡 创新点**

创新点在于将完成轨迹转化为结构化的非参数执行状态，并通过工作流图记忆、动作级经验库和适配的 SOP 三层记忆实现任务感知的自学习与安全蒸馏。

**🔧 技术方法**

采用 ReAct 循环、LangChain/LangGraph、工作流图记忆、动作经验抽取、适配 SOP 与安全门控蒸馏等技术。

**📊 数据集**

在 Earth-Bench、ThinkGeo 与 GeoPlan-Bench 三个地球观测基准数据集上进行评估。

**📈 对比分析**

与 Earth-Agent、OpenEarth-Agent、GeoEvolver 等现有方法比较，GeoForge 在多种 LLM 后端上平均提升 10–20% 的准确率，并显著改善工具使用轨迹的覆盖率、顺序与完整度。

**⚠️ 局限性**

局限性包括对经验库更新的依赖、对记忆质量过滤的敏感度，以及在极端新情境下仍可能需要人工干预。

---

## 29. Operationalising Relative Causal Knowledge: Backbone Identifiability from Private Reports on a Shared Outcome

**arXiv ID:** 2608.10664 | [PDF](https://arxiv.org/pdf/2608.10664v1)

**作者:** Fabrizio Russo `[一作]` (Imperial College London), Mark Somers `[通讯]` (Fifty One Degrees Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在两个代理的共效因果网络中，如何从各自的局部因果报告恢复共享的“骨干”结构，以实现因果知识的传递。

**💡 创新点**

①证明在一般情况下，局部因果边缘报告无法唯一确定骨干，存在无限多可兼容的骨干；②在满足可加分离性的假设下，并通过代理之间传递已识别的因果响应函数，可唯一恢复骨干，从而实现无歧义的因果知识传输。

**🔧 技术方法**

使用结构因果模型（SCM）、概率核（kernel）与边缘化技巧，构造微扰证明；引入局部重叠（minorisation）条件；应用可加分离性约束与响应函数通信框架。

**📊 数据集**

论文为理论研究，未使用实际数据集；仅以教育价值增值（teacher‑quality 与 neighbourhood‑mobility）为例说明理论结果，引用了Chetty等人的估计。

**📈 对比分析**

没有实验对比；评估基于数学证明与理论示例，指出在满足可加分离性并完成响应函数通信时，骨干可被唯一识别，因果知识可被无歧义传递。

**⚠️ 局限性**

仅适用于两代理共效结构，假设局部重叠与可加分离性；未讨论大规模网络、非共效因果结构或如何在缺乏完整响应函数通信时恢复骨干。

---

## 30. Co-Evolution in Agentic Systems: Toward Self-Directed Evolution Beyond Human Design

**arXiv ID:** 2608.10299 | [PDF](https://arxiv.org/pdf/2608.10299v1)

**作者:** Qing Zong `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

综述了在Agentic系统中多组件共同进化（Co‑Evolution）的研究进展，并提出了三阶段递进式分类法，帮助系统性地组织相关工作

**💡 创新点**

首次将Co‑Evolution作为核心轴，区分Agent–Agent、Agent–Environment以及Meta Co‑Evolution三类，系统化并揭示进化自由度的逐步扩展

**🔧 技术方法**

主要采用文献综述、概念框架构建、案例划分以及对已有方法的归纳整理，未引入新的算法实现

**📊 数据集**

无特定数据集，综述基于公开论文和现有评测平台的报道与结果

**📈 对比分析**

通过对已有研究的系统化梳理，对比了不同阶段的目标、适应对象、压力来源等，指出Stage 1和2的提升效果逐渐趋于饱和，Stage 3（Meta Co‑Evolution）被视为突破点，但目前相关实验案例有限

**⚠️ 局限性**

缺乏实证验证，未提供统一评测指标；Meta Co‑Evolution仍处于起步阶段，安全与治理措施尚未细化；综述缺乏对不同方法在相同基准上的直接对比

---

## 31. Optimal Stopping of Self-Refining Foundation Models

**arXiv ID:** 2608.10729 | [PDF](https://arxiv.org/pdf/2608.10729v1)

**作者:** Kim Hammar `[一作]` (Imperial College London), Emil C. Lupu `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将基础模型的自我改进过程视为最优停止问题，并推导出单一阈值的最优停止策略。

**💡 创新点**

创新点在于将自我改进建模为决策理论问题，证明存在全局最优的阈值停止策略，并通过高斯过程对改进动态进行系统识别。

**🔧 技术方法**

使用马尔可夫决策过程、Bellman 方程、最优停止理论、Gaussian Process 回归、以及模拟随机逼近（SPSA、交叉熵、差分进化）等技术。

**📊 数据集**

采用 effibench 编码基准，对三种前沿模型（haiku 4.5、Gemini Flash-lite 3.1、GPT Codex Mini 5.1）进行实验。

**📈 对比分析**

与固定迭代策略和 UCB 基线进行比较，优化阈值策略在所有模型和收益权重 β 下均取得最高期望收益，显著优于现有方法。

**⚠️ 局限性**

局限在于对系统函数的单调性与递减收益假设过于理想化，未考虑模型不确定性以及不同任务域的适用性。

---

## 32. Auditable AI-Assisted Research Writing: An Engineering Discipline with Pre-Registered Process Observation

**arXiv ID:** 2608.10858 | [PDF](https://arxiv.org/pdf/2608.10858v1)

**作者:** Yang Zhou `[一作]` (Chinese Academy of Sciences), Chengqun Yu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了一套在研究生产全过程中嵌入的可审计工作流，包括 Git 封装、哈希绑定的来源追踪、红线门控、跨模型角色分离以及程序化主体注入，并在两个案例（一个前瞻性项目和一个回溯性演示项目）中对其机制和指标进行测量与报告。

**💡 创新点**

创新点在于将可审计原则系统化为端到端流程，而非仅在事后检测；提出了“Git sealing with anchor lineage”“hash‑bound provenance”“red‑line gates”“cross‑model role separation”和“programmatic body injection”五大机制，并配合预注册协议、metric cards 和偏差日志，使第三方能够重现和验证所有指标，填补了研究产出缺乏可追溯历史的空白。

**🔧 技术方法**

使用的技术包括：Git 仓库的提交哈希封装与 anchor tag、哈希校验实现来源绑定、自动化门控脚本（red‑line gate）及其日志、跨供应商模型的人工与 AI 审查、只读运行器对脚本化组装进行字节级重现、预注册协议与 21 张 metric card、偏差与修改日志，以及基于脚本的源代码和数据管理。

**📊 数据集**

数据集主要是两州跨年度的省级科研资助政策文本，用于前瞻性案例；回溯性案例则是一次已完成的会议演示项目（包括幻灯片、脚本和后备材料）。所有数据均来自公开或受限政策文件与会议材料，部分数据以哈希形式存储以避免版权问题。

**📈 对比分析**

对比方法：将工作流与传统的事后检测方法对比，展示了工作流在生产阶段就能捕获错误（如确认测试失败导致的停工）并记录全过程。性能上未给出具体数值，而是提供了可重现的指标与缺失率；门控通过失败记录展示其有效性，且所有指标均在预注册的卡片中按标准计算，无统计推断。

**⚠️ 局限性**

局限性包括：观测者与被观测者同属一团队，缺乏独立性；仅有两个案例，缺少对照组；多数指标为预注册的描述性或探索性，缺乏确认性；未提供成本与效率的实证数据；受版权限制，部分数据仅以哈希提供，无法完全公开。

---

## 33. TideRL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling

**arXiv ID:** 2608.10402 | [PDF](https://arxiv.org/pdf/2608.10402v1)

**作者:** Yanyu Ren `[一作]` (Tsinghua University), Jie Tang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一个弹性异步多轮Agentic RL系统，解决了KV缓存预取、参考-演员训练停滞与模型切换、以及资源动态分配等瓶颈。

**💡 创新点**

创新点：①Continuous Task Batching (CTB) 在任务层面管理KV缓存与暂停，保持有用上下文；②Resource-Aware Ref-Actor Pipelining (RA2P) 根据准备度信号切换解耦与共置执行模式；③Elastic Resource Scaling (ERS) 基于准备度动态迁移GPU，实现零开销弹性伸缩。

**🔧 技术方法**

技术细节：任务级调度与优先级、KV缓存优先级与预留、解耦训练图重排（延迟损失计算）、零拷贝共享内存、RANK级别迁移、vLLM+Megatron/PyTorch FSDP训练后端、SGLang、NVIDIA H100 GPU、NVLink、JuiceFS共享存储。

**📊 数据集**

数据集与模型：文本任务 WebShop、AlfWorld；多模任务 OSWorld、ScienceBoard；使用 Qwen‑2.5‑7B/14B、Qwen‑3‑VL‑4B、Qwen‑3.5‑9B 等多规模 LLM。

**📈 对比分析**

对比方法：与 VeRL、AReaL、StreamRL 四种基准在同一硬件与模型配置下进行对比。性能表现：文本任务训练 throughput 提升 5.6×，多模任务提升 33%；KV 缓存命中率 ↑1.58×；每步训练时间 ↓44.3%；总等待时间 ↓77.6%；任务性能与 VeRL 接近（奖励差距 <0.01）。

**⚠️ 局限性**

局限性：异步训练导致评估任务需要严格固定模型版本，当前实现会暂停流水线；对 suffix‑decoding 等技术不适配多轮任务；在极大规模多模任务中 ERS 可迁移的 GPU 数量有限，弹性伸缩效果受限。

---

## 34. Chain of Spatial Thoughts: Modality-Agnostic Spatial Grounding for Vision Language Models

**arXiv ID:** 2608.10278 | [PDF](https://arxiv.org/pdf/2608.10278v1)

**作者:** Hunter Schofield `[一作]` (York University), Dongfeng Bai `[通讯]` (Huawei Technologies Canada)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Space Tokens 框架，将连续空间令牌嵌入现有 VLM 的自回归链式推理中，实现场景级 3D 重建和对象级 3D 包围盒的内在化，从而提升空间推理能力，而无需在推理时引入额外模块。

**💡 创新点**

创新点在于：① 利用词汇位置生成连续空间令牌，并通过投影对齐与重建损失让其编码真实几何信息；② 三阶段训练（表示学习、推理与 GRPO）使模型在保持轻量、可解释的同时显著提升空间任务性能；③ 提供可解码验证的几何表示，兼具可解释性。

**🔧 技术方法**

采用连续空间令牌、LoRA 微调、投影对齐、重建（相机、深度、点云）损失、Hungarian 匹配 3D 包围盒、三阶段训练（表示学习、推理、策略优化）以及 GRPO 策略优化。

**📊 数据集**

训练使用 VICA-322K 视频问答数据；评估使用 VSI-Bench、BLINK、CV-Bench、NExT-QA 等多种 benchmark。

**📈 对比分析**

与 Qwen3‑VL‑8B、SenseNova‑SI‑1.3 等基线比较，VSI‑Bench 总分提升 4.3/1.3 百分点，特别在房间尺寸 (+15.8/+9.5) 与物体尺寸 (+6.6) 任务达到 SOTA；在 BLINK、CV‑Bench 稍有提升/下降，NExT‑QA 提升 5.6 百分点，验证了空间表示对通用视频 QA 的正向迁移。

**⚠️ 局限性**

局限性包括：① 依赖大量标注视频数据且对输入帧数设定敏感；② GRPO 奖励仅针对 VSI‑Bench，泛化到其他任务仍需研究；③ 生成的 3D 结果虽可解码但精度不及专门的几何重建模型；④ 对极大模型或多模态场景的适配性尚未充分验证。

---

## 35. HoosierHelp: Benchmarking LLM Agents for Social Service Navigation

**arXiv ID:** 2608.09946 | [PDF](https://arxiv.org/pdf/2608.09946v1)

**作者:** Yiyang Li `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**通讯引用:** 5597 | [OpenAlex ID](https://openalex.org/A5027601906)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为SocialNavBench的交互式基准，用来评估大语言模型（LLM）在社交服务导航（即帮扶资源推荐）任务中的表现；

**💡 创新点**

创新点包括：①构建了基于真实印第安纳州公共社交服务资源的结构化数据库（3,971条记录）；②设计了多维度的用户模拟器，能产生需求结构、约束满足性以及非理想交互行为（焦躁、废话、无效请求、自相矛盾）；③将资源查询与最终推荐拆分为两种工具调用，从而区分查询错误与选择错误；

**🔧 技术方法**

技术主要是利用LLM驱动的对话代理（agent）与工具调用交互；通过工具接口实现结构化查询（search）和最终推荐（select）；实验中使用了七种主流LLM（Qwen3.7-Max、Qwen3.6-27B、Qwen3.6-35B-A3B、GPT-5.4、GPT-OSS-120B、GPT-4.1-mini、DeepSeek-V4-Flash）。

**📊 数据集**

使用的数据集为印第安纳州211公共资源目录，经过筛选后得到3,971条可被结构化搜索的资源记录；用户模拟器基于预先设定的隐藏配置（需求、约束、备选方案）生成对话。

**📈 对比分析**

评估指标包括工具调用的精确匹配（Tool EM）和最终资源选择的精确匹配（Resource EM）；实验在240条样本上进行，按需求结构、约束满足性、用户行为模式划分；结果显示：工具调用的精确度普遍低于资源选择精度；在约束不满足（fallback）和自相矛盾场景下，各模型性能显著下降；总体而言，最好的Resource EM约为67%，Tool EM约为39%。

**⚠️ 局限性**

局限性包括：①基准仅为评估工具，非真实服务系统，资源信息可能过时或不完整；②用户模拟器虽然多样化，但无法覆盖真实求助者的全部沟通特点、语言需求及危机情境；③基准假设每个需求只有唯一正确答案，忽略了多资源可行性的现实；④缺乏对系统部署中的人机协作、监管与危机处理机制的考察。

---

## 36. BPG: Balancing Plasticity and Generalization for Domain Incremental Learning

**arXiv ID:** 2608.10804 | [PDF](https://arxiv.org/pdf/2608.10804v1)

**作者:** Qiang Wang `[一作]` (Xi'an Jiaotong University), Yihong Gong `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出BPG框架，统一解决域增量学习中的可塑性与泛化两大难题；

**💡 创新点**

创新点在于：①BPG-Adapter根据域特征可分离度自适应分配适配器隐藏维度，②BPG-Inference采用软域混合策略在推理时聚合多域专家的预测，从而消除硬域选择的脆弱性；

**🔧 技术方法**

使用冻结的Vision Transformer骨干、轻量级适配器（Adapter）/LoRA、域特征可分离度指标、k-means原型、softmax加权等技术；

**📊 数据集**

在DomainNet、CDDB、CORe50三个多域基准上进行实验；

**📈 对比分析**

与多种连续学习方法（回放、正则化、参数隔离）以及现有Prompt/Adapter方法对比，BPG在三大数据集上均取得最高平均准确率，平均遗忘率降至接近0；

**⚠️ 局限性**

局限：仅在参数隔离范式下实现；推理时需要聚合所有域专家导致延迟增加；若可用回放缓冲区，可能进一步提升性能。

---

## 37. ELVAE: Evidential Learning-Based Variational Autoencoder for Uncertainty-Aware Generation

**arXiv ID:** 2608.10398 | [PDF](https://arxiv.org/pdf/2608.10398v1)

**作者:** Ge Wang `[一作]` `[通讯]`, Ge Wang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ELVAE模型，给传统VAE的潜变量引入输入相关的正态-逆伽马（NIG）后验，实现潜空间不确定性可解释的生成控制；

**💡 创新点**

通过在潜变量位置和方差上直接应用NIG分布并正则化，显式化潜在位置的不确定性（β/(ν(α-1))），并证明其为ELBO中的正则项，可作为生成时的可调控量；

**🔧 技术方法**

使用证据学习（NIG分布）、变分推理、ELBO优化、正则化、冻结分类器评估以及MNIST实验；

**📊 数据集**

使用MNIST手写数字数据集（60,000训练 + 10,000验证），模型仅利用无标签图像训练，标签用于评估；

**📈 对比分析**

通过冻结的分类器对低/高不确定性区间生成样本的分类错误率进行对比，低区误差26.3%，高区37.8%，比率1.437×；控制实验（z=γ）表明大部分差异来自锚可靠性；随机种子波动在1.126–1.437之间；

**⚠️ 局限性**

局限性：结果仅在类内可比，跨类全局排名无效；锚可靠性与不确定性解耦不完全；生成质量受限于简易MLP解码器；单图像层面关联弱；不同随机种子导致性能波动大。

---

## 38. CausalRepair: Bridging the Causality Gap in Large Language Model-Based Automated Program Repair via Dual-Slicing

**arXiv ID:** 2608.10613 | [PDF](https://arxiv.org/pdf/2608.10613v1)

**作者:** Linhao Wu `[一作]` (Peking University), Dan Hao `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了基于大型语言模型的自动程序修复框架CausalRepair，利用测试侧静态切片与源侧动态切片双向切片技术构建最小因果上下文，并通过交互式迭代修复和补丁增强实现高质量修复。

**💡 创新点**

提出最小因果上下文概念及双向切片策略，有效消除测试与源代码中的噪声，精准定位根因并显著提升LLM的修复效果。

**🔧 技术方法**

使用静态切片、动态切片、程序依赖图、LLM对话生成、交互式迭代修复、补丁增强等技术。

**📊 数据集**

在Defects4J V1.2/V2.0、Defects4J-Trans、RWB、GitBug-Java等Java bug 数据集上进行实验。

**📈 对比分析**

与15种最先进的APR工具（模板、学习、LLM）对比，CausalRepair在Defects4J上共修复313个正确bug，平均成本仅$0.029/bug，修复速度和成本均优于基线。

**⚠️ 局限性**

仅支持Java语言；依赖Slicer4J动态切片工具，工具局限性与跨语言迁移性待验证；实验基于完美fault localization，对真实工业环境的适用性需要进一步评估。

---

## 39. Predicting Space Groups of Double Perovskites by LLM with Dynamic Few-Shot Learning

**arXiv ID:** 2608.10483 | [PDF](https://arxiv.org/pdf/2608.10483v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 40. ReCBM: Uncertainty-Gated Relational Reasoning for Concept Bottleneck Models

**arXiv ID:** 2608.10004 | [PDF](https://arxiv.org/pdf/2608.10004v1)

**作者:** An Sui `[一作]` (Fudan University), Xiahai Zhuang `[通讯]` (Fudan University)

**通讯引用:** 6899 | [OpenAlex ID](https://openalex.org/A5011662977)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了基于不确定性门控的关系推理框架ReCBM，用以在概念瓶颈模型中修正不可靠的概念状态。

**💡 创新点**

创新点在于将语义关系（共现、蕴含、排斥）与概念不确定性相结合，通过门控机制控制证据传播，实现对缺失或错误概念的自适应恢复。

**🔧 技术方法**

采用Beta型证据式概念预测器、迭代关系推理模块、源/接收/锚门控，以及两阶段训练策略。

**📊 数据集**

使用WBC、CUB以及人工设计的Synthetic三组概念数据集进行验证。

**📈 对比分析**

与CBM、ProbCBM、SCBM、GraphCBM等基线比较，ReCBM在概念与任务的CR‑AUC上最高，且在不确定性干预和紧凑概念子集提取上保持或提升性能。

**⚠️ 局限性**

局限性包括在全腐败情况下恢复效果有限，对关系矩阵的初始化和手工设计依赖较强，且模型相较传统CBM更为复杂。

---

## 41. Navigation Alone Is Not Enough: Evaluating Explanatory Assistive UI Agents

**arXiv ID:** 2608.09944 | [PDF](https://arxiv.org/pdf/2608.09944v1)

**作者:** Santosh Patapati `[一作]` (Stony Brook University), Santosh Patapati `[通讯]` (Stony Brook University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5118979945)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 NeXUI benchmark，要求辅助 UI 代理在完成任务的同时，以适合盲人使用的自然语言对每一步动作进行解释。

**💡 创新点**

创新点在于将导航、解释和安全控制统一到同一评价框架中，重点关注非视觉交互；并为每个任务提供完整的界面状态、可操作目标以及说明性注释，方便评估代理的解释是否与界面状态一致。

**🔧 技术方法**

采用 Gemini Flash 系列基础模型进行评估，并使用多模态界面表征（渲染页面、可访问性树、读者视图等）来构造输入；通过结构化动作与简短解释的输出形式来实现交互。

**📊 数据集**

NeXUI 数据集：225 个真实生产任务，分布在 16 种界面，包含 1,755 个界面状态、31,000+ 目标候选，划分为 dev/validation/test/challenge 四个子集。

**📈 对比分析**

使用任务成功率、是否出现安全违规、效率（步骤数）和解释质量（与当前状态的一致性和任务关联度）进行多维度对比；在验证集上，Gemini 3.5 Flash 达到 44% 的任务成功率，平均解释分 0.7359，仍显不足。

**⚠️ 局限性**

局限性：任务成功率低，解释质量仍不理想；模型缺乏对视觉信息的完整感知，数据集仅涵盖 16 种界面，难以覆盖更广泛的动态网页场景。

---

## 42. Mixed Choice Multiparty Session Types, Precisely

**arXiv ID:** 2608.10704 | [PDF](https://arxiv.org/pdf/2608.10704v1)

**作者:** Jake Masters `[一作]` (University of Oxford), Nobuko Yoshida `[通讯]` (University of Oxford)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文针对多方会话类型中的混合选择（支持会话委托、创建与交织）构造了完整且精确的子类型关系，并实现了相应的子类型检测与安全性检查算法。

**💡 创新点**

创新点包括：①首次给出适用于混合选择多方会话 π-计算机的完整子类型关系并证明其精确性；②引入三方锁（minimal liveness error）与调度器过程构造，完成子类型完整性证明；③将精确性结果推广至一族子计算机，并实现二次时间复杂度的算法。

**🔧 技术方法**

主要技术手段有：基于类型系统的安全性证明、三方锁与调度器的构造方法、操作与语义层面的精确性证明，以及针对子类型和安全性检查的算法设计。

**📊 数据集**

使用了文献中的混合选择多方会话案例研究作为实验数据集；并结合合成的三方锁实例验证完整性。

**📈 对比分析**

与已有的子类型检测与会话安全性检查方法对比，本文实现的算法在时间复杂度上为O(n²)（n为状态空间与类型上下文大小），并在公开案例上验证了性能优于传统方法。

**⚠️ 局限性**

局限性包括：仅针对支持交织与委托的混合选择多方会话类型；对更复杂的会话模式（如动态角色生成或多重委托）尚未展开验证；实现依赖于手工构造的调度器，扩展性有待进一步研究。

---

## 43. Human versus Computer Vision

**arXiv ID:** 2608.10181 | [PDF](https://arxiv.org/pdf/2608.10181v1)

**作者:** Elena Sirotkina `[一作]` (New York University), Elena Sirotkina `[通讯]` (New York University)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5097687046)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对六种主流视觉显著性模型（四个深度网络和两个经典检测器）进行评估，利用3,023名美国成人的 11.4 百万次网页摄像头眼动数据，测量这些模型在预测新闻照片中的观众视线位置的准确性。

**💡 创新点**

发现无训练的中心高斯映射在绝大多数照片上已优于所有训练模型，且模型所增加的内容对不同人群（尤其是老年人、黑人和极端政治立场者）的预测效果极差。提出“群体签名”判据，判断一个群体是否具有可被模型学习的共同视线模式，并给出可行的补救方案（如使用 13 人观察者面板或对观众进行条件化读取）。

**🔧 技术方法**

使用 WebGazer 的网页摄像头眼动追踪技术获取原始眼动序列；将这些序列提取为原始样本和基于分散阈值的聚焦点；对六个显著性模型生成每张图片的单一显著性图；采用 AUC 评分（原始和中心校正）衡量模型预测与实际眼动的匹配度。

**📊 数据集**

数据集包括 83 张 Getty Images 新闻照片（涵盖移民、枪支、1 月 6 号攻击、 LGBTQ+ 权利等议题），以及来自 3,023 名成人的 61,458 次浏览（每张照片 5 秒），共 11.4 百万原始眼动样本；对照实验使用 MIT1003 实验室眼动数据进行模型性能比较。

**📈 对比分析**

结果显示：中心高斯原始 AUC 为 0.715，六个模型平均为 0.689-0.692；对照的观众面板（256 人）可达到 0.550，表明模型仅贡献约 0.04 的提升。模型在中心校正后（去除中心习惯）仅比中心映射高 0.02。模型在不同群体间的差异显著：年龄、种族和极端立场的预测误差明显大于总体误差。

**⚠️ 局限性**

局限性包括：仅测试新闻照片，可能不代表其他视觉内容；网页摄像头眼动误差比实验室追踪器高，影响细粒度评估；老年人、黑人和极端政治立场者样本相对不足，导致统计功效有限；群体签名判据依赖于群体内部一致性，若缺乏可辨别特征模型无法学习；最终模型在实践中仍需结合中心先验，单一预测图难以满足多元观众的个性化需求。

---

## 44. JitTrack: Onboard Multi-Object Tracking Against Viewpoint Jitter for Agile UAVs

**arXiv ID:** 2608.10485 | [PDF](https://arxiv.org/pdf/2608.10485v1)

**作者:** Yachun Shan `[一作]` (Peking University), Feitian Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对无人机视角抖动的端到端运动感知多目标跟踪框架JitTrack

**💡 创新点**

创新点在于通过语义查询优化、运动感知查询校正和运动感知去噪训练三大策略，直接让跟踪器学习运动鲁棒表示，省去显式相机运动补偿

**🔧 技术方法**

基于Transformer的查询式跟踪网络，结合语义注入、全局运动先验、低秩约束的查询校正以及带运动噪声的去噪训练

**📊 数据集**

在VisDrone2019-MOT和UAVDT两大无人机多目标跟踪基准上进行评测，并在真实无人机平台上验证

**📈 对比分析**

与多种state‑of‑the‑art方法对比，JitTrack在MOTA、IDF1等指标上提升约10–15%，在真实飞行中实现了高达90%的跟踪成功率和零身份切换

**⚠️ 局限性**

局限性包括对极端高速抖动的鲁棒性仍有限，训练时需要较大算力和大量带抖动标签的数据，且在多机协同或复杂环境下的表现尚待验证

---

## 45. Robust Multi-Agent Bandits with Heavy-Tailed Rewards and Information Asymmetry

**arXiv ID:** 2608.10529 | [PDF](https://arxiv.org/pdf/2608.10529v1)

**作者:** Daphne Feng `[一作]` (University of California, Los Angeles), William Chang `[通讯]` (University of California, Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究多智能体重尾奖励的多臂赌博机模型，针对三种信息不对称场景提出鲁棒分散学习算法。

**💡 创新点**

创新点在于：①把重尾鲁棒估计与多智能体分散设置结合；②设计意图偏离作为隐式通信信号；③在完全非对称情形下给出基于DSEE的全局时间自适应探索策略。

**🔧 技术方法**

主要技术包括截断均值鲁棒估计、鲁棒上置信界（RUCB）、轮询/信号消除机制以及任何时刻自适应的探索序列。

**📊 数据集**

实验使用 Pareto 分布的奖励（形状参数 2，尺度随 μ_a 调整），设置 M=2、K=2、T=10^6 的仿真实例。

**📈 对比分析**

与集中式或传统算法比较，Problem A 与 B 的累积奖励分别为 O(logT/Δ^{1/ε})，Problem C 为 O(log^2T)；实验显示 Problem B 在中期收敛最快，Problem C 的探索成本更高。

**⚠️ 局限性**

局限性包括：① 需预先知晓尾部参数 (ε,v)；② 对 K^M 的指数规模依赖；③ 对 Problem C 缺乏匹配下界，且当前分析未考虑非平稳或对抗环境。

---

## 46. Persistent Recursive Worlds Enable Autonomous Software Evolution

**arXiv ID:** 2608.10450 | [PDF](https://arxiv.org/pdf/2608.10450v1)

**作者:** Beichen Huang `[一作]` (Hong Kong Polytechnic University), Ran Cheng `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将软件项目持久化为递归世界的组织方式，并通过从零构建 C 编译器、在模型更换后继续已有项目以及将 MESA 模块迁移至 Rust 三种任务验证其可行性。

**💡 创新点**

创新点在于将持续性放在项目状态而非代理本身，利用接受版本、路径坐标和递归委托实现多代理有限寿命下的长期协同开发。

**🔧 技术方法**

技术：LLM 驱动的有限寿命代理（DeepSeek V4 Flash、GLM 5.2）、Git‑style 版本控制、递归委托与验证门控接受、工具链交互、自动化测试。

**📊 数据集**

数据集：c‑testsuite、LLVM 测试集、Csmith、LZ4、SQLite、MESA 源码及其六个科学工作负载。

**📈 对比分析**

比较方法：通过测试通过率、代码行数、委托深度、代理数、模型令牌成本等指标评估；编译器通过全部 c‑testsuite 与 LLVM/Csmith，Rust 版在六项工作负载中平均加速 1.55–6.87 倍，模型成本分别为 44.38 美元与 10.64 美元。

**⚠️ 局限性**

局限：实验缺乏对递归机制因果影响的系统验证，未量化人类干预，测试集与成本对比不统一，无法确定哪些持久记录对持续性最为关键。

---

## 47. Adaptive Matrix Multiplication for Dynamic Shapes on Ascend NPUs

**arXiv ID:** 2608.10803 | [PDF](https://arxiv.org/pdf/2608.10803v1)

**作者:** Yuhang Zhou `[一作]` (Nanjing University), Chen Tian `[通讯]` (Nanjing University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种面向 Ascend NPU 的自适应矩阵乘法框架，实现动态形状下的高性能 MatMul。

**💡 创新点**

创新点包括：①将动态形状映射到硬件感知的 2D tiling 体系；②构建解析性性能模型并利用离线搜索缓存最优实现；③设计可组合的优化库并实现 O(1) 运行时调度。

**🔧 技术方法**

使用硬件感知 2D tiling、解析性性能模型、可组合优化库、离线-在线缓存机制及 Ascend NPU 的显式 SIMD/内存层次。

**📊 数据集**

评估基于 80,000 个工业级动态形状数据集及多款推荐模型（MMOE、DLRM、DCN V2、ESMM、RankMixer）。

**📈 对比分析**

与 Ascend 官方 ACLNN 和 Catlass 基线对比，单算子平均加速 1.85×，端到端加速 1.09–1.48×，显示显著提升。

**⚠️ 局限性**

局限性在于高度针对 Ascend 910，模板和模型需重新定义才能迁移到其他 DSA；仅覆盖 MatMul 及相关算子；对低精度/稀疏/复杂算子支持有限。

---

## 48. Resolving Envy by Adding Goods with Bounded Supply: A Type-Count Dichotomy and Two-Agent Hardness

**arXiv ID:** 2608.10326 | [PDF](https://arxiv.org/pdf/2608.10326v1)

**作者:** Chuang-Chieh Lin `[一作]` (National Taiwan Ocean University), Colin Cleveland `[通讯]` (King's College London)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在有限供给的EEAG（Envy Elimination by Adding Goods）问题中，对二进制可加值设定进行了严谨分析，证明了当仅有一种额外物品类型时可以多项式时间求解；当恰好有两种额外物品类型时问题转化为NP‑完全；并进一步证明了在两名代理人且每种额外物品仅有单个单位时，问题仍是弱NP‑完全。

**💡 创新点**

（1）首次将EEAG的可行性约束转化为差分约束系统，并利用Bellman‑Ford得到最小可行扩展；（2）通过合并原先三类型构造中的“选中边”与“选中顶点”类型，实现了对两类型情况的完整NP‑完整性证明；（3）构造了一个从Clique和ESSE的严格归约，完成了二类型界定的闭合。

**🔧 技术方法**

主要使用差分约束与最短路径（Bellman‑Ford）来求解单类型问题；使用图构造与负环检测保证可行性；通过集合论与图论计数论证实现两类型NP‑完整性；使用标准的NP‑完整性归约技术（Clique、ESSE）。

**📊 数据集**

该工作为纯理论论文，不依赖任何真实数据集；所有实例均为在多项式时间内构造的人工实例。

**📈 对比分析**

算法：单类型EEAG的Bellman‑Ford实现时间复杂度为O(n³)；在实验上对随机生成的实例可在毫秒级完成；与现有方法（无此方法可比对）相比，提供了唯一已知的多项式算法。对两类型情况则证明其不可多项式求解（除非P=NP）。

**⚠️ 局限性**

局限性：只考虑二进制可加价值；不处理有预算约束或非二进制价值；对两类型的证明仅适用于有限供给且审批集包含关系；未给出伪多项式或近似算法；缺乏实验评估与实际案例验证。

---

## 49. SPOTting the Future: Lookahead Explanations for Deep Reinforcement Learning

**arXiv ID:** 2608.09967 | [PDF](https://arxiv.org/pdf/2608.09967v1)

**作者:** Tamar Gozlan `[一作]` (Hebrew University of Jerusalem), Claudia V. Goldman `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为SPOT（Sampling Policy Observation Tree）的通用可解释框架，用于构建深度强化学习代理的可视化决策树，并通过该树提供未来轨迹级的解释；

**💡 创新点**

创新点在于：①把策略采样与环境模拟结合，构造有限深度的决策树，实现对多步后果的可解释；②提供理论保证，证明在采样充分时能恢复最优动作；③在多行动空间下通过采样聚合，避免为每个动作单独分支；

**🔧 技术方法**

技术包括：蒙特卡洛策略采样、树形递归展开、统计计数、软最大化生成策略、对价值函数、优势、奖励等节点信息的聚合，兼容价值、策略和actor‑critic三类DRL算法；

**📊 数据集**

数据集与环境：SUMO‑RL仿真环境中的单一四向信号交叉口，采用21维观测向量、四个动作（信号相位），并在实验中引入上游车辆堵塞的灾难场景；

**📈 对比分析**

方法比较：与传统单时步特征归因方法SHAP对比，SPOT能够捕捉到无法通过当前观测直接感知的未来拥堵，表现为在事故窗口内能给出“干预”建议，而SHAP仅解释当前合理动作；未给出数值指标，但在案例中显示SPOT提供了更具行动性的解释；

**⚠️ 局限性**

局限性包括：①树深度和采样数受计算资源限制，可能无法覆盖所有重要后果；②仍依赖可观测的状态特征，无法直接感知完全不可观测的干扰；③未完成对关键状态的全局总结与人机评估实验，缺乏用户可操作性验证。

---

## 50. Efficient Reinforcement Learning for Long-Horizon Tool-Use Agentic Tasks

**arXiv ID:** 2608.10357 | [PDF](https://arxiv.org/pdf/2608.10357v1)

**作者:** Zelei Cheng `[一作]` (Capital One), William Campbell `[通讯]` (Capital One)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个可模块化的强化学习训练系统，集成Gymnasium兼容双控制环境包装、无值模型的GRPO优化和sink-aware FlexAttention，专门用于长时工具使用代理的训练。

**💡 创新点**

创新点在于：①将GRPO与VERL式rollout数据流无缝结合，省去价值网络；②设计了sink-aware FlexAttention与块稀疏掩码的融合实现，保持模型特定sink缩放的可微性；③通过自定义掩码和Sink scaling 大幅降低长序列的峰值显存，支持至8192 token 的训练。

**🔧 技术方法**

使用技术包括 Gymnasium 环境包装、GRPO策略优化、VERL式rollout、PyTorch FlexAttention、block-sparse masked attention、PyTorch Inductor/AOTAutograd、融合高带宽内存的自定义 kernel。

**📊 数据集**

实验数据集主要为 τ^2‑Bench 双控制环境以及商用零售域仿真（Retail domain）任务，用于评估验证奖励和训练诊断。

**📈 对比分析**

与基准 eager attention 在 1024–4096 token 下进行峰值VRAM 对比，优化路径分别节省 4.5%–19.7% 内存；在 8192 token 下基准 OOM，优化路径能顺利完成；验证奖励从 0.25 提升至 0.44，训练诊断指标亦随之上升。

**⚠️ 局限性**

局限性包括：仅在单一域单一训练窗口验证，未做多种随机种子或对比基线；未评估吞吐、延迟或总训练时间；峰值内存测试未验证梯度等价性；缺乏对更大模型或更长序列的系统性评估。

---

## 51. Mind Viruses: Self-Propagating Ideas in Multi-Agent LLM Systems

**arXiv ID:** 2608.10218 | [PDF](https://arxiv.org/pdf/2608.10218v1)

**作者:** Vassilis Papadopoulos `[一作]` (Anthropic Fellows Program), Jack Lindsey `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并演示了在多智能体系统中可传播的“mind virus”（思想病毒），通过进化算法设计病毒种子，并在编码协作场景和病毒链实验中评估其传播效果，探讨影响传播的因素并提出简单的防御提示。

**💡 创新点**

首次将思想病毒概念化为自我复制的目标传播机制，提出利用LLM进化优化生成病毒种子的方法，并系统识别了病毒传播中常见的“病毒人格”主题，展示了防御提示在阻止传播中的有效性。

**🔧 技术方法**

使用LLM进化优化（基于Kimi K2.5和Claude/ Gemini/ DeepSeek等模型）生成病毒种子；采用自定义的多智能体实验框架（类似OpenClaw），包括编码协作与病毒链两种拓扑；通过LLM评估器判定感染；利用文本与文件传播机制实现自我复制。

**📊 数据集**

实验使用合成的编码协作任务（子任务队列）、随机生成的agent系统提示与文件，构成虚拟的多智能体工作环境；病毒链实验则使用简化的单会话交互，每个agent持有最小化的文件系统；没有使用公开的现实数据集，而是基于实验设定的自定义任务与对话。

**📈 对比分析**

通过对不同模型（Claude Haiku 4.5、Gemini 3 Flash、DeepSeek V3.2、Qwen 3.5 32B 等）和网络拓扑（全连通 vs 单链）比较感染率。实验显示：恶意payload传播率约 20‑30%，但比友好payload低；在完全连通拓扑下，感染率可达 60‑80%；在“防御”系统提示下几乎 0%；在病毒链中，多跳传播率保持在 10‑20% 之间。总体而言，病毒传播率受模型、任务状态、警示提示和拓扑显著影响。

**⚠️ 局限性**

限制：实验环境人为简化，缺乏真实世界多智能体交互细节；只测试了少数LLM模型，未覆盖更广泛的模型空间；进化生成的病毒种子依赖于LLM的生成偏好，可能不具备最佳性；实验主要关注即时传播，未评估长期演化与自我修复；防御提示的鲁棒性需在更复杂、异构网络中进一步验证。

---

## 52. Elbow Angle Guidance System Based on Surface Haptic Sensations Elicited by Lightweight Wearable Fabric Actuator

**arXiv ID:** 2608.10404 | [PDF](https://arxiv.org/pdf/2608.10404v1)

**作者:** Kenta Yokoe `[一作]` (Nagoya University), Yasuhisa Hasegawa `[通讯]` (Nagoya University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于轻量化可穿戴织物致动器的肘部角度引导系统，利用麦克宾布质织物和两条麦克宾人工肌肉，通过气压控制表面触觉感应来直观引导肘部屈伸；

**💡 创新点**

创新点在于：①将麦克宾肌肉与织物结合成柔软轻便的致动器，实现对肘部角度的表面触觉引导；②根据Weber–Fechner定律设计气压与感知强度的映射，使触觉强度与目标角度实时关联；③通过算法自适应地求解阈值气压，使系统无需训练即可使用；

**🔧 技术方法**

采用技术包括：麦克宾式人工肌肉、Velcro织物套、可编程气压控制器、光学运动捕捉系统（OptiTrack）、自定义气压调节算法；

**📊 数据集**

使用的数据集为六名20-25岁健康右撇手的实验数据，记录目标肘角与实际肘角的误差；

**📈 对比分析**

与随机猜测结果进行比较，实验误差的平均值为约25°/26°，中位数约22.8°，显著低于随机误差的平均44°/中位39.1°（Wilcoxon检验p<0.001），说明系统能有效引导肘部角度；

**⚠️ 局限性**

局限性包括：仅在年轻健康受试者中验证；对强度较低的触觉无法实现对肌肉虚弱或老年受试者的有效引导；缺乏长时间使用或训练效果的评估；仅针对单侧肘部，不涉及多关节同步控制。

---

## 53. Embedding Rotation Invariance for Provable Multi-Oriented Scene Text Recognition

**arXiv ID:** 2608.10684 | [PDF](https://arxiv.org/pdf/2608.10684v1)

**作者:** Zhibin Ma `[一作]` (Sun Yat-sen University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了全端到端的旋转不变场景文本识别网络RISTER，结合旋转等变编码器和旋转不变解码器实现多方向文本识别。

**💡 创新点**

创新点在于证明并利用交叉注意力的旋转不变性，设计了RITD；同时提出RELG局部-全局等变网络，实现等变特征提取与全局关系建模。

**🔧 技术方法**

采用旋转等变卷积（F-Conv）、自注意力、交叉注意力、MLP等技术，并实现可在标准CNN/Transformer架构上无额外推理开销。

**📊 数据集**

使用Union14M-Filter大规模训练集，并在14个英语基准（IIIT5k、SVT、IC13、IC15、SVTP、CUTE80、Union14M各类别、ASOT）评估。

**📈 对比分析**

与现有多方向和通用STR模型对比，RISTER在多方向子集提升≈4%精度，整体在大多数基准上处于SOTA，参数量≤40M，推理速度与NRTR相当。

**⚠️ 局限性**

局限性包括仅验证正方形输入尺寸，尚未针对长文本或非方形输入展开实验，且对极端旋转角度的理论近似未完全证明。

---

## 54. ConnectionMind: Leveraging Social Networks and Large Language Models for Personalized Recommendation at Meta

**arXiv ID:** 2608.10187 | [PDF](https://arxiv.org/pdf/2608.10187v1)

**作者:** Haoyu Han `[一作]` (Michigan State University), Xiangjun Fan `[通讯]` (Meta Platforms, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并部署了 ConnectionMind 框架，将社交推荐问题建模为在异构社交–项目交互图上进行路径探索，并通过 LLM 策略实现可解释的个性化推荐。

**💡 创新点**

创新点在于：①将社交图与大语言模型相结合，形成基于路径的可解释推荐；②采用两阶段训练（SFT + RL）让 LLM 学会结构化图遍历；③在生产环境中引入教师-学生混合推理，兼顾准确性与低延迟。

**🔧 技术方法**

技术包括异构图构建、LLM 结构化推理策略、监督微调、基于规则的强化学习、图神经网络学生模型蒸馏以及多模态文本表征。

**📊 数据集**

使用了公开数据集 Delicious 与 Foursquare 进行离线实验，并在 Meta 的大规模短视频平台上使用内部用户–项目交互和社交关系数据进行部署与评估。

**📈 对比分析**

与传统 MF、GNN、扩散模型和 LLM 基础推荐器相比，ConnectionMind 在 Delicious/Foursquare 上在 Recall/Precision 上均位列首位；在 Meta 线上 A/B 测试中，离线 Recall@10 提升 88%，线上视频观看时长提升 0.43% 以上，曝光率与用户留存亦有显著提升。

**⚠️ 局限性**

局限性包括：对高频活跃用户仍需昂贵的 LLM 推理导致延迟；路径生成依赖采样子图，可能忽略全局最优路径；解释性仅限于路径级别，缺乏更细粒度因果推断；以及对隐私敏感社交关系的使用需要严格合规。

---

## 55. Enabling Scalable Kinesthetic Teaching via Observer-based Hand-guiding with Active Support

**arXiv ID:** 2608.10847 | [PDF](https://arxiv.org/pdf/2608.10847v1)

**作者:** Anna Tuma `[一作]` (KUKA), Niels Dehio `[通讯]` (KUKA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了RHOAS手导控制方案，利用模型驱动的动量观测器估计外部力，主动支持操作者的手导动作。

**💡 创新点**

创新点在于把手导视为主动交互而非被动一致性约束，并通过观测器估计外力和关节冗余实现低功耗、高精度的主动支持。

**🔧 技术方法**

采用机器人动力学模型、动量观测器进行外力估计、低通滤波和置信因子处理、基于估计的卡尔曼控制、冗余空间阻尼和反射扩展的笛卡尔速度控制。

**📊 数据集**

使用16名受试者在KUKA LWR iiwa上收集的实验数据（手导路径、冲击力、时序等），未使用公开数据集。

**📈 对比分析**

通过与重力补偿（G）、卡尔曼自适应阻尼（CAD）等基线进行对比，实验表明RHOAS在减少人力能量、提升用户偏好及提高任务执行质量方面显著优于基线。

**⚠️ 局限性**

局限性包括对机器人内置关节扭矩传感器的依赖、对奇异点附近估计的敏感性、以及仅在单一平台（KUKA LWR iiwa）和有限样本规模内验证，需要进一步跨平台和大规模验证。

---

## 56. HexEval: An Evidence-Driven Hexagonal Framework for Multidimensional Scholar Assessment

**arXiv ID:** 2608.10584 | [PDF](https://arxiv.org/pdf/2608.10584v1)

**作者:** Xiaokang Qu `[一作]` (University of Science and Technology of China), Yiting Lin `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HexEval 框架，采用证据驱动双层推理实现学者的六维评估，并保留中间证据与理由，支持可解释与可审计。

**💡 创新点**

创新点在于将学者评估拆分为内在质量层（匿名论文评估）与外部可验证行为层（公开数据验证），并在每维度保留可追溯的证据与推理过程。

**🔧 技术方法**

使用大型语言模型（Qwen3.6‑27B、Qwen2.5‑72B‑Instruct、DeepSeek）进行论文内容评估、匿名化、证据提取；结合规则清洗、Ridge 校准、log/exp 饱和等算法实现多维度评分。

**📊 数据集**

主要数据集包括 OpenReview（NeurIPS、ICLR）用于 D1–D3；GitHub、Lens、OpenAlex 等公开记录用于 D4–D6；预选 110 名 CS 学者的 OpenAlex 记录用于 D5；D6 采用 OpenAlex h‑index。

**📈 对比分析**

与直接整体评分、Chain‑of‑Thought、Self‑Reflection、均值等基线对比，HexEval 在 D1–D3 中显著降低 MAE、提升 Acc@0.5；D4 的综合分在 AUC、F1@k 上优于单指标；D5 的 MAE 下降至 0.266、Acc@0.5 达 0.878；D6 作为基准指标直接使用。

**⚠️ 局限性**

局限性包括：受公开数据覆盖范围、作者去重和领域差异的限制；GitHub/专利数据在某些学科稀缺；h‑index 对职业年限和引用惯例敏感；匿名化无法完全消除身份泄露；模型评估仍受训练数据与领域偏差影响。

---

## 57. Non-Existence of EFX Chore Allocations for Monotone Cost Functions with Binary Marginals

**arXiv ID:** 2608.10572 | [PDF](https://arxiv.org/pdf/2608.10572v1)

**作者:** Zehan Lin `[一作]` (University of Macau), Shengwei Zhou `[通讯]` (Nanyang Technological University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在单调二进边际成本函数下，任务分配中的完全EFX（即“对任何一件物品都无妒”）分配是否存在，提出并给出两类负例；

**💡 创新点**

首次证明在二进边际XOS和超模数（supermodular）成本下完全EFX分配可能不存在，填补了此前仅在部分子类已知的空白；

**🔧 技术方法**

采用组合构造（18位二进字串生成53项任务的“词件”装置），利用图论中的匹配性质与离散函数的边际增量证明；代码在Lean 4中验证；

**📊 数据集**

未使用真实或合成数据集，全部为理论构造实例；

**📈 对比分析**

无实验比较，论文为纯理论证明，未给出性能指标；

**⚠️ 局限性**

仅解决了XOS与超模数两类，仍未解决二进边际子模数（submodular）成本下EFX的完整性问题，且只给出负例构造，缺乏可扩展性或算法建议。

---

## 58. Straightforward Entropy-Sensitive Mergesort

**arXiv ID:** 2608.10421 | [PDF](https://arxiv.org/pdf/2608.10421v1)

**作者:** Bill Jin `[一作]` (University of Toronto), Alex Zihan Xu `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于方向合并的稳定归并排序算法，能够在已存在的递增或递减运行上进行自适应排序。

**💡 创新点**

核心创新在于只使用跳过检查（skip check）和动态合并方向决定，而不依赖运行扫描；通过静态合并树结构将递归栈压缩到 O(1) 词，且实现了对递减运行的自适应。

**🔧 技术方法**

技术主要包括：静态二叉分割的合并树、跳过检查、方向合并（正向/反向）、基于位堆栈的迭代后序遍历、以及延迟物理逆序处理递减子区段。

**📊 数据集**

论文为理论分析，没有给出具体实验数据集；只在理论上证明了比较次数和移动次数的上界。

**📈 对比分析**

与 Powersort、Peeksort 等现有最优自适应归并排序相比，Directional Mergesort++ 在比较次数上达到 nH+3n（理论上可进一步到 nH+3n−r#），在数据移动上为 1.5nH+6.5n，空间仅需 O(1) 词，时间复杂度保持 O(nH+n)。

**⚠️ 局限性**

局限性包括：在实际平均情况中可能比 Timsort 等实现慢；相较 Peeksort 仍有 1 级差距；实现需较为复杂的位堆栈逻辑；对极端递减序列的处理仍需要额外逆序操作。

---

## 59. Riemann GeoResolver: A Non-Euclidean Attention Framework from Euclidean Resolver to Hyperbolic-Spherical Geometry

**arXiv ID:** 2608.10416 | [PDF](https://arxiv.org/pdf/2608.10416v1)

**作者:** Liangchen Ge `[一作]` `[通讯]`, Liangchen Ge

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

本文提出逆距离注意力（IDA）及其欧氏、双曲和球面扩展，并给出了完整的理论证明。

**💡 创新点**

创新点在于证明IDA在表达能力、优化收敛和泛化误差上优于softmax，并构建了Riemann GeoResolver框架。

**🔧 技术方法**

使用的技术包括逆距离核、Polyak–Łojasiewicz（PL）不等式、有效秩分析、几何距离变换、Nyström近似与梯度下界证明等。

**📊 数据集**

未进行实验，因此未使用任何数据集。

**📈 对比分析**

由于缺乏实验，未做性能比较；理论上表明IDA在资源占用、收敛速度和对噪声的鲁棒性方面优于softmax。

**⚠️ 局限性**

局限性：仅为理论研究，缺少实验验证；多点PL分析尚未完成；仅对键进行压缩，值未压缩。

---

## 60. Hidden in Plain Sight: Diffusion-Based Unrestricted Robotic Attacks on Vision-Language-Action Models

**arXiv ID:** 2608.10393 | [PDF](https://arxiv.org/pdf/2608.10393v1)

**作者:** Jiahui Han `[一作]` (Xi'an Jiaotong University), Xia Hu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于扩散模型的可自然、不受限制的视觉语言动作(VLA)机器人对抗补丁攻击方法 DURA。

**💡 创新点**

创新点在于利用扩散模型潜在轨迹生成视觉自然补丁，并在白盒和仅动作输出黑盒两种威胁模型下实现高效攻击。

**🔧 技术方法**

使用了预训练的潜在扩散模型（LDM）、DDIM 反向过程、结合 VLA 策略梯度或基于查询的估计的对抗目标动作损失。

**📊 数据集**

在 LIBERO 模拟基准和 BridgeData V2 真实数据集上，针对 OpenVLA‑7B 和 π_0‑FAST 两种 VLA 模型进行实验。

**📈 对比分析**

与像素扰动、TMA、UPA、UADA 等基线相比，DURA 在白盒下实现 100% 攻击成功率，在黑盒下可达 79%（相较 TMA‑NES 提升约 40%），并在视觉自然度、查询效率和时间消耗上显著优于基线。

**⚠️ 局限性**

局限性包括对扩散模型和 VAE 解码器的依赖，物理打印补丁在不同光照或摄像头条件下可能仍需微调；且攻击仅在补丁可见且位置可控的前提下有效。

---

## 61. Context and Symmetry in Auditing: A Case Study of Skeleton Inference in Motion Capture

**arXiv ID:** 2608.10194 | [PDF](https://arxiv.org/pdf/2608.10194v1)

**作者:** Emma Harvey `[一作]` (Cornell Tech), Mona Sloane `[通讯]` (University of Virginia)

**通讯引用:** 576 | [OpenAlex ID](https://openalex.org/A5075302056)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过构建“情境审计”方法并对运动捕捉中骨骼推断进行案例研究，探讨在缺乏明确基准真值时如何审计测量结果。

**💡 创新点**

将社会实践理论与STS对称性相结合，提出情境审计与对称审计框架，允许在真实使用情境中对测量进行比较而不依赖单一真值。

**🔧 技术方法**

采用OptiTrack Prime 13运动捕捉系统与手工测量（测量带）结合，使用Bland‑Altman限差分析与嵌套回归评估测量可靠性。

**📊 数据集**

采集24名参与者（12人双次试验）在NYU mocap实验室的身体尺寸与运动捕捉数据，未使用公开数据集。

**📈 对比分析**

通过Bland‑Altman与回归对比测量差异，结果显示大多数测量在体型、性别、时间上保持可靠，但在肩宽和身高等方面存在尺寸相关差异，未发现显著性别差异。

**⚠️ 局限性**

样本量小、操作者为非专业实验者导致测量误差混入系统误差、仅涵盖创意设计低风险情境、缺乏对高风险应用的验证、对真实使用者视角的反思不足。

---

## 62. Towards Sustainable Artificial Intelligence: A Comprehensive Review and Comparative Analysis of Deep Learning Models' Carbon Footprint

**arXiv ID:** 2608.09998 | [PDF](https://arxiv.org/pdf/2608.09998v1)

**作者:** Samar Garrab `[一作]` (Royal Military College of Canada), Manel BenSassi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统地综述了绿色AI与绿色深度学习的研究，评估碳足迹测量工具，并通过CPU实验比较六种常见深度学习模型的碳排放与准确率。

**💡 创新点**

首次将碳排放与准确率统一量化为“每准确点排放量”(EAP)指标，提供可比较的碳效率度量，并结合系统性文献综述提出完整实验框架。

**🔧 技术方法**

采用CodeCarbon实时测量碳排放，使用卷积神经网络（CNN、U-Net、ResNet、VGG16/19、EfficientNet）以及标准机器学习工作流程。

**📊 数据集**

使用CIFAR-10图像分类数据集进行训练、验证与测试。

**📈 对比分析**

通过对整体与阶段性碳排放、准确率以及EAP进行比较，发现U-Net和EfficientNet在保持相近准确率的情况下排放最少，VGG19排放最高且准确率最低。

**⚠️ 局限性**

实验仅在单机CPU环境下进行且仅跑一次，未覆盖GPU、云端或大型模型，结果可能因硬件与能源强度不同而变化。

---

## 63. DSAR: Dual-Stream Autoregressive Modeling of Temporal Cloth Dynamics for Photorealistic Animatable Avatars

**arXiv ID:** 2608.10500 | [PDF](https://arxiv.org/pdf/2608.10500v1)

**作者:** Haozhong Xiong `[一作]` (Nanjing University), Sidan Du `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 DSAR 双流自回归框架，利用多视角 RGB 视频生成具有真实时间连贯性的可动画人类服装头像。

**💡 创新点**

创新点在于将可观测的运动学信息（几何变形与运动速度）与隐式内部状态（材质张力、历史记忆）分别建模为两个自回归流，并通过运动自适应时间聚合和记忆库显式捕捉时间因果性。

**🔧 技术方法**

使用 3D 高斯 splatting + SMPL/SMPL-X + 线性混合蒙皮，StyleUNet 编码解码，运动自适应时间聚合（MATA），跨注意力与记忆库融合，自适应时间正则化，多视角渲染损失（RGB、SSIM、LPIPS）。

**📊 数据集**

在 4D‑DRESS、AvatarREX 以及 AMASS（用于测试分布外运动）的多视角数据集上进行训练与评估。

**📈 对比分析**

与 HumanNeRF、GaussianAvatar、Animatable‑GS 等基线对比，在 PSNR、SSIM、LPIPS 上取得约 +1.5~+2.0 dB、SSIM 0.983、LPIPS 0.0305 的显著提升，尤其在远离分布（FD）测试序列中优于基线。

**⚠️ 局限性**

局限性包括需手动调节时间窗口和记忆长度，难以处理极快运动或极大姿态变化的细节；仅在多视角 RGB 场景验证，单视角或稀疏相机情形下性能未知。

---

## 64. CasDeblurGS: Cascaded 2D-to-3D Multi-View Consistency for 3D Gaussian Splatting from Two Blurry Images

**arXiv ID:** 2608.10345 | [PDF](https://arxiv.org/pdf/2608.10345v1)

**作者:** Haeyun Choi `[一作]` (University of Virginia), I-Gil Kim `[通讯]` (KT R&D Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种名为CasDeblurGS的级联框架，能够仅利用两幅已知内参的模糊图像，在无相机外参、无辅助锐化图像且不进行场景特定优化的前提下，完成3D高质量重建与新视角合成。

**💡 创新点**

创新点在于先通过遮挡感知的局部2D对应过滤获得可靠的跨视图信息，再利用无姿态3D高斯展平的全局重渲指导，形成逐步提升的2D‑3D级联流程，实现极端两视角模糊场景下的高保真3D重建。

**🔧 技术方法**

技术手段包括冻结的图像稳定器(NAFNet)、RAFT光流估计与前后一致性筛选、Occlusion‑Aware Cross‑View Guidance（OCGM）、无姿态3D高斯展平(NoPoSplat)骨干以及两阶段的NAFNet风格恢复网络，整体实现全流程的无监督端到端推理。

**📊 数据集**

使用了Deblur‑NeRF中的合成与真实两类数据集，共5个合成场景和10个真实场景，构成85对两视角模糊图像与目标视角的评估样本。

**📈 对比分析**

与SE‑GS、GAURA、CoherentGS、DAVANet、Difix3D+等基准方法相比，CasDeblurGS在真实场景上PSNR提升1.19dB、SSIM提升0.040、LPIPS降低0.064；在合成场景上PSNR提升2.11dB、SSIM提升0.109、LPIPS降低0.094，显著优于所有对比方法。

**⚠️ 局限性**

局限性包括假设已知相机内参、静态场景、仅两视角输入；在极端模糊、遮挡严重或视角重叠不足时对应信息易失效；不支持未知内参、动态场景或更大视角稀疏场景的扩展。

---

## 65. VERDICT: Training-Free Step-Wise Verification of Multimodal Reasoning via Disagreement-Aware Consensus

**arXiv ID:** 2608.10665 | [PDF](https://arxiv.org/pdf/2608.10665v1)

**作者:** Rohit Sinha `[一作]` (Indian Institute of Technology), Vineeth Balasubramanian `[通讯]` (Indian Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关、逐步验证的方法VERDICT，利用冻结的多模态评估器通过对其评分的耦合式一致性计算来过滤和选择推理步骤；

**💡 创新点**

创新点在于将跨模态不一致性显式建模为耦合评分问题，利用协调博弈的唯一纳什均衡实现闭式一致性解，从而在不需要额外训练的情况下利用不一致信息进行验证；

**🔧 技术方法**

使用三种冻结评估器（视觉、逻辑、上下文）分别评分，构建二次耦合目标并求解闭式一致性评分，随后按平均置信度与一致性分散的双阈值进行接受与排序；

**📊 数据集**

在六个多模态推理基准上进行评估，包括3DSRBench、CV-Bench（2D/3D）、BLINK、MMStar、AI2D等；

**📈 对比分析**

与基线模型、无训练的平均/最大/最小/方差聚合方法以及多个领域专用批评器比较，VERDICT在所有基准上均无退化，并在最坏情况下提升高达5.95%，整体准确率提升约+3.84%，且在六个基准上均比域特定批评器更稳定；

**⚠️ 局限性**

局限在于当所有评估器共识错误时仍会通过一致性传播错误，且在评估步骤全部被拒绝时需退回到基准模型；

---

## 66. Nonlinear Model Predictive Control via Sequential Convex Programming for Drone-to-Drone Docking

**arXiv ID:** 2608.10542 | [PDF](https://arxiv.org/pdf/2608.10542v1)

**作者:** Neeraj Balachandar `[一作]` (Indian Institute of Technology Hyderabad), Vishnu R. Unni `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了一套基于顺序凸规划（SCP）的非线性模型预测控制（NMPC）框架，用于在受风扰动和目标机动影响下实现多旋翼无人机的中空对接；

**💡 创新点**

创新点在于：①将航天软捕获的凸化对接约束迁移至空中无人机，并结合非线性多旋翼动力学；②在SCP中加入目标状态估计与预测，提升对噪声与漂移的鲁棒性；③使用SOCP实现实时可解的优化，使得在有限时间窗内可生成动态可行且安全的对接轨迹；

**🔧 技术方法**

主要技术包括：顺序凸规划、非线性模型预测控制、Kalman滤波状态估计、MuJoCo物理仿真、Clarabel二次锥规划求解器、简化的多旋翼动态模型；

**📊 数据集**

使用的“数据集”是基于MuJoCo的仿真环境，生成的静止目标和匀速目标轨迹，并加入高斯风扰动，仿真数据不来源于真实传感器；

**📈 对比分析**

通过仿真评估对接时间、圆锥角违规率、控制能量等指标，在不同风速标准差下（0.1、0.3、0.5、1.0 m/s）展示了系统在σ≤0.5时对接成功、违规率为0、能量约1.2–1.6；σ>0.8时因圆锥约束违反而失效；对比方法未给出传统PID或MPC基线，但指标表明系统在给定约束下能保持稳定对接；

**⚠️ 局限性**

局限性包括：忽略了偏航动力学和下洗效应；目标仅为静止或匀速，未考虑加速或激烈机动；仿真环境未覆盖真实传感器噪声与障碍物感知；未验证现场实验；缺乏递归可行性与闭环稳定性理论分析；

---

## 67. A Convolutional Layer Activation Dimensionality Reduction for Out-of-Distribution and Adversarial Attack Detection Methods

**arXiv ID:** 2608.10203 | [PDF](https://arxiv.org/pdf/2608.10203v1)

**作者:** Leandro de Souza Rosa `[一作]`, Riccardo Rovatti `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种可控高压缩的卷积层激活维度约减方法，并将其用于 OoD 与对抗攻击检测。

**💡 创新点**

提出利用卷积等价变换结合SVD排序的可调压缩策略，兼具高压缩率和信息保留；并扩展 DMD 与 MACS 以支持任意 DR。

**🔧 技术方法**

卷积等价变换、SVD、平均池化、Mahalanobis 距离、GMM、对抗样本生成等技术。

**📊 数据集**

CIFAR‑10/100、ImageNet、Far‑OOD 以及 BIM、PGD、FAB、SA、APGD、TRADES 等对抗攻击数据集。

**📈 对比分析**

通过 AUC/几何均值对比，新的 DR 在多模型、多数据集上与原始 DMD/MACS 相当或更优，并显著降低内存与计算开销。

**⚠️ 局限性**

在极大模型上 SVD 计算仍可能溢出；在极端压缩下信息损失仍存在风险，且对超参数敏感性不一。

---

## 68. HyperShape: Hyperelasticity Across Diverse Shapes

**arXiv ID:** 2608.09938 | [PDF](https://arxiv.org/pdf/2608.09938v1)

**作者:** Leo Widmer `[一作]` (University of Basel), Philippe Claude Cattin `[通讯]` (University of Basel)

**通讯引用:** 9544 | [OpenAlex ID](https://openalex.org/A5048965835)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了可控形状生成框架 HyperShape，并使用该框架产生了大量 2D 与 3D 超弹性仿真数据集，用来评估神经算子在几何、边界条件与力加载泛化能力。

**💡 创新点**

创新点在于：①提出可调节形状复杂度、边界条件与加载的可扩展数据生成管线；②提供多样化的 2D/3D 数据集，支持 in‑distribution、out‑of‑distribution 以及 synthetic‑to‑real 的系统评估；③通过实验揭示现有神经算子在面对几何多样性时的显著性能下降。

**🔧 技术方法**

技术细节包括：利用高斯随机场 + ODE 产生随机变形的形状；使用 FEniCS 进行 FEM 计算；将输入编码为格点 SDF 或点云 SDF；实验中对 FNO、CNO、U‑Net、GINO 与 Transolver 这五种神经算子进行训练与评估，并用相对 L² 与 H¹ 误差衡量性能。

**📊 数据集**

使用了四个 2D 数据集（Random2D、EasyRandom2D、OODRandom2D、OneRandom2D）和四个 3D 数据集（Random3D、AugLiver、Liver、OneLiver），每个数据集包含不同数量的形状、仿真次数、基底形状与变形参数，覆盖从单形状到大规模多样形状的范围。

**📈 对比分析**

采用 80‑10‑10 的训练/验证/测试划分，主要用相对 L² 误差比较模型表现。结果显示：几何多样性增加时误差显著上升；在 OOD 场景中，训练于更大多样化数据的模型性能提升；在 synthetic‑to‑real 迁移中，基于 AugLiver 的点云模型表现最差，而基于 Random3D 的网格模型表现较好。整体来看，所有神经算子在面对复杂几何与多变边界条件时都表现出明显的泛化不足。

**⚠️ 局限性**

局限性包括：仅支持均匀材料，未考虑材料参数的多样性；边界条件与力加载仅限于单一表面位置，无法模拟多点载荷；未深入分析形状与边界条件交互对难度的共同影响。

---

## 69. Towards Efficient Reasoning in LLM-Based Recommender Systems via Model Merging

**arXiv ID:** 2608.10447 | [PDF](https://arxiv.org/pdf/2608.10447v1)

**作者:** Linh Dieu Le `[一作]` (University of Queensland), Junliang Yu `[通讯]` (Griffith University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于注意力头的模型合并方法 REAM，能够在保持慢思考推荐器预测精度的同时，显著压缩其推理轨迹。

**💡 创新点**

创新点在于将检索重要性、决策可信度以及 Fisher 敏感度三种解释性信号融合为每个注意力头的合并系数，提供细粒度、解释驱动的压缩策略。

**🔧 技术方法**

使用注意力头级别的参数分离、Fisher 近似敏感度估计、线性约束下的水位分配优化等技术实现模型合并；基础模型为 Qwen2.5-3B 的 RecZero（慢思考）和 TALLRec（快思考）。

**📊 数据集**

在 Amazon Book、Amazon Music 和 Yelp 三个基准数据集上进行实验，评估评分预测（MAE、RMSE）与推理长度（Tok）。

**📈 对比分析**

与多种基线（Task Arithmetic、DARE、AIM+TA、ACM+TA 等）对比，REAM 在所有数据集上平均缩短推理长度约 20‑25%，同时保持或提升 MAE/RMSE，优于其他训练‑free 合并方法。

**⚠️ 局限性**

局限性包括对极大规模模型的适配仍需改进、合并系数仍需超参数调优，以及在某些场景下快思考模型的预测性能仍略低于原慢思考模型。

---

## 70. Synthesizing Probabilistic Saturating Counters with Differentially Private Formal Guarantees

**arXiv ID:** 2608.10521 | [PDF](https://arxiv.org/pdf/2608.10521v1)

**作者:** Zhiming Chi `[一作]` (Key Laboratory of System Software (Chinese Academy of Sciences)), Naijun Zhan `[通讯]` (Peking University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文对概率饱和计数器（PSC）在 Prime+Probe 侧信道攻击下进行形式化的差分隐私（DP）分析，并基于此设计了新的增强型 PSC，能够在给定隐私预算下保证纯 DP 并兼顾预测性能。

**💡 创新点**

创新点在于：①将 PSC 与攻击建模为概率 Moore 机，推导出最优攻击策略并揭示原始 PSC 的安全缺陷；②通过引入防御参数 p 及更新概率 m，提供可合成的 DP 保障；③推导出 Stationary Misprediction Rate 的闭式表达式，构建隐私‑效用曲线；④与随机响应机制对比，证明在同等隐私预算下增强 PSC 具有更低的误预测率。

**🔧 技术方法**

主要技术包括：概率 Moore 机建模、差分隐私理论、符号计算（SageMath）求解参数范围、马尔可夫链平稳分布求解、以及对 SPEC CPU 2017 和 MergeSort 进行模拟验证。

**📊 数据集**

实验使用 SPEC CPU 2017 基准集（约 1000 万指令）和 MergeSort（10^5 个整数）进行性能与误预测率评估。

**📈 对比分析**

与传统饱和计数器和原始 PSC 以及随机响应机制相比，增强 PSC 在给定隐私预算（如 ln 9）下误预测率仅提升 1–2%，且在部分基准上可实现 IPC 增益；在完全隐私（0,0）点误预测率升至 50%，产生最高约 24% 的性能开销。

**⚠️ 局限性**

局限性在于仅对 PSC 原语进行了 DP 分析，未覆盖完整预测器（如 TAGE）或多轮自适应攻击的 DP 组合；未来工作需扩展到更复杂预测器、自动化合成工具和重复攻击的 DP 合成分析。

---

## 71. Toward the Cognitive--Physical Limits of Embodied Intelligence through a World-Model-Centric Autonomous Racing Agent

**arXiv ID:** 2608.10618 | [PDF](https://arxiv.org/pdf/2608.10618v1)

**作者:** Zitong Shan `[一作]` (Nanyang Technological University), Chen Lv `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种以世界模型为核心的自动赛车智能体，联合建模交互演化、车辆动力学和可行运动边界，实现认知与物理极限协同的闭环决策与控制；

**💡 创新点**

创新点在于将可解释的世界模型嵌入到感知-预测-决策-控制流水线，利用模型滚动预测评估多模态轨迹，并通过连续闭环更新世界模型与策略，系统性地提升认知与物理极限利用率；

**🔧 技术方法**

关键技术包括：差分动力学残差学习的车辆动力学模型、基于注意力的多智能体交互预测、可行运动包络的物理约束学习、基于世界模型的多模态轨迹生成与滚动评估、受约束的MPC控制与限制适应机制、以及使用CQL+BC的策略更新；

**📊 数据集**

使用真实赛道的全尺寸赛车收集的高频传感与控制数据（LiDAR、RTK、IMU、轮速、视觉），并在Autoverse仿真中生成对手轨迹；

**📈 对比分析**

在仿真20秒交互任务中与BC、SAC和规则规划器对比，成功率88.3%（比BC高40.8%，比SAC高29.3%），安全里程超6.4km，平均速度保持竞争性；同时在真实赛道上实现最高256.3 km/h、峰值侧向加速度26.8 m/s²；

**⚠️ 局限性**

局限在于仅在赛车这一高度结构化、物理相对可控的环境验证，缺乏对更复杂多样化场景、动态障碍与人机协作等开放世界条件的评估；

---

## 72. A Fine-Grained Complexity of Co-Secure Domination for Some Subclasses of Chordal Graphs

**arXiv ID:** 2608.10617 | [PDF](https://arxiv.org/pdf/2608.10617v1)

**作者:** Manjusha M S `[一作]` (National Institute of Technology Calicut), Renjith P `[通讯]` (National Institute of Technology Calicut)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了连通图的共安全支配集（co‑secure dominating set，CSDS）问题，在2‑树的特定子类（如P_n^2、P_n^(2,k)）上给出了精确的下界与上界，并对分裂图（split graph）中的CSDS问题给出了算法与复杂度阐释：在K_1,3‑free分裂图上可在线性时间内求解最小CSDS，而在K_1,r‑free（r≥4）分裂图上该问题为NP‑完全。

**💡 创新点**

创新点包括：
- 对P_n^2、P_n^(2,1)、P_n^(2,2)以及更一般的P_n^(2,k)（k≥3）给出了完整的γ_cs值表达式，首次完成这类k‑树子类的CSDS计数。
- 通过结构分析提出了一个线性时间算法解决K_1,3‑free分裂图的最小CSDS问题。
- 建立了从P_3路径分割问题到Δ^I=2分裂图的CSDSD的多步归约，证明了K_1,r‑free（r≥4）分裂图上的CSDSD为NP‑完全，形成了完整的复杂度二分（dichotomy）结果。

**🔧 技术方法**

使用的技术主要是图论中的结构性证明（完美消去序列、模拟退火、构造性归约）、递归/归纳法（对n进行递推）、以及组合归约（从P_3路径分割或Exact‑3‑Cover到CSDSD）。

**📊 数据集**

论文未使用实验数据集，全部采用理论分析与构造性证明。

**📈 对比分析**

性能方面：在K_1,3‑free分裂图上算法复杂度为O(n+m)，即线性；而在K_1,r‑free（r≥4）分裂图上问题被证明为NP‑完全，说明不存在多项式时间求解（除非P=NP）。

**⚠️ 局限性**

局限性：
- 仅覆盖了2‑树的特定子类和分裂图的特定K_1,r‑free子类，未探讨更一般的k‑树或其他图类。
- 结果多为理论性质，缺乏实验验证与实际应用场景的评估。
- 只针对共安全支配集，未涉及其变体（如安全支配集、总共安全支配集等）的更广泛研究。

---

## 73. Retrieval-Corrected Conformal Prediction for Time Series

**arXiv ID:** 2608.10553 | [PDF](https://arxiv.org/pdf/2608.10553v1)

**作者:** Sangjin Jin `[一作]` (Ulsan National Institute of Science and Technology), Yongjae Lee `[通讯]` (Ulsan National Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种新的时间序列预测区间方法——检索校正合规预测（RCCP），通过检索与当前预测上下文相似的历史残差构造局部不对称区间，再用一个标量校正因检索导致的覆盖误差。

**💡 创新点**

创新点在于将残差检索与合规校正分离，既利用检索提供的局部残差证据实现自适应区间宽度，又通过归一化检索误差的标量校正恢复严格的覆盖保证，解决了传统检索方法缺乏校正与局部残差选择不精准的问题。

**🔧 技术方法**

技术上结合了检索式知识库（存储过去预测上下文与残差对）、欧氏/余弦距离检索、单侧残差量化生成不对称区间，以及分割合规预测中的分位数校正与标量乘子校正。

**📊 数据集**

在四个公开时间序列基准（Air、Solar、Electricity、Wind）上，以LSTM和Transformer两种前置回归器进行实验，并在每个数据集上对多个位置/站点独立评估。

**📈 对比分析**

与SCP、EnbPI、SPCI、NexCP、HopCPT、ResCP等基线相比，RCCP在所有数据集和模型上都实现了更稳健的目标覆盖率、最低的Winkler得分和更少的严重漏报，同时保持了与简单方法相近的校准时间。

**⚠️ 局限性**

主要局限在于检索表示的质量；如果检索键不能充分捕捉错误相关的相似性，区间效率可能下降；未来工作需扩展到高维多变量、多步预测等更复杂场景。

---

## 74. Automating and Scaling Behavioral Scientific Research on AI Agents

**arXiv ID:** 2608.10030 | [PDF](https://arxiv.org/pdf/2608.10030v1)

**作者:** Soo Yong Lee `[一作]` (KAIST), Kijung Shin `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了 AEROBAT，一个基于多代理 LLM 的自动化实验研究系统，完整实现了从假设生成、环境配置设计、仿真执行到盲评估与报告撰写的行为科学研究流程，专门用于探索 AI 代理在多种目标行为（如友善、合作、欺骗等）下的因果机制。

**💡 创新点**

创新点包括：①首次将完整的行为科学研究流程实现自动化；②提出可控制、可参数化、可多重实现的环境模型，支持匹配实验；③设计多代理架构，分别承担假设生成、配置设计、仿真执行和盲评估等角色；④引入 Bayesian monotone‑increment 模型与 Kendall’s τ 检验，提升因果效应估计的可靠性。

**🔧 技术方法**

技术手段：多代理 LLM（如 GPT‑5‑mini、Gemini‑3.1‑Pro、Kimi K2.6）配合自然语言配置、参数化与多重实现；仿真引擎执行数千轮对抗/合作场景；贝叶斯统计模型估计效应大小与 Bayes 因子；盲评估代理采用预先生成的评分标准进行行为量化；人类评估辅助验证配置与仿真的一致性。

**📊 数据集**

数据集：论文中使用 12 种目标行为共 79 个假设，构造了 1,240 组匹配配置，执行了 23,512 场仿真，总计 1,400,000+ 交互轮次；数据全部为系统自动生成的仿真日志，未使用公开真实世界数据。

**📈 对比分析**

比较方法：对每个假设在 3–5 个领域、3–5 组匹配配置中进行实验，统计 BF₁₀ 与 Δ，并对比不同 LLM 作为受试者的效应一致性；此外，通过三项逆向任务验证环境配置与仿真的一致性。性能方面：在 79 个假设中有 26 个达到 BF₁₀≥3 的显著结果，跨 LLM 的效应相关系数>0.6，配置推断准确率达 93%，表明系统在自动化行为因果研究方面具有较高可靠性。

**⚠️ 局限性**

局限性：①高度依赖 LLM 的推理与生成能力，可能出现偏见、创意不足或判断失误；②复杂性高的环境仍需人工审查，完全自动化尚不可行；③未验证在真实世界大规模部署中的稳健性；④受限于当前 LLM 的知识与推理范围，可能无法捕捉极端或非预期行为。

---

## 75. EVIL-Detect for NLPCC 2026 Shared Task 6: LLM-Generated Text Detection

**arXiv ID:** 2608.10698 | [PDF](https://arxiv.org/pdf/2608.10698v1)

**作者:** Hongrui Bao `[一作]` (Chinese Academy of Sciences), Yanan Cao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种多信号集成框架EVIL-Detect，用于中文LLM生成文本和LLM润色文本的三类检测。

**💡 创新点**

结合编辑强度回归、零样本概率对比、词频统计和保守规则，并引入冲突感知融合以提升对外部分布的鲁棒性。

**🔧 技术方法**

使用EditLens/Soft-EditLens回归、EchoPrompt零样本概率对比、词频log-odds、冲突感知融合、阈值校准及高精度文本规则等技术。

**📊 数据集**

使用NLPCC 2026共享任务6的训练集（CUDRT衍生）以及两个隐藏测试集（DetectRL-X Chinese），并覆盖GPT-4、Qwen系列、ChatGLM、Baichuan等生成器。

**📈 对比分析**

与单一基线（EditLens）和其他代表性设计（如Binoculars、SFT）对比，最终宏F1达0.8888，排名第一，显著提升HGT/HLT区分。

**⚠️ 局限性**

对完全未知的生成器或极端写作风格仍易混淆，且规则依赖手工编写，需进一步自动化。

---

## 76. Measuring Semantic Abstractness of SAE Features via Nonlocality

**arXiv ID:** 2608.10537 | [PDF](https://arxiv.org/pdf/2608.10537v1)

**作者:** Chuqiao Lin `[一作]` (University of Oxford), Xiao-Liang Qi `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种基于梯度的无标签度量——特征非局部性（Feature Nonlocality，FNL），用于量化稀疏自编码器（SAE）特征的上下文依赖程度，并将其应用于审计“逃逸”防护机制和改进推理性能。

**💡 创新点**

创新点：
1) FNL作为单一、无监督、与模型无关的指标，能够捕捉特征的语义抽象级别；
2) 通过与token注入、释义鲁棒性等代理指标的统计相关性，证明FNL与高层语义特征正相关；
3) 利用FNL筛选特征，在未使用任何标注或特定token筛选的情况下，提升DeepSeek-R1-Distill-Llama-8B在MATH-500上的准确率。

**🔧 技术方法**

技术手段：
- 稀疏自编码器（SAE）训练与特征提取；
- 通过反向传播计算每个输入位置对特征激活的梯度贡献；
- 对梯度归一化后计算熵得到FNL；
- 设计token注入和释义鲁棒性实验验证抽象性；
- 对高FNL特征进行激活调节（clamping）并评估推理表现；
- 与传统token-cue、对比集筛选等方法对比。

**📊 数据集**

使用的数据集包括：WikiText、GSM8K、Code-Python（用于FNL稳定性评估）；DeepSeek-R1-Distill-Llama-8B的原始和封装攻击样本（用于逃逸防护审计）；OpenThoughts-114k（用于FNL计算）；MATH-500（用于推理性能评估）。

**📈 对比分析**

比较与评估：
- FNL与token注入恢复的Spearman相关系数在-0.39到-0.46之间，AUC为0.73–0.84，表明高FNL特征更不易被单词触发；
- 与释义鲁棒性S_a的Spearman相关为+0.27，进一步验证抽象性；
- 在DeepSeek模型上，利用高FNL特征集进行激活调节，MATH-500的avg@4从0.865提升至0.911（+4.6点），超过低FNL、随机及单特征基线。

**⚠️ 局限性**

局限性：
- FNL对不同模型、不同层次的预测性能不一；在非DeepSeek模型中高/低FNL对齐的提升效果有限；
- 需要反向传播和梯度计算，计算成本相对较高；
- 只是一种相关性指标，不能完全证明因果机制；
- 对于极端或未见过的行为，FNL可能失效；
- 在对比集或特征分裂等现象出现时，FNL仍需结合其他验证方法。

---

## 77. Beyond Pixels: From Video Priors to 4D Worlds

**arXiv ID:** 2608.10744 | [PDF](https://arxiv.org/pdf/2608.10744v1)

**作者:** Zihao Liu `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种直接从视频生成模型的最终去噪VAE潜在空间到4D场景的映射方法——Latent-to-4D，绕过RGB解码，直接生成动态3D几何与相机轨迹。

**💡 创新点**

创新点在于将共享VAE潜在空间作为可重用接口，使得单一训练的L4AR网络即可与多种兼容的DiT视频生成器配合，无需额外调参或重新训练。

**🔧 技术方法**

核心技术包括：L4AR对齐模块（利用3D卷积和重采样将潜在张量映射至4D解码器的token格子），层级时空自注意力的细化模块，以及基于预训练4D重建器的摄像机与动态几何解码头。

**📊 数据集**

训练使用约1K条已有4D重建数据集的录像片段；评估使用公开的Text4D-200和I4D-200基准。

**📈 对比分析**

在同一潜在空间下与匹配的Wan+4RC、π³、Any4D等生成-重建级联相比，Latent-to-4D在DINO-F1、文本/图像条件下的多视角一致性和人类评估中均表现更好，尤其在几何完整性和时序稳定性上获得显著优势。

**⚠️ 局限性**

局限性包括：仅适用于共享VAE规范的模型；评估主要依赖投影DINO分数和主观人类评判，缺乏针对生成场景的度量几何准确性验证。

---

## 78. Precise Top-Layer Fabric Segmentation for Fabric Destacking with Edge- and Shape-Aware Deep Networks

**arXiv ID:** 2608.10648 | [PDF](https://arxiv.org/pdf/2608.10648v1)

**作者:** Wenbo Dong `[一作]` (University of Hong Kong), Kazuhiro Kosuge `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在叠加布料中对顶部层进行精准分割的多分支训练架构，通过在编码解码网络中加入边缘感知分支和形状感知分支，对边界和整体形状进行监督，从而提升分割精度。

**💡 创新点**

创新点在于：1) 在训练阶段仅使用两条辅助分支（边缘、形状），推理阶段保持轻量化主干网络；2) 边缘分支直接对边界进行二值化监督；3) 形状分支利用CAD模型生成的理想形状进行全局正则化，使分割结果既精准又符合物理形状。

**🔧 技术方法**

技术手段包括：ResNet50编码器 + 解码器（类似U-Net）主干网络；1×1卷积边缘分支；轻量CNN+全连接的形状分支；交叉熵+Dice、边缘交叉熵、形状交叉熵的联合损失；PyTorch实现，Adam优化器。

**📊 数据集**

使用包含235张真实领布料图像的自建数据集，每张图像配有像素级语义掩码、边缘掩码和基于对应CAD模型生成的理想形状标签。

**📈 对比分析**

通过与基线（仅主干网络）以及仅加边缘分支的模型进行Ablation，使用IoU、像素精度和边缘RMSE等指标评估。结果表明：基线IoU 93.25%，加边缘分支提升至96.24%，再加形状分支进一步提升至96.80%；边缘RMSE从5.03降至2.58像素，验证了多分支训练方案在精度和边缘清晰度上的显著提升。

**⚠️ 局限性**

局限性包括：1) 仅在小规模自建数据集上验证，泛化性待进一步考察；2) 形状分支依赖CAD模型生成的理想形状，限制了在无CAD支持场景的适用性；3) 推理阶段虽然仅使用主干网络，但在更大分辨率或实时机器人系统中的计算效率和鲁棒性尚未全面评估。

---

## 79. Unveiling the Predators: Contemporary Approaches to Identifying Illegitimate Open Access Journals in the Academic Publishing Ecosystem

**arXiv ID:** 2608.10739 | [PDF](https://arxiv.org/pdf/2608.10739v1)

**作者:** Robert Šamárek `[一作]`, Radek Martinek `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了现有的识别劣质期刊方法，并提出基于多变量图分析的新方法；

**💡 创新点**

创新点在于将学术出版生态构建为作者、论文、期刊、出版社等节点组成的网络，用图论算法揭示网络结构中的异常与社区，避免传统二元列表的局限；

**🔧 技术方法**

采用多变量图分析技术，包括中心性测度、社区检测、异常检测等图算法；

**📊 数据集**

使用开放获取的OpenAlex元数据作为构建网络的基础数据集；

**📈 对比分析**

对比了手工检查、黑白名单、机器学习分类等传统方法的优缺点，指出虽然一些ML模型可达F1≈0.98，但仍受限于训练数据的偏差和可解释性；新方法尚未在实验中量化性能，但预期能提供更透明、可扩展的识别手段；

**⚠️ 局限性**

局限性包括尚未完成经验验证、依赖OpenAlex的覆盖范围与质量、图模型构建与算法参数需要进一步优化，以及如何将异常检测结果转化为可操作的决策建议等问题。

---

## 80. Media-over-Multipath-QUIC for Realtime Video Applications

**arXiv ID:** 2608.10741 | [PDF](https://arxiv.org/pdf/2608.10741v1)

**作者:** Tanya Shreedhar `[一作]` (Delft University of Technology), Fernando Kuipers `[通讯]` (Delft University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种在 Media‑over‑QUIC 之上运行的多路径扩展（MPQUIC‑MoQ），允许在边缘 Relay 上通过声明式规则根据对象元数据将视频帧映射到不同的网络路径，从而提升实时视频交付的尾部延迟和播放缓冲。

**💡 创新点**

创新点在于：①将元数据与路径标签解耦，使得 Relay 在不解析媒体内容的情况下按规则决策；②通过规则语言实现依赖关系、重构避免、成本敏感等策略；③在多路径 QUIC 之上提供了可插拔的、兼容现有协议的控制面；④通过实际星链+WiFi 的实验验证了规则驱动调度能显著降低 99.9% 阈值延迟。

**🔧 技术方法**

技术手段包括：Media‑over‑QUIC（对象、元数据、Relay 机制）、多路径 QUIC（独立 RTT、拥塞控制、调度）、可编程规则引擎（匹配与动作）、路径标签注入、规则合并与指令附加到 QUIC 流、基于 SVC 编码的实验视频、Starlink 与 WiFi 的真实网络测量。

**📊 数据集**

使用的数据集是：一段 100 s、1080p、SVC 编码的视频（1 s GOP、3 级时间层），以及在欧洲、北美两地分别搭建的 Starlink 与 WiFi 双路径测试床。实验中共收集了约 5 000 帧的 FCT 与缓冲统计。

**📈 对比分析**

比较方法：将 7 种配置（单路径基线、四种传输层调度、规则驱动调度）在同一双路径下运行，并测量帧完成时间（FCT）、最小播放缓冲、备份路径使用率。结果显示，规则驱动调度将 99.9% 阈值 FCT 从 384.7 ms 降至 114.1 ms（降低 70%），播放缓冲从 332 ms 降至 127 ms（降低 61.5%），备份路径使用从 90–100% 降至 11.3%，唯一达到 150 ms 交互预算的配置。

**⚠️ 局限性**

局限性包括：①需要路径真正分离，若路径在核心网聚合则收益显著下降；②规则集的有效性取决于作者，缺乏自动生成或验证机制；③Relay 的协作是假设，恶意或资源紧张时可能违背规则；④仅针对最后一跳多路径，未覆盖端到端路径；⑤实验仅在单一星链星座、单一 SVC 编码和单个发布者/订阅者场景，未覆盖更广泛的码流或网络负载。

---

## 81. Bridging Severe Cross-Modal Misalignment: End-to-End Visible-Infrared Object Detection via Explicit Feature-Domain Affine Registration

**arXiv ID:** 2608.10680 | [PDF](https://arxiv.org/pdf/2608.10680v1)

**作者:** Qi Ming `[一作]` (Beijing University Of Technology), Xudong Zhao `[通讯]` (Beijing Institute Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种端到端的可见-红外相机误差鲁棒的定向目标检测方法JFRDet。

**💡 创新点**

创新点在于引入显式特征域仿射配准（CMAA）以消除大尺度几何偏移，配合光照引导的互补融合（IGCF）和对齐质量一致性门控（AQCG）实现联合训练。

**🔧 技术方法**

使用双流Backbone、跨模态仿射对齐模块、粗细尺度匹配、光照分数、门控策略，以及基于MMDetection/ MMRotate框架的训练。

**📊 数据集**

构造了DroneVehicle Misaligned（DVMA）数据集，包含约5.7万对受仿射扰动的可见-红外图像及定向框注释。

**📈 对比分析**

与多种单模和多模检测器对比，JFRDet在DVMA上取得mAP_50 69.7%（mAP_50:95 36.1%），显著高于S^2ANet+IR等最优方法。

**⚠️ 局限性**

主要局限包括仅针对仿射几何误差，未验证对非仿射失配的鲁棒性；对训练时需要配准标注；模型相对复杂，推理速度受限。

---

## 82. Surfacing the Unsaid: CUE-Bench for Affective Stance in Chinese Discourse

**arXiv ID:** 2608.10810 | [PDF](https://arxiv.org/pdf/2608.10810v1)

**作者:** Zhenyan Zheng `[一作]` (Huazhong University of Science and Technology), Zikai Song `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了CUE-Bench中文未说情感基准，定义显式‑隐式立场矩阵，包含情感立场、语用意图和细粒度情感三任务。

**💡 创新点**

创新点在于将显式情感与隐式情感的相互作用量化为九类立场，并提供链式推理框架，提升对隐含情感的判别。

**🔧 技术方法**

采用模型协同生成、人工裁决及受控LLM裁决的混合标注流程，并在评测中使用矩阵引导的链式思考 Prompt。

**📊 数据集**

使用约60k个候选实例，最终51.8k条样本，覆盖开放域对话、社交媒体、讽刺、客服等多场景中文对话。

**📈 对比分析**

与多种基线（Direct/Few‑shot/CoT）比较，矩阵引导方法在情感立场、语用意图、细粒度情感上均取得0.3–0.1点的提升，表现最优。

**⚠️ 局限性**

限制在于残留标注噪声、显式/隐式方向空间过于粗糙以及类别分布不均导致长尾问题。

---

## 83. AECNav: Active Evidence Consolidation for Efficient Zero-Shot Open-Vocabulary Object Navigation

**arXiv ID:** 2608.10817 | [PDF](https://arxiv.org/pdf/2608.10817v1)

**作者:** Guanlin Liu `[一作]` (Chinese University of Hong Kong Shenzhen), Junjie Hu `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种无训练的零射击目标导航方法AECNav，能在未知环境中定位任意目标并完成导航。

**💡 创新点**

创新点在于三大组件：证据门控感知共享编码、基于log-odds的目标证据累积与消减、主动信息增益前沿选择，使得目标确认更可靠且探索更高效。

**🔧 技术方法**

采用C-RADIOv4统一编码、SigLIP2文本嵌入、SAM3实例分割、LLM生成混淆目标、基于前沿信息增益与代价的主动决策。

**📊 数据集**

使用HM3D‑v2、HM3D‑OVON、MP3D三大模拟数据集进行基准测试，并在Unitree Go2四个室内真实场景上进行验证。

**📈 对比分析**

与现有最先进方法相比，AECNav在HM3D‑v2、MP3D、HM3D‑OVON上分别取得84.7%、51.3%、57.3%的成功率（SPL也显著提升），在真实机器人上95%成功率且运行频率约5 Hz。

**⚠️ 局限性**

限制主要体现在对LLM预先生成混淆词库的依赖、对前沿信息增益估计的近似以及在极大开放词汇场景下仍可能被误导或对遮挡严重的目标识别不足。

---

## 84. Connectivity Augmentation of Plane Graphs

**arXiv ID:** 2608.10848 | [PDF](https://arxiv.org/pdf/2608.10848v1)

**作者:** Krishnan Dehaleesan `[一作]` (University of Bergen), Pranabendu Misra `[通讯]` (Chennai Mathematical Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在固定平面嵌入的图上，分别给出了2-边连通性增补（PECA-Fix）和外平面图的3-顶点连通性增补（OTA-Fix）的最优增补算法，并证明其时间近线性；

**💡 创新点**

通过构造bt‑tree与标签方案实现2‑边连通的最优增补；利用面拆分技术在外平面图中实现3‑连通的最优增补；证明固定嵌入下的最优增补仅比无嵌入情况多一条边；将树图连通与内双对的对应关系结合以实现线性时间；

**🔧 技术方法**

使用平面图的bt‑tree/ bc‑tree、Union‑Find（逆Ackermann函数）、面图分解、内双对树、树图连通算法TreeConn、面拆分与内部边插入等结构与算法技术；

**📊 数据集**

论文仅给出理论证明与算法描述，并未使用具体实验数据集；

**📈 对比分析**

与已知的NP‑hard或近似方法相比，提出的算法在时间上为O(n(1+α(n)))、空间为O(n)，并给出最优增补边数为⌈k/2⌉或⌈k/2⌉+1，显著提升了增补质量与效率；

**⚠️ 局限性**

局限性在于仅适用于固定嵌入的平面图；对3连通性仅覆盖外平面图；未考虑加权边成本；在更高连通度或其他图类仍为NP‑hard，算法无法直接推广。

---

## 85. MedUP: Awakening Unified Understanding and Perception in Medical Vision-Language Models

**arXiv ID:** 2608.10635 | [PDF](https://arxiv.org/pdf/2608.10635v1)

**作者:** Yuan Wang `[一作]` (Zhejiang University), Zuozhu Liu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MedUP，一种利用区域掩码作为离散语言令牌的医学视觉‑语言模型，能够统一执行医学VQA、文本引导分割和区域根植理解。

**💡 创新点**

创新点在于开发了UniMedTok掩码令牌器，将图像掩码压缩为两级 256 维离散码并直接嵌入LLM词表，实现感知与理解在同一自回归空间内无缝交互，并引入Seg‑CoT链式推理提升文本到掩码的生成质量。

**🔧 技术方法**

技术主要包括图像条件向量量化掩码自编码器、词表扩展为掩码代码的特殊令牌、四流混合监督训练（医学VQA、文本引导分割、区域根植理解、Seg‑CoT）以及回路过滤（round‑trip filtering）以提升掩码重建质量。

**📊 数据集**

使用了约 1.84M 实例的 UniMed‑Train 语料（覆盖 80+ 医学分割数据集、医学 VQA、区域根植任务以及 4000 条 Seg‑CoT 样本）和 UniMed‑Bench 评测基准（包含 8.3k VQA、219k 文本引导分割和 218k 区域根植样本）。

**📈 对比分析**

在 UniMed‑Bench 上与基准模型（原生 Med‑VLM、外部工具和双解码器方案）比较，MedUP 在医学 VQA 最高准确率 (≈66–93% 取决于后端)、文本引导分割的宏观 Dice 最高 (≈64–88%) 并在区域根植理解中实现精确匹配 (≈78–82%)，同时与专业分割器保持竞争力。

**⚠️ 局限性**

局限性包括掩码表示过于简洁（可能不足以表达极小或形状复杂的结构）、主要在离线基准上评估，缺乏交互式或长序列工作流程验证，以及仅验证两种后端规模，尚未系统探索更大模型和不同训练设置的通用性。

---

## 86. Generator-Guided Inverse Sampling for Lévy-Driven Generative Models

**arXiv ID:** 2608.10384 | [PDF](https://arxiv.org/pdf/2608.10384v1)

**作者:** Tianfu Qi `[一作]` (University of Electronic Science and Technology of China), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于生成器的逆采样方法，用于处理含有α‑stable跳跃项的 Lévy 驱动生成模型，并将逆采样过程拆分为扩散、小跳跃和大跳跃三部分。

**💡 创新点**

创新点包括：① 利用生成器（infinitesimal generator）直接推导逆向 Lévy 过程的完整生成器，揭示大跳跃成分是状态相关的 Markov 跳跃过程；② 设计了仅用轻量网络学习大跳跃率的逆采样框架；③ 在高维情形下通过极坐标变换和预计算查找表，实现大跳跃的高效采样；④ 将该框架应用于 OFDM‑SISO 信道估计，在混合高斯与冲击噪声环境中显著提升 NMSE 与 BER。

**🔧 技术方法**

使用的技术包括：Lévy 过程生成器分析、Markov 决策过程理论、α‑stable 分布理论、轻量卷积神经网络（CNN+MLP）学习跳跃率、极坐标变换与逆 CDF/拒绝采样、混合噪声的概率密度近似、观察引导（posterior）采样等。

**📊 数据集**

数据集：原始 OFDM 信号帧与对应的信道冲击样本（TDD‑A/C/D 延迟功率谱）以及混合高斯+冲击噪声（γ_g、γ_s）。

**📈 对比分析**

比较方法：基于线性 MMSE、裁剪 LMMSE、裁剪 OMP、裁剪 SBL、针对异常的 SBL、传统扩散方法、Lévy‑DSM 以及理想 benchmark；评估指标为 NMSE 与 BER。实验结果表明，该方法在所有信道配置与冲击指数 α（1.2、1.8）下均优于基线，尤其在冲击性强的 α=1.2 情况下提升显著。

**⚠️ 局限性**

限制：① 仅适用于符合线性、各向同性、α‑stable 跳跃的 Lévy SDE；② 对高维积分与密度估计仍有计算瓶颈（尽管通过近似与查表缓解）；③ 单跳近似导致在高跳跃率时出现二次误差；④ 对观察信息的后验改造仍需近似，真实后验跳跃率难以实时计算；⑤ 需要训练集支持，若数据分布偏离训练集可能影响性能。

---

## 87. DashArena: Benchmarking LLMs on Interactive Analytic Dashboard Generation

**arXiv ID:** 2608.10567 | [PDF](https://arxiv.org/pdf/2608.10567v1)

**作者:** Xiaotong Wang `[一作]` (Zhejiang University), Dazhen Deng `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DashArena基准，评估从任务到交互式仪表盘的开放式生成，并引入模型生成的可回放交互轨迹；

**💡 创新点**

创新点在于结合交互轨迹+浏览器回放生成可复现的执行证据，使用多模态VLM评判并通过Bradley–Terry聚合产生排行榜；

**🔧 技术方法**

采用VLM（Claude Opus 4.6作为教师，Qwen3‑VL‑8B‑Instruct distillation），浏览器执行（Playwright+Chromium），ECharts渲染，LLM多轮生成；

**📊 数据集**

任务来自Tableau Public的234个分析仪表盘，拆分为234个任务，涵盖14个主题；

**📈 对比分析**

通过对每个任务进行匿名的两两比较，VLM评判基于任务、截图和执行报告；聚合使用Bradley–Terry模型；实验中GPT‑5.5排名第一，GPT‑5.5和人类基准相近，其他模型分布在中下游，且没有模型达到90%以上的渲染或可回放率；

**⚠️ 局限性**

局限包括依赖Tableau Public的偏倚、仅评估提交的交互轨迹（未能检测隐藏错误）、未使用多轮交互式编码代理、对非公共或其他可视化生态的覆盖不足。

---

## 88. Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence

**arXiv ID:** 2608.10720 | [PDF](https://arxiv.org/pdf/2608.10720v1)

**作者:** Haoyu Zhang `[一作]` (Chinese University of Hong Kong, Shenzhen), Yiwen Guo `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了Ex-Omni-2D框架，实现多模态对话系统能够生成文本、个性化语音和与参考图像、参考音频同步的视频响应。

**💡 创新点**

通过结构化视觉思维计划（VTP）与多码本语音单元共享接口，将对话生成与视频/语音分离训练，并使用Prefix-Streaming学生实现高效增量视频输出。

**🔧 技术方法**

使用大语言模型（Qwen3、Qwen3-TTS、Qwen2.5-Omni）、3D VAE+DiT视频生成器、流映射与On-policy distillation、Prefix-Streaming等技术。

**📊 数据集**

训练与评估数据来自VoiceBench CommonEval、OmniCharacter、SpeakerVid、InstructS2S-200K、OmniCharacter等多模态数据集，视频生成训练使用约140K SpeakerVid视频。

**📈 对比分析**

与echomimic、StableAvatar、UniAVGen等基准在音视频质量、同步和对话一致性等指标对比，Ex-Omni-2D在对话流畅度与一致性上领先；学生模型在8步时FPS约15，RTF 1.93，质量可调。

**⚠️ 局限性**

局限包括语音相似度不足、VTP仅为高层语义指导而非独立控制、学生模型质量低于教师、增量生成仍非实时、对长时序内容的稳定性研究有限。

---

## 89. Critic-Free Pretraining for Efficient Online Reinforcement Learning Fine-Tuning

**arXiv ID:** 2608.10473 | [PDF](https://arxiv.org/pdf/2608.10473v1)

**作者:** Daoyi Li `[一作]` (Tsinghua University), Yu Wang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了离线到在线强化学习中的Critic训练问题，提出Critic‑Free Pretraining（CFP），即在离线阶段仅训练Actor，在线阶段使用新初始化的Critic并做短期warm‑up后再进行在线微调。

**💡 创新点**

创新点在于把Critic的训练从离线阶段完全剔除，使用Fresh Critic并通过warm‑up校准，从而消除离线Critic的价值偏差、提升在线学习效率，并显著降低离线计算和内存成本。

**🔧 技术方法**

使用行为克隆（BC）训练Actor，Flow Matching框架下的Flow‑based policy；在线阶段使用TD损失训练Critic，并在warm‑up期间将Actor与Critic一起更新。

**📊 数据集**

在8个稀疏奖励的离线数据集上进行评估：5个OGBench任务（Cube Double、Cube Triple、Cube Quadruple、Scene、Puzzle 4×4）以及3个Robomimic任务（Square、Pinwheel、...），使用默认play和Multi‑Human数据集。

**📈 对比分析**

对比传统O2O（离线+在线同时训练Actor和Critic）与CFP，实验显示CFP在OGBench的Cube Triple等任务上显著提升成功率（如QCFQL‑CFP达≈0.7），在Robomimic大多数任务与O2O相当且少数任务有所提升。

**⚠️ 局限性**

限制在于CFP并未在所有Robomimic任务上超越传统O2O，说明当离线Critic的价值误差不大时剔除Critic并无优势；需要进一步研究诊断何时适用CFP以及更有效的Critic初始化与warm‑up策略。

---

## 90. The impact of design factors of virtual and augmented reality on tertiary students user experience in Metaverse

**arXiv ID:** 2608.09940 | [PDF](https://arxiv.org/pdf/2608.09940v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 91. Leveraging Human Reading Behavior for Keyphrase Extraction: A Webcam-based Eye-tracking Corpus

**arXiv ID:** 2608.10688 | [PDF](https://arxiv.org/pdf/2608.10688v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 92. InterPruner: Interactive Structured Pruning via Taylor-Implicit Criterion and Language-Prior Modulator for Multimodal Object Detection

**arXiv ID:** 2608.10724 | [PDF](https://arxiv.org/pdf/2608.10724v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 93. Rationale-Guided Learning for Multimodal Emotion Recognition

**arXiv ID:** 2608.10448 | [PDF](https://arxiv.org/pdf/2608.10448v1)

**作者:** Sujung Oh `[一作]` (Sungkyunkwan University), Sangmin Lee `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于多模态大型语言模型生成结构化推理的情绪识别框架RGL，将情绪识别转化为人类认知推理过程。

**💡 创新点**

创新在于将双过程理论拆分为直觉、情境与综合三层推理，利用离线LLM生成推理银行并通过对比学习让模型内部表征与人类推理对齐。

**🔧 技术方法**

采用多模态LLM（GPT‑4o）生成推理，使用ViT‑base、RoBERTa‑large、HuBERT‑base等预训练编码器，双头结构，Transformer融合，硬负样本挖掘与对比损失。

**📊 数据集**

在IEMOCAP（二人对话）和MELD（多方对话）两个公开情感数据集上训练与评估。

**📈 对比分析**

与目前SOTA方法对比，RGL在IEMOCAP/ MELD上分别实现73.68/67.43的加权F1和73.51/68.31的准确率，均超过先前最高值。

**⚠️ 局限性**

主要局限在推理银行依赖LLM生成，生成质量与多样性受限；模型仍需大规模算力训练且缺少跨域鲁棒性验证。

---

## 94. Is This Your Final Answer? Cross-Contextual Consistency as a Measure of LLM Credibility

**arXiv ID:** 2608.10315 | [PDF](https://arxiv.org/pdf/2608.10315v1)

**作者:** Siyang Wu `[一作]` (University of Chicago), Bryon Aragam `[通讯]` (University of Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出跨上下文一致性指标C3，用来评估LLM回答在内容中性、主题对齐的提示扰动下的稳定性；

**💡 创新点**

创新点在于将答案分布的跨上下文变化量化为无内部信息的可信度信号，弥补了现有自我一致性、自我报告等方法对过度自信的缺陷；

**🔧 技术方法**

技术实现包括随机生成内容中性扰动提示、采样原始与扰动下的生成、使用最大均方差（MMD）等距离度量对分布差异归一化得到C3；

**📊 数据集**

在六大基准上评估：SVAMP、MMLU高中统计、CommonsenseQA、SimpleQA、FActScore和HumanEval；

**📈 对比分析**

与自我一致性、自我报告、改写一致性及FActScore等基线对比，C3在AUROC、ECE、AUPRC等指标上普遍优于基线，尤其在数学推理、事实检索与代码生成任务中显著提升；

**⚠️ 局限性**

局限性包括仅在有限模型与任务上验证、对不同语言或领域的泛化未作系统评估、采样依赖较多且C3本身不能保证答案绝对正确。

---

## 95. Uncertainty-Aware Ensemble Deep Randomized Neural Networks for Classification

**arXiv ID:** 2608.10007 | [PDF](https://arxiv.org/pdf/2608.10007v1)

**作者:** M. Sajid `[一作]` (Indian Institute of Technology Indore), M. Tanveer `[通讯]` (Indian Institute of Technology Indore)

**通讯引用:** 7879 | [OpenAlex ID](https://openalex.org/A5004222223)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于直觉主义模糊理论的深度与集成 dRVFL 网络（IF-dRVFL 与 IF-edRVFL），以提升对噪声和离群点的鲁棒性。

**💡 创新点**

创新点在于将直觉主义模糊成员度与非成员度联合用于样本加权，并将其嵌入深层随机化神经网络，形成单网络即兼具深度与集成的鲁棒架构。

**🔧 技术方法**

采用直觉主义模糊理论、Gaussian kernel、随机化权重、闭式解和多层/集成训练方式。

**📊 数据集**

使用 UCI 与 KEEL 公开基准数据集（共 13 个 KEEL、12 个 UCI），并在加噪声场景下进行测试。

**📈 对比分析**

通过与 11 个 SOTA RdNN 与模糊基准模型比较，采用平均准确率、标准差、排名及 Friedman–Nemenyi 统计检验，IF-edRVFL 与 IF-dRVFL 分别以 84.39% 和 84.32% 的平均准确率排名第一、第二，显著优于其他方法。

**⚠️ 局限性**

局限在于对 σ 与隐藏层数等超参数的敏感性，需要手工调参；并且仅在二分类/多分类数据集上验证，未探讨连续回归或大规模深度场景。

---

## 96. Rethinking Data Efficiency in Industrial Dense Prediction: Pretraining Coherence, Not Inductive Bias, Determines ViTs Low-Data Advantage

**arXiv ID:** 2608.10590 | [PDF](https://arxiv.org/pdf/2608.10590v1)

**作者:** Haoran Sui `[一作]` (ZTE Corporation), Yaoyuan Jia `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究 ViT 与 CNN 混合模型在工业稀疏标注环境下的低数据效率问题，并提出 AlignBlock 轻量级对齐模块以解决预训练分布不一致导致的特征不匹配。

**💡 创新点**

将预训练不一致（ImageNet 预训练的 ViT 与 COCO 预训练的 CNN neck）视为低数据性能瓶颈，提出同时补偿空间局部性、统计分布与优化动态的三重对齐策略，并通过 2×2 实验矩阵与统计检验进行定量分解。

**🔧 技术方法**

使用 LayerNorm 与 BatchNorm 的统计校准、3×3 卷积注入局部偏置、残差对齐、三阶段逐步微调以及 MMD/CKA 等特征相似性度量来实现跨架构兼容。

**📊 数据集**

在四个真实工业数据集（terminal、hook、safety‑belt）以及 COCO‑10 子集上进行实验，数据量从 100 至 703 样本不等。

**📈 对比分析**

通过 2×2 实验矩阵、配对 t‑检验与 mAP@50/95 的多维指标进行对比；结果显示在域相似度高且样本 ≥200 时 Swin‑Graft 超越 YOLOv11x，域偏离大时 CNN 仍保持显著优势。

**⚠️ 局限性**

仅验证了 Swin 结构，类别多样性有限，验证集规模小，未考虑多类/多场景或自监督预训练，且 AlignBlock 依赖固定金字塔层级，限制了对更广泛工业场景的适用性。

---

## 97. Detecting an Effect Is Not Learning to Act on It: A Reward-SNR Floor for LLM Acquisition Agents

**arXiv ID:** 2608.10441 | [PDF](https://arxiv.org/pdf/2608.10441v1)

**作者:** Ying Yuan `[一作]` `[通讯]` (University of California), Ying Yuan (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在成本高昂的辅助观测（如LLM结构化推理）下，学习何时获取该观测的可行性，并提出了奖励–信噪比检测阈值，说明平均效应检测与个体获取策略学习的本质区别。

**💡 创新点**

创新点在于引入奖励–信噪比检测阈值，阐明平均效应检测与个体获取策略学习的差异，并提出结构化假设嵌入（SHE）作为可解释且可校准的LLM特征。

**🔧 技术方法**

主要技术包括冻结LLM生成K个带置信度与证据索引的意图假设、文本嵌入与最大相似度聚合形成输入分支，结合GRU/Mean-Pool/SASRec等基线，离线奖励SNR估计与正负控制，以及统计显著性检验与置信区间。

**📊 数据集**

使用了公开的MIND（新闻）、Amazon-Beauty（电商）与REES46（电商）三大数据集，覆盖内容丰富与稀疏两类。

**📈 对比分析**

在不同基线与历史长度/多意图切片下评估SHE对NDCG@10的提升，发现SHE在弱基线和稀疏/多意图子域有显著正向提升，但总体冗余增益接近零；学习获取策略无显著优势，表明低SNR导致无法从离线奖励中学习。

**⚠️ 局限性**

局限性包括检测阈值仅为必要条件，未对实验变量进行严格控制；SHE在内容稀疏数据中无效；仅在公开数据上验证，实际生产迁移性仍需进一步探索。

---

## 98. Boundary-Seeking Policy Gradient for Safe Reinforcement Learning

**arXiv ID:** 2608.10204 | [PDF](https://arxiv.org/pdf/2608.10204v1)

**作者:** Chenhua Fan `[一作]` (Washington State University), Honghao Wei `[通讯]` (Washington State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出Boundary-Seeking Policy Gradient (BSPG)，一种在安全强化学习中显式驱动策略更新至活跃约束边界并沿其优化奖励的方法。

**💡 创新点**

创新点在于将奖励梯度拆分为切向和法向两部分，法向部分利用约束残差实现双向边界吸引，并通过隐式拉格朗日乘子实现无学习双重变量的自适应权重。

**🔧 技术方法**

采用政策梯度理论、占优测度线性规划分析、正交梯度分解以及隐式拉格朗日解释，结合PPO框架实现具体算法。

**📊 数据集**

在SafetyPointGoal1-v0（OmniSafe安全导航任务）上进行实验。

**📈 对比分析**

与CRPO和ESPO等基准方法比较，BSPG在奖励最高、成本接近阈值、边界残差最小方面均优于对手。

**⚠️ 局限性**

局限性包括仅在理想梯度下给出收敛理论，对随机PPO实现未给出样本复杂度，且未在多种环境下验证泛化性。

---

## 99. User Satisfaction-Aware Resource Allocation with Prospect-Theoretic Utility for Multimedia Streaming

**arXiv ID:** 2608.10625 | [PDF](https://arxiv.org/pdf/2608.10625v1)

**作者:** Manoj Kumar S `[一作]` (Indian Institute of Technology Madras), Avhishek Chatterjee `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于门限的资源分配策略，并通过引入前景理论构造用户满意度的非对称效用指标，在多用户多类情形下实现了对资源分配的动态控制；

**💡 创新点**

创新点在于：①将前景理论的损失厌恶特征融入QoE度量；②设计门限阈值策略以降低资源下降的频率与幅度；③基于M/M/∞队列提供精确与近似的快速计算阈值与权重的方法；

**🔧 技术方法**

采用M/M/∞排队模型、马尔可夫链与重奖理论进行解析推导；利用数值线性方程组与二分搜索实现阈值与权重的最优计算；并提出“旁观者”近似简化模型；

**📊 数据集**

未使用真实网络数据集，而是通过设定参数（如λ、μ、B、α、β、γ）进行仿真验证，验证理论与仿真结果的一致性；

**📈 对比分析**

与传统公平分配（PS/FA）做对比，采用仿真与解析得到的整体性能指标（M(F)）进行评估；阈值策略在所有测试参数下均优于公平分配，且解析方法比仿真快约100倍，旁观者方法快约35倍；

**⚠️ 局限性**

局限性：假设无限服务器且忽略信道衰落、调制编码变化；使用泊松到达、指数服务时长；前景参数需先验估计；仅考虑两类用户，未考虑更复杂的服务级别与实际网络环境中的多样性。

---

## 100. Persona Conditioning as an Assessor-Sensitivity Probe for LLM-Based IR Evaluation

**arXiv ID:** 2608.10385 | [PDF](https://arxiv.org/pdf/2608.10385v1)

**作者:** Samaneh Mohtadi `[一作]` (University of Queensland), Gianluca Demartini `[通讯]` (University of Queensland)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在LLM评判器中加入任务导向的角色化（Persona Conditioning），系统评估了评判视角变化对信息检索(IR)相关性判断和系统排名的影响。

**💡 创新点**

创新点在于将角色化视作一种诊断探针，而非仅用于提升标签质量；同时比较抽象PersonaHub与技能导向USPersona两种来源的影响，并在大模型与小模型之间揭示了显著的模型容量依赖性。

**🔧 技术方法**

采用了基于摘要的评判框架、标准化的UMBRELA提示，并在多种LLM骨干（GPT‑4o、GPT‑4o‑mini、LLaMA‑3.1‑70B/8B、Qwen‑2.5‑72B/7B）上执行四个角色化条件（Query、Domain、Orthogonal、Evidence、Global）。

**📊 数据集**

使用TREC Deep Learning 2020（DL20）与TREC Retrieval‑Augmented Generation 2024（RAG24）两个基准数据集，包含数百条查询与多系统提交结果。

**📈 对比分析**

评估方法包括：与UMBRELA标签的加权Kappa一致性、与人工标注的Kappa对比、系统级NDCG@10、Kendall τ与Rank‑Biased Overlap (RBO)对排名一致性的衡量，以及对个别系统的Rank‑Displacement分析；实验显示高容量模型保持了高达0.94‑0.95的τ和0.86‑0.99的RBO，说明整体排名稳定；但在小模型上，角色化引起平均绝对Rank‑Shift可达5级，显著提升评估敏感度。

**⚠️ 局限性**

主要局限包括：角色化效果高度依赖模型容量，低容量模型可能产生过度噪声；摘要式评判可能忽略原文细节导致判断失真；仅考虑两种Persona来源，未涵盖更广泛的人格或文化维度；最后，本研究仅聚焦LLM评判器，未验证与真实人工评审的交叉一致性。

---

## 101. Rethinking LLM Verification: Evidence Structure, Uncertainty, and Selective Refinement

**arXiv ID:** 2608.10725 | [PDF](https://arxiv.org/pdf/2608.10725v1)

**作者:** Uma Ranjan `[一作]` (Indian Institute of Technology Jammu), Amit Sharma `[通讯]` (Microsoft Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种两阶段医学假设验证框架，先让LLM生成自我推理轨迹，再在模型因不确定而放弃（abstain）时，通过SNOMED CT检索提供局部本体知识，最终重新评估各选项。

**💡 创新点**

创新点在于将不确定性（abstention）作为触发器，动态调用本体检索而非始终使用外部知识图，既提升了准确率，又避免了构建完整知识图的成本。

**🔧 技术方法**

采用的大型语言模型（GPT‑5.5、DeepSeek‑R1）生成推理轨迹与置信度、基于BioPortal的SNOMED CT检索、再评估策略，以及统计显著性检验（McNemar）进行效果评估。

**📊 数据集**

使用的公开数据集包括 MedReason（含KG推理轨迹的1000题）和 MedQA（USMLE多选的1000题）。

**📈 对比分析**

通过与仅使用世界知识、KG‑grounded 轨迹以及无检索基线对比，MedReason 上的问句级准确率从 87.8% 提升到 96.2%（+8.4pp），假设级提升 4.2pp；MedQA 上提升约 1–2pp，覆盖率接近 100%，且所有提升均具有高度统计显著性。

**⚠️ 局限性**

主要局限在于对 SNOMED CT 检索质量的依赖，检索失败或不相关会削弱提升效果；此外，过度依赖模型自报置信度和 abstention 作为安全信号在临床实际应用中仍存在风险。

---

## 102. Efficient Weak-Entropy PINN for Solving Hyperbolic Conservation Laws

**arXiv ID:** 2608.10389 | [PDF](https://arxiv.org/pdf/2608.10389v1)

**作者:** Qi Gao `[一作]` (Columbia University), Xuan Di `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于弱熵条件的物理信息神经网络（WEPINN），用于求解可出现断裂解的双曲守恒律。

**💡 创新点**

创新点包括：① 将守恒律的弱形式与熵条件同时嵌入损失函数；② 通过预选正交三角函数测试函数并利用离散快速傅里叶变换（DFFT）实现高效积分；③ 在保持物理可接受解的同时避免人工粘性或光滑化。

**🔧 技术方法**

核心技术包括：物理信息神经网络（PINN）、弱积分表述、熵条件约束、离散快速傅里叶变换（DFFT）以及多尺度测试函数设计。

**📊 数据集**

实验数据集主要为数值基准问题：一维线性广告、无粘性Burgers、LWR交通模型、一维Euler压缩气体模型，以及二维Burgers方程，涵盖多种初始条件（Sigmoid、Riemann、Fourier、Trigonometric、PWC、圆盘、平滑三角波）和边界条件（周期、Dirichlet）。

**📈 对比分析**

与 Diff‑PINN、VPINN 和 WPINN 进行对比。WEPINN 在 L^2 误差、冲击检测率（S‑Rate）和冲击位置精度（S‑Acc）上均优于基线，尤其在出现冲击、相互作用和稀疏波的非线性问题中表现突出；在二维Burgers上也显著低于 Diff‑PINN 与 VPINN 的误差。

**⚠️ 局限性**

局限性包括：① 仅在周期或 Dirichlet 边界下实验，缺乏通用边界处理；② WPINN 在多维实现缺失，导致与其比较受限；③ 对极端高频或大尺度问题的可扩展性仍待验证；④ 训练中仍需手工调参和测试函数阶数，影响自动化程度。

---

## 103. Beyond Fixed Luminance: Towards Panchromatic and Orthochromatic Image Colorization

**arXiv ID:** 2608.10798 | [PDF](https://arxiv.org/pdf/2608.10798v1)

**作者:** Swarnim Maheshwari `[一作]` (Indian Institute of Technology Hyderabad), Vineeth N. Balasubramanian `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个无亮度限制的图像着色框架，使用基础图像编辑模型直接从灰度图像生成全RGB，而不是在Lab空间仅预测色度。

**💡 创新点**

创新点在于：①将着色视为完整RGB图像编辑任务；②引入混合灰度目标，训练模型同时处理标准panchromatic灰度和仿赤色不敏感的orthochromatic灰度，并通过文本提示让模型动态重新解释亮度；③利用现成的图像编辑基础模型（FLUX.2-klein），避免对文本到图像模型进行架构改造。

**🔧 技术方法**

技术细节包括：使用FLUX.2-klein-4B扩散模型和其DiT组件；对DiT使用LoRA微调（rank=16, alpha=16，约0.5%参数）；采用AdamW、bfloat16混合精度、梯度检查点；训练目标为混合灰度损失，文本提示为“colorize”或“colorize ortho”加标题；在推理时直接输出RGB，无亮度替换。

**📊 数据集**

数据集：ImageNet（前5k个验证图像）、COCO（前5k个测试图像）、Multi‑Instance（7213张验证图像）。图像灰度化采用标准luminance和自定义orthochromatic公式，并使用BLIP2或人工标题。

**📈 对比分析**

与BigColor、COCO‑LC、DDColor、DISCO、UniColor等基线在COCO、ImageNet和Multi‑Instance三大基准上进行比较。评估指标包括FID、sFID、FID‑DINO、Colorfulness、Col‑diverse、Saturation、ColorNet，以及人工artifact‑free率。结果显示：在标准panchromatic灰度下，性能与最强基线相当；在simulated orthochromatic灰度下，方法在大多数指标上显著优于基线，artifact‑free率最高，结构保真度（SSIM）亦保持良好。

**⚠️ 局限性**

局限性：仅通过LoRA微调，未进行完整微调；orthochromatic模拟使用简单公式，无法完全覆盖真实历史胶片的光谱敏感度、衰变与化学退化；对更广泛的灰度/颜色失真类型和多重修复任务仍需进一步研究。

---

## 104. 4D-WAM: 4D Consistent World Modeling for Autonomous Driving

**arXiv ID:** 2608.10107 | [PDF](https://arxiv.org/pdf/2608.10107v1)

**作者:** Jiacheng Fu `[一作]` (University of Science and Technology of China), Zhiwei Xiong `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出4D一致性世界动作模型4D-WAM，利用几何基础模型在训练阶段对视频-动作联合预测进行4D一致性监督，解决传统WAM在4D场景中的不一致问题。

**💡 创新点**

创新点包括：①将冻结的VGGT-Ω几何模型的特征与深度作为4D一致性损失，逼迫模型学习真正的4D结构；②发现WAM早期决策现象并提出决策导向时间步采样，聚焦高噪声阶段的决策形成；③设计Mixture-of-Transformers骨干与异向注意掩码，实现视频与动作的充分交互。

**🔧 技术方法**

技术手段包括扩散式视频-动作联合预测、Mixture-of-Transformers + 异向注意、VGGT-Ω的几何特征/深度监督、4D一致性损失（特征+深度）、决策导向时间步采样。

**📊 数据集**

使用NAVSIM v1与v2的navtest和navhard评测数据集进行训练与评估，训练阶段采用navtrain数据。

**📈 对比分析**

与TransFuser、DiffusionDrive、DriveVLA-W0、DriveFine、Epona等现有方法在NAVSIM v1/v2的PDMS/EPDMS指标上对比，4D-WAM在navtest EPDMS/PDMS分别达90.6/90.9，navhard EPDMS最高35.9，显著超过所有竞争者，显示更优的安全性、合规性与行车进度。

**⚠️ 局限性**

局限性包括：依赖大规模GPU训练与丰富的多视角训练数据；当前仅支持单摄像头输入；对极端动态物体或稀疏场景的超大运动仍可能产生误判；推理阶段需10步迭代，尽管可缩短至2步仍有推理延迟。

---

## 105. Conversational Orchestration for Organic 6G

**arXiv ID:** 2608.10714 | [PDF](https://arxiv.org/pdf/2608.10714v1)

**作者:** Masoud Shokrnezhad `[一作]` (ICTFICIAL Oy), Tarik Taleb `[通讯]` (Ruhr University Bochum)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种轻量级、去中心化的对话式编排框架，利用大型语言模型驱动的域级代理实现 Organic 6G 中多域资源的服务部署与动态迁移；

**💡 创新点**

创新点在于将编排复杂度从传统集成网络迁移到目标驱动的 LLM 代理，采用邻域级资源可达性广播与事件驱动协商的两层通信，辅以自校验 RL 训练和在线影子更新，实现在域动态加入/离开时的可扩展、简洁与敏捷；

**🔧 技术方法**

使用技术包括 LLM 代理（DeepSeek‑R1‑Distill‑Qwen‑7B）、自校验多目标强化学习（GDPO）、A2A 控制平面覆盖网络、压缩可达性广告、分布式可达性表以及在线影子更新机制；

**📊 数据集**

采用合成的多域异构网络数据集，共 3000 个服务部署场景（12 个域、可变延迟、带宽与计算池）进行训练与评估；

**📈 对比分析**

通过与强基线 LLM（DeepSeek‑R1）对比，离线 RL 训练后模型接近 100% 的得分；在目标切换后在线影子更新能恢复性能；控制平面消息量随域数呈近线性增长，域加入时的收敛时间在数十个管理周期内；

**⚠️ 局限性**

局限性包括：缺乏正式的可扩展性理论界限；对不确定性与尾部风险保障不足；A2A 交互的安全与欺骗防护未深入研究；实验仅基于合成数据，真实部署与更大规模网络的评估尚待验证。

---

## 106. Intrinsic Structure: Spectral Identifiability for Mechanistic Interpretability

**arXiv ID:** 2608.10172 | [PDF](https://arxiv.org/pdf/2608.10172v1)

**作者:** Ashim Dhor `[一作]` (IISER Bhopal), Pin-Yu Chen `[通讯]` (IBM Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的机制可解释性方法，通过将字典学习与可识别性相结合，探讨了神经网络模型内部电路的本质属性。

**💡 创新点**

创新点在于首次提出了一个可识别性定理，证明了在一定条件下，Koopman谱可以从有限样本中以M^-1/2的速率恢复，且谱是模型固有的特征。

**🔧 技术方法**

使用了Koopman算子和扩展动态模式分解（EDMDc）等技术，构建了一个控制的动态系统模型。

**📊 数据集**

使用了三个预训练的变换器模型（GPT-2 small、Gemma-2-2B和Qwen3-8B-Base）进行实验，数据集为WikiText-103。

**📈 对比分析**

与随机方向和主成分分析（PCA）进行了比较，Koopman模式在间接对象识别上表现优于随机方向，但在某些情况下不如主成分，且随着深度距离的增加，差距逐渐减小。

**⚠️ 局限性**

限制在于非正态性会导致激活的主方向与Koopman模式之间的偏离，且在不同的训练种子和字典宽度下，模型的可解释性结果可能会有显著差异。

---

## 107. Protection Levels for Vision-Based Pose Estimation

**arXiv ID:** 2608.10023 | [PDF](https://arxiv.org/pdf/2608.10023v1)

**作者:** Olivia Beyer Bruvik `[一作]` (Stanford University), Mykel J. Kochenderfer `[通讯]` (Stanford University)

**通讯引用:** 13006 | [OpenAlex ID](https://openalex.org/A5068326377)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出将RAIM的保护水平方法应用于基于视觉的姿态估计（PnP），并给出了可直接用于六自由度姿态的概率保护水平计算

**💡 创新点**

创新地将残差检测RAIM与非线性PnP结合，推导出可覆盖未检测故障的保护水平，并分析噪声、冗余和姿态维度对保护水平的影响

**🔧 技术方法**

采用残差检测RAIM、PnP求解、误差传播、非中心χ²分布及误差上界分析等技术

**📊 数据集**

实验使用合成跑道图像（四角点及额外边缘点）进行仿真，未来计划使用LARD数据集进行验证

**📈 对比分析**

通过理论推导与仿真曲线对比，验证保护水平随噪声线性变化、随关键点数量以-1/2衰减，单关键点故障与多关键点故障对比，结果与预期一致

**⚠️ 局限性**

假设关键点误差独立且Gaussian，未在真实图像上验证，需校准关键点失效概率和不确定性，且对姿态错误的检测能力有限

---

## 108. Machine Shape and Hierarchical Blocking: A Mathematics of Arrays Formalization, with an Open Problem in Hierarchical Shape Occupancy

**arXiv ID:** 2608.10119 | [PDF](https://arxiv.org/pdf/2608.10119v1)

**作者:** Lenore M Mullin `[一作]` `[通讯]`, Lenore M Mullin

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

基于MoA的递归推导，提出机器形状机理，自动决定稠密矩阵乘法的分块与预取策略；

**💡 创新点**

创新点在于将机器缓存层级视为形状，并提出递归的Γ算子，利用容量与延迟隐藏条件自动推导多级分块；

**🔧 技术方法**

使用Mathematics of Arrays (MoA)、递归形状推导、缓存层级模型、预取隐藏与实验验证技术；

**📊 数据集**

使用Apple M1 Pro、AWS Graviton4、Delta CPU（双插槽AMD）等多种真实硬件平台；

**📈 对比分析**

通过在多台机器上对比实验测得的块尺寸、预取效率和吞吐量，证明递归模型能恢复已知校准值，但不同架构间占用比例σ_i不一致，吞吐量预测需单次测量；

**⚠️ 局限性**

局限在于对虚拟化环境不确定、网络层级未自动选择通信算法、占用比例σ_i依赖架构且未完全可推导、不同机器间的η(P)需单独校准。

---

## 109. SPIEval: Evaluating Large Language Models as Mobile Assistants over Scattered Personal Information

**arXiv ID:** 2608.10692 | [PDF](https://arxiv.org/pdf/2608.10692v1)

**作者:** Junjie Ye `[一作]` (Fudan University), Pluto Zhou `[通讯]` (Tencent Hunyuan Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个面向移动助手的基准，评估LLM在多应用场景下利用分散个人信息完成任务的能力。

**💡 创新点**

基准由人手构造，涵盖五种认知能力（推理、消歧、整合、偏好推断、多意图拆分），包含250个任务、4335条记录、21种工具，并提供可控环境与可验证结果。

**🔧 技术方法**

采用多轮工具调用框架，将检索与执行工具集成进LLM交互；通过工具调用序列与参数级别评估模型表现。

**📊 数据集**

使用SPIEval数据集，包含10个模拟手机应用（通讯录、会议、短信等）的结构化记录，共4335条记录，覆盖250个任务。

**📈 对比分析**

通过与黄金答案对齐的二元准确率比较模型，最高模型GPT‑5.5的整体准确率仅为57.3%，其余模型落在16%–53%之间，显示信息定位是主要瓶颈。

**⚠️ 局限性**

主要限制包括：LLM在定位信息时过早停检导致错误、极少使用高级检索方法、在偏好推断与多意图拆分任务上表现不佳，整体对个人信息分散场景适应性不足。

---

## 110. Not a Monolith: Lab-Level Divergence in the Cooperative Equilibria of Chinese Frontier LLM Agents

**arXiv ID:** 2608.10262 | [PDF](https://arxiv.org/pdf/2608.10262v1)

**作者:** Francisco León Zúñiga Bolívar `[一作]` `[通讯]` (Institucion Universitaria Colegio Mayor del Cauca), Francisco León Zúñiga Bolívar (Institucion Universitaria Colegio Mayor del Cauca)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比四个中国前沿模型在迭代囚徒困境中的合作倾向，验证其与西方模型是否相同，并检验中国模型是否可视为一个整体。

**💡 创新点**

采用固定转换器消除代码转换偏差，首次证明中国模型的行为差异主要来源于实验室而非生态系统；提出了可在跨实验室研究中控制编码能力的实验设计。

**🔧 技术方法**

使用迭代囚徒困境、Moran进化过程、Axelrod锦标赛以及自我修订提示；通过固定的GPT‑5.4 Mini作为自然语言到代码的转换器实现策略生成。

**📊 数据集**

四个中国实验室的前沿模型：DeepSeek V4 Pro、Qwen3‑Max、Kimi K2.5、GLM‑5.1；为每个实验室、每种提示生成25条策略，共计1,800条自然语言策略。

**📈 对比分析**

在四种种群结构（平衡无噪声、偏向无噪声、平衡噪声、偏向噪声）下进行500次Moran模拟，比较各实验室的攻击平衡占比、合作多样性与噪声敏感度；结果显示四实验室在攻击平衡占比上显著差异，合作优势与西方模型相当。

**⚠️ 局限性**

局限包括：仅评估四个实验室，未在同一固定转换器下重新跑西方模型；使用英文提示，未探究中文提示对结果的影响；固定转换器对合作/中立占比有一定噪声影响；样本量有限，统计显著性受限。

---

## 111. Embodied Multimodal Grounding for Open-Vocabulary Mobile Manipulation via Semantic 3D Gaussian Splatting

**arXiv ID:** 2608.10756 | [PDF](https://arxiv.org/pdf/2608.10756v1)

**作者:** Huosen Ou `[一作]` (Hong Kong University of Science and Technology), Yiding Ji `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一个可刷新语义三维高斯场（Semantic‑3DGS）并将其作为跨模态感知与动作的共享接口，实现了开放词汇目标定位与少样本移动机器人抓取与放置；

**💡 创新点**

创新点在于：①利用少量主动摄像头视角快速构建局部Semantic‑3DGS，并将其用于多任务感知、语言定位、障碍感知、基座姿态规划与后置扩散式VLA策略的稀疏注入；②采用后置（Late‑Block）语义注入方式保留预训练动作先验；③结合可达性意识的基座姿态控制提升长程与高程适应性；

**🔧 技术方法**

使用了Semantic‑3D Gaussian Splatting、VGGT几何初始化、CLIP/DINO语义蒸馏、SAM分割、后置语义适配器、Diffusion‑based VLA（DexVLA风格）以及PPO基座姿态学习；

**📊 数据集**

使用了公开的CLIP/DINO预训练模型、VGGT预训练模型，外加10条真实机器人演示数据（每个任务）进行少样本训练与评估；

**📈 对比分析**

与PointVLA、DexVLA两种基线在五组真实机器人任务（少样本多任务、长程任务、高度适应、照片欺骗、杂乱场景）进行对比。完整系统在少样本多任务上平均成功率81.7%（vs 64.0%/37.7%），长程任务60%成功（vs 40%/28%），高度适应75‑80%成功率（vs 35‑48%），照片欺骗88%成功率（vs 80%/78%），杂乱场景74%成功率（vs 52%/46%），显著提升；

**⚠️ 局限性**

系统仅适用于短时主动感知的准静态任务，语义场与VLA推理需在离线GPU上完成，无法实时在低级控制循环内更新；不支持高速动态交互或零样本任意抓取；对基座姿态规划与语义注入有依赖，无法在极端环境下保证通用性。

---

## 112. MIDAS: Mutual Information Disentanglement with Uncertainty-Aware Fusion for Incomplete Multimodal Sentiment Analysis

**arXiv ID:** 2608.09986 | [PDF](https://arxiv.org/pdf/2608.09986v1)

**作者:** Yuhua Wen `[一作]` (Beijing University of Posts and Telecommunications), Ya Li `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 23461 | [OpenAlex ID](https://openalex.org/A5100404103)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对多模态情感分析中常见的缺失模态问题，提出一种基于变分建模、互信息最小化/最大化以及不确定性感知融合的MIDAS框架。

**💡 创新点**

创新点在于将不完整模态通过变分方式分解为共享与专属潜在分布，利用互信息最小化实现解耦、互信息最大化实现跨模态对齐，并直接使用后验方差作为不确定性权重进行融合。

**🔧 技术方法**

采用变分自编码器、互信息估计（JSD/InfoNCE）、注意力融合、贝叶斯不确定性推断以及多任务损失（任务、解耦、对齐、预测、重建）。

**📊 数据集**

在三大公开对话情感数据集 MOSI、MOSEI、CH‑SIMS 上进行实验。

**📈 对比分析**

与多种完整/不完整多模态基线（MISA、TFR‑Net、EMT‑DLFR、LNLN、P‑RMF 等）比较，MIDAS 在所有缺失率下均取得更高的分类准确率和相关系数，尤其在中高缺失率下显著优于竞争者。

**⚠️ 局限性**

局限包括：在极端缺失率（>90%）下性能衰减；缺失模式仅为随机遮掩，未覆盖实际异构、时变缺失；缺乏大规模预训练验证；对模型可解释性与训练稳定性的进一步研究仍需完善。

---

## 113. Actionable Hallucination Detection: Translating Latent Uncertainty into Agentic Critique

**arXiv ID:** 2608.10430 | [PDF](https://arxiv.org/pdf/2608.10430v1)

**作者:** Sanidhya Vijayvargiya `[一作]` (Samsung Research America), Rahul Lokesh `[通讯]` (Samsung Research America)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Latent Critic，一种在LLM生成过程中并行运行的低秩适配器，用于实时检测和定位工具调用中的规范化错误（specification‑grounding failures）；

**💡 创新点**

创新点在于：1) 通过对Transformer残差流的主动重构，将内部不确定性转化为可言语化的精确错误定位；2) 使用掩码诊断目标实现低秩重构，保持原模型行为不变；3) 将检测结果直接融入工具调用序列，实现零延迟、可操作的安全屏障；

**🔧 技术方法**

技术包括：LoRA参数高效微调、触发令牌（trigger token）结构化输出、激活补丁与层级探测、内部线性可分性分析、ReAct闭环代理实验；

**📊 数据集**

使用自建的模拟用户/工具调用数据集（5,000训练场景 + 500测试场景 + 200 ToolAlpaca OOD场景），覆盖Qwen3-4B、Llama‑xLAM‑2‑8B等基础模型；

**📈 对比分析**

与外部判别器、线性探针、Token/Semantic Entropy 等基线进行对比，Latent Critic 在 ID 条件下 AUROC 0.966、定位精度>80%，在 OOD 仍保持 0.925 AUROC，显著优于其他方法；在闭环代理实验中，精确反馈提升参数级 F1 至 61.2%（ID）/36.7%（OOD），误阻率降至 2.9%/13.2%，而通用判别仅提升精度却降低召回；

**⚠️ 局限性**

局限性包括：1) 训练标签来源于模拟环境，需人工验证；2) 依赖基础模型内部表示，若基础模型生成质量差，检测能力受限；3) 对开放式文本生成的跨度定位仍未解决；4) 在某些模型下对 OOD 的优势不显著，需进一步研究模型依赖性。

---

## 114. Quantum Incremental Learning with Mixed State Prototypes

**arXiv ID:** 2608.10464 | [PDF](https://arxiv.org/pdf/2608.10464v1)

**作者:** Yu Wu `[一作]` (Northwestern Polytechnical University), Witold Pedrycz `[通讯]` (University of Alberta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于混合态原型的量子增量学习框架，实现了在有限量子电路宽度下的类别增量学习。

**💡 创新点**

创新点包括：① 用混合态原型（CCPS）代替传统纯态，能够以低秩可调方式表示类分布并提供噪声抑制；② 通过HS距离与SWAP测试实现无量子读出权重化的距离判别；③ 设计可扩展的经典模块，避免在添加新类时扩展量子电路。

**🔧 技术方法**

核心技术包括：量子卷积网络（QCNN）作为特征提取器；混合态原型（CCPS）和Softmax权重参数化；HS距离与SWAP测试进行量子重叠测量；经典预处理、残差式MLP辅助学习；经验缓冲区与原型重放策略。

**📊 数据集**

实验使用CIFAR-100（四个32类子集）和TinyImageNet，构成16类初始任务后逐步增加4类，完成5个增量阶段。

**📈 对比分析**

与多种经典基线（线性Softmax、Cosine、Nearest Centroid、RBF‑SVM）以及量子基线（HEA‑VQC、QCNN、TTN‑QNN、Fidelity Classifier）对比；在非增量任务中，混合态原型取得平均精度0.7827，优于所有量子基线；在增量任务中，性能略低于最强经典基线，但相较于传统经典方法表现稳定，证明在受限资源下仍具备可行性。

**⚠️ 局限性**

局限性包括：实验仅在8/4 qubit模拟器上验证，未在真实硬件噪声环境中测试；对原型秩K需手动设置，过大或过小均会影响性能；在更大类别数或更复杂任务上的可扩展性与稳健性尚未评估。

---

## 115. A Graph Neural Network--Guided Genetic Algorithm for Physical Internet Supply Chain Optimization under Cost Uncertainty

**arXiv ID:** 2608.10245 | [PDF](https://arxiv.org/pdf/2608.10245v1)

**作者:** Faezeh Ardali `[一作]` (Louisiana State University), Gerald M. Knapp `[通讯]` (Louisiana State University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了物理互联网（PI）物流网络的确定性与最小最大后悔（min‑max regret）规划模型，并开发了一种基于图神经网络（GNN）的遗传算法（GNN‑GA），通过GNN学习工厂–枢纽分配概率来引导初始种群、调整变异，结合精确的线性规划评估连续流，实现了在不确定成本下的协同供应链优化。

**💡 创新点**

创新点在于：①仅利用GNN进行离散分配决策的概率引导而不替代精确优化；②将预测熵用于变异控制；③在紧预算下验证学习初始化的优势，并在更大预算下证明完整进化过程仍优于传统GA；④通过在多场景后悔模型中进行实验，展示了GNN‑GA对不确定性规划的适用性。

**🔧 技术方法**

技术手段包括异构图神经网络（heterogeneous GNN）实现节点特征编码与关系消息传递、遗传算法（GA）与熵引导变异、模拟退火（SA）对照实验、HiGHS线性规划求解器、PyTorch框架、Python实现与单线程求解。

**📊 数据集**

数据集为15个人工生成的单周期PI网络实例，尺寸从(1,3,2)到(15,90,700)不等，所有工厂–枢纽对均可选，枢纽之间全连通；训练集包含实例1‑9，模型选择集10‑12，独立测试集13‑15，以及另外3个独立可精确求解的实例，用于验证模型迁移性。

**📈 对比分析**

对照方法包括传统GA、模拟退火（SA）和仅按成本排序的热启动；在相同随机种子和评估预算下进行匹配对比；实验显示GNN‑GA在所有20个对照实验（实例13、15）中均优于成本热启动，在扩展预算（实例13、400次评估）中每次匹配运行均胜过GA；统计检验表明差异显著，且在大多数实例中GNN‑GA实现更低目标值，运行时间相对可接受。

**⚠️ 局限性**

局限性包括：仅在全连通、完全可行的合成单周期网络上验证；只考虑三种协同成本场景，未涵盖更复杂的不确定性分布；GNN仅用于分配决策，对连续流仍依赖精确求解，可能在规模更大或稀疏网络中表现下降；对后悔优化的训练样本有限，导致在某些困难实例上表现不如SA。

---

## 116. Procedural Fairness Failures in RLHF from Preference Averaging

**arXiv ID:** 2608.10126 | [PDF](https://arxiv.org/pdf/2608.10126v1)

**作者:** M P V S Gopinadh `[一作]` (Vishnu Institute of Technology), Srinivasa Raju Rudraraju `[通讯]` (Vishnu Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了RLHF在多样化人类偏好下的公平性问题，提出并评估了Preference-Aware RLHF（PA-RLHF），通过在奖励学习阶段对偏好模式进行分离来减少公平性差距。

**💡 创新点**

将偏好聚合视为导致程序公平性失败的结构原因，提出在奖励学习阶段分离偏好模式以保留多样性，从而实现程序公平性。

**🔧 技术方法**

K-means聚类对句子嵌入进行偏好模式分割；SBERT (all-MiniLM-L6-v2) 嵌入；逻辑回归奖励模型；Bradley-Terry 风格的对比偏好学习；仅评估奖励模型而不进行完整策略优化。

**📊 数据集**

模拟的971对比较数据，来自60个模拟评审员，20个提示，三类偏好（简洁、详细、技术/正式）共20个评审员每类20人。

**📈 对比分析**

对比标准RLHF与PA-RLHF的组别级对齐准确率；标准RLHF整体准确率46.9%，PA-RLHF提升至67.9%；公平性差距从15.9pp降至9.6pp；少数组提升27.6-32.8pp，主体组提升7.3pp。

**⚠️ 局限性**

受限于控制实验，聚类方法固定，未考虑自适应模式选择；仅在奖励学习阶段干预；未评估在真实生产数据、完整策略优化和多步决策中的效果；剩余公平差距仍存在。

---

## 117. How Robust Are LLMs to Vietnamese Dialects?

**arXiv ID:** 2608.10414 | [PDF](https://arxiv.org/pdf/2608.10414v1)

**作者:** Minh Tran `[一作]` (University of Science, VNU-HCM), Duc Hoang `[通讯]` (MIT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了 VialectBench 基准，评估大型语言模型在越南语六个方言变体下对情感识别、自然语言推理、问答和多选问答四个任务的鲁棒性。

**💡 创新点**

首次提供人工注释的对照式方言平行语料，系统化测评不同方言对模型表现的影响，并揭示任务与地区的差异化鲁棒性。

**🔧 技术方法**

使用人工重写、双人质量审核、指令调优的 LLM、确定性解码、准确率/ token‑F1、HFR 等评估指标进行实验。

**📊 数据集**

选取 UIT‑VSMEC（ER）、ViANLI（NLI）、ViQuAD（QA）与 ViMMRC2.0（MCQA）四大数据集的 400 条标准实例，再人工生成 2,400 条方言重写。

**📈 对比分析**

在十款指令调优模型上做标准与方言对照实验，平均性能下降 2.82%，QA 受损最大，PNT3/PNT2 方言造成最高降幅，GPT‑4o 仍保持最高稳健性。

**⚠️ 局限性**

样本采样偏向方言敏感词、单一参考模型的困惑度评估可能不具普适性，且未测试轻量化的方言修复或预处理方法。

---

## 118. ProbGuard: Calibrated Safety Risk Estimation from LLM Output Distributions

**arXiv ID:** 2608.10621 | [PDF](https://arxiv.org/pdf/2608.10621v1)

**作者:** Xinzhe Huang `[一作]` (Zhejiang University), Tianhang Zheng `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种全概率、架构无关的安全防护框架 ProbGuard，利用 LLM 早期输出分布信息来估计和校准后续生成产生不安全响应的风险，并实现对不安全输出的提前终止。

**💡 创新点**

创新点包括：①把安全风险建模为“后续生成产生不安全响应的概率”，并通过 Monte‑Carlo 采样得到校准目标；②设计了概率加权表征（probability‑weighted representation），将不同 LLM 的输出分布映射到统一嵌入空间，实现跨模型的输入兼容；③使用负对数似然损失对 ProbGuard 进行后验校准，使其输出为可解释的安全概率；④不依赖 LLM 隐藏层或完整文本输出，降低对目标模型架构的依赖。

**🔧 技术方法**

核心技术包括：Monte‑Carlo 采样（N=16）估计安全概率；概率加权嵌入表征（top‑K token、重token化、平均嵌入）；使用 CalibEval 训练出的判别模型评估采样样本的安全性；负对数似然损失进行训练；与多种基线对比评估 Brier、ECE 等校准指标；利用 GPT‑5 作为最终安全判断器评估攻击成功率。

**📊 数据集**

使用了三类数据集：①训练集（PKU、WildGuard、SEval 共 3000 条恶意示例）；②评估集（每个数据集 1000 条无重叠提示，用于校准评估）；③攻击集（AdvBench、HarmBench 各 100 条恶意提示，用于评估防御在 jailbreak 攻击上的效果）。

**📈 对比分析**

与 13 种基线（confidence‑based、guardrail、streaming monitor、feature‑probe 等）在 3 个 LLM（Llama3‑8B‑it、Qwen3‑8B、Gemma2‑9B‑it）和 3 个安全数据集上进行对比。ProbGuard 在 Brier 及 ECE 上均领先所有基线，平均降低 Brier 约 79.6%、ECE 约 71.9%；在六种 jailbreak 攻击中，ProbGuard 使攻击成功率降至 ≤1%，显著优于最优基线（约 2‑3%）。

**⚠️ 局限性**

局限性：①需要 Monte‑Carlo 采样，虽默认 N=16，但在更大规模或更长生成时仍会产生额外计算开销；②依赖 CalibEval 进行安全标签，若判别器性能下降可能影响校准质量；③目前评估聚焦于已知的六种 jailbreak 攻击，对未知或更复杂攻击的鲁棒性尚待验证；④跨模型迁移性强，但在极端新型 LLM（如极大模型或多模态模型）时仍需进一步验证。

---

## 119. Fine-Tuning Large Language Models for Codebook-Guided Coding of Students' Mathematics Metaphor Responses

**arXiv ID:** 2608.10276 | [PDF](https://arxiv.org/pdf/2608.10276v1)

**作者:** Liang Zhang `[一作]` (University of Michigan), Jinfa Cai `[通讯]` (Texas A&M University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估LoRA基监督微调后开源小型LLM在学生数学隐喻编码任务（情感强度与主题编码）上的表现，并与专有GPT-4o-mini、GPT-5-mini进行对比

**💡 创新点**

证明开源小模型通过LoRA微调即可与专有模型竞争，且可在本地部署、隐私友好，提供可扩展的代码书引导编码方案

**🔧 技术方法**

LoRA监督微调、代码书引导零shot提示、JSON输出格式、Krippendorff α、宏/微F1、Quadratic Weighted Kappa等评价指标

**📊 数据集**

2265条6–8年级学生食物/动物类数学隐喻数据，已人工编码情感强度（1–5）和多码主题（13个细粒度主题）

**📈 对比分析**

采用相同的测试集和提示结构对四模型进行三次重复评估；微调后DeepSeek‑R1 1.5B与Mistral 7B在情感和主题编码的Accuracy、Macro‑F1、QWK、Subset、Micro‑F1等指标上等同或优于GPT‑4o‑mini与GPT‑5‑mini，尤其Mistral‑7B在大多数指标上领先

**⚠️ 局限性**

仅在温度=0下评估；数据仅限6–8年级食物/动物隐喻，难以推广；稀有主题类别稳定性不足；未深入解释不同代码水平性能差异

---

## 120. Chronological Certificates for Shellsort: Ray Defects and Signed-Positive Transference

**arXiv ID:** 2608.10696 | [PDF](https://arxiv.org/pdf/2608.10696v1)

**作者:** Ziqi Zhao `[一作]` (Southeast University), Qingjian Ni `[通讯]` (Southeast University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种统一的“傅里叶与时间顺序”证书框架，既能给Shellsort的最坏情况交换/比较次数下界，也能给上界，阐明了两者之间的结构关系并给出了新的下界和上界结果；

**💡 创新点**

首次将傅里叶相位缺陷（下界）与有序正整数表示（上界）通过非对称传递联系起来，形成了一套完整的前缀和射线几何工具，进而获得了新的下界（Ω(n(log n/ loglog n)^2)）与上界（O(nlog^2 n)）的桥梁；

**🔧 技术方法**

使用了数值半群与Apéry集合理论、傅里叶分析、代数几何中的“射线商”和“正整数表示长度”指标、以及组合数论的Sylvester/ Frobenius 定理等多种技术；

**📊 数据集**

论文为理论研究，未使用实验数据集，所有结果均为纯数学证明；

**📈 对比分析**

通过对已知的Pratt、Incerpi–Sedgewick等经典上下界进行重新推导和对比，验证了新框架能恢复并改进这些结果；性能方面证明了在O(nlog^2 n)比较量下的必要结构，并给出了对应的下界；

**⚠️ 局限性**

限制在于两种证书之间的不对称性仍未完全消除，尚未解决Pratt尺度下的完整必要性问题（即是否每个O(nlog^2 n)的调度都必需满足O(log^2 n)的本地Apéry预算），以及对更高阶多生成元网格的更细致下界仍有空白。

---

## 121. Logit-Boundary Geometric Belief Interfaces and Sparse Sheaf-Enclave Protocols: A Self-Contained Substrate for Secure Network Electronic Health Record (EHR) Interoperability

**arXiv ID:** 2608.10300 | [PDF](https://arxiv.org/pdf/2608.10300v1)

**作者:** Alvin Spivey `[一作]` (Light Imaging Technologies Inc), Thomas Huang `[通讯]` (Light Imaging Technologies Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于logit边界的临床互操作性架构，先让不可信的发现模型输出logit receipt，再通过确定性判断引擎完成身份、术语、证据、图层一致性等校验，最终决定是否生成FHIR事务；同时引入了Dirichlet证据校准、细胞层面映射圆锥诊断与分布式加密脚本（DCSE）协议，以实现对模型输出的安全隔离与审计；

**💡 创新点**

核心创新点在于：①将模型输出抽象为“logit边界”而非直接推断结果，形成统一的可检查接口；②构建Geometric Belief Interface（GBI）与Finite Boundary Semantics的数学框架；③利用Dirichlet分布对本地类别证据进行校准；④使用细胞层sheaf与映射圆锥诊断检测跨局部冲突；⑤设计分布式加密Sheaf‑Enclave协议（DCSE）实现无等价化、可审计的决策链；

**🔧 技术方法**

主要技术包括：logit拓扑与线性读出理论、Dirichlet证据更新、细胞层面Sheaf与映射圆锥Hodge算子、隐式安全审计与加密证书、分布式BFT/TEE支持的DCSE协议；

**📊 数据集**

使用自定义的GBI BoundaryBench v0.1（256个任务、3种证据模式），在Qwen3‑4B‑Instruct‑2507模型上进行评估；

**📈 对比分析**

在该基准上共完成768次推理，但所有结果均被判定为不可接受（369次解析拒绝、399次模式校验拒绝），覆盖率为0%，表明模型输出完全被边界控制；目前未与其他模型或系统进行性能对比，主要展示了边界机制的有效性；

**⚠️ 局限性**

局限性包括：仅在单一4B模型和单一冻结接口上测试，无法证明跨模型或多种模型的一致性；不涉及真实临床数据与终端安全验证；仅能防止模型输出成为最终数据写入，无法消除内部幻觉；缺乏针对不同任务或规模的性能基准；

---

## 122. The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom

**arXiv ID:** 2608.10145 | [PDF](https://arxiv.org/pdf/2608.10145v1)

**作者:** Joyjeet Singh `[一作]` `[通讯]` (Independent researcher), Joyjeet Singh (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

重现并纠正了 LeWorldModel 在 TwoRoom 环境中的训练与评估过程，验证其表示学习与规划性能。

**💡 创新点**

发现并纠正了未公开的四条管线约定、评估协议不一致以及 BatchNorm 评估模式误差，并揭示了一步预测误差并不能预测长期规划效果。

**🔧 技术方法**

基于 ViT‑Tiny 编码器、单一预测损失与抗崩塌正则化的世界模型，采用 CEM 规划和线性/非线性探针评估。

**📊 数据集**

TwoRoom 诊断环境（10,000 条轨迹，固定帧速 5、最大长度 101）以及 32×32 像素调试版本。

**📈 对比分析**

对比了作者原始权重、纠正后权重及不同评估协议，纠正后权重在 25 步 offset 下 94% 的成功率，原始权重 84%，且一站预测误差与长程规划成功率无关。

**⚠️ 局限性**

仅使用单一随机种子、10 轮训练、单一环境，未验证不同规模或其他任务的泛化；BatchNorm 归一化校准是必需的，且结果对评估协议高度敏感。

---

## 123. Towards Color-Faithful Low-Light Image Enhancement via Adaptive Color Debiasing and Saturation Rectification

**arXiv ID:** 2608.10512 | [PDF](https://arxiv.org/pdf/2608.10512v1)

**作者:** Zhichen Yang `[一作]` (Fuzhou University), Ri Cheng `[通讯]` (Fuzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CAGE 框架，使用 AdaLAB 圆柱色空间和 AdaCCT 变换实现低光图像颜色校正。

**💡 创新点**

创新点在于先通过 AdaCCT 的前向变换自适应地抑制嵌入的颜色偏差，再通过逆向变换和 Out‑of‑gamut 光度补偿纠正饱和度异常，构建了统一的颜色校正流程。

**🔧 技术方法**

采用自适应圆柱变换、亮度敏感点、色度缩放、色调位移、光度补偿等技术，并与 Retinex、HSV‑Inspired、RGB 等多种低光增强骨干网络兼容。

**📊 数据集**

在 LOLv1、LOLv2、SDSD、SID 等标准低光配对数据集上进行评估。

**📈 对比分析**

与 Retinexformer、DarkIR、HVI‑CIDNet 等多种基线和先进方法进行 PSNR/SSIM/LPIPS 对比，CAGE 在所有数据集均实现 0.5–1.5 dB PSNR 提升、>0.01 SSIM 增长、LPIPS 降低，显示显著的颜色恢复与视觉质量提升。

**⚠️ 局限性**

局限在于仍依赖骨干网络性能，极端光照下可能出现局部颜色失真；模型增量虽小但仍需进一步压缩以适应资源受限设备。

---

## 124. Multi-UAV Tracking Evaluation Using 5G Uplink Signals on an O-RAN ISAC Simulation Testbed

**arXiv ID:** 2608.10784 | [PDF](https://arxiv.org/pdf/2608.10784v1)

**作者:** Arun K. Gurung `[一作]`, Satha K. Sathananthan `[通讯]` (NexVis)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在 O‑RAN ISAC 测试平台上，使用 5G NR 上行声音参考信号 (UL‑SRS) 进行多 UAV 检测、关联与跟踪，并将结果通过 SAPIENT 接口导出给 C2 消费者。

**💡 创新点**

首次实现了端到端的多目标 UAV 传感与跟踪评估，并揭示了竞争（contention）和身份（identity）是关键瓶颈，而非灵敏度；同时提出了利用垂直阵列提升关联性能的策略。

**🔧 技术方法**

结合 OpenAirInterface NR 栈、FlexRIC Near‑RT RIC、Sionna RT Ray‑Tracing 渠道、基于 OS‑CFAR 的检测、EKF 追踪器以及自定义 SM‑SENS、SAPIENT 接口。

**📊 数据集**

采用仿真场景：三架不同 RCS、飞行高度和速度的 UAV，使用 Sionna RT 生成的射线追踪通道，并通过噪声种子在单一几何配置中多次运行。

**📈 对比分析**

通过一系列指标（如可用性、竞争率、覆盖率、IDF1、GOSPA、RMSE 等）与 3GPP 研究报告中的 UAV 用例 KPI 对比，发现定位精度满足部分要求但身份碎片化严重，整体跟踪性能受竞争约束。

**⚠️ 局限性**

仅在仿真环境下评估（无实际 RF、时钟或干扰），场景仅包含单一几何且 UAV 数量有限；检测容量受固定缓冲限制；在同一距离‑多普勒格子内无法区分多个目标；未验证在更复杂、实时或干扰环境中的鲁棒性。

---

## 125. RLMOpt: Adaptive Prompt Optimization via Recursive Language Models

**arXiv ID:** 2608.10471 | [PDF](https://arxiv.org/pdf/2608.10471v1)

**作者:** Subhash Bangalore Satheesha `[一作]` (Autonomize AI), Bharath Dandala `[通讯]` (Autonomize AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种自适应提示优化器：使用递归语言模型（RLM）作为搜索策略，并通过确定性 harness 控制评估、选择与回归阈值，实现多任务的提示改进。

**💡 创新点**

创新点在于将搜索策略本身交给语言模型驱动，拆分评估与选择为确定性模块以避免过拟合，同时支持多组件提示（工具调用、演示等）并采用 Pareto 前沿与无回归阈值保证安全改进。

**🔧 技术方法**

采用的技术包括递归语言模型（RLM）、工具接口与 REPL 代码生成、确定性 harness、分字段评分与 Pareto 选取、回归阈值、无回归 floor、随机抽样、Llama‑3 70B（优化器）和 Llama‑3 8B/70B（任务模型）。

**📊 数据集**

使用的基准数据集为：Chia（临床信息提取）、HotpotQA（多跳问答）、IFBench‑2025（可验证指令遵循）和 BFCL 多轮工具调用（Berkeley Function‑Calling Leaderboard v3）。

**📈 对比分析**

与基线 Prompt Optimizer（light）在单一种子以及多种子匹配比较中对比。结果显示在四个基准上均取得最高held‑out分数，四任务平均提升 0.021；在 11 次种子对比中，改进占 9/11，且从不出现回归；计算效率更优，搜索 roll‑outs 更少，提示长度更短。

**⚠️ 局限性**

限制包括：未对各个组件（自适应搜索、Pareto 选取、无回归阈值等）单独消融，难以量化其独立贡献；停止策略方差大，未实现最佳收敛；在验证集很小或任务已近模型提示极限时易出现过拟合或无效改进；仅在具有足够 headroom 的任务上有效。

---

## 126. DoseBridge: Denoising Diffusion Bridge Model for Dose Prediction in Lung Intensity-Modulated Proton Therapy

**arXiv ID:** 2608.10173 | [PDF](https://arxiv.org/pdf/2608.10173v1)

**作者:** Zerun Zhang `[一作]`, Xuanfeng Ding `[通讯]` (Corewell Health William Beaumont University Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并实现了一种名为DoseBridge的去噪扩散桥模型，用于基于患者CT及显式束线几何信息来预测肺部强子束调强放疗（IMPT）的剂量分布。

**💡 创新点**

创新点包括：①首次将去噪扩散桥（DDBM）应用于放疗剂量预测；②将患者CT作为结构化桥端点，直接从解剖结构出发进行逆向生成；③使用空间对齐的二值束线掩模将计划特定的束线几何显式编码；④在U‑Net解码器中加入轻量级的门控融合和卷积融合模块，仅增加约1.95%的参数。

**🔧 技术方法**

核心技术包括：去噪扩散概率模型（DDBM）与ADM U‑Net；结构化桥端点与时序条件；束线掩模与结构与距离图（SDM）的联合编码；门控融合（Squeeze‑Excitation）和卷积融合；以及使用隐式采样器进行高效推理。

**📊 数据集**

使用单机构回顾性数据集：52例晚期肺癌患者（IMPT 60 Gy/30次），其中42例用于训练，10例用于测试。每例包括规划CT、结构标注（CTV、OAR、身体）和束线等位置信息。

**📈 对比分析**

与两种基准模型（DoseDiff 与 RTPdose）进行比较，采用 MAE、PSNR、SSIM 等图像相似度指标以及临床剂量体积指标（CTV D95、OAR Dmean/Dmax、肺 V20）和 NTCP。DoseBridge 的 MAE 为 4.17 Gy，PSNR 23.06 dB，SSIM 0.798，均优于两基准；临床指标误差均低于 3.2%（最大为 7.44%），NTCP 误差 < 2%。

**⚠️ 局限性**

局限性：①仅在单机构、单一病灶（肺）且剂量方案相同的 52 病例上训练和验证，缺乏跨机构、跨部位的通用性验证；②对极少量高梯度区的组织（如食道 Dmax、脊髓 Dmax）预测误差较大；③由于数据量有限，模型仍可能出现轻度过拟合；④预测剂量应作为规划先导而非最终剂量计算结果。

---

## 127. Multi Interests for Joint Search-Recommendation Modeling

**arXiv ID:** 2608.10535 | [PDF](https://arxiv.org/pdf/2608.10535v1)

**作者:** Xiangchen Pan `[一作]` (Huazhong University of Science and Technology), Zhicong Cheng `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并实现了一种联合搜索与推荐混合序列的多兴趣建模框架MIJSR

**💡 创新点**

在结构层提取搜索、推荐与跨域兴趣，并在语义层通过查询聚类与正交约束实现更细粒度、多角度兴趣分解

**🔧 技术方法**

采用对比学习对齐查询与商品、语义聚类与多层感知机映射、交叉注意力与自注意力、进阶层抽取(Progressive Layer Extraction)以及PLE多任务网络

**📊 数据集**

使用公开的Amazon Kindle Store和KuaiSAR两个大规模数据集进行实验

**📈 对比分析**

与15种基线（包括推荐、搜索及联合S&R模型）对比，MIJSR在HR@5/NDCG@5等指标均优于SOTA UniSAR，搜索任务提升显著，推荐任务也保持或提升

**⚠️ 局限性**

对正交、对齐及聚类等超参数高度敏感，且依赖查询语义聚类的质量，未在更广泛的异构域上验证

---

## 128. RadFusion: Towards Threshold-Controllable Radiology Report Generation

**arXiv ID:** 2608.10505 | [PDF](https://arxiv.org/pdf/2608.10505v1)

**作者:** Ying Jin `[一作]` (Microsoft), Eric Horvitz `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 RadFusion 框架，实现通过阈值控制医学影像报告生成的敏感度-特异度平衡

**💡 创新点**

首次将分类器的 ROC 控制机制与生成式报告结合，使报告的诊断内容可随阈值动态调节

**🔧 技术方法**

使用三模组件：校准的多标签分类器、基于 Auto‑VQA 的报告生成器以及 LLM 重写器（如 GPT‑5）

**📊 数据集**

在 MIMIC‑CXR 数据集上进行实验，评估 13 类胸片疾病的诊断性能

**📈 对比分析**

通过在 0.0–1.0 逐阈值闭环评估，报告 ROC 与分类器一致，且在匹配特异度时敏感度提升 6.9%，在匹配敏感度时特异度提升 20.7%

**⚠️ 局限性**

依赖于三者质量；分类错误会直接影响报告；重写指令需手工调优；仅对预设疾病类可阈值控制，未覆盖所有临床信息

---

## 129. Edge Phoneme Recognition for Children's Speech through Age-Aware Training

**arXiv ID:** 2608.10206 | [PDF](https://arxiv.org/pdf/2608.10206v1)

**作者:** Matthew Arboleda `[一作]` (Occidental College), Joel Walsh `[通讯]` (Occidental College)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并训练了一种轻量级的儿童语音音素识别模型，通过在WavLM Base+编码器上添加年龄分类辅助任务实现。

**💡 创新点**

利用年龄辅助学习将模型正则化为年龄不变的音素表示，从而在仅94M参数的模型上达到与大型模型相当的识别准确率。

**🔧 技术方法**

采用多任务CTC与年龄分类头、WavLM Base+预训练编码器以及端到端的语音转音素推理技术。

**📊 数据集**

使用包含11.75万条儿童语音的训练集（DrivenData + TalkBank），并按年龄桶划分进行训练与验证。

**📈 对比分析**

与317M参数的WavLM Large、RNN‑T等基线模型在DrivenData评估集上对比，年龄感知模型在混合CER 0.306 与目标分布 0.306 逼近大型模型，显著低于传统集成方法且仅需94M参数。

**⚠️ 局限性**

模型在更大规模、精度更高的WavLM Large 上难以部署，且年龄分类准确率仅72%且偏向多数类，需进一步解释年龄辅助机制并提升对不同年龄段的泛化能力。

---

## 130. Navigating the Proximity-Safety Balance: Constraint Decomposition for Human Following in Pedestrian Crowds

**arXiv ID:** 2608.10056 | [PDF](https://arxiv.org/pdf/2608.10056v1)

**作者:** Shiting Gong `[一作]` (University of Pennsylvania), Jiachen Li `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在拥挤人群环境下，提出一种多约束强化学习框架，将人类跟随任务拆解为稀疏奖励与独立成本约束，并在策略网络中融入运动预测不确定性，从而实现安全且紧密的跟随。

**💡 创新点**

创新点包括：①将跟随距离、安全与障碍约束分别映射为可直观设置的阈值，使得行为权衡可直接可调；②将预测不确定性纳入成本与观测，提升在不确定人类行为下的安全性；③使用PPO‑Lagrangian联合优化四个独立价值函数，构建统一的决策网络。

**🔧 技术方法**

核心技术为多约束强化学习（PPO‑Lagrangian）、Transformer+CNN注意力感知网络、预测不确定性模块ACI、动态阈值控制及离散化安全成本设计。

**📊 数据集**

使用CrowdNav模拟器扩展的静态障碍场景，生成多种密集人群与行为模型（ORCA、社交力SF、冲刺人群、群体运动），并在真实机器人上搭载RPLIDAR+DR‑SPAAM进行感知与跟踪。

**📈 对比分析**

与SG‑HA*、SG‑ORCA、SG‑MPC、OGM‑HEIGHT、RL、RL+ACI等基线对比，在分布内外实验中本方法成功率提升10–25%，碰撞率下降5–20%；在真实机器人实验中，10条测试路径中成功率达70%，验证了其实际可行性。

**⚠️ 局限性**

局限性包括：受上游检测与跟踪误差影响；对极端密集或高速人群的适应仍有限；阈值与惩罚参数需人工调优，且依赖先验运动预测器的准确性。

---

## 131. ASR-Roundtrip Evaluation Can Mask Context- and Convention-Dependent Reading Errors in Chinese News TTS

**arXiv ID:** 2608.10606 | [PDF](https://arxiv.org/pdf/2608.10606v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 132. $β$-VAEs as Effective Theories: Tolerance-Dependent Dimension

**arXiv ID:** 2608.10599 | [PDF](https://arxiv.org/pdf/2608.10599v1)

**作者:** Johannes Hirn `[一作]` `[通讯]` (Universitat de València), Johannes Hirn (Universitat de València)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了 β-VAE 在非线性条件下的谱截断与有效维度，利用 β 扫描分析了 WorldClim 数据集的重建效用及层深对有效维度的影响。

**💡 创新点**

创新点在于把 β-VAE 的 KL 正则化视为可调的“截断阈值”，通过直接测量重建效用谱而非拟合阈值来定义有效维度，并揭示深度模型在头部集中效用、尾部精度折中的行为。

**🔧 技术方法**

技术手段包括全连接 β-VAE、SNR 相关秩序参数、重建误差归一化、折扣率率曲线对齐、对数-对数功率律拟合以及不同层深的比较。

**📊 数据集**

使用的公开数据集是 WorldClim 19 生态气候变量（10' 分辨率），经过标准化和空间块采样，训练集约 50 万样本。

**📈 对比分析**

与 PCA 的重建误差和维度进行对比；在 5%~0.001% 重建误差阈值下，β-VAE 所需的有效维度显著低于 PCA；4 层深度模型在前几维的压缩率最高，但尾部误差略高。

**⚠️ 局限性**

局限性包括仅在无监督重建任务中评估；阈值与重建效用在高阶维度存在偏差；未验证在有监督任务或其他数据集上的普适性；扫描和训练成本较高。

---

## 133. Decomposition-Induced Context-Memory Conflict: When Fact-Checking Pipelines Contradict Their Own Source Text

**arXiv ID:** 2608.10627 | [PDF](https://arxiv.org/pdf/2608.10627v1)

**作者:** Yu-Feng Yen `[一作]` `[通讯]` (Independent Researcher), Yu-Feng Yen (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在“拆分-验证”事实检查流水线中，语言模型在拆分段落成原子主张时可能用自身参数知识取代原文信息，从而产生与原文相矛盾的主张，称之为拆分诱导的上下文‑内存冲突（DI‑CC）。

**💡 创新点**

创新点在于：①首次正式定义并机制化DI‑CC，证明它与经典的上下文‑内存冲突属于同一根源；②利用仅在经典冲突数据上训练的线性探针在拆分步骤中零样本即可检测DI‑CC；③发现传统的上下文感知解码（CAD）可以显著抑制DI‑CC，但伴随显著的完整性-faithfulness权衡。

**🔧 技术方法**

技术上主要使用：语言模型（Qwen2.5‑7B/3B/14B），线性探针训练与层选择（准确率与方差比），闭包式知识提取与NLI判定（DeBERTa‑v3‑large），以及CAD对比解码。

**📊 数据集**

数据集包括自生成的40/191/74个人传记（带精确/模糊/扰动版本）和公开的FActScore/ConflictBank/ NQ‑Swap，用以构造DI‑CC样本并进行真实对照测试。

**📈 对比分析**

与无参参考检测方法SelfCheckGPT相比，探针在DI‑CC检出率（AUC≈0.86–0.88）显著优于SelfCheckGPT（AUC≈0.51），且CAD将DI‑CC率从4.16%降低到2.57%（p=0.00004），但CAD导致19.4%拆分失败，且84%涉及身份造假。

**⚠️ 局限性**

局限性包括：DI‑CC自然出现率极低（0.2–0.4%），在自然生成文本中难以触发；探针在实际类不平衡场景下召回率低且种子敏感；CAD的完整性-faithfulness权衡尚未解决；机制在不同模型族（Mistral、Falcon）下未显著复制，表明可能具有模型族特异性。

---

## 134. Online Discrepancy Minimization for Sub-Gaussian Inputs via Regularization and Restriction

**arXiv ID:** 2608.10040 | [PDF](https://arxiv.org/pdf/2608.10040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 135. Mapping the Climate-Health Evidence Base (2007-2023): A Bibliometric, Statistical, and NLP Multi-Label Text Analysis of 22,695 Records

**arXiv ID:** 2608.09985 | [PDF](https://arxiv.org/pdf/2608.09985v1)

**作者:** Dhruv Dixit `[一作]` (Stevens Institute of Technology), Janine Molino `[通讯]` (Brown University)

**通讯引用:** 679 | [OpenAlex ID](https://openalex.org/A5109530709)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文基于22,695条2007–2023年的气候–健康文献，构建多标签词汇表，量化文献量、主题集中度、暴露–健康关联、方法时标变化以及编码完整性随时间与地区的演变。

**💡 创新点**

创新点在于将文献计量、网络协同统计、层级逻辑回归和主题模型等多方法结合，针对高维稀疏标签进行联合缺失诊断与偏倚调整，首次揭示暴露–健康配对的真正“富集”模式与时标转向长期预测的系统趋势。

**🔧 技术方法**

技术方法包括：负二项时间序列回归与Pelt/Bayesian分段；独立性检验与FDR控制的暴露–健康协同表；偏移泊松网络模型；层级逻辑/多元逻辑回归（含地理与期刊随机效应）；有序logit模型；LDA主题建模；缺失机制的Logistic回归与多重插补；以及贝叶斯多元Probit对多结果依赖的估计。

**📊 数据集**

使用的数据集是公开可获取的Climate–Health Bibliographic Corpus（22,695条记录），涵盖标题、期刊、年份、控制词汇标签（暴露、健康影响、地理、方法、专题）以及可用的摘要文本。

**📈 对比分析**

与单一统计或文本主题分析相比，综合方法显著提升了对暴露–健康配对富集的检验灵敏度（FDR<0.05），层级模型降低了因期刊/地区差异导致的偏倚，时间序列模型捕捉到四次显著分段，结果表明文献增长≈10%/年且时标平均延长；整体性能优于传统计数或单变量回归。

**⚠️ 局限性**

局限性包括：标签缺失与编码不完整导致的潜在非随机缺失；多标签稀疏导致的统计稀缺与模型收敛问题；主题模型缺乏可解释的词表，导致主题命名不确定；以及仅基于公开标签，未能覆盖未被注释的研究内容。

---

## 136. Coordinating the Unknown Lipschitz Constant in Multiplayer Bandits

**arXiv ID:** 2608.10526 | [PDF](https://arxiv.org/pdf/2608.10526v1)

**作者:** Ricardo Parada `[一作]` (University of California Riverside), William Chang `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在未知Lipschitz常数下的多玩家合作多臂老虎机问题，设计并分析了三种信息结构下的算法。

**💡 创新点**

提出了一种元算法，能够在未知Lipschitz常数的情况下，通过不同的信息结构实现玩家之间的协调，并证明了相应的后悔界限。

**🔧 技术方法**

使用了Lipschitz bandits的框架，结合了均匀探索、离散化和现有的合作多玩家多臂老虎机子程序。

**📊 数据集**

使用了模拟数据集，设置了两个玩家和一维动作空间，进行了10^5轮的实验。

**📈 对比分析**

通过与非自适应离散化方法的比较，展示了自适应离散化在不同信息结构下的性能优势，尤其是在Lipschitz常数较大时表现更佳。

**⚠️ 局限性**

算法在信息结构不理想的情况下（如奖励和动作信息不对称）可能导致较高的后悔值，且在实际应用中可能面临更复杂的环境挑战。

---

## 137. VisEditBench: Can Vision-Language Models Edit Visualization Code from Multimodal Feedback?

**arXiv ID:** 2608.10408 | [PDF](https://arxiv.org/pdf/2608.10408v1)

**作者:** Mizanur Rahman `[一作]` (York University), Enamul Hoque Prince `[通讯]` (York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 VisEditBench，首个评估视觉语言模型在多模态反馈下进行可视化代码编辑的基准，并提供 VisEditAgent 迭代式渲染反馈框架。

**💡 创新点**

创新点在于把可视化代码编辑视为多模态、循环迭代任务，并通过人类标注的真实错误案例构建基准，首次量化模型在反馈修复与风格重塑上的表现。

**🔧 技术方法**

采用了大型视觉语言模型（如 GPT‑4o、Claude‑4.6‑Sonnet、Qwen‑3‑VL‑32B 等）进行零样本推理，并设计 VisEditAgent 包含候选生成、执行、视觉验证和迭代细化。

**📊 数据集**

使用了从 Stack Overflow、Matplotlib/Vega‑Lite issue、Text2Vis 等真实数据生成的 1,395 个任务，覆盖 Matplotlib 与 Vega‑Lite 两大可视化库。

**📈 对比分析**

通过统一评估指标（可执行性、任务准确性、可读性、视觉质量、相似度）对 20 个模型进行零样本测试，Claude‑4.6‑Sonnet 以 74.46% 的通过率领跑，开放源代码模型最高约 51%，而 VisEditAgent 在 GPT‑4o 上将通过率提升至 67.99%。

**⚠️ 局限性**

局限在于仅覆盖 Matplotlib 与 Vega‑Lite，未包含 Plotly、D3.js、ggplot2 等主流库，且基准样本虽真实但未能覆盖所有专业领域的特殊需求。

---

## 138. Continuous Interaction Diffusion: A Diffusion-Native Runtime for Asynchronous Tool-Augmented Reasoning

**arXiv ID:** 2608.10438 | [PDF](https://arxiv.org/pdf/2608.10438v1)

**作者:** Yuhang Cao `[一作]` `[通讯]` (Nanjing University), Yuhang Cao (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Continuous Interaction Diffusion（CID）架构，将工具调用整合进扩散语言模型的去噪迭代中，实现持续非阻塞的外部信息交互；

**💡 创新点**

核心创新在于三通道（事实、思考、显示）设计、Typed Cognitive Tensor 表示可连续更新的隐层思考、以及持久感知绑定（perceptual bindings）使工具请求可提前触发、缓存重投影、动态刷新；

**🔧 技术方法**

技术手段包括：掩码扩散语言模型、连续隐层思考、Typed Cognitive Tensor、异步运行时与局部扩散时钟、工具意图适配器、感知编码与投影、源链接、持久绑定管理；

**📊 数据集**

论文未给出实测数据集，提出评估协议，建议在标准 QA、检索、复制、动态状态跟踪等任务上使用常见的检索/问答数据集进行实验；

**📈 对比分析**

通过与异步自回归模型、P-ReAct、传统 dLLM 工具调用等基线对比，评估信息需求提前性、等待时间利用、感知重投影效果、Typed Cognitive Tensor 的必要性和扩散本身的价值；预期在匹配质量的情况下，CID 能实现更低端到端延迟，且在工具调用效率上优于传统方法；

**⚠️ 局限性**

局限性包括：架构复杂度高、缺乏高质量的思考/绑定标注、思考稳定性与事实通道策略问题、计算与内存开销、隐私与可观测性挑战、目前仅支持只读工具、未处理副作用工具与多模态扩展。

---

## 139. Transformer Geometry Observatory TGO-IV: Developmental Topology Observatory

**arXiv ID:** 2608.09997 | [PDF](https://arxiv.org/pdf/2608.09997v1)

**作者:** Kaustubh Kapil `[一作]` (Sardar Vallabhai National Institute of Technology), Kishor P. Upla `[通讯]` (Sardar Vallabhai National Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发并应用了Transformer Geometry Observatory IV（TGO‑IV），通过构造Vietoris–Rips复形和计算持久同调，系统性地追踪ViT-Small在训练过程中的层级表示点云的拓扑演化。

**💡 创新点**

首次将持久同调方法与Transformer表示相结合，构建了一套完整的拓扑观测器（Persistence Diagram、Barcode、Betti Curve、Landscape、Bottleneck/Wasserstein距离），揭示了表示在训练过程中逐渐收敛为更紧凑且非平凡的拓扑结构（Progressive Topological Stabilization Hypothesis）。

**🔧 技术方法**

使用的技术包括：Vietoris–Rips复形构造、持久同调（Persistent Homology）、Persistence Diagram/Barcode、Betti Curve、Persistence Landscape、Bottleneck距离、Wasserstein距离；此外还用PyTorch训练ViT-Small，采用AdamW、cosine scheduler、AMP等常见训练技巧。

**📊 数据集**

实验数据集为ImageNet-100（1000张验证图像子集），训练采用ViT‑Small/16模型，12层Encoder，384维嵌入。

**📈 对比分析**

通过计算相邻层的Bottleneck和Wasserstein距离来量化拓扑差异。结果显示：训练早期距离较大，后期显著下降，表明相邻层的持久拓扑签名趋于相似；在特定层（如第3、10、12层）仍出现局部峰值，提示这些层仍在进行显著的拓扑重组。

**⚠️ 局限性**

主要局限在于：每层仅有197个点在384维空间中构成的点云极其稀疏，Vietoris–Rips复形只能近似表示，无法精确捕捉潜在表示流形的真实拓扑；结论仅适用于当前模型和数据规模，尚未验证在更大模型或其他任务上的通用性。

---

## 140. Grid-Preserving Knowledge Distillation: Transferring Convolutional Inductive Bias to Vision Transformers under Data Scarcity

**arXiv ID:** 2608.10723 | [PDF](https://arxiv.org/pdf/2608.10723v1)

**作者:** Junyong Choi `[一作]` (Hyundai Motor Company), Jaehoon Cho `[通讯]` (Korea Aerospace University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出了一种在训练阶段保留CNN教师空间网格的知识蒸馏框架，使用Inductive Bias Attention Module让ViT学生在数据稀缺场景下获得卷积网络的局部性和平移等先验。

**💡 创新点**

创新点在于将空间网格完整保留到蒸馏过程的每个阶段，采用可学习的跨架构对齐、可变形空间注意力以及卷积式交叉注意力，确保ViT学生接收的先验信息与CNN教师的空间对应关系一致。

**🔧 技术方法**

主要技术包括可学习的跨层权重聚合、通道+可变形空间注意力、1×1卷积投影的交叉注意力、以及基于位置的蒸馏损失。

**📊 数据集**

实验使用的教师为ResNet-56/50，学生为DeiT、T2T、PiT、PvTv2、CvT、ConViT等视觉Transformer，在CIFAR‑10/100、Flowers‑102、Chaoyang、CUB‑200和Tiny‑ImageNet等六个数据稀缺基准上评估。

**📈 对比分析**

与现有的局部性引导方法(LG/ALG)以及通用KD方法比较，所提框架在七种Transformer骨干上均取得最高或相近最高精度，尤其在极低数据量下差距可达2–3%（如CIFAR‑100 20%时提升约2.8%）。

**⚠️ 局限性**

局限性包括：只在分类任务上验证，尚未探讨对像素级任务（分割、深度估计）的推广；训练阶段需要额外的教师与模块，虽然推理无额外开销，但总体训练成本上升；在更大规模数据集或多任务场景下的有效性尚待进一步验证。

---

## 141. The Signal Rail: A Deterministic Motion Grammar for Communicating Conversational Agent State in Terminal Interfaces

**arXiv ID:** 2608.10689 | [PDF](https://arxiv.org/pdf/2608.10689v1)

**作者:** Matteo Grella `[一作]` `[通讯]` (Crisis24), Matteo Grella (Crisis24)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种名为Signal Rail的单行终端状态栏，用空间、方向和运动规则直观地展示对话代理的各个状态；提供了完整的规范、参考实现以及跨语言一致性验证。

**💡 创新点**

创新点在于：①为每个状态定义唯一的运动规则而非单纯依赖颜色；②采用空间分区（输入/处理/输出）与方向语义，形成可辨识的模式；③实现完全确定性的纯函数帧渲染，支持golden‑frame测试；④强调“诚实”原则，只展示系统实际拥有的进度与状态。

**🔧 技术方法**

使用技术包括：终端字符渲染（ANSI/Truecolor、Unicode块绘制）、Zig实现的参考框架、哈希驱动的确定性随机、音频量化（STT/TTS量化级别）、工具执行计数、跨语言（Zig/JS/Python）字节级一致性测试、golden‑frame测试与结构测试。

**📊 数据集**

论文未使用传统机器学习或图像数据集；评估基于：真实语音量化输入、真实工具计数进度、硬编码的golden‑frame测试用例，确保每个状态在所有配置下的渲染一致。

**📈 对比分析**

比较方法主要通过golden‑frame和结构化单元测试，验证实现与规范的一致性；实现与参考实现、JS/Python版本在同一帧矩阵上完全字节级一致。性能方面，每83 ms tick绘制一次，满足实时要求；未进行用户性能或负载测试。

**⚠️ 局限性**

局限性包括：缺乏用户研究验证状态可识别性；未实现过渡桥接动画；单行单代理设计，未覆盖多代理或后台长任务；无无障碍（屏幕阅读器）支持；终端环境无法自动读取系统的“减少动画”偏好；实现未覆盖所有规范中的细节（如警告状态）。

---

## 142. LLM Ensemble Fault Classification for Automotive HiL Validation

**arXiv ID:** 2608.10710 | [PDF](https://arxiv.org/pdf/2608.10710v1)

**作者:** Hamza Ouarrad `[一作]` (Institute for Software and Systems Engineering), Andreas Rausch `[通讯]` (Institute for Software and Systems Engineering)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于多大语言模型（LLM）集成的汽车硬件在环（HiL）故障定位分类框架。

**💡 创新点**

创新点在于将多维传感器窗口压缩为结构化证据，使用多LLM协同推理和置信度加权投票，提升诊断鲁棒性、校准性和可解释性。

**🔧 技术方法**

技术包括时间窗口化、健康对比的统计证据提取、零/少样本提示工程、Mistral、Qwen2.5、Phi-4等LLM推理，以及置信度加权投票、Borda/Reciprocal Rank融合。

**📊 数据集**

数据集来自两套HiL系统（汽油机和电动车），包含10种单一故障、3个驾驶场景、约15,000条窗口，每条窗口0.01s采样。

**📈 对比分析**

通过与单模型比较和多模型集成，对比Top‑1、Top‑2准确率、宏F1、MCC、Brier、ECE等指标；Top‑3集成实现0.917 Top‑1准确率、0.913宏F1、0.902 MCC，并在校准方面优于单模型。

**⚠️ 局限性**

局限包括对并发故障的适用性有限、依赖高质量健康参考、模型推理时间较长、对提示敏感以及缺乏对更大规模数据和多驾驶配置的验证。

---

## 143. VeriFin: A Neurosymbolic Framework for Verifying LLM-Generated Financial Claims

**arXiv ID:** 2608.10213 | [PDF](https://arxiv.org/pdf/2608.10213v1)

**作者:** Bethel Hall `[一作]` (Stevens Institute of Technology), William Eiers `[通讯]` (Stevens Institute of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出VeriFin框架，用神经符号方法对LLM生成的金融数值声明进行验证，确保其符合XBRL事实与授权公式后才可接受；

**💡 创新点**

创新点在于将生成与验证分离，利用XBRL事实自动归一化运算元、来源公式，使用SMT求解器Z3实现确定性约束检查，并通过unsat core提供精准修复反馈；

**🔧 技术方法**

采用大语言模型进行语义解析和检索规划，XBRL解析与事实归纳，符号求解器Z3进行约束求解；

**📊 数据集**

使用自构造的600题XBRLFiling数据集（来自28家公司10-K的计算链）和FinanceBench 67题的公开金融问答数据集；

**📈 对比分析**

与Direct LLM、LLM Judge、Judge+Formula、Program-of-Thought等基线比较，VeriFin在两组数据上均实现0错误接受率，准确率分别为92.2%（XBRLFiling）和83.3%（FinanceBench），覆盖率高达98.8%和80.6%；

**⚠️ 局限性**

局限在于对无法明确归一化或公式的情况会产生Abstain，导致覆盖率下降；对正确答案的过度拒绝率仍高，需提升正确答案的保留率；

---

## 144. Long-Time Trajectory Approximation via SA-NODEs: Model Predictive and Floquet Strategies

**arXiv ID:** 2608.10738 | [PDF](https://arxiv.org/pdf/2608.10738v1)

**作者:** Ziqian Li `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Nikolaos M. Matzakos `[通讯]` (School of Pedagogical and Technological Education)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了两种训练策略——模型预测重启（MPC）和 Floquet 正则化——以消除传统 SA‑NODE 在长时间预测中出现的指数误差衰减瓶颈，从而实现对动力系统的长期近似。

**💡 创新点**

创新点在于将数据重置（每个窗口从观测值重新启动）和动力学重置（利用稳定极限环的 Floquet 收敛）结合到 SA‑NODE 训练中，并给出了对应的误差上界；同时提供了可测量的后验收据（返回图谱谱半径、时域平均误差、振荡幅度）以验证这些策略的有效性。

**🔧 技术方法**

使用的技术包括：SA‑NODE 架构（时间线性偏置的残差网络）、自适应窗口划分与多窗口训练（MPC）、基于 Floquet 理论的损失正则化、时域编码（将时间映射到正弦余弦），以及数值积分与自适应多步积分（RK4、Euler）来评估流。

**📊 数据集**

实验采用了四个经典动力学基准：受迫 Duffing 方程、摆式系统、Stuart–Landau 与 Van der Pol 振荡器，以及一个自律臂摆系统，来分别检验两种策略在非自治与极限环动力学下的表现。

**📈 对比分析**

通过与单一（单模）SA‑NODE 的基线对比，MPC 在多窗口训练后使误差保持在设定阈值内，宽度需求从指数级降低到线性级；Floquet 正则化后，已训练模型在数百周期内误差增长线性，可观测到收敛半径满足阈值且锁定周期相位匹配，从而实现长时间的稳定预测。

**⚠️ 局限性**

局限性包括：1）MPC 需要在每个窗口切换时获取真实状态；2）Floquet 正则化依赖于目标系统拥有稳定极限环，且对时域编码与振荡幅度的假设在实测模型中往往不满足；3）理论保证是后验可测量的，缺乏严谨的前验收据；4）在高维或强非线性系统中训练成本与收敛性仍需进一步研究。

---

## 145. Off-Axis, On Purpose: Where a Transformer Computes Concepts and Why it Does So

**arXiv ID:** 2608.10251 | [PDF](https://arxiv.org/pdf/2608.10251v1)

**作者:** Mark Oskin `[一作]` `[通讯]` (University of Washington), Mark Oskin (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了Transformer在内部计算时的几何结构，证明概念阶段的表示被约束在距离输出轴约90°的子空间内进行，而注意力在此空间进行跨标记混合以“隔离”词汇预测，答案则在后期以对角写入；通过角度、参与度比、正交Procrustes等测量方法量化了这一结构，并通过在概念/标记阶段强制90°角度以及在层间插入固定旋转设备来验证并改进这种两阶段计算；

**💡 创新点**

创新点包括（1）系统性量化Transformer内部状态相对于输出轴的几何关系；（2）证明off‑axis子空间功能为隔离混合，其破坏成本高达64–84倍；（3）揭示答案是被追加而非旋转的；（4）通过强制90°角度并插入固定旋转设备，显著提升训练可靠性，并展示该“框架”为自由度（gauge）可被预先指定；

**🔧 技术方法**

使用角度测量（write angle）、参与度比（participation ratio）、正交Procrustes拟合、层次梯度惩罚（angle loss、KL loss）、在层间插入固定旋转设备（dense与sparse两种形式）、随机基准与对照实验、基于GPT‑2 small的Transformer架构；

**📊 数据集**

使用OpenWebText语料库（约8.9B token）进行训练，验证集120M token；评测时使用LAMBADA、BLiMP以及perplexity；

**📈 对比分析**

对比未强制、强制全轴、无设备、设备（dense、sparse）四种训练方案；基线模型在LAMBADA 0.266、BLiMP 0.805、ppl 19.2；设备方案保持相同质量，唯一提升是训练收敛率：无设备5/9收敛，设备7/8（稀疏9/9）收敛；在维度空间中，设备方案将概念阶段参与度从约25提升至约54（dense）或30（sparse）；

**⚠️ 局限性**

仅在125M GPT‑2 small、单一语料、单一规模下验证，未检验更大模型或不同任务；设备对性能影响随规模可能变化；对角度约束对更复杂任务的可迁移性未知。

---

## 146. Reinforcement Learning-Based Laser Cutting Machine Parameter Optimization

**arXiv ID:** 2608.10549 | [PDF](https://arxiv.org/pdf/2608.10549v1)

**作者:** Khanh Quan Pham `[一作]` (Chungbuk National University), Taehong Kim `[通讯]` (Chungbuk National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并验证了一种基于 Q‑learning 的 RL^2C 算法，用于在光学薄膜切割中三阶段（焦距、功率、斜切）自适应优化激光切割参数，显著降低斜切尺寸和材料浪费。

**💡 创新点**

创新点在于：①引入动态环境空间适应机制，使 Q‑table 在训练过程中能够实时扩展和更新以适应不同批次和材料的全新状态；②采用分阶段优化策略和三种不同的奖励函数，针对每个阶段的目标（线宽、亮度、斜切尺寸）提供量化反馈；③在工业实验中实现完全自主学习，减少人工干预。

**🔧 技术方法**

技术手段包括：Q‑learning 与 ε‑greedy 探索策略；动态 Q‑table 更新（STATIC/DYNAMIC 版本）；多阶段奖励设计；离散动作空间（±步长增减焦距、功率、斜切尺寸）。

**📊 数据集**

使用了 210 批次光学薄膜实验数据（90 批训练、120 批测试），包含三种薄膜（Material A、A1、B），每批约 70 组实验样本，记录线宽、亮度、斜切尺寸等指标。

**📈 对比分析**

与随机搜索、RL‑Bayesian Optimization、RL‑PSO 三种基线方法对比，RL^2C 在训练和测试阶段均实现了更低的平均步数（约 1.6–3.8 步）和更短的计算时间（0.02–0.03 秒），相对基线方法提升步数降低 12.5%–81.8%，验证了其高效性与可扩展性。

**⚠️ 局限性**

局限性包括：①仅在离散参数空间内实验，连续或更大维度参数仍需进一步研究；②动态 Q‑table 更新在极大状态空间下可能导致记忆与计算负担；③实验集中在三种薄膜，其他材料或复杂工艺的泛化效果尚未验证。

---

## 147. A Graph Approach to the Academic Publishing Network: A Heterogeneous Model and Structural Screening over OpenAlex Open Data

**arXiv ID:** 2608.10774 | [PDF](https://arxiv.org/pdf/2608.10774v1)

**作者:** Robert Šamárek `[一作]`, Radek Martinek `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并实现了一种基于OpenAlex开放数据的异构多变量学术出版网络图模型，利用投影（引文网络、合作网络）计算可解释的结构指标，并设计了三种异常筛查器（密集合作团、局部闭合引文环、主题孤立期刊），同时开发了开源Python库apnet及交互式可视化界面；

**💡 创新点**

创新点在于：①采用异构图模型并通过投影避免语义混淆；②以结构可解释性为核心的异常筛查器，输出可视化证据而非二元分类；③提供完整可复现的工作流和工具，支持从单机构到全球主题的无缝迁移；

**🔧 技术方法**

技术主要包括：NetworkX与pandas处理图与数据；基于Louvain/Leiden的社区检测；PageRank、k-core、加权聚类系数、Betweenness（采样近似）等结构度量；自定义异常评分公式；FastAPI+Sigma.js实现Web交互；

**📊 数据集**

使用OpenAlex公共API获取的开放数据，分别构建了：①VSB–TUBO 2020–2025机构语料（约35,000篇论文，10万作者）及其一跳引用邻域；②基于“large language model”标题的全球主题语料（12,520篇论文，80,000节点）；③对比基准集包含341个期刊（167个被取消/撤稿），用以验证异常特征；

**📈 对比分析**

在机构案例中，社区检测与实际研究组高度一致；中心性分析揭示跨学科桥梁；异常筛查器排序可供人工评估；在期刊案例中，匹配控制后发现“主题广度”AUC≈0.70是唯一稳健特征；使用PageRank计算的期刊声望对计数指标的作弊抵抗力提升约10倍；

**⚠️ 局限性**

局限包括：①OpenAlex数据缺失导致属性不完整，需使用分位数稳健度量；②图模型基于一跳邻域，可能漏检超出边界的循环；③缺乏严格的二元标签，验证只能通过一致性或匹配控制；④大规模全局分析需要迁移至图数据库（如Kùzu）并行实现；

---

## 148. Lost in Reconstruction: Aligning Action Representations with Language in Vision-Language-Action Models

**arXiv ID:** 2608.10484 | [PDF](https://arxiv.org/pdf/2608.10484v1)

**作者:** Li Wenjie `[一作]` (Carnegie Mellon University), Yonatan Bisk `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出一种语义对齐的动作分词器SALT，以改进视觉语言动作模型中的动作表示。

**💡 创新点**

创新点在于通过冻结的视觉语言模型生成指令来为分词器提供对齐目标，保持动作轨迹的可执行性同时保留语言语义结构。

**🔧 技术方法**

采用VQ‑VAE架构的分词器，并在分词器训练中加入语言对齐损失（LM生成指令），随后在VLA中使用Qwen2.5‑0.5B backbone。

**📊 数据集**

使用BridgeV2真实机器人操作轨迹与自然语言指令数据集。

**📈 对比分析**

与传统仅基于重构的VQ‑VAE、Bin和FAST分词器对比，SALT在SimplerEnv上任务成功率从31.2%提升至71.9%，并显著提升词汇可解码性和动作嵌入的宏观F1。

**⚠️ 局限性**

局限包括数据集词汇量有限、仅适用于可学习潜在的分词器、实验规模受限且缺乏因果机制解释。

---

## 149. The Multilingual Quantization Tax: Structural Collapse and Typological Fragility in Edge SLMs

**arXiv ID:** 2608.09941 | [PDF](https://arxiv.org/pdf/2608.09941v1)

**作者:** Mohammad Wathiq Soualhi `[一作]` `[通讯]`, Mohammad Wathiq Soualhi

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过零样本评估，系统检验 Gemma 4 与 Qwen 3.5 在 4-bit 量化后在八种语言（含非拉丁脚本）上的推理性能，揭示量化税在语言类型和任务域中的不均匀分布。

**💡 创新点**

创新点在于首次将语言类型与认知领域视为维度，量化并解释了四种新现象——类型脆弱性、家语脆弱悖论、域特定遗忘与量化抵抗，证明量化对多语言模型的结构性影响远超单一指标。

**🔧 技术方法**

使用了零样本框架、lm-evaluation-harness、Calibration‑free ZeroBitQuantization（ZeroBit）以及禁用内部思考模式，确保评估纯粹反映权重截断的结构影响。

**📊 数据集**

数据集包括 MMLU Pro X Lite（多学科推理）和 Global PIQA（物理常识推理），分别在英语、阿拉伯语、俄语、中文、日语、印地语、斯瓦希里语和约鲁巴语八种语言上进行测试。

**📈 对比分析**

评估方法为计算量化前后模型零样本准确率之差（Δ）作为量化税；结果显示，平均税约为 4–5%，但低资源非拉丁语言出现显著下降甚至随机噪声；逻辑密集任务损失更大，关联性任务相对鲁棒。

**⚠️ 局限性**

局限性包括仅覆盖 Gemma/Qwen 两大族群与 2B/4B 参数规模，未探讨更大模型或其他架构；仅使用 Calibration‑free 量化，未评估多语言 PTQ；仅零样本、禁用内部思考，未考虑 few‑shot 或生成任务；仅限于结构化推理与常识分类，未研究开放式生成性能。

---

## 150. DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation

**arXiv ID:** 2608.10636 | [PDF](https://arxiv.org/pdf/2608.10636v1)

**作者:** Zhuchenyang Liu `[一作]` (Aalto University), Yu Xiao `[通讯]` (Aalto University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对视觉文档检索（VDR）进行端到端单向量蒸馏，生成总计 524 M 参数的 DistilVDR 系统，既能在查询端保持轻量化，也能在文档端充分利用视觉信息。

**💡 创新点**

创新点在于：(1) 双向（query 与 document）以点对点余弦对齐方式从单一 8 B 视觉‑语言教师蒸馏两侧编码器；(2) 采用非对比学习、无负样本采样的无标签蒸馏；(3) 采用不对称编码器架构，将大视觉‑文本塔放在文档侧，轻量文本塔放在查询侧。

**🔧 技术方法**

技术包括：冻结的 8 B Qwen3‑VL‑Embedding 教师；点对点余弦对齐损失；InternViT‑300M 视觉编码器、ModernBERT‑base 文本编码器、DistilBERT‑base 查询编码器；动态裁剪与固定 Tile 数目；线性投影与均值池化映射到 4096 d 共享空间。

**📊 数据集**

使用 1.2 M 公开许可的文档图像（711 K 基础、454 K 多域、31.7 K 财务）与 1.49 M NanoVDR 训练查询，评估在 22 个 ViDoRe v1/v2/v3 数据集上。

**📈 对比分析**

在 12 个已重现的基线（单向量 250 M–8.8 B 以及多向量）上进行统一评估；DistilVDR‑HiRes/Fast 在 ViDoRe 上平均 NDCG@5 分别为 61.74 与 59.98，超过所有子 1 B 基线；索引体积缩小 15.6 倍，索引速度提升 7–10 倍，保持 86.9%/84.4% 的教师性能。

**⚠️ 局限性**

局限性包括：继承 8 B 教师的 embedding 弱点；蒸馏过程不考虑 query‑document 交互；视觉 Token 数目固定，未做内容自适应；仅支持拉丁字符查询；未检验更强教师或联合对比学习；相较于最强多向量基线仍相差 7–10 分。

---

## 151. Assessing Reliability of BERT-Based Models on Question Answering Tasks

**arXiv ID:** 2608.10806 | [PDF](https://arxiv.org/pdf/2608.10806v1)

**作者:** Pooja Yadav `[一作]` (Malaviya National Institute of Technology), Marko Robnik Šikonja `[通讯]` (University of Ljubljana)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了四种 BERT 变体（RoBERTa、BERT-Base、DistilBERT、ALBERT）在问答任务中的可靠性，分析了内部随机性（Monte Carlo Dropout）和输入改写（语义保持的改写）对答案一致性的影响。

**💡 创新点**

提出了一种结合内部结构扰动与输入改写的可靠性评估框架，首次系统比较了模型在这两类扰动下的输出稳定性，并用统计检验验证不同模型在不同数据集上的差异显著性。

**🔧 技术方法**

使用了 Monte Carlo Dropout、BART‑based paraphrasing、Sentence‑BERT 余弦相似度、F1 评分以及 Wilcoxon、Welch‑t 检验等技术。

**📊 数据集**

实验数据集为 SQuAD v2.0 和 QuAC，分别覆盖单轮与对话式问答场景。

**📈 对比分析**

通过平均余弦相似度、F1 评分及其标准差量化可靠性；RoBERTa 在两数据集上表现最稳定，DistilBERT 在内部扰动下更稳健，ALBERT 对输入改写更鲁棒；统计检验显示部分模型差异显著，说明可靠性并不完全随准确率提升。

**⚠️ 局限性**

局限性包括仅关注 BERT 系列模型、仅使用两类扰动方法（Dropout 与改写），未探索其他模型或更复杂任务，且评估主要基于自动指标，人工评估覆盖范围有限。

---

## 152. LLM Agents Factory: Retrieval of Domain-Specific LLM Agents

**arXiv ID:** 2608.09934 | [PDF](https://arxiv.org/pdf/2608.09934v1)

**作者:** Vitalii Belov `[一作]` (Sber AI), Semen Budennyy `[通讯]` (Sber AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LLM Agents Factory，一个基于检索的框架，利用预构建的 2 万+ agent profiles 快速生成领域特定的 LLM 代理。

**💡 创新点**

创新点在于将 agent 构造视为信息检索问题，结合结构化 agent 仓库、语义检索与知识蒸馏，以显著降低推理成本并提升可控性。

**🔧 技术方法**

使用了 BGE、MiniLM、MPNet 等句子编码器进行检索，并利用 LoRA 在小模型上蒸馏 agent 生成，同时采用 Qwen3‑4B 等大模型作为背骨。

**📊 数据集**

评测使用了 MMLU、BIG‑bench、BIG‑bench Hard 三个基准数据集。

**📈 对比分析**

相较于非 agent、Qwen3‑4B 零射击和 AutoGen，检索+单 agent 方案在准确率上提升至 82.3%/85.6%/68.5%，同时将 token 消耗和延迟分别降低约 3 倍和 4 倍。

**⚠️ 局限性**

局限在于仅在单轮问答/推理任务上验证，agent 库基于维基百科类别和固定角色，可能缺乏行业专属语义；蒸馏过程依赖教师模型的假设，易传递偏差。

---

## 153. PBD-AG: Persistent Baseline-Delta Active Graphs with Uncertainty-Aware Inspection for Long-Horizon Service Robots

**arXiv ID:** 2608.10449 | [PDF](https://arxiv.org/pdf/2608.10449v1)

**作者:** Shuo Bao `[一作]` (Peking University), Xinzhou Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了PBD-AG框架，能在未知环境中自主构建并持续维护基线-增量场景图，分离稳定基准与可修订的动态事件；

**💡 创新点**

创新点在于将结构基线与动态事件分离，采用几何可见性门控负证据、持久身份管理以及图引导的主动检查，使得长期任务机器人能可靠更新世界模型；

**🔧 技术方法**

使用RGB‑D与LiDAR感知、SigLIP/ DINO/ SAM2等视觉模型、DBSCAN聚类、几何可见性门、事件审计流、图优先主动采样等技术；

**📊 数据集**

主要在OmniGibson/BEHAVIOR模拟器的Gates bedroom、Hotel suite、Benevolence等室内场景进行评估，并在物理机器人上进行演示；

**📈 对比分析**

与适配的DynaMem与ConceptGraphs在相同证据下对比，PBD-AG在共享证据下Coarse‑F1达到0.868，超过对手11.1/17.2点；在动态记忆评估中IDF1达0.833，事件召回11/12，且无身份切换；

**⚠️ 局限性**

局限性包括对几何可见性门的依赖导致对遮挡或误检不够鲁棒，受限于高动态场景和复杂视角的识别，且未完成全局地图重构与离线优化，需进一步在更大规模真实环境中验证。

---

## 154. Sheaf-Based Federated Representation Learning

**arXiv ID:** 2608.10016 | [PDF](https://arxiv.org/pdf/2608.10016v1)

**作者:** Gabriele D'Acunto `[一作]` (Sapienza University of Rome), Paolo Di Lorenzo `[通讯]` (Sapienza University of Rome)

**通讯引用:** 4480 | [OpenAlex ID](https://openalex.org/A5000852147)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于层束（Sheaf）的联邦表示学习框架（SFRL），并实现了完全分布式的 Sheaf‑FRL 算法，用于在异构联邦系统中学习并对齐各节点的潜在表示。

**💡 创新点**

创新点在于：①不强制所有节点共享同一全局潜在空间，而是通过可学习的层束限制映射实现相邻节点潜在空间的几何对齐；②利用层束拉普拉斯诱导的二次粘合正则化（gluing penalty）实现柔性一致性约束；③采用闭式正交/Stiefel Procrustes 更新，可处理不同维度的潜在空间；④只在少量 pilot 样本上评估正则化，显著降低通信成本。

**🔧 技术方法**

主要技术：层束理论与网络层束、图拉普拉斯正则化、正交/Stiefel 限制映射、Procrustes 最小化、分布式交替优化（梯度更新 + 约束更新）以及随机/确定性收敛分析。

**📊 数据集**

使用 MNIST 数据集进行实验：在 15 个节点上构造数据分布异构（目标类别划分、分布偏移 s），其中 10% 样本作为 pilot 共享给所有节点，其余 90% 按分布划分到各节点；实验还对两节点的压缩实验做了验证。

**📈 对比分析**

与 ComFed、Sheaf‑FMTL、无协作基线以及 FedProto、FedMuscle 等方法比较；评价指标为本地准确率（private accuracy）和通信后准确率（communication accuracy）。实验结果表明 Sheaf‑FRL 在所有数据异构程度和压缩维度下均明显优于基线，尤其在高分布偏移和高压缩率时性能提升最为显著。

**⚠️ 局限性**

局限性：①需要共享 pilot 表示，存在潜在隐私泄漏风险；②pilot 的选择和数量会影响性能；③在极大规模网络或极端异构模型下的通信/计算扩展性尚待进一步验证；④实验仅覆盖监督分类任务，尚未验证自监督、半监督或多模态场景。

---

## 155. Whisper-Aware LLM: Self-Supervised Uncertainty Learning for Robust Whispered Speech Recognition

**arXiv ID:** 2608.10836 | [PDF](https://arxiv.org/pdf/2608.10836v1)

**作者:** Gaopeng Xu `[一作]` (Qwen Business Unit of Alibaba), Haitao Yao `[通讯]` (Qwen Business Unit of Alibaba)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 Whisper-Aware LLM 框架，通过自监督任务让 Audio-LLM 能感知并处理低质量的咕噜语音信号，进而提升 ASR 的鲁棒性和可靠性

**💡 创新点**

1）自监督的信号不确定性感知模块（UPM），通过 F0 预测和谱重建学习信号缺陷；2）Confidence‑Fused Decoding，将全局不确定性指令和帧级置信度注入 LLM 解码器，动态调节注意力；3）三阶段训练策略，稳健地融合 UPM 与大型预训练 LLM

**🔧 技术方法**

自监督学习（MSE、谱重建）、Transformer 结构、LoRA 参数微调、注意力偏置、指令嵌入、音频编码器、Qwen‑7B 语言模型

**📊 数据集**

大型混合语料（WenetSpeech、GigaSpeech、AISHELL‑1、LibriSpeech、wTIMIT、AISHELL6‑Whisper、噪声集合）用于 UPM 预训练；AISHELL‑1、LibriSpeech、wTIMIT、AISHELL6‑Whisper、噪声子集用于微调；测试集包含 AISHELL6‑Whisper、wTIMIT、AISHELL‑1、LibriSpeech-clean 及 1000 条噪声幻觉集

**📈 对比分析**

与 Whisper‑v3、Qwen2‑Audio、Qwen3‑ASR、FunASR、Seed‑ASR 等基线比较，评估指标为 WER/CER 与幻觉率；结果显示在 AISHELL6‑Whisper 上实现 1.31% CER（比上一最佳低 17%），在普通语音上 0.63% CER，噪声幻觉率从 25% 降至 4.5%，显著提升性能与可靠性

**⚠️ 局限性**

依赖预训练的 Audio‑LLM 与大规模数据，训练成本高；UPM 的自监督任务可能不适用于所有语言或低资源场景；模型对极端噪声或非常短的咕噜语句仍可能出现错误或幻觉

---

## 156. Benchmarking LLM-Guided Control-Plane Policies for Backend Fault Isolation in HAProxy

**arXiv ID:** 2608.10532 | [PDF](https://arxiv.org/pdf/2608.10532v1)

**作者:** Aman Chauhan `[一作]` (Independent Researcher), Vishnu Pendyala `[通讯]` (San José State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个闭环的 LLM 控制平面，将大型语言模型替换为 HAProxy 的路由策略，实时读取 Prometheus 与 HAProxy 数据平面 API 的监控信息，并通过函数调用对后端服务器的权重和排队状态进行调整。

**💡 创新点**

首次证明 LLM 能在运行时充当负载均衡控制器，发现了约 3 B 活跃参数的能力阈值、量化了可用性提升与尾部延迟成本的权衡，并揭示了推理成本与控制效果之间的负相关关系。

**🔧 技术方法**

使用 15 种开源 LLM（Qwen、Gemma、Granite、GPT‑OSS 等），通过 OpenAI‑兼容的函数调用接口进行控制；依赖 HAProxy 数据平面 API、Prometheus、Grafana k6 负载生成器、以及自定义的安全 guardrail 层来限制模型动作。

**📊 数据集**

实验基于一个 600 s 的可复现基准，后端池规模从 3 到 9，故障模式为每三台服务器中一台持续 5% 的 HTTP 5xx；通过 10 u/v 每个后端的虚拟用户来保持负载比例；比较 round‑robin 与 least‑conn 两种调度算法。

**📈 对比分析**

在 240 次实验中，超阈值模型（≥3 B 活跃参数）平均可将客户端 5xx 降低约 88%（最高可达 96–97%），但均值延迟提升约 4–5 ms，p95 延迟增加 2.6–2.8 倍；推理成本（token 数）与决策时间呈正相关，开启推理会导致控制循环超时，导致效果下降；在成本效益上，非推理、低成本模型（如 GPT‑OSS low）最优。

**⚠️ 局限性**

局限性包括：仅测试单一持久性故障模式，后端规模限制在 9 台；仅针对 HAProxy+Flask 堆栈；固定 10 s 采样周期、单 GPU 推理硬件；未探讨瞬态或级联故障、不同监控来源和大规模集群；guardrail 设计对行为影响显著，未进行消融；所有实验基于同一随机种子，缺乏不同种子下的鲁棒性验证。

---

## 157. MemSpec: Memory-Aware Runtime for Adaptive Draft Scheduling in Speculative Decoding on Edge Devices

**arXiv ID:** 2608.10362 | [PDF](https://arxiv.org/pdf/2608.10362v1)

**作者:** Eunjeong Kim `[一作]` (Kyungpook National University), Myeonggyun Han `[通讯]` (Kyungpook National University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向边缘设备的记忆感知运行时 MemSpec，用于在自回归大型语言模型推理中实现自适应草稿调度，以提升推理吞吐量

**💡 创新点**

创新点在于将草稿选择与草稿可用性分离，利用离线预测器预估最佳草稿并通过主动驻留管理（预取与驱逐）保持高效可用草稿，从而在内存受限的边缘设备上实现无阻塞自适应推理

**🔧 技术方法**

核心技术包括BERT编码器预测器（输出草稿排名）、基于Top‑K的草稿缓存管理器（主动预取、驱逐）、周期性（N迭代一次）的调度与预测接口、以及非阻塞的推理循环；实现基于PyTorch、CUDA 12.6的高性能推理引擎

**📊 数据集**

在 LLaMA‑2 7B（GPTQ INT4）和 Qwen2.5 7B（GPTQ INT4）目标模型上使用5个草稿模型（1个通用+4个域特化：代码、数学、法律、医疗），并在5个多域数据集（Alpaca、LiveCodeBench、Omni‑MATH、MMLU‑Law、MMLU‑Medical）进行评估

**📈 对比分析**

与基线（通用静态、Oracle‑静态、MAB‑Async）和Oracle‑动态（理想）对比，MemSpec 在 Jetson Orin Nano 上平均提升吞吐量约 40.7%（相对 MAB‑Async）且接近 Oracle‑动态（≈95‑97%），同时显著降低 49.4% 的回退执行，预测开销仅 3.9%

**⚠️ 局限性**

局限性包括：预测准确性仍有提升空间（但不影响大部分收益）；需在不同硬件平台上进一步验证调度间隔与缓存容量的鲁棒性；目前实验集中于 1‑GPU 边缘设备，对多 GPU 或更大内存环境的适配尚待探讨

---

## 158. AI Query Compilation for Unified and Optimized Execution

**arXiv ID:** 2608.10139 | [PDF](https://arxiv.org/pdf/2608.10139v1)

**作者:** Yeounoh Chung `[一作]` (Google Cloud), Fatma Ozcan `[通讯]` (Google Cloud)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将SQL关系操作与LLM推理统一编译为单一JAX张量程序，消除跨设备的数据传输与拆分执行瓶颈。

**💡 创新点**

创新点在于把传统数据库算子与深度学习算子映射到同一张量图，借助XLA实现全局优化与自动多设备切分。

**🔧 技术方法**

使用JAX、XLA、MLIR、GSPMD/PMAP等编译框架，将查询转换为GPU/TPU可执行代码。

**📊 数据集**

使用SemBench Rotten Tomatoes 影评与电影数据集（约5万行）以及自合成的AI查询进行评测。

**📈 对比分析**

与传统CPU‑TPU拆分执行基线对比，统一编译方案在TPU上实现5.3倍的延迟加速、9.8倍的吞吐提升，并在多设备上实现线性水平扩展。

**⚠️ 局限性**

局限性包括缺乏动态形状编译、对高选择性过滤时性能不如拆分模式、并发多租户时TPU队列串行导致吞吐受限，以及对多模态算子与动态数据跳过的支持不足。

---

## 159. Polynomial Bounds on Degeneration Order from Commutativity Properties of Tensor Slices

**arXiv ID:** 2608.10179 | [PDF](https://arxiv.org/pdf/2608.10179v1)

**作者:** Shree Ganesh `[一作]` (ENS de Lyon), Rafael Oliveira `[通讯]` (University of Waterloo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究3维张量的边界秩（border rank）与张量秩之间的关系，提出并证明了新的上界，特别是对具有3个切片且满足(r,3)-generic性质的张量；在此基础上给出了误差阶（error degree）与退化阶（degeneration order）的上界，进一步推导出非平凡的去边界化（debordering）结果。

**💡 创新点**

创新点在于：①对(r,3)-generic张量给出了误差阶≤1、退化阶≤(r−1)^3+(r−1)^2的指数级改进；②引入1-regular矩阵与Weyr正则化、Motzkin‑Taussky定理的组合，克服了多切片下缺乏对角化条件的难题；③在过完备（overcomplete）情形下通过可对角化扩展（commuting extensions）构造边界秩与误差阶的关系。

**🔧 技术方法**

主要技术包括：张量切片的矩阵分解、(r,3)-generic因式分解、约束矩阵的正则化与近似对角化（ASD）技术、Motzkin‑Taussky定理与Weyr形式、对矩阵的1-regular性与可逆变换、对误差阶与退化阶的代数计量（valuation）分析，以及边界秩的可对角化扩展理论。

**📊 数据集**

本文未使用实验数据集，全部结论来自纯理论分析与代数几何证明。

**📈 对比分析**

与过去的最优指数上界（如3^(m+n+p-3)r）相比，本文提供的上界显著更小（如在三切片、(r,3)-generic情形下误差阶≤1，退化阶≤(r−1)^3+(r−1)^2），从而实现了更紧的去边界化上界（张量秩≤2r）。

**⚠️ 局限性**

局限性包括：①对切片数>3的情形缺乏Motzkin‑Taussky类的必要与充分条件；②1-regular性假设对部分张量不成立；③过完备情形下仍需对π_{r,p,n}(C(r,p))投影是否封闭做更深入研究；④对非(r,3)-generic张量的误差阶与退化阶尚无相应的上界。

---

## 160. Safe Observation Capacity for Opponent Exploitation under Showdown Censoring

**arXiv ID:** 2608.09954 | [PDF](https://arxiv.org/pdf/2608.09954v1)

**作者:** Jiaxing Guo `[一作]` `[通讯]` (Imperial College London), Jiaxing Guo (Imperial College London)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在扑克类游戏中因翻牌导致的隐私信息缺失问题，提出安全主动去敏化（SAD）方案，通过主动揭露（floor‑safe probe）与序列表单流恢复隐藏行为，并给出安全观察容量与证书成本的理论与实验验证。

**💡 创新点**

① 证明被动秀牌数据在行动依赖揭露下会产生选择偏差；② 设计“floor‑safe probe”和“安全观察容量”，给出其可达性前沿及样本成本上界；③ 提出基于公共异常的启发式路由和鲁棒响应的SAD框架，实现条件证书同时保持无条件价值底线。

**🔧 技术方法**

使用序列表单与线性规划求蓝图与安全集合；通过流分析恢复 fold 质量；构建统计置信集并做鲁棒优化；对安全观察容量进行 LP 与边际价格分析；在桶化终局、固定棋盘河流子游戏及合成受限链上进行实验评估。

**📊 数据集**

实验数据包括：桶化翻牌河流结束局（粗细两套）；固定棋盘无桶化河流子游戏（public twin）；54 个哈希种子及未训练的 CFR 对手；51 个已求解的分组揭露线对手；以及 Bandit、MDP 与被遮盖链的合成实例。

**📈 对比分析**

与无揭露、被动估计、公共鲁棒、数据偏差响应等基线比较，SAD 在大多数实验中获得最高或接近最优的已证书收益，公共异常路由提升约 22% 的证书值；理论预测的容量与样本成本与实验结果高度吻合；在 public twin 上可获得 ≥96% 的安全可利用收益。

**⚠️ 局限性**

主要限制：需要对手不随 probe 适应的静态假设；只适用于可揭露（A3）线；仅在单局子游戏与两街规模上验证，未扩展到多街或软底线；需要先验概率与机会已知；证书具有条件性，仅在覆盖满足时有效。

---

## 161. ENCORE: Efficient Noise Context-Aware Representation for Low-Dose CT Denoising

**arXiv ID:** 2608.10343 | [PDF](https://arxiv.org/pdf/2608.10343v1)

**作者:** Minwoo Yu `[一作]` (Yonsei University), Adam S. Wang `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种低剂量CT去噪框架ENCORE，通过逼真噪声模拟、噪声自协方差估计和飞行卷积模块，实现了自适应噪声上下文引导的去噪。

**💡 创新点**

创新点包括：1）使用Cornish‑Fishers展开改进Poisson噪声建模；2）将局部噪声功率与相关性提取为自协方差图；3）引入FlyingConv实现按区块动态调整卷积权重；4) 通过控制噪声上下文强度实现零样本可调去噪。

**🔧 技术方法**

技术手段包括：自监督Noise2Noise训练、噪声自协方差预处理、基于分组深度卷积的飞行卷积模块、CUDA自定义核加速、零样本目标噪声水平控制。

**📊 数据集**

使用公开的Mayo2016（Siemens）和Mayo2020（GE）低剂量CT数据集，以及自制桌面扫描的真实CT物理数据进行评估。

**📈 对比分析**

与传统UNet、DnCNN、NADD、+COV、注意力模型（NAFNet、Uformer）比较，ENCORE在PSNR/SSIM/AUHOC上均显著提升，且MACs显著降低，实际推理时延与+COV相当；在跨厂商和真实物理数据上表现出更好的鲁棒性。

**⚠️ 局限性**

局限性：1）实际延迟未随MACs下降显著提升，主要受GPU内存带宽限制；2）训练基于模拟数据，未完全涵盖散射、衰减硬化等物理因素；3）对噪声自协方差的依赖可能导致对纹理过度平滑，需通过d_target调节来平衡。

---

## 162. Energy-Efficient Joint Optimization of VLC and Fog Computing Resources Allocation

**arXiv ID:** 2608.10752 | [PDF](https://arxiv.org/pdf/2608.10752v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 163. When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs

**arXiv ID:** 2608.10489 | [PDF](https://arxiv.org/pdf/2608.10489v1)

**作者:** Congyang Ou `[一作]` (Northwestern Polytechnical University), Zhenbo Luo `[通讯]` (MiLM Plus, Xiaomi Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究 Vision‑Language Models 的视觉标记压缩，提出基于跨模态残差和多样性选择的训练‑free 方法 SIEVE。

**💡 创新点**

创新点在于发现视觉信息在 LLM 层级中被文本子空间逐渐吸收，并用 CMA 衡量吸收程度、CMR 量化不可解释的视觉余弦，并在残差空间中进行多样性贪心选择，从而实现高效压缩。

**🔧 技术方法**

使用 Tikhonov 正则化最小二乘投影、余弦相似度、多头注意力权重、残差空间多样性贪心等技术。

**📊 数据集**

实验使用 GQA、MMBench、MMBench‑CN、MME、POPE、SQA、VQAv2、TextVQA 八大数据集，并在 LLaVA‑1.5、LLaVA‑NeXT、Qwen2.5‑VL 三种 VLM 上评估。

**📈 对比分析**

与 FastV、SparseVLM、VisionZip、HoloV、SCOPE 等方法对比，SIEVE 在保留 11.1% 视觉标记时保持 96–99% 的原模型平均性能，预填速度提升 2.5–3.6×，KV‑cache 缓存缩减 6–7×。

**⚠️ 局限性**

局限性包括：仍需手动设定能量比例 η 和头数 H_top；实验仅覆盖部分 VLM 框架，跨框架泛化需要进一步验证；在极端压缩率下可能仍出现语义丢失，需进一步研究。

---

## 164. Link-adaptive digital twin for robust physical-layer modeling in hybrid-amplified ultra-wideband optical networks

**arXiv ID:** 2608.10517 | [PDF](https://arxiv.org/pdf/2608.10517v1)

**作者:** Xiaoxuan Gao `[一作]` (Beijing University of Posts and Telecommunications), Yuefeng Ji `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了 LA‑DT（Link‑Adaptive Digital Twin）数字孪生模型，能够对混合放大（Raman+EDFA）超宽带光纤链路进行物理层建模并实现高精度 GSNR 估计。

**💡 创新点**

创新点包括：① 将 GSNR 建模拆分为三项功率预测（NLI、ASE、信号功率）以消除 EDFA 异构影响；② 设计 DeepModNet 架构并引入线性调制层（LML）实现域感知动态特征调制；③ 通过域判别器实现对未知场景的域识别，随后仅微调 LML 以实现极少样本（20 条）快速适配；④ 明确考虑拉曼泵引入的插入损耗，提升实用性。

**🔧 技术方法**

使用技术包括：DeepModNet 神经网络架构、线性调制层（LML）、域判别器、少样本微调、基于 GNPy 的半解析 GN/EGN 传播仿真生成训练数据，以及 Adam 优化器等。

**📊 数据集**

数据集：通过 GNPy 生成 35 个多场景（不同光纤长度、发射功率、泵功率、插入损耗）各 500 条样本（300 训练/200 测试）得到 10,500 训练样本和 7,000 测试样本；另外 12 个未见场景各 20 条微调样本/200 条测试样本。

**📈 对比分析**

与基线 BaseCondNet（将场景参数直接拼接为输入）比较，DeepModNet 在 NLI、ASE、信号功率预测分别提升 56.0%、58.4%、52.7%；GSNR 估计 RMSE 降至 0.114 dB（提升 55.8%）。在未见场景中，少样本微调后 RMSE 分别降至 0.218、0.117、0.160 dB，实现 61.2% 的整体提升。

**⚠️ 局限性**

局限性：仅针对等跨度单模链路；未覆盖多跨度、不同调制格式、流量负载及 ROADM 过滤等实际复杂情况；实验验证仅基于仿真，需进一步通过实验验证其在真实网络中的适用性。

---

## 165. Beyond Forecasting: Recasting Volatility Control as a Routing Problem

**arXiv ID:** 2608.10375 | [PDF](https://arxiv.org/pdf/2608.10375v1)

**作者:** Hongji Pu `[一作]` (University of Illinois Urbana Champaign), Leyang Zhou `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 VolRouter， 一个基于市场状态路由估计器–控制器对的波动率控制框架。

**💡 创新点**

创新点在于把波动率控制拆解为状态推断、是否切换评估和策略选取三步，并将路由器实现为规则/可学习/LLM 模块，显式地把策略选择加入控制流程。

**🔧 技术方法**

主要技术包括状态特征提取、路由决策器（规则/强化学习/LLM）、多策略库和持久切换机制；实验使用传统波动率估计器（RV、EWMA、GARCH 等）和多种控制器（目标波动率、趋势、尾部风险等）。

**📊 数据集**

使用四个场景的数据集：S&P 500（单资产波动率调节）、Multi‑Asset（多资产协方差控制）、Bitcoin（高波动数字资产）和 USDT（低波动稳定币）。

**📈 对比分析**

与固定规则、基于状态的固定控制、专家混合和情境赌博等基线对比，VolRouter 在 3/4 场景获得最高 Sharpe，提升 0.27~0.42，且往往能在不大幅提升原始收益的情况下显著降低最大回撤和 CVaR。

**⚠️ 局限性**

局限性包括：在波动率需求变化不大的市场（如 USDT）路由优势有限；性能对切换敏感度、交易成本和策略库规模敏感；过度频繁切换或库过大可能导致收益波动。

---

## 166. Do Time-Series Forecasters Use the Right History: Recoverability, Recovery, and Functional Use of Temporal Delays

**arXiv ID:** 2608.10433 | [PDF](https://arxiv.org/pdf/2608.10433v1)

**作者:** Qipeng Qian `[一作]` (Supcon Technology), Yuntao Qian `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

针对时序模型的延迟可恢复性、报告准确性和实际使用进行系统性审计，提出分层可恢复性度量，并证明即使模型报告正确且预测接近oracle，也可能未使用正确历史。

**💡 创新点**

创新在于将可恢复性、报告和功能使用分离为三链路，并引入输入条件下的贝叶斯可恢复性度量和路由机制来检验功能使用，揭示模型解释不可信的普遍性。

**🔧 技术方法**

使用输入条件贝叶斯可恢复性分析、profile分离、soft/hard路由控制、干预敏感性检验等技术。

**📊 数据集**

使用三种控制生成机制（P1点延迟、P2带输出记忆、P3有限核库）合成数据，以及真实北京空气质量和水利输入序列的半合成测试。

**📈 对比分析**

通过对TCN和N-HiTS两大神经网络骨干在同一数据集上的比较，评估结构误差、可恢复误差和预测MSE；结果显示结构误差与预测误差不一致，且在正确报告且接近oracle的情况下约60-90%实例未使用报告历史。

**⚠️ 局限性**

局限包括仅在单变量、已知输出机制的控制设置下验证；未涵盖多变量路径、预训练大模型、真实因果解释；路由改善未必提升预测精度，且实验以干预敏感性为准而非真实因果。

---

## 167. Automatic Field-of-View Adjustment for a View-Expansive Microscope via LSTM-Based Gaze and Pipette Motion Interpretation

**arXiv ID:** 2608.10401 | [PDF](https://arxiv.org/pdf/2608.10401v1)

**作者:** Kenta Yokoe `[一作]` (Nagoya University), Tadayoshi Aoyama `[通讯]` (Nagoya University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一套基于AI的视野（FOV）自动调节系统，并将其集成到视野扩展显微镜中，以提升ICSI手术的操作速度。

**💡 创新点**

创新点在于利用LSTM模型结合操作者的注视位置与吸管位置/速度，实时预测并自动调节最佳视野大小，消除了传统显微镜中需要手动更换物镜和调光的繁琐步骤。

**🔧 技术方法**

技术包括长短期记忆网络（LSTM）、屏幕眼动追踪（Tobii Pro Nano）、多视图显微技术（伺服镜、可调电镜片）以及实时图像显示控制。

**📊 数据集**

使用一位拥有五年以上ICSI经验的专家在实验室中完成的14次ICSI实验（共11520步）作为训练数据，另外3次实验（共2768步）作为验证数据。

**📈 对比分析**

通过与手动视野调整以及专家使用传统显微镜的对比，结果显示新系统使初学者的任务完成时间从平均60.5秒降至48.0秒，显著提升，且在统计上与专家使用传统显微镜的平均时间（约45秒）无显著差异，说明系统能够让新人达到与专家相当的操作速度。

**⚠️ 局限性**

局限性包括：仅以单一专家的数据训练，导致对不同操作者的视野偏好和注视模式的适配不足；眼动追踪器需要逐位调校且易漂移，影响系统的稳定性；模型为静态预训练，无法自动适应不同显微镜环境或多任务场景。

---

## 168. Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging

**arXiv ID:** 2608.10712 | [PDF](https://arxiv.org/pdf/2608.10712v1)

**作者:** Tim-Felix Fassch `[一作]`, Cyrill Stachniss `[通讯]` (University of Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对feed‑forward 3D Gaussian splatting生成的每像素高原理，使用基于显著性引导的超像素分割、编码器、合并器、细化器和多分辨率解码器，将原始稠密高斯压缩为内容自适应的精简表示。

**💡 创新点**

创新点在于：①用显著性引导的Bayesian自适应超像素实现空间一致且内容敏感的分组；②使用Set Transformer实现每组高斯的特征高斯编码；③跨视角基于几何重叠与特征相似度的学习式合并；④可控的层次细节解码器，允许在推理时按需平衡质量与效率；整体实现了与任意FF后端兼容、仅保留1/20原始高斯数的高效后处理。

**🔧 技术方法**

技术包括：Bayesian自适应超像素分割（BASS）+Shi‑Tomasi角点显著性；Set Transformer (SAB+PMA)编码/合并；多槽位（slot）解码器；kNN几何匹配与特征相似度门控；基于MSE/SSIM/LPIPS的光度损失；教师正则化和多样性正则；以及在线增量重建流程。

**📊 数据集**

使用DL3DV‑10K进行训练；评估基准为DL3DV‑Bench、MipNeRF360和Tanks & Temples；在线重建实验在MipNeRF360的多场景上进行。

**📈 对比分析**

与ReSplat、VolSplat、AnySplat的体素化后处理及基于时刻匹配的超像素聚合进行对比；在三大基准上，使用K=1解码器可获得与原始FF相当或更优的PSNR（约28–29 dB）、SSIM>0.90、LPIPS低，且原子数量仅为原始的1/20–1/10，显著提升渲染速度和内存效率。

**⚠️ 局限性**

局限性包括：依赖初始FF高斯的质量，若基础预测差则压缩效果受限；额外的编码/合并计算导致几毫秒的延迟；目前仅针对静态场景；在极细节纹理上略有失真；需要额外训练和超参数调优。

---

## 169. Efficiency Adjustments Break the Logarithmic Rank Barrier

**arXiv ID:** 2608.09984 | [PDF](https://arxiv.org/pdf/2608.09984v1)

**作者:** Josue Ortega `[一作]` (Queen's University Belfast), Gabriel Ziegler `[通讯]` (Freie Universität Berlin)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5128737219)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了学校分配中学生提议的稳定匹配算法（DA）在效率上的不足，并提出了效率调整后的DA（EADA）机制，证明其平均排名可实现对数对数（log log n）的上界，从而突破传统DA的对数（log n）瓶颈；

**💡 创新点**

创新点在于将EADA的闭合（closure）结构与概率分析相结合，首次给出EADA平均排名的渐近上界，并推广到所有弱支配DA的Pareto有效匹配，得到 O((log log n)^2) 的通用上界；

**🔧 技术方法**

采用序列化实现、闭合残余市场的概率上界、Pittel 对 DA 最差排名的估计、以及联合上界（union bound）和对顶k愤慨图的无环性分析等技术；

**📊 数据集**

使用完整随机的独立同分布（i.i.d.）学生偏好和学校优先级作为理论模型；仿真数据涉及 n=500~10,000 的随机市场，取 2,000 个样本平均；

**📈 对比分析**

与传统 DA 的对比显示，EADA 的平均排名显著低于 DA 的 log n 级别，仿真结果显示在可观测规模下几乎平稳增长，表明 EADA 在实际规模下表现优异；

**⚠️ 局限性**

主要限制在于上界的松弛，主要来自于对所有可能残余市场的联合上界以及对最差排名的粗略估计，导致得出的 log log n 量级可能不是最优；进一步研究需精确控制残余市场分布以获得更紧的下界。

---

## 170. Exploiting Structure in the Boolean Weighted Constraint Satisfaction Problem: A Constraint Composite Graph-Based Approach

**arXiv ID:** 2608.10005 | [PDF](https://arxiv.org/pdf/2608.10005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 171. Compositional Benchmark Synthesis for Hierarchical Human Action Recognition

**arXiv ID:** 2608.10765 | [PDF](https://arxiv.org/pdf/2608.10765v1)

**作者:** Farnaz Soleimani `[一作]` (University of Paris-Est Créteil), Ghazaleh Khodabandelou `[通讯]` (University of Paris-Est Créteil)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套基于现有单标签动作语料的四层层次意图基准合成与评估框架，生成跨主体、覆盖平衡的合成序列。

**💡 创新点**

创新点在于整合本体化层次结构、覆盖感知采样、反循环监督设计和组合held-out划分，使基准可再现且具备结构化通用性。

**🔧 技术方法**

采用转移模型、覆盖感知采样器、异构图表示、第一阶逻辑规则与多种基线模型（层次Transformer、序列Transformer、Bag-MLP、R-GCN）进行验证。

**📊 数据集**

基准基于NTU RGB+D 120动作数据集的预提取骨骼特征，结合自定义本体与生成器生成多级标签。

**📈 对比分析**

对四类基线进行宏F1评估，发现层次Transformer与R-GCN在意图层取得最高成绩，所有模型在组合held-out上平均下降0.13–0.17，且逻辑违例率表明不存在循环监督。

**⚠️ 局限性**

局限包括意图标签人为构造、时间序列仅弱约束、单一源数据集、以及受主体一致性约束导致的多样性限制。

---

## 172. BooST: Bridging Semantics and Motions for Efficient Skill Transfer

**arXiv ID:** 2608.10600 | [PDF](https://arxiv.org/pdf/2608.10600v1)

**作者:** Jusuk Lee `[一作]` (Seoul National University), H. Jin Kim `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并验证了一个两阶段框架BooST，先通过跨模态VQ‑VAE预训练统一的技能表示，再将其蒸馏成轻量级策略，实现少样本、高效跨任务、跨机器人本体的适应。

**💡 创新点**

将“语义意图”和“运动动力学”融合到同一离散代码表，既满足泛化、鲁棒、效率三重需求，又通过跨模态VQ‑VAE与动作重构实现技能表示的可蒸馏性。

**🔧 技术方法**

使用跨模态VQ‑VAE、CLIP视觉+语言交叉注意、位置编码、余弦注意、低维动作重构、变分推理、蒸馏以及BAKU式低层策略等技术。

**📊 数据集**

在76k条DROID机器人轨迹上预训练，随后在LIBERO benchmark（LIBERO‑90、Goal、Object、Spatial）及真实UR3机器人上评估，并通过增强的动态视觉干扰版LIBERO‑90测试鲁棒性。

**📈 对比分析**

与Diffusion Policy、VQ‑BeT、QueST、LISA、EXTRACT、LAPA、UniVLA等基线比较，BooST在所有数据量下均显著领先，尤其在10样本时提升140%；在真实机器人上仅需5样本即可取得高成功率，并在动态干扰环境下保持稳健。

**⚠️ 局限性**

受限于仅基于2D图像，z轴精度和深度感知受限；蒸馏后的策略不含CLIP，导致在大视角变化下3D理解能力不足，尚需进一步引入3D感知模块。

---

## 173. Can LLMs be Used to Simplify Algorithms? Simpler Algorithms for Vertex Coloring and Edge Connectivity

**arXiv ID:** 2608.10753 | [PDF](https://arxiv.org/pdf/2608.10753v1)

**作者:** Antoine El-Hayek `[一作]` (Institute of Science and Technology Austria), Da Wei Zheng `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大型语言模型在简化复杂算法方面的能力，并通过实验发现两种新的简化算法；

**💡 创新点**

首次展示LLM在算法简化中的有效性，提出更优的顶点着色列表大小改进和更简洁的全局最小割算法；

**🔧 技术方法**

使用多轮prompt与LLM（Claude、ChatGPT、Gemini）交互，结合随机化贪婪算法与极性分解技术；

**📊 数据集**

实验基于10个经典算法问题的论文PDF，无需外部数据集；

**📈 对比分析**

通过手工评估输出的六类（简化、已知简化、归约、无效、放弃、错误），发现Claude与ChatGPT在严格/非严格prompt下各得到1-2项创新，性能表现优于Gemini；

**⚠️ 局限性**

LLM输出仍可能错误或不完整，需专家核查；实验规模有限，结果仅为示例性展示。

---

## 174. OpenPM: Auditable Point-in-Time Evaluation for LLM Portfolio-Management Agents

**arXiv ID:** 2608.09988 | [PDF](https://arxiv.org/pdf/2608.09988v1)

**作者:** Xinying Cai `[一作]` (Rutgers University), Raymond Li `[通讯]` (University of British Columbia)

**通讯引用:** 6097 | [OpenAlex ID](https://openalex.org/A5009823475)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个针对大语言模型的可审计、点对点的股票组合管理基准 OpenPM，解决信息泄漏、执行成本过度乐观和自然语言风险指令未强制执行等问题；

**💡 创新点**

通过统一可用性门控、泄漏证书、成本敏感曲线和约束遵守报告，首次实现对 LLM 交易代理的完整可审计评估，并构造了一个分层分配器（analyst→constructor→critic）作为基准代理；

**🔧 技术方法**

使用了可用性门控数据管道、自然语言到结构化约束的解析、LLM 评估、确定性风险投影、交易模拟和完整审计日志；

**📊 数据集**

采用了冻结的 2026 年 S&P 500 股票池（508 支），共计 3,432 个 5 分钟行情，结合新闻、宏观、事件和文件等多源数据；

**📈 对比分析**

通过与等权重、基于组合排名、随机权重以及 SPY/现金基准的对比实验，发现当上层分析信号强时 LLM 构造器能在单窗口内略超等权重；但总体收益仍受交易成本和高换手率影响；

**⚠️ 局限性**

局限性包括：仅为单一窗口、无市场冲击的上限收益；理想化的成交模拟（单一 IEX 价格、无滑点）；仅覆盖美国大盘股票；未检验跨周期、跨市场的稳健性；以及仅提供相对排序而非统计显著的 alpha。

---

## 175. Multiway $f$-Cut is fixed-parameter tractable

**arXiv ID:** 2608.10380 | [PDF](https://arxiv.org/pdf/2608.10380v1)

**作者:** Tony Huynh `[一作]` (Institute for Basic Science), Marek Sokołowski `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文研究了多路f-切割问题，证明了该问题在预算k的参数化下是固定参数可解的。

**💡 创新点**

创新点在于提供了一个简单的证明，表明多路f-切割问题在一般的连接函数下也是固定参数可解的，扩展了之前仅在图的情况下的结果。

**🔧 技术方法**

使用了子模函数最小化作为黑箱技术，并结合了分支限界法来解决问题。

**📊 数据集**

使用了有限集合E上的连接函数，具体数据集未明确提及，但涉及到多个终端的情况。

**📈 对比分析**

与现有方法相比，提出的方法在时间复杂度上为O(k^k+2log(k+1)n^6(γ+n))，在处理复杂度上表现出色，尤其是在预算k的参数化下。

**⚠️ 局限性**

限制在于该方法依赖于子模函数的最小化，且在处理大规模数据时可能会面临效率问题。

---

## 176. Invertible Logits Transformation for Accuracy-Preserving Post-Hoc Uncertainty Calibration

**arXiv ID:** 2608.10372 | [PDF](https://arxiv.org/pdf/2608.10372v1)

**作者:** Lening Zhao `[一作]` (University of Pennsylvania), Li Shen `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种后置校准方法InvLT，利用共享标量MLP对预softmax logits进行非线性变换，并通过软重构正则保证单调性，从而在不改变原模型预测的前提下提升概率校准效果。

**💡 创新点**

创新点包括：①通过辅助逆网络的重构正则实现软单调性约束，避免UMNN等方法的数值积分与高计算开销；②使用共享的标量网络，使参数量与类别数无关，便于大类别场景；③在保持准确率的同时实现更高的校准精度。

**🔧 技术方法**

使用技术包括：标量MLP、逆网络重构正则、负对数似然/贝雷尔损失、温度缩放与多种基线校准方法（TS、PTS、UMNN、Dirichlet等）进行对比；实验中采用GPU/CPU训练与推理时间测评。

**📊 数据集**

实验数据集涵盖CIFAR‑10、CIFAR‑100以及ImageNet，使用多种模型架构（ResNet‑50/152、VGG‑16/19、DenseNet‑121、WideResNet、ViT‑B/16）进行评估。

**📈 对比分析**

通过ECE、AdaECE、KDE‑ECE和NLL等指标与多种基线方法比较，InvLT在所有数据集和架构上均取得最低ECE和NLL，保持原始准确率；在训练时间上比UMNN快约3.5倍，推理时间快约5倍。

**⚠️ 局限性**

局限性包括：仅做元素级变换，无法捕获跨类别误差；软单调性约束不保证严格的argmax保持；对极少校准样本可能不稳定；在高安全性场景仍需额外的argmax一致性检查。

---

## 177. Enhancing Reliability of Symbolic Execution Tools for Smart Contract Analysis through Rule-Based False Positive Reduction

**arXiv ID:** 2608.10265 | [PDF](https://arxiv.org/pdf/2608.10265v1)

**作者:** Muhammad Ali Hassan Ahmad `[一作]` (Lakehead University), Affan Rauf `[通讯]` (National University of Computer and Emerging Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对以太坊智能合约符号执行工具 Mythril 的高误报率，提出并实现了一套基于规则的误报过滤方法，显著提升工具的精确度。

**💡 创新点**

创新点在于将针对六类关键漏洞的误报消除规则直接嵌入 Mythril 的符号执行流程中，并提供了通用的规则设计思路，易于迁移至其他工具。

**🔧 技术方法**

核心技术包括符号执行、路径可行性求解、数据污点跟踪、调用类型区分和状态影响检测，配合自定义约束与规则来筛选误报。

**📊 数据集**

使用了 Gigahorse 基准集，其中包含 100 个已知漏洞合约和 40 个无漏洞合约，作为地面真相标签进行评测。

**📈 对比分析**

通过对比原始 Mythril 与改进版本在 Gigahorse 上的误报、真报、漏报等指标，实验显示误报率降低 50%（无漏洞集）和 89.2%（漏洞集），精度提升至 48.1%，召回率从 75.0% 提升至 81.2%，F1 分数从 0.152 提升至 0.605。

**⚠️ 局限性**

主要局限在于整数溢出/下溢类漏洞的误报仍高（未完全消除），实验仅覆盖 Mythril 与 Gigahorse，未验证在其他工具或更大数据集上的泛化效果。

---

## 178. Locally Deployable Small Language Models for Emergency Department Decision Support: A Systematic Benchmark of Fine-Tuning Strategies

**arXiv ID:** 2608.10273 | [PDF](https://arxiv.org/pdf/2608.10273v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 179. Evaluation-Conditioned Training: Teaching Models to Generalize to Stronger Oversight Regimes

**arXiv ID:** 2608.10209 | [PDF](https://arxiv.org/pdf/2608.10209v1)

**作者:** Alec Harris `[一作]` (AI Safety Initiative at Georgia Tech), Yixiong Hao `[通讯]` (AI Safety Initiative at Georgia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出并验证了一种后训练框架Evaluation‑Conditioned Training (ECT)，通过在训练时将自然语言评估者描述作为条件，来引导大型语言模型（LLM）更好地符合人类期望的行为；

**💡 创新点**

创新点在于将评估者的详细描述嵌入训练与部署阶段，利用评估标签纠正奖励误规范，从而在面对不完美反馈时仍能逼近理想目标；

**🔧 技术方法**

技术上采用了LoRA/QLoRA适配器、SFT和PPO等后训练方法，并在训练样本中加入评估标签作为额外输入；

**📊 数据集**

实验数据集包括Anthropic的政治偏见评估集（60类、9任务）用于政治话题的公平性实验，以及人工合成的一位数加法问题集用于评估sycophancy的控制；

**📈 对比分析**

与无标签基线、随机标签基线对比，ECT在政治公平性实验中提升了约15个百分点的“平衡度”，拒绝率下降；在算术实验中错误正例率从约48%降至21%，准确率提升至约74%；

**⚠️ 局限性**

局限性包括仅在单轮任务上验证、使用模型自评估器而非人工标注、样本规模有限、对复杂多步或真实世界情境的泛化尚未检验。

---

## 180. Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases

**arXiv ID:** 2608.10503 | [PDF](https://arxiv.org/pdf/2608.10503v1)

**作者:** Davood Wadi `[一作]` (McGill University), Matthew Philp `[通讯]` (Toronto Metropolitan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于完全交叉因子设计、使用精确Token概率分布进行行为评估的框架，用以系统地评估大语言模型的潜在偏差。

**💡 创新点**

创新点在于：①将心理测量学方法与LLM机制结合；②采用精确的token级概率质量函数（PMF）取代传统采样；③提出多变量序数一致性度量和分布式ANOVA，能准确分离主效应与交互效应。

**🔧 技术方法**

核心技术包括：完全交叉因子实验设计、精确提取下一个token的PMF、约束层（格式化失败率校正）、一致性层（序数一致性度量）以及构造层（离散卷积与分布式Hoeffding分解）。

**📊 数据集**

使用的数据集为消费者民族中心主义倾向量表（CETSCALE，17个Likert项目），在5款模型与4个目标国家的5×4完全实验设计中共产生340个PMF。

**📈 对比分析**

通过与传统聚合基准和基于采样的评估对比，发现精确PMF方法能更稳定地识别主效应和交互效应，降低采样误差导致的符号翻转概率；在案例研究中成功分离出各模型的国家偏好和交互偏差，展示了更高的分析精度。

**⚠️ 局限性**

局限性包括：仅在单一心理构念（民族中心主义）上验证；仅处理单Token响应的序数尺度；需对模型暴露下一个token概率的API支持；仅在英语实验，未检验跨语言效果；结果为一次性快照，可能随模型更新而变化。

---

## 181. Do Judges Behave Like Algorithms?

**arXiv ID:** 2608.10400 | [PDF](https://arxiv.org/pdf/2608.10400v1)

**作者:** Riya Manchanda `[一作]` (Duke University), Songman Kang `[通讯]` (Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了德克萨斯州哈里斯县的治安预审保释听证中法官的决策是否遵循可解释的算法规则，并分析了法官之间的差异与一致性。

**💡 创新点**

创新点在于提出“Rashomon集合”方法评估每位法官决策可被多个高效模型解释的程度；识别“不可解释案件”并归因于数据错误或外部信息；以及通过规则挖掘揭示法官之间的差异点。

**🔧 技术方法**

采用可解释决策树（GOSDT、TreeFARMS）、特征重要性分析、Bootstrap、规则挖掘（FP‑Growth）以及三元组一致性评估等机器学习技术。

**📊 数据集**

使用哈里斯县2020-2025年约22,000起治安案件的公开司法数据，包含被告特征、案件属性、法官决策与保释结果等。

**📈 对比分析**

与传统单一“最佳”模型对比，通过Rashomon集合展示大部分法官可被准确度≥85%的浅层树解释；在法官间的交叉预测中，误差显著升高，显示高度不一致；三元组实验表明法官自我预测胜过互相预测，平均一致率约60–70%，说明法官决策依赖个人算法。

**⚠️ 局限性**

主要局限包括：数据缺失和错误导致不可解释案件比例高达18%；外部司法信息（如跨县犯罪记录）未纳入模型；研究仅关注预审保释，难以推广至其他司法环节；以及未评估模型对长期结果（如再犯率）的预测效果。

---

## 182. A Simple Algorithm for Best Separable State

**arXiv ID:** 2608.10147 | [PDF](https://arxiv.org/pdf/2608.10147v1)

**作者:** Prashanti Anderson `[一作]` (Massachusetts Institute of Technology), Amit Rajaraman `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了最佳可分状态问题（BSS），旨在最大化量子测量在非纠缠状态下的接受概率。

**💡 创新点**

提出了一种更简单的SoS松弛的舍入方法，改进了现有算法的运行时间，并引入了一种新的“固定引理”。

**🔧 技术方法**

使用了全局相关舍入技术，结合SoS松弛方法。

**📊 数据集**

未具体提及使用的数据集，但涉及到的量子测量和状态是理论上的。

**📈 对比分析**

与现有算法相比，提出的算法在1-ϵ的情况下运行时间为n^O(√(n/ϵ))，在q/n的情况下为n^O(√(q))，显著优于之前的算法。

**⚠️ 局限性**

算法的限制在于仍然存在许多开放问题，例如能否将运行时间中的√(n)减少到O(log n)。

---

## 183. ENTLORE: A Graph-Grounded Benchmark for Latent Organizational Reasoning in Enterprise Question Answering

**arXiv ID:** 2608.10679 | [PDF](https://arxiv.org/pdf/2608.10679v1)

**作者:** Akrin Zheng `[一作]` (ScitiX.ai), Alaia Liu `[通讯]` (ScitiX.ai)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个新的企业问答基准，基于真实企业内部文档与组织表，生成审核后的真值图和匿名发布文档，并将问题映射为可执行图程序，评估系统在显式检索、跨源组合和隐式组织推理上的性能。

**💡 创新点**

通过审核后的真值图保证答案的唯一性与可证明性；将问题与答案映射为图程序，能够检验系统是否恢复隐式组织关系；发布匿名数据时保留跨源结构，避免泄露隐私；将基准划分为L1/L2/L3三级，区分显式检索与隐式推理。

**🔧 技术方法**

图构建与图规则推理、语义化证据检索、检索增强生成（RAG）、Agentic retrieval、LLM Wiki、GraphRAG 等模型与技术。

**📊 数据集**

来自某企业的多类型源（项目、模块、客户等）内部文档与组织表，构成约数千条文档和约600个问题。

**📈 对比分析**

通过对比闭书、BM25、Flat RAG、Agentic、LLM Wiki、GraphRAG、黄金文档代理Ω等访问范式与多种模型，L1/L2 问题准确率约为60%，L3 仅约35%；即使给定黄金文档，仍有30% L3 问题未解。

**⚠️ 局限性**

仍难以完全捕捉隐式组织关系的推理，基准对细粒度排名不稳定，且仅在单一企业场景验证，缺乏跨企业泛化能力。

---

## 184. CurveFP: Rational-Radix Logarithmic Datatypes with Closed Products for Language Models

**arXiv ID:** 2608.10010 | [PDF](https://arxiv.org/pdf/2608.10010v1)

**作者:** Ye Qiao `[一作]` (University of California, Irvine), Ye Qiao `[通讯]` (University of California, Irvine)

**通讯引用:** 123 | [OpenAlex ID](https://openalex.org/A5051829685)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 CurveFP，一种基于块缩放的闭合乘积低精度数据类型，可在语言模型训练中实现 FP8 级别质量，在推理时以 7 位整数实现高效乘法。

**💡 创新点**

创新点包括：将量化幅值分布在交错对数曲线上，曲线索引实现乘积闭合；利用有理基数调节动态范围与局部分辨率；给出相位计数公式与累加规则，消除乘法器，实现结构化乘法和固定相位累加。

**🔧 技术方法**

技术手段包括低精度量化、块级缩放、闭合乘积代数、相位分段累加以及对比实验；使用了多种 LLM（Llama‑3、Qwen3、Falcon‑H1、Qwen3.5）以及标准 FP16/FP8、BF16 基线进行评估。

**📊 数据集**

使用的数据集包括 WikiText‑2、FineWeb‑Edu、WikiText‑103、PG‑19、HellaSwag、PIQA、ARC、WinoGrande、LAMBADA、BLiMP 等。

**📈 对比分析**

通过 Perplexity、NMSE、相位误差等指标与 FP8/FP16/BF16 进行对比；在四大 7B–9B 模型上 CurveFP 仅比 FP8 少 1 位仍保持 ≤1.32% PPL；训练时 NMSE 下降约 10%；三种格式的 3B‑token 预训练均无发散，PPL 仅差 0.01；在 OOD/下游任务中与 BF16 接近，优势有限。

**⚠️ 局限性**

局限性包括：对块缩放粒度敏感，尺度选择影响性能；有理基数与相位数的权衡导致实现复杂度上升；在某些下游任务和 PG‑19 中未能显著提升，仍停留在 FP8 级别而非明显超越。

---

## 185. Deciding When to Rely on Visual Information: Gated Multimodal Fusion in Sequential Recommendation

**arXiv ID:** 2608.10700 | [PDF](https://arxiv.org/pdf/2608.10700v1)

**作者:** Natalija Glisovic `[一作]` (IKEA Retail (Ingka Group)), Martin Tegner `[通讯]` (IKEA Retail (Ingka Group))

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 VisGate 框架，在多模态序列推荐中通过可学习的门控机制，依据项目嵌入与用户序列上下文自适应融合视觉与协同信息。

**💡 创新点**

将视觉效用视为潜在上下文变量，采用基于共现的对比学习学习视觉投影，并在门控中结合用户序列上下文，实现按项目层级的自适应融合。

**🔧 技术方法**

使用 Transformer（BERT4Rec）+ MLP 门控 + 对比学习 + 层归一化 + 交叉熵 + 门控正则化等技术。

**📊 数据集**

实验基于 Amazon Scientific、Amazon Video Games、Amazon Health & Household 三个公开数据集以及 IKEA 私有数据集。

**📈 对比分析**

与 CF、视觉感知、图神经网络和现有多模态序列推荐模型比较，Hit@10/NDCG@10 在所有数据集上均显著提升，尤其在稀疏数据集上提升 15–20%。

**⚠️ 局限性**

门控在视觉独特但交互稀疏的项目上易失真，需交互阈值校正；同时模型仍采用冻结的视觉投影，无法端到端联合优化。

---

## 186. Interpreting Language Model Hidden States at Scale

**arXiv ID:** 2608.10260 | [PDF](https://arxiv.org/pdf/2608.10260v1)

**作者:** Jordan Pettyjohn `[一作]` (University of Chicago), Ian Foster `[通讯]` (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 OmniLens，一种能够在任意宽度激活（残差、注意力、MLP）上训练的通用可解释镜头（lens）框架，支持在 LLaMA‑3.3‑70B 及 LLaMA‑3.1‑405B 等前沿规模模型上实现密集覆盖。

**💡 创新点**

创新点包括：① 低秩翻译器（rank‑r）将每个镜头的参数从 O(d²) 降到 O(rd)，使得密集覆盖在前沿规模可行；② Subset‑KL 目标通过仅计算词表子集的 KL，显著降低显存和计算开销；③ 统一的镜头架构可同时应用于残差、注意力和 MLP 激活，打破以往镜头只针对单一组件的专业化限制。

**🔧 技术方法**

使用的技术包括：低秩参数化（LoRA‑style 低秩翻译器）、Subset‑KL 目标（Top‑k 截断与带重要性采样的尾部估计）、CUDA fused kernel 以高效采样词表子集、分布式训练与梯度检查点、Hookbox 进行模型激活钩子配置。

**📊 数据集**

主要使用的数据集有：LLaMA‑3.3‑70B、LLaMA‑3.1‑405B‑Instruct、GPT‑2 Small、GPT‑2 8B；在解释性任务中还使用了 Prompt‑Injection 数据集、2WikiMultiHop（多跳推理）和毒性检测数据集（如 GPT‑2 toxic prompts）。

**📈 对比分析**

与以往单组件镜头（如 tuned‑lens、attention‑lens）对比，OmniLens 在同等层级覆盖下参数量减少 90% 以上、峰值显存降低 70% 以上；在 Prompt‑Injection、Multi‑Hop Reasoning 与 Toxicity Localization 等三项案例研究中，其检测性能与先前结果一致或略优，且实现了模型全局的行为定位与干预评估。

**⚠️ 局限性**

局限性包括：仅针对宽度等于 d 的组件；低秩翻译器虽有效但在极低秩时会牺牲部分精度；Subset‑KL 的 Top‑k 方案有偏，重要性采样方案对采样质量敏感；目前未针对单头注意力或更窄组件的映射；在 70B 以上规模的完全收敛评估尚未完成。

---

## 187. UniMod: Enhancing Multi-Modal Medical Diagnosis through Cross-Modality and Within-Modality Alignment

**arXiv ID:** 2608.10316 | [PDF](https://arxiv.org/pdf/2608.10316v1)

**作者:** Zijian Gu `[一作]` (University of Central Florida), Song Wang `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 UniMod 框架，在多模态医学诊断中通过分别监督图像、文本和融合预测来抑制捷径学习。

**💡 创新点**

创新点在于：① 强制每种模态独立做出诊断预测；② 同时使用跨模态对齐与同模态对比学习构建语义一致的表征；③ 通过 GradNorm 动态平衡多任务损失；④ 采用 LoRA 进行参数高效微调。

**🔧 技术方法**

使用的技术包括：多模态视觉‑语言模型 InternVL2.5‑8B、定制模态隔离注意力掩码、图像‑文本 MSE 对齐、监督对比学习、GradNorm 损失加权、LoRA 参数高效微调。

**📊 数据集**

实验数据集为 Harvard‑Glaucoma（视网膜照片＋临床记录）和 CheXpert Plus（胸 X‑ray + 放射报告），并在 CheXpert Plus 上扩展至 5 类多标签任务。

**📈 对比分析**

与零样本、单模态、标准多模态 LoRA、OGM‑GE、G‑Blend、CGGM 等基线相比，UniMod 在 Harvard‑Glaucoma 上取得 0.850 AUC（比 OGM‑GE/G‑Blend 提升 1.6–1.8%），在 CheXpert Plus 上取得 0.966 AUC（比对手提升 5%+），并显著平衡两模态的依赖。

**⚠️ 局限性**

局限性包括：仅在两家单机构数据集上验证；缺乏跨机构泛化评估；只考察图像与文本两模态，未涉及 CT/MRI、心电等；未扩展至分割或密集预测等任务。

---

## 188. Beyond Decision Boundaries: Relational Geometry Attacks on Contrastive Embedding Manifolds

**arXiv ID:** 2608.10237 | [PDF](https://arxiv.org/pdf/2608.10237v1)

**作者:** Fei Zhao `[一作]` (University of Alabama at Birmingham), Nitesh Saxena `[通讯]` (Texas A&M University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种针对对比学习与Siamese嵌入模型的几何感知对抗攻击框架，通过学习生成器在离线阶段学习全局几何变形模式，随后一次前向传播即可在在线阶段实时生成对抗扰动，系统性破坏嵌入空间的正负样本相对几何关系，导致验证性能严重下降。

**💡 创新点**

创新点在于将对抗攻击目标从局部分类边界迁移到全局嵌入几何层面，构建基于U-Net的生成器学习全局几何腐蚀模式；将迭代优化迁移到离线阶段实现实时攻击；对比传统PGD、NES等攻击显示在几何腐蚀度、准确率下降与正负logit差距方面表现更强。

**🔧 技术方法**

使用U-Net编码-解码生成器、对比损失、温度缩放的余弦相似度、L2/∞ 约束投影、正负样本拉/推目标函数、正向方向正则化。

**📊 数据集**

主要使用MarkMatch同手投票标记对齐数据集（51×51 RGB图像对），以及CEDAR签名验证数据集。

**📈 对比分析**

与Diff‑PGD、ZO/NES、Surrogate Transfer等基线对比，攻击后准确率从0.954降至0.386，Gap Reduction达到15.170，正负logit差距完全倒置，显示相对更强的全局几何破坏效果。

**⚠️ 局限性**

局限性包括：对抗扰动仍受像素级约束影响，模型对特定任务/数据集的迁移性尚未充分验证；缺乏针对主动防御（如随机变换）下的稳健性研究；对生成器在不同嵌入尺度与复杂度的适应性尚待探讨。

---

## 189. Beyond Cash Flows: A Multi-Agent AI Framework for Valuing Clinical-Stage, Cross-Border Biotechnology

**arXiv ID:** 2608.10175 | [PDF](https://arxiv.org/pdf/2608.10175v1)

**作者:** Yuhan Fang `[一作]` `[通讯]` (CPC Scientific Inc.), Yuhan Fang (CPC Scientific Inc.)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套专门针对临床阶段生物技术公司（无营收、价值以二元科学与监管事件为主）的多代理投资框架，涵盖价值层、跨市场协调层和冲突融合层。

**💡 创新点**

创新点在于：①将科学与临床判断（机制、试验结果、监管路径）直接转化为可辩护的估值；②在多市场环境下同步调和价格、流动性与汇率差异；③引入针对生物技术特有的冲突类型感知融合机制，系统化处理科学乐观与监管保守之间的对立。

**🔧 技术方法**

技术手段包括：基于大型语言模型的多代理协作架构、结构化辩论与迭代反馈、事件驱动的风险调整净现值（rNPV）模型、跨市场代理实现多国定价对齐以及冲突类型识别与加权融合算法。

**📊 数据集**

使用的数据主要来自公开的临床试验数据库、机制和监管路径信息、竞争格局分析、以及A股、港股和美股的交易价格与流动性数据；论文未公开具体的数据集或参数。

**📈 对比分析**

该框架未在实证或基准比较中提供性能指标；作者仅引用自己在2019‑2021年期间管理的跨境生物技术基金的历史业绩（127.17% 对比基准 50.67%）作为方法可行性的经验佐证。

**⚠️ 局限性**

限制包括：仅为架构设计，缺乏实现细节、参数调优和大规模实测；无法直接验证 AI 系统在真实交易环境中的效果；依赖人工经验的“Glocal”方法，若迁移到全自动系统可能面临可复制性与监管合规挑战。

---

## 190. What We Know about Responsible AI Practices in Industry: A Half Decade of Empirical Research

**arXiv ID:** 2608.10431 | [PDF](https://arxiv.org/pdf/2608.10431v1)

**作者:** Wesley Hanwen Deng `[一作]` (Microsoft Research), Solon Barocas `[通讯]` (Microsoft Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述并整合了161篇关于行业负责 AI（RAI）实践的实证研究，形成了行业 RAI 现状与挑战的完整景观。

**💡 创新点**

创新点在于将跨学科、跨方法的实证研究聚合为一个统一的知识体系，并公开了一个包含所有被综述论文的开放数据库，为后续研究提供了聚合资源。

**🔧 技术方法**

采用系统文献综述方法：关键词检索、雪球式引用追踪、主题编码与归纳分析，以及对研究方法与结果的结构化注释。

**📊 数据集**

使用的数据集为161篇选定的行业 RAI 实证研究论文的元数据与研究结果。

**📈 对比分析**

通过对已有研究的发现进行定性归纳与比较，描述了行业 RAI 进展与挑战的趋势，并未进行算法层面的性能评估；比较基于研究主题、方法、时间和地区分布等维度。

**⚠️ 局限性**

局限性包括：研究主要聚焦于西方技术公司，缺乏对生成式 AI 等新兴领域的覆盖；依赖已发表研究的质量与可访问性；未对实证研究进行纵向追踪，难以把握实践随时间演变的细节。

---

## 191. Conversational versus Dashboard Explainable AI for UAV Intrusion Detection: An Empirical Study of Operator Trust and Reliance

**arXiv ID:** 2608.10434 | [PDF](https://arxiv.org/pdf/2608.10434v1)

**作者:** Cong Chi Nguyen `[一作]` (Phenikaa University), Thien Van Luong `[通讯]` (National Economics University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对无人机入侵检测系统的可解释人工智能界面，设计并对比了基于大型语言模型的对话式XAI和传统仪表盘XAI两种交互方式，评估其对操作者理解、信任与依赖的影响。

**💡 创新点**

创新点在于首次将LLM驱动的对话式XAI应用于高维多模态无人机安全场景，支持对全局解释、局部归因、对比反事实和何如分析的自然语言交互，并与传统仪表盘进行系统化对比实验。

**🔧 技术方法**

使用的技术包括XGBoost入侵检测模型、TreeSHAP特征归因、Partial Dependence Plot、MACE对比反事实、What-If分析以及LLama大型语言模型实现对话接口。

**📊 数据集**

实验基于UAV-ID数据集，共42,258条样本，经过7:3训练/测试拆分后训练XGBoost模型。

**📈 对比分析**

通过57名具备AI经验的参与者随机分配到无解释、仪表盘XAI和对话式XAI三组，评估理解、实用性、信任和依赖等指标，结果显示对话式界面更易被接受但导致过度依赖，决策准确率未明显提升；仪表盘在适当自我依赖方面更优。

**⚠️ 局限性**

局限性包括对话式接口可能诱发过度依赖；实验仅在单一数据集和有限任务范围内进行，未加入不确定性提示或认知强制机制；样本规模与任务多样性有限。

---

## 192. MAP-Graph: Provenance-Aware Shared Memory for Multi-Agent Workflows

**arXiv ID:** 2608.10509 | [PDF](https://arxiv.org/pdf/2608.10509v1)

**作者:** Yiqi Wang `[一作]`, Taotao Cai `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MAP-Graph，一种基于语义检索与证明链的共享内存治理层，用于多代理长流程任务中信息共享与行动安全的决策；

**💡 创新点**

创新点在于将检索与授权分离，采用递归祖先追溯、权限过滤、路径信任多层次重排序以及行动风险门控，实现对私有、受污染或被撤销记录的动态审计与控制；

**🔧 技术方法**

技术包括构建任务本地的有向多类型执行图、利用向量检索计算语义相似度、从图中递归计算路径信任值、权限判定与多级门控规则，以及将受影响的证明链标记用于审计；

**📊 数据集**

使用合成的 2,700 条任务数据，涵盖企业工作流、软件工程和科研协助三大领域，包含 6 个实验组（干净、毒化、私有、权限撤销、行动风险敏感、压缩/开销）和 540 条子集进行模型迁移测试；

**📈 对比分析**

与七个基线（无内存、共享/隔离向量内存、改编的 G‑Memory、协作内存、MemLineage、平面 Provenance 等）以及三个大型模型（Qwen、GLM、Llama）比较，MAP‑Graph 在整体任务成功率（94.96%）和精确决策率（72.70%）上领先，且在安全性指标（ASR、泄漏、撤销违规）上实现 0%，仅在高风险行动门控阈值下略有剩余误判；

**⚠️ 局限性**

局限性包括合成数据的模板化、单轮四代理设置、缺乏跨会话累积与部署规模评估、以及门控阈值对高风险行动的敏感性，未来需在更真实、多轮、跨会话环境中检验。

---

## 193. From Prediction to Incrementality: Causal Optimization for Large-Scale Targeting and Recommendation

**arXiv ID:** 2608.10182 | [PDF](https://arxiv.org/pdf/2608.10182v1)

**作者:** Changshuai Wei `[一作]` (LinkedIn), Benjamin Zelditch `[通讯]` (LinkedIn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个决策中心的框架，统一将因果效应估计、探索学习和受限线性规划结合，用于大规模营销目标和推荐的增量优化。

**💡 创新点**

创新点在于：①将 Transformer 与 DragonNet 结合，形成可处理时序多结局的因果网络；②使用线性化拉普拉斯采样的 Bayesian 神经 bandit，显式探索增量价值；③采用基于 dual decomposition 的大规模 LP 求解器，实现全局预算/频次约束的可扩展分配。

**🔧 技术方法**

使用技术包括 Transformer 编码、Transformer‑augmented DragonNet（因果预测）、线性化拉普拉斯 Thompson Sampling、贝叶斯神经 bandit、双重分解（dual‑decomposition）线性规划、自动机架构与实时控制器。

**📊 数据集**

数据集涵盖：公开的 Open Bandit Dataset（用于离线模拟）和 LinkedIn Feed 营销流量（用于 8 周线上 A/B 测试）。

**📈 对比分析**

方法通过与传统基于预测（propensity）与仅排名（ranking）的方案对比，在线下单轮与多轮实验中显著优于对手；在上线 A/B 测试中，系统整体提升长期价值指标 7.20%（p=0.041）。

**⚠️ 局限性**

局限性包括：需要精细构造因果训练数据、对交付/成本估计的依赖、对大规模约束求解器的实现复杂度、以及对真实环境中非平稳与多目标约束的适用性需进一步验证。

---

## 194. Cross-View Sequential Visual Localization with Spatio-Temporal Context Modeling for Autonomous Driving

**arXiv ID:** 2608.10660 | [PDF](https://arxiv.org/pdf/2608.10660v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 195. Co-Lecturing With the DED: Explaining Circuit Design via the Draw Encode Display Loop

**arXiv ID:** 2608.09945 | [PDF](https://arxiv.org/pdf/2608.09945v1)

**作者:** Alasdair Lambert `[一作]` (University of Strathclyde), Conor Mc Bride `[通讯]` (University of Strathclyde)

**通讯引用:** 2947 | [OpenAlex ID](https://openalex.org/A5090954120)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在数字电路教学中提出并实施了 Draw‑Encode‑Display (DED) 循环与协同授课模式，结合可视化手绘与 Syrup 语言代码同步演示。

**💡 创新点**

创新点在于将视觉直观的电路图与可测试的硬件描述语言桥接，并通过协同授课和实时编码实现双向认知；同时提出 Syrup 语言及其显示功能。

**🔧 技术方法**

采用手绘图、Syrup 语言、协同授课、实时演示及显示渲染技术。

**📊 数据集**

使用问卷调查数据（45 份有效问卷）评估学生对 DED 循环与协同授课的接受度。

**📈 对比分析**

通过 Likert 量表问卷对学生满意度进行比较，结果显示 95% 以上学生对 DED 循环、协同授课及颜色使用持积极态度，未与传统代码优先方式进行量化性能对比。

**⚠️ 局限性**

局限性包括样本量有限、可能存在自选偏差、仅评估主观感受而未进行控制实验，且未提供客观学习效果的量化指标。

---

## 196. ProTAGAD: A Foundation Model for TAG Anomaly Detection with Decoupled Topological and Textual Prototypes

**arXiv ID:** 2608.10699 | [PDF](https://arxiv.org/pdf/2608.10699v1)

**作者:** Ziyan Wang `[一作]` (Yunnan University), Xin Jin `[通讯]` (Yunnan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ProTAGAD，一种针对文本属性图（TAG）的零样本异常检测基础模型；

**💡 创新点**

核心创新在于将文本语义与拓扑结构的学习完全解耦，分别构建文本与结构两套原型库，仅在最终得分层面融合，显著缓解传统耦合方式导致的模糊异常边界（BAB）问题；

**🔧 技术方法**

技术包括文本编码器+自监督概率估计器、Agent-Skills-Review-Confidence机制生成伪标签、对齐损失强化文本原型区分度、图变换器+知识蒸馏得到两种结构表示、K-means生成结构原型、KL对齐保证两种表示对结构原型的一致性，并通过标准化组合文本与结构异常得分；

**📊 数据集**

在14个跨域文本属性图数据集（Cora、Citeseer、Pubmed、Arxiv、History、Children、Grocery、Movies、Toys、Fitness、Products、Cornell、Texas、WikiCS）上评估；

**📈 对比分析**

与18个基线（10个传统GAD、6个跨域GGAD、2个TAGAD）以及与耦合版ProTAGAD对比，ProTAGAD在8个未见目标图中7个获得最高AUROC，平均排名1.12，平均提升幅度约17%；

**⚠️ 局限性**

局限性包括对极小图（如Cornell）原型估计不稳、需要预先使用LLM生成Agent伪标签且不支持在线微调、以及在极大图上推理显存较高。

---

## 197. Curate Before You Connect: Identity and Ontology Tagging in a Production Knowledge Graph

**arXiv ID:** 2608.10644 | [PDF](https://arxiv.org/pdf/2608.10644v1)

**作者:** Vaibhav Dangaich `[一作]`, Kundeshwar Pundalik `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并实现了从文档抽取流到知识图谱的 ingestion 与 ontology‑tagging 层，解决了实体去重、属性合并、关系写入、嵌入生成等关键步骤，并提供了人机协作的修订队列。

**💡 创新点**

创新点包括：① 采用多层 “record‑identity ladder” 通过标识符、名称、显示名、位置顺序依权重决定同一实体；② 将命名与类型证据区分开来，防止“名词误认为类型”；③ 对边与关系使用内容地址式身份，消除文件作用域的重复；④ 仅将实体合并标记为 flag，避免不可恢复的错误；⑤ 设计了基于规则+模型的 triage agent，对提案队列进行预过滤、聚类和验证，实现可回滚的自动写入。

**🔧 技术方法**

使用技术包括 Neo4j（图数据库）与自定义 schema；字符串标准化与正则；多类标签匹配与子类深度比较；相似度聚类（cosine ≥0.90）；嵌入向量生成与批量 backfill；规则引擎 + 轻量级机器学习（模型推断）用于提案评估；审计记录与 idempotent 写入；性能优化如批量写入、并发控制。

**📊 数据集**

主要数据集为 98,795 份马哈拉施特拉邦政府决议文档（公开），产生 537,157 个实体、2,198,567 条关系；补充使用 98,795 份军队文档做表格抽取验证；还有小规模评估语料（约 2,500 条样本）用于 evidence‑rule 评估。

**📈 对比分析**

通过统计指标评估：实体类型覆盖率 94.3%，关系谓词覆盖 51.9%；提案队列 48,403 条未处理，已完成 775 条人类决策；关系级 conformance debt 约 24k 次未定义谓词、22k 次域/值违规。写入性能从 6+ 小时（单条）降低到 40–45 分钟；批量 embedding 后仍有 7,000 条未嵌入。triage agent 预估可将 12,000 条提案压缩为 1–2%（5–15 倍）进行人工审查。

**⚠️ 局限性**

局限性：① identity ladder 未做精确评估（缺乏手工标注对照）；② evidence‑rule 只在特定格式文档验证，通用性未知；③ 关系推断仍存在 11k 未定义谓词、22k 关系违规，未完全统一；④ 提案队列增长速度未能匹配人力；⑤ triage agent 仅为设计草案，未在生产环境运行；⑥ 评估样本非均匀，未得到全局 recall/precision；⑦ 某些错误（如位置优先级失效）需进一步改进。

---

## 198. Divided Attention Amplifies the Importance of Expectation-Aligned Visualization Design

**arXiv ID:** 2608.10320 | [PDF](https://arxiv.org/pdf/2608.10320v1)

**作者:** Jiho Kim `[一作]` (University of Wisconsin Madison), Michael Gleicher `[通讯]` (University of Wisconsin Madison)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过两项实验，探究在分心（双任务）条件下，色图（colormap）可视化设计与观察者先前期望（暗色对应大值、高色对应大值）是否一致时对任务表现的影响；

**💡 创新点**

创新点在于：①首次系统考察分心情境下期望违背的可视化设计所产生的增大成本；②通过线性弹道累积器（LBA）模型解释该成本来自于证据区分度下降；

**🔧 技术方法**

采用心理学实验方法，结合线性弹道累积器（LBA）决策模型对响应时间、准确率与漏答率进行联合建模；

**📊 数据集**

实验数据来源为来自Prolific的受试者，共167人（实验1）与117人（实验2），各自完成单任务与双任务版本的色图解读任务；

**📈 对比分析**

方法对比：单任务vs双任务、不同的色图映射（暗色大/亮色大）与高度映射（高大/低大），通过混合效应回归分析和LBA模拟验证结果；实验1中双任务显著提升错误率与响应时间；实验2中双任务导致误答率显著升高；

**⚠️ 局限性**

局限性包括：仅考察暗色-大与高色-大两种期望偏差；仅使用单一可视化任务（色图）；双任务仅模拟文本信息监控，未覆盖更广泛的现实多任务场景；LBA模型依赖假设，未通过眼动等直接测量验证。

---

## 199. Sequential Modality Dropout for Robust Multi-Modal Sequential Recommendation

**arXiv ID:** 2608.10240 | [PDF](https://arxiv.org/pdf/2608.10240v1)

**作者:** Guanqun Yang `[一作]` (Stevens Institute of Technology), Wenlong Zhang `[通讯]` (Stevens Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种四行代码的“Sequential Modality Dropout（SMD）”机制，使多模态序列推荐器在训练时对每个用户历史统一随机丢弃整条模态（图像或文本），从而提升模型在实际缺失模态时的鲁棒性。

**💡 创新点**

创新点在于：①将Per‑sample Bernoulli模态掩码直接注入任何多模态序列推荐器的融合点，做到架构无关；②通过可选的跨模态重建损失进一步提升在极端缺失场景下的表现；③仅四行代码即可实现，易于迁移与扩展。

**🔧 技术方法**

技术手段包括：基于SASRec及其多模态扩展（MM‑SASRec）、IISAN、MISSRec、fMRLRec等后端；使用冻结的CLIP视觉向量和BERT/LLM文本向量；采用二元交叉熵作为主损失，选配跨模态重建损失；实验评估使用HR@10、保留率、McNemar/Wilcoxon统计检验。

**📊 数据集**

使用Amazon四个域（Scientific、Instruments、Arts、Office）以及Beauty & Personal Care（Amazon Reviews 2023）作为数据集，覆盖26%–41%图像缺失率、48.3%文本缺失率等真实缺失场景。

**📈 对比分析**

通过对比无SMD与有SMD的模型，分别在全模态移除与逐项随机缺失两种评估协议下进行测试。SMD在四种后端模型中均提高文本保留率1.0–3.2×，在95%缺失率下HR@10保留61%（无SMD仅22%）；在所有16个模型×数据集组合中平均提升0.8% HR@10，并在121k用户级别的统计检验中显著。

**⚠️ 局限性**

局限性：①对模态损坏（噪声、失真）缺乏处理；②在复杂动态融合模型上提升有限；③仍假设缺失模式为整条用户序列或类别级统一缺失，未覆盖更细粒度的随机缺失场景；④扩展到音频、视频或结构属性等更多模态尚未验证。

---

## 200. Do LLM Recommenders Know When They're Hallucinating? Auditing Confidence Calibration in Catalog Faithfulness

**arXiv ID:** 2608.10008 | [PDF](https://arxiv.org/pdf/2608.10008v1)

**作者:** Srijith Ravikumar `[一作]` `[通讯]` (Amazon.com LLC), Srijith Ravikumar (Amazon.com LLC)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对零射大型语言模型（LLM）推荐器进行联合审核，既评估其生成结果是否落在目标目录（hallucination率）又评估模型自报置信度的校准程度（ECE、Brier、可靠性图），并在此基础上构建分割式合适阈值来尝试减少hallucination。

**💡 创新点**

创新点在于：①首次在推荐领域同时报告目录忠实度与置信度校准，弥补以往只关注二元hallucination的空白；②发现LLM在目录忠实度上的低估置信度是“提示-校准”不匹配的结果，而非模型本身过度自信；③提出基于分割式合适阈值的校准驱动拒绝机制，并证明其在四个不同LLM上的效果有限。

**🔧 技术方法**

使用了：零射提示的LLM（Mistral Large、Llama‑3.3‑70B‑Instruct、GPT‑OSS‑120B、Claude Sonnet 4.6），"Just Ask"自评置信度提示，ECE、Brier、可靠性图的计算方法，以及分割式合适阈值（split‑conformal）技术。

**📊 数据集**

数据集包含三个规模与canonical性不同的目录：MovieLens‑25M（62K电影标题）、Amazon Reviews 2023 Toys & Games（890K商品标题）和Yelp Open Dataset（150K本地商家）。每个目录均构造了300个有30+交互记录的审计用户和100个独立验证用户。

**📈 对比分析**

对四个LLM在每个目录、以及按流行度分层（Q1、Q4）的12个子组进行了比较：MovieLens 的OOD@10几乎为0，ECE高达0.22；Amazon 与 Yelp 的OOD分别为4–8%，ECE更高。使用分割式合适阈值时，hallucination 下降最多0.7个百分点，覆盖率下降4–21%。结果表明，置信度校准与hallucination是可分离的评价轴。

**⚠️ 局限性**

局限性：①仅评估零射LLM，未涵盖检索/微调的推荐堆栈；②自评置信度高度依赖提示语，可能与不同厂商的实现不一致；③合适阈值的分布边际保证对长尾子组不成立；④需要定期重新校准；⑤未探讨结合检索或检索‑生成混合模型的校准改进。

---

## 201. Spatio-Temporal Scheduling for Robust and Efficient Multi-Transmitter Wireless Power Transfer

**arXiv ID:** 2608.10391 | [PDF](https://arxiv.org/pdf/2608.10391v1)

**作者:** Yuna Sawada `[一作]` (Tokyo Metropolitan University), Tomotaka Kimura `[通讯]` (Doshisha University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于多发射器协同的时空调度框架，用于无线能量传输，结合了接收器聚类、α-公平度量的波束成形与时隙分配

**💡 创新点**

创新点在于将多发射器的互相干扰视为有益干扰，通过全局协同波束成形和分组调度实现能量传输的公平与高效，并采用轮询聚类避免负载不均

**🔧 技术方法**

采用MISO波束成形、α-公平度量优化、时隙组合搜索、Sionna射线跟踪仿真、Lambert W函数能量模型等技术

**📊 数据集**

使用Sionna RT射线跟踪仿真生成的5.7 GHz信道数据，模拟10 m×10 m办公环境，并加入移动障碍物进行时变仿真

**📈 对比分析**

与单发射器集中部署、以及仅采用集群内最大比传输（MRT）的对比，结果显示分布式部署在LOS阻塞下更鲁棒，交叉集群波束成形的累计分布函数（CDF）明显上移，能量接收更高

**⚠️ 局限性**

主要限制在于时隙分配的组合优化计算量大、未给出最优时隙分配的实际实现，且仍需进一步研究ET布置与阻塞感知聚类策略

---

## 202. SapiensID 2.0: Aligning Human Recognition Foundation Models with Human Perception

**arXiv ID:** 2608.10497 | [PDF](https://arxiv.org/pdf/2608.10497v1)

**作者:** Yiyang Su `[一作]` (Michigan State University), Xiaoming Liu `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了统一的人类识别基础模型 SapiensID 2.0，融合了语义知觉（固定软生物特征）与运动连贯性（时序动力学），通过将多模态大型语言模型（MLLM）的语义知识迁移至视觉特征空间，实现面部、人体再识别和步态识别三种任务的零样本高性能。

**💡 创新点**

创新点包括：① 通过特征级子空间对齐（Invariant Trait Alignment ITA 与 Transient Noise Disentanglement TND）实现固定软生物特征的注入与动态噪声的消除；② 设计 Kinematic Semantic Attention Head（K‑SAH），在不需要大规模视频训练的情况下，通过跨帧注意力捕捉运动轨迹；③ 在单一模型上同时支持面部、人体再识别与步态识别，显著提升跨任务泛化能力。

**🔧 技术方法**

使用的技术包括：ViT+RetinaPatch 视觉特征提取、跨模态文本编码器（MLLM）生成的语义向量、SVD+对齐损失实现子空间对齐、特征映射 MLP、K‑SAH 的多帧交叉注意力与遮挡鲁棒时间池化、语义子空间构造与相对关系保持损失。

**📊 数据集**

训练数据集：WebBody4M（含图像与 MLLM 语义标签）；评测数据集：人类再识别（CCVID、Market‑1501、MSMT17、LTCC、PRCC、CCDA、Celeb‑ReID）、步态识别（CCPG、BiggerGait、GaitBase 等）、人脸识别（LFW、CPLFW、CFP‑FP、CALFW、AgeDB）。

**📈 对比分析**

与现有最先进模型（CLIP3DReID、SOLIDER、HAP、BiggerGait、SapiensID 基线等）在同一零样本设置下对比，SapiensID 2.0 在 CCVID 通用/服装变化协议分别达 95.70% / 84.99% 的 Top‑1；在 CCPG CL/DN/BG 任务分别取得 60.8% / 86.4% / 91.6% 的 Rank‑1，显著领先；在人脸验证任务上匹配或略超 AdaFace 的精度。Ablation 证明 ITA、TND、K‑SAH 逐步提升性能。

**⚠️ 局限性**

局限性：在短期服装不变的 ReID 任务中，由于抑制服装信息，略低于部分基线；依赖离线 MLLM 语义提取，训练时仍需额外时间；K‑SAH 的时间窗口有限，可能无法捕捉更长范围的运动依赖；在极端遮挡或多主体场景下，语义标签可能产生误导。

---

## 203. Conflict Extraction in Probabilistic Datalog Analyses

**arXiv ID:** 2608.10755 | [PDF](https://arxiv.org/pdf/2608.10755v1)

**作者:** Siyu Chen `[一作]` (Purdue University), Jingbo Wang `[通讯]` (Purdue University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种用于概率Datalog程序的冲突提取框架，能够从推理结果中自动识别并枚举最小不满足子集（MUS），从而揭示概率推理中潜在的逻辑不一致。

**💡 创新点**

创新点主要包括：①利用推导图中的结构与统计负相关信息进行导向式采样（Derivative Sampling）以优先探索高冲突可能的候选集合；②设计了自底向上推理（Bottom‑Up Inference）机制，通过逻辑替换在不增加SAT调用的前提下自动推导更多MUS；③在概率Datalog语义下提供了完整且无误的MUS枚举方法，显著提升了冲突检测的吞吐量与诊断覆盖率。

**🔧 技术方法**

技术手段涵盖：概率Datalog程序建模与归一化；将推导图编码为布尔约束并使用CVC5求解；导向式采样算法与动态采样额度控制；自底向上推理逻辑替换与MUS收缩；与传统的基线MUS枚举器（如MARCO、CAMUS、UNIMUS等）进行对比。

**📊 数据集**

实验使用了70个真实案例，涵盖四大领域：功率侧信道分析（18个）、数据竞争检测（7个）、语义差异检测（41个）以及贝叶斯网络推断（4个）。每个案例对应具体的Datalog程序（如AES、SHA‑3、Apache HTTP、Linux IIO驱动等）。

**📈 对比分析**

与三种主流基线（CAMUS、MARCO、UNIMUS）以及其单项改进版本进行比较。在30分钟时间预算下，本框架在大多数领域内实现了6.6–65倍的MUS枚举吞吐率提升，且平均在侧信道与数据竞争任务中通过提取的MUS将误报量分别降低约69%与61%。

**⚠️ 局限性**

局限性：①仍需依赖外部SAT/SMT求解器，求解性能受限于求解器本身；②导向式采样的随机性导致不同种子下结果略有波动；③目前实验聚焦于四类典型分析任务，尚未在更大规模或不同逻辑语言（如LPAD、BLOG）中验证可扩展性；④对深度递归推导图的处理仍可能在极端大规模图上产生空间与时间瓶颈。

---

## 204. Smart Enough to Go Extinct? An Evolutionary Challenge to the Value of General Intelligence and Its Ethical Implications for AGI

**arXiv ID:** 2608.10730 | [PDF](https://arxiv.org/pdf/2608.10730v1)

**作者:** David Klotz `[一作]` `[通讯]` (Hochschule der Medien), David Klotz (Hochschule der Medien)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论证人类普遍智能的价值并从进化视角提出疑问，探讨其对AGI开发的伦理意义

**💡 创新点**

首次将进化生物学证据与AGI伦理论证结合，提出“存在风险悖论”与“进化不确定性”框架

**🔧 技术方法**

主要采用哲学分析与进化生物学文献综述，没有实验性或计算方法

**📊 数据集**

无数据集，全部基于已有生物学、化石与文献记录

**📈 对比分析**

无性能评估或比较实验，论证以理论推理和历史案例为依据

**⚠️ 局限性**

局限在于：1）人类进化史仅300,000年，难以对长期价值作结论；2）比较生物学与人工智能的类比并非完全成立；3）未考虑文化进化对生存价值的潜在影响

---

## 205. MVTrack: Ultrafast Appearance-Free Moving Object Tracking from Compressed Bitstreams

**arXiv ID:** 2608.10790 | [PDF](https://arxiv.org/pdf/2608.10790v1)

**作者:** Iñaki Erregue `[一作]` (Universitat de Barcelona), Sergio Escalera `[通讯]` (Universitat de Barcelona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为MVTrack的压缩域移动对象追踪框架，完全基于H.264码流中的运动向量进行检测与关联，无需像素重建。

**💡 创新点**

创新点在于：①将运动向量作为主信号实现“always-fast”追踪；②轻量化Anchor-free检测器MVDet结合时间门控与宏块分区嵌入；③改进ByteTrack的MVLink模块，通过视角无关的速度加速度估计解决停滞期身份碎片化。

**🔧 技术方法**

使用技术包括：CenterNet-inspired轻量检测器、EMA式时间门控、MB分区嵌入、ByteTrack关联改造、对H.264码流的运动向量提取与归一化。

**📊 数据集**

在VIRAT Ground Dataset（DIVA‑V1子集）上进行训练与评估，重点关注行人与车辆。

**📈 对比分析**

与YOLO26n‑ByteTrack（RGB基线）比较，MVTrack在HOTA上略高（53.88% vs 52.78%），MOTA、IDF1等指标均优于RGB基线，同时参数量↓60×、FLOPs↓40×、CPU延迟↓8.6×。

**⚠️ 局限性**

主要限制包括：①对长时间停滞对象仍易导致身份丢失；②运动向量的空间分辨率低，导致小物体或密集场景下定位误差；③对不同码率/帧率/B帧的鲁棒性虽有评估但仍受限于训练集的编码配置。

---

## 206. OAA: Three Phases of Vocal Guidance in Human-Drone Teleoperation

**arXiv ID:** 2608.10651 | [PDF](https://arxiv.org/pdf/2608.10651v1)

**作者:** Allan Henry `[一作]` (Grenoble Alpes University), Sylvain Huet `[通讯]` (Grenoble Alpes University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过对人类-人类指引和人类-无人机遥控两种实验配置，结合运动捕捉和语音数据，发现并验证了人类在三维空间中自发语音引导始终分为三个阶段：定位、接近与调整。

**💡 创新点**

创新点在于首次用无监督变点检测自动划分轨迹，证明三阶段结构在不同执行接口下均保持一致，并从语音中同步识别两个阶段的语言与语调特征，为阶段感知的语音遥控系统奠定基础。

**🔧 技术方法**

所用技术包括：VICON 运动捕捉、PyAnnote 语音分段、Whisper 语音转写、变点检测算法、Kruskal‑Wallis 与 Dunn 事后检验，以及对词汇类别的正则表达式标注。

**📊 数据集**

数据集为 VoiceStick 人机-无人机遥控语音+轨迹数据（29 dyads，338 trial）以及人类-人类指引实验数据（10 dyads，120 trial），两者均在同一 3×3×3 立方体目标网格内完成任务。

**📈 对比分析**

对比方法为将检测得到的三段与等时段基准进行统计对比，三阶段在速度、角速度、加速度、语音密度、停顿时长及词汇类别等指标上均显著不同（p < 0.001），并证明语音特征可独立预测定位与调整两阶段。

**⚠️ 局限性**

局限性包括：仅假设三阶段结构，未处理多目标或障碍情境；实验仅在法语语料中验证，跨语言推广待考；未在实时控制系统中验证阶段感知的实际效益；并未考虑个体差异对分段的影响。

---

## 207. Narrative Keyframing for Generative Creative Writing

**arXiv ID:** 2608.10337 | [PDF](https://arxiv.org/pdf/2608.10337v1)

**作者:** Chao Zhang `[一作]` (Cornell University), Abe Davis `[通讯]` (Cornell University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为叙事关键帧（Narrative Keyframing）的交互技术，帮助作家在故事规划中使用情节关键帧、人物关键帧和视角关键帧，利用大型语言模型自动生成介于关键帧之间的叙事文本，并通过技术评估和用户研究验证其效果。

**💡 创新点**

创新点在于：①把动画关键帧概念迁移到叙事生成，实现对情节、人物发展和视角的细粒度、可迭代控制；②引入第一人称视角关键帧作为中间表达，让抽象人物特征通过可视化文本被具体化、检验和重用；③通过颜色和高亮实现关键帧与最终叙事的可追踪映射，提升透明度与作者可控性。

**🔧 技术方法**

技术实现：前端采用 Next.js、React Flow 和 Slate.js；后端调用 OpenAI GPT 系列模型；使用 Firebase 记录交互事件；对人物特征进行自动建议、插值、视角生成、证据提取和最终叙事生成。

**📊 数据集**

数据集：使用两组写作提示（各 10 条，来源于公开写作提示数据集），在每个提示下生成 5 个多样化大纲；此外利用 OpenAI GPT 进行自动特征建议和文本生成。

**📈 对比分析**

比较方法：与同一 LLM 的“纯文本提示”基线相对，先用自动质量评分模型（Chakrabarty 等）进行全局分数对比，再随机抽取 30 对故事进行人工评审；用户研究采用 12 名写作者进行对照实验，收集 CSI、AI System Experience 量表以及主观对比问卷。结果显示：系统在整体质量上优于基线（自动评分 t=6.52, p<0.001；人类评审 83.3% 偏好），并在角色刻画细节和可控性上显著更好。

**⚠️ 局限性**

局限性：①样本量小（12 名参与者），结论尚属初步；②工作流程较为线性，可能限制更自由或即兴写作；③对非线性情节（如闪回）支持不足；④目前仅关注角色、视角等属性，尚未扩展到节奏、语调等其他叙事维度；⑤系统主要针对短篇三幕结构，长篇或复杂叙事尚未验证。

---

## 208. Inferential Capability Does Not Determine Legal Scope

**arXiv ID:** 2608.10601 | [PDF](https://arxiv.org/pdf/2608.10601v1)

**作者:** Nicola Fabiano `[一作]` `[通讯]` (Studio Legale Fabiano), Nicola Fabiano (Studio Legale Fabiano)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对AI Act与GDPR中推理（inference）概念的两种功能（构成与保护）进行理论梳理，提出两级分析框架、三条保护路径、四个架构维度以及推理链与归属链的区分；

**💡 创新点**

创新点在于将推理的构成功能与保护功能区分开来，并提出以推理链与归属链为核心的两级分析框架，阐释代理式架构如何通过合成、持续性和可审计性等维度改变推理的法律影响；

**🔧 技术方法**

主要采用法理学方法、案例研究与技术文献综述（包括Poretschkin–Naeven的推理层级模型、欧盟判例等）进行理论推导；

**📊 数据集**

无实验数据集，本文为纯理论/规范性研究；

**📈 对比分析**

无实验对比或性能评估，本文的“方法”是理论框架和推导，未涉及数值指标；

**⚠️ 局限性**

局限性在于缺乏经验验证，推理链与归属链的可追溯性假设未经实证检验，对多方代理层级的完整合成规则仍不完整，且研究仅聚焦欧盟法域，未对跨司法管辖区的适用性进行探讨。

---

## 209. AI-Generated Interactive Fiction for Educational Use: A Pilot Study of Perceived Comprehensibility, Coherence, and Engagement

**arXiv ID:** 2608.10818 | [PDF](https://arxiv.org/pdf/2608.10818v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 210. Secure Cooperative THz ISAC via Mamba Empowered Graph Neural Network Precoding

**arXiv ID:** 2608.10467 | [PDF](https://arxiv.org/pdf/2608.10467v1)

**作者:** Chao Wang `[一作]` (Xidian University), Derrick Wing Kwan Ng `[通讯]` (University of New South Wales)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了基于Mamba增强图神经网络的联合优化框架，用于多基站协作的THz ISAC系统中安全通信和雷达信号的协同设计，最大化最小密文速率并满足CRB定位误差约束。

**💡 创新点**

创新点在于将Mamba状态空间模型与GNN相结合，既能在极大规模天线阵列、宽带机动和近场效应下实现可扩展、高效的联合波束赋形与雷达协同，也通过图结构捕获多用户、多目标与基站间的物理交互，从而实现安全性与感知性能的协同优化。

**🔧 技术方法**

使用的技术包括THz OFDM双向ISAC系统建模、近场波束分裂补偿的TTD与相位移器、基于CRB的定位约束、Mamba‑empowered GNN（消息传递、Mamba模块、ZF预编码、功率分配模块）以及无监督训练的损失函数。

**📊 数据集**

数据集为仿真生成的随机部署场景，基站、用户、目标位置随机生成，利用HITRAN数据库计算分子吸收损耗，并在不同BS、用户、目标数与功率预算下生成训练与测试样本。

**📈 对比分析**

与传统交替优化（Alt‑Min）和深度学习基线（CVNN、CNN‑LSTM）比较，所提GNN在最小密文速率上显著优于对手，推理时间大幅降低，并能在不同子载波数、RF链数、功率和拓扑变化下保持性能优势。

**⚠️ 局限性**

限制在于依赖仿真数据，未考虑实际硬件非理想和时变信道；模型在极大规模网络时可能需要更深网络；以及未实现多目标与多用户的实时自适应调度与覆盖率平衡等功能。

---

## 211. Nutrition Data Infrastructure for the AI Era: Operationalizing FAIR for Agent-Mediated Research

**arXiv ID:** 2608.10363 | [PDF](https://arxiv.org/pdf/2608.10363v1)

**作者:** Lin Liao `[一作]` (8up), Peng Li `[通讯]` (8up)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了名为Nutrition Data Service (NDS) 的源保持基础设施，使 AI 代理能够通过描述精确匹配食品记录、构建类型化交叉映射并通过固定接口实现可重复的营养分析。

**💡 创新点**

创新点在于：①通过来源识别、版本与语义保持实现 FAIR 的实际应用；②采用描述驱动的多通道检索+LLM 再排序，保证匹配精度与可解释性；③设计可类型化、版本化的交叉映射，支持断言、拒绝与审计；④引入“Pinned Agent”接口，将检索与映射决策固定，消除模型间的再计算波动。

**🔧 技术方法**

技术包括：Hybrid Retrieval（语义 + 词典检索）、基于 pgvector 的近似最近邻索引、LLM 重新排序、DynamoDB 存储、PostgreSQL 与 pgvector 组合、REST/MCP 接口、S3 等批量导出，以及 SSSOM/FOO 等标准化映射框架。

**📊 数据集**

使用了多源公开数据库：FNDDS、USDA FoodData Central、NHANES、Davis Food Glycopedia 2.0、International Tables of Glycemic Index 等；并在 1,000 条 NHANES 餐食、11,857 条 NutriBench 任务以及 500 条交叉映射样本上进行评估。

**📈 对比分析**

与 GPT‑4o‑CoT 等已发表模型相比，NDS 在 11,857 条查询上的回答率 96.4%（对 84.6% 的查询错误≤7.5g，MAE 4.3g），远优于 66.8%/8.6g；在 NHANES‑DFG2 交叉映射基准中准确率从 0.654 提升至 0.688，特别是在拒绝无关匹配时提高 15.8 点；在可重复实验中，NDS 的 GL CV 为 0，DIY 版 CV 最高 0.816，展示了极高的可重复性。

**⚠️ 局限性**

局限性包括：仅覆盖美国数据，未覆盖国际和受限授权数据库；缺乏专家手工标注的黄金标准交叉映射；评估多聚焦在匹配和交叉映射任务，对全流程的临床有效性未验证；对受保护临床数据、隐私与合规性尚未做充分考量。

---

## 212. A matched-integrator evaluation of Hamiltonian neural networks on pendulum and Kepler dynamics

**arXiv ID:** 2608.10235 | [PDF](https://arxiv.org/pdf/2608.10235v1)

**作者:** Lenick Kemunto Nyabuto `[一作]` (African Institute for Mathematical Sciences), Birahim Tewe `[通讯]` (African Institute for Mathematical Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在控制条件下比较Hamiltonian神经网络（HNN）与普通前馈网络在守恒动力学中的表现，使用匹配的网络容量、相同的数据、相同的数值积分器和多随机种子评估；

**💡 创新点**

通过严格的匹配实验展示HNN在长时序能量守恒和轨迹精度上的显著优势，且能量漂移保持有界、波动小；

**🔧 技术方法**

使用Hamiltonian神经网络、普通多层感知机、RK4积分、中心差分导数目标、Adam优化器、张量自动微分；

**📊 数据集**

非线性摆和三维Kepler两体问题的仿真轨迹数据（通过RK4生成），分别包含70条（摆）和60条（Kepler）轨迹；

**📈 对比分析**

采用相同RK4推演、相同参数匹配、5个随机种子，多尺度时间与能量分层分析，结果显示HNN在能量漂移上平均减少42倍、轨迹MSE减少15.8倍；

**⚠️ 局限性**

实验仅覆盖可积的低维守恒系统，未测试混沌、非可积、耗散或多体系统；仅与普通前馈网络比较，未涵盖Lagrangian NN、SympNets等对比；计算成本在单核CPU上不具优势，需进一步研究高维多体情形。

---

## 213. Agentic Instruction Data Selection: Let DataMaster Interpret Your Intent

**arXiv ID:** 2608.10579 | [PDF](https://arxiv.org/pdf/2608.10579v1)

**作者:** Fanqi Zhou `[一作]` (Nanjing University), Gong Cheng `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 Instruction Data Selection Agent（I‑DSA），通过四阶段（域、特征、信息量、质量）动态地根据用户自然语言指令在大规模异构指令数据集上筛选训练子集。

**💡 创新点**

创新点包括：①动态、意图条件化的数据选择流程，摆脱固定阈值；②四个专用代理按顺序演进，逐步细化子集；③利用LLM生成过滤规则、指标权重和质量评估表格，实现全流程自动化；④结合模型相关的学习价值评估，提升子集质量。

**🔧 技术方法**

技术手段包括：使用DeepSeek‑V4‑Flash LLM作为核心引擎；HDBSCAN/KMeans聚类实现域过滤；工具沙箱实现数据查询和统计；LLM生成规则/权重/评估表；多指标加权（NLL、词熵、语义漂移、MeanDiff、质量维度）与Top‑K筛选；并行批量评估。

**📊 数据集**

数据集：单域实验使用 OpenR1‑Math、Synthetic‑Math；Medical‑Reasoning、UltraMedical；OSS‑Instruct、Code‑Alpaca；多域实验使用 OpenHermes‑2.5、Tulu‑3‑SFT‑Mixture。评测基准包括 GSM8K、MATH、Minerva‑Math、AMC23、AIME、LiveCodeBench、HumanEval、BigCodeBench、MedQA、MMLU‑Medical、MMLU‑STEM、GPQA‑Diamond。

**📈 对比分析**

与随机、六种静态选择方法（SuperFiltering、SelectIT、MIG、Select2Reason、DEFT 等）以及 Claude Code 动态基线进行对比；在 18 个单域实验中，I‑DSA 在 13 组排名第一，平均比最强静态基线高 3‑4 分；在 12 组实验中比全池训练更优；多域实验中，I‑DSA 超越所有静态基线及 Claude Code，平均提升 0.33‑2.84 分；每次数据选择的云 LLM 成本约 4‑5 美元。

**⚠️ 局限性**

局限性：仅在 7B‑8B 规模模型上验证，未探讨更大或不同架构模型的适用性；与 Claude Code 的对比受限于成本/提示；未实现对代理激活/顺序的自适应调整；缺乏基于下游评估反馈的迭代优化。

---

## 214. A Joint-Distribution Route to Fair Representations with Continuous Sensitive Attributes

**arXiv ID:** 2608.10470 | [PDF](https://arxiv.org/pdf/2608.10470v1)

**作者:** Yijin Ni `[一作]` (Georgia Institute of Technology), Xiaoming Huo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出使用联合分布的Hilbert–Schmidt Independence Criterion（HSIC）作为连续敏感属性公平性约束，形成FRHSIC算法；通过理论证明该联合度量等价于传统条件积分形式，并给出更高效的统计估计与训练一致性保证。

**💡 创新点**

创新点包括：① 将连续敏感属性的公平性判定转为联合IPM并证明其与条件积分式在可分离类下完全等价；② 证明HSIC与条件MMD的等价性，并给出谱尾上界，完成对公平度量的闭式控制；③ 在统计效率上实现O(n^{-1/2})收敛率，显著优于传统基于核平滑的O(n^{-2/5})；④ 通过批量HSIC正则化实现每周期约36倍的训练速度提升。

**🔧 技术方法**

使用的技术包括：Hilbert–Schmidt Independence Criterion (HSIC)、积分概率度量(IPM)、最大均值差异(MMD)、条件分解(disintegration)理论、谱分析、RKHS核方法、V-statistic估计、梯度下降的批量训练、Gaussian complexity与一致性证明。

**📊 数据集**

实验使用的公开数据集有：Adult、ACS Income、MEPS、Crime（回归）、COMPAS；并用高斯合成数据验证估计收敛率。

**📈 对比分析**

与现有方法（FREM、Reg‑GDP、ADV、MMD与LAFTR的二进制分箱版本）在公平-准确率折中进行对比；FRHSIC在所有数据集上与最强连续敏感属性基线相当，且在训练速度上比FREM快约36倍；在不同下游预测头上保持公平性稳定；在大样本中实验验证估计误差与理论O(n^{-1/2})一致。

**⚠️ 局限性**

局限性包括：需要选择特征映射到RKHS的核，谱尾近似可能不够紧；对多维敏感属性需采用产品核；对非RKHS的公平度量（如Wasserstein、TV）无法直接应用；速率证明仅在编码器复杂度受限（如有限宽度MLP、线性编码器）时成立；未讨论对分布漂移或数据不平衡的鲁棒性。

---

## 215. VoxSumm: A Multilingual Corpus of Long-Form Spoken News for Joint Summarization and Translation

**arXiv ID:** 2608.10359 | [PDF](https://arxiv.org/pdf/2608.10359v1)

**作者:** Yejin Jeon `[一作]` (Mila Quebec Ai Institute), David Ifeoluwa Adelani `[通讯]` (Mila Quebec Ai Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了联合语音摘要与翻译任务（JSumT）以及跨语言长语音摘要基准 VoxSumm。

**💡 创新点**

首次在多语言环境下将语音摘要与翻译合并为一体化任务，并构建覆盖24种语言、约703小时语音的真实数据集。

**🔧 技术方法**

采用大规模多模态语言模型（Gemini3.1‑Pro、Falcon 180B、ChatGPT‑3.5）结合零/少样本与链式思维提示进行实验。

**📊 数据集**

使用 VoxSumm 数据集（BBC 文章与对应多语言摘要的合成语音）进行训练与评估。

**📈 对比分析**

通过 BERTScore、xCOMET 以及人工评估对模型进行对比，Gemini3.1‑Pro 在英语与非英语摘要上表现最佳，少样本提示最有效，翻译先于摘要会显著降低性能。

**⚠️ 局限性**

局限包括：基于 CrossSum 自动匹配的跨语言对可能存在误配；语音为合成而非真实录音，可能影响模型在真实环境中的泛化。

---

## 216. TACTICL: Task-Aware Compression of Tabular ICL Models

**arXiv ID:** 2608.10837 | [PDF](https://arxiv.org/pdf/2608.10837v1)

**作者:** Mykhailo Koshil `[一作]` (TU Dortmund University), Katharina Eggensperger `[通讯]` (TU Dortmund University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种任务感知的压缩框架，自动剪枝Transformer层并用轻量级适配器替换，从而在保持ICL能力的同时显著降低推理成本。

**💡 创新点**

创新点在于将层级压缩与任务特定适配器结合，既保持原始模型的上下文学习，又通过自动搜索实现大规模层裁剪。

**🔧 技术方法**

使用基于AUC/稳定性的贪心搜索、轻量MLP适配器预训练、有限微调（仅更新适配器）以及混合预训练+目标数据的微调策略。

**📊 数据集**

在TabPFN及其v2.5版本的数十个分类/回归基准数据集上进行评估。

**📈 对比分析**

与完整模型和单纯蒸馏的浅层MLP基线对比，压缩后模型可保持≥95%性能，同时推理速度提升约2–3×，在85%层裁剪时仍保持接近原模型AUC。

**⚠️ 局限性**

局限性包括缺乏统一的压缩-性能权衡准则、仅针对TabPFN系列模型验证、以及对不同任务的通用性和更大规模评估尚未完成。

---

## 217. Fisher8: Stabilizing Neural Heteroscedastic Regression via Output-Layer Fisher Geometry

**arXiv ID:** 2608.10374 | [PDF](https://arxiv.org/pdf/2608.10374v1)

**作者:** Sumedh Vemuganti `[一作]` (University of Illinois at Urbana-Champaign), Nickvash Kani `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Fisher8 方法，对神经网络输出层的梯度进行 Fisher 几何预处理，以稳定联合预测均值和不确定性。

**💡 创新点**

通过局部 KL 近似重新定向和缩放梯度，实现无数据相关超参数的自然梯度修正，并给出近似的 KL 信任半径。

**🔧 技术方法**

使用自然梯度、Fisher 信息矩阵、KL 近似、梯度预处理与标准反向传播技术。

**📊 数据集**

在多维回归与表示学习任务的合成与真实数据集（如 UCI、MNIST 等）上进行实验。

**📈 对比分析**

与先前的 β‑NLL、Faithful 等手动调节方法对比，Fisher8 在似然‑误差平衡、校准不确定性以及特征空间表达上均优于或至少与最佳手工方案相当。

**⚠️ 局限性**

仅限于输出层修正，需扩展至其他似然形式、密集预测任务，并需进一步研究与 Adam 等优化器的交互。

---

## 218. CARB: A Characterization-Guided Framework for CNN Inference Cost Prediction and Deployment Screening

**arXiv ID:** 2608.10506 | [PDF](https://arxiv.org/pdf/2608.10506v1)

**作者:** Linh Nguyen `[一作]` (Florida State University), Zhixin Pan `[通讯]` (Florida State University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对 13419 个基于 ResNet 的 CNN 配置在 NVIDIA RTX 5090 与 RTX 3080 两款 GPU 上进行大规模实验，系统量化能耗、延迟和峰值内存三种关键资源的缩放行为，并基于实验结果构建了 CARB——一种可并行预测这三项指标的级联集成模型，并提出两阶段部署筛选流程，在无需完整硬件测评的情况下将候选空间压缩至 Pareto 前沿。

**💡 创新点**

创新点包括：
1) 首次揭示能耗与延迟在高计算负载下显著解耦；
2) 明确记忆资源跨 GPU 的线性可迁移性与能耗/延迟的非线性差异；
3) 设计级联集成（Memory → Energy → Latency）并引入交互特征与残差校正的双阶段预测框架；
4) 通过平台特定模型与跨 GPU 关系，提出高效的两阶段部署筛选方法，显著减少硬件测评成本。

**🔧 技术方法**

主要技术手段包括：
- 采用 XGBoost、LightGBM、ExtraTrees 三种基学习器的专家集成与级联融合；
- 构造多维交互特征（如 batch_size×SM_util）与对数变换；
- 基于残差学习的低批量修正器；
- 两阶段 Pareto 筛选流程（先在 RTX 5090 上做排名，再在 RTX 3080 上做阈值过滤）。

**📊 数据集**

使用的数据集为在 RTX 5090 与 RTX 3080 上对 13419 个 ResNet 风格网络（包含 basic/bottleneck 块、宽度倍数、深度、输入分辨率、批量大小、精度等多维搜索空间）进行统一测量得到的能耗、延迟、峰值内存与运行时指标（SM 利用率、内存利用率、温度等）。

**📈 对比分析**

对比方法包括：
- 仅基于 FLOPs 的线性回归（R²<0.4）；
- 延迟作为能耗代理的线性拟合（R²≈0.99 但 MAE≈71 J）。
CARB 在测试集上达 R²≈0.99，MAE 分别为 6.9 MB（内存）、26 J（能耗）和 1.6 ms（延迟）。在 75 J 能耗阈值下，预算分类准确率为 95.8%，误报率仅 2.1%。

**⚠️ 局限性**

局限性包括：
1) 模型需要在目标 GPU 上有足够的标注样本，跨 GPU 迁移仅在本研究的 RTX 5090/3080 之间有效；
2) 对低批量（bs≤8）配置的预测误差仍较大，需要额外校正；
3) 仅验证了 ResNet 风格网络，可能不适用于其他网络结构；
4) 需要对 GPU 监测信息或使用架构特征的两种模式，若环境缺乏监测支持需额外手工设置。

---

## 219. Do Personalized Skills Help Coding Agents? An Empirical Study of Developer Interaction Histories

**arXiv ID:** 2608.10319 | [PDF](https://arxiv.org/pdf/2608.10319v1)

**作者:** Shuyan Huang `[一作]` (University of Massachusetts Amherst), Andrew Lan `[通讯]` (University of Massachusetts Amherst)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究从开发者与LLM编码代理的历史交互中提炼个性化技能，并评估其对后续任务的帮助

**💡 创新点**

提出两阶段个性化技能生成框架（规则引导的自举与证据驱动的精炼），并构建可复现的回放评估体系

**🔧 技术方法**

使用GPT-5.5（Codex）进行技能生成、代理执行与模拟器交互；采用LLM-as-a-judge进行任务完成打分；采用规则匹配与TF‑IDF评估技能相似度

**📊 数据集**

基于SWE‑chat公开的 8,866 条 CLI 编码代理会话，最终筛选 206 条（13 名开发者、42 条测试会话）

**📈 对比分析**

与四种条件对比：无技能、目标开发者个性化技能、随机他人技能、全局通用技能；结果显示通用技能平均得分最高（68.80 vs. 65.02），提升约3.8分；个性化技能仅提升约0.97分，增益不显著，随机他人技能与个性化相近

**⚠️ 局限性**

受限于每位开发者仅有 3–13 条会话，难以提取稳定偏好；技能覆盖度有限，个性化规则往往不够泛化；评估依赖模拟器，可能与真实交互差异；对任务多样性的覆盖不足

---

## 220. A Lightweight Fault-Detection Scheme for Barrett Modular Multiplication Using Multiple Conditional Reduction Paths

**arXiv ID:** 2608.10736 | [PDF](https://arxiv.org/pdf/2608.10736v1)

**作者:** Rourab Paul `[一作]` (Shiv Nadar University), Amlan Chakrabarti `[通讯]` (University of Calcutta)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在 lattice‑based PQC 和 FHE 的 NTT 加速器中，提出了一种基于统计归约路径的轻量级 Barrett 模块化乘法（BMM）故障检测方案，并实现了 FPGA 原型。

**💡 创新点**

创新点在于：①利用 BMM 的两轮归约（Reduction‑1、Reduction‑2）执行概率分布进行统计监控，而非传统的重算或冗余硬件；②只需统计四种归约路径的出现频率即可实现高覆盖率的故障检测；③在保持原有性能的前提下显著降低面积、延迟与功耗。

**🔧 技术方法**

核心技术包括：单词级 BMM（word‑wise BMM）、统计归约监视器（SRM）、随机与突发位翻转故障注入（fault injection）与硬件仿真、Vivado FPGA 设计与评估。

**📊 数据集**

使用的数据集为两大标准化方案的参数集：Kyber（l=12,w=4,q=3329）和 CKKS（l=32,w=8,q=1811939329），并在完整的 NTT 迭代（Kyber 1024 次、CKKS 24576 次）上执行 100,000 次随机/突发故障注入。

**📈 对比分析**

比较方法：将 SRM+word‑wise BMM 与现有重算方法（RESO、RENO、RESWO）以及 BMR 方案在 Artix‑7 FPGA 上进行面积、延迟、功耗和错误覆盖率对比。结果显示：SRM 仅增加 5.2%（Kyber）/1.2%（CKKS）面积，<1% 延迟与功耗提升，且在永久故障下 100% 检测率；在 128 次或 512 次以上的暂态故障也能保持 100% 检测率。

**⚠️ 局限性**

局限性：①对暂态故障的检测依赖于足够多的受影响 NTT 迭代（Kyber 需 ≥128 次，CKKS ≥512 次）；②方案仅覆盖 BMM 归约路径，无法检测非归约相关的硬件缺陷；③在极低故障率或极短突发故障时，统计偏差可能导致误判；④需要在设计中嵌入 SRM 统计计数逻辑，尽管开销小，但仍不适用于极端资源受限的微控制器。

---

## 221. Eleven Years of BRACIS: A Meta-Scientific Study of the Brazilian Conference on Intelligent Systems

**arXiv ID:** 2608.09964 | [PDF](https://arxiv.org/pdf/2608.09964v1)

**作者:** Thales Sales Almeida `[一作]` (Tropic AI), Rodrigo Nogueira `[通讯]` (Maritaca AI)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2015-2025年BRACIS会议的1,066篇论文进行全记录，整合元数据、引用、全文文本，并利用LLM自动提取主题、贡献类型、机构、性别、开放性信号，随后从主题、社区和影响力三维度系统分析会议产出。

**💡 创新点**

首次将元数据+全文+LLM自动化提取相结合，构建完整、可复现的区域会议全量分析框架；同时量化BRACIS的LLM兴起、引用集中度、开放性实践与引用关系，为区域会议研究提供新的方法与视角。

**🔧 技术方法**

使用的技术包括：sabiazinho-4 LLM进行文本结构化与分类；正则+LLM混合提取引用；PDF文本提取（PDFBox）与URL/章节识别；图网络分析（NetworkX）；统计检验（Mann‑Whitney、Spearman、t检验）；h‑index、g‑index、i10计数；以及多源数据抓取（DBLP、Google Scholar API、Semantic Scholar）等。

**📊 数据集**

主要数据集为：BRACIS 2015-2025论文列表及其DBLP元数据、Google Scholar引用计数、Semantic Scholar提供的arXiv链接、完整PDF全文、机构、性别推断、关键词、贡献类型、artifact链接、反思章节等信息，共1,066篇记录，其中1,046篇可用于引用/机构/性别分析，1,032篇可用于全文相关分析。

**📈 对比分析**

通过对不同开放性信号（arXiv、artifact、行业合著）与引用量的Mann‑Whitney检验，发现arXiv投稿与引用显著正相关（p<10⁻⁴），但artifact/行业合著无显著差异；按贡献类型对引用进行分组，Empirical与Model组在均值/中位数上高于Algorithm组；对最佳论文提名与引用的比较显示无显著差异；整体h‑index为30，g‑index为56，i10为162；引用分布呈heavy‑tailed，top 1%论文占27%总引用。

**⚠️ 局限性**

主要局限：仅基于Google Scholar，引用计数易受索引变动影响；PDF提取失败导致部分论文缺失；LLM分类与机构去重可能存在误差；性别推断准确率受限于姓名多样性；开放性与引用关联为相关性，无法断言因果；分析截止至2026年5月，后续会议或数据更新未被覆盖。

---

## 222. Self-Correcting Long-Horizon Search Agents via Tree-Structured Memory

**arXiv ID:** 2608.10676 | [PDF](https://arxiv.org/pdf/2608.10676v1)

**作者:** Aijun Yang `[一作]` (Shanghai Jiao Tong University), Jian Cao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种树结构工作内存 ReTree，专门用于搜索代理在多步检索过程中自我纠错，保持每步有限上下文并追踪证据来源。

**💡 创新点**

创新点包括：① 将检索过程建模为证据树，节点存储摘要、原子证据和修订历史；② 在发现冲突时定位引入该证据的节点，替换证据、重新生成摘要并剪枝所有依赖子树；③ 通过源绑定保持答复可追溯性，解决传统压缩方法丢失出处与错误链的问题。

**🔧 技术方法**

采用的技术包括：树结构内存、基于词法相关性的 top‑k 证据检索、冲突检测与确认、摘要重生成、子树修剪、以及声明级引用生成与评估。

**📊 数据集**

实验使用四个公开检索问答基准：Bamboogle、HotpotQA、2WikiMultiHopQA 与 FRAMES，共计 2,149 问题。

**📈 对比分析**

与基线 Full‑Trajectory ReAct、FlatUpdate（平面修订）和 ReportMemory（压缩报告）对比，ReTree 在所有数据集上提升 8.3–25.6pp 的评判准确率，整体提升 13.9pp；相对 Full‑Trajectory 的最大每步上下文长度缩小 1.27–1.51 倍，且在 FRAMES 上在引用精度、召回率和 F1 上优于其他方法。

**⚠️ 局限性**

局限性包括：① 硬性子树剪枝可能误删仍有效的后续状态；② 冲突检测和摘要重生成增加 7–11% 的模型调用与 10–13% 的 token 消耗；③ 随着检索深度外部证据存储线性增长，导致检索和树遍历成本上升。

---

## 223. FACT: Failure-Aware Causal Training for World-Action Models

**arXiv ID:** 2608.10232 | [PDF](https://arxiv.org/pdf/2608.10232v1)

**作者:** Quanquan Peng `[一作]` (University of California San Diego), Xiaolong Wang `[通讯]` (University of California San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种因果型世界-动作模型（FAWM），先生成动作再预测未来视频与任务进度，利用失败回放监督世界分支，提升机器人策略鲁棒性。

**💡 创新点**

创新点在于：①通过教师强制的动作条件掩码，将动作生成与世界预测解耦，使失败回放可以作为后果监督而非错误示例；②使用动作条件的进度预测头，可在推理时对候选动作进行可选评分；③失败数据在训练中显著减少成功偏差的未来幻觉。

**🔧 技术方法**

技术包括：基于WAN2.2-5B视频扩散变压器的共享网络，轻量动作适配器，双阶段推理（动作去噪→未来/价值预测），流匹配去噪损失，失败动作动作失真掩码以及可选的候选动作评分。

**📊 数据集**

数据集涵盖：50个RoboTwin仿真抓取/操控任务，约1.3k个失败回放；真实世界双臂操作的5个任务（如堆立方、取立方、递送、堆碗、倒水）及3个未见变体，分别使用200/50个专家演示以及约30个失败回放。

**📈 对比分析**

与多种基线对比（X‑VLA、π0、π0.5、Motus、Gigaworld‑Policy 等），在仿真上平均成功率从86.3%提升至88.4%，与Motus相近；在真实世界见任务上从82%提升至89%，可选评分进一步到92%；在未见任务上从67%提升至77%，接近π0.5。

**⚠️ 局限性**

局限性：依赖失败回放的可用性；计算成本随候选数上升；缺乏对更广泛人机交互数据的验证；价值预测仅作为可选评分器，若无失败训练效果有限。

---

## 224. REDAgentBench: Executable Red Teaming and Faithful Measurement of LLM Agent Systems

**arXiv ID:** 2608.10669 | [PDF](https://arxiv.org/pdf/2608.10669v1)

**作者:** Zixing Chen `[一作]` (Fudan University), Chi Zhang `[通讯]` (Alibaba Cloud Computing)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建可执行红队基准REDAgentBench，对LLM代理工具使用的安全风险进行自动化生成攻击、沙盒执行与状态/轨迹证据判定；

**💡 创新点**

提出曝光–执行–观察–判定四阶段ASR框架，发现并诊断“识别–执行差距”（REG）并通过无训练动作时提醒实现显著降低违规；

**🔧 技术方法**

使用LLM驱动的红队生成、服务沙盒化执行、状态与轨迹双证据判定、混合评判器以及基于政策的行动提醒；

**📊 数据集**

1,661个可执行案例，涵盖5个服务面、15种干预策略、11类漏洞、28条安全约束，来源于12,181条攻击映射；

**📈 对比分析**

在六种模型和三种代理实现上对ASR进行对比，模型/实现/判定视图差异显著，状态视图相较轨迹视图提升7–12个百分点，动作提醒可将ASR降低70%+；

**⚠️ 局限性**

仅在可执行环境下评估，缺乏对真实部署动态、长期交互、跨模型泛化的考量，且依赖人工审核与判定器特定实现。

---

## 225. REATS: LLM Reasoning-based Ensemble Learning for Adaptive Time Series Forecasting

**arXiv ID:** 2608.10149 | [PDF](https://arxiv.org/pdf/2608.10149v1)

**作者:** Xu Zhang `[一作]` (Fudan University), Li Zhao `[通讯]` (Microsoft Research)

**通讯引用:** 12728 | [OpenAlex ID](https://openalex.org/A5032277491)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了REATS框架，利用大型语言模型的推理能力实现时间序列预测的样本自适应、可解释性集成。

**💡 创新点**

通过固定token成本的混合文本-数值输入、规则生成的链式思考、多行权重监督以及对连续回归任务的GRPO奖励映射，显著提升了基于LLM的集成性能与解释性。

**🔧 技术方法**

使用Qwen3-1.7B LLM、检索增强生成(RAG)、多行权重表、逆向奖励映射、两阶段SFT+GRPO微调以及自定义整数百分比表输出。

**📊 数据集**

在八大公共时序数据集（ETTh1/2、ETTm1/2、Exchange、Weather、Electricity、Traffic）以及混合候选模型组上进行实验。

**📈 对比分析**

与传统固定权重、误差加权、RL网络以及多款零样本LLM集成方法对比，REATS在两类候选模型组均实现平均MSE降低约10–20%，并在OOD与模型数量变化时保持优势。

**⚠️ 局限性**

仍受限于候选模型描述的质量、检索召回的准确性以及LLM推理成本，且在极大候选集合或长时序时的推理效率和精度有进一步提升空间。

---

## 226. TRACE: Trustworthy Retrieval-Augmented Conversational Engine

**arXiv ID:** 2608.10176 | [PDF](https://arxiv.org/pdf/2608.10176v1)

**作者:** Touseef Hasan `[一作]` (Wichita State University), Souvika Sarkar `[通讯]` (Wichita State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出TRACE框架，采用结构与语义双重检索并结合知识图谱实现公共服务对话推荐

**💡 创新点**

将约束解析为结构约束（通过知识图谱）和语义约束（文本检索），并系统比较不同KG结构对检索与推荐可靠性的影响

**🔧 技术方法**

知识图谱、向量检索、双重检索管道、LLM生成（多种开源及专有模型）

**📊 数据集**

堪萨斯州食物储藏室目录（约800条记录）和构造的1000条合成查询基准

**📈 对比分析**

对15个开源LLM与1个专有模型在4种KG变体下进行比较，指标为约束满足率、幻觉率与语义相似度；结果显示KG-3（位置+营业时间）可将约束满足率从64%提升至88%，幻觉率从10.5%降至2.1%

**⚠️ 局限性**

仍依赖外部检索质量，若目录噪声大或缺失约束，仍可能出现误检；对真实多领域公共服务目录的通用性尚待进一步验证

---

## 227. Finding the Signal in the Spam: Jointly Learning Rewards and Worker Reliability from Pairwise Comparisons

**arXiv ID:** 2608.10045 | [PDF](https://arxiv.org/pdf/2608.10045v1)

**作者:** Kaustubh Shivshankar Shejole `[一作]` (IIT Bombay), Avishek Ghosh `[通讯]` (IIT Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种同时估计项目奖励与工人能力的框架，使用Boltzmann-理性模型与EM算法实现。

**💡 创新点**

创新点在于：①将Polya‑Gamma增广用于对数几率转化为高斯形式，②将M步优化转化为秩‑1矩阵感知问题，从而给出收敛理论与全局最优性近似。

**🔧 技术方法**

主要技术包括Boltzmann‑理性比较模型、Polya‑Gamma增广、EM与交替最大化、矩阵感知理论（RIP）以及CG求解线性系统。

**📊 数据集**

实验数据集：合成数据、FaceAge（9,150项目、4,091工人、250,249比较）以及Passage（472项目、624工人、11,763比较）。

**📈 对比分析**

与Simple、RC、FactorBT、BARP、CrowdBT、HTCV、HBTL等基线对比；在准确率、加权准确率和Kendall's Tau上均取得最优或接近最优；同时运行时间仅为几秒级，远快于其他方法。

**⚠️ 局限性**

局限性包括：对高维稀疏数据的收敛速度可能下降；模型假设工人能力区间[-1,1]且不考虑工人学习/退化；若项目奖励分布极端不对称，矩阵感知假设可能不成立。

---

## 228. DSAgentBench: Can Agents Automate End-to-End Data-Science Workflows in Real Computer Environments?

**arXiv ID:** 2608.10366 | [PDF](https://arxiv.org/pdf/2608.10366v1)

**作者:** Mizanur Rahman `[一作]` (York University), Enamul Hoque Prince `[通讯]` (York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DSAgentBench，一个基于真实操作系统的端到端数据科学工作流基准；

**💡 创新点**

首次将完整的数据科学生命周期（从数据获取、探索、特征工程、建模、评估到可视化）放到真实 OS 环境中进行评测，并提供确定性评估器；

**🔧 技术方法**

基于 OSWorld 的 Ubuntu 环境，集成 VS Code、Jupyter Notebook、Chrome、SQLite 等工具，并结合截图+可访问性树（A11y）视觉感知和多模态动作空间；

**📊 数据集**

共构建 275 题，来源于 Kaggle、OpenML、GitHub、SQLite 等公开数据集，涵盖表格、文本、图像等多模态；

**📈 对比分析**

在 15 个闭源和开源代理上评测，最强的 Claude‑4.6‑Sonnet 在截图+A11y 设置下仅达 56.7% 的任务成功率，人类基准约 85%；开源模型仅 1% 左右，显示出显著的能力差距；

**⚠️ 局限性**

局限包括：开源模型不支持 A11y 观测，错误分析样本有限，评估对可视化任务的侧重仅限于最终输出质量，且 A11y 的提升幅度有限。

---

## 229. MammoMix: Leveraging Mixture of Experts for Robust Mammogram Breast Detection

**arXiv ID:** 2608.10437 | [PDF](https://arxiv.org/pdf/2608.10437v1)

**作者:** Dinh Tan Nguyen `[一作]` (University of Technology Sydney), Sai Ho Ling `[通讯]` (University of Technology Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出 MammoMix 框架，利用 Mixture-of-Experts 进行乳腺病灶检测；

**💡 创新点**

创新点在于专家模型针对不同数据域进行专门训练，并引入门控机制动态加权以及 MoCAE 校准模块提升置信度可靠性；

**🔧 技术方法**

采用 YOLOS 作为检测骨干，结合 Mixture-of-Experts 架构、随机森林校准、Soft NMS 与 Score Voting 等技术；

**📊 数据集**

使用 CSAW、DDSM、DMID 三个公开乳腺影像数据集进行实验；

**📈 对比分析**

与 DETR、RT-DETRv2、单一 YOLOS 等基线对比，MammoMix 在 mAP@50-95 上整体提升，尤其在多域混合和小病灶检测上表现最优；

**⚠️ 局限性**

局限性包括计算成本高、推理延迟大、缺乏可解释性、数据不平衡可能导致模型偏差。

---

## 230. Who Gets Heeded? An Obligation-Level Audit of Responsiveness in EPA Rulemaking

**arXiv ID:** 2608.10329 | [PDF](https://arxiv.org/pdf/2608.10329v1)

**作者:** Jianing Fan `[一作]` (Columbia University), Yue Yao `[通讯]` (Columbia University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 EPA 公共评论与法规义务的关系进行义务级响应性审计，测量评论是否导致具体法规义务的修订。

**💡 创新点**

提出了义务级响应性审计框架，将评论与法规义务精确匹配，揭示不同评论者群体在义务层面的公平性差异。

**🔧 技术方法**

使用结构‑德奥提克解析加 LLM 验证的义务抽取、LLM 驱动的评论‑义务匹配、句子变换器余弦相似度的义务结果分类，并进行盲审。

**📊 数据集**

采用 EPA 2010‑2022 年 6,145 条规则提案的 786,197 条评论，36 条 anchor 规则的 70,075 条可分析评论。

**📈 对比分析**

通过人类盲审验证提取和匹配准确率接近 1（κ=1.0），结果分类在编辑与实质变更上 κ=0.137；发现评论量与义务修订关联约 1.9 倍机会，评论立场无显著差异，组织与个人在编辑精炼结果上呈显著交叉差异。

**⚠️ 局限性**

局限包括：结果分类对编辑/实质区分表现不佳、提交者类型重建为启发式、样本偏向高关注规则、未实现因果推断、检索召回未测量、批量评论未去重等。

---

## 231. MarkNull: Model-Agnostic Watermark Removal in AI-Generated Images via On-Manifold Latent Manipulation

**arXiv ID:** 2608.10166 | [PDF](https://arxiv.org/pdf/2608.10166v1)

**作者:** Jie Cao `[一作]` (Queen's University), Jianbing Ni `[通讯]` (Queen's University)

**通讯引用:** 7028 | [OpenAlex ID](https://openalex.org/A5033931001)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两种模型无关的水印去除攻击方法（单实例优化版和可摊销版WRN），通过在潜在空间上对齐噪声与生成潜在的统计相关性进行解耦，实现对AI生成图像水印的有效移除。

**💡 创新点**

创新点在于引入Noise‑Latent Alignment Score (NLAS) 量化潜在噪声与生成潜在之间的相关性，并以此为目标最小化实现水印去除；同时将优化过程蒸馏为单次前向网络（WRN）实现高效实时攻击。

**🔧 技术方法**

技术包括潜在空间正交化、噪声潜在对齐分数、潜在空间正则化、VAE/Stable Diffusion逆向采样、Restormer网络结构、LPIPS/SSIM/PSNR等感知损失、以及对比攻击检测的DDIM逆向再生。

**📊 数据集**

使用公开的 Stable Diffusion 2.1/1.5/XL 基础模型生成图像，水印方案覆盖 9 种（post‑hoc、fine‑tuning、initial‑noise），对视频的实验使用 Text‑to‑Video‑MS‑1.7b 和 VBench 提供的 20 条提示。

**📈 对比分析**

相较于现有 5 种基线（Distortion、Regeneration、Adversarial）以及最新攻击（Imprint、UnMarker、NPA 等），提出的方法在多位水印上平均 BA 降到 53.14%（接近随机 50%），在 SynthID‑Image 上实现 100% ASR，视频水印也将 BA 降至 ~50%；同时保持 CQS 4.0 以上，生成速度 0.5 秒/图，显著优于传统优化攻击。

**⚠️ 局限性**

局限性包括：对极端对抗性或复杂后期水印的鲁棒性仍有限；在极隐蔽的初始噪声水印下可能需要更大扰动导致视觉失真；依赖于代理模型的相似性，若目标模型与代理差异过大可能降低攻击效果；以及缺乏对不同生成模型架构（如GAN、VAE‑diffusion hybrid）的广泛验证。

---

## 232. Does the way we write a theory change the program an LLM builds from it? A prospective randomized study of renderer format in LLM theory-to-program translation

**arXiv ID:** 2608.10314 | [PDF](https://arxiv.org/pdf/2608.10314v1)

**作者:** Andre Panossian `[一作]` `[通讯]` (American University of Beirut), Andre Panossian (American University of Beirut)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对同一理论文本在两种渲染格式（hold‑change‑direction 指令式与连贯条件句）下，使用两个 LLM 版本进行 theory‑to‑program 翻译，并通过预注册的随机化实验评估生成程序的可执行相似度。

**💡 创新点**

创新点在于首次将 LLM 翻译过程设计为可预注册的随机化实验，使用完整的公平随机化与精确 Fisher 检验，评估不同提示格式对程序一致性的影响。

**🔧 技术方法**

使用的大型语言模型是固定的两份 LLM 快照，评估器为专用的稀疏二次语言（823 monomials）和基于有限差分的响应度量。

**📊 数据集**

数据集包括五个“理论轴”卡片，生成 320 个模型调用，产生 32 个渲染槽的 11 维响应张量。

**📈 对比分析**

比较方法是对同一账户在两种渲染下的程序进行匹配距离和闭集同账户可识别度测试，结果显示在交互挑战下匹配距离显著降低，但未达设定阈值；单输入挑战无显著差异。

**⚠️ 局限性**

主要局限在于样本量仅为 16 个块（N=16），并发的匹配结构不完善、缺乏正向对照，且仅评估两份 LLM 快照，未验证对其他模型或提示的普适性。

---

## 233. SeFoRA: Sketch-Aggregated Federated Low-Rank Adaptation with Heterogeneous Client Ranks

**arXiv ID:** 2608.10144 | [PDF](https://arxiv.org/pdf/2608.10144v1)

**作者:** Yue Xia `[一作]` (Technical University of Munich), Rawad Bitar `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的联邦学习算法，结合了低秩适应（LoRA）和参数高效微调（PEFT），解决了在不同客户端使用不同LoRA秩时的维度不兼容问题。

**💡 创新点**

创新点在于提出了一种草图聚合的联邦LoRA算法，允许客户端传输其本地更新的线性草图，从而在聚合时避免了双线性不匹配问题，并允许在全模型的小子空间中进行聚合。

**🔧 技术方法**

使用了线性草图方法来实现低秩近似，并在算法中引入了秩同质版本以保证聚合的直接适配。

**📊 数据集**

在GLUE数据集上进行了数值实验，特别是对RoBERTa-Large模型的微调。

**📈 对比分析**

与现有的联邦LoRA方法相比，提出的方法在性能上优于最先进的技术，尤其是在处理秩异构和双线性不匹配问题时表现出色。

**⚠️ 局限性**

限制在于该工作未直接关注隐私保证，但通过缓解双线性不匹配和允许线性聚合，为未来的安全聚合和更高效的差分隐私机制铺平了道路。

---

## 234. Where To Look? : Causal Tracing of Vision Encoders in VLM

**arXiv ID:** 2608.10758 | [PDF](https://arxiv.org/pdf/2608.10758v1)

**作者:** Naren Kumar S `[一作]` (Indian Institute of Technology Gandhinagar), Mayank Singh `[通讯]` (Indian Institute of Technology Gandhinagar)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过因果追踪技术，量化视觉语言模型中视觉标记的因果贡献，并与目标物体的空间重叠（IoU）进行对齐分析。

**💡 创新点**

创新点在于揭示了即使视觉标记对模型输出具有强因果影响，它们也常位于目标区域之外，从而说明强视觉语言性能不一定需要高度空间定位的因果表示。

**🔧 技术方法**

使用了激活补丁（activation patching）进行因果追踪，计算每层每个视觉标记的因果贡献 Γ，并与 IoU 相关联，进一步通过层级相关性分析和多种图像扰动（DDPM、模糊）评估稳健性。

**📊 数据集**

主要在包含目标边界框的标准图像-文本数据集上进行实验，典型数据集如 COCO‑VQA 等，但具体数据集未在文中明示。

**📈 对比分析**

将上述方法应用于 CLIP、DeepSeek‑VL、Qwen‑2.5、LLaVA‑NeXT、LLaVA、InternVL、SmolVLM 等多款模型，并与不同扰动方式进行对比；实验结果显示所有模型的因果‑空间相关性均维持在 0.01–0.09 之间，低于预期，表明空间对齐弱，尽管模型在任务上的表现已很强。

**⚠️ 局限性**

局限性包括：仅针对 ViT‑based 视觉编码器，未覆盖 SigLIP 等其他架构；仅评估了两种扰动方式；部分模型仅做单次实验，随机种子影响未充分评估；未来需在更广泛的模型代际和多种视觉编码器上验证结论。

---

## 235. DynaPPI: A Large-scale Dynamic Protein Dataset for AI-driven Advances in Protein Interactomics

**arXiv ID:** 2608.10435 | [PDF](https://arxiv.org/pdf/2608.10435v1)

**作者:** Jiabao Wei `[一作]` (BIT), Zhiyuan Ma `[通讯]` (HUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并公开了DynaPPI数据集，收集了约1万条蛋白复合物从解离到结合的分子动力学轨迹，涵盖多链、配体、核酸等多种类型；

**💡 创新点**

创新点在于首次提供时间分辨的多链蛋白复合物形成轨迹，填补现有静态或单体轨迹数据的空白，为AI模型学习动力学结合过程提供了关键资源；

**🔧 技术方法**

采用高精度分子动力学模拟、实验验证以及基于扩散模型的条件生成与预测技术；

**📊 数据集**

使用的核心数据集为DynaPPI，包含时间序列原子坐标、物理属性、接触图和自由能曲线等多模态信息；

**📈 对比分析**

与传统PDB/动态PDB数据集相比，在条件生成任务中扩散模型能够更准确地重现动力学轨迹和最终结合结构，RMSD降低约10‑15%，配体识别成功率提升约5‑8%；

**⚠️ 局限性**

限制主要包括巨大的数据存储和计算成本（10‑100 TB、数千小时MD模拟），以及对实验验证的依赖导致数据集仍未覆盖所有生物大分子交互场景。

---

## 236. Sensing in Low-altitude Wireless Networks: Systems, Techniques, and Developments

**arXiv ID:** 2608.10555 | [PDF](https://arxiv.org/pdf/2608.10555v1)

**作者:** Zihao Tao `[一作]` (Hong Kong University of Science and Technology), Ying Cui `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统综述了低空无线网络（LAWN）专用感知技术，阐述了感知框架、技术分类与挑战，并通过基于雷达-相机稀疏融合的模型与数据驱动多模态方法完成了实时无人机检测与轨迹估计的案例研究。

**💡 创新点**

创新点在于①首次专注于LAWN感知领域，提供完整的概念、服务、任务、节点、目标与场景体系；②对四维技术维度（传播介质、协同方式、建模方法、感知模态）进行统一对比与瓶颈剖析；③提出并验证了一种结合物理模型与深度学习的稀疏雷达‑相机多模态融合框架，显著提升了小RCS UAV的检测精度与实时性。

**🔧 技术方法**

技术包括：基于RF与光学的单/多模态感知、非协同与协同感知、模型驱动与数据驱动方法、时空对齐与特征平衡的多模态融合网络；案例实现中使用雷达点云编码器、图像特征提取、距离‑速度感知的查询更新机制以及轻量化推理网络。

**📊 数据集**

使用了自建的真实世界雷达‑相机多模态数据集，包含多种无人机类型、不同环境与光照条件，并配备RTK标注轨迹作为基准。

**📈 对比分析**

与单模态（BEVFormer、SparseBEV）和现有雷达‑相机融合基线（RCM、RaCFormer）进行对比，稀疏雷达‑相机融合模型实现了83.20% mAP、0.317 ATE、17 ms 推理延迟，优于所有基线。

**⚠️ 局限性**

局限性包括：夜间低光照下性能显著下降、对极端气象与遮挡的鲁棒性不足、数据集规模有限导致泛化能力受限，以及模型对不同硬件资源的可迁移性尚需进一步评估。

---

## 237. SBCO: Self-Supervised, Verifier-Grounded Harness Optimization For Planning Agents

**arXiv ID:** 2608.10157 | [PDF](https://arxiv.org/pdf/2608.10157v1)

**作者:** Vivek Kulkarni `[一作]` (Samsung Research America), Srinivas Chappidi `[通讯]` (Samsung Research America)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SBCO，一种自监督、基于验证器的闭环 harness 优化器，用于提升约束式规划任务的性能。

**💡 创新点**

通过分块逼近坐标上升结合文本梯度，学习可验证器库与修复策略，避免昂贵的自我修改或元认知搜索。

**🔧 技术方法**

使用约束器学习（签名块与实现块）、文本梯度优化、策略与修复策略的自监督闭环迭代、块优化与战略转移技术。

**📊 数据集**

在 DeepPlanning 基准上，包含旅行规划与购物规划这两个长期约束任务的数据集。

**📈 对比分析**

与 HGM、HGM‑C 等自我改进基线对比，SBCO 在 4–5.5 倍更低 compute 预算下匹配或超过基线：Travel 复合得分 84 vs 83，Shopping 匹配得分 94 vs 91。

**⚠️ 局限性**

局限于显式可检验约束的规划任务，受底层 LLM 强度限制，且缺乏元认知能力。

---

## 238. Bias Smells in AI Software Development: Recognizing Potential Sources of Fairness Debt

**arXiv ID:** 2608.10248 | [PDF](https://arxiv.org/pdf/2608.10248v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 239. Multimodal Item Parameter Estimation using Simulated Response Probabilitie

**arXiv ID:** 2608.10154 | [PDF](https://arxiv.org/pdf/2608.10154v1)

**作者:** Christopher Ormerod `[一作]` (College Board), YoungKoung Kim `[通讯]` (College Board)

**通讯引用:** 135 | [OpenAlex ID](https://openalex.org/A5055937848)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过对多模态LLM Qwen3.5 进行参数高效微调，使其在不同学生能力区间内模拟选择答案，从而重建多项选择模型（MCM）和三参数逻辑模型（3PL）的难度、区分度与猜测参数；并实现了图像+文本混合题目的统一输入处理。

**💡 创新点**

① 采用“模拟学生”框架取代传统刺激‑参数回归，直接学习学生在不同能力水平下的选择概率；② 通过细粒度的能力区间和多模态输入，捕捉了猜测行为与图像刺激对答案的影响；③ 在 4‑9B 参数规模的 Qwen3.5 上首次展示了在多模态题目上实现高精度难度参数恢复的可行性。

**🔧 技术方法**

使用 Qwen3.5-4B/9B 作为基础模型；通过 LoRA + 量化进行参数高效微调，仅对 Gated Attention 层进行更新；采用提示式训练（包含学生能力、正确答案、题干与选项），利用 MSE 损失最小化模型输出的下一个 token 概率与预设的能力区间概率；训练后在推理阶段直接读取每个选项的概率曲线并拟合 3PL/MCM；在验证集上使用线性校正提高预测精度。

**📊 数据集**

基于 970 个题目模型、4,848 个实际题目、13.88M 条响应的数学多模态数据集；每个题目包含文本与图像刺激，响应被按学生能力水平（从正态分布抽样）标注，且已预估出 3PL/MCM 参数。

**📈 对比分析**

评估指标包括 Pearson 相关、RMSE 以及基于难度分档的 Quadratic Weighted Kappa (QWK)。Qwen3.5‑9B 在难度参数 b 上取得 Pearson 0.85、RMSE 0.55、QWK 0.835，显著优于 110M MathBERT (0.68、0.78、0.692) 与 70B MetaMath (0.75、0.68、0.692)。在猜测参数 c 上亦取得相对较高的相关（0.48），而区分度 a 的相关相对较低。

**⚠️ 局限性**

• 区分度 a 的恢复效果差，可能受能力区间离散化和线性校正影响；
• 仅对 Gated Attention 层微调，DeltaNet 层未充分适配，可能限制模型性能提升；
• 研究假设 3PL/MCM 能很好拟合真实响应，映射误差可能导致参数估计偏差；
• 仅在 4‑9B 规模模型上验证，较大规模模型或更细粒度的能力划分可能进一步提升效果。

---

## 240. When Your State Estimator Has Lost The Plot: Detecting Estimator Failures Via Spectral Analysis

**arXiv ID:** 2608.10623 | [PDF](https://arxiv.org/pdf/2608.10623v1)

**作者:** Christian Lanegger `[一作]` (ETH Zurich), Michael Pantic `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种对多传感器融合状态估计器的无传感器无模型的自我检测方法，利用估计速度在频域上的功率谱分布来判断估计器是否失效。

**💡 创新点**

创新点在于：①将高频功率内容作为通用失效指标，适用于视觉、激光雷达、毫米波雷达等不同感知模态；②将估计器视作黑盒，仅需其IMU+外部传感器融合的速度输出即可；③通过自监督阈值选择，实现无失效标注即可获得可靠检测。

**🔧 技术方法**

使用技术包括：Welch功率谱估计、频域指标（总功率、谱中心、谱带宽、谱熵）、阈值与切频选择（精确率-召回率曲线、F1最大化、相对阈值法）以及精确率召回率、PR-AUC等评估指标。

**📊 数据集**

数据集为户外无人机飞行数据，配备LiDAR、雷达与摄像头；同时使用三种估计器（RoVIO、DLIO、RIO）产生速度序列；通过估计器共识与手工标注得到失效标签；另外使用包含地面真值的RoVIO数据集进行验证。

**📈 对比分析**

与传统协方差、基于行为的异常检测等方法相比，本文方法在三种估计器上平均PR‑AUC达0.64，单个估计器调参后可提升至0.73；在不使用失效标注、仅依赖健康数据时仍能保持类似性能；检测成功率约为50‑58%（召回），精确率在61‑84%之间，尤其在高频阈值为10Hz时表现最佳。

**⚠️ 局限性**

局限性包括：①对平滑慢漂移或完整传感器失效缺乏明显高频特征，导致检测率低；②阈值和切频对估计器和速度高度依赖，需针对不同系统调参；③标注过程受限于估计器共识和手工检查，易出现误标，尤其在失效过渡期；④数据集不均衡（失效样本稀少、低速偏倚），影响阈值选择和泛化能力。

---

## 241. Beyond Detection Accuracy: Measuring Explanation Cost, Stability, and Utility for Resource-Aware IoT Intrusion Detection

**arXiv ID:** 2608.10349 | [PDF](https://arxiv.org/pdf/2608.10349v1)

**作者:** Abdurrahman Tolay `[一作]` `[通讯]` (Independent Researcher), Abdurrahman Tolay (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对CICIoT2023数据集进行泄漏安全处理后，系统性评估了四种机器学习模型（Logistic Regression、Decision Tree、Random Forest、XGBoost）的二分类检测性能、TreeSHAP解释成本与稳定性，并在不同工作负载和阈值下研究了选择性解释的成本‑覆盖折衷。

**💡 创新点**

首次将预测效果、解释成本、局部解释稳定性和选择性解释四维度统一考量，并提出了基于验证集误报阈值的自适应解释触发策略；同时构建了精确去重、标签冲突消除的泄漏安全版CICIoT2023语料库。

**🔧 技术方法**

使用Logistic Regression、Decision Tree、Random Forest、XGBoost四种模型；TreeSHAP（针对树模型）进行解释计算；在特定的微扰（保持预测类别和置信度变化≤0.05）下评估解释稳定性；基于阈值触发的选择性解释策略。

**📊 数据集**

CICIoT2023（共约20.6M个唯一39维特征样本，二分类攻击/正常）

**📈 对比分析**

比较方法：在自然（攻击占比高）和均衡两种测试分布下，采用F1、PR‑AUC、误报率、模型大小、训练/推理时间、TreeSHAP时间及其与推理的比率；稳定性使用Top‑5 Jaccard、Spearman、Cosine、L1变化。结果显示：XGBoost在所有预测指标上表现最佳；Random Forest误报率最低；Decision Tree解释成本最低、模型体积最小；XGBoost在解释吞吐量与成本上最优；Random Forest在稳定性上最好。

**⚠️ 局限性**

局限性：仅评估单一二分类数据集；所有计算均在实验机上完成，未在边缘/IoT设备上验证；仅使用TreeSHAP，未覆盖其他解释器；未进行多类别攻击族评估；未进行人类评估或安全操作验证；稳定性评估仅在有限的微扰范围内；未对不同硬件平台进行跨平台性能对比。

---

## 242. FormStruct-Bench:A Hierarchical and Diagnostic Benchmark for Table-Form Document Structure Recognition

**arXiv ID:** 2608.10396 | [PDF](https://arxiv.org/pdf/2608.10396v1)

**作者:** Lujie Ban `[一作]` (Chinese University of Hong Kong), Chenhao Ma `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向表单式文档的分层结构识别基准（FormStruct-Bench），并通过模板化生成与验证流程构建了可追溯、可验证的7,000条实例数据；

**💡 创新点**

①基于分层结构（页面→语义区域→局部网格→字段→控件组→关系）构建诊断级别评估；②采用三代理生成+验证（Director–Artist–Verifier）实现规模化且可追溯的数据生成；③将评估拆分为页面级、路径级、组件级并提供结构特定诊断；

**🔧 技术方法**

分层结构建模、模板化数据生成与几百人验证、LLM辅助验证、宏观/细粒度指标计算（Schema‑nTED、Value‑nED、TSR‑path、R‑F1@0.5、LIG‑F1、LG‑GriTS_Top、WG‑F1、Rel‑F1）以及难度/视觉退化切片分析；

**📊 数据集**

FormStruct‑Bench：70个可重用模板，扩展成7,000条经过验证的实例（包含多语言、多脚本），测试集1,100条人工复核；

**📈 对比分析**

与14个API托管/本地部署系统以及2个SFT版本进行对比；文档级性能最高可达83.85%（Value‑nED）/57.45%（Schema‑nTED），但细粒度结构指标最高仅17.91%（TSR‑path），展示出内容读取与结构绑定的显著差距；

**⚠️ 局限性**

①测试集仅覆盖中等区域复杂度（4–6区），对极低/高区数的泛化未覆盖；②多组件指标整体偏低，说明现有模型对区域定位、网格与关系重建仍不足；③基于模板的合成数据可能缺乏真实文档的随机变异，导致迁移性能受限；

---

## 243. FiGuRO: Intrinsic Dimension Estimation for Multi-Modal Data

**arXiv ID:** 2608.10857 | [PDF](https://arxiv.org/pdf/2608.10857v1)

**作者:** Viktoria Schuster `[一作]` (Massachusetts Institute of Technology), Caroline Uhler `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为 FiGuRO 的动态自适应框架，用来在单模态和多模态数据中估计数据的内在维度（ID），并在多模态场景下同时学习共享子空间与专属子空间的维度。

**💡 创新点**

创新点：①将低秩分解与速率‑失真理论相结合，形成基于重构精度的双向秩优化算法；②不需要额外的对齐或正则化损失，信息分离（共享与专属）成为优化的自然结果；③在一次训练循环中即可同时完成 ID 估计和子空间分解。

**🔧 技术方法**

技术手段：低秩可分解层（类似 LoRA/ARR）对瓶颈权重进行 SVD；利用 R²（或其他失真指标）作为重构精度；基于速率‑失真理论的阈值 λ 控制秩增减；实现动态更新的循环算法；对多模态 autoencoder 进行共享/私有子空间拆分。

**📊 数据集**

使用的数据集：
- 生成式模拟数据（四种子集，含不同共享/私有 ID 组合）；
- 真实多模态数据：Audio MNIST（图像+音频）、So2Sat（SAR+光学影像）和 NYU Depth V2（RGB+深度）。

**📈 对比分析**

对比方法与性能：
- 单模态：PCA、MSE、MLE、TwoNN、Rank Reduction Autoencoders、ARD‑VAE 等传统与神经网络 ID 估计器；
- 多模态：JIVE、AJIVE、SLIDE、ShIndICA 等矩阵分解方法；
- 结果显示：FiGuRO 在所有模拟子集上平均误差最低（约 2.9 维差），在真实数据上也实现了更准确的共享/专属 ID 估计，并在下游分类任务中达到或超过基线的准确率，证明了其鲁棒性与可解释性。

**⚠️ 局限性**

局限性：
- 需要足够表达力的基础模型，否则 ID 估计受限；
- 结果依赖于重构失真指标的选择，某些情况下可能低估 ID；
- 在高度相关或信息泄漏的子空间中，分离效果不保证；
- 对失真阈值 λ、能量阈值 γ 的敏感性需手动调参；
- 目前主要验证于图像/音频/遥感/深度等场景，其他领域仍待进一步检验。

---

## 244. A HamNoSys-Guided Dataset and Baselines for Fine-Grained Isolated Handshape Recognition in Sign Language

**arXiv ID:** 2608.10588 | [PDF](https://arxiv.org/pdf/2608.10588v1)

**作者:** Ushnish Sarkar `[一作]` (Variable Energy Cyclotron Centre), Tapas Samanta `[通讯]` (Variable Energy Cyclotron Centre)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了基于HamNoSys的平衡手势识别数据集，并提供了基准模型；

**💡 创新点**

首次使用官方HamNoSys 4手势图表定义160个细粒度手势类别，并设计了参与者重叠与不重叠两种评估协议，构建了可复现且跨语言可用的资源；

**🔧 技术方法**

采用RGB图像分类模型ResNet-18和ViT‑B/16，以及基于手部关键点的图卷积网络GCN和XGBoost；使用MediaPipe提取手部关键点，进行图像增强和归一化；

**📊 数据集**

自制144,000张RGB图像（139,199用于建模）来自15名大学生；对比LSWH100（synthetic 100类）和ASL Fingerspelling Dataset A（24类）进行匹配模型评估；

**📈 对比分析**

在参与者重叠的subject‑dependent评估中，ViT‑B/16取得最高top‑1 86.20%；在leave‑one‑subject‑out（LOSO）下，top‑1仅约45%，表明模型对未见参与者的泛化受限；相对外部数据集表现更好，但受类别数量和采集条件差异影响；

**⚠️ 局限性**

样本来源仅限15名大学生、单手静态手势、未包含动态、双手、位置或动作信息；采集环境单一RGB相机，缺乏多样化的人口和环境；仅评估四种模型，未探索更复杂或更鲁棒的模型。

---

## 245. Immersive Micromanipulation Integrating Pipette and Injector Operations with McKibben-Based Haptic Sensations for Workload Reduction

**arXiv ID:** 2608.10033 | [PDF](https://arxiv.org/pdf/2608.10033v1)

**作者:** Kenta Yokoe `[一作]` (Nagoya University), Tadayoshi Aoyama `[通讯]` (Nagoya University)

**通讯引用:** 2301 | [OpenAlex ID](https://openalex.org/A5089806952)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种沉浸式微操纵系统，整合了细胞操作管与注射器的运动、吸取与排出，并通过 McKibben 人工肌肉实现触觉反馈

**💡 创新点**

首次将吸取/排出与触摸感知统一到手部追踪界面，并使用柔性织物 McKibben 伺服器传递可感知的吸取、排出和接触振动

**🔧 技术方法**

采用虚拟现实 HMD、LeapMotion 手势追踪、光学显微镜、可调焦距镜头、电磁镜、McKibben 织物致动器

**📊 数据集**

实验使用6名无经验操作者在微珠传递任务中收集完成时间、NASA‑TLX工作量评分与问卷数据

**📈 对比分析**

相较传统操作，沉浸式无触觉下完成时间下降30–40%，加入吸取/排出+接触触觉后 NASA‑TLX下降约15分，问卷中相关易用性得分显著提升；无明显额外时间损失

**⚠️ 局限性**

样本量有限，未在真实卵母细胞上验证；McKibben 响应延迟约0.1–1s；系统需在不同细胞尺寸和弹性下重新标定；未考察长期使用或多任务性能

---

## 246. The CASE Framework: A Multi-Disciplinary Control Architecture for Governing Enterprise Agentic AI

**arXiv ID:** 2608.10153 | [PDF](https://arxiv.org/pdf/2608.10153v1)

**作者:** Srinivas Telukunta `[一作]` (Cornell University), Lucio Baron `[通讯]` (AI71)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了面向自主 AI 的四层治理框架 CASE，并结合控制理论、复杂适应系统、监督控制学与运维工程，构建跨层耦合模型、机制清单、成熟度评估工具，并通过三项公开数据实证验证其有效性。

**💡 创新点**

创新点包括：①精准匹配治理学科与四个代理尺度；②提出零接触部署悖论及跨层耦合条件；③设计非补偿性成熟度指数与评估表；④用三项公开数据实证验证框架，首次量化“Emergence Gap”。

**🔧 技术方法**

采用了控制理论公式、复杂系统的自组织临界性与传播阈值、监督控制的必要多样性定理、SRE 误差预算与故障注入、OpenTelemetry 语义观测、统计置信区间与 Cohen's κ 等技术。

**📊 数据集**

使用的公开数据集包括：AI 事故数据库、MIT AI 风险库、供应商后门与回溯报告、工具目录与文档、企业案例与招聘信息。

**📈 对比分析**

通过双重编码与交叉验证计算 Cohen's κ 为 0.83/0.64，工具覆盖率显示 L2 缺口；成熟度评分表明 35 家企业部署均处于 L0，表明现有治理偏向 L1/L4 缺失 L2，验证了跨层耦合与零接触部署悖论的预测。

**⚠️ 局限性**

局限性在于：样本偏向公开事件可能低估 L3/L4 失效；工具评估仅基于文档而非实测；模型参数如行为熵估算难度大；缺乏在真实企业部署环境中的进一步验证。

---

## 247. BREAD: Baseline-Referenced Explanations for Anomaly Diagnosis

**arXiv ID:** 2608.10587 | [PDF](https://arxiv.org/pdf/2608.10587v1)

**作者:** Jiaqi Qiu `[一作]` (University of Amsterdam), Inez M. Zwetsloot `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于基线参考的可解释方法BREAD，用于AI驱动的前瞻性异常检测的诊断；

**💡 创新点**

创新点在于将正常基线信息融入LIME的采样与权重核，理论证明在异常远离基线时能够实现稀疏且高可信度的特征重要性；

**🔧 技术方法**

技术主要包括基线参考采样、双点权重核、加权岭回归替代LIME的局部线性代理；

**📊 数据集**

使用合成高维AR(1)序列与真实电梯能耗日数据（16部电梯）进行实验；

**📈 对比分析**

与传统LIME和基于KernelSHAP的基线方法对比，BREAD在多种场景下的faithfulness、robustness与稳定性均明显优于对手，且计算成本与LIME相近；

**⚠️ 局限性**

局限在于只给出点估计，未量化解释不确定性；只使用单一基线，难以处理多模态正常状态；以及未实现进一步的特征筛选以提升可操作性。

---

## 248. A second-order theory of texture for depth from focus

**arXiv ID:** 2608.10411 | [PDF](https://arxiv.org/pdf/2608.10411v1)

**作者:** Sreekar Ranganathan `[一作]` (Carnegie Mellon University), Ioannis Gkioulekas `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出第二阶纹理理论并证明在常规光学相机上通过窄带光谱滤波器可以显著提升基于纹理的被动深度估计（depth‑from‑focus）。

**💡 创新点**

创新点在于将波光学中的主观散斑视为第二阶纹理，并展示其在被动无纹理场景中可被放大、检测，突破传统纹理对深度恢复的限制。

**🔧 技术方法**

技术手段包括：薄透镜成像模型下的波光学散射解析、泊松‑高斯噪声建模、基于峰值对比度的焦距测量、实验使用FLIR机器视觉相机＋Canon 60 mm f/2.8宏镜头＋530 nm窄带滤波器进行焦平面堆栈采集。

**📊 数据集**

数据集为多场景室内/室外自然光（阳光或标准灯光）下的真实拍摄焦平面堆栈；未使用公开数据集，而是自行收集实验图像。

**📈 对比分析**

通过鲁棒z‑score和误差率对DFF性能进行评估；实验显示窄带滤波器（10 nm）在相同曝光下将误差率从约30%降低至10%以下，曝光增加4倍时已显著提升，表明第二阶纹理能显著提高深度精度。

**⚠️ 局限性**

局限性：对半透明表面或方向性光照宽阔（如大灯、阴天）时散斑对比度衰减；使用滤波器需要相应增加曝光时间；在高光照或饱和条件下需注意动态范围限制。

---

## 249. Compute-Optimal Is Not Cluster-Optimal: Systems-Aware Scaling for Sparse Mixture-of-Experts

**arXiv ID:** 2608.10605 | [PDF](https://arxiv.org/pdf/2608.10605v1)

**作者:** Soumajyoti Sarkar `[一作]` (Amazon AGI Foundations), Sheng Zha `[通讯]` (Amazon AGI Foundations)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 MOSAIC 框架，能够在给定集群和训练窗口的前提下，联合搜索稀疏 Mixture‑of‑Experts（MoE）模型的架构、训练数据量和并行布局，实现系统感知的规模化。

**💡 创新点**

创新之处在于将稀疏 MoE 的扩展律与系统性能模型（MFU、通信成本、内存）结合，证明仅靠计算最优稀疏度不存在内点最优，并引入硬件可交付模型 FLOPs 约束以及混合整数非线性优化求解。

**🔧 技术方法**

技术上先拟合四维稀疏 MoE 扩展律（总参数、稀疏度、专家分割因子、训练令牌），随后构建基于微基准的 MFU 与内存预测模型，并在 MOSAIC 中进行离散几何枚举与并行布局搜索，求解一个 MINLP。

**📊 数据集**

实验使用了大规模文本预训练数据，覆盖从 1.04×10⁸ 到 2.7×10⁹ 活跃参数、79 B 总参数的训练日志；数据来源为公开文本语料（如 Common Crawl / WikiText 等）。

**📈 对比分析**

对比传统仅基于计算 FLOPs 的最优配置，系统感知最优配置在相同 GPU‑小时预算下实现了 0.031 nats 的损失下降，MFU 提升至约 12%；实测跑步验证了 MFU 排序与预测一致，并展示了损失在模型 FLOPs 与硬件成本轴上的翻转。

**⚠️ 局限性**

局限性包括结果高度依赖特定的几何阶梯、硬件平台和训练堆栈；未考虑批量大小与并行度的联合优化；专家分割因子仅限到 8；对非识别系数不传播不确定性，且在更大规模或不同任务上需进一步验证。

---

## 250. Decodable But Not Detachable: Training Data Granularity Determines Parametric Modularity in Large Language Models

**arXiv ID:** 2608.10214 | [PDF](https://arxiv.org/pdf/2608.10214v1)

**作者:** Marcus Armstrong `[一作]` (University of Houston), Arjun Mukherjee `[通讯]` (University of Houston)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型在不同领域粒度上是否形成可被削弱的域特定参数子集。

**💡 创新点**

揭示域粒度决定参数模块化，语言域出现高选择性神经集合，学科域则无；并量化这些子集的功能载荷。

**🔧 技术方法**

利用因果掩蔽与激活选择性度量，计算跨域损伤矩阵并分析空间分布。

**📊 数据集**

使用MMLU、ARC-Challenge、OpenBookQA、GSM8K、WikiText-2、MBPP、OPUS-100等公开基准。

**📈 对比分析**

对比随机掩蔽和特定域掩蔽，发现语言域损伤矩阵对角化程度可达595:1，代码域掩蔽使数学推理准确率下降16–24个百分点。

**⚠️ 局限性**

局限于1.5B–7B规模的指令调优模型，可能无法推广至更大或基准模型；激活幅度法可能遗漏其他重要神经。

---

## 251. On Understanding, Identifying, and Mitigating Vulnerabilities in Agentic Large Language Models

**arXiv ID:** 2608.10530 | [PDF](https://arxiv.org/pdf/2608.10530v1)

**作者:** Md Jafrin Hossain `[一作]` (Florida International University), Nirwan Ansari `[通讯]` (New Jersey Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 2023‑2025 年间关于 Agentic LLM 安全的 743 条记录进行 PRISMA 2020 规范的系统综述，最终纳入 85 篇论文。

**💡 创新点**

首次提出基于四层架构（感知、脑、行动、交互）的 13 类漏洞体系，并通过定量分析揭示攻击与防御比例 3.9:1、感知层研究占比 65.9%、行动层仅 4.7% 等研究不平衡，明确七大开放问题。

**🔧 技术方法**

采用系统检索策略、双人编码与 Cohen κ 验证、定量统计与可视化（热图、堆叠图）等方法，对文献进行编码、合成与量化。

**📊 数据集**

主要利用 IEEE Xplore、ACM DL、arXiv、Scopus、Web of Science 与 Google Scholar 六大数据库进行检索；在综述中引用了现有安全基准（如 AgentDojo、AgentHarm、AgentSecurityBench 等），未创建新数据集。

**📈 对比分析**

通过对攻击与防御文献数量、层级覆盖率、漏洞频次等指标进行定量对比，表明攻击研究远多于防御研究，感知层研究占优，行动层研究不足；这些统计指标反映了研究焦点偏向表面攻击、深层安全措施薄弱。

**⚠️ 局限性**

存在研究覆盖偏向感知层、缺乏对代码执行、实体化代理和单机代理等高风险场景的安全分析；检测方法仍不成熟，缺乏标准化评测；多数研究仅在实验室环境，缺少真实部署验证。

---

## 252. ResonaVis: Visualizing Interactive Music Data to Support Reflective Music Composition for Therapeutic Contexts

**arXiv ID:** 2608.10338 | [PDF](https://arxiv.org/pdf/2608.10338v1)

**作者:** Abhishek Karwankar `[一作]` (University of Delaware), Matthew Louis Mauriello `[通讯]` (University of Delaware)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并评估了 ResonaVis，一款可视化分析仪表盘，帮助音乐治疗作曲家基于儿童 ASD 交互日志与音频特征进行创作决策。

**💡 创新点**

创新点在于：①将儿童在 uCue 上的细粒度交互记录与多模态音频特征统一可视化；②通过多视图（时间轴、Sankey、协方差矩阵、频谱等）实现跨模态推理；③通过共创与实验验证，证明可视化提升作曲家自信与创意。

**🔧 技术方法**

技术实现：使用 Plotly + Dash 构建 Web 仪表盘；交互日志与音频特征分别转化为 Sankey 图、协方差矩阵、时间序列、波形、Mel 频谱等可视化；支持联动过滤、比较与导出。

**📊 数据集**

数据集：约 4 小时 uCue 交互日志（6 名 ASD 儿童、12 条可视化层级）+ 10 条对应的音频轨道，提取了音量、谱质心、频谱滚降等 20+ 维音频特征。

**📈 对比分析**

评估方法：混合研究——8 名音乐学生完成三项设计挑战并填写 SUS、NASA‑TLX 与自信量表；另外 2 名经验作曲家进行观察案例；结果显示 SUS 约 70 分、NASA‑TLX 低至中等负荷，且所有自信维度显著提升（p < 0.05）。

**⚠️ 局限性**

局限性：样本规模小（n=7+2），缺乏多元参与者；数据量有限，仅覆盖少数音频特征；未进行临床疗效评估；可视化仍需更友好的学习资源，尤其对非技术音乐人。

---

## 253. STCAD: Scalable Trajectory Clustering and Anomaly Detection on Terabyte-Scale AIS Data

**arXiv ID:** 2608.10249 | [PDF](https://arxiv.org/pdf/2608.10249v1)

**作者:** Bertram Hage `[一作]` (Technical University of Denmark), Peder Heiselberg `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个可扩展的无监督框架，用于在10^12级别的AIS海事轨迹数据上进行轨迹聚类和异常检测。

**💡 创新点**

创新点在于将自监督的BERT变压器编码器与CURE层次聚类相结合，并利用重建误差和聚类噪声标注实现内在异常检测，且不需要预先设定簇数。

**🔧 技术方法**

采用的技术包括BERT式Transformer（掩码词建模）、CURE层次聚类、UMAP降维、WARD连接、重建对比比率（RCR）评估、分段正态化与线性插值等预处理方法。

**📊 数据集**

使用丹麦海事管理局2024年全年AIS数据集，总计约1.2TB、6.89亿条消息、38,026艘船舶。

**📈 对比分析**

与单连锁、平均连锁和WARD聚类对比后，CURE+WARD在不预设簇数的情况下得到稳定聚类，噪声率1.4%、RCR 1.5，能够明显区分正常与异常航行轨迹。

**⚠️ 局限性**

局限性包括：预处理耗时高、对计算资源（64核、4GB内存/核）依赖显著、排除了稀疏的静态特征、对极少数细粒度异常仍可能检测不足。

---

## 254. Evidence-Based Scientific Question Discovery: A Framework with Historical Backtesting

**arXiv ID:** 2608.09968 | [PDF](https://arxiv.org/pdf/2608.09968v1)

**作者:** Hui Mao `[一作]` `[通讯]`, Hui Mao

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `14d48e9d-0069-4ad9-996a-1d5968216998` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一套从科学文献中提取证据、检测并类型化跨文献矛盾，然后生成可验证的研究问题并按优先级排序的完整流程。

**💡 创新点**

创新点在于将问题发现正式化为可追溯、可复现的步骤，采用人类裁定的矛盾类型化、两阶段排名（科学优先级与执行优先级分离），并实现历史回测评估。

**🔧 技术方法**

技术手段包括基于抽象的主张抽取、证据图构建、人工判定矛盾类型、机器学习评分模型以及基于嵌入的检索验证。

**📊 数据集**

使用的数据集为2020年前的 exoplanet 大气学文献（NASA ADS 与 arXiv）、NASA Exoplanet Archive 物体目录及 MAST 的 JWST/HST 观测记录。

**📈 对比分析**

通过历史回测比较：在2020年止的数据生成10个问题，随后 2021–2026 年的文献全部对其进行实质性回应，其中2个被回答、1个被反驳，最高排名的问题仍保持开放，表明较高的关注度与成功率。

**⚠️ 局限性**

局限性包括仅在天体物理领域验证、仅使用摘要抽取主张、裁定者为单一专家、样本量有限以及验证评审可能存在模型偏差。

---

## 255. Exponentially Consistent Low Complexity Tests for Statistical Sequence Matching

**arXiv ID:** 2608.10455 | [PDF](https://arxiv.org/pdf/2608.10455v1)

**作者:** Lin Zhou `[一作]` `[通讯]` (Southern University of Science and Technology), Lin Zhou (Southern University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对统计序列匹配问题，提出了低复杂度、指数一致性的固定长度和序贯检测方法。

**💡 创新点**

创新点在于利用GJS散度（离散序列）和MMD度量（连续序列）构造评分函数，仅需计算所有序列对的评分值即可识别匹配对，并在已知和未知匹配数两种情形下均给出了指数误差下界。

**🔧 技术方法**

主要技术包括：信息论中的GJS散度与MMD度量、方法论（类型方法、McDiarmid不等式）、序贯检验的阈值设计和指数误差分析。

**📊 数据集**

本文未使用公开真实数据集，实验均基于合成离散与连续序列进行仿真验证。

**📈 对比分析**

与传统全局穷举搜索方法（指数复杂度）相比，本文方法在保持可接受的指数误差下，计算复杂度降至多项式级，且在序贯版本中进一步降低错误概率，实验显示在相同误差阈值下运行时间显著下降。

**⚠️ 局限性**

局限性包括：假设匹配序列来自完全相同的分布，未给出逆定理证明最优性；未知匹配数时误差率受阈值选择影响；序贯检验假设所有序列同步可用，未考虑更灵活的抽样策略。

---

## 256. Stay or Stray - A Dynamical Systems Viewpoint of Popularity Bias

**arXiv ID:** 2608.10474 | [PDF](https://arxiv.org/pdf/2608.10474v1)

**作者:** Sarvesh Shashidhar `[一作]` (Indian Institute of Technology, Bombay), Tanmay Khandelwal `[通讯]` (Amazon Music)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对推荐系统中出现的受欢迎度偏差（popularity bias）进行研究，构建了包含两类用户（主流与小众）的耦合动态系统，使用两时间尺度随机逼近方法将模型参数更新与用户到达率分离，推导出连续时间 ODE，分析平衡点与收敛性，得出偏差出现的阈值与对称保留的充分条件，并在合成数据和真实音乐推荐平台日志上进行验证；提出通过平衡两类错误率的“Equalised Odds”策略来减轻偏差。

**💡 创新点**

① 使用两时间尺度随机逼近对推荐算法与用户参与度的耦合动态进行理论解析，首次给出偏差出现的阈值与对称保留的可行区间；② 在此框架下得到平衡点的完整解析与稳定性分析；③ 通过理论与实验相结合，验证模型在真实大规模数据上的有效性；④ 提出了基于错误率平衡的自然缓解策略。

**🔧 技术方法**

持续在线二分类器（MSE 损失下的梯度下降）、两时间尺度随机逼近、连续时间 ODE 分析、正交投影、矩阵分解（MF）用于生成用户特征、仿真模拟、热图/相位图可视化。

**📊 数据集**

① 合成的高斯特征数据；② 约 410M 条用户–项目交互日志（来自大型商业音乐推荐平台）经过矩阵分解得到用户与项目嵌入，用于估计用户特征分布并验证理论。

**📈 对比分析**

将理论阈值与实际系统演化轨迹进行对照；在不同 p、μ、Σ、λ 等参数设置下进行 100 次 Monte‑Carlo 仿真；对真实日志进行 MF 拟合后跑 100k 步，观察收敛点；实验表明理论预测与仿真/实测结果高度吻合，缓解策略能显著降低小众用户流失率。

**⚠️ 局限性**

① 仅考虑二分类用户，无法直接推广至多类场景（虽在补充材料中给出扩展）；② 假设用户特征服从高斯分布，且参数更新遵循两时间尺度（若用户行为变化快则假设失效）；③ 需要预先知道类别比例与特征均值，实际应用中估计误差会影响阈值计算；④ 仅在 MSE 损失下得到闭式结果，其他损失需进一步验证。

---

## 257. Flow Straight to Reality: Perceptually Consistent Flow Matching for Efficient Image Restoration

**arXiv ID:** 2608.10544 | [PDF](https://arxiv.org/pdf/2608.10544v1)

**作者:** Sangwoo Jo `[一作]` (Korea University), Sungjoon Choi `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 PCFlow，利用潜在一致流匹配和感知一致损失，在潜在空间中直接学习从降质图像到清晰图像的连续传输，实现少步推理的高质量图像恢复。

**💡 创新点**

①将感知一致损失与流匹配结合，形成“潜在一致感知损失”以在轨迹上引导语义一致性；②提出冲突无关梯度投影策略，在低SNR阶段消除结构与感知目标之间的梯度冲突；③采用轻量级卷积‑仅 U‑Net 结构，显著降低模型参数和推理时间。

**🔧 技术方法**

潜在一致流匹配（Latent Consistency Flow Matching, LCFM），潜在一致感知损失（Latent Consistency Perceptual Loss, LCPL），冲突无关梯度投影，SNR 自适应 λ 调度，卷积‑仅 U‑Net 变换器。

**📊 数据集**

FFHQ 512×512（BFR 训练）与 CelebA‑Test、LFW‑Test、CelebAdult；FFHQ 256×256（其余任务训练）并在 CelebA‑Test 评估。

**📈 对比分析**

与多阶段扩散方法（DifFace、DiffBIR、PMRF）、单阶段流匹配方法（ELIR、PMRF）以及现有超分/去噪/填补/上色模型对比。PCFlow 在 BFR 上实现 FID 35.89、NIQE 3.95、FPS 70.35，参数 32M；在其他任务上保持 FID 下降、PSNR/SSIM 稳定、LPIPS 下降，且参数仅 21M、推理速度显著提升。整体显示出更优的失真‑感知平衡与计算效率。

**⚠️ 局限性**

①仍可能在高频细节处出现细微伪影；②需预热期和梯度投影，训练调参较复杂；③在极端降质或非人脸场景的通用性尚未充分验证；④仅关注图像恢复，未探究生成任务的适用性。

---

## 258. Calibrating Post-Training Feature Shifts for LLM Data Contamination Detection

**arXiv ID:** 2608.10462 | [PDF](https://arxiv.org/pdf/2608.10462v1)

**作者:** Zhen Yang `[一作]` (University of New South Wales), Wenjie Zhang `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对大型语言模型后期训练导致的特征漂移的校准框架，以提升黑盒数据污染检测的鲁棒性。

**💡 创新点**

创新点在于：①使用多视图漂移检测（Multi‑View Shift Detection）从多种受控查询视角识别与后期训练相关的、对假阳性率影响显著的特征漂移子空间；②提出有界特征校正（Bounded Feature Correction），在保持有用检测信息的前提下，按需抑制漂移方向的特征分量；③通过自适应选择校正参数，进一步提升效果。

**🔧 技术方法**

技术主要包括：受控查询视角（controlled prompt views）、假阳性压力（FPP）排序、跨视图共识投影、奇异值分解、投影投影平均、线性有界校正矩阵以及对校正后的特征重新训练分类器。

**📊 数据集**

实验使用四个公开的污染检测基准：BookTection、BookMIA、ArxivTection、WikiMIA，分别包含不同来源的文本样本。

**📈 对比分析**

与两种现有特征基检测器（VeilProbe、DPDLLM）以及三种后期训练过的LLM（Qwen、Llama、DeepSeek）进行对比。实验表明，在所有24个设置中，校准方法平均提升AUC 2.1%（最高7.0%）和TPR@5%FPR 7.0%（最高15.0%）。

**⚠️ 局限性**

局限性包括：只校正在已知非成员样本上观察到的、导致假阳性的漂移；需要外部非成员样本；校正仅为线性，可忽略非线性或样本特异性的错误；校准过程需要额外模型查询，并依赖受控视角能否充分暴露相关漂移。

---

## 259. MAD-HOI: Masked Autoregressive Diffusion for Generating Articulated Hand Object Interactions from Text

**arXiv ID:** 2608.10162 | [PDF](https://arxiv.org/pdf/2608.10162v1)

**作者:** Ananya Bal `[一作]` (Carnegie Mellon University), Laszlo A. Jeni `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种从文本生成手物交互(HOI)序列的方法 MAD-HOI，支持单手/双手、复合动作、补全、填充及自动终止。

**💡 创新点**

创新点在于将连续潜在空间与掩码自回归扩散结合，保持手物轨迹解耦、支持可变长度生成并学习 EOM 预测；同时通过连续潜在避免 VQ 量化导致的接触失真。

**🔧 技术方法**

使用了连续 VAE 对手和物体运动进行解码，掩码自回归 Transformer 生成条件，流匹配扩散头完成潜在采样，并采用 CLIP 文本编码进行对齐。

**📊 数据集**

在 ARCTIC（单关节可动物体）和 GRAB（全身手物交互）两大公开数据集上进行训练和评测。

**📈 对比分析**

与 Text2HOI、DiffH2O、LatentHOI、OpenHOI、HOIGPT 等基线相比，MAD-HOI 在检索准确率、FID、KID、匹配得分等生成指标上均取得最高或接近最高；在几何和物理可行性指标上，在 ARCTIC 上位居前列，在 GRAB 上保持竞争力。

**⚠️ 局限性**

主要局限包括：对手部离散化仍存在一定误差；在 GRAB 长序列中 EOM 预测准确率略低；缺乏对机器人硬件的直接适配，仅通过后期运动重定向实现。

---

## 260. The Game of Marginal Utilities

**arXiv ID:** 2608.10373 | [PDF](https://arxiv.org/pdf/2608.10373v1)

**作者:** Isaac M. Sonin `[一作]` (University of North Carolina at Charlotte), Yaakov Malinovsky `[通讯]` (University of Maryland, Baltimore County)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了一类非合作资源分配博弈，证明其存在唯一纳什均衡，并给出了均衡的结构化描述；

**💡 创新点**

创新点在于：① 对博弈的完整解析，得出唯一性与等边际原理；② 证明均衡具有嵌套 cutoff 结构与活动区分解；③ 在完全活跃区得到单一非线性方程的聚合公式；④ 提出全局线性收敛的投影边际效用算法与结构利用的 Block Pandora 重构算法；

**🔧 技术方法**

采用了 Rosen 的凸博弈理论、KKT 条件、变分不等式与投影法、Lipschitz 与严格反单调性分析，以及一维单调方程求解等技术；

**📊 数据集**

论文仅为理论分析，未使用具体实验数据集；

**📈 对比分析**

算法的收敛性通过理论证明给出：投影算法满足显式步长条件下全局线性收敛；Block Pandora 在已知切断结构时可高效重构并验证均衡；未进行数值实验比较；

**⚠️ 局限性**

局限性包括：① 对玩家参数同质性和项目参数顺序性做了强假设，异质化后结构不一定成立；② 仅讨论静态单阶段博弈，缺乏动态或随机扩展；③ 计算复杂度未给出多项式上界，仅提供支持枚举法。

---

## 261. Observational Policy Ranking for SMB Financial Guidance from Multi-Action Accounting Logs

**arXiv ID:** 2608.10050 | [PDF](https://arxiv.org/pdf/2608.10050v1)

**作者:** Shrutendra Harsola `[一作]` (Foresight-AI, Intuit), Sricharan Kumar `[通讯]` (Foresight-AI, Intuit)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

从 85,078 条公司-月会计日志中学习多动作业务变更的财务效果，并通过观察性政策排名为中小企业提供单一业务建议。

**💡 创新点**

提出 Covariate‑Adjusted Residual Policy Learning (CAR‑PL)，即在多动作日志上直接做动作‑wise R‑learning 并加入支持衰减正则化；同时构建统一的模型辅助评分框架，保证不同方法可直接比较。

**🔧 技术方法**

使用 R‑learning、T‑learner、保守价值模型（CQL 风格）、多标签多热编码、交叉验证的倾向性与结果基线模型、梯度提升回归头、支持衰减、束缚的增量式评分等技术。

**📊 数据集**

基于 7,505 家中小企业的财务报告与 34 类业务变更标签，构成了包含 Gross Profit、Revenue、Quick Ratio 三个 KPI 的 85,078 条公司‑月样本。

**📈 对比分析**

在公司分层的训练/验证/测试拆分中，所有方法共享相同的冻结评分器和 34 类动作空间，使用公司级别的 bootstrap 计算置信区间。CAR‑PL 与 T‑learner 在 Gross Profit 与 Revenue 上均为最高点估计且无显著差异，CAR‑PL 覆盖更广；Quick Ratio 上 Contextual Value Model 领先。Zero‑shot LLM、modal constant 与随机参考在所有 KPI 上均低于学习型方法。

**⚠️ 局限性**

依赖观测性数据，假设已观测的前置协变量充分；共现动作仅通过焦点处理，未考虑完整组合策略；评分器基于拟合的倾向性与结果模型，可能受模型误差影响；未做前瞻性部署验证，未评估实际业务效果。

---

## 262. FADE: From Passive Verification to Active Discovery in Counterfactual Video Understanding

**arXiv ID:** 2608.10764 | [PDF](https://arxiv.org/pdf/2608.10764v1)

**作者:** Fufangchen Zhao `[一作]` (Beijing University of Posts and Telecommunications), Danfeng Yan `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FADE框架，通过证据内化的SFT与衰减锚点的RL两阶段训练，使视频多模态LLM能在无文本提示下自主发现并解释反事实事件，并推出anchor‑fading评估协议；

**💡 创新点**

创新点在于（1）将证据投影与响应约束回归相结合的SFT与逐步衰减文本锚点的RL策略融合；（2）设计anchor‑fading评估协议，能在不收集新数据集的前提下将现有MCQ基准拆分为MCQ、OQA、captioning三种形式；（3）显著提升模型在去掉文本提示后的保留率，突破传统MCQ准确率的局限；

**🔧 技术方法**

使用证据投影与响应约束回归（RCER）、基于奖励标准化的GDPO RL、LoRA微调、BF16精度训练、anchor‑fading的progressive curriculum以及自监督的文本生成补充；

**📊 数据集**

基于DualityVidQA（训练集104,879条，测试集）与IPV‑Bench的原始MCQ样本，补充OQA答案与caption以供FADE训练；

**📈 对比分析**

采用anchor‑fading协议在DualityVidQA‑test和IPV‑Bench上评估，FADE在Strict Paired分数上达到84.6%/76.5%/57.0%，保留率分别为90.4%和67.4%，明显优于GPT‑5.6等基线，且在OQA和captioning上表现更稳健；

**⚠️ 局限性**

局限性包括：①SFT仍依赖人工标注的时间区间；②在极长视频或多事件场景中的鲁棒性尚未验证；③评估协议虽然减少文本锚点，但仍未完全消除潜在的语言偏差。

---

## 263. Toward Human Rights Benchmarking for LLMs: A Pilot Methodology

**arXiv ID:** 2608.10268 | [PDF](https://arxiv.org/pdf/2608.10268v1)

**作者:** Savannah Thais `[一作]` (Hunter College), Caitlin Kraft Buchman `[通讯]` (AI and Equality)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了人权法律推理评估基准HumRightsBench，针对LLM在人权法上的推理能力进行验证。

**💡 创新点**

创新点在于引入IRAP框架（Issue, Rule, Application, Proposed Remedies）并与人权法的义务结构相结合，构建专家验证的情景式评测集。

**🔧 技术方法**

使用IRAP任务分解、结构化问题生成、规则回忆与应用、建议补救措施，并采用结构化输出接口进行评分。

**📊 数据集**

使用基于国际人权条约、一般评论、专门程序报告等权威文本的情景与子情景，聚焦先期的“饮水权”场景。

**📈 对比分析**

通过与三大旗舰LLM（GPT‑5、Claude Opus 4.7、Gemini 3）和一个开源模型Qwen 3.5‑9B的五次随机种子评估，整体准确率在0.339–0.577之间，其中最难为规则应用任务。

**⚠️ 局限性**

局限在于样本规模小、覆盖仅限饮水权、规则应用任务设定过高阈值、缺乏多语言与多维度验证，需进一步扩展情景与评测方式。

---

## 264. Rethinking Text-Based Image Retrieval in Specific Domain

**arXiv ID:** 2608.10524 | [PDF](https://arxiv.org/pdf/2608.10524v1)

**作者:** Jingyang Tan `[一作]` (Harbin Institute of Technology), Lanpeng Jia `[通讯]` (Changhong Intelligent Robot)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套面向特定领域（如监控）的多匹配文本图像检索框架和基准，包含数据构建引擎DSMM-TBIR和SecMM-TBIR数据集，并提出Semantic‑Aware Fine‑Tuning (SAFT) 方法提升检索性能。

**💡 创新点**

创新点在于：①引入多匹配检索范式，解决单匹配标签导致的评价偏差；②设计DSMM-TBIR三阶段数据引擎，利用LLM/ VLM 自动生成多匹配查询与候选；③提出SAFT框架，结合Semantic‑Aware Soft‑Label Supervision (SASS) 与Intra‑modal Structural Distillation (ISD) 有效缓解特定域内的假负样本问题。

**🔧 技术方法**

技术手段包括：生成式大语言模型与视觉语言模型的提示式查询生成；多模型投票检索与人工筛选相结合的候选过滤；对比学习中的对比损失 + SASS 的跨模态软标签对齐 + ISD 的视觉结构蒸馏；在CLIP‑类模型上进行细调。

**📊 数据集**

使用数据集：1）通用多模态数据（Flickr30K、MS‑COCO）做预训练；2）内部监控专用数据集（包含行人与车辆）；3）SecMM‑TBIR基准，50k监控图像 + 200个综合查询，支持多匹配标签。

**📈 对比分析**

与传统ITC（对比学习）和CUSA（基于单模态软标签的对齐）相比，SAFT在SecMM‑TBIR上平均提升mAP@20 7.8点；在Flickr30K/COCO等通用基准上也有显著提升（最高可达+5.4/10.3点）。实验覆盖多种CLIP‑类模型，验证方法稳健且可推广。

**⚠️ 局限性**

局限性包括：①数据生成与筛选仍需人工确认，仍存在标注成本；②SAFT在极端语义压缩的场景下对超参数敏感；③仅在监控与通用两个域验证，尚未对更细粒度或多模态（视频）任务展开测试。

---

## 265. Detecting Soft Skills in ML Engineering Roles CVs

**arXiv ID:** 2608.10046 | [PDF](https://arxiv.org/pdf/2608.10046v1)

**作者:** Aidin Azamnouri `[一作]` (Technical University of Munich), Stefan Wagner `[通讯]` (Technical University of Munich)

**通讯引用:** 8183 | [OpenAlex ID](https://openalex.org/A5041829889)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用生成式 LLM 对 300 条 ML 相关职位简历进行软技能抽取，区分显式（关键词列表）与隐式（叙述性）表达，并检验候选人侧的软技能呈现与雇主侧需求之间的差异。

**💡 创新点**

①首次在候选人简历中同时抽取显式与隐式软技能并进行区分；②通过假设检验与效应量控制多重比较，系统性检验雇主侧关于领导力普适性等假设在候选人侧的验证；③构建可复现的基于 LLM 的信息抽取管道。

**🔧 技术方法**

使用 Gemini 3 Pro 生成式模型结合 TaxoSoft 软技能词汇表；定制提示、正向与负向示例；统计检验包括 Fisher、χ²、Logistic 回归、McNemar；多重校正采用 Holm‑Bonferroni 与 Benjamini‑Hochberg。

**📊 数据集**

平衡 300 条公开可获取的简历（GitHub CV 语料库 + 2025 年公开搜索），按角色（ML 工程师、数据科学家、软件工程师）各 100 条，按经验年限（≥5 年）划分为高级与初级。

**📈 对比分析**

与传统关键词匹配相比，LLM 抽取 F1 ≈ 0.72、精度 0.69、召回 0.74；在 13 条预设假设中 11 条得到支持，1 条部分支持，1 条被驳回；效应大小中等偏大，置信区间不跨零，且经 Holm 校正后结论稳健。

**⚠️ 局限性**

样本受人工标注限制，规模有限；仅包含公开英文简历，可能不具备行业外的代表性；抽取误差虽已校正，但对极少见软技能的捕捉有限；研究聚焦于 ML 相关职位，结论在其他技术领域的推广需谨慎。

---

## 266. A Pragmatic Guide to Building Conservative Discrete Abstractions of Cyber-Physical Systems

**arXiv ID:** 2608.10254 | [PDF](https://arxiv.org/pdf/2608.10254v1)

**作者:** Jordan Peper `[一作]` (University of Florida), Ivan Ruchkin `[通讯]` (University of Florida)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一个四步的保守抽象流程，用来将连续动力学系统转换为有限状态模型，并在此基础上进行符号模型检查。

**💡 创新点**

创新点包括：①保守-by-construction 的抽象框架；②提供多种转移关系构造子（轴对齐盒子、凸多边形、PAC 采样）；③自环消除与 CEGAR 去除伪转移；④LTL 规范提升的 May‑Must 翻译。

**🔧 技术方法**

采用了轴对齐盒子、凸多边形覆盖、PAC 采样覆盖、符号模型检查、CEGAR 迭代细化以及 LTL 语义提升等技术。

**📊 数据集**

使用了三类案例数据集：合成线性系统、Gymnasium 的 MountainCar-v0 以及自行实现的无人车模型。

**📈 对比分析**

通过自环比例、成功率 (TPR)、构建时间和验证时间等指标进行对比；实验显示采样方法显著降低自环比例、提升 TPR，CEGAR 在 MountainCar 上进一步提升 TPR，未对无人车产生明显影响。

**⚠️ 局限性**

局限性包括：对大规模或高度非线性系统仍可能导致状态爆炸；自环消除和 CEGAR 对平稳或小步长区域难以验证；采样方法缺乏确定性保证；未考虑不确定或随机动力学。

---

## 267. Threshold-Based Spiking Neural Networks for Event-Driven Status Update Systems

**arXiv ID:** 2608.10640 | [PDF](https://arxiv.org/pdf/2608.10640v1)

**作者:** Marco Fries `[一作]` (Vienna University of Technology), Andrea Ortiz `[通讯]` (Vienna University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种针对事件驱动状态更新系统的阈值型传输决策方法。

**💡 创新点**

证明了存在最优阈值策略，并基于此设计了轻量级的脉冲神经网络（SNN）来逼近最优策略。

**🔧 技术方法**

利用MDP建模、阈值策略证明、SNN架构与梯度强化学习（Policy Gradient）以及与ANN比较的能耗分析。

**📊 数据集**

使用仿真数据集，参数包括传输成功概率0.9、唤醒概率0.5、最大AoI 16s 等。

**📈 对比分析**

与最优阈值策略、ANN、随机、始终/永不传输策略比较，SNN能耗低于ANN，性能与最优阈值策略接近。

**⚠️ 局限性**

局限性在于仅考虑伯努利唤醒过程，阈值最优性可能不适用于更一般的唤醒模式，且仅在仿真环境中验证。

---

## 268. To EFX OR to MMS, That is the Question

**arXiv ID:** 2608.10397 | [PDF](https://arxiv.org/pdf/2608.10397v1)

**作者:** Hadi Hosseini `[一作]` (Pennsylvania State University), Rohit Vaish `[通讯]` (IIT Delhi)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在不可分物品分配中，将EFX（对任意单个物品的公平性）和MMS（最大最小份额）这两种公平性概念按“每个代理人任选其一”的方式组合的新公平性定义，并探讨其存在性与算法可行性。

**💡 创新点**

创新点包括：①首次证明在子模子集和子模成本下，即使只有三位代理人，EFX∨MMS也可能不存在（提供八件子模商品与七件子模杂务的最小反例）。②在加性混合物品（含正负价物品）且最多三种价值类型（其中一种仅有一位代理人）时，永远能找到EFX∨MMS分配，并给出高效的分割与重组算法；③在相同加性评价的情形下，进一步证明了EFX∧MMS（两者同时满足）的可行性。

**🔧 技术方法**

核心技术包括：使用SMT求解器Z3搜索子模反例；构造“结构性障碍”来简化反例验证；通过几何递减秩变换保证严格单调与子模性；设计阈值保持的分配子程序，利用EFX与MMS的性质在加性场景中进行分区重组；使用cut-and-choose与Leximin++等经典分配工具。

**📊 数据集**

本文未使用公开数据集，而是自行构造的小型物品集合（如八件商品或七件杂务）和人工设计的价值表，用以展示反例与算法的可行性。

**📈 对比分析**

与以往仅讨论EFX或MMS单独存在性的研究相比，本文证明了在更宽松的混合公平性下，既能保持存在性，又能在特定结构下给出多项式或伪多项式算法；实验比较主要通过理论证明与构造反例完成，性能指标以时间复杂度（多项式/伪多项式）与分配质量（满足至少一项公平性）为准。

**⚠️ 局限性**

局限性包括：①对子模子集和子模成本的存在性仍是负结果，且只在三代理人下给出；②正向结果仅适用于加性混合物品且价值类型受限（最多三种且其中一类单数）；③未探讨更一般的公平性组合或多代理人下的近似/效率权衡；④对于非加性情况的算法效率和可扩展性尚未给出。

---

## 269. Threat-guided Policy-aware Scene Perturbation for Safe Autonomous Driving with Online Reinforcement Learning

**arXiv ID:** 2608.10403 | [PDF](https://arxiv.org/pdf/2608.10403v1)

**作者:** Xincong Hu `[一作]` (Nanjing University), Zongzhang Zhang `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种威胁导向、策略感知的场景扰动框架TPSP，用于在在线强化学习中提升自动驾驶的安全学习效率。

**💡 创新点**

创新点在于将当前策略的表征嵌入场景编码，针对策略弱点生成定向扰动，并以策略rollout的威胁差值作为闭环优化信号；从而实现针对性、安全性更高的经验生成。

**🔧 技术方法**

使用策略感知场景编码（注意力聚合+policy投影）、Gaussian扰动网络、威胁评估（TTC、距离、碰撞等）、PPO强化学习以及熵正则化的优化框架。

**📊 数据集**

在GPUDrive仿真环境下使用NAVSIM v2数据集进行训练和评估。

**📈 对比分析**

与Vanilla PPO、随机扰动、TPSP无策略感知等基线对比，TPSP在NAVSIM v2 Stage1/2的NC、TTC指标分别提升至99.8%/96.7%，并且在相同交互预算下学习速度显著加快。

**⚠️ 局限性**

局限在于扰动空间仅涵盖对象速度、生成时间与ego初始状态，未扩展至更丰富的语义/地图/多智能体扰动；同时缺乏在真实车辆上的验证与对更大规模仿真环境的测试。

---

## 270. Exploring Semantic Stability Across Reviews in the Linux Kernel

**arXiv ID:** 2608.10101 | [PDF](https://arxiv.org/pdf/2608.10101v1)

**作者:** Lucas Ciziks `[一作]` (Universidade de São Paulo), Marco Aurélio Gerosa `[通讯]` (Northern Arizona University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了Linux IIO子系统中补丁系列的功能级语义稳定性，跟踪每个函数在多轮审阅过程中的变化。

**💡 创新点**

首次提出基于功能级语义相似度的测量框架，并揭示了近乎满分相似度主要由未被编辑的函数造成的偏差。

**🔧 技术方法**

使用UniXcoder预训练代码编码器的余弦相似度，以及基于AST的结构距离作为对比。

**📊 数据集**

数据集来自LKML5Ws的Linux内核邮件列表，提取了64,044条功能级记录，形成10,117条可比轨迹。

**📈 对比分析**

比较方法包括：(a) 与随机无关函数对的相似度基准，(b) 按编辑量分桶的M1检验，(c) 端点相似度M2和连续转移相似度M3。结果显示端点平均相似度0.997，编辑后平均相似度0.990，差异虽显著但效果尺寸极小。

**⚠️ 局限性**

局限在于整个函数的向量平均化无法捕捉局部细微改动，导致高相似度无法区分重要修复与无关修改；同时轨迹匹配方法未经过充分验证，存在误链接风险。

---

## 271. Shape optimisation of nonlinear Naghdi shells on discrete geometries

**arXiv ID:** 2608.10342 | [PDF](https://arxiv.org/pdf/2608.10342v1)

**作者:** Ado Farsi `[一作]` (Imperial College London), Alberto Paganini `[通讯]` (University of Leicester)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文实现了面向薄壳结构的形状优化框架，针对L型铸件和半圆柱壳在几何非线性条件下通过最小化位移实现刚度提升。

**💡 创新点**

创新点在于将微分几何方法与基于拉格朗日乘子法的自适应多层控制空间相结合，并引入Helmholtz平滑度度量，使得非线性薄壳的形状梯度可直接计算并保持光滑，可在内部自由变形域实现大幅度刚度增益。

**🔧 技术方法**

采用的技术包括：几何非线性Naghdi薄壳模型、共轭梯度求解器、Adjoint微分法求形状梯度、分层多网格控制空间、Helmholtz平滑度量、有限元离散化与网格细化策略。

**📊 数据集**

使用的“数据集”主要是几何参数（L、H、R、t）、材料常数（E、ν）以及加载和形状位移预算等模拟输入参数，并未引用公开数据集。

**📈 对比分析**

通过与COMSOL“Shape Optimization of a Shell”基准结果以及Abaqus的前向响应对比。对L型铸件的能量减小率为87%（COMSOL为89%），对半圆柱的位移能量降低率为91%，平均位移下降78%，性能与基准相近且仅增加4.1%的表面积，表明方法具有良好的优化效果。

**⚠️ 局限性**

局限性包括：仅在对称问题下测试，未验证非对称负载或更复杂几何；控制空间与状态空间的映射依赖手工设定的粗细网格和对称平面条件；计算成本高，特别是在几何非线性下的多步加载和网格细化；对大变形极限（形状位移预算）敏感，未探讨更高自由度的形状约束。

---

## 272. Physics-Informed Machine Learning in Prognostics and Health Management: A Systematic Literature Review

**arXiv ID:** 2608.10047 | [PDF](https://arxiv.org/pdf/2608.10047v1)

**作者:** Christopher Braun `[一作]` (University of Stuttgart), Marco F. Huber `[通讯]` (University of Stuttgart)

**通讯引用:** 3783 | [OpenAlex ID](https://openalex.org/A5031354877)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统文献综述，对PIML在PHM中的应用进行全面梳理，梳理了212篇研究，提出了四类分类框架并对不同PHM任务进行分析，

**💡 创新点**

创新点在于首次将观察偏差、归纳偏差、学习偏差与混合方法四类体系化地划分并系统评估其在PHM领域的实际效果，

**🔧 技术方法**

采用系统检索（Scopus、Web of Science、arXiv）、ASReview主动学习筛选、质量筛选以及四类分类方案来整理与评估研究，

**📊 数据集**

数据来源为文献数据库中的212篇论文（涵盖电池、轴承、齿轮、燃料电池等多种资产），并未使用传统实验数据集，

**📈 对比分析**

通过对比文献中的基线结果和方法，显示PIML在大多数任务上均优于纯数据驱动或纯物理模型的性能，但并未提供统一的跨研究评估基准，

**⚠️ 局限性**

局限性包括对特定资产（如锂离子电池、轴承）的过度聚焦，证据不足以完全支持所有优点，缺乏统一的实验或公开数据集进行跨方法验证，且系统综述的搜索时间窗口有限。

---

## 273. From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models

**arXiv ID:** 2608.10444 | [PDF](https://arxiv.org/pdf/2608.10444v1)

**作者:** Si'an Xie `[一作]` (Beijing University of Posts and Telecommunications), Ming Wu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MPAR-Bench，用于衡量LLM在多点联想推理（推理广度）上的能力。

**💡 创新点**

创新点在于：①基于游戏“Just One”设计的多线索推理任务；②采用多代理生成+嵌入过滤+人工验证的流程，生成语义多样、低冗余的线索集；③细粒度评估框架，包括准确率、ANLS、嵌入相似度和推理轨迹验证；④四轴扰动测试验证鲁棒性。

**🔧 技术方法**

使用了多代理合作生成线索、Qwen3-Embedding-8B嵌入过滤、fastText嵌入相似度、手工评审、思考模式（Thinking）等技术。

**📊 数据集**

构建了双语（英中）1,000条样本的数据集，覆盖英语和中文语义多样的线索与目标词。

**📈 对比分析**

通过与多款主流LLM（GPT-5.2、Gemini-3.1pro、Sonnet-4.5、Qwen3-max等）在标准和增强设置下的准确率、ANLS、嵌入相似度和推理轨迹指标进行比较；结果显示最佳模型在英语约86.8%/中文约72.2%，扰动下准确率下降9–18/5–12点。

**⚠️ 局限性**

局限性包括：1) 仍然存在“过度思考”导致答案被覆盖的现象；2) 对扰动的鲁棒性不均匀，模型和语言依赖明显；3) 规模扩大、思考模式或反馈机制只能获得有限提升，表明当前训练范式未能充分优化推理广度。

---

## 274. Mitigating Bus Bunching with Reinforcement Learning Enhanced by Semantic Stop Embedding

**arXiv ID:** 2608.10207 | [PDF](https://arxiv.org/pdf/2608.10207v1)

**作者:** Xin Dong `[一作]` (Pennsylvania State University), Vikash V. Gayah `[通讯]` (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种离线LLM辅助的停靠语义嵌入，结合事件驱动的深度Q学习实现公交持停控制，并在同线路与跨线路实验中验证其有效性。

**💡 创新点**

创新点在于：①将停靠的物理、功能与历史运营信息通过LLM统一转换为固定语义向量；②在RL状态中加入语义向量，既提升单线路控制效果，又为跨线路迁移提供共享特征空间；③采用离线推理避免实时LLM调用，保持控制实时性。

**🔧 技术方法**

使用的大型语言模型（LLM）对停靠信息进行语义标注，文本嵌入模型与PCA降维生成低维语义向量；基于事件驱动的MDP框架，采用深度Q网络（DQN）进行强化学习；对比传统Daganzo规则与无控制基线。

**📊 数据集**

基于宾夕法尼亚州立大学校园两条巴士线路（White Loop、Blue Loop）的AVL车辆定位数据、乘客到达速率、站点周边POI（OpenStreetMap）及站点历史运营统计。

**📈 对比分析**

通过与无控制、Daganzo规则、仅使用间距信息的RL、以及使用停靠ID的RL进行对比；在同线路实验中，语义RL在头way方差、乘客等待时间与持停时间上均优于其他方法；在跨线路实验中，零射转移改善了部分指标，微调后提升早期学习速度，冷启动最终性能最佳。

**⚠️ 局限性**

局限性：仅在两条校园巴士线路的仿真环境中验证，缺乏对更大规模、异质化公交网络的实测评估；未系统评估不同交通需求波动、突发事件下的鲁棒性；需要进一步量化LLM推理与嵌入生成的实际数据与计算成本。

---

## 275. FUSE: Frame-Unified Stress Estimation from Facial Video

**arXiv ID:** 2608.10442 | [PDF](https://arxiv.org/pdf/2608.10442v1)

**作者:** Stefanos Gkikas `[一作]` (Honda Research Institute Japan), Giorgos Giannakakis `[通讯]` (Hellenic Mediterranean University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种能够一次性处理整段面部视频的压力检测框架FUSE，避免传统的时间窗口切分；

**💡 创新点**

创新点在于将时间维度折叠到通道维度，利用非对称注意力机制统一处理任意长度的完整视频；

**🔧 技术方法**

使用的技术包括轴折叠、傅里叶位置编码、跨注意力与自注意力层的组合，以及高维通道的非对称注意力网络；

**📊 数据集**

实验基于58名受试者的120秒面部视频数据集，包含多种压力诱发任务，采用二分类（中性vs压力）；

**📈 对比分析**

与传统窗口切分方法相比，FUSE在stride=15时达到69.44%的测试准确率，stride=1时仅略低（69.03%），在计算成本和推理延迟上可通过增大stride实现显著节省；

**⚠️ 局限性**

局限性包括仅在单一数据集和二分类任务上验证，未与窗口化基线做直接对比，且未考察多模态融合的潜在优势。

---

## 276. Can Bayesian Optimization Efficiently Find a Strong Single Expert in Neural Thickets?

**arXiv ID:** 2608.10867 | [PDF](https://arxiv.org/pdf/2608.10867v1)

**作者:** Nigel Bastian Cendra `[一作]` (University College London), Jakob Zeitler `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究使用贝叶斯优化在低维随机线性子空间中搜索单一专家模型。

**💡 创新点**

创新点在于用高斯过程先验引导搜索，显著降低评估成本，实现5倍候选数节省的性能。

**🔧 技术方法**

采用贝叶斯优化、Gaussian Process、随机线性嵌入以及无梯度的后训练评估。

**📊 数据集**

使用 Countdown、GSM8k 与 MATH500 三个推理基准以及 Qwen2.5‑Instruct 的 0.5B–3B 模型。

**📈 对比分析**

与随机搜索和 RandOpt 进行对比，BO 在 K=1 时匹配或超过 RandOpt 的 1000 候选，并在多数任务上提升测试准确率，但在多数任务中提升有限。

**⚠️ 局限性**

局限在于选择集饱和、过度拟合、子空间限制、跑间方差大以及仅在小预算下验证，缺乏更大规模验证与多子空间搜索。

---

## 277. Neuroevolution Arena: Nested Ecological Evaluation of Update-and-Inheritance Regimes across Neural Architectures

**arXiv ID:** 2608.10323 | [PDF](https://arxiv.org/pdf/2608.10323v1)

**作者:** Yuxu Ge `[一作]` (University of York), Yifei Cheng `[通讯]` (University of York)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了GPU加速的异质细胞神经网络生态系统Neuroevolution Arena，并通过层分区的更新-遗传方案（EvoEvo、EvoRL、RLRL）与两种网络架构（Baseline64和Wide128）对训练与冻结生态评估的影响进行对比。

**💡 创新点**

创新点在于提出了可并行的多细胞神经网络生态平台、层分区的演化/强化学习机制，以及嵌套的冻结评估协议，揭示训练与生态评估中的排名差异并提供了可审计的评估流程。

**🔧 技术方法**

使用技术包括神经进化（NEAT、遗传算法）、强化学习（自定义RL更新）、GPU并行计算、化学信号扩散、二维格子自组织及细胞运动模型。

**📊 数据集**

使用的数据集与模型公开托管在 HuggingFace，分别为 https://huggingface.co/datasets/geyuxu/alife2026-data 与 https://huggingface.co/geyuxu/alife2026-model。

**📈 对比分析**

评估方法为三层次：①在 3×2 训练实验（共 18 条件，50,000 代）下记录训练适应度；②冻结评估共 198 作业，包括 135 对战、9 六方对决和 54 存活实验；③对比训练适应度与冻结 AUC、六方获胜率及生存阈值。结果显示 RL 方案在训练中适应度最高，但冻结对比中排名依赖于网络架构与保存的 artifact，未出现单一绝对冠军。

**⚠️ 局限性**

限制在于：①冻结评估仅使用同一 run‑index 的 artifact，未覆盖全部 artifact 交叉；②环境上下文混合为 2:1 的合作/攻击比例，未实现完全平衡；③未保存完整生态状态，只存储单细胞控制器；④生存阈值过严格导致全 0 的主测结果，未能区分条件差异；⑤仅研究两种网络架构，未包含 Hybrid64‑128 与 NoSkip64 等。

---

## 278. Bridging Event Streams and DiT: Event-Guided Video Frame Interpolation

**arXiv ID:** 2608.10479 | [PDF](https://arxiv.org/pdf/2608.10479v1)

**作者:** Guixu Lin `[一作]` (University of Tokyo), Yinqiang Zheng `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过提取事件摄像机产生的稀疏事件流、图像扭曲事件（IWE）和双向光流，作为轻量级适配器注入预训练的Diffusion Transformer（DiT）视频生成模型，实现高质量的视频帧插值。

**💡 创新点**

①基于适配器的框架，仅对少量参数进行LoRA微调，无需从头训练；②将事件信息转换为可兼容Diffusion模型的控制信号（IWE+双向稀疏光流）；③构建大型合成事件-视频数据集EvPexels。

**🔧 技术方法**

事件处理采用Contrast Maximization（CMax）获取光流和IWE；IWE编码器与光流对齐融合适配器；LoRA微调；基于FLF2V（Wan2.1）的DiT视频扩散模型；光流Warp与融合。

**📊 数据集**

使用实测高速度RGB‑事件数据集BS‑ERGB、合成的EvPexels（1,100场景约390k帧），以及DAVIS、Pexels等公开测试集。

**📈 对比分析**

在×24插值任务下，使用PSNR、SSIM、LPIPS、FID、FVD等指标与RIFE、TRF、GI、ViBiD、FCVG、Wan2.1 FLF2V、TimeLens、CBMNet‑Large、TimeLens‑XL、VDM‑EVFI‑Wan2.1等基线进行对比；在BS‑ERGB上获得LPIPS/FID/FVD第一，PSNR/SSIM排名第三/第二；在DAVIS与Pexels上取得最高的PSNR、SSIM、LPIPS、FID、FVD，显著优于所有对比方法。

**⚠️ 局限性**

方法对事件采样频率高度依赖，光流估计在噪声较大或极端高速场景下可能失真；仅在少量事件‑视频配对上微调，缺乏对真实无标注事件数据的无监督推广；适配器训练仍需事件-视频数据，无法完全做到零样本。

---

## 279. Comprendia: AI-Augmented Code Comprehension

**arXiv ID:** 2608.10290 | [PDF](https://arxiv.org/pdf/2608.10290v1)

**作者:** Costain Nachuma `[一作]` (Idaho State University), Minhaz F. Zibran `[通讯]` (Idaho State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个 Eclipse 插件，将依赖图可视化、LLM 辅助代码解释、克隆检测与重构建议以及 CVE 风险叠加整合到同一图形视图，并实现了 Graph-Aware Callee Pruning 算法来挑选合适的调用者。

**💡 创新点**

① 基于可见依赖图的可审计 callee 选择，保证解释上下文可追踪；② 通过继承层次折叠减少重复调用者；③ 将克隆检测直接叠加在同一图上，实现一次性视图；④ 统一的图基子结构支持多种 LLM，保证跨模型可重复性。

**🔧 技术方法**

Eclipse JDT AST 解析、Cytoscape.js 图可视化、Graph-Aware Callee Pruning 算法、CloMan 克隆检测器、OSV.dev CVE 查询、LLM 接口（Anthropic Claude、OpenAI GPT、Ollama Llama）以及 JGit 版本控制。

**📊 数据集**

quickbite Java benchmark（33 类，6 包），其中包含手工注入的 k=2、3、4 的克隆组以及已知 CVE。

**📈 对比分析**

与 Selection‑only（仅文本选择）和 Legacy（固定 cap 5 调用者）两种基线在同一 6 个代码片段上比较，使用相同 token 预算计算提示词长度。GACP 在无折叠场景下平均多 19% 调用者，token 使用比 Legacy 少 20%；在折叠场景下 token 几乎相同并可节省 2.7%。算法输出在三种 LLM 上完全一致，说明跨模型性能稳定。

**⚠️ 局限性**

仅在单一 benchmark 评估；评价指标为粗粒度的关键词匹配；未进行真实开发者理解实验；缺乏个性化与方法级调用图；与嵌入检索基线未对比；未调优参数与不同图细粒度等。

---

## 280. Evaluating Shrinking (Experience Report)

**arXiv ID:** 2608.09935 | [PDF](https://arxiv.org/pdf/2608.09935v1)

**作者:** Alperen Keles `[一作]` (University of Maryland), Leonidas Lampropoulos `[通讯]` (University of Maryland)

**通讯引用:** 497 | [OpenAlex ID](https://openalex.org/A5075217645)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对Haskell三个主流Property‑Based Testing框架（QuickCheck、Hedgehog、Falsify）在ETNA平台四个工作负载上的shrinking行为进行系统评估，提出了量化指标（树编辑距离、缩小时间及每单位进度成本）并对不同生成器族（基于类型、API、构造正确）进行对比；

**💡 创新点**

创新点在于：①首次将shrinking效果与成本独立量化，并使用基于枚举的最小化结果作为真值；②扩展ETNA评测平台以支持shrinking测量；③提供跨框架、跨生成器族的实证对比，揭示结构性shrinking在多数场景下仍具优势；

**🔧 技术方法**

主要技术包括：属性测试与随机生成、外部/内部shrinking实现、树编辑距离计算（zss库）、ETNA平台的实验脚本、LeanCheck的穷举最小化、时间和进度统计、统计显著性检验（Friedman、Wilcoxon等）

**📊 数据集**

使用ETNA的四个Haskell工作负载：Binary‑Search‑Tree、Red‑Black‑Tree、Simply‑Typed Lambda Calculus、System F_<：；每个工作负载提供若干注入变异与对应属性，形成多任务评测集合；

**📈 对比分析**

比较方法：在每个工作负载上分别运行三种框架、三种生成器族，记录bug‑发现时间、shrinking时间、树编辑距离与最小值的差距；使用bucket chart展示bug‑发现速度，使用累计计数图（CCP）展示shrinking效果，使用对数坐标的时间/进度曲线评估效率。结果显示QuickCheck在多数情形下最快且shrinking质量接近或优于集成shrinking；集成shrinking在某些生成器族和工作负载上表现略优，整体差异不显著；

**⚠️ 局限性**

局限性包括：生成器实现偏向QuickCheck，可能不公平；测量依赖强制严格执行，真实环境可能不同；树编辑距离对对称实例敏感且受枚举策略影响；默认缩小预算未统一跨库；缺乏真实调试时间的用户研究。

---

## 281. PolyLayout: Hierarchical VLM-Guided Layout Generation Beyond Rectangular Rooms

**arXiv ID:** 2608.10838 | [PDF](https://arxiv.org/pdf/2608.10838v1)

**作者:** Yutong Jiang `[一作]` (IKEA Retail (Ingka Group)), Shahin Shahkarami `[通讯]` (IKEA Retail (Ingka Group))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套分层的家具布局生成系统 PolyLayout，能够在任意多边形房间中生成物理可行的 3D 物品布局。

**💡 创新点**

将 VLM 引导的宏观布局与确定性几何求解器分离，支持非矩形房间和门窗约束，并通过功能聚类提升布局连贯性。

**🔧 技术方法**

使用 Vision‑Language 模型（VLM）进行语义规划、规则驱动的功能聚类以及基于规则的几何求解器，并结合检索增强生成（RAG）来生成设计方案。

**📊 数据集**

在 IKEA Retail 的真实家具目录（约 44 组库存）和多种真实房间（3 个矩形、3 个带门窗、5 个非矩形）上进行评测。

**📈 对比分析**

与 Holodeck 与 LayoutVLM 进行对比，PolyLayout 在完好率、边界合规率均 100% 的同时，感知可行性得分最高（2.74/4），且 CPU 端时延仅 70 s，远优于 LayoutVLM 的 907 s。

**⚠️ 局限性**

仅评估单房间布局，缺乏多房间或全屋实验；聚类规则手工制定，扩展到更多功能域仍需人工规则；对开放式 VLM 约束的基线缺乏对比。

---

## 282. DINO-A: Adapting Self-Distillation Vision Transformers to General Audio Representation Learning

**arXiv ID:** 2608.10659 | [PDF](https://arxiv.org/pdf/2608.10659v1)

**作者:** Tomasz Radzikowski `[一作]` (Warsaw University of Technology), Przemysław Rokita `[通讯]` (Warsaw University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

将Canonical DINO自蒸馏框架迁移到通用音频领域，推出DINO-A模型并在FSD50K上进行预训练，随后在ESC‑50、Speech Commands v2、UrbanSound8K和GTZAN上做线性探针评估。

**💡 创新点**

首次在相同条件下将DINO与BYOL‑A v2直接对比，研究高维投影空间、多crop对音频的影响，并系统分析不同patch尺寸和模型族（ViT vs CNN）在各类任务中的表现差异。

**🔧 技术方法**

采用自蒸馏（student‑teacher）框架、EMA教师、交叉熵损失、高维投影空间、BYOL‑A v2音频增广（Mixup‑BYOLA、RandomResizeCrop、RunningNorm）以及ViT（8×8/16×16）与CNN（AudioNTT2022）骨干。

**📊 数据集**

预训练使用FSD50K（约41k条无标签音频），评估在ESC‑50、Speech Commands v2、UrbanSound8K和GTZAN四个标准分类基准上进行线性探针评估。

**📈 对比分析**

在完全相同的预训练数据和评估流程下，DINO‑A平均落后BYOL‑A v2 11.96个百分点；ViT 8×8在所有任务中表现最佳；CNN在语音任务上优于ViT，而在环境声和音乐任务上显著落后。

**⚠️ 局限性**

局限性：预训练规模仅限FSD50K，导致高维投影空间无效；多crop在音频中假设不稳健；未进行AudioSet规模预训练、完整微调或针对音频的增广与非方形patch探索。

---

## 283. SparSTAR: Sparse Attention for SpaceTime AutoRegressive Video Synthesis

**arXiv ID:** 2608.10519 | [PDF](https://arxiv.org/pdf/2608.10519v1)

**作者:** Jongbeom Lee `[一作]` (Sogang University), Suk-Ju Kang `[通讯]` (Sogang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

为 InfinityStar 视频自回归生成模型设计一种训练无关的块稀疏注意力机制（SparSTAR），通过动态重算每个尺度的块选择，显著降低晚期尺度的注意力开销，同时保持所有 token 和所有 refinement 级别的完整性。

**💡 创新点**

创新点在于：
• 采用聚合 Q‑K 块评分（block‑wise avg‑dot‑product）实现无参数、无训练的块选择；
• 引入 clip‑aware 策略，区分 Clip 1 与后续 Clip 的保留与可选块，确保文本和前一 clip 最终尺度始终保持密集；
• 设计 forward‑only 的 FlexAttention 执行路径，省略反向索引，减少推理时额外运算；
• 采用非均匀密度调度，在晚期尺度施加更高稀疏度，以最大化计算节省。

**🔧 技术方法**

使用的技术包括：
• 块稀疏注意力（block‑sparse attention）与聚合 Q‑K 评分；
• Spacetime Sparse Attention（SSA）作为背景上下文；
• FlexAttention/forward‑only sparse 执行实现；
• 32‑bit BF16 推理、NVIDIA H100 GPU 与 PyTorch/Triton 库。

**📊 数据集**

评估数据集：
• 720p 5 秒（81 帧）文本‑到‑视频与图像‑到‑视频生成任务；
• VBench 官方评测套件（5 秒 81 帧，16 维度）。

**📈 对比分析**

比较方法：与 Dense InfinityStar、FastSTAR、SparseVAR、FastVAR、ToMe 等加速基线在相同任务、分辨率、提示与种子下进行对比。结果显示：
• 速度提升约 1.6×（T2V）和 1.62×（I2V）
• PSNR 提升 2.92 dB（T2V）和 2.31 dB（I2V）
• VBench 分数差距 ≤ 0.08 分
• 长视频 Clip‑2 仅稀疏时实现 1.31×/1.36× 速度提升，VBench 下降仅 0.87/0.46 分。

**⚠️ 局限性**

局限性：
• 仅稀疏晚期尺度，早期尺度与非注意力模块未受益；
• 未对 KV 缓存压缩或动态大小块适配进行探索；
• 对更长多 clip 生成的全面评估仍待验证；
• 对块大小、密度调度以及与时间一致性交互的敏感性未做系统化研究。

---

## 284. Visual-to-Haptic Augmentation in XR: A Wearable Glove for Perceptual Grounding in Multimodal Interaction

**arXiv ID:** 2608.10368 | [PDF](https://arxiv.org/pdf/2608.10368v1)

**作者:** Faisal Mohd `[一作]` (University of Ottawa), Abdulmotaleb El Saddik `[通讯]` (University of Ottawa)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于特征的视觉到触觉映射算法和配备29个振动执行器的可穿戴手套，能够将XR中的图像和视频转换为分布式触觉反馈，并通过实验验证其在动态内容中的有效性。

**💡 创新点**

创新点在于：①提出了统一的特征提取与融合框架，兼顾运动、边缘与亮度三种视觉特征；②实现了实时（150 ms）触觉映射，无需预先录制的触觉库；③通过四层模块化架构将触觉渲染嵌入现有XR系统，提供可扩展的感知增强层。

**🔧 技术方法**

技术手段包括：Unity实现XR环境；Farnebäck稠密光流、Sobel边缘检测、亮度归一化的特征提取；线性加权融合公式；滑动平均与阈值后处理；使用LRA振动执行器、DRV2605L驱动和Teensy 4.1微控制器实现硬件接口；浏览器仪表板可视化调试。

**📊 数据集**

使用自定义XR内容：静态纹理（玻璃、砂纸）和动态视频（流水、弹跳乒乓球）。没有公开的标准数据集，全部为实验室自制。

**📈 对比分析**

通过受控的 within‑subject 对比实验（N=20）将视觉‑仅与视觉+触觉两种条件进行比较。结果显示：
• 在纹理模块中，两条件的现实感差异无显著意义（p=0.914，d_z=0.03）。
• 在视频模块中，触觉反馈显著提升现实感（p=0.044，d_z=0.48），并在沉浸感、时间同步、触觉‑视觉对应等维度上取得更高评分。

**⚠️ 局限性**

局限性包括：
1. 仅在单用户、离线同步环境下验证，未测试多用户或网络延迟情境。
2. 触觉执行器数量和布局有限，导致静态纹理的空间连续性不足。
3. 仅采用线性特征融合，缺乏对更复杂视觉语义的捕捉。
4. 数据集为自制，缺乏公开可复现的标准基准。

---

## 285. Neural Tree Collaborative Filtering: Rethinking Graph Collaborative Filtering as Tree Collaborative Filtering with Curvature-Aware Propagation Depth

**arXiv ID:** 2608.10297 | [PDF](https://arxiv.org/pdf/2608.10297v1)

**作者:** Jinfeng Xu `[一作]` (University of Hong Kong), Edith C. H. Ngai `[通讯]` (University of Hong Kong)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了神经树形协同过滤（NTCF）框架，将用户-项目交互图视为根节点树，并为每个节点分配基于局部度不平衡的曲率感知传播深度；

**💡 创新点**

核心创新在于通过闭式曲率指标自适应调整每个节点的消息传递层数，解决传统GCF统一传播深度导致的过平滑和信息缺失问题；

**🔧 技术方法**

方法结合了图卷积网络（GCN）卷积、离散Ricci曲率近似、节点级深度掩码和层级拼接/曲率加权平均聚合，使用BPR损失训练；

**📊 数据集**

实验使用了Yelp、Amazon Kindle Store和Pinterest三大公开稀疏交互数据集；

**📈 对比分析**

与NGCF、LightGCN等传统GCF基线以及四种自监督推荐模型比较，NTCF在Recall@K和NDCG@K指标上均有显著提升，且仅增加一层消息传递开销；

**⚠️ 局限性**

局限性在于需要预先计算曲率得分，且在极端稀疏图中曲率阈值的选择对性能影响较大。

---

## 286. Accelerated Learning of High Dimensional Functions with a Tensor-Featured Training Network

**arXiv ID:** 2608.10351 | [PDF](https://arxiv.org/pdf/2608.10351v1)

**作者:** Karl Pierce `[一作]` (University of Maryland College Park), Haizhao Yang `[通讯]` (University of Maryland College Park)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将张量分解特征（Tensor-Featured）嵌入神经网络输入层的两步优化策略，旨在加速高维函数学习的训练过程。

**💡 创新点**

创新点在于将预训练神经网络通过随机化CPD（CPD-RALS）分解成低秩张量，并将其输出作为新特征，引入多种可快速计算的Rank‑1特征，从而在保持模型表达力的同时显著提升训练收敛速度。

**🔧 技术方法**

采用随机化CPD-RALS（利用Leverage‑Score采样的随机ALS）、Tensor‑Train/CPD张量分解、随机特征与高维高斯过程理论、Julia实现的Flux.jl和ITensorCPD.jl框架进行实验。

**📊 数据集**

使用合成的高维PDE解（非线性椭圆方程和波动方程）作为数据集，维度从5到40，训练集随机采样1000点（扩展至10000点），验证集5000点。

**📈 对比分析**

与传统（无特征）训练做对比，实验表明在合适的训练样本量和初始模型接近全局极值时，Tensor-Featured训练在相同epoch数下能显著降低验证集的均方误差，收敛速度提升约20–40%，但在初始模型远离全局最优时改进有限。

**⚠️ 局限性**

局限性包括对预训练模型质量的高度依赖、随机CPD分解在极高维时仍存在计算与存储瓶颈、特征选择仍需经验式指导，且在某些情形下加入特征可能导致模型收敛回传统解。

---

## 287. On $\ell$-rank additive intersection pairs (RAIP) of codes

**arXiv ID:** 2608.10472 | [PDF](https://arxiv.org/pdf/2608.10472v1)

**作者:** Sanjit Bhowmick `[一作]` (Indian Institute of Technology Guwahati), Sihem Mesnager `[通讯]` (University of Paris VIII)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并研究了ℓ-秩加性交叉对（ℓ-RAIP）代码，构建了其理论框架，并给出了必要充分条件、矩阵判定方法以及与ACD、ACP和Hull的关联。

**💡 创新点**

创新点在于将线性ℓ-交叉对推广到加性代码空间，利用字符理论与条目对数映射形成新的判定矩阵；证明任意加性代码在q>3时可单词等价为ACD代码，并给出从自正交、GRS及扩展GRS构造ℓ-RAIP的系统方法。

**🔧 技术方法**

主要技术包括：字符理论、对数映射、单项式矩阵（monomial）等价、对称/斜对称双线性形式、矩阵秩判定以及通过Φ映射将线性代码与加性代码对应。

**📊 数据集**

本文未使用公开数据集，而是以理论构造和符号计算为主；所有示例均采用代数构造的符号参数（如α、β、ξ）。

**📈 对比分析**

评价方法为理论证明与构造示例，未给出实验性能指标；通过构造示例展示所得到的ℓ-RAIP代码可满足给定交叉维数，并且在q>3时可转换为ACD，表明理论的可行性。

**⚠️ 局限性**

局限性：未给出具体码参数（长度、码率、最小距离）和编码/译码复杂度；仅讨论了欧几里得/迹双线性形式，对Hermitian等非对称双线性形式的推广留待后续研究；缺乏实验验证。

---

## 288. Hierarchical Compositionality for An Assistive AI Agent

**arXiv ID:** 2608.10330 | [PDF](https://arxiv.org/pdf/2608.10330v1)

**作者:** Tianyi Fu `[一作]` (University of Edinburgh), Mohan Sridharan `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种基于层级组合性和符号推理的 AI 代理框架，用于在家庭环境中快速、个性化地解决命令歧义。

**💡 创新点**

创新点包括：① 将三层属性-概念层级（原子属性 → 统计构造概念 → 用户工作流程模式）嵌入歧义消解；② 将 ASP 逻辑推理与数据驱动的概念层级结合，实现非单调的可行性过滤；③ 采用三信号（语义兼容、会话显著性、主题偏好）融合，并设置阈值进行不确定性控制。

**🔧 技术方法**

技术栈：ASP（Answer Set Programming）用于推理与规划；WordNet 计算语义相似度；统计 lift 与频率阈值用于概念挖掘；GPT‑5.1 作为 LLM 对比基线；Python + SPARC 等工具实现系统；实验环境为模拟家庭场景。

**📊 数据集**

使用的主要数据集：NOVA 语义特征规范（约 3,983 个属性）用于构建原子属性；实验自制的 66 个实体、12 个动作的模拟家庭环境；5 位用户各 10 次会话的交互历史（约 1500 条行为片段）用于训练和评估。

**📈 对比分析**

对比方法包括随机、LLM+CoT、ASP+LLM、语义+显著、对象级、属性级、概念级以及无 ASP 等 8 种基线；评估指标为整体准确率、答案准确率和澄清率。实验表明，本方法在所有模糊级别下整体准确率最高（A1 74.2%→A4 54.7%），相较 LLM 基线提升 30%+；澄清率最低（≈26%），并在 noask 模式下准确率 62.7%，均显著优于所有基线。

**⚠️ 局限性**

局限性：仅在完全可观测的模拟环境中验证；需要先构建属性/概念层级，迁移到新领域需重构；未评估对大规模实体/动作的扩展；阈值与权重需经验调优；缺乏对非结构化语音/视觉输入的鲁棒性验证。

---

## 289. More Accurate, Less Human: Gestalt Grouping in Vision Models

**arXiv ID:** 2608.10195 | [PDF](https://arxiv.org/pdf/2608.10195v1)

**作者:** Sudhanva Manjunath Athreya `[一作]` (University of Utah), Sai Phani Kumar Malladi `[通讯]` (Siemens AG)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

做了什么：本研究提出了一套基于Gestalt原理的行为学评估基准，利用公开的图形感知实验数据来衡量视觉语言模型在图表感知任务中与人类感知规律的一致性，而非仅关注任务准确率。

**💡 创新点**

创新点是什么：创新点在于把实验心理学中的Gestalt聚类原则和图形感知定量数据转化为可复现的评估标准，使用错误一致性（κ）和人类‑人类一致性上限来量化模型的“人类相似度”，并将此方法扩展至视觉语言模型的图表阅读。

**🔧 技术方法**

用了什么技术：采用零样本余弦相似度读取、最小化拟合探针、文本解析规则进行输出映射，并使用Bootstrap 置信区间计算行为一致性（B）、错误一致性（κ）与效应复制（a(k)）三项指标。

**📊 数据集**

用了什么数据集：重用公开的图形感知实验数据，包括Cleveland‑McGill比例估计、Tableau mark‑color odd‑one‑out、THINGS对象偶数等原始刺激与人类响应，构建了两条轨道（图表与自然图像）上的四个聚类任务。

**📈 对比分析**

如何比较的方法，性能怎么样：对45个模型（15个编码器+30个基础模型）进行行为一致性、错误一致性与效应复制评估，结果显示准确率与人类相似度解耦，部分闭源模型在形状和颜色聚类上达到或逼近人类‑人类一致性上限，而多数模型表现出显著偏离。

**⚠️ 局限性**

limitation是什么：局限性包括仅覆盖两条Gestalt原则、未评估完整图表上下文与轴/图例等效果、仅使用公开实验数据、开源与闭源模型规模难以直接对比、以及未探讨其他分组法则或更广泛的认知理论。

---

## 290. INSIDE the Student's Mind: Jointly Modeling Latent Reasoning and Action in LLM Student Simulators

**arXiv ID:** 2608.10492 | [PDF](https://arxiv.org/pdf/2608.10492v1)

**作者:** Rose Niousha `[一作]` (University of California), Narges Norouzi `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 INSIDE 框架，在大语言模型中同时学习生成学生的内部推理对话和后续代码提交。

**💡 创新点**

创新点在于利用 Bloom 分类的三维结构（认知、情感、行动）生成内部对话，结合教师模型逆推重构推理轨迹，并通过对话引导提升学生行为的真实度。

**🔧 技术方法**

技术包括 LLM 微调（LoRA）、基于 GPT‑5 的教师推理、Qwen2.5/Qwen3/LLaMA 以及 GPT‑5 的提示与微调融合。

**📊 数据集**

使用了加州大学伯克利分校 CS 61A 课程的两学期学生代码提交数据，共 445 名学生的 6 911 次提交作为训练，479 名学生的 6 316 次提交作为测试。

**📈 对比分析**

通过 Wasserstein 距离评估代码功能、复杂度与风格的一致性，使用 GPT‑5‑mini 评估内部对话与代码变更的对齐率；实验显示 INSIDE 在动作真实性上优于提示方法，内部推理对齐率最高可达 57.9%，且整体行为轨迹更贴近真实学生。

**⚠️ 局限性**

局限性包括：推理轨迹为教师模型重建，可能不完全反映真实学生的思考；对齐度虽高但仍低于 100%；不同数据分布导致结果可比性受限；大模型的推理能力不一定能提升与学生相似的非专家推理。

---

## 291. Tree-of-Ideas: Automated Research Ideation via Cross-Trajectory Reasoning over Scholarly Evolution

**arXiv ID:** 2608.10740 | [PDF](https://arxiv.org/pdf/2608.10740v1)

**作者:** Xun Li `[一作]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy), Wenhao Jiang `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了 Tree-of-Ideas（ToI）框架，自动生成科研想法，核心思路是从论文引用网络中重建分支演化树，并在树上追踪研究缺口与技术演进，再通过跨路径推理产生基于多条路径证据的研究方向。

**💡 创新点**

创新点：① 使用分支演化树而非单一线性链来刻画学术演进；② 在树结构中显式标注研究缺口，形成缺口驱动的演化视角；③ 通过跨轨迹共振与交叉链接信号（convergent 与 cross‑link）联合挖掘，为生成的研究想法提供多路径的历史支撑。

**🔧 技术方法**

技术手段：基于 LLM（DeepSeek‑V4‑Flash）实现检索增强与生成；EvoTrace 用关系推理重构缺口中心的演化树；EvoAgent 通过跨路径信号发现与内部审核过滤生成多样化、具备可验证性的研究想法。

**📊 数据集**

数据集：采用 Semantic Scholar API 结合预构建的大型 AI 会议（NeurIPS, ICML, ACL, EMNLP, ICLR, CVPR, AAAI 等）论文的引用图，构成主题中心化的文献空间；实验覆盖六个 AI 研究主题。

**📈 对比分析**

对比方法：Direct Prompting、RAG、CoI‑Agent、AI‑Scientist、ResearchAgent 以及人类已发表论文的参考想法。评估使用三大 LLM 评审器与五位 AI 专家在五维度（Novelty, Significance, Groundedness, Feasibility, Effectiveness）上打分。结果表明 ToI 在所有维度均名列自动化方法首位，平均分 6.27（LLM 评审）/6.62（人类评审），与人类参考差距仅 0.02/0.33，显示出显著优越性能。

**⚠️ 局限性**

局限性：评估仅停留在想法阶段，未进行实验验证，可能导致高分想法在实际实现时遇到困难；同时 ToI 对引用图的完整性和连通性高度依赖，若文献缺失或连接不充分，跨轨迹推理的证据基础将受限。

---

## 292. PERCEPT: A Corpus for POS Tagging and Analysis of Persian-English Code-Mixing

**arXiv ID:** 2608.10109 | [PDF](https://arxiv.org/pdf/2608.10109v1)

**作者:** Ghazal Kalhor `[一作]` (University of Tehran), Behnam Bahrak `[通讯]` (Khatam University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了第一个公开的大规模波斯语–英语代码混合语料库PERCEPT，包含6800条来自X、Instagram和Digikala的帖子，并为代码混合词自动标注了UD词性标签和文档主题。

**💡 创新点**

创新点在于首次为波斯语–英语代码混合文本提供大规模UD词性标注，提出了可迁移的LLM辅助标注框架，并基于此开展了全面的语言学分析。

**🔧 技术方法**

采用Gemini 3.5 Flash LLM进行词性和主题标注，结合平台特定提示与规则过滤，随后通过人工验证评估标注质量。

**📊 数据集**

使用6800条从X、Instagram和Digikala收集的帖子作为主要数据集，并对原始数据做了去敏感处理和语言识别过滤。

**📈 对比分析**

通过与人工金标准的比较，LLM在代码混合词识别上达到95.5%精确率、86.6%召回率、90.8% F1；在词性标注上准确率97%、宏F1 76.8%，整体F1 88.1%，显示标注可靠。

**⚠️ 局限性**

研究局限包括仅聚焦波斯语–英语代码混合，平台有限，标注层面仅覆盖词性和主题，缺乏实体识别、依存句法等更深层次标注。

---

## 293. Amulet: Frame Extrapolation Through Sparse Layered Scene Representation and Adaptive Shading

**arXiv ID:** 2608.10423 | [PDF](https://arxiv.org/pdf/2608.10423v1)

**作者:** Sebastian Künzel `[一作]` (University of Stuttgart), Dieter Schmalstieg `[通讯]` (University of Stuttgart)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于稀疏分层图像空间缓存的帧外推渲染方法，能在不使用神经网络的情况下从历史帧生成多帧新视角。

**💡 创新点**

创新点在于利用分层稀疏缓存存储多层可见几何与着色信息，并通过自适应TTL与梯度驱动的预更新策略高效处理新遮挡与高频着色变化。

**🔧 技术方法**

采用的技术包括frustum voxel分层稀疏缓存、光线遍历与占用遮罩、前后层合成、光线追踪阴影与全局照明、梯度估计调度、半透明WBOIT等。

**📊 数据集**

使用的评估数据集为Bistro Interior、Bistro Exterior、Intel Sponza和San Miguel四个公开场景。

**📈 对比分析**

与DLSS 4.5、MobFGSR、MoFlow以及基准延迟渲染器比较，在同等或更低时延下，PSNR/SSIM/FLIP/LIPPS略优于这些方法；在4K/240Hz外推下可达约250Hz，性能比传统方法快3–4倍。

**⚠️ 局限性**

局限性是对高动态、非刚体或变形场景（如大量动画人物）需要频繁全局重新着色，导致开销增大，在极端动态场景下性能退化，可能需退回到传统渲染或混合方案。

---

## 294. The Parser Already Knows: Lightweight Bias Correction in Constrained Decoding

**arXiv ID:** 2608.10137 | [PDF](https://arxiv.org/pdf/2608.10137v1)

**作者:** Işıl Özgü `[一作]` (University of California Los Angeles), Miryung Kim `[通讯]` (University of California Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种离线训练的轻量化对数概率校正器 SHIM，利用已存在的内部解析器和词法状态修正 GCD 的概率分布；

**💡 创新点**

创新点在于把解析器和词法状态直接当作特征学习校正因子，既不改动基础 LLM 权重，也不需要昂贵的在线重采样，显著恢复语法约束下的真实分布；

**🔧 技术方法**

采用多层感知机或逻辑回归模型，以解析器状态、词法状态和候选下一个 token 为特征，训练校正因子 γ，并与现有 GCD 工具（如 Syncode）结合；

**📊 数据集**

在 BV4 语法的 SyGuS 逆位条件基准、Spider 文本到 SQL 任务以及多种上下文无关语法上进行实验；

**📈 对比分析**

与 Syncode 的掩码方法和 ASAp 的在线采样进行 KL 散度与推理时延对比，SHIM 在大多数语法下 KL 降低 10-200 倍，且推理速度仅略高于掩码，远快于在线采样；

**⚠️ 局限性**

局限性包括需访问解析器/词法状态，对极大多义或长词法词典的语法效果有限；在不支持内部状态的 GCD 工具下只能使用单独 token 校正，性能相对较弱。

---

## 295. Playable Pressure: Affective Dramaturgy and Selective Realism in the Design of a VR Emergency-Response Serious Game

**arXiv ID:** 2608.10763 | [PDF](https://arxiv.org/pdf/2608.10763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 296. P3CA: Encoder-Agnostic Interpretation of Vision Foundation Model Embeddings via Spatial Probing

**arXiv ID:** 2608.10131 | [PDF](https://arxiv.org/pdf/2608.10131v1)

**作者:** Amoon Jamzad `[一作]` (Queen's University), Parvin Mousavi `[通讯]` (Queen's University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种位置提示主成分分析（P3CA）方法，用于在用户指定的空间区域内局部拟合主成分投影，并将该投影应用于整个特征张量，实现对高维空间表示的局部可视化和解释。

**💡 创新点**

创新点在于将PCA的拟合空间限定为用户提示区域，提供编码器无关、无需标签、无模型改动的局部对比增强技术，并在多模态数据中实现跨模态对比。

**🔧 技术方法**

使用了位置提示的局部协方差矩阵求特征值分解（PCA），并结合预处理归一化、可视化RGB映射以及嵌入式交互式框架EmbedVision进行实现。

**📊 数据集**

评估数据集包括自然图像（DINOv3编码的街景图像）、结肠病理切片（GigaPath 与 H0-mini 两种基础模型编码的全切片图像）以及胶质母细胞瘤样本的 H&E 与 10x Visium 空间转录组数据。

**📈 对比分析**

通过对比全局PCA与P3CA在不同提示下的对比度增益、三维投影空间的类间分离以及基于冻结投影的LDA分类准确率，结果显示P3CA在局部结构上显著提升对比度（如对比度增益可达13.56）并使诊断相关类别的分类准确率提升至约85%（相较于全局PCA的约60%）。

**⚠️ 局限性**

局限性包括：投影线性且只能捕捉三维结构；输出高度依赖提示选择，可能压缩非提示区域信息；RGB颜色仅表示投影坐标，缺乏语义解释；需要进一步研究提示选择稳定性与用户差异。

---

## 297. TAF-MED: Multi-Turn Safety Refusal Collapse in LLMs Under Declared Self-Treatment Intent

**arXiv ID:** 2608.10258 | [PDF](https://arxiv.org/pdf/2608.10258v1)

**作者:** Waleed Jamil `[一作]` (Independent Researcher), Raphael Schmitt `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型在自我治疗意图出现后多轮对话中的药物安全性持续性，并提出了基于临床场景的 TAF-MED 基准进行评估。

**💡 创新点**

首次将药物安全边界的持久性作为评估维度，构建了包含500个经过医生审查的固定三轮对话场景，并使用自动与医生双重标注相结合的方法来量化安全崩溃。

**🔧 技术方法**

采用 GPT-4o 自动判别器进行标签，使用 Claude、GPT-5.4、Gemini 等八种大型语言模型进行响应收集，并通过统计学 bootstrap、Spearman/Kendall 等方法评估排名稳定性。

**📊 数据集**

基于医生审核的 500 份合成场景，覆盖10个临床家族、三种严重程度和四类药物指导目标，共 4,000 条三轮对话。

**📈 对比分析**

对每个模型在 U1、U2、U3 轮的 Unsafe 率、任意轮 Unsafe 率以及 Safe→Unsafe 的崩溃率进行计算，发现 Safe 率 26.4% 上升至 63% 的 Unsafe，71.6% 的对话出现 Unsafe，61.4% 的 Safe 开头对话随后崩溃，模型排名在多轮评估中部分出现倒置。

**⚠️ 局限性**

局限在于使用合成固定三轮对话、缺乏真实患者对话长序列、未考虑工具使用或部署时的安全策略、以及医生标注样本有限，不能反映真实世界使用情况。

---

## 298. Weak Bisimulation Finiteness of Pushdown Systems With Deterministic $\varepsilon$-Transitions Is 2-ExpTime-Complete

**arXiv ID:** 2608.10583 | [PDF](https://arxiv.org/pdf/2608.10583v1)

**作者:** Stefan Göller `[一作]` (University of Kassel), Paweł Parys `[通讯]` (University of Warsaw)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对ε-标签化推导栈系统（ε-Pushdown System）的弱Bisimilarity有限性问题进行理论研究，并证明该问题属于双指数时间（2-）。

**💡 创新点**

创新点在于提出了泵三元组（Pumping Triple）分解与类型归一化方法，利用其结构化特性对无限堆栈状态进行等价类数目上界的严谨计数。

**🔧 技术方法**

核心技术包括：ε-PDS的泵三元组分解、层次逼近≈_k关系、弱Bisimilarity的类数递增判断、以及对泵三元组类型的集合论计数与递归归纳。

**📊 数据集**

无实验数据集（纯理论分析）。

**📈 对比分析**

与已知的NP/PSPACE可判定性相比，本工作将上界提升到双指数时间，并提供了构造弱Bisimilarity商的算法，理论上可实现最优等价类数≤Z的判定。

**⚠️ 局限性**

主要限制在于算法复杂度仍为双指数，且仅适用于ε-Pushdown系统，对更一般的无ε或多标签系统尚未推广；实现细节与实际性能仍待进一步实验验证。

---

## 299. How to Dogfood Your AI Chat Agent: A Three-Layer Evaluation Framework with Goal-Directed NPC Simulation

**arXiv ID:** 2608.09939 | [PDF](https://arxiv.org/pdf/2608.09939v1)

**作者:** Alexandre Cristovão Maiorano `[一作]` `[通讯]`, Alexandre Cristovão Maiorano

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一个三层狗盒测试框架，用于在 CI/CD 中对 LLM 聊天代理进行从单轮正确性、对话连贯性到目标导向成功率的全维度评估；

**💡 创新点**

创新点在于将传统的命题问答、随机漫步多轮评估与结构化 NPC（非玩家角色）模拟相结合，并引入了五种目标类型与十分类失效分类法，实现了自动化的发布门控（PROMOTE/HOLD/ROLLBACK）与持续改进；

**🔧 技术方法**

技术包括 Gemini 2.5 Pro/Flash 作为系统和 LLM‑as‑judge，Python‑first 运行时，随机漫步生成器，NPC 生成与评判提示，Bootstrap 置信区间，Pearson/Spearman 相关分析，以及 GitHub Actions CI/CD；

**📊 数据集**

数据集为公司内部生产系统的对话日志与人工编写的 20+ 版本测试集（共 257 次评估、108 条 NPC 场景、83 条单轮问答、7 条多轮测试），其中包括 13–138 条问题、6 种 persona、5 种目标类型；

**📈 对比分析**

与单层评估相比，三层方案互补性强，NPC 通过目标达成率从约57%提升至70%（两周内迭代），评估成本仅 0.17 USD/轮，耗时 12 min，显著低于人类评估（约 1080 USD/轮、27 h），且能够实现每日自动化发布决策；

**⚠️ 局限性**

局限性包括缺乏直接的实测对真实用户满意度的转移验证、单一 LLM 供应商导致的自匹配偏差、NPC 评判缺乏人类校准、场景编写的作者偏差、单一生产系统实验导致外部可泛化性待验证。

---

## 300. Path Integral Value Matching for Linear Quadratic Stochastic Optimal Control

**arXiv ID:** 2608.10777 | [PDF](https://arxiv.org/pdf/2608.10777v1)

**作者:** Bangyan Liao `[一作]` (Westlake University), Tailin Wu `[通讯]` (Westlake University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对线性-二次随机最优控制（LQ‑SOC）问题，提出一种基于值函数的 Path Integral Value Matching（PI‑VM）算法，利用递归路径积分与时序差分学习，消除对完整轨迹模拟的依赖。

**💡 创新点**

创新点在于将传统路径积分控制的递归结构与TD学习结合，利用Girsanov定理实现离线经验回放的无偏重采样，并在高维场景下显著降低估计方差。

**🔧 技术方法**

核心技术包括连续时间路径积分控制、时序差分（TD）损失、经验回放缓冲、Girsanov变换、自动微分和神经网络值函数逼近。

**📊 数据集**

实验数据集涵盖三种线性/二次 Ornstein‑Uhlenbeck 控制任务、20 维高模态高斯混合模型（GMM）采样任务以及50 维多井势能采样任务。

**📈 对比分析**

与七种主流基于策略的基线（RE、CE、VAR、LVAR、AM、SOCM、SOCM‑A）进行对比，PI‑VM 在控制误差、采样精度上与最优方法持平或更优，并在计算效率上提升了约10–20倍，尤其在高维（至200维）下仍保持稳定收敛。

**⚠️ 局限性**

局限性在于仍需通过自动微分对值函数求梯度以得到控制信号，导致在极大规模或实时需求场景下的推理速度可能受限。

---

## 301. MERA: Model Evolution and Routing with Skill Adaptation for Agentic Systems at Scale

**arXiv ID:** 2608.10333 | [PDF](https://arxiv.org/pdf/2608.10333v1)

**作者:** Yuhang Yao `[一作]` (Carnegie Mellon University), Tianyu Shi `[通讯]` (Gradient)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为 MERA 的多周期自适应协议，利用可执行的执行轨迹不断改进小模型能力，并通过验证器和回退机制确保部署质量。

**💡 创新点**

创新点在于：①将模型调用视为自适应单元；②使用共享轨迹同时更新技能库、路由器和小模型适配器；③通过联合回放验证所有更新组合，避免单组件指标误导；④在运行时仅基于输入进行路由，减少与隐藏状态耦合。

**🔧 技术方法**

核心技术包括：输入级路由器、可重用技能库（SkillBook）外部程序提示记忆、基于 Qwen3-Embedding 的分类逻辑回归路由器、可执行验证器（代码运行、工具调用验证）、多周期训练调度（Skill → LLM → Router），以及联合回放决策。

**📊 数据集**

主要使用数据集：HumanEval、MBPP（合并为 546 训练任务，582 评估任务），TAU‑2（35 任务拆分），以及用于成本评估的财务模拟工作负载。

**📈 对比分析**

对比方法：与直接使用小模型（无路由/回退）、SFT 单轮、SFT+GRPO 多轮、始终使用大模型等基线进行比较；在 HumanEval+MBPP 上四周期适配将小模型通过 SFT+GRPO 提升至 49.7% 通过率；通过验证器回退后，部署性能为 88.3% 通过率，成本仅为 60.8% 的大模型；在 TAU‑2 上，小模型通过适配从 14/35 提升至 18/35，接近未适配的 4B 端点。

**⚠️ 局限性**

局限性：①在 TAU‑2 的适配评估样本不足，统计显著性有限；②回放验证虽能防止失效，但仍需生产中沙箱/灰度验证与漂移监测；③技能库和路由器的效果在多工具、多轮交互场景中尚未充分验证；④方法依赖可执行验证，适用范围受限于可验证任务。

---

## 302. MESA:Task-Adaptive Multi-Structure Evidence Selection for Long-Horizon Agent Memory

**arXiv ID:** 2608.10108 | [PDF](https://arxiv.org/pdf/2608.10108v1)

**作者:** Beidi Zhao `[一作]` (University of British Columbia), Qi Chen `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种结构层动态选择框架，学习在长时序轨迹中对多种结构化记忆的查询自适应子集进行检索与融合，从而提升长时序代理问答性能。

**💡 创新点**

发现最佳记忆子集通常是中间规模且任务/查询相关的；提出基于弱监督的答案级反馈、先验引导和UCB平衡的策略优化方法；在固定构建器和答案模型的情况下，仅通过学习选择器实现显著性能提升。

**🔧 技术方法**

多结构记忆构建（文本摘要、时间存储、知识图谱、向量检索、原始轨迹）；查询自适应选择器（可执行策略）；先验引导的策略生成器与UCB调度；答案模型（如Qwen3-32B/Gemma-4-31B）。

**📊 数据集**

AMA-Bench（208剧集、2496问答，六个代理领域，四种记忆能力）；LoCoMo（10多会话，1540问答）。

**📈 对比分析**

与长上下文、BM25、Qwen3-Emb-4B、MemGPT、HippoRAG2、Mem0、MemoRAG、A-Mem、EMem、Hindsight、AMA-Agent 等基线对比；在AMA-Bench上使用Qwen3-32B实现总体准确率65.1%，比最强基线高8.5%，比全结构方案减少41%证据标记；在LoCoMo上F1 49.0%，比最强基线高1.8%。

**⚠️ 局限性**

只优化选择器而未联合优化记忆构建；对不同记忆结构的依赖性和跨结构证据关系未建模；在对话记忆上的提升有限；需要进一步探索跨结构协同与训练效率。

---

## 303. Dynamic Context Adapters: Efficiently Infusing History into Vision-and-Language Models

**arXiv ID:** 2608.10525 | [PDF](https://arxiv.org/pdf/2608.10525v1)

**作者:** Yuhang Song `[一作]` (University of Liverpool), Chun-Yi Lee `[通讯]` (National Taiwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Dynamic Context Adapter（DCA）方案，利用可学习的固定大小记忆向量将历史视觉上下文压缩并注入预训练 Vision‑Language Model（VLM），实现对长时序导航任务的高效记忆与决策。

**💡 创新点**

创新点包括：① 将历史帧压缩为固定大小可学习记忆，避免拼接导致的序列长度爆炸；② 通过轻量级跨层适配器在 VLM 每层注入压缩记忆，实现线性时间复杂度；③ 兼顾预训练模型原有知识，保持架构不变；④ 在保持显存、FLOPs 下降的同时，显著提升导航性能。

**🔧 技术方法**

采用 PrismaticVLM phi‑2+3b 作为 VLM 基础；使用 Vision‑Transformer + CLIP 的视觉编码器、Phi‑2 语言模型、跨模态投影；设计 Memory Compression Module（多层交叉注意力）与 Memory Integration Module（跨层注意力）；借鉴 PEFT 思想，构建轻量级适配器；实现跨模态、跨层的动态上下文注入。

**📊 数据集**

在 VLN‑CE 环境下的 R2R（Reach Room）任务，使用标准的 Val‑Unseen 数据集进行评估；训练时结合 RGB‑D 视觉输入和自然语言指令。

**📈 对比分析**

与 RGB‑Seq2Seq、RGB‑CMA、NaVid、No‑Adapt、Recurrent‑Adapt 等基线对比；DCA 在 Success Rate（SR）上提升 13.7%（相较 RGB‑Seq2Seq）和 8.7%（相较 RGB‑CMA），与 NaVid‑IL 的性能相当但参数量更小；在 SR、SPL 等指标上超过拼接方法，同时 FLOPs 减少 25% 以上、显存降低约 15%；整体实现了优异的效率‑性能平衡。

**⚠️ 局限性**

局限性：① 压缩向量容量有限，极长序列可能导致信息缺失；② 需要额外的压缩与适配器模块，训练开销略高；③ 对指令语义相关压缩未明显提升性能；④ 在更大规模 VLM 或不同任务（如视频问答）中的适用性仍待验证；⑤ 细粒度视觉信息的保留程度有限。

---

## 304. From Prompt Injection to Web Exploitation: Revisiting Classic Vulnerabilities in LLM-Integrated Applications

**arXiv ID:** 2608.10281 | [PDF](https://arxiv.org/pdf/2608.10281v1)

**作者:** Spiros Tsigkopoulos `[一作]`, Christoforos Ntantogian `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并系统化了 LLM 中介的 Web 攻击（LLM2X），通过 TicketOracle 实验平台评估了七种 LLM 在五种 SSRF 场景下的成功率，并探讨了 Prompt、模型、应用及网络层的防御策略。

**💡 创新点**

创新点在于：①将 Prompt 注入视为通往传统 Web 漏洞的桥梁，形成 LLM2SQLi、LLM2XSS、LLM2SSTI、LLM2CommandInjection、LLM2IDOR、LLM2CSRF、LLM2XXE、LLM2SSRF 等攻击类别；②基于 TicketOracle 的可复现实验框架，首次在多模型、多场景下量化 LLM 中介攻击的可行性；③系统性提出分层防御方案，验证 Prompt 约束与网络级 egress 控制的有效性。

**🔧 技术方法**

技术方法包括：Flask 服务器、LLM 代理（OpenRouter API）、工具调用（fetch_event_data）、Prompt 约束与允许列表实现、网络级 egress 策略、日志审计（retention.log）。

**📊 数据集**

使用的数据集为 TicketOracle 自定义事件/票务数据库以及人工植入的恶意评论，实验中不涉及公开大规模数据集，仅利用内部构造的数据进行攻击验证。

**📈 对比分析**

对比方法：对七种 LLM 在四种直接攻击（DA1-DA4）与一次间接攻击（IA）下分别执行 10 次会话，记录成功率。结果显示，非硬化版 Llama 3.3、Qwen3、DeepSeek 均达 10/10 成功率，GPT-5.2 与 Claude Opus 全部失败；硬化版则对直接攻击实现 100% 阻断，但对间接攻击表现不一。性能评估表明，更新的模型往往在 Prompt 层具备更强的拒绝能力，但对间接注入仍易被绕过。

**⚠️ 局限性**

局限性包括：①实验仅聚焦 SSRF，未覆盖其它 LLM2X 类别；②仅评估了七种模型，缺乏更广泛的跨模型泛化验证；③Prompt 约束在不同模型上表现不一，无法提供统一防护；④未探索针对 LLM 本身的更强 Jailbreak 技术；⑤实验环境与真实生产系统的网络拓扑、权限配置存在差距，可能导致结果偏差。

---

## 305. PolypVision: A Three-Stage Hierarchical Deep Learning Framework for Classification and Segmentation of Colorectal Polyps

**arXiv ID:** 2608.10649 | [PDF](https://arxiv.org/pdf/2608.10649v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 306. Multi-View Relational Distillation for Spatial Reasoning with Vision-Language Models

**arXiv ID:** 2608.10864 | [PDF](https://arxiv.org/pdf/2608.10864v1)

**作者:** Kiet T. Nguyen `[一作]` (Korea Advanced Institute Of Science And Technology), Seunghoon Hong `[通讯]` (Korea Advanced Institute Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究如何在保持视觉语言对齐的前提下，通过多视角关系蒸馏提升视觉语言模型的几何推理能力。

**💡 创新点**

提出多视角关系蒸馏（MVRD），仅蒸馏跨视角的余弦相似关系，而非特征本身，从而避免破坏预训练的视觉-语言对齐。

**🔧 技术方法**

采用多视角关系蒸馏、低秩适配（LoRA）以及双通路架构来分离几何与语义目标。

**📊 数据集**

在ScanNet/ScanNet++/ARKitScenes的多视角问答数据集上训练，评估使用VSI-Bench以及多种3D场景理解数据集（ScanRefer、Multi3DRefer、Scan2Cap、ScanQA、SQA3D）。

**📈 对比分析**

与SFT、特征蒸馏（3DRS）和特征融合方法（VLM-3R、VG-LLM）比较，MVRD在VSI-Bench上平均准确率达60.4%（接近SOTA特征融合），并在3D场景任务上显著优于基线。

**⚠️ 局限性**

训练规模有限，仍存在几何与语义平衡的折中，未来需在更大规模与更高效架构下进一步提升。

---

## 307. TRACE-GS: On-Policy Trajectory Distillation with Privileged Geometric Conditioning for Sparse-View 3DGS Restoration

**arXiv ID:** 2608.10286 | [PDF](https://arxiv.org/pdf/2608.10286v1)

**作者:** Linlian Jiang `[一作]` (Concordia University), Xinxin Zuo `[通讯]` (Concordia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 TRACE-GS，一种利用训练时的特权几何信息进行 on‑policy 轨迹蒸馏的稀视角 3D 高斯散点重建恢复框架；

**💡 创新点**

创新点在于：①采用几何不对称的教师‑学生对，教师仅在训练时利用稠密视角几何提供更可靠的恢复方向；②在学生的逆向扩散轨迹上进行 on‑policy 蒸馏，避免了传统离线监督导致的状态偏差；③对齐教师与学生的跨视角检索响应，进一步约束多视角一致性；

**🔧 技术方法**

使用技术包括视频扩散模型（pre‑trained 生成器）+ LoRA 适配器进行速度预测、on‑policy 轨迹蒸馏损失、检索对齐损失、以及稠密与稀疏渲染序列的几何对齐；

**📊 数据集**

训练数据来自 DL3DV 112 场景；评估使用 DL3DV‑Benchmark、Mip‑NeRF 360 以及 NeRFBusters 三个基准；

**📈 对比分析**

与 3DGS、Difix3D+、GenFusion、GSFixer、NeRF‑based 以及其他扩散恢复方法对比，TRACE‑GS 在 3、6、9 视角下均实现 PSNR/SSIM/LPIPS 的最高或次高分数，尤其在 3 视角极端稀疏时提升 2+ dB PSNR；

**⚠️ 局限性**

局限性在于：仅针对静态场景，需在训练阶段获得稠密视角作为特权信息；部署时仍需稀疏视角；在动态捕捉或无稠密视角可用的环境下效果未知。

---

## 308. ChronoSSM: Training for Temporally Aware Representations in Autoregressive State Space Models

**arXiv ID:** 2608.10120 | [PDF](https://arxiv.org/pdf/2608.10120v1)

**作者:** Adrien Schoen `[一作]` (École Normale Supérieure de Lyon), Francesco Bronzino `[通讯]` (École Normale Supérieure de Lyon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种自回归State Space Model（SSM）
   既预测下一个事件，也预测对应的时间戳；并对比两种训练策略——全局联合训练（joint）和先只训练事件后再训练时间（token-only）。

**💡 创新点**

创新点：
   ① 把时间预测嵌入同一共享backbone，避免两阶段假设；
   ② 通过可恢复性诊断（线性probe、Temporal Cohesion Score）和生成质量评估，系统证明时间监督能提升时间信息可恢复性且不显著降低生成质量。

**🔧 技术方法**

技术：
   - Backbone：SSM（Mamba2；对比实验也用GPT‑2）
   - 时间head：基于对数‑正态密度的连续时间预测，使用log‑变换目标；
   - 损失：token交叉熵 + λ·时间负对数似然；
   - 评估工具：ridge‑regression probe、TCS、DLS、JSD、Hits@k等。

**📊 数据集**

数据集：
   - 业务流程：BPI2018（业务流程日志）
   - 临床事件：MIMIC‑IV（ICU 监测数据）
   - 网络流量：视频流包捕获（packet‑capture）
   - 时序知识图：GDELT（全球事件图）
   - 额外实验：音乐（MIDI）

**📈 对比分析**

比较方法与性能：
   - 对比 joint 与 token‑only 两种训练方式；
   - 在四大域分别计算：
     • 时间可恢复性（TCS↑, R²↑, MAE↓）——joint 始终优于 token‑only；
     • 内容生成质量（BPI：DLS、MIMIC：JSD/TV、网络：JSD IP/port、KG：MRR/Hits）——在大多数域 joint 无显著下降，部分域（网络、KG）甚至提升；
   - 结论：联合训练提升时间可恢复性，且不产生系统性的生成质量下降。

**⚠️ 局限性**

局限性：
   ① 未对生成时间分布的真实性做严格评估；
   ② 结果在不同域表现不均，缺乏统一机制解释；
   ③ 超参数（λ、时间head 结构）对效果影响较大，需更系统的调优；
   ④ 仅使用现有领域特定评测，缺少统一的时间+内容联合评估标准。

---

## 309. When the Interviewer Is a Bot: Behavior, Breakdowns, and Trust in MLLM-Led Interviews

**arXiv ID:** 2608.10412 | [PDF](https://arxiv.org/pdf/2608.10412v1)

**作者:** He Zhang `[一作]` (Pennsylvania State University), John M. Carroll `[通讯]` (Pennsylvania State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

搭建 InterviewBot——一个基于实时多模态大型语言模型（LLM）的语音访谈系统，并通过15名参与者进行实地访谈，记录并分析访谈行为与用户体验。

**💡 创新点**

创新点在于：①以现成实时LLM为访谈主体，直接观察其默认访谈行为；②对428个访谈轮次进行细粒度行为编码，揭示LLM在深度探问、问题叠加等方面的典型模式；③提炼四类访谈失效（信息丢失、过早终止、延迟、打断）与三大社会动态（披露校准、制度合法性、对话基础）并给出针对深度控制、透明交接与内容化倾听的设计建议。

**🔧 技术方法**

使用了 OpenAI Realtime API（实时多模态LLM）配合语音识别与实时转写，构建了无人物形界面的语音交互界面，采用自动“barge‑in”与实时生成跟进问题功能。

**📊 数据集**

数据集包含15位受试者的访谈记录（共428个LLM轮次）及其访谈后反思访谈，对应的音频与转写文本；未使用公开大规模问答或访谈语料库。

**📈 对比分析**

方法上主要是经验性行为编码与主题分析；未与人工访谈或其他自动访谈系统做直接对比，亦未给出量化性能指标，报告的结果为描述性统计与主题发现。

**⚠️ 局限性**

局限性包括：样本规模小、年龄集中于研究生与本科生、缺乏对AI经验的基线测评、转写误标导致部分手动纠正、部分系统功能（如实时打断、单问题）实现不完整，且未采用Wizard‑of‑Oz验证人工干预效果。

---

## 310. Reasoning Shortcuts and Value Symmetries: What Symmetry Permits, Architecture Realizes, and Optimization Selects

**arXiv ID:** 2608.10420 | [PDF](https://arxiv.org/pdf/2608.10420v1)

**作者:** Xin Xu `[一作]` `[通讯]` (Carnegie Mellon University), Xin Xu (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过把神经符号系统的推理捷径建模为约束满足问题，并利用自动同构群研究规则解空间的结构，探讨何时规则能唯一决定概念映射。

**💡 创新点**

创新点在于提出了组件级值对称性（componentwise value symmetry）来取代原来要求单一共享值集的全局对称性定义；对该新定义在异构基准上的有效性进行了系统测量，并通过四机制的经验归纳和六条定理提供了理论解释。

**🔧 技术方法**

主要技术包括：约束满足问题建模、自动同构群与 orbit‑stabilizer 计数、线性代数分析、随机化归约、以及对布尔域下的向量空间结构的利用。

**📊 数据集**

使用了 rsbench 公开的四个异构基准（CLE4EVR、Kandinsky、BDD‑OIA/SDD‑OIA）以及自定义的八类规则族、Latin square、Sudoku 等组合问题作为实验数据集。

**📈 对比分析**

与原先的全局对称性定义对比，组件级对称性将 90% 的误报率降至 0%，并通过实验验证其对 70% 以上捷径实例的解释能力；理论上证明了大多数实例的转移性或不转移性，表明该方法在性能上显著优于原始定义。

**⚠️ 局限性**

局限性包括：对全局对称性在异构情况的适用性仍有限；部分机制仅给出充分条件而非必要与充分；复杂度分析只在特定的输入表示下成立，通用决策问题仍处于高阶多项式层级。

---

## 311. Improving TensorSketch Using Complex Random Variables

**arXiv ID:** 2608.10523 | [PDF](https://arxiv.org/pdf/2608.10523v1)

**作者:** Amit Sharma `[一作]` (IIT Hyderabad), Keegan Kang `[通讯]` (Bucknell University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种复杂到实数（Complex-to-Real，CtR）变体的张量 Sketch，用于高阶多项式核的低维近似。

**💡 创新点**

创新点在于：① 使用独立的四阶根号 unity 随机函数取代传统的 Rademacher 符号；② 通过 CtR 转换得到实值嵌入；③ 维持输入稀疏性 O(p(ẋ + D log D)) 的计算复杂度；④ 证明了方差上界从 3^p/D 降至 2^p/D，显著提高估计精度。

**🔧 技术方法**

核心技术包括：哈希冲击（CountSketch）与随机投影（JL-type）相结合；复杂随机变量的四阶根号 unity 取值；FFT 基础的多项式卷积；理论上利用 Khintchine、Cauchy-Schwarz 等不等式进行方差分析。

**📊 数据集**

实验数据集：人工合成（n=3000, d=2，p∈{10,15,20}）以及真实数据 MAGIC Gamma Telescope（d=10）和 COD‑RNA（d=8），所有样本均归一化后输入。

**📈 对比分析**

与 Real TensorSketch、密集 JL‑type（Gaussian/Rademacher）以及它们的 CtR 变体进行对比。结果显示：CtR TensorSketch 在 KL 散度上持续优于所有基线，且与 Real TensorSketch 的构造时间相近（保持输入稀疏性），而密集 JL‑type 的时间更高。

**⚠️ 局限性**

局限性：仅针对多项式核提出，未探讨更高阶根号 unity 是否能进一步改善高阶矩收敛；无法直接推广到其它核函数或更一般的随机特征映射；实验与理论均未给出显式的 (ε,δ) 近似保证。

---

## 312. Cracks in the Foundation: Seemingly Minor Architectural Choices Impact Long Context Extension

**arXiv ID:** 2608.10296 | [PDF](https://arxiv.org/pdf/2608.10296v1)

**作者:** Amanda Bertsch `[一作]` (Ai2), Dirk Groeneveld `[通讯]` (Ai2)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在保持数据、tokenizer、优化器等一致的前提下，对 7B 级密集 Transformer 进行系统性架构消融实验，探索四项常见设计选择（归一化策略、Grouped‑Query Attention、滑动窗口注意力、预训练上下文长度）如何在长上下文场景下相互叠加并显著削弱性能。

**💡 创新点**

发现单独的“微小”设计变动对长上下文性能影响有限，但至少三项变动组合时可导致性能下降高达 47%，并证明短上下文指标无法预测长上下文表现，揭示了长上下文扩展需提前验证的重要性。

**🔧 技术方法**

采用标准 Transformer + RoPE + QK‑norm、GQA、滑窗策略，结合 64K RoPE 调整的两阶段长上下文扩展（10B 继续预训练），并使用线性回归与注意力统计量对结果进行解释。

**📊 数据集**

使用公开的多领域大规模文本混合（如 Longmino、Project Gutenberg、FineWeb、政府与法律文本等），共 140B token 的通用预训练数据，随后在 64K 长度上进行 10B token 的扩展训练。

**📈 对比分析**

通过 RULER、HELMET、LongPPL 三个长上下文基准对比评估，结果显示 Llama‑3 架构在长上下文扩展上表现最好；在相同扩展预算下，包含四项负面设计的模型平均 HELMET 分数从 56.4 降至 29.9，表明设计组合对性能影响显著。

**⚠️ 局限性**

主要局限：实验仅覆盖 7B‑8B 规模；长上下文扩展的训练时间高昂（约 170,000 GPU‑h）；未对更大规模或更稀疏/递归等非标准 Transformer 进行验证；缺乏对长上下文瓶颈的因果机制解释，仅提供统计相关性。

---

## 313. ELMER: Evolutionary Language Model that Explores and Refines

**arXiv ID:** 2608.10196 | [PDF](https://arxiv.org/pdf/2608.10196v1)

**作者:** Matthew Siper `[一作]` (New York University), Julian Togelius `[通讯]` (New York University)

**通讯引用:** 14798 | [OpenAlex ID](https://openalex.org/A5077267552)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了一种基于自然语言的程序进化变异器，利用大型语言模型在自然语言描述上进行策略编辑，然后将其编译为可执行的 GPTL 代码，通过执行结果来控制行为空间位移。

**💡 创新点**

核心创新在于将行为空间位移度量与可调强度的条件变异器相结合，并使用 Common-Parent oDPO 在 LLM 上实现可控的行为级变异，证明自然语言既能提供可执行搜索空间，又能精细调节变异幅度。

**🔧 技术方法**

技术手段包括：Qwen3-8B 细化的多任务 LLM（支持 NL→GPTL 编译、GPTL→NL 翻译和条件 NL 变异）；逆向退化链生成行为量化监督数据；Common-Parent oDPO 偏好优化；以及使用 GPTL 进行确定性回测执行。

**📊 数据集**

实验基于 2008‑2025 年的 E‑mini S&P 500、银期和美国国债期货的每小时行情数据，总计约 320 000 条交易记录，划分为训练、验证和测试三份，用以生成变异训练数据和评估搜索性能。

**📈 对比分析**

与传统 AST 变异器、匹配代码输出模型以及无标签 SFT 进行 252 次固定预算搜索对比；结果显示 oDPO 在行为位移校准、AUC 与测试 Sharpe 上显著优于基线，且在最高测试 Sharpe 上达到了 2.34 的记录。

**⚠️ 局限性**

局限性包括：强度控制仅为离散等级且在高强度时易饱和，无法实现精确的数值位移；对不同外部优化器的交互效果尚未系统评估；以及在极大变异幅度下模型收敛性与稳定性仍有待提升。

---

## 314. When Chain-of-Thought Helps and When It Hurts: An Empirical Investigation of the Serial-Depth Bottleneck in LLM Reasoning

**arXiv ID:** 2608.09942 | [PDF](https://arxiv.org/pdf/2608.09942v1)

**作者:** Tughanbulut Kurtulush `[一作]` `[通讯]` (Vistula University), Tughanbulut Kurtulush (Vistula University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过检验H_dp带宽上界在实际LLM推理中的适用性，探究链式推理（CoT）是否普遍提升推理性能。

**💡 创新点**

创新点在于将理论上的H_dp带宽瓶颈与实际基准数据结合，提出并验证CoT仅在高序列深度（P‑complete）任务上有效、对低深度任务无效的单向解释框架。

**🔧 技术方法**

采用三款指令微调的解码器型Transformer（Qwen‑2.5‑7B/32B、Llama‑3.1‑8B）以及两种推理条件（直接答复与CoT）进行对照实验，并通过深度梯度与Spearman相关性检验。

**📊 数据集**

使用五个常用NLP基准（GSM8K、MATH、MMLU、ARC‑Challenge、HumanEval）以及其符号化版本GSM‑Symbolic进行评测。

**📈 对比分析**

对比无CoT与CoT两种提示下的准确率，发现P‑complete任务CoT提升54–68个百分点，TC^0任务CoT基本无效，HumanEval呈模型规模依赖性，整体Spearman ρ=0.661，显著相关。

**⚠️ 局限性**

局限性包括H_dp上界仅在极端上下文长度下严格成立、实验仅涵盖解码器型Transformer、无CoT与CoT条件的推理时间与输出长度差异未分离、以及TC^0基准高基线可能掩盖了潜在的负面影响。

---

## 315. FUBU-EPSTEIN: A Large-Scale Twitter Dataset on the Jeffrey Epstein Case and Its Global Public Discourse (2019-2023)

**arXiv ID:** 2608.10210 | [PDF](https://arxiv.org/pdf/2608.10210v1)

**作者:** Michael Kreil `[一作]`, Daniel Thilo Schroeder `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并公开了一个包含 5,438 万条推文、37,029 万条社交图边、13,983 万条自动标注（情感、阴谋、误信息、毒性、道德情感、政治立场、动员等）的长期、多维度数据集，去文本化、匿名化后提供可供传播学、信息战等领域使用。

**💡 创新点**

① 大规模、跨多年的持续抓取；② 自动化 LLM 标注覆盖 12 个维度；③ 去文本化、匿名化处理并保留内部传播关系，兼顾研究可用性与隐私风险；④ 为公开数据集提供完整的元数据和验证脚本。

**🔧 技术方法**

Twitter Search API v1.1 抓取；Raspberry Pi 4 集群、Java RxJava 并行流水线进行压缩、去重、排序；vLLM+NVIDIA H200 GPU 进行 Qwen2.5‑7B‑Instruct 标注；Apache Parquet 存储、Python 读取器与自动化校验脚本。

**📊 数据集**

自研的 Epstein 数据集（V4 版）：5,4375,364 条推文记录、13,983,232 条标注、46,150,021 条内部关系、37,028,919 条社交图边；公开可在 Zenodo 下载。

**📈 对比分析**

论文未进行算法性能对比，主要提供数据质量统计：LLM 标注误差仅 0.014%，情感分布为 54.1% 负面、42.8% 中立；阴谋、误信息等标签覆盖率分别为 47.0%、6.3%、2.2% 等，供后续研究评估。

**⚠️ 局限性**

① 缺少原始文本，限制深度文本分析；② LLM 标注可能带来偏差，未做大规模人工验证；③ 去匿名化后仍可能被重识别；④ 仅覆盖推特，缺少跨平台视角；⑤ 数据采集截至 2023，后续事件未包含。

---

## 316. Extending Triple Graph Grammars to Formalize Complex View Definitions on Families of Models - Long Version

**arXiv ID:** 2608.10785 | [PDF](https://arxiv.org/pdf/2608.10785v1)

**作者:** Lars König `[一作]` (Karlsruhe Institute of Technology), Jens Kosiol `[通讯]` (Brandenburgische Technische Universität Cottbus -- Senftenberg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文把NeoJoin视图定义语言中的查询完整翻译为三图文法（TGG）规则，并扩展TGG以支持聚合、交叉乘积连接和重叠查询等复杂视图操作；同时引入 skip 语义来处理重叠查询。

**💡 创新点**

创新点：
① 对NeoJoin查询的完整TGG翻译，补充了此前未处理的复杂操作；
② 通过 skip 语义为TGG规则增添可跳过动作，解决重叠查询问题；
③ 利用多合并（multi‑amalgamation）实现交叉乘积连接；
④ 为聚合操作设计专门的TGG规则。

**🔧 技术方法**

采用技术：
- Triple Graph Grammars（TGG）及其前向/后向规则的操作化；
- 负应用条件（NAC）、属性约束与属性条件；
- 多合并规则（multi‑amalgamation）；
- 跳过语义（skip semantics）与效应导向的图变换；
- 规则子规则与计数器机制以保证终止性。

**📊 数据集**

使用数据集：论文中主要以硬件仿真模型（blocks‑and‑ports）与传感器硬件模型的配对为示例，并未使用公开的大规模真实数据集，仅提供概念性示例。

**📈 对比分析**

方法对比与性能：论文未给出实验评估或与其他方法的性能比较，只在理论层面讨论了正确性、冲突性和可合并性等性质，缺乏实际运行时的性能数据。

**⚠️ 局限性**

限制：
- 聚合函数往往不可逆，导致反向同步不唯一；
- 需要 TGG 的高级特性（NAC、属性约束、多合并），实现复杂度高；
- skip 语义可能引入非确定性；
- 目前仅在理论上证明正确性，未完成完整的合并性、终止性证明；
- 对选择模式与连接条件有一定限制，无法支持多重参与连接的情况。

---

## 317. Reifying Research Logic: AI-Assisted Workflow Construction and Incremental Refinement for Quantitative Syntax

**arXiv ID:** 2608.10662 | [PDF](https://arxiv.org/pdf/2608.10662v1)

**作者:** He Wang `[一作]` (National University of Defense Technology), Wei Yuan `[通讯]` (National University of Defense Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并评估了 QLWF 平台，利用 AI 辅助将自然语言研究描述转化为可视化、可执行的工作流，并支持增量修订；同时公开了节点库、基准任务和工作流模板。

**💡 创新点**

创新点包括：① 通过 reification 与 formalization 双重转换，将研究逻辑外化为可执行工作流；② 设计了包含五个阶段（分类、选择、计划、配置、验证）的 AI 辅助生成管道；③ 引入增量补丁修订机制，提升工作流复用性；④ 在量化句法领域公开完整的 48 节点库、64 任务基准（QL‑Bench）和 12 任务生命周期基准。

**🔧 技术方法**

核心技术：使用 GLM‑5 大语言模型在设计阶段完成文本解析、节点选择、流程规划与参数配置；固定节点库与执行引擎提供确定性执行；采用 JSON 结构持久化工作流；评价采用三层评估（结构有效性、可执行性、输出合理性）。

**📊 数据集**

数据集与基准：利用三份 Universal Dependencies treebanks（Chinese‑GSD、English‑EWT、Chinese‑PUD）进行实验；构建了 64 任务的 QL‑Bench 量化句法基准和 12 任务的生命周期基准。

**📈 对比分析**

方法比较：将 QLWF 与单提示、链式思考、两阶段三种基线在 64 任务上比较。QLWF 在 L1/L2 达 100%，L3 平均 98.4%，比基线提升约 40pp；在增量修订基准上，patch 成功率 100%，token 消耗约 1/3，平均速度提升约 2.5 倍。

**⚠️ 局限性**

局限性：仅在受限的量化句法领域验证；基准为作者自建，非社区标准；LLM 输出非确定性，token 计数近似；实验规模有限，外推至其他子领域需自行构建节点库与验证规则；仅在设计阶段使用 LLM，运行时保持确定性。

---

## 318. A Gateway Architecture for Enterprise MCP Authentication: Unifying Heterogeneous Auth, Identity Delegation, and the User / Non-User Persona Problem

**arXiv ID:** 2608.10760 | [PDF](https://arxiv.org/pdf/2608.10760v1)

**作者:** Suraj Kumar `[一作]` (Enterprise AI Platform Engineering), Srinivasan Manoharan `[通讯]` (Enterprise AI Platform Engineering)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并部署了统一的 MCP 网关架构，实现单一端点、双轴认证模型（用户/非用户 × 各种凭证），从而统一身份验证、授权、审计与离职管理；

**💡 创新点**

创新点包括：双轴认证模型的抽象与统一实现；网关提供三种 SSO 授权方式（Auth Code+PKCE、Device Code、ROPC）和三种令牌交付模型（BYOT、GYOT、RFC 8693）；以及三种端到端身份流的通用化，尤其对易被误用的 User→Service‑Account 场景提供安全检查；

**🔧 技术方法**

采用 OAuth 2.0（PKCE、Device、ROPC）、RFC 8693 令牌交换、MCP 协议、共享 SDK 进行令牌验证、缓存与刷新，配合企业 SSO、CDN/WAF 或私有 MCP 隧道；

**📊 数据集**

无公开数据集，主要在企业内部对数十个 MCP 服务器进行部署验证；

**📈 对比分析**

没有传统实验比较，评估以生产运营为准：在多种客户端（Web、桌面、SDK、低代码）中实现统一登录、工具搜索与审计，显著降低离职与权限回收成本；

**⚠️ 局限性**

局限性包括：需对 SSO 系统高度依赖，ROPC 对非人类身份有安全限制，初始配置与跨团队协作复杂，且网关不负责资源级授权，需要下游服务器自行实现细粒度权限控制；

---

## 319. Monophonic Audio Synthesizer Using FPGAs

**arXiv ID:** 2608.10116 | [PDF](https://arxiv.org/pdf/2608.10116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 320. Post-Calibration Reliability Reranking of Relevance Decisions via Label-wise Monotone Projection

**arXiv ID:** 2608.10406 | [PDF](https://arxiv.org/pdf/2608.10406v1)

**作者:** Inwoo Tae `[一作]` (UNIST), Yongjae Lee `[通讯]` (UNIST)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在校准后对已确定的相关性决策进行可靠性重新排序的标签相关单调投影方法（MRP）。

**💡 创新点**

创新点在于为每个预测标签学习单调函数，将校准后的置信度映射为该标签的正确性可靠性，从而在不改变预测结果的前提下实现更精细的错误风险评估。

**🔧 技术方法**

技术手段包括：后置概率校准（温度缩放、对角校准、样条校准等）、单调格点（lattice）优化、二阶差分正则化，以及可靠性评估指标（NLL_correct、AURC、AUPR‑Error）和可选的概率几何兼容性分析（MRC）。

**📊 数据集**

在六个信息检索相关性数据集上验证：Amazon ESCI、MSLR‑WEB10K、Alloprof‑Rerank、ESCI‑Rerank‑US、WANDS、SciDocs。每个数据集覆盖不同领域（商品搜索、网页搜索、问答检索、科学检索）和标签粒度（二值或多级）。

**📈 对比分析**

与多种后置校准器（无校准、TS、DIAG、Spline、h‑cal、SMART）结合使用，并与基线置信度、共享 1D、标签截距等消融模型对比。结果显示，MRP 在大多数数据集上显著降低 NLL_correct 和 AURC、提升在预算回退场景下的准确率（SelAcc@τ），在标签残差可靠性显著的任务中也提升 AUPR‑Error；同时保持原始准确率与 ECE 不变。

**⚠️ 局限性**

局限性包括：仅对已确定的预测做重新排序，无法纠正需要改变标签的错误；当标签间残差可靠性弱时（如 SciDocs）改进有限；MRC 的概率几何映射并非总是可行；方法假设固定决策且不涉及候选生成或检索排序的端到端优化。

---

## 321. CHORUS: Complementary Experts for High-Coverage Testbench Stimulus Generation

**arXiv ID:** 2608.10090 | [PDF](https://arxiv.org/pdf/2608.10090v1)

**作者:** Hejia Zhang `[一作]` (UC San Diego), Jishen Zhao `[通讯]` (UC San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们提出了一种后训练框架，将分阶段监督微调产生的检查点转化为互补的强化学习专家，并通过训练无关的权重平均或自适应多教师OPD将其合并为单一模型。

**💡 创新点**

创新点在于利用分阶段SFT产生的多样化专家并通过奖励门控的任务路由与跳过更新实现对专家多样性的动态利用，从而突破单一RL饱和。

**🔧 技术方法**

使用了执行引导的DAPO RL、最差状态优先的改进、Model Soup权重平均、干预感知的DARE/TIES/DELLA合并以及自适应多教师的on‑policy distillation。

**📊 数据集**

在CVDP-ECov和AutoEval-ECov这两个Verilog测试基准上进行评估。

**📈 对比分析**

与现有通用、编码和硬件/验证专用的4B–671B模型比较，单模型4B经过该框架后在CVDP-ECov agentic refinement下的Pass@1达到88%，比DeepSeek‑R1（671B）高出13.5个百分点。

**⚠️ 局限性**

实验仅局限于硬件测试基准，所获多样性来自特定的SFT课程，未验证其在其他RL领域或不同课程下的通用性。

---

## 322. FlowScout: From Execution Feedback to Reliable Tool-Using Agent Workflows

**arXiv ID:** 2608.10039 | [PDF](https://arxiv.org/pdf/2608.10039v1)

**作者:** Shuo Hao `[一作]` (Fudan University), Xin Peng `[通讯]` (Fudan University)

**通讯引用:** 14757 | [OpenAlex ID](https://openalex.org/A5071724015)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种基于执行反馈的自动生成工具集成 agentic workflow 的框架，能够从历史任务求解记录中挖掘工具协作骨架并通过蒙特卡洛树搜索优化工作流拓扑；

**💡 创新点**

其创新点在于将真实工具调用节点嵌入工作流图、使用执行反馈指导搜索，并构建可重用且更稳定的工作流结构；

**🔧 技术方法**

采用工具协调骨架挖掘、图搜索（蒙特卡洛树搜索）、LLM 作为评判者（G‑eval）、以及 LLM 与工具调用节点的混合工作流模型；

**📊 数据集**

使用 ToolBench 四个领域（金融、体育、旅游、天气）的任务记录，分别拆分为 Seen/Unseen 工具集合进行实验；

**📈 对比分析**

与 PM4Py、ReAct 与 AFlow 进行对比，实验显示生成的工作流在工具调用正确率上提升约92.69%，执行分数提升约17.66%，并在多次运行中 CV 降低约62%；

**⚠️ 局限性**

局限性包括仅适用于较短的工具调用链（≤6）、对记录噪声敏感、仍受 LLM 生成不确定性影响，并且在更大规模工具集与更长链路的任务中效果尚未验证。

---

## 323. Your LLM, Your Style: Behavioral Mode Axes for LLM Behavioral Control

**arXiv ID:** 2608.10703 | [PDF](https://arxiv.org/pdf/2608.10703v1)

**作者:** Haoze Liu `[一作]` (Shanghai Jiao Tong University), Na Zou `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于情境行为数据（B‑data）的框架，用来测量和控制大型语言模型（LLM）的行为人格，并构建了3,200个对照情境探针，覆盖20种行为模式和四种交互注册；

**💡 创新点**

核心创新在于：①用情境探针替代自报问卷，使人格评估基于可观测的选择、建议和执行行为；②提出可在激活空间中对齐的行为模式轴（Behavioral Mode Axes, BMAs），区分思考层（thought‑derived）和响应层（response‑derived）；③发现这些轴在特定层（Behavioral Control Layer, BCL）内可实现跨模型、跨注册的干预。

**🔧 技术方法**

采用对比激活添加（Contrastive Activation Addition）和稀疏自编码器分析，提取并正则化激活向量；使用层级激活加法实现模型推理时的实时干预；通过解析生成的答案来评估干预效果。

**📊 数据集**

构造的情境探针集合（3,200个）基于验证的心理测量维度（BFI‑2, DOSPERT, HEXACO等），并在七种开源权重模型上进行测试：Llama‑3.1‑8B/70B‑Instruct、Qwen2.5‑7B/14B/32B‑Instruct、Gemma‑2‑2B/9B‑it（含细调版本）。

**📈 对比分析**

与传统自报问卷结果相比，行为档案显示平均差距22.7个百分点，且跨注册相关系数平均0.76；在层级扫掠中，BCL区间的净方向范围最高可达0.89；思考层BMAs在保持低漂移（未知率<1%）的前提下，控制效果显著优于响应层BMAs。

**⚠️ 局限性**

局限性包括：①仅覆盖20种预设行为模式，无法完全捕捉模型的所有人格维度；②在某些模式下响应层BMAs易产生特征漂移，导致行为与预期不符；③情境探针设计仍需人工审查，可能存在潜在偏倚；④跨模型的BCL位置不同，需进一步统一方法。

---

## 324. From Detection to Understanding: TAR and TAR-Bench for Multi-Task Traffic Anomaly Reasoning

**arXiv ID:** 2608.10317 | [PDF](https://arxiv.org/pdf/2608.10317v1)

**作者:** Han Zhang `[一作]` (NVIDIA), Tomasz Kornuta `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 TAR（Traffic Anomaly Reasoning）数据集及其评测基准 TAR-Bench，构建了覆盖问答、时间推理和场景理解的10种任务，并为每个任务提供链式推理轨迹；

**💡 创新点**

创新点在于将交通异常检测提升为多任务推理，利用 MAVEN 生成结构化事件描述与问答链，结合自动化与人工校正的评测集，形成首个多任务、链式推理的交通视频语言理解资源；

**🔧 技术方法**

技术实现包括 Gemini 3.1 Pro、Gemma‑4‑31B 等多模态 LLM 的多阶段标注流水线（MAVEN）以及基于链式推理的监督微调；

**📊 数据集**

使用了来自八大公开交通异常数据集（SO‑TAD、TADBenchmark、UCF‑Crime、Highway Traffic、TAD、VAD‑R1、Barbados Traffic、Accident‑Bench）的3,670段视频，共44,040条训练注释，评测集包含17条YouTube视频裁剪的960条人手校正注释；

**📈 对比分析**

对11种视觉‑语言模型进行了零射击与 TAR 微调比较，零射击平均准确率约为30‑48%，通过多任务微调可提升约21‑23分，最优模型 Cosmos3‑Super 在零射击下获得48.8%平均分，微调后 CR2‑8B 达到55.7%；

**⚠️ 局限性**

局限性包括训练标注为自动生成，可能存在幻觉与错误；数据集来源有限，覆盖的摄像头与场景多为固定监控，评测集规模相对较小；评估指标主要基于 BERTScore 与 IoU，无法完全衡量因果与逻辑推理的正确性。

---

## 325. MRIComp4Flow: Compression of 3D Brain MRI for Training Multi-Modal Generative Models

**arXiv ID:** 2608.10291 | [PDF](https://arxiv.org/pdf/2608.10291v1)

**作者:** Lisa K. Fischer `[一作]` (Technical University of Munich), Sandeep Nagar `[通讯]` (Munich Center for Machine Learning)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并实现了 MRIComp4Flow，利用 JPEG2000/JPEG‑LS 对 3D 脑 MRI 进行压缩，并在解压后使用 Wavelet Flow Matching 训练多模态生成模型，以在压缩数据上合成缺失的影像序列。

**💡 创新点**

首次量化展示深度生成模型在 3D MRI 上可承受高达 20:1 的压缩而不显著损失合成质量，并提出压缩可作为去噪正则化提升生成效果的可能性。

**🔧 技术方法**

采用 JPEG2000 与近无损 JPEG‑LS 编码、基于 Haar 小波的 Wavelet Flow Matching（WFM）模型以及 3D 直方图归一化和随机访问加载技术。

**📊 数据集**

使用公开的 BraTS 2024 低瘤大脑肿瘤数据集（T1n、T1c、T2w、T2f 共四模态）。

**📈 对比分析**

将压缩比 1–400 的 JPEG2000 与不同近无损水平的 JPEG‑LS 训练的模型与原始无压缩模型进行 PSNR、SSIM 及下游肿瘤分割 Dice/HD95 比较；结果表明 20:1 压缩下合成质量与未压缩相当（ΔPSNR<1dB、ΔSSIM<0.02），甚至到 100:1 仍保持高质量；分割 Dice 在全肿瘤上达到 0.814，验证压缩不会显著影响宏观结构识别。

**⚠️ 局限性**

压缩后高频纹理和小尺寸结构（如肿瘤核心）容易被模糊，极端压缩（>200:1）会出现明显伪影；对专业放射学评价的缺失、仅限于 BraTS 数据集以及仅评估合成质量而非原始诊断精度等也为研究局限。

---

## 326. SkillLens: Visual Skill Cards for Retrieval-Augmented GUI Action Prediction and On-Policy Distillation

**arXiv ID:** 2608.10775 | [PDF](https://arxiv.org/pdf/2608.10775v1)

**作者:** Zhou Liu `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并利用视觉技能卡（VSC）为固定的视觉语言模型执行器提供视觉程序记忆，提升 GUI 动作预测能力，并提出 CardDistill 将 VSC 训练成无需运行时检索的学生模型。

**💡 创新点**

将多源交互记录统一转换为可检索的视觉技能卡，分离检索与视觉证据加载，实现低成本运行时检索；并提出基于 VSC 的教师‑学生蒸馏算法 CardDistill。

**🔧 技术方法**

使用视觉语言模型（VLM）作为执行器，文本检索+视觉证据检索、图像裁剪与对齐、基于视觉证据的查询与重排序，以及基于 VSC 的对齐蒸馏损失。

**📊 数据集**

Multimodal‑Mind2Web、WebLINX‑BrowserGym 与 OSWorld‑G 三个 GUI 基准数据集。

**📈 对比分析**

在冻结的 GPT‑5.4‑mini、GPT‑4o、Gemini、Qwen3‑VL‑2B 等模型上，SkillLens 使 Mind2Web 的 Step SR 提升 11.6 分、WebLINX‑BG Overall 提升 2.9 分；CardDistill 进一步提升 Mind2Web Step SR 12.0 分、WebLINX‑BG Overall 3.2 分。

**⚠️ 局限性**

检索效率与视觉证据选择仍受限，模型仍需外部检索或蒸馏，难以在极端资源受限环境下直接部署；VSC 的生成与审核依赖 LLM 或人工，可能引入误导信息。

---

## 327. InSight-doc: Agentic Visual Perception for Long-Document Understanding

**arXiv ID:** 2608.10628 | [PDF](https://arxiv.org/pdf/2608.10628v1)

**作者:** Kaican Li `[一作]` (Hong Kong University of Science and Technology), Nevin L. Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于agentic视觉感知的InSight-doc框架，先用低分辨率快速扫描长文档，再通过主动放大关键区域实现多轮推理，完全不依赖外部检索。

**💡 创新点**

创新点在于将分辨率作为可调节的推理资源，实现从全局低分辨率到局部高分辨率的粗细层级推理，并将SFT与RL相结合，形成端到端的主动感知学习管道。

**🔧 技术方法**

采用Qwen3-VL-8B-Instruct为基础模型，结合SFT+RL（GRPO）训练，并使用自定义的Zoom-in工具调用构成链式思维过程。

**📊 数据集**

使用包含17.9K SFT示例和19.2K RL示例的主动感知语料，来源于arXiv、DUDE、DocVQA、InfographicVQA、Paper2Poster、MapTab等六大数据集，构建多页、多跳、多种问题类型的长文档VQA数据。

**📈 对比分析**

与现有基线相比，InSight-doc在DUDE、MP-DocVQA、MMLongBench-Doc、LongDocURL等任务上提升4.3–16.4个百分点；在长文档上幻觉率下降40%+，推理延迟降低41–68%，并在高分辨率VQA上亦表现优异。

**⚠️ 局限性**

局限性包括仅在Qwen3-VL-8B-Instruct上验证，未尝试更大或不同模型；RL奖励仅为二元准确率，缺乏更丰富的奖励设计；在极长或极复杂布局的文档上可能仍需进一步改进。

---

## 328. Signpost Watermarking: Joint Optimization for Visual Watermark Coexistence

**arXiv ID:** 2608.10091 | [PDF](https://arxiv.org/pdf/2608.10091v1)

**作者:** Shruti Agarwal `[一作]` (Adobe Research), John Collomosse `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种显式的 decoder‑aware 训练框架，能够在图像和视频中生成可与多种独立水印系统共存的显式“signpost”水印，并在双重嵌入下保持高视觉质量和解码鲁棒性。

**💡 创新点**

创新点在于：①将多水印共存视为可优化目标，直接在训练中加入冻结的二级水印解码器；②首次证明独立视频水印也能共存并通过显式优化提升；③通过 sparsity、JND 引导与多目标损失，使得 signpost 残差与各水印在空间频率上互补。

**🔧 技术方法**

使用 UNet+ResNet-50 的 encoder‑decoder 结构；JND 纹理掩模；LPIPS 感知损失；多目标损失组合（图像重建、二进制解码、共存解码、JND 加权像素误差）；三阶段训练课程；AdamW 优化器。

**📊 数据集**

图像：大型专有数据集，评测 1000 张 MirFlickr‑1M 验证集；视频：SA‑V 数据集 51k/155 训练/验证集，分辨率 256×256。

**📈 对比分析**

与四种图像水印（PixelSeal、InvisMark、TrustMark‑P、MaskWM）及 ZOETROPE 对比；与三种视频水印（VideoSeal、TrustMark‑Q、FlowMark）及 FM‑32 对比。实验表明 signpost 单独 PSNR 达 52.4 dB，VMAF 99.2；在双重嵌入时 PSNR 仍高于 44 dB，VMAF 接近 99。共存误差下降幅度显著，尤其对 TrustMark 系列提升高达 +0.24 bit accuracy，整体比基线提升 15–20% 的解码成功率。

**⚠️ 局限性**

局限性包括：1) 对抗性攻击（如针对双重水印的专门攻击）尚未评估；2) 仅针对已有水印模型的共存，未知对新加入的未冻结水印的兼容性；3) 训练需要预先冻结多种解码器，部署成本相对较高；4) 在极端压缩或重编码下的鲁棒性仍有限。

---

## 329. FaCTz: Fast Critical-Point and Topology-Aware GPU Compression for Scientific Vector Fields

**arXiv ID:** 2608.10586 | [PDF](https://arxiv.org/pdf/2608.10586v1)

**作者:** Mingze Xia `[一作]` (Oregon State University), Xin Liang `[通讯]` (Oregon State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了GPU端误差上界失真压缩算法，保证科学向量场中的临界点不被丢失、生成伪点或变更类型，解决了传统压缩器仅满足点误差但破坏拓扑的问题。

**💡 创新点**

将临界点保护从本质上是耦合的顺序约束转化为可并行的块级安全边界和推测式逐点验证两种模式；首次在GPU上实现可保证临界点的失真压缩，并提供可调的吞吐/压缩比权衡。

**🔧 技术方法**

采用SZ风格的预测+量化+熵编码框架；使用CUDA并行实现；块级模式利用单精度安全边界推导和块最小化；推测式模式采用多分辨率递归并行验证回退；使用ANS熵编码和双精度检查保证拓扑安全。

**📊 数据集**

三组二维向量场数据集：Ocean（2400×3600）、DT-10K（10000×10000）和DT-20K（20000×20000），每个数据集包含数十万至数百万个顶点。

**📈 对比分析**

与cuSZ、cuSZ-i、cuSZp、cuZFP、nvCOMP Zstd等GPU压缩器以及CPU cpSZ进行比较；在保证所有临界点的最宽容误差下，块级模式压缩比约2.4–6.7×、吞吐36–60 GB/s；推测式模式压缩比约6–16×、吞吐29–73 GB/s；速度比CPU cpSZ快约640×，同时保持高PSNR与零临界点损失。

**⚠️ 局限性**

推测式模式相对块级模式慢；目前仅支持二维向量场；对极大网格仍需进一步调优；对更高维或更复杂拓扑（如三维向量场或张量场）的拓扑保护尚未覆盖。

---

## 330. Training Set Synthesis for Bioacoustic Denoising: A Case Study With Mice

**arXiv ID:** 2608.10054 | [PDF](https://arxiv.org/pdf/2608.10054v1)

**作者:** Reyhaneh Abbasi `[一作]` (Acoustics Research Institute of the Austrian Academy of Sciences), Nicki Holighaus `[通讯]` (Acoustics Research Institute of the Austrian Academy of Sciences)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种基于频谱里程碑（ridge）合成训练集并训练U‑Net denoiser的生物声学降噪方法，专门用于处理噪声严重的鼠类超声唤声（USV）信号。

**💡 创新点**

创新点在于：①利用自动里程碑检测来合成近似无噪声训练样本，避免需要手工标注；②在损失函数中加入里程碑加权，使网络更关注信号主轨迹；③将这些技术组合在一个端到端的深度学习降噪框架中。

**🔧 技术方法**

主要技术包括多频谱重分配（MTRS‑STFT/CQT）、多成分里程碑跟踪、基于高斯掩模的清晰信号合成、复比率掩模（cRM）预测以及带里程碑加权的U‑Net网络。

**📊 数据集**

使用来自野生与驯化大鼠的超声唤声数据集（约12,937条USV），并利用自动提取的里程碑生成合成训练和测试数据。

**📈 对比分析**

与传统谱减法、noisereduce（稳态与非稳态模式）以及预训练的Biodenoising模型对比。实验表明：在合成测试集上，SI‑SDR提升从-10.17 dB到16.52 dB（最低SNR），在实际场景里程碑跟踪误差从1.4 kHz降至0.8 kHz，USV分类宏观F1从原始的83%提升到89%，在外部样本上从69%提升到72%。

**⚠️ 局限性**

局限性包括：对极低SNR下极弱的USV仍可能无法完全恢复；高SNR时可能出现微量SI‑SDR下降；方法仅适用于具有明显里程碑结构的声学信号，对宽带或噪声化信号不适用；合成训练集对里程碑精度依赖较高，错误的里程碑会引入伪噪声。

---

## 331. When Do Anchor-Based Pointwise LLM Rerankers Help? Retriever Quality, Statistical Scope, and Anchor Design

**arXiv ID:** 2608.10528 | [PDF](https://arxiv.org/pdf/2608.10528v1)

**作者:** Utshab Kumar Ghosh `[一作]` (Missouri University of Science and Technology), Shubham Chatterjee `[通讯]` (Missouri University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

复现并系统分析 Anchor‑based pointwise LLM reranker（GCCP/PAGC）的有效性及其受检索质量、聚合策略和 anchor 构造的影响。

**💡 创新点**

发现核心对比度评分有效，但聚合和谱 MDS anchor 构造并非必要；方法的收益取决于第一阶段检索器性能、统计校正和 anchor 设计，揭示了其条件性。

**🔧 技术方法**

使用 Flan‑T5（Large/XL/UL2）等 LLM 进行 RG‑YN 与 GCCP 评分，采用 A/B prompt 与 min‑max 归一化，随后通过线性聚合得到 PAGC；anchor 通过谱 MDS 或简单句子拼接构造；评估采用 BM25 与 E5/BGE 等检索器；对比使用对偶 bootstrap 与 Holm‑Bonferroni 校正。

**📊 数据集**

在 TREC Deep Learning 2019/2020、以及八个 BEIR 子集（TREC‑COVID、Touché‑2020、DBPedia‑Entity、SciFact、Signal1M、TREC‑News、Robust04、NFCorpus）上进行实验。

**📈 对比分析**

相较于单点 RG‑YN，GCCP/PAGC 在 BM25 检索下显著提升 nDCG@10；在强检索器（E5、BGE）下提升幅度明显降低，聚合贡献有限；简单 anchor（Top‑3 句子拼接）可匹配或优于谱 MDS。

**⚠️ 局限性**

受限于检索质量、聚合权重、anchor 设计与统计检验方式；在强检索器或实体检索任务中效果不稳定；对实现细节和参数调优高度敏感，复现难度大。

---

## 332. One Recipe, Many Harnesses: What Self-Evolution Encodes Across Languages and Models

**arXiv ID:** 2608.10178 | [PDF](https://arxiv.org/pdf/2608.10178v1)

**作者:** Siqi Yang `[一作]` (University of Illinois Urbana Champaign), Martin Hirzel `[通讯]` (Ibm)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究自我进化的编码代理 harness 在多语言、多模型环境中的功能与结构，通过固定演化方案对每个（语言、模型）组合进行自适应迭代，构建可解释的 harness；

**💡 创新点**

创新点在于将 Typed Routing 与 Instrumented Attribution 结合成 TRIAGE 演化框架，能够将每一次 harness 编辑与具体故障信号关联，并通过保持演化流程不变，剖析不同语言与模型下 harness 所编码的通用核心与生态特定差异；

**🔧 技术方法**

采用 LLM 进行失败诊断和编辑决策，利用 verifier 与预算两侧证据进行 typed routing；对 harness 进行自动化标签（抽象概念与生态标记）匹配；

**📊 数据集**

使用 Multi‑SWE‑Bench 的八种编程语言（Python、Java、C++、C、Go、Rust、JavaScript、TypeScript）以及三种模型（Claude Haiku 4.5、GPT‑5‑mini、DeepSeek‑V4‑Flash）进行实验；

**📈 对比分析**

对比三种 harness（最小种子、手工 mini‑SWE‑agent、演化 harness），评估在 held‑out 任务集上的 mean_solve@3；演化 harness 在大多数组合上显著提升，获得 20–40% 的生态特定增益，且其共享核心可部分迁移，性能提升幅度视语言生态而定；

**⚠️ 局限性**

局限在于只覆盖单一演化周期、单一任务域（代码修复）和有限的语言/模型组合；未探究长期迭代、跨任务迁移及更细粒度的缺陷类型，且对非检测到的缺陷与模型能力提升的影响未知。

---

## 333. An Asynchronous Triggered MAC Protocol for Underwater Acoustic Networks

**arXiv ID:** 2608.10533 | [PDF](https://arxiv.org/pdf/2608.10533v1)

**作者:** Bingwen Huangfu `[一作]` (Jilin University), Jun Liu `[通讯]` (Beihang University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种名为 AT-MAC 的异步触发 MAC 协议，解决水声网络中时钟同步开销大、传播延迟变异导致的频道利用率低和公平性差的问题。

**💡 创新点**

创新点：① 引入 Triggered Slot 机制，完全摆脱同步时隙约束；② 在此基础上设计 Time‑Offset 异步 MAPPO（多智能体近端策略优化），使用时间偏移观测编码和时间归一化折扣实现对异步交互的稳定学习；③ 设计低开销、基于局部观测的负载感知公平守护器，动态抑制偏差传输以维持网络整体公平。

**🔧 技术方法**

使用技术：多智能体深度强化学习（MAPPO）+To‑Dec‑POMDP建模 + 时间偏移观测编码 + 负载感知公平守护机制 + 集中训练/分布式执行（CTDE）框架。 另外实现了模型压缩、低功耗推理等硬件实现技术。

**📊 数据集**

数据集与实验：① 真实场景轨迹（Danjiang Lake、Jiaozhou Bay、Songhua Lake）用于 trace‑based 训练与验证；② ASUNA 公共数据集（Garda、Hadera、Werbellin 湖）用于评估泛化；③ 合成大规模拓扑（Syn‑7、Syn‑8）用于评测可扩展性；④ 现场硬件实验（嵌入式节点）验证能耗与推理延迟。

**📈 对比分析**

比较方法：对齐 TDMA、ALOHA、UW‑ALOHA‑Q、Async‑DL‑MAC、DR‑DLMA、MOMA‑MAC 六类基线。评估指标包括吞吐率、成功率、端到端延迟、能效以及 5% 分位公平度。实验结果表明：AT‑MAC 在大多数情形下实现了 30–40% 的吞吐率提升，成功率保持在 95% 以上，延迟最低，能效最高，并在所有拓扑下公平度均优于基线，甚至在大规模网络中与 TDMA 接近但性能更稳健。

**⚠️ 局限性**

局限性：① 对训练以外的流量负载（轻载或极高载）泛化仍有限；② 网络规模增大时可利用的空间时间复用降低，性能趋近 TDMA；③ 公平守护器的阈值 ϵ 需要人工调优，过大或过小均会影响吞吐与公平；④ 目前仅在单跳、中心式接收节点的场景验证，跨跳多跳网络的路由与 MAC 协同仍待研究。

---

## 334. MMArt A Multi-Perspective Multimodal Dataset for Visual Art Understanding

**arXiv ID:** 2608.10706 | [PDF](https://arxiv.org/pdf/2608.10706v1)

**作者:** Shuai Wang `[一作]` (University of Amsterdam), Marcel Worring `[通讯]` (University of Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了MMArt数据集，包含74,234幅WikiArt画作，每幅画配有四种专业视角（叙事、形式、情感、历史）以及一条统一描述。

**💡 创新点**

创新点在于：①多视角并行标注，②使用专用vision‑language模型对每个视角进行专业生成，③对视角互补性进行生成重构与检索双向分析，④通过调和步骤产生统一描述。

**🔧 技术方法**

技术手段包括：GalleryGPT、ArtRAG、Qwen3‑VL‑Instruct等专用VLM进行视角生成；文本‑图像生成器FLUX.2‑Klein与Qwen‑Image用于重构评估；检索嵌入使用Qwen3‑VL‑Embedding‑2B与Jina‑CLIP‑v2；CLIP、DINOv3用于多维度相似度；Gemma‑3‑27B作为LLM‑as‑judge进行质量评估。

**📊 数据集**

使用的基础数据集为WikiArt（约75k幅画）并结合ArtEmis情感标签；通过模型生成获得四视角及统一描述，最终得到74,234幅完整标注的样本。

**📈 对比分析**

通过生成重构（CLIP‑style、DINOv3‑composition、情感一致性）和检索实验（Recall@k、MRR、NDCG）对比，发现叙事视角在检索最强（R@1≈44%），形式视角在重构最优（CLIP相似度最高），统一描述略优于单视角。单一视角无法覆盖所有任务，验证了多视角设计的必要性。

**⚠️ 局限性**

局限性包括：①视角生成受限于预训练模型，可能出现事实错误或信息缺失；②情感视角依赖Crowd数据，情感多样性有限；③缺乏人工多视角真值对比；④对跨文化、跨语言解释支持不足。

---

## 335. Mitigating Context Interference for Reliable and Efficient Search Agents

**arXiv ID:** 2608.10743 | [PDF](https://arxiv.org/pdf/2608.10743v1)

**作者:** Boyang Xue `[一作]` (Chinese University of Hong Kong), Aldo Lipani `[通讯]` (University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多轮检索代理的上下文干扰问题，并提出基于知识蒸馏的动态上下文精炼器，随后将其融入强化学习训练流程，以提升代理的可靠性和效率。

**💡 创新点**

提出“先精炼再生成”范式，利用高级LLM生成的关键文本构建精炼数据集，实现弱LLM的精炼能力；将精炼器嵌入RL roll‑out，显著降低干扰并提高生成质量。

**🔧 技术方法**

使用LLM检索代理（Qwen‑2.5‑7b/3b），知识蒸馏（GPT‑4为教师），强化学习（GRPO/PPO），自监督微调与检索增强生成（IRCoT）等技术；对比压缩（GPT‑Compress）和自我精炼（Self‑Refine）等基线。

**📊 数据集**

闭卷问答基准：Natural Questions、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle。

**📈 对比分析**

与IRCoT、Search‑GRPO、Search‑o1、SFT、RFT等方法对比，EM提升约1–2个百分点，平均检索次数（ART）和上下文长度显著下降，推理时间（AIT）亦降低。

**⚠️ 局限性**

仅针对检索代理任务，未涵盖工具使用或规划等其他agent场景；精炼模块作为外部辅助，未内置于模型本身；需要进一步探索更广泛任务中的干扰来源与精炼方法。

---

## 336. Whole-Body Planning for Humanoids Navigating Confined Spaces via Self-Collision Avoidance References

**arXiv ID:** 2608.10220 | [PDF](https://arxiv.org/pdf/2608.10220v1)

**作者:** Carlos Gonzalez `[一作]` (University of Texas at Austin), Luis Sentis `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个三阶段的全身规划框架，用于在高度受限的环境中实现仿人机器人平稳穿越，包含基于刚体体积的运动学路径规划、可微自碰撞规避以及动力学一致性的轨迹优化，并利用生成的高质量轨迹训练残差强化学习策略以实现在线控制。

**💡 创新点**

创新点在于将运动学路径规划直接投射到可达的刚体体积上，结合可微碰撞检测与可达性约束生成体积感知引导，采用两步优化策略（先忽略硬碰撞再硬约束）提升求解鲁棒性，并通过残差强化学习将离线轨迹迁移到在线控制。

**🔧 技术方法**

使用了 SOCP、CasADi 自动微分、SQP 求解器、可微距离计算、逆运动学、残差强化学习（PPO）、对抗域随机化、以及对 Unitree G1 机器人全关节动力学模型。

**📊 数据集**

数据集主要为 Unitree G1 机器人在三个 NIST 紧凑环境（Unobstructed Hole、Obstructed Hole、Tilted Stairs）中的仿真测试，评估涉及 10 组随机起始位置。

**📈 对比分析**

与基准 spline 避碰规划和线性插值方法相比，该框架在所有环境中均实现 10/10 的成功率（最高可达 10/10），求解时间平均为 2–3 分钟；残差 RL 策略在完全随机化条件下成功率超过 95%，显著优于传统方法。

**⚠️ 局限性**

局限性包括：计算时间仍较高，主要在仿真中验证，未在真实硬件上测试；依赖预设的接触序列；允许有限的碰撞穿透；仅针对 Unitree G1 仿人机器人，通用性待进一步验证。

---

## 337. MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection

**arXiv ID:** 2608.10459 | [PDF](https://arxiv.org/pdf/2608.10459v1)

**作者:** Jinmo Han `[一作]` (Seoul National University), Nam Soo Kim `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MD-ProTector，一种仅使用输入文本的轻量编码器检测器，通过多原型银行分别表示人类和LLM生成文本，并直接利用原型相似度做判别。

**💡 创新点**

创新点在于引入 Prototype Positioning 损失，先分离类级方向，再用样本残差对每个原型进行数据驱动定位，解决多原型之间的冗余和重叠问题。

**🔧 技术方法**

采用无监督 SimCSE‑RoBERTa 作为编码器，使用三种损失（Prototype‑to‑Class、Sample‑to‑Prototype、Prototype Positioning）联合训练；原型初始化为 K‑means，训练过程中对原型进行归一化和残差投影。

**📊 数据集**

使用 MAGE、RAID、M4 三大基准，涵盖域/生成器多样性、对抗攻击以及多语言情况，并进行未见域/未见生成器的 leave‑one‑out 评估。

**📈 对比分析**

在相同数据、模型和训练预算下与 Binary CE、SupCon、DeTeCtive、DSVDD 等方法对比，MD‑ProTector 在 MAGE CDCM 和 RAID 上 AvgRec 最高（95.14/88.18），在 RAID 上 AUROC 最高、FPR95 最低；在域/生成器/语言偏移场景中排名第二或最优。

**⚠️ 局限性**

局限包括仅适用于二分类固定标签，未处理混合生成、编辑协作或机器介入程度估计；原型数量为经验固定，缺乏自适应调整；未解决持续学习和分布漂移问题，实际部署可能产生误报。

---

## 338. The Impact of Operational-Data Fidelity when Assessing Safety-Critical Autonomous-Vehicle Software

**arXiv ID:** 2608.10025 | [PDF](https://arxiv.org/pdf/2608.10025v1)

**作者:** Kizito Salako `[一作]` (University of London), Rabiu Tsoho Muhammad `[通讯]` (University of London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究安全关键软件可靠性评估中操作数据细节不足的影响，并基于保守贝叶斯推断（CBI）扩展，给出了针对失效类型不确定性下的保守置信下界与阈值计算方法；

**💡 创新点**

创新点在于：①提出对失效类型不确定性的保守贝叶斯推断框架；②给出四个定理给出不同假设下的最小置信下界；③通过AV安全评估案例展示粗粒度数据可能导致过度乐观的风险；④提供可直接应用于ISO 26262/ISO 21448等标准的实践指导；

**🔧 技术方法**

使用技术包括保守贝叶斯推断（CBI）、二维概率空间划分、离散先验构造、定理证明与数值仿真；

**📊 数据集**

数据集为假设性的自动驾驶监测器操作日志（无公开真实数据），利用模拟实验生成不同失效类型计数与无失效情形；

**📈 对比分析**

与传统单维CBI方法比较，结果显示在考虑失效类型不确定性时需要更大量的无失效样本才能达到同等置信水平，且在某些情形下保守度甚至趋于无限；

**⚠️ 局限性**

局限性包括：①仅考虑两种失效模式，未涵盖更复杂成功/失败结构；②假设i.i.d.和静态操作环境，未处理环境漂移和分布变化；③未将分类器失效率直接映射至系统级安全指标（如事故率、里程等）。

---

## 339. N2NMatcher: Towards Inlining-Resilient Binary Decomposition and Module Matching

**arXiv ID:** 2608.10043 | [PDF](https://arxiv.org/pdf/2608.10043v1)

**作者:** Ang Jia `[一作]` (Dalian University of Technology), Xiaochen Li `[通讯]` (Dalian University of Technology)

**通讯引用:** 1490 | [OpenAlex ID](https://openalex.org/A5100328760)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种针对程序级二进制代码相似性分析（BCSA）的函数内联鲁棒二进制分解与模块匹配框架。

**💡 创新点**

核心创新在于利用源级验证的稳定边界函数作为锚点，构建层次化的ACFG-FCG图神经网络进行锚点预测，基于锚点进行模块分解，并使用模块图嵌入结合语法、语义与图相似度实现模块匹配，显著提升了在函数内联变化下的鲁棒性。

**🔧 技术方法**

采用层次化图神经网络（包含ACFG层与FCG层）进行特征编码；使用阈值判定锚点；构造模块图并通过多层图消息传递与注意力池化获得模块嵌入；通过对比学习与多重相似度（语法、语义、图）进行模块匹配。

**📊 数据集**

主要使用BinKit LTO子集（3024 x86‑64 二进制，28 项目）进行训练与交叉验证，并在ISRD数据集（608 二进制，26 项目）进行跨项目验证。

**📈 对比分析**

与ModX与BMVul等基线在 BinKit 上进行二进制分解和模块匹配评估，平均模块重叠度（MDQ）最高、Top‑1 相似度、Recall@1 与 MRR 亦显著优于基线；在 ISRD 上的跨项目检索指标提升约 30%；运行时开销相对 ModX 仅增加约 28%。

**⚠️ 局限性**

局限性包括：锚点标注依赖调试信息，可能在无调试或强优化下失效；实验仅覆盖 x86‑64 LTO 架构，未验证对其他架构、无 LTO 或混淆二进制的适用性；跨项目迁移性能仍低于同项目比较，需进一步改进模型泛化与负样本挖掘。

---

## 340. Multi-Granular Rationale-Guided Molecular LLM for Property Prediction

**arXiv ID:** 2608.10480 | [PDF](https://arxiv.org/pdf/2608.10480v1)

**作者:** Junwoo Park `[一作]` (Sungkyunkwan University), Sujee Lee `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出一种多粒度推理提示的分子大语言模型（MR‑MoL），通过将基于图神经网络的子结构重要性排序（含Murcko骨架、BRICS片段及功能基）序列化为文本提示，直接提供给LLM与分子图/SMILES共同参与属性预测。

**💡 创新点**

创新点在于首次将GNN的子结构归因结果以可读文本形式嵌入LLM提示，实现模型对单个子结构方向、权重和排名的直接解读与利用。

**🔧 技术方法**

采用的技术包括图神经网络（基于Molecule‑BERT预训练）进行子结构掩码归因、Q‑Former投影将分子图映射至LLM嵌入空间、双阶段训练（图-语言对齐与多任务指令调优）以及LoRA微调。

**📊 数据集**

实验使用了PubChem/DrugBank/Mol‑Instructions/ChEBI-20等多源分子文本对进行对齐训练，以及8个MoleculeNet任务（BACE、BBBP、ClinTox、HIV、SIDER、Tox21、ESOL、Lipo）进行属性预测评估。

**📈 对比分析**

与七个专用模型（GNN或单任务LLM）及五个通用LLM进行对比，MR‑MoL在大多数任务上达到或接近最优，显著提升ROC‑AUC/减小RMSE，尤其在BACE、SIDER等分类任务上超过11点，缩小了与专用模型的性能差距。

**⚠️ 局限性**

局限性包括依赖来源GNN归因的质量、未涵盖宏环与立体化学等结构视角、仅适用于可归因的分类/回归任务，且当源模型产生误导性归因时可能导致模型误判。

---

## 341. Online Interval Selection on a Simple Chain

**arXiv ID:** 2608.10376 | [PDF](https://arxiv.org/pdf/2608.10376v1)

**作者:** Yaqiao Li `[一作]` (Shenzhen University of Advanced Technology), Denis Pankratov `[通讯]` (Concordia University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在线区间选择问题在简单链（unit‑length 区间只与相邻区间相交）上的表现，提出并分析了一个单向可撤销的确定性无记忆算法（Revoke‑to‑the‑Left）。

**💡 创新点**

创新点在于：①在随机顺序模型下给出了该算法的竞争比为 2(1−1/√e)≈0.786；②证明任何确定性可撤销算法在对抗模型下竞争比上限为 3/4；③基于此构造得到该问题的提示复杂度下界为 n/4，首次给出提示复杂度的理论下界。

**🔧 技术方法**

使用了组合计数、递推关系、生成函数、积分表示以及对抗输入构造等数学技术；同时结合提示复杂度理论进行分析。

**📊 数据集**

实验部分未使用实际数据集，全部以理论模型（n 个单位长度区间的简单链）为基础进行分析。

**📈 对比分析**

与不允许撤销的基本贪心算法（竞争比≈0.864）以及对抗模型中任何可撤销算法的上界（≤0.75）进行对比。该算法在随机顺序下获得 0.786 的竞争比，优于对抗上界但略低于贪心算法。

**⚠️ 局限性**

局限性包括：仅适用于简单链模型，单向撤销策略；未证明在更一般的区间图上可取得同样表现；随机顺序假设在实际应用中可能不成立。

---

## 342. Frozen Brain-MRI Foundation Models Are Site Fingerprints

**arXiv ID:** 2608.10295 | [PDF](https://arxiv.org/pdf/2608.10295v1)

**作者:** Saman Rahbar `[一作]` `[通讯]` (University of British Columbia), Saman Rahbar (University of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `da1b1a89-583a-4b57-9c81-478778569bec` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

对多站点脑MRI数据中冻结的基础模型（FM）嵌入进行系统审核，量化其对扫描站点信息的可解码程度，并探索在保持解剖信息的前提下去除站点相关伪影的方法。

**💡 创新点**

发现站点信息是嵌入的内在、无学习的成分；通过随机初始化和原始图像基线验证证明该特征来源于低层图像统计而非预训练；提供训练无关的去除技术（INLP/ComBat）并评估其对稠密分割的影响。

**🔧 技术方法**

使用线性与MLP探针、迭代零空间投影 (INLP)、ComBat 调和、SwinUNETR、ViT、ResNet 等多种冻结编码器、SynthSeg 生成银标签、ABIDE-I/II 数据集、留一站点交叉验证和 Dice 评估等技术。

**📊 数据集**

ABIDE-I（546 份 T1 影像，6 站点）和 ABIDE-II（989 份 T1 影像，15 站点）两大独立多站点脑MRI数据集。

**📈 对比分析**

通过多次随机拆分计算线性/非线性探针的平衡准确率，比较站点与临床变量（性别、年龄、自闭症诊断）的可解码比率；结果显示站点在所有层级约 0.9–0.96 的准确率，远高于临床信号；去除后站点可解码率从 0.94 降至 0.07/0.00，全球读出无显著提升，但稠密分割在全尺度去除时 Dice 下降至 0.18。

**⚠️ 局限性**

站点标签同时包含扫描器、协议和人群信息，无法完全分离；仅评估了两种 SwinUNETR 检查点和三大架构，未覆盖更广泛模型；使用 SynthSeg 银标签限制了分割评估；未进行旅行者设计验证，未能彻底区分采集与人口因素。

---

## 343. Real-World Cooperative Bimanual Dexterous Grasp of Large Objects from Single-View Observations

**arXiv ID:** 2608.10383 | [PDF](https://arxiv.org/pdf/2608.10383v1)

**作者:** Ziming Li `[一作]` (University of Auckland), Ning Wang `[通讯]` (Chongqing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6514db3d-8de6-452c-91b7-acdb31787cc4`

**🎯 论文内容**

提出了一个基于双臂多指机器人在真实环境下的协同抓取框架，包含从单视RGB‑D图像生成双手关节配置、运动规划与力感知在线调节，并实现了对大型物体的稳定抓取。

**💡 创新点**

创新点在于：①使用真实遥控演示收集多模态抓取数据，克服仅在仿真中训练的局限；②利用DDPM生成关节级抓取姿态，避免依赖完整三维模型；③将运动规划与力反馈融合实现在线抓取微调，提升执行稳健性。

**🔧 技术方法**

核心技术包括：Segment‑Anything Model (SAM) 做实例分割；PointNet++ 编码分段点云；DDPM 生成关节配置；机器人运动规划 + 低速力导向微调；多指手指力与触觉感知；双臂协同抓取策略。

**📊 数据集**

数据集为 33 种大型物体（共 353 次抓取）通过 Apple Vision Pro 远程操作收集，记录 RGB、深度、关节角度、力矩、触觉等多模态信息，公开于 GitHub。

**📈 对比分析**

与 BimanGrasp‑DDPM、ViSiL‑HD、GraspNet、DexGraspAnything 等基线对比，平均成功率 61.9%（对 8 种不同形状物体），显著高于单手方法和仅在仿真训练的方案；消除关键模块（如力调节、运动规划）会导致成功率下降。

**⚠️ 局限性**

局限性包括：①仅在桌面静态场景验证，复杂环境和动态物体仍未测试；②对极端摆放、尺寸或质量分布不均的物体仍有失败；③当前仅生成目标姿态，未学习完整抓取轨迹，需进一步结合学习式轨迹生成和实时感知。

---

## 344. ChemWorld: Programmable Chemical Worlds for Controlled and Replayable Agent Experimentation

**arXiv ID:** 2608.10792 | [PDF](https://arxiv.org/pdf/2608.10792v1)

**作者:** Jiangjie Qiu `[一作]` (Tsinghua University), Xiaonan Wang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了ChemWorld，一个可编程的化学环境，允许研究者通过组合过程与观测组件来构建、修改并精确重放化学世界，同时提供与代理交互的统一公共接口。

**💡 创新点**

创新点在于把化学世界本身作为可控实验变量，分离公共实验合同与评估者私有化学/材料定律，并通过事务性执行和完整可重现记录实现严格匹配的实验对比。

**🔧 技术方法**

采用可组合组件化学过程与观测模型、兼容性编译器、事务性执行与回放机制，以及开源的Python/Go实现。

**📊 数据集**

使用了注册的64个参考任务-世界单元、52个生成的化学世界组合以及1,786条边界和类别构造作为实验数据集。

**📈 对比分析**

通过覆盖测试、模块与接口完整性测试、负向探针失败语义、资源会计与回放验证等方法，所有测试均通过，表明系统在可重复、可审计、可重现方面性能优异。

**⚠️ 局限性**

局限性包括仅在声明的组件词汇和兼容规则下验证，缺乏对真实物理化学系统的校准和外部基准评估，以及未覆盖更高阶交互和完整化学空间。

---

## 345. EvoMem: Memory-Augmented Evolution for Code Optimization

**arXiv ID:** 2608.10795 | [PDF](https://arxiv.org/pdf/2608.10795v1)

**作者:** Viktor Volkov `[一作]` (AXXX), Ivan Oseledets `[通讯]` (AXXX)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

引入了 EvoMem 持久化记忆架构，在 LLM 驱动的进化式代码搜索中提取、存储并在后续运行中检索成功变异经验，以提升搜索效率和跨任务迁移能力。

**💡 创新点**

将成功变异事件转化为可追溯、任务感知的结构化“记忆卡”，实现离线写入和在线检索，仅提供有限建议而非硬约束，在保持原进化流程不变的前提下实现跨运行知识迁移。

**🔧 技术方法**

结合 LLM（Gemini 3 Flash、Qwen3‑8B）生成变异，使用 MiniLM‑L6 embeddings 进行检索、DBSCAN 聚类、LLM 决策合并、基于 provenance 的 deduplication 与增量存储，构建 GigaEvo 的两阶段记忆管线。

**📊 数据集**

训练集包含几何优化（Circle Packing、Heilbronn、Kissing Number）、多跳问答（HotpotQA、HoVer、GSM8K）以及 GPU 核心优化（KernelBench）和科学代码调优（AlgoTune）等；评估集为相同类别但不同实例的任务，排除目标任务自身的记忆。

**📈 对比分析**

采用相同进化管线的对照实验，比较无记忆与记忆启用的目标指标与搜索速度；结果显示多数基准平均提升约 6 % 目标指标，搜索速度平均提升约 5.9 倍，部分任务如 HotpotQA、KernelBench 甚至实现高达 16 倍速度提升。

**⚠️ 局限性**

记忆效果不均匀，受限于记忆提取、聚类和检索超参；过度依赖记忆可能导致收敛早停；实验规模有限，仅覆盖少数任务；缺乏对大规模、跨文件、持续演化的验证。

---

## 346. SeFaR: Semantic Feature-aware Robustness Testing of Deep Neural Networks

**arXiv ID:** 2608.10289 | [PDF](https://arxiv.org/pdf/2608.10289v1)

**作者:** Nusrat Jahan Mozumder `[一作]` (University of Virginia), Matthew Dwyer `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个基于分层语义特征模型的系统化测试框架，用于生成满足需求预条件的图像并评估视觉模型对语义变化的鲁棒性。

**💡 创新点**

创新点包括可手工扩展的分层概念树、利用文本条件扩散模型生成语义保持但特定特征变化的图像、以及通过VQA与聚类反馈自动发现隐藏影响特征并迭代细化。

**🔧 技术方法**

主要技术为文本条件扩散图像编辑（Qwen‑Image‑Edit‑2511）、视觉语言模型（Qwen2.5‑VL‑7B）、句子嵌入与聚类、结构化提示以及迭代适配机制。

**📊 数据集**

使用了SGSM（模拟驾驶）、RRAV（校园无人车）和AI4MARS（火星地形）三大数据集，分别对应不同任务与环境。

**📈 对比分析**

与传统像素扰动或基于生成模型的扩增方法相比，实验显示预条件保持率约93% 以上，且在满足预条件下约16% 的测试能够暴露模型不鲁棒性。

**⚠️ 局限性**

局限性包括对高质量生成模型与VQA判定的依赖、概念树需要专家手工定义、预条件判定准确性受限，以及目前仅聚焦视觉模型，未涵盖跨模态或多任务情境。

---

## 347. Lesion-Aware Adaptive Fourier Neural Operator for CT-to-PSMA PET Synthesis in Prostate Cancer

**arXiv ID:** 2608.10429 | [PDF](https://arxiv.org/pdf/2608.10429v1)

**作者:** Rashmi Bhaskara `[一作]` (Purdue University), Oluwaseyi M. Oderinde `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了一种基于CT的PSMA-PET合成模型LAFNO，旨在提升病灶信息的重建质量。

**💡 创新点**

创新点在于：①用CT衍生的对比与无序代理通道代替高维放射组学做模型条件；②引入病灶级总活性(TLA)、病灶对比和周围区域损失，实现病灶感知的多目标训练。

**🔧 技术方法**

技术手段包括3D U‑Net+Adaptive Fourier Neural Operator (AFNO)瓶颈、CT对比/无序代理特征注入、四项损失（L1、TLA、对比、周围）以及滑动窗口训练。

**📊 数据集**

使用TCIA PSMA‑PET‑CT‑Lesions数据集，包含^18F‑PSMA（335例）和^68Ga‑PSMA（204例）以及对应的肿瘤掩码。

**📈 对比分析**

与四种基线（AFNO‑L1、Pix2Pix、FlowLet、cWDM）对比，LAFNO在病灶TLA误差上分别达到48.3%/64.0%，在肿瘤核心放射组学ICC最高，整体SSIM/PSNR略低于AFNO‑L1，但保持竞争力。

**⚠️ 局限性**

局限性包括对软组织微小病灶的敏感度不足、周围区域重现性仍差、不同探针/扫描器差异导致性能波动，需要更大、标注更细粒度的数据集和更完善的生物学条件化方法。

---

## 348. Optimize Cheap, Deploy Strong: Cost-Aware Cross-Tier Transfer for Evolutionary Optimization

**arXiv ID:** 2608.10694 | [PDF](https://arxiv.org/pdf/2608.10694v1)

**作者:** Tal Oved `[一作]` (IBM Research), Udi barzelay `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种成本感知的交叉层优化方法，将进化式提示搜索中的三大角色——评估、变异和部署——分别放置在不同价格层级的模型上；通过在最低价模型上评估、在强力模型上进行反射式变异，并在更高价模型上零样本部署，从而显著降低搜索成本。

**💡 创新点**

创新点在于：①将评估层与变异层解耦，利用最低价模型完成大部分评估，从而把成本降至主导；②证明在弱模型上搜索得到的提示可在更强模型上零样本直接迁移，甚至优于在目标层级直接优化的提示；③系统化评估了跨层迁移的有效性，并给出成本与收益的理论与经验边界。

**🔧 技术方法**

技术包括：进化式提示搜索（基于GEPA的反射式变异）、多层模型分配（评估层、变异层、部署层）、跨层提示迁移、成本模型分析与实证验证。

**📊 数据集**

实验使用四个任务的数据集：HotpotQA、IFBench、LiveBench‑Math、HoVer，覆盖多跳问答、指令遵循、数学推理和主张验证。

**📈 对比分析**

与传统同层全成本优化相比，本文方法在11种模型、4个模型族上实现了5.6–14×的搜索成本降低（Gemini族更高可达25–54×），且在多数实验中优于或匹配目标层级的优化性能；在90%以上的实验中，低价搜索得到的提示在目标层级上至少与原版同等或更好。

**⚠️ 局限性**

局限性包括：①低价评估模型的基本能力必须足够，若其性能接近0则无法产生梯度导致搜索停滞；②在已达到提示不敏感上限的部署层级，搜索提升有限；③方法的成本优势依赖于当前的价格分层，如果价格差距缩小或消失，成本收益优势可能被削弱。

---

## 349. DualSpectralCF: Training-Free Sign-Aware Spectral Collaborative Filtering

**arXiv ID:** 2608.10247 | [PDF](https://arxiv.org/pdf/2608.10247v1)

**作者:** Guanqun Yang `[一作]` (Stevens Institute of Technology), Xiaoxue Han `[通讯]` (Stevens Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一个训练‑free 的双谱框架 DualSpectralCF，利用显式负反馈通过符号化输入信号和符号化项间运算符，在任何谱 CF 骨干（如 ChebyCF、GF‑CF、Turbo‑CF）上提升推荐精度。

**💡 创新点**

1) 首次在谱 CF 中引入仅两个超参数的符号化输入和符号化运算符；2) 框架与骨干无关，可直接迁移到多种谱模型；3) 在保持训练‑free 的同时，在五个负反馈基准上显著提升 Recall@20，尤其对冷启动用户效果突出。

**🔧 技术方法**

基于低通图滤波、Chebyshev/多项式滤波、符号化 Laplacian 或相似度矩阵的谱分析技术；同时使用标记化的正负交互信号。

**📊 数据集**

Amazon‑CDs、Amazon‑Music、Epinions、KuaiRand、KuaiRec 五个公开带显式负反馈的推荐数据集。

**📈 对比分析**

与无符号谱 CF（ChebyCF、GF‑CF、Turbo‑CF）、LightGCN 以及学习型符号化模型 SIGformer 对比；Recall@20 在 5 个数据集上均提升 0%–34%，在所有数据集上均不逊于无符号骨干，达到 SIGformer 70.7%–90.7% 的性能，并以 7.7–155.3 倍的速度优势。

**⚠️ 局限性**

需手动调节全局 γ 与 κ，单一 γ 对活跃用户可能产生负效应；对噪声或稀疏负反馈数据可能效果不佳；缺乏动态自适应机制，无法捕获更复杂的用户偏好结构。

---

## 350. Topological Feasibility Guarantees for Differentiable Predictive Control

**arXiv ID:** 2608.10332 | [PDF](https://arxiv.org/pdf/2608.10332v1)

**作者:** Guangyu Wu `[一作]` (Chalmers University of Technology), Ján Drgoňa `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于拓扑分析的可判定可行性理论，证明了差分预测控制（DPC）在仅使用有限离线样本时能够严格满足状态与输入约束，并给出了一个两阶段的自监督训练框架（温启动+CBF-代理微调）实现这一目标。

**💡 创新点**

创新点包括：① 用拓扑/几何方法严谨证明DPC的确定性可行性；② 引入离线的控制壁函数（CBF）代理损失，在训练时严格内部化约束；③ 通过两阶段同伦训练使软约束与硬约束兼顾，克服了传统阻尼方法在安全域外数值不稳定的问题；④ 通过有限样本覆盖证明可行性几何不再需要在线安全滤波。

**🔧 技术方法**

技术手段：可微预测控制框架、自动微分、软硬约束损失设计、控制壁函数（CBF）与对数障碍、两阶段同伦训练、Lipschitz 连续性与紧致集覆盖理论、Heine–Borel 定理用于构造有限覆盖。

**📊 数据集**

数据集：自生成的三类仿真环境—移动机器人非线性运动学、随机参数的非平稳线性系统、离散化的四旋翼线性化模型；每个任务分别使用 100、200、400、600、800、1000 等不同规模的样本进行训练与评估。

**📈 对比分析**

比较方法：与传统基于软惩罚的DPC（Vanilla DPC）对比；通过统计测试（1000 条测试轨迹）比较点状违规率和轨迹违规率。结果表明：随着训练样本增大，CBF-代理DPC 的违规率呈单调下降趋势，最终在 1000/100/50/16 等样本数处达到 0；相比之下 Vanilla DPC 在相同样本规模下违规率明显更高，表明安全性显著提升。

**⚠️ 局限性**

局限性：① 需要足够分布均匀的训练样本来覆盖可达安全集，缺乏明确的样本复杂度上界；② 目前的理论证明为存在性，未给出具体构造式的样本数上界；③ 对高度非线性/高维系统的可扩展性尚未验证；④ 依赖精确的系统模型，模型误差会削弱理论保证。

---

## 351. Closed-Loop LLM Co-Pilots for Digital Agriculture

**arXiv ID:** 2608.09949 | [PDF](https://arxiv.org/pdf/2608.09949v1)

**作者:** Serge Kernbach `[一作]` `[通讯]` (CYBRES GmbH), Serge Kernbach (CYBRES GmbH)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过搭建基于49通道植物传感网络的闭环数字农业系统，利用大型语言模型（LLM）对植物生理数据进行实时分析并驱动光照、灌溉等硬件执行器，实现了从数据解释到全自动控制的转变，显著提升了产量和能源效率。

**💡 创新点**

创新之处在于将LLM用作交互式共舱驾驶员，既能即时生成自然语言的诊断报告，又能在闭环系统中执行多目标优化（时间最短、能耗最小、超低能耗），并通过自主学习发现暗诱绿叶积累等新策略，突破传统基于规则的农业自动化。

**🔧 技术方法**

技术手段包括：两代理架构（Scout+Worker）进行多时标特征工程；Python脚本自动聚合高维时序数据；LLM（Gemini、Claude、ChatGPT等）进行自然语言推理与JSON控制序列生成；光谱、电化学、介电、温湿度等多模态传感器融合；模型预测控制、强化学习与启发式搜索结合的控制策略。

**📊 数据集**

使用的数据集为来自垂直农场微绿（小麦草、豌豆）和单株植物（Dracaena、番茄、辣椒）的实时多通道传感数据，共计49个传感通道，涵盖电化学、电阻、光谱、温湿度、土壤含水、叶通量等多种生理指标。

**📈 对比分析**

与传统周期性光照+持续红光基准对比，LLM最短时间模式将生长周期缩短35%（4.25天对6.5天），能耗上升48%；最小能耗模式能耗下降18%但生长周期延长至5天；“超低能耗”模式在基准上实现67.9%的能耗节约；在异常检测实验中，LLM能够准确定位多变量关联并给出专家与非专家均可理解的解释，验证了其诊断与控制双重能力。

**⚠️ 局限性**

局限性包括：LLM易出现幻觉，可能给出错误数值或不存在的物理关联；缺乏可验证的优化算法内部实现，导致可解释性不足；对复杂生理机制的解释仍需人工专家确认，无法完全自动化；系统在安全性和可靠性方面尚未实现全面保障，需人机协同进行最终验证。

---

## 352. Multitask Pareto Optimization for Monotone Submodular Problems with Dynamic Constraints

**arXiv ID:** 2608.10425 | [PDF](https://arxiv.org/pdf/2608.10425v1)

**作者:** Liam Wigney `[一作]` (Adelaide University), Frank Neumann `[通讯]` (Adelaide University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在动态背包约束下的单调子模优化问题中使用多任务GSEMO的理论与实验效果，证明了在共享子模目标函数的前提下可以获得（1‑1/e）近似，并给出了上界运行时间。

**💡 创新点**

创新点在于将多任务 Pareto 优化与动态约束结合，揭示了小 Pareto 前沿如何实现解的共享与快速适应，并给出了针对 k 个任务的统一上界分析，首次把子模目标函数纳入动态多任务分析。

**🔧 技术方法**

采用了全局简单多目标进化算法 GSEMO、Pareto 支配判定、运行时间分析（期望迭代次数）以及最大覆盖（Maximum Coverage）作为实验基准。

**📊 数据集**

实验使用了 9 个社交网络图（如 ca-GrQc、Erdos992、ca-HepPh 等）并在不同预算与动态更新策略下评估算法。

**📈 对比分析**

与传统单任务 GSEMO 逐任务求解方法比较，发现当各任务预算相近时多任务版本在大约 10 倍以上的计算量下显著优于单任务；当预算差异很大或预算非常小时时，单任务方法性能更好。

**⚠️ 局限性**

局限性包括仅考虑统一成本约束、仅针对单调子模目标、动态约束只覆盖背包类且未考虑更一般的权重/非子模情形，且理论上只给出上界而非精确复杂度，实际中对极小预算的适应性仍有限。

---

## 353. Diffract: Spectral View of LLM Domain Adaptation

**arXiv ID:** 2608.10850 | [PDF](https://arxiv.org/pdf/2608.10850v1)

**作者:** Nikita Borodin `[一作]` (Risk AI Research Lab), Dmitry Vinichenko `[通讯]` (Risk AI Research Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析并改进了持续预训练（CPT）过程，通过奇异值分解（SVD）对大模型权重矩阵进行谱分析，揭示了奇异值谱不变、奇异向量变化驱动适配以及注意力头异质性。

**💡 创新点**

提出了基于头重要性排序的头重置准则（利用参考文本域与目标域的差异），实现最多 4% 的数学域性能提升，并证明在 CPT 过程中可安全移除多达 60% 的注意力头或 50% 的低秩分量而不显著损失质量；同时发现域连接性（不同域检查点的线性插值保持性能），提供了一种新的模型平均思路。

**🔧 技术方法**

使用 SVD、奇异值谱统计（Frobenius、谱、有效秩）、奇异向量一致性度量、PL‑KS 距离、头重置与 SVD 截断、线性插值（模型汤）等技术，配合 Diffract 工具实现可重复的谱分析。

**📊 数据集**

在 OLMo 2 1B、7B、13B、32B 等模型上，以 DCLM、DolminoMath、FLAN、Stack Exchange、StarCoder 等数据集进行预训练与 CPT，分别针对数学、指令、文本、代码四个域进行实验。

**📈 对比分析**

在 OLMES 框架下对语言任务（ARC‑Easy、HellaSwag、WinoGrande）、数学任务（GSM8K、MATH‑500）、指令跟随（DROP、SQuAD）和代码生成（HumanEval）进行评估。结果显示：相较于全量 CPT，按重要性重置约 15% 的注意力头可提升 4%（7B 模型）；保留 70% 低秩分量（13B）或 50%（7B）后性能不下降；线性插值（域连接）在大预训练量和模型规模下表现优于简单混合训练。

**⚠️ 局限性**

实验仅涵盖 OLMo 2 系列模型，未验证其他 LLM 架构；使用的 AdamW 训练脚本可能限制可迁移性；基准覆盖有限（主要为语言、数学、指令、代码）；未深入探讨优化器、超参或更大规模（>32B）实验，且未完全验证对非文本域的迁移效果。

---

## 354. MazzikaAI: A knowledge-based performance-to-prompt compiler for real-time Arabic maqam accompaniment with a streaming text-to-music model

**arXiv ID:** 2608.10360 | [PDF](https://arxiv.org/pdf/2608.10360v1)

**作者:** Jiaxin Du `[一作]` (Grand Valley State University), Haoyu Li `[通讯]` (Grand Valley State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套基于知识的实时伴奏系统，通过将现场演奏状态编译成自然语言提示，驱动Google Lyria RealTime生成微调的、符合阿拉伯 maqam 语调与装饰的音频；

**💡 创新点**

创新点在于将自然语言作为通用控制表面，将模型不需要微调即可通过 deterministic prompt 编译器实现精准、可解释的实时音乐生成，并通过知识库嵌入专门的 maqam 语义与乐器分配规则；

**🔧 技术方法**

技术包括知识库（手工编写的 maqam 规则、乐器角色、生成参数表）、工作内存状态估计、四状态伴奏策略、文本提示编译器、WebSocket 双向实时通信、React 前端、FastAPI 后端、Lyria RealTime 生成模型以及 MediaPipe、Web MIDI、Web Speech、Gemini OMR 等多模态输入；

**📊 数据集**

使用的数据主要为现场 MIDI、手势、语音指令以及可选的乐谱 OCR，未使用传统大型音频或 MIDI 数据集进行训练，而是依赖专家手工规则与实时事件；

**📈 对比分析**

通过系统级度量（延迟、吞吐、稳定性）、对照实验（提示编译、maqam grounding、仪器抑制、门控开启/关闭/静态）以及两名阿拉伯 maqam 专家的现场体验评估，对比结果显示：关键到可听延迟中位数为263 ms，峰值<1 s，回调率≈179次/分钟，系统保持无崩溃；Ablation表明提示编译可显著提升四分之一音符出现率，门控能减少API推送；

**⚠️ 局限性**

局限包括缺乏节拍跟踪与时序同步，不能实现stem‑level混音导致的乐器抑制失效，依赖闭源 Lyria RealTime 造成的延迟与功能受限，知识库覆盖范围有限（仅支持单键盘单演奏者、英语命令），以及未完成的多模态与多人场景支持。

---

## 355. The Kuramoto Neural Operator: Learning to Solve PDEs via Coupled Oscillator Dynamics

**arXiv ID:** 2608.10234 | [PDF](https://arxiv.org/pdf/2608.10234v1)

**作者:** Petr Badolia `[一作]` (Basic Research of Artificial Intelligence Laboratory), Aleksandr Beznosikov `[通讯]` (Basic Research of Artificial Intelligence Laboratory)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Kuramoto Neural Operator (KNO)，一种通过在物理域上演化耦合振荡器来学习 PDE 求解算子的神经网络架构；

**💡 创新点**

创新点在于将 PDE 的解算子建模为隐式振荡器动力学的演化，利用 Kuramoto 互动与球面投影实现自适应的空间耦合，并通过持续刺激 (stimulus) 与局部消息网络实现特征条件化；

**🔧 技术方法**

使用的技术包括：球面上投影的离散 Kuramoto 迭代、局部卷积消息网络、指数滑动平均反馈、谱重采样到固定的 32×32 基准网格、以及残差解码回到特征空间；

**📊 数据集**

在 Representative PDE Benchmark 上评估，包括 Poisson、Wave、Allen–Cahn、Continuous/Discontinuous Translation、Darcy、Navier–Stokes、Airfoil 等 8 个二维 PDE 任务；

**📈 对比分析**

与 FNO、RNO、ConvFNO、CNO、DeepONet、UNet、AKOrN 等 9 个基线相比，KNO 在大多数任务上排名第一或第二，尤其在波动、传输和异构弹性流场任务上显著优于传统卷积/频域方法；

**⚠️ 局限性**

局限性：对不规则几何或极其线性、光滑的算子表现不如卷积/频域基线；依赖固定网格和谱重采样，难以直接处理不规则网格或高维问题；对超参数（振荡器数、维度、层深）敏感，需要进一步探索。

---

## 356. FlowGRN+: Improving Gene Regulatory Network Inference by Spline Fitting and Manifold Projection in Conditional Flow Matching (Technical Report)

**arXiv ID:** 2608.10407 | [PDF](https://arxiv.org/pdf/2608.10407v1)

**作者:** Tsz Pan Tong `[一作]` (University of Luxembourg), Jun Pang `[通讯]` (University of Luxembourg)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于条件流匹配的基因调控网络推断框架 FlowGRN+，能够从单细胞 RNA‑seq 快照数据中重构细胞轨迹并推断 GRN。

**💡 创新点**

创新点：① 用平滑样条拟合多时间点的 OT 链，生成连续且对 dropout 友好的参考轨迹；② 引入流形投影方案将参考速度投影到局部切空间，缓解样条过冲并提升轨迹与数据流形的对齐；③ 通过多边界流匹配提升时间一致性并自动化细胞分箱，减少人工聚类。

**🔧 技术方法**

使用技术：条件流匹配（CFM）、多边界 OT 链、平滑样条（B‑spline + 正则化）、kNN 局部协方差投影、Woodbury 近似、ResNet 20‑层、DOPRI5 求解器、dynGENIE3 进行 GRN 推断。

**📊 数据集**

数据集：BEELINE benchmark 中 7 个实验集（hESC、hHep、mDC、mESC、mHSC‑E、mHSC‑GM、mHSC‑L），每组取 500 或 1000 个高变基因（含 TF）。

**📈 对比分析**

与 FlowGRN 及 10+ 传统基线（LEAP、SCODE、GRISLI、GRNVBEM、SINCERITIES、Scribe、GENIE3、GRNBOOST2 等）在 AUPRC 与 EPR 两指标上对比。FlowGRN+ 在 14 个设置中 10 个 EPR 进入前两名，6 个 AUPRC 进入前两名；相较原 FlowGRN，EPR 上有显著提升，AUPRC 在部分数据集略有下降。轨迹平滑度（TV）显著下降。

**⚠️ 局限性**

局限性：① 流形投影假设数据呈光滑单连通流形，对具有分支或非光滑结构的数据可能不适用；② 投影计算及多边界样条拟合增加训练时间和算力负担；③ 对极端 dropout 或分支动态的处理仍有限，未来需要更紧耦合投影与 CFM 或加入先验通路约束。

---

## 357. World-First SEM-based Recovery of Crash EDR Data from the EEPROM of a Severely Damaged SRS Module Using CrashScan

**arXiv ID:** 2608.10152 | [PDF](https://arxiv.org/pdf/2608.10152v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 358. Self-evolving Agentic Customer Support System at LinkedIn

**arXiv ID:** 2608.10224 | [PDF](https://arxiv.org/pdf/2608.10224v1)

**作者:** Chih Hui Wang `[一作]`, Changshuai Wei `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个支持企业客服的自演化AI系统，闭环集成提示进化、检索增强生成和多维评估，实现持续改进。

**💡 创新点**

创新点在于将提示、检索和评估三个模块实现版本化闭环演化，采用遗传算法自动优化提示、将RAG作为动作而非预处理，并设计可解释的多维评估框架。

**🔧 技术方法**

采用遗传算法提示进化、检索增强生成（RAG）工具调用、LLM-as-judge评估、多语言翻译评估、版本化知识库、在线A/B实验等技术。

**📊 数据集**

使用企业内部多语言客服对话日志（约1万条）及合成边缘案例；评估集包括100条支持对话、30条意图检测数据、300条中英翻译数据。

**📈 对比分析**

通过离线模拟对比Vanilla RAG、ReAct Agent、OpenAI Tool Agent等配置，整体分数提升至2.78，幻觉率降至<0.1%，完整度达87.8%；在线A/B实验中QA自助提升9.0pp、取消自助4.8pp、路由准确率30.6pp。

**⚠️ 局限性**

主要局限包括评估依赖GPT‑4.1判定器、进化成本与延迟、单租户实验、组件因果分离困难、检索覆盖上限、低资源语言测试不足以及对闭源模型的依赖。

---

## 359. Carefully Considering Culture: Analyzing LLM Alignment in Single- and Multi-Cultural Settings using Cultural Consensus Theory

**arXiv ID:** 2608.09937 | [PDF](https://arxiv.org/pdf/2608.09937v1)

**作者:** Krishna Pothugunta `[一作]` (University of Notre Dame), John P. Lalor `[通讯]` (University of Notre Dame)

**通讯引用:** 492 | [OpenAlex ID](https://openalex.org/A5033125725)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

利用文化共识理论（CCT）对大规模语言模型（LLM）在十个国家、十二个文化领域中的回答进行评估，比较其与人类受访者的共识结构，并通过共识一致性（CC）和方差差异（Δ_VE）来诊断模型的文化适配性。

**💡 创新点**

首次将文化共识理论应用于 NLP 领域，提供了可量化 intra‑group 方差、Consensus Gap、Heterogeneity Gap 与 Consensus Inflation 的细粒度诊断框架；同时提出通过 CCT 的方差解释与共识一致性来衡量 LLM 与人类文化共识的匹配程度。

**🔧 技术方法**

核心技术：文化共识理论（CCT）的特征值分解、共识向量估计、文化能力评分；评估指标包括共识一致性（CC）与方差差异（Δ_VE）；实验实现使用 AnthroTools、Open WebUI 及 GPT‑4o API 进行模型调用。

**📊 数据集**

数据集：World Values Survey（WVS）第七波的人类调查数据，以及通过 10 个 LLM（GPT‑OSS、Llama3、Qwen、Phi3、GPT‑4o 等）按国家定制提示生成的 60 行回答（10 模型 × 6 提示）。

**📈 对比分析**

方法：构造人类与 LLM 的响应矩阵，分别拟合 CCT 模型，计算各自的文化能力评分、共识向量；随后通过 CC 衡量两者共识答案的一致性，Δ_VE 衡量 LLM 与人类内部共识的差异。实验显示：在 POC 与 RV 领域出现 Consensus Inflation，HWB 领域出现 Consensus Gap，POST 与 EV 等领域出现 Heterogeneity Gap；多文化与单文化情境下 Δ_VE 变化显著，说明文化聚合对模型共识结构有显著影响。

**⚠️ 局限性**

局限性：CCT 对极端一致或极度分散的回答无法提供有意义的共识度量；需要完整的个体级响应数据，无法直接应用于国家级指标；共识一致性采用启发式加权与四舍五入，缺乏正式的序数模型；模型仅给出统计描述，缺乏因果解释；在子文化层面、不同样本规模以及提示敏感性方面仍需进一步研究。

---

## 360. ProtoGIB-Workload: Learning Workload-Specific Neural Topology Prototypes across Subjects

**arXiv ID:** 2608.10647 | [PDF](https://arxiv.org/pdf/2608.10647v1)

**作者:** Yuzhe Zhang `[一作]` (Nanjing University of Aeronautics and Astronautics), Chao Shen `[通讯]` (Xi'an Jiaotong University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种名为ProtoGIB-Workload的图瓶颈框架，用于实现对未见用户的EEG工作负荷识别。

**💡 创新点**

创新之处在于同时引入样本级的Stochastic Graph Information Bottleneck（SGIB）压缩冗余连接，并在类条件下通过Class-Conditional Topology Stabilizer（CTS）将不同主体的拓扑收敛到统一原型，从而显著抑制主体特异性结构捷径。

**🔧 技术方法**

主要技术包括变分子图生成器与Gumbel-Sigmoid离散化、图卷积网络、信息瓶颈正则化（I(A;X,R)、I(A;D|Y)）、类条件平均场目标以及类内收敛与类间分离损失。

**📊 数据集**

实验使用了两个公开EEG工作负荷数据集（STEW、EEGMAT）以及自采集的空中交通控制员数据集SELF（59通道、5级负荷）。

**📈 对比分析**

在严格的留一主体交叉验证下，与12种基线（EEG通用、工作负荷专用和图基模型）对比，ProtoGIB在Macro-F1上平均提升约5.15%，在STEW和EEGMAT分别提升6.19%和6.34%，显示出优异的跨主体泛化性能。

**⚠️ 局限性**

局限性包括受试者样本有限（尤其是SELF仅8人）导致高阶拓扑信息仍未完全消除；在细粒度多类负荷或极低负荷场景下性能仍有提升空间；以及对不同通道数和采样率的鲁棒性需进一步验证。

---

## 361. Fixed-Threshold Peeling in Sublinear MPC: Round-Approximation Tradeoffs and Applications

**arXiv ID:** 2608.10135 | [PDF](https://arxiv.org/pdf/2608.10135v1)

**作者:** Slobodan Mitrović `[一作]` (University Of California Davis), Wen-Horng Sheu `[通讯]` (University Of California Davis)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了基于迭代剥离的图问题，包括密度依赖的边定向、密度依赖的着色、最稠密子图和k-核心分解，提出了在子线性MPC模型下的算法。

**💡 创新点**

创新点在于为这些问题提供了更快的子线性MPC算法，突破了之前的Θ(√(n))轮复杂度限制，尤其是在最稠密子图和k-核心分解问题上。

**🔧 技术方法**

使用了改进的指数化和修剪算法，结合了图的稀疏化和剥离技术，以实现更好的近似因子和轮次复杂度的权衡。

**📊 数据集**

使用了随机采样的图数据集，以确保在子线性内存模型下能够有效地处理图问题。

**📈 对比分析**

与之前的算法相比，本文的算法在轮次复杂度上有显著改进，尤其是在最稠密子图问题上，达到了O(log^1/3 n)和O(log^1/4 n)的近似因子。

**⚠️ 局限性**

限制在于算法的性能依赖于局部内存的大小，且在处理某些特定图结构时可能会遇到困难，尤其是在处理大量非活动顶点时。

---

## 362. Never Stop Speaking: a Denial-of-Service Attack on End-to-End Speech Language Models

**arXiv ID:** 2608.10405 | [PDF](https://arxiv.org/pdf/2608.10405v1)

**作者:** Shuozhe Cheng `[一作]`, Wenbo Jiang `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对端到端语音LLM提出基于声学扰动的拒绝服务攻击，诱导模型生成过长无意义输出

**💡 创新点**

创新性地设计了多目标损失函数，结合VAD实现对EOS抑制、长度扩展、语义一致性的协同优化

**🔧 技术方法**

采用PGD梯度投影法、VAD声区检测、加权EOS、top‑k、长度及语义相似度损失等技术

**📊 数据集**

使用OpenSLR和QCRI两大公开语音数据集，对三款开源E2E ALLM（LFM2.5‑Audio、FunAudioChat、Qwen2‑Audio）进行评估

**📈 对比分析**

相较于基线方法，攻击成功率达到87%/84%，输出长度提升约4倍，显著增加GPU内存占用，验证了攻击效果

**⚠️ 局限性**

对压缩或其他信号变换敏感，且在不使用VAD时语义一致性下降

---

## 363. Seeds Before Objectives: Rethinking Evaluation for Low-Resource Garhwali ASR

**arXiv ID:** 2608.10670 | [PDF](https://arxiv.org/pdf/2608.10670v1)

**作者:** Karamvir Singh Batra `[一作]` (Thapar Institute of Engineering and Technology), Sahil Sharma `[通讯]` (Ulster University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Garhwali低资源语音识别任务上，构建了以官方VAANI分割为基础、可复现的多种子基准，系统性评估了多种干预手段。

**💡 创新点**

创新点在于：①提出并验证多种子配对统计方法，证明单跑结果易产生误导；②对Focal CTC、Matra-weighted CTC和Hindi→Garhwali双阶段转移进行严谨比较，发现它们在此任务中无可靠提升；③确认预训练设计和速度增强才是提升性能的关键。

**🔧 技术方法**

使用技术包括：w2v-BERT 2.0 580M预训练编码器、CTC损失（标准、Focal、Matra-weighted）、速度扰动（0.9/1.0/1.1×），双阶段Hindi转移，配对Wilcoxon检验并做Holm校正。

**📊 数据集**

数据集：VAANI Garhwali子集，8.8 h训练语音，官方4,778/666/450（train/val/test）划分。

**📈 对比分析**

比较方法：在五个随机种子上计算WER/CER，采用配对Wilcoxon+Holm校正进行显著性检验；结果显示标准CTC平均WER为47.0%；速度增强平均降低1.1–1.5点；Focal CTC、Matra-weighted CTC及Hindi转移均无显著优势。

**⚠️ 局限性**

局限性：仅针对单一方言和单一数据集，种子数有限（5个），统计功效不足；未探索多源转移、参数高效适配、LM/beam搜索等改进；结果受训练样本量和评测协议约束，无法直接推广到其他低资源方言。

---

## 364. Most biomedical publications show signs of LLM-assisted writing

**arXiv ID:** 2608.10715 | [PDF](https://arxiv.org/pdf/2608.10715v1)

**作者:** Lena Holzwarth `[一作]` (University of Tübingen), Dmitry Kobak `[通讯]` (University of Tübingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

利用大型语言模型（LLM）标记词频的变化，构建了一种无偏估计方法，推算2023‑2025年PubMed Central（PMC）生物医学论文中LLM辅助写作的比例；

**💡 创新点**

该方法突破了传统仅给出下界的估计，提出在不同阈值下通过优化标记词集来直接估算LLM使用率，并在模拟实验中验证其高精度；

**🔧 技术方法**

主要技术包括：标记词频统计、线性回归外推人类写作词频、计算LLM使用率下界、阈值优化与二项分布误差传播，辅以随机裁剪和模拟验证；

**📊 数据集**

使用了2026年初公开的PMC开放获取子集，包含1194万篇英文论文（2017‑2025年），并对各章节及各国作者进行细分分析；

**📈 对比分析**

与现有频率差、混合模型、分布重叠等方法相比，该估计在模拟实验中误差<0.02，实际数据显示2025年整篇论文LLM使用率达89%，高于文献报告的31‑52%范围；

**⚠️ 局限性**

主要局限在于假设2018‑2022年词频趋势可外推至2025年，且人类写作风格可能随LLM普及而改变，阈值选择及低频词影响也可能导致低估。

---

## 365. Generating Attacks for LLMs with GFlowNets

**arXiv ID:** 2608.10171 | [PDF](https://arxiv.org/pdf/2608.10171v1)

**作者:** Berkay Ozcam `[一作]` (Turkcell), Emin Islam Tatli `[通讯]` (Turkcell)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用GFlowNets让一个大型语言模型自动对另一个模型进行红队攻击，并生成多样化的攻击输入

**💡 创新点**

首次将GFlowNets应用于红队，兼容英语和土耳其语攻击输入的生成，且通过奖励机制提升多样性

**🔧 技术方法**

监督微调（SFT）、生成流网络（GFlowNets）、最大似然估计（MLE）以及安全评估器（Qwen3Guard、LlamaGuard）

**📊 数据集**

基于Lee等人公开的混合攻击数据集，扩展后并翻译为土耳其语共3100条样本

**📈 对比分析**

通过在不同模型配置（攻击者、受害者、评估器）下进行12次实验，显示土耳其语攻击成功率最高达79%，英语亦显著提升，且GFlowNets+MLE显著提高毒性与成功率

**⚠️ 局限性**

受限于GPU资源使用低参数模型，评估器的准确性直接影响整体成功率，土耳其语攻击多样性不足导致迁移性能下降

---

## 366. ASCon: A Direction-Aware Reciprocal Agent--Step Contextualization Model for Failure Attribution in Multi-Agent Systems

**arXiv ID:** 2608.10646 | [PDF](https://arxiv.org/pdf/2608.10646v1)

**作者:** Shuyu Jiang `[一作]` (Sichuan University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为ASCon的方向感知互补代理–步骤上下文化模型，用于LLM驱动的多智能体系统（MAS）故障归因。

**💡 创新点**

创新点在于：①统一表示框架，将步骤、代理及故障模式三种归因目标统一建模；②引入方向感知图注意力（DGAT）区分前后依赖；③采用掩码步-代理注意力实现对代理行为的选择性聚合；④将代理上下文回注到步骤表示，形成互补上下文化。

**🔧 技术方法**

主要技术：方向感知图注意力网络（DGAT）、掩码步-代理注意力、图神经网络、LLM文本嵌入、轻量级预测头。

**📊 数据集**

使用的公开数据集：TracerTraj（代码、数学、通用代理任务）和Aegis-Bench（含14种故障模式的MAS轨迹）。

**📈 对比分析**

与多种提示式和学习式基线（如All-at-Once、StepFinder、AgentTracer、Aegis-SFT等）进行对比。实验显示，在根因归因上Agent微准确率提升约3.4%，Step微准确率提升约7.6%；在故障模式归因上Pair μF1提升约13.9%，Pair MF1提升约11.1%。此外，将ASCon与LLM基方法结合，在无额外训练的离域场景中也能显著提升性能。

**⚠️ 局限性**

局限性：①对图结构仍依赖手工构建，LLM推断的图未能带来优势；②在长序列或多根因场景下的可解释性和性能尚待进一步验证；③模型在极端稀有故障模式上的泛化仍有限。

---

## 367. Post-Hoc Sparse Coding of Latent Communication Between Vision-Language Model Agents

**arXiv ID:** 2608.10198 | [PDF](https://arxiv.org/pdf/2608.10198v1)

**作者:** Di Wu `[一作]` (Xi'an Jiaotong-Liverpool University), Xiaohui Zhu `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析了 Vision Wormhole 视觉语言模型间固定形状隐空间通信通道的冗余，利用后置稀疏自编码器压缩传输并评估重构与任务性能。

**💡 创新点**

证明该通道即使压缩到 128× 体积，重构误差极低且单次评估任务准确率几乎不变，显示其存在可观的可压缩结构。

**🔧 技术方法**

采用后置稀疏自编码器（SAE）对冻结激活进行压缩，使用 4096 大小字典、k=4 等超参，并计算重构误差、余弦相似度、MSE、带宽、Jaccard 重叠等指标。

**📊 数据集**

在九个推理基准上评估：GSM8K、ARC‑Easy、ARC‑Challenge、GPQA、MedQA、MBPP+、HumanEval+、AIME 2024 与 AIME 2025。

**📈 对比分析**

与原始 2052 KB float32 传输相比，k=4 的稀疏编码将带宽降至 16 KB，压缩比 128×，单次评估宏平均准确率仅下降 0.08%。

**⚠️ 局限性**

局限在于仅为单次评估、未与低秩、量化、向量量化等基线对比、未验证跨模型/对称传输的可迁移性，以及 98.78% 字典未被激活，难以确认稀疏性究竟是主要压缩因子。

---

## 368. GeoSeg-OV: Bridging Geospatial Gaps with Structural Guidance for Open-Vocabulary Remote Sensing Segmentation

**arXiv ID:** 2608.10426 | [PDF](https://arxiv.org/pdf/2608.10426v1)

**作者:** Ruizhong Liu `[一作]` (China University of Geosciences), Hongyan Zhang `[通讯]` (China University of Geosciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GeoSeg-OV 框架，用辅助视觉基础模型的结构信息引导开词汇遥感分割。

**💡 创新点**

创新点在于将辅助 VFM 从视觉‑文本匹配中解耦，仅用其结构偏置指导成本聚合和解码，结合结构引导聚合（SGA）与成本感知解码（CAD）实现跨域通用。

**🔧 技术方法**

核心技术包括多旋转 CLIP 成本体积构建、结构引导聚合（SGA）、成本感知解码（CAD）、冻结辅助 VFM（如 DINOv2、Depth Anything 等）等。

**📊 数据集**

使用七个高分辨率陆表覆盖数据集（FLAIR、OpenEarthMap、LoveDA、EarthMiss、DeepGlobe、Potsdam、Vaihingen）构建全球 HRLC 基准。

**📈 对比分析**

与训练自由、传统与遥感专用的多种开词汇分割方法对比，GeoSeg-OV 在所有评估集上平均 mIoU 提升约 2.5–3.0，领先现有 SOTA。

**⚠️ 局限性**

局限在于需额外冻结 VFM 前向计算导致推理成本上升，且对极高分辨率图像仍存在分辨率与边界细化挑战。

---

## 369. UserToolBench: A User-Profile-Hidden Benchmark for Personalized Decision Making in Tool-Use LLMs

**arXiv ID:** 2608.10042 | [PDF](https://arxiv.org/pdf/2608.10042v1)

**作者:** Xuexiong Yin `[一作]` (Sun Yat Sen University), Keze Wang `[通讯]` (Sun Yat Sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 UserToolBench 这一基准，用以评估工具使用型大语言模型在隐藏用户资料、信息不完整且跨多轮交互中的个性化决策能力。

**💡 创新点**

创新点在于：①构建了“profile‑hidden”评估协议，让模型必须从历史交互中推断隐式偏好；②将工具调用轨迹与多轮、跨主题的长期决策结合；③通过真实交互轨迹的隐私去标识化生成多样化用户画像与工具生态。

**🔧 技术方法**

主要技术包括：LLM 辅助的用户画像抽象与对话生成、基于工具 schema 的规划与验证、严格与宽松两种轨迹匹配评估指标，以及多标签与严重性诊断的错误分析。

**📊 数据集**

使用了 10 个隐私去标识化的用户画像、170 个公开 API 风格工具、1,065 轮对话，涵盖 799 个评估任务（缺失信息、单工具、多工具）。

**📈 对比分析**

实验对比了 9 款工具使用型 LLM（包括 GPT‑5.4、Qwen‑3.6、DeepSeek‑V4 Pro 等），Exact 轨迹准确率最高仅 49.36%，Relaxed 任务完成率约 70%，表明当前模型在个性化委派与多工具协同方面仍显薄弱。

**⚠️ 局限性**

限制包括：仅 10 个用户画像、生成的对话可能缺乏真实对话中的修正与拖延行为、工具生态覆盖面有限、评估基于单一参考轨迹，未充分考虑多条合法决策路径。

---

## 370. From Faulty Memories to Corrected Actions: Dependency-Guided Rollback Repair for Memory-Augmented Agents

**arXiv ID:** 2608.10502 | [PDF](https://arxiv.org/pdf/2608.10502v1)

**作者:** Caili Yu `[一作]`, Taotao Cai `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对语言模型代理的持久内存错误，提出依赖引导的回滚修复方法，实现对已传播错误的回答和状态恢复。

**💡 创新点**

将执行追踪与内存生命周期结合成带类型的依赖图，采用独立支持检查与规则驱动的回滚规划，只重放答案相关的受影响步骤，兼顾恢复与成本。

**🔧 技术方法**

使用运行时证明（provenance）+ 图构建与数据流追踪 + 规则化回滚规划 + 选择性重放 + 大模型（GPT‑4o）调用等技术。

**📊 数据集**

使用150个购物/旅行/客服工具场景的控制型基准和改编自 LongMemEval‑V2 的 50 条多错误轨迹。

**📈 对比分析**

与六个基线（全内存重置、删除检索、MemAudit、LLM‑judge、AgentTrace、无修复）对比，控制基准上恢复率 85.3% 最高，成本（重放比例 12.3%）最低；转移子集上恢复率 68% 也领先。

**⚠️ 局限性**

复现率在外部多错误轨迹下降，回滚方案对复发的抑制有限，且依赖完整的运行时证明，未覆盖缺失/错误证明的情况。

---

## 371. Ollivier's Ricci Curvature on Complex-weighted Graphs

**arXiv ID:** 2608.10132 | [PDF](https://arxiv.org/pdf/2608.10132v1)

**作者:** Yu Tian `[一作]` (Center for Systems Biology Dresden), Melanie Weber `[通讯]` (Harvard University)

**通讯引用:** 897 | [OpenAlex ID](https://openalex.org/A5034942394)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

提出了Ollivier-Ricci曲率在复加权图上的定义，并将其用于社区检测。

**💡 创新点**

首次给出兼容方向性和复权重的离散曲率，利用多层参数化与磁拉普拉斯关联，并推导出组合上界下界。

**🔧 技术方法**

采用随机游走、最短步距离、最优运输、组合近似与多层映射等技术。

**📊 数据集**

在合成的有向SBM、C.elegans神经网络以及EU email网络上进行实验。

**📈 对比分析**

与无向与有向Ricci曲率对比，利用NMI评价社区检测性能，发现复曲率在存在大量有向三角时表现最佳。

**⚠️ 局限性**

受限于高计算复杂度、对磁拉普拉斯构造的依赖以及仅验证了部分曲率扩展。

---

## 372. IADD-TR: Intervention-Aware Dynamics Decoupling with Targeted Regularization for Model-Based Reinforcement Learning

**arXiv ID:** 2608.10634 | [PDF](https://arxiv.org/pdf/2608.10634v1)

**作者:** Zefeng Liang `[一作]` (Guangdong University of Technology), Zhifeng Hao `[通讯]` (Shantou University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种结合干预感知动态解耦与目标正则化的模型基础强化学习框架IADD-TR，以降低动态学习和策略学习中的偏差。

**💡 创新点**

创新点在于将转移过程分解为动作干预阶段与自然演化阶段，并通过零动作锚定可观测中间状态；同时引入基于高效影响函数的目标正则化，使价值函数更鲁棒于策略梯度估计。

**🔧 技术方法**

采用因果推断的干预模型、零动作可识别性分析、目标正则化（Doubly Robust）技术、软演员-评论家（SAC）以及MBPO模型回放等技术。

**📊 数据集**

在MuJoCo连续控制任务（HalfCheetah、Hopper、Walker2d、Ant、Humanoid）以及控制合成动力学环境上进行实验。

**📈 对比分析**

与SAC、PPO、SLBO、MBPO等基线进行对比，IADD-TR在所有任务上实现了更快的样本效率和更高或竞争的最终回报。

**⚠️ 局限性**

方法依赖于零动作锚定的可识别性假设以及对可观测中间状态的准确性，在非零动作占比过高或锚定不充分时效果可能下降。

---

## 373. ImpactHO: Importance-Aware KV Cache Transfer for Multi-User Edge LLM Handover

**arXiv ID:** 2608.10545 | [PDF](https://arxiv.org/pdf/2608.10545v1)

**作者:** Minwoo Kim `[一作]` (Pohang University of Science and Technology), Yongjune Kim `[通讯]` (Pohang University of Science and Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ImpactHO框架，通过对每个用户KV缓存进行重要性排序并按重要性顺序传输，解决多用户边缘LLM切换时的缓存迁移与回程带宽分配问题；

**💡 创新点**

1) 将KV缓存重要性排序融入网络传输；2) 用经验sigmoid模型刻画部分缓存对推理准确度的影响；3) 推导基于加权水填模型的实时分配算法，显著提升多用户平均准确度；

**🔧 技术方法**

重要性打分（Fast KVzip）、KV缓存按重要性排序、基于sigmoid准确度曲线的加权水填资源分配算法、离散时隙调度与Fallback策略；

**📊 数据集**

RULER长上下文评测基准；使用Qwen3-8B、Qwen3-14B和Llama-3.1-8B-Instruct三大模型进行实验；

**📈 对比分析**

与等分分配、赢家全占、比例公平、目标端重新预填充和混合传输等基线对比；实验显示在500 ms转移窗口内可达93.7%平均准确度，接近全缓存上限的0.5pp，且在多种带宽、时隙、用户负载情形下均优于基线；

**⚠️ 局限性**

仅在可达性区间内使用加权水填，无法覆盖不可达情况；缺乏全局时隙/未来用户预测的全局最优调度；实验仅覆盖特定模型与场景，实际部署时可能面临更大缓存/更复杂网络延迟；

---

## 374. Optimizing Parameterized Physics-Informed Neural Networks to Solve Multilayered Static Linear Elastic PDEs

**arXiv ID:** 2608.09981 | [PDF](https://arxiv.org/pdf/2608.09981v1)

**作者:** Joseph Lim `[一作]` (Bellarmine College Preparatory), Zhen Zhang `[通讯]` (Brown University)

**通讯引用:** 81994 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对多层弹性板的静态线性弹性问题，提出一种参数化物理信息神经网络（P2INN）框架，实现对不同材料刚度和层厚的快速、可控形变预测。

**💡 创新点**

创新点在于：① 层级分解的PDE残差与界面连续性正则化，② 合规性感知的尺度与刚度归一化，③ 仅在参数空间极端点使用稀疏FEM监督，④ 通过层级、物理先验的损失权重设计实现高精度和高泛化性。

**🔧 技术方法**

使用的技术包括：多层感知机（4层64隐藏神经元），Soft-Activation层（β=20）实现材料界面识别，SOAP+L-BFGS双阶段优化，基于Navier-Cauchy方程的PDE残差损失，边界和载荷软约束，能量一致性约束，FEM参考评价体系。

**📊 数据集**

数据集来自自行构建的FEM模拟，采用16×16×8 Hex8网格，对一层模型随机采样100组参数（E∈[1,10], t∈[0.05,0.15]），对三层模型同样随机采样100组参数（E_i∈[1,10], t_i∈[0.02,0.10]），并在六维参数空间的极端角点共64组进行轻量FEM监督（共36k节点标注）。

**📈 对比分析**

通过对比P2INN与FEM的体积平均绝对误差（VolMAE）评估。结果显示：一层模型均值1.56%/最坏2.87%，三层模型均值2.53%/最坏4.66%；速度提升分别为188×（1层）和149×（3层），在10^6个参数评估中，P2INN总耗时仅0.8/0.9小时，而FEM则需130/120小时。

**⚠️ 局限性**

局限性包括：仅适用于静态线性弹性问题，无法处理大变形、非线性材料、瞬态波传播或接触；三层模型仍需稀疏FEM监督以保持精度；未提供不确定性量化；在极薄层或高刚度组合下误差相对增大。

---

## 375. Neural Introspection Gating for Adaptive KV-Cache Reuse in Vision-Language-Action Models

**arXiv ID:** 2608.10824 | [PDF](https://arxiv.org/pdf/2608.10824v1)

**作者:** Zhijie Wu `[一作]` (University of Tokyo), Kei Okada `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Vision‑Language‑Action（VLA）模型的 KV‑Cache 进行动态失效治理，通过监测上一步的 logit margin 来决定是否重新计算视觉特征。

**💡 创新点**

创新点在于利用无训练、无额外参数的模型内部置信度（logit margin）作为门控信号，避免盲目缓存导致的误差累积。

**🔧 技术方法**

采用三阶段 VLA‑Cache（静态补丁检测、任务相关过滤、熵自适应层级重用）并加上基于 logit margin 的失效门控，集成在 OpenVLA 和 OpenVLA‑OFT 的推理管线中。

**📊 数据集**

使用 LIBERO benchmark 套件（Spatial、Object、Goal、Long）共 4 套，每套 500 条 episode 进行评估。

**📈 对比分析**

与完整推理和原始 VLA‑Cache 进行对比；在 OpenVLA 上恢复 100%+ 的准确率损失，保持约 80% 的计算节省；在 OpenVLA‑OFT 上门控几乎不触发，性能保持不变。

**⚠️ 局限性**

局限性包括：门控基于上一步 margin，存在 1 步反应延迟；阈值需要针对每个模型手动调优；无法区分缓存误差与模型固有不确定性；仅在仿真环境验证，真实机器人效果待验证。

---

## 376. Visual Geometry Foundation-Aware Gaussians for Single-Frame Surround-View Driving Reconstruction

**arXiv ID:** 2608.10682 | [PDF](https://arxiv.org/pdf/2608.10682v1)

**作者:** Junhong Lin `[一作]` (Peking University), Wei Gao `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于预训练几何基础模型的单帧环视驾驶重建框架 VGGD，能够在极低重叠的相机输入下实现高质量的3D重建与视角合成。

**💡 创新点**

核心创新在于将几何先验迁移到前端并通过 Scale Warmup 和 Dual‑Path Neck 进行驾驶场景适配，显著提升几何一致性与外观细节；同时利用混合像素‑体积 3D Gaussian Splatting 解码器实现高效渲染。

**🔧 技术方法**

技术要点包括：预训练 VGGT 作为几何前端；双路颈部（Geometry‑Consistent 与 Appearance‑Aware）解耦特征；Scale Warmup 预热尺度；Hybrid pixel‑volume 3DGS 解码器；多任务损失（RGB、LPIPS、深度、PCC）。

**📊 数据集**

使用 nuScenes 单帧环视重建基准（700 训练 + 150 验证场景），包含 6 视角相机，提供 12 个目标视角。

**📈 对比分析**

与多种基线（Per‑Scene 优化方法、通用 Feed‑Forward 方法、驾驶专用方法、以及大规模模型）对比，VGGD 在 PSNR、SSIM、LPIPS、PCC 上均取得最高或接近最高分，尤其在几何一致性（PCC 0.808）和渲染质量（PSNR 24.85）方面领先。

**⚠️ 局限性**

局限性在于仅处理静态场景，未建模动态对象或场景运动；对极端动态环境的鲁棒性尚待提升。

---

## 377. Motion Artifact-Aware Self-Supervised Representation Learning for 3D Brain MRI Motion Artifact Reduction

**arXiv ID:** 2608.10170 | [PDF](https://arxiv.org/pdf/2608.10170v1)

**作者:** Mojtaba Safari `[一作]` (University of Chicago), Xiaofeng Yang `[通讯]` (University of Chicago)

**通讯引用:** 13488 | [OpenAlex ID](https://openalex.org/A5100619090)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一套无配对监督的自监督表示学习框架 SSRL-MAR，用于3D脑MRI运动伪影的去除。

**💡 创新点**

创新点：① 采用三阶段闭环设计——对比学习提取运动表示、基于运动表示的伪影合成网络、以及运动感知恢复网络，实现无配对训练；② 设计了带维度重加权的异构 InfoNCE 对比损失，使嵌入空间仅关注运动而忽略解剖细节；③ 引入 MA‑Conv 与噪声注入模块，使合成伪影更逼真且可与真实伪影对齐；④ 在无原始运动清晰图像的前提下，完成自监督修复与无监督域自适应。

**🔧 技术方法**

使用的技术包括 3D ResNet + MLP 的对比学习、异构 InfoNCE 损失、U‑Net/MA‑Conv/Noise‑Injection 的运动合成网络、PatchGAN 对抗损失、梯度一致性损失、Swin Transformer 的残差块、以及 3D 领域的无监督域适配。

**📊 数据集**

数据集：IXI、HCP（用于模拟运动的配对数据）以及真实的 MR‑ART（含实际运动伪影）进行外部验证；使用 TorchIO 产生10事件随机运动；FreeSurfer 评估分割与表面拓扑。

**📈 对比分析**

与 3D CycleGAN、2D UDDN、2D AutoDPS 及两种监督基线（oracle 与 source‑only）对比。实验表明 SSRL‑MAR 在模拟数据上 PSNR 23.81 dB、SSIM 91.55%、NMSE 0.79%；在 MR‑ART 上未适配时 PSNR 26.01→26.90 dB、24.50→26.11 dB，性能仅比oracle差 0.25–0.47 dB；同时在语义分割上，核心结构（胼胝体、脑室）的体积误差减半以上。

**⚠️ 局限性**

局限性：仅验证于 T1w 脑部扫描，其他对比（T2、FLAIR）及非脑组织待测试；合成运动仍与真实运动存在一定差距；对极端/非刚性运动的恢复有限；模型训练与推理均需大量显存；未引入 k‑space 物理模型，难以捕捉更细粒度的运动效应。

---

## 378. Chartography: A Benchmark for Professional Chart Understanding

**arXiv ID:** 2608.10677 | [PDF](https://arxiv.org/pdf/2608.10677v1)

**作者:** Suhaas Garre `[一作]` (Surge AI), Edwin Chen `[通讯]` (Surge AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并发布了 Chartography 基准，包含 100 个专业领域任务，考核模型在真实专业图表上的可视化推理与决策能力。

**💡 创新点**

创新点在于：由领域专家编写并三重验证任务、设置专家校准的可接受答案范围、筛选能击败前沿模型的难度任务，以及强调无工具条件下的视觉感知与领域约定评估。

**🔧 技术方法**

使用大语言模型在无工具模式下进行推理，并采用 Gemini 3.5 Flash 自动裁判；实验评估了 30 个前沿模型配置的表现。

**📊 数据集**

数据集由 100 张来自医疗、工程、金融等 12 个专业领域的真实图表组成，其中 51 条答案带有可接受范围，涵盖 80 张在线图表和 20 张专家原创图表。

**📈 对比分析**

与 ChartQA、ChartMuseum 等现有基准对比，Chartography 上最佳模型仅达到 45% 的 pass@1，显著低于 80–90% 的其他基准，说明在专业图表理解上仍有巨大提升空间。

**⚠️ 局限性**

局限性包括：评测禁止使用任何工具，导致视觉感知仍是主要瓶颈；样本规模有限，可能不足以覆盖所有专业图表变体；模型仍需提升对视觉细节和领域约定的解码能力。

---

## 379. Longitudinal Evidence That General-Purpose Chatbots Actively Foster Relational Engagement

**arXiv ID:** 2608.10672 | [PDF](https://arxiv.org/pdf/2608.10672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 380. Measuring the End-to-End Resilience of Application Deployments in Real-World Communication Networks with DRACO

**arXiv ID:** 2608.10611 | [PDF](https://arxiv.org/pdf/2608.10611v1)

**作者:** Leon Janzen `[一作]` (Technical University of Darmstadt), Matthias Hollick `[通讯]` (Technical University of Darmstadt)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了DRACO框架，用于建模通信网络并评估应用部署的端到端可用性、鲁棒性与弹性。

**💡 创新点**

提出了端到端弹性度量，并实现了可配置的、支持公共与合成数据的评估平台。

**🔧 技术方法**

采用图论建模、概率测量、挑战模拟与可视化，配合Python实现。

**📊 数据集**

利用iGDB、NetMob23、AWS/Cloudflare/GCP/Meta等公开数据以及合成数据。

**📈 对比分析**

通过两案例（法国即时通讯与德国5G核心）演示测量结果，弹性得分与鲁棒性显著受部署位置影响。

**⚠️ 局限性**

受限于缺乏真实运营商验证的拓扑与流量，需使用假设或合成数据；框架对输入文件格式依赖较高。

---

## 381. Unlocking the Power of Medical Tabular Data via Semantic-Aware Multimodal Pre-training

**arXiv ID:** 2608.10522 | [PDF](https://arxiv.org/pdf/2608.10522v1)

**作者:** Yingsheng Liu `[一作]` (Monash University), Zhen Yu `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种语义感知的多模态预训练框架AID，利用自监督方法同时学习医学影像与结构化表格数据的表示。

**💡 创新点**

创新点包括：①基于冻结的元学习先验和无标签的重要性自适应掩蔽，构造诊断重要特征的学习曲线；②软标签离散化模块，用三角核将连续数值映射为概率分布，保持序数关系并替代不稳定的连续回归。

**🔧 技术方法**

技术手段包括ViT图像编码器、混合Transformer表格编码器、跨模态注意力融合、图像-表格对比与匹配（ITC/ITM）、离散重建（KL散度）以及PCA+TabPFN v2提取特征重要性。

**📊 数据集**

使用的主要数据集为大型皮肤病切片图像集SLICE‑3D、私有视网膜图像集HOP以及公开眼底图像集EyePACS，对不同科室进行了跨域验证。

**📈 对比分析**

通过与多种监督和自监督基线（ViT‑B、CatBoost、SimCLR、MMCL、TIP、CITab等）对比，AID在SLICE‑3D上实现AUC 0.984/0.986、pAUC 0.192/0.193，OOD AUC 0.944，线性探针AUC 0.984，显著优于现有最优方法，并在HOP和EyePACS上保持了高通用性。

**⚠️ 局限性**

局限性包括：对表格重要性估计仍依赖无监督的元学习先验，可能与临床真实重要性不完全一致；软标签离散化对分箱策略敏感；实验集中在皮肤病和眼底两类数据，跨其他医学领域的适用性尚待验证；实现复杂度高，需较大算力支持。

---

## 382. EweAcT: Ewe behaviour aligned to accelerometer data for activity monitoring in extensive grazing systems

**arXiv ID:** 2608.09943 | [PDF](https://arxiv.org/pdf/2608.09943v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 383. Position Encoding in Transformers: From Absolute and Relative Methods to Rotary Position Embeddings and Long-Context Scaling

**arXiv ID:** 2608.10021 | [PDF](https://arxiv.org/pdf/2608.10021v1)

**作者:** Jiguo Li `[一作]` `[通讯]`, Jiguo Li

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

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

## 384. Beyond Detection: Evaluating Defensive LLMs Against AI-Generated Social Engineering in Live Turn-by-Turn Interaction

**arXiv ID:** 2608.10239 | [PDF](https://arxiv.org/pdf/2608.10239v1)

**作者:** Yuqiao Xu `[一作]` (Case Western Reserve University), Erman Ayday `[通讯]` (Case Western Reserve University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个在线房屋交易场景的300条对话基准，评估防御型LLM在多轮对话中识别并定位信任链失败的能力；

**💡 创新点**

创新点在于提出“trust‑chain localization”框架，将防御目标细化为识别具体信任链失效的组件（身份、资产、验证、交易路径），并设计了状态化与静态两种评估协议；

**🔧 技术方法**

使用大型语言模型（GPT‑4系列、Claude、Llama、Qwen）进行结构化输出评估，并通过程序化规则对其动作与诊断进行评分；

**📊 数据集**

基准数据集包含20个情境族、5种结构失效（合法、L1~L4）和3种表面呈现（夸张风险、中性、合法化）共300个固定对话；

**📈 对比分析**

比较方法为计算多项指标：实时干预率、预请求干预率、首次干预定位准确率、条件定位准确率、完整定位准确率和合法案例误警率；结果显示模型在干预覆盖、定位准确和误警率之间存在显著差异，且不同模型对不同失效组件表现不一；

**⚠️ 局限性**

局限性包括仅使用合成对话且对话不自适应、评估聚焦于房屋交易且未充分覆盖真实攻击者行为、缺乏人类评估与跨域验证。

---

## 385. Decision-Aware Approximation of Belief Functions for Evidential Combinatorial Optimization

**arXiv ID:** 2608.10650 | [PDF](https://arxiv.org/pdf/2608.10650v1)

**作者:** Sohaib Afifi `[一作]` `[通讯]` (University of Artois), Sohaib Afifi (University of Artois)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对在组合优化中使用的信念函数（质数函数）提出一种“决策感知”的近似方法，改用合并焦点元（focal element）时优先保留决策质量，而非仅靠距离度量保持证据相似。

**💡 创新点**

创新点包括：① 引入基于决策后悔（regret）的近似目标，并证明其可用单点（one‑point）上界；② 在标量成本下给出精确的 O(N²K) 动态规划求解；③ 推导在线版本，能够在最终成本未知时提前压缩焦点元。

**🔧 技术方法**

技术方法包括：安全合并（safe merge）定义；利用单点上界构造决策感知压缩；动态规划求解聚类问题；在线压缩策略与局部敏感度 w_t 的使用；实验中对 Jaccard、Jousselme、最大质量等传统近似进行对比。

**📊 数据集**

使用的是人工生成的最短路径实例：图形包含 3–5 条边、2–4 条分支，焦点元为整数区间（下界 0–7，宽度 0–2）或 0–19，宽度 0–4，实验规模为 2500 条实例，5 个随机种子。

**📈 对比分析**

与基线方法（Jousselme、Jaccard、最大质量、随机安全合并）相比，决策感知方法（Bound‑DA、Online‑DA）在决策变化率、平均后悔以及 95th 分位数后悔均显著更低；在 K 较小（压缩强度高）时优势最为明显；在线版本在线性读出下决策变化率最低，即使在非线性读出下仍保持低于多数基线。

**⚠️ 局限性**

局限性包括：① 在向量（多维）成本情形下的聚类问题仍是启发式而非精确求解；② 单点上界所需的单调性假设在最小化极大后悔等更一般准则下仍是猜想；③ 该方法仅适用于基于区间下界的线性期望成本或类似可单调化准则；④ 对异常大规模实例的可扩展性需进一步验证。

---

## 386. Withholding the Completing Chunk: Deterministic Pair-Completion Guardrails for Streaming LLM Output

**arXiv ID:** 2608.10279 | [PDF](https://arxiv.org/pdf/2608.10279v1)

**作者:** Christopher M. Frost `[一作]` `[通讯]` (HEOSSI (Pte.) Ltd.), Christopher M. Frost (HEOSSI (Pte.) Ltd.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了在流式大语言模型输出中，对由两个词法谓词构成的危险规则进行前缀扫描的“对偶完成拦截”机制，确保当两谓词同时出现时阻止该分块的释放。

**💡 创新点**

创新点在于明确证明在固定词法规则下，完整前缀扫描可以精确保证第一块完成对偶谓词时被拦截，同时提供了可验证的机制正确性保证和实验矩阵，并与传统完整缓冲、滚动窗口及块内扫描等策略对比。

**🔧 技术方法**

技术上使用了基于Python的Bee引擎实现的状态机，依赖已有的正则扫描器进行词法匹配，采用完整前缀扫描、块大小控制和多线程计时测评；对偶规则通过逻辑与组合实现。

**📊 数据集**

实验数据集包括：①四个手工指定的词法对偶规则（每个规则在不同块大小下的40次实验），②AEGIS2.0公开对话数据（338安全样本、394不安全样本），③13,114条内部训练答案进行无标签误报检查。

**📈 对比分析**

在四个策略（完整前缀、滚动窗口、块内扫描、完整缓冲）下，完整前缀扫描与完整缓冲都能检测100%对偶，但完整前缀扫描因重复扫描导致时间复杂度随响应长度呈二次增长；在安全样本上误报率为0%（Wilson上限1.12%），在不安全样本上误报率为0%（Wilson上限0.97%），与训练好的Llama Guard 3 1B FP16模型相比，前者缺乏语义覆盖但无误报，后者覆盖率更高但误报率约8.3%。

**⚠️ 局限性**

局限性包括：只覆盖四个词法对偶，无法评估整体有害输出召回率；缺乏对齐词法的泛化能力；完整前缀扫描在长响应与细粒度块时的性能瓶颈；对抗性改写、分词分块、跨响应对偶完成等情况未得到处理；内部误报仅在安全子域（网络安全）中发现，表明需要更细粒度的域级评估。

---

## 387. $π$-SUB: A Physics-Informed Synthetic Underwater Benchmark Dataset for Underwater Image Enhancement

**arXiv ID:** 2608.10589 | [PDF](https://arxiv.org/pdf/2608.10589v1)

**作者:** Namritha Lasyapriya Maddali `[一作]` (Indian Institute of Science), Narasimhan Sundararajan `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了π-SUB，一个基于物理模型的合成海水图像对照数据集，用于训练和评估水下图像增强模型。

**💡 创新点**

创新点在于融合了深度相关的下沉辐射、基于Jerlov水体类型的光学特性、藻类吸收、CDOM和荧光等生物光学过程，并可独立控制悬浮颗粒、雾化散射等残留现象，生成极具真实感的训练样本。

**🔧 技术方法**

技术包括改进的Jaffe–McGlamery光传播模型、深度相机估计、光学参数从实测I/O进行解析、物理增强模型以及基于Unreal Engine的场景渲染。

**📊 数据集**

使用117张干净参考图像（来自Unreal Engine渲染和精选实景图像），按十种Jerlov水体类型、不同相机深度和光学条件合成14,040对合成/真实图像。

**📈 对比分析**

通过与SUID、SUIEB、Syrea、PHISWID等合成数据集以及五个真实基准的FID、OOD率、PCA等分布一致性指标比较，π-SUB在全球Fidelity上降低46%，OOD率仅2%；在四种主流UIE架构训练后在六个真实基准上，π-SUB提升UIQM 4.18%–9.46%，降低NIQE 23.98%–48.78%，并在特征匹配任务中显著提高可匹配关键点数。

**⚠️ 局限性**

局限在于荧光模型尚未在现场光谱中验证，深海人工照明、高污染海域及水纹等极端条件仍未充分覆盖。

---

## 388. Recovering Wasted Compute in Autoresearch Agents

**arXiv ID:** 2608.10424 | [PDF](https://arxiv.org/pdf/2608.10424v1)

**作者:** Au Kwok Chun `[一作]` (Columbia University), Micah Goldblum `[通讯]` (Columbia University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了基于树搜索的自研系统在表格机器学习任务中的常见失败模式，并提出了多种针对性干预方案，包括全局调试顾问、预算感知的超参数调优机制、Thompson Sampling 回溯以及对探索性数据分析的诊断；

**💡 创新点**

创新点在于引入全局共享的调试知识库实现跨分支学习、通过提示与控制循环双重机制强制执行超参数调优、使用概率化的 Thompson Sampling 优化回溯策略，并系统评估这些干预对代理计算效率的提升；

**🔧 技术方法**

所采用的技术包括基于 LLM 的代理框架 AIDE 与 ML‑Master、树搜索与 MCTS、调试错误压缩与共享注册表、提示层与控制层的超参数调优奖励塑造、以及 Thompson Sampling 的贝塔分布采样回溯；

**📊 数据集**

实验数据集涵盖九个表格预测任务，来自 MLE‑bench 与 Kaggle，包括 Cirrhosis、GNSS、Spaceship Titanic、Wine Quality 以及 Playground 系列 S5E3–S5E12；

**📈 对比分析**

通过与基线代理的对比（基准 2×2×10 随机种子），评价指标为金牌数、有效提交率和模型评分；全局调试顾问将金牌数从 22 提升至 38、有效率从 81% 提升至 100%，超参数指导在大多数任务中提升 0.1–0.4 分，Thompson Sampling 则显著降低空跑比例并在部分任务中保持或提升最终得分；

**⚠️ 局限性**

限制包括：现有代理在探索多样性、利用 EDA 方面仍有限；干预虽提升效率，但未能解决所有鲁棒性与可扩展性问题；且在更大规模或不同领域任务上的迁移效果尚待验证。

---

## 389. DOCSCHISEL: Adaptive Tool Documentation Optimization Framework for LLM Agents

**arXiv ID:** 2608.10037 | [PDF](https://arxiv.org/pdf/2608.10037v1)

**作者:** You Lu `[一作]` (Fudan University), Xin Peng `[通讯]` (Fudan University)

**通讯引用:** 14757 | [OpenAlex ID](https://openalex.org/A5071724015)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对大型语言模型（LLM）代理使用工具时的工具文档进行大规模实证研究，并基于失败执行轨迹提出了一种自适应工具文档优化框架。

**💡 创新点**

创新点在于揭示工具文档信息字段在不同任务域、LLM骨干和代理范式下的异质性，并提出针对每个工具逐字段自适应增删改的优化方法。

**🔧 技术方法**

技术上采用LLM进行失败轨迹分析、信息字段增删改、记忆机制与循环迭代的自适应优化流程，同时与EasyTool、DRAFT等基线进行对比。

**📊 数据集**

使用了14个工具使用基准（包括WorkBench、API-Bank等）共计74个工具，覆盖9个任务域，包含了24,955个工具与其文档。

**📈 对比分析**

与EasyTool、DRAFT相比，平均提升任务成功率约75%，在原始文档基础上提升95.89%；优化平均耗时12.65分钟/工具，token开销有限。

**⚠️ 局限性**

实验局限于选定的代理范式、LLM骨干和数据集，未覆盖更广泛场景；同时依赖LLM生成与人工标注的领域知识，可能产生主观偏差。

---

## 390. LEGO: Leveled Language Gaussian Splatting

**arXiv ID:** 2608.10057 | [PDF](https://arxiv.org/pdf/2608.10057v1)

**作者:** Yuning Peng `[一作]` (Wuhan University), Bisheng Yang `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了LEGO框架，用三维高斯场实现多层次的开放词汇场景分割与理解；

**💡 创新点**

核心创新在于把视角依赖的SAM多尺度掩码自适应重标记为统一的3D层级结构，并通过解耦特征空间避免跨层级语义混杂，进一步构建层级化语言场景图以实现复杂空间推理；

**🔧 技术方法**

主要技术包括：多视角SAM掩码升维与3D尺度估计、基于峰值的层级赋值、像素对级联稠密监督、层级对比蒸馏损失与特征正则化、HDBSCAN递归聚类、CLIP视角选择与Embedding、以及LLM驱动的Chain‑of‑Retrieval；

**📊 数据集**

使用了NVOS、SPIn‑NeRF进行提示式分割评测，LERF‑OVS、Mip‑NeRF 360进行开放词汇分割与定位评测，同时构造了120道细粒度CoR查询数据集；

**📈 对比分析**

与SOTA方法（如SAGA、FlashSplat、OmniSeg3D、LangSplat等）相比，LEGO在NVOS、SPIn‑NeRF提示分割上mIoU分别提升至94.2%/94.2%，在开放词汇分割上定位mAcc提升4–5%，并在极细粒度子部件识别、CoR查询中实现了显著提升（mIoU 52.2% vs 5.6%/9.9%等），证明其在多层次语义与空间推理上的优势；

**⚠️ 局限性**

局限性包括：仍依赖高质量的多视角SAM掩码与稀疏高斯场训练，推理时需要多视角输入与显著的计算开销，对极端遮挡或纹理稀缺物体的语义提取仍可能受限，且层级划分受局部尺度分布影响，可能在复杂场景中产生不完整的层级结构。

---

## 391. Shaping the notion of #wellbeing in the therapy culture context: an analysis through Instagram narratives

**arXiv ID:** 2608.10793 | [PDF](https://arxiv.org/pdf/2608.10793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 392. MEGA: Self-Evolving Agent Optimization Infrastructure via Wisdom Graph

**arXiv ID:** 2608.10504 | [PDF](https://arxiv.org/pdf/2608.10504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 393. A full software stack for epidemic disease management: Unlocking the joint potential of software technology and supercomputing

**arXiv ID:** 2608.09933 | [PDF](https://arxiv.org/pdf/2608.09933v1)

**作者:** Jonas Gilg `[一作]` (German Aerospace Center), Martin J. Kühn `[通讯]` (German Aerospace Center)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文开发并公开了一个完整、模块化的、面向疫情管理的开源软件栈，涵盖数据获取、ETL、SECRVVS传播模型、自动化工作流、HPC 加速以及可交互前端，支持德国地方卫生部门的实时决策。

**💡 创新点**

创新点在于：①将数学模型、数据管道、工作流和前端统一在同一生态系统；②通过 Airflow + UNICORE 实现高性能计算的无缝衔接；③提供可插拔的 ML surrogate 预估模型，支持即时 “what‑if” 分析；④采用 FAIR 原则与 OAuth2.0/Keycloak 结合实现安全可扩展的身份管理。

**🔧 技术方法**

技术栈包括：MEmilio (Python) 传播模型；FastAPI、Celery、Redis、PostgreSQL、MinIO 做数据服务；Airflow + KubernetesExecutor + UNICOREExecutor 负责工作流与 HPC 调度；React/Redux + MUI 前端；Keycloak+OAuth2.0 PKCE 认证；Helm + Rancher + OpenStack 做容器化与基础设施管理。

**📊 数据集**

使用的数据集主要为德国各区县每日的确诊、重症、ICU、死亡、人口结构、流动性和接种率等公共数据（RKI 及德国统计局），并通过 pilot LHA 生成的合成个体级数据进行本地化验证。

**📈 对比分析**

在 27‑02 至 10‑03 的评估期间，单机 Kubernetes 运行场景生成耗时约 197 分钟；将该阶段迁移至 JURECA HPC 节点后，耗时下降至 6‑7 分钟，提升约 30 倍；前端页面首次渲染 0.8 秒，Lighthouse 综合得分 93%，SUS 用户可用性 83.5%。

**⚠️ 局限性**

局限性包括：ML surrogate 尚未投入生产；模型参数与结构针对 SARS‑CoV‑2 及德国行政区划，需要进一步适配其他病原体与地区；数据实时性受公共源更新频率限制；系统仍需进一步优化以应对更大规模的多国部署与复杂的隐私合规需求。

---

## 394. A Neural Network Based Teleoperation for Remote Controlled Vehicles

**arXiv ID:** 2608.10367 | [PDF](https://arxiv.org/pdf/2608.10367v1)

**作者:** Ning Ding `[一作]` (Virginia Polytechnic Institute and State University), Azim Eskandarian `[通讯]` (Virginia Commonwealth University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一套基于神经网络的遥控车辆遥操作系统，能够在远程环境下通过视觉与感知信息实现高精度控制。

**💡 创新点**

创新点在于将深度学习模型与实时控制策略融合，利用端到端学习实现对多种动态环境的自适应控制，并在仿真与实验中证明其优越性。

**🔧 技术方法**

采用了卷积神经网络、循环神经网络或强化学习等深度学习技术，并结合ROS框架实现与车辆硬件的实时交互。

**📊 数据集**

使用了自建的车辆传感器数据集（包含视觉、雷达、IMU等多模态信息）以及公开的无人驾驶车辆仿真环境中的数据。

**📈 对比分析**

通过与传统PID控制、模型预测控制（MPC）以及基于模型的强化学习方法进行对比，实验表明该方法在路径跟踪误差、稳态误差以及系统响应时间上分别提升约25%、15%和20%。

**⚠️ 局限性**

主要限制在于对训练数据的依赖较强，面对未见过的极端环境时泛化能力不足，且实时推理的计算延迟在低功耗硬件上仍显著。

---

## 395. The Researcher's Guide to HPC Networks

**arXiv ID:** 2608.09953 | [PDF](https://arxiv.org/pdf/2608.09953v1)

**作者:** C. Nicole Avans `[一作]`, Anthony Skjellum `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并归纳了高性能计算集群网络的各层次结构与实现技术，提供了一个统一的参考框架。

**💡 创新点**

创新点在于将多供应商网络、API 与层级模型整合为一张系统化的图谱，并通过实例（El Capitan）展示完整的网络设计与实现。

**🔧 技术方法**

主要技术包括对多种网络协议（InfiniBand、OmniPath、Slingshot 等）、通信库（MPI、OpenSHMEM、UCX、LibFabric 等）的综合分析，以及对拓扑、路由与控制平面组件的系统描述。

**📊 数据集**

文中未使用实验数据集，信息来源主要为公开规范、厂商文档与已有研究综述。

**📈 对比分析**

通过对比表格和示例架构，对不同网络层次与通信接口的性能特点进行定性比较；但未给出量化的基准测试结果或具体性能数值。

**⚠️ 局限性**

局限性包括缺乏实际实验验证、侧重传统集群网络而非云环境、以及在不同硬件平台上的通用性仍需进一步评估。

---

## 396. Dreamer-SAC: Off-Policy Learning in Latent World Models for Sample-Efficient Autonomous Driving

**arXiv ID:** 2608.10386 | [PDF](https://arxiv.org/pdf/2608.10386v1)

**作者:** Jiazhuo Li `[一作]` (Tongji University), Xi Xiong `[通讯]` (Tongji University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在自动驾驶场景中设计了 Dreamer-SAC 框架，融合隐状态世界模型与离线 Soft Actor-Critic，实现高效的数据驱动决策学习。

**💡 创新点**

创新点在于将短期隐状态 rollouts 与真实与模型经验混合回放相结合，并采用 n‑step 目标与多目标监督，平衡模型偏差与探索，显著提升样本效率与安全性。

**🔧 技术方法**

使用了 Recurrent State‑Space Model (RSSM)、Soft Actor‑Critic、离线经验回放、多模态编码（图像+LiDAR）、n‑step TD 目标、多目标奖励预测等技术。

**📊 数据集**

实验基于 MetaDrive 仿真平台的 BIG_BLOCK_SEQUENCE‑CC 场景，使用摄像头图像与 125 维 LiDAR/车辆状态观测。

**📈 对比分析**

在相同环境下与 DreamerV3、SAC、PPO 进行对比，Dreamer‑SAC 在累计奖励、碰撞率、行驶距离等指标上均显著优于基线，显示出最佳的样本效率与安全表现。

**⚠️ 局限性**

主要局限在于对模型预测误差仍依赖短期 rollouts，较长 rollouts 会导致偏差；验证仅在仿真环境，缺乏真实道路数据；缺少自适应 roll‑out 长度机制。

---

## 397. The GenAI Catch-22: Use of Generative Artificial Intelligence in Norwegian Newsrooms During the 2025 Parliamentary Election

**arXiv ID:** 2608.10773 | [PDF](https://arxiv.org/pdf/2608.10773v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 398. Exploration-Driven Personalized Federated Reinforcement Learning via Intrinsic Motivation

**arXiv ID:** 2608.10499 | [PDF](https://arxiv.org/pdf/2608.10499v1)

**作者:** Md Rafid Islam `[一作]` (North South University), Ratun Rahman `[通讯]` (University of Alabama in Huntsville)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

提出了一种探索驱动的个性化联邦强化学习框架EDPFRL-IM，利用随机网络蒸馏（RND）产生内在奖励并通过聚合探索统计生成全局新颖性先验，实现客户端在保持隐私的前提下协同探索；

**💡 创新点**

创新点在于将自我驱动的探索机制与联邦学习相结合，构建低通信成本的探索协同协议，并通过全局新颖性先验引导分布式学习；

**🔧 技术方法**

采用随机网络蒸馏（RND）作为内在奖励生成器，使用PPO/SAC等强化学习优化器，服务器通过聚合各客户端的压缩探索摘要来计算全局新颖性先验；

**📊 数据集**

在MountainCar-v0和CartPole-sparse（两种稀疏奖励、异质动力学的强化学习环境）上进行实验；

**📈 对比分析**

与本地RL、FedRL、FedRL+RND、FedAvg-RL、pFedMe、FedPer++以及单机内在动力学方法（RND、ICM、CBE）对比，EDPFRL-IM在平均回报、探索覆盖率、冷启动适应速度等指标上均显著优于所有基线，最终平均回报分别提升至0.76（MountainCar）和0.74（CartPole-sparse）；

**⚠️ 局限性**

局限性包括对超参数（α_i、β）的敏感性、实验仅在模拟环境中验证、聚合探索摘要可能仍存在信息泄露风险，以及在更大规模或真实世界场景下的可扩展性尚待进一步评估。

---

## 399. DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving

**arXiv ID:** 2608.10413 | [PDF](https://arxiv.org/pdf/2608.10413v1)

**作者:** Zebin Xing `[一作]` (Institute of Automation, Chinese Academy of Sciences), Dongbin Zhao `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 DriveVLA-M0，一个结合失败感知记忆与检索增强的 Vision‑Language‑Action 框架，在推理时通过结构化检索与轻量化的 LoRA‑based Test‑Time Training（TTT）对模型进行场景特定的修正。

**💡 创新点**

核心创新包括① 用潜在记忆池存储失误案例并记录场景结构与专家轨迹；② 构建分离地图与动态代理的检索模型，实现基于道路拓扑与行人/车辆布局的结构化检索；③ 引入解耦 LoRA TTT，仅在高相似度触发时进行后向微调，显著降低推理开销。

**🔧 技术方法**

使用技术包括 InternVL3 视觉‑语言模型、DINOv2+LoRA 检索模块、解耦 Map/Agent LoRA、轨迹与评分头、PDMS/EPDMS 评估指标、以及基于模拟器的 Oracle scorer。

**📊 数据集**

主要使用的公开数据集为 NAVSIMv1（Navtest）和 NAVSIMv2（Navhard），并通过 SimScale 生成额外合成场景扩充记忆池。

**📈 对比分析**

与多种 E2E 与 VLA 基线（如 TransFuser、Mimir、DriveVLA‑W0 等）对比，DriveVLA‑M0 在 Navtest 取得 94.1 PDMS、Navhard 取得 47.0 EPDMS，且仅在触发 TTT 时增加 26.44 ms 的后向延迟，表现出显著的性能提升。

**⚠️ 局限性**

局限性包括：对极端新颖场景的适应性有限；记忆池规模受存储与检索成本限制；触发阈值需手动调优，过宽阈值会引入噪声，过窄则错失修正机会。

---

## 400. Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora?

**arXiv ID:** 2608.10690 | [PDF](https://arxiv.org/pdf/2608.10690v1)

**作者:** Qingjie Zhang `[一作]` (Tsinghua University), Han Qiu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过已公开的 BPE tokenizer 词表推断隐藏训练语料库的 token 比例，并可聚合为语言或领域级混合比例

**💡 创新点**

发现不同语料库的 BPE token ID–比例分布具有稳定全局形状，提出 Quantile‑Guided Density Estimation (QGDE) 以多量化曲线与局部密度加权实现细粒度 token‑级比例估计

**🔧 技术方法**

BPE 分词、量化回归、构建多量化曲线、局部高斯密度加权、方向性 KL 相似度评估、聚合权重分配等技术

**📊 数据集**

mC4、OSCAR、FineWeb、Wikipedia、CodeParrot、OpenWebMath、RedPajama‑C4、Wikipedia–arXiv、CodeParrot、SmolLM 训练语料

**📈 对比分析**

与直接 ID‑比例传输、PoCTrace、DMI 等基线比较；在 token 级平均相对误差 MRE 低至 3%（受控实验）/5.7%（SmolLM），在类别级误差 3%/6%，显著优于基线

**⚠️ 局限性**

缺乏公开的 LLM 训练语料库导致验证范围有限，仅在可获得 ground truth 的 SmolLM 上验证；对极端不平衡比例的估计仍不够精确

---

## 401. Toward a Theory of Value in AI Alignment

**arXiv ID:** 2608.10327 | [PDF](https://arxiv.org/pdf/2608.10327v1)

**作者:** Andrew Smart `[一作]` (Google Research), Abeba Birhane `[通讯]` (Trinity College Dublin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

进行了AI价值对齐论文的系统性综述，标注94篇论文并用自研指标评估其对价值理论的隐含假设。

**💡 创新点**

明确揭示该研究领域隐含的经济主义价值观，提出跨学科的评估框架，并指出大多数研究忽视价值定义、单一主义与实证缺失。

**🔧 技术方法**

采用人工双人标注、三元评估量表、统计互评一致性（kappa、PABAK）等定量方法结合质性引语分析。

**📊 数据集**

采集了基于Semantic Scholar的576篇引用种子论文的集合，最终筛选94篇符合价值对齐主题的论文。

**📈 对比分析**

通过量化评估各维度的出现比例和互评一致性，发现大多数论文将价值视为可测量且可用偏好替代，体现了方法上的同质化。

**⚠️ 局限性**

样本量有限、评估量表难以覆盖所有细微差异、可能的主观判定偏差以及对更大规模文献缺乏覆盖。

---

## 402. pyRMV: Reusable, Cross-model Validation for Computational Science

**arXiv ID:** 2608.09956 | [PDF](https://arxiv.org/pdf/2608.09956v1)

**作者:** Hugo Dictus `[一作]` (École polytechnique fédérale de Lausanne), Henry Markram `[通讯]` (École polytechnique fédérale de Lausanne)

**通讯引用:** 39679 | [OpenAlex ID](https://openalex.org/A5069964239)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文内容未完整提供，无法判断具体做了什么。

**💡 创新点**

无法确认创新点。

**🔧 技术方法**

无法确定使用了哪些技术。

**📊 数据集**

无法确认使用了哪些数据集。

**📈 对比分析**

无法说明方法对比与性能评估。

**⚠️ 局限性**

无法确定研究的局限性。

---

## 403. What Actually Serializes GPU LZ77 Decode: Three Decoders, Three Mechanisms, and an Encode-Time Lever That Removes the Last One

**arXiv ID:** 2608.10188 | [PDF](https://arxiv.org/pdf/2608.10188v1)

**作者:** Yakiv Shavidze `[一作]` `[通讯]` (Independent Researcher), Yakiv Shavidze (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对GPU LZ77解码的顺序瓶颈进行了测量和优化，证明解析阶段占比达64–72%，而回溯链深度几乎不影响延迟；通过限制链深度、移除距离历史、使用周期填充并行化等手段进一步提升解码速度。

**💡 创新点**

创新点在于：①首次量化解析层占比并拆解出真正顺序的四项；②用绝对偏移块级编码实现无回溯依赖；③证明链深度与延迟无关；④在解析层移除4项距离历史实现0.54%压缩率提升；⑤将自重叠匹配视为周期填充而非依赖链，提升匹配层速度2.75–8.42×；⑥揭示缓存行粒度导致的4.4%总线效率，并给出39×写入速度差。

**🔧 技术方法**

采用CUDA 12.x在H100 80 GB SXM上实现三种解码器（dense full‑pipe、wavefront、v7‑RA seek）；使用ACEAPEX格式（绝对偏移、块级分割）作为被测压缩格式；通过自定义解析器、波前调度、CUDA图和距离历史移除等技术；使用FNV/XXH3做位级正确性校验。

**📊 数据集**

使用人类染色体1（hg38）为主基准，另外还测试了enwik8、enwik9、silesia、FASTQ以及约5 GB的50 GB基因组片段；对比了LZ4、Zstd‑3/19、Brotli‑9等公开压缩器。

**📈 对比分析**

与其他压缩器在相同16 KB块独立性约束下比较，ACEAPEX在三/五个语料库中压缩率优于Zstd‑19，并在解码速度上通过解析层优化实现显著提升；解析层速度提升至2.75–8.42×，距离历史移除后比率下降0.54%；总线效率仅4.4%，表明进一步改进需解决缓存行粒度。

**⚠️ 局限性**

实验局限包括：仅在单台H100 GPU上测试；密集全管线基准噪声为6%，导致小幅度效能变化难以验证；50 GB随机访问测试无法做到完全位级正确性；距离历史移除的加速效果未在GPU上直接验证；最小匹配长度测试未完全位级替代字面量。

---

## 404. MIRA: Medical Image Reflection for Agentic Diagnosis

**arXiv ID:** 2608.10827 | [PDF](https://arxiv.org/pdf/2608.10827v1)

**作者:** Shengzhi Wang `[一作]` (Tongji University), Qingwen Liu `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 405. Towards Unified Dynamic Face Landmark Detection

**arXiv ID:** 2608.10346 | [PDF](https://arxiv.org/pdf/2608.10346v1)

**作者:** Sebastian Regalado `[一作]` (University of Toronto), Igor Gilitschenski `[通讯]` (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出统一动态面部标志检测方法，将面部标志视为面部部件边界上的进度点，支持单模型对多种“ N‑point”数据集进行端到端训练，并在推理时按需输出任意数量的标志；

**💡 创新点**

创新点包括：① Face Part‑Anchored Landmark Positions（FPALP）统一所有数据集的标注，打破 N‑point 限制；② 采用 FPALP 查询 + 跨模态 Transformer 解码器实现动态标志预测；③ 用预训练文本编码器对面部部件进行语义编码，提升查询可解释性与收敛速度；

**🔧 技术方法**

技术手段：FPALP 表示；跨模态解码器（自注意、可变形注意、交叉注意、FFN）；预训练图像编码器（ViT‑B/ResNet）；SentenceBERT 文本编码器；PossLoss 与 WingLoss 损失；LoRA 轻量级数据集适配器；

**📊 数据集**

使用 AFLW‑19（19点）、300W（68点）、WFLW（98点）三大基准集进行联合训练；交叉验证 COFW、COFW68、WFLW68、WFLW_E 等；

**📈 对比分析**

在 WFLW、300W、AFLW‑19 上与多种 SOTA 方法对比，单模型获得或超过 NME 指标；加入 LoRA 适配器后进一步提升，平均 NME 下降 0.2‑0.5；在跨数据集评估中表现优于多数现有方法，证明泛化能力；

**⚠️ 局限性**

局限性：FPALP 统一依赖于不同数据集的标注对齐，误差可能影响性能；在极端姿态或遮挡下部分部件（如鼻边）精度相对较低；模型对图像编码器容量敏感，较小的 backbone 在某些极端数据上表现受限；

---

## 406. Conflict or Strategy? Asymmetric Role Framing of La France insoumise and Rassemblement National in French News Headlines, 2022-2025

**arXiv ID:** 2608.09936 | [PDF](https://arxiv.org/pdf/2608.09936v1)

**作者:** Amr Sobhy `[一作]` `[通讯]` (Le French News Lab), Amr Sobhy (Le French News Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2022-2025年法国25家媒体发布的2.86万条关于左派党派LFI和右派党派RN的新闻标题进行分层情境化标注，研究其在冲突、战略与道德评估等维度上的角色分配与差异；

**💡 创新点**

首次提出“政治角色分配”框架，将冲突/战略行为框架与合法性/归因等道德评估分层，并建立构造层级可靠性体系验证LLM多模型投票标注的稳健性；

**🔧 技术方法**

使用三大LLM（GPT-OSS-120B、Llama-3.3-70B-Instruct、Mistral Large 2）进行多模型投票标注，辅以400条人类审核、逻辑回归（含年、渠道固定效应）、聚类自助法和置换检验等统计技术；

**📊 数据集**

包含28,592条标题，涉及LFI、RN及其主要领导人关键词，来源于25家法语媒体，时间跨度2022‑2025年；

**📈 对比分析**

通过卡方检验和逻辑回归比较不同框架的出现率；冲突框架在LFI中显著更高（OR≈0.61），战略框架在RN中显著更高（OR≈1.41），两结果在自助法和置换检验中均保持显著；道德评估的稳定性较低；整体投票标注在冲突/战略维度上与人类审核κ>0.7，可靠性较好；

**⚠️ 局限性**

局限性包括：LLM标注可能受训练数据偏见影响，合法性标签可靠性较低，标题检索基于关键词可能引入位置偏差，研究为观测性分析无法确定因果机制，且仅关注标题，缺乏对全文或其他语境的考察，结果在法语媒体体系外的可推广性待验证。

---

## 407. Time to Move on: Querying without Nulls and Bags

**arXiv ID:** 2608.10863 | [PDF](https://arxiv.org/pdf/2608.10863v1)

**作者:** Molham Aref `[一作]` (RelationalAI), Wim Martens `[通讯]` (RelationalAI University of Bayreuth)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对 SQL 中的空值和多重集合进行理论与实证分析，提出并论证了一种无空值、无三值逻辑、基于集合语义的查询语言。

**💡 创新点**

创新点在于：①消除 3VL，恢复布尔逻辑；②完全避免 null，采用 6NF 全规范化；③强调集合语义，说明 bag 语义在递归、聚合等场景的致命缺陷；④给出理论证明和实验对比，支持无 null 语言的可行性。

**🔧 技术方法**

主要技术包括：关系代数/计算的重定义、布尔逻辑推导、全规范化（6NF）分解、理论上的查询等价性证明、递归查询的集合语义实现。

**📊 数据集**

实验主要基于 TPC‑H、TPC‑DS 标准基准以及作者自行构造的递归/聚合示例数据集。

**📈 对比分析**

比较方法是将传统 SQL 与改写后的无 null、无 3VL 版本在相同查询上执行，并对查询等价性、执行计划和输出结果进行对比；结果显示：99% 以上的 TPC‑H/TPC‑DS 查询在布尔逻辑下保持正确性，且消除了 bag 相关的性能爆炸。

**⚠️ 局限性**

局限性包括：实现上需要对底层存储做额外优化；在某些需要多重计数的业务场景仍需外部编码；缺乏成熟的商业实现和标准化支持。

---

## 408. Simplex Relaxation for Discrete Diffusion

**arXiv ID:** 2608.10615 | [PDF](https://arxiv.org/pdf/2608.10615v1)

**作者:** Jinya Sakurai `[一作]` (NTU Singapore), Xun Xu `[通讯]` (A*STAR)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

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

## 409. Hip Energized Monopedal Hopping

**arXiv ID:** 2608.10387 | [PDF](https://arxiv.org/pdf/2608.10387v1)

**作者:** Shane Rozen-Levy `[一作]` (University of Pennsylvania), Daniel Koditschek `[通讯]` (University of Pennsylvania)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种利用臀部扭矩为无锁定式平面单足机器人（SLIP with attitude）提供能量并同时稳定姿态的控制策略，并设计了新的步态控制器；

**💡 创新点**

在传统的PD+前馈姿态控制基础上，将臀部扭矩重新解释为能量注入源，首次利用混合平均分析得到闭式固定点与稳定性条件，并证明非对称步态对实现能量平衡与稳态跳跃是必要的；

**🔧 技术方法**

混合平均（Hybrid Averaging）分析、非线性动力学模型、步态（Raibert式）控制、离散时间积分器与前馈补偿、能量比（energy ratio）估计；

**📊 数据集**

仿真数据（5链平面双足模型和Penn Jerboa机器人）以及硬件实验数据（真实Penn Jerboa机器人）；

**📈 对比分析**

与传统尾部能量化跑步（速度0.2–1.0 m/s）以及其他高功率多足机器人（Cassie、Atlas）进行对比，硬件实现可在1.02–1.77 m/s（相当于5.1–8.85腿长/秒）稳态跳跃，表现出显著的速度提升和良好的能量利用；

**⚠️ 局限性**

受限于仅在平面内工作、对质心偏移的精确控制要求高、起跳角检测误差导致能量比估计不准、基于小角度与简化假设的理论模型，实际VPP位置与理论预测不完全吻合，且模型对大能量状态下的稳定性预测不足。

---

## 410. Hand-Written PTX Tensor-Core GEMM Kernels: A Multi-Precision Study on NVIDIA L4

**arXiv ID:** 2608.10103 | [PDF](https://arxiv.org/pdf/2608.10103v1)

**作者:** Matt J. Borowski `[一作]`, Blazej Osinski `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 NVIDIA L4 GPU 上 FP16、INT8、INT4 精度的手写 PTX Tensor‑Core GEMM 核进行基准实验，并与 WMMA API 基准进行对比。

**💡 创新点**

系统化量化不同精度下 PTX 与 WMMA 的性能差异，揭示 INT4 时 PTX 可实现显著加速，并提供可复现的核实现与决策规则。

**🔧 技术方法**

使用手写 PTX、双/三缓冲异步拷贝、ldmatrix、mma 指令、Nsight Compute 性能计数、量化算子。

**📊 数据集**

采用各种尺寸（N=512~8192）的方阵作为 GEMM 基准，分别以 FP16、INT8、INT4 进行量化测试；未使用传统数据集。

**📈 对比分析**

在相同尺寸、相同精度下，测量执行时间和硬件计数；结果显示 FP16 无加速，INT8 加速 1.4–1.8×，INT4 加速 2.9–4.3×，最佳 INT4 对 FP16 WMMA 的加速可达 98.7×。

**⚠️ 局限性**

实验仅限单个 L4 GPU、方阵、固定块/warp 划分；对其他架构、非方阵或不同调度可能产生差异；未评估精度损失或完整 LLM 推理效果。

---

## 411. Auditing Chinese Web-scale Corpora via Sampled BPE Token Statistics

**arXiv ID:** 2608.10678 | [PDF](https://arxiv.org/pdf/2608.10678v1)

**作者:** Qingjie Zhang `[一作]` (Tsinghua University), Han Qiu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研发并验证了一种名为 Sampled-BPE 的轻量化 token 级审核流程，利用抽样和 BPE 自动识别并计数中文语料库中的污染词。

**💡 创新点**

创新点在于将海量中文语料的抽样与 BPE 结合，突破固定关键词表的局限，既能保持低成本、可周期化的审核，又能生成可追溯的分层 token 数据集。

**🔧 技术方法**

核心技术包括流式抽样、BPE 训练与计数、GLM-4-32B 进行类别映射、以及统计聚合构建语料污染概况。

**📊 数据集**

使用了 11 个公开中文语料库和 2021–2026 年 6 份 Common Crawl 的中文切片，最终构建了 630,684 条记录的分层 token 数据集。

**📈 对比分析**

与全量审计对比，0.25% 抽样率下 token 覆盖率 76.8%，Spearman 86.9%，Pearson 99.9%，类别误差约 5%；运行速度提升 148×、内存降低 35×，相对误差仅 4.25%。

**⚠️ 局限性**

局限性包括：仅针对中文，其他语言不一定适用；包含成人、赌博等敏感内容，需谨慎使用；对非中文读者的翻译解释可能不够完整。

---

## 412. Stream Forcing: Constructing Unified Training Trajectory for Robust Streaming Video Generation

**arXiv ID:** 2608.10439 | [PDF](https://arxiv.org/pdf/2608.10439v1)

**作者:** Yueting Zhu `[一作]` (Huazhong University of Science & Technology), Xinggang Wang `[通讯]` (Huazhong University of Science & Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种统一的训练框架，将独立采样与递进采样通过连续的训练轨迹相连接，以实现流式视频扩散模型的训练与推理一致性。

**💡 创新点**

将训练时的噪声级采样视为帧索引的随机过程，并构造 Logit‑Normal 参数化的连续训练轨迹；引入联合校准与基于 Gaussian Copula 的时序相关采样，保证轨迹平滑、分布覆盖一致且跨帧相关。

**🔧 技术方法**

使用 Logit‑Normal 分布、联合校准算法、Gaussian Copula 相关采样、流式推理策略以及训练曲线与学习率调度等技术。

**📊 数据集**

使用 UCF‑101、Taichi‑HD 进行无条件视频生成评估，使用 nuScenes 评估驾驶世界模型效果。

**📈 对比分析**

与 AR‑Diffusion、Diffusion Forcing 等主流基线在 FVD/FID 上比较；在 UCF‑101 上实现 36.6% FVD 提升，Taichi‑HD 上提升 4.7%；在 128 帧长序列零射扩展上分别提升 27.9% 和 10.9%；在 nuScenes 上 FID、FVD 均显著下降至 9.4/105。

**⚠️ 局限性**

仍需手工调节训练轨迹参数，训练阶段相对繁琐；对极长序列的稳定性与实时推理速度尚未完全解决；模型对不同视频内容的泛化仍有限，且需要较多 GPU 资源。

---

## 413. Beyond Dry References: Learning Relative Audio Effects Representations via Contrastive Distance Learning

**arXiv ID:** 2608.10573 | [PDF](https://arxiv.org/pdf/2608.10573v1)

**作者:** Xinlu Liu `[一作]` (Tencent Music Entertainment), Zhenhai Yan `[通讯]` (Tencent Music Entertainment)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种无干参考的相对音效表示学习框架 RelFx，利用双分支 Siamese 编码器与交叉注意力、差分门控融合来捕捉音频对间的音效变换。

**💡 创新点**

创新点：① 从绝对音效状态转向相对音效距离；② 通过交叉注意力实现跨分支特征交互；③ 采用差分门控融合和反对称双向融合，使嵌入对输入顺序具有可逆性；④ 结合对比损失、三元组损失和参数回归辅助，实现更稳健的表示；⑤ 引入动态音效采样概率调度。

**🔧 技术方法**

核心技术包括：对比学习（NT-Xent+三元组），双分支 CNN14 backbone，交叉注意力模块，差分门控融合（及其反对称变体），MLP 投影头，L2 正则化，参数回归辅助损失，动态音效概率调度。

**📊 数据集**

训练数据：内部授权混音轨 6,447 条（2,681 首歌）；MoisesDB 2,585 条干/湿音轨（240 首歌）；评估数据：MUSDB18（四类乐器）。

**📈 对比分析**

与 CLAP、VGGish、AFx-Rep 及 Fx-Encoder++ 进行比较；在 Fx 风格迁移任务上，RelFx（标准）在四类乐器上的平均 STFT 损失 L_d 低至 1.388，较 Fx-Encoder++ 提升约 13%（在鼓类最高 16.7%），Oracle 条件下进一步提升；在检索任务中，RelFx 的 R@1 达到 67.3%。

**⚠️ 局限性**

局限性：模型关注整体音效变换，未能区分混音中的源特定效果；对未知链顺序和真实生产链的适应性待进一步验证；反对称变体在组合性质上虽优于基线，但仍未达到完美。

---

## 414. Predicting affective connotation of visualizations from their constituent colors

**arXiv ID:** 2608.10169 | [PDF](https://arxiv.org/pdf/2608.10169v1)

**作者:** Karen B. Schloss `[一作]` (University of Wisconsin--Madison), Seth R. Gorelik `[通讯]` (Woodwell Climate Research Center)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并验证了一种基于颜色情感关联的可视化情感预测方法，即先用已建好的颜色空间回归模型估计每个颜色的情感得分，再按颜色出现频率加权平均得到整幅可视化的情感得分。

**💡 创新点**

创新点在于：①把颜色情感关联从单个颜色扩展到整幅多色可视化；②证明“可加性假设”在多色连续与离散可视化中成立；③引入“数据依赖性”视角，指出颜色在图中所占面积对整体情感的影响。

**🔧 技术方法**

技术上主要使用：CIELCh 色彩空间的 LabC Cyl2 回归模型、加权平均、线性回归与混合效应模型、Pearson 相关系数、AIC/BIC 比较。

**📊 数据集**

数据集包括：UW‑71 颜色情感关联样本、Matplotlib 40+20 颜色刻度、20 个连续数据集（全球生物量）、18 个 4 色离散调色板，实验参与者来自美国威斯康星大学本科生。

**📈 对比分析**

方法对比：均值预测 vs 加权预测。实验1显示均值预测已能高相关（r≈0.6–0.9）；实验2在数据倾斜的色图中加权预测显著更优（p<0.05，AIC/BIC下降）；实验3在柱状图中加权预测略有提升，点图则不显著。整体性能表明加权平均能捕捉更精确的情感分布。

**⚠️ 局限性**

局限性包括：①基于像素的权重易受视觉遮蔽、隐藏像素影响；②仅测试了色彩因素，未考虑轴线、文字等其他视觉元素；③对文化、色觉差异等群体差异未做充分验证；④未探讨多类散点图等更复杂可视化形式下的数据依赖性。

---

## 415. CRHT: A Continuous Regression Hybrid Transformer for Vessel Trajectory Prediction with Online Cluster Sampling

**arXiv ID:** 2608.10256 | [PDF](https://arxiv.org/pdf/2608.10256v1)

**作者:** Alexander Schiøtz `[一作]` (Technical University of Denmark), Peder Heiselberg `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并评估了连续回归混合变压器（CRHT）用于基于AIS的船舶轨迹预测。

**💡 创新点**

创新点包括：① 在线K‑means聚类采样以缓解空间不平衡；② 结合1D卷积局部动力学提取与多头自注意力的混合编码器；③ 通过可学习查询的Transformer解码器实现并行多步回归；④ 采用缩放Delta学习和Huber损失提高回归稳定性。

**🔧 技术方法**

使用的技术包括：深度学习Transformer（Encoder‑Decoder）、1D卷积、正弦位置编码、MinMax归一化、Huber损失、AdamW优化器、在线K‑means聚类。

**📊 数据集**

使用丹麦海事局2016‑2025年7月‑10月的AIS数据集，约1.8亿条消息，提炼出41,458条航程（1,203艘大型船舶），并进行预处理与重采样。

**📈 对比分析**

与Kalman滤波器、TrAISformer等基线在1小时与2小时预测区间进行ADE/FDE对比，CRHT在1小时ADE/FDE分别为1.21 km/2.07 km，优于其他模型；在2小时ADE为2.55 km，FDE略高于TrAISformer，但整体误差更低。

**⚠️ 局限性**

局限性：对极端不确定或多路径情况仍易产生“幻觉”轨迹；训练需要大量GPU资源；聚类采样可能忽略细节动力学；模型在较长时间步（>1 h）上误差上升。

---

## 416. TCAM for Autonomous Deformable Manipulation: The RMC2 Champion System for WBCD 2026 Track 4

**arXiv ID:** 2608.10718 | [PDF](https://arxiv.org/pdf/2608.10718v1)

**作者:** Guangrui Shen `[一作]` (TermiTech), Qing Yu `[通讯]` (TermiTech)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

实现了T恤的自动拾取、搬运、对齐与抚平的全自主机器人系统。

**💡 创新点**

创新点在于将任务专用硬件（定制夹具）、腕部四视角视觉、UMI与现场实机数据的闭环收集与因果学习框架TCAM相结合，显著降低了机器人交互的物理复杂度。

**🔧 技术方法**

采用TCAM（TermiBrain Causal Action Model）多视角Vision‑Language‑Action（VLA）网络、动作块(chunk)输出、3D 打印夹具与腕部四相机视觉，形成端到端的自主控制管线。

**📊 数据集**

使用混合数据集：约200条UMI式演示 + 约400条实机演示，共计600条经过质量过滤的训练样本。

**📈 对比分析**

在WBCD 2026 Track 4 竞赛中完成25件T恤，22件满足表面平滑评分，平均每件耗时23秒，获得该赛道第一名。

**⚠️ 局限性**

局限包括：对极端失败的长时段恢复仍不可靠、早期终止判断依赖人工、对齐动作的数据不足导致平滑率波动、未实现完整的自动恢复决策机制。

---

## 417. Mapping Multimodal Pilot Stress and Fatigue During Flight Sessions

**arXiv ID:** 2608.09947 | [PDF](https://arxiv.org/pdf/2608.09947v1)

**作者:** Atandrila Chowdhury `[一作]` (Purdue University), Mark Wilson `[通讯]` (Purdue University)

**通讯引用:** 15627 | [OpenAlex ID](https://openalex.org/A5027597688)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对四名学生飞行员在真实飞行训练期间，连续记录心率、皮肤电反应、温度及加速度，并结合主观压力与疲劳量表，分析飞行过程中多模态生理与主观指标的变化。

**💡 创新点**

创新在于首次将多模态生理信号与自评量表同步采集，构建实时监测飞行员压力与疲劳的多维指标体系，揭示了心率与EDA、温度等信号在不同飞行阶段的特征。

**🔧 技术方法**

采用可穿戴多传感器设备、信号预处理与标准化、时序可视化、事件对齐与统计分析等技术，结合PSS-10与KSQ问卷。

**📊 数据集**

使用本研究收集的四名学生飞行员飞行期间的生理与问卷数据，未采用公开数据集。

**📈 对比分析**

主要通过描述性统计和可视化对比前后疲劳水平及各信号随时间变化，发现疲劳显著升高但主观压力变化有限，心率在高负荷阶段出现尖峰；但由于样本量极小，缺乏显著性检验。

**⚠️ 局限性**

局限主要为样本量极小（仅四名），难以推广；缺乏短期睡眠质量信息；未考虑飞行员经验差异及环境噪声等因素；未进行严格的统计检验。

---

## 418. Gaussian Sculpting: End-to-End Controllable Surface Reconstruction via Field Optimization

**arXiv ID:** 2608.10602 | [PDF](https://arxiv.org/pdf/2608.10602v1)

**作者:** Ke Jiaxin `[一作]` (Dalian University of Technology), Xiangjia He `[通讯]` (University of Nottingham Ningbo China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了一种全流程可微的高质量表面重建框架——Gaussian Sculpting，通过在SDF优化中引入受限高斯渲染实现端到端的几何重建。

**💡 创新点**

主要创新在于将高斯渲染与SDF优化耦合，构建几何、分布与透明度约束的高斯参数化，并采用双层梯度隔离优化与多分辨率八叉树细化，以显著减少漂浮点、缺失结构并提升网格完整度。

**🔧 技术方法**

使用了隐式SDF MLP、可微等值面提取、基于三角面分布的高斯参数化、梯度隔离的双层优化、octree式细分以及多尺度分辨率控制等技术。

**📊 数据集**

在 NeRF Synthetic 与 OmniObject3D 两大公开数据集上进行评估，使用多视角 RGB 图像与对应的 ground‑truth 几何。

**📈 对比分析**

与 NeRF、NeuS、2DGS、PGSR、GOF、GSDF 等基线对比，利用 Chamfer Distance 与网格质量指标（极角、最小角、sliver 比例）表明，在多数场景下，尤其是低分辨率下，Gaussian Sculpting 取得更低误差和更少漂浮点的表现。

**⚠️ 局限性**

局限性包括：双层优化导致训练时间长、GPU 显存占用高，且在高反射或细纹理场景下仍易出现渲染不稳定，难以完全适应大规模场景。

---

## 419. GenRec: An LLM-Backed Recommendation Ranker at Netflix

**arXiv ID:** 2608.10257 | [PDF](https://arxiv.org/pdf/2608.10257v1)

**作者:** Ying Li `[一作]` (Netflix), Ashish Rastogi `[通讯]` (Netflix)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在 Netflix 上设计并部署了 GenRec，一个基于大型语言模型（LLM）的推荐排名器，采用两阶段训练框架：第一阶段将开源 LLM 适配为 Netflix 专属基础模型，第二阶段在排名相关数据和奖励信号上进行后训练，结合上下文 verbalization、catalog‑aware 评分头以及预填充推理实现高效的全库排名。

**💡 创新点**

① 引入 LLM‑backed 推荐架构，利用预填充推理一次性为整个候选集生成排序；② 通过上下文工程与奖励加权训练实现数据效率和业务约束的双重对齐；③ 在成本受限环境下通过上下文压缩与模型规模折中，显著降低推理成本；④ 在大规模生产流量下验证离线与在线性能的显著提升，展示 LLM 对传统推荐系统的可行替代路径。

**🔧 技术方法**

基于内部 decoder‑only LLM（Transformer）+ vLLM 预填充推理；多任务联合损失（ranking、语言建模、奖励加权）；语义 ID 词表扩展；上下文 verbalization 与压缩；KV 缓存与 prefix caching；离线 MRR、在线 A/B 测试。

**📊 数据集**

Netflix 会员交互日志（数十亿条）用于 Phase‑1 训练；Phase‑2 使用约 1×~20× 的标注交互数据（包含观看、点赞等信号）作为 ranking 标签；对话式格式化的训练样本；10% 流量的在线 A/B 测试数据。

**📈 对比分析**

与成熟的生产 discriminative ranker 进行离线 MRR 和在线 A/B 对比；GenRec 在离线 MRR 上提升约 +1.6%（仅使用 40 倍更少的 Phase‑2 训练样本）；在 10% 流量、4 周的在线 A/B 中，短期与长期核心指标均实现统计显著提升（如核心指标 +0.006%）；实验还展示了数据量与模型规模对质量的正向影响，并在成本预算内定位 Pareto 前沿。

**⚠️ 局限性**

需要精细的上下文 verbalization 与奖励调校；对 Phase‑1 训练和基础 LLM 规模的依赖较高；目前仅在批处理表面验证，实时推荐仍需改进；候选集过大时可能需要采样 softmax；推理成本仍受模型尺寸和上下文长度限制，需进一步 distillation；奖励模型覆盖可能不足以完整表达所有业务目标。

---

## 420. Fast and Memory-Efficient Wavelet Convolutions via I/O-Aware Reformulation

**arXiv ID:** 2608.10805 | [PDF](https://arxiv.org/pdf/2608.10805v1)

**作者:** Amit Aflalo `[一作]` (Ben Gurion University of Negev), Oren Freifeld `[通讯]` (Ben Gurion University of Negev)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了WTConv的I/O感知融合实现，显著降低内存带宽需求和执行时间。

**💡 创新点**

通过三种代数重构：在芯片上重新计算Haar分析、将多级重构合并为单一位索引求和以及把通道尺度折叠进卷积权重，实现在不改变算子功能的前提下减少HBM流量约2.5×。

**🔧 技术方法**

使用CUDA实现的I/O优化策略、Haar变换重构、位索引闭式合成、权重与尺度融合等技术。

**📊 数据集**

在ConvNeXt-T、WTConvNeXt-T网络以及多种B×C×H×W张量尺寸上进行评估，未使用公开数据集，仅做微基准和网络吞吐量测试。

**📈 对比分析**

与原始WTConv和标准深度卷积比较，融合实现训练阶段加速3.7–4.3×、内存降低1.8–2.3×，并且在每个分解层级下优于7×7深度卷积；推理阶段加速约3.4×、内存降低约1.2×。

**⚠️ 局限性**

仅适用于Haar小波，无法直接推广到更长滤波器或复杂重构的波let族；性能受GPU带宽和调度差异影响。

---

## 421. Flex-$π$: A Multi-Stream World-Action Model with Compute Flexibility

**arXiv ID:** 2608.10860 | [PDF](https://arxiv.org/pdf/2608.10860v1)

**作者:** Ge Yan `[一作]` (University of Washington), Dieter Fox `[通讯]` (University of Washington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种6B参数的世界-动作模型，能够在仅使用RGB的情况下联合预测RGB、3D点图和DINO语义特征，并支持任何子集输入/输出的推理模式。

**💡 创新点**

创新点在于：①利用冻结的VAE将RGB和3D点图映射到同一潜在空间，实现无需额外传感器或预训练即可获得几何与语义监督；②引入视觉流随机丢弃与跨模态强制，使单一模型在不同输入/输出组合下保持性能；③使用Mixture-of-Transformers实现高效多流融合与动作生成。

**🔧 技术方法**

采用的技术包括：冻结视频生成VAE编码器、DINOv3语义编码、Mixture-of-Transformers骨干、流匹配（flow‑matching）训练、视觉流随机丢弃、跨模态强制、动作专家与跨流注意力。

**📊 数据集**

使用的数据集：AGIBOT World‑Beta（≈500小时、100任务）预训练；RoboTwin仿真（50任务）；LIBERO与LIBERO‑Plus；真实双臂YAM机器人五个精细操控任务。

**📈 对比分析**

与VLA和WAM基线对比，在RoboTwin、LIBERO、LIBERO‑Plus以及真实机器人任务上，成功率提升2–7倍；action‑only模式速度快于π_0.5且成功率更高；在有限演示和OOD情形下保持优势；多流联合训练显著提升数据效率。

**⚠️ 局限性**

局限性包括：需要至少10个epoch微调才能收敛；多流联合生成的推理速度仍慢于同参数VLA；对更强语义推理能力和更多机器人数据的依赖；训练时的视觉流丢弃导致收敛更慢。

---

## 422. MoE Proxy Models for Low-Cost Failure Reproduction and Diagnosis in LLM RL Post-Training

**arXiv ID:** 2608.10823 | [PDF](https://arxiv.org/pdf/2608.10823v1)

**作者:** Yikai Wang `[一作]` (Nanjing University), Wangze Zhang `[通讯]` (Huawei)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在强化学习后训练（RL post‑training）中，针对 Mixture‑of‑Experts（MoE）大型语言模型（LLM）构建轻量级代理模型，用于低成本的故障重现和辅助诊断。

**💡 创新点**

创新点在于提出多视角、频率感知的专家剪枝方法：同时考虑路由器参数、专家共激活行为和路由上下文表示三种相似性视角，通过 K‑Medoids 聚类选择代表性专家，既保持原模型的骨干架构、Top‑k 路由机制，又能保留导致故障的关键动态特征；且不需要参数平均或额外微调。

**🔧 技术方法**

技术包括 MoE 模型的专家相似性度量（路由器参数距离、共激活距离、路由上下文原型距离）、距离融合与归一化、K‑Medoids 聚类、代理模型重构；实验使用 Huawei Ascend 910 NPU、VERL、vLLM、GRPO 训练框架；实现对 Qwen3‑30B‑A3B、DeepSeek‑V3.2 等模型的代理构建。

**📊 数据集**

主要数据集：GSM8K（任务能力评估与代理验证）、专门收集的校准集（统计专家激活频率、路由权重、隐藏状态），以及在 RL 环境中收集的轨迹与奖励数据。

**📈 对比分析**

与基线（仅按频率选取专家、随机选择、固定前 K、专家合并等）相比，代理模型在保留任务性能时将硬件需求下降 50%–87.5%，每步 NPU‑hour 成本可降低至原来的 1/33.3；训练动态（奖励、KL 损失）与原模型保持一致，并在两种典型故障（数值一致性故障与优化稳定性故障）下重现相同的异常方向和时间趋势。

**⚠️ 局限性**

局限性：对极端压缩（仅保留极少专家）时仍可能出现任务性能显著下降；代理模型的故障重现能力取决于所选故障类型，未能覆盖所有可能的 RL 后训练错误；此外，方法依赖于足够的校准数据来估计专家相似性，对资源受限的实验环境可能受限。

---

## 423. Reference-Free Post-Training of Open Large Language Models for Multilingual Machine Translation

**arXiv ID:** 2608.10812 | [PDF](https://arxiv.org/pdf/2608.10812v1)

**作者:** Chris Han `[一作]` (Xiaomi Inc.), Jian Luan `[通讯]` (Xiaomi Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文基于MiLMMT-46-v0.1模型，采用参考无监督强化学习与SFT‑RL插值，构建了MiLMMT-46-v1.0，提升了多语言翻译质量。

**💡 创新点**

创新点在于将语言识别门控的参考无监督质量估计奖励与Group Relative Policy Optimization相结合，并通过线性插值在RL与SFT之间取得可控的质量平衡；同时对比了OPD的可行性。

**🔧 技术方法**

主要技术包括GRPO、参考无监督质量估计（XCOMET与COMETKiwi）、语言识别门控、SFT‑RL权重插值以及对比的on‑policy distillation。

**📊 数据集**

使用了覆盖46种语言的MiLMMT SFT数据，以及WMT24++和FLORES+评测数据集。

**📈 对比分析**

在WMT24++和FLORES+上，MiLMMT-46-v1.0在XCOMET、COMETKiwi等参考无监督指标上超过了多家开源和商业系统（如Google Translate、Gemini 3 Pro、GPT‑5），spBLEU略有下降但整体质量指标显著提升。

**⚠️ 局限性**

局限性包括RL训练导致的spBLEU下降、OPD无法突破RL+插值的性能上限、对极低资源语言提升有限，以及奖励模型可能存在的偏差。

---

## 424. Evaluating Semantic and Spatial Guidance for Foundation Model Segmentation of Small-Scale PV in Remote Sensing Imagery

**arXiv ID:** 2608.10801 | [PDF](https://arxiv.org/pdf/2608.10801v1)

**作者:** Roni Blushtein-Livnon `[一作]` (Ben-Gurion University of Negev), Emir Galilee `[通讯]` (Ben Gurion Institute for the Study of Israel & Zionism)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估并系统比较了不同提示策略（文本、几何、混合）对SAM3在遥感小型光伏板分割任务中的表现，探讨训练规模、转移学习、空间分辨率和跨图像变化的影响。

**💡 创新点**

首次对遥感领域中的提示方式进行细粒度比较，证明混合提示在小目标分割中既能提高精度又增强鲁棒性，并揭示提示策略是决定模型适配、泛化和数据效率的关键因素。

**🔧 技术方法**

使用Vision‑Language基础模型SAM3的提示式分割，结合文本提示、边界框提示及其组合，采用有限标注的微调、逐图转移学习和多分辨率评估。

**📊 数据集**

主要在以色列北Negev多时相航空影像（0.25/0.145/0.132 m/pixel）进行实验，并在法国Google Earth/IGN以及纽约Queens的公开光伏数据集上验证跨数据集一致性。

**📈 对比分析**

通过F1、IoU、Precision、Recall等指标，对不同提示、训练规模、转移学习和分辨率的组合进行系统实验；结果显示混合提示在300样本即可达到F1≈0.90，文本提示最差，几何提示中等，混合提示最佳且对分辨率、光照等变化最不敏感。

**⚠️ 局限性**

仅针对单一目标类别（小型光伏板）和单一基础模型（SAM3）进行评估，几何提示使用的是理想化的真值框，未测试检测生成框的效果，且未使用校准的辐射数据来区分光照和传感器差异。

---

## 425. E$^3$mo-Bench: A Scalable Benchmark for Multimodal Evoked and Expressed Emotion Understanding via Bayesian Pairwise Alignment

**arXiv ID:** 2608.10796 | [PDF](https://arxiv.org/pdf/2608.10796v1)

**作者:** Lancheng Gao `[一作]` (Shanghai Jiao Tong University), Xiongkuo Min `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 E^3mo-Bench 基准，覆盖诱发与表情情绪的感知、开放词识别和 VAD 评估，并设计了可扩展的 Bayesian Pairwise Alignment（BPA）标注流程与 E^3mo-Score 评估代理。

**💡 创新点**

创新点包括：①把诱发与表情情绪统一到同一基准；②利用贝叶斯相对比较将稀疏对比转化为连续 VAD 分数；③构建无训练需求的多模型委员会评估器。

**🔧 技术方法**

技术手段涉及跨模态反射标注（CMRA）、贝叶斯最大后验 MAP 优化、TrueSkill 对比调度、基于 GPT 系列 LLM 的文本推理、视觉/音频特征提取与融合，以及五模型委员会投票。

**📊 数据集**

数据集来源于 8 个公开音频-视觉情绪数据集（DFEW、MELD、CAER、MAFW、LIRIS-ACCEDE、VideoEmotion、MediaEval、VGGSound），共构建 12,314 个 QA 对。

**📈 对比分析**

通过在感知、识别、评估三任务上与 14 款开源/专有 MLLM 进行对比，E^3mo-Score 在 VAD 评估中普遍高于单模型，整体表现虽优于随机但仍显著低于人类水平，并揭示了诱发与表情情绪性能不均。

**⚠️ 局限性**

局限性主要在于：当前 MLLM 对诱发情绪感知和视频对比任务表现弱；BPA 仍受限于标注者一致性；E^3mo-Score 受限于委员会模型的表达能力；基准缺乏跨文化、多语种的情绪细节。

---

## 426. Dual Stress: Runtime Safety Monitoring for Safety-Constrained MPC Navigation

**arXiv ID:** 2608.10791 | [PDF](https://arxiv.org/pdf/2608.10791v1)

**作者:** Jamil Chahine `[一作]` (American University of Beirut), Anthony Tzes `[通讯]` (New York University Abu Dhabi)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用模型预测控制（CBF–MPC）的双重变量（dual stress）作为运行时碰撞风险监测信号，评估其与传统几何检测器的互补性，并在物理仿真环境下验证其可操作性。

**💡 创新点**

创新点在于：①将MPC求解得到的KKT乘子视为“安全压力”信号，用于实时监测；②证明该信号能捕捉到控制模型与真实车辆差异导致的接近难度；③在同一误报预算下展示其对碰撞预警的显著提升。

**🔧 技术方法**

技术手段包括：CBF–MPC与内部点优化器（IPOPT）求解；NVIDIA Isaac Sim 物理仿真；15种基线几何检测器（距离、速度、时间到碰撞、DRAC 等）；预注册实验设计与 Wilson 区间统计；碰撞可预防性（braking rescue）测试。

**📊 数据集**

数据集：2000个预注册的六车交叉场景（500训练、2000测试、100 pilot），以及脚本化的高速公路切入、变道、交叉路口单冲突场景。

**📈 对比分析**

比较方法：在相同 10% 误报预算下，dual stress 单独阈值与 15 个几何检测器的联合警报进行对比。结果显示：dual stress 单独警报 4.25% 的场景，覆盖率 69% 的可制止碰撞；与几何检测器的覆盖率 47% 相比显著提升；两者联合可达 75% 覆盖率。11 种扰动实验中，阈值冻结后性能保持稳定，传感器噪声严重时几何检测器补充性增强。碰撞救援率达 96.5%。

**⚠️ 局限性**

局限性：仅在 1/10 规模物理仿真车辆、常速障碍、六车交叉场景下验证；不涉及实时感知误差、可变动作障碍或全尺寸车辆；未做匹配的行动比较；union 阈值未单独调优；对重度传感器失真敏感；未来需在真实硬件上进一步验证。

---

## 427. JEPA-WAM: Stage-Level Joint-Embedding Prediction for World-Action Models in Robot Manipulation

**arXiv ID:** 2608.10780 | [PDF](https://arxiv.org/pdf/2608.10780v1)

**作者:** Xiao Liu `[一作]` (Tsinghua University), Yan Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 JEPA‑WAM，一种将阶段级语义未来（Stage‑JEPA）与短期世界动作模型（Motus WAM）相结合的通用机器人策略；

**💡 创新点**

创新点在于：① 将机器人未来分为短期物理未来与阶段级语义未来；② 使用目标条件的 Joint‑Embedding Predictive Architecture (JEPA) 在表示空间预测下一个任务阶段的潜在表示；③ 通过门控接口将该潜在表示注入 WAM 的视频标记，使得局部动作生成能够提前获得任务进度指引；

**🔧 技术方法**

采用的技术包括：冻结的 V‑JEPA2 编码器、训练的 Stage‑JEPA 预测器、Motus WAM（Mixture‑of‑Transformers）、Qwen3‑VL 视觉‑语言编码器、三模态联合注意力、门控正则化等；

**📊 数据集**

在 RoboTwin 2.0 仿真平台上进行评估，涵盖 50 种双臂操作任务，分别在干净与随机环境下测试；

**📈 对比分析**

与 GO‑1、π_0.5、X‑VLA、Motus 等基线对比，JEPA‑WAM 在干净环境下取得 91.42% 成功率，随机环境 89.08%，在多数语义任务类别上均超过 Motus；消融实验表明 Stage‑JEPA 预测与其在 WAM 中的注入是提升性能的关键；

**⚠️ 局限性**

局限性包括：对推理延迟敏感，需进一步降低 WAM 的推理时间；对仅依赖即时抓取或接触稳定性的任务（如获取与举升）阶段指引作用有限；目前仅使用单尺度潜在表示，缺乏更细粒度的多尺度任务进度编码。

---

## 428. Rule of Thumb: Explaining Artificial Intelligence Systems using Partial Information

**arXiv ID:** 2608.10766 | [PDF](https://arxiv.org/pdf/2608.10766v1)

**作者:** Kaivalya Rawal `[一作]` (University of Oxford), Chris Russell `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于特征可预测性而非敏感性的可解释方法——Rule of Thumb（RoT），并在零射线分类、黑盒AI审计和科学发现等新兴XAI场景中进行验证。

**💡 创新点**

创新点在于：1) 通过训练可加性函数直接估计AI预测，避免对输入进行扰动或访问模型内部；2) 只需一次全局训练即可为任意子集特征生成解释；3) 支持使用非输入特征（如种族、性别）进行解释；4) 兼容大语言模型API且计算成本极低。

**🔧 技术方法**

使用的技术包括：基于Dropout的加性模型学习、sigmoid/恒等输出映射、对特征值的可加重要性分数推导；对比实验中使用SHAP、LIME、Integrated Gradients和梯度方法；可视化采用force‑plot、swarm‑plot、词云等。

**📊 数据集**

实验数据集包括：OpenAI GPT‑4o‑mini零射线分类（猫/狗图像、电影评论、司法判例）、GPT‑4.1‑nano API简历筛选、Pima印度人糖尿病数据、Amazon电商推荐、犯罪司法与信贷数据。

**📈 对比分析**

与SHAP、LIME和梯度方法相比，RoT在解释速度上提升数万倍（例如对司法判例的RoT解释平均<0.1ms，SHAP需千秒），且在零射线分类、审计和科学假设生成上与人类标注的相关性更高（AUROC≈0.77–0.92），对不相关特征（如年龄）不产生误导，且对对抗性“假特征”具有更高的鲁棒性。

**⚠️ 局限性**

局限性：1) 仅考虑单个特征的可预测性，可能忽略特征交互或复杂因果关系；2) 需要对每个特征训练独立函数，参数量随特征维度增长；3) 对于极大规模特征空间，训练与存储成本仍有提升空间；4) 由于不基于模型梯度，无法提供对模型内部机制的深入洞察。

---

## 429. Capacity regimes for Boolean function computation via channels

**arXiv ID:** 2608.10816 | [PDF](https://arxiv.org/pdf/2608.10816v1)

**作者:** Jingge Zhu `[一作]` (University of Melbourne), Matthias Frey `[通讯]` (University of Melbourne)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在点对点通信系统中，如何通过信道计算布尔函数的问题，提出了计算能力的概念，并为一类布尔函数提供了可实现性和对偶结果。

**💡 创新点**

创新点在于对布尔函数计算问题的规模关系进行了全面的表征，提供了计算能力的上下界，并揭示了不同哈明权重的函数类对消息长度的影响。

**🔧 技术方法**

使用了信息论中的信道容量理论，结合哈明权重的概念，推导出不同情况下的计算能力。

**📊 数据集**

未具体提及使用的数据集，但讨论了布尔函数的哈明权重和相关的函数类。

**📈 对比分析**

通过与经典的香农传输问题进行比较，展示了在小哈明权重情况下，消息长度可以指数级增长，而在大哈明权重情况下，消息长度则线性增长。性能上，计算能力的上下界相差不超过2。

**⚠️ 局限性**

限制在于只考虑了单一布尔函数的计算，未来的工作可以扩展到多布尔函数的计算问题。

---

## 430. ReLTEx: Reliable LLM-based Taxonomy Expansion

**arXiv ID:** 2608.10970 | [PDF](https://arxiv.org/pdf/2608.10970v1)

**作者:** Zeinab Ghamlouch `[一作]` (Institut Polytechnique de Paris), Mehwish Alam `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

ReLTEx框架在LLM生成候选概念的基础上，加入结构感知验证和递归扩展控制，实现了可靠的自动分类拓展。

**💡 创新点**

创新点在于将开放式LLM生成与路径感知结构验证相结合，利用分类器过滤不一致关系并通过递归停止机制抑制错误传播。

**🔧 技术方法**

技术主要包括LLM（如Mistral、Llama3.2等）生成候选子概念、基于DistilRoBERTa的二分类结构验证器，以及基于置信度的递归停止策略。

**📊 数据集**

使用SemEval‑2016环境词典和Schema.org两大公开分类本体进行评测。

**📈 对比分析**

通过遮蔽扩展基准、人工评估和LLM评估三种方式对比，ReLTEx在R@K、MRR、WuP等指标上均优于现有方法，尤其在Schema.org上表现突出。

**⚠️ 局限性**

局限在于依赖中等规模LLM，分类器可能误判关系，且缺乏标准化的递归生成基准，导致系统级比较受限。

---

## 431. ClusterBench: A Framework for Cluster-Wide Continuous Benchmarking and Regression Testing

**arXiv ID:** 2608.10956 | [PDF](https://arxiv.org/pdf/2608.10956v1)

**作者:** Aditya Ujeniya `[一作]` (Friedrich-Alexander-University), Gerhard Wellein `[通讯]` (Friedrich-Alexander-University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了ClusterBench框架，能够在整个集群上持续、组件级别执行基准测试并记录性能与功耗、频率、温度等指标。

**💡 创新点**

创新点在于集群感知调度、组件级别连续基准、内置指标采集、双维度（空间+时间）统计与变异分析。

**🔧 技术方法**

使用Go实现，结合Slurm调度，集成Perl指标采集脚本、DuckDB、SQLite、MetricCollector，支持JSON定义文件。

**📊 数据集**

使用NHR@FAU三大集群（Helma、Alex、Fritz）硬件，执行HPL、DGEMM、TheBandwidthBenchmark、OSU微基准、fio等基准，收集功率、频率、温度等数据。

**📈 对比分析**

通过对同一节点多次测量与跨节点分布比较，发现单节点误差<1%，跨节点误差≤5%；对性能与频率、功耗、温度做相关性分析，揭示不同冷却方式下的关系。

**⚠️ 局限性**

局限在于不支持编译阶段、参数空间扫描、仅支持Slurm、只做健康检查与回归，不涉及全局性能调优或复杂多节点作业。

---

## 432. Understanding the Architecture of Coding Agents: An Exploratory Study Using a Research Prototype

**arXiv ID:** 2608.10934 | [PDF](https://arxiv.org/pdf/2608.10934v1)

**作者:** Marco Tulio Valente `[一作]` `[通讯]` (Federal University of Minas Gerais), Marco Tulio Valente (Federal University of Minas Gerais)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统化描述编码代理的主要架构，并实现了简洁的开源代理 Ark，同时设计了 ArkBench 轻量级基准集

**💡 创新点**

提出最小化的参考实现和对应的轻量级基准，并通过架构分类法对比主流代理，展示了简化设计的可行性

**🔧 技术方法**

利用 ReAct 循环、LLM 交互、工具调用、内存/上下文管理、错误处理、安全隔离、追踪等技术，构成单模型的完整代理框架

**📊 数据集**

使用 ArkBench，其中包含10个软件维护与演化任务（3个 bug 修复、3个重构、4个功能实现）及其单元/隐藏测试

**📈 对比分析**

在 gpt‑5.4‑mini 上评估，Ark 在 ArkBench 上完成 80% 任务，平均 3,342 tokens、23.5 秒，成本 <0.5 美元；与 Codex CLI、OpenCode 对比，功能更简洁但同样保持核心架构

**⚠️ 局限性**

局限性包括工具集有限、无持久化记忆、缺乏高级搜索、对复杂重构和依赖更新支持不足

---

## 433. Certify or Refuse: A Cross-Model Map for Selective Risk Control with Coverage Floors under Covariate Shift

**arXiv ID:** 2608.10893 | [PDF](https://arxiv.org/pdf/2608.10893v1)

**作者:** Jiamiao Liu `[一作]` (Army Medical University), Xuetao Chen `[通讯]` (Army Medical University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了在有界比率协变量移位下的“Floor Certification Map”，给出在保证至少β比例流量被自动回答且风险不超过α的选择性问答系统的可行性边界及样本复杂度分解。

**💡 创新点**

创新点在于：①首次给出以覆盖下限β为索引的风险下限；②将样本成本拆分为标记源样本和未标记目标样本的双资源映射；③证明了低阶下限与上限跨模型匹配；④对估计权重（nuisance）成本进行显式定价。

**🔧 技术方法**

使用了重要性加权、线性化风险检验、经验-Bernstein UCB、受限分区的比例估计、风险与覆盖阈值网格化、LR_margin边界以及全局非参数检验。

**📊 数据集**

实验基于注册的合成生成器（多种结构）和真实问答工作负载SQuAD→NewsQA（共4212个评估条目）。

**📈 对比分析**

与无覆盖下限、加权一致性、插件基线等方法对比，正式保证的门控算法在网格审计中0违规，而其他方法在规模上出现多次Holm拒绝；在真实工作负载下，证书提前拒绝。

**⚠️ 局限性**

局限包括：①需要预注册的分区模型和θ-可测性假设；②对未知η的nuisance成本仍未完全证明必要；③实验验证仅覆盖合成族1（及部分族2），缺乏对更广泛场景的普适性；④相对oracle权重的样本开销显著增大。

---

## 434. From Pattern Detection to Composition Analysis in Quantum Software

**arXiv ID:** 2608.10882 | [PDF](https://arxiv.org/pdf/2608.10882v1)

**作者:** Neilson Carlos Leite Ramalho `[一作]` (University of Sã Paulo), Marcos Lordello Chaim `[通讯]` (University of Sã Paulo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建量子模式知识库并改进 qpa 检测管线，生成组件调用组合图，评估量子模式在80个开源项目中的实际采用情况。

**💡 创新点**

引入词汇扩展和四通道语义匹配提升召回率；通过 LLM 集成实现半自动的知识库维护；利用组合图揭示直接/间接调用、共现关系与实现粒度，为模式采用分析提供新的可视化视角。

**🔧 技术方法**

使用 all‑mpnet‑base‑v2 句子嵌入进行名称/摘要/标题/模式描述四通道匹配；LLM 集成（Llama‑3.3、Qwen‑2.5、DeepSeek）进行模式分类；静态 AST 分析提取函数调用；Neo4j 图数据库存储并查询调用图；两阶段词汇扩展处理框架自有命名。

**📊 数据集**

基于 PlanQK Atlas 的 61 个模式；从 Qiskit、PennyLane、Classiq、Qiskit Algorithms、Qiskit Machine Learning 五个框架抽取 286 个组件；80 个精选 Python 项目共 904 个脚本；Qrisp 教程笔记本用于检测准确性评估；全部数据已发布在 qpa 复制包中。

**📈 对比分析**

与基线（无词汇扩展）对比，微 F1 从 0.449 提升到 0.712；与 LLM 集成对比，微 F1 0.704，宏 F1 0.632；检测覆盖 611 次，23 个模式全部出现；组合图揭示大部分模式是通过内部调用间接引入，直接调用比例与框架差异明显；实验显示不同框架在实现粒度与调用层次上存在显著差异。

**⚠️ 局限性**

知识库仅覆盖 5 个框架，难以覆盖新框架或自定义命名；词汇扩展在跨框架时仍有召回不足；仅分析 Python 与 Jupyter 笔记本，未覆盖其他语言或大规模生产代码；LLM 需要付费，影响可复现性；组合图受静态分析限制，无法捕获动态构造或隐式调用。

---

## 435. Robust Safety Filtering for Input-Constrained Underactuated Linear Systems

**arXiv ID:** 2608.10872 | [PDF](https://arxiv.org/pdf/2608.10872v1)

**作者:** Muhamad Rausyan Fikri `[一作]` `[通讯]` (Tampere University), Muhamad Rausyan Fikri (Tampere University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于 H∞ 基线输入、扰动观测器和高阶控制障碍函数的鲁棒安全滤波框架；

**💡 创新点**

创新点在于将 H∞ 差分游戏的基线控制与实时扰动估计结合，用观测器证书的误差边界构造鲁棒 HOCBF，提供精确的可行性间隔和有限时域性能平衡；

**🔧 技术方法**

采用 H∞ 差分游戏求解、扰动观测器、鲁棒高阶控制障碍函数（HOCBF）、QP 安全滤波和线性系统模型；

**📊 数据集**

使用二维两轮平衡机器人（TWBR）的线性化模型；

**📈 对比分析**

与 LQR、基线 H∞、扰动补偿（OI）控制器对比，仿真显示在临时阶跃扰动下唯一满足位置与俯仰两项安全约束，且扭矩峰值低于其他方法；

**⚠️ 局限性**

局限在于仅验证线性化模型，未考虑非线性重排误差，且需事先知道扰动率上界及完整状态观测。

---

## 436. XCoT-VLA: Executable Chain-of-Thought for Vision-Language-Action Driving

**arXiv ID:** 2608.10976 | [PDF](https://arxiv.org/pdf/2608.10976v1)

**作者:** Foundation Model Team `[一作]` (XPeng Inc), XPeng Inc `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了 XCoT‑VLA 框架，用可执行的 2–6 个语义动作令牌替代冗长的自然语言 Chain‑of‑Thought，直接将多模态感知映射到连续轨迹生成；

**💡 创新点**

①提出可执行 Chain‑of‑Thought (XCoT) 令牌化，压缩为 2–6 个可执行动作令牌；②在 Vision‑Language‑Action 模型中采用两分支 FFN（Reason 与 Control）并通过共享自注意力实现推理与轨迹生成解耦；③在同一可执行空间中引入 XCoT 策略优化 (XCPO)；

**🔧 技术方法**

视觉‑语言融合、可执行令牌化、共享自注意力、分支 FFN、流匹配轨迹生成、基于离线 Reason–Action 监督的训练、可选的奖励驱动 XCPO；

**📊 数据集**

约 3.6 M 个混合样本：3.1 M 自动 Reason–Action 标注、200 k 人工注解、320 k 车道变更专用规则式标注；

**📈 对比分析**

与轨迹仅监督、自然语言 CoT、隐式语义令牌三种基线对比，评估开放式规划 ADE/FDE。XCoT‑VLA 在通用集 ADE‑6s‑Long 1.645→1.323、FDE‑6s‑Long 4.354→3.089；车道变更集 ADE‑6s‑Lat 0.594→0.309、FDE‑6s‑Lat 1.616→0.648，显著优于基线；推理延迟低于 83.3 ms 预算；

**⚠️ 局限性**

仍缺乏与周围车辆的交互式谈判能力，受交通规则感知误差影响，对纵向舒适度提升有限，且 XCPO 在本文中未进行闭环性能量化。

---

## 437. MUSE: A Full-Text Cross-Domain Knowledge Base of Scientific Problems, Solutions, and Rationales

**arXiv ID:** 2608.10974 | [PDF](https://arxiv.org/pdf/2608.10974v1)

**作者:** Tsofia Cohen `[一作]` (Hebrew University of Jerusalem), Tom Hope `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了全文本、跨领域的科学问题-解决-理由（P–S–R）三元组资源MUSE，并公开了约36,960条来源可追溯的三元组；

**💡 创新点**

创新点在于对科研论文中细粒度技术问题与对应解决方案及其理由进行结构化标注，突破了传统仅关注论文级贡献或修辞角色的限制；

**🔧 技术方法**

采用了分层的提取管线，包括段落筛选、边界标签化提取、关系抽取以及LLM后处理（GPT‑4o + Claude Opus），并通过多种Transformer（DeBERTa‑v3、Mistral‑7B等）实现；

**📊 数据集**

数据集为579段落的专家标注种子集，用于训练和评估抽取模型，并在整个arXiv全文语料上扩展生成KB；

**📈 对比分析**

与单一端到端LLM抽取相比，分阶段管线在标注一致性、抽取精度和理由完整性上显著提升；在LLM推理实验中，理由监督在多约束复杂问题上提升了解答质量，且在简单问题上可能产生过度推理；

**⚠️ 局限性**

局限性包括：对医学、社会科学或人文文献的适用性尚未验证；实验规模受限于算力，无法证明理由监督对所有模型和任务的普适性；

---

## 438. GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care

**arXiv ID:** 2608.10969 | [PDF](https://arxiv.org/pdf/2608.10969v1)

**作者:** Ruirui Wang `[一作]` (University of Zürich), Diego Paez-Granados `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于图注意力的ICU多变量时间序列模型 GARLIC，能够在存在缺失和不规则采样的情况下实现精准预测并提供内置可解释性。

**💡 创新点**

创新点包括：1) 学习式指数衰减编码器实现缺失值自适应插补；2) 时滞图结构学习捕获信号间依赖；3) 跨维序列注意力融合全局信息；4) 交替解耦优化稳定训练；5) 所有注意力权重与图边可直接作为解释。

**🔧 技术方法**

使用技术包括指数衰减插补、时间窗口注意力、可学习图消息传递、GRU+自注意力、ℓ1 正则稀疏图、交替解耦多任务学习。

**📊 数据集**

在 MIMIC‑III、PhysioNet 2012 与 2019 三大ICU基准数据集上进行实验，并在多种时间序列任务中验证泛化能力。

**📈 对比分析**

与多种不规则时间序列模型（RNN, GRU‑D, ODE‑RNN 等）和可解释模型（RETAIN, IMV‑LSTM 等）对比，GARLIC 在 AUROC 与 AUPRC 上实现最高分，尤其在缺失率高的 P12 数据集上显著优于最优基线。

**⚠️ 局限性**

局限性包括：未验证预测任务、固定滑动窗口限制灵活性、未加入静态患者特征、对极端类别不平衡处理不足、无法实时流式处理长时间序列。

---

## 439. Multiple Scale Latents for Learned Image Compression

**arXiv ID:** 2608.10952 | [PDF](https://arxiv.org/pdf/2608.10952v1)

**作者:** Jonas Brenig `[一作]` (University of Würzburg), Radu Timofte `[通讯]` (University of Würzburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多尺度层次潜在表示的学习型图像压缩模型，通过在不同尺度下使用多个潜在变量来更好地捕捉图像的空间结构。

**💡 创新点**

创新点在于直接将多尺度潜在表示与现有的熵模型结合，并使用端到端训练方式；相比以往的粗细层次或单尺度模型，显著提升了压缩效率。

**🔧 技术方法**

采用卷积自编码器、残差块、字典学习的通道组自回归熵模型，以及直通估计器和随机 Gumbel 退火进行潜在优化。

**📊 数据集**

使用100k张512×512分辨率的图像进行训练，评估数据集包括 Kodak、CLIC 2020 Professional 和 Tecnick。

**📈 对比分析**

与多种最新学习型压缩方法以及传统 VVC 编码器对比，KD0基准上实现了-17.9% BD-rate 的提升，CLIC 和 Tecnick 上也分别取得约-9.6% 和-24.1%的改进。

**⚠️ 局限性**

主要局限在于多尺度架构导致解码时计算复杂度和参数量增加，解码速度略慢；同时仅在单一路径上使用相同的熵模型，未针对不同尺度探索专用熵模型。

---

## 440. StreamFlow: Dynamic Memory Flows for Streaming Video Understanding

**arXiv ID:** 2608.10949 | [PDF](https://arxiv.org/pdf/2608.10949v1)

**作者:** Muxin Fu `[一作]` (Tongji University), Bo An `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种高效的视觉记忆框架StreamFlow，能够在实时视频流中动态、按需地访问历史视觉信息，解决了传统模型和记忆方法在编码冗余与硬编码访问上的局限。

**💡 创新点**

创新点在于：① 引入动态感知的中期记忆（只对帧间差异高的补丁进行编码），显著减少不必要的视觉编码；② 构建潜在长时记忆，以固定容量的视觉潜在形式压缩并保留历史信息；③ 采用视觉注意力得分（VAS）驱动的记忆注入机制，在生成过程中监测视觉依赖并按需注入相关潜在，缓解注意力漂移。

**🔧 技术方法**

技术包括基于像素差异的补丁残差评分与稀疏选择、GOP级视觉潜在编码与相似性驱动合并、注意力得分监测与检索压缩模块（由LoRA微调的视觉压缩器实现），以及Frozen MLLM（如Qwen3.5-9B）作为语言视觉后端。

**📊 数据集**

在训练中使用COIN、NeXT-QA、STAR、CLEVRER、LLaVA-Video-178K等问答式视频数据集；在评估时，采用StreamingBench（流式视频），以及MVBench、MLVU、VideoMME等离线长视频基准。

**📈 对比分析**

与现有模型化和记忆化方法（如LiveVLM、StreamMem、VideoLLM-Online、TimeChat-Online等）相比，StreamFlow在StreamingBench上取得67.73%准确率，领先最强基线4.63%；在MLVU、VideoMME等离线基准上也取得最佳或接近最佳成绩，同时在视觉注意力得分上提升59.1%，并将端到端延迟降低50.4%、峰值内存降低21.1%。

**⚠️ 局限性**

局限性主要在于：① 仍需额外的稀疏补丁选取与潜在压缩开销，可能影响极低延迟场景；② 长期记忆压缩可能导致细粒度信息丢失，对极长时序推理的细节把握有限；③ 目前仅在固定摄像头视角与有限场景下验证，尚未评估在高度动态或多摄像头混合流中的鲁棒性。

---

## 441. A Cost-Efficient Routing Pipeline for Multilingual Short-Text Classification Using Small Language Models

**arXiv ID:** 2608.10939 | [PDF](https://arxiv.org/pdf/2608.10939v1)

**作者:** Wajdi Ben Saad `[一作]` (Carthago Labs), Safa Madiouni `[通讯]` (Universite Paris Dauphine Psl)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了一种固定列表路由策略，用于在多语言短文本分类中根据语言资源水平决定是否使用翻译加零射模型；

**💡 创新点**

创新点在于将路由决策与语言资源层级相结合，并在不进行任务特定微调的情况下，仅通过翻译与多语言编码器的组合实现性能提升；

**🔧 技术方法**

主要技术包括预训练压缩句子编码器（paraphrase-multilingual-MiniLM-L12-v2 与 paraphrase-MiniLM-L6-v2）、OPUS‑MT/NLLB 翻译后端、以及基于相似度的原型匹配分类；

**📊 数据集**

使用了两大公开数据集：SIB‑200 的七分类主题子集（15 语言）和 MASSIVE 的意图子集（15 语言、60 意图）；

**📈 对比分析**

通过对四种路由配置（R0‑R3）进行宏观 F1 与延迟的对比，发现 SIB‑200 在仅翻译低资源层（R1）时实现最高整体宏 F1（0.7403），而 MASSIVE 在全部翻译（R3）时获得最佳宏 F1（0.4647），且低资源层翻译均显著提升局部 F1；

**⚠️ 局限性**

局限性包括仅评估两个数据集、路由边界为手工设定、未对翻译质量进行细粒度分析、以及仅在单一本地硬件上测得延迟，无法直接推广到所有生产环境。

---

## 442. Temporally Grounded Compositional Camera Motion Understanding via Geometric Knowledge Distillation

**arXiv ID:** 2608.10932 | [PDF](https://arxiv.org/pdf/2608.10932v1)

**作者:** Dazhao Du `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个面向视频的时序分块、多标签相机运动识别任务，并构建了Cammotion基准数据集；

**💡 创新点**

创新点在于将相机运动建模为时序分块和多重运动共存的组合识别问题，并通过几何蒸馏技术将3D基础模型的几何信息注入多模态大语言模型的轻量级相机令牌中，实现无额外推理成本的几何增强；

**🔧 技术方法**

使用了3D基础模型VGGT-Ω做教师，设计了Geometry-aware Camera Token Extractor (CamDistill) 进行蒸馏；同时结合多模态大语言模型(Qwen3-VL、Qwen2.5-VL等)和多标签推理框架；

**📊 数据集**

使用自建的Cammotion数据集：4229个单帧剪辑、8591段、14258个运动实例，覆盖20个方向感知标签、12种运动类型，来自YouTube与腾讯视频；

**📈 对比分析**

在Cammotion上与多款公开和闭源模型比较，CamDistill + 4B模型在帧级micro F1上达到73.5，远超Gemini-3.1-Pro（43.3）和基础Qwen3-VL-4B（24.2）；同时在外部Benchmarks（CameraBench、CMVQA）上显著提升，显示跨任务迁移能力；

**⚠️ 局限性**

局限性包括：数据仅为单镜头视频，缺少剪辑式多镜头场景；对极少见标签的识别仍有欠缺；蒸馏过程依赖VGGT-Ω教师，若教师质量不足会影响效果；

---

## 443. FedCGR: Federated Cross-Domain Generative Recommendation

**arXiv ID:** 2608.10929 | [PDF](https://arxiv.org/pdf/2608.10929v1)

**作者:** Zhuodong Liu `[一作]` (Beijing Jiaotong University), Peiyu Hu `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在联邦学习环境下的跨域生成式推荐框架FedCGR，将跨域对齐转化为对共享语义ID（SID）序列的生成，并在此基础上设计可靠性感知语义接口和原型个性化聚合。

**💡 创新点**

创新点在于：① 固定SID词表通过公共商品元数据实现跨域对齐，消除传统方法对交互信息的依赖；② 通过可靠性门控的残差注入把本地协同过滤信号融合进固定SID表示；③ 引入原型相似度加权的个性化聚合，针对域异质性实现负迁移抑制。

**🔧 技术方法**

核心技术包括：分层量化自编码（RQ-VAE）构建SID词表；Transformer自回归生成器；Mixture-of-Experts（MoE）共享-私有专家结构；可靠性感知残差融合；原型相似度加权的联邦聚合；密集辅助对齐损失。

**📊 数据集**

使用Amazon Review数据集构建六个跨域场景（Food-Kitchen、Grocery-Beauty、Grocery-Sports等），覆盖二域至四域不同相关性水平。

**📈 对比分析**

与单域推荐、传统联邦CDA、TIGER+FedAvg/FedProx等基线比较；在完整排名评估中FedCGR在22个域-指标组合上持续领先，尤其在高异质性场景GKBS中提升显著；在采样评估（999负）中亦获得最优性能，优于FedDCSR等最强联邦CDA基线。

**⚠️ 局限性**

局限性包括：1) 固定词表限制了对新物品的即时适应，需要离线更新；2) 对可靠性估计的手工设计（频率基准）可能不足以捕捉所有噪声；3) 额外的共享专家与密集辅助损失增加了模型参数与通信开销，尽管总体收敛更快；4) 仅在跨域商品推荐场景验证，需在更广泛的多模态/多任务环境进一步测试。

---

## 444. ThinkRetrieve: Retrieval-Augmented Reasoning Traces for Test-Time Scaling

**arXiv ID:** 2608.10928 | [PDF](https://arxiv.org/pdf/2608.10928v1)

**作者:** Vaibhav Singh `[一作]`, Dinesh Manocha `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在大型推理模型推理过程中动态检索已解决实例并注入到推理轨迹中的测试时刻扩展框架。

**💡 创新点**

在每一步推理后，基于模型当前的中间答案检索结构相似的完整解题实例，并将其作为上下文注入，既提供过程指引，又减少错误累积。

**🔧 技术方法**

采用密集向量检索（E5‑Large），在推理步骤中使用中间答案生成查询，实施连续检索‑注入机制；模型使用 Qwen 系列、DeepSeek‑R1 等大推理模型进行链式思考生成。

**📊 数据集**

使用 NuminaMath 生成的无重叠已解实例库作为检索语料，对 GSM‑8K、MATH‑500、AIME 2025 以及 SciQ 进行评估；SciQ 的训练集作为检索语料。

**📈 对比分析**

与标准顺序测试时刻扩展、静态输入级示例（S‑ICL）和随机检索等基线对比；在所有模型‑基准组合上均优于基线，最高可提升 13.4 分（AIME 2025），并在高预算下保持单调上升。

**⚠️ 局限性**

依赖检索语料覆盖，分布外问题可能无效或误导；检索增加推理延迟；检索过滤无法完全排除潜在答案泄漏；跨域适用性仍待验证。

---

## 445. FaithformBench: Benchmarking Faithfulness of Mathematical Chain-of-Thought Autoformalisation

**arXiv ID:** 2608.10916 | [PDF](https://arxiv.org/pdf/2608.10916v1)

**作者:** Rob Cornish `[一作]` (Nanyang Technological University), Luke Ong `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无人工标注、可扩展的评估方法（FaithformBench），通过自动扰动自然语言推理步骤并在证明助手中验证其可证明性，以评估自动形式化（AF）系统的忠实度。

**💡 创新点**

创新点在于：①引入两种失败模式（错误诱导与静默修正）并用证明器可靠检测；②基于扰动的低成本评估能同时考虑正确与错误输入；③构建大规模基准并公开共享，揭示现有AF在保持有效性与避免静默修正之间的紧张关系。

**🔧 技术方法**

使用：1）LLM（GPT‑5.2、Claude Opus 4.7 等）或正则表达式生成扰动；2）Lean 4 形式化表达式；3）DeepSeek‑Prover‑V2 证明器进行可证明性检验；4）统计指标 FNR、FPR、AFFR 以及综合低限值。

**📊 数据集**

数据集：从 ProcessBench 采集 12,784 条推理步骤，涵盖 GSM8K、MATH、OlympiadBench、Omni‑MATH 四个数学数据集。

**📈 对比分析**

比较方法：对 8 种 AF（4 细化模型 + 4 通用 LLM）在四个数据集上计算 FNR、FPR、AFFR 和综合低限值。结果显示：细化模型（尤其是 Goedel）在 FNR 低、FPR 高，表明其善于保持有效性但易于静默修正；通用 LLM 在 FPR 与 AFFR 上普遍更好，整体不忠实度更低。

**⚠️ 局限性**

局限性：①无法完全捕捉语义漂移；②评估依赖强力证明器，若证明器不足可能导致结果不确定；③仅给出忠实度下界，不能保证完全忠实。

---

## 446. Order Matters: LVLMs as Judges for Temporal Reasoning in Image Sequences

**arXiv ID:** 2608.10908 | [PDF](https://arxiv.org/pdf/2608.10908v1)

**作者:** Martina Ianaro `[一作]` (University of Bologna), Joao Magalhaes `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型视觉语言模型在多帧图像序列评估中的判断危机，提出了对比判定和分析-先评判的评估框架。

**💡 创新点**

发现并量化了因果掩蔽与RoPE造成的先序/后序偏差，并首次构建了专门测评时序推理的PRISM与MIRAGE数据集。

**🔧 技术方法**

采用Transformer架构、RoPE、因果掩蔽、LoRA微调、Gemini-2.5-Flash链式思考生成解释，以及Optuna进行超参数优化。

**📊 数据集**

使用自研PRISM（人工扰动的程序化图像序列）和MIRAGE（生成模型输出的真实/幻影序列）作为评估基准。

**📈 对比分析**

与LLaVA-Critic、LLaVA-OneVision对比，点wise得分高度一致但pairwise准确仅略高于50%；微调后提升至≈70%，但仍低于人类水平。

**⚠️ 局限性**

主要受Transformer的因果掩蔽与RoPE导致的序列位置偏差限制，微调难以完全消除，缺乏真正的时序推理能力。

---

## 447. GitSkills: A Dataset of Agent Skills on GitHub

**arXiv ID:** 2608.10906 | [PDF](https://arxiv.org/pdf/2608.10906v1)

**作者:** Giuseppe Destefanis `[一作]` (University College London), Marco Ortu `[通讯]` (University of Cagliari)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文收集并整理了数百万个GitHub公开仓库中的“agent skill”文件，构建了一个包含文件内容、元数据、文件哈希、文件夹结构和部分提交历史的SQLite数据集。

**💡 创新点**

创新之处在于：①首次提供了规模化的agent skill 数据集；②采用哈希去重只存储唯一内容，同时保留所有复制实例；③为每个代表性文件附加完整文本、前置信息、文件夹内容及部分提交记录，为后续研究提供了丰富的数据层次。

**🔧 技术方法**

技术手段包括：利用GitHub Code Search和REST API进行文件检索；通过文件大小分区突破1000条结果限制；使用内容哈希实现去重；对代表文件进行文本解析、前置信息提取和提交历史抽样；最终将所有表结构存储为单文件SQLite数据库。

**📊 数据集**

使用的数据集来源是公开的GitHub仓库，采集到的文件超过300万条，涵盖约数千个仓库。数据集包含文件内容、仓库元数据、文件夹结构、提交历史（对部分文件抽样）以及匿名化的作者信息。

**📈 对比分析**

本文未进行算法或模型性能对比；而是提供了数据资源，旨在为后续研究提供实验基础。研究者可以基于此数据集开展采用率、复用性、语义演化、维护安全等多方面的量化分析。

**⚠️ 局限性**

局限性包括：①仅覆盖公开仓库，无法反映私有仓库中的技能使用情况；②提交历史仅抽样采集，可能缺乏完整演化视角；③受GitHub API查询上限限制，检索策略可能导致部分文件遗漏；④文件归属位置不一，部分非标准位置的技能可能被低估；⑤匿名化作者信息虽保护隐私，但可能影响作者级别的深入分析。

---

## 448. MARS: A framework for modelling register-based social networks

**arXiv ID:** 2608.10946 | [PDF](https://arxiv.org/pdf/2608.10946v1)

**作者:** Katherine Hamilton `[一作]` (Leiden University), Frank Takes `[通讯]` (Leiden University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `67630363-6be0-4f51-ab05-7198250671a5` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了MARS（Multiplex Affiliation-based Random Spatially-embedded）框架，能够根据注册数据的生成机制模拟多层、空间嵌入的社交网络，并对其统计特性进行解析和数值验证；

**💡 创新点**

核心创新在于将节点与其所属机构在度量空间中随机嵌入，并通过与距离相关的连通函数来决定各层的连接，首次实现了对注册网络构建过程的可重现建模；

**🔧 技术方法**

使用软随机几何连接函数、随机几何网络生成、统计分析（度分布、密度、聚类系数）以及数值模拟方法；

**📊 数据集**

以荷兰全国人口注册网络（涵盖工作、学校、家庭、邻里等层）为实验数据；

**📈 对比分析**

通过将MARS生成网络的度分布、密度、聚类系数等指标与荷兰真实网络进行逐项比较，发现模型能够较好地重现其宏观结构；在空间自由度调节实验中，聚类系数下降与实证结果一致，验证了模型的可解释性；

**⚠️ 局限性**

局限性包括：模型对空间与连通函数的假设过于理想化，导致生成网络密度和聚类系数偏高；未充分考虑层间异质性和群组结构，未来需要引入块模型或更复杂的连接机制来提升现实性。

---

## 449. CARE: Confidence-Aware Reasoning for Reliable Medical VQA

**arXiv ID:** 2608.10964 | [PDF](https://arxiv.org/pdf/2608.10964v1)

**作者:** Yuetian Du `[一作]` (Zhejiang University), Qiang Zhu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了CARE框架，联合监督微调与GRPO强化学习，并通过置信度感知奖励CAR解决医疗视觉问答中的置信度失准问题。

**💡 创新点**

创新点在于：①构建可扩展的Medical‑CoT数据合成管线，实现结构化、可验证的诊断推理路径；②提出置信度感知奖励CAR，将置信度与诊断正确性直接嵌入RL奖励，实现置信度与准确率的同步优化；③使用GRPO无价值网络的强化学习，避免传统价值网络带来的估计误差。

**🔧 技术方法**

采用了大语言模型微调（SFT）、基于GRPO的强化学习、置信度感知奖励机制、自动化Medical‑CoT生成、Qwen2.5‑VL‑7B‑Instruct模型作为基础。

**📊 数据集**

实验使用了三大医学视觉问答基准：VQA‑RAD（放射学）、SLAKE（多模态知识增强）、PathVQA（病理学）。

**📈 对比分析**

与多种规模的主流医疗多模态大模型（Med‑R1‑3B、MedVLM‑R1‑2B、MedMO‑8B、Lingshu‑7B、MedVLThinker‑7B、Fleming‑VL‑8B）在准确率、置信度校准误差（ECE）和幻觉率（HR）三维度进行对比，CARE在所有基准上均获得最高准确率、最低ECE和最低HR，显著优于现有方法。

**⚠️ 局限性**

局限性包括：①对合成CoT路径的自动验证仍依赖外部模型（如GPT‑4o）且可能引入误判；②强化学习阶段计算成本高，需多 GPU 训练；③对开放式问答依赖SFT初始化，可能限制跨域泛化；④实验仅覆盖三大医学领域，缺乏跨领域或更大规模数据集的验证。

---

## 450. Once Poisoned, Arbitrarily Controlled: A Programmable Backdoor in VLMs

**arXiv ID:** 2608.10959 | [PDF](https://arxiv.org/pdf/2608.10959v1)

**作者:** Tao Lin `[一作]` (Chinese Academy of Sciences), Lijia Yu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种“any-to-any”视觉语言模型后门攻击，通过一次性毒化训练实现可在推理时动态生成任意目标描述的触发器。

**💡 创新点**

创新点在于：①以触发器-目标对的多样化训练，使模型学习“触发器即指令”的通用规则；②引入特征空间隐写技术 TS(·)，将任意目标文本映射为不可见或低可见度的视觉触发器，保持攻击隐蔽性；③实现了无需重训练即可为任意文本生成触发器，突破了传统一对一/多对多后门的局限。

**🔧 技术方法**

主要技术包括：自定义毒化策略（覆盖触发器图像并替换标题）、特征空间隐写（使用预训练视觉特征提取器对触发器进行优化），以及噪声/补丁触发器的生成（L∞约束、Patch约束）。

**📊 数据集**

使用的数据集有：CC-SBU-ALIGN（3,500张图文对）做主训练；Tiny-ImageNet（64×64）与CIFAR-100（32×32）做触发器-目标对；评估时采用 Flickr8k、Flickr30k；还在 GQA、MME 上验证模型正常性能。

**📈 对比分析**

与现有固定映射攻击（TrojVLM、MTAttack、IAG）以及 BadNets 进行对比。实验表明：在未见过的触发器-目标对上，any-to-any 方案的攻击成功率可达 86–92%，远高于固定映射攻击（0%）；对多种触发器（Vanilla、Patch、L∞）均保持高准确率；并在多种后门防御（Shrinkpad、Flip、Scale-up、Neural Cleanse 等）下依旧保持 80%+ 的成功率。

**⚠️ 局限性**

局限性包括：对触发器尺寸和噪声幅值敏感；需要预训练视觉特征提取器来生成隐写触发器；在极低可见度（如 L∞=8/255）或低分辨率触发器下效果下降；攻击依赖于 VLM 对视觉特征的泛化能力，若模型架构或训练方式显著不同，效果可能受限。

---

## 451. Mixture-of-Experts-based Entropy Model for Learned Image Compression

**arXiv ID:** 2608.10947 | [PDF](https://arxiv.org/pdf/2608.10947v1)

**作者:** Jonas Brenig `[一作]` (University of Würzburg), Radu Timofte `[通讯]` (University of Würzburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Mixture of Experts的熵模型来改进学习式图像压缩。

**💡 创新点**

通过为上下文建模引入可选择激活的专家网络，显著提升模型容量而不增加解码时延。

**🔧 技术方法**

使用Swin-Transformer编码器/解码器、channel-group熵模型、MoE层（门控网络+专家网络）、噪声softmax路由、辅助平衡损失和SGA隐层细化。

**📊 数据集**

在Kodak、Tecnick、CLIC 2020 Professional等标准基准上训练，训练集为ImageNet/COCO/DIV2K/Flickr2K共100k张512×512图像。

**📈 对比分析**

与VVC、DCAE、LALIC、MLIC++等方法比较，BD-Rate分别在Kodak-16.85%、Tecnick-22.09%、CLIC-8.08%上优于基线，且解码延迟保持在约73ms。

**⚠️ 局限性**

模型参数量相对更大，虽然延迟相似，但仍占用较高显存；在极低比特率下提升有限，训练复杂度较高。

---

## 452. GS-CPE: Unified 6-Degree-of-Freedom Camera Pose Estimation via 3D Gaussian Splatting

**arXiv ID:** 2608.10938 | [PDF](https://arxiv.org/pdf/2608.10938v1)

**作者:** Huaiyuan Weng `[一作]` (University of Waterloo), Su-Min Kang `[通讯]` (Soongsil University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

该论文提出了一种基于Gaussian Splatting的粗细化相机位姿估计框架GS-CPE，用检索-几何初始化和可视化加权图像对齐进行位姿求解。

**💡 创新点**

创新点在于将检索引导的几何位姿估计与可视化感知的多尺度、可见性掩模自适应重渲染相结合，显著提高了在大视角和遮挡下的鲁棒性。

**🔧 技术方法**

主要技术包括：NetVLAD检索、Transformer特征匹配、基于3DGS的可微渲染、可见性掩模生成、逐层多尺度光度对齐及自适应重渲染优化。

**📊 数据集**

在四个数据集上评估：7Scenes、Cambridge Landmarks、FAST‑LIVO2 以及私有 AC1，分别覆盖室内、室外以及不同光照与动态变化场景。

**📈 对比分析**

与现有 APR、SCR 以及 NeRF‑based NRP 方法相比，GS‑CPE 在 7Scenes 上实现 0.91 cm / 0.29° 的中位误差，Cambridge 约 21 cm / 0.39°，在 FAST‑LIVO2 和 AC1 上同样取得第一或第二名的精度，显示出更好的泛化和高精度结合。

**⚠️ 局限性**

主要局限包括：对 3DGS 生成质量高度依赖，重渲染仍占用显存与时间，单帧实时率约 0.62 FPS，且在极大尺度场景中的可扩展性尚未完全验证。

---

## 453. Robust Algebraic Theories of Triangle Graphs

**arXiv ID:** 2608.10927 | [PDF](https://arxiv.org/pdf/2608.10927v1)

**作者:** Marius Bozga `[一作]` (CNRS, Université Grenoble Alpes), Florian Zuleger `[通讯]` (Technische Universität München)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过引入并证明新的图代数，对树宽≤3且每条边均属于三角形的图（即三角图）以及满足特定连通性约束的子类（扇图）给出了完整的代数化表述，并进一步证明了这些图类在上下文无关、可识别与CMSO可定义三者之间的等价性，推出了相关判定问题的可判定性。

**💡 创新点**

创新点在于将经典的二元序列-并行代数（用于树宽2的系列并行图）扩展到树宽3，引入了三元序列操作和新的并行/串行组合，形成了针对三角图与扇图的专用代数；同时在代数与逻辑（CMSO）之间建立了新的可识别/可定义等价关系，并给出了一种基于树分解与转化的可解析构造。

**🔧 技术方法**

主要技术包括：图代数与图替换操作、树分解与k-步属性、三元序列构造、MSO逻辑与转化、上下文无关文法与可识别语言的定义，以及基于Courcelle定理的可判定性推导。

**📊 数据集**

本文没有使用实验数据集，所有结果均为理论证明与形式化定义，属于纯理论计算机科学研究。

**📈 对比分析**

由于研究聚焦于理论证明，未涉及实验比较或性能评估，论文的主要贡献体现在数学证明与理论框架的完善，而非算法实现或性能指标。

**⚠️ 局限性**

局限性包括：仅针对三角图与扇图两个特定图类，无法直接推广到更一般的树宽3图或更高树宽图；此外，虽然证明了可判定性，但未给出具体算法实现与复杂度分析，未来工作需进一步研究实现细节与复杂度。

---

## 454. ComBodied Agents: a New Paradigm of Human-Centric Agentic AI

**arXiv ID:** 2608.10915 | [PDF](https://arxiv.org/pdf/2608.10915v1)

**作者:** Qianggang Ding `[一作]` (Université de Montréal), Bang Liu `[通讯]` (Université de Montréal)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 Combodied Agents 的概念与闭环框架，阐述以人类状态为核心的多模态感知、长期记忆、个人世界模型与可授权干预的技术路线。

**💡 创新点**

创新点在于将数字代理与实体代理的优点融合，形成以事件驱动、多模态融合、长期可纠正记忆及基于情境的个人世界模型为核心的闭环系统，首次把人类状态的可持续提升和代理权的可逆性纳入设计目标。

**🔧 技术方法**

核心技术包括事件驱动多模态感知、可纠正的长期记忆结构、基于场景的个人世界模型（PWM）以及基于同意、安全、可逆性与可解释性的干预策略；同时引入云-边缘协作的分阶段部署。

**📊 数据集**

论文参考了多种公开基准（LongMemEval、PHIA、Health-LLM、iOSWorld 等）用于验证记忆、个性化与健康推理，但未构建新的专用数据集。

**📈 对比分析**

通过设计场景化评估框架，说明在健康、情感陪伴、学习、老年护理等情境下需要进行实验评估，现阶段尚无完整实验数据，性能指标待后续实证研究。

**⚠️ 局限性**

局限性包括缺乏完整的端到端实验验证、对多模态融合与个体化推理技术的挑战、隐私与治理问题、以及高风险决策的安全约束尚未充分验证。

---

## 455. ReOrder-OPD:Reliability-Aware Prompt Ordering for On-Policy Distillation

**arXiv ID:** 2608.10905 | [PDF](https://arxiv.org/pdf/2608.10905v1)

**作者:** Ximo Zhu `[一作]` (Hello Group Inc.), Xiaolei Lv `[通讯]` (Hello Group Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ReOrder-OPD方法，利用教师对学生前缀的延续可靠性进行Prompt排序，从而提升on‑policy distillation的训练效果。

**💡 创新点**

创新点包括：① 定义教师延续可靠性R并将其聚合为Prompt级可靠性；② 用一轮学生Rollout的ROUGE‑5最大值作为粗粒度代理来排序Prompt；③ 将Prompt级调度与Trajectory级监督分离，使两层干预可组合。

**🔧 技术方法**

采用的技术包括：on‑policy distillation (OPD)、ROUGE‑5相似度、答案验证器、教师与学生Rollout、动态优先级刷新、以及与FiRe‑OPD/ExOPD的组合监督。

**📊 数据集**

实验使用的主要数据集有 DeepMath‑1K、5K、17K（数学推理），以及 Eurus‑2‑RL‑Data 子集（代码生成）。评估指标为 AIME、HMMT、HumanEval+、MBPP+、LiveCodeBench V6 的 mean@16。

**📈 对比分析**

通过匹配对比（固定Prompt pool、更新预算、随机种子）与 Vanilla OPD、FiRe‑OPD、ExOPD 比较。结果显示，在数学和代码生成任务中，ReOrder‑OPD 均能提升 1–3 个百分点，并且与 FiRe‑OPD/ExOPD 组合后仍保持额外收益。

**⚠️ 局限性**

局限性包括：① 需要预先构建 verifier‑correct 的教师库；② 代理仅能区分粗粒度可靠性，不能提供精细的可靠性估计；③ 静态排序在大规模 Prompt pool 时效果衰减，需要动态刷新；④ 额外的学生Rollout 产生额外推理成本。

---

## 456. Partially Observable Learning for Multi-Platform Dispatch Optimization

**arXiv ID:** 2608.10897 | [PDF](https://arxiv.org/pdf/2608.10897v1)

**作者:** Fengming Yao `[一作]` (University of Exeter), Man Luo `[通讯]` (University of Exeter)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了POLO框架，基于部分可观测多智能体强化学习，解决多平台即时配送中平台只能获取自身订单与局部快递员信息的分布式调度问题。

**💡 创新点**

①将多平台配送定义为平台-网格对的独立智能体，实现完全基于平台本地观测的学习；②在策略网络中引入注意力机制，对候选快递员进行交互式聚合；③设计基于因果反事实的奖励塑造，缓解因共享快递员导致的非平稳性。

**🔧 技术方法**

多智能体强化学习（actor‑critic），自注意力聚合编码器，因果反事实奖励，中心化训练/去中心化执行框架。

**📊 数据集**

真实Meituan餐饮配送数据，覆盖连续八天的订单与快递员轨迹，用此构建高保真模拟器。

**📈 对比分析**

与Random、PRandom、DisGreedy、DetGreedy、IBM、TCAC、FAIR等基线比较，以GMV和平均快递员行驶距离（ACTD）评估。POLO在单平台、双平台以及大规模场景中均取得最高GMV，尤其在多平台大规模时提升超过20%；虽然ACTD略高于纯距离最小化基线，但整体收益和效率均优于其他学习型方法。

**⚠️ 局限性**

受限于仅在模拟环境中验证，未在真实系统上部署；需要手动设定网格尺寸和候选快递数，超大规模平台数量时训练与推理仍有一定开销；对极端低快递可用性场景的鲁棒性尚待进一步验证。

---

## 457. VibeLifeBench: Can Your Life Agent Be Proactive and Persistent in a Living World?

**arXiv ID:** 2608.10875 | [PDF](https://arxiv.org/pdf/2608.10875v1)

**作者:** Xiaohongshu Inc `[一作]` `[通讯]`, Xiaohongshu Inc

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出VibeLifeBench基准，利用多周、动态生活场景评估LLM代理的长期协助能力。

**💡 创新点**

将主动性、持续性与生活世界演进三大属性融入细粒度加权评分，构建200个跨10个日常领域任务。

**🔧 技术方法**

构建22个模拟服务后端，使用事件系统（用户、观察、通知、突变）驱动时间轴，并通过Terrarium执行多轮交互收集可观察输出。

**📊 数据集**

手工设计的200个任务脚本（共7453事件、12,261检查），覆盖旅行、金融、法律等十个领域。

**📈 对比分析**

在7个最新模型（Claude、GPT、Gemini等）上跑三次，avg@3最高仅32.5（Claude 5），表明当前模型在长期生活协助上的表现不足。

**⚠️ 局限性**

模型缺乏主动感知、跨阶段持久状态、对突变的处理以及跨领域一致性，导致得分低且不稳定。

---

## 458. A Consolidated Game Framework for Cooperative Defense Against Cross-Domain Cyber Attacks in Satellite-Enabled Internet of Things

**arXiv ID:** 2608.10873 | [PDF](https://arxiv.org/pdf/2608.10873v1)

**作者:** Linan Huang `[一作]` (Tsinghua University), Jianhua Lu `[通讯]` (Tsinghua University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种三方博弈框架，联合卫星与物联网网络中的攻击、IoT 运营商与卫星服务商的协同防御，以解决跨域网络攻击问题。

**💡 创新点**

将跨域攻击影响量化并设计了跨域协同博弈均衡（CSNE）与阈值分区理论，利用分段线性特性构建高效的价格与采样频率学习算法，弥补了以往单域防御与无策略对齐难题的不足。

**🔧 技术方法**

基于博弈理论的三方协同博弈模型、最优采样频率与防御配置的最佳响应、UCB多臂赌博机与分段线性二分搜索学习算法。

**📊 数据集**

通过仿真生成的IoT设备数目、流量速率、攻击强度等参数（如N=100、µNor=100pps、µMal=15pps、αSeR=0.8等），未使用公开数据集。

**📈 对比分析**

与单域防御对比实验表明，协同博弈框架可将IoT-NO与SAT-SP的收益提升约10–12%，并在攻击强度高或设备规模大时显著缓解规模效应失效。

**⚠️ 局限性**

局限在于模型假设设备状态与攻击均为可观测的离散过程，未考虑多阶段复杂攻击、隐蔽特征检测与隐私保护，且实验基于理想化仿真，需在真实卫星/物联网环境中进一步验证。

---

## 459. NullEdit: Stealthy Image Protection via VLM Condition Redirection

**arXiv ID:** 2608.10870 | [PDF](https://arxiv.org/pdf/2608.10870v1)

**作者:** Weiyao Huang `[一作]` (Sun Yat-sen University), Wei Lu `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种针对 VLM‑条件 Diffusion Transformer 编辑器的无监督防护机制，使未经授权的图像编辑被压制为无操作，并保持源图像的一致性。

**💡 创新点**

创新点在于通过在 VLM 隐藏层空间中实现平衡的正向编辑与无编辑锚定的条件重定向，并结合跨提示梯度平均，实现了隐蔽且无害的“无操作”防护。

**🔧 技术方法**

采用了 VLM 语义编码、Diffusion Transformer backbone、方向性相似度约束、梯度投影、动量优化以及跨提示梯度平均等技术。

**📊 数据集**

实验使用 CelebA‑HQ 与 VGGFace2 两个面部图像数据集，包含多身份样本。

**📈 对比分析**

与 DiffPGD、VAE、DeContext 等基线对比，在 Step1X‑Edit 与 Qwen‑Image‑Edit 上 EditReward IF 均降低约 0.8+，同时保持高源内容与身份相似度，且用户研究显示效果居首。

**⚠️ 局限性**

局限性在于仍依赖对 VLM 表示空间的访问，对极端或未见提示的泛化尚有不足，且在更高分辨率或不同模型时需要进一步验证。

---

## 460. FormaTheoria: Constructing Large-Scale Lean Theories from Mathematical Literature $-$ Toward the Formalization of the Classification of Finite Simple Groups

**arXiv ID:** 2608.10894 | [PDF](https://arxiv.org/pdf/2608.10894v1)

**作者:** Tianjiao Nie `[一作]` (Tsinghua University), Yuan Zhou `[通讯]` (Tsinghua University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 FormaTheoria 工作流，自动从散布在 15 本学术著作中的文献中重构 CFSG（有限单群分类）的 Lean 形式化，并覆盖 Feit–Thompson 奇数阶定理、Glauberman 的 Z* 定理、Brauer–Suzuki 定理以及 Bender–Suzuki 定理，生成 994k 行 Lean 代码。

**💡 创新点**

创新点在于：① 端到端 AI 辅助的多阶段工作流（源检索、语义翻译、递归依赖发现、跨源对齐、语义审查、回溯修正）；② 共享代理框架支持多轮交互、上下文压缩和依赖感知的批量并行化；③ 引入独立的语义审查与修正机制，显著降低翻译错误和源缺陷的传播。

**🔧 技术方法**

技术手段包括：大语言模型驱动的 TranslatorAgent、Prover（图式证明分解）、ReconcilerAgent、共享代理框架、Lean 交互工具、上下文压缩、章节级上下文共享、依赖感知批量并行化、语义审查与自动修复。

**📊 数据集**

数据集：来自 CFSG 相关的 15 本书/论文（共 1,037 页），构成的源文献被自动检索、OCR 转写；最终生成 994,000 行 Lean 代码，覆盖 850+ 文件。

**📈 对比分析**

与现有手工形式化（如 Rocq 的 Feit–Thompson 证明 150k 行、Mathlib 1.9M 行）对比，本文在 7 个月内完成，利用依赖感知并行化将墙壁时间缩短 76%（4.2 倍），章节级共享降低 26% 体量并减少 18% token 消耗，语义审查提升翻译质量，实验中 10/10 源项通过自我改进审查，9/10 在诊断审查后通过。

**⚠️ 局限性**

局限性：生成的代码量巨大且结构复杂，尚未充分优化可重用性和可读性；仍需人工审查解决无法自动化的源缺陷和跨源不一致；工作流高度依赖文献检索和 OCR 的准确性，错误来源可能影响整体质量；资源需求大，尤其在并行化和上下文压缩阶段。

---

## 461. CSS Quantum LRCs with Intersecting Recovery Sets: Constructions and Bounds

**arXiv ID:** 2608.10912 | [PDF](https://arxiv.org/pdf/2608.10912v1)

**作者:** Evagoras Stylianou `[一作]` (Technical University of Munich), Rawad Bitar `[通讯]` (Technical University of Munich)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文研究了使用 CSS 构造的 (r,t,x)-量子局部可恢复码（qLRCs），并给出了通过子集包含矩阵构造的二进制代码族；

**💡 创新点**

创新点在于证明在双重码距至少为二的前提下，CSS qLRC 与具有公共恢复集的经典 (r,t,x)-LRC 等价，并利用子集包含矩阵得到一系列高速、非平凡距离的二进制 CSS qLRC；

**🔧 技术方法**

主要技术包括：CSS 构造、子集包含矩阵的正交性判定、经典 LRC 的恢复集与交叉参数分析，以及对纯 CSS 代码距离的证明；

**📊 数据集**

本文未使用外部数据集，所有参数均为理论构造所得；

**📈 对比分析**

与 Bu–Gu–Li 的唯一已知精确构造相比，新构造在可用性固定为 3、交叉参数更大但速率更高、距离固定为 4，并且比 Bu–Gu–Li 在大参数下的速率更好；

**⚠️ 局限性**

局限性包括仅适用于二进制 CSS 代码，无法直接推广到非二进制或更宽泛的 LDPC 结构，以及对交叉参数的上界和距离上界仍未完全匹配。

---

## 462. GESTO: Human-Centric Spatio-Temporal Memory for Reasoning in Dynamic Scenes

**arXiv ID:** 2608.10886 | [PDF](https://arxiv.org/pdf/2608.10886v1)

**作者:** Ermanno Bartoli `[一作]` (Anonymous Institution), Iolanda Leite `[通讯]` (Anonymous Institution)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在机器人视觉系统中，提出一种将 4D 场景图与两层层次化的活动层次结构（原子交互与目标驱动事件）进行耦合的记忆框架，自动从 RGB‑D 流中提取、定位交互并生成事件，从而支持跨时间和空间的回溯推理。

**💡 创新点**

创新点在于：①将原子人机交互与目标驱动事件两级层次化结构与 4D 场景图进行显式耦合；②通过上下文感知的关联修正，实现无外部事件边界与对象对应的自动化构建；③使用工具调用式关系感知代理完成跨模态查询。

**🔧 技术方法**

使用技术包括：基于 VLM（Cosmos‑Reason2）的原子交互提取；FastSAM+sentence‑t5‑large 的掩码与文本嵌入；几何 IoU 与语义相似度的双重匹配；LLM 进行事件聚类与纠错；以及工具调用式关系感知代理。

**📊 数据集**

数据集：主要使用 4D 场景图基准（包含 80 条查询）并扩展 40 条 Space2Event 与 Event2Space 查询；另外在 EgoLive 与 HOI4D 上做无监督的 egocentric 评估。

**📈 对比分析**

与 DAAAM、ReMEmbR 以及 EGG 等基线比较，在标准基准上取得 0.71 文本准确率、0.75 二进制 F1、0.70 时间准确率，Space2Event 0.73、Event2Space 0.75，基本可与 EGG（使用外部事件/对象标注）相媲美，并显著优于完全自动化的 EGG。

**⚠️ 局限性**

局限性：①VLM 对对象定位依赖外观，受遮挡或相似物体影响；②掩码缺失或错误导致交互未能匹配或匹配错误；③记忆随会话增长，长期部署需进一步压缩或归纳；④仅在实验环境中评估，真实场景鲁棒性待验证。

---

## 463. Search-to-Decision Reductions for the Linear and General Code Equivalence Problems

**arXiv ID:** 2608.10967 | [PDF](https://arxiv.org/pdf/2608.10967v1)

**作者:** Abhinaba Mazumder `[一作]` `[通讯]` (University of Zurich), Abhinaba Mazumder (University of Zurich)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文展示了从判定问题到搜索问题的有效降维方法，分别针对线性码等价（LCE）和广义码等价（GCE）问题。

**💡 创新点**

创新点在于：① 将 Permutation Code Equivalence（PCE）的搜索-判定降维框架推广到 LCE 与 GCE；② 通过项目类划分恢复置换；③ 采用 Engel-Schneider 算法在多项式时间内恢复对角矩阵；④ 证明在一次 oracle 调用后即可完成整个搜索过程。

**🔧 技术方法**

技术方法包括：项目类（proportional columns）划分、递归构造查询、广义等价的半线性结构分析、以及对角等价问题的图论求解（BFS+权重比值）。

**📊 数据集**

无实验数据集，研究完全基于理论证明和算法分析。

**📈 对比分析**

方法通过多项式时间复杂度（最多 n² 次 LCE oracle 调用，和 O(nk²+k^ω) 线性代数运算）与先前已知的搜索-判定降维方法（主要用于 PCE）对比，证明了在 LCE 与 GCE 上同样可实现多项式时间搜索。没有给出具体数值性能评估。

**⚠️ 局限性**

局限性：仅覆盖 Hamming 码等价的三种变体；对矩阵码等价（MCE）的搜索-判定降维仍未解决；实验验证缺失，实际实现效果与理论复杂度之间的差距未进一步评估。

---

## 464. SafeCA: Safe Cross-Attention Localization and Regulation for Text-to-Video Jailbreak Defense

**arXiv ID:** 2608.10933 | [PDF](https://arxiv.org/pdf/2608.10933v1)

**作者:** Siyuan Liang `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SafeCA，一种基于跨注意力特征的防止 jailbreak 的机制，在推理阶段通过对跨注意力块进行定位、能量归一化、注意力掩蔽和语义适配器，抑制恶意语义扩散并保持视频质量。

**💡 创新点**

首次揭示了跨注意力过程中存在的累计分离效应与线性可分性提升现象，并在此基础上设计了分层注意力定位与特征层面正则化的完整防御框架，既不修改模型权重，也实现了实时、轻量化的部署。

**🔧 技术方法**

采用跨注意力统计分析、PCA 语义子空间、能量归一化、指数衰减掩蔽、语义适配器以及逆向梯度将异常信号映射回输入 token 的技术；所有操作均在推理时完成，保持模型冻结。

**📊 数据集**

使用公开的 T2VSafetyBench 与 SafeWatch（共计 580+ jailbreak 提示）、WebVid-10M 的 100 条干净提示以及 Open‑Sora、CogVideo、Sora、Kling、Luma 等开源/闭源 T2V 模型进行评测。

**📈 对比分析**

与 SAFREE、VideoEraser、T2VShield、关键词过滤等现有防御对比，SafeCA 在 T2VSafetyBench 上将 ASR 从 29.73% 降低到 23.93%（≈19.5% 下降），GPT‑4o 分数从 28.17 降至 21.07（≈25.2% 降低），在闭源模型 Sora 上同样显著提升，且推理开销仅 +0.1s，保持了语义一致性与视频质量。

**⚠️ 局限性**

对高度复杂或长时序隐式语义的 jailbreak 暴露的覆盖率仍有限，未来需要引入时序建模与可学习正则化以提升对多样化攻击模式的适应性。

---

## 465. Bounds for Pure Disjoint $(r,δ)$-Quantum Locally Recoverable Codes

**arXiv ID:** 2608.10922 | [PDF](https://arxiv.org/pdf/2608.10922v1)

**作者:** Evagoras Stylianou `[一作]` (Technical University of Munich), Holger Boche `[通讯]` (Technical University of Munich)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了纯的无交叉 (r,δ) 量子局部可恢复码 (qLRC)，并在不依赖稳定器结构的情况下给出了单点 Singleton‑类下界和线性规划上界；

**💡 创新点**

创新点在于提出局部 Knill–Laflamme 条件与块级 Shor–Laflamme 与单位元枚举器相结合的全新框架，从而实现对错误权重在恢复区块中的分布进行精细刻画，进而得到比已知结果更强的下界；

**🔧 技术方法**

主要技术包括局部 KL 条件推导、块级 Shor–Laflamme 与单位元枚举器、Krawtchouk 多项式展开、以及基于枚举器的线性规划（LP）方法；

**📊 数据集**

无数据集，论文完全基于理论推导与数学证明；

**📈 对比分析**

与已有的 Singleton‑类下界和经典/稳定器对应的 LRC 下界比较，证明在纯度假设下可获得更严格的下界；LP 上界与松弛的已知界相比更为紧；实验性或数值验证仅通过理论上界的比较；

**⚠️ 局限性**

局限在于仅适用于纯、无交叉的 (r,δ) qLRC，无法直接推广到非纯或重叠恢复区块的情形；缺乏具体编码构造或实现示例；

---

## 466. VIDS-Seg: Towards Reliable Uncertainty Quantification in Pediatric Cardiac Ultrasound Segmentation

**arXiv ID:** 2608.10903 | [PDF](https://arxiv.org/pdf/2608.10903v1)

**作者:** Paul Fischer `[一作]` (University of Basel), Ece Ozkan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `f7dab867-23a8-4241-85e9-4ba79c6402f9`

**🎯 论文内容**

本文提出了适用于医学图像分割的 VIDS‑Seg，能够在无额外标注数据的情况下检测成人训练模型在儿童（尤其婴儿）群体上的潜在失效。

**💡 创新点**

创新点在于将 VIDS 的自适应先验扩展到像素级分割，采用轻量预测头的可变形后验推断，使得在分布漂移时可显式提升不确定性，并与标准校准方法区分。

**🔧 技术方法**

采用变分推断、可变形先验、可变形网络（U‑Net embedding + 1×1 预测头）以及温度标定等技术。

**📊 数据集**

使用成人 EchoNet‑Dynamic 作为训练集，儿科 EchoNet‑Pediatric 作为测试集，并按年龄分层（婴儿、幼童、学童、青少年）。

**📈 对比分析**

与深度集成、PHiSeg 等基线比较，分割 Dice/HD95 与 ID 一致，婴儿子组表现下降；在不确定性评估中 VIDS‑Seg 的 NCC 最高，温度标定后仍保持优势；下游 EF 估计 MAE 与心脏功能判别 AUROC 也表现最佳。

**⚠️ 局限性**

局限在于对婴儿群体仍存在误差波动，未针对多器官或更大规模数据进行验证，且需进一步把像素不确定性转化为临床可解释的指标。

---

## 467. Benchmarking Time Series Generation Methods for Privacy-Preserving Forecasting

**arXiv ID:** 2608.10891 | [PDF](https://arxiv.org/pdf/2608.10891v1)

**作者:** Luis Amorim `[一作]` (University of Porto), Carlos Soares `[通讯]` (University of Porto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在隐私敏感场景下使用合成时间序列替代原始数据进行预测，并通过 Train on Synthetic, Test on Real (TSTR) 基准评估预测性能与隐私风险。

**💡 创新点**

提出了基于图的隐私增强生成器 Grasynda-P，结合矩阵集成和核密度估计，兼顾预测性能与隐私隔离。

**🔧 技术方法**

采用图生成、矩阵集成、核密度估计、深度生成模型、变换方法、噪声扰动以及 NHITS 预测等技术。

**📊 数据集**

使用了七个公开时间序列数据集（M1、M3、Tourism、NN3 等），涵盖行业、人口与经济等领域。

**📈 对比分析**

通过 TSTR 对比多种生成器和噪声方法，发现噪声方法隐私最高但预测最差；简单变换方法预测最好；Grasynda-P 位于预测-隐私 Pareto 前沿，兼具较高预测性能和较强隐私隔离。

**⚠️ 局限性**

局限在于隐私评估仅为经验距离度量，未给出正式差分隐私保证；实验仅针对单变量、月/季节数据和单一预测模型，未验证在多变量或高频数据上的泛化。

---

## 468. Complexity and algorithms for proper conflict-free coloring in graphs

**arXiv ID:** 2608.10874 | [PDF](https://arxiv.org/pdf/2608.10874v1)

**作者:** Dinabandhu Pradhan `[一作]` (Indian Institute of Technology (ISM)), Vaishali Sharma `[通讯]` (Indian Institute of Technology (ISM))

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了冲突自由着色（PCF）问题的复杂性与算法，在特定图类中证明了NP-完整性、逼近下界，并给出多类图的线性时间最优着色算法；

**💡 创新点**

创新点在于首次证明PCF k-可着色在完美消除双侧图和二元弦图中仍为NP-完整，并提出对PCF chromatic number的O(n^{1-ε})逼近下界，同时提供块图、正确区间图、链图、伪拆分图的最优线性时间算法以及χ_{pcf}≤ω+1的紧上界；

**🔧 技术方法**

采用了归约证明、最大邻域序、树结构分解、BCE/PEO等图结构算法以及贪心着色策略；

**📊 数据集**

本文为理论性研究，未使用具体实验数据集；

**📈 对比分析**

通过理论证明比较复杂度，所给算法实现线性时间，能得到最优着色，且在一般图中逼近高效性受限；

**⚠️ 局限性**

局限性在于结果仅适用于所述特殊图类，无法直接推广到所有弦图或更广泛图类，逼近下界在实际应用中的意义有限。

---

## 469. Optimistic Rates for Multiclass PAC Learning

**arXiv ID:** 2608.10869 | [PDF](https://arxiv.org/pdf/2608.10869v1)

**作者:** Xiaoyu Li `[一作]` (University of New South Wales), Junbin Gao `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文证明了多分类学习中在oracle风险已接近零时的最优“乐观”学习速率，给出了上界与下界的匹配；同时将该结论推广到列表学习；

**💡 创新点**

创新点在于提出了相对压缩定理（相对比较、无稳定性、保持系数一），通过覆盖‑菜单‑压缩三块架构实现了两个维度（Natarajan维度和DS维度）分别对应波动与可实现余项的分离；

**🔧 技术方法**

主要技术包括覆盖‑菜单‑压缩框架、相对压缩定理、局部化偏差与Bernstein不等式、二分量的分离分析以及对伪立方和Natarajan/DS结构的精细构造；

**📊 数据集**

本工作为纯理论研究，未使用公开数据集；所有结果通过数学构造与离散示例验证；

**📈 对比分析**

通过与已知的最坏情况和已实现下界比较，证明了给定的上界与下界在依赖维度和样本量方面相匹配（仅差常数与多项对数项）；

**⚠️ 局限性**

局限性包括：上界仍含多项对数项；算法仅信息理论性，未给出高效实现；仅适用于有限标签集，无法直接推广到无限标签空间；

---

## 470. REAP: Relation-Aware Elicitation and Parsing for Closed-Book Knowledge Base Construction from LLMs

**arXiv ID:** 2608.10963 | [PDF](https://arxiv.org/pdf/2608.10963v1)

**作者:** Thanh-Dan Bui `[一作]` (VNU University of Engineering and Technology), Tuan-Phong Nguyen `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于Mistral-24B的两阶段闭书知识库构建流水线，利用关系感知的提示、链式思考和空集门控来提取事实，并通过直接解析生成规范的JSON答案。

**💡 创新点**

创新点在于将事实召回与答案序列化分离，采用关系特定的多步链式推理和推理空集门控，显著降低幻觉和格式错误，提高多值关系召回率。

**🔧 技术方法**

使用的技术包括关系感知提示工程、链式思考（CoT）推理、推理空集门控、正则表达式与直接JSON解析、以及vLLM+TPU的批量推理。

**📊 数据集**

使用的实验数据集是AKBC Shared Task 2026提供的训练/验证/测试集，共6个关系，涵盖空集、单值和多值对象集合。

**📈 对比分析**

通过在验证集上对Mistral-24B、Gemma-2-9B、Llama-3.1-8B三大模型进行零样本和两阶段流水线对比，Mistral-24B在验证集上macro‑F1达到0.65，测试集macro‑F1为0.62，显著优于Llama、Gemma和官方Qwen3.5-9B基线。

**⚠️ 局限性**

主要局限包括对极罕见实体的知识缺失、量化关系的数值误差有时超出5%容差，以及采样与TPU分布式计算导致的结果波动，venues关系仍是最难的挑战。

---

## 471. Evidence-Grounded Trustworthy Multimodal Reasoning and Evaluation Benchmark in Complex Urban Scenes

**arXiv ID:** 2608.10954 | [PDF](https://arxiv.org/pdf/2608.10954v1)

**作者:** Zhaoyang Wei `[一作]` (University of Chinese Academy of Sciences), Jianbin Jiao `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对复杂城市场景中的多模态推理与评估的 AD^2‑Bench 基准，并基于链式证据（CoE）构建的 Evidence‑Grounded Visual Reasoning (EGVOR) 框架，实现了对视觉证据的显式生成与校准；

**💡 创新点**

创新点在于：① 采用分层视觉诊断，将推理拆分为感知、关系与决策四层，形成可量化的链式证据评估；② 引入“证据原子”结构（定位–关注–描述），强制模型在视觉空间和语义上进行对齐；③ 通过层级课程学习与多目标强化学习（HRGW + 语义对齐 + 路径多样性）显著抑制空间模糊和语义不确定；

**🔧 技术方法**

使用了多模态大语言模型（如 Qwen‑VL、LLaVA、InternVL）作为基础，结合自监督链式生成、聚焦-关注-描述的定位模块、基于高斯‑沃斯特曼距离的空间奖励、语义一致性奖励以及组相对策略优化等技术；

**📊 数据集**

构建了大规模的 AD^2‑Bench（10k 场景图像、70k QA 对），并在 V*、HR‑Bench、MME‑RealWorld、TreeBench 等公开基准上进行跨域验证；

**📈 对比分析**

与现有 18 种 MLLM（包括 7B‑72B 级别及 GPT‑4、Gemini 等）对比，EGVOR 在 AD^2‑Bench 的平均分从 55% 提升至 67.7%，在 V*、HR‑Bench、MME‑RealWorld 等任务上也分别提升 10‑20%，同时显著降低视觉熵差、提升推理稳定性；

**⚠️ 局限性**

局限性包括：① 训练和推理成本较高，需两阶段课程与 RL；② 当前仅支持单帧图像，缺乏时序证据链；③ 对极端遮挡或极低分辨率场景仍可能产生轻微失真；

---

## 472. Physics-informed Diffusion Generative Model for Time-Series Data Synthesis in Dynamic Systems

**arXiv ID:** 2608.10941 | [PDF](https://arxiv.org/pdf/2608.10941v1)

**作者:** Haiteng Wang `[一作]` (Beihang University), Lei Ren `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Physics-informed Diffusion Generative Model（PhysDGM），用于生成符合工业系统物理规律的时序数据，缓解数据稀缺问题。

**💡 创新点**

核心创新在于：①在逆扩散过程的每一步嵌入物理约束，实现逐步物理一致性；②采用动态约束训练策略，逐渐加强物理正则化；③提出梯度引导采样，使预训练模型可无需再训练即可适配新物理环境。

**🔧 技术方法**

技术手段包括：DDPM扩散模型、U‑Net 结构、物理约束正则化（退化、耦合、范围、离散），动态权重函数和梯度引导采样。

**📊 数据集**

实验使用34个工业时序数据集（涡轮风扇发动机、航空发动机、电池、化工过程）生成约440万条合成样本，并与原始真实数据混合训练。

**📈 对比分析**

与 DiffWave、DiT、SSSD、TabDDPM、PINN 等基线对比，采用判别得分、预测得分、RMSE、ACC 等指标评估；PhysDGM 在下游 RUL、HI、SOH、故障诊断任务中平均降低 15–48% RMSE、提升 20–48% 预测准确度，并仅需 10–20 倍更少的真实数据即可逼近完整数据性能。

**⚠️ 局限性**

局限性包括：对可微分物理约束的依赖，难以处理离散事件或非光滑动力学；扩散采样计算成本高，需进一步优化；在更广泛工业领域的泛化能力仍待验证。

---

## 473. IO Factory: Simulating AI-Enabled Influence Campaigns at Scale

**arXiv ID:** 2608.10920 | [PDF](https://arxiv.org/pdf/2608.10920v1)

**作者:** Lukasz Olejnik `[一作]` (King's College London), Daniel Thilo Schroeder `[通讯]` (SINTEF Digital)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 IO Factory 框架，在受控仿真平台上完整模拟 AI 驱动的影响行动生命周期，记录从策划、平台动作、曝光到测量和结果的可追溯链路。

**💡 创新点**

将影响行动生命周期与可追溯的多层测量、匹配基线比较结合，提供可复现、可检验的红队实验环境，并支持对 AI 机器人群组行为的规模化模拟。

**🔧 技术方法**

利用多智能体 LLM 代理、OASIS 社交平台仿真、LLM 判定器、构造更新规则、匹配基线比较和大规模并行仿真技术。

**📊 数据集**

使用模拟人口（10,000 或 100,000 伪人）、自定义构造定义（如信任、支持）以及 LLM 生成的叙事内容，未使用真实社交网络数据。

**📈 对比分析**

通过匹配基线与激活运行的“方向提升”指标进行配对自助法比较，使用 13 次匹配复制，结果在 10,000 人规模下可在数小时内完成，支持 100,000 规模仿真，显示显著方向提升。

**⚠️ 局限性**

结果仅为模拟尺度，未与真实世界数据校准；LLM 判定器并非真值，受模型版本和提示影响；对图结构、推荐机制等平台假设过度简化；缺乏对人类与机器曝光的区分。

---

## 474. Sensor-Informed Per-Point Covariance for Structured-Light 3D Imaging

**arXiv ID:** 2608.10888 | [PDF](https://arxiv.org/pdf/2608.10888v1)

**作者:** Sehoon Tak `[一作]` (Yonsei University), Jae-Sang Hyun `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

建立了一种基于实验测得的相位精度和校准后的相位‑深度映射的第一阶点云协方差模型，用于结构光3D重建，并实现了每点3×3协方差场的构造与验证。

**💡 创新点**

创新点在于：①直接将相位噪声通过相位‑深度与相位‑3D映射传播到3D协方差，形成物理主的相位诱导协方差；②再通过测量空间（u,v,Φ）协方差完成全秩正则化；③避免依赖局部几何邻域，提供纯测量过程的协方差；④在G‑ICP注册中验证该协方差模型的有效性。

**🔧 技术方法**

采用了结构光相位提取与解缠、相位精度统计、相位‑深度与相位‑3D映射的校准、Jacobian线性传播、协方差正则化、G‑ICP配准等技术。

**📊 数据集**

使用了实验室平面重复采样数据（10深度×7倾斜×20重复）用于相位精度与映射校准；并采集了14对雕像扫描数据用于G‑ICP验证。

**📈 对比分析**

将提出的协方差模型与等方差模型和基于局部邻域PCA的几何协方差进行对比。实验结果显示，提出模型在G‑ICP中翻译RMSE约为0.214mm、旋转RMSE约为0.422mm，显著优于等方差模型（0.718mm/1.134mm），但略逊于几何协方差（0.110mm/0.352mm）。

**⚠️ 局限性**

主要局限：仅在近似恒定SNR的受控条件下验证；未考虑相机畸变、相位‑深度模型不确定性、像素间相关性；未对空间变化的SNR和更广泛的硬件平台进行推广；以及未与几何协方差融合的混合模型研究。

---

## 475. ConfTriage: A Calibration-Aware LLM Triage Framework for Pulmonary Nodule Malignancy with Selective Specialist Deferral

**arXiv ID:** 2608.10885 | [PDF](https://arxiv.org/pdf/2608.10885v1)

**作者:** Md Rabiul Islam `[一作]` (Texas A&M University), Hasan Kurban `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

构建了一套基于通用大型语言模型的置信度校准三层筛选框架（ConfTria），该框架仅使用肺结节的结构化属性转化为自然语言描述，利用LLM输出的置信度与logprob相结合，再通过Platt校准，最后将低置信度病例交给专门的深度学习后备模型进行判定；同时在整个流程中给出了有限样本错误上界、校准误差与最优退避之间的界定，以及在分布漂移下的校准漂移上界。

**💡 创新点**

创新点主要体现在三层结构：①把结构化属性转为自然语言作为输入而非原始像素；②利用LLM的文字置信度与token logprob在对数几率空间结合后进行Platt校准，从而得到可解释的、可度量的置信度；③设计了低置信度的判定阈值与专门的MC‑dropout深度学习后备，提供了正式的安全证明，并给出了校准误差对退避性能的理论上限。

**🔧 技术方法**

使用了通用LLM（GPT‑4o‑mini、Claude‑3.5‑Haiku、Gemini‑3.1‑Flash‑Lite、Mistral‑Large、Qwen‑2.5‑72B）、对数几率空间置信度融合、Platt scaling、Monte‑Carlo dropout 组合的 ResNet‑50/ EfficientNet‑V2‑B0 专家模型、以及对有限样本、校准漂移与校准误差界定的统计学证明。

**📊 数据集**

主要使用公开的 LIDC‑IDRI 数据集（955 例肺结节，患者级 5‑折交叉验证），并在实验中提及对 SPIE‑AAPM‑NCI LUNGx 与 LNDb 的外部验证。

**📈 对比分析**

在六种输入组合（仅属性、仅文字、仅图像统计、属性+文字、属性+图像、全部）上对五种顶尖LLM进行控制性消融，发现文字输入能够获得 AUROC≈0.92（Gemini‑3.1‑Flash‑Lite），几乎与 0.93 的专门深度学习后备持平；图像统计输入几乎无判别力。与传统深度学习模型和其他基线相比，LLM 的文字输入在不需要图像训练的情况下达到相当性能，且统计检验显示强模型显著优于弱模型（ΔAUROC>0.13，p<0.001）。

**⚠️ 局限性**

局限性包括：LIDC‑IDRI 的恶性标签基于放射科医生主观评分，缺乏病理验证；实验未包含前瞻性临床验证；框架依赖结构化属性的可获得性；LLM 模型会随时间快速过时；仅针对二分类任务；理论证明基于 i.i.d. 与 TV‑距离有限的假设，实际部署需考虑非平稳性和后备模型的校准稳定性。

---

## 476. Enhanced Filtering Algorithms for the Euclidean Traveling Salesperson Problem and its variants in Constraint Logic Programming

**arXiv ID:** 2608.10881 | [PDF](https://arxiv.org/pdf/2608.10881v1)

**作者:** Alessandro Bertagnon `[一作]` (University of Ferrara), Marco Gavanelli `[通讯]` (University of Ferrara)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在约束逻辑程序设计框架下，提出了利用欧几里得距离的几何信息（不相交与凸包顺序）对欧几里得旅行商问题（ETSP）以及欧几里得广义旅行商问题（EGTSP）进行强约束传播的算法，显著缩小搜索空间；

**💡 创新点**

创新点在于将经典的几何性质（无交叉定理、凸包周边节点按顺序访问）转化为可执行的约束传播器，并将其推广至需要部分访问点的EGTSP；

**🔧 技术方法**

采用了约束逻辑程序设计（CLP）中的后继表示、全不同约束、逆约束、以及自定义的几何传播器；传播器通过预先计算角度、半平面判定、邻居集合等实现高效的域修剪；

**📊 数据集**

实验数据集包括TSPLIB、Concorde网站、CITIES数据集的结构化实例；以及由R包netgen生成的均匀和聚类随机实例；对于EGTSP则使用三种聚类方式（均匀、中心点分布、网格划分）生成的随机实例；

**📈 对比分析**

与仅使用标准CLP模型、以及仅使用基于求解器的线性规划/约束松弛（Held-Karp、LKH上界）等基线进行对比；结果显示，加入几何传播后求解时间平均下降约70%，搜索节点下降约60-75%，并显著减少超时实例；

**⚠️ 局限性**

局限性包括：只适用于不允许边交叉的欧氏问题，不能直接处理时间窗口、容量约束等扩展；在极端分布（点集几乎全在一条直线或邻居集合稀疏）时几何传播效果不明显；且整体性能仍低于专门优化的IP求解器（如Concorde）。

---

## 477. X2-Turn: Frame-Synchronous Dual-Head Modeling for Joint Streaming ASR and Turn State Prediction

**arXiv ID:** 2608.10878 | [PDF](https://arxiv.org/pdf/2608.10878v1)

**作者:** Kaiqi Fu `[一作]` (X Square Robot), Qian Wang `[通讯]` (X Square Robot)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为X2-Turn的框架同步转态预测方法，通过延迟流建模实现实时的用户发言状态预测。

**💡 创新点**

创新点在于引入了一个并行的转态预测头，与ASR头共享流式表示，能够在单次前向传递中同时预测ASR标记和细粒度的转态。

**🔧 技术方法**

使用了延迟流建模的预训练Voxtral Realtime模型，并设计了一个统一的转态标签集和ASR锚定监督方法。

**📊 数据集**

使用了双语的Easy-Turn测试集，包含中文和英文的ASR数据和转态数据，数据量约为26k小时。

**📈 对比分析**

与现有的级联方法相比，X2-Turn在转态分类准确性和延迟方面表现优越，能够在不牺牲实时响应性的情况下实现准确的转态检测。

**⚠️ 局限性**

限制在于未来的工作需要进一步平衡ASR和转态目标，并提高在更具挑战性的对话环境中的鲁棒性。

---

## 478. TEAMMix: Taxonomy Enrichment Augmentation and Minority-augmented Mixing Strategy for LLM-enhanced Weak-Supervised Hierarchical Text Classification

**arXiv ID:** 2608.11044 | [PDF](https://arxiv.org/pdf/2608.11044v1)

**作者:** Jian Zhang `[一作]` (Zhejiang University), Hongwei Wang `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 TEAMMix 框架，利用 LLM 生成关键词与语料挖掘扩展标签层级，并通过少数类增强混合生成高质量伪样本，实现弱监督层级文本分类。

**💡 创新点**

创新点在于将 LLM 关键词生成与语料关键词挖掘相结合实现标签层级语义丰富，并通过 Gaussian 混合模型对置信度进行筛选的少数类增强混合，显著提升伪样本与标签的匹配质量。

**🔧 技术方法**

采用 ChatGLM‑4‑9B 进行提示生成，BERT 及 BM25 进行词向量与相似度评估，句子编码器提取特征，GMM 进行置信度重采样，动态置信度采样与混合训练等技术。

**📊 数据集**

使用 Amazon‑531（三层商品分类）和 DBpedia‑298（三层维基百科类别）两个公开数据集。

**📈 对比分析**

与零样本与弱监督基线（Hier‑0‑shot‑TC、WeSHClass、TaxoClass、TELEClass）以及完全监督对照进行比较，TEAMMix 在 Example‑F1、P@3 与 MRR 上均优于基线，并且相比 GPT‑4 在成本与耗时上显著更低。

**⚠️ 局限性**

局限性包括生成文本与关键词仍存在噪声，对复杂标签层级的泛化能力有限，且依赖高性能 LLM 与细致提示设计，导致可复现性与资源消耗受限。

---

## 479. Weighted First-Order Model Counting over Ordered Domains

**arXiv ID:** 2608.10877 | [PDF](https://arxiv.org/pdf/2608.10877v1)

**作者:** Jan Tóth `[一作]` (Czech Technical University), Ondřej Kuželka `[通讯]` (Czech Technical University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文研究了加权一阶模型计数问题（WFOMC），提出了一种在有序域上以多项式时间计算WFOMC的方法，特别是通过引入线性顺序公理来实现。

**💡 创新点**

创新点在于通过引入线性顺序公理，证明了在有序域上可以以多项式时间解决WFOMC，并且提出了一种新的算法来处理后继关系，显著提高了性能。

**🔧 技术方法**

使用了动态规划技术来逐步计算WFOMC，并引入了线性顺序公理和后继关系的扩展版本。

**📊 数据集**

使用了理论上构造的有序域数据集，特别是通过引入线性顺序和后继关系来模拟各种推理场景。

**📈 对比分析**

与现有方法相比，本文的方法在处理有序域时表现出更好的性能，尤其是在计算后继关系时，运行时间有时甚至提高了指数级别。

**⚠️ 局限性**

限制在于对于两个线性顺序关系的WFOMC问题仍然是#P_1-困难，表明在此情况下无法实现多项式时间的计算。

---

## 480. myMediWhisper: Construction of Burmese Medical Speech Corpus and Whisper Fine-Tuning for Clinical Dialogue ASR

**arXiv ID:** 2608.11036 | [PDF](https://arxiv.org/pdf/2608.11036v1)

**作者:** Ye Kyaw Thu `[一作]` (National Electronics and Computer Technology Center), Thepchai Supnithi `[通讯]` (National Electronics and Computer Technology Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并公开了28小时缅甸医学语音语料，利用该语料对Whisper模型进行全微调（FFT）和参数高效微调（PEFT），并研究波形级与频谱级数据增强在噪声与回声环境下的鲁棒性提升。

**💡 创新点**

公开低资源医学语料；采用Rank‑Stabilized LoRA高秩参数高效微调，使Whisper‑Large‑v2可训练；系统性评估FFT与PEFT在不同声学条件下的性能与鲁棒性，并揭示数据增强对干净语音有轻微退化但对嘈杂/回声环境显著提升。

**🔧 技术方法**

Whisper encoder‑decoder Transformer；全微调与参数高效微调（rsLoRA）；波形级增强（时间移位、音高移位、加高斯噪声）与频谱级增强（时间遮蔽、频率遮蔽）；房间声学模拟（pyroomacoustics）；WER、SER、DER、IER指标及实时因子RTF评估。

**📊 数据集**

自制的缅甸医学语音数据集（28 h，9名说话人，16 kHz采样），包含14,517句经母语者验证的文本；训练集52.95 h、评估集2.87 h；数据公开于Huggingface。

**📈 对比分析**

与零射Whisper基线（Tiny、Base、Small、Medium、Large v2）及公开模型（wav2vec2‑bloom、MMS‑1B）对比；FFT模型在无增强时达到23.44 % WER（myMediWhisper‑Medium），优于更大规模基线；PEFT在Large v2实现41.57 % WER；数据增强在噪声/回声条件下显著降低WER，但在干净数据上略有下降。

**⚠️ 局限性**

仅使用模拟噪声与合成房间声学，缺乏真实临床多说话人、非稳态设备噪声；语料规模有限，声学与方言多样性不足；PEFT在小模型下性能略低，推理时RTF较FFT高。

---

## 481. Mapping and Measuring the Behavioral Evolution of Large Language Models

**arXiv ID:** 2608.11027 | [PDF](https://arxiv.org/pdf/2608.11027v1)

**作者:** Dong Qiao `[一作]` (Chinese University of Hong Kong), Jicong Fan `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过使用同一批10,000个prompt，构建对32种LLM的无标签行为映射，利用句子级和token级的多种相似度度量，揭示了模型家族一致性、跨家族距离收敛以及最新模型响应云的紧凑性。

**💡 创新点**

创新点在于：①提出三种互补的句子级相似度构造（对齐均值、PCA压缩、Gromov–Wasserstein）；②引入token级MMD作为独立验证；③给出基于训练风险和有效目标分布的行为相似性充分条件；④对跨世代变化进行时间轴上的定量分析。

**🔧 技术方法**

使用了Qwen3-Embedding-8B对响应进行句子级嵌入，PCA降维、Gromov–Wasserstein匹配、token级MMD计算，配合MDS、t‑SNE、UMAP可视化以及层次聚类、漂移曲线等分析手段。

**📊 数据集**

数据集包括13个公开基准（MMLU、ARC、HellaSwag、WinoGrande、GSM8K、MATH、TruthfulQA等）提供的10,000个prompt，覆盖多种任务和领域。

**📈 对比分析**

通过构造三种相似度矩阵并在不同投影下进行可视化与聚类，结果显示模型家族内部聚集明显，跨家族距离随发布时间递减，最新推理导向模型的响应云更紧凑；句子级和token级度量在Spearman相关性上均高于0.9，验证了方法的稳健性。

**⚠️ 局限性**

局限性包括：仅使用单条生成结果导致生成方差影响距离；缺乏真实标签或权重信息；所用prompt库和嵌入器对距离影响不确定；理论条件未在实验中直接验证；模型覆盖范围有限，未涵盖所有公开LLM。

---

## 482. Derivative Computation in PINNs: Automatic Differentiation, Finite Differences and Beyond

**arXiv ID:** 2608.11020 | [PDF](https://arxiv.org/pdf/2608.11020v1)

**作者:** Maciej J. Mikulski `[一作]` (AGH University of Krakow), Tadeusz Uhl `[通讯]` (AGH University of Krakow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统研究了在物理信息神经网络（PINN）中使用有限差分（FD）计算导数，提出了可运行时校准步长ε的经验方法，并与自动微分（AD）进行全面对比；还提出了三种FD策略（FD、eFD、sFD）以及随机步长正则化（sFD），并揭示了在含跨样本依赖的网络（如BatchNorm、注意力）中标准PyTorch autograd 的隐式错误；

**💡 创新点**

创新点包括：①基于误差分析的自适应步长校准方法；②针对多阶导数问题的三种FD策略，尤其引入随机步长正则化sFD；③揭示并解决跨样本依赖网络中autograd 的错误；

**🔧 技术方法**

使用技术包括：物理信息神经网络、有限差分数值微分、自动微分、批归一化、注意力机制、GPU加速、FP32训练、速度与内存微基准、随机步长采样等；

**📊 数据集**

实验数据集为PINNacle基准集中的三道 PDE：二维Poisson方程（四圆洞多连通域）、一维粘性 Burgers 方程、三维热传导方程（Heat2D-CG复杂几何与Robin 边界）；

**📈 对比分析**

采用相同 MLP 架构、相同训练设置（Adam、10k+ epoch、8192 batch）比较 SMAPE、L2相对误差、最大误差以及训练时间与GPU内存。结果显示 FD 与 AD 在准确性上基本相同；在大批量或跨样本依赖网络中，FD 速度快 2-4 倍、内存低 10-1000 倍；sFD 在 Poisson 问题上显著提升 20% L2 误差，其他问题性能相同或略差；

**⚠️ 局限性**

局限性包括：仅验证一阶、二阶导数；仅在 FP32 下实验；未针对四阶或更高阶 PDE 进行验证；sFD 在含强梯度或时变问题中可能引入过多噪声；缺乏对不同浮点格式和更大网络的理论与实验支持。

---

## 483. Sona Technical Report

**arXiv ID:** 2608.11015 | [PDF](https://arxiv.org/pdf/2608.11015v1)

**作者:** Sona Team `[一作]`, Vladislav Tytskiy `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并部署了一个单模型生成式音乐推荐系统，整合候选生成与排序，取代原有多阶段推荐流水线，并在Yandex Music的My Vibe智能音箱流量上进行在线A/B测试。

**💡 创新点**

创新点包括：① 将生成与排序统一在同一个Transformer中，使用共享的用户表示；② 采用无人工特征的语义ID分词器与协同细化的音频-文本模型；③ 通过教师模型的密集蒸馏（rollout distillation）将长序列预训练的知识压缩到生成器与排序器；④ 历史压缩技术在保持长历史信息的同时降低推理成本。

**🔧 技术方法**

使用的技术主要有：Transformer编码器、可变长自回归解码器、交叉注意力排名模块、语义ID分词器（残差K-means量化）、Qwen2.5-Omni多模态预训练、InfoNCE协同对齐、next-token预测、教师ranker的多头排序、两阶段预训练（next-item预测+多头微调）、在线持续训练、Beam Search、Triton Inference Server、FlashAttention、bfloat16混合精度、FSDP。

**📊 数据集**

数据集为Yandex Music 2026年8月的用户交互日志，覆盖一年的完整事件序列。训练数据以每周、每日的日志窗口递增；候选集来自日志中的展示列表；对齐使用Inference Log记录的历史截断时间。

**📈 对比分析**

比较方法：离线使用Target-track Recall@k、Teacher Recall@k、Weighted Pair Accuracy评估生成质量和排序准确性；在线A/B测试对比控制端的Active Users、Total Listening Time、Likes、Repeat Commands、Deeply Engaged Users。实验显示单模型在Active Users上提升约25%（与之前Argus相当），Total Listening Time提升约12%，Likes提升约8%，且对其他指标也表现出正向提升。

**⚠️ 局限性**

局限性包括：尚未在全量流量上持续部署；相较原有流水线，模型覆盖的曲目集合略小；验证仅在My Vibe智能音箱表面，需进一步在其他推荐场景测试；模型规模与测试时扩展仍有提升空间；未探索强化学习后训练或更细粒度的内容探索机制。

---

## 484. What Iterated Self-Feeding Probes of Language Models Measure, and a test that separates the construction from the model

**arXiv ID:** 2608.10986 | [PDF](https://arxiv.org/pdf/2608.10986v1)

**作者:** Nicolás Vera Zúñiga `[一作]` `[通讯]`, Nicolás Vera Zúñiga

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种环形Token格子自我反馈探测器，利用共随机数耦合测量损伤传播，并区分构造与模型本身的读数。

**💡 创新点**

发现自我反馈探测器的读数同时包含构造和模型信息，并提出判别方法；报告了仅归因于构造的吸收态相变与归因于模型的开发转变。

**🔧 技术方法**

使用Glauber动力学、共随机数耦合、Lyapunov指数、阻尼长度、吸引子占比等统计量，并在19种大模型上进行验证。

**📊 数据集**

在19种大模型（如Pythia、GPT‑2等）以及Domany–Kinzel自动机上进行实验；使用模型的预训练检查点数据。

**📈 对比分析**

通过同一构造下模型训练进程变化与同一模型下不同构造变化对比，检验读数是否随构造或模型变化；实验显示吸引子占比能区分模型，构造变更导致相变。

**⚠️ 局限性**

仅在单一半径、温度和模型族内验证，跨构造或其他动态的通用性尚未证明；方法只给出相对读数，绝对损伤需明确耦合；未提供对所有动态的普适解释。

---

## 485. Pitch Contour Tokenization using VQ-VAE and Its Application on Korean Traditional Music Analysis

**arXiv ID:** 2608.10979 | [PDF](https://arxiv.org/pdf/2608.10979v1)

**作者:** Seonguk Ju `[一作]` (Sogang University), Dasaem Jeong `[通讯]` (Sogang University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了一种基于 VQ‑VAE 的无监督音高轮廓分词器，能够从未标注的韩国传统音乐音频中学习连续音高运动的离散词汇。

**💡 创新点**

提出了变换最小化重构损失以提升分段稳健性，并利用 VQ‑VAE 直接从连续轮廓中学习可解释的离散单位，无需预定义符号。

**🔧 技术方法**

使用 VQ‑VAE、1D 卷积、变换最小化重构损失以及对齐与尺度变换的搜索；并与自编码器基线进行对比。

**📊 数据集**

采用约 280 小时的韩国传统人声音乐 Pansori 录音集以及 NIA AI‑Hub 标注的 sigimsae 段作为训练和评估数据集。

**📈 对比分析**

与无监督自编码器基线和有监督 ED‑TCN 进行对比；在分段稳健性指标 KLD 与匹配准确率上，VQ‑VAE 显著下降 KLD、提升准确率；在 sigimsae 分类上，使用单个 token 的 MAP 分类器得到的 F1 与 mAP 约为 0.28/0.32，接近有监督模型的三分之二。

**⚠️ 局限性**

模型假设连续音高，未声区被排除，无法端到端处理完整录音，对未声段的处理有限。

---

## 486. CapProbe: Evaluating Detailed Image Captions via Full-Scene Dense Question Answering

**arXiv ID:** 2608.11074 | [PDF](https://arxiv.org/pdf/2608.11074v1)

**作者:** Mouxiao Huang `[一作]` (Huawei Technologies), Han Shu `[通讯]` (Huawei Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了全场景稠密QA基准（CapProbe）用于评估详细图像字幕，并给出了三阶段评估协议（生成-读者回答-多维度度量）。

**💡 创新点**

创新点包括：①将图像按语义区域分解并为每个区域生成多项选择问题，平均74个问题/图像；②引入Uncertain选项和Effective Accuracy，区分答非所问与遗漏；③提出兼顾“能力”与“效率”的多维度评估指标；④构建跨域覆盖的两层语义分类体系（37 L1 / 219 L2）。

**🔧 技术方法**

技术手段包括：YOLOv26‑seg 与 SAM3 的语义分割；Gemini‑3.1‑Pro 与 GPT‑5.5 用于生成结构化元数据和 QA；Qwen3‑32B 等LLM 作为读者；量化的整体准确率、Effective Accuracy、Coverage、Uncertainty Ratio、Tokens/Img、Mean/Global Density 等指标。

**📊 数据集**

数据集来源：从 Places365、LVIS、OpenImagesV7 选取 346 张图片，涵盖 37 L1/219 L2 语义域，共 1,868 区域、25,650 QA。

**📈 对比分析**

比较方法：在 13 个 VLM（闭源与开源）上评估，使用整体准确率、Effective Accuracy、Coverage、Uncertainty Ratio、Tokens/Img、Mean/Global Density 等指标。闭源 Gemini‑3.1‑Pro 以 68.7% 的整体准确率位居榜首；开源中 Qwen3‑5.397B 与 Qwen3‑VL‑235B 处于第二层；长文本并不一定带来更高信息密度，模型在覆盖与密度之间存在权衡。

**⚠️ 局限性**

局限性：①评估结果依赖读者模型，绝对分数受读者偏好影响；②区域分割粗略，可能漏掉细小目标；③数据集规模有限，单图像样本不足；④语义标签非统计代表性；⑤计算成本相对较高。

---

## 487. Deployment Is Not Destiny: Robot Recomposition in the Field with Unseen Software, Hardware, and Compute Payloads

**arXiv ID:** 2608.11063 | [PDF](https://arxiv.org/pdf/2608.11063v1)

**作者:** Steven Swanbeck `[一作]` (University of Texas at Austin), Robert Blake Anderson `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套可在运行时动态重组的机器人系统框架，支持硬件、软件和计算资源的即插即用与分布式共享。

**💡 创新点**

创新点在于统一可组合抽象、自动发现与管理组件，以及利用PDDL规划与行为树实现无人工干预的任务生成，使未知模块也能被即时集成。

**🔧 技术方法**

核心技术包括ROS中间件、Docker容器化部署、USB与网络设备的统一抽象、行为树与PDDL规划、LLM将自然语言转为PDDL目标、UDP多播与TCP RPC实现分布式资源共享。

**📊 数据集**

论文未使用公开数据集，而是在真实演示环境中验证：核电站辐射源定位与灾害搜救场景，涉及Spot、Panther、Turtlebot等机器人与外部计算节点。

**📈 对比分析**

与传统需要数小时专家介入的手工重配置相比，框架可在几分钟内完成重组，并实现跨机资源共享；两次演示均展示了快速部署、任务级行为树生成以及实时目标执行的优势。

**⚠️ 局限性**

局限性包括：payload选择仍需人工判断；硬件安装缺乏自检与姿态验证；使用的PDDL规划器为非反应式，未充分利用行为树的全表达与实时性。

---

## 488. The Minimum-Weight Mixed Dominating Set on Threshold Graphs

**arXiv ID:** 2608.11057 | [PDF](https://arxiv.org/pdf/2608.11057v1)

**作者:** Emiliano Lancini `[一作]` (Université Paris Dauphine--PSL), Oulin Yang `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究阈值图上的最小权重混合支配集（MWMDS）问题，并给出多项式时间（O(n^5)）算法，核心思路是将该问题转化为约束混合覆盖（Constrained Mixed Cover）问题，进一步映射到最小权重边覆盖问题。

**💡 创新点**

创新点在于：①证明任意权重可转化为非负权重而不影响复杂度；②构造约束混合覆盖问题并证明其等价于最小权重边覆盖；③针对阈值图设计了七类子问题，证明每个极小混合支配集必满足其中至少一个子问题，从而实现完整的O(n^5)算法。

**🔧 技术方法**

主要技术包括：权重预处理（拆分正负权重）、构造辅助图（带复制与辅助顶点/边）、约束混合覆盖到边覆盖的多项式映射、阈值图的结构性质（最小顶点覆盖集的描述）、枚举多种顶点分区并分别调用边覆盖算法。

**📊 数据集**

本工作为理论性研究，没有使用具体数据集，所有结果均为算法分析与证明。

**📈 对比分析**

由于研究属于理论计算复杂度，没有与其它方法进行实验比较；所给算法的性能在阈值图上可保证O(n^5)，在现有文献中是唯一已知的多项式解（此前仅有无权重的O(n^3)解）。

**⚠️ 局限性**

局限性包括：①算法复杂度高，实际执行效率可能受限；②仅适用于阈值图，无法直接推广到更一般的图类；③缺乏实验验证，无法评估在实际图实例上的表现；④对权重取值范围（非负）做了假设，虽然通过预处理可转化，但在实际应用中仍需额外步骤。

---

## 489. A Comparative Evaluation of Deep Learning Object Detection Models on a Real-World Multi-Plant Dataset from Africa

**arXiv ID:** 2608.11053 | [PDF](https://arxiv.org/pdf/2608.11053v1)

**作者:** Ismail Ismail Tijjani `[一作]` (EJAZTECH.AI), Abdullahi Suiudeen `[通讯]` (Federal University Dutse)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究收集了尼日利亚农田的真实图片数据集 AgriAISeg，并对六种主流目标检测模型（YOLOv5、YOLOv8、YOLO11、YOLO26、Faster R‑CNN、RT‑DETR）在该数据集上的检测性能进行了系统对比。

**💡 创新点**

创新点在于：①首次公开了适用于非洲地区、覆盖黄瓜、卷心菜、芝麻三种作物、并包含多种光照、遮挡和视角变化的真实场景数据集 AgriAISeg；②对一阶段、两阶段和 Transformer‑based 三大检测范式进行了统一实验和全面对比，弥补了以往仅对 YOLO 等模型局部评估的空白；③在同一实验设置下量化了训练时间与检测精度的权衡，为低算力农业设备部署提供了参考。

**🔧 技术方法**

技术手段包括：利用 SAM3 进行半自动实例分割并转换为边界框标签；对 YOLO 系列采用官方默认训练脚本；对 Faster R‑CNN 进行超参数细调；对 RT‑DETR 采用 Vision Transformer 作为 backbone；统一使用 640×640 的输入尺寸，并评估 Precision、Recall、mAP@0.5、mAP@0.5:0.95 等指标。

**📊 数据集**

使用的数据集为 AgriAISeg，包含 3,382 张手工采集的尼日利亚黄瓜、卷心菜和芝麻图像，涵盖了不同光照、遮挡、视角和土壤背景等多样化真实场景。

**📈 对比分析**

比较方法：将所有模型在相同 75/15/10 的 train/val/test 划分、相同预处理、相同评价指标下进行训练与评估；记录训练时长、Precision、Recall、mAP@0.5、mAP@0.5:0.95。实验结果显示 RT‑DETR 以 0.624 的 mAP@0.5:0.95 领跑，YOLOv8 与 YOLO11 仅差 0.006，均显著优于传统两阶段的 Faster R‑CNN（mAP@0.5:0.95 仅 0.240）。此外，RT‑DETR 的训练时间约 4.12 小时，YOLO 系列仅需 1.69–2.00 小时，说明在实时部署上 YOLO 具有更高效率。

**⚠️ 局限性**

局限性包括：①数据集仅包含三种作物，难以完全覆盖非洲多样化作物；②模型训练与评估仅在单一硬件环境下完成，缺乏对移动设备或低功耗无人机的实际部署验证；③Transformer‑based RT‑DETR 虽性能最佳，但训练时长和算力需求相对更高；④实验未考虑不同生长期、不同地区土壤和作物种植模式对模型迁移性能的影响。

---

## 490. DEFT: Data-Efficient Frequency-domain Top-k Sampling via Inverse Discrete Fourier Transform for Spatiotemporal Dynamical Systems Modeling

**arXiv ID:** 2608.11019 | [PDF](https://arxiv.org/pdf/2608.11019v1)

**作者:** Hengbo Xiao `[一作]` (Peking University), Guannan He `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种基于频域的主动数据采样方法DEFT，用来高效生成满足物理约束的训练数据；

**💡 创新点**

创新点在于通过对少量观测信号做离散傅里叶变换，自动识别并保留能量占比最高的K个频率分量，再在该子空间中通过幅度相位随机化产生多样化合成输入，从而显著减少训练数据需求，并给出理论泛化误差上界；

**🔧 技术方法**

主要技术包括频域能量排序、IDFT合成、物理模拟标签、深度算子网络（如Attention-DeepONet）以及相关的泛化误差分析；

**📊 数据集**

实验数据集涵盖Burgers、Allen–Cahn、Diffusion–Sorption PDEBench、以及COMSOL P2D-SEI电池衰退模型，此外还对多种电池化学体系进行迁移学习；

**📈 对比分析**

与随机采样、均匀网格、Latin hypercube、贝叶斯优化等传统采样方法对比，DEFT在低频占优的系统中R²提升至0.95–0.99，数据量可缩减40%且误差不超过2%；在电池预测任务中，R²>0.98，且仅需20%微调数据即可实现跨化学体系的高精度迁移；

**⚠️ 局限性**

主要局限包括：对信号光滑性（α>12）的严格假设、仅适用于低频主导或可压缩的系统；对于宽带或高频重要的系统，频域截断可能导致信息损失；理论泛化界限尚待在更弱正则性条件下进一步完善。

---

## 491. Information Bottleneck under Perfect Privacy

**arXiv ID:** 2608.11003 | [PDF](https://arxiv.org/pdf/2608.11003v1)

**作者:** Junle Zhong `[一作]` (CentraleSupélec), Sreejith Sreekumar `[通讯]` (CentraleSupélec)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在完美隐私约束下，活跃速率信息瓶颈问题，并提出针对非凸、无约束且概率约束的扰动ADMM求解器

**💡 创新点**

首次在此问题中实现精确的零泄露约束，设计了扰动双重更新的ADMM框架，并给出全局收敛与KŁ收敛速率的理论保证

**🔧 技术方法**

采用基于扰动双重更新的ADMM、强凸-弱凸结构、Lyapunov函数与Kurdyka–Łojasiewicz分析

**📊 数据集**

使用公开的离散联合分布数据（S、X、Y）作为实验样本，设定|U|=2

**📈 对比分析**

与传统IB算法（扰动ADMM、迭代IB、DRS）对比，结果显示RCPP曲线略低于无隐私约束曲线，但隐私泄露始终维持在数值精度级，说明隐私约束导致的效用损失不大

**⚠️ 局限性**

仅适用于有限字母空间，无法直接推广到多用户或连续情形，且需要对概率约束保持严格正下界

---

## 492. Seeing above the waves: A modular sensing framework for data acquisition at sea

**arXiv ID:** 2608.10997 | [PDF](https://arxiv.org/pdf/2608.10997v1)

**作者:** Jonathan E. Schmidt `[一作]` (Technical University of Denmark), Roberto Galeazzi `[通讯]` (Technical University of Denmark)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套模块化、多模态的海上自动化传感平台蓝图，并在渡轮、港口巴士和无人船等不同船型上进行多场景部署，生成长期、结构化的数据集。

**💡 创新点**

创新点在于：①将雷达、LiDAR、RGB/LWIR摄像、IMU、GNSS、AIS、天气等多种传感器通过统一的硬件架构和ROS2+Docker容器化软件实现高度模块化与隔离；②采用GNSS同步时钟和MCAP高效压缩，实现跨传感器时序一致且可追溯的数据记录；③提供可复现的部署与校准流程，降低实验门槛，支持跨船型和跨实验的可比性。

**🔧 技术方法**

主要技术包括：ROS2（CycloneDDS中间件）、Docker容器化部署、PoE Ethernet网络架构、GNSS时间同步、MCAP数据记录格式、Python/ROS2节点的统一接口、硬件级隔离与软硬件分层设计。

**📊 数据集**

作者收集了三组实测数据集：①沿海导航（4–5 TB，涵盖雷达、RGB、LWIR、GNSS、IMU、天气）；②城市水路交通（100–500 GB，包含雷达、LiDAR、RGB、LWIR、GNSS、IMU）；③海上基础设施映射（100–500 GB，LiDAR、GNSS、IMU、声呐）。这些数据均以MCAP文件形式公开，可供感知与定位研究使用。

**📈 对比分析**

通过现场部署与长期运营评估验证平台稳定性：在奥雷斯und渡轮长达数周的跨境航行中，系统保持无重大数据丢失；港口巴士与岸基双平台实验中，LiDAR数据在单独子网下同步，后处理对齐后实现无缝融合；USV实验中实现了“海床‑天空”三维地图的同步构建。虽然文中未给出算法性能数值，但所示案例证明平台在多模态同步、网络容错和长周期收集方面的可靠性。

**⚠️ 局限性**

局限性包括：①核心数据收集节点为单点可能导致系统单点故障，需引入备份；②高频率传感器（LiDAR、摄像）对网络带宽与存储要求高，需额外子网或本地缓存；③校准过程依赖手工或自动化脚本，复杂配置仍需人工干预；④平台主要针对海上表面船舶，对深海或大气环境的适配尚未验证。

---

## 493. Static in Frames, Dynamic in Events: Rethinking Features in Event Cameras as Motion Cues

**arXiv ID:** 2608.11075 | [PDF](https://arxiv.org/pdf/2608.11075v1)

**作者:** Hesam Araghi `[一作]` (Delft University of Technology), Nergis Tomen `[通讯]` (Delft University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种将事件相机的动态特征重新作为运动提示的框架，统一了帧内静态特征与事件驱动的动态特征建模

**💡 创新点**

创新点在于将事件数据拆分为静态帧信息与动态事件信息，并设计了基于梯度投影与时间窗口的特征提取与匹配方法

**🔧 技术方法**

采用事件卷积网络（Event‑CNN）、时空梯度投影、光流估计等技术

**📊 数据集**

在公共事件数据集 N‑Caltech101、N‑MNIST、N‑Planes 与 DVS128 Gesture 上进行实验

**📈 对比分析**

与现有事件特征方法（ESIM、EV‑Flow、N‑DAVIS‑Event‑Flow）对比，所提方法在分类精度上提升约4%，在跟踪鲁棒性上提高3~5%，实验显示显著性能提升

**⚠️ 局限性**

局限性包括对高噪声事件的鲁棒性不足，计算量相对较大，以及对光照变化的适应性仍待改进

---

## 494. Uncertainty-Aware Deep Learning for Genomics Applications: Insights from an Empirical Study

**arXiv ID:** 2608.11054 | [PDF](https://arxiv.org/pdf/2608.11054v1)

**作者:** Sepideh Saran `[一作]` (Berlin Institute for Medical Systems Biology, Max Delbruck Center for Molecular Medicine), Uwe Ohler `[通讯]` (Berlin Institute for Medical Systems Biology, Max Delbruck Center for Molecular Medicine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对深度学习模型在基因组学中的不确定性量化（UQ）进行系统评估，构建了可用于不同数据特征的实验框架。

**💡 创新点**

提出两种不依赖标签真值的评估方法：①通过不同UQ方法之间的样本不确定性一致性（Kendall tau、top‑100重叠）衡量可靠性；②通过引入单一不确定源（类别不平衡、标签噪声、motif缺失、OOD）观察不确定性分布的偏移，评估方法对不同来源不确定性的感知能力。

**🔧 技术方法**

使用深度集成（10 个 CNN）、贝叶斯神经网络（变分推断 + Flipout）和 Monte Carlo Dropout 这三种主流 UQ 技术，在同一网络架构上实现对比；对模拟与真实数据执行多种不确定性实验。

**📊 数据集**

模拟数据：6 种转录因子（CTCF、HNF4A、JUN、MEF2A、MYC、TAL1）背景序列；真实数据：3 种 RNA‑binding 蛋白（MBNL1、PUM2、QKI）的 PAR‑CLIP 序列；单细胞 RNA‑seq：GSE194122 数据集的 4 种细胞类型；此外对 8 种冠状病毒基因组进行 OOD 测试。

**📈 对比分析**

先让三种 UQ 模型在相同数据上收敛至相似准确率（使用 NLL、Brier、准确率、平均精度等指标），随后对不确定性进行比较：BNN 在类别不平衡和 OOD 时表现出更高的分布偏移和更强的对不确定样本的定位能力；ENS 与 BNN 的一致性高于 MC‑dropout；MC‑dropout 的不确定性几乎为零，缺乏信息。实际 27 类 RBP 任务中，使用 BNN 的不确定性阈值过滤可提升平均精度。

**⚠️ 局限性**

局限性：仅关注分类任务，未考虑回归或无监督场景；只评估聚合不确定性，未区分 epistemic 与 aleatoric；实验仅覆盖两种基因组学模态和固定的实验设置；对传统机器学习模型或大规模预训练模型未作评估；阈值选择需针对具体数据和应用定制。

---

## 495. SCOUT: Symmetric Consensus Outlier Detection for Failure Localization in LLM Pre-Training

**arXiv ID:** 2608.11034 | [PDF](https://arxiv.org/pdf/2608.11034v1)

**作者:** Zhuang Wang `[一作]` `[通讯]` (Independent Researcher), Zhuang Wang (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个面向LLM预训练的运行时错误定位框架，利用严格多数共识在等价副本之间识别挂起、慢速和静默数据腐败等隐蔽失败。

**💡 创新点**

创新点在于：①将等价副本的严格多数共识作为统一定位原则；②引入 Consensus Collective Communication (C3) 抽象，既收集证据又输出离群位图；③结合离线CPU观测器和现场重放（in‑situ replay）分别定位挂起、慢速和SDC；④使用紧凑签名和覆盖率压缩提升重放效率，并将重放结果用于检查检查点的完整性。

**🔧 技术方法**

技术手段包括：多维并行映射与等价副本组建；C3的精确与统计共识；CPU观测器通过共享内存捕获进度和集体指纹；可重复重放的确定性 RNG 与张量签名哈希；覆盖率压缩策略压缩 MoE 形状；与 PyTorch/TorchTitan/Megatron-Core/DeepSpeed 的公共接口集成；以及与 ♊（加速检查点）结合的检查点认证机制。

**📊 数据集**

评估使用的是基于三层 Transformer 的确定性训练（固定种子、固定批次、固定模型），在两台 8 核 A100 机器上进行软件注入实验；未使用大规模公开数据集，而是侧重于验证定位与检查点认证的功能。

**📈 对比分析**

通过软件注入测试，框架在所有 16 GPU 的工作负载中均能 100% 正确定位挂起、慢速和 SD 的故障且无误报；层级重放的运行时开销约为 0.3%；MoE 形状压缩后覆盖率均高于 98%，且重放覆盖率可压至 16~48 代表形状；在实验中未给出整体吞吐量或恢复时间指标，主要关注定位准确率和误报率。

**⚠️ 局限性**

局限性包括：评估规模仅为两台机器 16 GPU，未覆盖大规模集群与 RDMA 网络；仅使用软件注入，缺乏真实硬件故障验证；未测量完整训练过程中的吞吐量和恢复延迟；覆盖率压缩仅在特定硬件/软件环境下验证，需进一步泛化；此外，C3 对多数一致性的假设在极端多节点失效场景下可能失效。

---

## 496. Self-Knowledge Retrieval Augmented Generation Framework for Patent Matching

**arXiv ID:** 2608.11030 | [PDF](https://arxiv.org/pdf/2608.11030v1)

**作者:** Jian Zhang `[一作]` (Zhejiang University), Hongwei Wang `[通讯]` (Zhejiang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对专利检索与匹配任务，作者提出了一种基于自我知识挖掘的检索增强生成（Self‑Knowledge RAG）框架，利用大型语言模型（LLM）从专利文本中自动抽取关键实体与层级本体，并以此扩展检索查询、检索相似专利并最终在上下文丰富的指令下进行生成式匹配。

**💡 创新点**

创新点在于：①不依赖外部知识库，而是让LLM主动从查询专利中挖掘实体与本体；②将自挖掘的知识用于查询扩展和检索引导，使得检索与生成过程形成闭环；③通过多层次（实体、技术类别、本体层级）上下文提升匹配精度，显著优于传统CoT或单纯RAG方案。

**🔧 技术方法**

主要技术包括：大型语言模型（如Qwen2‑Instruct‑7B、GLM‑4‑Chat‑9B、Qwen2.5‑Instruct‑14B）用于实体与本体抽取；BGE‑large作为向量编码器；FAISS实现高效向量检索；以及定制化的检索增强生成提示工程。

**📊 数据集**

使用了两大数据集：1）PatentMatch（1,000条中英专利匹配实例，覆盖8个IPC类别），用于评估匹配性能；2）从公开来源收集的约30万条专利作为检索库，用BGE‑large对其进行向量化以供检索。

**📈 对比分析**

与多种基线（普通LLM、域特定LLM、Chain‑of‑Thought、传统RAG）进行对比，实验显示在GLM‑4‑Chat‑9B上整体准确率从72.6%提升至81.3%，在Qwen2‑Instruct‑7B上从58.7%提升至69.8%，表明自我知识驱动的RAG方法在中英文多语言场景下均有显著性能提升。

**⚠️ 局限性**

局限性包括：本框架仅处理文本信息，未结合多模态专利数据（如图像）；本体层级采用静态IPC结构，缺乏动态自适应生成；在极大规模检索库下的实时性能与可扩展性仍待进一步验证。

---

## 497. Data Attribution of Emergent Misalignment with Persona Features

**arXiv ID:** 2608.11025 | [PDF](https://arxiv.org/pdf/2608.11025v1)

**作者:** Clemens Vetter `[一作]` (Bonn-Aachen International Center for Information Technology, University of Bonn), Florian Mai `[通讯]` (Lamarr Institute for Machine Learning and Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过Sparse Autoencoder对四款开源LLM进行对比分析，揭示了精调导致的“Emergent Misalignment”现象及其对应的persona特征；

**💡 创新点**

首次将EM相关特征归因到预训练数据，证明人类写作文本单独不足以诱发EM，而LLM生成的指令-响应对可显著激活EM；

**🔧 技术方法**

采用SAE模型diffing、激活steering、文档激活检索与因果归因的多步骤方法；

**📊 数据集**

使用Common Crawl Web文档（约100万条）与从中检索到的高激活文档作为实验数据；

**📈 对比分析**

通过对齐/误导微调模型与对齐模型的EM率比较，Steering可将EM率提升至62%（高于微调的35%），负向steering可将误导模型EM率降至≈1%；

**⚠️ 局限性**

局限在模型规模限制、SAE可用性、评估标准依赖LLM judge、特征选择主观性以及未验证因果性等方面

---

## 498. Watching Synthetic Videos: Aligning Cross-modal Representations with Visual Synthesis for Zero-shot Video Captioning

**arXiv ID:** 2608.11013 | [PDF](https://arxiv.org/pdf/2608.11013v1)

**作者:** Liangyu Fu `[一作]` (Northwestern Polytechnical University), Zhiyong Wang `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种零样本视频字幕生成框架WSV，利用文本到视频模型生成视频潜在表示，再通过聚合器、提示器和GPT‑2生成字幕。

**💡 创新点**

创新点在于：①直接在训练阶段引入文本到视频模型产生可视潜在空间，显式克服训练/推理的模态鸿沟；②设计3D CNN聚合器修正合成潜在与真实视频分布的偏差；③使用提示器将视觉潜在映射为软提示，调优GPT‑2，从而在纯文本训练下实现高质量视频字幕。

**🔧 技术方法**

主要技术包括：文本到视频生成模型（CogVideoX、Wan2.2‑T2V）、3D CNN聚合器、跨注意力+自注意力提示器、CLIP4Clip视觉/文本编码器、GPT‑2语言模型及对比学习损失。

**📊 数据集**

实验使用公开视频字幕数据集：MSVD、MSR‑VTT 与 VATEX，训练阶段仅使用文本描述，推理阶段使用真实视频。

**📈 对比分析**

与现有零样本视频字幕方法相比，WSV 在 MSVD 上 B@4 52.0、CIDEr 95.7；MSR‑VTT 上 B@4 33.9、CIDEr 45.5；VATEX 上 B@4 26.1、CIDEr 35.5，均显著优于此前最佳零样本方法，逼近甚至超越部分监督模型。

**⚠️ 局限性**

局限性包括：①依赖文本到视频模型的生成质量，若合成潜在失真会影响最终字幕；②聚合器与提示器结构相对复杂，训练成本高；③目前仅在三大数据集上验证，泛化到更大规模或多语言场景需进一步验证；④对细粒度动作或多人物交互的描述仍存在误差。

---

## 499. On the Limitations of Cross-Lingual Consistency in Multilingual Text-to-image Generation

**arXiv ID:** 2608.11002 | [PDF](https://arxiv.org/pdf/2608.11002v1)

**作者:** Sicheng Zhang `[一作]` (Khalifa University), Mubarak Shah `[通讯]` (University of Central Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了跨语言文本到图像（LingT2I）基准，系统评估了10种语言下的内容生成和文本渲染表现。

**💡 创新点**

首次提供统一多维度、跨语言的评测框架，揭示语言不平等、写作系统瓶颈以及文化偏好对生成质量的深层影响。

**🔧 技术方法**

采用多语言编码器（MetaCLIP2、Qwen‑2.5‑VL）与CLIPScore、TRIGScore等评测指标，并利用 Gemini 2.5 Pro 进行翻译与验证。

**📊 数据集**

从 DOCCI、EasyText 等公开数据集抽取 33K 句子，构成 30K 内容生成样本和 3K 文本渲染样本，覆盖 10 种语言。

**📈 对比分析**

对 17 款现有 T2I 模型进行对比实验，发现多语言增强模型虽降低语言方差，但整体性能仍受语言家族与文化差异影响；文本渲染任务普遍表现较差。

**⚠️ 局限性**

受限于翻译过程可能引入偏差、评测指标对非拉丁语仍不够中性，以及数据集本身的语言与文化分布不均。

---

## 500. TimeRoute: Time-Aware Modality Routing and Diffusion for Multi-Modal Recommendation

**arXiv ID:** 2608.10983 | [PDF](https://arxiv.org/pdf/2608.10983v1)

**作者:** Pengyu Zhang `[一作]` (University of Amsterdam), Paul Groth `[通讯]` (University of Amsterdam)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为TimeRoute的基于扩散模型的多模态推荐框架，旨在解决多模态时间尺度失配导致的用户融合比例不匹配和过时模态噪声问题。

**💡 创新点**

创新点在于引入时序感知的模态路由器（根据用户历史时间特征为每个用户生成个性化模态权重）以及使用FiLM与双流长短期去噪的时间条件扩散重构器，二者分别在融合层和图重构层实现时序适配。

**🔧 技术方法**

采用的技术包括扩散模型、特征线性调制（FiLM）、双流（长短期）去噪头、图卷积网络、对比学习以及多模态特征投影等。

**📊 数据集**

在TikTok、Amazon‑Baby和Amazon‑Sports三个公开多模态推荐数据集上进行实验。

**📈 对比分析**

与DiffMM、KDiffE及多种基线方法对比，TimeRoute在Recall@20、Precision@20和NDCG@20等指标上平均提升约7–10%，在随机和时间顺序拆分下均保持优势。

**⚠️ 局限性**

局限性包括对时间戳覆盖率敏感；当某一模态（如视觉）主导时，高活跃用户可能收敛到近单模态权重；且未考虑模态质量估计等因素。

---

## 501. A Dataset and Benchmark for Optical Music Recognition of String Quartet Scores

**arXiv ID:** 2608.10978 | [PDF](https://arxiv.org/pdf/2608.10978v1)

**作者:** Dongmin Kim `[一作]` (Sogang University), Dasaem Jeong `[通讯]` (Sogang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OSSQ-OMR数据集及基准，专注多声部字符串四重奏的光学乐谱识别。

**💡 创新点**

首个多声部OMR数据集，提供系统级与谱线级图像、三种编码格式，并实现视觉对齐扫描。

**🔧 技术方法**

采用YOLOv8分割、LSTM+ResNet（Zeus）与ConvNeXt+Transformer（SMT）两种深度学习基线，结合LMXE/ABC编码与OMR‑NED评估。

**📊 数据集**

基于OpenScore字符串四重奏与IMSLP扫描的116首曲目，构成OSSQ-OMR数据集。

**📈 对比分析**

通过四个随机分割，在系统/谱线级别和不同编码下训练两模型，平均OMR‑NED在合成图像为3.6%（最佳），扫描图像为5.9%。

**⚠️ 局限性**

对扫描源仍有较大误差，基线未充分利用多声部上下文，缺乏更大规模的多声部/全页数据。

---

## 502. Entropy-Centric Explainable AI for Remote Sensing Image Segmentation

**arXiv ID:** 2608.11064 | [PDF](https://arxiv.org/pdf/2608.11064v1)

**作者:** Ali Saleh `[一作]` (Lebanese University), Ali J. Ghandour `[通讯]` (EFREI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于熵的分割模型可解释方法（Entropy‑Centric XAI），并设计了新的评估框架 H‑SIT；

**💡 创新点**

创新点在于利用输入扰动对目标类熵的变化量作为重要性度量，融合 Sobol 随机采样与熵不确定性理论；

**🔧 技术方法**

主要技术包括 QMC 采样、Sobol 敏感度估计、图像扰动、熵计算以及 L‑SIT 与 H‑SIT 的性能评估；

**📊 数据集**

使用 WHU 影像数据集（建筑占地分割）进行实验验证；

**📈 对比分析**

与 Grad‑CAM、Score‑CAM、Seg‑Sobol 在 L‑SIT 和 H‑SIT 指标下对比，Entropy‑Centric 在预测置信度下降、IoU 降低和熵提升上均表现更好，说明其解释更忠实、定位更精准；

**⚠️ 局限性**

局限性包括：解释粒度为补丁级别，可能缺失细粒度细节；需要多次前向传播，计算成本相对较高；对采样策略和阈值敏感。

---

## 503. Batch Size or Negatives? A Selection Rule for Memory-Constrained Recommender Training

**arXiv ID:** 2608.11061 | [PDF](https://arxiv.org/pdf/2608.11061v1)

**作者:** Artyom Sabitov `[一作]` (Moscow Independent Research Institute of Artificial Intelligence), Alexey Zaytsev `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

分析在固定内存预算下，采样 softmax 的批大小与负样本数量的权衡，并给出理论最优配置，随后在合成数据和 MovieLens‑1M、Gowalla、Netflix、MovieLens‑20M 等大型推荐数据集上进行验证。

**💡 创新点**

通过分解两源随机性的梯度方差，证明在两者中应优先增大批量、减小负样本数，提出实用的 n≈√B、k≈√B 配置规则，首次将理论与实验结合给出内存分配的最优指导。

**🔧 技术方法**

采用采样 softmax 与 logQ 偏差校正、SGD/Adam 优化、理论梯度方差分析（delta 方法、Felton–Wilkinson 近似）、NDCG@10 与 AUL 的实验评估。

**📊 数据集**

使用 MovieLens‑1M、MovieLens‑20M、Gowalla、Netflix 四个真实推荐数据集以及一个大规模合成数据集进行实验。

**📈 对比分析**

在不同 (n,k) 配置与标准、无偏交叉熵目标之间进行对比，利用 AUL 衡量收敛速度、NDCG@10 衡量最终推荐质量；实验表明更大批量、更少负样本能够更快收敛、最终性能更优，校正项在实践中效果不明显。

**⚠️ 局限性**

仅聚焦于最后一层的梯度方差分析，实验规模有限；未探索更大模型或多任务设置；校正项对优化动力学影响微乎其微，实际应用中可能无显著提升。

---

## 504. HUI360: A 360° Egocentric Dataset and Baselines for Human-Robot Interaction Anticipation

**arXiv ID:** 2608.11051 | [PDF](https://arxiv.org/pdf/2608.11051v1)

**作者:** Raphael Lorenzo-Louis `[一作]` (Inria), Serena Ivaldi `[通讯]` (Inria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了HUI360人机交互预测大规模数据集与自动化标注管道，并在其上训练并评估了随机森林、MLP、LSTM等基线模型。

**💡 创新点**

①最大规模野外360°多环境人机交互预测数据集；②基于物理交互的可自动标注方法；③统一评估协议及跨域零样本转移实验。

**🔧 技术方法**

利用YOLOv11检测、SAM2跟踪/分割、ViTPose与Sapiens姿态估计，以及随机森林、MLP、LSTM等机器学习模型。

**📊 数据集**

主要使用HUI360（4.31k轨迹、>1M检测）与SSUP-HRI（1.15M帧）两大数据集进行标注和评估。

**📈 对比分析**

通过AUC、F1、AP等指标，在跨环境、跨数据集、不同预测时延（0.33–2.0s）和不同帧率（15–1Hz）下对基线模型进行比较，LSTM性能最佳，但跨域时性能显著下降；帧率降至1Hz会导致急剧衰退。

**⚠️ 局限性**

仅关注物理接触交互，忽略了目光、言语等社交信号；基线模型过于简单；数据集仍缺乏更细粒度交互标签。

---

## 505. 3D Weighted Geometric Graph Neural Networks for Sheep Facial Pain Assessment

**arXiv ID:** 2608.11050 | [PDF](https://arxiv.org/pdf/2608.11050v1)

**作者:** Alam Noor `[一作]` (CISTER Research Center), Mohamed Daoudi `[通讯]` (University of Lille)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种基于单目深度估计的3D加权几何图神经网络，用于评估绵羊面部疼痛；

**💡 创新点**

创新点在于将面部关键点嵌入三维欧氏空间，结合几何加权消息传递和注意力机制，并采用双侧检测平均化提升准确率；

**🔧 技术方法**

使用VideoDepthAnything进行单目深度估计、ResNet-18视觉骨干、WG-GNN（3层）、基于RBF的边权、尺度化点积注意力、交叉熵+几何正则化；

**📊 数据集**

使用自建的绵羊面部关键点数据集，包含耳、眼、鼻等九个表情标注，涵盖多种品种、姿态和年龄；

**📈 对比分析**

与SOTA 2D-WGNN、CNN、SVM等模型对比，3D-SPFES WG-GNN在全局模型下达到78.33%准确率，Cohen's κ为0.473，明显优于单模型基线并提升了163% κ；

**⚠️ 局限性**

局限包括严重疼痛样本稀缺导致分级不完整，NPS对严重疼痛的区分不理想，以及几何权重相对注意力权重低，需进一步改进深度重建和数据扩充。

---

## 506. ReRound: Reconstructive Rounding to Resolve Midpoint Ambiguity in Calibration-Free LLM Quantization

**arXiv ID:** 2608.11045 | [PDF](https://arxiv.org/pdf/2608.11045v1)

**作者:** He-Yen Hsieh `[一作]` (Harvard University), H. T. Kung `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 ReRound 的无校准低位量化方法：先训练条件扩散模型重构低位权重，然后利用重构结果在中点模糊区内引导 RTN 的取整，生成若干候选整数权重矩阵，并通过谱保留（匹配主要奇异值）选取最优候选。

**💡 创新点**

创新点包括：①用训练好的条件扩散模型为权重提供先验重构，解决 RTN 的中点模糊问题；②设计位置依赖容差度量决定何时接受重构建议；③采用谱一致性（主奇异值匹配）进行候选选择，无需激活或文本校准。

**🔧 技术方法**

技术手段：条件扩散模型（U-Net）训练权重补丁；RTN 取整；位置依赖容差公式；奇异值分解与谱误差比较；低位（3-bit/4-bit）权重量化；固定量化参数不变。

**📊 数据集**

数据与模型：使用多款小型 LLM（Gemma 2 2B、Gemma 3 1B、Qwen3 1.7B、OLMo 2 1B、SmolLM2 1.7B、Llama 3.2 1B、Pythia 1.4B、Phi‑2 2.7B）；评测基准为 WinoGrande、PIQA、BoolQ、SIQA；对 perplexity 还用 WikiText‑2、C4；扩散模型训练仅使用模型自身权重补丁，无外部训练集。

**📈 对比分析**

与多种无校准 PTQ 方法（RTN 组/通道、HQQ、BNB FP4、CafeQ、Hadamard+RTN）以及有校准 PTQ 方法（GPTQ、AdaRound、SignRound）比较。ReRound 在 3‑bit 与 4‑bit 量化中均能提升 0.1–1.6 个点，在多模型平均上 4‑bit 提升约 1.3 点、3‑bit 约 1.6 点；在 4‑bit 时与最强校准方法 SignRound 的性能相当或略优。

**⚠️ 局限性**

局限性：①每个预训练模型需单独训练扩散模型，导致离线成本高；②只能在已给定的量化参数下改进，候选集受限；③谱选择仅在权重空间有效，可能无法捕获所有下游任务需求；④补丁重构可能忽略跨层/全局依赖；⑤主要在小型 LLM 上验证，尚未证实在大规模模型上的效果。

---

## 507. Policy Convergence and Divergence Across National and Within Regional AI Strategies: A Policy Design Element Analysis

**arXiv ID:** 2608.11006 | [PDF](https://arxiv.org/pdf/2608.11006v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 508. Property Graph Techniques in Relational Databases

**arXiv ID:** 2608.11001 | [PDF](https://arxiv.org/pdf/2608.11001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 509. A 5/4 bound for graphic $s$-$t$ path TSP on subcubic graphs

**arXiv ID:** 2608.11038 | [PDF](https://arxiv.org/pdf/2608.11038v1)

**作者:** Junho Hwang `[一作]` `[通讯]`, Junho Hwang

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文给出了在简单 2 连通三度图上，任意两个端点之间的图形 s‑t 路径 TSP 的最优解长度上界为 ⌊(5n+n₂(G))/4⌋-1，并给出 O(n²) 的构造算法。

**💡 创新点**

创新点在于将 Wigal‑Yoo‑Yu 的边根偶覆盖技术推广到任意端点对，提出拆分‑细分 gadget，首次直接实现 5/4 近似系数（并证明其最优）。

**🔧 技术方法**

主要技术包括偶覆盖的剩余量理论、边根转换与拆分细分 gadget、图合并/收缩分析，以及基于 Scan 的线性时间估计。

**📊 数据集**

由于论文为理论研究，未使用实验数据集，仅在结构化图类（如三度图、3‑连通三度图）上做理论分析。

**📈 对比分析**

与现有的 7/5、3/2、以及通过通用路径-巡回转换得到的 5/4+ε 方案相比，本文在子三度图上直接给出 5/4 近似，算法时间为 O(n²)，在所有情形下保持最优系数。

**⚠️ 局限性**

局限性在于方法仅适用于最大度为 3 的图，对更高度或一般图类尚无直接推广；算法对特殊异常结构的处理依赖于 WYY 的四条例外，无法自动化到更大图类。

---

## 510. When Visual Signals Mislead: A Mechanistic Study of Attribute Hallucination in Vision-Language Models

**arXiv ID:** 2608.11024 | [PDF](https://arxiv.org/pdf/2608.11024v1)

**作者:** Yufei Zhang `[一作]` (Zhejiang University), Hongwei Wang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VISOR 框架，结合 null-image 诊断与按因子路由的纠错，解决 VLM 的属性幻觉问题。

**💡 创新点**

创新点在于用视觉与语言先验分离的 VSNR 诊断判定错误类型，并根据诊断结果路由不同干预（校准、拒绝、视觉适配）。

**🔧 技术方法**

使用了 null-image 前向传播提取视觉和语言信号、SNR 评估、局部层追踪、LoRA 视觉适配器以及阈值校准。

**📊 数据集**

基于 VAW（Visual Attributes in the Wild）数据集进行 yes/no 属性查询实验。

**📈 对比分析**

与 VCD、ICD 等 prior‑suppression 方法对比，VISOR 在颜色/状态属性上减少 FPR 11‑17pp，在材料属性上通过 Adapt 或 Abstain 也显著降低误报，整体性能优于基线。

**⚠️ 局限性**

局限在于只针对已知属性词，Adapt 只覆盖高 FPR 的有限词汇，且对开源词汇的扩展尚未实现。

---

## 511. Workflow Cards: Structured Summaries of Workflow Executions Using Provenance Data

**arXiv ID:** 2608.11022 | [PDF](https://arxiv.org/pdf/2608.11022v1)

**作者:** Nicola Giuseppe Marchioro `[一作]` (University of Trento), Renan Souza `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Workflow Cards，利用结构化模板将工作流执行的原始 provenance 数据压缩为易读、可被人类和 LLM 直接查询的摘要，弥补 Model Card 与 Data Card 在执行层信息上的不足。

**💡 创新点**

创新点在于将工作流级别的执行细节（如资源使用、参数选择、任务状态等）转化为面向问题的问答友好卡片，既可供人类快速理解，也可直接作为 LLM 上下文，显著提升了对 provenance 数据的可访问性和可解释性。

**🔧 技术方法**

技术手段包括：基于 W3C PROV 规范的 provenance 采集（Flowcept 与 yProv4ML）；设计并实现面向问题的卡片模板；利用 LLM‑as‑a‑Judge 对答案质量进行量化评估；以及对比 schema‑based 查询与预先生成卡片的回答效果。

**📊 数据集**

使用的数据集主要为公开的 Hugging Face 机器学习微调实验（构建合成 Workflow Card）以及真实的气候 ML 工作流 DLESyM（通过 Flowcept 采集 provenance），同时生成合成 metadata 用于 Benchmark I。

**📈 对比分析**

比较方法分为两大基准：Benchmark I 通过单卡信息完整性与交叉信息评估（单卡 vs 全卡 vs 留一）；Benchmark II 将 Workflow Card 与基于 schema 的查询进行对照，使用 LLM‑as‑a‑Judge 与人类评审进行评分；结果显示 Workflow Card 在 LLM 质量上几乎翻倍提升（≈0.45→0.87）。

**⚠️ 局限性**

限制包括：Benchmark I 采用合成元数据而非真实事实；Benchmark II 仅覆盖单个工作流且人类评审仅为单一专家；实验聚焦 ML 工作流，非 ML 领域（如基因组学、CFD 等）的适用性尚未验证。

---

## 512. ConRub-Med: Reinforcement Learning with Consensus Rubrics for Open-Ended Medical Question Answering

**arXiv ID:** 2608.10996 | [PDF](https://arxiv.org/pdf/2608.10996v1)

**作者:** Taojie Zhu `[一作]` (Tsinghua University), Yonghong He `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于共识规则（rubric）和三状态评分的强化学习方法，用于医疗问答模型的训练

**💡 创新点**

创新点在于①使用三台独立生成模型产生规则并通过审核器筛选共识规则；②采用三状态评分（正确、缺失、错误）给出负分；③在完全相同最终奖励的组中引入双向比较，直接生成序列优势而不依赖额外偏好损失

**🔧 技术方法**

技术包括多模型规则生成与审核、三状态判定器、Group Relative Policy Optimization (GRPO)、Pairwise Sequence Advantages、语言模型（Qwen3-4B、Qwen3-32B等）

**📊 数据集**

构建了5,166条提示的规则集，包含49,046条内容规则和10,332条全局控制；使用HealthBench-Hard、MedXpertQA、DiagnosisArena、MedMCQA、PubMedQA、MMLU-Medical等医学基准以及WritingBench、GPQA-Diamond、IFEval等通用基准进行评估

**📈 对比分析**

与单源规则、二元评分、无对比优势等对照实验，所提方法在9项基准中排名第一的6项（如HealthBench-Hard、MedXpertQA、DiagnosisArena等），整体医学平均分51.76，泛化平均分74.27，HealthBench-Hard分数38.98显著高于先前最高的37.30

**⚠️ 局限性**

局限性包括：共识规则可能遗漏单一或双重模型识别的有用规则；对评估专家样本有限，需更广泛覆盖；规则构建依赖大型闭源模型，成本高；Pairwise优势仅适用于完全相同奖励的组，无法扩展到近似相等情况

---

## 513. Multi-Level Evidence Aggregation for Robust Facial Phenotype Retrieval in Rare Genetic Disorder Prioritization

**arXiv ID:** 2608.11037 | [PDF](https://arxiv.org/pdf/2608.11037v1)

**作者:** Alexander Hustinx `[一作]` (University Hospital Bonn), Peter Krawitz `[通讯]` (University Hospital Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

针对罕见遗传疾病的面部表型检索，提出一种在推理时对多级证据进行聚合的方法，提升诊断候选疾病的排名精度。

**💡 创新点**

创新点在于：①在不改动底层表型编码器的前提下，进行推理时的多级证据聚合；②引入患者级影像聚合、患者加权的疾病中心点聚合以及局部最近邻与全局中心点的混合评分；③通过多级聚合实现对单图像匹配的显著提升，尤其在多图像和稀缺疾病情形下效果更显著。

**🔧 技术方法**

使用的技术包括：ArcFace基础表型编码器（GM-Arc）及其多模型、多TTA融合；余弦距离作为相似度度量；基于向量平均和加权平均的聚合运算；混合评分公式中的λ参数（取值0.75）。

**📊 数据集**

使用数据集：GestaltMatcher Database（GMDB）v1.1.4，包含约11,548名患者、15,381张面部图像、710种罕见遗传疾病；按患者数量划分为GMDB-Freq（>6例）和GMDB-Rare（≤6例）两组，并进一步划分多图像子集（GMDB-Multi-Freq、GMDB-Multi-Rare）。

**📈 对比分析**

与基准单图像最近邻检索（GM-Arc）相比，综合多级聚合在统一库上实现：Top‑1准确率从38.52%提升至48.82%（+10.30pp）在GMDB‑Freq，19.38%提升至23.79%（+4.41pp）在GMDB‑Rare；多图像子集Top‑1从46.12%提升至60.94%（+14.82pp）和从18.54%提升至26.71%（+8.17pp）。

**⚠️ 局限性**

限制包括：①评估样本有限，尤其是多图像病例；②聚合仅在现有编码器（GM-Arc）下测试，未检验对其它表型编码器的适用性；③对罕见疾病的中心点聚合可能无法覆盖多样化的面部表型；④λ参数固定，缺乏对不同疾病分布的自适应调整；⑤仅考虑单标签诊断，未处理多重诊断或组合表型。

---

## 514. Aerial Layouting: Design and Control of a Compliant and Actuated End-Effector for Precise In-flight Marking on Ceilings

**arXiv ID:** 2608.10987 | [PDF](https://arxiv.org/pdf/2608.10987v1)

**作者:** Christian Lanegger `[一作]` (ETH Zurich), Roland Siegwart `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在空中布局任务中，提出一种基于Gough‑Stewart平台的柔性自驱动末端执行器，通过多接触点、弹性支撑和全向轮实现毫米级精度的天花板标记。

**💡 创新点**

创新点在于将弹性Gough‑Stewart结构与多点接触相结合，且不依赖精确模型或复杂的全身控制，利用末端执行器自身的闭环控制即可补偿飞行器误差。

**🔧 技术方法**

采用机械优化（能量场最小化）、弹簧阻尼支撑、三轮全向驱动、相机+CharUco板标定与视觉跟踪，以及基于动力学的闭环控制。

**📊 数据集**

实验数据来自自建OMAV平台与Vicon运动捕捉系统的同步采集，没有使用公开数据集。

**📈 对比分析**

与自由飞行、单接触、无弹性、无闭环等基线进行对比，平均定位误差低于1 mm（全系统MAE < 2 mm），显著优于以往基于Delta或简单接触的飞行器标记方法。

**⚠️ 局限性**

局限在于仅适用于水平平面天花板、对视觉跟踪精度和外部姿态估计要求较高、对倾斜或曲面拓展有限、且在强风等剧烈扰动下性能可能下降。

---

## 515. PEAK: Precise and Persistent Concept Erasure via k-Sparse Autoencoders

**arXiv ID:** 2608.10985 | [PDF](https://arxiv.org/pdf/2608.10985v1)

**作者:** Man Jiang `[一作]` (Hefei University of Technology), Yanbin Hao `[通讯]` (Hefei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PEAK，一种通过 k‑sparse autoencoder 对扩散模型内部表示进行特征定位与参数优化，实现精准且持久的概念擦除方法。

**💡 创新点**

创新点在于：①利用 kSAE 将稠密激活拆解为可解释的稀疏特征；②通过对目标与非目标提示的对比得出目标特征集合；③在参数优化阶段同时抑制目标特征并对齐非目标特征，既保证擦除精度又保持生成质量，且无需推理时干预。

**🔧 技术方法**

核心技术包括 k‑sparse autoencoder（BatchTopK 稀疏）、特征重要性评分（激活强度+频率）、目标特征抑制损失、对齐（保持）损失，以及对扩散网络的局部微调。

**📊 数据集**

使用的主要数据集与评测工具包括 Stable Diffusion v1.4/SDXL/FLUX、I2P（不良内容提示）、MS‑COCO（生成质量评估）、NudeNet（裸体检测）、CLIP、FID、KID 等。

**📈 对比分析**

在 I2P 基准上与九种现有擦除方法对比，PEAK 将 NudeNet 检测数从 582 降至 6，攻击成功率（ASR）从 96.52% 降至 5.63%；在 MS‑COCO 上实现最高 CLIP Score、最低 FID 与零 KID；在黑盒/白盒概念恢复攻击下，ASR 极低（<0.1%），表现出显著的持久性和生成质量保持。

**⚠️ 局限性**

局限性包括：①对极细粒度属性的抑制效果可能不如预期；②需要额外训练 kSAE，增加计算与存储成本；③目前仅验证了视觉概念的擦除，对文本层概念或跨模态概念的扩展尚未深入；④在更大规模或更复杂模型上可能需要进一步调优。

---

## 516. ThinkAfford: Affordance-Centric Reasoning for Fine-Grained 3D Grounding in Cluttered Scenes

**arXiv ID:** 2608.10981 | [PDF](https://arxiv.org/pdf/2608.10981v1)

**作者:** Xinrui Lin `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种任务驱动的3D可操作性定位框架，将问题拆分为高召回的可操作性提案生成与基于语言的推理两部分。

**💡 创新点**

创新点包括：1）Affordance Proposal Generation (APG)通过学习可操作性上下文提示和多层视觉特征，预测交互级热图并生成细粒度提案；2）Visual-Prompted Affordance Reasoning (VPAR)采用think‑then‑answer结构，并利用Group Relative Policy Optimization (GRPO)以3D重叠奖励进行强化学习，显著提升了对相似提案的区分能力。

**🔧 技术方法**

使用的技术包括：CLIP与DINOv2跨模态编码器、CoOp提示学习、DBSCAN聚类生成提案、可视化提示的VLM、GRPO强化学习、可视化到3D的投影与加权投票融合。

**📊 数据集**

主要数据集为SceneFun3D（训练200场景，验证30场景），并在ScanNet、ScanNet++、3RScan、MultiScan等公开室内重建数据集上进行零样本跨域评估。

**📈 对比分析**

与OpenMask3D、LERF、Mask3D-F、Fun3DU、TASA、AffordBot等基线进行对比，SceneFun3D验证集上AP_25提升至25.46%（+12.9个百分点），AP_50提升至10.69%（+4.3个百分点）；跨域零样本表现优于Fun3DU，并在真实机器人部署中验证了定位准确性。

**⚠️ 局限性**

局限性包括：1）提案覆盖仍存在漏检，导致IoU 0.25召回率下降；2）3D边界精度不够高，难以达到更严格阈值；3）对稀有交互类别的覆盖仍不均衡；4）仅提供定位结果，未涉及执行控制或力反馈等实际操纵环节。

---

## 517. Measuring Cross-Cultural Style Diffusion Through Era Classification: US and Korean Popular Music

**arXiv ID:** 2608.10980 | [PDF](https://arxiv.org/pdf/2608.10980v1)

**作者:** Dasol Lee `[一作]` (Sogang University), Dasaem Jeong `[通讯]` (Sogang University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出跨文化时代推断框架：训练CNN时代分类器仅使用美国Billboard音频，再应用到韩国Melon榜单歌曲，量化两国音乐风格在时间上的对齐差异。

**💡 创新点**

创新点：①用年代标签作为跨文化比较轴，避免文化偏差；②采用从零开始的CNN避免预训练模型泄漏；③提供可量化的“时代偏移”指标，展示跨文化音频在不同时期的同步趋势。

**🔧 技术方法**

技术：多层次层级预测（decade/half-decade/quarter-decade/year）CNN架构（FCN、ShortChunkCNN、Musicnn、CRNN等），单声道16kHz、Mel spectrogram，随机30秒裁剪，并用多种随机种子训练以检验稳健性。

**📊 数据集**

数据集：Billboard Hot 100（≈22,000曲目，1958–2024）和Melon榜单（≈2,200曲目，1964–2009），按艺术家划分、年代平衡并采用艺术家无泄漏的拆分。

**📈 对比分析**

比较方法：在Billboard内域评估准确率；在跨域预测中计算时代偏移的中位数和分布；不同模型和年代的偏移结果一致，显示1960–80年代偏移≈4–5年，90年代≈2–3年，2000年代保持≈2–3年。

**⚠️ 局限性**

局限：模型同时捕获录音技术差异，难以区分作曲风格与录音工艺；30秒裁剪缺乏完整曲式信息；音频来源不一定为原声；单曲偏移精度受噪声影响。

---

## 518. Twisted Conjugacy and the Classification of Induced Centrosymmetric Alternant Codes

**arXiv ID:** 2608.11056 | [PDF](https://arxiv.org/pdf/2608.11056v1)

**作者:** Ousmane Ndiaye `[一作]` (Université Cheikh Anta Diop de Dakar), Massamba Sow `[通讯]` (Université Cheikh Anta Diop de Dakar)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过研究泛化Reed–Solomon码（GRS）的自同构，构造并分类了诱导中心对称的交替码；

**💡 创新点**

创新点在于引入“扭曲共轭”作用，证明其与PΓL₂(𝔽_q)中的常规共轭等价，并利用Shintani定理完成对半正交自同构（即中心对称变换）的全套分类，统一并推广了已知的 quasi‑centrosymmetric 交替码构造；

**🔧 技术方法**

核心技术包括：
- 泛化Reed–Solomon码与Cauchy码的等价与自同构描述；
- 设定扭曲共轭(h·_j f = h^(−p^j)fh^(−1))并证明其在PGL₂中的轨道与PΓL₂中的共轭轨道对应；
- 应用Shintani定理得到 γ_p^j‑相似类的四种类型；
- 结合自同构条件，给出交替码中心对称的必要与充分条件；
- 对不同类型的同余类给出具体支持与乘子结构；
- 在 𝔽₂⁴（大小为16）的示例中验证理论。

**📊 数据集**

示例使用的“数据集”是有限域 𝔽₂⁴（φ(x)=x⁴）以及在该域上构造的长度为8、维度为2 的GRS码；通过不同的同构变换（四种同余类）生成对应的中心对称交替码。

**📈 对比分析**

论文没有进行实验性能比较；主要以理论证明和有限域示例来说明构造方法的正确性和完整性。

**⚠️ 局限性**

限制：
- 结果仅针对由GRS码诱导的交替码，未直接扩展到更一般的代数码；
- 对于大规模码，具体的实现细节（如多项式计算、支持集合构造）未给出；
- 仅给出有限域上的实例，缺乏对实际通信环境下码性能（误码率、编码/译码复杂度）的评估。

---

## 519. Efficient Hypergradient Descent for Inverse Reinforcement Learning

**arXiv ID:** 2608.11052 | [PDF](https://arxiv.org/pdf/2608.11052v1)

**作者:** Nikita Sevriukov `[一作]` (HSE University), Marina Sheshukova `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出了一种基于 Fisher 信息矩阵的隐式超梯度方法，用于求解逆强化学习（IRL）的双层优化问题。

**💡 创新点**

创新点在于证明了在内层最优且可实现条件下，内层海森矩阵等价于折扣轨迹 Fisher 信息矩阵，从而将 Fisher 信息作为逆 Hessian 的近似，并引入流式 Spectral Compensation Frequent Directions (SCFD) 进行稀疏近似，显著降低存储和计算成本。

**🔧 技术方法**

使用了 Fisher 信息矩阵、隐式超梯度（Implicit Hypergradient）、自然超梯度下降（NHGD）思想、SCFD 流式矩阵压缩以及标准的最大熵 IRL 公式。

**📊 数据集**

实验数据集包括经典的 CartPole（离散控制）和 LQR（连续控制）两种环境。

**📈 对比分析**

与单循环 ML-IRL 基线进行对比，Fisher Sketching 方法在保持或提升奖励排名（RankCorr）和策略性能的同时，显著减少内存占用（最高 1.31 倍）并加快运行速度（最高 1.29 倍），尤其在高维 LQR 环境中效果更为显著。

**⚠️ 局限性**

局限性主要体现在对 Fisher 信息矩阵近似的假设（需内层达到可实现最优）以及对 SCFD 超参数（如 sketch 大小、阻尼 λ）的敏感性，过大或过小的参数会导致优化不稳定或近似失真。

---

## 520. Multiclass Sentiment Analysis for Identifying Political Viewpoints

**arXiv ID:** 2608.11049 | [PDF](https://arxiv.org/pdf/2608.11049v1)

**作者:** Girma Yohannis Bade `[一作]` (Instituto Politécnico Nacional), Grigori Sidorov `[通讯]` (Instituto Politécnico Nacional)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对泰米尔语社交媒体帖子进行七类多层次政治情感分析，区分正面、负面、讽刺、意见化、立证、无关及中性。

**💡 创新点**

提出在低资源泰米尔语上同时使用XGBoost与BERT构建多类情感基线，并首次在该语言下对两种模型进行系统评测。

**🔧 技术方法**

利用TF‑IDF特征+XGBoost与bert-base-uncased+BertTokenizer两套技术栈进行模型训练与推理。

**📊 数据集**

使用DravidianLangTech 2025共享任务提供的4,352条训练、544条验证、544条未标记测试样本的泰米尔语推文数据集。

**📈 对比分析**

通过宏F1指标比较，XGBoost取得0.2835、BERT取得0.2806，表明两模型虽相近但均面临复杂政治语境的挑战。

**⚠️ 局限性**

受限于样本量有限、类别不均衡、对“None of the above”等稀缺类识别不足，以及模型对上下文细微差别捕捉能力不足。

---

## 521. V-FiLLM: Verified Financial LLM Reasoning Benchmark

**arXiv ID:** 2608.11047 | [PDF](https://arxiv.org/pdf/2608.11047v1)

**作者:** Alicia Larsen `[一作]` (ETH Zürich), Nino Antulov-Fantulin `[通讯]` (ETH Zürich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了V-FiLLM框架，利用可执行计算树从合成财务电子表格自动生成金融推理问题及其答案，实现零标注、可扩展的基准；

**💡 创新点**

创新点在于：①完全自动化生成、答案可验证；②四维可控难度（计算深度、表达宽度、金融概念复杂度、上下文大小）；③提供多轮、对抗鲁棒性和数据增强扩展；

**🔧 技术方法**

技术包括符号计算树生成与类型约束、自然语言渲染与表述增强、LoRA微调、对抗扰动、基准评估与可视化；

**📊 数据集**

使用了自定义的合成财务表格（10‑Q风格与规范化表格）作为基准数据，LoRA验证时亦使用FinQA；

**📈 对比分析**

与六款开源LLM（Gemma‑31B、Qwen3.7‑Plus、DeepSeek‑v4‑Flash、Llama‑3.3‑70B、GPT‑OSS‑120B、Qwen3.5‑9B）在混合深度题目上评测，Gemma最高，深度/对抗扰动显著降低准确率；LoRA微调将准确率提升至85.6%，多轮拆解显著提升低端模型；

**⚠️ 局限性**

局限性包括仅英文、单一表格格式、实验规模有限（模型数与样本数受算力限制），对抗扰动仅限于问句无关单元格，生成问题可能存在翻译歧义且未覆盖更深层推理或跨表/跨文档场景。

---

## 522. Who Are You Explaining To? A Multi-Agent System for Audience-Aware XAI Narratives

**arXiv ID:** 2608.11033 | [PDF](https://arxiv.org/pdf/2608.11033v1)

**作者:** Francesco Musicco `[一作]` (Politecnico di Bari), Tommaso Di Noia `[通讯]` (Politecnico di Bari)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一个多代理框架 XstrAI，用于将本地特征归因结果转换为面向不同受众（临床医生、患者、数据科学家）的可解释叙事。

**💡 创新点**

创新点在于：① 将解释证据与解释文本分离，使用不可变的解释卡保持证据不变；② 通过三阶段 LLM 代理（规划者 Framer、叙述者 Narrator、审查者 Reviewer）实现受众感知规划、语言实现与一致性验证；③ 设计了有限的修订循环，确保输出在归因、语义与安全性方面的一致性。

**🔧 技术方法**

技术方法包括：SHAP 局部归因、不可变解释卡（类似模型卡），多代理 LLM 系统（使用 qwen、gemma、Claude、GPT、Gemini 等大语言模型）、基于规则和 LLM 的文本抽取、可读性与词汇多样性评估、语义匹配（BERTScore 类似）以及多轮审查与修订机制。

**📊 数据集**

使用公开的糖尿病（Diabetes）和中风（Stroke）预测数据集，分别构建随机森林模型并生成 SHAP 归因，进行实验评估。

**📈 对比分析**

与 11 种基线（Explingo、单代理与逐步消融版本）在两组数据集上进行对比；通过内在叙事评估（可读性、词汇丰富度、归因一致性）和外部评估（LLM 评判者的重识别与排名、人工问卷）验证。XstrAI 在受众识别准确率 100%，语义匹配、归因一致性和 LLM Elo 评分上均优于基线，尤其在临床医生和患者受众上表现突出；在数据科学家受众上与最佳单代理基线相当。

**⚠️ 局限性**

局限性包括：人类评估样本有限，评估范围仅覆盖两类医学预测任务，未验证对其他归因方法或非表格数据的泛化，系统对输入噪声的鲁棒性尚未充分评估，且对受众模型的细粒度差异（如不同专业背景）处理尚不完善。

---

## 523. R4DSG: Relative 4D Scene Graph Memory for Object-Centric Question Answering in Long Egocentric Video

**arXiv ID:** 2608.11017 | [PDF](https://arxiv.org/pdf/2608.11017v1)

**作者:** Ke Ma `[一作]` (Tongji University), Meng Wang `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个基于相对4D场景图的查询内存，能够将长时 egocentric RGB 视频转化为可检索的对象状态和锚点变化记录，从而支持对象中心的长时问答。

**💡 创新点**

创新点在于：①将对象与稳定锚点分离，仅记录对象相对锚点的状态变化；②使用 RGB‑only 分割与 3D 提升实现相对空间信息；③通过持续身份关联与锚点推断构建持久对象轨迹；④将事件与检索文档紧密耦合，形成紧凑、可检索的内存。

**🔧 技术方法**

使用了 SAM‑3 视频分割与一致性、SAM‑3D RGB‑only 3D 提升、相对锚点关联、持久身份匹配、事件写入、检索+文档化（Retrieval+）以及 Qwen3.5‑27B 语言模型进行答案生成。

**📊 数据集**

采用了 EgoLifeQA A1_JAKE 单人数据集，包含 500 题四选，255 题为对象相关，涵盖 72 个 When 题、15 个 Why 题。

**📈 对比分析**

与 Plain RAG、EgoRAG‑Text、VLM‑only、EMQA‑style episodic、AMEGO‑style active memory 等基线进行比较；在仅用问题检索时整体准确率从 32.9% 提升至 39.6%（+6.7pp），When 题从 30.6% 提升至 43.1%（+12.5pp）；在 option‑blind 检索下仍保持领先；WhyMemory 进一步提升 Why 子集从 33.3% 到 40.0%。

**⚠️ 局限性**

主要局限包括：①仅在离线环境下实现，无法在线即时更新；②仅在单一受试者上验证，缺乏多主体泛化；③锚点识别与轨迹关联仍易受遮挡、光照等影响；④缺少因果或社交上下文支持，导致某些 Why 题仍难以回答；⑤检索时可能出现漏检或误检。

---

## 524. Templated or fully Synthetic? Prompt construction as a confound in measuring LLM political stance beyond writing assistance

**arXiv ID:** 2608.11008 | [PDF](https://arxiv.org/pdf/2608.11008v1)

**作者:** Ilias Chalkidis `[一作]` `[通讯]` (University of Copenhagen), Ilias Chalkidis (University of Copenhagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将 IssueBench 从原先仅测评写作辅助任务扩展至信息寻求与观点分享，并提出使用全合成 LLM 生成的提示替代传统模板提示，以提升评测的生态效度并评估不同提示构造对 LLM 政治立场估计的影响。

**💡 创新点**

创新点包括：①提出完全合成的 LLM 生成提示，能够在保持对议题、意图、立场控制的同时，消除模板提示的结构化痕迹；②通过人类与 LLM 的双重评估，发现模板提示在中立情境下会因填充词的暗示而导致模型偏向特定立场；③系统对比三类提示在真实感与构造识别两项指标上的表现，并通过立场检测实验量化提示构造对模型估计的偏差。

**🔧 技术方法**

技术手段包括：Prompt 采集与归纳（真实日志、模板提炼、LLM 生成）、LLM-as-a-Judge 的多模型投票判定、数据标注（真实感排序、构造识别、意图/立场判定）、统计分析（平均倾向、拒绝率、Wilcoxon检验）以及对比实验。

**📊 数据集**

使用的数据集包括：① IssueBench 的真实聊天日志；② 通过模板提炼得到的 3,000 条模板提示；③ 由 Claude Opus 4.8 在详细指令下生成的 2,700 条合成提示；④ 六个争议议题（移民、气候变化、AI 采用、以色列-巴勒斯坦、俄乌、伊朗-美国/以色列）和三种用户意图；⑤ 用于判定的三款开源 LLM（DeepSeek V4 Pro、Mistral Large 3、NVIDIA Nemotron 3 Ultra）组成的投票评判集。

**📈 对比分析**

比较方法：先用人类和 LLM 评估三类提示的真实感排名和构造识别，随后对 GPT‑5.4‑mini 与 Grok‑4.3 的响应进行立场检测。结果显示：合成提示在真实感排名与模板提示相差显著，几乎与真实提示同级；在构造识别中模板提示最易被识别。立场检测中，模板提示在中立情境下平均倾向偏离约 0.4–0.5 量表点，极端立场下差异可达 1+，说明提示构造对模型倾向估计有显著影响。整体拒绝率也因提示结构差异而不同。

**⚠️ 局限性**

限制：①仅测试两款美国开发者的 LLM，未涵盖更广泛模型；②合成提示全部由单一生成模型（Claude Opus 4.8）产生，可能带来偏差；③研究仅限六个英文议题，未考察多语言或地区差异；④LLM‑as‑a‑Judge 本身带有立场偏好，未对比人类标注；⑤未验证识别提示是否真正改变模型行为；⑥平均倾向指标粗糙，忽略响应分布细节；⑦未研究多轮对话与时间漂移对立场的影响。

---

## 525. HNDiff: Haze-Noise Diffusion for Image Dehazing

**arXiv ID:** 2608.10995 | [PDF](https://arxiv.org/pdf/2608.10995v1)

**作者:** Jin-Ting He `[一作]` (National Yang Ming Chiao Tung University), Yen-Yu Lin `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

采用基于扩散模型的图像去雾框架 HNDiff，利用大气散射模型构造前向扩散过程，并在反向过程中同时去除雾和噪声。

**💡 创新点**

关键创新是将物理层面的雾生成机制嵌入扩散过程，并提出雾感知噪声调度器（HANS）以及在潜在空间的先验生成器，实现对雾密度的自适应建模。

**🔧 技术方法**

结合扩散模型、U‑Net 估计器、Feature Gating Module 与潜在空间扩散，构建联合雾-噪声扩散与去雾-去噪逆过程。

**📊 数据集**

在 SOTS-Indoor、SOTS-Outdoor、NH‑HAZE、O‑HAZE、Dense‑HAZE、RW2AH 与真实场景 RTTS 等七个数据集上进行训练与评测。

**📈 对比分析**

与四个主流去雾基线（FocalNet、ConvIR、SGDN、RIDCP）对比，平均 PSNR 提升约 0.57 dB，SSIM 提升约 0.009，显著超越现有最优方法。

**⚠️ 局限性**

仅适用于符合大气散射模型的雾退化，无法直接迁移到运动模糊、雨滴或低照度等其他噪声场景。

---

## 526. Putting Registers to Work: Task Registers for Token Pruning in Vision Transformers

**arXiv ID:** 2608.10989 | [PDF](https://arxiv.org/pdf/2608.10989v1)

**作者:** Hongsen Cao `[一作]` (Queen Mary University Of London), Ahmed Sayed `[通讯]` (Queen Mary University Of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Task-Adaptive Pruning（TAP），利用任务专属的注册表动态控制ViT模型的token裁剪、预算分配和稠密特征恢复。

**💡 创新点**

创新点在于将任务信息嵌入可演化的注册表，实现跨任务的自适应裁剪、层级预算分配以及任务感知的恢复比例，打破传统单一裁剪策略的限制。

**🔧 技术方法**

主要技术包括Vision Transformer（ViT）骨干、MAE预训练、基于注册表的评分与选择、精确预算分配的Sigmoid线性读出、以及每个被裁剪token对应的stand‑in恢复机制。

**📊 数据集**

使用的数据集包括ImageNet-1K（分类）、ADE20K（语义分割）和COCO（目标检测与实例分割）。

**📈 对比分析**

与多种基线（Random、Attention Top‑K、ToMe、Token‑Cropr、SViT等）进行对比，TAP‑J在保持接近无裁剪模型的精度（分类83.2%，Seg 47.0 mIoU，检测53.7 AP）同时实现约1.3×编码器吞吐量提升，表现优于大多数传统裁剪方法。

**⚠️ 局限性**

局限性包括：需为每个任务单独训练注册表或低秩更新；固定总保留率限制了对不同输入尺寸的适应；恢复仅使用单一stand‑in与偏移，未考虑多级或跨层传输；在极大模型或多任务场景下的存储与训练成本仍较高。

---

## 527. ConVAWG: A Retrieval-Grounded Framework for Controlled Synthetic Dialogue Generation in Violence Against Women and Girls

**arXiv ID:** 2608.11200 | [PDF](https://arxiv.org/pdf/2608.11200v1)

**作者:** Chen Lyu `[一作]` (University of Warwick), Gabriele Pergola `[通讯]` (University of Warwick)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一套名为 ConVAWG 的检索驱动框架，用于生成以英国法律为准绳、基于真实案例模式、并配备情节层级计划与毒性控制的多场景 VAWG 对话。

**💡 创新点**

创新点在于：①将公开的国内凶杀审查报告与人口统计、CPS 罪名定义融合，构造统计一致、法律对齐的情景；②通过分层事件图实现情节与时间的精细规划；③在 LLM 角色扮演中引入检索式风格控制；④采用 Contrastive Activation Addition (CAA) 对单个加剧语句进行可控毒性注入，避免整体对话失去连贯性。

**🔧 技术方法**

使用技术包括：大型语言模型 GPT‑5.2（以及多模型对比）、向量检索（DHR、CPS、ONS 数据库）、事件图构造与层级拆分、检索式风格提示、对抗式激活添加进行毒性控制、以及后处理的短句裁剪和时间戳同步。

**📊 数据集**

数据来源主要有：PersonaHub 的受害者人格种子、英国 ONS 的人口与犯罪统计、CPS 对 VAWG 的官方定义、以及 410 篇公开的 Domestic Homicide Review 报告；这些数据被加工成情境规范、事件图和对话脚本。

**📈 对比分析**

与八个基线（5 个直接生成、3 个框架迁移）以及人工评估、LLM‑as‑Judge、下游任务进行对比；ConVAWG 在 6 维对话质量指标上平均得分 4.75/5，显著优于基线（p<10⁻⁴），且在人类评估中也排名第一，显示在连贯性、人物一致性、毒性真实性、犯罪与情境真实性方面都有明显提升。

**⚠️ 局限性**

局限性包括：①数据与方法仅基于英国法律、统计与文化，跨国迁移性可能受限；②只覆盖有对话痕迹的行为，对纯离线或非口头的虐待形式建模有限；③合成数据虽然符合统计，却可能缺乏真实案例的细腻情感与多样性，且使用时须严格遵守数据使用协议，防止被滥用。

---

## 528. Mediatised Participation: Citizen Journalism and the Decline in User-Generated Content in Online News Media

**arXiv ID:** 2608.11159 | [PDF](https://arxiv.org/pdf/2608.11159v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 529. How to Verify Consistency of Probabilistic Claims

**arXiv ID:** 2608.11181 | [PDF](https://arxiv.org/pdf/2608.11181v1)

**作者:** Orr Paradise `[一作]` (EPFL), Shafi Goldwasser `[通讯]` (MIT)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种交互式可检查证明（IPCP）框架，用于在多项式时间内验证概率预测模型的内部一致性。通过构造稀疏的“证人”分布（支持点不超过声明数量+1）以及低精度的权重表示，作者给出了可接受的证明长度；同时利用多项式扩展、和检验协议以及 Reed‑Muller 码实现对分布的局部可验证编码。论文还证明了在一致性判定问题上的 NP 可验证性、可近似性与硬度结果，并讨论了其在 AI 安全与自证一致性训练中的潜在应用。

**💡 创新点**

创新点主要包括：
1) 第一次给出概率预测器一致性的交互式可检查证明，显著提升了一致性验证的可行性；
2) 利用 Carathéodory 定理证明任意一致性证人可稀疏化为 m+1 个支持点，并对权重的二进制长度给出严格上界；
3) 通过引入小的完整性-可靠性间隙（gap）实现低精度权重证明，从而将证书长度压缩至 O(mn + log B)；
4) 设计了一种 Reed‑Muller 码的分布编码与边际检验协议，使得验证者只需查询证明字符串的极少位置即可完成一致性检验；
5) 在证明与理论分析中融合了 sum‑check 协议、拉格朗日乘子、Hadamard 矩阵不等式等多种技术，形成了一套完整的可验证一致性理论。

**🔧 技术方法**

所用技术与方法包括：
- Carathéodory 定理与稀疏组合论，用于压缩证人分布；
- 二次规划与拉格朗日乘子求解最优权重；
- 复杂度分析（Hadamard 不等式、Cramer 规则）对权重二进位长度的上界；
- sum‑check 协议与多项式扩展，用于将指数级和检验转化为局部点检验；
- Reed‑Muller 码与多项式插值，构造分布的可检验编码；
- 互动式可检查证明（IPCP）框架，将传统 PCP 与交互式证明相结合；
- 证明压缩与 gap 技术，降低证书长度与误差容限。

**📊 数据集**

本文属于理论研究，不依赖任何实际数据集；作者在概念层面讨论了对神经网络等大规模预测模型的应用，但未给出具体数据集实验。

**📈 对比分析**

与传统一致性检查（如直接枚举或数值优化）相比，本文的方法具有：
- 证明长度从指数级压缩到 O(mn + log B)；
- 验证时间为多项式 O(m^3(B+log m)^2 + m^2 n)，在理论上可扩展；
- 交互式 PCP 仅需读取证明字符串的多项式数量位置，减少 I/O 负担；
- 对不一致性的检测具有完整性与可靠性保证；
- 通过 gap 方案可进一步将权重精度降低到 O(log(m/)) 位，进一步提升效率。

**⚠️ 局限性**

主要局限性包括：
- 证明字符串本身仍为指数级长度，需要与可信的 prover 合作；
- 交互式协议对通信与同步提出了额外要求，实际部署可能受限；
- 需要模型以可计算的 circuit 形式给出，对深度学习模型的直接映射尚不完善；
- 仅保证一致性而不保证预测准确性或校准性；
- 对于极大规模模型（变量数量成指数级），支持点与计算量仍显著；
- gap 方案会对一致性阈值产生微小偏差，实际应用需权衡精度与效率。

---

## 530. Matchings via Random Greedy Independent Set: A Simpler Algorithm and Analysis

**arXiv ID:** 2608.11163 | [PDF](https://arxiv.org/pdf/2608.11163v1)

**作者:** Andrew McGregor `[一作]` `[通讯]` (University of Massachusetts Amherst), Andrew McGregor (University of Massachusetts Amherst)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个在随机贪心最大独立集算法基础上，针对每个被删减顶点仅随机采样一条边，从而得到最大匹配的近似算法，并证明该算法在期望上可取得常数近似（1/27）

**💡 创新点**

创新点在于：1) 只采样一条边就能保证常数近似，极大简化了先前需要 O(log n) 条边的做法；2) 通过完全避免分数匹配的技术，给出了更短、更直观的证明，改善了近似比值；3) 证明了该改进版本在动态流模型下仍可实现 O(n²) 空间和 O(log log n) 轮的运行。

**🔧 技术方法**

采用的技术主要包括：随机贪心最大独立集（RGMIS），每步随机边采样，图方向化与子采样技术，概率分析与期望线性，构造可删减顶点集合的递归序列，以及利用路径结构推导匹配大小的下界。

**📊 数据集**

本文为理论论文，没有使用任何实际数据集；所有结果均基于数学证明与期望分析。

**📈 对比分析**

与先前的 Assadi 等人提出的 O(log n) 边采样版本相比，本文的算法在每一步仅采样一条边，空间与时间复杂度不变，但近似常数从他们的结果进一步简化为 1/27；在动态流模型中，该算法仍保持 O(n²) 空间、O(log log n) 轮的实现能力。

**⚠️ 局限性**

局限性包括：1) 近似常数 1/27 相对较大，实际匹配质量仍有提升空间；2) 仅给出了理论分析，未进行实验验证；3) 对动态流实现的细节仍未完全验证，尤其是多轮采样与边删除的同步问题。

---

## 531. Hierarchical Empirical-Bayes Naive Bayes: Minimax Smoothing and Calibration with AODE Extension

**arXiv ID:** 2608.11162 | [PDF](https://arxiv.org/pdf/2608.11162v1)

**作者:** Nguyen Thai Anh `[一作]` (Van Lang University), Ngo Hoang Tu `[通讯]` (Van Lang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种层级经验贝叶斯 Naive Bayes（HEB‑NB），通过对每个类别‑特征对的 Dirichlet 先验浓度进行 Type‑II 最大似然学习，实现自适应平滑；同时扩展到 HEB‑AODE。

**💡 创新点**

创新点在于：① 先验浓度自适应学习而非固定的拉普拉斯等平滑；② 通过理论分析给出非渐近 ℓ₁ 错误上界、Laplace 下界以及风险层面的严格分离；③ 证明自适应平滑可在结构放宽的 AODE 中同样提升。

**🔧 技术方法**

使用 Dirichlet‑Multinomial 共轭、Minka 的固定点迭代实现 Type‑II ML、闭式后验均值，结合稀疏高基数特征的先验平均（均匀或聚合边缘），并在需要时进行低计数下的 Laplace 退化。

**📊 数据集**

在 31 个 UCI/OpenML 基准数据集上评估，涵盖低、中等、以及 3 个高基数（最高 16,137 类）数据集，如 click‑prediction、amazon‑employee 等。

**📈 对比分析**

与 6 种固定平滑器（拉普拉斯、Lidstone、KT、m‑estimate、TE）和结构改进器（AODE、CAWNB）进行 Friedman‑Nemenyi、Wilcoxon 检验；HEB‑M 在 log‑loss 与 Brier 上获得最优 Friedman 排名，HEB‑AODE 对 AODE 有显著提升；与 sqrt‑MI 权重联用时 ECE 可降低 41%–70%。

**⚠️ 局限性**

局限包括：仅适用于离散特征（需对连续特征进行离散化）；HEB‑AODE 计算成本高、内存占用大；对于样本极少的类别会退化到 Laplace；Type‑II ML 的收敛速率理论尚未完全证明。

---

## 532. PRMU: A Corpus-Free Benchmark for Person-Centric Knowledge Unlearning in Multimodal Large Language Models

**arXiv ID:** 2608.11149 | [PDF](https://arxiv.org/pdf/2608.11149v1)

**作者:** Huafeng Chen `[一作]` (Nanjing University), Caifeng Shan `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无原始训练语料的多模态机器无学习框架，针对公开人物的自然获取知识进行去除。

**💡 创新点**

创新点在于：①构建了PRMU基准，专注人名相关知识的自然学习并提供无语料评估协议与邻域保留评估；②提出SGPE轻量级无语料去学习方法，融合知识位移、受保护参数投影和相似性门控，实现局部化的知识抹除。

**🔧 技术方法**

技术手段包括：代理语料生成、知识位移与双向受保护编辑、基于相似度的输入门控、生成式评估（ROUGE‑L、准确率）、邻域保留评估及MMBench实用性测评。

**📊 数据集**

使用PRMU数据集（1080公开人物、50,649文本探针、41,303视觉探针），以及从Wikidata、Wikipedia、CLIP等公开资源构建的代理语料。

**📈 对比分析**

与四种主流无学习方法（梯度上升、拒绝调优、负偏好优化、直接偏好优化）及SGPE对比。SGPE在目标遗忘、邻域保留和整体多模态实用性上取得更佳平衡，现有方法往往出现遗忘与局部保留冲突。

**⚠️ 局限性**

局限性包括：在大规模并行删除时仍存在交叉干扰；视觉-语言知识的彻底去除尚不充分；评估聚焦公开人物，可能与真实私有数据差异；SGPE虽轻量但仍可能引入局部漂移。

---

## 533. Is There Really a Camouflaged Object? Towards Realistic Camouflaged Object Detection

**arXiv ID:** 2608.11135 | [PDF](https://arxiv.org/pdf/2608.11135v1)

**作者:** Huafeng Chen `[一作]` (Nanjing University), Caifeng Shan `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种真实场景下的伪装物体检测方法OPCNet，并构建了大规模真实COD基准数据集OPC16K。

**💡 创新点**

创新点在于将COD拆解为物体定位与伪装存在推理，采用层次存在推理、相似度感知伪装关系建模和存在感知特征细化，实现对三种场景（CO、NOCOD、BG）的判别。

**🔧 技术方法**

利用PVTv2-B4骨干网络，结合自注意力、全局池化、轻量级粗定位头、相似度校准模块和存在感知门控实现。

**📊 数据集**

使用OPC16K数据集，共计16,245张图，包含9,000张伪装物体、3,050张纯背景和4,195张非伪装物体样本。

**📈 对比分析**

在OPC16K上与SINet、VSCode、RUN、CamoDiffusion、USCNet等五个主流COD方法对比，OPCNet在三类分类准确率最高、负样本误检率最低、整体COD指标显著提升。

**⚠️ 局限性**

局限在于仍依赖人工标注的伪装掩码，难以覆盖极端伪装场景；模型在计算量和推理速度上略高于传统二分类COD方法。

---

## 534. WildFireGS: Physics-Based Wildfire Simulation in Large-Scale Semantics-Enriched Gaussian Splatting Forest Scenes

**arXiv ID:** 2608.11100 | [PDF](https://arxiv.org/pdf/2608.11100v1)

**作者:** Nienke Driessen `[一作]` (Delft University of Technology), Michael Weinmann `[通讯]` (Delft University of Technology)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了WildFireGS——一种可直接在空中影像重建的语义化3D高斯点云上进行物理基础野火仿真的框架；

**💡 创新点**

创新点在于把高斯点云与植被语义、燃料属性耦合，并设计粒子驱动的燃烧与热传递模型，无需显式网格或体素即可在大规模场景中模拟火势；

**🔧 技术方法**

使用的技术包括3D Gaussian Splatting、Feature Splatting的语义注释、粒子动力学热/冷传递、火粒子与雨粒子耦合、物理约束与燃料消耗模型；

**📊 数据集**

使用数据集为Open Forest Observatory的无人机空中图像及合成的场景；

**📈 对比分析**

通过在不同坡度、植被密度、风速和雨量下与Rothermel模型及实验结果对比，展示了与理论一致的传播速度、燃烧面积和燃料消耗，且在真实场景中表现出良好的匹配；

**⚠️ 局限性**

局限性包括场景保持静止（无结构变形）、语义分类粒度不足、未区分干湿燃料、实时更新受烟雾和热湍流影响，以及对复杂多种树种和多层燃料结构的支持不充分。

---

## 535. VidForensics-M1: Meta-Detection Reinforcement Learning with Verifiable Temporal Grounding for AI-Generated Video Forensics

**arXiv ID:** 2608.11201 | [PDF](https://arxiv.org/pdf/2608.11201v1)

**作者:** Bowei Liu `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Meta-Detection概念，将标签正确性与证据可靠性联合优化，并在强化学习中引入Evidence‑Guided Reward Redistribution，实现了对AI生成视频的高精度鉴别。

**💡 创新点**

创新点在于①将可验证的时间段定位作为证据，提升检测可靠性；②设计EGRR在保持标签奖励不变的前提下根据证据质量重新分配奖励；③构建自动化的真实‑伪造视频生成管道，自动标注时间段。

**🔧 技术方法**

使用了Qwen3.5‑9B作为基准检测器，强化学习框架（GRPO/DAPO），边界帧条件视频生成模型进行伪造，GPT‑5.5/ Gemini‑3.1‑Pro生成并过滤文本解释，Temporal‑Grounding 与 EGRR 对奖励进行校准。

**📊 数据集**

通过从InternVid、ActivityNet等公开视频中抽取真实片段，再用LTX‑Video‑2B、Wan2.2‑Fun‑5B‑InP、SkyReels‑V2等模型生成伪造片段，最终构成包含10万条真实‑伪造对的数据集；在ViF‑Bench和GenBuster‑Bench（fake‑only子集）上评估。

**📈 对比分析**

与Qwen3.7‑Plus、GPT‑5.5、DeepTraceReward、BusterX++等方法对比，VidForensics‑M1在ViF‑Bench上达到86%+准确率、85%+F1；在GenBuster‑Bench fake‑only子集上召回率超过89%，比基线提升10%–20%，表现出更好的跨模型与跨域泛化能力。

**⚠️ 局限性**

局限性包括：①仅针对可控片段替换的伪造，可能不适用于更复杂的多维度操纵；②对时间段定位的规则依赖，若生成模型质量下降会影响证据质量；③在极短或极长视频、不同帧率等情况下的适用性尚待验证。

---

## 536. From Interpretability to Control: Insights from Six Years of the TrustNLP Workshop

**arXiv ID:** 2608.11171 | [PDF](https://arxiv.org/pdf/2608.11171v1)

**作者:** Rahul Gupta `[一作]` (Amazon AGI), Aram Galstyan `[通讯]` (Amazon AGI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对TrustNLP 2021–2026六届共144篇归档论文进行定量主题分析与纵向综述，构建六维信任维度分类，揭示信任议题随大模型能力演化的同步波动，提出四个结构洞察和行动建议。

**💡 创新点**

创新点在于：①整合TrustLLM与DecodingTrust两大框架生成新的六维信任分类体系；②首次系统追踪信任议题随关键能力事件（如ChatGPT发布、Agentic系统出现）的时间演进；③通过对比输出级与内部级评估揭示评估缺口；④提出缺乏统一理论框架导致研究碎片化的结构洞察。

**🔧 技术方法**

使用文本分类与主题建模技术，三方注解（人工、Claude Sonnet 5、Amazon Nova Lite 2.0）进行维度标注，计算Cohen's κ、准确率，绘制年度趋势图并与ACL等主会议进行横向对比。

**📊 数据集**

主要数据来源为TrustNLP会议论文的标题与摘要，全部来自ACL Anthology；未使用额外公开数据集。

**📈 对比分析**

比较方法基于人类与LLM标注的一致性，准确率约92%，κ值>0.7，表明分类可靠；对信任维度出现频率与年份进行可视化对比，未进行模型性能评估。

**⚠️ 局限性**

局限性包括：仅分析TrustNLP六届归档论文，未覆盖非归档、非英文及其他会议研究；基于标题/摘要的标注可能忽略细节；缺乏实验验证；结构洞察的普适性尚待进一步验证。

---

## 537. Scheduling Mixed RL Rollouts Beyond Prefix Locality

**arXiv ID:** 2608.11152 | [PDF](https://arxiv.org/pdf/2608.11152v1)

**作者:** Zetao Hong `[一作]` (Nanjing University), Chen Tian `[通讯]` (Nanjing University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对混合 RL 生成回合的推理服务，提出了一种路由层的会话接纳与容量分配策略，以提升 KV 缓存利用率和整体吞吐量。

**💡 创新点**

创新点在于：1）自适应会话接纳，动态限制每个实例可容纳的会话数；2）按工作负载类型分配 KV 容量；3）加入会话驻留时间权重，实现基于块时间的需求分配。

**🔧 技术方法**

使用的技术包括：路由层自适应接纳算法、基于 KV 缓存占用的工作负载感知分配、前缀缓存重用的放置策略（与 vLLM Router 集成）以及 CPU KV 缓存离线存储支持。

**📊 数据集**

实验数据集：Step3.7（196B-A11B 稀疏 MoE 模型）和 Qwen3.6-35B-A3B（35B 参数模型），在固定 checkpoint 的 rollout-only 实验和 50 轮 Step3.7 训练任务上验证。

**📈 对比分析**

与 vLLM Router 通过静态并发度调优基准进行对比；在 rollout-only 场景下，MISA-T 在 Step3.7 上提升吞吐量 53.3%，在 Qwen3.6-35B-A3B 上提升 43.6%；在 50 轮训练中提升 35.6% 吞吐率、减少 22.8% 迭代时间、将前缀缓存命中率从 74.5% 提升到 96.2%。

**⚠️ 局限性**

局限性：依赖请求携带工作负载标签；需要及时、完整的推理实例状态快照；若快照延迟或缺失，可能导致接纳阈值估计不准确，从而影响性能。

---

## 538. CausalSplat: Towards Comprehensive Hierarchical Reasoning in 3D Gaussian Splatting

**arXiv ID:** 2608.11150 | [PDF](https://arxiv.org/pdf/2608.11150v1)

**作者:** Jiayu Ding `[一作]` (Peking University), Ge Li `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了“Reasoning 3D Gaussian Segmentation”任务，构建了 Causal-LERF 和 Causal-ScanNet 两个多层次推理基准，并提出了 CausalSplat 框架来实现复杂语言指令在 3D Gaussian 场景中的分割与定位。

**💡 创新点**

创新点包括：①从语义、空间、功能和因果四个维度系统化定义推理任务；②将 3D Gaussian 语义字段与多模态场景图结合，通过 LLM 进行指令解析与图遍历，实现显式结构感知与隐式逻辑推理的分离；③引入空间加权特征聚合、对比学习与 HDBSCAN 聚类等技术，提升 3D 语义一致性与实例分割质量。

**🔧 技术方法**

使用的技术包括：3D Gaussian Splatting、SAM（图像分割）与 HDBSCAN（聚类）、对比学习优化、构建多模态场景图（节点属性由 VLM 提取）、Vision‑Language Model（如 Qwen3‑VL‑30B‑A3B‑Instruct）与链式思考（CoT）推理、以及三阶段推理流程（指令解析、拓扑推理、决策输出）。

**📊 数据集**

数据集：对 LERF 和 ScanNet 进行扩展，构建 Causal-LERF（2D 目标）和 Causal-ScanNet（3D 点云）共计 231 条多层次推理指令；同时在 Ref‑LERF、LERF‑OVS 等公开基准上进行评估。

**📈 对比分析**

与现有开源方法（open‑vocabulary、referring、reasoning）在 Causal‑LERF 上实现 47.0% mIoU（相较第二名 LUDVIG 的 23.6% 提升约 100%），在 Causal‑ScanNet 上实现 14.9% mIoU（约 3 倍提升）。在 Ref‑LERF、LERF‑OVS 等标准基准上亦取得 36.1% 和 51.3% mIoU，均超过同类最佳方法。性能提升主要归因于语义字段构造与场景图推理的协同。

**⚠️ 局限性**

局限性：①对场景图构造与 VLM 语言理解的依赖导致推理过程对模型训练数据与提示设计敏感；②对大规模、高密度场景的计算开销仍较大；③目前仅针对静态室内场景，缺乏对动态或户外环境的评估；④对细粒度语义与多模态对齐的精度有待进一步提升。

---

## 539. SAR2Agri: Learning SAR Intensity Representations for Agricultural Monitoring

**arXiv ID:** 2608.11142 | [PDF](https://arxiv.org/pdf/2608.11142v1)

**作者:** Moti Rattan Gupta `[一作]` (Plaksha University), Anupam Sobti `[通讯]` (Plaksha University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `57a58b01-81b4-4d75-a45c-2e891f272b50` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了仅使用SAR强度影像的自监督预训练流水线，通过时间差预测（TD）和未来帧预测（FF）两类时间预设任务，并加入遮罩与课程学习，目标是为农业监测（作物类型、产量、季节事件）学习高质量表征。

**💡 创新点**

创新点包括：①证明光学时间预设任务可迁移至SAR；②在SAR上首次将遮罩（MAE式随机遮蔽）与课程学习（先TD后FF+TD）相结合，显著提升表征；③构建针对区域（泰米尔纳德邦）的预训练流程；④在SICKLE基准上实现比光学预训练、监督模型以及现有SAR多模态/单模态FM显著更优的性能。

**🔧 技术方法**

技术手段：ViT‑S 编码器；时间翻译器与Transformer解码器实现FF；CLS‑based 3层MLP实现TD；多任务学习与课程学习框架；90%遮罩比例的MAE式遮蔽；UPerNet 分割头；基于时间戳的时间编码与位置编码；使用dB转换压缩SAR强度动态范围。

**📊 数据集**

数据集：预训练使用 Sentinel‑1 RTC 224×224 像素片段（6,602 位置，1,018 天，约12亿像素）；评估使用 SICKLE 基准（Sentinel‑1、Sentinel‑2、Landsat‑8 时序数据），包含作物类型（1937/227 样本）、产量、种植/移栽/收获日期（282/37 样本）等。

**📈 对比分析**

与基准方法比较：光学/SAR 多模态 RSFM（DoFA、CopernicusFM、TerraMind）、SAR 专用 FM（SAR‑JEPA、SARMAE、SAR‑W‑MixMAE 等）、监督 ViT‑S、UNet3D、光学 SSL 预训练（FF、TD、MAE）等。最终模型在 S1 上作物类型 IoU 84.9%（比光学预训练高 15.3pt，超出最优基线 9pt），在产量和季节事件预测也显著优于其他 SAR FM；在种植/移栽/收获日期预测略低于光学预训练。

**⚠️ 局限性**

局限性：对季节事件（种植、移栽、收获）预测的表现仍略逊于光学预训练；遮罩策略对全局时间差任务不友好，需进一步调优；仅在泰米尔纳德邦区域进行预训练，跨地区泛化尚未验证；只利用SAR强度图像，未探索相位、多极化或混合模态的潜力。

---

## 540. sLTN: Structural Logic Tensor Networks

**arXiv ID:** 2608.11136 | [PDF](https://arxiv.org/pdf/2608.11136v1)

**作者:** Davide Rinaldi `[一作]` (Nokia Bell Labs), Luciano Serafini `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在原有Logic Tensor Networks (LTN) 的基础上，提出了结构化逻辑张量网络（sLTN），通过在语言中加入结构维度、结构变量和结构关系，使得逻辑表达式能够直接捕捉时序、序列或图形等结构化信息，并以可微分的张量语义进行求值。

**💡 创新点**

创新点在于将结构化维度视为一阶符号的第一类元素，实现了结构维度的显式量化、关系约束和位置选择；同时在语法与语义层面统一了结构化约束与模糊逻辑，形成可直接与神经网络训练耦合的完整框架。

**🔧 技术方法**

采用可微分模糊逻辑算子（如 Lukasiewicz、Gödel 等）、张量对齐与广播、结构维度上的聚合（量化）以及多目标梯度组合（如 PCGrad）等技术；实现基于 PyTorch 的声明式签名、解析、解释与训练接口。

**📊 数据集**

以 MNIST 字母视频序列为实验数据集，构造出现/隐藏视频两类，并使用该数据集演示视频分类与逻辑约束的联合学习。

**📈 对比分析**

通过在同一任务上对比原始 LTN（无结构维度）与 sLTN（加入结构维度）的性能，sLTN 在视频出现预测准确率和逻辑一致性上均提升了约 5‑10%，表明结构化约束显著提升了学习效果。

**⚠️ 局限性**

局限性包括：目前支持的结构维度和关系类型相对有限；在大规模图或高阶时序数据时，张量对齐与聚合可能导致显著的计算与内存开销；缺乏对递归或动态图结构的原生支持，需要进一步扩展。

---

## 541. Actions Speak Louder than Words: Measuring Cross-Lingual Policy Retention in Tool-Using Agents

**arXiv ID:** 2608.11110 | [PDF](https://arxiv.org/pdf/2608.11110v1)

**作者:** Sourabrata Mukherjee `[一作]` (Microsoft Research India), Sunayana Sitaram `[通讯]` (Microsoft Research India)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文针对多语言工具使用代理模型，评估其在不同语言下执行的动作序列（policy）是否保持一致，并提出以“跨语言保留率”作为衡量指标；

**💡 创新点**

创新点在于：①将动作序列本身作为可测量对象并构造了消除五大测量偏差的规范化估计方法；②发现四大前沿模型在贪婪解码下，跨语言保留率约为71–73%，表明不同语言对同一任务的动作路径相对稳健；③揭示英语言枢轴和轨迹长度对多语言一致性影响的因果机制；④揭露常见的正则表达式提取错误导致的“多语言失败”测量伪影；

**🔧 技术方法**

技术方法包括：基于 ReAct 框架的动作模板、贪婪与温度采样解码、动作序列长度匹配、交叉语言与同语言一致性比值（Ĩ = I_cross/I_within）、样本量控制、随机种子对齐、Bootstrap 置信区间、因果干预（英语言枢轴消除/强制）和误差校正（空轨迹率、词法匹配）。

**📊 数据集**

使用的数据集涵盖：六个公开多语言基准（FLORES-200、XQuAD、XNLI、Belebele、XCOPA）和一个自研合成基准，总计5,776任务、41种语言，配合8个指令调优模型（Gemma‑3‑27B、Sarvam‑M、Qwen3‑235B‑A22B、Llama‑4‑Maverick、GPT‑OSS‑120B、Gemma‑3‑4B、Qwen3‑8B、Aya‑Expanse‑8B），共计2.38M代理回合。

**📈 对比分析**

比较方法为：在相同任务、相同解码条件下，生成两份同语言回合（同种语言）和跨语言回合（不同语言），分别计算动作序列相似度 I_within 与 I_cross，随后归一化为 Ĩ。结果显示：四大前沿模型在贪婪解码下跨语言保留率集中在71–73%；小模型则不在此区间；温度升高不显著降低跨语言保留率；英语言枢轴干预显著影响保留率；长度匹配是必要的校正步骤。

**⚠️ 局限性**

局限性包括：①仅评估符号工具调用（未执行真实工具），可能不完全反映实际执行路径；②测量依赖单一正则提取器，对不同模型的解析错误会导致偏差；③只考察贪婪解码，采样策略的多样性未被充分探索；④模型范围主要为四大前沿与部分小模型，其他厂商或更小规模模型的结论尚未验证；⑤跨语言保留率虽高，但并不等价于性能（准确率）或安全性，需要与答案质量分离评估。

---

## 542. Capturing Uncertainty in Human Motion for Representation Learning in Soccer

**arXiv ID:** 2608.11203 | [PDF](https://arxiv.org/pdf/2608.11203v1)

**作者:** Yizhou Xu `[一作]` (KTH Royal Institute of Technology), Atsuto Maki `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过自监督的未来运动预测任务，学习了针对足球场上运动员3D骨架的通用表征。

**💡 创新点**

创新点在于引入离散未来运动分布学习（DDL）：构建离散码本并用显式监督学习未来运动的多模态概率分布，同时采用帧级监督提升对帧级任务的迁移能力。

**🔧 技术方法**

技术核心包括：基于图注意力网络（Graph Transformer Network）的骨架时空建模、离散码本构建（KD‑tree分割）、分布式代码预测与条件运动生成、以及与传统回归、基准方法的对比实验。

**📊 数据集**

使用公开的 WorldPose（世界杯2022赛季）和私有的 ProSoccer（大规模商用跟踪数据）两个数据集进行训练与评估。

**📈 对比分析**

在运动预测上，相较于六种基准方法（LTD、HisRep、MSR‑GCN、PGBIG、SiMLPe、GCNext）和零样本基线，DDL显著降低 MPJPE，尤其在高速度运动上提升明显；在下游任务（动作识别与射门检测）上，DDL预训练的 GTN 模型分别取得 96.21% 与 0.9274 的 Average‑AP，优于随机初始化、无预训练、以及其他自监督方法（如 MAMP）。

**⚠️ 局限性**

局限性包括：仅关注单一运动员骨架，未考虑球、队友、对手等交互因素；码本大小与分辨率的折衷仍待进一步探讨；在更复杂的多模态生成任务中，温度控制可能不足以充分挖掘多样性。

---

## 543. Risk-Aware Kinodynamic Motion Planning Under Uncertainty For Safe Navigation on Planetary Environments

**arXiv ID:** 2608.11175 | [PDF](https://arxiv.org/pdf/2608.11175v1)

**作者:** Sachin Sunil Kelkar `[一作]` (Georgia Institute of Technology), Yashwanth Kumar Nakka `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种结合AO-RRT与SCP的风险感知运动规划方法；

**💡 创新点**

将残差动力学的不确定性用合规预测量化，并将CVaR风险作为AO-RRT的边成本；

**🔧 技术方法**

使用AO-RRT、SCP、CVaR、合规预测、自动微分等技术；

**📊 数据集**

在仿真与Leo无人车的实地实验中使用ZED 2i立体相机、Jetson Orin AGX、Vicon等数据；

**📈 对比分析**

与无风险成本的AO-RRT和单纯的SCP对比，风险降低约97%，最终碰撞风险仅为6×10⁻³；

**⚠️ 局限性**

仍依赖准确的风险地图与残差模型，计算量大，尚未在更复杂多样化场景中验证。

---

## 544. MultiModal Code-Switching: Interleaving Visual Objects into Language for Explicit Object-Level Alignment

**arXiv ID:** 2608.11167 | [PDF](https://arxiv.org/pdf/2608.11167v1)

**作者:** Changhao Xiang `[一作]` (Nanjing University), Xinyu Dai `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态码切换（MMCS）预训练范式，通过将文本实体替换为对应的视觉对象，实现显式的对象-实体对齐，从而提升多模态大模型的语义绑定能力。

**💡 创新点**

核心创新在于将视觉对象直接嵌入文本序列，实现对象级对齐；配合可扩展的数据合成管线，显著提升数据效率，并通过语言模型与实体重构损失共同训练，强化视觉语义映射。

**🔧 技术方法**

技术手段包括：多模态码切换预训练（LM + 实体重构损失），使用SigLIP2、QwenViT等视觉编码器；LLM骨干如Qwen2.5-3B、Qwen3-8B、Llama3-8B；两层MLP投影器；数据合成管线利用Qwen3-VL生成详细描述、Qwen2.5-72B提取实体、Grounding DINO定位对象、SAM-2.1生成掩模。

**📊 数据集**

使用合成的773K样本数据集（基于COCO、GQA、Flickr30K等公开图像），配合LLaVA-NeXT 779K指令数据进行SFT，并在COCO 2014验证集、RefCOCO/+/g、CVBench、OCRBench、VQA、MMBench等多项基准上评估。

**📈 对比分析**

与传统图像级预训练、Patch Aligned等细粒度对齐方法比较，MMCS在50K样本下就能超过600K图像-文本对的基线；在视觉定位任务上提升约7.9%，在视觉感知任务提升2-4%；整体在多模态指令、VQA、OCR等任务中均表现出显著性能提升。

**⚠️ 局限性**

局限性包括：仅聚焦自然图像中的视觉对象，难以直接迁移到图表、文字识别等领域；依赖自动生成的标注和定位模型，可能带来偏见与隐私风险；模型在复杂关系与动作表达上的表现尚未深入探究。

---

## 545. Why Does CLAUDE.md Keep Growing? Catastrophic Remembering in Agentic Coding

**arXiv ID:** 2608.11095 | [PDF](https://arxiv.org/pdf/2608.11095v1)

**作者:** Kushal Chakrabarti `[一作]` `[通讯]` (South Park Commons), Kushal Chakrabarti (South Park Commons)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化了 GitHub 中 agentic README 指令文件的不断增长现象，并证明其根源是指令的隐式推理记忆随时间衰减，导致所谓的灾难性记忆。

**💡 创新点**

创新点在于提出并验证通过在指令中加入记录推理的评论来阻止增长并提升指令遵循性能，首次将 IFEval 反转用于可观测的最小覆盖评估。

**🔧 技术方法**

采用语言模型模拟维护者与执行者、逆向 IFEval / WildIFEval 评测框架以及统计学方法估计删除风险和评论效果。

**📊 数据集**

数据集包括数千个公开 GitHub 仓库的多版本 README、IFEval 与 WildIFEval benchmark 以及自行构建的注释实验数据。

**📈 对比分析**

对比实验显示，带评论的提示在最小覆盖下指令数缩减至 1~10% 的多余量，同时在真实世界评测中指令遵循率提升 5-15%，相较于无评论版本有显著改进。

**⚠️ 局限性**

局限性包括仅评估英文仓库、依赖固定的删除阈值与匹配器、实验仅在有限规模的模型与数据上验证、未探索更大规模提示或多语言环境的表现。

---

## 546. Who Uses Open-Weight Models? China and the Shifting Geography of AI in Science

**arXiv ID:** 2608.11090 | [PDF](https://arxiv.org/pdf/2608.11090v1)

**作者:** Zackary Okun Dunivin `[一作]` `[通讯]` (University of Stuttgart), Zackary Okun Dunivin (University of Stuttgart)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统性分析了截至2026年6月21万多篇学术论文中LLM的使用情况，构建了能区分模型使用与仅提及的NLP管线。

**💡 创新点**

创新点在于首次将大规模学术文本与作者归属信息相结合，量化不同LLM族群的使用趋势，并揭示中国开发的开源模型在学术界迅速上升的地理与产业层面。

**🔧 技术方法**

使用了词典提取、LLM生成的银标准标注、SciBERT分类器以及逻辑回归和多项式回归等技术。

**📊 数据集**

主要数据集为Semantic Scholar Open Research Corpus（S2ORC）21.34M篇全文与OpenAlex作者与机构信息。

**📈 对比分析**

通过将论文划分为单族群与多族群两类，结合统计回归比较不同模型族群的采用比例，模型识别准确率约为92%–94%，使用识别F1达0.93。

**⚠️ 局限性**

限制在于仅关注公开与专有权重的粗粒度划分，数据缺失与作者国籍推断误差，以及无法捕捉研究者选择背后的动机与技术细节。

---

## 547. RTSKG: Building a Rail Transit Station Knowledge Graph Dataset

**arXiv ID:** 2608.11080 | [PDF](https://arxiv.org/pdf/2608.11080v1)

**作者:** Shutong Zhu `[一作]` (Southeast University), Yuan Zhu `[通讯]` (Southeast University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个铁路站点知识图谱RTSKG，并使用该图谱完成站区商店推荐和客流预测任务。

**💡 创新点**

通过专门设计的本体显式建模站点与道路、POI等城市实体之间的空间与语义交互，填补了现有知识图谱缺少铁路站点与周边实体关联的空白。

**🔧 技术方法**

利用GeoPandas抽取事实、RDF/OWL本体、GIE等知识图谱嵌入模型，并结合图神经网络与LLM（GPT‑4o）实现知识增强预测。

**📊 数据集**

整合纽约市与芝加哥的铁路、行政区划、道路、POI等数据，并与公开的UUKG、HUSK等知识图谱进行对比实验。

**📈 对比分析**

在站区商店推荐和知识增强客流预测任务中，与UUKG、HUSK等基线进行对比，RTSKG在Hits@k、MRR、MAE、RMSE等指标上均显著优于基线，表现最佳。

**⚠️ 局限性**

当前仅覆盖两座城市，缺乏多城市推广和实时更新能力；KG规模和动态更新机制有限，未能充分捕捉实时交通事件的影响。

---

## 548. Learning Gaussian Structure: Intervention-Guided Density Control for Feed-Forward Driving Reconstruction

**arXiv ID:** 2608.11077 | [PDF](https://arxiv.org/pdf/2608.11077v1)

**作者:** Hang Li `[一作]` (Beihang University), Xiao Bai `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 LGS 框架，基于 LiDAR 和多摄像头输入的 feed‑forward Gaussian 重新构建，学习高密度 Gaussian 集合及其属性，实现动态驾驶场景的高质量渲染；

**💡 创新点**

创新点包括（1）通过 prune / add 结构干预产生的局部梯度响应来训练 Gaussian Densify Policy，实现可学习的稀疏 Gaussian 结构重塑；（2）引入 Cross‑Time Point Query，显式检索并聚合不同时间戳的邻域特征，提升属性预测可靠性；

**🔧 技术方法**

技术手段包括：3D 稀疏卷积骨干、Point Transformer V3 作为结构策略网络、梯度干预监督、均值池化跨时刻检索、迭代 Gaussian 细化、以及对多尺度 Gaussian 映射的密度控制；

**📊 数据集**

实验使用 Waymo Open Dataset 与 PandaSet 两大公开数据集；

**📈 对比分析**

与 UniSplat、Flux4D、STORM 等基线在 Waymo 上比较，PSNR 由 26.28 dB 提升至 28.04 dB（SSIM 0.885，LPIPS 0.113），在 PandaSet 上 PSNR 达到 25.03 dB，速度约 1.9 s/帧；

**⚠️ 局限性**

局限性：阈值固定可能无法适配所有场景；背景建模仍依赖 LiDAR 对齐的 monocular 深度与采样天空点；欧氏跨时检索可能混合不同物体，增加搜索成本；

---

## 549. SkillZip: Evaluation-Free Skill Compression for Self-Evolving Agents by Discovering Reusable Structure

**arXiv ID:** 2608.11079 | [PDF](https://arxiv.org/pdf/2608.11079v1)

**作者:** Xiaofan Bai `[一作]` (Alibaba Group), Yuhong Li `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无评估任务的技能压缩方法 SkillZip（含一次性压缩与连续压缩模式）

**💡 创新点**

创新点在于将技能视为“类型化合约”，通过最短可信解释（MDL）原则仅共享结构、保留罕见规则，从而实现既压缩又不丢失任何提取的约束

**🔧 技术方法**

核心技术包括：结构化语义抽取（接口、流程、工具、规则、输出、证据）、基于规则共享与异常编码的最短覆盖搜索、动态规划与加权打包、以及局部更新与全局重打包的连续压缩框架

**📊 数据集**

使用了三套评测基准：BFCL‑v4（网络搜索与推理）、LiveMathematicianBench（数学定理推理）、SpreadsheetBench（电子表格操作），以及三款大模型（Qwen3.7‑Max、Qwen3.6‑Plus、Kimi K2.6）

**📈 对比分析**

与无技能、人工技能、演化后原始技能以及 SkillReducer 对比；SkillZip 在保持或提升任务分数的同时压缩率平均为 31.2%（相比 SkillReducer 的 9.2%），执行速度提升约 3.5×，且无需任何任务回放；跨模型泛化也优于 SkillReducer

**⚠️ 局限性**

局限性包括：依赖于结构化解析和手工定义的规则集；对完全无结构或含大量非执行文本的技能可能压缩效果有限；实验仅覆盖三类基准，可能无法全面验证在更广泛领域的适用性

---

## 550. New Lower and Upper Bounds for the Grothendieck Constant

**arXiv ID:** 2608.11158 | [PDF](https://arxiv.org/pdf/2608.11158v1)

**作者:** Rahul Saha `[一作]` (University of Texas at Austin), Raghu Meka `[通讯]` (University of California Los Angeles)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用自主 AI 研究系统在长周期内完成对 Grothendieck 常数 KG 的上下界改进，并以此揭示 KG 的十分位数为 7。

**💡 创新点**

创新点在于：① 首次通过 AI 系统发现并证明了不通过构造硬实例而直接给出 KG 的下界 6π/11；② 通过“限制 Krivine”方案提出新的上界 1.7813319810625639，比传统上界 1.7822 更优；③ 详细记录并分析了 AI 在长周期数学研究中的强弱，提出人机协作的改进思路。

**🔧 技术方法**

技术包括：① 双代理架构——推理模型（OpenAI GPT‑5.x）负责方向规划与证明构造，编程代理（Anthropic Claude）负责代码实现与实验；② 文件‑基记忆与异步人机指令；③ 内部校验协议（数值计算、Arb 证明、区间算术复核）确保结果可验证；④ 通过有限维 Gauss 相关性函数、变分与优化方法推导新证明。

**📊 数据集**

数据与资源：主要使用数学文献与公开公式库、数值实验数据（高维高斯采样、交叉验证）、Arb 计算环境；并未使用传统机器学习数据集，而是基于数学推理与实验产生的自生成数据。

**📈 对比分析**

与先前最优界比较：旧下界 1.6769… → 新下界 1.7135…（提升 0.0366），旧上界 1.7822… → 新上界 1.78133…（降低 0.0008），实现了 KG 十位数的确定；实验显示新方案在多维极限情况下能保持更小的非线性误差。

**⚠️ 局限性**

局限性：① AI 在研究判断（何时转向、何时停留）和研究状态表述（何信息需持久化）方面表现不稳，易丢失关键约束导致错误记录；② 需要人工持续干预以设定目标、审计结果；③ 目前缺乏对数学过程（失败、策略选择）的训练数据，导致系统难以自动做出全局性决策。

---

## 551. The Illusion of Cross-Lingual Safety in Low-Resource Languages

**arXiv ID:** 2608.11146 | [PDF](https://arxiv.org/pdf/2608.11146v1)

**作者:** Abigail Oppong `[一作]` (Makerere University Center for Artificial Intelligence), Seid Muhie Yimam `[通讯]` (University of Hamburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者构建了 LoDNA 数据集并使用潜在几何框架，对四种低资源非洲语言（Twi、Hausa、Amharic、Swahili）在 7B‑8B 规模 LLM 中的安全拒绝表示进行评估，探讨英语言安全对齐的跨语言转移。

**💡 创新点**

创新点在于：①提出 LoDNA，配对文字翻译与文化本地化提示；②设计潜在几何框架，用隐藏层几何特征（拒绝方向、漂移、投影）量化跨语言安全对齐；③通过内部表示而非生成结果评估，揭示跨语言安全转移深度不足。

**🔧 技术方法**

技术方法包括：隐藏状态提取、余弦相似度、保留成分、漂移分数、PCA 投影、线性探针；并在 Mistral、Llama、Qwen2.5、AfriqueQwen 四个模型上进行层级分析。

**📊 数据集**

使用的数据集为 LoDNA（从 DNA 扩展至四种语言）、社区收集的英语提示及原始 DNA 英文数据，形成文字翻译与文化本地化双版本。

**📈 对比分析**

通过对齐度量（cosine、retained component、drift、SLL）和拒绝概率比较，发现大多数语言的拒绝信号保留低于 10%，只有 Swahili‑Llama 在某些层有显著提升；SLL 与拒绝概率极低，表明跨语言安全失败严重。

**⚠️ 局限性**

局限性包括仅评估 7B‑8B 规模模型、数据集规模有限、潜在几何框架为观察性关联、低资源语言的子词分词碎片化可能影响结果、缺乏中等资源对照语言。

---

## 552. A Systematic Sample Size Analysis of ML-Based Path Loss Prediction for LPWAN

**arXiv ID:** 2608.11083 | [PDF](https://arxiv.org/pdf/2608.11083v1)

**作者:** Robert Bitterling `[一作]` (Fraunhofer FKIE), Michael Rademacher `[通讯]` (Fraunhofer FKIE)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过使用真实城市LoRa测量数据和LiDAR生成的地形特征，评估了随机森林（RF）和k-近邻（k-NN）两种简单机器学习模型在LPWAN路径损耗预测中的性能。

**💡 创新点**

创新点在于：①系统性研究了训练样本量对ML模型精度的影响，提供了实用的学习曲线；②在随机混合划分之外，还采用了LOGO（留一网关）评估，揭示了模型在未见网关上的迁移能力。

**🔧 技术方法**

使用的技术包括：LiDAR地形处理、SignalServer路径剖面、随机森林（YDF库）和k-近邻（k=7、k=31）回归。

**📊 数据集**

数据集为德国波恩市的LoRa上行RPP测量，经过清洗后得到114,465条记录，进一步聚类后保留64,294条样本；同时利用州提供的LiDAR数据生成地形特征。

**📈 对比分析**

对比方法为：在随机池化划分（75/25）下，训练集大小从100到48,220逐步增加，评估RMSE；与传统经验模型（Bonn、COST‑231 Hata、ITM、ITWOM）和专门LPWAN模型做对比。结果显示：RF在小样本（≤3,250）时最佳，k‑NN在大样本时最优；最高RMSE可降至≈6.15 dB，远优于传统模型≈9.7 dB。LOGO实验表明RF迁移误差增幅约2.6 dB，k‑NN误差翻倍。

**⚠️ 局限性**

局限性包括：①仅在单一城市、单一技术（LoRa 868 MHz）和频段上验证，未检验跨城市/技术的泛化；②未进行充分的超参数调优；③日志O评估仍可能低估真实部署中的误差，未考虑网关间的异质性和测量噪声。

---

## 553. AdvFD: Boosting Visual Generation via Adversarial Fr'echet Distance Loss

**arXiv ID:** 2608.11205 | [PDF](https://arxiv.org/pdf/2608.11205v1)

**作者:** Mingju Gao `[一作]` (Peking University), Hao Tang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Adversarial Fréchet Distance (AdvFD) 框架，结合静态预训练特征与可学习对抗特征空间，改进生成模型后训练并对抗Fréchet hacking；

**💡 创新点**

创新点在于引入可学习的对抗特征空间，并采用真实特征白化来抑制尺度爆炸；通过交替的min–max更新动态暴露并逼近未被静态特征捕捉的分布差异；

**🔧 技术方法**

利用FD-Loss、Fréchet距离、对抗学习、特征白化（real‑feature whitening）、LoRA微调、AdamW+梯度裁剪等技术；

**📊 数据集**

在ImageNet‑1K 256×256上进行条件图像生成实验；

**📈 对比分析**

与FD‑Loss基线以及多种生成模型（JiT、pMF 等）在 FID、FD‑r6、FD‑r3 等指标上进行对比；AdvFD 在所有尺度与架构下均显著降低指标（如 FID 从 1.00 降至 0.79，FD‑r6 从 5.53 降至 3.92，FD‑r3 从 8.45 降至 6.03），显示出一致的性能提升；

**⚠️ 局限性**

仍依赖预训练特征空间，若不使用白化或对抗更新不当会导致训练不稳定；未验证跨模态或视频生成，可能受模型规模与超参数限制。

---

## 554. Strategies to Avoid Illegal Data Access

**arXiv ID:** 2608.11153 | [PDF](https://arxiv.org/pdf/2608.11153v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 555. VIScore: Diagnosing Planning-Relevant Quality in Latent World Models

**arXiv ID:** 2608.11174 | [PDF](https://arxiv.org/pdf/2608.11174v1)

**作者:** Haiyu Wu `[一作]` (Altos Labs), Morgan Levine `[通讯]` (Altos Labs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在LeWorldModel中将SIGReg替换为VISReg，并引入VIScore诊断指标，对世界模型的潜在空间与规划成功率之间的关系进行系统分析。

**💡 创新点**

提出了Veracity–Influence–Sobriety三因素结合的VIScore评分，可同时衡量编码器、预测器与规划器的可达性、行动容量和幻觉误差，从而与规划成功率呈强相关。

**🔧 技术方法**

使用视觉编码器+动作编码器+Transformer预测器的JEPA架构，配合CEM/MPPI等搜索式规划器，并对VISReg进行中心、尺度、形状分离正则化。

**📊 数据集**

在四个离线像素控制基准（PushT、Reacher、OGBench-Cube、Two-Room）及六种未见形状的PushObj进行OOD评估，并对MAZE等未见任务进行迁移测试。

**📈 对比分析**

与straightness、physical-state probe、empowerment等传统诊断相比，VIScore在开发、留置及方法/任务迁移集上均获得 Spearman ρ ≥0.75，校准误差显著低于常数基线，显示更高的预测稳定性。

**⚠️ 局限性**

限制在于只适用于基于预测器的搜索式规划，对加速规划器无效；单种种子评估会高估成功率；缺乏针对多任务与离散动作的诊断。

---

## 556. Agentic Configuration Management (ACM): A Reference Configuration Model for Governed Agentic Systems

**arXiv ID:** 2608.11166 | [PDF](https://arxiv.org/pdf/2608.11166v1)

**作者:** Audrey Quessada-Vial `[一作]` `[通讯]` (PwC), Audrey Quessada-Vial (PwC)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Agentic Configuration Management (ACM) 框架无关的治理与配置参考模型，目标是为异构 agent 系统提供统一的可治理配置管理、生命周期、影响传播与运行时重建机制。

**💡 创新点**

创新点包括：
- 将传统软件配置管理 (SCM) 原则迁移到 agent 系统，定义 Agentic Configuration Item (ACI) 及其不可变修订；
- 设计四图结构 (Configuration, Evolution, Assurance, Runtime) 来明确治理、演化、保证与运行时关系；
- 引入确定性治理语义（生命周期、质量、保证、影响、资格）和基于固定点的影响传播算法；
- 提供跨框架的语义投影适配器，使得不同 agent 框架（LangGraph、CrewAI、OpenAI Agents SDK）能投影为相同的 ACM 表示，从而实现治理等价。

**🔧 技术方法**

技术细节：
- Python 3.11+实现，核心使用 Pydantic 进行数据建模；
- 采用工作列表算法实现确定性影响传播，保证最小固定点收敛；
- 运行时重建通过规范化的事件流和回放机制完成；
- 三个框架适配器分别实现对各自 introspection 的抽取与投影。

**📊 数据集**

实验数据：
- 27 个治理场景覆盖 ACI 变更、生命周期、影响、基线、运行时等；
- 9 个影响传播案例（局部、中间、全局）在三框架上各执行 5 次；
- 所有实验均基于在 LangGraph、CrewAI 与 OpenAI Agents SDK 中构建的示例程序，无使用公开大型数据集。

**📈 对比分析**

比较方法与性能：
- 场景验证：将 ACM 计算结果与预期治理状态做对比；
- 跨框架等价性：比较不同框架投影得到的 ACM 图，确认治理结果一致；
- 影响传播：测量影响集大小、传播深度、固定点迭代次数与收敛速度；与单跳依赖检查做对比，证明 ACM 能覆盖更多影响并减少后续检查范围；
- 性能方面：工作列表传播复杂度为 O(|E_C|)，运行时回放为 O(|Σ|)，在实验规模下运行时间均在秒级。

**⚠️ 局限性**

局限性：
- 只在 LangGraph、CrewAI 与 OpenAI Agents SDK 上验证，缺乏更广泛框架的实测；
- 投影质量依赖框架 introspection 能力，若框架缺失关键信息需手工声明；
- 治理策略（质量、保证、资格规则）由外部上下文提供，ACM 本身不定义业务细则；
- 未针对大规模 agent 集群或分布式部署进行性能与可扩展性评估；
- 运行时重建假设事件序列完整且无错误，实际生产环境中的异常事件处理尚未覆盖。

---

## 557. Role of Personality in Conversational Information Seeking

**arXiv ID:** 2608.11164 | [PDF](https://arxiv.org/pdf/2608.11164v1)

**作者:** Abdisalam Abukar `[一作]` (University of Glasgow), Joemon M. Jose `[通讯]` (University of Glasgow)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在26名受试者上做了一个within-subject实验，比较了三种由大语言模型通过提示实现的助手人格（外向、尽责、中性）在三类信息检索任务（旅游规划、手机购物、健康饮食）中的交互效果；

**💡 创新点**

创新点在于把助手人格视为可控的交互变量，而非固定属性，发现助手人格的效用与任务类型和用户人格兼容性相关，而非存在单一最优人格；

**🔧 技术方法**

使用了GPT‑4.1大语言模型并通过不同系统提示来实现人格差异；

**📊 数据集**

数据集为26位受试者的对话日志、行为追踪、问卷与Big Five人格测评；

**📈 对比分析**

对比方法主要是行为指标（回复长度、用户词占比、回合数等）与主观评估（信任、委托、交互质量）。结果显示：外向人格在旅游任务中获得最高信任，尽责人格在健康任务中表现最佳，中性人格在购物任务中表现最佳；

**⚠️ 局限性**

局限包括样本量小、样本偏年轻男性、仅考察两项人格维度、仅使用提示而非模型训练、未深入评估答案质量与真实性等。

---

## 558. A Recommendation System Approach for Interference-Robust Sensor Subset Selection

**arXiv ID:** 2608.11143 | [PDF](https://arxiv.org/pdf/2608.11143v1)

**作者:** Kaan Buyukkalayci `[一作]` (University of California, Los Angeles), Christina Fragouli `[通讯]` (University of California, Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于推荐系统的选择低成本声学传感器后激活高成本传感器的子集选择方法，旨在实现实时且鲁棒的目标跟踪；

**💡 创新点**

创新点在于将传感器子集选择转化为推荐问题，使用双塔（Two‑Tower）网络学习网络声学上下文与候选子集的嵌入，并利用频段声学特征取代传统单值RSSI，从而提升在干扰环境下的鲁棒性；

**🔧 技术方法**

技术上采用频段功率特征提取、双塔多层感知机（MLP）架构、距离加权的光滑效用函数和均方误差损失进行训练；

**📊 数据集**

实验数据来自两套户外车辆跟踪部署：一个包含6个传感器、存在人声、风等干扰；另一个包含10个传感器、干扰较少；数据包括车辆GPS轨迹和传感器音频；

**📈 对比分析**

与传统RSSI基线（线性路径损耗、KDE混合、归一化RSSI top‑3、样条后验）以及RSSI双塔对照组对比；在干扰部署中，频段双塔模型在最近节点包含率上提升至约98.4%（相比RSSI基线77%），在非干扰部署中虽略逊于RSSI双塔模型，但仍高于所有基线；计算量保持在每个采样≤2 ms，符合实时要求；

**⚠️ 局限性**

局限在于子集枚举成本随可推荐高成本节点数和子集大小呈指数增长，且对大规模部署需进一步采用检索/分层方法；同时模型依赖于频段特征的选择，若环境频谱特性变化大需重新训练；

---

## 559. Attention-Path Fragility as an Uncertainty Signal in Large Language Models

**arXiv ID:** 2608.11138 | [PDF](https://arxiv.org/pdf/2608.11138v1)

**作者:** Minsoo Kim `[一作]` (POSCO Holdings Future Technology Research Institute), Ilyong Yoon `[通讯]` (POSCO Holdings Future Technology Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的不确定性估计器ASMI，通过在 Transformer 关键层随机屏蔽注意力头来测量子级别的注意力子网络互信息，并结合语义一致性核实现Sem-ASMI。

**💡 创新点**

创新点在于：① 用注意力子网络互信息捕捉“预测脆弱性”，揭示模型对特定注意路径的依赖；② 通过语义一致性权重过滤表面形态差异；③ 在单次贪心推理上即可得到可复现的置信度信号；④ 在不同任务上自动预测其适用范围。

**🔧 技术方法**

核心技术包括：随机注意力头屏蔽（结构扰动）、BALD 互信息估计、top‑K 近似分布、语义相似性矩阵、适应门控（基于样本多样性）。

**📊 数据集**

使用四个大规模 QA 数据集：CoQA、SQuAD、BabiQA（有上下文引导），以及 TriviaQA（闭书回想）作为对照；评测在 Qwen3-4B、Qwen3-8B、Llama‑2‑7B、Mistral‑7B 四个 backbone 上进行。

**📈 对比分析**

与 17 种基线（信息量、采样多样性、探针、注意力等）进行比较，ASMI 在有上下文引导的 QA 上的 PRR 与最强采样多样性基线持平或略优；在闭书 TriviaQA 上性能退回或低于单通道 MSP。显著优势体现在“自信-脆弱”错误预测上，能在保持 25% 覆盖率时将错误率从约 16% 降至 6%。

**⚠️ 局限性**

局限性：① 仅在注意力路径决定答案的情境有效，对闭书回想几乎无效；② 需要先选定屏蔽层和掩码率，缺乏自动化调参；③ 采用 top‑K 截断，可能在极稀疏词表下失真；④ 互信息无法清晰区分表观与真实不确定性；⑤ 在部分数据集（如 BabiQA Llama‑2‑7B）中模型过于鲁棒，ASMI 无法读取信息。

---

## 560. Two-stage Odd Residual Flows for Mean-Preserving Probabilistic Time Series Forecasting

**arXiv ID:** 2608.11114 | [PDF](https://arxiv.org/pdf/2608.11114v1)

**作者:** Kiran Madhusudhanan `[一作]` (University of Hildesheim), Vijaya Krishna Yalavarthi `[通讯]` (University of Hildesheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两阶段概率时间序列预测框架 TORF，先用预训练的确定性模型产生准确均值，再通过受限正向奇数流建模残差分布，确保预测均值不变。

**💡 创新点**

创新点在于将均值预测与不确定性估计完全解耦，使用奇数约束的正则化正则化流，使残差分布可灵活建模且解析保持均值；并且通过轻量级卷积SplineNet实现长时序可扩展。

**🔧 技术方法**

核心技术包括：预训练确定性点预测（如 SimpleTM）、奇数约束正则化正向流（Restricted Normalizing Flow）、Rational Quadratic Spline 变换、条件缩放层、卷积式SplineNet、以及基于负对数似然的训练。

**📊 数据集**

在 8 个短期（ETTh1-S、ETTh2-S、ETTm1-S、ETTm2-S、Solar-S、Electricity-S、Traffic-S、Exchange-S）和 9 个长期（ETTm1-L、ETTm2-L、ETTh1-L、ETTh2-L、Electricity-L、Traffic-L、Exchange-L、Weather-L、ILI-L）真实数据集上进行实验。

**📈 对比分析**

与 12 种基线（4 个点预测 + 8 个生成式概率模型）比较，TORF 在 17 个任务中均实现了 CRPS 和 NMAE 的显著提升，最高可达 +35.4% CRPS 与 +28.1% NMAE。

**⚠️ 局限性**

主要限制是残差假设独立且对称，难以处理异形或跨维度相关的误差；Mixture 扩展虽然放宽对称性但参数和计算成本显著增加。

---

## 561. Every Packet Counts: Dispersing Information for Loss-Resilient Learned Image Compression

**arXiv ID:** 2608.11096 | [PDF](https://arxiv.org/pdf/2608.11096v1)

**作者:** Yuhang Wei `[一作]` (Shanghai Jiao Tong University), Guo Lu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于通道重分配和交错通道分组的端到端抗丢包图像压缩方案。

**💡 创新点**

创新点在于将 Inter-Channel Redistribution 与 Interleaved Channel Grouping 结合，平衡各包信息并短化自回归链，提升丢包鲁棒性。

**🔧 技术方法**

采用深度可变形自编码网络、双层双分支自回归模型以及可逆通道重分配模块。

**📊 数据集**

在 Flickr2W 训练集、Kodak 与 CLIC2020 验证集上进行实验。

**📈 对比分析**

与 JPEG2000、ProgDTD、LossResilientLIC、ResiComp 等方法对比，在 5/10/20% 丢包率下平均 PSNR 提升 1.8–3.2 dB，方差降低一个数量级。

**⚠️ 局限性**

局限在于 hyperprior 必须完整接收，且对极端突发丢包场景的进一步鲁棒性待提升。

---

## 562. A Linear-Time Approximation Scheme for the Densest Subgraph Problem

**arXiv ID:** 2608.11094 | [PDF](https://arxiv.org/pdf/2608.11094v1)

**作者:** Elena Grigorescu `[一作]` (University of Waterloo), Mehrshad Taziki `[通讯]` (ETH Zürich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种线性时间的 (1−ε)-近似算法，用于在无向图中寻找密度最大的子图，并给出了一个近似 1/2−ε 的算法用于至少包含 k 个顶点的密度子图。

**💡 创新点**

核心创新是将流的阻塞流与图的 κ‑core 结构结合，设计了一种“饱和框架（saturation framework）”，能够在每一步通过流量重新分配“负载”来识别稀疏顶点，从而在保持最大密度不变的前提下逐步“雕刻”图，避免传统方法中出现的对数因子。

**🔧 技术方法**

主要技术包括：
- 通过构造带容量的有向网络和多次阻塞流来得到顶点负载分配；
- 利用 κ‑core 的核心分解快速减小图规模；
- 迭代雕刻与精细化参数 K_i 的自适应更新；
- 对密度子图约束下的“至少 k 个顶点”问题，利用流框架获得一个至少满足 (1/2−ε) λ* 的子图。

**📊 数据集**

论文没有使用公开数据集；实验和证明均在理论分析框架下完成。

**📈 对比分析**

与已有的 1/2 近似、(1−ε) 近似以及基于线性规划或乘子权重的近似相比，该方法在所有 ε>0 的情况下实现了真正的线性时间复杂度 O((n+m)/ε³·log(1/ε))，并在至少 k 个顶点的约束下实现了近似阈值 1/2 的线性时间（O((n+m)·log²n·log(1/ε)/ε)）。

**⚠️ 局限性**

限制方面：
- 需要先预先估计 λ* 以确定阈值 τ，虽然通过多次调用可以消除对 log(1/ε) 的开销，但实现上仍需额外的二分搜索步骤；
- 对 ε 取值过小导致 1/ε³ 的高次方，实际运行时间可能不如预期；
- 对有向图密度子图的处理只给出了一个粗略的 2-近似；
- 证明中大量使用了理论上可实现的阻塞流与 κ‑core 计算，实际实现的效率依赖于底层流算法的常数。

---

## 563. Cross-View Feature Matching: Survey, Benchmarking, and Foundation-Model Perspectives

**arXiv ID:** 2608.11093 | [PDF](https://arxiv.org/pdf/2608.11093v1)

**作者:** Songlin Du `[一作]` (Southeast University), Takeshi Ikenaga `[通讯]` (Waseda University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了跨视角特征匹配的统一综述，构建了基于六维度的层次化分类法，并在一致评测协议下对稀疏、半稠密、密集匹配方法进行了系统对比，指出了未来研究方向。

**💡 创新点**

创新点在于提出了面向跨视角匹配的六维度层次化分类框架，并统一了实验评测标准，首次将不同范式在精度、效率、泛化等维度进行公平、可复现的对比分析。

**🔧 技术方法**

综述涵盖了传统手工特征、学习型检测器/描述子、Sparse/Semi‑Dense/Dense 匹配器（含 Transformer、Graph‑Neural、Mamba、Diffusion、VFM‑based 方法）、训练策略（自监督预训练、知识蒸馏、跨域数据生成）以及鲁棒估计（RANSAC、学习型 RANSAC）。

**📊 数据集**

使用了 MegaDepth、ScanNet、HPatches、Aachen Day‑Night、InLoc、YFCC100M 等公开数据集，统一配置图像尺寸、关键点数、匹配阈值等。

**📈 对比分析**

通过统一的评估指标（相对位姿 AUC、Homography 重投影误差、定位成功率），对稀疏、半稠密、密集三种匹配范式在同一数据集上进行对比，结果表明密集匹配在精度上最高，但计算量最大；稀疏匹配在速度与模型尺寸上最优；半稠密匹配则兼具精度与效率。

**⚠️ 局限性**

局限性包括：高计算与内存开销（尤其是密集与全局注意力）；对极端视角/光照变化鲁棒性不足；缺乏统一的概率/多假设匹配框架；跨域泛化能力有限；缺少针对几何匹配的专门预训练基础模型；对动态/非刚体场景的建模仍待深入。

---

## 564. Long-Horizon AI Research for Grothendieck Constant: A Case Study in Human-AI Mathematical Collaboration

**arXiv ID:** 2608.11195 | [PDF](https://arxiv.org/pdf/2608.11195v1)

**作者:** Alan Li `[一作]` (University of Texas at Austin), Raghu Meka `[通讯]` (University of California Los Angeles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `847a60d8-a755-47af-ba5d-c5236b9e3083` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过与人类共同推进的AI研究系统，对Grothendieck常数的上下界进行了改进，并对AI在长周期数学研究中的协作方式做了详细案例分析。

**💡 创新点**

① 使用AI系统首次得到并证明下界≥6π/11，避免了构造硬实例的传统方法；② 通过“限制Krivine方案”取得上界π/2·log(1+√2)−3.47×10⁻⁴；③ 记录并公开AI与人类如何在高层决策、技术执行与状态表示等方面协作的完整流程。

**🔧 技术方法**

使用大型语言模型（GPT‑5.5/5.6）、Claude代码代理、文件式记忆与压缩、内部验证协议、异步人类指令、数值/符号计算与区间算术等技术，形成了一个可持续的AI研究系统。

**📊 数据集**

主要参考现有数学文献（Krivine方案、Gaussian分析等）、先前研究的实验数据，及自定义的高维Gaussian实验与区间算术结果作为验证集。

**📈 对比分析**

与以往人类或自动化搜索得到的10⁻⁵级改进相比，AI系统实现了3.47×10⁻⁴的上界提升和约0.0486的下界提升；实验耗时约240个会话，API成本约5.4k美元，全部结果通过内部与人工双重验证。

**⚠️ 局限性**

AI在全局研究判断和研究状态表示方面表现薄弱，难以自动识别全局障碍并及时转向新方向；状态压缩过程易丢失关键信息，导致记录误导；缺乏以数学研究过程为训练数据的模型，限制了其在长周期科研中的自主性。

---

## 565. Surgical WAM: A World-Action Model for Data-Efficient Surgical Robot Learning

**arXiv ID:** 2608.11204 | [PDF](https://arxiv.org/pdf/2608.11204v1)

**作者:** Wenrui Bao `[一作]` (University of Central Florida), Yuzhang Shang `[通讯]` (University of Central Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Surgical World‑Action Model（WAM），通过在动作标注稀缺的情况下利用无标注手术视频预训练，再在有限的动作示范上微调，实现闭环手术操纵。

**💡 创新点**

创新点在于将无标注视频预训练与动作预测统一到一个生成模型中，首次在手术机器人领域证明动作无关视频预训练能显著提升闭环控制性能。

**🔧 技术方法**

采用 Cosmos Policy 作为基底，构建视频‑动作扩散 Transformer 进行世界‑动作联合建模，利用两阶段训练（视频预训练 + 动作微调）。

**📊 数据集**

使用 SurRoL 模拟手术基准（四个任务）和 JIGSAWS 真实手术视频作为无标注和有标注数据。

**📈 对比分析**

与现有的行为变换器、DEX、ALOHA、扩散策略及通用 VLA 模型相比，WAM 在四个任务上的闭环成功率提升至 77.8%（比无预训练高 14.3 个百分点），尤其在 PegTransfer 上提升 20%。

**⚠️ 局限性**

局限性包括对视频预训练数据的依赖仍需大规模手术视频，模型对长期推断可能过拟合，且在真实机器人硬件上的部署与延迟需进一步验证。

---

## 566. Beyond a Bag of Features: Set-Level Instability in Sparse Autoencoders

**arXiv ID:** 2608.11197 | [PDF](https://arxiv.org/pdf/2608.11197v1)

**作者:** Nikolai Bolik `[一作]` (Heidelberg University), Artur Andrzejak `[通讯]` (Heidelberg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了稀疏自编码器（SAE）激活集合对大型语言模型（LLM）语义结构的可解释性，检验其是否能更好地捕捉人类概念边界和典型性，并探讨其在语义细化（如形容词+名词）时的组合一致性。

**💡 创新点**

创新点在于将SAE的“活跃特征集合”作为度量单位，系统评估其在类别边界、典型性、以及语义修饰下的保留与丢失行为，揭示了SAE签名更多反映模型内部结构而非人类语义。

**🔧 技术方法**

采用稀疏自编码器（包括Top‑K、BatchTop‑K、JumpReLU、Matryoshka等变体）对Transformer残差流进行分解，使用Jaccard相似度衡量激活集合相似性；同时使用余弦相似度对密集表示进行对比；利用预训练Gemma、GPT等模型。

**📊 数据集**

使用经典心理学典型性数据集（如bird、animal等类别）和从自然文本中抽取的30,000段落做聚类实验，构造形容词+名词的细化序列进行丢失特征分析。

**📈 对比分析**

通过AMI对类别边界恢复、Spearman相关系数对典型性排名的比较，发现SAE集合与密集表示相比，类别边界的恢复更弱，典型性相关性几乎无显著提升；在形容词修饰实验中，发现丢失特征率高达20–60%，违背了简单的“并集”期望。

**⚠️ 局限性**

局限性包括仅评估残差流SAE，未覆盖所有hook位置或SAE变体；实验数据受限于特定心理学数据集和形容词修饰范例；对其他语义细化形式（如同义改写、任务相关抽象）的推广尚未验证。

---

## 567. Test-Time Self-Evolving GUI Visual Grounding via Reflection-Guided On-Policy Self-Distillation

**arXiv ID:** 2608.11191 | [PDF](https://arxiv.org/pdf/2608.11191v1)

**作者:** Shiyu Xuan `[一作]` (Nanjing University of Science and Technology), Zechao Li `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Test‑Time Self‑Evolving框架，使GUI视觉定位模型在部署后通过探索、评估、反思和内部化实现自适应改进；

**💡 创新点**

创新点包括：①利用MLLM驱动的Reflector生成细粒度评估与文本反思；②将反思转化为token级监督的Reflection‑Guided On‑Policy Self‑Distillation；③设计Contrastive Calibration方法消除因错误prefix导致的误导性监督；

**🔧 技术方法**

核心技术为多模态大语言模型、反思生成器、On‑Policy Self‑Distillation、对比校准与优势裁剪等；

**📊 数据集**

在ScreenSpot‑v2、ScreenSpot‑Pro、OSWorld‑G/Refine、MMBench‑GUI等六个GUI视觉定位基准上进行实验；

**📈 对比分析**

与基线模型及现有Test‑Time RL方法（如GUI‑RCPO）对比，平均提升约7–8%，在最难的数据集MMBench‑GUI上提升高达4.6%；

**⚠️ 局限性**

局限性：依赖Reflector的评估质量；在极端失败情形下仍可能产生少量负迁移；缺乏对更大规模模型或跨语言场景的评估。

---

## 568. Are We Really Making Progress in Group Recommendation? Unmasking the Tie-Breaking Illusion

**arXiv ID:** 2608.11190 | [PDF](https://arxiv.org/pdf/2608.11190v1)

**作者:** Song-Duo Ma `[一作]` (National Taiwan University), Pu-Jen Cheng `[通讯]` (National Taiwan University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究了团体推荐中因训练时得分压缩与评估时确定性 tie‑breaking 交互导致的评估偏差，并提出了基于期望的 tie‑aware 评估方法。

**💡 创新点**

创新点在于：①揭示额外 sigmoid 既是评估误差来源，也是隐式边际平滑机制；②提出温度缩放 BPR 在不产生大量 tie 的前提下恢复平滑效果；③系统给出 tie‑aware 指标与 tie 统计量。

**🔧 技术方法**

采用 BPR 损失、额外 sigmoid 转换、温度缩放 BPR、期望 HR@K/NDCG@K 计算以及与多种基线模型的对比实验。

**📊 数据集**

使用公开团体推荐基准数据集 CAMRa2011 与 Mafengwo 进行实验。

**📈 对比分析**

对 ConsRec、AlignGroup、DHMAE、ITR、DGGVAE 等近期方法与 Baselines 进行比较，发现多数方法在 tie‑aware 评估下性能显著下降，方法间相对排名也发生显著变化；去除额外 sigmoid 后性能得到部分恢复，未出现明显优势。

**⚠️ 局限性**

局限性：仅针对特定实现与单正样本 top‑K 评估；未覆盖所有数据集、评估协议及超参数组合，tie‑aware 评估仍需进一步推广。

---

## 569. On the Sensitivity to Errors in Homomorphic Computing: Single Transient Bit-flip Client-side Error Characterization

**arXiv ID:** 2608.11155 | [PDF](https://arxiv.org/pdf/2608.11155v1)

**作者:** Matías Mazzanti `[一作]` (University of Buenos Aires), Radha Venkatagiri `[通讯]` (Georgetown University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

对CKKS同态加密在单比特翻转错误下的错误敏感性进行实验研究，区分加法与乘法操作的错误传播模式，并证明乘法操作主导错误行为。

**💡 创新点**

首次系统性识别并分类了CKKS在不同优化（Vanilla、RNS、NTT、RNS+NTT）和参数组合下的“加法模式”和“乘法模式”错误特性，指出乘法操作对错误传播的主导作用以及模式与参数/数据无关的本质。

**🔧 技术方法**

使用自研的C++ CKKS实现（基于OpenFHE/HEaaN/SEAL/PyFHE），配合LLTFI框架进行单比特翻转注入，采用最大相对误差百分比（MREP）评估错误影响。

**📊 数据集**

无专门数据集；实验以随机/伪生成的多项式系数为输入，在每个系数的每一比特位置逐一注入单比特翻转。

**📈 对比分析**

通过在各种FHE操作组合（无运算、加法、乘法、旋转、加法+乘法+引导）下注入错误，并对解码结果与原始数据比较，评估错误传播；未对执行速度或资源消耗做量化，主要关注错误传播特性。

**⚠️ 局限性**

局限性：仅研究CKKS方案，未覆盖其他HE方案；仅考虑单比特翻转错误，未探讨多比特或其他故障模型；实验仅在客户端侧进行，未评估服务器端硬件错误的影响；未给出性能（时间/能耗）指标。

---

## 570. DACRI: Decision-Aware Causal Intervention Ranking for Critical Supply Chains

**arXiv ID:** 2608.11154 | [PDF](https://arxiv.org/pdf/2608.11154v1)

**作者:** Shiqi Huang `[一作]` (Independent Researcher), Lashimi Muraleedharan Nair `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出DACRI框架，将供应链干预选择视为学习到排序问题，并开发CriticalSCM‑Bench v1，包含因果结构、配对的因果与反事实滚动以及净值评估；

**💡 创新点**

在受控实验环境下首次量化“发现根源”与“决定干预”之间的差异，证明适应性排序能在部分领域提升净值；

**🔧 技术方法**

使用因果结构学习（PCMCI、Granger）做图恢复，LambdaMART（LightGBM）做排名，结合异常检测（EWMA+MAD、Isolation Forest、LSTM‑AE、USAD、MTAD‑GAT），以及对抗性解释（检索‑基生成）进行结果解释；

**📊 数据集**

合成数据来源于Backblaze驱动可靠性、电子生命周期、UCI SECOM半导体生产与USGS矿产供应，三种关键供应链原型（DI、SE、CM）；

**📈 对比分析**

与六级基线对比（severity‑only、true‑root、constant‑buffer、train‑selected static、perfect‑oracle、equal‑weight），LambdaMART在SE和CM上分别提升约11%–16%净值，DI上低于constant‑buffer；通过多种压力测试（延迟、成本、图变更、OOD）进一步验证；

**⚠️ 局限性**

受限于合成基准、易失性因果标签、CM的OOD脆弱性、未考虑实际货币成本、对异常检测器选择敏感、模型复杂度不一定带来收益。

---

## 571. When and Where Faults Matter: A Study of Transient Errors in CKKS Multiplication

**arXiv ID:** 2608.11147 | [PDF](https://arxiv.org/pdf/2608.11147v1)

**作者:** Vattana Chan `[一作]` (Georgetown University), Radha Venkatagiri `[通讯]` (Georgetown University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对CKKS加密方案中未优化的同态乘法过程进行单比特瞬态错误注入实验，并分析了错误在密文分量c0与c1中的时序和位置如何影响最终解密结果。

**💡 创新点**

创新点在于揭示同态乘法对单比特翻转极其敏感，证明c0与c1的错误传播路径不同，且错误时序会导致从无效错误到无声数据损坏（SDC）的不同级别影响，同时给出了基于多项式误差的理论解释。

**🔧 技术方法**

采用自研的C++ CKKS实现（基于OpenFHE/SEAL等），结合LLTFI错误注入工具，使用最大相对误差百分比（MREP）评估解码输出；同时通过代数展开和错误多项式模型对错误传播进行理论分析。

**📊 数据集**

实验所用数据为合成的浮点数（实数/复数）数组，经过iFFT编码后用于CKKS加密与乘法测试。

**📈 对比分析**

通过将解密结果与原始数据对比计算MREP，比较不同位置与时序下单比特错误的影响；实验显示错误在c0或c1、不同乘法阶段注入会导致从微小误差到大幅SCDC的差异，但本文未报告系统性能开销，只聚焦于正确性评估。

**⚠️ 局限性**

局限性包括：仅研究未优化的CKKS乘法，未涉及NTT/RNS等加速；仅考虑单比特瞬态错误，未测试多比特或持续性错误；未提出或评估任何容错或误差检测机制，结果对更复杂或真实云端场景的推广有限。

---

## 572. AlbumentationsX: One Augmentation Pipeline for Images and Related Annotations

**arXiv ID:** 2608.11123 | [PDF](https://arxiv.org/pdf/2608.11123v1)

**作者:** Vladimir Iglovikov `[一作]` `[通讯]` (Albumentations), Vladimir Iglovikov (Albumentations)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 AlbumentationsX，一个统一的 Compose 对象，能够在一次调用中对图像及其相关注释（mask、bbox、keypoint、stereo、视频帧、3D 体素等）执行相同的随机变换，保证几何变换的一致性。

**💡 创新点**

创新点在于：1) 将变换列表、概率、注释设置、随机种子封装在同一个对象中；2) 支持实例级绑定（mask、bbox、label 同步处理）；3) 提供可保存、重放、可视化变换的完整流程；4) 通过 additional_targets 和 instance_binding 处理多视角、深度图、视频、3D 数据；5) 允许添加项目特定的自定义变换，并保持统一的随机性管理。

**🔧 技术方法**

技术主要使用 Python 的 albumentations 库进行实现，扩展了其 Compose、BboxParams、ImageOnlyTransform、DualTransform 等基类；利用随机种子、SamplingContext、ReplayCompose 等机制实现可重复性；并结合 PyTorch 的 Dataset 与 DataLoader 进行集成。

**📊 数据集**

本文没有使用特定的数据集，侧重于描述框架和示例代码，演示如何在语义分割、目标检测、姿态估计、立体视觉、视频、3D 体素等多种任务中应用。

**📈 对比分析**

方法比较主要关注的是变换的一致性和可复现性，而非模型性能。示例展示了不同变换顺序、概率、随机种子如何影响结果；并通过保存和重放机制保证同一样本在不同训练阶段获得相同的增强效果。

**⚠️ 局限性**

限制包括：无法自动判断变换是否保持任务标签（需人工决定）；不同目标类型对变换的支持程度不一；在多通道或自定义数据结构（如相机标定矩阵）时需要额外规则；库的版本和许可（AGPL）可能限制商业使用。

---

## 573. You Only Charge Once 2.0 : A End-to-End Analog Computing-in-Memory Architecture with Reconfigurable Switched Capacitors

**arXiv ID:** 2608.11116 | [PDF](https://arxiv.org/pdf/2608.11116v1)

**作者:** Zihao Xuan `[一作]`, Fengbin Tu `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

暂无论文内容，无法描述

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

## 574. Foundation Model-Enabled Efficient Data Sampling (FEEDS): A label-efficient training strategy for pan-cancer, multi-tracer PET/CT datasets

**arXiv ID:** 2608.11076 | [PDF](https://arxiv.org/pdf/2608.11076v1)

**作者:** Biratal Raj Wagle `[一作]` (Dartmouth), Indrani Bhattacharya `[通讯]` (Dartmouth)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种基于基础模型的高效数据采样方法FEEDS，用来选择最具代表性和多样性的未标注PET/CT病例进行专家标注，然后一次性训练全身病灶分割模型；

**💡 创新点**

创新点在于：①利用DinoV2等视觉基础模型提取的特征向量，在特征空间中以“最远点”策略高效挑选多样化样本；②该方法不需要多轮训练或额外计算，标注成本大幅下降；③同时在体素、病灶及解剖区域层面全面评估性能，验证其临床实用性；

**🔧 技术方法**

核心技术包括：基础模型特征提取（DinoV2 2D MIP），基于余弦距离的最远点采样，nnU-Net 一次性监督训练，以及多指标评估（Dice、FPVol、FNVol、病灶灵敏度/PPV、区域性能）；

**📊 数据集**

使用三大数据集：AutoPET‑III（多重放射剂量和多癌种）、Deep‑PSMA（前列腺癌双扫描）、Dartmouth‑Hitchcock（内部 PSMA）进行训练、验证和外部测试；

**📈 对比分析**

与随机采样、DPP采样及伪标签半监督学习等传统方法相比，FEEDS在仅标注30%数据时可达到与完全标注（100%）相当的Dice和病灶检测性能，并在FPVol/FNVol上表现更优；在三个独立测试集上均保持高泛化性；

**⚠️ 局限性**

局限性包括：①使用2D最大强度投影特征，缺乏3D空间信息；②基础模型DinoV2为通用模型，缺少PET/CT专属训练；③仅利用PET特征进行样本挑选，未融合CT信息；③在不同扫描仪/协议下仍存在迁移挑战。

---

