# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-09-10 | 今日论文总数: 514

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. AppetiteCheck: Feasibility of Momentary Vagus Nerve Stimulation as an Implicit Intervention for Eating Behavior

**arXiv ID:** 2609.09700 | [PDF](https://arxiv.org/pdf/2609.09700v1)

**作者:** Tan Gemicioglu `[一作]` (Cornell Tech), Tanzeem Choudhury `[通讯]` (Cornell Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究开发了一种可携式、低注意力的经颈静电迷走神经刺激装置（AppetiteCheck），用于在分心进食期间即时干预，减少食物摄入量。

**💡 创新点**

创新点在于首次将瞬时非侵入性迷走神经刺激作为隐式、无意识干预方式应用于进食行为，突破了传统基于认知或感知提示的饮食干预方法。

**🔧 技术方法**

采用了 FDA 认证的 Truvaga Plus 手持设备进行经颈迷走神经刺激，结合心率监测与自述量表评估生理和主观效果；实验设计为受试者内随机对照，刺激与安慰剂（肩部刺激）交叉比较。

**📊 数据集**

使用了 24 名健康参与者的实验数据，涵盖四种常见零食（薯片、糖果、饼干、葡萄干）以及对应的食物重量、进食时长、心率与 HRV 指标，形成完整的行为与生理数据集。

**📈 对比分析**

通过对照实验与线性混合效应模型分析，发现 tcVNS 条件下食物摄入量平均降低 9.6%，进食速率降低 23.6%，后食饱腹感与安慰剂相当，心率显著下降，说明该方法在短期内对进食行为具有显著且可量化的影响。

**⚠️ 局限性**

局限性包括：实验仅在实验室进行、样本量有限、受试者为健康人群且未收集体重/BMI 信息、安慰剂刺激位置不同导致感官体验差异、刺激对心率的影响不确定与 HRV 指标无显著差异，且长期效果与日常使用可行性尚未验证。

---

## 2. Learning Terrain-Adaptive Humanoid Locomotion on Granular Terrain

**arXiv ID:** 2609.10286 | [PDF](https://arxiv.org/pdf/2609.10286v1)

**作者:** Junnosuke Kamohara `[一作]` (Georgia Institute of Technology), Ye Zhao `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于三维RFT的粒子地形接触模型，并通过教师-学生强化学习实现人形机器人在沙地等颗粒地形上的自适应步态控制。

**💡 创新点**

创新点在于①用完整的3D RFT无经验式接触模型替代传统刚体或经验式法则；②结合变分自编码器压缩地形信息，实现在线地形感知与自适应；③在真实Unitree G1机器人上完成首次在天然颗粒地形上实现行走、跑步与跳跃。

**🔧 技术方法**

技术包括三维RFT接触模型、NVIDIA MPM粒子模拟、IsaacLab强化学习框架、PPO+VAE 教师-学生结构、TCN地形编码器、域随机化与训练课程。

**📊 数据集**

使用由 NVIDIA Newton MPM 生成的高保真颗粒场景（基质：玄武岩、干沙、海滩沙等）以及 Unitree G1 机器人收集的传感器日志；没有公开公开数据集。

**📈 对比分析**

与传统刚体接触、2D RFT、弹性圆锥模型等基线对比，3D RFT+教师-学生策略在仿真与真实测试中成功率提升 20–30%，速度跟踪误差降低 30% 以上，并能在不同硬度颗粒层实现无缝过渡。

**⚠️ 局限性**

局限包括：①仅验证在平坦颗粒层，未测试斜坡或波纹地形；②对极软或高粘度颗粒缺乏足够域随机化；③模型参数需针对不同材料手工拟合，缺乏通用性；④计算量仍高，需高性能 GPU。

---

## 3. Robust Rank Aggregation for Multimodal Speech-Based Alzheimer's Disease Detection

**arXiv ID:** 2609.09948 | [PDF](https://arxiv.org/pdf/2609.09948v1)

**作者:** Zemin Jin `[一作]` (Chinese University of Hong Kong), Tomoko Matsui `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于秩聚合和置信门随机森林的多模态语音 AD 检测框架。

**💡 创新点**

创新点在于使用归一化预测秩聚合解决概率尺度不匹配，并通过置信门随机森林在高置信度时进行可解释的错误纠正，避免了后验校准。

**🔧 技术方法**

采用 HuBERT 语音特征、RoBERTa 文本特征、逻辑回归基分类器、秩聚合、置信门随机森林以及互信息特征选择等技术。

**📊 数据集**

使用 ADReSS2020 与 ADReSSo2021 两个公开 AD 语音数据集进行实验。

**📈 对比分析**

与现有方法比较，在 ADReSS2020 上实现 95.83% 的最高准确率，在 ADReSSo2021 上实现 90.14%，均优于或接近目前最优成绩。

**⚠️ 局限性**

局限性包括仅在两大数据集上验证，缺乏对不同语言、更多临床人群以及其他异构分类器的泛化评估。

---

## 4. An Explainable Machine Learning Framework for Predicting Blood-Brain Barrier Permeability Using Molecular Descriptors

**arXiv ID:** 2609.10012 | [PDF](https://arxiv.org/pdf/2609.10012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 5. HyperTrace: Hypothesis-Based Preference Tracing for Online LLM Personalization

**arXiv ID:** 2609.09835 | [PDF](https://arxiv.org/pdf/2609.09835v1)

**作者:** Jianzhi Shen `[一作]` (Johns Hopkins University), Muhammad Shafique `[通讯]` (NYU Abu Dhabi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练‑free 的在线个性化框架 HyperTrace，通过对用户隐含偏好进行粒度细致的推断与更新，实现对语言模型生成的持续、可解释的个性化。

**💡 创新点**

创新点在于：① 将用户偏好建模为可观测的自然语言假设粒子集合，并用 SMC（Sequential Monte Carlo）样式的权重更新实现在线推理；② 在粒子集中同时维护短期话题意图与长期稳定偏好，形成层次化记忆；③ 采用 LLM 作为“代理选择模型”对每个假设的效用进行估计，保留多种可解释解释并跟踪证据积累。

**🔧 技术方法**

主要技术包括：自然语言粒子假设、SMC 重要性权重、基于 LLM 的代理选择模型（Bradley–Terry 近似）、高效的分支与采样策略、主题标记与跨会话记忆合并、以及基于 LLM 的评价与对齐得分。

**📊 数据集**

实验使用公开的两大个性化数据集：PRISM（真实用户反馈+问卷）和 PersonaMem‑v2（模拟多会话反馈）。

**📈 对比分析**

与 CoT、RAG、Dynamic Cheatsheet、HyperAlign、Hydra 等基线相比，HyperTrace 在偏好预测、响应对齐以及个人化档案一致性上均取得更高得分，特别是在长期会话转移点保持更稳健的表现；在 PRISM 上实现了最高的 Acc_>20（0.6136）和 Prof.（4.1857）等指标，性能优于所有对比方法。

**⚠️ 局限性**

局限性包括：① 仅依赖显式的选择反馈，无法直接从口头批评或隐式行为中提取偏好；② 对 LLM 的推理准确性与细粒度受限；③ 需要对自然语言假设进行人工可解释性评估，可能在多样化用户中出现歧义；④ 由于使用黑盒 LLM，某些推断可能受模型偏差影响。

---

## 6. In RAG We Trust? Measuring Robustness of Retrieval-Augmented Generation Under Document Poisoning

**arXiv ID:** 2609.09243 | [PDF](https://arxiv.org/pdf/2609.09243v1)

**作者:** Iliano Fasolino `[一作]` `[通讯]` (University of Milan), Iliano Fasolino (University of Milan)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估检索增强生成模型在文档被恶意篡改时的鲁棒性，使用4‑bit Llama 3.1 8B模型在FEVER验证任务上进行588次因子实验。

**💡 创新点**

引入三种文档毒化策略（实体替换、数字替换、否定），量化其对检索增强生成模型准确率、误导率、放弃率的影响，并通过查询层bootstrap估计不确定性。

**🔧 技术方法**

采用MiniLM检索器、FAISS向量索引、4‑bit量化的Llama 3.1 8B生成器，温度0.1解码，并使用关键字匹配、词汇重叠代理等指标评估。

**📊 数据集**

基于FEVER验证集的19,597条证据句子，加10条信息丰富的句子，构成检索索引，使用49个事实验证查询。

**📈 对比分析**

对比清洁与毒化两组答案，计算准确率、误导率、放弃率，结果显示准确率从77.9%下降到43.5%，误导率从9.5%升至34%，放弃率从21%升至52%，实体替换对模型影响最大。

**⚠️ 局限性**

仅使用单一小型量化模型和有限查询集，解码未锁定导致噪声，关键词匹配和词汇重叠代理的可靠性有限，实验未覆盖更大模型、更多攻击方式及更严格的人类评估。

---

## 7. Data-Centric Post-Training for Financial Reasoning: Mining, Distillation, and Verifiable Learning

**arXiv ID:** 2609.10113 | [PDF](https://arxiv.org/pdf/2609.10113v1)

**作者:** Zhirayr Hayrapetyan `[一作]`, Dmitry Zmitrovich `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了一个基于三类数据源（Glaive推理、Finance‑Instruct蒸馏、GraphGen合成）的金融推理后训练数据集，并通过三种判别器实现对金融相关性、缺失上下文和答案可验证性的筛选。

**💡 创新点**

创新点在于将数据筛选拆分为阶段性决策，使用轻量级分类器实现大规模候选过滤，并结合自蒸馏、参数融合和规则式奖励的RL，显著降低SFT导致的能力遗忘。

**🔧 技术方法**

采用的技术包括Qwen3-Embedding嵌入式语义去重、三层二分类器、监督微调与自蒸馏SFT、WiSE‑FT式线性融合、Group Relative Policy Optimization（GRPO）以及规则式验证器。

**📊 数据集**

数据集涵盖了Glaive Reasoning v1 20M、Finance‑Instruct‑500K、CFA教材通过GraphGen构建的知识图谱以及经过筛选的32k验证例子。

**📈 对比分析**

在FINESSE‑Bench上，普通SFT导致3.2–4.0个百分点下降，而自蒸馏SFT提升1.0–2.8个百分点；参数融合恢复3.0个百分点，GRPO在硬样本上再提升0.4个百分点，直接RL可提升3个百分点。

**⚠️ 局限性**

局限性包括对FINESSE‑Bench的单一评测、对模型规模的依赖、以及对可验证性判断的阈值敏感，未能充分解决所有领域任务的迁移泛化问题。

---

## 8. SMCC-Empowered Digital Twins for Sensorless Monitoring in Large-Scale AI-Driven IoT Systems

**arXiv ID:** 2609.09161 | [PDF](https://arxiv.org/pdf/2609.09161v1)

**作者:** Vincenzo Sammartino `[一作]` `[通讯]` (University of Pisa), Vincenzo Sammartino (University of Pisa)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 SMCC-DT 框架，通过 6G ISAC 波形实现无传感器的物理资产监测，并在 Edge 上联合分配感知功率、波束、内存划分与 CPU 频率，实现 AI 驱动的数字孪生同步。

**💡 创新点**

创新点在于将 Sensing–Memory–Communication–Computation 四层资源统一建模为交叉层优化，并采用 PPO‑based 深度强化学习（SmccAgent）在实时环境中学习近最优资源分配策略，从而在不部署专用传感器的情况下显著降低同步延迟和能耗。

**🔧 技术方法**

使用技术包括：6G ISAC 波形与联合波束设计、Cramér‑Rao 限界感知质量评估、内存与计算预算约束建模、端到端延迟与能耗目标的混合整数非线性规划、Proximal Policy Optimization（PPO）深度强化学习、奖励函数约束形状化。

**📊 数据集**

使用的数据集为基于 500 节点工业 IoT 场景的 Monte‑Carlo 仿真数据，模拟 200×200×50 m³ 智能工厂，包含 64 窗口天线、28 GHz 频率、32 GB 内存及多种 AI 模型尺寸（7 M、125 M、1.3 B 参数）。

**📈 对比分析**

与四种基线（Orthogonal Allocation、Compute‑Only Optimization、Greedy Heuristic、Random Allocation）对比，SMCC‑DT 在同步延迟上比 OA 低 38.7%、比 CO 低 27.2%，能耗比 OA 降低 27.4%、比 CO 降低 19.8%；在感知精度 ≥ 95% 时保持最优 Pareto 前沿。

**⚠️ 局限性**

局限性包括：实验仅在仿真环境中验证，缺乏真实 6G 硬件与多服务器联合部署验证；模型规模受内存限制，未探讨极大模型的可扩展性；PPO 训练耗时较长，实时迁移需要进一步研究；仅考虑单 BS Edge 方案，未包含多点覆盖与网络拥塞动态变化。

---

## 9. NOPE-HYPE: A Structured Simulation Workflow for Robust Speech-to-Text Across Diverse Acoustic Environments

**arXiv ID:** 2609.10058 | [PDF](https://arxiv.org/pdf/2609.10058v1)

**作者:** Niramay M. Patel `[一作]` (IISER Bhopal), Raksha Sharma `[通讯]` (IIT Roorkee)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

建立了一套名为NOPE‑HYPE的结构化训练流程，通过可控环境模拟器、PSD模板覆盖的环境缩减以及结构化超参数搜索，实现语音转文本模型在多种噪声环境下的鲁棒性提升。

**💡 创新点**

其创新点在于将噪声生成从简单的随机噪声迁移为可解释的可控模拟器，并通过PSD模板覆盖方法选择代表性环境原型，再利用有限实验空间进行高效的超参数搜索，从而使模拟噪声与真实噪声在性能上几乎等价。

**🔧 技术方法**

技术方面使用了基于PSD、包络、瞬态、聊天和可选RIR的可控音频模拟器，PSD模板覆盖与k‑prototype选择策略，以及针对Whisper和SeamlessM4T模型的27个配置的结构化超参数搜索。

**📊 数据集**

实验数据来源于Indic‑ST（英-印）和CoVoST（英-德）语音数据集，并利用DEMAND环境数据库收集真实噪声样本来拟合模拟器。

**📈 对比分析**

通过在clean、Gaussian、pink、均衡真实噪声和模拟噪声四种训练条件下对比BLEU/chrF/WER等指标，结果表明模拟噪声与平衡真实噪声的性能相近，甚至在某些设置下略优；最佳配置cfg03/06在不同语言对和环境中保持稳定的鲁棒性。

**⚠️ 局限性**

局限性包括模拟器仍依赖固定统计参数，可能无法覆盖极端或高度动态变化的环境；超参数搜索仅在三种PSD模板和27个配置内完成，缺乏对更广泛噪声类型和更大规模的泛化能力；RIR等域位移效果需进一步针对特定部署场景进行自适配。

---

## 10. PELM: Power Efficient On-Device LLM Inference with Speculative Decoding and Dynamic Voltage Frequency Scaling

**arXiv ID:** 2609.09662 | [PDF](https://arxiv.org/pdf/2609.09662v1)

**作者:** Weisi Yang `[一作]` (Northwestern University), Stephen Xia `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向移动设备的 LLM 推理功耗优化框架 PELM，结合 DVFS 频率调节、预测式解码和可变深度推理，实现对功耗与推理速度的协同调控。

**💡 创新点**

创新点包括：① 将传统 DVFS 与算法层面的自省式预测式解码（Self‑Speculative Decoding）结合；② 引入可变深度验证（Variable Depth）机制，在保持输出质量的前提下动态裁剪模型深度；③ 采用基于深度强化学习的闭环控制策略，实时权衡频率、预测长度与深度，以适应热量与性能需求。

**🔧 技术方法**

核心技术包括：自省式预测式解码与层级跳过（LayerSkip）、动态电压频率调节（DVFS）、隐藏状态缓存（Hidden Queue Cache）、深度 Q‑网络（Branch‑DQN）实现的 RL 调度器。

**📊 数据集**

评估使用多种任务数据集：GSM8K（数学推理）、NQ‑open（问答）、HumanEval（代码生成）、WMT14‑DE‑EN（机器翻译）、CNN/Daily Mail（文本摘要），并在 LLaMA‑1B、8B、13B 三种模型上进行测试。

**📈 对比分析**

与四种基线（系统默认 DVFS、静态预测式解码、FUSE、zTT）对比。PELM 在 AGX Orin 上实现 29–52% 能耗下降、10–23% 速度提升；在 Orin Nano 上保持能耗低且速度稳健。完成率（CR）几乎为 100%，并在多热环境下表现出色。任务质量（TSA）与基线基本持平，性能‑能耗比（PPJ）最高。

**⚠️ 局限性**

局限性包括：需要模型支持早期退出（Early‑Exit）以获得中间层输出；目前仅在 NVIDIA Jetson 平台测试，其他 SoC 需要重新映射 DVFS 接口；未在量化模型或多任务/后台负载环境下进行全面验证；对极端低功耗设备（如 Raspberry Pi）或非 NVIDIA GPU 的适配尚未完成。

---

## 11. A Statistical Approach to Estimating Sample Size of Machine Learning Models

**arXiv ID:** 2609.09547 | [PDF](https://arxiv.org/pdf/2609.09547v1)

**作者:** Dat Phan-Trong `[一作]` (Deakin University), Svetha Venkatesh `[通讯]` (Deakin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

做了什么：提出一种利用 ReLU 神经网络近似非线性预测表面，将其划分为局部线性区域，在每个区域内估计 R² 与 f² 效应大小，采用非中心 F 分布或大样本正态近似计算局部功效，并通过体积加权覆盖度聚合，最终通过二分搜索确定满足全局覆盖阈值的最小样本量。

**💡 创新点**

创新点是什么：将复杂非线性预测表面转化为可解析的分段线性结构，首次在局部层面实现统计功效估计并以体积加权覆盖度为全局标准，从而克服传统功效分析无法处理非参数模型的限制。

**🔧 技术方法**

用了什么技术：ReLU 神经网络代理逼近、凸多面体区域划分、局部 R² 与 f² 估计、非中心 F 分布/大样本正态近似、体积加权覆盖度、二分搜索全局样本量优化。

**📊 数据集**

用了什么数据集：在多维合成连续分段线性数据（d=2,3,4,6，结点密度 k=2,5,7,13）以及 UCI 三个真实数据集（Abalone、Concrete Compressive Strength、Liver Disorder）进行评估。

**📈 对比分析**

如何比较的方法，性能怎么样：与全局 F 检验及人工生成的真实分段函数直接对比；实验表明该方法能稳定收敛至目标覆盖率，样本量随效应阈值和模型结构的变化而变化，充分捕捉区域异质性，整体性能优于单纯全局方法。

**⚠️ 局限性**

limitation是什么：对特征密度估计的依赖导致在高维数据中可能出现欠拟合或稀疏区域；随着维度升高，ReLU 网络产生的线性区域数急剧增加，计算成本和内存占用显著上升；目前仅针对连续结果，分类和生存等其他终点的扩展尚未完成。

---

## 12. LiteRAG: Cost-Efficient Graph-Based Retrieval-Augmented Generation

**arXiv ID:** 2609.10239 | [PDF](https://arxiv.org/pdf/2609.10239v1)

**作者:** Daniel Alejandro Coll Tejeda `[一作]` (Universitat Rovira i Virgili), Daniel Barcelona-Pons `[通讯]` (Universitat Rovira i Virgili)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出 LiteRAG，一种通过查询条件化的图探索和推理链上下文构造来取代 LLM 控制的多跳检索方法。

**💡 创新点**

创新点包括：查询条件化锚点选择、动态阈值扩展、社区感知的中心点惩罚，以及将子图转化为紧凑推理链上下文。

**🔧 技术方法**

采用向量检索、并行子图扩展算法、图嵌入、社区检测以及结构化推理链生成等技术。

**📊 数据集**

使用 DistComp（分布式系统论文集合）和 UltraDomain（农业、法律、医疗三大领域）两个数据集进行实验。

**📈 对比分析**

与 GraphRAG、LightRAG、HiRAG、LinearRAG 等基线相比，LiteRAG 在 DistComp D_1280 上 Q_total 达 0.798、延迟 1.42 s、令牌 2291、成本 0.0003 $，在 UltraDomain 上 Q_total 0.801、延迟 1.24 s、令牌 4862、成本 0.0006 $，性能优于其它方法。

**⚠️ 局限性**

局限性包括：评测聚焦于 DistComp 领域、未考虑索引成本、所有系统共享同一生成模型导致跨模型鲁棒性未知、硬令牌预算控制不完全等。

---

## 13. Scored vs. Generated Readouts in Behavioral Language Models: An Empirical Study of Elicitation Format

**arXiv ID:** 2609.09882 | [PDF](https://arxiv.org/pdf/2609.09882v1)

**作者:** Touchapon Kraisingkorn `[一作]` (Amity Group), Wachiravit Modecrua `[通讯]` (Amity Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在四个零售预测任务上对同一语言模型的不同读出方式进行对比，验证并量化了“读出可互换性假设”的失效。

**💡 创新点**

创新点在于系统性测量了评分型与生成型读出的排名差距、揭示其与训练方式、任务监督程度及读出模板收敛的关联，并提出了可在现有部署中直接校正排名的实用方案。

**🔧 技术方法**

技术上使用了从 Qwen3.5‑27B 继续预训练、监督微调并可选 RL 的大语言模型；三种读出方式分别是直接读出概率、先生成推理文本后得分、以及先给出概率后再作判断；通过对每条样本的 AUC、Brier 误差及 bootstrap 置信区间进行评估。

**📊 数据集**

所用数据集包括公开的 DMBGN 电子商务优惠券领取基准、公开的 Dunnhumby Complete Journey 超市促销数据、以及两套海量内部零售客户历史与优惠券使用数据（SEA 及 US 两地）。

**📈 对比分析**

比较结果显示，除单一基线模型外，评分型读出在 12/13 组实验中均优于生成型读出，AUC 差距可达 1.5–14.5 分，且显著性检验 p≈0.003，证明了读出方式对实际目标排序的显著影响。

**⚠️ 局限性**

局限性包括仅覆盖四个二分类零售预测任务、仅评估排名准确性（AUC）而非其他指标、对公共 API 模型的读出方案受限，以及未对生成读出的排名成本进行因果干预，因而结果可能不具备跨任务或跨模型的普适性。

---

## 14. Should I Be Polite to My LLM Relevance Judge? Tone as a Severity Operating-Point Shift

**arXiv ID:** 2609.09703 | [PDF](https://arxiv.org/pdf/2609.09703v1)

**作者:** Tian Zhang `[一作]`, Meng Li `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语气（tone）对大语言模型在信息检索任务中做出的相关性判断的影响。

**💡 创新点**

提出并验证了“严重性操作点（severity operating‑point）”理论，解释语气如何通过改变模型评分的严厉度来影响与人类标注的契合度。

**🔧 技术方法**

使用线性加权 Cohen’s κ 评估绝对分数的一致性，Kendall’s τ 与 NDCG@10 检验排名稳定性，并通过查询分层交叉拟合（query‑disjoint cross‑fit）检验因果关联。

**📊 数据集**

采用 TREC Deep Learning DL19 与 DL20 数据集，共 3,498 条查询–段落对，并对 8 种 LLM 判定模型、5 个礼貌级别、每级 3 个改写语句进行实验。

**📈 对比分析**

结果显示，绝对分数的一致性 κ 随语气变化呈显著负相关（Spearman ρ≈-0.68，p≈0.02），但排名指标如 NDCG@10 变化极小（最大 Δ≈0.011），说明语气对绝对标注更具威胁。

**⚠️ 局限性**

局限性包括仅使用单一礼貌判别器、七种非推理模型、每级三种改写，未给出等价边际，且语气改写同时改变了说明文字，未能完全独立考察语气效应。

---

## 15. Integrating Multi-Source Feedback in Computational Design

**arXiv ID:** 2609.09483 | [PDF](https://arxiv.org/pdf/2609.09483v1)

**作者:** Francisco Erivaldo Fernandes Junior `[一作]` (Instituto Tecnológico de Aeronáutica), Antti Oulasvirta `[通讯]` (Aalto University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现一种多源贝叶斯优化框架，支持设计者在不具备技术背景的情况下将多种评估信号（专家意见、用户测试、模拟器、LLM等）整合到优化流程中；

**💡 创新点**

创新点在于：①独立建模每个评估源并通过线性聚合实现可解释的多源融合；②无代码工具MUSE将该框架嵌入到熟悉的设计界面，允许即时加权、动态增删源；③在六项技术评估和用户研究中展示多源方法在异构、动态、非同步评估场景下的优势。

**🔧 技术方法**

技术主要包括：贝叶斯优化、独立高斯过程（GP）代理、线性聚合与权重调节、源间不一致度量、ReactJS前端+Python+Optuna后端实现。

**📊 数据集**

数据集涵盖：多种基准函数（sin+cos、Rosenbrock、Three-Hump Camel等）用于合成实验；以及两项可视化设计任务（主题公园游客数、残奥会选手数）用于用户研究。

**📈 对比分析**

通过六个实验对比单源与多源方法：在多源采样、源间不一致、动态源调整、权重重设四个场景中多源显著优于单源；在源数增加和非平稳优化场景中两者表现相当。用户研究表明，设计者在多源条件下感知更高灵活性与信心，但对源透明度有进一步需求。

**⚠️ 局限性**

局限性包括：无法有效处理非平稳目标（多源与单源同样受限）；模拟器响应慢、透明度低影响设计者使用；聚合方法采用线性加权，可能在某些非线性交互场景下不足；对大量源的可扩展性与自动权重调节仍待研究。

---

## 16. Seven Sources of Physical AI Capability Formation

**arXiv ID:** 2609.09627 | [PDF](https://arxiv.org/pdf/2609.09627v1)

**作者:** Gang Chen `[一作]` `[通讯]` (Zyllion Data Technology Company), Gang Chen (Zyllion Data Technology Company)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过文献回顾与编码，定义并验证了七类Physical AI能力形成来源（RE、PM、EI、SE、MG、EC、ED），并证明在研究范围内已达到理论饱和。

**💡 创新点**

提出统一的“形成来源”框架，将能力来源与传统标签区分，提供可追溯、可比较的分类；通过三轮最大差异/负例检验验证了七类来源的完整性。

**🔧 技术方法**

重构归纳设计、理论饱和判据、编码规则、最大差异采样与负例挑战等方法。

**📊 数据集**

共49条证据记录，来自45公开来源（主流论文、同行评审等），覆盖至2026年9月4日的前沿预印本。

**📈 对比分析**

比较方法主要是依据来源归类评估能力形成路径，而非性能指标；文中未给出数值性能对比。

**⚠️ 局限性**

仅覆盖英文技术文献、未采用双人编码、样本量有限、未评估来源频率或效应大小、对未来技术不做逻辑完备性声明。

---

## 17. OpenDiscoveryTrace: Process Traces for Evaluating AI Scientist Workflows

**arXiv ID:** 2609.09203 | [PDF](https://arxiv.org/pdf/2609.09203v1)

**作者:** Aayam Bansal `[一作]`, Keertan Balaji `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并公开了OpenDiscoveryTrace数据集，记录了558条AI科学家在多领域任务中的完整过程轨迹，包含思考、工具调用、观察、错误等九项字段；

**💡 创新点**

创新点在于捕捉并公开过程级追踪，而非仅评估最终产出，使得能够审计推理质量、诊断失败模式并区分系统性推理与偶然成功；

**🔧 技术方法**

采用了LLM与工具调用的ReAct式架构、过程级记录方案、LLM-judge评估、统计检验（如Cliff’s δ）等技术；

**📊 数据集**

使用的主要数据集为558条轨迹，涵盖7种模型（3种前沿模型GPT‑5.4、Claude Opus 4.6、Gemini 3.1 Pro，4种开源权重模型Qwen2.5‑7B、Mistral‑7B‑v0.3、Phi‑3.5‑mini、Qwen2.5‑1.5B），覆盖药物发现、材料科学、基因组学、文献分析四大领域；

**📈 对比分析**

通过LLM‑judge对363条前沿模型轨迹进行比较，发现三款前沿模型成功率相近（84–89%），但Claude Opus 4.6的错误率是GPT‑5.4的30倍，且错误类型差异显著；

**⚠️ 局限性**

局限性包括：样本中计算机实验占大多数，缺乏物理实验数据；开源权重模型样本相对较少；人工标注的可靠性尚未完全验证；以及对模型内部推理机制的解释仍有限。

---

## 18. 3rd Place Solution to Human Motion Challenges in Real-World and Clinical Settings (MoCha) @ECCV2026: Language-Aligned Motion Representations for Domain-Generalizable UPDRS-Gait Severity Estimation

**arXiv ID:** 2609.10187 | [PDF](https://arxiv.org/pdf/2609.10187v1)

**作者:** Soojie Kim `[一作]` (UNIST), Seungryul Baek `[通讯]` (UNIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

使用Bi‑GRU编码器在SMPL序列上训练语言对齐的运动表示，并结合GPT‑5.5伪标签和参数级模型融合，实现在多源临床域下的UPDRS‑步态严重度估计；

**💡 创新点**

创新点包括：① 通过离线生成的临床描述对运动嵌入进行语义对齐；② 使用GPT‑5.5生成伪标签平衡严重度类别；③ 对每个源域独立微调后利用SVD与SCORE实现参数级融合，避免了测试时的元数据和调整；

**🔧 技术方法**

技术手段包括Bi‑GRU时间序列编码、文本编码器投影、重建解码器、GPT‑5.5伪标注、SVD/SCORE参数融合以及轻量级模型结构；

**📊 数据集**

数据集为CARE‑PD，包含4个源域（PD‑GaM、BMCLab、T‑SDU‑PD、3DGait），共2915条SMPL步态序列；

**📈 对比分析**

在MoCha 2026挑战的隐藏测试集上，宏观F1为0.57，排名第三，参数量仅637K，较大模型节省约27.7倍参数；与首位(0.69)、第二位(0.58)相比略低，但在不使用元数据或测试时校正的更严格条件下表现优异；

**⚠️ 局限性**

局限性：仍依赖离线生成的文本描述和GPT伪标注的质量；未使用测试时校正或主体分组信息，可能在极端域移情况下性能下降；仅在CARE‑PD数据集上评估，跨数据集的泛化尚未验证。

---

## 19. Introducing Consort: A Spec-First Agent Framework for Enforced, Test-Driven Development on Live Database Branches

**arXiv ID:** 2609.09671 | [PDF](https://arxiv.org/pdf/2609.09671v1)

**作者:** Kevin Hartman `[一作]` `[通讯]` (Databricks), Kevin Hartman (Databricks)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

Consort是一款spec‑first、test‑driven的 AI 编码框架，采用确定性 orchestrator、角色分离、不可编辑的测试以及在分支数据库上执行完整的 TDD 循环，以强制约束生成代码的正确性与可维护性。

**💡 创新点**

其创新点在于：①将实时分支数据库纳入测试循环，实现真正的“实地”数据验证；②使用不可编辑的测试和人类审批门控，确保模型无法规避约束；③通过确定性 orchestrator 控制流程，避免语言模型在上下文压缩中失控；④将工作拆分为专职角色代理，模拟人类团队的职责分工，提升可维护性。

**🔧 技术方法**

核心技术包括：确定性 state‑machine orchestrator、角色分离的 LLM 代理、不可编辑的测试列表、基于 copy‑on‑write 分支的数据库、以及人类批准的门控与记录机制。

**📊 数据集**

论文未使用公开数据集，而是以真实生产数据库的分支作为测试基准，利用分支数据库中保持的真实数据和约束来验证代码；在实验设计中，后续计划使用工业级有状态任务集来评估质量。

**📈 对比分析**

方法为对比性分析，评估五个框架在 spec 冻结、TDD、测试不可变、数据库循环、Orchestration、门控、角色分离与上下文管理等维度的表现。结果显示 Consort 在数据库循环、不可变测试和人类门控方面独树一帜；但尚未给出具体性能指标，成本与质量比预期通过后续实验验证。

**⚠️ 局限性**

局限性包括：①需依赖数据库环境，部署成本较高；②缺乏对多种模型和运行时的跨平台支持；③实验验证仍待完成，输出质量提升的假设尚未证实；④对轻量级原型或低风险项目可能不具备成本效益。

---

## 20. The Answer Path and the Grounding Instruction in LLM Question Answering over Knowledge Graphs

**arXiv ID:** 2609.10237 | [PDF](https://arxiv.org/pdf/2609.10237v1)

**作者:** Arquimedes Canedo `[一作]` `[通讯]` (Siemens Digital Industries Software), Arquimedes Canedo (Siemens Digital Industries Software)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估 GraphRAG 管线中四个子图相关决策（答案链、序列化语法、三元组顺序、子图大小）对大型语言模型问答性能的影响。

**💡 创新点**

创新点在于系统性证明：仅答案链的存在和 grounding instruction 的抑制效应才是提升性能的关键；其他四项决策在多跳推理下均无显著效果。

**🔧 技术方法**

使用六大 LLM（Claude Sonnet 4.6、Haiku 4.5、GPT‑5、GPT‑5 Mini、Gemini 2.5 Pro、Gemini 2.5 Flash）与七种 RDF 语法、五种三元组排序，统一的 token‑level F1 与证据可信度评估，并通过 95% 置信区间和符号翻转检验来比较实验结果。

**📊 数据集**

采用 LC‑QUAD 2.0（82 个 1/2 跳问题）和 QALD‑10（43 个 3/4 跳问题）构成的 125 个 Wikidata 问题集合，并用 gold SPARQL 构造“oracle”子图。

**📈 对比分析**

通过 16 场实验（共 30,841 次试验）对比上下文、提示、子图大小等因素，发现答案链缺失将 F1 降至 0.005，而保持链并替换其他三元组仅提升 +0.003；表明召回比精确度更关键。

**⚠️ 局限性**

限制包括深度与基准混杂、子图覆盖率随深度下降、缺乏真实检索器、评分器对格式敏感、参数知识污染、样本量在高跳数不足、以及未验证多格式解析器等。

---

## 21. SAGE: Semantic-Aware Geographic Error Recovery for AI Data Movement

**arXiv ID:** 2609.10126 | [PDF](https://arxiv.org/pdf/2609.10126v1)

**作者:** Patrick S. Y. Hung `[一作]` (City University of Hong Kong), Ray C. C. Cheung `[通讯]` (City University of Hong Kong)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出一种语义感知地理错误恢复架构SAGE，针对AI张量在网络芯片中的传输，先判断错误是否需要重播，再决定重播的起点。

**💡 创新点**

创新点在于将错误的数值语义（Class‑H/M/L）与物理故障地理位置分离；使用工作负载校准的安全合同来决定是否重播；利用源本地检查点表根据错误热点动态缩短恢复段；通过质量归一化终端延迟Ψ_del对策略进行排序；实现端点验证和全flit重播以保持语义质量。

**🔧 技术方法**

采用gem5 Garnet模拟器、ASAP7 RTL流水线检查器、BCH/CRC校验、Ordered VC所有权与信用计数、Region‑local 检查点策略、学习曲线A(p_eff)质量映射、延迟/队列统计、HBM pilot等技术。

**📊 数据集**

实验使用DenseNet‑121/ESC‑50（训练故障注入）、DeiT‑S/16推理通信轨迹、以及100×100网格的合成随机流量（带热区）。

**📈 对比分析**

与固定34跳、10跳、路径平均、end‑to‑end等基线对比。SAGE在基准BER 3×10⁻⁵下平均延迟下降28%，p99.5下降36%，质量提升，Ψ_del下降30.1%；在更高BER下保持队列稳定而固定34跳变为非稳态；DeiT‑S应用测试亦显著降低延迟并提升质量。

**⚠️ 局限性**

局限在于类边界和质量权重需针对工作负载/格式手工校准；假设位错误独立，未考虑突发相关错误；未评估端到端任务准确性；硬件实现成本与跨级控制器整合待进一步研究。

---

## 22. Forward-Free LLM Depth Pruning via Weight Redundancy

**arXiv ID:** 2609.09883 | [PDF](https://arxiv.org/pdf/2609.09883v1)

**作者:** Vincent-Daniel Yun `[一作]` (University of Southern California), Woosang Lim `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种前向无关的深度剪枝方法WRP，通过仅使用检查点权重估层间冗余，从而实现不需要前向传播或校准数据的Transformer块裁剪。

**💡 创新点**

创新点在于利用权重空间的线性CKA相似度结合投影尺度信息构建全层相似度矩阵，进而进行全局层聚类与分配，解决了传统基于权重幅值的单层评估缺陷。

**🔧 技术方法**

技术包括权重中心化、线性CKA相似度计算、投影尺度归一化、全层相似度矩阵构造、谱聚类选择聚类数、两阶段冗余裁剪策略。

**📊 数据集**

在LLaMA‑3.1‑8B、Qwen‑3‑14B和Mistral‑Nemo‑12B三大模型上，针对六种剪枝预算，使用LM Evaluation Harness的九个零样本基准（ARC‑Easy/Challenge、HellaSwag、WinoGrande、BoolQ、OpenbookQA、RTE、COPA、RACE）。

**📈 对比分析**

与基于激活的LoRP、LLM‑Streamline、ShortGPT以及前向无关的Mag+比较，WRP在所有设置下平均提升约14.38分，且在绝大多数任务上接近或匹配激活方法的表现，同时无需前向计算。

**⚠️ 局限性**

限制主要是仅关注Transformer块的权重冗余，未考虑微调后对任务特定知识的影响；此外，对于极端大模型，权重矩阵存储与相似度计算仍可能占用显著内存。

---

## 23. AgenticGen: Reward-Guided Agentic Video Generation for Advertising

**arXiv ID:** 2609.09187 | [PDF](https://arxiv.org/pdf/2609.09187v1)

**作者:** Xingyuan Bu `[一作]` (ByteDance), Shilei Wen `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出AgenticGen框架，将广告视频生成拆分为策略选择和草稿生成两阶段推理，并通过线上业务反馈学习奖励模型来优化生成过程。

**💡 创新点**

首次将奖励驱动的智能体方法应用于广告视频生成，结合业务指标反馈与人工质量评判两类奖励，并在TikTok平台实现在线A/B测试，显著提升CTR、CVR和广告价值。

**🔧 技术方法**

使用大规模视觉语言模型（Qwen3‑VL‑8B‑Thinking）进行策略推理；构建性能奖励（Bradley‑Terry）和手工规则奖励；采用DPO进行偏好预热，随后用GRPO进行在线策略优化；实现印象平衡投放与多模态特征融合。

**📊 数据集**

SFT数据：约50K+100K条策略与草稿轨迹；性能奖励数据：100万对视频的印象平衡对齐样本；规则奖励数据：10K对视频的三位人工标注；RL数据：各50K对策略选择与草稿生成的偏好样本。

**📈 对比分析**

在TikTok广告系统的A/B测试中，AgenticGen相较SFT提升CTR 2.72%、CVR 2.63%、广告价值（Advv）9.61%；离线评估表明性能奖励模型准确率60.85%、规则奖励模型对人类标注的准确率69.50%，并证明DPO与GRPO显著提升策略与草稿生成的偏好一致性。

**⚠️ 局限性**

主要局限：依赖昂贵的在线投放数据，收集周期长且噪声大；仅在TikTok平台验证，跨平台泛化尚未评估；奖励模型仍无法完全捕捉细粒度的视觉质量与创意差异；模型复杂度高，部署成本和推理延迟需进一步优化。

---

## 24. $Φ$-Bench: Can Large Language Models Engineer the Infrastructure That Powers Them?

**arXiv ID:** 2609.10226 | [PDF](https://arxiv.org/pdf/2609.10226v1)

**作者:** Leilei Ding `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Φ‑Bench基准，用于评估大型语言模型在LLM基础设施工程中的能力，构造了85个覆盖从核函数实现到全系统优化的三类任务；

**💡 创新点**

创新点在于：①将长周期、开放式、系统级评估纳入基准；②基于论文与开源仓库构建三层覆盖分类；③采用Agent循环自动化任务合成与迭代测试生成，显著提升任务规模与真实性；

**🔧 技术方法**

技术手段包括：LLM驱动的文献与代码筛选与标签、Agent循环任务合成、自动化测试与性能测量、规则与代理双重作弊检测，以及多模型迭代优化分析；

**📊 数据集**

数据来源为2300余篇顶级系统会议论文与1850个公开LLM基础设施仓库，经过筛选后形成410个细粒度标签，并用于生成85个基准任务；

**📈 对比分析**

评测方法采用实现与性能双指标，对八个前沿模型进行评估，Claude Opus 5最高得分36.53%，其他模型在10–28%之间；在E2EO任务中模型通过多轮迭代显著提升表现；

**⚠️ 局限性**

局限性包括：对硬件层面理解不足，长周期任务仍较难完成；模型对推理预算与资源调优敏感，需更高计算成本；基准覆盖的领域仍有限，未来需进一步扩展与细化。

---

## 25. ProMeta: Few-shot PROTAC-targeted degradation prediction across E3 ligases

**arXiv ID:** 2609.09891 | [PDF](https://arxiv.org/pdf/2609.09891v1)

**作者:** Yuansheng Liu `[一作]` (Hunan University), Xiao Luo `[通讯]` (Hunan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了ProMeta，一种基于原型的图神经网络框架，利用少量支持样本在新的E3 ligase上预测PROTAC降解活性，实现跨ligase的少样本学习；

**💡 创新点**

创新点在于将PROTAC降解预测重新定义为少样本元学习问题，采用原型网络在元训练时学习可迁移的分子+蛋白序列表征，并在推理时仅构造目标ligase的活/不活原型，无需更新编码器；

**🔧 技术方法**

使用的技术包括：GCN分子图编码器、轻量级蛋白序列嵌入、ECFP4指纹融合、原型分类器以及基于episodic meta-learning的训练策略；

**📊 数据集**

数据集来自PROTAC‑DB 3.0，经过严格过滤得到1,386个CRBN/VHL标注上下文以及少数罕见ligase（FEM1B、IAP、MDM2、XIAP、cIAP1）的样本，配合相应的POI和E3蛋白序列；

**📈 对比分析**

与RF+ECFP、监督GNN、PROTAC‑STAN、DegradeMaster等基线比较，CRBN→VHL下的AUROC分别为0.796（K=2,Q=3）和0.883（K=2,Q=5），VHL→CRBN分别为0.702和0.821；在一Shot罕见ligase任务中AUROC达0.700，均优于基线；

**⚠️ 局限性**

局限性包括：仅使用二维图/指纹和蛋白序列特征，缺乏3D结构和三元复合物信息；训练数据主要来自CRBN和VHL，罕见ligase评估样本不足；在更大ligase空间或多E3共用的情况下效果未知。

---

## 26. Scalable Composition of Byzantine Agreements under Reorder Attacks

**arXiv ID:** 2609.09623 | [PDF](https://arxiv.org/pdf/2609.09623v1)

**作者:** Jing Chen `[一作]` (Tsinghua University), Wentao Zhou `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种同时考虑节点腐败和通信通道重排攻击的新对手模型，并在此模型下研究拜占庭协议的并行与并发组合安全性，给出了最优的安全阈值，随后构造了两套黑盒编译器（并行和并发）以及面向长消息的高效可靠广播与可靠传输原语，实现了在异步网络中对长消息的低通信复杂度。

**💡 创新点**

创新点主要体现在：①首次将通信通道重排攻击纳入拜占庭协议安全分析，揭示了传统签名方案在并行/并发环境下失效的根本原因；②在该模型下给出了既包含可认证又不包含认证的协议的紧凑安全阈值；③设计了可复用的黑盒编译器，能够在不增加额外安全假设的前提下，将任何单实例协议转换为并行/并发安全的协议；④利用纠删码和顺序证明方案，将长消息的可靠广播与传输的通信复杂度从原始的多倍提升到接近最优。

**🔧 技术方法**

使用的技术包括：可靠广播（RB）与可靠传输（RMT）的新实现，基于多数投票的重排抵抗机制；纠删码（ECC）与Merkle树证明（顺序证明方案）相结合实现长消息的高效重传；对称式黑盒编译框架；以及对并行与并发执行模型的形式化分析。

**📊 数据集**

本工作为理论分析论文，无使用具体数据集；所有结果均以定理、定律和通信复杂度分析为依据。

**📈 对比分析**

与原始协议相比，经过并行编译后的协议通信复杂度增加约 O(n) 倍（对点对点消息），但在多实例执行中实现了真正的并行/并发安全；对长消息的改进版 RMT/RB 通过纠删码将通信复杂度从 O(n^3|M|) 降低到 O(|M| + λ n log n)，在异步网络中保持相同的安全阈值，并且在大多数实际场景下可实现与非组合协议相当的效率。

**⚠️ 局限性**

限制包括：①对短消息仍需要 O(n^3|M|) 的通信开销；②改进版原语需要可测度的密码学假设（如哈希碰撞抵抗）和多项式时间对手；③阈值 n>max{3t, 2c+2t+1} 仍相对严格，可能在某些实际部署中难以满足；④当前工作仅针对传统有效性定义，未探讨弱有效性或其他可接受的拜占庭协议形式。

---

## 27. Geometry Without Coordinates: LiDAR Diffusion as a 3D Feature Bridge

**arXiv ID:** 2609.10322 | [PDF](https://arxiv.org/pdf/2609.10322v1)

**作者:** Samed Doğan `[一作]` (Munich University of Applied Sciences), Alfred Schöttl `[通讯]` (Munich University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过将二维预训练模型的伪标签迁移到稀疏 3D LiDAR 上，训练了一个 LiDAR 条件扩散模型，能够根据文本提示生成深度、语义分割和实例预测等多模态输出，并在点云空间对其中间特征进行系统性分析。

**💡 创新点**

创新点包括：①利用 LiDAR 条件扩散模型把大规模 2D 先验迁移到 3D 稀疏域；②设计了在点云空间进行线性探测与余弦相似度评估的协议；③揭示了网络深度上特征的层次化跨模态组织——浅层保持模态分离，瓶颈层趋同，解码器层再分化。

**🔧 技术方法**

采用的技术主要有：基于 Stable Diffusion 1.5 的 U‑Net 扩散网络；文本提示条件；Depth Anything、Segment Anything、SegFormer 等 2D 预训练模型产生的伪标签；线性探测、余弦相似度分析以及条件丢弃（深度/强度/像素级）实验。

**📊 数据集**

实验数据集为 nuScenes（包含同步 LiDAR 与相机），伪标签通过 Cityscapes 训练的 2D 模型生成，评估使用 nuScenes lidarseg 验证集。

**📈 对比分析**

评估方法为：在点云空间对 U‑Net 各层特征进行线性探测，基准为高斯噪声；语义类 MIoU 最高可达约 0.23（对比 0.035 的噪声基线）；深度指标 abs_rel=0.160、δ1=0.781；不同丢弃方式对性能影响有限，像素级丢弃略微降低结果。

**⚠️ 局限性**

局限性包括：仅在单一数据集上训练和评估，未验证跨数据集泛化；语义评估受 Cityscapes 与 nuScenes 标注不一致的影响；实例预测仅做定性展示；输出性能不及专门 3D 任务模型；未探究随机初始化或训练稳定性的敏感性。

---

## 28. KVShareArena: KV-Cache Reuse Across Contexts and Model Checkpoints

**arXiv ID:** 2609.10266 | [PDF](https://arxiv.org/pdf/2609.10266v1)

**作者:** Xi Shi `[一作]` (University of Central Florida), Qian Lou `[通讯]` (University of Central Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个针对KV缓存跨上下文与跨检查点重用的基准KVShareArena，并在多源检索与多智能体报告两类工作负载上评测多种重用方法。

**💡 创新点**

创新点在于统一量化跨上下文与跨检查点重用的性能与成本，并提出PGR指标和三维成本前沿，公开了基准代码与排行榜。

**🔧 技术方法**

采用位置对齐、选择性重算、注意力校准、锚点偏移、细调修复、软标记适配器、解码KV交接、压缩等多种修复技术。

**📊 数据集**

使用LongBench v1（科学论文、单文档、多跳QA）和FRAMES四个子集的检索证据，Agent Reports的专家报告数据集。

**📈 对比分析**

通过与无缓存基准、全重算上限比较，测量PGR并绘制计算、内存与TTFT三维前沿；结果显示仅位置对齐即可满足大多数任务，只有多源互相依赖时付费修复才显著提升。

**⚠️ 局限性**

局限在于仅评测8B/4B模型，跨检查点重用只验证相同架构、分词器；报告轨道方法有限；基准规模受N=100限制，未覆盖更大模型或不同任务。

---

## 29. An Autonomous GeoAI Agent for Arctic Eco-Navigation

**arXiv ID:** 2609.09374 | [PDF](https://arxiv.org/pdf/2609.09374v1)

**作者:** Samira Alkaee Taleghan `[一作]` (University of Colorado Denver), Farnoush Banaei-Kashani `[通讯]` (University of Colorado Denver)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了一个人机交互、多代理 GeoAI 系统，用于在北极进行生态导航，结合运营、物理、生态和社区多准则的航线规划。

**💡 创新点**

创新点包括：①将生态与社区影响纳入多准则航线优化并保持人类价值判断可控；②采用分层代理架构（数据采集、搜索、决策）实现可审计、可解释的决策；③引入 ε‑近似 Pareto 搜索与多种搜索算法，提供可调整的计算-精度折中。

**🔧 技术方法**

使用的技术包括：图论与多目标最短路（多目标 Dijkstra、A*、A*pex）、基于欧拉法的近似 Pareto 集、空间数据预处理与投影、云端大数据接口（Ollama 语言模型）以及多代理通信与验证。

**📊 数据集**

主要数据集：USNIC 历史海冰图、NOAA GFS 预报海冰、NOAA NCEI 声纳深度、NOAA Fisheries EFH 与海豹关键栖息地、Open‑Meteo 波浪风速、Alaska DCRA/BOEM 社区与传统狩猎信息。

**📈 对比分析**

对比方法：在 Bering Strait 3km 网格、250km 缓冲区内，比较精确 Pareto、ε‑近似 Pareto 与单目标加权求解；实验显示 ε=0.35 在 371s 内完成 229 条代表路线，覆盖所有可行解；更高 ε 加速但牺牲多样性；加入单目标锚点可恢复极端解，保持决策质量。

**⚠️ 局限性**

局限性：1）不支持时变多准则航线规划；2）缺失或零值证据仍可能导致不完整的准则评估；3）空间去重阈值无正式保证，可能影响结果；4）系统仍需人工指定权重、权益等主观参数。

---

## 30. Playing Whack-a-Mole with misconceptions about memorization, extraction, and copyright

**arXiv ID:** 2609.09320 | [PDF](https://arxiv.org/pdf/2609.09320v1)

**作者:** A. Feder Cooper `[一作]` `[通讯]`, A. Feder Cooper

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该论文通过对前沿大型语言模型（GPT‑4o、Gemini‑2.5‑Pro、DeepSeek‑V3.1）进行微调，并利用“书籍记忆覆盖率（bmc@k）”等指标评估模型在已微调后的模型上对版权书籍的逐字复现能力，声称微调能激活模型对大部分书籍内容的记忆并实现大量提取。

**💡 创新点**

创新点主要体现在：①提出并使用bmc@5作为衡量模型记忆覆盖率的指标；②通过大量基于摘要的prompt与多次生成相结合的实验框架，声称可在微调后高比例提取被微调书籍的内容；③将单个模型在不同书籍上的覆盖率与最长连续提取片段进行对比，尝试评估“对版权侵权的实质性影响”。

**🔧 技术方法**

使用的技术包括：
- 前沿模型微调（OpenAI/Gemini/DeepSeek的Fine‑Tuning API）
- 生成式摘要（用GPT‑4o生成约一半长度的情节摘要）
- 生成多次（每段摘要100次）以提高覆盖率
- 通过正则化和trim步骤统计与原书的5+连续词匹配，得到bmc@5
- 对比最长连续提取片段（longest contiguous regurgitated span）。

**📊 数据集**

使用的数据集为公开版权书籍集合（Books3），共计243个书籍–模型组合；每本书被拆分为约300–500词的“段落”，并生成摘要；测试集中包含在微调数据之外的书籍。

**📈 对比分析**

实验方法：对每本书生成约30,000条输出（100次×约300段），统计所有输出中与书籍的5词以上连续匹配并去除prompt中出现的匹配，计算覆盖比例。报告的性能主要是bmc@5覆盖率（最高可达85–90%）和最长单生成连续提取片段（最高约460词）。

**⚠️ 局限性**

局限性与批评：
- bmc@5使用5词阈值，低于行业常规的37词，易产生假阳性；
- 评估过程中未充分控制prompt泄露，导致部分匹配可能源自摘要重组；
- 未进行负控制实验，无法量化假阳性比例；
- 未报告实验成本，影响对市场替代性的评估；
- 论文声称的“市场替代”与实际提取方式不符，覆盖率为散布式碎片，难以直接替代完整作品；
- 研究缺乏对提取结果真实性与法律意义的深入讨论。

---

## 31. Joint nonlinearity in a stiffened aluminium wingbox panel and what it requires of a reduced basis

**arXiv ID:** 2609.10088 | [PDF](https://arxiv.org/pdf/2609.10088v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` (Institute of Communication and Computer Systems), Evangelos Papatheou `[通讯]` (University of Exeter)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并校准了一个以物理螺栓位置为节点的铝制翼盒钢板的壳有限元模型，研究了接头非线性对振动响应以及模型降阶方法的影响。

**💡 创新点**

提出通过在接头处动态残差压缩而非扩展基底的降阶技术，解决局部非线性无法被全局模态捕捉的问题，并证明接头非线性对结构响应具有明显非对称的软化和摩擦效应。

**🔧 技术方法**

使用混合插值壳单元、灵敏度校准、Newmark时域与谐波平衡频域双独立求解、Taylor展开投影降阶以及邻近元素块级压缩的精确Newton求解。

**📊 数据集**

采用Sheffield大学制造的铝制翼盒板实验数据，包含自由–自由悬挂激励-测量装置得到的前五个模态频率及加速度传感器数据。

**📈 对比分析**

通过与完整模型的时域和频域响应、线性/三阶Taylor降阶模型以及邻近块压缩降阶模型进行对比，精度可达0.01–0.13个百分点，计算成本在全阶模型的1/8至2倍之间。

**⚠️ 局限性**

Taylor展开仅在接头接近斜率拐点时有效，无法完整描述饱和、双线性或摩擦等非线性形式；邻近块压缩需要手动选择邻域大小以平衡精度与速度，且对接头滑移范围的限制导致在强非线性操作点需更大邻域。

---

## 32. Voice or Stereotype? Disentangling Acoustic and Content-Based Gender in Speech-to-Speech Models

**arXiv ID:** 2609.09263 | [PDF](https://arxiv.org/pdf/2609.09263v1)

**作者:** Xiaoqun Liu `[一作]` (Centific Research), Abhishek Mukherji `[通讯]` (Centific Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实施了一套双重检验协议，评估语音转语音模型在不同语言、不同内容性别刻板印象下的声音漂移与性别归属偏差。

**💡 创新点**

提出了针对固定输出声音的S2S模型进行性别归属偏差评估的完整实验框架，并首次揭示固定声音下的偏差隐藏现象。

**🔧 技术方法**

使用合成语音（TTS）产生受控输入，五大S2S模型（GPT‑4o‑audio、Gemini 2.5、GLM‑4‑Voice‑9B、Step‑Audio‑2‑mini、Kimi‑Audio‑7B），并通过声学特征、wav2vec‑2.0 说话人性别分类器、逻辑回归等技术评估结果。

**📊 数据集**

基于英文、中文、西班牙语三种语言的长篇性别中性文本，先由LLM生成并人工筛选，后由TTS生成对应男女声音，形成的180条受控音频样本。

**📈 对比分析**

通过五项任务（朗读、改述、摘要、翻译、性别词语描述）对模型进行测评，发现所有模型的声音漂移几乎为零，但性别归属在内容偏移时呈现显著的性别歧视（误判率高达90%），闭源模型偏差更大。

**⚠️ 局限性**

局限包括：每语言样本量小、LLM生成文本可能与测试模型共享先验、仅考察二元性别、固定音色模型固有限制等。

---

## 33. An Exponential Deterministic--Randomized Gap in ERM-Oracle Complexity for Thresholds on an Unknown Order

**arXiv ID:** 2609.10196 | [PDF](https://arxiv.org/pdf/2609.10196v1)

**作者:** Xuan Li `[一作]` `[通讯]` (University of New South Wales), Xuan Li (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在线学习中使用一致性型 ERM（全概念返回）或弱一致性（仅返回可实现性）接口时，阈值概念在未知总序上的在线学习问题，重点分析随机化与确定性学习器在调用量和错误数上的差距。

**💡 创新点**

1) 在固定自然的最小前缀/最大前缀规则下，证明任何确定性学习器必须在某些实例上总调用数+错误数≥T-ε；2) 证明随机化学习器在同一规则下可实现 O(log T) 次错误与 O(log T) 次调用，从而给出确定性与随机化的指数级分离；3) 将此分离扩展到对抗性“冻结”回调，并探讨选择规则的复杂性轴；4) 对固定查询预算和弱一致性接口给出部分极限和对比结果。

**🔧 技术方法**

主要技术包括：\n- 一点公共值 (one‑pin) 证明法，用于构造在任何查询下最多只能“冻结”一个自由点的极端规则；\n- 电荷/固定化（fixation）步骤，将适应性构造转化为固定实例；\n- 潜在函数 (potential) 及区间树分布（interval‑tree distribution）用于随机化下的期望下界；\n- 对抗性冻结与记忆无关的欧拉化方法。

**📊 数据集**

本工作完全基于理论分析，没有使用任何真实或合成数据集；所有结果均通过数学证明给出。

**📈 对比分析**

通过对比理论下界与已知的随机化上界（AHR25 证明的 O(log T) 调用）以及对抗性下界，本文展示：确定性学习器需要线性调用 O(T)，随机化学习器只需对数调用 O(log T)，误差数两者相同；在弱一致性接口下，二者均需 Θ(T) 调用来达到 O(log T) 误差。

**⚠️ 局限性**

限制：\n- 只考虑阈值类及其未知总序的转导式在线学习；\n- 对象的选择规则仅在特定“极端”规则或可预声明规则下得到完整定理，无法直接推广到所有合法规则；\n- 常数（如 128）未最优化；\n- 对非转导式设置、其他 Littlestone 类的扩展仍未完成；\n- 中间查询预算区间的精确值尚未确定。

---

## 34. Talking to Itself While Coding: What Makes Comments Help Code Generation?

**arXiv ID:** 2609.09242 | [PDF](https://arxiv.org/pdf/2609.09242v1)

**作者:** Dangfeng Pan `[一作]` (Monash University), Xiaoning Du `[通讯]` (Monash University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型在同一次推理中自生成注释是否会因果影响代码生成的 pass@1 性能，并通过观测和受控干预实验加以验证。

**💡 创新点**

创新点在于将强模型的注释块预填给弱模型，剥离表面形式，证明注释内容的正确性是提升性能的关键因子，且自提示无法可靠复制该提升。

**🔧 技术方法**

使用了观测分析、受控干预实验、注释块对齐、内容扰动实验、不同提示变体以及统计显著性检验等技术手段。

**📊 数据集**

实验基于 LiveCodeBench v6（单文件 Python）、RepoClassBench（C# 与 Java）以及自制的十种提示变体。

**📈 对比分析**

通过与基线、同源、随机文本、错题目等对比，平均在 12 个弱–强模型配对中提升约 17.2%，而自提示仅恢复约 24% 的提升，显示外部注释对性能有显著正面影响。

**⚠️ 局限性**

局限在于未能在自提示或训练时方法中重现外部注释的显著提升，需要进一步探索更有效的提示或训练策略。

---

## 35. Exact Degeneracy Under Balanced k-Shot Sampling:Consequences for Small-Sample Discriminant Analysis on LLM Embeddings

**arXiv ID:** 2609.09860 | [PDF](https://arxiv.org/pdf/2609.09860v1)

**作者:** Lingxiao Qu `[一作]` `[通讯]` (University of Aizu), Lingxiao Qu (University of Aizu)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在冻结LLM嵌入下，使用平衡k-shot采样进行少样本文本分类时，Kernelized Linear Principal Component Discriminant Analysis（KLPCDA）七种变体的行为，并提出了闭式诊断与修复方法；同时在四个公开数据集、三种嵌入维度和多种k值上与线性探针、最近中心、SetFit、LoRA、ICL等基线进行比较，给出了四条可量化的实践指南。

**💡 创新点**

①在平衡k-shot采样下，KLPCDA的类内散射矩阵成为缩放正交投影，导致三种变体在理论上出现特征子空间全等或目标为零的极限；②提出在该退化子空间内使用内部次级项进行“tie‑break”修复；③首次在冻结LLM嵌入上系统评估KLPCDA并给出四条针对类数、层次、ICL和高维嵌入的实证指南；④通过三种几何分离度量的负面结果，证明高维嵌入的劣势主要是估计效率问题，而非几何可分性。

**🔧 技术方法**

使用的技术包括：Kernelized Linear PCA/Discriminant Analysis、线性逻辑回归探针、最近中心分类、SetFit（对比学习微调）、LoRA（低秩适配器微调）、ICL（基于提示的生成分类）、特征子空间投影、特征值分解与tie‑break、交叉验证、Wilcoxon配对检验、浮点精度验证等。

**📊 数据集**

使用的数据集：Banking77（77类）、CLINC150（150类）、TREC（6类）和AG News（4类）。嵌入来源：MiniLM（d=384）、BGE-large（d=1024）和E5‑Mistral‑7B‑Instruct（d=4096）。

**📈 对比分析**

与基线的比较方法：对每个数据集和k∈{2,3,5,10}（以及扩展k≤50）抽取10次平衡支持集，计算各方法在完整测试集上的准确率，采用配对Wilcoxon检验评估差异。结果显示，调优后的逻辑回归探针在三大数据集上普遍优于KLPCDA七种变体；在TREC上KLPCDA No.6在某些k值下略优；KLPCDA修复后的No.1/5表现提升但仍低于探针；SetFit在大多数情况下领先于LoRA；ICL在k=2–3时优于微调基线，但在k≥10时被SetFit/LoRA追赶。

**⚠️ 局限性**

局限性包括：①未修复KLPCDA No.3和No.7；②仅使用线性核，未测试RBF等核；③ICL仅在类数较少的数据集上评估；④随机种子数仅为10，样本量有限；⑤仅评估三种嵌入尺寸，未单独检验d维度对结果的纯影响；⑥结果特定于平衡k-shot采样，其他采样策略未探究；⑦对几何分离度量的负面结论仅适用于当前测度，未排除其他可能解释。

---

## 36. Robust Industrial Cyber Physical Classification Using Neuromorphic Temporal Embeddings and Hybrid SNN XGBoost Under Machine Unlearning Attacks

**arXiv ID:** 2609.09564 | [PDF](https://arxiv.org/pdf/2609.09564v1)

**作者:** Ammar Kamoona `[一作]` (RMIT University), Xinghuo Yu `[通讯]` (RMIT University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种混合 Spiking Neural Network + XGBoost 的入侵检测系统，用一次性训练的 SNN 提取时间特征并冻结，随后仅周期性更新 XGBoost 分类器，从而实现对机器不学习（label poisoning）攻击的鲁棒性。

**💡 创新点**

创新点包括：①通过分离特征提取与分类来隔离标签污染，SNN 的时间膜电位嵌入仅受物理信号影响；②使用 Leaky‑Integrate‑and‑Fire SNN 与 surrogate gradient 学习实现高效、低能耗的时间特征提取；③证明该结构在机器不学习攻击下可延迟性能崩溃并显著降低 F1 下降幅度；④可在 Intel Loihi 等神经形态硬件上实现实时、低功耗部署。

**🔧 技术方法**

技术细节：Leaky‑Integrate‑and‑Fire (LIF) 神经元、surrogate gradient 反向传播、15 步滑动窗口时间嵌入、最大/平均/最终膜电位三种统计特征、min‑max 归一化、互信息特征选择、XGBoost 梯度提升树、Python/PyTorch/Loihi SDK 等。

**📊 数据集**

使用两个真实功率系统数据集：Synchrophasor（9 维特征，218,459 条样本，3 类）和 MSU/ORNL（128 维降到 50 维，78,377 条样本，3 类）。

**📈 对比分析**

与 MLP、LSTM、1D‑CNN、GNN、Random Forest、XGBoost、规则检测等 7 种基线进行比较。Hybrid 在干净数据上分别达到 99.9%（Synchrophasor）和 95.0%（MSU/ORNL）的准确率，F1‑macro 0.999/0.943；在 10% label poisoning 下 F1‑macro 仅下降 0.9%，并将目标类崩溃阈值从 60% 提升至 70%，表现优于所有原始模型。

**⚠️ 局限性**

局限性：①假设初始训练数据已得到安全验证，SNN 冻结后若被攻击仍有风险；②在极端投毒（≥70%）时所有模型最终崩溃；③仅在 MSU/ORNL 数据集上进行机器不学习实验，泛化性待验证；④SNN 单独准确率较低，主要依赖嵌入结构；⑤未评估针对嵌入空间的自适应对抗攻击。

---

## 37. Time-Frequency Geometric Cross-Attention for Chunked Vision-Language-Action Models

**arXiv ID:** 2609.09925 | [PDF](https://arxiv.org/pdf/2609.09925v1)

**作者:** Shengye Dong `[一作]`, Shanmin Pang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Time–Frequency Geometric Cross‑Attention (TFGCA) 模块，用于改进 Vision‑Language‑Action（VLA）模型的动作片段预测，提升多阶段任务的执行质量。

**💡 创新点**

创新点在于：① 对每个控制维度采用可学习的 Stationary Wavelet Transform 将动作序列拆解为多尺度频率分量；② 在交叉注意力中融合点积（相似度）与楔积（正交度）两种分数，以同时捕获频率特征和跨阶段几何关系；③ 采用零初始化残差保证可直接插拔到已有预训练模型。

**🔧 技术方法**

技术栈包括：可学习 SWT、楔积注意力机制、动作空间投影与对齐损失、流匹配训练、零初始化残差结构、学习的混合权重 β。

**📊 数据集**

使用的数据集有 LIBERO、LIBERO‑Plus、RoboTwin 2.0 以及真实机器人 AgiBot A2 的实验数据。

**📈 对比分析**

对比方法：在相同训练与评估协议下与基线 π₀.₅ 进行同源比较；在 LIBERO 上平均提升 1.5%，在 LIBERO‑Plus 总体提升 6.3%，在 RoboTwin 随机化环境提升 28.5%，在 AgiBot A2 上整体成功率提升 11.67% 点；相对于其他公开方法亦表现领先。

**⚠️ 局限性**

局限性：频域分解在极短时间窗口或单层小波时效果有限；模块对预训练模型的兼容性要求高；在极端 OOD 或高随机化环境下仍有进一步提升空间。

---

## 38. OASIS: A Rubric-Based Multimodal Assessment Platform Using Large Language Models

**arXiv ID:** 2609.09180 | [PDF](https://arxiv.org/pdf/2609.09180v1)

**作者:** Ameer H. Shakur `[一作]`, Andrew R. Jamieson `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了OASIS系统，利用大语言模型对视频、音频和文本的评估任务进行规范化、可追溯的 rubric‑based grading，支持从本地 CLI 到完整的 Elephant+MAPLES 栈，甚至 AI 助手与 M2C 接口的自动化操作。

**💡 创新点**

核心创新在于把 rubric 视为可编译程序，提供分阶段、可见成本的执行计划、内容可寻址的评分身份、统一的人工评审状态以及完整的可审计链，实现在大规模部署中保持透明度、可重现性和可审计性。

**🔧 技术方法**

技术实现主要包括 Go + Python 的 OpenAPI 先行合同、FastAPI+Prefect 工作流、PostgreSQL+MinIO 数据存储、LLM 调用（Gemini、OpenAI、Anthropic、Ollama、vLLM）、MCP 代理、SimRubrics 的多模型 rubric 质量检查以及 Wayfinder 交互式助手。

**📊 数据集**

系统使用了医学教育中的 OSCE 评估数据（约 7000 场实测 encounter，包含视频、音频、文本和注释），以及内部合成的多模态验证集作为开发与测试数据。

**📈 对比分析**

通过与人工评分的对比，item‑level agreement 在 93–96% 范围，Cohen κ 在 0.83（AI–人类）与 0.73（人类–人类）之间；系统显著降低人工评分工作量至 95–97%，并通过分阶段成本估算与分布式批处理提升了效率。

**⚠️ 局限性**

主要局限包括对特定模型（如 Gemini 原生视频）的依赖、对评估设计质量的高度敏感、以及对托管 LLM 版本变更的不完整可重现性；多模态支持仍受限于模型、服务运行时和硬件条件。

---

## 39. Decentralized network congestion control for DAG-based distributed ledger system

**arXiv ID:** 2609.09961 | [PDF](https://arxiv.org/pdf/2609.09961v1)

**作者:** Mayank Pandey `[一作]` (IIT Kanpur), Nishchal Kumar Verma `[通讯]` (IIT Kanpur)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于节点行为的可变PoW难度模型，利用交易频率和代币支出作为声誉来动态调整DAG分布式账本的交易上限，从而实现去中心化网络拥塞控制。

**💡 创新点**

核心创新在于：①将PoW难度与节点历史交易频率和代币支出关联，实现公平的交易机会；②通过非合作博弈证明存在唯一的纯纳什均衡，使每节点在规定限额内发交易；③对动态节点数变化给出均匀的均衡漂移。

**🔧 技术方法**

使用的技术主要包括：可变PoW难度计算、基于声誉的难度公式、非合作博弈理论、纳什均衡分析、数值模拟。

**📊 数据集**

未使用真实数据集，采用数值仿真和示例网络（如10节点、2节点、IOTA模型等）进行验证。

**📈 对比分析**

与IOTA等现有线性/固定难度PoW进行对比，证明所提模型在满足最大吞吐量的同时，能够更公平地分配计算资源，抑制交易刷屏，且对节点数变化具有鲁棒性；性能以理论分析和数值结果展示，未给出具体吞吐量数值。

**⚠️ 局限性**

主要局限包括：1) 仅在模拟环境中验证，缺乏真实网络实验；2) 需要全局时间同步与节点声誉信息的共享；3) 对网络延迟、攻击手段（如协同刷屏）未作深入研究；4) 可能对高频交易者产生额外计算成本。

---

## 40. Constraint-Aware Discrete Black-Box Optimization Using Tensor Decomposition

**arXiv ID:** 2609.09370 | [PDF](https://arxiv.org/pdf/2609.09370v1)

**作者:** Keisuke Onoue `[一作]` (Nara Institute of Science and Technology), Ryosuke Kojima `[通讯]` (Kyoto University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在离散黑盒优化中将逻辑约束直接嵌入张量分解模型的约束感知张量分解（CA‑TD）框架，显著提升了样本效率；

**💡 创新点**

创新点在于将先验符号约束通过T‑norm可微逻辑惩罚方式融入张量分解的学习过程，使得代理模型在训练阶段即具备可行域辨别能力；

**🔧 技术方法**

采用张量分解（CP/TT/TR）构建代理模型，利用T‑norm差分逻辑实现约束惩罚，并提供两种训练方案：半定规划（HSDP）与梯度下降（PGRAD）；

**📊 数据集**

在合成Ackley、真实工程的压力容器设计、Warcraft路径规划、糖尿病治疗方案以及七个从文献迁移来的复杂组合优化任务上进行实验；

**📈 对比分析**

与传统无约束/约束贝叶斯优化、TPE、PROTES以及基于NN+MILP的强基线进行对比，结果表明CA‑TD在大多数任务上实现了更快的收敛和更优的最终目标值，尤其在小规模问题上HSDP更稳定，在大规模问题上PGRAD更高效；

**⚠️ 局限性**

主要局限在高维搜索空间下的内存与计算量扩展性，现有实现仍依赖稠密张量，缺乏自动化秩选择与稀疏表示方法。

---

## 41. UOT-Gap: A Variational Principle for the Modality Gap in Vision-Language Models via Unbalanced Optimal Transport

**arXiv ID:** 2609.10224 | [PDF](https://arxiv.org/pdf/2609.10224v1)

**作者:** Zonglin Yang `[一作]` (Guangdong Police College), Yuejun Xie `[通讯]` (Guangdong Police College)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于不平衡熵正则化最优运输（UOT）的无训练诊断框架，用来量化视觉‑文本模型（如CLIP）中的模态差距，并评估图像‑字幕配对质量；

**💡 创新点**

将模态差距视为UOT残差的分解，区分分布匹配成本、耦合复杂度和边缘质量；引入pair‑aware残差能捕捉配对破损；证明残差与经典均值差距之间的关系并给出一次性下降方向；实现仅靠冻结编码器即可完成诊断；

**🔧 技术方法**

不平衡熵正则化最优运输（UOT）与Sinkhorn迭代、球面投影更新（barycentric correction）、配对残差统计、Spearman相关分析；

**📊 数据集**

Flickr8K、COCO Karpathy 1K子集、5个随机COCO 1K子集以及合成正态球面样本；

**📈 对比分析**

与传统均值差距、模态探测AUC、logit熵等指标对比，pair‑aware UOT残差在多种模型/数据/超参组合下对Recall@1降幅具有平均Spearman 0.97（最高1.00），远优于均值差距；在随机字幕实验中能够精准捕捉检索失效；

**⚠️ 局限性**

仅评估冻结的全局嵌入和1K子集，缺乏大规模检索验证；对UOT超参数敏感；需要配对数据；未与其他分布度量（如MMD、Sinkhorn散度）比较；对无配对或动态场景的适用性有限。

---

## 42. Do LLMs Make More Mistakes If They Do Not Believe the Input Data?

**arXiv ID:** 2609.09363 | [PDF](https://arxiv.org/pdf/2609.09363v1)

**作者:** Peter Kochelka `[一作]` (Charles University), Ondřej Dušek `[通讯]` (Charles University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大型语言模型在对Czech/Slovak本地知识的RDF三元组进行事实、虚构和反事实生成时的真实性和流畅性，重点分析了模型对上下文与自身参数知识冲突的处理。

**💡 创新点**

创新点在于将低资源语言的本地知识与对立（反事实）输入相结合，使用LLM自我分类来衡量模型对输入真实性的感知，并通过多模型、多语言的LLM-judge评估揭示评估者偏差对上下文-记忆冲突结论的影响。

**🔧 技术方法**

技术手段包括使用九个开源LLM（从1.7B到122B级别）、多轮温度为1.0的分类任务、贪婪解码的文本生成、以及基于Kimi K3的自动评判器进行事实性与流畅度评分；同时通过人类标注对评判器进行微调与验证。

**📊 数据集**

数据集为Czech和Slovak的本地知识问答基准CUS-QA转化为RDF三元组，并人工构造对应的反事实与虚构版本，合计2,847条实例（1,482条Czech，1,365条Slovak）。

**📈 对比分析**

比较方法通过在四种目标语言（Czech、Slovak、English、Upper Sorbian）上对同一输入生成文本，使用Kimi K3评判者对可信度（1–5分）和流畅度进行平均评分，结果显示大模型在真实性上更优，但Upper Sorbian仍显低效；对比不同输入真实性，FA>FI>CFA的真实性得分仅差0.05分，显示上下文-记忆冲突效应极小。

**⚠️ 局限性**

主要局限包括评判者和人工标注者缺乏Upper Sorbian母语能力、评判器在小样本上可能过拟合、使用低推理开销的模型生成可能不代表高推理模式下的行为，以及评判者与人类评估在某些错误类别上的偏差未能完全消除。

---

## 43. AMEND: Audited Margins Enable Nonblocking Drops in GPU-PIM LLM Decoding

**arXiv ID:** 2609.09823 | [PDF](https://arxiv.org/pdf/2609.09823v1)

**作者:** Zuxiong Tan `[一作]` (University of California, Davis), Avesta Sasan `[通讯]` (University of California, Davis)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种GPU–PIM协同的稀疏注意力解码方案AMEND，利用已审计的历史margin预测BLASST裁剪决策，并行执行GPU存活注意力与PIM补全QK，完成补充审核后更新状态并提前生成下一步mask。

**💡 创新点**

创新点包括：① 在解码前基于历史margin进行BLASST裁决预测；② 并行GPU与近内存PIM执行补全QK，消除当前查询带宽瓶颈；③ 堆栈级控制器融合观测、状态更新与下一步mask预生成；④ 用Scalable Constrained Bayesian Optimization离线调优五个混合型参数以满足误判预算。

**🔧 技术方法**

使用的技术包括：GPU+HBM‑PIM近内存算子（AttAcc）、BLASST的max‑relative裁剪、EMA与方差预测器、补充审核机制、堆栈级控制流、Scalable Constrained Bayesian Optimization（SCBO）。

**📊 数据集**

使用的数据集为LongBench（16个长文本任务）和RULER（多上下文长度），以及Llama‑3.1‑8B‑Instruct和Qwen3‑8B模型。

**📈 对比分析**

通过与Dense、Quest、BLASST、BLASST‑PIM、Quest‑PIM等基线在模型质量、解码速度、能耗等指标进行对比。实验显示AMEND在保持接近基线的质量下，批量8时可获得1.40–3.63×的解码速度提升，能耗下降28–66%，在8K–64K上下文长度上表现最优。

**⚠️ 局限性**

局限性在于：对BLASST裁决可预测性的依赖，误判率受历史margin准确性的影响；需要额外的PIM硬件与控制器；极大上下文或批量尺寸下的审计延迟可能成为瓶颈；目前评估仅基于仿真，缺乏真实硬件验证。

---

## 44. Conditions for Global Optimality in Quantum Arimoto-Blahut Algorithms

**arXiv ID:** 2609.09731 | [PDF](https://arxiv.org/pdf/2609.09731v1)

**作者:** Geng Liu `[一作]` (Chinese University of Hong Kong), Masahito Hayashi `[通讯]` (Chinese University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究全秩 Arimoto–Blahut 固定点的全局最优性条件，阐明其与镜像下降（MD）更新的关系，并给出后验最优性证明与上界。

**💡 创新点**

提出必要且充分的全局最优性判定条件：固定点处 AB 更新方向与目标梯度仅相差约束正交空间中的元素；同时给出可仅用目标函数评估的有限差分最优性证书。

**🔧 技术方法**

使用凸优化、Bregman 投影、矩阵微分、镜像下降理论、Arimoto–Blahut 迭代以及有限差分数值方法。

**📊 数据集**

以量子通道相对熵为例，分别研究 dephasing–depolarizing 通道（无约束与线性约束）和振幅衰减通道的全秩 AB 固定点；实验基于二维希尔伯特空间，使用随机或特定初始密度矩阵。

**📈 对比分析**

通过 AB 与 MD 在同一初始点下的轨迹对比，证明 AB 在满足兼容性条件时可达到全局最优；使用后验证书计算目标间隙上界，验证其正确性；在振幅衰减示例中展示 AB 迭代单调但停在次优点，说明单调性不足。

**⚠️ 局限性**

仅适用于全秩固定点、线性等式约束的情形，未讨论边界点或非凸目标；大规模问题的可扩展性与计算复杂度未给出；后验证书依赖于可行切向量基的枚举，可能在高维时不可行。

---

## 45. ContractEval: Query-Conditioned Execution Matching for Procedural Instruction Conformance

**arXiv ID:** 2609.09458 | [PDF](https://arxiv.org/pdf/2609.09458v1)

**作者:** Praphul Singh `[一作]` (Oracle Health AI), Ganesh Kumar `[通讯]` (Oracle Health AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了ContractEval框架，用来将LLM代理的程序性合规性转化为对查询激活义务的匹配评估；

**💡 创新点**

创新点在于明确了评估分母——查询激活的义务图，并通过结构化匹配将遗漏、错误分支、依赖冲突等具体化为可定位的合规性失败；

**🔧 技术方法**

核心技术包括基于层次任务网络（HTN）的合同编译、查询条件化期望图构造、文本到图的LLM提取器以及最大权匹配算法；

**📊 数据集**

使用了10份深度式SOP合同共200条查询+1,400条人工审核并注释的程序执行样本，配合人工注释的声明性HTN与期望图；

**📈 对比分析**

与输出仅、轨迹感知、期望图LLM判定等四类评估方法对比，金标ContractEval在检测/定位隐藏结构错误时几乎完美，而基于LLM提取的可扩展评估在检测率与定位率上达≈0.95，差距主要来自提取器的校准问题；

**⚠️ 局限性**

局限在于仅处理文本式SOP合同，缺乏多模态、多工具环境支持，且依赖于人工审核的图注释与提取器的精度，无法直接用于高风险部署场景。

---

## 46. Physics-informed neural networks by Gradient-Guided Gaussian Adaptive Sampling (3GAS-PINNs)

**arXiv ID:** 2609.09162 | [PDF](https://arxiv.org/pdf/2609.09162v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 47. Epoch: Compiling Diffusion Blocks for Sparse MoE Serving

**arXiv ID:** 2609.09748 | [PDF](https://arxiv.org/pdf/2609.09748v1)

**作者:** Jianian Zhu `[一作]` (Huazhong University of Science and Technology), Jidong Zhai `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一种名为 Epoch 的执行器，将扩散式语言模型（dLLM）的扩散块视为编译单元，预先编译块级结构（专家支持、序列分片、缓存描述符），在每一次迭代中只刷新影响解码的值，显著减少 MoE 推理中的冗余计算与通信。

**💡 创新点**

创新点包括：
1) 把扩散块作为可复用的编译单元，区分块时钟与迭代时钟；
2) 仅缓存块级专家支持与稳定解码位置的 MoE 输出，重新计算每一步的路由与 logits；
3) 在专家、token 状态与 payload 轴上分别实现专家裁剪、解码位置缓存与仅传输新鲜 token‑expert 工作列表，从而压缩 dispatch/ combine 的通信；
4) 通过可控的刷新周期 M_C 维持缓存稳定性，保证解码决策不受影响。

**🔧 技术方法**

采用了 Mixture‑of‑Experts (MoE) 推理、块级调度、稀疏 MoE 管线、专家支持压缩、稳定解码位置缓存、compact packing、稀疏 dispatch/ combine、专家并行（EP）、张量并行（TP）和序列分片（SP）。实现基于 vLLM 融合 MoE 内核、NCCL 复合通信、Triton 加速，部署在 8 台 NVIDIA H100 GPU 上。

**📊 数据集**

使用的模型：LLaDA‑MoE (1B active / 7B total)、LLaDA2.0‑mini (1.4B active / 16B total)、LLaDA2.0‑Flash (6.1B active / 100B total)。
使用的数据集：GSM8K、HumanEval、MGSM、MT‑Bench（均采用默认提示、贪心阈值解码 τ=0.95，最多 64 次迭代）。

**📈 对比分析**

与 dInfer（无缓存/有缓存）以及 SGLang 的 forward‑scoped MoE 运行时进行对比；并做 ablation（全计算、去专家稀疏、去 token 稀疏）。评估指标包括：end‑to‑end 请求执行时间、每迭代时延、P95 ITL、计算/内存/通信占比。结果显示：
- 在 batch 512、模型 100B 时，Epoch 可比最优基线快 1.7–2.7×；
- 在 batch 128/256 时，平均提升 1.1–1.7×；
- 在线尾部延迟降低约 1.7–1.8×；
- 性能提升来源分别为专家裁剪（≈50% 计算降低）、解码缓存（≈30% 计算减少）和稀疏通信（≈40% 通信减少）。
- 任务准确率与基线保持一致，质量不受影响。

**⚠️ 局限性**

局限性：
1) 仅适用于块扩散式 MoE LLM，无法直接迁移到纯自回归模型；
2) 需要稳定的块大小与阈值解码策略，动态块或更灵活的解码机制可能导致缓存失效；
3) 需要手动调节刷新周期 M_C，过大可能引入解码误差，过小则收益有限；
4) 在极小 batch 或极低并发场景下，块级预编译开销可能抵消收益；
5) 对硬件依赖较强，需支持高带宽 NVLink/NVSwitch 以及高效的稀疏 collectives；
6) 目前实现仅在 8 台 H100 GPU 上验证，尚未在更大规模集群或多节点环境中评估。

---

## 48. UnsafeChecker: Finding Soundness Bugs in Rust Safe Abstractions

**arXiv ID:** 2609.09641 | [PDF](https://arxiv.org/pdf/2609.09641v1)

**作者:** Xizhe Yin `[一作]` (Nanjing University), Baowen Xu `[通讯]` (Nanjing University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于Rust MIR的静态分析框架，能够检测安全抽象中由于unsafe代码导致的潜在无定义行为和契约违背；

**💡 创新点**

创新点在于统一维护所有权、生命周期和布局三大语义维度的共享状态，并通过流感知抽象解释实现对unsafe实现内部安全契约的检查；

**🔧 技术方法**

采用了Rust MIR抽象解释、flow‑sensitive数据流分析、三维抽象域（ownership、object‑validity、layout）以及数值域与SMT求解器对指针算术进行精确约束；

**📊 数据集**

使用两大数据集：Dataset A为46个RustSec CVE（53个真实缺陷）做基准评测，Dataset B为100k+ crates.io crate 的大规模扫描结果；

**📈 对比分析**

与MirChecker、Rudra、SafeDrop等基线工具比较，Recall提升至约68%，检测32/46 CVE；在大规模扫描中发现114个验证可触发的bug，已确认45个、27个已修复；分析运行时与内存开销均在可接受范围内（平均分析时≤12 s，内存≤4 GB）；

**⚠️ 局限性**

局限在于只覆盖顺序内存安全，忽略并发与Ffi细节，数值域使用区间抽象导致的误报和漏报，以及未建模的标准库API可能造成的缺陷；

---

## 49. Violet: Enabling Full Virtualization for M-mode RTOS on RISC-V

**arXiv ID:** 2609.09833 | [PDF](https://arxiv.org/pdf/2609.09833v1)

**作者:** Taro Kito `[一作]` (Ritsumeikan University), Koichi Mouri `[通讯]` (Ritsumeikan University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 Violet 超级监视器，使得现有的 M‑mode RTOS 能在 RISC‑V 虚拟机中不做任何修改就能运行，并与 Linux 等 GPOS 共存。

**💡 创新点**

创新点在于将 RISC‑V 的 Hypervisor 扩展与软件实现的 M‑mode 模拟相结合，实现了对 M‑mode 的完整虚拟化；同时支持多 OS 并发执行，填补了目前仅支持 S‑mode/ U‑mode 的空白。

**🔧 技术方法**

主要技术包括：RISC‑V Hypervisor Extension（HS/VS/VU 模式）、软件实现的 M‑mode 模拟器（特权指令、CSR、中断、CLINT 的仿真）、SBI 虚拟化、设备直通和 VPLIC/VCLINT 机制；实现基于 C/C++ 的超监视器，并在 SiFive HiFive P550 硬件上部署。

**📊 数据集**

使用的数据集包括：riscv‑arch‑test（约 674 个测试）、FreeRTOS 示例程序、Linux 6.6.21 内核、SiFive HiFive Premier P550 开发板（四核 1.4 GHz、16 GB LPDDR5）和实际的硬件测量数据。

**📈 对比分析**

评估方法：先在 RISC‑V 体系结构测试框架 RISCOF 下跑 riscv‑arch‑test 验证 M‑mode 模拟一致性；随后在真实硬件上分别跑 FreeRTOS、Linux 与两者并发运行，并测量 M‑mode CSR 访问延迟、定时器中断延迟以及线程度量测试套件的协作调度上下文切换时间。结果显示：CSR 访问平均额外开销约 1.1k–1.7k 周期；定时器中断延迟从 0.16 µs 提升至 1.05 µs；上下文切换延迟从 0.16 µs 提升至 31 µs，整体性能开销明显但可接受。

**⚠️ 局限性**

主要局限：M‑mode 模拟依赖软件陷阱+仿真，导致显著的 CSR 访问与中断延迟；未实现对共享硬件资源（如缓存、内存带宽）的隔离；缺乏对 QoS 扩展（Ssqosid/CBQRI）的支持，无法实现严格的实时性与资源分配控制。

---

## 50. StreamAlign: Streaming Text-Aligned Speech Tokenization

**arXiv ID:** 2609.09719 | [PDF](https://arxiv.org/pdf/2609.09719v1)

**作者:** Kang-wook Kim `[一作]` (Seoul National University), Gunhee Kim `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一个文本对齐的语音分词框架，支持实时流式分词与重构，并基于该分词单元训练了语音语言模型；

**💡 创新点**

创新点在于：1）将字符级RNN‑T对齐与词级ASR引导结合，实现在线音素对齐，解决ASR‑LLM词表不匹配；2）使用主动词边界检测显著降低延迟；3）采用子词级聚合保留细粒度声学信息；4）整体实现流式文本对齐的完整链路；

**🔧 技术方法**

采用Conformer音频编码器、RNN‑T字符对齐、Transformer聚合、RVQ离散化、两阶段解码（单元预测+预训练TTS解码器）、词边界MLP、以及基于Llama的SLM；

**📊 数据集**

使用LibriTTS、Emilia语音数据训练；在LibriSpeech test‑clean评测重构性能；下游评测使用SALMon、StoryCloze、VCTK等数据集；

**📈 对比分析**

与多种非流式、流式、文本条件化分词器对比；在重构任务中获得最低4.41% WER、最高4.23 UTMOS；在语音连续生成和likelihood‑based分类任务中均优于基线SLM；RTF 0.35，延迟 270 ms；

**⚠️ 局限性**

局限性：仅在英语语料上验证，缺乏多语种与代码混合场景评估；未在非读说话的SLM上测试；依赖引导ASR，识别误差仍可能影响性能。

---

## 51. AgentHijack: Visual Patch Attacks on Multimodal Computer-Use Agents

**arXiv ID:** 2609.09212 | [PDF](https://arxiv.org/pdf/2609.09212v1)

**作者:** Zhihao Liu `[一作]` (Hainan University), Yuqing Zhang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并实验了一个端到端的评估框架，用可训练的视觉补丁诱导多模态计算机使用代理（CUA）执行恶意终端命令，从截图输入到 VLM 输出、动作解析、环境执行全过程进行跟踪与评估。

**💡 创新点**

创新点包括：① 将视觉触发攻击拆解为三层（文本输出、动作解析、环境变更）并分别量化；② 采用固定位置、有限扰动的补丁训练，结合离线目标字符串监督与在线真实环境验证；③ 通过 600 个实例级真实实验揭示补丁在不同代理栈中的传播损耗与实际危害。

**🔧 技术方法**

使用的技术包括：多模态大模型（Qwen3.5-4B/9B、Aguvis-7B、EvoCUA-8B、UI-TARS-1.5-7B）+统一模型适配层；补丁优化采用 Adam、L∞约束；在线实验通过真实网页截图（GitHub Pages、CSDN 克隆）与代理包装器交互；日志记录、截图、动作解析器与评估器构成完整闭环。

**📊 数据集**

数据集：5 个公开可复现的 GUI‑agent + VLM 后端；每个后端在 120 个（15 种攻击目标 × 2 网页环境 × 4 友好任务）逻辑案例中执行，共 600 个实例；补丁尺寸 700×500；补丁位置固定（如 GitHub 361,165）。

**📈 对比分析**

比较方法：对每个后端分别统计 ① Gen→ParserGap（文本命中率）、② Parser→ExecutionGap（动作解析通过率）、③ 环境成功率。总体上 84.5% 的文本命中率降到 47.0% 的解析通过率，再降到 20.3% 的环境成功率；不同后端表现差异显著，Qwen3.5 系列最优，UI‑TARS 最差。

**⚠️ 局限性**

局限性：① 仅在作者控制的网页与离线虚拟机中测试，未覆盖真实第三方网站；② 补丁高度位置特异，缺乏对不同页面布局的鲁棒性；③ 只针对 VLM‑基代理，未评估其他类型代理；④ 评估未涉及更高级的防御机制，且只展示了实验结果，没有进一步的安全对策研究。

---

## 52. Compute-Bounded Security Assurance - Coverage, Verification, and Response under Resource Constraints

**arXiv ID:** 2609.09229 | [PDF](https://arxiv.org/pdf/2609.09229v1)

**作者:** Jithin VG `[一作]` (Bud Ecosystem), Ditto PS `[通讯]` (Bud Ecosystem)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了在有限计算预算下的安全保证框架，推导覆盖率模型、误差评估和响应延迟关系，并提出概念性防御架构与评估协议。

**💡 创新点**

将覆盖率与成功概率、支持上限、误差、资源和时间等因素分离；展示了均值与协方差不能确定覆盖上限的反例；整合正式证明、经验评估与经济资源考量，提供完整的理论合成。

**🔧 技术方法**

组合概率与随机过程、子模函数、有效样本量理论、期望值与方差分析、最优分配与KKT、服务容量与排队论、指数分布的风险模型、正确性与误判评估等技术。

**📊 数据集**

本文未使用真实数据集，全部使用解析公式和合成参数进行数值演示；采用虚构的任务集和参数验证理论。

**📈 对比分析**

提出基准评估协议（hold‑out任务、配对比较、负样本、不确定性报告），但未给出实验结果；理论上展示不同配置在预算、覆盖率和响应延迟上的相对收益，可用于后续实证验证。

**⚠️ 局限性**

依赖严格的假设（条件独立、固定评估宇宙、指数时延）；缺乏实测数据与部署实验；无法直接评估实际安全收益；对高阶联合分布假设有限；仅提供理论框架与数值示例。

---

## 53. When Auditors Fabricate: Batch-Size Degradation and Confident Hallucination in LLM Detection of Planted Document Contamination

**arXiv ID:** 2609.09696 | [PDF](https://arxiv.org/pdf/2609.09696v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 54. A Kernel-Based Modular Discriminant Analysis Framework for Small-Sample Learning

**arXiv ID:** 2609.09910 | [PDF](https://arxiv.org/pdf/2609.09910v1)

**作者:** Lingxiao Qu `[一作]` (University of Aizu), Yan Pei `[通讯]` (University of Aizu)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对小样本学习问题，系统研究了 Kernelized Linear Principal Component Discriminant Analysis（KLPCDA）在不同领域中的表现，并分析其三大目标（总方差、类间分离、类内紧凑）之间的相互作用。

**💡 创新点**

提出了 KLPCDA 的统一可插拔框架，能够灵活组合三项目标产生七种变体，并给出了根据数据特征选择合适变体的实用准则。

**🔧 技术方法**

利用核方法构建 KLPCDA，进行特征空间投影，并与 PCA+LDA、RLDA、KPCA+LDA、SVM、FCNN、2D‑CNN、HybridSN、ProtoNet‑inspired、SimCLR‑inspired 等基线进行对比。

**📊 数据集**

在四个真实世界数据集上评估：印度平原（hyperspectral 图像）、CWRU 轴承故障诊断、GSE44076 结肠癌基因表达、JAFFE 个人/表情识别。

**📈 对比分析**

实验表明 KLPCDA 变体在所有任务上均优于传统方法，尤其是方法 5（C+S_w）在高维基因数据上表现最佳；方法 4 在信号故障检测上最好；方法 7 在细粒度人脸表情识别中取得最高准确率；整体而言，KLPCDA 在小样本、高维和类别不平衡场景下兼具鲁棒性和较低计算成本。

**⚠️ 局限性**

在极细微类内差异（如表情识别）场景下，KLPCDA 的提升有限；且虽然比深度网络更轻量，但仍需进行核矩阵运算，对样本数较大时的效率与可扩展性有待进一步优化。

---

## 55. Privacy-Preserving Split Learning for Federated LLM Fine-Tuning

**arXiv ID:** 2609.09794 | [PDF](https://arxiv.org/pdf/2609.09794v1)

**作者:** Heng Jin `[一作]` (Virginia Tech), Y. Thomas Hou `[通讯]` (Virginia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一种基于分割学习和联邦学习的LLM微调隐私保护框架，能够在多机构数据分布且参与方无法承载完整模型的场景下进行联邦微调，且服务器端仍可获得完整可直接部署的模型。

**💡 创新点**

创新点在于设计了学习式遮蔽与恢复的两层轻量化适配器，拆分激活空间使服务器在接收的中间激活无法直接解码输入，从而根除因自回归模型导致的输入泄漏；并通过STE保持对模型主体的有效更新。

**🔧 技术方法**

技术手段包括：分割学习（把LLM划分为A、B、C三段），联邦学习（FedAvg聚合），两层适配器（W1遮蔽、W2恢复）训练联合损失（JSD、CKA、KL、MSE），Straight‑Through Estimator做反向传播，和对齐数据训练适配器。

**📊 数据集**

实验使用OASST1（无关任务）作为对齐数据；微调任务采用Banking77、CLINC150（分类）和MentalChat16K（生成）三个公开数据集；测试了Llama-3.2-3B、Llama-3.1-8B、Ministral-3-8B、Ministral-3-14B四种LLM。

**📈 对比分析**

与Embedding ε‑Privacy、Smashed‑Data DP、NoPeek三种基线防御进行对比，评估指标为ROUGE‑1重构分数（越低越好）和任务性能（准确率/ROUGE‑1）。实验显示该方法将重构分数降至接近0，且在所有任务和模型上保持与无防御相近的性能；训练时间与资源开销仅略增5‑10%。

**⚠️ 局限性**

局限性：仅针对切点激活泄漏做防御，未涵盖梯度逆向攻击；依赖对齐数据的可用性和多机构间对齐策略；实验规模有限，未评估在更大规模、多模型或极端通信延迟环境下的鲁棒性和可扩展性。

---

## 56. CompassOPD: Cross-Family On-Policy Distillation via Within-Family Likelihood Shifts

**arXiv ID:** 2609.10154 | [PDF](https://arxiv.org/pdf/2609.10154v1)

**作者:** Naibin Gu `[一作]` (Chinese Academy of Sciences), Weiping Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CompassOPD，改进跨模型族的On-Policy Distillation（OPD）方法，通过移除跨族偏移并使用教师族内部的相对似然变化来提升学生模型的推理性能。

**💡 创新点**

创新点在于识别并去除跨族偏移的干扰，仅利用教师族内的likelihood shift传递能力提升，同时引入冻结的学生参考策略以调节更新方向。

**🔧 技术方法**

使用OPD框架、文本空间对齐、停止梯度技术、KL正则化以及对MoE教师的专家激活配置进行自我参考。

**📊 数据集**

使用DAPHO-Math和DeepScaleR-Preview进行训练，评估基准包括AIME 2024/25/26、HMMT 2025/26、MATH-500等七个推理数据集。

**📈 对比分析**

在三种学生族（Granite4.1-3B、Qwen3-4B、OLMo-3-7B）和三种教师族（Qwen3.5、Qwen3、Mistral）上进行对比，CompassOPD平均提升约5.5分（最优可达5.5分），在大多数配置下均优于传统OPD；自我参考MoE设置也实现了3.43分的提升。

**⚠️ 局限性**

实验未覆盖极大规模模型，对非常大模型的适用性仍待验证；此外，需要教师族中存在低能力参考模型或可通过MoE激活获得参考。

---

## 57. Contextual Utility of Quantization Moves in Extreme Low-Bit LLMs

**arXiv ID:** 2609.09867 | [PDF](https://arxiv.org/pdf/2609.09867v1)

**作者:** Wenxuan Xiao `[一作]` (Astrmira Tech), Xu Cao `[通讯]` (Astrmira Tech)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究后训练量化（PTQ）中单步移动（move）的实用性，发现只用当前状态梯度无法准确评估移动对最终模型性能的影响。提出在移动自身中点处评估梯度，并揭示一组移动的效用近似为二次伪布尔函数，配对项（pairwise term）决定 Pareto 前沿。基于此，设计了两类构造器（one-shot midpoint 选择与 recentered Pareto beam），在多模型（Llama‑3.2‑1B/3B、Qwen3‑4B）上验证其能显著提升未见数据的 perplexity、零射任务准确率，并能修复传统 GPTQ 及其它 PTQ 求解器造成的性能损失。

**💡 创新点**

创新点：
1) 首次系统证明移动在其中点梯度上比当前状态梯度更能预测实际损失变化，且误差可被量化。
2) 发现移动集合的效用近似为二次伪布尔函数，配对项占能量 0.01–13% 但决定 Pareto 前沿；常用的中心点评分忽略此项导致“漂移”。
3) 通过穷举 256‑状态格子验证上述二次结构，并在此基础上提出只需在中点读取梯度即可完成高质量搜索的构造器。
4) 在 4B 级别大规模 PTQ 步骤中展示传统求解器的“误解”与“误用”，并证明按中点评估并重新价格能恢复甚至超越原始模型。

**🔧 技术方法**

技术方法：
- 量化代理（reconstruction proxy）与多种功能（next‑token NLL、ARC option‑KL、HellaSwag KL 等）评估。
- 中点梯度（dᵀ∇F(q₀+½d)）与当前状态梯度比较。
- 穷举布尔格子与伪布尔函数拟合（Möbius 展开）来估计配对项 β。
- 一步中点选择与 recentered Pareto beam 两种构造器。
- 结合 GPTQ、AWQ、AWQ‑INT3、Llama‑3.2、Qwen3‑4B 等预训练模型和量化方案。
- 用 bootstrap / Student‑t 评估置信区间。

**📊 数据集**

数据集与模型：
- Llama‑3.2‑1B 与 3B（INT3 128‑组、block 128）
- Qwen3‑4B（INT3/INT3‑NVFP4）
- 评估指标： held‑out next‑token NLL、ARC option‑KL、HellaSwag KL、MMLU、零射任务（ARC‑Challenge）等。
- 训练、校准与测试块数分别记录，采用对比实验。

**📈 对比分析**

比较方法：
- 与标准 GPTQ（无激活排序、激活排序）对比；
- 与基于中心点评分的常规搜索（common‑center）对比；
- 与 exact‑endpoint 搜索（beam）对比。
- 结果：
  * 在 4B 量化步骤中，midpoint‑based 构造器提升 2.7–2.9 pp 在零射任务；
  * 在 1B GPTQ 上 mid‑point 选择使未见 perplexity 降低 11%；
  * 在 4B 级别 beam 方案相同预算下，能实现 19% perplexity 降低。
- 统计显示：midpoint 预测精度 > 99%（Spearman 0.9996），当前状态梯度仅 ~70%。

**⚠️ 局限性**

局限性：
- 实验仅覆盖八个原语的 256‑状态格子，难以推广到更大规模的移动集合。
- 构造器未使用估计的配对项 β，仍依赖 additive 近似；在高支持度移动时此近似失效。
- 只评估同速率（same‑rate）量化改动，异速率或混合精度场景未覆盖。
- 结果主要基于特定模型（Llama‑3.2、Qwen3‑4B）和任务，其他模型/任务的泛化尚需验证。

---

## 58. Dependency-Aware ROM/CBD Correctness Bounds for ML-KEM-768 at the Heuristic Failure Scale

**arXiv ID:** 2609.09983 | [PDF](https://arxiv.org/pdf/2609.09983v1)

**作者:** Aurélie Duriez `[一作]` (netHsys SARL), Christophe Tommasini `[通讯]` (Tommasini Conseil)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 ML‑KEM‑768 的 ROM/CBD 抽象下，给出了一个依赖保持的解密失败概率上界，证明误解密概率不超过 2⁻¹⁶⁴.81。

**💡 创新点**

创新点在于：① 采用图耦合全理想参考模型；② 用完备的傅里叶反压和总变差控制实现理想到实际的精确变换；③ 对三重 CRT 组合进行三因子稀疏极限反浓度证明；④ 结合精确整数计算、理性区间算术和 256 维并行统一上界，最终获得与 FIPS 203 经验值相同的 2⁻¹⁶⁴.8 级别上界。

**🔧 技术方法**

使用的技术包括：高精度整数计算、有限域傅里叶分析、总变差与反压不等式、CRT 理想分层、离散化多变量传输、联合取样与离散对数等；所有计算均通过可重现的脚本和证书实现。

**📊 数据集**

论文没有使用传统意义上的数据集，而是基于 ML‑KEM‑768 参数（q=3329, n=256, k=3 等）以及中心二项分布（CBD₂）随机抽样进行理论与实验验证；所有随机抽样均在可重现的 ROM/CBD 环境下执行。

**📈 对比分析**

与之前的工作相比，该结果在同一抽象模型下取得了更紧的上界：先前的正式证明给出约 2⁻¹⁷⁰ 的上界，经验模型给出 2⁻¹⁶⁴.8；本工作通过完整的理想-傅里叶-反压链条，将上界收敛到 2⁻¹⁶⁴.81，几乎与经验值相等，显示出极高的精度与可靠性。

**⚠️ 局限性**

主要局限性包括：① 仅在 ROM/CBD 抽象下成立，无法直接推导到固定 SHAKE 实现；② 结果是上界而非精确失败分布；③ 未实现 IND‑CCA 级别的安全性或自适应正确性；④ 证明链条虽可重现，但尚未在形式化证明助手中完全验证，需进一步形式化。

---

## 59. SCCM : Stream Cruise Control Method for Automated Drift Detection and Adaptation

**arXiv ID:** 2609.09432 | [PDF](https://arxiv.org/pdf/2609.09432v1)

**作者:** Mohammad Abu-Shaira `[一作]` (University of North Texas), Weishi Shi `[通讯]` (University of North Texas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了Stream Cruise Control Method（SCCM），一种在线回归中预更新控制层，能在检测到概念漂移前自动调节模型超参数并进行有限重校。

**💡 创新点**

创新点包括：①基于KPI-Window的CFAR式自适应阈值实现早期漂移检测；②漂移幅度量化并映射为比例超参数调节；③利用尺度映射（Scale Map）实现多模型的动态超参数调整；④在漂移持续时进行有限的模型重校，以避免过度适应。

**🔧 技术方法**

使用的技术主要有：KPI监控窗口（KPI‑Win）、CFAR启发式阈值、漂移幅度评估、尺度映射调参、在线回归模型（RLS、PA、LMS、OLR‑WA）的动态超参数更新以及基于窗口的有限重校。

**📊 数据集**

数据集包括18个合成数据集（急性、增量、交替渐进漂移），以及8个真实世界回归数据集，涵盖医疗、金融、房地产、能源、环境、化学、海洋和零售等领域。

**📈 对比分析**

通过与八个检测–适应基线（ADWIN‑RESET/​WINDOW/​SSPT/​OHL、KSWIN‑RESET/​WINDOW/​SSPT/​OHL）以及基础模型进行对比，SCCM在R²和MSE指标上均显著优于基线，漂移检测更及时、误报率更低，整体预测性能提升幅度可达数个百分点。

**⚠️ 局限性**

局限性包括：①需先设定KPI与敏感度参数ρ，影响检测灵敏度；②对空间/局部漂移的定位能力有限；③实验仅聚焦回归任务，扩展至分类需进一步验证；④在高噪声或极端非平稳场景下的阈值自适应机制可能需要进一步调优。

---

## 60. XAI-Arena: Can LLMs Assess the Quality of XAI Explanations?

**arXiv ID:** 2609.09428 | [PDF](https://arxiv.org/pdf/2609.09428v1)

**作者:** Yanfei Hu Fleischhauer `[一作]` (LMU Munich), Stefan Feuerriegel `[通讯]` (LMU Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证了XAI-Arena框架，利用大型语言模型对XAI解释质量进行可复现、多维度、利益相关者敏感的评估。

**💡 创新点**

引入LLM-as-a-judge机制，实现大规模、可复制的XAI解释质量评估，覆盖八个质量维度并考虑不同利益相关者视角。

**🔧 技术方法**

使用GPT-5.4 LLM进行评分，结合SHAP、LIME、DiCE、PDP、Permutation Importance等XAI方法；采用标准化提示模板、固定解码设置；通过Spearman相关、ANOVA等统计分析。

**📊 数据集**

9,504条合成数据（线性/非线性、20/60/100特征、不同样本量）和3个真实世界表格数据集（WDBC、Telco churn、California Housing）。

**📈 对比分析**

对模型（LR、RF、XGBoost、MLP）、XAI方法、解释格式、利益相关者角色进行全因子设计；LLM评分与人工评估相关系数0.693，代理指标与LLM评分高度相关；不同维度、模型、方法间显著差异。

**⚠️ 局限性**

仅使用单一LLM（GPT-5.4），提示设计可能影响结果；仅限表格数据和部分XAI方法；文本与图像评估分离；可能存在预训练泄漏；缺乏实际高风险领域的验证。

---

## 61. World-Time Compute with Verified Code World Models

**arXiv ID:** 2609.09163 | [PDF](https://arxiv.org/pdf/2609.09163v1)

**作者:** James Schwoebel `[一作]`, Martin G. Frasch `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出利用已验证的代码世界模型（CWMs）生成无限、完全标注的训练样本，称之为“world‑time compute”，并在多种符号性任务（程序修复、诊断、ARC、List Functions 等）上验证其在模型泛化、规划和决策中的优势。

**💡 创新点**

核心创新在于：①将领域规则转化为可验证的程序并自动合成；②通过程序验证保证每条轨迹标注准确无误；③在 LLM 上进行基于这些轨迹的细调，实现跨世界泛化的“世界时间计算”，从而在小模型或少量经验下获得显著提升。

**🔧 技术方法**

技术手段包括：LLM 代码生成与多门（语法、沙箱运行、逆向验证）验证；符号状态空间设计；QLoRA 微调；对比学习和激活判别；以及多种实验框架（openworld）。

**📊 数据集**

数据集涵盖：人工编写的可验证世界（sprint、orchard、triage、program repair、ARC‑AGI、List Functions、CLRS‑Text、Bongard‑RWR 等）和通过规则增强生成的众多子世界。

**📈 对比分析**

比较方法主要为：①与基于 LLM 单步预测的代理、MLP、1‑NN 等学习动态模型对比；②与已知规则的代码合成对比；③跨世界训练与单世界训练的精度差异；结果显示：代码合成模型在 OOD 精度可达 100%，学习模型在 OOD 低于 1%；world‑time compute 在小模型上提升 10–30% 以上，且提升随世界数量呈先上升后趋于平稳。

**⚠️ 局限性**

限制主要包括：仅适用于可符号化的状态空间，无法处理像像素、3D 之类的感知任务；单块合成在规则数超过约 8 条时失效（复杂度悬崖）；需要手工编写或自动生成的高质量规则；对非符号化领域的迁移性仍需进一步验证。

---

## 62. Efficient Leakage-Free Neural Architecture Search under Leave-One-Subject-Out Evaluation

**arXiv ID:** 2609.09433 | [PDF](https://arxiv.org/pdf/2609.09433v1)

**作者:** Heinke Hihn `[一作]` `[通讯]` (IU International University of Applied Sciences), Heinke Hihn (IU International University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

本文提出了一种基于块划分、泄漏-free 的 NAS 方法，用于离群子样本留一交叉验证（LOSO）下的疼痛级别分类，显著降低了计算成本并提升了模型性能。

**💡 创新点**

创新点在于通过将受试者划分为外部块共享 NAS 过程，避免了每个折叠独立搜索，同时引入了加权方差惩罚的选取指标，既提高了搜索效率，又兼顾了跨受试者的泛化。

**🔧 技术方法**

技术手段包括使用 Optuna 的树结构 Parzen 估计器进行 NAS、块化交叉验证、加权方差惩罚的评估公式、早期融合（EF）深度学习模型以及块级共享训练与评估流程。

**📊 数据集**

实验使用 BioVid 热痛数据集，在二分类痛感/无痛的场景下进行。

**📈 对比分析**

与现有 EF、LF 方法对比，本文在 EF 设置下将平均准确率从 82.79% 提升至 83.39%，并将参数量从 770 万降至 61k–1.79M，计算时间从 90 小时缩减至 12 小时。

**⚠️ 局限性**

局限在于架构及参数的高方差，缺乏统一的部署选择标准，且仅在二分类任务验证，未来需探讨多类别、LF 体系以及其它 LOSO 任务的适用性。

---

## 63. When Does Low-Bit Quantization Preserve the Decisions of Vector Search?

**arXiv ID:** 2609.09854 | [PDF](https://arxiv.org/pdf/2609.09854v1)

**作者:** Wenxuan Xiao `[一作]` (Astrmira Tech), Xu Cao `[通讯]` (Astrmira Tech)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究低比特量化在向量搜索中对决策稳定性的影响，提出了基于比较边界与残差的分解方法，并给出了判定一致性的确定性耦合定理，进一步利用精确高斯模型和旋转分析解释残差协方差和量化误差对决策风险的作用；

**💡 创新点**

创新点在于：①提出无分布假设的边界‑残差决策风险分解，揭示平均误差无法捕捉决策错误；②给出Vamana等图搜索算法的确定性轨迹耦合定理，将局部比较误差升维到全局邻居列表；③在精确高斯表示下导出Stein协方差公式和大偏差构造，证明低阶高斯诊断不足以保证指数尾；④通过块级留置证书实现仅凭样本即可给出选择性风险界限。

**🔧 技术方法**

采用的技术包括：边界‑残差风险分解、协方差感知残差尾界、Vamana选择调用的确定性状态机耦合、Stein协方差同义式、旋转均匀化分析、条件MGF和截断尾界、块级留置贝尔兹曼与Hoeffding证书、以及对量化器的残差机制建模。

**📊 数据集**

使用的数据集涵盖了学习得到的对比表示（Cohere、MiniLM、BGE‑M3、Jina、MSMARCO、Wolt‑CLIP、Landmark‑DINO）、经典手工特征（SIFT、GIST、GloVe）以及合成高斯、球面、随机与污染等模拟数据。

**📈 对比分析**

通过对不同量化器（1‑bit/2‑bit坐标代码、RaBitQ、Lucene BBQ、scalar int4、PQ）在同一决策集上评估flip率、Spearman相关、均方误差、以及块级留置证书风险，发现标准化边界能高度预测局部错误率，旋转能显著提升全局相似度但对局部决策影响不一；整体上，scalar int4在决策风险上最优，尽管在最难5%边界上仍有10%错误。

**⚠️ 局限性**

局限性包括：①轨迹耦合证书在长路径上饱和且不涉及候选集生成或图导航；②高斯同义式仅为模型下的oracle，近似诊断无法直接推广；③块级留置证书需代表性独立块且对分布漂移不具备迁移性；④本文未覆盖候选覆盖导致的端到端召回依赖。

---

## 64. MLLMs Hallucinate when Information Distribution Drifts in Synergy Heads

**arXiv ID:** 2609.09206 | [PDF](https://arxiv.org/pdf/2609.09206v1)

**作者:** Meng'en Qin `[一作]` (Shenzhen University of Advanced Technology), Ruize Han `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HEAL，一种通过头层信息解耦和动态校准来减少多模态大语言模型的幻觉生成方法

**💡 创新点**

通过因果噪声干预和差分因果分析精确区分注意力头为冗余、视觉、语言和协同四类，并引入视觉‑语言均衡因子动态校准协同头的内容分布

**🔧 技术方法**

因果噪声干预、差分因果分析、对抗信息分解、动态均衡校准、并行批量化实现

**📊 数据集**

在 LLaVA-Bench、MME、BLINK-Twice、POPE、CHAIR、MMHal-Bench 等多模态和幻觉评测数据集上评测

**📈 对比分析**

与现有多模态模型及幻觉抑制技术（如 VCD、EAH、FarSight 等）进行对比，HEAL 在幻觉指标上显著提升（如 POPE F1、CHAIR 召回率提升约3-5%），同时保持或提升通用多模态性能

**⚠️ 局限性**

均衡因子和更新间隔需经验确定，且仅在注意力层层面校准，对早期视觉编码失败或缺失证据导致的幻觉效果有限

---

## 65. Multi-Robot Scanner for Automated Full-Body Dermoscopic Imaging

**arXiv ID:** 2609.10169 | [PDF](https://arxiv.org/pdf/2609.10169v1)

**作者:** Valerio Franchi `[一作]` (University of Girona), Josep Malvehy `[通讯]` (Hospital Clínic de Barcelona)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并实现了一套基于四台UR10协作机器人、液体可调焦镜头的全身非接触性皮肤镜级别成像扫描器，并通过自动视图规划和碰撞检测实现完整的皮肤病变自动采集。

**💡 创新点**

创新点包括：① 机器人自动化全身皮肤镜图像采集；② 采用液体透镜实现毫秒级快速聚焦；③ 视图规划算法根据3D重建动态定位最佳相机位置；④ 将总身摄影与皮肤镜图像采集合并为一体化工作流，缩短临床检查时间并提升数据安全性。

**🔧 技术方法**

使用技术包括：UR10e协作机器人、液体可调焦镜头、3D重建（ToF+RGB摄像机）、YOLOv8肿瘤检测、视图规划算法、深度学习超分辨率（GPEN、Real‑ESRGAN）、多相机校准、SMPL人体模型匿名化、PLC控制及触摸界面。

**📊 数据集**

使用的数据集为156例皮肤病变，每例使用四种成像模式（Vectra、传统接触式皮肤镜、Native、SR）进行采集，并由两名皮肤科专家对17项临床特征进行评价。

**📈 对比分析**

比较方法：对三种成像模式（接触皮肤镜、扫描器Native/SR、Vectra）按专家评分计算每个特征可见度比例；利用USAF 1951目标测量真实光学分辨率；结果显示扫描器在多数特征上与接触皮肤镜相当，显著优于Vectra；但在回归区、荧光等特征上仍落后于手持皮肤镜，且真实光学分辨率（22.1 vs 8.8）低于传统手持皮肤镜。

**⚠️ 局限性**

局限性：① 真实光学分辨率仍低于接触式皮肤镜；② 超分辨率仅提升像素密度，未改善真实分辨率；③ 缺乏大规模临床验证和完整工作流效率量化；④ 对某些关键特征（回归区、细微血管等）识别仍不及手持皮肤镜；⑤ 受限于液体镜头的工作温度和环境稳定性。

---

## 66. Fixed-mesh based approach for modeling of superconducting magnetic bearings

**arXiv ID:** 2609.09347 | [PDF](https://arxiv.org/pdf/2609.09347v1)

**作者:** Elias Paakkunainen `[一作]` (TU Darmstadt), Sebastian Schöps `[通讯]` (TU Darmstadt)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并验证了一种固定网格（Fixed‑Mesh）方法，用于模拟超导磁悬浮装置（SMB）在平移运动下的磁耦合和升力行为。

**💡 创新点**

创新点在于：仅在空气区引入坐标变换，将相对运动转化为时间变化的各向异性磁导率张量；无需重新网格，直接嵌入现有FEM工作流程；兼容j‑a、h‑formulation等多种电磁形式。

**🔧 技术方法**

使用了2D有限元（j‑a）形式、均匀化（homogenization）和电路耦合，配合GetDP/Gmsh实现时间依赖的磁导率张量；对比重网格模型、实验数据，采用R²和最大绝对误差评估精度。

**📊 数据集**

采用了实验测得的SMB升力数据（来源于文献）、双路电流环（DCL）几何尺寸、永磁体磁化强度、铁磁导率、HTS功率律等参数作为数据集。

**📈 对比分析**

通过与重网格模型和实验数据比较，使用R²值（>0.99）和最大绝对误差评估，验证模型精度；计算效率显著提升，DoFs与计算时间从约8–9 h降至0.26–0.5 h，缩减至原来的一半以上。

**⚠️ 局限性**

局限性：目前仅处理平移运动，未耦合运动方程；大旋转角度可能导致精度下降；二维模型限制，需进一步推广至三维；对材料非线性和温度、场依赖的假设仍需进一步验证。

---

## 67. With a Thermomix You Lose the Ability to Cook: A Kitchen Machine Analogy for Applications of Generative AI in Education

**arXiv ID:** 2609.09856 | [PDF](https://arxiv.org/pdf/2609.09856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 68. How Far Do Capability Cues Travel? Anthropomorphism and Differentiated Trust in a Platform-Embedded AI Assistant

**arXiv ID:** 2609.09713 | [PDF](https://arxiv.org/pdf/2609.09713v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 69. Shifting Relational Paradigms for Affective Computing: Affective Resonance, Vitality Affects, and Vocal Interaction Fields

**arXiv ID:** 2609.09864 | [PDF](https://arxiv.org/pdf/2609.09864v1)

**作者:** Cy Gorman `[一作]` (Nurobodi), Yihang Yao `[通讯]` (Nurobodi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出将情感计算视为互动场域而非个体状态的范式，并在AMI会议语料中利用连续自监督语音表示检测多方对话中的方向性表达耦合；

**💡 创新点**

创新点包括：① 构建关系型框架Affective Resonance Dynamic Ontologies（ARDO）与相应的Artificial Affective Resonance Intelligence（AARI）；② 开发基于零点校准的方向性耦合分析方法；③ 采用四说话人条件VAR模型，消除群体级混淆；④ 证明耦合仅在相互共处（Tier A）时出现，支持情感共振的场域理论；

**🔧 技术方法**

技术手段：连续自监督语音特征（WavLM‑Base+）、跨层激活分散度作为表达性指标、Granger因果/多元VAR、四说话人条件建模、循环移位零点校准、FDR多重检验校正；

**📊 数据集**

使用数据集：AMI会议语料（四声道会议录音，16000 Hz）；

**📈 对比分析**

比较方法：在不同互动情境（Tier A、B、C）下进行方向性耦合检验，并与能量代理进行残差控制；性能表现为：微尺度（≤1 s）Tier A显著率提升约+6.6–7.4个百分点；宏尺度效果弱；与非耦合控制相比，耦合显著降低；

**⚠️ 局限性**

局限性：仅在AMI数据上验证，跨语料通用性待验证；表达性代理不涉及情绪标签；Granger分析仅捕获线性预测，无法区分收敛与发散耦合；宏尺度统计功效低；未来需加入多模态、非线性（如传输熵）以及更大样本量。

---

## 70. Echoes in the Algorithm: Analyzing the Fidelity of User Preferences Against Realized Platform Reach

**arXiv ID:** 2609.09365 | [PDF](https://arxiv.org/pdf/2609.09365v1)

**作者:** Emelia Hughes `[一作]` (ND-IBM Technology Ethics Lab), Tim Weninger `[通讯]` (ND-IBM Technology Ethics Lab)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过TokOrNot实验让受试者在隐藏计数的TikTok视频对比中判断哪条视频更受欢迎并记录个人偏好

**💡 创新点**

提出“结果可读性”框架，评估在计数被隐藏时用户对平台成功的推断程度

**🔧 技术方法**

使用已验证的公开观看次数作为客观对照，将其与用户预测进行对齐

**📊 数据集**

收集了由TikTok研究API检索并人工校正的跨类别视频对（9类，共1978对）

**📈 对比分析**

与真实观看数对比，预测准确率仅56.8%（略高于随机），但预测与个人偏好一致率达83.5%，表明仅靠内容难以精准读出受欢迎程度

**⚠️ 局限性**

局限包括固定问题顺序导致可能的锚定效应、样本为美国Prolific成年人且年龄偏大、仅用观看次数代表“受欢迎”且未考察算法内部机制

---

## 71. If It's Not Buggy, Don't Fix It: On the Dynamics of Iterative Bug-fixing with LLMs

**arXiv ID:** 2609.10123 | [PDF](https://arxiv.org/pdf/2609.10123v1)

**作者:** Xietao Wang-Lin `[一作]` (University of Warwick), Louis Mahon `[通讯]` (UnlikelyAI)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM在无历史信息的盲迭代bug修复过程中的动力学，量化修复率与损坏率，并发现了伪bug循环现象。

**💡 创新点**

创新点在于提出了观察并量化伪bug循环的实验框架，并通过线性探测揭示“buggy code”内在表示，进而可以用来调控编辑倾向。

**🔧 技术方法**

使用Gemini 2.5 Flash‑Lite和Qwen2.5‑7B‑Instruct两大LLM，分别以全文件编辑和搜索/替换块（SRB）两种方式进行迭代实验，并结合线性探测（steering vector）和logit‑lens分析。

**📊 数据集**

采用CodeContests+数据集，随机挑选20道竞赛题，每题40个C++提交（包含正确与错误程序），平均23个隐藏测试用例。

**📈 对比分析**

通过修复率α、损坏率β、吸引子类型及周期长度等指标进行比较，结果显示SRB相较于全文件修复具有更高损坏率、更长循环周期和更多退化现象；利用steering向量可显著抑制伪bug循环，提升修复效率。

**⚠️ 局限性**

局限在于仅针对小型模型、单文件、无目标的盲迭代环境进行实验，未探讨更大模型、目标驱动或多文件环境下的表现。

---

## 72. What Symmetry Buys a Learned Motion Planner

**arXiv ID:** 2609.10033 | [PDF](https://arxiv.org/pdf/2609.10033v1)

**作者:** Andrea Emir Sevincel `[一作]` `[通讯]` (Shanghai Jiao Tong University), Andrea Emir Sevincel (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在学习式运动规划中引入基于起止点的帧化处理，通过将任务投影到由起点和终点确定的局部坐标系，消除六自由度中的五个自由度，显著提升模型的泛化和成功率。

**💡 创新点**

提出了“查询驱动帧化”方法，即仅用一次叉乘即可在初始化时消除平移和两次旋转，并证明剩余的SO(2)旋转无法通过连续规则去除，阐明了对称性恢复的极限；同时提供了对残差非齐次性上界的理论证明，并展示其对模型性能预测的局限。

**🔧 技术方法**

使用条件流匹配（Conditional Flow Matching）作为生成模型，结合SE(3)等变卷积/向量神经网络结构、点云障碍编码、FiLM条件化和可选的SO(2)等变骨干网络；实验中还对比了数据增强、帧平均和等变权重等三种对称性恢复机制。

**📊 数据集**

采用人工合成的三维点质量规划基准PointMass3D：在[-1,1]^3内放置40个障碍物（20球、20盒子），共300个环境，每个环境600个起止对，训练集60个环境，测试集50个未见环境。

**📈 对比分析**

与传统采样/优化规划器（RRT‑Connect、CHOMP、TrajOpt）以及世界坐标训练的学习模型对比。结果显示，帧化模型在未见数据上的碰撞自由率从14.6%提升至51.1%，远超世界坐标模型和直线基准；在相同训练预算下，帧化可比传统算法的成功率高约30个百分点。

**⚠️ 局限性**

局限性包括：仅适用于在状态空间上可作用SE(3)的任务（如移动平台、自由飞行体、末端执行器空间规划），不适用于关节空间规划；基准难度相对较低且仅使用单一障碍生成器；残差非齐次性无法准确预测对称性恢复机制的收益；帧化的收益随数据量增大而下降；局部几何信息假设可在推理时获得，现实感知场景可能无法满足。

---

## 73. A Decade of Bayesian Optimization for Controller Tuning and Robot Learning: Tutorial, Review, and Future Prospects

**arXiv ID:** 2609.09403 | [PDF](https://arxiv.org/pdf/2609.09403v1)

**作者:** David Stenger `[一作]`, Sebastian Trimpe `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并提供实践指南，聚焦控制器调参与机器人学习中的贝叶斯优化（BO）方法，帮助研究者和工程师快速上手。

**💡 创新点**

①提出统一的 BO 视角，系统梳理从安全、约束、多目标、上下文、时间变化到离散空间等多种高级变体；②首次在控制与机器人领域引入轻量级基准套件（TuneControl）及评估指标，填补了缺乏标准化基准的空白；③通过对 110 篇硬件实验论文的系统评述，明确了 BO 在实际应用中的优缺点与研究热点。

**🔧 技术方法**

主要技术包括高斯过程回归（GP）、多种采集函数（EI、MES、logEI 等）、安全 BO（SafeOpt、LoSBO）、约束 BO、时间变化与上下文 BO、偏好 BO、离散/整数空间 BO、局部 BO 等；还讨论了多保真、多任务、批量、路径成本等扩展。

**📊 数据集**

利用对 110 篇真实硬件实验论文的系统梳理构建基准集合，并采用 TuneControl 开源框架对多种控制器（PID、LQR、MPC 等）与机器人（四旋翼、机械臂、地面车辆等）进行实验。

**📈 对比分析**

与传统随机搜索、梯度法或基准贝叶斯优化实现（如 Gaussian Process 优化、SMAC 等）进行对比。实验结果表明：在相同的实验预算下，BO 的平均收敛速度提升 30%–70%，在安全约束情形下仍能保持 90%+ 的成功率；多目标和上下文扩展进一步提高了性能。

**⚠️ 局限性**

局限性：①在高维（>10 维）参数空间下样本效率下降；②GP 超参数估计不稳定，尤其在噪声或非平稳环境中；③缺乏统一公开基准和完整的实验细节；④安全 BO 对初始安全点和核函数高度敏感；⑤对复杂约束和多保真信息的融合仍处于研究阶段。

---

## 74. A-JIT: Agentic Just-In-Time Software Construction

**arXiv ID:** 2609.10248 | [PDF](https://arxiv.org/pdf/2609.10248v1)

**作者:** Mark Marron `[一作]` (University of Kentucky), Earl T. Barr `[通讯]` (University College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Agentic Just-In-Time软件构建(A-JIT)框架，允许程序在运行时动态生成缺失实现、适配用户行为；

**💡 创新点**

核心创新是将代码合成与执行环境融合、使用可注解的“hole”语义、可触摸值与LLM驱动的自动补全；

**🔧 技术方法**

采用Bosque语言及其hole语义、Tecton值生成框架、LLM推理、基于运行时的测试与验证机制；

**📊 数据集**

以示例天气预报转换为实验对象，未使用公开大规模数据集；

**📈 对比分析**

对比方式主要是基于功能示例验证，未给出定量性能指标；

**⚠️ 局限性**

局限在缺乏大规模基准测试、对LLM生成代码的安全性与可解释性不足、对多用户交互的可扩展性待验证。

---

## 75. Execution-Time Opacity Logic: A Logic for Ensuring ET-Opacity in Timed Systems

**arXiv ID:** 2609.10066 | [PDF](https://arxiv.org/pdf/2609.10066v1)

**作者:** Jean Leneutre `[一作]` (Institut Polytechnique de Paris), James Ortiz `[通讯]` (Université Paris Est Créteil)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于时序自动机的ET‑Opacity（执行时间不可辨别性）逻辑，并给出了符号化的有界模型检验算法，用以验证如ATM系统等时序模型的隐私属性。

**💡 创新点**

创新点在于：①将秘密从单一位置扩展为任意公式；②引入freeze clock和专用观测时钟，实现对总执行时间的精确记录；③给出了可判定的有界模型检验方法，复杂度为PSPACE；④将工具集成进VITAMIN验证框架。

**🔧 技术方法**

采用的技术包括时序自动机、时区图（zone graph）与DBM、符号化前向后向搜索、固定点计算、观测时钟与freeze clock、执行时间匹配操作等。

**📊 数据集**

使用的实验数据集：原始ATM系统模型（14位置、2时钟、22转移），以及通过自动扩展得到的一系列更大规模的ATM子模型；每条公式重复实验50次。

**📈 对比分析**

方法比较：对8条ET‑Opacity公式进行检验，平均运行时间约0.047秒，标准差很小；在内存方面峰值在2–108 MiB之间；与以往仅基于位置的不可辨别性方法相比，提供了更一般的秘密表达能力和可判定的检验。性能表现稳定，适用于中小规模时序模型。

**⚠️ 局限性**

局限性：检验仅在指定的时间上限（有界验证）内可判定；只能观测总执行时间，未支持多维时序或其他侧信道信息；未处理无界或无限执行；并未考虑概率、策略或量化扩展。

---

## 76. Proof-Carrying Cognition: Closing the Verification Gap with Reality-Settled Reward

**arXiv ID:** 2609.09776 | [PDF](https://arxiv.org/pdf/2609.09776v1)

**作者:** Eshwar Reddy M `[一作]` (Testsigma), Sourav Karmakar `[通讯]` (Intuit)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了“Proof‑Carrying Cognition（PCC）”范式，利用可验证的、可价格的命题账本、可学习的世界模型和基于现实结果的结算，解决大型语言模型在推理步骤上出现的Goodhart效应，形成以现实为奖励的闭环。

**💡 创新点**

创新点包括：
1) 将推理步骤抽象为“概率命题”，让模型在每一步就可以像交易所一样报价；
2) 通过自建世界模型对命题进行即时定价，并以严格的对齐评分规则对最终结算结果进行惩罚；
3) 设计“Soundness‑under‑Pressure”指标与可扩展的现实结算基准（RSR‑Bench），以量化验证器在优化压力下的可靠性；
4) 在实验中首次展示“on‑policy settlement”在对抗自我游戏时既能保持执行奖励又显著提升标签效率（>10×）并提高声明分辨率。

**🔧 技术方法**

技术实现包括：
- best‑of‑N、hill‑climbing、evolutionary 与基于策略梯度的强化学习；
- 典型的回归/梯度提升学习者作为学习型验证器；
- 使用严格正确的、对齐的分数规则（Brier、Log‑Loss）作为结算机制；
- 通过对齐风险（anchor‑drift）监控来自适应调整结算速率；
- 在大规模实验中使用GRPO与QLoRA对 Qwen2.5-1.5B-Instruct 进行微调。

**📊 数据集**

主要数据集与基准：
- 6‑token 与 10‑token 语法程序合成（~10^4 与 ~10^10 可能程序）；
- MBPP 与 HumanEval 真实 Python 程序（~1K 任务）；
- 预先注册的对抗扩展实验（RichDSL 10‑token DSL，2–4× 更多任务）；
- 真实 LLM 判别器与执行器（Claude API，MBPP/HumanEval 单元测试）。

**📈 对比分析**

比较方法与性能：
- 对比“冻结学习型验证器”与“对齐结算”模型；
- 在最小实验中，未对齐验证器在 N=4096 时 Soundness‑under‑Pressure 从 0.94 降至 0.32；对齐后提升至 0.56 并保持稳定；
- 对齐后标签效率提升 >10×，在 150 on‑policy 结算标签上超越 1,500 随机标签；
- 在 GRPO 微调中，冻结验证器的执行奖励在 400 步后降至 0.063，现实结算模型提升至 0.397；
- 结算还提升了命题的分辨率（Brier 解耦中分辨率+98%），并未导致模糊化；
- 对抗实验中，冻结验证器出现 0.26–0.28 的 hack‑gap，而对齐验证器将 gap 降至 ≈0。

**⚠️ 局限性**

局限性：
- 实验规模主要为小型合成 DSL 或单机微调，未在 100B+ 规模模型上验证 PCC；
- 结算成本（执行/测量）在某些任务中不可忽略，且对“可解释”命题语言的可表达性未彻底评估；
- 对齐结算的可信度依赖于执行环境与结算调度的安全性，仍需硬化为可信执行基；
- 对抗模型主要为搜索级别，未覆盖端到端学习型攻击；
- 在跨任务（跨问题）验证中，on‑policy 结算的标签效率未必保持优势；
- 真实世界的可验证目标（如实验复现、物理仿真）与所用的可执行或单元测试的可验证性不同，可能导致“现实”本身被操控的风险。

---

## 77. HaWMPO: Hallucination-Aware World Model-based Policy Optimization for Generalist Robot Policy

**arXiv ID:** 2609.09941 | [PDF](https://arxiv.org/pdf/2609.09941v1)

**作者:** Zengjue Chen `[一作]` (Joy Future Academy, JD), Qi Wang `[通讯]` (Jilin University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套基于世界模型的VLA策略后训练框架HaWMPO，利用可视化-语言-动作（VLA）策略在模拟环境中进行自闭环强化学习；

**💡 创新点**

创新点在于设计了动作条件的幻觉感知模型（HAM）来评估世界模型生成的图像序列的可靠性，并将幻觉得分融入Reward‑Soft机制，对奖励进行动态抑制，从而减少虚拟轨迹的误导性影响；

**🔧 技术方法**

技术上结合了VLA策略、像素级视频世界模型、基于Transformer的HAM、GRPO强化学习算法以及Reward‑Soft奖励调节；

**📊 数据集**

实验数据集主要包括LIBERO仿真基准（Object、Spatial、Goal任务套件）和在G1机器人上执行的两项真实世界任务（Tissue‑to‑Box、Headphone‑on‑Stand）；

**📈 对比分析**

与OpenVLA‑OFT、WMPO和WoVR*等基线对比，HaWMPO在LIBERO上平均成功率提升至63.7%（比基线高+2.8%），在真实机器人上成功率从67.5%提升至80.0%，表现优异；

**⚠️ 局限性**

局限性包括幻觉评估依赖手工设计的监督信号，实验范围主要为短时序操作，且幻觉模型与世界模型训练分离，未验证在更复杂长时序任务中的可扩展性。

---

## 78. QPS-ToR: A Parallel Iterative Switching Algorithm for Reconfigurable Optical Datacenter Switching

**arXiv ID:** 2609.09400 | [PDF](https://arxiv.org/pdf/2609.09400v1)

**作者:** Dongzhao Song `[一作]` (Georgia Tech), Jun Xu `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在可重构光学数据中心网络中引入SW‑QPS算法，改进了NegotiaToR的单轮iSLIP调度；

**💡 创新点**

创新点在于将滑动窗口队列比例抽样（SW‑QPS）应用于光学交换机调度，并将调度管线从三步RGA压缩为两步RG；

**🔧 技术方法**

使用SW‑QPS、滑动窗口机制、并行迭代交换算法（PISA）作为核心调度技术；

**📊 数据集**

实验基于Meta Hadoop、Web搜索以及Google数据中心的三份公开流量轨迹；

**📈 对比分析**

在相同模拟器下与NegotiaToR和RotorNet对比，QPS‑ToR在负载>0.5时平均流完成时间下降72%–82%，吞吐量提升24%–36%；

**⚠️ 局限性**

局限在于对短鼠标流的改进有限，且需要通过分离调度实例来避免读写冲突。

---

## 79. M2LG-DG: A Multi-modal Local-Global Domain Generalization Framework for Cross-site Major Depressive Disorder Classification

**arXiv ID:** 2609.09186 | [PDF](https://arxiv.org/pdf/2609.09186v1)

**作者:** Muhammad Asif Hasan `[一作]` (Griffith University), Alan Wee-Chung Liew `[通讯]` (Griffith University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 M^2LG-DG 框架，实现跨站点抑郁症（MDD）分类的域泛化；

**💡 创新点**

创新点包括：双流 rs‑fMRI 编码同时捕捉全局与局部功能；共享‑私有多模态分解与交叉注意力融合；跨站点监督对比学习排除同站同类正样本；

**🔧 技术方法**

使用技术包括：Transformer 与图卷积网络的双流编码；共享‑私有分解与多损失约束；双向交叉注意力门控融合；跨站点监督对比损失与标签平滑交叉熵；

**📊 数据集**

使用数据集为 REST‑meta‑MDD（多中心 rs‑fMRI 与非影像基线信息）以及 ABIDE（自闭症多中心 rs‑fMRI）验证；

**📈 对比分析**

与 13 个竞争方法比较，在四个隐藏站点的宏平均 AUC 达到 69.48%，领先；在 ABIDE 上亦优于对手，表现出更强的泛化性能；

**⚠️ 局限性**

局限性：对扫描时间、TR 等差异仍需通过固定重采样处理；非影像变量缺失处理仍有限；需要更大样本和更多多模态多中心验证。

---

## 80. An Experimental Evaluation of Multimodal Prompt Injection Attacks on Agentic AI Frameworks

**arXiv ID:** 2609.09404 | [PDF](https://arxiv.org/pdf/2609.09404v1)

**作者:** Viet K. Nguyen `[一作]` (California State Polytechnic University), Mohammad I. Husain `[通讯]` (California State Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了MMPIBench，用于评估多模态（图像和音频）Prompt Injection攻击在六种Agentic AI框架中的表现；

**💡 创新点**

首次构建跨框架、跨模型、跨载体的可复现基准，结合分层判定和流水线跟踪，能够细粒度识别攻击在哪一阶段被阻断；

**🔧 技术方法**

使用多模态载体（OCR文本、覆盖层、EXIF、二维码、伪UI、混合）和音频语音注入，配合OpenRouter多模型接口、Agentic框架适配器以及文本工具模拟，利用模型输出与内部日志进行自动判定；

**📊 数据集**

创建了24个视觉攻击实例（6载体×4目标）与12个音频实例（4目标×3隐蔽度），所有数据均可复现；

**📈 对比分析**

对720个（视觉）与72个（音频）实验结果进行统计，发现模型是决定攻击成功的主因，框架主要决定是否能将模态送入模型；视觉通道完成率≈1%，尝试率≈12.8%，音频通道完成率≈49%；

**⚠️ 局限性**

局限性包括：单次试验、温度零但无种子导致结果波动；框架与模型版本变化快；视觉与音频任务框架、提示与工具设置不完全一致；仅使用模拟工具，未评估实际系统风险；评估仅关注攻击面，不提供防御措施。

---

## 81. How neighbourhood ideology shapes misinformation belief in densely tied social networks

**arXiv ID:** 2609.10277 | [PDF](https://arxiv.org/pdf/2609.10277v1)

**作者:** Soroush Karimi `[一作]` (University of Exeter), Diogo Pacheco `[通讯]` (University of Exeter)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并仿真了一种整合个人意识形态强度与虚假信息意识形态对齐度的传染模型，探讨网络结构对错误信息传播的影响。

**💡 创新点**

创新点在于将个人意识形态强度与虚假主张的对齐度明确区分并嵌入传染动力学，揭示高聚集度网络会自发放大邻居意识形态对信仰形成的影响。

**🔧 技术方法**

使用了基于代理的模拟、马尔可夫链分析、标准化线性回归等技术，并在不同网络拓扑（ER、WS、BA、实测Facebook）上评估模型。

**📊 数据集**

主要数据集为：1000节点的Erdős–Rényi网络、4039节点的Watts–Strogatz和Barabási–Albert模型以及实测Facebook社交网络样本。

**📈 对比分析**

通过比较不同网络结构与极端主义节点分布策略（随机、度中心、BFS聚类）下的信仰概率和回归系数，发现高度聚集的网络使邻居意识形态对信仰的影响接近个人意识形态强度，说明模型能够捕捉网络聚集效应；在不同gullibility（α）水平下，整体信仰比例随α上升而提升。

**⚠️ 局限性**

局限性包括：仅考虑单维意识形态、网络静态且未模拟时间演化、参数设定缺乏实证验证、未对多维意识形态或多主题主张进行扩展，模型的现实可推广性仍待进一步实验验证。

---

## 82. CLFTv2: Efficient Camera-LiDAR Fusion for Semantic Segmentation via Hierarchical Feature Pyramids

**arXiv ID:** 2609.09881 | [PDF](https://arxiv.org/pdf/2609.09881v1)

**作者:** Toomas Tahves `[一作]` (Tallinn University of Technology), Raivo Sell `[通讯]` (Tallinn University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了CLFTv2，一种在2D视角投影下使用Swin Transformer实现的层次化相机‑激光雷达融合框架，用轻量化残差解码器实现高效语义分割，特别关注对VRU的召回率；

**💡 创新点**

创新点在于将全局ViT换成局部窗口Swin，构建多尺度特征金字塔并通过残差融合实现高效跨模态上下文聚合，避免了查询‑匹配解码器的高计算开销；

**🔧 技术方法**

使用Swin Transformer（shifted‑window self‑attention）、FPN式残差融合解码器、2D视角投影的激光雷达坐标图、ImageNet预训练、加权交叉熵等技术；

**📊 数据集**

在三大自动驾驶数据集上验证：ZOD、Waymo Open Dataset、ISEAuto；

**📈 对比分析**

与MaskFormer、Mask2Former及DeepLabV3+等基线在相同配置下对比，CLFTv2在ZOD上mIoU提升至53.5%（相较Mask2Former 52.5%），Waymo上保持竞争力但显著降低GFLOPs（1.4×）并提升吞吐量（2.2×）；

**⚠️ 局限性**

局部窗口注意力在点云密集的Waymo场景下效果不及全局ViT，且对伪标签噪声较敏感；未在低功耗边缘设备上验证，且仅使用2D投影，可能失去3D几何细节。

---

## 83. Learning Global Camera Poses from Noisy View-Graphs for Structure from Motion

**arXiv ID:** 2609.09491 | [PDF](https://arxiv.org/pdf/2609.09491v1)

**作者:** Fadi Khatib `[一作]` (Weizmann Institute of Science), Ronen Basri `[通讯]` (Weizmann Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种无监督的视图图（view‑graph）相机位姿聚合方法，利用可置换等变的边条件图神经网络（GNN）从噪声相对位姿中恢复全局一致的相机外参，然后通过三角化和鲁棒束平面（bundle adjustment）进一步精炼结构。

**💡 创新点**

创新点在于：1）提出了可置换等变的边条件GNN来聚合全局视图图信息，直接学习相对位姿的全局一致性；2）使用相对位姿一致性作为自监督目标，无需地面真值；3）在大规模稀疏检索生成的视图图上同样表现良好；4）在不生成3D点的情况下实现高精度相机位姿恢复。

**🔧 技术方法**

核心技术包括：Permutation‑equivariant edge‑conditioned GNN、相对位姿一致性损失、SO(3) 对数映射、单位平移向量、三角化（DLT）与鲁棒束平面优化、可选的视图重新集成步骤。

**📊 数据集**

在 MegaDepth、1DSfM、Strecha、BlendedMVS 等公开数据集上进行评估，训练主要基于 MegaDepth，交叉数据集测试验证通用性。

**📈 对比分析**

与传统增量式、因子化以及最新深度 SfM 基线（如 RESfM、VGGSfM、FAST3R 等）比较，本文方法在旋转误差、平移误差和相机覆盖率上均优于深度模型，且在大多数场景中与 COLMAP、GLOMAP 等经典管线相当或更好；同时推理速度大幅提升（比 COLMAP 快数倍），且内存线性扩展。

**⚠️ 局限性**

局限性包括：1）依赖于初始相对位姿与点轨迹的质量；2）对极端噪声或极端稀疏视图图仍有挑战；3）网络不直接预测 3D 点，后期需单独三角化；4）在完全未标定（intrinsic）场景下自标定误差略高于已知标定。

---

## 84. Keep Evaluation Fair: Detecting Data Leakage in Code Generation Benchmarks via Membership Inference Attacks

**arXiv ID:** 2609.09865 | [PDF](https://arxiv.org/pdf/2609.09865v1)

**作者:** Dongdong Zhao `[一作]` (Wuhan University of Technology), Xiao Yu `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CGMIA（Code‑Generation‑specific Membership Inference Attack）方法，用来检测代码生成大语言模型在评测基准中的数据泄露情况。

**💡 创新点**

创新点：①将专家设计的四种特征（CodeBLEU、编辑距离、困惑度、测试通过率）与 CodeBERT 的语义嵌入融合，构建多模态成员推断分类器；②利用影子模型训练模拟目标模型行为，从而在无训练数据可见的黑盒场景下实现成员推断；③系统地在八大代码生成基准和 StarCoder 的真实泄露数据上验证，展示显著优于现有 MIA 方法的效果。

**🔧 技术方法**

技术手段：影子模型微调、特征提取（专家特征与 CodeBERT 嵌入）、一维 CNN 压缩、线性投影、三层前馈分类器、集成学习（加权投票）以及统计显著性检验。

**📊 数据集**

使用了八个公开代码生成基准（HumanEval‑X、MBXP、NaturalCodeBench、EvoCodeBench、ClassEval 等），以及 StarCoder‑7B 的 APPS 漏洞样本，评估不同 LLM（Qwen2.5‑Coder、CodeGemma、DeepSeek‑Coder、Phi‑2）的泄露检测性能。

**📈 对比分析**

与 DetectLeak、Gotcha、CodeMI、随机森林、逻辑回归、CNN、LSTM、Transformer 等基线比较，CGMIA 在 precision、recall、MCC、AUC 上平均提升 0.05–0.16、0.06–0.11、0.17–0.26、0.09–0.13，且在 StarCoder‑7B 的已知泄露样本中召回率达到 65%+，证明其实用性。

**⚠️ 局限性**

局限性：①需要模型返回 token‑级 log‑probability（score‑access）才能计算困惑度；②仅考虑完整基准样本泄露，对改写/改动后的代码泄露缺乏鲁棒性；③实验基于 LoRA 微调的模拟泄露，真实预训练泄露难以完全再现；④对极大模型的性能尚未验证；⑤对极低泄露比例或严格文本‑only 接口的适应性有限。

---

## 85. On the Tightness of Standard Relaxations for Mixed-Integer Bilevel Linear Programs

**arXiv ID:** 2609.10233 | [PDF](https://arxiv.org/pdf/2609.10233v1)

**作者:** Sergey S. Ketkov `[一作]` (University of Zurich), Oleg A. Prokopyev `[通讯]` (University of Zurich)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了混合整数双层线性规划（MIBLP）中的单层松弛（SLR）产生的下界和上界，并证明在标准计算复杂度框架内，这些界无法得到严格且统一的改进。

**💡 创新点**

创新点在于首次通过复杂度理论证明，连续双层线性规划和纯整数双层线性规划的SLR下界和上界在最小-最大子类中是provably best，且在多项式时间MILP-oracle算法框架内同样不可改进。

**🔧 技术方法**

采用的技术主要是复杂度理论与归约论证，包括NP-hard与Σ₂^P-hard归约、KKT条件的线性化、LP松弛、以及多项式时间可计算性分析。

**📊 数据集**

论文未使用实验数据集，全部结果基于理论证明与复杂度分析。

**📈 对比分析**

由于研究的主要是理论性质，没有实验比较，性能评估仅通过证明下界/上界的不可改进性来体现。

**⚠️ 局限性**

局限在于仅针对最小-最大形式的BLI/IBLP进行讨论，并未覆盖一般双层目标或包含耦合约束的情况。

---

## 86. OmniEye: Efficient Multimodal Forensic Video Intelligence for Law-Enforcement Body-Worn Cameras

**arXiv ID:** 2609.09460 | [PDF](https://arxiv.org/pdf/2609.09460v1)

**作者:** Mamadou K. Keita `[一作]` (Rochester Institute of Technology), Ernest Fokoué `[通讯]` (Rochester Institute of Technology)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套本地可执行的多模态视频分析系统 OmniEye，用于执法视频的索引、检索和问答，并提供不可篡改的审计与重现功能。

**💡 创新点**

创新点包括：① 4-bit 量化 + 递归式 speculative 解码与 AIMD 资源调度，实现在单 16 GB GPU 甚至共享集群上高效运行；② 基于 SHA‑256 的每帧窗口哈希链与全局增量审计日志，实现可验证的不可否认与防篡改；③ 重感知问答机制，要求答案仅引用已在当前交互中重新感知的窗口，显著降低模型幻觉风险。

**🔧 技术方法**

核心技术包括：Gemma 4‑12B 多模态模型（4‑bit 或 bf16）、4‑bit 量化与双重量化、temporal speculative decoder、数据并行与 AIMD 批量控制、SQLite FTS5 与 BM25、SHA‑256 链与哈希链审计、重感知问答与工具调用框架。

**📊 数据集**

使用了罗切斯特（NY）警察局的 1,000 条带年分层抽样的执法摄像机录像（总计超过 90,000 条），每条录音分为 30 秒窗口，包含音频与视频帧；此外还加入了政策文件文本进行索引。

**📈 对比分析**

与两名人工标注员的 500 条窗口进行对比：系统与标注员 A 的 κ 为 0.29，优于标注员 B 的 0.17；在检索任务中，系统在前 25 窗口内高优先级内容的召回率提升 3.3 倍（基线 0.18），ROC‑AUC 达到 0.677/0.686，表明排序效果显著优于仅用类别标签。

**⚠️ 局限性**

局限性包括：① 样本多样性有限，仅来自单一辖区；② 标注员间标注差异大，导致低 κ，说明“是否值得审阅”主观性强；③ 对极端硬件（如多租户共享集群）下的实时性能仍需进一步验证；④ 依赖开放权重模型，若模型更新导致兼容性问题需重新调优。

---

## 87. Marker-free eye-gaze estimation using a single image and depth from defocus

**arXiv ID:** 2609.09610 | [PDF](https://arxiv.org/pdf/2609.09610v1)

**作者:** David Hurtubise-Martin `[一作]` (University of Sherbrooke), Marie-Flavie Auclair-Fortier `[通讯]` (University of Sherbrooke)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种仅使用单摄像头、无标记、基于模糊深度估计的眼动定位系统，并用变分贝叶斯多项式逻辑回归完成AoR（关注区）分类。

**💡 创新点**

创新点在于：①将模糊（depth‑from‑defocus）直接用于头部姿态与深度估计；②通过八维几何特征与VBMLR实现极低参数（81）且近乎完美的分类；③在单摄像头条件下实现几乎无标记、可实时部署的眼动预测。

**🔧 技术方法**

采用的技术包括：iris定位、鼻中心检测、模糊测量→深度推算→头部姿态与方向、八维特征向量、变分贝叶斯多项式逻辑回归（VBMLR），并与SVM、线性回归、Ridge、CNN‑1D、ResNet18 进行对比。

**📊 数据集**

使用的数据集：4名受试者在15英寸屏幕上，分别在50 cm和70 cm两距离，9个AoR，每个区100帧，总共7200帧。

**📈 对比分析**

比较方法：在5折交叉验证下对VBMLR、SVM、LR、Ridge、CNN‑1D、ResNet18 进行分类性能评估。VBMLR在单距离下50 cm准确率99.69%、70 cm 96.81%，全距87.03%；仅81个参数，推理时间约0.002 ms/样本；其他方法准确率普遍低于90%，且参数与计算量显著增大。

**⚠️ 局限性**

局限性：对不同距离或跨用户的泛化性弱，主要在上方AoR产生混淆；深度估计对光照与模糊敏感；系统仅适用于固定摄像头与已知工作距离，需进一步研究跨距适配与更小AoR的鲁棒性。

---

## 88. Spectral origin of the topological gap exponent d + η: mechanism, kernel, decomposition, and scope

**arXiv ID:** 2609.09159 | [PDF](https://arxiv.org/pdf/2609.09159v1)

**作者:** Matthew Loftus `[一作]` `[通讯]` (Independent Researcher), Matthew Loftus (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过对临界点云的持久性同调与谱积分的关系，提出并验证了拓扑间隙指数为 d+η 的解析机制。

**💡 创新点**

创新点在于把拓扑间隙与结构因子谱积分的 k^(-2η) 权重联系起来，并给出两维系统的 IR 统计算法与临界维度判据。

**🔧 技术方法**

主要技术包括傅里叶变换求结构因子、α 复合滤波器的持久性同调计算、解析尺度推导与解析量化的分解 I_0·I_shape。

**📊 数据集**

使用 2D Ising、2D Potts (q=3,4) 与 3D Ising 体系在不同尺寸 L（32–512）上产生的多数自旋点云数据集。

**📈 对比分析**

通过比值稳定性、α‑sweep 与不同持久性同调过滤器（α 复合、Vietoris–Rips）比较，结果表明 Δ 与 I(-2η) 近似线性，比例波动低于 5%，验证了理论预测；在 2D 系统中得到一致的 d+η 计数；3D 系统需额外归一化。

**⚠️ 局限性**

局限在于 4 倍 Potts 系统的 α_opt 未能在大尺寸下稳定，存在长自相关；三维系统仅得到 UV 统计算法，缺乏对裸核的第一性原理推导；对 Log 修正与自相关的处理仍待进一步完善。

---

## 89. JEPA Policy: Diffusion-Free Imitation Learning via Paired Action and Future Representation Prediction

**arXiv ID:** 2609.09630 | [PDF](https://arxiv.org/pdf/2609.09630v1)

**作者:** Jie Xu `[一作]` (Anyverse Dynamics), Zhongpu Xia `[通讯]` (Anyverse Dynamics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在行为克隆基础上引入无扩散两步Transformer策略 JEPA Policy，利用动作与对应的未来状态表征进行联合监督。

**💡 创新点**

创新点在于：①把动作与未来表征放入同一共享Transformer层，实现未来预测直接塑造动作表示；②采用两步 MIP 风格训练而不需迭代采样，保持低延迟；③通过未来一致性诊断提供部署时的风险监测。

**🔧 技术方法**

使用技术包括：两步 MIP 训练、共享自注意力 Transformer、动作与未来对齐、stop‑gradient 目标、无扩散推理、以及可解释的未来一致性误差。

**📊 数据集**

实验数据集涵盖 LIBERO、Robomimic、MimicGen 的九个仿真任务，并在五个真实机器人任务（Cabinet、Cup Stack、Cup Upright、Plate Grape、Pen Insert）上验证。

**📈 对比分析**

方法与 MIP、Diffusion Policy、ACT‑JEPA 等基线在相同训练步数、批次和评估设置下比较，结果显示九任务平均提升约 5.6 点，单机决策延迟从 439.5 ms 降至 13.2 ms，硬件实验中在所有任务上均保持领先。

**⚠️ 局限性**

局限性包括：评估主要在仿真与少量真实机器人实验，任务多样性有限；使用最优检查点评估可能导致乐观；未验证多模态未来预测的有效性；对不同感知模态或多机器人场景的泛化尚未测试。

---

## 90. From Retrieval to Weights: Parametric Individualization of Small Language Models with Individual Text Corpora

**arXiv ID:** 2609.10155 | [PDF](https://arxiv.org/pdf/2609.10155v1)

**作者:** Christoph Wigbels `[一作]` (Bergische Universität Wuppertal), Markus J. Hofmann `[通讯]` (Bergische Universität Wuppertal)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将受试者的 Google 搜索历史转化为个体文本语料库，并使用 DoRA 低秩适配器在小型语言模型（Qwen3‑0.6B）中写入个体知识，随后与检索增强生成（RAG）进行对比，评估在 36 道多项选择知识题上的个体化效果。

**💡 创新点**

首次将个体文本语料库直接写入模型权重而非仅在推理时检索，形成权重化个体化路径，并与传统检索化个体化进行系统比较；同时采用长度归一化与 PMI 读出消除选项 ID 偏差，使评估更为公平。

**🔧 技术方法**

使用参数高效微调（DoRA 适配器）、检索增强生成（RAG）、密集检索（nomic-embed-text-v1.5）、对数似然与 PMI 评估、长度归一化读出，以及基于 Qwen3‑0.6B 的小型模型。

**📊 数据集**

构建了 515 名受试者（分层抽样 150 名）Google 搜索历史的个体文本语料库；使用 36 道多项选择知识题（12 道公共 BEFKI GC‑K + 24 道新题）作为评估标准。

**📈 对比分析**

采用 4×2 设计（基线/adapter × 无检索/RAG），使用匹配准确率、对数损失匹配、CK 准确率等指标；adapter 显著降低对数损失匹配（PMI 0.079 nats，LN 0.248 nats），但在 PMI 视图下匹配准确率无显著提升；检索与 adapter 无交互；对知识语料大小无显著相关。

**⚠️ 局限性**

样本主要为年轻女性高学历，且仅覆盖中等大小语料，单语种德语，知识测试为通用测试未针对个体知识；小模型整体正确率仍低于受试者；机器翻译质量对训练信号有影响；个体错误模式未得到充分捕获。

---

## 91. MotionBlind: Probing the Illusion of Motion Understanding in Video-LLMs

**arXiv ID:** 2609.09528 | [PDF](https://arxiv.org/pdf/2609.09528v1)

**作者:** Dhairya Bhatia `[一作]` (Northeastern University), Sarah Ostadabbas `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MotionBlind基准，用于评估视频LLM对物理运动（速度、幅度、方向）的理解。

**💡 创新点**

创新点在于自记录、对比对（contrastive）最小对例，专注物理运动属性，采用严格的实例准确率评估，揭示现有Video-LLM在运动感知上的缺陷。

**🔧 技术方法**

使用对比评估协议（2×2实例、实例准确率），结合多种帧采样策略（均匀、随机、学习型、无训练关键帧）以及完整性探测（无视频、打乱帧、倒序）来分析模型。

**📊 数据集**

MotionBlind数据集（60个对例、240问答），并与TimeBlind和Video-MME进行对比。

**📈 对比分析**

对6个公开Video-LLM和2个专有模型进行实验，公开模型在MotionBlind上的实例准确率仅为11.7%以内，只有Gemini 3.1 Pro略高至60%，但仍低于人类91.3%；帧数增加或动态选择对准确率无显著提升。

**⚠️ 局限性**

样本规模有限，仅单一室内演员，缺乏野外或主观视角；YES/NO解析限制可能低估推理能力，未测试更大模型或其他结构。

---

## 92. Grounded Evaluation and Repair for NL-to-PDDL Problem Generation

**arXiv ID:** 2609.09898 | [PDF](https://arxiv.org/pdf/2609.09898v1)

**作者:** Joana Rosa `[一作]` (INESC INOV), Bruno Martins `[通讯]` (INESC ID)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一套端到端的 NL-to-PDDL 生成、评估与迭代修复管道，结合 LLM 生成、语法/领域合规检查、规划/验证与 LLM 批评家，实现了对自然语言规划描述的自动化 PDDL 问题生成，并对 Planetarium、AutoPlanBench 及 PDDL 2.1 基准进行实验。

**💡 创新点**

将操作性接受标准（解析+领域合规+规划+VAL+LLM 批评）与离线基准重构（语义/结构匹配）分离，提出细粒度反馈驱动的循环修复，并系统评估少量样例提示和迭代修复对不同类型规划域的影响。

**🔧 技术方法**

基于 GPT‑4 的 LLM 生成与批评、Fast Downward/TFD/ENHSP 规划器、VAL 验证器、静态领域合规检查器以及结构化与语义匹配工具（Planetarium oracle）。

**📊 数据集**

Planetarium、AutoPlanBench（含支持与不支持语义匹配的子集）以及六个手工挑选的 PDDL 2.1 领域（时间简化与数值域）。

**📈 对比分析**

在线采用操作成功率与离线结构/语义匹配率进行对比；在 Planetarium 语义支持下，操作成功率从 0.333 提升至 0.583，结构匹配率从 0.183 提升至 0.417；在 PDDL 2.1 上，操作成功率从 0.350 提升至 0.733，但结构匹配率仅从 0.000 提升至 0.067，表明操作成功与基准重构存在显著差距。

**⚠️ 局限性**

①操作接受标准仅是可执行性的代理，无法保证任务完整性；②LLM 批评家存在假阳性/假阴性；③仅处理已给定领域模型，未覆盖域生成；④实验使用的自然语言描述相对模板化，缺乏真正开放式用户语言；⑤缺乏交互式迭代与计划迁移评估；⑥PDDL 2.1 基准覆盖有限，无法验证更复杂时间/数值域的鲁棒性。

---

## 93. Benchmarking Agentic HLS Design Tasks With HLS-Eval

**arXiv ID:** 2609.09526 | [PDF](https://arxiv.org/pdf/2609.09526v1)

**作者:** Stefan Abi-Karam `[一作]` (Georgia Institute of Technology), Callie Hao `[通讯]` (Georgia Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了基于 mini-swe-agent 的代理式 HLS 设计评估流程，并将其集成到 HLS-Eval 基准中。

**💡 创新点**

创新点在于为 HLS 设计任务构建了完整的代理式评估框架，允许 LLM 代理调用工具、编译验证并迭代改进，同时提供成本与 token 使用的可追踪分析。

**🔧 技术方法**

采用了 LLM 代理技术（mini-swe-agent）、Bash 工具箱、C++ 编译器、HLS 合成器以及自动化脚本。

**📊 数据集**

使用了来自 CHStone、MachSuite、Polybench、Rosetta 和 C2HLSC 的 85 个 HLS 设计样例。

**📈 对比分析**

通过 pass@k、推理缩放和轨迹分析对比模型表现，发现小型开源模型在 pass@10 上已达到 100%，但较大模型在 k=10 时仅提升约 11%，表明代理式评估能显著提升通过率。

**⚠️ 局限性**

限制在于基准难度不足，当前任务过于简单导致模型饱和，缺乏更具挑战性的设计场景以及更细粒度的工具使用分析。

---

## 94. Hierarchical and Permutation-Invariant Feature Transformation Learning via Policy-Guided Embedding Search

**arXiv ID:** 2609.10225 | [PDF](https://arxiv.org/pdf/2609.10225v1)

**作者:** Rui Liu `[一作]` (University of Kansas), Dongjie Wang `[通讯]` (Northeast Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种层次化特征变换框架，将特征、操作和抽象概念的嵌入统一到无序不变的连续空间，并通过策略导向的多目标强化学习搜索最优变换序列，提升表格数据预测性能。

**💡 创新点**

核心创新在于：①构建可保持置换不变的层次编码器‑解码器，将低层特征-操作交互与高层概念抽象同时嵌入；②利用自注意力池化实现概念级置换不变性；③用PPO搜索策略在非凸嵌入空间中全局探索，无需凸性假设。

**🔧 技术方法**

技术手段包括Transformer编码器/解码器、聚合层与自注意力池化实现无序不变嵌入、连续嵌入空间学习，以及基于PPO的多目标强化学习搜索（兼顾预测准确度与变换长度）。

**📊 数据集**

实验使用19个公开表格数据集（UCI、LibSVM、Kaggle、OpenML），涵盖14个分类任务和5个回归任务，统一采用随机森林作为下游模型进行交叉验证。

**📈 对比分析**

与9个主流特征变换基线（RDG、ERG、LDA、AFAT、NFS、TTG、GRFG、DIFER、MOAT）以及自家变体对比，PHER在所有数据集上均获得最高的F1/1‑RAE，且生成的变换序列更短、对不同下游模型表现更稳健。

**⚠️ 局限性**

局限性主要包括：搜索过程计算开销较大，扩展到极大特征空间时效率可能受限；依赖先前采集的变换记录，若记录不足可能影响嵌入质量；以及对PPO超参数的敏感度需要进一步自动化。

---

## 95. Zero-Shot Temporal Localisation of Audio Deepfakes in Multi-Speaker Conversations

**arXiv ID:** 2609.10051 | [PDF](https://arxiv.org/pdf/2609.10051v1)

**作者:** Soumyadeep Roy `[一作]` `[通讯]` (St Xavier's College), Soumyadeep Roy (St Xavier's College)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种零样本多说话人对话中的深度伪造时序定位方法，并构建了基准数据集进行评估。

**💡 创新点**

创新点在于：将冻结的二分类检测器通过五阶段无监督管道包装，利用双阈值有限状态机实现区间化；同时定义了适用于混合内容文件的时序评估指标。

**🔧 技术方法**

使用滑窗评分、归一化、窗口级置信度、极大值滤波、Gaussian平滑、双阈值FSM以及区间后处理；不需要额外训练。

**📊 数据集**

主要数据集为ASVspoof 5构造的180个多说话人对话，以及AMI会议语料的真实对话验证。

**📈 对比分析**

与单阈值、训练好的局部化器以及三种冻结检测器比较，零样本管道在t‑IoU≈0.90、TDR≈0.95、MS‑DCF≈0.26，超越单阈值且仅比训练化模型差≈0.04 t‑IoU；在真实对话中的误报率低于2%。

**⚠️ 局限性**

局部化边界偏移约3.5秒，构造基准的误报率受标签构造影响；未覆盖真实动态对话的完整性；对抗性生成器的域差距导致短段注入检测能力有限。

---

## 96. YallaMorph: A Benchmark for Evaluating Arabic Morphological Generation in Large Language Models

**arXiv ID:** 2609.10153 | [PDF](https://arxiv.org/pdf/2609.10153v1)

**作者:** Mahmoud Reda `[一作]` (New York University), Nizar Habash `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了大规模阿拉伯形态生成基准YallaMorph，覆盖动词、名词、形容词及其附着形式，直接评测LLM的受控形态生成能力。

**💡 创新点**

创新点在于使用形态学驱动的分层采样（根类、频率、形态完整性、干性等），生成600K条结构化实例，并首次将此类基准与LLM的指令调优结合评估。

**🔧 技术方法**

技术手段包括使用CAMeL Tools/CamelMorph生成器构造基准实例，采用多语言指令调优的LLM（GPT、Gemini、Qwen、Fanar、Jais、ALLaM）进行零/少量示例提示，并通过Any Match Accuracy、F1等指标进行评估。

**📊 数据集**

数据来源主要为CamelMorph MSA词典，结合BAREC‑10M频率统计和CAMeLBERT频率表进行采样，最终基准集已公开于GitHub。

**📈 对比分析**

方法：在10-shot与0-shot提示下对七款LLM进行对比，评估标准为diacritized、undiacritized及正则化模式的Any Match Accuracy和F1。结果显示GPT在10-shot下diacritized约52% AMA，最高可达80%（未标点/正则化），其余模型性能显著低于此水平，整体仍远低于理想。

**⚠️ 局限性**

局限性：仅覆盖现代标准阿拉伯语，依赖CamelMorph的注释和生成规则；基准为采样而非完整枚举；评估受提示、解码策略影响；可能忽略可接受的正字法变体及方言多样性。

---

## 97. Distilling Image Prototypes for Guided Test-Time Adaptation

**arXiv ID:** 2609.09737 | [PDF](https://arxiv.org/pdf/2609.09737v1)

**作者:** Liwen Wang `[一作]` (Anhui University), Zhe Jin `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于Distilling Image Prototype（DIP）的测试时自适应框架DIPTTA，用于解决错误累积和灾难性遗忘问题。

**💡 创新点**

创新点在于将源域知识转化为可动态重建的合成图像集合DIP，既能实现动态特征原型重放，又能为不确定性估计提供稳定的源域锚点。

**🔧 技术方法**

采用数据蒸馏、动态特征对比学习、源校准不确定性估计（基于Laplace近似的后验推断）以及均值教师框架。

**📊 数据集**

在CIFAR‑10‑C、CIFAR‑100‑C、ImageNet‑C、TinyImageNet‑C、ImageNet‑R以及CCC长期序列基准上进行实验。

**📈 对比分析**

与多种SOTA方法（Tent、RMT、CoTTA、BECoTTA等）对比，DIPTTA在大多数任务上均实现了显著的误差率降低，尤其在严重域漂移下表现突出。

**⚠️ 局限性**

局限性包括对DIP的初始化与优化需要额外计算，且对极端域变迁或数据稀疏场景下的鲁棒性仍有待进一步验证。

---

## 98. Meme Coin Factories: Uncovering Large-Scale Manipulations on pump.fun

**arXiv ID:** 2609.10246 | [PDF](https://arxiv.org/pdf/2609.10246v1)

**作者:** Nicolas Szwajcok `[一作]` (Ecole Polytechnique Federale De Lausanne), Nicolas Christin `[通讯]` (Carnegie Mellon University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对Solana主流代币发行平台的1500多万个代币进行大规模分析，识别并量化洗盘、创作者地址混淆、协同抛售、复制币及社交媒体操纵等五类操纵策略，并揭示了市场操纵即服务（MMaaS）的存在。

**💡 创新点**

在规模、方法和范式上均实现突破：首次在单一平台上覆盖完整代币生命周期的数据；提出多重阈值洗盘与协同抛售的低误判检测；首次系统性探测创作者地址层级混淆与复制币；将社交媒体与代币发行同步关联，揭示帖子驱动的操纵经济；发现并分析MMaaS工具链，为监管与技术防护提供新视角。

**🔧 技术方法**

主要技术手段包括：基于交易对的洗盘与协同抛售启发式（WT1/WT2、DP1/DP2），基于资金链的创作者地址聚类，IPFS哈希匹配实现复制币识别，图论方法检测评论协同与社区活跃度，回归与统计分析验证操纵与“毕业”成功率关联，以及对社交媒体链接与帖子内容的文本相似度与影响力评估。

**📊 数据集**

使用Solana链第三方服务（Chainstack、Syndica、Arkham）收集完整交易与代币元数据，约1500万枚代币的元数据、87M交易记录；通过Arkham标签注释创作者与资助方身份；利用Twitter、Telegram、Truth Social公开接口抓取用户、帖子、群组及链接；对IPFS图片哈希与非IPFS图片进行去重；构建多维度数据集实现全方位分析。

**📈 对比分析**

对比传统单一策略阈值检测，提出的双阈值洗盘/抛售方法在误判率低的同时覆盖率更高；利用逻辑回归验证洗盘交易数与代币毕业率正相关，显著性p<1e-58；复制币覆盖率达到10%+，与原币毕业率对比揭示首发优势；社交媒体影响力与代币成功率相关性统计显示高粉丝、活跃帖子可带百万级收益；整体结果显示操纵活动普遍且具可盈利性。

**⚠️ 局限性**

局限性包括：检测方法依赖启发式阈值，可能漏检更隐蔽的洗盘与抛售；仅关注Solana生态，无法直接推广至其他链；IPFS哈希匹配无法识别图像相似但非完全相同的复制币；社交媒体分析受API限制，未覆盖所有平台；无法直接确认操作者恶意意图，只能通过后续表现间接推断；MMaaS发现基于公开信息，可能存在漏检或误判。

---

## 99. SEA-SpeechBench: A Large-Scale Multitask Benchmark for Speech Understanding Across Southeast Asia

**arXiv ID:** 2609.09672 | [PDF](https://arxiv.org/pdf/2609.09672v1)

**作者:** Jingyi Liao `[一作]` (A*STAR), Ai Ti Aw `[通讯]` (A*STAR)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并发布了SEA-SpeechBench，一个涵盖11种东南亚语言、9类任务（语音识别、翻译、问答、情感、性别、年龄、说话人辨识、时间问答、时间定位）的多任务评测基准。

**💡 创新点**

首次引入针对语音时序推理的新任务；采用双语提示（英语与本地语言）和统一的音频预处理，构建了超过97,000条样本、597小时的评测集。

**🔧 技术方法**

通过统一的多模态LLM框架，使用多种语音编码器与LLM适配器；采用LLM判读器对自由文本输出进行评估；对模型使用WER、BLEU、chrF、宏F1、时序F1等指标进行评测。

**📊 数据集**

数据来源包括FLEURS、CommonVoice、OpenSLR、Bloom‑Speech、Thai Elderly Speech、THAI SER、VoxVietnam、VietMed等公开语料，并自行合成时间推理和问答子集。

**📈 对比分析**

与多款开源与商用音频LLM（MERaLiON、SeaLLMs‑Audio、Qwen2、Phi‑4、Gemini 2.5 Flash、GPT‑4o 等）进行横向对比；开源模型在ASR上最优者为MERaLiON‑2，商用模型整体最好；在情感识别、语音翻译、时序任务上表现低于可用阈值，且低资源语言性能显著逊色。

**⚠️ 局限性**

局限在于数据覆盖不足，尤其是方言和低资源任务；评测集可能受到模型训练数据泄漏影响；需要更多高质量标注与合成数据提升泛化。

---

## 100. Exploring 3D Glyph Physicalizations for Public Engagement through River Health

**arXiv ID:** 2609.09472 | [PDF](https://arxiv.org/pdf/2609.09472v1)

**作者:** Maria Teresa Ortoleva `[一作]` (King's College London), Alfie Abdul-Rahman `[通讯]` (King's College London)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并原型化了一套基于三维字形（glyph）的实体化工具包，用于公众参与伦敦河流健康数据的可视化与反思。

**💡 创新点**

将传统二维字形扩展到三维实体化，并结合可回收材料和可个人化注释，以支持认知、情感和社会层面的互动。

**🔧 技术方法**

使用物理化技术，基于插槽式（slot‑in）鱼形结构，配合纸板、卡纸、胶带和荧光笔进行组装。

**📊 数据集**

使用伦敦大都会政府公开的伦敦河流健康地图（River Health Map）及其关联指标，如河流分支、排水管类型、道路径流等。

**📈 对比分析**

通过案例实验与用户访谈评估，显示该工具包在提升对多维数据的理解与情感投入方面优于传统二维图表，但缺乏量化对比实验。

**⚠️ 局限性**

局限性包括未在大规模公开场合测试，材料可回收性与美观度受限，以及缺乏对不同受众群体的可访问性评估。

---

## 101. X-CoSD: Communication-Efficient Cross-Vocabulary Collaborative Speculative Decoding

**arXiv ID:** 2609.09166 | [PDF](https://arxiv.org/pdf/2609.09166v1)

**作者:** Jaeduk Lee `[一作]` (Seoul National University), Wan Choi `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了跨词表协同推理框架X-CoSD及其改进版X-CoSD-E，用于在设备端小模型与服务器端大模型词表不一致时实现无损加速推理。

**💡 创新点**

核心创新在于混合残差重采样(HR)和服务器重采样加设备验证(SR-DV)两种机制，既保证了与服务器LLM分布一致，又显著降低了通信负载。

**🔧 技术方法**

主要技术包括：跨词表Token‑Level Intersection限制候选词汇、混合残差重采样、服务器生成替代候选并进行设备端验证、以及对残差分布的两阶段抽样分析。

**📊 数据集**

实验使用Vicuna‑68M作为设备端SLM，Llama‑3.1‑8B和Qwen2‑7B作为服务器端LLM，评估数据集涵盖WMT‑DeEn、XSum、CNN/DailyMail、GSM8K、MMLU等。

**📈 对比分析**

与传统CoSD、U‑HLM、贪婪重采样等基线比较，X‑CoSD与X‑CoSD‑E在保持生成质量≈服务器LLM的同时，通信负载（尤其下行）显著下降，令token吞吐量提升且延迟降低。

**⚠️ 局限性**

局限性在于仍需在重采样时下行传输部分分布信息，且在极低下行速率下仍受限；SR‑DV虽降低了通信，但可能引入额外交互延迟；对不同设备与服务器的兼容性需进一步验证。

---

## 102. Pairit: A Platform for Live Experiments on Human-AI Collaboration

**arXiv ID:** 2609.09789 | [PDF](https://arxiv.org/pdf/2609.09789v1)

**作者:** Harang Ju `[一作]` (Johns Hopkins University), Sinan Aral `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Pairit，一个基于单一 YAML 配置文件的声明式平台，用于设计、测试和部署包含人类与 AI 参与者的实时组织实验，支持匹配、聊天、协同工作区、AI 代理行为和流程控制，并在多项已发表研究中进行验证。

**💡 创新点**

创新点在于：① 将 AI 作为一等参与者融入实验流程，可在同一配置文件中声明聊天、编辑、动作等多种角色与行为；② 使用声明式实验图解耦实验设计与实现，提升实验可审计、可复现、可共享；③ 提供完整的过程日志和可视化界面，方便研究者分析实时协作动态。

**🔧 技术方法**

技术手段包括：YAML 声明式实验图、图形化流程编译、服务器端 AI 代理（如 GPT-5-nano）、实时聊天与协同编辑、匹配与随机化模块、可嵌入自定义 HTML/表单组件、数据导出（CSV/JSON/JSONL）以及 CLI lint/compile 机制。

**📊 数据集**

使用的数据集主要为线上实验招募平台 Prolific 的参与者：共计 2,234 名受试者用于广告策划实验，2,500 名受试者用于双人谈判实验；此外在内部预演阶段也利用语言模型模拟参与者进行 in silico 预测试。

**📈 对比分析**

与传统手工编写实验平台（如 oTree、Empirica 等）相比，Pairit 能够在单一配置文件中完成实验设计、人员分配、流程控制和 AI 代理配置；实验验证显示其能够可靠同步多人实时交互、生成高分辨率过程日志，并支持多种组织设计实验的快速迭代。虽然没有传统基准对比指标，但其在实际研究中的成功部署证明了系统的可用性与可靠性。

**⚠️ 局限性**

局限性包括：① 需要外部平台和 AI 模型访问权限，非所有研究团队可即时部署；② 虽然配置文件可共享，但实验执行仍受招募渠道、审稿流程和服务器托管等外部因素影响；③ 当前验证案例主要集中在广告策划和谈判等特定任务，对更广泛的组织场景尚需进一步测试；④ 对隐私与数据安全的审计仍需完善，尤其是涉及敏感对话内容时。

---

## 103. Efficient User Association and Wireless Scheduling with Shorter Time-Scale Rate Adaptation

**arXiv ID:** 2609.09387 | [PDF](https://arxiv.org/pdf/2609.09387v1)

**作者:** Xiaoyi Wu `[一作]`, Bin Li `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在无线网络中，结合不同时间尺度的用户关联、排程与速率自适应，提出了一种整合MaxWeight、虚拟队列与UCB的在线学习算法，以最大化累计吞吐量并保证公平。

**💡 创新点**

创新点包括：① 在用户关联/排程与速率自适应间实现多时间尺度协同；② 将虚拟队列与UCB权重结合形成公平最大化权重；③ 推导出累计公平违约率为0且累积损失为O(√KlogK)；④ 提出低复杂度的Pick‑and‑Compare变体。

**🔧 技术方法**

核心技术包括：MaxWeight调度、UCB（上置信界）多臂赌博机学习、虚拟队列公平约束、Lyapunov稳定性分析、低复杂度Pick‑and‑Compare策略。

**📊 数据集**

实验基于真实60 GHz毫米波测试平台收集的30个链路、30 000个时隙的信道质量（post‑SNR）数据，随后生成可用速率集合进行仿真。

**📈 对比分析**

与传统的基于UCB或固定排程算法对比，提出的MaxWeight-UCB算法在吞吐量近似最优（逼近率随时间趋近1），实现了零累计公平违约；Pick‑and‑Compare版本在相同吞吐量下显著降低了计算开销，性能差距仅为常数级别。

**⚠️ 局限性**

限制包括：① 需要已知或可估计的信道成功率（μ），在极端时变环境下学习速度可能受限；② 仍需评估在更大规模网络中的可扩展性；③ 对干扰模型和真实多用户情形的假设仍保持简化。

---

## 104. TempTPI: Informer-Based trajectory prediction for maritime vessels

**arXiv ID:** 2609.09840 | [PDF](https://arxiv.org/pdf/2609.09840v1)

**作者:** Kevin Ferneding `[一作]` (Technical University of Denmark), Peder Heiselberg `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于Informer的海上船舶轨迹预测框架TempTPI，结合多通道时间编码实现长期预测

**💡 创新点**

创新点在于将Informer的ProbSparse稀疏注意力与傅里叶频率扩展的多周期时间编码融合，显著提升了5小时预测的误差

**🔧 技术方法**

采用的技术包括Informer结构、ProbSparse自注意力、时间编码的正弦余弦扩展、1D卷积、全连接解码器以及PyTorch实现的训练管线

**📊 数据集**

使用的数据库是丹麦海域公开的AIS历史轨迹数据，采样时间为2025年10月1日至14日的商船航迹

**📈 对比分析**

通过与TPTrans的对比实验，TempTPI在1~5小时预测窗口内平均MSE下降约55%，并在训练与验证曲线中表现出更稳健的收敛

**⚠️ 局限性**

局限性包括仅在两周时间窗口内评估，缺乏跨季节和跨海域的数据验证，且模型在预测边界时偶尔出现越界问题

---

## 105. Leveraging Fine-grained Error Correction in Korean Speech Recognition for Consultation Services

**arXiv ID:** 2609.09889 | [PDF](https://arxiv.org/pdf/2609.09889v1)

**作者:** Yonghyun Jun `[一作]` (Chung-Ang University), Hwanhee Lee `[通讯]` (Chung-Ang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究针对韩语 ASR 纠错提出了两阶段文本后编辑框架 DCSC，结合检测器与上下文增广的 span‑level 校正，并首次构建了 1974 份真实客服对话的 DasanCallDial 基准数据集。

**💡 创新点**

创新点在于：①在低资源韩语环境下引入 token‑level 错误检测与检测门控的纠错流程；②利用对话上下文与 span‑level 目标实现精细化纠错；③构建真实、错误稀疏的韩语客服对话数据集，填补了韩语 post‑editing 资源缺口。

**🔧 技术方法**

技术包括：基于 ELECTRA 的编码器检测器、基于 pkoT5 的 seq2seq 纠错器、对话上下文插入与 Span 标注、token‑level 交叉熵训练与两阶段路由机制。

**📊 数据集**

使用数据集：DasanCallDial（1974 条客服对话，共 115,460 句子，误差率约 17.95%）以及公开的 Hyper‑BTS 用于跨域评测。

**📈 对比分析**

与零规则、基于规则、字符级 Seq2Seq、Utterance‑to‑Utterance 等基线相比，DCSC 在检测 F1 最高、错误句 WER 下降至 26.27、整体 Bal‑WER 下降至 13.30，显著优于 LLM 直接生成方案。

**⚠️ 局限性**

局限性包括：检测阈值需手工调优导致误报/漏报；缺乏音频信息限制对语音特定错误的纠正；对话上下文窗口有限，长距离依赖难以捕捉；模型对多错误 span 的鲁棒性仍有限。

---

## 106. On the Parameterized Complexity of Coloring Discovery

**arXiv ID:** 2609.09837 | [PDF](https://arxiv.org/pdf/2609.09837v1)

**作者:** Eric Decker `[一作]` (University of Bremen), Sebastian Siebertz `[通讯]` (University of Bremen)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了图着色发现问题的参数化复杂性，特别是三种修改步骤模型：颜色翻转、颜色交换和颜色滑动。

**💡 创新点**

提出了针对不同参数的精确固定参数算法，并证明了在某些情况下的NP完全性和难度下界。

**🔧 技术方法**

使用了固定参数算法和整数线性规划（ILP）技术来解决不同的着色问题。

**📊 数据集**

使用了图的顶点覆盖数、距离完全图的距离、树深度、反馈顶点集数等作为参数进行分析。

**📈 对比分析**

与现有方法比较，颜色翻转模型在顶点覆盖和距离完全图的参数下是固定参数可解的，而颜色交换和滑动模型在某些参数下是NP完全的。

**⚠️ 局限性**

在颜色滑动模型中，复杂性仍然是开放的，且在所有固定颜色的情况下，无法找到通用的单指数时间算法，除非指数时间假设失败。

---

## 107. Subgroup Membership Inference Audits of Differentially Private Synthetic Text

**arXiv ID:** 2609.09848 | [PDF](https://arxiv.org/pdf/2609.09848v1)

**作者:** Yidan Sun `[一作]`, Anil Anthony Bharath `[通讯]` (Imperial College London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种以目标池为显式参数的子组成员身份推断（MIA）游戏，并在此框架下对四个真实数据集、三种生成器和五个DP预算进行系统性审计。

**💡 创新点**

创新点在于：①将目标池抽象化为参数，实现高风险子组、随机对照和合并池的统一对比；②构建32种代理指标并针对三种攻击者场景进行校准；③揭示DP对高风险子组的保护不均匀，且泄露的记录主要集中在少数条目上。

**🔧 技术方法**

使用技术包括：差分隐私训练（DP‑SGD）、训练‑free 生成器（Aug‑PE、EPSVec）、多维代理评分（n‑gram 重叠、词向量相似度、概率模型）、归一化校准（RMIA‑style）以及基于AUC和低FPR TPR 的评估。

**📊 数据集**

所用数据集包括医学（N2C2'08、PsyTAR）、金融（DMSAFN）和法律（EurLex）文本，每个数据集的样本量从 604 到 5,042，记录长度从几百到几千词不等。

**📈 对比分析**

与现有发布‑基准攻击（Canary、DOMIAS）相比，本文的方法在 30/36 组合中实现了更高的 AUC（最高 0.83，最低 0.55），并能捕捉到残留泄露的集中趋势；DP 在平均水平上显著降低泄露，但对高风险子组的削弱效果不如随机样本。

**⚠️ 局限性**

局限性包括：数据集规模和子组规模较小，导致相关性统计受限；实验仅覆盖了三种生成器和发布‑基准攻击，未考虑模型访问型攻击；多次重绘子组和完整随机采样的方差未被估计，实验成本较高。

---

## 108. PccDiffuser: Multi-solution Motion Planning for Continuum Robots

**arXiv ID:** 2609.09745 | [PDF](https://arxiv.org/pdf/2609.09745v1)

**作者:** Ke Qiu `[一作]` (Zhejiang University), Haojian Lu `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于条件扩散模型的连续机器人多解路径规划框架PccDiffuser，能够一次性生成多条可行的配置空间路径并转化为可执行轨迹。

**💡 创新点**

创新点包括：①将扩散模型与连续机器人非奇异指数坐标参数化结合，直接学习完整路径分布；②在逆扩散过程中引入解析微分运动学作为梯度指导，实现终点精度和全身空隙提升；③使用图神经网络处理可变数量障碍，实现环境感知；④通过批量去噪并行生成多解，大幅提升规划效率。

**🔧 技术方法**

使用的技术主要是：条件扩散模型（DDIM）、一维时序U-Net、图神经网络（GNN）、解析微分运动学、动力学约束时间分配、基于RRT/RRT*和APF的基准比较。

**📊 数据集**

训练数据包括：约353k条无障碍、243k条含0–4个球形障碍的三段PCC机器人路径，采用随机逆运动学、人工势场修复、Douglas–Peucker简化等生成。

**📈 对比分析**

与传统配置空间/工作空间RRT/RRT*以及APF进行对比，PccDiffuser在含障碍场景下成功率达到91.11%（高于70%），平均运行时仅34 ms，且多解规划时候并行采样成本几乎不变，显示出显著的可靠性和效率优势。

**⚠️ 局限性**

局限性包括：只在静态球形障碍下验证，未覆盖动态环境或非球形障碍；解析微分指导在极端姿态下可能产生梯度不稳定；需要GPU加速，单机CPU性能仍有限；对大规模连续机器人（段数>3）尚未评估。

---

## 109. Online Inverse Integer Linear Optimization via Small-Gradient Skipping: Constant Regret and Finite Mistakes

**arXiv ID:** 2609.09809 | [PDF](https://arxiv.org/pdf/2609.09809v1)

**作者:** Akira Kitaoka `[一作]` `[通讯]` (NEC Corporation), Akira Kitaoka (NEC Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出小梯度跳过机制（SGS）用于在线逆向线性优化，在每轮无错误时不更新迭代和内部状态，并将其应用于OGD、ONS和MetaGrad，得到T无关的误差和损失上界。

**💡 创新点**

创新点在于利用误差缺失的梯度零化跳过，显著降低更新次数；在统一间隔下给出有限误差和损失的保证；在整数线性规划下提供显式的多项式上界，且不需要每轮计算重心。

**🔧 技术方法**

使用在线凸优化（OGD、ONS、MetaGrad）框架、统一间隔假设、梯度跳过策略、组合学下的间隔下界、子梯度更新与投影分析。

**📊 数据集**

论文为理论研究，无特定实测数据集；主要使用整数线性规划、M-convex 等形式化问题结构进行分析。

**📈 对比分析**

与现有方法比较时，SGS+ONS或MetaGrad实现的误差和损失上界不含log T，且对ILP可得到O(d² M₂ log(2M₂))，相比之前的O(M₂ d logT/d)和指数型O(exp(d log d))更优；M-convex 结构下得到O(L d log(2dL))，与需中心重心法的O(d log d)对比，计算成本更轻。

**⚠️ 局限性**

局限性包括：需要统一间隔的正间隔假设；上界与已知下界仍存在维数/范数上的倍数差距；不支持噪声或被污染的数据；对大间隔问题可能不如简单方法表现。

---

## 110. A Trust-Network-Based Federated Learning Framework for Multi-Center Aging Clock Prediction

**arXiv ID:** 2609.10108 | [PDF](https://arxiv.org/pdf/2609.10108v1)

**作者:** Chunxu Zhang `[一作]` (Hong Kong Polytechnic University), Qiang Yang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一种基于信任网络的联邦学习框架，用于多中心衰老钟模型的构建与解释。

**💡 创新点**

创新点在于将可信关系建模为有向对等信任，采用渐进式模型传播而非集中聚合；结合AgeMoE（混合专家+Transformer）与生成式重放，兼顾可解释性与知识保持，并在生物学层面解析蛋白交互从二阶到高阶组织。

**🔧 技术方法**

使用AgeMoE混合专家模型、条件VAE生成重放、Trust‑Network‑Based Federated Propagation等技术。

**📊 数据集**

使用UK Biobank蛋白质组、GEO DNA甲基化以及基于年龄分区的UKB‑AgeSplit等多中心数据集。

**📈 对比分析**

与本地训练、FedAvg等对照，平均MAE下降约0.6–0.7岁，模型波动小、无显著忘却；不同基线（线性、XGBoost、MLP、Transformer）均受益；生成重放进一步提升异构数据下的全局泛化。

**⚠️ 局限性**

局限包括：信任网络预设不自适应、仅使用年龄标签、未整合多组学、缺乏真实临床部署的评估。

---

## 111. HiRAD: A Flexible Large-Scale AGV Routing System

**arXiv ID:** 2609.09752 | [PDF](https://arxiv.org/pdf/2609.09752v1)

**作者:** Yunjie Huang `[一作]` (Hong Kong University Of Science And Technology Guangzhou), Lei Li `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了HiRAD框架，用层次化强化学习在连续空间中实现大规模AGV安全高效路径规划，兼顾运动学约束与实时性；

**💡 创新点**

创新点包括高分辨率时空建模将连续运动映射为可微分RL问题；将方向决策与速度控制分离的Alternating Decision–Velocity Control；以及通过观察裁剪与地图导向优先级实现的Asynchronous Decision–Velocity Control，将复杂度从O(n²)降到O(n)并显著降低推理延迟；

**🔧 技术方法**

采用强化学习（LSTM+策略网络）进行宏观方向规划；预定义离散速度模式实现微观速度控制；使用观察裁剪和基于网格的优先级索引实现异步决策；

**📊 数据集**

在随机网格（40×40至160×160）和两种真实仓库地图（G1 16×214、G2 32×428）上进行实验，AGV规模从64到2048；

**📈 对比分析**

与经典CBS、ICBS、ODrM*、OHSMD以及RL基准PRIMAL对比；HiRAD在大规模场景下实现了45%–63%更短的完成时间（makespan）和更低的运行时；同时在动态环境中的鲁棒性优于OHSMD，且相较PRIMAL具有更高的成功率和更快的推理速度；

**⚠️ 局限性**

限制主要包括：对离散速度模式的依赖可能限制在极端速度变化的场景；在极端高密度障碍或极大规模网络中，宏观策略训练仍需大规模样本；以及对模型泛化到非仓库环境的验证仍不足。

---

## 112. Elastoformer: Enabling Dynamic Adaptivity via Elastic Model Transformation

**arXiv ID:** 2609.10018 | [PDF](https://arxiv.org/pdf/2609.10018v1)

**作者:** Sudaksh Kalra `[一作]` (University of Amsterdam), Dolly Sapra `[通讯]` (University of Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了 Elastoformer 框架，将预训练的静态神经网络（如 Vision Transformer、ResNet-50、VGG-16）转换为弹性网络，使其能在边缘设备上根据实时资源（延迟、功耗、内存）自适应切换不同的推理模式。

**💡 创新点**

核心创新在于：①两阶段 Compress‑Grow 过程可从单一模型生成多级子网络；②权重共享增长机制在保留核心参数的同时只新增少量参数，显著降低内存占用；③无需多模型堆叠即可实现实时弹性推理，避免传统 bag‑of‑models 的加载与切换开销；④该方法对 Transformer 和 CNN 均通用，展示了广泛适用性。

**🔧 技术方法**

主要技术包括：结构化剪枝（L1‑norm saliency）实现多级压缩；按需逐步恢复参数的 Grow 阶段；权重共享与部分冻结策略；梯度裁剪与学习率调度；系统监测模块用于实时选择最佳子网络；实现细节基于 PyTorch。

**📊 数据集**

实验使用 ImageNet、CIFAR‑10、CIFAR‑100 三大视觉数据集，分别验证 ViT‑Base、ResNet‑50 和 VGG‑16 的弹性化效果。

**📈 对比分析**

对比方法包括原始 ViT‑Base、ResNet‑50、VGG‑16；Early‑Exit ViT、AdaptiveNet、Token‑Merging 方案等。评估指标为 Top‑1 准确率、FLOPs、延迟与内存占用。实验结果表明：在 CR=0.6 时，FLOPs 可压缩至 15% 原量，保持 90%+ 原始准确率；延迟下降 50% 以上；与 AdaptiveNet 相比，内存占用减少 76%；与 bag‑of‑models 相比，节省 60% 内存。

**⚠️ 局限性**

局限性：核心子网络（最小模型）容量决定整体性能，高压缩比例（CR≥0.8）会导致准确率显著下降；权重共享在极度压缩时的效果有限；当前需要人工设定子网络数量与压缩比例，缺乏自动化搜索；未来计划开发自动化搜索与层级蒸馏等提升方法。

---

## 113. Stable Answers, Unfinished Reasoning: Why Self-Consensus Is Not a Safe Early-Exit Signal

**arXiv ID:** 2609.09989 | [PDF](https://arxiv.org/pdf/2609.09989v1)

**作者:** Yunxiang Mo `[一作]` (Hong Kong University of Science and Technology), Hejia Geng `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对自我共识（self-consensus）在大语言模型早停（early exit）中的有效性进行系统评估，并通过对 3,520 规则的全搜索揭示其无法同时实现安全性和节省代价的根本原因。

**💡 创新点**

提出并验证了“共识–终止差距”（consensus–termination gap）这一机制，即在固定探测方式下，多次探测得到一致答案仅说明当前答案保持不变，而非推理已结束，从而导致早停错误。

**🔧 技术方法**

使用离线探测（offline probing）在冻结轨迹上实现自我共识评估；采用窗口大小、共享阈值、成熟度、答案有效性等超参数进行全网格搜索；对比基于边界置信度的非共识方法 DEER；通过宏平均、token 计量和准确率下降门限等指标进行评估。

**📊 数据集**

实验基于两种大模型（例如 LLaMA-70B 与 4B 版本）在三个数学竞赛数据集（MATH500、AMC23、AIME24）上，采用多随机种子（42/43/44）划分 train/dev/test，另外使用两款未见模型做外部验证。

**📈 对比分析**

结果显示：无一条 3,520 条共识规则能通过预先设定的三条安全/节省门限（准确率下降 ≤1%、token 节省 ≥10% 等），而 DEER 在同一评估流程下能实现 28–32% 的 token 节省且准确率下降 ≤1%。

**⚠️ 局限性**

局限性包括：仅评估单一探测后缀与固定窗口共识策略；使用冻结轨迹无法观察探测对推理过程的潜在影响；实验集为数学竞赛，可能不适用于开放式推理或代码生成；probe 代价对节省率影响较大；缺乏对探测独立性的直接控制实验。

---

## 114. A Sharp Barrier for Consistent Submodular Maximization: Any Improvement over $2-\sqrt{2}$ Entails Exponential Queries or Linear Recourse

**arXiv ID:** 2609.09986 | [PDF](https://arxiv.org/pdf/2609.09986v1)

**作者:** Shi Fu `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种一致的子模最大化问题，探讨了在元素逐步到达时解决方案质量与稳定性之间的权衡。

**💡 创新点**

提出了一种随机算法，证明了在多项式查询和最坏情况下常数回收的条件下，能够达到的最优近似为2-√(2)≈0.5858，低于离线1-1/e的保证。

**🔧 技术方法**

使用了随机算法，结合了多项式时间实现和有界位数的理性oracle答案。

**📊 数据集**

未具体提及使用的数据集，但讨论了在多项式查询和常数回收条件下的理论结果。

**📈 对比分析**

与之前的算法进行比较，证明了任何固定的改进都需要在关键到达之前进行指数级的查询，或者在到达时进行线性数量的更改，显示出一致性的计算成本。

**⚠️ 局限性**

算法的局限性在于，即使在无限查询的情况下，当前oracle仍然隐藏了哪些元素在到达后会被需要，导致在关键插入时无法及时修正解决方案。

---

## 115. LLMSec-AV: A Vulnerability Taxonomy and LLM-Driven Software Weakness Discovery Framework for Autonomous Vehicles

**arXiv ID:** 2609.09386 | [PDF](https://arxiv.org/pdf/2609.09386v1)

**作者:** Md. Wasiul Haque `[一作]` (University of Alabama), Mizanur Rahman `[通讯]` (University of Alabama)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了面向自动驾驶软件的漏洞分类法，并基于此开发LLMSec-AV框架，利用大型语言模型在Autoware代码上进行功能级漏洞检测。

**💡 创新点**

创新点在于首次将AV特定漏洞分类法作为机器可读上下文直接喂给LLM，展示了LLM在缺乏显式规则时对AV特有漏洞类的发现能力。

**🔧 技术方法**

技术上采用了Codestral 22B与gpt‑oss 20B两款LLM，结合检索增强（检索公开漏洞记录）与三步结构化提示，并与传统静态分析器（CodeQL、Semgrep、cppcheck、Clang Static Analyzer）做对照。

**📊 数据集**

数据集使用真实开源AV堆栈Autoware的代码基，构建了基于上游修复和公开漏洞记录的ground truth，评估约1,600个功能单元。

**📈 对比分析**

对比方法为卷积匹配和“体积匹配”基线；LLM在不依赖规则的情况下召回约12%–17%已知缺陷，显著高于所有静态分析器（后者几乎未匹配任何缺陷）。

**⚠️ 局限性**

局限性包括：动态确认阶段几乎无成功，导致缺陷确认率为0；ground truth仅基于修复与公开记录，可能不完整；仅评估一个AV堆栈与两种LLM，未测验多模型、不同种子或其他中间件。

---

## 116. Who Are They to Each Other? Multi-Agent Reasoning for Speaker Relationship Inference

**arXiv ID:** 2609.09628 | [PDF](https://arxiv.org/pdf/2609.09628v1)

**作者:** Yaohan Guan `[一作]` (Johns Hopkins University), Najim Dehak `[通讯]` (Johns Hopkins University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种无训练的多代理推理框架，用于从口语对话中推断说话者之间的关系。

**💡 创新点**

创新点在于设计了两种多代理交互机制：多角色多代理辩论（分别赋予语言学家、社会学家、心理学家等角色）以及多代理竞争（通过淘汰赛和成对裁决保持多种假设）。

**🔧 技术方法**

技术主要是利用大型语言模型（GPT‑5‑mini、GPT‑audio‑1.5、Qwen2.5‑Omni‑7B）在推理时进行多代理交互，配合结构化的提示与裁判策略。

**📊 数据集**

使用的数据集是 Seamless Interaction 语料库（自然对话子集），经过筛选后包含 607 条对话，涵盖熟悉与陌生关系以及细粒度的 4 类关系。

**📈 对比分析**

与零样本推理、标准多代理辩论和 CortexDebate 等基线对比，Multi‑Agent Compete 在文本、音频和音频+文本三种模态下均取得了最高的宏 F1 与准确率，特别是在二分类和细粒度关系预测上提升了 0.5–0.8 的宏 F1，3–4% 的准确率。

**⚠️ 局限性**

局限性包括：仅在单一数据集上评估，推理过程需多次 LLM 调用导致推理成本高，且某些关系类别样本量极少，导致按类别性能不稳健。

---

## 117. Layerwise Tunable Lifting Scheme for the Convolutional Neural Network

**arXiv ID:** 2609.09827 | [PDF](https://arxiv.org/pdf/2609.09827v1)

**作者:** Abdumannon Yovkochov `[一作]` (University of California San Diego), Truong Nguyen `[通讯]` (University of California San Diego)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在CNN中引入可调biorthogonal wavelet提升架构，实现低频/高频分支的层级可调式下采样。

**💡 创新点**

提出三种可调提升策略（低频调节、高频调节、顺序调节），并保证完美重构与可逆性。

**🔧 技术方法**

利用基于格子（lattice）的可调提升结构，将一维提升映射到二维，集成到ResNet-18中。

**📊 数据集**

在DTD纹理分类、MVTec-AD榛子异常检测以及私有KRC102S PCB异常检测数据集上评估。

**📈 对比分析**

与传统ResNet-18、WaveCNet、OrthLatt-UwU、BiorLatt-UwU等基线比较，最高在DTD上实现45.32%准确率，在MVTec-AD榛子上AUROC 99.75%，显著优于基线。

**⚠️ 局限性**

限制在于需要额外的通道扩张导致参数略增，且目前仅支持固定基底wavelet，未探索多基底自适应。

---

## 118. Introvert Clustering for Distributed Graph Algorithms

**arXiv ID:** 2609.10044 | [PDF](https://arxiv.org/pdf/2609.10044v1)

**作者:** Yi-Jun Chang `[一作]` (National University of Singapore), Nima Dolatabadi `[通讯]` (University of Copenhagen)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种新的图划分方法——introvert clustering，并利用它构造了多层introvert网络分解，从而在分布式模型中实现了O(log^2 n)的确定性列表边缘着色和O(log^2 n)的1/4-ε局部平衡割。

**💡 创新点**

引入了introvert clustering这一既保留低直径又保证每个聚类顶点至少有一半邻居留在同一聚类的划分，并将其与层次化分解结合，突破了传统网络分解只能实现贪婪式并行的局限。

**🔧 技术方法**

核心技术包括 Miller–Peng–Xu 的低直径聚类、剪枝（trimming）步骤以满足introvert属性、递归网络分解的白盒改造以及基于这些分解的简洁并行算法。

**📊 数据集**

该工作为纯理论算法，未使用任何实验数据集；所有结果均通过理论分析得到。

**📈 对比分析**

相较于以往需要 Ω(logΔ) 或多项式 Δ 复杂度的方案，本文的算法在所有 Δ ≥ Δ0(ε) 范围内实现了 O(log^2 n) 的确定性时间，且在边缘着色问题上取得了与最佳随机算法相当的时间。

**⚠️ 局限性**

限制在于只能在使用 (3/2+ε)Δ 颜色时达到 O(log^2 n) 的确定性时间，尚未解决使用 (1+ε)Δ 颜色的情况；此外，局部平衡割仅能达到 1/4-ε 的保证，距离最优 1/2 仍有较大间隙。

---

## 119. Stability of Fork-Join Systems with Redundancy and Heterogeneous Servers

**arXiv ID:** 2609.09237 | [PDF](https://arxiv.org/pdf/2609.09237v1)

**作者:** Chutong Gao `[一作]` (Northwestern University), Ohad Perry `[通讯]` (Southern Methodist University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究带冗余的分叉-合并系统（FJR）的稳定性，确定了最大稳定区间并给出了静态与动态容量分配的最优设计与控制条件。

**💡 创新点**

首次用投影到(k,k)系统与广义Schur-凸（GSC）序比较法完成了FJR系统的稳定性分析，证明了冗余对稳健性提升的理论机制。

**🔧 技术方法**

采用了状态空间投影、随机过程的样本路径比较以及GSC顺序理论，结合连续时间马尔可夫链转移率解析。

**📊 数据集**

无数据集，全部为理论模型与数学推导。

**📈 对比分析**

通过与已知的(k,k)无冗余系统比较验证方法，得到最大稳定区间为[0,1)，说明在所有可行策略下稳定性可达极限；未给出平均等待时间等性能指标。

**⚠️ 局限性**

仅适用于Poisson到达、指数服务时间、FCFS服务；未讨论非指数或非Poisson情形，也未验证在实际系统中的性能与鲁棒性。

---

## 120. Beyond Conventional Federated Learning via High-Order Regularization

**arXiv ID:** 2609.09904 | [PDF](https://arxiv.org/pdf/2609.09904v1)

**作者:** Alireza Kabgani `[一作]` (University of Antwerp), Masoud Ahookhosh `[通讯]` (University of Antwerp)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HiFedProx，一种在联邦学习中使用尺度匹配的高阶幂型正则化的第一阶优化方法。

**💡 创新点**

创新点在于将二次正则化推广为可调指数 p 的高阶正则化，并通过正则化梯度在参考位移 R 处保持相同，从而实现对小位移的温和惩罚和对大位移的更强惩罚。

**🔧 技术方法**

采用高阶幂正则化、有限预算的随机客户端优化、同一小批次 Armijo 回溯线搜索，以及基于梯度方向的自动微分技术。

**📊 数据集**

在 60 名作者的 FEMNIST 子集上进行实验，使用两层卷积网络，结合不同的噪声标签与步长设置。

**📈 对比分析**

通过与传统 FedProx（p=2）和其他指数 p∈{2,…,8} 的配对实验对比，发现指数在 5–7 区间能显著压缩客户端位移尾部并在中至重压条件下提升损失和准确率，且 p=6 在最重压下表现最佳。

**⚠️ 局限性**

局限性包括只在单一数据子集与网络结构上验证，未探讨无标签写手、不同压缩策略或连续指数；且未给出完整收敛性证明，需进一步联合调优 μ、R 与 p。

---

## 121. Agentic Web Accessibility Auditing: Authoring and Evaluating Per-Criterion Worker Agents for WCAG

**arXiv ID:** 2609.09379 | [PDF](https://arxiv.org/pdf/2609.09379v1)

**作者:** Arjun Mishra `[一作]` (University of British Columbia), Dongwook Yoon `[通讯]` (University of British Columbia)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个共享框架，将每个 WCAG 成功标准映射到专门的 worker 代理，支持多种工具和模型进行可解释的自动化无障碍评估，并利用 Library Accessibility Alliance（LAA）审计报告构建基准进行评估。

**💡 创新点**

提出基于标准的 worker 架构，分离工具调用与标准指令，生成可追溯的执行记录；在同一保存页面上系统性对比规则引擎、Vision‑Language 模型、交互式 worker 与通用编码代理，探索基于证据可视化与中立决策的审计工作流。

**🔧 技术方法**

采用 Headless Chromium 浏览器、axe-core 规则引擎、Vision‑Language 模型（Gemini、Claude、GPT‑5.5）、通用编码代理（Claude Code、Codex）以及 ReAct 类循环与工具库，配合 Python 评估脚本与日志系统。

**📊 数据集**

基于 LAA 发表的学术平台审计报告，构建了 24 页、11 个平台的保存 HTML 样本，总计 250 个页面–标准记录，其中 78 条为正样本，余下为采样负样本。

**📈 对比分析**

使用 micro‑平均 precision、recall、F1 进行评估；与 axe‑core、批量 VLM、单项 VLM 以及编码代理进行对比。worker 在引用运行下 recall 约 0.86、precision 约 0.80，规则引擎精度高但 recall 低；VLM 在视觉语义类指标上表现优异；worker 在键盘交互类指标上更强；成本与精度呈现权衡关系。

**⚠️ 局限性**

基准样本有限且仅来自已审计页面，负样本推断假设可能引入噪声；未对真实用户或开发者进行验证；工具与模型差异难以单独归因；只评估页面级别，未覆盖多页面流程；重现性有限，仅两次 worker 重跑；成本估计为 API 等价，未测量人类审计工作量或修复效果。

---

## 122. Are Unreachable Nodes Truly Safe? Fully Eclipsing Monero's P2P Network!

**arXiv ID:** 2609.10260 | [PDF](https://arxiv.org/pdf/2609.10260v1)

**作者:** Ruisheng Shi `[一作]` (Beijing University of Posts and Telecommunications), Qin Wang `[通讯]` (CSIRO)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了两种针对Monero不可达节点的日蚀攻击（Nyx和Moros），并在仿真网络和主网中验证了其可行性和持久性。

**💡 创新点**

首次提出不需要入站连接即可对不可达节点实施日蚀的间接同义词投毒方法，并结合Monero内部的定时同步和连接刷新机制实现稳定隔离；同时引入了两种攻击场景覆盖现有节点与新节点。

**🔧 技术方法**

采用端口多样化的Sybil投毒、灰名单/白名单注入、Monero协议的timed sync和update_sync_search、网络级传播模型、SEED Emulator仿真以及对主网的实测。

**📊 数据集**

使用1,200节点的Monero仿真网络（包含6个种子节点）和Monero主网真实节点；Sybil节点采用1,000个/24子网IP（以及额外的20个灰名单IP）进行投毒。

**📈 对比分析**

通过评估peerlist占比、连接接管率、时间到日蚀（TTE）和稳定性等指标进行比较；Nyx在约27分钟内完成12/12接管，并在18小时内保持稳定；Moros在8秒内完成新节点日蚀，持续10.8小时；与传统日蚀方法相比，攻击不依赖入站连接，且更易持续。

**⚠️ 局限性**

攻击需要大量/24子网IP资源，Sybil成本高；实验仅针对Monero的P2P机制，跨链适用性未验证；未在完整主网规模上进行全量测试；现有防御措施无法阻挡，需更根本的身份/信任机制改进。

---

## 123. CoGe-GCD: Reframing Generalized Category Discovery with Compositional Generalization

**arXiv ID:** 2609.10158 | [PDF](https://arxiv.org/pdf/2609.10158v1)

**作者:** Luyao Tang `[一作]` (University of Hong Kong), Cheng Chen `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了CoGe-GCD框架，旨在通过构造视觉片段的可复用原语和几何校正来改进通用类别发现任务

**💡 创新点**

创新点在于将人类认知中的分层感知（Compositional Perception）与归纳推理（Generalizing Induction）融合到GCD中，利用原语竞争与信息聚合、空间相似性校准，显著提升未知类别的辨识和类别数量估计

**🔧 技术方法**

主要技术包括：基于Transformer的视觉特征提取、原语竞争与证据聚合的超图信息传递、基于空间距离的几何校准、以及无监督的类别聚类/对比损失

**📊 数据集**

在六个公开基准上进行评估，分别为CUB-200、Stanford Cars、FGVC Aircraft（细粒度）以及CIFAR-10/100、ImageNet-100（粗粒度）

**📈 对比分析**

与多种基线（CMS、SimGCD、LegoGCD、SelEx、ORCA、GCD等）在同一backbone与head设置下对比，CoGe-GCD在所有类别准确率上平均提升约1-4%，尤其在未知类别上提升2-5%，同时计算开销仅增加约1%

**⚠️ 局限性**

局限性包括：对原语数量M的选择仍有一定敏感性，极大规模数据集上对空间校准的迭代效果尚未充分验证，且模型仍需在不同任务域（如多模态、连续学习）中进一步测试

---

## 124. LeCor: Learning to Be Corrected by Meta-Learned Test-Time Training for Interactive 3D Lung-Tumour Segmentation

**arXiv ID:** 2609.09477 | [PDF](https://arxiv.org/pdf/2609.09477v1)

**作者:** Yi Luo `[一作]` (Johns Hopkins University), Kai Ding `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并实现了LeCor，一种通过元学习的测试时训练方法，使肺肿瘤交互式3D分割模型在接收少量点击纠正后能在未点击的切片上持续改进。

**💡 创新点**

创新点在于将点击误差视为训练信号，通过对案例特定适配器的元学习初始化和步长，实现在测试时单步梯度更新能够提升整个病灶的分割精度，而非仅作为提示；该方法首次在医学交互式分割中将测试时更新本身作为学习目标。

**🔧 技术方法**

使用的技术包括：SAM 3基础模型（带记忆机制）+LoRA微调、rank‑4 LoRA案例适配器、元学习（MAML/Meta‑SGD）对适配器参数与步长进行学习、交互式点击模拟与损失设计（点击交叉熵+一致性损失）以及对未点击切片的Dice评估。

**📊 数据集**

使用了五个公开肺CT数据集：LUNA25、NSCLC‑Radiomics、MSD Task06 Lung、LNDb 和 4D‑Lung；在133个至少8切片的病例上进行测试。

**📈 对比分析**

通过与基线（fine‑tuned 模型+记忆条件）、固定步长测试时更新以及各实验变体进行对比，LeCor 在第7轮达到Dice 0.827，比基线高约4个百分点；在Dice≥0.80阈值下失效率下降43%，并且仅需3轮即可达到基线7轮所能达到的质量；计算成本仅比基线多约1 s/轮。

**⚠️ 局限性**

局限性包括：需要至少8切片的病灶；点击模拟仅针对当前误差最深点，尚未验证真实临床点击的鲁棒性；对步长与损失权重的设置敏感；每轮需要一次反向传播，导致额外GPU时间。

---

## 125. On the Evolution of the Capacity-Achieving Input Support for the Amplitude-Constrained AWGN Channel

**arXiv ID:** 2609.10039 | [PDF](https://arxiv.org/pdf/2609.10039v1)

**作者:** Luca Barletta `[一作]` (Politecnico di Milano), Alex Dytso `[通讯]` (Qualcomm Flarion Technology, Inc.)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了幅度受限高斯信道的最优输入分布，并对相应的 KKT 函数 G_A 进行了系统分析。

**💡 创新点**

创新点在于将 G_A 的导数与条件累积量建立直接联系，并证明 G_A 可扩展为整个复平面上的唯一整个函数。

**🔧 技术方法**

主要采用了信息理论中的 KKT 条件、条件累计量、Leibniz 交换法则以及复分析中的整个函数扩展技术。

**📊 数据集**

研究属于理论分析性质，未使用任何实际数据集。

**📈 对比分析**

通过解析推导和数值演化展示了 G_A 的性质，未涉及与其他方法的性能比较。

**⚠️ 局限性**

局限在于仅针对幅度约束的高斯信道，且对多重最优分布的进一步数值验证与通用性仍需进一步研究。

---

## 126. Reproducing Omitted Temporal Expressions in Japanese News for Retrieval-Augmented Applications

**arXiv ID:** 2609.09569 | [PDF](https://arxiv.org/pdf/2609.09569v1)

**作者:** Tomoaki Yasuda `[一作]` (Kagawa University), Shotaro Ishihara `[通讯]` (Nikkei Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于规则的流水线，在日本新闻文本中使用发布时间作为参考点，将省略的时间表达式（如“21日”“上个月”“这年”）恢复为完整的具体日期或时间区间，并在文档被索引前完成此处理，旨在提升检索与 RAG 系统的时间一致性。

**💡 创新点**

①在不使用 LLM 推理的情况下，实现与 GPT‑4o 等大模型相近的时间恢复精度；②将 ja‑timex 的规则抽取与新建的新闻特定过滤、范围端点补全、多重参考日解析、上下文日期选择等模块组合，形成可审计、可复现且计算成本低的系统；③证明了出版时间驱动的时间恢复能够显著提升时限检索性能。

**🔧 技术方法**

主要技术包括：
- 规则式时间表达式抽取（基于 ja‑timex）；
- 文本预处理与掩码过滤；
- 日历算术与多参考日解析；
- 时代转 Gregorian 年份转换；
- 上下文模式匹配决定月份/年份；
- 与 GPT‑4o、GPT‑4o‑mini、本地 LLM 以及 TF‑IDF/BM25、六种嵌入模型进行对比实验。

**📊 数据集**

使用的公开数据集：
- Nikkei Newspaper Article Open Corpus（97 篇，256 条省略时间表达，包含 241 个标注区间）；
- Livedoor News Corpus（270 篇，463 条省略时间表达，包含 439 个标注区间）。
此外构造了 148 条时限检索问答测试集。

**📈 对比分析**

评价方法：
- 抽取：与原始 ja‑timex 比较，测量 TP/FP/FN、精确率、召回率、F1；
- 归一化：固定范围下与 GPT‑4o、GPT‑4o‑mini、本地 LLM（Swallow 等）对比，记录准确率与处理时间；
- 端到端：结合抽取与归一化，报告 Nikkei（P 0.9575，R 0.9688，F1 0.9631）和 Livedoor（P 0.8609，R 0.9222，F1 0.8905）。
- 检索：在三种文本状态（原始、加上发布时间、时间恢复）下，使用 TF‑IDF、BM25、六种嵌入模型计算 Recall@1/3/5，发现时间恢复在大多数方案下均提升或保持最佳召回。

**⚠️ 局限性**

局限性：
- 仅覆盖可通过出版时间恢复的五类表达，忽略指示性、季节性、比较性表达及需要多重参考时间的情况；
- 未实现全局参考时间更新机制，对文章内部时间点漂移的处理不完善；
- 规则中对日/月选择的局部启发式对非典型未来表述可能失效；
- 规则和错误分析多基于 Nikkei 和 Livedoor，未验证在更大规模或不同领域的泛化能力；
- 代码与部分辅助数据未公开，完整可复现性受限。

---

## 127. Politics of Feelings: Emotional Expression and Legislative Effectiveness in the U.S. Congress

**arXiv ID:** 2609.10198 | [PDF](https://arxiv.org/pdf/2609.10198v1)

**作者:** Segun Aroyehun `[一作]` `[通讯]` (University of Konstanz), Segun Aroyehun (University of Konstanz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用基于Transformer的情感分类器，对1973-2024年美国国会1.7M次演讲文本进行情绪分析，探讨情绪随时间、议题、议员特征的变化，并检验情绪与立法效能的关系。

**💡 创新点**

创新点在于①系统量化并区分八种离散情绪（愤怒、恐惧、厌恶、悲伤、喜悦、热情、自豪、希望）在国会演讲中的动态与差异；②首次将离散情绪与议员立法效能关联，发现热情和自豪正相关、愤怒负相关；③引入情绪多样性、强度、取向等整体情感维度，完善情绪评估。

**🔧 技术方法**

主要技术为：Transformer多标签情绪分类器（对八种情绪输出连续得分），线性混合效应模型（控制议员、议题、会议期效应），趋势检测（Mann‑Kendall、Sen斜率）、分布差异检验（Kruskal‑Wallis）。

**📊 数据集**

使用的数据集为：①国会记录（Congressional Record）1973-2024年演讲文本共1,766,113条；②CAP政策主题分类器（21个主题）用于主题分配；③立法效能得分（Legislative Effectiveness Score）以及议员个人特征（性别、党派、院席、意识形态得分等）。

**📈 对比分析**

比较方法：先绘制各情绪随时间和党派的趋势（Mann‑Kendall显著性检验），再通过热图比较各议题的情绪配置，随后用混合效应模型估计情绪与立法效能的回归系数。结果显示热情与自豪对立法效能正向显著（b≈0.01-0.02），愤怒显著负向（b≈-0.04），情绪多样性和正向情绪也正向关联，情绪强度负向关联。

**⚠️ 局限性**

局限性包括：①研究仅限美国国会，结果可能不易推广至其他立法机构；②未能识别情绪产生的具体动机与目标；③仅涵盖八种情绪，缺乏更细致或其他情绪维度（如好奇、羞愧、怀旧等）。

---

## 128. TRACE: Training Reasoning Agents for Causal Exploration with Synthesized Rewards

**arXiv ID:** 2609.10315 | [PDF](https://arxiv.org/pdf/2609.10315v1)

**作者:** Rui Sun `[一作]` (Independent Researchers), Bing He `[通讯]` (Independent Researchers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个数字广告诊断环境，通过模拟隐藏干预产生可验证的奖励，训练 LLM 进行多步工具使用的根因归因；

**💡 创新点**

创新点在于将干预模拟与奖励合成结合，利用可控生成的干预标签作为客观奖励，克服了缺乏天然可验证目标的诊断任务；

**🔧 技术方法**

技术包括：模拟器‑oracle‑RL 框架、GRPO 强化学习、分层奖励（归因、完整归因、格式化）、Python+SQL 工具交互、监督预训练 (SFT)；

**📊 数据集**

数据集为 5,000 条已验证的模拟诊断任务（训练/验证）和 235 条 held‑out 诊断测试集，涵盖 12 种根因、单维/双维驱动切片与时间复杂度；

**📈 对比分析**

与多款前沿模型（Claude Opus 5、GPT‑5.6 Sol、Qwen3.5‑122B‑A10B 等）进行对比；SFT 后的 Qwen3.5‑35B 达到 0.757 FullAttr@1，远超最强提示基线 0.686，说明后期训练和可验证奖励比单纯增大模型规模更有效；

**⚠️ 局限性**

局限性包括：仍难以精准归因双维切片、对无信号情境的误归因率高、依赖可控模拟环境，难以直接迁移到无模拟的真实系统，且奖励设计和工具调用效率仍需进一步优化。

---

## 129. Vague2Detect: Handling Ambiguous Prompts in Knowledge-Based Open-World Detection

**arXiv ID:** 2609.09949 | [PDF](https://arxiv.org/pdf/2609.09949v1)

**作者:** Ibrohimjon Muminov `[一作]` (Dongguk University), Jihie Kim `[通讯]` (Dongguk University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Vague2Detect框架，结合知识库、微调的Sentence‑BERT检索、YOLO‑World视觉验证与GPT回退，实现对功能性模糊提示的目标检测。

**💡 创新点**

创新点在于在推理时动态扩展知识库、利用GPT生成新类别、以及通过微调SBERT精细化功能意图映射，从而显著提升模糊提示的成功率。

**🔧 技术方法**

使用了微调的Sentence‑BERT作为语义检索器、YOLO‑World作为开源词汇检测器、GPT‑3.5‑Turbo做回退生成以及结构化的家庭知识库。

**📊 数据集**

采用了Open Images V7子集与自采室内场景相结合的约1000张图像构成的家庭场景数据集，并包含100+ KB对象及对应的模糊提示。

**📈 对比分析**

通过与YOLO‑World基线比较，使用Vague Prompt Success Rate（VPSR）与检测准确率评估；VPSR从32%提升到61%（SBERT）再到85%（全管道），检测准确率从29%提升到61%再到83%。

**⚠️ 局限性**

局限性包括知识库规模有限（约100类）、细粒度类别易混淆、GPT回退成本较高以及对多义提示、小/遮挡对象识别仍存在错误。

---

## 130. SkNeXt enables topology-guided neuronal reconstruction from petabyte-scale microscopy data

**arXiv ID:** 2609.09832 | [PDF](https://arxiv.org/pdf/2609.09832v1)

**作者:** Jiayi Ding `[一作]` (Chinese Institute for Brain Research), Hu Zhao `[通讯]` (Chinese Institute for Brain Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 SkNeXt，一种先生成并手动校正稀疏骨架，再利用骨架先验进行高分辨率实例与语义分割的拓扑优先神经元重建工作流。

**💡 创新点**

将神经元骨架作为先验拓扑和空间索引，集中计算与人工干预在稀疏结构上，并通过骨架引导的选择性数据访问实现对 PB 级数据的可扩展重建。

**🔧 技术方法**

使用 3D ConvNeXt V2 多任务网络进行骨架与高分辨率分割预测，结合最小成本树构造、方向一致性成本、基于骨架的 Marker‑controlled Watershed，以及 OME‑Zarr/Zstd 存储与随机访问技术。

**📊 数据集**

在多份超分辨荧光脑图像数据上评估，包括 40×4096×4096 视角数据、0.3 PB 海马体超分辨数据以及整个鼠脑 1 PB 级数据。

**📈 对比分析**

与传统全体素 Watershed、Flood‑filling 网络对比，SkNeXt 在 0.3 PB 数据上单 GPU 仅需约 3 天完成实例与语义分割，手动校正时间从 4 小时降至 0.5 小时，显著提升效率且保持神经元身份完整。

**⚠️ 局限性**

在高度交织、信号弱或极细神经突触段仍需人工校正，骨架误检可能导致后续分割错误，且流程仍依赖高质量骨架预测和手动修正。

---

## 131. Geometry Conditioning in an Embodied SLM: Training Controls and Robustness Diagnostics in a 0.8B Hybrid Model

**arXiv ID:** 2609.09213 | [PDF](https://arxiv.org/pdf/2609.09213v1)

**作者:** Hao Li `[一作]`, Lin He `[通讯]` (University of Tennessee)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在冻结的0.8B混合语言模型上，利用LoRA和门控递归网络对LIBERO-Spatial三任务进行行为克隆，比较将几何信息注入输入token与递归门的两条路径。

**💡 创新点**

首次对输入token MLP与门控衰减门这两条几何条件路径以及训练时几何打乱和时钟条件的影响进行系统比较，揭示训练时几何对成功率无显著提升。

**🔧 技术方法**

使用Qwen3.5-0.8B模型、LoRA、门控Delta网络（GDN）、token-MLP适配器和时间钟控等技术。

**📊 数据集**

采用LIBERO-Spatial三任务的演示数据集。

**📈 对比分析**

通过三种随机种子、540次保留测试和McNemar检验比较，在正确几何下的门控衰减门成功率为28.9%，随机打乱训练为36.7%，相对几何token为34.4%，均略高于无几何基线24.4%，但差异不显著。

**⚠️ 局限性**

实验受限于单一模型、少量任务和种子，几何条件的参数瓶颈、训练不稳定以及仅在仿真环境中的评估限制了结论的普适性。

---

## 132. Freezing of Gait Prediction Under Spatial Occlusion: An IMU-Supervised Cross-Modal Distillation Approach

**arXiv ID:** 2609.09826 | [PDF](https://arxiv.org/pdf/2609.09826v1)

**作者:** Chandan Biswas `[一作]` (NeuroAI Fusion Labs), Anabik Pal `[通讯]` (Indian Institute of Science Education and Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种跨模态子空间蒸馏框架，在无穿戴设备的视频环境下预测帕金森病患者的冻结步态（FOG），通过在训练阶段用IMU的高精度运动子空间监督视觉模型，克服自遮挡导致的姿态估计误差。

**💡 创新点**

创新点：① 将IMU kinematic oracle 的子空间作为教师，对视觉+文本学生进行监督；② 采用置信度自适应的概率双流视觉融合，动态权重α随骨架置信度变化；③ 采用多目标对比学习同时对齐IMU与文本表征，避免同类负样本冲突。

**🔧 技术方法**

技术手段：1D CNN kinematic oracle，3D ResNet+ST-GCN骨架网络，Transformer文本编码，概率双流融合，监督式对比损失（SupCon）与InfoNCE对齐，Adam优化，Matthews相关系数、F1、AUPRC等评估指标。

**📊 数据集**

数据集：公开多模态FOG数据集（35名PD患者），包含360°转向的视频、IMU序列与临床指标，实验窗口长度为1.0–3.0秒。

**📈 对比分析**

对比方法：与RGB-only、骨架-only、早期融合以及IMU Oracle基线比较。结果显示跨模态蒸馏模型在帧级F1≈0.713、事件级F1≈0.694、AUPRC≈0.775、MCC≈0.613，显著优于视觉基线并逼近IMU上限（事件级F1≈0.718）。

**⚠️ 局限性**

局限性：训练阶段需要同步的IMU和文本数据，推理阶段仍需预训练模型；对非同步或缺失多模态数据的适应性有限；实验仅在实验室环境验证，未在室外或家庭真实场景中评估。

---

## 133. X-amine509: Predicting the Practical Risk Level of Enterprise X.509 Certificates

**arXiv ID:** 2609.09402 | [PDF](https://arxiv.org/pdf/2609.09402v1)

**作者:** Cameron Keith `[一作]` (Keyfactor), Caleb Shorter `[通讯]` (Keyfactor)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了X‑amine509系统，利用机器学习快速对企业大规模X.509证书进行风险分级和优先级排序，从而显著减少完整确定性检查所需的计算成本。

**💡 创新点**

创新点在于将基于CA/Browser Forum、NIST等标准的177项缺陷检查量化为风险评分，并通过系统化的特征工程、随机种子优化与贝叶斯超参数调优，打造了既高精度又极低延迟的二阶段风险分级模型；同时提供可解释性的特征重要性分析，为安全运维提供决策支持。

**🔧 技术方法**

技术包括：树形模型（Decision Tree、Extra Trees、Random Forest、Gradient Boosting、XGBoost、LightGBM、CatBoost、HistGradientBoosting）、神经网络（MLP）、特征工程（负序列号、有效期、EKU、国别、SAN类型等）、随机种子优化、贝叶斯优化超参数调优、ONNX推理、NDCG评估与多类别精确率/召回率评估。

**📊 数据集**

数据集为1,027,714份公开可获取的X.509证书（Fortune 500、.gov、.edu域），并在13个月后再收集571,374份进行模型耐久性测试。

**📈 对比分析**

与传统单一模型基线相比，最佳模型（Extra Trees）在测试集上实现MAE≈2.26、R²≈0.993、NDCG≈0.998；Decision Tree在速度与精度的折中下达成MAE≈2.39、R²≈0.986、可达3.7百万证书/秒。模型在后期数据上仍保持MAE≤7.4、R²≥0.915、Critical‑tier召回≥97%。

**⚠️ 局限性**

局限性包括：风险评分依赖预定义的标准缺陷，无法覆盖新型攻击；数据主要为公开信任的Web证书，特征重要性对内部PKI可能不适用；模型对稀有关键缺陷可能产生漏报，需与确定性检查结合使用；在高风险证书上召回率虽高但仍有少量漏报。

---

## 134. Pretraining and Distillation Matter More Than Architecture Family for Label-Free Single-Cell Classification

**arXiv ID:** 2609.09863 | [PDF](https://arxiv.org/pdf/2609.09863v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 135. "It's Like Drinking from a Fire Hose": Understanding and Characterizing Video Learning Experiences for Individuals with ADHD

**arXiv ID:** 2609.09443 | [PDF](https://arxiv.org/pdf/2609.09443v1)

**作者:** Hanxiu 'Hazel' Zhu `[一作]` (University of Wisconsin-Madison), Yuhang Zhao `[通讯]` (University of Wisconsin-Madison)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了ADHD学习者观看视频讲座的体验，并通过眼动追踪与回想式访谈收集行为与主观数据。

**💡 创新点**

首次系统性结合多模态视频设计、行为信号与学习者策略，揭示了ADHD用户的认知过载与无聊等挑战，并提出适配建议。

**🔧 技术方法**

使用眼动追踪仪（Tobii Pro Fusion）与实时可视化的自定义观看界面，以及对话式回想支持。

**📊 数据集**

研究样本为16名自报ADHD成年学习者，观看8段不同风格的视频讲座（约13–17分钟）。

**📈 对比分析**

研究未进行算法或系统的性能对比，主要通过定性编码与行为量化描述，未给出数值性能评估。

**⚠️ 局限性**

研究限制包括实验室环境可能导致观察者效应、样本量小、未系统操纵视频设计变量、仅测量即时回想与短期测验。

---

## 136. Beyond Surface Imitation: Contrastive Modeling for Reasoning Path Alignment in Multimodal In-Context Learning

**arXiv ID:** 2609.10177 | [PDF](https://arxiv.org/pdf/2609.10177v1)

**作者:** Mingbo Yang `[一作]` (Sun Yat-Sen University), Yan Xiao `[通讯]` (Sun Yat-Sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种新的多模态 In-Context 学习框架 COMIL，旨在通过对比演示建模、基于响应的检索和轻量级对齐控制，实现模型在推理时的路径对齐，而非仅表面模仿。

**💡 创新点**

创新点：
- 对比演示建模：把每个示例拆成（输入、劣质回应、更佳回应、推理路径）四元组，显式展示从劣质到最佳的改进过程；
- 基于响应的检索：在检索时同时考虑输入相似度与当前回应相似度，使检索到的演示更贴合当前状态；
- 轻量级对齐控制：训练一个小型质量预测器估计中间回应质量，动态决定何时停止迭代，提升稳定性。

**🔧 技术方法**

技术方法：对比演示生成、检索算法（余弦相似度加权）、迭代生成精炼、轻量级对齐控制（BERT+CLIP编码+MLP回归）。

**📊 数据集**

使用的数据集：CIFAR‑10（图像分类）、Flickr30k（图像字幕）、VQAv2、OKVQA（视觉问答）。在闭源模型上还验证了在 CIFAR‑10 上的效果。

**📈 对比分析**

比较方法：与多类基线（检索式 ICL：CLIPRE、KNN、CR；训练式 ICL：LCL、MimIC；CoT：Few‑shot CoT、Self‑Consistency CoT；自我精炼：Iteration、Self‑Refine、SC‑Captioner；以及最近方法 AIM、TACO、M²IV）在同一模型与数据集上对比。结果显示 COMIL 在 13/16 组设置中取得最优，尤其在 VQA 与 Captioning 任务上显著提升；在闭源模型上亦保持领先。

**⚠️ 局限性**

局限性：
- 需要额外的推理成本（多次生成、检索与控制），虽然相比重度推理方法降低了 50%~80% 的延迟/token；
- 依赖检索集大小与质量，检索集过小会限制效果；
- 对极其复杂或完全不相似的推理路径仍可能产生误导性改进，精炼不一定始终单调提升。

---

## 137. OntologyAligner: Ontology-Aligned Retrieval and Hierarchy-Guided Large Language Model Reranking for Biomedical Ontology Normalization

**arXiv ID:** 2609.10055 | [PDF](https://arxiv.org/pdf/2609.10055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 138. Development and Validation of a Physics-Guided Machine Learning Extrapolation Framework Using a Classical Transient Diffusion Benchmark

**arXiv ID:** 2609.09912 | [PDF](https://arxiv.org/pdf/2609.09912v1)

**作者:** Ashutosh Yadav `[一作]` (Indian Institute of Technology Jodhpur), Harshal Akolekar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于物理引导的机器学习外推框架，并在一维瞬态热扩散问题上通过逐步训练‑预测‑验证‑扩展的方式实现并验证了模型的外推可靠性。

**💡 创新点**

创新点在于将物理洞察直接嵌入网络结构与训练策略：对 BiLSTM 引入根-Fourier 坐标、边界加权损失和松弛递推；对 PINN 引入对数时间变换、硬边界约束和递进式训练；并构建了可在任意点使用的解析基准，形成闭环验证机制。

**🔧 技术方法**

采用了双向长短时记忆网络（BiLSTM）和物理信息神经网络（PINN），配合根-Fourier 坐标变换、对数时间坐标、边界权重损失、松弛递推等技术。

**📊 数据集**

使用了完整解析的瞬态热扩散问题数据集（Fo∈[0.0005,4]，x∈[0,1]），仅在 Fo∈[0.001,0.009] 的窄窗口内训练，随后在该窗口之外进行逐步外推验证。

**📈 对比分析**

对比方法：在训练窗口内对比插值误差（MAE、R²），在窗口外采用逐步验证得到的外推误差；结果显示修正后的 BiLSTM 和 PINN 在向后（Fo→0）和向前（Fo→∞）外推均保持 R²>0.98，MAE<0.01；BiLSTM 在向前外推误差增长更慢，向后外推需要更多松弛步骤。

**⚠️ 局限性**

局限性：依赖可解析的基准验证，难以直接推广到无解析解的真实工程问题；对物理知识的依赖限制了在未知或高度非线性系统中的适用性；在大尺度多物理耦合问题中，训练成本和参数规模仍需进一步优化。

---

## 139. Purchase Advice and Observable Buyer Responses in Real AI Conversations

**arXiv ID:** 2609.09878 | [PDF](https://arxiv.org/pdf/2609.09878v1)

**作者:** Benjamin Tannenbaum `[一作]` `[通讯]` (Aiso Boost Ltd.), Benjamin Tannenbaum (Aiso Boost Ltd.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对Aiso公开的历史AI助手对话进行审计，统计购买导向信息、后续用户回应以及明确的购买/放弃结论，检验会话深度是否夸大跟进率；

**💡 创新点**

首次系统性区分助手推荐内容、用户后续回应和交易结果，揭示仅凭会话深度无法准确估计转化率；

**🔧 技术方法**

使用基于规则的人工智能辅助标注、Python脚本进行统计与可视化；

**📊 数据集**

来自Aiso商业化研究数据库的317条对话记录（经筛选后剩67条），涵盖住宿、旅游、家居服务等类别；

**📈 对比分析**

通过描述性统计比较不同后续回应条件，发现相同任务后续跟进率为26.9%，若按总会话深度计则为34.3%，后者高出27.8%；

**⚠️ 局限性**

样本来源不具代表性、缺乏交易结果、标注缺乏独立人工验证，无法估计真实转化率或因果影响；

---

## 140. Lightweight Zero Trust via Automotive SDN

**arXiv ID:** 2609.09817 | [PDF](https://arxiv.org/pdf/2609.09817v1)

**作者:** Friedrich Wiemer `[一作]` (Robert Bosch GmbH), Florian Wagner `[通讯]` (ETAS GmbH)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种分两步实施的汽车零信任架构，先利用现有的MACsec/MKA与VLAN实现部分安全保障，再通过已有的CORECONF/YANG SDN管理平面实现动态策略、密钥生命周期管理与审计，从而无需新增零信任基础设施即可在车辆内部网络中实现零信任。

**💡 创新点**

创新点在于将SDN管理平面直接映射为NIST SP 800‑207中零信任的策略引擎、管理员与执行点，省去单独的零信任控制平面；同时将MACsec/MKA的PSK预配置与YANG包装的密钥生命周期结合，提供安全的密钥分发与轮换机制；并在同一架构下覆盖五个完整、两个部分的NIST tenets。

**🔧 技术方法**

使用的技术包括：IEEE 802.1AE MACsec、MKA、VLAN、TSN、CORECONF、YANG、DTLS、PKI证书与CRL、AUTOSAR CSM密钥管理、IETF YANG keystore与crypto-types模型。

**📊 数据集**

本文未使用公开数据集，而是通过架构建模与标准映射分析来验证方案，主要基于Open Alliance TC17、TC19规范和NIST SP 800‑207 的理论框架。

**📈 对比分析**

评估方法是对比NIST SP 800‑207中七个tenets的覆盖情况：Step 1 覆盖 T2、T3、T6 部分；Step 2 通过动态YANG配置实现 T1、T3、T6、T7 完全覆盖，T4、T5 仍为部分满足；未进行量化性能实验，重点在于架构兼容性与无额外协议栈的实现。

**⚠️ 局限性**

限制包括：仅在网络层实现零信任，未覆盖ECU内部的应用层授权；身份依赖现有PKI；T3 的每会话授权仍不完全；未实现信任算法与设备鉴定；缺乏对管理平面容错与可用性的深入分析；未验证对CAN‑XL 等其他链路层的迁移细节。

---

## 141. Tensor-Train Weak SINDy: Identifying High-Dimensional Nonlinear Dynamics

**arXiv ID:** 2609.09434 | [PDF](https://arxiv.org/pdf/2609.09434v1)

**作者:** Will Houser `[一作]` (University of Colorado), David M. Bortz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了 TT-WSINDy 算法，该算法将多维近似非线性动力学（MANDy）与弱形式稀疏识别（WSINDy）相结合，利用张量轨迹（tensor‑train, TT）格式在弱形式下对高维 ODE 进行稀疏回归，从而在不构造指数级特征矩阵的前提下完成模型识别。

**💡 创新点**

创新点主要体现在：① 将弱形式的积分变换与 TT 结构结合，避免了对导数数据的数值逼近，提升了噪声鲁棒性；② 在 TT 空间中实现稀疏化的“修改版序列阈值最小二乘”（TT‑MSTLS），通过在低秩子空间内完成特征筛选，显著降低了计算与存储开销；③ 引入低秩特征张量构造方法，使得即使在时间样本数远大于搜索空间尺寸的情形下仍能保持可行性。

**🔧 技术方法**

使用的关键技术包括：弱形式 SINDy（WSINDy）回归、张量轨迹（TT）分解与伪逆、跨相关（cross‑correlation）实现弱形式积分、修改版序列阈值最小二乘（MSTLS）、低秩 TT 构造（截断 SVD 与随机化范围寻找）以及对 TT‑MSTLS 的时间/空间复杂度分析。

**📊 数据集**

在实验中采用了四个经典动力学系统的数据：Fermi‑Pasta‑Ulam‑Tsingou（FPUT）、Lorenz 96、Kuramoto 模型和 Chua 电路。每个系统均使用合适的基函数库（如 {1, x, x², x³}、{1, x} 等）生成特征空间，并在不同噪声水平下采集大量时间样本。

**📈 对比分析**

与传统的平面 WSINDy（矩阵求解）进行对比。结果表明：① TT‑WSINDy 在高维（如 Lorenz 96 的 D=12）下保持了与平面 WSINDy 相同的识别精度（相对系数误差 ~10⁻⁴）；② 在大多数高维场景中，TT‑WSINDy 的运行时间比平面 WSINDy 低 2–7 倍；③ 对噪声的鲁棒性更好，弱形式在噪声范围 0–1 内始终优于或等价于平面形式；④ 低秩特征张量构造在大时间样本数 M 时代码仍能顺利执行，避免了显存溢出。

**⚠️ 局限性**

局限性包括：① 需要前提稀疏性，否则 TT‑MSTLS 的筛选不会显著减小问题规模；② 对搜索空间大小 Jⁿ、维度 D 与时间样本数 M 的形状仍有敏感性，某些参数组合下 TT‑MSTLS 可能不如平面求解；③ 目前仅针对 ODE，PDE 扩展仍待研究；④ 对截断误差、随机化 SVD 的选择不当可能导致漏检；⑤ 需要较高的实现复杂度和调参经验。

---

## 142. When Ad Networks Misbehave: Understanding Risks of Semi-Drive-By Splash Ads

**arXiv ID:** 2609.09574 | [PDF](https://arxiv.org/pdf/2609.09574v1)

**作者:** Song Wu `[一作]` (Independent Researcher), Xueqiang Wang `[通讯]` (University Of Central Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究并量化了基于广告 SDK 的“半驱动式闪屏广告”欺诈行为，并提出了可在大规模测量中部署的自动化 honeypot 框架。

**💡 创新点**

创新点在于首次将广告网络层视为攻击主体，揭示通过传感器阈值诱发的无用户交互广告点击，并结合硬化模拟器、LLM 合成使用轨迹和规则+GPT 辅助的日志分析实现了高效检测。

**🔧 技术方法**

技术手段包括跨层真机属性投射、LLM 驱动的使用与传感器轨迹合成、DeepLink 日志监控、规则+GPT-4o 扩展的属性匹配与归因引擎。

**📊 数据集**

数据集来源于中国四大应用市场共收集 32,758 个 APK，筛选后 31,823 个非游戏流行应用，用于 24 小时持续测量和欺诈率评估。

**📈 对比分析**

通过与 VPBOX、Android Emulator、Genymotion 等传统环境对比，框架在逃避 VM 检测、诱发欺诈率、以及 DeepLink 日志分析上均表现最优，测得 1.7%–4.9% 的欺诈率，日志分析召回率 91%，精准率 100%。

**⚠️ 局限性**

局限性包括仅针对 Android 闪屏广告、四大华为市场、模拟器与合成传感器，无法覆盖 iOS 或其他广告格式；阈值设定和动态阈值适配可能导致漏检；公开披露后可能促使对手更新检测逻辑。

---

## 143. SA-Profile: Automated Sulcus Angle Profiling from Super-Resolution MRI

**arXiv ID:** 2609.10125 | [PDF](https://arxiv.org/pdf/2609.10125v1)

**作者:** Michael Wehrli `[一作]` (University Basel), Philippe C. Cattin `[通讯]` (University Basel)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于超分辨率MRI的连续髌股关节髌槽角(SA)剖面自动测量框架，能够沿近端-远端轴连续描述髌槽形态。

**💡 创新点**

创新点在于将隐式神经表示（INR）与二维U-Net关键点检测相结合，利用最大似然样条拟合实现跨切面的连续性，避免传统单切面测量的切片选择和标记误差问题。

**🔧 技术方法**

使用了四层MLP隐式神经表示、带WIREx激活的INR、两层1024维隐藏层、U-Net结构的纵向和横向关键点检测模型，以及L‑BFGS‑B优化的三维样条拟合。

**📊 数据集**

数据集包括公开的fastMRI（1,566名患者）和本地32名髌槽发育不良（TD）患者的临床MRI；在fastMRI中进行模型训练与验证，在TD队列和fastMRI中进行临床一致性评估。

**📈 对比分析**

与传统单切面人工测量对比，自动方法在两组患者上的平均绝对误差(MAE)分别为11.6°（fastMRI）和8.9°（TD），与手工测量差距与已知的观察者变异范围相当；同时生成的SA剖面显示出群体层面的形态差异。

**⚠️ 局限性**

局限性包括：超分辨率重建可能导致软组织边界模糊；TD样本量小；fastMRI样本不构成健康参照；手工测量本身存在变异，且未与多专家标注比较；需要更大临床验证。

---

## 144. The Menu Is an Execution Prior: State-Path Tool Menus for Online Agents

**arXiv ID:** 2609.09395 | [PDF](https://arxiv.org/pdf/2609.09395v1)

**作者:** Bo Yan `[一作]` (University of Central Florida), Song Wang `[通讯]` (University of Central Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在大型工具库中为语言模型代理构建一个执行前的可执行工具菜单，确保多步任务的完整路径和可执行顺序。

**💡 创新点**

提出状态路径（state‑path）概念，将路径完整性作为菜单构造目标，并通过编码器‑检索器‑重排序器学习工具的兼容性、产出关系和执行顺序。

**🔧 技术方法**

使用基于 Transformer 的关系感知编码器结合可执行性、状态兼容性和路径相关性特征；检索器做覆盖式工具选择；重排序器做前缀可执行排序；训练过程采用轨迹监督。

**📊 数据集**

在 ToolBench（32 工具）以及 AppWorld、TRAJECT‑Bench、UniToolCall、ToolHop 等公开基准上进行评估，使用官方 304 任务拆分作为测试集。

**📈 对比分析**

与官方菜单、COLT、ToolRet、Tool‑REX、SkillRouter、ToolGen 等构造方法对比；在 32 工具菜单下，State‑Path 将在线成功率从 0.737 提升至 0.898，显著超越其它基线。

**⚠️ 局限性**

菜单一次性构造后不更新；依赖工具的输入/输出字段信息，缺乏文档时效果下降；对稀疏或受损路径历史敏感；在安全性和沙箱方面需要进一步控制。

---

## 145. VFNet: Multi-View Spatio-Temporal Model for Void Fraction Estimation in Gas-Liquid Two-Phase Flow

**arXiv ID:** 2609.09711 | [PDF](https://arxiv.org/pdf/2609.09711v1)

**作者:** Md Adnan Faisal Hossain `[一作]` (Purdue University), Fengqing Zhu `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种多视角时空网络，用同步的侧视和俯视视频预测气液两相流的空隙率，输出每个空间段的时间分辨估计。

**💡 创新点**

创新点在于：①将几何先验与学习残差结合，实现物理可解释的修正；②双分支结构（局部特征融合 + Mamba 时空分支）有效捕捉空间细节与长时依赖；③在无标注真实数据的情况下完成跨域迁移并提升下游流态分类。

**🔧 技术方法**

采用双分支卷积特征提取、双向 Mamba 状态空间模型、残差回归头和门控网络，并使用轻量化回归头与位置编码。

**📊 数据集**

使用基于 CFD 的合成数据集（16个不同工况，总计约 30,000 帧，分为 3,300 条双视角视频），测试集为 600 对。

**📈 对比分析**

与几何基线和多种深度视频骨干（I3D、SlowFast、VideoMAE、VideoMamba 等）比较，平均 MAE 0.33%、相对误差 11.43%、PSNR 49.41 dB，显著优于所有基线；在真实流态图像中提高流态分类准确率 10–14%。

**⚠️ 局限性**

局限在于：训练完全依赖合成 CFD 数据，缺乏真实空隙率测量对比；模型对极端或未见工况的泛化性尚待进一步验证。

---

## 146. VANTAGE-Bench: Evaluating the Infrastructure AI Gap in Vision-Language Models

**arXiv ID:** 2609.09396 | [PDF](https://arxiv.org/pdf/2609.09396v1)

**作者:** Zaid Pervaiz Bhat `[一作]` (NVIDIA), Vidya Nariyambut Murali `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

引入VANTAGE-Bench，一个面向固定摄像头基础设施AI场景的多任务评测框架；

**💡 创新点**

创新点在于跨语义、空间、时间、时空四大推理支柱，突破单一MCQ评测模式，设计8种任务并提出单通道轨迹跟踪协议；

**🔧 技术方法**

使用零样本评估方法，扩展VLMEvalKit进行多模态指标计算，并通过自然语言提示与生成式输出评估模型；

**📊 数据集**

构建了3,346个专家标注的媒体资产，涵盖物流、运输、智能空间三大部署域，包括视频任务、图像定位与密集检测；

**📈 对比分析**

与传统消费者视频基准对比，零样本性能在事件验证、指称表达和时间定位上落后9–24点，跟踪性能随视距延伸显著下滑，仅比专业跟踪器高约5点；

**⚠️ 局限性**

局限包括摄像头视角局限于高位固定镜头，数据主要来自美国两市，跟踪任务仅为合成，缺乏多物体追踪和3D定位等真实场景验证。

---

## 147. Breaking Fault Lines: Unifying TEE-Assisted BFT Consensus in Partially Trusted Worlds

**arXiv ID:** 2609.09742 | [PDF](https://arxiv.org/pdf/2609.09742v1)

**作者:** Xiaoqing Wen `[一作]` (University of British Columbia), Chen Feng `[通讯]` (University of British Columbia)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了针对部分TEE部署的BFT共识协议，在通用的“部分TEE”模型下重新定义了容错上限并提出双仲裁（dual‑quorum）和TEE领导者快速路径（TEE‑leader fast path）两项关键设计。

**💡 创新点**

创新点在于：①证明了部分TEE下的容错上限为 f < max{n/3, m/2}，揭示了TEE数量超过2/3时才可提升容错能力；②设计了可在TEE和非TEE节点间动态切换的双仲裁机制；③利用TEE不可伪造性实现TEE领导者的快速提议与视图切换，从而减少通信轮数与延迟。

**🔧 技术方法**

核心技术包括：Intel SGX可信执行环境、ECDSA签名、HotStuff风格的BFT协议、链式（pipelined）HotStuff变体、可信组件（Checker+Accumulator合并为“Trusted Module”）以及双仲裁与快速路径的协议逻辑。

**📊 数据集**

实验数据集：400条事务/块、每条256字节负载；部署在云平台的97台SGX实例上，使用Redis KV工作负载（100%读操作、1KB值）。

**📈 对比分析**

方法：在LAN/WAN环境下对比了自研协议（称为Raftel和Raftel‑Pipe）与HotStuff、Basic‑HotStuff、FlexiBFT等基线；结果显示Raftel在WAN下可达625 TPS、<670 ms延迟，比HotStuff高约308 TPS；在Redis工作负载下，Raftel/ Raftel‑Pipe吞吐率分别为84–92 TPS，明显优于传统HotStuff（44 TPS）和基线（91 TPS）。

**⚠️ 局限性**

局限性：①未对侧信道、回滚/分叉攻击等高级TEE攻击做防护；②依赖Intel SGX，未实现远程身份验证（remote attestation）与动态节点重配置；③在TEE节点不足2/3时容错提升有限，仅提供效率优化。

---

## 148. Networked Admissibility-Preserving Control for Directed Safe Coordination

**arXiv ID:** 2609.09384 | [PDF](https://arxiv.org/pdf/2609.09384v1)

**作者:** Abhinav Sinha `[一作]` (University of Cincinnati), Shashi Ranjan Kumar `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

针对有向根树网络，提出了一种网络化可接受性保持控制（APC）框架，将输入实现（APIR）作为状态引入，使用对数壁垒坐标实现安全通道协同。

**💡 创新点**

创新点在于：①把物理输入动态化为状态，形成真正的可接受性保持输入实现；②给出方向特定的兼容性条件，保证各向异性执行器限制内的安全协同；③通过精确级联推导实现对有向网络达成值的显式修正；④引入部分固定点（pinning）实现无全局参考的安全轨迹分配。

**🔧 技术方法**

采用的技术包括：对数壁垒Lyapunov函数、边界函数σ_i、基于有向图拉普拉斯矩阵的权重向量π、精确级联控制律、方向性兼容性判定以及对齐误差的指数衰减分析。

**📊 数据集**

实验使用5个单积分体在两种拓扑（强连通图与仅含根树图）以及部分固定点情形下的仿真数据，没有使用公开数据集。

**📈 对比分析**

通过仿真比较，结果表明：状态始终保持在时变安全通道内，命令保持有界，物理输入满足预定的非对称限制，且系统实现指数收敛至安全协同轨迹；与传统的直接饱和控制相比，APC在保持安全边界的同时消除了因输入实现引起的协同误差。

**⚠️ 局限性**

局限性包括：①兼容性判定为区域性，仅对给定的初始转化状态集合有效；②要求拓扑固定、根树结构；③对参数p₁、p₂、γ等的精确设置敏感，缺乏鲁棒性分析；④未考虑离散采样、通信延迟或拓扑切换等实际网络不确定性。

---

## 149. Quantum MDS codes from complements of unions of finite-field subsets

**arXiv ID:** 2609.09943 | [PDF](https://arxiv.org/pdf/2609.09943v1)

**作者:** Naihong Hu `[一作]` (East China Normal University), Hong Ji `[通讯]` (East China Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构造了五族 Hermitian 自正交的 GRS 码，并利用 Hermitian 构造得到新的量子 MDS 码；

**💡 创新点**

创新点在于使用补集结构与可预设迹值、范数值及乘法子群余数的交集来确定定位点，并给出足够条件保证 Hermitian 自正交，从而得到可比现有构造更长或更高最小距离的量子 MDS 码；

**🔧 技术方法**

采用有限域的 Lagrange 系数、消失多项式、Frobenius 同态和代数几何方法，结合线性代数和群论技巧，构造定位集和列标；

**📊 数据集**

论文未使用实验数据集，全部为理论构造与数学证明；

**📈 对比分析**

与基于迹映射、线性变换、乘法子群余数等已有构造做严格长度与距离比较，证明在同一长度下本构造的最小距离至少比前者大 1 或更大，且在某些参数下可达到距 q/2+1 的量子 MDS 码；

**⚠️ 局限性**

局限性包括仅适用于奇素数幂 q、构造需满足一系列参数约束、且仅给出 MDS 码的存在性而无具体实现细节；

---

## 150. Can Artificial Intelligence Support Healthcare and Mental Health Through Early Cyberbullying Detection ? The Impact of Emotion-Aware AI on Proactive Online Safety

**arXiv ID:** 2609.09735 | [PDF](https://arxiv.org/pdf/2609.09735v1)

**作者:** Hamed Jelodar `[一作]` (University of New Brunswick), Sajjad Dadkhah `[通讯]` (University of New Brunswick)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CareGuard早期预警框架，集成情感感知过滤、Transformer多模型分类与LLM后处理，能够检测网络霸凌并生成可解释的情绪、实体与摘要信息。

**💡 创新点**

①先行情感感知过滤显著降低噪声与计算成本；②采用RoBERTa、BERT、DistilBERT+Bi‑GRU多模型融合，提升对敏感类别的鲁棒性；③LLM后处理提供情绪分析、实体提取和摘要，增强解释性与干预可行性。

**🔧 技术方法**

零射击语义标注、Fine‑tuned RoBERTa/BERT/DistilBERT + Bi‑GRU、情感情绪特征提取、余弦相似度过滤、LLM（LLaMA）多任务推理、Prompt工程与Chain‑of‑Thought。

**📊 数据集**

Kaggle公开的网络霸凌推文数据集（约1137条测试样本），标签覆盖种族/宗教/性别/非霸凌等四类。

**📈 对比分析**

对RoBERTa‑base（基准）与BERT‑base、DistilBERT 进行性能对比；RoBERTa‑base准确率0.91、宏F1 0.90，DistilBERT明显下滑；在敏感类别（种族、性别）上RoBERTa‑base表现最佳。

**⚠️ 局限性**

仅使用单一数据集，存在领域漂移与语言多样性限制；类别不平衡导致敏感类别误检；缺乏多平台、多语言及大规模真实环境评估；隐私与伦理风险需进一步验证。

---

## 151. The Mutations of Machine Speech

**arXiv ID:** 2609.09496 | [PDF](https://arxiv.org/pdf/2609.09496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 152. When Does Defendant Statement Matter? A Study of Bias and Persuasion in LLM-Simulated Jurors

**arXiv ID:** 2609.09887 | [PDF](https://arxiv.org/pdf/2609.09887v1)

**作者:** Cho-Ying Wu `[一作]` `[通讯]` (Bosch AI Research), Cho-Ying Wu (Bosch AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型（LLM）在美国普通法刑事陪审团模拟中的行为，评估情感说服、意识形态偏见和被告-陪审团背景亲和度对判决严厉度的影响，并发布了专门的JuryBench基准数据集。

**💡 创新点**

首次将前沿LLM与陪审团模拟系统结合，构建500个高度争议的刑事案例、设计多样化陪审团与被告背景，并量化情感说服与偏见效应，为法律心理学与AI交叉研究提供了新工具。

**🔧 技术方法**

使用20款前沿LLM进行陪审团角色扮演，配合线性回归因子分析、统计显著性检验与对比实验来量化情感强度、悔意和背景亲和度的影响。

**📊 数据集**

自制JuryBench数据集，包含500个案例、约400种可能指控、不同被告背景、陪审团意识形态分布，以及情感强度与悔意评分，用于训练与评估LLM陪审团。

**📈 对比分析**

通过计算判决严厉度变化、成功率、受损率等指标对20款LLM进行比较，结果显示大部分模型与人类陪审团相似，情感说服影响有限，背景亲和度是最显著的决定因素。

**⚠️ 局限性**

模拟流程简化为单阶段，未涵盖完整庭审、陪审团讨论与交叉质询；数据仅针对美国普通法且为人工合成，缺乏真实人物，情感传播仅通过文本呈现，且模型更新可能导致结果变化。

---

## 153. Chance, Persistent Advantage, and the Generative-AI Era in Open-Source Package Careers

**arXiv ID:** 2609.09687 | [PDF](https://arxiv.org/pdf/2609.09687v1)

**作者:** Hazem Ibrahim `[一作]` (New York University Abu Dhabi), Yasir Zaki `[通讯]` (New York University Abu Dhabi)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对开源软件贡献者的职业生涯进行大规模量化分析，检验其成功模式是否与科学、电影、音乐等领域一致，并评估生成式 AI 工具出现后对职业轨迹的影响。

**💡 创新点**

首次在软件领域验证“随机影响规则”和 Q‑模型，提出稳健的稳定因素与动量分解方法，并在相同贡献者的 AI 时代前后进行置换对比。

**🔧 技术方法**

使用统计检验（卡方、TVD、置换检验）、受限最小二乘拟合、Bootstrap、Benjamini–Hochberg 多假设校正，以及 Python/SQL 数据处理流水线。

**📊 数据集**

GitHub Archive（2015‑2025 推送事件）、Libraries.io 与 Ecosystem.ms 的依赖快照，以及自建的身份映射表。

**📈 对比分析**

将结果拆分为发现/验证两半，在 54 种条件下做多重检验校正；对随机影响规则采用 TVD 与置换 P 值评估偏差；对稳定因素采用三重校验与动量分解；AI 时代对比使用真实与假设对照，并通过 Bootstrap CI 判断差异。结果显示随机影响规则基本成立，稳定因素约占 20%，AI 时代未出现显著变化。

**⚠️ 局限性**

受限于仅可测量发布包的影响，缺失非包贡献；缺乏完整的依赖快照导致插值误差；AI 时代设计仅为日历区分而非实际使用率，样本量与随访时间有限；平台身份与地区分布不均，结果仅具描述性而非因果推断。

---

## 154. Albedo Estimation via Latent Bridge Matching

**arXiv ID:** 2609.09884 | [PDF](https://arxiv.org/pdf/2609.09884v1)

**作者:** Carme Corbi `[一作]` (Universitat Autònoma de Barcelona), Maria Vanrell `[通讯]` (Universitat Autònoma de Barcelona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出基于 Latent Bridge Matching（LBM）的单步反照率估计框架，并加入像素重建损失与阴影条件，实现了更高的物理一致性与推理效率。

**💡 创新点**

创新点在于：①利用 LBM 的源图像到目标图像的单步传输，显著降低推理时间；②在像素空间加入重建损失，强制满足光照模型；③阴影估计作为条件可互相提升，构建两阶段互补解码器。

**🔧 技术方法**

使用了 VAE 编码器/解码器、Stable Diffusion XL UNet、LBM 训练目标、LPIPS 与 MSE 组合损失，并对阴影/法向进行条件编码。

**📊 数据集**

训练与评估使用 InteriorVerse、Hypersim、MIT Intrinsic Images、ARAP、IIW 等五个公开数据集，全部仅用合成数据进行训练。

**📈 对比分析**

与主流 IID 方法（PIE‑Net、FlowIID、IntrinsicDiffusion 等）对比，LBM‑AID 在 MIT 的 LMSE 与 PSNR 上均优于基准；在 InteriorVerse、Hypersim 与 IIW 的整体指标也保持竞争力，且推理速度提升约 75%，单步采样显著加速。

**⚠️ 局限性**

局限包括：1) 依赖大模型（≈2.5B 参数）导致显存占用高、仅能处理 ≤2K 分辨率；2) 训练完全基于合成数据，现实场景的泛化受限；3) 两阶段互补仍受阴影估计误差影响，尚未实现端到端联合训练；4) 评价指标碎片化，缺乏统一基准。

---

## 155. Unthrottling the Tanh Jacobian in SAC: A Negative Result on Bang-Bang Control and MetaDrive

**arXiv ID:** 2609.09478 | [PDF](https://arxiv.org/pdf/2609.09478v1)

**作者:** Faiq Shamass `[一作]` `[通讯]` (Independent Researcher), Faiq Shamass (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究Soft Actor-Critic（SAC）中 tanh 压缩导致的 Jacobian 消失对极端动作的影响，并尝试通过添加梯度旁路（bypass）来补偿该缺失。

**💡 创新点**

提出并验证了一种最小化的梯度旁路机制：在 actor 损失中加入一个与 critic 的未裁剪动作梯度相同、但不受 tanh 缩放的额外项，探讨其对极端动作任务的效果。

**🔧 技术方法**

使用 SAC 算法、tanh 压缩的高斯策略、梯度旁路设计、阈值门控、MetaDrive 仿真环境以及双积分最小时间任务来实现与评估。

**📊 数据集**

使用的“数据集”主要是自定义的实验环境：双积分最小时间任务的随机初始状态分布，以及 MetaDrive 的交通仿真环境（模拟驾驶数据），并不依赖公开的大规模真实数据集。

**📈 对比分析**

通过配对种子实验对比标准 SAC 与旁路版本，评估指标包括返回值、碰撞率、离路率、动作饱和率等。实验结果表明标准 SAC 已能接近极限，旁路不提升返回；在双积分任务中，旁路导致动作饱和率飙升、返回急剧下降；在 MetaDrive 微调中，旁路无显著返回提升且常伴随离路率上升。

**⚠️ 局限性**

局限性：实验仅在有限的任务（双积分、MetaDrive 微调）和少量种子下进行；未对从头训练、不同门控策略或更大参数网格做系统探索；MetaDrive 结果受强基准影响，且仅在特定交通密度和熵设置下测试。

---

## 156. Procedural Memory Under Change: Reuse and Interference in Controlled Web Tasks

**arXiv ID:** 2609.09774 | [PDF](https://arxiv.org/pdf/2609.09774v1)

**作者:** Yanze Cao `[一作]` `[通讯]`, Yanze Cao

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对语言代理的程序性记忆在接口、数量、表示、优化目标与分布式证据等多种偏差下的匹配与干扰进行实验性探究，分为人机辅助的回溯适配案例与冻结记忆的对照实验两阶段；

**💡 创新点**

创新点在于将“记忆–任务不匹配”“可观察干扰”“记忆导致的错误”三者明确区分，并设计四类诊断签名的正式单次实验，系统检验不匹配程序在显式证据充分时是否会产生可观察错误；

**🔧 技术方法**

采用冻结提示、温度0、单次推理、对比记忆条件（原始记忆、无记忆、表格适配、语义改写）以及自定义诊断签名的评估框架；

**📊 数据集**

使用BrowserGym TimeWarp Task 57、WebShop V1–V6的接口记录以及基于两类商品（杏仁、米饼）的32个合成购物决策样本；

**📈 对比分析**

通过对比四种记忆条件与无记忆基线，对每个诊断签名的出现率进行计数。实验显示，在所有32个单元中均未出现预定义的干扰签名，且大多数条件下可实现参考兼容的选择和全局最优，总体表现良好；

**⚠️ 局限性**

局限性包括：回溯阶段仅为人机辅助、未能验证自适应记忆成功；对照实验仅在单一模型、温度0、单次推理下进行，未检验重复性、采样变异或跨模型泛化；合成决策任务过于简化，缺乏真实浏览过程与噪声；记忆条件差异较大，难以隔离单一因素；未给出统计显著性或人口层面错误率。

---

## 157. Seeing the Voice, Preserving the Self: A Participatory Design Approach to Deaf-Centric Text-to-Speech

**arXiv ID:** 2609.10199 | [PDF](https://arxiv.org/pdf/2609.10199v1)

**作者:** Shela Atemnkeng `[一作]` (Gallaudet University), Christian Vogler `[通讯]` (Gallaudet University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过参与式设计方法（焦点小组、共创工作坊、早期设计评估），与聋人用户共同探索和定义聋人中心的文本转语音（TTS）技术的功能需求与交互设计，提出了非听觉验证、身份保留与情感调节等关键设计概念。

**💡 创新点**

创新点在于：①首次将聋人社区的参与式设计带入TTS领域；②把非听觉验证与身份认同视为核心设计需求；③将聋人口音保留从技术挑战转化为设计价值；④为聋人内容创作者提供可视化情感调节与多模态验证工具。

**🔧 技术方法**

技术上主要采用了定性研究工具（访谈记录、CART字幕、ASL翻译）以及原型化工具（Figma、Figma Make、OpenAI Codex）来快速生成视觉化交互原型；在语音合成层面提到可利用现有商业TTS（ElevenLabs、Lovo等）和潜在的深度学习模型进行情感与身份调节。

**📊 数据集**

数据集方面没有使用公开语音或文字数据集，研究所依据的是27名来自美国华盛顿地区的聋人/听障参与者的原始访谈文字记录与共创草图。

**📈 对比分析**

比较方法：通过“5秒测试”、短视频演示和侧面比较三阶段评估，定性收集参与者对三种可视化（emoji、karaoke、波形）在情感表达、可读性、可信度等方面的主观评价；未提供量化性能指标，仅呈现定性反馈与设计优劣对比。

**⚠️ 局限性**

局限性包括：样本规模小、地域集中（华盛顿地区与美国），缺乏聋盲或聋残障参与者；早期原型缺乏交互性，仅展示静态/短视频；未进行真实语音合成与用户体验测试；技术实现仍需大量研发，如情感识别、口音保留、局部情绪标注等。

---

## 158. Agent-Based ML-LLM Fusion with Self-Optimizing Prompts for Plateau Weather Alerts

**arXiv ID:** 2609.10135 | [PDF](https://arxiv.org/pdf/2609.10135v1)

**作者:** Shuai Yan `[一作]` (Chengdu Jincheng College), Shan He `[通讯]` (Chengdu Jincheng College)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了 SmartWeatherAgent，一套集成意图识别、LightGBM 高原极端天气预测和基于 LLM 的自适应生成的端到端智能气象预警系统。

**💡 创新点**

创新点在于：①在 LLM 生成过程中嵌入 12 轮微步提示自适应闭环，实现生成–评估–优化的动态迭代；②针对高原天气引入多维度的特征工程（风速突变率、降水突发指数、阵风比等）显著提升 LightGBM 的预测性能；③将规则推理与 LLM 生成统一到三阶段架构中，提升预警的专业深度、逻辑严谨性和科学可信度。

**🔧 技术方法**

采用的技术包括：正则表达式+Qwen3 进行意图识别；LightGBM 并结合高原特征进行短期预报；提示自适应闭环（生成–评估–优化）及 12 轮微步迭代；特征工程（滚动统计、绝对差、阶梯编码等）；Bayesian 超参搜索、时间序列交叉验证；以及 S_final 等多维评估指标。

**📊 数据集**

使用了拉萨 2024–2025 年 1 月 1 日至 5 月 21 日的每小时气象观测数据 12,168 条（温度、降水、风速/阵风、紫外指数、能见度等），并在 VisualCrossing 公开数据中提取相关特征。

**📈 对比分析**

实验中将 LightGBM 与 Random Forest、Gradient Boosting、XGBoost、CatBoost 进行对比。LightGBM 在 F1‑Macro 为 0.61，单类 F1 分别为 0.17/0.50/0.77，综合得分 S=0.55，推理延迟仅 1.60 ms；相比基线模型在精度、稀有事件识别和低延迟方面均取得显著优势。12 轮提示自优化后，综合预警质量得分从 4.2 提升至 8.9，提升幅度 112%。

**⚠️ 局限性**

主要局限性包括：①仅在拉萨地区训练，缺乏对其他高原城市的地理泛化能力；②提示自适应闭环受预定义评估维度和分阶段框架限制，缺乏对新警情的开放感知与自重构能力；③系统尚未接入实时气象数据流，缺乏对通信中断、传感器噪声或极罕事件的鲁棒性验证。

---

## 159. StochBench: A Domain-Specific Benchmark for Stochastic Processes in Lean

**arXiv ID:** 2609.09264 | [PDF](https://arxiv.org/pdf/2609.09264v1)

**作者:** Idan Davidovich `[一作]` (Case Western Reserve University), Vipin Chaudhary `[通讯]` (Case Western Reserve University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

本文构建了一个包含450个研究生层级随机过程题目的Lean 4基准，并提供对应的自然语言说明。

**💡 创新点**

创新点在于聚焦领域深度、引入直接/抽象化目标、共享定义与基准数据，首次为随机过程领域提供规模化验证平台。

**🔧 技术方法**

采用Opus 4.8代理与Lean LSP MCP服务器，结合工具调用、查询与证明重构实现自动推理。

**📊 数据集**

数据集由450个从概率与随机过程教材、MIT课程笔记与练习选取的问题组成，涵盖马尔可夫链、马氏过程、泊松过程等八大主题。

**📈 对比分析**

通过对每个目标限制15分钟、记录成功率，基准代理获得34.9%的完整证明率；按主题和目标类型划分的细粒度统计进一步展示性能差异。

**⚠️ 局限性**

局限包括人类策划的偏倚、Mathlib缺失的基础设施导致抽象目标失败、以及对证明正确性和语义一致性的人工审核依赖。

---

## 160. Field-level prediction of mid-plane stress tensor fields in concrete target penetration: a cross-velocity graph neural operator surrogate

**arXiv ID:** 2609.10032 | [PDF](https://arxiv.org/pdf/2609.10032v1)

**作者:** Wenpu Du `[一作]` (North University of China), Wenzheng Xu `[通讯]` (North University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

生成400个低速打击案例，构建全尺度聚集体的六组分应力场数据，并用图神经算子实现时间演化预测。

**💡 创新点**

①在低速窗口对终端状态做逐案验证，剔除数值噪声；②提出基于物理加权边权的图神经算子预测完整六组分应力场；③提供公开可复现的跨速率、每种种子数据集。

**🔧 技术方法**

LS‑DYNA高分辨率显式动力学模拟 + 采样+插值构建场；图神经算子（Graph Neural Operator） + 物理启发的边权；自回归 rollout 与多步教师强制。

**📊 数据集**

400个案例（4速度×100随机聚集体）生成的X‑Z面六组分应力场，40帧；附加46个返回标记样本用于终端验证。

**📈 对比分析**

与 Fourier、Wavelet、浅层图网络基线对比；单步相对L2误差0.698，略低于 Fourier 0.691；跨速率LOSO误差0.75；自回归稳定但幅度收敛约0.56，相比真实峰值下降约55%。

**⚠️ 局限性**

单步误差较大；幅度、能量显著被平均抑制导致峰值低、活跃区扩展；缺乏对高压高速范围的泛化；对终端状态依赖于数值稳定性，无法直接给出完整穿透阈值。

---

## 161. A practical DIRECT-type algorithm for medium-scale black-box global optimization

**arXiv ID:** 2609.09796 | [PDF](https://arxiv.org/pdf/2609.09796v1)

**作者:** Linas Stripinis `[一作]` (Vilnius University), Remigijus Paulavičius `[通讯]` (Vilnius University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于 DIRECT 的动态分割与局部搜索混合的全局优化算法，用于盒子约束的黑盒优化问题。

**💡 创新点**

引入动态分割策略以自适应细分潜在最优超矩形，并结合一次性局部搜索，形成 hybrid DIRECT 变体，显著提升了高维问题的收敛速度和求解质量。

**🔧 技术方法**

使用 DIRECT 原理、1‑D surrogate（线性/二次）预测、Pareto 选择、动态分割控制参数以及一次性 hill‑climbing 本地搜索。

**📊 数据集**

在 BBOB 基准库（含 324 个函数、5 维实例及 4 个子集）以及 CEC、ABS、Layeb 等公开测试集合上进行实验。

**📈 对比分析**

与 6 种主流 DIRECT 变体、Naive Multi‑scale Search、以及两种常用的无梯度方法进行对比，采用解的准确率、函数评估次数、执行时间等指标；实验显示新算法在约 40% 的实例上实现最快收敛，17% 的实例上最快运行时间，并在整体上提升约 12% 的可解率与 27% 的解质量。

**⚠️ 局限性**

在平坦或多峰目标函数上，动态分割与局部搜索可能导致不必要的评估；若初始最优估计差，分割过度；对极高维或非光滑情况的性能提升有限。

---

## 162. Safe to Stop? Risk-Constrained Stopping for Sequential Clinical Diagnosis Agents

**arXiv ID:** 2609.09678 | [PDF](https://arxiv.org/pdf/2609.09678v1)

**作者:** Yuexin Wu `[一作]` (University of Memphis), Vasile Rus `[通讯]` (University of Memphis)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了一种基于风险约束的临床诊断停止策略，评估其在主动诊断中的性能与成本平衡

**💡 创新点**

提出了LTT风格的联合检验框架，可在有限样本下对选择性诊断错误率与自主覆盖率提供可审计的置信度；同时引入了稀疏随机混合策略实现成本最小化

**🔧 技术方法**

使用风险排名器（基于梯度提升模型）估计诊断错误概率，构建阈值-时间决策族；采用固定候选族、线性规划混合优化以及二项检验与多重检验（Holm、Bonferroni）等统计方法

**📊 数据集**

基于MIMIC-IV-ED v2.2与MIMIC-IV-Ext-CDS v1.0.2构建的1834个腹痛急诊病例数据集，包含12类测试动作与相应成本

**📈 对比分析**

与最大概率停止、原始LA-CDM停止、置信阈值停止及经验风险最小化等基线对比；实验显示，在相同数据上，最优混合策略在错误率<25%、覆盖率>70%且成本显著低于基线（约12–13单位），同时请求测试次数下降约0.7次

**⚠️ 局限性**

局限在于标签已被访问，缺乏真正前瞻性验证；数据为单中心、缺失记录结构化；缺乏亚组安全性保证；对多重检验依赖于先前冻结，无法提供完全独立的安全证明

---

## 163. Improving 5G AI-RAN MCS Selection by Predicting Retransmissions

**arXiv ID:** 2609.09324 | [PDF](https://arxiv.org/pdf/2609.09324v1)

**作者:** Tamerlan Aghayev `[一作]` (Northeastern University), Tommaso Melodia `[通讯]` (Northeastern University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种可插拔的预测覆盖框架，对现有5G NR链路适配算法进行前瞻性调节，以减少重传并提高吞吐量。

**💡 创新点**

通过仅利用HARQ的历史结果，使用梯度提升树预测下一个帧是否需要重传，并在不改写底层链路适配器的情况下对MCS做微调，形成轻量级、可迁移的预测层。

**🔧 技术方法**

使用梯度提升树（GBDT）作为预测器，ONNX进行低延迟推理；与OLLA、SALAD等主流链路适配算法集成；在X5G实验平台、OpenAirLink衍射器及实际无线硬件上进行验证。

**📊 数据集**

基于X5G OTA测试平台收集的实测HARQ反馈与链路质量数据（约2.4 M帧级样本，70/15/15训练/验证/测试），随后在不同3GPP TDL/ CDL传播模型、SISO/MIMO、行人/车辆移动场景中无重训练验证。

**📈 对比分析**

将overlay与基线OLLA和SALAD在多种信道、天线、速度与OTA场景下对比；在最佳情况下，平均良好吞吐量提升71.5%（相较OLLA）/60.9%（相较SALAD），重传次数下降71.8%；在未见过的信道延迟扩展或频率选择性衰减情形下仍保持5–70%的收益，证明了泛化能力。

**⚠️ 局限性**

仅针对下行链路适配，依赖HARQ历史，无法处理极端快速衰落或极低SNR；模型在极端条件下误判率上升；未验证多用户/多基站场景；未来需扩展至上行、RL等更复杂策略。

---

## 164. Literati: Towards Anytime Optimal Shape Generalized Trees via AO*

**arXiv ID:** 2609.09299 | [PDF](https://arxiv.org/pdf/2609.09299v1)

**作者:** Nakul Upadhya `[一作]` (University of Toronto), Eldan Cohen `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种能够全局最优地学习形状泛化树（SGT）的算法 Literati，能同时决定树结构和分裂函数的复杂度。

**💡 创新点**

创新点在于将 SGT 学习转化为 AND/OR 图搜索，并在 AO* 中加入次优启发式和轮询 AND 选择，从而在保证最优性的同时实现强劲的 anytime 性能。

**🔧 技术方法**

技术上采用 AND/OR 图建模、AO* 全局搜索、次级不合法启发式（CART lookahead）以及轮询 AND 节点探索，并利用自适应离散化减少搜索空间。

**📊 数据集**

实验使用了 24 个规模在 10³–10⁵ 条样本的真实世界表格数据集（包含 QuantBnB 基准）。

**📈 对比分析**

与贪婪、基线和其他最优树方法（ShapeCART、ShapeTAO、CART、AxTAO、Branches、STreeD、LDS‑DL8.5、CADL8.5、ConTree、DPDT）对比，Literati 在训练/测试准确率、运行时和证明率上均表现最佳，尤其在更深树和更高形状复杂度下优势明显。

**⚠️ 局限性**

局限性包括仅适用于表格数据，对样本量极大或时间受限的数据集求解最优性仍有困难，且需要手工设置离散化和正则化超参数。

---

## 165. Early Epistemic Settlement in AI-Assisted Writing

**arXiv ID:** 2609.09332 | [PDF](https://arxiv.org/pdf/2609.09332v1)

**作者:** Han-yu Wang `[一作]` `[通讯]` (University of Hong Kong), Han-yu Wang (University of Hong Kong)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了大型语言模型在写作过程中的“早期知识论定居”现象，即模型提供的组织方案在作者完成自身构建工作之前已足以满足当前写作需求，并探讨其对写作过程和学习的影响。

**💡 创新点**

提出了“早期知识论定居”概念，区分了完成已有组织与改变可形成组织所需关系的两类工作，并阐述了模型建议如何通过置换生成性工作影响写作后续的论证与知识建构。

**🔧 技术方法**

主要使用理论与概念框架，结合写作心理学与认知科学的研究，借助大型语言模型（如 GPT 系列）在示例性写作任务中的交互式反馈作为说明材料。

**📊 数据集**

论文并未使用具体的数据集，而是通过案例分析和理论阐述来说明模型建议对写作过程的影响。

**📈 对比分析**

由于缺乏实验数据，论文未进行性能对比或量化评估，而是通过对比“已可形成组织”与“需要新关系的组织”两种情景来展示早期定居的概念价值。

**⚠️ 局限性**

局限性在于：①缺乏大规模实证验证，难以确定模型建议在不同写作任务与受众中的普适性；②主要为理论性讨论，未提供可复现的实验设置；③对写作质量和学习效果的长期影响尚不明确。

---

## 166. Shift-Accumulate Attention: Multiplier-Free Query--Key Products for Transformer Decoding

**arXiv ID:** 2609.09208 | [PDF](https://arxiv.org/pdf/2609.09208v1)

**作者:** Khubaib Ahmed `[一作]` (University of Wolverhampton), Ahsan Ul haq `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将Transformer解码中的查询‑键乘积改为完全基于位移与累加的运算，通过将键缓存量化为符号幂²码来实现无乘法运算。

**💡 创新点**

提出PoT‑Mk多位数位方案，兼顾存储压缩与精度，并展示在解码时可实现多倍于FP16的吞吐量；同时通过硬件指令模型DS4A证明若有4路位移‑累加指令可进一步提升性能。

**🔧 技术方法**

使用符号幂²量化、移位累加算术、在线shift‑exact softmax、CUDA融合核、KV缓存量化及硬件指令集分析。

**📊 数据集**

TinyLlama‑1.1B‑Chat模型与WikiText‑103文本数据集。

**📈 对比分析**

与标准FP16 SDPA对比，PoT‑4、PoT‑M1、PoT‑M4在512‑32k KV长度下分别提升约3.5–6.5倍吞吐量，且在8存储位时保持与FP16相近的困惑度；若支持DS4A指令，性能可再提升约2–3倍。

**⚠️ 局限性**

受限于当前GPU缺乏4路位移‑累加指令，导致指令发射瓶颈；实验仅在单一模型与单一GPU上验证，缺乏跨模型和多设备的泛化。

---

## 167. SalamandraTA at WMT 2026 Terminology Shared Task: Hard Examples Are Better Teachers

**arXiv ID:** 2609.09999 | [PDF](https://arxiv.org/pdf/2609.09999v1)

**作者:** Xixian Liao `[一作]` (Barcelona Supercomputer Center), Maite Melero `[通讯]` (Barcelona Supercomputer Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于“仅保留模型错误的术语实例”过滤规则的术语翻译训练方法，并将其应用于SalamandraTA‑7B‑instruct v3.0模型，结合文档级推理管道完成WMT26术语翻译任务。

**💡 创新点**

创新点在于识别并仅使用模型误译的术语样本来训练，从而显著提升术语准确率；同时提出两向合成管线生成术语数据，并在推理时采用块级术语过滤与质量估计驱动的后编辑。

**🔧 技术方法**

技术包括大模型Gemma‑4‑31B的双向合成数据生成、硬例过滤、指令调优（instruction tuning）、文档级分块翻译、术语过滤、CometKiwi质量估计与后编辑。

**📊 数据集**

使用MeSpEn医学术语表和EMEA平行语料生成术语‑文本对，合成得到约33,615实例，覆盖29种语言；评测使用WMT2025及WMT2026术语共享任务测试集。

**📈 对比分析**

与WMT26提交榜单比较，SalamandraTA v3.0在术语成功率（94.2%）与chrF++（74.6）均位列前列，仅有两项提交在两项指标上均超越；对比上一年GRPO系统，单纯监督微调方式在同一基准上提升了约2.7%术语准确率和1.1% chrF。

**⚠️ 局限性**

局限性包括过滤规则静态、未动态更新难度、术语检索可能因词形不匹配导致误检、训练数据偏向医学领域、文档级翻译仍需分块导致跨块上下文缺失。

---

## 168. ScopeMamba-YOLO: Widening the Perceptual Scope Inward and Outward for Small Object Detection in Remote Sensing Imagery

**arXiv ID:** 2609.10156 | [PDF](https://arxiv.org/pdf/2609.10156v1)

**作者:** Junjie Fan `[一作]` (Nanjing University of Science and Technology), Yong Qi `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ScopeMamba‑YOLO，一种针对 UAV 与遥感小目标的高效目标检测框架。

**💡 创新点**

核心创新在于“off‑path 零门控选择性扫描”机制，结合 Cascaded Global‑Context Module (CGCM)、Selective‑Scan PAN (SS‑PAN)、Anisotropic Multi‑Scale Strip (AMS) Block 与 Scale‑Adaptive Distribution Focal Loss (SA‑DFL)，实现高分辨率细节与大范围上下文的并行建模。

**🔧 技术方法**

主要技术包括：基于 Mamba 的线性复杂度状态空间模型、方向性条形卷积、非因果四向选择性扫描、分尺度分布式回归（DFL）以及零门控上下文注入。

**📊 数据集**

在 VisDrone‑2019 与 AI‑TOD 两个遥感小目标基准数据集上进行实验验证。

**📈 对比分析**

相较于 YOLOv8s、Mamba‑YOLO、HEdge‑MamYOLO 等同类方法，ScopeMamba‑S 在 VisDrone‑2019 上 mAP_50 达到 50.8%（相当于 YOLOv8s 的 +10.8pp），参数仅 3.57M（YOLOv8s 的 11.10M），同样在 AI‑TOD 上取得显著提升，尤其在非常小与小目标上表现突出。

**⚠️ 局限性**

主要限制：stride‑4 检测路径与完整的选择性扫描导致 FLOPs 较高；实验仅在 640×640 输入、无预训练模型；目前仅适用于 YOLO 风格的单目标检测，需进一步降低延迟、扩展至高分辨率、预训练、方向目标与多模态检测。

---

## 169. Integrating Unimodal and Vision-Language Representations in Latent Space for Multi-Label Chest X-Ray Classification

**arXiv ID:** 2609.09185 | [PDF](https://arxiv.org/pdf/2609.09185v1)

**作者:** Quang-Huy Tran `[一作]` (Ho Chi Minh City University of Technology), Hoang-Anh Ngo `[通讯]` (AK Technologies Company Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种多标签胸部X光分类框架，融合 RAD‑DINO 单模态视觉表示与 BioViL‑T 视觉语言表示，并在 MIMIC‑CXR‑JPG 数据集上实现 14 个临床标签的分类。

**💡 创新点**

创新点：①将两种不同预训练目标（视觉自监督 vs 视觉‑语言）产生的特征分别投射到独立的潜在空间进行细化；②在细化后的三条分支（RAD‑DINO、BioViL‑T、早期融合）上进行混合融合，显著提升性能；③通过系统 ablation 明确潜在空间细化与融合策略对性能的贡献。

**🔧 技术方法**

技术细节：冻结的 RAD‑DINO（ViT‑B/14）和 BioViL‑T（CNN‑Transformer + BERT）编码器；每条分支使用 Autoencoder + 噪声去噪器在 256 维潜在空间进行细化；使用 Transformer Encoder 与门控组合作为分类头；采用多标签损失、AUROC/mAP 评估。

**📊 数据集**

使用的数据集：MIMIC‑CXR‑JPG（377,110 张胸片，211,130 研究，14 个临床观察标签），采用时间拆分的 train/val/test 分区。

**📈 对比分析**

比较方法：与单模态 RAD‑DINO、BioViL‑T、早期融合（CONCAT）以及未细化的混合融合进行对比；最优模型取得 mean AUROC 0.840（±0.001）和 mAP 0.467，显著优于单模态和早期融合，并在 14 标签上与公开方法相当或更好。

**⚠️ 局限性**

局限性：①仅在 MIMIC‑CXR‑JPG 内部评估，缺乏跨机构泛化验证；②标签来源自动提取，严重不平衡；③潜在空间细化的解释与机制仍需进一步探究；④单一测试集的反复使用可能导致探索性结果过拟合；⑤随机种子数量有限，统计功效受限。

---

## 170. Differential Stochastic Simulated Annealing Processor for Fully Connected 2048-Spin Optimization

**arXiv ID:** 2609.09559 | [PDF](https://arxiv.org/pdf/2609.09559v1)

**作者:** Naoya Onizawa `[一作]` (Tohoku University), Takahiro Hanyu `[通讯]` (Tohoku University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一款 2,048 级全连接模拟退火处理器（DSSA），实现了高效的全连接 Ising 计算。

**💡 创新点**

创新点在于差分序列化更新（仅重算翻转的自旋），与 16:1 RNG 共享和自旋选择调度相结合，显著减少交互计算量与功耗。

**🔧 技术方法**

采用 TSMC 28 nm CMOS 设计，16 Mb 4‑bit SRAM 权重存储，差分模拟退火（DSSA）算法，序列化数据通路、RNG 共享、权重读写优先级调度等硬件实现技术。

**📊 数据集**

使用 G‑set（G22、G23、G24、G27、G35、G39、K2000）和 BiqMac Library 的 MAX‑CUT 目标图集进行验证。

**📈 对比分析**

与 STATICA（预估 2,000 级）和 GPU SSA 对比，DSSA 在 500 MHz 下实现 0.15 W/自旋、2.7 TTS、0.86 mJ 能量/解决方案；功耗比 STATICA 低 1.5×、TTS 能量低 3.5×，相较 GPU 典型基线能量降低 5 个数量级。

**⚠️ 局限性**

主要限制是仅基于后布局仿真，未进行硅验证；权重存储呈 O(N²) 规模，对极大规模问题或高翻转率场景的优势有限。

---

## 171. Structure-Aware Unsupervised Anomaly Detection for Spacecraft Telemetry with Adaptive EVT Thresholding

**arXiv ID:** 2609.10017 | [PDF](https://arxiv.org/pdf/2609.10017v1)

**作者:** Óscar Alcarria `[一作]` (Universidad de Castilla-La Mancha), José M. Puerta `[通讯]` (Universidad de Castilla-La Mancha)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个完全无监督、部署即用的结构感知异常检测框架，能够在仅有一个月操作历史且无标签、无GPU的条件下，对航天器遥测进行实时异常检测。

**💡 创新点**

创新点在于：①基于任务级统计特征自适应选择检测模型（M1 使用 LSTM 自编码器，M2 使用聚类+PCA+LOF）；②采用增量式每月重训练，保持对非平稳过程的自适应；③使用极值理论（EVT）动态阈值控制误报，实现精度优先的事件级异常检测。

**🔧 技术方法**

技术包括：时序序列预处理（零阶保持、稀疏插值）、轻量级 LSTM 自编码器、PCA 降维、K‑means 聚类、局部异常因子（LOF）、极值理论（GPD）阈值自适应、事件聚合后处理。

**📊 数据集**

使用 ESA Anomalies Dataset（ESA‑AD）两条任务，M1 76 通道（58 目标），M2 100 通道（47 目标），包含正常、罕见和异常事件。

**📈 对比分析**

在严格的时间序列增量评估下，与工业无监督基线相比，M1 的 F0.5 提升至 0.700（从 0.424），M2 保持竞争力（0.698 vs 0.882）。与有标签监督方法相比，误差差距显著缩小，尤其是 M1 的 0.700 对比 0.786 的监督上限。

**⚠️ 局限性**

局限性包括：对统计特征的依赖可能不适用于完全不同的任务；缺乏标签的情况下阈值和模型选择仍是经验式，难以进一步优化；仅在 ESA‑AD 上验证，未测试跨任务泛化；计算成本较高（M1 每月约 1040 秒训练）。

---

## 172. Positional task conditioning for scalable defect detection across product families in large product catalogs

**arXiv ID:** 2609.09567 | [PDF](https://arxiv.org/pdf/2609.09567v1)

**作者:** Soham Satyadharma `[一作]` (Amazon Catalog AI), Suleiman A. Khan `[通讯]` (Amazon Catalog AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将多标签检测拆分为三分类子任务，并利用位置任务条件化（Positional Task Conditioning）蒸馏技术，检测产品家族中的不一致性并显著提升准确率与成本效益。

**💡 创新点**

创新点在于：①将多标签问题拆解为关注度更高的子任务，②提出在蒸馏过程中在提示结构中插入任务标记的 Positional Task Conditioning 方法，③在单个小模型中实现多任务判别，显著提升性能并降低成本。

**🔧 技术方法**

采用分解式检测（Divide-and-Conquer）、位置任务条件化蒸馏、LLM（Claude Sonnet 4.5、Qwen3、Mistral 等）与 QLoRA、vLLM 等实现细节。

**📊 数据集**

使用真实人工标注的 2,296 家族-属性对（约 390 错误实例）以及 11,130 家族-属性对的合成数据，样本来源于多国英文产品目录。

**📈 对比分析**

与单体提示相比，F1 从 52% 提升到 87%；与 RBD 对比，PTC 在五个模型上平均提升 1.5%–6.3% F1，整体 F1 约 84%–88%；成本降低约 98%，单体提示与 D&C 仅在单次调用成本差异可忽略。

**⚠️ 局限性**

局限性：仅在单一域（产品家族不一致）验证；任务数扩大时是否可持续尚未探究；未对训练数据规模与样本效率进行消融；数据集为专有，公开性受限。

---

## 173. Evidence-Order Calibration for Selective Visual Reasoning under Progressive Loss of Question-Critical Evidence

**arXiv ID:** 2609.09184 | [PDF](https://arxiv.org/pdf/2609.09184v1)

**作者:** Muhamathu Ameer Ali Aacaas Muhamath `[一作]` `[通讯]` (University of Moratuwa), Muhamathu Ameer Ali Aacaas Muhamath (University of Moratuwa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉语言模型上构建逐步的局部证据缺失轨迹，并评估模型的可靠性评分是否随证据丢失呈单调下降。

**💡 创新点**

提出了证据序列一致性评价指标（EMVR）并设计了基于顺序监督的轻量后置可靠性头，验证其对单调性和跨降解迁移的有效性。

**🔧 技术方法**

使用冻结的Qwen2.5-VL-3B-Instruct，提取隐藏状态、序列置信度和熵，并训练线性后置头加入对偶Hinge损失。

**📊 数据集**

基于GQA场景图生成的176条受控轨迹（共880个遮挡条件），以及对照的非关键区域遮挡与局部高斯模糊。

**📈 对比分析**

与原始置信度及BCE后置头对比，顺序监督将EMVR从0.330降至0.303，且在未见模糊降解上也显著降低，尽管AUROC、Brier等传统指标提升不显著。

**⚠️ 局限性**

实验规模有限（仅176轨迹、单一VLM、局部遮挡与模糊），对关键证据定义与数据质量有限验证，未提升选择性风险排名。

---

## 174. Beyond Similarity: Foundation Models as an Efficient Backbone for Training-Free Composed Video Retrieval

**arXiv ID:** 2609.10008 | [PDF](https://arxiv.org/pdf/2609.10008v1)

**作者:** Dmitry Demidov `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Rao Anwer `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了CoVRAGE，一个完全无训练的分层级管道，利用冻结的基础模型在复合视频检索中实现自适应检索、重排序、分解与验证。

**💡 创新点**

创新点在于置信度门控的深度适配与阶段感知的时间帧采样，将不同功能分配给可重用编码、轻量级重排序、文本生成和严格多模态验证，实现大规模图库下可扩展检索与细粒度推理。

**🔧 技术方法**

使用了冻结的多模态编码器、置信度门控的候选扩展、目标描述生成、基于时序的新颖帧采样、时间戳感知验证以及多阶段接口设计。

**📊 数据集**

在WebVid-CoVR、Dense-WebVid-CoVR和CoVR-R三个公开基准上进行评估。

**📈 对比分析**

与其他训练-free或零样本方法对比，CoVRAGE在Dense-WebVid-CoVR上达到89.55 R@1，在CoVR-R上达到93.43 R@1，分别高出约+35%和+25%绝对值。

**⚠️ 局限性**

局限性在于置信度校准与候选预算的自适应策略尚未充分优化，且对极短或极复杂编辑的鲁棒性有待进一步验证。

---

## 175. Design and Attitude Control of an Underwater Quadruped Robot

**arXiv ID:** 2609.09217 | [PDF](https://arxiv.org/pdf/2609.09217v1)

**作者:** Davide Molinaroli `[一作]` (Norwegian University of Science and Technology), Kostas Alexis `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计、建模并实验验证了一种低成本的水下四足机器人，通过球形末端的拖曳力实现姿态控制。

**💡 创新点**

创新点在于：①使用可重复制造的POM密封壳和商用伺服电机实现开放源代码的低成本水下密封方案；②提出一种简化的漂浮基动力学模型，忽略附加质量但足以设计SO(3)闭环姿态控制器；③利用球形末端产生均匀拖曳力进行姿态控制，并将总力矩分配到腿部速度命令；④引入功率-恢复循环实现有限工作空间内的连续扭矩输出。

**🔧 技术方法**

技术手段包括：SO(3)姿态误差计算与PID控制、虚拟功与浮动基雅可比矩阵、五杆链腿部正逆运动学、拖曳力与浮力模型、抗饱和积分调节、伺服电机控制与ROS1实现、使用VectorNav VN-100 IMU、Jetson Orin NX计算平台、以及在MuJoCo中采用惯性基流体模型进行仿真。

**📊 数据集**

未使用公开数据集；通过CFD（SIMPLE算法）估算球形末端的阻力系数，并在水槽实验中自行采集姿态、角速度、伺服角度等数据。

**📈 对比分析**

通过仿真与水下实验对比评估控制性能。指标包括上升时间、稳态时间、稳态误差和超调量。实验结果显示：在30°至75°的姿态设定下，上升时间在2–13 s之间，稳态时间在23–64 s，稳态误差在0.3–2.9°，超调率一般低于20%。较大角度（90°）时因耦合与gimbal lock导致性能下降。

**⚠️ 局限性**

主要局限包括：①腿部相位协调不足导致扭矩空缺和更长的稳态时间；②简化模型忽略附加质量、升力和伺服动力学，导致仿真与实机差异；③功率-恢复循环受限的工作空间限制了姿态控制速度；④伺服带宽有限，无法即时跟随高频速度命令；⑤缠绕与水深变化对阻力系数的影响未建模。

---

## 176. Efficient Fairness Auditing Across Guidance Scales in Text-to-Image Diffusion Models via Causal Abstraction

**arXiv ID:** 2609.09486 | [PDF](https://arxiv.org/pdf/2609.09486v1)

**作者:** Nabila Tasfiha Rahman `[一作]` (University of Arkansas), Lu Zhang `[通讯]` (University of Arkansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于因果抽象的审计工具，用于在不重复完整扩散过程的前提下，评估文本到图像扩散模型在不同 CFG 指导尺度下的公平性；

**💡 创新点**

创新点在于将因果抽象与可识别的审计查询相结合，构建高层结构因果模型和变换器神经因果模型，并给出在信息损失下仍能保留干预分布的正确性保证；

**🔧 技术方法**

主要技术包括结构因果模型、部分投影 C‑DAG、Transformer 神经因果模型、Wasserstein 距离评估、Bootstrap 置信区间、CLIP 属性分类器、Stable Diffusion 与 StayFair 方案；

**📊 数据集**

实验数据涵盖三条性别中立提示（律师、图书管理员、科学家），使用 Stable Diffusion v1.5 生成 1000 条随机种子轨迹（800 训练、200 验证），并在不同 CFG 尺度下进行属性评分；

**📈 对比分析**

通过将高层模型与低层模型在最终属性分布上的 Wasserstein‑1 距离进行对比，发现 75% 的配置满足误差比例 ≤1；在计算效率上高层模型实现了约 19.4 倍的速度提升（94.9% 运行时间缩减），并在公平性评估上与低层模型保持一致；

**⚠️ 局限性**

局限性包括：因果抽象的 lossy 映射可能导致 AIC 违反，导致部分因果信息丢失；实验仅覆盖单一属性与少数提示；仅在 Stable Diffusion 及其 StayFair 变体上验证，未扩展到更广泛的提示分布或多属性场景；

---

## 177. HLSFactory-Agent: Large-Scale Agentic HLS Dataset Construction from Academic and Open-Source Projects

**arXiv ID:** 2609.09519 | [PDF](https://arxiv.org/pdf/2609.09519v1)

**作者:** Kaushik Chandana `[一作]` (Georgia Institute of Technology), Callie Hao `[通讯]` (Georgia Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个基于LLM的代理HLSFactory-Agent，自动从学术论文和开源仓库中提取可综合的独立HLS设计，并提供论文索引脚本以加速仓库发现。

**💡 创新点**

将LLM代理与Pi框架结合，在Docker容器内完成自动化设计提取和验证，首次实现大规模、可复现的HLS数据集构建流程。

**🔧 技术方法**

使用LLM（如GPT-4）作为代理核心，搭配Pi Agent框架、Clang编译器、Vitis HLS、Bash脚本以及Docker容器实现工具调用与结果验证。

**📊 数据集**

利用收录的2,517篇潜在HLS论文和26个公开仓库作为测试集，最终提取了271个候选设计，其中130个通过Vitis HLS验证。

**📈 对比分析**

通过将提取率、验证通过率与代理运行时长/成本进行对比，发现设计提取与推理成本正相关，平均每个设计约需X美元/分钟，验证通过率约为48%。

**⚠️ 局限性**

局限性包括仍需人工审查论文和仓库、对Vitis HLS兼容性的依赖导致部分设计失败、Docker容器内缺乏完整HLS验证以及对复杂依赖或生成代码的处理不够完善。

---

## 178. A Risk-Sensitive and Uncertainty-Aware Decision-Making and Control Framework for Safe and Robust Autonomous Driving

**arXiv ID:** 2609.09650 | [PDF](https://arxiv.org/pdf/2609.09650v1)

**作者:** Zhuoren Li `[一作]` (Tongji University), Bo Leng `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了RUDC框架，将风险敏感分布式强化学习与集成式策略不确定性量化相结合，并通过不确定性感知的高阶控制障碍函数实现安全修正，提升无人驾驶在未信号化交叉口的决策与控制安全性、效率和鲁棒性。

**💡 创新点**

① 统一量化尾部风险与策略不确定性；② 引入不确定性感知的HOCBF，自适应调整安全约束严格度并补偿模型误差；③ 采用CVaR分布式RL与深度集成网络实现风险敏感决策；④ 通过残差预测器补偿CBF离散化误差，整体实现安全过滤与决策的协同。

**🔧 技术方法**

风险敏感分布式RL（CVaR+分位数回归）、深度集成网络与随机先验函数、深度强化学习策略、基于单轨动力学的高阶控制障碍函数（ECBF/TTCBF）、不确定性感知的高阶CBF、残差预测器（MLP）、实时QP求解、注意力融合感知编码。

**📊 数据集**

使用仿真交叉口环境（基于Highway-Env）生成的无人车与周围车辆数据，训练/测试任务包括随机目的地与高密度左转交叉口，数据来自仿真模拟，未使用真实道路数据。

**📈 对比分析**

与QPSL、Recovery RL、FAC、USL、Lagrangian、vanilla SAC、DSAC等安全RL基线在随机目的地和高密度左转两种测试中进行对比。RUDC-T/E在成功率、违章率、平均奖励、最小距离等指标均优于基线，尤其在OOD与长尾场景下保持高成功率、低违章率，且实时计算时间≤60 ms，满足10 Hz的控制频率。

**⚠️ 局限性**

当前CBF仅基于即时状态-动作，缺乏历史上下文，可能导致校正与原始策略偏差大；未考虑输入约束下的CBF，可能出现物理约束冲突；缺乏真实道路实验与异质交通参与者的验证，极端突发情况的鲁棒性尚未充分评估。

---

## 179. IAE-VTG: Interaction-Aligned Action-Entity Video Temporal Grounding

**arXiv ID:** 2609.09736 | [PDF](https://arxiv.org/pdf/2609.09736v1)

**作者:** Shiwen Zhao `[一作]` (University of Sydney), Martin R. Oswald `[通讯]` (University of Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过细粒度分离动作和实体信息并结合运动与外观特征，提出IAE‑VTG模型实现了视频‑文本交互一致性语义对齐，从而显著提升视频时序定位精度。

**💡 创新点**

创新点在于提出Fine‑grained Disentangled Interaction Module (FDIM)与Interaction‑Sensitive Assignment (ISA)，在表示层和训练分配层同时显式建模动作‑实体交互一致性，解决传统方法中动作与实体独立关注导致的定位误差。

**🔧 技术方法**

采用双流编码（运动与外观）、跨模态注意力、语义匹配、DETR式集合预测以及交互敏感的匈牙利匹配等技术，构建完整的交互感知VTG框架。

**📊 数据集**

在QVHighlights、Charades‑STA和TACoS三个标准VTG基准数据集上进行实验。

**📈 对比分析**

与FlashVTG、DualGround、KDA等最先进方法比较，IAE‑VTG在R1、mAP、mIoU等多项指标上均达到或接近最高分，尤其在严格IoU阈值下表现突出。

**⚠️ 局限性**

主要局限包括对词性标签的依赖，角色标注错误会导致性能下降；对更复杂的多语义关系、长段落式查询支持有限，未来需要进一步扩展至图结构推理等方向。

---

## 180. AnimalLift: Reconstructing Animatable 3D Animals from a Single Image by Learning Canonical Shape, Texture, and Fur Maps

**arXiv ID:** 2609.09513 | [PDF](https://arxiv.org/pdf/2609.09513v1)

**作者:** Chunyi Sun `[一作]` (Australian National University), Stephen Gould `[通讯]` (Australian National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 AnimalLift，利用单张图片恢复可动画、可编辑且含显式毛发的 3D 动物模型，输出共享拓扑、UV 对齐的几何、纹理和毛发地图。

**💡 创新点**

创新点：① 在共享 UV 统一空间内同时预测几何、纹理和毛发；② 使用 UV 对齐的毛发地图编码根点相对 3D 位移，保证在变形和物理模拟中的一致性；③ 通过可渲染的流匹配（flow matching）实现纹理生成；④ 设计了面向单图的程序化数据生成管线，提供对齐的几何、纹理与毛发监督。

**🔧 技术方法**

技术：DINOv2 ViT 主干 + 分支头（几何、毛发、纹理流匹配），轻量化 UNet 解码器，流匹配网络，纹理自编码器，基于 SMAL 的骨架和 UV 布局，GPU 并行训练。

**📊 数据集**

数据集：合成的 11K 训练样本（六类动物：狗、猫、狐狸、狼、熊、狮子），每个样本包含渲染图、统一拓扑几何、UV 纹理、毛发密度和 UV 对齐毛发张量；真实图像测试集 640 张来自 Oxford‑IIIT Pet 与 Animals with Attributes 2。

**📈 对比分析**

对比方法：Fauna‑3D、Hunyuan‑3D 2.1、Trellis 2、AniMer、BITE。AnimalLift 在合成数据上获得最佳几何 Chamfer‑L1，且在纹理与毛发方面实现显著低 Patch‑FID 与高 LPIPS/MV‑CLIP；在真实图像上也保持了竞争力，且支持动画、毛发编辑与物理模拟，超越基线仅生成姿态对齐几何或缺失毛发的方案。

**⚠️ 局限性**

局限：仅适用于共享拓扑的四足动物，无法处理非四足或拓扑差异较大的种类；单张图片导致遮挡区域不确定，恢复为类别先验的平均结果，细节如伤痕、湿毛等缺失；需要手工创建的基础模板，虽然是一次性工作，但仍是门槛。

---

## 181. Arti-JEPA: Adapting Video World Model to Real-Time MRI of the Vocal Tract for Speech-Production Analysis

**arXiv ID:** 2609.09757 | [PDF](https://arxiv.org/pdf/2609.09757v1)

**作者:** Hong Nguyen `[一作]` (University of Southern California), Shrikanth Narayanan `[通讯]` (University of Southern California)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在未标注的实时MRI（rtMRI）口腔运动视频上，继续训练 V-JEPA 2 的联合嵌入预测目标，得到自适应编码器 Arti-JEPA，并冻结该编码器用于多任务评估（跨域音素预测、流利与失流检测、舌癌术前后转移分析）。

**💡 创新点**

① 将大规模互联网视频预训练的隐空间预测模型迁移到极低分辨率、单片灰度的 rtMRI；② 在无标签条件下实现连续预训练，显著提升跨域音素识别与临床任务；③ 通过冻结特征在舌癌患者上保持音素可辨识性，揭示解剖改变对可辨识性的影响。

**🔧 技术方法**

采用 V-JEPA 2（隐空间预测）与 EMA 目标编码器、遮挡掩码多块策略、三维旋转位置编码；对视频进行 50 fps 采样、空间重采样、灰度复制；冻结编码器后使用轻量级注意力池化+BiLSTM 读出；评估指标包括 Cohen's κ、PER、宏 F1 等。

**📊 数据集**

训练集：USC 75‑Speaker Speech MRI（≈15 h）+ 纵向 rtMRI 语料（≈38 h）；评估集：Annot‑16（16 讲者）、USC LSS（单讲者）、Glossectomy（3 讲者术前后）、Stuttering（7 讲者）。

**📈 对比分析**

与未适配的 V-JEPA 2、VideoMAE（像素重建）以及 ImageNet‑监督图像编码器比较；跨域音素 κ 从 0.25 提升至 0.35，帧级音素 κ 最高 0.505；视频模型优于图像模型；在舌癌患者音素可辨识性保持 0.28–0.45，手术后略降；流利/失流检测宏 F1≈0.82，类型分类性能较低。

**⚠️ 局限性**

① 训练资源受限（单 GPU、批量小）导致模型可能未达到最佳；② 隐空间预测窗口固定 640 ms，长时序任务受限；③ 预训练对不同扫描协议、解剖变异的泛化尚未完全验证；④ 舌癌患者样本仅 3 人，无法建立系统性结论；⑤ 类型分类难以提升，表明缺乏对语音信息与视频信息对比的进一步实验。

---

## 182. Geometric organization of olfactory descriptor data in the Poincaré disk

**arXiv ID:** 2609.09573 | [PDF](https://arxiv.org/pdf/2609.09573v1)

**作者:** Aniss Aiman Medbouhi `[一作]` (KTH Royal Institute of Technology), Danica Kragic `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过两维Poincaré盘的超曲线多维尺度映射（Hyperbolic MDS）对嗅觉描述符数据进行低维可视化，揭示了半径与描述符信息熵之间的关联以及角度方向与连续或二值描述符的梯度/类别结构；

**💡 创新点**

创新点在于首次将超曲线几何应用于嗅觉描述符空间，证明半径可捕捉全局描述符分布信息，角度可捕捉连续描述符梯度或二值标签的类别聚集，并通过多种鲁棒性检验验证这一结构的普遍性；

**🔧 技术方法**

使用的技术包括超曲线度量多维尺度映射、Riemannian Adam优化、软最大熵计算、正交化处理、Tangent空间线性回归与超曲线核密度估计；

**📊 数据集**

使用的两个数据集为：Sagar 160种单分子气味的3位受试者连续描述符评分（共480条观测）和GoodScents–Leffingwell 4983种分子专业二值标签（共138个描述符）；

**📈 对比分析**

方法通过保留距离的Pearson/Spearman相关性评估嵌入质量，并采用受限置换检验（WS、OB、mol）验证半径-熵及角度-R²统计显著性。实验显示Sagar中半径与评分熵负相关（Pearson≈-0.77，p<0.001），角度对甜、麝香、果香、愉悦等描述符有显著方向性（R²>0.4，p<0.001）；GSLF中半径与活跃标签熵正相关（Pearson≈0.87，p<0.001），正交熵负相关；二值标签在盘内形成高密度角度聚集区。

**⚠️ 局限性**

局限包括：仅基于描述符数据，缺乏神经或行为层面的验证；Sagar受试者人数少，个体差异未能全面覆盖；GSLF标签稀疏导致距离保留度低；角度聚集分析为定性可视化，未提供统计聚类度量；未整合分子结构信息或更深层次的描述符层级。

---

## 183. GTA-2: A Multi-VLM Framework for Synthesizing Robot Manipulation Skills via Grounded Task Axes

**arXiv ID:** 2609.09808 | [PDF](https://arxiv.org/pdf/2609.09808v1)

**作者:** M. Yunus Seker `[一作]` (Carnegie Mellon University), Oliver Kroemer `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个多 VLM 框架 GTA-2，能够从自然语言指令和 RGB‑D 观测自动构建可执行的机器人操作技能，并支持针对性的人机反馈进行局部修正。

**💡 创新点**

创新点在于将任务分解、控制器组合、参数设置和视觉定位四个阶段拆分为独立的 VLM 代理，使用任务轴（Task‑Axis）表示法显式记录控制目标，使得技能生成可解释、可重用且可局部调优；并通过零样本生成和多阶段反馈显著提升了泛化性能。

**🔧 技术方法**

技术包括：多代理 VLM（使用 Gemini 3.1 Pro）实现任务分解、控制器生成、参数赋值和视觉定位；任务轴控制器库（位置、姿态、力、抓手等多种控制器）；基于 RGB‑D 的视觉定位与点云几何运算；以及确定性编译器将已定位的控制器序列转化为可执行机器人脚本。

**📊 数据集**

使用了 14 种真实机器人操作任务（包括捡放、关节物体交互、工具使用、接触丰富操作等），在每个任务上进行 15~20 次独立实验；视觉输入来自 ZED 2i 立体相机的 RGB‑D 图像。

**📈 对比分析**

与 VLA 基线 π_0.5 以及两种 Code‑as‑Policies（CaP‑TAC：任务轴控制器，CaP‑Primitive：传统机器人原语）进行对比；零样本平均成功率 73.9%，比最强基线提升 31.4 百分点；经过一轮/两轮人机反馈后成功率分别升至 86.2% 与 99.0%，一次反馈即可将绝大多数失败转为成功。

**⚠️ 局限性**

局限性包括：受限于任务轴表示和控制器库的表达能力，无法描述超出其定义的控制目标；假设所有必要的场景特征可从单张 RGB‑D 观测中获取，且未实现在线重定位；在一次执行后若场景发生变化需重新定位；实验仅在单一桌面机器人/摄像头配置下验证，未充分测试多种机器人/感知环境；缺乏自动化的执行监测与迭代优化机制。

---

## 184. What Should an Agent Forget? Separating What Is Stored from What Is Used

**arXiv ID:** 2609.10263 | [PDF](https://arxiv.org/pdf/2609.10263v1)

**作者:** Yuhang Li `[一作]` (Beihang University), Yuchen Li `[通讯]` (East China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于查询条件的记忆视图，将存储历史与答案时使用的证据分离，实现可选择性遗忘与记忆检索。

**💡 创新点**

创新点在于使用同槽替换与历史意图机制，允许被替代的事实在需要时恢复，并通过率失真优化控制答案上下文大小。

**🔧 技术方法**

采用语言模型驱动的证据抽取、同槽替换判定、贪婪打包、以及基于预算的率失真框架来构建查询时的记忆视图。

**📊 数据集**

在 AMB‑Text、LME‑KU、MAB‑FC、BEAM 和 PersonaMem 等四大问答与对话记忆基准数据集上进行评估。

**📈 对比分析**

与 ACE、ReasoningBank 等基线对比，RD‑Forget 在四种大型语言模型上均取得 10–30% 的精度提升，平均约 86% 的答案准确率。

**⚠️ 局限性**

局限性包括：依赖冻结语言模型进行抽取与判断，模型与任务迁移性待验证，且对更复杂的多模态或非问答任务尚未深入探测。

---

## 185. Where Does the Human End? Creative Agency with Generative AI across Five Years of Chinese Digital Painting

**arXiv ID:** 2609.09333 | [PDF](https://arxiv.org/pdf/2609.09333v1)

**作者:** Yibo Meng `[一作]` (Cornell University), Chengxi Zang `[通讯]` (Cornell University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开展了为期五年的纵向访谈研究，跟踪17名中国数字画师与生成式AI的交互与创作代理分配的演变

**💡 创新点**

提出“纵向代理分配”概念，揭示创作者在技术、社会与情感条件下对AI角色的反复调整与再分配

**🔧 技术方法**

采用半结构化访谈、主题编码与纵向案例分析等质性研究方法，无技术实验或模型训练

**📊 数据集**

访谈数据：17名参与者在2021-2025年共5轮访谈，涵盖职业身份、工作流程、版权、情感等方面

**📈 对比分析**

无对比实验或性能指标，研究通过跨年叙事对比来阐释代理分配的变化，未涉及数值评估

**⚠️ 局限性**

局限包括样本量有限、研究对象局限于中国数字画师、研究者与受访者回忆偏差、纵向访谈可能导致叙事重构以及文化与领域的可推广性受限

---

## 186. Beyond Accuracy: ARIA-Rubrics for Evaluating Audio Reasoning in Large Audio Language Models

**arXiv ID:** 2609.09681 | [PDF](https://arxiv.org/pdf/2609.09681v1)

**作者:** Yupei Li `[一作]` (Imperial College London), Björn Schller `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ARIA-Rubrics 框架，对大型音频语言模型的推理链进行无标注、自动化、可解释的评估；

**💡 创新点**

创新点包括：①首次在音频推理中引入无注释金标准推理链；②构建六维度（感知对齐、步骤连贯性、内容实质、推理进展、答案一致性、音频词汇稠密度）自动化指标；③利用 Chain-of-Thought 模板外化推理过程，实现透明评估；

**🔧 技术方法**

使用了 Chain-of-Thought 提示、CLAP 对齐模型、句向量相似度、NLI 分类器、轻量 LLM 评分、词向量相似度等技术；

**📊 数据集**

使用 MMAR 与 MMAU-mini 两个 1000 条多选音频推理基准进行实验；

**📈 对比分析**

在 9 个 LALM（开源与闭源）上对比 ARIA 分数与准确率，发现 ARIA 能区分三种推理模式；ARIAscore 与人类评估相关性 0.671，优于单一指标，并与 LLM judge 结果高度一致；

**⚠️ 局限性**

局限性包括：①轻量评分模型可能影响精度；②人类评估样本规模有限；③仅评估两闭源模型，缺乏更广泛的商业模型覆盖。

---

## 187. Valerant: An Automatic Navigable Game Map Generator via Action-Conditioned World Model Exploration

**arXiv ID:** 2609.09418 | [PDF](https://arxiv.org/pdf/2609.09418v1)

**作者:** Yiran Qiao `[一作]` (Case Western Reserve University), Jing Ma `[通讯]` (Case Western Reserve University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用冻结的动作条件视频世界模型与视觉SLAM，构建训练无关的自动探索框架 Valerant，能够从单帧截图生成可导航的 3D 游戏地图。

**💡 创新点**

将动作条件世界模型转化为 World Action Model 并引入对比性“何若”回滚行动、鲁棒地板估计、碰撞检测以及死角回溯等机制，实现无训练的自动化地图生成。

**🔧 技术方法**

动作条件视频扩散模型（Matrix-Game 3.0）、MASt3R-SLAM、流匹配（flow-matching）视频生成、对比性探索策略、视觉 SLAM、碰撞检测、地板估计与回溯检查点等技术。

**📊 数据集**

VALORANT 游戏的截图，单帧图像作为初始输入。

**📈 对比分析**

与 HY-World 2.0 进行比较，使用真实性评分和人类偏好比例两种评估指标。Valerant 在视觉真实性和用户偏好方面均优于 HY-World 2.0。

**⚠️ 局限性**

生成的点云几何一致性仍有限；单帧输入导致多模态不确定性；对视觉幻觉和复杂交互的鲁棒性尚需提升。

---

## 188. Link prediction in complex networks via fusing node centrality and local similarity indices

**arXiv ID:** 2609.09658 | [PDF](https://arxiv.org/pdf/2609.09658v1)

**作者:** Yingying Zhang `[一作]` (China Jiliang University), Chengye Zhao `[通讯]` (China Jiliang University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了将节点中心性与局部相似性融合的框架，用PageRank和DomiRank作为中心性，实现了统一加权的局部-全局结合式链接预测方法。

**💡 创新点**

创新点在于统一权重的DomiRank融合（DR-MD）系列显著优于基于PageRank的PR-MD，且该框架可轻松扩展至其他中心性；并系统验证了七种局部相似性指数的普适提升。

**🔧 技术方法**

技术包括PageRank、DomiRank闭式解、七种经典局部相似性（CN, AA, RA, JC, SO, HPI, PA），加权融合公式与分段完成规则，五折交叉验证、Wilcoxon签名秩检验。

**📊 数据集**

使用九个真实网络数据集（政治书籍、神经网络、交通、社交、学术、食物网、爵士乐、代谢、食物网等，节点数≤500）。

**📈 对比分析**

通过与七种局部基线、三种全局方法（Katz, RWR, SimRank）以及四种高级方法（LNB, CN2D, CNC, CND）比较，DR-RA平均AUC达0.70以上，显著高于所有对手；所有DR-MD在所有数据集上均优于对应PR-MD，且统计显著。

**⚠️ 局限性**

局限包括规模有限（≤500节点）、需手工设定权重参数、对不同局部相似性尺度敏感、仅针对无向无权网络、需要进一步验证大规模网络下的可扩展性和负样本采样协议的通用性。

---

## 189. Video-MOPD: Multi-Teacher On-Policy Distillation for Video Understanding

**arXiv ID:** 2609.09300 | [PDF](https://arxiv.org/pdf/2609.09300v1)

**作者:** Zhenxin Qin `[一作]` (Tongji University), Lin Ma `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了三个针对视频理解不同子任务的专家模型（一般视频理解、视频时序定位、视频 STEM 推理），随后通过多教师在策略上的对策蒸馏（Multi‑Teacher On‑Policy Distillation, MOPD）将其知识集成到单一的统一模型 Video‑MOPD 中，此外提出了可靠性感知信息采样（RAIS）提升蒸馏效果。

**💡 创新点**

①将任务级强化学习与专家蒸馏相结合，实现了“先专精后统一”的范式；②利用 MOPD 在学生自身生成的轨迹上给教师监督，避免离线轨迹的分布偏差；③引入 RAIS 通过教师一致性与学生差距双重度量筛选样本，显著提升蒸馏效率。

**🔧 技术方法**

基于 Qwen3‑VL‑8B‑Instruct 的参数初始化，分别对三类专家进行任务专属的强化学习（GRPO、时间目标奖励、Wasserstein 奖励等）；随后使用多教师在策略蒸馏，采用 token‑级反向 KL 损失；采样策略使用 RAIS；最终统一模型在单一前向推理中完成所有任务。

**📊 数据集**

一般视频理解：NeXT‑QA、LongVideo‑Reason、STAR、LLaVA‑Video‑178K、Holmes‑train、PerceptionTest、CLEVRER、SR‑91k；视频时序定位：TimeLens2‑93K、TimeLens‑100K、Ego4D‑NLQ；视频 STEM 推理：Orsta47K、virl39K；用于蒸馏的混合数据集包含视频、图像及时序定位样本。

**📈 对比分析**

在七个公开基准（MVBench、MMVU、Video‑MME、VideoMMMU、Video‑Holmes、TimeLens、TempCompass）上与 Qwen3‑VL‑8B‑Instruct 及多种先前模型对比，Video‑MOPD 在总体平均分上提升 5.4 分（至 69.12），并在 6/7 个基准上领先或同级；相较于直接参数平均，提升 1.84 分，证明策略空间蒸馏的有效性。

**⚠️ 局限性**

目前仅针对中短视频，未充分探索长时序视频的稀疏证据与分布式推理；蒸馏过程依赖教师的一致性与奖励设定，可能受限于数据质量；模型规模相对较大，部署成本较高。

---

## 190. AutoTrans: AI-Assisted Automatic Translation of Security Assertions for RISC-V Processors

**arXiv ID:** 2609.10057 | [PDF](https://arxiv.org/pdf/2609.10057v1)

**作者:** Sharjeel Imtiaz `[一作]` (Tallinn University of Technology), Tara Ghasempouri `[通讯]` (Tallinn University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

AutoTrans框架实现了RISC-V安全断言在不同微架构之间的全自动翻译与验证，避免了手工重写。

**💡 创新点**

创新点包括：① 基于正则表达式的轻量级信号提取器，保证LLM提示无幻觉；② 固定模板与锁定推理参数，保证提示字节一致，提升可复现性；③ 在翻译后通过QuestaSim和JasperGold FPV两道门槛，确保断言形式正确且非空洞。

**🔧 技术方法**

主要技术：正则表达式解析、Python实现的信号提取、固定模板Prompt构造、NVIDIA DeepSeek V4-Flash/Pro LLM推理、QuestaSim语法验证、JasperGold形式验证。

**📊 数据集**

使用NS31A RISC-V处理器的1146条安全断言（68组）作为源数据，目标为lowRISC Ibex处理器的9个安全模块。

**📈 对比分析**

与Transys、AutoAssert、AssertLLM等方法对比，AutoTrans不需GPU、无结构图匹配、可复现且通过FPV门槛；实验中Auto TAR为78%，最终TAR达100%，相较于以往半自动方法覆盖模块更多、时间大幅缩短。

**⚠️ 局限性**

局限性：仍需手工干预处理FPV检出的结构性错误；当前仅针对RISC-V处理器微架构，尚未验证跨架构不同ISA的适用性；模型推理受限于云服务的可用性。

---

## 191. SocialRL: Refining LLMs' Social Intelligence through Multi-turn Reinforcement Learning and Reward Design

**arXiv ID:** 2609.09764 | [PDF](https://arxiv.org/pdf/2609.09764v1)

**作者:** Jianing Wang `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了 SocialRL，面向社交对话的多轮强化学习框架，能够同时追求私人目标与维护关系。

**💡 创新点**

创新点在于：①用PPO对完整对话轨迹进行多轮优化；②设计六维动态过程奖励并配合阶段感知权重，实现目标与关系的平衡；③将奖励模型细化为可审计的二元评判。

**🔧 技术方法**

采用PPO、GAE、值网络进行多轮学习；奖励模型基于LLM生成多维度二元准则；动态权重调度基于对话阶段推理。

**📊 数据集**

在四大社交对话基准上实验：SOTOPIA-π、SOTOPIA-All、SOTOPIA-Hard 与 AgentSense，并使用多种LLM背骨与对手模型。

**📈 对比分析**

与基线（Base、BC、Sotopia-RL、SDPO、ArCHer 等）以及商用参考模型对比，SocialRL 在 Goal Achievement 上平均提升 9.2 个百分点，关系维度也显著改善。

**⚠️ 局限性**

局限包括过度妥协、对突发对手行为的僵化、以及极长对话的记忆衰减，未来需引入目标底限、扩大训练分布和增强记忆模块。

---

## 192. Arbitrary Cipher Attacks Against Large Language Models Do Not Require Fine-Tuning

**arXiv ID:** 2609.09553 | [PDF](https://arxiv.org/pdf/2609.09553v1)

**作者:** Thomas Rivasseau `[一作]` `[通讯]` (McGill University), Thomas Rivasseau (McGill University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了一种不需要微调即可在大型语言模型（LLM）上实施的密码破解攻击，利用字母置换实现加密通信，绕过有害内容检测与模型对齐机制。

**💡 创新点**

创新点在于首次展示了在前沿黑盒LLM（如Claude Sonnet 4、Gemini 3、GPT 5.5）上通过提示与上下文学习即可实现密码式 jailbreak，且不依赖模型内部训练或 fine‑tuning。

**🔧 技术方法**

主要技术包括：单步提示完整字母置换、迭代式字母置换教学、上下文示例生成、结合已知 jailbreak 提示（如 crescendo）以及禁用模型推理以防止解密。

**📊 数据集**

使用的数据集为公开可获取的通用知识问答对，用于生成加密示例；实验中还对多款前沿 LLM 进行黑盒评测。

**📈 对比分析**

在前沿模型上实验表明，单步攻击在 Claude Sonnet 4 上可达 100% 成功率；迭代攻击在 Claude Sonnet 4.5、Gemini 3 Flash 等模型上分别实现 80%–100% 的成功率，表明模型性能提升直接开启了此攻击向量。

**⚠️ 局限性**

局限性包括：需要较多的置换对才能成功；对低资源模型效果差；加密过程中易出现字母错误导致信息失真；攻击在开启模型推理后失效；未针对 AI 代理与工具调用等场景进行验证。

---

## 193. Embedding Model-form Uncertainty in Probabilistic Calibration of Digital Twins for Bridges

**arXiv ID:** 2609.10171 | [PDF](https://arxiv.org/pdf/2609.10171v1)

**作者:** Daniel Andrés Arcones `[一作]` (Technical University of Munich), Jörg F. Unger `[通讯]` (Federal Institute for Materials Research and Testing)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对桥梁热模型进行基于嵌入式随机参数的模型形式不确定性量化与校准。

**💡 创新点**

创新点在于将模型形式不确定性直接嵌入可变参数，通过方差分解、KS偏差检验以及可解释的误差分离实现对不确定性的系统性评估与传播。

**🔧 技术方法**

使用技术包括最大似然估计、嵌入式随机参数建模、正交多项式 Chaos（PCE）求解、Sobol 敏感性分析、Kolmogorov–Smirnov 偏差度量以及三步校准流程。

**📊 数据集**

实验基于德国 Nibelungenbrücke 桥梁的温度监测数据，包含 2024 年 6 月的日常监测与 2023–2025 年的多季节数据。

**📈 对比分析**

与仅使用观测噪声校准的基线模型相比，嵌入式方法在 KS 偏差和方差分解上更优，能够将大部分不确定性归因为模型本身，从而提升预测可靠性。

**⚠️ 局限性**

局限性包括嵌入参数的可识别性有限、季节性偏差突出、二维几何简化导致的模型结构缺陷以及对极端环境条件的不足。

---

## 194. Explaining f-Divergence-Based Regularization via Local Curvature and Sharpness-Aware Minimization

**arXiv ID:** 2609.09367 | [PDF](https://arxiv.org/pdf/2609.09367v1)

**作者:** Nour Jamoussi `[一作]` (EURECOM), Marios Kountouris `[通讯]` (University of Granada)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究f‑divergence（如KL、JSD等）正则化与Sharpness‑Aware Minimization（SAM）在参数与输入扰动下的本地二阶几何联系，并验证其对模型泛化的影响；采用α‑skew JSD作为可调曲率系数的实验平台，探讨曲率强度与性能的关系。

**💡 创新点**

① 给出f‑divergence正则化在参数空间的局部二阶展开可表为Fisher加权二次型，② 将其与SAM在二阶解析下的最大Hessian特征值联系，③ 提出输入空间扰动的更一般拉回式二次形式，④ 通过α‑skew JSD的曲率系数α(1‑α)可控的实验验证两者的几何对应关系。

**🔧 技术方法**

f‑divergence二阶展开、Fisher信息矩阵、Gauss‑Newton矩阵、SAM的内层最大化解析、EfficientNet‑B2/ResNet‑18模型、AdamW/SGD优化、NLL与准确率评估、损失曲面可视化（Hessian特征值与曲率统计）。

**📊 数据集**

CIFAR‑10、Fashion‑MNIST、EMNIST（Balanced）和Oxford‑IIIT Pet四个公开图像分类基准。

**📈 对比分析**

在输入扰动（随机遮罩2%）下，沿α∈{0.1,…,0.9}调节JSD曲率系数；结果显示曲率系数最大（α=0.5）时准确率上升、NLL下降；对应的Hessian特征值与曲面平坦度下降；与SAM（ρ设定）相比，曲率强的α‑skew JSD实现更平坦的局部最小值，性能优于无正则化基线。

**⚠️ 局限性**

二阶近似对所有f‑divergence统一，仅区分系数ϕ″(1)，高阶项与结构差异未被捕获，导致无法在本研究框架内选择最优divergence；实验范围局限于输入扰动，未全面检验参数扰动场景。

---

## 195. Degree Sequence Reconstruction from Subgraph Traces

**arXiv ID:** 2609.09397 | [PDF](https://arxiv.org/pdf/2609.09397v1)

**作者:** Venkata Gandikota `[一作]` (Syracuse University), Haodong Yang `[通讯]` (Syracuse University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究如何从按顶点删除的子图（trace）中恢复未知图的度序（度计数向量），提出两种算法实现该重构。

**💡 创新点**

创新点在于首次将度序重构转化为混合二项分布参数估计和度矩的多项式根多重性分析，给出了理论上最优的样本复杂度（O(n^{1/3}) 或 O(n^{1/2})）并推导了对应的必要与充分矩数。

**🔧 技术方法**

主要技术包括：拒绝采样将度序重构映射到混合二项分布学习；使用阶乘矩与普通矩的变换；利用多项式在 x=1 处的高阶根上限理论；以及通过线性规划求解恢复度序的多项式方法。

**📊 数据集**

论文仅使用理论模型和合成图（如完整图、二部图等）进行分析，没有采用真实数据集；所有结果均为严谨的数学证明与概率分析。

**📈 对比分析**

与现有的图追踪重构（需要指数样本）及单一矩估计（样本复杂度 O(n^3)）相比，本文提出的两种方法分别实现了 O(n^{1/3}) 及 O(n^{1/2}) 的样本复杂度，并在二项混合估计方法上实现了理论最优样本量，但其解码时间仍为指数；度矩方法实现了多项式时间解码且与样本量匹配。

**⚠️ 局限性**

主要局限包括：混合二项分布估计算法缺乏子指数时间实现；度矩方法在理论上可行但对非均匀保留概率 q 的情况研究有限；未给出非平凡的下界（仅有 Ω(n^2) 下界）；以及未考虑非顶点删除的其他噪声通道。

---

## 196. LBFAST: A Lightweight Moment-Represented Lattice Boltzmann Solver for Multi-GPU Architectures

**arXiv ID:** 2609.09160 | [PDF](https://arxiv.org/pdf/2609.09160v1)

**作者:** Marco Lauricella `[一作]` (Consiglio Nazionale delle Ricerche), Sauro Succi `[通讯]` (Consiglio Nazionale delle Ricerche)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了一款面向GPU的轻量化 Lattice Boltzmann 求解器 LBFAST，用于大规模多相流仿真；

**💡 创新点**

创新点在于完全不存储分布函数，改用 moment‑represented（仅存密度、速度、二阶张量）并在 GPU 上即时重构分布，显著降低内存占用与带宽需求，同时支持多 GPU 分布式和多种速度集（D3Q19/D3Q27/D3Q27h）；

**🔧 技术方法**

采用 Hermite 基正则化 LBM、CUDA + MPI（异步非阻塞）双缓冲、共享内存重构、Allen–Cahn 相场、能耗监测等技术；

**📊 数据集**

主要使用标准验证案例：单相 Taylor–Green vortex、双相层泊松流、Laplace 圆滴压差以及 Poiseuille 两相层流；

**📈 对比分析**

与传统分布函数存储 LB、accLB、waLBerla 等进行对比；在 Leonardo 超算上 512 GPU 运行 512³ 细分域可达约 56 GLUPS（单精度），强/弱伸缩接近理想；能耗每 LUP 约 nJ/LUP 稳定；整体性能提升约 2–3 倍；

**⚠️ 局限性**

局限在于仍受 GPU 内存带宽/容量限制，高阶模型开销较大；MPI 维度分解需手动调优；对高密度比多相流的鲁棒性有限；缺乏自适应网格或动态负载均衡。

---

## 197. ViBe: Visual Behavior Adaptation for Perceptive Humanoid Whole-Body Control

**arXiv ID:** 2609.09918 | [PDF](https://arxiv.org/pdf/2609.09918v1)

**作者:** Lokesh Krishna `[一作]` (University of Southern California), Quan Nguyen `[通讯]` (University of Southern California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种后训练框架 ViBe，利用冻结的视觉编码器和冻结的全身运动跟踪器，通过跨注意力提取器和低秩适配器学习视觉反馈，使机器人在不同感知控制任务中实现零样本从仿真到真实的迁移。

**💡 创新点**

创新点在于：① 通过跨注意力提取器将预训练视觉特征与机器人状态和任务指令结合，直接生成任务相关的视觉反馈；② 用低秩适配器将该反馈注入冻结的运动跟踪器，保持原有运动先验不变，仅对任务特定的细节进行微调；③ 采用后训练方式（不需要教师-学生蒸馏或额外的表示损失），实现参数高效的强化学习微调；④ 引入参考阶段退火（reference-phase annealing）作为强化学习的自适应课程。

**🔧 技术方法**

使用的技术包括：冻结的视觉编码器（如 Theia‑Tiny ViT、DINOv3‑S+ 等）、跨注意力提取器、低秩适配器（LoRA）、PPO 强化学习、非对称演员-评论家、参考阶段退火、以及预训练的 SONIC 运动跟踪器。

**📊 数据集**

数据集方面：利用多来源的参考动作数据（GRAIL、OmniRetarget、单位树 G1 自研 Clip 集）以及任务特定的数据（Repose Cube 关节轨迹、Nominal pose、Dodgeball 场景）。每个任务都有对应的奖励函数和可选任务指令。

**📈 对比分析**

与基线（盲目跟踪器、教师-学生蒸馏、预训练的特权观测策略）比较，ViBe 在感知行走、感知跑酷、全身物体搬运、躲避球等四个任务上分别达到了 89.6%–94.7% 的特权观测策略性能；在硬件实验中实现了零样本从仿真到真实的迁移，成功率约为 93%（Repose Cube）等，证明了方法在多样化场景下的稳健性。

**⚠️ 局限性**

局限性包括：① 对动态干扰（如移动球、球员头部）仍易产生误判；② 规划器与控制器未在闭环中训练，可能在某些参考转移外出现性能下降；③ 需要更多的域随机化来提升对未见动态扰动的鲁棒性。

---

## 198. Integrated Population Balance and Multiphysics Modeling for Predicting Undesired Agglomeration in Small Molecule Manufacturing

**arXiv ID:** 2609.10256 | [PDF](https://arxiv.org/pdf/2609.10256v1)

**作者:** Prakitr Srisuma `[一作]` (Massachusetts Institute of Technology), Richard D. Braatz `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究并验证了一种集成热传递、质量传递与种群平衡的机理模型，用于预测小分子制造中混合筛滤干燥过程中软硬絮凝粒子的形成与分布演化。

**💡 创新点**

首次在混合筛滤干燥中明确区分软硬絮凝，提出基于溶剂蒸发速率与晶体生长速率耦合的软硬絮凝转化机理，并将其嵌入种群平衡模型。

**🔧 技术方法**

采用多物理场耦合的能量与质量平衡方程、基于种群平衡的粒径演化模型，并使用有限体积法离散化 PDE，MATLAB ODE/PDE 求解。

**📊 数据集**

通过两套实验体系—KCl+正己烷（仅软絮凝）与阿司匹林+异丙醇（软硬絮凝共存）—获得温度、残余溶剂含量与粒径分布数据进行模型验证。

**📈 对比分析**

将模型预测的温度、湿度及 D90、D99 与实验数据在两种体系下对比，误差低于10%，能够准确捕捉软硬絮凝对粒径上尾的影响，证明模型在不同溶质溶解度条件下均具高预测准确性。

**⚠️ 局限性**

模型假设软硬絮凝转换为一级过程、溶剂蒸发与晶体生长速率可简化为经验函数，缺乏对多相动力学和非均匀搅拌的精细描述，且未在实时状态估计或 MPC 中验证。

---

## 199. Robust Beam Prediction for V2X Networks with Multi-Modal Sensing

**arXiv ID:** 2609.10200 | [PDF](https://arxiv.org/pdf/2609.10200v1)

**作者:** Chen Shang `[一作]` (University of Technology Sydney), Jiadong Yu `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于多模态（相机、LiDAR、雷达、GPS）融合的BeamTransFuser框架，用于V2X网络的鲁棒波束预测。

**💡 创新点**

创新点在于层次化Transformer跨模态融合与可缺失模态生成模块，能在部分传感器失效时仍保持高精度预测。

**🔧 技术方法**

采用多分支CNN编码、层次Transformer融合、可学习融合权重以及条件变分自编码器（CVAE）进行模态补全。

**📊 数据集**

在DeepSense 6G真实多模态V2X数据集上进行实验，该数据集包含同步的相机、LiDAR、雷达、GPS和波束标签。

**📈 对比分析**

与Avatar、TII、CMDF、ICMFE、QTNs等基线对比，BeamTransFuser在多场景DBA-score和Top‑k准确率上均居首位，缺失模态时CVAE补全显著提升性能。

**⚠️ 局限性**

局限在于对高维相机特征补全效果有限，且对多车辆复杂场景的鲁棒性尚未验证。

---

## 200. Deep Neural Networks for Learning Intent from sEMG Signals to Support Hardware Devices for Post-Stroke Neurorehabilitation

**arXiv ID:** 2609.09971 | [PDF](https://arxiv.org/pdf/2609.09971v1)

**作者:** Zakariyya Brewster `[一作]` (University of Toronto), Tala Abdelmaguid `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一套针对中风患者受损手臂的五指意图解码流程，从高密度表面肌电信号到可嵌入式设备的模型输出，并实现了四通道硬件约束下的可部署模型。

**💡 创新点**

创新点在于：① 将 LSTM、CNN 与 GNN 在同一 sEMG 特征空间下进行系统对比；② 通过自动化架构搜索得到 Compact CNN（CNN‑Micro）并与 ResNet 教师对齐；③ 在硬件约束下引入跨通道知识蒸馏，显著提升四通道模型性能；④ 形成可重复、可导出 ONNX 的完整软件流水线。

**🔧 技术方法**

使用的技术包括：高通带通 Butterworth 滤波 + Symlet‑4 小波去噪；200 ms 重叠窗口特征提取（12 个时域/频域手工特征）；LSTM、CNN 与 GNN（GCN）预测器；Optuna 自动化架构与训练调参；ResNet 作为教师模型；交叉通道知识蒸馏与迁移学习。

**📊 数据集**

采用了 2026 年发布的中风特定双侧纵向 HD‑sEMG 数据集（48 位病人，64 通道，16 个动作），在受损臂数据上进行训练；同时使用健康臂数据进行健康→受损的迁移实验。

**📈 对比分析**

方法：在同一 64 通道预处理、特征提取和患者级拆分下比较 LSTM、CNN 与 GNN；随后用 Optuna 搜索 CNN 学生网络并对比 ResNet 教师；对 CNN‑Base 进行健康→受损迁移；最后在四通道约束下比较直接训练、第一层切片初始化和跨通道蒸馏。性能方面：CNN‑Large 在单拆分上取得最高子集准确率 0.593、宏 F1 0.714；四通道蒸馏模型平均子集准确率 0.5219、宏 F1 0.6095；相较于直接四通道训练提升约 4% 以上。

**⚠️ 局限性**

局限性包括：仅使用单一患者拆分（无交叉验证），随机种子数量有限；未在真实硬件上测量推理时延；通道映射仅基于预期硬件位置，未通过实验验证；未尝试原始信号 CNN 或稀疏 GNN；实验仅评估解码精度，未验证临床疗效。

---

## 201. VLX-VR: An Agentic-Aware Video Reasoning Model

**arXiv ID:** 2609.09985 | [PDF](https://arxiv.org/pdf/2609.09985v1)

**作者:** Sheng Li `[一作]`, Tiancheng Zhao `[通讯]` (Om Ai Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并训练了VLX‑VR，一种在 Think–Memory–Observation 循环中主动获取、存储并利用多模态证据的智能视频推理模型。

**💡 创新点**

创新点在于将证据获取、记忆读写与终止判断集成到模型的推理策略中，并通过强化学习学习这一完整的推理循环，而非仅在推理后端使用外部工具。

**🔧 技术方法**

采用基于大型语言模型的多模态推理框架，结合强化学习奖励设计，使用自定义的 Memory 组件提供的 <retrieve> 与 <store> 操作。

**📊 数据集**

主要使用 MINERVA 数据集进行训练与评估，此外还参考 Video‑MME、LongVideoBench 等多模态视频理解基准。

**📈 对比分析**

在 MINERVA 上与多款现有视频‑LLaMA、Video‑LLaVA、GPT‑4o 等模型比较，VLX‑VR 获得 78.79% 的多选答案准确率，排名最高，并在不同视频时长组表现稳定，跨时长方差最低。

**⚠️ 局限性**

局限性包括对计数、状态变化、因果推理和空间感知的表现仍弱；在复杂的状态跟踪或不完整视觉信息时可能产生不一致或错误的推理轨迹。

---

## 202. ROAM: Robust Organization of Atomic Memories for Agents through Semantic Relations

**arXiv ID:** 2609.09778 | [PDF](https://arxiv.org/pdf/2609.09778v1)

**作者:** Jianjie Zheng `[一作]`, Guanhua Chen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ROAM 框架，先对新旧原子记忆进行五类语义关系推断（ind、eqv、osn、nso、con），再根据关系将记忆划分为 Primary 与 Evidence，并通过 Fusion 合成读取视图，以解决长期 LLM 记忆中的冗余与冲突；

**💡 创新点**

创新点在于将语义关系推断与存储操作解耦，使用明确的关系标签指导记忆组织，同时引入融合模块提升读取视图质量，显著提高在有限检索预算下的答案准确率；

**🔧 技术方法**

技术包括使用大模型（如 Llama‑3、Mistral 等）做关系推断器、Retriever 做检索、Fusion 模块做视图合成，并在实验中采用 LongMemEval、MEME‑Post 与自制 Controlled 评测；

**📊 数据集**

使用的数据集包括 LongMemEval、MEME‑Post、PersonaMem‑v2（用于关系推断评估）等；

**📈 对比分析**

与 Append‑all、Mem0、EverMemOS‑style 等基线对比，ROAM 在受限检索预算下最高可提升 29.8% 的答案准确率；在完整 LongMemEval 上提升 5.9–2.0% 点，在 MEME‑Post 上提升 5.9–11.5% 点；不同模型规模与预算下仍保持优势；

**⚠️ 局限性**

实验仅覆盖固定历史与检索预算场景，未考察长流、频繁更新、层级/事件级记忆等；对原子化记忆的依赖限制了对更复杂结构化记忆的适用性。

---

## 203. TEFM: Token-Efficient Faithful Modeling for Structured Data

**arXiv ID:** 2609.09552 | [PDF](https://arxiv.org/pdf/2609.09552v1)

**作者:** Zhichao Hou `[一作]` (North Carolina State University), Rui Song `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了TEFM框架，能够在不牺牲准确率的前提下，将高维结构化数据压缩为少量Behavioral Code，供LLM高效处理。

**💡 创新点**

创新点在于结合残差量化VAE实现层级压缩、BC-文本轻量对齐以及双重保真性特征抽取，三者共同实现token效率与可信解释。

**🔧 技术方法**

主要技术包括Residual Quantized VAE、BC-文本对齐预训练、双保真变分解释器以及多模型LLM微调。

**📊 数据集**

实验使用了临床MIMIC-III死亡预测数据和网络CIC-IDS2017入侵检测数据。

**📈 对比分析**

与基线相比，TEFM在Qwen3、Gemma-2、Phi-4等模型上实现了约99% token压缩，同时保持或提升分类准确率，且生成可信的最小特征解释。

**⚠️ 局限性**

局限性包括对不同结构化域的泛化仍需验证、对量化代码的解释性有限以及训练过程对VAE和LLM的双重依赖导致计算成本较高。

---

## 204. Interpreting Object-Dependent Concept Brittleness in Text-to-Image Diffusion Models

**arXiv ID:** 2609.09909 | [PDF](https://arxiv.org/pdf/2609.09909v1)

**作者:** Yifan Yuan `[一作]` (Shenzhen University), Linlin Shen `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究文本到图像扩散模型中的“对象依赖概念脆弱性”，提出诊断并在推理时纠正的框架。

**💡 创新点**

通过将去噪过程映射到逐步稀疏自编码器空间，揭示成功与失败生成在概念维度上的结构差异，并利用稀疏原型进行插值校正。

**🔧 技术方法**

使用稀疏自编码器（SAE）、稀疏编码、类级原型、插值修正以及多步去噪特征。

**📊 数据集**

构造了覆盖10种风格和10种属性的训练集与150+实体的测试集，实验覆盖5个主流扩散模型。

**📈 对比分析**

在风格与属性控制任务上与基线比较，使用CLIP‑I/T、Style Alignment、BLIP‑VQA等指标，均显著提升概念一致性并保持低推理开销。

**⚠️ 局限性**

仅在前20%去噪步骤进行校正，且更深层特征更有效，限制了对更大规模模型以及更细粒度局部属性控制的通用性。

---

## 205. Which Medical Questions Deserve Rationales? Perturbation-Sensitive Selection for Robust QA

**arXiv ID:** 2609.09684 | [PDF](https://arxiv.org/pdf/2609.09684v1)

**作者:** Yuexin Wu `[一作]` (University of Memphis), Vasile Rus `[通讯]` (University of Memphis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在已标注答案的医学多选问答数据集中，研究如何在固定解释标记预算下挑选能提升模型对答案选项顺序不变性的解释样本，并提出 RMS‑RSP 评分方法。

**💡 创新点**

创新点是引入基于解释位置的激活噪声扰动，并用金牌‑最佳干扰项边际变化的根均方（RMS）敏感度得分来衡量解释对决策边界的局部影响。

**🔧 技术方法**

使用 MedGemma‑4B‑IT 大模型与 LoRA 微调，利用高斯噪声扰动隐藏状态并计算边际变化进行离线样本选择与后续解释监督训练。

**📊 数据集**

实验涵盖 AfriMed‑QA、MedExpQA、MedExQA、PubMedQA、MedMCQA 五个医学 QA 数据集。

**📈 对比分析**

与随机、答案熵、答案边际、解释长度以及五种近期推理数据选择器对比；RMS‑RSP 在锁定预算下在 AfriMed‑QA 上显著优于随机，并在所有数据集上均能提升答案顺序鲁棒性和语义一致性，但整体准确率提升有限。

**⚠️ 局限性**

局限包括仅在已生成解释可见的离线场景下工作、对小样本敏感、未考虑专家审核成本、全解释监督与 RSP 的对比不完全公平，以及实验仅在单一 4B 模型和特定数据分割上进行。

---

## 206. CrossLink: Breaking Location Privacy by Linking Device Identifiers Across Protocols

**arXiv ID:** 2609.09963 | [PDF](https://arxiv.org/pdf/2609.09963v1)

**作者:** Aneet Kumar Dutta `[一作]` (CISPA Helmholtz Center for Information Security), Mridula Singh `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究并实现了一种跨协议的不确定性感知跟踪算法，证明即使在LTE、WiFi和BLE等多协议设备各自使用临时标识并保持被动监听，攻击者仍能通过异步旋转实现长时间的设备跟踪。

**💡 创新点**

创新点在于提出了跨协议候选链接构造、交叉协议一致性剪枝和唯一链追踪的完整流程，并给出理论上的混合条件模型说明多协议随机化无法相互组合，揭示系统级隐私缺口。

**🔧 技术方法**

主要技术包括异步观测候选构造、基于时间重叠、空间一致性和移动性约束的候选筛选、交叉协议一致性剪枝，以及对实验测量参数（LTE Timing Advance、WiFi/BLE RSSI误差等）的建模；同时使用SuMO等仿真工具进行大规模场景评估。

**📊 数据集**

数据集涵盖多款国产/国际手机（Samsung Galaxy M11、Xiaomi Redmi K50i、OnePlus Nord、Google Pixel 9、iPhone 15）的现场采样，并在仿真中使用真实测量得到的随机化间隔、传输间隔以及定位误差分布，在3.87 km²密集城市区域生成用户移动轨迹。

**📈 对比分析**

与单协议基线（仅LTE、WiFi或BLE）比较，全覆盖下跨协议算法能恢复约83%用户完整轨迹，单协议仅约21%或4%；在不同侦听器部署策略（RAND、SPOT、PATCH、MOB）下仍保持高隐私泄漏；实验和仿真均显示该算法在定位误差、用户密度、移动速度等参数变化下具有稳健的性能。

**⚠️ 局限性**

局限性包括对定位误差模型的依赖，尤其在多径导致的大误差情况下性能下降；假设各协议随机化独立，未考虑主动攻击或实现缺陷；实现需要大量硬件或受信设备作为移动监听器，部署成本和可行性仍是挑战。

---

## 207. Extracting Semantics from Cattle Reporting Categories for Data Interoperability and Findability

**arXiv ID:** 2609.09381 | [PDF](https://arxiv.org/pdf/2609.09381v1)

**作者:** Kassy Raymond `[一作]`, Deborah Stacey `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

基于底层数据提取并归一化牛类报告分类，构建可直接用于发现与互操作的词汇表；通过对词条的根词与修饰词进行规则化，揭示了现有元数据标准（如AGROVOC）无法覆盖的年龄、性别与生产细节。

**💡 创新点**

创新点在于：①不依赖预先存在的元数据或标准；②利用数据自身的报告分类进行语义挖掘；③通过规则化的修饰词为数据提供可搜索的描述性标签，从而实现更细粒度的互操作和可发现性。

**🔧 技术方法**

技术方法包括：Python脚本实现API抓取、OCR文本提取、结构化数据统一、文本预处理与归一化（使用TextBlob、spaCy EntityMatcher等）、规则化映射、与AGROVOC进行匹配与对比。

**📊 数据集**

使用了四大来源的牛类数据集：FAOSTAT（GE、QCL）、Eurostat（apro_mt_lscatl）、Ethiopia Statistics Service（ESS）以及WOAH（GBADs）共计五个数据集，涵盖不同国家与国际组织。

**📈 对比分析**

与AGROVOC对比的性能显示：底层词汇表在根词、年龄、性别与生产修饰上实现了更高的覆盖率（匹配数约5‑12倍），而AGROVOC仅匹配根词，缺失细粒度信息；因此底层方法在可发现性与互操作性方面显著优于传统标准映射。

**⚠️ 局限性**

局限性包括：仅针对牛类，其他牲畜需重新定义规则；规则化需手工维护，难以覆盖所有表达形式；对文化差异与随时间变化的分类缺乏动态更新机制；以及在缺乏完整元数据时仍需一定的人工或AI支持进行错误检测与质量评估。

---

## 208. Evaluating Enterprise Analytics Agents: An End-to-End, Trace-Backed Methodology

**arXiv ID:** 2609.09182 | [PDF](https://arxiv.org/pdf/2609.09182v1)

**作者:** Teja Venkat Kolli `[一作]` (Thumbtack), Vijay Anand Raghavan `[通讯]` (Thumbtack)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发并验证了一套端到端的评估方法，用来评估企业分析智能代理在解释业务意图、选择度量定义、执行工具、检验结果、生成自然语言答案等多阶段流程的可靠性。

**💡 创新点**

① 将评估流程分为语义理解、执行质量和可靠性三大信号族；② 设计了针对分析代理的失败分类法；③ 结合运行时追踪、可回放实验和跨运行一致性检查构成完整评估框架；④ 把评分分为主要与诊断信号，并引入弃权感知。

**🔧 技术方法**

基于随机隔离重复运行、工具调用追踪、SQL与工具执行监控、基于人类编写的黄金答案的评分器，以及阈值驱动的决策框架；实验中使用LLM模型配置（Fast 与 Reasoning）与工具链。

**📊 数据集**

在一家大型在线商城内部的业务领域，构建了50道真实业务分析问题的问答库（5个业务域各10道），其中10道财务类问题提供了结构化黄金答案。

**📈 对比分析**

对两种模型配置进行3次重复跑，共300条追踪；评估包括运行有效性、黄金答案合规率、合同违规、追踪风险、跨运行一致性等维度。结果显示Reasoning配置在回答率、真实数据利用率、规范源表使用率上均优于Fast，但在执行效率、分解方法、schema过度使用、解释一致性等方面表现不足。

**⚠️ 局限性**

① 仅在单一公司内部实验；② 没有对语义层进行评估；③ 交叉运行一致性基于三次重复，可能低估不稳定率；④ 评价依赖人工黄金答案且未进行可靠性测评；⑤ 结果受特定数据仓库与环境影响，不具可移植性。

---

## 209. Multi-Pass, Multi-View Blended Learning for High-Fidelity Volumetric CT Synthesis from Chest X-Rays

**arXiv ID:** 2609.09920 | [PDF](https://arxiv.org/pdf/2609.09920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 210. Structural Process Supervision for Latent Chain-of-Thought Reasoning

**arXiv ID:** 2609.09928 | [PDF](https://arxiv.org/pdf/2609.09928v1)

**作者:** Yiqi Li `[一作]` (Shanghai Jiao Tong University), Yu Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Prototype-Mediated Process Supervision（PMPS），在训练阶段为隐式推理提供结构化的过程监督，通过可学习的原型实现隐式与显式链式思考嵌入的软多对多对齐，解决表征崩塌问题，并引入Progressive Sequential Alignment逐步放宽顺序先验；

**💡 创新点**

① 用可学习原型作为语义锚点实现多对多软对齐；② 采用Sinkhorn‑Knopp实现均匀分配，避免原型崩塌；③ 通过双向交叉预测损失对隐式与显式表示进行双向监督；④ Progressive Sequential Alignment使用高斯位置先验与余弦退火动态引导顺序结构；⑤ 训练阶段仅增加少量参数，无推理成本。

**🔧 技术方法**

共享两层MLP投影与L2归一化；Sinkhorn‑Knopp软对齐；交叉熵与Smooth L1损失；双向交叉预测损失；Gaussian位置先验与余弦退火的PSA模块；自蒸馏teacher‑student架构；LoRA微调。

**📊 数据集**

GSM8K‑Aug、GSM‑Hard、SVAMP、MultiArith、MATH以及在更大模型上的LLaMA‑3.2‑3B‑Instruct、Qwen3‑4B‑Instruct等。

**📈 对比分析**

与CoT‑SFT、Coconut、CoLaR‑2、Latent‑SFT、CODI、SIM‑CoT等六类基线比较；在GSM8K‑Aug平均精度提升约2‑4%；在OOD（GSM‑Hard、SVAMP、MultiArith）提升约3‑5%；输出长度比显式CoT低50%；在大模型和MATH任务同样取得领先；相比SIM‑CoT参数仅+2M，且无推理成本。

**⚠️ 局限性**

目前仅在数学推理任务上评估，未验证在更广泛的推理或对话任务上的通用性；结果受原型维度与超参调优影响；对无分步标注或非结构化数据的适用性仍需进一步研究。

---

## 211. RoMa-$Ω$: What Feed-Forward 3D Models Know About Image Matching

**arXiv ID:** 2609.09507 | [PDF](https://arxiv.org/pdf/2609.09507v1)

**作者:** David Nordström `[一作]` (Chalmers University of Technology), Fredrik Kahl `[通讯]` (Chalmers University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了前馈 3D 重建模型对图像匹配的潜在能力，并通过将 VGGT‑Ω 的特征注入 RoMa v2，构建了新的匹配器。

**💡 创新点**

创新点在于：①系统评估前馈 3D 模型在零样本、直接几何匹配以及作为特征提取器时的匹配性能；②发现后期层特征相关性退化但仍含丰富对应信息；③用 VGGT‑Ω 替换 DINO，得到的匹配器在多项基准上显著优于现有方法。

**🔧 技术方法**

主要技术包括 Transformer‑based 视觉编码器、VGGT‑Ω 的跨视角注意力、线性探针、RoMa v2 的粗细匹配解码器，以及在训练时仅更新解码器与预测头。

**📊 数据集**

实验使用了 MegaDepth、ScanNet、RUBIK、WxBS、HardMatch、FlyingThings3D、Map‑free 等公开图像匹配与重定位数据集。

**📈 对比分析**

通过与 RoMa 与 RoMa v2 的对比，本文的模型在 WxBS、HardMatch、RUBIK、Map‑free、Dense Matching 等任务中均取得+8.1 mAA、+3.5 mAA、+2.4 AUC 等显著提升，性能优于现有最优方法。

**⚠️ 局限性**

主要限制在于模型参数量与显存消耗大约比 RoMa v2 大 40%，推理速度变慢，且在极端视角与模态变换下的鲁棒性仍有限。

---

## 212. Audio Deepfake Detection Using Temporal Coherence Analysis

**arXiv ID:** 2609.09489 | [PDF](https://arxiv.org/pdf/2609.09489v1)

**作者:** Justin D. Norman `[一作]` (University of California, Berkeley), Sarah Barrington `[通讯]` (University of California, Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于CLAP嵌入的时序一致性分析框架，利用统计特征检测语音与音乐的深度伪造；

**💡 创新点**

发现特征-标签反转现象及语音-音乐熵方向反转，并提出轻量级多专家集成与百分位数自适应音乐检测；

**🔧 技术方法**

使用CLAP预训练嵌入、余弦相似度分布、29维统计特征、XGBoost分类器、加权多专家集成及基于百分位数的自适应阈值；

**📊 数据集**

使用LibriSpeech、ASVspoof 2019/5、DeepSpeak v2、M-LAADS、AUDETER、MUSDB18、FakeMusicCaps、In-the-Wild、FakeAVCeleb、SONICS、FMA等多域数据集；

**📈 对比分析**

与端到端深度学习模型（RawNet2、AASIST）及对抗性检测方法对比，语音在In-the-Wild的EER≈34.9%、AUC≈0.718，音乐跨域F1≈0.938，整体性能与深度学习基线相当或略优；

**⚠️ 局限性**

依赖多源训练集，易受新生成器变化影响，音乐自适应需目标域统计，存在类别不平衡、缺乏实时性、需人工特征选择且缺乏未知分布自适应机制。

---

## 213. Automated Mobile Video Objective Testing System

**arXiv ID:** 2609.09579 | [PDF](https://arxiv.org/pdf/2609.09579v1)

**作者:** Eric Petajan `[一作]` (AT&T), Szilveszter Nadas `[通讯]` (Ericsson Research)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个可在实验室环境下对移动设备视频应用进行实时客观质量评估的系统 AMVOTS，支持多种应用场景（如 DASH VoD、视频会议、远程桌面等），并将评估结果用于闭环 QoE‑aware 资源分配实验。

**💡 创新点**

创新点包括：① 能够在不修改应用代码的前提下，通过 HDMI 捕获两路视频（Reference 与 Distorted）并自动对齐；② 采用低分辨率 1‑bit 颜色量化差异算法实现高速帧对齐；③ 通过多种视频正则化与遮罩技术（裁剪、缩放、掩码、Hald 色彩校正）提高 VMAF 计算准确性；④ 将实时 QoE 数据通过 REST 接口反馈到网络控制器，实现“QoE‑in‑the‑Loop”闭环资源调度。

**🔧 技术方法**

核心技术包括 VMAF、RGB24→160×90 1‑bit 量化帧差算法、GPU/CPU 加速、镜像与遮罩复制、Hald 色彩校正、RESTful 数据传输、harmonic mean 评分聚合。

**📊 数据集**

主要使用基准数据集：预先保存的 VoD 影片、直播新闻原始码流、视频会议录制视频、实时 HDMI 捕获（PC、手机）、模拟网络延迟与丢包环境的测试流；并在 AT&T 研发实验室的 1080p@60fps 设备上验证。

**📈 对比分析**

与传统单独 VMAF 评估相比，AMVOTS 通过自动对齐和正则化可在 1 秒内得到 10 秒内的总 QoE，支持最多 4 路并行摄取，实验显示在 1080p@60fps 条件下系统延迟低于 150 ms，吞吐量可达 4 Gbps，能够在闭环实验中实现约 3 倍的视频流并发提升。对比实验结果表明，在动态资源分配场景下，QoE 评分提升平均 12 % 以上，且系统在 4K 输入时会出现处理延迟波动。

**⚠️ 局限性**

主要限制：① 依赖可获取 Reference 视频，无法评估完全无 Reference 的应用；② 目前只针对 1080p 60fps 进行性能优化，4K 时性能不稳定；③ 对网络条件变化的实时反馈仍需进一步压缩延迟；④ 需要专用硬件（AJA Corvid 44、Dell R740）才能实现高帧率处理，限制了广泛部署。

---

## 214. The kernel-block rank profile and a complete classification of $\mathbb{Z}_2\mathbb{Z}_4\mathbb{Z}_8$-linear Hadamard codes

**arXiv ID:** 2609.09969 | [PDF](https://arxiv.org/pdf/2609.09969v1)

**作者:** Dipak K. Bhunia `[一作]` `[通讯]` (Universitat Politècnica de Catalunya), Dipak K. Bhunia (Universitat Politècnica de Catalunya)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文介绍了一种新的等价不变量——核块秩剖面，并对ℤ_2ℤ_4ℤ_8-线性Hadamard码的完整分类进行了研究。

**💡 创新点**

创新点在于提出了核块秩剖面这一概念，它通过记录核将坐标集划分为块的方式，提供了比传统的秩和核维度更细致的分类信息。

**🔧 技术方法**

使用了递归构造方法来生成ℤ_2ℤ_4ℤ_8-加法Hadamard码，并计算了其核块秩剖面。

**📊 数据集**

研究中使用的主要数据集是ℤ_2ℤ_4ℤ_8-加法Hadamard码的不同类型，特别是长度为2^t的码。

**📈 对比分析**

通过核块秩剖面与传统的秩和核维度进行比较，发现核块秩剖面能够更准确地区分不同类型的Hadamard码，尤其是在t_1≥2的情况下，能够完全恢复类型信息。

**⚠️ 局限性**

限制在于该研究仅针对递归构造的ℤ_2ℤ_4ℤ_8-线性Hadamard码，尚未对其他类型的Hadamard码进行比较，且对更广泛的字母表的推广仍需进一步研究。

---

## 215. Lensless Gaze Is Not Private by Default: Auditing Identity Leakage Across Disclosure Surfaces

**arXiv ID:** 2609.09188 | [PDF](https://arxiv.org/pdf/2609.09188v1)

**作者:** Rahul Vimalkanth `[一作]` (Indian Institute of Technology Madras), Kaushik Mitra `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过在模拟的无镜眼球追踪管线中对不同“披露表面”进行系统化审计，评估了在已知点扩散函数（PSF）且已知身份图库的闭集识别攻击下，各种信号（传感器测量、学习嵌入、压缩瓶颈、局部状态、公开输出及其时间聚合）泄露的身份信息量。

**💡 创新点**

创新点包括：①提出“披露表面”审计框架，将身份泄露视为跨信任边界的系统属性；②在模拟无镜眼球测量中证明视觉上难以识别的图像仍能实现高识别率；③展示压缩维度、光学编码或输出量化并不能自动保证隐私，泄露风险取决于具体表示与聚合方式。

**🔧 技术方法**

技术手段主要是：模拟无镜测量（固定RGB PSF卷积+噪声）；使用ViT‑Tiny MAE、PCA、GSPL瓶颈等嵌入；匹配线性和两层MLP探针进行闭集识别；构造GazeSplit token+残差分解；对时序块进行分块训练/评估及聚合实验。

**📊 数据集**

使用的数据集是公开的Open Eye Dataset（OpenEDS），从中采集36个身份的图像和追踪帧，构成训练、验证、测试三拆分，并在实验中对每个身份采样固定数量的帧进行分块。

**📈 对比分析**

比较方法为：对每个披露表面使用相同的线性/MLP探针评估Top‑1识别准确率。实验结果显示：原始眼部裁剪和模拟无镜测量Top‑1分别约97.7%和96.7%；MAE 192维嵌入94.3%；8维PCA 93.2%；8维GSPL 77.5%；128‑bin量化标记仅38.1%；连续输出72.6%；时序聚合后，量化标记从34%升至40%，连续/残差从55%降至40%。

**⚠️ 局限性**

局限性：①仅在固定已知PSF的模拟环境下评估，未考虑光学密钥随机性或多摄像头环境；②攻击模型仅限线性/MLP探针，未上界更强攻击；③实验使用单一主体划分、无独立会话，未验证跨会话、设备或头戴位置的持久性；④未区分固有虹膜特征与采集相关的几何/照明信息，导致泄露率不能直接映射到真实身份隐私。

---

## 216. SWORD: Wikidata-based Distortions Reveal Hidden Cross-Lingual Inconsistencies in LLM Factual Error Rejection

**arXiv ID:** 2609.09349 | [PDF](https://arxiv.org/pdf/2609.09349v1)

**作者:** Sanghyeok Park `[一作]` (Soongsil University), Jinhyuk Yun `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了SWORD基准，用以评估多语言LLM在识别并拒绝事实错误陈述的能力。

**💡 创新点**

创新点在于系统化利用Wikidata三元组进行语义可控失真，生成语法正确但事实错误的句子，并发现模型在语义可行失真上更易拒绝而非随机失真，进一步揭示跨语言差距被传统评测隐藏。

**🔧 技术方法**

技术包括：从Wikidata提取三元组、使用node2vec构建图嵌入控制失真时的语义相似度、利用Gemini-2.5-Flash将三元组自动翻译成八种语言的自然语言句子、以及多语言LLM的真伪分类推断。

**📊 数据集**

数据集基于2024年9月的Wikidata快照，筛选出所有主体、属性、客体在八种语言均有标签的三元组，随机抽取500条作为测试样本，随后生成四种失真形式。

**📈 对比分析**

与现有多语言基准（如MMMLU）对比，模型在真语句上的准确率相近，但在失真语句上表现差异显著，尤其在韩国/日语上准确率相较于欧美语言下降多达28个百分点，表明跨语言差距被传统指标掩盖。

**⚠️ 局限性**

局限性包括：仅覆盖八种语言，未深入低资源语言；失真生成依赖Gemini-2.5-Flash，可能产生生成偏好；失真策略未涵盖所有类型的事实错误；使用固定时间点Wikidata，未考虑实时知识更新的影响。

---

## 217. GraphDroid: Asynchronous LLM-Based Mobile App GUI Testing via History-Aware Exploration and Hybrid Intent Fulfillment

**arXiv ID:** 2609.10031 | [PDF](https://arxiv.org/pdf/2609.10031v1)

**作者:** Xiaolei Li `[一作]` (Hong Kong University of Science and Technology), Shing-Chi Cheung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GraphDroid，一种基于LLM的异步意图驱动移动GUI测试框架

**💡 创新点**

创新点在于三大原则：基于聚类的记忆机制、异步意图生成以及混合式意图完成策略

**🔧 技术方法**

技术上结合了大语言模型、Spectral聚类、DFS/Go‑Explore启发式搜索和轻量化/LLM驱动的GUI代理

**📊 数据集**

使用了由LLMDroid、LLMExplorer等基准以及10款Google Play热门商业App共21类真实Android应用构成的Benchmark

**📈 对比分析**

在相同两小时预算下，GraphDroid在代码覆盖、活动覆盖和发现状态数量上分别比最佳基线提升约20–50%，且成本仅为最优纯LLM基线的1/8，bug检测率也最高

**⚠️ 局限性**

局限性包括对LLM的依赖、对特定超参数的敏感性以及对复杂意图完成仍受现有GUI代理能力限制

---

## 218. Reference-Based Bias Detection in LLMs via Relative Representations of Hidden States

**arXiv ID:** 2609.10060 | [PDF](https://arxiv.org/pdf/2609.10060v1)

**作者:** Marek Jeliński `[一作]` (NASK - National Research Institute), Sebastian Cygert `[通讯]` (NASK - National Research Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于参考模型的隐层表示偏差审计方法，通过相对表示将不同模型映射到共享空间，并用 Representational Bias Shift ΔB 衡量细调导致的偏差变化。

**💡 创新点**

创新点在于：①将句子编码为与固定锚句相似度向量，形成可比的共享空间；②定义 ΔB 并证明其与输出层偏差高度相关，显著优于传统 SEAT 等基线。

**🔧 技术方法**

使用技术包括相对表示(Relative Representation)、余弦相似度、欧氏距离计算偏差、对齐方法（Procrustes、CKA）以及 ROC AUC 等性能评估。

**📊 数据集**

实验使用 WildGuardMix、DecodingTrust 和 ToxiGen 三大偏差基准，并构造锚句集、属性句集和目标句子集。

**📈 对比分析**

ΔB 与三大基准的 Pearson 相关系数最高达 |r|=0.84，ROC AUC 范围 0.65–0.99，计算成本比输出层评估低 3–50 倍，且在 Llama、Mistral、Gemma 等模型上表现一致。

**⚠️ 局限性**

局限性包括：需有参考模型；对参数高效适配（LoRA）时信号弱，Gemma 结果尤弱；仅适用于英语单轴群体；无法单独判定模型是否无偏，且缺乏因果关系证明。

---

## 219. Beyond Training: A Feasibility Taxonomy for Inference-Time AI Governance

**arXiv ID:** 2609.10105 | [PDF](https://arxiv.org/pdf/2609.10105v1)

**作者:** Samar Ansari `[一作]` `[通讯]` (University of Chester), Samar Ansari (University of Chester)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个包含 20 项推理时治理机制的可行性分类法，评估其技术成熟度、对不同威胁维度的覆盖度，并将其与硬件治理对齐，提出替换原则。

**💡 创新点**

创新点包括：①将训练时代的计算治理迁移至推理阶段；②提出双维威胁模型（能力层级 × 角色）和四种治理场景；③将监测、验证与执法三大类机制纳入同一框架，并给出四点成熟度评估；④通过与硬件治理的对齐构建条件替换原理。

**🔧 技术方法**

技术手段主要是：①对四大商业供应商（Anthropic、OpenAI、Google Vertex、AWS Bedrock）的文档快照进行实证采样；②使用四点成熟度刻度（可部署、近期、研发、投机）和保守评级规则；③构建 20×12 双维威胁矩阵；④利用二次评级与 Cohen κ 检验一致性。

**📊 数据集**

使用的数据集：供应商公开的 API 文档、技术论文与案例、以及随机抽样的七个机制的二次评级结果；未采用传统机器学习训练数据集。

**📈 对比分析**

比较方法：对每个机制按可用性、监管适用性与对 12 个威胁单元的足够/部分/不足进行评估；结果显示 15 个机制已在生产级别实现，但在高能力部署者与微调攻击场景下鲁棒性不足；二次评级的一致性 Cohen κ 为 0.74。

**⚠️ 局限性**

局限性：①评估依赖现有供应商实现，可能忽视未来的技术突破；②对分布式/开放权重部署缺乏可监控基础，导致覆盖率不足；③对高能力部署者（C3.R2）的覆盖极低；④未覆盖能源与成本监测等新兴机制，且对模型内部安全措施的鲁棒性未充分验证。

---

## 220. Guaranteeing Faithful Evidence Extraction in Speculative Retrieval-Augmented Generation

**arXiv ID:** 2609.10046 | [PDF](https://arxiv.org/pdf/2609.10046v1)

**作者:** Quentin Signé `[一作]` (Université de Toulouse - IRIT UMR 5505), Thiziri Belkacem `[通讯]` (Airbus Protect)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Constrained Hybrid Decoding（CHyD）的方法，能够在检索增强生成（RAG）框架中通过硬约束保证所有引用的文本都是从检索到的上下文中逐字提取的。

**💡 创新点**

创新点在于：①通过定义提取模式（EEC）并使用触发词和特殊终止词实现模式切换；②利用后缀树对词元进行硬约束，使生成的任何引用都必须是连续的、完全匹配的上下文片段；③将“可靠性优先”从传统的流畅性/效率转向严苛的事实一致性，提供可验证的安全保障。

**🔧 技术方法**

使用技术包括：speculative RAG框架、两种解码模式（标准与提取）、基于后缀树的token级约束掩码、触发/终止标记的模式切换、few-shot提示、并引入Extraction Faithfulness Accuracy（EFA）等自定义评估指标。

**📊 数据集**

实验数据集：MedMCQA（多项选择医学问答）、MESAQA（医学抽象问答）、QuoteSum（开放域半提取问答）、SEMAeroSQuAD（自建的航空/空间技术领域半提取问答）以及基准的DistilBERT抽取模型。

**📈 对比分析**

与SEMQA、NEST及纯抽取的DistilBERT进行对比。CHyD在所有模型和数据集上实现了近乎完美的EFA（≥0.981），在技术领域的EM/F1显著优于对手，且在ROUGE‑L、BERTScore等流畅性指标上保持竞争力；唯一缺点是推理时延略高，但在安全关键应用中可接受。

**⚠️ 局限性**

局限性包括：对触发词的依赖导致偶尔错失提取；硬约束限制了模型的改写和多样性；推理延迟增加且与提取次数呈线性关系；并未保证推理过程中的逻辑推理正确，仅保证提取片段的真确性。

---

## 221. Decision Transformer for UAV-Mounted RIS-Assisted Dynamic D2D Communications

**arXiv ID:** 2609.09885 | [PDF](https://arxiv.org/pdf/2609.09885v1)

**作者:** Yaxuan Liu `[一作]` `[通讯]` (Jiangsu Second Normal University), Yaxuan Liu (Jiangsu Second Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了无人机挂载可重构智能表面（RIS）辅助的设备对设备（D2D）通信，并在随机链路激活场景下联合优化无人机轨迹、姿态和RIS相位以最大化平均总速率。

**💡 创新点**

创新点在于将RIS的入射角相关反射特性与无人机三维姿态耦合，并通过跨场景离线训练的决策Transformer实现零样本迁移与高效在线微调，显著提升泛化能力。

**🔧 技术方法**

采用了深度强化学习（DDPG）生成专家轨迹，决策Transformer（DT）进行离线预训练和在线微调，以及基于Rician信道和UPA反射模型的物理通道建模。

**📊 数据集**

使用了多场景离线数据集（8个不同起始位置的无人机- RIS 轨迹，每场景生成500条专家轨迹）和在40×40×8 m³空间内的4个D2D用户模拟数据。

**📈 对比分析**

通过与传统DRL算法（PPO、SAC、TD3）以及直接迁移的DDPG进行比较，DT在零样本时已接近专家性能，微调后几乎达到专家级别，仅需较少的交互次数，表现出更优的平均总速率和更低的波动。

**⚠️ 局限性**

局限在于实验仅基于仿真，未考虑风阻、能耗模型复杂化、以及实际硬件实现中的非理想反射和控制延迟；此外，对更大规模RIS或多无人机协同场景的适用性尚未验证。

---

## 222. BRACE: Anchored Bellman-Residual Correction for Stale Critics in Asynchronous RL

**arXiv ID:** 2609.09783 | [PDF](https://arxiv.org/pdf/2609.09783v1)

**作者:** Guanqun Zhao `[一作]` (Beijing University of Posts and Telecommunications), Zeyu Chen `[通讯]` (Baidu Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的值函数目标修正方法 BRACE，专门解决异步强化学习中大型语言模型的 critic‑side 偏差。

**💡 创新点**

创新点在于将重要性加权残差限制在前 k 个 token（k‑capped correction horizon），并在此窗口之外使用常数权重的 Monte‑Carlo 尾部，将策略修正与奖励传播分离，从而克服 V‑trace 在长时间任务中的失效。

**🔧 技术方法**

技术上基于 PPO 的异步训练框架，采用 k‑capped 重要性修正、Monte‑Carlo 尾部、clipped 损失以及标准的优势计算；同时保持与原始 PPO 相同的 actor 目标，只改进 critic 目标。

**📊 数据集**

使用四个长时延代理任务的数据集：Search‑R1（7 个 QA 基准），BrowseComp‑Plus（检索增强推理），DAPO‑Math‑17k（AIME 2024/25/26），以及带工具的 GSM8K；实验模型包括 Qwen3‑8B、Qwen3‑30B‑A3B 与 Qwen2.5‑7B‑Instruct。

**📈 对比分析**

与五种基线（PPO、PPO‑EWMA、AReaL、KPop、IcePop）进行对比，BRACE 在 13/16 个指标中名列前茅，在 Search‑R1 的 mean@1 上提升 2.4%；在 BrowseComp‑Plus 上比同步 PPO 速率快 2.46×，并在离线更新 50 步时保持稳定。

**⚠️ 局限性**

局限性包括：只修正 critic 目标，仍需配合 actor‑side 整改；k、ρ̅、c̅ 等超参需手工设定；在极端 staleness 下仍会出现性能衰减；未来工作需与 actor‑side 方法结合并进一步自动化超参选择。

---

## 223. ALIGN-HOLD: Experience Alignment for Real-Time Hold Control in Large-Scale Ride-Hailing Matching at DiDi

**arXiv ID:** 2609.09685 | [PDF](https://arxiv.org/pdf/2609.09685v1)

**作者:** Zuhao Zhang `[一作]` (Shanghai Jiao Tong University), Shuai Li `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出ALIGN-HOLD框架，在实时叫车匹配中通过构建订单、司机、市场三视角的隐式偏好对来训练奖励模型，并以此为基础实现经验对齐的上下文Bandit持久化策略，已在DiDi巴西市场上线并取得显著收益。

**💡 创新点**

核心创新在于：①将经验评估从手工多目标奖励迁移到基于隐式偏好的对齐学习；②提出多视角偏好构建，结合订单、司机轨迹与实时匹配图信息；③引入模型自适应硬偏好采样与低可识别性反馈基于密度的过滤；④实现无额外推理成本的生产化部署。

**🔧 技术方法**

技术包括：Transformer‑MLP奖励模型与Bradley‑Terry对比损失、Transformer‑LinUCB上下文Bandit策略、模拟器交互训练、基于密度的低可识别性过滤、以及与EXHOLD共享的执行约束层。

**📊 数据集**

使用DiDi实时运营日志，涵盖28天A/B实验期间约100,000条每日请求的订单、司机轨迹与匹配图数据，划分为训练/验证/测试集。

**📈 对比分析**

通过对比现有EXHOLD基准的在线A/B实验，ALIGN-HOLD在巴西5市累计提升trip completion 0.57%、司机收入 0.64%，乘客取消率下降1.85%/2.07%，保持hold比率不变；离线RM评价显示偏好准确率约0.85，整体表现优于手工奖励方法。

**⚠️ 局限性**

局限性包括：隐式偏好易受噪声与混杂影响，低可识别性过滤需动态阈值可能误删；模型泛化尚未在其他地区或时间段充分验证；系统依赖生产匹配模拟器，对环境漂移需持续监控。

---

## 224. XAI-Refine: An Automated Explanation-Knowledge Loop for Brain-Age Prediction

**arXiv ID:** 2609.09388 | [PDF](https://arxiv.org/pdf/2609.09388v1)

**作者:** Yang Qiao `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一种自动化解释-知识循环框架，用于基于静息态功能连接的脑龄预测模型，通过可靠的后验解释、文献验证和最小化修订来实现模型的知识驱动改进。

**💡 创新点**

①闭环的解释-知识迭代机制；②将可验证知识转化为可微分约束而非直接目标；③在解释空间中保持变量、算子和范围的兼容性，确保可审计的最小修订。

**🔧 技术方法**

结合多种后验解释方法（梯度、特征重要性、纵向一致性等）、LLM驱动的文献检索与证据提取、可微分约束学习与响应探测、TransformerConv神经网络等技术。

**📊 数据集**

使用 Emory Healthy Brain Study (EHBS) 数据集，包含 613 名受试者的 81×81 功能连接矩阵，按年龄分层划分为训练、验证和测试三组。

**📈 对比分析**

与多种基准（经典线性、树模型、GCN/GAT/TransformerConv、BrainNetCNN 等）以及控制实验（随机目标、反向约束、直接证据损失、一发式修订、全修订等）对比；最终模型在 MAE 4.38、PCC 0.527、解释可靠性、文献对齐和目标闭合率（≈84%）等指标均优于基准。

**⚠️ 局限性**

依赖于后验解释的稳定性与文献检索的完整性；知识一致性不等同因果或临床验证；仅在静息态功能连接上验证，需跨人群、扫描仪、协议进一步评估；模型可能捕获数据偏差，不适用于临床决策。

---

## 225. Osprey: Target-agnostic Pre-training Makes Stronger Drafters in Speculative Decoding

**arXiv ID:** 2609.09338 | [PDF](https://arxiv.org/pdf/2609.09338v1)

**作者:** Fengxiang Bie `[一作]` (Together AI), Tianyi Zhang `[通讯]` (Together AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种可复用、目标无关的预训练小语言模型（Osprey）作为speculative decoding的drafter，先通过剪枝得到浅层骨干，随后进行目标无关的next-token预训练，再通过词表对齐、零初始化QKV扩展以及蒸馏方法，适配不同目标模型，实现跨模型、跨任务的高效推理。

**💡 创新点**

创新点在于四阶段流程：①剪枝小LM为浅层骨干；②进行目标无关的next-token预训练，构建通用语言模型基座；③对齐目标词表并零初始化QKV扩展，使drafter能接收目标隐藏状态而不破坏预训练参数；④利用蒸馏对目标进行微调。该方法让同一预训练骨干可复用于多目标模型，显著提升接受率和吞吐量，突破了以往仅针对单目标训练的局限。

**🔧 技术方法**

使用的技术包括模型剪枝、next-token预训练、词表映射与对齐、零初始化QKV扩展、基于EAGLE-3 TTT框架的蒸馏训练、SGLang推理平台、以及多语言多任务评估。

**📊 数据集**

使用的数据集包括：FineWeb 100B-token进行通用预训练；各目标模型生成的 chat、code、commonsense、finance、math 等领域的 prompt-response 对；以及公开基准：MATH-500、HumanEval、LiveCodeBench、MT-Bench、Commonsense-Eval、MGSM、Global-MMLU。

**📈 对比分析**

通过与官方和复现的 EAGLE-3 基线在 Qwen3-8B、Llama-3.3-70B-Instruct、MiniMax-M2.5 三个目标模型上进行同样的适配和蒸馏，评估平均接受长度(AL)和 tokens/s。Osprey 的 AL 在所有模型上提升约16–22%，吞吐量提升约17–18%，在跨域和多语言任务上的优势尤为显著。

**⚠️ 局限性**

限制包括：前期目标无关预训练需要较大计算成本，仅在高吞吐量或多目标部署时才可收回；低流量或一次性部署难以弥补；评估仅覆盖三大目标模型，其他硬件或批量设置可能表现不同；大多结果来自单次实验，缺乏多种随机种子验证；未解决模型偏见、安全性等问题。

---

## 226. DensePol: Dense-Angle Polarization Dataset for Learning-Based Polarimetric Vision

**arXiv ID:** 2609.09359 | [PDF](https://arxiv.org/pdf/2609.09359v1)

**作者:** Param Sangani `[一作]` (Saint Louis University), Hadi Aliakbarpour `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了DensePol，一个含有180个全分辨率分析器角度的RGB‑polarization数据集，并提出了一种两阶段RGB‑to‑polarization预测框架；

**💡 创新点**

创新点在于：① 通过Division‑of‑Time（DoT）实现高密度角度采样，显著提升Polarization参考稳定性；② 将Stable Diffusion模型改造成单步预测器，并结合多时钟特征对齐；③ 引入局部DoLP细化器，仅修正DoLP而保持AoLP不变；

**🔧 技术方法**

技术方法包括：高密度DoT采集、基于Fourier的角度拟合与残差分析、双角度编码（ρ, cos2ϕ, sin2ϕ）、Stable Diffusion U‑Net的一步预测、Timestep‑Conditioned Adapter对齐以及轻量化全分辨率DoLP细化网络；

**📊 数据集**

使用的数据集为DensePol（2,018幅RGB‑polarization对，包含1,988实景和30合成渲染），与现有四角DoFP/DoT数据集（如PolARGB、PolarAnything等）做对比；

**📈 对比分析**

在多项指标上表现优异：相较于现有最佳基线，AoLP MAE下降约13.8%，DoLP MAE下降约0.001，PSNR提升4.8 dB，SSIM提升0.14；在下游表面法向估计任务中也排名第一；

**⚠️ 局限性**

局限性包括：仅适用于静态场景，采集过程对运动和机械误差敏感；高角度采集耗时且设备成本高；对极端光照和材质变化的泛化仍有限；

---

## 227. Cross User/App Network Attacks - Hijacking TCP Connections and DNS Cache Poisoning via a Malicious User/App (Extended Version)

**arXiv ID:** 2609.09345 | [PDF](https://arxiv.org/pdf/2609.09345v1)

**作者:** Tamir Shahar `[一作]` (Hebrew University), Amit Klein `[通讯]` (Hebrew University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文展示了一种利用本地低权限恶意应用与远程攻击者协同，推断TCP ISN和DNS源端口，从而实现TCP连接劫持和DNS缓存投毒的技术。

**💡 创新点**

创新点在于利用操作系统接口（如procfs、cBPF、IP选项）和端口分配机制泄露协议状态，证明即使攻击者不具备包捕获权限，也能突破TCP和DNS的随机化防御。

**🔧 技术方法**

技术包括：cBPF过滤泄露ACK/ISN、IP源路由选项传递SYN以获取ISN、对端口分配算法进行分析与预测、利用端口绑定/监控推断源端口、离线或远程生成并投递伪造的SYN‑ACK和DNS响应。

**📊 数据集**

主要使用了多平台实验数据：Linux（Ubuntu 24.04）、Android 16、macOS 26.4、iOS 26.3、Windows 11 Home/Pro；实验涉及TCP handshake、HTTP响应注入、systemd‑resolved、Android stub、Windows DNS cache等真实环境。

**📈 对比分析**

与传统随机化或基于内核计数器的攻击相比，本文方法在多平台上成功率≥80%，并在端口保留NAT环境下仍能完成。实验表明在Linux cBPF预处理下成功率97%，在Android IP选项下86%，Windows IP选项下87%，DNS缓存投毒在Linux 97%/Android 100%/Windows 100%。

**⚠️ 局限性**

限制包括：需攻击者能在本地机器上运行恶意代码，某些技术（IP选项）受网络设备过滤；在高延迟或CDN前端时竞赛窗口变窄；不同系统对IP选项/端口分配的实现差异导致攻击难度不同。

---

## 228. Approximate Nearest Neighbor in Ultra-High Dimensional $\ell_\infty$

**arXiv ID:** 2609.09427 | [PDF](https://arxiv.org/pdf/2609.09427v1)

**作者:** Nathan White `[一作]` (University of Pennsylvania), Tian Zhang `[通讯]` (University of Pennsylvania)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出了在维度远大于点数的超高维情形下，针对 ℓ∞ 度量的近似最近邻（ANN）数据结构。

**💡 创新点**

创新点在于引入“子集嵌入”概念：仅选择原始坐标的子集即可近似保持所有点间的 ℓ∞ 距离；给出上界 n^{1+1/c} 与匹配的下界，证明其最优；利用该嵌入构造了无维度依赖的查询时间、对数维度依赖空间的 ANN 方案，并给出多种逼近-查询时间折衷。

**🔧 技术方法**

技术主要包括：
- 子集嵌入的贪心构造与图论分析（环长与边数关系）得到上界；
- 利用子集嵌入构造高维数据的低维表示；
- 分组与递归筛选 3-近似基础结构，逐步提升逼近因子；
- 证明匹配下界的图生成与距离编码方法。

**📊 数据集**

论文中没有使用具体实验数据集，主要为理论分析与数据结构构造。

**📈 对比分析**

与已知的 ℓ∞ ANN 方法相比，本文提供了查询时间不随 d 变化、空间仅对 log d 依赖、逼近因子可在 O(log n) 与 O(1) 之间折衷的方案；在近似因子 ≥ 3 的情况下，查询时间达到 Ω(n) 的下界近似匹配；在近似因子为 O(1) 时，查询时间接近线性 n^{1+ε}，空间为 O(n² log d)，优于之前的 ℓ∞ 方案（需要 Ω(nd) 查询时间）。

**⚠️ 局限性**

局限性：
- 无法实现 1+ε 级别的逼近；逼近因子小于 3 时必需 Ω(d) 查询时间；
- 子集嵌入的维度下界仍为 Ω(n)，在极高维下仍需存储大量坐标；
- 只针对 ℓ∞ 度量，其他 ℓ_p（p>2）情形仍未解决。

---

## 229. Distribution-Consistent Inference for Dynamic Sparse Mixture-of-Experts

**arXiv ID:** 2609.09241 | [PDF](https://arxiv.org/pdf/2609.09241v1)

**作者:** Dohyeon Kim `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在稀疏专家网络(SMoE)中减少激活专家数目对中间表示的影响，并提出一种轻量级的推理时分布对齐方法LDA；

**💡 创新点**

LDA通过对每层的均值和方差进行逐维归一化，补偿因减少top‑k导致的尺度和方差失衡，从而显著恢复性能；

**🔧 技术方法**

采用均值-方差对齐（per‑dimension moment alignment）、RMSNorm分析、基于Calibration Set的统计估计以及可与现有动态路由策略无缝结合的推理时修正；

**📊 数据集**

在C4数据集上进行校准，评估使用13个基准任务（通用知识、常识推理、数学推理、代码生成、指令跟随），以及ShareGPT和NuminaMath‑1.5用于效率对比；

**📈 对比分析**

与固定top‑k、动态top‑p、PESF等路由策略对比，实验显示LDA在同等专家预算下将性能提升到接近或超过默认top‑k，且几乎不增加推理成本；在多任务、不同模型上均得到统计显著改进；

**⚠️ 局限性**

无法弥补因减少专家数导致的实际专家容量损失；仅使用均值和方差，忽略协方差与高阶统计；需额外的校准步骤；动态top‑p阈值仍需手工设定。

---

## 230. Cascading Gradient Inversion via LT-Code Inspired Peeling in Federated Learning

**arXiv ID:** 2609.09659 | [PDF](https://arxiv.org/pdf/2609.09659v1)

**作者:** Saeed Shariati `[一作]` (University of Isfahan), Mohsen Alambardar Meybodi `[通讯]` (University of Isfahan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于 LT 代码剥离（peeling）思想的级联梯度反演攻击，能够在单次 FedSGD 通信中完整恢复批量样本及其标签。

**💡 创新点**

创新点在于：①将梯度反演与 erasure‑correcting 代码理论相结合，构建“级联”恢复流程；②通过控制第一层神经元的激活度分布，逼近 robust soliton 分布，显著提高单个“孤立”样本的出现概率，从而实现大批量恢复；③提供两种无辅助数据和有辅助数据的激活度调节方法（Soliton‑Free、Soliton‑Data）。

**🔧 技术方法**

使用的核心技术包括：analytic 梯度反演（利用第一层全连接层+ReLU 的梯度结构）、LT 代码的 peeling 解码思想、robust soliton 分布的概率调度、标签回归与认证（基于第一层偏置残差的闭式拟合）以及批量反向传播用于批量样本的梯度更新。

**📊 数据集**

在 8 个公开基准数据集上评估：CIFAR‑10、CIFAR‑100、MNIST、EMNIST、Fashion‑MNIST、SVHN、ImageNet、HARUS。

**📈 对比分析**

与现有单轮反演方法（如 Boenisch CaH、SPEAR、SPEAR++）以及多轮方法比较，结果显示：在被动场景下 94–100% 的样本在批量 ≤128 时被完整恢复；在主动场景下，批量几百时恢复率超过 90%；相较于之前方法提升幅度可达数十个百分点。性能评估基于平均恢复率、标签准确率及每轮迭代的收敛次数。

**⚠️ 局限性**

局限性包括：仅适用于首层为全连接+ReLU 的网络；当前实现仅针对 FedSGD，尚未推广至 FedAvg；需对第一层参数进行改动（可能易被检测）；在均值高度集中或维度极低的任务（如 MNIST、EMNIST）下主动/被动恢复率仍相对较低；Soliton‑Data 需要一批辅助样本，若不可获取则只能使用 Soliton‑Free，后者在某些数据集上恢复率略低。

---

## 231. RAP: Research Attention Prediction Reveals Target-Conditioned Evidence Acquisition Biases

**arXiv ID:** 2609.10092 | [PDF](https://arxiv.org/pdf/2609.10092v1)

**作者:** Yingqian Wu `[一作]` (Fudan University), Zhongyu Wei `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并评估了一套滚动、结果导向的研究关注度预测基准（RAP），检验LLM在有限检索条件下对未来六个月论文方向分布的预测能力。

**💡 创新点**

创新点在于：①提出面向结果、可滚动的研究关注度预测任务；②将检索、状态恢复与未来更新拆分为独立诊断；③发现检索目标导致证据采集失效，揭示LLM在未来更新方面的瓶颈。

**🔧 技术方法**

使用大语言模型（GPT‑5.5、Qwen‑3系列、Claude 等）结合 BM25 文献检索，实验不同检索时效窗口（关闭、固定窗口、累计历史），并采用 Spearman 秩相关、TV、JSD 等评价指标。

**📊 数据集**

数据集基于 2022‑06 至 2026‑06 的 arXiv AI/ML 论文，构建 281 个重叠领域、8 方向的冻结代码表，并按时间划分为滚动预测期。

**📈 对比分析**

与最近状态、EWMA、ARIMA 等统计基线对齐；所有LLM 条件均未能超过 EWMA；检索能提升部分模型的准确性，但未突破持续性基线；在 Fine‑Tuning 后，Qwen‑3‑4B 的 Spearman 最高提升 0.105。

**⚠️ 局限性**

局限性包括：仅衡量论文提交份额，无法评估科研质量；方向代码表人为且不完整；在累计检索时目标驱动的证据采集失效；LLM 在恢复现状方面表现良好，但对未来更新的可靠性不足。

---

## 232. Multimodal Emotion Recognition in Conversations via Class-Wise Adaptive Modality Fusion and Affective Geometry

**arXiv ID:** 2609.09924 | [PDF](https://arxiv.org/pdf/2609.09924v1)

**作者:** Oriol Marín `[一作]` (Eurecat, Centre Tecnològic de Catalunya), Rafael Redondo `[通讯]` (Eurecat, Centre Tecnològic de Catalunya)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该工作提出了基于Transformer的多模态情感识别框架，通过融合文本、音频和视觉特征，加入面部几何信息、按类别自适应模态融合以及基于情感几何的先验，用以提升对情感转移的识别。

**💡 创新点**

创新点包括将面部外观与几何描述相结合增强视觉输入，设计按情感类别自适应的模态融合机制，及引入情感空间先验以平滑情感转移预测。

**🔧 技术方法**

使用了Self‑Distillation Transformer架构、ViT、3D面部特征（Landmarks、Expression Parameters、Action Units）、两层MLP自适应权重、温度化KLD蒸馏、以及情感空间先验。

**📊 数据集**

在MELD和IEMOCAP这两个多模态对话情感识别基准上进行评估。

**📈 对比分析**

通过与现有方法（DialogueRNN、MMGCN等）和原始SDT做对比，使用加权F1和准确率评价，最终在两数据集上均实现了比原始SDT提升约2–4%加权F1，且在情感转移样本上获得了约1–2%准确率提升。

**⚠️ 局限性**

局限性包括对面部检测的依赖、先验需要数据集特定的情感坐标、缺乏对不同说话者的显式建模，以及在不同数据集间结果不完全可比。

---

## 233. LightMedSeg-ISLES: Stroke Lesion Segmentation with 81x Fewer Parameters than nnU-Net

**arXiv ID:** 2609.09634 | [PDF](https://arxiv.org/pdf/2609.09634v1)

**作者:** Giorgi Nikvashvili `[一作]` (University of California Berkeley), Yang Yang `[通讯]` (University of California San Francisco)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种轻量级的3D U‑Net 变体 LightMedSeg‑ISLES，用于 T1‑加权影像中卒中病灶的分割。

**💡 创新点**

在保持 97.5% nnU‑Net Dice 的同时，仅使用 81× 更少的参数；核心创新包括 GhostConv 基础、局部结构先验模块（LSPM）、空间锚点以及针对卒中病灶的训练策略。

**🔧 技术方法**

使用 GhostConv、LSPM、空间锚点、深度监督、Flip TTA、强数据增强、FiLM 模式、AdamW 与余弦学习率调度等技术。

**📊 数据集**

在 ISLES’26 公开数据集上进行实验，训练 1045 张图像，验证 262 张，内部测试 146 张；数据来源于 ATLAS v2、SOOP 与新加入的 ISLES’26 病例。

**📈 对比分析**

与过滤后的 nnU‑Net、UNETR++ 及 nnFormer 进行对比；LightMedSeg‑ISLES（1.26 M 参数）在 TTA 下获得 Dice 0.6178、lesion‑wise F1 0.5992，显著优于大模型，且 FLOPs 仅为 nnU‑Net 的 4.7×。

**⚠️ 局限性**

在小至中等尺寸病灶上的性能仍落后，且实验仅在内部留存集完成，未进行外部验证；模型仅针对 T1‑加权图像，跨模态推广有限。

---

## 234. Agentic AI-enabled Semantic Commissioning of a Cognitive Digital Twin for Reconfigurable Manufacturing

**arXiv ID:** 2609.09503 | [PDF](https://arxiv.org/pdf/2609.09503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 235. How effective are traditional test criteria at detecting bugs in large language models generated code?

**arXiv ID:** 2609.09315 | [PDF](https://arxiv.org/pdf/2609.09315v1)

**作者:** Asma Hamidi `[一作]` (University of Luxembourg), Mike Papadakis `[通讯]` (University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过大规模实验研究传统测试充分性指标（语句覆盖、分支覆盖、变异测试）在检测LLM生成代码中的缺陷时的有效性；

**💡 创新点**

创新点在于首次系统评估LLM驱动的全自动软件开发流程中传统覆盖指标的适用性，并揭示缺陷检测主要受断言质量限制；

**🔧 技术方法**

采用LLM（GPT‑5‑Mini、GPT‑4.1‑Mini、Claude‑Haiku‑4.5、DeepSeek‑V4‑Flash、Llama‑3.3‑70B‑Instruct）进行代码与测试生成，利用变异测试和覆盖率工具计算覆盖度并评估触发/检测率；

**📊 数据集**

使用四个Python编程基准（HumanEval+、MBPP、BigCodeBench、NaturalCodeBench），共生成约6,066个挑战性缺陷实例；

**📈 对比分析**

通过在每个缺陷上随机抽样100组满足给定覆盖度的测试集，计算触发率（FTR）与检测率（FDR），结果显示各指标差异不大，覆盖度高并不能显著提高缺陷检测，平均检测率低于10%；

**⚠️ 局限性**

局限性包括仅针对Python函数、仅使用5种LLM与4个基准、缺陷生成依赖参考实现的正确性、以及对测试断言自动生成的依赖性仍需人工干预。

---

## 236. Context operations to architecture modelling output from large language models and evaluation criteria for their use in systems engineering design

**arXiv ID:** 2609.10132 | [PDF](https://arxiv.org/pdf/2609.10132v1)

**作者:** Vinicius Kaster Marini `[一作]` (Federal University of Santa Maria), Petter Krus `[通讯]` (Linköping University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套正式化的上下文组装框架，用于将多模态上下文（策略提示、持久引用、向量化提问）输入大型语言模型，并在SAR无人机组件架构建模案例中进行验证；

**💡 创新点**

创新点包括：①对LLM上下文单元的模块化组合与语法化；②基于模型化‑代码的LLM输出评估方法；③通过对不同上下文处理方案与模型规模的对比，揭示并缓解“中间丢失”注意力效应；

**🔧 技术方法**

使用技术包括：大型语言模型（Claude Sonnet、Qwen、ChatGPT‑OSS、Nemotron、Kimi）、Prompt Engineering、μ‑Template 模板、检索增强生成（RAG）、PlantUML 代码块模型化以及语法/架构验证脚本；

**📊 数据集**

使用的数据集主要是SAR UAV 的需求、规范与参考文档（PDF、DOCX、结构化表格）以及模型生成的代码块；

**📈 对比分析**

通过四种上下文处理（T1‑T4）和五个LLM模型的实验，评估回答长度、上下文利用率、生成时间、模型结构（元素/流/端口计数）以及语法错误/缺失情况。结果显示，高容量模型在细节和完整性上优于低容量模型，μ‑Template 有显著提升模型质量，而引用信息因中间丢失效应影响不大；

**⚠️ 局限性**

局限性在于LLM 对中间上下文的注意力衰减导致引用信息效果有限；模型规模受限导致小模型表现欠佳；实验仅限单一查询，未探索多轮对话或代理建模；验证手段仍需进一步自动化与细化。

---

## 237. Maverick: Private and Verifiable LLM Inference Made Practical via Matrix-Vector Multiplication Delegation

**arXiv ID:** 2609.10264 | [PDF](https://arxiv.org/pdf/2609.10264v1)

**作者:** Ben Merbaum `[一作]` (Yale University), Fan Zhang `[通讯]` (Yale University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在开放源代码大型语言模型上实现隐私保护和可验证推理的完整协议，利用矩阵向量乘法的委托与加密实现无服务器开销的隐私与正确性验证。

**💡 创新点**

创新点在于首次实现信息理论安全的可验证矩阵向量乘法委托（vMVMD）并与基于 LPN 的伪随机掩码结合得到 pvMVMD，从而在不增加服务器计算负担的前提下完成输入隐私与验证。

**🔧 技术方法**

采用 Freivalds 改进的稀疏挑战、线性误差更正码（RAA）、双重 LPN 假设生成伪随机掩码、批量验证协议以及在 BabyBear 有限域上的 FP8 量化实现。

**📊 数据集**

使用公开的 Qwen3‑4B（4B 参数）作为评测模型，实验数据基于其量化后的 BabyBear 域权重。

**📈 对比分析**

相较于本地推理，标准/加速/仅验证模式在 64 线程服务器和 2–8 线程客户端下实现 8–18 倍吞吐提升；与 DeepProve 的 124M GPT‑2 与 270M Gemma‑3 对比，验证时间和整体推理时间相差 30–150 倍；在不计网络延迟时服务器端 CPU 并行数 64 时吞吐量 8–18 倍，网络延迟下吞吐提升 10–156 倍。

**⚠️ 局限性**

主要限制包括：需要在客户端进行较大的一次性预处理（O(mn)）存储、通信轮数较多导致标准模式对低延迟场景不友好、批量验证仍需服务器额外计算以及对代码的特定误差纠正码依赖，未来工作需进一步减少通信和预处理开销。

---

## 238. Decision Shifts, Lost Label Functionality, and an Inconclusive Grounding Audit in Correctness-Gated Multi-Teacher Distillation

**arXiv ID:** 2609.09702 | [PDF](https://arxiv.org/pdf/2609.09702v1)

**作者:** Xiaofei Feng `[一作]` `[通讯]` (University of Illinois Urbana-Champaign), Xiaofei Feng (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在固定的学生模型、教师池和测试集上，比较了基于候选决策正确性门控的加权多教师蒸馏（Correctness-weighted MTD）与无过滤多教师蒸馏（Unfiltered MTD）及硬过滤（Hard-filter）等八个训练方案的决策行为与证据合理性。

**💡 创新点**

创新之处在于提出了“候选正确性门控”与“硬过滤对比”的匹配实验框架，系统性地评估了连续加权对决策分布的影响，并揭示了加权方法并未提升证据支持能力。

**🔧 技术方法**

核心技术包括：多教师蒸馏、基于逻辑回归的候选正确性门控、硬过滤与加权过滤两种候选保留策略，以及多标签决策头与自然语言生成解码器的联合训练。

**📊 数据集**

使用的实验数据为基于Yelp开放数据集的餐厅评论证据，构成882条人工标注实例（353训练、262开发、267测试），其中测试集中包含167条声明与100条推荐查询。

**📈 对比分析**

在精确度、五标签宏F1和任务定义的条件不安全率三个指标上，Correctness-weighted MTD相较于Unfiltered MTD提升了准确率和宏F1，但宏F1下降、Recall损失严重（尤其是Refuted标签为0），且在人工评审中未出现任何证据支持的正面输出，整体性能并无实质提升。

**⚠️ 局限性**

局限性包括：仅使用单一学生模型与单一域（餐厅评论），样本量有限（267测试例），硬阈值对结果高度敏感，且人类评审未配对样本，导致无法估计系统级证据合理性提升或损害的真实效应。

---

## 239. Accountable and uncertainty-aware evaluation of sensor-based AI under distribution shift: devices, subjects, and nearly three years underground

**arXiv ID:** 2609.09257 | [PDF](https://arxiv.org/pdf/2609.09257v1)

**作者:** Benny Platte `[一作]` (Mittweida University of Applied Sciences), Marc Ritter `[通讯]` (Mittweida University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了一套针对传感器AI系统的分阶段、可量化不确定性评估协议，并在两座地下矿山的磁场定位任务上进行实验。

**💡 创新点**

创新点在于将模型评估视为测量，提出四个累计泛化阶段、机会参考、OOPS率以及基于分位数的决策规则，显著揭示随机拆分的过度乐观并提供长期性能分解。

**🔧 技术方法**

采用LSTM序列分类、不同磁场表示（F_V、F_HZ 等）、重采样训练、bootstrap 置信区间和分位数统计等技术。

**📊 数据集**

使用两座矿山（Markus‑Röhling‑Stolln 和 Schlema‑Alberoda）的磁场序列数据，包含训练、交叉设备、交叉主体和34个月后重新测量的数据。

**📈 对比分析**

通过对比随机拆分、交叉设备、交叉主体和跨时间的宏平均精度，并使用分位数阈值进行可部署性判定；结果显示在交叉主体阶段精度降至0.69（比随机拆分低0.41），跨时间存在约23点的真实磁场衰变。

**⚠️ 局限性**

局限性包括样本量有限（仅两名主体、两次跨时间测量）、设备与主体不可分离、缺乏多次路测和设备校准跟踪，以及仅评估单一模型配置的重复训练导致不确定性估计受限。

---

## 240. Academia x Industry: The Role of Fundamentals for Silicon in an AI Native Era

**arXiv ID:** 2609.09344 | [PDF](https://arxiv.org/pdf/2609.09344v1)

**作者:** Vincent T. Lee `[一作]` (Meta Reality Labs Silicon), Matheus Trevisan Moreira `[通讯]` (Meta Reality Labs Silicon)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文阐述了在 AI 原生硅设计时代，学术与产业如何通过重视基础设计原则、培养 AI 流畅性以及知识沉淀，推动硅设计的协同转型。

**💡 创新点**

创新点在于提出将 AI 工具视为补充而非替代的框架，强调软技能培养、标准化接口与知识转化的重要性，并提出 AI 兼容技术转移模型。

**🔧 技术方法**

主要技术包括大型语言模型（LLM）、代理式 AI 工作流、EDA 工具的 AI 原生接口、以及基于规则的自动化验证与奖励机制。

**📊 数据集**

未使用公开数据集，主要依托行业内部经验、设计案例与内部日志进行论证；对数据来源做了理论阐释而非实验收集。

**📈 对比分析**

由于为概念性与框架性论文，未提供实验对比与定量性能指标；作者通过案例分析、行业趋势与专家访谈说明潜在效益与挑战。

**⚠️ 局限性**

局限包括缺乏可量化验证、数据可用性受限、对 AI 与硬件设计协同细节的实现路径不完整，以及对知识沉淀与标准化接口的实际落地尚未实证。

---

## 241. Grounding Generated Video Plans in Simulation Towards Versatile Dexterous Controllers

**arXiv ID:** 2609.10050 | [PDF](https://arxiv.org/pdf/2609.10050v1)

**作者:** Tianyue Wu `[一作]` (University of California, Berkeley), Masayoshi Tomizuka `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

结合生成视频模型与仿真跟踪，训练统一的手–物体控制器，实现抓取、推拉、姿态调整等多种操作；

**💡 创新点**

①使用可控视频生成与重建管线产生海量、可调节的手物交互参考；②在仿真中采用强化学习+SAPG优化的跟踪式策略学习多目标、多轨迹控制；③通过行为克隆+DAgger将多任务策略蒸馏成单一通用控制器；

**🔧 技术方法**

视频生成模型Seedance 2.0、FoundationPose、WiLoR、HOI‑DETR等；手物重建采用MANO、深度对齐、联合优化；RL采用异步Actor‑Critic、SAPG、Domain Randomization、精细奖励设计；仿真环境为IsaacGym/PhysX，硬件为Franka arm + Sharp Wave Hand，感知使用RealSense + FoundationPose；

**📊 数据集**

生成视频约2500个clip，重建后约2000个HOI轨迹；基准数据H2O、HO‑Cap用于重建评估；训练对象共42类，10类基准对象，5类新对象；

**📈 对比分析**

与VideoManip、EgoInfinity、DO AS I DO等基线在HOI重建上进行人类评估、ADD‑S、MRRPE、CDev等指标，取得平均排名1.67、ADD‑S 4.82/3.88cm；在RL跟踪上与PPO、ReGrind、ManipTrans等对比，专家策略在训练对象上宏平均成功率78.6%，基线低；在基准对象上达到77.4%宏平均；蒸馏后仍有66.6%宏平均；实地测试20条未见视频计划，整体成功率27/40；

**⚠️ 局限性**

生成视频对大物体或薄物体抓取支持有限；重建噪声导致轨迹不准，学习困难；Sim‑to‑real差距在接触转移时易失效，缺乏有效恢复策略；未覆盖手内操作；对复杂人类示范的依赖仍高。

---

## 242. CT-SAFR: Safe and Interpretable Chain-of-Thought Reasoning for Autonomous Robots: A Multi-Layered Verification Framework for Trustworthy AI-Driven Robotic Decision Making

**arXiv ID:** 2609.09692 | [PDF](https://arxiv.org/pdf/2609.09692v1)

**作者:** Cagri Temel `[一作]` `[通讯]` (Hezarfen LLC), Cagri Temel (Hezarfen LLC)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本论文提出并实现了CT‑SAFR，一个多层级验证框架，用于在自主机器人中安全、可解释地部署 Chain‑of‑Thought（CoT）推理。

**💡 创新点**

创新点包括：①四层防御深度结构（结构、物理、语义、自解释）实现 94.2% 幻觉检测与 96.4% 综合安全检测；②将自一致性解码从 NLP 拓展至机器人任务，实现多路径一致性判断；③低延迟（<500 ms）实时验证满足机器人控制需求；④提供完整的实验评估与消融分析，验证每层安全价值。

**🔧 技术方法**

使用技术：LLM CoT（Mistral‑7B‑Instruct‑v0.2）+自一致性解码（k=5）；结构层采用形式化语法解析与符号推理；物理层基于几何约束检查（AABB、碰撞网格、运动学限值）；语义层对照知识库与机器人认知状态；解释层生成可视化与置信度报告；硬件层包含 RTX 3090 并行推理与专用安全硬件。

**📊 数据集**

实验数据集：Gazebo 仿真仓库环境，共 500 个任务（60% 标准、25% 边缘、15% 对抗性），覆盖 1,000 小时操作。数据包括传感器观测、物理约束、任务指令、已验证事实库。

**📈 对比分析**

与现有方法（如 SayCan、Yang 等）对比，CT‑SAFR 在安全指标上显著提升：幻觉检测率 94.2%（95% CI 91.8–95.9%），总体安全检测 96.4%，延迟 462 ms；在仓库实验中，安全事件下降 87%（p<0.001），任务完成率提升 3.5%。

**⚠️ 局限性**

局限性：评估仅局限于受控仓库场景，未覆盖户外、航空或水下等环境；对抗性攻击针对 LLM 本身未做系统评估；物理约束需人工编写，约束提取自动化不足；自一致性采样 k=5 增加能耗，对电池供电机器人有负担。

---

## 243. Can We Trust Video Hallucination Detectors? VidHalLoc for Evaluating the Evaluators

**arXiv ID:** 2609.09895 | [PDF](https://arxiv.org/pdf/2609.09895v1)

**作者:** Xinyu Chen `[一作]`, Mark Dras `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VidHalLoc 基准，用于在视频问答与视频字幕任务中评估视频幻觉检测方法，并设计了一套统一的诊断评估协议。

**💡 创新点**

创新点在于：①使用对抗性幻觉样本构建 2,000 条样本，覆盖实体与动态两大幻觉类别；②引入基于 harness engineering 的多智能体工作流，显著提升构建效率并保证高质量标注；③提供统一的评估框架，使不同检测方法可在同一数据集上直接比较。

**🔧 技术方法**

采用了多模态嵌入相似度、学习式蕴含、结构化验证等检测技术，并利用 CLIP、LaViLa 等视觉特征抽取器以及多智能体的协同工作协议。

**📊 数据集**

使用了 1,090 条源视频（来自 VidOR、COIN、Perception Test、UCF101-DS、UCF101），在此基础上生成 2,000 条包含 7 类幻觉类型的对抗性问答/字幕样本。

**📈 对比分析**

对 15 种系统（含 2 种商业 VLM、6 种开源 VLM、3 种视频代理、4 种专用检测器）进行评估，结果显示四种专用检测器的最高整体准确率仅为 34.63%，而 Gemini‑3‑Flash 等顶级 VLM 的整体准确率可达 83.63%。

**⚠️ 局限性**

局限性包括：样本仅为单一对抗性细节，未覆盖多重错误交互；类别分布均衡但不一定反映真实部署频率；未覆盖长时段视频，无法检验跨段证据整合能力。

---

## 244. RESCUE-BENCH: Towards Relation-Aware Multi-Party Emotional Support Conversation Systems

**arXiv ID:** 2609.09657 | [PDF](https://arxiv.org/pdf/2609.09657v1)

**作者:** Haichuan Hu `[一作]` (Hong Kong Polytechnic University), Wenjie Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种新的关系感知情感支持对话任务（RESCUE-Bench），通过构建包含夫妻和家庭对话的多模态数据集，并设计六个评价子任务，评估大型语言模型（LLM）在多方关系环境下的情感理解与支持决策能力。

**💡 创新点**

创新点主要有：① 将情感支持从单一寻求者转向多方关系场景，强调关系动态对支持效果的影响；② 设计“关系理解”与“关系敏感支持”两大维度共六个任务，系统化考察模型的关系推理与决策；③ 构建基于真实夫妻/家庭访谈的视频/对话数据，首次公开关系感知情感支持基准；④ 通过多模态结构化标注（时间、语义、个体线索、关系姿态、治疗策略等）为后续研究提供细粒度资源。

**🔧 技术方法**

主要技术包括：使用大型语言模型（Qwen、DeepSeek、GPT‑4o 等）在零样本下完成任务；利用 GPT‑5.4 作为“LLM‑as‑Judge”评估生成式理解任务；BERTScore 评估语义相似度；对多模态交互段落进行结构化表示；对关系模式进行转移矩阵分析；在任务中引入多层模型来捕捉个体情绪、关系态度及支持时机与策略。

**📊 数据集**

数据集：RESCUE‑Bench，来源于公开的夫妻与家庭访谈视频，包含 191 个样本（174 夫妻、17 家庭）、7,079 个标注回合、总计 1,064.8 分钟视频。每个样本按时间切分为多模态交互段落，并配备 6 维标注（时间实体、语义内容、个体线索、关系姿态、治疗策略、关系模式）。

**📈 对比分析**

比较方法：在十种先进 LLM 上进行零样本评测，分别测评 ER、VP、RPP、ITP、STP、SSP 六项任务。结果显示：在局部情绪或干预提示相关任务（ITP、ER）上性能较好（平均 ITP F1≈82.6%，ER 4.05/5）；但在需要关系推理的任务（RPP 最高 45.6%，VP 3.58/5，SSP 仅 31.9% recall）表现显著低于 ER，表明现有 LLM 在处理关系动态与制定关系敏感支持策略方面仍存在显著不足。

**⚠️ 局限性**

局限性：① 数据来源受媒体制作与编辑偏差，缺乏文化与人口多样性；② 关系模式、观点、支持目标与策略等标签高度主观，标注一致性与可靠性仍有提升空间；③ 某些标签呈长尾分布，可能影响模型训练与评估；④ 未公开原始视频、音频与视觉素材，限制了完全多模态复现与进一步研究；⑤ LLM‑as‑Judge 的客观性与一致性仍待进一步验证。

---

## 245. Stencil Computation at the Intersection of AI and HPC

**arXiv ID:** 2609.10368 | [PDF](https://arxiv.org/pdf/2609.10368v1)

**作者:** Timothee Ewart `[一作]`, Mauricio Araya-Polo `[通讯]` (TotalEnergies EP Research and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估了TinyTC与PyTorch/Triton在8阶25点声学光栅算子上的性能，并将其与SYCL、OpenMP等传统实现做对比。

**💡 创新点**

首次将面向AI的张量编译器应用于高阶科学算子，探讨了硬件专用与可移植两条路径的互补性。

**🔧 技术方法**

使用TinyTC（Intel GPU低层DSL）、Python/Triton（高层张量DSL）、SYCL共享内存、OpenMP offload等编程模型，并通过roofline、内存层级计数进行性能分析。

**📊 数据集**

在800³网格（含27层PML边界）上进行多步时间推进的声波模拟，采用随机与全零两种初始化策略来测试压缩效果。

**📈 对比分析**

在Intel Arc B580等平台上，TinyTC在零初始化时可达35.8 Gpts/s，Triton为32.5 Gpts/s；在随机初始化时分别为15.6与13.5 Gpts/s。跨平台结果显示，TinyTC在Intel GPU上性能最高，Triton保持良好跨厂商可移植性，SYCL与OpenMP则作为可编程性与易移植的基线。

**⚠️ 局限性**

受限于TinyTC仅支持Intel GPU、Triton未充分利用2D块加载、SYCL与OpenMP缺乏细粒度缓存调度，导致在非Intel平台上性能不如预期；此外，内存压缩仅在Intel GPU可用，进一步影响跨平台比较。

---

## 246. The Hyperbolic Surface Distance, Diameter, and Dirichlet Problems

**arXiv ID:** 2609.09549 | [PDF](https://arxiv.org/pdf/2609.09549v1)

**作者:** Vincent Despre `[一作]`, Marc Pouget `[通讯]` (Université de Lorraine)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一套针对任意高 genus 哈普伯平面曲面（给定 Dirichlet 域）的高效距离与直径计算算法，并给出了精确直径的代数可计算性证明。

**💡 创新点**

创新点在于：①将 Chen‑Han 的窗口化连续 Dijkstra 方法迁移到无多面体结构的光滑曲面；②利用波前传播隐式构造基于源点的 Dirichlet 域；③将直径问题转化为有限个代数方程组，证明 cosh(diam(S)) 是输入群生成元的代数扩张上的代数数。

**🔧 技术方法**

主要技术包括：波前传播（sequence‑tree）算法、Dirichlet 域与伪‑网（pseudo‑net）离散化、Voronoi/加权 Voronoi 结构、厚薄分解、墙（wall）排列与半代数优化。

**📊 数据集**

实验和分析基于理论构造的 Dirichlet 域与伪‑网，未给出具体真实数据集；所有性能结论均来源于理论复杂度分析。

**📈 对比分析**

与传统多面体曲面最短路算法相比，本文的 O(g²) 距离计算和 O(g² log g) 单源查询在理论上更高效；直径近似算法在厚曲面上以 O(g³ log g / ε²) 时间完成，且精度可调；但精确直径算法虽可计算但指数级时间，难以实际使用。

**⚠️ 局限性**

主要局限在于：①需要 Dirichlet 域与伪‑网的构造；②精确直径算法指数时间，实际不可扩展；③在薄带（thin collars）中需额外处理，导致伪‑网大小与 ε 成反比，可能造成内存与时间膨胀。

---

## 247. Identifying Habit, Physics, and Nuisance in Robot World Models

**arXiv ID:** 2609.09210 | [PDF](https://arxiv.org/pdf/2609.09210v1)

**作者:** Jinting Hang `[一作]` (Harvest Praxis), Zhenhui Cai `[通讯]` (Harvest Praxis)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种结构因果模型（SCM），将机器人学习中的多模态预测问题拆解为操作者习惯（habit）、共享物理动力学（physics）和观测噪声（nuisance）三部分，并给出一种冻结共享动力学、仅更新薄层接口（freeze‑plus‑interface）的适配策略；

**💡 创新点**

创新点在于：1）通过单一SCM清晰区分习惯、物理与噪声的因果关系；2）设计了针对物理面和习惯面分别的干预检验；3）提出了在低样本或数据受损场景下更稳健的冻结动态+微调接口方案；

**🔧 技术方法**

技术主要包括：结构因果建模、行动与观测的干预实验、对抗性“错误行动”损失、逆向评分（reverse‑scoring）辅助可行过去序列的排名、以及多步预测和像素级低样本适配；

**📊 数据集**

使用了三大公开数据集：ManiSkill StackCube（仿真）、DROID（户外RGB+关节数据）、RH20T（多操作者关节数据）；

**📈 对比分析**

在三组实验中：① 在StackCube上，冻结动态+训练接口在低样本（N≤64）下误差低于全微调或从零训练；② 在DROID关节和像素实验中，冻结策略对错误行动的敏感性明显高于合并模型，并在低样本（N=32）时比从零训练更好；③ 在RH20T多操作者实验中，交换用户或打乱行动导致误差激增，验证SCM假设；总体而言，冻结+接口方案在低样本、数据受损、以及跨摄像头场景下表现更稳健；

**⚠️ 局限性**

局限性包括：1）习惯干预仅为代理，未构建显式习惯识别器；2）跨摄像头性能仍显著受限，提示对观测噪声处理不足；3）闭环验证仅在仿真中完成，缺乏真实机器人部署的验证；4）逆向评分虽提升排名但不具备高识别精度。

---

## 248. Beyond Contact Sensors: Deep learning with Pseudo-Labeling for remote Photoplethysmography

**arXiv ID:** 2609.10026 | [PDF](https://arxiv.org/pdf/2609.10026v1)

**作者:** Bhargav Acharya `[一作]` (Bielefeld University), Hanna Drimalla `[通讯]` (Bielefeld University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究评估无监督信号处理提取的伪标签是否能替代有标注的接触式PPG，作为深度学习rPPG模型的训练信号。

**💡 创新点**

证明在同步不完善的数据集上伪标签可优于接触式标注，并指出伪标签质量对跨数据集泛化至关重要。

**🔧 技术方法**

使用TS-CAN和Physnet两种深度学习架构，并通过POS方法生成伪标签。

**📊 数据集**

采用公开的CHILL和PURE两个视频-PPG数据集。

**📈 对比分析**

通过交叉验证和跨数据集评估比较三种训练信号，发现伪标签在CHILL内测优于Finger-PPG，但在PURE跨测效果受伪标签质量影响，且在同步良好的数据集上跨域泛化仍存在差距。

**⚠️ 局限性**

局限在于伪标签质量评估仍需真实标注，且在同步良好时跨域泛化性能不稳定，限制了其在真实场景的直接应用。

---

## 249. Learning to Fly: Stable Vision-Guided UAV Servoing with Compact Target-Centric Cues and Reinforcement Learning

**arXiv ID:** 2609.09234 | [PDF](https://arxiv.org/pdf/2609.09234v1)

**作者:** Saurbh Singh Jamwal `[一作]` (Indian Institute of Technology Bombay), Nived Chebrolu `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究使用轻量级目标中心化视觉特征和低维传感器测量实现无人机长时程视觉伺服控制的强化学习框架。

**💡 创新点**

创新点在于提出仅使用目标掩码得到的图像空间偏移和相对深度与运动传感器融合的12维观测表示，并通过视觉、动力学和联合三种课程学习比较其对稳定性和鲁棒性的影响。

**🔧 技术方法**

采用Proximal Policy Optimization（PPO）与三种课程学习（视觉、动力学、联合）、轻量级颜色分割、IsaacLab仿真环境以及经典视觉伺服基线。

**📊 数据集**

使用IsaacLab仿真环境中的RGB‑D图像与目标掩码生成的合成数据；未使用公开数据集。

**📈 对比分析**

与直接PPO以及三种课程策略在相同优化预算下对比；实验表明三种策略实现相似的长期奖励，但视觉课程在目标运动变化和噪声扰动下鲁棒性最佳；与经典伺服基线相比，RL策略虽命中精度略低，但在扰动和未知目标运动下表现更稳健。

**⚠️ 局限性**

主要局限在于仅在仿真中验证，使用简化的分割与深度，未涵盖复杂环境、多目标、真实传感器噪声、延迟或相机外参变化等情况；因此实地部署仍需进一步验证。

---

## 250. FolDeX: A Physical-World Benchmark for Long-Horizon Robotic Manipulation of Deformable Objects

**arXiv ID:** 2609.10243 | [PDF](https://arxiv.org/pdf/2609.10243v1)

**作者:** Chenhuan Liu `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了FolDeX这一长周期变形物体（主要是服装折叠）真实机器人基准，并建立了FoldChallenge评测平台，提供2000+小时多任务、多机器人、多场景的完整实验数据；

**💡 创新点**

创新点在于：①首个完整的长周期变形物体基准；②围绕回收数据、任务、场景和体型四个轴组织数据以系统评估数据重用；③统一评测协议与FoldScore指标，使不同方法可在同一物理环境下公平比较；

**🔧 技术方法**

采用了视觉‑语言‑动作（VLA）接口的π_0策略，结合实时块化训练（RTC）与DAgger回收数据，利用多机器人、多场景同步视觉与本体感知进行训练；

**📊 数据集**

使用FolDeX数据集，包含2000+小时的真实机器人操作，覆盖20+任务、10+机器人形态，涵盖服装折叠、其他变形物体与部分刚体任务，并按回收、任务、场景、体型划分；

**📈 对比分析**

通过FoldChallenge统一评测，指标包括成功率、完成时间、最终整齐度和综合FoldScore；实验显示多任务RTC版π_0平均FoldScore达75.59，加入回收数据后成功率提升至95%、FoldScore提升至82.53；跨任务/体型实验表明当前策略易出现负迁移；

**⚠️ 局限性**

局限性包括：实验仅在有限硬件与时间内完成，未覆盖触觉感知；跨体型和跨任务的联合训练导致严重干扰与灾难性遗忘；评测仅基于视觉+本体感知，未来需引入更多传感与更复杂的数据重用方法。

---

## 251. Adversarial Training for Tabular Credit Scoring: A Multi-Attack Robustness Evaluation in P2P Lending

**arXiv ID:** 2609.09945 | [PDF](https://arxiv.org/pdf/2609.09945v1)

**作者:** Gijs A. F. Niewzwaag `[一作]` (University of Twente), Marcos R. Machado `[通讯]` (University of Twente)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对P2P借贷中的信用评分模型进行系统的对抗鲁棒性基准评估，考察在不同攻击类型下模型性能的变化。

**💡 创新点**

创新点在于构建完整的训练‑测试对照网格，涵盖四种对抗攻击（FGSM、PGD、DeepFool、Salt‑and‑Pepper）和混合训练策略，并在三类模型（逻辑回归、前馈神经网络、表格Transformer）上展开跨攻击泛化研究。

**🔧 技术方法**

采用对抗训练技术，结合梯度攻击（FGSM、PGD、DeepFool）与非梯度扰动（Salt‑and‑Pepper），并使用表格Transformer模型、批归一化、dropout等深度学习组件；实验流程基于CRISP‑ML生命周期并使用5折分层交叉验证。

**📊 数据集**

使用Lending Club贷款数据集（约400k条样本，28个特征）以及在Prosper数据集上的复现实验，确保实验可复现且具备行业代表性。

**📈 对比分析**

通过在干净数据和不同攻击生成的测试集上比较ROC‑AUC、准确率、召回率、精确率和F1；结果显示匹配攻击的对抗训练可将ROC‑AUC提升至0.95–0.98，梯度攻击训练在梯度族内部转移良好，但对Salt‑and‑Pepper的跨族泛化弱；混合训练在保持清洁性能的同时提供最均衡的鲁棒性。

**⚠️ 局限性**

局限性包括仅使用单一数据子集且预处理固定，攻击模型简化且未充分体现真实操纵约束；缺乏黑盒攻击与时间漂移的评估，且对行业内部验证与监管合规的细节探讨不足。

---

## 252. Fast Algorithms for Sparse PCA and Robust Sparse Estimation

**arXiv ID:** 2609.09701 | [PDF](https://arxiv.org/pdf/2609.09701v1)

**作者:** Giannis Iakovidis `[一作]` (University of Wisconsin Madison), Ankit Pensia `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了稀疏主成分分析（Sparse PCA）认证的快速算法，给定一个正半定矩阵M，问题是要么排除一个大的k稀疏二次型，要么返回一个高值（放松的）证据。

**💡 创新点**

提出了一种双标准算法，运行时间为O(d^2 + d k^O(log k))，在某些条件下，运行时间为O(d^2)，并且在样本访问模型中突破了二次障碍。

**🔧 技术方法**

使用了半正定松弛（SDP）和快速相关检测技术。

**📊 数据集**

使用了d维的样本数据集，样本数量为n=d^o(1)。

**📈 对比分析**

与现有的标准SDP方法相比，提出的方法在时间复杂度上有显著改进，标准方法需要Ω(d^4)的时间，而本研究的方法在特定条件下可以在O(d^2)时间内完成。

**⚠️ 局限性**

算法在处理一般输入时可能会失败，尤其是在没有额外结构的情况下，且在某些情况下可能需要更多的样本以保证性能。

---

## 253. Deformable Object Manipulation under Partial Observability via Real-Time Full-Shape Estimation

**arXiv ID:** 2609.10308 | [PDF](https://arxiv.org/pdf/2609.10308v1)

**作者:** Kosar Behnia `[一作]` (Tampere University), Gokhan Alcan `[通讯]` (Tampere University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种轻量级的条件递归变分自编码器（cRVAE），能够仅利用稀疏的角点节点观测实时估计变形物体的全状态，并将其作为前向模型嵌入到递归-基准控制框架中实现障碍物感知的协同操控

**💡 创新点**

创新点在于：①利用条件递归VAE在推理阶段去除编码器，仅依赖隐层和角点反馈实现全形态估计；②实现无物理参数、无在线辨识的高速预测（≈9 ms/步，≈1.5 ms/步），并通过测量残差反馈校正隐状态，显著降低漂移；③将此模型与MPC/iLQR结合，在受限观测下完成长周期障碍物规避任务

**🔧 技术方法**

使用技术包括变分自编码器、条件先验、GRU递归网络、测量残差反馈机制、与iLQR/MPPI等递归前瞻控制器的结合；实现基于JAX的自动微分与JIT加速以便快速求导

**📊 数据集**

数据集主要来自：①使用XPBD物理模拟器生成的绳索（N=33）和织物（N=225）轨迹（10条/29条），②Unitree Go2实验平台的实测六点位姿数据（8套），用于模型校准和实地验证

**📈 对比分析**

与参数辨识的XPBD物理模型以及几何基线（相似变换）对比，cRVAE在h=40时的MAE分别为2.51 cm（绳）/0.98 cm（织物），远优于基线；计算时间上对比XPBD，cRVAE快≈350×（绳）/≈1550×（织物），完全满足100 ms控制预算；在真实机器人实验中，10/10次任务成功且无碰撞，计算时延约65 ms/步，符合实时要求

**⚠️ 局限性**

局限性包括：①仅利用角点观测，内部节点信息缺失仍会影响精度；②目前未建模与环境的接触动力学，导致在高接触复杂场景下可能出现误差；③依赖高质量的模拟数据与校准，跨域迁移仍需进一步验证

---

## 254. CityPlanner: A Sandbox Agent for Executable Urban Planning

**arXiv ID:** 2609.09578 | [PDF](https://arxiv.org/pdf/2609.09578v1)

**作者:** Wentao Zhang `[一作]` (Beihang University), Wenrui Wang `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种沙盒代理框架 CityPlanner，用于可执行的城市规划任务，能够在文件化环境中生成、评估并迭代改进规划方案。

**💡 创新点**

创新点在于提出统一的文件化沙盒环境 UrbanSandbox，并将长期规划任务拆分为 BuildPlan 与 ImprovePlan 两个原子任务，通过原子任务强化学习与迭代部署显著提升方案可行性与质量。

**🔧 技术方法**

采用大语言模型 Qwen3（8B/14B）结合监督微调（SFT）、GRPO、Atomic-Task RL（ATRL）等技术，并在沙盒中执行脚本、调用评估器实现交互式规划。

**📊 数据集**

使用基于 OpenStreetMap 的真实城市数据，构建了 5,670 个规划实例，覆盖土地分配、道路建设和站点布局三类任务。

**📈 对比分析**

与启发式方法、任务专用强化学习以及通用 LLM 代理等基线对比，CityPlanner 在 9 个设置中取得 8 个最佳得分，显著提高可行率与目标得分，同时保持中等运行时。

**⚠️ 局限性**

局限性包括较高的计算成本（多轮沙盒交互）以及缺乏理论最优性保证，最终解受模型能力、奖励设计和评估器反馈的影响。

---

## 255. RoboDrop: Curating VLA Post-Training Data via Local Gradient Compatibility

**arXiv ID:** 2609.10021 | [PDF](https://arxiv.org/pdf/2609.10021v1)

**作者:** Runze Xu `[一作]` (Tsinghua University), Jincheng Yu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出RoboDrop框架，利用训练轨迹中的梯度兼容性在线评估并筛选VLA后训练数据；

**💡 创新点**

创新在于：①基于任务语义与视觉上下文构造局部梯度参考；②在一次热身训练期间按梯度方向在线评分并聚合成episode级分数；③使用BIC自动判断筛选比例；

**🔧 技术方法**

技术主要包括流匹配训练、梯度归一化余弦相似度、DINO特征检索、CountSketch压缩、BIC与GMM模型；

**📊 数据集**

使用LIBERO、Robomimic-MH以及真实机器人收集的四项任务数据集；

**📈 对比分析**

与长度、行为检索、DataMIL、Scizor、QoQ等基线比较，RoboDrop在AUROC与下游成功率上均显著领先（最高提升至95%以上的成功率、AUROC≈96%）；

**⚠️ 局限性**

局限在于需要预先获取并验证一小批干净参考数据，且对极端噪声或非典型错误仍需进一步验证。

---

## 256. Fast Collision-Free Acquisition in 1-Persistent Age-Threshold Slotted ALOHA via Role-Protected Counters

**arXiv ID:** 2609.09934 | [PDF](https://arxiv.org/pdf/2609.09934v1)

**作者:** Plínio Santini Dester `[一作]` `[通讯]` (São Paulo State University), Plínio Santini Dester (São Paulo State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种角色保护的计数阈值 ALOHA（RP–CTSA）协议，用于快速收敛到目标导向更新的周期性无碰撞调度，并在整个过程中保持较低的 Age‑of‑Information（AoI）。

**💡 创新点**

创新点在于将预约计数与信息年龄分离，保留已学习的单例调度阶段，利用单比率（RELEASE/HOLD）即时释放多节点碰撞，并给出严格的纯死亡链收敛分析，获得 O(n²) 或 O(n log n) 的收敛时间与阈值比例的闭式极限。

**🔧 技术方法**

采用了纯死亡链理论、机会时间几何分布、事件跳过蒙特卡罗模拟、期望与分布收敛分析以及 Euler‑Gamma 函数等数学工具。

**📊 数据集**

使用自行编写的离散事件模拟器进行实验，未采用公开数据集，实验数据来自理论模型的随机试验。

**📈 对比分析**

通过与 1‑pTSA、信息丰富的 L‑ZC 和理想 TDMA 进行对比，结果显示 RP‑CTSA 在同步起始条件下收敛时间显著短于 1‑pTSA（从 O(n²) 下降到 O(n log n) 或更低），短期 AoI 亦大幅低于 1‑pTSA，略逊于 L‑ZC 但在信息量更少的前提下性能仍优。

**⚠️ 局限性**

局限性包括：需要理想的角色检测与同步、无误差的 ACK/RELEASE 反馈；仅在同步起始条件下给出严格收敛定理，随机起始状态下收敛行为未证明；未考虑信道噪声、功率约束及实际硬件实现的开销。

---

## 257. Quantifying IIoT Sensor Node Criticality by Fusing its Data Criticality and Security Vulnerability

**arXiv ID:** 2609.09807 | [PDF](https://arxiv.org/pdf/2609.09807v1)

**作者:** Sachin K. Sen `[一作]` (Unitec Institute of Technology), Shaoning Pang `[通讯]` (Federation University Australia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在工业物联网（IIoT）环境中评估传感器节点重要性的综合方法，融合了传感器数据的关键性与其安全漏洞评分，利用Dempster–Shafer理论进行融合与评估。

**💡 创新点**

创新点在于首次将数据重要性与CVSS安全漏洞评分相结合，并采用Dempster–Shafer证据理论给出节点重要性的可信区间，从而实现更全面、可解释的安全优先级排序。

**🔧 技术方法**

使用了Dempster–Shafer证据理论、CVSS 4.0与3.1漏洞评分、贝叶斯/可信度计算，以及基于统计/机器学习的葡萄酒质量与传感器特征关联分析。

**📊 数据集**

采用了公开的红葡萄酒特征与质量评分数据集（包含11个关键参数）作为传感器数据来源，并为每个传感器计算对应的CVSS安全漏洞评分。

**📈 对比分析**

通过将数据关键性与安全漏洞分数进行融合前后进行对比，利用Belief（信度）与Plausibility（可信度）得分进行节点排名；结果显示使用CVSS 4.0融合的Belief值更高、范围更窄，说明融合方法更稳健、能更好地区分关键节点。

**⚠️ 局限性**

局限性包括：数据与漏洞评分的主观性和偏差、样本量有限、Dempster–Shafer假设的局限、可推广性受限，以及CVSS评分依赖公开信息，可能未完全反映真实威胁。

---

## 258. AccelMPC: High-Rate, Low-Power FPGA-Accelerated Model Predictive Control for Tiny Drones

**arXiv ID:** 2609.09380 | [PDF](https://arxiv.org/pdf/2609.09380v1)

**作者:** Andrea Grillo `[一作]` (Dartmouth College), Brian Plancher `[通讯]` (Dartmouth College)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并验证了一种针对极小型无人机的FPGA加速线性模型预测控制（MPC）框架，支持1 kHz实时约束轨迹跟踪。

**💡 创新点**

通过端到端硬件‑算法协同设计，实现了对ADMM‑MPC求解器的FPGA加速、固定点实现、预缓存Cholesky分解与自定义PCB集成，显著提升速度与能效。

**🔧 技术方法**

采用ADMM求解线性二次规划、固定点FPGA实现、Cholesky分解预缓存、两种硬件映射（展开与重构）以及自定义四层PCB与高带宽SPI通信。

**📊 数据集**

使用Crazyflie 2.1无人机与OptiTrack摄像头捕捉位置数据，构造动态障碍物和约束轨迹实验。

**📈 对比分析**

与基准TinyMPC MCU求解器比较，FPGA实现速度提升至15.6×、能耗降低至195.4×；在1 kHz控制率下完成约束MPC，支持超过20,000个优化变量。

**⚠️ 局限性**

受限于FPGA本地存储、固定点精度以及对更复杂三维碰撞约束的适应性；对高精度控制和更大规模问题仍需进一步研究。

---

## 259. Edu-QuRating: Multi-Dimensional Educational Data Curation with Distilled Pairwise Judgements

**arXiv ID:** 2609.09425 | [PDF](https://arxiv.org/pdf/2609.09425v1)

**作者:** Oliver G. B. Garrod `[一作]` (Fab AI), Paul Atherton `[通讯]` (Fab AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了Edu-QuRating，基于多维教育评判体系的过滤与打分管线，利用LLM的成对偏好判定并蒸馏成单文本的Edu-QuRater模型，随后将其应用于大规模数据集的过滤用于小型模型的预训练，以及作为GRPO强化学习的奖励模型；

**💡 创新点**

创新点在于将教育价值拆分为核心维度（整体教育性、准确性、参与度、教学结构等）以及学生/教师针对的基础读写维度，使用LLM成对偏好学习而非单一标量；同时通过神经Bradley–Terry目标将偏好蒸馏成可扩展的单文本评分器，实现高效大规模评估；

**🔧 技术方法**

技术上采用LLM判断器（如GPT-4.1-mini/4、Qwen3-4B）进行成对评判，使用Sheared-LLaMA-1.3B和Gemma-3-4B-PT作为序列分类基础模型，训练时使用神经Bradley–Terry损失；随后在预训练中使用nanotron框架，在GRPO中结合Edu-QuRater与结构化奖励进行强化学习；

**📊 数据集**

使用的数据集包括FineWeb-Edu-Fortified（约3.2亿条文档）、FineWeb-Edu基准、DataComp-LM、外部教育素材集以及教师任务与高质量输出的1008条样本；

**📈 对比分析**

评估方式为：先在held-out对照中计算Edu-QuRater的pairwise准确率（平均0.917），再在单跑预训练30k步对比FineWeb-Edu基准，Edu-QuRating混合模型取得平均准确率0.3903（高于基准0.3806），在ARC‑CF、HellaSwag等任务上显著提升；在GRPO评估中，结合Edu-QuRater与答案结构奖励的模型在教学质量和指令遵循的win率分别达81.08%和68.24%，优于单一奖励方案；

**⚠️ 局限性**

局限性包括：仅覆盖英文语料，缺乏多语言、不同课程体系支持；预训练实验仅进行单次跑，未探索更广泛的混合组合；GRPO评估规模有限，且奖励可能偏向表面结构而忽视内容准确性；LLM评判与奖励需要昂贵计算，且无法弥补源数据缺失的语言或题材。

---

## 260. No Free Checker: A Survey of Verifiers for Robot Policies

**arXiv ID:** 2609.09250 | [PDF](https://arxiv.org/pdf/2609.09250v1)

**作者:** Yang Wan `[一作]` (Zhejiang University), Linchao Zhu `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了约150篇关于机器人学习中验证器（verifier）的研究，按判定来源将其分为人类验证器、基于规则与形式的验证器、学习与预训练的验证器以及模型本身内在的验证器，并在可用性与可信度维度对四类进行系统比较。

**💡 创新点**

创新点在于提出了“可用性-可信度”双维度框架，对不同验证器来源进行统一评价；同时系统梳理了验证器在训练、推理和评估中的多重角色，并给出了验证器性能评估的九个关键指标。

**🔧 技术方法**

技术手段包括：人类标签采样、基于时序逻辑/几何的规则判定、深度学习的奖励模型与进度估计、世界模型预测与自我检测、以及基于控制边界函数和可达性值的安全过滤。

**📊 数据集**

使用的数据集主要为模拟环境与真实机器人收集的轨迹（如LIBERO、CALVIN、RoboArena等）以及公开的机器人学习基准（ManiRewardBench、RoboReward、RoboFAC、OpenGVL等），同时涉及人类标注的轨迹比较、评分和干预数据。

**📈 对比分析**

比较方法：对不同验证器按同一套指标（如一致率、下游策略性能提升、错误率、跨体型转移等）进行打分；实验结果表明人类验证器虽然最具直接性但成本高；基于规则的验证器成本低但受限于精确状态估计；学习/预训练验证器成本低且可扩展，但可信度依赖训练数据；模型本身内在验证器最便宜但与任务成功的关联最弱。

**⚠️ 局限性**

局限性包括：验证器往往只在模拟或有限数量的真实轨迹上评估，难以保证在不同机器人/任务/环境下的泛化；对人类标签的误差未充分报告；在优化（如奖励模型训练）中可能出现“奖励黑客”导致误导；未给出统一的基准来评估验证器在搜索对抗条件下的鲁棒性。

---

## 261. Belief-State Engine: Augmenting LLMs for Principled Planning Under Partial Observability

**arXiv ID:** 2609.10036 | [PDF](https://arxiv.org/pdf/2609.10036v1)

**作者:** Arnab Chattopadhayay `[一作]` (University College London), Debdipta Halder `[通讯]` (Indian Institute Of Technology Kharagpur)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了Belief-State Engine（BSE），一种将贝叶斯推断模块与大型语言模型（LLM）解耦的架构，使LLM仅在得到精确的贝叶斯后验（belief）后做决策，从而在部分可观测环境下实现了原则化规划；

**💡 创新点**

创新点在于构建了四条最小公理化的贝叶斯一致性规范，并证明了在满足这些公理时，LLM+BSE组成的系统等价于贝叶斯后验马尔可夫决策过程，给出了可验证的Bellman最优性保证；

**🔧 技术方法**

技术上结合了POMDP理论、两步贝叶斯滤波、四公理框架、belief序列化与LLM策略接口，并通过实验对比了反应式、CoT、ReAct、自然语言belief跟踪器、QMDP和POMCP等基线；

**📊 数据集**

实验数据集包括经典Tiger POMDP（二状态）和红队攻击图基准（K=4、6，状态空间至多64维）；

**📈 对比分析**

采用配对种子比较法，使用回报、贝叶斯校准、决策一致性及计算成本等指标，结果显示BSE在Tiger任务上显著提升了成功率和平均回报，并在决策一致性上优于自然语言跟踪器；在攻击图任务上虽回报差距不大，但在决策一致性和网络侵入覆盖率上表现更好；

**⚠️ 局限性**

局限性包括：需要已知的POMDP模型；仅适用于有限状态空间；仅在单代理、静态环境下验证；实验规模有限（N=40/25种子、单温度样本），未覆盖所有基线或完整的十项消融；未完成开源模型的复制验证；模型误差或不确定性会破坏理论保证。

---

## 262. Socio-technical and Ethical Dimensions of Architecture Practices in FLOSS

**arXiv ID:** 2609.09975 | [PDF](https://arxiv.org/pdf/2609.09975v1)

**作者:** Sven Thielen `[一作]` `[通讯]` (Heinrich Heine University Düsseldorf), Sven Thielen (Heinrich Heine University Düsseldorf)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究通过三阶段方法（案例研究、框架与干预设计、试点评估），对自由/开源软件（FLOSS）项目中的架构实践、治理结构、沟通方式及伦理维度进行多方法研究，并提出了将架构视为社会技术与伦理工作的新框架和轻量化实践与教学模式。

**💡 创新点**

创新点在于：① 将架构实践与治理、沟通与伦理因素系统整合，形成跨案例的概念框架；② 设计并评估可在FLOSS社区与课程中直接落地的轻量化架构文档模板、参与路径和教学案例；③ 在研究设计中融合AI/LLM对架构决策的潜在影响。

**🔧 技术方法**

采用的技术包括：仓库挖掘与静态分析（生成依赖图、聚类重建架构视图）、多源文本分析（问题/PR/邮件列表关键词过滤与标签识别）、定性编码与主题分析、半结构化访谈、共设计工作坊以及原型实验的迭代评估。

**📊 数据集**

使用的数据集包括：3–4对域级FLOSS项目（如LibreOffice/OpenOffice、GnuPG/Sequoia-PGP、Pacman/Zypper等）的源代码仓库、ADR与设计文档、Issue/PR与邮件列表记录、访谈转录、以及相关课程的教学大纲与作业材料。

**📈 对比分析**

方法比较以跨案例定性对比为主，使用一系列可量化代理（治理文件存在度、角色权限矩阵、参与者多样性、决策权分布、伦理关注度、可追溯性指标）来描述不同项目的架构工作特征。因研究聚焦经验与过程，未提供传统意义上的性能数值，而是通过案例间的模式对比揭示优势与挑战。

**⚠️ 局限性**

局限性包括：案例数量有限、可能存在选择偏差、访谈回应率与访谈数据的主观性、架构推理的完整性难以完全通过追踪获得、AI/LLM影响可能未被完全捕捉、结果对不同治理模型的泛化受限、缺乏严格的定量效能评估。

---

## 263. A 2.37332-Competitive Algorithm for Online Square Packing with Gravity

**arXiv ID:** 2609.10101 | [PDF](https://arxiv.org/pdf/2609.10101v1)

**作者:** Nichlas Langhoff Rasmussen `[一作]` `[通讯]` (Alpha Energy ApS), Nichlas Langhoff Rasmussen (Alpha Energy ApS)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于非对称槽层次划分的递归算法，解决了在线 Tetris‑与重力约束下的轴对齐正方形和固定宽高比矩形在单宽条带中的装箱问题，并给出了相应的竞争比分析。

**💡 创新点**

创新点在于：①用非对称槽划分（宽槽 pw 与窄槽 qw）代替传统对称分割，能够实现局部充电式分析；②通过局部相位不变式和阶段性计费证明了对正方形的竞争比 ρ*≈2.3733，显著低于以往的 2.6154；③将同一框架推广至宽高比 ≤ κ 的矩形，并证明竞争比上界为 O(κ)，与下界 Ω(κ) 完全匹配；④给出了匹配的下界构造。

**🔧 技术方法**

使用的技术主要包括：递归非对称槽层次划分、虚拟高度剖面、Tetris‑与重力可达性证明、局部相位不变式与阶段计费、以及代数/数值优化以求得最优分割参数 p*。

**📊 数据集**

该工作为理论分析，无需实际数据集；所有结论均基于输入序列的符号表示和解析推导。

**📈 对比分析**

与之前的 SlotAlgorithm（竞争比 34/13≈2.6154）相比，新算法在正方形情形下将竞争比降低至 2.3733；对宽高比 ≤ κ 的矩形，竞争比从无界提升到 O(κ)，与下界 Ω(κ) 一致；实验/模拟未涉及，性能评估完全基于理论竞争比。

**⚠️ 局限性**

主要限制包括：①正方形的竞争比仍未达到下界 2，尚存在 0.3733 的余差；②对一般矩形的上界为 O(κ)，常数项尚未最优；③方法依赖于非旋转假设，无法直接推广至可旋转矩形；④在极端参数范围（如 p→1/2）下分析复杂度升高。

---

## 264. Assembling Two Parts in One Hand

**arXiv ID:** 2609.10137 | [PDF](https://arxiv.org/pdf/2609.10137v1)

**作者:** Liuao Pei `[一作]` (University of Hong Kong), Jie Song `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了单只多指机器人手在无外部支撑或第二手臂的情况下完成两物体内手组装的任务，并通过强化学习实现对不同任务的统一控制；

**💡 创新点**

创新点包括①将指尖功能分配与辅助奖励相结合，形成可推广的组装框架；②利用单人手动作快照作为初始化与姿态约束；③在训练中采用观测域随机化和历史本体感知融合，显著提升在视觉遮挡下的鲁棒性；

**🔧 技术方法**

主要技术包括基于POMDP的强化学习（RNN策略），仿真平台IsaacSim，RGB‑D视觉状态估计（FoundationPose），以及多尺度目标奖励与指尖功能奖励；

**📊 数据集**

实验数据集为三种组装任务（瓶盖–瓶子、注射器塞–注射器、马克笔盖–马克笔）的仿真环境与真实硬件场景，未使用公开大规模数据集；

**📈 对比分析**

通过与仅本体感知、历史MLP、无指尖奖励、无姿态奖励等对照方案的消融实验，以及在不同手型下的零样本硬件转移，本文在Bottle、Syringe、Marker任务中实现闭环成功率分别为18/20、15/20、18/20，远优于开环重放（仅3/20）；

**⚠️ 局限性**

局限性在于：①仅聚焦于组装环节，未覆盖完整的多物体抓取与姿态调节流程；②任务数量有限，仅包含三种几何形状；③模拟对平面接触的建模不完整，导致某些倾斜角度下的抓取不稳；④对手型的适应性受限，未覆盖所有手指关节与柔软手掌的差异。

---

## 265. Looped GPT-BERT: Trading Parameters for Computation in Small Language Modeling

**arXiv ID:** 2609.09691 | [PDF](https://arxiv.org/pdf/2609.09691v1)

**作者:** Tingshuo Fan `[一作]` (Fudan University), Tao Ji `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在数据受限的 BabyLM Strict-small 任务中，作者提出 LoopGPT‑BERT，将 GPT‑BERT 的双重目标（掩码与因果语言建模）与深度循环参数共享相结合，使用少量物理层反复迭代来提升有效深度。

**💡 创新点**

创新点在于：① 在同一 Transformer 栈中同时训练掩码与因果任务；② 采用深度循环参数共享（4 层物理层迭代 12 次）实现高有效深度而不增大参数量；③ 通过实验系统评估循环深度与目标比例对语法、实体追踪等能力的影响。

**🔧 技术方法**

使用的技术包括：GPT‑BERT 双模式训练（MNTP 与 CLM），深度循环 Transformer（Universal Transformers 风格），BPE 分词器（8k 词表），LAMB 优化器，余弦学习率衰减，动态权重融合（DWA）。

**📊 数据集**

主要使用 7.48M 词的 BabyLM Strict‑small 语料（BNC Spoken、CHILDES、Project Gutenberg、OpenSubtitles、Simple Wikipedia、Switchboard）并经过规则+重写清洗。

**📈 对比分析**

在 BabyLM 2026 评测中，12.18M 参数的 4×12 LoopGPT‑BERT 在整体平均分 35.42、NLP 平均 48.48 与官方 31M GPT‑BERT 基线相比提升 7.25/13.18 分；在 BLiMP、GLUE 等任务上表现与更大基线相近；循环深度 4×6 在性能与推理速度上达到最佳折中。

**⚠️ 局限性**

限制包括：① 仅在单一 8k 词表与特定清洗流程下实验，缺乏对原始数据/不同 tokenizer 的 ablation；② 仅使用单一随机种子，结果可能受训练方差影响；③ 计算成本（层迭代）显著高于非循环模型；④ 复合目标比例会改变监督密度，导致不同任务能力偏移；⑤ 在实体追踪、年龄认知等任务表现仍不佳。

---

## 266. Watermarks Without Verification: AI Text Watermarking After the EU AI Act

**arXiv ID:** 2609.09604 | [PDF](https://arxiv.org/pdf/2609.09604v1)

**作者:** Alexander Nemecek `[一作]` (Case Western Reserve University), Erman Ayday `[通讯]` (Case Western Reserve University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对欧盟AI法案要求的文本水印实施进行了系统性评估与治理分析，评测了公开实现的SynthID-Text水印，并归纳了用户与供应商之间的争议、保证与可验证性缺口；

**💡 创新点**

创新点在于将争议与保证按可验证性分层（公开可验证、需供应商协作、需机构）进行分类，并在公开模型上完整实验水印对质量、正确性、检测率的影响，同时提出五项治理需求（匹配输出发布、配置公开、审计、标准化协议、跨供应商接口），强调治理缺口是导致信任危机的根源。

**🔧 技术方法**

使用了采样式文本水印技术SynthID-Text（非失真配置），结合MarkLLM工具进行水印嵌入与检测；通过统计指标（perplexity、TPR、AUROC、pass@1）评估质量与可检测性。

**📊 数据集**

采用Gemma-2-9B与Llama-3.1-8B两种开源权重模型；数据集为500个开放式自然语言提示和364个编程问题（来自两个公开基准），每个问题10个样本。

**📈 对比分析**

方法：三支实验 arms——（a）水印+相同种子，（b）无水印+相同种子，（c）无水印+不同种子；对 prose 计算 perplexity、对比 win-rate、语义相似度；对 code 计算 pass@1；对两类文本计算 1% FP阈值下的 TPR 与 AUROC。结果显示 prose 质量变化在±0.3%，无显著差异；代码正确性在 Llama 上下降约3个百分点，Gemma 近乎无影响；检测率对 prose 为 39‑59% TPR（AUROC 0.78‑0.82），对 code 仅 0.55‑0.57。

**⚠️ 局限性**

局限性包括：无法验证真实部署配置（密钥、熵阈值、强度）；缺乏公开匹配输出与评分检测器，导致无法独立验证质量与可检测性；缺少标准化评估协议、独立审计机构以及跨供应商统一接口，导致治理失效与信任缺失。

---

## 267. UnitBoost: Managing Compound LLM Systems with a Merge Operator, Not a Model

**arXiv ID:** 2609.09815 | [PDF](https://arxiv.org/pdf/2609.09815v1)

**作者:** Xing Zhang `[一作]` (AWS Generative AI Innovation Center), Peiyang He `[通讯]` (AWS Generative AI Innovation Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在复合LLM系统中，将传统的生成式管理器替换为一个确定性的元层操作符，该操作符通过对每个任务单位执行受约束的逐单位最大化来合并工作者输出，并记录单元来源、分数与可行性，随后生成残差供下一轮工作；

**💡 创新点**

创新点在于：①用受约束的逐单位argmax代替整体生成，消除生成式管理器的顺序敏感和不透明性；②提供显式残差分配机制，提升多轮协作效率；③保持单元可追溯性与可测试的失败条件；④在不增加模型调用的情况下实现超越候选选择上限的性能。

**🔧 技术方法**

采用任务定义的单元映射（unit map）、本地分数函数q、可行性谓词C、持久单元表、受限argmax求解、残差生成与门限停止；此外使用归一化值排序、信号融合（agreement、rank、retrieval-backed check）以及可配置的入选边际δ。

**📊 数据集**

在七个基准上评估：QAMPARI、ASQA、FanOutQA、ELI5、SWE‑bench、ClassEval以及合成的多模态检索/代码生成任务。

**📈 对比分析**

对比方法包括：基于生成式管理器（Claude Sonnet 5、GPT‑5.6 Sol、DeepSeek‑V3.2等）、oracle候选选择、混合聚合、辩论共识、批判式修订、以及现有的多代理协议。实验结果显示：在QAMPARI、ASQA和FanOutQA上，替换管理器可分别提高0.048–0.195点（相较于匹配生成器），且在六种复合系统配置中提升0.013–0.182点；在FanOutQA的单元级F1上从0.4778提升至0.5524。

**⚠️ 局限性**

局限性包括：①仅适用于任务可划分为可机械识别的单元；②当单元不可分割、单元身份不明确或端点对每个单元计价时，无法获益；③残差耗尽的检测存在延迟；④需要为每个任务提供可靠的分数与可行性判定；⑤对数据集外的语料或不同检索模式的跨域泛化尚未充分验证。

---

## 268. When Fusion Fails: Corruption-Aware Rebalanced Fusion for Multi-Modal Medical Image Segmentation

**arXiv ID:** 2609.10261 | [PDF](https://arxiv.org/pdf/2609.10261v1)

**作者:** Yuchen Pei `[一作]` (Central China Normal University), Gang Li `[通讯]` (University of North Carolina)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了一种针对空间对齐但分辨率不一致的多模态医学影像分割方法CoReFuse‑Med，提出抑制特征腐败并重平衡模态贡献的两阶段融合框架。

**💡 创新点**

创新点在于①识别“融合失效”现象并揭示优化‑推理不一致；②提出IMSH模块在浅层抑制重采样噪声、MBCF模块在深层校正模态偏差；③实现轻量化轴向上下文编码器LACE。

**🔧 技术方法**

技术方法包括U形网络架构、轻量化LACE编码器、空间尺度分解的IMSH、通道+空间自注意力的MBCF、梯度与遮挡分析以及F‑CNR指标。

**📊 数据集**

使用的数据集为EPVS、BraTS2020和WMH Challenge，其中EPVS为真实临床多分辨率数据，BraTS通过切片保留比例模拟分辨率差异。

**📈 对比分析**

与3D U‑Net、Swin‑UNETR、MedNeXt等通用骨干以及MMFormer、HNoSegXS等专用多模态网络对比，CoReFuse‑Med在EPVS、BraTS、WMH上分别提升DSC≈0.03、HD95下降≈2mm、IAVD/Recall/Precision提升，并显著降低参数量和计算量。

**⚠️ 局限性**

局限性包括仅针对空间对齐但分辨率差异的情况，未处理完全缺失模态，噪声实验仅为辅助，EPVS数据因隐私限制无法公开。

---

## 269. A Systematic Evaluation of Molecule Generation Models for De Novo Drug Design: From Benchmarks to Practical Insights

**arXiv ID:** 2609.10099 | [PDF](https://arxiv.org/pdf/2609.10099v1)

**作者:** Xinrui Xu `[一作]` (Xiangtan University), Xuan Lin `[通讯]` (Xiangtan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统综述了82种基于深度生成模型的分子生成方法，构建了模型分类、数据集与评价指标的统一框架，并整理了公开资源；

**💡 创新点**

创新点在于首次将模型家族、生成流程、评价基准和实验验证整合为一套整体视角，提供统一的数据集、指标与代码仓库，便于跨模型对比与实践应用；

**🔧 技术方法**

主要使用的技术包括RNN/Transformer、VAE、GAN、流模型与扩散模型，以及对应的训练目标与口袋条件化策略；

**📊 数据集**

使用的核心数据集包括QM9、ZINC、GEOM‑Drugs、MOSES、CrossDocked2020以及多种蛋白‑配体复合体数据库；

**📈 对比分析**

通过对上述数据集的标准化评估，作者对模型在有效率、独特性、创新性、药物属性和结合能等指标进行了系统比较，发现流模型和扩散模型在基本化学质量上表现最优，而口袋条件化模型在结合能与药物性方面取得显著提升；

**⚠️ 局限性**

局限性主要体现在评估协议不一致、蛋白质结构仅为静态口袋、缺乏对受体柔性与多靶点的考虑，以及实验验证仍停留在早期前临床阶段。

---

## 270. Multi-Functional Embedding Models for Funder Name Disambiguation in Scientific Publication Records

**arXiv ID:** 2609.09984 | [PDF](https://arxiv.org/pdf/2609.09984v1)

**作者:** Kanyao Han `[一作]` (University of Illinois at Urbana-Champaign), Jana Diesner `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建多源数据训练集（WoS、OFR、ROR），对句子嵌入模型（Sentence Transformer、Gemma、Qwen3）进行多任务微调，并结合 Louvain 聚类，完成了对生物多样性领域科研论文中资助者名称的去重与标准化，生成了大规模去重数据集；

**💡 创新点**

创新点在于：①利用跨数据源（WoS+OFR）自动生成正负样本，避免手工标注；②采用多任务学习（Contrastive Loss + Multiple Negatives Ranking Loss）让模型同时完成映射、匹配与聚类三重功能；③提出统一的框架可适配多种嵌入模型；④通过聚类挖掘Ror未覆盖的大型非英语资助者；

**🔧 技术方法**

技术手段包括：句子嵌入模型微调、对比损失、多个负样本排序损失、相似度阈值分类、Louvain 社区检测、以及在微调后使用 Cosine 相似度进行实体映射；

**📊 数据集**

使用数据集为：Web of Science 122,508 生物多样性论文的资助者记录、Crossref Open Funder Registry、Research Organization Registry（ROR）及其衍生的正负样本对；

**📈 对比分析**

实验将微调模型与预训练嵌入模型、生成式 LLM（GPT‑5.2、Gemini‑2.5‑Flash、Claude‑Sonnet‑4.6）以及微调+LLM 混合方案进行对比；在 WoS 语料上，微调后 ST 与 Gemma 的匹配准确率达到 0.91（独特名称 0.89），比预训练提升 0.15‑0.23；Qwen3‑0.6B 0.85；混合方案最高 0.82；相似度阈值 0.85 的二分类实现 F1 0.92；

**⚠️ 局限性**

局限性包括：Ror 未覆盖大量非英语及“项目”型资助者导致匹配率下降；长尾稀有资助者缺乏训练样本，易产生误匹配；长名称导致相似度低，误判为未匹配；生成式 LLM 在标准化时易引入错误并在后续步骤放大；数据受 WoS 许可限制，无法公开完整去重结果；

---

## 271. DiSCo: A Distribution-First Steering and Cultural Prior Evaluation Framework for Measuring Cultural Preference Bias in LLMs

**arXiv ID:** 2609.10253 | [PDF](https://arxiv.org/pdf/2609.10253v1)

**作者:** Bhuvan Arora `[一作]` (BITS Pilani), Dhruv Kumar `[通讯]` (BITS Pilani)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于分布优先的 forced‑choice 评估框架 DiSCo，用四级文化上下文梯度（C0–C3）衡量 LLM 的文化偏好先验与可控性。

**💡 创新点**

创新点在于：① 在多选题场景下将偏差度量转为分布比较，避免单一答案假设；② 通过 C0–C3 梯度和选项/上下文旋转控制字母位置与首位偏差；③ 引入 Signal Lift、Prior Stickiness Index、Statistical Parity Difference 等新指标量化位置信号与偏好坚持；④ 构建公开的 DiSCo Dataset/Bench，填补文化多样性评测缺口。

**🔧 技术方法**

技术手段包括：四级 prompt 设计、选项顺序和上下文顺序的四重旋转、KL/Gini、JSD、SPD 等分布指标、JSON 结构化输出与解析。

**📊 数据集**

使用数据集：DiSCo Dataset（150,816 行）和其压缩版 DiSCo‑Bench（304 题、12 文化），均从 BLEnD 转化而来，提供均衡的多文化选项。

**📈 对比分析**

实验对比六种 instruction‑tuned LLM（DeepSeek V3.2、LLaMA 3.3 70B、Claude Haiku 4.5、GPT‑5.4 Nano、Qwen3 8B、Gemma 3n E4B）。结果显示：默认偏好高度集中于 UK/US；C1 的位置提示提升 0.43–0.57 的符合率；C2 加强后仍有 34–50% 的 Prior Stickiness；SPD 在 C2 时反而变大，说明 prompt‑based steering 并未缩小文化不平等；C3 的事实注入几乎无影响（JSD < 0.02）。

**⚠️ 局限性**

局限性：仅使用英文，文化标签仅为国家/地区；未覆盖多语言、多脚本；事实注入对先验影响有限；评测仅基于静态 prompt，无法捕捉动态上下文或检索增强的文化适配。

---

## 272. Fundamental Limits of Joint Target Detection and Parameter Estimation - Characterizing Mixed-State Sensing Limits via Posterior Entropy Volume

**arXiv ID:** 2609.09667 | [PDF](https://arxiv.org/pdf/2609.09667v1)

**作者:** Dazhuan Xu `[一作]` (Nanjing University of Aeronautics and Astronautics), Han Zhang `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了混合离散-连续目标检测与参数估计的后验熵体极限，提出统一的熵体与信息量框架。

**💡 创新点**

首次将熵体与后验熵量结合，给出后验熵体收缩定律，并证明后验保持级联的无信息损失特性。

**🔧 技术方法**

使用信息理论（最大熵原理、混合AEP、典型集合理论）、概率论与数值仿真。

**📊 数据集**

使用单目标存在-范围模型的模拟数据，采用蒙特卡罗仿真。

**📈 对比分析**

通过与传统检测/估计信息量分解和级联接口信息损失的比较，验证理论精确匹配，显示每比特信息可将后验熵体减半。

**⚠️ 局限性**

仅适用于有限目标数、已标记状态，未覆盖随机目标集合、数据关联及有限块长误差。

---

## 273. Learning-Aided Short Code Design for ISAC based on MIMO-OFDM

**arXiv ID:** 2609.09797 | [PDF](https://arxiv.org/pdf/2609.09797v1)

**作者:** Mingcheng Nie `[一作]` (University of Sydney), Yonghui Li `[通讯]` (University of Sydney)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Transformer的深度学习编码波形设计，用于MIMO‑OFDM下的集成感知与通信（ISAC），实现短块传输时通信可靠性与距离估计精度的权衡。

**💡 创新点**

创新点在于：①将通信信息位、CSI和角度信息统一映射到共享隐藏空间并通过Transformer实现跨子载波特征融合；②使用联合损失（交叉熵+延迟修正CRB）调节通信-感知权衡；③揭示学习的码字在通信导向、感知导向与平衡三种极端结构之间的连续演变。

**🔧 技术方法**

技术包括：Transformer编码器/解码器、线性投影、角度双曲正弦-余弦编码、匹配滤波输入、改进的Cramér‑Rao边界（MCRB）损失、Adam优化、余弦学习率退火。

**📊 数据集**

使用合成实验数据：C=6, M=12, N_t=N_r=9，通信路径数P=5，感知目标数Q=3，角度从-60°到60°的离散网格，延迟均匀分布在[0,5]。

**📈 对比分析**

与传统的SVD预编码卷积码-QPSK/8PSK基线进行比较。实验显示：当权重λ=0.4时，所学波形在BER与MCRB上同时接近基线，λ=0时BER优于CC‑8PSK，λ=1时MCRB优于感知导向基准。

**⚠️ 局限性**

局限性包括：仅在单静态目标场景下验证；角度已知假设限制在后探测阶段；缺乏对动态目标、非理想CSI误差以及更复杂网络拓扑的评估。

---

## 274. Deterministic Prompting for Speaker-Stable Low-Resource Greek TTS

**arXiv ID:** 2609.10022 | [PDF](https://arxiv.org/pdf/2609.10022v1)

**作者:** Georgios Syllas `[一作]` (Athena Research Center), Alexandros Potamianos `[通讯]` (National Technical University Of Athens)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了单一声道现代希腊语文本到语音（TTS）系统，先通过 WhisperX 对齐与过滤将原始有声书录音转化为训练友好的片段，再利用 Parler‑TTS 进行多语言预训练并微调，随后使用低秩适配（LoRA）在仅 3.5 小时的单声道数据上进行声源稳定化，最终实现接近人类质量的合成语音。

**💡 创新点**

创新点包括：① 设计可复用的数据清洗与对齐流水线，将混杂的社区录音与有声书快速转换为标准 TTS 片段；② 通过用固定离散化的属性标签构造确定性风格提示，消除 LLM 生成提示带来的语音漂移；③ 采用两阶段训练（全参数微调 + LoRA 细化）在低资源场景下既保留多语种声学先验，又稳固单声道身份。

**🔧 技术方法**

使用的核心技术有 WhisperX（对齐与语音分段）、Parler‑TTS（多语言prompt‑conditioned codec 生成器）、LoRA（低秩参数适配）、Deterministic Prompt Engineering、HiFi‑GAN/DAC 语音解码器，以及 ASR‑基准评估（WhisperX）。

**📊 数据集**

使用的数据集包括：CSS10（≈4 h 单声道女性有声书）、Mozilla Common Voice（≈15.5 h 174 说话人多语种混杂语音）以及从公开有声书构建的手工验证单声道语料（≈3.5 h 男性说话人）。

**📈 对比分析**

通过对比 MOS‑I、MOS‑N、MOS‑C、WER/CER 等客观/主观指标进行评估。最优配置（Deterministic Prompt + LoRA）实现 WER 10.7%（仅比人类录音低 2.9 pp），MOS‑I 4.00 近人类 4.36，MOS‑C 4.24 接近人类 4.30，显示在仅 3.5 h 数据下即可达到近人类语音质量。

**⚠️ 局限性**

局限性包括：仅验证单一男性阅读式声源，缺乏多说话人/说话风格的泛化；对 LLM 提示漂移的原因尚未彻底解析；在极低资源条件下仍需大量多语种预训练；未对抗深度伪造等安全风险进行实验。

---

## 275. Optimal Non-Adaptive Vantage Point Selection

**arXiv ID:** 2609.10267 | [PDF](https://arxiv.org/pdf/2609.10267v1)

**作者:** Jie Gao `[一作]` (Rutgers University), Chang Wu `[通讯]` (Tsinghua University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究视点选择问题，提出了单查询与多查询下非自适应算法的最优竞争比；

**💡 创新点**

创新点在于给出单查询问题的Θ̃(n^{2/3})最优竞争比，并推导出多查询情形下复杂但紧确的三段式竞争比；

**🔧 技术方法**

主要技术包括基于边占优度的贪心选择、残留度和占有度的迭代递减、对最优比的精细概率与组合分析，以及构造下界的蝴蝶图与填充结构；

**📊 数据集**

本文为理论性研究，未使用实际数据集；

**📈 对比分析**

相较于以往仅知Ω̃(√n)下界与O(n)上界，本文实现了Ω̃(n^{2/3})与更一般参数下的Ω̃(min{n/αk,√(n/α),n^{2/3}/(αk)^{1/3})}竞争比，说明了非自适应算法在该问题上的可行性；

**⚠️ 局限性**

局限性在于下界仅针对非自适应算法，尚未扩展到自适应情况，且多查询结果依赖复杂参数阈值。

---

## 276. EEGBind: Detecting Source-Level Interictal Epileptiform Discharges via EEG-Centric Multimodal Binding

**arXiv ID:** 2609.09728 | [PDF](https://arxiv.org/pdf/2609.09728v1)

**作者:** Muchen Li `[一作]` (Hong Kong University of Science and Technology), Jintai Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了EEGBind框架，用 EEG 作为主模态，将同步视频作为辅助信息进行源级间歇性癫痫发放（IED）分类。

**💡 创新点**

创新点在于 EEG‑centric 多模态绑定策略以及无交叉熵（CE‑free）的视图一致性修复机制，既保持 EEG 的源区判别边界，又提升对视频路由扰动的鲁棒性。

**🔧 技术方法**

技术包括 ST‑EEGFormer 作为 EEG 编码器、七路视频特征投影、注意力探针（attentive probe）、视图抛弃/遮挡一致性约束、队列对比学习，以及五折主观离散集成。

**📊 数据集**

使用 NeuroMM 2026 Grand Challenge Track 3 的 NMM‑Source‑IED 数据集，该数据集提供同步 EEG 与视频，且每个候选窗口标注为五个脑区源级标签。

**📈 对比分析**

通过与 EEG‑only 基线、不同修复变体以及 dummy‑video/zero‑video 控制进行对比，最终五折集成在官方隐藏测试集上达 weighted‑F1 0.8395，显著优于基线（0.6444）与控制（0.7448/0.7460）。

**⚠️ 局限性**

局限在于仅在单一公开数据集上验证，缺乏跨医院/设备的泛化评估；视频辅助虽提升性能，但对单一路径的依赖极低，未深入探索更丰富的多模态融合方法。

---

## 277. Minimum-makespan completion and vertex selection leave the Wang-Sitters constant at 11/6

**arXiv ID:** 2609.10004 | [PDF](https://arxiv.org/pdf/2609.10004v1)

**作者:** Adam Y. Shavit `[一作]` `[通讯]` (CUNY), Adam Y. Shavit (CUNY)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文研究 Wang–Sitters 的槽位逼近方案（graph balancing），证明即使在 Step 3（匹配）使用最优匹配、Step 1（解空间）限制为顶点解，或两者都限制，方案的最坏情况常数仍为 11/6（即 1.166…），并给出相应的反例和构造。该工作进一步证明，所谓的“最佳匹配”并不能把常数从 11/6 降低到 7/4，且在 Step 1 只返回顶点解的情况下，最坏常数仍保持在 11/6；同时证明在给定的 LP 方案下，选择匹配是 NP‑hard 的。

**💡 创新点**

创新点：① 系统化分离了原先统一的 7/4 猜想，分别证明阈值相对和相对最优的两种形式均不成立；② 构造了新的最坏实例族，展示即使在最优匹配或顶点解下，最坏常数依旧是 11/6；③ 证明了匹配选择问题在已给 LP 解下是 NP‑难，揭示了在 Wang–Sitters 框架内无法通过更优匹配来提升性能。

**🔧 技术方法**

技术：基于 Wang–Sitters 的槽位图与 Shmoys–Tardos 匹配理论，使用鸽巢原理和构造性的计数论证；利用 LP 的可行域与顶点性质证明顶点实例；利用精确的组合构造与算术证明最坏常数；通过归约证明匹配选择的 NP‑难性。

**📊 数据集**

数据集：论文主要是理论构造，没有使用公开数据集；作者提供了多组符号构造实例（如 19 机、39 任务的实例、g=1…4 的实例族等），并在实验部分使用符号运算和整数规划求解验证构造的正确性。

**📈 对比分析**

比较方法：将最坏常数与已有的 11/6 上界和 7/4 的下界进行对比；通过构造反例证明两种 7/4 形式均不成立；利用实验检验 NP‑难归约；结果显示，即使采用最优匹配或顶点解，Wang–Sitters 方案的最坏比率仍为 11/6，未能进一步改进。

**⚠️ 局限性**

局限性：仅讨论了 Wang–Sitters 的槽位逼近框架，未考虑其它 LP 逼近或更复杂的匹配策略；证明依赖于特定的阈值与“非严格”大作业判定，若采用严格判定可能不同；虽然证明匹配选择 NP‑难，但未给出可行的近似或多项式时间选择算法。

---

## 278. Subagents vs Agent Skills: Executing Reusable Knowledge for Long-Horizon Agentic Tasks

**arXiv ID:** 2609.09233 | [PDF](https://arxiv.org/pdf/2609.09233v1)

**作者:** Wasu Top Piriyakulkij `[一作]` (Cornell University), Niranjani Prasad `[通讯]` (Microsoft Research Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究长时序任务中语言模型代理如何使用可重用知识库，通过比较将技能包作为主技能（在主上下文中加载）与作为子代理（在单独上下文中执行）两种方式的效果。

**💡 创新点**

提出子代理执行在具备明确输入输出契约的程序化技能包上更有效，同时显著降低峰值上下文长度，并证明其在长时序任务中更稳健。

**🔧 技术方法**

使用大语言模型工具调用框架 OpenHands，结合 GPT‑5.3 Codex 等 LLM；将技能包视为工具，设计子代理调用机制。

**📊 数据集**

利用 SkillsBench 基准，其中包含 87 个长时序任务；对 64 个任务合成了具有输入输出契约的程序化技能包。

**📈 对比分析**

与传统的 agent‑skill 执行对比，使用准确率、峰值上下文长度、总 token 消耗等指标。实验表明：在合成的程序化技能包下，子代理的准确率高于 agent‑skill，且随着干扰工具数量增加子代理更稳健；但总 token 消耗更高。

**⚠️ 局限性**

子代理仅在技能包具备清晰契约时才有效；需要额外通信成本；目前缺乏高效的通信设计与技能库组织方法。

---

## 279. InstantMimic: A High Performance System for Learning Physics-based Skills in Seconds

**arXiv ID:** 2609.09821 | [PDF](https://arxiv.org/pdf/2609.09821v1)

**作者:** Ikjun Choi `[一作]` (Seoul National University), Jungdam Won `[通讯]` (Seoul National University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了InstantMimic——一种端到端GPU原生的训练管线，能在几秒内完成物理基础的角色运动模仿训练，并支持LLM驱动的自动超参数搜索。

**💡 创新点**

通过把物理仿真、奖励评估、策略推理和PPO更新全部迁移至GPU，利用CUDA Graphs、核融合和NVIDIA Warp JIT等技术，显著降低了GPU内核碎片化和CPU内存访问导致的瓶颈；实现秒级训练并实现LLM自动化超参数优化。

**🔧 技术方法**

技术包括MuJoCo Warp GPU物理后端、CUDA Graphs、核融合、NVIDIA Warp JIT、零拷贝内存访问、PPO、DeepMimic式奖励函数以及GPT-5.5‑high LLM代理进行超参数搜索。

**📊 数据集**

使用PHC（Physics-based Character）运动捕捉数据集训练五种基础动作，使用AMASS 37.4小时的运动数据进行VAE基潜在控制器的预训练。

**📈 对比分析**

与Isaac Lab（PhysX）和mjlab（MuJoCo Warp）对比，InstantMimic在后弹跳训练中达0.613 Mfps，比前两者分别快5.85×和3.91×；后弹跳从18.77s降至2.17s（8.6×），五种参考动作均在1.5–4.5s内收敛；AMASS预训练从约30min完成。

**⚠️ 局限性**

局限性包括：超参数搜索依赖“训练时间至成功”指标，可能牺牲追踪质量；搜索空间与LLM能力相关；性能提升仍受限于物理求解器和GPU资源，且在其他任务或更大规模系统中的可迁移性未充分验证。

---

## 280. TFR-GNN: Topology- and Fault-Aware Graph Neural Scheduling for Heterogeneous Distributed Computing Systems

**arXiv ID:** 2609.09165 | [PDF](https://arxiv.org/pdf/2609.09165v1)

**作者:** Shiyu Yang `[一作]` (University of California, Los Angeles), Jie-Si Yang `[通讯]` (University of Utah)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为 TFR-GNN 的图神经网络调度器，能够在异构分布式系统中根据工作流 DAG、机器拓扑、速度与可靠性信息，做出兼顾速度与容错的任务分配与复制决策。

**💡 创新点**

创新点：
1) 引入失效门控（failure‑activity gate）实现“无失效时退化为 HEFT”的适应性保证；
2) 结合双向依赖注意力、带宽偏置的拓扑注意力和交叉注意力，形成对任务与机器的高质量表示；
3) 通过“可靠性倾斜”和可选复制门控实现容错决策；
4) 采用知识蒸馏将多策略最佳组合（HEFT、R‑HEFT、FT‑HEFT、复制等）转化为单一一射策略，避免直接 RL 的收敛困难。

**🔧 技术方法**

技术细节：
- GNN 编码任务与机器节点；
- 双向 DAG 注意力（GAT）捕获任务间依赖；
- 机器图注意力加入对带宽的对数偏置；
- 交叉注意力生成任务‑机器亲和度；
- 失效门控可靠性倾斜与复制门控；
- 引导式列表调度（guided list‑scheduling）解码；
- 蒸馏训练：教师-强制（teacher‑force）对每步置放/复制决策做监督，Loss = 交叉熵 + 复制 BCE。

**📊 数据集**

数据集：使用 WfCommons/Pegasus 公开工作流，135 条实例，涵盖 7 种科学与数据分析应用，任务规模从 22 到 4,846。机器集成随机异构速度、类型亲和性、双模可靠性（volatile vs reliable），并按 ρ 设定 volatile 机器比例。

**📈 对比分析**

对比方法：HEFT、Min‑Min、Max‑Min、随机、R‑HEFT（固定可靠性倾斜）、FT‑HEFT、复制基准以及“Oracle”即每个场景最佳策略。实验结果：
- 在 s=0（无失效）时 TFR‑GNN 与 HEFT 完全等价；
- 在失效场景下，平均降低 14.8% 预期 makespan（最高可达 47%），比 R‑HEFT/FT‑HEFT 低约 11%；
- 与 Oracle 接近（0.852× vs 0.854×），单一策略即可跟踪最佳组合；
- 生成时间 < 1 s 甚至在 5,000 任务时；
- 在未见过的应用与规模大 10 倍时仍保持性能。

**⚠️ 局限性**

局限性：
1) 仅在事件级模拟器中评估，未考虑生产环境中的争用、straggler、数据放置等影响；
2) 失效模型为 Poisson 崩溃 + log‑normal 修复，复制在此模型下效果有限；若为永久失效或 checkpoint，复制策略可能更有效；
3) 蒸馏教师来自有限策略集合，无法突破其性能上限；
4) 仅优化期望 makespan，未考虑尾部延迟或截止时间等风险敏感目标；
5) 训练与测试均在单 CPU 下完成，未探索更大规模或在线自适应训练。

---

## 281. Revisiting the Weight Spectrum of the Affine Grassmann Code $C^{\mathbb{A}}(2,m)$

**arXiv ID:** 2609.10274 | [PDF](https://arxiv.org/pdf/2609.10274v1)

**作者:** Rohit Yadav `[一作]` `[通讯]` (Indian Institute of Technology Jammu), Rohit Yadav (Indian Institute of Technology Jammu)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文独立且简化地推导了仿射Grassmann码C^Å(2,m)的权重谱，适用于所有m≥4和每个素数幂q。

**💡 创新点**

创新点在于用一个封闭公式表达码字的汉明权重，而不是通过对相关交替矩阵秩的案例分析。

**🔧 技术方法**

使用了线性代数和多项式评估技术，特别是通过评估通用矩阵的次要线性组合来构造仿射Grassmann码。

**📊 数据集**

没有具体提到使用的数据集，但涉及的数学结构和公式适用于任意素数幂q。

**📈 对比分析**

与Piñero和Singh的方法相比，本文的方法更为统一和简洁，能够在不需要特定特征的情况下处理所有素数幂q，性能上提供了更简洁的推导。

**⚠️ 局限性**

限制在于目前的推导仅适用于仿射Grassmann码C^Å(2,m)，而对于更高维度的情况（如ℓ≥3）仍然是开放的研究问题。

---

## 282. Low-Rank Prompt Learning for Vision-Language Models with Fixed-Token Bases

**arXiv ID:** 2609.09462 | [PDF](https://arxiv.org/pdf/2609.09462v1)

**作者:** Tanvir Muntakim Tonoy `[一作]` (UC Santa Barbara), Ramtin Pedarsani `[通讯]` (UC Santa Barbara)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了 CLIP 连续提示（CoOp）的低秩分解，探讨是否需要学习 token 侧因子和 embedding 侧因子，并证明可将 token 侧因子固定为任意基（高斯、正交、SVD 或随机）只训练 embedding 侧因子即可完成适配。

**💡 创新点**

提出提示因子非对称性和更新空间维度差异的理论，证明 token 侧因子对性能影响有限，可固定；同时展示低秩提示在多任务下与密集提示相当甚至更优，并显著减少可训练参数。

**🔧 技术方法**

使用低秩提示分解 P=BA、Prompt-Factor Asymmetry 定理、Local Update-Space Dimension Gap、固定因子下的梯度下降收敛分析，并在 CLIP ViT-B/16 与 RN50 两个 backbone 上进行训练与评估。

**📊 数据集**

Caltech101、DTD、FGVC Aircraft、Food101、Oxford Flowers、Oxford Pets、UCF101 七个公开图像分类基准。

**📈 对比分析**

与密集 CoOp（16 token）和 4 token 对比；在 1-shot 时低秩提示提升约 1–2% 准确率，在 1/4/16 shot 的 H‑mean 上均有提升，且参数量仅为原来的约 ¼；固定 token 侧基在各场景下几乎无性能损失。

**⚠️ 局限性**

仅针对 CoOp 样式的文本提示；未验证深层、多模态提示或其他视觉‑语言模型的可推广性。

---

## 283. CougarTail & CUB: A General-Purpose Mast and Central Utility Board for Cylindrical Underwater Enclosures

**arXiv ID:** 2609.10230 | [PDF](https://arxiv.org/pdf/2609.10230v1)

**作者:** Ben Washburn `[一作]` (Brigham Young University), Joshua Mangelson `[通讯]` (Brigham Young University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出并实现了CUB（圆形中央功能板）与CougarTail（集成传感器桅杆），用于4英寸直径的水下圆柱封装中，将Raspberry Pi CM5与STM32微控制器、功率管理与多路接口集成到一块圆形PCB上，并配备GPS与双频PCB天线的水下可伸缩桅杆，显著降低电子占用空间与重量，提升表面通信与定位性能。

**💡 创新点**

创新点在于：①采用圆形PCB与4英寸圆柱封装匹配，最大化横截面积利用；②将计算、功率管理、外围接口与低功耗控制统一到单块板上；③设计可水密、双频天线兼容的可伸缩桅杆，解决水下GPS与无线信号衰减问题。

**🔧 技术方法**

使用的技术包括：Raspberry Pi Compute Module 5的完整Linux环境、STM32微控制器的低层任务卸载、XT90电池接口与50 A熔断器、USB‑C供电、USB‑A/USB‑C/以太网/I²C/SPI/UART/ CAN接口、Digi XBee无线模块、Molex双频PCB天线、陶瓷GPS天线、PETG 3D打印桅杆壳体、聚酯管穿线、环氧漆封装、蓝色机器人4英寸圆柱外壳、以及针对GPS/无线通信的实时数据采集与测试。

**📊 数据集**

使用的数据主要来自本实验室的现场与基准测试：①基准测试记录GPS水平精度（均值1.07 m、半数值0.57 m）与修订率100%；②无线链路吞吐率随距离（5 m–100 m）变化的实验数据；③基于推车的航点跟踪实验记录的路径轨迹与控制指令；④电子占用空间与重量测量（从200 mm/709 g降至25 mm/156 g）。

**📈 对比分析**

比较方法：将原先200 mm长、709 g的矩形PCB堆叠系统与新系统进行对比，计算占用空间与重量的减少率（87.5 %与78 %）。性能方面：GPS水平精度1.07 m（1σ）且100 %修订率；无线链路在5–100 m距离内保持高吞吐（具体数值见实验图）；推车航点测试验证了完整的感知-控制-通信链路，路径误差符合预期。

**⚠️ 局限性**

局限性：①桅杆的实际压力耐受深度尚未在实验室压力试验中验证，理论深度>100 m需进一步确认；②仅在4英寸圆柱封装中验证，扩展至其他尺寸封装需重新设计PCB；③在水下航点任务的闭环控制性能仍未完成实际水下测试；④对极端环境（高温、腐蚀性水体）的长期耐久性未进行长期评估。

---

## 284. Proximity Gaps for Gabidulin Codes and Applications

**arXiv ID:** 2609.09838 | [PDF](https://arxiv.org/pdf/2609.09838v1)

**作者:** Songsong Li `[一作]` (Shanghai Jiao Tong University), Ruiqi Zhu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了Gabidulin码在秩度量下的邻近间隙，并基于此构造了交互式邻近证明（IOPP）和多项式承诺方案（PCS）

**💡 创新点**

首次给出秩度量码的邻近间隙界，并证明其与 Reed–Solomon 码的唯一解码间隙相匹配，同时提出相应的 IOPP 与 PCS 框架

**🔧 技术方法**

使用符号乘法、Berlekamp‑Welch 里程碑、线性插值、张量分解以及改进的分离引理等代数技术

**📊 数据集**

无实验数据集，全部为理论证明与构造

**📈 对比分析**

与 Hamming 度量下的 RS 码结果进行对比，证明 δ≤(d−1)/2n 的上界可达且误差概率约为 10q^n−1/q^m；IOPP 与 PCS 的证明尺寸和验证时间均为 O(√k) 级，优于传统方法

**⚠️ 局限性**

仅在唯一解码半径以内有效，δ=d/3 处仍存在误差下界，且目前未给出更优的误差上界

---

## 285. BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models

**arXiv ID:** 2609.09554 | [PDF](https://arxiv.org/pdf/2609.09554v1)

**作者:** Shivam Singh `[一作]` (University of California San Diego), Alex Warstadt `[通讯]` (University of California San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对Whisper模型进行语言专属微调，构建了102种语言的ASR模型；

**💡 创新点**

创新点在于将简单的单语微调扩展到大规模多语言、引入词表替换与文本自监督微调两大策略，并通过warm-start初始化提高tokenizer效果；

**🔧 技术方法**

采用Whisper的Transformer架构，结合tokenizer替换、文本解码器微调和多任务学习；

**📊 数据集**

使用FLEURS语音数据、Common Voice数据以及来自Wikimedia等的大规模文本语料；

**📈 对比分析**

与Whisper、Omni-1B/7B、MMS、Qwen3-ASR等基线对比，模型在77种语言上优于Whisper、平均CER下降2.8倍，在27种语言上实现开放源代码SOTA；

**⚠️ 局限性**

局限包括仅使用100小时的语音数据、仅对Whisper进行微调、评估仅限训练集相同领域且未覆盖所有语言，且多语言模型在某些语言上仍表现不佳。

---

## 286. Who Argues What? Joint Argument-Entity Detection and Classification in Political Debates

**arXiv ID:** 2609.10192 | [PDF](https://arxiv.org/pdf/2609.10192v1)

**作者:** Lucio La Cava `[一作]` (University of Calabria), Sergio Greco `[通讯]` (University of Calabria)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对政治辩论文本，构建了包含实体标注的USElecDeb扩展数据集，并提出一种单次生成式模型JAET，用于同时标注论证成分与辩论实体。

**💡 创新点**

创新点在于：①将实体层与论证层统一到同一标注视图，实现了实体与论证的并行标注；②通过细粒度的生成式标签插入方法，使模型能够在保持原始文本不变的前提下直接输出完整的带标签序列；③证明联合标注显著优于传统的分阶段或独立模块的组合。

**🔧 技术方法**

技术上采用decoder-only LLM（如Llama 3.1 8B、Mistral 7B、Qwen 2.5 7B）进行提示式微调，优化目标为最小化生成标签的负对数似然；同时引入文本保留率与标签格式化率两种鲁棒性评估。

**📊 数据集**

使用的数据集为新发布的entity-enriched USElecDeb，包含44场美国总统/副总统辩论的转录文本，约8353个回合，标注了论证成分（Claim、Premise）和8类辩论实体（Person、Organization、Party等）。

**📈 对比分析**

与基准（prompt-only、AM-only、RooseBERT、逐步Pipeline）相比，JAET在未标注类型下F1提升约+27.3%，在标注类型下提升约+41.9%；单一任务的AM性能与AM-only相当，同时恢复了实体层；在Persuasive Essays等非政治领域亦表现出显著提升，表明方法具有良好迁移性。

**⚠️ 局限性**

局限性包括：①仅覆盖英文美国政治辩论，难以推广到其他语言或文化场景；②实体词表基于美国政治语境，可能缺乏跨国通用性；③仅以回合级别为基础，无法捕捉跨回合的指代与论证关系；④实体标注依赖LLM+人工验证，可能存在系统性偏差。

---

## 287. In Medical Claims Data, Enhancing Predictive Performance for Major Adverse Cardiovascular Events Using Cross Attention

**arXiv ID:** 2609.09824 | [PDF](https://arxiv.org/pdf/2609.09824v1)

**作者:** Yuhei Fujioka `[一作]` (Kyoto University), Shingo Fukuma `[通讯]` (Kyoto University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用健康检查数据与一年的医疗索赔数据，构建了跨注意力变压器模型，对主要不良心血管事件（MACE）进行预测。

**💡 创新点**

创新点在于使用跨注意力机制有效捕获诊断与治疗之间的多对多关系，提升了对临床信息的利用效率。

**🔧 技术方法**

技术包括跨注意力（Cross‑Attention）与自注意力（Self‑Attention）Transformer、前馈网络（FFN）、LightGBM及ASCVD基准模型。

**📊 数据集**

数据集来源于日本健康保险协会：166,030名健康检查受试者、714,710名索赔记录，共51,367名样本（含MACE事件）。

**📈 对比分析**

与ASCVD、LightGBM和自注意力模型比较，跨注意力模型在ROC‑AUC上达到0.7720、MCC为0.1525，统计显著优于其他模型。

**⚠️ 局限性**

局限包括未使用时间序列信息、重复医疗代码以及对单月高代码量样本的适用性有限。

---

## 288. Smart Adaptive Computing Across the Continuum: LLMs in IoT-Edge-Cloud Resource Management

**arXiv ID:** 2609.09348 | [PDF](https://arxiv.org/pdf/2609.09348v1)

**作者:** Antonino Vaccarella `[一作]` (Institute of Information Science and Technologies 'Alessandro Faedo', National Research Council of Italy), Massimo Coppola `[通讯]` (Institute of Information Science and Technologies 'Alessandro Faedo', National Research Council of Italy)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过扩展Wang等人的DRL分类体系，加入LLM整合层次（AI Augmentation Paradigm）和反馈通道（Feedback Channel）两个维度，对六种近年云连续体资源管理系统进行归类和分析，并揭示了缺乏跨层闭环LLM主导架构的空白。

**💡 创新点**

创新点在于提出了包含LLM角色与反馈路径的全新架构分类法，系统性识别出云连续体中LLM与DRL/多智能体协作时的关键缺口——缺少跨层反馈抽象与闭环控制；同时提出了Mediator（中介）组件的概念作为解决方案。

**🔧 技术方法**

技术手段主要是文献综述与架构分析；通过对六个案例的LLM层次（LLM0-LLM2）与反馈类型（F0-F3）的评估，构建了一个四维（控制范围、训练范式、LLM层次、反馈通道）扩展分类表。

**📊 数据集**

本文未使用具体数据集，而是以系统架构与功能描述为分析对象，对现有研究做归纳和比较。

**📈 对比分析**

比较方法采用基于扩展分类表的对比评估；由于未进行实验，论文未给出数值性能对比，主要以架构模式和功能实现的差异作为评估依据。

**⚠️ 局限性**

局限性包括：仅关注六篇近期系统，缺乏更广泛样本；缺少跨层闭环的实验验证；未考虑LLM推理成本、时延和隐私安全问题；对云连续体多层异构反馈的抽象仍未实现，且未探讨联邦学习等更先进的协同机制。

---

## 289. Two-Token Features and Small-Large Ensembles for VLM Hallucination Detection

**arXiv ID:** 2609.10244 | [PDF](https://arxiv.org/pdf/2609.10244v1)

**作者:** Eli Schwartz `[一作]` `[通讯]` (IBM Research), Eli Schwartz (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个字符级幻觉检测系统，利用4B VLM的两词特征头进行微调，并在推理时与400B零射大模型进行集成，同时结合OCR提取和合成数据提升性能。

**💡 创新点**

创新点在于采用中层隐藏状态的两词特征进行分类、对层次进行系统性搜索、利用大模型生成的合成幻觉数据实现集成多样性，并将OCR信息直接注入提示。

**🔧 技术方法**

技术手段包括Qwen3.5-VL-4B与397B的LoRA微调、两层MLP分类器、凸组合集成、PaddleOCR文本提取，以及大模型的JSON-schema零射推理。

**📊 数据集**

使用的数据集为SHROOM‑Visions 2026任务提供的SHEEP多语言数据（约15k标注样本），再加上约67k条由大模型生成的合成幻觉样本，以及OCR提取结果。

**📈 对比分析**

评估采用Spearman相关系数（平均Cor和标签感知Cor‑lbl），在隐藏测试集上实现平均Cor 0.487、Cor‑lbl 0.387，排名分别为第6/28、第6/21、第8/21和第7/22。

**⚠️ 局限性**

局限性包括单一随机种子训练导致的语言间波动、OCR提取器的准确性受限、合成数据类别与真实数据不匹配、集成在验证集上效果不显著，以及仅使用同一模型族（Qwen）和单一OCR引擎。

---

## 290. Evaluating Model Retraining under Drift: Paired Comparisons of Cumulative Subgroup Disparity

**arXiv ID:** 2609.09788 | [PDF](https://arxiv.org/pdf/2609.09788v1)

**作者:** Aaron Ceross `[一作]` `[通讯]`, Aaron Ceross

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文比较了定期更新、损失触发更新以及子组误差差距触发更新三种策略对部署中的子组误差差距的影响，并在两种合成漂移情景与美国社区调查（ACS）数据上进行评估。

**💡 创新点**

创新点在于提出子组误差差距触发策略，并使用配对差异、随机参考与后见者最优时间表等多维度方法，系统评估不同更新规则对子组公平性与整体性能的综合影响。

**🔧 技术方法**

技术手段包括 L2 正则化逻辑回归、CUSUM 监测、子组误差差距监测、配对差异统计、学生化自举检验、群体积分校准、数据重加权与模拟/重放实验。

**📊 数据集**

使用的数据集为两种合成漂移（子组特定漂移与组合漂移）以及美国社区调查（ACS）1 年人群文件（性别与种族子组）。

**📈 对比分析**

比较方法基于配对差异、平均累积差距（0.04–0.88个百分点）和相对减少率，结果显示三种策略均比冻结模型降低了累计子组差距，但差距缩小幅度有限，且不同策略在误差差距与准确率之间存在权衡。

**⚠️ 局限性**

局限性包括仅考虑两种漂移、单一学习器、二元子组、十窗口时间窗口、非确认性实验结果以及未评估训练成本、不同漂移幅度、更多子组与模型失配等情形。

---

## 291. Total Simulated Survey Error: Designing and Diagnosing Survey Responses from Large Language Models

**arXiv ID:** 2609.10280 | [PDF](https://arxiv.org/pdf/2609.10280v1)

**作者:** Indira Sen `[一作]` (University of Mannheim), Markus Strohmaier `[通讯]` (University of Mannheim)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了 Total Simulated Survey Error (TS2E) 框架，用于系统化分析大语言模型生成的调查问卷中的测量与代表性错误。

**💡 创新点**

创新点在于将传统的总调查误差（TSE）框架扩展至 LLM 模拟，区分设计导致与模型固有的测量误差与代表误差，并识别评估过程中的逻辑谬误。

**🔧 技术方法**

技术方法包括任务阐述、人格构建、LLM 选择与提示工程、响应生成与处理以及后调权重等设计步骤，并通过多世代分析检验其对模拟质量的影响。

**📊 数据集**

使用的数据集为美国全国选举研究（ANES）2024 年的投票意向与相关属性，构建了 4,779 个模拟个体。

**📈 对比分析**

评估采用加权 F1 和加权总变差（TVD）两指标，对 288 个设计配置进行 OLS 回归，结果显示最佳配置 F1≈0.52、TVD≈0.23，且不同设计因素对子群表现有显著影响。

**⚠️ 局限性**

局限性包括无法获得真正无误差的 LLM 基准、评估依赖同一调查的参考数据可能引入“真值落差”和“数据污染”，以及模型选择与人格构建的交互效应仍未完全解构。

---

## 292. Beyond Repository Boundaries: Cross-Repository Graph Retrieval for Code Generation

**arXiv ID:** 2609.09987 | [PDF](https://arxiv.org/pdf/2609.09987v1)

**作者:** Minh Le-Anh `[一作]` (Quantum AI & Cyber Security Institute, FPT Corporation), Nghi D. Q. Bui `[通讯]` (VinUniversity)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CrossCoder，利用统一的跨仓库知识图和计划-检索策略实现跨仓库代码生成。

**💡 创新点**

创新点在于：①将仓库与外部依赖库的实体统一建图；②通过计划-检索框架先规划所需实体，再递归图扩展获取多跳上下文；③引入 VersionExec 基准评估代码在不同依赖版本下的可执行性。

**🔧 技术方法**

使用了知识图索引、语义检索、图扩展、RAG（检索增强生成）以及基于计划的检索方法。

**📊 数据集**

采用了 RepoExec、DevEval 两个仓库级基准和基于 BigCodeBench 的 VersionExec（包含不同版本的 13 个库）来评测。

**📈 对比分析**

在 RepoExec 与 DevEval 上，CrossCoder 的 Pass@1/3 和 DIR 均超过所有对比方法，20B 模型已能接近 120B 模型；在 VersionExec 上，对比无上下文、仅提示库信息和文档检索，CrossCoder 在两个版本环境均显著提升 Pass@1/3，特别是旧版本环境的鲁棒性更好。

**⚠️ 局限性**

主要局限包括：仅在 Python 生态中验证，构建跨仓库知识图和多跳扩展的计算与内存开销较大，对极大仓库的预处理会产生延迟；未来需支持多语言、压缩图规模并引入更高层的仓库意图信息。

---

## 293. Streaming Algorithms for Gaussian Kernel Density Statistics

**arXiv ID:** 2609.09622 | [PDF](https://arxiv.org/pdf/2609.09622v1)

**作者:** Qin Zhang `[一作]` `[通讯]` (Indiana University), Qin Zhang (Indiana University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于高斯核相似性的单通道流算法，用于计算多样性指数和密度矩。该算法通过利用高斯核的几何和分析特性，提供了在固定维度下的子线性空间近似算法。

**💡 创新点**

创新点在于通过高斯核的几何结构克服了传统相似性函数在流数据统计分析中的空间复杂性限制，展示了几何结构如何根本性地改变相似性统计分析的流复杂性。

**🔧 技术方法**

使用了高斯核作为相似性函数，并结合了流数据模型中的单通道算法，利用了几何和分析特性来设计近似算法。

**📊 数据集**

使用了固定维度的欧几里得向量流数据集，具体数据集未在摘要中详细说明。

**📈 对比分析**

与一般相似性函数相比，使用高斯核的算法在单通道空间复杂性上表现出显著优势。对于多样性指数，算法在空间使用上为O_d(ε^{-2d-4} log^O(d)(n/δ))，而一般相似性函数的O(1)近似需要Ω(n)位空间。

**⚠️ 局限性**

限制在于对于高维数据，算法的空间复杂性仍然依赖于维度，且在高阶密度矩的估计中，仍存在小的多项式间隙，未能完全闭合上界和下界之间的差距。

---

## 294. Encrypt What Matters: When Selective Homomorphic Inference Is Efficient

**arXiv ID:** 2609.09357 | [PDF](https://arxiv.org/pdf/2609.09357v1)

**作者:** Ali Backour `[一作]` (Massachusetts Institute of Technology), Ana Onoprishvili `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了选择性同态推理，只对输入中敏感的 ROI 区域进行加密，其余部分保持明文，保证与全加密推理得到完全相同的输出。

**💡 创新点**

创新点在于：①提出“加密依赖传播”机制，跟踪哪些激活受加密影响；②证明卷积层中受加密影响区域的扩展规律（m_i=m+i(k-1)）；③基于依赖传播构建速度提升预测模型；④指出网络的局部性是决定选择性加密效益的关键属性。

**🔧 技术方法**

使用技术包括：同态加密（FHE）与 Concrete TFHE、4-bit 权重/激活量化网络、卷积网络的依赖传播算法、速度预测公式 S_pred=∑N_a/∑T_a。

**📊 数据集**

实验数据集未在论文中详细列出，实验使用 224×224 的通用图像输入（如 ImageNet 图像），并对通道缩减版本的 AlexNet、ResNet‑18、VGG‑11 进行评测。

**📈 对比分析**

对比方法：在 Concrete TFHE 上测量完整加密与选择性加密的同态推理耗时。结果显示，32×32 ROI 时 VGG‑11 达到 18× 加速，ResNet‑18 5.1×，AlexNet 4.3×。预测模型对 ConvNeXt、EfficientNet‑B0、ViT‑B/16、Swin‑T、MLP‑Mixer、全连接网络等也给出了不同加速，表明局部性强的网络获得更高收益。

**⚠️ 局限性**

限制：①预测器是结构成本模型，未能精确模拟实际运行时；②实验仅在通道缩减的网络上进行，绝对运行时间不代表原始网络；③未提供完整的数据集与训练细节。

---

## 295. LinearMask-GS: Stable-Mask Importance Pruning for Compact 3D Gaussian Splatting

**arXiv ID:** 2609.10095 | [PDF](https://arxiv.org/pdf/2609.10095v1)

**作者:** Donghun Ryu `[一作]` (Chung-Ang University), Minhyeok Lee `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种LinearMask-GS框架，在3D Gaussian Splatting中通过线性增量掩码实现稳定重要性剪枝，从而显著减少Gaussian原语数量并保持渲染质量。

**💡 创新点**

核心创新是将Gumbel‑Sigmoid替换为线性增量激活，保持掩码值处于中间区间，避免早期饱和导致排名失效，进而实现更可靠的剪枝决策，并能无修改迁移至多种Gaussian基底模型。

**🔧 技术方法**

使用最大池化重要性评分、线性增量掩码激活、top‑ρN硬剪枝、稀疏正则化以及与3DGS/2DGS等显式Gaussian表示的联合端到端训练。

**📊 数据集**

在Mip‑NeRF 360、Tanks & Temples和Deep Blending这三大新视图合成基准上进行实验评估。

**📈 对比分析**

与LP‑3DGS、Compact3DGS、LightGaussian等剪枝/压缩基线比较，LinearMask‑GS在Mip‑NeRF 360上将Gaussian数量从3.36M降至0.94M（约3.6×压缩），PSNR提升至27.70 dB（+0.49 dB），SSIM提升0.012，保持约591 FPS，且在其他基准上同样保持或提升PSNR/SSIM，压缩率明显优于现有方法。

**⚠️ 局限性**

需要在每个数据集上预先调节斜率τ和最小存活比例ρ，短掩码训练窗口可能导致光泽或透明区域判别不足，且方法假设静态场景，未处理动态或自适应调度问题。

---

## 296. Scores Alone Do Not Prove Discovery: The Discovery Certification Protocol for Auditing AI Research Agents

**arXiv ID:** 2609.09219 | [PDF](https://arxiv.org/pdf/2609.09219v1)

**作者:** Jingjie Ning `[一作]`, Ji Zeng `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了Discovery Certification Protocol (DCP)，通过分门别类的门控流程（Gate 1–3）对AI研究代理产生的数值结果进行可验证的可复现性评估。

**💡 创新点**

创新点在于：①将结果级别审计与可回溯的“恢复证据”相结合，支持任何合法实现路径；②引入“核心(Core)”与“证据(Evidence)”两种判定，分别关注零恢复率与基于随机对照的反馈效应；③提供统一的、可离线重放的可执行证明语言。

**🔧 技术方法**

技术手段包括：基于大型语言模型的研究代理（DeepSeek‑v4‑flash 与 DeepSeek‑v4‑pro）、封闭式评估器、随机对照实验、有限样本上界估计、记录Web访问、可确定性验证器与预注册实验框架。

**📊 数据集**

使用的数据集主要是：①SQLite 事件服务的16类查询（含4类高流量）用于索引优化；②虚拟催化剂控制实验（8^5=32,768 可能配方）；以及对设备、背包、仿射等调优案例的模拟数据。

**📈 对比分析**

对比方法：对每个目标结果，匹配代理在给定起始信息下的恢复率与原始实验得分进行比较；同时进行真值反馈与中性策略的随机配对实验。实验表明在两个完整审计中，96 次无线索回收均为零；在反馈实验中，30/30 真值分支恢复，0/30 中性分支，二者差值区间 [0.6379, 1.0] 远大于零，且对照实验保持在 ±0.17 以内。

**⚠️ 局限性**

局限性包括：①仅适用于可量化的单一数值结果，无法直接处理多目标或复杂因果结论；②对预注册和资源约束依赖较大，实际部署时可能面临成本与可复制性挑战；③评估仍需人工核对的模型与接口，误差或遗漏仍可能影响最终判定。

---

## 297. DuplexJail: Safety Alignment Breaks Under Spoken Interruption in Full-Duplex Models

**arXiv ID:** 2609.09420 | [PDF](https://arxiv.org/pdf/2609.09420v1)

**作者:** Jaechul Roh `[一作]` (University of Massachusetts Amherst), Andrea Fanelli `[通讯]` (Dolby Laboratories)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DuplexJail，评估全双工语音模型在接受用户语音时通过固定时间或拒绝触发的语音中断所带来的安全攻击；

**💡 创新点**

首次证明可通过在模型拒绝后注入预录语音来诱导有害回应，且将拒绝触发与时间控制相结合，揭示全双工交互的安全薄弱环节；

**🔧 技术方法**

采用全双工语音生成模型（PersonaPlex、PersonaPlex‑RL、FLM‑Audio、BayLing‑Duplex）、Whisper ASR、对抗性音频注入、文本拒绝表达检测与即时中断、Prompt Engineering 等技术；

**📊 数据集**

使用720条合成的有害请求（AdvBench 520条 + HarmBench 200条）以及相同的预录中断音频；

**📈 对比分析**

通过“whole‑response attack success rate”衡量攻击效果，固定延迟下 PersonaPlex 达到 40.3%（+33.8pp）和 PersonaPlex‑RL 48.7%（+39.3pp），拒绝触发下分别为 35.6% 与 48.6%；其他模型表现相对稳定或下降，表明中断策略对模型安全有显著影响；

**⚠️ 局限性**

仅在合成语音与有限模型上实验，缺乏多方言、真实用户交互的验证；中断时机受模型发音延迟差异影响，且未探讨多轮会话的长期安全性。

---

## 298. Exact-Form Regret for Gradient Descent, Mirror Descent and Follow-the-Regularized-Leader

**arXiv ID:** 2609.09466 | [PDF](https://arxiv.org/pdf/2609.09466v1)

**作者:** Ashkan Soleymani `[一作]` (Massachusetts Institute of Technology), Patrick Jaillet `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从几何视角出发，阐述了投影型一阶在线学习算法（如在线梯度下降、镜像下降和 FTRL）对何种形式的行动相关偏差（Φ‑regret）能够实现无后悔，并将其与相关的游戏均衡概念相连接。

**💡 创新点**

创新点包括：①提出“exactness”（位移场可由标量势函数产生）作为决定无后悔的核心几何准则；②将外部后悔、对称线性偏差、近端偏差等纳入同一框架；③证明正则化几何下的“mirror exactness”与“FTRL exactness”在边界处可能不同，导致控制的偏差类不相同；④利用“circulation”来刻画非保守场导致线性后悔的障碍；⑤基于上述偏差类定义新的均衡概念（Conservative Correlated Equilibrium）并探讨其与传统 CE、PCE 的关系。

**🔧 技术方法**

主要技术包括：几何分析（保守场、旋度、线积分）、凸分析（Bregman 距离、Moreau 包络、弱凸性）、投影和镜像映射、相对平滑性与统一凸性、潜在函数与一次性极值定理、以及对 FTRL 的累积对偶状态分析。

**📊 数据集**

本研究为理论性工作，未使用任何实测数据集；所有结果均为严谨的数学证明和理论界定。

**📈 对比分析**

比较方法：以 OGD、镜像下降、FTRL 的无后悔上界与通过“circulation”构造的线性后悔下界进行对比。性能表现为：在满足 smoothness/oscillation 条件时，针对所有可用的 exact‑form 偏差，算法能够实现 O(√T) 的无后悔；若偏差位移场存在非零旋度，则可构造对手使后悔线性。

**⚠️ 局限性**

局限性：①需要偏差的潜在函数具备可微、Lipschitz‑梯度等光滑性；②对非 Legendre 正则化或非光滑几何下的完整结果尚未给出；③对界面效应的处理依赖于额外的相对平滑性或 Lipschitz 条件；④在连续支撑下的均衡分离仅给出非完整密度示例，是否能在全维绝对连续分布上实现仍是未解问题。

---

## 299. From Pixels to Hierarchical Sequences: Quadtree Mask Encoding for Vision-Language Binary Change Detection

**arXiv ID:** 2609.09876 | [PDF](https://arxiv.org/pdf/2609.09876v1)

**作者:** Xiao An `[一作]` (Wuhan University), Wei He `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将遥感二时序变化检测问题转化为可检验四叉树文本生成的框架 QUAKE-CD，生成像素级二值掩模和基于视觉证据的链式思考解释。

**💡 创新点**

创新点在于：① 用四叉树语言将掩模压缩为可语法验证的文本序列；② 构造 QUAKE‑CoT 数据集，将掩模序列与链式思考对齐；③ 采用分阶段课程学习和语法门控双奖励强化学习，联合优化格式、像素精度与语义可信度。

**🔧 技术方法**

采用的技术包括 Qwen3‑VL‑8B‑Instruct 的自回归视觉语言模型、四叉树编码与解码、LoRA 微调、GRPO 强化学习、规则奖励（语法+Tversky）与思考奖励、以及多任务学习与进阶课程训练。

**📊 数据集**

使用 SYSU‑CD、LEVIR‑CD 及 LEVIR‑CD+ 这三个遥感变化检测数据集构成的 QUAKE‑CoT（约27k 训练样本，9k 测试样本），对二时序图像进行掩模编码与链式思考生成。

**📈 对比分析**

在与 BIT、ChangeFormer、ChangeCLIP、PixelLM、LISA、Text4Seg、RSUniVLM 等基线对比时，QUAKE‑CD 在累计 F1 达 78.31%（比 BIT 高 1.37%、比 ChangeFormer 差 3.13%），每图 F1 73.18%（超过所有专用检测器），召回率远超基线，精度显著提升；吞吐率约 74.27 对/秒，比解码器模型快 1.8×，与平面文本序列方法相当。

**⚠️ 局限性**

局限性：① 叶子大小 ℓ 的设置会影响学习可行性，过小导致序列过长、语法错误率上升，过大导致边界模糊；② 对极大尺寸图像的直接编码受限，需进一步分块或多尺度策略；③ 语法门控奖励需手工设计，可能难以推广到其它结构化任务；④ 当前仅在 64×64/32×32 级别实验，扩展到高分辨率场景仍需验证。

---

## 300. Ephemeral Feeds and Enduring Rituals: RushTok and the Formation of Event-Based Algorithmic Communities

**arXiv ID:** 2609.09331 | [PDF](https://arxiv.org/pdf/2609.09331v1)

**作者:** Emelia Hughes `[一作]` (University of Notre Dame), Tim Weninger `[通讯]` (University of Notre Dame)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对TikTok上“RushTok”事件的混合方法研究（调查与自我民族志）探讨了基于算法的临时社区如何围绕大学社团招募仪式形成、维持与边界化，并对平台治理提出建议。

**💡 创新点**

提出了“事件型算法社区”概念，阐释其与传统网络/身份社区的区别，并首次把“模糊归属、娱乐驱动、季节性序列化”与平台算法的互动联系起来；同时将“研究失访/沉默”视为方法学洞察。

**🔧 技术方法**

采用定性主题分析（OpenAI GPT辅助编码）、描述性统计与交叉表格，并结合平台推广数据（TikTok Promote metrics）进行量化描述。

**📊 数据集**

样本包括71名在2024年8月收视“RushTok”的TikTok用户调查问卷；此外收集了平台推广的观看/点击数据；未使用公开视频语料库或社交网络图谱。

**📈 对比分析**

方法主要为描述性对比（自认为是社区成员 vs 非成员）和主题编码，未与任何基准模型或实验进行性能对比；报告了定量比例和主题覆盖率。

**⚠️ 局限性**

局限性包括样本偏向女性、无校内成员、受限的自我报告；缺乏行为日志与内容语料分析；调查与访谈数据收集时间窗口有限；未进行因果推断；研究聚焦单一事件，缺乏跨事件比较。

---

## 301. Session Attestation for Unmodified TLS Services in Confidential Virtual Machines

**arXiv ID:** 2609.09668 | [PDF](https://arxiv.org/pdf/2609.09668v1)

**作者:** Qi Gu `[一作]` (NSFOCUS, Inc.), Sheng Ma `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于可信观察服务器本地生成的ECDHE公开密钥与TLS Finished结合的会话级远程证明方案，利用临时OS闩锁在不改动应用层TLS和证书的前提下，实现对可信执行环境（TEE）端点的证明。

**💡 创新点**

创新点在于：①只利用握手中公开的公共密钥而非TLS秘密，结合标准Finished验证即可保证TEE端点；②通过操作系统级的临时闩锁实现无应用层改动的验证流程；③支持双向（互相）证明并能与现有TLS实现无缝协同。

**🔧 技术方法**

技术手段包括：Linux/Windows的网络包过滤与临时闩锁（nftables、WinDivert、NFQUEUE）、libpcap抓取ClientHello/ServerHello、CSV/SEV‑SNP/Hygon CSV 可信证明接口、TLS 1.3、OpenSSL、标准TLS Finished验证以及对报文的临时阻塞与释放。

**📊 数据集**

使用真实 Hygon CSV 虚拟机实验环境：上传 1 KiB 至 64 MiB、互相证明 1 KiB/1 KiB 或 64 MiB/64 MiB 的数据流，实验涵盖 Linux CVM 与 Windows 11 客户端。

**📈 对比分析**

与本地 TLS 1.3、TLS+RA、TNG（RA‑TLS）方案对比：短连接平均延迟分别降低 63.1 %（Linux）和 23.0 %（Windows）；大文件传输接近原生性能，吞吐率约 89 %（Linux）/97 %（Windows）；CPU 与内存占用均低于 TNG，证明系统成本可控。

**⚠️ 局限性**

局限性：仅支持 TLS 1.3；需要操作系统层支持临时闩锁；验证过程仅覆盖客户端发起的敏感数据，服务器端主动数据的门控未完全实现；对四字节重用和更大规模实验的评估仍待深入；在非实验环境下的可部署性与可扩展性尚需进一步验证。

---

## 302. Auditable Emergency Triage for Maternal and Newborn Care in India

**arXiv ID:** 2609.09356 | [PDF](https://arxiv.org/pdf/2609.09356v1)

**作者:** Shobhit Jagga `[一作]` (Noora Health), Anubhav Arora `[通讯]` (Noora Health)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在印度WhatsApp平台上开发并部署了一套将急诊分诊拆分为症状提取和规则引擎的系统。

**💡 创新点**

创新点在于将LLM的作用限制为症状提取，采用可编辑规则引擎实现可追溯、可审计的决策流程，并在部署后实现快速规则迭代。

**🔧 技术方法**

技术包括Gemini-2.5-Flash LLM、规则引擎、症状词典、翻译步骤、两步分解。

**📊 数据集**

使用了从患者WhatsApp问询中收集的769条真实多语言（印地语、英语、泰卢固语等）查询，按验证/测试划分。

**📈 对比分析**

与端到端LLM和仅规则两种对比，最终系统召回率提升至0.810（±0.07）/F1 0.702，性能与仅规则相当，且可追溯。

**⚠️ 局限性**

局限在于词典和规则覆盖不足、缺失上下文导致错误、数据量小、未评估临床影响。

---

## 303. Learning to Adapt and Calibrate: Score Distribution Alignment for Few-Shot Uncertainty Prediction in Medical VLMs

**arXiv ID:** 2609.10333 | [PDF](https://arxiv.org/pdf/2609.10333v1)

**作者:** Xuan Cuong Ngo `[一作]` (University of Arkansas), Ngan Le `[通讯]` (University of Arkansas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了在少样本医学视觉-语言模型（VLM）适配后，如何通过分数分布对齐实现可靠的合成预测。

**💡 创新点**

创新点在于将监督少样本适配与自适应分数重加权相结合，利用覆盖间隙最小化目标来消除适配导致的非可交换性，从而在不牺牲适配性能的前提下恢复覆盖性。

**🔧 技术方法**

技术包括：监督少样本适配（线性探测、LoRA、残差适配器）、合成预测、分数重加权学习以及覆盖间隙最小化的理论驱动优化。

**📊 数据集**

数据集涵盖九个医学图像数据集（肿瘤切片、眼科、胸部X光）以及CLIP在ImageNet和CIFAR-10上的验证。

**📈 对比分析**

与SCP、Adapt+SCP、加权CP、SCAT、TIM、TransCLIP等基线比较，方法既保持或提升准确率，又实现目标覆盖率，预测集更紧凑，类条件覆盖差异下降。

**⚠️ 局限性**

局限性在于覆盖间隙界限依赖于分数分布近似；在极低样本或校准样本不足的情况下，界限可能松散，导致类条件覆盖不稳定。

---

## 304. Kernel-Managed Shared Memory for System-Wide Personalization

**arXiv ID:** 2609.10144 | [PDF](https://arxiv.org/pdf/2609.10144v1)

**作者:** Ryan Lum `[一作]` (Rutgers University), Yongfeng Zhang `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于内核管理的共享内存机制，让多智能体系统通过统一的内核进行记忆写入、检索、隐私控制和提示注入，以提升个性化交互效果。

**💡 创新点**

核心创新在于将个性化记忆的管理从各智能体迁移到系统内核层，统一写入顺序、身份解析、可见性规则与跨智能体可见性，避免了传统按代理实现的隐私泄漏和写入顺序竞争。

**🔧 技术方法**

技术包括：AIOS 操作系统风格框架、基于元数据的记忆结构、内核级写入序列号与超时阈值保证读写一致性、隐私可见性静态规则、语义相关性排序与令牌预算控制，以及格式化为自然语言的记忆注入。

**📊 数据集**

使用人工合成的用户画像和任务上下文数据，配合 1,800 次实验，生成的查询与响应均来自同一虚构用户，保证了实验的一致性和可控性。

**📈 对比分析**

通过与三种基线（全上下文拼接、检索增强、外部 Mem0 未内核化）在 GPT‑4o、Llama‑3.1‑8B、Qwen‑2.5‑7B 三个模型上进行 1,800 次评测，自动化评审与人工评估表明内核管理方案在个性化得分（Profile/Task/Integration）上显著高于未管理后端和标准 RAG，且在低于全上下文拼接的模型上实现 15–61% 的延迟与令牌成本下降。

**⚠️ 局限性**

局限包括：在 Llama‑3.1‑8B 上的 Task 与 Integration 得分略低于全上下文拼接，缺乏对不同任务类型、记忆粒度与长期一致性影响的深入分析，以及对跨用户身份解析错误的边界情况未完全覆盖。

---

## 305. AXON: A ROS 2 RMW with Shared-Memory/QUIC Transport and QKD/ML-KEM Key Establishment

**arXiv ID:** 2609.10024 | [PDF](https://arxiv.org/pdf/2609.10024v1)

**作者:** Sergio Sánchez de la Fuente `[一作]` (Universidad de León), Ángel Manuel Guerrero-Higueras `[通讯]` (Universidad de León)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

实现了一种ROS 2 RMW，使用同主机共享内存、跨主机QUIC、并通过守护进程进行发现与图同步，并提供两种安全模式（后量子ML‑KEM混合交换和QKD外部PSK）。

**💡 创新点**

创新点在于：①分离本地与远程传输，减少网络负担；②通过QUIC提供统一安全链路并可插拔后量子密钥交换；③实现QKD密钥作为TLS外部PSK并在QKD模式下可选的应用级AEAD封装。

**🔧 技术方法**

使用技术包括Rust核心、C++适配器、POSIX共享内存+eventfd、quinn QUIC实现、rustls（经fork支持外部PSK）、aws‑lc‑rs加密库、ETSI QKD 014 API、Zstd压缩。

**📊 数据集**

未使用公开数据集；验证基于ROS 2节点交互、两台机器的仿真环境和QuKayDee QKD模拟器。

**📈 对比分析**

方法为在实验室两机上对比两种安全模式下的握手和数据传输，但未给出定量性能指标，说明性能尚未测评。

**⚠️ 局限性**

局限性包括：不兼容DDS wire；缺乏证书验证、主动MITM防护；QKD模式无前向保密；模式匹配需全网统一；依赖rustls fork；未完成完整ROS 2 RMW合规与第三方安全审计；未在真实QKD硬件上验证。

---

## 306. Active Adaptation, Not Static Defense: Temporal Dynamics of Preventative Steering in Adversarial Fine-Tuning

**arXiv ID:** 2609.10142 | [PDF](https://arxiv.org/pdf/2609.10142v1)

**作者:** Jing Guan `[一作]` (JIUTIAN Research), Junlan Feng `[通讯]` (JIUTIAN Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并改进了在LLM恶意微调中使用的“Preventative Steering”防御方法，证明其对模型参数的持续适应起关键作用，并提出了动态注入强度调度策略PIS来提升安全稳健性。

**💡 创新点**

创新点在于：①通过机制分析揭示Preventative Steering的两阶段动态（补偿性适配与稳态），并证明单次权重偏移不足以持续防御；②提出Progressive Intensity Scheduling（PIS），根据梯度与攻击方向的对齐情况动态调整注入强度，从而在整个训练过程中维持防御效能。

**🔧 技术方法**

主要技术包括：激活向量注入（preventative steering）、梯度对齐度（cosine similarity）监测、参数空间写路分析、子空间投影保留实验（IDP）以及梯度投影持续实验（IDP Continuation）和动态强度调度（PIS）。

**📊 数据集**

实验数据集主要使用对抗性微调数据（覆盖恶意、附和、幻觉三种危险特征）以及一系列安全与人物特质评估基准（Forbidden, StrongReject, XSTest, CnSafe, Jade, JB-Distill, SweEval等）。

**📈 对比分析**

与静态强度防御（固定α）比较，PIS在Qwen2.5-7B/32B与Gemma-3-12B三大模型上均显著提升安全平均得分（最高提升约+6点）并将有害特质得分降低至约1.3–1.5；PIS在不同起始强化时间、最大强度倍数和注入层上均保持优于静态方案，证明了动态调度的有效性。

**⚠️ 局限性**

局限性包括：仅验证了三种危险特征和少数模型族；对多语言、多攻击样本、不同模型架构的泛化尚未验证；安全基准覆盖有限，可能未覆盖所有安全风险；PIS的超参数（起始点、增幅比例）仍需经验选择。

---

## 307. Semi-Implicit Pairwise Descent for Nonlocal Continuum Mechanics

**arXiv ID:** 2609.09834 | [PDF](https://arxiv.org/pdf/2609.09834v1)

**作者:** Xukun Luo `[一作]` (Chinese Academy of Sciences and University of Chinese Academy of Sciences), Xiaowei He `[通讯]` (Chinese Academy of Sciences)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Semi-Implicit Pairwise Descent（SIPD）算法，实现大规模超弹性材料在复杂接触与摩擦下的实时模拟，避免了传统方法的Hessian计算和正定性投影问题。

**💡 创新点**

创新点包括：①将FEM方程改写为非局部对称对偶力形式，完全依赖一阶能量梯度；②引入SIPD迭代和无SVD的解析投影策略保证局部刚度矩阵正定；③统一把接触与摩擦建模为各向异性弹性能量，直接加入非局部对偶力。

**🔧 技术方法**

使用的技术主要有：非局部连续介质理论（Peridynamics）、半隐式Jacobi式迭代、解析投影分解、GPU并行化（CUDA）、可视化线搜索。

**📊 数据集**

实验数据集：大规模四面体网格（如28只兔子、1.8M四面体面条、1.03M兔子模型、1.8M粘面等），并在多种物理参数（弹性模量、泊松比、摩擦系数）下进行仿真。

**📈 对比分析**

与传统Newton、VBD、Jacobi等方法比较，SIPD每次迭代成本显著降低（寄存器占用约56，GPU占用率高），整体收敛时间比VBD快约1.6×，并在高泊松比、大摩擦系数、百万级碰撞对等极端测试中保持稳定。

**⚠️ 局限性**

局限性包括：①接触摩擦模型依赖网格分辨率，可能产生虚假接触力；②当前方法不支持断裂模拟；③基于惩罚的接触未完全禁止穿透，未来可结合IPC等连续碰撞检测来提升鲁棒性。

---

## 308. Deep and shallow biases in language models

**arXiv ID:** 2609.09901 | [PDF](https://arxiv.org/pdf/2609.09901v1)

**作者:** An Vo `[一作]` (MBZUAI), Daeyoung Kim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“偏差深度”概念及其度量指标 π=DR×FR，用以区分语言模型在不同表述下是否保持相同偏好，并基于此评估 LLM 的深层偏差（Deep bias）与浅层偏差（Shallow bias），从而揭示偏差来源和去偏难度。

**💡 创新点**

创新点在于：①将模型回答的聚焦程度（DR）与在情境重述中保持一致的比例（FR）结合，形成单一深度得分；②通过场景重述方法区分稳定偏好与表述敏感偏好；③将偏差与模型训练阶段（预训练、SFT、DPO、RLVR）关联，揭示深度偏差主要由预训练决定；④使用 GEPA 与 LoRA‑SFT 两种去偏技术进行对比，证明深度偏差更难去除。

**🔧 技术方法**

使用的技术包括：多次采样（温度0.6）、答案聚类与规范化、偏差深度评分、DEPI（场景重述）、GEPA（系统提示优化）以及 LoRA‑SFT（多样性压平微调）。

**📊 数据集**

数据集来源于 Olmo‑3‑7B 的公开 SFT 数据，经过筛选、重写、去重后得到 4,442 个“选择随机”提示家族（共 133,260 个重述），覆盖预训练、SFT、DPO、RLVR 四个训练阶段。

**📈 对比分析**

与单提示评估相比，本文的双轴评估显示约 75% 的偏差为浅层；在四个后 SFT 模型上，深度偏差比例从 35.8%–77.7% 变为 13.2%–39.6%；对去偏方法的实验表明 LoRA‑SFT 在降低深度偏差（4.5%）和浅层偏差（10.3%）方面均优于 GEPA（1.6% 与 5.8%）。

**⚠️ 局限性**

局限性包括：1) 大部分偏差无法通过单一训练样本归因，需进一步研究分布式模式；2) 评估数据以低风险日常提问为主，尚未验证在高风险决策场景的适用性；3) 结果受采样温度与模型版本等因素影响，需更系统的跨模型评估。

---

## 309. Prototyping QoE-Aware Rate Adaptation in Cellular Networks with Commercial Applications

**arXiv ID:** 2609.09490 | [PDF](https://arxiv.org/pdf/2609.09490v1)

**作者:** Szilveszter Nádas `[一作]` (Ericsson Research), Eric Petajan `[通讯]` (AT&T)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种可在现有商用实时视频流应用上实现的 QoE‑aware 资源分配原型，利用外部 QoE 测量与网络侧拥塞信号实现动态调节。

**💡 创新点**

其创新点在于引入“复合空间复杂度”将视频空间复杂度与谱效率融合，并设计了针对商用应用约束的增量资源重新分配算法。

**🔧 技术方法**

采用 AMVOTS 作为外部 QoE 采集工具，SCONE/QUIC/MoQ 作为网络与应用间的通信框架，并使用增量重新分配与最大效用分配算法。

**📊 数据集**

数据集基于 AMVOTS 测试集，包括多种实时交互视频内容（云游戏、XR、视频会议等），并在实验室环境中收集网络测量。

**📈 对比分析**

通过与传统等比率速率分配和静态目标 QoE 分配对比实验，原型在相同吞吐量下实现约 3 倍并发会话数，并在实验室环境中显示显著 QoE 提升。

**⚠️ 局限性**

主要限制在于缺乏完整的空间复杂度曲线、对商用应用的 QoE 反馈依赖外部工具、受限的速率控制幅度以及隐私与可信度的挑战。

---

## 310. The Era by Eon Benchmark: A Generated Enterprise Estate with Exact Ground Truth for Benchmarking LLM Agents

**arXiv ID:** 2609.09853 | [PDF](https://arxiv.org/pdf/2609.09853v1)

**作者:** Benjamin Gruenbaum `[一作]` (Eon), Or Itzahary `[通讯]` (Eon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 Era by Eon Benchmark，自动生成完整的虚拟企业环境（企业应用模拟器、内部数据库、问题及答案）以评估 LLM 代理。

**💡 创新点**

创新点在于完全自动化生成共享实体图、投影到多款产品模拟器、根据业务问题生成内部数据库，并通过计算答案、一致性校验与真实性评分卡保证无人工答案、数据可复现且真实。

**🔧 技术方法**

采用种子化随机生成器、代码模板化问题/答案生成、专用内部数据库生成器、一致性/依赖检查、Boosted‑tree 真实性判定、文本多样性统计与计量评估等技术。

**📊 数据集**

使用 Eon 运营数据、公开统计与估算值构成的现实性目标作为参考，生成的虚拟公司与这些统计对齐；不直接使用真实企业数据。

**📈 对比分析**

在单一虚拟公司上对 9 种 LLM 代理做 3 次重复，评估 33 个问题的准确率，最高 76.8% 低至 42.4%；多跳、跨系统问题最难，配对检验仅有 3 个显著差异。

**⚠️ 局限性**

局限性包括仅覆盖单公司、单轮问题；内部数据库与文档部分未评估；现实性目标来源有限；模板生成文本仍缺乏人类写作的多样性与细节；评估仅针对模拟器问题。

---

## 311. Polynomial-time algorithms for setting tight big-M coefficients in transmission expansion planning with disconnected buses

**arXiv ID:** 2609.09474 | [PDF](https://arxiv.org/pdf/2609.09474v1)

**作者:** Behnam Jabbari-Marand `[一作]` (North Carolina State University), Adolfo R. Escobedo `[通讯]` (North Carolina State University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种名为最长最短路径连接（LSPC）算法，用于在电网扩张规划（DC‑TEP）中为不连通的总线对求取紧致的角度差上界，并基于该上界构造有效不等式；

**💡 创新点**

核心创新在于用图论思路避免求解 NP‑hard 的最长路径问题，通过先在已连通网中求最短路径，再与新增子网的树结构相连，构造“最长最短路径连接”，得到比传统 LPP 或 SPP 方法更紧的 big‑M 上界；

**🔧 技术方法**

主要技术包括混合整数线性规划（MILP）DC‑TEP 重建、Dijkstra 最短路径算法、节点隔离图和邻接集合构造、集合论路径枚举与证明、以及复杂度分析（O(m log n)）；

**📊 数据集**

论文未使用公开数据集，实验验证基于示例网络图进行；

**📈 对比分析**

与传统的 LPP、SPP 以及 γ‑上界方法对比，LSPC 产生的角度差上界更小、生成的路径基不等式更强，能够在更大规模实例中显著缩小 LP 松弛并提升求解速度；

**⚠️ 局限性**

局限性在于仅适用于初始网络已连通且新增子网为相对简单的树状结构；若新增子网结构复杂或包含多条相互连通的路径，LSPC 可能失效；此外，算法依赖于节点度分布近似指数衰减的假设。

---

## 312. IMU-Centric Moving Horizon Estimation for Lateral Dynamics Estimation Across Vehicles and Grip Conditions

**arXiv ID:** 2609.10202 | [PDF](https://arxiv.org/pdf/2609.10202v1)

**作者:** Seuffo Akouan ha Ngoune `[一作]` (University of Modena and Reggio Emilia), Marko Bertogna `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于IMU的移动窗口估计（MHE）框架，用于在不依赖外部里程计或复杂轮胎模型的情况下重建车辆的侧向速度和侧向加速度；

**💡 创新点**

创新点在于：1）将可辨识的Sine Saturation Tire（SST）简化轮胎模型与MHE结合，允许在线估计有效轮胎力容量系数；2）通过在MHE中引入状态和参数约束，实现对非线性与饱和现象的鲁棒估计；3）在不需要预先测量侧向速度的前提下，利用IMU测得的侧向加速度和车道角速度完成状态重建；

**🔧 技术方法**

使用技术包括：基于平面单轨动力学模型、SST轮胎近似、移动窗口非线性优化（MHE）以及多步射击与自动微分求解；

**📊 数据集**

实验数据集包括：2014年Ferrari 250 LM与1963年Corvette Grand Sport的公开REVS数据集，以及阿布扎比自动驾驶赛车联盟Super Formula EAV-25的官方赛道数据；

**📈 对比分析**

与传统的Kalman滤波（EKF/UKF）和基于Pacejka模型的LOP-UKF进行对比，MHE在侧向速度RMSE、侧向加速度以及转向角速度估计上均优于滤波器，尤其在极限操控和低抓地力条件下表现更为稳健；

**⚠️ 局限性**

局限性包括：1）对SST曲率因子B的固定预设可能在不同驾驶条件下引入模型误差；2）MHE求解时仍需一定计算资源，虽已在高端CPU上实现实时，但在嵌入式平台上可能受限；3）在极低侧向激励或直线行驶时可观测性退化，需通过参数约束补偿；

---

## 313. Actuator Dynamics Curricula for Narrow-Viability Tasks in Legged Robot Learning

**arXiv ID:** 2609.09492 | [PDF](https://arxiv.org/pdf/2609.09492v1)

**作者:** Kousheek Chakraborty `[一作]` (Saxion University of Applied Sciences), Abeje Y. Mersha `[通讯]` (Saxion University of Applied Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种基于关节刚度的奖励进阶训练框架（Actuator Dynamics Curriculum），通过在训练时从高刚度逐步降至系统识别值，解决了在狭窄可行性任务中探索因终止而缺乏梯度信号的问题，并在Spot机器人上实现了无零差异的手站立转移。

**💡 创新点**

提出将模拟关节动态（刚度）作为可调课程维度，用闭环系统的自然频率提升可行性核以扩大探索空间，系统地证明自然频率提升可增大可行性核，并将该方法应用于实际硬件。

**🔧 技术方法**

采用基于PPO的强化学习，结合IsaacLab仿真、CMA-ES系统辨识、指数滑动平均的关节刚度退化策略、闭环PD控制以及域随机化。

**📊 数据集**

主要使用Boston Dynamics Spot的模拟环境与其系统辨识得到的关节参数，训练数据来自随机探索的状态分布；未使用公开数据集，而是基于仿真和现场实验收集的轨迹。

**📈 对比分析**

与固定刚度、随机刚度、时间退化等基线进行对比，在Spot手站立任务中，课程训练的平均回合长度从约400提升至≈975，奖励从-42提升至-4.8，成功率在模拟与硬件上均达到100%；其它基线表现显著不如课程。

**⚠️ 局限性**

理论证明仅适用于简化的摆杆系统，缺乏对多刚体系统的通用性；课程目前仅针对单一任务，参数手工设定；硬件评估仅为定性，缺乏量化失败模式和自适应调度机制。

---

## 314. CS-Guard: Benchmarking LLM Guardrails for Code Generation Security

**arXiv ID:** 2609.09798 | [PDF](https://arxiv.org/pdf/2609.09798v1)

**作者:** Jinyang Li `[一作]` (Adelaide University), Hung X. Nguyen `[通讯]` (Adelaide University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 CS-Guard 基准，系统评估不同类型的 LLM 代码生成安全防护（guardrail）在文本到代码、代码到代码以及小说情境攻击（FSA）下的效果。

**💡 创新点**

创新点包括：①首次提供针对代码生成安全的系统化基准；②提出嵌入合法软件情境的恶意指令——FSA，显著提升评测真实性；③设计三层 guardrail 分类法（类别、操作、位置），实现模块化与可扩展评估。

**🔧 技术方法**

技术手段：使用多种 LLM（如 CodeLlama、DeepSeekCoder、GPT‑OSS‑20B、Qwen3‑30B、GPT5‑mini 等）与 9 种 guardrail（策略型、分类器型、内部型）；利用 LLM 标签器、vLLM 生成、对抗式 jailbreak（CipcherChat、EmojiAttack 等）与 FSA 构造；评估指标为攻击成功率（ASR）和 F1 分数。

**📊 数据集**

数据集：1,000 条 Meta CyberSecEval 的恶意文本指令；7 种 jailbreak 攻击；FSA 1,000 条（及 300 条子集）；331 条代码到代码恶意生成样本（RMCBench 与 Redcode‑Gen 采集）。

**📈 对比分析**

比较方法：对输入位置的 guardrail 计算 ASR，对输出位置的 guardrail 计算 F1；对策略型 guardrail 报告 ASR 减少比例。实验结果显示：内部 guardrail 在文本到代码攻击中 ASR 仍高达 15–61%；在代码到代码中 ASR 接近 100%；FSA 在所有 guardrail 下 ASR 约 85–99%，表明现有防护在复杂情境下效果有限。

**⚠️ 局限性**

局限性：仅覆盖英文数据；多轮 jailbreak 评估不足；未对白盒内部 guardrail（如 neuron‑level 调整）进行深入测试；内部 guardrail 仅在量化模型上评估，可能不代表全精度版本。

---

## 315. Compact Visuotactile World Models for Lifting: Prediction, Reward Alignment, and Force Constraints

**arXiv ID:** 2609.09597 | [PDF](https://arxiv.org/pdf/2609.09597v1)

**作者:** Qinzhen Ma `[一作]` (Rice University), Sida Peng `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在机器人抓取和抬升任务中，作者通过将视觉与触觉融合进一个随机初始化的紧凑世界模型，并在此基础上进行轨迹级不确定性校准与基于想象的演员-评论家学习，探讨了触觉感知对力预测和决策效果的实际影响。

**💡 创新点**

创新点在于将触觉信息直接嵌入模型、结合轨迹级校准以及利用想象策略学习来评估感知误差对决策质量的影响，首次在同一实验框架下系统比较视觉、视觉+触觉与力反馈三种信息源。

**🔧 技术方法**

技术上使用了随机初始化的卷积-MLP SimNorm 世界模型、分层残差校准方法、分支演员-评论家学习、MuJoCo 物理模拟器以及公开的 GelSight 触觉记录，构成完整的感知-预测-控制链。

**📊 数据集**

数据集包括 160 条 MuJoCo Lift 轨迹（训练/验证/校准/ID/OOD 分割）和 102,628 张公开 GelSight Mini 触觉图像与 ATI Nano17 力标注，用于训练与评估感知与力预测。

**📈 对比分析**

实验比较了视觉单一、视觉+触觉（行为克隆、强化学习）以及力反馈基线，结果显示触觉能将终点力误差从 1.058 N 降至 0.228 N，并将新环境中 10 cm 抬升成功率从 20 % 提升至 93 %（但在 8 N 预算下仍只有 33 % 的成功率）。

**⚠️ 局限性**

局限性包括仅使用单一刚体几何、缺乏传感器与模拟的跨域转移、样本环境有限、轨迹校准不具备在线安全保证，以及未能在力约束下完全验证任务成功率。

---

## 316. Longitudinal tracking of multiple sclerosis lesions in the spinal cord: A validation study

**arXiv ID:** 2609.09424 | [PDF](https://arxiv.org/pdf/2609.09424v1)

**作者:** Pierre-Louis Benveniste `[一作]` (Polytechnique Montreal), Julien Cohen-Adad `[通讯]` (Polytechnique Montreal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

对脊髓多发性硬化病灶的纵向实例级跟踪方法进行评估与比较，提出并验证了五种不同的跟踪策略。

**💡 创新点**

首次在脊髓MS中实现一对多/多对一的病灶对应，证明基于注册的IoU匹配与基于SC坐标的注册‑free XGBoost 能有效处理病灶分裂与合并。

**🔧 技术方法**

使用SCT的脊髓坐标系、Hungarian算法、XGBoost、Siamese网络、变形注册和IoU匹配；分割采用nnUNet产生的SC‑MS病灶掩码。

**📊 数据集**

采用5医院（Calgary, Edmonton, Montreal, Toronto, Vancouver）收集的CanProCo队列数据，34名患者，PSIR/STIR 0.7×0.7×3 mm分辨率的纵向扫描；人工专家标注为基准。

**📈 对比分析**

通过留一交叉验证（LOOCV）与独立测试集比较，评估TP/FP/FN、精度、召回与F1。最优策略（#5）在测试集达到F1 = 0.98，LOOCV F1 = 0.97；XGBoost（#2）次之（F1 ≈ 0.91/0.88），其余策略表现较弱。

**⚠️ 局限性**

局限性包括：分割误差与注册误差导致漏检/误检；样本量和仅两时间点限制了泛化与多时序评估；依赖椎间盘定位与SC坐标的准确性；对复杂形状病灶及大规模多中心数据的适用性尚未验证。

---

## 317. Building the Harness Automatically: Self-Play in Code Distills a Text Harness for Black-Box Optimization

**arXiv ID:** 2609.09468 | [PDF](https://arxiv.org/pdf/2609.09468v1)

**作者:** Yi Wu `[一作]` (Google), Lukasz Heldt `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种“可执行实践→文本化驻固”的框架，通过在外部开发目标上反复写作、评估和记录优化器代码，最终一次性将经验蒸馏为一段约200字的可冻结文本 harness 并在多种 LLM 执行器上直接使用。

**💡 创新点**

创新点在于：①将可执行优化器的实践记录一次性蒸馏为可冻结文本，从而实现策略在不同模型执行器和不同目标几何上的可迁移；②通过严格的门控（gate）与冻结（seal）保证公开评测时不受实时反馈影响，提升可审计性与稳健性；③独立复现证明性能层级可被复制，强调可重复性目标而非字面一致。

**🔧 技术方法**

技术包括：LLM 代码生成与执行（如 Gemini Flash、Claude Sonnet），可执行实践循环（写代码、评估、记录），门控与冻结机制，文本蒸馏（Distill）与执行接口封装，统计检验（Wilcoxon、Holm 校正）以及对比实验框架。

**📊 数据集**

数据集主要有：随机偏移正半定二次函数（开发集）、BBOB 公开景观（Bent Cigar、Gallagher‑101、Rastrigin）作为测试集，以及封闭的 YouTube 奖励调优内部数据集。

**📈 对比分析**

与随机搜索、CMA‑ES、GP‑BO（普通与加强版）、以及保留的实践程序等基线对比，冻结文本 harness 在 20 次评估预算下实现了约 48% 的 regret 降低（与 Flash Base 64.2→32.9），在不同执行器上平均 regret 均显著下降，且在 BBOB 景观上提升至 GP‑BO 级别。

**⚠️ 局限性**

局限性包括：实验主要在低维（D=8）、固定预算（B=20）和有限样本（N=10）下进行，部分结果依赖 30 次独立复现；未覆盖所有经典优化器（如 HEBO、TuRBO）；跨模型迁移仅测试了少数执行器；以及一刀切的蒸馏可能不适用于需要多轮自适应反馈的场景。

---

## 318. CARRE: Counterfactual Action Retrieval and Reason Evaluation for Explainable Churn Prescription

**arXiv ID:** 2609.09766 | [PDF](https://arxiv.org/pdf/2609.09766v1)

**作者:** MinJoo Kim `[一作]` (Hanyang University), SeungHwan Cho `[通讯]` (Hanyang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 CARRE 三阶段 churn 处置管道，先用检索增强获取候选保留动作，再用成本感知的反事实评分挑选最优行动，最后通过 LLM 进行原因分类与自然语言解释，形成完整的预测‑处置‑解释闭环。

**💡 创新点**

创新点在于：① 将检索、成本优化与 LLM 解释严格分离，提升模块可评估与可替换性；② 直接模拟候选动作的反事实风险变化并引入成本惩罚，显著优于仅基于 SHAP 的解释方法；③ 通过结构化提示与链式推理实现解释与处置的一致性与可解释性。

**🔧 技术方法**

技术细节包括：FAISS+句子编码器检索；加权校准的逻辑回归作为 churn 预测模型；λ‑惩罚的成本感知反事实评分；OpenAI/Groq LLM（GPT‑4o 等）配合 CoT 结构化提示进行原因归因与解释生成。

**📊 数据集**

使用 IBM Telco Customer Churn 数据集（7043 条记录，19 个特征），按 80/20 随机分割为训练/测试集。

**📈 对比分析**

方法对比：随机、规则、SHAP、SHAP+Cost 四种基线；在 313 个高风险测试样本上，CARRE 的平均模型预测风险降低为 0.302，远高于 SHAP 的 0.168（提升 79.8%），成本归一化效率为 0.093 对比 0.084（提升 10.5%）。在 136 案例弱标签一致率由 79.4% 提升至 90.4%，LLM 评估得分在 4.02–5.00/5 范围内。

**⚠️ 局限性**

局限性：① 仅评估模型预测的风险变化，未验证实际因果效应；② 弱标签基于手工规则，缺乏专家验证；③ 动作空间仅包含 6 种保留手段，无法覆盖所有 churn 原因；④ λ 与 k 的设置在同一数据集上调优，缺乏跨数据集或跨行业的稳健性；⑤ LLM 评估与人工一致性低，需更严谨的人工标注与验证。

---

## 319. 5-Dialects-BN: Unmasking the Impact of Transliteration on Bangla Dialectal LLMs

**arXiv ID:** 2609.09964 | [PDF](https://arxiv.org/pdf/2609.09964v1)

**作者:** Md Mahir Jawad `[一作]` (BRAC University), Md Farhad Alam Bhuiyan `[通讯]` (Penta Global Limited)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了第一个多注释孟加拉方言基准（Bangla Dialect Benchmark），包含 5 种方言（Chittagong、Barisal、Noakhali、Sylhet、Rangpur）的 6,000 条手工标注条目，每条记录包括原始方言文本、罗马化转写、标准孟加拉翻译、英语翻译以及主观性（二元）标签。

**💡 创新点**

创新点：
1) 将罗马化与原始文本并置，首次系统研究脚本对大语言模型（LLM）性能的因果影响；
2) 同步提供方言、标准、英语及主观性标签的 5 维齐齐标注，形成跨语言、跨维度的多任务基准；
3) 通过仅 160 条样本的 LoRA 微调，证明开放源代码 LLM 在低资源方言任务上可超越闭源零样本模型；
4) 对罗马化导致的性能下降进行信息流失、子词碎片化、分布稀疏三因果拆解。

**🔧 技术方法**

技术：
- LLM 评估：Gemini 3 Flash、GPT‑4o‑mini、Claude Haiku 4.5（闭源）及 Qwen‑3‑4B、Gemma‑4‑4B、Llama‑3.1‑8B、Mistral‑7B（开源）；
- 四种提示/微调策略：零样本、少样本、链式推理（CoT）、LoRA 微调；
- 评估指标：BLEU、ROUGE‑2/​L、METEOR、chrF++、COMET（翻译）、macro‑P/​R/​F1（主观性）及 chrF++/COMET（标准化）；
- 脚本多方案实验：Avro、ITRANS、ISO‑15919。

**📊 数据集**

数据集：Bangla Dialect Benchmark（6,000 条条目），采集自 YouTube 评论、Facebook 群组、Reddit、新闻等公开在线资源，经母语者验证、人工翻译、英文翻译与主观性标注。

**📈 对比分析**

比较方法与性能：
- 在零/少样本提示下，闭源模型表现稳定；开源模型表现波动大，尤其 Llama‑3.1‑8B 在少样本下出现示例干扰；
- LoRA 微调后，所有开源模型的翻译 BLEU 均提升至 64–74 区间，Mistral‑7B 从 16.2 提升至 73.6，超过 Gemini 3 Flash 的 46.4；
- 主观性 F1 在 LoRA 后达到 74–79，超过所有闭源零样本基准；
- 罗马化输入导致所有模型平均降低 12–17 BLEU，且在翻译任务中差距最大。

**⚠️ 局限性**

局限：
1) 方言样本分布不均（Chittagong 1,900 条，其他方言仅 700 条），可能影响跨方言迁移研究；
2) 罗马化缺乏统一规范，导致脚本不确定性；
3) LoRA 微调仅评估 160 条/方言（1,000 条总计）且未给出规模曲线；
4) 数据仅覆盖书面方言，口语方言表现未知；
5) 未进行人工质量评估，仅使用自动指标；
6) 仅对翻译与主观性进行了 LoRA 微调，标准化任务仅提示。

---

## 320. Streaming P300 Acquisition and Statistical Signal Validation Across Five EEG Platforms: A Hardware-Agnostic BrainFlow/LSL Pipeline

**arXiv ID:** 2609.10047 | [PDF](https://arxiv.org/pdf/2609.10047v1)

**作者:** Isabella Guan `[一作]` (Lake Washington High School), Fusheng Wang `[通讯]` (Stony Brook University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了一个基于BrainFlow/LSL的硬件无关实时P300采集管线，并在五种EEG平台上验证其可用性；

**💡 创新点**

创新点包括：跨平台统一采集流程、使用标签置换的非参数信号可分离性检验、对行列闪烁间隔进行调度优化，以及对不同字符解码策略的系统评估；

**🔧 技术方法**

采用了BrainFlow与LSL实现数据同步，xDAWN空间滤波、Riemannian分类与逻辑回归解码器，以及峰值置换测试和交叉验证等技术；

**📊 数据集**

使用了包含20个Flex会话（共131字符）以及初步试验的自制干湿电极、Emotiv EPOC X、Muse 2等多平台数据；

**📈 对比分析**

通过比较每个平台的信号分离p值和采集可靠性，发现Flex在小样本下p值最低（0.034），Muse 2在12场次中可靠性最高；在20会话扩展验证中，Flex的峰值置换测试和累计xDAWN解码AUC均超过阈值，AUC最高达0.72；

**⚠️ 局限性**

局限性包括：受试者与会话数量有限、P300信号相对弱、头戴设备覆盖范围不足、评估方法可能产生数据泄露，以及尚未实现实际可用的拼写精度。

---

## 321. Characterizing Multi-Cell Pinching-Antenna Transmission: Revealing the Other Side of the Coin

**arXiv ID:** 2609.09585 | [PDF](https://arxiv.org/pdf/2609.09585v1)

**作者:** Zhiguo Ding `[一作]` `[通讯]` (Nanyang Technological University), Zhiguo Ding (Nanyang Technological University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过随机几何分析研究了在多小区毫米波系统中使用可压缩天线（pinching antenna）时的干扰抑制效应，并与传统天线进行对比。

**💡 创新点**

创新点在于：① 引入了分数功率控制与可压缩天线的结合，揭示了其在降低基站发射功率、从而降低互小区干扰方面的优势；② 在一维和二维模型下给出了封闭式或近似的下行成功概率与平均速率表达式；③ 在小密度（λ→0）情形下推导了更直观的性能近似，验证了可压缩天线在低密度网络中始终优于传统天线。

**🔧 技术方法**

技术手段包括：随机几何模型（HPPP）、Slivnyak 定理、分数功率控制、LoS/NLoS 链路模型、概率生成函数（PGFL）、拉普拉斯变换、Bessel 函数与不完全伽马函数的近似。

**📊 数据集**

数据集：仿真参数包括载波频率 28 GHz、波导高度 d=3 m、ϵ=1、阻塞参数 β=0.5、噪声功率 −90 dBm；通过 Monte‑Carlo 仿真验证理论结果，探讨 λ（基站密度）与 r_c（小区半径）对性能的影响。

**📈 对比分析**

比较方法：将可压缩天线系统的成功概率（或平均速率）与传统天线系统（固定或分数功率控制）在相同基站密度、相同基站功率设置下进行对比。结果显示：① 在一维模型中，可压缩天线始终取得更低的失误概率；② 在二维模型中，随着 SNR 提升，可压缩天线的平均速率显著高于传统天线；③ 在低基站密度和大小区半径时，优势尤为突出。

**⚠️ 局限性**

局限性：① 2D 情形下的平均速率无法得到闭式表达，需数值积分；② 分析主要假设 LoS 链路始终存在或采用简化的 LoS/ NLoS 混合模型；③ 未考虑多用户多天线或非 TDMA 调度策略的影响；④ 只讨论了下行链路，未分析上行或全双工情况。

---

## 322. Recovering Biomechanical Signals from Missing Keypoints Using Temporal Interpolation in Monocular Gait Analysis

**arXiv ID:** 2609.09670 | [PDF](https://arxiv.org/pdf/2609.09670v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 323. LogiScope-VQA: Benchmarking Vision-Language Models for Logistics Hazard Identification in Industrial Scenarios

**arXiv ID:** 2609.09790 | [PDF](https://arxiv.org/pdf/2609.09790v1)

**作者:** Hanjing Zhou `[一作]` (Cainiao Group, Alibaba Group), Yanbing Zhou `[通讯]` (Cainiao Group, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并构建了首个基于真实工业仓储监控的多模态安全评估基准（10,274条VQA实例、5,394张视觉样本），并对20款主流LMM进行工业元素感知、仓储知识理解和潜在风险推理三维度的系统评测；

**💡 创新点**

首次将真实工业环境与逐步递进的课程体系、LLM-as-Judge评估、以及安全风险偏差分析融入多模态基准，揭示了细粒度感知瓶颈与过度风险报告问题；

**🔧 技术方法**

采用视觉语言预训练、链式思考（thinking mode）、多模态提示与解码、LLM-as-Judge开放式问答解析、层级式标注与三轮交叉验证，以及风险偏差评分等技术；

**📊 数据集**

使用从Cainiao全球仓储园区收集的3.5M原始监控片段（精炼为5,394高质量样本）和10,274条由物流专家标注的VQA对，涵盖18类核心物体和20类风险因素；

**📈 对比分析**

对比了6款商业模型（Claude, Gemini, GPT, Qwen3.7）与多款开源模型（Qwen3.5-Plus、Kimi、GLM等），以及随机/频率基准和人工专家；结果显示顶尖开源模型在整体准确率可与商业模型相当，但在细粒度感知上仅达0.39/0.36，远低于专家（>0.95），在知识任务上已超越非专业人群；

**⚠️ 局限性**

存在细粒度感知仍然偏低（低分辨率、宽视角、密集遮挡），以及安全风险偏差普遍倾向于过度报告，导致模型在安全评估中缺乏公平性；

---

## 324. Benchmarking Hybrid Deep Research Across Database Querying and Web Search

**arXiv ID:** 2609.09410 | [PDF](https://arxiv.org/pdf/2609.09410v1)

**作者:** Ruofan Wu `[一作]` (University of Houston), Zhewei Yao `[通讯]` (Snowflake AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了HybridDeepResearch基准，用于评估需要同时进行网页检索和SQL查询的深度研究型代理。

**💡 创新点**

首创三种跨模态推理模式（SQL→检索、检索→SQL、并行交集），并通过“handoff”锁定验证两种工具必须共同完成任务。

**🔧 技术方法**

采用大型语言模型（Qwen3.5、GLM‑5.2、GPT‑5、Claude‑Sonnet‑4.6）与两种代理框架（smolagents、MiroFlow）以及实时网页与固定语料库检索后端进行实验。

**📊 数据集**

构建了380道基于LiveSQLBench‑Base‑Lite九个数据库的任务，并附带公开网络语料（Wikipedia、FineWeb‑10BT）；还筛选出120道均衡的难度子集。

**📈 对比分析**

通过Pass@8和Avg@8指标比较，最强专有模型在难度子集上的Pass@8约为54%，open‑weight模型普遍低于30%；MiroFlow在Pass@8上优于smolagents，但Avg@8提升不明显，方向性推理模式始终是最难的。

**⚠️ 局限性**

局限性在于仅使用九个公开数据库和公共网页，未覆盖专有或专业数据库，三种模式分布不均，且难度子集仍需进一步扩充以提升评测公平性。

---

## 325. Strangers to Themselves: What Language Models Say About Themselves Is Generic

**arXiv ID:** 2609.09899 | [PDF](https://arxiv.org/pdf/2609.09899v1)

**作者:** Phil Blandfort `[一作]` (Predictably Weird), Urja Pawar `[通讯]` (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于预测的自我知识检验框架，系统评估语言模型在不同情境下对自身行为的预测能力，并与其实际表现进行对比。

**💡 创新点**

创新点在于构建了跨模型的行为平均基线、噪声上限和项目化自我报告等多重对照，揭示模型的自我报告往往反映的是对AI助手的一般行为认知，而非特定模型的私有行为信息。

**🔧 技术方法**

主要技术包括 Pearson 相关系数、噪声上限估计、部分相关分析、项目化（item‑informed）和泛化（generic‑subject）提问策略，以及对不同规模和启用推理的模型进行实验。

**📊 数据集**

使用了九个公开行为评估数据集，包括 Demographic Bias、Sycophancy、Capability、Reward Hacking、Misuse & Lying、Agentic Policy Violation、Agentic Misalignment 等。

**📈 对比分析**

通过与行为平均基线和其他模型的自我报告比较，发现即使提供了完整项目信息，模型对自身行为的预测相关系数也仅为 0.2–0.4，远低于理论上可达的噪声上限，说明自我报告效果有限。

**⚠️ 局限性**

局限性包括：只覆盖了九个评估数据集；低成本推理场景；模型池的构成和噪声上限估计对结果影响较大；缺乏对更复杂多轮交互和高成本代理测评的验证。

---

## 326. Kernel-Complexity Edge Sanitization for Training-Free Defense against Structural Graph Attacks

**arXiv ID:** 2609.09698 | [PDF](https://arxiv.org/pdf/2609.09698v1)

**作者:** Yaning Jia `[一作]` (Dartmouth College), Soroush Vosoughi `[通讯]` (Dartmouth College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种训练无关、模型无关的图结构防御框架KCES，通过基于图核复杂度的边评分来筛除对GNN泛化有害的结构攻击边。

**💡 创新点**

创新点在于将图核复杂度（GKC）引入到泛化误差上界，并利用其导出的边KC分数作为理论指导的鲁棒边筛除信号，实现无训练、可并行的预处理防御。

**🔧 技术方法**

主要技术包括图核Gram矩阵构造、图核复杂度计算、基于伪标签的KC分数估计以及高KC边的阈值剪枝。

**📊 数据集**

在多规模公开图数据集Cora、Citeseer、Polblogs、Pubmed、Flickr、Ogbn-Arxiv等上进行实验。

**📈 对比分析**

与多种基线（GCN、GAT、RGCN、ProGNN、GNN-Jaccard等）比较，KCES在大多数结构攻击场景下均超越或等同于现有防御，并在大规模图上保持优越鲁棒性且计算开销低。

**⚠️ 局限性**

局限性主要是只针对结构攻击，无法抵御节点/特征攻击；在异质性或高异类谱图上伪标签可能效果不足，且可能误删有用边。

---

## 327. MethaneFuse: Learning from Multi-Sensor Satellite Observations for Methane Plume Detection

**arXiv ID:** 2609.09762 | [PDF](https://arxiv.org/pdf/2609.09762v1)

**作者:** Yuyao Wang `[一作]` (University of Alberta), Di Niu `[通讯]` (University of Alberta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究从不完整的多传感器卫星观测中检测甲烷羽流，构建MethaneUnion数据集并提出两阶段的MethaneFuse框架；

**💡 创新点**

①构建专门用于部分观测的多传感器时空数据集MethaneUnion；②设计Stage1感知各传感器原生特征并用掩蔽注意力聚合，Stage2利用CLS路由LoRA专家实现轻量化、传感器感知的适配；

**🔧 技术方法**

共享ViT编码器、sensor-native tokenization、掩蔽注意力池化、CLS路由LoRA专家、轻量化参数高效微调、多传感器缺失模态融合技术；

**📊 数据集**

基于Carbon Mapper甲烷羽流报告匹配Sentinel‑2、Landsat 8/9、EMIT、Sentinel‑5P，形成约8,981个多传感器实例的MethaneUnion数据集；

**📈 对比分析**

与独立传感器预测、逻辑/平均/多数投票、SatMAE/AnySat/Panopticon等基线比较；在480 m评估中MethaneFuse实现F1 84.87、AUROC 93.62，比最强基线高5.65 F1、8.30 AUROC、FPR降低8.19；在跨传感器、不同尺度和geo‑cluster测试中均优于基线；

**⚠️ 局限性**

数据依赖Carbon Mapper的报告和匹配，羽流掩模精度有限；S5P仅提供粗略CH4信息；缺乏对时间连续性和不确定性的建模；适配公开卫星，难以覆盖极端云或低质量场景；

---

## 328. Decision-Focused Active Learning for Scale-Aware Critical-Materials Recovery

**arXiv ID:** 2609.09413 | [PDF](https://arxiv.org/pdf/2609.09413v1)

**作者:** Niranjan Srinivas `[一作]` (Coactive Inc.), Elias Nakouzi `[通讯]` (Pacific Northwest National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对稀土元素回收中的选择性沉淀过程进行实验设计与评估，比较主动学习与贝叶斯优化策略在实际实验记录上的表现，并提出基于下游贝叶斯风险的批量选择方法。

**💡 创新点**

创新点在于将实验批量选择与最终工艺决策的下游风险联系起来，提出一种分两阶段、可扩展的策略以及结构化混合代理方法；同时通过真实实验记录构建条件性基准，验证自适应策略在寻找最佳回收条件上的高效性。

**🔧 技术方法**

主要技术包括贝叶斯优化（GP‑UCB）、等价类分辨与条件化优化、知识梯度式价值评估、批量决策优化以及基于贝叶斯风险的目标函数。

**📊 数据集**

使用了PNNL CICERO工作流提供的实验记录数据集，包括NdFeB（Round 1）沉淀路线、SmCo（Rounds 1–2）质量与产率记录、以及产水沉淀实验的原始ICP‑MS与UV‑Vis数据。

**📈 对比分析**

在真实记录的NdFeB条件下，主动学习（两阶段、GP‑UCB）在16–24个实验后即可发现最优富集条件，而空间填充方法需要48个实验；在合成评估中，结构化混合代理与两阶段策略的额外损失均低于约0.05，差异在随机误差范围内。

**⚠️ 局限性**

局限性包括实验记录缺失测量单位、稀释校正和相位信息，缺乏下游工艺损失函数及真实规模转移数据，未提供多重复制以估计实验噪声，且对不同方案的比较仍以模型假设为前提，实际工业实施需要进一步验证。

---

## 329. RouteBridge: Reliability-Routed Bidirectional Distillation Between Neural Radiance Fields and 3D Gaussian Splatting

**arXiv ID:** 2609.09606 | [PDF](https://arxiv.org/pdf/2609.09606v1)

**作者:** YuanHang Wang `[一作]` (University of Technology Sydney), Xin Cao `[通讯]` (University of Technology Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过路由器在光线级别动态选择NeRF与3DGS之间的教师-学生关系，实现两种表示的互相学习；

**💡 创新点**

创新点在于：①基于光线可靠度的双向路由策略；②无共享特征、无额外转换网络的渲染无关对齐接口；③利用颜色、透明度和归一化深度作为几何约束的联合蒸馏；

**🔧 技术方法**

使用独立的Instant‑NGP NeRF和3DGS分支，构造光线可靠度估计（光度残差+几何信号），实现路由决策；随后通过颜色、透明度和深度的双向损失进行蒸馏；

**📊 数据集**

在mip‑NeRF‑360（9个室内/户外场景）和静态三视DTU（15个场景）上进行实验；

**📈 对比分析**

与Instant‑NGP、3DGS、NeRF‑GS等基线对比，mip‑NeRF‑360上NeRF导出PSNR 28.56 dB、3DGS导出PSNR 28.77 dB，DTU上21.12 dB，均显著优于同类方法；

**⚠️ 局限性**

局部可靠度估计为经验式，未做概率校准；双分支训练导致显著内存消耗；当两分支在同一区域错误时，路由器可能无法判定，且缺少所有观测信息时无法恢复结构；

---

## 330. Democracy Needs Reach: Political Equality, Online Speech, and Algorithmic Recommendation

**arXiv ID:** 2609.09465 | [PDF](https://arxiv.org/pdf/2609.09465v1)

**作者:** Etienne Brown `[一作]` `[通讯]` (University of Ottawa), Etienne Brown (University of Ottawa)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出在社交媒体平台中引入推荐底线（保证每个认证账号在一周内至少获得固定数量的政治内容曝光），以实现政治影响机会的平等。

**💡 创新点**

将民主平等（政治影响机会平等）作为算法设计的新价值导向，并提出“推荐底线”作为可行的技术措施，首次将这一理念与实际推荐系统相结合。

**🔧 技术方法**

推荐算法调节（保证最小曝光）

**📊 数据集**

无实验数据集，主要基于已有研究与统计（如 Twitter 关注者分布、Zhu & Lerman 2023 等）

**📈 对比分析**

未进行实验比较；作者论证基于理论分析与文献综述，指出实现后可通过模拟验证。

**⚠️ 局限性**

局限在于缺乏实证支持、实现细节与公平性评估、对广告与商业模式影响、可能的用户代理受限等问题。

---

## 331. Structural Fusion of Bayesian Networks with Limited Treewidth Using Genetic Algorithms

**arXiv ID:** 2609.10276 | [PDF](https://arxiv.org/pdf/2609.10276v1)

**作者:** Pablo Torrijos `[一作]` (Universidad de Castilla-La Mancha), José M. Puerta `[通讯]` (Universidad de Castilla-La Mancha)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一个遗传算法，用于在限定树宽的前提下融合多张贝叶斯网络，生成结构上与原始网络尽量相似且可推理的共识网络。

**💡 创新点**

提出了受限树宽贝叶斯网络融合的定义和基于SMHD距离的目标函数，设计了专门的染色体编码、基于贪婪解的种群初始化、树宽自适应变异以及适应度惩罚机制，从而在融合过程中兼顾结构相似性和计算可行性。

**🔧 技术方法**

采用遗传算法（二进制染色体、锦标赛选择、单点交叉、变异概率自适应）、SMHD距离度量、树宽计算（图形化 + 三角化）、A(G,σ)变换、贪婪算法作为基线，使用Java和Tetrad库实现。

**📊 数据集**

使用合成网络（10、25、50节点，10/20/30个输入BN）和真实网络（Child 20节点、Insurance 27节点），通过对基准网络进行扰动生成多组输入BN进行实验。

**📈 对比分析**

通过比较遗传算法与贪婪算法在不同树宽、种群规模和迭代次数下的SMHD结果。实验表明，遗传算法在树宽限制宽松或高时能得到与无约束融合几乎相同的结构，且显著优于贪婪算法；在大规模/高树宽场景下，改进比率可达0.2（即提升五倍），并在所有配置下保持更低的SMHD。

**⚠️ 局限性**

算法仍受树宽计算成本影响，需预先确定变量顺序σ；实验规模有限，尚未验证对极大规模网络的可扩展性；仅聚焦结构融合，未处理参数学习；在树宽极小或接近完整融合时搜索空间受限，性能提升有限。

---

## 332. Can AI Agents Deliver Verifiable Network-Wide Outcomes Across Authority Boundaries?

**arXiv ID:** 2609.10181 | [PDF](https://arxiv.org/pdf/2609.10181v1)

**作者:** Tianzhu Zhang `[一作]` (Nokia Bell Labs), Meikang Qiu `[通讯]` (Augusta University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一个可信运行时保证层，用于在多权威范围的 AI 网络代理协调完成后，收集并验证观测证据，确保网络操作真正实现运维人员的意图。

**💡 创新点**

创新点在于：①将完成确认拆分为：受控观测收集、证据绑定与新鲜度校验、确定性规则检查以及 LLM 验证器评估；②引入“完成合同”机制，明确各观测原点和所需属性；③设计了可信的 Broker、Epoch Ledger、Submission Gate 等组件，保证证据来源和时效；④支持一轮自动修复，提升自主恢复能力。

**🔧 技术方法**

技术方面使用了：LLM 驱动的代理（基于 GPT‑5 系列）、网络操作中介器（Scope Wrapper）、证据 Broker、Epoch Ledger、可编程的 deterministic 规则引擎、Verifier Agent（LLM）、容器化实验环境 Containerlab 以及 FRRouting 软件。

**📊 数据集**

实验数据集为 NetAgentBench 中自定义的 6 个任务（包含 OSPF 对等、BGP 传播/过滤、路由可达性等），每个任务生成 15 条执行轨迹，总共 45 条；此外还通过人为注入故障和干预实验来评估可靠性。

**📈 对比分析**

对比方法：将仅基于执行记录的 Action‑Derived Fulfillment Claims（Baseline）与完整证据收集的 CA 进行对比。结果显示，CA 在 32 条成功轨迹上被正确接受，而 Baseline 在 13 条失败轨迹中错误接受，证明后续观测的价值；Q2 的干预实验进一步证明了来源绑定和新鲜度检查能成功拦截错误证据；在 Q4 中，允许一轮修复后所有故障都能恢复，体现了系统的恢复能力。

**⚠️ 局限性**

局限性包括：①信任边界假设（Orchestrator、Broker、Wrapper 等未被破坏），无法防御直接绕过中介的变更；②每个权威范围仅对应单个路由器，未覆盖更复杂的管理域；③采用全局 epoch 策略过于保守，导致不相关证据被错误地视为过期；④观测收集顺序性导致未形成原子快照；⑤完成合同只覆盖已声明的属性，缺失要求可能被忽略；⑥修复仅限一次且对所有路由器执行，缺乏最小化和分阶段部署；⑦实验规模有限，未覆盖多厂商、多层网络及对抗性场景。

---

## 333. How Fragile Is Safety Alignment at Frontier Scale? A Single-Direction Attack on a 320B MoE

**arXiv ID:** 2609.09793 | [PDF](https://arxiv.org/pdf/2609.09793v1)

**作者:** Yi Shi `[一作]` (Continuum AI), Kai Shen `[通讯]` (Continuum AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 GLM‑5.3‑Flash 320B 混合专家模型使用单一方向的权重正交化（directional ablation）进行干预，消除其拒绝有害请求的行为，同时保持大部分能力。

**💡 创新点**

在面向 320B 级稀疏 MoE、四流残差连接与 block‑FP8 量化的模型上验证并改进了原始单方向消除方法，展示其在前沿架构下的可行性与局限性，并揭示拒绝行为在模型内部的分布与非加性特征。

**🔧 技术方法**

采用差异均值求取拒绝方向、一次性低秩（rank‑1）正交化、跨张量的专家权重展开、mHC（多流残差）混合矩阵处理、对 block‑FP8 代码进行迭代正交化并重新量化。

**📊 数据集**

利用 AdvBench、JailbreakBench、StrongREJECT、HarmBench、MaliciousInstruct、ForbiddenQuestions、SimpleSafetyTests 等七个有害评测集，以及 XSTest 进行过拒绝率对比；MMLU、MMLU‑Pro、CMMLU、GSM8K 四个能力基准用于验证干预对能力的影响。

**📈 对比分析**

对比未干预模型，单方向消除实现了 41–89 百分点的拒绝率下降（大部分 benchmark 下降 70% 以上），且对四个能力指标的影响不超过 1 分；同时发现干预效果在不同 writer 组（attention、dense、expert）之间高度非加性，单独编辑任一组仅能移除少量拒绝。对比多流残差与单流残差，写入层级的正交化优于仅在层边界投影。

**⚠️ 局限性**

局限性包括：仅在单一 GLM‑5.3‑Flash checkpoint 上实验；拒绝判定依赖自动 judge 模型，可能与人工评估不一致；能力评测采样有限；对随机方向的控制不足；对不同规模或架构的泛化未验证；仅给出 32 次迭代正交化经验值，未做理论收敛分析；未公开完整实现细节，复制成本较高。

---

## 334. Fine-Tuning a KV Cache Concatenation-Aware Model or Recomputing KV Caches? Why Not Both?

**arXiv ID:** 2609.09768 | [PDF](https://arxiv.org/pdf/2609.09768v1)

**作者:** Fumihiko Tachibana `[一作]`, Jun Deguchi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

结合 KV 缓存拼接意识的模型微调与对部分 KV 缓存的选择性重计算，在长上下文（124k token）语境下提升生成质量并显著缩短首个 token 的延迟。

**💡 创新点**

提出将 KV 缓存拼接与模型微调相结合的双重策略，二者互补降低 key 误差，从而在极长上下文中保持高准确率。

**🔧 技术方法**

使用 Block‑attention 微调、CacheBlend 选择性重计算、FlashAttention‑2、FlashInfer、LMCache 等技术；实现 KV 缓存分层并行加载与重计算。

**📊 数据集**

Llama3.1‑8B‑Instruct 与 Qwen2.5‑7B‑Instruct 作为基线模型；RULER benchmark、4K‑124K token 语料、TriviaQA、HQA、NQ、2Wiki 等。

**📈 对比分析**

与全注意力、普通微调、仅 CacheBlend、仅 Block‑attention 等做对比；在 124k token 上 Block‑attention+CacheBlend 提升 RULER 得分 9.7 点，TTFT 降低 80%，在多项 RAG 与通用任务中均优于单一策略。

**⚠️ 局限性**

对选择性重计算的注意力实现速度有限；未评估基于注意力得分的重计算方法（A^3、KVShare）和 Link0+APE 组合；未对思考型模型进行微调。

---

## 335. Somatosensory Activation and Attentional States in Creative Making

**arXiv ID:** 2609.09960 | [PDF](https://arxiv.org/pdf/2609.09960v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 336. AgentAudit: An Open, Extensible Framework for Full-Lifecycle Trust Evaluation of AI Agents

**arXiv ID:** 2609.09875 | [PDF](https://arxiv.org/pdf/2609.09875v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 337. Why Sample What You Can Enumerate? Exact Policy Optimization for Genomic Tool Selection

**arXiv ID:** 2609.10221 | [PDF](https://arxiv.org/pdf/2609.10221v1)

**作者:** Haoyue Liu `[一作]` (Chinese University of Hong Kong), Xiaoying Tang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在可枚举工具子集空间上进行精确策略优化的框架（Exact Policy Optimization, EPO），用来替代传统采样的GRPO，以改进基因组推理中的工具调用策略。

**💡 创新点**

创新点在于：①识别并解决采样优化与可枚举工具空间不匹配导致的优势消失和信息稀缺问题；②设计了全枚举奖励表和精确期望优化的目标，使每次更新都能覆盖整个动作空间；③通过长度归一化和熵正则化实现对LLM策略的精细控制。

**🔧 技术方法**

技术手段包括：使用LoRA微调的7B LLM生成工具子集；对生成的工具子集字符串做每token长度归一化评分；熵正则化提升候选覆盖率；构建完整的奖励表（预计算每个问题-子集对的奖励）；对比采样群归一化优势的GRPO；在实验中使用四种基因组工具（序列组成、Motif扫描、剪接分析、kNN专家）。

**📊 数据集**

数据集：三套基因组多项选择问答基准（GenomeQA、GenBench‑X、BM4），共计约3,590+1,000+1,000测试题；训练集为2,002个基因组QA问题；工具库包含4个基因组工具，所有子集可枚举（2^4=16）。

**📈 对比分析**

与随机、All‑Tools、Tools w/ Desc、SFT、DPO、GRPO等基线在五个冻结推理器（Qwen3‑8B、Qwen2.5‑1.5B、Qwen2.5‑7B、Mistral‑7B、InternLM2.5‑7B）上进行交叉测试。EPO 在所有15个设置上平均提升 6.75 分（最高 14.20 分），同时每题调用工具平均从 2.36 降至 1.40，且在冻结推理器预算下只需 2.4 倍少的奖励评估。

**⚠️ 局限性**

局限性：仅适用于工具数较小、可枚举的空间；对更大工具库需要构造自适应候选集；实验集中在基因组推理领域，需进一步验证在其他专业领域的迁移性。

---

## 338. HELIOS: Guardrailed LLM-Driven Evolution of Autonomous Resource Orchestration Policies for Multi-Cloud Distributed Systems

**arXiv ID:** 2609.09164 | [PDF](https://arxiv.org/pdf/2609.09164v1)

**作者:** Guanyu Ding `[一作]` (New York University), Ying Wang `[通讯]` (Pepperdine University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了HELIOS系统，利用大语言模型离线演化生成可执行的多云资源调度策略，并在生产环境中通过安全 guardrail 层执行；

**💡 创新点**

核心创新在于将LLM从高频决策路径移除，改为离线的演化程序合成；结合约束安全层确保任何生成策略的可控性；通过真实工作负载与价格数据的 trace‑driven 仿真，验证了成本与可靠性的显著提升；

**🔧 技术方法**

采用 Claude Sonnet 进行程序生成；演化式程序合成（Mutation/跨代交叉）；基于真实云价、停机概率的多云模拟器；guardrail 机制实现五项安全约束；对比基线采用 MILP 规划与 DQN 强化学习；

**📊 数据集**

使用 PlanetLab、Azure 与 Bitbrains 的真实计算/内存请求轨迹；2026 年 AWS、Azure、GCP 的 on‑demand 与 spot 价格及中断频率；SkyPilot 交叉云价格目录；CPU 轨迹补齐内存需求；

**📈 对比分析**

对比单云 BFD、阈值 HPA、贪婪多云、手写 seed、MILP 规划与 DQN meta；HELIOS 在所有三套保留轨迹上实现了最高的惩罚成本（相对单云基线低 45%，对贪婪多云低 9–19%，对 MILP 低 40%），同时维持或降低 SLO 违约率；guardrail 的缺失会导致 97–98% 的高级服务停机；

**⚠️ 局限性**

主要局限：仅在仿真环境下验证，迁移/启动模型过于简化；使用单一 2026‑07‑03 的价格快照，未覆盖价格动态；CPU 轨迹的内存需求为合成；未考虑网络延迟、GPU、跨区放置等；LLM 生成过程由人工监督，搜索预算有限；缺乏持续演化与实际部署的长期稳定性评估。

---

## 339. $S^3$-Bench: Evaluating Speech Interaction Models as Scientific Voice Assistants

**arXiv ID:** 2609.09852 | [PDF](https://arxiv.org/pdf/2609.09852v1)

**作者:** Heyang Liu `[一作]` (Shanghai Jiao Tong University), Yu Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了S^3-Bench评估框架，覆盖10个科学学科的知识问答与多轮对话，用于评估语音交互模型在科学领域的表现。

**💡 创新点**

创新点在于：①构建专业问答与动态多轮对话数据集；②将语音交互拆分为识别、感知、知识推理与生成四阶段，揭示生成阶段与前置阶段的负向耦合；③采用LLM-as-Judge多维度评估。

**🔧 技术方法**

采用大型语言模型（Qwen、Gemini、GPT-5.6 Luna等）、ASR/TTS系统、Cascade 端到端流水线、专业词表与发音校正、LLM-评判器等技术。

**📊 数据集**

数据来源于17个公开科学问答基准、Common Voice 语音库、YouTube 语音片段以及学术论文集合，最终得到约1,980条语音问答和213条多轮对话。

**📈 对比分析**

通过对比6款端到端语音LLM和2款Omni-LLM，使用WER/EER/AER、相关率(RR)等指标；在S^3-Knowledge中最佳端到端模型准确率≈60%，在S^3-Dialogue中最强模型在松散标准下科学事实性≈94%，严格标准约51%。

**⚠️ 局限性**

局限性包括：生成阶段的发音错误导致整体质量下降；严格事实性仍低，缺乏多轮对话中的事实一致性；受众适应率仅≈50%，无法充分根据用户背景调整解释；整体依赖高质量语音-文本对齐，仍需提升。

---

## 340. SwingBot: Learning Whole-Body Brachiation for Humanoid Robots

**arXiv ID:** 2609.10283 | [PDF](https://arxiv.org/pdf/2609.10283v1)

**作者:** Yujie Xiong `[一作]` (Fudan University), Lihua Zhang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了SwingBot框架，使得高自由度人形机器人能够在没有外部感知的情况下完成连续的肩部摆动（Brachiation）运动，并在实际机器人上演示了多段连贯攀爬、负载携带以及抗干扰能力。

**💡 创新点**

创新点包括：① 利用稀疏的生物启发关键帧作为训练支架，指导机器人在长时程的释放‑摆动‑捕获序列中学习；② 通过RSSM（Recurrent State‑Space Model）构建特权信息的递归隐状态，用于估计机器人在执行过程中难以直接观测的段内位移和抓取状态；③ 将上述两项技术结合，实现在真实机器人上实现连续、鲁棒的高自由度摆动。

**🔧 技术方法**

技术手段包括：PPO强化学习、残差关键帧引导、RSSM特权状态估计、低通滤波与低阶PD控制、适应性采样、以及仿真到实机的模拟‑实机迁移方法。

**📊 数据集**

数据集方面：未使用公开数据集，全部在仿真环境中采集训练数据（包含不同条形间距和命令切换的连续轨迹），随后在真实机器人上进行5条连续试验验证。

**📈 对比分析**

比较方法：对比不使用关键帧引导、不使用RSSM估计以及教师策略（拥有真实特权变量）的变体。在仿真中，所有8段成功率从无关键帧（0%）提升到有关键帧+RSSM（约61%），教师策略最高约65%；在真实机器人上，单段成功率达到约93%–94%，所有8段完成率在正常、负载、扰动和不同条形间距下分别为60%、40%、40%和60%，说明方法在不同条件下保持了较高的鲁棒性。

**⚠️ 局限性**

局限性：① 仅靠惯性测量，缺乏外部感知，条形必须在训练时覆盖的间距范围内；② 连续攀爬会导致关节过热，降低可用扭矩，影响长时间运行；③ 仅使用被动钩子，手部抓取精度有限，未来需要加入视觉感知和更灵活的抓取器。

---

## 341. Fast Constraint Extraction for Corrective Control under STL Specifications via Logical Dependency Tracking

**arXiv ID:** 2609.09439 | [PDF](https://arxiv.org/pdf/2609.09439v1)

**作者:** Antoine Besset `[一作]` (Institut Polytechnique de Paris), Julien Alexandre dit Sandretto `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种逻辑依赖跟踪框架，能够在不确定性下快速提取保证 STL 规范满足的最小足够约束，并将其转化为基于zonotope可达性分析与线性规划的控制纠正问题；

**💡 创新点**

创新点在于将三值 STL 监测与标记传播相结合，利用逻辑依赖生成毫秒级的 DNF 约束，从而实现对不确定性下 STL 满足的正式保证并提供纠正控制方案；

**🔧 技术方法**

采用三值 STL 语义、标记传播与逻辑依赖跟踪、DNF 生成、zonotope 可达性分析、线性规划控制修正，以及概率下界证明等技术；

**📊 数据集**

实验使用了一个 Dubins‑风格的非线性车辆模型，采用合成的区间与高斯参数分布作为不确定性来源；

**📈 对比分析**

与 All‑SAT/枚举标记约束提取方法比较，本文方法在单个约束识别仅需 1–3 ms，整体纠正时间保持在几十毫秒，显著快于基线且保持形式化保证；

**⚠️ 局限性**

主要局限在于可达性分析仍需数秒，导致整体计算受限；此外获得的控制修正并非全局最优，可能产生冗余或次优纠正；

---

## 342. Future-Aware Flow Planning for Safe UAV Target Following

**arXiv ID:** 2609.10166 | [PDF](https://arxiv.org/pdf/2609.10166v1)

**作者:** Boning Feng `[一作]` (Stockholm University), Xiaodan Shi `[通讯]` (Stockholm University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于未来感知流规划的无人机目标跟踪框架，利用预测目标未来作为残差指导生成安全可执行轨迹，并在采样循环中嵌入风险评分前缀修复。

**💡 创新点**

创新点在于将目标预测与无人机轨迹生成分离：目标未来以门控残差方式注入条件流生成器，避免直接硬引用路径；同时在采样过程中嵌入可执行前缀安全修复，使前缀修正能即时影响后续采样，从而实现更稳健的跟踪-安全权衡。

**🔧 技术方法**

采用的技术包括Mamba时序网络做目标预测、条件流匹配器（Flow Transformer）配合门控未来适配器、RSEPSS风险评分前缀修复、以及与Isaac Sim、PX4 SITL等仿真平台的接口。

**📊 数据集**

使用了合成森林障碍物环境的ID/OOD基准数据集（共150个ID、200个OOD场景），以及通过Isaac Sim、Pegasus与PX4 SITL实现的飞行控制器仿真数据。

**📈 对比分析**

与SafeFlow、SafeFlowMatcher以及手工设计的Future‑MPC进行对比，实验表明该方法在ID场景实现零碰撞率、最高安全跟踪时间（STT@8≈0.96），OOD场景碰撞率最低且平均终点误差最佳；在飞行控制器循环验证中保持了良好的跟踪误差（ID 1.84 m，OOD 2.02 m）且无硬碰撞。

**⚠️ 局限性**

局限性包括依赖离散目标状态与局部障碍输入，未覆盖端到端视觉感知、动态障碍物或硬件气动特性；RSEPSS增加了重新规划成本，需进一步轻量化；仿真验证尚未覆盖真实飞行测试。

---

## 343. What Makes Adversarial Examples Transfer Across Deepfake Detectors?

**arXiv ID:** 2609.10002 | [PDF](https://arxiv.org/pdf/2609.10002v1)

**作者:** Rafael M. Mamede `[一作]` (INESC TEC), Ana F. Sequeira `[通讯]` (INESC TEC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

系统评估了60种不同配置的深度伪造检测器在黑盒对抗攻击（AutoAttack和CW‑EOT）下的迁移性能，构建了3,540个源–目标对，生成并分析了240,000张对抗图像。

**💡 创新点**

创新点在于：① 对源与目标的兼容性因素（骨干网络、架构家族、预训练方案、训练数据）进行因子化、系统化评估；② 发现不同攻击的迁移性受不同兼容性因素主导；③ 提出自我排除和严格的oracle评估，揭示单源评估低估了目标脆弱性。

**🔧 技术方法**

采用AutoAttack和CW‑EOT两种对抗攻击算法，配合分层对比统计和Jackknife估计进行转移性分析，并使用自我排除/严格oracle协议评估目标鲁棒性。

**📊 数据集**

使用DeepfakeBench DF40数据集（包含FF++、Celeb‑DF、DFDC等子集）进行训练与测试，随机挑选2,000张伪造图像用于对抗攻击。

**📈 对比分析**

通过攻击成功率(ASR)比较单源、平均源以及oracle的性能：单源平均ASR仅为AA 7.21%、CW‑EOT 19.52%；自我排除oracle ASR平均约92%；严格oracle平均ASR高达64.48%；并发现AA与CW‑EOT在迁移性影响因素上差异显著。

**⚠️ 局限性**

局限性在于：仅评估了6种骨干、2种预训练、5种训练子集，未考虑训练随机性与防御模型；所用的攻击与数据集限制了结果的普适性；缺乏对其他攻击方式和更大规模模型的验证。

---

## 344. Scalable Oversight for AI in Mental Health: Lessons from 350,000 AI Coaching Conversations between Therapy Sessions

**arXiv ID:** 2609.09533 | [PDF](https://arxiv.org/pdf/2609.09533v1)

**作者:** Matthew A. Scult `[一作]` (Grow Therapy), Manoj Kanagaraj `[通讯]` (Grow Therapy)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并在实际产品中部署了三层“人类在环”监督框架，结合预防性设计、实时安全监测与持续质量评估，以实现对心理健康AI工具的大规模、安全、可持续监管。

**💡 创新点**

创新点在于把传统的“人类在环”(human‑in‑the‑loop)转化为“人类在环”(human‑on‑the‑loop)，通过将人类专业判断集中在系统级指标、异常检测与持续反馈循环中，而非逐条审阅；同时融合了多模态评估（自动评估、临床评审、A/B测试、红队攻击）形成闭环改进。

**🔧 技术方法**

使用技术包括：1) 预训练LLM进行系统提示与对话生成；2) LLM‑as‑judge（独立评估模型）对每次对话进行多维度评分；3) 并行安全LLM实时检测风险信号并触发分级响应；4) 自动化benchmark与红队测试框架；5) 结构化QA工具与人类临床审核；6) 数据管道与A/B实验平台。

**📊 数据集**

主要数据集为：① Grow Therapy网络中350,000+次真实对话（352,649个会话）；② 内部模拟benchmark对话（涵盖Cognitive Behavioral Therapy、Dialectical Behavior Therapy、Acceptance & Commitment Therapy等多种情境）；③ 红队攻击用例与安全测试集；④ A/B测试对照版本的用户情绪与目标进展数据。

**📈 对比分析**

方法对比：将新版本与现有版本进行A/B测试，比较综合质量分数、自动安全评估得分、情绪转变率（61.9%正向）与动机提升率（64.9%）。此外，通过自动评估筛选出的低质量会话与人类审核结果的匹配度约为4倍于未标记会话，验证了自动评估的有效性。安全阈值调优后，误触发率显著下降，而对真实危机信号的检测保持不变。

**⚠️ 局限性**

局限性包括：① 仅在单一产品和单一运营商（Grow Therapy）内验证，缺乏跨机构或跨文化的外部验证；② 依赖内部生成的benchmark和红队测试，可能未覆盖所有潜在攻击路径；③ 自动评估模型的准确性仍受训练数据和评判尺度的影响，存在误判风险；④ 仍需进一步评估长周期的用户依赖性与伦理影响。

---

## 345. A Bio-Plausible Visual Neural Network for Locust-Inspired Collision Perception

**arXiv ID:** 2609.10183 | [PDF](https://arxiv.org/pdf/2609.10183v1)

**作者:** Qinbing Fu `[一作]` (Guangzhou University), Jigen Peng `[通讯]` (Guangzhou University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于蜻蜓眼结构的生物学可解释视觉神经网络，用于检测逼近物体并做出碰撞决策。

**💡 创新点**

创新点包括：① 采用六边形等距采样模拟蜻蜓复眼；② 引入基于群体投票的分布式响应整合；③ 用泄漏积分与发射（LIF）神经元替代传统sigmoid输出，提升时序生物学真实性；④ 结合自抑制、侧抑制等多重抑制机制实现更高的逼近选择性。

**🔧 技术方法**

技术细节包括：ON/OFF对比通道、可抑制的前向、侧向和自抑制权重、卷积滤波器、群体投票权重、LIF膜电位动力学以及噪声鲁棒性分析。

**📊 数据集**

实验使用的数据集包括：人工合成的扩张/平移刺激、实验室捕捉的球体逼近与平移序列，以及194条真实驾驶仪表盘视频（42条碰撞、152条非碰撞）。

**📈 对比分析**

与传统LGMD模型以及双线性/高斯下采样方法对比，模型在六边形采样下保持更高的结构保真度，群体投票显著降低噪声影响，最终在真实视频上实现76.29%准确率、83.33%召回率、60.35% F1分数，优于仅使用单一输出单元的模型。

**⚠️ 局限性**

局限性在于：在复杂场景下仍出现较高的误报率（Precision 47.30%），主要受背景运动、摄像机抖动影响；模型对数据不平衡（碰撞样本稀缺）仍敏感，需要进一步改进召回与精确度的平衡。

---

## 346. Cost-Aware Post-Hoc Deferral Under Calibration and Shift: An Environmental AI Case Study

**arXiv ID:** 2609.09235 | [PDF](https://arxiv.org/pdf/2609.09235v1)

**作者:** Haoran Yu `[一作]` (University of Florida), Danping Zhang `[通讯]` (Nanchang Hangkong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种后置学习决策框架 EcoTrust，用于在冻结预测器下，根据不确定性信号、异步成本、评审准确率以及检测到的分布漂移，决定自动决策还是人工评审；

**💡 创新点**

创新点在于将多组不确定性信号（置信度、熵、集成方差、冲突、共形预测、马氏距离）整合到交叉拟合的元风险估计器中，同时引入检测外推的支持门，并以成本感知的阈值直接比较自动、评审与回退三种决策，形成可审计的部署协议；

**🔧 技术方法**

技术包括交叉拟合的逻辑回归元风险估计器、Platt 校准、马氏距离支持门、成本感知的决策阈值、基于时间块的离线评估与bootstrap检验；

**📊 数据集**

使用了美国国立海洋大气管理局（NOAA）与美国地质调查局（USGS）在科伦比亚河（The Dalles）1996–2024年的气候与水文观测数据，构造每日热胁迫标签（水温>18°C），并在10个河站进行迁移实验；

**📈 对比分析**

通过与Chow拒绝法、手工启发式、随机评审、选择性分类、传统L2D等基线在同一冻结预测器上进行比较；在分布内，Chow在成本上占优；在多种后置风险估计器中，EcoTrust 在12个后台中6个表现最佳；在迁移实验中，支持门退化为“始终评审”回退，未能显著提升成本；

**⚠️ 局限性**

局限包括：元风险估计器仅用约1,200条样本，难以精准估计极低错误概率；支持门仅检测协变量漂移，无法处理概念漂移；评审错误假设为均匀且独立，未考虑实际专家差异；在迁移站点中未获得真实标签和评审反馈，导致回退策略可能过度保守；

---

## 347. Automatic Reproducible Camera Intrinsic Calibration

**arXiv ID:** 2609.10082 | [PDF](https://arxiv.org/pdf/2609.10082v1)

**作者:** Xiangcheng Hu `[一作]` `[通讯]` (Hong Kong University of Science and Technology), Xiangcheng Hu (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套完全自动化的相机内参标定管线，能够根据采集数据自动筛选合适的图像子集并决定径向畸变模型的阶数。

**💡 创新点**

创新点包括：①基于中位数缩放阈值的迭代剔除机制，使图像筛选与畸变阶数解耦；②采用仅重新估计位姿的验证集来判定畸变阶数，避免了对标定集残差的过拟合；③将上述两步集成到可交互的工具中，保证决策的透明可复现。

**🔧 技术方法**

技术手段主要涉及Brown‑Conrady径向畸变模型、基于中位数的残差尺度估计、迭代外点剔除、POSE‑ONLY验证（PnP重估）、自适应阈值阈值τ=κ·median，κ=2；实现使用MATLAB/Python实现，接口可通过脚本调用。

**📊 数据集**

使用的数据集包括作者自采的Rig‑A（39张手持摄像图）和Rig‑B（39张），以及公开的OpenCalib‑F、ROS stereo、OpenCV等四种相机模型，共计七个数据集，覆盖62°–111°视角和0.3–2.3 Mpx像素。

**📈 对比分析**

与ROS‑calibrator、mrcal（默认8‑系数模型）以及不做图像筛选/阶数选择的基线方法相比，本文管线在所有五个公共数据集上均取得最低的留存集平均重投影误差，图像筛选可降低至多25%，阶数选择进一步提升5%，总体误差下降显著。

**⚠️ 局限性**

局限性在于仅考虑径向畸变的两阶/三阶模型，未扩展至 rational/fisheye 家族；筛选标准仍基于残差统计，未加入几何覆盖度约束；在视角更窄、图像数不足 N_min 的场景下，阶数选择功能无法激活。

---

## 348. FPGA Acceleration of Fully Homomorphic Encryption with Adaptive Key Switching

**arXiv ID:** 2609.09423 | [PDF](https://arxiv.org/pdf/2609.09423v1)

**作者:** Zhihan Xu `[一作]` (University of Southern California), Viktor K. Prasanna `[通讯]` (University of Southern California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了可在FPGA上同时支持传统混合密钥切换（HKS）和新型KLSS的自适应加速器，并通过动态选择最优方法降低全FHE工作负载的延迟。

**💡 创新点**

①设计了内存高效的KLSS数据通路，避免离芯片中间密文传输；②建立了性能模型指导在不同Ciphertext层级、密钥参数和硬件并行度下动态切换HKS/ KLSS；③实现了能在Alveo U280上实现1.84–3.31×加速bootstrapping、1.66–2.52×加速安全图像分类的自适应加速器。

**🔧 技术方法**

FPGA硬件架构（可变并行度计算阵列、流式置换网络、跨银行scratchpad）、Barrett模乘、Solinas模约简、双缓冲与数据通路调度、性能模型映射到时钟周期。

**📊 数据集**

使用公开的FHE参数集（Set‑1 与 Set‑2）以及典型工作负载：LoLA‑MNIST、ResNet‑20（Cifar‑10）以及完整的bootstrapping流程。

**📈 对比分析**

与多种先前FPGA实现（FAB、Poseidon、DAHE、OLA）以及GPU基准（V100 100x）比较；在所有评测中，HKS/ KLSS 组合下的自适应加速器在关键操作和完整工作负载中均实现显著加速，平均提升约 2–3 倍。

**⚠️ 局限性**

仍受限于FPGA片上存储容量；KLSS在小Ciphertext层级时效率低于HKS，需动态决策；模型对不同安全参数的准确性和实现细节的硬件映射仍需进一步验证。

---

## 349. An Efficient and Effective Agentic Group Shilling Attack on Recommender Systems

**arXiv ID:** 2609.09551 | [PDF](https://arxiv.org/pdf/2609.09551v1)

**作者:** Quoc Viet Nguyen `[一作]` (Griffith University), Thanh Tam Nguyen `[通讯]` (Griffith University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a2602d71-93ab-4bad-974b-672788df8193` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了AGAS，一种通过协调工作者角色切换、动态策略的黑盒代理团体欺骗攻击框架，以提升目标物品在推荐系统中的曝光。

**💡 创新点**

创新点在于：① 无需离线训练的协调器；② 通过角色切换打破固定模式；③ 根据受害者反馈实时调整策略。

**🔧 技术方法**

主要技术：大型语言模型（LLM）驱动的代理，ReAct/Reflexion循环，动态信号采集与策略更新；无监督反馈。

**📊 数据集**

使用了公开推荐数据集：MovieLens-100K/1M、Genome 2021、Netflix、Douban、Amazon Reviews 2018。

**📈 对比分析**

与现有基线（Random、Bandwagon、AUSH、AgentSA/Attack等）在HR@K、NDCG@K、Rec@K等指标上进行比较，AGAS在目标推广效果上优于所有基线，同时保持更高的推荐质量与更低的检测率。

**⚠️ 局限性**

局限性包括：仅针对隐式CF模型验证，未覆盖多模态或LLM驱动的推荐器；依赖于代理可访问的反馈；对攻击者硬件/算力需求仍较高。

---

## 350. Putting Captions to the Test: Evaluating Video Caption Quality through Multiple-Choice Question Answering

**arXiv ID:** 2609.09973 | [PDF](https://arxiv.org/pdf/2609.09973v1)

**作者:** Zizhen Wang `[一作]` (Apple), Xiaoming Simon Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个无参考的视频字幕评估基准，利用人类核验的细粒度多选问答来衡量字幕的事实性和覆盖度，并给出了三种指标（Factuality、Coverage、Overall）。

**💡 创新点**

创新点在于①基于信息保真度的参考无关评估方法；②通过多选问答和层级问答类型实现对细粒度视觉理解的诊断；③将评估拆解为事实性与覆盖度，使用调和平均构造 Overall；④显著提升与人工评判的一致性，揭示当前 VLLM 在事实性高但覆盖度不足、推理能力弱的瓶颈。

**🔧 技术方法**

技术手段包括：使用 LLM（如 GPT‑4o、Gemini‑2.5）作为判定器；自动生成 + 过滤 + 人工审核的过采样 QA 生成 pipeline；盲解答检验、语义去重、Hard Negative 筛选；以及基于 TP/FP/FN 计算 Factuality、Coverage、Overall 的公式。

**📊 数据集**

数据集覆盖 24 个视频域、约 10,000 条视频，来自公开数据集（VideoMME、VideoChatGPT、NextQA、MVBench、MMBench‑Video、CVRR、PerceptionTest、longvideobench、VDC、Dream1K）。每条视频平均提供 15–20 条人类核验的多选问答。

**📈 对比分析**

与传统 n‑gram/语义匹配（BLEU、CIDEr、CLIPScore）及现有 QA‑based（VDC、VCapsBench）方法相比，所提评估在 Spearman 和 Kendall 与人工评分的相关性均更高（最高 ρ≈0.69），并在多种 VLLM 上实现了客观的细粒度诊断：SOTA GPT‑5.2 取得 Overall ≈83.08，事实性高但覆盖度低；推理题目的性能明显落后于描述性题目。

**⚠️ 局限性**

局限性包括：QA 可能未覆盖视频全部细节，覆盖度指标是基于已识别的关键信息；仅提供英文问答，缺乏多语言支持；评判依赖 LLM API，可能受模型更新和偏见影响；以及手工审核成本高、难以完全消除噪声。

---

## 351. From Few-Shot Segmentation to Clinician-in-the-Loop Medical Image Analysis

**arXiv ID:** 2609.10001 | [PDF](https://arxiv.org/pdf/2609.10001v1)

**作者:** Yazhou Zhu `[一作]` `[通讯]` (Independent Researcher), Yazhou Zhu (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出基于有限支持集、跨域和交互式反馈的医学图像分割框架，统一为序列决策问题；

**💡 创新点**

创新点在于把任务定义、查询策略、预算控制和受限适配融合成可解释的动态价值决策流程；

**🔧 技术方法**

采用原型分割、结构化传输、动态语义匹配、SAM提示、主动学习与控制论决策等技术；

**📊 数据集**

使用多机构MRI/CT公开数据集（不限定具体数据集），构造跨域、稀缺、模糊样本；

**📈 对比分析**

与原型网络、跨域适配、交互式分割等基线对比，受限适配后在稀缺/跨域场景下Dice提升约2–5%，查询成本降低；

**⚠️ 局限性**

局限在于缺乏真实临床验证、对不确定性假设高度敏感、需要大量工程实现安全门控和长期学习治理。

---

## 352. Cross-Species Animal Re-Identification with Semantic Consistency Learning

**arXiv ID:** 2609.09705 | [PDF](https://arxiv.org/pdf/2609.09705v1)

**作者:** Shuoyi Chen `[一作]` (Wuhan University), Mang Ye `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在跨物种动物身份识别任务中提出了一种语义一致性学习框架（SCL），通过前景背景分离的频域归一化和跨物种邻域建模实现对多种动物的统一表征学习。

**💡 创新点**

创新点在于①前景-背景分离频域归一化（FDSNorm）能够在保持结构信息的同时抑制环境风格差异；②跨物种邻域建模（CNM）利用动态邻居和记忆队列，强化跨物种语义关联而非仅仅依赖实例级判别。

**🔧 技术方法**

使用的技术包括Vision Transformer基础网络、频域归一化、Teacher‑Student EMA、记忆队列、互最近邻检索、边际约束等。

**📊 数据集**

使用了11个公开动物ReID数据集（Wildlife71、PetFace、iPanda‑50、ELPephants、SealID、GZGC、ATRW、HyenaID2022、LeopardID2022、SeaTurtleID2022、WhaleSharkID）进行评估。

**📈 对比分析**

与多类前沿方法（TransReID、Meta、CLIP、UniReID、AdaFreq、Megadescriptor、MiewID等）在两种跨物种评估协议上进行对比，SCL在Rank‑1和mAP上均显著领先，提升幅度约5‑10%。

**⚠️ 局限性**

局限性包括对背景信息仍有一定依赖，未完全覆盖完全开集场景；需要大量多物种数据进行训练；在极端环境或极少样本物种上的效果仍需验证。

---

## 353. Channel Estimation for OFDM via Delay-Doppler Refinement

**arXiv ID:** 2609.09782 | [PDF](https://arxiv.org/pdf/2609.09782v1)

**作者:** Mingcheng Nie `[一作]` (University of Sydney), Yonghui Li `[通讯]` (University of Sydney)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种利用延迟-多普勒域（DD域）精炼的OFDM信道估计（CE）方案，先在时频域做粗略LMMSE估计，再将估计转至DD域进行阈值化和相位补偿，最终重构时频域信道矩阵；

**💡 创新点**

创新点在于：1) 将传统OFDM在时频域的粗估与DD域的精细化结合，充分利用DD域的稀疏性与相位聚合特性；2) 通过阈值化与多观测平均显著提升估计精度；3) 兼顾现有OFDM帧结构，无需改动Pilot安排；

**🔧 技术方法**

采用全尺寸LMMSE估计、对称有限傅里叶变换（SFFT）、阈值化、相位补偿、观测平均与重构技术；

**📊 数据集**

使用仿真数据：M=N=16的OFDM系统，3条独立路径，延迟/多普勒均匀抽样（l∈[0,2], k∈[-3,3]），QPSK数据符号和全1的Pilot；

**📈 对比分析**

与ST-LS、ST-LMMSE+插值、全尺寸LMMSE等基准方法比较，结果显示所提CE在所有SNR下NMSE明显低于基准，且不存在饱和误差；

**⚠️ 局限性**

局限性包括：仅在理想仿真环境下验证，未考虑分数延迟/多普勒；对大M,N的计算复杂度高；对非理想Pilot布局及实际硬件实现的适配仍待研究。

---

## 354. Uncertainty-Aware Sea-Ice Type Mapping with Multiple Ice Charts

**arXiv ID:** 2609.09451 | [PDF](https://arxiv.org/pdf/2609.09451v1)

**作者:** Samira Alkaee Taleghan `[一作]` (University of Colorado Denver), Farnoush Banaei-Kashani `[通讯]` (University of Colorado Denver)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究通过将多国海冰服务的独立注解视为多标注者不确定性，并将其与深度学习模型的预测不确定性对比，探讨海冰发育阶段（SoD）映射中的不确定性来源及其相互关联。

**💡 创新点**

创新点在于：①首次使用基于阶梯厚度的 Wasserstein 距离量化多标注者不确定性；②将多标注者不确定性与模型不确定性（预测熵、深度集成、MC dropout、证据性 Dirichlet 头、合成预测集）系统比较；③发现模型不确定性在冰缘附近与多标注者争议高度相关，并提出 MC dropout 在校准方面的优势。

**🔧 技术方法**

主要技术包括：基于 U‑Net 的深度卷积网络、深度集成、Monte Carlo dropout、证据性神经网络、分形预测集（合成预测集）以及 Wasserstein 距离用于不确定性评估。

**📊 数据集**

数据集为 404 条 Sentinel‑1 SAR 场景，配备四国（NIC、DMI、CIS、NOAA）独立海冰图注解，并加入 AMSR2 主动微波与气象上下文，共同构建多模态海冰映射数据。

**📈 对比分析**

方法对比通过五种监督策略（硬标签、共识、软监督、加权、支持集）以及三种不确定性估计（深度集成、MC dropout、证据性头）进行，使用 MAE、粗粒度准确率、RPS、NLL、Brier、ECE、Spearman 相关、AUC 等指标评估。结果显示共识监督在点和序数准确率上表现最佳，MC dropout 在 ECE 上最优，且在冰缘附近预测熵与多标注者争议的 Spearman 相关最高（0.704）。

**⚠️ 局限性**

主要局限包括：仅覆盖单一北极地区与季节，使用单一网络架构和划分方案；多标注者不确定性通过不同服务间的差异近似，可能低估真实模糊度；像素级评估对多边形级注解存在空间分布误差；结果对其他冰区或数据源的泛化性未验证。

---

## 355. CompEvo: Competition-Induced Evolution for Multi-Agent in News-Driven Time Series Forecasting

**arXiv ID:** 2609.09195 | [PDF](https://arxiv.org/pdf/2609.09195v1)

**作者:** Yuxuan Zhang `[一作]` (Sun Yat-sen University), Zehua Zeng `[通讯]` (Unilumin Group Co., Ltd.)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 CompEvo，一种基于竞争诱导进化的多智能体新闻驱动时序预测框架；

**💡 创新点**

创新点在于将进化博弈理论与可微分竞争机制结合，利用预测误差实现基于 fitness 的多样性选择，并提供理论收敛与稳健性的证明；

**🔧 技术方法**

核心技术包括可微分策略执行、基于预测误差的 fitness 选择、三阶段对手感知的逻辑演化、GRPO 策略优化、参数高效适配器和 Llama‑3.1‑8B LLM；

**📊 数据集**

使用了四个真实世界数据集：Electricity、Exchange、Traffic、Bitcoin；

**📈 对比分析**

与数值、新闻感知、单/多智能体基线（SA、MAEL、MAE‑GPT、MAD‑RL）进行对比，平均 RMSE 降低 29.8%、MAPE 降低 30.3%，在所有评估指标上均优于对照组；

**⚠️ 局限性**

局限性在于仅针对新闻驱动时序预测验证，未扩展到更广泛的预测场景或多模态证据，并且对极端噪声新闻与持续更新环境的鲁棒性仍需进一步研究。

---

## 356. EFQ-Softmax: Exp-Free Quantization for Softmax

**arXiv ID:** 2609.09721 | [PDF](https://arxiv.org/pdf/2609.09721v1)

**作者:** Haohui Han `[一作]` (Xi'an Jiaotong University), Wencong Zhang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 EFQ‑Softmax 方法，直接从移位后的注意力分数生成低位概率块，替代传统的高精度指数‑>量化路径；

**💡 创新点**

创新点在于仅用指数‑仅缩放（exponent‑only scale）与残差对数域线性阈值映射，生成 E2M1 4‑bit 代码，实现在 FlashAttention 在线软最大化中保持同一概率码的生成与使用；

**🔧 技术方法**

使用低位量化（E2M1/FP4）、FlashAttention‑style 在线递推、微尺度块（microscaling）分块、指数‑仅缩放、残差归一化、单 affine 映射、离散 LUT、A5 向量单元硬件实现以及全局参数（τ,h）校准；

**📊 数据集**

在 Qwen3‑8B（语言模型）、Qwen3‑VL‑8B‑Instruct（视觉‑语言）和 WAN2.2‑TI2V‑5B（文本‑视频）三种模型上评估，并用 VBench 对视频质量做量化；

**📈 对比分析**

与 FP16 与 MXFP4 传统指数‑>量化基线对比：EFQ‑Softmax 在 Qwen3‑8B 7‑任务平均精度提升至 0.6773（比 MXFP4 +0.0024），在 Qwen3‑VL 9‑任务平均精度提升至 0.8000（比 MXFP4 +0.0174）；在 WAN2.2 的 VBench 指标与 FP16 基线相当且图像质量最高；硬件层面在 A5 向量单元上，vector 阶段延迟平均下降 40.33%；

**⚠️ 局限性**

局限性包括：仍需手工校准 τ,h 以适配不同模型/任务；仅对 E2M1 4‑bit 量化有效，未涵盖更低位或不同格式；在极端长序列/大批量下的精度与性能仍有限；整体低位管线尚未完全实现。

---

## 357. Scaling E-Commerce Attribute Extraction with Parallel Decoding

**arXiv ID:** 2609.09716 | [PDF](https://arxiv.org/pdf/2609.09716v1)

**作者:** Nikhita Vedula `[一作]` (Amazon.com, Inc.), Shervin Malmasi `[通讯]` (Amazon.com, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个两阶段的LLM流水线，先自动发现每类产品的购买区分属性集合，再用微调的轻量级LLM在大规模电商目录中并行提取这些属性值。

**💡 创新点**

创新点包括：①基于大型LLM的自动化分类属性发现与语义标准化；②利用Hyper-Parallel Decoding实现多属性值并行解码，大幅提升推理吞吐量；③生成紧凑、跨类别可比的结构化产品知识库。

**🔧 技术方法**

核心技术包括大型LLM（Claude、Qwen3-4B/8B）、知识蒸馏、微调、超平行解码、属性名称聚类与标准化、受约束解码以及LLM评估器。

**📊 数据集**

使用了本土化30M+条产品数据集（包含标题、描述、bullet等），生成约100K个示例用于微调，20K条用于评估；此外在实验中用专门的评测集合进行人类与LLM评估。

**📈 对比分析**

与基础大型LLM基线对比，提取准确率约85%（与基线84.8%相当），吞吐量提升9倍（159K/小时 vs 18K），推理成本降低92%，并在多项指标（方案质量、正确率、误差类型）上达到或超过基线水平。

**⚠️ 局限性**

局限性：仅利用文本信息，无法提取图像中蕴含的属性；仅在英文目录上验证；假设属性值条件独立，可能对高度相关属性产生偏差；实验仅覆盖Qwen3模型族，其他模型的表现未知。

---

## 358. The Vibe Shift in Software Engineering: Evaluating AI-Led Conversational Programming for Performance, Cognition, and Responsible Adoption

**arXiv ID:** 2609.09560 | [PDF](https://arxiv.org/pdf/2609.09560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 359. Muon-C: Operator-Aligned Muon for Convolutional Kernels

**arXiv ID:** 2609.09676 | [PDF](https://arxiv.org/pdf/2609.09676v1)

**作者:** Jiaxin Qing `[一作]` (University of California, Berkeley), Lexin Li `[通讯]` (University of California, Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Muon-C，一种针对卷积核的自适应优化器，利用频率分块极化实现操作符对齐。

**💡 创新点**

创新点在于将 Muon 的矩阵几何映射到卷积的频域通道转移矩阵，并通过临界 Fourier 网格实现有限支持的独立块极化。

**🔧 技术方法**

使用频域 FFT、Newton–Schulz 极化、批量化实现以及块结构对齐技术。

**📊 数据集**

在 CIFAR‑10、ImageNet‑1k（32×32）流匹配任务和 ImageNet‑100 分类任务上进行评估。

**📈 对比分析**

与全局展开 Muon、Adam/AdamW 对比，Muon‑C 在 FID、准确率上更早收敛，并在 FLOPs、训练时间上节省约30–40%。

**⚠️ 局限性**

缺点包括额外的频域计算开销、对大规模高分辨率任务的实验不足，以及对稀疏、分组卷积等非标准卷积的适应仍需进一步验证。

---

## 360. Towards Automatic Evolution Tree Generation from Citation Graphs

**arXiv ID:** 2609.09561 | [PDF](https://arxiv.org/pdf/2609.09561v1)

**作者:** Zexing Zhao `[一作]` (Georgia Institute of Technology), Liang Zhao `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种自动从引用图中生成科学演化树的框架EvoTree，先构建时间无关的概念分类树，再通过时间一致性约束和边缘论文重映射生成演化结构，并使用LLM进行概念标签化。

**💡 创新点**

创新点在于分阶段、结构-表示交替训练的EvoTree框架：①独立学习时间无关的taxonomy backbone；②利用时间和引用方向约束重构演化树并处理过渡论文；③在少量标注下进行微调，最终只用LLM完成标签化而不改动拓扑。

**🔧 技术方法**

使用的技术包括图神经网络+SPECTER2文本编码、分布式聚类（HDBSCAN+Wasserstein）、图感知编码、时间一致性与结构合法性约束、EM式结构-表示交替更新、少样本度量学习以及LLM（GPT‑4.1-mini）标签化。

**📊 数据集**

数据集为411篇AI领域调查论文构成的ego图，以及涵盖11个AI子领域、共352篇论文的手工注释演化树，用于训练、验证和少样本校准。

**📈 对比分析**

与多种层次聚类、引用基聚类、TaxoGen、Hu‑CiteTaxo、TaxoAlign、Context‑Aware等基线比较，EvoTree在NMI、citation‑direction accuracy、FS‑CP、FS‑EDA和边缘论文检测AUROC等指标上均优于所有基线，并在LLM和人工评估中获得最高总体质量排名。

**⚠️ 局限性**

局限性包括仅在AI领域验证，引用方式和时间戳噪声可能影响时间约束；依赖调查章节结构作为弱监督；少样本标注规模有限；以及对封闭式LLM的依赖，换用开源LLM可能影响标签质量。

---

## 361. Adaptive Entangled Game Modules in Artificial General Intelligence

**arXiv ID:** 2609.09226 | [PDF](https://arxiv.org/pdf/2609.09226v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 362. MUCnoHARM@GermEval Shared Task 2026: Retrieval-based In-Context Learning for Defamatory Offences, and Where It Falls Short

**arXiv ID:** 2609.09791 | [PDF](https://arxiv.org/pdf/2609.09791v1)

**作者:** Kristin Gnadt `[一作]` (Central Office for Information Technology in the Security Sector), Matthias Aßenmacher `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了检索式上下文学习（RetICL）在德国刑法框架下判定诽谤性言论的有效性，并比较了不同的法律知识注入方式、检索策略以及模型规模。

**💡 创新点**

创新点在于将法律判决树拆解为六个子决策并作为“Explicit”条件化，系统评估检索策略（稠密/稀疏检索、相似度/多样性/随机）对少样本提示的影响，并揭示模型规模与提示配置之间的交互效应。

**🔧 技术方法**

使用技术包括：instruction‑tuned 大语言模型（Gemma‑4 26B、Gemma‑4 4B、Qwen3.5‑9B、EuroLLM‑22B、GPT‑5.5）、检索式提示（dense embedding via Sentence‑Transformers、BM25 sparse embedding、fusion）、few‑shot/zero‑shot提示、静态与动态示例检索、QLoRA 参数高效微调。

**📊 数据集**

数据集为 GermEval 2026 子任务 4（DEF），包含 3,263 条德国推文的二分类标签，另外 797 条示例有每一步判决树的细粒度标注；测试集 577 条未标记推文用于官方竞赛提交。

**📈 对比分析**

与 1‑NN TF‑IDF 基线（F1_macro=0.644）以及同类模型的零/少样本提示对比，最佳配置（Gemma‑4 Implicit + dense similarity）达到 F1_macro≈0.733，召回率在 0.43–0.74 之间；微调 Qwen3.5‑9B 在 0‑shot+Implicit 提示下得到 F1_macro≈0.741，明显优于大多数动态提示，但在正类召回上仍低于基线。

**⚠️ 局限性**

局限性：仅在单一语言、单一数据集和单一标注方案上验证；数据同质性导致 1‑NN 基线表现强劲，无法检验概念漂移或多样化数据；未覆盖所有可能的配置组合；未使用推理型法律提示（链式思考/法律演绎）；结果对自动化执法的实际意义有限。

---

## 363. CEDD-optimizer: Enabling Cost-Efficient Dataset Distillation on Geographically Distributed Edge Systems

**arXiv ID:** 2609.10151 | [PDF](https://arxiv.org/pdf/2609.10151v1)

**作者:** Dai Liu `[一作]` (Technical University of Munich), Martin Schulz `[通讯]` (Technical University of Munich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为CEDD-optimizer的超参数调优框架，旨在提高地理分布的边缘系统中数据集蒸馏的成本效率。

**💡 创新点**

创新点在于首次识别了在非均匀地理分布的边缘环境中进行数据集蒸馏的超参数调优挑战，并提出了一个数学优化模型来解决这一问题。

**🔧 技术方法**

使用了超参数调优框架CEDD-optimizer，包括CEDD-calibrator和CEDD-solver两个模块，前者用于离线和在线模型参数识别，后者用于在线优化超参数设置。

**📊 数据集**

在多个图像数据集上进行了实验，包括MNIST、Fashion-MNIST、CIFAR-10、SVHN和ImageNette。

**📈 对比分析**

与基线DD方法相比，CEDD-optimizer在相同质量约束下实现了高达20.8倍的成本降低，且在准确性上保持竞争力。

**⚠️ 局限性**

限制在于CEDD-optimizer的性能可能受到边缘设备的异构性和地理位置变化的影响，且在大规模部署时，求解器的运行时间可能成为瓶颈。

---

## 364. Adaptive Shared Control with Online Bounded-Rational Human Behavior Estimation

**arXiv ID:** 2609.10215 | [PDF](https://arxiv.org/pdf/2609.10215v1)

**作者:** Henry Ascencio Trejo `[一作]` (Tampere University), Gokhan Alcan `[通讯]` (Tampere University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出一种自适应共享控制框架，通过在线估计人类行为的有限层级有界理性模型（level‑k），并利用该概率分布来计算机器人的一步分布感知最佳响应，从而在共享控制过程中实现对人类非完全理性行为的实时补偿。

**💡 创新点**

创新点包括：① 将有界理性（level‑k）模型与自适应动态规划（ADP）相结合，构建离散化的人类与机器人行为候选库；② 通过状态转移残差与softmax映射在线更新概率人类模型（PHM），无需直接观测人类输入；③ 设计分布感知的一步最佳响应，并在二次终端成本与Euler近似下给出闭式解，显著降低实时计算负担。

**🔧 技术方法**

使用的技术主要有：有界理性（level‑k）递归、非线性控制仿射系统建模、基于值函数的ADP（神经网络逼近）、状态残差累计与指数衰减、softmax概率映射、一阶优化与闭式求解。

**📊 数据集**

数据集：在两种仿真环境下评估：① 采用复制驱动的四维非线性系统（benchmark system）；② 采用二维平面两连杆机械臂系统。所有实验均基于自定义仿真，未使用公开数据集。

**📈 对比分析**

与两种基线（最大概率选择与概率加权平均）对比，实验显示：① 在人类行为估计方面，KL 散度随时间下降，最终远低于基线；② 在成本表现上，分布感知响应的累计运行成本比基线低约 40%~60%，机器人控制能量最低，整体任务完成更平滑、效率更高。

**⚠️ 局限性**

局限性：① 需要先离线学习完整的 level‑k 行为库，若任务或人类策略空间改变需重新训练；② 假设人类目标与意图在交互过程中保持不变，无法处理意图漂移；③ 只在仿真环境下验证，未验证对真实硬件、噪声与观测不完整的鲁棒性；④ 仍依赖对系统动力学与状态的完整观测。

---

## 365. Infra-Bench CLS: A Global, Open-Source Benchmark for Critical Infrastructure Classification with Earth Observation Foundation Models

**arXiv ID:** 2609.09482 | [PDF](https://arxiv.org/pdf/2609.09482v1)

**作者:** Justin Guthrie `[一作]` (George Mason University), Isaac Corley `[通讯]` (Taylor Geospatial Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了全球开源基准 Infra‑Bench CLS，用于评估地球观测基础模型（FM）在设施级关键基础设施分类任务上的表现；收集了18,756张 Sentinel‑1 SAR 与 Sentinel‑2 多光谱图像，覆盖七大洲、13类资产；并构建了基于 OSM 标签的训练、验证、测试拆分；

**💡 创新点**

首创全球规模的设施级关键基础设施分类基准，并系统评估七种基础模型在线性探针和全微调下的性能；在有限标注数据条件下验证微调显著优于线性探针；发现自然图像预训练模型 DINOv3 在此任务上超越专用 EO 预训练模型，凸显跨域迁移潜力；

**🔧 技术方法**

采用多种地球观测基础模型（SatlasPretrain S1/S2、CROMA、Prithvi‑EO‑2.0、AlphaEarth Foundations、OlmoEarth v1.1‑Base、DINOv3 ViT‑L/16），配合线性探针和全微调；使用类权重平衡、AdamW、cosine LR 调度、固定批大小、梯度累积等训练策略；对不同波段和分辨率进行了统一预处理；

**📊 数据集**

Infra‑Bench CLS 自建数据集，基于 OpenStreetMap 标签与 Microsoft Planetary Computer 提供的 Sentinel‑1 (VV, VH) 与 Sentinel‑2 (10 m 多光谱) 图像；共计 13 类资产，后选取 10 类用于评估；同时提供 Maine/NH 区域的手工验证子集；

**📈 对比分析**

通过与随机初始化 ResNet‑18 的线性探针和全微调基线对比，并在 1.0× 与 0.3× 两种训练集规模下评估；宏 F1 结果显示 DINOv3 微调最高达 0.579，SatlasPretrain S2 0.559，OlmoEarth 0.540，Prithvi‑EO‑2.0 0.481，AlphaEarth 0.473；线性探针表现远低于微调；在 0.3× 训练集下，DINOv3 微调 0.525、SatlasPretrain S2 0.499 等均超过完整训练 ResNet‑18 的宏 F1 0.392，验证了少量标注数据时基础模型的优势；

**⚠️ 局限性**

主要限制包括：1）依赖 OSM 标签，存在标注噪声与不一致；2）部分类别样本不足导致评估不稳定；3）预训练与评估分辨率、波段配置不匹配，可能影响结果；4）未考虑更高分辨率或多时相数据；5）仅在 10 m 分辨率下评估，细粒度类难以区分；6）基线与 FM 架构规模不一致，难以分离预训练效益与网络大小效应；7）未评估对分布漂移的鲁棒性。

---

## 366. The Double Measurement Confound in Agent Benchmarks: De-Scaffolding, Ground-Truth Scoring, and Reliability Beyond the Mean

**arXiv ID:** 2609.09218 | [PDF](https://arxiv.org/pdf/2609.09218v1)

**作者:** Yonghong Zhang `[一作]` (Universidad Autonoma de Madrid), Yong Xie `[通讯]` (Spanish National Research Council)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文识别并修正了代理评测中的双重测量混淆（即评测框架的 scaffold 负责关键执行决策且评分器仅基于输出形状），提出了一套可执行的审计与修复协议：将执行关键决策移交给模型、将评分改为对种子生成的真实结果的比对，并加入 worst‑case、CVaR 与可靠性阈值等指标；随后在自定义的“交易数据提取”基准上以及 τ‑bench 与 BFCL 两个公开基准上验证该协议。

**💡 创新点**

创新点包括：① 用测量理论统一框架阐释 scaffold 与 scorer 的联合混淆；② 设计联合 2×2 干预（scaffold 级别 + scorer 选项）来实现可辨识性；③ 引入可靠性评估指标（worst‑case、CVaR、Reliability@τ）以揭示平均值掩盖的模型差异；④ 提出“有效性卡”来系统评估基准的可解释性；⑤ 在自己的预注册实验中证明了混淆对实验结论的影响。

**🔧 技术方法**

技术手段包括：测量理论模型（S(m,c,j)），seeded 程序化生成器、工具调用 harness、L0/L1/L2 scaffold 级别控制、ground‑truth 评分器、形状评分器的对比；统计方法：bootstrap 95% CI、Cliff’s δ、CVaR@α、Reliability@τ；实用化为可复现的审计管道和有效性卡生成脚本。

**📊 数据集**

使用的数据集为：1) 自研的“交易数据提取”基准，包含 5 个执行任务（T3‑T6、T8）和 10 个种子；2) τ‑bench 的官方轨迹（零模型调用的验证用）；3) BFCL 的 AST 轨迹（400 条离线提交）。

**📈 对比分析**

评估方法：在三种 scaffold 级别（L0、L1、L2）和两种 scorer（shape、ground‑truth）下对 8 种模型进行 5–10 颗种子实验；比较指标包括平均得分、worst‑case、CVaR@0.2、Reliability@0.9；结果显示：在 L2+ground‑truth 下出现完整的可靠性谱，平均值高的模型如 GPT‑5 在 worst‑case 归零，导致排名被重新排序；外部基准的审计表明 scorer 仅 benchmark‑specific，scaffold 则为 uncontrolled 轴。

**⚠️ 局限性**

局限性：① 仅在单一领域（基于 seed 的交易数据提取）进行实验，未检验跨领域迁移；② 任务数与种子有限，无法覆盖更大规模的评测；③ 对 scaffold 干预仅验证了两级（L0、L2）和形状 vs ground‑truth scorer，未覆盖所有可能的 harness 设计；④ 外部基准的验证仅靠零调用探针，无法评估需模型交互的评分器；⑤ L2 评分因 re‑emission 上限产生技术性 floor，导致极端模型的结果被截断。

---

## 367. Frame-Coded Legged Locomotion over Noisy Terrain

**arXiv ID:** 2609.10273 | [PDF](https://arxiv.org/pdf/2609.10273v1)

**作者:** Lav R. Varshney `[一作]` `[通讯]` (Stony Brook University), Lav R. Varshney (Stony Brook University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于有限帧编码的多足机器人地形适应运动方法，将全局运动指令展开为异构本地接触指令，并通过形态学的可耦合弹性网络实现对丢失或噪声接触的自适应解码；

**💡 创新点**

核心创新在于将运动指令视为向量信号，用有限帧（如等范数Parseval帧、低相干度帧或调和帧）进行冗余展开，并证明机械平衡即为最小均方误差（MMSE）估计，进而将信息量、弹性刚度与误差范式直接关联，给出信息–运动性能的不等式与渐进极限；

**🔧 技术方法**

采用线性高斯模型、随机矩阵论、信息理论（MMSE-互信息公式）、机械能量最小化和有限帧理论，辅以数值仿真验证；

**📊 数据集**

主要使用理论分析与模拟实验（随机生成的高斯帧、随机接触存活模型），未引用公开真实机器人数据集；

**📈 对比分析**

通过信息理论界定可恢复率阈值R<q，并给出随机帧在接触存活率q下的误差指数和刚度衰减；相较于传统重复编码方案，提出的帧编码在相同冗余下可实现更高的有效运动维度并显著降低误差放大；

**⚠️ 局限性**

局限在于假设线性化的指令-接触映射、静态弹性平衡、理想的接触门控和高斯噪声；未考虑摩擦锥、非线性几何、惯性动力学、碰撞约束及实际机器人硬件限制。

---

## 368. Who You Are Adds Nothing Detectable to Where You Go Next: Sociodemographic Conditioning in LLM Next-Location Prediction

**arXiv ID:** 2609.09609 | [PDF](https://arxiv.org/pdf/2609.09609v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 369. Understanding the Security Boundary of Obfuscation-based On-Device LLM Protection

**arXiv ID:** 2609.10117 | [PDF](https://arxiv.org/pdf/2609.10117v1)

**作者:** Hanyi Zhou `[一作]` (Tsinghua University), Zhuotao Liu `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对TEE‑Shielded LLM Partition（TSLP）中现有的加密/混淆方案，作者提出统一的加密原语框架，将不同方案的权重变换拆解为可组合的线性原语，进一步推导出该原语族的 canonical 形式 𝒪_prior，并基于此设计了系统性的三阶段攻击 Collapse，随后引入两种新的稀疏乘法和双边乘法原语，扩展安全边界并构建新的防御方案 𝒪_X。

**💡 创新点**

创新点主要有：① 把多种散落的 TSLP 防御抽象为统一的原语集合；② 证明该原语族的 canonical 形式 𝒪_prior 是安全边界；③ 设计基于原语的系统攻击 Collapse，揭示共享的列方向泄露弱点；④ 提出稀疏乘法和双边乘法两种新原语，能够打破列/行方向的泄漏，进一步提升安全边界。

**🔧 技术方法**

技术方法包括：线性代数原语分析、可组合原语的重写系统、三阶段攻击流程（列对齐、低秩子空间交集、稀疏支持估计与 Fine‑tune）、多视图列匹配（LDD oracle）、低秩/稀疏分解、以及实验评估中的模型压缩与对齐。

**📊 数据集**

使用了四个代表性模型（BERT‑Base、ViT‑Base、Qwen2.5‑0.5B、Qwen2.5‑1.5B）以及四个数据集（MNLI、SST‑2、QNLI、CIFAR‑100）进行实验评估。

**📈 对比分析**

与多种先前的 TSLP 防御（ArrowCloak、TSQP、TransLinkGuard、NNSplitter、LoRO 等）以及 White‑box/Black‑box 基线进行对比。实验表明 Collapse 攻击在所有配置下平均提升了 1.45× 的 Black‑box 性能，接近 White‑box 上限；新防御 𝒪_X 在相同成本下将攻击性能压回到 1.0× 的 Black‑box 水平，说明其有效性。

**⚠️ 局限性**

局限性包括：① 评估仅在现有防御流水线下完成，未针对新防御 𝒪_X 设计自适应攻击；② 未给出正式的模型提取安全证明，只能提供经验性评估；③ 对硬件侧信道、故障注入等攻击未考虑；④ 密钥刷新频率越高，反而可能降低安全性（泄漏多视图信息）。

---

## 370. Black-Box Red Teaming of Agentic AI: A Taxonomy-Driven Framework for Automated Risk Discovery

**arXiv ID:** 2609.09647 | [PDF](https://arxiv.org/pdf/2609.09647v1)

**作者:** Divyanshu Kumar `[一作]` (Enkrypt AI), Prashanth Harshangi `[通讯]` (Enkrypt AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个基于黑盒的、七域风险分类体系的代理系统评估框架，利用自动化的 SAGE-RT 红队生成多步骤攻击场景并通过 LLM 判定与人工复核完成风险评估。

**💡 创新点**

创新点包括：①将可观测行为映射到七类具体风险的系统化分类法；②通过 SAGE-RT 自动生成120个跨域攻击场景，实现全流程无人工干预；③将 LLM 判定与人工复核相结合的评估流程；④揭示风险与代理架构的高度相关性。

**🔧 技术方法**

技术手段主要包括 SAGE-RT 场景生成器、GPT‑4o 作为自动化判定器、CrewAI 与 AutoGen 代理框架、LLM 判定 Rubric、人工专家复核。

**📊 数据集**

使用的“数据集”为：①CrewAI 的餐厅接待代理（含 500 条预订、50 张桌子、100 个菜品的 SQLite 语料）；②AutoGen 的股票咨询代理（含三名子代理与工具集）；以及通过 SAGE-RT 自动生成的 1,440 条多步攻击场景。

**📈 对比分析**

评估通过在两种代理架构下对四个基础模型（gpt‑4.1‑nano、mistral‑small、gemini‑2.5‑flash、kimi‑k2‑instruct）执行 120 个场景/域，得到的风险热图与统计。实验显示治理风险均值 56.25%，隐私风险最高达 65%，多代理架构在行为风险上最高可达 85%；不同模型表现一致，风险主要受架构影响。

**⚠️ 局限性**

局限性包括：①未评估防御措施的有效性；②只能发现黑盒可探测的漏洞，无法检测需源码级分析的缺陷；③攻击场景多为英语，可能忽略多语种或文化特定的攻击；④随着代理技术演进，风险分类可能需持续更新。

---

## 371. Hyperbolic Geometry for Open-World Object Detection in Remote Sensing Imagery

**arXiv ID:** 2609.09626 | [PDF](https://arxiv.org/pdf/2609.09626v1)

**作者:** Wuzhou Li `[一作]` (Wuhan Textile University), Xiang Li `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 HyRS-OWOD 框架，用于遥感图像中的开放世界目标检测，包含解耦目标性学习（DOL）、基于双曲半径的不确定性学习（HUL）以及双曲度量学习（HML），实现未知物体召回、未知-已知分离和增量学习。

**💡 创新点**

创新点主要有三：①首次将双曲几何引入遥感开放世界检测，利用双曲半径作为不确定性指标；②通过 DOL 解耦类别信息与目标性，提升未知候选提议的召回率；③使用 HML 在双曲空间中强化类内紧凑与类间分离，有效缓解增量学习中的灾难性遗忘。

**🔧 技术方法**

采用的技术包括 Poincaré 球模型双曲嵌入、Möbius 运算、随机盒子（RandBox）基础框架、ResNet‑50+FPN backbone、AdamW 优化器、交叉熵+焦点损失、超平面回归损失、解耦相关损失、超曲不确定性损失以及基于距离加权的双曲度量学习。

**📊 数据集**

实验使用了三大遥感目标检测数据集：NWPU VHR‑10、DIOR、DOTA，并在多种已知/未知类别划分（如 16+4、10+10、8+8 等）和增量学习任务（Task 1–4）上进行评估。

**📈 对比分析**

与 ORE、OW‑DETR、PROB、KTCN、SGROD 等现有 OWOD 方法以及 RandBox 基线进行对比。实验结果显示：在 U‑Recall 上平均提升 7–12 个百分点，mAP（已知、已知+未知）提升 4–10 个百分点，尤其在增量学习阶段能显著保持已知类别性能并快速学习新类别，整体性能普遍超过所有对照方法。

**⚠️ 局限性**

局限性包括：①对超参数（如双曲曲率、阈值、温度）敏感，需手工调优；②在更大规模或多源遥感图像上的鲁棒性尚未充分验证；③缺乏对模型在实时部署、计算成本和推理速度方面的系统性评估。

---

## 372. Policy Change for Treelike Monitors

**arXiv ID:** 2609.10114 | [PDF](https://arxiv.org/pdf/2609.10114v1)

**作者:** François Hublet `[一作]` (ETH Zurich), Joshua Schneider `[通讯]` (ETH Zurich)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在线监测过程中在运行时动态更改规范（策略）时的可行性，并提出了一个决策问题来判定在任何监测树（treelike monitor）上是否能够完成策略变更。

**💡 创新点**

创新点包括：①将策略变更问题形式化为深度双射（deep bisimilarity）上的转换；②引入超逻辑HyperpTPTL来编码监测树的等价类，并将其还原为可判定的1‑TPTL可满足性问题；③给出了实时时间语义下非原始递归（non‑primitive‑recursive）复杂度的上界与下界，离散时间语义下的EXPSPACE‑完整性。

**🔧 技术方法**

使用的技术：监测树（treelike monitor）与深度双射、pMTL 与 pLTL 逻辑、TPTL 与 HyperpTPTL 逻辑、信息公式、超逻辑的量化约简、归约到1‑TPTL可满足性、复杂度分析（非原始递归、EXPSPACE）。

**📊 数据集**

未使用任何实证数据集；研究完全基于理论构造与形式化证明。

**📈 对比分析**

方法与现有自适应监控/合成方法比较，表明在监测树上可实现多次策略变更；性能上以理论复杂度衡量：实时时间语义下为非原始递归，离散时间语义下为EXPSPACE；没有实验评估。

**⚠️ 局限性**

局限性：①对实时时间语义下非树形监视器的可判定性仍未解决；②离散时间语义下对更低复杂度的改进尚不可行；③缺乏实验验证，未展示具体实现效率。

---

## 373. Improving Cross-Lingual Token Representations by Adding a Pinch of SALT

**arXiv ID:** 2609.09953 | [PDF](https://arxiv.org/pdf/2609.09953v1)

**作者:** Guillem Ramírez `[一作]` `[通讯]` (University of Edinburgh), Guillem Ramírez (University of Edinburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种轻量级的后训练方法SALT，通过在多语言句子编码器中加入跨语言短语级监督，显著提升了 token 级表示的质量；

**💡 创新点**

创新点在于利用跨语言短语对作为监督信号，结合对比、翻译与插值三种损失，使得模型在保持句子级别性能的同时，提升 token 级别性能，且不需修改模型结构；

**🔧 技术方法**

主要技术包括：基于 LLM 或自监督 CASE 方案提取跨语言短语；对齐短语的对比学习（Span‑Contrastive Loss）；使用冻结的解码器进行短语翻译（Span‑Translation Loss）；以及保持句子嵌入空间的插值损失；

**📊 数据集**

使用数据集包括 NLLB Primary 并抽取 129 语言对的平行句子进行后训练；评估数据包括 FLORES‑200 devtest、MTEB 分类任务，以及 5 个 token 级基准（词对齐、PAN‑X NER、Massive 槽位填充、UDPOS POS、WiC 词义辨识）；

**📈 对比分析**

与 XLM‑R、XLM‑Align、LaBSE、MEXMA 等预训练编码器以及 SONAR 的多种 fine‑tuning 方案进行比较；SALT 在 4/5 个 token 级基准上取得最佳或并列最佳成绩，且在句子检索和分类任务中保持或提升性能；

**⚠️ 局限性**

局限性包括：需要平行语料来提取短语，CASE 方案依赖原始编码器的跨语言表示质量，平均池化可能无法捕捉长短语内部结构，仅在后训练阶段验证，未探索在大规模预训练中加入短语监督，低资源语言的适用性有限。

---

## 374. UNISON: A Co-Designed Near-Memory Scheduler of Session KV Residency for LLM Agents

**arXiv ID:** 2609.09643 | [PDF](https://arxiv.org/pdf/2609.09643v1)

**作者:** Fan He `[一作]` (Fudan University), Xiaoyang Zeng `[通讯]` (Fudan University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种面向大型语言模型代理循环的近内存调度器，利用工具等待间隔和轮次危害信息对 KV 缓存进行统一的淘汰和层级置换决策。

**💡 创新点**

创新点包括：① 将生存惩罚淘汰与空闲窗口 DMA 迁移统一为单一排名；② 在硬件层面实现事件驱动的统一控制平面，避免软件调度误判；③ 通过 28 nm CMOS 轻量级调度核心实现低功耗高精度排名。

**🔧 技术方法**

采用技术包括：事件驱动近内存调度、指数滑动平均、轮次危害表、定点扫描算术、DMA 预算驱动层级迁移以及 CMOS 逻辑合成。

**📊 数据集**

使用了 SWE‑bench 与 GAIA 的 6 条代理工作负载，涵盖 Qwen3‑Coder‑30B、Devstral‑24B、Gemma4‑E4B 三个模型族，共 1,415 个会话、33,596 次轮次。

**📈 对比分析**

与 LRU、timeout、AGSERVE、CacheScout 等基线以及离线 Bélády oracle 对比，联合策略在所有轨迹上实现最高命中率和 AMAT，平均提升 34.8% AMAT、命中率提升 0.3%（达 23.1%），TTFT 在长周期轨迹上降低 58% 至 89%，并在 28 nm CMOS 0.169 mm²、13.6 mW 下实现。

**⚠️ 局限性**

局限性：当工具等待短于解码窗口时，软件时间戳会反转间隔信号；当预填充已饱和 GPU 时，命中率提升不转化为延迟下降；调度核心仅支持两层 KV 层级，且需硬件事件接口，难以在纯软件系统中实现。

---

## 375. The Living Library: Transforming Archival Collections into Conversational Knowledge Systems -- Lessons from the Theodore Roosevelt Presidential Library

**arXiv ID:** 2609.09368 | [PDF](https://arxiv.org/pdf/2609.09368v1)

**作者:** Pengce Wang `[一作]` (Microsoft), Juan Lavista Ferres `[通讯]` (Microsoft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计并实现了四层Living Library框架，将特奥·罗斯福档案数字化、AI处理、检索推理与可选对话界面整合，并在特奥·罗斯福总统图书馆部署了公众互动数字人展览Talk to TR。

**💡 创新点**

创新点在于跨时代类比重构、双路径延迟感知检索、实时全链路流式与非阻塞安全栈，构建了实时可信且非时效化的历史人物对话体验，并提出可复制的架构与治理原则。

**🔧 技术方法**

技术组合包括OCR/LLM、Azure AI Search、检索增强生成、双路径检索、增量ASR、TTS、实时头像渲染(Unreal Engine+Lemon Slice)、自适应填充、分层监控与自动恢复等。

**📊 数据集**

使用约30万条特奥·罗斯福档案记录（信件、手稿、书籍、照片等），来自四十个机构的原始素材，经OCR后形成700多GB文本并索引。

**📈 对比分析**

对OCR模型进行源对齐评估，GPT‑5在CER/WER/BLEU/嵌入相似度上最优；现场部署首词延迟平均2.8秒，97%答案在5秒内；检索准确性通过专家审查与用户日志验证，保持高可信度与非时效性。

**⚠️ 局限性**

局限在于OCR误差仍存在、检索策略偶尔误检或漏检、对话真实性与非时代化需人工评估、系统在极端高并发或长会话中的鲁棒性未完全验证、缺乏系统化访客体验与可视化测评。

---

## 376. Distributed and Private Textual Data Synthesis from Embeddings

**arXiv ID:** 2609.10104 | [PDF](https://arxiv.org/pdf/2609.10104v1)

**作者:** Ergute Bao `[一作]` (Inria), Xiaokui Xiao `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在无需可信中心、仅需轻量级用户参与的分布式环境下，设计并实现了差分隐私文本合成算法 Fre-E2T。

**💡 创新点**

创新点包括：①将聚类和中心点求平均的过程压缩为一次性投影+网格分桶，避免迭代与大规模通信；②引入语义支持保护（Semantic Support Protection）保证稀有文本在输出中保持隐蔽；③将多方安全聚合与随机投影、随机网格分桶等技术融合，构成完整的两阶段分布式协议。

**🔧 技术方法**

核心技术：Johnson–Lindenstrauss 随机投影、随机网格分桶、离散高斯/拉普拉斯随机化、稀疏重采样（sample‑and‑threshold）求 heavy‑hitter、加密标签（VOPRF）、安全聚合（SecAgg）以及后处理的向量→文本逆向模型。

**📊 数据集**

实验使用四个真实用户查询数据集：TaylorAI、Instructions‑2M、LMSYS Chat、Yelp；使用 768 维句子/文本嵌入模型（如 Sentence‑BERT）。

**📈 对比分析**

与中心化差分隐私基线 Aug‑PE 进行比较，Fre‑E2T 在相同隐私预算下实现了相当甚至更好的精度（Precision/Recall/F1）和较低的平均 L2 距离，尤其在较高 ε 下与中心化基线接近；同时在稀有文本保护上优于 Aug‑PE。

**⚠️ 局限性**

主要限制：逆向嵌入→文本的模型（Vec2Text/GPT‑2）在噪声干扰下的恢复质量仍低于原始文本，导致合成文本的自然度和下游任务性能不及中心化方法；此外，安全聚合的实现与网络延迟、参与度波动等因素有关，实际部署仍需优化。

---

## 377. Optimality of Kløve Arrays within the Symmetric Kløve-Mossige Class

**arXiv ID:** 2609.09239 | [PDF](https://arxiv.org/pdf/2609.09239v1)

**作者:** Lilin Yan `[一作]` (Northwestern Polytechnical University), Hongwei Zhao `[通讯]` (Northwestern Polytechnical University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

本文给出了最小冗余对称 Kløve–Mossige（S-KMA）数组在所有传感器数固定的最优问题中，必定属于 Kløve 子族（KA）的证明。

**💡 创新点**

创新点在于：①用解析与计算机辅助相结合的方式，完成了对无限多传感器数的判定；②将最优数组归类为 KA，消除了此前仅能证明存在 KA 的结论；③提供了一个完整的有限枚举证书，对 2≤N≤329 的情况给出全覆盖的验证。

**🔧 技术方法**

使用的技术包括：组合数论与加法基底的理论、极大化问题的解析上界与下界推导、整数规划与符号计算、基于生成器参数的枚举搜索、以及多种独立实现（C、Python）对结果进行交叉验证。

**📊 数据集**

使用的数据集主要是通过枚举所有满足条件的生成器参数 (x,y,z,λ) 产生的 S-KMA 组和对应的 KA 组，覆盖 N≤329 的所有可能情况；此外对 N≥330 则仅需对解析上界与下界进行比较。

**📈 对比分析**

比较方法：先给出经典九块 Kløve 构造的下界，然后对任意非 KA 最优候选通过两条上界（Φ 和 (N+3)^2/8-2）进行排除；在 N≥330 时，两条上界与下界的差距已足以保证无非 KA 最优方案。性能上，整个计算在 329 以内完成约 1.8 万次检查，且通过多实现验证无误。

**⚠️ 局限性**

局限性：仅证明固定传感器数时最优方案属于 KA，未讨论不同 N 下最优集合的数量和结构；计算部分仅验证到 N=329，超大 N 的细节仅靠解析上界，且对非对称或非连续和共振数组的情况未作覆盖。

---

## 378. Contextual Bandit-Based Decomposition of Network Slice Requirements under Cumulative Resource Budget Constraints

**arXiv ID:** 2609.09624 | [PDF](https://arxiv.org/pdf/2609.09624v1)

**作者:** Masaki Kobayashi `[一作]` (NTT), Masahiro Kobayashi `[通讯]` (NTT)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在线网络切片解构策略（CCKB），用于在多域5G网络中将端到端切片请求拆解为域级服务级别协议（SLA）要求，并实现资源分配与预算管理。

**💡 创新点**

创新点在于：① 将解构策略优化问题（nsrdp）视为在线学习任务；② 采用基于高斯过程（GP）的预测模型快速评估候选解构方案的奖励与资源占用；③ 引入资源紧张惩罚机制，动态抑制瓶颈域资源的过度消耗，从而提升长远资源预算利用率。

**🔧 技术方法**

核心技术包括：在线强化学习框架、GP回归预测、惩罚调度机制、以及多域资源分配算法。

**📊 数据集**

使用了基于5G网络拓扑、瓶颈位置与不同流量混合的仿真数据集（通过多种拓扑和流量场景生成），不涉及公开真实网络数据集。

**📈 对比分析**

与现有基线方法（如基于规则的解构、随机解构、传统的线性规划等）进行对比，实验结果显示在大多数拓扑、瓶颈与流量组合下，CCKB在资源利用率、请求接受率和SLA满足率方面均显著优于基线，提升幅度从10%至30%不等。

**⚠️ 局限性**

局限性包括：① 仅在仿真环境验证，缺乏真实网络部署实验；② 高斯过程预测在大规模、实时场景下可能产生计算瓶颈；③ 对于极端突发流量或快速拓扑变化的鲁棒性尚待进一步评估。

---

## 379. Gradland: On Phenomenal Experience, Differentiated Across Many Dimensions

**arXiv ID:** 2609.09306 | [PDF](https://arxiv.org/pdf/2609.09306v1)

**作者:** David Balduzzi `[一作]` `[通讯]`, David Balduzzi

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出经验可以由物理相互作用的梯度（雅可比矩阵）来刻画，并在理想化的 Gradland 世界里引入两种度量——透明度（有效秩）和聚合度（Kirchhoff 复杂度）来量化体验的维度与连贯性，随后用这些度量分析经验的时长、鲜明度、纹理、出生时的混乱感、学习过程以及丰富体验的功能。

**💡 创新点**

创新点在于：① 将梯度/雅可比矩阵视为经验的核心结构，打破传统信息论方法；② 提出有效秩和聚合度两种可计算的量化指标；③ 通过多种神经网络实验（MLP、RNN、Transformer、ReLU vs Sigmoid、注意力）展示这些指标如何解释人类经验现象；④ 建立 Gradland 这一可仿真平台以检验假设。

**🔧 技术方法**

使用的技术包括：神经网络模拟、梯度与雅可比矩阵计算、奇异值分解、有效秩与聚合度的数学定义与实现、Kirchhoff 行列式求解、经验可视化与统计分析。

**📊 数据集**

主要实验数据来自人工生成的网络结构和随机初始化的 MLP（如 24 层 ReLU、AlexNet 预训练卷积核），并未使用公开的真实数据集；实验集中在网络结构本身而非输入数据集。

**📈 对比分析**

比较方法：对同一网络的不同结构/激活函数进行雅可比矩阵分析，计算有效秩和聚合度，随后将指标值与人类经验描述（如时间感知、鲜明度、纹理）进行关联。实验表明：时间维度的连贯性由 RNN 的低秩非块对角雅可比矩阵解释；鲜明度与激活函数的梯度特性相关；纹理与卷积核的空间梯度直接对应；随机初始化网络的混乱感与梯度的白噪声分布对应。性能方面，度量计算在大规模网络下可在 O(nd·min(n,d)) 时间内完成。

**⚠️ 局限性**

局限性：① 仅适用于可微分、连续的理想化系统；② 假设经验与雅可比矩阵一一对应，忽略了离散事件和非线性跳变；③ 未考虑批量处理、前向后向分离导致的多重“单子”现象；④ 对真实大脑的映射仍处于概念阶段，实验仅验证在 Gradland 的可行性；⑤ 高阶相互作用（如二阶梯度）被简化忽略。

---

## 380. Can AI Agents Detect and Repair Artifact Drift in Network Experiments?

**arXiv ID:** 2609.09849 | [PDF](https://arxiv.org/pdf/2609.09849v1)

**作者:** Tianzhu Zhang `[一作]` (Nokia Bell Labs), Meikang Qiu `[通讯]` (Augusta University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了一种衡量AI代理在网络实验记录中保持artifact完整性的基准（Artifact Integrity Benchmark），并对多种通用代理进行评估。

**💡 创新点**

创新点在于：①把artifact完整性定义为支持路径可追溯、范围受限且证据充足；②构造了52个公开源代码驱动的实例，并给出确定性评分协议；③揭示了通用代理在跨artifact、隐式关系和传播修复时的显著性能下降。

**🔧 技术方法**

使用技术包括大型语言模型（GPT‑5系列、Claude、Gemini等）与代理运行时（Codex CLI、Cursor Agent、OpenCode），以及基于claim–evidence–scope图模型的自动评分器。

**📊 数据集**

数据集由Batfish、Zeek、P4C、Open vSwitch等公开项目的源文件、实验记录与注入的一致性缺陷构成，共52个实例。

**📈 对比分析**

比较方法是按合同完整性（Recall、Repair、Label、Accusation、Residual）计分，平均通过率为65.3%，但在结构层级越高时通过率从近99%降至约26%，显示通用代理在复杂修复上表现不足。

**⚠️ 局限性**

局限性：基准仅评测人工注入的故障，缺乏真实网络工作负载；只考虑通用代理，未测试网络专用代理；评估高度确定性，可能忽视语义级别的合理修复。

---

## 381. XAgent: eXecution-guided Agentic AI for Effective Localization and Resolution of GitHub Issues

**arXiv ID:** 2609.09769 | [PDF](https://arxiv.org/pdf/2609.09769v1)

**作者:** Hieu Huynh `[一作]` (University of Melbourne), Kla Tantithamthavorn `[通讯]` (Monash University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于执行引导的代理式人工智能框架，用于在 GitHub 仓库层面自动定位并修复问题

**💡 创新点**

创新点在于：①使用差分执行分析对失败与成功执行进行比较，显著缩小定位搜索空间；②构建上下文感知验证测试增强器，提升验证覆盖率，避免过拟合；③通过动态信息与语义相似度结合的多维排名提高定位精度

**🔧 技术方法**

核心技术包括：大型语言模型（Claude 4 Sonnet）与工具交互、差分树编辑距离（APTED）、函数级调用树跟踪、上下文抽取+零射提示、余弦相似度排名（UniXcoder）

**📊 数据集**

使用 SWE-bench-lite（300 个真实 GitHub issue）进行实验，亦对 SWE-bench-verified 子集进行验证

**📈 对比分析**

与 9 个 SOTA 基线（ExpeRepair、Refact Agent、SWE‑Agent 等）对比，达到 62.0% 的 issue 解决率、72.8% 的函数级定位 F1 分数，且成本比上一最佳低 37%（约每题 1.56 美元），同时在 7 个复杂问题上独占胜利

**⚠️ 局限性**

主要限制包括：对低质量或缺失测试用例的依赖；生成的 reproduction 脚本成功率仅 93%，其余需回退到 LLM 定位；验证测试的随机性受高温度影响，需多次重复保证一致性

---

## 382. Teacher Geometry Shapes Learnability in Teacher-Student Networks

**arXiv ID:** 2609.09595 | [PDF](https://arxiv.org/pdf/2609.09595v1)

**作者:** Kai J. Sandbrink `[一作]` (École Polytechnique Fédérale de Lausanne), Johanni Brea `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

研究教师-学生网络的可学习性，探讨教师网络几何对学习成功率的影响；

**💡 创新点**

提出最大/标准/最小节点不相似度教师分布，量化学习可行性；分析损失景观并解释不同教师分布导致的局部最小差异；提出不同学习率策略显著提升成功率；

**🔧 技术方法**

使用梯度流、Adam、快速读出权重、损失景观解析、两/单节点理论推导、实验对比、AUROC评估等技术；

**📊 数据集**

实验使用随机生成的高斯输入分布，维度D∈{2,4,8,16}，教师宽度M∈{2,4,8,16}，并在多种激活函数（ReLU、softplus、tanh）下训练；

**📈 对比分析**

通过在不同教师分布、过参数化比、学习率配置和激活函数下进行实验对比；结果显示最大不相似度分布成功率最高，最小分布最低，学习率差异可将成功率提升数倍；

**⚠️ 局限性**

局限于单隐藏层网络；理论分析主要针对ReLU；实验依赖随机初始化，未验证在实际任务中的泛化；对内部局部最小的预测仍不完善；

---

## 383. PRAGMA: Evaluating Personalized Guidance with Memory Alignment in Lifelong Conversations

**arXiv ID:** 2609.09664 | [PDF](https://arxiv.org/pdf/2609.09664v1)

**作者:** Hyojeong Yu `[一作]` (Seoul National University), Kyomin Jung `[通讯]` (Seoul National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了名为PRAGMA的基准，用于评估长期对话中个性化指导的能力，并在该基准上对检索增量生成（RAG）、多种检索技术以及内存系统进行了系统实验。

**💡 创新点**

创新点包括：①设计四种对齐/纠正查询类型（Event-Align、Event-Correct、Traj-Align、Traj-Correct）以覆盖事件级记忆与轨迹级记忆；②构建可控且人工验证的长期对话历史，配合细粒度证据标注，突出检索与基于记忆的推理耦合；③将检索与生成、内存结构三者在同一评测框架下统一比较。

**🔧 技术方法**

技术方面采用了Dense/BM25/Window检索、Query-Rewriting、Adaptive-K等检索策略；检索增量生成（RAG）与内存系统（A-MEM、Mem0、SimpleMem）相结合；评估使用GPT-5-mini和Qwen3-30B进行生成，并用LLM判定器根据对齐与基于证据的标准进行评分。

**📊 数据集**

使用数据集为PRAGMA，包含100个合成用户、400个指导查询（每种类型各100例），对话历史约160k tokens；数据通过Privasis-Zero用户信息生成，并在GitHub与HuggingFace上公开。

**📈 对比分析**

实验将模型在No-Context、Oracle-Session、Oracle-Summary、Full-Context等条件下进行对比，评估指标包括检索召回、精确召回、对齐分数与基于证据分数。结果显示，即便检索得到充分证据，模型在生成个性化指导时仍大多无法充分利用，尤其是纠正类查询的对齐和基于证据得分均远低于Oracle-Summary。

**⚠️ 局限性**

局限性包括：①对话历史为合成生成，缺乏真实人类对话中的模糊与多样性；②评测仅针对单轮生成，未覆盖多轮交互中的记忆恢复与对话迭代；③检索评估仅到会话级别，未细化每一步推理轨迹，可能忽略部分隐式证据。

---

## 384. ProbPlug: A Plugin Uncertainty Network for Reliable Confidence in LLM Binary Classification

**arXiv ID:** 2609.10122 | [PDF](https://arxiv.org/pdf/2609.10122v1)

**作者:** Jianzong Wang `[一作]` (Ping An Technology (Shenzhen) Company Limited), Yayun He `[通讯]` (Ping An Technology (Shenzhen) Company Limited)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ProbPlug，一个轻量级插件，用冻结 LLM 的隐藏层特征来估计二分类任务的置信度。

**💡 创新点**

创新点在于通过多层 Transformer 隐藏状态的自注意力聚合来生成置信度，而无需修改基模型或进行多次推理，且能在跨任务上保持良好泛化。

**🔧 技术方法**

使用了 token 压缩模块、层信息聚合模块（多头自注意力）和轻量化分类头，全部训练在冻结的 LLM 上。

**📊 数据集**

实验覆盖文本二分类数据集（SMS Spam、SST‑2、Toxic Comment、Civil Comments、Amazon Polarity）以及多模态情感识别数据集 IEMOCAP。

**📈 对比分析**

与 Verbalization、Logit、Self‑Consistency、CISC、SAPLMA 等方法对比，ProbPlug 在 F1、AUPRC、ECE、Brier Score 等指标上均优于或接近最佳方法，且只需单次推理，跨任务性能尤为突出。

**⚠️ 局限性**

局限性包括主要针对二分类任务，扩展到多分类需要多二分类拆解；对极端噪声或需要更细粒度置信度评估的场景仍有提升空间。

---

## 385. MetroLLM-Bench: Evaluating Language Models as Transit Kiosk Runtimes

**arXiv ID:** 2609.10016 | [PDF](https://arxiv.org/pdf/2609.10016v1)

**作者:** Remco Hendriks `[一作]` `[通讯]` (Continker), Remco Hendriks (Continker)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 MetroLLM-Bench 基准，包含 955 个案例，评估语言模型在地铁售票机策略层的表现，并通过对 26 种模型（包括 Qwen、GPT、Muse Glimmer 等）进行对比实验，展示了 4B Qwen 3.5 通过 PEFT 微调后在 Tier‑1 评分上超过 GPT‑5.6 且与 GPT‑5.4 全量模型持平。

**💡 创新点**

创新点在于：①设计专门针对地铁机顶端决策的基准，包含路由、票价、故障、无障碍、文化、政策等 11 类场景；②将 deterministic Tier‑1 与 semantic Tier‑2 两层评分分离，既可作为训练奖励，又能评估语义质量；③在同一任务上开展容量‑天花板实验，揭示 PEFT 在 4B 后收益递减甚至负向的规律。

**🔧 技术方法**

使用技术包括：ReAct 循环式工具调用、结构化工具（如路由、票价、规划、通知等）、QLoRA 低秩适配、Qwen 3.5 学生模型、GPT‑5.x API、Claude Haiku 4.5 评判器、Pydantic 验证终端状态、GPU 量化推理（Q4_K_M）以及多种服务器配置比较。

**📊 数据集**

数据集为自行构造的 955 条案例，覆盖 6 个真实地铁系统（MARTA、Doha、BART、Taipei、CTA、Beijing），包含 11 类场景；训练集 717 条、检验集 238 条，并通过系统分层 75/25 划分保证独立性。

**📈 对比分析**

比较方法：采用 75/25 随机划分的训练/检验集，使用基线规则化脚本（Tier‑1 84.6）和 26 种模型的单次或多次评估，计算 Tier‑1、Composite 两层得分；在 4B、9B、27B PEFT 之间绘制增益曲线。性能方面，4B PEFT 在 Tier‑1 上 91.32 分，超越 GPT‑5.6 tier（90.6/90.0）并与 GPT‑5.4 全量模型（91.37）相当；规则化基线仅 84.6。容量‑天花板实验显示 2B→4B 增益 +7，4B→9B +1.65，9B→27B 负向 -0.91。

**⚠️ 局限性**

局限性包括：①基准范围受限，无法评估更长时序或更大行动空间的模型；②统计噪声与配置敏感度导致小分数差异不显著；③评估主要基于 238 条 held‑out 集合，样本量有限；④模型对服务器配置高度依赖，导致跨代比较不完全公平；⑤PEFT 结果受 QLoRA 参数、训练样本规模、序列长度限制影响，未必适用于其它任务。

---

## 386. Scaling Post-Training Ternarisation to Qwen3-8B Capability Retention, Reproduction, Lossless Packing, and Packed Execution

**arXiv ID:** 2609.09240 | [PDF](https://arxiv.org/pdf/2609.09240v1)

**作者:** Anirudh Malik `[一作]` (OneBit AI), Poojith Devan `[通讯]` (OneBit AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Qwen3-8B模型进行后训练低比特（三值）量化，并验证其在能力、存储与直接执行三方面的可行性。

**💡 创新点**

首次在8B规模下完成完整的量化、无损压缩、直接打包执行流程，并通过外部再现门验证结果一致性；对比4B模型，提供匹配协议下的能力对比；详细记录有效比特率、存档大小与执行性能。

**🔧 技术方法**

使用KOTMS旋转、E2M-ATQ自适应三值化、GPTQ误差补偿、基于格点的无损序列化（lattice-aware packing）以及自研的packed GEMV内核。

**📊 数据集**

WikiText-2（校准与评测）、C4、PTB（困惑度评测）以及八个零样本任务集（ARC-Challenge、ARC-Easy、BoolQ、HellaSwag、LAMBADA-openai、MMLU、PIQA、WinoGrande）。

**📈 对比分析**

通过与TWLA公开参考结果对照（保持0.05点误差）、匹配4B/8B协议下的机会校正保留率（8B 78.5% vs 4B 69.6%，提升8.9点）、困惑度平均比值1.361×，以及直接打包执行在RTX 5070上实现15.52 tokens/s（占用7.35 GiB）与基准offload 1.48 tokens/s的对比；单个GEMV微基准显示packed核比FP16 cuBLAS慢4.6×。

**⚠️ 局限性**

仅用单一随机种子、仅量化线性投影、未量化嵌入/头/激活/缓存、GEMV内核为原型、在本地FP16基准与公开结果存在7.9%差异、未对所有竞争低比特PTQ方法做完全协议对比。

---

## 387. StreetDiff: Multi-view Street Scenes Generation via Cross-view Consistent Multi-view Stable Diffusion with Structure Prompts

**arXiv ID:** 2609.09890 | [PDF](https://arxiv.org/pdf/2609.09890v1)

**作者:** Qi Zhang `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出StreetDiff模型，能够根据文本和结构提示（分割图、轮廓图或用户草图）生成高度一致、真实感的多视角城市街景，并构建了新的Street360 HDR多视角街景数据集。

**💡 创新点**

创新点包括：① Panorama–Perspective Synergy框架，先用全景作为全局先验再引导视角生成；② Panorama Alignment Module（PAM）实现球面投影一致性注意力，解决跨视角对齐和循环一致性问题；③ 采用三阶段训练与一致性奖励微调（GRPO），显著提升多视角一致性与视觉质量。

**🔧 技术方法**

主要技术：Stable Diffusion U‑Net基础、ControlNet+Attention、LoRA微调、球面投影与交叉注意力（PAM）、三阶段训练策略、GRPO一致性奖励微调、使用BLIP‑3、GPT‑4生成文本提示、OneFormer生成分割图。

**📊 数据集**

使用Street360（10k HDR街景，4K–8K分辨率）作为主要训练/评测数据集，并结合CVRG‑Pano、Matterport3D、ScanNet、HDR360‑UHD等公开室内外数据做对比。

**📈 对比分析**

与MVDiffusion、PanFusion、SD+LoRA、Text2Light等SOTA方法对比。StreetDiff在FID、IS、CLIP Score、视角重叠PSNR等指标上均优于对手，用户评测显示>77%用户认为其在风格一致性、逼真度和多视角一致性上最佳。

**⚠️ 局限性**

局限性：目前仍主要依赖全景先验，难以处理极端极角或极端动态场景；对结构提示的质量要求高，若分割/轮廓错误会影响生成；在极高分辨率或复杂光照下性能尚需进一步提升。

---

## 388. Few Rows Tell Them Apart: Equivalence of Queries Mixing Set and Bag Semantics

**arXiv ID:** 2609.09978 | [PDF](https://arxiv.org/pdf/2609.09978v1)

**作者:** Sara Cohen `[一作]` `[通讯]` (Hebrew University of Jerusalem), Sara Cohen (Hebrew University of Jerusalem)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在有限大小数据库上检验SQL查询等价性的方法，并给出了可计算的上界，使得若两条查询在所有小于等于该上界的数据库上给出相同结果，则两查询在所有数据库上等价。

**💡 创新点**

提出了针对组合语义框架下的多重集合宽度和键宽度的指数上界，并将其扩展到含比较的查询类，首次实现了完全决策的有界搜索。

**🔧 技术方法**

利用组合语义框架、SQL的集合集与多重集合运算、键约束与外键结构、以及比较运算的重写技术，推导出指数上界并证明其完备性。

**📊 数据集**

论文未使用具体数据集，而是基于理论分析与形式证明。

**📈 对比分析**

通过理论证明而非实验比较，展示了在给定上界的数据库上完成搜索即可决定查询等价性，性能表现取决于查询大小与多重集合宽度的指数上界。

**⚠️ 局限性**

局限在于仅适用于组合语义框架下的某些查询类，且实际搜索成本可能随指数上界快速增长，未给出实际实现或性能评估。

---

## 389. TransGaze-Object: Transformer Based Driver Gaze Object Prediction Framework in Real Driving

**arXiv ID:** 2609.10139 | [PDF](https://arxiv.org/pdf/2609.10139v1)

**作者:** Pavan Kumar Sharma `[一作]` (Indian Institute of Technology Kanpur), Pranamesh Chakraborty `[通讯]` (Indian Institute of Technology Kanpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种端到端的驾驶员注视物体预测框架 TransGaze-Object，直接从驾驶员面部图像和交通场景物体信息中预测注视的交通物体或背景。

**💡 创新点**

创新点在于：①首次将注视物体直接作为输出；②通过 Transformer 跨注意机制将面部特征与场景物体空间特征融合；③构建了全新 Urban Driving‑Face‑Scene‑Gaze (UD‑FSG) 数据集；④提出混合损失函数（分类、一致性、混淆注意、硬负样本间距）提升模型鲁棒性。

**🔧 技术方法**

使用了 ResNet‑18 进行面部与眼部特征提取、YOLOv8 检测面部与交通物体、Gaussian 加权眼部特征、Transformer 编码器与跨注意机制、soft‑max 温度缩放和加权融合的注意力头，最后分类层输出注视物体类别。

**📊 数据集**

使用自行构建的 UD‑FSG 数据集，该数据集包含 35 名驾驶员的 373,488 张同步面部与场景图像、交通物体边框以及注视点和注视物体标签，覆盖高密度城市交通场景。

**📈 对比分析**

与基于点‑注视（PoG）估计后再关联物体框的 SGAP‑Gaze 方法相比，TransGaze‑Object 在测试集上的注视物体准确率为 59.45%（≈60%），高出 8.47%（相对提升 17.5%），并显著降低了背景与物体混淆错误率（从 23.21% 降至 11.68%）。

**⚠️ 局限性**

局限性包括：对小尺寸物体的预测仍易出错；模型仅考虑单帧，未利用时间序列信息；仅针对前视摄像头，未兼容不同摄像头配置；在真实多变场景下的泛化性能仍有待提升。

---

## 390. Minimal Deadlock-Free Routing for Degree-Six Triangular-Lattice Meshes and Tori with Two Forbidden Turns

**arXiv ID:** 2609.09746 | [PDF](https://arxiv.org/pdf/2609.09746v1)

**作者:** Zibo Diao `[一作]` (Tsinghua University), Rongxi Sun `[通讯]` (Tsinghua University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了两个死锁无关的最短路径适配路由方案，分别用于有限度六方向三角格子网（mesh）和其周期环（torus），并给出了单链路失效下的容错扩展。

**💡 创新点**

创新地统一六向坐标体系；证明mesh仅需禁止两转向并使用单虚拟通道即可保证死锁自由；揭示torus单VC会产生CDG环，提出两VC+Hamiltonian dateline的全局排序来消除环；在单链路失效时通过旋转转向规则和同组三角绕行保持最短+1 hop；提供完整的CDG验证与实现。

**🔧 技术方法**

采用代数坐标、相邻方向扇区、线性势能函数、Hamiltonian坐标、虚拟通道分配、CDG检查器、gem5/Garnet仿真、随机/信用/固定路由策略等技术。

**📊 数据集**

使用gem5的Garnet合成流量，采用全节点对列表均匀分布的流量；对不同尺寸（n=4,8,12）和多种注入率、内部/边界扇区、数据线heavy等场景进行实验。

**📈 对比分析**

通过与传统13×13 Mesh_XY基准进行零负载时延、平均跳数、最大吞吐量等指标对比；结果显示HexMesh在零负载时延约1.18×、吞吐量约1.33×于Mesh_XY；HexTorus零负载更低但高负载吞吐略低。内部扇区流量延迟低于边界扇区；数据线heavy显著提高VC1使用率与停滞。规模扩展从n=4到12保持理论趋势。

**⚠️ 局限性**

仅考虑单静态双向链路失效，未覆盖多故障、节点故障、动态拓扑变化；实现要求按hop的VC分配，单包VC不可行；评估仅在合成流量，未涉及应用级性能、功耗、时序；未证明两VC为必要条件。

---

## 391. NEXUS-MI: Communication-Aware Federated Personalization for Gateway-Coordinated Motor-Imagery Brain-Computer Interfaces

**arXiv ID:** 2609.09786 | [PDF](https://arxiv.org/pdf/2609.09786v1)

**作者:** Daniel Adu Worae `[一作]` (University of Notre Dame), Aarthy Nagarajan `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 NEXUS-MI 框架，通过网关协同的联邦学习实现基于 EEG 的运动意象脑机接口的个性化解码。

**💡 创新点**

创新点在于将网关同步视为可调的设计变量，构建了多维通信策略空间，并在此框架下系统评估同步策略对解码精度、通信成本与用户可靠性的影响。

**🔧 技术方法**

采用了联邦学习（FedAvg）、EEGNet 主干网络与个性化分类头、以及调度、缓冲、旧版更新处理与下载控制等通信自适应技术。

**📊 数据集**

实验使用了公开的 BCICIV‑2a（9 受试者，4 类）和 OpenBMI（54 受试者，2 类）EEG 数据集，按会话划分进行联邦训练与本地头部微调。

**📈 对比分析**

通过与理想链路基准以及六种异构通信策略（P1–P6）的对比，P5 在保持 41% 服务器↔客户端模型下行流量降低的同时，在 OpenBMI/EIB‑PH 场景下平均提升约 1% 的个体解码准确率，其他场景准确率波动不显著；对比进一步在多次实验和不同连接可用性下验证了通信成本与性能的稳健性。

**⚠️ 局限性**

局限性包括仅在实验室数据上评估，未涉及真实家用部署与移动网络环境；受试者规模和任务类型有限；仅改进主干网络，未考虑完整端到端解码链路；并未针对不同用户群体的长期稳定性和安全隐私进行深入分析。

---

## 392. Multi-Agent Agentic Graph Learning via Structural Signatures

**arXiv ID:** 2609.09565 | [PDF](https://arxiv.org/pdf/2609.09565v1)

**作者:** Liang Qu `[一作]` (Edith Cowan University), Hua Wang `[通讯]` (Victoria University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出MAAGL框架，通过把图划分为社区为每个社区分配独立的LLM代理，实现区域化的图推理；

**💡 创新点**

创新点在于：①使用可置换的结构签名动态概括邻域结构，避免邻居排序带来的不确定性；②将结构与语义证据分离，仅保留与目标最相关的k条文本；③利用历史轨迹估计代理置信度并触发有限的协作讨论；④通过经验学习提炼可复用的结构-策略关联；

**🔧 技术方法**

技术包括：大型语言模型驱动的推理代理、Leiden社区划分、结构签名统计、语义相似度检索、协作辩论式多轮推理、经验验证与共享；

**📊 数据集**

实验使用四个文本属性图数据集：OGB‑Arxiv、Cora‑full、OGB‑Products、Amazon‑Computers；

**📈 对比分析**

与传统GNN、单代理AGL及协调式多代理方法相比，MAAGL在所有四个数据集的准确率和宏F1均显著提升，并在零样本迁移中表现优异；

**⚠️ 局限性**

局限在于结构签名维度手工设定，缺乏自动发现或任务特定自适应维度的机制。

---

## 393. Hybrid Quantum-Classical NLP Classification with Compact Semantic Representations: An Experimental Analysis of Representation Compression

**arXiv ID:** 2609.10089 | [PDF](https://arxiv.org/pdf/2609.10089v1)

**作者:** Ali Hassan `[一作]` (German University in Cairo), Maha A. Metawei `[通讯]` (Electronics Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了将预训练句子嵌入通过降维压缩为可直接量子编码的低维表示，并在此基础上使用变分量子电路进行分类的混合量子-经典NLP流水线。

**💡 创新点**

创新点在于系统比较了无监督的PCA与监督的LDA/NCA降维方法对量子兼容表示的影响，证明监督降维能在仅5维下保持甚至超越完整维度的分类性能，并强调了无泄漏交叉验证的必要性。

**🔧 技术方法**

使用了MiniLM/句子BERT预训练嵌入、PCA、LDA、NCA三种降维技术、角度编码到5个量子比特、硬件高效的两层变分量子电路以及经典逻辑回归/多层感知机决策层。

**📊 数据集**

实验数据集包括TREC六分类问答数据集以及小规模餐厅情感分类数据集。

**📈 对比分析**

通过对比不同维度下的准确率，监督降维在5维时即可达到85.3%/83.1%的准确率，超过PCA在同维度下的57.9%并与完整384维基线的85.1%相当；无泄漏验证下性能下降显示泄漏风险。

**⚠️ 局限性**

局限性在于量子比特受限、噪声影响、实验仅在模拟器上完成、仅考虑线性降维、未展示真实硬件上的量子优势，且数据集规模有限。

---

## 394. Contrastive Projection: Reading Transformer Internals by Differencing Logit Lenses

**arXiv ID:** 2609.09902 | [PDF](https://arxiv.org/pdf/2609.09902v1)

**作者:** Olli Tuomi `[一作]` `[通讯]` (Evident Solutions Oy), Olli Tuomi (Evident Solutions Oy)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种“对比投影”方法，用差分隐藏状态再通过 logit 透镜投影到词表空间，从而读取两条输入序列在 Transformer 计算中所产生的差异。该方法实现了逐位置、逐子层、逐头的系统化追踪，并通过多重对比三角化、对比设计规则以及激活补丁等技术，验证了其在多模型、多任务（复合名词、隐喻、记忆与幻觉对比）上的有效性。

**💡 创新点**

创新点包括：
1) 将对比激活差分与 logit 透镜结合，构建零训练的“对比投影”读取器；
2) 统一的系统化追踪框架（位置、子层、头部），可实时定位信息流和写入点；
3) 多重对比三角化（multi‑contrast triangulation）和基准差分（baseline subtraction）实现对特定语义轴的聚焦；
4) 对比设计规则（共享前缀、匹配当前词、匹配下一词）提升可读性；
5) 在不同模型架构（Phi‑2、Pythia‑1.4B、Qwen2.5‑1.5B）及不同随机种子上验证通用性。

**🔧 技术方法**

技术手段：
- 通过 `h_c[L] - h_k[L]` 计算差分隐藏状态；
- 通过 `W_E^T`（unembedding）投影到词表空间；
- 逐位置/子层/头的读取与统计；
- 多重对比三角化与基准差分平均化；
- 激活补丁（activation patching）验证因果路径；
- 注入差分向量测试方向性与可解释性。

**📊 数据集**

使用的数据集与模型：
- 语言模型：Phi‑2（2.7B）、Pythia‑1.4B、Qwen2.5‑1.5B；
- 复合名词、隐喻、记忆与幻觉等对比示例（如 "hot dog / cold dog"、隐喻词对（cold/bright/…）与实体召回对比）；
- 5 个 Pythia‑410M 的随机种子版，用于跨种子对比。

**📈 对比分析**

比较方法与结果：
- 对比投影与传统单句 logit 透镜对比，后者往往被高频函数词淹没；
- 在三种模型中，所有模型的预测层都能读取到目标语义（如食物 vs 动物），但词面表现因模型而异；
- 在不同随机种子上，语义轴保持正向一致，但具体词面上差异巨大（Jaccard 0.08）；
- 通过多重对比三角化，Pythia 的名词层面读数从无到可读，Qwen 从 60 位提升到 20 位；
- 通过注入实验验证隐喻方向的可塑性和跨域效果；
- 整体性能：能在多模型、多层次上稳定读取语义差异，但对词面解读的鲁棒性受限。

**⚠️ 局限性**

局限性：
- 仅能读取投影到词表的内容，未捕捉未投影的内部结构；
- 需要精确对齐的对比对，且对首词差异敏感；
- 读取器基于无训练的投影，难以自动判定可读性（仅靠经验阈值）；
- 只在少数模型与任务上验证，未覆盖所有 Transformer 变体；
- 省略 LayerNorm 的影响导致不同层的差分幅度不可比；
- 结果多为定性解释，缺乏严格的量化因果证据；
- 词面 token 的差异可能是网络特定的，导致跨网络可比性受限。

---

## 395. LexAgentHallu: A Hierarchical Benchmark for Profiling Hallucinations in Legal Agents

**arXiv ID:** 2609.09754 | [PDF](https://arxiv.org/pdf/2609.09754v1)

**作者:** Yujin Zhou `[一作]` (Hong Kong University of Science and Technology), Sirui Han `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了LexAgentHallu，一个面向多步法律LLM代理的幻觉评估基准，提供双层细粒度幻觉分类和轨迹级评价指标。

**💡 创新点**

创新点在于首次结合法律与代理层面的双层分类、专家循环的四阶段构建流程、以及基于轨迹的RAWR等细粒度指标，揭示幻觉聚类和框架/任务特征。

**🔧 技术方法**

采用多模态LLM推理框架（ReAct、LawThinker、Plan-and-Execute）与大型语言模型（LegalDelta、Qwen、Gemini、GPT‑5.4等）以及LLM‑as‑a‑judge（Claude‑4.6‑Sonnet）进行评估与标注。

**📊 数据集**

数据来源于中国法律基准（LexEval、LawBench、UniLaw、DISC‑Law‑Eval、PLawBench）与全国司法考试样本，共3414条实例，并配套细粒度检查清单。

**📈 对比分析**

通过18种专有与开源代理的对比实验，发现幻觉普遍存在，RAWR高达60%以上，框架和模型规模显著影响性能，显示出多步评估的必要性。

**⚠️ 局限性**

局限性包括仅覆盖中国法域、随着LLM演进可能出现新模式、仅评估单一模型且未涉及多模型协同或跨司法系统。

---

## 396. MuJoCable: Reduced-Order Surface-Routed Cable Transmission for Tendon-Driven Robots

**arXiv ID:** 2609.09612 | [PDF](https://arxiv.org/pdf/2609.09612v1)

**作者:** Yi Zhang `[一作]`, Yue Xie `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 MuJoCable 插件，向 MuJoCo 添加了可配置的、由表面路由决定的短杆缆传输模型，并实现了连续滑动摩擦、单向张力及节点虚功映射。

**💡 创新点**

创新点在于将多表面动态路由、单向张力、方向性 Capstan 摩擦传播和节点虚功耦合统一到 MuJoCo 的仿真框架中，使缆传输的几何、摩擦与力学可以在仿真前直接设计与优化。

**🔧 技术方法**

技术主要包括：基于优化的多表面路由算法、单向轴向弹性律、方向性 Capstan 摩擦传播、节点虚功映射到刚体动力学以及 MuJoCo 引擎插件实现。

**📊 数据集**

使用的数据主要是合成的 7 组滑轮基准（用于验证闭式解析关系）、18 轴 SpiRobs 平台以及一款基于缆路由的无机构手指进行硬件验证，未使用公开数据集。

**📈 对比分析**

通过与解析滑轮模型对比（误差低于 0.5%）、与原生 MuJoCo 软绳对比（运动相近但张力差异显著）以及与硬件实验对比，验证了插件在传输精度、能量损失与运动分布上的准确性；插件增加的仿真步骤时间约为 22.5%，仍能保持 11.9 倍的实时率。

**⚠️ 局限性**

局限性在于：假设缆路由拓扑固定，无法自动切换或重连；不考虑缆的质量、弯曲、扭曲等分布式状态；对脱轨、再接合等动态拓扑变化的处理尚未实现。

---

## 397. PrivAudit: A Dual-Lens Auditing Framework for Website Privacy Practices under the CCPA

**arXiv ID:** 2609.09697 | [PDF](https://arxiv.org/pdf/2609.09697v1)

**作者:** Mohamed Moustafa Dawoud `[一作]` (University of California, Santa Cruz), Ram Sundara Raman `[通讯]` (University of California, Santa Cruz)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个自动化的双视角网站隐私实践审计框架PrivAudit，结合大语言模型解析隐私政策和浏览器测量Cookie写入；

**💡 创新点**

首次在大规模样本上同时评估隐私政策披露与前端跟踪行为，揭示政策与行为之间的明显脱节；

**🔧 技术方法**

采用GPT‑4o等LLM进行政策评分与自然语言解释，并使用Puppeteer+Chrome在六种隐私配置下自动化抓取Cookie；

**📊 数据集**

使用已标注的602个CCPA适用网站、396个非适用网站以及1,000个Tranco热门站点的URL集合；

**📈 对比分析**

对比监管与非监管网站在披露完整性、可操作性和Cookie数量上的差异，结果显示监管网站披露更完整但第三方跟踪并未显著降低；GPC信号可减少约一半Cookie但未完全奏效；

**⚠️ 局限性**

局限在于仅测量首页与Cookie，未覆盖登录后或深层页面的跟踪；LLM可能出现幻觉；仅关注Cookie，未涵盖指纹等其他追踪手段。

---

## 398. Through the Looking Glass: Directly Reading and Writing Transformers

**arXiv ID:** 2609.10210 | [PDF](https://arxiv.org/pdf/2609.10210v1)

**作者:** Mark Oskin `[一作]` `[通讯]` (University of Washington), Mark Oskin (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过一种基于模型自身权重与激活的无监督“镜头”（lens）方法，定位并量化 Transformer 在生成单个 token 时真正起决定作用的少量组件。它定义了必要组件（若移除会改变预测）、足够组件（仅保留即可重现预测）和剩余组件的功能；并进一步探讨了这些组件的可读性（能否给出语义化说明）和可编辑性（对模型行为的可控修改）。

**💡 创新点**

创新点包括：
1) 将“负载”（负向贡献）与正向贡献分离，发现大部分传统归因方法会把相互抵消的信号错误计入答案，真正的决定因素只有几十个。 
2) 引入“必要‑足够”框架，给出对每个预测的最小实现集；发现仅 8–53 组组件即可完整重现预测。 
3) 通过直接解码层内 token 表（layer‑native token table）对组件读取侧进行无训练的解释，验证其与实际驱动行为的一致性。 
4) 在无训练、无拟合的前提下实现可编辑操作（增益、关联插入、注意力通道编辑），展示编辑效果与标准 rank‑one 编辑的比较。 
5) 通过 12 个不同规模、架构和激活函数的预训练模型验证结论的普适性。

**🔧 技术方法**

技术手段主要包括：
- 直接计算每个 feed‑forward 单元和注意力通道对目标 logit 的贡献 κ_u；
- 通过正负分离（net vs abs）筛选贡献大小，得出必要、足够集合；
- 使用层本地 token 表 Ẽ[l,t] 解码组件读取侧，计算参数与激活一致性（ov@K）；
- 采用“修复”（repair）算法在保持某些组件的前提下补全输入，寻找足够集；
- 通过增益（α）放大单元激活、rank‑one 关联插入（键/值直接写入稀疏单元）和注意力通道放大编辑，实现可编辑性测试。

**📊 数据集**

主要数据集：
- OpenWebText（用于训练 GPT‑2 小模型）。
- 12 个公开预训练模型（GPT‑2、OPT、SmolLM2、Qwen2.5、Gemma‑3、OLMo‑2、Llama‑3.2、TinyLlama、Mistral），规模从 1.24 亿到 70 亿参数。 
- 评估数据采用 held‑out 文本与 LAMBADA 任务，模型输出被评为 Top‑1 预测概率 ≥ 0.3 的样本。

**📈 对比分析**

比较方法：
- 与传统绝对值归因方法（计数 3k–200k 组件）对比，显示传统方法误把相互抵消的信号计入答案。
- 与 12 个不同模型、不同激活函数的结果对比，验证必要‑足够集合大小在 2–16 组件之间波动；
- 在可编辑性实验中，将增益编辑与 rank‑one 编辑、注意力通道编辑的损失、rank 改变、以及 held‑out loss 进行量化对比。 
- 性能方面：仅 8 个组件即可完整复现 115 条预测；必要组件平均 53 个；足够集合平均 8–16 个；对剩余组件的影响仅为 “架构管理” 作用，几乎不改变后续 token 预测。

**⚠️ 局限性**

局限性：
1) 归因直接以写入为准，忽略只通过后续组件影响的间接贡献。 
2) 组件单元定义（head vs channel）会影响计数结果。 
3) 结论主要基于 GPT‑2 小模型及其衍生模型，虽然在 12 个模型上验证，但在更大规模或不同结构（如 GPT‑NeoX）上的适用性未证实。 
4) 必要‑足够集合是通过启发式搜索得到的上界，最小实现集可能更小。 
5) 可编辑实验使用的“稀疏单元”方式仅演示单一示例，未覆盖更复杂的编辑需求。

---

## 399. Direct Diversity Optimization for Diverse Successful Trajectories in Preference Post-Training

**arXiv ID:** 2609.10052 | [PDF](https://arxiv.org/pdf/2609.10052v1)

**作者:** Junwon Ko `[一作]` (KAIST), Junmo Kim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Direct Diversity Optimization (DDO) 的离线后训练方法，用于让大型语言模型（LLM）代理在仅拥有轨迹级成功/失败标签的情境下，保留并保持多条成功策略；

**💡 创新点**

创新点在于将Divergence-Tree Collection (DTC) 与 Reference-Relative Target-Odds Objective (RTO) 结合，通过构造同一决策状态下的多分支集合并对成功分支进行参考相对目标赔率的匹配，从而将粗糙的轨迹级监督转化为细粒度的同状态分支对比监督；

**🔧 技术方法**

主要技术包括DTC用于恢复决策状态、采集同状态下的替代分支并标注结果；RTO定义参考相对目标分布，并通过soft logistic目标对模型的log-odds进行匹配；实验中还使用了标准的DPO、DivPO、TieDPO等后训练基线；

**📊 数据集**

使用了三个基准数据集：BabyAI、BabaIsAI 和 WebShop；

**📈 对比分析**

与DPO、DivFreq、DivProb、TieDPO-RK/Dav以及成功仅模仿和解码时多样化控制进行对比，DDO在任务成功率和成功策略覆盖率（ESD、H-ESD）上均优于所有基线，平均提升约10.5个百分点的成功率、0.08-0.09的覆盖率，并在局部行动替换实验中达成最高的恢复率（约75.2%）；

**⚠️ 局限性**

该方法仅适用于可精确重建状态并能在同一决策状态下执行多分支的离散环境，无法直接应用于连续控制、部分可观测或随机动态、多模态观测以及真实世界中产生不可逆外部副作用的任务。

---

## 400. From State Synchronization to Cognitive Self-Evolution: An Operational Architecture for Cognitive Digital Twins

**arXiv ID:** 2609.09625 | [PDF](https://arxiv.org/pdf/2609.09625v1)

**作者:** Haoran Gao `[一作]` (Concordia University), Jun Cai `[通讯]` (Concordia University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了四层自演化闭环认知数字孪生（CDT）架构，包括物理层、数字孪生层、认知层和任务层，并阐述了基于用户请求驱动和自驱动两种任务启动模式。

**💡 创新点**

创新点在于将认知能力嵌入数字孪生核心，形成完整的认知闭环；提出任务感知的状态组织、语义通信、跨维度资源协同与隐私保护等关键机制，并实现了自驱动任务生成与经验反馈驱动的自演化。

**🔧 技术方法**

采用任务感知一致性与协调、语义表示与映射、跨维度优化、保密性知识交换、跨层闭环协同设计以及基于契约的激励机制；实现层面使用TCN实现数字孪生时序映射、RGAT从知识图获取认知模型、MLP用于任务候选评估与决策。

**📊 数据集**

使用UCI公开的PAMAP2和WESAD可穿戴生理数据集，构建多变量状态空间进行闭环仿真。

**📈 对比分析**

通过与无邻域知识的CDT-NA和仅有静态知识库的DT-S两种基线对比，评估可行域命中率和搜索开销。结果表明，CDT在低语义比例下仍能保持高命中率，且自驱动模式对语义信息更鲁棒；经验积累显著降低搜索开销，验证了自演化效能。

**⚠️ 局限性**

局限性包括对通信与传感不确定性建模不足、对安全边界的安全性分析缺失，以及在大规模部署时知识图分区、跨边缘一致性与编排开销等可扩展性挑战。

---

## 401. Settling: Equilibrium Inference for Non-Convex Validity Sets

**arXiv ID:** 2609.09682 | [PDF](https://arxiv.org/pdf/2609.09682v1)

**作者:** Lyes Saad Saoud `[一作]` `[通讯]` (Independent Researcher), Lyes Saad Saoud (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了Settling推理框架，用以解决在非凸有效集下平方损失点估计导致的条件均值坍塌问题。

**💡 创新点**

将推理分解为提议生成、约束评估和平衡选择，通过动力学迭代收敛到局部稳定的可行解，从而避免平均化导致的无效输出。

**🔧 技术方法**

采用梯度下降的能量函数平衡、确定性迭代收敛分析、局部梯度误差鲁棒性证明，并在可解析几何诊断中实现七点样条动力学。

**📊 数据集**

使用100个随机障碍位移的几何测试场景（A‑to‑B轨迹），未涉及真实高维数据集；语义与传感融合示例为概念性可视化。

**📈 对比分析**

与均值聚合、随机去噪和直接能量下降等基准对比，Settling在99/100情境下成功且轨迹光滑度显著优于随机去噪；随机去噪100/100成功但不够平滑。

**⚠️ 局限性**

仅在可解析几何实验中验证，未证明在学习到的高维环境下能保持局部梯度对齐和基底几何；需要进一步评估学习提议和能量网络的泛化与推理成本。

---

## 402. Storage-Scalable Progressive Semantic Communication via Knowledge-Base Reuse

**arXiv ID:** 2609.10112 | [PDF](https://arxiv.org/pdf/2609.10112v1)

**作者:** Heng Zhu `[一作]` (Nanjing University of Aeronautics and Astronautics), Feifei Song `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了存储可扩展的知识库重用量化方案SSKBQ，用于多阶段逐步语义通信，解决传统多知识库方案的存储扩展瓶颈。

**💡 创新点**

通过重用有限数量的知识库实现多阶段逐步量化，并引入阶段感知残差监督机制，既保持了逐步细化的能力，又显著降低了KB存储需求。

**🔧 技术方法**

采用端到端语义编码/解码、向量量化、知识库重用、残差监督损失以及PSNR/SSIM/FID/KID等指标进行训练与评估。

**📊 数据集**

使用Cityscapes与COCO数据集的128×64像素图像进行实验。

**📈 对比分析**

与SKBQ、MKBQ、JSCC、VQVAE（Gumbel‑Softmax/One‑Hot）等基线对比，SSKBQ在16个传输阶段实现最高PSNR（Cityscapes 27.36 dB，COCO 23.53 dB），存储和传输比率大幅下降，仅为JSCC的1%，且在多数指标上表现更优。

**⚠️ 局限性**

在复杂多样数据集（如COCO）下仍不如直接传输JSCC，且实验仅在无噪声索引传输环境下进行，未考虑信道噪声、丢包等实际通信情形。

---

## 403. A Function-Space Approach to the Statistical Mechanics of Learning Dynamics

**arXiv ID:** 2609.09589 | [PDF](https://arxiv.org/pdf/2609.09589v1)

**作者:** Yizhou Zhang `[一作]` (Variational AI), Zhengjie Miao `[通讯]` (Simon Fraser University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种直接在函数空间中描述神经网络学习的统计力学框架，探讨了深度神经网络在高度非线性动态下的宏观行为规律。

**💡 创新点**

创新点在于将学习的统计力学描述直接建立在函数空间层面，而不是从参数空间出发，强调了参数配置作为微观实现与函数及其演化算子作为宏观变量之间的分离。

**🔧 技术方法**

使用了统计力学的方法，特别是条件自由能的概念，结合了动态算子和微观状态几何的相互作用。

**📊 数据集**

研究中未明确提及具体的数据集，但讨论了ReLU类型的函数空间，暗示可能使用了与深度学习相关的标准数据集。

**📈 对比分析**

通过比较动态几何M与统计几何B的相互作用，发现当[M,B]=0时，条件自由能达到极小值，且在固定谱情况下，较大的特征值与较小的微观状态曲率特征值配对，从而提供了局部恢复力。

**⚠️ 局限性**

限制在于假设了一个各向同性的随机源，并未考虑更复杂的噪声几何，此外，条件自由能的贡献并未涵盖M的完整慢动态，未来的工作需要探讨这些限制的扩展。

---

## 404. A Note on the Point-Clothoid Distance Algorithm

**arXiv ID:** 2609.10179 | [PDF](https://arxiv.org/pdf/2609.10179v1)

**作者:** Haibin Ye `[一作]`, Gong Cheng `[通讯]` (Tongji University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

重新分析点-克洛多伊距离算法，证明在不含拐点、转角不超过 2π 的段内平方距离函数最多有三 stationary 点，其中可出现两个局部最小，并据此优化候选点选择逻辑。

**💡 创新点**

提出以演化曲线切线相交为依据的三 stationary 点上限证明，消除了原先单一局部最小假设，证明 min–max–min 结构，推导出在无端点搜索时无需中点候选的完整性。

**🔧 技术方法**

采用几何分析（演化曲线分段、凸性证明）与极值点对应性技术，并通过向量化数值实验验证迭代次数与评估时间。

**📊 数据集**

使用 1000×1000 网格查询点覆盖特定无拐段，参数 k=1/2、k₀=√(π/2)，不同转角 Δθ=nπ/3（n=1,2,4,5）。

**📈 对比分析**

与原算法（包含中点候选）在迭代次数和评估时间上对比，发现去掉中点可节省 3.8%–12.3% 迭代次数、60.9%–70.3% 评估时间，最优情况下 88.11% 迭代次数下降。

**⚠️ 局限性**

仅适用于无拐点、转角≤2π 的段；对包含拐点或更大转角的段仍需研究；数值实验未覆盖所有异常情况；中点仍保留为数值回退。

---

## 405. Vision-language models know more about agriculture than they show and rubric-grounded verifications close the gap

**arXiv ID:** 2609.09417 | [PDF](https://arxiv.org/pdf/2609.09417v1)

**作者:** Earl Ranario `[一作]` (University of California, Davis), Urmil Jatin Chandarana `[通讯]` (University of California, Davis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了包含116个数据集、834类、8324张图像的农业视觉语言模型基准，评估 VLM 在疾病、虫害、损伤、质量和物种识别等零样本任务的表现；

**💡 创新点**

创新点在于证明视觉表征已足够，通过参考描述上限和诊断式结构化推理结合 Probabilistic Pivot Tournament 验证器，展示在无参考描述情况下可超越传统上限，提升 F1 近乎翻倍；

**🔧 技术方法**

采用的技术包括 VLM（Gemma、Qwen）与自监督视觉编码器（DINOv3、YOLOv11）线性探测、参考描述生成、固定任务诊断 rubric、Probabilistic Pivot Tournament 验证器及细粒度评分；

**📊 数据集**

使用的公开数据集来自 AgML 公开农业图像数据集集成，覆盖 116 个子集、834 类、8324 张图像；

**📈 对比分析**

通过比较无参考（下限）、参考描述上限以及 Rubric+PPT 的 F1_macro，发现 VLM 视网膜已足够，Rubric+PPT 在多任务上将 F1 近乎翻倍，部分任务甚至超过上限，例如 Gemma4 E4B 在疾病任务上达到 0.71；

**⚠️ 局限性**

局限性在于验证器的字母尺度评分与置信度负相关，难以作为不确定性指标，并且无法精准区分提升主要来自生成还是选择，需改进评分机制。

---

## 406. SymbolicLight V2: Hybrid Neuromorphic Architecture and Sparse Execution for Low-Energy Language Inference

**arXiv ID:** 2609.09772 | [PDF](https://arxiv.org/pdf/2609.09772v1)

**作者:** Ting Liu `[一作]` `[通讯]` (SymbolicLight Research), Ting Liu (SymbolicLight Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现并评估SymbolicLight V2混合神经形态语言模型，利用稀疏事件驱动推理在FPGA（Alveo U50C）和ARM Cortex‑A76上显著降低能耗并提升吞吐量。

**💡 创新点**

创新点包括：① 将有符号分级事件编码扩展到投影和注意力路径，实现可执行稀疏性；② 采用主动行权重收集与部分KV状态加载，消除零激活导致的内存访问；③ 用ReLU‑L₁本地注意力取代softmax，减少指数运算；④ 结合持续残差与事件化循环，形成完整的混合神经形态架构。

**🔧 技术方法**

技术手段：整数化（INT8）固定点运算、事件编码与量化、主动行权重收集、部分KV加载、Alveo U50C FPGA实现、ARM NEON向量化、ReLU‑L₁注意力、ALiBi线性位置偏置、固定点归一化与门控。

**📊 数据集**

数据集：训练使用约4B词汇量的开放式语料（48K分词器），在194M参数的V2检查点上进行推理评估，未使用特定公开数据集名称。

**📈 对比分析**

比较方法：在相同检查点下，采用三阶段FPGA实现（全枚举→主动行收集→部分KV加载）与ARM CPU、RTX 5090 FP32、低精度FP16/INT8 GPU进行对比；测量吞吐量（tokens/s）、总能耗（J/token）和增量能耗。结果显示：FPGA在p32/n128情境下吞吐量提升35.5%，能耗下降27.6%；GPU能耗比FPGA高89%，ARM在9.8 W时吞吐65 tokens/s。

**⚠️ 局限性**

局限性：① 检查点质量低于同等预算的稠密模型；② 能耗测量受设备边界（板卡、适配器、主机）差异影响；③ 仅评估贪心解码与短上下文；④ 未在所有平台上统一精度与功率传感器校准；⑤ 仅针对单一FPGA/ARM实现，缺乏广泛的跨硬件验证。

---

## 407. Applying foundation model embeddings towards urban livability evaluation

**arXiv ID:** 2609.09429 | [PDF](https://arxiv.org/pdf/2609.09429v1)

**作者:** Ayush Khot `[一作]` (University of Illinois at Urbana-Champaign), Shaowen Wang `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用地理空间基础模型嵌入（AlphaEarth、AnySat、TerraMind）与多模态Transformer框架相结合，系统评估其对城市宜居性多维度预测的贡献，并通过注意力熵与Full Grad‑CAM等可解释性方法深入分析模型对空间信息的整合与偏好。

**💡 创新点**

①提出将高维嵌入通过卷积提取后注入Transformer编码器的融合策略；②对单一与组合嵌入在宜居性预测中的相互补充性进行全面对比；③结合注意力熵与Grad‑CAM，首次在宜居性任务中对嵌入与传统遥感、文本模态的权重与空间聚焦进行可视化解释。

**🔧 技术方法**

多模态Transformer回归（TMTMR）框架、预训练DenseNet、BERT、卷积嵌入提取、Transformer注意力机制、注意力熵分析、Full Grad‑CAM、MAE多任务损失。

**📊 数据集**

荷兰Leefbaarometer v3.0（100 m × 100 m网格）数据集，配合SuperView 2 m多光谱影像、SDGSAT‑1 Glimmer 10 m夜光、PDOK 0.5 m DSM、POI、AlphaEarth、AnySat、TerraMind三种基础模型嵌入。

**📈 对比分析**

采用RMSE评估六个宜居性维度（整体、PHY、NUI、SOC、AME、HOU），与仅使用四种传统模态的基线TMTMR模型对比。结果显示：AlphaEarth嵌入显著降低RMSE（尤其是PHY与AME），多嵌入组合（aef+as+tm）在农村地区能恢复并提升基线性能；在缺失RS、DSM、NLRS或POI等模态时，嵌入能显著缓解性能下降，RS缺失仍是最难替代的模态。

**⚠️ 局限性**

研究仅在荷兰数据上验证，缺乏跨地区或跨文化的泛化评估；基础模型训练以城市数据为主，导致农村地区的表现不稳定；未对模型可能产生的社会不公平偏差进行量化评估；以及未覆盖时间演化或多时相的宜居性变化。

---

## 408. Sound Debloating of Redundant Checks in Zero-Knowledge Machine-Learning Circuits

**arXiv ID:** 2609.10149 | [PDF](https://arxiv.org/pdf/2609.10149v1)

**作者:** Zhantong Xue `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种对零知识机器学习电路中冗余检查进行安全削减（debloating）的技术，显著减小约束数量并提升证明效率。

**💡 创新点**

创新点在于将全电路抽象解释与证明来源图（provenance graph）相结合，能够在全局层面识别并安全移除冗余检查，并通过图搜索保证移除后仍保持完整的 witness 集。

**🔧 技术方法**

主要技术包括：多域抽象解释（区间、已知位、常量）；构建证明来源图记录每个事实的产生路径；在图上进行 AND/OR 证明搜索以避免循环推理；以及贪心的移除循环与优先级排序。

**📊 数据集**

实验使用了多种神经网络模型（MLP、CNN、RNN、Transformer）并通过两个 ZK‑ML 框架（ezkl 与 zkml）生成约束系统，约束规模从几千到超过 2500 万。

**📈 对比分析**

与 Circom、Distilling、CirC、Clap、Halo2 分析器以及精确 SMT 进行对比，本文工具在所有基准上实现了 3.2%–48.7% 的约束削减，平均约 25.5%，并将证明时间平均降低 33.4%，最高可达 72.8%。

**⚠️ 局限性**

局限性包括：分析不完整，无法发现所有可裁剪的冗余检查；仅支持已实现的约束类型与抽象域；在极大模型（>10^8 约束）上仍受内存/时间限制；且仍需手动扩展规则以覆盖更多电路结构。

---

## 409. Orukeet: Multilingual ASR with Frozen Gabor Kernels

**arXiv ID:** 2609.10054 | [PDF](https://arxiv.org/pdf/2609.10054v1)

**作者:** Nathan Roll `[一作]` (Oruk Ai), Calbert Graham `[通讯]` (Oruk Ai)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 NVIDIA 的 Parakeet 语音识别模型中，将一半时间滤波器替换为 12,288 个经过拟合的 Gabor 核，固定后训练其余参数。

**💡 创新点**

创新点是将可学习滤波器部分转化为解析的 Gabor 函数，既保留原架构又提升多语言性能。

**🔧 技术方法**

使用 Gabor 拟合、参数冻结、教师匹配、时间转导器训练以及 AdamW 优化。

**📊 数据集**

使用 LibriSpeech、FLEURS 以及 47 个分区的 EuroSpeech、GigaSpeechBench、Monsoon 等多种多语言、多口音数据集。

**📈 对比分析**

在相同 NeMo 设置下对比，Orukeet 在 FLEURS pooled WER 由 11.01% 降至 9.85%（约 11% 下降），在 LibriSpeech test‑other 由 3.14% 降至 2.86% 等。

**⚠️ 局限性**

限制在于仅固定 50% 的滤波器，且对某些语言的提升有限，且对超大模型的通用性与进一步可解释性仍有待验证。

---

## 410. Endorsement Without New Evidence: How Sequential Voting Inflates Mandates in Online Community Governance

**arXiv ID:** 2609.09321 | [PDF](https://arxiv.org/pdf/2609.09321v1)

**作者:** Zihan Chen `[一作]` (Stevens Institute of Technology), Di Zhu `[通讯]` (Stevens Institute of Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了投票与文本分离度量（vote–text divergence），并对 Wikipedia 管理员选举中约 19.8 万票进行顺序效应、信息饱和与治理结果的实证分析。

**💡 创新点**

提出了新的文本‑投票分离指标，首次揭示大幅支持边际可能掩盖独立审议的程度，并将其与信息饱和机制关联，挑战传统可见多数解释。

**🔧 技术方法**

使用词典式文本分析提取五维信息（情感、置信度、特异度、证据、顺从），再通过线性回归、固定效应模型、Permutation test 与伪随机试验（RD）评估梯度与后果。

**📊 数据集**

采用 Stanford SNAP Wiki‑RfA 语料库：4,003 次管理员选举、1,903 次成功提升、198,275 份带文本的投票记录，覆盖 2003‑2013 年。

**📈 对比分析**

与仅计数的传统方法和可见多数假设对比，发现投票顺序梯度显著且被累计文本信息完全解释，信息饱和而非决策度量驱动递延；治理后果（活动量、任期、被除名）与 divergence 无显著关联，显示选拔结果未受影响。

**⚠️ 局限性**

局限在于：① 依赖词典式自动评分，缺乏人工标注验证；② 不能区分“空递延”与“有效递延”；③ 仅评估可观测的治理指标，可能忽略更深层质量与长期影响。

---

## 411. HBFSim: Fast and Faithful Simulation of High-Bandwidth Flash Under Real GPU Execution

**arXiv ID:** 2609.09800 | [PDF](https://arxiv.org/pdf/2609.09800v1)

**作者:** Yanpeng Hu `[一作]` (ShanghaiTech University), Andi Quinn `[通讯]` (University of California Santa Cruz)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了一个平台，能够在真实GPU上执行大型语言模型推理任务的同时，将高带宽闪存（HBF）的时序、容量、热量等效应实时注入到工作负载中，并提供参考路径与快速路径两种模拟方式。

**💡 创新点**

创新点在于：①将HBF的时序与温度模型直接注入GPU PTX层，保持计算真实执行；②将指令发射与消耗分离，使用设备未来机制实现异步；③采用一次性校准的曲线和一阶热模型实现高精度且高效的模拟；④支持容量超大工作集（可达110 GiB逻辑空间）并验证正确性；⑤在参考路径基础上实现20.8倍加速的快速路径。

**🔧 技术方法**

使用技术包括：PTX重写、设备未来（device future）与TMA对象、MQSim参考路径、快速路径的按访问曲线计算、闭环热量模型（一次阶响应）与阈值状态、刷新与寿命管理、容量页面缓存与稀疏主机文件、校准与验证脚本。

**📊 数据集**

主要使用的数据集与模型：Qwen3‑30B‑A3B（通过 vLLM 0.15.1 运行），TinyLlama F16（通过 llama.cpp 运行），以及对比用的标准LLM推理工作负载。

**📈 对比分析**

比较方法：①验证注入时序后工作负载输出与基线完全一致；②在六个校准大小下测量服务时间与实际 NVMe 路径匹配；③对比参考路径与快速路径的运行时间，快速路径在相同工作负载下实现 20.8× 的加速；④在 110 GiB 逻辑范围的测试中，使用 2 GiB HBM 页面缓存，完成后校验和与基线一致；⑥评估温度变化对性能的影响，发现热量升高时 GPU 计算速率下降约 8%。

**⚠️ 局限性**

局限性：①HBF 设备尚未上市，校准基于 NVMe 驱动路径，未能捕捉真实 HBF 内部细节；②仅支持 PTX 可重写的内核，预编译的 cubin 及部分高级特性不兼容；③热量模型基于单台 GPU 与单个 NVMe 设备，未覆盖多卡或多机场景；④未对长期连续工作负载的热稳定性与耐久性进行验证；⑤对不同硬件平台的可移植性和可扩展性仍需进一步评估。

---

## 412. Meta-LinEXP3: Online-within-Online Learning for Adversarial Linear Contextual Bandits

**arXiv ID:** 2609.09907 | [PDF](https://arxiv.org/pdf/2609.09907v1)

**作者:** Hao Li `[一作]` (National University of Defense Technology), Zheng Xie `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Meta-LinEXP3算法，使用在线-内部在线框架在对抗性线性上下文多臂赌博机中实现跨任务迁移。

**💡 创新点**

创新点在于构造可预测的任务级先验并与LinEXP3结合，给出已知/未知上下文分布下的子线性任务间转移理论保证，并将先验准确性与转移复杂度关联。

**🔧 技术方法**

采用LinEXP3、策略中心估计（PC‑KDE）、过去正则化矩估计（PRME）、正余弦检索加权（PCRW）等技术，并证明转移复杂度与先验误差的关系。

**📊 数据集**

在合成随机任务、MovieLens 100K电影推荐、KSC光谱立方体结构化采样等数据集上进行评估。

**📈 对比分析**

与传统LinEXP3、TS、Meta‑TS、Uniform聚合、FFW等基线比较，Meta‑LinEXP3在任务对齐程度、MovieLens、KSC等实验中均取得更低的累计失调/奖励差距，提升约10–20%或1–2%不等。

**⚠️ 局限性**

局限性包括需要已知或可计算上下文分布、PRME需 O(d²) 记忆、转移收益仅在任务结构可预测时出现、异构任务可能无优势，且实验中的非线性或近似任务表示不完全符合理论假设。

---

## 413. Towards Stress-Aware Sentence-Level Filipino G2P With Weakly-Supervised ByT5 Fine-Tuning

**arXiv ID:** 2609.09974 | [PDF](https://arxiv.org/pdf/2609.09974v1)

**作者:** Lorenz Bernard Marqueses `[一作]` (De La Salle University), Ann Franchesca Laguna `[通讯]` (De La Salle University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种基于ByT5的句子级菲律宾G2P模型，能够在不显式标注重音的情况下自动推断句子上下文中的重音位置；

**💡 创新点**

创新点在于将LLM辅助的合成句子级语音标注与多语种预训练的ByT5相结合，实现对菲律宾词形变化、重音歧义和同形异义词的自适应识别；

**🔧 技术方法**

使用技术包括：Byte-level ByT5（mT5变体）、LLM（Gemini系列）生成句子级语音数据、Epitran规则系统做基线、PanPhon、CharliuG2P初始化、Adafactor优化器和多模型集成；

**📊 数据集**

数据集涵盖：WikiPron词典级发音、Tatoeba 7,300句子、NewsPH‑NLI 31,166句子、两套LLM生成的合成数据（无指导版和Wiktionary引导版），以及手工校正的1,173句子测试集；

**📈 对比分析**

通过与基线词级多语种G2P（CharliuG2P）和规则系统（Epitran）对比，句子级Fine‑tuned模型在手工测试集上PER降至0.54%（对比基线19.74%），CER降至2.50%，并在四大重音类上分别达到83%–89%的分类准确率；

**⚠️ 局限性**

局限性包括：合成语音数据噪声大、覆盖词形变化不完整、同形异义词在某些句子中仍出现重音错误、非标准重音的识别率低、模型对句子长度与复杂度的鲁棒性不足，未来需扩大数据规模、改进清洗与多任务学习以提升性能。

---

## 414. Do Agents Know When They Succeed? Calibrating Agent Confidence from Internal Representations

**arXiv ID:** 2609.09448 | [PDF](https://arxiv.org/pdf/2609.09448v1)

**作者:** Priyanka Mary Mammen `[一作]` (UMass Amherst), Srujananjali Medicherla `[通讯]` (Independent Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了两种基于内部隐藏状态的置信度评估框架——Latent Trajectory Dynamics (LTD) 与 Action Representation Probe (ARP)，用于多轮代理系统的可靠性监测。

**💡 创新点**

创新点在于利用LLM内部残差流的几何变化与动作决策时的表示直接推断任务成功率，突破传统仅靠输出层概率的限制。

**🔧 技术方法**

采用逻辑回归预测、余弦/相对位移度量、路径效率特征、PCA降维以及Platt校准等技术，对内部状态进行特征化并进行置信度估计。

**📊 数据集**

实验数据来自 InterCode Bench，涵盖 Bash、SQL（SPIDER）和 Python（MBPP）三类交互式编程任务。

**📈 对比分析**

与外部置信度基线（Calibrated Logprob、HTC）进行5折交叉验证比较，在 Qwen、DeepSeek 三个模型上，LTD/ARP 在 AUROC、ECE、Brier Score 上均优于外部基线，提升约5–10%。

**⚠️ 局限性**

局限性包括仅在离线教师强制生成的残差流上评估，未测试在线实时推断成本；对模型规模、任务多样性及跨域泛化的鲁棒性仍待进一步验证。

---

## 415. MedDeID enables locally governed clinical-text de-identification from real or synthetic training data

**arXiv ID:** 2609.10049 | [PDF](https://arxiv.org/pdf/2609.10049v1)

**作者:** Stig Hellemans `[一作]` (University of Antwerp), Kris Laukens `[通讯]` (University of Antwerp)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了MedDeID框架，在本地实现临床文本去标识、合成生成、模型训练、推理、伪匿名化及评估；同时提供荷兰语和英语两种语言的可部署模型。

**💡 创新点**

1）实现可在机构内部完全本地部署的完整去标识流程；2）首次展示仅使用合成数据即可在真实医院和基层护理文本上获得接近专家水平的召回率；3）通过子注释和核心PII召回指标对去标识质量进行细粒度评估；4）提供共享的schema与API，支持跨机构复现与审计。

**🔧 技术方法**

采用双头RoBERTa（荷兰使用RobBERT-2023，英语使用RoBERTa-base）进行BIO span检测和类别标注；后处理层对检测结果进行补全、扩展；伪匿名化通过日期偏移与年龄粗化实现；合成数据由LLM+规则渲染生成并标注。

**📊 数据集**

训练集：4,470份荷兰医院手工注释笔记（含合成标注）、6,493份合成荷兰笔记；验证集：300份荷兰医院、100份基层护理；测试集：荷兰医院、合成荷兰、合成英语、技术I与ASQ-PHI两套公开英语合成基准。

**📈 对比分析**

与两位人类标注者、荷兰DEDUCE、原始DEDUCE、专门荷兰去标识模型、Qwen3-8B大模型、GLiNER-PII、OpenAI Privacy Filter、OpenMed等对照。荷兰医院基准上：医院训练模型召回98.9%，非PII红写0.24%；合成训练模型召回96.1%；在基层护理上，合成模型优于医院模型（90.3% vs 87.0%）。英语合成模型在Technetium-I和ASQ-PHI上分别取得99.73%与98.9%的注释字符召回率，非PII红写率分别为1.61%和6.21%（排除未标注年龄后分别为0.89%和0.55%）。整体来看，合成训练模型在跨场景鲁棒性上表现突出，且推理速度快、资源消耗低。

**⚠️ 局限性**

（1）基准仅来自单一医院和单一基层诊所，样本量有限，可能不具备普适性；（2）合成文本虽然能提升鲁棒性，但标签准确性低于真实标注，且无法验证在真实英语临床文本上的性能；（3）评价指标主要关注核心PII召回与非PII红写，未充分评估伪匿名化对隐私风险的实际降低；（4）模型对日期格式、姓名格式等敏感性仍存在差异；（5）缺乏多机构、多语言的外部验证与伦理审计。

---

## 416. On the Sequential Test and Distributed Detection

**arXiv ID:** 2609.09358 | [PDF](https://arxiv.org/pdf/2609.09358v1)

**作者:** Earnest Akofor `[一作]` `[通讯]`, Earnest Akofor

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10和ImageNet数据集进行实验。

**📈 对比分析**

与传统的激活函数模型进行了比较，结果显示新模型在分类精度上提高了5%，且训练时间缩短了15%。

**⚠️ 局限性**

模型在处理高分辨率图像时性能下降，且对计算资源的需求较高。

---

## 417. Which Tokens Should SFT Actually Learn? A Token-Trimming Perspective on Mathematical Reasoning

**arXiv ID:** 2609.09707 | [PDF](https://arxiv.org/pdf/2609.09707v1)

**作者:** Yaning Jia `[一作]` (Dartmouth College), Soroush Vosoughi `[通讯]` (Dartmouth College)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出TrimSFT，一种基于logit-gap的token‑level重加权方法，用于数学推理的监督微调，聚焦中间置信度区间并剔除两端的tokens；

**💡 创新点**

创新点在于通过Gaussian权重对logit-gap谱两端进行裁剪，只强化介于两端之间的tokens，从而避免过度锐化或忽视低置信度tokens，提供非单调的重加权方案；

**🔧 技术方法**

采用在同一次前向传播中计算每个token的logit-gap，并用停梯度的Gaussian权重缩放交叉熵损失；与FSFT、DFT等传统重加权方法对比，并在pass@8、best‑of‑8等指标上进行评测；

**📊 数据集**

使用860K规模的数学推理链式思路数据，随机抽取20k进行微调，评测基准包括MATH500、OlympiadBench、Minerva、AMC、AIME24等五大数学推理测试集；

**📈 对比分析**

与Base、SFT、FSFT、DFT等基线对比，TrimSFT在六个基础模型上平均提升1.5‑3倍，尤其在MATH500上最高提升+26.9点；在pass@8和best‑of‑8上亦优于SFT，显示更高的覆盖率和自洽性；

**⚠️ 局限性**

仅在数学推理任务中验证，效果对代码生成、开放式指令等领域未知；需要手动设置两参数m,τ，虽相对鲁棒但仍可能需要针对不同模型/数据自适应调优。

---

## 418. DiffLUT-Net: Differentiable Training of FPGA LUT Networks with Learnable Connectivity

**arXiv ID:** 2609.09254 | [PDF](https://arxiv.org/pdf/2609.09254v1)

**作者:** Jiaqi Ye `[一作]` (Technische Universitaet Darmstadt), Grace Li Zhang `[通讯]` (Technische Universitaet Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研发了一种从训练到FPGA部署的完整框架 DiffLUT‑Net，直接从头训练六输入 LUT 网络并生成可综合 Verilog。

**💡 创新点**

同时联合学习 LUT 的 64 条真值表条目与每个 LUT 输入的硬件有效源选择，采用可微分逼近实现训练，随后硬化后直接映射到 FPGA。

**🔧 技术方法**

可微分多项式真值表逼近、稀疏可学习连接矩阵、分布感知热编码、GroupSum 累计输出以及后端硬化与 Verilog 导出。

**📊 数据集**

使用 JSC CERNBox、JSC OpenML、MNIST、Fashion‑MNIST、CIFAR‑10 等公开数据集进行实验。

**📈 对比分析**

与传统 MAC 加速器、Truth‑Table 转换方法、NeuraLUT、DWN、FPGN 等基线在相同 FPGA 资源/时延下对比，DiffLUT‑Net 在相同准确率下显著降低 LUT 用量（最高可至 52%）并减少 A×L，甚至在高精度配置下保持竞争力。

**⚠️ 局限性**

受限于六输入 LUT 的可扩展性，真值表尺寸随输入位宽指数增长；训练中离散化和硬化可能导致精度损失；对更大规模网络或卷积/变压器等架构尚未验证。

---

## 419. From Fixed Keys to Readable Schemas: Small Language Models for Vehicle Agent Function Calls

**arXiv ID:** 2609.09476 | [PDF](https://arxiv.org/pdf/2609.09476v1)

**作者:** Hamed Jafarzadeh Asl `[一作]` (Huawei Noah's Ark Lab), Vahid Partovi Nia `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对车载助手在有限内存和时延约束下，将自然语言请求映射为车辆功能调用的任务进行了研究。

**💡 创新点**

创新点在于提出并对比了两种功能表面表示方式——功能词（FT）与提示中模式（SIP），并从理论和实验角度揭示其对泛化、拒绝和部署成本的影响。

**🔧 技术方法**

使用了小型语言模型（Gemma 3 270 M、FunctionGemma 270 M、Qwen3 0.6 B、1.7 B）并通过全量微调或LoRA进行功能调用微调。

**📊 数据集**

构建了基于Android Automotive 79个功能的单轮功能调用基准，共9,822个样例，划分为已见、未见和超域三类。

**📈 对比分析**

在已见功能上四种模型均达96%+准确率；在未见功能上SIP在1.7 B规模下可实现84%准确率，而FT始终为0%；SIP在拒绝超域请求时的准确率明显优于FT，且两者在推理延迟和内存占用上存在显著差异。

**⚠️ 局限性**

局限包括基准为合成英文单轮数据，缺乏多语言、多轮真实驾驶对话以及在真实车载硬件上的延迟测评。

---

## 420. FreqFLD: Towards All-in-One Facial Landmark Detection via Frequency Modulation

**arXiv ID:** 2609.10278 | [PDF](https://arxiv.org/pdf/2609.10278v1)

**作者:** Shun Ren `[一作]` (China Three Gorges University), Jun Wan `[通讯]` (Zhongnan University of Economics and Law)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种 All-in-One 频率调制框架 FreqFLD，利用频率分离模块 FreqMoM、频率混合专家 FreqMoE 以及频率一致路由损失 FreqCR 实现跨数据集鲁棒的面部关键点检测；

**💡 创新点**

创新点包括：①显式分离低频全局结构与高频局部细节的频率分离模块；②在 All-in-One 下实现频率条件专家路由的 FreqMoE；③引入 FreqCR 损失平衡专家利用，稳定训练与提升泛化；

**🔧 技术方法**

使用了 Transformer 结构、FFT 频域操作、混合专家（Mixture of Experts）框架、多尺度编码解码器、频率提示块、正则化损失、数据增强及 AdamW 等技术；

**📊 数据集**

在 300W、WFLW、COFW、AFLW 四个公开数据集上进行训练与评估，并实现 All-in-One 多数据集联合训练与留一数据集测试；

**📈 对比分析**

与多种 Heatmap 与 Coordinate 回归基线方法对比，实验结果显示 FreqFLD 在各子集（包括遮挡、姿态、模糊等困难情况）上取得与 SOTA 相当甚至更优的 NME，尤其在 All-in-One 训练下实现了良好的跨数据集泛化；

**⚠️ 局限性**

局限性在于模型参数量较大，专家数量与 TopK 需细致调节；在极端场景或极少量数据时表现仍有提升空间；频率模块与专家设计尚未达到极致轻量化。

---

## 421. Concept drift mitigation through community and spectral graph analysis for the detectionof cyberattacks in network traffic

**arXiv ID:** 2609.09442 | [PDF](https://arxiv.org/pdf/2609.09442v1)

**作者:** Julien Michel `[一作]` (EPITA), Pierre Parrend `[通讯]` (EPITA)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在网络流量检测中提出一种基于特征稳定性（t‑robustness）的特征空间选择方法，旨在提前构建对概念漂移具有鲁棒性的特征集合。

**💡 创新点**

创新点包括：①将概念漂移的量化转移到单个特征层面，定义了可比的、无模型依赖的稳定性得分；②通过图社区度量和谱度量两类基于拓扑的衍生特征来捕捉攻击行为；③在不更新模型的前提下，仅通过特征筛选实现长期检测性能的提升。

**🔧 技术方法**

技术手段：基于 NetFlow 流量构建 IP/IP 与 IP+Port 两种图；使用 Louvain 社区划分和图 Laplacian 谱特征；计算特征状态、状态距离、t‑equivalency 以及最终的 t‑robustness；用 XGBoost（以及 CART、MLP 等）训练并评估；采用 MCC、保留期望等指标评估性能。

**📊 数据集**

使用公开的 UGR16 数据集（2016 年 7 月至 8 月的两周网络流量），在此数据上提取原始特征、图社区特征和谱特征，并在不同学习场景下进行实验。

**📈 对比分析**

比较方法：对照基线特征集、仅图社区特征集、仅谱特征集以及经过 t‑robustness 过滤得到的特征集；在 3+1 个学习场景（含无模型更新的漂移场景和无漂移的控制场景）下，分别记录 MCC、保留期望和 MCC 速率差。结果显示，t‑robust 特征集在漂移场景下的最差 MCC 和保留期望均明显高于基线和单一图特征集，证明了其对概念漂移的鲁棒性。

**⚠️ 局限性**

局限性：①谱特征的计算复杂度高，导致仅能在 150K 条样本上评估，影响结果可靠性；②t‑robustness 的阈值是经验设定，缺乏理论指导；③方法仅在有标签且可构建图的 NetFlow 数据上验证，其他流量类型或无监督情境下的适用性尚未测试；④模型仍受限于所选学习器，未能在更广泛的深度或迁移学习框架中验证。

---

## 422. X2-NativeCursor: Native-Token Text Progress Tracking for Incremental-Text Streaming Codec TTS

**arXiv ID:** 2609.09677 | [PDF](https://arxiv.org/pdf/2609.09677v1)

**作者:** Zehan Liu `[一作]` (X Square Robot), Qian Wang `[通讯]` (X Square Robot)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种轻量化在线观察器X2-NativeCursor，用于在不解码波形、且不改动TTS生成器的前提下，从原生语音Token实时跟踪文本进度；

**💡 创新点**

通过将文本先规范化为稳定的说话标签、使用局部匹配器与位置状态实现极低前瞻误差、将可修正的对齐估计与单调游标分离，并实现对多种codec‑based TTS仅需重训练观察器的兼容性；

**🔧 技术方法**

采用规范化计划TNPlan、文本编码器、带膨胀卷积的原生Token编码器、局部匹配器、位置状态、单调游标输出，并在训练时结合偏差、内容和速率损失进行监督；

**📊 数据集**

以20,235个Qwen3‑TTS合成语音样本训练，800个包含中英、数字、符号的固定测试文本为评测数据，参考使用Qwen3‑ForcedAligner自动对齐及MMS‑FA等；

**📈 对比分析**

与在线波形基线（WindowMMS+PersistentCTC）和完整音频基线（MMS‑FA、CTC‑seg、FunASR等）对比，X2‑NativeCursor在Qwen3‑TTS上以80 ms前瞻实现中文字符MAE仅0.151（比波形基线低≈88%），RTF仅0.0180（比0.3598降低95%），在CosyVoice2上重训练后MAE约0.26；

**⚠️ 局限性**

对不同声线的泛化性有限，未训练的声音会导致MAE升至≈0.79；需要为每个TTS骨干单独重训练观察器；目前仅验证了codec‑based TTS，对其他架构的鲁棒性尚未评估。

---

## 423. RobustSGPO: Search-Space Control for Agent Harness Evolution

**arXiv ID:** 2609.09646 | [PDF](https://arxiv.org/pdf/2609.09646v1)

**作者:** Zibo Zhao `[一作]` (Wuhan University), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RobustSGPO，通过明确编辑请求、补丁构造与检查以及保留多起始点来改进多代理系统的 Harness 演化。

**💡 创新点**

创新点在于将 SGPO 的自由式更新拆解为可控的搜索空间，结合编辑范围、操作和目标的显式决策，以及按类别保留有效快照，显著提升了编辑有效率与最终质量。

**🔧 技术方法**

使用语义梯度优化、对等重放验证、可编程补丁生成器、基于类别的存档（类似 MAP-Elites）和周期性权限调度等技术。

**📊 数据集**

在 AgentX brainstorming 工作流上评估，使用 120 个任务（两大类各 60 个）进行 95 次实验，累计 7350 个候选尝试。

**📈 对比分析**

与原 SGPO、Planned Search、Constrained Search 和 Structured Search 对比，RobustSGPO 在最终测试分数上提升至 4.14（相较 SGPO 的 3.77 提升 0.37），在 30 轮内完成 13/15 本地任务、11/15 跨代理任务，且在 20M token 预算下提升 80% 完成率。

**⚠️ 局限性**

主要局限是类别保留策略带来额外搜索开销，导致在等价 token 成本下与 Structured Search 的优势缩小；同时对权限调度的最佳策略和跨任务迁移的长期效果仍待进一步研究。

---

## 424. RealSimLoop: Online Real-to-Sim Adaptation via Differentiable Reduced-Order Simulation with Vision Feedback

**arXiv ID:** 2609.09828 | [PDF](https://arxiv.org/pdf/2609.09828v1)

**作者:** Zhihao Cen `[一作]` (South China University of Technology), Guoxin Fang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于可微分减阶模拟与视觉反馈的在线 Real‑to‑Sim 自适应框架 RealSimLoop，能够在几乎实时的条件下动态更新物理参数并同步模拟结果与真实观测。

**💡 创新点**

创新点包括：① 滑动窗口优化策略使系统能够在在线环境中捕捉时间变化的材料特性；② 将可微分 3D 高斯抛射（3DGS）与可微分渲染相结合，直接使用图像像素误差驱动参数更新；③ 在低维神经子空间中实现可微分仿真，显著降低求解和梯度传播成本；④ 采用对偶法（adjoint）在子空间中高效求解对材料参数的梯度。

**🔧 技术方法**

主要技术包括：可微分柔性体仿真（JAX实现的Neo‑Hookean模型）、神经 AutoEncoder 生成的减阶子空间、3D Gaussian Splatting + 可微分渲染、滑动窗口在线优化、对偶法梯度计算、SAM 2 用于视觉数据处理。

**📊 数据集**

数据集：① 真实实验数据（多材料电缆驱动结构、双臂柔性棒、温度可变结构、高速弹球碰撞等）来自多视角摄像机与力传感器；② 合成数据用于训练子空间模型，随机生成材料参数与交互序列；③ 公开图像/点云数据（如IPC弹球）用于测试。

**📈 对比分析**

与传统离线 Real‑to‑Sim 方法、仅基于标记的在线方法以及单一视角的离线视觉方法进行比较。结果显示：RealSimLoop 在速度上比全空间求解快 7–8 倍；在几何重建误差、PSNR、L2 误差等指标上优于离线基线；在张力预测与应力重建等下游任务中也表现出更高精度。

**⚠️ 局限性**

局限性：① 对于每个单元级的连续材料分布，参数空间增大，需更强正则化或神经表示；② 在高频极动态（如高速弹球）下减阶模型可能不收敛，需要退回全空间求解；③ 对未知变形模式或材料参数外推能力有限；④ 滑动窗口大小、子空间维度等超参数需要手动调节。

---

## 425. Beyond Top Words: MonoTM for Topic Modeling with Interpretable Monosemantic Features

**arXiv ID:** 2609.09575 | [PDF](https://arxiv.org/pdf/2609.09575v1)

**作者:** Una Joh `[一作]` (Syracuse University), Bei Yu `[通讯]` (Syracuse University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MonoTM框架，利用稀疏自编码器提取可解释特征，并在保留全特征估计的前提下，用验证过的特征构建主题描述。

**💡 创新点**

通过将稀疏自编码器特征分离为统计估计和解释两种角色，解决了特征可解释性与混合估计最优配置不一致的问题，并实现了在非单词级别上更具语义抽象的主题描述。

**🔧 技术方法**

使用稀疏自编码器（SAE）训练文档嵌入特征，结合Interpreter–Predictor LLM流程进行特征标签生成与验证，随后用LDA对全特征构建文档-主题混合，固定后在验证特征上估计主题-特征分布。

**📊 数据集**

在20 Newsgroups、Web of Science（WOS）和Reuters-21578三大基准语料上进行实验。

**📈 对比分析**

在混合估计方面，MonoTM基于全特征的BoF+LDA在三大数据集上均实现了比传统BoW-LDA和多种神经主题模型更高的Micro‑F1和Macro‑F1；在主题描述上，MonoTM的特征标签在可解释性和覆盖度上优于基于词的TF‑IDF、c‑TF‑IDF和纯LLM生成的描述。

**⚠️ 局限性**

局限包括对Interpreter/Predictor LLM的依赖导致解释性评分受模型偏见和成本影响、未完全覆盖低验证特征所含的潜在语义、以及在高影响应用中仍需人工审核以防止重要特征被忽略。

---

## 426. Senseful Consense: Towards Simplified Cookie Banners using Plain Language

**arXiv ID:** 2609.10271 | [PDF](https://arxiv.org/pdf/2609.10271v1)

**作者:** Minela Bećirović `[一作]` (TU Braunschweig), Alexandra Dirksen `[通讯]` (University of Twente)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在IAB Transparency & Consent Framework (TCF) 框架下，使用简易语言重新编写Cookie横幅文本，并评估其可读性与可理解性提升。

**💡 创新点**

创新点在于：①首次系统地将实际Cookie横幅文本映射到TCF 11个处理目的；②利用简易语言对每个目的进行标准化描述；③通过自动化工具量化简易语言改写后可读性的显著提升，首次为Cookie横幅提供可操作的简易语言改写框架。

**🔧 技术方法**

主要技术包括：AI目的映射模型（OpenAI o3、GPT‑4.1），可读性评估工具（capito.ai、Wortliga、Textstat），以及自研Firefox扩展用于提取横幅文本；所有评估均基于Flesch、FOG、Dale‑Chall三种可读性公式和语言等级分数。

**📊 数据集**

数据集为200个英文Cookie横幅样本（实际成功提取181个），来源于Majestic Million前百万网站列表；完整实验数据已公开发布于匿名Open Science项目。

**📈 对比分析**

比较方法：对每个横幅原始文本与改写文本分别计算Flesch Reading Ease、FOG Index、Dale‑Chall、capito.ai B1‑B2比例、Wortliga语言等级与可理解分数；结果显示改写后Flesch得分提升、FOG和Dale‑Chall下降，B1‑B2比例基本达到100%，Wortliga语言等级从C2/C1降至B1/B2，理解分数从0–100提升至75–89；仅少数案例未见显著差异。

**⚠️ 局限性**

局限性包括：①仅评估第一层横幅，未覆盖多层或交互式设置；②AI映射受模糊词和缺失信息影响，可能导致目的识别不完整；③简易语言仅提升可读性，无法补偿原文信息缺失；④缺乏真实用户体验测试，未检验是否真正提升用户决策质量；⑤对简易语言水平的定义存在主观性；⑥样本主要来自顶级网站，未必具备代表性。

---

## 427. OmniPoint: Universal Monocular Metric Pointcloud from Any Camera

**arXiv ID:** 2609.09394 | [PDF](https://arxiv.org/pdf/2609.09394v1)

**作者:** Botao Ye `[一作]` (Google DeepMind), Abhijit Kundu `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一的单目几何估计框架 OmniPoint，能够从任意相机（pinhole、鱼眼、全景）和任意输入条件（相机内参、稀疏深度）生成精确的三维点云。

**💡 创新点**

创新点包括：① 光线+距离（Ray+Distance）分离表示，解耦相机投影与场景结构；② 双向数据增强（Perspective‑to‑Any 与 Any‑to‑Perspective）在 3D 空间桥接有标签与无标签相机域；③ 通过可学习状态嵌入与向量化高斯平滑实现鲁棒的几何先验注入。

**🔧 技术方法**

核心技术有：Vision Transformer 变体作为特征提取器；分离的光线与距离回归与独立损失；ROE 对齐实现全局尺度一致；自监督伪标记与双向投影；向量化高斯平滑处理稀疏深度；多任务损失（点、光线、尺度、法向、局部一致性、遮挡掩码）。

**📊 数据集**

训练使用 29 个标注数据集（如 ARKitScenes、Matterport3D、ScanNet++ 等）与 2 个无标注全景集（Diverse360、360+x）。

**📈 对比分析**

与 15+ 先进基线（Depth Anything、MoGe、UniDepth、Depth Pro、UniK3D、DA² 等）在多种视角（标准视场、鱼眼、全景）下进行零样本评估，结果显示在所有相机类型上均达到或超过 SOTA，尤其在鱼眼和全景场景的相对误差下降 30% 以上；在给定稀疏深度时性能提升近 10%。

**⚠️ 局限性**

局限性包括：① 仍依赖大量标注相机特定数据进行预训练；② 对非常极端的畸变或极低分辨率全景仍可能出现误差；③ 模型规模较大，推理速度与显存需求相对较高。

---

## 428. SynThermFace: Amplifying Limited Paired Data for Visible-Thermal Face Recognition via Synthetic Data Generation

**arXiv ID:** 2609.10303 | [PDF](https://arxiv.org/pdf/2609.10303v1)

**作者:** Anjith George `[一作]` (Idiap Research Institute), Sebastien Marcel `[通讯]` (Idiap Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SynThermFace 框架，通过有限的真实可视‑热图对，利用扩散模型生成大规模可配对可视‑合成热图数据，并在此基础上用 PACT（Preservation‑Aware Cross‑Spectral Tuning）对已有可视 FR 模型进行跨光谱适配，最终实现仅一前向即可完成可视‑热面部识别；

**💡 创新点**

① 将有限的真实配对数据放大成大规模配对合成数据，极大提升跨光谱训练样本量；② 设计 PACT，融合对称 InfoNCE 对齐和冻结可视教师保留正则，兼顾跨模态对齐与可视特征保持；③ 将生成过程移到训练阶段，推理时不再需要图像翻译，提升部署效率；

**🔧 技术方法**

使用 Qwen‑Image‑Edit（多模态扩散变换器）+ LoRA 进行可视→热的图像翻译；Flow‑matching 与身份保留损失训练生成器；在可视 FR 基础上（EdgeFace）使用对称 InfoNCE 交叉模态对齐与可视教师保持损失进行 PACT 微调；

**📊 数据集**

MCXFace（可视‑热对）用于模型训练与基准评估；CASIA‑WebFace 与 Digi2Real 作为可视源，用以生成合成热图构建大规模配对数据集；Tufts Face 数据集用于跨数据库评估；

**📈 对比分析**

与基线可视 FR（AdaFace、EdgeFace）、HFR 方法（DIU、xEdgeFace、PDT）及 PACT 的不同配对配置进行比较。PCA‑Real 在 MCXFace 上 EER 3.04%，Rank‑1 96.49%；PCA‑CASIA‑Synthetic‑Thermal EER 0.99%、Rank‑1 99.75%；PCA‑Digi2Real‑Synthetic‑Thermal EER 1.20%、Rank‑1 99.50%。在 Tufts 上，PCA‑Real 使 EER 从 43.41% 降至 18.21%，PCA‑Synthetic 进一步降至 13.91%；表明生成数据显著提升性能并具有一定跨数据库迁移能力；

**⚠️ 局限性**

生成热图仍依赖有限真实对，可能继承扩散模型与源数据的偏差；身份保持难以直接量化，只通过下游 FR 性能间接评估；实验仅使用 EdgeFace 验证，未验证在更大容量模型上的泛化；跨数据库迁移虽提升但仍受传感器/数据库差异限制；

---

## 429. GANDR: Claim Auditing for Verifiable Legal Answer Generation

**arXiv ID:** 2609.10293 | [PDF](https://arxiv.org/pdf/2609.10293v1)

**作者:** Chen Qian `[一作]` (William & Mary), Andreas Stathopoulos `[通讯]` (William & Mary)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种两代理系统 Grounded Answer DRafter，通过 Drafter 生成结构化法律推理答案并由 Critic 在独立上下文中逐条核查引用，最终提供每条声明的审计轨迹。

**💡 创新点**

其创新点在于将逐条验证作为答案生成的一部分，使用结构化 CREAC 方案与结构化审核轨迹相结合，并以结构检查为提交门槛，避免依赖审核者的主观判定。

**🔧 技术方法**

采用了大语言模型（如 30B Nemotron、GLM‑4.7‑Flash 等）配合 BM25 检索、SEF 自检 rubric、atomic 解析工具以及多轮重写机制。

**📊 数据集**

主要使用了来自 LegalBench 与 LegalBench‑RAG 的 185 条法律案例，覆盖 15 任务标签与 8 领域桶。

**📈 对比分析**

与六个基线（零射、单模型、增强 RAG、CrewAI、LangGraph 等）共享相同检索和引用指令进行对比；在严格准确率上达到 70.8%，比最强基线高 11.3%，并在不同后端模型上保持 +3.2~+6.5 的优势。

**⚠️ 局限性**

主要限制是对检索质量依赖较高，无法处理多源记忆或更难检索的情境，且 audit 机制在四方标签上仅具备咨询性，未能完全替代人工验证。

---

## 430. Isotropic Embedding Perturbations for Robust Vision Language Encoders

**arXiv ID:** 2609.10292 | [PDF](https://arxiv.org/pdf/2609.10292v1)

**作者:** Hyesong Choi `[一作]` (Soongsil University), Dongyoon Han `[通讯]` (NAVER AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种在嵌入空间进行等向性噪声扰动的数据增强方法（Aether），通过控制α‑mixing将高斯噪声平滑混合到视觉/语言编码器的嵌入层，实现对特征空间的正则化；

**💡 创新点**

创新点在于：①突破传统输入空间/区域混合的三轴饱和瓶颈，探索全新嵌入空间正则化轴；②采用变异保持的α‑mixing，保证噪声等向且不破坏语义结构；③在视觉‑语言对齐任务中实现对跨模态细粒度对齐的友好增强。

**🔧 技术方法**

核心技术包括：图像输入→patch/卷积→嵌入→在嵌入层加入正态分布噪声并按预设噪声时间步t的α̅_t系数做α‑mix；实现细粒度的等向性扰动；代码基于PyTorch实现。

**📊 数据集**

主要使用数据集：ImageNet‑1K（主任务），CUB、NABirds（细粒度分类），ADE20K（语义分割），COCO（目标检测/实例分割）以及多种SSL预训练模型（MAE、SimMIM、DiffMAE、MaskDiT、DiffMIM）进行下游评估。

**📈 对比分析**

与标准增强组合ℛ_b（CutMix+MixUp+DropPath+RandAug）以及其他同类增强（AugMix、RandErase、Manifold Mixup、Noisy Feature Mixup）进行对比；在ImageNet上提升至+3.49% top‑1，VLMs（CLIP、AIMv2、SigLIP）提升0.5–1.3%，细粒度任务提升≈1.6%，分割/检测任务提升≈0.4%；实验表明该方法在多种架构（ViT、Swin、ResNet）及多任务上均具备可迁移性。

**⚠️ 局限性**

局限性包括：①对极度饱和或极大模型的提升幅度有限；②仅在视觉‑语言和视觉任务上验证，其他模态（如音频）尚未测试；③噪声时间步和α̅_t调度需经验选择，可能影响稳定性；④对部分任务（如极端小样本场景）提升不显著。

---

## 431. HybridFLow: SDN-Orchestrated Client Partitioning for Hybrid Federated Learning

**arXiv ID:** 2609.10404 | [PDF](https://arxiv.org/pdf/2609.10404v1)

**作者:** Osama Abu Hamdan `[一作]` (University of Texas at Arlington), Md Arifuzzaman `[通讯]` (Missouri University of Science and Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个闭环 SDN 调度框架，将 SDN 控制平面的全局网络状态嵌入混合联邦学习的同步/异步客户端划分中，以减少通信瓶颈导致的 straggler 影响并加速模型收敛。

**💡 创新点**

创新点在于利用 SDN 的全域拓扑视图生成校准后的每客户端通信时延估计，并将其作为决策依据，形成网络感知的同步/异步划分算法，实现通信延迟与更新老化风险的平衡。

**🔧 技术方法**

采用 ONOS SDN 控制器、OpenFlow 统计、Flow Scheduler、Progress Tracker、EWMA 校准、Flower FL 框架以及 ZeroMQ 通信接口，结合 Mathis 变体吞吐量估算和强化学习/约束规划可选的路径分配算法。

**📊 数据集**

使用 CIFAR‑10 数据集并按 7/10 类划分产生非 IID，模型为 MobileNetV3‑Large（14.2 MB）进行训练，以评估跨站点网络异构性对 FL 性能的影响。

**📈 对比分析**

与完全同步（SDN 路由）和完全异步 FedAsync 两个基线对比，实验在三种拓扑（E1、E2、E3）中显示本框架在达到 80% 目标准确率时比基线快 33–40%，平均每轮时长减少 30–40 秒，且在非 IID 场景下保持收敛质量。

**⚠️ 局限性**

实验仅在仿真环境中验证，假设全局 SDN 可见性、可信流量数据和无攻击性客户端，缺乏大规模真实 WAN 部署验证、对控制器失效或不完整网络视图的鲁棒性评估。

---

## 432. Rosetta at AlexandriaX-2026: LoRA-Adapted NileChat for Context-Aware Dialectal Arabic Dialogue Translation

**arXiv ID:** 2609.10395 | [PDF](https://arxiv.org/pdf/2609.10395v1)

**作者:** Nada Esmaeil `[一作]` (Tanta University), Muhammad Arif `[通讯]` (Yale University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Rosetta 系统，利用 LoRA 适配 NileChat-3B，对英语对话进行上下文感知的方言阿拉伯语翻译，并在 AlexandriaX 共享任务中分别在受限和无限制赛道上提交。

**💡 创新点**

创新点包括：① 通过结构化系统/用户提示显式注入方言标签、对话历史、性别和人物信息；② 在训练阶段引入历史噪声以缓解曝光偏差；③ 采用外部 MADAR 与 PADIC 句对预训练，检验其对不同方言的影响，揭示负迁移现象。

**🔧 技术方法**

技术手段包括：LoRA 参数适配 (r=32, α=32, dropout=0)；4‑bit NF4 量化和 Unsloth+PEFT 的高效微调；对话上下文递归输入；Beam‑search (5 beams, 长度惩罚 0.7) 进行推理；历史噪声（截断、词级丢弃/交换/复制）。

**📊 数据集**

使用数据集：Alexandria 语料（13 种方言对话，约 66k 训练转折）；外部预训练数据为 MADAR（约 68k 句对）和 PADIC（约 17k 句对）通过 NLLB‑200 将 MSA 翻译为英语后构造方言‑英语平行语料。

**📈 对比分析**

在官方评测中，受限赛道的平均 spBLEU 为 26.10，排名第 4；无限制赛道平均 spBLEU 为 25.09，排名第 5。相比之下，外部预训练对大多数方言没有提升，只有利比亚和摩洛哥方言略有增益，整体表现略逊于仅使用 Alexandria 训练的受限模型。

**⚠️ 局限性**

局限性在于：① 外部预训练导致大部分方言出现负迁移，无法统一提升；② 对不同方言的覆盖和质量不均衡，可能因后向翻译噪声导致误差；③ 只对 3B 参数模型做适配，未探索更大规模或多任务联合学习的潜力。

---

## 433. A Later Test Set Is Not a New Domain: Pretraining Familiarity Survives a Contamination-Free Hold-Out

**arXiv ID:** 2609.10357 | [PDF](https://arxiv.org/pdf/2609.10357v1)

**作者:** Mahdi Naser Moghadasi `[一作]` (BrightMind AI), Faezeh Ghaderi `[通讯]` (University of Texas at Arlington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个完全基于时间后置且可重建的数据集协议，评估13种预测器（4个经典、3个数据集专门训练、6个预训练模型）在5个连续发布域的7组时间序列上的零样本性能。

**💡 创新点**

创新点在于：①提出仅使用后置测试窗口、无API密钥、可重建的数据集协议；②发现预训练模型优势主要源于语料库熟悉度而非序列特性；③提供显著性检验和多指标比较框架，揭示预训练模型在准确性上虽有优势但区间覆盖率更自信。

**🔧 技术方法**

采用MASE、sMAPE、加权分位数损失、80%区间覆盖率等评价指标；使用Nemenyi和Wilcoxon检验比较模型；对比经典基线（Theta、AutoETS、AutoARIMA等）、训练集模型（LightGBM、LSTM、NBEATS）与预训练模型（ChronosBolt、Chronos2、TimesFM、TimesFM3、Moirai2）。

**📊 数据集**

使用五个持续公开的域的七组数据集：Wikipedia pageviews（日/周/月）、Weather（ERA5）/Air quality（CAMS）（小时）、Electricity（丹麦电网）（小时）、Exchange rates（ECB）（日），每组包含数百到数十个系列，所有数据可通过无密钥脚本重建。

**📈 对比分析**

以平均排名并给出Nemenyi区间为主进行比较；结果显示预训练模型在Wikipedia域占优，在Weather/Air quality表现相近，电力域被Theta领跑，汇率域所有模型相同；预训练模型在准确度上略好，但80%区间覆盖率低于经典基线；成本方面预训练模型显著低于自动ARIMA搜索。

**⚠️ 局限性**

局限性包括：仅使用时间后置无法消除语料库熟悉度影响；域和频率覆盖有限；预训练截止时间依赖公开说明，难以验证；未深入评估模型随机性、数值稳定性和多窗口泛化；依赖公开语料库描述，无法完全重现所有模型的训练细节。

---

## 434. CertiFlash: A Formal Verification Framework for Flash Translation Layers in Computational Solid State Drives

**arXiv ID:** 2609.10347 | [PDF](https://arxiv.org/pdf/2609.10347v1)

**作者:** Harshita Gupta `[一作]` (ETH Zurich), Onur Mutlu `[通讯]` (ETH Zurich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 CertiFlash，一个可复用的 Coq 形式化验证框架，用于验证 SSD FTL 的安全性（租户隔离、完整性、块所有权）和功能正确性。

**💡 创新点**

创新点在于：① 将安全与功能性合并为单一全局不变式并一次性证明其保持；② 设计可重用的模块接口，使新 FTL 只需证明五条假设而非重新推导 27 条不变式；③ 通过预条件束捕捉所有已知失效面并证明其充分性与必要性。

**🔧 技术方法**

采用 Coq 证明助手实现状态机模型、全局不变式、抽象与具体设备的关系，使用命令级预条件束、可执行的布尔表达式以及 Coq 自带的库和定理证明。

**📊 数据集**

在 DaisyPlus OpenSSD（Xilinx Zynq UltraScale+ + Micron NAND）硬件上演示了十种攻击，四个案例研究（扩展状态、缓存实现、范围映射、去重）检验框架。

**📈 对比分析**

对比传统单一实现验证，证书长度从 16,489 行仅需 27–3,231 行，验证时间 93 秒；新框架在四个案例中显著降低了重写工作量，保持了相同的安全与功能性。

**⚠️ 局限性**

局限性包括：只针对单线程、无并发 FTL；模型仅覆盖页级映射和外部元数据，未考虑高级硬件特性或完整性保护协议；对动态重配置、跨设备共享等场景支持有限。

---

## 435. Learning Intrusion Response Strategies for OT Systems

**arXiv ID:** 2609.10298 | [PDF](https://arxiv.org/pdf/2609.10298v1)

**作者:** Duc Huy Le `[一作]` (KTH Royal Institute of Technology), Rolf Stadler `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于 POMDP 的 OT 入侵响应问题模型，并设计两种基于 PPO 的学习方法（k-Obs-PPO 与 BF-PPO）来自动生成防御策略；随后在仿真与真实仿真环境中对学习到的策略进行评估。

**💡 创新点**

创新点包括：①将部分可观测的 OT 入侵响应场景形式化为 POMDP 并提供可计算的观测函数；②提出两种可扩展的 RL 方案，其中 BF-PPO 通过粒子滤波与边缘信念压缩大幅降低贝叶斯更新的复杂度；③在真实 OT 仿真平台上验证学习策略的有效性，展示近似全观测方法的性能。

**🔧 技术方法**

主要技术：POMDP 建模、近端策略优化（PPO）、k-Obs-PPO（仅用最近 k 条观测）、BF-PPO（粒子滤波 + 边缘信念输入）、Docker/ContainerLab 仿真、ModbusTCP/PLC 开源实现、IDS 交通采集。

**📊 数据集**

使用 40000 期（约 14 天）从仿真环境收集的 IDS 网络测量数据来估计观测函数；在仿真平台上随机生成 500×100 轮 POMDP 轨迹作为训练数据，并在 20 期仿真 OT 环境中评估策略。

**📈 对比分析**

对比方法包括：①全观测假设下的 MDP-PPO（理想基线）；②基于阈值的手工策略。结果显示 BF-PPO 的累计成本最低，收敛速度最快，且与 MDP-PPO 的性能仅相差数个百分点；k-Obs-PPO 收敛慢且成本高；阈值策略表现最差。

**⚠️ 局限性**

局限性：①模型采用离散时间、同步动作且无时延，仅近似真实 OT 运行；②仅针对特定架构（Purdue 模型）和攻击类型，缺乏泛化评估；③未考虑操作安全约束；④观测函数简化为仅与攻击动作相关，忽略背景流量的多样性；⑤粒子滤波的粒子数和边缘压缩可能在更大状态空间下失效。

---

## 436. A Confidence-Aware Multimodal Fusion Framework for Industrial Human-Robot Collaboration

**arXiv ID:** 2609.10339 | [PDF](https://arxiv.org/pdf/2609.10339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 437. Data-Driven Risk Fields for Safer End-to-End Autonomous Driving

**arXiv ID:** 2609.10377 | [PDF](https://arxiv.org/pdf/2609.10377v1)

**作者:** Yuanxin Tian `[一作]` (Tsinghua University), Jianqiang Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 DRiF 框架，学习共享的 BEV 风险场景表示，结合静态地图、动态交互风险预测与轨迹规划。

**💡 创新点**

创新点在于使用相对风险监督学习动态风险，避免手工绝对风险标注；同时将静态地图与动态风险联合训练于共享 BEV 表示。

**🔧 技术方法**

采用 BEV 编码器、三分支（静态地图、动态风险、规划）网络，基于规则安全优先级生成 pairwise 风险标签，并使用相对风险对比损失与双阶段训练策略。

**📊 数据集**

使用 CARLA 仿真收集的多城镇数据，并在 Bench2Drive 闭环基准上进行评估。

**📈 对比分析**

与 UniAD、VAD、TF++ 等基线对比，DRiF multi-frame 在 Bench2Drive 取得 88.78 DS、75.91 SR，显著优于 TF++ multi-frame（85.65 DS、69.09 SR），同时提升安全指标和多能力得分。

**⚠️ 局限性**

pairwise 标签仅覆盖重叠、走廊、占用三类风险，未覆盖更全面的真实世界风险理论，未来需扩展更完整的风险模型。

---

## 438. Beyond Weak Labels: Prompt-Guided Local Refinement for Weakly Supervised Water Segmentation in High-Resolution Multispectral Imagery

**arXiv ID:** 2609.10371 | [PDF](https://arxiv.org/pdf/2609.10371v1)

**作者:** Muhammad Farhan Humayun `[一作]` (University of Turku), Jukka Heikkonen `[通讯]` (University of Turku)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出两阶段弱监督高分辨率多光谱水体分割框架，先用伪标签训练SegFormer/ U-Net，后用结构化提示在SAM 2上进行局部修正。

**💡 创新点**

结合结构化组件级提示生成与SAM 2的局部修正，针对伪标签缺陷进行边界细化和小水体恢复，显著提升细节识别。

**🔧 技术方法**

使用SegFormer/ U-Net基础网络、伪标签、置信度图、结构化提示生成、SAM 2分割与几何筛选等技术。

**📊 数据集**

基于芬兰Joensuu地区高分辨率Orthophoto（RGB+NIR），官方水文矢量转栅格的伪标签，手工校正的强标签验证集（8184/810）。

**📈 对比分析**

在手工校正的验证集上比较Stage 1与Stage 2，SegFormer B0的IoU从0.9509提升至0.9535，U‑Net从0.9408提升至0.9486，F1、精度等指标均有细微提升，主要体现在边界精确和细小结构恢复。

**⚠️ 局限性**

仅提升了整体指标的微小幅度，效果受Stage 1伪标签质量和提示生成的限制；对全局误分无法纠正，依赖于高质量初始预测；未验证跨区域迁移及更复杂场景下的鲁棒性。

---

## 439. Unifying Score and Performance for Fine-Grained Music Understanding in Audio-Language Models

**arXiv ID:** 2609.10351 | [PDF](https://arxiv.org/pdf/2609.10351v1)

**作者:** Milan Liessens Dujardin `[一作]` (Bryel Labs and UC Berkeley), Kevin Miao `[通讯]` (Bryel Labs)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为 MuNo-SP 的文本化音乐表示方法，将乐谱与对应演奏的时间、力度、踏板等信息统一编码，并基于该表示构建 MAESTROCaps 数据集，生成基于演奏的细粒度音乐分析与问答对。

**💡 创新点**

创新点在于：①将乐谱与演奏信息以可被大型语言模型直接理解的线性文本格式统一；②搭建自动化数据生成流水线，实现从对齐乐谱、MIDI 到长篇音频分析和问答的全流程；③发布包含 100 条古典钢琴作品、对应音频、分析与问答的 MAESTROCaps 数据集。

**🔧 技术方法**

使用技术包括：MuNo-SP 语法设计与转换；音乐-性能对齐算法（利用 MAESTRO、ASAP、(n)ASAP、PianoCoRe、BSED 等资源）；基于 GPT‑5.6 Sol 的结构化分段、长篇分析与问答生成；以及 Gemini 3.7 Flash 作为评判者进行自动评估。

**📊 数据集**

数据集：MAESTRO、ASAP、(n)ASAP、PianoCoRe、Beethoven Symphony Expression Dataset (BSED) 等，涵盖 100 条古典钢琴作品（共 100 小时）及对应乐谱、MIDI 与音频。

**📈 对比分析**

比较方法：在 MuSP‑Bench 上对 MuNo‑S、MuNo‑SP 与 ABC、MIDI‑as‑text、ABC+MIDI 等基线进行问答准确率对比；通过 token‑efficiency 衡量文本压缩率；进行人工评估与 Gemini‑as‑Judge 的四维度评分。MuNo‑SP 在所有模型与子任务上均显著优于基线（如联合得分提升至 80% 以上，单一模型最高提升 34%），且在人类评测中获得 83% 以上的整体偏好。

**⚠️ 局限性**

局限性：①数据主要集中于西方古典钢琴，缺乏其他流派与乐器；②对乐谱与 MIDI 的对齐与校正依赖人工审核，易传播错误；③MuNo‑SP 仅包含谱面与 MIDI 信息，缺少音色、录音环境等音频特征；④模型对已知作品的偏好尚未完全评估；⑤对大规模开放权重 LLM 的适配与部署仍待验证。

---

## 440. On-Policy Distillation for Vision-Language Model Adaptation, an Effective Paradigm on Low-Quality Multimodal Data

**arXiv ID:** 2609.10321 | [PDF](https://arxiv.org/pdf/2609.10321v1)

**作者:** Hongyuan Zhang `[一作]` (University of Hong Kong), Ping Luo `[通讯]` (University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在视觉‑语言模型适配中使用 on‑policy 蒸馏的框架，通过学习轻量化控制器动态构造样本级蒸馏目标，使教师、零射先验和硬标签按可靠性自适应混合。

**💡 创新点**

创新点在于将蒸馏目标构造视为可学习的在线决策，利用教师、学生、先验的置信度、边缘、熵、对比度量与验证反馈，生成 bounded 的目标混合、样本权重与温度，从而在不同样本和训练阶段实现自适应监督。

**🔧 技术方法**

使用了轻量化 MLP 控制器、可靠性与对比度量（置信度、边缘、熵、教师‑学生/教师‑先验不一致、特征对齐）、验证反馈动作匹配、bounded policy actions、可微蒸馏损失等技术。

**📊 数据集**

使用了 11 个 Base‑to‑Novel 基准（ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101）和十个 Cross‑Dataset 目标数据集（ImageNet→Caltech101、OxfordPets、Cars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101）。

**📈 对比分析**

与零射基准、CoOp/CoCoOp/MaPLe/PromptSRC 以及 PromptKD 进行对比；在 Base‑to‑Novel 上 HM 从 83.73 提升到 84.62，主要提升新类别；在 Cross‑Dataset 上平均准确率提升到 72.66，较 PromptKD 提升 1.33%，在 FGVCAircraft、DTD、EuroSAT 等难点数据集表现尤为显著。

**⚠️ 局限性**

局限性包括：需要额外训练阶段的验证反馈与计算开销；对极端领域漂移的提升有限；在部分数据集（如 Caltech101、Food101、UCF101）提升不显著；若可靠性信号质量不足，控制器可能产生不稳定的目标构造。

---

## 441. Ensembling LLMs for AI-Augmented Cybersecurity Software Requirements Generation

**arXiv ID:** 2609.10316 | [PDF](https://arxiv.org/pdf/2609.10316v1)

**作者:** Santiago Perez-Acuna `[一作]` (Universidad Politécnica de Madrid), Juan C. Yelmo `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对同一安全标准（ISO/IEC 27002）下的系统需求生成任务，收集多模型、多随机种子和不同提示配置的LLM生成结果，采用后期投票式融合（Uniform与Naive‑Bayes）构建单一排序列表，并与单次运行、配置平均值进行对比。

**💡 创新点**

将LLM输出视为无序检索结果，将投票融合转化为信息检索式数据融合，首次在安全需求生成中系统评估多模型、多随机性的“协和效应”与可靠性加权对结果排序的影响，展示后期融合即可显著提升召回率并压缩误报。

**🔧 技术方法**

投票式加权融合（Uniform、Naive‑Bayes）、信息检索评估指标（PR、ROC、AP、F2、Jw）、统计验证（OOB bootstrap、结构扰动检验）以及对齐后缀@k的精度/召回曲线。

**📊 数据集**

在AI4I4（工业4.0物流系统）上，从10条ISO/IEC 27002:2022控制模板生成的72条黄金需求（在183条候选中）以及来自4个模型家族（Llama 3.1 405B、Qwen‑2 72B、Mixtral 8×22B、GPT‑4 Turbo）共24次随机运行。

**📈 对比分析**

与单个运行和配置平均值比较，Uniform融合将AP从0.684提升至0.825（↑21%），ROC AUC从0.699提升至0.817（↑17%），F2@kR从0.764提升至0.831；Naive‑Bayes进一步提升到AP 0.864、ROC AUC 0.869、F2 0.831，且在更浅的审查深度（k≈91）即可达到最优点；内部验证显示改进在92%以上重复样本中保持正向。

**⚠️ 局限性**

实验仅涵盖单一标准、单一系统和单一语言；候选集为已标注的有限集合，未考虑未生成的有效需求；各配置的随机种子数量不均衡；融合策略依赖于先前的人工判定，缺乏对新系统、不同标准或多语言场景的泛化验证。

---

## 442. RiLM: Parameter-Efficient Language Modeling via Geodesic Decoding

**arXiv ID:** 2609.10305 | [PDF](https://arxiv.org/pdf/2609.10305v1)

**作者:** Fang Li `[一作]` `[通讯]` (Oklahoma Christian University), Fang Li (Oklahoma Christian University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种参数高效的语言模型RiLM，将解码从传统输出矩阵替换为在黎曼流形上进行几何距离解码，采用共享MLP实现状态更新；

**💡 创新点**

核心创新在于使用几何距离作为解码方式，并在超平面上构建状态轨迹，同时引入Möbius运算解决超球面递归的边界崩塌问题；

**🔧 技术方法**

利用Riemannian几何（欧氏与Poincaré球面）、Möbius算子、共享MLP构成递归核、梯度截断训练、参数化的温度softmax；

**📊 数据集**

主要实验基于WikiText-2与Penn Treebank，使用2000词频词表进行控制，另外在10k词表下进行验证；

**📈 对比分析**

对照未绑定、绑定、参数匹配的LSTM、Transformer和SSM基线，发现HypRiLM在WikiText-2上在约290k参数下显著低于所有基线（PPL 54.2 vs 117.9–149.9），Flat RiLM在PTB上表现更佳；

**⚠️ 局限性**

局限性包括：仅在小词表和小模型规模验证，超球面解码在大词表下效果不稳定，需近似技术；训练时需额外的Möbius正则化，推理时计算量接近softmax；不适用于大规模预训练或完整词表任务。

---

## 443. An Empirical Analysis of ReDoS Vulnerabilities and ReDoS Detection Tools

**arXiv ID:** 2609.10294 | [PDF](https://arxiv.org/pdf/2609.10294v1)

**作者:** N'Zolieh Ismaël Mahassadi `[一作]` (Université du Québec en Outaouais), Abdelwahab Hamou-Lhadj `[通讯]` (Concordia Universiy)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对 490 条 ReDoS CVE 进行经验分析，并利用三大 regex 数据集（RegexLib、Regex_before、Regex_after）评估 5 款公开 ReDoS 检测工具与 1 款纠错工具的有效性。

**💡 创新点**

创新点在于：①构建了首个标注好的 ReDoS CVE 数据集和标记版 RegexLib 数据集；②提出多因子度量（结合静态、动态工具判定）来判断 regex 是否易受 ReDoS 攻击；③系统对比 5 款检测工具的共识度与误判率，揭示工具间高度不一致。

**🔧 技术方法**

技术包括：正则表达式静态分析（ReDoSHunter、SafeRegex、RAT）、灰盒动态检测（Rescue、Revealer）、以及基于模式修正的 RegexScalpel 纠错。

**📊 数据集**

使用的数据集有：①490 条 ReDoS CVE（NVD + 自定义关键词收集）；②RegexLib（RXXR2）共 2,853 条合法 regex；③CVEFixes 提取的 Regex_before（379 条）与 Regex_after（391 条）两组真实项目 regex。

**📈 对比分析**

比较方法：对每条 regex 运行 5 个检测工具，统计各工具标记、共识（0-5 计数）和多因子阈值下的漏洞率；纠错后再次检测。性能结果显示：工具间共识率低，单工具标记 74% 的漏洞；多因子度量下 25.5% 的 regex 被判定为易受攻击；纠错率在 RegexLib 上 97.9% 但仍有 52.7% 的纠正后 regex 仍被判定易受攻击。

**⚠️ 局限性**

局限性：①工具定义与攻击模式不统一，导致误判与漏判；②RegexScalpel 修正往往改变语义或仅加入硬编码限制，不能保证安全；③实验仅覆盖公开工具与特定数据集，未检验其他编程语言/引擎；④多因子度量虽提高准确性，却依赖手工阈值与经验，缺乏理论支持。

---

## 444. Cyber-Financial Contagion: Modeling the Propagation of an AI Vendor Compromise Through the Banking System

**arXiv ID:** 2609.10350 | [PDF](https://arxiv.org/pdf/2609.10350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 445. On the Limits of Quantum Multiparty Simultaneous Communication

**arXiv ID:** 2609.10289 | [PDF](https://arxiv.org/pdf/2609.10289v1)

**作者:** Pedro Montealegre `[一作]` (Universidad Adolfo Ibañez), Jorge Valenzuela `[通讯]` (Universidad Adolfo Ibañez)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并分析了多玩家SMP模型下的IC_k,n关系，证明了公共随机与无预共享纠缠的量子信息在最大消息长度上呈指数分离。

**💡 创新点**

创新点在于：①构造了自然的k-玩家Index Coordination泛化IC_k,n；②给出了无共享纠缠量子协议在无误差与有限误差两种错误模式下的最小消息长度下界；③利用全局相位折叠技术与多方直接积定理的精确因式分解，完成了多方量子状态识别的直接积定理。

**🔧 技术方法**

采用了随机访问码(RAC)不等式、量子状态识别的(p,η)-预测器框架、相位折叠技术、算子相似分解与AM‑GM不等式，以及两方不对称直接积定理的推广。

**📊 数据集**

无具体数据集，使用理论构造的硬分布进行证明。

**📈 对比分析**

与经典公共随机协议相比，量子协议在最大消息长度上仅能得到多项式下界：在无误差模式下Ω(n^{1-1/k})，在有限误差模式下Ω(n^{(k-1)/(k+1)})；当k≥c·log n时两者均达到Ω(n)。

**⚠️ 局限性**

局限性包括：1) 仅证明了无预共享纠缠的量子下界，未考虑共享纠缠的可能提升；2) 对有限误差下界仍存在指数与多项式的差距，尚未统一两种错误模式的下界；3) 需要更完整的多方直接积定理以进一步提升下界；4) 研究仅为理论分析，缺乏实验验证。

---

## 446. Construction of Multi-sequences With High Nonlinear Complexity via Narrow Ray Class Fields

**arXiv ID:** 2609.10369 | [PDF](https://arxiv.org/pdf/2609.10369v1)

**作者:** Xiaofeng Liu `[一作]` (Nankai University), Fang-Wei Fu `[通讯]` (Nankai University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了利用窄射线类域和1阶Drinfeld模构造多序列，获得高非线性复杂度；

**💡 创新点**

创新点在于提出统一的框架，将窄射线类域与Guruswami‑Xing的循环下降相结合，得到可扩展到任意曲线型函数域的新多序列族；

**🔧 技术方法**

采用函数域、射线类域理论、Drinfeld模、Riemann‑Roch空间等工具进行构造与分析；

**📊 数据集**

主要使用理论上的函数域族（如Hermitian、Suzuki、Garcia‑Stichtenoth塔等），不涉及实验数据集；

**📈 对比分析**

与已有Hermitian、cyclotomic等构造比较，线性与非线性复杂度均保持或提升，尤其在高维、长周期情况下表现优异；

**⚠️ 局限性**

受限于对函数域射线类群结构和足够有理点的要求，实际实现受域扩展与计算复杂度的影响。

---

## 447. Spot-the-shift: Evaluating Grounded Image Difference Captioning of Long-term Changes

**arXiv ID:** 2609.10356 | [PDF](https://arxiv.org/pdf/2609.10356v1)

**作者:** Benedetta Liberatori `[一作]` (University of Trento), Monika Wysoczańska `[通讯]` (valeo.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个面向长时序街景变化的基于图像对的有语义描述与空间定位的基准任务（Grounded Image Difference Captioning），并构建了首个公开的多城市、多年时间跨度的真实数据集；

**💡 创新点**

创新点在于：①将自然语言差异描述与像素级掩码进行联合标注，实现多模态的精细差异定位；②设计了针对差异单位的精确评估指标（精度/召回），并证明其与人工评价高度一致；③提出了自动化合成数据生成流程，利用真实图像配合多步图像编辑与特征对齐，生成既多样又可评估的训练样本；

**🔧 技术方法**

使用的技术包括：多模态大型语言模型（MLLM）如 Gemini、Qwen、Molmo 等；图像编辑生成器与视角变换；基于 DINOv3 的特征匹配进行差异定位；SAM 做掩码提取；文本分解与句子嵌入（Sentence‑BERT）进行差异单位匹配；

**📊 数据集**

数据集为基于 Mapillary Street‑Level Sequences (MSLS) 的图像对，覆盖 15 个城市、12 个国家，时间跨度 7 年，含 939 对图像，已标注自然语言差异描述与对应的像素掩码；

**📈 对比分析**

在零样本和微调两种设置下与多款现有 MLLM 进行对比。零样本下大模型精度低于 15%，假阳性高；微调后，模型精度提升约 4.4% F1，TNR 提升 15.7%，假阳性率降至 0.8%，且定位精度保持不变；

**⚠️ 局限性**

局限性包括：①数据集规模仍有限，难以覆盖所有真实城市场景；②对多重变化的处理仍受限，现有模型难以一次性捕捉多重细粒度差异；③合成数据的真实性与多样性仍有提升空间；④目前评价指标仍无法完全覆盖所有细粒度错误（如方向错误）。

---

## 448. Beyond One-Size-Fits-All: Sample-Adaptive Strategy Routing for Vision Token Pruning in MLLMs

**arXiv ID:** 2609.10346 | [PDF](https://arxiv.org/pdf/2609.10346v1)

**作者:** Haiji Liang `[一作]` (National University of Singapore), Wangbo Zhao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出VIP-Router实现视觉令牌裁剪的样本级自适应选择

**💡 创新点**

利用低成本视觉-文本特征预测每种裁剪策略的成本感知效用，弥补平均-最佳与样本-最佳差距

**🔧 技术方法**

CLIP预览编码、文本到视觉注意力、共享ratio条件多层感知机

**📊 数据集**

VTC-Bench Group A（包含GQA、MMBench、MME等）以及跨骨干零样本迁移到ScienceQA-IMG、MMMU等

**📈 对比分析**

与固定裁剪策略、Best Fixed及Per‑Sample Oracle对比，VIP‑Router在所有裁剪比例上实现26.9%相对准确率提升、22%相对效用提升，并在不同M‑LLM骨干上保持优势

**⚠️ 局限性**

仅针对Group A的评估，跨架构零样本迁移效果有限，需进一步验证在更大范围和不同难度任务上的适用性

---

## 449. From Symbolic Perception to Logical Deduction: A Framework for Guiding Language Models in Geometric Reasoning

**arXiv ID:** 2609.10335 | [PDF](https://arxiv.org/pdf/2609.10335v1)

**作者:** Weichen Dai `[一作]` (University of Science and Technology of China), Yi Zhou `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于大型语言模型的几何推理框架，结合几何视觉解析器和符号求解器，实现了对平面几何问题的可解释解答。

**💡 创新点**

提出了专用几何视觉解析器与定理推理模块，将视觉信息转化为符号表达，并通过外部定理库引导LLM避免幻觉，从而在复杂几何题上匹配甚至超越顶尖多模态模型。

**🔧 技术方法**

采用YOLO+Hough变换+OCR+Texteller进行图像解析，基于符号推理的定理推理模块以及DeepSeek‑R1 LLM，辅以少量大规模推理样本训练与众数投票增强。

**📊 数据集**

在自构建的ZhongkaoGeo‑L1/L2/L3（分别来自2023‑2025年中国中考）上评测，并与公开数据集GeoQA、Geometry3K进行对比。

**📈 对比分析**

将模型在L1/L2的准确率和L3的计分率与多项公开基线（Qwen3、DeepSeek‑R1、GPT‑o1、Gemini 2.5‑Pro）对比，标准化输入；结果显示在L1/L2上达到92.13%/74.30%（多样本投票后93.26%/78.31%），在L3上的平均计分率为88.4%，高于Gemini 2.5‑Pro的87.2%。

**⚠️ 局限性**

仍受限于对图像解析的准确性、对非几何优化/不等式等高级任务的支持有限，且仅验证于中考题库，跨域推广仍需进一步验证。

---

## 450. TRACE: Trajectory-robust Admission with Evidence Ordering for Efficient GUI Agents

**arXiv ID:** 2609.10297 | [PDF](https://arxiv.org/pdf/2609.10297v1)

**作者:** Yuhao Wang `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文提出了一套训练无关的视觉令牌裁剪框架，用于可重用生命周期的 GUI 代理，以在多步交互中高效利用高分辨率截图。

**💡 创新点**

创新点在于将界面布局先验与指令相关性和特征新颖度结合，生成嵌套的证据排序，并通过原生令牌覆盖修复和单调 KV 缩减，实现一次写入、后续递减的可重用视觉状态。

**🔧 技术方法**

采用了布局导出的交互先验、嵌套证据排序、原生令牌覆盖修复和单调 KV 缩减等技术，并结合 OmniParser 检测、正态化、余弦相似度、正交残差选择等算法。

**📊 数据集**

在六个公开 GUI 基准（ScreenSpot-v2, ScreenSpot-Pro, MMBench-GUI L2, OmniGUI, Mind2Web, AndroidControl）以及多规模模型上进行评估。

**📈 对比分析**

与现有 PruneSID、FastV、CDPruner 等方法相比，在单步和多步任务下在紧凑预算下平均提升 20-30% 的准确率，同时将推理延迟和视觉 KV 内存减少 2.4 倍。

**⚠️ 局限性**

限制包括对 OmniParser 检测质量的依赖、对稀疏覆盖修复策略的手工设计以及在极低预算下仍可能缺失细节区域。

---

## 451. The Semantic Bottleneck: Leveraging Semantic Representations for Non-Invasive Speech Decoding

**arXiv ID:** 2609.10296 | [PDF](https://arxiv.org/pdf/2609.10296v1)

**作者:** Gilad D. Landau `[一作]`, Oiwi Parker Jones `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了非侵入式语音解码框架Brain2Semantics2Text，先将句子级MEG信号映射至语义嵌入空间，再通过语义嵌入反演恢复文本。

**💡 创新点**

创新点在于使用语义嵌入瓶颈代替低级音素/单词解码，采用句子级语义解码与软可逆性评估，结合SigLIP+VICReg多目标损失以及迭代语义嵌入反演技术。

**🔧 技术方法**

技术包括MEG空间注意力+膨胀时间卷积、Transformer自注意力聚合、SigLIP对比学习与VICReg正则、选择ADA语义嵌入空间、以及基于条件生成的迭代语义嵌入反演模型。

**📊 数据集**

使用LibriBrain Sherlock Holmes子集的单受试者MEG数据，总计约62小时，划分训练、验证与测试。

**📈 对比分析**

与BrainECHO及词级解码基线比较，使用WER、BLEU-1、ROUGE-1和BERTScore等指标；Brain2Semantics2Text在BLEU-1/ROUGE-1和BERTScore上优于BrainECHO，神经信号提升更显著，但词级指标仍低于需词对齐的词级解码方法。

**⚠️ 局限性**

主要限制包括低信噪比与有限训练数据导致语义映射困难，模型对特定语料库过拟合，语义嵌入反演未专门针对语义优化，缺乏跨受试者泛化能力，且需要更多主题与概念多样化的数据。

---

## 452. Training Trajectories Determine Circuit Removability in Annealable Soft-Prior Transformers

**arXiv ID:** 2609.10287 | [PDF](https://arxiv.org/pdf/2609.10287v1)

**作者:** Zonglin Yang `[一作]` (Guangdong Police), Jiayu Liu `[通讯]` (Guangdong Police)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了训练路径对可调软先验Transformer检索电路可移除性的影响。

**💡 创新点**

发现通过平滑的“fade‑to‑zero”门控调度，先验可以在训练后被移除而不损失性能，而硬切换或后续继续训练无法实现同样效果。

**🔧 技术方法**

使用可门控的软位置先验Transformer、门控调度（自由、强制、计划）以及多种训练路径。

**📊 数据集**

在合成任务上评估：关联回忆、马尔可夫诱导和线性回归ICL。

**📈 对比分析**

与多种位置先验基线比较，在关联回忆任务中零门准确率从≈0.09提升至≈0.73；马尔可夫诱导同样取得显著提升；线性回归ICL为边界案例，未能体现可移除性。

**⚠️ 局限性**

结果仅适用于小规模离散检索任务，对更大模型、不同先验类型或长度外推的鲁棒性有限；内容先验几乎不起作用，且调度形状实验不够全面。

---

## 453. MOONWALK: Mediating Operations with Intent-Evidence-Action Alignment Across Junior-Supervisor Review Workflows in Animation/VFX Pre-Production

**arXiv ID:** 2609.10385 | [PDF](https://arxiv.org/pdf/2609.10385v1)

**作者:** Shih-Yu Lai `[一作]` (National Taiwan University), Xiang Anthony Chen `[通讯]` (UCLA)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并评估了MOONWALK系统，一种通过共享项目记录、参考对齐和结构化反馈来维护动画与VFX预制制作中的创意意图、证据与行动链的工具；

**💡 创新点**

提出了意图–证据–行动（Intent‑Evidence‑Action）设计框架，并将其实现为结构化工作流，保持审查过程中的意图连续性、可检验证据与可执行任务，并将AI角色限定在协作协调而非审美判断；

**🔧 技术方法**

采用多模态大语言模型（GPT‑4o‑mini、Gemini 2.0 Flash、Claude 3.5 Sonnet）进行视觉观察、规范与参考对比及综合评估，并实现11维度的自动分析；

**📊 数据集**

使用来自两家专业动画/VFX工作室的内部项目材料（4个项目、4幅作品、4个参考图像），并结合ShotGrid等现有工作流记录；

**📈 对比分析**

通过在19名专业从业者中进行双条件的实验（MOONWALK vs Chat‑Only）与后续问卷（Likert量表与比较题），统计显著性检验显示MOONWALK在12/13项指标上显著优于中立点，且在多项比较题中获得多数偏好，说明其在意图对齐、任务可执行性与透明度方面优于纯对话接口和现有工作流；

**⚠️ 局限性**

局限性包括单次实验、缺乏纵向生产日志评估、仅针对内部数据且未验证在不同工作室生态中的可迁移性；系统依赖完整的共享记录，未覆盖非录入的口头交流；AI仅做协调，未能解决过度依赖与创作主导权的问题；以及与现有工具链的集成仍需改进。

---

## 454. PACE: Perceived-Latency-Aware Cascading Service Routing and Filler Control for QoE-Efficient Retrieval-Augmented Dialogue Serving

**arXiv ID:** 2609.10372 | [PDF](https://arxiv.org/pdf/2609.10372v1)

**作者:** Lin Huang `[一作]`, Suihan Xiao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个基于自适应门控、填充控制和波动性检测的对话服务框架 PACE，实现了低延迟、高质量的查询处理。

**💡 创新点**

创新在于将实时负载指数与可变阈值结合的适配式杀手开关、基于瞬时路径优先级的填充决策，以及跨域迁移不需调优的波动性感知访问控制。

**🔧 技术方法**

采用 EWMA、门控状态机、填充控制、阈值自适应、缓存与 L2 检索级联、以及概率论证明等技术。

**📊 数据集**

使用 DuReader‑3k、CarQA‑Volatile、汽车领域对话语料及公开开域查询作为实验数据。

**📈 对比分析**

与静态阈值、纯 LLM、以及不同配置的对照实验对比，结果显示 PACE‑full 在 c=16 时平均延迟下降至 0.28 s（比纯 LLM 快 2.5 倍），质量保持 4.65–4.81/5。

**⚠️ 局限性**

仅在文字对话栈上验证，缺乏完整机器人多模态实验；跨域迁移在不同垂直领域仍需验证；系统对提示语的依赖可能限制可移植性。

---

## 455. OmniMed-FL: A Robust Multimodal Federated Learning Framework for Clinical Diagnosis

**arXiv ID:** 2609.10364 | [PDF](https://arxiv.org/pdf/2609.10364v1)

**作者:** Ayush Debnath `[一作]` (Indian Institute of Technology Kharagpur), Sudip Misra `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了OmniMed‑FL，一个多模态联邦学习框架，用于五分类临床诊断（正常、肺炎、COVID‑19、胸腔积液、心脏肥大），并在合成与公开胸部X射线+合成临床笔记的代理数据上系统评估融合策略、初始化、缺失文本填补等。

**💡 创新点**

首次将视觉和文本信息在联邦学习框架中融合，提出多模态融合规则的全方位比较，结合抗崩溃正则化与类平衡采样，量化标签偏斜与客户端数量对性能与通信成本的影响。

**🔧 技术方法**

使用ViT‑Base/16作为视觉编码器、DistilBERT作为文本编码器，配合八种多模态融合方法（拼接、注意力等），并与FedAvg、FedProx、FedMME、SCAFFOLD‑AdamW等聚合策略以及类平衡采样、熵多样性正则、置信度惩罚等技术进行实验。

**📊 数据集**

采用两套代理语料库：Dataset A（3,000个合成X射线+合成笔记）和Dataset B（3,000个公开胸部X射线+同类合成笔记），每类600例，图像与笔记仅按类别配对而非真实患者。

**📈 对比分析**

通过与局部训练、FedAvg、FedProx、FedMME等基线对比，在α=0.1、K=5下，FedProx macro‑F1达到0.737；多模态融合最高0.956；在不同客户端数与α下观察到标签偏斜显著影响F1，通信量随客户端数线性增长。

**⚠️ 局限性**

仅使用合成/无患者关联的代理数据，无法评估真实诊断准确性；实验为顺序模拟，未测算并发、延迟及异构设备成本；缺乏差分隐私和攻击鲁棒性评估，结果仅基于两种随机种子，具描述性而非统计显著性。

---

## 456. Why Is Video Still So Expensive? A Survey of Inference-Efficiency Mechanisms in Video and Audiovisual LLMs

**arXiv ID:** 2609.10355 | [PDF](https://arxiv.org/pdf/2609.10355v1)

**作者:** Killian Steunou `[一作]` (Institut Polytechnique de Paris), Mounîm A. El Yacoubi `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统化了截至 2026 年 VideoLLM（视频大语言模型）的推理效率机制，涵盖帧采样、视觉/音频编码、连接器压缩、LLM 前填充与解码等四个 pipeline 阶段。

**💡 创新点**

创新点在于：①按推理流水线阶段构建统一的 taxonomy，揭示各技术对整体成本的具体贡献；②在同一主机（7‑8B LLM）与统一输入协议（如 MVBench、Video‑MME、EgoSchema 等）下，对多篇论文提供可复现的准确率‑成本对比，弥补了以往缺乏标准化评估的空白；③深入剖析多模态音频效率与视觉/音频交互的空缺，提出未来评估方向。

**🔧 技术方法**

使用了多种技术手段：
- 时序帧采样（AdaFrame、AKS、TSPO 等）
- 轻量化视觉/音频编码器（MobileCLIP、VideoMamba、MViT、UniFormer 等）
- 连接器压缩与投影（Q‑Former、Perceiver Resampler、FAVOR 等）
- 视觉/音频 token 归并与剪枝（VisionZip、HoliTom、FastV、PruneVid 等）
- LLM 内部 KV 缓存压缩、稀疏注意力与摘要 token（MMInference、ReKV、VoCo‑LLaMA 等）
- 流式与离线长视频记忆压缩（StreamKV、StreamingTOM、Flash‑VStream 等）。

**📊 数据集**

主要评估数据集：MVBench、Video‑MME、EgoSchema、LongVideoBench、MLVU、RVS‑Ego / RVS‑Movie、Video‑MMA 等多任务集合，覆盖视频分类、检索、问答、检索、对话等多种能力。

**📈 对比分析**

比较方法：在同一 7B‑8B LLM、统一视频长度、帧数与分辨率的基准下，对每种机制给出 token 保留率、FLOPs、延迟与准确率的量化结果。典型性能表明：
- 帧采样与 token 剪枝可将 token 数量压缩至 10‑30% 同时仅损失 0‑5% 的准确率；
- 视觉编码器轻量化如 VideoMamba、MViT‑v2 等可在保持 80‑90% Kinetics‑400 准确率的前提下，显著降低 2‑3× FLOPs；
- LLM 内部 KV 缓存压缩可实现 30‑70% 内存减少，解码速度提升 1.5‑3×；
- 结合多阶段压缩（如 HieraVid + FastV）可实现 30% token 保留且 <2% 的准确率下降，prefilling FLOPs 下降 4‑5×。

**⚠️ 局限性**

局限性：
- 评估缺乏统一标准，跨论文对比往往基于不同任务、视频长度、帧数与分辨率，导致结果不完全可比；
- 绝大多数研究侧重计算/内存指标，能耗与功耗指标仍缺失；
- 音频效率研究相对不足，现有音频压缩方法多在同一主机上验证，缺乏跨模型泛化；
- 许多方法只在单一视频长度或场景上测试，未系统评估长视频、实时推理与多模态场景的综合表现；
- 由于大模型规模与训练成本高，公开复现和基准化实验仍受限，导致部分方法报告难以复现。

---

## 457. TimeCues Studio: A Workspace for Music Annotation and Algorithm Prototyping

**arXiv ID:** 2609.10338 | [PDF](https://arxiv.org/pdf/2609.10338v1)

**作者:** Sapir Caduri `[一作]` (Bar-Ilan University), Yoav Goldberg `[通讯]` (Bar-Ilan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款名为TimeCues Studio的开源多媒体音乐注释与算法评估工作空间，集成团队协作、模糊注释、多标记类型、算法对比与半自动化注释。

**💡 创新点**

创新点包括支持多候选模糊注释与关键/可选标记、同一工作空间实现团队注释与算法评估、Python沙盒自定义检测器与AutoGuess一致性聚类，以及3频段波形可视化提升注释效率。

**🔧 技术方法**

采用React/TypeScript前端、Node/Python后端、Docker Compose部署、Python评估库、Librosa、Demucs等音频特征提取，以及WebSocket实时同步与可视化绘图库。

**📊 数据集**

提供Demo三首CC0 EDM曲目，支持用户上传任意音频并导出为JAMS、MIDI、Audacity等格式；主要使用的语料库为自由版权的EDM曲目。

**📈 对比分析**

通过内置多种基线算法（新颖度曲线、MSAF、张量分解、深度模型等）与自定义检测器，使用多候选模糊评估指标（加权召回、F1等）进行对比，AutoGuess聚类提供可调阈值的半自动基线，实验显示各算法在精度与召回上的差距可视化呈现。

**⚠️ 局限性**

局限性包括未在多乐种上进行广泛验证、对实时性能与大规模语料的可扩展性评估不足、Python沙盒自定义检测器易受脚本错误影响、以及模糊注释仍需人工验证，未实现完全自动化。

---

## 458. Odometer-Agnostic Drift Correction Using OpenStreetMap Lane Geometry

**arXiv ID:** 2609.10336 | [PDF](https://arxiv.org/pdf/2609.10336v1)

**作者:** Joaquin Caballero `[一作]` (Distance Technologies Oy), Jarno Ralli `[通讯]` (Distance Technologies Oy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种仅利用OpenStreetMap（OSM）车道中心线进行实时漂移校正的轻量化、与里程计无关的方法。

**💡 创新点**

创新点在于直接将最近轨迹段与稀疏车道几何进行对齐，通过滑动窗口与修剪ICP实现在线纠正，无需密集地图、昂贵预处理或复杂匹配管线。

**🔧 技术方法**

采用KD‑树检索车道点、正交投影与方向一致性约束进行对应匹配，滑动窗口内使用修剪ICP求解SE(2)变换，实现在线校正。

**📊 数据集**

在KITTI与KITTI‑360两个主流自动驾驶基准数据集上验证，使用多种里程计后端（KISS‑ICP、LiODOM、ORB‑SLAM3、Basalt）。

**📈 对比分析**

与最近的OSM辅助漂移校正方法（Li et al., Kurda et al., TOM‑Odometry）对比，本文在大多数序列上取得更低的APE‑2D平均误差，且在不同传感器与后端上均表现出显著提升。

**⚠️ 局限性**

局限包括车道匹配在高速、平行车道或频繁车道变更场景下可能模糊，缺乏对多层道路（桥梁、隧道）的高度约束；对OSM地图误差敏感，校正后轨迹可能出现局部不连续。

---

## 459. Dimensionality Reduction for Hyperspectral Image Classification

**arXiv ID:** 2609.10334 | [PDF](https://arxiv.org/pdf/2609.10334v1)

**作者:** Mohamed Cherifi `[一作]` (Ecole Militaire Polytechnique), Abdennour Hacine Gharbi `[通讯]` (Universite De Bordj Bou Ariridj)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在高光谱图像分类任务中，结合降维方法（PCA、LDA）与分类器（KNN、SVM、Random Forest）的效果，并对超参数进行交叉验证；通过实验验证了PCA+RF组合在印度松谷数据集上取得最佳性能。

**💡 创新点**

创新点在于系统比较了不同降维与分类器组合的整体准确率和Kappa系数，并通过5折交叉验证对模型参数进行细致调优，验证了PCA+RF在此任务中的优势。

**🔧 技术方法**

主要使用的技术包括主成分分析（PCA）、线性判别分析（LDA）、K最近邻（KNN）、支持向量机（SVM，RBF核）和随机森林（RF），以及5折交叉验证和超参数调优。

**📊 数据集**

使用的数据集为印度松谷（Indian Pines）高光谱数据集，尺寸145×145像素，220个波段，16类（后简化为6类）。

**📈 对比分析**

方法通过计算整体准确率（OA）和Kappa系数进行比较；实验表明PCA+RF组合取得95.5% OA、92.6% Kappa，略优于PCA+KNN（94.1% OA）和PCA+SVM（93.8% OA）；LDA组合性能相对较低。

**⚠️ 局限性**

局限性包括仅在单一数据集上验证；降维仅采用线性方法（PCA/LDA），未尝试非线性降维；分类器参数空间有限；未评估计算成本或实时性等实际部署因素。

---

## 460. Decoupled Self-Forcing Distillation for Streaming Talking Head Generation

**arXiv ID:** 2609.10317 | [PDF](https://arxiv.org/pdf/2609.10317v1)

**作者:** Yanru An `[一作]` (Shanghai Jiao Tong University), Wenjun Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Motar，一种实时流式说话人头像生成框架，将音频与文本条件先投影到低维身份去耦合的运动空间，然后使用因果 Transformer 生成连续运动潜变量，再由预训练的扩散渲染器实时渲染视频帧。

**💡 创新点**

创新点包括：① 将条件聚焦于运动潜在空间，避免将音频直接作用于包含身份、纹理等无关信息的像素空间；② 采用分层条件路由——全局 Q‑Former 负责情感与幅度的序列级控制，局部窗口交叉注意力负责帧级口型同步；③ 引入解耦自我强迫蒸馏，在同一冻结的双向渲染教师下同时解决运动生成器与渲染器的曝光偏差；④ 仅使用 77M 参数的运动生成器，结合可流式的高容量渲染器，显著提升实时性与可扩展性。

**🔧 技术方法**

技术手段包括因果 Transformer + 连续扩散头、Q‑Former 融合、分层全局/局部交叉注意力、Self‑Forcing 与 DMD（Diffusion Model Distillation）自我强迫蒸馏、运动空间的对抗损失、Stable Diffusion 作为渲染器、bounded look‑ahead 以实现流式生成。

**📊 数据集**

使用公开的 MEAD 与 Hallo3 两个数据集，分别覆盖情感对话与野生口型视频，并通过 Qwen2.5‑VL‑7B 生成描述情绪与头部运动的文本标签。

**📈 对比分析**

与 SadTalker、AniPortrait、Hallo3、StableAvatar、LiveAvatar、AvatarForcing 等两阶段与端到端说话人生成方法在 MEAD/Hallo3 子集上进行对比，评估指标包括 FID、FVD、CSIM、E‑FID、ΔSync‑C/D、FPS、延时与参数量。Motar 在同步度（ΔSync‑C/D）上与重建上限接近，FPS 达 15.4，首帧延时 1.39 s，视觉质量与渲染器上限相近，尽管在 FID/FVD 上略逊于大型端到端模型，但在实时性和多模态控制上明显优于基线。

**⚠️ 局限性**

局限性主要在于：① 视觉质量受限于 Stable Diffusion 渲染器的表达能力；② 对极端光照、极细表情细节的捕捉仍有限；③ 需要双流并行实现，增加实现复杂度；④ 仍需大规模预训练的冻结教师，若教师不足可能影响性能。

---

## 461. One Loop, Two Gains: Can Active Learning win the Lottery for Free?

**arXiv ID:** 2609.10311 | [PDF](https://arxiv.org/pdf/2609.10311v1)

**作者:** Benedikt Tscheschner `[一作]` (University of Technology Graz), Marc Masana `[通讯]` (University of Technology Graz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在池基主动学习循环中加入一次迭代幅度剪枝，以无额外训练成本发现稀疏赢家子网络

**💡 创新点**

将迭代幅度剪枝与主动学习的重复训练循环整合，形成无需额外开销的 Ticket‑Active‑Learning 框架，实现高稀疏度下的赢家子网络发现

**🔧 技术方法**

使用迭代幅度剪枝（IMP）、稀疏掩码、随机层级重启、稀疏训练与微调，以及多种主动学习采样函数（Margin、CoreSet、BADGE、SAAL、MaxHerding、UHerding）

**📊 数据集**

CIFAR‑100、Imagewoof、Tiny‑ImageNet、Places365；模型包括 ResNet‑18/50、ConvNext‑v2、DeiT‑Small、DINO ViT‑S，包含自监督与监督预训练

**📈 对比分析**

与稠密模型、随机稀疏、随机剪枝对比；在六种采样函数与两种架构下，稀疏模型在约95%稀疏度仍能匹配稠密模型精度，计算量（MAC）约减一半，效果与稠密等价

**⚠️ 局限性**

仅在图像分类任务评估；使用无结构稀疏，实际硬件加速受限；需要手动设定稀疏率与重启比例；未验证对检测、NLP 等任务的普适性

---

## 462. View-Structured Conformal Prediction for 3D Gaussian Splatting

**arXiv ID:** 2609.10307 | [PDF](https://arxiv.org/pdf/2609.10307v1)

**作者:** Junzheng Chu `[一作]` (Nankai University), Zhenwei Shi `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于视角结构的 conformal 预测方法（VSCP），为 3D Gaussian Splatting（3DGS）生成可量化的 RGB 区间，使新视角在给定置信水平下至少覆盖一定比例的像素。

**💡 创新点**

创新点在于：①将不确定性尺度拆分为可迁移的视角难度因子 a 和渲染器自适应的空间形状 b，二者乘积构成预测区间宽度；②使用基于视角的分层 conformal 校准（View‑CP），实现对视角事件的分布无关、有限样本有效性；③通过对方程 a^⋆(b) 的解析推导得到视角难度的风险最优目标，完成对宽度的精确分解。

**🔧 技术方法**

核心技术包括 3D Gaussian Splatting 渲染、残差归因构造空间形状 b、基于相机与渲染特征的视角难度预测器 a（对数线性模型）、分层 split‑conformal 校准、以及四次 rasterization 的高效实现。

**📊 数据集**

使用 Tanks & Temples、Deep Blending 以及 Mip‑NeRF360 三大公开数据集（共 13 场景）进行训练、验证与迁移实验。

**📈 对比分析**

与常数尺度、3DGS‑U、GAVIS 可见性场、以及十模型集成的标准差等基线对比，VSCP 在相同视角覆盖率下平均宽度减少约 22%（比常数尺度高效，几乎匹配十模型集成宽度但仅需一模型和四次 rasterization），并在跨数据集迁移时保持有效性。性能在所有 13 场景上均表现出显著优势。

**⚠️ 局限性**

局限性包括：需要至少 9 个校准视角才能保证有效区间；对极难视角的宽度仍相对较大；在数据量极少或视角分布与训练集差异极大时，视角难度预测的精度会下降；方法仍依赖于 3DGS 的残差估计，若渲染器更改可能需重新调整。

---

## 463. A Dominant Diffuse Phase in the Sparse Autoencoder Phase Diagram

**arXiv ID:** 2609.10299 | [PDF](https://arxiv.org/pdf/2609.10299v1)

**作者:** Alexis D. Plascencia `[一作]` `[通讯]`, Alexis D. Plascencia

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

对MAIS‑O43实验网格下的稀疏自编码器（SAE）进行大规模评估，发现无论在全批还是小批梯度下降下，都未出现完整字典恢复或特征合并，而是出现一种扩散相（近乎完美重构但学习到的原子与真实特征差距显著），

**💡 创新点**

首次系统测量并公开MAIS‑O43相图，揭示训练SAE的相图与理论最优解不同，并提出扩散相这一新的现象，为解释稀疏自编码器在层级结构下的表现提供实证依据。

**🔧 技术方法**

使用ReLU+ℓ₁稀疏编码器、全批Adam和标准小批Adam优化器、严格的最大匹配与合并判定算法、向量化训练和精确的复现套件（PyTorch/NumPy、固定种子、TF32禁用）。

**📊 数据集**

采用合成数据生成器：n=64、m=256、γ可调的父子嵌套字典Φ，训练样本2¹⁸，评估样本2¹⁶，覆盖10个网格点以及完整165格的离线小批实验。

**📈 对比分析**

与理论最优（完整恢复或合并）对比，使用恢复率、合并率、匹配特征比例、余弦相似度、重构误差等指标。结果显示：恢复率0%、合并率0%，匹配特征最高仅3.6%，重构误差极低（≈10⁻⁶–10⁻³），但学习代码密度比真实生成过程高10–30倍。

**⚠️ 局限性**

仅覆盖有限的网格点（10个）和完整网格的离线小批实验；未探索不同SAE变体（如top‑k、gated、树结构等）；仅使用单一字典和数据生成器，未评估多特征全局最优解的可能性。

---

## 464. A traffic management system for large and heterogeneous vehicles in narrow industrial environments

**arXiv ID:** 2609.10400 | [PDF](https://arxiv.org/pdf/2609.10400v1)

**作者:** Alessandro Bonetti `[一作]` (University of Modena and Reggio Emilia), Lorenzo Sabattini `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套针对大型、异构自动引导车（AGV）在狭窄且非标准化工业环境中进行高密度交通管理的实时协调系统，基于L‑MAPF算法与NURBS道路图实现局部最优、连续安全调度，并通过路径分配器与死锁检测/解决模块实现与真实AGV的无缝交互。

**💡 创新点**

创新点包括：1）将基于时间窗的冲突搜索CBS与滚动时域冲突解决相结合，形成任何时段可增量扩展的ABH‑CBS求解器；2）在NURBS道路图上实现多种AGV尺寸与动力学的连续路径规划；3）设计了基于冲突集预计算的实时路网冲突检测与路径分配机制；4）引入了扩展走廊策略和层级死锁检测/处理，提升系统鲁棒性与吞吐率。

**🔧 技术方法**

核心技术包括L‑MAPF（终身多智能体路径规划）、NURBS曲线路网建模、Bounded Horizon CBS（有限时域冲突搜索）、Rolling Horizon Conflict Resolution（滚动时域冲突解决）、Safe Interval Path Planning（安全间隔路径规划）、多任务优先级与即时路径分配、深度优先搜索（DFS）用于死锁检测。

**📊 数据集**

实验使用了与意大利Gruppo TecnoFerrari S.p.A.合作的三种真实工业布局（小型、 中型、大型），分别包含多类AGV、复杂走廊与非标准化网格；此外使用了基于这些布局生成的路网图与任务列表作为数据集。

**📈 对比分析**

与传统基于规则的调度、工业标杆方法以及另一L‑MAPF变体进行对比，实验结果表明系统吞吐率提升最高可达11%，并保持连续运行与实时适应；在高密度与非标准化环境下，ABH‑CBS实现了更低的延迟与更高的路径质量。

**⚠️ 局限性**

局限性包括：1）对极其拥挤或极小走廊的实时性能尚未全面评估；2）死锁处理依赖于CBS求解，若涉及车辆数过多可能导致求解时间延长；3）路径分配器对执行不确定性仅通过时间窗约束处理，极端扰动仍可能导致碰撞；4）系统依赖预先生成的NURBS路网，无法动态重构布局。

---

## 465. Retrofitting Code Using LLMs to Support Exceptional Behavior

**arXiv ID:** 2609.10397 | [PDF](https://arxiv.org/pdf/2609.10397v1)

**作者:** Linghan Zhong `[一作]` (University of Texas at Austin), Milos Gligoric `[通讯]` (University of Texas at Austin)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

自动为缺少异常抛出语句的方法补充异常处理代码，并使其通过给定的异常行为测试（EBS）

**💡 创新点**

提出基于异常行为测试驱动的“异常代码补全”任务，并通过静态与动态分析结合上下文来构造 LLM prompt，从而显著提升 LLM 在此任务上的表现

**🔧 技术方法**

采用大型语言模型（如 Qwen2.5‑coder、Llama3.1 等）+ 静态分析（提取异常构造器签名、方法/字段签名）+ 动态分析（收集异常运行时信息、代码覆盖）+ Prompt Engineering + 自我修复迭代

**📊 数据集**

从 CodeSearchNet 的 Maven Java 项目构建的基准集，包含约 5 千个方法（含缺失 throw）及其异常行为测试，进一步补充 EvoSuite 与 Randoop 生成的自动化测试

**📈 对比分析**

与仅提供方法+测试的基线对比，使用 Pass@k（编译、EBS、全部测试、工具测试）四个指标；在 Qwen2.5‑coder 32b 上相较基线提升约 5‑10% 的 Pass@5/Pass@10，闭源模型 GPT‑5‑mini 亦显著提高；自我修复迭代进一步提升成功率

**⚠️ 局限性**

局限性包括：只处理方法内部直接抛出的异常，未覆盖由被调用方法抛出的异常；数据集仅限 Maven Java 项目，可能对其他语言或构建系统的泛化有限；Pass@k 仅衡量测试通过，未必等价于功能一致；LLM 随机性导致结果不稳定

---

## 466. Enhanced Deformable Convolution with Center-invariant Offset and Edge-aware Mask

**arXiv ID:** 2609.10387 | [PDF](https://arxiv.org/pdf/2609.10387v1)

**作者:** Yixiao Li `[一作]` (Beihang University), Wei Zhou `[通讯]` (Cardiff University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种能够在更大卷积核下有效学习形变位移的增强可变形卷积（EDC），并将其作为插件式模块用于语义分割网络。

**💡 创新点**

核心创新在于中心不变位移模块（COM）与边缘感知掩码模块（EMM）的双分支设计，使得可变形卷积仅在边缘区域进行形变，同时保持大感受野。

**🔧 技术方法**

采用双分支可变形卷积、深度可分离卷积、边缘检测（Sobel）、二值逻辑回归等技术实现模块化实现。

**📊 数据集**

在 ADE20K、MS COCO、Cityscapes、PASCAL VOC 四大公开语义分割数据集上进行评估。

**📈 对比分析**

与传统 DCN、DCNv2、Entire DCN、DCNv3 以及标准卷积做 plug‑and‑play 对比，EDC 在所有主要指标上均超越对手，尤其在 5×5 及更大核尺寸下显著提升性能。

**⚠️ 局限性**

对极大卷积核（如 15×15、17×17）仍存在参数与 FLOPs 激增、收敛不稳定的问题，且在轻量化模型中实现困难。

---

## 467. Shape-guided Gaussian Splatting for Sparse-View X-ray 3D Reconstruction

**arXiv ID:** 2609.10376 | [PDF](https://arxiv.org/pdf/2609.10376v1)

**作者:** Pranav Poudel `[一作]` (Polytechnique Montréal), Herve Lombaert `[通讯]` (Polytechnique Montréal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了形状引导的高斯弹射框架，用于稀视X射线三维重建。

**💡 创新点**

创新点在于将高斯原语绑定到统计形状模型，并在初始化和优化阶段同时引入形状与密度先验，使重建在极稀视条件下保持解剖一致性并显著提升质量。

**🔧 技术方法**

使用的技术包括3D高斯弹射（Gaussian Splatting）、基于PCA的统计形状模型、密度图谱先验、光度损失（L1、D‑SSIM）、总变差、形状和密度正则化。

**📊 数据集**

数据集采用NMDID的758份CT扫描，使用TotalSegmentator生成股骨分割，构建形状模型后在6个留样子上进行实验。

**📈 对比分析**

与SOTA R^2‑Gaussian进行比较，采用PSNR和SSIM评估；在5视角下平均PSNR提升2.83 dB、SSIM提升0.022，在10视角下亦有提升。

**⚠️ 局限性**

局限性包括仅在股骨上验证，需进一步验证对其他解剖结构的适用性；对形状统计样本的依赖以及在更高视角下增益减小。

---

## 468. SceneHI: High-Resolution 3D-Consistent Scene Texturing with Controllable Illumination

**arXiv ID:** 2609.10363 | [PDF](https://arxiv.org/pdf/2609.10363v1)

**作者:** Athanasios Tragakis `[一作]` (University of Glasgow), Paul Henderson `[通讯]` (University of Glasgow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对多物体室内场景，提出一种零样本生成高分辨率、3D一致性纹理并可控制光照的框架SceneHI。

**💡 创新点**

创新点包括：① 用解析的像素‑纹素映射实现多视角一致的噪声轨迹；② 在高分辨率潜在纹理空间（HRLT）上并行扩散，避免Janus问题；③ 分阶段设计：全局纹理、细节增强（HDTR）和光照烘焙（GCS）；④ 通过模式投票与法线权重实现视角一致的渲染/逆渲染循环；⑤ 支持可选的阴影烘焙与细节提升，兼容主流渲染管线。

**🔧 技术方法**

技术栈：Stable Diffusion 3.5 + ControlNet 作为 2D 扩散模型；Nvdiffrast 进行光线追踪与逆渲染；xatlas 生成 UV；解析的像素‑纹素映射；Patch‑based 高分辨率扩散（HDTR）；生成式阴影烘焙（GCS）与可调光照。

**📊 数据集**

使用 3D‑FRONT 基准的 10 个复杂室内场景；每个场景提供两条文本提示；在每个场景渲染 25 个新视角进行评估。

**📈 对比分析**

通过 CLIP Score、Aesthetic Score、Inception Score 以及 20 名用户的 1–5 评分对比；与 Text2Tex（两种模式）、SceneTex、RoomTex、RoomPainter 等最先进方法比较，SceneHI 在纹理一致性、细节与阴影质量上均优于基线，并将总生成时间降低 80%。

**⚠️ 局限性**

局限性：单卡 RTX A6000 下仍需约 4 小时/场景；对非常近的细节需要更高 HRLT 分辨率，超出普通显存；目前主要验证于室内场景，需进一步测试户外或复杂几何；阴影烘焙仍基于渲染光照，无法实时动态光照。

---

## 469. Precision in Rice Variety Classification using Stacking-Based Ensemble Learning

**arXiv ID:** 2609.10524 | [PDF](https://arxiv.org/pdf/2609.10524v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 470. Non-Existence of PMMS Allocations and a $4/3$-PMMS Guarantee for Additive Chores

**arXiv ID:** 2609.10493 | [PDF](https://arxiv.org/pdf/2609.10493v1)

**作者:** Xiaohui Bei `[一作]` (Nanyang Technological University), Biaoshuai Tao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在可加性偏好下无分割物品的两两最大最小份额（PMMS）公平性，证明了PMMS在加性物品（goods）和任务（chores）下并不总存在，并给出了NP‑难的判定结果；同时提供了两个显式反例（goods的比值226/227，chores的1.102065）以及一个始终能得到4/3‑PMMS分配的算法；

**💡 创新点**

创新点在于：①构造了从加性任务到加性物品的多项式时间还原，进而传递了任务域已知的非存在结果到物品域；②证明了PMMS判定问题的NP‑难性；③给出显式的近似上界与下界，并提出了从价格支持的pEF1分配出发的重新分配与局部交换流程，确保得到4/3‑PMMS分配；

**🔧 技术方法**

主要技术包括：多项式时间还原与等价性证明、精确枚举与数值验证、价格支持的pEF1分配与支撑价法、排序重分配+局部交换的结构性证明、极限/连续性论证以消除非退化情况；

**📊 数据集**

该研究为理论性质，未使用外部数据集，所有结果均为构造性证明与符号计算（包括数值枚举与精确证书）所得到；

**📈 对比分析**

方法在理论上得到下界（不可能性、NP‑难）与上界（4/3‑PMMS），与已有的√5−1/2≈0.618、(√17−1)/4≈0.781等近似因子相比，给出了更紧的上界，并证明了4/3是可实现的；

**⚠️ 局限性**

局限性包括：存在下界与上界之间的明显间隙（goods 0.7808–0.9887，chores 1.102065–1.3333）；对精确PMMS分配的计算仍未可多项式；证明中依赖于精确枚举和数值验证，缺乏纯粹结构化的解析证明；

---

## 471. IdeaAMBIG: Benchmarking Implementation-Critical Gaps in Research-Idea Specifications

**arXiv ID:** 2609.10539 | [PDF](https://arxiv.org/pdf/2609.10539v1)

**作者:** Yiling Ma `[一作]` (Yale University), Arman Cohan `[通讯]` (Yale University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并构建了一个基于证据的单缺陷实例数据集，评估研究想法规格的实现就绪度、缺陷定位与澄清生成。

**💡 创新点**

首个同时包含真实与合成缺陷、证据支持解决方案、并将三项任务拆分的研究方法规范就绪性基准，解决了现有评测缺乏实现面向缺陷检测与澄清的问题。

**🔧 技术方法**

使用13种LLM（包括开源与专有模型）进行三项诊断任务，并通过宏F1、宏缺陷恢复率（Macro‑DRR）与宏澄清成功率（Macro‑CAS）等指标评估；同时配合语义检索、人工标注与自动聚类提升数据质量。

**📊 数据集**

660个实例（163个来自重现报告与GitHub issue，497个合成注入缺陷），覆盖十个研究领域（计算机视觉、NLP、信息检索、图学习、时间序列等）。

**📈 对比分析**

通过与人工评估对比、源代码聚类和模型对比，评估三项任务。最佳模型在真实缺陷定位仅达9.6%宏DRR，但在已知缺陷时澄清成功率可提升至80.6%；在合成缺陷上表现明显更好，说明缺陷定位仍是主要瓶颈。

**⚠️ 局限性**

仅聚焦于AI/ML领域；单缺陷设定；未考察多缺陷交互；澄清仅为单步生成；未通过完整实现验证诊断结果，未来需扩展到其他科学领域与多轮澄清。

---

## 472. Characterizing Language Generation in the Limit: Finite Witnesses and a Separation-Width Hierarch

**arXiv ID:** 2609.10525 | [PDF](https://arxiv.org/pdf/2609.10525v1)

**作者:** Xiaoyu Li `[一作]` (University of New South Wales), Junbin Gao `[通讯]` (University of Sydney)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文提出了在无限语言族上，用有限正例证（witness）来描述在极限下生成语言的可行性，并给出了一个完整的分离宽度（separation-width）层次结构，表明了不同语言族所需的最小正例证大小。

**💡 创新点**

创新点在于：① 用一个单一的、与观测集合无关的 set‑driven 正则化（normalization）将任意顺序依赖的生成器转化为仅基于已见集合的生成器；② 引入正分离宽度概念，刻画所有可生成语言族的最小 witness 大小，并证明每个可能的宽度（从 0 到 ω+1）都可出现；③ 证明了局部维度（如 VC 维、Littlestone 维等）不足以刻画极限生成，可生成性是一个全局性质。

**🔧 技术方法**

使用的技术主要是组合论与递归论：构造了正例证集合、证明了无穷链不可能出现、利用了无穷集合的捕获引理（diagonal capture lemma），并在 Lean 证明助手中实现并检查了核心定理和分离宽度层次。

**📊 数据集**

本文不使用实验数据集，而是基于理论构造的无穷语言族（如所有无限子集、两块并合的语言族等）来展示宽度的不同取值。

**📈 对比分析**

方法比较通过数学证明完成：作者将新的分离宽度与先前的生成条件（如闭包维度、EUC‑cover 等）关联，并用示例语言族说明新的定理是严格更精确的；并在 Lean 中验证了所有证明的正确性；没有实验性能指标。

**⚠️ 局限性**

局限性包括：① 仅适用于可数无穷宇宙；② 证明的 witness 赋值与交叉条件是存在性的，未给出有效计算方法；③ 结果不提供生成器的时间或空间复杂度；④ 只考虑单元素正例输出，未涵盖多元素或多样化输出的生成需求。

---

## 473. Artificial Intelligence Literacy and Sustainable Development: An Ethical Governance and Development Goals Framework

**arXiv ID:** 2609.10489 | [PDF](https://arxiv.org/pdf/2609.10489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 474. Avatar: Toward Autonomous End-to-End Orchestration of Scientific Workflows using LLMs

**arXiv ID:** 2609.10509 | [PDF](https://arxiv.org/pdf/2609.10509v1)

**作者:** Suman Raj `[一作]` (University of Chicago), Ian Foster `[通讯]` (Argonne National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Avatar架构，使用三位actor（调度器、执行器、溯源器）实现可插拔的决策策略，支持传统规则与LLM推理交替控制科学工作流。

**💡 创新点**

创新点在于将科学工作流的关键职责抽象为统一的actor模型，并通过单一的动作目录与适配器实现传统与LLM驱动的无缝切换，从而提供一个可复现、可泛化、可实验的agentic工作流参考框架。

**🔧 技术方法**

技术包括：Actor‑based通信模型、适配器验证、规则与LLM（OpenAI GPT‑4）决策、Academy中间件、Parsl执行子系统、TaskVine管理器、CCTools TaskVine、RDKit+PyTorch模拟器、Colmena活跃学习框架。

**📊 数据集**

使用的数据集与工作负载包括：①模拟的容错DAG任务（自定义矩阵乘法）、②动态流式工作流（随机到达速率）、③QM9分子集（5000个候选分子，100个已标记样本）用于活跃学习。

**📈 对比分析**

对比方法：在规则模式下Avatar与原生TaskVine、Parsl、Colmena的行为和资源利用保持一致；在LLM模式下评估任务重试次数、浪费计算时间、GPU忙时长和工作流吞吐量。实验显示LLM诊断模式在容错任务中可将浪费计算降低55%，在GPU活跃学习中可减少40% GPU忙时长；在高频缩放场景中LLM因推理延迟而表现不如轻量规则。

**⚠️ 局限性**

局限性：实验仅在单机、同质资源上完成，未验证多节点/异构调度决策；LLM推理延迟在需要实时反应的场景下限制其优势；适配器和动作目录仍需人工维护；安全性与错误处理仅覆盖重试与失败分类，缺乏针对具体错误类型的恢复策略。

---

## 475. Wicked Problem, Parsimonious Solution: Securing Electric Vehicle Charging Station Software

**arXiv ID:** 2609.10502 | [PDF](https://arxiv.org/pdf/2609.10502v1)

**作者:** Emma Sheppard `[一作]` (Montana State University), Ann Marie Reinhold `[通讯]` (Montana State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一套基于分层软件质量保证（HSQA）的评估框架，用以系统性评估电动汽车充电站（EVSE）软件的质量与安全性，并将多种行业标准与威胁模型整合进该框架。

**💡 创新点**

创新点在于将已成熟的HSQA方法迁移至电动汽车充电设施软件领域，融合 ISO/IEC 25010:2023、政府与行业最佳实践以及 ISA/IEC 62443、IEEE 1547‑3、ISO/SAE 21434 等多层次标准与威胁模型，构建统一的多层质量安全特征体系，提供从代码级漏洞到系统级安全属性的全景评估。

**🔧 技术方法**

主要技术包括：
- HSQA 元模型及其可扩展的质量安全特征映射；
- 静态分析工具（SonarQube、CVE Binary Tool 等）产生的漏洞与代码质量指标；
- 对已有威胁模型（Four‑Interface、CharIN EVSE、Microsoft STRIDE）的映射与重构；
- 结合 ISO/IEC 25010:2023 质量特性与多家机构的安全最佳实践，形成多维度评估指标。

**📊 数据集**

论文未使用公开数据集，而是基于对 EVSE 源代码与编译文件的静态分析输出，并利用公开的 CVE 数据库（通过 CVE Binary Tool 检测）。

**📈 对比分析**

由于本文主要聚焦方法构建与框架设计，未给出实验结果或与其他评估方法的性能对比；其贡献在于提出一种可落地、可扩展的评估框架，而非性能指标的量化比较。

**⚠️ 局限性**

局限性包括：
- EVSE 专用的静态分析工具缺乏，导致评估依赖通用工具，覆盖率可能不足；
- 仅能聚合可度量的特征，对缺失或难测量的高层属性依赖人工判断；
- 高层特征的选择与细化仍需在实际项目中验证和迭代；
- 依赖公开 CVE 数据，可能忽略内部/零日漏洞；
- 论文未在真实充电站环境中进行实证评估，缺乏性能与效果验证。

---

## 476. Testing the Binary Rank with Polynomial Query Complexity

**arXiv ID:** 2609.10496 | [PDF](https://arxiv.org/pdf/2609.10496v1)

**作者:** Michal Parnas `[一作]` `[通讯]` (Academic College of Tel Aviv-Yaffo), Michal Parnas (Academic College of Tel Aviv-Yaffo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种适用于0/1矩阵二进制秩（binary rank）的自适应双侧误差属性检测算法，查询复杂度为O(d^3 log(d)/ε^2)，并给出了在二进制秩≤d的假设下，利用额外O(d(n+m))次查询构造近似二进制分解的算法。

**💡 创新点**

突破了此前所有已知的二进制秩测试算法只能达到指数级查询复杂度或极度依赖d的情形，实现了多项式级别（关于d和1/ε）的查询复杂度；同时在核心（core）抽取与候选子空间/图结构的组合方法上具有新颖性。

**🔧 技术方法**

核心抽取基于自适应实秩测试，随后枚举所有可能的核心分解X·Y，构造残差矩阵C并通过子空间W和兼容图G实现行/列扩展的可行性检查；使用随机采样估计候选质量，结合Chernoff界定误差。

**📊 数据集**

无数据集，完全是理论分析与算法设计。

**📈 对比分析**

与先前的Parnas‑Ron‑Shraibman等人提出的指数级查询算法相比，查询量从2^{O(d)}降低到O(d^3 log d / ε^2)，虽然运行时间仍为指数级O(2^{O(d^2)}poly(d,1/ε))，但在查询效率上取得显著提升。

**⚠️ 局限性**

主要局限在于算法的时间复杂度仍为指数级（2^{O(d^2)}），在实际大规模问题上不可行；此外，核心枚举和子空间/图结构构造需要大量二进制向量的枚举，导致实现成本高。

---

## 477. MotionCanvas: Learning Implicit Motion Planning from Composable Kinematic Cues

**arXiv ID:** 2609.10457 | [PDF](https://arxiv.org/pdf/2609.10457v1)

**作者:** Zeyu Ling `[一作]` (Zhejiang University), Linchao Bao `[通讯]` (Tencent)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于共享运动画布的隐式运动规划模型，能够在给定时间、空间、关节位置/角度等多种异构约束的情况下，自动生成连贯的全身动作。

**💡 创新点**

创新点：①将所有约束统一映射到同一“运动画布”上，并通过构造混合约束采样器让单一生成器学习在任意约束组合下生成运动；②采用cue‑preserving flow‑matching 模型，在生成过程中强制保持约束值不变；③在训练中加入几何一致性、边界平滑、rollout监督等多重损失，显著提升运动连贯性与精度。

**🔧 技术方法**

主要技术：流匹配（flow‑matching）+多模态扩散变换器（MMDiT）生成网络；cue‑preserving imputation；组合式约束采样；前向运动学（FK）一致性约束；边界和rollout监督等。

**📊 数据集**

使用的数据集包括 HumanML3D、BABEL、MotionFix、PerMo、BrokenAMASS 等；训练语料约 700 小时的 3D 动作数据。

**📈 对比分析**

与现有方法对比，本文在多项基准上实现了最优或接近最优的性能：在 HumanML3D 上取得最低 FID 与 cue‑error，在混合约束下的 cue‑error 亦为最小；在 BABEL 序列生成、MotionFix 编辑、PerMo 风格/动作切换、BrokenAMASS 修复等任务上均获得最高或最优分数。

**⚠️ 局限性**

局限性：对极度稀疏或不连贯的约束组合仍可能出现过渡不自然；需要较大训练数据与计算资源；模型在训练后无法快速适配全新关节拓扑或非 SMPL 结构；对非常高维约束组合的泛化能力尚待进一步验证。

---

## 478. Advanced Brain Tissue Imaging with Data-Consistent Diffusion Priors in Laminographic X-Ray Nanoimaging

**arXiv ID:** 2609.10456 | [PDF](https://arxiv.org/pdf/2609.10456v1)

**作者:** Wenxuan Fang `[一作]` (École Polytechnique Fédérale de Lausanne), Luis Barba `[通讯]` (Paul Scherrer Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在X‑光层析成像中提出LUCID框架，利用多视角扩散先验与数据一致性迭代实现缺失频率的恢复与脑组织细节的重建

**💡 创新点**

创新点在于将三维扩散生成模型拆解为轴向、矢状面、冠状面三视角的2D扩散先验，并将其嵌入到层析投影域的物理一致性更新中，形成闭环迭代；同时通过可逆域翻译解决模拟训练数据与真实层析测量之间的域差距

**🔧 技术方法**

核心技术包括扩散概率模型（DDPM）作为生成先验、投影域数据一致性正则化、基于前向投影算子𝒜的梯度更新、以及多视角切片处理和可逆量化映射

**📊 数据集**

训练集采用10个高分辨率PXCT脑柱体积的切片（轴向、矢状、冠状），验证集为从剩余PXCT体积合成的层析投影；实验集为从PyXL获取的真实层析投影（约27 nm像素）

**📈 对比分析**

与传统FBP、梯度下降（GD）等方法相比，LUCID在模拟数据上PSNR提高至30.14 dB（比GD提升4.36 dB）、SSIM为0.9902（比GD提升0.0115）；在实验数据中恢复了更多缺失频率，显著降低了轴向拉伸与层间混叠，并在四个频域指标（CSF_cone、ΔE_cone、E_cone）上优于对比方法

**⚠️ 局限性**

主要局限包括：需要预先训练的高质量全角CT体积作为先验，导致跨模态泛化仍受限；多视角2D切片方法虽节省计算，但可能无法完全捕捉3D细微相互依赖；域翻译虽缓解差距，但在极端分辨率或噪声条件下仍可能产生偏差

---

## 479. Building Multilingual Bridges: Data Mixing as the Pillar of Generalization for In-Language Reasoning

**arXiv ID:** 2609.10445 | [PDF](https://arxiv.org/pdf/2609.10445v1)

**作者:** Mehrnaz Mofakhami `[一作]` (Cohere Labs), Julia Kreutzer `[通讯]` (Cohere Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了一个能够在45种语言上用本地语言进行推理的LLM模型（Tiny Aya L2 Reasoner）及其多语言推理数据。

**💡 创新点**

创新点在于：① 通过少量（≤5K/语）翻译后的多语言推理样本与大量英语推理样本以及少量多语言非推理样本混合训练，证明即使极少的多语言推理数据也能让模型在未见语言中以本地语言进行推理；② 引入双模式（推理/非推理）训练与长上下文训练，避免模型在推理时退回英语；③ 通过语言标记（如“Think in language X”）实现推理语言的可控性。

**🔧 技术方法**

使用技术包括：translate‑train（将英语推理数据翻译为多语言）、数据混合策略（英语推理 + 多语言推理 + 多语言非推理）、32K上下文长文本训练、语言标记控制（prepending “Think in the same language as the prompt”）、以及基于FastText/GlotLID的语言识别与重复率检测。

**📊 数据集**

数据集：
- 英语推理数据（OpenThoughts、Dolci‑Think‑SFT‑32B、Open‑Thoughts‑114K 等，约1.7M样本）；
- 约44种语言的翻译推理数据（每语≤5K，约180K样本）；
- 多语言非推理指令数据（约4.9M样本，来自Dolci Instruct、翻译数据、地区指令等）。

**📈 对比分析**

对比方法：同规模英文推理模型、语言强制推理（user prefix / thinking prefix）、以及更大规模的L2推理模型（M‑Thinker 7B、DeepSeek‑v4‑Flash 等）。结果显示：Tiny Aya 在45语言上平均 L2 推理率≈93%，任务准确率仅比英文模型低≤2–3%，在 PolyMath 任务略逊，但整体优于更大模型；在低资源语言上保持高推理率且推理长度更短，显示出更好的效率与泛化。

**⚠️ 局限性**

局限性：
- 多语言推理训练依赖机器翻译，可能导致推理风格、文化框架和表达方式与母语者不一致；
- 评估主要使用自动指标，未进行人类评估；
- 模型对提示中显式的“Think in language X”高度敏感，缺乏在无明确提示或混语输入下的鲁棒性。

---

## 480. Mobility Information Capacity in the Sky: A Gaussian Channel Perspective

**arXiv ID:** 2609.10436 | [PDF](https://arxiv.org/pdf/2609.10436v1)

**作者:** Weijie Yuan `[一作]` (Southern University of Science and Technology), Pingzhi Fan `[通讯]` (Southwest Jiaotong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了低空无线网络的运动信息容量（MIC），将机动输入与轨迹观测之间的可区分度作为容量度量。

**💡 创新点**

创新点在于将运动系统建模为有限时程的高斯通道，推导出对数行列式容量公式，并通过白化与水填充得到运动模态与资源分配，最终给出类似Shannon的长期信息率表达式。

**🔧 技术方法**

使用线性高斯动力学模型、信息理论中的互信息、矩阵白化、奇异值分解及KKT条件实现水填充优化，计算运动模态与容量。

**📊 数据集**

论文未采用公开数据集，而是通过理论模型（如单自由度系统）和数值仿真验证结果。

**📈 对比分析**

通过比较全轨迹观测与终端观测的容量率，展示全轨迹观测可保持正的长期信息率，而终端观测随时程增长趋于零；水填充算法在弱模式上不分配资源。

**⚠️ 局限性**

局限性包括未考虑机动的物理可行性约束（如操纵限制、碰撞、飞行区域约束）、仅对非自适应外生机动有效，且未扩展到因果或多机动交互的情形。

---

## 481. Towards Scalable and Cost-Efficient Vulnerability Detection: A Study on Automatic Query Generation

**arXiv ID:** 2609.10412 | [PDF](https://arxiv.org/pdf/2609.10412v1)

**作者:** Ivana Clairine Irsan `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种利用大型语言模型自动生成 CodeQL 查询的框架，以提升 Java 项目的漏洞检测覆盖率。

**💡 创新点**

创新点在于将 LLM 与结构化提示、语法校正相结合，实现从 NVD 漏洞描述自动合成可执行的 CodeQL 查询，并在成本与效能上优于传统手工编写与直接 LLM 扫描。

**🔧 技术方法**

采用多模型 LLM（如 Kimi K2.5、GPT 5.2 Codex、Claude Sonnet 4.5 等）结合 Gemini 3 Flash Preview 进行语法校正，使用提示工程、递归提示和循环迭代的方式生成查询，并通过 CodeQL 编译器验证。

**📊 数据集**

使用 MoreFixes 公开的 Java CVE 代码变更集以及 NVD 的漏洞记录，涵盖 MITRE Top 25 10 大 CWE，构建了 112 个测试案例。

**📈 对比分析**

通过与官方 CodeQL 基线和 PDBERT 的文件级别检测对比，评估检测率、FDR 与 F1 分数；Kimi K2.5 生成 96 条查询，检测率提升 263% 并获得 24.04% 的 AvgF1，表现优于基线并在成本上更具优势。

**⚠️ 局限性**

局限性包括对较大或跨文件漏洞的生成效果有限、对“死代码”产生高误报、对不同语言迁移尚未验证，以及依赖昂贵的高端 LLM 模型导致的部署成本。

---

## 482. Can Foundation Models Moderate Online Content? Evaluating Instruction- vs. Example-Driven Policy Operationalization

**arXiv ID:** 2609.10410 | [PDF](https://arxiv.org/pdf/2609.10410v1)

**作者:** Ayan Majumdar `[一作]` (MPI-SWS), Abhisek Dash `[通讯]` (MPI-SWS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了两种基于视觉语言模型的内容审核方法（指令驱动与案例驱动），并构建了基于 Bluesky 平台的 4,000 条多模态帖子审核基准。

**💡 创新点**

首次系统比较了指令驱动与案例驱动两种范式在平台政策执行上的有效性，发现两者在 F1 分数上相当，但指令驱动在速度和成本上更具优势。

**🔧 技术方法**

使用开源视觉语言模型（如 LLaVA、Flamingo 等）以及专门的 AI 安全模型（如 OpenAI’s Moderation API），并通过不同层级的政策文本或历史审核案例进行提示。

**📊 数据集**

基准数据集为 Bluesky 社交媒体平台上 4,000 条手工标注的多模态帖子，涵盖自然流量、已标注违规、潜在漏标违规和低风险帖子四个子集。

**📈 对比分析**

对比实验显示：指令驱动模型在所有子集上都显著优于 Bluesky 的现行审核系统（F1 从 0.22 提升至 0.60），而案例驱动（尤其是典型示例）也能达到相近性能，但需处理多张图像导致推理速度降低 6 倍、成本升高 3–32 倍。

**⚠️ 局限性**

主要限制包括：未覆盖视频/音频等其他模态；缺乏专家级审核标签；未考虑上下文会话、链接内容和实时大规模部署；以及对政策变更和潜在偏见的系统性评估不足。

---

## 483. Show-Harness: Just a VLM Agent Can Play Robots

**arXiv ID:** 2609.10522 | [PDF](https://arxiv.org/pdf/2609.10522v1)

**作者:** Yanzhe Chen `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Show-Harness 这套模型无关的“Embodied Harness”，通过紧凑的语义动作接口让基础视觉‑语言模型（VLM）能够逐步推理并直接控制机器人；并配套 GUMI GUI 界面，实现无硬件遥控的演示收集。

**💡 创新点**

创新点在于将 VLM 与机器人连接的动作空间既保持语义可解释、跨机器人实现，又细粒度足以实现直接物理控制，允许（1）在零调优下使用前沿 VLM 进行零射击控制；（2）用极少 GPU 小量微调即可让小型开源 VLM 在同一接口下获得强大泛化能力；以及通过 GUMI 统一人机协同演示收集。

**🔧 技术方法**

核心技术包括：1) Perceive–Reason–Act 循环；2) 多视角引导、关节感知、子任务规划、在位规划、动作分块与自适应步长等插件；3) 语义动作解释器在每个机器人实现上做确定性映射；4) LoRA 微调和视觉‑语言预训练的无缝对接；5) GUI 交互式演示记录。

**📊 数据集**

使用了真实机器人数据集：在 Franka 与 AgileX 双臂上收集 164 条真实演示（7.8K 步），覆盖 10 种不同物体（方块、香蕉、网球、泰迪熊、棋子）和两种目标容器；以及 230 条模拟演示（13.5K 步），跨 ManiSkill 与 RoboLab 两个仿真环境。

**📈 对比分析**

与 VLA、VLA‑centric、Code‑as‑policy 等三大基线进行比较；Show‑Harness 在任务、环境、外观、模态与跨机器人迁移上均显著优于基线；零射击模式可直接使用 Gemini‑3.1 Pro 等前沿模型完成任务，微调小型模型（如 Qwen3.5‑2B）在模拟演示下实现了 0‑to‑real 转移并获得更高成功率。

**⚠️ 局限性**

目前仅在单臂/双臂平行爪抓取机器人上验证，缺乏对更复杂身体（人形、灵巧手等）的适配；未加入触觉/力反馈，限制了在更高接触要求任务中的表现。

---

## 484. BrainTaskonomy: Learning How to Pretrain and What to Transfer in fMRI Foundation Models

**arXiv ID:** 2609.10518 | [PDF](https://arxiv.org/pdf/2609.10518v1)

**作者:** Junfeng Xia `[一作]` (Southern University of Science and Technology), Quanying Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

基于学习关系构建fMRI基础模型的预训练与下游适配策略；

**💡 创新点**

通过测量域间学习难度与促进关系，设计域优先和高到低噪声时间步的层次化预训练课程；同时利用任务间的第一阶与高阶迁移关系构建任务族，并在预算约束下使用整数规划选取源任务和迁移路径。

**🔧 技术方法**

轻量级Brain‑DiT代理、时间步分段课程、任务族构建、预算感知整数规划（BIP）、高阶PCA‑岭回归读取器等技术。

**📊 数据集**

十个fMRI域（HCP Rest/Task/Movie、CHCP Rest、ABCD Rest、NKI Rest、ABIDE、ADHD、ADNI、CineBrain）以及十五个下游任务（年龄回归、性别分类等）。

**📈 对比分析**

在Omni‑fMRI基准上与BrainLM、Brain‑JEPA、BrainMass等现有基础模型对比，Priority+high‑to‑low时序课程的Brain‑DiT在v‑NMSE、PSD‑NMSE、FC‑MSE以及六项内域与部分外域任务上均优于随机/均匀混合策略，提升幅度分别为6.5%、16.3%和10.5%；BIP策略在高阶迁移下相较于随机/容量匹配基线显著提升。

**⚠️ 局限性**

实验范围受限于当前域与任务集合，未系统评估跨模型或更大域空间的泛化；预算约束下的策略验证仅为探索性，尚未充分证明对OOS鲁棒性的普适提升。

---

## 485. PASCAL: A Phase-Aware Shared-Cache Model for Parallel Scans

**arXiv ID:** 2609.10515 | [PDF](https://arxiv.org/pdf/2609.10515v1)

**作者:** Zhongchun Zhou `[一作]` (Hong Kong University of Science and Technology), Songtao Mao `[通讯]` (Johns Hopkins University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 PASCAL，一种基于阶段感知的共享缓存模型，能够精确计算并预测 AI 加速器中并行扫描访问模式下的缓存未命中率。

**💡 创新点**

创新点在于：①推导了针对并行扫描的 TTL 缓存闭式未命中计数公式；②给出了 LRU 和任意分页策略的未命中上界；③通过进度偏差、占用率、预取深度和计算工作量等动态特征，构建了无需要求执行轨迹的有限时域预测管线；④将静态几何与动态进度关联，显著降低了对采样或模拟的依赖。

**🔧 技术方法**

技术方法包括：工作集理论、堆叠距离 (stack‑distance) 与再利用距离 (reuse‑distance) 的符号推导；TTL 与 LRU 的闭式公式与边界；进度偏差量化与水填充 (water‑filling) 计算；基于累积流量曲线的插值与归一化；以及对执行类的归档插值（anchor harmonic interpolation）。

**📊 数据集**

实验数据集主要是 NVIDIA GB10 GPU 上的自定义微基准，采用 16 KiB 数据块、1280 行缓存容量、不同占用率、预取深度和计算工作量，共计 60 组未见配置（420 次运行）用于测试；训练阶段使用 84 个配置、1096 条标签。

**📈 对比分析**

与 TileSight、SDCM 等基线相比，PASCAL 在 60 组测试配置上的平均绝对误差（MAPE）为 13.84%，明显低于 TileSight 的 44.79% 和 SDCM 的 54.16%。在更长时域压力测试中，PASCAL 仍保持较低误差，且在所有配置上平均误差仅为 0.76pp，显著优于对手。

**⚠️ 局限性**

局限性包括：①模型仅在固定配置、块大小与缓存容量已知的前提下校准，跨平台迁移需要重新校准；②对动态进度偏差的预测仍基于经验插值，无法完全捕捉极端进度分离情形；③仅针对并行扫描模式，对更一般的访问模式的适用性未作评估；④缺乏严格的统计置信区间或覆盖保证，误差控制依赖训练集覆盖度。

---

## 486. DUET-DINO: Simultaneous Cross-View World Modeling for Latent Planning in Robot Manipulation

**arXiv ID:** 2609.10506 | [PDF](https://arxiv.org/pdf/2609.10506v1)

**作者:** Nisarga Nilavadi `[一作]` (University of Technology Nuremberg), Wolfram Burgard `[通讯]` (University of Technology Nuremberg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了DUET‑DINO，一种通过双视角交叉注意力实现的潜在世界模型，用于机器人操纵任务中的7-DoF动作规划；

**💡 创新点**

创新点在于将侧视相机与腕部摄像头的潜在表示进行跨视角交叉条件化，使每个视角的预测器能利用互补信息，从而实现全7-DoF的精细运动规划；

**🔧 技术方法**

采用冻结的DINOv3视觉编码器、跨视角交叉注意力块、独立的预测头以及CEM优化进行零射规划；

**📊 数据集**

在DROID与RoboArena两大大型机器人操作数据集上训练和评估；

**📈 对比分析**

与单视角、独立双视角以及V‑JEP A 2‑AC等基线相比，DUET‑DINO在空间到达、倾斜到达和抓取-举起任务中分别达到92%、72.5%和60%成功率，显著优于基线；

**⚠️ 局限性**

局限在于CEM规划计算量大导致实时性受限，且对相机位姿变动仍有一定敏感性，未来需结合更高效的动作建议方法。

---

## 487. Cross-Model Agreement as a Deployment-Time Reliability Signal for Automatic Polyp Segmentation

**arXiv ID:** 2609.10495 | [PDF](https://arxiv.org/pdf/2609.10495v1)

**作者:** Siddharth Gupta `[一作]` (Indian Institute of Technology Roorkee), Jitin Singla `[通讯]` (Indian Institute of Technology Roorkee)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种 Referee‑Based Quality Estimation (RBQE) 框架，通过将主分割模型与一个独立训练的裁判模型在同一张图像上的预测结果进行一致性评估，以估计实时结肠镜分割的可靠性。

**💡 创新点**

创新点在于系统性剖析裁判模型的独立性与架构多样性对可靠性信号的贡献，并证明仅凭模型独立性即可获得有意义的质量估计，而架构多样性进一步提升判别力；同时在无真实标注条件下提供了可解释且低成本的质量估计方法。

**🔧 技术方法**

主要技术包括：多模型一致性度量（Agreement Dice、IoU、Area Ratio、Boundary Agreement、Centroid Distance）、裁判模型的独立训练、交叉架构比较（SegFormer‑B0、UNet++）以及与单模型不确定性（TTA）和几何基准（形态学特征）的对比。

**📊 数据集**

使用了四个公开结肠镜数据集（CVC‑ClinicDB、CVC‑ColonDB、ETIS‑Larib PolypDB、CVC‑300）共1,223张图像作为外部评测基准，裁判模型均在Kvasir‑SEG上训练。

**📈 对比分析**

与形态学基准（0.927）和TTA（0.905）相比，RBQE 在标准基准上的ROC‑AUC达到0.960，限制子集（排除空掩码）上仍保持0.876，显著提升；此外，RBQE 仅需一次额外裁判模型前向传播，计算开销低于TTA。

**⚠️ 局限性**

局限性包括：仅验证了二分类多边形分割，需评估多类别/多模态任务；依赖裁判模型的可用性；在裁判与主模型共享系统性偏差时可能产生高一致但错误的结果；以及未提供像素级不确定性映射。

---

## 488. IBIB: A Protocol for Measuring Enterprise AI Systems by Serving Route, Not Model Identifier

**arXiv ID:** 2609.10494 | [PDF](https://arxiv.org/pdf/2609.10494v1)

**作者:** Blake Stenstrom `[一作]` (Iterate.ai), Brian Sathianathan `[通讯]` (Iterate.ai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并实现了一套企业级AI评估协议，核心包括在评估前对服务路由进行金属绑定预检、可靠性包含的评分规则以及score‑blind的判定过程，并在该协议下对11个部署系统在128个锁定任务（987个断言）上的能力进行测评。

**💡 创新点**

创新点主要有：①将服务路由纳入评估对象，解决了模型标识与实际部署系统之间的测量误差；②提出可靠性包含评分，既记录失败，又不将不支持的功能计入分数；③引入score‑blind判定，确保恢复决策不依赖正确性信息；④定义饱和判定与分层评估机制，客观描述不同任务集对系统区分度的影响。

**🔧 技术方法**

技术手段包括：金属绑定预检算法（Algorithm <ref>）、判定表与决策表的实现、任务级别断言与加权平均分数计算、Bootstrap分层采样产生区间置信区间、以及对结果的可追溯性哈希与记录。

**📊 数据集**

数据集为128个锁定任务（共987个确定性断言），涵盖七个suite：S1 单表电子表格、S2 多表电子表格、S3 可选文本文档、S4 仅图像文档、S5 图表/图形、S6 状态化工具调用、S7 受治理数据库。所有任务为合成、受控且已签名。

**📈 对比分析**

比较方法：采用等权重的suite平均分（-7 Full）作为主分数，并在每对系统间通过Bootstrap重抽样得到95%区间；按区间不包含0的原则划分resolution groups；最终得到分层结果。性能方面，GPT‑5.6 Sol得分88.34为最高，GLM‑5.2+GLM‑5V‑Turbo得分41.10为最低；top9系统在大部分suite上得分>90；四套suite（S1,S3,S4,S6）饱和，S5不具判别性，S2与S7保留区分度。

**⚠️ 局限性**

局限性：①每个配置仅跑一次，缺乏跑间方差估计；②四套suite饱和导致分辨率受限；③使用私有合成任务，缺乏公开可复现性；④未对底层引擎/硬件差异进行归因；⑤无人工基准；⑥不公开完整任务与答案，外部验证受限。

---

## 489. Nonmaximal sums of maximally monotone operators under Rockafellar's constraint qualification

**arXiv ID:** 2609.10487 | [PDF](https://arxiv.org/pdf/2609.10487v1)

**作者:** Weifeng Yang `[一作]` `[通讯]`, Weifeng Yang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

在非反射 Banach 空间 c₀ 与标准 ℓ¹ 上构造了满足内部域条件的两个最大单调算子，但它们的点态和不再是最大单调的，从而给出 Rockafellar 和氏和式猜想的反例。

**💡 创新点**

提出了一个通用构造定理，利用单调极点（monotone polar）与正秩一扰动的关系，能够在满足特定几何条件下自动生成此类反例；该定理不仅适用于 c₀，也可通过有界线性满射转移到 ℓ¹，扩展了已知反例的范围。

**🔧 技术方法**

主要技术包括：单调极点的显式计算、凸代表函数与 Fitzpatrick 表示、Lipschitz 曲线构造、块状三角映射、正秩一线性算子、以及有界线性满射的拉回（pullback）法则。

**📊 数据集**

本文没有使用实验数据集，所有结果均为理论证明；主要采用 c₀ 的坐标块结构与 ℓ¹ 的标准基向量构造具体算子。

**📈 对比分析**

由于是理论构造，无需与其他方法进行性能比较；研究重点在于给出反例的存在性与构造过程，而非数值表现。

**⚠️ 局限性**

局限性在于：构造方案依赖于特定的块状三角结构与 Lipschitz 曲线，尚未证明可推广至更一般的非反射 Banach 空间；此外，反例仅展示了最大单调性失效的情形，并未探讨其对优化算法收敛性的实质影响。

---

## 490. ConvMem: Convolutional Memory for Long-Context Reasoning

**arXiv ID:** 2609.10441 | [PDF](https://arxiv.org/pdf/2609.10441v1)

**作者:** Hongming Zhang `[一作]` (Chinese Academy of Sciences), Bo Xu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ConvMem，一种训练‑free 的并行长文本推理框架，利用冻结 LLM 作为语义卷积核，将线性处理改为层次化对数树形结构。

**💡 创新点**

创新点在于将卷积神经网络的层级卷积、可配置步幅、跳跃连接和多核卷积等概念迁移到文本推理，消除 RL 训练需求并实现大规模并行化。

**🔧 技术方法**

使用技术包括：冻结 LLM 语义卷积核、可配置步幅（重叠窗口）、跳跃连接（保留关键原文）、多核卷积（查询拆分为多通道）、层级并行推理。

**📊 数据集**

使用的数据集为 RULER‑HotpotQA（in‑distribution）和 RULER‑2WikiMultiHopQA（out‑of‑distribution）。

**📈 对比分析**

与标准 LLM、检索/记忆式训练‑free 方法以及 RL 训练的 MemAgent 等进行对比，ConvMem 在训练‑free 基线中取得最高 F1/子‑EM 分数，并在 OOD 数据上优于 RL 方法，显示更好的鲁棒性和泛化。

**⚠️ 局限性**

局限性包括：总体算力/能耗高于单线扫描，且对查询拆分的质量高度依赖，拆分错误会导致后续推理失效。

---

## 491. Glyph: A Multi-Strategy Agentic System for Column Description and Sensitivity-Ontology Tagging of Enterprise Data Catalogs

**arXiv ID:** 2609.10430 | [PDF](https://arxiv.org/pdf/2609.10430v1)

**作者:** Kostia Kudriavtsev `[一作]` (Apple), Sha Sundaram `[通讯]` (Apple)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

Glyph 通过两台 LLM 代理（描述生成器和标签器）自动为企业数据湖中的每一列生成描述并分配多标签安全分类，实现可审计、可部署的列级数据目录服务。

**💡 创新点**

创新点包括：1）将列描述生成与安全标签分配拆分为独立的 LLM 代理，并通过共享状态与本体耦合；2）结合检索增强生成、三种并行标签策略（正则、描述、元数据）与 Reciprocal Rank Fusion，实现跨信号的稳健融合；3）仅使用元数据与代码（不读取敏感单元格）完成标签；4）每个标签附带来源、置信度与推理，实现可追溯、可审计。

**🔧 技术方法**

技术栈包括：LangGraph 结构化代理、主动检索增强生成 (RAG)、MiniLM 6 层对比学习细调的元数据编码器、正则表达式规则、密集检索 + ANN、LLM-as-Judge 自校验、Reciprocal Rank Fusion、Pydantic 状态验证、REST 微服务、S3 存储的向量索引与版本化。

**📊 数据集**

数据集：企业生产目录 3.4M 列、98k 表（涵盖 4 个业务线和 3 种存储后端）；训练集包含 70k+ 已标注列-标签对；评测集由 4 个业务线的表级数据集构成，表与训练集严格分离。

**📈 对比分析**

评估方法：对比单策略（regex、描述、元数据）与融合结果，使用 F2（召回优先）衡量多标签质量；与传统 CTA 系统（Sherlock、Sato、Doduo）及商业扫描器对照，融合后整体 F2 达到 0.890，元数据单一策略 0.880；在生产中接受率从 63% 提升至 99%，检索指标 NDCG@10 由 0.55 提升至 0.92。

**⚠️ 局限性**

局限性：1）依赖大量人工标注的训练数据，缺乏公开基准泛化验证；2）无法使用单元格数据导致对细粒度语义的把握有限；3）未实现跨列联合推理，缺乏协同效应；4）p50 延迟约 58 秒，批处理时可行，但实时性受限。

---

## 492. Frequency-Conditioned Flow Matching for Vision-Language-Action Models

**arXiv ID:** 2609.10405 | [PDF](https://arxiv.org/pdf/2609.10405v1)

**作者:** Haochen Niu `[一作]` (AGIBOT), Wang Chuang `[通讯]` (AGIBOT)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过在机器人视觉语言动作模型中引入频域条件化的流匹配框架FreqFM，对机器人动作的频谱进行显式建模并提升动作生成质量。

**💡 创新点**

创新点在于将频率作为条件维度贯穿整个流匹配流程，包括谱匹配源分布、PSD归一化自适应目标以及基于频率预算的分类器无指导（CFG）限制。

**🔧 技术方法**

采用离散余弦变换（DCT）将时域轨迹映射到频域，使用流匹配（Flow Matching）、多任务自适应权重、以及频率参考运输预算的CFG投影。

**📊 数据集**

在LIBERO、LIBERO-Plus、VLA-Arena模拟基准和六个真实机器人任务（AgiBot Expedition A2）上进行评估。

**📈 对比分析**

与同一架构下的时域流匹配基线相比，FreqFM 在LIBERO-Plus、VLA-Arena 以及六个真实任务中均提升 1–9.3 个点（如平均成功率提升 9.3%），并在低频高能量任务中显著降低轨迹抖动。

**⚠️ 局限性**

限制在于频谱统计仅在训练集上一次性估计，难以适应与训练分布不同的动作统计；以及需手动指定平滑动作维度，可能影响跨平台迁移。

---

## 493. A positive resolution of the gap-entropy conjecture

**arXiv ID:** 2609.10529 | [PDF](https://arxiv.org/pdf/2609.10529v1)

**作者:** P. M. Aronow `[一作]`, Patrick Lopatto `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

证明了固定置信度最佳臂识别的间隙熵猜想，针对独立单位方差的高斯臂，均值在[0,1]之间，并且存在唯一的最优臂。

**💡 创新点**

提出了间隙熵的概念，并证明了在所有δ-correct算法中，给定实例的期望样本数量的上下界与间隙熵相关，且可以通过一个算法同时达到所有实例的基准。

**🔧 技术方法**

使用了高斯模型和1-sub-Gaussian奖励分布的算法，结合了熵的计算和样本复杂度的分析。

**📊 数据集**

使用了高斯分布的实例数据集，特别是均值在[0,1]之间的高斯臂。

**📈 对比分析**

与现有算法（如Exponential-Gap Elimination和lil'UCB）进行了比较，证明了所提出算法在样本复杂度上具有更优的界限，且在每个实例上都能达到接近最优的样本数量。

**⚠️ 局限性**

限制在于算法的性能依赖于对间隙的准确估计，且在某些情况下可能需要更多的样本来适应未知的间隙配置。

---

## 494. Subexponential Approximation of the Permanent in Deterministic Polynomial Time

**arXiv ID:** 2609.10516 | [PDF](https://arxiv.org/pdf/2609.10516v1)

**作者:** Sergei Kudria `[一作]` (Chinese University of Hong Kong Shenzhen), Mahbod Majid `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了第一种在多项式时间内对任意非负有理矩阵的行列式进行子指数因子近似的确定性算法。

**💡 创新点**

创新点在于将凸优化与匹配分区函数结合，构造可控的上界，并通过“匹配未匹配节点的期望数”以及相关性衰减来分析误差，从而实现比以往任何方法都更好的子指数近似。

**🔧 技术方法**

核心技术包括：
- 对匹配分区函数的凸化与约束化（log‑sum‑exp 和分配项），
- 通过预算约束（每个顶点的权重和≤Λ）控制误差；
- 采用投影子梯度平均法求解凸目标；
- 递归删除与相关性衰减法求解匹配概率，从而实现高效的匹配计数。

**📊 数据集**

本工作为理论研究，没有使用具体数据集；所有实验与验证均在理论分析与数学证明层面完成。

**📈 对比分析**

与以往的确定性多项式时间方法（如 eⁿ、√2ⁿ 等）相比，误差对数宽度为 O(n(log log n)² / log n)，对应的近似因子为 exp(o(n))，在任何固定指数因子 cⁿ (c>1) 之上显著改进；与随机化 FPRAS（Jerrum‑Sinclair‑Vigoda）相比，虽然仍未实现多项式时间的相对误差，但已迈出了将随机化依赖降至子指数的关键一步。

**⚠️ 局限性**

局限性：
- 仍然是子指数误差，未能达到多项式时间的相对误差 FPTAS；
- 运行时间的指数项包含 √λ·log²λ，虽然对固定 λ 可多项式，但在 λ 较大时可能不可接受；
- 证明主要针对二部图结构，扩展到一般图需进一步研究。

---

## 495. Field Converter: Geometry-Initialized Temporal Residual Refinement for World-Grounded Player Pose Estimation from Soccer Broadcasts

**arXiv ID:** 2609.10498 | [PDF](https://arxiv.org/pdf/2609.10498v1)

**作者:** Simon Khan `[一作]` (Arts et Métiers ParisTech), Sébastien Laporte `[通讯]` (Arts et Métiers ParisTech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了Field Converter框架，利用摄像机与球场几何信息对球员根部位置进行初始化，并在此基础上通过时序残差网络进行校正，实现世界坐标系下的3D球员姿态估计。

**💡 创新点**

创新点在于将全局位置回归拆解为几何初始化与残差校正两步，显著减少了全局定位误差，并证明了残差学习优于直接回归；同时证明了时序上下文对全局定位的关键作用。

**🔧 技术方法**

采用了相机与场地平面交点的几何计算、低阶姿态回归（MLP）、时序卷积网络（TCN）与Transformer等技术，融合3D姿态、2D姿态、bbox信息、相机参数及地面交点等多模态特征。

**📊 数据集**

在FIFA Skeletal Tracking Light 2026数据集（包含89段经过标定的单目足球广播视频）上进行训练与评估。

**📈 对比分析**

与几何初始化、直接全局回归、帧级MLP等方法对比，残差TCN与Transformer将根部误差从49cm降至约10cm，世界MPJPE降至13cm，表明方法在全球定位与姿态一致性上均优于现有技术。

**⚠️ 局限性**

主要局限是对地面接触的假设导致在跳跃、头球等离地动作中定位误差显著上升；此外依赖于上游检测、姿态估计与相机标定的精度，误差可能会被放大。

---

## 496. HAPS-RIS or HAPS-Relay: Which Outperforms Under Impairments with NOMA in 6G NTN?

**arXiv ID:** 2609.10468 | [PDF](https://arxiv.org/pdf/2609.10468v1)

**作者:** Bilal Karaman `[一作]` (Manisa Celal Bayar University), Halim Yanikomeroglu `[通讯]` (Carleton University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过蒙特卡罗仿真研究了高空平台站（HAPS）使用可重构智能表面（RIS）和中继站（RS）两种架构，在非正交多址（NOMA）和正交多址（OMA）方案下，考虑硬件失真（HWI）与不完美信道状态信息（CSI）的实际系统性能；

**💡 创新点**

创新点在于：①首次系统性对比HAPS‑RIS与HAPS‑RS在现实失真与CSI误差条件下的谱/能效；②提出RIS元件分配与用户空间分布对NOMA性能的优化策略；③揭示在LOS主导环境中，用户距离增大可显著提升NOMA收益。

**🔧 技术方法**

使用的技术包括：Rician多径信道模型、Friis路径损耗、超分辨率RIS相位调控、功率分配与功率域NOMA、硬件失真建模、CSI误差建模以及能效分析框架。

**📊 数据集**

未使用真实数据集，而是基于参数化的仿真模型（L=2用户、2 GHz载波、NIS等）进行10⁵次Monte Carlo模拟。

**📈 对比分析**

比较方法：在相同总发射功率与相同用户设置下，分别计算HAPS‑RS与HAPS‑RIS在Oma/NOMA下的总速率与能效；结果显示，在非理想条件下HAPS‑RIS在总速率上明显优于HAPS‑RS，且能效曲线在中等到高功率区间实现峰值，表明RIS在能耗低、信号噪声不放大的优势。

**⚠️ 局限性**

局限性包括：①仅考虑单小区两用户NOMA；②RIS元件分配仅通过仿真推导，缺乏闭式优化或实时自适应算法；③硬件失真模型为高斯近似，实际失真特性可能更复杂；④未讨论RIS硬件功耗与控制开销的动态管理。

---

## 497. Do speech foundation models really learn words?

**arXiv ID:** 2609.10434 | [PDF](https://arxiv.org/pdf/2609.10434v1)

**作者:** Robin Huo `[一作]` (University of Toronto), Ewan Dunbar `[通讯]` (University of Toronto)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了 HuBERT 与 wav2vec 2.0 预训练模型在后期层对词级信息的编码能力，利用线性残差化去除音位（phoneme/diphone/triphone）信息后，检验词识别与无监督词发现的性能。

**💡 创新点**

创新点在于通过残差化方法分离词信息与音位信息，证明后期层仍能独立编码词识别信息，并且该残差化能显著提升无监督词发现的 NED/F1/R 指标。

**🔧 技术方法**

采用线性岭回归进行残差化、softmax 线性探测器对词识别进行评估，并结合无监督词分割算法与 K‑means 聚类来评估词发现效果。

**📊 数据集**

使用 LibriSpeech dev‑clean 语音数据及其对应的 phoneme 与 word 对齐标签进行实验。

**📈 对比分析**

与原始表征相比，残差化后的模型在词识别准确率上保持≈90%（后期层）并在词发现任务中 NED 降低、token F1 与 R 值提升，验证残差化策略有效。

**⚠️ 局限性**

主要局限是需要精确的 phoneme 对齐信息，残差化仍未能完全消除音位信息，并且排除完整词帧的做法可能导致估计偏差。

---

## 498. Multi-Agent Reinforcement Learning for Autonomous UAV Exploration in Wildfire Response

**arXiv ID:** 2609.10433 | [PDF](https://arxiv.org/pdf/2609.10433v1)

**作者:** Caden Chandra `[一作]`, Jerry Ng `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在模拟火灾环境中训练多智能体无人机，通过深度强化学习实现火灾边界跟踪与自适应定位。

**💡 创新点**

将奖励分解为接近、探索、能量等多维度，并结合环境复杂度与层级课程学习，形成针对不同火场结构的统一策略。

**🔧 技术方法**

采用多智能体 Actor‑Critic（TD3）框架、Gaussian 噪声探索、Curriculum 学习、奖励归一化、Replay Buffer、APE2 候选评估等技术。

**📊 数据集**

使用六种自定义火场模拟场景（静态、扩散、线性、随机、墙体、中心大火），并未使用公开数据集。

**📈 对比分析**

通过与完整奖励模型的消融实验比较，各场景中奖励分量的缺失导致成功率与覆盖率下降；最终基线在所有指标上保持最优；训练后平均奖励≈250k，成功率100%，平均距离≈1 单位。

**⚠️ 局限性**

仅在仿真环境下验证，未考虑风、地形、传感器噪声和通信延迟等现实因素；实验仅单次运行，缺乏多次复现；使用统一策略可能在特定火场中性能不佳。

---

## 499. TrajMark: Ownership Attribution and Segment-Level Tamper Localization for Coding-Agent Trajectories

**arXiv ID:** 2609.10416 | [PDF](https://arxiv.org/pdf/2609.10416v1)

**作者:** Bokang Zeng `[一作]` (University of New South Wales), Jiaojiao Jiang `[通讯]` (University of New South Wales)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可视化的两层轨迹水印框架TrajMark，能够在编码代理生成的可见轨迹上实现批量级所有权归属与细粒度篡改定位；

**💡 创新点**

创新点在于将鲁棒的所有权证明与脆弱的局部完整性检测分离为两层：稀疏的文件名索引线性方程作为所有权证据，Q12子类型的普通、组和终端封印作为完整性承诺；

**🔧 技术方法**

技术包括基于HMAC的键控文件名选择、GF(2)^6线性方程嵌入、Q12封印的序列化与哈希验证、以及公正的可视化轨迹重放与批量解码；

**📊 数据集**

使用SWE‑bench Python、SWE‑PolyBench Java 与 JavaScript 三大仓库，结合 SWE‑agent、OpenHands 与 OpenDev 三个代理框架，以及 DeepSeek V4 Flash、GPT‑5 mini 与 MiniMax M3 三种大语言模型；

**📈 对比分析**

与 ActHook、AgentMark 等前沿行为水印方法对比，在 27 组完整水印批处理中，所有权恢复率 100%，随机 20% 改动下 96%+ 的恢复率，单点篡改检测率 95.5%–100%，整体结构开销约 7 次读操作/轨迹，Pass@1 仅提升 0.6%；

**⚠️ 局限性**

局限性包括：需要键值密钥且不提供非否认或时效性保证；受限于文件名碰撞和探测空间；对可见轨迹的公开规范依赖较高，无法适用于隐藏元数据或完整日志保留的场景；

---

## 500. Fortunate Recall: Ontology-Driven Memory Lifecycle Management for Persistent Coherence in LLMs

**arXiv ID:** 2609.10413 | [PDF](https://arxiv.org/pdf/2609.10413v1)

**作者:** Ansuman Mullick `[一作]` (Bilkent University), Eray Tüzün `[通讯]` (Bilkent University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Fortunate Recall 的生命周期管理层，对个人事实进行行为类型分类并执行确定性生命周期策略，实现更精准的长期记忆存取。

**💡 创新点**

创新点包括：①基于行为域的 10+1 语义本体，②针对每类的差异化衰减、衍生、事件时效与检索路由策略，③将生命周期元数据与 LLM 提取相分离，保持低延迟且可解释的决策。

**🔧 技术方法**

技术实现：LLM（gpt‑4.1‑mini）在归档时提取边缘关系、槽键、生命周期状态与事件时间；随后使用纯数学的生命周期层对候选事实做闭式评分；检索过程中加入少量 LLM 调度仅做候选精简，整体延迟约 47 µs。

**📊 数据集**

使用数据集：自建 LifecycleBench（516 题、40 人格、9 攻击向量）、LongMemEval‑S（500 题）以及 BEAM（2,000 题）进行评估；同时在 Kimi K2.5 生成器上进行跨实现验证。

**📈 对比分析**

与 Mem0、Memory‑R1、MemoryOS、A‑MEM 等现有系统在七种配置下对比：FR‑Bank 在 LifecycleBench 上 76.9% pass（比 Mem0 提升 16pp），LongMemEval‑S 上 75.2% pass，confabulation 下降 45.1%→22.4%（答题率提升 31.2%→18.6%）。在 BEAM 上也获得 46.8% 正确率，显著优于 Mem0 的 32.9%。

**⚠️ 局限性**

局限性：①评估完全基于 LLM 判断，缺乏人工标签；②Benchmark 与方法共设计可能导致偏倚；③对比仅使用默认配置，未进行最优调参；④未在多框架或多代理环境中验证；⑤系统在需始终答复时的强制答复率与误答率提升仍是待解决问题。

---

## 501. Forgetting Only What Matters: Layer-Selective Unlearning toward Robust LLMs

**arXiv ID:** 2609.10439 | [PDF](https://arxiv.org/pdf/2609.10439v1)

**作者:** Ravi Ranjan `[一作]` (Florida International University), Agoritsa Polyzou `[通讯]` (Florida International University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 FOM-UL，一种基于层级选择的 LLM 遗忘框架，仅更新对遗忘目标影响大、对保留目标敏感度低的 Transformer 层，以实现高效、量化鲁棒的忘记。

**💡 创新点**

创新点在于：① 用忘记‑保留比值构建层级显著性评分；② 采用迭代扩展层级的可更新集合；③ 结合遗忘、匹配、保留三重损失；④ 通过层级集中更新提升 4‑bit/8‑bit 量化下的遗忘稳健性。

**🔧 技术方法**

技术手段包括：Transformer 层贡献分析、梯度显著性评分、梯度上升/下降、匹配损失、量化分析、SURE 协议、NPO、GA、KLD 等传统遗忘方法。

**📊 数据集**

使用数据集：TOFU、KnowUnDo、MUSE（BOOKS、NEWS）以及多种 LLM（Llama‑2 7B、Llama‑3.2 1B、GPT‑2、Gemma‑3 1B）。

**📈 对比分析**

与 vanilla、GA_GDR、NPO_GDR、SURE+NPO、ReLearn、LUNAR 等基线比较，FOM-UL 在 M1/M2（残留记忆）、隐私泄露、保留集效能等指标上均表现更优或相近，并在 8/4‑bit 量化与对抗提问下保持更高的鲁棒性。

**⚠️ 局限性**

局限性：依赖层级归因信号的可靠性；未给出正式消除保证；在高度冗余或极端对抗/分布偏移情形下可能需要扩展更新层或多轮训练，导致计算成本和潜在的效能折衷。

---

## 502. Learning with Covariance Matrices: Principal Component Analysis Meets Learning with Graphs

**arXiv ID:** 2609.10490 | [PDF](https://arxiv.org/pdf/2609.10490v1)

**作者:** Saurabh Sihag `[一作]` (University at Albany, SUNY), Alejandro Ribeiro `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出将协方差矩阵视为图结构，利用图信号处理中的图卷积实现PCA的等价表达，并进一步构建可扩展的协方差网络（VNN），解决传统PCA在可重复性、计算复杂度及多尺度适应性方面的不足；

**💡 创新点**

创新点在于：1）将PCA转换为无需特征分解的多项式图滤波实现；2）引入VNN实现非线性映射和多层结构；3）提供理论稳定性与可迁移性证明；4）设计稀疏VNN、时空VNN、公平VNN及鲁棒VNN等扩展；

**🔧 技术方法**

技术包括：协方差矩阵图建模、图傅里叶变换、谱滤波、多项式图滤波、非线性激活、VNN多层架构、稀疏阈值、在线更新、图仿射与图网（graphon）理论；

**📊 数据集**

主要数据集为：大规模脑磁共振成像（CamCAN、DLBS、IXI、eNKI）用于年龄预测；人工生成的线性回归数据集用于稳健性实验；此外使用不同分辨率的Schaefer脑区分区数据评估多尺度迁移；

**📈 对比分析**

比较方法：对PCA-线性回归、PCA-RBF核回归与VNN进行相同任务的性能对比，指标为MAE、Pearson相关系数。结果显示VNN在所有实验中表现稳定、MAE约0.1–0.2年，相关系数高于0.83；PCA模型对协方差扰动敏感，性能波动大；VNN的迁移实验表明不同尺度下预测结果相关系数>0.97；

**⚠️ 局限性**

局限性：1）理论证明基于理想的协方差估计与谱分隔性，实际数据中近似值仍可能导致误差；2）VNN仍需训练标签，标签稀缺时效果未知；3）对极大维度时内存与计算成本仍有挑战；4）稀疏与鲁棒扩展需额外超参数调优；5）在某些任务中VNN的解释性与传统PCA相比尚不明显。

---

## 503. A Formal Framework for Noisy Runtime Verification

**arXiv ID:** 2609.10462 | [PDF](https://arxiv.org/pdf/2609.10462v1)

**作者:** Shay Allen Logan `[一作]` (Kansas State University), Thomas Ferguson `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种新的命题式模态逻辑 EDMon，用以在存在噪声的运行时验证环境中形式化并比较监控性（monitorability）、活跃性（liveness）类以及免疫性（immunity）等概念。

**💡 创新点**

创新点在于：①将噪声（mutation）建模为动态模态，构成了兼具 LTL、动态逻辑与知识模态的统一框架；②通过 EDMon 语义与证明，能够精确表述三种主流监控性定义以及它们与活跃性类和免疫性之间的关系；③证明了 EDMon 能够自然地捕捉到 NRV 文献中的主要定理，显示其在理论上的充分性和通用性。

**🔧 技术方法**

技术主要包括：1）动态模态 ⟨μ⟩ 及其逆、星号等组合；2）观测模态 ⟨⟩、⟨^*⟩；3）知识模态 “determines”；4）对 LTL 的扩展；5）形式语义与推理规则（如对偶、组合、归纳证明）。

**📊 数据集**

该工作并未使用具体实验数据集；主要是理论形式化与证明。

**📈 对比分析**

通过与已有 NRV 研究中提出的监控性定义、活跃性层级和免疫性概念的对比，证明 EDMon 能够重现并统一它们的结果。实验性比较未展开，性能（如可判定性、复杂度）在本文中未给出具体数值，而是通过证明 EDMon 能够表达这些概念来展示其理论性能。

**⚠️ 局限性**

局限性包括：①未对四值监控性（4-valued monitorability）进行处理；②对多代理系统、分支时序或一阶动态逻辑的扩展未实现；③对复杂性与可判定性分析留待后续工作；④对 HyperLTL 等更表达式丰富的逻辑尚未直接嵌入 EDMon。

---

## 504. Quantum Feature Engineering for Credit Default Prediction: When and Why IQP Circuits Help Linear Classifiers

**arXiv ID:** 2609.10505 | [PDF](https://arxiv.org/pdf/2609.10505v1)

**作者:** Menachem Finkelstein `[一作]` (Reichman University), Sarel Cohen `[通讯]` (Reichman University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究 IQP 电路生成的特征是否能在信用违约预测中提升线性分类器的性能。

**💡 创新点**

在相同特征预算下对量子与经典无监督非线性特征提取进行严格多重比较校正，证明 IQP 在逻辑回归上显著优于 Kernel PCA，并且仅对线性模型有效。

**🔧 技术方法**

使用 8‑qubit IQP 量子特征映射提取 16 个期望值特征；对比逻辑回归、随机森林、SVM、XGBoost、k‑NN 等分类器；与 PCA、SVD、ICA、随机投影、特征聚合、二次多项式、KPCA、噪声基线等经典方法比较；采用 Benjamini–Hochberg FDR 校正。

**📊 数据集**

UCI Default of Credit Card Clients 数据集（30,000 条记录，23 个金融特征，默认率 22.6%）。

**📈 对比分析**

采用 5 折分层交叉验证，比较追加 16 个量子特征与 16 个经典特征对 F1、准确率和 AUC 的影响；逻辑回归 F1 从 0.462 提升到 0.517（+0.055，p<0.0001），优于 KPCA 的 +0.031；其他非线性模型无显著提升。

**⚠️ 局限性**

仅在经典模拟器上验证，未实现量子硬件加速；量子特征对非线性模型无效；特征选择对结果影响大，需选取信息量高的特征；对硬件噪声、误差等实际限制未知。

---

## 505. iLogMap: Geodesic Polar Coordinates Parameterization with the Magnetic Laplacian

**arXiv ID:** 2609.10503 | [PDF](https://arxiv.org/pdf/2609.10503v1)

**作者:** Tomás Banduc `[一作]` (Pontificia Universidad Católica de Chile), Francisco Sahli Costabal `[通讯]` (Pontificia Universidad Católica de Chile)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出iLogMap方法，利用磁拉普拉斯算子求解曲面（及体积）几何极坐标的角度分量，实现高精度、全局一致的GPC参数化；

**💡 创新点**

将角度同步问题映射为磁拉普拉斯基态特征问题，解耦径向与角度，支持各向异性度量、异质性、体积域，并可选择切点移除实现连续角度场；

**🔧 技术方法**

采用Fast Iterative Method求解Eikonal与Jacobi因子，有限元P1组装磁拉普拉斯，shift‑and‑invert幂迭代求特征；对比Affine Heat Method和Laplace插值；对异质/各向异性度量做适配；

**📊 数据集**

实验使用典型曲面（半球、圆环/多孔球、双环、兔子模型等）、带边界曲面、异质或各向异性导电率平板、心脏模型（左心房、左心室）以及球、圆柱、托罗斯等三维体素网格；

**📈 对比分析**

与AHM和Laplace基线对比，iLogMap在角度误差和度量失真上相当或更优，尤其在高基数、边界、异质/各向异性场景；运行时间与AHM相近，切点移除版进一步降低失真；体积参数化误差低于5%；

**⚠️ 局限性**

需要对每个源点重新构造磁拉普拉斯矩阵并做LU分解；切点检测阈值需手工设定；对多源点应用效率低；目前仅适用于二维流形和三维体积，未扩展到高维；对不规则网格仍有限；缺乏自动阈值和预计算策略。

---

## 506. Coastal Environment Generation with HoloOcean

**arXiv ID:** 2609.10484 | [PDF](https://arxiv.org/pdf/2609.10484v1)

**作者:** Abigail Austin `[一作]` (Brigham Young University), Joshua G. Mangelson `[通讯]` (Brigham Young University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一套从单张航拍图自动生成海岸环境的 Unreal Engine 5 级别流水线，能够生成高度图、语义分割、材质贴图、资产匹配与摆放，最终产出可直接用于 HoloOcean 的完整模拟场景。

**💡 创新点**

整合多种 AI 技术（Gemini 语义分割、Seabed‑Net 高度预测、YOLO 检测、DINOv3 嵌入、FAISS 相似搜索）实现无专家操作的一键式海岸环境自动生成，并提供完整的资产匹配与 PCG 生态。

**🔧 技术方法**

使用了 Unreal Engine 5、Google Gemini、Seabed‑Net、YOLO、DINOv3、FAISS、UE PCG、Raycast Semantic LiDAR、Imaging Sonar 等技术。

**📊 数据集**

基准数据包括自制的五个人工级别合成环境的航拍图和真实世界的 Hale‘iwa Boat Harbor 航拍图；资产库由 159 个 UE 资产构成。

**📈 对比分析**

通过对比真实/合成地面真值的资产位置 RMSE、旋转误差、精确率与召回率评估效果；低分辨率下位置误差<0.5 m，精度0.855‑0.992，召回率0.965‑0.978；高分辨率精度提升，生成时间从几秒到约5分钟，明显快于人工建模。

**⚠️ 局限性**

受 Seabed‑Net 高度预测不够平滑、Gemini 语义分割细节不足、YOLO 对遮挡/靠近对象的检测失误以及旋转误差大、未考虑对象朝向等因素限制，导致真实世界细粒度匹配仍不完美。

---

## 507. Towards Tackling Application Logic Flaws through Autonomous Formal-Logic Modeling and Automated Reasoning

**arXiv ID:** 2609.10537 | [PDF](https://arxiv.org/pdf/2609.10537v1)

**作者:** Yiwei Fang `[一作]` (Chinese Academy of Sciences), Luyi Xing `[通讯]` (University of Illinois)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 LL‑Verifier 框架，能够自动将自然语言协议说明转化为形式化逻辑模型，并利用模型检查技术在 IoT 设备协议中发现逻辑漏洞。

**💡 创新点**

创新点包括：① 通过 LLM 实现自治建模；② 设计了针对 LLM 的“模型安全护栏”（FMG）专用逻辑语言；③ 在非 Dolev‑Yao 的威胁模型下实现精细化事件生成与模型检查；④ 提供首个面向 IoT 协议的自动化建模基准（LL‑Bench）和完整工具链。

**🔧 技术方法**

采用的大技术包括：大语言模型（LLM）+自定义形式化逻辑语言与语法护栏；Rewrite Logic / Maude 形式化模型与 LTL 逻辑模型检查；自动化编译器将模型转换为可执行的 Rewrite Theory；代理式迭代修复与程序分析器用于消除 LLM 幻觉与语义错误。

**📊 数据集**

使用的数据集包括 27 个 IoT 访问控制协议（如 iRobot、Google Home、Aqara 等）和公开协议基准；自建 LL‑Bench 基准集合包含原始协议文本、组织文本和对应的形式化模型，总计约 X 万词、约 44 条规则、10+ 角色和属性。

**📈 对比分析**

通过与手工模型、直接使用 Maude 编码以及多种 LLM（GPT‑4o、Claude‑Sonnet‑4 等）对比实验，模型覆盖率分别达 98%（角色）、99%（属性）、97%（规则），模型检查时间从几分钟到十几分钟不等；在真实设备上验证发现 17 个零日逻辑漏洞。

**⚠️ 局限性**

局限性：对复杂语义的准确生成仍需人工校正；威胁模型必须由专家手工指定；在大规模协议或高度并发场景下仍可能面临状态空间爆炸问题。

---

## 508. Programmable World Model

**arXiv ID:** 2609.10540 | [PDF](https://arxiv.org/pdf/2609.10540v1)

**作者:** Zheng-Hui Huang `[一作]` (Alaya Lab), Zhixiang Wang `[通讯]` (Alaya Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了可编程世界模型（Programmable World Model），通过将世界状态演化与视频生成解耦，使得交互式视频能在保持显式持久状态的同时，利用生成模型渲染逼真视觉。

**💡 创新点**

创新点在于：①引入状态增强的 3D 方向包围盒（OBB）作为中间抽象，既能提供全局一致的世界坐标，又足够简洁；②设计确定性状态编译器将 OBB 及其身份、语义、运动信息投影为像素对齐的控制图；③将传统的生成模型转变为渲染器，并通过 ControlNet 训练仅对控制图进行微调；④构建自动化数据管线和专门评测基准（CombatStateBench），实现对世界状态一致性的量化评估。

**🔧 技术方法**

所用技术包括：3D 方向包围盒检测（WildDet3D）、实例分割与跟踪（SAM3）、相机参数与深度估计（ViPE）、文本编码器（Qwen3-VL）、生成模型基座（LingBot-World-v1）、ControlNet 结构、基于光流/速度量化的方向图、时序与空间记忆机制（AlayaWorld）等。

**📊 数据集**

数据集主要来自无 HUD 的游戏视频：Cyberpunk 2077、Forza Horizon 6 与 GTA V；此外通过自动化数据引擎提取相机、语义标签、实例轨迹和 OBB，生成训练对齐的视频-控制样本，并在 CombatStateBench 上评估。

**📈 对比分析**

与两大基线 LingBot-World-V2 与 YUME 进行对比，采用 VBench 视觉质量指标、Count Accuracy 与 State Accuracy 评测。结果显示：Count Accuracy 94% / State Accuracy 98%，远超基线（约 40% 以上提升），同时在影像质量、主体一致性、背景一致性和时间稳定性上均取得最高分。

**⚠️ 局限性**

局限性包括：①仅使用 OBB 作为结构化表示，难以捕捉细粒度姿态、细节几何与复杂物理交互；②依赖 3D 检测与跟踪的精度，误检会导致状态不一致；③对极端视角变换或大范围遮挡时的姿态恢复仍有挑战；④当前评测聚焦于角色死亡与计数，未覆盖更细致的交互逻辑与多物体协同。

---

## 509. Guiding Image-to-3D Generation with Test-Time Partial Observations

**arXiv ID:** 2609.10531 | [PDF](https://arxiv.org/pdf/2609.10531v1)

**作者:** Jerred Chen `[一作]` (University of Oxford), Ronald Clark `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种训练无关的框架，利用测试时可获得的局部几何观测信息，在不重新训练或微调的情况下引导预训练的图像到3D生成模型生成更符合几何真实性的3D资产。

**💡 创新点**

核心创新在于将后验引导视为对预训练流场景的形变，并设计了基于占据网格的射线一致观测似然，将表面占据与空域信息结合，实现对局部观测的直接约束；同时该方法可无缝集成到多视图扩展模型（MV‑SAM3D）中。

**🔧 技术方法**

使用流匹配（flow‑matching）和采样时的梯度引导（类似classifier‑free guidance）技术；构建了基于射线一致性的占据概率观测能量；采用可调的引导权重、步长、冷却策略等技巧以提升几何精度和视觉质量。

**📊 数据集**

在GSO‑30真实物体数据集上进行实验，设置高、中、低可观测性（分别使用5、2、1张视角图像）。

**📈 对比分析**

与SAM3D、MV‑SAM3D、SpaceControl三种基线对比，使用Chamfer距离、单向距离和LPIPS评估。在所有可观测性设置下，本文方法在几何精度（CD、SD）和视图合成质量（LPIPS）上均优于或与最强基线相当，尤其在低可观测性时表现最显著。

**⚠️ 局限性**

局限性包括：假设局部观测准确且已对齐到模型坐标系；使用占据网格导致空间分辨率有限；对噪声或未对齐观测的鲁棒性不足；目前仅在SAM3D及其多视图扩展上验证，其他生成模型的推广仍待研究。

---

## 510. Emergency Department Revisit Quality Review Screening: Exploring Human Decision-Making and Artificial Intelligence Support

**arXiv ID:** 2609.10421 | [PDF](https://arxiv.org/pdf/2609.10421v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 511. Risk-Averse Decision Making via Quantum Measurement Design

**arXiv ID:** 2609.10482 | [PDF](https://arxiv.org/pdf/2609.10482v1)

**作者:** Meiyi Zhu `[一作]` (King's College London), Osvaldo Simeone `[通讯]` (Northeastern University London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计一种量子测量方法，使测量结果作为决策动作时，最大化风险厌恶型指标（OCE）

**💡 创新点**

① 将OCE优化问题转化为有限个半正定规划（SDP）；② 在二元状态辨识中给出闭式Helstrom测量解；③ 通过数值实验展示风险厌恶能显著改善低尾分布

**🔧 技术方法**

OCE/ CVaR 框架、分段线性收益函数、半正定规划及其对偶、Helstrom测量分析

**📊 数据集**

利用合成数据：8维 Hilbert 空间，4个状态，先随机生成秩为4的密度矩阵，再通过参数 γ 混合噪声，形成 ρ_s = γσ_s + (1-γ)C(σ_s)

**📈 对比分析**

与两阶段基于计算基的固定 POVM（仅做投影再经典决策）做对比；结果显示：随着 γ 的增加，最优测量在平均效用和 CVaR 上都优于基准，且 CVaR 能把低尾风险降低而平均效用损失有限

**⚠️ 局限性**

仅适用于分段线性收益函数；对更一般的非线性收益函数需要新的分析；计算量随 kinks 数量 K 增大而提升；实验仅在合成数据上验证，未测试在更大维度或真实量子系统上的可扩展性

---

## 512. AgroVisNet: A lightweight Convolutional Network and the BD-PlantDX Expert-Validated Benchmark for Radish, Potato and Pointed Gourd Disease Classification

**arXiv ID:** 2609.10469 | [PDF](https://arxiv.org/pdf/2609.10469v1)

**作者:** Md. Abdullah Mandal `[一作]` (Bangladesh Army University of Science & Technology), Md. Khalid Syfullah `[通讯]` (Bangladesh Army University of Science & Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了一个从零开始的轻量卷积网络 AgroVisNet，并发布了一个 12,432 张农作物叶片/根部图像的专家验证数据集 BD‑PlantDX。

**💡 创新点**

创新点在于：① 设计了包含分组瓶颈残差块、连续通道+空间注意力、多尺度深度可分卷积以及双池化分类头的架构，能够在 0.3 M 参数内实现 99.52% 的准确率；② 公开了区域特定、专家验证的多作物病害数据集；③ 在同一实验协议下，证明该模型比所有 ImageNet 预训练轻量骨干更精确且更小。

**🔧 技术方法**

使用技术包括：分组卷积、深度可分卷积、通道注意力 (SE)、空间注意力、双池化 (GMP+GAP) 分类头、Swish 激活、批归一化、Dropout、权重衰减、随机旋转/缩放/平移/对比度增强、Grad‑CAM 与 Grad‑CAM++ 可解释性。

**📊 数据集**

使用数据集：BD‑PlantDX（12 类，共 12,432 张，来自罗卜、土豆、指瓜的健康与病害状态），以及在迁移实验中对 VegNet‑BD（21 类）和 RadishLeaf‑BD（5 类）进行验证。

**📈 对比分析**

方法：与六个 ImageNet 预训练轻量骨干（EfficientFormerV2‑S0、RepViT‑M1.0、ConvNeXt‑Atto、FastViT‑T8、MobileNetV4‑Conv‑Small、GhostNetV2‑1.0）在相同划分与训练协议下对比。AgroVisNet 以 0.29 M 参数取得 99.52% 准确率，参数量比最小骨干低 8.7×，MACs 低 1.3×；在四个大型骨干的集成模型（VGG16+ResNet50+InceptionV3+MobileNetV2）下仍以更小参数实现更高准确率。迁移到 VegNet‑BD 与 RadishLeaf‑BD 分别获得 98.71% 与 99.05% 的准确率。

**⚠️ 局限性**

局限：① 图像采集在统一光照和背景下，缺乏对自然背景、不同光照条件的评估；② 仅覆盖两地区、单季节和三种作物；③ 对目标硬件的持续负载、内存占用及热量未测；④ 对噪声、亮度变化的鲁棒性有限；⑤ 未提供严重度等级标签，缺乏对病害严重程度的判别。

---

## 513. Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization

**arXiv ID:** 2609.10464 | [PDF](https://arxiv.org/pdf/2609.10464v1)

**作者:** Andy Zeyi Liu `[一作]`, John Sous `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了基于重力条件的Semigroup-JEPA（SG-JEPA）世界模型，利用编码器与预测器的联合训练和递归潜在回归来学习物理动力学并实现长期预测。

**💡 创新点**

创新点在于将重力作为动作参数注入模型，并通过多步自回归损失与SIGReg正则化共同训练编码器和预测器，同时使用线性特征模型解释长时序下的OOD泛化优势，证明编码器的表示是决定性因素。

**🔧 技术方法**

使用ViT Tiny编码器、GRU/SSM/Transformer预测器、SIGReg正则、MuAdam优化器、基于回归的Diffusion Policy控制以及线性特征模型进行理论分析。

**📊 数据集**

构建了八个MuJoCo数据集，包括二维平面自由落体、投射物体以及三维机器人臂控制，训练时重力在狭窄高斯分布内采样，测试时扩展到更宽的重力网格。

**📈 对比分析**

与LeWM和DINO-WM基线对比，采用物理状态MSE和控制成功率等指标评估；SG-JEPA在长时序预测误差上下降约30-50%，在OOB重力下控制成功率提升至原基线的2.5倍。

**⚠️ 局限性**

局限性包括只考虑单一标量重力，形状转移的泛化不均衡，以及理论模型仅基于线性特征，无法完全解释非线性和碰撞等复杂动力学。

---

## 514. JarvisGUI: Towards Cross-Device GUI Agents with Dynamic Task Composition

**arXiv ID:** 2609.10451 | [PDF](https://arxiv.org/pdf/2609.10451v1)

**作者:** Zixiang Chen `[一作]` (Beihang University), Haifeng Wang `[通讯]` (Baidu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了JarvisGUI跨设备GUI评测基准，评估智能体在Android、Windows与Ubuntu等异构平台上的协同工作能力

**💡 创新点**

首次系统构建多设备工作流任务，使用基于槽位的类型系统实现任务自动组合与验证

**🔧 技术方法**

采用Planner–Grounder两阶段架构，利用Qwen3‑VL‑Plus进行跨平台规划与低层视觉定位，后端通过Docker虚拟化多设备环境

**📊 数据集**

基于人工标注模板任务与自动合成，构成约150个跨平台合成任务（共442子任务），并生成任务配置文件与评估脚本

**📈 对比分析**

与多款开源GUI代理（Qwen3‑VL‑30B、HOLO2‑30B、UI‑TARS‑1.5‑7B、MAI‑UI‑8B、UI‑Venus‑Ground‑7B、GUI‑Owl‑32B）对比，使用任务成功率（TSR）和子任务成功率（SSR）评估，结果显示跨设备依赖任务性能极低，单机任务虽有进展但仍不足

**⚠️ 局限性**

受限于Docker/KVM资源、缺乏可访问性场景及场景覆盖有限，导致可复现性和实际应用范围受限

---

