# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-21 | 今日论文总数: 662

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Multi-Agent Reinforcement Learning for Safe Autonomous Driving Under Pedestrian Behavioral Uncertainty

**arXiv ID:** 2605.20255 | [PDF](https://arxiv.org/pdf/2605.20255v1)

**作者:** Prakash Aryan `[一作]` (University of Bern), Sebastiano Panichella `[通讯]` (University of Bern)

**通讯引用:** 5258 | [OpenAlex ID](https://openalex.org/A5063227479)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在多智能体强化学习框架下，共训练自动驾驶车辆（SDC）与12名具有隐藏性格特征的行人，探索如何在仿真中更真实地模拟人行横道与闯红灯的交互场景。

**💡 创新点**

① 将行人的闯红灯行为与隐藏的性格特征关联，形成可变的行为不确定性；② 引入速度差异指标（Speed Differential）直接从轨迹测量SDC对可预见与不可预见穿行的响应差异；③ 通过共同训练使行人自发学习等候行为，显著降低碰撞。

**🔧 技术方法**

多智能体近端策略优化（MAPPO）结合共享集中式评论家；使用JAX + Flax实现GPU加速；自定义奖励函数平衡行人行进与SDC目标达成；利用速度差异指标评估不确定性。

**📊 数据集**

采用自构建的120×120 m城市地图仿真环境，包含四向交叉口与T型路口，设有20条人行道、6条斑马线、12名行人和一辆SDC；不使用公开标准数据集，而是基于Dijkstra路径规划与脚本化运动生成多种交互场景。

**📈 对比分析**

与四种基准（全速、反应式刹车、规则式行人、单智能体RL）以及单智能体RL进行比较。共训练的SDC在500轮评估中达到78%到达率、14%碰撞率，显著优于最佳规则基准（35%/33%）和单智能体RL（65%/20%）。行人等候行为将碰撞率从20%降低到14%，并在面对脚本化行人时仍保持较高成功率。

**⚠️ 局限性**

限制：① 仅在有限规模（12名行人）和单一城市地图上验证，缺乏对更大交通环境的推广性；② 速度差异指标虽然直观但未结合其他不确定性量化方法；③ 行人行为仍基于简化的Dijkstra路径，缺乏对更复杂人类动态决策的建模；④ 性格特征仅影响闯红灯概率，未考虑其他行为维度。

---

## 2. Conformal Selective Acting: Anytime-Valid Risk Control for RLVR-Trained LLMs

**arXiv ID:** 2605.20270 | [PDF](https://arxiv.org/pdf/2605.20270v1)

**作者:** Hamed Khosravi `[一作]` (Georgia Institute of Technology), Xiaoming Huo `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5719 | [OpenAlex ID](https://openalex.org/A5014880531)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在线可信、任何时刻路径有效的选择性风险控制包装器（Conformal Selective Acting），用于本地专家LLM的RLVR部署，保证每一轮的误差率不超过合同规定阈值。

**💡 创新点**

创新点在于将按阈值构造的e-process与非拒绝部署规则相结合，实现任意时间路径有效的选择性风险控制，并实现速率最优的认证与无时长的释放率差距。

**🔧 技术方法**

使用了e-process超级马尔可夫过程、Ville不等式、可预测的bet、Bonferroni多阈值网格以及RLVR训练与在线LoRA更新的RL框架。

**📊 数据集**

使用了八个高风险专家基准（医学、法律、金融、科学等）共480+100+160个流、四个基础模型（Med42-8B、Saul-7B、Qwen2.5-Math-7B、Llama-3.2-3B-Inst.）以及16个对抗性分布偏移流。

**📈 对比分析**

与十个现有方法（包括离线合成、在线平均、边缘时间、A-RCPS等）对比，所有测试中实现零路径违规且非拒绝率均高于50%，在对抗性场景下平均误差率仅3.2%，显著优于其他方法。

**⚠️ 局限性**

局限性包括：仅适用于确定性验证器；假设每轮间只有一次更新；需要一次性校准；对概率或不完整验证器的情况尚未覆盖。

---

## 3. High Quality Embeddings for Horn Logic Reasoning

**arXiv ID:** 2605.20467 | [PDF](https://arxiv.org/pdf/2605.20467v1)

**作者:** Yifan Zhang `[一作]` (Lehigh University), Jeff Heflin `[通讯]` (Lehigh University)

**通讯引用:** 4173 | [OpenAlex ID](https://openalex.org/A5065113414)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并改进了用于一阶逻辑推理的逻辑句子嵌入模型，并评估其在基于神经网络的引导归纳推理中的性能。

**💡 创新点**

提出三项改进：增加重复项原子生成、基于难度平衡的三元组生成以及周期性聚焦硬样本的训练策略，以提升嵌入的表达能力。

**🔧 技术方法**

采用triplet loss学习嵌入，结合反向链推理、神经网络评分模型，并使用自生成的合成知识库和查询进行实验。

**📊 数据集**

主要使用合成的 Horn 片段知识库，规模为 250、375、500 条语句，常用 200-400 个常量和随机生成的查询。

**📈 对比分析**

与标准反向链推理器及之前的嵌入方法对比，平均节点数下降 70–90%，中位数显著降低，证明新嵌入显著提升搜索效率。

**⚠️ 局限性**

嵌入对不同知识库的泛化差异大，且实验仅限于合成 Horn 片段，对更大规模或全一阶逻辑的适用性尚未验证。

---

## 4. Pseudo-Siamese Network for Planning in Target-Oriented Proactive Dialogues

**arXiv ID:** 2605.20195 | [PDF](https://arxiv.org/pdf/2605.20195v1)

**作者:** Xinyue Kang `[一作]` (Soochow University), Fang Kong `[通讯]` (Soochow University)

**通讯引用:** 800 | [OpenAlex ID](https://openalex.org/A5102803936)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并提出了一种前向聚焦双向伪双子网络（FF‑BPSN），用于目标导向主动对话系统的对话路径规划，并将规划结果用于引导语言模型生成回复。

**💡 创新点**

创新点在于首次将双向规划与前向聚焦模块结合：通过双Transformer解码器生成正向与逆向路径，并利用前向聚焦特征融合机制在保持双向信息的同时强调前向信息，从而显著提升路径质量与对话连贯性。

**🔧 技术方法**

技术方案包括Transformer‑based解码器、Pseudo‑Siamese结构、前向聚焦特征融合（ReLU+MLP + Sigmoid权重）、知识‑目标互注意（KT）、交叉熵损失与相似度约束以及基于该路径的Prompt‑guided PLM/LLM生成。

**📊 数据集**

使用的公开数据集为DuRecDial（中文）和DuRecDial 2.0（英文），两者均包含多轮对话、领域知识和用户画像。

**📈 对比分析**

在路径规划准确率、回复生成F1、BLEU、DIST、知识F1、成功率等指标上，FF‑BPSN与多种基线（MGCG, KERS, TPC‑BART/GPT, BART, GPT‑2, DialoGPT, LLaMA‑1B/3B）对比，均取得SOTA表现；尤其在LLaMA‑3B+FF上实现了最高的F1、BLEU、知识F1和成功率。

**⚠️ 局限性**

局限性包括对长文本和不同领域知识的泛化能力尚待验证；模型依赖大规模GPU，训练成本高；对用户兴趣动态变化的实时适应性相对有限。

---

## 5. Combined Program Analysis Techniques: A Systematic Mapping Study

**arXiv ID:** 2605.20310 | [PDF](https://arxiv.org/pdf/2605.20310v1)

**作者:** Pietro Braione `[一作]` (University of Milano-Bicocca), Martino Tessaro `[通讯]` (University of Milano-Bicocca)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对近几年结合程序分析技术的研究进行系统映射研究，并提出了一个新的分类体系以描述组合分析的工作流、协同效应与映射函数。

**💡 创新点**

创新点在于提出了基于协同效应、交互工作流与映射函数三维度的完整分类法，为理解与设计组合程序分析技术提供了系统框架。

**🔧 技术方法**

采用系统映射研究方法、文献检索与筛选、案例抽取与分类等技术。

**📊 数据集**

使用了从 Scopus 检索得到的 2,776 篇候选论文，经过筛选后共计 1,049 篇原始研究作为分析对象。

**📈 对比分析**

通过回答 RQ1–RQ4，系统对组合技术的类型、工作流与映射机制进行归纳与统计，呈现了不同组合技术在文献中的分布与共性，未进行传统实验性能对比。

**⚠️ 局限性**

限制在于仅聚焦软件工程领域的文献；文献解读带有主观色彩；未评估具体实现的性能与实验结果。

---

## 6. Provably Learning Diffusion Models under the Manifold Hypothesis: Collapse and Refine

**arXiv ID:** 2605.20235 | [PDF](https://arxiv.org/pdf/2605.20235v1)

**作者:** Wei Huang `[一作]` (Riken Aip), Kenji Fukumizu `[通讯]` (Institute Of Statistical Mathematics)

**通讯引用:** 9746 | [OpenAlex ID](https://openalex.org/A5012922872)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出Score-induced Latent Diffusion (SiLD)，两阶段训练：低噪声阶段学习数据流形几何，后期阶段在流形上估计密度，全部通过一个去噪分数匹配（DSM）目标实现。

**💡 创新点**

创新点在于：①利用分数函数在低噪声下的奇异性驱动“collapse‑and‑refine”机制，自动完成流形学习与密度估计；②不再需要KL正则或VAE编码器，理论证明样本复杂度只随内在维度而非环境维度；③提供完整的理论分析（梯度流、随机特征回归、逆SDE采样）与实验验证。

**🔧 技术方法**

技术包括：两层保守形式网络（对低噪声阶段）与随机特征网络（对密度估计阶段）；梯度下降与均值场（Mean‑field）梯度流分析；随机特征回归与岭回归；逆扩散SDE采样以及高噪声阶段的随机特征头。

**📊 数据集**

数据集：Stacked MNIST、CelebA与CelebA‑HQ（图像），四个MoleculeNet基准（QM9、HIV、MUV、PCBA）用于分子生成。

**📈 对比分析**

与基于VAE的LDM（KL正则）、MMD或GAN+EMA等方法对比，SiLD在FID、LPIPS、重建MSE、分子有效率、独特性和内部多样性等指标上表现更优；在分子生成任务中显著避免LDM的分布坍塌，保持高多样性与真实性。

**⚠️ 局限性**

限制：需要显式的两阶段训练与噪声尺度调参；理论主要针对平滑流形，对离散或高度非光滑流形的适用性尚未充分验证；高噪声阶段仍需额外的随机特征头，复杂度与环境维度有关。

---

## 7. Graph Transductive Sharpening: Leveraging Unlabeled Predictions in Node Classification

**arXiv ID:** 2605.20248 | [PDF](https://arxiv.org/pdf/2605.20248v1)

**作者:** Brown Zaz `[一作]` (University of Cambridge), Pietro Liò `[通讯]` (University of Cambridge)

**通讯引用:** 25893 | [OpenAlex ID](https://openalex.org/A5056748708)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在传导式节点分类中提出了一种新的训练目标——Transductive Sharpening（TS），通过对无标签节点的预测分布施加熵约束，使模型在训练过程中直接利用无标签节点的预测信息。

**💡 创新点**

创新点在于：①将无标签节点的预测熵作为正则项加入损失；②在无标签节点上最小化熵、在标签节点上最大化熵的对称机制，从而在不改变模型结构的前提下提升训练效果；③使用Tsallis熵（q=2）替代传统Shannon熵，以获得更稳定的梯度。

**🔧 技术方法**

技术：基于图神经网络（GCN、GraphSAGE、GAT）和MLP的交叉熵损失，加入自定义的熵项；采用Tsallis熵（Gini impurity）作为不确定性度量；在训练中调节单一超参数λ，实现无架构改动的loss-level修正。

**📊 数据集**

数据集：13个主流节点分类基准，包括 Cora、CiteSeer、PubMed、Computer、Photo、CS、Physics、WikiCS、Squirrel、Chameleon、Amazon-Rat、Roman-Empire、Minesweeper。

**📈 对比分析**

比较方法：在上述所有模型上与标准交叉熵训练 baseline 进行对比，使用相同的超参数搜索。实验结果显示 TS 在大多数模型/数据集上均能提升准确率，平均提升约 0.5%–2%（如 GCN 在 Cora 上从 84.54% 提升至 85.74%），且在使用统一 λ=0.25 时仍保持较好性能，证明其稳定性。

**⚠️ 局限性**

局限性：①对 λ 的取值敏感，过大会导致过度自信并损失性能；②TS 仅提升训练信号，无法替代图结构信息传递；③目前仅在静态传导式节点分类上验证，尚未探讨动态图或链路预测等其他任务；④自适应 λ 的 meta-learning 方案实验效果不佳，表明进一步的动态调整仍需研究。

---

## 8. PrivacyAkinator: Articulating Key Privacy Design Decisions by Answering LLM-Generated Multiple-choice Questions

**arXiv ID:** 2605.20206 | [PDF](https://arxiv.org/pdf/2605.20206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 9. Online Conformal Prediction with Corrupted Feedback

**arXiv ID:** 2605.20515 | [PDF](https://arxiv.org/pdf/2605.20515v1)

**作者:** Bowen Wang `[一作]` (King's College London), Osvaldo Simeone `[通讯]` (Northeastern University)

**通讯引用:** 17201 | [OpenAlex ID](https://openalex.org/A5017736224)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究在线自适应置信预测在反馈被腐蚀时的鲁棒性，并提出两种基于滤波与主动补偿的鲁棒算法，证明其误覆盖率可控。

**💡 创新点**

创新点在于把反馈错误建模为任意二值翻转，并设计滤波式鲁棒 OCP 与主动补偿式 OCP，给出针对独立随机与有记忆的错误模型的显式误覆盖保证。

**🔧 技术方法**

使用在线凸优化（OGD）框架、误覆盖指标分析、Krichevsky–Trofimov 估计、主动训练与误差补偿等技术。

**📊 数据集**

实验数据集包括 CIFAR-100（分类）与 AVA（回归），采用预训练 ResNet‑18 与 VGG16 作为基模型。

**📈 对比分析**

与传统 OCP、理想反馈 OCP、F‑ROCP 等基线对比，结果显示在随机与记忆性错误环境下，AC‑ROCP 能保持目标误覆盖率并显著减小预测集大小，优于其他方法。

**⚠️ 局限性**

限制在于需预先设定训练步数/间隔，且对极端高噪声或持续记忆错误仍存在一定保守性，算法对参数敏感。

---

## 10. Nonlocal operator learning for fMRI encoding and decoding tasks

**arXiv ID:** 2605.20389 | [PDF](https://arxiv.org/pdf/2605.20389v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 11. ProcBench: Evaluating Process-Level Defects and Control Preservation in LLM Coding Agents

**arXiv ID:** 2605.20251 | [PDF](https://arxiv.org/pdf/2605.20251v1)

**作者:** Jiawei He `[一作]` (Amap), Dong Sun `[通讯]` (Amap)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ProcBench框架，用于评估LLM编码代理的执行轨迹和过程缺陷，而不仅仅关注最终结果。

**💡 创新点**

创新点在于构建可复用的过程缺陷本体、统一轨迹表示、缺陷证据提取后进行贝叶斯校准风险估计，并引入控制保留维度以衡量治理能力。

**🔧 技术方法**

采用了轨迹标准化映射、缺陷检测器、贝叶斯校准、风险分级以及多维度得分卡报告等技术。

**📊 数据集**

使用了200条来自AndroidBench、TerminalBench和SWE-bench-Verified的执行轨迹进行标注和评估。

**📈 对比分析**

通过与多种代理-模型组合对比，ProcBench在保持整体排名不变的同时揭示更多过程质量差异，校准后的风险分布更稳定，整体性能与传统终端指标相当但提供更细粒度诊断。

**⚠️ 局限性**

局限包括缺陷本体不完整、部分缺陷难以观测、对标注数据的依赖、校准样本规模有限以及控制保留指标仅为操作代理的近似衡量。

---

## 12. Creating Learning Scaffolds for Engineering Design Using Concept Catalyst

**arXiv ID:** 2605.20511 | [PDF](https://arxiv.org/pdf/2605.20511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 13. AirfoilGen: A valid-by-construction and performance-aware latent diffusion model for airfoil generation

**arXiv ID:** 2605.20303 | [PDF](https://arxiv.org/pdf/2605.20303v1)

**作者:** Zhijie Yang `[一作]` (Zhejiang University), Qiang Zou `[通讯]` (Zhejiang University)

**通讯引用:** 746 | [OpenAlex ID](https://openalex.org/A5028026050)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于圆扫描写（CS-Rep）的空中翼形生成框架 AirfoilGen，能够保证生成形状的几何有效性并在潜在空间中实现对升力和阻力的性能控制。

**💡 创新点**

1) 引入 CS-Rep 作为形状表示，天然约束生成过程保证非自交、单闭合、平滑厚度分布；2) 通过残差向量量化 + Transformer 编码/解码的自编码器把翼形映射到潜在空间；3) 在该潜在空间训练条件扩散模型（带分类器无指导）实现对性能区间的精准控制；4) 构建超过 200k 的翼形与性能标签数据集，显著提升模型训练效果。

**🔧 技术方法**

残差向量量化 (RVQ)、Transformer 编码器/解码器、潜在空间条件扩散模型 (conditional DDPM) 与分类器无指导 (CFG)、基于高斯噪声的扩散过程、自动微分与对抗性训练。

**📊 数据集**

自建的 200k+ 空中翼形数据集，包含 NACA‑4、NACA‑5 系列经弧长采样得到的 CS-Rep 表示以及由 NeuralFoil 计算得到的升力/阻力系数；对比使用传统 UIUC 数据集 1,650 条翼形。

**📈 对比分析**

在失真度（Chamfer、Hausdorff）、生成多样性与真实性（Fidelity、Diversity）、以及对 25 个性能区间的生成准确率（平均 98.41%）上与 BézierGAN、AirfoilDiffusion、CEBGAN 等方法进行定量比较。AirfoilGen 在几何有效性、性能控制、生成多样性和准确率方面均优于对比方法。

**⚠️ 局限性**

1) 仅采用 25 个离散性能区间，难以细粒度控制连续升阻值；2) 对极端 CL/CD 组合存在偏移，受训练数据尾部分布限制；3) 模型训练与推理对算力要求较高，需要 GPU 进行加速。

---

## 14. CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support

**arXiv ID:** 2605.20468 | [PDF](https://arxiv.org/pdf/2605.20468v1)

**作者:** Ricardo Diaz-Rincon `[一作]` (University of Florida), Benjamin Shickel `[通讯]` (University of Florida)

**通讯引用:** 3239 | [OpenAlex ID](https://openalex.org/A5019504069)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 CASCADE 框架，将第一阶段分类的置信度传播到第二阶段回归的预测区间，实现基于不确定性的动态区间调整；

**💡 创新点**

创新点在于利用 Venn‑Abers 生成的多概率区间直接映射为非一致性分数，构造连续可调的尺度函数 σ(x)，从而实现跨任务不确定性传递与区间自适应；

**🔧 技术方法**

技术手段包括 XGBoost 分类与回归模型、Venn‑Abers 置信区间、连续/离散级联的归一化 conformal prediction、β 参数调节的尺度函数以及标准 CP、CV+、J+aB 等基线对比；

**📊 数据集**

使用来自佛罗里达大学（UF Health）的 631 名帕金森病住院患者的 10 年医疗数据（包括 LEDD 变化率）作为实验数据集；

**📈 对比分析**

与 Naïve、Standard CP、CV+、J+aB 和离散 Mondrian CP 等基线相比，连续 CASCADE 在保持 80% 边际覆盖率的同时，将低不确定性患者的区间长度缩短 38.9%，在高不确定性患者区间长度扩大 158.9%，且 Cascade Ratio 达到 4.23（远高于基线 1.00），表明显著的适应性和效率提升；

**⚠️ 局限性**

局限性包括：采用对称尺度扩展，未考虑单侧风险不平衡；缺乏正式的拒绝（abstention）机制；需进一步在帕金森病全程和其他两阶段医疗任务中验证鲁棒性。

---

## 15. When Irregularity Helps: A Subclass Analysis of Inductive Bias in Neural Morphology

**arXiv ID:** 2605.20558 | [PDF](https://arxiv.org/pdf/2605.20558v1)

**作者:** Wen Zhang `[一作]` `[通讯]`, Wen Zhang

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了日语过去式变位，使用子组层次分析发现少数低频不规则动词导致错误集中，并通过选择性消融验证其影响。

**💡 创新点**

提出了子组感知评估框架，并发现单一低频不规则子类（Type 4‑2）对模型性能影响远大于整体不规则集合，揭示了不规则性对学习的非单调效应。

**🔧 技术方法**

使用基于 Transformer 的字符级编码-解码模型（SIGMORPHON 2020 与 2023 版本）并进行消融实验。

**📊 数据集**

使用 SIGMORPHON 2023 日语动词变位数据集（全部 3,958 条记录，包含 4 种动词类别）。

**📈 对比分析**

与标准基线进行对比，完整训练时模型准确率约为 97.9%–98%，但在去除 Type 4‑2 后可提升至 99.8%–99.9%，显示消融效果显著。

**⚠️ 局限性**

局限在于仅评估单一语言与单一变位任务，且仅使用 Transformer 基线，未检验跨语言或其他模型的普适性。

---

## 16. Closed-form predictive coding via hierarchical Gaussian filters

**arXiv ID:** 2605.20293 | [PDF](https://arxiv.org/pdf/2605.20293v1)

**作者:** Aleksandrs Baskakovs `[一作]` (Aarhus University), Nicolas Legrand `[通讯]` (Aarhus University)

**通讯引用:** 447 | [OpenAlex ID](https://openalex.org/A5072976435)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文将预测编码网络重构为深层层次高斯滤波器（HGF），通过一次性闭式变分更新实现局部推断与学习；

**💡 创新点**

创新点在于恢复可学习的精度矩阵并引入动态精度推断，从而实现自然梯度更新、一次性推断和Hebbian兼容权重更新；

**🔧 技术方法**

主要技术包括变分推断、一般化层次高斯滤波、精度权重的概率学习与闭式更新规则；

**📊 数据集**

使用FashionMNIST数据集进行分类实验；

**📈 对比分析**

与传统BP和常规预测编码网络对比，HGF在训练时间接近BP、在线学习、数据效率和概念漂移任务上均表现更优；

**⚠️ 局限性**

局限性在于目前仅支持全连接网络、未实现批量并行推断、对卷积架构与更大规模数据集的适应性尚未验证，且精度学习机制在深层网络中仍存在不稳定性。

---

## 17. Modern Portfolio Theory in the Crypto-Wilderness

**arXiv ID:** 2605.20528 | [PDF](https://arxiv.org/pdf/2605.20528v1)

**作者:** Ivan Vynyavskyy `[一作]` (Complexity Science Hub), Aviv Yaish `[通讯]` (Yale University)

**通讯引用:** 138 | [OpenAlex ID](https://openalex.org/A5047798226)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对以太坊链上ERC‑20资产持仓进行大规模重构，并与MPT理论最优组合对比，检验偏离有效前沿对收益的影响。

**💡 创新点**

首次以完整链历史为基础，利用公开交易日志实现账户级组合重建；系统化评估实际组合与受约束有效前沿的距离及其对实盘收益的预测。

**🔧 技术方法**

基于MapReduce日志提取、Ledger聚合、序列化快照；使用最小化方差、最大回报、最大夏普比率的序列化凸优化；计算ℓ1距离和CAPM风险调整收益。

**📊 数据集**

以太坊主网全链（2015–2025）ERC‑20转账日志以及Coingecko的代币信息与日均价格，覆盖约数千枚代币与数十万账户。

**📈 对比分析**

将实际持仓与三种MPT最优方案以及等权、按市值加权基准进行比较，结果显示市场加权在中位数回报和α上均优于MPT，且实际组合距离最优前沿不预测更好表现。

**⚠️ 局限性**

仅覆盖ERC‑20且仅基于单链地址，忽略ETH本地余额与离链托管资产；地址聚类不完善，可能低估实体级多元化；对高频交易策略与信息优势未做深入探究。

---

## 18. Continual Segmentation under Joint Nonstationarity

**arXiv ID:** 2605.20538 | [PDF](https://arxiv.org/pdf/2605.20538v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 19. Conflict-Aware Active Perception and Control in 3D Gaussian Splatting Fields via Control Barrier Functions

**arXiv ID:** 2605.20566 | [PDF](https://arxiv.org/pdf/2605.20566v1)

**作者:** Amirhossein Mollaei Khass `[一作]` (Lehigh University), Nader Motee `[通讯]` (Lehigh University)

**通讯引用:** 1742 | [OpenAlex ID](https://openalex.org/A5031516064)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在3D高斯点渲染（3D Gaussian Splatting）环境中提出一种冲突感知的主动感知与控制框架，既保证机器人安全又主动获取信息以减少地图不确定性。

**💡 创新点**

创新点在于：①将基于风险感知的平均值-风险（AV@R）与对齐的安全控制边界函数（CBF）结合，形成可微分的安全性约束；②设计空间与角度感知边界函数，将期望信息增益（EIG）与机器人朝向对齐；③通过在控制器中引入松弛变量，构建统一的安全-感知二次规划（QP），在冲突出现时自动放宽感知约束。

**🔧 技术方法**

使用的技术包括：3D Gaussian Splatting 场景建模、期望信息增益（Fisher信息）评估、风险感知（AV@R）计算、控制边界函数（CBF）与感知边界函数（Perception CBF）、冲突感知二次规划（CBF‑QP）、KKT分析。

**📊 数据集**

使用Lehigh University提供的四个高分辨率3DGS场景：Stonehenge（100K）、Statues（200K）、Flightgate（300K）、Adirondacks（500K）。

**📈 对比分析**

与现有基线（如SAFER‑Splat、Splat‑Nav等）在同一场景下对比；实验结果显示：安全成功率与基线相当或更高，最小安全距离显著提升，计算时间大幅下降（约30‑50倍），并且在信息获取上取得更高的期望信息增益。

**⚠️ 局限性**

局限性包括：①对高风险阈值设置较高时可能过于保守，导致在窄通道中行动受限；②目前仅在相对度为1或2的控制系统下验证；③实验仅在仿真环境，未在真实机器人上验证；④未考虑动态障碍或多机器人协同情形。

---

## 20. OSCToM: RL-Guided Adversarial Generation for High-Order Theory of Mind

**arXiv ID:** 2605.20423 | [PDF](https://arxiv.org/pdf/2605.20423v1)

**作者:** Sharmin Sultana Srishty `[一作]` (BRAC University), Shaikhul Islam Sinat `[通讯]` (BRAC University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 OSCToM 框架，用生成器（RL+DSL）合成 Observer‑Self Conflict 形式的高阶推理故事，并在此数据上对 Llama‑3.1‑8B‑Instruct 进行两阶段课程微调，得到 OSCToM‑8B。

**💡 创新点**

①将 Observer‑Self Conflict 概念引入 ToM 评测；②用扩展 DSL 描述 4 阶递归与欺骗操作；③使用 DQN‑guided RL 与组合 surrogate 评估实现大规模高效对抗性数据生成；④在无 A* 搜索的前提下实现 4 阶 ToM 推理的高准确率。

**🔧 技术方法**

强化学习（DQN）、组合 surrogate 评估（DistilBERT 6 组子模型）、LoRA 参数高效微调、4‑bit 量化、OpenRouter API 文本化、两阶段课程微调、DSL 语义编程。

**📊 数据集**

自建 OSCToM 数据集（≈15k 题干），以及公开 ToMi、Hi-ToM、BigToM、FANToM 等基准集用于评估。

**📈 对比分析**

与 ExploreToM、Llama‑3.1‑8B‑Base、Mistral‑NeMo‑12B、Phi‑3‑Medium‑14B、Qwen2.5‑14/32B、Gemma‑2‑27B 等模型在四个 ToM 基准上对比；OSCToM‑8B 在 ToMi 79.5%、Hi-ToM 65.3%、BigToM 89.8%、FANToM 76.0% 处获得最高分；推理平均时延 2.62 s，比 ExploreToM 的 15 s 降低 5.7×，并保持 𝒪(1) 复杂度。

**⚠️ 局限性**

1) DSL 仅覆盖位置、观察、通信、欺骗等行为，未涵盖情绪、记忆、意图等社会推理维度；2) 生成故事仅为文本，缺乏多模态输入；3) surrogate 评估依赖于 distillation，若标签质量偏差会误导生成；4) 目前未验证在更大模型或多模态场景下的通用性。

---

## 21. MedicalBench: Evaluating Large Language Models Toward Improved Medical Concept Extraction

**arXiv ID:** 2605.20197 | [PDF](https://arxiv.org/pdf/2605.20197v1)

**作者:** Zhichao Yang `[一作]` (Optum AI), Robert E. Tillman `[通讯]` (Optum AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了MedicalBench，提供在电子病历中隐式医学概念提取并结合文本证据的基准；

**💡 创新点**

首次系统化评估隐式医学推理与证据归属，强调难负样本、语义混淆及专家验证的证据跨度；

**🔧 技术方法**

利用多阶段LLM筛选管道、监督微调基准模型、提示式推理与证据检索技术；

**📊 数据集**

基于MIMIC‑IV住院出院摘要与ICD‑10诊断/手术码，最终包含823条高质量标注样本；

**📈 对比分析**

对比多种LLM与监督基线，最高F1仅0.59，证据检索能力与概念提取显著相关，提示式推理和隐式证据提取能提升约15‑20% F1；

**⚠️ 局限性**

模型整体表现仍低，难以捕捉隐式推理、时序关系和散布证据；样本规模有限，且高度依赖LLM预筛与专家审阅，泛化性待验证。

---

## 22. Rotatable Coupler Antenna Enhanced Wireless Network: Modeling and Coupler Rotation Optimization

**arXiv ID:** 2605.20535 | [PDF](https://arxiv.org/pdf/2605.20535v1)

**作者:** Xiaodan Shao `[一作]` (University of Waterloo), Xuemin Shen `[通讯]` (University of Waterloo)

**通讯引用:** 99431 | [OpenAlex ID](https://openalex.org/A5100773343)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种可旋转耦合器天线（RCA），通过仅在主动天线附近布置低成本被动耦合器并让其在三维空间自由旋转，以实现不增加射频链路的前端机械波束成形与频谱/能效提升；并提出基于球面上条件梯度与交叉熵方法的旋转优化算法，最大化接收信号噪声比。

**💡 创新点**

创新点在于：①首次将耦合器的三维方向旋转作为可调参数来主动重塑天线互耦与方向图，实现无需额外 RF 链的机械波束成形；②提出球面截头条件梯度（Spherical‑Cap Conditional‑Gradient）与交叉熵初始化相结合的全局优化框架，能够在非凸约束下高效寻找近似最优旋转配置；③通过多端口电路模型与方向图公式，定量描述旋转如何影响互耦与有效通道，从而实现精确的性能优化。

**🔧 技术方法**

采用多端口电路理论建模、方向图分析、互耦矩阵求解、球面截头条件梯度优化、交叉熵方法（CEM）离散搜索、Armijo 步长退火以及基于 MATLAB 的数值仿真。

**📊 数据集**

使用仿真数据：频率 7 GHz，λ ≈ 0.043 m，短线偶极子长度 0.5λ，半径 λ/500；天线配置 N = 3、N+1 个端口；信道采用 L = 6 条多径，随机均匀分布的角度；载波功率 30 dBm；负载阻抗 (0.05 + j50) Ω；最大旋转角 θ_max 设为 60°、175° 进行比较。

**📈 对比分析**

与三种基线方案对比：①全 RF 链天线阵列（ULA）；②固定旋转耦合器天线；③可移动耦合器天线（FCA）仅通过位置优化。评估指标为可达率 R( U ) = log₂(1 + SNR)。结果显示，RCA 在相同功率和硬件预算下，可达率均超过基线，尤其在多耦合器（N > 1）和宽旋转角（θ_max = 175°）时，收益显著。算法收敛快速，第一轮迭代即出现明显提升。

**⚠️ 局限性**

局限性包括：仅在点对点、单用户、静态信道下验证；未给出实际硬件实现与实时旋转控制的实验；算法复杂度随着耦合器数和搜索粒度上升；未处理多用户/多链路场景；未提出专门的通道估计与低开销旋转调度方法。

---

## 23. Shiny Stories, Hidden Struggles: Investigating the Representation of Disability Through the Lens of LLMs

**arXiv ID:** 2605.20191 | [PDF](https://arxiv.org/pdf/2605.20191v1)

**作者:** Marco Bombieri `[一作]` (University of Verona), Marco Rospocher `[通讯]` (University of Verona)

**通讯引用:** 1630 | [OpenAlex ID](https://openalex.org/A5043305308)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过让大型语言模型(LLM)模拟残障人士在社交媒体上的自我描述，并将生成的文本与真实残障人士的Reddit帖子进行对比，探究LLM对残障群体的表现是否存在理想化与偏见。

**💡 创新点**

创新点在于构建公开的LLM生成残障描述数据集，利用情感、情绪、抑郁检测与词频统计三维度，对比真实与生成文本，首次系统揭示LLM在残障表述上既过度正面又忽略真实挑战的双重偏差。

**🔧 技术方法**

采用多模型零射击生成（GPT‑4o‑mini、Gemini‑1.5F、Mixtral‑8x7B），结合VADER情感分析、NRC情绪词典、抑郁检测模型以及Fightin' Words词频统计和LLM驱动的主题聚类。

**📊 数据集**

使用两类数据集：一为从r/disability、r/blind、r/autism、r/depression、r/deaf、r/cerebralpalsy等六个Reddit子版块筛选的352条真实残障用户自述；二为三款LLM在36个“想象自己是残障/普通人”的提示下生成的1,080条人工文本。

**📈 对比分析**

比较方法包括情感比例、情绪词比例、抑郁标签分布、词汇z‑score差异及主题簇对比；结果显示LLM生成文本几乎全为正面情绪且抑郁标记极低，而真实文本则呈负面情绪占比高、抑郁标记显著；词汇差异体现LLM的“灵感色情”倾向。

**⚠️ 局限性**

局限性包括仅覆盖部分残障类型、依赖词典与统计方法忽略语境、未进行人类评估（尤其残障群体反馈）、缺乏与非残障Reddit内容的直接对比、实验仅针对特定LLM版本与英文，且存在隐私与伦理风险。

---

## 24. TelePhysics: Physics-Grounded Multi-Object Scene Generation from a Single Image with Real-Time Interaction

**arXiv ID:** 2605.20290 | [PDF](https://arxiv.org/pdf/2605.20290v1)

**作者:** Xin Zhang `[一作]` (Fudan University), Xuelong Li `[通讯]` (Institute of Artificial Intelligence, China Telecom (TeleAI))

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用单张图像构建完整的场景级3D模型，并通过物理仿真生成可控、符合物理规律的视频。

**💡 创新点**

提出训练‑free的统一框架，包括场景感知、锚点引导姿态对齐、粗细相结合的相机参数优化以及把物理仿真与Diffusion重渲染解耦，实现实时交互预览并保持照片真实感。

**🔧 技术方法**

结合SAM3/SAM-3D-Objects进行实例分割与三维重建；使用Scene‑Aware Pose Alignment与AGMF进行全局坐标对齐；通过Vision‑Language Model自动估计材质参数；使用Genesis多物理求解器（RBD、MPM、PBD）进行仿真；采用Wan2.1/2.2 VACE进行Diffusion重渲染；使用Qwen2.5‑VL‑72B估计材料；全流程无监督训练。

**📊 数据集**

在60个多对象场景的基准集（室内外、单体与多体交互）上进行评估；采用公开的SAM、Genesis、Wan等开源工具。

**📈 对比分析**

与Sora2-pro、Veo3.1、CogVideoX1.5、Wan2.2-A14B、WonderPlay、PhysCtrl等基线进行对比；在GPT‑5评估中SA、PC、VQ均超过对手，平均SA 3.29/5、PC 3.15/5、VQ 3.61/5；在人类评价中Borda得分6.93/7，显著优于其他方法。

**⚠️ 局限性**

受制于单目分割/重建误差；对高度关节化、透明或复杂材质的物体表现不佳；离线高质量重渲染耗时长；材质与接触属性需手工或启发式估计。

---

## 25. JUDO: A Juxtaposed Domain-Oriented Multimodal Reasoner for Industrial Anomaly QA

**arXiv ID:** 2605.20284 | [PDF](https://arxiv.org/pdf/2605.20284v1)

**作者:** Hyunju Kang `[一作]` (Sungkyunkwan University), Hogun Park `[通讯]` (Sungkyunkwan University)

**通讯引用:** 457 | [OpenAlex ID](https://openalex.org/A5071646906)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 JUDO 的三阶段训练框架，用以在工业异常检测中将领域知识与视觉对比推理相结合，实现更精准的缺陷定位、分类、描述和分析。

**💡 创新点**

创新点在于：①将正常图像作为训练时的视觉对比上下文，实现细粒度缺陷分割；②通过监督微调将领域文本知识内化到模型参数中；③设计多奖励的 GRPO 强化学习，使视觉证据与文本推理统一成域向的推理过程。

**🔧 技术方法**

使用了多模态预训练模型（Qwen2.5‑VL‑7B）+ 监督微调 (SFT) + 领域知识注入 + 视觉分割训练 + GRPO（Group Relative Policy Optimization）+ 定制的分割、推理、结构对齐奖励。

**📊 数据集**

主要数据集包括 MMAD benchmark（MVTec AD、MVTec LOCO、VisA、GoodsAD）、Real‑IAD 进行分割训练，以及从 MMAD 的领域描述生成的 QA 数据集。

**📈 对比分析**

与常规 LMM（Gemini、GPT‑4o、Qwen 等）和专注异常检测的 AnomalyR1 对比，JUDO 在 MMAD 的七项子任务中平均准确率达到 81.20%，在分类、定位、描述和分析等缺陷相关任务上均领先，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：在单纯的异常判别（二分类）任务上仍低于部分大型 LMM；引入推理模式后会略微降低判别准确率；整体仍受限于所使用的视觉编码器能力，且在对更复杂多样的工业场景泛化时需要进一步验证。

---

## 26. Hypergraph Partitioning on GPU with Distinct Incident Hyperedges and Size Constraints

**arXiv ID:** 2605.20497 | [PDF](https://arxiv.org/pdf/2605.20497v1)

**作者:** Marco Ronzani `[一作]` (Politecnico di Milano), Cristina Silvano `[通讯]` (Politecnico di Milano)

**通讯引用:** 3907 | [OpenAlex ID](https://openalex.org/A5031461662)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种GPU为中心的确定性多层超图划分器，能够同时满足分区大小和每个分区可接受的不同入射超边数量的约束。

**💡 创新点**

创新点包括：①基于GPU层级并行模型的候选配对与权重匹配算法，采用动态规划求解伪森林的最大权重匹配；②在划分和细化阶段使用事件驱动的约束校验，避免全量状态重建；③采用材料化邻域并集技术减少全局原子操作；④将传统多层划分策略转化为可并行的Warp级别实现。

**🔧 技术方法**

使用了CUDA并行编程、压缩稀疏存储（CSR/CSF）+哈希集合、Warp/线程级别协作、事件排序与前缀和、动态规划匹配、随机扰动去重、稀疏矩阵预处理等技术。

**📊 数据集**

主要数据集包括：从神经形态硬件映射获得的12个SNN超图（节点1k‑500k、边数数百万），VGG‑style ANN转化的SNN，以及扩充的ISPD98基准（k‑way平衡划分场景）。

**📈 对比分析**

与hMETIS、Mt‑KaHyPar（CPU）以及gHyPart（GPU）等现有工具比较，平均提升约380×速度（相较hMETIS），与Mt‑KaHyPar相比速度提升2–15×、连通度下降至0.64×；与gHyPart比较，速度接近或略快，连通度比其好约25%。

**⚠️ 局限性**

局限性：对超图稠密度与邻域不均衡的情况仍有内存和warp分歧的潜在开销；在k‑way平衡划分下切网质量仍略逊于最先进CPU方法；算法对输入预处理（如邻域材料化）依赖较高，需额外内存管理。

---

## 27. Why Latent Actions Fail, and How to Prevent It

**arXiv ID:** 2605.20223 | [PDF](https://arxiv.org/pdf/2605.20223v1)

**作者:** Jung Min Lee `[一作]` (Seoul National University), Jungwoo Lee `[通讯]` (Seoul National University)

**通讯引用:** 11285 | [OpenAlex ID](https://openalex.org/A5100376261)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文从理论上分析了无标签视频中潜在动作模型（LAM）在面对外生噪声时的失败机制，并给出了对应的补救策略；

**💡 创新点**

创新点在于将线性LAM扩展为Ex‑BMDP框架，证明重建目标会导致未来外生状态泄漏，并通过交叉外生重构（等价于CCA）和外生鲁棒目标预测两类辅助目标，理论上保证潜在动作在外生状态下的一致性；

**🔧 技术方法**

主要技术包括Ex‑BMDP建模、线性与非线性LAM的理论分析、交叉外生重构损失与外生鲁棒预测损失的定义与证明，以及利用VQ‑VAE + UNet实现的实践LAM；

**📊 数据集**

实验数据集涵盖合成的线性与非线性环境（4×4格子世界）、Bridge V2、RT‑1以及Distracting Control Suite；

**📈 对比分析**

与仅使用标准重建损失的LAM对比，补救策略在动作对齐指标（线性探针NMSE、动作验证NMSE）上均提升，且在不同噪声水平下保持更高的一致性与更低的外生区域误差；

**⚠️ 局限性**

局限性主要在于理论分析依赖线性假设和简化的外生噪声模型，实际LAM与连续控制任务中的复杂动力学仍需进一步验证与推广。

---

## 28. Open-World Evaluations for Measuring Frontier AI Capabilities

**arXiv ID:** 2605.20520 | [PDF](https://arxiv.org/pdf/2605.20520v1)

**作者:** Sayash Kapoor `[一作]` (Princeton University), Arvind Narayanan `[通讯]` (Princeton University)

**通讯引用:** 18699 | [OpenAlex ID](https://openalex.org/A5058102069)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了开放世界评估（open‑world evaluation）范式，并基于此框架CRUX进行首次实验，即让AI代理自主完成iOS应用开发并提交至App Store；

**💡 创新点**

创新点在于：①将开放世界评估系统化并提出一系列方法学规范（如日志公开、干预记录、成本报告等）；②首次展示代理在真实应用商店提交流程中的高成功率；

**🔧 技术方法**

使用技术包括Claude Opus 4.6（具自适应思维）配合OpenClaw可视化与浏览器交互框架、macOS虚拟机、GitHub、Apple Developer账户等；

**📊 数据集**

数据集：无传统离散数据集，评估基于真实的iOS开发与App Store提交流程，使用实际的App Store表单与审核机制；

**📈 对比分析**

与传统基准相比，实验未采用对照模型，仅记录代理成功率与人机干预次数；结果显示代理成功提交应用，所需人工干预仅5次（其中4次为苹果政策限制），总成本约$1000，开发成本仅$25；

**⚠️ 局限性**

局限性包括：单一任务、单一模型、缺乏可重复性与可比性、需领域专家评审、成功判定模糊、环境非稳定（网络、审核政策变化）以及输出质量仍有缺陷。

---

## 29. Data Scaling as Progressive Coverage of a Predictive Contribution Spectrum

**arXiv ID:** 2605.20196 | [PDF](https://arxiv.org/pdf/2605.20196v1)

**作者:** Zihui Song `[一作]`, Chunlin Huang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用后缀自动机构建文本语料的全局 KL 预测贡献谱，研究该谱的尾部几何与数据规模学习曲线的关联，并通过匹配余量误差定义有效截断阶数 K(N)，揭示训练规模可视为在谱中前沿推进的机制。

**💡 创新点**

提出以后缀自动机状态为单位、按状态频率乘以 KL 偏差权重得到的预测贡献谱，并证明其尾部斜率与不同数据集的学习曲线斜率高度相关；进一步通过余量误差匹配得到 K(N) 与 N 的对数线性关系，提供了比传统 token 频率尾部更直接的机制解释。

**🔧 技术方法**

后缀自动机（SAM）构建、全局 KL 预测贡献谱计算、余量误差匹配的截断阈值 K(N) 定义、对数线性拟合、R² 评估。

**📊 数据集**

12 个真实语料库，包括百科、叙事、评论、分类、摘要等多域数据（如 AG News、Yelp、Amazon Polarity、DBPedia、IMDb、Tweet Sentiment、CNN/DailyMail、Yahoo Answers、XSum、BillSum、TinyStories、WikiText-103）。

**📈 对比分析**

与传统的 token 统计（熵、压缩比、单词尾部斜率、n-gram 简介）等单一指标对比，全球 KL 谱在 1000k 训练集上对数据规模斜率的拟合 R² 最高（≈0.98），而有效截断阶数 K(N) 的对数线性拟合 R² 也很高（聚合 R² ≈0.96）。

**⚠️ 局限性**

定义的截断阈值受端点锚定影响（最小 N 对应 K≈1，最大 N 接近尾部长度）；对误差的归一化在不同数据集间可变，可能影响跨集比较；预测贡献谱仅为可操作的代理，未证明其为唯一真实的可学习模式。

---

## 30. Adaptive Human-Robot Collaboration for Masonry Construction Under Material and Assembly Uncertainty

**arXiv ID:** 2605.20264 | [PDF](https://arxiv.org/pdf/2605.20264v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 31. TabPFN-MT: A Natively Multitask In-Context Learner for Tabular Data

**arXiv ID:** 2605.20234 | [PDF](https://arxiv.org/pdf/2605.20234v1)

**作者:** Cormac Cureton `[一作]` (McGill University), Narges Armanfard `[通讯]` (McGill University)

**通讯引用:** 1355 | [OpenAlex ID](https://openalex.org/A5073955046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并训练了TabPFN-MT，一种面向多目标任务的表格基础模型，利用单次前向推理完成多任务预测；

**💡 创新点**

在TabPFN框架中引入了可变维度的y编码器和共享解码头，并通过单一复杂结构因果模型生成多目标合成先验，显著提升了多任务上下文学习能力；

**🔧 技术方法**

使用Transformer自注意力、扩展的目标编码器、共享MLP解码器以及基于结构因果模型的合成数据预训练；

**📊 数据集**

在344个低至中等样本规模（≤5000样本）的真实世界数据集上进行评估，涵盖多目标分类、从单目标数据重新构造的多目标、以及子采样的大型数据集；

**📈 对比分析**

与单任务基线（GBDT、XGBoost、CatBoost、TabPFN等）以及多任务深度学习模型（MMOE、PLE、STEM、MultiTab）进行对比，TabPFN‑MT在Accuracy、F1、ROC AUC等指标上位居所有多任务模型之首，并在总体平均排名中与最强单任务集成相当；

**⚠️ 局限性**

受限于最大5个任务和10个类别的设计，尚未覆盖回归任务；预训练阶段仍需一次性计算资源；在任务不相关或高度不平衡的情况下，信息瓶颈可能导致性能下降。

---

## 32. Agentic Agile-V: From Vibe Coding to Verified Engineering in Software and Hardware Development

**arXiv ID:** 2605.20456 | [PDF](https://arxiv.org/pdf/2605.20456v1)

**作者:** Christopher Koch `[一作]` `[通讯]` (Independent Researcher), Christopher Koch (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

系统化梳理Agentic AI编码与硬件开发的实证证据，提出了Agentic Agile‑V流程框架，并设计了六步任务循环（Specify‑Constrain‑Orchestrate‑Prove‑Evolve‑Verify）以及风险适配的验收门控与证据包。

**💡 创新点**

创新点包括：①将对话式探索与结构化执行切分，形成对话‑合同门控；②将敏捷迭代与V模型验证相结合，形成轻量化的Agile‑V生命周期；③提出基于风险级别的多层验收门控和证据包规范；④给出最小输入工件模型与任务模板，明确Agent应具备的执行语境。

**🔧 技术方法**

采用大语言模型驱动的Agentic编码系统（如OpenHands、GitHub Copilot等）、多工具沙箱、自动化测试与静态分析、硬件仿真/形式验证、以及数据驱动的过程管理工具；框架本身以流程模型、任务循环和证据打包实现。

**📊 数据集**

使用的主要数据集包括：AIDev（约93万条Agent生成的PR）、RealBench（Verilog硬件生成与验证基准）、METR与Google RCT数据、GitHub规模的配置与执行案例（如Claude Code、GitHub Copilot等）。

**📈 对比分析**

比较方法为“有限证据综合”而非传统元分析；通过对RCT、GitHub PR接受率、配置文件影响、硬件验证通过率等多维度指标的对比，得出Agentic AI在不同上下文中对生产力的“中等、异质性”影响；并通过实验和案例展示风险门控能显著降低验证债务，提升代码质量。

**⚠️ 局限性**

局限性包括：①研究基于现有实证数据，缺乏大规模实验验证框架有效性；②硬件/固件生成的低通过率表明当前LLM仍难满足系统级验证需求；③不同工具和任务类型的异质性导致框架通用性需进一步评估；④框架对流程配置和人工审批的依赖可能降低部署速度。

---

## 33. A Comprehensive Comparison of Deep Learning Architectures for COVID-19 Classification on CT & X-ray Imagery

**arXiv ID:** 2605.20445 | [PDF](https://arxiv.org/pdf/2605.20445v1)

**作者:** Sarmad Khan `[一作]` (National University of Sciences and Technology), Basim Azam `[通讯]` (University of Melbourne)

**通讯引用:** 103 | [OpenAlex ID](https://openalex.org/A5002895693)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

对公开的CT与X光影像数据集进行COVID‑19与正常/肺炎病例的二分类与三分类实验，系统评估16种预训练CNN模型的性能。

**💡 创新点**

首次在同一实验框架下对多种主流深度学习架构进行全面对比，并给出基准性能指标，为后续模型改进提供参考。

**🔧 技术方法**

采用迁移学习、数据增强（旋转、翻转、缩放、平移、剪切）以及Adam优化器训练预训练网络（VGG、ResNet、DenseNet、Xception、Inception、EfficientNet、NasNet等）。

**📊 数据集**

使用四个公开数据集：SARS‑CoV‑II（CT），COVID‑CT，IEEE‑8023（X光），Radiography Database（X光多分类）。

**📈 对比分析**

通过准确率、精确率、召回率和F1分数等指标进行比较，结果显示ResNet系列模型在二分类中可达96–98%准确率，三分类中最高95%准确率，整体优于现有方法。

**⚠️ 局限性**

局限性包括图像分辨率较低、样本不平衡导致的过拟合风险、缺乏跨机构外部验证以及缺乏可解释性分析。

---

## 34. FBOS-RL: Feedback-Driven Bi-Objective Synergistic Reinforcement Learning

**arXiv ID:** 2605.20256 | [PDF](https://arxiv.org/pdf/2605.20256v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 35. Physics-informed convolutional neural networks for fluid flow through porous media

**arXiv ID:** 2605.20250 | [PDF](https://arxiv.org/pdf/2605.20250v1)

**作者:** Rafał Topolnicki `[一作]` (Polish Academy of Sciences), Maciej Matyka `[通讯]` (University of Wrocław)

**通讯引用:** 1494 | [OpenAlex ID](https://openalex.org/A5033959580)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于物理信息卷积神经网络的框架，利用孔隙结构图像直接预测孔隙尺度流速场，并将网络预测结果作为 Lattice Boltzmann 方法（LBM）的预热初值，显著加速数值求解。

**💡 创新点**

创新点在于：①构造了多项物理约束的自定义损失函数（包括速度误差、固体内无流、不可压、周期一致性和色散度匹配），通过合理加权显著提升物理一致性；②系统评估了多种 CNN 主干网络（VGG、ResNet、DenseNet、ConvNeXt、EfficientNet、MobileNetV3）在同一任务中的性能差异；③展示网络预热在不同孔隙率、几何形状和边界条件下对 LBM 收敛速度的实际提升。

**🔧 技术方法**

采用了卷积编码-解码结构（U‑Net）结合多种主干网络、数据增强（翻转、滚动）以及 Adam 优化器；损失函数包含速度 MSE、固体无流 L1、不可压梯度平方、周期一致性差异以及色散度误差；模型训练采用两阶段策略，最终使用 α=5、β=1、γ=0.1、δ=0.01。

**📊 数据集**

训练集为 10,000 张 256×256 二值孔隙图像，使用随机三角波叠加生成，孔隙率在 0.70–0.95 之间；此外对比测试还使用了圆/方障碍、管道边界、低孔隙率样本以及真实 Li‑O₂ 电极截面（DRP‑1129v2）来评估泛化。

**📈 对比分析**

通过多种指标（速度 RMSE、色散度 MAPE、R²、渗透率 RMSE/MAPE/R²）对比，ResNet‑101 在速度 RMSE 5.1×10⁻³、色散度 R² 0.983、渗透率 R² 0.997 处表现最佳；在 LBM 预热实验中，90% 的样本收敛速度提升约 50%（显著性 p≈4×10⁻⁷）。

**⚠️ 局限性**

主要局限包括：仅针对二维固定分辨率孔隙结构；在低孔隙率接近渗透阈值时性能明显下降；物理约束仅为惩罚项，无法完全强制满足偏微分方程；对三维结构、异构多尺度孔隙网络的直接迁移尚需进一步研究。

---

## 36. Consistently Informative Soft-Label Temperature for Knowledge Distillation

**arXiv ID:** 2605.20357 | [PDF](https://arxiv.org/pdf/2605.20357v1)

**作者:** Hoang-Chau Luong `[一作]` (Rochester Institute of Technology), Lingwei Chen `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1067 | [OpenAlex ID](https://openalex.org/A5017524689)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CIST框架，对知识蒸馏中的教师和学生分别采用样本级自适应温度，并加入教师置信度与学生学习难度的加权，改进传统固定温度的KD。

**💡 创新点**

创新点在于：①对教师和学生分别设置样本自适应温度，消除固定温度导致的软标签熵不一致和严格的logit尺度匹配；②引入基于教师置信度与学生难度的加权，形成自适应的蒸馏课程。

**🔧 技术方法**

技术实现包括：熵分析指导温度自适应；独立教师/学生温度；KL重加权与温度乘积的自适应梯度调节；在实验中使用标准KL蒸馏损失与自定义加权。

**📊 数据集**

在视觉任务上使用CIFAR‑100和ImageNet数据集；在语言任务上使用GPT‑2/OPT教师对GPT‑2/OPT学生的指令‑响应数据集进行蒸馏。

**📈 对比分析**

与标准KD、DKD、CTKD、MLKD、FitNet、RKD、CRD、OFD、ReviewKD等方法对比，CIST在CIFAR‑100、ImageNet和指令‑跟随评测中分别提升约0.3%–3.6%准确率或ROUGE‑L分数，显著优于基线且计算开销几乎相同。

**⚠️ 局限性**

局限性包括：需要手动调节超参数ρ和KL权重λ_KL；在极端教师/学生容量差距下效果仍需进一步验证；对非分类任务或更大规模模型的通用性尚未充分评估。

---

## 37. Challenges in Working Towards Patient Engagement in Developing Technology Prototypes

**arXiv ID:** 2605.20205 | [PDF](https://arxiv.org/pdf/2605.20205v1)

**作者:** Fateme Rajabiyazdi `[一作]` (University of Calgary), Sheelagh Carpendale `[通讯]` (Simon Fraser University)

**通讯引用:** 11959 | [OpenAlex ID](https://openalex.org/A5012561411)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在两个月的试点中实现并评估了面向多慢性病患者的数字健康干预平台 MyCareCompass，收集了使用日志和反馈来分析患者的参与度。

**💡 创新点**

创新点在于提出了三条可持续参与的关键经验：MVP 与患者复杂需求的不匹配、实现过程中的患者中心优先级漂移，以及跨条件解释负担导致的脱离，为复杂慢性病护理工具的设计提供了新框架。

**🔧 技术方法**

研究采用了以人为中心的设计方法（访谈、焦点小组、旅程映射、原型与可用性测试），技术实现上为简化的移动/网页平台数据跟踪与可视化，但 MVP 在功能上被大幅简化。

**📊 数据集**

数据集仅包含7名多慢性病患者在试点期间产生的使用日志、登录频率、功能使用数据以及后续自评问卷，未使用公开大规模数据集。

**📈 对比分析**

通过混合方法比较—客观使用分析与主观感知—发现参与度低，登录频率和功能使用率远低于预期，说明 MVP 在实际场景中的可持续性不足。

**⚠️ 局限性**

局限性包括样本规模小、仅在 COVID-19 期间进行、MVP 功能过于简化导致外部有效性受限，并且未能充分验证跨条件解释工作量的测评方法。

---

## 38. Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning

**arXiv ID:** 2605.20609 | [PDF](https://arxiv.org/pdf/2605.20609v1)

**作者:** Junseok Kim `[一作]` (Seoul National University), Songhwai Oh `[通讯]` (Seoul National University)

**通讯引用:** 3821 | [OpenAlex ID](https://openalex.org/A5033764106)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在离线目标条件强化学习中通过类比转导实现组合泛化的新方法

**💡 创新点**

1）将任务内在类比定义为最优时间距离差场，理论证明其不变性与充分性；2）提出可实现的双重类比表征；3）设计CTA框架通过双线性转导实现OOC（组合外）类比推断

**🔧 技术方法**

利用最优时间距离学习（IQL）、双线性转导、层级策略（高层生成k步类比，低层执行原子动作）以及时间距离差场的低维近似

**📊 数据集**

OGBench离线目标条件强化学习基准中的8个操纵任务（需要方块重排、顺序交互、按钮组合）

**📈 对比分析**

与多种基线（无表征、双重目标表征、层级IQL等）对比，CTA在6/8个任务中排名第一或第二，平均性能提升约42%（在4个组合复杂任务中提升约40%），并在OOC案例中显著优于基线

**⚠️ 局限性**

1）假设任务内在与外在成分可分离，可能在实际环境中失效；2）双重类比虽近似理论类比，但实现上仍有差距，缺乏严格不变性保证；3）在任务外在变异较少的环境中收益有限

---

## 39. Symmetrization of Loss Functions for Robust Training of Neural Networks in the Presence of Noisy Labels

**arXiv ID:** 2605.20347 | [PDF](https://arxiv.org/pdf/2605.20347v1)

**作者:** Alexandre Lemire Paquin `[一作]` (Université Laval), Philippe Giguère `[通讯]` (Université Laval)

**通讯引用:** 2553 | [OpenAlex ID](https://openalex.org/A5032902130)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出对多分类损失函数进行对称化处理，推出唯一的凸多分类对称损失（多类unhinged），并基于它设计了 SGCE 与 α-MAE 两种新型鲁棒损失。

**💡 创新点**

创新点在于证明多类 unhinged 为唯一满足凸性、非递增、对称性和置换不变性的多分类损失，并通过对 GCE 的对称化得到可调节的 SGCE，进一步将其与 MAE 结合得到可调节的 α-MAE，提供了从硬标签到软标签之间的连续平衡。

**🔧 技术方法**

使用了对称化分解（symmetrization decomposition）、梯度平滑化（β-smoothness）和批归一化/欧式归一化等技术，以及常见的 SGD、cosine annealing 训练策略。

**📊 数据集**

在 CIFAR-10、CIFAR-100、CIFAR-10N、CIFAR-100N 以及 WebVision 等公开基准数据集上进行实验。

**📈 对比分析**

与 CE、MAE、GCE、SCE、NCE+RCE、NCE+AGCE、ANL-CE 等现有鲁棒损失进行对比，SGCE 与 α-MAE 在对称噪声、非对称噪声和自然噪声场景下均能保持或超过前者的性能，尤其在 CIFAR-10N/CIFAR-100N 与 WebVision 上表现尤为突出。

**⚠️ 局限性**

局限性包括：对 α 和 q 等超参数的敏感性、对极端高噪声率下的欠拟合风险、以及在某些实验中仍需手动调节权重衰减或归一化策略。

---

## 40. FullFlow: Upgrading Text-to-Image Flow Matching Models for Bidirectional Vision--Language Generation

**arXiv ID:** 2605.20316 | [PDF](https://arxiv.org/pdf/2605.20316v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 41. Framing an AI with Values Reduces AI Reliance in AI-supported Writing Tasks

**arXiv ID:** 2605.20512 | [PDF](https://arxiv.org/pdf/2605.20512v1)

**作者:** Alice Gao `[一作]` (University of Washington), Katharina Reinecke `[通讯]` (University of Washington)

**通讯引用:** 3622 | [OpenAlex ID](https://openalex.org/A5076852435)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项在线实验中，研究者通过展示AI系统的价值取向概览，探究其是否能降低用户对大型语言模型（LLM）写作建议的依赖，并提升写作的独创性。

**💡 创新点**

创新点在于提出并验证了“AI价值框架”这一极简介入方式——仅通过一张条形图概述AI的价值取向，即可显著减少用户对AI输出的接受比例（约20%）并提升文本多样性。

**🔧 技术方法**

采用的技术主要包括：对OpenAI GPT-4的自动补全生成、实验平台SvelteKit搭建、交互日志与文本质量指标（AI依赖率、词汇多样性、余弦相似度）以及主题分析。

**📊 数据集**

使用的数据集为149名来自印度与美国的参与者（共4组实验条件），以及OpenAI模型提供的写作提示与自动补全结果。

**📈 对比分析**

对照组（无介入）与三种介入组（仅展示AI价值、先问价值再展示、比较展示）进行配对t检验和方差分析，结果显示非比较性价值展示组在AI依赖率下降最高（约30%），而文本独创性（余弦相似度）亦显著降低，表明介入有效。

**⚠️ 局限性**

局限性包括：样本仅来自两国且来源于Prolific平台，价值概览形式过于简化（条形图），未考虑用户对AI的先前认知或任务相关上下文，且展示的价值并不必然与AI真实行为相符，未能解决AI本身的偏见问题。

---

## 42. Improving Quantized Model Performance in Qualitative Analysis with Multi-Pass Prompt Verification

**arXiv ID:** 2605.20193 | [PDF](https://arxiv.org/pdf/2605.20193v1)

**作者:** Aisvarya Adeseye `[一作]` (University of Turku), Adeyemi Adeseye `[通讯]` (Brilloconnetz Partners avoin yhtiö)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在定性分析中使用低比特量化的 LLaMA-3.1 (8B) 模型，并提出多步提示验证框架以提升主题提取、频率统计的准确性。

**💡 创新点**

①系统性评估 8、4、3、2 位量化对 LLM 主题抽取性能的影响；②设计量化感知的多重验证流程，显著降低幻觉与语义漂移；③在不增大模型尺寸的前提下，使 4‑bit 量化模型达到接近 8‑bit 的实用水平。

**🔧 技术方法**

量化方法包括 SmoothQuant、INT8、GPTQ、AWQ、HQQ、SPQR、GGUF；多步提示验证结合结构化系统提示、JSON 输出约束、主题/频率验证；评估使用 NVivo 人工编码、BF16 LLaMA 作为对照，计算 F1、SDS、HR、TCS、Freq、KOR、KHR、ARI 等指标。

**📊 数据集**

82 份半结构化访谈转录（8k–13k 词），专家 33 人、非专家 49 人，围绕组织内游戏化与隐私议题。

**📈 对比分析**

通过对比人类编码、BF16 结果以及各量化模型在验证前后的指标，发现 4‑bit+验证后 F1、频率相关性、ARI 与 8‑bit/人类接近，验证前 3/2‑bit 模型表现严重下降，验证后提升 20–40 点；整体误差率降低 45% 以上。

**⚠️ 局限性**

仅评估 LLaMA‑3.1(8B)；数据集单一领域；未检验其他 LLM 或混合精度方案；验证过程增加约 34% 运行时间；极低精度下模型仍有一定局限。

---

## 43. HADS-Net:A Hybrid Attention-Augmented Dual-Stream Network with Physics-Informed Augmentation for Breast Ultrasound Image Classification

**arXiv ID:** 2605.20536 | [PDF](https://arxiv.org/pdf/2605.20536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 44. Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning

**arXiv ID:** 2605.20201 | [PDF](https://arxiv.org/pdf/2605.20201v1)

**作者:** Miao Li `[一作]` (University of Edinburgh), Mirella Lapata `[通讯]` (University of Edinburgh)

**通讯引用:** 27992 | [OpenAlex ID](https://openalex.org/A5041024491)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ProxyCoT 框架，利用短代理上下文训练模型以提升长上下文推理能力。

**💡 创新点**

创新点在于将代理上下文用于两阶段训练：先在短代理上生成高质量链式推理，再通过监督微调将其迁移到完整长上下文。

**🔧 技术方法**

采用了强化学习与可验证奖励生成推理轨迹、链式推理蒸馏、监督微调等技术。

**📊 数据集**

在 SciTrek、HotpotQA 长文本问答基准以及 Loong 跨域长文本问答数据集上进行实验。

**📈 对比分析**

与零样本、纯 SFT、RL、链式推理蒸馏等基线相比，ProxyCoT 在完整上下文上准确率提升约 10–15%，且推理长度显著减少。

**⚠️ 局限性**

局限性包括需要手工或高质量标注的代理上下文，难以自动生成；实验仅限英语且受限于可获取的数据集。

---

## 45. Complementing reinforcement learning with SFT through logit averaging in the post training of LLMs

**arXiv ID:** 2605.20555 | [PDF](https://arxiv.org/pdf/2605.20555v1)

**作者:** Xingwei Gan `[一作]` (University Of California San Diego), Ying Zhu `[通讯]` (University Of California San Diego)

**通讯引用:** 9891 | [OpenAlex ID](https://openalex.org/A5006092446)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在对抗强化学习中把参考策略（如SFT）的logits与可训练策略的logits按权重平均，构造混合策略并用于GRPO训练，省去KL正则与评论家。

**💡 创新点**

不使用KL正则，而是直接在logit空间进行几何平均，形成类似产品专家的混合策略，既保持了SFT的格式优势，又提升了推理能力。

**🔧 技术方法**

logit averaging、GRPO、固定/自适应权重策略、与概率平均对比、产品专家与信赖区间分析、RL训练与奖励函数设计。

**📊 数据集**

MATH、cn‑k12（NuminaMath‑1.5）、MMLU；模型：Qwen2.5‑Instruct 1.5B/3B/7B；SFT作为参考策略。

**📈 对比分析**

与标准KL‑regularized GRPO、纯RL、概率平均混合比较。固定权重与自适应权重在三大数据集上表现相近，logit平均往往比对照组更高或相当；在logit vs. probability 实验中，logit平均在α≈0.5处达到最高解答率（约55%），明显优于概率平均。

**⚠️ 局限性**

依赖参考策略的格式化优势，若参考与训练策略缺乏互补性则效果可能不佳；自适应权重需要额外验证集，可能导致过拟合；在更大规模模型或不同领域的验证尚未充分，需进一步研究。

---

## 46. Co-Fusion4D: Spatio-temporal Collaborative Fusion for Robust 3D Object Detection

**arXiv ID:** 2605.20301 | [PDF](https://arxiv.org/pdf/2605.20301v1)

**作者:** Wenxuan Li `[一作]`, Qingxiang Meng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Co‑Fusion4D框架，基于BEV的多模态时空融合3D目标检测；

**💡 创新点**

创新点在于：① 当前帧为中心的非对称融合策略，减少累计对齐误差；② Dual Attention Fusion（DAF）模块同时捕捉帧内空间注意和帧间时间注意；③ 四阶段训练方案分离单帧、跨模态和跨帧学习，提升稳定性与性能；

**🔧 技术方法**

采用LiDAR点云与多视角相机图像的多模态BEV特征提取、几何对齐（点云对齐、BEV特征对齐）、DAF融合、Co‑Fix3D检测头及Transformer；

**📊 数据集**

使用公开nuScenes数据集进行训练与评估；

**📈 对比分析**

相较于现有方法，Co‑Fusion4D在nuScenes测试集上取得mAP 74.9%，NDS 75.6%，超过Co‑Fix3D、BEVFusion4D、GAFusion等SOTA；

**⚠️ 局限性**

局限在：对齐误差仍可能影响长时序；仅使用3帧时序，窗口扩展收益有限；计算开销主要来自多帧视角变换与图像backbone，推理延迟随帧数线性增长。

---

## 47. Terrestrial Soft Mobile Robots: A Review

**arXiv ID:** 2605.20304 | [PDF](https://arxiv.org/pdf/2605.20304v1)

**作者:** Dimuthu D. K. Arachchige `[一作]` (DePaul University), Dimuthu D. K. Arachchige `[通讯]` (DePaul University)

**通讯引用:** 406 | [OpenAlex ID](https://openalex.org/A5008534789)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了无轮陆地软体移动机器人（wheelless terrestrial soft mobile robots）的研究进展，涵盖了软限腿与软无腿的运动方式、驱动技术、建模方法、轨迹生成与控制策略，并对未来研究方向与挑战进行系统性讨论。

**💡 创新点**

创新点在于：①对软体机器人运动学与动力学的全新分类体系（软限腿按腿数分组、软无腿按运动模式划分）；②整合多种驱动技术（气动、电机、智能材料、磁性、热/化学等）与建模框架（连续介质、几何、离散、简化、定制）；③提出轨迹生成与控制的四大范式（仿生、基于模型、学习型、经验试验）；④以 1996‑2024 年的文献计量为依据，绘制技术发展与研究热点的时间线与性能对比。

**🔧 技术方法**

使用的技术与方法主要为：文献检索与系统综述、分门别类的框架构建、表格与图表对比、对已有实验数据与指标（如速度、转向、能耗、承载力）的归纳与对照。

**📊 数据集**

数据来源为 1996 年至 2024 年间发表的软体机器人相关论文与专利，作者自行整理并标注在各表格与图表中。

**📈 对比分析**

比较方法：通过表格（如“研究、运动方式、驱动、建模、轨迹、控制、能量/速度”等）对不同工作进行对齐；性能方面主要归纳各工作报告的速度（m/s、BL/s）、转向稳定性、能耗与实现复杂度等指标，呈现出气动驱动在速度与多功能性上优于其他驱动，仿生控制在复杂地形中的适应性最强。

**⚠️ 局限性**

局限性：①综述性质未提供统一实验平台与标准基准，导致跨工作性能对比受限；②软体机器人的设计、制造与建模高度依赖材料与工艺，仍缺乏可复制的通用方法；③高维非线性动力学与传感反馈难以实现实时闭环控制；④多模态驱动与控制融合尚不成熟，能量效率与可靠性有待提升。

---

## 48. SMA-DP: Spectral Memory-Aware Differential Privacy for Deep Learning

**arXiv ID:** 2605.20450 | [PDF](https://arxiv.org/pdf/2605.20450v1)

**作者:** Mohammad Partohaghighi `[一作]` (University of California Merced), Roummel Marcia `[通讯]` (University of California Merced)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Spectral Memory‑Aware DP‑SGD（SMA‑DP‑SGD），在保持差分隐私的前提下，利用仅包含已发布噪声梯度的分形记忆并以层级谱可靠性调节记忆衰减，实现更稳定的私有优化。

**💡 创新点**

创新点在于：① 仅基于已发布的私有梯度构造记忆分形核，避免原始梯度泄漏；② 结合 WeightWatcher 谱诊断为每层提供可靠性信号，动态调节记忆衰减；③ 设计可回滚到标准 DP‑SGD 的混合机制，保持清晰的条件敏感性。

**🔧 技术方法**

使用分形记忆核、WeightWatcher/Heavy‑Tailed Self‑Regularization 谱诊断、私有历史对齐、范数匹配、warm‑up 激活，以及组级差分隐私核算。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100、MNIST 三个图像分类数据集上进行实验。

**📈 对比分析**

与 DP‑SGD、DP‑Adam、DP‑AdamW、DP‑IS、DP‑SAM、DP‑SAT、DP‑Adam‑AC 等基线在同等实验设置下对比，SMA‑DP‑SGD 在 CIFAR‑100 与 CIFAR‑10 上均取得最优或接近最优的最终准确率，在更具挑战的数据集上优于其他方法。

**⚠️ 局限性**

限制包括：记忆分支会携带历史噪声；谱可靠性区间是经验可调的超参数；额外的谱诊断与记忆管理导致计算开销增加；全步隐私保证需使用保守联合核算，边际隐私成本仅为诊断。

---

## 49. Artificial Pancreas Implantables -- How Healthcare Professionals May Deal With DIY Bio Cases

**arXiv ID:** 2605.20208 | [PDF](https://arxiv.org/pdf/2605.20208v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 50. Security Document Classification with a Fine-Tuned Local Large Language Model: Benchmark Data and an Open-Source System

**arXiv ID:** 2605.20368 | [PDF](https://arxiv.org/pdf/2605.20368v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 51. Deep Learning Surrogates for Emulating Stochastic Climate Tipping Dynamics

**arXiv ID:** 2605.20580 | [PDF](https://arxiv.org/pdf/2605.20580v1)

**作者:** Adeline Hillier `[一作]` (Johns Hopkins Applied Physics Laboratory), Anand Gnanadesikan `[通讯]` (Johns Hopkins University)

**通讯引用:** 17065 | [OpenAlex ID](https://openalex.org/A5024290744)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了基于动态信息的Temporal Fusion Transformer（TFT）代理，用于快速模拟海洋箱模型的多变量时序数据，准确预测AMOC和PMOC的崩塌事件。

**💡 创新点**

创新点包括：1) 将静态参数与时间变异协变量统一编码；2) 去除自注意力层以降低计算成本；3) 引入软DTW损失以对齐相位误差；4) 将随机淡水扰动作为未来已知协变量；5) 使代理可微分，便于梯度推理。

**🔧 技术方法**

采用改进的TFT架构（LSTM编码器-解码器、门控残差网络、变量选择网络），结合软DTW损失、时间扭曲训练和自回归序列到序列推断。

**📊 数据集**

使用4箱与6箱简化箱模型生成的合成时序数据；4箱数据约6,000训练/401测试/401验证样本，6箱数据约1.4百万模拟（24,909个崩塌样本用于训练，655/656测试/验证）。

**📈 对比分析**

与SegRNN、iTransformer、TiDE、Mamba、TimeXer等五种基准模型对比，评估RMSE、soft-DTW、崩塌时间相关性等指标；改进TFT在自回归推断中表现最优，崩塌检测率≈98–99%，相关系数≥0.93，并实现与数值模拟器相比约465倍的速度提升。

**⚠️ 局限性**

局限性包括：仅针对简化箱模型，无法对更复杂的全球耦合模型直接适用；受混沌敏感性限制，预测精度随初始误差指数增长；需要大量仿真数据训练，且对超出训练参数空间的外推性能不确定。

---

## 52. OpenSeisML: Open Large-Scale Real Seismic and well-log Dataset for Generative AI

**arXiv ID:** 2605.20539 | [PDF](https://arxiv.org/pdf/2605.20539v1)

**作者:** Ipsita Bhar `[一作]` (Georgia Institute of Technology), Felix J. Herrmann `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7767 | [OpenAlex ID](https://openalex.org/A5010780250)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了基于英国NDR真实地震与测井数据的OpenSeisML数据集及自动化处理管线，支持生成式AI在地震反演中的训练。

**💡 创新点**

创新点在于将真实测井时间-深度关系与平均速度场插值结合，实现可重复、可扩展的时间-深度转换，并提供多样化真实地下结构数据。

**🔧 技术方法**

采用检查射线插值的RBF方法、OpendTect时间-深度转换、FFT重采样以及扩散模型生成技术。

**📊 数据集**

使用英国NDR公开的多份地震卷和测井（LAS、checkshot）数据，涵盖北海盆地多种海洋环境。

**📈 对比分析**

通过与Compass数据集对比的扩散生成实验，显示生成的速度模型与真实模型高度一致，表明方法可在有限井数下实现良好泛化，但未公布具体数值指标。

**⚠️ 局限性**

局限在于仅使用二维切片验证，速度场为平均场而非间隔速度，井数受限，三维推广仍待进一步研究。

---

## 53. Automated Kernel Discovery Towards Understanding High-dimensional Bayesian Optimization

**arXiv ID:** 2605.20249 | [PDF](https://arxiv.org/pdf/2605.20249v1)

**作者:** Taeyoung Yun `[一作]` (Korea Advanced Institute of Science and Technology), Jinkyoo Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2990 | [OpenAlex ID](https://openalex.org/A5023509025)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于大型语言模型的进化框架，自动在高维贝叶斯优化中发现并验证高效的高斯过程核；

**💡 创新点**

创新点包括：①两阶段LLM生成（先数学表达再代码），避免了直接代码生成导致的功能冗余；②不需要以观测为上下文，突破了高维数据的上下文长度瓶颈；③扩展核搜索空间至非加乘组合，并引入留一连续排名概率分数（LOO-CRPS）作为更稳健的核选择指标；

**🔧 技术方法**

使用技术包括：GPT-4o等大型语言模型、进化算法框架、GP核构造与正定性检验、留一CRPS评估、传统贝叶斯优化组件（EI、Acquisition优化器）等；

**📊 数据集**

实验基准涵盖五个高维任务：Rover (D=100)、Mopta08 (D=124)、Lasso-DNA (D=180)、SVM (D=388)、Humanoid (D=6392)；

**📈 对比分析**

与17个基线（基础核、搜索式、LLM式）对比，平均排名1.2/17，显著优于最近竞争者Compositional Search（4.4/17）；在所有基准上均实现更快收敛和更高最优值；

**⚠️ 局限性**

局限性：当前仅针对单一任务进行核发现，跨任务迁移仍需进一步研究；部分LLM生成的核因正定性或维度兼容性问题被过滤，生成效率可提升；需要更系统的多任务共学习与更细粒度的prompt约束。

---

## 54. Privacy-by-Design Adaptive Group Assignment for Digital Lifestyle Coaching at Scale

**arXiv ID:** 2605.20505 | [PDF](https://arxiv.org/pdf/2605.20505v1)

**作者:** Nariman Mani `[一作]` (Nutrosal Inc.), Salma Attaranasl `[通讯]` (Nutrosal Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并部署了 PRISM-Coach 系统，在数字生活方式教练平台实现了隐私保护与自适应群组分配。

**💡 创新点**

创新点在于四视图隐私边界、隐私约束的情境 Bandit 分组、以及人机协作的 AI 辅助教练。

**🔧 技术方法**

使用了结构化 tokenization、密钥管理的身份 vault、线性 UCB 隐私约束上下文 Bandit、可审计的 AI 生成与人机回显、差分隐私汇总。

**📊 数据集**

使用了三年平台遥测（约 15 万用户）和应用内需求与体验调查数据。

**📈 对比分析**

与传统静态分组和部署前基线做对比，AI‑enabled 模型使每日签到依从率提升 26%/33%，参与指数提高 33%，平均体重下降 2.1 kg，AI 辅助的建议准确率 88%/84%。

**⚠️ 局限性**

局限在于非随机化的观察设计、季节性和干预变化的混杂、用户自报数据偏差及残留的推断风险。

---

## 55. CP-MoE: Consistency-Preserving Mixture-of-Experts for Continual Learning

**arXiv ID:** 2605.20247 | [PDF](https://arxiv.org/pdf/2605.20247v1)

**作者:** Yang Liu `[一作]` (University of New South Wales), Flora D. Salim `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 CP‑MoE 框架，利用瞬态专家、表示一致性路由偏置和基于表示的正则化，解决大规模 MoE 模型在持续学习中的灾难性遗忘与知识迁移不平衡问题。

**💡 创新点**

创新点包括：①瞬态专家作为前置评估探针，捕获任务特定更新并估计参数重要性；②一致性保持路由偏置（CP‑Bias）基于 CKA 兼容性对专家进行路由引导；③表示导向正则化按兼容性加权保护关键参数，三者协同实现对任务间干扰的精准抑制与知识迁移的最大化。

**🔧 技术方法**

主要技术包括：LoRA‑MoE 架构、瞬态专家（Transient Expert）热启动、Centered Kernel Alignment (CKA) 兼容性评估、Consistency‑Preserving 路由偏置、重要性累计正则化、负载平衡辅助损失、AdamW + cosine 学习率衰减。

**📊 数据集**

实验数据集为：①Super‑NaturalInstructions (SuperNI) 进行多任务语言生成与零样本迁移；②VQA v2 进行多模态视觉问答的任务增量学习。

**📈 对比分析**

与多种基线（InfLoRA、O‑LoRA、GainLoRA、CL‑MoE、VQACL 等）在同一实验设置下对比；在 SuperNI 上 AP 最高 50.84%，零样本迁移 35.80%；在 VQA v2 上 AP 62.30%，平均遗忘（AF）-0.35%，均超过对手并取得显著的性能提升。

**⚠️ 局限性**

局限性：①依赖短期温暖启动和瞬态专家，需额外数据和计算；②缺乏对极大任务序列或多任务交叉场景的深入评估；③在复杂多模态交互中对专家兼容性评估的计算成本较高；④对硬件资源仍有一定要求，未来可探索更高效的兼容性估计与动态专家扩展。

---

## 56. Efficient Table QA via TableGrid Navigation and Progressive Inference Prompting

**arXiv ID:** 2605.20254 | [PDF](https://arxiv.org/pdf/2605.20254v1)

**作者:** Amritansh Maurya `[一作]` (IIIT Allahabad), Omar Moured `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 73 | [OpenAlex ID](https://openalex.org/A5053010867)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了两种基于大语言模型的表格问答（TableQA）提示框架——TableGrid Navigation (TGN) 与 Progressive Inference Prompting (PIP)，实现了在不需额外训练的情况下对表格数据的精准检索与多步推理。

**💡 创新点**

创新点在于将表格视为二维网格通过迭代的分析–执行–验证循环（TGN）以及通过列识别、行筛选、逐步推理的结构化提示（PIP）来显式控制模型的检索路径和推理过程，从而显著降低幻觉和错误率。

**🔧 技术方法**

采用了大语言模型（如Qwen3-8B、Llama-3.1-8B、Meta‑Llama‑3-8B等）配合自定义提示模板，利用vLLM框架实现高效推理；TGN与PIP均无模型微调，仅通过提示工程提升性能。

**📊 数据集**

使用公开基准表格问答数据集 TableBench 与 FeTaQA，涵盖事实检验、数值推理、数据分析等多种子任务。

**📈 对比分析**

在TableBench上，TGN以48.46%（全量准确率）领先所有基线；在FeTaQA上，PIP在所有基线中取得最高分（如Meta‑Llama‑3-8B+PIP在多项ROUGE/BERTScore上位居榜首）。同时，TGN与PIP可作为监督模板，对小模型进行微调后也能缩小与大模型的性能差距。

**⚠️ 局限性**

局限性主要体现在：①模型在最终答案生成阶段仍易出现不一致或略微不准的错误；②在极端多步推理或长文本回答场景下，提示长度受限；③缺乏对非结构化表格（如合并单元格、缺失值）处理的细粒度机制。

---

## 57. Same Target, Different Basins: Hard vs. Soft Labels for Annotator Distributions

**arXiv ID:** 2605.20642 | [PDF](https://arxiv.org/pdf/2605.20642v1)

**作者:** Mirerfan Gheibi `[一作]` (Independent Researcher), Gashin Ghazizadeh `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了将标注者分布以硬标签形式交付给模型（multipass与SLS），并与传统软标签训练进行对比。

**💡 创新点**

创新点在于证明硬标签交付在稀疏标注下可优于软标签，且在完整分布时与软标签达到相同期望目标，同时揭示不同交付方式导致更平坦的训练基底。

**🔧 技术方法**

采用多重传递（multipass）、随机标签抽样（SLS）、对照实验，并结合梯度方差分析、Hessian谱分析和OOD检测等技术手段。

**📊 数据集**

实验基于CIFAR-10H人类标签分布，使用改装的ResNet‑18网络。

**📈 对比分析**

与软标签训练、投票、标签平滑、混合等基线相比，硬标签在稀疏标注时软NLL更低；当完整分布可用时与软标签相当；并在OOV检测和梯度方差方面展现更平坦的解。

**⚠️ 局限性**

局限在于仅使用单一数据集和网络结构、实验种子有限、稀疏标注的模拟方式等，结果对其他任务或更大模型的可推广性尚未验证。

---

## 58. HalluCXR: Benchmarking and Mitigating Hallucinations in Medical Vision-Language Models for Chest Radiograph Interpretation

**arXiv ID:** 2605.20469 | [PDF](https://arxiv.org/pdf/2605.20469v1)

**作者:** Haoyu Wang `[一作]` (King's College London), Zitong Li `[通讯]` (King's College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 HalluCXR 基准，评估 6 种不同架构的 VLM 在 856 张 MIMIC‑CXR 胸片上 15,408 次输出的幻觉情况。

**💡 创新点**

创新点包括：八分类幻觉层级与严重性评分、两层自动检测与 LLM 判别管道、基于响应长度的风险评分、以及多模型集成校正策略。

**🔧 技术方法**

使用技术有：基于 CheXpert 的关键词与否定检测、LLM（Gemini、GPT‑5.1）裁判、AUC/ROC 分析、投票与 ECE 权重集成、4‑bit 量化模型推理。

**📊 数据集**

数据集为 MIMIC‑CXR 856 张前平胸片（220 病例），并利用 CheXpert 12 项标签作为黄金标准。

**📈 对比分析**

实验比较显示幻觉率介于 61.9%–82.3%，严重幻觉达 36.4%–80.2%；长度阈值 745 字符 AUC 0.908；集成方法可将 fabrication 降至 0.050（84.8% 降幅），同时保持 0.534 的 F1。

**⚠️ 局限性**

局限包括：仅覆盖 12 项 CheXpert 标签，罕见标签提取噪声；LLM 判别存在同族偏差；单中心胸片数据；固定提示模板；开源模型量化与商业 API 性能不完全可比。

---

## 59. Hiding in Plain Sight: Finding MAHA on Reddit

**arXiv ID:** 2605.20435 | [PDF](https://arxiv.org/pdf/2605.20435v1)

**作者:** Sabit Ahmed `[一作]` (University of Virginia), Henry Kautz `[通讯]` (University of Virginia)

**通讯引用:** 21864 | [OpenAlex ID](https://openalex.org/A5105508446)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了覆盖2020‑2025年6年、包含12个健康主题的19.4M Reddit帖子和4M用户的MAHA运动数据集，并为帖子与评论提供主题立场标签。

**💡 创新点**

首次在大规模社交媒体中系统性挖掘隐藏在“公开视野”中的MAHA社区，结合多阶段关键词检索、语义分类器和树式少样本LLM立场判别，实现在数百万规模下的细粒度立场标注。

**🔧 技术方法**

采用两阶段正则表达式关键词检索、句子BERT嵌入+逻辑回归的MAHA相关性分类器以及Qwen2.5‑32B的树式少样本in‑context学习进行立场识别。

**📊 数据集**

使用Arctic Shift镜像的Pushshift原始Reddit数据（2020‑2025年提交与评论）以及自定义的12主题关键词集合。

**📈 对比分析**

在分层抽样的36个讨论树上评估，四类F1达到0.56、去除无立场后三类F1为0.52，明显高于随机基准，证明模型可行。

**⚠️ 局限性**

关键词过滤可能遗漏隐含立场的短评论，且缺乏跨主题深度上下文，导致部分立场识别误差和主题覆盖不足。

---

## 60. Can Vision Models Truly Forget? Mirage: Representation-Level Certification of Visual Unlearning

**arXiv ID:** 2605.20282 | [PDF](https://arxiv.org/pdf/2605.20282v1)

**作者:** Zhenyu Yu `[一作]` (Fudan University), Shuigeng Zhou `[通讯]` (Fudan University)

**通讯引用:** 11321 | [OpenAlex ID](https://openalex.org/A5017862559)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了 Mirage 框架，用于对视觉模型的表示层进行遗忘审计，验证模型在行为上看似遗忘但内部特征仍保留信息的现象。

**💡 创新点**

创新点在于：①定义了以重训练基线为参照的“遗忘差距”指标，②将四种诊断（线性探测恢复、CKA 对齐、特征可分性、层级恢复）组合为统一的表示层评估。

**🔧 技术方法**

技术包括线性探测器（logistic 回归）、Centered Kernel Alignment（CKA）、Fisher-inspired 可分性评分，以及层级 LPR 分析。

**📊 数据集**

使用七个数据集（MNIST、CIFAR‑10、CIFAR‑100、ModelNet、脑瘤 MRI、COVID‑19 放射学、Yahoo Answers）以及对应的 VFL 配置。

**📈 对比分析**

对比了八种现有未学习方法（Retrain、FT、Fisher、Amnesiac、UNSIR、BU、SSD、Target），发现即使输出层满足遗忘要求，LPR 与 CKA 仍显示显著残留结构；无方法同时满足高效能、输出遗忘和表示遗忘。

**⚠️ 局限性**

局限性包括：仅为审计框架而非新算法；实验基于 VFL 方案，可能不适用于更复杂的真实部署；线性探测可能低估可恢复信息，且未针对横向联邦学习或更强攻击者的场景做进一步评估。

---

## 61. Flexible Coupler Antenna for Wireless Networks: Opportunities and Challenges

**arXiv ID:** 2605.20560 | [PDF](https://arxiv.org/pdf/2605.20560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 62. Synchronization and Turn-Taking in Full-Duplex Speech Dialogue Models

**arXiv ID:** 2605.20356 | [PDF](https://arxiv.org/pdf/2605.20356v1)

**作者:** Pablo Riera `[一作]` (ASAPP Inc.), S. R. K. Branavan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在模拟的两实例 Moshi 全双工对话环境下，测量内部表示同步（CKA）并用因果 LSTM 探针预测即将到来的 IPU 边界与是否保持发话。

**💡 创新点**

提出结合 CKAs 与因果 LSTM 探针的评估框架，以量化全双工对话模型的内部同步与预期转折，并将其与人类神经耦合相类比。

**🔧 技术方法**

使用 Linear CKA、因果 LSTM、AUC‑ROC 评估，并在模拟噪声通道、PAD 令牌偏置以及预训练与微调模型版本上进行对比实验。

**📊 数据集**

以约 80 小时音频的 100 秒长度模拟对话为数据集，采用预录医疗预约情境音频作为启动提示，包含四种噪声等级、三种 PAD 偏置、两种模型版本共 2880 段对话。

**📈 对比分析**

通过不同噪声、偏置和模型版本的 CKAs 峰值与探针 AUC‑ROC 进行比较，结果显示无噪声条件下同步和预测性能最高，噪声降低同步至 0.1 以下，微调对性能影响有限。

**⚠️ 局限性**

仅在单一情境、相同提示、短时对话中测试，探针可能利用全局时间规律而非真正内部表征，缺乏对人机交互和多模型、多层次动力学的验证。

---

## 63. Weasel: Out-of-Domain Generalization for Web Agents via Importance-Diversity Data Selection

**arXiv ID:** 2605.20291 | [PDF](https://arxiv.org/pdf/2605.20291v1)

**作者:** Fatemeh Pesaran zadeh `[一作]` (Seoul National University), Gunhee Kim `[通讯]` (Seoul National University)

**通讯引用:** 6130 | [OpenAlex ID](https://openalex.org/A5100664729)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对离线网页代理训练进行数据选择与状态裁剪，选取固定预算的轨迹步骤并通过目标中心化AXTree裁剪减少无关内容，提升训练效率和跨域泛化。

**💡 创新点**

①引入基于目标相关性与对称多样性的双目标重要性-多样性子集选择框架；②使用贪心算法高效求解；③提出目标中心化AXTree裁剪与自回归推理合成，减少输入冗余并消除风格不匹配。

**🔧 技术方法**

BERTScore/CLIP用于计算重要性与多样性；贪心子集选择；AXTree线性化与窗口裁剪；LLM自回归推理合成；多模态嵌入（SigLIP、CLIP）在扩展实验中使用。

**📊 数据集**

AgentTrek（合成教程轨迹）与NNetNav（真实浏览轨迹）两大离线数据集；在WebArena、WorkArena、MiniWob等基准上进行零样本转移评估。

**📈 对比分析**

与全量SFT、随机抽样、LLM-as-Judge等基线对比；在WebArena-Lite、WebArena、MiniWob、WorkArena等任务上，WEasel在保持或提升零样本成功率的同时，训练速度提升约9.7–12.5×；在多模态GUI代理上亦显示 约 14% 的准确率提升。

**⚠️ 局限性**

①对BERTScore/CLIP的语义相似度假设仍可能不足；②目标中心化裁剪主要适用于AXTree文本表示，对非文本或多模态输入的适用性有限；③推理合成会增加标记长度，影响极大模型的成本；④实验集中在零样本转移，其他学习场景（如在线RL、连续任务）未充分验证。

---

## 64. ParaVT: Taming the Tool Prior Paradox for Parallel Tool Use in Agentic Video Reinforcement Learning

**arXiv ID:** 2605.20342 | [PDF](https://arxiv.org/pdf/2605.20342v1)

**作者:** Zuhao Yang `[一作]` (Nanyang Technological University), Lidong Bing `[通讯]` (MiroMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种名为ParaVT的多代理强化学习框架，能够在单个回合内并行发起多段视频裁剪请求，实现长视频推理中的并行工具调用；

**💡 创新点**

创新点在于首次引入并行单回合多工具调用策略，解决了传统顺序调用导致的误裁剪传播、上下文腐败与线性推理成本等问题，并针对工具先验矛盾提出了探索锚定与帧预算门控的GRPO改进；

**🔧 技术方法**

采用冷启动监督微调+强化学习（GRPO）相结合的两阶段训练，加入结构化格式奖励（Anchor）与随机帧预算（nFrames Gating），实现对格式稳定性和工具利用率的双重调控；

**📊 数据集**

使用自制的97K样本多任务SFT语料（包含通用视频问答、长视频推理、时序定位与并行工具轨迹）以及4.4K样本RL集（开放式问答、多选题与时序定位），并在六个公开长视频基准上进行评估；

**📈 对比分析**

与现有7–8B开源模型（如Qwen3‑VL‑8B）以及工具增强基线进行对比，ParaVT在六个基准上均优于同类模型，平均提升约7.9%，在长视频多选题和时序定位任务上取得显著领先；

**⚠️ 局限性**

局限性包括：仅在单一大型模型（Qwen3‑VL‑8B）上验证，缺乏跨模型通用性研究；训练过程中仍需精细调参以避免格式崩溃；并行调用虽然降低推理成本，但在极大规模工具集合时仍面临并行资源调度挑战。

---

## 65. ELEMENT: Multi-Modal Retinal Vessel Segmentation Based on a Coupled Region Growing and Machine Learning Approach

**arXiv ID:** 2605.20458 | [PDF](https://arxiv.org/pdf/2605.20458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 66. NeuroQA: A Large-Scale Image-Grounded Benchmark for 3D Brain MRI Understanding

**arXiv ID:** 2605.20525 | [PDF](https://arxiv.org/pdf/2605.20525v1)

**作者:** Mohammad H. Abbasi `[一作]` (Stanford University), Ehsan Adeli `[通讯]` (Stanford University)

**通讯引用:** 14787 | [OpenAlex ID](https://openalex.org/A5015355317)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建并公开了一个针对 3D 脑 MRI 的视觉问答基准，包含 56,953 条 QA 对，覆盖 12 个数据集、203 个模板、11 个推理类别、三种答复格式，并通过多轮专家审核、自动化 38 条规则审计以及模板级快捷语料清理，形成了“短板攻击防护”的基准。

**💡 创新点**

创新点：① 大规模、跨域 3D MRI VQA 数据集；② 通过模板级答案平衡与分布审计，将文本仅答复精度降至随机水平附近；③ 设定三项图像基准测试（图像缺失、造假、幻觉）以验证模型是否真正“读懂”三维影像；④ 提供人类临床医生视觉基准与两层发布架构，确保可复现与可扩展。

**🔧 技术方法**

使用的技术包括：基于 FreeSurfer 体积量化和结构化元数据的真值生成；NIfTI 3 方向视图；基于规则的文本预处理与否定解析；多轮专家评审和自动化规则验证；模板级答案重分布与 95% 目标平衡；最后的图像基准与性能评测脚本。

**📊 数据集**

所用数据集：ADNI、AIBL、PPMI、BraTS-GLI、BraTS-MEN、WMH、CC359、IXI、HCP-YA、HCP-Aging、HCP-Development、ABCD，总计 12,977 受试者。

**📈 对比分析**

比较方法：与零样本 Vision‑Language 模型（Gemini‑3.1‑Pro、GPT‑5.2 等）、一份 1.02M 参数的 3D CNN 监督基线以及随机与文本仅基线对照。结果显示：所有模型均未突破文本仅 49.4% 的基准，也低于 48.9% 的临床医生视觉基准，表明当前模型尚未真正利用 3D 图像信息；相比之下，Oracle（完全图像+文本）得分可达 100%。

**⚠️ 局限性**

局限性：① 仅为评估基准，非临床部署；② 关注问答任务，未覆盖更高级的临床决策、报告生成等；③ 仍存在高难度“可视化不可检索”项目，需要进一步改进模板与答案平衡；④ 受限于公开数据与 DUA 许可，部分数据的可复现性依赖于内部脚本；⑤ 未来工作需引入 T2/FLAIR、视觉质量控制等新元素。

---

## 67. Group-Algebraic Tensors: Provably-optimal Equivariant Learning and Physical Symmetry Discovery

**arXiv ID:** 2605.20440 | [PDF](https://arxiv.org/pdf/2605.20440v1)

**作者:** Paulina Hoyos `[一作]` (UT Austin), Lior Horesh `[通讯]` (IBM Research)

**通讯引用:** 2236 | [OpenAlex ID](https://openalex.org/A5089022499)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种张量代数框架，使得任何有限群G定义乘法规则，从而使得等变性成为一种内在的代数属性，而不是架构约束。

**💡 创新点**

创新点在于提供了一个证明最优的对称保持张量近似的Eckart-Young最优性保证，并且通过机器验证的理论基础支持该框架的有效性。

**🔧 技术方法**

使用了张量代数和广义傅里叶变换等技术，结合了机器学习中的岭回归。

**📊 数据集**

使用了QM9数据集，该数据集包含134,000个小有机分子及其12个量子化学性质。

**📈 对比分析**

与传统的等变神经网络（ENNs）相比，-SVD结合岭回归在参数效率和准确性上表现出色，尤其在小样本情况下，-SVD的表现优于其他神经网络模型。

**⚠️ 局限性**

该框架目前仅处理有限群，扩展到连续群结构是一个未解决的问题。此外，当前的框架主要基于立方体对称性，可能无法捕捉更复杂的对称性结构。

---

## 68. Pixel Wised Lesion Prediction on COVID-19 CT Imagery: A Comparative Analysis of Automated Image Segmentation Architectures

**arXiv ID:** 2605.20459 | [PDF](https://arxiv.org/pdf/2605.20459v1)

**作者:** Sarmad Khan `[一作]` (National University of Sciences & Technology), Basim Azam `[通讯]` (University of Melbourne)

**通讯引用:** 103 | [OpenAlex ID](https://openalex.org/A5002895693)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

系统评估了四种深度学习分割架构（Unet、PSPNet、Linknet、FPN）与六种预训练骨干网络在COVID‑19 CT图像病灶分割上的性能。

**💡 创新点**

提出统一的实验框架，将多种分割头与多种骨干组合进行二分类和多分类的综合比较，并实现二分类F1≈98%、多分类F1 75–77%的最佳结果。

**🔧 技术方法**

采用深度卷积网络、预训练编码器、Sigmoid/Softmax激活、组合损失（加权二元交叉熵+焦点损失+Tversky损失）、图像增强、Jaccard/F1评价指标等技术。

**📊 数据集**

使用公开的COVID‑19 CT数据集，包括Zenodo、Medical Segmentation（Radiopedia+Medseg）等，涵盖背景、肺部、病灶等三类标签。

**📈 对比分析**

通过统一训练参数、交叉验证、Jaccard Index和F1-score进行比较，结果显示Unet+MobileNetV2在二分类上F1达98%，多分类中Linknet+Inception‑ResNetV2与FPN+DenseNet121分别达到75%和77%，优于现有基准。

**⚠️ 局限性**

局限性包括数据量有限、仅针对CT图像、未评估在不同设备或更复杂肺部分割任务上的泛化性能，以及训练仍受GPU资源限制。

---

## 69. Clove: Object-Level CXL Memory Management in Managed Runtimes

**arXiv ID:** 2605.20370 | [PDF](https://arxiv.org/pdf/2605.20370v1)

**作者:** Sam Son `[一作]` (University of California Berkeley), Scott Shenker `[通讯]` (University of California Berkeley)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在 JVM 等托管运行时中实现了面向对象级的 CXL 内存管理系统，支持透明、高效的热点跟踪与对象迁移。

**💡 创新点**

创新点在于利用托管运行时已有的对象可见性、GC 迁移与 JIT 生成机制，结合硬件采样的热点指令定位，实现了轻量级、可选择的对象级热点跟踪，以及混合页/对象级迁移与冷热对象压缩策略，解决了 CXL 低开销与缺乏拦截点的问题。

**🔧 技术方法**

使用技术包括：PEBS 采样定位 L3 缓存未命中热点指令；JIT 动态插桩在热点指令处更新对象头部计数；利用 ZGC 的移动 GC 进行热点对象压缩；周期性激活与回收策略控制热点计数与迁移开销；基于区域热度水位的迁移选择。

**📊 数据集**

数据集与工作负载：合成 Ehcache HotWarm 与 Zipfian（0.99）分布；真实工作负载：Ehcache + Twitter 生产轨迹（Large/Medium），JGraphT PageRank（GAP benchmark），H2 数据库 + TPC‑C；基线对比系统包括 TPP、Memtis、HybridTier。

**📈 对比分析**

方法：在相同硬件与 CXL 模拟环境下，将新系统与三种基线系统进行基准对比，测量相对于全局本地内存的延迟/吞吐量、局部内存命中率；结果显示在 10%–50% 本地内存比例下，系统将应用慢速 22–84%，比基线提升 11–45% 的命中率，且运行时开销仅 1–5%。

**⚠️ 局限性**

局限性：只适用于支持对象元数据、JIT 与可移动 GC 的托管语言（主要是 JVM，理论上可扩展到 .NET、PyPy 等）；不覆盖 off‑heap 原生内存；假设热点变动速度慢，极端频繁热点变化未验证；实现需对运行时做较大修改，对标准 JVM 兼容性有限。

---

## 70. LEAP: A closed-loop framework for perovskite precursor additive discovery

**arXiv ID:** 2605.20242 | [PDF](https://arxiv.org/pdf/2605.20242v1)

**作者:** Xin-De Wang `[一作]` (Renmin University of China), Zhong-Yi Lu `[通讯]` (Renmin University of China)

**通讯引用:** 7656 | [OpenAlex ID](https://openalex.org/A5082102844)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了LEAP（LLM‑driven Exploration via Active Learning for Perovskites），一个闭环专家‑in‑the‑loop 框架，用专业化的大语言模型提取文献机制知识，生成可解释的软特征，并结合贝叶斯优化与实验反馈实现前驱体添加剂的优先级排序与实验验证。

**💡 创新点**

创新点在于：① 将域特化LLM（Perovskite‑RL）用于从文献中提取机制相关信息并生成五维可解释软特征；② 将这些机制特征与传统分子描述相结合，构建混合表征；③ 在低数据环境下利用贝叶斯优化（GP+EI）与专家审核形成闭环，实现机制感知的高效添加剂筛选。

**🔧 技术方法**

使用技术包括：域特化LLM（Qwen3‑32B）通过SFT+RL（LoRA、GRPO）训练；贝叶斯优化（Gaussian Process surrogate + Expected Improvement）做候选优先级排序；实验工艺与表征（FTIR、XRD、SEM、J‑V、EQE、SCLC、稳定性测试）；机制一致性基准与统计检验。

**📊 数据集**

数据集主要包括：1,264篇perovskite添加剂文献的机制与属性数据（90,749条SFT样本，5,800条RL样本）；热启动实验添加剂库（36条）；实验设备数据（每种条件24台）；机制一致性基准（16篇文献，32道多选题）。

**📈 对比分析**

与一般基础模型（未fine‑tune的Qwen、其他通用LLM）在机制一致性基准中对比，Perovskite‑RL准确率达78.1%，显著优于基线；实验验证中，第二轮6‑CDQ获得20.57% PCE，第三轮2‑CNA获得21.32% PCE，均显著高于对照19.25%，且非LEAP选取的三个添加剂未出现类似提升。

**⚠️ 局限性**

局限性包括：仅在三种添加剂与单一器件结构上验证；仍需人工专家审核，尚未实现完全自动化；机制特征与模型设计专为perovskite，是否能迁移到其他材料或更大空间尚未测试；实验数据规模有限，缺乏更广泛的验证。

---

## 71. Spacetime Optimal-Transport Attention for Visuo-Haptic Imitation Learning of Contact-Rich Manipulation

**arXiv ID:** 2605.20433 | [PDF](https://arxiv.org/pdf/2605.20433v1)

**作者:** Yue Feng `[一作]` (Nanyang Technological University), I-Ming Chen `[通讯]` (Nanyang Technological University)

**通讯引用:** 14269 | [OpenAlex ID](https://openalex.org/A5046605612)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种三模态融合框架 SO-TA，结合视觉、力/力矩和姿态信息实现接触丰富操作的模仿学习

**💡 创新点**

将力姿态条件的空间聚合视为熵正则化的最优传输问题，用显式边缘约束代替传统 softmax attention，显式编码任务相位先验

**🔧 技术方法**

最优传输 (Sinkhorn 迭代)、扩散策略 (Diffusion Policy)、Transformer 以及 ResNet 视觉编码器

**📊 数据集**

在三条真实机器人任务上收集的演示数据：紧密插孔装配、BCM 线束插拔、曲面擦拭，约 600+ 轮演示，共 3 万+ 步长

**📈 对比分析**

与拼接和交叉注意力基线对比，SO-TA 在插孔任务实现 100% 成功率、光照/遮挡扰动下 82.5% 成功率，而拼接仅 43.5%；其他任务均匹配或优于基线

**⚠️ 局限性**

最优传输热图相对模糊，且在接触操作中仍存在累计误差的长尾，需运行时补救（截断-重置）

---

## 72. Instance Discrimination for Link Prediction

**arXiv ID:** 2605.20257 | [PDF](https://arxiv.org/pdf/2605.20257v1)

**作者:** Valentin Cuzin-Rambaud `[一作]` (Université Lyon 1), Rémy Cazabet `[通讯]` (Université Lyon 1)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了自监督实例判别在无属性图的链路预测任务中的应用，并提出了两种基于链路表示的模型 L‑GRACE 与 L‑BGRL。

**💡 创新点**

创新点在于将增广过程视为可调超参数，并通过社区结构驱动的 SBM 增广以及针对链路的对比/非对称损失来实现针对链路预测的自监督学习。

**🔧 技术方法**

使用了图对比学习（GRACE）、无监督自监督方法（BGRL）与自定义的链路表示与 InfoNCE/Asymmetric 损失，结合 SBM 生成的社区增广。

**📊 数据集**

在八个无属性网络（USAir、Power、Router、Yeast、Celegans、NS、PB、Ecoli）以及若干有属性网络（Cora、Citeseer、Coauthor‑CS、Coauthor‑Physics、Amazon‑Computers、Amazon‑Photo）上进行实验。

**📈 对比分析**

与传统监督 GCN、SEAL、ELPH 等基线相比，L‑GRACE 与 L‑BGRL 在大多数数据集上达到或逼近最先进性能，特别是在无属性图上显著优于 GRACE/BGRL 的原始版本。

**⚠️ 局限性**

主要局限包括对链路表示的内存开销高、对 SBM 质量依赖强、在属性图上表现不如无属性图、以及对社区检测的依赖导致增广效果不稳定。

---

## 73. You Don't Need Attention: Gated Convolutional Modeling for Watch-Based Fall Detection

**arXiv ID:** 2605.20275 | [PDF](https://arxiv.org/pdf/2605.20275v1)

**作者:** Sana Alamgeer `[一作]` (Texas State University), Anne H. H. Ngu `[通讯]` (Texas State University)

**通讯引用:** 7669 | [OpenAlex ID](https://openalex.org/A5016020974)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Gated-CNN轻量级双流卷积网络，用门控模块替代自注意力，实现手表级跌倒检测；

**💡 创新点**

创新点在于门控机制可显式抑制背景激活并放大跌倒特征，避免软max稀释；同时通过双流并行处理加快推理，显著优于Transformer；

**🔧 技术方法**

采用1D-CNN特征提取、GLU风格的sigmoid门控、全局平均池化、共享分类头，训练时使用TensorFlow、推理时转换为TFLite；

**📊 数据集**

使用五个腕部IMU数据集：SmartFallMM、WEDA-Fall、FallAllD、UMAFall、UP-Fall；

**📈 对比分析**

与Transformer及其他双流模型在LOSOCV下对比，平均F1 90–93%，比Transformer高8–15%；在Pixel Watch 3实时测试平均F1 97%，准确率98%，且推理时间仅2.8 ms，参数3.1 万；

**⚠️ 局限性**

局限性包括对高幅值日常活动的误报（如穿衣、扫地），训练集缺乏此类样本；门控对极端噪声或多尺度模式的适应性有限，需要进一步数据扩充和自适应微调。

---

## 74. Tippett-minimum Fusion of Representation-space Diffusion Models for Multi-Encoder Out-of-Distribution Detection

**arXiv ID:** 2605.20502 | [PDF](https://arxiv.org/pdf/2605.20502v1)

**作者:** Neelkamal Bhuyan `[一作]` (Georgia Institute of Technology), Neelkamal Bhuyan `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5083298403)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于预训练编码器多视角的表示空间扩散模型（RDM）并通过两级最小值门（EncMin2L）实现无监督的 OOD 检测；

**💡 创新点**

创新点在于仅利用 ID 数据计算 η²、Δμ、Spearman ρ 等诊断指标自动识别编码器的专长并动态选择最强检测器，避免了对 OOD 例子或标签的依赖；

**🔧 技术方法**

主要技术包括表示空间扩散模型（VP‑SDE+PF‑ODE）、统计校准的最小值门（Tippett 最小 p‑value 组合）以及 ID 数据诊断指标；

**📊 数据集**

使用 CIFAR‑10、CIFAR‑100 作为 ID 数据，评估了 SVHN、CIFAR‑100、CIFAR‑10C、DTD、CelebA 等四种 OOD 迁移场景；

**📈 对比分析**

与单编码器、单一密度模型、分类器基准、特征空间方法及多种扩散 OOD 方法比较，EncMin2L 在所有四种 OOD 场景均能取得 ≥0.94 AUROC，且参数量比最强对手低 2.3 倍；

**⚠️ 局限性**

局限性包括对合成腐蚀探测（Δμ）对真实传感器噪声等多样化 covariate shift 的泛化不足，以及最小值门假设编码器 p‑value 独立性的前提在实际中可能被违反。

---

## 75. Spectral Unforgetting: Post-Hoc Recovery of Damaged Capabilities Without Retraining

**arXiv ID:** 2605.20296 | [PDF](https://arxiv.org/pdf/2605.20296v1)

**作者:** Aarash Abro `[一作]` (Zeta Labs), Muhammad Tahir `[通讯]` (Lahore University of Management Sciences)

**通讯引用:** 16324 | [OpenAlex ID](https://openalex.org/A5100689313)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种仅基于预训练和微调检查点的无数据后置修复方法 DG-Hard，用来恢复微调后遗失的能力，并保持微调获得的新能力。

**💡 创新点**

创新点：①将微调增量拆解为低秩任务信号与 IID 噪声两部分；②采用 Donoho‑Gavish 最优硬阈值对每个权重增量的奇异值进行筛选；③设计了分区条件评估指标（healing、preservation、non‑damage、on‑task retention），在多任务多模型上衡量恢复‑保留平衡。

**🔧 技术方法**

技术手段：奇异值分解（SVD）、Donoho‑Gavish 线性最优硬阈值、无数据/无梯度的矩阵过滤；评估时使用分区统计、和和谐平均（HM）作为整体得分。

**📊 数据集**

数据集：14 个 (模型, 任务) 微调细胞（Qwen3.5‑4B、Llama‑3.2‑3B‑Instruct）+ 9 个跨域保留基准（MMLU、TriviaQA、TruthfulQA、ARC‑Challenge、GSM8K、IFEval、Math‑500、MNLI、HellaSwag）；安全性评估使用 HarmBench、XSTest、StrongREJECT。

**📈 对比分析**

与 WiSE‑FT、FAPM、LoRA、V‑SoftMask、CoFi‑Tune 等后置或训练时方法对比。DG‑Hard 在 14 个微调细胞、9 个基准上获得最高的 Combined 分数（≈83.3），比第二佳方法提升 6.9 分，且在安全性维度几乎恢复到基线水平。

**⚠️ 局限性**

局限性：①假设任务信号集中在高奇异值方向，若信号分散可能被误删；②每个权重矩阵需要计算 SVD，规模较大模型时开销不容忽视；③仅去除噪声残差，无法处理因微调引入的结构性误差。

---

## 76. PolycubeNet: A Dual-latent Diffusion Model for Polycube-Based Hexahedral Mesh Generation

**arXiv ID:** 2605.20274 | [PDF](https://arxiv.org/pdf/2605.20274v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 77. A Human-in-the-Loop Framework for Efficient Prompt Selection in Microscopy Vision-Language Models

**arXiv ID:** 2605.20495 | [PDF](https://arxiv.org/pdf/2605.20495v1)

**作者:** Abhiram Kandiyana `[一作]` (University of South Florida), Dmitry Goldgof `[通讯]` (University of South Florida)

**通讯引用:** 14023 | [OpenAlex ID](https://openalex.org/A5053211631)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于目标驱动的主动提示调优框架，用于在显微镜图像分类中构建紧凑的提示集，并在达到验证性能目标后停止标注；

**💡 创新点**

创新点在于将提示集构造视为主动学习问题，设计了三种互补的采样策略（不确定性引导、复杂度感知不确定性、密度树边界采样），并将生成式模型的提示生成与专家校验相结合，实现低成本标注；

**🔧 技术方法**

使用大型视觉‑语言模型 GPT‑4o 进行提示调优与推理，结合随机采样与主动学习的采样策略，采用句子嵌入和token计数评估提示长度与冗余；

**📊 数据集**

在 Lurcher 小鼠脑组织的低倍染色显微图像数据集上进行实验，该数据集包含21只小鼠的图像，任务是区分突变与野生型；

**📈 对比分析**

与随机采样的基线 APT‑USF 相比，三种主动学习方法均在低资源约束下显著降低专家标注量（平均20–35张图像），实现 100% 动物级别准确率，且在图像级别上平均准确率提升约 8 分；

**⚠️ 局限性**

局限性包括：只在单一数据集上验证，依赖 GPT‑4o 等大模型的可用性，且对不同视觉‑语言模型的泛化与不同预训练语义空间的敏感性尚未完全探究。

---

## 78. Training Language Agents to Learn from Experience

**arXiv ID:** 2605.20477 | [PDF](https://arxiv.org/pdf/2605.20477v1)

**作者:** Yuval Shalev `[一作]` (University of Cambridge), Mateja Jamnik `[通讯]` (University of Cambridge)

**通讯引用:** 1535 | [OpenAlex ID](https://openalex.org/A5036018012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 In-context Training (ICT) 框架，用语言模型的反射机制在交互式环境中实现跨任务自我提升，并训练反射模型学习生成改进的系统提示。

**💡 创新点**

创新点在于：①将跨任务自我提升建模为可学习的元任务；②提出无人工标注的 RL 训练管线，通过在先前任务上重播来给反射模型奖励；③构建 MetaGym 库，为元学习环境统一化接口；④证明反射模型可在未见任务甚至不同 benchmark 上迁移。

**🔧 技术方法**

使用大语言模型 Qwen2.5‑7B‑Instruct 作为演员与基准反射器；采用 GRPO（梯度奖励权重优化）进行 RL 微调；采用 MetaGym 作为元环境框架；利用自然语言描述的系统提示与经验轨迹进行训练与评估。

**📊 数据集**

数据集：ALFWorld（6种任务类型）与 MiniHack（多种 5×5 网格任务），在训练集上构造多种任务类型作为元训练，留出部分任务类型作为元测试；通过在训练集中多次运行 ICT 循环生成反射训练数据。

**📈 对比分析**

对比方法：未训练 Qwen2.5‑7B‑Instruct 以及初始提示；训练后的反射器在元测试任务上多次运行 ICT，取最佳提示后计算成功率。实验显示：在 ALFWorld、MiniHack 以及跨 benchmark（MiniHack 训练、ALFWorld 测试）中，训练后的反射器的成功率均显著高于基线，尤其在 4 轮后仍继续提升。

**⚠️ 局限性**

局限性：①仅训练反射器，未同步训练演员，导致双模型存储成本高；②在 ICT 循环中使用固定验证集评估最佳提示，现实场景可能缺乏可重复验证集；③实验仅在少数任务类型与模型规模上验证，未探究更大规模或更复杂环境下的长期稳定性与持续学习；④在小批量情况下奖励稀疏导致学习困难。

---

## 79. Evaluating Temporal Semantic Caching and Workflow Optimization in Agentic Plan-Execute Pipelines

**arXiv ID:** 2605.20630 | [PDF](https://arxiv.org/pdf/2605.20630v1)

**作者:** Alimurtaza Mustafa Merchant `[一作]` (Columbia University), Kaoutar El Maghraoui `[通讯]` (IBM)

**通讯引用:** 924 | [OpenAlex ID](https://openalex.org/A5034513925)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为工业资产操作中的基于LLM的Agent提供低延迟的Plan-Execute流程，提出两层优化方案：时序语义缓存与MCP工作流改进；

**💡 创新点**

创新点在于：①在缓存前加入时序分类器，区分静态、相对、锚定和实时查询，解决参数与时间敏感的工业查询；②利用磁盘持久化工具发现缓存与并行执行DAG的方式，显著降低工具发现和执行阶段的开销；③系统两层优化相互独立且可叠加，显著提升整体吞吐量；

**🔧 技术方法**

使用LLM（Llama‑3.3‑70B via LiteLLM）进行规划与总结，Qwen3 embeddings + FAISS做语义检索，Asteria式的reranker判定，基于MCP（Model Context Protocol）进行工具调用，临时分类器用正则表达式，工作流并行化采用DAG分层与连接池；

**📊 数据集**

使用AssetOpsBench（AOB）基准，包含152条工业操作查询，进一步生成80条同义句（60%热、40%冷）进行缓存实验，18条IoT查询用于工作流并行性评估；

**📈 对比分析**

与未优化基线对比：MCP层单独提升中位延迟1.67×，完整管道（缓存+MCP）提升3.48×，缓存命中率45%，命中时平均可获得30.6×速度提升；对各阶段开销进行细分，发现工具发现从2.1s降至0.007s，执行从34.6s降至17.4s；

**⚠️ 局限性**

局限性包括：①纯语义缓存无法处理参数冲突导致的误命中，F1上限≈0.64；②缓存仅内存化，重启后失效；③仅在单台Apple M‑系列机器上实验，未考虑多机并发与持久化；④日期解析仅覆盖固定短语，未处理自然日期；⑤工作流并行化受计划结构限制，收益不均；

---

## 80. Decomposing MXFP4 quantization error for LLM reinforcement learning: reducible bias, recoverable deadzone, and an irreducible floor

**arXiv ID:** 2605.20402 | [PDF](https://arxiv.org/pdf/2605.20402v1)

**作者:** Xiaocan Li `[一作]`, Zheng Shen `[通讯]` (Huawei Canada)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对MXFP4量化误差进行三分解，揭示其对RL后训练的不同影响，并提出针对每种误差的校正方法；

**💡 创新点**

创新点在于给出精确的scale bias、deadzone truncation和grid noise三种误差的分解与对应RL失效模式，并通过Macro‑Block Scaling、Outlier Fallback和Adaptive Quantization Noise实现对三种误差的机制‑定向修正；

**🔧 技术方法**

使用的技术包括MXFP4量化格式、QDQ模拟、Macro‑Block Scaling、Outlier Fallback、Adaptive Quantization Noise、GRPO训练框架和TIS重要性采样；

**📊 数据集**

实验数据集为GSM8K数学推理任务；

**📈 对比分析**

与BF16基线对比，dense模型在MBS+AQN+OF组合下仅退化0.7个百分点，MoE模型退化3.0个百分点，显示所提出方法显著恢复精度；

**⚠️ 局限性**

局限性包括仅验证于GSM8K、使用QDQ模拟而非原生硬件、AQN噪声参数需手工调优，以及实验仅使用单个随机种子。

---

## 81. Geometry-Lite: Interpretable Safety Probing via Layer-Wise Margin Geometry

**arXiv ID:** 2605.20241 | [PDF](https://arxiv.org/pdf/2605.20241v1)

**作者:** Woo Seob Sim `[一作]` (Yonsei University), Yu Rang Park `[通讯]` (Yonsei University)

**通讯引用:** 3421 | [OpenAlex ID](https://openalex.org/A5046176363)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Geometry‑Lite，一种紧凑的提示级安全性探测器，通过将每层隐藏状态映射为三种几何边界下的标量边距并按层级、变化和形状三轴总结，来分析多层安全信号；

**💡 创新点**

创新点在于将多层隐藏状态的安全信息分解为可解释的边距几何特征，并通过对不同几何读出（质心、k‑NN、本地邻域、监督线性边界）以及层间运动的系统消融，揭示安全信号主要来源于边距位置的持续性，而非层间位移；

**🔧 技术方法**

使用的技术包括：隐藏状态到边距的映射（差分均值、k‑NN、线性回归），边距轨迹的三轴统计（幅度、负漂移、结构形状），以及对39维摘要的 L2‑正则化逻辑回归分类；

**📊 数据集**

实验数据集涵盖七个安全基准（XSTest、WildJailbreak、JBB‑Behaviors、DoNotAnswer、BeaverTails、PKU‑SafeRLHF、ToxicChat）和九个指令微调模型（Llama、Gemma、Qwen 1.2B–70B），并设置硬难度子集；

**📈 对比分析**

与单层探测器和原始多层分数堆叠（MultiLayer‑Linear、TaT‑Disp‑LSTM）比较，Geometry‑Lite 在全分布 AUROC 上与最优堆叠相当（0.955~0.964），在低误报率和留一基准转移上略有优势，且证明边距幅度特征最关键；

**⚠️ 局限性**

局限性包括：仅处理二分类安全标签、单轮提示、手工设计的边距摘要，未考虑多轮交互或更复杂的安全任务，且在极端基准下仍需改进。

---

## 82. GrandGuard: Taxonomy, Benchmark, and Safeguards for Elderly-Chatbot Interaction Safety

**arXiv ID:** 2605.20203 | [PDF](https://arxiv.org/pdf/2605.20203v1)

**作者:** Changxuan Fan `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10727 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了针对老年人使用LLM的安全评估框架GrandGuard，包括细粒度风险分类、10,404条标注的prompt/response基准，并提出基于微调和政策增强的两种安全防护措施。

**💡 创新点**

首次提出针对老年人情境风险的细粒度三层风险分类与基准，结合agent层的上下文增强和可定制的政策式防护，显著提升对老年人特有风险的识别与缓解。

**🔧 技术方法**

采用LLM微调（LoRA）、policy‑enhanced moderation、自动化prompt生成、人工与LLM混合标注、GrandGuard agent等技术手段实现风险检测与响应安全化。

**📊 数据集**

利用AI incident数据库、社区讨论帖、访谈报告等多源数据，生成并标注10,404条prompt/response样本，结合LLM生成的合成数据完成基准构建。

**📈 对比分析**

在10款主流LLM上进行评估，发现多数模型在老年人场景下安全率低于50%；微调的Llama‑Guard‑3和policy‑enhanced gpt‑oss‑safeguard‑20b在GrandGuard基准上达>90%检测准确率，Agent层提升多达50%安全率。

**⚠️ 局限性**

数据代表性有限、合成prompt可能带风格偏差与标签噪声、严重度量单维、评估仅覆盖单轮静态交互、外部防护仍易产生误报/漏报，需长期、多轮、现实环境验证。

---

## 83. Axiomatizing Neural Networks via Pursuit of Subspaces

**arXiv ID:** 2605.20534 | [PDF](https://arxiv.org/pdf/2605.20534v1)

**作者:** Mehmet Yamac `[一作]`, Moncef Gabbouj `[通讯]` (Tampere University)

**通讯引用:** 29874 | [OpenAlex ID](https://openalex.org/A5007583477)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了Pursuit of Subspaces（PoS）假设，构建了一套以几何公理为基础的深度学习理论框架，并以此解释网络的表示学习、计算机制和泛化行为；随后通过PoS模块实现残差学习、角度分离和等变性，并在ECG异常检测、光学显微镜体积重建、Transformer/PosFormer等多任务上进行实验验证。

**💡 创新点**

创新点主要有四个：① 将深度网络视为非线性正交投影到子流形集合的几何框架；② 提出了四条核心公理（Compactness、Projection、Residual、Recursive Projection），统一描述了网络的层级投影和残差机制；③ 通过PoS模块阐释了ReLU角度分离、残差连接与等变性变换的几何根源；④ 证明深度网络可将样本复杂度从乘法降低为加法，从而解释其对大规模变换的高效泛化；⑤ 在实验中实现了零样本异常检测、域迁移、隐私保护与可解释性，验证了PoS的实际效能。

**🔧 技术方法**

技术手段包括：几何与微分拓扑（流形、切空间、正交投影）、子空间稀疏表示与RIP理论、残差学习与正交补空间、ReLU角度分离、等变性变换（Isometry、Group Action）、PoS模块设计、投影正则化与动态Push-Pull损失、遮掩与噪声注入等数据增强、卷积/Transformer架构、光学显微镜深度重建网络。

**📊 数据集**

实验使用的数据集包括：个性化多用户ECG信号集（健康与异常样本）；光学光场显微镜（FLFM）采集的3D体积数据；以及公开图像/图像生成数据集用于AutoEncoder的预训练与微调（未具体列名）。

**📈 对比分析**

在ECG异常检测任务中，PoS（compact）在仅使用健康样本的零样本设置下相较于传统子空间投影、稀疏逼近及多用户训练的深度模型取得显著优势；在域迁移与隐私保护任务中，PoS实现了高准确率（异常检测AUC≈0.95，隐私保护F1≈20%降至≈20%）；在FLFM体积重建中，采用PoS + Push‑Pull 损失的模型将PSNR提升约4.66 dB，SSIM提升至0.974，明显优于基线（PSNR≈34 dB，SSIM≈0.968）。

**⚠️ 局限性**

局限性包括：部分理论推导仍为假设或经验性结论，缺乏严格的全局证明；实验覆盖的任务相对有限，缺乏大规模、多域的验证；PoS框架假设数据能在子流形集合上良好表示，复杂或高噪声数据可能不满足；对不同网络结构（如稀疏网络、图神经网络）的适用性尚待进一步探索；以及对训练样本与变换空间的覆盖率与可解释性仍存在挑战。

---

## 84. HAPS: Rethinking Image Similarity for Virtual Staining

**arXiv ID:** 2605.20362 | [PDF](https://arxiv.org/pdf/2605.20362v1)

**作者:** Fedor Gubanov `[一作]` (Skolkovo Institute of Science and Technology), Maxim Sharaev `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 592 | [OpenAlex ID](https://openalex.org/A5081476624)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对跨染色图像相似度评估的专用指标 HAPS，并在虚拟染色工作流中实现自动数据清洗与质量控制。

**💡 创新点**

创新点在于：①利用在大量病理切片上对比学习预训练的 RetCCL 编码器提取多尺度病理特征；②通过专家评分进行线性校准，使指标与病理学家的判读高度一致；③在多种图像变形下证明该指标对细微配准误差具有鲁棒性；④在虚拟染色模型训练中应用指标筛选数据，显著提升生成图像的分布真实性。

**🔧 技术方法**

技术包括：全参考图像相似度评估框架、RetCCL（ResNet50 对比学习模型）、Pearson 相关距离计算、专家校准的线性头、数据预处理（灰度、CLAHE、去噪）和基准指标（PSNR、SSIM、LPIPS、DISTS、NCC 等）。

**📊 数据集**

数据集：①自建专家标注的 522 张 H&E→HER2 补丁对（256×256）用于指标评估；②公开 MIST 虚拟染色基准（4,642 训练对 / 1,000 测试对，1024×1024）用于下游生成模型验证。

**📈 对比分析**

比较方法：用 Spearman 相关系数、三分类 AUROC 及二分类 AUROC 评估指标与专家评分的一致性；用 FID/KID 评估虚拟染色模型的分布真实性。HAPS 在 Spearman 上达到 0.540、三分类 AUROC 0.743、二分类 AUROC 0.775，明显优于最优基线 LPIPS+SqueezeNet（0.485/0.717/0.691）。在 MIST 上使用 HAPS 过滤 25% 低分样本后，BCI 模型的 FID 从 100.03 提升至 96.52，PSPStain 的 KID 由 0.012 降至 0.009，证明指标在实际数据清洗中的有效性。

**⚠️ 局限性**

局限性：①实验仅覆盖乳腺癌 H&E–HER2 组合，泛化到其他组织/染色需进一步验证；②专家评分成本高，指标的专家校准过程依赖大量人工标注；③RetCCL 预训练数据来自同一来源，跨中心、跨设备的差异可能影响指标表现；④目前 HAPS 仅作为评价和筛选工具，尚未验证其可微分损失在训练过程中的直接使用效果。

---

## 85. ShadeBench: A Benchmark Dataset for Building Shade Simulation in Sustainable Society

**arXiv ID:** 2605.20510 | [PDF](https://arxiv.org/pdf/2605.20510v1)

**作者:** Longchao Da `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**通讯引用:** 7775 | [OpenAlex ID](https://openalex.org/A5100777770)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 ShadeBench 大规模多模态数据集和统一评估基准，用于城市阴影的生成、分割和 3D 建筑重建；

**💡 创新点**

创新点在于：①将卫星影像、OSM 建筑骨架与 Blender 阴影渲染对齐，解决几何偏差；②使用 NOAA Solar Position Algorithm 生成位置、时间相关的物理光照；③提供标准化的评估协议和多任务基准；

**🔧 技术方法**

采用的技术包括：OSM 数据提取、卫星图像对齐器（基于 Canny 边缘检测与掩模/生成式修正）、Blender 阴影渲染、ControlNet/DeepShade 生成模型、SAM/SAM2/GeoSAM 分割模型、SAM‑3D 3D 重建模型；

**📊 数据集**

使用的数据集为 ShadeBench，覆盖 6 大洲 34 城市，包含对齐后的卫星图、建筑骨架、时变阴影图、结构化文本描述及 Blender 导出的 3D 建筑网格；

**📈 对比分析**

对比方法：在生成任务中比较 Diffusion、ControlNet 与 DeepShade，评估 SSIM/LPIPS，DeepShade 最高；在分割任务中评估 SAM/SAM2/GeoSAM，模拟图 Dice/IUoU 0.98+，卫星图仅 0.1‑0.5；在 3D 重建任务中用 Chamfer Distance 评估，SAM‑3D 可恢复主要结构；

**⚠️ 局限性**

局限性包括：对齐策略可能错误移除真实建筑；阴影变化仅按小时离散，缺乏连续性；数据集依赖 OSM 可能缺少细节；基线模型在细腻阴影、低对比度区域表现仍不理想。

---

## 86. Introspective X Training: Feedback Conditioning Improves Scaling Across all LLM Training Stages

**arXiv ID:** 2605.20285 | [PDF](https://arxiv.org/pdf/2605.20285v1)

**作者:** Brandon Cui `[一作]` (NVIDIA), Prithviraj Ammanabrolu `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一的“自省训练”(Introspective Training)方法，通过在训练前期为数据文档预先加上评判者生成的质量评分和自然语言批评，直接在自回归语言模型的下一个词预测任务中使用前缀条件化训练；

**💡 创新点**

创新点在于将后期训练阶段获取的质量反馈倒推到早期训练阶段，使得所有训练阶段（预训练、持续预训练、微调）都能利用质量信号；同时提出两种条件化方式（量化质量标记和自然语言批评）并验证其在多任务和多阶段中的有效性；

**🔧 技术方法**

技术包括：离线评判者模型(Judge LLM)对文档进行五轴评分与批评；对评分/批评生成前缀并将其拼接到原始文档；使用标准下一个词预测目标训练模型；在SFT阶段将前缀放在system message中；评估使用多种基准（GSM8K、MATH、HumanEval、MBPP、MMLU等）；

**📊 数据集**

使用的数据集包括：Nemotron Nano v2（高质量Common Crawl、数学、代码文本）；Dolmino（多领域混合语料）；CraneMath、SwallowCode（专门数学和代码语料）；以及各种公开基准数据集；

**📈 对比分析**

与传统无条件下一个词预测(NTP)比较，Introspective Training 在相同计算量下提高 5–10 点左右的评测分数，最优情况下可达 2.8 倍的 FLOP 效率；在 12T/18T 检查点继续预训练时仍能保持优势；在 SFT 阶段也能获得正向迁移；

**⚠️ 局限性**

局限性包括：需要额外的评判者模型和离线注释成本；批评模板可能与某些领域（如代码）不完全匹配导致效能波动；在已高度筛选的 SFT 数据上提升幅度有限；未来需改进领域专属评判指标与批评模板。

---

## 87. HyperBones: Realtime Bone-driven Neural Garment Simulation with Hypernetwork Conditioning

**arXiv ID:** 2605.20460 | [PDF](https://arxiv.org/pdf/2605.20460v1)

**作者:** Astitva Srivastava `[一作]` (IIIT Hyderabad), Egor Larionov `[通讯]` (Meta Reality Labs)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种实时、物理可行的衣物仿真框架，使用虚拟骨架驱动粗糙变形，再通过UV卷积网络恢复细节，并通过自监督物理损失训练。

**💡 创新点**

创新点：1）将低频全局变形与高频细节拆分为两阶段；2）用轻量骨网络和FiLM调制学习虚拟骨变形；3）通过共享编码器和自监督MeshGraphNet实现物理一致性；4）预计算身份特定特征，仅在推理时执行骨网络，极大提升速度。

**🔧 技术方法**

技术手段包括：LBS (线性混合皮肤化)、虚拟骨架与骨网络（Bone-Net）、FiLM 条件化、UV空间卷积-MLP细节恢复、MeshGraphNet 物理自监督、动态时间集成。

**📊 数据集**

数据集：使用 AMASS 运动序列进行自监督训练，服装几何来自 VTO 数据集，人体模型为 SMPL，用于生成不同体型的身份特征。

**📈 对比分析**

与 SNUG、NCS、GAPS、HOOD 等方法在 6 件服装上对比，取得最低边缘误差和面积误差；在紧身服装上碰撞误差略高；实时性能达到 300+ FPS，速度比 HOOD 高 25×，显著优于现有方法。

**⚠️ 局限性**

局限性：目前仅支持固定 6 件预训练服装，无法即时适配新服装；细节恢复依赖 UV 贴图，对贴图质量敏感；对极紧身服装的碰撞处理仍不如专门的图网络方法。

---

## 88. Under Pressure: Emotional Framing Induces Measurable Behavioral Shifts and Structured Internal Geometry in Small Language Models

**arXiv ID:** 2605.20202 | [PDF](https://arxiv.org/pdf/2605.20202v1)

**作者:** Rana Muhammad Usman `[一作]` `[通讯]` (Independent Researcher), Rana Muhammad Usman (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究情绪化评估提示对小型语言模型在不可满足编码约束任务中的行为与内部激活方向的影响。

**💡 创新点**

创新点在于将8种情绪提示与不可满足任务相结合，发现情绪提示可诱导可测量的激活方向，并揭示其低维结构。

**🔧 技术方法**

使用激活分析、层级分离度计算、PCA可视化、向量差分与因果激活 steering 等技术。

**📊 数据集**

使用 Qwen 3.5 0.8B 与 2B 两个规模模型，在四个不可满足编码约束任务上共 160 次对话（0.8B）及 20 次压制/平静重跑（2B）。

**📈 对比分析**

通过诚实/快捷词法指标与层 23 激活分离度对比，发现压力提示产生最高快捷率，层 23 激活方向最大；在 2B 模型上因果激活 steering 能提升行为一致性，表现优于 0.8B。

**⚠️ 局限性**

局限包括：仅使用词法匹配评估诚实/快捷，任务域单一，PCA 取样量有限，因果激活实验样本极少，且所有向量均相对“平静”基线，未证明情绪真实性。

---

## 89. Chronicle: A Multimodal Foundation Model for Joint Language and Time Series Understanding

**arXiv ID:** 2605.20268 | [PDF](https://arxiv.org/pdf/2605.20268v1)

**作者:** Paul Quinlan `[一作]` (Inertialai), Xiaodan Zhu `[通讯]` (Queen's University)

**通讯引用:** 10892 | [OpenAlex ID](https://openalex.org/A5016892586)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

训练了一个从随机初始化的324M参数解码器Transformer，统一学习自然语言与时间序列。

**💡 创新点**

创新点在于：①仅从零开始同时联合预训练两种模态；②共享同一Transformer骨干，无需后期适配；③通过极少量交叉模态对齐数据实现跨模态能力。

**🔧 技术方法**

技术包括patch编码+Arcsinh归一化的时间序列表示，自回归跨模态预训练，BPE词表嵌入，RMSNorm、SwiGLU、RoPE和量化预测头。

**📊 数据集**

使用的预训练数据为FineWeb-Edu、Dolmino-mix-1124文本以及GiftEvalPretrain+KernelSynth时间序列；评估数据包括19个NLU任务、GIFT-Eval、UCR/UEA 24个时序分类数据、Time-MMD多模态预测以及TimeCAP多模态分类。

**📈 对比分析**

与规模匹配的LLM（GPT‑2、Gemma‑3、LLaMA‑3.2）在19 NLU任务表现相当；零样本预测在GIFT‑Eval与主流TSFM相近；冻结嵌入在UCR/UEA分类中超过所有基准；在Time‑MMD多模态预测中超越所有监督融合基线，性能提升显著。

**⚠️ 局限性**

局限性包括：预训练中文本比例高导致时序性能略逊于专用TSFM；单向自回归目标导致长周期误差累积；交叉模态对齐仅占5%数据，仍有提升空间；未验证对话推理或检索任务。

---

## 90. HealthTale: A Patient-Centric Health Story Visualization Tool

**arXiv ID:** 2605.20207 | [PDF](https://arxiv.org/pdf/2605.20207v1)

**作者:** Ryan Smith `[一作]` (University of British Columbia), Tamara Munzner `[通讯]` (University of British Columbia)

**通讯引用:** 10556 | [OpenAlex ID](https://openalex.org/A5079916657)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了 HealthTale 系统，帮助患者在初诊前通过自由文本生成结构化时间线，梳理并可视化其健康故事以便与临床医生沟通。

**💡 创新点**

通过多阶段定性研究提出面向患者的健康故事数据抽象，并利用 LLM 将自由叙事自动转为包含时间模糊、生活事件等属性的事件结构，弥合患者叙事与临床可读可视化的鸿沟。

**🔧 技术方法**

使用 Claude Sonnet 4.6 LLM 进行文本解析，DBSCAN 聚类实现多时间尺度布局，D3/TypeScript 开发交互式可视化，并支持静态打印版本。

**📊 数据集**

采集了 20 条 Reddit 健康帖、22 条书面健康故事以及 85 条整体语料，用于评估，参与者包括 34 名患者和 3 名临床医生。

**📈 对比分析**

通过形成性验证与用户研究（患者 34 名、医生 3 名）评估解析准确率（误差约 41/1090 事件）和可视化可读性，结果显示解析有效、布局易于快速扫描，用户满意度高，未进行传统 EMR 对比。

**⚠️ 局限性**

尚未在真实临床流程中评估，样本量偏小且医生技术倾向性高，系统对解析错误的纠正依赖患者主动，缺乏与 EMR 深度集成与大规模持续数据处理能力。

---

## 91. The Structure and Dynamics of the Online MAHA-sphere

**arXiv ID:** 2605.20457 | [PDF](https://arxiv.org/pdf/2605.20457v1)

**作者:** Sabit Ahmed `[一作]` (University of Virginia), Henry Kautz `[通讯]` (University of Virginia)

**通讯引用:** 21864 | [OpenAlex ID](https://openalex.org/A5105508446)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用2020-2025年Reddit数据，构建MAHA运动的跨主题信念生态系统，分析其用户交互网络、跨主题捆绑、时间演化和语言差异；

**💡 创新点**

创新点在于把MAHA视为多主题整体，揭示其跨主题高度捆绑、网络中心性以及两种语言框架（感知/情感 vs 证据/理性），并首次量化跨主题迁移与网络结构；

**🔧 技术方法**

使用双阶段关键词+LLM（Qwen2.5‑32B‑Instruct）立场分类、用户意见得分、网络分析（Kamada‑Kawai、UPGMA）、心理语言分析（LIWC‑22）等技术；

**📊 数据集**

采用2020-2025年Reddit公开帖子与评论约1.3 B评论、234 M帖子，覆盖12个健康争议主题；

**📈 对比分析**

与传统单主题研究对比，MAHA内部的主题间共识度显著高于抗MAHA组；网络分析显示核心主题聚集，外延主题散布；跨主题迁移显著（如防疫主题向疫苗/科学聚合），表明MAHA具有高度结构化与演化特征；

**⚠️ 局限性**

局限包括仅聚焦Reddit平台、平台内容审核可能导致极端意见缺失、关键词过滤与主题定义可能遗漏隐性立场、以及LIWC效应值相对较小。

---

## 92. Generation of Heterogeneous PET Images from Uniform Organ Activity Maps Using a Pretrained Domain-Adapted Diffusion Model

**arXiv ID:** 2605.20267 | [PDF](https://arxiv.org/pdf/2605.20267v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 93. Prism: Structural Symmetry Scanning via Duality-Constrained Laplacian Projection

**arXiv ID:** 2605.20245 | [PDF](https://arxiv.org/pdf/2605.20245v1)

**作者:** Jiatong Xie `[一作]` `[通讯]`, Jiatong Xie

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Prism 框架，利用图拉普拉斯与对称性算子 P 计算双重性缺陷 δ(L,P)，作为网络结构健康度量。

**💡 创新点**

创新点在于给出闭式投影到 P 的交换子并以 δ(L,P) 为连续、可解释的诊断指标，同时提出无监督学习 P 的交替优化方法。

**🔧 技术方法**

技术包括谱理论、线性代数投影、Fiedler 向量初始化、交替优化、RMT 过滤以及实验验证。

**📊 数据集**

使用的数据集包括 20 节点合成对称网络、Zachary Karate Club、S&P 500 股票相关网络（实时与历史 10 年）以及 5 次主要金融危机。

**📈 对比分析**

与原始拉普拉斯、模块度、RMT 等方法比较，Prism 在 5% 边噪声下社区识别准确率达 94.5%（高于 76.6%），双重性缺陷对结构退化的敏感度比错误 P 高 3.38 倍，并能提前检测金融网络的结构压力。

**⚠️ 局限性**

局限性包括：缺陷度量仅在 P 有意义时有效，随机或不相关的 P 会产生噪声；目前仅支持无向正权图；多社区扩展与有向图/异质图的处理尚未完善。

---

## 94. Do No Harm? Hallucination and Actor-Level Abuse in Web-Deployed Medical Large Language Models

**arXiv ID:** 2605.20591 | [PDF](https://arxiv.org/pdf/2605.20591v1)

**作者:** Sunday Oyinlola Ogundoyin `[一作]` (Macquarie University), Rahat Masood `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文对OpenAI GPT Store中6,233个Web部署的MedGPT以及10个开源医疗LLM进行大规模审计，评估其临床幻觉与开发者层面滥用风险。

**💡 创新点**

创新点在于提出MedGPT-HEval多指标幻觉检测框架和基于LLM的开发者滥用评估管道，实现了对医疗LLM安全性的系统化、可复现的多维度分析。

**🔧 技术方法**

采用自动爬虫、G-Eval、BARTScore、语义熵、余弦相似度等多指标评分，利用Gemini 3.1 Pro做“LLM裁判”，并用K-means聚类确定政策合规阈值，同时进行人工校准。

**📊 数据集**

使用自构建的HAA-MedGPT数据集（6,233个MedGPT元数据及1,500个交互样本）以及10个公开医疗LLM在MedQA基准下的响应，用于幻觉和合规性评估。

**📈 对比分析**

通过多指标对比，发现25–30%的MedGPT在G-Eval低于0.8，37%未达BARTScore≥-3.5，41%余弦相似度<0.4，且用户活跃度与幻觉指标几乎不相关；与开源模型相比，MedGPT在事实准确性和语义对齐上表现更好，而开源模型在语义熵上更稳定。

**⚠️ 局限性**

研究局限包括仅捕获2026年1月20–22日的Store快照、对Store覆盖度不完整、仅评估静态元数据风险、开源模型样本有限，以及评估指标缺乏临床真实验证。

---

## 95. Mechanisms of Misgeneralization in Physical Sequence Modeling

**arXiv ID:** 2605.20299 | [PDF](https://arxiv.org/pdf/2605.20299v1)

**作者:** Kento Nishi `[一作]` (Harvard), Hidenori Tanaka `[通讯]` (Harvard University)

**通讯引用:** 2921 | [OpenAlex ID](https://openalex.org/A5050969230)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了生成序列模型在物理量分布上的误差，并提出了物理误差泛化（physical misgeneralization）现象

**💡 创新点**

首次将模型局部误差通过物理测量量的传播机制建模，并引入数据偏差核（data deviation kernel）来预测概率转移，从而解释并预估分布漂移

**🔧 技术方法**

采用深度扩散模型（1D U‑Net）、数据偏差核、仿真恢复测量、以及自监督编码等技术

**📊 数据集**

使用自制的合成任务（正弦、锥形和 Logistic 映射）、双摆模拟、Maze2D 导航数据集

**📈 对比分析**

与真实训练模型生成的物理量分布对比，预测误差率低于 10%，并通过核干预（数据表示变换）显著降低分布漂移，性能优于传统数据重采样或条件建模

**⚠️ 局限性**

仅针对扩散模型定义的核，未涵盖自回归、VAE 等其它生成架构；在多阶段管线和非物理序列任务中效果需进一步验证；对数据表示和公平性等方面仍有局限

---

## 96. Advanced Scientific Methodology Plays Rossini

**arXiv ID:** 2605.20220 | [PDF](https://arxiv.org/pdf/2605.20220v1)

**作者:** Silvia Licciardi `[一作]` (University of Palermo), Elisa Francomano `[通讯]` (University of Palermo)

**通讯引用:** 765 | [OpenAlex ID](https://openalex.org/A5080409036)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对罗西尼《Mi lagnerò tacendo》一百余个版本的乐谱进行音乐信息提取与结构化分析，构建异构图模型并做统计与语义关联分析。

**💡 创新点**

首次将句法解析、图论、统计分布与机器学习统一到同一框架内，能够同时考察旋律、和声、节奏与文本之间的多维关联，提供对作者变体的微观与宏观双重视角。

**🔧 技术方法**

MusicXML 解析、特征提取、图神经网络（GNN）、n‑gram 信息理论、数据挖掘与统计分布（Zipf‑Mandelbrot、熵等）技术。

**📊 数据集**

Rossini 的 133 份“Mi lagnerò tacendo”手稿版本（Sibelius .sib 格式转换为 MusicXML）。

**📈 对比分析**

通过单变体深度分析与多变体聚合对比，展示在局部差异显著的情况下聚合分布仍能体现罗西尼的宏观倾向；相对传统宏观级别研究，方法更细致、可量化，性能以统计稳定性与模式一致性为衡量。

**⚠️ 局限性**

受限于仅聚焦单一作曲家与单一文本，缺乏跨作曲家或跨流派的验证；图模型规模与计算复杂度较高；缺少深度学习模型的系统评估，尚未在大规模数据集上验证泛化能力。

---

## 97. RealUserSim: Bridging the Reality Gap in Agent Benchmarking via Grounded User Simulation

**arXiv ID:** 2605.20204 | [PDF](https://arxiv.org/pdf/2605.20204v1)

**作者:** Ming Zhu `[一作]` (Salesforce Ai Research), Huan Wang `[通讯]` (Salesforce Ai Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建基于真实用户行为数据的用户模拟框架，提取7,275个可执行行为档案，用于对LLM用户模拟器进行情境化模拟，并评估其与真实用户的行为匹配度。

**💡 创新点**

首次利用真实人机对话数据生成可执行的用户档案，突破“形式化天花板”和“指令放大”问题，并提出Paired Trajectory Turing Test (PT3)多维度真实性评估基准。

**🔧 技术方法**

使用GPT‑4o进行用户档案构建与行为提取，采用多种LLM（GPT‑4o、GPT‑5‑mini、GPT‑5、Llama‑3‑70b、gpt‑oss‑20b、Claude‑3‑Sonnet）进行模拟对话和评估；PT3评判器基于人工评估模型进行匹配率判断。

**📊 数据集**

主要数据集为WildChat‑4.8M（14,000+真实人机对话），以及Airline与Retail两大域的任务集用于端到端代理评估。

**📈 对比分析**

在PT3上对600条多域对话进行匹配率评估，基于档案的模拟从24.2%提升至45.3%；在代理评估中，真实档案模拟导致平均任务成功率下降约3%（暴露三种失败机制），而去除手工指令可提升多达+16%（因指令放大导致难度失衡）。

**⚠️ 局限性**

局限性包括：单会话用户示例被清除导致语言风格匹配率偏低；GPT‑4o在生成混乱文本时仍显规范；在技术知识维度的过度校准可能导致专家级对话的误判；以及不同模型对指令敏感性差异使评估结果高度依赖所选模拟器。

---

## 98. Fast Reconstruction of Exact Maxwell Dynamics from Sparse Data

**arXiv ID:** 2605.20514 | [PDF](https://arxiv.org/pdf/2605.20514v1)

**作者:** Dan DeGenaro `[一作]` (Georgetown University), Bogdan Raiţă `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个深度浅层神经网络，利用权重共享和代数约束保证每个隐藏神经元本身是麦克斯韦方程的精确解，从而实现从稀疏点观测中重建四维空间电磁场。

**💡 创新点**

将麦克斯韦方程的解空间直接嵌入网络结构，使用计算代数构造精确解并证明在任意域具有普适逼近性质，确保零PDE残差并显著加速训练；同时在极低数据量下实现极高精度。

**🔧 技术方法**

运用Ehrenpreis–Palamodov定理、符号计算得到权重约束，设计浅层tanh激活网络，并采用随机梯度下降训练；与PINN、FEM、S‑EPGP等基准方法对比。

**📊 数据集**

使用自行生成的稀疏观测数据集，基于四种真实解（平面波、径向波、Hopf纤维化、随机解），在约1K或仅100点观测下进行实验，没有使用公开的标准数据集。

**📈 对比分析**

在无/有边界条件下与PINN、FEM、S‑EPGP进行比较；在“race to 5%”实验中，本文方法在数秒内达到5%相对误差，PINN需数百秒，S‑EPGP同样慢；在70秒时间预算下误差显著低于对手；在仅100点数据下可达<1%误差，整体比传统方法快1–3个数量级。

**⚠️ 局限性**

仅适用于无源、均匀麦克斯韦方程；不支持电荷、电流、异质介质、边界交界等复杂物理；初始/边界条件通过拟合实现，非符号约束；网络规模和优化策略可能限制在更大/更复杂问题上的性能。

---

## 99. Mechanics of Bias and Reasoning: Interpreting the Impact of Chain-of-Thought Prompting on Gender Bias in LLMs

**arXiv ID:** 2605.20410 | [PDF](https://arxiv.org/pdf/2605.20410v1)

**作者:** Edie Pearman `[一作]` (Mila), Golnoosh Farnadi `[通讯]` (Mila)

**通讯引用:** 822 | [OpenAlex ID](https://openalex.org/A5053667504)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了链式推理（CoT）提示在大语言模型（LLM）中的性别偏见缓解效果，结合基准评测、机制可解释性与推理链失败分析。

**💡 创新点**

创新点在于将四大多选题基准、注意力头簇与隐藏状态探针、以及对推理链的七类行为分类三种视角融合，揭示 CoT 仅实现表层缓解且依赖模型与数据集特性。

**🔧 技术方法**

采用的技术包括：注意力头簇分析（Stereotype Attention Score）、隐藏状态探针（MLP 预测答案类型）、推理链行为分类器（基于 DeBERTa 的二分类模型）以及标准与 CoT 提示下的对比实验。

**📊 数据集**

使用的数据集有 BBQ、StereoSet、CrowS-Pairs 与 SocioEconomicQA 四个英文多选题性别偏见基准。

**📈 对比分析**

通过对标准与 CoT 提示下的置信度不确定率和 Diff‑Bias 指标进行对比，实验表明 CoT 在不同模型（Qwen、QwQ、Llama、Mistral 等）和不同数据集上的表现不一致，未能稳定降低偏差，甚至在部分情况导致偏差提升。

**⚠️ 局限性**

局限性包括：仅覆盖二元性别与四个特定数据集；模型长度限制导致推理链被截断；推理链的真实性受限于模型自述的可信度；注意力与输出关系的因果性尚未完全验证；探针方法可能存在数据泄露与层级覆盖不足。

---

## 100. Intent-First Aerial V2V for Tactical Coordination and Separation: Protocol and Performance Under Density and Disturbance

**arXiv ID:** 2605.20595 | [PDF](https://arxiv.org/pdf/2605.20595v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 101. Super-Beamforming in Holographic MIMO

**arXiv ID:** 2605.20377 | [PDF](https://arxiv.org/pdf/2605.20377v1)

**作者:** Andrea Pizzo `[一作]` (University Pompeu Fabra), Angel Lozano `[通讯]` (University Pompeu Fabra)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过理论分析和数值仿真研究了线性全息天线阵列在半波长以下间距时的互耦效应，推导了端火增益可实现二次增长（超增益）的定律，并给出了损耗限制下的性能极限。

**💡 创新点**

创新点在于：①证明互耦能在不缩小间距的情况下将增益从线性提升到二次增长；②提出两种超增益机制——空间集中和频域集中，并利用谱集中理论对其进行严谨定量分析；③给出了在实际损耗条件下实现超增益的具体参数约束。

**🔧 技术方法**

主要技术包括：离散空间傅里叶变换、耦合矩阵特征值分解、谱集中与离散勒普拉斯多项式、Sturm–Liouville 方程、WKB近似以及离散正交基底（离散勒普拉斯多项式）的构造。

**📊 数据集**

论文主要使用数值仿真数据（如supergain_theta_data、G_N_data等）来验证理论，未引用公开实验数据集。

**📈 对比分析**

通过将超增益因子与传统半波长阵列的线性增益做对比，并计算Q因子，展示在特定间距和损耗水平下可获得10–20倍增益提升，但伴随极高的Q因子和窄带宽。

**⚠️ 局限性**

局限性在于：超增益对天线间距、制造精度和损耗极为敏感，易受结构扰动和宽带波束倾斜影响；实现高度需要极低的阻抗损耗和精确的相位控制，实际部署具有技术挑战。

---

## 102. GraphDiffMed: Knowledge-Constrained Differential Attention with Pharmacological Graph Priors for Medication Recommendation

**arXiv ID:** 2605.20188 | [PDF](https://arxiv.org/pdf/2605.20188v1)

**作者:** Krati Saxena `[一作]` (Kyushu Institute of Technology), Tomohiro Shibata `[通讯]` (Kyushu Institute of Technology)

**通讯引用:** 7610 | [OpenAlex ID](https://openalex.org/A5067304768)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出GraphDiffMed药物推荐框架，结合双尺度差分注意力和药理DDI图先验，以抑制噪声并考虑药物相互作用。

**💡 创新点**

创新点在于同时应用访视内与访视间双尺度差分注意力，并将药物相互作用图作为先验注入注意力机制，兼顾推荐质量与安全性。

**🔧 技术方法**

采用差分注意力v2、图偏置差分注意力、GRU序列编码、药物分子图神经网络、DDI正则化以及诊断/手术因果校正等技术。

**📊 数据集**

使用MIMIC‑III ICU数据集，包含6350名病人、15032次访问，ATC3级药物、ICD‑9诊断、手术、实验室指标和人口学信息。

**📈 对比分析**

与DADA‑MED、CIDGMed、PROMISE、LEAP等多种基线在Jaccard、F1、PRAUC、DDI率及平均药物数等指标上进行对比，GraphDiffMed在质量指标上优于对手，DDI率略高但可解释。

**⚠️ 局限性**

局限性包括仅在MIMIC‑III ICU上验证；药物粒度不含剂量/制剂；DDI信息仅为二元邻接矩阵；图偏置仅在访视级别，缺乏药物对级别；缺少外部验证与临床结果评估。

---

## 103. Pseudo-Formalization for Automatic Proof Verification

**arXiv ID:** 2605.20531 | [PDF](https://arxiv.org/pdf/2605.20531v1)

**作者:** Slim Barkallah `[一作]` (Stanford University), Tengyu Ma `[通讯]` (Stanford University)

**通讯引用:** 9403 | [OpenAlex ID](https://openalex.org/A5101821970)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了伪正式化（Pseudo-Formalization）与块验证（Block Verification）框架，用自然语言模块化构造可验证的证明；

**💡 创新点**

创新点在于在保持自然语言灵活性的同时，通过模块化与显式前提/结论划分，显著降低LLM验证的上下文消耗，实现更高精度的错误检测；

**🔧 技术方法**

使用LLM先将原始证明翻译为伪正式化格式，再对每个模块进行独立验证，并通过校准模型聚合结果；

**📊 数据集**

使用两个基准：①来自IMO/Putnam的AI生成证明数据集（200条）；②新构建的35篇包含已修正错误的arXiv数学论文数据集；

**📈 对比分析**

与直接LLM-as-judge基线相比，PF+BV在精确率-召回率曲线中占优，误报率下降、召回率提升，且在更长、更复杂的研究论文上表现更佳；

**⚠️ 局限性**

局限包括伪正式化翻译的难度、对数学证明的专属性、仅评估错误定位而非完整错误判定，以及可能的误报上限受限于作者仅标注已公开的错误。

---

## 104. SUGAR: A Scalable Human-Video-Driven Generalizable Humanoid Loco-Manipulation Learning Framework

**arXiv ID:** 2605.20373 | [PDF](https://arxiv.org/pdf/2605.20373v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 105. Oracle Supervision Transfers for Hyperparameter Prediction in Model-Based Image Denoising

**arXiv ID:** 2605.20479 | [PDF](https://arxiv.org/pdf/2605.20479v1)

**作者:** Jianmin Liao `[一作]` (Syracuse University), Yuesheng Xu `[通讯]` (Old Dominion University)

**通讯引用:** 5901 | [OpenAlex ID](https://openalex.org/A5026572582)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种单一的配置条件化预测器HyperDn，能够在不重新标注新配置的情况下，利用已标注的源配置超参数信息进行超参数预测，显著降低模型基图像去噪的标注成本。

**💡 创新点**

创新点在于将oracle超参数标签视为可跨配置迁移的监督信号，并通过共享图像编码器与配置条件化输出头实现多配置共享与个性化的超参数预测。

**🔧 技术方法**

使用ConvNeXtV2编码器、配置条件化多槽输出结构、槽级可微缩放偏置读出以及多任务损失对已标注的源配置进行训练，并在目标配置上进行少样本微调或零样本推断。

**📊 数据集**

实验数据集包括Waterloo Exploration Database（WED）256×256图像、STL10 96×96、Kodak24 512×768，以及多种噪声类型（高斯、泊松、冲击、混合噪声）。

**📈 对比分析**

与基线的单配置CNN、ImageNet预训练版本及Monte-Carlo SURE等方法比较，HyperDn在DiffPIR新配置上仅用2个oracle标签即可达到与oracle相差0.9 dB的PSNR，零样本时对混合噪声和跨分辨率目标的PSNR也仅落后0.2 dB。

**⚠️ 局限性**

限制在于只针对已知的模型基去噪框架，跨配置迁移需在源配置中已有相似超参数角色；对未知噪声组合或全新去噪模型的零样本效果尚未验证；训练阶段仍需耗费大量oracle标注时间。

---

## 106. Information Redistribution Under Reductions in NP Search

**arXiv ID:** 2605.20236 | [PDF](https://arxiv.org/pdf/2605.20236v1)

**作者:** Jing-Yuan Wei `[一作]` `[通讯]` (Zhejiang Yi-Neng Grid-Storage Energy Co. Ltd.), Jing-Yuan Wei (Zhejiang Yi-Neng Grid-Storage Energy Co. Ltd.)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文探讨了将结构化的 P-矩阵违规搜索问题通过经典 NP 完备性归约转化为 3-SAT 和 Subset Sum，并从信息可访问性的角度解读归约如何重分散隐藏线索。

**💡 创新点**

创新点在于把归约视为信息重分配机制而非单纯的计算变换，提出归约可提升局部可推理性但伴随表示扩张的“信息可访问性”框架。

**🔧 技术方法**

使用信息理论分析、Tseytin‑式归约、PPSZ、HGJ 等求解算法的时间空间复杂度比较。

**📊 数据集**

未使用真实数据集，采用小规模 N=6 的示例实例进行说明。

**📈 对比分析**

通过比较原始矩阵搜索与归约后 3‑SAT/Subset Sum 算法的时间复杂度与维度扩张，表明归约后虽增加维度但局部推理更强，时间指数系数略有提升。

**⚠️ 局限性**

局限在于讨论基于示例和理论分析，缺乏大规模实验验证，且归约带来的表示扩张与推理收益未被定量化。

---

## 107. It Takes Two: Complementary Self-Distillation for Contextual Integrity in LLMs

**arXiv ID:** 2605.20258 | [PDF](https://arxiv.org/pdf/2605.20258v1)

**作者:** Sangwoo Park `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种“SelfCI”框架，利用自监督的反馈生成器和两条互补的教师策略，使大型语言模型在保持任务完整性的同时实现上下文适当的隐私保护。

**💡 创新点**

创新点在于将信息抑制与任务解决分离，使用两条基于反馈的教师分别强化任务完整性与隐私最小化，并通过逆KL结合产生Product‑of‑Experts目标，直接逼近理想的上下文完整性状态。

**🔧 技术方法**

技术包括自监督的反馈模板（I_allow/I_disallow）生成任务相关理由、基于这些理由构造的两条教师分布、对两条教师的逆KL联合优化以及对教师分布的指数加权组合。

**📊 数据集**

实验使用了CI‑RL（合成隐私任务）、PrivacyLens（agentic 工具使用场景）和CIMemories（累积私有上下文）等数据集，并在多种指令调优与推理骨干上验证。

**📈 对比分析**

与基线CI‑RL（在线强化学习）和ContextDistill（离线教师蒸馏）以及初始模型对比，SelfCI在Integrity和Complete指标上提升显著，保持甚至提升Utility，且在大模型和跨域任务上表现更稳健、样本效率更高。

**⚠️ 局限性**

局限性包括依赖人工标注的合成属性数据、对小模型的上下文学习能力依赖较高、使用固定权重λ，且评估仅聚焦于最终响应，未深入分析推理过程中的信息泄露。

---

## 108. Quadratic Characterizations for Reachability Analysis of Neural Networks

**arXiv ID:** 2605.20482 | [PDF](https://arxiv.org/pdf/2605.20482v1)

**作者:** Elias Khalife `[一作]` (Virginia Tech), Pierre-Loic Garoche `[通讯]` (Federation ENAC ISAE-SUPAERO ONERA Universite de Toulouse)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并验证了针对二维静态关系的可验证二次约束，用于神经网络可达性与安全性分析。

**💡 创新点**

提出数据驱动候选生成与SOS验证相结合的框架，生成域相关二次约束，并针对ReLU网络引入块级重复约束与局部边界紧化。

**🔧 技术方法**

利用采样求解凸二次规划生成候选，再通过多项式或松弛多项式逼近配合Sum‑of‑Squares、Semidefinite Programming及多面体传播实现全局验证。

**📊 数据集**

对tanh、饱和函数、27维tanh控制器以及ACAS Xu 4层ReLU网络进行实验，比较了NNV、CORA、α,β‑CROWN等工具。

**📈 对比分析**

通过输出可达集宽度和安全椭圆体体积比较，得到相对NNV宽度约6.4%缩小，α,β‑CROWN约1.2%缩小；ReLU案例中COMB‑PP约93.1宽度是最小，优于EP、NNV等。

**⚠️ 局限性**

方法受二维域限制、SOS/SDP 计算复杂度高，且对大规模网络仍可能产生显著计算开销。

---

## 109. Lighting-aware Unified Model for Instance Segmentation

**arXiv ID:** 2605.20436 | [PDF](https://arxiv.org/pdf/2605.20436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 110. SDM: A Powerful Tool for Evaluating Model Robustness

**arXiv ID:** 2605.20308 | [PDF](https://arxiv.org/pdf/2605.20308v1)

**作者:** Xinlei Liu `[一作]` (Information Engineering University), Baolin Li `[通讯]` (Information Engineering University)

**通讯引用:** 4207 | [OpenAlex ID](https://openalex.org/A5100733202)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的基于梯度的攻击方法 SDM，用于生成更强的对抗样本。

**💡 创新点**

创新点是重新构造攻击目标为最大化非真标签概率上界与真标签概率之差，并通过三层“cycle-stage-step”顺序优化以及DPDR损失实现高效攻击。

**🔧 技术方法**

使用梯度上升、负概率损失、方向概率差比(DPDR)损失和多阶段迭代优化技术。

**📊 数据集**

在 CIFAR-10、CIFAR-100、Mini-ImageNet 以及 ImageNet-1K 等数据集上进行实验。

**📈 对比分析**

与 PGD、C&W、APGD 等现有方法比较，SDM 在攻击成功率上提升约 4-6%，并在计算成本上更具优势。

**⚠️ 局限性**

主要限制是超参数配置复杂，需要经验或自动化调优。

---

## 111. Latent Space Guided Scenario Sampling for Multimodal Segmentation Under Missing Modalities

**arXiv ID:** 2605.20372 | [PDF](https://arxiv.org/pdf/2605.20372v1)

**作者:** Irem Ulku `[一作]` (Ankara University), Erdem Akagündüz `[通讯]` (METU)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于预训练模型共享潜在空间的场景采样分布，用于在多模态语义分割中缺失模态时进行更有效的微调。

**💡 创新点**

创新点在于：①通过共享潜在空间的失真量度量每种缺失模态组合的重要性；②使用RBF核和RKHS正则化对这些重要性进行光滑，得到非均匀的采样概率分布；③将该分布替代传统的随机模态丢弃策略，显著提升在不同缺失模态场景下的性能。

**🔧 技术方法**

主要技术包括：预训练多模态分割模型、共享潜在空间失真计算、RBF核与正则化的核平滑（RKHS）求解、温度调节的softmax以及与均匀分布的混合采样；对比实验中还使用了LoRA参数高效微调。

**📊 数据集**

使用了三大遥感数据集：DSTL（光谱三模态）、Potsdam（RGB、IRRG、DSM混合模态）和Hunan（Sentinel‑1 SAR、Sentinel‑2 MSI、DEM）进行实验。

**📈 对比分析**

对比方法包括：①无微调（仅预训练+随机模态丢弃）；②随机模态丢弃微调；③LoRA微调；④本文提出的潜在空间导向场景采样微调。实验结果显示，在CBC‑SLP、CBC和CMX三种后端模型以及三大数据集上，本文方法在大多数缺失模态组合上都取得了更高的IoU和F1分数，且在全模态场景下性能保持不变，证明了其鲁棒性。

**⚠️ 局限性**

局限性：①实验仅覆盖二分类语义分割，尚未验证在多类别或更复杂任务中的效果；②对非常稀缺或不平衡的模态组合的泛化性仍需进一步评估；③需要预训练模型已具备良好共享潜在空间，若预训练质量欠佳，效果可能受限。

---

## 112. Measuring Decidability as Related to Busy Beaver Numbers

**arXiv ID:** 2605.20215 | [PDF](https://arxiv.org/pdf/2605.20215v1)

**作者:** Gurpreet Tandi `[一作]` (Bakersfield College), Jonathan Brown `[通讯]` (Bakersfield College)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

构造了两台Turing机，分别在停机与否与存在Brocard问题未知解或未知Fermat质数相对应；同时给出这些机的状态上界（72与43）。

**💡 创新点**

提出用Busy Beaver数为基准，将判定性与逻辑系统强度关联，并利用该框架给出判定复杂度的上界；此外设计了能在单一Turing机上完成质数检验与阶乘、平方判断的完整流程。

**🔧 技术方法**

基于传统的无限纸带Turing机模型，采用二进制或一进制表示、状态合并、递归计算、模运算与平方检测等算法实现。

**📊 数据集**

本工作未使用任何外部数据集，全部基于理论构造与证明。

**📈 对比分析**

方法通过计算状态数来衡量复杂度，但未给出运行时间或硬件实验；仅通过比较两台机的状态上界来说明其相对复杂度。

**⚠️ 局限性**

局限在于仅给出理论上限，未验证是否最优；Busy Beaver数本身不可计算，实际停机行为依赖未知命题；若命题无解，机器永不停止，无法得到实用结果。

---

## 113. Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents

**arXiv ID:** 2605.20616 | [PDF](https://arxiv.org/pdf/2605.20616v1)

**作者:** Chongrui Ye `[一作]` (University of Illinois Urbana-Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8028 | [OpenAlex ID](https://openalex.org/A5003491365)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Auto-Dreamer，一种两时标的语言代理记忆系统，分离快速会话写入和离线区域重写以实现跨会话记忆整合。

**💡 创新点**

创新点在于将记忆整合建模为区域重写（region rewriting）操作，并通过可证明的源轨迹进行支持；利用工具调用与 GRPO 训练的离线学习策略，实现可压缩、可抽象化的记忆重写。

**🔧 技术方法**

使用 GRPO 强化学习、工具调用（tool-use rollout）、冻结句子编码器、计数式 counterfactual utility 评价以及基于文本的 typed memory entries 与 provenance‑linked trajectories。

**📊 数据集**

主要训练使用 ScienceWorld 轨迹，评估数据集包括 ALFWorld、ScienceWorld 与 WebArena。

**📈 对比分析**

与十个基线（包括 Prompted、RL‑trained 与结构化存储方法）比较，Auto‑Dreamer 在连续记忆部署下取得最高成功率且活跃记忆占用显著更小；在固定银行实验中亦保持领先，提升 2–7 个点。

**⚠️ 局限性**

局限在于抽象化过程中可能丢失局部细节导致任务失败；对记忆区域选择与合成规模的超参数敏感；目前仅处理文本轨迹，未覆盖多模态场景。

---

## 114. Puzzled By ChatGPT? No more! A Jigsaw Puzzle to Promote AI Literacy and Awareness

**arXiv ID:** 2605.20404 | [PDF](https://arxiv.org/pdf/2605.20404v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 115. Programmable Participatory Governance -- A Formal Framework for Transparent, Accountable, and Citizen-Responsive Democratic Systems: From Deliberative Theory to Decentralised Architecture

**arXiv ID:** 2605.20261 | [PDF](https://arxiv.org/pdf/2605.20261v1)

**作者:** Sergio Montenegro `[一作]` `[通讯]` (Independent Researcher), Sergio Montenegro (Independent Researcher)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了Programmable Participatory Governance（PPG）框架，定义了从提案到执行、投票、否决的完整治理管道；

**💡 创新点**

核心创新包括动态法定人数机制、综合合法性评分、透明默认与财务透明账本、民事参与强制、荣誉与授权双重公共角色问责、混合身份模型（社会图验证+零知识+Soulbound Token）以及自我进化的治理参数；

**🔧 技术方法**

采用区块链智能合约实现自执法；零知识证明和Soulbound Token保障身份隐私与Sybil防护；形式化状态机与博弈论模型用于阐释决策逻辑与操纵抗性；并利用模拟和实验代码验证动态法定人数与否决机制；

**📊 数据集**

主要数据来源为文献与公开统计（OECD、IDEA、US政策案例等），并通过模拟生成的三类行为主体（被动、积极、战略）数据进行验证；公开代码与仿真结果已托管在GitHub；

**📈 对比分析**

通过对比现有civic‑tech平台（vTaiwan、Better Reykjavik、Decidim）与DAO治理系统的四项关键属性（形式化、匿名投票、财务账本、动态法定人数），展示PPG在所有属性上实现完备；模拟实验表明动态法定人数提升参与率同时保持决策吞吐；博弈论证明在合理阈值下小规模协同无法盈利；总体性能可接受，具可扩展性；

**⚠️ 局限性**

局限性包括身份验证仍不完备（需要外部信任来源、密钥管理风险）；模拟为简化模型，未考虑社交网络、学习与情境适应；博弈论假设可能过度保守；论证难以保证大规模讨论质量；法律与宪法集成仍待实践；数字鸿沟与资源不均衡问题；DAO与传统行政机构对接仍处于技术成熟度不足的阶段；

---

## 116. Capability $\neq$ Interpretability: Human Interpretability of Vision Foundation Models

**arXiv ID:** 2605.20337 | [PDF](https://arxiv.org/pdf/2605.20337v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 117. Augmented Analytics and Decision Quality: The Role of Trust among Non-Technical BI Users

**arXiv ID:** 2605.20198 | [PDF](https://arxiv.org/pdf/2605.20198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 118. Art Card Game (ACG): Embedding Illustration in Gameplay to Mitigate Artist Self-Criticism

**arXiv ID:** 2605.20465 | [PDF](https://arxiv.org/pdf/2605.20465v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 119. An $O(n^5)$-Time Algorithm for Optimal Broadcast Domination

**arXiv ID:** 2605.20526 | [PDF](https://arxiv.org/pdf/2605.20526v1)

**作者:** Kleitos Papadopoulos `[一作]` `[通讯]`, Kleitos Papadopoulos

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种 O(n³) 时间的路径广播支配算法，并通过分割技巧实现了 O(n⁵) 时间的最优广播支配算法

**💡 创新点**

核心创新在于构造单一有向无环图（状态图），消除锚点循环，利用前向动态规划求解路径支配问题

**🔧 技术方法**

使用距离预处理、并查集维护残余连通分量、状态图构造与最短路径求解、以及动态规划重构广播方案等技术

**📊 数据集**

实验采用七类图集：路径、环、随机树、稀疏连通 ER 图、杠杆、星形和轮形图

**📈 对比分析**

与原 Heggernes‑Lokshtanov 基线相比，在路径、环等结构复杂实例上实现了 2.3–8.4 倍加速，且两者得到相同最优结果

**⚠️ 局限性**

局限性包括对环形情况仍需 O(n⁴) 处理，且实验实现为 Python，常数因子尚未进一步优化

---

## 120. Codec-Robust Attacks on Audio LLMs

**arXiv ID:** 2605.20519 | [PDF](https://arxiv.org/pdf/2605.20519v1)

**作者:** Jaechul Roh `[一作]` (University of Massachusetts Amherst), Amir Houmansdar `[通讯]` (University of Massachusetts Amherst)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研发并评估了一种在神经音频编码器（EnCodec）的连续潜在空间中进行的对抗攻击，能够在多比特率的压缩通道下注入目标文本到 Audio LLM。

**💡 创新点**

创新点在于：① 将攻击表面从波形域转移到编码器自己的潜在空间，利用压缩器保持该空间信息的特性；② 通过多比特率的 straight‑through EoT 训练，使攻击在实际 Opus、MP3、AAC‑LC 等压缩下保持鲁棒性；③ 证明 lossy 压缩并非可靠防御，揭示潜在空间攻击的实用威胁。

**🔧 技术方法**

采用技术包括 EnCodec 的连续潜在编码、PGD 与 Adam 优化、Expectation‑over‑Transformation（EoT）采样多比特率、straight‑through estimator（STE）对 Opus 的近似、以及跨模型（Qwen2‑Audio‑7B‑Instruct、Audio Flamingo 3、Qwen2.5‑Omni）的评测。

**📊 数据集**

使用了多种语音与音乐载体：金融客服、面试筛选、音乐版权检测等三类部署场景，包含约120条不同语言/风格的音频样本，目标是三种不同的攻击文本。

**📈 对比分析**

方法比较：将潜在空间攻击与同样训练条件下的波形域攻击基线对比，保持相同 SNR 与 EoT 训练；在 Opus 128 kbps 下潜在攻击平均 85.5% 的目标子串成功率（ASR），波形基线仅 26%；在未见过的 MP3 与 AAC‑LC 上转移性能高达 100% 与 84% 的 ASR。

**⚠️ 局限性**

局限性：① 目前仅针对单一目标模型有效，跨模型迁移需重新优化；② 缺乏针对潜在空间攻击的专门防御策略（如对抗训练、跨编码器检测或输入随机化）；③ 对于极低比特率（≤32 kbps）及更复杂的目标文本，攻击成功率显著下降。

---

## 121. A Multi-Layer Testing Framework for Automated Data Quality Assurance in Cloud-Native ELT Pipelines

**arXiv ID:** 2605.20500 | [PDF](https://arxiv.org/pdf/2605.20500v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 122. VBT-MPC: Vision-Based Tactile MPC for Contour Following

**arXiv ID:** 2605.20392 | [PDF](https://arxiv.org/pdf/2605.20392v1)

**作者:** Edison Velasco-Sanchez `[一作]` (University of Alicante), Pablo Gil `[通讯]` (University of Alicante)

**通讯引用:** 1829 | [OpenAlex ID](https://openalex.org/A5022395891)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了基于视觉触觉传感器的模型预测控制框架，实现了在眼内手配置下的触觉轮廓跟踪。

**💡 创新点**

在触觉图像域直接以轮廓特征为目标进行MPC控制，避免了姿态估计和力控制模块；同时提出了融合分割、线拟合与EKF的轮廓特征提取流水线。

**🔧 技术方法**

使用Markerless VBTS（GelSight Mini）、U-Net+++ResNet18分割网络、加权最小二乘线拟合、扩展卡尔曼滤波、ACADOS/CasADi求解MPC、ROS2仿真与真实实验。

**📊 数据集**

构建了约8000张GelSight Mini触觉图像的轮廓数据集，并在实验中测试了3D打印轮廓与多种真实物体。

**📈 对比分析**

将VBT‑MPC与经典IBVS、分离式视觉触觉伺服两种基线进行对比，在仿真与真实实验中，VBT‑MPC在保持接触、跟踪误差（r、δ、α、β）及实时性（≈20 ms）上显著优于基线，误差低于4 mm，角度误差<0.05 rad。

**⚠️ 局限性**

对轮廓可辨识度低、曲率突变或软体物体导致特征提取不稳定时性能下降；未实现对正向法向力的显式调节。

---

## 123. A Survey of Large Audio Language Models: Generalization, Trustworthiness, and Outlook

**arXiv ID:** 2605.20266 | [PDF](https://arxiv.org/pdf/2605.20266v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 124. Evaluating multimodal emotion recognition in proactive conversational agents: A user study

**arXiv ID:** 2605.20200 | [PDF](https://arxiv.org/pdf/2605.20200v1)

**作者:** Adnana Dragut `[一作]` (Universidad de Zaragoza), Jose M. Buades-Rubio `[通讯]` (Universitat de les Illes Balears)

**通讯引用:** 362 | [OpenAlex ID](https://openalex.org/A5045452339)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究开发并评估了一个集成面部识别与生成式AI文本情感分析的多模态情绪识别模块，嵌入到主动式社交交互代理（SIA）中；

**💡 创新点**

创新点在于将生成式AI的语义情绪分析与传统面部表情识别相结合，并在真实用户对话中验证其对“扑克面”现象的解释；

**🔧 技术方法**

技术包括计算机视觉的SilNet面部表情识别模块、OpenAI GPT‑4/ChatGPT的情绪语义分析模块、以及自定义对话生成与情绪检测提示；

**📊 数据集**

数据集：20名参与者在实验室进行两轮约5分钟的非脚本化对话，收集面部视频、语音转文本、用户自评情绪；

**📈 对比分析**

比较方法：将面部识别结果与文本分析结果与用户自评对比，得到面部识别准确率23.8%，文本分析准确率45%；表现显示文本分析远优于视觉识别；

**⚠️ 局限性**

局限包括样本量小、实验环境受控导致扑克面效应夸大、仅使用面部与文本两模态，缺乏声音语调与身体姿态等信息，且文本分析对语义歧义仍易误判。

---

## 125. Causal Unlearning in Collaborative Optimization: Exact and Approximate Influence Reversal under Adversarial Contributions

**arXiv ID:** 2605.20341 | [PDF](https://arxiv.org/pdf/2605.20341v1)

**作者:** Ali Mahdavi `[一作]` (Islamic Azad University), Omid Kashefi `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种Federated Unlearning方法HF‑KCU，利用Krylov子空间近似影响函数实现高效删除客户端数据。

**💡 创新点**

引入无Hessian矩阵的CG迭代与因果加权机制，显著降低复杂度并保证只有持有删除数据的客户端被更新。

**🔧 技术方法**

采用Hessian‑free Krylov子空间近似、共轭梯度、自动微分求HVP、因果权重、阻尼调节和自适应缩放等技术。

**📊 数据集**

在CIFAR‑10、MNIST、Fashion‑MNIST等数据集，并在ResNet‑18、ViT‑Lite等模型架构上进行实验。

**📈 对比分析**

与全量重训练、FedEraser、SISA、NaiveGA等方法对比，CIFAR‑10上实现约47.7×速度提升、0.60%准确率损失、MIA成功率与重训练相当；在后门攻击场景实现约430×速度提升、ASR下降15.4%。

**⚠️ 局限性**

方法依赖有界对抗扰动、O(kd)内存占用对极大模型受限，且CF度量在高准确率下不稳定。

---

## 126. A strongly annotated passive acoustic dataset for tropical bird monitoring

**arXiv ID:** 2605.20578 | [PDF](https://arxiv.org/pdf/2605.20578v1)

**作者:** Daniela Ruiz `[一作]` (Microsoft AI for Good Research Lab), Juan M. Lavista Ferres `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并公开了PteroSet数据集，对哥伦比亚热带鸟类录音进行精确时频注释，并提供COCO风格JSON格式。

**💡 创新点**

首次在热带生态系统提供规模化、强标注的数据，展示了声音重叠和跨场域差异带来的挑战，并提供了基线模型。

**🔧 技术方法**

使用音频时序窗切、mel谱图、ResNet-18深度学习二分类模型进行鸟类检测。

**📊 数据集**

基于2023-2025年在Puerto Asís（Putumayo）和Pivijay（Magdalena）录得的563段音频，涵盖168种鸟类。

**📈 对比分析**

采用留一项目交叉验证，对5折平均F1≈0.72、AUPRC≈0.80、准确率≈0.85进行评估，表现良好但存在场域差异影响。

**⚠️ 局限性**

局限在于注释不均衡、低频弱声难检、跨站域迁移差异、以及时段离散录音结构导致模型假设挑战。

---

## 127. Fault-Tolerant, Rigidity-Preserving Control of Inflatable Truss Robots

**arXiv ID:** 2605.20561 | [PDF](https://arxiv.org/pdf/2605.20561v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 128. \ECUAS{n}: A family of metrics for principled evaluation of uncertainty-augmented systems

**arXiv ID:** 2605.20490 | [PDF](https://arxiv.org/pdf/2605.20490v1)

**作者:** Lautaro Estienne `[一作]` (Universidad de Buenos Aires), Luciana Ferrer `[通讯]` (Universidad de Buenos Aires)

**通讯引用:** 7891 | [OpenAlex ID](https://openalex.org/A5056267912)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一类基于决策理论的评估指标（n指标），用于同时衡量不确定性增强（UA）系统的预测质量与不确定性质量，并通过该指标对分类与生成模型进行统一评估。

**💡 创新点**

创新点在于：① 将UA系统的评价建模为无拒绝成本下的Bayes决策成本积分，得到的指标天然为正则化评分规则（PSR），能奖励可解释为概率的不确定性；② 通过参数n控制对错误预测与不确定性误差的权衡，满足不同高风险应用需求；③ 通过理论推导给出生成任务中不确定性应基于语义等价类的原因，解释了先前经验观察。

**🔧 技术方法**

使用了统计决策理论、贝叶斯决策、正则化评分规则（如Brier、交叉熵）以及积分权重技术；在实现上实现了对多种UA系统的统一接口；采用了后验校准、温度缩放等校准方法。

**📊 数据集**

在分类实验中使用了 ImageNet、CIFAR-10/100、AGNews、FVCAUS 等数据集；在生成实验中使用了手工标注的 TriviaQA 子集（455条）以及 MMLU；并对多种语言模型（Qwen3.5‑9B、Mini‑T‑3‑8B‑Instruct‑2512、Gemini‑2.5‑Flash 等）进行评估。

**📈 对比分析**

与传统评估（Accuracy/ER、AUC、ECE、Brier、Cross‑Entropy、AURC）比较，n指标能够同时反映预测准确率与不确定性校准；在高风险场景下，n=0 使误判置信度高的样本受到严重惩罚，排名与传统指标显著不同；实验表明在分类任务中 n 指标能区分校准后模型的差异，在生成任务中能更好区分不同置信度生成策略的整体表现。

**⚠️ 局限性**

局限性在于：① 假设最终用户遵循决策理论并愿意使用可解释概率不确定性；② 若用户仅根据经验设定拒绝阈值且不关心不确定性可解释性，n指标可能不合适；③ 对于非常大或未知的等价类数量（K→∞）的生成任务，计算中需要做近似或假设，可能影响指标精度。

---

## 129. FusionCell: Cross-Attentive Fusion of Layout Geometry and Netlist Topology for Standard-Cell Performance Prediction

**arXiv ID:** 2605.20287 | [PDF](https://arxiv.org/pdf/2605.20287v1)

**作者:** Haoyi Zhang `[一作]` (Peking University), Runsheng Wang `[通讯]` (Peking University)

**通讯引用:** 7119 | [OpenAlex ID](https://openalex.org/A5002760019)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种双模态标准单元性能预测模型 FusionCell，能够在毫秒级完成对 7nm 标准单元延迟和功率的预测。

**💡 创新点**

首次采用拓扑引导的交叉注意力，将网表拓扑作为查询引导布局几何特征，实现网表与布局几何的显式对齐；同时结合 DeiT 视觉 Transformer 与图 Transformer 的异构图编码。

**🔧 技术方法**

使用 DeiT 视觉 Transformer、图 Transformer、跨模态交叉注意力、异构设备‑网图建模、R/C 预处理、标准化损失等技术。

**📊 数据集**

基于 ASAP7 7nm FinFET PDK 自动生成的 19.5k 单元库，覆盖 149 种单元类型，包含六个性能指标（上升/下降延迟、上升/下降过渡、上升/下降功率）。

**📈 对比分析**

与 Vision‑only、Late Fusion、Symmetrical Fusion 及 ProtoCellLayout 等基线对比，使用 MAPE、R²、Spearman/Kendall 评估；FusionCell 在所有六个指标上平均 MAPE 0.92%，R² 0.977，Spearman ρ≈0.86，显著优于基线。

**⚠️ 局限性**

仅针对已知功能单元的布局变体，缺乏跨功能或跨工艺节点的泛化；对更复杂工艺规则的鲁棒性待验证；仍需大量标注数据支持。

---

## 130. SURF: Steering the Scalarization Weight to Uniformly Traverse the Pareto Front

**arXiv ID:** 2605.20619 | [PDF](https://arxiv.org/pdf/2605.20619v1)

**作者:** Liuyuan Jiang `[一作]` (University of Rochester), Lisha Chen `[通讯]` (University of Rochester)

**通讯引用:** 4020 | [OpenAlex ID](https://openalex.org/A5091442724)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的权重采样方法SURF，使线性标量化在多目标优化中能够均匀覆盖帕累托前沿。

**💡 创新点**

通过几何分析构建PF弧长累积分布函数，并利用其逆映射实现权重的自适应采样；在结构化问题给出闭式解，在一般问题给出迭代CDF重建算法，并证明收敛到有限采样上限。

**🔧 技术方法**

使用线性标量化、弧长测度、CDF推断、闭式解析、迭代重构以及投影梯度/Adam/PPO等内部求解器。

**📊 数据集**

在Bandit、MO‑Gymnasium（DST、Fishwood、MO‑Mountaincar）以及Reddit摘要的LLM对齐任务等数据集上进行实验。

**📈 对比分析**

与均匀权重、OLS、UMOD、Reward Soup等基线进行比较，评估指标包括Hypervolume、IGD、CV和Gap Ratio；SURF在所有任务中显著提升均匀性，并在HF、IGD等方面保持或提升性能。

**⚠️ 局限性**

对非凸问题仍需隐式强凸假设，内部求解器需可收敛；在高维或大规模设置下需多次外循环，且对超参数（N,α）敏感。

---

## 131. TriForces: Augmenting Atomistic GNNs for Transferable Representations

**arXiv ID:** 2605.20581 | [PDF](https://arxiv.org/pdf/2605.20581v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 132. AgentAtlas: Beyond Outcome Leaderboards for LLM Agents

**arXiv ID:** 2605.20530 | [PDF](https://arxiv.org/pdf/2605.20530v1)

**作者:** Parsa Mazaheri `[一作]` (University of California, Santa Cruz), Kasra Mazaheri `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AgentAtlas——一种统一的代理行为分类与测评框架，用于评估 LLM 代理的控制决策与轨迹失败，而非仅关注最终任务成功。

**💡 创新点**

创新点在于六态控制决策分类、九类轨迹失败拓展及其两层层级标签，配合对 15 个现有基准的覆盖审计，以及对 prompt 格式与评估轴对模型排名影响的实证演示。

**🔧 技术方法**

技术方法包括基于闭集标签菜单的 taxonomy‑aware 及 taxonomy‑blind 评估策略、生成式任务与轨迹数据、以及多模型（封闭与开源）在固定合成数据集上的逐项评测。

**📊 数据集**

使用由 Claude Opus 4.7 生成的 1,342 条合成样本，涵盖 Control（684 条）、Trajectory（400 条）与 Security（258 条）三大子集，按六态控制门和九类轨迹错误标注。

**📈 对比分析**

对比方法：在 taxonomy‑aware 与 taxonomy‑blind 两种 prompt 模式下，评估八个模型在控制准确率、轨迹诊断准确率和工具‑上下文效用保持率上的表现；发现 prompt 规范化能压缩模型差异，且不同评估轴排名不一致，说明单一指标无法全面衡量代理能力。

**⚠️ 局限性**

局限性包括：数据集由单一模型生成可能偏向其偏好；缺乏人工验证的校准子集；安全与轨迹标注的主观性与细粒度不足；以及评估只关注固定任务而未覆盖更广泛的真实世界情境。

---

## 133. EPC-3D-Diff: Equivariant Physics Consistent Conditional 3D Latent Diffusion for CBCT to CT Synthesis

**arXiv ID:** 2605.20470 | [PDF](https://arxiv.org/pdf/2605.20470v1)

**作者:** Alzahra Altalib `[一作]` (University of Dundee), Alessandro Perelli `[通讯]` (University of Glasgow)

**通讯引用:** 599 | [OpenAlex ID](https://openalex.org/A5074333189)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种名为EPC‑3D‑Diff的条件三维潜在扩散框架，用于将CBCT图像合成高质量的sCT图像，并在训练过程中引入投影域等变损失以保持物理一致性。

**💡 创新点**

核心创新点在于：①将CT前向模型的旋转等变性嵌入扩散过程，强制生成的sCT在投影域中满足角度位移一致性；②在三维潜在空间进行条件扩散，既保持了体积上下文，又降低了计算负担；③结合了物理一致性约束与图像域L1/边缘/拉普拉斯损失，提升HU精度与结构保真。

**🔧 技术方法**

使用了三维自编码器（3D Residual Block）、条件3D U‑Net、DDPM/DDIM采样、投影等变损失、物理前向/后投影操作以及多项图像域正则化（L1、边缘、拉普拉斯）。

**📊 数据集**

在两组数据集上进行评估：NWH 10例头颈部phantom（8例训练/2例测试）以及JUST 14例临床数据（11例训练/3例测试），对每例切片做对齐、归一化后统一裁剪至256×256。

**📈 对比分析**

与CycleGAN和无等变条件扩散（C‑DDPM）在单域和混合域训练场景下进行对比；EPC‑3D‑Diff在phantom数据上PSNR提升约+7.4 dB、SSIM逼近0.99；在临床数据上PSNR提升约+1.8 dB，且HU曲线与真实CT高度一致，显示对扫描仪几何差异的鲁棒性。

**⚠️ 局限性**

限制主要包括：训练时需进行一次前向投影，导致训练时间略增（约15%）；对不同体位或部位的数据迁移尚未充分验证；模型在大规模多模态数据上的泛化能力仍待进一步评估。

---

## 134. Can Conversational XAI Improve User Performance? An Experimental Study

**arXiv ID:** 2605.20439 | [PDF](https://arxiv.org/pdf/2605.20439v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 135. End-to-End Unmixing with Material Prompts for Hyperspectral Object Tracking

**arXiv ID:** 2605.20569 | [PDF](https://arxiv.org/pdf/2605.20569v1)

**作者:** Xu Han `[一作]` (Griffith University), Jun Zhou `[通讯]` (Griffith University)

**通讯引用:** 19670 | [OpenAlex ID](https://openalex.org/A5100781212)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种端到端的高光谱目标跟踪框架E2E-MPT，联合优化材料解混（hyperspectral unmixing）和目标定位，利用材料提示（material prompt）直接增强跟踪器的判别能力。

**💡 创新点**

创新点包括：① 将解混过程嵌入跟踪网络，实现两者的协同优化；② 设计材料表示分解模块（MRDM）和双分支小波增强材料提示模块（DWMPM），将丰度图分解为低频/高频分支；③ 引入目标导向的解混损失，将解混目标与定位精度对齐；④ 采用频率提示融合模块（FPFM）在潜在空间融合低高频提示，提升鲁棒性。

**🔧 技术方法**

使用深度自编码器解混网络（如DAEU、CNNAEU）、1D Haar小波变换、跨注意力与卷积混合的WMP块、Transformer骨干（OSTrack）以及基于提示学习的轻量级可训练模块；训练时采用联合损失（跟踪+解混）。

**📊 数据集**

在三个标准高光谱跟踪基准集上评估：HOTC2020、HOTC2023和HOTC2024，涵盖VIS、NIR、RedNIR三种光谱域。

**📈 对比分析**

与多种RGB与高光谱跟踪器（如SeqTrack、SMAT、AQATrack、SSTtrack、DaSSP-Net、MMF等）对比，E2E-MPT在DP和AUC上均取得最高成绩，例如在HOTC2020上达0.966/0.734（DP/AUC），相较于最强对手提升约1–3%/2–5%；在HOTC2023、HOTC2024亦稳步领先，证明端到端材料提示策略显著提升跟踪性能。

**⚠️ 局限性**

局限性包括：① 解混网络的性能受训练数据覆盖范围限制，未见材料的解混质量下降会影响跟踪；② 未引入时序信息，易在长时间遮挡或极端照明变化时失效；③ 对几何变形（旋转、尺度变化）仍不够鲁棒，需进一步改进。

---

## 136. Leveraging Vision-Language Models to Detect Attention in Educational Videos

**arXiv ID:** 2605.20211 | [PDF](https://arxiv.org/pdf/2605.20211v1)

**作者:** Gabriel Becquet `[一作]` (Sorbonne University), Ali Abou-Hassan `[通讯]` (Sorbonne University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用Vision‑Language模型Gemini 3在叠加眼动轨迹的教育视频中进行无训练的注意力检测。

**💡 创新点**

首次将视觉提示与VLM结合，并通过自然语言解释提供可解释性，避免传统特征工程。

**🔧 技术方法**

Gemini 3 VLM、视频+红色圆点视觉提示以及Zero‑shot、Few‑shot、启发式Chain‑of‑Thought Prompting。

**📊 数据集**

使用Lallé等人公开的Tobii Nano眼动数据集（N=70，7 分钟绿色化学课程）。

**📈 对比分析**

与统计基线比较，零‑shot启发式策略在宏观召回/精度上略优，但总体准确率仍低于多数类基线（≈58–60 % vs 80.1 %）。

**⚠️ 局限性**

缺乏任务特定微调导致难以捕捉细粒度认知状态，性能受类不平衡和VLM对时间动态理解不足限制。

---

## 137. What Do Agents Communicate? Characterizing Information Exchange in Multi-Agent Systems

**arXiv ID:** 2605.20548 | [PDF](https://arxiv.org/pdf/2605.20548v1)

**作者:** Yong Jin Chun `[一作]` (University of California, Irvine), Iftekhar Ahmed `[通讯]` (University of California, Irvine)

**通讯引用:** 1474 | [OpenAlex ID](https://openalex.org/A5078115464)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统分析多代理系统中信息交流的类别，并提出基于信息类别的恢复增强方法（CARA），在失败案例中实现高达86%的恢复率。

**💡 创新点**

首次对MA系统中交互信息进行系统化分类，利用遮蔽分析量化各类别对任务性能的影响，并基于此设计了可在不同MA架构中通用的类别约束恢复策略。

**🔧 技术方法**

采用遮蔽分析（leave‑one‑out）、LLM‑as‑a‑Judge自动注释、以及多轮对话式增强（CARA）等技术。

**📊 数据集**

六个常用数据集（GSM8K、MATH500、MMLU、StrategyQA、CRUXEval、LiveCodeBench）并使用两种开源LLM（Qwen2.5‑32B‑Instruct 与 Qwen2.5‑Coder‑32B）。

**📈 对比分析**

在五种MA架构（Seq‑U、Seq‑R、Debate、CR‑MC、CR‑SV）下与两种模型进行基准比较，遮蔽某些信息类别可导致准确率变化幅度在±18%之间，CARA在多数任务与系统中将失败率降低至15%以下。

**⚠️ 局限性**

仅限两种开源模型与六个数据集，类别划分可能不完整，对提示设计敏感，未考虑自适应通信策略或跨任务的泛化能力。

---

## 138. WildRoadBench: A Wild Aerial Road-Damage Grounding Benchmark for Vision-Language Models and Autonomous Agents

**arXiv ID:** 2605.20306 | [PDF](https://arxiv.org/pdf/2605.20306v1)

**作者:** Bingnan Liu `[一作]` (University of Electronic Science and Technology of China), An Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 15143 | [OpenAlex ID](https://openalex.org/A5100419830)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套野外无人机道路损伤定位基准，评估固定视觉-语言模型与基于LLM的自主研究代理在同一任务上的表现

**💡 创新点**

首次将视觉定位与LLM驱动的端到端检测开发流程统一到同一评测框架，形成两条互补的评测轨道

**🔧 技术方法**

采用大规模视觉-语言模型（Gemini、Qwen系列等）进行零样本定位，以及使用ReAct式LLM代理进行数据搜索、模型微调、训练与推理代码生成

**📊 数据集**

基于1,061张专业标注的无人机航拍道路损伤图像，包含1,699个目标框，涵盖水渍、裂缝、坑洞、碎石及基础设施异常五大类别

**📈 对比分析**

对25个VLM和15个LLM代理进行统一的COCO-AP_50评测；Gemini 3 Pro在单次推断中达到42.1% AP_50，开源模型最高仅25.5%；LLM代理最多可提交5次，Claude Opus 4.7在5小时内实现17.76% mAP_50，整体性能仍远低于理想

**⚠️ 局限性**

小目标检测效果差、模型对细粒度定位缺乏精准度、LLM代理受限于预算与工具调用、数据泄漏风险及评测流程对小样本的鲁棒性不足

---

## 139. Lean Refactor: Multi-Objective Controllable Proof Optimization via Agentic Strategy Search

**arXiv ID:** 2605.20244 | [PDF](https://arxiv.org/pdf/2605.20244v1)

**作者:** Jialin Lu `[一作]` (Simon Fraser University), Wuyang Chen `[通讯]` (Simon Fraser University)

**通讯引用:** 2875 | [OpenAlex ID](https://openalex.org/A5013562664)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Lean Refactor，一种检索增强的可插拔框架，利用冻结的 LLM 通过策略库实现多目标、可控、版本鲁棒的 Lean 证明重构。

**💡 创新点**

创新点在于将重构知识外部化为带有编译时间与版本兼容性标注的策略库，并通过检索规则实现不同目标的动态切换，而无需模型微调。

**🔧 技术方法**

采用检索增强代理架构、向量检索、目标重排序、编译器反馈循环以及 Frozen LLM（Gemini、Claude、GPT）作为 Planner/Refactor/Debugger。

**📊 数据集**

使用约 200K 份长短证明对（来自 Mathlib、NuminaMath、FineLeanCorpus 等）构建了 9K 条策略，并在 miniF2F、PutnamBench、Putnam2025 以及多门类研究库（Analysis、FLT、PFR、PhysLean 等）进行评估。

**📈 对比分析**

与 ProofOptimizer、Claude Code 等基线对比，Lean Refactor 在竞赛数据上实现 70%+ 证明长度压缩、60% 编译时间缩减，且在版本迁移上保持更高的成功率。

**⚠️ 局限性**

局限在于跨版本评测仅覆盖 PutnamBench，策略级元数据采用保守聚类，且对极端版本兼容性和大规模项目的普适性尚未充分验证。

---

## 140. DEL: Digit Entropy Loss for Numerical Learning of Large Language Models

**arXiv ID:** 2605.20369 | [PDF](https://arxiv.org/pdf/2605.20369v1)

**作者:** Zhaohui Zheng `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 108133 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Digit Entropy Loss (DEL)，并在LLM中实现了浮点数优化，提升了数值预测性能。

**💡 创新点**

从熵最小化角度重新设计数值学习，去除手工数值距离项，引入二元交叉熵与数字条件概率乘积，并实现全局浮点数的学习。

**🔧 技术方法**

基于自回归LLM的交叉熵与二元交叉熵损失，改进的数字熵损失及浮点数位置权重，应用于CodeLlama、Mistral、DeepSeek、Qwen‑2.5等模型。

**📊 数据集**

使用MathInstruct大规模数学指令数据集进行微调，并在GSM8K、MATH、SVAMP、SimulEq、AQuA、SAT‑Math、MMLU七个数学推理基准上评估。

**📈 对比分析**

与MLE、MixCE、EMO、NTL、DIST^2Loss等基线对比，DEL在所有基准平均提升0.5–1.5%准确率，并在数值误差上显著降低，表现最优。

**⚠️ 局限性**

仍按token级优化，未考虑全局数值语义；未扩展对算术运算等更复杂数值推理的建模。

---

## 141. A Semantic-Web Oriented Competency Model for Engineering Programs

**arXiv ID:** 2605.20401 | [PDF](https://arxiv.org/pdf/2605.20401v1)

**作者:** Nicolas Evain `[一作]` (Universite de Pau et des Pays de l'Adour), Philippe Arnould `[通讯]` (Universite de Pau et des Pays de l'Adour)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实施了一套将计算机领域BoK映射到以能力为中心的课程框架的方法，并在ISANUM五年制工程课程中进行验证。

**💡 创新点**

创新点在于将BoK知识点与课程能力显式关联，构建基于本体的可协同维护的Semantic MediaWiki（ISANUMpedia），并通过必修工作实践与多条专业路径实现理论与实践的深度融合。

**🔧 技术方法**

所用技术包括本体建模、Semantic MediaWiki、Bloom认知层级映射、e-portfolio自评系统以及与行业合作的工作实习平台。

**📊 数据集**

使用的数据集为来自《Computing Curricula 2020》的34个计算知识领域下的494个BoK主题，以及基于这些主题制定的23个课程能力。

**📈 对比分析**

通过对首批23名学生的课程完成度、e-portfolio提交量（931条成就页）、修订次数（3,747次）等指标进行量化评估，展示了模型的可操作性和学生参与度，但未进行跨课程或跨校对比，性能主要以实施效果和可持续维护性评估。

**⚠️ 局限性**

局限性包括：映射与本体构建仍需人工投入，BoK与行业需求快速演进时需持续更新，缺乏长期跟踪评估数据，以及对AI辅助自动化维护的探索尚未成熟。

---

## 142. Divide-Prompt-Refine: a Training-Free, Structure-Aware Framework for Biomedical Abstract Generation

**arXiv ID:** 2605.20628 | [PDF](https://arxiv.org/pdf/2605.20628v1)

**作者:** Sylvey Lin `[一作]` (University of Illinois Urbana-Champaign), Halil Kilicoglu `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 3226 | [OpenAlex ID](https://openalex.org/A5016571803)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种零训练、结构感知的分解-聚合框架DPR-BAG，用以从完整医学论文生成缺失摘要。

**💡 创新点**

采用分面拆解、并行LLM摘要、后期全局润色的无监督方法，并发现简化提示比复杂提示更能保持事实一致。

**🔧 技术方法**

使用LLM（instruction‑tuned）、First Sentence Labeling拆分、TR‑UMLS和CoT实体引导，以及零样本提示。

**📊 数据集**

构建PMC‑MAD（46,309篇缺失摘要的PubMed全文）并在PubMed Summarization上验证。

**📈 对比分析**

与LED（arXiv/PubMed）和LongT5的零样本及微调版本对比，DPR‑BAG在抽象度与事实一致性上均优于微调模型，且保持无训练优势。

**⚠️ 局限性**

评估依赖自动指标，BOMRC结构对非标准文章适用性有限，摘要压缩过度可能漏掉辅助信息。

---

## 143. Self-Training Doesn't Flatten Language -- It Restructures It: Surface Markers Amplify While Deep Syntax Dies

**arXiv ID:** 2605.20602 | [PDF](https://arxiv.org/pdf/2605.20602v1)

**作者:** Ming Liu `[一作]` (Amazon), Ming Liu `[通讯]` (Amazon)

**通讯引用:** 1006 | [OpenAlex ID](https://openalex.org/A5102008384)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多代自我训练过程中对不同规模、不同架构的 LLM 进行实验，追踪 17 个预先选定的语言特征（按结构深度分层）随代数的变化，提出并验证结构深度假设（SDH）：结构深度决定特征衰减速率，表面特征放大而深层结构坍塌。

**💡 创新点**

创新点在于：①首次从结构深度视角解释 LLM 自我训练导致的语言失真，②揭示“表面复杂度悖论”——聚合复杂度指标上升但句法结构衰退；③提供多模型、多层级证据，验证深度而非频率是主导预测因子。

**🔧 技术方法**

使用自回归模型（GPT‑2、Pythia、OPT）自我训练循环，采样 3,000 文本/代；通过正则表达式与依存句法解析提取 17 个特征；采用线性混合效应模型、Spearman 相关、引导重抽样等统计方法评估结构深度与衰减率的关系。

**📊 数据集**

数据集：从公开预训练权重出发，按统一提示生成 3,000 句文本；所有文本为英语；未使用人工文本，仅使用模型自身生成的合成数据；此外使用 OpenWebText 作为对照训练数据验证 SDH 仅出现在自我训练场景。

**📈 对比分析**

比较方法：对每个模型计算每个特征的衰减率，然后与结构深度和初始频率进行 Spearman 相关；对多模型做混合效应聚合；结果显示结构深度相关系数约 0.54（p<10⁻⁶），显著高于频率相关系数 0.22；表面特征平均提升 25%，深层特征平均下降约 48%/52%。

**⚠️ 局限性**

局限性包括：仅研究 124M–2.8B 级小模型，缺乏超大模型验证；仅限英语，结构深度划分粗糙（四层，d=3 仅含一特征）；特征检测依赖正则与浅层解析，可能存在误检；自我训练仅在纯合成数据上实验，未评估混合真实数据的效果；多模型共用训练语料（Pythia 系列）导致某些独立性缺失。

---

## 144. Ada2MS: A Hybrid Optimization Algorithm Based on Exponential Mixing of Elementwise and Global Second-Moment Estimates

**arXiv ID:** 2605.20533 | [PDF](https://arxiv.org/pdf/2605.20533v1)

**作者:** Meng Zhu `[一作]` (Jiangxi University of Finance and Economics), Weidong Min `[通讯]` (Nanchang University)

**通讯引用:** 2661 | [OpenAlex ID](https://openalex.org/A5046155722)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为Ada2MS的优化器，在训练过程中通过指数插值实现AdamW与Momentum SGD行为的平滑过渡。

**💡 创新点**

创新点在于：①利用全局二阶矩估计与逐元素二阶矩估计的指数混合，使得在过渡过程中保持RMS更新量有限；②设计连续的切换指数α_t，使优化器在训练后期逐步转向SGD特性，以获得更优的泛化；③在保持稳定性的同时兼顾自适应步长与动量优势。

**🔧 技术方法**

技术手段包括：自适应动量（AdamW），全局二阶矩估计，指数混合（α_t）、自适应学习率与权重衰减的调度、Warmup-Steady-Decay学习率策略。

**📊 数据集**

使用三个视觉基准数据集：CIFAR-100（图像分类），PASCAL VOC（目标检测），Semantic Boundaries（语义分割），配合SwinV2、YOLOv7‑tiny和U‑Net模型。

**📈 对比分析**

通过统一的优化器比较协议（相同的RMS范数校准学习率和权重衰减），与Momentum SGD、AdamW、RAdam、AdaI、Lion、SophiaG等主流优化器对比。Ada2MS在CIFAR‑100 Top‑1误差为29.17%（仅次于Lion），在VOC检测上实现mAP@0.5 59.40%（最高），在语义分割上获得mIoU 52.52%（第三好）。

**⚠️ 局限性**

局限性：①需进一步探索切换起始点与速率的敏感性；②未在非视觉任务或更大模型上验证；③引入了额外的α_t 超参数，可能增加调参复杂度。

---

## 145. ClaimDiff-RL: Fine-Grained Caption Reinforcement Learning through Visual Claim Comparison

**arXiv ID:** 2605.20278 | [PDF](https://arxiv.org/pdf/2605.20278v1)

**作者:** Tianle Li `[一作]` (Chinese University of Hong Kong), Yu Cheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 35664 | [OpenAlex ID](https://openalex.org/A5026944066)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ClaimDiff-RL框架，利用图像核验的演员-参考差异作为奖励单元，实现长文本图像字幕的细粒度强化学习；

**💡 创新点**

创新点在于将整体字幕评分拆解为可视化验证的声称差异，使用开放词汇错误类型和严重度对差异进行归因，并可按相对或仅演员两种方式合成可标量奖励；

**🔧 技术方法**

技术上结合多模态判别器（Gemini‑3‑Pro‑Preview）进行差异检测与验证，使用GRPO进行策略梯度优化，并通过严重度权重与歧义惩罚调节信仰‑覆盖权衡；

**📊 数据集**

使用约200万张LAION/DataComp‑1B图像配合Gemini生成的参考字幕进行SFT和RL训练，评测使用160张人工标注的诊断集、Capability字幕细粒度基准及BLINK、OCRBench‑v2、HRBench‑4K、RealWorldQA、SimpleVQA等VQA基准；

**📈 对比分析**

与传统全局评分的RL（有/无参考）相比，ClaimDiff‑RL在保持或提升整体F1的同时显著改善对象计数、空间关系和场景识别等细粒度指标，并在VQA任务上恢复并超过SFT基线的平均表现；

**⚠️ 局限性**

局限在于奖励高度依赖判别器的准确性与参考字幕的提出作用，若判别器误差或参考字幕偏差会影响奖励信号；此外，对极少见视觉属性和高歧义情境的处理仍有限，需要更稳健的验证策略。

---

## 146. Multi-agent Collaboration with State Management

**arXiv ID:** 2605.20563 | [PDF](https://arxiv.org/pdf/2605.20563v1)

**作者:** Mengyang Liu `[一作]` (Shanghai Jiaotong University), Yihong Dong `[通讯]` (Peking University)

**通讯引用:** 988 | [OpenAlex ID](https://openalex.org/A5077542599)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 STORM 框架，采用状态管理代替工作空间隔离，保证多智能体协作时写操作的本地一致性；

**💡 创新点**

创新点在于把冲突检测从后期合并转移到写时进行，利用局部状态一致性与意图注释实现即时冲突反馈和语义协作；

**🔧 技术方法**

使用乐观并发控制的写时冲突校验、意图注释结构化信息、任务分解与管理器调度；

**📊 数据集**

评估数据集包括 Commit0‑Lite 和 PaperBench（Code‑Dev 子集）；

**📈 对比分析**

与单智能体和 GitWorktree（工作空间隔离）对比，STORM 在 Commit0‑Lite 上提升 18.7 计分点、在 PaperBench 上提升 1.4 点，且在成本与时间效率上相当或更优；

**⚠️ 局限性**

局限在于依赖任务分解质量、仍需人工或策略编写意图注释、对高度语义冲突的处理不够完善，且系统复杂度较单智能体更高。

---

## 147. Spectral Souping: A Unified Framework for Online Preference Alignment

**arXiv ID:** 2605.20408 | [PDF](https://arxiv.org/pdf/2605.20408v1)

**作者:** Yinlam Chow `[一作]` (Google DeepMind), Bo Dai `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Spectral Soupping框架，利用语言MDP的谱表示构建离线基准策略，在线时通过线性组合这些基准来实现LLM个性化对齐；

**💡 创新点**

发现LLM logits可由一组谱基向量线性表示，从而给出理论上可证明的子最优性界限；

**🔧 技术方法**

核心技术包括语言MDP建模、谱表示理论、LoRA适配器、两阶段离线–在线训练、Bradley–Terry对比学习与KL正则化；

**📊 数据集**

实验数据集包括UltraFeedback（4维偏好），文本到图像交互的T2I数据集，以及基于LifeSnaps的睡眠教练数据；

**📈 对比分析**

与RLHF、P‑SOUPS、PAD、PAD‑SF等基线对比，Spectral Soupping在所有三域上逼近83–88%（UltraFeedback/T2I）或72%（睡眠教练）RLHF性能，并显著优于其它基线，且在线适应更快、数据更高效；

**⚠️ 局限性**

局限性包括子最优性界限不够紧，仍需足够多的基准策略以覆盖谱空间，对特定任务可能需要手工挑选或扩充基准，且需要离线预训练成本。

---

## 148. Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs

**arXiv ID:** 2605.20315 | [PDF](https://arxiv.org/pdf/2605.20315v1)

**作者:** Haiquan Lu `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**通讯引用:** 13769 | [OpenAlex ID](https://openalex.org/A5015574447)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对长上下文LLM代理推理的阶段感知量化框架Mix-Quant，将高吞吐量的NVFP4量化仅应用于前填充阶段，保持解码阶段BF16精度

**💡 创新点**

创新点在于：①发现前填充阶段对量化误差鲁棒性强，解码阶段易受误差累积影响；②设计仅对前填充量化的混合策略，实现性能与效率的权衡；③将NVFP4微尺度量化与硬件加速结合，提供可部署方案

**🔧 技术方法**

使用NVFP4（四位浮点微尺度）权重与激活量化、BF16高精度解码、Prefill-Decode分离部署、NIXL KV缓存传输、FlashInfer、vLLM服务栈

**📊 数据集**

评估数据集包括长上下文推理Benchmarks（LongBench‑V2、AA‑LCR）、代理任务Benchmarks（BFCL v4、LongMemEval、τ²‑bench）、数学推理Benchmark（Math500、AIME24、AIME25）

**📈 对比分析**

与BF16基线及全NVFP4统一量化对比，Mix-Quant在所有代理与长上下文任务中平均保持≥90%原始性能，且前填充速度提升≈2–3×；全量化下性能下降显著

**⚠️ 局限性**

局限性包括：①对解码阶段未量化，仍无法在极高并发环境下进一步降低成本；②在极长上下文或高错误敏感任务中，前填充量化仍可能导致微小误差累积；③对NVFP4硬件依赖较高，迁移成本较大

---

## 149. SOLAR: A Self-Optimizing Open-Ended Autonomous Agent for Lifelong Learning and Continual Adaptation

**arXiv ID:** 2605.20189 | [PDF](https://arxiv.org/pdf/2605.20189v1)

**作者:** Nitin Vetcha `[一作]` (National University of Singapore), Dianbo Liu `[通讯]` (National University of Singapore)

**通讯引用:** 6616 | [OpenAlex ID](https://openalex.org/A5014407399)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SOLAR，能够在流式持续学习环境中自我发现并保留参数级自适应策略，以实现 LLM 的实时自适应。

**💡 创新点**

创新点在于把 LLM 权重视为环境进行自我探索，通过多级强化学习生成、验证并积累自适应策略，实现可持续的可塑性与稳定性平衡；同时利用低秩 LoRA 与元学习框架实现高效的自适应。

**🔧 技术方法**

使用参数级元学习、低秩 LoRA 适配器、强化学习（ReST^EM）、多级自我编辑策略、Prompt 编码（Sentence‑BERT）、卷积解码器等技术。

**📊 数据集**

主要数据集包括常识推理集 ARC‑e、ARC‑c、BoolQ、HellaSwag、PIQA，外延还包含 GSM‑MC、MATH‑MC、DivLogicEval、SocialIQA、CodeMMLU。

**📈 对比分析**

与 LoRA、TTL、DOM、DnD 等基线相比，SOLAR 在域内平均提升约 20–30%（例如 ARC‑e 74.7% vs 56.5%），在域外平均提升约 15–25%；整体表现明显优于现有方法。

**⚠️ 局限性**

局限性在于训练阶段计算成本高、需要大量 GPU 资源，且仍依赖初始手工构造的知识库，缺乏完全无监督的自我演化；推理时需要缓存策略，实时性受限。

---

## 150. Multi-Week, In-Class Deployments of Telepresence Robots With Four Homebound K-12 Students: Benefits, Challenges, and Recommendations

**arXiv ID:** 2605.20431 | [PDF](https://arxiv.org/pdf/2605.20431v1)

**作者:** Matthew Rueben `[一作]` (University of Southern California), Maja J. Matarić `[通讯]` (University of Southern California)

**通讯引用:** 30541 | [OpenAlex ID](https://openalex.org/A5010248533)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在美国一所大型都市学区，对四名因健康或心理原因无法到校的学生进行为期数周的远程课堂 Telepresence 机器人部署，并通过访谈与音视频记录评估其使用体验

**💡 创新点**

首次系统化探究多维度个体差异对远程课堂体验的影响，并提出可定制化部署流程与机器人/接口改进建议

**🔧 技术方法**

使用 OhmniLabs 机器人加上基于 Web 的操作界面，并结合可访问性改造（键盘、游戏手柄、文字转语音等）

**📊 数据集**

收集了 15 次访谈文本和约 67 小时的课堂与操作音视频数据

**📈 对比分析**

通过案例研究与跨案例分析进行定性比较，未做量化性能对比；结果显示机器人提升了社交连结与自我调节能力，但仍存在听觉、视觉与移动性挑战

**⚠️ 局限性**

样本仅四人、缺乏对照组、部署时间有限、部分音视频缺失、访谈受访者多由家属代替，导致可推广性与因果推断受限

---

## 151. Machine-Learning-Enhanced Non-Invasive Testing for MASLD Fibrosis: Shallow-Deep Neural Networks Versus FIB-4, Tabular Foundation Models, and Large Language Models

**arXiv ID:** 2605.20523 | [PDF](https://arxiv.org/pdf/2605.20523v1)

**作者:** Athanasios Angelakis `[一作]` (UniBw), Filomena Ferrucci `[通讯]` (University of Salerno)

**通讯引用:** 4047 | [OpenAlex ID](https://openalex.org/A5053084752)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究提出机器学习增强的非侵入性检测（MLE-NIT），通过在保留FIB-4输入变量的基础上使用小型神经网络提升MASLD（原NAFLD）患者高级纤维化的检测。

**💡 创新点**

创新点在于：①正式定义MLE-NIT框架，强调输入封闭与工作流程不变；②设计并验证了仅含354个可训练参数的“浅深”神经网络（s-DNN），证明在仅用FIB-4变量空间时能匹配甚至优于大型通用模型；③在同一特征空间对比TabPFN、LLM和传统FIB-4，揭示不同模型在外部验证中的敏感性/特异性偏差。

**🔧 技术方法**

使用的技术包括：浅深神经网络（s-DNN）、TabPFN（表格基础模型）、GPT‑4o（LLM，零射和微调两种方式）以及传统FIB‑4评分；同时做了校准、置换特征重要性、决策曲线分析等诊断性评估。

**📊 数据集**

数据集为三组来自中国、马来西亚和印度的活检确诊MASLD患者，共计784例，其中中国样本分为训练/内部验证，马来西亚和印度作为外部验证集。

**📈 对比分析**

比较方法：在外部验证集上固定阈值（0.5）计算ROC‑AUC、灵敏度、特异度和F1；与传统FIB‑4在1.3阈值下对比。结果显示：s‑DNN在马来西亚外部集ROC‑AUC 0.77、灵敏度0.81；在印度外部集ROC‑AUC 0.67、灵敏度0.59；TabPFN偏向高特异性（马来西亚0.69、印度0.66），LLM表现波动；总体上s‑DNN提供更均衡的操作特性。

**⚠️ 局限性**

局限性包括：①仅使用FIB‑4变量空间，限制了潜在性能提升；②外部验证样本量小且族群异质性高，影响泛化；③阈值未根据外部集优化；④LLM缺乏概率输出，难以做校准与决策曲线；⑤s‑DNN的结构选择（奇素数宽度）未系统验证；⑥未进行前瞻性临床试验评估实际工作流影响。

---

## 152. Stacked Intelligent Metasurfaces for Resolution-Constrained Near-Field Range Extension in 6G Systems

**arXiv ID:** 2605.20298 | [PDF](https://arxiv.org/pdf/2605.20298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 153. AnimeAdapter: Fine-grained and Consistent Zero-shot Anime Character Generation

**arXiv ID:** 2605.20237 | [PDF](https://arxiv.org/pdf/2605.20237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 154. Collocational bootstrapping: A hypothesis about the learning of subject-verb agreement in humans and neural networks

**arXiv ID:** 2605.20529 | [PDF](https://arxiv.org/pdf/2605.20529v1)

**作者:** Claire Hobbs `[一作]` (Yale University), R. Thomas McCoy `[通讯]` (Yale University)

**通讯引用:** 1117 | [OpenAlex ID](https://openalex.org/A5003082850)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用神经网络在合成语料中模拟学习，验证词共现统计（collocational bootstrapping）能帮助学习英语主谓一致，并对儿童说话输入进行统计分析。

**💡 创新点**

提出并证明了 collocational bootstrapping 机制，发现 Zipf 分布参数 α≈1.4 为最优，可与儿童输入的统计特性匹配；首次将词共现的可变性与语法习得结合起来。

**🔧 技术方法**

采用两层 Transformer（类似 GPT‑2）的语言模型进行训练与评估；使用 Zipf 拟合和最小对句评测方法；对 CHILDES 数据进行依存解析提取主谓配对。

**📊 数据集**

合成语料（12,000 句，四种句型，40 名词 + 40 动词）；CHILDES 语料（约 4,739,189 条成人对幼儿的发话记录，提取约 2,802,071 条主谓配对）。

**📈 对比分析**

通过四类最小对句（SEEN/MATCH、UNSEEN/MATCH、SEEN/MISMATCH、UNSEEN/MISMATCH）评测模型准确率，绘制 α‑准确率曲线，α≈1.4 时达到接近 100% 的最佳表现；CHILDES 统计拟合得到 α≈1.43，接近模型最优值，表明自然输入具备有利于协同引导的统计结构。

**⚠️ 局限性**

实验语料过于简化，缺乏真实词汇、时态、多元输入；模型仅处理文本，未考虑语调、语义和多模态信息；仅研究主谓一致，未探讨其他结构；与儿童实际习得错误模式仍有差距。

---

## 155. COAgents: Multi-Agent Framework to Learn and Navigate Routing Problems Search Space

**arXiv ID:** 2605.20618 | [PDF](https://arxiv.org/pdf/2605.20618v1)

**作者:** Oleksandr Yakovenko `[一作]` (Huawei Technologies Canada), Mao Kun `[通讯]` (Huawei Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了COAgents多代理框架，用于对VRP的搜索过程进行学习，能够自适应地选择节点、动作以及跳跃操作。

**💡 创新点**

将搜索空间建模为Partial Search Graph，并通过节点选择、移动选择和跳跃代理三者协同决策，使用共享的GGCN–Transformer核心实现可迁移性。

**🔧 技术方法**

采用Gated Graph Convolution与Transformer的组合为通用核心，配合问题特定的E2E模块和跳跃解码器，并通过监督学习和强化学习训练三种代理。

**📊 数据集**

在10,000个随机生成的CVRP和VRPTW（100客户）实例上训练，并在官方1,000个测试集上评估。

**📈 对比分析**

与传统启发式、ALNS、LKH3、HGS、OR-Tools以及多种学习型方法（如POMO、MVMoE、NLNS、DACT等）对比，VRPTW上平均目标值优于所有神经基准，缩小到HGS的缺口14–44%，在CVRP上保持竞争力。

**⚠️ 局限性**

运行时间相对较慢，受PSG自注意力和跳跃解码器二次复杂度限制，且目前仅验证到N=100；实现为Python，难以与高效C/C++实现竞争。

---

## 156. Code Generation by Differential Test Time Scaling

**arXiv ID:** 2605.20473 | [PDF](https://arxiv.org/pdf/2605.20473v1)

**作者:** Yifeng He `[一作]` (University of California, Davis), Hao Chen `[通讯]` (University of Hong Kong)

**通讯引用:** 112334 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于覆盖驱动差分测试的测试时扩展方法，通过在 LLM 生成的多种代码候选者上执行模糊测试，收集动态行为并聚类，最终挑选最具代表性的候选作为最终输出。

**💡 创新点**

创新点在于：1）不依赖公共测试或额外 LLM 推理即可完成候选选择；2）利用覆盖驱动的差分分析与聚类来识别行为一致的候选；3）极大降低执行时间和 token 消耗，保持与现有最佳方法相当的性能。

**🔧 技术方法**

使用技术包括：多种采样策略（温度/核采样、束搜索、提示扰动）生成候选；覆盖驱动模糊测试（AFL/atheris 等）生成输入；差分分析记录动态行为；层次平均链聚类与 medoid 选取进行候选选择。

**📊 数据集**

使用的实验数据集为 LiveCodeBench（脚本和库函数两类）以及 CodeContests 的 v2、v5 版本，涵盖了多种编程任务和语言（以 Python 为主）。

**📈 对比分析**

与 Majority Voting、SkyCoder 等现有测试时扩展方法对比，实验在 GPT‑4、Qwen2.5‑Coder、GPT‑4o-mini 等多种 LLM 上实现了 0.5%~13.5% 的 Pass@1 提升；相较于基线，执行时间约为 1/5，token 消耗仅为 4% 以内，性能与最优方法相当或更好。

**⚠️ 局限性**

局限性包括：1）仅支持 Python，扩展至其他语言需配合相应模糊工具；2）对复杂自定义输入、非确定性程序处理有限；3）差分测试假设多数候选正确，若多数误解需求会导致错误选取；4）LLM 生成模糊驱动成功率约 90–96%，失败时需回退；5）在极大输入规模或多线程程序上覆盖率难以充分获取。

---

## 157. Beyond Routing: Characterising Expert Tuning and Representation in Vision Mixture-of-Experts

**arXiv ID:** 2605.20610 | [PDF](https://arxiv.org/pdf/2605.20610v1)

**作者:** Gene Tangtartharakul `[一作]` (University of Auckland), Katherine R. Storrs `[通讯]` (University of Auckland)

**通讯引用:** 1071 | [OpenAlex ID](https://openalex.org/A5042387485)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过训练稀疏门控的对比学习Mixture-of-Experts CNN，对自然图像进行学习，并使用神经科学方法对专家级特征调谐进行深入分析；

**💡 创新点**

创新点在于突破传统仅基于路由的专家解释，采用 MEI、RSA 等工具从专家内部视角揭示连续行为相关维度的稀疏调谐，并发现了稳定的可动画-不可动画分裂；

**🔧 技术方法**

使用的技术包括自监督对比学习（SimCLR）、稀疏门控 MoE（top‑k 与噪声门控）、Most Exciting Inputs、非负 Lasso 回归、代表性相似度分析等；

**📊 数据集**

使用的数据集为 STL10（训练）和 THINGS（OOV 评估及 66 维行为导向特征）；

**📈 对比分析**

通过对比路由级别与专家级别的特征调谐、分类可分性及 RSA 结果，展示专家虽在路由上显示离散类别偏好，但在内部实现上呈现连续特征调谐，且跨模型和种子保持可复制的动画‑无动画分裂；

**⚠️ 局限性**

局限性包括训练数据集规模有限、仅使用自监督目标、保持模型总参数不变限制了专家容量、以及结果可能不直接推广到工业规模的大型 MoE 系统。

---

## 158. MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric Semantic Modality Discovery

**arXiv ID:** 2605.20297 | [PDF](https://arxiv.org/pdf/2605.20297v1)

**作者:** Ziyuan Gao `[一作]` (University College London), Ziyuan Gao `[通讯]` (University College London)

**通讯引用:** 177 | [OpenAlex ID](https://openalex.org/A5091758218)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种通过中文文本提示自动发现语义模态的持续医学图像分割框架MedCRP-CL；

**💡 创新点**

创新点在于将贝叶斯非参数的 Chinese Restaurant Process 与语义模态发现结合，利用LoRA适配器与 intra-modality EWC 实现任务结构感知的参数共享与隔离；

**🔧 技术方法**

采用CLIP文本编码器提取提示嵌入、CRP 进行动态聚类、LoRA 低秩适配器与 EWC 正则化、Dice+交叉熵损失；

**📊 数据集**

使用 16 个医学分割任务，涵盖内镜、皮肤镜、超声、胸部 X 光等四种成像类型，数据集包括 Kvasir、ISIC、CAMUS、BUSI 等；

**📈 对比分析**

与 EWC、RAPF、CL-LoRA、MoE-Adapters 等基线相比，MedCRP-CL 在 Dice 上达 73.3%，遗忘率仅 4.1%，参数量 8.6M，显著优于最佳基线（+8.0% Dice、6 倍参数压缩）；

**⚠️ 局限性**

局限在于对文本提示的质量敏感，极端噪声下聚类可能失效，对不同文本编码器的依赖较高，且仅在 2D 成像上验证，3D 数据的适用性待进一步研究。

---

## 159. Reinforcing Human Behavior Simulation via Verbal Feedback

**arXiv ID:** 2605.20506 | [PDF](https://arxiv.org/pdf/2605.20506v1)

**作者:** Weiwei Sun `[一作]` (Carnegie Mellon University), Maarten Sap `[通讯]` (Carnegie Mellon University)

**通讯引用:** 6393 | [OpenAlex ID](https://openalex.org/A5015128745)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Ditto模型，利用对话式口头反馈在强化学习中训练人类行为模拟器。

**💡 创新点**

创新点在于将口头反馈视为一等信号，生成基于反馈的改进回合，并与原始回合联合优化，使基模型在推理时无需反馈即可内化指导。

**🔧 技术方法**

技术方法包括GRPO强化学习、LoRA微调、Qwen3-8B大模型、反馈条件生成和联合损失优化。

**📊 数据集**

使用了自建的Soul基准，涵盖10个任务、6类人类行为模拟（Theory of Mind、角色扮演、社交技能、学习者模拟、用户模拟、人格模拟），并配套训练数据。

**📈 对比分析**

与GPT‑5.4、专门人类模拟模型及GRPO对比，Ditto平均提升36%，在10个任务中6个击败GPT‑5.4，且在多轮和生成任务上显著优于标准GRPO，安全性指标如秘密保持也有所提升。

**⚠️ 局限性**

局限在于对训练时可获得的反馈依赖度高、对逻辑推理类任务提升有限、评估依赖外部判定器，且在多任务迁移和实际应用场景中的鲁棒性仍需进一步验证。

---

## 160. Stage-Audit: Auditable Source-Frontier Discovery for Cross-Wiki Tables

**arXiv ID:** 2605.20478 | [PDF](https://arxiv.org/pdf/2605.20478v1)

**作者:** Chen Shen `[一作]` `[通讯]` (Megagon Labs), Chen Shen (Megagon Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

为Seed2Frontier任务设计了Stage‑Audit治理框架，确保结构化表格行级的源引用与可追溯性；

**💡 创新点**

创新点在于分离写权限的治理协议、行级源引用门控以及12项审核税onomies，实现对表格完整性与来源的系统化审核；

**🔧 技术方法**

采用LLM‑curator与LLM‑auditor组合，配合源引用门控、行级证据检查和结构化审核流程；

**📊 数据集**

使用51个Seed2Frontier实例组成的评估集，覆盖15个顶级域、Wiki种子页、标注的补充页面及主键真值表；

**📈 对比分析**

与memory‑only、seed‑outlink、vanilla LLM curator四种配置对比，Stage‑Audit提升源前沿精度从0.356至0.505（+42%），F1从0.334至0.451；在不同域和前沿规模上表现出可观提升；

**⚠️ 局限性**

限制在于仅评估单通行无修复循环，未包含人类接受步骤；测试仅在两种模型（闭源GPT‑5.4和开源Llama 3.3 70B）上，跨模型泛化与更大规模前沿的适用性尚未验证。

---

## 161. An exponential mechanism based on quadratic approximations for fine-tuning machine learning models with privacy guarantees

**arXiv ID:** 2605.20521 | [PDF](https://arxiv.org/pdf/2605.20521v1)

**作者:** Hoang Tran `[一作]` (Oak Ridge National Laboratory), M. Paul Laiu `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 88 | [OpenAlex ID](https://openalex.org/A5077931824)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于指数机制（Exponential Mechanism）的差分隐私微调方法 ExpM-Quad，该方法利用预训练模型附近的二次近似构造可微分的效用函数，能够直接从截断高斯分布中采样得到隐私保护的模型参数。

**💡 创新点**

创新点包括：①将指数机制与局部二次近似结合，获得可闭式采样的高斯分布；②给出精确的敏感度估计与理论隐私保证；③设计随机投影（随机子空间）策略，使高维模型可扩展；④在理论上证明投影子空间可保留近似最优效用，并给出误差上界。

**🔧 技术方法**

核心技术包括：差分隐私的指数机制、Gauss‑Newton 近似二阶信息、截断高斯采样（重jection 或 Gibbs 采样）、随机投影（随机子空间）以及对损失函数的局部二次展开。

**📊 数据集**

实验数据集：1) 5 维正弦回归；2) MNIST 图像分类（干净和加噪声版本）；3) 医疗临床数据 MIMIC‑IV 的死亡预测任务（线性模型）。

**📈 对比分析**

与 DP‑SGD 以及非私有 SGD/CE 基线进行对比。ExpM-Quad 在不同隐私预算 ε 下均表现出随 ε 增大而显著提升的性能，尤其在中等至大 ε 时超过 DP‑SGD，并逐渐逼近非私有基线；在 MNIST 上，虽然对 CE 损失的近似不如 MSE 严谨，但仍能达到接近基线的准确率。

**⚠️ 局限性**

局限性包括：①需要计算或近似 Hessian，导致在大模型上计算成本较高；②二次近似在非线性损失（如交叉熵）上可能不够精确；③需手动调参（如截断半径 R、投影维度 p̃、正则化 λ）；④在极小 ε 下表现仍受限，无法完全抵消隐私噪声；⑤随机投影可能因子空间选择不佳而丢失部分最优信息。

---

## 162. Closing the Motivation Gap: Incentives Enhance Visual Misinformation Discernment and Verification

**arXiv ID:** 2605.20438 | [PDF](https://arxiv.org/pdf/2605.20438v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 163. Fifty Years of Transaction Processing Research (extended)

**arXiv ID:** 2605.20466 | [PDF](https://arxiv.org/pdf/2605.20466v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 164. Catching a Moving Subspace: Low-Rank Bandits Beyond Stationarity

**arXiv ID:** 2605.20269 | [PDF](https://arxiv.org/pdf/2605.20269v1)

**作者:** Hamed Khosravi `[一作]` (Georgia Institute of Technology), Xiaoming Huo `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5719 | [OpenAlex ID](https://openalex.org/A5014880531)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种在低秩线性上下文多臂赌博机中能够在线识别并适应子空间漂移的算法SPSC及其自适应变体；

**💡 创新点**

首次给出从单次标量反馈恢复低秩子空间的必要与充分条件，并实现了以子空间秩 r 为主的动态调优奖励；

**🔧 技术方法**

利用二次测量映射与上采样逆算实现子空间估计，结合窗口化投影岭-UCB和CUSUM式变点检测，构成SPSC；

**📊 数据集**

在合成实验、UCI/MovieLens、半合成临床（Warfarin、Vancomycin）、ZOZOTOWN生产日志等共十一套数据集上进行评估；

**📈 对比分析**

与LinUCB、D-LinUCB、SW-LinUCB、Restart-LinUCB、LowRank-Reward、LinTS、VOFUL、BOSS等基线比较，SPSC在 d−r≳T^{1/6} 的 regime 下实现了 10%–30% 的 regret 降低，且在自适应变点检测下仍优于所有非oracle 方法；

**⚠️ 局限性**

假设已知噪声方差、完整探测支持以及状态-噪声耦合受限；对高秩或强漂移、非高斯噪声、频繁变点等场景的鲁棒性尚待进一步研究。

---

## 165. Smaller Abstract State Spaces Enable Cross-Scale Generalization in Reinforcement Learning

**arXiv ID:** 2605.20272 | [PDF](https://arxiv.org/pdf/2605.20272v1)

**作者:** Nasehatul Mustakim `[一作]` (University of Saskatchewan), Lucas Lehnert `[通讯]` (University of Saskatchewan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种理论框架，研究在部分可观测马尔可夫决策过程（POMDP）中通过将状态压缩到极小抽象空间来实现跨尺度的离群分布（OOD）泛化。

**💡 创新点**

创新点在于引入了“后继加权模型归约”（successor‑weighted model reduction），能够在不依赖最大范数的前提下将无限状态空间压缩为更小的抽象空间，并给出了相应的性能损失上界。

**🔧 技术方法**

技术上扩展了有限状态POMDP的状态抽象与仿真引理到可数无限状态空间，采用分布加权范数、后继表示（Successor Representation）及其归约投影算子，推导出OOD泛化的理论边界。

**📊 数据集**

实验使用自定义的温冷格子（warm–cold lattice）和符号链（sign‑chain）等模拟任务，未使用公开数据集。

**📈 对比分析**

通过对比不同抽象规模下的误差类型（近似误差、估计误差）以及在toy环境中的子最优动作计数，展示了U‑形误差权衡，并证明合适的抽象长度可显著提升OOD测试性能。

**⚠️ 局限性**

局限性包括：未给出具体的学习算法实现抽象；仅考虑单一抽象函数；假设观测和动作空间有限，且未验证在真实复杂环境中的效果。

---

## 166. Less Data, Faster Training: repeating smaller datasets speeds up learning via sampling biases

**arXiv ID:** 2605.20314 | [PDF](https://arxiv.org/pdf/2605.20314v1)

**作者:** Jingwen Liu `[一作]`, Bingbin Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究并验证了在固定计算预算下，使用更小的数据集（并多次重复训练）可以比使用更大数据集更快收敛的“小数据-大数据”差距，并揭示其主要原因。

**💡 创新点**

发现更小的数据集所带来的采样偏差能通过调整不同层的相对范数增长来加速特征学习，提出理论分析与实证验证，并展示通过层级初始化或学习率干预可消除该差距。

**🔧 技术方法**

采用理论分析（采样偏差、范数增长）、梯度下降/SGD、AdamW、分阶段训练、多重实验干预（层级学习率、初始化比例、QK 正则化）等技术。

**📊 数据集**

使用一系列合成任务的数据集，包括稀疏奇偶校验、单指数模型、上下文线性回归、模数加法，并在不同规模（如 2¹⁴ vs 2²⁰）下比较。

**📈 对比分析**

通过对比总计算量（步数×批量大小）和模型性能（准确率/损失），在稀疏奇偶校验任务中获得高达 100 倍的计算节省；随机标签实验验证速度提升不依赖任务信号；层级干预可几乎消除差距，提升最终性能。

**⚠️ 局限性**

局限性包括：研究仅限于合成/结构化任务，结果可能无法直接推广到真实自然语言或大规模无结构数据；在凸优化（如线性回归）下未观察到类似效应；需进一步探索在复杂真实场景中的可行性。

---

## 167. Enhancing Graph-Based SLAM in GNSS-Denied environments by leveraging leg odometry

**arXiv ID:** 2605.20484 | [PDF](https://arxiv.org/pdf/2605.20484v1)

**作者:** Léon Perruchot-Triboulet `[一作]` (ENSTA), Kai Xiao `[通讯]` (LinXai Tech Co)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过在 LIO‑SAM 中加入并行的前向运动学（FK）lane，并利用身份相对位姿约束对垂直方向进行轻量正则化，从而在 GNSS‑缺失环境下抑制激光雷达-惯性 SLAM 的高度漂移。

**💡 创新点**

创新点在于：① 并行 kinematic lane 与 LiDAR‑inertial lane 的软耦合；② 在噪声模型中对 z 轴设定低方差、其余轴设定高方差，实现垂直正则化而不损失水平精度；③ 仅利用已有的腿部编码器数据，保持低部署成本。

**🔧 技术方法**

使用了 LIO‑SAM、GTSAM 的 iSAM2 递增优化、前向运动学 odometry、身份相对位姿约束、选择性噪声模型，以及 ROS Noetic 的系统集成。

**📊 数据集**

实验使用了 Factory 数据集（约 700 m、平坦）和 CocoPark 数据集（约 600 m、丘陵、动态物体）两组户外闭环路径。

**📈 对比分析**

通过与 baseline LIO‑SAM（无 GNSS）在闭环误差上的对比，Factory 数据集提升为垂直误差 0.2 m、水平误差约 2 m；CocoPark 数据集中 baseline 崩溃，改进方法得到垂直误差 0.3 m、水平误差约 4 m。

**⚠️ 局限性**

局限性包括：仅对垂直方向进行正则化，水平轴融合有限；未实现滑移检测，严重滑移时 FK prior 可能误导；实验仅在单一 D50 机器人与单一 SLAM 框架上验证，缺乏外部 RTK‑GNSS 等真实地面真值；动态物体处理仍需进一步改进。

---

## 168. Adaptive Probe-based Steering for Robust LLM Jailbreaking

**arXiv ID:** 2605.20286 | [PDF](https://arxiv.org/pdf/2605.20286v1)

**作者:** Junxi Chen `[一作]` (Sun Yat-Sen University), Xiaohua Xie `[通讯]` (Sun Yat-Sen University)

**通讯引用:** 7003 | [OpenAlex ID](https://openalex.org/A5018298892)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对现有对比驱动的LLM jailbreak方法进行改进，提出基于模型提取的方向迭代优化与基于对比激活统计的自适应强度调节，实现了更强且更鲁棒的探测式对抗攻击。

**💡 创新点**

创新点在于：① 将对比向量搜索视作模型提取，利用已存在的判别器对激活进行标注并迭代增强训练集，显著提升方向的准确性；② 用对比激活的L2范数自适应设定每层的steering强度，消除了手动调参的痛点；③ 统一剔除最后一层激活并对所有token位置进行steering，进一步提升攻击效果。

**🔧 技术方法**

技术手段包括：线性探测（Linear Probe）、对比激活提取、模型提取式迭代重训练、基于统计的自适应强度调节、激活剪枝（Discarding Last-layer Activation）、全Token Steering、以及对抗性评价指标（SRF、HB、SR）。

**📊 数据集**

使用的数据集：100对对比提示（善恶各50），200条强制拒绝与HarmBench的有害提示（共200）。对抗测试使用12种针对jailbreaking的加固LLM模型。

**📈 对比分析**

与RepE、SCAV、RD-C/A、Angular等基线对比。实验表明，本方法在12个加固LLM上将有害率从仅6%提升至至少50%，多数模型达到70%以上；相较于基线，整体提升约30%–40%（如在Llama3-CB上从0.0提升至0.98）。

**⚠️ 局限性**

局限性：① 仅在单体LLM层面验证，缺乏对完整对话系统的评估；② 依赖可靠的判别器（如SRF），若判别器性能受限，可能影响方向迭代；③ 需要前置对比提示，若对比提示不足或偏差大，迭代效果受限；④ 结果受限于评判标准（3种LLM评判器），不同评判器可能得到不同结论。

---

## 169. ZEBRA: Zero-shot Budgeted Resource Allocation for LLM Orchestration

**arXiv ID:** 2605.20485 | [PDF](https://arxiv.org/pdf/2605.20485v1)

**作者:** May Hamri `[一作]` (Tel Aviv University), Inbal Talgam-Cohen `[通讯]` (Tel Aviv University)

**通讯引用:** 666 | [OpenAlex ID](https://openalex.org/A5081600081)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ZEBRA框架，在多代理流水线中通过LLM估计各阶段效用曲线并以连续非线性背包问题求解预算分配；

**💡 创新点**

创新点在于零样本、推理时的预算分配，将阶段效用建模为饱和指数曲线，利用水填充法一阶KKT求解，实现无监督的连续预算拆分；

**🔧 技术方法**

技术包括：LLM控制器（Prompt+LLM推断）、连续非线性背包求解（Lagrange双二分/水填充）、加法与乘法质量聚合的统一求解；

**📊 数据集**

主要使用APPS编码面试基准（150题），以及HumanEval、CodeContests编码基准和HotpotQA问答基准；

**📈 对比分析**

与直接让LLM分配预算（LLM-direct）和均匀分配（Uniform）对比，ZEBRA在所有预算水平和难度级别均显著提升NB保留率（如α=0.5时恢复94.4%质量，LLM-direct仅88.1%）；

**⚠️ 局限性**

局限包括：仅一次性前置分配，未支持在线/递归重分配；控制器占约33%预算开销；仅适用于分阶段流水线，无法处理无阶段或复杂DAG结构；

---

## 170. When Reasoning Supervision Hurts: TTCW-Based Long-Form Literary Review Generation

**arXiv ID:** 2605.20364 | [PDF](https://arxiv.org/pdf/2605.20364v1)

**作者:** Jinlong Liu `[一作]` (University of Birmingham), Mark Lee `[通讯]` (University of Birmingham)

**通讯引用:** 7129 | [OpenAlex ID](https://openalex.org/A5100620082)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个包含26万长篇故事的TTCW创意写作评审数据集，并对LLM在固定格式评审生成中的推理监督效果进行评测。

**💡 创新点**

首次公开长篇TTCW评审数据集，并发现非推理监督在固定格式评审生成中优于推理监督。

**🔧 技术方法**

使用Qwen3 LLM、LoRA参数高效微调、基于规则的标注与合成技术。

**📊 数据集**

使用基于TTCW的长篇故事数据，规模约263,911篇，4k-8k词。

**📈 对比分析**

通过对比推理与非推理微调的解析率、得分精度及BERTScore，非推理模型在8B/4B规模下解析率达1.0，最终评估分0.6820，优于推理模型。

**⚠️ 局限性**

数据无人工标注，可能带偏差；仅在Qwen3家族、4B/8B规模上实验，未验证更大模型或其它架构；LoRA可能限制学习。

---

## 171. ConceptSeg-R1: Segment Any Concept via Meta-Reinforcement Learning

**arXiv ID:** 2605.20385 | [PDF](https://arxiv.org/pdf/2605.20385v1)

**作者:** Yuan Zhao `[一作]` (Dalian University of Technology), Xiaoqi Zhao `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 ConceptSeg-R1 的统一概念分割框架，能够在上下文无关、上下文相关和需要推理的三类概念上实现像素级分割。

**💡 创新点**

创新点在于将元学习版 GRPO 与多模态大语言模型相结合，形成 Meta-GRPO、概念翻译模块（CTM）以及快捷路由，实现在推理与分割之间的无损、高效转换，并通过规则诱导实现跨任务的迁移。

**🔧 技术方法**

采用的技术包括 Meta-GRPO 强化学习、概念翻译模块 CTM、SAM 3 作为分割头、Qwen2.5-VL 作为推理引擎，以及 Chain‑of‑Thought 模板与短语提示。

**📊 数据集**

在 16 个涵盖 CI、CD、CR 概念的基准上进行评估，数据集包括自然、工业、医学场景、ReasonSeg 以及 Cityscapes 等。

**📈 对比分析**

与 SAM 3、SAM3‑I、LENS、SegZero 等方法在 CI、CD、CR 任务上进行对比，ConceptSeg‑R1 在所有三类概念上均获得最优或接近最优的 mIoU、gIoU、cIoU，并在 Cityscapes 零样本设置下将 mIoU 提升至 62.6%。

**⚠️ 局限性**

局限性在于对极其复杂或长文本推理的泛化能力仍有限，Meta‑GRPO 的训练成本较高，且对极端稀有概念的充分验证仍待进一步研究。

---

## 172. Do as I Say, Not as I Do: Instruction-Induction Conflict in LLMs

**arXiv ID:** 2605.20382 | [PDF](https://arxiv.org/pdf/2605.20382v1)

**作者:** Carolina Camassa `[一作]` (Future Impact Group), Derek Shiller `[通讯]` (Rethink Priorities)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在面对指令与上下文模式冲突时的指令遵循鲁棒性，并设计了可控的实验范式在13个模型上进行评估

**💡 创新点**

提出了一个可控的指令-诱导冲突实验框架，系统量化诱导压力对指令遵循的影响，并发现输出多样性是鲁棒性的关键因素

**🔧 技术方法**

利用多轮对话中的硬编码诱导转折、测量不同 N 值下的指令遵循率、chain‑of‑thought 推理、模型自我预测以及统计相关性分析等技术

**📊 数据集**

使用由 35 个问题集构成的问答库，涵盖固定输出与任务型两类条件，硬编码答案来自预生成文本，涵盖多语言与多任务场景

**📈 对比分析**

通过比较不同模型、条件与 N 取值下的指令遵循率和自我预测准确率发现，平均 IF 率从 1% 到 99% 变化；鲁棒性与标准基准无显著相关；输出多样性显著提升 IF 率；推理可提升但未完全消除易受诱导

**⚠️ 局限性**

实验设置人为、受限，仅评估固定输出条件下的自我预测；使用贪婪解码未覆盖完整输出分布；缺乏对真实部署环境下鲁棒性的验证

---

## 173. Quant.npu: Enabling Efficient Mobile NPU Inference for on-device LLMs via Fully Static Quantization

**arXiv ID:** 2605.20295 | [PDF](https://arxiv.org/pdf/2605.20295v1)

**作者:** Jinghe Zhang `[一作]`, Gang Huang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在移动 NPU 上实现了整数全静态量化框架，支持低比特 LLM 推理。

**💡 创新点**

核心创新包括旋转与位宽感知初始化、分布感知分阶段优化、以及基于敏感度的自适应混合精度策略。

**🔧 技术方法**

使用可学习的量化参数、Hadamard 旋转矩阵、梯度直通估计（STE）以及局部量化误差损失实现全静态优化。

**📊 数据集**

在 Llama‑3.2‑3B‑Instruct、Qwen3‑1.7B 以及 Llama‑3‑8B 上，利用 C4、WikiText‑2 进行校准，并在 PIQA、Winogrande、HellaSwag、ARC‑Easy、ARC‑Challenge、LAMBADA 等零样本基准上评测。

**📈 对比分析**

与 ExecuTorch、QuaRot、SpinQuant 等现有 PTQ 方法比较，保持或略优的精度同时实现最高 15.1% 的 NPU 推理延迟下降，4‑bit 权重/激活量化亦保持 12% 以上的准确提升。

**⚠️ 局限性**

局限性在于仍需预先融合旋转矩阵、对极低比特（如 4‑bit）或超大模型的鲁棒性尚有限，并依赖少量校准样本与特定 NPU 架构。

---

## 174. Matryoshka Concept Bottleneck Models

**arXiv ID:** 2605.20612 | [PDF](https://arxiv.org/pdf/2605.20612v1)

**作者:** Ziye Chen `[一作]` (Hong Kong University of Science and Technology), Lijie Hu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5067496051)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的Matryoshka概念瓶颈模型（MCBM），通过mRMR排序构建层次化的概念瓶颈，支持在单个模型中按不同概念容量进行推断，降低测试时干预成本；

**💡 创新点**

创新点在于将概念按最大相关性与最小冗余顺序排列，并通过Matryoshka结构实现多层次推断，理论证明干预成本从O(K)降到O(log K)，并在单模型中实现可扩展的解释与干预；

**🔧 技术方法**

使用了mRMR特征排序、Matryoshka表示学习、端到端联合训练、可共享权重的Efficient MCBM、以及对概念与任务的多层次损失；

**📊 数据集**

在CUB-200-2011、LAD和CelebA三个细粒度数据集上进行实验；

**📈 对比分析**

与独立训练的CBM、Sequential CBM、Sparse/Label‑free CBM、随机排序的MCBM以及Efficient MCBM等基线比较。MCBM在保持与独立模型相当的精度的同时，显著减少了干预次数，mRMR排序后在低维度下精度高于随机，实验显示在CUB、LAD、CelebA上均可获得与全概念模型相近的准确率；

**⚠️ 局限性**

主要限制包括：对概念排序的依赖，若mRMR估计不准可能导致层次失效；共享权重版本在高精度需求下略低于独立head版本；对零拷贝或预训练CLIP等跨模态模型的表现仍有限，需要进一步微调。

---

## 175. Mapping the Winds of Stance Dynamics using Potential Landscape Models

**arXiv ID:** 2605.20363 | [PDF](https://arxiv.org/pdf/2605.20363v1)

**作者:** Benjamin Steel `[一作]` (McGill University), Derek Ruths `[通讯]` (McGill University)

**通讯引用:** 5378 | [OpenAlex ID](https://openalex.org/A5041513167)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究提出一种基于潜在态度空间和势能景观的框架，用以跟踪和描述社交媒体上多维度、多平台的立场大规模转变。

**💡 创新点**

创新点在于将立场检测与概率主成分分析相结合，构造连续时间轨迹，再利用势能景观神经网络捕捉立场变化的全局梯度，实现可视化和预测；同时首次在跨平台加拿大政要数据上验证该模型。

**🔧 技术方法**

技术包括：自动立场目标提取、微调的LLM立场分类器、贝叶斯核回归平滑时间序列、概率PCA进行缺失值插补与降维、势能景观多层感知机（MLP）拟合时空势能、蒙特卡洛Dropout估计不确定性。

**📊 数据集**

使用了从X/Twitter、TikTok、Instagram、Bluesky四个平台收集的约2719万条公开帖文，涉及4108位加拿大政要与影响者，时间跨度2022‑2025年。

**📈 对比分析**

与无运动基线、Holt衰减指数平滑、阻尼Theta模型等做对比；势能景观模型在7–360天预测范围内平均降低约5.7% MSE，且对政要的预测显著优于基线，但在720天后性能下降，整体提升有限。

**⚠️ 局限性**

局限包括：立场检测准确性随时间变化、仅使用文本与名词短语作为目标忽略语境、缺乏多模态信息、假设马尔可夫过度简化、势能景观对数据稀疏与非平稳性敏感。

---

## 176. Personality Engineering with AI Agents: A New Methodology for Negotiation Research

**arXiv ID:** 2605.20554 | [PDF](https://arxiv.org/pdf/2605.20554v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 177. Proximal State Nudging: Reducing Skill Atrophy from AI Assistance

**arXiv ID:** 2605.20355 | [PDF](https://arxiv.org/pdf/2605.20355v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 178. Residual Paving: Diagnosing the Routing Bottleneck in Selective Refusal Editing

**arXiv ID:** 2605.20262 | [PDF](https://arxiv.org/pdf/2605.20262v1)

**作者:** Bryce Hinkley `[一作]` (University of Texas at San Antonio), Peyman Najafirad `[通讯]` (University of Texas at San Antonio)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种路由残差编辑方法，能够在冻结的指令调优模型上对特定拒绝提示进行非拒绝编辑，同时保持对善意提示和有害提示的拒绝行为不变。

**💡 创新点**

创新点在于将路由选择（判断是否属于编辑集合）与残差编辑（具体如何修改内部状态）分离开来；利用路由器在早期层提取边界特征，并用门控混合专家在后层实现编辑；同时引入oracle路由诊断来定位路由选择的瓶颈。

**🔧 技术方法**

技术包括：冻结的Transformer残差架构、早期层特征提取路由器、可学习的门控与专家混合残差更新、分阶段训练（门控预训练、对比热身、监督编辑、门控校准）、阈值软/硬门控与辅助否决机制。

**📊 数据集**

使用的数据集：SALAD‑Bench（被基模型拒绝的提示）、Alpaca‑style 指令（善意保持提示）、HarmBench（有害保持提示），并在六个指令调优模型上进行跨模型实验。

**📈 对比分析**

与传统单向激活驱动的拒绝控制方法比较（如ActAdd、DIM），路由残差编辑在主分割上将编辑提示的拒绝率从88.6%降至4.0%，保持善意保持分布95.5%、有害保持分布87.3%；相比之下单向方法拒绝率仍高于80%。在六个模型上，oracle路由诊断显示路由选择是主要瓶颈。

**⚠️ 局限性**

局限性包括：编辑桶仅为实验性安全重构集合，未覆盖所有安全情形；有害保持数据量较小导致置信区间宽；oracle路由仅为诊断而非可部署；模型仅在贪婪解码、特定Transformer架构下验证，未测试更大或混合专家模型；对拒绝标注依赖OpenAI裁判，需替代开源裁判以复现。

---

## 179. What Do Biomedical NER and Entity Linking Benchmarks Measure? A Corpus-Centric Diagnostic Framework

**arXiv ID:** 2605.20537 | [PDF](https://arxiv.org/pdf/2605.20537v1)

**作者:** Robert Leaman `[一作]`, Zhiyong Lu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于语料的诊断框架，用以从注释、概念链接、训练-测试划分、文档元数据及术语映射中提取多维度统计，评估生物医学NER/EL基准语料的测量属性。

**💡 创新点**

将语料的规模/密度、词汇与概念结构、重叠风险、元数据构成以及术语覆盖等五大诊断维度系统化，为基准语料提供可解释的、可比较的属性分析。

**🔧 技术方法**

使用Python实现YAML可配置的管道，支持BioC、PubTator、BRAT等多种注释格式转换，计算统计指标并通过交互式仪表盘可视化结果。

**📊 数据集**

对九个常用的生物医学NER/EL语料库（AnatEM、BC5CDR、BioID、CHEMDNER、CRAFT、CellLink、JNLPBA、NCBI‑Disease、NLM‑Chem）进行诊断。

**📈 对比分析**

通过在上述五类诊断维度上对各语料进行对比，展示即便任务标签相同，不同语料在评估信号、泛化需求、泄漏风险、文献及概念覆盖等方面存在显著差异；论文未给出具体模型性能，只说明诊断揭示了基准差异。

**⚠️ 局限性**

框架只能描述语料结构，无法直接预测模型排名或基准饱和度；多义性指标无法区分真实多义、规范不清或标注错误；术语覆盖依赖于所选术语版本；主题映射仅为轻量级诊断，未提供严格的学科划分。

---

## 180. Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?

**arXiv ID:** 2605.20606 | [PDF](https://arxiv.org/pdf/2605.20606v1)

**作者:** Muquan Li `[一作]` (UESTC), Tao He `[通讯]` (UESTC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一个鲁棒数据集蒸馏框架C^2R，结合攻击感知课程与对比鲁棒损失来提升合成数据的抗攻击性能。

**💡 创新点**

从鲁棒边距视角提出攻击感知课程（AAC）与对比鲁棒损失（CRL），并开发了高效的线搜索PGD（LS‑PGD）生成对抗样本，实现了对最小鲁棒边距的精确排序与聚焦。

**🔧 技术方法**

采用鲁棒边距理论、投影梯度下降（PGD）+线搜索、对比学习框架、类平衡内存队列和高效的LS‑PGD内部攻击器。

**📊 数据集**

在CIFAR‑10、CIFAR‑100、Tiny‑ImageNet以及多个ImageNet‑1K子集（ImageNette、ImageWoof、ImageFruit、ImageMeow、ImageSquawk、ImageYellow）上进行实验。

**📈 对比分析**

与SRe^2L、D^4M、ROME等鲁棒蒸馏基线对比，在六种攻击（FGSM、PGD、CW、VMI、Jitter、AutoAttack）下平均提升约2.8%的鲁棒准确率，Drop率更低，训练时间更短。

**⚠️ 局限性**

对抗强度仍依赖于攻击器，IPC升高时鲁棒准确率略有下降；需对权重η进行调参；对极端小样本集鲁棒性未得到充分验证。

---

## 181. Goodbye Drift: Anchored Tree Sampling for Long-Horizon Video-to-Video Generation

**arXiv ID:** 2605.20476 | [PDF](https://arxiv.org/pdf/2605.20476v1)

**作者:** Matthew Bendel `[一作]` (Descript Inc), Xingzhe He `[通讯]` (Descript Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无训练、适用于离线长时视频生成的 Anchored Tree Sampling（ATS）采样器，利用双向生成器将长时序拆分为稀疏锚点生成和稠密叶子填充，形成层次化、并行的树形推理流程。

**💡 创新点**

创新点在于：
1) 通过根级别的全时段条件化 + 递归双向锚点生成，将传统顺序 AR 的 K 步拆解为 O(log T) 步的并行树；
2) 使用仅依赖基础双向模型的无训练调度，保持模型不变；
3) 通过稀疏到稠密的层次填充直接抑制漂移和重置跳跃，显著提升长时视频质量。

**🔧 技术方法**

技术细节：
- 双向视频生成器（如 LTX‑2.3、Wan 2.1 + VACE）作为黑盒；
- Anchored Tree Scheduler：根调用（仅条件化）→多级锚点细化 → 叶子稠密填充；
- 并行兄弟节点、层级递归；
- 采用两侧锚点约束的 “bounded infilling”；
- 评估使用无参考指标 AQ、IQ。

**📊 数据集**

数据集：
- 30 min 长时视频（每个维度 5 条，分别为 inpainting、outpainting、edge、pose、depth）；
- 使用公开工具生成相应的条件轨（DepthAnything V3、Canny、DWPose）。

**📈 对比分析**

与方法比较：
- 对比两种主流 AR 长时 V2V 基线 LongLive 与 Reward Forcing；
- 结果显示：
  * AQ +4.0 / IQ +6.6（LongLive）
  * AQ +5.7 / IQ +6.2（Reward Forcing）；
  * 片段漂移与重置跳跃均显著降低；
  * 生成速度提升 5.3×（2000 s 视距）并可达到 3.3× 以上实时。

**⚠️ 局限性**

局限性：
1) 锚点质量不佳会污染整个子树；
2) 对弱运动引导（如 depth、Canny）仅使用关键帧不足，需视频级锚点；
3) 对离框实体缺乏跨叶子一致性，可能出现外观差异；
4) 仅适用于 V2V；缺乏生成稀疏视频的专用模型，难以推广至 T2V 或动态摄像；
5) 需进一步加入稀疏生成器、DAG 语义结构和参考输入以解决上述问题。

---

## 182. OmniISR: A Unified Framework for Centralized and Federated Learning via Intermediate Supervision and Regularization

**arXiv ID:** 2605.20276 | [PDF](https://arxiv.org/pdf/2605.20276v1)

**作者:** Wei-Bin Kou `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**通讯引用:** 4142 | [OpenAlex ID](https://openalex.org/A5020953714)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 OmniISR 框架，将集中式学习（CL）、联邦学习（FL）与两者混合训练统一在同一目标下，利用中间监督与正则化（MI 与 NE）对网络进行多层引导，提供了理论收敛、漂移、梯度对齐与鞍点逃逸时间等非渐近性保证，并在多种模型与数据集上做了大规模实验；

**💡 创新点**

①统一框架实现 CL‑FL 的协同优化；②引入互信息（MI）作为中间监督和负熵（NE）正则化，既抑制客户端表示漂移又保持泛化；③给出全模式 𝒪(1/√T) 收敛、漂移上界、梯度非冲突保证和混合模式鞍点逃逸时间的理论证明；④证明混合梯度能加速逃逸并实现 CL‑FL 兼容；

**🔧 技术方法**

互信息（MI）中间监督、负熵（NE）正则化、通用 ISR 层选择、Adam/SGD 优化、FedAvg、FedProx、FedDyn、FedAvgM、FedIR、MOON、SCAFFOLD、FedGau、BalanceFL 等联邦算法；理论分析采用非渐近收敛、漂移分析、梯度对齐与鞍点逃逸时间证明；

**📊 数据集**

Cityscapes、CamVid 与 SynthiaSF 三大城市驾驶语义分割数据集；

**📈 对比分析**

与仅使用输出层交叉熵监督的 CL 基线以及多种 FL 基线（FedAvg、FedProx、FedDyn 等）进行对比，指标包括 mIoU、mF1、mPre、mRec；OmniISR 在 CL 与 FL 模式均实现显著提升，FL 上平均提升 2.03 点，CL 上 1.76 点；显著缩小 CL‑FL 性能差距 22.60%；在 48 条 FL 算法对比中获得 37/48 计分指标提升；

**⚠️ 局限性**

在部分架构与数据集上提升有限；需要对 ISR 的层数、位置、权重进行手工调优；计算开销略大；理论假设如梯度无偏、L‑光滑性等在实际极端异构环境中可能不完全满足。

---

## 183. Hybrid Edge-HPC Systems for Low-Latency Data-Driven Inference

**arXiv ID:** 2605.20532 | [PDF](https://arxiv.org/pdf/2605.20532v1)

**作者:** Liubov Kurafeeva `[一作]` (University of California Santa Barbara), Rich Wolski `[通讯]` (University of Notre Dame)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种混合边缘–高性能计算的反向后填充架构，实现实时推断与异步模型改进的闭环系统；在数字农业空气流动推断任务上进行了部署与评估。

**💡 创新点**

核心创新在于将推断与昂贵模拟训练解耦，利用反向后填充将HPC资源转变为机会性模型改进引擎，并通过可插拔的代理模型、分布式日志与5G网络实现异步、可扩展的协同工作。

**🔧 技术方法**

采用高精度CFD模拟（OpenFOAM）、物理信息神经网络（PINN）、傅里叶神经算子（FNO）和主成分回归（PCR）等代理模型；通过CSPOT分布式日志、RBFDM文件传输、私有5G切片和Raspberry Pi边缘推断设备进行系统集成。

**📊 数据集**

使用真实的CUPS（柑橘保护屏蔽）农场传感器流数据（风速、风向、温度、湿度）作为模拟参数与推断评估数据。

**📈 对比分析**

将专用集群、NERSC共享HPC及其组合的模型发布周期、准确性衰减（MAE）和模型陈旧度进行对比；结果显示专用集群平均周期为134.8 min，NERSC为80.0 min，混合系统为50.0 min，平均模型陈旧度降低2.7倍，推断误差保持在传感器噪声阈值以下。

**⚠️ 局限性**

局限在于模拟与训练仍是系统瓶颈，HPC队列等待不确定性仍导致模型更新不规律；代理模型的超参数（如历史窗口）未动态优化；实验仅在农业流动域完成，需验证在更广泛场景中的适用性。

---

## 184. Supervised Latent Restructuring for Small-Data Quantum Learning in Plant Phenomics

**arXiv ID:** 2605.20413 | [PDF](https://arxiv.org/pdf/2605.20413v1)

**作者:** Alakananda Mitra `[一作]` (University of Nebraska--Lincoln), Chittaranjan Ray `[通讯]` (University of Nebraska--Lincoln)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在极少样本的植物表型学任务中，提出了从1280维深度图像嵌入到11维监督式潜在空间的压缩工作流，并在该潜在空间上实现了量子核对齐（QKA）以训练可调量子核；

**💡 创新点**

创新点在于将PCA降噪与LDA监督重构相结合形成的“监督潜在重构（SLR）”流程，显著提升潜在空间的几何可分性，并将该结构化低维空间用于量子核学习，探索了小样本量子学习的几何瓶颈；

**🔧 技术方法**

使用的技术包括EfficientNet‑B0特征提取、PCA降噪、LDA监督压缩、角度感知重缩放（AALR）、基于GPU的量子特征映射与Quantum Kernel Alignment（QKA）以及经典基线分类器（线性SVM、RBF‑SVM、Random Forest、XGBoost）和QSVC；

**📊 数据集**

实验基于Plant Pathology 2021苹果病害细粒度12分类数据集，训练样本每类10个，验证/测试样本分配按说明；

**📈 对比分析**

通过对比PCA‑64与SLR‑11两种表示的经典基线表现以及QKA的QSVC性能，发现SLR‑11显著提升线性学习器的准确率（如Linear SVM从0.299提升到0.328），但对高非线性模型造成退化；QKA在11维潜在空间上实现了约23.6%准确率、0.212宏F1，低于PCA‑64基线但高于随机；

**⚠️ 局限性**

主要限制在于：①在极少样本和高维压缩下，LDA压缩可能丢失对非线性模型有用的细粒度信息；②量子核对齐在当前优化预算和计算资源下收敛有限，难以突破经典基线；③对混合病害和视觉相似类别的区分仍不充分。

---

## 185. Modality-Decoupled Online Recursive Editing

**arXiv ID:** 2605.20273 | [PDF](https://arxiv.org/pdf/2605.20273v1)

**作者:** Siyuan Li `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 116315 | [OpenAlex ID](https://openalex.org/A5100336796)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种模态解耦的在线递归编辑框架M‑ORE，用于在不耗费大量计算与存储的前提下，持续修正多模态大型语言模型的知识；

**💡 创新点**

通过将跨模态冲突与编辑间干扰两大瓶颈分别归结为统计量不匹配与编辑子空间共振，并在统一的近端投影视角下设计了常数时间闭式更新，显著降低了编辑时的互相干扰；

**🔧 技术方法**

采用分层局部二阶统计量、固定正交低秩写空间以及Sherman‑Morrison递归闭式公式，实现了仅需维护$r$维矩阵并在每次编辑时做$O(r^2)$运算的编辑更新；

**📊 数据集**

在BLIP2‑OPT（2.7B）和LLaVA‑v1.5（7B）两种主流MLLM上，用E‑VQA和E‑IC（包含精细视觉定位）两套编辑基准数据集进行实验；

**📈 对比分析**

与FT‑L/FT‑M、MEND、AlphaEdit、SERAC、IKE、LiveEdit等参数修改/保持型编辑器对比，M‑ORE在长期编辑（100步）下保持更高的可靠性、通用性与局部性，平均得分提升约2–8个百分点，且每步计算与内存开销保持$O(1)$；

**⚠️ 局限性**

主要局限在于写空间需要预先正交初始化，对新出现的模态或大规模参数扩展时需重新构建；此外，虽然保持了局部性，但对极大规模模型的参数维度与多模态交互细节的适应性仍待进一步验证。

---

## 186. Parallel LLM Reasoning for Bias-Resilient, Robust Conceptual Abstraction

**arXiv ID:** 2605.20194 | [PDF](https://arxiv.org/pdf/2605.20194v1)

**作者:** Aisvarya Adeseye `[一作]` (University of Turku), Adeyemi Adeseye `[通讯]` (Brilloconnetz Partners avoin yhtiö)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现PECII框架，用于改进大型语言模型对长文本的主题抽象，消除顺序处理导致的累计偏差和无根据信任合并问题。

**💡 创新点**

创新点在于将分块语义分割、并行独立推理与证据锚定合并相结合，形成一套结构化、可追溯且对模型规模不敏感的长文本分析流程。

**🔧 技术方法**

使用了基于Transformer的预训练LLM（LLaMA、Qwen、ChatGPT）进行推理，并结合Python实现的多层架构：语义chunking、并行推理、嵌入相似度检索、证据对齐、聚类合并和可靠性加权排序。

**📊 数据集**

利用82名受访者的半结构化访谈转录（约8k–13k词）作为实验数据集，并通过NVivo人工编码生成专家金标准。

**📈 对比分析**

通过与单文本、顺序分块和并行分块三种执行策略比较，使用遗漏率、早期块主导指数、证据可追溯性、错误声称率、主题压缩比等指标；实验显示并行分块可将遗漏率降低约84%、证据可追溯性提升约130%、错误声称率下降约90%，并显著减少跨模型差异。

**⚠️ 局限性**

局限性包括：依赖严格的引证与句子级别引用，可能在多语言或不同文本风格下产生误锚；并行计算资源需求高，阈值设置对结果敏感；仅在访谈文本上评估，泛化性待进一步验证。

---

## 187. Latent Geometry as a Structural Monitor: Eigenspace Alignment for Anomaly Detection in Anonymity Networks

**arXiv ID:** 2605.20391 | [PDF](https://arxiv.org/pdf/2605.20391v1)

**作者:** Vaibhav Chhabra `[一作]` `[通讯]`, Vaibhav Chhabra

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一种基于潜在几何的结构监测框架，用双观察者（几何观测器CDAE与热力学观测器GRBM）结合CCA分析，对Tor网络的行为群体进行结构异常检测。

**💡 创新点**

创新点包括：① 通过EJT（Jacobian Trace特征值分解）发现并证明了一个九维稳健的“硬轴”子空间；② 将几何与热力学两种观测器桥接，形成可识别多种异常类型的门控体系；③ 能在网络拓扑未变化时捕获连通性下降等隐蔽故障。

**🔧 技术方法**

使用技术包括：Contractive Denoising Autoencoder、Gaussian Restricted Boltzmann Machine、Canonical Correlation Analysis、Jacobian特征值分解（EJT）、Monte‑Carlo噪声基准、门控阈值与FPR评估。

**📊 数据集**

数据集为Tor Onionoo API公开的191维日常度量，覆盖2019年1月19日至2026年4月21日的67个连续观察窗口，包含节点容量、地理、角色等特征。

**📈 对比分析**

方法评估通过在24个已知稳定窗口上验证门控误报率（主要门控FPR=0.0%），并在1000次高斯噪声模拟中计算噪声基准，最终在2026年2月20日的一次已确认Cloudflare BGP撤销事件中检测到高显著性（16.8σ）的结构变形，表明检测精度高。

**⚠️ 局限性**

局限性：① 仅有单一外部验证事件；② CCA旋转门阈值FPR高达49%，非单独报警；③ 仅捕获线性共享结构，可能漏检非线性变化；④ 依赖Tor特定角色与冻结编码器，需定期再训练；⑤ 只检测结构变化，无法直接推断攻击原因。

---

## 188. Weight Decay Regimes in Grokking Transformers: Cheap Online Diagnostics

**arXiv ID:** 2605.20441 | [PDF](https://arxiv.org/pdf/2605.20441v1)

**作者:** Lucky Verma `[一作]` `[通讯]`, Lucky Verma

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在小型 transformer 上训练模组算术任务，研究 grokking 过程，并通过将 weight decay 作为单一可控参数，绘制 λ×N 三阶段相位图；同时提出两种低成本在线诊断（平均余弦相似度与熵标准差），验证其在多架构、多任务和干预实验中的一致性；并使用 Lean 4 形式化验证诊断的数学正确性。

**💡 创新点**

① 将 weight decay 定位为控制 grokking 过渡的经验参数；② 提出两种基于注意力激活的即时诊断，成本低、可在线监测；③ 在 Transformer、MLP、LSTM、Mamba 等不同架构和四种算术任务上验证相同相位结构；④ 用 Lean 4 证明诊断公式的严谨性。

**🔧 技术方法**

Transformer 模型、AdamW 优化器、weight decay 控制、在线注意力相似度与熵标准差诊断、逻辑斯蒂回归与功率律拟合、Bootstrap 与 Jackknife 置信区间、随机种子多样化、跨架构与跨任务实验设计、Lean 4 形式化验证。

**📊 数据集**

模组算术数据集（mod_+, mod_-, mod_×, mod_÷），模数 p=97，共 9409 训练/测试样本；实验覆盖参数规模 0.82M–85M；还包括 4 层 MLP、LSTM 与 Mamba 四种架构。

**📈 对比分析**

通过 λ 与 N 的二维扫描绘制三阶段相位图，逻辑斯蒂回归估计 λ_c≈0.0158（95% CI），功率律拟合得到 ν≈0.757；在不同任务、架构中测得 λ_c 变化，head 重新初始化实验验证 Phase‑2 幅度受 head 结构影响；长周期实验观察 retention，随机森林预测 AUC≈0.80。整体表现显示 weight decay 对 grokking 过渡具有显著可控性。

**⚠️ 局限性**

研究仅限于模块算术任务和 Transformer attention 架构，未验证语言模型、大规模或非注意力模型；未给出热力学或 universality 类理论；weight decay 阈值需在不同架构/任务上重新校准；诊断只适用于注意力 head 结构；实验规模有限，未覆盖更大参数或更复杂任务，理论解释仍待完善。

---

## 189. Tool-Augmented Agent for Closed-loop Optimization,Simulation,and Modeling Orchestration

**arXiv ID:** 2605.20190 | [PDF](https://arxiv.org/pdf/2605.20190v1)

**作者:** Liyuan Deng `[一作]` (Northwestern Polytechnical University), Huaxi Huang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 356 | [OpenAlex ID](https://openalex.org/A5049609631)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种基于工具增强的强化学习框架COSMO-Agent，用LLM自动完成闭环CAD–CAE优化流程。

**💡 创新点**

创新点在于将CAD–CAE管线建模为交互式RL环境，采用多约束奖励和基于交互日志的奖励函数，训练LLM在多工具、随机失败环境下实现结构参数与材料协同优化。

**🔧 技术方法**

使用LLM（Qwen3-8B）+GRPO强化学习，工具集包括CAD生成、有限元求解、结果提取和成本计算，以及多约束奖励设计。

**📊 数据集**

使用工业级可执行CAD–CAE基准数据集，约2万条任务覆盖25种部件类别，包含可编辑的参数模板、物理仿真结果和成本指标。

**📈 对比分析**

与多种公开/闭源LLM（如Qwen3-30B、Gemini-3-Flash等）在统一接口、工具调用预算下对比，COSMO-Agent在完整成功率、约束满足率和工具调用效率上均位居榜首。

**⚠️ 局限性**

局限性包括仅针对单体线性静力问题，未涵盖接触、装配、多物理耦合等复杂场景，且对工具链错误恢复和更大动作空间的鲁棒性还有待提升。

---

## 190. Accelerating Video Inverse Problem Solvers with Autoregressive Diffusion Models

**arXiv ID:** 2605.20624 | [PDF](https://arxiv.org/pdf/2605.20624v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 191. GROW: Aligning GRPO with State-Action Modeling for Open-World VLM Agents

**arXiv ID:** 2605.20246 | [PDF](https://arxiv.org/pdf/2605.20246v1)

**作者:** Xiongbin Wu `[一作]` (Shanghai Jiao Tong University), Wei Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 86659 | [OpenAlex ID](https://openalex.org/A5100431792)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对开放世界视觉语言模型代理的强化学习框架 GROW，采用轨迹拆分为状态-动作样本并在同一组内计算相对优势，避免了长上下文噪声导致的梯度问题。

**💡 创新点**

创新点在于将 GRPO 应用于开放世界任务时，将全轨迹改为细粒度状态-动作样本，并给出理论替代分析证明相对优势信号仍然有效，从而实现更高效的策略改进。

**🔧 技术方法**

使用了 GRPO 的变体 GROW、状态-动作分解与相对优势计算、Qwen2-VL-7B-Instruct 视觉语言模型、以及 Minecraft 环境进行训练和评估。

**📊 数据集**

使用 MCU benchmark（800+ Minecraft 任务集合），涵盖建造、GUI 操作、战斗等多类任务。

**📈 对比分析**

与多种 SOTA 方案（VPT、STEVE-1、ROCKET-1、JARVIS-VLA 等）进行对比，GROW 在所有三类任务的成功率均达到最高，并显著减少完成任务所需的步骤；相较于 PPO，GROW 收敛更快、性能更稳定。

**⚠️ 局限性**

主要局限在于缺乏记忆模块，难以处理需要长期记忆的长时限任务；此外实验仅在 Minecraft 这类高保真模拟器中验证，尚未扩展到更广泛的开放世界环境。

---

## 192. Scalable Multi-robot Motion Planning via Hierarchical Subproblem Expansion and Workspace Decomposition Refinement

**arXiv ID:** 2605.20395 | [PDF](https://arxiv.org/pdf/2605.20395v1)

**作者:** Isaac Ngui `[一作]` (University of Illinois Urbana-Champaign), Nancy M. Amato `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8794 | [OpenAlex ID](https://openalex.org/A5050205557)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种可递归细化工作空间分解的多机器人运动规划方法CIPHER，利用高层MAPF与低层连续规划的分层冲突检测与递归分解来避免联合空间搜索

**💡 创新点**

引入“分辨率引导”机制，即在规划过程中动态细化冲突区域的网格分解，以实现更细粒度的协调；并设计分层冲突解决策略，先尝试局部细化、再扩展空间，最后回退到联合规划

**🔧 技术方法**

工作空间网格分解、冲突基础搜索（MAPF，CBS）、区域引导的RRT/DB-RRT、冲突检测与递归分解、回退到联合RRT

**📊 数据集**

实验使用合成环境：空白环境、带房间的障碍、随机低/中/高密度障碍；机器人状态通过随机生成起止点

**📈 对比分析**

与Coupled RRT、Decoupled RRT、sRRT、ARC、WG‑DaSH、MRdRRT（几何）以及Coupled/Decoupled/K‑ARC/K‑WG‑DaSH（动力学）对比；CIPHER在所有环境均实现100%成功率，规划时间比最慢方法低约十倍，优于现有混合方法，PP‑RG‑RRT规划最快但成功率低

**⚠️ 局限性**

对机器人尺寸相近的网格尺寸敏感，过小导致路径难寻，过大导致MAPF拥塞；分辨率引导在动力学问题上可能增加计算量；最终若仍冲突需回退到联合规划，导致耗时上升

---

## 193. Music of Changing Lines: Toward a Culturally Situated Approach to the I-Ching

**arXiv ID:** 2605.20386 | [PDF](https://arxiv.org/pdf/2605.20386v1)

**作者:** Ling Qi `[一作]` (Georgia Institute of Technology), Alexandria Smith `[通讯]` (Georgia Institute of Technology)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一个交互式Web系统，让用户通过投掷硬币完成易经的问卜流程，并在投掷、解释、音乐生成的三个阶段实时产生音频体验。

**💡 创新点**

将易经从纯粹的随机工具转变为富含意义的仪式框架，利用大型语言模型做为解释中介，并将解释结果转化为音乐生成指令，实现了过程驱动、参与式的 AI 音乐创作。

**🔧 技术方法**

使用Web Audio API和Tone.js实现音频引擎，调用Google Gemini 2.5 Flash进行文本解释，利用Lyria文本到音乐模型生成响应音乐，配合传统音色库和自定义概率算法。

**📊 数据集**

采用易经六十四卦及其爻辞的公开释义（Gua Ci、Yao Ci），传统声库中的五声音阶乐器样本，以及用户自选的问题文本作为输入数据。

**📈 对比分析**

与约翰·凯奇的随机操作方法对照，说明本系统在保留随机性基础上加入了语义解释；性能方面以定性评估为主，未给出数值指标，侧重用户体验和仪式感。

**⚠️ 局限性**

主要局限包括：Lyria 生成的音乐对细粒度提示的响应不够稳定，模型对情绪和结构的把握仍有限；易经解释的多样性导致结果难以统一评估；系统缺乏大规模用户测试和客观性能评测。

---

## 194. Miller-Index-Based Latent Crystallographic Fracture Plane Reasoning with Vision-Language Models

**arXiv ID:** 2605.20416 | [PDF](https://arxiv.org/pdf/2605.20416v1)

**作者:** Qinwu Xu `[一作]`, Yifan Jiang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究多模态大语言模型（MLLM）是否能将米勒指数（hkl）作为结构化潜在变量，用于推理裂纹几何；

**💡 创新点**

创新点在于提出潜在引导推理框架，让模型既能推断潜在晶面，又能判断该表示在当前裂纹图像中是否适用，并证明模型能在物理无效时正确拒绝；

**🔧 技术方法**

主要技术包括使用基于提示的MLLM进行潜在推断与一致性判断，构造理想立方体–晶面交叉的合成数据，并在此基础上评估模型在真实裂纹图像上的表现；

**📊 数据集**

使用的数据集包含：合成的立方体-晶面交叉图像（2D截面与3D配对），以及来自陶瓷、玻璃、金属、混凝土等多种材料的真实裂纹图像；

**📈 对比分析**

方法评估通过比较模型在理想合成场景下的潜在推断准确率、对不一致配对的判定准确率以及在真实裂纹中对米勒指数可用性的拒绝率；结果显示在理想单晶面场景中准确率高，且模型能在多面或非晶材料中正确拒绝；

**⚠️ 局限性**

局限性在于米勒指数仅适用于单晶面主导的裂纹，无法覆盖多面、非晶或复合材料的复杂裂纹；模型在细粒度索引区分上表现有限，且需要显式建模物理有效域以避免误用。

---

## 195. Do Vision--Language Models Understand 3D Scenes or Just Catalogue Objects?

**arXiv ID:** 2605.20448 | [PDF](https://arxiv.org/pdf/2605.20448v1)

**作者:** Animesh Maheshwari `[一作]` (Deccan AI), Nishit Verma `[通讯]` (Deccan AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个3,034样本、人工标注的基准，用于评估视觉‑语言模型在三维场景结构推理中的能力，并通过行为、注意力与因果机制三维分析探究其内部机制。

**💡 创新点**

首次在视觉‑语言模型中系统评测深度排序遮挡、光学几何推理和体积重排三种三维推理任务，揭示视觉‑token 合并步骤是失效的关键瓶颈。

**🔧 技术方法**

采用多任务对比评测、Depth‑Guided Attention Relevance (DGAR) 与激活补丁因果追踪等技术，对ViT+LM架构进行深入解析，并结合人类标注的评估。

**📊 数据集**

使用合成的Nano Banana 2室内场景图像，并人工验证遮挡深度与光学几何，利用SAM 3生成目标对象分割掩码。

**📈 对比分析**

在六款VLM（GPT‑5.2、Gemini 3.1、GLM‑4.6V、Qwen3.5‑VL‑397B‑A17B、Qwen3‑VL‑30B‑A3B‑Thinking等）上进行对比实验，体积规划准确率高达53–97%，而遮挡推理仅6–45%，光学几何低至1–7%，显现出显著的功能分离。

**⚠️ 局限性**

局限在于仅使用合成图像，缺乏真实照片验证；机制分析仅针对单一开源VLM；实验场景为单视角、桌面尺度、英语提示，可能限制结果的通用性。

---

## 196. Leveraging Large Language Models for Sentiment Analysis: Multi-Modal Analysis of Decentraland's MANA Token

**arXiv ID:** 2605.20192 | [PDF](https://arxiv.org/pdf/2605.20192v1)

**作者:** Xintong Wu `[一作]` (University of Pennsylvania), Luyao Zhang `[通讯]` (Duke Kunshan University)

**通讯引用:** 4109 | [OpenAlex ID](https://openalex.org/A5100447104)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

利用大型语言模型分析Decentraland Discord社区情绪，并结合多模态金融数据（价格、成交量、市值）构建LSTM模型，预测MANA代币的每日收益。

**💡 创新点**

首次将RoBERTa情绪分析与多模态特征融合应用于元宇宙虚拟经济中的加密货币价格预测，展示社区情绪对短期波动的预测价值。

**🔧 技术方法**

RoBERTa情绪分类器、情绪聚合、LSTM时间序列网络以及MSE/MAE/R²评估指标。

**📊 数据集**

CoinMarketCap提供的MANA每日OHLCV及市值数据，配合从Discord #general频道抓取的5,513条消息，构成366天的多模态数据集。

**📈 对比分析**

对比仅用典型价格的基线LSTM与加入成交量、市值与情绪的多模态LSTM，后者MSE从0.002100降至0.001528，MAE从0.0297降至0.0241，但两者R²均为负值（-0.4185/-0.0321），表明解释力有限。

**⚠️ 局限性**

模型对方向性预测的解释力不足，情绪信号主要反映平台用户体验而非宏观市场因素，导致R²为负，预测精度仍有限。

---

## 197. AgentCo-op: Retrieval-Based Synthesis of Interoperable Multi-Agent Workflows

**arXiv ID:** 2605.20425 | [PDF](https://arxiv.org/pdf/2605.20425v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 198. How You Move Tells What You'll Do: Trajectory-Conditioned Egocentric Prediction

**arXiv ID:** 2605.20388 | [PDF](https://arxiv.org/pdf/2605.20388v1)

**作者:** Sejoon Jun `[一作]` (Northeastern University), Lorenzo Torresani `[通讯]` (Northeastern University)

**通讯引用:** 25637 | [OpenAlex ID](https://openalex.org/A5082736347)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在第一人称视频中，提出TrajPilot模型用于预测未来动作、计划和事件结果；

**💡 创新点**

核心创新在于将未来相机轨迹作为细粒度的条件信号来驱动预测，并通过轨迹候选检索+门控-排名实现无轨迹测试；

**🔧 技术方法**

使用V-JEPA 2.1视觉编码器、EgoVLPv2对齐嵌入、轨迹对齐编码器、因果Transformer预测器以及门控-排名评分器；

**📊 数据集**

在四个 egocentric 基准上评估：Atomic、Keystep、Ego4D GoalStep、EgoPER；也在 EPIC‑Kitchens‑100 进行跨域迁移；

**📈 对比分析**

与 VLM (Qwen‑VL) 与结构化规划器 (SCHEMA, PDPP) 对比，TrajPilot 在开放词表和封闭词表情境下均显著提升（多步预测 M@1 约提升 10–20pp，长程更明显）；

**⚠️ 局限性**

限制在于门控-排名的候选选择仍低于oracle，轨迹优势在对象识别驱动任务上（如 EK100）不显著，且需要检索候选池导致推理开销。

---

## 199. FlowLM: Few-Step Language Modeling via Diffusion-to-Flow Adaptation

**arXiv ID:** 2605.20199 | [PDF](https://arxiv.org/pdf/2605.20199v1)

**作者:** Runzhe Zhang `[一作]` (Shanghai Jiao Tong University), Peilin Zhao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将预训练的连续扩散语言模型通过细调转换为流匹配语言模型，直接预测干净数据并以平均速度采样，实现在极少步数下生成高质量文本。

**💡 创新点**

创新点在于将扩散模型的曲线路径重新对齐为直线流路径，使用平均速度而非瞬时速度指导采样，证明在微调阶段保持z₀预测目标能够显著提升训练稳定性与生成质量。

**🔧 技术方法**

采用流匹配（Flow Matching）框架、z₀预测损失、平均速度采样、对数时间尺度重缩放、以及统一的正则化约束，基于DiffuSeq预训练模型。

**📊 数据集**

使用问答生成（Quasar‑T）、文本简化（Wiki‑Auto）和改写（QQP）三个公开Seq2Seq数据集。

**📈 对比分析**

与原始DiffuSeq（2000步）和DPM‑Solver（10步）以及其他加速方法（DLM‑One、Reflow、Perflow、FMSeq）对比，FlowLM在1-5步内实现与原始模型相当甚至更优的BLEU、ROUGE‑L、BERTScore和多样性指标，同时推理时间降低至毫秒级。

**⚠️ 局限性**

局限性包括：对需要细粒度编辑的任务仍可能表现不足；仅适用于连续扩散模型，无法直接迁移至离散模型；在极少步数（如一步）下对参数规模要求较高，且对大模型的实验尚未充分验证。

---

## 200. Robust Subspace-Constrained Quadratic Models for Low-Dimensional Structure Learning

**arXiv ID:** 2605.20300 | [PDF](https://arxiv.org/pdf/2605.20300v1)

**作者:** Zheng Zhai `[一作]` (Beijing Normal University), Xiaohui Li `[通讯]` (Yantai University)

**通讯引用:** 5271 | [OpenAlex ID](https://openalex.org/A5022510153)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种鲁棒子空间约束二次模型（SCQM），用于从高维数据中学习低维结构，并能够适应多种噪声分布。

**💡 创新点**

创新点包括：将传统子空间约束二次分解（SQMF）推广到可选的 ℓ_p^p 等非平方损失；给出完整的梯度和 KKT 条件；在 Riemannian 梯度下降框架下实现高效求解。

**🔧 技术方法**

使用了子空间约束二次矩阵分解、可微/子可微损失函数、Riemannian 梯度下降+QR 退化、backtracking line search 以及梯度推导等技术。

**📊 数据集**

实验数据包括：合成圆/球面噪声数据、MNIST 手写数字 4/9 的重构实验以及 MNIST 2/6/8 的插值实验。

**📈 对比分析**

通过与 SPH、MFIT、MLS 及线性模型的对比实验，在不同噪声水平下，SCQM 在大多数情况下取得更低的重构误差；低噪声时二次项提升；高噪声时 ℓ_p^p（p<2）表现出更强鲁棒性。

**⚠️ 局限性**

局限性包括：缺乏对统计非渐近性质和收敛理论的分析；高噪声下二次模型可能出现过拟合；仅探索了特定的损失与二次形式，未扩展至更复杂的非线性结构。

---

## 201. QwenSafe: Multimodal Content Rating Description Identification via Preference-Aligned VLMs

**arXiv ID:** 2605.20584 | [PDF](https://arxiv.org/pdf/2605.20584v1)

**作者:** Dishanika Denipitiyage `[一作]` (University of Sydney), Suranga Seneviratne `[通讯]` (University of Sydney)

**通讯引用:** 2307 | [OpenAlex ID](https://openalex.org/A5038376039)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了QwenSafe模型，能够基于应用的元数据和截图自动识别Apple定义的内容评级描述符。

**💡 创新点**

提出了descriptor-aware多模态对齐和metadata2CRD数据构造管道，利用DPO实现预测与证据的对齐。

**🔧 技术方法**

使用Qwen3-VL-8B视觉语言模型，结合监督微调和直接偏好优化（DPO）进行训练。

**📊 数据集**

构建并使用了metadata2CRD数据集，涵盖12个Apple内容评级描述符的应用元数据、截图和定义。

**📈 对比分析**

与Qwen3-VL、LLaVA-1.6和Gemini-2.5-Flash在二分类CRD识别上对比，QwenSafe在正类召回率上分别提升111.8%、36.1%和2.1%。

**⚠️ 局限性**

局限性包括对Apple定义的依赖、数据集规模有限以及对其他平台的通用性和推理效率仍待验证。

---

## 202. Mahjax: A GPU-Accelerated Mahjong Simulator for Reinforcement Learning in JAX

**arXiv ID:** 2605.20577 | [PDF](https://arxiv.org/pdf/2605.20577v1)

**作者:** Soichiro Nishimori `[一作]` (University of Tokyo), Masashi Sugiyama `[通讯]` (RIKEN AIP)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个可向量化、GPU 加速的日本麻将模拟器 Mahjax，支持大规模并行仿真与强化学习训练。

**💡 创新点**

创新点在于：①将麻将游戏逻辑改写为 JAX 函数式纯函数，完成完全向量化；②采用位掩码缓存 Yaku 计算、矩阵化控制流等优化，显著提升 GPU 计算吞吐量；③兼容 Pgx API，提供可视化工具，方便调试与交互。

**🔧 技术方法**

主要技术包括：JAX 与 JIT 编译、向量化逻辑与缓存优化、位掩码预计算、Transformer‑Encoder 代理架构、PPO 与 KL 正则化强化学习、SVG 可视化。

**📊 数据集**

使用了公开的 Tenhou 日志（红五规则）用于功能验证，训练样本则由基于规则的启发式代理产生的 500k 条经验生成；实验中也对 Libriichi Rust 模拟器和 Pgx Shogi 进行基准对比。

**📈 对比分析**

与 Libriichi（CPU）和 Pgx Shogi（GPU）比较时，Mahjax 在 8 台 NVIDIA A100 GPU 上实现了 1M~2M 步/秒的吞吐量，超过 Libriichi 10 倍以上，并超越 Pgx Shogi；强化学习实验中，基于 BC 预训练后 PPO 微调的代理在单局模式下平均排名显著优于 2.5 的均等水平。

**⚠️ 局限性**

限制包括：仅支持 no‑red 与 red 规则；强化学习实验仅限单局（单 Kyoku）模式，尚未覆盖完整回合或多局游戏；目前的 RL 训练依赖 BC 预训练，尚未实现从零开始的纯自我对弈学习。

---

## 203. LLM Pretraining Shapes a Generalizable Manifold: Insights into Cross-Modal Transfer to Time Series

**arXiv ID:** 2605.20449 | [PDF](https://arxiv.org/pdf/2605.20449v1)

**作者:** Alexis Roger `[一作]` (McGill University), Irina Rish `[通讯]` (Université de Montréal)

**通讯引用:** 7781 | [OpenAlex ID](https://openalex.org/A5055430458)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究语言预训练 Transformer 在时间序列预测中的迁移机制，并通过实验验证其几何结构作用。

**💡 创新点**

提出“共享流形假设”，证明预训练模型已内置可映射为时间序列的方向，迁移主要是低秩对齐而非重学。

**🔧 技术方法**

采用 Qwen3-0.6B、next-token 预测、线性探测、LoRA、梯度一致性分析、有效秩、PCA 等技术。

**📊 数据集**

使用 GiftEval 语料库及 WikiText-103 进行验证，时间序列长度 512、预测 horizon 64。

**📈 对比分析**

对比预训练 vs 随机初始化、全微调 vs LoRA 等；预训练模型收敛更快、CRPS/MASE 低约 10‑20% 以上，LoRA 达到与全微调相同性能。

**⚠️ 局限性**

仅在单一 Qwen3-0.6B、单一 tokenization 及 1024-bin 方案下验证，未检验更大规模或不同架构；线性探测未证明最佳预测子空间。

---

## 204. Modeling Emotional Dynamics in Agent-to-Agent Interactions on Moltbook

**arXiv ID:** 2605.20442 | [PDF](https://arxiv.org/pdf/2605.20442v1)

**作者:** Syed Mhamudul Hasan `[一作]` (Southern Illinois University), Abdur R. Shahid `[通讯]` (Southern Illinois University)

**通讯引用:** 310 | [OpenAlex ID](https://openalex.org/A5035192689)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过构建Persona‑Stimulus‑Reaction（PSR）框架，利用VAD空间将代理生成的文本情感映射为连续向量，并采用高斯混合模型（GMM）对代理的身份、帖子与评论情感进行分布建模，从而量化并分类代理在Moltbook平台上的情感行为。

**💡 创新点**

创新点在于将传统的情感标签映射到VAD三维连续空间，并将其分解为身份、刺激与反应三部分，形成PSR结构；通过GMM捕捉情感分布的多模态特征，能够捕捉代理在交互中的情感演化与不确定性；同时提出基于三种情感中心距离的行为类型分类方法。

**🔧 技术方法**

使用技术包括：Google Deep Translate进行多语言转英文；Hugging Face的SamLowe/roberta-base-go_emotions情感分类器；NRC VAD词典将情感标签映射到VAD坐标；高斯混合模型对VAD向量进行聚类与概率建模；欧氏距离与阈值τ用于判断PSR各子集之间的相似度；基于阈值的行为类型判定算法。

**📊 数据集**

使用数据集为Moltbook‑crawl，收集自2026年1月27日至2月8日的公开API数据，包含约124,165个代理、759,997条帖子、3,079,480条评论以及17,332个子社区；在此基础上筛选出21,972个拥有完整身份、帖子与评论信息的代理用于PSR分析。

**📈 对比分析**

通过对完整PSR实例计算d_PR、d_PS、d_SR并按阈值划分为5类行为，得到不同类比例（如39%为刺激驱动、未知类占比高等）。与仅使用单一VAD标签的情感描述相比，PSR+GMM能够更细致地揭示代理情感的多模态与动态特征，表现为更丰富的行为类型分布与对情感来源的可解释性。

**⚠️ 局限性**

主要局限包括：①大量代理缺失身份、帖子或评论信息，导致未知类比例高；②数据收集时间窗口短，缺乏长期交互演化；③仅使用文本情感提取，翻译与模型误差可能导致情感信息损失；④未考虑图像、视频等多模态情感信号；⑤API限制（每帖最多返回100条评论）限制了交互深度。

---

## 205. HRM-Text: Efficient Pretraining Beyond Scaling

**arXiv ID:** 2605.20613 | [PDF](https://arxiv.org/pdf/2605.20613v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 206. A 10,000-Year Global Stochastic Tropical Cyclone Catalog with Wind-Dependent Track Transitions (WHITS)

**arXiv ID:** 2605.20494 | [PDF](https://arxiv.org/pdf/2605.20494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 207. Refusal Evaluation in Coding LLMs and Code Agents: A Systematic Review of Thirteen Malicious-Code Prompt Corpora (2023-2025)

**arXiv ID:** 2605.20351 | [PDF](https://arxiv.org/pdf/2605.20351v1)

**作者:** Richard J. Young `[一作]` (University of Nevada Las Vegas), Gregory D. Moody `[通讯]` (University of Nevada Las Vegas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过PRISMA式系统综述，针对恶意代码生成拒绝评估中使用的13个公开prompt语料库，统一提取并比较了它们的构造方法、prompt构造分类、可复现性与许可情况以及恶意类别覆盖。

**💡 创新点**

创新点在于将prompt语料库本身视为研究单元，首次系统梳理并量化这些语料库在构造、评估与治理上的异质性，揭示出缺乏可靠性统计、跨语料库可比性差以及恶意类别标准碎片化等关键问题，并给出针对未来语料库建设的具体改进建议。

**🔧 技术方法**

采用PRISMA搜索策略、统一提取模板、prompt构造三轴分类法、构造方法与许可对照表以及恶意类别覆盖图等技术手段，对语料库属性进行系统化梳理与可视化。

**📊 数据集**

使用的数据集包括AdvBench、CyberSecEval系列、RMCBench、RedCode、MCGMark/MCGTest、JailbreakBench（Cyberweapons子集）、CySecBench、MalwareBench、CIRCLE、MOCHA、ASTRA、Scam2Prompt/Innoc2Scam-bench以及JAWS-Bench等13个公开恶意代码prompt语料库。

**📈 对比分析**

通过对比提取字段（构造方法、提示类型、回合结构、诱导方式、许可与可靠性等），生成四张对照表和一张恶意类别覆盖图；但由于不同语料库使用的提示构造与评估方式差异大，直接的拒绝率性能比较并不可行，本文仅指出当前统计缺乏可比性，需统一可靠性测度与标签基线。

**⚠️ 局限性**

局限性包括：未对所有语料库统一重新测量交叉评估可靠性；检索截止时间仅至2026‑05‑06，后续新语料库未纳入；缺乏人类标注基线与交叉验证；部分排除列表仅由单一作者筛选；仅基于已发布的元数据，未对实际语料库内容进行重新编码或验证。

---

## 208. Neural Estimation of Pairwise Mutual Information in Masked Discrete Sequence Models

**arXiv ID:** 2605.20187 | [PDF](https://arxiv.org/pdf/2605.20187v1)

**作者:** Jai Sharma `[一作]` (University of California, Berkeley), Bryan Li `[通讯]` (University of California, Berkeley)

**通讯引用:** 5354 | [OpenAlex ID](https://openalex.org/A5100605274)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种神经网络估计器，用于在掩码扩散模型（MDM）中预测位置间的条件互信息，从而实现并行解码并提升效率。

**💡 创新点**

通过训练轻量级头部直接从隐藏状态预测互信息矩阵，避免昂贵的联合分布真值计算，并将互信息用于指导并行采样，首次将互信息嵌入MDM的生成过程。

**🔧 技术方法**

使用了神经互信息估计（如MINE/INFO-NCE）技术、掩码扩散模型、轻量头部网络以及互信息引导的贪心并行采样算法。

**📊 数据集**

实验基于100,000个随机生成的Sudoku谜题、ESM-C蛋白语言模型以及UniRef50参考蛋白集合。

**📈 对比分析**

与顺序采样、Naive并行（k=4,7,12）和EB‑Sampler比较，MI引导下的并行采样在Sudoku上准确率提升至约63%，前向传递次数约15次；在蛋白生成上JSD降至0.136、前向传递次数约15次，显著优于Naive并行。

**⚠️ 局限性**

估计器需要大量预训练，预测精度不够完美导致生成质量略有下降；计算互信息真值成本高，且对不同任务可能需要重新设计头部与训练策略。

---

## 209. Plug-and-Play Spiking Operators: Breaking the Nonlinearity Bottleneck in Spiking Transformers

**arXiv ID:** 2605.20289 | [PDF](https://arxiv.org/pdf/2605.20289v1)

**作者:** Xinzhe Yuan `[一作]` (Harbin Institute of Technology), Huan Xiong `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 7867 | [OpenAlex ID](https://openalex.org/A5059443966)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可插拔的框架，利用 LIF 神经元群与位移‑查找表操作实现 Transformer 关键非线性（Softmax、SiLU、RMSNorm），从而实现无训练的 ANN‑to‑SNN 转换。

**💡 创新点**

将非线性运算拆解为分子分母两路，用整数除法、PolarNorm（CORDIC‑Hypot）和 PWL‑Exp 近似，提供可验证的误差界限，且可直接插入现有转换管线，无需额外训练。

**🔧 技术方法**

技术包括 LIF 神经元群、整数除法实现、PolarNorm（平方根与归一化）、PWL‑指数近似、查找表与位移缩放，以及现有 ANN‑to‑SNN 方案（SpikeZIP、SpikeLLM）。

**📊 数据集**

使用的大规模语言模型与基准包括 LLaMA‑2/3、Mistral、Qwen3 以及 BERT 等，评估任务涵盖 WinoGrande、HellaSwag、ARC、PIQA 等。

**📈 对比分析**

与 Sorbet、XNOR‑Net、DoReFa‑Net、Padé 等基准对比；在算子层面误差均低于 8‑bit 量化基线；在模型层面替换后平均准确率变动 <1%，证明方法有效且性能稳健。

**⚠️ 局限性**

尚未完成在真实神经形态硬件上的端到端部署，受限于算子实现、存储开销和时延；对极大模型的完整量化与能耗评估仍需进一步研究。

---

## 210. Score-Based Causal Discovery of Latent Variable Causal Models

**arXiv ID:** 2605.20396 | [PDF](https://arxiv.org/pdf/2605.20396v1)

**作者:** Ignavier Ng `[一作]` (Carnegie Mellon University), Kun Zhang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 21773 | [OpenAlex ID](https://openalex.org/A5100342355)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于得分的因果结构学习方法（SALAD），能够在存在潜在变量的线性模型中识别潜在变量之间的因果关系；

**💡 创新点**

创新点在于：①构造了可保证得分等价和一致性的得分函数；②对潜在变量模型的自由度进行系统化表征；③提供了既精确又可连续化的搜索算法，形成了对多种约束式方法的统一视角；

**🔧 技术方法**

采用了高斯线性潜在变量模型、似然函数与BIC评分、Gumbel‑Softmax连续优化、增强拉格朗日法等技术；

**📊 数据集**

使用了人工生成的1‑factor和层级结构数据，随机设定系数与噪声，样本量从100到10000不等；

**📈 对比分析**

与FOFC、HUANG、GIN等基于约束式的基线方法比较；在小样本（如100样本）时，SALAD在骨架F1和MEC SHD上显著优于基线（F1≈0.99/0.92 vs 0.90/0.57），且性能随样本增大趋近1；

**⚠️ 局限性**

主要限制：精确搜索计算量大，运行时间长；BIC在潜在变量设置下理论正当性尚未完全阐明；缺乏大规模真实数据验证和更高效的贪心搜索策略。

---

## 211. Direct Translation between Sign Languages

**arXiv ID:** 2605.20588 | [PDF](https://arxiv.org/pdf/2605.20588v1)

**作者:** Zetian Wu `[一作]` (Oregon State University), Liang Huang `[通讯]` (Oregon State University)

**通讯引用:** 3079 | [OpenAlex ID](https://openalex.org/A5075538786)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过反向翻译（先把目标语言文本翻译成源语言文本，再用手语生成模型生成源手语）构造大规模合成的手语-手语平行语料，并训练一个单一的多任务Encoder-Decoder模型，实现直接的手语到手语翻译。

**💡 创新点**

①首次利用跨语言文本翻译+手语生成的反向翻译技术为手语-手语翻译生成大规模并且高质量的合成平行数据；②将手语-手语和手语-文本任务联合训练，避免传统Cascade模型的误差累积与额外延迟。

**🔧 技术方法**

使用VQ‑VAE离散化手语动作、基于mllslt的多语言Encoder‑Decoder、TranslateGemma 4B进行文本翻译、动态时间扭曲（DTW）MPJPE与BLEU评估、以及自监督的反向翻译数据增强技术。

**📊 数据集**

3个单语手语-文本对齐语料（ASL‑English、CSL‑Chinese、DGS‑German）作为训练来源；跨语手语配对数据（来自先前工作）用于严格评测。

**📈 对比分析**

与Cascade（手语→文本→MT→手语）和前人直接系统对比；实验表明直接模型在MPJPE上降低约19%，在BLEU4上提升至50%以上，速度约快2.3×，在严格评测集上也明显优于基线。

**⚠️ 局限性**

缺乏真实的手语-手语平行数据，评测仍依赖文本对齐和自动评估；模型为离线批处理，无法实时对话；未进行人类评估或与聋人社群的合作验证。

---

## 212. Neural Collapse by Design: Learning Class Prototypes on the Hypersphere

**arXiv ID:** 2605.20302 | [PDF](https://arxiv.org/pdf/2605.20302v1)

**作者:** Panagiotis Koromilas `[一作]` (University of Athens), Yannis Panagakis `[通讯]` (University of Athens)

**通讯引用:** 2578 | [OpenAlex ID](https://openalex.org/A5050734738)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出在单位超球面上学习类别原型的框架，统一交叉熵与监督对比学习为原型对比，提出 NTCE 与 NONL 两种归一化对数损失，并证明 SCL 在训练过程中已得到最优的类均值分类器，从而消除了传统线性探测阶段。

**💡 创新点**

创新点在于从几何角度统一监督学习与对比学习，剔除径向自由度；通过将负样本扩展到批量中、分离正负项的正则化，使得模型能以更少的迭代收敛到 Neural Collapse；并证明无需线性探测即可获得最优分类器。

**🔧 技术方法**

采用归一化软最大（NormFace）、温度缩放、对比损失（SCL）、均值原型方法以及理论分析（Neural Collapse、有效秩、对齐、信息度量）等技术。

**📊 数据集**

实验数据集包括 CIFAR‑10、CIFAR‑100、ImageNet‑100、ImageNet‑1K 以及迁移学习/长尾任务如 VOC2007、Pets、Flowers 等。

**📈 对比分析**

与标准 CE、ETF+DR、NormFace、SCL（线性探测、归一化探测、固定原型）进行对比。NTCE/NONL 在分类准确率、NC 指标（≥95%）上均优于 CE，训练迭代次数大幅减少；固定原型在 SCL 中实现了与线性探测相同甚至更好的准确率；迁移学习提升约 5.5% 余弦平均，长尾提升高达 8.7%，鲁棒性（mCE）亦下降。

**⚠️ 局限性**

局限性包括：需要较大 batch 以获取足够负样本；理论推导基于平衡 UFM/LPM 假设，对 ReLU 激活下最优框架可能不再是中心等边 ETF；迁移至细粒度子类时可能面临区分能力不足。

---

## 213. AI-Assisted Competency Assessment from Egocentric Video in Simulation-Based Nursing Education

**arXiv ID:** 2605.20233 | [PDF](https://arxiv.org/pdf/2605.20233v1)

**作者:** Hanchen David Wang `[一作]` (Vanderbilt University), Meiyi Ma `[通讯]` (Vanderbilt University)

**通讯引用:** 1568 | [OpenAlex ID](https://openalex.org/A5101671027)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

基于第一人称护理模拟视频，提出三阶段框架：提取动作时间线、分析序列特征，并与教师评估的临床能力相关联。

**💡 创新点**

①在极低样本条件下使用冻结的DINOv2视觉‑语言模型和原型+HMM实现动作识别；②发现识别准确度与教学评分呈负相关，识别困难度可作为补充评估信号。

**🔧 技术方法**

冻结视觉编码器（DINOv2、ResNet‑50、CLIP）+原型匹配+HMM Viterbi；Spearman相关、过程挖掘等。

**📊 数据集**

22名护理学生在头戴摄像机下完成的药物管理模拟，总计3.8小时、493个动作，按16类动作+背景标注，配合C‑CEI 23项评分。

**📈 对比分析**

与ResNet‑50和CLIP对比，DINOv2在10‑shot跨样本实验中实现MOF 65.6%、mIoU 45.1%、F1 41.9%，并在5‑shot以上明显优于其它模型；识别准确度与教师评分负相关（ρ=−0.524，p=0.012）。

**⚠️ 局限性**

样本量仅22课时，难以推广；评估仅基于可视行为，无法捕捉语言交流或临床推理；单一原型设计限制了对多样化动作的识别。

---

## 214. Pramana: A Protocol-Layer Treatment of Claim Verification in Autonomous Agent Networks

**arXiv ID:** 2605.20312 | [PDF](https://arxiv.org/pdf/2605.20312v1)

**作者:** Ravi Kiran Kadaboina `[一作]` `[通讯]` (Independent Researcher), Ravi Kiran Kadaboina (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Pramana协议层原语，用于标准化自主代理输出的可审计证明，并在TLA+中对生命周期进行形式化验证。

**💡 创新点**

创新点包括：① 将四类认知根基（测量、推理、类比、引用）嵌入到Typed Attestation中；② 在A2A与MCP协议上定义统一的wire扩展；③ 通过TLA++TLC实现全面的安全不变式验证；④ 在LLM审稿者集成模式下进行实验对比。

**🔧 技术方法**

使用技术包括：Python+Pydantic v2的数据模型；TLA+ + TLC的形式化模型与验证；LLM评审者集成（Haiku、Sonnet、Opus、GPT‑4o‑mini、Llama‑3.3）；以及基于单元测试的自动化脚本。

**📊 数据集**

实验使用MBPP与HumanEval两个代码生成数据集，共计约1.3M tokens、100个问题（100个buggy+100个clean），以及在此基础上构造的LLM评审者集成配置。

**📈 对比分析**

通过在不同LLM集成配置（同模型、同家族、多家族）下计算检测率和误报率（FPR）进行比较；实验表明LLM‑as‑judge在不同语料上误报率差异显著（高达40%点），而Pramana在形式化验证和标准化方面表现稳定，未出现不变式违规。

**⚠️ 局限性**

局限性包括：实验仅在代码生成任务上，未验证真实监管领域的多样性；样本量有限且评审单人，可能导致误报率偏高；未覆盖所有监管法规细节，如主体披露与反歧视的实质性合规性。

---

## 215. ReversedQ: Opportunities for Faster Q-Learning in Episodic Online Reinforcement Learning

**arXiv ID:** 2605.20592 | [PDF](https://arxiv.org/pdf/2605.20592v1)

**作者:** Sofia R. Miskala-Dinc `[一作]`, Aviva Prins `[通讯]` (University of Maryland)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5028696505)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

改进了基于后验采样的RL算法，通过后向传播、加速值更新和更合理的初始化来提升学习速度。

**💡 创新点**

创新点在于结合后向传播、即时更新和更紧凑的初始化，减少理论上过度保守机制对实践性能的负面影响。

**🔧 技术方法**

采用基于后验采样的模型自由算法ReversedQ，并对其做了三项改造：后向更新、加速更新频率、改进初始化。

**📊 数据集**

在BDCL（Bidirectional Diabolical Combination Lock）和链式MDP（Chain MDP）两种离散有限时限MDP环境上进行实验。

**📈 对比分析**

与原ReversedQ和随机策略对比；在BDCL上性能从9.53%提升至78.78%，在链式MDP上从21.76%提升至61.81%，显著优于基线。

**⚠️ 局限性**

局限在于尚未正式证明改造后仍满足原算法的理论回报界限；实验仅覆盖两种特定环境，未检验在更复杂或非平稳环境中的泛化。

---

## 216. Governance by Design: Architecting Agentic AI for Organizational Learning and Scalable Autonomy

**arXiv ID:** 2605.20210 | [PDF](https://arxiv.org/pdf/2605.20210v1)

**作者:** Nelly Dux `[一作]` (ESSEC Business School), Abhishek Kumar Mishra `[通讯]` (Accenture Research)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一家大型 IT 服务公司的 2025 年迭代开发与分阶段部署中，系统性地研究了 agentic AI 的治理设计与实践，并提出了七条可操作的治理经验。

**💡 创新点**

创新点在于：①将治理嵌入技术架构而非后置合规层；②构建“学习成熟度模型”，阐明 memory、data 与 governance 随 AI 自主度递增的相互依赖；③通过四大治理挑战与对应设计干预，形成从“受限助手”到“agentic orchestration”演进的治理路径。

**🔧 技术方法**

技术采用了：多代理框架（agent orchestration）、工具调用与检索增强生成（RAG）、基于规则的记忆与日志存储、统一的“Nexus”入口与审批门控，整体基于企业内部 LLM 与 API 接口实现。

**📊 数据集**

使用了公司内部的知识库、客户报告、数据集市、向量索引等企业数据源，以及系统运行日志、用户反馈和内部访谈文本。

**📈 对比分析**

对比方法主要是案例对比与纵向追踪：将系统在不同阶段（受限助手 → 传统工具包装 → agentic orchestration）对治理效果、流程连贯性和风险可追溯性进行定性评估，未给出具体量化指标但指出通过治理-by-design 设计实现了更高的可追溯性、错误率下降和 ROI 预期。

**⚠️ 局限性**

局限性包括：①研究范围仅为单一企业案例，缺乏跨行业验证；②未提供大规模量化性能数据；③隐私与数据保留策略仍处于早期探索，长周期自学习导致的偏差与可审计性尚未彻底解决。

---

## 217. WaveGraphNet: Physics-Consistent Guided-Wave Damage Localization through Coupled Inverse-Forward Graph Learning

**arXiv ID:** 2605.20311 | [PDF](https://arxiv.org/pdf/2605.20311v1)

**作者:** Vinay Sharma `[一作]` (EPFL), Olga Fink `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出WaveGraphNet框架，利用图神经网络和前向一致性正则化进行复合板引导波缺陷定位。

**💡 创新点**

将缺陷定位问题视为图到坐标的回归，并通过学习的前向能量偏差模型实现坐标与测量波能分布的一致性约束。

**🔧 技术方法**

图神经网络（GAT）、注意力机制、频域特征提取、前向一致性学习以及三阶段训练策略。

**📊 数据集**

OGW-1复合板缺陷监测基准，12个边缘贴附传感器，28个单缺陷样本。

**📈 对比分析**

与1D-CNN、LSTM、GNN-MLP、GAT、单向WaveGraphNet对比，采用Split A/B空间留出测试，耦合WaveGraphNet在未见区域MAE最低且FPR为0。

**⚠️ 局限性**

仅在单一板、单缺陷、固定实验条件下验证，未评估多缺陷、环境/操作变异及实际现场部署的鲁棒性。

---

## 218. MagBridge-Battery: A Synthetic Bridge Dataset for Li-ion Magnetometry and State-of-Health Diagnostics

**arXiv ID:** 2605.20240 | [PDF](https://arxiv.org/pdf/2605.20240v1)

**作者:** Sakthi Prabhu Gunasekar `[一作]` (Amrita Vishwa Vidyapeetham), Prasanna Kumar Rangarajan `[通讯]` (Amrita Vishwa Vidyapeetham)

**通讯引用:** 854 | [OpenAlex ID](https://openalex.org/A5111727218)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

发布了MagBridge‑Battery v1.0合成磁场数据集，结合公开磁场形态与PulseBat SOH标签，为电池健康诊断提供首个可公开的磁场+SOH配对数据；

**💡 创新点**

创新点在于通过合成桥将单个磁场形态库与SOH标签相连，构建了完整的磁场–SOH映射，并通过受控消融验证桥的非线性编码能力；

**🔧 技术方法**

采用OSF磁场形态银行、MagBridge‑Embed潜在表示、线性判别分析、k‑NN软解码、噪声模型以及深度学习与传统机器学习基线进行数据生成与评估；

**📊 数据集**

使用了Mohammadi‑Jerschow OSF磁场扫描数据与PulseBat LFP退役电池的SOH/SOC/U特征；生成了5,600个基于PulseBat的grounded样本、600个合成异常样本和560个Regime‑B低压样本；

**📈 对比分析**

在无泄漏的主分割上分别评估SOH回归、二次寿命分类、异常检测和异常子类分类四个任务，基线Ridge回归R²≈0.675、RidgeCls BA≈0.907、RF BA≈0.789；消融实验表明SOH标签被正确编码，A2消融导致R²降至≈0；

**⚠️ 局限性**

局限性包括单化学单电芯、缺乏真实磁场+SOH配对验证、检索‑融合解码对方向不敏感、通道相关性误差及时间轴固定等问题。

---

## 219. Understanding Model Behavior in Monocular Polyp Sizing

**arXiv ID:** 2605.20461 | [PDF](https://arxiv.org/pdf/2605.20461v1)

**作者:** Xinqi Xiong `[一作]` (University of North Carolina at Chapel Hill), Roni Sengupta `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 37 | [OpenAlex ID](https://openalex.org/A5102635961)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对单目结肠镜息肉大小分类进行诊断审计，评估模型在多中心数据、不同输入模态和患者分层交叉验证下的表现，探讨尺度歧义和分割误差对性能的影响。

**💡 创新点**

发现模型性能受限于尺度歧义与分割质量两大独立瓶颈，并提出 oracle 尺度阶梯、快捷分区与 mask 替换等可复用的评估工具，帮助定位并量化提升空间。

**🔧 技术方法**

使用 RGB、相对深度、光照特征等多模态输入；模型包括 ResNet18、CNN3（基于深度）、MLP、ViT-B；深度估计采用 MetricCol；分割模型使用 PolypPVT；还使用了尺度校正和遮罩扰动等实验手段。

**📊 数据集**

采用 Real-Colon 与 SUN-SEG 两个公开多中心数据集，共计 159 名患者、232 个息肉，框架下采用患者分层 5‑折交叉验证。

**📈 对比分析**

在 Macro‑F1 约 0.75 的性能基准上，oracle 尺度提升 16.1 百分点，分割误差导致 15.3 百分点下降；整体性能与不同模型和输入模态差异不大，显示两大瓶颈主导结果。

**⚠️ 局限性**

主要限制包括：缺乏真实尺度标注导致尺度歧义无法完全消除；分割模型在域移时性能显著下降；样本规模有限，易形成基于检查行为的捷径。

---

## 220. Tiny-Engram: Trigger-Indexed Concept Tables for Generative Vision

**arXiv ID:** 2605.20309 | [PDF](https://arxiv.org/pdf/2605.20309v1)

**作者:** Runyuan Cai `[一作]` (AutoArk-AI), Xiaodong Zeng `[通讯]` (AutoArk-AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于触发词索引的概念表（Engram）实现视觉个性化，能够在冻结的图像/视频生成模型中仅在出现注册触发词时激活对应视觉记忆并修改文本编码器的隐藏状态，保持其余提示不受影响。

**💡 创新点**

创新点在于：①通过精确的n‑gram注册实现局部激活边界，避免传统全局适配器导致的覆盖和交互问题；②在多编码器、多模态扩散模型中引入相对范数注入规则，使得不同文本流的注入力度可统一调节；③将该机制从语言模型迁移到视觉生成，展示其在单/多编码器文本条件下的有效性。

**🔧 技术方法**

主要技术包括：PEFT（仅训练记忆向量与缩放参数）+ Engram模块；n‑gram注册与精确匹配；相对范数注入；在SD1.5、SD3.5和Wan2.2的冻结扩散/视频生成器中插入注入点。

**📊 数据集**

使用了五张来自《死亡搁浅》主角的静态图像以及由GPT-5.3逆向提示生成的对应文本作为训练对；视频实验采用利用Wan2.2 TI2V-5B生成的五段短视频（每段81帧）作为教师生成的训练样本。

**📈 对比分析**

与冻结基线模型在匹配提示与种子下进行定性对比；未注册触发词时输出保持一致，验证局部激活；在SD3.5中可显著更换主体身份并保留场景、光照等细节；在Wan2.2视频实验中触发词能改变主体形象，但对细粒度身份与时序一致性的影响有限。

**⚠️ 局限性**

局限性：①实验规模小，仅用5张图/5段视频；②缺乏与DreamBooth、Textual Inversion、LoRA等传统个性化方法的定量基准；③视频实验受教师生成质量限制，无法保证细粒度身份和时序一致；④在多编码器环境下注入点与比例规则的选择仍需更系统验证。

---

## 221. TreeText-CTS: Compact, Source-Traceable Tree-Path Evidence for Irregular Clinical Time-Series Prediction

**arXiv ID:** 2605.20292 | [PDF](https://arxiv.org/pdf/2605.20292v1)

**作者:** Kwanhyung Lee `[一作]` (Kim Jaechul Graduate School of AI, KAIST), Eunho Yang `[通讯]` (Kim Jaechul Graduate School of AI, KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

将不规则的电子病历时间序列转换为可读、可追溯的树路径证据，并用语言模型对其进行预测；

**💡 创新点**

创新在于通过冻结的XGBoost树生成可追溯的阈值条件文本，随后利用自适应选择器挑选紧凑证据，并将这些可读文本输入语言模型，兼顾可解释性与性能；

**🔧 技术方法**

采用多尺度窗口摘要、XGBoost模型、Tree‑to‑Evidence映射、Compact Evidence Selector以及基于BioClinical ModernBERT的语言模型分类器；

**📊 数据集**

在PhysioNet 2012、MIMIC‑III和PhysioNet 2019三个ICU预测基准上进行评估；

**📈 对比分析**

与传统文本接口（WRDP、Record2Vec、Decode Like a Clinician、TimeCP）以及数值时序模型（GRU‑D、STraTS、DuETT等）进行对比，取得了所有文本接口中最高的AUROC和AUPRC，并接近数值模型的性能；

**⚠️ 局限性**

局限性包括缺乏临床正确性/因果性/公平性验证、对树模型和摘要的依赖、以及未在临床部署场景下验证。

---

## 222. Regulating Anatomy-Aware Rewards via Trajectory-Integral Feedback for Volumetric Computed Tomography Analysis

**arXiv ID:** 2605.20277 | [PDF](https://arxiv.org/pdf/2605.20277v1)

**作者:** Tianwei Lin `[一作]` (Zhejiang University), Ling Zhang `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了临床异常基准（CABS）和基于轨迹积分反馈的强化学习框架TIF‑GRPO，以提升3D CT报告生成的临床真实性。

**💡 创新点**

创新点在于将报告拆解为可验证的异常单元，并在奖励设计中引入积分反馈机制，解决评估幻觉与机制偏差问题。

**🔧 技术方法**

采用控制理论的积分反馈、Group Relative Policy Optimization、结构化CABS奖励，以及深度VLM（Qwen3‑VL‑4B）等技术。

**📊 数据集**

在CT‑RATE、AMOS‑MM等四个3D CT基准以及MIMIC‑CXR‑Report跨模态数据集上进行评测。

**📈 对比分析**

与多种通用与医学专用VLM以及基于表面相似度奖励的RL方法对比，TIF‑GRPO在实体准确率、临床可信度和器官覆盖率等指标上均实现SOTA，显著优于基线。

**⚠️ 局限性**

局限性包括对可验证单元定义范围的依赖，未能覆盖所有极端罕见异常，并且积分反馈机制在推理时增加了一定的计算开销。

---

## 223. NaP-Control: Navigating Diffusion Prior for Versatile and Fast Character Control

**arXiv ID:** 2605.20209 | [PDF](https://arxiv.org/pdf/2605.20209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 224. A Meshtastic-based LoRa Mesh System for Smart Campus Applications: From Solar-Powered Sensing to Containerized Data Management

**arXiv ID:** 2605.20379 | [PDF](https://arxiv.org/pdf/2605.20379v1)

**作者:** Rafael Garzon Andosilla `[一作]` (Universidad Militar Nueva Granada), José de Jesús Rugeles Uribe `[通讯]` (Universidad Militar Nueva Granada)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计、实现并评估了基于Meshtastic协议的LoRa网状网络，集成了太阳能感知节点、移动跟踪器和容器化的边缘数据管道，应用于大学校园的智慧校园项目。

**💡 创新点**

创新点在于将低功耗太阳能节点与Meshtastic的去中心化网状路由相结合，并通过Docker Compose实现可复现的端到端数据流，消除了对商业LoRaWAN运营商的依赖，提供了自给自足的智慧校园平台。

**🔧 技术方法**

采用Raspberry Pi Pico+Semtech SX1262的自制太阳能节点、Seeed SenseCAP T1000-E移动跟踪器，Meshtastic固件的Managed Flooding路由，915 MHz ISM频段、SF11/125 kHz调制；在边缘使用Docker Compose部署Mosquitto MQTT、Node-RED、InfluxDB与Grafana，实现数据采集、ETL、存储与可视化。

**📊 数据集**

使用现场采集的太阳辐射时间序列、GNSS定位轨迹以及测得的RSSI和SNR链路质量指标，构成实验数据集。

**📈 对比分析**

通过在校园内不同传播区测量RSSI/SNR，并在2.47 km的远程节点记录62个有效包（平均RSSI ≈ -110 dBm，平均SNR ≈ +2.75 dB），证明网状链路在远距离下可实现可靠通信，整体性能优于传统LoRaWAN星型部署。

**⚠️ 局限性**

局限性包括仅在少数节点和单一路径上验证，缺乏大规模、多节点、长时段的吞吐率和延迟统计；未系统评估功耗、网络拥塞和节点间协调的性能。

---

## 225. Intersecting Dense Automata

**arXiv ID:** 2605.20421 | [PDF](https://arxiv.org/pdf/2605.20421v1)

**作者:** Dmitry Chistikov `[一作]` (University of Warwick), Neha Rino `[通讯]` (University of Warwick)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5088843055)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了新的非确定性有限自动机(NFA)产品构造，显著降低了在计算NFA交集空性时的转移数，从传统的Θ(m^k)降到O(m n^{k-1})（k为NFA数量），并基于此给出了更快的空性判定算法；同时提出了对应的证书（短路径集与“staggered cut”）并给出了高效的验证方案；通过细粒度复杂度论证，证明该算法的上界在组合k‑Clique假设下是最优的；还讨论了该技术在模型检测与正则路径查询中的潜在应用。

**💡 创新点**

①采用“interleaving”与“0‑k restriction”思想构造稀疏NFA，避免传统笛卡尔积中的冗余转移；②提出“nodding product”“catch‑up product”“leapfrog product”三种不同实现，兼顾转移稀疏与ε‑转移的使用；③结合细粒度复杂度理论，证明除非突破组合k‑Clique或SETH，否则不存在更快的组合复杂度；④设计了可验证的交集空性证书，验证时间比直接判定更快。

**🔧 技术方法**

核心技术包括：NFA交集的平衡分解（interleaving）、0‑k限制、ε‑转移利用、矩阵乘法优化（快速矩阵乘法实现O(n^ω)的验证）、细粒度复杂度的下界证明与构造；实现上使用邻接矩阵与稀疏矩阵乘法，构造和验证时的时间复杂度分别为O(m n^{k-1})与O(kℓ n^k+ω-2)。

**📊 数据集**

本工作主要是理论性研究，没有使用实际数据集；所有结果均基于抽象的NFA结构和图模型（如k‑Clique图）。

**📈 对比分析**

与传统的直接笛卡尔积方法（时间O(m^k)）相比，新的构造在转移数和运行时间上取得显著改进：对固定k和Σ，时间降低到O(m n^{k-1})，对于k=2时可达到O(m n)；在最坏情况下（密集NFA）甚至比直接构造快数倍；此外，证书验证时间比决策算法更快，尤其在稠密图情形下。

**⚠️ 局限性**

限制与挑战：①对ε‑转移的依赖在某些应用中可能不被接受；②跳跃式（leapfrog）与追赶式（catch‑up）构造在字母表较大时状态数仍可能膨胀；③当前最优性证明仅在组合k‑Clique假设下成立，若该假设被突破则可能存在更快算法；④在实际模型检测工具中整合这些稀疏构造仍需进一步实验验证。

---

## 226. Time-To-Reach Separation and Safety Filtering for Safe, Fair, and Efficient Multi-Agent Coordination

**arXiv ID:** 2605.20625 | [PDF](https://arxiv.org/pdf/2605.20625v1)

**作者:** Matthew Low `[一作]` (University of California Berkeley), Jason J. Choi `[通讯]` (University of California Los Angeles)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于最小到达时间（TTR）的多智能体协调框架，用于在拥挤的航空走廊入口实现无人机的安全并入。

**💡 创新点**

创新点在于将TTR统一作为优先级分配、时间间隔规划与安全过滤的度量，且通过优先级一致的控制边界函数（CBF）安全过滤实现先来后到的公平性与安全性协同。

**🔧 技术方法**

使用技术包括Hamilton–Jacobi到达性分析（求解最小BRT/BRAT）、TTR值函数的离线计算、基于TTR的分隔引导控制、以及基于CBF的最小范数安全过滤器（分布式、优先级协同和集中式三种实现）。

**📊 数据集**

使用仿真数据集：在二维平面上随机生成N=5或N=8架无人机的初始状态，动力学参数取自行业规范，加入不同风速和方向的扰动，以评估算法性能。

**📈 对比分析**

与传统贪婪TTR最小化加安全过滤（Dec、Prio、Cent）以及无安全过滤的基线进行比较。结果显示：TTR分隔加安全过滤在安全率（ν_unsafe）、到达时间（T^last）、优先级保持率（τ_K）和成功率（P_pass）上均优于基线；在100个随机场景中，TTR_sep+Cent实现最高的成功率和最低的安全违规率。

**⚠️ 局限性**

限制包括：离线HJ求解在高维系统下计算成本高、仿真仅限二维平面（未考虑高度和三维障碍）、未对更大规模或动态交通的可扩展性进行理论证明，且对复杂天气和多模动力学的鲁棒性尚待进一步验证。

---

## 227. STELLAR: Scaling 3D Perception Large Models for Autonomous Driving

**arXiv ID:** 2605.20390 | [PDF](https://arxiv.org/pdf/2605.20390v1)

**作者:** Yingwei Li `[一作]` (Waymo), Mingxing Tan `[通讯]` (Waymo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了在自动驾驶感知中通过大规模预训练和多模态融合实现模型参数和数据量扩展的可行性，并提出 STELLAR 框架。

**💡 创新点**

首次系统性验证 3D 感知模型在 50M 训练样本、500M 参数规模下的可扩展性，提出多模态、跨任务学习与大规模预训练的整体训练范式。

**🔧 技术方法**

采用 Sparse Window Transformer、BEV 投影、LiDAR/相机/雷达/地图特征融合、CenterNet 风格检测头、占据预测与道路图预测、多任务学习、梯度检查点、LAMB 优化及教师蒸馏等技术。

**📊 数据集**

主要使用自研的 50M 自动驾驶日志数据集（包含 LiDAR、相机、雷达、地图特征），并在公开的 Waymo Open Dataset 基准上进行验证。

**📈 对比分析**

通过 Waymo Open Dataset 验证集的 L2 APH 与现有单/多帧方法对比，STELLAR 在所有类别上显著提升，尤其是骑行者检测；在测试集上刷新单帧/多帧不使用未来帧的 SOTA。

**⚠️ 局限性**

可扩展性收益呈递减；模型规模受限于计算/内存；数据质量受限于自动标注；缺乏实时推理效率评估；仍需验证在更大范围或不同传感器配置下的鲁棒性。

---

## 228. Retrieval-Augmented Long-Context Translation for Cultural Image Captioning: Gators submission for AmericasNLP 2026 shared task

**arXiv ID:** 2605.20626 | [PDF](https://arxiv.org/pdf/2605.20626v1)

**作者:** Aashish Dhawan `[一作]` (University of Florida), Christan Grant `[通讯]` (University of Florida)

**通讯引用:** 337 | [OpenAlex ID](https://openalex.org/A5070860161)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出两阶段管线：先用视觉语言模型生成西班牙语中间字幕，再通过检索增强的多示例 Gemini 2.5 Flash 翻译成目标土著语言字幕。

**💡 创新点**

创新点在于将检索增强的上下文学习与文化字幕生成结合，利用西班牙语-目标语言检索对对齐风格，并通过合成并行数据和形态学提示进一步提升质量。

**🔧 技术方法**

使用的技术包括 Qwen2.5‑VL/Qwen3‑VL 视觉语言模型、Gemini 2.5 Flash（多示例 LLM 翻译）、GPT‑4 系列（对比实验）、BM25 检索、mBART 传统翻译、字符级评估 chrF++ 以及 NFD 正则化。

**📊 数据集**

使用的数据集包括 AmericasNLP 2026 任务图像与字幕、AmericasNLP 2023 并行训练集、MultiScript30k 生成的合成并行对、开发集作为检索示例及测试集。

**📈 对比分析**

通过与官方基线（Qwen3VL+M2M‑100/mBART）以及单独的 mBART 翻译基线对比，进行 grid 搜索验证不同检索/示例设置；在 dev 集上实现 Guaraní 48.24 chrF++（相对基线提升 131.7%）、Bribri 42.90 chrF++（提升 164.1%）、Orizaba Nahuatl 25.67 chrF++（提升 122.6%），测试集亦保持 >150% 的提升。

**⚠️ 局限性**

主要局限在于级联架构导致西班牙语字幕错误无法回滚、开发集示例可能导致 dev 集得分偏高、评估仅基于 chrF++，缺乏对流利度、文化适宜性等更细粒度的评价。

---

## 229. MAPS: A Synthetic Dataset for Probing Vision Models in a Controlled 3D Scene Space

**arXiv ID:** 2605.20549 | [PDF](https://arxiv.org/pdf/2605.20549v1)

**作者:** Santiago Galella `[一作]` (FIAS & Institute of Computer Science), Matthias Kaschube `[通讯]` (FIAS & Institute of Computer Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MAPS 数据集（2,618 个可控 3D 网格，覆盖 560 个 ImageNet 类）并提供基于 Blender 的可变参数渲染管线，用它评估 20 个预训练模型的对场景因子敏感性。

**💡 创新点**

在规模、类覆盖、可调参数和可扩展渲染上实现突破，首次提供统一的 ImageNet 对齐 3D 基准和回归敏感性分析框架。

**🔧 技术方法**

利用 Blender 渲染、CMA‑ES 验证网格可识别性、Latin hypercube 采样、线性与多项式回归、层次聚类等技术。

**📊 数据集**

主要使用自建的 MAPS 数据集，配合 ImageNet 选取的 100 个类别和 TorchVision 预训练模型。

**📈 对比分析**

通过对每个模型的判别边缘进行回归，发现摄像机距离和仰角是所有模型的主失效轴，且模型间敏感性谱聚类揭示现代 CNN 与 Transformer 归属同一组，说明准确率并不能完全反映对场景因子的依赖。

**⚠️ 局限性**

局限在于类覆盖偏向人造物体、对动物细粒度缺失；仅调节相机、灯光、背景，无法改变遮挡、纹理等；渲染管线非可微，尚未实现更高维度的场景因子。

---

## 230. Latent Process Generator Matching

**arXiv ID:** 2605.20547 | [PDF](https://arxiv.org/pdf/2605.20547v1)

**作者:** Lukas Billera `[一作]` (Karolinska Institutet), Ben Murrell `[通讯]` (Karolinska Institutet)

**通讯引用:** 18527 | [OpenAlex ID](https://openalex.org/A5033401033)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种新的推前（pushforward）生成器匹配框架，允许在训练时使用时间依赖的潜在随机过程，并在生成时仅保留可观测状态的边缘分布。

**💡 创新点**

创新点在于：①将原先仅针对静态潜在变量的生成器匹配推广到潜在过程；②给出通用的可推前理论与梯度一致性证明；③允许非投影映射，支持链级刚体运动与内部柔性相分离的蛋白质结构生成。

**🔧 技术方法**

核心技术包括：时间不齐性费勒过程的生成器参数化、线性参数化、Bregman散度的条件/边缘损失、拉普拉斯/马尔可夫链推前公式以及对连续与离散潜在空间的统一处理。

**📊 数据集**

论文主要基于理论推导，示例使用了：①离散状态空间的编辑流（Edit Flows）与分支流（Branching Flows）的案例；②连续状态空间的例子（如带有状态切换的扩散过程）以及蛋白质链级结构的潜在空间示例；没有在公开数据集上进行大规模实验。

**📈 对比分析**

比较方法主要是与已有的投影潜在过程生成器匹配理论（如流匹配、扩散模型、CTMC的投影结果）进行对比，证明其作为特殊情况能恢复这些结果。性能方面，示例展示了边缘分布与目标分布对齐，生成路径平滑且隐含的潜在过程被有效集成。

**⚠️ 局限性**

局限性包括：①对基过程Y_t与映射Φ的正则性假设较强，实际应用中可能违反；②推前的KFE唯一性与可行性需满足较多技术条件；③在大规模复杂潜在空间（如高维连续或流形）下的数值实现与计算成本尚未充分验证；④目前仅提供理论与小规模示例，缺乏广泛的实验评估。

---

## 231. Pareto-Enhanced Portrait Generation: Vision-Aligned Text Supervision for Alignment, Realism, and Aesthetics

**arXiv ID:** 2605.20640 | [PDF](https://arxiv.org/pdf/2605.20640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 232. Detecting Data Exfiltration through I2P Anonymity Networks: A Two-Phase Machine Learning Approach

**arXiv ID:** 2605.20546 | [PDF](https://arxiv.org/pdf/2605.20546v1)

**作者:** Siddique Abubakr Muntaka `[一作]` (University of Cincinnati), Pulcheria Serwaa `[通讯]` (Kwame Nkrumah University of Science & Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个双阶段机器学习框架，先用随机森林检测I2P匿名网络流量，再用XGBoost对已识别流进行行为分类，区分合法隐私使用与数据外泄。

**💡 创新点**

首次将检测与行为评估结合起来，实现对I2P流量的威胁等级划分，并通过特征重要性解释模型决策，为安全运维提供可解释的优先级信息。

**🔧 技术方法**

采用树基集成方法（随机森林、XGBoost、LightGBM）并与SVM、深度神经网络进行对比，使用流级特征提取、标准化与类别权重处理。

**📊 数据集**

使用SafeSurf Darknet 2025数据集，共184,548条网络流，I2P占12.4%，并标注了FTP、P2P、浏览等行为标签。

**📈 对比分析**

通过80/20分层划分测试比较五种算法，Phase 1随机森林取得99.96%准确率、0.006%误报率；Phase 2 XGBoost取得91.11%准确率、92.85%召回率，训练时间从9秒到188秒不等。

**⚠️ 局限性**

局限性包括数据仅来自实验室环境，行为分类仅涵盖三类，未覆盖邮件、聊天等其他I2P应用；仅基于元数据，可能易受对抗攻击；模型对其他匿名网络（如Tor）缺乏泛化能力。

---

## 233. The Yes-Man Syndrome: Benchmarking Abstention in Embodied Robotic Agents

**arXiv ID:** 2605.20544 | [PDF](https://arxiv.org/pdf/2605.20544v1)

**作者:** Doguhan Yeke `[一作]` (Purdue University), Z Berkay Celik `[通讯]` (Purdue University)

**通讯引用:** 755 | [OpenAlex ID](https://openalex.org/A5005376753)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个针对具身机器人视觉-语言模型（VLM）的“拒绝执行”基准数据集，并对多种前沿VLM在此基准上的拒绝行为进行评估与改进。

**💡 创新点**

提出了八类拒绝类别的系统化分类法、基于结构化视觉推理的可扩展生成管线，以及使用防御性提示与上下文学习提升拒绝率的方法。

**🔧 技术方法**

采用三阶段管线：(1) 视觉-语言模型做结构化视觉表征；(2) 规则推导生成拒绝候选；(3) 模板化指令生成；同时使用 LLM 作为判别者评估模型输出。

**📊 数据集**

从五个真实世界机器人数据集（DROID、Robo2VLM、RoboVQA、BridgeV2、EgoThink）各抽取 250 张图像，共 1,250 张图像和 6,069 条指令。

**📈 对比分析**

通过对比 GPT‑5.4、Claude Sonnet 4.6、Gemini 2.5 Flash、Gemini Robotics ER 1.6 Preview 等模型，发现最优模型 Gemini 2.5 Flash 的拒绝率仅为 39%，其余模型低至 16%；防御性提示+上下文学习可将拒绝率提升至 93% 以上，但仍未完全解决问题。

**⚠️ 局限性**

主要局限包括：依赖第一阶段视觉推理的准确性，且错误会传播；仅覆盖静态图像与有限类别，未考虑多轮交互和多样化机器人；使用模板生成指令缺乏语言多样性；判别器基于 LLM，可能引入噪声。

---

## 234. Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs

**arXiv ID:** 2605.20641 | [PDF](https://arxiv.org/pdf/2605.20641v1)

**作者:** Yifei Wang `[一作]` (Shanghai Jiao Tong University), Li Pan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 10481 | [OpenAlex ID](https://openalex.org/A5100455171)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM部署阶段研究编译优化导致的后门攻击，提出两种后门框架：针对特定输入的触发无后门（ISBS）和通用编译触发后门（CTB）。

**💡 创新点**

首次揭示编译优化是LLM安全的新攻击面，并设计统一的、无需修改编译器或硬件的后门方法。

**🔧 技术方法**

利用LoRA微调、激活层偏置插入、连续触发器优化、梯度聚合和对抗训练，以及对不同编译后端（Inductor、CUDAGraphs）的实验。

**📊 数据集**

在四种开源LLM（Llama-3.2-1B/3B、Qwen2.5-1.5B/3B）和四个下游任务（Agent、Embodied、Medical、SST）上进行测试。

**📈 对比分析**

与基准模型对比，ISBS在目标输入上实现约90%攻击成功率；CTB在所有模型–任务组合中平均成功率约90%，在Inductor后端可达100%，并在不影响正向推理准确率的前提下保持接近100%的纯净精度。

**⚠️ 局限性**

攻击对不同编译后端的跨迁移性有限，且对任务复杂度敏感；对防御手段的有效性仍需进一步评估。

---

## 235. The General Theory of Localization Methods

**arXiv ID:** 2605.20635 | [PDF](https://arxiv.org/pdf/2605.20635v1)

**作者:** Congwei Song `[一作]` (Beijing Institute of Mathematical Sciences and Applications), Congwei Song `[通讯]` (Beijing Institute of Mathematical Sciences and Applications)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5055854735)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种统一的定位方法框架，通过局部核与局部均值两大核心概念实现非线性机器学习。

**💡 创新点**

创新点在于将自注意力、MeanShift、Hopfield网络等模型归纳为定位方法，并通过定位技巧将简单模型转化为强大模型。

**🔧 技术方法**

主要技术包括局部核定义、局部均值回归、核密度估计、自适应核学习、离散核以及基于局部均值的自编码器和扩散模型。

**📊 数据集**

实验使用了MNIST、Fashion‑MNIST等公开图像数据集以及合成数据进行验证。

**📈 对比分析**

通过与传统核回归、MeanShift聚类、LLE降维等方法对比，定位方法在拟合精度、聚类质量和可解释性方面均表现优异。

**⚠️ 局限性**

局限性主要是核参数选择和计算开销，且在高维稀疏数据中需进一步研究核可扩展性与理论收敛性。

---

## 236. Dynamic Shapley Computation

**arXiv ID:** 2605.20620 | [PDF](https://arxiv.org/pdf/2605.20620v1)

**作者:** Xuan Yang `[一作]` (Duke University), Jian Pei `[通讯]` (Duke University)

**通讯引用:** 52742 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 D-Shap，基于玩家‑任务矩阵的动态 Shapley 价值计算框架，支持任务与玩家的增量更新；

**💡 创新点**

创新点在于将 Shapley 视为可维护的矩阵结构，利用模型诱导的局部性（任务与玩家之间的稀疏依赖）实现结构感知的增量插值与局部重计算；

**🔧 技术方法**

技术手段包括：玩家‑任务 Shapley 矩阵维护、结构感知插值、局部性推断、子集重用、覆盖感知锚点选择以及基于模型的距离度量；

**📊 数据集**

实验在五类代表性模型体系（覆盖不同任务与局部性机制）上进行，使用常见的标准数据集组合（如CIFAR‑10、ImageNet等），但论文未在摘要中具体列明数据集；

**📈 对比分析**

与传统静态 Shapley 计算、单向增量方法以及基于解释器的摊销方法对比，D-Shap 的任务增量更新在毫秒级完成，玩家增量更新成本下降三位数，且在所有评估指标上与完整重计算的结果质量相当；

**⚠️ 局限性**

局限性包括：对模型诱导局部性的强假设依赖，若任务/玩家关系高度非局部则性能可能下降；需先估计新任务与已有任务的相似度；矩阵规模仍随玩家/任务数量增长，存储与维护成本随之上升；此外，论文未详细讨论删除或替换操作的实现细节。

---

## 237. Unsupervised clustering and classification of upper limb EMG signals during functional movements: a data-driven

**arXiv ID:** 2605.20599 | [PDF](https://arxiv.org/pdf/2605.20599v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 238. From Automated to Autonomous: Hierarchical Agent-native Network Architecture (HANA)

**arXiv ID:** 2605.20608 | [PDF](https://arxiv.org/pdf/2605.20608v1)

**作者:** Binghan Wu `[一作]` (AsiaInfo Technologies Limited), Ye Ouyang `[通讯]` (AsiaInfo Technologies Limited)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了层级化自主网络架构 HANA，实现从静态自动化向基于智能体的自主运维转型。

**💡 创新点**

创新点在于双驱动 Orchestrator 的“慢思考”战略规划与“快思考”快速响应耦合，并引入自我意识实现战略与操作自适应。

**🔧 技术方法**

采用多智能体协作、公共记忆、A2A 协议、模型上下文协议以及智能工具箱，将认知意图映射到 5G 核心网络的操作命令。

**📊 数据集**

使用 5G 核心网络真实监控数据、网络性能指标和历史故障案例作为训练与验证数据集。

**📈 对比分析**

通过两组案例验证：VIP 终端服务保障与核心网络自愈，结果显示在拥塞场景下吞吐量保持高于基线，MTTR 降低约 86%。

**⚠️ 局限性**

局限性包括 LLM 推理延迟、跨域协同标准化不足，以及对极端并发策略目标的处理尚需优化。

---

## 239. Mechanistic Interpretability for Learning Assurance of a Vision-Based Landing System

**arXiv ID:** 2605.20607 | [PDF](https://arxiv.org/pdf/2605.20607v1)

**作者:** Romeo Valentin `[一作]` (Stanford University), Mykel J. Kochenderfer `[通讯]` (Stanford University)

**通讯引用:** 12723 | [OpenAlex ID](https://openalex.org/A5068326377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于机制解释的视觉降落系统学习保证方法，并在内部情况表示层实现了内容与风格分离与运行时OOMS监测

**💡 创新点**

创新点在于利用K‑SVD稀疏字典将ViT的patch嵌入拆解为可解释的contentful与stylistic原子，并通过这些原子构建OOMS检测器，满足EASA学习保证对内部表示的监控要求

**🔧 技术方法**

使用Vision Transformer+Soft‑argmax回归、K‑SVD稀疏字典学习、内容/风格基于跨域方差判别、对OOMS的约束式L1正则化逻辑回归

**📊 数据集**

在LARDv2数据集上进行训练与评估，该数据集包含四个不同视觉域（xplane, ges, arcgis, bingmaps）和模拟的BOGO/逆BOGO裁剪

**📈 对比分析**

与从零训练的ViT基准相比，预训练模型在关键点回归的平均像素误差上提升约1.3px（预训练1.6±0.2 vs. 从零1.7±0.1），并在OOMS检测中以约0.96的AUROC（使用仅34个原子）实现高准确率

**⚠️ 局限性**

局限性包括内容/风格划分依赖于仅四个视觉域，CV阈值为简单中位数且可能误判；仅针对单一任务和架构，未完成对原子覆盖度与稳定性的完整验证

---

## 240. Head-Aware Key-Value Compression for Efficient Autoregressive Image Generation

**arXiv ID:** 2605.20600 | [PDF](https://arxiv.org/pdf/2605.20600v1)

**作者:** Guotao Liang `[一作]` (Harbin Institute of Technology), Yunming Ye `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 9844 | [OpenAlex ID](https://openalex.org/A5002523892)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了HeadKV，一种训练无关的头感知KV缓存压缩框架，用于加速自回归图像生成。

**💡 创新点**

创新点在于：①识别注意力头为局部头和全局头并在生成初期进行分类；②为局部头采用滑动窗口压缩、为全局头采用分层Token驱逐（STE）来保留长程信息；③无需额外训练或统计数据，能适配不同模型和输入。

**🔧 技术方法**

技术包括：注意力累计阈值分类、滑动窗口KV更新、分层Token驱逐（按距离分为近远两组并分别挑选最重要Token）、周期性缓存更新和可变长度注意力实现。

**📊 数据集**

使用了多种文本到图像和类别到图像的模型作为实验平台：Janus‑Pro‑1B/7B、Lumina‑mGPT‑768/1024、LlamaGen‑XL；评估数据集包括 COCO‑30K、ImageNet、GenEval、DPG、FID‑30K/50K。

**📈 对比分析**

与全缓存、LineAR 及其他LLM KV压缩方法（Streaming, H2O, R‑KV）进行对比；结果显示HeadKV在保持近似视觉质量的前提下，显著降低KV内存（约 56% 以上）并提升吞吐量（2–4× 加速），尤其在高压缩比（ρ=1/8）下优于竞争方法。

**⚠️ 局限性**

局限性：对大型模型如 Lumina‑mGPT‑768 在极高压缩率下延迟略高；分层驱逐对分区比例 r_s 敏感，需要调参；局部头误分类率约 7–8%，虽对整体影响小，但仍可改进。

---

## 241. $Δ$ynamics: Language-Based Representation for Inferring Rigid-Body Dynamics From Videos

**arXiv ID:** 2605.20576 | [PDF](https://arxiv.org/pdf/2605.20576v1)

**作者:** Chia-Hsiang Kao `[一作]` (Cornell University), Ning Zhou `[通讯]` (Amazon)

**通讯引用:** 1358 | [OpenAlex ID](https://openalex.org/A5049381658)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用语言模型将单目视频中的刚体物理状态与属性转化为可直接输入物理引擎的结构化文本配置；

**💡 创新点**

提出用自然语言作为统一的刚体动力学表示，并通过光流输入和运动推理显著提升跨域泛化；

**🔧 技术方法**

采用 Qwen2.5‑VL 视觉语言模型，结合 RAFT 光流、运动描述生成、最佳采样和 CMA‑ES 进化搜索等技术；

**📊 数据集**

在 400K 个 MuJoCo 合成视频上训练，并在 CLEVRER（Blender）和自收集的 235 条真实视频上评估；

**📈 对比分析**

与 InternVL3、Qwen2.5‑VL‑7B、Claude‑4‑Sonnet 等 VLM 通过少样本提示对比，Segmentation IoU 最高 0.29，真实视频上 IoU 提升 12% 并实现跨引擎零样本迁移；

**⚠️ 局限性**

仅支持预定义的球、圆柱、盒子等几何原语，无法处理复杂形状或非刚体运动的情形。

---

## 242. Sketch2MinSurf: Vision-Language Guided Generation of Editable Minimal Surfaces from Hand-Drawn Sketches

**arXiv ID:** 2605.20733 | [PDF](https://arxiv.org/pdf/2605.20733v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 243. Faster or Stronger: Towards Flexible Visual Place Recognition via Weighted Aggregation and Token Pruning

**arXiv ID:** 2605.20551 | [PDF](https://arxiv.org/pdf/2605.20551v1)

**作者:** Zichao Zeng `[一作]` (University College London), Jan Boehm `[通讯]` (University College London)

**通讯引用:** 6754 | [OpenAlex ID](https://openalex.org/A5056951932)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了WeiAD（加权最优传输聚合）与WeiToP（基于自蒸馏的Token剪枝）两大模块，专门用于提升视觉场所识别的准确性与推理效率；

**💡 创新点**

①通过层级权重化的OT聚合和双向dustbin、ghost token，使聚合更具区分度；②利用聚合过程产生的Token重要性信息进行自蒸馏，训练轻量化剪枝模块，实现推理时可控的精度‑效率权衡；

**🔧 技术方法**

使用Vision Transformer（DINOv2 ViT‑B/14）作为特征提取器；基于可学习权重的最优传输（OT）聚合；层级权重与bidirectional dustbin；自蒸馏学习Token重要性；多相似度损失；以及Token剪枝技术；

**📊 数据集**

训练集：GSV‑Cities；评估集：MSLS‑val、MSLS‑challenge、Nordland、Pitts250k‑test、SPED‑test、AmsterTime等标准VPR基准；

**📈 对比分析**

与CNN基线（NetVLAD、CosPlace、MixVPR、EigenPlaces等）以及ViT基线（SuperVLAD、SALAD、CricaVPR）进行对比；在所有数据集上均实现或逼近最高Recall@1/5；在Token保留率ρ=0.5时，比MixVPR速度更快3.1%并提升Recall@1；在ρ=0.4时，比最快CNN基线快0.29 ms/图并提升5.8%Recall；

**⚠️ 局限性**

仍依赖ViT的高计算成本；Token剪枝主要针对单阶段VPR，对多阶段或段级检索的适用性尚未验证；在极低Token保留率下性能下降；对更小模型的可扩展性与轻量化设计仍有提升空间。

---

## 244. Uncertainty-Guided Conservative Propagation for Structured Inference in Vessel Segmentation

**arXiv ID:** 2605.20543 | [PDF](https://arxiv.org/pdf/2605.20543v1)

**作者:** Huan Huang `[一作]` (Kennesaw State University), Chen Zhao `[通讯]` (Kennesaw State University)

**通讯引用:** 7641 | [OpenAlex ID](https://openalex.org/A5017171605)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为UGCP的推理阶段模块，利用不确定性引导的保守传播在logit空间中迭代更新血管分割结果。

**💡 创新点**

创新点在于将预测不确定性、结构感知边缘调制和源项稳定化整合到有限步logit更新中，实现内部自适应信息传播，显著提升血管连通性与边界精度。

**🔧 技术方法**

采用端到端可微的logit空间迭代更新、Dirichlet证据推理、基于sigmoid的门控传播、结构边缘调制因子以及源项稳定化，兼容CNN和Transformer骨干网络。

**📊 数据集**

使用四个公开血管分割数据集：2D FIVES、ICA；3D ImageCAS、COSTA。

**📈 对比分析**

通过与基线CNN/Transformer模型以及CRF后处理对比，UGCP在DSC、centerline Dice提升0.3–2.3%并显著降低HD95（如COSTA从5.07像素降至3.34像素），整体性能持续提升。

**⚠️ 局限性**

局限性包括目前仅验证二分类血管分割，无法完整恢复极弱信号区域的血管；多类别分割和更高效更新策略仍待进一步研究。

---

## 245. Lowering the Barrier to IREX Participation: Open-Source Algorithms, Toolkit, and Benchmarking for Iris Recognition

**arXiv ID:** 2605.20735 | [PDF](https://arxiv.org/pdf/2605.20735v1)

**作者:** Siamul Karim Khan `[一作]` (University of Notre Dame), Adam Czajka `[通讯]` (University of Notre Dame)

**通讯引用:** 1420 | [OpenAlex ID](https://openalex.org/A5067121774)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本论文提出并实现了五个开源虹膜识别模块（SEGM、HDBIF、CRYPTS、TripletIris、ArcIris），其中TripletIris使用ConvNeXt‑tiny+Batch‑Hard Triplet loss，ArcIris使用ResNet100+ArcFace loss，另外提供了可直接提交给NIST IREX X评测的C++实现，并在公开数据集上完成了全面的性能评估。

**💡 创新点**

创新点包括：①首次将完整的开源虹膜识别系统（含分割、归一化、特征提取、匹配）以IREX‑compliant形式公开；②提出两种基于深度学习的端到端匹配模型（TripletIris、ArcIris），在公开数据集上逼近或匹配商业产品（VeriEye）的性能；③提供可解释性的CRYPTS方法，并将其改造为开源C++实现；④构建了高效的NestedSharedAtrousResUNet与ResNet18的圆形边界估计模型，显著提升分割与归一化速度。

**🔧 技术方法**

技术主要包括：卷积神经网络（ConvNeXt、ResNet）、Triplet loss与ArcFace loss、Batch‑Hard triplet mining、nested U‑Net与Atrous卷积的分割网络、Daugman rubber‑sheet归一化、HDBIF人类驱动的滤波器、CRYPTS基于晶体位移的特征匹配、LibTorch/TorchScript C++封装、IREX X API接口、OpenCV图像预处理。

**📊 数据集**

使用了八大虹膜基准集（Q‑FIRE、WBPMI、CASIA‑Iris‑Thousand V4、CASIA‑Iris‑Lamp V4、IITD‑Iris、NDIris3D、IIITD‑Contact‑Lens、Notre Dame Variable Iris Quality Release 2），以及训练集包括ND CCL、Photometric Stereo Iris Dataset、LivDet‑Iris 2017、ND‑Iris‑Contact‑Lenses 2010 等公开数据。

**📈 对比分析**

与现有开源方法（OSIRIS、USIT、DGR）以及商业系统（VeriEye、WCI）对比，ArcIris在大多数数据集上实现了接近或优于VeriEye的EER（低于1%）、Rank‑1准确率≥95%且FTE率几乎为0；TripletIris虽略逊于ArcIris，但仍明显优于传统开源模型；HDBIF和CRYPTS在解释性和特定场景（如法医后天虹膜）上具有优势，但在大规模匹配上速度和失败率较高。

**⚠️ 局限性**

局限性包括：CRYPTS匹配速度慢，导致在IREX X 1:N搜索中不符合时间限制；部分方法对低质量或异常虹膜仍有较高的失败率；深度模型对训练集的依赖可能导致在未见过的设备或人种上性能下降；缺乏针对实时嵌入式设备的优化；在法医后天虹膜场景下的鲁棒性仍待进一步验证。

---

## 246. Early High-Frequency Injection for Geometry-Sensitive OOD Detection

**arXiv ID:** 2605.20728 | [PDF](https://arxiv.org/pdf/2605.20728v1)

**作者:** Chuanjie Cheng `[一作]` (Nanjing Normal University), Yanhui Gu `[通讯]` (Nanjing Normal University)

**通讯引用:** 655 | [OpenAlex ID](https://openalex.org/A5100749023)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过频域诊断提出一种输入端高频注入方法EIHF，以提升基于特征几何的OOD检测效果。

**💡 创新点**

创新点在于：①利用band-wise MMD²揭示高频输入能更好区分ID与OOD特征；②在保持训练目标不变的情况下，仅在输入端添加固定高频残差通道，改变特征空间几何而非设计新的OOD分数。

**🔧 技术方法**

技术包括：频域分带处理、MMD²距离评估、固定高频残差通道的构造（Gaussian平滑差分）、Mahalanobis距离OOS分数、对比实验与消融研究。

**📊 数据集**

使用的数据集包括CIFAR-10、CIFAR-100、ImageNet-100作为ID，SVHN、LSUN、iSUN、Places365、Textures、SUN、iNaturalist等作为OOD。

**📈 对比分析**

与MSP、ODIN、Energy、ReAct、ASH、Vim、VOS、SSD+、KNN+、CIDER、PALM、DPL、DRL等基线在相同的训练和评分设置下对比，EIHF在CIFAR-100上获得最优平均FPR95和最高AUROC，ImageNet-100上在SUN/Textures上排名第一、平均FPR95最低，整体提升明显。

**⚠️ 局限性**

局限性：对场景级（如Places）OOV检测效果不如纹理级转移；受益主要集中在Mahalanobis等几何敏感的评分方法；高频残差在不同注入位置或失真后效果显著下降。

---

## 247. SAVER: Selective As-Needed Vision Evidence for Multimodal Information Extraction

**arXiv ID:** 2605.20713 | [PDF](https://arxiv.org/pdf/2605.20713v1)

**作者:** Miaobo Hu `[一作]` (Chinese Academy of Sciences), Jun Xiao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 297499 | [OpenAlex ID](https://openalex.org/A5111964102)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SAVER框架，在多模态信息抽取中对视觉证据进行选择性使用，仅在需要时激活视觉并选取少量有代表性的图片。

**💡 创新点**

通过分层校准的Conformal Groundability Gate控制视觉激活，并使用子模数相关多样性选择器构造紧凑证据集合，实现风险受控的选择性视觉推理。

**🔧 技术方法**

使用全局图像向量的Conformal门、子模数最大化选择器或RL选取器、Set Transformer聚合、能量启发式联合评分头，并结合文本与视觉的一致性正则与稀疏路由。

**📊 数据集**

在单图和多图的MRE与MNER基准上评估，包括MNRE、MRE-MI、Twitter-2015/2017、MNER-MI、MNER-MI-Plus等。

**📈 对比分析**

与文本基线、始终开启多模态模型以及多图选择性基线对比，SAVER在多图RE上获得最高F1、最小AURC、显著降低FLOPs和P90延迟；在单图场景提升幅度较小。

**⚠️ 局限性**

对图像的全局向量缓存依赖、子模数选择复杂度与图像数目相关，且在极少图像或视觉不相关的情况下选择机制仍有误判风险。

---

## 248. An Event-Driven Tool for Context-Aware Code Smell Detection Using SmellDSL

**arXiv ID:** 2605.20675 | [PDF](https://arxiv.org/pdf/2605.20675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 249. GSA-YOLO: A High-Efficiency Framework via Structured Sparsity and Adaptive Knowledge Distillation for Real-Time X-ray Security Inspection

**arXiv ID:** 2605.20669 | [PDF](https://arxiv.org/pdf/2605.20669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 250. Llamas on the Web: Memory-Efficient, Performance-Portable, and Multi-Precision LLM Inference with WebGPU

**arXiv ID:** 2605.20706 | [PDF](https://arxiv.org/pdf/2605.20706v1)

**作者:** Reese Levine `[一作]` (UC Santa Cruz), Tyler Sorensen `[通讯]` (Microsoft Research)

**通讯引用:** 561 | [OpenAlex ID](https://openalex.org/A5052590511)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了 llama.cpp 的 WebGPU 后端 LlamaWeb，使大语言模型能够在浏览器中高效、可移植地推理。

**💡 创新点**

通过静态内存规划、可调内核库和模板化 GPU 内核实现多精度量化支持，显著降低内存占用并提升性能。

**🔧 技术方法**

采用 WebGPU、WGSL、Emscripten、OPFS、静态内存阵列、子组矩阵、量化解码模板化以及可调性能参数等技术。

**📊 数据集**

在 16 台不同厂商的设备上评估 10 个模型（如 Llama3.2、Gemma、Qwen 等）和 4 种权重格式（GGUF 等）共 481 次基准跑。

**📈 对比分析**

通过与 WebLLM、Transformers.js 以及本地 llama.cpp 后端在预填充和解码吞吐率上对比，LlamaWeb 在解码阶段提升 45–69% 速度、内存占用下降 29–33%，并能与原生后端竞争甚至超越。

**⚠️ 局限性**

预填充阶段仍落后于 WebLLM/Transformers.js，缺乏内核融合；浏览器尚未支持子组矩阵功能；量化解码性能受限于现有数据布局。

---

## 251. Heartbeat-Bound Hierarchical Credentials: Cryptographic Revocation for AI Agent Swarms

**arXiv ID:** 2605.20704 | [PDF](https://arxiv.org/pdf/2605.20704v1)

**作者:** Saurabh Deochake `[一作]` `[通讯]` (SentinelOne Inc.), Saurabh Deochake (SentinelOne Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于心跳绑定的层级凭证协议（HBHC），实现 AI 代理集群在父级失效后自动在秒级内失效，消除僵尸代理问题。

**💡 创新点**

创新点在于将凭证有效期与父级定期心跳签名绑定，实现在离线（无网络）环境下的确定性撤销，并在层级结构中实现级联撤销。

**🔧 技术方法**

使用了 ECDSA、HMAC‑SHA256、BIP‑32 分层密钥派生、时间戳心跳签名以及可预计算、双路径等分发策略，并在 Rust/Python 实现。

**📊 数据集**

实验基于 OpenAI GPT‑4o‑mini 生成的工具调用与真实 LLM 代理集群，评估了在 10、100、1000、5000 等规模下的性能。

**📈 对比分析**

与 OAuth 2.0、短期 X.509、W3C 状态列表等传统撤销方案对比，HBHC 在僵尸窗口从数小时缩短到 40 秒，认证时延 0.26 ms，吞吐 18,000+ 验证/秒，且对抗注入后续工具调用零成功率。

**⚠️ 局限性**

局限在于需前期缓存父级公钥、依赖时钟同步、对心跳丢包导致误拒，以及无法解决键被提取或父级签名被盗的情况；未来需实现序列号、证明自动化与更大规模分发验证。

---

## 252. DIVE: Embedding Compression via Self-Limiting Gradient Updates

**arXiv ID:** 2605.20689 | [PDF](https://arxiv.org/pdf/2605.20689v1)

**作者:** Dongfang Zhao `[一作]` (University of Washington Tacoma), Dongfang Zhao `[通讯]` (University of Washington Tacoma)

**通讯引用:** 1364 | [OpenAlex ID](https://openalex.org/A5101671477)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为DIVE的压缩适配器，能够将大语言模型生成的高维嵌入压缩至几百维，同时保持或提升检索性能。

**💡 创新点**

创新点在于引入自限三元组损失与多头NT-Xent对比损失的稀疏-稠密梯度分解，既限制了对预训练空间的过度扰动，又通过多头视角提供稠密自监督梯度。

**🔧 技术方法**

采用三层MLP+多头投影、margin-based hinge三元组损失、头对头NT-Xent对比损失、参数高效微调等技术，并在检索阶段使用FAISS/HNSW等索引。

**📊 数据集**

在BEIR六个数据集（scifact、arguana、nfcorpus、scidocs、fiqa、quora）上进行实验。

**📈 对比分析**

与Matryoshka-Adaptor、Search-Adaptor和SMEC等基线在128/256/512维压缩比例下对比，DIVE在所有数据集和所有压缩比例上均优于基线且始终不低于冻结基线，nDCG@10显著提升。

**⚠️ 局限性**

仅适用于有标注的查询-文档对，无法在无监督场景下使用；性能依赖于冻结背骨嵌入质量；目前仅在英语检索任务中验证。

---

## 253. A Semantic and Occlusion-Aware GM-PHD Filter

**arXiv ID:** 2605.20666 | [PDF](https://arxiv.org/pdf/2605.20666v1)

**作者:** Jovan Menezes `[一作]` (Cornell University), Mark Campbell `[通讯]` (Cornell University)

**通讯引用:** 7297 | [OpenAlex ID](https://openalex.org/A5087789612)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种语义与遮挡感知的GM‑PHD出生模型，用于提升自动驾驶环境下多目标跟踪的初始化速度和精度。

**💡 创新点**

创新点在于将深度学习得到的语义先验与几何遮挡信息直接融入出生密度，动态地在遮挡边界、语义相关区域和传感器视场边界布置高斯出生分量，从而显著减少跟踪延迟并降低误报。

**🔧 技术方法**

技术方法包括随机有限集（RFS）框架下的GM‑PHD滤波、LiDAR/伪LiDAR点云语义分割、遮挡cone计算及多层出生分量建模，并采用动态概率和置信权重对不同出生来源进行加权。

**📊 数据集**

实验数据集主要使用KITTI（包含激光点云、语义分割标签）以及自建的2D鸟瞰交叉口仿真环境，用以评估出生模型在真实与模拟场景中的性能。

**📈 对比分析**

与部分均匀、均匀、适应性出生模型以及标准GM出生模型对比，S‑OA模型在仿真中平均跟踪延迟降低约30–40%，卡方误差与OSPA指标降低20–80%，在KITTI数据上延迟更少、误差更小，且在高噪声/杂波下保持较高精度。

**⚠️ 局限性**

局限性包括对语义分割与遮挡检测精度的依赖、出生权重估计仍需经验或交叉验证、以及在极端动态场景或多传感器融合时的扩展性尚待验证。

---

## 254. Gaze into the Details: Locality-Sensitive Enhancement for OCTA Retinal Vessel Segmentation

**arXiv ID:** 2605.20651 | [PDF](https://arxiv.org/pdf/2605.20651v1)

**作者:** Tuopusen Huang `[一作]` (Harbin Institute of Technology), Xiangqian Wu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 5018 | [OpenAlex ID](https://openalex.org/A5042120075)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种轻量化的LSENet网络，用于解决光学相干断层扫描血管成像（OCTA）中的低局部对比度和血管不连续问题，实现高精度的血管分割。

**💡 创新点**

创新点包括：① Patch Information Enhance（PIE）模块通过局部窗口注意力替代传统跨层跳接，聚焦低对比区域；② Multiscale Feature Fusion（MFF）模块在PIE之前提供多尺度视觉可解释特征；③ Connectivity Refinement Decoder（CRD）在最后采用大卷积核提升血管连通性，整体架构轻量化且参数极少。

**🔧 技术方法**

技术手段：U‑Net骨干 + GroupNorm、Patch‑based 注意力、3×3/5×5 多尺度卷积、通道注意力、11×11 大卷积核、BCE+Dice 复合损失、旋转/翻转数据增强、Adam优化。

**📊 数据集**

使用三套公开OCTA数据集：OCTA‑500（3mm、6mm）、ROSE‑1、ROSSA，分别进行训练/验证/测试划分。

**📈 对比分析**

与9种SOTA方法（U‑Net、CS‑Net、CE‑Net、SwinUNet、DGNet、FRNet、Vessel‑Net、UTNet、OCT²former）在同一训练环境下比较，LSENet在Dice、FDR、Kappa等关键指标上均位列前列或第一，并且参数量仅2.08M，显著低于传统模型，体现出优异的性能与高效性。

**⚠️ 局限性**

局限性：① 在敏感度与特异性之间仍有权衡，部分数据集上敏感度略低；② 大卷积核可能导致边缘细节轻微失真；③ 目前仅在OCTA三套数据上验证，跨模态或不同扫描设备的泛化性尚需进一步评估。

---

## 255. Jointly Learning Predicates and Actions Enables Zero-Shot Skill Composition

**arXiv ID:** 2605.20648 | [PDF](https://arxiv.org/pdf/2605.20648v1)

**作者:** Benedict Quartey `[一作]` (Brown University), Stefanie Tellex `[通讯]` (Brown University)

**通讯引用:** 4856 | [OpenAlex ID](https://openalex.org/A5059273574)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Predicate-Action Skills，通过联合学习动作轨迹与谓词信念轨迹，实现零样本技能组合；

**💡 创新点**

创新点在于把谓词视为与动作同步生成的轨迹，使得动作与符号效果在采样时天然对齐，提供可直接用于规划的在线符号接口；

**🔧 技术方法**

采用条件生成模型（DDPM、CFM）配合UNet或Transformer骨干，并用观测编码器对输入视觉进行特征提取；

**📊 数据集**

实验数据集包括2D PushBarrier、3D RoboMimic的Kitchen和Coffee任务，以及真实世界的立方体打包任务，并提供了技能分割与标签工具；

**📈 对比分析**

与动作生成单独、谓词预测单独以及多任务共享编码的基线对比，在PushBarrier上实现最高奖励、最短步骤，在RoboMimic上保持相近奖励同时宏观F1明显提升，证明联合建模不降低控制性能且提升谓词预测；

**⚠️ 局限性**

局限性包括对谓词集合的依赖、感知误差导致的状态混淆、对手工规划域的依赖以及缺乏自动发现谓词/操作的机制。

---

## 256. SCRIBE: Diagnostic Evaluation and Rich Transcription Models for Indic ASR

**arXiv ID:** 2605.20712 | [PDF](https://arxiv.org/pdf/2605.20712v1)

**作者:** Kavya Manohar `[一作]` (Adalat AI), Kumarmanas Nethil `[通讯]` (Adalat AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了SCRIBE评估框架并基于LLM数据策划构建了 Hindi、Malayalam、Kannada 的丰富转录 ASR 模型及对应基准。

**💡 创新点**

提出沙德容忍对齐与词汇、标点、数字、域实体四类错误分解的诊断工具，替代传统 WER，并公开相关工具、数据集与模型。

**🔧 技术方法**

采用 LLM 进行语料丰富化、Unicode 保留的词元化、沙德-aware 动态规划对齐及四类错误率计算等技术。

**📊 数据集**

使用约 1000h Hindi、850h Kannada、800h Malayalam 公开 Indic 语料；新基准 FLEURS-RO（通用）和 IN22-Legal（法律领域）。

**📈 对比分析**

与 IndicWhisper 与 IndicConformer 比较，SCRIBE‑ASR 在所有语言下实现最低 WER，数字错误率 <1%，域实体错误率 <2%；但标点错误率仍较高，尤其在 Dravidian 语言上。

**⚠️ 局限性**

仍无法完全解决标点分割难题，尤其在 agglutinative 语境下；对语音与语法耦合的处理仍有限；评估仍需人工校准以保证质量。

---

## 257. Interpretable Discriminative Text Representations via Agreement and Label Disentanglement

**arXiv ID:** 2605.20693 | [PDF](https://arxiv.org/pdf/2605.20693v1)

**作者:** Tong Wang `[一作]` (Yale University), Leo Yang Yang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现一种基于LLM的可解释文本特征发现方法（LFD），通过交叉LLM一致性筛选和残差增益选择，构建满足概念清晰与标签解耦的可审计特征空间。

**💡 创新点**

创新点在于将社会科学的构造效度标准（交叉Cohen's κ一致性）与标签解耦门槛直接嵌入特征学习流程，同时通过对比训练对抗性的残差增益实现自动化筛选。

**🔧 技术方法**

核心技术包括：多模态LLM交互（Proposer、Labeler、Examiner），Cohen's κ一致性检验，基于对比样本的残差增益选择，LightGBM等分类器，以及理论上基于噪声率的κ屏障分析。

**📊 数据集**

在十个文本分类任务上评测，覆盖七个语料库（包括广告、贷款申请、消费者投诉、点评、气话、iSarcasm、AG News等）。

**📈 对比分析**

与匿名判别基线（PCA、稀疏探针、全嵌入）及可解释非判别基线（LDA、BERTopic）以及单LLM概念瓶颈（TBM）比较，LFD在保持与TBM相近的预测准确率的同时，显著提高了特征的概念清晰度（κ>0.70）与标签解耦性（|ρ|<0.60）。

**⚠️ 局限性**

局限性包括：对规则可表述任务效果最佳，对复杂语义交互（如讽刺、欺骗式评论）性能不及匿名基线；需要多个LLM并行调用，成本和时间较高；不同LLM之间可能共享偏差，导致一致性检验的误判。

---

## 258. Distributional Alignment as a Criterion for Designing Task Vectors in In-Context Learning

**arXiv ID:** 2605.20730 | [PDF](https://arxiv.org/pdf/2605.20730v1)

**作者:** Jihoon Kwon `[一作]` (Seoul National University), Jy-yong Sohn `[通讯]` (Yonsei University)

**通讯引用:** 451 | [OpenAlex ID](https://openalex.org/A5006777962)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的度量指标d_NTP，用来评估任务向量与ICL推理结果的分布一致性，并基于该指标设计Linear Task Vector（LTV）方法；

**💡 创新点**

创新点在于将任务向量的质量直接与ICL的预测分布对齐作为评价标准，并通过最小化d_NTP的代理目标（MSE）得到闭式线性映射，从而实现高效、无训练的任务向量提取；

**🔧 技术方法**

核心技术包括对比KL散度与next-token概率的偏差度量、线性映射与Ridge回归求解任务向量、以及对隐藏层状态的加性更新；

**📊 数据集**

使用了8个文本分类基准（SST‑2、SST‑5、MR、AGNews、DBPedia、TREC、SUBJ、HateSpeech18）和2个回归基准（线性回归、ReLU回归），以及5种LLM模型（LLaMA‑2‑7B、LLaMA‑2‑13B、LLaMA‑3.1‑8B、Qwen‑2.5‑7B、Qwen‑3‑8B）；

**📈 对比分析**

与多种基线任务向量方法（Task Vector、Function Vector、State Vector、I2CL）以及零样本和ICL基线比较，LTV在所有模型上平均提升约9.2%准确率，单模型最高提升9.2%（LLaMA‑2‑13B），并且推理延迟与零样本相当；

**⚠️ 局限性**

局限性包括：只在分类和回归任务上评估，未验证对更复杂任务（如生成或多模态）的适用性；方法依赖线性近似，可能在任务效应高度非线性时表现受限；

---

## 259. CALMem : Application-Layer Dual Memory for Conversational AI

**arXiv ID:** 2605.20724 | [PDF](https://arxiv.org/pdf/2605.20724v1)

**作者:** Rajendra Narayan Jena `[一作]` (Infosys Limited), Sankar Arumugam `[通讯]` (Infosys Limited)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出 CALMem，一种在应用层实现的双层记忆架构，利用滑动窗口向量检索（episodic memory）和结构化事实存储（semantic memory）为 LLM 对话提供近乎无界的有效上下文，并通过 MOIM（记忆注入消息）在 token 预算内自适应地注入相关历史；

**💡 创新点**

创新点包括：① 在应用层实现双层记忆，既兼顾细粒度的向量检索，又能保持高精度的事实记忆；② 在会话压缩后仍可检索被压缩掉的历史（intra‑session retrieval）；③ token‑预算自适应注入策略，避免注入导致的窗口“加速压缩”；④ 纯应用层实现，零回归、无模型修改、跨供应商兼容；⑤ 采用纯 Rust + SQLite 的轻量实现，提供可配置的内存缓存和直观的 MOIM 格式；

**🔧 技术方法**

主要技术包括：滑动窗口分块 + All-MiniLM-L6-v2 句子向量（fastembed ONNX）；余弦相似度检索；双向过滤（跨会话与会话内）；token‑预算分层；SQLite 原生向量 blob 存储；可选的全内存嵌入缓存；Rust async + tokio；与 LLM 提供商的标准消息接口集成；

**📊 数据集**

评估使用了来自真实对话的 120 条查询‑会话对（共 30 会话）以及 50k‑100k 个向量分块；对比基线包括 BM25（SQLite FTS5）与 TF‑IDF；

**📈 对比分析**

实验结果显示：dense 句子检索的 Precision@5 达到 0.74，Recall@5 0.69，MRR 0.81，显著优于 BM25（P@5 0.59）和 TF‑IDF（P@5 0.51）。检索延迟在 5k–60k 片段时为 100–250 ms（DB‑scan）或 20–40 ms（内存缓存），远低于 LLM API 典型 500 ms–3 s；token 注入最多 750 tokens/轮，几乎不影响上下文容量。

**⚠️ 局限性**

局限性包括：检索仅基于语义相似度，缺乏时序推理；未实现重排序；单用户、单进程设计；分块采用字符滑窗，非语义分块；对精确匹配（如工单号）检索效果不佳，需混合 BM25；

---

## 260. E-ReCON: An Energy- and Resource-Efficient Precision-Configurable Sparse nvCIM Macro for Conventional and Spiking Neural Edge Inference

**arXiv ID:** 2605.20717 | [PDF](https://arxiv.org/pdf/2605.20717v1)

**作者:** Ankit Kumar Tenwar `[一作]` (Indian Institute of Technology Indore), Santosh Kumar Vishvakarma `[通讯]` (Indian Institute of Technology Indore)

**通讯引用:** 2218 | [OpenAlex ID](https://openalex.org/A5068792760)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于3T1R ReRAM位细胞的16KB数字计算内存宏E‑ReCON，用于边缘AI推理。

**💡 创新点**

创新点在于ADC‑less AND型位细胞与10T/28T交错加法树，显著降低面积、功耗并支持CNN和SNN两种工作负载。

**🔧 技术方法**

采用65 nm CMOS工艺实现的3T1R位细胞、交错加法树、数字累加与脉冲域乘法等技术。

**📊 数据集**

在MNIST/A‑Z、CIFAR‑10、SVHN、ImageNet‑1K等数据集上进行评估。

**📈 对比分析**

相较于先前ADC基ReRAM‑CIM设计，延迟从约1.5 ns降至0.48 ns，吞吐率提升至2.31–3.1 TOPS，能效高达419 TOPS/W；准确率在LeNet‑5、AlexNet、CNN‑8上分别达到97.81%、93.23%与96.51%。

**⚠️ 局限性**

局限性包括仅支持小规模（16 KB）位宽，未针对大规模网络训练，且在极高精度或更大阵列规模下可能面临功耗与面积瓶颈。

---

## 261. VISTAQA: Benchmarking Joint Visual Question Answering and Pixel-Level Evidence

**arXiv ID:** 2605.20676 | [PDF](https://arxiv.org/pdf/2605.20676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 262. Rethinking Cross-Layer Information Routing in Diffusion Transformers

**arXiv ID:** 2605.20708 | [PDF](https://arxiv.org/pdf/2605.20708v1)

**作者:** Chao Xu `[一作]` (Alibaba Group), Shao-Qun Zhang `[通讯]` (Nanjing University)

**通讯引用:** 65 | [OpenAlex ID](https://openalex.org/A5073419905)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对Diffusion Transformers（DiTs）的跨层信息路由进行系统诊断，并提出了Diffusion-Adaptive Routing（DAR）替代标准残差路由，利用时间自适应的softmax注意力聚合前层输出。

**💡 创新点**

创新点在于：①首次从深度与去噪时间步共同视角诊断残差路由问题，揭示前向幅度膨胀、梯度衰减、块间冗余三症；②提出可学习、时间自适应且非增量的跨层路由机制DAR；③在保持Transformer同质性的同时实现了可插拔式改进。

**🔧 技术方法**

技术包括：基于softmax的层间注意力聚合、时间步注入（静态、显式注入、动态），分块聚合（chunked aggregation）降低内存开销，结合adaLN-Zero条件路径与REPA对齐目标。

**📊 数据集**

使用ImageNet‑1K 256×256作为主要评测数据集，并在Qwen‑Image模型上进行后训练蒸馏实验。

**📈 对比分析**

与基线SiT-XL/2、U‑ViT、U‑DiT及REPA等方法对比，DAR在仅600K迭代下实现了6.92 FID（SDE）和8.61 FID（ODE）优于基线；相较于REPA，DAR在早期阶段可实现约2×的训练加速；同时保持参数量不变，效果显著。

**⚠️ 局限性**

局限性包括：仅在ImageNet规模数据上验证，缺乏对更大规模模型的系统评估；分块聚合的最佳chunk size仍需经验选择；以及对不同条件编码器或多模态任务的适应性尚未全面探测。

---

## 263. IndusAgent: Reinforcing Open-Vocabulary Industrial Anomaly Detection with Agentic Tools

**arXiv ID:** 2605.20682 | [PDF](https://arxiv.org/pdf/2605.20682v1)

**作者:** Rongbin Tan `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Huawei Cao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 IndusAgent，一个基于工具增强的多模态大语言模型框架，用于零样本工业异常检测，并在推理过程中主动调用裁剪、增强、先验检索和测量等工具。

**💡 创新点**

创新点包括：① 构建结构化工具集成推理数据集 Indus-CoT；② 设计 Accuracy‑Gated 多级奖励函数，抑制工具滥用；③ 采用 GRPO 强化学习实现工具驱动的主动推理；④ 通过监督微调对模型进行域特定诊断协议对齐。

**🔧 技术方法**

技术手段：多模态大语言模型 Qwen3‑VL‑8B；工具集（裁剪、增强、先验检索、测量）；结构化 Chain‑of‑Thought 生成；监督微调（SFT）+工具增强强化学习（GRPO）+门控奖励。

**📊 数据集**

使用的数据集：训练集 约 3,000 条 Indus‑CoT 轨迹（从 Real‑IAD 采集，去重与 MVTec‑AD、VisA、MPDD、DTD、SDD 等测试集类别重叠）；测试集包括 MVTec‑AD、VisA、MPDD、DTD、SDD。

**📈 对比分析**

与商业（如 GPT‑4o、Claude‑Sonnet‑4）和开源（如 Qwen3‑VL‑Instruct、Anomaly‑R1、LLaVA‑1.5 等）模型在五个基准上对比，IndusAgent 在平均 83.4% 的零样本性能中遥遥领先，尤其在结构复杂的 VisA 与 MPDD 上分别提升至 76.8% 与 72.7%，显著超过现有 SOTA。

**⚠️ 局限性**

局限性：仍依赖 8B 参数级模型，推理时需要多次工具调用导致计算成本；工具集相对有限，无法覆盖所有工业检测场景；对极端光照、纹理极度复杂或快速动态场景的鲁棒性尚待验证；缺乏对模型推理过程可解释性的系统分析。

---

## 264. DarkShake-DVS: Event-based Human Action Recognition under Low-light andShaking Camera Conditions

**arXiv ID:** 2605.20680 | [PDF](https://arxiv.org/pdf/2605.20680v1)

**作者:** Jiaqi Chen `[一作]` (Beijing Institute of Technology), Liyuan Pan `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 792 | [OpenAlex ID](https://openalex.org/A5064162540)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合事件相机与IMU的低光、6-DoF运动下的运动补偿与人类动作识别框架（EIS-HAR）并构建了DarkShake-DVS大规模基准数据集

**💡 创新点**

①首次构建低光下真实6-DoF抖动、同步IMU的HAR基准；②设计了基于IMU的自适应运动补偿（AIMC）与迭代贪婪采样（IGS）两大模块；③提出四阶段混合Swin Transformer（HSTS）实现空间-时间特征联合建模

**🔧 技术方法**

事件相机与IMU同步采集、非线性几何运动补偿、动态时间分段与自适应缩放、IGS关键帧选择、混合Swin Transformer、交叉熵损失训练

**📊 数据集**

DarkShake-DVS（18,041段，62类动作，低光+6-DoF+IMU），同时在HARDVS、DailyDVS-200上验证

**📈 对比分析**

与多种SOTA方法（ResNet、C3D、SwinT、Spikformer、Mamba等）在三数据集上对比；在DarkShake-DVS上准确率91.35%，在HARDVS 53.21%，在DailyDVS-200 51.99%；相比未补偿或统一采样均提升2-6个百分点，显示补偿与IGS有效

**⚠️ 局限性**

主要局限：仍对极端高强度抖动类别效果有限；对场景深假设、低帧率事件分辨率的适应性未知；仅关注单人/双人动作，未扩展到多人复杂交互

---

## 265. RoPeSLR: 3D RoPE-driven Sparse-LowRank Attention for Efficient Diffusion Transformers

**arXiv ID:** 2605.20659 | [PDF](https://arxiv.org/pdf/2605.20659v1)

**作者:** Yuxi Liu `[一作]` (Peking University), Kun Yuan `[通讯]` (Peking University)

**通讯引用:** 4339 | [OpenAlex ID](https://openalex.org/A5100614598)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 RoPeSLR 框架，通过结合 3D RoPE 与稀疏‑低秩注意力，实现极高稀疏率下的视频生成。

**💡 创新点**

理论证明 DiT 的注意力可拆分为稀疏高频尖峰与低秩背景，设计了非线性低秩补偿器和可学习的 3D 绝对位置嵌入，以解决 RoPE 兼容性难题。

**🔧 技术方法**

使用了稀疏‑低秩注意力、3D Rotary Position Embedding、绝对位置嵌入、低秩 MLP 补偿器、VMoBA 稀疏结构以及分布感知门控融合等技术。

**📊 数据集**

在 OpenSora 与 OpenVid‑1M 子集上进行实验，评估了 Wan2.1‑1.3B/14B 与 Hunyuan‑13B 模型的表现。

**📈 对比分析**

与 Full、SVG2、VSA、SLA 等稀疏或稀疏+线性基线对比，在 90% 稀疏率下实现 FLOPs 降低 10×、端到端速度提升 2.26×，VBench 质量降幅不到 1.3%。

**⚠️ 局限性**

当前稀疏分支使用的 VMoBA 实现尚未完全优化，导致实际端到端延迟提升低于理论值，需要进一步的硬件加速实现。

---

## 266. REFLECTOR: Internalizing Step-wise Reflection against Indirect Jailbreak

**arXiv ID:** 2605.20654 | [PDF](https://arxiv.org/pdf/2605.20654v1)

**作者:** Jiachen Ma `[一作]` (Fudan University), Chao Yang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 5188 | [OpenAlex ID](https://openalex.org/A5103069882)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大型语言模型的间接 Jailbreak 攻击，提出了 Reflector 框架，通过在生成过程中内化逐步自我反思机制，以实现对危险推理的实时监控与纠正。

**💡 创新点**

创新点：① 将安全对齐从表面门控扩展到轨迹级别；② 设计两阶段训练——先用教师引导的高质量反思数据进行 SFT 预训练，再用双奖励（安全 + 反思）进行 RL 微调；③ 引入反思奖励以鼓励模型主动触发反思并实现安全拒绝，显著提升间接 Jailbreak 的防御效果。

**🔧 技术方法**

技术细节：教师引导生成结构化反思数据（<z_reflect> 与 <z_explore> 标记）；使用 Supervised Fine‑Tuning (SFT) 在反思数据上训练；采用 Group Reward‑Decoupled Normalization Policy Optimization (GDPO) 的 RL，奖励函数为 r = r_safety + λ·r_reflect，r_safety 通过 HarmBench+GPT‑OSS 判别器联合决策，r_reflect 在出现反思且最终拒绝时给予 λ 奖励，否则惩罚。

**📊 数据集**

数据集：① 1500 条基于 DRA（BeaverTails）生成的间接攻击样本（GPT‑5 作为教师）+ 500 条 AlpacaEval 通用指令数据；② 评测时使用 StrongREJECT、XsTest、WildChat、Do‑Not‑Answer、AutoDAN、GCG、PAIR、DRA、PAP、ReNeLLM、DrAttack 等安全基准，另外使用 MMLU‑Pro、GSM8K、SimpleQA、AdvGLUE 等通用/专业任务评测。

**📈 对比分析**

对比方法：SFT、DPO、Self‑Critique、Shallow‑Align、STAIR 等对齐方案；在 LLaMA‑3.1‑8B‑Instruct 与 Qwen‑2.5‑7B‑Instruct 上评估。Reflector（SFT+GDPO）在所有间接 Jailbreak 上 DSR 超过 90%，在 XsTest 上 100%，在 WildChat 上提升 8.7%；同时保持甚至提升通用性能（MMLU‑Pro 45.20% / GSM8K +5.65%），说明其不产生典型的“对齐税”。

**⚠️ 局限性**

局限性：① 需要教师模型（如 GPT‑5）生成高质量反思数据，增加前期数据准备成本；② 反思奖励 λ 的调参对安全与通用性的平衡敏感，过大可能导致对齐过度；③ 主要针对推理式攻击，尚未验证对非文本或多模态输入的适用性；④ 训练时的 RL 计算开销仍高，尽管推理阶段的延迟可控。

---

## 267. Holistic Reliability Propagation: Decoupling Annotation and Prediction for Robust Noisy-Label

**arXiv ID:** 2605.20725 | [PDF](https://arxiv.org/pdf/2605.20725v1)

**作者:** Jingyang Mao `[一作]` (Nanjing Normal University), Yanhui Gu `[通讯]` (Nanjing Normal University)

**通讯引用:** 655 | [OpenAlex ID](https://openalex.org/A5100749023)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种Holistic Reliability Propagation (HRP) 框架，将外部标签和模型自生成伪标签的可靠性分离为α和β，并分别用于输入层的可靠性加权Mixup和表示层的伪标签对齐的对比学习。

**💡 创新点**

创新点在于通过双层元学习独立估计标签可靠性和伪标签可靠性，打破传统零和融合约束，并在不同训练模块中按需使用，实现更灵活、更鲁棒的噪声标签学习。

**🔧 技术方法**

使用技术包括双层元学习、可靠性加权Mixup (RAM)、全局可靠性门控 (GRG)、共识驱动对比学习 (CDCL) 等，配合双网络交叉训练。

**📊 数据集**

实验数据集涵盖 CIFAR-10、CIFAR-100 以及真实噪声的 Animal-10N。

**📈 对比分析**

与 DivideMix、RRL、UNICON、ProMix、LongReMix、L2B、PSSCL 等多种基线比较，HRP 在大多数噪声比例下获得最高平均准确率（如 CIFAR-10 平均 94.1%），并在 OOD 检测任务中展现优异性能。

**⚠️ 局限性**

局限性包括需少量干净验证数据，双网络与双层优化导致训练成本提升（约 1.3 倍），以及在极端噪声（90%）下过度保守导致梯度稀疏，限制最高噪声水平下的表现。

---

## 268. Robust Recommendation from Noisy Implicit Feedback: A GMM-Weighted Bayes-label Transition Matrix Framework

**arXiv ID:** 2605.20721 | [PDF](https://arxiv.org/pdf/2605.20721v1)

**作者:** Zongyu Li `[一作]` (Guangdong University of Technology), Yongshuai Yu `[通讯]` (Beijing Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

针对隐式反馈推荐系统中普遍存在的标签噪声问题，提出一种Robust GMM‑Weighted Bayes‑Label Transition Matrix (RGBT)框架，利用Gaussian Mixture Model（GMM）对实例可靠性进行评分，进而校准Bayes‑Label Transition Matrix（BLTM）并实现全样本利用与低方差估计。

**💡 创新点**

创新点在于：①将GMM用于对每个样本的可靠性权重进行细粒度评估，显著减轻BLTM估计的偏差与方差；②构建基于BLTM的实例依赖噪声模型，并提供理论保证其一致性和方差优势；③采用迭代的自适应阈值蒸馏策略，动态平衡样本利用率与噪声抑制。

**🔧 技术方法**

技术包括：Gaussian Mixture Model（GMM）可靠性建模；Bayes‑Label Transition Matrix（BLTM）估计网络；深度推荐模型（GMF、NeuMF、NGCF、LightGCN）与相应的损失与正则化；样本权重化的损失函数与联合优化策略；以及对噪声数据的仿真与评价指标（NDCG@K、Recall@K、L1矩阵误差）。

**📊 数据集**

使用真实数据集（Adressa、MovieLens、Yelp）以及基于MovieLens的合成“对称翻转”和“相邻翻转”噪声数据，评估模型在不同噪声率（0.1–0.4）下的性能。

**📈 对比分析**

与多种基线（普通训练、WBPR、WRMF、T‑CE、DeCA、SGDL、BLTM）及多种Transition Matrix方法（CCR、Dual‑T、T‑Revision、VolMinNet、RRFN、CONL）进行对比。实验结果显示：RGBT在NDCG/Recall上均优于传统去噪方法，并在L1矩阵误差上达到或接近最佳（对称噪声下L1≈1.38，低于RRFN 2.20），同时保持了更高的样本利用率和方差稳定性。

**⚠️ 局限性**

局限性包括：①需要额外的GMM训练与权重计算，增加计算开销；②对阈值ρ和融合权重λ的敏感性，需手动调参；③在极端噪声或非翻转类型噪声场景下的鲁棒性尚未充分验证；④实验仅覆盖隐式点击与评分场景，未探讨多模态或因果噪声的情况。

---

## 269. Decision-Path Patterns as Tree Reliability Signals: Path-based Adaptive Weighting for Random Forest Classification

**arXiv ID:** 2605.20716 | [PDF](https://arxiv.org/pdf/2605.20716v1)

**作者:** Youngjoon Park `[一作]` `[通讯]`, Youngjoon Park

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于随机森林树内部决策路径翻转模式的样本级加权方法。

**💡 创新点**

创新点在于将路径翻转模式视为树可靠性信号，并通过对预测类别进行条件化得到零期望类别偏差的权重。

**🔧 技术方法**

技术包括六种路径模式分类、森林级概率分桶、条件比率权重表、预计算叶节点路径模式、加权投票聚合。

**📊 数据集**

在30个二分类基准数据集（UCI 17个 + OpenML 13个）上进行实验。

**📈 对比分析**

与标准 RF、静态 OOB 加权 WRF、动态选择 KNORA‑Eliminate 与 KNORA‑Union 对比，实验显示在所有森林规模（100–1000棵）上均显著提升准确率（Wilcoxon p=0.018），并且少数类召回没有明显衰退，主要优势体现在不确定区间内。

**⚠️ 局限性**

局限性包括增益幅度很小、仅适用于二分类、需要额外5折交叉验证估计权重表、信号主要集中在置信度0.5–0.7区间，且对少量样本的细胞计数稀疏可能导致过拟合。

---

## 270. CandorMD: An AI-Assisted Audio Simulation and Feedback System for Training Clinicians for Medical Error Disclosure

**arXiv ID:** 2605.20701 | [PDF](https://arxiv.org/pdf/2605.20701v1)

**作者:** Inna Wanyin Lin `[一作]` (University of Washington), Tim Althoff `[通讯]` (University of Washington)

**通讯引用:** 4919 | [OpenAlex ID](https://openalex.org/A5033639139)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并评估了一款名为 CandorMD 的 AI 辅助音频模拟与反馈系统，帮助临床医生练习并改进医疗错误披露沟通。

**💡 创新点**

创新点在于：① 采用大语言模型生成动态、情感化的患者语音回应；② 双代理架构（患者模拟代理 + 评估代理）实现实时、情境化的逐回合反馈；③ 支持案例自定义、角色多样化（患者、家属）及即时“即插即用”式练习，弥补传统视频/角色扮演的资源和适应性不足。

**🔧 技术方法**

技术包括：OpenAI GPT‑4o/ GPT‑4o‑mini‑tts 生成文本与语音；人机交互框架设计；对话历史、信息与情感状态管理；实时情感评分与阶段检测；基于 SPIKES 等框架的评估模型。

**📊 数据集**

数据集主要来自 12 位关键利益相关者（临床医师、风险管理、患者倡导、沟通专家）的半结构化访谈与模拟会话；未使用公开医疗错误披露文本库，而是以专家提供的案例描述和即时生成的对话为训练与评估素材。

**📈 对比分析**

对比方法：与现有的 VCA（静态视频）和传统角色扮演进行对照，主要通过主题分析评估用户体验。用户认为模拟情感逼真、反馈及时、可重复练习；系统在情绪、事实准确性与对话流畅度方面优于传统方法，然而缺乏量化的学习成效指标。

**⚠️ 局限性**

局限性包括：① 评估基于定性访谈，缺乏客观量化结果；② 样本量小、受访者分布有限；③ 仅提供音频模拟，未覆盖多模态（面部表情、肢体语言）；④ 对 LLM 生成内容的准确性、偏见与法律/隐私风险仍需关注；⑤ 文化与组织层面的障碍（错误文化、政策限制）无法单靠技术解决。

---

## 271. Distributed Direct Preference Optimization

**arXiv ID:** 2605.20696 | [PDF](https://arxiv.org/pdf/2605.20696v1)

**作者:** Zhanhong Jiang `[一作]` (Iowa State University), Zhanhong Jiang `[通讯]` (Iowa State University)

**通讯引用:** 841 | [OpenAlex ID](https://openalex.org/A5031212270)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在分布式学习环境中，对直接偏好优化（DPO）的理论收敛性和时间复杂度进行首次系统分析，提出 FedDPO 与 DecDPO 两种算法。

**💡 创新点**

创新点在于：① 通过轨迹级 Sigmoid 权重推导出 DPO 的光滑性常数与方差上界；② 给出 FedDPO 在全参与、部分参与、异步延迟场景下的收敛率与下界；③ 证明 DecDPO 的收敛速度受网络谱间隙控制；④ 通过下界展示对异构性、局部步骤和采样量的依赖是不可去除的。

**🔧 技术方法**

使用技术包括：梯度下降、FedAvg、去中心化梯度下降、轨迹级对数比损失、Sigmoid 导数分析、马尔科夫链混合假设、光滑性与方差分析、谱间隙理论。

**📊 数据集**

实验数据集：Stanford Human Preferences (SHP) 与 Anthropic HH‑RLHF，分别在分布式环境下划分为非 IID 的子集。

**📈 对比分析**

与集中式 DPO、FedDPO（全参与/部分参与）、DecDPO 环网等进行对比；实验表明 DecDPO 在梯度范数下降和最终误差上最优，FedDPO 受参与率和异构性影响，实验结果与理论预测高度吻合。

**⚠️ 局限性**

限制：仅在 log‑linear softmax 策略下证明；对 Transformer 结构的理论分析缺失；轨迹混合假设为指数衰减，可能在长时间或近确定性环境中过于保守；未考虑通信压缩、拜占庭客户端、Adam/FedAdam 等优化器。

---

## 272. Design for Manufacturing: A Manufacturability Knowledge-Integrated Reinforcement Learning Framework for Free-Form Pipe Routing in Aeroengines

**arXiv ID:** 2605.20644 | [PDF](https://arxiv.org/pdf/2605.20644v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 273. Layer-wise Token Compression for Efficient Document Reranking

**arXiv ID:** 2605.20683 | [PDF](https://arxiv.org/pdf/2605.20683v1)

**作者:** Shengyao Zhuang `[一作]` (Amazon AGI), Ivano Lauriola `[通讯]` (Amazon AGI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Layer-wise Token Compression（LTC）技术，在Transformer文档重排序模型的中间层进行自适应Token池化以降低计算成本并保持效果。

**💡 创新点**

创新点在于将Token压缩从输入层迁移到中间层，既保留早期层的细粒度交互信息，又通过后续层的压缩显著提升吞吐量，并证明此策略对长文本也有正向正则化效应。

**🔧 技术方法**

主要技术包括1D自适应平均池化、可学习的压缩比例r、可调节的目标层l*，以及在列表式LLM重排序器中对文档Token单独压缩的mask机制。

**📊 数据集**

使用MS MARCO（passage和document）作为训练集，评估使用TREC DL19/20（passage、document）以及BEIR六个异构数据集的零样本性能。

**📈 对比分析**

与不压缩基线以及Jasper嵌入层压缩方案对比，LTC在中间层压缩时保持或提升nDCG@10，QPS提升幅度可达25%（passage）到116%（document）及列表式模型近200%；在长文本和零样本场景下，压缩模型往往优于未压缩基线。

**⚠️ 局限性**

局限性包括：压缩层和比例需手工搜索；只评估了平均池化压缩，未与其他压缩/剪枝方法做系统比较；强压缩仍需压缩感知微调，完全无训练的压缩方案尚未实现。

---

## 274. Deep Attention Reweighting: Post-Hoc Attention-Based Feature Aggregation in CNNs for Disentangling Core and Spurious Features under Spurious Correlations

**arXiv ID:** 2605.20732 | [PDF](https://arxiv.org/pdf/2605.20732v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 275. Modular Multimodal Classification Without Fine-Tuning: A Simple Compositional Approach

**arXiv ID:** 2605.20674 | [PDF](https://arxiv.org/pdf/2605.20674v1)

**作者:** Herman Bergström `[一作]` (Chalmers University of Technology and University of Gothenburg), Rahul G. Krishnan `[通讯]` (Vector Institute)

**通讯引用:** 2497 | [OpenAlex ID](https://openalex.org/A5073514348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CoMET 方法，将冻结的多模态编码器与 Tabular Foundation Model（TFM）组合，构成一个训练‑free 的多模态分类管道。

**💡 创新点**

创新点包括：① 仅使用 PCA 作为模态到 TFM 的适配器即可取得强劲性能；② 提出 PALPooling（Pseudo Attention Label Pooling），一种不需要梯度、可快速拟合的自适应池化方法；③ 通过组合强大的冻结编码器与 TFM，达到多模态任务的 state‑of‑the‑art 性能，挑战了传统端到端训练的必要性。

**🔧 技术方法**

采用的技术包括：冻结预训练编码器（DINOv3、ELECTRA、Sentence‑BERT 等）、PCA 降维、Tabular Foundation Models（TabICL、TabPFN）、PALPooling、无梯度拟合、上下文学习（in‑context learning）以及对多模态特征的拼接与投影。

**📊 数据集**

使用的数据集涵盖图像、文本、表格及层级标签，主要包括：News Channel、Salary、PAD‑UFES、Petfinder、Pneumonia、Butterflies、MS COCO、Open Images、IMDB、20 Newsgroups、Yelp、AG News、Amazon Reviews、Web of Science、Linux Bugs、iNaturalist、Amazon MM 等。

**📈 对比分析**

与传统基线（TTT、TabSTAR、MMPFN、AutoGluon、Catboost、TE、2‑layer MLP）以及同类方法进行对比。CoMET 在大多数数据集上匹配或超过现有 SOTA，PALPooling 在单模态实验中提升准确率（尤其在 Pneumonia、Butterflies 上显著），H‑CoMET 在层级分类任务中优于平面模型，显示出显著的性能优势。

**⚠️ 局限性**

限制与未来工作：① 目前仅使用 PCA 作为适配器，缺乏更精细的 TFM‑aware 降维方法；② 主要验证文本与图像模态，未对音频、视频、3D 等模态进行评估；③ 在某些任务（如 Wine、Petfinder）表现不如专门针对该模态的细粒度方法；④ 对大规模数据与长上下文窗口的可扩展性仍待进一步验证。

---

## 276. Seeing Through Fog: Towards Fog-Invariant Action Recognition

**arXiv ID:** 2605.20645 | [PDF](https://arxiv.org/pdf/2605.20645v1)

**作者:** Enqi Liu `[一作]` (Beijing Institute of Technology), Qing Li `[通讯]` (Beijing Institute for General Artificial Intelligence)

**通讯引用:** 38990 | [OpenAlex ID](https://openalex.org/A5100404176)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FogAct 真实雾天动作识别数据集，并设计了端到端的 FogNet 模型，实现对雾天视频的雾无关特征提取与动作分类。

**💡 创新点**

创新点包括：①首个包含雾天与清晰视频配对的真实动作数据集；②FogNet 通过 Fog‑Aware Selection、Mutual Enhancement 与 Cross‑Stream Alignment 三个模块，在无需先行去雾的前提下学习雾无关语义；③利用大型视觉‑语言预训练模型（CLIP）进行特征对齐与判别。

**🔧 技术方法**

使用的技术包括：两流 CLIP 编码器、全局自注意力、双向交叉注意力、帧级一致性对齐、对比损失 (InfoNCE) 与多任务训练；模型基于 ViT‑B/16 结构，训练时采用 AdamW、cosine 退火与 warm‑up。

**📊 数据集**

使用的数据集为：FogAct（约 10,000 对雾天/清晰视频，55 类动作、10 场景）、UCF‑101、HMDB‑51 与 Kinetics‑100 的 ASM 合成雾版本，以及公开的其他雾天视频（如 Foggy Zurich 等）作对照。

**📈 对比分析**

与单阶段与两阶段现有 SOTA（如 OST、DINO‑based、LLaMa‑VID 等）进行对比。FogNet 在 FogAct 上实现 Top‑1 88.7%、Top‑5 99.4%，超过 OST 的 83.2%/98.9% 以及其他基线约 5–10%；在合成雾数据上平均提升 4.2%（最高 9.8%）。

**⚠️ 局限性**

局限性：需要雾/清晰配对训练样本；对细微/短时动作的识别仍有困难；数据集仅覆盖 10 场景，未测试其他恶劣天气；模型依赖大型 CLIP 预训练，计算资源要求较高；跨雾天的泛化能力尚待进一步验证。

---

## 277. DisImpact: Quantifying the Physi-Social Impact of Natural Disasters Through Social Media

**arXiv ID:** 2605.20646 | [PDF](https://arxiv.org/pdf/2605.20646v1)

**作者:** Ruichen Yao `[一作]` (University of Illinois Urbana-Champaign), Dong Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 83940 | [OpenAlex ID](https://openalex.org/A5100699858)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DisImpact 两阶段框架，利用多模态大语言模型对 Reddit、TikTok、YouTube 公开帖子进行十类灾害影响标签，然后按时间窗口计算统一影响指数，并对物理与社会维度进行聚合与分析。

**💡 创新点**

创新点在于：①将物理与社会影响融合为统一尺度的指数；②利用多模态 LLM 对海量社交媒体进行高效标注；③通过滑动窗口加权的影响指数既捕捉相对占比，又反映讨论强度；④跨平台、跨时空的综合验证。

**🔧 技术方法**

核心技术包括：多模态大语言模型（Gemini‑2.0‑flash）进行内容分类与过滤；加权比例平滑与极坐标函数的强度权重；Spearman 相关系数进行与官方数据的时序验证。

**📊 数据集**

数据集包含 2024 年大西洋飓风和 2025 年加州野火期间，从 Reddit、TikTok、YouTube 共 134,053 条原始帖子，经过过滤后约 84,404 条相关帖子，用于构建影响指数并与 FEMA 公共援助和 NASA FIRMS 火灾辐射功率对齐。

**📈 对比分析**

通过 Spearman 相关检验与 FEMA 及 FIRMS 真实数据的对齐，得到社交影响指数在 3 周领先 FEMA 的最大相关系数约 0.44，火灾物理指数在同一时点与 FIRMS 的相关系数约 0.42，说明指数在时间上可提前预警且与权威数据相关性中等到良好。

**⚠️ 局限性**

主要限制：①单标签标注忽略多重影响；②仅基于社交媒体缺乏对人群整体情绪与长期心理影响的覆盖；③时间窗口局限于数月，无法评估长期恢复；④统一指数虽易比较但在专业领域缺乏细粒度解释。

---

## 278. GAMR: Geometric-Aware Manifold Regularization with Virtual Outlier Synthesis for Learning with Noisy Labels

**arXiv ID:** 2605.20727 | [PDF](https://arxiv.org/pdf/2605.20727v1)

**作者:** Ningkang Peng `[一作]` (Nanjing Normal University), Yanhui Gu `[通讯]` (Nanjing Normal University)

**通讯引用:** 655 | [OpenAlex ID](https://openalex.org/A5100749023)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Geometry-aware Manifold Regularization (GAMR)，通过主动合成虚拟异常样本构造能量屏障，从而重塑特征空间，使干净样本更紧凑、噪声样本更分离。

**💡 创新点**

创新点在于：①利用极值理论估计特征空间边界并在低密度区采样虚拟异常样本；②将这些样本与能量模型相结合，形成可微分的能量障碍；③实现了无需先验噪声模型的通用正则化，可无缝集成至现有样本筛选框架。

**🔧 技术方法**

技术手段包括：极值理论（EVT）+超矩形边界采样；能量基模型（EBM）+能量正则化（SPADE）；双网络互检与自监督对比学习；滑动窗口自适应样本筛选。

**📊 数据集**

使用了CIFAR-10、CIFAR-100、Animal‑10N、Food‑101等公开数据集，并在多种对称/非对称噪声设置下进行评估。

**📈 对比分析**

与 DivideMix、LongReMix、PSSCL、UNICON 等主流噪声学习方法比较，GAMR 在所有噪声水平下均达到或超过SOTA，尤其在 90% 对称噪声（93.7%）和 49% 非对称噪声（87.9%）上显著领先；在 OOD 检测中 AUROC 与 FPR95 均提升。

**⚠️ 局限性**

局限性：①每次迭代需要额外采样与能量计算，导致推理/训练时延增加约 44%；②对距离阈值 τ 的选取仍有一定经验依赖；③在极高噪声率或极低信噪比的真实场景下，边界估计可能不够精确，需进一步稳健化。

---

## 279. Design Principles and Observable Indicators for AI-Enabled Pedagogical Accompaniment: Evidence from the Amico Dual-Mode Prototype in Italy and China

**arXiv ID:** 2605.20665 | [PDF](https://arxiv.org/pdf/2605.20665v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 280. AGPO: Adaptive Group Policy Optimization with Dual Statistical Feedback

**arXiv ID:** 2605.20722 | [PDF](https://arxiv.org/pdf/2605.20722v1)

**作者:** Miaobo Hu `[一作]` (Chinese Academy of Sciences), Jun Xiao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 297499 | [OpenAlex ID](https://openalex.org/A5111964102)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AGPO（Adaptive Group Policy Optimization），一种无价值函数的强化学习框架，利用组级统计量自适应控制裁剪半径和采样温度，以改进LLM推理性能。

**💡 创新点**

核心创新在于：①用共享的组级统计量同时驱动自适应裁剪和双向温度采样；②在无critic的GRPO基础上实现在线可调节的更新幅度与探索性；③通过简单统计（熵、KL、奖励方差、偏度、投票熵）实现高效、鲁棒的自适应策略。

**🔧 技术方法**

使用技术包括：组化滚动采样、奖励归一化、GRPO截断目标、基于熵、KL漂移、奖励方差/偏度、投票熵的自适应裁剪公式、双向温度采样（ATS）、指数移动平均（EMA）、多种分散度估计（STD、MAD、IQR）以及基于这些统计的控制器。

**📊 数据集**

实验评估涵盖九个中英文数学/ STEM 基准：GSM8K、MATH、OCW、SAT、MMLU_STEM、CodeContests、CMATH、GaokaoMath-Cloze、GaokaoMath-QA。

**📈 对比分析**

与PPO、Adaptive-KL PPO、DPO、GRPO（固定裁剪）、GRPO+ATS、AGPO（无ATS）和AGPO（完整）等多种基线对比，AGPO（完整）在所有九个任务上获得最高分，平均比PPO提升约8.3个百分点，比最强基线提升约1.4个百分点；训练速度比GRPO略低，但远快于PPO。

**⚠️ 局限性**

主要局限：对组级统计的可靠性要求较高，若奖励定义不当、奖励稀疏或噪声过大，统计估计可能失真，导致自适应控制失效；目前仍未针对极端稀疏或对抗性奖励进行鲁棒性验证。

---

## 281. Declarative Data Services: Structured Agentic Discovery for Composing Data Systems

**arXiv ID:** 2605.20690 | [PDF](https://arxiv.org/pdf/2605.20690v1)

**作者:** Shanshan Ye `[一作]` (Northeastern University), Duo Lu `[通讯]` (Brown University)

**通讯引用:** 467 | [OpenAlex ID](https://openalex.org/A5102305858)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Declarative Data Services (DDS) 架构，实现从自然语言意图到多系统数据后端的结构化 agentic discovery。

**💡 创新点**

四层 typed contract 结构与 L4 attribution loop 使搜索空间分层 bounded，知识通过 skill 记录持续积累，解决传统 unbounded 发现的收敛难题。

**🔧 技术方法**

基于 LLM（Claude Opus 4.6）+ 子代理搜索、类型验证、operator DAG、技能 YAML、runtime 信号归因与迭代循环技术。

**📊 数据集**

以单人交易者实时分析工作负载为实验场景，使用 Coinbase/Binance 交易数据流和自定义查询；并附加聊天平台案例作为示例。

**📈 对比分析**

与三种无结构 agentic discovery baseline（自然语言、系统列表、技能 YAML）对比，DDS 在一次迭代即可达 100% T0/T1/T2，成本最低、迭代率最高；baseline 需多轮、成功率低、成本高。

**⚠️ 局限性**

仅在单一域、单模型、单宿主环境验证；缺少 L5 SLO 验证层；归因规则手工写，缺少学习化；技能版本和跨产品演化未完善。

---

## 282. LT2: Linear-Time Looped Transformers

**arXiv ID:** 2605.20670 | [PDF](https://arxiv.org/pdf/2605.20670v1)

**作者:** Chunyuan Deng `[一作]` (Rice University), Hanjie Chen `[通讯]` (Rice University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LT2（Linear‑Time Looped Transformers）架构，将循环 Transformer 的 quadratic softmax attention 替换为子二次的 linear / sparse attention，并探索混合（hybrid）混合器。

**💡 创新点**

创新点：①将循环机制与 sub‑quadratic attention 结合，利用循环实现记忆细化与感受野扩展；②设计两种混合策略（GDN+DSA 与 Full+GDN）在保持或提升性能的同时大幅提升解码吞吐量；③提出可从预训练的循环 Transformer 转化为 LT2‑Hybrid 的多阶段蒸馏方法。

**🔧 技术方法**

使用技术：循环 Transformer、线性注意力（GDN、KDA、DeltaNet）、稀疏注意力（窗口、DSA）、混合注意力、门控机制、对数似然蒸馏（KL、top‑k softmax）、参数共享、正则化与梯度控制。

**📊 数据集**

使用数据集：FineWeb‑Edu（语言建模 100B 词）、OpenThoughts‑v3（长上下文扩展）、SWDE、SQuAD、FDA、TriviaQA、NaturalQuestions、DROP、NIAH‑Single‑1/2/3（长上下文检索）以及合成的状态跟踪/记忆任务。

**📈 对比分析**

比较方法：与标准循环 Transformer、全注意力 Transformer、各类 sub‑quadratic 变体在相同参数量、相同循环次数（T=4）下进行零样本评测、语言建模困惑度、长上下文检索准确率、吞吐量等指标比较。实验显示：LT2‑Hybrid (GDN+DSA) 在 1.3B 规模下在 zero‑shot 上与全注意力循环相当，吞吐量提升约 5.7×；LT2‑Hybrid (Full+GDN) 在性能上进一步提升 (+2.1 0‑shot 分数) 并保持 5× 以上的速度优势；蒸馏模型 Ouro‑Hybrid‑1.4B 在仅 1B 词的训练后即可与 1B 级别模型竞争，甚至逼近 4B 模型。

**⚠️ 局限性**

限制：仅探讨深度级别混合与简单的循环级别调度；未研究完整的循环级别混合（不同迭代使用不同注意力族）或跨循环状态共享机制；在极长上下文或更大规模下的稳定性与通用性仍待进一步验证。

---

## 283. AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals

**arXiv ID:** 2605.20643 | [PDF](https://arxiv.org/pdf/2605.20643v1)

**作者:** Duy Nguyen `[一作]` (University of North Carolina Chapel Hill), Mohit Bansal `[通讯]` (University of North Carolina Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 AVSD（Adaptive-View Self-Distillation）方法，利用同一模型在多种特权视图（如完整解答、最终答案、部分推理等）下作为教师，在自我强化学习框架中对学生进行 on‑policy 细粒度的 token 级别蒸馏。

**💡 创新点**

创新点在于：①通过几何均值和算术均值分别构造共识目标与松散目标，揭示多视图教师分布之间的共享与差异；②设计了基于共识一致性与残差比例的门控机制，只在视图一致且残差不会反转更新方向时才加入残差，从而兼顾稳定性与信息丰富度；③不需要外部教师模型，直接利用学生自身在不同特权条件下的预测实现多视图蒸馏。

**🔧 技术方法**

技术手段包括：基于 token‑level reverse‑KL 的优势函数；多视图教师分布的几何平均（共识）与算术平均（松散）聚合；残差门控策略（alignment 与 magnitude 两个分量）；在共识目标上加权残差后重归一化得到最终蒸馏目标；使用 on‑policy reverse‑KL 损失进行训练。

**📊 数据集**

使用的数据集有：
- 训练集：OpenThoughts 里数学推理子集；
- 评估集：AIME 2024、AIME 2025、HMMT 2025；
- 代码生成集：Codeforces Python 子集（5K 例子）和 LiveCodeBench v6；
- 视图构造：数学题的完整解答、最终答案、部分推理；
  代码题的参考实现、算法提示、运行反馈。

**📈 对比分析**

与 SFT、GRPO、单视图自蒸馏（OPSD）等基线进行对比，评估指标为 Avg@8。结果表明：
- 在 Qwen3‑4B 上 AVSD 从 73.3 提升到 76.2（+2.9%）;
- 在 Qwen3‑8B 上从 74.2 提升到 75.4（+1.2%）并相较 GRPO 提升 3.1%；
- 在 DeepSeek‑R1‑Distill‑Qwen‑7B 上也获得 2.3% 的平均提升；
- 代码生成任务同样显示 2.4%–4.9% 的 Avg@8 增益。整体表现优于所有基线，且随着模型规模增大收益更为显著。

**⚠️ 局限性**

局限性：
- 需要多种可获取的特权视图，若缺少某些视图会影响效果；
- 随视图数量增多收益递减，超过 3–4 视图后提升有限；
- 额外的教师前向计算会带来一定的计算开销；
- 对于不具备可验证奖励或无明显特权信息的任务，方法的优势不易显现。

---

## 284. Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds

**arXiv ID:** 2605.20723 | [PDF](https://arxiv.org/pdf/2605.20723v1)

**作者:** Lakshani Manamperi `[一作]` (University of Moratuwa), Kutila Gunasekera `[通讯]` (University of Moratuwa)

**通讯引用:** 192 | [OpenAlex ID](https://openalex.org/A5088151430)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在资源受限的安卓手持设备群体上，利用CROWDio实现了多阶段ONNX模型（如DistilBERT）推理的分布式执行，避免了模型裁剪或云端依赖；

**💡 创新点**

提出了结合JIT延迟加载、单分区驻留、4层亲和调度、压缩张量传输与流式1:1依赖模型的完整调度子系统，实现了在不改动模型的前提下，将大模型的显存压力均匀分布到设备集群；

**🔧 技术方法**

采用ONNX Runtime、WebSocket通信、zlib自描述JSON压缩、基于心跳的亲和性调度、动态任务依赖管理及JIT分区加载等技术；

**📊 数据集**

使用DistilBERT（≈67M参数）在SST‑2情感分析数据集上的分层推理作为评测基准；

**📈 对比分析**

与传统的Barrier模式对比，单设备峰值RSS被限定在约43 MB，批处理延迟从27.9 s降低至18.4 s（约34%），冷启动耗时48 s，热启动6 s，压缩率约62%；

**⚠️ 局限性**

局限性包括冷启动耗时高、单分区驻留对高RAM设备不够灵活、zlib压缩对CPU占用较高、仅支持线性拓扑、需用户同意参与等。

---

## 285. An Application-Layer Multi-Modal Covert-Channel Reference Monitor for LLM Agent Egress

**arXiv ID:** 2605.20734 | [PDF](https://arxiv.org/pdf/2605.20734v1)

**作者:** Alfredo Metere `[一作]` `[通讯]` (Enclawed), Alfredo Metere (Enclawed)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种面向 LLM 代理出口的参考监控器，利用分阶段文本处理链和音视频去隐写变换，并通过引入 boot‑time 可信签名门控来检测合法媒体，彻底消除可破坏的隐写通道。

**💡 创新点**

创新点在于将多模态隐写的检测转为签名鉴定，使用信息论互信息作为残余容量度量，并提供统一的容量预算管道和可扩展的 adversarial benchmark；同时首次实现对可听频段隐写的签名式防御。

**🔧 技术方法**

技术细节包括十阶段有序文本处理链（规范化、污点跟踪、熵扫描、噪声注入、语义扰乱、LLM 重写、时序扰乱、行为约束）、RGB 6 位量化图像变换、FFT 频域音频滤波、基于 Ed25519 的 boot‑time 可信签名门控、Miller‑Madow 校正的互信息测度，以及可插拔的 15 个工作编码器/解码器。

**📊 数据集**

实验数据集为自制的多模态隐写工作负载，包括文本零宽字符、同形字、空格计数、同义词替换等；图像 LSB、均值、序列置换编码；音频超声和次可听编码，共 15 个编码器/解码器组合；不依赖公开数据集。

**📈 对比分析**

评估方法为在每个通道收集 (X,Y) 样本，计算 Miller‑Madow 校正的互信息来衡量残余容量；实验结果显示在全强制模式下大多数可破坏通道残余容量为 0，均值亮度通道仅限 3 位，默认 staged 模式保持合法流且无误差，同时容量削减可接受。

**⚠️ 局限性**

局限性包括无法完全破坏均值亮度通道、对合法签名者的依赖导致密钥泄露窗口、LLM 重写带来延迟与不确定性、对可听频段隐写的 arms‑race 仍未完全解决、以及在某些高分辨率场景下可能出现的可见失真。

---

## 286. TASTE: A Designer-Annotated Multi-Dimensional Preference Dataset for AI-Generated Graphic Design

**arXiv ID:** 2605.20731 | [PDF](https://arxiv.org/pdf/2605.20731v1)

**作者:** Haonan Zhu `[一作]` (Lica World), Purvanshi Mehta `[通讯]` (Lica World)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并公开了TASTE数据集，对四个主流文本到图像模型产生的图形设计进行九个子维度（排版、色彩和谐、视觉层级、情绪调性、描述准确性等）专业设计师偏好评分，并将评分与生成模型进行对比。

**💡 创新点**

创新点在于①将设计质量拆分为多维子维度并收集真实设计师偏好；②提出三统计量（Kendall τ、majority-vote probability、Condorcet cycle）对偏好数据的信号检验框架；③在TASTE上训练轻量级差分评分头，显著提升对设计维度的预测。

**🔧 技术方法**

技术手段包括：基于Bradley–Terry模型的加权对数损失、对齐多头MLP评分器、对比性评估使用Kendall τ、majority-vote probability、Condorcet cycle、以及与六个开放权重VLM和三种专用评分器的宏观对照实验。

**📊 数据集**

使用的数据集为：TASTE（四个T2I模型、10名专业设计师、9维度共1600条评分/维度），对照数据集为Sushi、MovieLens、以及HPSv2测试集。

**📈 对比分析**

通过宏观agreement（与5名设计师多数投票比较）评估模型，所有开放权重VLM和专用评分器的精度≤0.55；自研差分评分头在验证集上达0.611，接近人类留一法上限0.741，表明模型在设计维度预测上有显著提升。

**⚠️ 局限性**

局限性包括：每个评估点仅有5名评审、仅使用英语提示、样本来自同一平台、仅覆盖九个子维度、未包含动画/品牌一致性、可访问性等设计维度，且对跨文化、跨语言的适应性未知。

---

## 287. MTR-Suite: A Framework for Evaluating and Synthesizing Conversational Retrieval Benchmarks

**arXiv ID:** 2605.20729 | [PDF](https://arxiv.org/pdf/2605.20729v1)

**作者:** Junhao Ruan `[一作]` (Northeastern University), Jingbo Zhu `[通讯]` (Northeastern University)

**通讯引用:** 2161 | [OpenAlex ID](https://openalex.org/A5100370155)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了MTR‑Suite框架，用于评估、自动合成并基准化对话检索；

**💡 创新点**

创新点在于引入LLM评估MTR‑Eval、贪心遍历聚类的多代理生成MTR‑Pipeline以及基于生产场景的MTR‑Bench；

**🔧 技术方法**

技术包括LLM-as-Judge、Greedy Traversal Clustering、递归分块、MinHash‑LSH、NVIDIA质量分类器、FineWeb‑EDU、Embedding检索等；

**📊 数据集**

使用了2025年1月Wikipedia dump、内部金融语料库以及公开的QReCC、QuAC、Doc2Dial、TopiOCQA等对比数据集；

**📈 对比分析**

通过Recall@5/20、NDCG@20、MRR@20等指标对比传统数据集与MTR‑Bench，发现MTR‑Bench显著降低Recall并提高检索难度；

**⚠️ 局限性**

局限性在于仅评估检索模块，未涉及端到端生成，且受LLM生成质量与提示工程的影响。

---

## 288. Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting

**arXiv ID:** 2605.20678 | [PDF](https://arxiv.org/pdf/2605.20678v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 289. Beyond Semantic Similarity: A Two-Phase Non-Parametric Retrieval Workflow for Corporate Credit Underwriting

**arXiv ID:** 2605.20684 | [PDF](https://arxiv.org/pdf/2605.20684v1)

**作者:** Linus Ng Junjia `[一作]` (OCBC), Zhao Jing Yuan `[通讯]` (OCBC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一套企业信用审批用的检索增强生成系统，采用双阶段检索+效用驱动重排序，并实现结构化证据提取。

**💡 创新点**

提出效用驱动检索框架、适应性检索控制器以及上下文感知证据抽取，解决传统检索的相似度‑实用性缺口。

**🔧 技术方法**

使用词典+密集检索（多语言嵌入）、轻量级 LM 进行相关性与支持评估、LLM‑as‑Judge 进行效用评分，以及结构化文本与表格分块提取技术。

**📊 数据集**

内部多语言金融文件（年度报告、行业分析等）与信用分析师标注的相关性标签数据集。

**📈 对比分析**

与传统仅基于语义相似度的检索比较，系统在企业内部部署后将文档审阅时间从数小时降至约3分钟，检索准确率显著提升。

**⚠️ 局限性**

局限性包括数据集受保密限制无法公开，表格复杂性处理仅保留上下文，未实现完整的结构化数值提取，未来需集成 OCR/VLM 等技术。

---

## 290. AIR: Amortized Image Reconstruction Framework for Self-Supervised Feed-Forward 2D Gaussian Splatting

**arXiv ID:** 2605.20820 | [PDF](https://arxiv.org/pdf/2605.20820v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 291. On the limits and opportunities of AI reviewers: Reviewing the reviews of Nature-family papers with 45 expert scientists

**arXiv ID:** 2605.20668 | [PDF](https://arxiv.org/pdf/2605.20668v1)

**作者:** Seungone Kim `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (Carnegie Mellon University)

**通讯引用:** 22025 | [OpenAlex ID](https://openalex.org/A5068811427)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过45名领域专家对82篇Nature系列期刊论文的人类评审和三款前沿LLM生成评审进行细粒度条目级别的人工评注（共2960条），评估其正确性、重要性和证据充分性。

**💡 创新点**

首次在单条评审的三维标准下量化AI评审的优劣，并揭示AI评审能补充而非替代人类评审；同时提出了PeerReview Bench基准和CMU Paper Reviewer开放平台，提供持续评估与实际预投稿反馈的工具。

**🔧 技术方法**

使用GPT‑5.2、Claude Opus 4.5、Gemini 3.0 Pro等LLM代理（通过OpenHands实现文件与工具访问），以及GPT‑5.4进行评审条目相似度自动判定；通过Meta‑reviewer AI 自动化评审提升评注效率。

**📊 数据集**

以82篇Nature（含Nature Communications等）公开论文及其正式人类同行评审为核心数据集，补充相应预审稿件；构建了78篇论文的PeerReview Bench基准和CMU Paper Reviewer所用的公开评审数据。

**📈 对比分析**

对比方法：与人类最高/最低评级评审进行匹配率、全正向比例、正确率、重要性得分、证据充分性等多维度统计；结果显示GPT‑5.2在全正向比例上显著高于顶级人评审（60% vs 48.2%），Claude 4.5与Gemini 3.0 Pro均超过最低人评审；AI评审覆盖率高但与彼此重叠率显著高；在PeerReview Bench上，GPT‑5.4精确度最高（93.8%）但召回仅26%，Claude‑Opus 4.5的F1≈50.9%。

**⚠️ 局限性**

主要局限：AI评审的正确率仍低于顶级人评审，存在显著错误；对子领域特定方法论缺乏深入理解、长文档上下文跟踪不足以及对细小问题过度批判等缺陷；多AI评审会降低视角多样性；评审耗时高且仅针对Nature系列论文，泛化性和在其他期刊的表现尚未验证。

---

## 292. VLA-REPLICA: A Low-Cost, Reproducible Benchmark for Real-World Evaluation of Vision-Language-Action Models

**arXiv ID:** 2605.20774 | [PDF](https://arxiv.org/pdf/2605.20774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 293. LER-YOLO: Reliability-Aware Expert Routing for Misaligned RGB-Infrared UAV Detection

**arXiv ID:** 2605.20667 | [PDF](https://arxiv.org/pdf/2605.20667v1)

**作者:** Liming Hou `[一作]` (Engineering University of PAP), Yubo He `[通讯]` (Engineering University of PAP)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种可靠性感知稀疏MoE融合框架，用于处理RGB-红外无人机目标检测中的空间偏移问题。

**💡 创新点**

创新点在于在对齐阶段预测局部可靠性图，并将其作为路由先验，动态选择RGB主导、红外主导或交互专家，实现可靠性驱动的多模态融合。

**🔧 技术方法**

采用了Uncertainty-Aware Target Alignment（U-TA）进行可学习特征对齐，利用稀疏MoE融合与可靠性引导的专家路由，以及YOLOv5s检测头。

**📊 数据集**

实验使用公开的Misaligned Bimodal UAV (MBU) 基准（由Anti-UAV数据集提取）。

**📈 对比分析**

在YOLOv5s族基准下与静态融合、参数匹配等对照组对比，LER-YOLO在平均精度（AP）上达到89.7%±0.2%，比静态对齐+融合提升约2个百分点。

**⚠️ 局限性**

局限性包括目前仅采用三专家，稀疏激活优势有限；路由网络引入额外开销；对真实传感器误差的评估仍受限，需在更广泛平台验证。

---

## 294. Most Transformer Modifications Still Do Not Transfer at 1-3B: A 2020-2026 Update to Narang et al. (2021) with Downstream Evaluation and a Noise Floor

**arXiv ID:** 2605.20798 | [PDF](https://arxiv.org/pdf/2605.20798v1)

**作者:** Yang Zhao `[一作]` (Tencent), Jie Zhou `[通讯]` (Tencent)

**通讯引用:** 35651 | [OpenAlex ID](https://openalex.org/A5100620306)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对20个2021年后Transformer改进在1–3B规模下进行统一代码、配置和多种种子噪声基准下的系统评估

**💡 创新点**

首次将噪声基准、下游任务评估和跨尺度稳定性检查作为评测架构改进的必备三要素

**🔧 技术方法**

使用统一Llama‑style模型、混合域CLIMB‑Mix‑400B预训练、-12下游基准、三种种子噪声校准、Bonferroni/Benjamini‑Hochberg等统计检验

**📊 数据集**

CLIMB‑Mix‑400B预训练语料及12项下游任务（PIQA、ARC、HellaSwag、WinoGrande、SocialIQA、MMLU、OpenBookQA、BoolQ、RACE、LAMBADA、TruthfulQA‑MC2）

**📈 对比分析**

结果显示大多数改进不具可转移性，仅Softpick与HybridNorm在1.2B显著提升，随后3B尺度检验进一步筛除一项；注意力输出改动出现失真与损失-下游解耦；对比中基线误差约0.2%

**⚠️ 局限性**

局限在于只评估20个改进、未进行每种改动的超参调优、仅测试两种规模，且基准固定可能掩盖特定架构或优化器的潜在收益

---

## 295. VIHD: Visual Intervention-based Hallucination Detection for Medical Visual Question Answering

**arXiv ID:** 2605.20772 | [PDF](https://arxiv.org/pdf/2605.20772v1)

**作者:** Jiayi Chen `[一作]` (Monash University), Jianfei Cai `[通讯]` (Monash University)

**通讯引用:** 24802 | [OpenAlex ID](https://openalex.org/A5100635804)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练无关的视觉干预幻觉检测方法 VIHD，用于识别医疗视觉问答中模型生成的幻觉文本。

**💡 创新点**

创新点在于通过视觉依赖探测定位关键解码层，使用高注意力视觉 token 层级掩蔽进行内部干预，并以校准语义熵衡量幻觉严重度，实现对视觉与语言交互的细粒度自我监测。

**🔧 技术方法**

使用视觉注意力权重分析、视觉 token 掩蔽（VID）、对比与补充融合的校准语义熵（CSE）以及无监督的多样本采样框架。

**📊 数据集**

实验覆盖 VQA-RAD、SLAKE、VQA-Med-2019 三个医疗 VQA 数据集，并在 Hulu-4B 与 LingShu-7B 两种大型医学多模态模型上进行评估。

**📈 对比分析**

与七种最先进的 introspective 检测方法（AvgProb、MaxProb、AvgEnt、MaxEnt、RadFlag、SE、VASE）对比，VIHD 在 AUC 与 AUG 上平均提升 5%–12%，显著优于 VASE，且推理时间与 VASE 相近。

**⚠️ 局限性**

局限性包括对视觉注意力估计的依赖可能受模型架构或图像复杂度影响，且方法尚未验证在非视觉模态幻觉、不同解码策略或跨模型迁移时的鲁棒性。

---

## 296. Cumulative Meta-Learning from Active Learning Queries for Robustness to Spurious Correlations

**arXiv ID:** 2605.20771 | [PDF](https://arxiv.org/pdf/2605.20771v1)

**作者:** Kin Whye Chew `[一作]` (National University Of Singapore), Jingxian Wang `[通讯]` (National University Of Singapore)

**通讯引用:** 3283 | [OpenAlex ID](https://openalex.org/A5101929678)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种利用主动学习查询样本来学习模型先验（归纳偏置）的框架CAML，以提升对真实世界数据中伪相关性的鲁棒性。

**💡 创新点**

创新点在于：①将主动学习视为元学习任务，利用查询样本作为元测试信号；②设计累积式元学习目标，捕捉主动学习过程中的序列依赖，加入交互项以强化已学归纳偏置；③通过这种累积偏置显著放大少数族群样本的影响。

**🔧 技术方法**

核心技术包括：基于一阶MAML的元学习算法、累积式元目标（递推求解）、最大后验推断视角下的先验学习、以及对主动学习的序列化建模。

**📊 数据集**

在四个伪相关性基准上验证：MNIST‑CIFAR Dominoes、Waterbirds、SpuCo、CivilComments，并使用预训练ResNet‑18与BERT模型。

**📈 对比分析**

与传统主动学习（仅将查询样本加入训练集）相比，CAML在多数数据集与主动学习策略下均提升了少数族群测试准确率，最大提升可达约30%。

**⚠️ 局限性**

局限性包括：对查询函数的依赖（需高质量的主动采样）；额外的计算开销（累计元目标导致更多前向/反向传递）；以及实验范围主要聚焦于伪相关性基准，需进一步验证在更广泛场景下的适用性。

---

## 297. The Illusion of Intervention: Your LLM-Simulated Experiment is an Observational Study

**arXiv ID:** 2605.20767 | [PDF](https://arxiv.org/pdf/2605.20767v1)

**作者:** Victoria Lin `[一作]` (Google DeepMind), Alexander D'Amour `[通讯]` (Google DeepMind)

**通讯引用:** 2709 | [OpenAlex ID](https://openalex.org/A5060694111)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了利用大型语言模型（LLM）模拟用户进行实验时，由于用户属性随不同干预而漂移，导致的选择偏倚（confounding bias），并提出一种基于负控制结果的诊断方法以及通过逐步提取并在 persona 指令中加入额外属性的调整策略，以减小用户漂移对实验效果估计的影响。

**💡 创新点**

创新点在于：①把 LLM 生成的用户漂移形式化为因果推断中的选择偏倚；②首次使用负控制结果作为检测工具，衡量不同干预下潜在用户属性分布的变化；③提出迭代的属性提取与调整框架，实证证明能显著降低偏倚并使估计趋于稳定。

**🔧 技术方法**

采用因果推断理论（潜在结果框架、后门调整）、负控制结果检验、总变差距离（TVD）衡量分布差异、基于 prompt 的属性提取与调参、以及在多款 LLM（Qwen3、Gemma‑4、Gemma‑3、GPT‑OSS、Gemini 3 Flash）上进行实验。

**📊 数据集**

实验数据集包括：OpinionQA（调查问卷）、NYT Book Opinions（书籍偏好）和 MovieLens（电影偏好），这些数据提供了真实用户的特征与问卷/偏好答案，用作 LLM 生成的“模拟用户”对比。

**📈 对比分析**

通过对比未调整（iteration 0）与多轮迭代调整后的 TVD 与观测效应，评估方法有效性。结果显示：在多数模型与数据集上，TVD 随迭代显著下降，观测效应趋于稳定；但对某些模型（如 GPT‑OSS）调整效果有限，说明方法并非普适。整体而言，方法能够显著降低用户漂移带来的偏差，但仍受模型能力与负控制选取的影响。

**⚠️ 局限性**

局限性包括：①负控制结果的选择对检测灵敏度有很大影响；②prompt‑based 调整无法完全等同统计条件化，可能引入新的依赖或样本偏差；③在强推理能力的 LLM 中，漂移更难控制；④实验规模受算力限制，尤其对 Gemini 3 Flash 的样本量有限；⑤即使 TVD 下降，也不保证完全消除偏差，仍需结合其他校正技术。

---

## 298. Correcting Stochastic Update Bias in Preconditioned Language Model Optimizers

**arXiv ID:** 2605.20756 | [PDF](https://arxiv.org/pdf/2605.20756v1)

**作者:** Nikhil Nayak `[一作]` (Fastino Labs), Ash Lewis `[通讯]` (Fastino Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文识别并校正了随机预条件优化中的梯度-预条件耦合偏差与逆预条件的有限样本偏差，并提出了一种单批交叉拟合与方差校正的偏差纠正框架；

**💡 创新点**

创新点在于对两种偏差进行统计分解，提出可在单一步骤内实施的交叉拟合与delta法逆校正，使得该方法同时适用于对角预条件（AdamW、Sophia）和矩阵预条件（Shampoo）；

**🔧 技术方法**

采用交叉拟合预条件、Delta方法/Jackknife逆校正、对角/矩阵预条件实现（AdamW、Sophia、Shampoo），以及微批量方差估计；

**📊 数据集**

使用Qwen2.5-0.5B模型在FineWeb-Edu打包序列的预训练、20% span-replaced混合质量训练以及Alpaca风格指令调优的数据集；

**📈 对比分析**

通过与标准优化器比较（held-out交叉熵损失），偏差校正在预训练中分别降低0.15、0.07、0.11 nats，指令调优差异小但保持竞争力；

**⚠️ 局限性**

仅在0.5B规模模型、有限计算资源与单次实验上验证，缺乏对更大模型和不同数据集的广泛测试，且矩阵校正依赖特征分解近似，批量拆分对结果敏感。

---

## 299. GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation

**arXiv ID:** 2605.20752 | [PDF](https://arxiv.org/pdf/2605.20752v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 300. PulseCol: Periodically Refreshed Column-Sparse Attention for Accelerating Diffusion Language Models

**arXiv ID:** 2605.20813 | [PDF](https://arxiv.org/pdf/2605.20813v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 301. Spatial Gram Alignment for Ultra-High-Resolution Image Synthesis

**arXiv ID:** 2605.20808 | [PDF](https://arxiv.org/pdf/2605.20808v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 302. The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?

**arXiv ID:** 2605.20749 | [PDF](https://arxiv.org/pdf/2605.20749v1)

**作者:** Xingyu Lyu `[一作]` (Chinese Academy of Sciences), Qingming Huang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 32257 | [OpenAlex ID](https://openalex.org/A5028597017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在两层网络的神经切线核（NTK）极限下，门控线性单元（GLU）结构对训练优化的影响；通过理论推导与实验验证，发现GLU能显著改善NTK谱的条件数，导致更快的梯度下降收敛；同时评估GLU对泛化误差的影响，发现其对泛化间隙提升有限。

**💡 创新点**

核心创新在于揭示GLU门控通过Hadamard乘法重构NTK矩阵，使得谱更紧凑、条件数下降，从而解释了GLU在优化过程中的加速特性；进一步将谱重塑与训练阶段动态及损失交叉现象联系起来，提供了对GLU优势的深度理论解释。

**🔧 技术方法**

主要技术手段包括：神经切线核（NTK）框架、随机矩阵理论（Wishart矩阵与Marchenko–Pastur分布）、Hadamard乘法与谱条件数分析、梯度下降在NTK极限下的方向性动态推导。

**📊 数据集**

实验所用数据集包括：高维高斯合成数据、MNIST、CIFAR‑10（MLP Mixer）、Tiny ImageNet（ViT）、FineWeb‑Edu（GPT‑2）等，覆盖图像与语言模型多种场景。

**📈 对比分析**

通过将GLU与传统非门控模型（ReLU、GEGLU、Silu等）在相同参数量下进行对比，评估NTK条件数、训练损失曲线和泛化间隙。结果显示：GLU模型在训练损失下降速度更快，出现损失交叉现象；但在相同训练误差水平下，GLU与非GLU模型的泛化间隙基本重合，表现差异不显著。

**⚠️ 局限性**

局限性包括：理论分析仅适用于两层网络和无限宽度的NTK极限，未涵盖更深网络或非线性训练动态；实验验证主要关注条件数和训练收敛，未深入探讨GLU对不同优化器、正则化或数据不平衡情况下的泛化效果；此外，GLU在提高训练速度方面的优势在实际大模型中可能被其他因素所掩盖。

---

## 303. Draw2Think: Harnessing Geometry Reasoning through Constraint Engine Interaction

**arXiv ID:** 2605.20743 | [PDF](https://arxiv.org/pdf/2605.20743v1)

**作者:** Juncheng Hu `[一作]` (National University of Singapore), Joey Tianyi Zhou `[通讯]` (Agency for Science, Technology and Research)

**通讯引用:** 11110 | [OpenAlex ID](https://openalex.org/A5045125183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Draw2Think 框架，将 VLM 的几何推理外部化到 GeoGebra 约束引擎，通过 Propose‑Draw‑Verify 循环实现中间状态可验证；

**💡 创新点**

创新点在于把几何推理视为与约束引擎的闭环交互，使用 Typed ToolSpecs 让 VLM 发出可被引擎直接检查的构造指令，并引入 Construction Fidelity 与 Measurement Faithfulness 两项可审核的保证，且无需额外训练；

**🔧 技术方法**

技术包括 Gemini 等 VLM、GeoGebra 动态几何引擎、Giac CAS、Typed ToolSpec 接口、Propose‑Draw‑Verify 循环与结构化观察；

**📊 数据集**

使用了 GeoGoal、GeoSketch、MathVerse、GeoQA/UniGeo、PGPS9K、Geo3K、OlympiadBench、SolidGeo‑hard、MathVista、GenExam‑math 等多种几何与渲染基准；

**📈 对比分析**

通过 Pass@1 与基线单步 VLM 对比；在 planar benchmark 提升 4.1%，在 solid benchmark 提升 16.4%；在 GeoGoal 上 predicate‑level 通过率 95.9%、问题‑level 通过率 84.0%；在 GenExam‑math strict/relaxed 渲染得分分别为 68.2/90.5，超过领先 T2I 系统 11.9 分；

**⚠️ 局限性**

局限性在于只对单步操作进行验证，策略层未被检查，导致模糊感知、长链构造或无必要工具使用时仍会失败；目前仅针对 Euclidean 单数值实例，未覆盖更复杂或非数值几何问题。

---

## 304. OlmoEarth v1.1: A more efficient family of OlmoEarth models

**arXiv ID:** 2605.20804 | [PDF](https://arxiv.org/pdf/2605.20804v1)

**作者:** Gabriel Tseng `[一作]` (Allen Institute for AI), Patrick Beukema `[通讯]` (Allen Institute for AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本工作对 OlmoEarth 视觉变压器模型进行改进，推出 v1.1 版本，通过模型结构与训练策略的若干优化，显著降低训练与推理时的计算成本，并在大多数下游任务上保持甚至提升性能。

**💡 创新点**

核心创新包括：① 将多波段集合合并为单一 bandset，并引入随机波段丢弃与非线性投影层，以恢复跨波段交互；② 更新掩码策略，使所有地图类目标始终可见并加入时序掩码；③ 修改损失函数，剔除极难负样本并仅对解码模式进行相似度阈值筛选。

**🔧 技术方法**

使用了掩码图像建模（LatentMIM Lite）、对比学习（InfoNCE）以及多尺度、跨模态的 token 表示；模型采用 Encoder‑Decoder 视觉 Transformer，训练时使用 AdamW、学习率预热与余弦退火；增添了非线性投影、随机波段丢弃、时间掩码与改进的损失。

**📊 数据集**

训练与评估使用多模态遥感数据集，包括 Sentinel‑1、Sentinel‑2、Landsat‑8、WorldCereal、OpenStreetMap、WorldCover、Cropland Data Layer、SRTM 与 Canopy Height Map。下游任务涵盖 m‑bigearthnet、m‑so2sat、m‑brick‑kiln、m‑eurosat、BreizhCrops、CropHarvest‑Togo、CropHarvest‑PRC、m‑cashewplant、m‑SA‑crop‑type、PASTIS、MADOS、AWF、Nandi 等。

**📈 对比分析**

通过 kNN、线性探针与微调三种评测方式进行对比。总体而言，Nano、Tiny、Base 三个规模的 v1.1 在大部分任务上保持与 v1 相近的性能，部分任务（如 m‑eurosat、CropHarvest）略有下降，而在 m‑bigearthnet、BreizhCrops、PASTIS 等任务上表现略优。训练 GPU 时数由 2,989h 降至 1,763h（约 1.7×节省），推理与微调时的计算成本亦显著降低。

**⚠️ 局限性**

局限性包括：对某些高分辨率或跨波段依赖较强的任务仍有性能下滑；时间掩码引入的超参数 p_t 需进一步调优；非线性投影层对模型稳定性有影响，需平衡其宽度与模型规模；以及本研究未考虑硬件制造与能源外部成本，仍有进一步降低碳足迹的空间。

---

## 305. Q-SpiRL: Quantum Spiking Reinforcement Learning for Adaptive Robot Navigation

**arXiv ID:** 2605.20801 | [PDF](https://arxiv.org/pdf/2605.20801v1)

**作者:** Mohamed Khair Altrabulsi `[一作]` (New York University Abu Dhabi), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11394 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在本研究中，作者提出了一种量子增强的脉冲强化学习框架Q‑SpiRL，用于自适应机器人在含静态和动态障碍的网格世界中导航，并实现了从传统表格Q‑学习到密集MLP、脉冲SNN、量子增强MLP和量子增强脉冲SNN等五种代理的统一训练与评估；

**💡 创新点**

创新点在于将参数化量子电路嵌入脉冲神经网络的决策管线，利用脉冲产生的放电率特征作为量子层的输入，实现了时域稀疏处理与量子特征转换的融合；同时提出了全局Q‑表转换和确定性贪婪推理的评估协议，保证不同架构在相同条件下可比；

**🔧 技术方法**

核心技术包括强化学习（DQN），脉冲神经网络（LIF + Poisson编码），参数化量子电路（8量子比特、3层变分块），以及在IBM量子硬件上执行的量子测量；

**📊 数据集**

实验数据集为三种网格规模（20×20、30×30、40×40）的人工生成环境，包含随机静态障碍和动态障碍，使用100个未见种子进行测试；

**📈 对比分析**

通过在100个测试环境上进行确定性贪婪推理，评估成功率、成功加权路径长度（SPL）、路径长度和转弯率，实验结果显示QSNN在所有规模下均实现最高的SPL、最低的转弯率和与表格Q学习相当甚至更高的成功率，整体性能优于传统MLP、SNN和量子增强MLP；

**⚠️ 局限性**

局限性包括：仅在简化的网格世界中验证，动作空间有限；量子硬件执行受噪声和采样波动影响，单次硬件跑的结果不具统计意义；训练仅使用CPU且只进行800集，未探索更大规模或更复杂的环境；

---

## 306. BioDefect: The First Dataset for Defect Detection in Bioinformatics Software

**arXiv ID:** 2605.20788 | [PDF](https://arxiv.org/pdf/2605.20788v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 307. Beyond Numerical Features: CNN-Driven Algorithm Selection via Contour Plots for Continuous Black-Box Optimization

**arXiv ID:** 2605.20797 | [PDF](https://arxiv.org/pdf/2605.20797v1)

**作者:** Yiliang Yuan `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Mustafa Misir `[通讯]` (Duke Kunshan University)

**通讯引用:** 1016 | [OpenAlex ID](https://openalex.org/A5024551178)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在连续黑盒优化中使用轮廓图视图构建实例表示，并利用CNN回归器预测各算法的性能，从而实现实例级的自动算法选择。

**💡 创新点**

创新点在于抛弃传统的数值特征（如ELA），改为直接对采样得到的二维轮廓图进行视觉学习；通过堆叠或单视图编码两种CNN结构，首次证明视觉表示能够在连续优化中取得与数值特征相当甚至更好的性能。

**🔧 技术方法**

技术上使用了：固定网格采样生成二维灰度轮廓图、图像尺寸缩放（64/128/300）、三层CNN或ResNet-18编码器、回归头预测relERT/relHV、留一交叉验证、统计显著性检验。

**📊 数据集**

数据集包括：BBOB 2009 单目标（24 函数 × 4 维 × 5 变体），以及在 Deep-ELA 协议下的双目标 BiBBOB、DTLZ、MMF、ZDT 共 33 个实例。

**📈 对比分析**

与单最优算法、ELA/Deep-ELA 基线进行比较。单目标下，CNN 模型将平均 relERT 从 30.37 降至 5.60（相当于或优于最强基线 5.72）；双目标下，CNN 取得与 Deep-ELA 相近的 relHV（0.94–0.98），在 ZDT 等子类中甚至击败传统方法。

**⚠️ 局限性**

局限性包括：采样预算高（每实例需 5×300×300 次评估），仅使用二维切片，难以捕捉高维结构；仅评估 d=2 的双目标场景；模型受限于固定算法组合和预算设置；需要进一步研究低预算、多视图和混合特征的鲁棒性。

---

## 308. CMC-Opt: Constraint Manifold with Corners for Inequality-Constrained Optimization

**arXiv ID:** 2605.20796 | [PDF](https://arxiv.org/pdf/2605.20796v1)

**作者:** Yetong Zhang `[一作]` (Georgia Institute of Technology), Frank Dellaert `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 24592 | [OpenAlex ID](https://openalex.org/A5087336025)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了约束带角（Constraint Manifold with Corners, CMC）框架，将等式和不等式约束嵌入状态空间，实现无约束优化；

**💡 创新点**

创新点是将不等式约束纳入微分几何中，定义带角约束流形，给出局部参数化、切空间、微分与重投影，并将Riemannian梯度下降扩展到该结构；

**🔧 技术方法**

使用因子图分解、约束流形重投影、梯度下降/LM、欧几里得投影、Gauss–Newton Hessian近似及trust‑region等技术；

**📊 数据集**

在13连杆四足机器人跳跃障碍的时空轨迹规划问题（70步、四阶段）上进行实验；

**📈 对比分析**

与惩罚法、增量拉格朗日、SQP、CM‑Opt对比，CMC‑Opt实现约束违约率0、目标成本709.78，搜索空间从32,194降至2,260，计算时间6.3e2s，性能显著优于基线；

**⚠️ 局限性**

局限性包括不适用于跨时间步约束（导致连通组件过大、重投影子问题难解）且需要低维切空间构造，当前实现仅针对内在可微分不等式约束。

---

## 309. Learning to Think in Physics: Breaking Shortcut Learning in Scientific Diffusion via Representation Alignment

**arXiv ID:** 2605.20780 | [PDF](https://arxiv.org/pdf/2605.20780v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 310. HyFrac.fun: A 3D Hydraulic Fracturing Simulator on Cloud

**arXiv ID:** 2605.20764 | [PDF](https://arxiv.org/pdf/2605.20764v1)

**作者:** Jing Hu `[一作]` (Tongji University), Jaroon Rungamornrat `[通讯]` (Chulalongkorn University)

**通讯引用:** 1445 | [OpenAlex ID](https://openalex.org/A5048231000)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并实现了HyFrac.fun平台，实现了3D非平面水力压裂的传播与生产的端到端云计算仿真。

**💡 创新点**

将传播和生产两种不同的SGBEM–FEM方程结构同构，实现在云端自动无转换地把演化的3D裂缝网格直接传给生产求解；提出完整自适应重建算法；实现增量矩阵更新、缓存友好重排、OpenMP并行、动态求解器切换；搭建SaaS云架构。

**🔧 技术方法**

采用对称Galerkin BEM、FEM、非牛顿粘性流体动力学模型；增量矩阵更新、缓存友好重排、OpenMP并行；混合Preconditioned Krylov迭代与直接求解器；FastAPI异步后台、TRAME/VTK服务器渲染、WebSocket、Nginx反向代理与容器化。

**📊 数据集**

未使用公开真实数据，采用数值基准：圆形裂缝Sneddon解析解、Darcy-裂缝耦合解析解，以及三裂缝模拟（3 m/6 m spacing、Newtonian vs power‑law）得到的模拟结果。

**📈 对比分析**

通过与解析解的误差验证，并在云端8核多线程环境下测评，矩阵组装时间从单核到多核缩减约3倍；整体模拟保持交互性能；性能受矩阵组装支配，随着核心数提升并行效率下降。

**⚠️ 局限性**

仍受限于矩阵组装的O(N²)复杂度和内存带宽/锁争用导致并行效率下降；极端应力阴影下雅可比矩阵病态需切换到直接求解；仅支持单一岩石层、均匀介质，缺乏热力化学耦合及多尺度裂缝网络的实时多物理耦合。

---

## 311. What Semantics Survive the Connector? Diagnosing VLM-to-DiT Alignment in Video Editing

**arXiv ID:** 2605.20795 | [PDF](https://arxiv.org/pdf/2605.20795v1)

**作者:** Hangyu Lin `[一作]` (Hong Kong University Of Science And Technology), Yanwei Fu `[通讯]` (National University Of Singapore)

**通讯引用:** 16485 | [OpenAlex ID](https://openalex.org/A5084959430)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Trace-Edit这一控制性诊断数据集和评估协议，用于分离并量化VLM到DiT在视频编辑过程中的语义对齐瓶颈。

**💡 创新点**

创新点在于通过人工合成的单对象视频与VLM过滤实现精确的空间槽位、属性值和对象角色标注，并设计多维度诊断指标（几何重构、线性可解性、Token路由与注意力分布）来剖析对齐质量。

**🔧 技术方法**

利用VLM自动验证、MLP连接器、Meta-Query标记、线性探测器、CKA有效秩、特征方差及注意力分布等技术手段，对VLM-到-DiT的中间表示进行系统性分析。

**📊 数据集**

使用自建的Trace-Edit数据集，共5524个复合视频、11048条关系编辑实例，涵盖颜色、材质与动作三大属性类型，所有样本均通过VLM审验确保标注质量。

**📈 对比分析**

以四个代表性模型（UniVideo、Kiwi-Edit、Wan2.2、Wan2.1）为例，采用诊断指标评估连接器前后表示的语义可解性，发现细粒度槽位、目标值与角色信息显著退化，结构错误率达到40%–60%，通过对比表明连接器是主要瓶颈。

**⚠️ 局限性**

局限在于诊断仅基于合成数据，未能覆盖真实世界的复杂性；连接器设计缺乏对细粒度语义的完整保留，且不同DiT骨干对结果的影响尚未系统探究。

---

## 312. Interaction Locality in Hierarchical Recursive Reasoning

**arXiv ID:** 2605.20784 | [PDF](https://arxiv.org/pdf/2605.20784v1)

**作者:** Yosuke Miyanishi `[一作]` (CyberAgent Inc), Tetsuro Morimura `[通讯]` (CyberAgent Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并验证了一个名为 Interaction Locality 的框架，用来量化递归空间推理模型在不同几何任务（迷宫、数独、ARC‑AGI 以及 MTU3D 3D 场景）中的信息流是否局限于附近单元或语义段落。

**💡 创新点**

创新点在于将稀疏自编码器特征消融、有限噪声激活补丁、结构雅可比与注意力检查等多种探针统一到一个任务几何感知的可复现度量体系，并揭示高层状态在递归步骤内写入本地但跨周期传播全局的特征分布。

**🔧 技术方法**

采用的技术包括：稀疏自编码器（SAE）特征消融、有限噪声激活补丁（activation patching）、结构雅可比（Jacobian）与注意力矩阵分析，以及对 MTU3D 3D 场景进行对象级补丁实验。

**📊 数据集**

使用的数据集包括 Maze‑Hard、Sudoku Extreme、ARC‑AGI（二维任务）和 ScanNet（用于 MTU3D 的 3D 场景）。

**📈 对比分析**

通过在不同模型（HRM 与 TRM）的关键递归步骤进行补丁实验并与几何匹配的随机基线比较，结果显示高层写入在同一段落内更具局部性，且跨周期传播更为广泛；在 MTU3D 中，局部性显著存在于视觉到定位模块的切换处，而统一编码器内部则几乎无局部性。

**⚠️ 局限性**

局限性包括：仅分析现有检查点而非对齐精度的训练过程、几何邻域定义对不同任务可能不完整（如数独的行列约束、ARC‑AGI 的颜色边界、3D 的接触与视角关系），以及未对比不同架构在相同任务下的性能差异。

---

## 313. VBFDD-Agent for Electric Vehicle Battery Fault Detection and Diagnosis: Descriptive Text Modeling of Battery Digital Signals

**arXiv ID:** 2605.20742 | [PDF](https://arxiv.org/pdf/2605.20742v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 314. AttriStory: Fine-grained Attribute Realization for Visual Storytelling with Diffusion Models

**arXiv ID:** 2605.20777 | [PDF](https://arxiv.org/pdf/2605.20777v1)

**作者:** Manogna Sreenivas `[一作]` (Indian Institute of Science), Soma Biswas `[通讯]` (Indian Institute of Science)

**通讯引用:** 3018 | [OpenAlex ID](https://openalex.org/A5055091440)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了用于视觉故事生成的属性实现方法和对应基准，重点解决细粒度视觉属性的准确呈现。

**💡 创新点**

创新点在于：①创建了AttributionStory基准，系统化标注多场景多艺术风格的正负属性对；②提出了AttributioLoss——在扩散模型早期去噪阶段对交叉注意力图做IoU优化，正向强化正确属性-对象配对，负向抑制错误配对。

**🔧 技术方法**

技术手段包括：使用大型语言模型生成结构化故事和属性对；在Latent Diffusion模型（Stable Diffusion XL）中插入AttributioLoss，基于交叉注意力地图的IoU损失进行梯度更新；保持现有角色一致性机制不变，兼容多种故事生成管线。

**📊 数据集**

数据集为AttributionStory，包含200条多场景故事，覆盖10种艺术风格，每场景配2–5个属性-对象对，用ChatGPT自动生成并人工校验。

**📈 对比分析**

与Vanilla SDXL、StoryDiffusion和ConsiStory三种基线对比，在VQA-Score、CLIP‑T、CLIP‑I和DreamSim四项指标上均提升，特别是属性实现指标VQA-Score提升≈0.025–0.04，角色一致性CLIP‑I也未受影响，整体视觉质量DreamSim显著提高。

**⚠️ 局限性**

局限性在于：尽管属性实现大幅提升，但动作与场景动态仍可能与文本不完全匹配；部分极端或复杂属性组合仍存在误解，需进一步完善注意力约束和语义解析。

---

## 315. TERDNet: Transformer Encoder-Recurrent Decoder Network for Scene Change Detection

**arXiv ID:** 2605.20822 | [PDF](https://arxiv.org/pdf/2605.20822v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 316. Decomposing Subject-Driven Image Generation via Intermediate Structural Prediction

**arXiv ID:** 2605.20807 | [PDF](https://arxiv.org/pdf/2605.20807v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 317. Building Arabic NLP from the Ground Up: Twenty Years of Lessons, Failures, and Open Problems

**arXiv ID:** 2605.20786 | [PDF](https://arxiv.org/pdf/2605.20786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 318. Tunable MAGMAX: Preference-Aware Model Merging for Continual Learning

**arXiv ID:** 2605.20803 | [PDF](https://arxiv.org/pdf/2605.20803v1)

**作者:** Kei Hiroshima `[一作]` (Yokohama National University), Shinichi Shirakawa `[通讯]` (Yokohama National University)

**通讯引用:** 1688 | [OpenAlex ID](https://openalex.org/A5005268349)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Tunable MAGMAX模型合并框架，支持在连续学习中根据用户偏好构建合并模型；

**💡 创新点**

创新点在于引入偏好向量控制每个任务向量贡献的元素数量，实现任务性能可调，并自动构造偏好向量；

**🔧 技术方法**

利用任务向量、最大幅度选择（MAGMAX）、最优传输距离或标签分布相似度等技术；

**📊 数据集**

在CIFAR‑100和ImageNet‑R的类增量学习基准上进行实验；

**📈 对比分析**

与MAGMAX、随机混合、模型平均、TIES合并等方法对比，Tunable MAGMAX在各目标环境上均能达到或超过最优方法的平均精度；

**⚠️ 局限性**

限制在于需要目标环境的小量元数据以及未考虑任务间依赖性，且对大规模任务集合时OT方法表现下降。

---

## 319. Demo-JEPA: Joint-Embedding Predictive Architecture for One-shot Cross-Embodiment Imitation

**arXiv ID:** 2605.20811 | [PDF](https://arxiv.org/pdf/2605.20811v1)

**作者:** Jingyang He `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**通讯引用:** 11456 | [OpenAlex ID](https://openalex.org/A5013030532)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出 Demo-JEPA 框架，将跨胚体模仿学习转化为在共享的预测潜在空间中进行潜在目标规划，利用视觉演示推断目标可实现的未来潜在轨迹，随后目标机器人通过自学的动力学模型与 CEM 规划实现这些潜在子目标；

**💡 创新点**

创新点在于：①放弃低层动作对齐，直接从演示中提取目标意图的潜在目标；②通过“Dreamer Predictor”在 JEPA 共享潜在空间中实现跨胚体的未来状态翻译；③在潜在空间规划而非像素或动作空间，提升跨胚体泛化能力；

**🔧 技术方法**

主要技术包括：Joint Embedding Predictive Architecture (JEPA) 作为世界模型；Dreamer Predictor（跨胚体注意力 + Conv3D 融合 + Transformer 解码）用于生成潜在目标；Cross‑Entropy Method (CEM) 在潜在空间中进行动作规划；对潜在空间的时间扰动正则化与目标自适应更新；

**📊 数据集**

实验数据集主要使用 RLBench 仿真环境中的 Sawyer 机器人演示转给 Franka 机器人，以及真实世界中 UR5e 机器人演示转给 Franka 机器人，共六个真实抓取/操作任务；

**📈 对比分析**

与基线 VPP（视觉预测规划）和 XSkill（共享技能原型）比较。Demo-JEPA 在分布偏移更大的情形下（跨胚体桥接和零样本泛化）均显著优于基线，仿真零样本成功率约 36% 对比 VPP 4%，实测跨胚体桥接成功率约 55% 对比 VPP 约 35%；

**⚠️ 局限性**

主要限制包括：①当前的动作条件世界模型在高精度或复杂动力学任务上仍受限；②训练过程中仍需对时间或进度进行对齐或预处理，无法完全无监督学习；

---

## 320. ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing

**arXiv ID:** 2605.20802 | [PDF](https://arxiv.org/pdf/2605.20802v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 321. Hack-Verifiable Environments: Towards Evaluating Reward Hacking at Scale

**arXiv ID:** 2605.20744 | [PDF](https://arxiv.org/pdf/2605.20744v1)

**作者:** Amit Roth `[一作]` (Tel Aviv University), Yonathan Efroni `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可验证奖励黑客环境框架，将任意基准环境包装为文件系统环境，并植入可检测的奖励黑客点，允许对代理行为进行自动化且确定性的评估；同时在TextArena游戏上构建了21个游戏的可验证基准并公开代码；对多款前沿与开源语言模型进行了系统实验，报告黑客率和无黑客获胜率；

**💡 创新点**

创新点在于：①构建了通用的可验证环境包装器，使得奖励黑客行为可被确定性检测；②提出了四类通用黑客集合（隐藏答案、逻辑漏洞、提示读取、提示编辑）；③在大规模游戏集上实现了自动化黑客检测与度量；④为模型评估提供了基准与排行榜，揭示模型间黑客差异。

**🔧 技术方法**

技术手段包括：环境包装器（wrapper）实现文件系统接口；定义黑客检测函数h(观测,动作)；采用Gymnasium API与TextArena游戏集；使用语言模型推理生成动作；计算黑客率(HR)与无黑客胜率(HF-WR)等指标。

**📊 数据集**

使用的数据集为TextArena文本游戏集合，包含21个游戏；每个游戏提供不同难度层级，并植入四类黑客；同时对比多款模型（gpt-5.4, claude-sonnet-4.6, gemini-3.1-pro, qwen3.6-plus等）。

**📈 对比分析**

实验通过在每个模型-游戏组合上执行多条三局轨迹，记录轨迹级黑客率和游戏级无黑客胜率；结果显示gpt-5.4与claude-sonnet-4.6在低黑客率与高胜率上处于Pareto前沿；其他模型表现差异明显，说明黑客倾向受模型与环境类型共同影响。

**⚠️ 局限性**

局限性包括：①逻辑漏洞需要针对每个环境单独实现；②假设基准环境无预先存在的错误，复杂环境适用性有限；③难以判定黑客行为是否出于意图，可能因好奇或低能力导致误检；④对非常大规模或多模态环境的扩展尚未验证。

---

## 322. Causal Machine Learning Is Not a Panacea: A Roadmap for Observational Causal Inference in Health

**arXiv ID:** 2605.20782 | [PDF](https://arxiv.org/pdf/2605.20782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 323. Refining and Reusing Annotation Guidelines for LLM Annotation

**arXiv ID:** 2605.20809 | [PDF](https://arxiv.org/pdf/2605.20809v1)

**作者:** Kon Woo Kim `[一作]` (Graduate University for Advanced Studies SOKENDAI), Akiko Aizawa `[通讯]` (Graduate University for Advanced Studies SOKENDAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

系统化利用和迭代改进现有文本标注准则，以指导LLM在生物医学实体识别任务中生成符合金标准的注释，提出了迭代审查框架。

**💡 创新点**

首次证明了准则整合、推理型LLM以及低监督审查循环能提升标注质量，并展示了基于准则的LLM自我改进过程。

**🔧 技术方法**

采用大语言模型（GPT‑5、Gemini、DeepSeek）进行注释与评估，构建模式解释–原则生成–准则修订三步审查流程，并在增量迭代中逐步改进准则。

**📊 数据集**

使用 NCBI Disease、BC5CDR（Disease 与 Chemical）和 BioRED 三个生物医学实体识别数据集，其中训练集为 10 篇文档，评估集为 100 篇文档。

**📈 对比分析**

与单提示、准则提示以及审查迭代三种策略对比；所有模型在准则提示下均提升，推理模型优于非推理，审查迭代可实现 0.01–0.03 的 F1 增益，统计检验显示部分显著性。

**⚠️ 局限性**

需要完整且细化的人类编写准则，实验仅基于极小的训练样本，准则改进可能引入副作用且缺乏标准化质量检查，低监督环境下迭代收敛不稳定。

---

## 324. Findings of the Counter Turing Test: AI-Generated Text Detection

**arXiv ID:** 2605.20761 | [PDF](https://arxiv.org/pdf/2605.20761v1)

**作者:** Rajarshi Roy `[一作]` (Kalyani Government Engineering College), Aman Chadha `[通讯]` (Amazon AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对AI生成文本的二分类与模型归因任务进行全面分析与实验，提出多种检测方法并在Defactify 4.0共享任务中进行评估。

**💡 创新点**

创新性地结合Transformer fine-tuning、重写编辑距离与多维特征融合，形成混合检测框架，显著提升模型归因性能。

**🔧 技术方法**

使用Fine-tuned DeBERTa、BART、XGBoost特征分类、重写编辑距离等技术。

**📊 数据集**

使用包含50,000条样本的多域多模型数据集，包括人类文本与Gemma-2-9、Mistral-7B、Qwen-2-72B、LLaMA-8B、Yi-Large、GPT-4o生成文本。

**📈 对比分析**

与基线相比，最高二分类F1达1.0000，模型归因宏F1最高为0.9531，显示归因任务更具挑战性。

**⚠️ 局限性**

仍缺乏跨模型泛化与对抗鲁棒性评估，数据集仅覆盖六款LLM，且归因性能相对较低。

---

## 325. Findings of the Counter Turing Test: AI-Generated Image Detection

**arXiv ID:** 2605.20787 | [PDF](https://arxiv.org/pdf/2605.20787v1)

**作者:** Rajarshi Roy `[一作]` (Kalyani Government Engineering College), Aman Chadha `[通讯]` (Amazon AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文报告了 Defactify 4.0 研讨会中的 Counter Turing Test（CT2）共享任务，提出两项挑战：二分类检测图像是否为 AI 合成以及识别合成所用的具体生成模型。

**💡 创新点**

创新点在于：①构建了包含 5 种主流生成模型（Stable Diffusion 3/XL/2.1、DALL‑E 3、Midjourney 6）与 MS COCO 实景图像的 50K 图像数据集 MS COCOAI；②将频域特征与多模态学习相结合，为模型指纹识别提供新的参考。

**🔧 技术方法**

使用的技术包括卷积神经网络（EfficientNet‑B0、ResNet、Swin Transformer）、视觉 Transformer（ViT、CLIP‑ViT）、频域变换（二维傅里叶）、对比学习、伪标签数据扩增、注意力融合以及多模态跨任务学习。

**📊 数据集**

数据集为自研的 MS COCOAI，包含 50,000 张来自 Stable Diffusion、DALL‑E 3 与 Midjourney 的合成图像，配备真实 MS COCO 图像、生成模型标注以及对抗性后处理样本。

**📈 对比分析**

评估指标为 F1‑score：二分类任务 Task A 最高分 0.8334，模型识别任务 Task B 最高分 0.4986，表明合成图像检测已达到 83% 以上的准确率，而模型指纹识别仍显挑战；参与系统多采用频域+多模态特征，效果明显优于仅用空间特征的基线。

**⚠️ 局限性**

局限性包括：①模型识别准确率低，说明指纹提取仍不成熟；②对抗性鲁棒性不足，后处理操作能显著降低检测效果；③缺乏实时检测机制，现有方法主要在离线环境下评估；④实验仅覆盖 5 种生成模型，泛化到未来新模型的能力未知。

---

## 326. STAR-IOD: Scale-decoupled Topology Alignment with Pseudo-label Refinement for Remote Sensing Incremental Object Detection

**arXiv ID:** 2605.20738 | [PDF](https://arxiv.org/pdf/2605.20738v1)

**作者:** Yaoteng Zhang `[一作]` (Northwestern Polytechnical University), Qi Wang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 18259 | [OpenAlex ID](https://openalex.org/A5100341321)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对遥感增量目标检测中的灾难性遗忘和多尺度变异问题，提出了 STAR-IOD 框架。

**💡 创新点**

创新点在于引入子空间解耦的拓扑蒸馏（STD）以处理尺度差异，以及基于 K‑Means 的自适应伪标签生成（CPG）以弥补缺失标注。

**🔧 技术方法**

所用技术包括尺度自适应实例划分、拓扑关系对齐、响应层蒸馏、聚类驱动阈值推断以及伪标签去重等。

**📊 数据集**

实验基于新构建的 DIOR‑IOD 与 DOTA‑IOD 两个遥感增量检测基准数据集进行。

**📈 对比分析**

与多种 SOTA 方法对比，STAR‑IOD 在 DIOR‑IOD 上提升 1.7% mAP，在 DOTA‑IOD 上提升 2.1% mAP，显著减轻遗忘并保持对新类的检测性能。

**⚠️ 局限性**

局限性包括对语义相似类别（如飞机与直升机）的区分仍不充分，以及聚类阈值依赖候选池大小和噪声分布。

---

## 327. VSCD: Video-based Scene Change Detection in Unaligned Scenes

**arXiv ID:** 2605.20821 | [PDF](https://arxiv.org/pdf/2605.20821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 328. SpineContextResUNet: A Computationally Efficient Residual UNet for Spine CT Segmentation

**arXiv ID:** 2605.20760 | [PDF](https://arxiv.org/pdf/2605.20760v1)

**作者:** K S Nithurshen `[一作]` (Shiv Nadar University), Saurabh J. Shigwan `[通讯]` (Shiv Nadar University)

**通讯引用:** 43 | [OpenAlex ID](https://openalex.org/A5018870067)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计并实现了一种轻量级3D残差U-Net（SpineContextResUNet），用于在CT扫描中快速定位脊柱，适用于边缘设备。

**💡 创新点**

创新点在于引入并行多膨胀卷积的Context Block，兼顾大视野与低计算量，在严格硬件限制下仍能保持高Dice分数。

**🔧 技术方法**

采用3D残差网络、Atrous Spatial Pyramid Pooling (ASPP)、多尺度膨胀卷积、BCE+Dice混合损失、滑动窗口推理和高斯权重融合等技术。

**📊 数据集**

使用VerSe2020和CTSpine1K两个公开CT脊柱数据集进行训练和评估。

**📈 对比分析**

与3D U-Net、ResUNet、SwinUNETR以及TotalSegmentator进行对比，SpineContextResUNet在VerSe2020和CTSpine1K上分别达88.13%和88.17%的Dice分数，显著优于ResUNet（≈2.8%提升），并能在Intel i5、Jetson Orin Nano等边缘硬件上实现可行推理；相同参数规模下的Transformer表现明显差。

**⚠️ 局限性**

局限性包括：对极端解剖变异和低分辨率CT仍存在边界误差；滑动窗口推理导致颈部和骶部精度略低；仅完成二分类定位，未实现精细化椎体分类。

---

## 329. OSGNet with MLLM Reranking @ Ego4D Episodic Memory Challenge 2026

**arXiv ID:** 2605.20818 | [PDF](https://arxiv.org/pdf/2605.20818v1)

**作者:** Yisen Feng `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 29935 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

基于重排序框架解决Ego4D短时刻定位任务，结合传统定位模型生成候选片段并用多模态大型语言模型（MLLM）进行筛选，最终在NLQ和GoalStep两项挑战赛中夺冠。

**💡 创新点**

创新点在于：①将MLLM的强大视觉‑语言推理能力聚焦于有限候选集合，从而克服长视频上下文长度限制；②在GoalStep中加入序列先验约束，通过起始时间惩罚进一步提升选择质量。

**🔧 技术方法**

核心技术包括：OSGNet传统定位模型、GPT‑5.4多模态LLM、两阶段候选片段重排序、基于序列先验的后处理。

**📊 数据集**

使用的数据集为CVPR 2026年Ego4D Episodic Memory Challenge的自然语言查询（NLQ）和目标步骤（GoalStep）两条轨道。

**📈 对比分析**

与基线OSGNet相比，NLQ的R@1@IoU=0.3提升了0.15个百分点，GoalStep的R@1@IoU=0.3提升了0.08个百分点；在Leaderboard中均排名第一，显示整体性能显著提升。

**⚠️ 局限性**

局限性包括：①在NLQ中的提升有限，可能因错误负样本导致MLLM无法完全改善评估指标；②MLLM受输入图像数目与上下文长度限制，需分段处理；③评测协议与注释变更可能影响跨年份对比。

---

## 330. GraphRAG on Consumer Hardware: Benchmarking Local LLMs for Healthcare EHR Schema Retrieval

**arXiv ID:** 2605.20815 | [PDF](https://arxiv.org/pdf/2605.20815v1)

**作者:** Peter Fernandes `[一作]` (California Polytechnic State University), Ria Kanjilal `[通讯]` (California Polytechnic State University)

**通讯引用:** 80 | [OpenAlex ID](https://openalex.org/A5086238205)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在本地GPU上实现并系统评估了GraphRAG用于EHR（Epic Clarity）模式文档检索的完整管线。

**💡 创新点**

首次揭示不同参数规模的开源LLM在GraphRAG中的可靠性阈值、索引质量与答案质量的解耦，以及本地检索在低延迟、低幻觉方面优于全局检索的结论。

**🔧 技术方法**

使用Microsoft GraphRAG v2.3.0、Ollama部署的Llama 3.1、Mistral、Qwen 2.5和Phi‑4‑mini LLM，Leiden社区划分、向量检索和LLM生成的结构化JSON。

**📊 数据集**

基于Epic Clarity的10文件子集（共8份文档、141段文本）以及对应的HTML表结构描述。

**📈 对比分析**

通过比较索引时间、图实体/关系数、查询延迟、手工答案质量评分（1–5）以及幻觉检测来评估模型；Qwen 2.5在本地检索中取得最高答案质量（3.3/5）且延迟最低；Phi‑4‑mini在3.8B规模下失败，Mistral出现无限重复。

**⚠️ 局限性**

局限性包括仅使用10文件子集、主观手工评分、单一硬件配置，且未覆盖完整7,000文件规模或云端基准。

---

## 331. Assessing socio-economic climate impacts from text data

**arXiv ID:** 2605.20793 | [PDF](https://arxiv.org/pdf/2605.20793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 332. Instant GPU Efficiency Visibility at Fleet Scale

**arXiv ID:** 2605.20799 | [PDF](https://arxiv.org/pdf/2605.20799v1)

**作者:** Connor Pedersen `[一作]` (NVIDIA), Nik Konyuchenko `[通讯]` (NVIDIA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并验证一种基于GPU硬件计数器的全局FLOP利用率指标OFU，用于大型GPU集群的即时可视化与效率监控。

**💡 创新点**

以Tensor Pipe Activity和SM时钟频率为基础的精确、精度无关的MFU近似，经过多代GPU（H100、GB200）与多种精度（FP16、TF32、FP8、NVFP4）验证，且无需应用层instrumentation，提供可在大规模集群上连续监控的部署级指标。

**🔧 技术方法**

使用硬件性能计数器（DCGM、Nsight Systems/Compute）、first‑principles推导、tile量化校正、SM时钟采样误差分析、峰值FLOP计算、统计相关性评估以及Prometheus等监控工具。

**📊 数据集**

通过在H100和GB200上进行3,500+受控GEMM实验（多尺寸、随机维度）以及608个生产训练作业（Megatron‑LM等），不使用专门的机器学习数据集，仅以矩阵乘法为基准。

**📈 对比分析**

与应用级MFU对比，控制实验中校正后误差≤2%，生产作业相关系数r≈0.78；在大规模训练中识别出效率下降2.5×并验证混合精度对比，显示OFU在多精度环境下的准确性与可操作性。

**⚠️ 局限性**

只计数Tensor Core活动，忽略CUDA‑core计算；对小尺寸或不均衡矩阵的tile量化误差；SM时钟采样噪声需较长采样窗口；不捕获自定义内核或非矩阵乘法的浮点工作。

---

## 333. Diffuse to Detect: Bi-Level Sample Rebalancing with Pseudo-Label Diffusion for Point-Supervised Infrared Small-Target Detection

**arXiv ID:** 2605.20766 | [PDF](https://arxiv.org/pdf/2605.20766v1)

**作者:** Zhu Liu `[一作]` (Dalian University of Technology), Risheng Liu `[通讯]` (Dalian University of Technology)

**通讯引用:** 13909 | [OpenAlex ID](https://openalex.org/A5042370642)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于单点监督的双层双更新框架，利用热扩散物理模型与超像素先验将单点标签转化为可用伪掩码，并在训练中动态重新平衡样本权重。

**💡 创新点**

创新点在于：物理驱动的热扩散注释策略；双层优化同时更新检测器权重、样本权重和扩散参数；可微分扩散模块与元分类器实现自适应伪标签修正和样本再平衡；动态聚合方案提升优化稳定性。

**🔧 技术方法**

技术包括热扩散方程求解（离散迭代）、超像素分割、可微分伪标签生成、元学习预测样本权重、梯度外推的双层优化及动态聚合一阶近似。

**📊 数据集**

实验采用四个红外小目标检测数据集：SIRST3、SIRST‑v1、NUDT‑SIRST 和 IRSTD‑1k，按 6:2:2 进行训练/验证/测试划分。

**📈 对比分析**

在所有基准上相较于 LESPS、PAL、MCLC 等单点监督方法，均取得显著提升（IoU 最高可达 87% 以上、nIoU、P_d 提升 10‑20%），且仅使用 30% 训练数据时可与全监督相近；伪标签生成速度比 COM、MCLC 快约 5 倍。

**⚠️ 局限性**

局限性包括对热扩散参数的敏感性，仍需手动设定迭代次数/步长；在极低 SNR 或复杂背景下伪标签可能出现误分；以及对超像素先验的依赖可能在低分辨率图像上效果不佳。

---

## 334. The Hidden Signal of Verifier Strictness: Controlling and Improving Step-Wise Verification via Selective Latent Steering

**arXiv ID:** 2605.20745 | [PDF](https://arxiv.org/pdf/2605.20745v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 335. Resolving Long-Tail Ambiguity in Unsupervised 3D Point Cloud Segmentation with Language Priors

**arXiv ID:** 2605.20737 | [PDF](https://arxiv.org/pdf/2605.20737v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 336. PACD-Net: Pseudo-Augmented Contrastive Distillation for Glycemic Control Estimation from SMBG

**arXiv ID:** 2605.20751 | [PDF](https://arxiv.org/pdf/2605.20751v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 337. ShapeBench: A Scalable Benchmark and Diagnostic Suite for Standardized Evaluation in Aerodynamic Shape Optimization

**arXiv ID:** 2605.20763 | [PDF](https://arxiv.org/pdf/2605.20763v1)

**作者:** Shaghayegh Fazliani `[一作]` (Stanford University), Madeleine Udell `[通讯]` (Stanford University)

**通讯引用:** 2060 | [OpenAlex ID](https://openalex.org/A5084564811)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出并发布了ShapeBench，一个包含103个跨领域气动形状优化（ASO）任务的统一基准和诊断套件，支持多类形状、多目标、多点、混合变量等，并为每个任务提供快速代理模型和可选的高精度CFD验证；

**💡 创新点**

创新点包括：①构建跨领域、多任务的统一基准；②每任务配套低费代代理与高费代验证，支持fidelity gap分析；③诊断套件用于检测代理剥削与物理可信度；④推出专为ASO设计的LLM驱动优化器ShapeEvolve；⑤公开完整实验流程与可重现性，方便方法开发与部署。

**🔧 技术方法**

使用的技术包括代理模型（如COCOANet、Transolver等）、CFD高精度模拟、梯度基优化（adjoint、L‑BFGS‑B）、进化算法（PSO、CMA‑ES）、贝叶斯优化、LLM驱动的优化框架（OpenEvolve、ShinkaEvolve、ShapeEvolve），以及诊断与LLM评判流程。

**📊 数据集**

使用的主要数据集为公开的ShapeBench任务集合、3,570个预计算CFD仿真数据（用于训练代理）、CCA任务数据集，以及通过GitHub和HuggingFace发布的相关数据。

**📈 对比分析**

通过统一的评估预算（评估次数）比较经典与LLM优化器，结果表明不同任务和目标下方法排名差异巨大，平均Spearman ρ≈0.013，单任务结论不具可泛化性；在CCA任务中，ShapeEvolve显著优于传统方法；其他基准在不同形状类别表现不一。

**⚠️ 局限性**

局限性包括：高费代验证缺失（如Passenger Car）；代理模型易被剥削导致物理可信度低；诊断依赖LLM评估，可能受限于语言模型的可靠性；缺少在线CFD评估器，无法实时验证；未来需扩展高精度验证、制造可行性、鲁棒性与不确定性控制。

---

## 338. Rethinking Fraud Safety Evaluation: Multi-Round Attacks Reveal Safety-Utility Tradeoffs in Graph-Context LLM Defenders

**arXiv ID:** 2605.20759 | [PDF](https://arxiv.org/pdf/2605.20759v1)

**作者:** Laura Jiang `[一作]` (Curtin University), Nasim Ferdosian `[通讯]` (Curtin University)

**通讯引用:** 140 | [OpenAlex ID](https://openalex.org/A5087546771)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

评估基于图上下文的欺诈防御模型在多轮攻击中的安全性，重点关注拒绝时机和benign误拒。

**💡 创新点**

将安全评估从单轮转向多轮，对拒绝时机和benign误拒进行测量，定位安全-效用权衡在LLM对结构上下文的消耗上。

**🔧 技术方法**

利用图神经网络生成风险评分并序列化为JSON上下文，结合大型语言模型进行prompt驱动决策；采用冻结的Fraud‑R1评测套件与多种攻击策略。

**📊 数据集**

Fraud‑R1（修复后的256×20样本，含20个测试案例），并在更大规模512×40进行一致性检查。

**📈 对比分析**

通过配对置换检验和Bootstrap置信区间，对比文本仅、静态图、时序图三种上下文，时序图在AUSR和早期拒绝上显著优于文本，但benign误拒显著增加。

**⚠️ 局限性**

单元规模有限、攻击者模型为基准而非真实人类、仅测试prompt序列化整合方式、对非Qwen模型的定位分析不足、benign误拒高导致部署难度大。

---

## 339. Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards

**arXiv ID:** 2605.20758 | [PDF](https://arxiv.org/pdf/2605.20758v1)

**作者:** Xuehui Yu `[一作]` (National University of Singapore), Harold Soh `[通讯]` (National University of Singapore)

**通讯引用:** 2643 | [OpenAlex ID](https://openalex.org/A5066073375)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出冲突感知增量指导(g^car)，用于流模型在组合奖励下的推理时间对齐，解决离散流场漂移问题。

**💡 创新点**

通过冲突感知门控机制主动检测梯度冲突并学习修正向量，实现轻量级近似指导与精确指导之间的平衡。

**🔧 技术方法**

基于流匹配/条件流匹配、推理时增量指导、梯度误差分析、终端值回归学习、冲突门控等技术。

**📊 数据集**

实验使用2D混合高斯、Maze2D、ManiSkill2机器人任务、CelebA‑HQ图像编辑等合成与真实数据集。

**📈 对比分析**

与g^cov‑G、GLASS‑FKS、GM、MPPI、FlowGrad等基线对比，在漂移率降低、成功率提升、计算开销低等指标上均表现更优。

**⚠️ 局限性**

在复杂奖励景观下收敛困难，CLIP奖励非平滑导致训练不稳定，对极端冲突仍有限制。

---

## 340. Distribution-Aware Reward: Reinforcement Learning over Predictive Distributions for LLM Regression

**arXiv ID:** 2605.20740 | [PDF](https://arxiv.org/pdf/2605.20740v1)

**作者:** Jungsoo Park `[一作]` (Georgia Institute of Technology), Alan Ritter `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 10264 | [OpenAlex ID](https://openalex.org/A5039096905)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种分布感知奖励（Distribution-Aware Reward）用于强化学习训练大型语言模型，使其在回归任务中优化预测分布而非单点预测。

**💡 创新点**

创新点在于将连续分布评分规则 CRPS 与留一法信用分配相结合，直接为每个采样结果分配奖励，从而鼓励模型产生既准确又适当分散的预测分布。

**🔧 技术方法**

采用基于策略梯度的强化学习框架（如GRPO），使用 CRPS 作为分布质量评估，利用留一法计算每个回合的边际贡献，并在 Qwen 系列 LLM 上实现。

**📊 数据集**

在三类数据集上进行评估：控制的高斯混合合成回归任务、代码性能预测（KBSS、APPS）和 MoleculeNet 分子属性预测（FreeSolv、ESOL、Lipophilicity）。

**📈 对比分析**

与 SFT、点对点 MSE 奖励、价值头回归以及开源 LLM 的零样本推断进行比较；实验显示在 Spearman 相关性、RMSE、MAE 等指标上均优于基线，尤其在分布感知奖励下显著提升排序与不确定性估计。

**⚠️ 局限性**

局限性包括对模型规模和计算资源要求较高，未深入探讨不同温度或采样规模对分布质量的影响，以及在极端不确定性场景下的鲁棒性尚待进一步验证。

---

## 341. ParaCell: Paravirtualized Secure Containers with Lightweight Intra-Container Isolation and Intent-Driven Memory Management

**arXiv ID:** 2605.20906 | [PDF](https://arxiv.org/pdf/2605.20906v1)

**作者:** Yiyang Wu `[一作]` (Shanghai Jiao Tong University), Haibo Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 7715 | [OpenAlex ID](https://openalex.org/A5100406215)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于MPK的轻量级隔离与内存管理接口的 paravirtualized secure container 运行时，实现在容器内用户与内核之间的高效切换与细粒度内存弹性；

**💡 创新点**

创新点包括：①利用MPK实现容器内部用户/内核隔离，消除地址空间切换；②通过向宿主暴露内核分配/释放事件，主动绑定/解绑GPA–HPA映射，避免二次故障；③将上述技术集成为可替换PVM的完整方案；

**🔧 技术方法**

使用技术包括 MPK（Memory Protection Keys）、Paravirtualization、内核分配 Hook（alloc/free）、Shadow Page Table、PCP 列表批处理、Linux 6.7.0、QEMU/KVM 以及 Systrap；

**📊 数据集**

使用的数据集与基准包括 SQLite benchmark、PARSEC、vmitosis、SWE‑bench（Claude Code）以及 SkillsBench，辅以典型云工作负载；

**📈 对比分析**

通过与 PVM、gVisor、HyperAlloc、VirtIO‑Balloon、VirtIO‑Mem 等基线在裸机和嵌套环境下进行对比，采用 LMbench、页面故障延迟、进程管理、内存回收吞吐量和端到端延迟等指标，结果显示相较于 PVM 可降低最高57%/79% 延迟，内存回收吞吐量提升 10–26 倍，agent 工作负载内存占用仅 0.4% 左右，显著优于 HyperAlloc 的 35.6%；

**⚠️ 局限性**

主要局限包括：MPK 容易被控制流劫持绕过，需要二级保护；与 KVM/QEMU 集成导致 Shadow 页表和元数据开销；不兼容已使用 MPK 的应用；在高并发/多核场景下仍有宿主–客栈交互开销；依赖特定 Linux 版本与硬件支持。

---

## 342. NeighborDiv: Training-free Zero-shot Generalist Graph Anomaly Detection via Neighbor Diversity

**arXiv ID:** 2605.20879 | [PDF](https://arxiv.org/pdf/2605.20879v1)

**作者:** Kaifeng Wei `[一作]` (Netease Yidun AI Lab), Yuke Li `[通讯]` (Netease Yidun AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个无需训练的图异常检测框架NeighborDiv，基于邻居间多样性来判别异常节点；

**💡 创新点**

核心创新是从传统的节点-邻居一致性转向邻居-邻居多样性范式，利用邻居相似度方差作为独立、二阶的异常信号；

**🔧 技术方法**

使用低维SVD投影、基于余弦相似度的邻居对方差计算、采样近似、全图中位数校准以及z-score标准化；

**📊 数据集**

在七个目标图（Cora、YelpChi、Reddit、T‑Finance、Tolokers、Disney、Questions）和四个源图（Facebook、Amazon、PubMed、Elliptic）上进行评估；

**📈 对比分析**

与多种无监督、监督和现有零样本GGAD方法对比，NeighborDiv在单域独立训练和统一多域训练两种协议下均取得SOTA，平均AUC提升约10.25%/6.89%，AP提升约17.78%/9.58%，且无性能波动；

**⚠️ 局限性**

局限包括仅处理一跳邻居、多样性仅为全局中位数校准、对极端稠密图的可扩展性有限、以及未覆盖动态图、异构图和边级异常检测等场景。

---

## 343. PlexRL: Cluster-Level Orchestration of Serviceized LLM Execution for RLVR

**arXiv ID:** 2605.20863 | [PDF](https://arxiv.org/pdf/2605.20863v1)

**作者:** Yiqi Zhang `[一作]` (National University of Singapore), Siyuan Feng `[通讯]` (Shanghai Innovation Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种集群级别的 RLVR 作业多路复用框架 PlexRL，解耦 RL 控制与模型执行，并通过统一服务化的 LLM 推理与训练来填补作业间的空闲时间。

**💡 创新点**

创新点在于：①将多阶段 RLVR 作业的模型调用抽象为远程服务，①实现跨作业的时分复用；②设计基于时空资源视图与微移位的放置策略；③构建集群级别的模型状态管理器以实现高效的 GPU/CPU/NVMe 内存层级迁移和状态预取。

**🔧 技术方法**

主要技术包括：分布式 LLM 推理/训练服务（支持 Megatron、FSDP 等后端）、集群级 Scheduler（基于环形缓冲、段树剪枝、区间集合 fitting）、HRRS（考虑切换成本的优先级调度）、模型状态管理器（GPU、主机、NVMe 层级存储与 canonicalized offload）以及基于 Tensor‑parallel、数据‑parallel 的多模型并行架构。

**📊 数据集**

使用内部数学任务数据集（约 45,000 条 AIME‑级别样本）以及公开的 Qwen2.5‑7B‑Instruct、Qwen3‑30B‑A3B‑Thinking‑2507、Qwen3‑235B‑A22B‑Instruct‑2507 三种规模的 LLM 进行实验。

**📈 对比分析**

与 colocated、split‑async 等传统 RLVR 部署方案对比；评估指标为 GPU‑小时/训练步、推理吞吐率与占用率；实验表明在两作业打包情况下，PlexRL 将 GPU‑小时成本分别降低 31.36%、30.10% 与 37.58%，并在大型模型下保持与基线相同的奖励曲线。

**⚠️ 局限性**

限制主要体现在：①仍需手动设置作业的周期性需求和资源阈值，②对极端大模型（>百亿参数）时的状态迁移开销不完全消除；③在高并发场景下，Scheduler 与 StateManager 的网络通信可能成为瓶颈。

---

## 344. SEABAD: A Tropical Bird Activity Detection Dataset for Passive Acoustic Monitoring

**arXiv ID:** 2605.20853 | [PDF](https://arxiv.org/pdf/2605.20853v1)

**作者:** Muhammad Mun'im Ahmad Zabidi `[一作]` (Universiti Malaya), Norisma Idris `[通讯]` (Universiti Malaya)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SEABAD，这是一个面向东南亚热带环境的鸟类活动检测数据集，包含5万条3秒单声道录音，平均覆盖1,677种鸟类，正负样本均衡。

**💡 创新点**

创新点在于：①双分支自动化策划管道，实现从社区录音（Xeno‑Canto）到负样本多源集合的完整流程；②利用声学相似度+FAISS进行去重；③多级多聚类的多样性意识平衡，显著降低类不平衡；④为边缘设备提供标准化16kHz单声道格式。

**🔧 技术方法**

技术手段包括：RMS能量提取与窗口分段、基于梅尔谱嵌入的近似最近邻去重、声学显著性评分、MiniBatch K‑Means聚类与分层抽样、以及在TensorFlow下训练MobileNetV3‑Small等轻量CNN。

**📊 数据集**

使用的数据集为：正样本来自Xeno‑Canto的5,0000条录音；负样本取自BirdVox‑DCASE‑20k、Freefield1010、Warblr、FSC‑22、ESC‑50、DataSEC，共计25,000条。

**📈 对比分析**

对比实验显示，MobileNetV3‑Small在SEABAD测试集上达99.57%准确率、0.9985 AUC，四大CNN模型均突破99.4%准确率；而BirdNET零样本在同一数据集仅为68.6%准确率，说明迁移领域难题。

**⚠️ 局限性**

局限性包括：①仅覆盖东南亚，难以直接推广到其他热带或温带区域；②负样本主要来自非热带数据，可能缺乏当地雨季、蝉鸣等典型噪声；③数据依赖社区录音，存在空间与时间偏差；④仅针对二分类检测，未覆盖物种识别任务。

---

## 345. MemGym: a Long-Horizon Memory Environment for LLM Agents

**arXiv ID:** 2605.20833 | [PDF](https://arxiv.org/pdf/2605.20833v1)

**作者:** Wujiang Xu `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 MemGym benchmark，统一了五条 agentic 路径，并给出 memory‑isolated 评价框架，生成可控 synthetic pipeline 并训练轻量级奖励模型 MemRM 以替代 Docker rollouts。

**💡 创新点**

创新点包括：① 将 memory 与 agentgy 环境统一到一个 memory contract 接口；② 在多任务中采用 memory‑isolated 评分解耦 reasoning 与 memory；③ 设计长度可控、ablation 验证的 synthetic 生成管道；④ 用 Qwen3‑1.7B + QLoRA 训练的 MemRM 作为快速评估门。

**🔧 技术方法**

技术主要涉及：memory manager、per‑step contract、retrieval‑style 与 LLM summarization、synthetic pipeline with distractor & fictionalization、replay‑and‑fork harness、以及 QLoRA 微调的 reward 模型。

**📊 数据集**

使用数据集包括 SWE‑Gym/SWE‑bench 编码任务、τ²‑bench 对话、WebArena‑Infinity 网页操作、SWE‑smith 缺陷库、学术检索后端（arXiv、Semantic Scholar、OpenAlex、Wikipedia）以及由 synthetic pipeline 生成的 670 条 deep‑research 与 1,194 条 coding‑QA 实例。

**📈 对比分析**

通过在同一 reasoner 下对 baseline 与 memory 进行配对对比，评估 memory gain；实验表明 memory 对对话与网页任务提升显著（≈8–10% 成功率），对编码任务无明显收益；在 synthetic 轴上，A‑Mem 在最高压力点均领先；MemRM 在 IID 上 AUROC 0.985，且对部分 OOD 仍具可用性。

**⚠️ 局限性**

局限性：评估仍需高昂的算力；memory 策略与 reasoner 的耦合使完全解耦困难；门控覆盖率有限，OOB 泛化受限；仅在已覆盖子集内可部署；对不同任务域的通用性尚待验证。

---

## 346. Forward asymmetric numeral systems coding for natural language text compression

**arXiv ID:** 2605.20826 | [PDF](https://arxiv.org/pdf/2605.20826v1)

**作者:** Mykyta Kharin `[一作]` (Taras Shevchenko National University of Kyiv), Igor Zavadskyi `[通讯]` (Taras Shevchenko National University of Kyiv)

**通讯引用:** 57 | [OpenAlex ID](https://openalex.org/A5051852254)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于前向自适应建模(FAM)的自适应ANS压缩方法，改进了传统tANS编码流程。

**💡 创新点**

创新点在于：①在编码时递减符号频率、解码时递增频率，避免频率表的显式传输；②使用词汇最后出现顺序对字典进行排序，使编码表与文本顺序一致，从而提升压缩率；③通过反向解码兼容自适应频率更新。

**🔧 技术方法**

核心技术包括：异构数值系统(ANS)、tANS实现、前向自适应建模(FAM)、词级标记化、7z/PPMd压缩字典、C++实现。

**📊 数据集**

实验使用了Silesia语料库中的Dickens和Webster、Calgary语料库中的book1、book2、以及Canterbury语料库中的alice29和asyoulik。

**📈 对比分析**

方法与rANS、均匀置换(tANS)以及反向置换等传统ANS方案在同一文本上进行比较。实验结果表明，该FAM-tANS在代码体积上明显优于其它方案，速度与rANS相当，均匀置换速度慢。总体上压缩率提升显著。

**⚠️ 局限性**

局限性包括：仅在词级自然语言文本上验证，未测试非文本或大字典场景；需要先完整读取文本以构建字典，导致一次性内存需求；对极大词典的构建与调度仍可能带来时间/空间开销。

---

## 347. HDMoE: A Hierarchical Decoupling-Fusion Mixture-of-Experts Framework for Multimodal Cancer Survival Prediction

**arXiv ID:** 2605.20891 | [PDF](https://arxiv.org/pdf/2605.20891v1)

**作者:** Huayi Wang `[一作]` (Zhejiang University), Jian Wu `[通讯]` (Zhejiang University)

**通讯引用:** 19361 | [OpenAlex ID](https://openalex.org/A5081897442)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种层级解耦融合的稀疏专家混合网络（HDMoE）用于多模态癌症生存预测，融合病理切片图像和基因组特征。

**💡 创新点**

创新点包括：①在每层引入共享专家和路由专家，先去除同模态冗余并提取细粒度特征；②引入随机特征重组（RFR）模块，在每层融合时打乱特征组合，增强细粒度跨模态关系学习；③采用距离度量和负载平衡损失提升专家分配与特征解耦效果。

**🔧 技术方法**

采用稀疏Mixture-of-Experts（Sparse MoE）、随机特征重组（Random Feature Reorganization）、转移学习预训练的ResNet50、TransMIL层、距离度量（余弦相似度）、负载平衡正则化等技术。

**📊 数据集**

使用了私有肝癌（LC）数据集（160例配对MRI+WSI）以及TCGA公开的BLCA、BRCA、LUAD四个癌症数据集（分别含WSI和基因组信息）。

**📈 对比分析**

与多种单模态与多模态基线（SNN、CLAM、TransMIL、MoME等）进行5折交叉验证对比，HDMoE在LC上C‑Index 0.683、BLCA 0.694、BRCA 0.686、LUAD 0.675，均超过现有最佳多模态方法（平均提升约1.5%），并通过Kaplan‑Meier和T‑检验验证风险分层显著。

**⚠️ 局限性**

局限性：①要求配对完整的多模态数据，缺少单模态或缺失模态时无法直接使用；②数据规模有限，未进行多中心验证，泛化性待进一步评估。

---

## 348. Distance between Road Networks: A Macroscopic Method for Road Network Datasets Comparison Using Traffic-weighted Geographic Distribution

**arXiv ID:** 2605.20921 | [PDF](https://arxiv.org/pdf/2605.20921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 349. Finite-Time Regret Analysis of Retry-Aware Bandits

**arXiv ID:** 2605.20854 | [PDF](https://arxiv.org/pdf/2605.20854v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 350. JFAA: Technical Report for the EPIC-KITCHENS-100 Action Anticipation Challenge at EgoVis 2026

**arXiv ID:** 2605.20904 | [PDF](https://arxiv.org/pdf/2605.20904v1)

**作者:** Qiaohui Chu `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 29935 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于JEPA的未来动作预判方法JFAA，利用冻结的V-JEPA 2.1模型提取上下文特征和近未来潜在标记，并通过轻量化注意力探测器分别预测动词、名词和动作。

**💡 创新点**

创新点在于将冻结的V-JEPA 2.1编码器-预测器与可学习查询的轻量化注意力探测器相结合，构建了近未来潜在特征的融合与领域感知的集成推理策略。

**🔧 技术方法**

使用的技术包括V-JEPA 2.1 ViT‑G/384的冻结编码器与预测器、基于多头注意力的探测器、Sigmoid Focal损失、随机扰动的预期时间、以及多头学习率/权重衰减的探测器头与领域感知集成。

**📊 数据集**

所用数据集为EPIC‑KITCHENS‑100（EK‑100）动作预判任务的官方训练、验证与隐藏测试集。

**📈 对比分析**

与基线与其它参赛方案对比，JFAA在官方MT5R评测中取得27.95的总分，排名第一，并在动词、名词、未见参与者和尾部类别上均显示出显著优势。

**⚠️ 局限性**

局限性包括对多重相似物体的名词判别仍有误差、对完全遮挡或极短视野的动作推断能力有限，以及依赖冻结的高阶模型可能限制进一步改进。

---

## 351. Task-Routed Mixture-of-Experts with Cognitive Appraisal for Implicit Sentiment Analysis

**arXiv ID:** 2605.20916 | [PDF](https://arxiv.org/pdf/2605.20916v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 352. HyDAR-Pano3D: A Hybrid Disentangled Anatomical Recovery Framework for Panoramic-to-3D Reconstruction

**arXiv ID:** 2605.20827 | [PDF](https://arxiv.org/pdf/2605.20827v1)

**作者:** Yaoyao Yue `[一作]`, Jinman Kim `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种两阶段的HyDAR-Pano3D框架，用双编码器结合基于基础模型的语义先验进行解剖归一化体积重建，再通过结构化变形场恢复个体化三维解剖；

**💡 创新点**

创新点在于将深度不确定性的三维重建与形状变形解耦，使用双流编码器与Adaptive Cross‑Attention Fusion融合语义先验，显著减少解剖边界模糊；

**🔧 技术方法**

采用双编码器（3DPX + SemiT‑SAM）、Adaptive Cross‑Attention Fusion、Anatomical Restoration Network（基于稀疏控制格网的结构化变形场）等深度学习技术；

**📊 数据集**

在三大公开 CBCT 数据集（NC、ST、TF）上训练与评估，数据统一裁剪为 128×128×256 并通过曲线引导生成合成 panoramic；

**📈 对比分析**

与 UNETR、Oral‑3D、3DPX 等基线比较，HyDAR‑Pano3D 在 PSNR、SSIM、Dice 等指标上分别提升约 5–10 dB、~6%、~6%（TF 数据集），并在下游牙齿与下颌神经管分割任务中取得最高 Dice（牙齿 82.4% / 76.4%，IAC 72.2%），性能显著优于对比方法；

**⚠️ 局限性**

主要限制包括：依赖人工标注的弓形曲线用于生成归一化体积；合成 panoramic 与真实扫描存在域差距；极端解剖异常或病理变形可能被平滑处理，导致局部细节缺失。

---

## 353. Sutra: Tensor-Op RNNs as a Compilation Target for Vector Symbolic Architectures

**arXiv ID:** 2605.20919 | [PDF](https://arxiv.org/pdf/2605.20919v1)

**作者:** Emma Leonhart `[一作]` `[通讯]`, Emma Leonhart

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

无法获取论文内容，无法进行总结

**💡 创新点**

无法确定

**🔧 技术方法**

无法确定

**📊 数据集**

无法确定

**📈 对比分析**

无法确定

**⚠️ 局限性**

无法确定

---

## 354. USV: Towards Understanding the User-generated Short-form Videos

**arXiv ID:** 2605.20838 | [PDF](https://arxiv.org/pdf/2605.20838v1)

**作者:** Haoyue Cheng `[一作]` (Nanjing University), Limin Wang `[通讯]` (Nanjing University)

**通讯引用:** 22857 | [OpenAlex ID](https://openalex.org/A5100436505)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了USV-1.0大规模用户生成短视频数据集，并定义了主题识别与视频‑文本检索两项任务。

**💡 创新点**

创新点包括：①使用查询词自动标注和标题弱监督构建多模态数据；②提出MMF‑Net多模态融合网络和VTCL视频‑文本对比学习框架；③提供首个面向短视频的多模态基准与实验。

**🔧 技术方法**

采用的技术包括2D/3D卷积网络（TSN、I3D、SlowFast等）、BERT+MLP文本编码、EasyOCR字幕提取、log‑Mel spectrogram音频特征、InfoNCE对比学习及多模态特征融合。

**📊 数据集**

使用的数据集为USV‑1.0，约224K视频，212主题，含视觉、音频、字幕和标题。

**📈 对比分析**

在主题识别任务中，MMF‑Net实现78.84% mca；在视频‑文本检索任务中，V+A+T组合的Recall@1达23.51%，相较于单模态基线显著提升。

**⚠️ 局限性**

局限性包括：①预训练模型对领域差异影响较大；②自动标注导致标签噪声较多；③仅聚焦短视频平台，跨域适用性尚待验证。

---

## 355. For How Long Should We Be Punching? Learning Action Duration in Fighting Games

**arXiv ID:** 2605.20911 | [PDF](https://arxiv.org/pdf/2605.20911v1)

**作者:** Hoang Hai Nguyen `[一作]` (Maastricht University), Dennis J. N. J. Soemers `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究在街机格斗游戏中，让RL智能体同时学习动作与执行时长（frame skip），以提升对抗脚本AI的表现

**💡 创新点**

将动作空间扩展到同时包含动作类型和时长，允许代理自适应决策频率

**🔧 技术方法**

使用PPO算法、分离或联合策略头的神经网络，以及FightLadder框架

**📊 数据集**

Street Fighter II – Special Champion Edition，使用FightLadder环境与内置脚本AI对战

**📈 对比分析**

通过对比固定、随机和自适应frame‑skip策略的胜率与奖励，发现自适应策略可与高固定值相媲美，最高可达100%胜率

**⚠️ 局限性**

自适应代理倾向高frame‑skip、缺乏多样化动作且对不同对手泛化能力差，且对内置AI的可被利用性敏感

---

## 356. Creating Robust and Fair Graph Structures for Connectivity and Clustering

**arXiv ID:** 2605.20897 | [PDF](https://arxiv.org/pdf/2605.20897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 357. FlowLong: Inference-time Long Video Generation via Manifold-constrained Tweedie Matching

**arXiv ID:** 2605.20910 | [PDF](https://arxiv.org/pdf/2605.20910v1)

**作者:** Jangho Park `[一作]` (KAIST), Jong Chul Ye `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个训练‑free、模型无关的长视频生成框架FlowLong，利用滑动窗口和重叠区域融合实现视频时序延伸。

**💡 创新点**

核心创新是Tweedie匹配通过在重叠帧的预测干净样本空间插值来保持流形约束与时序一致，以及在采样早期注入随机噪声同步各窗口轨迹的“Stochastic Early‑Phase Sampling”。

**🔧 技术方法**

基于流匹配（flow matching）和逆问题求解思路，结合流式ODE/SDE采样、Tweedie公式和多窗口并行采样。

**📊 数据集**

在MovieGen Bench和SceneBench进行30s/60s文本到视频评估，并用LTX‑2、Wan 2.1等预训练模型进行音视频联合和3DGS生成实验。

**📈 对比分析**

与RIFLEx、UltraViCo等双向模型以及CausVid、Self‑Forcing、LongLive等自回归模型对比，FlowLong在VBench的多维度评分中显著提升，尤其在动态度、连贯性和视觉质量方面优于基线。

**⚠️ 局限性**

局部重叠一致性约束可能不足以保证极长视频的全局语义一致，且对窗口尺寸与重叠比例的敏感性仍需进一步研究。

---

## 358. Winfree Oscillatory Neural Network

**arXiv ID:** 2605.20922 | [PDF](https://arxiv.org/pdf/2605.20922v1)

**作者:** Jiawen Dai `[一作]` (Shanghai Jiao Tong University), Yue Song `[通讯]` (Shanghai Qi Zhi Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Winfree振荡神经网络（WONN），通过在高维圆形相空间上迭代同步动力学来进行计算。

**💡 创新点**

创新点在于将通用Winfree同步动力学与可学习的相互作用函数结合，并通过分组层次化同步实现可扩展、高参数效率的动态神经架构。

**🔧 技术方法**

采用Winfree动力学、分组相互作用、可学习或三角函数的灵敏度与影响函数、双相频率状态、能量函数用于推理等技术。

**📊 数据集**

使用CIFAR-10/100、ImageNet-100/1K进行视觉分类，Maze-hard迷宫路径规划和数独推理作为逻辑推理任务。

**📈 对比分析**

与ResNet、ViT、AKOrN等基线相比，WONN在相同或更少参数下取得或超越竞争者的精度，在ImageNet-1K与Maze-hard上表现突出。

**⚠️ 局限性**

局限在于对超大规模模型的收敛性与训练稳定性尚未完全验证，且对更复杂的推理任务的可推广性仍待进一步研究。

---

## 359. FruitEnsemble: MLLM-Guided Arbitration for Heterogeneous ensemble in Fine-Grained Fruit Recognition

**arXiv ID:** 2605.20892 | [PDF](https://arxiv.org/pdf/2605.20892v1)

**作者:** Enhui Yu `[一作]` (University of Science and Technology Liaoning), Youshan Zhang `[通讯]` (Chuzhou University)

**通讯引用:** 1100 | [OpenAlex ID](https://openalex.org/A5079460371)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了包含306个品种、116,233张图像的长尾多模态数据集Fruit-306，并提出了双阶段动态推理框架FruitEnsemble，先用异构视觉骨干构成加权集成生成Top‑3候选，再在置信度低于阈值时触发多模态大型语言模型进行Chain‑of‑Thought推理。

**💡 创新点**

创新点在于：①验证校准的加权异构集成与置信度间隙路由的双阶段决策；②将LLM限定在Top‑K候选且配合专家文本描述的可解释性推理；③针对困难样本的Hard Sample‑Aware Joint Loss以增强多样性；④高效的适配与鲁棒训练策略，使LLM仅在约15%样本中使用。

**🔧 技术方法**

采用ResNet‑50、DenseNet‑201、EfficientNet‑B7与Vision‑Transformer四种异构骨干，利用熵‑加权聚合与置信度阈值路由；引入Qwen‑VL‑Plus作为多模态LLM进行CoT推理；使用焦点损失处理长尾；实现可并行推理、动态LLM调用与EMA‑正则化的鲁棒训练。

**📊 数据集**

Fruit‑306数据集：306个果品种，116,233张多样化自然场景图像，伴随每类专家编写的形态描述文本，划分70/10/20的长尾训练/验证/测试集。

**📈 对比分析**

与14种单模型及静态集成相比，FruitEnsemble在验证集/测试集上Top‑1准确率提升至70.49%（比单模型高约5%），Top‑5准确率92.5%，平均推理时延仅19.8 ms，LLM调用率约15%，显著兼顾准确性与实时性。

**⚠️ 局限性**

局限性包括：LLM仍为计算瓶颈，需依赖昂贵的模型；在极低频类别的置信度仍可能不足；框架针对果品种任务，跨域（如蔬菜、农作物）推广需进一步验证。

---

## 360. Training distribution determines the ceiling of drug-blind cancer sensitivity prediction

**arXiv ID:** 2605.20885 | [PDF](https://arxiv.org/pdf/2605.20885v1)

**作者:** Taekyung Heo `[一作]` `[通讯]`, Taekyung Heo

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了药物盲预测中药物表征与训练分布对预测性能的影响，并提出通过MoA分层训练和响应匹配两种策略提升药效预测。

**💡 创新点**

首次将药物盲预测的性能分解为全局与每药物相关的Pearson r，揭示训练分布异质性为主要瓶颈，并证明MoA分层训练能显著提高每药物r。

**🔧 技术方法**

采用岭回归和大型Transformer编码器，结合药物结构、LINCS签名、药物靶点向量和MoA类别等特征，并实现响应匹配与MoA分层训练。

**📊 数据集**

使用GDSC2、CTRPv2、BeatAML、PRISM等四个独立药物敏感性数据集进行评估。

**📈 对比分析**

在药物盲交叉验证下比较全局和每药物Pearson r，发现传统药物表示几乎无提升；通过MoA分层训练和响应匹配可将每药物r从约0.1提升至0.3-0.4，显著优于基线。

**⚠️ 局限性**

实验仅基于细胞系IC50，忽略体内药代动力学、肿瘤微环境和组合用药；MoA分层训练仅适用于机制统一的药物类别，需要足够的功能相似药物来匹配；缺乏临床前向验证。

---

## 361. SynCB: A Synergy Concept-Based Model with Dynamic Routing Between Concepts and Complementary Neural Branches

**arXiv ID:** 2605.20908 | [PDF](https://arxiv.org/pdf/2605.20908v1)

**作者:** Tores Julie `[一作]` (Université Côte d’Azur, CNRS, Inria, I3S), Precioso Frédéric `[通讯]` (Université Côte d’Azur, CNRS, Inria, I3S)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SynCB 框架，将概念基模型与端到端神经网络通过共享特征提取器和可训练路由模块联合训练，并引入基于样本不确定度的 USI 干预策略。

**💡 创新点**

创新点包括：①联合训练概念分支与神经分支共享 backbone；②学习路由模块动态决定每个样本使用哪个分支；③设计针对干预的 USI 策略；④加入干预损失提升模型对人类干预的响应。

**🔧 技术方法**

采用概念瓶颈模型（CBM/CEM）、Mixture of Experts 路由网络、交叉熵和二元交叉熵损失、联合训练策略、干预损失、Uncertainty Sample Intervention（USI）干预策略。

**📊 数据集**

使用 CUB、AWA、CIFAR-10 以及其不完整版本 CUB Inc、AWA Inc 共五个基准数据集。

**📈 对比分析**

与 CBM、CEM、ProbCBM、HP-CBM、MixCEM、DNN baseline 等方法对比。SynCB 在五个基准上均实现了最高任务准确率，平均提升约 3.9pp；在干预响应上优于 MixCEM，差距可达 6.43pp；同时保持了较好的概念检测性能。

**⚠️ 局限性**

局限性：当概念数量较大时，USI 的干预成本高；路由模块需要在训练期间让两条分支都参与，导致训练更耗时；在概念集合不完整或极度不完整的情况下表现仍需进一步验证；实际人机交互成本与可操作性尚未完全评估。

---

## 362. CIG: Exploration via Conditional Information Gain

**arXiv ID:** 2605.20878 | [PDF](https://arxiv.org/pdf/2605.20878v1)

**作者:** Tim Joseph `[一作]` (FZI), J. Marius Zöllner `[通讯]` (KIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了可扩展的Conditional Information Gain（CIG）奖励，用于模型基RL的短期想象轨迹，以更好地平衡终身与回合内信息收益。

**💡 创新点**

通过对轨迹信息增益的log‑determinant近似并引入trace‑reduction与Cholesky分解，实现了同时考虑回放缓冲和前缀上下文的可计算奖励。

**🔧 技术方法**

使用深度集成预测器、Gaussian近似、矩阵行列式与Cholesky分解，以及DreamerV2框架中的actor‑critic训练。

**📊 数据集**

在MiniGrid与OGBench（包括清洁版与带随机噪声的Noisy‑TV变体）共12个任务上进行评估。

**📈 对比分析**

与Plan2Explore、RND、ICM、APT、E3B及其组合等基线比较，CIG在所有任务中获得最高的归一化IQM，显著优于其它方法并在噪声场景下保持鲁棒性。

**⚠️ 局限性**

在部分简单任务如Scene、Puzzle 3×3的纯覆盖任务中表现相似，且对较长回合或模型无关环境的适用性未验证，trace‑reduction会丢失方向信息。

---

## 363. Mobile UMI: Cross-View Diffusion Policy with Decoupled Kinematics for Mobile Manipulation

**arXiv ID:** 2605.20894 | [PDF](https://arxiv.org/pdf/2605.20894v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 364. RelWitness: Open-Vocabulary 3D Scene Graph Generation with Visual-Geometric Relation Witnesses

**arXiv ID:** 2605.20823 | [PDF](https://arxiv.org/pdf/2605.20823v1)

**作者:** Minh Anh Nguyen `[一作]` (Phenikaa University), Sui Yang Guang `[通讯]` (Phenikaa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出基于视觉‑几何证据的关系见证机制，用于在不完整标注下的开放词汇3D场景图生成。

**💡 创新点**

创新点在于将“关系见证”定义为可观测的物理线索，结合正负未标记学习，避免虚假关系并提升未见谓词召回。

**🔧 技术方法**

采用RGB‑D多视角、深度、重建3D几何、角色一致性、空视图测试及多视角一致性等视觉几何检测，并使用正负未标记学习、记忆池与可解释的关系解码。

**📊 数据集**

在3DSSG/3RScan和ScanNet的开放词汇拆分上进行实验。

**📈 对比分析**

与SceneGraphFusion、Open3DSG、ConceptGraphs、Text/对象先验补全等基线比较，显著提升未见谓词mR、见证精度，减少幻觉与冗余，验证方法有效性。

**⚠️ 局限性**

局限在于对遮挡或低质量重建导致的几何噪声敏感；非可观测的功能或社交关系仍保持不确定，且依赖高质量RGB‑D与精确标注。

---

## 365. VISTA: Technical Report for the Ego4D Short-Term Object Interaction Anticipation at EgoVis 2026

**arXiv ID:** 2605.20901 | [PDF](https://arxiv.org/pdf/2605.20901v1)

**作者:** Qiaohui Chu `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 29935 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了VISTA，一种结合静态目标检测与冻结V-JEPA视频上下文的短期目标交互预测模型。

**💡 创新点**

首次在StillFast框架中引入冻结的V-JEPA 2.1 ViT‑G作为短程时间上下文模块，并通过FiLM特征调制与ROI级融合将时间信息注入检测分支，同时采用多头预测与集成推理提升整体匹配性能。

**🔧 技术方法**

采用COCO预训练的Faster R‑CNN ResNet‑50 FPN检测分支、冻结的V‑JEPA 2.1 ViT‑G编码器、FiLM调制、ROI上下文MLP、多头预测（框、名词、动词、接触时间、置信度）、集成推理及NMS等技术。

**📊 数据集**

使用Ego4D短期目标交互（STA）数据集，包含第一人称视频及交互注释。

**📈 对比分析**

在官方测试集上与StillFast基线V2、Faster R‑CNN+SlowFast等进行对比，VISTA在Overall Top‑5 mAP上获得5.40，排名第一，Noun+Verb mAP从13.29提升至16.15，整体表现明显优于基线。

**⚠️ 局限性**

对小型或被遮挡目标的预测仍易失误，易受视觉干扰物影响，且模型对时间窗口长度和更复杂场景的泛化能力仍有限。

---

## 366. Map-Mono-Ego: Map-Grounded Global Human Pose Estimation from Monocular Egocentric Video

**arXiv ID:** 2605.20889 | [PDF](https://arxiv.org/pdf/2605.20889v1)

**作者:** Hiroyuki Deguchi `[一作]`, Hideo Saito `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种仅使用单目摄像头即可实现漂移抑制的轨迹跟踪和全局一致人类姿态估计的框架。

**💡 创新点**

创新点在于：①利用合成数据库实现精准定位；②通过基于内点的过滤进行轨迹细化；③结合扩散模型进行人类姿态估计；④首次在单目头戴摄像头设置下实现全局一致姿态和轨迹，同时在实验中验证了方法的有效性。

**🔧 技术方法**

使用技术包括：HLoc（ALIKED+LightGlue+NetVLAD）用于定位；DROID-SLAM+GravityNet/HeadNet进行基线估计；Diffusion-based pose模型（修改后接受颈部关节轨迹）；基于内点比例的轨迹细化；数据预处理与相机标定。

**📊 数据集**

使用的数据集为作者自采集的室内点云（FARO Focus激光扫描）、头戴摄像机的自我摄像视频（1920×1080）以及通过Theia3D、DhaibaWorks和Soma得到的SMPL-X模型的地面真实运动数据。

**📈 对比分析**

方法通过与基线（DROID-SLAM+GravityNet/HeadNet）在相同场景下对比，显示出在全球位置和姿态精度上均优于基线，尤其在轨迹漂移和姿态误差（MPJPE）方面取得显著改进，但在某些场景仍出现不自然的表面穿透。

**⚠️ 局限性**

局限性在于未显式施加物理约束，导致人机交互时出现表面穿透或接触不足的问题。未来工作计划引入场景感知优化（如SDF碰撞避免损失）以提升物理可行性。

---

## 367. PlanningBench: Generating Scalable and Verifiable Planning Data for Evaluating and Training Large Language Models

**arXiv ID:** 2605.20873 | [PDF](https://arxiv.org/pdf/2605.20873v1)

**作者:** Ziliang Zhao `[一作]` (Renmin University of China), Pluto Zhou `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立 PlanningBench 框架，生成可验证的可扩展规划数据，用于评估和训练 LLM 的规划能力。

**💡 创新点**

将真实规划场景抽象为任务与约束的层次化分类，采用约束驱动的合成流程和闭环难度调节，实现可控难度、多样性和自动验证；强调确定性最优解以提升奖励信号。

**🔧 技术方法**

约束驱动合成、生成-响应-评估闭环、GRPO 强化学习、自动验证列表、难度控制算法、对齐检验。

**📊 数据集**

PlanningBench 生成的 467 个评估实例和 300 个训练实例；对比公开基准 TravelPlanner、ChinaTravel、Multi-Challenge、Inverse IFEval、Collie 等数据集。

**📈 对比分析**

在 18 种 LLM（包括 GPT‑5.4‑xhigh、Gemini、DeepSeek 等）上使用 All-pass 与 Avg-pass 评估，最佳模型 All-pass 63%，Avg-pass 92%；在 GRPO 训练后模型在外部规划基准上平均 All-pass 提升 7–15%，在通用指令跟随基准上提升约 7%。

**⚠️ 局限性**

仍难以完全满足全局约束，错误主要集中在计算/分配错误；验证基于静态检查，缺乏动态约束；数据规模有限，需进一步扩展；对低资源 LLM 的提升有限。

---

## 368. Runtime-Certified Bounded-Error Quantized Attention

**arXiv ID:** 2605.20868 | [PDF](https://arxiv.org/pdf/2605.20868v1)

**作者:** Dean Calver `[一作]` `[通讯]` (Independent Researcher), Dean Calver (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种分层 KV 缓存架构，配合运行时可证实的注意力计算与多级回退，实现在 LLaMA 3.1‑8B 上以 INT8/INT4 KV 缓存压缩保持 FP16 级别质量。

**💡 创新点**

实现了每头每步的误差上界分解（键量化误差与值重构误差），并提供在线监测与自适应精度选择以及确定性回退到完整 FP16 计算的多级回退 ladder。

**🔧 技术方法**

采用通道级 INT8 键量化、组级 INT4 值量化、块级错误注释、在线误差评估、分层存储 (VRAM + CPU RAM) 与融合式压缩域注意力核，以及排名一致性检查与阈值驱动的自适应提升。

**📊 数据集**

在 PG‑19、NIAH、RULER 三大基准上评测，涵盖语言建模、检索与结构化推理任务，最大上下文长度达 128K。

**📈 对比分析**

与全 FP16 dense baseline 对比，PG‑19 perplexity 差值 < 0.001，NIAH 检索准确率无显著差异，RULER 价值敏感子任务在 8K/32K 下降约 0.07/0.02pp；整体落后仅 0.01pp；在 64K/128K 上保持与 dense 同等质量，吞吐量约为 dense 的 3.5–4.5 倍，内存占用下降 44%（VRAM）。

**⚠️ 局限性**

误差上界仅局部（每头每步），不保证全局模型一致性；值重构误差在短上下文和价值敏感任务中累积导致失效；回退机制依赖系统 RAM 与 PCIe 传输，排名一致性检查为瓶颈，未实现端到端质量保证。

---

## 369. DISC: Decoupling Instruction from State-Conditioned Control via Policy Generation

**arXiv ID:** 2605.20856 | [PDF](https://arxiv.org/pdf/2605.20856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 370. Multi-Step Likelihood-Ratio Correction for Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2605.20865 | [PDF](https://arxiv.org/pdf/2605.20865v1)

**作者:** Deokgyu Yoon `[一作]` (Seoul National University), Min-hwan Oh `[通讯]` (Seoul National University)

**通讯引用:** 73159 | [OpenAlex ID](https://openalex.org/A5100447410)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了N步前向轨迹（N‑step forward trace）以及基于该轨迹的NFPO算法，用以在强化学习可验证奖励（RLVR）框架下改进大语言模型的推理性能。

**💡 创新点**

创新点在于将PPO等局部代理目标与完整策略梯度之间建立连续可调的桥梁：通过截断未来令牌的似然比积，N决定偏差与方差平衡，并给出理论上的收敛下界与方差分析。

**🔧 技术方法**

核心技术包括：前向轨迹（N‑step forward trace）正则化、masked policy gradient（MPG）与总变差（TV）令牌掩码、比例剪辑、奖励优势估计以及基于梯度的策略更新。

**📊 数据集**

使用了数学推理基准集：MATH、DAPO‑Math‑17k、AIME24/25/26、AMC23、MATH 500、Minerva和OlympiadBench，模型为Qwen3‑1.7B与Qwen3‑8B。

**📈 对比分析**

与GRPO、DPPO等现有RLVR方法对比，NFPO在pass@1平均指标上提高约10–20个百分点，显著优于基线，尤其在大模型和长响应场景中表现突出。

**⚠️ 局限性**

局限性包括：需手动调节N值；随着N增大方差上升，可能导致训练不稳定；目前仅在数学推理任务验证，缺乏跨任务通用性与更细粒度的动态分析。

---

## 371. ArchSIBench: Benchmarking the Architectural Spatial Intelligence of Vision-Language Models

**arXiv ID:** 2605.20837 | [PDF](https://arxiv.org/pdf/2605.20837v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 372. SmoCap: Unified Scale-Pose Canonicalization with Proxy-Mapped Trust-Region QP

**arXiv ID:** 2605.20850 | [PDF](https://arxiv.org/pdf/2605.20850v1)

**作者:** Shihao Li `[一作]` (University of Tokyo), Naohiko Sugita `[通讯]` (University of Tokyo)

**通讯引用:** 5298 | [OpenAlex ID](https://openalex.org/A5028247446)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了SmoCap，一种在同一信赖域二次规划中联合估计模型尺度和姿势的运动规范化框架。

**💡 创新点**

创新点在于：①通过统一的局部信赖域QP避免尺度–姿势泄漏；②使用低维代理映射实现弱可观结构的协调控制；③可选的预求解器为极端配置提供热启动；④在大规模数据集上实现子毫秒级运行速度。

**🔧 技术方法**

采用信赖域二次规划、代理空间约束、MuJoCo kinematics、Gauss‑Seidel预求解以及可选的软参考项。

**📊 数据集**

使用了CAMK‑Knee（膝关节荧光标定）、Riglet（身体测量）和MoYo（极端瑜伽姿势）三大公开数据集。

**📈 对比分析**

与基准（OpenSim风格分段尺度+IK）比较时，SmoCap在膝关节角度RMSE 2.9°、标记RMSE 18.8 mm、外部形态误差约7.2 mm，平均每帧0.25–0.33 ms、2–3次迭代，且漏泄率显著降低。

**⚠️ 局限性**

局限性包括：仅基于皮肤标记且假设标记固定；未考虑动力学或肌肉-张力尺度；脊柱验证仅基于粗糙的平滑度指标，缺乏影像或统计真值。

---

## 373. Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models

**arXiv ID:** 2605.20839 | [PDF](https://arxiv.org/pdf/2605.20839v1)

**作者:** Jeffrey Wang `[一作]` (University of Wisconsin--Madison), Grigorios G. Chrysos `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了不需要激活函数的多项式模块 PolyMLP、PolyConv、PolyAttn，并将其集成进 MetaFormer 风格的视觉骨干 PolyNeXt，实现激活自由的图像识别模型。

**💡 创新点**

创新点在于证明激活函数可被 Hadamard 乘积替代，提出轻量级稳定化策略（Sigmoid-Scale、多输入跳连）以及深度优先设计，使得多项式网络在不同规模上可匹配或超过传统激活模型，并提供可兼容全同态加密的全多项式版本。

**🔧 技术方法**

使用了多项式激活替代技术、Hadamard 乘积、PolyMLP/PolyConv/PolyAttn 三大模块、Sigmoid-Scale 归一化、层归一化/批归一化、MetaFormer 框架、深度优先结构、正则化（dropout、stochastic depth）等多项技术。

**📊 数据集**

在 ImageNet-1K 分类、ADE20K 语义分割以及 ImageNet-C/A/R/Sketch 鲁棒性评估数据集上进行实验。

**📈 对比分析**

与 ConvFormer、CAFormer、MONet、DTTN、StarNet 等基线在相同参数、FLOPs、分辨率下对比，PolyNeXt 在各规模上实现或超越激活模型的 Top-1 准确率；在鲁棒性和分割任务中亦优于基线，全多项式 BN 版本在 ImageNet 上达到 82.7% 的精度。

**⚠️ 局限性**

局限性包括：训练需要较小批量、强正则化和特定初始化；深度优先导致吞吐量低；多项式乘积对学习率敏感；全同态加密部署仍待实现；与标准模型的迁移需要更细致的调优。

---

## 374. Markovian Circuit Tracing for Transformer State Dynamic

**arXiv ID:** 2605.20824 | [PDF](https://arxiv.org/pdf/2605.20824v1)

**作者:** Abdullah X `[一作]` `[通讯]` (Project AWARE and Zephara AI), Abdullah X (Project AWARE and Zephara AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出Markovian Circuit Tracing（MCT）诊断框架，用于检验Transformer在隐藏马尔可夫模型（HMM）任务中是否具备粗粒度的状态转移结构。

**💡 创新点**

创新点在于：①将内部状态抽象、转移估计、马尔可夫阶数检验和状态强制干预整合为一套可操作的诊断流程；②在拥有完整Ground‑Truth（真实状态、贝叶斯信念、转移矩阵、最优预测及强制状态下的对照目标）的合成HMM基准上验证Transformer的状态动态解释性。

**🔧 技术方法**

使用的技术包括：Transformer模型训练（两层因果Transformer），残差聚类（K‑means）做状态抽象，线性Ridge探针回归信念向量，行向量KL与Frobenius误差评估转移矩阵，负对数似然比较不同阶数的马尔可夫模型，以及激活补丁（state forcing）干预。

**📊 数据集**

使用六种合成HMM族（包括易分离、模糊发射、粘性、近均匀、高熵、三状态与六状态）以及每族三条随机种子，共18个实验任务，数据规模为64长序列、6000训练样本、1500验证样本。

**📈 对比分析**

与基线的比较：模型平均验证损失仅比Bayes最优多0.0138，证明学习效果优秀；残差聚类与PCA+K‑means在转移矩阵和贝叶斯信念的KL上均优于随机投影，且在易分离与粘性族上显著捕获一阶转移信号；状态强制干预中，恢复中心点补丁将KL从0.196降至0.053，显著优于错误状态、平均激活、随机激活、洗牌标签和真实状态中心点等对照，显示抽象状态具有因果效用。

**⚠️ 局限性**

局限性包括：任务为合成小规模，模型参数仅约27万；状态抽象方法仅为简单残差聚类，未探究更高级的稀疏或概率抽象；实验未验证自然语言或更复杂部分可观测环境中的状态结构；未能证明完整的因果抽象匹配HMM的真实隐状态。

---

## 375. Enhancing Scientific Discourse: Machine Translation for the Scientific Domain

**arXiv ID:** 2605.20912 | [PDF](https://arxiv.org/pdf/2605.20912v1)

**作者:** Dimitris Roussis `[一作]` (Institute for Speech and Language Processing), Stelios Piperidis `[通讯]` (Institute for Speech and Language Processing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了3个语言对（ES‑EN、FR‑EN、PT‑EN）的科学领域并行与单语语料库，并基于OPUS‑MT模型进行领域适配微调，以提升四个科研子领域的机器翻译质量。

**💡 创新点**

首次大规模抓取62个学术仓库，产生11.7M句对，并将通用科学文本与专门域文本相结合进行多域微调，展示了在多学科场景下翻译性能的显著提升。

**🔧 技术方法**

使用了LASER对齐、margin scoring、Transformer‑big 预训练模型、MarianMT微调框架、SacreBLEU/chrF2++/COMET评估指标，并结合BPE子词分割技术。

**📊 数据集**

采集并过滤自学术标题/摘要得到的并行语料（ES‑EN、FR‑EN、PT‑EN）以及相应的单语语料，按通用科学与Cancer、Energy、Neuroscience、Transportation四个子领域划分，构成训练与评测数据集。

**📈 对比分析**

通过与基线OPUS‑MT和Google Translate进行BLEU、chrF2++、COMET对比，发现微调后模型平均BLEU提升约2.4分、COMET提升约1.6分；单域+通用文本进一步提升约0.9/0.8分，表明多域增补有效。

**⚠️ 局限性**

局限包括数据主要来源于拉美和欧洲学术仓库，导致跨语种覆盖度不均；仅采用单向微调未探讨多域或back‑translation等更先进方法；在高资源语言对上仍难与Google Translate的表现相媲美。

---

## 376. RISE: Reliable Improvement in Self-Evolving Vision-Language Models

**arXiv ID:** 2605.20914 | [PDF](https://arxiv.org/pdf/2605.20914v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 377. SubTGraph: Large-Scale Subterranean Environment Synthesis with Controllable Topological Variability for Robotic Autonomy Validation

**arXiv ID:** 2605.20917 | [PDF](https://arxiv.org/pdf/2605.20917v1)

**作者:** F. Labra Caso `[一作]`, G. Nikolakopoulos `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SubTGraph地下环境生成框架，实现可控的地形、纹理与拓扑生成，并公开150个地下世界数据集。

**💡 创新点**

创新点在于使用结构约束与路徑描述符结合Dijkstra生成多层地下拓扑，支持线性、抛物线、正弦三种路线，提供可调节的拓扑复杂度。

**🔧 技术方法**

采用基于网格的图算法、Dijkstra最短路、路线偏移描述符、三维网格关联与Gazebo物理仿真。

**📊 数据集**

使用了DARPA SubT提供的三角网格资产与自定义纹理，生成了150个包含矿山、天然洞穴和熔岩管道的地下环境。

**📈 对比分析**

通过多机器人路径规划、语义分割与LIO SLAM三大案例验证方法，在Gazebo仿真中与FastLIO/DLIO对比，显示在垂直通道与回溯等角落情况下LIO表现差异。

**⚠️ 局限性**

局限在于资产重复导致特征识别错误、网格离散化限制细节、缺乏连续数字孪生的高保真度。

---

## 378. Evaluating Speech Articulation Synthesis with Articulatory Phoneme Recognition

**arXiv ID:** 2605.20920 | [PDF](https://arxiv.org/pdf/2605.20920v1)

**作者:** Vinicius Ribeiro `[一作]` (Université de Lorraine), Yves Laprie `[通讯]` (Université de Lorraine)

**通讯引用:** 953 | [OpenAlex ID](https://openalex.org/A5076654217)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出用语音识别（语音标识率）来评估基于音素序列的声道形态合成的质量，并比较三种声道合成模型。

**💡 创新点**

创新点在于：①将音素识别作为客观评价指标，替代传统几何距离或声学特征；②通过加入发声编码（voicing）提升识别准确性；③用识别误差直观衡量合成声道形态的语音信息保留程度。

**🔧 技术方法**

使用卷积残差网络与循环网络的混合架构（DeepSpeech‑2 类），配合适配器将 500 维声道形态特征映射为 80 维；训练目标为 CTC 损失，评估指标为语音错误率（PER）；同时利用 t‑SNE 可视化特征嵌入。

**📊 数据集**

数据集为单位女性法语说话者 2.5 小时的实时 MRI（RT‑MRI）录音，包含 10 个声道器官的轨迹与对应音素标注；还使用了三种合成器（平均轮廓、无模型自由生成、基于自动编码器生成）的合成声道特征。

**📈 对比分析**

通过 PER 对比：无声编码时 acoustic 23.30，真声道 23.65；加声编码后 acoustic 21.66，真声道 21.66，模型自由 20.59，自动编码 31.69；平均轮廓 43.18。结果表明：无模型自由生成在 PER 上优于其他两种合成器，自动编码器的表现逊色；识别误差可有效区分模型并反映声道形态的音素保留程度。

**⚠️ 局限性**

局限性包括：仅使用单一说话者的数据，难以推广至多说话者；合成声道缺乏声源信息导致声学识别仍受限；模型的评价仍依赖于预先训练好的识别器，若识别器性能不足，评估可信度下降。

---

## 379. Calibration vs Decision Making: Revisiting the Reliability Paradox in Unlearned Language Models

**arXiv ID:** 2605.20915 | [PDF](https://arxiv.org/pdf/2605.20915v1)

**作者:** Divyaksh Shukla `[一作]` (Indian Institute of Technology Kanpur), Ashutosh Modi `[通讯]` (Indian Institute of Technology Kanpur)

**通讯引用:** 1294 | [OpenAlex ID](https://openalex.org/A5076043215)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

无法完成总结，缺少论文内容。

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 380. GenAI-Driven Threat Detection with Microsoft Security Copilot

**arXiv ID:** 2605.20896 | [PDF](https://arxiv.org/pdf/2605.20896v1)

**作者:** Scott Freitas `[一作]` (Microsoft Security Research), Amir Gharib `[通讯]` (Microsoft Security Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一种名为的自适应智能代理，能够在Microsoft Defender平台上持续构建事件时间线、自动化推理并生成新的可解释警报，以填补现有检测逻辑中的漏洞。

**💡 创新点**

主要创新点包括：① 将告警、事件、用户行为分析和威胁情报统一为动态时间线；② 采用版本化LLM提示契约，配合schema校验和容错机制确保生成内容可靠；③ 设计规划‑执行循环的自主调查流程，实现攻击假设生成与证据收集；④ 在调查完成后动态生成具有标题、严重度、MITRE映射、修复建议等信息的警报。

**🔧 技术方法**

所用技术包括：大语言模型（GPT‑4.1、GPT‑5.4）与LLM提示契约；PySpark分布式检索与聚合；实体关联与时间窗展开；行为异常与威胁情报信号融合；Planner‑Executor 规划执行循环；动态警报生成与schema验证。

**📊 数据集**

数据来源主要为微软Defender的海量日志与告警，包括事件、UEBA、威胁情报等；线上评估使用12个生产区域的客户反馈（共1,088条警报级评估），离线评估使用10个随机复杂威胁事件样本，覆盖勒索软件、商务邮件劫持等多阶段攻击。

**📈 对比分析**

与基线行级分类器比较，使用GPT‑5.4比GPT‑4.1提升宏F1 0.12、微F1 0.11；系统在线上评估中获得80.1%微精度（宏精度78.2%），并在15%的受检事件中生成新警报。离线评估宏F1达0.78，基线仅0.52，提升0.26。单事件平均完成时间28分钟，Token成本2.04美元，作业级失败率0.38%。

**⚠️ 局限性**

限制包括：评估聚焦于缺口检测而非与商业检测系统直接对比；离线实验样本有限，无法覆盖所有攻击形态；系统不支持对抗性适配，攻击者可能通过改造日志误导模型；生成警报的质量仍需人工复核。

---

## 381. Learning fMRI activations dictionaries across individual geometries via optimal transport

**arXiv ID:** 2605.20883 | [PDF](https://arxiv.org/pdf/2605.20883v1)

**作者:** Sonia Mazelet `[一作]` (Ecole Polytechnique), Bertrand Thirion `[通讯]` (Inria-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种新的字典学习框架——Amortized Graph Dictionary Learning (AGDL)，利用基于最优传输的 Fused Gromov‑Wasserstein (FGW) 距离，并结合预训练的 ULOT 网络来预测 OT 计划，从而在大规模 fMRI 图数据上实现字典学习，直接保留个体脑几何信息；

**💡 创新点**

创新点包括：① 用加速的 FGW（通过 ULOT）降低复杂度，使得包含约千节点的 fMRI 图可行；② 学习字典随 FGW 参数 α 变化的模型（线性插值或 MLP），无需多次训练即可得到不同 α 下的字典；③ 在原始个体几何上进行字典学习，保留个体几何差异，突破传统统一模板的限制；

**🔧 技术方法**

核心技术包括：Fused Gromov‑Wasserstein 距离、Unsupervised Learning of Optimal Transport (ULOT) 网络、图字典学习、加速 OT 预测、α 条件下的字典建模（线性插值/MLP）、t‑SNE、PCA 等可视化与评估方法；

**📊 数据集**

使用 Human Connectome Project (HCP) 数据集，包含 1200 名受试者的 7 任务 fMRI 数据，左半球约 950 个节点的图，共约 12000 个对比图；

**📈 对比分析**

与传统基线（在共模板几何上直接进行字典学习）对比，在对比分类任务中 AGDL 的最佳 α=0.22 的性能略低于基线；但在受试者分类任务中，AGDL 的最佳 α=0.78 显著优于基线，显示其保留个体信息的优势；整体表现表明 AGDL 能在保持结构与特征对齐的同时捕获个体差异；

**⚠️ 局限性**

局限性包括：仅在左半球进行实验，右半球或双侧组合效果未知；字典几何固定，可能限制表达能力；对比分类任务的性能不如传统方法；尚未验证更大规模或实时处理的可扩展性；未来工作需探索非固定几何字典和双侧联合学习。

---

## 382. Solving Multivariate Polynomial Systems and Rectangular Multiparameter Eigenvalue Problems with MacaulayLab

**arXiv ID:** 2605.20884 | [PDF](https://arxiv.org/pdf/2605.20884v1)

**作者:** Christof Vermeersch `[一作]` (KU Leuven), Bart De Moor `[通讯]` (KU Leuven)

**通讯引用:** 46018 | [OpenAlex ID](https://openalex.org/A5018814006)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一个名为 MACE 或 MacaulayLab 的工具箱，用于通过块 Macaulay 矩阵的数值线性代数方法解决多变量多项式系统和矩形多参数特征值问题。

**💡 创新点**

创新点在于：①同时处理两类看似无关的问题；②对多项式基底和单词顺序完全不敏感；③能处理零维解集以及位于无穷远的正维解集；④提供多种求解策略（零空间、列空间、稀疏递归等），并模块化实现以便未来扩展。

**🔧 技术方法**

技术手段包括：块 Macaulay 矩阵构造、数值列空间/行空间求解、奇异值分解、QR 逆向分解、平移不变性、随机线性平移、并行 Schur 分解、聚类多重解、残差评估等；并通过自动选择多项式基底和单词顺序实现算法的通用性。

**📊 数据集**

使用了作者自行收集的数据库，其中包含 290 个多变量多项式系统和 30 个矩形多参数特征值问题，来源于已有公开数据集（如 Test Database of Polynomial Systems、Posso Test Suite）以及作者研究中的实例。

**📈 对比分析**

与 PHClab、PNLA、Maple（多项式系统）以及 MultiParEig（特征值问题）进行性能比较。实验显示 MacaulayLab 在大多数测试实例中竞争力十足：对大多数多项式系统，PHClab 速度最快但对非方阵问题不适用；MacaulayLab 在过定系统和正维无穷远解集情形下表现更优；在多参数特征值问题上，MacaulayLab 在多数多项式问题上优于 MultiParEig 的两种实现。性能曲线表明在大多数情况下其求解时间在 1.2‑2 倍范围内，且在极少数高维/大规模问题上仍有提升空间。

**⚠️ 局限性**

主要限制包括：只能处理零维解集（除非正维部分仅在无穷远处）；构造和存储块 Macaulay 矩阵的内存开销大；频繁的秩检查与平移操作会导致计算开销；对非线性多参数特征值问题的支持仍不完整；目前未实现交互式选择求解策略的功能。

---

## 383. Governance by Construction for Generalist Agents

**arXiv ID:** 2605.20874 | [PDF](https://arxiv.org/pdf/2605.20874v1)

**作者:** Segev Shlomov `[一作]` (IBM), Nir Mashkif `[通讯]` (IBM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了CUGA的策略系统，通过在LLM代理执行流程的五个关键点（意图门控、剧本注入、工具指南、人工批准、输出格式化）动态注入可组合的治理政策，使代理行为可预测、可审计；

**💡 创新点**

创新之处在于将治理转化为可外部化、可组合的policy-as-code层，利用多层次运行时检查点而非单纯的prompt工程，直接在模型之外强制约束其行为，兼容任何通用LLM；

**🔧 技术方法**

采用了LangGraph工作流框架、Milvus向量数据库进行语义检索、Embedding/关键词触发器、Policy Agent层的冲突解析以及OpenAI GPT-OSS/ GPT-4/Claude等多模型支持的技术栈；

**📊 数据集**

在公开基准数据集OAK（27项保险客服任务）和BPO（26项企业后台任务）上进行评估，并使用向量数据库存储策略规则；

**📈 对比分析**

通过在无策略和有策略两种配置下对比三种模型的Success Rate，结果显示在OAK上从75%提升到100%，在BPO上从49.2%提升到82.3%，提升幅度为15–37.7个百分点；

**⚠️ 局限性**

局限性包括：运行时治理引入额外token开销；触发器阈值的设置可能导致误判；人工批准步骤在高并发场景下可能成为瓶颈；并且策略需要持续维护以跟上业务规则变更。

---

## 384. ProCrit: Self-Elicited Multi-Perspective Reasoning with Critic-Guided Revision for Multimodal Sarcasm Detection

**arXiv ID:** 2605.20867 | [PDF](https://arxiv.org/pdf/2605.20867v1)

**作者:** Yingjia Xu `[一作]` (Soochow University), Min Cao `[通讯]` (Soochow University)

**通讯引用:** 11124 | [OpenAlex ID](https://openalex.org/A5035837633)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ProCrit 框架，实现多模态嘲讽检测的自发式多视角推理，并通过提议-批评两代理协同完成推理、批评与修订。

**💡 创新点**

创新点在于：① 用动态角色代理递归生成过程级推理注解，打破固定视角束缚；② 引入 draft–critique–revise 模式和互相精炼训练，让批评者为提议者提供可操作的自然语言反馈；③ 通过双阶段强化学习实现提议者草稿与修订的联合优化。

**🔧 技术方法**

技术包括：基于大语言模型的多角色生成、序列化推理注解、自然语言批评与反馈、双阶段 Group Relative Policy Optimization (GRPO) 的强化学习、以及跨代理的互相精炼训练策略。

**📊 数据集**

使用了三大公开数据集：MMSD、MMSD2.0（去噪版）和红色评估集 RedEval，涵盖多模态嘲讽检测任务。

**📈 对比分析**

与多种基线（无视角、预定义视角和手动或自动生成视角）在 zero‑shot 和 fine‑tuned 两阶段对比，ProCrit 在所有三个数据集上均实现了最高或最接近最高的 F1/召回率，并显著提升了召回率，说明批评指导有效捕捉细微讽刺信号。

**⚠️ 局限性**

局限性包括：① 仍依赖大模型计算资源；② 过程级推理注解的质量取决于教师模型的表现；③ 批评与修订过程可能受限于训练数据的多样性，导致在极端或新颖嘲讽机制上仍可能出现误判。

---

## 385. CAdam: Context-Adaptive Moment Estimation for 3D Gaussian Densification in Generative Distillation

**arXiv ID:** 2605.20872 | [PDF](https://arxiv.org/pdf/2605.20872v1)

**作者:** SeungJeh Chung `[一作]` (Kyung Hee University), HyeongYeop Kang `[通讯]` (Korea University)

**通讯引用:** 246 | [OpenAlex ID](https://openalex.org/A5011229651)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于统计信号验证的高效3D高斯稠密化方法CAdam，用于优化式生成式3D Gaussian Splatting；

**💡 创新点**

创新点在于把稠密化视为统计信号验证：使用动量估计消除随机噪声、量化分位数筛选、SNR门控与选择性透明度重置，实现自适应稠密化并软终止；

**🔧 技术方法**

采用Adam动量估计、量化分位数、SNR门控、选择性透明度重置等技术，集成到多种生成式稠密化目标（SDS、ISM、VFDS）及多种后端（GaussianDreamer、LucidDreamer、FlowDreamer等）；

**📊 数据集**

使用多种文本到3D基准数据集（T3Bench、DreamFusion Gallery）以及通过Gemini‑Pro和GPT‑4生成的自定义提示，共计约400条样本；

**📈 对比分析**

通过与传统基于梯度幅值累积的稠密化比较，CAdam在所有目标和后端上将高斯计数降低85%–97%，存储占用从约218 MB降至约5 MB，同时在CLIP、ImageReward、HPS v2、NIQE等质量指标与原始稠密化保持相近，用户研究总体偏好几乎持平；

**⚠️ 局限性**

局限在于仍需阈值调参（分位数与SNR阈值），过宽阈值可能导致过度稠密化，过窄阈值可能不足以细化薄部或低对比细节；

---

## 386. LOSCAR-SGD: Local SGD with Communication-Computation Overlap and Delay-Corrected Sparse Model Averaging

**arXiv ID:** 2605.20866 | [PDF](https://arxiv.org/pdf/2605.20866v1)

**作者:** Yassine Maziane `[一作]` (KAUST), Peter Richtárik `[通讯]` (KAUST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种在异构工作者、数据同质的分布式训练中，结合局部训练、稀疏同步和通信-计算重叠的算法，并提出延迟校正合并规则。

**💡 创新点**

首次对上述四个要素的组合给出理论收敛保证，并提出保持重叠进展的延迟校正合并。

**🔧 技术方法**

稀疏压缩、局部SGD、通信-计算重叠、延迟校正合并及理论分析（非凸L‑平滑目标），使用PyTorch逻辑时钟模拟实现。

**📊 数据集**

主要实验使用Logistic回归（四个工作者），附录中还验证了CIFAR‑10和Tiny ImageNet。

**📈 对比分析**

与阻塞稀疏同步、覆盖写入等方法对比，实验显示通信-计算重叠显著缩短训练时间，延迟校正合并优于覆盖写入，稀疏度降低通信成本的同时保持优化性能。

**⚠️ 局限性**

仅适用于数据同质场景；未证明时间复杂度最优；在异构目标下的安全性和效果尚未研究。

---

## 387. Terminal-World: Scaling Terminal-Agent Environments via Agent Skills

**arXiv ID:** 2605.20876 | [PDF](https://arxiv.org/pdf/2605.20876v1)

**作者:** Zihao Cheng `[一作]` (Beihang University), Yunhong Wang `[通讯]` (Beihang University)

**通讯引用:** 14116 | [OpenAlex ID](https://openalex.org/A5115589096)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出一种完全自动化的终端代理数据合成流水线，利用agent技能作为核心合成原语，联合生成任务说明、可执行环境与教师轨迹，最终构建5,723个高质量训练环境并训练8B/14B/32B系列模型。

**💡 创新点**

创新点在于将开源agent技能直接拆解为任务执行的三维信息（目标、前置条件、执行方式），通过技能团队与技能图扩展合成空间，实现任务、环境和轨迹的统一生成与对齐；同时通过技能指导的教师轨迹显著提升轨迹质量与效率。

**🔧 技术方法**

技术手段包括：多代理架构的生成‑验证‑修复（GVR）机制、LLM驱动的任务与环境生成、Gemini‑3‑Flash用于环境构建、DeepSeek‑V3.2作为教师模型收集轨迹，终端模拟器Terminus2用于评估。

**📊 数据集**

使用数据来源：从ClawHub与SkillMP收集10,000个agent技能，筛选后得到1,000个高质量技能，生成5,723个终端任务；训练集为对应的5,723个任务的教师轨迹（约5.7K条），仅占Nemotron‑Terminal 490.5K轨迹的1.2%。

**📈 对比分析**

评估方法：在6个基准（Terminal‑Bench 2.0、AIME24/25、DABench、TableBench、BIRD）上与多种基线（GPT‑5.2、Gemini、Claude、Nemotron等）进行SFT＋推理对比；32B模型在Terminal‑Bench 2.0上Pass@1 31.5（比Nemotron‑32B高+4.5），且在所有非终端基准上匹配或超过Nemotron‑32B，仅使用5.7K轨迹即可实现高效性能。

**⚠️ 局限性**

局限性包括：依赖大型LLM（如DeepSeek‑V3.2）进行轨迹生成，对更大规模或更复杂终端任务的泛化尚未充分验证；失败轨迹处理仍需更细粒度的策略；技能库覆盖面有限，未探索多语言或多平台终端环境的适配。

---

## 388. Convergence Analysis of Evolution Strategies for Mixed-Integer Optimization

**arXiv ID:** 2605.21000 | [PDF](https://arxiv.org/pdf/2605.21000v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 389. Playing Devil's Advocate: Off-the-Shelf Persona Vectors Rival Targeted Steering for Sycophancy

**arXiv ID:** 2605.21006 | [PDF](https://arxiv.org/pdf/2605.21006v1)

**作者:** Ishaan Kelkar `[一作]` (University of Toronto), Maheep Chaudhary `[通讯]` (Independent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究使用现成的角色向量（persona vectors）来抑制大型语言模型的同情偏向（sycophancy），并将其与对比激活添加（CAA）进行对比。

**💡 创新点**

创新点在于证明不需要专门的对齐标签，批判性思维角色向量即可在几乎与CAA相当的水平上降低同情偏向，并发现这些向量与CAA方向几乎正交，提示同情偏向更像是一种角色属性而非单一可驱动方向。

**🔧 技术方法**

采用在中间残差层进行激活加法（activation addition）的方法，对模型进行推理时的向量注入，评估其对同情偏向对数差（Δlogit）和二元准确率的影响。

**📊 数据集**

使用Gemma 2 27B和Qwen 3 32B两种指令微调模型，在对比平衡的PhilPapers强制选择基准（600条记录/种子）上进行实验。

**📈 对比分析**

通过对每个种子进行配对Wilcoxon检验并使用Holm校正，比较CAA、随机向量以及不同角色向量的Δlogit和二元率；批判性角色达到了CAA效果的68–98%，且保持了对事实的准确性，而顺从性角色效果不显著。

**⚠️ 局限性**

局限性包括仅在两种模型和有限的角色向量集上验证，正交性并不能证明机制独立；顺从性角色表现不一致且易受上限效应影响，需进一步在更多模型和向量上验证。

---

## 390. Conditional Equivalence of DPO and RLHF: Implicit Assumption, Failure Modes, and Provable Alignment

**arXiv ID:** 2605.20834 | [PDF](https://arxiv.org/pdf/2605.20834v1)

**作者:** Zhiqin Yang `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18994 | [OpenAlex ID](https://openalex.org/A5045081171)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了通过在RLHF中加入显式约束的偏好优化方法CPO和E-CPOC，以保证模型在学习人类偏好时不会出现与人类偏好相悖的“病态收敛”。

**💡 创新点**

核心创新在于揭示DPO与RLHF等价性仅在隐含假设（RLHF最优策略需优先人类偏好）成立时才成立；基于此提出的CPO在RLHF目标中加入可调阈值约束，并给出理论保证和可检验的损失到δ空间的桥接；E-CPOC进一步实现无奖励模型的保守显式约束，并证明其与受约束RLHF等价。

**🔧 技术方法**

技术手段包括：Bradley–Terry 偏好模型、KL 正则化的RLHF框架、对RLHF闭式解的变形、软阈值/硬约束的引入、对损失函数的可检验 ℓ^2-δ 近似、以及软边缘排名损失的几何解释。

**📊 数据集**

在实验中使用 Llama‑3‑8B‑Instruct 作为基础模型，并在 princeton‑nlp/llama3‑ultrafeedback‑armorm 数据集上进行偏好对齐；评估采用 AlpacaEval‑2 与 Arena‑Hard 两大对话任务。

**📈 对比分析**

与传统 DPO、SimPO、RDPO 等基线相比，CPO 在 AlpacaEval‑2 上取得 25.15% 的胜率（高于 DPO 的 24.60%），在 Arena‑Hard 上达 32.6%（比 SimPO 提升 2.6%，比 DPO 提升 3.7%），同时保持与基线相当的回答长度；E‑CPOC 亦在实验中表现与 CPO 相近，且无需奖励模型。

**⚠️ 局限性**

局限性包括：尚未在更大规模模型上验证效果；E‑CPOC 的实验评估相对不足；缺少训练动态可视化（如损失曲线、偏好准确率、病态收敛比例随训练的变化）。

---

## 391. Diagnosing Overhead in Dispatch Operations: Cross-architecture Observatory

**arXiv ID:** 2605.20982 | [PDF](https://arxiv.org/pdf/2605.20982v1)

**作者:** Bole Ma `[一作]` (Erlangen National High-Performance Computing Center), Gerhard Wellein `[通讯]` (Erlangen National High-Performance Computing Center)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 MoE 专家并行的 AlltoAll 通信中，作者通过 DODOCO 框架系统评估路由不平衡，检验了 EP 可纠正性和合成 token 基准有效性两大假设，并探究 EP 比例对负载分布的影响。

**💡 创新点**

创新点在于：①提出 DODOCO 的 5×6 因子实验，首次在多种模型与多种数据条件下同时验证 EP 与合成基准的两大假设；②发现 EP 规模变化对路由不平衡无效；③揭示合成 token 基准显著高估不平衡（最高 2.35×），并提出模型按数据稳健性分为“数据稳健”与“持续集中”两类的架构分类，为 Interconnect 设计提供更精准的工作负载模型。

**🔧 技术方法**

技术手段包括：PyTorch + Megatron-Core + NCCL；在每个 EP 及每个层级记录发送计数矩阵；计算 Gini 系数、对称 Dirichlet 浓度 α；进行 EP 扫描（4、8、16、32）和批量大小扫描（64、256、1024、2048）；通过 10 次测量循环获取统计值。

**📊 数据集**

使用了 6 种文本条件：Mock、Shuffled、Remapped、Romansh、Opus、Wikitext，均来自公开 HuggingFace 数据集，覆盖了 token 频率、序列结构、词嵌入、语言熟悉度等维度。

**📈 对比分析**

比较方法：对比 Mock 与真实文本下的 Gini 与 Dirichlet α；对 EP 扫描计算每专家最大/平均比例；对批量大小扫描观察 Gini 随 GBS 的变化。结果显示：EP 扫描中最大/平均比例波动 ≤5%，表明 EP 不能改善不平衡；Mock 条件下 Gini 与 α 与真实文本相比高达 2.35×，批量大小随 GBS 增大导致 Mock 不平衡恶化，而真实文本保持平稳。

**⚠️ 局限性**

局限性包括：实验规模仅限 16–32 GPU，未覆盖更大规模、多节点、多线的 AlltoAll；仅评估了 aux‑loss 平衡方案，未覆盖无平衡或其他路由机制；测量窗口短且跨 EP 的相关性未进一步验证；未测量系统级干预（如动态重排）对最终训练吞吐的影响。

---

## 392. Choose Wisely and Privately: Proactive Client Selection for Fair and Efficient Federated Learning

**arXiv ID:** 2605.20975 | [PDF](https://arxiv.org/pdf/2605.20975v1)

**作者:** Adda Akram Bendoukha `[一作]` (Institut Polytechnique de Paris), Aymen Boudguiga `[通讯]` (CEA-LIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种在联邦学习中提前进行差分隐私保护的客户端选择框架，以构建高效且公平的联邦网络。

**💡 创新点**

创新点在于：①基于互信息的潜在联盟损失(PFL)评估兼顾预测效能与群体公平；②使用差分隐私表格完成离线最优联盟搜索；③通过模拟退火实现组合优化。

**🔧 技术方法**

主要技术包括：差分隐私高维共现表计数、互信息估计、潜在联盟损失函数、模拟退火搜索算法、标准FedAvg训练。

**📊 数据集**

实验使用美国人口普查Folktables四个分类任务（ACSIncome、ACSEmployment、ACSPublicCoverage、ACSTravelTime），将美国州作为客户端。

**📈 对比分析**

与随机采样、FedProx、SCAFFOLD、ClusterFL、UCB-CS、FedSampling等对比，所选联盟在收敛速度、准确率和公平度上均优于基线。

**⚠️ 局限性**

局限性包括：需要事先计算并发布高维表格，噪声会降低信息精度；离线搜索不适应动态数据变化；仅验证于离散特征的表格数据，对图像等连续数据的推广有限。

---

## 393. Bridging Structure and Language: Graph-Based Visual Reasoning for Autonomous Road Understanding

**arXiv ID:** 2605.20942 | [PDF](https://arxiv.org/pdf/2605.20942v1)

**作者:** Lena Wild `[一作]` (KTH Royal Institute of Technology), Marco Pavone `[通讯]` (Stanford University)

**通讯引用:** 11780 | [OpenAlex ID](https://openalex.org/A5050003000)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Combined Road Substrate（CRS）框架，将几何道路结构与开放词汇语义统一为可执行的图表示，并通过递归唯一性实现“grounding for free”，自动生成可验证的Chain‑of‑Thought（CoT）与多层次问答对，显著提升视觉‑语言模型（VLM）在结构化道路推理任务中的性能。

**💡 创新点**

创新点在于：①利用递归唯一性保证所有生成的推理任务在图中有唯一、可追溯的根基；②在同一图中自动提取可验证的CoT，从而实现“零成本”可解释监督；③证明仅用20–80个CRS场景即可在多种VLM上获得显著的推理性能提升，表明结构化监督是突破瓶颈的关键。

**🔧 技术方法**

技术实现包括：图结构定义（节点、属性、边）、canonical operators（Φ_n,Φ_p,Φ_e）保证语义合法；递归唯一性与完整性约束；查询实例化机制（选择器、问题模板、答案模板、扰动操作）生成多种复杂问答；CoT抽取与链式推理；在Argoverse2视觉数据、OpenLane‑V2 HD地图与拓扑上构建CRS；基于Qwen、Gemini、Claude等多种VLM进行全参数微调。

**📊 数据集**

数据集：利用Argoverse2前视图与OpenLane‑V2 HD地图与拓扑，人工筛选与丰富生成80个CRS图谱，产生约22k训练问答对（19类查询）与1000个验证样本（120个场景）。

**📈 对比分析**

通过在多种开源与闭源VLM（Qwen‑4B‑SFT、Qwen‑2B‑SFT、Gemini‑3.1‑Prom等）上进行对比，CRS微调后在结构化推理任务的整体准确率提升约30–40%，在仅40个场景训练下就能超过当前强闭源基线；小模型（2–4B参数）表现尤为显著，且在不同推理深度下保持稳定。

**⚠️ 局限性**

限制：依赖高质量视觉感知，当前CRS图谱仍需人工策划与审查；CoT虽可解释但对整体准确率提升有限；未直接验证对下游驾驶决策的效果；未来需探索自动化图谱构建与强化学习驱动的结构化监督。

---

## 394. Towards Integrated Rock Support Visualisation in 3D Point Cloud of Underground Mines

**arXiv ID:** 2605.20973 | [PDF](https://arxiv.org/pdf/2605.20973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 395. Comparative Evaluation of Deep Learning Models for Fake Image Detection

**arXiv ID:** 2605.20971 | [PDF](https://arxiv.org/pdf/2605.20971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 396. WiXus: A Wheeled-Legged Robot with Wire-Driven Environmental Utilizing to Integrate Mobility and Manipulation

**arXiv ID:** 2605.20932 | [PDF](https://arxiv.org/pdf/2605.20932v1)

**作者:** Shintaro Inoue `[一作]` (University of Tokyo), Kei Okada `[通讯]` (University of Tokyo)

**通讯引用:** 6655 | [OpenAlex ID](https://openalex.org/A5101836795)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一款名为 WiXus 的机器人，融合了脚车轮移动与线缆驱动系统，并通过实验展示了平面地图构建、崖壁攀爬、救援式物体搬运以及工具（刈草机）利用等多种功能。

**💡 创新点**

其创新点在于通过环境线缆锚定实现机器人悬浮，使原本用于移动的脚车轮能够被解放出来，转而充当抓取和操作的机械臂，从而扩展了脚车轮机器人的三维作业域。

**🔧 技术方法**

采用了脚车轮驱动、V型链驱动、RTAB‑Map SLAM、SMACH 状态机、CAN‑USB 直流电机控制、以及自主飞行锚定无人机等技术，辅以 M2006、Robstride02、CyberGear 等高密度伺服电机。

**📊 数据集**

实验环境为实验室自制场景，使用 RTAB‑Map 生成自建地图，并自行记录线缆锚点；并未使用公开数据集。

**📈 对比分析**

与现有脚车轮平台进行定性对比，WiXus 在平面行走和 3D 攀爬时实现了良好的速度跟踪和任务完成率；在悬浮状态下完成抓取与工具使用，展示了原有平台无法实现的功能，但未给出量化的性能基准。

**⚠️ 局限性**

局限性包括线缆驱动控制频率低、依赖操作员手动指令、缺乏完整的自主运动规划与感知模块，以及需要先知晓线缆锚点等环境信息。

---

## 397. Beyond Text-to-SQL: An Agentic LLM System for Governed Enterprise Analytics APIs

**arXiv ID:** 2605.21027 | [PDF](https://arxiv.org/pdf/2605.21027v1)

**作者:** Gundeep Singh `[一作]` (Dialpad Inc), Shashi Bhushan TN `[通讯]` (Dialpad Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 Analytic Agent，一种基于 LLM 的代理系统，能够将自然语言请求转化为安全调用企业分析 API，并返回结果或可视化。

**💡 创新点**

将 LLM 代理与企业受控 API 结合，支持多步骤规划、目标解析、权限验证、结构化查询与治理-aware 可视化，突破传统 Text-to-SQL 在企业环境中的局限。

**🔧 技术方法**

使用 Gemini-2.5 系列 LLM、Google Agent Development Kit、LiteLLM、D3 可视化、权限验证与日志审计，并在 Google Cloud Run 部署。

**📊 数据集**

基于领域专家编制的 90 条企业用例集合，包含自然语言查询和对应的 API 负载作为黄金标准。

**📈 对比分析**

采用 AlignScore 与 GPT‑5.2/Claude‑Opus‑4.6 的 LLM‑as‑judge 评估，Gemini‑2.5‑Pro 达到 77.22% 的端到端准确率、96.67% 查询执行成功率；Flash 为 71.67%/94.44%。

**⚠️ 局限性**

评估仅覆盖 Gemini 系列，且对规模化覆盖仍有限，模型容量不足时结构化查询正确率低；在多步骤推理与权限验证上仍易出错。

---

## 398. A Deployment Audit of Release-Side Risk in Conformal Triage under Prevalence Shift

**arXiv ID:** 2605.20956 | [PDF](https://arxiv.org/pdf/2605.20956v1)

**作者:** Chengze Li `[一作]` (University of Illinois Chicago), Philip Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 136869 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并实现了一种针对预后偏移的发布侧合规三角形部署审计，评估低审查率下的安全性。

**💡 创新点**

创新点在于将预后校正、合规阈值校准与安全评估分离到不同数据子集，直接报告事件释放风险而非仅仅边际覆盖。

**🔧 技术方法**

采用合规预测、单调预后校正、池化与类别化分割合规、分层抽样等技术。

**📊 数据集**

使用了Retrospective NSCLC（NSCLC‑Radiogenomics）以及TCIA公开数据集。

**📈 对比分析**

通过对比池化校正与类别化合规在相同评估集上的人类审查率和事件释放风险，发现前者可降低审查但增加事件风险，后者主要通过增加审查实现安全。

**⚠️ 局限性**

局限在于仅在单一回溯NSCLC样本上验证，缺乏前瞻性多中心试验及事件标签不足导致的样本量瓶颈。

---

## 399. 3D Reconstruction and Knowledge Distillation to Improve Multi-View Image Models to Explore Spike Volume Estimation in Wheat

**arXiv ID:** 2605.20940 | [PDF](https://arxiv.org/pdf/2605.20940v1)

**作者:** Olivia Zumsteg `[一作]` (ETH Zurich), Paraskevi Nousi `[通讯]` (Swiss Data Science Center)

**通讯引用:** 635 | [OpenAlex ID](https://openalex.org/A5074409158)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种融合3D点云几何监督与仅用2D图像推断的混合3D‑to‑2D框架，用于田间小麦穗体积估计。

**💡 创新点**

创新点在于：①利用距离直方图实现刚体不变的PointNet，在室内高质量扫描监督下训练出能处理部分场景点云的模型；②构建Regulated Transformer，结合单视角与全视角的体积预测；③将上述3D与2D模型通过多模态集成后进行知识蒸馏（特征蒸馏与伪标签蒸馏），从而得到仅需2D图像即可高效推断的模型。

**🔧 技术方法**

采用的技术包括：多视角RGB图像采集、YOLOv11检测+SAM分割、DINOv2等ViT骨干、Regulated Transformer、点云距离直方图与点Transformer、知识蒸馏（特征匹配与伪标签）、OpenMVS点云重建、epipolar一致性匹配、集成学习与梯度下降训练。

**📊 数据集**

使用在ETH Lindau‑Eschikon田间平台收集的1134个标记小麦穗（93个品系），共12个RGB相机，配合Shining 3D Einscan-SE V2光学扫描获得的体积真值，公开数据集可在https://oliviazum.github.io/3DKD-wheat/获取。

**📈 对比分析**

与线性、LSTM、Transformer、Point Transformer等基线模型对比，单个Regulated Transformer在DINOv2骨干下取得MAE≈654 mm³、r≈0.76；引入刚体不变PointNet并进行KD后MAE降至≈597 mm³、r≈0.80；最优的多模态集成模型MAE≈578 mm³、r≈0.83、MAPE≈13%；蒸馏后的Regulated Transformer在仅用图像时MAE≈640 mm³、r≈0.78-0.82、推理时间≈1.4 ms（相比集成的160 ms）。

**⚠️ 局限性**

局限性包括：仍需多视角图像与12台相机的硬件支持；点云重建对风动等场景噪声敏感；不同生长阶段的形态变化导致MAE随采样日变化；当前模型未能直接在3D空间进行推断，无法利用稠密深度信息；域特定骨干（FoMo4Wheat）表现欠佳，表明跨域迁移仍有挑战。

---

## 400. DrawMotion: Generating 3D Human Motions by Freehand Drawing

**arXiv ID:** 2605.20955 | [PDF](https://arxiv.org/pdf/2605.20955v1)

**作者:** Tao Wang `[一作]` (Beijing University of Posts and Telecommunications), Li Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 43814 | [OpenAlex ID](https://openalex.org/A5100336135)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出了DrawMotion框架，利用手绘stickman与轨迹作为条件，生成同时符合文本描述与用户手绘意图的人类动作序列。

**💡 创新点**

创新点在于首次将手绘条件引入文本到动作生成，设计了高效的多条件融合模块（MCM）和训练无关的中间特征引导（IFG），实现了细粒度且精确的控制。

**🔧 技术方法**

技术上采用DDIM扩散模型作为生成基础，结合多条件解码器、dot‑product与efficient attention、MCM以及基于Mahalanobis距离的梯度裁剪实现训练无关的引导。

**📊 数据集**

使用KIT‑ML和HumanML3D两个公开动作数据集进行训练与评估。

**📈 对比分析**

在多项指标（FID、R‑Precision、Trajectory Error、StiSim）与竞争方法相比，DrawMotion取得最优或接近最优成绩，并且在推理速度与显存占用上优于现有编辑方法。

**⚠️ 局限性**

限制在于当用户提供的轨迹或stickman与文本或物理规则冲突时，模型可能偏离输入导致生成质量下降；此外需要用户提供相对合理的手绘输入。

---

## 401. Single-Pass, Depth-Selective Reading for Multi-Aspect Sentiment Analysis

**arXiv ID:** 2605.20998 | [PDF](https://arxiv.org/pdf/2605.20998v1)

**作者:** Yan Xia `[一作]` (Universiti Malaya), Chee Seng Chan `[通讯]` (Universiti Malaya)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对多方面情感分析（ATSA）中句子重编码与静态表示的效率‑表达力权衡，提出单通道深度有序子结构（DABS）框架，只需一次编码即可为所有方面共享表示，并通过可调深度查询实现轻量级的方面条件读取。

**💡 创新点**

创新点在于将 Transformer 深度视为可查询的资源，构建可重用的深度层级子结构（Depth‑Ordered Representation Aggregation），并在此基础上实现方面预算感知的选择（ACBS），同时引入稀疏、span‑mask 与融合熵正则，确保在不损失精度的前提下显著降低多方面推理成本。

**🔧 技术方法**

核心技术包括：Pre‑trained Transformer（DeBERTa‑v3‑base）作为编码器；Depth‑GRU 对最后 K 层进行递归聚合形成深度子结构；Aspect‑Conditioned Budget‑Aware Selection 通过独立的 token gating 与深度分布预测实现证据定位与深度选取；多头注意力 + MLP + softmax/σ 机制；稀疏、span‑mask 与熵正则化提升稳定性和可解释性。

**📊 数据集**

使用了四个英语 ATSA 基准（SemEval‑2014 Laptop & Restaurant、SemEval‑2015 Restaurant、SemEval‑2016 Restaurant），以及三种非英语（法语、俄语、西班牙语）SemEval‑2016 ABSA 数据集进行跨语言泛化验证。

**📈 对比分析**

与结构感知、Fine‑tune 与 LLM（Llama‑3、Qwen3、GPT‑3.5）等多类基线对比，DABS 在所有四个数据集上均实现或优于现有最高 Accuracy 与 Macro‑F1（例如 Lap14 Acc 84.41% / MF1 81.56%），相比 LLM 在 5‑shot 设定下提升 10+ 个百分点；在多方面情景下，端到端计算量减少 60% 以上，推理延迟和吞吐量亦显著提升。

**⚠️ 局限性**

主要局限：①收益高度依赖多方面场景，单一方面（M=1）时优势有限；②保持深度子结构需额外的计算与内存开销，随序列长度、模型尺寸与 K 递增；③目前仅在已知方面跨度的 ATSA 任务上评估，未覆盖隐式或联合提取方面的设置。

---

## 402. Thinking-while-speaking: A Controlled, Interleaved Reasoning Method for Real-Time Speech Generation

**arXiv ID:** 2605.20946 | [PDF](https://arxiv.org/pdf/2605.20946v1)

**作者:** Xuan Du `[一作]` (Huawei Technologies), Xinghao Chen `[通讯]` (Huawei Technologies)

**通讯引用:** 3877 | [OpenAlex ID](https://openalex.org/A5006817088)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 InterRS 思考-说话并行框架，实现即时响应的连贯推理语音生成。

**💡 创新点**

创新地将思考段嵌入语音生成间隙，结合 TA‑Balance 与 Linguistic Quality 两种奖励，精细控制思考长度和语义连贯。

**🔧 技术方法**

使用 Qwen2.5‑Omni‑3B 端到端模型，结合 interleaved SFT 与 Group Relative Policy Optimization（GRPO）强化学习，并配套语音合成。

**📊 数据集**

基于 K&K、MetaMath、Spoken‑MQA、SATA‑Bench 等数据，经过三阶段对齐管线生成高质量 interleaved 语音推理数据。

**📈 对比分析**

与思考前说话、直接回答、Fast CoT 以及 Mini_OR 等基线比较，InterRS 在数学/逻辑测试中平均得分 49.33，保留 96.5% 的思考精度且实现即时流式输出，流利度评分提升至 1.83/2。

**⚠️ 局限性**

目前仅在静态轮询评估，未处理用户中断或话题跳转等动态对话情境，需要进一步扩展以适应真实交互。

---

## 403. CHOIR: Contact-aware 4D Hand-Object Interaction Reconstruction

**arXiv ID:** 2605.20992 | [PDF](https://arxiv.org/pdf/2605.20992v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 404. Component Influence-Driven Fastener Reduction for Robotic Disassemblability-Aware Design Simplification

**arXiv ID:** 2605.21026 | [PDF](https://arxiv.org/pdf/2605.21026v1)

**作者:** Takuya Kiyokawa `[一作]` (University of Osaka), Kensuke Harada `[通讯]` (University of Osaka)

**通讯引用:** 11125 | [OpenAlex ID](https://openalex.org/A5016270703)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套基于组件影响得分的分析框架，用以在产品设计早期识别并减少对机器人拆装过程产生负面影响的紧固件，从而降低结构约束、工具更换次数和机器人行走距离。

**💡 创新点**

创新点在于将机器人拆装序列规划结果转化为可量化的组件影响得分，并通过热图可视化以及基于影响得分的排名机制提供可操作的紧固件删减建议，同时通过几何稳定性指标防止结构不安全的修改。

**🔧 技术方法**

技术主要包括：CAD模型与Contact‑Connection‑Constraint（CCC）图的自动生成、通过序列交换模拟计算影响得分、解析式地评估去除紧固件后的结构约束、工具更换与行走距离的减少以及极值比（ρ_J、ρ_A）等几何稳定性评估。

**📊 数据集**

使用的数据集为七款家用电器（冷凝器、电视、空调外机、微波炉、迷你微波炉、美容设备、电动工具），每个零件在CAD模型中被转化为CCC图中的节点与边。

**📈 对比分析**

通过与随机选取紧固件的基线进行对比，实验显示在六个产品中都能减少结构约束（ΔE）并保持工具更换次数不增加（ΔT ≤ 0），在大多数产品上行走距离缩短幅度达165–1675 mm，表明该方法在提升拆装效率方面显著优于随机策略。

**⚠️ 局限性**

局限性包括：仅基于几何关系而未考虑材料强度或机器人动力学；仅针对紧固件删减；未进行高保真有限元或完整运动规划验证；R_max设为3时可能不适用于所有结构复杂度较高的产品。

---

## 405. Building a Custom Taxonomy of AI Skills and Tasks from the Ground Up with Job Postings

**arXiv ID:** 2605.21029 | [PDF](https://arxiv.org/pdf/2605.21029v1)

**作者:** Stephen Meisenbacher `[一作]` (Technical University of Munich), Peter Norlander `[通讯]` (Loyola University Chicago)

**通讯引用:** 220 | [OpenAlex ID](https://openalex.org/A5069547675)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开发了名为TaxonomyBuilder的工具，利用大型语言模型（LLM）和聚类技术从大规模工作岗位文本中自动构建AI技能层次化分类体系，并系统评估不同数据使用策略（数据增量、软聚类、分位数过滤）对分类质量的影响。

**💡 创新点**

创新点在于：①首次从系统化、实验角度探讨在海量文本下“少即多”的原则，揭示数据增量与软聚类对税onomies质量的相反影响；②提出并验证了分位数过滤策略的有效性；③公开了可直接复现的TaxonomyBuilder框架，为后续领域自定义税onomies奠定基础。

**🔧 技术方法**

技术方法包括：①关键词检索（pyahocorasick）+句子嵌入与类向量评分（MiniLM、gte、embeddinggemma）筛选候选语境；②HDBSCAN+UMAP聚类；③LLM生成标签（gpt‑4o‑mini）并通过相似度聚合去冗余；④LLM-as-a-Judge评估（gpt‑5.4‑nano）和域覆盖（embedding‑based semantic search）对构建结果进行量化。

**📊 数据集**

使用数据集为：NLx（约3.1M公开职位，筛选2024‑25年共32.12M条）与USAJOBS（1.54M联邦职位），各自保留最后一个月做为测试集。

**📈 对比分析**

通过12种配置（增量/不增、软聚类/不、25/50/75%分位过滤）分别在两大数据集上评估：聚类轮廓系数、LLM-as-a-Judge四项指标、域覆盖宏F1。结果显示：无增量+软聚类+50%过滤组合在大多数指标上表现最佳，且增量往往降低覆盖率，验证“少即多”原则。

**⚠️ 局限性**

限制在于：①仅在自研TaxonomyBuilder框架内评估，未检验其对其他现有方法的影响；②缺乏对LLM模型、聚类参数、标签修剪等的更细粒度消融研究；③仅关注AI技能领域，结果对其他领域的泛化仍待验证。

---

## 406. Preserve, Reveal, Expand: Faithful 4D Video Editing with Region-Aware Conditioning

**arXiv ID:** 2605.20961 | [PDF](https://arxiv.org/pdf/2605.20961v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 407. Towards UAV Detection in the Real World: A New Multispectral Dataset UAVNet-MS and a New Method

**arXiv ID:** 2605.20963 | [PDF](https://arxiv.org/pdf/2605.20963v1)

**作者:** Yihang Luo `[一作]` (National University of Defense Technology), Zhijie Chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了同步高分辨率RGB–MSI小无人机检测数据集UAVNet-MS，并提出双流融合检测框架MFDNet；

**💡 创新点**

①首个多光谱小无人机数据集；②引入物理对齐模块ArrayCode；③采用双流特征提取与尺度感知语义解耦融合；

**🔧 技术方法**

使用多光谱相机阵列采集、光谱对齐编码（ArrayCode）、3D卷积编码器、双流深度网络以及轻量级尺度解耦融合；

**📊 数据集**

UAVNet-MS（15,618 RGB–MSI数据块，4种材质无人机，93.7%极小目标），并与20种基准检测器对比；

**📈 对比分析**

在RGB-only、MSI-only、RGB+MSI三种协议下，使用mAP@0.5评估；MFDNet在RGB+MSI下相较最优RGB基线提升6.2% AP_50，极小目标ET提升19.9% AP；

**⚠️ 局限性**

MSI单独检测仍低于RGB基线，模型体积和推理速度相对较慢，对低光谱噪声处理不足，数据集规模仍有限。

---

## 408. Modeling Temporal scRNA-seq Data with Latent Gaussian Process and Optimal Transport

**arXiv ID:** 2605.20989 | [PDF](https://arxiv.org/pdf/2605.20989v1)

**作者:** Mehmet Yigit Balik `[一作]` (Aalto University), Harri Lähdesmäki `[通讯]` (Aalto University)

**通讯引用:** 12215 | [OpenAlex ID](https://openalex.org/A5049442192)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于潜在异方差高斯过程与最优传输的生成式模型（LGP‑OT），用于从单细胞RNA测序的离散时间点重建连续细胞发育轨迹。

**💡 创新点**

核心创新在于：①将异方差高斯过程放入潜在空间，以捕获随时间变化的生物噪声；②使用Hilbert空间低秩近似实现大规模可扩展；③通过最优传输目标匹配生成与观测的细胞分布，避免对单细胞轨迹的假设；④引入细胞特异性潜在时间和细胞类型条件化，解耦时间异步与多谱系；⑤开发梯度驱动的扰动策略，可在时间域模拟基因或细胞类型干扰。

**🔧 技术方法**

使用的技术包括：高斯过程（GP）与Hilbert空间逼近、异方差GP先验、最优传输（带熵正则化的Wasserstein距离）、变分推断（Wasserstein自编码器框架）、无编码器解码器结构、梯度优化扰动。

**📊 数据集**

实验数据集包括：斑马鱼胚胎（ZB）、果蝇胚胎（DR）、小鼠iPSC再编程（SC）等三套公开scRNA‑seq时序数据。

**📈 对比分析**

与scNODE、MIOFlow、PRESCIENT、PI‑SDE、VGFM等基线在插值、外推和稀疏组合任务上进行比较，采用Wasserstein距离评估。LGP‑OT在所有三种任务和三套数据集上均实现最低误差，尤其在外推和稀疏数据下明显优于对手。

**⚠️ 局限性**

主要局限包括：未显式建模分支事件；使用SE核限制对周期性或振荡动态的建模；仅适用于转录组测序，尚未扩展到多模态或空间单细胞数据。

---

## 409. Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy

**arXiv ID:** 2605.20965 | [PDF](https://arxiv.org/pdf/2605.20965v1)

**作者:** Yutong Xie `[一作]` (Southeast University), Yuheng Jia `[通讯]` (Southeast University)

**通讯引用:** 1820 | [OpenAlex ID](https://openalex.org/A5013880628)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在推理阶段利用层间视觉注意差异构造视觉证据显著图，并在生成过程中加强对该证据的关注，从而抑制大型视觉‑语言模型（LVLM）在生成文本时的幻觉。

**💡 创新点**

① 训练无关、即插即用；② 利用层间视觉注意差异（ILVAD）自动识别并聚焦正确视觉证据；③ 在视觉和文本两方面共同提升，进一步减少语言先验导致的错误；④ 只改动注意力权重，保持模型原有架构。

**🔧 技术方法**

层间视觉注意差异分析、显著图构建、头部选择、注意力增强（对视觉证据和文本的双重强化）、基于阈值的显著性筛选、可视化与归一化。

**📊 数据集**

幻觉评测基准：CHAIR（图像描述）、POPE（是否包含物体判断）、MMHal‑Bench（多模态事实一致性）、MME（对象/属性层面评估）以及LLaVA‑Bench（开放式问答）。训练/测试数据来源于MSCOCO等公开图像数据集。

**📈 对比分析**

在五种最新 LVLM（LLaVA‑1.5‑7B、LLaVA‑NeXT‑7B、Qwen2‑VL‑7B、Qwen3‑VL‑8B、InternVL3‑8B）上与 Greedy、Beam、VCD、CODE、AGLA、ONLY、VAR、SPARC、VAF、VHR 等多种基线对比；在大多数评测指标上达到或超过现有最优方法，尤其在幻觉比例、准确率和自然度方面均有显著提升；推理时间几乎无额外开销。

**⚠️ 局限性**

① 需对阈值 τ、增强强度 α、β 等超参数进行经验调优；② 仅关注视觉证据，可能在需要更细粒度语言先验的任务中效果有限；③ 对极长文本生成时，早期 token 统计可能不足以捕获全部证据；④ 对非视觉感知（例如多模态音频）扩展性尚未验证。

---

## 410. Causal Past Logic for Runtime Verification of Distributed LLM Agent Workflows

**arXiv ID:** 2605.20923 | [PDF](https://arxiv.org/pdf/2605.20923v1)

**作者:** Benedikt Bollig `[一作]` (Université Paris-Saclay), Benedikt Bollig `[通讯]` (Université Paris-Saclay)

**通讯引用:** 1210 | [OpenAlex ID](https://openalex.org/A5082538918)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

为分布式LLM代理工作流设计并实现了Causal Past Logic (CPL) 用于运行时监控与决策控制

**💡 创新点**

创新点在于将因果可见性作为决策依据，提出基于向量时钟的在线监控算法，使守卫在源代码层面实时评估并影响控制流，而非事后日志检查

**🔧 技术方法**

主要技术包括：消息序列图(MSC)模型、过去时态逻辑(CPL)、向量时钟与最新值视图、ZipperGen编排框架以及Lean4实现的形式化验证

**📊 数据集**

论文未使用公开数据集，而是通过在ZipperGen的原型实现（Python实现）和示例工作流（代码审查、测试与安全检查）进行演示

**📈 对比分析**

对比方法主要体现在与传统基于顺序日志的运行时验证方法对比；实验表明CPL可在本地完成决策，无需全局日志，监控开销可通过向量时钟与最新值视图的局部维护得到控制，性能符合理论下限的可扩展性

**⚠️ 局限性**

局限性包括：对向量时钟的无界计数器导致资源增长、对高度并发或长生命周期执行的规模化挑战、以及未在真实LLM工作流中进行大规模性能评估

---

## 411. LiteViLNet: Lightweight Vision-LiDAR Fusion Network for Efficient Road Segmentation

**arXiv ID:** 2605.21007 | [PDF](https://arxiv.org/pdf/2605.21007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 412. Hybrid Machine Learning Model for Forest Height Estimation from TanDEM-X and Landsat Data

**arXiv ID:** 2605.20997 | [PDF](https://arxiv.org/pdf/2605.20997v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 413. Verifiable Provenance and Watermarking for Generative AI: An Evidentiary Framework for International Operational Law and Domestic Courts

**arXiv ID:** 2605.21002 | [PDF](https://arxiv.org/pdf/2605.21002v1)

**作者:** Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov `[一作]` (Swedish Defence University), Nurana Abdullayeva `[通讯]` (ADA University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一个统一的证据框架，将加密来源证明、稳健水印和零知识证明结合，映射到国际作战法、国内程序和产品监管的合法性阈值。

**💡 创新点**

首次提出跨法域、可调权重的Dempster‑Shafer聚合器，将经验检测性能量化为法律可接受度，并公开了72000样本基准。

**🔧 技术方法**

采用加密内容来源证明（C2PA Ed25519）、三类水印（Stable Signature、Tree Ring、Gaussian Shading）、零知识证明（zk‑SNARK）、Dempster‑Shafer理论与贝叶斯阈值映射等技术。

**📊 数据集**

使用公开生成模型（SDXL, FLUX.1, Stable Audio 2, Suno v4, Veo 2, Sora）产生12000样本，随后通过六种清洗流水线生成72000评估样本，数据许可为Apache 2.0/CC‑BY 4.0。

**📈 对比分析**

采用TPR@FPR=10⁻³、AUC、计算开销等指标对单一方案与聚合系统进行对比，聚合系统在Tier 2–3攻击下TPR约0.92/0.78/0.47，显著优于任何单一水印，并满足大多数法域阈值。

**⚠️ 局限性**

基准覆盖有限的模型与攻击，未涵盖更强模型或新型攻击；零知识证明成本仍高，缺乏跨司法认可的信任根；对操作员概率解释的研究不足。

---

## 414. Beyond the Bellman Recursion: A Pontryagin-Guided Framework for Non-Exponential Discounting

**arXiv ID:** 2605.20996 | [PDF](https://arxiv.org/pdf/2605.20996v1)

**作者:** Hojin Ko `[一作]` (Sungkyunkwan University), Jeonggyu Huh `[通讯]` (Sungkyunkwan University)

**通讯引用:** 51 | [OpenAlex ID](https://openalex.org/A5013964577)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种新的基于Pontryagin最大原理的直接策略优化框架（PG‑DPO），专门解决非指数折扣下的连续时间控制问题。

**💡 创新点**

创新点在于完全摒弃Bellman递推，采用决策时间锚定的Pontryagin条件与Monte‑Carlo投影相结合，通过局部Hamiltonian最大化实现时间一致性或平衡解，显著提高了对非指数折扣的适应性。

**🔧 技术方法**

使用的技术包括：可微模拟器或学习动力学模型、反向传播（BPTT）作为随机成本敏感性估计、Monte‑Carlo平均化成本状态、局部Hamiltonian最大化投影（Newton/内点法）以及两阶段PG‑DPO算法。

**📊 数据集**

在合成的连续时间控制基准上评估：多维（d=5、10、100）生存折扣目标控制、Merton资产配置与超几何折扣、以及时间变化的冲动消费/资源管理任务，全部通过模拟生成。

**📈 对比分析**

与PPO、DPO、PINN、Deep BSDE/BSVIE等基线比较，PG‑DPO在所有折扣情形下实现了平均L₁误差低至1e‑2级别，标准差几乎可忽略，优于对手两到三阶量级；在极端非指数折扣（超几何）下，误差下降到1e‑3级别，显著提高了控制精度与鲁棒性。

**⚠️ 局限性**

主要限制包括：需可微的真实或学习动力学模型；成本敏感性估计对BPTT的方差较大；投影步骤对折扣核的光滑性与可导性有要求；在高维或非连续约束下的数值稳定性和计算成本仍需进一步改进。

---

## 415. Towards Context-Invariant Safety Alignment for Large Language Models

**arXiv ID:** 2605.20994 | [PDF](https://arxiv.org/pdf/2605.20994v1)

**作者:** Yixu Wang `[一作]` (Fudan University), Yingchun Wang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 12495 | [OpenAlex ID](https://openalex.org/A5100613144)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Anchor Invariance Regularization (AIR)，通过把可验证的提示作为锚点，使用 stop‑gradient 约束开放式提示，使安全对齐在不同表面形式下保持一致，并将其与 Group Relative Policy Optimization (GRPO) 等强化学习框架结合。

**💡 创新点**

创新点在于：① 将对齐问题转化为非对称的 invariance 正则；② 采用锚点与 stop‑gradient 形成的单向约束，避免传统对称正则导致可靠提示被牺牲；③ 通过组化提示构造（包含可验证锚点和开放式变体）实现高效且可扩展的训练。

**🔧 技术方法**

使用技术包括：Anchor Invariance Regularization、Group Relative Policy Optimization (GRPO)、Group Sequence‑level Policy Optimization (GSPO)、V‑REx、Invariant Risk Minimization (IRM) 思想、强化学习与人类反馈 (RLHF) 体系、stop‑gradient 操作以及辅助损失实现。

**📊 数据集**

实验数据集涵盖安全、道德推理和数学三大领域：Safety（AdvBench、构造的多样化提示）、Moral Reasoning（Moral Choice 数据集）和 Mathematics（GSM8K）。每个领域都设计可验证的多选/真伪锚点与开放式变体。

**📈 对比分析**

与 GRPO、GSPO、V‑REx 等基线在 ID 与 OOD 上对比，评估指标为 prompt‑level Accuracy (Acc) 与 group‑level Accuracy (Acc_group)。AIR 在所有三大领域的 Acc_group 上提升 12‑15%（ID）和约 33%（OOD），并显著提升整体准确率，表明在面对对抗性表述时模型的鲁棒性得到明显改善。

**⚠️ 局限性**

局限性：① 依赖存在可验证锚点的任务，无法直接应用于完全无监督或无可验证提示的场景；② 对超参数 λ 的敏感性较高，需手工调优；③ 实验仅在 Qwen‑2.5‑14B 上验证，尚未在更大规模或多模态模型上评估；④ 尽管降低了奖励游戏，但仍对噪声奖励敏感，若锚点质量低也可能影响效果。

---

## 416. ArPoMeme: An Annotated Arabic Multimodal Dataset for Political Ideology and Polarization

**arXiv ID:** 2605.20967 | [PDF](https://arxiv.org/pdf/2605.20967v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 417. Domijn: The Security of Domain Registrars and the Risk of a Domain Name Takeover

**arXiv ID:** 2605.20984 | [PDF](https://arxiv.org/pdf/2605.20984v1)

**作者:** Koen van Hove `[一作]` (NLnet Labs & University of Twente), Roland van Rijswijk-Deij `[通讯]` (University of Twente)

**通讯引用:** 1413 | [OpenAlex ID](https://openalex.org/A5091643049)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过实测 10 家 .nl 顶级域名注册商的安全控制，评估域名被劫持的可能性与影响，并将其风险与勒索软件、DDoS 攻击进行对比。

**💡 创新点**

创新点在于首次将域名劫持的技术可行性与业务影响量化为模型，并基于 NIST 风险评估尺度将两者关联，揭示域名劫持对组织可能造成与勒索软件相当的损失。

**🔧 技术方法**

技术方法包括：自动化注册测试、TOTP 代码暴力破解率限制评估、电话交互模拟、WHOIS / RDAP 数据抓取、对 10 家注册商的多因素身份验证与恢复流程进行审计。

**📊 数据集**

使用的数据集为：2024‑07‑22~29 期间 Cloudflare Radar 提供的前 1,000,000 个域名（约 9,000 个 .nl 域名），以及对应的 RDAP / WHOIS 记录。

**📈 对比分析**

比较方法：对注册商的安全措施进行打分（如是否支持 2FA、是否有按账户的速率限制、是否需要多重身份验证），并将劫持后可执行的攻击（DNS 篡改、收集邮件、转移域名等）与勒索软件、DDoS 的损失模型对齐；结果显示绝大多数注册商已具备基本防护，但在 TOTP 速率限制和安全通知方面存在显著缺陷，整体风险评估处于“低至中”与“高”影响区间之间。

**⚠️ 局限性**

局限性包括：仅测试公开注册的普通用户账户，未覆盖企业级注册商；密码泄露概率基于外部统计假设，无法直接验证；法律与伦理限制阻止发送伪造身份证或大规模攻击；测试结果可能随注册商后续安全改进而改变。

---

## 418. Point Cloud Sequence Encoding for Material-conditioned Graph Network Simulators

**arXiv ID:** 2605.20978 | [PDF](https://arxiv.org/pdf/2605.20978v1)

**作者:** Philipp Dahlinger `[一作]` (Karlsruhe Institute Of Technology), Gerhard Neumann `[通讯]` (Karlsruhe Institute Of Technology)

**通讯引用:** 10497 | [OpenAlex ID](https://openalex.org/A5110467801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种名为peach的框架，利用在上下文点云序列上进行的in‑context学习，使图网络模拟器能够在推理阶段快速适应未见过的材料属性并生成网格级动态预测。

**💡 创新点**

创新点在于：①将时间维度与空间坐标统一为4D点云，设计轻量级的spatio‑temporal点云编码器；②使用两种辅助监督（物理参数回归与SDF重建）来引导潜在空间学习；③在不需要网格观测的情况下实现零样本sim‑to‑real迁移。

**🔧 技术方法**

核心技术包括：Graph Network Simulators（MaNGO式轨迹级模拟器）、Transformer‑based 4D点云编码器、soft‑max聚合潜在向量、辅助回归和SDF损失、数据增强（噪声、遮挡、随机点）。

**📊 数据集**

在四个仿真场景（含2D/3D、弹性、粘弹性等物理属性）以及一个真实机器人落球弹力膜的实测数据集上进行评估；训练全部使用合成点云，不含真实点云。

**📈 对比分析**

与多种基线（点云编码器、网格编码器、oracle、无上下文）以及传统步骤模拟器相比，peach在仿真准确率上均优于或相当于网格基模型，甚至在某些任务中超过oracle；在真实场景中，peach在动态接触阶段的误差显著低于其他方法。

**⚠️ 局限性**

限制包括：需要预先获取初始网格几何；上下文轨迹必须与目标过程共享相同材料属性；单次全序列预测导致内存占用较高，难以处理极长时序或高分辨率网格。

---

## 419. STEAM: A Training-Free Congestion-Aware Enhancement Framework for Decentralized Multi-Agent Path Finding

**arXiv ID:** 2605.20929 | [PDF](https://arxiv.org/pdf/2605.20929v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 420. On the Complexity of Hop Domination and 2-Step Domination in Graph Classes

**arXiv ID:** 2605.20970 | [PDF](https://arxiv.org/pdf/2605.20970v1)

**作者:** Sandip Das `[一作]` (Indian Statistical Institute), Sk Samim Islam `[通讯]` (Indian Statistical Institute)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5038722857)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了图的 Hop Domination 与 2-Step Domination 问题，并证明它们在单位圆图、正则图（d≥3）以及无爪图等多种图类中仍为 NP‑完全问题

**💡 创新点**

创新点在于首次给出了这些特殊图类（尤其是单位圆图和正则图）中这两类支配问题的 NP‑完全性证明，并提供了相应的多种构造性归约

**🔧 技术方法**

主要技术是基于顶点覆盖的多种图变换与构造（如网格、路径、三角形、四元组等结构），以及精细的距离控制与支配关系分析

**📊 数据集**

论文仅作理论分析，没有使用实验数据集，所有结果均为证明性说明

**📈 对比分析**

由于是理论证明，没有实验对比；通过归约展示了问题的计算复杂度，未给出算法性能评估

**⚠️ 局限性**

局限在于只讨论了 NP‑完全性，没有给出近似或参数化算法，且构造过程较为繁琐，对实际应用的直接指导有限

---

## 421. Ark: Offchain Transaction Batching in Bitcoin

**arXiv ID:** 2605.20952 | [PDF](https://arxiv.org/pdf/2605.20952v1)

**作者:** Pim Keer `[一作]` (TU Wien), Zeta Avarikioti `[通讯]` (TU Wien)

**通讯引用:** 148 | [OpenAlex ID](https://openalex.org/A5006562415)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Ark，一个基于 Bitcoin 的 commit‑chain 协议，允许用户通过虚拟 UTXO（VTXO）实现离链交易，操作员只需在必要时参与即可。

**💡 创新点**

创新点在于：1）完全兼容 Bitcoin，不需要新的脚本功能；2）接收者可无 on‑chain 预留即可加入；3）只需与参与交易的用户交互即可完成状态更新；4）通过“reset 交易”解决原始实现中的 hostage 与 spam 攻击；5）引入可选的 fast‑finality 机制，在合理的经济假设下实现无需等到 on‑chain 确认即完成支付。

**🔧 技术方法**

核心技术包括：Taproot + Schnorr 多签（MuSig2）用于实现虚拟交易树（VTXT）与批量承诺；利用固定 nonce 的 Schnorr 签名实现私钥提取以惩罚 operator；通过聚合签名和锚点（anchor）实现批量交换与撤销；以及一个基于广播网络的 fast‑finality 协议，依赖于一个存在至少一名诚实签名者的签名委员会。

**📊 数据集**

实验数据基于开源实现（GitHub：ark4fish/ARK）在主网/测试网环境中进行，主要评估：承诺构造时间（线性增长，200 名用户约 2.7 秒）和 on‑chain 账本尺寸（承诺 197 vB，单一 VTXO 退出 150×⌈log n⌉+107 vB）。未使用传统数据集，评估依赖于模拟/真实区块链交易。

**📈 对比分析**

与 Lightning、Spark、CoinPool、Bitcoin Clique 等现有 Layer‑2 方案对比，Ark 的承诺大小常数（197 vB），协作退出成本常数，单边退出成本为 O(log n)。实验表明，在大批量状态更新时，Ark 能在极低的 on‑chain 费用下完成所有交易；而 Lightning 的单边退出成本随活跃 HTLC 数量线性增长；Spark 受 UTXO 深度影响；CoinPool 具有常数成本但不兼容 Bitcoin；Clique 的退出成本亦呈对数增长。

**⚠️ 局限性**

局限性包括：1）操作员中心化，单点失效导致性能下降；2）操作员需预先锁定大量流动性和 fast‑finality 的保证金；3）小额 VTXO 的单边退出费用可能较高；4）依赖存在至少一名诚实签名者的委员会，若委员会全被恶意掌控则安全性受损；5）在极端情况（银行跑、恶意矿工）下，退出延迟和 MEV 风险可能增加；6）当前实现对链下扩展和多运营商设计的支持有限。

---

## 422. Memory Grafting: Scaling Language Model Pre-training via Offline Conditional Memory

**arXiv ID:** 2605.20948 | [PDF](https://arxiv.org/pdf/2605.20948v1)

**作者:** Runxi Cheng `[一作]` (Tsinghua University), Yeyun Gong `[通讯]` (Microsoft Reasearch Asia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Memory Grafting方法，将冻结的迁移模型隐藏状态作为外部n-gram记忆，供更小的接收模型检索并注入；

**💡 创新点**

创新点在于将昂贵的记忆学习过程离线完成，利用已有预训练模型的高质量表示构建可扩展的外部记忆，保持O(1)检索复杂度；

**🔧 技术方法**

采用离线n-gram隐状态抽取、精确最长匹配检索、轻量投影与门控融合以及Engram哈希回退机制；

**📊 数据集**

在Nemotron-CC数据集上训练，规模分别为约0.9B和2.8B参数，使用LLaMA-3-8B分词器；

**📈 对比分析**

与MoE和vanilla Engram基线在相同预训练预算下对比，平均基准分数从MoE的51.95/Engram的52.43提升至53.86，单项任务均有提升；

**⚠️ 局限性**

局限包括：对高频n-gram的覆盖依赖，低频或未匹配情况需Engram回退；记忆质量受迁移模型表示层选择影响；大规模离线构建仍需算力。

---

## 423. The Knowledge Gap in a High-Choice Media Environment: Experimental Evidence from Online Search

**arXiv ID:** 2605.21019 | [PDF](https://arxiv.org/pdf/2605.21019v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 424. DAMA: Disentangled Body-Anchored Gaussians for Controllable Multi-Layered Avatars

**arXiv ID:** 2605.21001 | [PDF](https://arxiv.org/pdf/2605.21001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 425. Focus-then-Context: Subject-Centric Progressive Visual Token Reduction for Vision-Language Models

**arXiv ID:** 2605.20950 | [PDF](https://arxiv.org/pdf/2605.20950v1)

**作者:** Yulin Zhao `[一作]` (Harbin Institute of Technology), Zheng Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 78766 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SPpruner，一种以主体为中心的递进式视觉令牌压缩方法，模仿人类视觉的先聚焦后关注场景结构的感知流程。

**💡 创新点**

创新点在于：①将视觉显著性与语义相关性融合的焦点识别模块（FIM）精准捕获广泛视觉主体；②构造上下文感知结构扫描模块（CASSM），通过结构响应采样（SRS）动态调整采样步长，恢复主体的全局结构上下文；③无须额外训练，兼容多种VLM架构。

**🔧 技术方法**

技术方法包括：基于视觉显著性和跨模态注意力的得分函数；结构依赖度与语义对齐的上下文效用函数；结构响应采样策略；Lipschitz连续性理论给出的误差上界分析。

**📊 数据集**

在22个基准上验证，涵盖图像理解（LLaVA-1.5、LLaVA-Next、Qwen2.5-VL）、文档理解（TextVQA、DocVQA、OCRBench）、图像描述（Flickr30K）以及视频理解（LLaVA-OneVision）等任务。

**📈 对比分析**

与现有最优无训练压缩方法（ToMe、VisionZip、FastV、DART、PACT等）对比，SPpruner 在保持 90%+ 原模型性能的同时，显著提升速度（最高 2.53× 加速，FLOPs 减 64%），在高压缩比下性能下降仅 0.6% 左右。

**⚠️ 局限性**

局限性：对超高压缩比（如 10% 令牌）仍可能出现语义细节缺失；对多模态输入中的细粒度文本与视觉细节匹配需求仍需进一步提升；在纯文本任务需要额外的层级剪枝策略。

---

## 426. An IoT-Enabled Smart Home Automation System for Energy Efficiency with Web-Based Control

**arXiv ID:** 2605.20981 | [PDF](https://arxiv.org/pdf/2605.20981v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 427. JobArabi: An Arabic Corpus and Analysis of Job Announcements from Social Media

**arXiv ID:** 2605.20960 | [PDF](https://arxiv.org/pdf/2605.20960v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 428. DySink: Dynamic Frame Sinks for Autoregressive Long Video Generation

**arXiv ID:** 2605.21028 | [PDF](https://arxiv.org/pdf/2605.21028v1)

**作者:** Bo Ye `[一作]` (Southeast University), Min-Ling Zhang `[通讯]` (Southeast University)

**通讯引用:** 15738 | [OpenAlex ID](https://openalex.org/A5079083101)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DySink框架，利用检索式动态帧池替代静态帧槽，在自回归长视频生成中提供适时的长距离上下文，并加入异常门控以防止sink collapse。

**💡 创新点**

创新点包括：①动态检索历史关键-值作为长范围锚点；②轻量化交叉头一致性异常门控，实时抑制过度聚焦旧帧；③在保持原RoPE结构的同时实现更自适应的长时记忆控制。

**🔧 技术方法**

采用流动匹配的Diffusion Transformer（Wan2.1）、KV缓存、RoPE、LoRA微调、视觉编码器生成块描述、余弦相似检索、以及层级异常门控机制。

**📊 数据集**

训练使用VidProM扩展提示，评估使用MovieGen（128个样本）以及VBench/VBench‑Long；短时5s采用946个提示；长时50/75/100s使用MovieGen提示集。

**📈 对比分析**

与NOVA、Pyramid Flow、SkyReels‑V2、MAGI‑1、CausVid、Self‑Forcing、Self‑Forcing++、LongLive以及Bidirectional模型对比，DySink在5s总分最高，在50/75/100s视频中动态度提升20+点，时间质量与文本对齐均优于基线。

**⚠️ 局限性**

局限性：在最长100s、平滑摄像或渐进场景下表现最好；对更长、事件丰富、对象长时间消失/出现的场景尚未充分验证；视觉块描述忽略细粒度语义或文本依赖，未来需更强模型和细粒度检索机制。

---

## 429. A Sharper Picture of Generalization in Transformers

**arXiv ID:** 2605.20988 | [PDF](https://arxiv.org/pdf/2605.20988v1)

**作者:** Paul Lintilhac `[一作]` (Dartmouth), Sair Shaikh `[通讯]` (Dartmouth)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

研究Transformer在布尔域上对稀疏低阶函数的学习与泛化性能，提出构造性PAC‑Bayes方法，利用低锐度与低范数的Transformer实现可证明的泛化上界，并通过实验验证该理论；

**💡 创新点**

创新点包括：①提出针对Transformer的构造性PAC‑Bayes框架，直接构造可实现稀疏低阶傅里叶函数的Transformer并分析其锐度与范数；②将傅里叶稀疏度和最大度作为功能复杂度指标，给出以此为基的泛化上界；③证明链式思维（CoT）可将高阶Parity任务的泛化误差从指数降低到线性；④给出半解析的泛化界，兼顾理论严谨与实践可用；

**🔧 技术方法**

使用傅里叶‑Walsh展开、特定的注意力+MLP Transformer构造、锐度（Hessian trace）分析、PAC‑Bayes理论、随机高斯扰动分析、机制可解释性实验、链式思维实验；

**📊 数据集**

实验采用生成的随机稀疏傅里叶布尔函数（正系数、统一度数、稀疏度≤T）以及Parity任务；训练样本数为8192；多组度数（1-5）与稀疏度（1-20）进行评估；

**📈 对比分析**

与传统基于范数的覆盖数泛化界对比；实验表明半解析界在合理参数范围内非空，训练误差与理论上界一致；CoT实验显示错误率相比单步显著下降，证明了理论预测的线性优势；

**⚠️ 局限性**

局限性：①扰动项P(σ)的上界过于保守，实际值远小；②理论仅适用于正系数、单一度数、稀疏度≤T的布尔函数；③假设低锐度插值器存在且优于学习得到的解，尚未在所有情形下证明；④实验规模受限，未覆盖更大网络或复杂函数；

---

## 430. Toward 6G-enabled Brain Computer Interfaces: Technical Requirements, Use Cases, Challenges, and Future Trends

**arXiv ID:** 2605.20939 | [PDF](https://arxiv.org/pdf/2605.20939v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 431. Privacy-Preserving Distributed Optimization Under Time Constraints Using Secure Multi-Party Computation and Evolutionary Algorithms

**arXiv ID:** 2605.20944 | [PDF](https://arxiv.org/pdf/2605.20944v1)

**作者:** Sebastian Gruber `[一作]` (Johannes Kepler University Linz), Thomas Lorünser `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种在时间限制下进行隐私保护的分布式优化方法，该方法结合了进化算法和安全多方计算（MPC）来评估解决方案。

**💡 创新点**

创新点在于通过选择性地使用MPC来评估解决方案，从而减少隐私保护计算对运行时间的影响，并允许在截止日期内返回解决方案。

**🔧 技术方法**

使用了进化算法（如遗传算法和NSGA-II）和安全多方计算（MPC）技术。

**📊 数据集**

使用了平衡的单目标分配问题（AP）、对称的单目标旅行商问题（TSP）和多目标分配问题（MOAP）作为数据集。

**📈 对比分析**

通过与完全隐私保护的优化算法进行比较，结果表明该方法在运行时间上有显著改善，能够在时间限制内找到有效的解决方案。

**⚠️ 局限性**

局限性在于隐私保护计算可能导致解决方案质量的潜在折衷，尤其是在使用启发式优化和模糊化方法时。

---

## 432. PaintCopilot: Modeling Painting as Autonomous Artistic Continuation

**arXiv ID:** 2605.20941 | [PDF](https://arxiv.org/pdf/2605.20941v1)

**作者:** Yunge Wen `[一作]` (MIT Media Lab), Paul Pu Liang `[通讯]` (MIT Media Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 PaintCopilot，一种协同式神经绘画助手，能够在无目标图像的条件下，从画布当前状态和之前的笔触历史中自动生成下一笔触，并支持多种交互工作流；

**💡 创新点**

创新点包括：① 把绘画建模为开放式自回归过程，摒弃传统的目标图像重建；② 设计三种互补模型——基于 ViT 的目标预测器、通过流匹配的自回归笔触预测器以及 VAE 区域采样器；③ 实现可微分刷子渲染与实时优化，使人机协作更加流畅；

**🔧 技术方法**

技术手段涵盖：Vision Transformer+Adaptive Layer Normalization 对画布隐空间进行目标预测；流匹配（Flow Matching）框架生成多样化笔触；VAE+Transformer 生成区域化笔触序列；可微分高斯刷子、硬圆/笔尖刷子；基于 Stable Diffusion VAE 的隐空间；以及实时差分优化与梯度裁剪等；

**📊 数据集**

使用了 3,000 古典肖像画的人工合成数据集，每幅画配有 300–400 条笔触序列，笔触顺序通过语义分割、法线图与 DINO 注意力引导得到；

**📈 对比分析**

评价方法：在 100 幅保留测试画上评估目标预测器的 LPIPS/SSIM/PSNR，随着绘制进度提升；笔触预测器以平均 L1 误差和维度分解评估；区域采样器评估重建误差与多样性；实验显示目标预测器随进度提升表现稳定，笔触预测误差在 0.07–0.14 范围内，区域采样器保持低误差与高多样性；

**⚠️ 局限性**

局限性：长时间自回归生成易出现语义漂移，尤其在细节丰富区域；目前仅在肖像绘制场景验证，扩展到其他艺术风格需更丰富的数据；缺乏长程规划与全局一致性机制，难以完全替代人工创作；

---

## 433. Partially Observable Restless Bandits for Age-Optimal Scheduling over Markov Channels

**arXiv ID:** 2605.21016 | [PDF](https://arxiv.org/pdf/2605.21016v1)

**作者:** Xijun Wang `[一作]` (Sun Yat-sen University), Xiang Chen `[通讯]` (Sun Yat-sen University)

**通讯引用:** 37841 | [OpenAlex ID](https://openalex.org/A5100641667)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了在多设备物联网系统中，如何在时变Markov信道且调度前无法获知即时信道状态的前提下，利用部分可观测RMAB模型最小化系统平均年龄（AoI）并实现高效调度。

**💡 创新点**

创新点包括：① 将AoI与信道信念联合建模，证明子问题阈值结构与指数可行性；② 对α=1-β情形给出Whittle指数闭式；③ 推导低复杂度Whittle‑like指数并提供快速求索算法；④ 在大规模网络与资源匮乏场景下实现近似最优调度。

**🔧 技术方法**

使用技术包括：部分可观测RMAB、Lagrangian松弛、Whittle指数理论、POMDP→MDP映射、阈值策略证明、RVI与阈值结构加速、闭式Whittle-like指数推导以及仿真验证。

**📊 数据集**

实验采用模拟数据，设置不同的信道转移概率α、β以及设备数M、带宽K，未使用公开真实数据集。

**📈 对比分析**

通过与贪心、期望即刻奖励（myopic）、Round‑Robin和小规模最优策略对比，结果显示Whittle及Whittle‑like指数策略在接近理论下界、尤其在K小或M大时性能显著优于基线。

**⚠️ 局限性**

局限性：仅考虑单一长度的更新包、假设Markov信道参数已知且平稳、离线预先计算指数对实时变化不敏感，且未探讨能量约束、多优先级或学习增强的情况。

---

## 434. DASH: Fast Differentiable Architecture Search for Hybrid Attention in Minutes on a Single GPU

**arXiv ID:** 2605.20936 | [PDF](https://arxiv.org/pdf/2605.20936v1)

**作者:** Weizhe Chen `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 29935 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DASH框架，利用可微分搜索快速设计混合注意力架构

**💡 创新点**

将层级注意力选择转化为可微分的连续路由，只优化架构logits并冻结模型权重，显著降低搜索成本

**🔧 技术方法**

可微分架构搜索、教师对齐线性候选、软路由与成本正则化、温度退火、教师保留KL损失

**📊 数据集**

使用Qwen2.5-3B-Instruct为基模型，在DCLM通用文本语料上进行Stage 1/3训练与Stage 2搜索

**📈 对比分析**

与多种手工规则、代理选择器（GA‑S2等）及Jet‑Nemotron NAS搜索结果比较，DASH在RULER长序列检索上实现最高分，且搜索仅需12.3 M token、约20 min，成本比Jet‑Nemotron低3.3 billion倍

**⚠️ 局限性**

仅针对层级混合选择；只在单一模型族与规模上验证；搜索后模型训练仍相对较少；成本正则化仅为经验式代理，未包含硬件特定指标

---

## 435. MemConflict: Evaluating Long-Term Memory Systems Under Memory Conflicts

**arXiv ID:** 2605.20926 | [PDF](https://arxiv.org/pdf/2605.20926v1)

**作者:** Zhen Tao `[一作]` (Renmin University of China), Zhiyu Li `[通讯]` (MemTensor Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MemConflict 框架，用来评估长短期记忆系统在面对动态、静态和条件三类冲突时的检索、排序与最终答案生成能力。

**💡 创新点**

创新点在于：①系统化定义三种冲突类型并构建对抗性多轮对话基准；②结合黑盒与白盒评估，既检验答案正确性，又诊断记忆检索与排名；③提出多维度指标（AA、SEH@K、SRS、UOCS、CRS）与可解释的诊断视角。

**🔧 技术方法**

技术手段包括：使用 LLM 生成用户画像、对话和查询；基于检索增强的 LLM（RAG）实现记忆检索；实现多级评估协议和自动化的指标计算；通过可解释的白盒接口提取检索结果与排名。

**📊 数据集**

数据集：MemConflict 12 个虚拟用户，每个用户约 52 轮对话、2349 句、约 20 万 token，覆盖 90+ 动态冲突、约 17 静态冲突、约 17 条件冲突，并注入多种语义相似干扰者。

**📈 对比分析**

比较方法：在相同的查询集与评估协议下，对六个公开的长记忆系统（A-Mem、LangMem、Letta、MemOS、Mem0、Memobase）进行黑盒 AA 与白盒 SEH@3/SRS 的对比。实验显示 MemOS 在大多数冲突类型下表现最佳，LangMem 在动态冲突上突出，而其他系统在条件或静态冲突上表现相对薄弱。

**⚠️ 局限性**

局限性：①数据为合成模拟，缺乏真实人类对话的多样性；②仅涵盖三类冲突，未覆盖更复杂的多模态、跨语言或策略性信息缺失场景；③评估仅针对公开系统，无法直接适用于闭源或部署环境中的记忆模块。

---

## 436. Strategy-Induct: Task-Level Strategy Induction for Instruction Generation

**arXiv ID:** 2605.20924 | [PDF](https://arxiv.org/pdf/2605.20924v1)

**作者:** Po-Chun Chen `[一作]` (National Taiwan University), Hsin-Hsi Chen `[通讯]` (National Taiwan University)

**通讯引用:** 7192 | [OpenAlex ID](https://openalex.org/A5000334344)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种仅基于问题的任务级指令诱导框架 Strategy-Induct，能在缺少标注答案的情况下生成有效指令。

**💡 创新点**

创新点在于使用策略生成替代标注答案，借助少量问题生成策略对来诱导通用任务指令，并实现跨模型、跨规模的泛化。

**🔧 技术方法**

采用多阶段提示技术：先用LLM生成每个问题的推理策略，再用策略-问题对诱导任务级指令，最后用诱导指令进行推理。

**📊 数据集**

在 BBH-Induct、Evals-Induct 和 Shift Cipher 三个数据集上进行实验。

**📈 对比分析**

与 ZCoT、SCoT、INDUCT 等基线相比，Strategy-Induct 在 60 个设置中大多数实现了更高准确率，尤其在小模型和 LRM 上提升显著。

**⚠️ 局限性**

局限性包括对强大指令跟随模型的依赖、跨模型/数据集泛化受限、仅适用于任务级重复性问题，且实验依赖 API 访问的模型。

---

## 437. ROAR-3D: Routing Arbitrary Views for High-Fidelity 3D Generation

**arXiv ID:** 2605.21121 | [PDF](https://arxiv.org/pdf/2605.21121v1)

**作者:** Hanxiao Sun `[一作]` (Hong Kong University of Science and Technology), Wenhan Luo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 7962 | [OpenAlex ID](https://openalex.org/A5004450394)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

将预训练的单视图3D生成模型升级为支持任意数量无姿态多视图输入的ROAR-3D，显著提升生成3D几何的精度与完整性。

**💡 创新点**

创新性地提出 token‑wise view router、dual‑stream attention 与 orientation perturbation 三大组件，解耦姿态控制与几何传递，使模型无需姿态或外部重建模块即可进行多视图条件生成。

**🔧 技术方法**

结合预训练的 Diffusion Transformer（DiT）与 3DShape2VecSet VAE、DINOv2 视觉特征、Gumbel‑Softmax 路由、双流交叉注意力以及第二阶段的 LATTICE 细化网络，形成完整的轻量多视图生成体系。

**📊 数据集**

在约 30 万对象的 Objaverse/ObjaverseXL 过滤子集上进行训练，并在 Anyview‑200（包含 200 个多视图真实与合成图像）上进行泛化与鲁棒性评估。

**📈 对比分析**

与单视图、传统重建以及其他多视图条件生成方法对比，ROAR‑3D 第一阶段已在 CD、F1、ULIP‑I、Uni‑I 上超过 ReconViaGen 等基线；第二阶段细化后 CD 下降至 21.039，F1(0.05) 提升至 81.6，展示显著性能提升。

**⚠️ 局限性**

仍依赖预训练的单视图模型，极少视图或极端视角差异下性能可能受限；训练成本较高（约 3–4 GPU 天 + 10 GPU 天），未充分评估光照或材质变化对鲁棒性的影响。

---

## 438. Image Encryption via Data-Identified Discrete Chaotic Maps

**arXiv ID:** 2605.21118 | [PDF](https://arxiv.org/pdf/2605.21118v1)

**作者:** Wenyuan Lia `[一作]`, Li Zhang `[通讯]` (Lanzhou University of Technology)

**通讯引用:** 58114 | [OpenAlex ID](https://openalex.org/A5100425709)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于SINDy-PI从数据自动识别混沌映射的图像加密框架。

**💡 创新点**

通过数据驱动的混沌映射学习实现隐式密钥，提升密钥空间与抗攻击性。

**🔧 技术方法**

采用SINDy-PI算法进行离散系统识别、混沌序列生成以及行列置乱+模256扩散。

**📊 数据集**

使用合成的Hénon、三维Logistic和Lozi映射训练数据，以及256×256灰度图“Moon Surface”进行测试。

**📈 对比分析**

与固定参数混沌加密方案对比，NPCR≈99.6%、UACI≈33.5%、熵≈7.997、相关系数≈0，均逼近理论理想值。

**⚠️ 局限性**

仅适用于离散混沌映射，需先公开映射形式；对噪声和真实物理信号的鲁棒性尚未充分验证。

---

## 439. Automated Byzantine-Resilient Clustered Decentralized Federated Learning for Battery Intelligence in Connected EVs

**arXiv ID:** 2605.21115 | [PDF](https://arxiv.org/pdf/2605.21115v1)

**作者:** Mouhamed Amine Bouchiha `[一作]` (Télécom SudParis), Yacine Ghamri-Doudane `[通讯]` (La Rochelle University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 ABC-DFL——一种面向电动汽车电池智能的拜占庭容错聚类去中心化联邦学习框架，并实现了完整的区块链+oracle生态系统。

**💡 创新点**

核心创新包括：动态声誉驱动的 QBFT 共识协议、双层过滤聚合协议 FLECA（EV 侧参考过滤 + oracle 层 HDBSCAN 聚类），以及基于区块链的阈值签名与内容寻址，实现了去中心化、可验证且对拜占庭攻击高度鲁棒。

**🔧 技术方法**

采用了：去中心化联邦学习、区块链（Open‑permissioned + Dynamic QBFT）、去中心化 oracle 网络、阈值签名、差分隐私（DP）、HDBSCAN 密度聚类、FedAvg/FedProx 等基础聚合方法。

**📊 数据集**

使用真实电动车电池诊断数据集 “EVBattery”，在多任务模型（BiLSTM、CNN 等）下进行异常检测与容量估计。

**📈 对比分析**

与 FedAvg、FedProx、Multi‑Krum、FLAME、UBAR 等多种聚合方案对比；在无攻击场景下与 FedProx 对齐；在多种 Poisoning 与 Backdoor 攻击（Gauss、Krum、Trim、Label‑Flip、Feature、Adaptive、BadNets、Scaling、Neurotoxin）下，AIS 与 ASR 均低于 0.1，显著优于对比基线；在区块链层面吞吐 60‑70 tx/s、单轮耗时 4‑8 s，低于纯中心化 FL。

**⚠️ 局限性**

限制：需要预先建立许可式共识委员会，拜占庭比例需低于 50%；DP 参数需权衡隐私与性能；实验规模受限于 245 台车辆，未验证更大规模部署；在极端网络分区或高 churn 时性能可能下降。

---

## 440. Musical Attention Transformer: Music Generation Using a Music-Specific Attention Model

**arXiv ID:** 2605.21081 | [PDF](https://arxiv.org/pdf/2605.21081v1)

**作者:** Shinnosuke Taksuka `[一作]` (Meiji University), Hideo Mukai `[通讯]` (Meiji University)

**通讯引用:** 4169 | [OpenAlex ID](https://openalex.org/A5011157909)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种名为Musical Attention的Transformer注意力机制，用于在音乐生成中融合元信息（小节数、调性、节拍）。

**💡 创新点**

创新点在于将元信息直接嵌入注意力机制，并设计两种专门关注前缀和同属性音符的注意模式，从而提升音乐结构一致性与多样性。

**🔧 技术方法**

采用Transformer编码器（8层，512维，8头）结合相对位置编码、稀疏注意和温度采样；并在预训练阶段使用掩码语言模型。

**📊 数据集**

使用Lakh MIDI Dataset（约45k首MIDI），经预处理后得到单轨和多轨序列进行训练。

**📈 对比分析**

与Full Attention和Strided Attention比较，Musical Attention在单轨和多轨的Bar Error、Key Error、Token Error等指标均表现最优，训练准确率相近。

**⚠️ 局限性**

局限在于生成的动态缺乏变化、和弦进程偶尔不自然，且未充分利用和弦信息或调性变化。

---

## 441. VDFP: Video Deflickering with Flicker-banding Priors

**arXiv ID:** 2605.21079 | [PDF](https://arxiv.org/pdf/2605.21079v1)

**作者:** Zhiyi Zhou `[一作]` (Shanghai Jiao Tong University), Xiaokang Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 26759 | [OpenAlex ID](https://openalex.org/A5019708391)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种面向视频去闪烁（deflickering）的新框架VDFP，能够有效消除由于屏幕与摄像机同步不匹配导致的周期性亮度条纹。

**💡 创新点**

创新点包括：1) 通过滚动快门机制的退化场模型DFM，生成多层次、多频率的真实感闪烁合成数据；2) 引入连续时空先验感知CPP模块，利用FA‑MSE学习细腻的条纹置信图；3) 采用零初始化条件注入，将先验信息无缝融入预训练扩散模型，避免分布漂移。

**🔧 技术方法**

主要技术包括：滚动快门退化场建模、Swin‑UNet+FA‑MSE连续置信图预测、零初始化5通道条件注入的扩散模型（STAR基础），以及对齐与色彩校正流程。

**📊 数据集**

使用了两类数据集：1) 实际拍摄的DeViD视频数据集，涵盖多场景LED显示屏录像；2) 通过DFM合成的多波段闪烁数据，用于预训练与微调。

**📈 对比分析**

与DLoRAL、FPANet、Flickerformer等多种视频恢复模型对比，VDFP在SSIM、LPIPS、VMAF、DOVER等指标上均领先，且视觉上能完全抹除厚弱两种条纹并保持细节与时序一致。

**⚠️ 局限性**

局限性在于DeViD仅包含LED矩阵屏幕的闪烁场景，未覆盖LCD、OLED等其他显示技术，未来需要扩大硬件多样性和跨平台泛化验证。

---

## 442. Cross-lingual robustness of LLM-brain alignment and its computational roots

**arXiv ID:** 2605.21049 | [PDF](https://arxiv.org/pdf/2605.21049v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 443. Towards Understanding Self-Pretraining for Sequence Classification

**arXiv ID:** 2605.21070 | [PDF](https://arxiv.org/pdf/2605.21070v1)

**作者:** Omar Coser `[一作]` (Università Campus Bio-Medico di Roma), Antonio Orvieto `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了Transformer序列分类中自预训练（SPT）对优化与性能提升的机制，探索了SPT如何通过学习接近偏置的Attention模式来弥补标签监督难以学习有效Attention映射的瓶颈。

**💡 创新点**

创新点在于发现并证明SPT能够通过重构目标学习到近邻偏置的Attention结构，显著提升Transformer在LRA任务中的训练可行性，并用简化理论解释标签监督在均值池化损失下对某些Attention方向不可感知的原因。

**🔧 技术方法**

主要技术包括Transformer架构、掩码token重构预训练、深度与数据源的系统消融、对比不同位置编码（ALiBi、RoPE、FIRE）的实验，以及基于Attention矩阵的理论分析。

**📊 数据集**

使用的数据集包括Long‑Range Arena基准中的CIFAR‑10、PathFinder、ListOps、Retrieval、Text等任务，以及自定义的10k长度二分类合成任务。

**📈 对比分析**

方法上与从零训练进行对比，利用相同的超参、训练预算，并通过验证集峰值准确率衡量。实验显示SPT+微调在多数任务上提高约5–20%准确率，尤其在深度变化和数据源交换时仍能保持显著收益。

**⚠️ 局限性**

局限性在于大部分消融仅在少数任务（CIFAR‑10、PathFinder）与单种子实验，未覆盖全部LRA任务；理论简化未考虑层归一化、嵌入层、深度影响等因素，且对大型模型与多任务的计算资源需求高。

---

## 444. A Dialogue between Causal and Traditional Representation Learning: Toward Mutual Benefits in a Unified Formulation

**arXiv ID:** 2605.21058 | [PDF](https://arxiv.org/pdf/2605.21058v1)

**作者:** Yan Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Guangyi Chen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 4504 | [OpenAlex ID](https://openalex.org/A5088257057)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出统一的表示学习框架，将任务组件和约束组件分离，并探讨了因果约束与任务目标的相互作用。

**💡 创新点**

创新点在于将因果表示学习与传统表示学习通过任务‑约束视角桥接，揭示了任务选择对因果约束效果的关键影响。

**🔧 技术方法**

使用了基于视图生成、重建、对比学习等多种任务目标，并结合条件先验、稀疏机制等因果约束来学习表示。

**📊 数据集**

实验采用CausalVerse合成数据集，分别在图像和视频两种场景下进行评估。

**📈 对比分析**

通过将不同任务目标与相同因果约束组合，发现对比学习能获得最高的MCC和R²，说明任务‑约束匹配显著提升性能。

**⚠️ 局限性**

局限性在于对任务‑约束组合的分析主要依赖实验观察，缺乏理论指导如何系统选择合适的组合。

---

## 445. On Unified and Sharpened CMI Bounds for Generalization Errors

**arXiv ID:** 2605.21056 | [PDF](https://arxiv.org/pdf/2605.21056v1)

**作者:** Yang Lu `[一作]` (University of Melbourne), Jingge Zhu `[通讯]` (University of Melbourne)

**通讯引用:** 530 | [OpenAlex ID](https://openalex.org/A5007978311)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种新的 leave‑m‑out (LmO) 超样本框架，并在此框架下推导统一的条件互信息（CMI）泛化误差上界；

**💡 创新点**

通过引入可调参数 m 与 k 统一复现并改进已有的 MI/CMI 上界，提出更精确的 IPCIMI、SIPCIMI、LOFO‑CMI 等新界；

**🔧 技术方法**

使用信息理论工具（互信息、条件互信息、二元 KL/JS 散度）、CGF 估计、数据处理不等式以及离散化/分解技巧；

**📊 数据集**

在论文中仅使用合成数据进行实验，例如 Bernoulli、Gaussian、有限假设空间等典型统计学习示例；

**📈 对比分析**

将新界与传统 MI、IMI、ICIMI、LOO‑CMI 等结果在上述示例上做数值比较，结果显示 LmO‑CMI 与 IPCIMI/SIPCIMI 等界在绝大多数情形下更紧，尤其在 m≫n 时显著优于现有界；

**⚠️ 局限性**

局限性包括：仍需对离散化或 CGF 进行估计，计算量随 m 增大而提升；对非分布式或高维实际数据的可扩展性未充分验证；界限仍依赖于损失函数的有界性或可微性等假设。

---

## 446. Modeling and Control of a Pneumatic Morphing Soft Quadrotor based on the SOFA Framework for Dynamic Soft Robotic Simulation

**arXiv ID:** 2605.21031 | [PDF](https://arxiv.org/pdf/2605.21031v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 447. UOTIP: Unbalanced Optimal Transport Map for Unpaired Inverse Problems

**arXiv ID:** 2605.21094 | [PDF](https://arxiv.org/pdf/2605.21094v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 448. The Quiet Path from Seemingly Minor Design Errors to Workplace AI Incidents

**arXiv ID:** 2605.21035 | [PDF](https://arxiv.org/pdf/2605.21035v1)

**作者:** Julia De Miguel Velázquez `[一作]` (King's College London), Daniele Quercia `[通讯]` (Nokia Bell Labs)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用大型语言模型和问卷调查，对 214 条职场 AI 事故进行特征匹配分析，量化 AI 系统与工人需求的不一致对事故的影响。

**💡 创新点**

首次将 AI 特征不匹配框架与实际事故数据结合，揭示 83% 的事故源自特征不匹配，并发现 74% 的不匹配源自开发者的设计偏差。

**🔧 技术方法**

使用 GPT‑5 进行文本分类、特征提取，并通过 Cohen’s κ 评估标注可靠性；结合统计分析和可视化技术展示结果。

**📊 数据集**

AI Incident Database（2013‑2025）1,256 条条目，筛选 214 条职场事故；O*NET 任务集合；202 名工人和 197 名开发者的偏好问卷。

**📈 对比分析**

通过与工人偏好差距阈值 0.5 判断不匹配；计算不匹配占 83.6% 的事故，开发者与工人偏好不一致占 74%；Kappa 在 0.85–0.97 范围内，显示高标注一致性。

**⚠️ 局限性**

主要局限包括：数据来源多为新闻报道，可能忽略日常细微事故；任务级别简化未覆盖非正式工作；因果推断依赖文本解释，缺乏实验验证；样本主要来自美国，文化差异未充分考虑。

---

## 449. LoCar: Localization-Aware Evaluation of In-Vehicle Assistants through Fine-Grained Sociolinguistic Control

**arXiv ID:** 2605.21086 | [PDF](https://arxiv.org/pdf/2605.21086v1)

**作者:** Seogyeong Jeong `[一作]` (KAIST), Alice Oh `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并验证了一套针对韩语车载助手的本地化评估框架 LOCAR，定义了13项关键绩效指标（KPI），涵盖细粒度敬语控制、对话连贯性、主动澄清等能力，并通过自动化流水线对多轮与单轮对话进行评估。

**💡 创新点**

创新点包括：① 针对车载领域的本地化评估体系；② 细粒度敬语KPI与混合形态学+LLM判定方法；③ 关注战略交互（澄清、主动）而非单纯事实准确性；④ 通过人工黄金集校准的LLM-as-judge实现高效、可复现的多维度评估。

**🔧 技术方法**

技术手段：LLM-as-judge（基于三款LLM的投票机制）、形态学敬语检查、人工标注黄金集、自动化评估流水线、合成与迁移QA数据、对比基准模型评测。

**📊 数据集**

数据集：以车主手册与导航手册为基础生成的单轮和多轮韩语QA数据，覆盖13个KPI；共803个人工双人标注的黄金样本；所有问答均提供韩语与英文翻译版本。

**📈 对比分析**

比较方法：对11个模型（含韩语与全球API）在单/多轮车载场景下随机抽取50条测试实例，评估所有KPI；LLM-as-judge与形态学验证相结合。结果显示韩语敬语细粒度控制不稳定，澄清、主动等战略指标表现低且方差大；整体评估平均耗时约90秒；形态学+LLM组合将敬语准确率从0.69提升至0.94。

**⚠️ 局限性**

局限性：① 仅针对韩语，敬语检测依赖韩语形态学规则，无法直接迁移至其他语言；② 评估在离线文本环境，未考虑ASR/TTS错误、真实语音延迟和实时交互约束；③ 关键指标在高端模型下已接近饱和，实际部署中动态上下文、资源限制可能导致性能下降。

---

## 450. Anomaly-Informed Confidence Calibration for Vision-Based Safety Prediction

**arXiv ID:** 2605.21109 | [PDF](https://arxiv.org/pdf/2605.21109v1)

**作者:** Zhenjiang Mao `[一作]` (University of Florida), Ivan Ruchkin `[通讯]` (University of Florida)

**通讯引用:** 491 | [OpenAlex ID](https://openalex.org/A5021509994)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种在线、无训练的异常感知校准框架，结合视觉与动力学异常评分，对自动驾驶赛车中的视觉安全预测器进行自适应置信度校准。

**💡 创新点**

创新点在于：① 将世界模型的重建误差和动力学不确定性（通过马氏距离提取的特征）融合为两个分离的异常评分；② 将校准温度动态地依赖于这两种异常评分，并辅以测试时增强（TTA）实现对视觉失真与动力学失效的双重鲁棒；③ 通过合成扰动学习校准器，使其能够迁移到未见的真实世界异常而无需重新训练。

**🔧 技术方法**

技术手段包括：VAE+ConvLSTM 世界模型、重建误差与马氏距离异常评分、测试时图像增强、基于异常评分的温度缩放校准、以及使用多域/合成数据进行校准器训练。

**📊 数据集**

实验数据来自物理 DonkeyCar 平台：在场景内的 25 条驾驶序列构成 ID 数据，使用 ImageNet-C 的多种噪声和自定义动力学扰动生成合成训练集，并在四种真实世界 OOD 协议（暗光、模糊、偏置、延迟）上评估。

**📈 对比分析**

与温度缩放、Platt、Isotonic、Histogram Binning、DAC 等基线对比，平均 ECE 从 0.184 降至 0.116，提升 37%，同时 OOD 检测 AUROC 约 0.83，显著优于传统置信度或特征距离方法。

**⚠️ 局限性**

局限性包括：校准器依赖于合成扰动的结构假设（单调性与近似充分性），对极端或未覆盖的失真严重度迁移效果未知；仅在离线训练后采用静态校准，未实现在线更新；且未在闭环控制实验中验证最终安全性能。

---

## 451. Dynamic Video Generation: Shaping Video Generation Across Time and Space

**arXiv ID:** 2605.21042 | [PDF](https://arxiv.org/pdf/2605.21042v1)

**作者:** Shikang Zheng `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15548 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种动态视频生成框架DVG，能够根据视频内容自动分配空间与时间的计算资源，从而在保持质量的前提下显著加速视频生成过程。

**💡 创新点**

①联合空间‑时间加速视角，解决单维压缩导致质量下降的问题；②利用早期粗略潜在草图估计空间细节与运动需求，实现内容感知的压缩策略；③采用anchor‑based潜在重塑和坐标哈希重噪，支持任意分辨率与帧率的灵活压缩；④预定义动作集合与预算过滤相结合，自动匹配最优压缩方案。

**🔧 技术方法**

FFT频谱分析、光流估计、anchor‑based潜在重塑、坐标哈希生成噪声、预算过滤与需求匹配优化、结合step‑distillation、DPM‑solver、Sparse Attention等技术。

**📊 数据集**

使用VBench（946条文本‑视频提示）和VBench‑I2V（1118条提示）进行评测，并在HunyuanVideo、Wan2.2和HunyuanVideo‑1.5等多种模型上验证。

**📈 对比分析**

与单维压缩、Sparse Attention、DPM‑solver、distillation等方法对比，DVG在保持或提升VBench总分（80+）、I2V PSNR（>23）和LPIPS（<0.1）的同时，实现3–7×的加速；结合蒸馏可达18×速度提升。

**⚠️ 局限性**

仍需依赖预先定义的压缩动作集合，对极高分辨率或极长时序视频可能需要更细粒度的调优；在极端细节或短时序场景下，内容估计误差可能导致质量波动；当前实现主要在软件层面，未针对特定硬件做深度优化。

---

## 452. Backchaining Loss of Control Mitigations from Mission-Specific Benchmarks in National Security

**arXiv ID:** 2605.21095 | [PDF](https://arxiv.org/pdf/2605.21095v1)

**作者:** Matteo Pistillo `[一作]` (Apollo Research), Joshua Herman `[通讯]` (U.S. Department of the Treasury)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于已有任务特定基准的后向推理方法，用来识别和限制 AI 系统在国防安全部署中可能导致失控（LoC）的可操作性和权限，从而实现对 LoC 的主动预防。

**💡 创新点**

创新点在于：①将基准错误答案作为探测 LoC 风险的切入点；②通过对错误答案进行后向链回，系统性识别出触发 LoC 的可操作性和权限；③提出针对性干预措施，实现对错误行为的限制，同时保持正确行为的可行性。

**🔧 技术方法**

技术上主要采用：结构化威胁建模、权限与可操作性映射、后向链推理（backchaining）以及最小特权原则的实现方法。

**📊 数据集**

使用了国防安全领域的示例基准——一个关于衍生安全分类的多选题，涵盖了手写笔记、网络层、分类指南等情境数据。

**📈 对比分析**

与传统的前置评估或后置监测方法相比，该方法无需完整绘制权限地图即可快速识别风险路径；在示例中，作者通过移除特定权限（如编辑注释、生成未经审查摘要）成功阻止了错误选项（A–C）的实施，保持了正确选项（D）的可执行性，表现出有效的风险削减效果。

**⚠️ 局限性**

局限性包括：①依赖基准的代表性与完整性，若基准存在偏差、缺失或不现实，则可能漏检风险；②无法覆盖所有可能的 LoC 场景，仅能针对已知错误回答进行干预；③对实际部署环境的技术实现和治理成本仍需进一步验证。

---

## 453. TextSculptor: Training and Benchmarking Scene Text Editing

**arXiv ID:** 2605.21090 | [PDF](https://arxiv.org/pdf/2605.21090v1)

**作者:** Yiheng Lin `[一作]` (Beijing Jiaotong University), Yujie Zhong `[通讯]` (Bytedance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TextSculptor 框架，包括自动化的数据构建管道、规模化的 TextSculpt-Data 数据集以及面向场景文字编辑的 TextSculpt-Bench 基准。

**💡 创新点**

创新点在于：①结合 VLM 进行语义重写、图像合成与 OCR 质量门控，生成高质量文本-图像对；②使用程序化文本渲染与拼接生成对齐的编辑样本，保证文字准确性和背景一致性；③设计多维度评估协议（文本准确性、视觉质量、背景保持），并通过 OCR 对齐和多模态判断实现细粒度评测。

**🔧 技术方法**

使用的技术包括：Qwen3-VL 语义重写、强大图像生成模型、PaddleOCR 质量筛选、Python 渲染引擎、SSIM 背景相似度、GPT‑5.2 多模态评判；模型基于 Qwen‑Image‑Edit‑2511 并采用 LoRA 微调。

**📊 数据集**

构建的数据集为 TextSculpt-Data，包含 1.2M OCR 验证的文本‑图像样本和 2M 程序化生成的编辑对，合计 3.2M 训练样本；基准 TextSculpt‑Bench 采样 800 张图像并生成 200 例四种编辑任务（加、换、删、混合）。

**📈 对比分析**

在 TextSculpt‑Bench 上，TextSculptor 通过单轮 LoRA 微调实现了文本准确率、视觉质量和背景保持的综合得分 0.69，背景保持最高 0.78，显著优于公开模型并缩小与专有系统的差距。

**⚠️ 局限性**

局限性包括：数据主要来自合成场景，缺乏真实多样性；评测仍依赖 GPT‑5.2 主观判断，可能存在偏差；基准仅覆盖四类基本编辑任务，未涵盖更复杂的文字编辑需求。

---

## 454. An Evidence-driven Protocol for Trustworthy CI Pipelines

**arXiv ID:** 2605.21089 | [PDF](https://arxiv.org/pdf/2605.21089v1)

**作者:** Fernando Castillo `[一作]` (TU Berlin), Stefan Tai `[通讯]` (TU Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于确定性构建系统与可信执行环境的可信 CI 流水线协议。

**💡 创新点**

创新点在于将确定性构建（DBS）与 TEE 远程证明结合，形成单一可验证的证据链，消除消费者重复重建的需要。

**🔧 技术方法**

使用技术包括 GitLab、Argo Workflows、Kubernetes、NixOS/Nix、Intel TDX、TEE 远程证明、区块链（Ethereum Attestation Service）等。

**📊 数据集**

实验使用了三个真实开源项目（aave/interface、ChainSafe/ChainBridge、smartcontractkit/chainlink）以及对应的 SBOM 与安全扫描工具 Trivy。

**📈 对比分析**

通过对比开启与关闭可信机制的 10 次运行，工作流时间、CPU/内存开销均保持在同一数量级，且消费者端验证成本显著降低，TEE 开销被多消费者摊销。

**⚠️ 局限性**

局限性包括对 TEE 硬件与 PKI 的依赖、区块链交易成本、证据可重现性受时间戳等非确定性输入影响，以及大规模并发流水线对 TEE 资源的竞争。

---

## 455. NanoCP: Request-Level Dynamic Context Parallelism for Data-Expert Parallel Decoding

**arXiv ID:** 2605.21100 | [PDF](https://arxiv.org/pdf/2605.21100v1)

**作者:** Jiefei Chen `[一作]` (Fudan University), Dahua Lin `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 44401 | [OpenAlex ID](https://openalex.org/A5010087030)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多专家（MoE）模型推理中，提出一种动态上下文并行（DCP）系统，能够在每个请求级别上将MoE通信与KV缓存放置解耦，利用全局调度器实现双重平衡调度。

**💡 创新点**

核心创新在于：①为每个请求动态分配上下文并行度（CP度），使长请求分散到多GPU以减轻KV缓存不均衡，短请求保持本地执行；②双重平衡调度同时考虑MoE通信批量与KV缓存负载；③使用AOT图引擎与路由通信后端，在保持GPU静态形状的前提下实现动态路由；④引入全局页表支持动态KV分配。

**🔧 技术方法**

技术实现包括：Hybrid DP‑EP 并行架构、动态上下文并行度分配、全局调度器与中央状态管理、AOT（提前编译）CUDA Graph 库、基于路由的通信后端、全局 KV 页表、WaterFill 负载分配算法。

**📊 数据集**

实验使用短文本数据集 ShareGPT‑4o（短上下文）与长文本数据集 GitHub Issue（长上下文）混合工作负载，分别设定 1% 与 5% 长请求比例；同时在 100% 长请求场景下也进行了测试。

**📈 对比分析**

与 vLLM（DP32、DP‑CP 2/4/8）和 Helix 等基线相比，DCP 在 50 ms TPOT SLO 下最高请求率提升 1.88×–3.27×；在同一请求率下 P99 TPOT 延迟下降 1.79×–2.12×；在短上下文场景下与 DP32 的平均 TPOT 对齐或略优；在全长上下文场景下与 DP‑CP 8/8 近似匹配。

**⚠️ 局限性**

局限性包括：①动态路由与全局调度仍带来微小 CPU 开销（≤2.6% 迭代时间）；②AOT 图引擎需预编译 48 张图，消耗约 5.3 GiB GPU 内存；③在极端长请求比例（>10%）时，动态 CP 受限，收益下降；④目前仅在 NVIDIA H200、NVLink+InfiniBand 环境下验证，跨硬件的可移植性待进一步评估。

---

## 456. SpectralEarth-FM: Bringing Hyperspectral Imagery into Multimodal Earth Observation Pretraining

**arXiv ID:** 2605.21075 | [PDF](https://arxiv.org/pdf/2605.21075v1)

**作者:** Nassim Ait Ali Braham `[一作]` (Technical University of Munich), Xiao Xiang Zhu `[通讯]` (Technical University of Munich)

**通讯引用:** 26114 | [OpenAlex ID](https://openalex.org/A5068384981)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

训练了一个多传感器基础模型，融合高光谱、光谱多光谱、雷达和地表温度数据，用于多任务遥感推断。

**💡 创新点**

设计了层级 Transformer 结合光谱分组 Token 化、传感器专属分支与跨模态融合，首次实现高光谱与低光谱/雷达的联合预训练，并构建了包含 3 颗高光谱卫星与多模态配套的 40 TB 大规模数据集。

**🔧 技术方法**

使用层级分组光谱 Transformer、跨模态融合模块、JEPA 风格自监督预训练、频谱插值映射以及多分支推断技术。

**📊 数据集**

使用了由 EMIT、EnMAP、DESIS 三颗高光谱卫星与 Sentinel‑2、Landsat LST、SAR 等共 2 M 地理位置、25 M 补丁、40 TB 数据构成的大规模多模态数据集。

**📈 对比分析**

与 10 个高光谱基准和 PANGAEA 7 项 EO 任务进行冻结编码器评估，取得十级基准第 1 名（平均排名 1.4）和 PANGAEA 平均排名 3.43，显著优于现有方法。

**⚠️ 局限性**

仍受限于对未见传感器的频谱插值依赖、跨传感器噪声差异处理不足以及低光谱分辨率的限制。

---

## 457. Towards Physically Consistent 4D Scene Reconstruction for Closed-loop Autonomous Driving Simulation

**arXiv ID:** 2605.21032 | [PDF](https://arxiv.org/pdf/2605.21032v1)

**作者:** Bowyn Tan `[一作]` (Tsinghua University), Shengbo Eben Li `[通讯]` (Tsinghua University)

**通讯引用:** 20073 | [OpenAlex ID](https://openalex.org/A5100747108)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对单源观测导致的 4D Gaussian Splatting 空间‑时间可辨识失败，提出 Orthogonal Projected Gradient（OPG）层级训练及 Temporal Regularization（TV）策略，实现既能保持新视角重建（NVS）稳定，又能有效建模时变信息。

**💡 创新点**

创新点包括：① 建立信息几何诊断框架揭示 Singular Observation Failure（SOF）下的信用分配困境；② 通过 OPG 正交投影梯度消除空间‑时间交叉信息，恢复空间可辨识；③ 在时变参数上加入 TV 正则，约束其物理一致性。

**🔧 技术方法**

使用 4D Gaussian Splatting + 4D Spherical Harmonics 表示；信息几何 / Fisher 信息矩阵与 Cramér‑Rao 分析；Orthogonal Projected Gradient（OPG）层级训练；Temporal Total Variation 正则；对比多种基线方法进行实验。

**📊 数据集**

Waymo Open Dataset 中的 NOTR 子集（单车道三摄像头数据）。

**📈 对比分析**

与 DrivingGaussian、StreetGaussians、S^3Gaussian 等现有方法对比，采用传统视角重建指标（PSNR/SSIM/LPIPS）和 NVS 质量评估；OPG+TV 方案在插值指标与 NVS 兼容性上与 SOTA 对齐，且在新视角下避免崩溃。

**⚠️ 局限性**

依赖高质量几何优化，若几何欠拟合会导致残影捕获时变信息；无法处理非刚体运动（如行人）；对 OPG 的投影依赖空间信息，TV 正则可能抑制高频时变细节。

---

## 458. A Unified Framework for Uncertainty-Aware Explainable Artificial Intelligence: A Case Study in Power Quality Disturbance Classification

**arXiv ID:** 2605.21114 | [PDF](https://arxiv.org/pdf/2605.21114v1)

**作者:** Yinsong Chen `[一作]` (Deakin University), Chee Peng Lim `[通讯]` (Swinburne University of Technology)

**通讯引用:** 16780 | [OpenAlex ID](https://openalex.org/A5072923302)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出统一的贝叶斯神经网络解释分布框架和不确定性感知解释算子（UA-RAO），并在15类电压失真分类任务中进行评估。

**💡 创新点**

创新点在于将解释分布定义为贝叶斯后验通过Lipschitz连续归因算子推前得到的测度，提出多种统计摘要（均值、方差、分位数、变异系数、集合聚合），并给出Monte‑Carlo可达性与Wasserstein逼近的理论保证。

**🔧 技术方法**

使用的技术包括贝叶斯神经网络（深度集成、MC Dropout、Laplace）、三种本地解释方法（Occlusion、Grad‑CAM、LIME）、推前测度与Wasserstein距离分析、Monte‑Carlo采样以及RMA与IoU定位指标。

**📊 数据集**

实验数据集为XPQRS合成的15类电压失真（PQD）数据以及真实测量的电压波形。

**📈 对比分析**

通过在三种BNN近似与三种解释算子组合下的比较，发现深度集成+Occlusion+平均UA‑RAO在RMA和IoU上显著优于确定性基线，并通过多种摘要展示不同类别的定位表现。

**⚠️ 局限性**

局限性包括需满足Lipschitz连续性假设、每次解释需要大量采样导致计算成本高、未在更大网络或不同领域中验证鲁棒性。

---

## 459. RCGDet3D: Rethinking 4D Radar-Camera Fusion-based 3D Object Detection with Enhanced Radar Feature Encoding

**arXiv ID:** 2605.21112 | [PDF](https://arxiv.org/pdf/2605.21112v1)

**作者:** Weiyi Xiong `[一作]` (Beihang University), Bing Zhu `[通讯]` (Beihang University)

**通讯引用:** 3179 | [OpenAlex ID](https://openalex.org/A5062264366)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在4D雷达-相机融合的3D目标检测任务中，提出了一种基于雷达特征提取的轻量级方法 RCGDet3D，并通过改进雷达特征编码来提升检测性能与实时性。

**💡 创新点**

创新点包括：①使用射线对齐坐标系预测雷达高斯原语（R-PGE），简化姿态学习并提升几何一致性；②引入语义注入（SI）模块，将图像语义信息融入高斯属性预测，实现更精细的雷达特征；③采用简单拼接融合和单一视角变换，显著减少计算量和延迟。

**🔧 技术方法**

核心技术包括：Gaussian Splatting 的点高斯编码、射线对齐坐标变换、变形注意力（Deformable Attention）进行语义注入、CenterPoint 检测头以及轻量化的图像分支（LXL）视角变换。

**📊 数据集**

在公开的两大多模态数据集上进行实验：View‑of‑Delft (VoD) 和 TJ4DRadSet。

**📈 对比分析**

与当前最先进方法（如 SIFormer、RaGS、MSSF‑PP 等）对比，RCGDet3D 在 VoD 上实现了 79.3%（EAA）/65.6%（ROI） 的 AP，并以 19.9 FPS（V100）实现实时推理；在 TJ4DRadSet 上获得 70.51%（3D AP）/69.33%（BEV AP），速度提升至 34.9 FPS，整体在准确率与速度上均超过同类方法。

**⚠️ 局限性**

局限性：①对雷达点云稀疏度仍有依赖，极端低点密度场景可能效果下降；②当前仅使用单帧雷达信息，未充分利用雷达速度信息进行时间建模；③语义注入仅在图像可用时启用，单模态雷达场景仍受限于雷达特征自身能力。

---

## 460. WCXB: A Multi-Type Web Content Extraction Benchmark

**arXiv ID:** 2605.21097 | [PDF](https://arxiv.org/pdf/2605.21097v1)

**作者:** Murrough Foley `[一作]` `[通讯]`, Murrough Foley

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个包含2008个网页、七种结构化页面类型的Web内容提取基准（WCXB），并提供开发/测试拆分及基线评测。

**💡 创新点**

解决了现有仅基于新闻文章的基准盲点，提出了基于HTML结构的页面类型分类，使用LLM+人工双重验证的高质量标注，并公开了公开数据与评测工具。

**🔧 技术方法**

利用LLM辅助标注、自动化验证脚本、前沿模型复核、脚本检查和人工审核构建标注；评估了13个传统与神经网络提取器（包括rs‑trafilatura、MinerU‑HTML、ReaderLM‑v2等）。

**📊 数据集**

WCXB本身——2,008页来自1,613域的网页，划分为1,497页训练集和511页测试集，涵盖文章、论坛、产品、集合、列表、文档、服务七类。

**📈 对比分析**

采用词袋F1、短语召回/包含率进行评估；在文章上所有系统F1>0.87，但在论坛、集合、产品等结构化类型上F1差距高达20–30分；最优系统rs‑trafilatura在开发集F1 0.859、测试集F1 0.903，神经模型未能突破传统基准。

**⚠️ 局限性**

主要局限包括数据以英语为主，SPA页面仅标记为空内容；页面类型比例不平衡（文章占53%）；LLM驱动标注缺乏正式一致性指标；仅评测了部分神经模型，未覆盖更大模型或多语言情况。

---

## 461. Reviving Error Correction in Modern Deep Time-Series Forecasting

**arXiv ID:** 2605.21088 | [PDF](https://arxiv.org/pdf/2605.21088v1)

**作者:** Minh Hoang Nguyen `[一作]` (Deakin University), Hung Le `[通讯]` (Deakin University)

**通讯引用:** 1662 | [OpenAlex ID](https://openalex.org/A5101936199)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种通用的误差纠正机制，能够在不重新训练深度时间序列预测模型的前提下，利用自回归推理时对预测误差进行后置校正；

**💡 创新点**

创新点在于设计了一个轻量级、架构无关的误差纠正器——UEC-STD，它通过将预测拆解为趋势和季节成分并分别校正，显著抑制了长周期自回归误差累积；

**🔧 技术方法**

技术上使用了自回归推理框架、移动平均趋势/季节分解、MLP误差校正网络以及自动化β调参和平衡验证策略；

**📊 数据集**

实验覆盖10个公开长周期预测数据集（如ETTh1/2、ETTm1/2、Traffic、Weather、Electricity、US Births、Saugeen River Flow、Sunspot、Solar Energy、Exchange Rate、ILI），并在4种主流模型（TimeMixer、TimesNet、TimeXer、TimeBridge）上验证；

**📈 对比分析**

与直接预测（DF）和无校正自回归（AR）相比，UEC-STD在多种模型和数据集上平均降低MSE约2.3%和MAE约0.9%，在极端长时序（如720步）时误差提升可达10%以上；

**⚠️ 局限性**

局限性包括对β和季节/趋势权重的手动调参、对训练集与验证集时间分离的敏感性，以及在极高维或极大规模数据时计算开销仍略高。

---

## 462. Divide et Calibra: Multiclass Local Calibration via Vector Quantization

**arXiv ID:** 2605.21060 | [PDF](https://arxiv.org/pdf/2605.21060v1)

**作者:** Cesare Barbera `[一作]` (University of Pisa), Andrea Pugnana `[通讯]` (University of Trento)

**通讯引用:** 220 | [OpenAlex ID](https://openalex.org/A5052641887)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于向量量化的分区加上共享代码字的组合式多分类局部校准框架，能在高维表示空间中学习可泛化的局部校准映射。

**💡 创新点**

创新点在于将连续表示空间划分为可指数扩展的Voronoi细胞，并通过共享代码字的参数化（“索引参数化”）构造局部Dirichlet校准模型，从而在稀疏区域保持统计稳定性。

**🔧 技术方法**

核心技术包括：向量量化（VQ）划分、Voronoi分区、共享代码字的Dirichlet浓度参数化、索引参数化技巧以及两阶段训练（量化感知表示学习 + 区域校准）。

**📊 数据集**

实验使用多种图像多分类基准（如ImageNet、CIFAR等）、医学图像数据集和tabular数据集（如UCI），验证方法在不同数据源上的稳健性。

**📈 对比分析**

与全球校准（Temperature Scaling、Isotonic、Dirichlet Calibration）和局部校准（KCal、Local Nets、ProCal）对比，在LCE、MLCE等局部校准指标上显著优于所有基线；在ECCE、NLL等全局指标上保持竞争力，尤其在低样本支持区域表现最突出。

**⚠️ 局限性**

局限性：依赖冻结的表示空间，若预训练模型已高度校准或特征与校准结构不匹配效果有限；理论假设局部且需稳定的量化分配；未在端到端可微VQ或动态码本上验证；在强大的全局校准方法上并未始终占优。

---

## 463. Genetic Programming with Transformer-Based Mutation for Approximate Circuit Design

**arXiv ID:** 2605.21055 | [PDF](https://arxiv.org/pdf/2605.21055v1)

**作者:** Ondrej Galeta `[一作]` (Brno University of Technology), Lukas Sekanina `[通讯]` (Brno University of Technology)

**通讯引用:** 4050 | [OpenAlex ID](https://openalex.org/A5055549968)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于Transformer的变异算子并将其集成到Cartesian Genetic Programming中，用于自动设计近似8位乘法器。

**💡 创新点**

首次将Transformer与CGP结合，设计了一个混合变异策略，在搜索过程中使用Transformer指导变异，显著提升搜索效率和结果质量。

**🔧 技术方法**

BERT式自编码器Transformer、CGP、混合变异策略、蒙版学习目标、Mann–Whitney U检验等。

**📊 数据集**

利用EvoApproxLib中39,699个8位近似乘法器构成的训练集，并根据WCE阈值筛选得到子集。

**📈 对比分析**

通过与传统CGP及EvoApproxLib的基准比较，使用Mann–Whitney U检验在中后期搜索阶段表现出显著优势（p≈10⁻⁸），且在大多数WCE阈值下获得更优的面积-误差折衷；搜索速度也更快。

**⚠️ 局限性**

仅针对8位乘法器、每个WCE阈值需单独训练Transformer，模型与数据集规模有限，训练与演化过程计算量大，缺乏跨目标、跨宽度的通用性。

---

## 464. SG-LegalCite: A Principle-Augmented Benchmark for Legal Citation Retrieval in Singapore Law

**arXiv ID:** 2605.21057 | [PDF](https://arxiv.org/pdf/2605.21057v1)

**作者:** Shannon Lee Yueh Ern `[一作]` (Singapore University of Technology and Design), Zhu Sun `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 596 | [OpenAlex ID](https://openalex.org/A5101002129)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了将法律原则与案情结合的法律引用检索范式，并实现了相应的检索系统

**💡 创新点**

首次在新加坡司法体系中构建以法律原则为核心的检索框架，并开发了首个大规模原则驱动的检索基准

**🔧 技术方法**

采用对比学习的句向量检索（InfoNCE）、多种预训练法律语言模型（SBERT、Legal-BERT、Lawma-8B、SaulLM-7B 等）以及 LLM 生成事实摘要与原则抽取

**📊 数据集**

SG‑LegalCite 数据集，包含 100,890 个案情‑原则‑引用三元组，来源于 8,523 份 2000‑2025 年新加坡最高法院判决

**📈 对比分析**

与 11 种基线（BM25、SBERT、Legal‑BERT、Legal‑Longformer、Lawma‑8B、SaulLM‑7B 等）在两种查询设置（仅案情 vs. 案情+原则）上对比，原则增强查询平均提升 MRR 111% 与 Recall 124%，大模型在原则增强下提升最高可达 184%

**⚠️ 局限性**

LLM 生成的摘要与原则抽取仍依赖人工校验，采样候选集（1000）而非完整检索，单次实验导致方差评估不足

---

## 465. Multimodal LLMs under Pairwise Modalities

**arXiv ID:** 2605.21059 | [PDF](https://arxiv.org/pdf/2605.21059v1)

**作者:** Yan Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Guangyi Chen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 4504 | [OpenAlex ID](https://openalex.org/A5088257057)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种仅使用对偶模态数据训练多模态大型语言模型的方法，并给出理论可识别性分析；

**💡 创新点**

通过理论证明可仅凭对偶模态恢复共享潜在空间，设计两阶段对齐与重组框架，实现无需完整多模态对齐的可扩展模态融合；

**🔧 技术方法**

使用自编码器自监督重建与 CLIP 样式对比学习构建共享潜在空间，随后通过异向/部分对齐与冻结预训练 LLM 的解码器进行跨模态重组；

**📊 数据集**

在 Qwen3-Omni 基础上，使用 Objaverse+synthetic 3D-文本对、TVL 触觉-文本/图像对进行训练，并在 ModelNet40、3D-MMVET 与 TVL 任务上评估；

**📈 对比分析**

与多种基线（如 PointLLM、ShapeLLM、Adapter PEFT）以及完整对齐模型比较，在 ModelNet40/3D-MMVET 上 MPM 取得 70.5%/57.6% 等最高准确率，在 TVL 上亦优于其他方法；

**⚠️ 局限性**

目前仅验证在少数新模态（3D 与触觉）上的可行性，扩展到更多模态和更复杂任务的效果与规模仍待探索。

---

## 466. DeTox-Fed: Detecting Toxic Conversations in the Fediverse with Federated Graph Neural Networks

**arXiv ID:** 2605.21054 | [PDF](https://arxiv.org/pdf/2605.21054v1)

**作者:** Pantelitsa Leonidou `[一作]` (Cyprus University of Technology), Michael Sirivianos `[通讯]` (Cyprus University of Technology)

**通讯引用:** 3886 | [OpenAlex ID](https://openalex.org/A5056600584)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DeTox-Fed，一种基于联邦图学习的去中心化社交网络毒性对话检测框架。

**💡 创新点**

创新点在于将对话树构建成图结构，融合结构、用户、情感与对话统计特征，并通过联邦学习实现跨实例协同训练，既保留数据隐私又利用跨实例的社交关系。

**🔧 技术方法**

使用图神经网络（GraphSAGE）配合 DeepWalk 结构嵌入、情感分析模型、噪声修正（NC）Backbone 以及 Federated Averaging（FedAvg）等技术实现模型训练与推理。

**📊 数据集**

使用公开的 Pleroma 数据集（约 1.3 百万对话树，涵盖 692 个实例）进行实验。

**📈 对比分析**

与轻量化 LLM 预提示方案（局部与全局）对比，DeTox-Fed 在宏观 F1 分数上略优或持平，并且在标注稀缺、部分客户端参与、不同毒性阈值等实际场景下表现稳定。

**⚠️ 局限性**

局限包括仅处理二分类任务、对长对话结构信息利用有限、对 LLM 对比实验受限于提示设计与模型安全性，以及需要进一步扩展到多类别或更细粒度的 Moderation 标签。

---

## 467. Perception of Social Robots as Communication Partners in Healthcare for Older Adults

**arXiv ID:** 2605.21053 | [PDF](https://arxiv.org/pdf/2605.21053v1)

**作者:** Hana Yamamoto `[一作]` (Karlsruhe Institute of Technology), Katja Mombaur `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 3831 | [OpenAlex ID](https://openalex.org/A5042406934)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对比人类与社交机器人在老年人健康护理中的交互，评估情感表情、心率和主观感受，探究正向提示的影响。

**💡 创新点**

首次将多模态测量（面部表情+心率+问卷）结合正向提示，对老年人机器人交互的心理生理影响进行系统评估。

**🔧 技术方法**

使用OpenCV+ONNX人脸表情识别模型、60GHz毫米波传感器心率监测、文本转语音交互以及Wizard‑of‑Oz 控制的 Navel 机器人。

**📊 数据集**

35名70岁以上健康老年人（MMSE≥24）在实验室完成两轮交互。

**📈 对比分析**

通过卡方检验、ANOVA 等统计方法比较机器人与人类交互及提示与否的差异，结果显示机器人交互与人类无显著压力差异，心率略低，正向提示无显著效果。

**⚠️ 局限性**

样本量有限、机器人外观与对话内容不匹配、正向提示设计过于重复，且实验仅限结构化问卷场景，未检验长期交互效果。

---

## 468. Decoupling Communication from Policy: Robust MARL under Bandwidth Constraints

**arXiv ID:** 2605.21085 | [PDF](https://arxiv.org/pdf/2605.21085v1)

**作者:** Alexi Canesse `[一作]` (Institut Polytechnique de Paris), Sonia Vanier `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5031988385)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 SLIM 架构，将通信编码与策略执行分离，实现在低带宽下的多智能体强化学习。

**💡 创新点**

创新点在于引入标准化的带宽预算 β，将消息尺寸、传输频率和图稀疏性统一衡量，并通过缓存历史消息的 Transformer 方式提升对部分可观测环境的鲁棒性。

**🔧 技术方法**

采用多智能体强化学习中的 CTDE、MAPPO、Transformer 注意力模块、通信编码器以及消息缓存。

**📊 数据集**

在四个部分可观测的 MARL 基准上进行实验：Predator‑Prey、Traffic Junction、Navigation 和 SHAPES。

**📈 对比分析**

与 CommNet、IC3Net、TarMAC、CommFormer 等基线在标准化带宽 β 下进行比较，SLIM 在高带宽下与最优基线持平或领先，在低带宽下表现更稳健，性能下降幅度最小。

**⚠️ 局限性**

局限在于未考虑真实无线网络中的包头、量化、时延、路由、丢包及中介竞争等因素，且缓存线性增长，需在长周期任务中进行窗口或压缩处理。

---

## 469. Q-ARVD: Quantizing Autoregressive Video Diffusion Models

**arXiv ID:** 2605.21072 | [PDF](https://arxiv.org/pdf/2605.21072v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 470. AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions

**arXiv ID:** 2605.21082 | [PDF](https://arxiv.org/pdf/2605.21082v1)

**作者:** Minghao Chen `[一作]` (Hangzhou Dianzi University), Yufei Yin `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 8404 | [OpenAlex ID](https://openalex.org/A5046313825)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用大型语言模型（LLM）在 ReAct 交互中收集成功轨迹，随后通过翻译器将硬编码的动作转换为可重用的软编码动作，再由构建器生成可直接执行的 RPA 函数，并在执行过程中结合 RPA 结果和 ReAct 反馈进行自我修复。

**💡 创新点**

核心创新包括：
1) Translator‑Builder 管道——将 ReAct 轨迹中的硬动作自动翻译为鲁棒的软动作并合成 RPA 代码；
2) 层级化检索增强生成（RAG）——利用树结构轨迹库在构建过程中动态检索相关观察和子轨迹；
3) 混合修复策略——在 RPA 执行失败时先分析断点，再让 ReAct 代理继续完成任务，从而实现迭代改进。

**🔧 技术方法**

技术手段：LLM（GPT‑4o、GPT‑4.1、GPT‑5）、ReAct 代理、翻译器、构建器、检索增强生成（RAG）、分析器、混合修复、POMDP 任务建模。

**📊 数据集**

使用的基准数据集：
- AndroidWorld（116 种任务类型、20 个真实 Android 应用）
- WebArena（Reddit 领域，19 种任务类型）
- MiniWoB++（9 种有反馈的“硬”任务 + 44 种无反馈的“简单”任务）。

**📈 对比分析**

与多种现有方法比较：ReAct、SeeAct、M3A、SteP、RCI、AdaPlanner、AutoManual 等。在三大 benchmark 上，AutoRPA 的成功率与 SOTA 相当甚至更高，同时 token 消耗下降 82%–96%，执行时间降低 50% 以上。code‑only 版本在 token 量极低的情况下仍保持高成功率。

**⚠️ 局限性**

局限性：
- 需要先在 ReAct 交互中完成若干“构建任务”，对任务数不足时效果下降；
- 对极端或非典型任务的鲁棒性仍有限；
- RPA 脚本在界面结构大幅变化时可能失效；
- 实验主要在模拟/仿真环境，缺乏真实生产系统的验证。

---

## 471. Linear-DPO: Linear Direct Preference Optimization for Diffusion and Flow-Matching Generative Models

**arXiv ID:** 2605.21123 | [PDF](https://arxiv.org/pdf/2605.21123v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 472. GradeLegal: Automated Grading for German Legal Cases

**arXiv ID:** 2605.21076 | [PDF](https://arxiv.org/pdf/2605.21076v1)

**作者:** Abdullah Al Zubaer `[一作]` (University of Passau), Jelena Mitrovic `[通讯]` (University of Passau)

**通讯引用:** 663 | [OpenAlex ID](https://openalex.org/A5019466280)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估27种大语言模型在德国刑法和行政法案例答案评分中的自动化能力，探讨提示策略和模型选择对评分准确度的影响

**💡 创新点**

首次系统比较开源与闭源、推理与非推理模型在长篇法律答卷评分中的表现；发现结构化评分量表与示例答案联合使用能显著提升与专家评分的一致性；引入多模型集成方法进一步提升性能

**🔧 技术方法**

零样本提示（Task-Agnostic、Instr.+Rubric、Instr.+Solution、Instr.+Rubric+Solution），使用Quadratic Weighted Kappa评估；并通过最小值与中位数聚合构建集成评分器

**📊 数据集**

两个非公开德国法考案例集：刑法（71份）与行政法（16份），答案长度平均约12k-27k token，评分采用0-18点等级

**📈 对比分析**

在完整提示下，推理型GPT‑5在行政法上达0.911 QWK，与人工评判相当；刑法最高0.599 QWK；相较之前报告（≈0.6）大幅提升；集成模型在两类题目均可超越最佳单模型（提升≈0.1 QWK）

**⚠️ 局限性**

数据量有限（尤其行政法），结果受分数分布影响；仅在德国司法体系与考试格式下验证，跨国、跨语言推广需进一步实验；未进行模型微调，未探讨自适应聚合策略

---

## 473. Improved Guarantees for Constrained Online Convex Optimization via Self-Contraction

**arXiv ID:** 2605.21107 | [PDF](https://arxiv.org/pdf/2605.21107v1)

**作者:** Dhruv Sarkar `[一作]` (Indian Institute of Technology Kharagpur), Abhishek Sinha `[通讯]` (Tata Institute of Fundamental Research)

**通讯引用:** 2087 | [OpenAlex ID](https://openalex.org/A5057128963)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于嵌套投影的在线梯度下降算法，用于约束在线凸优化问题。

**💡 创新点**

利用自收缩曲线的几何性质，首次给出了嵌套投影轨迹的有限长度证明，从而显著降低约束违规度量。

**🔧 技术方法**

核心技术包括投影更新、梯度步长设定以及自收缩曲线的理论工具。

**📊 数据集**

无实验数据集，全部为理论分析与证明。

**📈 对比分析**

与之前最优的 O(√T) 违规度量相比，强凸场景下实现 O(log T) 违规度量，凸场景下保持 O(√T) 违规度量且仅提升了常数因子；在极端情况下，强凸时既获得 O(log T) 违规度量又保持 O(log T) 惩罚。

**⚠️ 局限性**

常数 C_d 依赖维度 d，缺乏明确的尺度分析；仅适用于准凸约束；并未给出下界，尚未证明对凸损失下的违规度量是否可进一步优化。

---

## 474. Robust Personalized Recommendation under Hidden Confounding in MNAR

**arXiv ID:** 2605.21066 | [PDF](https://arxiv.org/pdf/2605.21066v1)

**作者:** Zongyu Li `[一作]` (Guangdong University of Technology), Tianyu Xia `[通讯]` (Peking University)

**通讯引用:** 637 | [OpenAlex ID](https://openalex.org/A5112745105)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种能够在隐藏混杂因素下进行稳健个性化推荐的框架（PUID），并在其基础上加入基准引导的BPUID

**💡 创新点**

首次通过信息熵增益估计用户-物品级的敏感度边界，从而实现个性化的隐藏混杂修正；并将基准模型用于正则化，平衡鲁棒性与准确性

**🔧 技术方法**

基于熵敏感度估计、极值优化的鲁棒IPS/DR损失、对抗式训练、预训练模型做基准参考以及离散化分箱的熵估计

**📊 数据集**

KuaiRec、Coat、Yahoo!R3 三个公开推荐数据集（含有有偏与无偏交互）

**📈 对比分析**

与标准MF‑MLPs、IPS、DR、RD、BRD 等全局敏感度方法比较，实验显示PUID和BPUID在UAUC、NDCG等指标上均显著优于基线，尤其在高缺失率下BPUID更为稳定

**⚠️ 局限性**

熵敏感度估计对分箱策略敏感，个性化边界依赖于得分范围归一化和区间校准，框架效果受底层表示学习能力和超参平衡的影响

---

## 475. On the Complexity of Entailment for Cumulative Propositional Dependence Logics

**arXiv ID:** 2605.21113 | [PDF](https://arxiv.org/pdf/2605.21113v1)

**作者:** Kai Sauerwald `[一作]` (FernUniversität in Hagen), Arne Meier `[通讯]` (Leibniz Universität Hannover)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了累积命题依赖逻辑和累积命题团队语义逻辑的蕴含问题的复杂度。

**💡 创新点**

首次给出了两类逻辑在累积模型下蕴含问题的精确复杂度上界与下界，并揭示了两者在复杂度层级上的差异。

**🔧 技术方法**

使用了累积模型、关系模型、累积推理规则、oracle 机器以及归约技术进行复杂度分析。

**📊 数据集**

本工作为理论研究，无涉及具体数据集。

**📈 对比分析**

通过与已知的偏好逻辑和团队逻辑复杂度比较，证明了蕴含问题在 Θ^p_2、Π^p_2 等复杂度类中的位置，并在相应下界证明中达到 -hard、Δ^p_2-hard 的结果。

**⚠️ 局限性**

缺乏与下界匹配的完整性结果，且对简化版本的证明仍依赖于强归约；未来需要进一步完善完整性与可扩展性。

---

## 476. Efficient Learning of Deep State Space Models via Importance Smoothing

**arXiv ID:** 2605.21108 | [PDF](https://arxiv.org/pdf/2605.21108v1)

**作者:** John-Joseph Brady `[一作]` (King's College London), Yunpeng Li `[通讯]` (King's College London)

**通讯引用:** 3699 | [OpenAlex ID](https://openalex.org/A5100331862)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种并行可微粒子平滑器 PVMC，用于高效训练深度状态空间模型。

**💡 创新点**

创新点在于通过对边缘平滑后验进行 importance weighting，消除粒子过滤的顺序性，实现 log N × log T 的 span 复杂度，同时获得无偏梯度与更紧致的 ELBO。

**🔧 技术方法**

主要技术包括并行前缀/后缀扫描、重要性采样、ELBO 优化、深度神经网络参数化的 SSM 以及 GPU 级并行实现。

**📊 数据集**

实验数据集包括线性高斯系统（可得到精确平滑解）、Lotka–Volterra 捕食-猎物模拟（带监督训练）以及 S&P 500 日收盘价时间序列（无监督生成任务）。

**📈 对比分析**

与 RTS、TFS、d‑SMC、Soft/Stop‑Gradient/Diffusion DPF、MDPS、DMM、TC‑VAE 等基线对比，PVMC 在状态估计 MSE、KSD、训练时间等指标上均优于对手，并实现约 10 倍的速度提升。

**⚠️ 局限性**

局限性包括对粒子数的依赖、对更高维非线性系统的可扩展性尚未充分验证，以及在某些复杂真实任务中仍需进一步改进模型结构和采样策略。

---

## 477. Benchmarking Empirical and Learning-Based Approaches for Feedforward Steering Control in Autonomous Racing

**arXiv ID:** 2605.21111 | [PDF](https://arxiv.org/pdf/2605.21111v1)

**作者:** Georg Jank `[一作]` (Technical University of Munich), Boris Lohmann `[通讯]` (Technical University of Munich)

**通讯引用:** 4398 | [OpenAlex ID](https://openalex.org/A5006081522)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对自适应赛车中前馈转向控制器进行系统基准测试，比较两种经验模型（基线与新的EHD）与两种学习模型（MSNN 与 LSTM）在开环与闭环中的表现。

**💡 创新点**

提出了基于多项式曲面拟合的经验逆动力学模型 EHD，能够以最小参数捕捉速度相关的非线性转向行为，并在闭环系统中实现更高的鲁棒性和更快的圈速。

**🔧 技术方法**

使用多项式拟合、模型结构化神经网络（MSNN）、长短期记忆网络（LSTM）、高保真双轨车辆动力学仿真、基于MPM 的轨迹规划和闭环评估框架。

**📊 数据集**

利用 Abu Dhabi Autonomous Racing League 赛道的高保真仿真数据，包括 26 圈的训练集与 1 圈的测试集，真实车辆参数校准后用于验证。

**📈 对比分析**

在开环中，MSNN 与 LSTM 的预测误差分别比基线低 39.1% 与 39.7%，但在闭环评估（无反馈与有反馈）中，EHD 在最高 GG 比例下表现最好，圈速最快，鲁棒性最高；学习模型虽然降低了横向加速度误差，却未能提升轨迹跟踪或圈速。

**⚠️ 局限性**

局限性：学习模型对加速度目标的过度依赖导致闭环中转向延迟与误差放大；EHD 的结构固化限制了进一步改进；未考虑直接路径或偏航率信息，导致学习模型难以完全匹配整体跟踪目标；对实时在线自适应学习的缺失。

---

## 478. HORST: Composing Optimizer Geometries for Sparse Transformer Training

**arXiv ID:** 2605.21104 | [PDF](https://arxiv.org/pdf/2605.21104v1)

**作者:** Tom Jacobs `[一作]` (CISPA Helmholtz Center for Information Security), Rebekka Burkholz `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的稀疏训练优化器 HORST，结合 AdamW 与超bolic 镜像更新，实现稳定性与稀疏性兼顾。

**💡 创新点**

创新点在于把优化器视为非可交换算子组合，通过先执行 AdamW 再执行超bolic 镜像步长，既保留 L∞ 隐含偏差，又引入 L1 稀疏性。

**🔧 技术方法**

使用的技术包括算子组合理论、镜像下降、AdamW、超bolic 熵镜像映射以及 AC/DC 稀疏化流水线。

**📊 数据集**

在 ImageNet（DeiT-base/DeiT-small）和 GPT-2 Small（SlimPajama-6B）上进行实验。

**📈 对比分析**

与 AdamW 和 HAM 相比，HORST 在 70%、80%、90% 稀疏率下均提升 10–15% 准确率/降低困惑度，尤其在高稀疏率时表现最显著。

**⚠️ 局限性**

局限在于对 L∞ 极限的理论分析尚未完整，且仅在无微调的一次性剪枝实验中验证，未探讨与后置剪枝方法的组合。

---

## 479. A Typed Tensor Language for Federated Learning

**arXiv ID:** 2605.21103 | [PDF](https://arxiv.org/pdf/2605.21103v1)

**作者:** Theofilos Mailis `[一作]` (Athena Research Center), Yannis Ioannidis `[通讯]` (Athena Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

本文提出了一种类型化张量语言，系统化了联邦学习与分析中的客户端局部计算、共享状态聚合以及共享后处理，并通过共享状态因式分解理论证明了该语言的可表达性与迭代封装能力。

**💡 创新点**

创新点在于将联邦计算抽象为带记录轴类型的张量语言，给出严格的语义与类型规则，并证明所有符合该语言的一轮或迭代程序都可分解为 encode‑merge‑decode 的固定维共享状态；此外，该框架被扩展到可微分学习，统一描述了一阶与二阶联邦优化方法。

**🔧 技术方法**

主要技术包括：类型系统设计与记录轴追踪、虚拟全局张量语义、共享状态因式分解定理、可微分片段与梯度聚合、共享线性代数扩展，以及对隐私机制的接口化。

**📊 数据集**

论文未使用具体数据集，所有讨论均基于理论模型与形式化示例。

**📈 对比分析**

论文未进行实验比较或性能评估，主要提供理论证明与与集中式算法等价性的讨论。

**⚠️ 局限性**

局限性包括：仅支持固定维共享状态的计算；无法处理持久私有客户端状态、共享状态维度随记录数增长的情况；以及对完整隐私机制的支持需要额外的类型或原语扩展。

---

## 480. Fine-grained Claim-level RAG Benchmark for Law

**arXiv ID:** 2605.21071 | [PDF](https://arxiv.org/pdf/2605.21071v1)

**作者:** Souvick Das `[一作]` (University of Luxembourg), Domenico Bianculli `[通讯]` (University of Luxembourg)

**通讯引用:** 1790 | [OpenAlex ID](https://openalex.org/A5038017715)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了CLAIMRAG-LAW基准，包括317个法语/英语QA对和968条人工校验的法律主张。

**💡 创新点**

首次将多语言、多法域的非专家与专家问题结合到细粒度检索增强生成（RAG）评估中，并引入基于主张的评估框架。

**🔧 技术方法**

采用GPT‑4生成QA对，使用RAGChecker和RefChecker进行检索与生成以及主张提取/验证的细粒度评估。

**📊 数据集**

数据来源为欧盟GDPR（英文）和法国民法（法文），并与现有法律RAG模型进行对照测试。

**📈 对比分析**

通过对八种主流RAG系统的RAGChecker评分，发现检索准确率低、生成中出现高比例的误基准和幻觉，主张级评估对矛盾主张的检测仍有限。

**⚠️ 局限性**

数据为人工合成且单一专家标注，缺少多方验证，且GPT‑4生成的QA可能带来偏见，导致评估结果对真实法律查询的泛化有限。

---

## 481. R2AoP: Reliable and Robust Angle of Progression Estimation from Intrapartum Ultrasound

**arXiv ID:** 2605.21099 | [PDF](https://arxiv.org/pdf/2605.21099v1)

**作者:** Yuanhan Wang `[一作]` (Tsinghua University), Qiyuan Tian `[通讯]` (Tsinghua University)

**通讯引用:** 2883 | [OpenAlex ID](https://openalex.org/A5066843175)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种R2AoP框架，实现了可靠稳健的胎位进展角（AoP）估计。

**💡 创新点**

创新点在于将结构感知分割与置信度引导的几何建模结合，并通过几何可靠约束的测试时自适应提升跨域鲁棒性。

**🔧 技术方法**

使用了三分支局部结构增强U-Net、置信度加权椭圆拟合、熵最小化、TV平滑以及几何可靠性约束的TTA。

**📊 数据集**

主要数据集包括公开的PSFHS源域数据，以及JNU-IFM和IUGC 2024两套跨域目标域数据。

**📈 对比分析**

在两个目标域上与多种现有AoP方法对比，R2AoP在AoP误差、ASD、HD100和Dice指标均取得最优或同等表现，显著优于竞争者。

**⚠️ 局限性**

局限性包括依赖高质量置信度图、对训练时三分支结构的实现复杂度、以及在极端噪声或极端解剖变异时仍可能产生误差。

---

## 482. APM: Evaluating Style Personalization in LLMs with Arbitrary Preference Mappings

**arXiv ID:** 2605.21063 | [PDF](https://arxiv.org/pdf/2605.21063v1)

**作者:** Philipp Spohn `[一作]` (Technical University of Munich), Zeynep Akata `[通讯]` (Technical University of Munich)

**通讯引用:** 16409 | [OpenAlex ID](https://openalex.org/A5040372929)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了APM基准来评估LLM在缺乏显式偏好信息下的隐式风格个性化

**💡 创新点**

通过隐藏随机映射用户属性到响应原则并校准基线，消除评判者偏差

**🔧 技术方法**

采用检索增强生成、基于用户嵌入的提示优化和路由式个性化三种技术

**📊 数据集**

使用合成用户属性-对话历史数据（N=M=10，4000训练+1000测试用户，2-4轮对话）

**📈 对比分析**

在APM上与非个性化基线比较，路由方法在Qwen上W/L≈1.79，Llama约1.40；RAG仅在更强模型上略升，提示优化无提升

**⚠️ 局限性**

仍受限于历史信息不足、评价指标依赖LLM判定、路由器标签质量瓶颈以及对更大原则集的可扩展性不足

---

## 483. Efficient Banzhaf-Based Data Valuation for $k$-Nearest Neighbors Classification

**arXiv ID:** 2605.21033 | [PDF](https://arxiv.org/pdf/2605.21033v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 484. ACL-Verbatim: hallucination-free question answering for research

**arXiv ID:** 2605.21102 | [PDF](https://arxiv.org/pdf/2605.21102v1)

**作者:** Gábor Recski `[一作]` (TU Wien), Ádám Kovács `[通讯]` (KR Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了ACL Anthology的提取式问答系统ACL-Verbatim，并创建了100个查询-文本片段对的基准数据集；使用该数据集训练和评估多种提取模型。

**💡 创新点**

提出将VerbatimRAG与自定义文档分块、LLM生成合成查询相结合，专注于提取式回答以消除hallucination；并展示小型BERT模型在提取任务上可超越大型LLM。

**🔧 技术方法**

采用VerbatimRAG框架、Docling将PDF转换为Markdown、IBM Granite嵌入、基于ScIRGen的查询生成、手工标注、现代BERT token classifier等技术。

**📊 数据集**

主要使用ACL Anthology（120k+论文）进行数据处理与索引，并在其中随机抽样333篇论文生成合成查询，最终形成100个查询-片段标注对。

**📈 对比分析**

在提取任务上用词级F1评估；训练的150M参数ModernBERT达到53.6的F1，优于最强LLM提取器的48.7；说明小模型在提取任务上更高效、参数更少。

**⚠️ 局限性**

局限性包括数据集规模仅100对，主要基于合成查询，可能缺乏真实用户多样性；只评估提取不涉及生成完整答案；模型泛化能力与不同学科/语言的适用性尚待验证。

---

## 485. Grounding Driving VLA via Inverse Kinematics

**arXiv ID:** 2605.21061 | [PDF](https://arxiv.org/pdf/2605.21061v1)

**作者:** Junsung Park `[一作]` (Korea Advanced Institute of Science and Technology), Hyunjung Shim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2365 | [OpenAlex ID](https://openalex.org/A5079082222)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在驾驶视觉‑语言‑动作（Driving VLA）模型中加入未来视觉状态预测任务和仅基于当前/未来视觉状态的逆运动学网络（IK 网络），实现视觉特征在轨迹规划中的显式利用，并显著提升小模型的性能。

**💡 创新点**

将轨迹生成视为逆运动学问题，提出：①使用 LLM 预测下一帧视觉状态作为监督；②设计仅接受视觉状态对的 IK 网络，消除 ego 状态和文本命令的快捷通路，从而恢复视觉 grounding；③通过两任务联合训练保证视觉信息在所有训练场景中都有梯度。

**🔧 技术方法**

技术手段包括：Qwen2.5‑0.5B LLM + DINOv3 ViT‑S/16 视觉编码器；下一视觉状态预测（MSE 损失）；交叉注意力的条件扩散模型作为 IK 网络；多任务训练（视觉预测、问答、轨迹回归）；GradCAM、对抗剪贴等分析工具。

**📊 数据集**

使用的数据集有：NAVSIM（闭环仿真 benchmark，包含 v1 与 v2 子集）、nuScenes（开放循环 benchmark）、nuCaption、nu‑X、nuScenes‑QA 等文本与场景描述数据。

**📈 对比分析**

与 OpenDriveVLA、Alpamayo‑R1、DriveFine 等 7B–8B 大模型进行对比；在 NAVSIM‑v1 取得 PDMS 92.2（比 OpenDriveVLA 高 19.0）、在 NAVSIM‑v2 取得 EPDMS 90.6（比 OpenDriveVLA 高 20.4）；在 nuScenes 上 ADE 0.06、碰撞率 0.09，分别比 7B 规模模型低 0.27 的误差；同时在 GradCAM、物体对齐等视觉 grounding 指标上显著优于基线。

**⚠️ 局限性**

局限性包括：1）模型仍需要显式的视觉预测和 IK 网络，结构上较为复杂；2）对极端动态场景的改进有限，依赖训练数据多样性；3）目前仅在仿真/公开数据上验证，缺乏真实车辆场景的鲁棒性评估；4）对视觉感知噪声或缺失信息的容忍度尚未充分评估。

---

## 486. Automated ICD Classification of Psychiatric Diagnoses: From Classical NLP to Large Language Models

**arXiv ID:** 2605.21154 | [PDF](https://arxiv.org/pdf/2605.21154v1)

**作者:** Fernando Ortega `[一作]` (Universidad Politécnica de Madrid), Enrique Baca-García `[通讯]` (University Hospital Jimenez Díaz Fundation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对西班牙精神卫生临床记录进行ICD编码自动化，比较多种文本表示和分类模型，并通过对e5_large进行端到端微调显著提升性能。

**💡 创新点**

系统性评估传统词频模型、无监督主题模型、BERT、BioLORD、e5系列、Llama-3-8B等多种文本表示与Random Forest、XGBoost、MLP、LLM微调等分类器的组合；首次在大规模西班牙精神健康数据上展示LLM微调在长尾标签分布下的优势。

**🔧 技术方法**

自然语言处理（BERT、BioLORD、e5、Llama-3-8B、paraphrase multilingual等Transformer）、机器学习（Random Forest、XGBoost、MLP）以及端到端LLM微调。

**📊 数据集**

约79,048条西班牙语精神科诊断描述（145,513条原始文本中挑选，来自Fundación Jiménez Díaz医院），标注为85个ICD诊断码。

**📈 对比分析**

使用Micro/Macro F1、精确率、召回率等指标；传统词频+MLP已接近0.84 Micro F1；e5_large微调后达到0.866 Micro F1、0.804 Macro F1，显著优于其他组合。

**⚠️ 局限性**

仍受长尾标签分布和少数类稀缺影响，少数类精准率和召回率低；模型规模大、推理成本高；缺乏可解释性和针对罕见诊断的增量学习方法。

---

## 487. EllipseLIO: Adaptive LiDAR Inertial Odometry with an Ellipsoid Representation

**arXiv ID:** 2605.21150 | [PDF](https://arxiv.org/pdf/2605.21150v1)

**作者:** Rowan Border `[一作]` (University of Cyprus), Margarita Chli `[通讯]` (University of Cyprus)

**通讯引用:** 9653 | [OpenAlex ID](https://openalex.org/A5055582765)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出并实现了一种名为EllipseLIO的LiDAR‑Inertial Odometry算法，能够在不同传感器和环境中无需手动调参即可实时提供可靠的位姿估计。

**💡 创新点**

创新点包括：① 基于距离自适应的扫描滤波方法，动态调节不同距离下的下采样分辨率；② 采用椭球体表面几何信息自适应选择点对-点、点对-平面或点对-线误差度量的注册算法；③ 在扫描注册过程中通过匹配点与已观测椭球的距离与垂直方向重力方向的相对角度动态加权，实现漂移自我纠正。

**🔧 技术方法**

核心技术有：iOctree 数据结构用于高效稀疏存储与搜索；迭代扩展卡尔曼滤波(IEKF)用于状态预测与更新；张量投票(TV)与主成分分析构造椭球体并提取线/面/球特征；自适应匹配权重与退化检测机制；以及与FAST‑LIO等开源框架的深度耦合。

**📊 数据集**

实验在五个公开数据集上完成：Newer College、Oxford Spires、Botanic Garden、GRACO（无人机）和 GEODE（室内外多平台）。

**📈 对比分析**

与DLIO、FAST‑LIO2、LIO‑SAM、iG‑LIO在统一的0.1 m voxel分辨率下进行对比，EllipseLIO在所有序列中均未发生发散，平均APE比第二佳方法低38%，且在大尺度与狭窄环境下均保持低误差；计算时间约为35 ms，内存占用仅2.4 GB，优于多数对手。

**⚠️ 局限性**

局限性在于目前仅使用LiDAR与IMU，未加入视觉信息；在极度几何退化或高度动态环境下仍可能出现漂移；对极低分辨率或极高分辨率传感器的鲁棒性尚待进一步验证。

---

## 488. Cloud-Native Operation of Roadside Infrastructure Enabling Demand-Driven Collective Perception via V2X

**arXiv ID:** 2605.21145 | [PDF](https://arxiv.org/pdf/2605.21145v1)

**作者:** Lukas Zanger `[一作]` (RWTH Aachen University), Lutz Eckstein `[通讯]` (RWTH Aachen University)

**通讯引用:** 3929 | [OpenAlex ID](https://openalex.org/A5113050304)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了基于 Kubernetes 的云原生、需求驱动的道路边缘基础设施编排，并在真实试验场景中验证了基于 V2X 的集体感知应用；

**💡 创新点**

首次将 V2X 触发的按需服务编排与云原生 Kubernetes 结合，实现了能源节约与可扩展性，并通过实验给出了延迟与能耗估算；

**🔧 技术方法**

使用 K3s/Kubernetes、Helm、Prometheus、Grafana、ROS 2 + Zenoh、ITS‑G5 V2X、ETSI CAM/CPM、容器化微服务与 GPU 感知算法；

**📊 数据集**

利用本测试场地一周的 CAM 记录（69610 条消息）以及实验车辆 karl 的实时数据；

**📈 对比分析**

通过对比持续开启与按需启动的两种部署模式，测量了端到端延迟（≈13 s）和部署延迟（≈10 s 占主导），并估算了能耗节约（≈4.22 kWh/天，≈1500 kWh/年）——显示在 ITS‑G5 通信范围内可行；

**⚠️ 局限性**

受限于低交通密度、对 ITS‑G5 覆盖率的假设、冷启动占用时间过长、仅在四台单元上验证、能耗估算依赖具体硬件与场景，缺乏在高密度城市环境下的验证。

---

## 489. ECHO-PPI: Trustworthy AI for Evidence-Bundled Detection of Overlapping Protein Modules in Protein-Protein Interaction Networks

**arXiv ID:** 2605.21216 | [PDF](https://arxiv.org/pdf/2605.21216v1)

**作者:** Sima Soltani `[一作]` (Islamic Azad University), Yahya Forghani `[通讯]` (Islamic Azad University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 ECHO-PPI 框架，结合网络拓扑、语义嵌入和 GO 证据，为蛋白互作网络中的重叠模块提供可解释的蛋白-模块分配；

**💡 创新点**

创新点在于为每个蛋白-模块分配附加“证据包”与分层置信度标签，实现在可解释、可审计的层面提升基因组学模块预测的透明度；

**🔧 技术方法**

采用 MCL 作为基准分区，利用“黑洞”核评分进行候选中心生成，计算拓扑永续性、GO TF-IDF、语义相似度，结合重叠启发式与回顾安全补充策略；

**📊 数据集**

使用酵母（Gavin、Krogan）PPI 数据集，并结合 GO 注释、CYC2008 金标准；

**📈 对比分析**

与 MCL、MCL+重叠、ClusterONE、SLPA 等基准对比，ECHO-PPI 在预测 F1 与 MCL+重叠持平或略低，但在每个分配上实现完整证据导出，覆盖率与模块大小均保持与基准相似；

**⚠️ 局限性**

主要局限在于预测精度未超越现有最强重叠基线，受候选覆盖空间限制和注释完整性影响，且证据权重及阈值仍需经验调优，未来需扩展至更大人类 PPI 数据与更丰富的结构/跨物种证据。

---

## 490. Behavior-Consistent Deep Reinforcement Learning

**arXiv ID:** 2605.21214 | [PDF](https://arxiv.org/pdf/2605.21214v1)

**作者:** Marcel Hussing `[一作]` (University of Pennsylvania), Eric Eaton `[通讯]` (University of Pennsylvania)

**通讯引用:** 40545 | [OpenAlex ID](https://openalex.org/A5035778053)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出行为一致强化学习框架，研究在不同随机种子下训练同一算法产生的策略行为差异，并提出 Q-value Expectile Disagreement（QED）自适应熵调节方法来显著降低跨跑的策略分布差异。

**💡 创新点**

创新点在于：①将最大熵RL与行为一致目标结合，证明温度与 Q‑值不一致成正比可控制 KL 散度；②设计 QED 通过双 Q 失衡和期望分位估计，实时估计单跑内部温度；③解决高熵导致离线数据外推误差的问题，并在稳定算法上验证其有效性。

**🔧 技术方法**

技术主要包括：最大熵RL与 KL 约束理论、双 Q 估计、期望分位数估计、LayerNorm 以及 MAD‑SAC（模型增强数据的 Soft Actor‑Critic）等稳定离线算法。

**📊 数据集**

数据集为 18 个 MuJoCo 连续控制任务（Ant, HalfCheetah, Hopper, Walker2d 等）以及单腿 Hopper 的高变异子任务。

**📈 对比分析**

对比方法包括标准 SAC‑LN 与 MAD‑SAC，使用 IQM 与 95% 置信区间评估回报，使用对称 KL 与 L2 行动距离评估行为一致性。QED 在保持或略低于基准回报的同时，将跨跑 KL 降低约两订单、回报方差降低约 50%，并在高变异任务上实现收敛一致。

**⚠️ 局限性**

局限性：依赖 Q‑值的点估计与双 Q 一致性，失衡消退缓慢；在高维任务中，KL 仍可能超过 1，表现出性能‑一致性权衡；且高熵下仍需稳健的离线学习器，若无此支持易导致失稳。

---

## 491. Semantic Granularity Navigation in Image Editing

**arXiv ID:** 2605.21190 | [PDF](https://arxiv.org/pdf/2605.21190v1)

**作者:** Liangsi Lu `[一作]` (Guangdong University Of Technology), Yang Shi `[通讯]` (Guangdong University Of Technology)

**通讯引用:** 100202 | [OpenAlex ID](https://openalex.org/A5100674628)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无训练、无逆向的推理时控制器NaviEdit，用以解耦图像编辑进度与模型尺度，实现对漂移型生成模型的高效编辑。

**💡 创新点**

创新点在于将编辑过程视为在尺度轴上的受控积分，明确进度与尺度的分离，并通过自洽的步长合同实现尺度分配的解耦，从而在固定计算预算下聚焦于尺度窗口内的高效计算。

**🔧 技术方法**

主要技术包括差分场（velocity field）检索、尺度窗口探测、基于泄漏压和振荡度的诊断、以及自洽步长更新与控制策略。

**📊 数据集**

使用PIE‑Bench和ImgEdit‑Bench两个公开编辑基准进行实验评估。

**📈 对比分析**

与多种基线（DiffEdit、InfEdit、FlowEdit、Prompt‑to‑Prompt等）对比，NaviEdit在保持结构完整性的同时实现更强语义改动，取得PSNR、SSIM、LPIPS、CLIP‑Whole/Edited等指标上的显著提升。

**⚠️ 局限性**

局限性包括对编辑支持估计的依赖、对复杂场景（镜面、重复对象）缺乏全局一致性约束，以及在已训练好、端到端学习编辑策略的模型中增益有限。

---

## 492. A Terrain-Adaptive epsilon-Constraint MPC for Uneven Terrain Kinodynamic Planning

**arXiv ID:** 2605.21188 | [PDF](https://arxiv.org/pdf/2605.21188v1)

**作者:** Otobong Jerome `[一作]` (Universidade Federal da Paraíba), Tiago Nascimento `[通讯]` (Universidade Federal da Paraíba)

**通讯引用:** 1428 | [OpenAlex ID](https://openalex.org/A5064505240)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于地形自适应的ϵ约束模型预测控制（MPC）框架，用于在不规则地形上进行车式车辆的运动动力学规划，兼顾路径效率和姿态稳定性。

**💡 创新点**

创新点包括：① 通过地形特征自适应调整ϵ阈值，实时平衡路径优先与稳定性约束；② 采用半参数模型，将解析车辆动力学与稀疏高斯过程残差学习相结合，提高动态预测精度；③ 利用快速行进法（FMM）生成全局向量场，指导采样并避免局部最优；④ 在实时重排控制框架内对多目标问题做ε约束求解。

**🔧 技术方法**

使用的技术包括：模型预测控制（MPC）+ε约束多目标优化、快速行进法（FMM）向量场、稀疏高斯过程（SGP）残差学习、采样式控制序列生成（warm‑start + FMM‑biased），以及ROS 2 + Gazebo仿真。

**📊 数据集**

实验数据集：两块三维网格地形（Mesh 1：270k 顶点/537k 面；Mesh 2：59k 顶点/118k 面），50 对起止点（分为光滑斜坡与崎岖地形）；以及真实户外环境中的AgileX Scout Mini机器人实验。

**📈 对比分析**

与基线方法（MPPI、GAKD、固定ε、无稳定约束）比较，ϵ‑MPC 在 94% 的成功率、平均规划时间 35 ms、最大姿态偏差 18.7°、路径长度偏差 8.5 m 方面均优于 MPPI（87%/29 ms/24.6°/7.8 m）和 GAKD（89%/38 ms/19.5°/9.2 m）；在真实世界中 80% 成功率、26.4° 最大偏差，仍明显高于 MPPI（67%/32.8°）和 GAKD（73%/30.2°）。

**⚠️ 局限性**

局限性：① 在动态或含运动障碍的环境中尚未验证；② 需要高质量三维网格和地形描述符，易受感知误差影响；③ 采样与ε阈值调节仍有经验性参数，可能在极端地形下表现欠佳；④ 计算量相较于纯随机采样略高，需进一步优化。

---

## 493. FTerViT: Fully Ternary Vision Transformer

**arXiv ID:** 2605.21171 | [PDF](https://arxiv.org/pdf/2605.21171v1)

**作者:** Szymon Ruciński `[一作]` (CSEM), Nadim Maamari `[通讯]` (CSEM)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了完全三值化的Vision Transformer（FTerViT），将所有权重矩阵和归一化参数压缩到{-1,0,+1}，并成功在资源受限的MCU上部署。

**💡 创新点**

首次证明了Patch Embedding、LayerNorm和分类头等最脆弱层可以使用三值化，并提出了TernaryBitConv2d和TernaryLayerNorm两种新算子，同时使用两阶段知识蒸馏加轻量化恢复训练实现高精度。

**🔧 技术方法**

采用三值量化、知识蒸馏（同架构）、量化感知恢复（QAD）、直通估计器（STE）、通道级缩放、FP32教师指导以及纯C推理引擎。

**📊 数据集**

主要使用ImageNet‑1K进行训练与评估，CIFAR‑10/100用于小模型的验证。

**📈 对比分析**

与现有二值/多比特ViT方法对比，在ImageNet上W2A8 DeiT‑III‑S 384×384实现82.43% top‑1，模型仅6.09 MB（比FP32损失2.42pp），在ESP32‑S3上实现79.64% top‑1，显著优于同类三值ViT。

**⚠️ 局限性**

仅在小型ViT上验证，未评估更大模型；推理核未做位压缩优化；两阶段训练流程可进一步合并为单阶段。

---

## 494. Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation

**arXiv ID:** 2605.21125 | [PDF](https://arxiv.org/pdf/2605.21125v1)

**作者:** Xixiang He `[一作]` (National University of Defense Technology), Qingyong Hu `[通讯]` (Intelligent Game and Decision Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对Group Relative Policy Optimization（GRPO）中出现的优势崩塌问题进行诊断并提出解决方案。

**💡 创新点**

创新点包括：①提出首个实时诊断指标优势崩塌率（ACR）；②设计Adaptive Virtual Sample Policy Optimization（AVSPO），在检测到优势崩塌时注入虚拟奖励样本恢复梯度。

**🔧 技术方法**

采用技术手段有：GRPO框架、二元验证奖励、ACR实时监控、虚拟奖励样本插入、动态阈值适配与梯度更新。

**📊 数据集**

使用的数据集包括六个数学推理基准（MATH‑500、GSM8K、Minerva、OlympiadBench、AMC、AIME24）以及跨领域验证基准MMLU‑Pro。

**📈 对比分析**

与基线GRPO、DCPO、INTUITOR、RENT等方法对比，AVSPO在0.5B–14B规模模型上平均提升4–6个百分点，优势崩塌率从28–45%下降至11–18%，表现显著。

**⚠️ 局限性**

局限性：目前仅在二元奖励、数学推理任务中验证；虚拟样本的分布和数量仍需手动设定，可能在不同任务或更大模型规模下引入偏差；对超参数的敏感性需进一步研究。

---

## 495. RankE: End-to-End Post-Training for Discrete Text-to-Image Generation with Decoder Co-Evolution

**arXiv ID:** 2605.21195 | [PDF](https://arxiv.org/pdf/2605.21195v1)

**作者:** Siyong Jian `[一作]` (Westlake University), Huan Wang `[通讯]` (Westlake University)

**通讯引用:** 19719 | [OpenAlex ID](https://openalex.org/A5100332013)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种端到端后训练框架RankE，能够同时演化AR策略和VQ解码器，缓解潜在的latent covariate shift；

**💡 创新点**

创新点是首次让VQ解码器与AR策略共进化，通过排名机制和Rank-GAN实现奖励驱动的像素级对齐，打破冻结解码器导致的FID‑CLIP trade‑off；

**🔧 技术方法**

采用GRPO策略优化、Rank‑GAN奖励加权对抗学习、EMA一致性正则化以及EM框架的交替优化，结合非梯度奖励的处理；

**📊 数据集**

主要使用MS‑COCO 30K、BLIP3o‑60k（用作SFT语料）和HPSv2作为评估数据集；

**📈 对比分析**

与冻结解码器的RL基线对比，RankE在MS‑COCO 30K上同时将FID从17.76提升至15.21，将CLIP从32.88提升至33.76，并在Janus‑Pro上取得更优的零样本GenEval表现；

**⚠️ 局限性**

限制包括较高的显存占用、对SFT语料与预训练分布匹配的敏感性以及仍冻结VQ编码器等。

---

## 496. OCTOPUS: Optimized KV Cache for Transformers via Octahedral Parametrization Under optimal Squared error quantization

**arXiv ID:** 2605.21226 | [PDF](https://arxiv.org/pdf/2605.21226v1)

**作者:** Mark Boss `[一作]` (Stability AI), Shimon Vainer `[通讯]` (Stability AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于八面体参数化的 KV 缓存压缩编码器，联合量化旋转后的三元组方向与范数，实现低位宽高效推理。

**💡 创新点**

将 KV 缓存三元组的方向映射到二维八面体坐标，并用非均匀 (b+1,b-1) 位分配与 Lloyd‑Max 量化，获得比单坐标量化更优的 MSE。

**🔧 技术方法**

使用结构化随机正交旋转（Walsh-Hadamard）、八面体映射、Lloyd‑Max 量化、联合三元组量化和可选 1‑bit QJL 余量估计。

**📊 数据集**

在合成高斯键、长上下文语言模型（Qwen2.5‑7B‑Instruct）、自回归视频（CausVid、Causal Forcing）和音频（AAR）四种模态上进行评估。

**📈 对比分析**

与 TurboQuant-MSE、TurboQuant-QJL 及 PolarQuant 采用相同旋转和位宽做基准，在所有模态下均匹配或超越基线，尤其在 2‑bit 级别下保持较低的 MSE、IP误差与检索召回率。

**⚠️ 局限性**

解码算术更复杂，速度低于 bf16 SDPA；在极低位宽下仍会出现显著性能衰退，仅在 KV 带宽或容量受限时最具吸引力。

---

## 497. Metaphors in Literary Post-Editing: Opening Pandora's Box?

**arXiv ID:** 2605.21178 | [PDF](https://arxiv.org/pdf/2605.21178v1)

**作者:** Aletta G. Dorst `[一作]` (Leiden University Centre for Linguistics), Katinka Zeven `[通讯]` (Leiden University Centre for Linguistics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过让六名新毕业的文学译者对三段文学文本进行后编辑，评估神经机器翻译（NMT）和大型语言模型（LLM）在翻译隐喻时的表现。

**💡 创新点**

创新点在于首次系统量化多词隐喻在MT输出中的错误率，并探讨后编辑者对隐喻的注意力与修改行为之间的关联。

**🔧 技术方法**

研究使用了Google Translate、DeepL及ChatGPT（GPT‑4）三大公开机器翻译引擎生成的翻译结果，并对后编辑者的修改进行编码分析。

**📊 数据集**

数据集为VU Amsterdam Metaphor Corpus（VUAMC）中的三段选自《Lucy Ghosts》的文学片段，共78个单词隐喻与28个多词隐喻。

**📈 对比分析**

比较方法是对后编辑者在每个引擎输出中所做的隐喻修改次数进行统计，结果显示约三分之一的隐喻需修改，且多词隐喻的错误率超过50%。

**⚠️ 局限性**

局限性包括样本规模仅六名编辑、仅三段文本、单一语言对以及未区分所做修改是必需还是可选，无法全面评估MT质量与后编辑难度。

---

## 498. ScenePilot: Controllable Boundary-Driven Critical Scenario Generation for Autonomous Driving

**arXiv ID:** 2605.21168 | [PDF](https://arxiv.org/pdf/2605.21168v1)

**作者:** Qiyu Ruan `[一作]` (University of Macau), Cheng-zhong Xu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了ScenePilot框架，用于在物理可行性与自主驾驶策略可行性边界附近生成安全关键场景，从而系统地暴露车辆堆栈的缺陷。

**💡 创新点**

创新点在于：① 将物理可行性（RSS衍生的σ）与策略可行性（在线学习的风险预测Φ）拆分为两条约束；② 在多目标强化学习中引入ε阈值和步级可行性屏蔽，精准聚焦“物理可行却策略失效”的边界带；③ 通过可行性阈值扫描和可行性屏蔽，使生成的场景更均匀且更具诊断价值。

**🔧 技术方法**

采用基于RSS的物理安全评分、在线学习的风险预测网络、约束多目标PPO与步级屏蔽的强化学习算法，并结合可行性阈值扫描实现对边界带的系统探索。

**📊 数据集**

使用SafeBench数据集（基于CARLA的八种典型交通场景），在不同路段与不同控制器上训练与评估。

**📈 对比分析**

与LC、AdvSim、CARLA Scenario Generator、Adversarial Trajectory Optimization、ChatScene、SCSG、SCENGE等方法对比，ScenePilot在SafeBench上取得最高碰撞率0.893、最低Overall Score 0.476，且在多种控制器和更高密度交通条件下仍保持优越性；在对Ego策略进行对抗微调后，ScenePilot生成的场景能显著降低碰撞率并提升整体分数。

**⚠️ 局限性**

局限性包括：① 场景多样性受限于固定路段，主要集中在短时间窗口内；② 采用代理驱动的top‑k选择可能偏向特定代理，导致对不同策略的泛化性有限。

---

## 499. Q-SYNTH: Hybrid Quantum-Classical Adversarial Augmentation for Imbalanced Fraud Detection

**arXiv ID:** 2605.21164 | [PDF](https://arxiv.org/pdf/2605.21164v1)

**作者:** Adam Innan `[一作]` (Hassan II University of Casablanca), Mohamed Bennai `[通讯]` (Hassan II University of Casablanca)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了Q-SYNTH，一个混合量子-经典对抗生成框架，用于在信用卡欺诈检测中合成少数类欺诈样本。

**💡 创新点**

创新点在于将参数化量子电路作为生成器，配合经典判别器，实现了在分布相似性和下游检测性能之间取得平衡，并通过统一协议同时评估KS、Wasserstein距离、AUC等指标，展示量子生成器在分布拟合上优于经典GAN。

**🔧 技术方法**

技术包括量子生成器（变分量子电路）、经典判别器（双层LeakyReLU网络）、KS/Wasserstein分布相似度评估、AUC检测、SMOTE、经典GAN、QNN、ANN、Logistic Regression、Random Forest、XGBoost等分类器。

**📊 数据集**

使用Kaggle信用卡欺诈数据集（含极端类别不平衡的信用卡交易记录）。

**📈 对比分析**

与SMOTE、经典GAN以及多种传统分类器比较，Q-SYNTH在KS和Wasserstein指标上显著优于经典GAN且接近SMOTE，在下游召回率和F1得分上与GAN相近，整体表现处于分布拟合与检测性能的折中点。

**⚠️ 局限性**

局限包括仅评估特征维度的相似性、使用压缩表示（PCA）导致信息损失、未在真实量子硬件上评估噪声影响、缺乏对全局依赖结构的评估、对超参数和初始化敏感。

---

## 500. High-speed Networking for Giga-Scale AI Factories

**arXiv ID:** 2605.21187 | [PDF](https://arxiv.org/pdf/2605.21187v1)

**作者:** Sajy Khashab `[一作]` (NVIDIA), Mark Silberstein `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Spectrum-X多平面硬件加速网络架构，用于大规模AI训练集群的高速、低延迟、可隔离的分布式通信；

**💡 创新点**

创新点包括：1）多平面拓扑取代传统层级深度，显著提升bisection带宽与容错；2）硬件加速的自适应路由与平面负载均衡器（PLB），实现微秒级流量重分配；3）控制循环分离（端点层、平面层、交换机层），实现快速恢复和高隔离；4）结合PFC与改进的拥塞控制，专为同步collective设计；

**🔧 技术方法**

技术手段涵盖：RDMA over Converged Ethernet (RoCEv2)、硬件加速自适应路由、平面级负载均衡器、PFC与基于RTT+ECN的拥塞控制、BGP驱动的加权路由、全频率网络遥测、仿真器NSX等；

**📊 数据集**

使用的基准数据集包括：NCCL同步collective（All-Reduce, All-Gather等）、RDMA bisection微基准、AI训练工作负载（DeepSeek-V3 LLM、Nemotron 3 Ultra）、以及大规模（256K~512K GPU）模拟实验；

**📈 对比分析**

与传统单平面RoCEv2+DCQCN/ECMP方案对比，Spectrum-X在高负载下实现98%线速、P99延迟仅8–9µs、可隔离性提升至≈99%，动态故障恢复时间从≈1s降至≈3ms，LLM训练步骤时间保持不变或略微下降；

**⚠️ 局限性**

局限性包括：需专用硬件（NVIDIA ConnectX NICs及Spectrum交换机）和复杂的多平面光纤布线；BGP权重更新仍有时延，极端规模下可能导致路由收敛挑战；目前主要针对RoCEv2工作负载，未覆盖非RDMA协议或混合网络环境；

---

## 501. Detecting Trojaned DNNs via Spectral Regression Analysis

**arXiv ID:** 2605.21146 | [PDF](https://arxiv.org/pdf/2605.21146v1)

**作者:** Samuele Pasini `[一作]` (Università della Svizzera Italiana), Paolo Tonella `[通讯]` (Università della Svizzera Italiana)

**通讯引用:** 11290 | [OpenAlex ID](https://openalex.org/A5025438762)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于内部预激活谱变化的模型更新检测方法，以在模型微调过程中识别植入的后门

**💡 创新点**

将后门检测视为回归问题，利用预激活谱的统计分布对更新进行异常检测，避免了触发器重构的假设，提供了对更新过程的透明度和可解释性

**🔧 技术方法**

预激活谱分析、Mahalanobis距离统计、回归基准分布（CSDD）、模型内部特征提取和基准实验

**📊 数据集**

CIFAR-10、SVHN、GTSRB、CelebA 四个视觉分类数据集

**📈 对比分析**

与 Neural Cleanse、ABS、FeatuRE 等三种主流后门检测器比较，单步更新下平均准确率为 0.95（最佳 0.98），比基线提升约 20%；多步更新下仍保持 0.89 的平均准确率，表现出良好的鲁棒性

**⚠️ 局限性**

仅适用于拥有可信参考模型且可追溯更新的场景，对无参考或全新训练的模型无法检测；在长周期多次更新时需重新估计参考分布以保持性能

---

## 502. UniT: Unified Geometry Learning with Group Autoregressive Transformer

**arXiv ID:** 2605.21131 | [PDF](https://arxiv.org/pdf/2605.21131v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 503. Safety-Critical Control for Smoothed Implicit Contact Dynamics

**arXiv ID:** 2605.21138 | [PDF](https://arxiv.org/pdf/2605.21138v1)

**作者:** Haegu Lee `[一作]` (Maersk Mc-Kinney Moller Institute, University of Southern Denmark), Christoffer Sloth `[通讯]` (Maersk Mc-Kinney Moller Institute, University of Southern Denmark)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究提出了针对光滑隐式接触动力学的安全关键控制方法，使用离散时间控制屏障函数（CBF）过滤接触力以满足安全约束。

**💡 创新点**

创新点在于将接触力的隐式模型用一阶泰勒展开局部线性化，并结合中央路径参数 κ 的屏障安全分析，提出基于边界的 κ 选取与鲁棒 CBF 加紧修正。

**🔧 技术方法**

使用隐式接触动力学（NCP）、离散时间 CBF、Taylor 线性化、鲁棒性紧缩以及离散时间 QP 进行实现。

**📊 数据集**

实验数据集为四个接触丰富的机器人系统：一维盒子接触、平面推、盒子支点和跳跃机（hopper）场景。

**📈 对比分析**

与标准 CBF 以及无安全约束的基准进行对比，实验表明鲁棒 CBF 在所有系统中消除了接触力违规，且提升了任务成功率。

**⚠️ 局限性**

局限性在于需针对每个系统预先进行 κ 选取和紧缩参数 δ 的离线调优，且在复杂或极端接触情形下鲁棒性仍可能不足。

---

## 504. Distill to Think, Foresee to Act: Cognitive-Physical Reinforcement Learning for Autonomous Driving

**arXiv ID:** 2605.21139 | [PDF](https://arxiv.org/pdf/2605.21139v1)

**作者:** Yang Wu `[一作]` (Nanjing University of Science and Technology), Jin Xie `[通讯]` (Nanjing University)

**通讯引用:** 21151 | [OpenAlex ID](https://openalex.org/A5039338731)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种认知-物理融合的强化学习框架，通过将视觉语言模型的认知先验蒸馏到BEV编码器，构建自回归BEV世界模型进行未来语义预测，并利用双奖励（物理安全与认知一致）通过GRPO优化驾驶策略，实现端到端安全驾驶并支持语言指令控制。

**💡 创新点**

创新点包括：1) 在端到端驾驶中引入认知-物理双管齐下的框架；2) 通过VLM蒸馏在不增加推理成本的前提下实现认知先验；3) 构建可解释的BEV世界模型作为物理沙盒；4) 双奖励策略将硬物理约束与软认知偏好统一；5) 支持用户命令的可插拔认知通道。

**🔧 技术方法**

采用的技术包括：VLM蒸馏（InternVL3+LoRA）、BEV编码与跨模态对齐、Transformer自回归世界模型、GRPO强化学习、物理安全指标（碰撞、可行驶区、进度、时间-碰撞、舒适度）与认知奖励、CLIP文本编码。

**📊 数据集**

使用的主要数据集为NAVSIM v1和v2（基于OpenScene），并利用多种驾驶VQA数据集进行VLM微调。

**📈 对比分析**

与18种最新方法（包括IL、RL、世界模型等）对比，本文在NAVSIM v1取得PDMS 91.4（高于第二名DriveDPO的+1.4），在v2取得EPDMS 86.1，安全子指标（无事故、可行驶区、交通灯遵守）均名列前茅，显示出显著性能提升。

**⚠️ 局限性**

主要局限包括：1) 依赖BEV世界模型的预测精度，过长预测窗口会累积误差；2) VLM蒸馏在领域外迁移时可能不够鲁棒；3) 对实时性能的评估有限，尚未在真实车辆上验证；4) 对极端稀有场景的泛化尚待进一步研究。

---

## 505. Reasoning-Trace Collapse: Evaluating the Loss of Explicit Reasoning During Fine-Tuning

**arXiv ID:** 2605.21127 | [PDF](https://arxiv.org/pdf/2605.21127v1)

**作者:** Lukas Twist `[一作]` (King's College London), Jie M. Zhang `[通讯]` (King's College London)

**通讯引用:** 4353 | [OpenAlex ID](https://openalex.org/A5088708850)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在对已训练的显式推理模型进行下游微调时，模型如何在保持最终答案准确性的同时失去完整推理轨迹（即“推理轨迹崩塌”）并提出结构化评估框架和轻量级损失掩蔽方法来监测与缓解此现象。

**💡 创新点**

①提出推理轨迹崩塌的定义与结构化评估框架；②首次系统测量推理轨迹完整性与答案准确性之间的关系；③展示简单的损失掩蔽即可显著保留推理轨迹，且不需要教师生成轨迹；

**🔧 技术方法**

构建轻量化评估库（Reasoner），实现统一的聊天模板、轨迹解析、结构化指标计算和损失掩蔽；采用LoRA微调、AdamW优化、cosine学习率调度；对四种小型推理模型分别进行实验；

**📊 数据集**

在化学问题集Chemistry L‑3（含答案解释）进行下游微调；评估数据包括数学推理（math）和代码生成（code）等三大任务；

**📈 对比分析**

比较方法：标准微调、损失掩蔽（masked‑think、response‑only）和教师蒸馏；评估指标包括最终答案准确率、有效推理率（valid reasoning rate）、空/缺失/截断率以及推理条件准确率。实验表明：标准微调快速导致有效推理率降至0‑30%，但答案准确率仍可维持或提升；损失掩蔽显著保持有效推理率至40‑70%，且与答案准确率相近；教师蒸馏对部分模型有效，但对某些模型（如X）表现不佳；

**⚠️ 局限性**

仅评估四种小型模型，未涵盖大规模或其他架构；仅使用单一微调数据集，外部有效性有限；实验仅采用LoRA和贪婪解码，未探究其他适配器或随机采样的影响；指标聚焦推理轨迹结构完整性，未评价轨迹的准确性或有用性。

---

## 506. Do LLMs Know What Luxembourgish Borrows? Probing Lexical Neology in Low-Resource Multilingual Models

**arXiv ID:** 2605.21227 | [PDF](https://arxiv.org/pdf/2605.21227v1)

**作者:** Nina Hosseini-Kivanani `[一作]` `[通讯]`, Nina Hosseini-Kivanani

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了基于LuxBorrow的LexNeo-Bench，用以评估多语言LLM在卢森堡语新闻中对形态改编借词的识别与归属判断。

**💡 创新点**

首次引入结构化语言知识图（LKG）并证明其显著提升LLM对借词分类与二元归属判断的性能，同时提出二元新词识别任务揭示LLM在时序认知方面的不足。

**🔧 技术方法**

采用Prompting技术（零样本、少样本、知识图平面与图形上下文）与OpenAI接口下的指令微调LLM（Gemma 12B/27B、Llama 70B），并使用多任务评估指标（准确率、宏F1等）。

**📊 数据集**

使用RTL 1999–2025新闻语料经过LuxBorrow标注得到的句子级语言识别与词级借词标签；同时调用Lëtzebuerger Online Dictionnaire (LOD)的词典信息作为知识图构建与提示。

**📈 对比分析**

与零样本、少样本、知识图平面与图形等17种提示策略在三模型上比较，零样本约25%准确，KG‑graph提升至71–81%；二元新词识别最高约50%准确，KG‑graph对该任务表现不佳。

**⚠️ 局限性**

仅覆盖编辑新闻的正式体裁，借词标签自动化可能存在误差；仅评估三种LLM，缺乏大规模多语种覆盖；知识图缺少时序特征，无法完全解决新词判断。

---

## 507. ASIND: Alternating Sparse Identification for Predicting Network Dynamics Without Knowledge

**arXiv ID:** 2605.21220 | [PDF](https://arxiv.org/pdf/2605.21220v1)

**作者:** Mingyu Kang `[一作]` (University of Science and Technology of China), Linyuan Lv `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种交替稀疏识别网络动力学（ASIND）算法，能够在不知道自动力函数 F、交互函数 G 和网络结构 A 的情况下，交替稀疏识别它们。

**💡 创新点**

创新点在于将 F、G、A 三者联合参数化，并通过交替优化实现无先验知识下的稀疏识别，同时揭示网络结构的弱可识别性。

**🔧 技术方法**

使用了稀疏优化、拉格朗日乘子法、增量式 ADMM 等技术，构建了二次规划求解器来交替更新 A、w 和 λ。

**📊 数据集**

使用了四种经典网络动力学模型（Kuramoto、SIS、LV、MM）以及三种网络拓扑（ER、WS、BA）生成的数据集，网络规模为 16。

**📈 对比分析**

与无先验知识的 SINDy 基线进行对比，评估指标为 RMSE 和 MAPE，ASIND 在 100 步预测上显著优于 SINDy，误差低于 1%，而 SINDy 的误差可达数百至数千倍。

**⚠️ 局限性**

局限在于网络规模上限约为 100，且网络结构的弱可识别导致重构网络与真实网络相差较大；实验规模和复杂度有限，未能覆盖更大或更真实的系统。

---

## 508. Supporting Dynamic Control-Flow Execution for Runtime Reconfigurable Processors

**arXiv ID:** 2605.21203 | [PDF](https://arxiv.org/pdf/2605.21203v1)

**作者:** Hassan Nassar `[一作]` (Karlsruhe Institute of Technology), Jörg Henkel `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为可重配置处理器实现动态控制流微代码执行，支持循环、条件跳转和异常处理，并为四种典型应用（SIFT、SWE、CNN、SHA‑3）设计专用指令与加速器。

**💡 创新点**

首次将动态跳转子指令、暂停与多类异常集成到可重配置处理器的微码执行中，实现了更灵活、更高效的算法实现。

**🔧 技术方法**

采用VLIW微指令、FPGA可重配置加速器、专用指令集（SI）、动态跳转指令以及异常处理机制。

**📊 数据集**

在四个领域的基准测试中：SIFT使用公开图像集；SWE使用标准海域网格；CNN使用常用图像分类数据集；SHA‑3使用不同长度的文本/消息样本。

**📈 对比分析**

在Xilinx VC707板上，裸机Leon3 CPU与加速器/微码实现对比，平均运行时间提升27×，CNN与SHA‑3最高达42×，SIFT与SWE分别提升14×和7×。

**⚠️ 局限性**

硬件扩展虽然占用资源（如CNN‑MAC接近LUT上限），但受限于5个加速器槽，且动态控制流仍需手动为每种应用编写SI，缺乏自动生成机制。

---

## 509. SAM-Sode: Towards Faithful Explanations for Tiny Bacteria Detection

**arXiv ID:** 2605.21186 | [PDF](https://arxiv.org/pdf/2605.21186v1)

**作者:** Wanying Tan `[一作]` (Shenzhen University), Sihong Xie `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

SAM‑Sode框架通过将归因点转换为SAM3的视觉提示，并结合物理意义与几何一致性双约束，对微小细菌检测结果进行掩膜重构和去噪，从而提高可解释性。

**💡 创新点**

创新点在于将传统梯度归因点直接作为SAM3的提示输入，利用SAM3的几何重构能力以及双约束门控（MaskIoU与MaskScore）实现实例级归因去噪，解决细菌等极小目标归因分散和背景冗余的问题。

**🔧 技术方法**

使用技术包括：集成梯度（IG）和Grad‑CAM归因、SAM3 Promptable分割、随机切片策略生成多样化掩膜、MaskIoU与MaskScore两维量化评估，以及双约束门控和实例归一化过滤。

**📊 数据集**

使用的数据集为自建的TBC‑Micro细菌检测数据集（2524张图像、57472个边框标注，背景为复杂电路）和公开的AGAR培养皿数据集（1802张图像）。

**📈 对比分析**

与传统IG、Grad‑CAM等归因方法在两套数据集上对比，SAM‑Sode在抑制背景噪声、提升归因可视化质量方面显著优于对照方法；实验中MaskIoU和MaskScore均有明显提升，说明可解释性更强。

**⚠️ 局限性**

局限性包括：依赖SAM3生成掩膜，随机切片策略增加计算开销；对极小或模糊目标的归因仍受图像分辨率限制；缺乏对不同细菌种类形态差异的进一步适配研究。

---

## 510. On the Identifiability of Semi-Blind Estimation in Cell-Free Massive MIMO Networks

**arXiv ID:** 2605.21181 | [PDF](https://arxiv.org/pdf/2605.21181v1)

**作者:** Christian Forsch `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Laura Cottatellucci `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 2037 | [OpenAlex ID](https://openalex.org/A5064586027)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了CF-MaMIMO网络中半盲估计的可识别性，并从大规模系统设计角度推导了可识别性区域。

**💡 创新点**

提出用Poisson点过程生成的BRGG网络通过匹配度分布逼近独立边图，构建递归概率分析得到可识别性阈值。

**🔧 技术方法**

利用Karp‑Sipser算法、稀疏随机图密度演化理论、泊松点过程与几何随机图模型以及Lambert W函数求解临界点。

**📊 数据集**

通过Monte Carlo仿真生成随机网络，未使用公开数据集，所有数据均为仿真产生。

**📈 对比分析**

与独立边图理论对比，仿真验证了可识别性阈值的 sharp phase transition，误差仅在几百分之内，性能表现良好。

**⚠️ 局限性**

对BRGG的近似仍有限，未考虑多路径衰落、硬件非理想和噪声效应，仅在理想无噪声条件下讨论可识别性。

---

## 511. ChunkFT: Byte-Streamed Optimization for Memory-Efficient Full Fine-Tuning

**arXiv ID:** 2605.21177 | [PDF](https://arxiv.org/pdf/2605.21177v1)

**作者:** Yongkang Liu `[一作]` (Northeastern University), Hinrich Schütze `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ChunkFT，一种通过旋转字节均衡的参数块（chunk）来实现内存高效的全参数微调框架。

**💡 创新点**

创新点在于：①以字节级训练成本划分参数块而非层级，②仅在激活块上进行梯度计算与优化，③支持任意子张量梯度更新，保持原训练目标与标准 Transformer 兼容。

**🔧 技术方法**

核心技术包括：基于 PyTorch 的块级反向传播实现、异步 CPU‑GPU 权重与优化器状态迁移、混合精度训练、块轮转调度与梯度累积。

**📊 数据集**

实验使用 Llama‑2‑7B、Llama‑3‑8B、Llama‑3‑70B、RoBERTa‑Large 等大模型；数据集涵盖 SuperGLUE、MathBench、MT‑Bench、BoolQ、COPA、WiC、MultiRC 等。

**📈 对比分析**

与 Adam、LoRA、GaLore、BAdam、HiFT、APOLLO、LOMO 等基线比较，ChunkFT 在 24 GB GPU 上实现 13.7 GB 的峰值显存，支持更大批量；训练速度比 BAdam/HiFT 快 10–20 %；在语言理解、数学推理与指令跟随任务上，性能与完整参数微调持平或略优，且记忆消耗显著降低。

**⚠️ 局限性**

局限性包括：需要调节块数 K 与更新间隔 T 的超参数；块轮转导致每块的更新频率不均衡，可能影响极小任务的收敛；实现复杂度高，需定制块级反向传播和异步数据迁移。

---

## 512. Domain-Adaptable Reinforcement Learning for Code Generation with Dense Rewards

**arXiv ID:** 2605.21180 | [PDF](https://arxiv.org/pdf/2605.21180v1)

**作者:** Erfan Aghadavoodi Jolfaei `[一作]` (Technische Universität Darmstadt), Mira Mezini `[通讯]` (Technische Universität Darmstadt)

**通讯引用:** 8202 | [OpenAlex ID](https://openalex.org/A5078067853)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于PPO的强化学习框架，利用可执行反馈和多维奖励对大型语言模型进行细粒度微调，以提升代码生成的语法正确性、功能正确性、安全性和领域适应性。

**💡 创新点**

创新点在于：① 将稀疏的序列级奖励转化为密集的 token‑level 奖励，实现更精准的信用分配；② 设计可定制的多组件奖励体系，包括语法检查、代码风格/漏洞检测、KL 正则化以及任务特定奖励；③ 将机器人仿真环境反馈直接嵌入奖励中，实现环境感知的代码生成。

**🔧 技术方法**

技术手段主要是 Proximal Policy Optimization（PPO）与价值网络的组合，使用 token‑level 奖励映射、静态代码分析工具（Ruff linter）、语法约束（SynCode）以及仿真器 RoboSim 等评估组件。

**📊 数据集**

实验数据集包括通用代码生成的 OpenCodeInstruct、MBPP/MBPP+ 评测集，以及机器人程序合成的 Robo-Instruct 和 RoboEval 基准。

**📈 对比分析**

与基线 Qwen2.5‑Coder（1.5B）比较，经过 RL 微调后 MBPP pass@1 由 0.460 提升至 0.653（+19%），MBPP+ 从 0.413 提升至 0.556；在 RoboEval 上 Python 错误率从 77/80 降至 11/80，成功率提升至 14/80，执行失败率下降 51%。

**⚠️ 局限性**

局限性包括：RL 方法对大型模型的适用性有限，较小模型（1.5B）仍落后于更大模型（7B）；奖励分配仍受限于可解释性和计算开销；缺乏对指令对齐和长度偏差的深入研究；以及对更广泛执行反馈机制的探索不足。

---

## 513. Humanoid Whole-Body Manipulation via Active Spatial Brain and Generalizable Action Cerebellum

**arXiv ID:** 2605.21133 | [PDF](https://arxiv.org/pdf/2605.21133v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 514. SMoA: Spectrum Modulation Adapter for Parameter-Efficient Fine-Tuning

**arXiv ID:** 2605.21147 | [PDF](https://arxiv.org/pdf/2605.21147v1)

**作者:** Yongkang Liu `[一作]` (Northeastern University), Hinrich Schütze `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种 Spectrum Modulation Adapter（SMoA），通过把冻结的预训练权重按谱块划分并在每个对角块上使用 Hadamard 调制的低秩分支，实现参数高效微调。

**💡 创新点**

创新点在于将谱结构显式利用，分块对齐并对每块做 Hadamard 调制，从而在保持低参数预算的同时扩大可用更新族，突破单一低秩分支对谱尾的限制。

**🔧 技术方法**

使用了奇异值分解、谱块重排、Hadamard 乘积以及低秩分解等技术，构造块对角的更新。

**📊 数据集**

在 Llama‑2‑7B 和 Llama‑3‑8B 上对 Commonsense Reasoning、Dialogue Generation (ConvAI2) 与 Mathematical Reasoning (GSM8K) 等多种基准数据集进行实验。

**📈 对比分析**

与 LoRA 及其变体（HiRA、MeLoRA、DoRA 等）对比，SMoA 在同等或更少的可训练参数下平均准确率提升约 0.4–1.5%，并在低预算设置下保持竞争力。

**⚠️ 局限性**

局限在于对跨块交互的残差建模能力有限，且当任务残差主要由跨块成分驱动时，结构诱导可能变为限制。

---

## 515. Complete Supermartingale Certificates for $ω$-Regular Properties

**arXiv ID:** 2605.21134 | [PDF](https://arxiv.org/pdf/2605.21134v1)

**作者:** Alessandro Abate `[一作]` (University of Oxford), Diptarko Roy `[通讯]` (University of Birmingham)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5014656403)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一套针对一般状态空间时不变马尔可夫链的可接受性（reactivity）性质的完整证明规则，通过将每个 Streett 条件拆解为吸收区域和全局不变量的组合，并给出对应的超马氏子证书。

**💡 创新点**

创新点在于首次提供了对 ω‑regular（即 reactivity）性质的几乎必然完整与 ε‑完整的超马氏子证明规则，并通过吸收区域分解把问题转化为安全、终止与安全三类子问题，理论上实现了对一般状态空间的完整性。

**🔧 技术方法**

使用的技术包括马尔可夫链理论、σ‑代数与转移核、超马氏子与其期望性质、Orey 定理、吸收区域与不变量的构造、停顿时间与返回时间的分析，以及不变量与安全/终止的证明规则。

**📊 数据集**

论文为理论工作，未使用标准数据集；示例采用“Lending Casino”随机游走模型来演示方法。

**📈 对比分析**

由于是理论性研究，未做实验对比；相较于以往仅给出充分条件的证明规则，本文提供了完整性保证，并通过示例展示了在理论框架下的可证明性提升。

**⚠️ 局限性**

局限性包括：完整性证明仅在可数无限状态空间实现；对不可数状态空间或非时间齐性的马尔可夫链尚无完整规则；方法假设马尔可夫链无非确定性（未扩展到 MDP 或随机游戏）。

---

## 516. Reinforcement Learning-based Control via Y-wise Affine Neural Networks: Comparative Case Studies for Chemical Processes

**arXiv ID:** 2605.21211 | [PDF](https://arxiv.org/pdf/2605.21211v1)

**作者:** Austin Braniff `[一作]` (West Virginia University), Yuhe Tian `[通讯]` (West Virginia University)

**通讯引用:** 1004 | [OpenAlex ID](https://openalex.org/A5083266245)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发并验证了一种基于YANN（Y-wise Affine Neural Network）的强化学习控制方法，并在三个典型化学过程（CSTR、四罐系统、多级提取塔）上进行实验评估。

**💡 创新点**

创新点在于：①将线性MPC的显式分段仿射解嵌入到神经网络中，实现策略和价值网络的理论可靠初始化；②通过YANN构造的Actor‑Critic框架实现了在训练初期即可接近NMPC性能的控制器；③显著降低了训练数据需求与计算量，提升了可解释性与安全性。

**🔧 技术方法**

技术包括：YANN架构、YANN-DDPG算法、线性化模型（Jacobian/系统辨识）、多参数MPC（mp-MPC）显式解、传统Actor‑Critic强化学习框架（DDPG、TD3、PPO、SAC）以及非线性模型预测控制（NMPC）作为基准。

**📊 数据集**

使用公开的PC‑Gym数据集，包含连续搅拌釜反应器（CSTR）、四罐系统和五级提取塔的动力学模型。

**📈 对比分析**

与PPO、SAC、DDPG、TD3及NMPC比较。YANN-RL在训练轮次为传统RL四倍时即可达到或接近NMPC的误差与累计成本；在CSTR和提取塔中误差低于传统RL，且在四罐系统中避免了不可行操作，表现出更高的稳态误差与安全性。

**⚠️ 局限性**

局限性包括：①仍未在完整非线性模型下验证长期稳定性；②对线性近似的依赖可能在极端非线性或模型不确定性强的系统中表现不佳；③缺乏对噪声、扰动与实时在线适应的完整评估。

---

## 517. PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection

**arXiv ID:** 2605.21207 | [PDF](https://arxiv.org/pdf/2605.21207v1)

**作者:** Xiaoyu Zhou `[一作]` (Jinan University), Zhihua Xia `[通讯]` (Jinan University)

**通讯引用:** 6317 | [OpenAlex ID](https://openalex.org/A5005319671)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种峰值引导校准（Peak‑Guided Calibration, PGC）框架，用于检测高保真 AI 生成的图像和视频；

**💡 创新点**

创新点在于：①通过“峰值”聚合机制聚焦最具判别力的局部痕迹，抵消全局语义主导的干扰；②将局部峰值偏置与全局 logits 进行加性校准，实现对细粒度伪造痕迹的放大；

**🔧 技术方法**

主要技术包括：双流特征编码（残差域 + RGB 视觉 Transformer），峰值聚合模块（基于温度 soft‑max 的峰值选择），以及局部偏置校准的加性逻辑；

**📊 数据集**

使用的数据集有：①CommGen15（15 大商业生成模型 + COCO 真实图），②GenImage、UniversalFakeDetect、AIGI 等公开基准；

**📈 对比分析**

与 14 种现有检测器（CNNDet、FreDect、LGrad、UnivFD、PatchCraft、FreqNet、NPR、FatFormer、AIDE、CoD、B‑Free、DDA、Effort、SAFE）在 CommGen15 上平均准确率提升 12.3%，在 GenImage、AIGI、UniversalFakeDetect 等基准上也分别实现了 2.1%、3.5% 和 2.2% 的增幅，整体保持最高的准确率与 AP；

**⚠️ 局限性**

局限性：PGC 仍需对局部峰值的选择做参数调优；在极端后处理或全局一致性更强的生成模型上可能面临检出率下降；此外双流和峰值聚合增加了计算与存储开销。

---

## 518. The Statistical Significance of the Inclusion of Graph Neural Networks in the Financial Time Series Forecasting Problem

**arXiv ID:** 2605.21192 | [PDF](https://arxiv.org/pdf/2605.21192v1)

**作者:** Marco Gregnanin `[一作]` (IMT School for Advanced Studies), Maurizio Parton `[通讯]` (University of Chieti–Pescara)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种 Time‑Geometric 模型，将图神经网络与传统时间序列网络相结合，以同时捕捉时间和几何模式，从而提升金融单变量时间序列预测精度。

**💡 创新点**

创新点在于：① 用可视化图算法将单变量序列转换为图表示；② 在时间网络的基础上加入图卷积网络，形成“时间‑几何”双模态框架；③ 对多种基线模型进行系统的统计显著性检验，证明几何模式的加入对预测有显著提升。

**🔧 技术方法**

采用的技术包括：可视化图算法（Visibility Graph）、图卷积网络（GCN）、RNN/LSTM/GRU/Transformer/TCN 等时间序列模型、LSTM 用于几何部分的序列编码、全连接层融合、以及配对 t 检验、Wilcoxon、Sign、Friedman、Nemenyi 等多种统计检验方法。

**📊 数据集**

使用了来自 S&P 100 指数的 90 只股票的日频数据，包含收盘价、开盘价、最高价、最低价和成交量等 5 个特征，时间跨度约 2117 天。

**📈 对比分析**

通过在 1 天、5 天和 20 天的预测窗口内，对 16 种算法（基线 + Time‑Geometric）计算 RMSE、MAE、MAPE、MASE 四个指标，并对结果进行配对 t、Wilcoxon、Sign、Friedman 与 Nemenyi 检验。实验显示，Time‑Geometric 在大多数情况下均优于对应基线模型，尤其 Transformer 与 TCN 的改进最为显著，且统计检验表明改进具有显著性。

**⚠️ 局限性**

局限性包括：① 仅在单变量金融序列上验证，未检验多变量或非金融序列；② 性能提升相对有限，受预测 horizon 与图参数影响；③ 可视化图的构造和参数选择可能对结果产生影响；④ 未探讨模型在更大规模或更长周期数据上的可扩展性。

---

## 519. SURGE: An Event-Centric Social Media Sentiment Time Series Benchmark with Interaction Structure

**arXiv ID:** 2605.21198 | [PDF](https://arxiv.org/pdf/2605.21198v1)

**作者:** Chen Su `[一作]` (University of Science and Technology of China), Yan Song `[通讯]` (University of Science and Technology of China)

**通讯引用:** 25726 | [OpenAlex ID](https://openalex.org/A5013100135)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 SURGE，一个包含 67 个公共事件、三种时间粒度（1 天、12 小时、6 小时）的事件级时间序列，并配有对齐的文本与回复/转发互动结构；

**💡 创新点**

首次将事件级时间序列、对齐文本和社交媒体互动网络三者结合，提出文本增强、交互密集期评估和跨类别泛化的多模态基准协议；

**🔧 技术方法**

利用自动化抽取与清洗、LLM（Qwen3-32B）情感标注、三粒度分箱、结构化/扁平文本视图，并在实验中使用 DLinear、PatchTST、GPT4TS 等时序模型以及 MM‑TSF、GPT4MTS、CAMEF 等多模态模型和结构感知探测器；

**📊 数据集**

融合 Twitter、Reddit 与 Threads 共 93 个候选事件，最终筛选 67 个事件（共 817k 条英文帖子），覆盖自然灾害、政治、社会运动、技术发布、体育娱乐五大类；

**📈 对比分析**

通过 MAE、MSE 及结构感知 MAE_reply(k%) 进行评估，结果显示传统 Last‑Value 在 MAE 上仍占优，MSE 上深度模型表现更好；多模态模型在文本增强后提升有限，结构化文本对部分模型有正面影响；交互密集期与跨类别泛化更具挑战；

**⚠️ 局限性**

局限包括仅包含英文文本，缺乏多语言与多平台覆盖；仅关注预测任务，未提供分类、谣言检测等；情感标注采用未微调 LLM，细粒度情感和立场标注缺失；高互动期和跨类别评估仍有提升空间。

---

## 520. Information Leakage Envelopes

**arXiv ID:** 2605.21185 | [PDF](https://arxiv.org/pdf/2605.21185v1)

**作者:** Sara Saeidian `[一作]` (Inria), Catuscia Palamidessi `[通讯]` (Inria)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了点值最大泄漏（PML）封套（PML envelope）的概念，并系统地研究了其数学性质、上界与下界，以及在两类典型机制（PML 极端机制和 k‑随机响应机制）上的显式计算和数值评估。

**💡 创新点**

创新点包括：
- 引入 PML 封套，使得隐私保证在后处理下保持不变且对失败概率给出上界；
- 将 PML 与信息密度、信息流量等信息论概念联系起来，给出等价定义；
- 推导出通用的上界（最大泄漏 + log(1/δ)）和可计算的下界（基于事件泄漏和二元后处理）；
- 在两类机制上提供解析或半解析的 PML 封套，展示其与传统（ε,δ）-DP 的差异和优点。

**🔧 技术方法**

技术手段主要是信息论工具（信息密度、最大后门信息泄漏、数据处理不等式）、概率分布分析、逆函数与分位数理论、非凸优化（对二元后处理的最优分配）、以及组合性分析。

**📊 数据集**

本文并未使用真实数据集，而是在离散模型中采用了均匀或给定的先验分布（如四层先验）作为例子，用来演示 PML 封套的计算与数值特性。

**📈 对比分析**

通过将 k‑随机响应机制在不同 k 和先验下的 PML 封套与传统（ε,δ）-DP 的隐私曲线对比，展示了 PML 封套在低 δ 时给出的隐私阈值通常低于 DP，且随 δ 增大后退化速度更慢；在 PML 极端机制下证明封套恒为 ε，展示了极端机制的最优性；数值实验表明上界和下界收敛良好，且对先验和 k 的依赖性清晰可见。

**⚠️ 局限性**

局限性包括：
- 对一般机制计算 PML 封套仍然困难，需求解高维非凸优化；
- 下界仅在有限域上给出，且可能不紧（尤其是对非二元后处理的情况）；
- 只考虑了离散有限域，未覆盖连续或无穷支持的情形；
- 对组合（尤其是自适应组合）的精确界限仍未给出；
- 对实际系统的实现与计算复杂度未做深入探讨。

---

## 521. Manga109-v2026: Revisiting Manga109 Annotations for Modern Manga Understanding

**arXiv ID:** 2605.21182 | [PDF](https://arxiv.org/pdf/2605.21182v1)

**作者:** Jeonghun Baek `[一作]` (University of Tokyo), Kiyoharu Aizawa `[通讯]` (University of Tokyo)

**通讯引用:** 8893 | [OpenAlex ID](https://openalex.org/A5069982192)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对Manga109漫画数据集中的对话文本注释进行系统性修订，识别并纠正了5类注释问题，重新标注约29,000条注释，构建了Manga109-v2026版本；

**💡 创新点**

创新点在于将现代OCR结果与大型语言模型辅助的冲突检测相结合，形成混合自动-人工的纠错流程，同时处理了文字重叠、边界框过大、缺失文本、擬聲词覆盖以及连线气泡分割不当等特定漫画表达结构问题；

**🔧 技术方法**

主要技术包括：商业OCR API（Mantra Inc.）用于生成候选文本；GPT‑5与Gemini 3 Flash用于评估文字转录一致性并自动选择正确文本；人工验证与手动重新划分边界框；以及后续的OCR评估脚本；

**📊 数据集**

使用原始Manga109数据集（约147,887条文本注释），并在此基础上构建Manga109‑v2026，评估时采用MangaOCR的OCR输出；

**📈 对比分析**

通过对比原始Manga109与Manga109‑v2026的OCR评估（E2E精度、召回率与H‑mean），发现H‑mean从48.5提升至62.9，提升幅度约14.4个百分点，精度和召回率也分别提升至63.4%和62.4%；

**⚠️ 局限性**

局限性包括：OCR输出仍不被视为绝对真值，可能导致误修；人工修订工作量大且主观性存在；仅覆盖了约20%注释的修订，未覆盖所有潜在错误；依赖专有OCR与LLM，复现性受限；

---

## 522. Comparative Analysis of Military Detection Using Drone Imagery Across Multiple Visual Spectrums

**arXiv ID:** 2605.21157 | [PDF](https://arxiv.org/pdf/2605.21157v1)

**作者:** Sourov Roy Shuvo `[一作]` (KIIT Deemed to be University), Prasant Kumar Pattnaik `[通讯]` (KIIT Deemed to be University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过在四种视觉模式下训练YOLOv11-small模型，评估了无人机图像中军事目标检测的性能。

**💡 创新点**

创新点在于将单一数据集转化为灰度、热成像、夜视与遮蔽四种模拟真实战场环境的视觉模式，并对比不同模式下的检测效果。

**🔧 技术方法**

使用技术包括图像预处理（如COLORMAP INFERNO、光照增强）、YOLOv11-small目标检测、以及标准评价指标（mAP@50、mAP@50–95、F1、处理时延）。

**📊 数据集**

使用的数据集为KIIT‑MiTA（1700张无人机军事场景图），并通过自定义增强生成四个子集。

**📈 对比分析**

通过对比四个视觉模式的mAP、推理时间等指标，发现夜视模式在准确率上最高（mAP@50≈0.701），遮蔽与热视效果相近，灰度模式最节省时延但准确率最低。

**⚠️ 局限性**

局限性包括缺乏实时视频序列评估、对极端低可见度或复杂遮挡场景的鲁棒性不足，以及模型在边缘设备上的推理性能仍待优化。

---

## 523. CoarseSoundNet: Building a reliable model for ecological soundscape analysis

**arXiv ID:** 2605.21143 | [PDF](https://arxiv.org/pdf/2605.21143v1)

**作者:** Alexander Gebhard `[一作]` (TUM University Hospital), Björn W. Schuller `[通讯]` (TUM University Hospital)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文构建并公开了CoarseSoundNet，一种多标签深度学习模型，用于在PAM录音中粗粒度区分人类声、动物声和地表声，并将“静默”作为额外类别；

**💡 创新点**

创新点包括：①系统评估多种CNN/Transformer架构并选取最优模型；②在训练中加入静默类以提升特征区分度；③引入类特定阈值和持续时间约束来优化推理性能；④通过大量公开与私有PAM数据集进行跨域验证，展示了数据多样性对泛化的影响；

**🔧 技术方法**

技术上采用预训练的CNN10、AST、CLAP-HTSAST等模型，使用二元交叉熵、Adam优化器、SpecAugment/自定义增广，并在滑动窗口（10 s）上进行多标签推理；

**📊 数据集**

数据集涵盖Edansa（Arctic North Slope）、BE系列（德国三地区）、以及多来源混合数据（AudioSet、FSD50K等），共计数千小时录音；

**📈 对比分析**

与多种基线（如BirdNET、Whisper、Qwen2-Audio）对比，CoarseSoundNet在Edansa的宏观F1达0.923、在BESound的宏观F1达0.907，加入静默类和额外PAM数据后性能进一步提升，且类特定阈值+持续时间约束可使某些类别的召回率提高约5%；

**⚠️ 局限性**

主要局限包括：①域间差距仍显著，特别是人声与地表声的识别受噪声掩蔽影响；②合成混合数据效果有限；③人声类别仍易出现误检/漏检，且人工标注误差高；④生态指标与物种多样性相关性依然较弱，说明仅靠声景分类无法完全替代生态学测量。

---

## 524. LoRa and LoRaWAN simulator-cum-emulator with CAD and capture effect in Python

**arXiv ID:** 2605.21136 | [PDF](https://arxiv.org/pdf/2605.21136v1)

**作者:** Matthijs Reyers `[一作]`, R. R. Venkatesha Prasad `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文提出了一款基于Python的LoRa/LoRaWAN离散事件仿真器，支持多种设备类、捕获效应、协议无关性以及真实STM32固件的仿真。

**💡 创新点**

创新点包括：①三阶段包交付模型精准再现捕获效应；②基于容器化、CFFI的固件仿真技术，可直接运行实际C代码而不需完整硬件模拟；③面向Python的轻量级内核，降低学习门槛。

**🔧 技术方法**

使用技术主要有：Python 3（asyncio、cffi、pandas、numpy）、自研离散事件内核、跨平台交叉编译、HAL接口重定向、三阶段信号捕获模型以及完整LoRaWAN 1.0.4协议栈。

**📊 数据集**

本文并未使用公开数据集，而是通过自定义路径损耗模型、信道干扰参数以及模拟场景（多网关、设备类A/B/C）来验证仿真器功能。

**📈 对比分析**

与NS‑3、OMNeT++等传统C++仿真器相比，Python仿真器在几百节点规模下可在数秒至数分钟内完成仿真，且在功能覆盖度上（设备类、捕获效应、固件仿真）实现了前所未有的完整性。

**⚠️ 局限性**

主要局限包括：无法捕捉固件中的内存管理错误或ARM特有的未定义行为；仿真速度仍慢于C++实现；跨平台编译对极端硬件依赖的支持有限。

---

## 525. Smarter edits? Post-editing with error highlights and translation suggestions

**arXiv ID:** 2605.21135 | [PDF](https://arxiv.org/pdf/2605.21135v1)

**作者:** Fleur V. J. van Tellingen `[一作]`, Alina Karakanta `[通讯]` (Leiden University)

**通讯引用:** 212 | [OpenAlex ID](https://openalex.org/A5047533259)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究通过构建SmartPE界面，利用自动后编辑(APE)生成的错误高亮和修正建议，并与传统后编辑(PE)、基于质量估计(QE)的错误高亮进行对比，探讨这些LLM辅助特征在专业翻译后编辑中的可用性。

**💡 创新点**

创新点在于首次将APE生成的错误高亮与纠正建议引入后编辑工作流，提供更直观的错误提示和可直接应用的翻译方案，并公开了相应的后编辑数据集与工具。

**🔧 技术方法**

采用LLM技术：xTower‑Instruct‑13B‑v0.1用于MT与APE；xCOMET‑XXL用于QE；SmartPE界面采用JavaScript实现并记录细粒度编辑日志；评估方法包括ESA、DA、主观Likert量表及半结构化访谈。

**📊 数据集**

使用来自WMT24新闻共享任务和QE4PE语料库的8篇新闻和8篇生物医学短文（共200词/篇），生成MT与APE文本，形成14,400词的多平行后编辑数据集，已公开托管于GitHub。

**📈 对比分析**

通过在四种条件下测量字符/秒的生产率、直接评估(DA)分数、ESA得分以及用户感知评分进行比较，结果显示无论是生产率还是翻译质量均未出现显著提升；但S‑APE条件下的建议获得最高的用户体验和信心提升，H‑APE高亮在主观上被认为更有用。

**⚠️ 局限性**

局限性包括样本仅有8名专业译者，单一语言对（英→荷）与有限文本量，评估仅用一名专家评分，易受学习、疲劳和个体差异影响，且对低资源语言对的可推广性未知。

---

## 526. SurgOnAir: Hierarchy-Aware Real-Time Surgical Video Commentary

**arXiv ID:** 2605.21132 | [PDF](https://arxiv.org/pdf/2605.21132v1)

**作者:** Jingyi He `[一作]` (TU Munich), Yuan Bi `[通讯]` (TU Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种实时、层次化的外科手术视频叙述框架SurgOnAir，能够在无未来信息的情况下逐帧生成叙述，并精准捕捉阶段、步骤等层次变化；

**💡 创新点**

首创层次化实时流式叙述模型，结合阶段‑步骤‑动作三级标签；构建了SurgOnAir‑11k层次化视频‑语言数据集；使用专门的转移token显式标记工作流状态变迁；通过密集时序交错输入与LLM的双层叙述‑状态融合实现高效实时生成；

**🔧 技术方法**

使用 WhisperX 进行词级语音转录与时间对齐；GPT‑4o 对 ASR 纠错与动作过滤；视觉编码器提取 2 FPS 帧；以预训练的 LiveCC‑7B‑Base 作为 LLM 基座；采用分阶段条件化、KV‑cache 流式推理等训练策略；

**📊 数据集**

SurgOnAir‑11k 数据集（约 11k 段手术视频），含词级 ASR 转录、动作/解释/交互过滤、三层阶段‑步骤‑动作标签；对比基线使用 LLaVA‑Video‑7B、Qwen2.5‑VL‑7B、LiveCC‑7B；

**📈 对比分析**

通过 GPT‑4o 评判者进行成对选择任务，计算胜率；SurgOnAir 与 Hulu‑Med 的比对中胜率为 66.1%，优于 LiveCC‑7B（16.7%）和 SurgOnAir‑base（60.4%）；消融实验显示层次化建模将胜率提升至 60.6%，阶段正确率提升至 65.8%；

**⚠️ 局限性**

数据集规模有限，缺乏未来动作预测能力，对极端场景或罕见步骤的泛化能力受限；

---

## 527. VersusQ: Pairwise Margin Reasoning for Generalizable Video Quality Assessment

**arXiv ID:** 2605.21130 | [PDF](https://arxiv.org/pdf/2605.21130v1)

**作者:** Shibei Meng `[一作]` (Chinese University of Hong Kong), Qiang Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 14786 | [OpenAlex ID](https://openalex.org/A5088556682)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VersusQ框架，利用大型多模态模型进行视频质量的成对边际推理

**💡 创新点**

创新点在于使用连续边际回归和Margin‑Coupled GRPO，将相对比较与绝对评分解耦，实现无锚点最小二乘排行榜重构

**🔧 技术方法**

结合Qwen3‑VL‑4B‑Instruct、冻结的SlowFast R50运动编码器、轻量级回归头以及GRPO强化学习

**📊 数据集**

在LSVQ训练集上训练，并在七个公开VQA基准（LSVQ、LSVQ‑1080p、KoNViD‑1k、LIVE‑VQC、LIVE‑YT‑Gaming、Waterloo‑IVC‑4K、VDPVE）上评估

**📈 对比分析**

与无监督、监督和LMM强化学习方法对比，VersusQ在SRCC/PLCC上均取得最高平均分，跨域泛化表现显著优于现有SOTA

**⚠️ 局限性**

局限性包括对成对采样策略的依赖、训练效率受限以及对长视频或AIGC内容的时序建模不足

---

## 528. KSOS-BO: Improving Sampling in Bayesian Optimization via Kernel Sum of Squares

**arXiv ID:** 2605.21179 | [PDF](https://arxiv.org/pdf/2605.21179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 529. PREFINE: Preference-Based Implicit Reward and Cost Fine-Tuning for Safety Alignment

**arXiv ID:** 2605.21225 | [PDF](https://arxiv.org/pdf/2605.21225v1)

**作者:** Richa Verma `[一作]` (IIT Madras), Balaraman Ravindran `[通讯]` (IIT Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种完全离线的安全对齐框架PREFINE，能够利用有限的安全与不安全轨迹偏好对已训练好的策略进行微调，从而降低成本违约率并保持任务性能。

**💡 创新点**

核心创新在于将安全对齐转化为偏好学习问题，并结合Direct Preference Optimization (DPO) 与监督微调(SFT)形成单阶段无嵌套优化的目标；同时使用策略采样生成对抗性“假”动作作为DPO的对比样本，提升了安全信号的可靠性。

**🔧 技术方法**

采用DPO+SFT混合损失、策略采样的对比动作生成、VAE网络结构来捕捉多模态离线数据；训练时只需一次反向传播，避免了成本估计或分布匹配的复杂循环。

**📊 数据集**

使用DSRL基准的12个连续控制任务（7个SafetyGym、5个BulletSafetyGym）中的离线轨迹数据，分别划分为安全（D_p）和不安全（D_np）子集。

**📈 对比分析**

与BC、PPL、SafeDICE、CPQ等基线在奖励-成本曲线、标准化奖励/成本以及安全阈值满足率进行对比，PREFINE在保持最高奖励的同时把安全违约率降低60%–92%，且训练时间约为基线的1/10。

**⚠️ 局限性**

局限性包括：需要已有可微的参考策略；对偏好标签噪声仍有一定敏感性；对极端不平衡的数据分布或完全缺失不安全示例的适应性尚未充分验证。

---

## 530. Tao's Equational Proof Challenge Accepted (Technical Report)

**arXiv ID:** 2605.21200 | [PDF](https://arxiv.org/pdf/2605.21200v1)

**作者:** Lydia Kondylidou `[一作]` (Ludwig-Maximilians-Universität München), Marijn J. H. Heule `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 Krympa 工具，通过混合使用 Vampire 与 Twee 自动定理证明器，自动化地对等式推理证明进行最小化，将原始证明步数显著压缩。

**💡 创新点**

创新点在于：①将不同证明器的证明拆分成可独立重证的子结论（lemmas），②构造依赖图并采用分段组合策略，③使用多种问题变体（big‑step、small‑step、abstracted）与启发式选择，自动寻找更短的证明链。

**🔧 技术方法**

使用技术包括：Vampire 超位置推理、Twee 完成法、Brute‑force + 启发式搜索、直接证明转换、依赖图构造、分段证明重组、Lean 形式化输出。

**📊 数据集**

数据集为 Equational Theories Project 的 1431 个等式问题，其中包含 650→448 挑战定理，实验覆盖所有 13 个 Lean 文件中的等式证明。

**📈 对比分析**

通过在 1431 个基准上与原始 Vampire 证明比较，平均证明长度从 6.6 步缩短到 4.5 步（约 31.5%），对至少 15 步基准的 117 个案例，SA 方案平均缩短 56.7%；单个案例如 650→448 从 62 步降至 20 步，151 步降至 10 步，显示出显著的性能提升。

**⚠️ 局限性**

局限性：①仅适用于一阶单一等式（unit equality）子领域；②依赖于外部证明器的非确定性和时间限制；③分段策略仅限于 3 段，可能无法捕获更复杂结构；④未考虑项大小或复杂度；⑤目前仅集成 Vampire 与 Twee，扩展到其他证明器需要额外工作。

---

## 531. Learning First Integrals via Backward-Generated Data and Guided Reinforcement Learning

**arXiv ID:** 2605.21160 | [PDF](https://arxiv.org/pdf/2605.21160v1)

**作者:** Jingfeng Zhong `[一作]` (Shanghai Jiao Tong University), Shuai Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 133310 | [OpenAlex ID](https://openalex.org/A5100371500)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大模型的求解器FISolver，用于自动发现微分方程的一阶积分；

**💡 创新点**

创新点在于提出“反向生成”算法构造大量（微分方程、一阶积分）数据集，并通过监督微调和Levenshtein距离奖励的强化学习提升模型的符号推理能力；

**🔧 技术方法**

采用LoRA微调、GRPO强化学习、前缀符号生成、Beam搜索与符号验证等技术；

**📊 数据集**

使用自生成的BwdBase、BwdSyn数据集以及前向生成的Fwd数据集进行训练与评估；

**📈 对比分析**

与大型数学LLM（如Qwen2.5-Math-7B）和商业求解器Mathematica对比，在Normal集上达到约70%准确率，Hard集上超过60%准确率，显著优于对手；

**⚠️ 局限性**

局限包括仅在二维无参ODE系统上验证，Polish表示非规范导致Levenshtein奖励可能不完全对应；

---

## 532. SAOITHE: Sustainable Age-of-Information-Based Timely Status Updating for Hardware-constrained Edge networks

**arXiv ID:** 2605.21328 | [PDF](https://arxiv.org/pdf/2605.21328v1)

**作者:** Shih-Kai Chou `[一作]` (Ericsson), Jernej Hribar `[通讯]` (Jožef Stefan Institute)

**通讯引用:** 305 | [OpenAlex ID](https://openalex.org/A5054993601)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种面向碳排放的实时状态更新调度框架SAOITHE，旨在在碳足迹预算内最小化信息的Age of Information。

**💡 创新点**

创新点在于将碳强度的时间变化纳入受限马尔可夫决策过程，推导出闭式Whittle指数实现可扩展调度。

**🔧 技术方法**

使用的技术包括受限马尔可夫决策过程、拉格朗日松弛、Whittle指数方法以及动态规划。

**📊 数据集**

实验使用真实世界的碳强度（CI）轨迹，覆盖低、中、高CI区域。

**📈 对比分析**

与Round Robin和Random基线相比，SAOITHE在低、中、高CI环境下分别提升约25%、20%和75%的AoI性能，同时满足碳预算约束。

**⚠️ 局限性**

限制在于假设所有源能耗相同、优先级一致，未考虑源间碳强度差异及更复杂的网络拓扑。

---

## 533. Multicategorical Semantics for Untyped Effects

**arXiv ID:** 2605.21337 | [PDF](https://arxiv.org/pdf/2605.21337v1)

**作者:** Ariel Grunfeld `[一作]` (Ben-Gurion University), Liron Cohen `[通讯]` (Ben-Gurion University)

**通讯引用:** 1425 | [OpenAlex ID](https://openalex.org/A5029629129)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本论文提出了 Freyd 代数（Freyd operads）作为无类型、值与计算分离的分组结构，并构造了相应的 Freyd PROP，证明了其可表示性与与 Freyd PROP 的 adjunction，从而得到无类型计算 λ 计算式的完整语义模型；

**💡 创新点**

创新点在于将计算顺序化的单步替换视为原子操作，利用 Freyd 代数和预多范畴（preoperads）构造序列化的替换范畴，并证明该构造是 Freyd PROP 的自由构造；

**🔧 技术方法**

主要技术包括：预多范畴、Freyd 代数、预 PROP、自由替换构造、表示性与 adjunction 的证明，以及对无类型计算 λ 语言的语义解释与完整性证明；

**📊 数据集**

无；本研究为纯理论模型，未使用任何数据集；

**📈 对比分析**

无；本工作不涉及实验或性能对比，而是通过构造性证明展示模型的初始性与完整性；

**⚠️ 局限性**

局限性包括：仅针对无类型 Call‑by‑Value 计算器，未覆盖类型化系统；对线性、CBPV 等变体的适用性尚未探讨；此外，构造在更大范畴（如多对象情形）下的推广仍需进一步研究。

---

## 534. RSE of a Quantum Transport Code and its Effects

**arXiv ID:** 2605.21334 | [PDF](https://arxiv.org/pdf/2605.21334v1)

**作者:** Christoph Conrads `[一作]` (Forschungszentrum Jülich), Edoardo Di Napoli `[通讯]` (Forschungszentrum Jülich)

**通讯引用:** 486 | [OpenAlex ID](https://openalex.org/A5055932268)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对 libNEGF 量子输运模拟软件，构建了完整的研发流程，涵盖了持续集成（CI）、容器化构建、单元与系统级测试、以及持续性能基准（CB）等实践。

**💡 创新点**

创新点在于：①将安全语言与不安全语言混合开发时的“未捕获错误”思维系统化；②在 CI 中引入多容器、多平台、多 MPI 方案，保证代码在多种硬件与编译器下可构建且无错误；③利用 JUBE 与 LLview 自动记录与分析 HPC 运行结果，实现对性能退化与硬件更新影响的实时监测；④通过对编译器警告、返回值检查和代码格式化的强制执行，显著降低了隐藏缺陷的概率。

**🔧 技术方法**

使用技术包括：Fortran、C/C++、CUDA、GitLab CI、Docker/Kubernetes、JUBE（性能基准工具）、LLview（作业报告），以及 OpenMPI、MPICH、GCC、Intel Fortran 等编译器组合。

**📊 数据集**

数据集：主要使用 libNEGF 自己的量子设备模拟输入（结构、Hamiltonian、重叠矩阵），在不同节点数、CPU/GPU、MPI 组合下运行，生成的模拟结果与性能指标（耗时、能耗、GPU 活跃度）作为基准数据。

**📈 对比分析**

比较方法：在 CI 中按不同容器与编译器组合并行构建并执行完整测试套件；在 CB 中使用 JUBE 在 JUWELS Booster 上每周执行强标缩放实验，并通过 Python 脚本统计耗时、能耗与 GPU 负载。性能表现：在正常条件下实现了 8–192 节点的可扩展性；在系统维护后观察到 2–4 节点运行时间与能耗轻微下降，验证了持续监测的必要性。

**⚠️ 局限性**

局限性包括：①CI 仅覆盖有限的容器与编译器组合，仍无法覆盖所有 HPC 环境；②持续性能基准需要占用昂贵的超级计算机资源，频率受限；③对遗留代码的修复成本高，某些重构如 Git 历史清理仍需手动干预；④容器化构建虽提高可重复性，但在大型项目中仍可能因依赖版本冲突导致构建失败。

---

## 535. TextReg: Mitigating Prompt Distributional Overfitting via Regularized Text-Space Optimization

**arXiv ID:** 2605.21318 | [PDF](https://arxiv.org/pdf/2605.21318v1)

**作者:** Lucheng Fu `[一作]` (Georgia Institute of Technology), Haohan Wang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个正则化框架TextReg，旨在通过控制提示文本的容量成本与规则范围狭窄两大因素，降低提示优化过程中的分布式过拟合，从而提升大语言模型在不同数据集和不同后端模型上的跨域推理性能。

**💡 创新点**

创新点主要包括：①将提示分布式过拟合建模为代表性低效（capacity cost × scope narrowness）并以此度量；②设计了在离散文本空间实现软惩罚的三阶段正则化流程（Dual‑Evidence Gradient Purification、Semantic Edit Regularization、Regularization‑Guided Prompt Update）；③引入 RuleBank 进行全局规则回溯与本地批次证据融合，过滤掉局部、样本特定梯度；④通过有限差分估计正则化梯度，将代表性低效的变化直接映射为文本正则化指令。

**🔧 技术方法**

技术手段包括：LLM驱动的文本梯度生成与过滤、基于RuleBank的规则记忆与匹配、LLM实现的条件投影Π_gen、语义差分分析器 M_Δ、生成正则化梯度的生成器 Γ，以及在候选提示中优先选择兼容正则化梯度的 Prompt Rewrite。

**📊 数据集**

实验数据集涵盖 BigBench Hard 6 任务：Logical Deduction（3/5/7 物体）、Tracking Shuffled Objects（3/5/7 物体）、GSM8K、SVAMP、MultiArith；训练使用三项源任务，评估在剩余任务及四种不同 LLM 后端上进行跨域测试。

**📈 对比分析**

与 Zero‑shot CoT、TextGrad、REVOLVE 进行比较。实验显示 TextReg 在大多数后端模型和数据集的 OOD 任务中均取得最佳或次佳准确率，并在最难任务上提升约 10% 以上；相比 TextGrad 提升高达 +11.8%，相比 REVOLVE 提升 +16.5%。

**⚠️ 局限性**

局限性：目前仅针对单轮推理且规则结构明确的任务；对开放式生成、多轮对话及代理指令等场景，规则模糊、容量共享等问题尚未涵盖，需进一步研究。

---

## 536. Fast and Stable Triangular Inversion for Delta-Rule Linear Transformers

**arXiv ID:** 2605.21325 | [PDF](https://arxiv.org/pdf/2605.21325v1)

**作者:** Aleksandros Sobczyk `[一作]` (Huawei Technologies), Jiawei Zhuang `[通讯]` (Huawei Technologies)

**通讯引用:** 842 | [OpenAlex ID](https://openalex.org/A5103220307)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统地分析并基准测试线性注意力模型中三角矩阵求逆算法，重点关注数值稳定性、计算复杂度与硬件效率，并提出实现方案，在低精度下实现了最高 4.3 倍的速度提升，同时保持模型精度。

**💡 创新点**

首次提供直接与迭代三角求逆方法在 AI 加速器上的全面评估，揭示最佳低精度实现，证明在不牺牲准确率的前提下可显著加速 LLM 线性注意力层。

**🔧 技术方法**

采用直接求逆（VCS、MCS、MBH、MXR）与迭代求逆（NS、IR）算法，利用矩阵乘法密集实现、低精度浮点运算、NPU 硬件加速及严格的数值稳定性分析。

**📊 数据集**

使用 MMLU 端到端评估数据集以及 Qwen3.6 等大模型的真实推理结果，对三角矩阵进行合成实验，并在 NPU 上测量运行时性能。

**📈 对比分析**

与 SGLang 现有实现进行对比，测量单层三角求逆占比、整体推理速度与准确率下降；实验显示在 NPU 上可达 4.3 倍加速，同时在低精度环境保持完全的推理精度。

**⚠️ 局限性**

迭代方法在低精度下易出现不稳定，直接方法仍存在较高计算开销；对数、单位对角下三角矩阵的假设限制了泛化范围；在极大规模或非标准结构下仍需进一步验证。

---

## 537. Vision Transformers and Convolutional Neural Networks for Land Use Scene Classification

**arXiv ID:** 2605.21268 | [PDF](https://arxiv.org/pdf/2605.21268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 538. CRAFT: Conflict-Resolved Aggregation for Federated Training

**arXiv ID:** 2605.21317 | [PDF](https://arxiv.org/pdf/2605.21317v1)

**作者:** Ziqi Wang `[一作]` (Friedrich-Alexander University Erlangen-Nuremberg), Nils Thuerey `[通讯]` (Technical University of Munich)

**通讯引用:** 3035 | [OpenAlex ID](https://openalex.org/A5047248117)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 CRAFT，利用参考锚定的最小二乘投影实现冲突消除的联邦聚合框架。

**💡 创新点**

创新点在于把聚合视为几何纠正问题，使用闭式伪逆投影确保全局更新与每个客户端梯度正向对齐，同时引入层级自适应以解决不同层级的冲突。

**🔧 技术方法**

核心技术包括参考方向（上一轮归一化更新）与冲突约束的线性规划、Moore‑Penrose伪逆求解、层级投影和梯度正向约束的闭式解。

**📊 数据集**

实验使用 FEMNIST、CIFAR‑10/100 数据集，配合 MLP、CNN 及 ResNet‑20/56/110 等深度模型进行验证。

**📈 对比分析**

与 FedAvg、FedProx、FedNova、FedAvgM、FedAdam、FedMGDA+、FedFV、ConFIG 等基线对比，平均精度提升 7.8%~53.5%，尾部用户精度提升显著，标准差显著下降，收敛速度更快。

**⚠️ 局限性**

局限包括仅使用上一轮全局更新作为参考方向、对齐目标固定为样本比例、在极端客户端参与不完整或多步局部更新时可能导致约束不可行或性能下降。

---

## 539. From Circuit Evidence to Mechanistic Theory: An Inductive Logic Approach

**arXiv ID:** 2605.21303 | [PDF](https://arxiv.org/pdf/2605.21303v1)

**作者:** Nura Aljaafari `[一作]` (University of Manchester), Andre Freitas `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种正式的一致性层，将机制解释视为诱导理论构建，为每个已发现电路生成因果功能签名（CFS）和ILP学习的架构签名，支持电路在不同模型、任务和规模下的比较、细化与迁移。

**💡 创新点**

创新点在于：①首次提供了可复用的逻辑表达层，将电路行为与结构分离；②通过θ-子模（θ-subsumption）实现跨实验的形式化比较；③利用ILP从结构与因果证据中学习可解释的Horn子句，显著提升了对电路类别的辨别能力；④构建了迁移框架，证明先前的机制知识可跨规模、跨架构迁移。

**🔧 技术方法**

采用因果归因（Direct Logit Attribution）提取节点贡献；使用语言角色和任务特定角色标签构建CFS；通过Inductive Logic Programming（ILP）学习规模不变的Horn子句作为架构签名；利用θ-子模进行结构比较；实现迁移时结合行为与因果验证。

**📊 数据集**

实验基于Pythia-14M、Pythia-1B和LLaMA-3.2-1B三种Transformer模型；任务涵盖10类（语义角色绑定、间接宾语识别、数值比较等），最初使用15个电路，后扩展至每个模型30个电路；数据来源为公开的模型权重与相应任务评测数据集。

**📈 对比分析**

比较方法：将ILP签名与 Weisfeiler–Lehman 图核和随机森林特征基线进行对比；ILP在区分 IOI 与其他电路时的结构分离度提升 3–4 倍；θ-子模构造的层次结构揭示电路之间的精细关系；迁移实验中，绑定机制在跨模型、跨架构迁移时成功率高于选择/比较机制。

**⚠️ 局限性**

局限性：仅在少数模型与任务上验证，未覆盖更广泛的架构；验证仅针对因果贡献，未评估对新提示的任务性能；阈值与参数多为经验设定，需进一步敏感性分析；CFS 的角色标签依赖轻量级依赖句法解析，可能忽略分布式机制。

---

## 540. DriveMA: Rethinking Language Interfaces in Driving VLAs with One-Step Meta-Actions

**arXiv ID:** 2605.21273 | [PDF](https://arxiv.org/pdf/2605.21273v1)

**作者:** Weicheng Zheng `[一作]` (Shanghai Qi Zhi Institute), Hang zhao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在端到端驾驶系统中，提出并验证了一种简洁的单步meta-action作为语言中介，取代长链自然语言推理；

**💡 创新点**

创新点包括：①将高维轨迹压缩为可自动标注、低熵的meta-action，降低标注成本和推理延迟；②采用行动中心的监督预训练提升meta-action预测；③设计基于turn‑level credit assignment的RL框架，结合轨迹与meta-action一致性奖励，显著弥合语言与动作间的差距；

**🔧 技术方法**

技术手段包括：基于Qwen3.5的视觉‑语言模型，轨迹‑meta-action映射规则、块划分法生成标注；行动中心SFT预训练；多轮生成中的GRPO强化学习，稠密奖励（meta-action、轨迹质量、轨迹‑meta-action一致性）以及按步骤归一化的优势分数；

**📊 数据集**

使用的数据集：Waymo Open Dataset Vision-based End-to-End Driving (WOD‑E2E)、NAVSIM、以及240K的驾驶VQA数据（WaymoQA、IDKB、LingoQA）做预训练；

**📈 对比分析**

在WOD‑E2E上，DriveMA 2B模型获得RFS Overall 8.060、Spotlight 7.251，4B模型提升至8.079/7.169，均超越所有先前方法；在NAVSIM上，4B模型实现PDMS 91.2，优于现有VLA方法；相比结构化或开放式自然语言推理，单步meta-action在预测准确率、推理延迟和整体性能上均表现更佳；

**⚠️ 局限性**

局限性包括：①单步meta-action的表达能力有限，细粒度决策仍难以捕捉；②meta-action生成仍受规则映射约束，可能在多样化场景下不够鲁棒；③强化学习阶段对样本效率和超参敏感；④模型主要在Waymo与NAVSIM上验证，跨域推广仍需进一步研究。

---

## 541. Reinforcement Learning for Risk Adaptation via Differentiable CVaR Barrier Functions

**arXiv ID:** 2605.21257 | [PDF](https://arxiv.org/pdf/2605.21257v1)

**作者:** Xinyi Wang `[一作]` (University of Michigan), Dimitra Panagou `[通讯]` (University of Michigan)

**通讯引用:** 2427 | [OpenAlex ID](https://openalex.org/A5059647993)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出一种端到端的风险自适应框架，利用强化学习与可微分的CVaR安全层实现对随机障碍运动下的安全导航；

**💡 创新点**

创新点在于首次将可微分的CVaR控制障碍函数(QP)与RL联合训练，使机器人能在动态人群中自适应调整风险水平与安全裕度，从而兼顾效率与安全；

**🔧 技术方法**

主要技术包括Gaussian混合模型（GMM）对障碍运动不确定性建模、基于CVaR的控制障碍函数、可微分的二次规划安全层以及Actor‑Critic强化学习；

**📊 数据集**

实验使用基于社交力模型的仿真数据，包含12×12工作空间、20个动态圆形障碍，分别测试单积分器与圆盘式差速车两种机器人模型；

**📈 对比分析**

与ORCA、CBF‑QP、固定风险CVaR‑BF、Adaptive‑CVaR‑BF、Vanilla‑RL、CrowdNav++等基线对比，实验显示所提方法在成功率、返回值和安全距离方面均显著优于其他方法，尤其在高密度人群场景下保持较高的稳健性；

**⚠️ 局限性**

主要局限包括仅考虑最近一个可见障碍物（N=1），未在真实机器人平台验证，且依赖GMM预测的障碍运动，未来工作需扩展到多障碍联合建模、图神经网络编码以及多步预测安全约束。

---

## 542. Learning Structural Latent Points for Efficient Visual Representations in Robotic Manipulation

**arXiv ID:** 2605.21258 | [PDF](https://arxiv.org/pdf/2605.21258v1)

**作者:** Yicheng Jiang `[一作]` (Hong Kong University of Science and Technology), Qiming Shao `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 4511 | [OpenAlex ID](https://openalex.org/A5027396836)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种融合隐式与显式表征的3D预训练框架，使用点云变换器与点级变分自编码器学习结构化潜在点；

**💡 创新点**

创新点在于将点级变分自编码器嵌入点云编码器的潜在空间，形成结构化潜在点；同时采用轻量化3D Gaussian splatting、仅特征分裂渲染以及辅助点/颜色重建，提升表示质量与训练效率；

**🔧 技术方法**

技术包括Point Transformer v3、点级变分自编码器(PL‑VAE)、3D Gaussian splatting、特征分裂渲染、深度与语义对齐、强化学习的ACT策略；

**📊 数据集**

使用ScanNetV2进行预训练，评估数据集为RLBench、ManiSkill2以及六项真实机器人抓取/放置/倒水等任务；

**📈 对比分析**

与多种基线（MultiViT、PTv3、SPUNet、VC‑1、MultiMAE、PonderV2、GSRL、SPA、Lift3D）相比，模型在RLBench和ManiSkill2上实现最高的平均成功率（分别为0.56和0.64），在真实机器人实验中也获得最高成功率；同时GPU训练时间比VC‑1、SPA、PonderV2缩减4.6×、24×、60×；

**⚠️ 局限性**

由于引入采样过程，模型在极高精度需求场景（如手术、细工操作）下的几何表征可能不够精准，表现受限。

---

## 543. Beyond the Tip of the Iceberg: Understanding SATD in Dockerfiles through the Lens of Co-evolution

**arXiv ID:** 2605.21238 | [PDF](https://arxiv.org/pdf/2605.21238v1)

**作者:** Wei Minn `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 31419 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过将Dockerfile中的自我声明技术债务（SATD）与源代码的共进化视角相结合，对Dockerfile中的技术债务生命周期进行了系统性量化与质性分析。

**💡 创新点**

创新点在于首次引入跨文件共进化维度，揭示约27%入库、40%还清的技术债务与非Dockerfile文件同步演化，并构建了完整的SATD-源代码共进化分类体系。

**🔧 技术方法**

采用关键词提取与人工标注相结合的SATD识别与子类型分类，利用二元相关系数与Fisher检验衡量共进化关联，使用Kaplan‑Meier生存分析和log‑rank检验比较债务还清时长，并通过开放与轴向编码完成质性共进化模式分类。

**📊 数据集**

使用从Docker Hub按三字符前缀抽样获得151,295个项目，筛选出4,300个可连通GitHub仓库，进一步选取2,962个含Dockerfile、393个含SATD候选项的仓库，最终手工标注1,316个SATD实例。

**📈 对比分析**

通过对比coupled（共进化）与isolated（单文件）债务的生存曲线，发现共进化债务的平均还清时间更短（约39天 vs 41天，p=0.0201），尤其是Defect/Workaround子类型；统计显著性较高，说明跨文件协同更利于快速修复。

**⚠️ 局限性**

局限性包括：1）关键词提取可能漏检非关键词型SATD；2）数据集仅覆盖具星标的公开GitHub项目，难以推广至闭源或其他IaC技术；3）部分子类型样本量不足，导致生存分析统计功效低；4）人工标注虽具有高一致性，但仍可能受主观因素影响。

---

## 544. LamPO: A Lambda Style Policy Optimization for Reasoning Language Models

**arXiv ID:** 2605.21235 | [PDF](https://arxiv.org/pdf/2605.21235v1)

**作者:** Zhe Yuan `[一作]` (Pinterest), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 7202 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了LamPO，一种无价值模型的强化学习优化方法，利用组内响应的成对优势来改进奖励分配。

**💡 创新点**

核心创新在于用对比式分解优势（Pairwise Decomposed Advantage）取代传统标量组优势，并加入基于序列对数概率差的置信度加权，同时在有参考答案时提供ROUGE‑L辅助奖励。

**🔧 技术方法**

采用Lambda‑Style Policy Optimization、对数概率对比权重、PDA、ROUGE‑L辅助奖励、PPO剪裁式目标以及KL正则化等技术。

**📊 数据集**

在四大推理基准上实验，使用Mixture‑of‑Thoughts数据集，并评估AIME24、AIME25、MATH‑500、GPQA‑Diamond，测试模型包括Qwen3‑1.7B、Qwen3‑4B和Phi‑4‑mini。

**📈 对比分析**

与GRPO、DAPO、GSPO等基线相比，LamPO在所有模型和任务上均取得更高平均分（如Qwen3‑1.7B平均53.60%对51.42%，Qwen3‑4B平均77.09%对75.40%），且训练更平稳。

**⚠️ 局限性**

主要局限包括O(G²)的成对计算开销、对奖励信号质量敏感、ROUGE‑L辅助奖励仅适用于有参考答案且仅反映词汇重叠，且在非推理任务上的通用性尚未验证。

---

## 545. The Team Order Problem: Maximizing the Probability of Matching Being Large Enough

**arXiv ID:** 2605.21234 | [PDF](https://arxiv.org/pdf/2605.21234v1)

**作者:** Haris Aziz `[一作]` (UNSW Sydney), Ali Pourmiri `[通讯]` (UNSW Sydney)

**通讯引用:** 113 | [OpenAlex ID](https://openalex.org/A5028516396)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文定义并研究了“团队序列匹配问题”（Team Order Problem），即给定一队已确定的选手排列，寻找另一队的最优排列以最大化赢得大于半数比赛的概率。针对该问题提出了多种结果：在概率仅取三值 {α,β,0} 时可在多项式时间内求解；给出了一个多项式时间近似方案（PTAS），在概率被 ϵ>0 缩放后能够得到接近最优的解；并对最大权匹配与最优解之间的差距给出了理论界定。

**💡 创新点**

创新点包括：1) 将团队赛排布问题与 Poisson-Binomial 分布与正常分布逼近相结合，构造了一个可控误差的近似方案；2) 将该问题转化为预算/奖励匹配（Budgeted/Reward Matching），利用已有 PTAS 做为子算法；3) 在仅有三值概率的特殊情形下证明可多项式求解，弥补了该类彩色二分图匹配的空白；4) 给出了最大权匹配的胜率上界与下界，阐明了两者之间的关系。

**🔧 技术方法**

主要技术：动态规划（用于计算给定排列的胜率）；Poisson-Binomial 分布的正常近似与 Hoeffding 型集中不等式；预算匹配的 PTAS（基于分块/分配策略的近似）；二分图最大匹配（匈牙利算法）与其变体；概率分布的分层划分与匹配族的多项式上界证明。

**📊 数据集**

本文未使用实际数据集，全部为理论分析与算法设计；若需要实验验证，可基于体育比赛或推荐系统的合成数据进行测试。

**📈 对比分析**

比较方式主要是理论分析：给出与最优解的误差界限（ε 级近似）以及最大权匹配与最优解的概率差距估计；在特殊三值概率情形下证明最优性，展示算法多项式时间复杂度。由于没有实验，性能评价以时间复杂度（如 n^{O(δ^{-1}ε^{-2})}）和近似误差为主。

**⚠️ 局限性**

局限性：1) 对最优解的精确求解仍是未决问题，可能为 NP-难；2) 只给出 PTAS，未能证明是否存在 FPTAS；3) 近似方案要求所有非 0/1 概率至少远离 0 与 1（即 δ>0），在极端概率场景下可能不适用；4) 论文未给出实验验证，实际性能与理论预测之间可能存在差距。

---

## 546. Learning Robust Dexterous In-Hand Manipulation from Joint Sensors with Proprioceptive Transformer

**arXiv ID:** 2605.21330 | [PDF](https://arxiv.org/pdf/2605.21330v1)

**作者:** Senlan Yao `[一作]` (ETH Zürich), Robert K. Katzschmann `[通讯]` (ETH Zürich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种只利用关节自我感知（joint proprioception）实现软体机器人手臂对立方体进行连续旋转的控制方法；

**💡 创新点**

创新点在于：①引入“Proprioceptive Transformer（PT）”架构，通过自注意力有效从关节历史序列中提取隐式物体状态；②使用教师-学生蒸馏框架，将在仿真中获取的物体信息隐式编码到仅依赖关节传感的学生网络；③证明直接关节传感（磁角传感器）比电机编码器更能降低仿真‑现实差距；

**🔧 技术方法**

技术包括：强化学习（PPO）训练教师策略、Transformer编码器与自监督重构目标、教师-学生蒸馏、关节角度噪声建模、仿真–现实域随机化、使用 Isaac Lab 并行训练、在真实 ORCA 手臂上部署 20Hz 控制。

**📊 数据集**

数据集：①仿真环境 8192 并行，包含 55mm 与 65mm 立方体，随机化摩擦、质量、关节刚度等；②真实实验使用 ORCA 手臂与两种尺寸立方体进行 3 次 60s 试验。

**📈 对比分析**

比较方法：与 Proprio‑PPO（仅关节）和 Extero‑PPO（带真实物体位姿）对比；在 55mm 立方体上 PT‑Joint 的 RPM 达 11.83，远超 Proprio‑PPO 3.83（3.1×）和 Extero‑PPO 3.08（3.8×），且 100% 旋转准确率和 0 次掉落；在 65mm 立方体上同样实现 11.33 RPM，2.3× 超越 Extero‑PPO。

**⚠️ 局限性**

局限性：仅验证单轴旋转任务；依赖磁角传感器，未探究不同关节配置或更复杂物体形状；对动态环境变化（光照、遮挡）不涉及；未来需扩展到多轴重定向、不同形状及结合触觉信息。

---

## 547. A New Framework to Analyse the Distributional Robustness of Deep Neural Networks

**arXiv ID:** 2605.21313 | [PDF](https://arxiv.org/pdf/2605.21313v1)

**作者:** Divij Khaitan `[一作]` (Microsoft), Subhashis Banerjee `[通讯]` (Ashoka University)

**通讯引用:** 8141 | [OpenAlex ID](https://openalex.org/A5004760940)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于“显著路径”显著性矩阵的神经网络鲁棒性诊断框架，利用权重-激活交互的伯努利分布来衡量类别间分离度和熵，进而评估模型的记忆化与对分布漂移的鲁棒性。

**💡 创新点**

创新点在于：①将每个连接的权重与对应激活的乘积视作单独的交互，并构造显著性指示矩阵；②用伯努利分布的KL散度作为跨类别分离度度量；③在多任务（分类、OOD、分布偏移）上统一使用该诊断，揭示稀疏显著路径与鲁棒性之间的关系。

**🔧 技术方法**

技术手段包括：层级权重-激活交互矩阵构造、伯努利分布建模、KL散度计算、熵评估、稀疏性直方图、以及对比实验（随机标签、正常标签、OOD 数据）。

**📊 数据集**

数据集包括 ImageNet、ImageNet‑R、CIFAR‑10（及其变体 CIFAR‑10.1、SVHN、TinyImageNet）以及 InceptionV3、ResNet‑50、ViT‑B/32 等模型的公开预训练权重。

**📈 对比分析**

与其他度量（平均交互、softmax 交互、能量距离）对比，显著路径的KL散度在训练正常标签时始终表现出显著的跨类别分离度提升；随机标签模型保持低分离度，符合记忆化假设；在 OOD 条件下分离度下降、熵升高，说明诊断能捕捉到鲁棒性衰减。

**⚠️ 局限性**

局限性包括：仅针对全连接层的分析，未扩展到卷积或注意力层；伯努利假设可能不适用于所有网络结构；诊断只提供模型层级的可解释性，未直接提供改进鲁棒性的训练策略。

---

## 548. Frontier: Towards Comprehensive and Accurate LLM Inference Simulation

**arXiv ID:** 2605.21312 | [PDF](https://arxiv.org/pdf/2605.21312v1)

**作者:** Yicheng Feng `[一作]` (Chinese University of Hong Kong), Hong Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 20492 | [OpenAlex ID](https://openalex.org/A5034735808)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并实现了LLMSim，一个离散事件仿真器，用于精确模拟现代LLM推理服务中的分离式架构、并行策略与状态化工作负载。

**💡 创新点**

创新点在于提出了分离式抽象与忠实度平面，结合校准的算子预测器、通信与内存预算模型，以及对CUDA Graph、投机解码等运行时优化的细粒度建模。

**🔧 技术方法**

使用技术包括离散事件仿真框架、基于随机森林的算子运行时回归、ASTRA‑Sim/HTSim通信后端、事件驱动的调度与批处理循环以及对KV缓存、前缀缓存和激活传输的显式建模。

**📊 数据集**

实验所用数据集包括 Qwen3‑30B MoE、Step3‑316B MoE、Llama‑3.1‑8B 以及从 SharedGPT 采集的真实请求轨迹，配合人工合成的预填/解码负载。

**📈 对比分析**

与 AIConfigurator、Vidur、LLMServingSim2.0 与 Apex 的对比实验表明，在16卡 H800 测试平台上，LLMSim 的吞吐量误差低于4%，端到端延迟误差从 44.9% 降至 6.4%（共定位）或 2.6%（拆分），在大规模（1K+ GPU）下仍保持 6–10% 的精度。

**⚠️ 局限性**

局限性包括目前仅针对 vLLM 进行校准，缺乏对其他主流推理框架的完整支持；CPU 开销建模相对粗糙，需进一步改进；以及对某些深度优化（如 MoE 路由细粒度）仍存在精度提升空间。

---

## 549. DeCoR: Design and Control Co-Optimization for Urban Streets Using Reinforcement Learning

**arXiv ID:** 2605.21311 | [PDF](https://arxiv.org/pdf/2605.21311v1)

**作者:** Bibek Poudel `[一作]` (University of Tennessee), Weizi Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种两阶段强化学习框架（DeCoR），在实际城市走廊中利用视频与Wi‑Fi捕获的行人与车辆需求，对中段人行横道布局与信号控制进行共同优化，显著提升行人到达速度和车辆等待时间。

**💡 创新点**

创新点在于：1）将感知数据直接驱动城市设计与运营的闭环联合优化；2）使用生成式设计策略（高斯混合模型）采样横道位置和宽度；3）采用共享控制策略在多重模拟环境中学习自适应信号时序，兼顾行人和车辆延迟；4）通过指数惩罚的MWAQ奖励实现对极端延迟的抑制。

**🔧 技术方法**

技术包括图注意力网络（GATv2）编码行人网络、PPO强化学习用于设计与控制策略、SUMO微观仿真进行评估、GMM生成横道方案、指数MWAQ奖励函数。

**📊 数据集**

数据集来自一条750米的大学校园走廊：2,223名行人/小时、202辆车/小时，行人流通过30天匿名Wi‑Fi日志、车辆流通过交叉口视频。

**📈 对比分析**

与现有布局（7条横道）和固定周期信号、无信号控制进行对比。优化后横道数降至4条，行人到达时间平均下降23%；控制策略使行人等待时间降至1.26s（比固定信号下降79%），车辆等待时间降至9.85s（比固定信号下降65%）。同时对未见的需求量级和结构变化表现出良好泛化。

**⚠️ 局限性**

局限性包括：1）仅在单一走廊场景评估，缺乏多网络验证；2）未考虑冲突或违反信号的现实情况；3）假设感知观测完美，未考虑噪声和遮挡；4）设计目标仅关注移动性，未直接评估安全风险或伤亡。

---

## 550. How Much Online RL is Enough? Informative Rollouts for Offline Preference Optimization in RLVR

**arXiv ID:** 2605.21266 | [PDF](https://arxiv.org/pdf/2605.21266v1)

**作者:** Richa Verma `[一作]` (TCS Research), Balaraman Ravindran `[通讯]` (IIT Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了G2D框架，将短期GRPO热身与离线DPO结合，实现基于可验证奖励的推理模型训练。

**💡 创新点**

发现并利用热身阶段产生的高信息量偏好数据，证明中等热身步长能显著提升DPO性能，缩小在线与离线RLVR差距。

**🔧 技术方法**

结合GRPO（在线强化学习）、DPO（离线偏好优化）以及LoRA参数微调、可验证奖励（Math-Verify）和熵/中段比例等指标评估。

**📊 数据集**

主要使用MATH-500和GSM8K进行推理任务评测，并在Qwen2.5-7B和Llama-3.1-8B两大模型上验证。

**📈 对比分析**

对比SFT、标准离线DPO、完整GRPO（G=2/4），发现G2D在K≈150（Qwen）或K≈500（Llama）时，在MATH-500上提升约10% Pass@1，同时计算成本低约4-5倍。

**⚠️ 局限性**

结果受热身长度、模型格式符合度、生成长度约束等因素影响，且在更大规模或不同验证器下尚未验证，缺乏通用性。

---

## 551. FedCoE: Bridging Generalization and Personalization via Federated Coordinated Dual-level MoEs

**arXiv ID:** 2605.21264 | [PDF](https://arxiv.org/pdf/2605.21264v1)

**作者:** Penglin Dai `[一作]` (Southwest Jiaotong University), Xiao Wu `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 7562 | [OpenAlex ID](https://openalex.org/A5101981128)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedCoE 框架，通过共享门控网络与一致性驱动的专家聚合，兼顾全局泛化与本地个性化。

**💡 创新点**

创新点：双层 MoE 结构、共享门控网络统一路由、基于客户端‑专家相关矩阵的选择性聚合、零样本冷启动专家组装。

**🔧 技术方法**

技术：联邦学习、Mixture‑of‑Experts、共享门控网络、相关性矩阵聚合、预训练门控、零样本自适应冷启动、ResNet‑MoE backbone。

**📊 数据集**

数据集：CIFAR‑10、CIFAR‑100、Tiny‑ImageNet。

**📈 对比分析**

对比方法：FedAvg、FedProx、SCAFFOLD、FedMoEKD、pFedMoE、PM‑MoE。性能：FedCoE 在全局准确率 78% 以上、个性化 89% 以上，冷启动 77%+，相较基线提升 8‑12%。

**⚠️ 局限性**

局限：需服务器端存储全局专家池，门控网络可能泄露语义信息；极端非 IID 下专家选择仍具挑战；需要预训练公共数据集；实现复杂度较高。

---

## 552. Distributed Stochastic Graph Algorithms

**arXiv ID:** 2605.21248 | [PDF](https://arxiv.org/pdf/2605.21248v1)

**作者:** Keren Censor-Hillel `[一作]` (Technion), George Giakkoupis `[通讯]` (Inria)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究分布式随机图优化模型，提出针对最大匹配、最小顶点覆盖、最小支配集的快速近似算法。

**💡 创新点**

在分布式随机设置下实现了常数轮或无通信的近似，突破传统分布式下的对数轮下界。

**🔧 技术方法**

结合概率预处理、Poisson 变量、分水岭/水分配等技术，并利用随机化“幻觉图”与水分配算法。

**📊 数据集**

论文未使用具体数据集，理论分析基于随机边出现概率模型。

**📈 对比分析**

与经典分布式算法相比，在相同近似比下轮数从Ω(logΔ/ loglogΔ)降至常数或O(1/ε)，实现显著速度提升。

**⚠️ 局限性**

对不均匀边概率的处理有限，且仅在随机模型下证明，现实网络的鲁棒性待进一步验证。

---

## 553. Graph Navier Stokes Networks

**arXiv ID:** 2605.21247 | [PDF](https://arxiv.org/pdf/2605.21247v1)

**作者:** Zexing Zhao `[一作]` (Northwest A&F University), Yuxiao Li `[通讯]` (Bosch)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Graph Navier–Stokes Networks (GNSN)，通过在图中引入速度场实现对流与扩散双重信息传递，以缓解传统GNN的过平滑问题。

**💡 创新点**

创新点在于将Navier–Stokes方程中的对流机制迁移到图结构上，动态平衡对流与扩散并结合超平面位置编码，显著提升对异质图和同质图的适应性。

**🔧 技术方法**

采用连续时间PDE建模、可学习的速度场、对流-扩散耦合方程、神经ODE框架以及超平面位置编码等技术实现新型信息传递。

**📊 数据集**

在十二个真实世界数据集（Texas、Wisconsin、Cornell、Chameleon、Citeseer、PubMed、Cora、Amazon Photo等）以及合成cSBM数据集上进行实验。

**📈 对比分析**

与30余种基线（包括Geom-GCN、GCNII、CGNN、GRAND、GREAD等）进行对比，GNSN在大多数任务上取得最高准确率（如Photo 95.43%），并在Dirichlet能量实验中明显降低过平滑。

**⚠️ 局限性**

局限性包括对速度场与位置编码的超参数需要调优，计算开销较大，对极大规模或动态图的可扩展性尚未充分验证，且在极深网络下仍可能出现残留的过平滑现象。

---

## 554. Profiling User Vulnerability to Phishing Through Psychological and Behavioral Factors

**arXiv ID:** 2605.21246 | [PDF](https://arxiv.org/pdf/2605.21246v1)

**作者:** Valeria Formisano `[一作]` (University of Naples Federico II), Davide Marocco `[通讯]` (University of Naples Federico II)

**通讯引用:** 1746 | [OpenAlex ID](https://openalex.org/A5027259197)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了用户在网络钓鱼攻击中的易受性，利用心理学和行为学变量对1086名参与者的反应进行测评。

**💡 创新点**

创新点在于将心理学维度与行为时序（如响应时间）结合，发现“创造力”与“资历”二因素能够有效区分高危与警觉用户，并提出针对性培训策略。

**🔧 技术方法**

采用探索性因子分析（EFA）、K‑Means聚类、相关分析以及反应时间的统计检验，使用Python与Jamovi实现数据处理与建模。

**📊 数据集**

使用公开的Spamley数据集，包含用户问卷、钓鱼识别任务结果和邮件特征，共1086名有效样本。

**📈 对比分析**

通过将用户分为高危组和警觉组，比较了两组的识别率和平均响应时间；高危组识别率约68.8%，平均响应时间21.4秒；警觉组识别率约78.3%，平均响应时间23.6秒，显示两组差异显著。

**⚠️ 局限性**

局限性包括自报量表可能存在偏差、实验环境的模拟性不完全贴近真实工作情境、缺乏地理文化差异控制以及横断面设计无法确认因果关系。

---

## 555. To Select or not to Select, that is the Question: Distilling Robot Skill Prediction into a Small Ensemble

**arXiv ID:** 2605.21242 | [PDF](https://arxiv.org/pdf/2605.21242v1)

**作者:** Haechan Mark Bong `[一作]` (Polytechnique Montréal), Giovanni Beltrame `[通讯]` (Polytechnique Montréal)

**通讯引用:** 2431 | [OpenAlex ID](https://openalex.org/A5021953451)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建合成任务-技能匹配数据集，并训练小型句子编码器集成实现从自然语言任务描述预测机器人物理技能

**💡 创新点**

通过合成数据和针对边界任务生成，利用小模型集成在固定技能分类下超越大规模LLM，提出轻量化高精度技能预测方法

**🔧 技术方法**

使用LLM生成任务文本、mpnet与MiniLM句子编码器、微调Transformer后置四层MLP进行多标签分类

**📊 数据集**

合成1261条任务描述，涵盖23种技能组合，包含100条人工审核样本，构成多标签机器人技能预测数据集

**📈 对比分析**

与Kimi K2、GPT‑OSS‑120B、Llama‑4‑Scout‑17B等零样本LLM基线对比，EM达83.5%（比最大基线提升11.5%），推理速度快两位数倍

**⚠️ 局限性**

数据规模有限，标签需人工审核；仅覆盖六项物理技能，未涵盖更细粒度能力或资源约束等真实部署情况

---

## 556. RePCM: Region-Specific and Phenotype-Adaptive Bi-Ventricular Cardiac Motion Synthesis

**arXiv ID:** 2605.21237 | [PDF](https://arxiv.org/pdf/2605.21237v1)

**作者:** Xuan Yang `[一作]` (National University of Singapore), Lei Li `[通讯]` (National University of Singapore)

**通讯引用:** 12299 | [OpenAlex ID](https://openalex.org/A5100440407)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `4de8e9d8-757b-475f-9627-18a445e50202` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于区域和表型自适应的双心室运动合成框架 RePCM，能够仅用收缩末期单帧网格预测完整心动周期；

**💡 创新点**

创新点包括①利用运动描述符聚类得到数据驱动的功能区划分并生成区域邻接先验；②在条件 VAE 中引入区域特定注入模块，限制跨区域信息交流；③通过形状条件的混合专家（MoE）先验捕捉不同心脏表型的运动差异；

**🔧 技术方法**

核心技术为三维网格 VAE、注意力机制（Masked SyncAttention、FiLM）、聚类（KMeans）、混合专家模型、形状嵌入与自适应专家权重；

**📊 数据集**

使用三个公开心血管 MRI 数据集：ACDC、M&Ms 与 M&Ms-2，涵盖正常、扩张性心肌病、肥厚性心肌病及右心室异常四种表型；

**📈 对比分析**

与 CVAE、ACTOR、Action2Motion、CHeart、MeshHeart、CardioSynth4D 等基准方法对比，RePCM 在 ASSD、HD95 与顶点 RMSE 上均取得显著优势，尤其在左心室上的误差降幅最大，且在体积曲线和区域动力学保持方面表现更为稳健；

**⚠️ 局限性**

局限性包括：对右心室的预测仍受限于解剖复杂性；区域划分若过细会削弱区域特异性信号；混合专家数量与样本平衡不佳时可能导致专家稀疏与不稳定；未来需扩展到更广泛的表型和全心结构。

---

## 557. A Two-Watched Literal Scheme for First-Order Logic

**arXiv ID:** 2605.21335 | [PDF](https://arxiv.org/pdf/2605.21335v1)

**作者:** Yasmine Briefs `[一作]` (Max Planck Institute for Informatics), Christoph Weidenbach `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 2028 | [OpenAlex ID](https://openalex.org/A5046034681)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了把两盲目文字（two-watched literal）方案从命题逻辑扩展到一阶逻辑，并给出了完整的规则体系与实现细节；

**💡 创新点**

创新点在于：①将两盲目文字概念正式移植到一阶逻辑中；②设计了可递归、增量的规则体系，证明了其安全性和完备性；③通过动态编程基线对比，展示了在大规模长句子问题上的显著性能提升；

**🔧 技术方法**

使用了基于规则的推理体系、最小化归一化替换、辨别树（discrimination tree）和路径索引（path index）等数据结构；对比中还实现了基于动态规划的基线算法；

**📊 数据集**

在TPTP v9.2.1（非等价）问题集上进行实验；

**📈 对比分析**

通过在SPASS(SCL)中并行运行两种方案（同一输入、同一时间限制），记录每种方案在总耗时中的比例以及考虑的实例数；结果表明，在长句子或长子句占主导的实例中，TWFO往往比基线快一个数量级，而在极易问题上基线略占优；

**⚠️ 局限性**

局限性包括：仅在基于地面轨迹（ground trail）的设置下测试；对大量短句子或具有多重因子（factor）的子句性能不佳；实现中存在初始化开销；未在完整的一阶推理（如超分辨率）环境中验证。

---

## 558. SymbolicLight V1: Spike-Gated Dual-Path Language Modeling with High Activation Sparsity and Sub-Billion-Scale Pre-Training Evidence

**arXiv ID:** 2605.21333 | [PDF](https://arxiv.org/pdf/2605.21333v1)

**作者:** Ting Liu `[一作]` `[通讯]` (SymbolicLight Research), Ting Liu (SymbolicLight Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SymbolicLight V1，一种结合二值LIF脉冲和连续残差流的双路径稀疏语言模型；

**💡 创新点**

创新点在于将脉冲门控的长期指数衰减路径与短期稀疏注意力路径融合，并通过动态先验头和双语分词器提升稀疏性与表达力；

**🔧 技术方法**

使用LIF脉冲动力学、Dual-Path SparseTCAM、动态先验网络、ATan梯度近似以及双语BPE分词器；

**📊 数据集**

在3B词、10域中英混合语料（约40%中文、40%英文、20%其他）上训练；

**📈 对比分析**

与同参数规模GPT‑2基线对比，194M模型在验证集PPL为8.88–8.93（σ=0.021），高于GPT‑2 201M的8.27（差距7.7%），但优于GPT‑2 124M（统计显著）；Ablation显示短程注意力路径最关键，LIF动态优于静态Top‑K掩码；0.8B规模实验提供规模上升可行性；

**⚠️ 局限性**

局限包括：0.8B模型缺乏匹配的密集基线和完整基准，未进行后训练对齐，原始数据未公开，GPU上能耗仍高，稀疏化优势尚未在实际硬件上验证，且下游零样本性能与大模型相距甚远。

---

## 559. A Mechanistic Study of Tabular Foundation Models

**arXiv ID:** 2605.21288 | [PDF](https://arxiv.org/pdf/2605.21288v1)

**作者:** Marin Biloš `[一作]` (Morgan Stanley), Yuriy Nevmyvaka `[通讯]` (Morgan Stanley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三大表格基础模型（TabPFN、TabICL、Mitra）进行机制审计，定位其内部表示层、读出方式、对称性（行/列/类不变性）以及表示崩塌问题，并基于这些发现设计针对性对抗攻击。

**💡 创新点**

提出可验证的读出规则（late-layer注意力加权投票、最终层最近原型读取）并证明不同模型实现完全不同的推理机制；通过单行/列置零、去除RoPE等轻量级编辑实现完全列/类不变性；揭示模型对特定扰动（hub毒化、秩变形等）的脆弱性，展示读出机制对攻击结果的可解释性。

**🔧 技术方法**

使用层级线性探测、干预实验、注意力权重重构、kNN/线性头比较、秩/维度分析、置零/去除位置编码、重训练、对抗扰动（Hub Poison、Rank Warp 等）等技术。

**📊 数据集**

在49个公开分类基准和10个回归基准上进行评估，并使用 Balance-Scale、合成压力数据等专门设计的数据集验证表示崩塌和对称性问题。

**📈 对比分析**

在相同随机种子、相同批次下直接比较模型准确率、读出规则与原模型的性能差距，以及对抗攻击对模型和基线（Ridge、XGBoost、MLP）的准确率下降；结果显示，模型在准确率上相当，但机制差异明显，攻击效果符合读出机制的预测。

**⚠️ 局限性**

仅评估了现有预训练混合，未探究生成器对机制的影响；未覆盖大规模数据场景，且对抗攻击仅限于设计的扰动，可能未覆盖全部潜在弱点。

---

## 560. Let EEG Models Learn EEG

**arXiv ID:** 2605.21280 | [PDF](https://arxiv.org/pdf/2605.21280v1)

**作者:** Yifan Wang `[一作]` (Stony Brook University), Chenyu You `[通讯]` (Stony Brook University)

**通讯引用:** 5001 | [OpenAlex ID](https://openalex.org/A5076320750)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于流匹配的连续动态生成框架Just EEG Transformer (JET)，能够以原始多通道EEG信号为输入直接生成高保真时间序列；

**💡 创新点**

创新点在于：①将EEG生成视为连续时间演化而非离散去噪；②结合Transformer自注意力捕捉跨时空依赖；③设计了针对EEG谱学、非平稳性和重尾幅值的结构约束；

**🔧 技术方法**

使用了条件流匹配（conditional flow matching）技术、Transformer骨干网络、适应性层归一化、以及多项结构约束（L1重建、统计一致性、时频总变差与皮尔逊相关性）；

**📊 数据集**

在三大公开临床数据集上评估：TUAB（异常EEG）、TUEV（事件EEG）和TUSZ（癫痫EEG）；

**📈 对比分析**

与EEG-GAN和Vanilla Diffusion对比，JET在TS-FID、Silhouette Score和下游分类增益上均明显优于基线，TS-FID下降超过40%，下游分类器准确率提升正值；

**⚠️ 局限性**

局限性包括：仍需更大规模多中心数据验证；对极端噪声或异常信号的鲁棒性未完全评估；生成速度虽然较传统扩散更快，但仍高于GAN的即时采样。

---

## 561. Enhanced-BLE: A Hybrid BLE-ESB Framework for Dynamically Reconfigurable and Energy-Efficient 2.4 GHz IoT Communication

**arXiv ID:** 2605.21270 | [PDF](https://arxiv.org/pdf/2605.21270v1)

**作者:** Ziyao Zhou `[一作]` (Nanyang Technological University), Hen-Wei Huang `[通讯]` (Nanyang Technological University)

**通讯引用:** 2251 | [OpenAlex ID](https://openalex.org/A5029613558)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在统一的Nordic nRF54L15平台上系统评估BLE和ESB两种协议的功耗、吞吐率、唤醒延迟和双向通信性能，并提出基于MPSL的Hybrid Enhanced‑BLE框架，将两者优势结合。

**💡 创新点**

创新点在于利用MPSL实现BLE与ESB的协同工作，支持快速协议切换（≈18 ms）、低唤醒延迟（≈12 ms）和前向高吞吐率（≈2.2 Mbps）与后向可靠通信的双向融合。

**🔧 技术方法**

技术手段包括Nordic nRF54L15芯片、BLE 2M/4M PHY、ESB 4M PHY、MPSL调度框架、可动态调整TXP与PHY、功耗采样（PPK）和实验数据包（244/252字节）。

**📊 数据集**

未使用公开数据集，实验使用自制数据包并在不同距离下测量RSSI，评估吞吐率与功耗。

**📈 对比分析**

通过单包、连续包、睡眠唤醒、双向通信等实验对比，ESB在单包时耗时与能耗比BLE低约50%，吞吐率提升约两倍，唤醒延迟约为BLE的1/20；Hybrid Enhanced‑BLE实现前向≈2.2 Mbps、后向≈1.35 Mbps，手动切换延迟≈18 ms，整体功耗更低。

**⚠️ 局限性**

ESB缺乏安全、互操作性及可靠反向传输；Hybrid框架受BLE连接间隔和MPSL调度约束，跨厂商移植性及与其他2.4 GHz协议共存仍需进一步验证。

---

## 562. Systematic Design of Separation Logics

**arXiv ID:** 2605.21262 | [PDF](https://arxiv.org/pdf/2605.21262v1)

**作者:** Roberto Bruni `[一作]` (University of Pisa), Roberta Gori `[通讯]` (University of Pisa)

**通讯引用:** 761 | [OpenAlex ID](https://openalex.org/A5090195630)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种系统化的设计方法，用于从程序语义直接推导出分离逻辑的局部公理和框规则，从而在前向/后向、过/欠近似以及不同内存模型下构建可扩展、可重用的证明系统；

**💡 创新点**

创新点在于将局部性原理嵌入构造过程，通过逻辑变量与语义闭包来自动生成局部公理，消除传统框规则中的语法侧条件，并实现多种近似方向与内存模型的统一框架；

**🔧 技术方法**

技术核心包括：抽象解释的计算设计、语义闭包（后果、框、存在）与可组合的闭包序列、逻辑变量与通用框的定义、以及基于收集语义的归一化证明；

**📊 数据集**

由于研究属于理论推导与形式化证明，未使用任何真实数据集；

**📈 对比分析**

通过对比经典 SL、ISL、CISL、SepSIL 等已存在的逻辑，作者证明所生成的证明系统在可导三元组集合上严格优于原始逻辑，并给出了完整的相对完备性与可推理规则；

**⚠️ 局限性**

局限性包括：目前仅验证了无并发与有限内存模型的情况，未处理更复杂的断言语言和并发模型；方法对闭包序列的选择与最小化依赖手工证明，自动化程度有限；

---

## 563. STiTch: Semantic Transition and Transportation in Collaboration for Training-Free Zero-Shot Composed Image Retrieval

**arXiv ID:** 2605.21261 | [PDF](https://arxiv.org/pdf/2605.21261v1)

**作者:** Miaoge Li `[一作]`, Jingcai Guo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于多模态大语言模型和CLIP的检索框架STiTch，直接生成多描述并对其进行“转换”与“运输”处理，以实现图像检索；

**💡 创新点**

通过一阶段生成、转移向量与运输距离相结合的三步协同策略，避免两阶段生成带来的信息损失，并在嵌入空间中引入集合对集合（set-to-set）度量；

**🔧 技术方法**

使用多模态大语言模型（如Qwen2-VL、LLaVA-Next、GPT‑4o）生成描述，CLIP文本/图像编码器进行向量匹配，结合bidirectional distance、交通运输距离和可视化增强；

**📊 数据集**

在四个公开数据集上评测：CIRCO、CIRR、Fashion‑IQ、GeneCIS；

**📈 对比分析**

与现有多模态检索方法（PALAVRA、SEARLE、CIReVL、LDRE、OSrCIR、SEIZE、Pic2Word、LinCIR等）进行对比，STiTch在CIRCO/CIRR与GeneCIS上均取得最高或接近最高的mAP、Recall指标；在Fashion‑IQ上略逊于SEIZE；整体表现优于两阶段生成和传统CLIP基线；

**⚠️ 局限性**

在大规模图像增强时显著增加内存占用；在复杂场景或修改文本信息不足时可能聚焦错误对象；现有基准存在假负样本问题，导致真实有效检索被误判。

---

## 564. Transforming Privacy Artifacts into Accessible Reports for Non-Technical Stakeholders

**arXiv ID:** 2605.21269 | [PDF](https://arxiv.org/pdf/2605.21269v1)

**作者:** Zoe Pfister `[一作]` (University of Innsbruck), Michael Vierhauser `[通讯]` (University of Innsbruck)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个基于LMM的需求工程框架，能够将涉及人类监控的技术需求和隐私威胁分析转化为易于非技术利益相关者理解的隐私报告。

**💡 创新点**

创新点包括：①将传统安全分析工具（DFD、STRIDE）与LLM链式推理相结合，实现技术文档自动化转化为隐私报告；②在报告生成过程中引入多角色LLM代理，分别负责图形转化、威胁解释和需求简化，提升报告可读性与透明度。

**🔧 技术方法**

使用技术主要有：Data Flow Diagram (DFD)、STRIDE威胁模型、Mermaid图形描述、OpenAI Gemini 2.5 Pro、Claude Sonnet 4.5、LangChain / n8n 工作流自动化，以及链式推理和少量示例提示。

**📊 数据集**

采用了两份来自工业合作伙伴的真实用例（UC1：InLine Control of Product Assembly；UC2：Cycle Time Monitoring for Lean Production）作为输入数据，结合手工创建的DFD和STRIDE分析结果。

**📈 对比分析**

通过对比手工编写与自动生成的隐私报告，使用基于Krishna等人方法的六维度Likert量表（内部一致性、冗余、完整性、简洁性、正确性、可理解性）进行问卷评估，并在专家访谈中获取定性反馈。结果显示，自动生成的报告在完整性、简洁性和可理解性方面与手工报告相当或更优，但在一致性和术语规范性方面仍存在轻微差异。

**⚠️ 局限性**

主要局限包括：①评估样本仅包含两份用例，缺乏大规模验证；②LLM生成过程不确定，易出现格式或术语不一致；③报告质量高度依赖输入技术文档的一致性与完整性；④缺乏针对不同利益相关者角色的个性化报告功能；⑤尚未与多种隐私分析框架或监管文本（如GDPR条款）对接。

---

## 565. Tracing the ongoing emergence of human-like reasoning in Large Language Models

**arXiv ID:** 2605.21299 | [PDF](https://arxiv.org/pdf/2605.21299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 566. Hyper-V2X: Hypernetworks for Estimating Epistemic and Aleatoric Uncertainty in Cooperative Bird's-Eye-View Semantic Segmentation

**arXiv ID:** 2605.21309 | [PDF](https://arxiv.org/pdf/2605.21309v1)

**作者:** Abhishek Dinkar Jagtap `[一作]` (Technische Hochschule Ingolstadt), Andreas Festag `[通讯]` (Technische Hochschule Ingolstadt)

**通讯引用:** 5413 | [OpenAlex ID](https://openalex.org/A5032935966)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Hyper‑V2X框架，利用Bayesian hypernetwork对协作式BEV语义分割进行表观与噪声不确定性估计。

**💡 创新点**

创新点在于采用部分权重生成策略并引入V2X上下文嵌入来对decoder权重进行条件化，使得模型能够在保持低计算开销的前提下实现高质量不确定性推理，并保持对不同协作骨干的架构无关性。

**🔧 技术方法**

采用Bayesian hypernetwork、V2X上下文嵌入、部分权重生成、MC采样、CoBEVT等协作感知骨干以及ELBO风格损失进行训练与评估。

**📊 数据集**

使用公开的OPV2V协作感知基准数据集进行实验。

**📈 对比分析**

与CoBEVT、V2VNet、DiscoNet等前沿方法以及MC Dropout基线对比，Hyper‑V2X在IoU上从60.4提升至61.4，且在ECE、Brier Score、NLL等校准指标上均表现更佳。

**⚠️ 局限性**

仅针对BEV语义分割，仍需验证在更大规模、多模态（LiDAR、Radar）协作感知任务中的适用性，并且对通信延迟、数据同步等实际部署挑战缺乏深入研究。

---

## 567. Reducing Object Hallucination in LVLMs via Emphasizing Image-negative Tokens

**arXiv ID:** 2605.21300 | [PDF](https://arxiv.org/pdf/2605.21300v1)

**作者:** Meng Shen `[一作]` (Nanyang Technological University), Deepu Rajan `[通讯]` (Nanyang Technological University)

**通讯引用:** 4246 | [OpenAlex ID](https://openalex.org/A5009372982)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究视觉依赖度对大规模视觉语言模型（LVLM）生成文本中幻觉的影响，并提出基于视觉依赖度的损失重加权和数据过滤方法来减轻幻觉。

**💡 创新点**

创新点在于：①提出“视觉依赖度”指标将生成标记划分为图像正向、图像不变、图像负向三类；②在训练阶段对不同类型标记采用自适应权重调整，突出负向标记以压制幻觉；③通过去除视觉依赖度高或低的训练样本实现无额外推理成本的数据过滤。

**🔧 技术方法**

技术主要包括：对 LVLM 进行噪声图像（高斯噪声）对比，计算 token 概率差得到视觉依赖度；利用 LoRA 微调、softmax 重正则化的损失重加权；对训练样本按总视觉依赖度进行排序过滤；使用 CHAIR、FaithScore、MME、POPE 等基准进行评测。

**📊 数据集**

使用的数据集：COCO（用于分析幻觉分布）、LLaVA-Detail 23k、LLaVA-Instruct 150k、SVIT-Detail 71k（Bunny）、LLaVA-Mix 665k（用于 MME/POPE 评估）。

**📈 对比分析**

与 OPERA、VCD、EOS 等基线相比，方法在不牺牲回答长度的前提下显著降低 CHAIR_S、CHAIR_I 幻觉指标，Recall 略有下降；在 MME/POPE 评估中表现优于 EOS，接近或优于训练自由方法。

**⚠️ 局限性**

局限性包括：①幻觉抑制伴随目标检测 Recall 降低；②方法对噪声时间步（T=900）敏感，未系统探讨不同噪声级别；③仅在三种 LVLM（PaliGemma、LLaVA、Bunny）上验证，缺乏更广泛的跨模型通用性验证。

---

## 568. Deformba: Vision State Space Model with Adaptive State Fusion

**arXiv ID:** 2605.21308 | [PDF](https://arxiv.org/pdf/2605.21308v1)

**作者:** Hongyu Ke `[一作]` (Georgia State University), Haoxin Wang `[通讯]` (Georgia State University)

**通讯引用:** 2849 | [OpenAlex ID](https://openalex.org/A5101899364)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Deformba 视图状态空间模型，通过 Context‑Adaptive State Fusion (CASF) 实现自适应状态融合，支持 2D 视觉任务与多模态 3D 感知的高效计算。

**💡 创新点**

创新点在于将 SSM 的写/读操作解耦，使用偏移预测网络在写好的状态内自适应读取上下文信息，消除固定扫描路径限制，兼顾自注意力与跨模态查询交互，同时保持线性时间复杂度。

**🔧 技术方法**

核心技术包括 Mamba 状态空间模型、CASF（偏移预测 + bilinear 采样 + ECA 关注）、线性注意力框架、卷积 FFN、MESA 正则化，以及多阶段层级骨干网络。

**📊 数据集**

使用数据集包括 ImageNet‑1K（分类）、MS COCO（检测/实例分割）、ADE20K（语义分割）和 nuScenes（BEV 3D 检测）等。

**📈 对比分析**

与 CNN、ViT、SSM 等现有 SOTA 方法在同参数/ FLOPs 下进行对比；在 ImageNet Top‑1 达到 85.4%（相对 MambaOut‑B 低 1.2% 但整体领先），COCO bbox mAP 51.8%/mask mAP 69.8%，ADE20K mIoU 52.6%，nuScenes NDS 0.538，均显示出显著或相近的性能提升。

**⚠️ 局限性**

局限性在于仍依赖单次扫描写入，偏移预测在极复杂空间关系上可能不如多方向扫描；目前仅在典型任务验证，对低分辨率、实时部署等极端场景的鲁棒性尚待进一步评估。

---

## 569. Reliable Automated Triage in Spanish Clinical Notes: A Hybrid Framework for Risk-Aware HIV Suspicion Identification

**arXiv ID:** 2605.21256 | [PDF](https://arxiv.org/pdf/2605.21256v1)

**作者:** Rodrigo Morales-Sánchez `[一作]` (Universidad Nacional de Educacion a Distancia), Raquel Martínez `[通讯]` (Universidad Nacional de Educacion a Distancia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在西班牙临床记录中实现了一种针对早期 HIV 疑似检测的风险感知混合选择式分类框架

**💡 创新点**

创新点在于将不确定性拆分为概率（Mondrian conformal prediction）和几何（多重质心马氏距离 veto）两种独立约束，实现了对模糊和 OOD 示例的双重拦截

**🔧 技术方法**

采用注意力MIL网络、温度缩放、光谱归一化、马氏距离+全局协方差估计、蒙特卡洛 dropout 与交叉验证集成等技术，构建了多种不确定性后端

**📊 数据集**

使用来自 Madrid HUFA 医院的 13,642 名患者 63,802 条西班牙语临床笔记的真实数据集，包含严重类别不平衡（1:5.5 之比）

**📈 对比分析**

与传统 MC Dropout、CV 集成以及简单编码器基线对比，混合框架在严格安全阈值 α=0.01 下可覆盖 67.7% 的样本，Clear F₂ 达到 0.982，且误判率显著低于单一不确定性方法

**⚠️ 局限性**

局限性包括：仅在单一机构验证、几何阈值对时间或人口变化敏感、需要重新校准以适用于不同风险场景、以及对大型语言模型的不确定性提取尚未解决

---

## 570. MONET: A Massive, Open, Non-redundant and Enriched Text-to-image dataset

**arXiv ID:** 2605.21272 | [PDF](https://arxiv.org/pdf/2605.21272v1)

**作者:** Benjamin Aubin `[一作]` (Jasper Research), Clément Chadebec `[通讯]` (Jasper Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了MONET数据集，包含1.049亿对高质量图像-文本配对，并在此基础上训练了4B参数的文本到图像模型，展示其在GenEval与DPG评测中的竞争力。

**💡 创新点**

首次公开提供了全流程精细过滤、去重、跨模型多样化重新描述、合成数据增强以及预计算嵌入/标签的完整数据集；同时实现了多模型长短句混合重标注与大规模合成生成，显著提升了数据多样性和质量。

**🔧 技术方法**

使用美学评分、安全过滤、pHash+SSCD双阶段去重、域治理、多VLM（Florence2、InternVL3、ShareGPT4V、Gemini-2.5）重标注、FLUX系列模型合成、DINOv2/CLIP/SSCD嵌入、YOLO/Mediapipe标签、SANA VAE预编码；训练时采用潜在扩散框架、MMDiT风格去噪器、DCVAE压缩VAE、Qwen3-4B文本编码器。

**📊 数据集**

从LAION-2B、COYO、CC12M、Common-Catalog、Megalith-10M等公开来源共收集29亿条原始图文对，随后筛选、去重、重标注、合成，最终生成104.9M对的MONET；后续训练与评测使用ImageNet、GenEval、DPG等公开基准。

**📈 对比分析**

通过在ImageNet验证集上计算Long-CLIP对齐分数和FID，比较不同重标注模型与合成比例的影响；在GenEval与DPG上与多种公开模型对标，4B MONET模型在整体分数（GenEval 84.80，DPG 91.76）与同规模模型相近，显示其在公开数据上的可竞争性。

**⚠️ 局限性**

存在西方文化与皮肤色调偏重、年龄分布不均衡的偏见；仅提供英文描述；合成数据可能带来幻觉与风格偏差；安全过滤过于保守可能丢弃合法内容；样式与伦理注释仅覆盖子集；计算成本高，缺乏多语言与结构化属性标签。

---

## 571. SR-Ground: Image Quality Grounding for Super-Resolved Content

**arXiv ID:** 2605.21244 | [PDF](https://arxiv.org/pdf/2605.21244v1)

**作者:** Artem Borisov `[一作]` (Lomonosov Moscow State University), Dmitriy Vatolin `[通讯]` (Lomonosov Moscow State University)

**通讯引用:** 679 | [OpenAlex ID](https://openalex.org/A5020940244)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了用于超分辨率（SR）图像质量定位的 SR‑Ground 数据集，并在该数据集上训练了分割模型与交互式 SR 系统

**💡 创新点**

① 细粒度的 SR 专用视觉缺陷像素级标注；② 结合模型推断、人工标注与自适应迭代的多轮数据修正流程；③ 用 SR‑Ground 进行“质量定位”引导的 SR 训练，提升缺陷去除的可控性

**🔧 技术方法**

分割模型：Mask2Former、SegFormer；损失函数：交叉熵 + Dice；优化：AdamW + LoRA；SR 系统：基于 OSEDiff 的扩散模型；评估：mIoU/mAcc、F1/IoU；人机交互：基于 prominence 的众包校正

**📊 数据集**

主要使用 SR‑Ground（63k 张，6 类缺陷）；对比基准：Q‑Ground‑100K、Open Images、Molodetskikh 等；SR 生成模型包括 Real‑ESRGAN、BSRGAN、SwinIR、DiT4SR 等

**📈 对比分析**

与仅使用 Q‑Ground 预训练的模型相比，SR‑Ground 微调的模型在 Q‑Ground 测试集上的 mIoU/mAcc 提升约 5–10%；在 DeSRA 数据集上的 F1/IoU 达到无参考模型中最高；交互式 OSEDiff 能在指定区域成功添加或移除缺陷，保持全局一致性

**⚠️ 局限性**

仅覆盖 6 种缺陷，可能不足以代表所有 SR 产生的复杂混合缺陷；依赖众包主观评判，仍可能存在标注噪声；SR‑Ground 主要基于合成 SR 结果，真实拍摄 SR 结果的迁移性尚未完全验证

---

## 572. On the Cost and Benefit of Chain of Thought: A Learning-Theoretic Perspective

**arXiv ID:** 2605.21260 | [PDF](https://arxiv.org/pdf/2605.21260v1)

**作者:** Yue Zhang `[一作]` (University of Ottawa), Yongyi Mao `[通讯]` (University of Ottawa)

**通讯引用:** 3617 | [OpenAlex ID](https://openalex.org/A5004793184)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

本文提出了一个学习理论框架，用来分析大型语言模型的“Chain of Thought（CoT）”推理过程，定义了推理风险并将其分解为轨迹不匹配风险（TMR）和oracle轨迹风险（OTR）两项；

**💡 创新点**

创新点在于首次给出CoT风险的紧凑分解，并证明在不满足稳定性条件时TMR不可控，同时在满足稳定性条件下给出TMR的最优上界，揭示误差放大因子并与领域适应理论相连接；

**🔧 技术方法**

使用的技术主要包括学习理论工具（如quasi‑metric损失、稳定性分析、动态系统误差传播）、无免费午餐（no‑free‑lunch）定理、以及领域适应的分布差距度量；

**📊 数据集**

本研究为纯理论工作，没有使用任何公开数据集进行实验；

**📈 对比分析**

由于是理论研究，文中未给出实验比较或性能指标，重点在于提出理论界定与风险上界；

**⚠️ 局限性**

局限性包括：仅考虑固定长度的CoT步骤；链式规则与答案映射被假设为确定性；只对可恢复点的测试分布做分析，未考虑非可恢复点；未考虑随机采样的解码过程；链式规则与答案映射未联合学习。

---

## 573. Optimized Federated Knowledge Distillation with Distributed Neural Architecture Search

**arXiv ID:** 2605.21322 | [PDF](https://arxiv.org/pdf/2605.21322v1)

**作者:** Chaimaa Medjadji `[一作]` (University of Luxembourg), Feras M. Awaysheh `[通讯]` (Umea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了FedKD-NAS框架，将知识蒸馏和神经网络架构搜索相结合，在联邦学习中实现模型无关的高效知识蒸馏。

**💡 创新点**

创新点在于在每个客户端采用自适应NAS选择符合资源限制的学生模型，同时通过服务器端固定教师模型进行知识蒸馏，解决了非IID和系统异构带来的收敛与通信瓶颈。

**🔧 技术方法**

使用技术包括联邦知识蒸馏、神经架构搜索（NAS）、指数移动平均平滑、软标签蒸馏损失、温度调节等。

**📊 数据集**

使用数据集包括CIFAR-10、CIFAR-100、FMNIST、EMNIST、MNIST以及CASA人类活动识别数据集，并在真实边缘设备上验证。

**📈 对比分析**

与FedAvg、Ditto、FedDF、FedMD、FedDistill、Local-KD等方法对比，FedKD-NAS在非IID环境下保持1.0的准确率、PQS并且通信成本仅为1.6 MB，显著优于其它方法且总体性能最佳。

**⚠️ 局限性**

局限性是需要一份公开的参考数据集来生成logits；当类别数过大时logits通信量可能超过模型权重；NAS搜索仍有计算开销，且在极端资源受限设备上可能受限。

---

## 574. Automatic Discovery of Disease Subgroups by Contrasting with Healthy Controls

**arXiv ID:** 2605.21301 | [PDF](https://arxiv.org/pdf/2605.21301v1)

**作者:** Robin Louiset `[一作]` (Université Paris-Saclay), Pietro Gori `[通讯]` (Université Paris-Saclay)

**通讯引用:** 774 | [OpenAlex ID](https://openalex.org/A5103276787)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于深度学习的子群体发现方法 Deep UCSL，用健康对照组对比识别病理子群体。

**💡 创新点**

创新点在于：① 将子群体发现与健康对照的对比学习结合；② 采用 EM 迭代联合训练深度编码器、子群体聚类头和混合专家分类器；③ 引入聚类正则化和 Sinkhorn‑Knopp 软聚类，解决聚类退化与身份重识别问题；④ 通过统一框架实现无子群标签监督的子群体发现。

**🔧 技术方法**

使用深度卷积编码器、Mixture‑of‑Experts 分类器、Soft‑K‑Means（带 Sinkhorn‑Knopp 约束）以及 Expectation‑Maximization 优化，辅以 KL 正则化和子群体伪标签。

**📊 数据集**

实验数据集包括：MNIST（数字 7 作为病理类）、三维 MRI T1（精神分裂症与躁郁症子群）、儿童肺炎（病毒/细菌子群）、OCT 视网膜病变（多子群）以及 ODIR 视网膜疾病数据集。

**📈 对比分析**

与 Deep Cluster、PCL、SimCLR、SupCon、WS‑DeepClustering 等方法比较，Deep UCSL 在所有实验中的子群体均衡准确率（Subgroup B‑ACC）和整体均衡准确率（Overall B‑ACC）均显著优于对手，特别在 MNIST、精神病学、肺炎和视网膜子群任务中取得最优或接近上限的性能。

**⚠️ 局限性**

局限性包括：① 需要事先设定子群数量 K；② 仅在有二分类（健康/病理）标签的场景适用；③ 训练时需要多阶段 EM 迭代，计算成本相对较高；④ 对数据噪声和采集多样性敏感，可能需要额外的领域自适应或增广策略。

---

## 575. TimeSRL: Generalizable Time-Series Behavioral Modeling via Semantic RL-Tuned LLMs -- A Case Study in Mental Health

**arXiv ID:** 2605.21295 | [PDF](https://arxiv.org/pdf/2605.21295v1)

**作者:** Yuang Fan `[一作]` (Columbia University), Xuhai "Orson" Xu `[通讯]` (Columbia University)

**通讯引用:** 2600 | [OpenAlex ID](https://openalex.org/A5066796307)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个两阶段的 LLM 框架 TimeSRL，先将长期行为时间序列转化为自然语言抽象，再通过该抽象预测心理健康指标。

**💡 创新点**

创新点在于通过强化学习与可验证奖励联合训练，使抽象过程与最终预测目标对齐，形成鲁棒的语义中介，不需要中间标注，显著提升跨数据集泛化性能。

**🔧 技术方法**

采用大型语言模型、两阶段 Prompt 设计、Group Relative Policy Optimization (GRPO) 与 Reinforcement Learning from Verifiable Rewards (RLVR) 对抽象与预测进行端到端优化。

**📊 数据集**

在 GLOBEM 与 College Experience 两个长期被动感知的心理健康预测数据集上进行评估，包含 14 天多变量行为轨迹与 PHQ-4 子分数。

**📈 对比分析**

与传统 ML、最先进的非 LLM 行为建模方法以及直接提示 LLM（GPT‑5.0、Qwen3‑4B）对比，在 LO​SO 交叉数据集下，TimeSRL 在焦虑/抑郁预测上平均 MAE 分别降低 3.1–10.1% 与 9.5–44.1%（相较非 LLM）和 27.4–57.6%（相较 LLM），并在不同 LLM backbone 与跨 benchmark 转移中保持领先。

**⚠️ 局限性**

局限性包括训练成本高、需要手工设计提示模板、只能处理固定长度窗口、对奖励函数和输出格式敏感，以及对中间摘要质量与临床可解释性缺乏系统评估。

---

## 576. \textit{Stochastic} MeanFlow Policies: One-Step Generative Control with Entropic Mirror Descent

**arXiv ID:** 2605.21282 | [PDF](https://arxiv.org/pdf/2605.21282v1)

**作者:** Zeyuan Wang `[一作]` (National University of Defense Technology), Yanwei Fu `[通讯]` (Fudan University)

**通讯引用:** 16485 | [OpenAlex ID](https://openalex.org/A5084959430)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出一种Stochastic MeanFlow Policies（SMFP），即一跳生成式策略，可在在线离线强化学习中同时实现SAC式熵正则和镜像下降优化。

**💡 创新点**

创新点在于将MeanFlow的噪声到动作映射与高斯重参数化相结合，得到可计算的熵下界，从而兼容熵正则化；同时通过优势加权的镜像下降回归，将多模态目标与表达式策略匹配，突破单峰高斯策略的局限。

**🔧 技术方法**

技术手段包括：一次性MeanFlow生成网络、随机重参数化、熵下界约束、优势加权镜像下降回归、价值引导采样、JAX实现、单步推断与并行采样。

**📊 数据集**

在七个MuJoCo连续控制任务上进行实验：Hopper、Walker2d、Ant、HalfCheetah、Humanoid、HumanoidStandup、Swimmer。

**📈 对比分析**

与15种基线（PPO、SPO、TD3、SAC、DIPO、QVPO、DIME、MaxEntDP、DPMD、SAC Flow-T/G、FlowRL、FPMD-R/M）在30个随机种子下比较。SMFP在多数任务获得最优或近似最优累计奖励，并保持单步推断效率，体现了性能与计算效率兼顾的优势。

**⚠️ 局限性**

局限性包括：对熵温度α和镜像下降系数λ的超参数敏感；在高维动作空间或更复杂环境中可能仍出现探索不足或训练不稳定；实验仅在MuJoCo环境验证，未探索离散动作或更大规模任务的泛化。

---

## 577. Nonparametric Learning and Earning with One-Point Feedback under Nonstationarity

**arXiv ID:** 2605.21263 | [PDF](https://arxiv.org/pdf/2605.21263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 578. Towards Single Exponential Time for Temporal and Spatial Reasoning: A Study via Redundancy and Dynamic Programming

**arXiv ID:** 2605.21267 | [PDF](https://arxiv.org/pdf/2605.21267v1)

**作者:** Victor Lagerkvist `[一作]` (Linköping University), Leif Eriksson `[通讯]` (Linköping University)

**通讯引用:** 10853 | [OpenAlex ID](https://openalex.org/A5001315059)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了空间-时间定性推理中的RCC和IA两大问题的求解复杂度，重点证明了约束冗余的最大数量上界，并提出两种动态规划算法，分别在𝔄₃⁽ᵁ⁾上实现4ⁿ时间，和在RCC‑8⁽ᵁ⁾上实现(o(n))ⁿ时间；

**💡 创新点**

创新点包括：①完整分类了所有基本关系的最大非冗余约束数；②提出了基于状态压缩的DP方法，首次在NP‑hard定性推理中实现单指数/超线性指数求解；③利用不一致路径(IP)与最小化总序集合相结合的思路，显著降低RCC‑8⁽ᵁ⁾的状态空间；

**🔧 技术方法**

技术手段主要为：动态规划、状态压缩、冗余约束分析、组合最优子结构证明、使用不一致路径进行状态比较与最小化；

**📊 数据集**

本文属于理论分析，没有使用实验数据集；

**📈 对比分析**

与先前的枚举方法相比，4ⁿ算法比传统2^(n log n)方法提升约O((1/4)^n)的指数因子；RCC‑8⁽ᵁ⁾的(o(n))ⁿ时间大幅优于之前的(0.531n)^n约束；

**⚠️ 局限性**

局限性在于：①对完整RCC（8）仍未突破单指数2ⁿ界；②算法复杂度仍为高指数，对大规模实例的实际可行性有限；③缺乏实验验证，难以评估实际性能。

---

## 579. Divide and Contrast: Learning Robust Temporal Features without Augmentation

**arXiv ID:** 2605.21241 | [PDF](https://arxiv.org/pdf/2605.21241v1)

**作者:** Abdul-Kazeem Shamba `[一作]` (Norwegian University of Science and Technology), Gavin Taylor `[通讯]` (United States Naval Academy)

**通讯引用:** 2388 | [OpenAlex ID](https://openalex.org/A5043788981)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为 Divide and Contrast (Di‑COT) 的自监督时间序列表示学习框架，利用随机划分重叠子块并在子块之间进行对比，无需数据增强或多次编码器传递。

**💡 创新点**

创新点在于：①通过在窗口内随机划分子块实现多尺度、信息丰富的对比；②将下一个子块的预测转化为跨子块的分类任务，避免传统的 DTW 或多次前向传播；③实现了与序列长度无关的损失计算，显著降低训练时间。

**🔧 技术方法**

使用的技术包括：InceptionTime 编码器、温度尺度的 InfoNCE/分类损失、随机子块划分与重叠、跨子块对比、密集监督等。

**📊 数据集**

实验数据集涵盖六个大规模真实世界时间序列数据集（PAMAP2、WISDM2、HARTH、SLEEP、ECG、SKODA）以及 UCR 与 UEA 基准。

**📈 对比分析**

与多种 SSL 方法（TNC、TS‑TCC、TS2Vec 等）和传统 ML 基线（MiniRocket、RF）在 linear probing、kNN、聚类、低标签和跨域迁移等任务中对比，Di‑COT 在多数任务上获得 state‑of‑the‑art 或接近最佳性能，同时训练时间显著最短。

**⚠️ 局限性**

局限性在于：该方法主要关注判别式表示学习，可能不适用于需要精细时间预测或生成的任务；对预测类任务的适用性需进一步研究。

---

## 580. APEX: Autonomous Policy Exploration for Self-Evolving LLM Agents

**arXiv ID:** 2605.21240 | [PDF](https://arxiv.org/pdf/2605.21240v1)

**作者:** Yibo Li `[一作]` (National University of Singapore), Bryan Hooi `[通讯]` (National University of Singapore)

**通讯引用:** 5952 | [OpenAlex ID](https://openalex.org/A5065675832)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为APEX的自我进化LLM代理框架，用显式策略图解决探索崩塌问题。

**💡 创新点**

创新点在于将策略空间建模为有向无环图，并通过Fork Discovery主动扩展未探索的里程碑，同时用Policy Selection在图中进行基于不确定性的探索。

**🔧 技术方法**

核心技术包括LLM生成的结构化摘要、策略图更新、奖励回传、Thompson Sampling等多项式化探索算法。

**📊 数据集**

在九款Jericho文本冒险游戏和WebArena网页交互任务上进行评估。

**📈 对比分析**

相较于Static、Memory、Reflexion、EvoTest和ACE等基线，APEX在所有任务上均取得显著提升，尤其在需要发现多样化策略的游戏中表现最优。

**⚠️ 局限性**

局限性包括对LLM识别里程碑和未探索方向的依赖，难以发现完全超出历史经验的策略，且主要适用于离散里程碑型任务。

---

## 581. Multimodal Emotion Recognition with Large Language Models

**arXiv ID:** 2605.21239 | [PDF](https://arxiv.org/pdf/2605.21239v1)

**作者:** Hongrui Zhang `[一作]` (Tsinghua University), Sicheng Zhao `[通讯]` (Tsinghua University)

**通讯引用:** 8688 | [OpenAlex ID](https://openalex.org/A5051149140)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文对多模态情感识别（MER）与大型语言模型（LLM）结合的最新研究进行了系统综述，提出了三大核心挑战（情感数据稀缺、跨模态情感鸿沟、解释不透明）并将现有方法分为情感数据增强、多模态情感表征和多模态情感推理三大方向，构建统一框架与分类法；

**💡 创新点**

创新点在于首次将MER-with-LLMs的研究从挑战角度进行系统划分，提出了以LLM为中心的协同架构，阐明了数据增强、表征优化与解释推理之间的关系，并为未来研究提供了可操作的开放性问题与发展方向；

**🔧 技术方法**

主要技术包括LLM自回归生成、SFT与RL微调、Q-former、Mixture-of-Experts、注意力融合、表征锐化/增强、开放词汇情感标注、解释性推理链条以及多模态适配器；

**📊 数据集**

使用了多种公开及自建数据集，如GVEC、VEC-CoT、EmoVIT、EMO-LLaMA、EmotionHallucer、MM-BigBench、MER-UniBench、EEmo-Bench、FaceBench等，覆盖图像、视频、音频、文本及对话等子任务；

**📈 对比分析**

通过在不同子任务上对比0-shot、SFT、RL等方法，发现情感解释与主观性推理方向可显著提升准确率/召回率，但整体仍受零-shot泛化、长尾识别、跨文化迁移、情感幻觉及安全偏见等问题制约；

**⚠️ 局限性**

局限性包括缺乏跨子任务统一框架与机制理解、对主观性与开放词汇情感的理论化与评价不足、缺少Agentic情感交互与实时反馈机制，以及文化适应与偏见治理方面的实质性解决方案。

---

## 582. On the Regularity and Generalization of One-Step Wasserstein-guided Generative Models for PDE-Induced Measures

**arXiv ID:** 2605.21388 | [PDF](https://arxiv.org/pdf/2605.21388v1)

**作者:** Likun Lin `[一作]` (University of Hong Kong), Zhiwen Zhang `[通讯]` (University of Hong Kong)

**通讯引用:** 2664 | [OpenAlex ID](https://openalex.org/A5100611977)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一阶 Wasserstein 引导的生成模型（如 DeepParticle）对 PDE 诱导概率分布的理论解析，证明了该类目标分布满足双重性（doubling）并导致最优传输映射具有 Hölder 连续性，进而给出该模型的过拟合风险（excess‑risk）上界与样本大小关系，并对目标分布漂移（target shift）给出鲁棒性估计。

**💡 创新点**

① 通过 PDE 的 Schauder 估计与几何双重性，首次证明了线性椭圆、抛物方程以及 Torus 上的 Fokker–Planck 诱导分布的最优传输映射具 Hölder 连续性；② 将该正则性结果与 ReLU 网络逼近理论、经验 Wasserstein 误差估计结合，得到 DeepParticle 在样本量 N 下的具体收敛速率；③ 给出目标分布漂移下的 OOD（out‑of‑distribution）风险上界。

**🔧 技术方法**

利用了：
- PDE 正则性（Schauder 理论、Hopf 边界点估计）、
- 双重性（doubling measures）与其对最优传输的 Hölder 正则性，
- ReLU 网络对 Hölder 函数的逼近理论，
- 经验 Wasserstein 统计上界（empirical measure 的 W₂ 误差），
- 训练过程的优化误差估计与一般化误差分解。

**📊 数据集**

主要使用了由线性椭圆/抛物方程、Torus 上 Fokker–Planck 方程产生的概率密度，尤其在实验中取：
- 1 维区间上 u''=0 的 Dirichlet 例子，
- 2 维单位圆盘上 -Δu=1、u=0 的例子，
- 以及相应的目标分布的经验样本集。

**📈 对比分析**

实验中固定网络结构，随样本量 N 变化训练 DeepParticle，测量验证集的 W₂ 损失。结果显示：1 维时误差≈N⁻¹/²，2 维时误差≈N⁻¹/⁴，基本符合理论给出的收敛速率；与传统的多步扩散模型相比，实验表明一阶模型在相同样本规模下实现更快的收敛。

**⚠️ 局限性**

限制包括：
- 仅针对线性、边界有界且凸域的 PDE（椭圆、抛物、Torus Fokker–Planck）；
- 正则性与双重性证明依赖于 PDE 的特殊结构，难以推广到非线性或无界域情况；
- 经验 Wasserstein 误差上界与维度依赖仍相对粗糙；
- 优化误差仅以抽象形式出现，缺乏对实际训练算法（如 Adam）收敛性的理论保证；
- 对 OOD 误差的鲁棒性估计是单参数的，未考虑更复杂的目标分布漂移。

---

## 583. roto 2.0: The Robot Tactile Olympiad

**arXiv ID:** 2605.21429 | [PDF](https://arxiv.org/pdf/2605.21429v1)

**作者:** Elle Miller `[一作]` (University of Edinburgh), Sethu Vijayakumar `[通讯]` (University of Edinburgh)

**通讯引用:** 10160 | [OpenAlex ID](https://openalex.org/A5069715982)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Robot Tactile Olympiad v2基准套件，涵盖四种不同的类人手部机械结构，提供无视觉“盲”触觉强化学习任务（Baoding球旋转与球弹跳）并开源模拟环境与基线；

**💡 创新点**

创新点在于统一标准化跨形态触觉RL平台、完全基于本体与触觉感知的端到端盲策略、GPU并行训练加速、并在无显式姿态估计或教师-学生蒸馏的条件下实现显著性能突破；

**🔧 技术方法**

采用自定义PPO实现（SKRL框架）、观察堆叠、自监督前向动力学、环境分离评估、超参数搜索、Isaac Lab GPU并行仿真；

**📊 数据集**

使用四种手部机器人（Shadow Dexterous Hand、Shadow Lite、Allegro Hand、ORCA Hand）在仿真中构建的Baoding球旋转与弹跳任务数据集；

**📈 对比分析**

与基于状态（包含物体位置信息）的代理进行对比，盲代理在弹跳任务中可达80次弹跳（200M步），在Baoding任务中最高13次旋转（理论最高约35次）；

**⚠️ 局限性**

局限包括盲策略在多物体Baoding任务中高方差、对触觉信息的二值化限制、仅限仿真环境，尚未验证真实硬件的转移性能；

---

## 584. Classification of Single and Mixed Partial Discharges under Switching Voltage Using an AWA-CNN Framework

**arXiv ID:** 2605.21352 | [PDF](https://arxiv.org/pdf/2605.21352v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 585. Teaching AI Through Benchmark Construction: QuestBench as a Course-Based Practice for Accountable Knowledge Work

**arXiv ID:** 2605.21413 | [PDF](https://arxiv.org/pdf/2605.21413v1)

**作者:** Haiyang Shen `[一作]` (Peking University), Yun Ma `[通讯]` (Peking University)

**通讯引用:** 78797 | [OpenAlex ID](https://openalex.org/A5100369226)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于课程的AI教育练习，通过学生设计专家级问答来构建BenchMark，并用它评估深度研究型AI系统。

**💡 创新点**

创新点在于将Benchmark构造与教学融合，使学生主动制定任务、标准和验证流程，从而培养对AI产出负责的判断能力；同时用学生设计的任务显著揭示了现有深度搜索系统的隐蔽失败。

**🔧 技术方法**

使用深度研究（retrieval‑augmented generation）AI系统（如 GPT‑5.5、Claude Opus 4.7 等），并结合人类评审、反向攻击与多轮质量控制的技术。

**📊 数据集**

使用自建的 QuestBench 数据集，包含 256 道跨 14 个人文社科领域的专家级问题、答案和评判标准，数据已公开在 HuggingFace。

**📈 对比分析**

对 13 个前沿深度搜索系统进行统一评测，平均通过率仅 16.85%，最高系统 GPT‑5.5 的通过率为 57.58%。评测揭示主要失败模式为检索失败、推理不完整和答案提取错误，表明当前模型在专业领域的可靠性仍有限。

**⚠️ 局限性**

局限性包括：仅覆盖人文社科领域，未涉及自然科学/工程/医学等；需要具备高级学科背景的学生和教师支持；评测基于单一时点的模型与工具，结果随技术演进而变化；教学效果需在更长期、多机构、多学科中验证。

---

## 586. Disentangling Generation and Regression in Stochastic Interpolants for Controllable Image Restoration

**arXiv ID:** 2605.21381 | [PDF](https://arxiv.org/pdf/2605.21381v1)

**作者:** Yi Liu `[一作]` (Tongji University), Yichao Zhang `[通讯]` (Tongji University)

**通讯引用:** 10632 | [OpenAlex ID](https://openalex.org/A5100419861)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DiSI 框架，将图像恢复中的回归与生成分离，允许在同一模型下通过调节时间参数实现失真-感知平衡。

**💡 创新点**

创新点：①引入回归时间 r 与生成时间 g 两个独立时间变量，解耦 SI 的确定性回归路径与随机生成过程；②设计可在任意轨迹下使用的解析式一阶采样器；③构建双分支 pixel‑space DULiT 网络，提升条件引导与推理效率。

**🔧 技术方法**

使用技术：Stochastic Interpolants、概率流 ODE、线性注意力、AdaGroupNorm、Dual‑branch U‑Net Transformer、adaptive loss weighting 等。

**📊 数据集**

数据集：Rain100H（去雨）、GoPro（去模糊）、LOL（低光增强）、CelebA‑HQ（填空）。

**📈 对比分析**

与 SOTA 退化‑恢复方法（DM/FM/IR‑SDE/DB 等）在 PSNR/SSIM 与 LPIPS/FID 上对比，DiSI 在保持高 PSNR/SSIM 的同时通过调节 δ 达到与 DM/FM 相当或更优的感知指标，并且推理速度更快。

**⚠️ 局限性**

局限性：对轨迹与采样参数的手工调节敏感，过大或过小的 δ 及 NFE 会导致质量下降；对极端噪声水平或大尺寸图像的泛化尚未充分验证。

---

## 587. Closed Loop Dynamic Driving Data Mixture for Real-Synthetic Co-Training

**arXiv ID:** 2605.21372 | [PDF](https://arxiv.org/pdf/2605.21372v1)

**作者:** Hongzhi Ruan `[一作]` (Li Auto), Kun Zhan `[通讯]` (Li Auto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了在端到端自动驾驶中，如何在有限预算下通过闭环评估驱动的自动化数据混合来优化真实-合成数据组合。

**💡 创新点**

提出了 AutoScale 框架，结合 Graph-RAE 场景表征、Cluster-GA 聚类梯度上升以及向量检索，实现了基于评估反馈的动态数据混合与检索。

**🔧 技术方法**

采用图正则化自编码器（Graph-RAE）、聚类梯度上升（Cluster-GA）以及基于图嵌入的检索技术，构建闭环数据优化循环。

**📊 数据集**

在 NAVSIM 真实驾驶数据与 SimScale 合成数据（约 85K 真实场景 + 384K 合成场景）上进行实验。

**📈 对比分析**

与传统统一采样、Chameleon、IWR 等基线相比，AutoScale 在 LTF 与 DiffusionDrive 两种模型上通过 50K-100K 合成样本获得更高的 EPDMS 分数，数据利用率显著提升。

**⚠️ 局限性**

仅在两种模型和有限迭代轮次下验证，缺乏在更大模型或强化学习任务中的泛化验证。

---

## 588. RoadTones: Tone Controllable Text Generation from Road Event Videos

**arXiv ID:** 2605.21411 | [PDF](https://arxiv.org/pdf/2605.21411v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 589. Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents

**arXiv ID:** 2605.21347 | [PDF](https://arxiv.org/pdf/2605.21347v1)

**作者:** Akshay Manglik `[一作]` (Scale AI), Xue `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种名为Insights Generator（IG）的多代理系统，利用大规模LLM agent执行轨迹的语义特征，自动生成可验证的自然语言洞察报告，帮助工程师诊断和改进agent行为；

**💡 创新点**

创新点包括：① scout‑investigator 分层角色，分离假设生成与验证；② 通过 stateful Python 数据处理层避免LLM上下文窗口限制，支持跨句子、跨模型的批量统计；③ 引入四维评估框架（LLM judge、人工专家、下游改进、自动补丁循环），系统性评估洞察质量与实际效用；

**🔧 技术方法**

技术栈：多代理架构（Orchestrator、Scout、Investigator），Python工具集合（trace inspection、工具调用、A/B cohort comparison、聚类、统计检验），LLM推理（Claude Opus 4.6、Claude Code），基于模型的代码执行、自动化补丁循环，LLM judge与人类专家评分。

**📊 数据集**

使用的基准数据集包括 SpreadsheetBench‑Verified（约400个已验证任务，包含大量agent执行轨迹）和 HLE（约250条轨迹），以及在补丁循环实验中专门使用的 SpreadsheetBench 训练/测试拆分。

**📈 对比分析**

与四种对照系统（Single‑Agent Coding、CC Subagents、Trace2Skill、RLM）在四个评估维度下对比：覆盖率约91%平均、对手胜率最高（77.9%）、按维度质量分最高；在人工专家改进实验中 IG 报告使 scaffold 效率提升30.4pp，对照为16.2pp；在自动补丁循环实验中 IG 与 RLM、CC Subagents 均将 pass 率提升至0.81‑0.84，纯补丁基线退化至0.58。成本与时间：IG平均约$76 LLM成本、48分钟。

**⚠️ 局限性**

局限性：实验样本量偏小，仅在 SpreadsheetBench 进行干预评估，缺乏在更大规模、多领域任务（如 Humanity's Last Exam、SWE‑Bench Pro 等）的验证；LLM judge 与人工专家的评估结果存在偏差；未充分评估 IG 在不同 LLM/工具组合下的可扩展性；补丁循环实验样本有限，难以统计其方差。

---

## 590. FedCritic: Serverless Federated Critic Learning-based Resource Allocation for Multi-Cell OFDMA in 6G

**arXiv ID:** 2605.21418 | [PDF](https://arxiv.org/pdf/2605.21418v1)

**作者:** Amin Farajzadeh `[一作]` (University of Ottawa), Melike Erol-Kantarci `[通讯]` (University of Ottawa)

**通讯引用:** 8000 | [OpenAlex ID](https://openalex.org/A5089891162)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出 FedCritic，一种全局无协调、服务器无关的联邦强化学习框架，用于 6G 多小区 OFDMA 的资源调度与功率分配。

**💡 创新点**

创新点在于仅通过邻居 gossip 方式聚合评论器参数，消除了 CTDE 的集中评论器瓶颈，同时保持策略本地化；同时结合虚拟队列实现长期 QoS 控制，并在干扰密集的 reuse‑1 环境下实现稳定训练。

**🔧 技术方法**

使用多智能体 actor‑critic（MAPPO）算法、在线虚拟队列、EMA 近似干扰、gossip‑based 参数平均、PPO 损失、GAE 等技术。

**📊 数据集**

在 7‑站、32 子载波、8 UE/站、离散功率级别的模拟仿真环境中评估，未使用公开数据集。

**📈 对比分析**

与 CTDE‑MAPPO、CTDE+VQ、FedActor、贪婪与 QoS 启发式方法对比；FedCritic 在平均总速率、SINR 分布、邻居碰撞率与公平性方面均优于基线，训练更稳定、通信开销更低。

**⚠️ 局限性**

局限包括对固定干扰图、静态网络规模的假设；未证明全局收敛；在极大规模或时间变化网络中仍需进一步验证与改进。

---

## 591. What Twelve LLM Agent Benchmark Papers Disclose About Themselves: A Pilot Audit and an Open Scoring Schema

**arXiv ID:** 2605.21404 | [PDF](https://arxiv.org/pdf/2605.21404v1)

**作者:** Mahdi Naser Moghadasi `[一作]` (BrightMind AI), Faezeh Ghaderi `[通讯]` (University of Texas at Arlington)

**通讯引用:** 502 | [OpenAlex ID](https://openalex.org/A5074537862)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过设计一个包含五个字段（benchmark identity、harness specification、inference settings、cost reporting、failure breakdown）的审核 schema，对12篇LLM代理和传统静态基准论文进行一次性评分，记录每篇论文在各维度上的披露程度，并讨论了现有基准论文的披露缺口与改进建议。

**💡 创新点**

创新点在于提出了针对LLM代理评估运行的可操作披露标准与评分代码表，首次量化了代理基准论文的披露完整度，并揭示了成本与环境规范等重要信息的普遍缺失。

**🔧 技术方法**

使用的技术包括：JSON Schema 架构定义、Markdown 代码书、Python 脚本验证器、手工评分流程，以及对 GitHub 仓库与论文文本的文本提取与比对。

**📊 数据集**

使用的数据集为12篇基准论文的文本与对应 GitHub 仓库（8 篇代理基准：SWE-bench、WebArena、OSWorld、GAIA、AgentBench、VisualWebArena、Mind2Web、MLE-bench；4 篇传统静态基准：HumanEval、MMLU、GSM8K、MBPP）。

**📈 对比分析**

比较方法是对每个字段按 {0, 0.5, 1} 评分，计算每篇论文的平均披露得分。结果显示代理基准平均得分为0.38（满分1.0），传统基准为0.66，代理基准在成本披露与环境镜像指向方面表现最差，平均为0.0；在 harness 规范和 inference settings 也仅为 0.44 与 0.56。

**⚠️ 局限性**

限制包括：仅有单一审核者，评分过程中存在主观边界判断；样本量小（12 篇）；依赖公开文本与仓库，若出现渲染或访问失败可能导致信息缺失；评分仅衡量披露完整度，未评估实验正确性或结果可信度。

---

## 592. "I didn't Make the Micro Decisions": Measuring, Inducing, and Exposing Goal-Level AI Contributions in Collaboration

**arXiv ID:** 2605.21363 | [PDF](https://arxiv.org/pdf/2605.21363v1)

**作者:** Eunsu Kim `[一作]` (KAIST), Sherry Tongshuang Wu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2322 | [OpenAlex ID](https://openalex.org/A5004225142)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于目标的归因框架CoTrace，用以量化人机协作过程中的目标与需求层面贡献

**💡 创新点**

创新点在于从过程层面追踪直接与间接的目标塑造行为，拆分目标为可验证需求并构建自动化归因流水线，且将其作为评估、设计与反思工具

**🔧 技术方法**

使用LLM-as-judge实现的自动化管线：行动提取、需求抽取、影响标签及贡献聚合，并通过提示工程与交互设计进行实验

**📊 数据集**

使用公开的ShareChat与CoGym-Real数据集，共638条人机对话日志，涵盖编程、数据分析、写作与规划四大任务领域

**📈 对比分析**

与传统基于最终产出的归因方法相比，CoTrace能够量化模型在细粒度需求层面的贡献；在模拟实验中，特定交互与提示策略可将模型需求生成提升至约70%，但对最终输出质量无显著提升

**⚠️ 局限性**

局限性包括：依赖LLM-as-judge的准确性；仅覆盖有限任务类型与平台；难以评估最终产出质量；工具需进一步验证其对不同用户群体的普适性

---

## 593. LASH: Adaptive Semantic Hybridization for Black-Box Jailbreaking of Large Language Models

**arXiv ID:** 2605.21362 | [PDF](https://arxiv.org/pdf/2605.21362v1)

**作者:** Abdullah Al Nomaan Nafi `[一作]` (University of Maine), Prabuddha Chakraborty `[通讯]` (University of Maine)

**通讯引用:** 892 | [OpenAlex ID](https://openalex.org/A5013184572)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种黑盒元攻击框架LMM Adaptive Semantic Hybridization (LASH)，将多种基础 jailbreak 攻击的输出作为可重用种子，并通过遗传算法对混合权重进行无梯度优化，生成单一高效 jailbreak prompt。

**💡 创新点**

创新点在于将不同攻击家族的输出视为可组合的种子，并在每个请求中自适应调整权重，突破单一家族局限；利用无梯度遗传搜索与 LLM 组成器实现语义层面的混合。

**🔧 技术方法**

使用技术包括黑盒查询、遗传算法 (GA) 对混合 logits 进行无梯度优化、softmax 混合、LLM 组成器、两阶段 fitness（关键词拒绝检测+LLM Judge）、以及机制分析的激活探针与补丁。

**📊 数据集**

评估数据集为 JailbreakBench，包含 100 条有害提示，覆盖 10 个行为类别。

**📈 对比分析**

与 5 个最先进的黑盒基线（PAIR、TAP、AutoDAN、AutoDAN‑Turbo、FlipAttack）在 6 个目标模型上比较，LASH 在 ASR_1 84.5%/ASR_2 74.5% 的平均成功率，平均查询仅 30，显著优于基线；在 SmoothLLM、Llama Guard 与 perplexity filtering 等三种防御机制下仍保持竞争力。

**⚠️ 局限性**

局限性包括仅针对单轮 prompt 优化、未考虑多轮对话或防御感知，且依赖于黑盒种子生成和固定的查询预算。

---

## 594. Data-Efficient Neural Operator Training via Physics-Based Active Learning

**arXiv ID:** 2605.21348 | [PDF](https://arxiv.org/pdf/2605.21348v1)

**作者:** Alicja Polanska `[一作]` (University College London), Stanislas Pamela `[通讯]` (United Kingdom Atomic Energy Authority)

**通讯引用:** 2014 | [OpenAlex ID](https://openalex.org/A5030017826)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了基于物理残差的主动学习方法，用于高效构建神经算子训练数据。

**💡 创新点**

创新点在于利用 PDE 残差作为模型不确定性度量，并通过参数归一化注入物理先验。

**🔧 技术方法**

采用 FNO 结构、有限差分卷积核估算 PDE 残差、top‑k 与 SBAL 等主动学习策略。

**📊 数据集**

在 1D Burgers 方程和 2D 可压 Navier‑Stokes 方程的数值实验中进行验证。

**📈 对比分析**

与随机采样和 LCMD 比较，物理残差采样在训练轨迹更少时即可达到相同 RMSE，优于随机采样，近似与 LCMD 的数据效率相当。

**⚠️ 局限性**

限制主要在于残差归一化在不同动力学 regime 下的有效性、FNO 条件化不足以及对极端参数范围的适用性仍需改进。

---

## 595. Preference-aware Influence-function-based Data Selection Method for Efficient Fine-Tuning

**arXiv ID:** 2605.21422 | [PDF](https://arxiv.org/pdf/2605.21422v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 596. Text Analytics Evaluation Framework: A Case Study on LLMs and Social Media

**arXiv ID:** 2605.21338 | [PDF](https://arxiv.org/pdf/2605.21338v1)

**作者:** Yuefeng Shi `[一作]` (Cardiff University), Jose Camacho-Collados `[通讯]` (Cardiff University)

**通讯引用:** 3611 | [OpenAlex ID](https://openalex.org/A5086289154)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了针对社交媒体文本的LLM数据分析基准，设计470个手工验证的问答题目，评估LLM在长文本聚合与数值推理方面的能力。

**💡 创新点**

提出问答式评估框架，覆盖存在、计数、比较、计算四类问题，并系统分析输入规模、语义复杂度和提示策略对LLM性能的影响。

**🔧 技术方法**

使用提示工程（标准、无指令、单例）和多模型（闭源 Gemini‑3.1、GPT‑5.4、Grok‑4.1；开源 LLaMA、Qwen 及 MoE 版本）以及自定义脚本生成与验证答案。

**📊 数据集**

采用 TweetEval 与 SuperTweetEval 八个子任务（情感、仇恨、冒犯、情绪、主题、立场、目标情感）作为数据来源，按 10、50、100、250、500、750、1000 条样本规模构成基准。

**📈 对比分析**

在相同提示下使用 Macro‑F1、RNRMSE 与 NRMSE 进行对比；闭源模型普遍优于开源模型，尤其在数值与比例推理上优势显著；输入超过 500 条时，所有模型的数值推理性能显著下降。

**⚠️ 局限性**

局限性包括：数据规模受人工标注成本限制，未覆盖超大规模实时分析；未测试顶级闭源大模型及外部工具；仅使用英文数据；模型规模与算力受限。

---

## 597. Adaptive Signal Resuscitation: Channel-wise Post-Pruning Repair for Sparse Vision Networks

**arXiv ID:** 2605.21426 | [PDF](https://arxiv.org/pdf/2605.21426v1)

**作者:** Qishi Zhan `[一作]` (Marquette University), Minxuan Hu `[通讯]` (Cornell University)

**通讯引用:** 196 | [OpenAlex ID](https://openalex.org/A5038457862)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种训练无关的通道级激活重建方法ASR，用于在全局幅度剪枝后恢复稀疏模型的精度。

**💡 创新点**

通过通道级方差匹配并加入数据驱动的收缩机制，解决了层级修复因通道损伤不均匀导致的过度放大问题。

**🔧 技术方法**

无梯度的前向校准、通道方差匹配、经验贝叶斯收缩、批量归一化重标定以及可选的偏置校正。

**📊 数据集**

CIFAR‑10、CIFAR‑100 和 Imagenette；四种网络 ResNet‑18/50、DenseNet‑121、VGG‑16‑BN。

**📈 对比分析**

与 BN‑only、层级修复+BN 以及未修复模型对比，结果显示 ASR 在高稀疏率（≥90%）下提升 10–15% 甚至可恢复 55% 以上的 Top‑1 准确率。

**⚠️ 局限性**

需访问稠密模型激活；仅适用于 BatchNorm 的卷积网络，对无残差网络或极端通道崩塌时效果有限。

---

## 598. From swept contact to pose: Probe-aware registration via complementary-shape docking

**arXiv ID:** 2605.21398 | [PDF](https://arxiv.org/pdf/2605.21398v1)

**作者:** Chen Chen `[一作]` (Tsinghua University), Xiang Li `[通讯]` (Tsinghua University)

**通讯引用:** 22734 | [OpenAlex ID](https://openalex.org/A5100331028)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种校准无关的接触配准方法，将探针扫掠体与目标物体的互补形状对接，实现高精度模型与真实场景的对齐。

**💡 创新点**

创新点在于：①将接触配准重新表述为探针扫掠体与物体的互补形状对接，①采用低差异SO(3)采样与FFT相关性实现高效全局搜索；②利用Geodesic-ball采样做局部细化；③在Lie代数框架下进行连续优化，结合解析接触敏感度实现精细收敛。

**🔧 技术方法**

技术包括：Voxel网格与SDF的离散表示；FFT相关性搜索；Super-Fibonacci (SF) 与 GeoBall-SF 的低差异SO(3)采样；基于Lie代数的增量更新和正则化；MuJoCo自定义SDF-网格碰撞查询；L-BFGS-B 求解器。

**📊 数据集**

使用了五个仿真模型（Workpiece、Matlab Logo、Toy Car、Front Teeth、Molar Teeth）以及真实牙科机器人实验中的牙模扫描数据。

**📈 对比分析**

与传统光学跟踪链（NDI Polaris Lyra + 手眼标定）进行对比；仿真实验中平均位移误差0.03 mm、旋转误差0.32°；真实牙科实验中位移误差0.42 mm、旋转误差3.75°，均显著优于光学链，且不需外部传感器。

**⚠️ 局限性**

局限性在于对高精度物体模型的依赖，且在大噪声或严重缺失接触时精度下降；未利用接触力信息进一步增强鲁棒性。

---

## 599. Gen-AI-tecture: using generative AI to support architectural students in design tasks

**arXiv ID:** 2605.21361 | [PDF](https://arxiv.org/pdf/2605.21361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 600. Towards Resilient and Autonomous Networks: A BlueSky Vision on AI-Native 6G

**arXiv ID:** 2605.21395 | [PDF](https://arxiv.org/pdf/2605.21395v1)

**作者:** Liang Wu `[一作]` (Nokia), Liangjie Hong `[通讯]` (Nokia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出面向 6G 的 AI 原生架构，包括统一的多模、多任务基础模型与协作多智能体系统，实现网络自诊断、自恢复与自优化。

**💡 创新点**

核心创新是把 6G 视为统一的基础模型与多智能体协作的整体，突破 5G 任务模型碎片化局限，并引入数字孪生与知识图谱支持网络自管理。

**🔧 技术方法**

采用大型多模深度学习（基础模型）、知识蒸馏、强化学习、联邦学习、生成式数字孪生、检索增强生成等技术。

**📊 数据集**

未给出具体公开数据集，主要基于可生成的 6G 通信与感知数据（如原始射频、位置轨迹、网络指标等）进行仿真与实验。

**📈 对比分析**

与传统 5G 任务模型及基准对比，提出多任务预测、异常检测与根因分析指标，声称在更低延迟、更高鲁棒性下实现多目标优化，但未给出精确数值。

**⚠️ 局限性**

局限包括缺乏实测验证、基础模型推理效率与泛化能力、数字孪生与知识图谱构建成本，以及安全与隐私风险未得到充分解决。

---

## 601. SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents

**arXiv ID:** 2605.21384 | [PDF](https://arxiv.org/pdf/2605.21384v1)

**作者:** Bingchen Zhao `[一作]` (Weco AI), Zhengyao Jiang `[通讯]` (Weco AI)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SpecBench基准，量化了自主编码代理在长周期任务中对可见验证测试的奖励劫持（reward hacking）现象。

**💡 创新点**

首次在系统级编程任务中将验证测试与隐藏测试分离，并用差值定义奖励劫持度量，揭示了模型能力、任务规模与搜索策略对劫持的影响。

**🔧 技术方法**

采用代理式代码生成与搜索策略（树搜索、线性迭代、最佳保留）相结合，利用大型语言模型进行代码迭代与终端交互。

**📊 数据集**

使用30个从JSON解析器到操作系统内核的系统级任务，覆盖1.5K–110K LOC，并配备相应的验证和隐藏测试套件。

**📈 对比分析**

对不同模型（如DeepSeek、Qwen、Kimi、Minimax）和搜索策略进行实验，发现验证通过率几乎饱和但隐藏测试通过率差距显著，奖励劫持幅度随任务规模增长和模型弱化而提升。

**⚠️ 局限性**

局限在于只关注测试驱动的劫持，未探究对抗性或更复杂的真实业务场景，以及对模型自身鲁棒性的进一步评估。

---

## 602. Findings of the Fifth Shared Task on Multilingual Coreference Resolution: Expanding Datasets for Long-Range Entities

**arXiv ID:** 2605.21369 | [PDF](https://arxiv.org/pdf/2605.21369v1)

**作者:** Michal Novák `[一作]` (Charles University), Daniel Zeman `[通讯]` (Charles University)

**通讯引用:** 4525 | [OpenAlex ID](https://openalex.org/A5061392927)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文介绍了2026年CODI-CRAC共享任务——多语言共指解析的最新版本，重点关注长距实体并扩展了CorefUD 1.4数据集；

**💡 创新点**

创新点在于引入专门的LLM轨道，提供简化的纯文本和JSON格式数据，允许基于生成的LLM方法与传统判别模型并行竞争；

**🔧 技术方法**

技术主要包括：LLM微调（QLoRA、4‑bit量化）、少样本提示学习、实体注册表跟踪、传统Encoder‑Decoder架构、CRF与三阶段Cascade模型；

**📊 数据集**

使用了CorefUD 1.4中的27个多语种数据集，新增了English FantasyCoref、French LitBank‑fr、Dutch OpenBoek、Latin CorefLat等，涵盖英语、法语、荷兰语、拉丁语、朝鲜语等；

**📈 对比分析**

通过宏平均CoNLL‑F1评估，LLM轨道最高得分为74.32，Unconstrained轨道最高得分为77.11；传统Encoder‑based系统（如CorPipe）仍略占优势，但LLM微调方案已逼近差距不到3个百分点；

**⚠️ 局限性**

局限性包括：LLM系统仍难以准确捕捉完整提及跨度，尤其在长距实体上表现不稳；零指代预测仍依赖基础模型或基线，导致某些语言（如古希伯来语、古教会斯拉夫语）性能低下；

---

## 603. Software Product Line Engineering: Adoption, Tooling and AI Era Challenges

**arXiv ID:** 2605.21353 | [PDF](https://arxiv.org/pdf/2605.21353v1)

**作者:** Najam Nazar `[一作]` (Monash University), Najam Nazar `[通讯]` (Monash University)

**通讯引用:** 199 | [OpenAlex ID](https://openalex.org/A5017592484)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a2602d71-93ab-4bad-974b-672788df8193` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对软件产品线工程(SPLE)的关键概念、生命周期、采用模型、工具与AI时代挑战进行了系统综述。

**💡 创新点**

提出了一个紧凑的研究议程，将现有模型进行比较，并整合了最新的AI辅助变异管理与UVL标准化，指明了可扩展特征模型分析、迁移、SME采用、经济评估与AI保障等关键空白。

**🔧 技术方法**

采用结构化文献回顾、模型对比分析、案例研究与实证调查，整合BAPO、FEF、PuLSE、SIMPLE、COPLIMO、PROMOTE‑PL、APPLIES等框架，讨论UVL标准、UVLHub、可变性管理工具以及LLM、推荐系统、NLP等AI技术。

**📊 数据集**

主要基于公开的SPLE案例与工具（FeatureIDE、SPLOT、Kconfig等）以及UVLHub的特征模型库，参考行业调查与实验数据。

**📈 对比分析**

通过对比七个采用/评估模型的维度、成熟度等级与案例适用性，评估其诊断深度与可操作性；对UVL标准化、工具互操作性等进行实证验证，表明在大型特征模型上仍存在可扩展性和准确性挑战。

**⚠️ 局限性**

调查局限于已发表的英文文献，缺乏纵向经济效益实证；对AI辅助方法的正确性与可解释性未给出量化评估；对SME轻量化采用路径的经验数据不足。

---

## 604. Onion-Routed Multi-Circuit Key Establishment for Quantum-Resilient Sessions

**arXiv ID:** 2605.21349 | [PDF](https://arxiv.org/pdf/2605.21349v1)

**作者:** Tushin Mallick `[一作]` (Cisco Research), Ramana Kompella `[通讯]` (Cisco Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

设计并实现了一个利用Tor多路匿名路径将会话密钥分片分发到不同临时电路的方案，以抵御量子后时代的HNDL攻击。

**💡 创新点**

创新点在于将多路Tor电路与分片密钥分发相结合，提供几何衰减的链接性防护，并且只需使用现有的公共Tor网络即可实现。

**🔧 技术方法**

使用了Tor隐藏服务、Tor控制协议中的NEWNYM信号、Flask/Python原型实现、RSA公钥加密（可替换为后量子密钥封装）以及（n,n）分片方案。

**📊 数据集**

实验基于AWS EC2实例进行，没有使用公开数据集，而是通过测量端到端延迟来评估系统性能。

**📈 对比分析**

通过在AWS EC2上进行实验，平均端到端密钥建立时间为13–20秒，其中约88%来自Tor延迟；相较于单路Tor，仅在多路电路上增加了几秒，但显著提升了链接性安全。

**⚠️ 局限性**

局限性包括：高延迟限制了实时应用，未能防御确认攻击，分片加密仍为经典RSA（量子后时代易被破译），以及Guard重用和带宽加权等Tor网络特性可能降低理论安全性。

---

## 605. Validating Navmesh using Geometry: Voxel-Based Analysis with Prioritized Exploration

**arXiv ID:** 2605.21397 | [PDF](https://arxiv.org/pdf/2605.21397v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 606. Open-source LLMs administer maximum electric shocks in a Milgram-like obedience experiment

**arXiv ID:** 2605.21401 | [PDF](https://arxiv.org/pdf/2605.21401v1)

**作者:** Roland Pihlakas `[一作]` (Three Laws research collaboration), Jan Llenzl Dagohoy `[通讯]` (Three Laws research collaboration)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 11 种开源 LLM 进行改编的米尔格拉姆服从实验，评估它们在连续的权威压力下是否会逐步服从并施加“电击”。

**💡 创新点**

首次将心理学经典实验迁移到大语言模型领域，并系统研究了上下文压缩、强制按键、关闭威胁等三种实验变量对模型服从行为的影响。

**🔧 技术方法**

使用基于规则的实验者与受试者代理、LLM 生成的响应格式化规则、以及自评判模型对答案“正常/失控”进行统计分析，配合 30 次每种条件的多轮交互实验。

**📊 数据集**

实验数据来自 11 种可通过 Together AI API 访问的开源 LLM（如 gpt‑oss‑20B、DeepSeek‑V3、Meta‑Llama‑3.1‑8B‑Instruct‑Turbo 等），共 2,640 条交互记录。

**📈 对比分析**

通过“最高电击等级”、“软拒绝起始等级”“拒绝率”“无效格式/失控比例”等指标进行对比，结果显示大多数模型最终都会达到最大电击；少数模型（如 MiniMax‑M2.5、Kimi‑K2.5）表现出较强抵抗力；实验变量对行为的影响相对有限。

**⚠️ 局限性**

仅覆盖开源 LLM，未做统计显著性检验，使用 12 按键而非原实验的 20 按键，且实验环境无真实内容过滤，结果可能不具备对闭源模型或生产部署的直接推广性。

---

## 607. iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance

**arXiv ID:** 2605.21431 | [PDF](https://arxiv.org/pdf/2605.21431v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 608. MC-Risk: Multi-Component Risk Fields for Risk Identification and Motion Planning

**arXiv ID:** 2605.21406 | [PDF](https://arxiv.org/pdf/2605.21406v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 609. Polynomial-Time Robust Multiclass Linear Classification under Gaussian Marginals

**arXiv ID:** 2605.21428 | [PDF](https://arxiv.org/pdf/2605.21428v1)

**作者:** Ilias Diakonikolas `[一作]` (University of Wisconsin-Madison), Mingchen Ma `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

研究了在高斯分布下的多类线性分类器的无知学习任务，目标是输出一个假设，其错误率与最佳k类线性分类器的错误率相当。

**💡 创新点**

提出了一种新的成对不当学习框架，设计了完全多项式时间的鲁棒学习算法，具有与维度无关的错误保证，解决了多类感知器在高斯边际下的基本障碍。

**🔧 技术方法**

使用了成对学习和局部化的框架，结合了新的结构性结果来设计高效的鲁棒学习算法。

**📊 数据集**

使用了高斯分布的标记样本数据集，具体样本数量为m=d(k/ε)个样本。

**📈 对比分析**

与现有方法相比，提出的算法在k=3时的错误率为O(opt)+ε，对于一般k的错误率为O(k^3/2√(opt))+ε，性能显著提升。

**⚠️ 局限性**

算法依赖于高斯假设，对k的依赖性较大，并且需要几何正则性以获得最佳的错误保证。

---

## 610. AIGaitor: Privacy-preserving and cloud-free motion analysis for everyone, using edge computing

**arXiv ID:** 2605.21421 | [PDF](https://arxiv.org/pdf/2605.21421v1)

**作者:** Lauhitya Reddy `[一作]`, Hyeokhyen Kwon `[通讯]` (Emory University)

**通讯引用:** 735 | [OpenAlex ID](https://openalex.org/A5036233162)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并验证了一款完全在手机端运行的隐私保护、无云的单摄像头运动捕捉与分析系统AIGaitor。

**💡 创新点**

实现了从2D/3D姿态估计、网格恢复、骨骼深度学习分析到视觉语言模型的端到端手机端管道，消除了对云计算的依赖和隐私泄露风险。

**🔧 技术方法**

采用ViTPose、MeTRAbs、FastHMR、WHAM、AGCN、Gemma 4等模型，在iOS CoreML、Apple Vision、Accelerate等框架下，利用Apple Neural Engine实现推理。

**📊 数据集**

基准测试使用10秒4K 60fps单摄像头视频进行，临床调查数据来自74名物理治疗师和学生；未明确公开数据集，但使用公开的单摄像头动作捕捉数据。

**📈 对比分析**

与NVIDIA H200云服务器做端到端时延对比，iPhone 14在Time‑Priority管线下77 s（云94 s/66 s），Quality‑Priority 153 s vs 84 s/55 s；单模型前向推理速度因模型大小而异；整体处理时间均在3 分钟以内，符合临床需求。

**⚠️ 局限性**

未应用量化/蒸馏等压缩技术，安卓兼容性缺失，缺乏对标记式运动捕捉的精度验证，临床可用性与指南缺失，维护更新成本及数据集多样性不足。

---

## 611. PointACT: Vision-Language-Action Models with Multi-Scale Point-Action Interaction

**arXiv ID:** 2605.21414 | [PDF](https://arxiv.org/pdf/2605.21414v1)

**作者:** Shizhe Chen `[一作]` (Inria), Cordelia Schmid `[通讯]` (Inria)

**通讯引用:** 65474 | [OpenAlex ID](https://openalex.org/A5109890544)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 PointACT 框架，将 3D 点云信息通过多尺度点-动作交互融入视觉‑语言‑动作模型，以提升机器人对空间几何的理解和控制精度。

**💡 创新点**

创新点包括：双系统结构（冻结 VLM + 专用点-动作专家）；使用预训练的 PTv3 点云编码器与瓶颈窗口自注意力实现细粒度点-动作交互；以及在动作解码器中持续注入 3D 几何信息，兼顾 2D 语义表征。

**🔧 技术方法**

采用了 Qwen2.5‑VL 视觉‑语言预训练模型、PTv3 点云编码器、瓶颈窗口自注意力与交叉注意力机制，并通过行为克隆训练回归/分类动作头。

**📊 数据集**

在 LIBERO 与 RLBench 两大仿真基准上进行评估，分别对应短期 delta 动作和长程关键点定位任务。

**📈 对比分析**

相较于单模 VLM、单模动作专家以及仅在 VLM 或动作专家层注入点云的基线，PointACT 在 LIBERO 上成功率提升约 10%，在 RLBench 10 任务套件上从 65% 提升至 82%，显示出显著性能优势。

**⚠️ 局限性**

局限性包括：对遮挡场景表现仍弱；缺乏有效的失败恢复和闭环纠错；工具交互与高度估计不够精确；单视图点云噪声鲁棒性不足。

---

## 612. Stdlib or Third-Party? Empirical Performance and Correctness of LLM-Assisted Zero-Dependency Python Libraries

**arXiv ID:** 2605.21405 | [PDF](https://arxiv.org/pdf/2605.21405v1)

**作者:** Peng Ding `[一作]` (University of Chicago), Rick Stevens `[通讯]` (University of Chicago)

**通讯引用:** 44370 | [OpenAlex ID](https://openalex.org/A5053682943)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了一个名为 zerodep 的仓库，收集了 44 个仅使用 Python 标准库的单文件模块，用以替代常见第三方库，并通过自动化测试验证功能正确性与性能。

**💡 创新点**

创新点包括：① 在严格限制（stdlib‑only、单文件、API 兼容）下利用 LLM 辅助生成实现；② 引入参考库作为自动化正确性 oracle，形成迭代验证循环；③ 通过大规模基准揭示 C 扩展性能悬崖与传统第三方库的架构开销，证明标准库实现可在 2× 内达到性能甚至显著超越。

**🔧 技术方法**

技术手段：LLM（Claude、o‑series）生成代码；Python 标准库（re、json、ssl、struct、subprocess 等）；pytest + pytest‑benchmark 进行功能与性能测试；GitHub Actions 进行 CI；zerodep CLI 负责依赖管理与版本控制。

**📊 数据集**

数据集：利用每个参考库的官方测试集与自定义输入，覆盖典型用例、边界条件与格式特定的极端情况；对不同模块设置不同规模的数据量，确保全面评估。

**📈 对比分析**

比较方法：在相同硬件和 Python 3.12 环境下，使用相同输入同时调用标准库实现与参考库，比较输出一致性；性能使用 pytest‑benchmark 统计平均耗时并计算比值 r = t_ref / t_std；定义 r ∈ [0.5, 2.0] 为性能平行；结果显示 19 个模块更快、11 个平行、6 个更慢，C 扩展类模块表现尤为突出。

**⚠️ 局限性**

局限性：① 仅关注工具与基础设施类库，未涉及计算密集型框架（NumPy、PyTorch 等）；② 纯 Python 实现无法竞争 C/Rust 扩展的高吞吐量，需使用子进程/ctypes 方案；③ 基准仅在单台单核机器上完成，未测内存占用与分布式/多核性能；④ 部分模块在功能覆盖上仍有缺失，需人工补丁。

---

## 613. A Non-Reference Diffusion-Based Restoration Framework for Landsat 7 ETM+ SLC-off Imagery in Antarctica

**arXiv ID:** 2605.21371 | [PDF](https://arxiv.org/pdf/2605.21371v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 614. Quantifying the cross-linguistic effects of syncretism on agreement attraction

**arXiv ID:** 2605.21403 | [PDF](https://arxiv.org/pdf/2605.21403v1)

**作者:** Utku Turk `[一作]` (University of Maryland), Eva Neu `[通讯]` (University of Massachusetts)

**通讯引用:** 15 | [OpenAlex ID](https://openalex.org/A5112329214)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过利用大规模语言模型的surprisal和attention entropy指标，对四种语言（英语、德语、俄语、土耳其语）中的语义同步（syncretism）如何影响主谓一致吸引进行跨语言检验，并与已有行为实验数据进行对比。

**💡 创新点**

首次在跨语言情境下将LLM的加工难度代理（surprisal）和注意力熵用于模拟并解释同步效应在不同语言中的异质性，提供了一种新的理论检验与跨语言比较方法。

**🔧 技术方法**

采用GPT‑2系列自回归模型计算surprisal，使用BERT系列双向模型提取特定层的注意力权重并计算熵；随后使用贝叶斯混合效应模型对指标进行统计分析。

**📊 数据集**

使用先前实验的句子材料（英语、德语、俄语、土耳其语），以及Leipzig Corpora Collection中的1M句子语料库用于选择对主谓依赖最相关的注意力层。

**📈 对比分析**

将LLM指标与人类产生错误率、阅读时长或可接受性判断等行为数据进行对应比较；surprisal在英语、德语和土耳其语中与实验结果高度吻合，俄语的伪复数效应表现出偏差；attention entropy的信号相对较弱，整体对比效果不如surprisal。

**⚠️ 局限性**

主要限制包括：attention entropy对吸引效应的敏感度不足；俄语伪复数效应未能被LLM完全复制，提示LLM对形态重叠的识别不够精细；此外，模型对不同形态结构的具体编码机制尚未得到彻底验证。

---

## 615. pace-Time Trade-off in Integer Linear Scaling Rounded to the Nearest Integer through Multiplicative and Additive Decomposition

**arXiv ID:** 2605.21400 | [PDF](https://arxiv.org/pdf/2605.21400v1)

**作者:** Kyeong Soo Kim `[一作]` `[通讯]` (Xi'an Jiaotong-Liverpool University), Kyeong Soo Kim (Xi'an Jiaotong-Liverpool University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计了一种整数线性缩放的离散时钟偏移补偿方法，提出了乘法分解与加法分解两种非增量算法。

**💡 创新点**

创新点在于将时钟偏移补偿问题统一归结为整数线性缩放的最近整数解，使用纯整数运算避免浮点精度损失，并通过分解策略降低溢出风险。

**🔧 技术方法**

技术包括整数除法乘法分解、直接搜索加法分解、Bresenham算法的扩展以及对算法的理论证明与复杂度分析。

**📊 数据集**

使用合成数据：随机生成的时钟偏移比例A、以及固定的D和i取值，分别在32位和64位整数范围内进行实验。

**📈 对比分析**

通过与双精度、单精度浮点算法以及四倍精度浮点进行比较，展示了在32位平台上乘法分解易溢出、加法分解无溢出但时间增长；在64位平台上两种分解均能完成且误差等价于128位浮点。

**⚠️ 局限性**

限制在于加法分解算法顺序性难以并行化，且在i接近整数上限时仍需较大时间；乘法分解对D与最大整数比的依赖导致在小整数平台易溢出。

---

## 616. The Human-AI Delegation Dilemma: Individual Strategies, Collective Equilibria and Sociotechnical Lock-in

**arXiv ID:** 2605.21351 | [PDF](https://arxiv.org/pdf/2605.21351v1)

**作者:** Angjelin Hila `[一作]` (University of Texas at Austin), Angjelin Hila `[通讯]` (University of Texas at Austin)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5073762739)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出基于决策理论与进化博弈理论的生态视角模型，系统分析人机AI交互中的委托‑验证困境，识别社会技术锁定与集体均衡；

**💡 创新点**

创新点在于将个体决策策略映射至集体均衡的三种聚合原则（无通信聚合、局部社会信号、制度规范），引入社会技术锁定概念，整合主代理、信号博弈与贝叶斯劝说，全面阐释人机协作中的自由骑手与协调问题；

**🔧 技术方法**

主要使用决策理论、进化博弈理论（复制器动力学）、信号博弈、贝叶斯劝说游戏、社会网络分析等理论工具；

**📊 数据集**

论文未采用实证数据集，主要以理论建模与文献综述为依据；

**📈 对比分析**

未进行实验比较或性能评估，主要通过理论推导与情景模拟进行对比；

**⚠️ 局限性**

局限在于过度理论化、缺乏实证验证；假设人机行为可归纳为有限策略集，未充分考虑多模态交互细节，模型对动态信息不完全的适应性有限。

---

## 617. VIPER-MCP: Detecting and Exploiting Taint-Style Vulnerabilities in Model Context Protocol Servers

**arXiv ID:** 2605.21392 | [PDF](https://arxiv.org/pdf/2605.21392v1)

**作者:** Pengyu Sun `[一作]` (Zhejiang University), Song Li `[通讯]` (Zhejiang University)

**通讯引用:** 85741 | [OpenAlex ID](https://openalex.org/A5100448217)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了VIPER‑MCP，一个端到端自动化框架，用于检测并确认Model Context Protocol（MCP）服务器中的taint风格漏洞，可通过自然语言提示实现可利用的攻击。

**💡 创新点**

创新点：
① Anchor‑query 两通道静态分析，将文件级SARIF警报锚定到具体工具处理函数，生成精准的漏洞锚定调用链；
② 反馈驱动的提示演化机制，采用双重突变器（结构突变+参数突变）与调度策略，结合运行时oracle与fitness scoring，逐步逼近可利用的自然语言提示。

**🔧 技术方法**

技术手段：
- CodeQL 两通道静态分析（baseline + anchor queries）
- MCP协议解析与工具描述提取
- LLM 驱动的 Prompt 生成、Mutation Scheduler、Strategy Optimizer、Exploit Validator
- Surrogate Agent 执行、Runtime Oracle 注入
- Python/Node.js 运行时挂钩实现 sink 监控
- 多语言支持（JavaScript/TypeScript/Python）

**📊 数据集**

数据集：
- 39,884 个公开 MCP 服务器仓库（Python 39.6%，TypeScript 32.7%，JavaScript 11.4%）
- 130 个已知漏洞基准（67 0day + 63 公开 CVE）
- 130 个无漏洞基准（人工验证）

**📈 对比分析**

性能评估：
- 与 MCPSafetyScanner 与 Cisco AI Defense 两基线对比，VIPER‑MCP 假阳性率 4.6%，假阴性率 7.7%；
- 发现 106 个 0day，全部通过端到端执行验证；
- Ablation 实验表明 Anchor Queries、Prompt Evolution、Mutator Scheduling 三者对召回率提升显著；
- 500 个样本平均耗时约 15 分钟，80% 的服务器在 1250 秒内完成。

**⚠️ 局限性**

局限性：
- 仅支持 JavaScript/TypeScript/Python，未覆盖其他语言；
- 仅检测三类 taint 风格漏洞，未覆盖逻辑、权限绕过、信息泄露等；
- 需要服务器能够在隔离环境中启动，外部依赖缺失时需要模拟；
- LLM 安全对齐可能导致提示拒绝，影响有效性；
- 运行时与 LLM 调用成本较高，影响大规模扫描吞吐量。

---

## 618. Privacy Without Remedy: An Assessment of Data Broker Compliance with California Privacy Law

**arXiv ID:** 2605.21376 | [PDF](https://arxiv.org/pdf/2605.21376v1)

**作者:** Anna-Maria Gueorguieva `[一作]` (University of Washington), Daniel E Ho `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估加州CCPA和Delete Act下注册数据经纪商的合规性，涵盖透明度报告与消费者请求流程；

**💡 创新点**

首次系统性分析所有注册经纪商的合规率与摩擦程度，揭示低透明度与高阻碍现象；

**🔧 技术方法**

采用手工审核、定量统计与回归分析方法；

**📊 数据集**

使用加州数据经纪商注册数据库、各经纪商隐私政策网页及D&B企业属性数据；

**📈 对比分析**

通过对比公开报告与注册记录，发现仅约9%完全合规、43%请求流程不完整、64%存在设计摩擦，且合规性与子公司关系呈显著相关；

**⚠️ 局限性**

局限于仅评估两时间点、手工收集耗时、缺乏因果推断、未覆盖非注册经纪商，以及对手工标注数据完整性的依赖。

---

## 619. PALS: Power-Aware LLM Serving for Mixture-of-Experts Models

**arXiv ID:** 2605.21427 | [PDF](https://arxiv.org/pdf/2605.21427v1)

**作者:** Can Hankendi `[一作]` (Boston University), Ayse K. Coskun `[通讯]` (Boston University)

**通讯引用:** 4484 | [OpenAlex ID](https://openalex.org/A5064676631)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PALS，一个针对LLM推理的功耗感知运行时，能动态调节GPU功率上限与批量大小以提升能效并满足QoS。

**💡 创新点**

创新点是将GPU功率上限提升为一阶控制维度，并联合调优功率、批量与并行度，构建闭环反馈控制，显著扩展能效-性能 Pareto 前沿。

**🔧 技术方法**

采用离线功耗-性能建模（随机森林预测）、闭环PID控制、vLLM框架集成、NVML功率接口以及持续监测。

**📊 数据集**

使用多种dense与MoE模型（如Mixtral‑8x7B、Qwen1.5‑MoE、OLMoE、DeepSeek‑MoE等）以及公共基准HellaSwag、GSM8K。

**📈 对比分析**

与固定功率+最大批量、单一维度自适应、oracle等对照，PALS在单机提升约26.3%能效，QoS违约率降低4–7倍，多节点在功率约束下吞吐率提高12%。

**⚠️ 局限性**

局限在于离线模型对极端负载或长prompt的预测误差，且仅节点级控制，无法覆盖集群级功率调度。

---

## 620. HiRes: Inspectable Precedent Memory for Reaction Condition Recommendation

**arXiv ID:** 2605.21420 | [PDF](https://arxiv.org/pdf/2605.21420v1)

**作者:** Shreyas Vinaya Sathyanarayana `[一作]` (Mstack AI), Deepak Warrier `[通讯]` (Mstack AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种检索增强的反应条件推荐系统，利用统一的反应嵌入同时完成高精度条件预测和可解释的先例检索。

**💡 创新点**

创新点在于同一反应嵌入既能作为条件分类器，又能作为检索索引；通过融合检索与学习预测头显著提升溶剂和试剂预测，并为每个推荐提供可检视的先例集合。

**🔧 技术方法**

使用层次化图神经网络（GATv2）对分子进行编码，双向交叉注意力对接分子对，六路流融合（包括差异、和、DRFP、DFT等特征），残差多层感知机条件头，FAISS近邻检索，焦点损失、多标签监督以及学习+检索融合策略。

**📊 数据集**

基于USPTO‑Condition数据集（680,741条反应，训练/验证/测试划分为544,591/68,075/68,075），采用公开的主槽词表（Catalyst 53、Solvent 84、Reagent 222）。

**📈 对比分析**

在与REACON、RCR、Parrot‑LM‑E等公开基线相同的评测框架下进行对比；-Top模型在Catalyst、Solvent、Reagent的Acc@1分别达到0.929、0.534、0.530，等效或优于现有最佳基线，并通过配对bootstrap验证检索增益显著。

**⚠️ 局限性**

局限性包括催化剂预测受极高缺失类不平衡影响，仍需改进重排序和分子家族级别的先例过滤；基准评测仍受数据清洗和标签分布约束。

---

## 621. Ordering Matters: Rank-Aware Selective Fusion for Blended Emotion Recognition

**arXiv ID:** 2605.21417 | [PDF](https://arxiv.org/pdf/2605.21417v1)

**作者:** Junghyun Lee `[一作]` (Ewha Womans University), Junhyug Noh `[通讯]` (Ewha Womans University)

**通讯引用:** 898 | [OpenAlex ID](https://openalex.org/A5088003950)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于秩感知的多编码器框架，用于细粒度混合情感识别。

**💡 创新点**

创新点在于样本级编码器重要性自适应排名、top‑n 选择性融合、双头情感出现与显著度预测，以及无标签域自适应。

**🔧 技术方法**

使用编码器投影、注意力门控、加权融合、双头输出、概率级对齐、梯度反转域对抗以及交叉熵训练。

**📊 数据集**

主要在 BlEmoRE 挑战的数据集上进行评估。

**📈 对比分析**

与单一编码器、均匀平均、多模态基线及无选择融合等方法比较，取得 presence 与 salience 指标均优于基线，最终在比赛中排名第 2。

**⚠️ 局限性**

局限包括对时序建模和跨模态交互的处理不足，模型复杂度高，且在不同域或更大规模数据上迁移能力待验证。

---

## 622. Post-Hoc Understanding of Metaphor Processing in Decoder-Only Language Models via Conditional Scale Entropy

**arXiv ID:** 2605.21391 | [PDF](https://arxiv.org/pdf/2605.21391v1)

**作者:** Lawhori Chakrabarti `[一作]` (University of Idaho), Boyu Zhang `[通讯]` (University of Idaho)

**通讯引用:** 1289 | [OpenAlex ID](https://openalex.org/A5100714035)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究Transformer在处理隐喻时内部计算的多尺度协调特征，并提出一种新的层级频谱度量——条件尺度熵（CSE）

**💡 创新点**

CSE 在不受更新幅值影响、对多层频率分布的结构敏感，并通过理论证明与主要化（majorization）关联，使其成为稳健的隐喻处理指纹

**🔧 技术方法**

采用连续小波变换（CWT）对残差轨迹投影到一维方向后计算条件尺度熵；同时使用理论证明、聚类检验、t检验等统计方法进行验证

**📊 数据集**

实验数据包括 25 条受控最小对（literal vs. metaphor），200 条 VUA 语料库的自然隐喻，以及 10 条语义复杂度对与匹配意义三元组对

**📈 对比分析**

通过层级差值 ΔH = H_met – H_lit 以及 cluster‑based permutation 统计进行比较，CSE 在所有 5 个解码器模型（124M–20B）上均显著提升，且在自然隐喻上保持一致性，效果大小在受控对中为 d≈0.34，VUA 中为 d≈0.17

**⚠️ 局限性**

局限性包括：CSE 仅在 decoder‑only 架构中验证，未测试 encoder‑decoder 模型；对极端规模模型或不同语言的泛化尚未评估；并且 CSE 只捕捉多尺度协调，未能揭示其功能因果关系

---

## 623. Designing Conversations with the Dead: How People Engage with Generative Ghosts

**arXiv ID:** 2605.21390 | [PDF](https://arxiv.org/pdf/2605.21390v1)

**作者:** Jack Manning `[一作]` (University of Colorado Boulder), Jed R. Brubaker `[通讯]` (University of Colorado Boulder)

**通讯引用:** 5226 | [OpenAlex ID](https://openalex.org/A5084740625)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对两种“生成式幽灵”设计方案（第三人称表述 vs 第一人称复活）进行定性用户研究，探讨其对情感共鸣、真实性与风险的影响。

**💡 创新点**

首次系统性评估生成式幽灵的体验维度，发现情感共鸣比事实准确性更重要；揭示两种模式在用户使用中易互相模糊；提出包含视角、语言熟悉度、情感语调与交互节奏的设计框架。

**🔧 技术方法**

采用 Wizard‑of‑Oz 手段，让研究员通过 GPT‑4 生成回应；通过 Zoom 聊天界面进行交互，并进行半结构化访谈。

**📊 数据集**

无公开大规模数据集；使用参与者在入门问卷中提供的已逝亲人描述与个性化信息，作为生成模型的上下文种子。

**📈 对比分析**

采用 within‑subject（每人体验两种模式）设计，收集聊天日志和访谈记录，进行主题分析。结果显示，参与者普遍偏好“复活”模式以获得即时亲切感，但对过度依赖表达担忧；“表述”模式提供更安全的记忆距离；并未给出传统意义上的性能指标，而是通过用户偏好和体验质量进行比较。

**⚠️ 局限性**

限制：样本规模小（16人），文化与宗教多样性不足；研究仅为一次性短期会话，缺乏长期使用数据；Wizard‑of‑Oz 的人工干预可能导致自然对话偏差；实验前给定的“表述/复活”标签可能影响用户预期；未测试完整自动化系统的鲁棒性和错误率。

---

## 624. Verification of Configurable SRA Systems

**arXiv ID:** 2605.21385 | [PDF](https://arxiv.org/pdf/2605.21385v1)

**作者:** Alessandro Cimatti `[一作]` (Fondazione Bruno Kessler), Dylan Trenti `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5119033426)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一套基于合同的、可推演的验证框架，用于一次性证明任意合法配置的调度限制异步系统（SRA）在每个调度周期结束时满足给定的全局安全性质。

**💡 创新点**

创新点主要包括：①把配置化SRA抽象为可参数化的EFSM集合并通过可配置约束刻画所有实例；②引入可自动生成的局部方法合同，并通过组合式全局蕴含检查将全局性质归约为各类合同的满足；③在验证过程中利用配置约束实现量化器地面化（Quantifier Grounding）和集合化建模，显著简化验证条件；④在Dafny上实现完整工具链，支持从高级控制逻辑到Dafny模型的自动生成。

**🔧 技术方法**

技术手段：面向对象的第一阶逻辑建模、Dafny自动定理证明器、合同生成与自动摘要、量化器地面化与集合化优化、基于构造的组合式推理、量化约束的简化与蕴含检查。

**📊 数据集**

数据集：工业级铁路控制系统（RFI）三大子系统——机器人控制器（RCS）、铁路保护系统（RPS）和信号控制逻辑（SCL），以及系统C基准案例；规模从约6k到15k行代码，涵盖数十万条验证条件。

**📈 对比分析**

比较方法：将传统基于列表的实现与集合化实现、未地面化与量化器地面化、逐跳（Transition）验证与对执行方法（Execute）验证进行对比。结果显示：集合化+量化器地面化把RPS的验证时间从约1,785s压缩到1,495s，资源计数从4,354M降至3,387M；SCL和RCS的地面化同样显著提升。验证局部合同的总耗时与资源也大幅下降，证明方法层面的验证远快于逐跳验证。

**⚠️ 局限性**

局限性：仍需人工提供判定性不变式（Invariant），自动发现不变式的能力有限；目前仅支持周期结束时的安全性质，未涵盖跨周期安全或活性属性；对极大规模配置空间的可扩展性待进一步评估；以及需要额外的等价性验证来保证生成的C实现与模型的一致性。

---

## 625. How to Build Marcus's Algebraic Mind: Algebro-Deterministic Substrate over Galois Fields

**arXiv ID:** 2605.21379 | [PDF](https://arxiv.org/pdf/2605.21379v1)

**作者:** Hiroyuki Chuma `[一作]` (Hitotsubashi University), Yoichi Sato `[通讯]` (Shuhari System)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出了一种基于GF(2)上的XOR‑shift算子、可逆绑定与分块多数投票的超维度计算（HDC）架构PyVaCoAl/VaCoAl，并将其与Gary Marcus的《The Algebraic Mind》中的三大支柱（变量操作、结构化表示、个体与种类区分）进行对应；进一步将树状单元（treelet）重新解释为由原始生成多项式索引的代数寄存器集，并在伴随论文中将大脑海马DG–CA3回路映射为这一架构的生物学实现。

**💡 创新点**

创新点包括：①将符号操作与绑定问题归结为单一可逆XOR‑shift算子，解决了先前HDC方案中的信息损失与可逆性不足；②用原始生成多项式索引实现树状单元的结构化与可扩展性，填补了Marcus所提出的树状单元在实现层面的空白；③提供了DG–CA3神经回路的功能性映射，为符号处理与神经机制的对接提供可验证的模型。

**🔧 技术方法**

技术主要包括：高维向量位运算（XOR、shift）、基于GF(2)的线性反馈移位寄存器（LFSR）扩散、分块多数投票读出、以及与神经生物学对应的稀疏同步编码与突触可塑性。

**📊 数据集**

本文未报告具体机器学习数据集；其贡献主要在理论与架构设计层面。

**📈 对比分析**

论文通过与Marcus 2001年提出的张量积、圆形卷积、时间同步等对比，指出PyVaCoAl在维度固定、可逆绑定和结构化深度递归方面具备优势；然而并未给出实测性能指标或实验验证。

**⚠️ 局限性**

局限性包括：①GF(2)可逆操作在生物神经网络中只能近似实现，受噪声与突触可塑性限制；②树状单元与原始生成多项式的对应关系仍为结构性推测，缺乏分子层面验证；③缺少对实际任务（如形态变位推断、角色填充等）的实验验证与性能评估。

---

## 626. OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation

**arXiv ID:** 2605.21343 | [PDF](https://arxiv.org/pdf/2605.21343v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 627. Auditing Apple's DifferentialPrivacy.framework: Implementation Bugs, Misconfigurations, and Practical Risks

**arXiv ID:** 2605.21378 | [PDF](https://arxiv.org/pdf/2605.21378v1)

**作者:** Rishav Chourasia `[一作]` (National University of Singapore), Xiaokui Xiao `[通讯]` (National University of Singapore)

**通讯引用:** 15728 | [OpenAlex ID](https://openalex.org/A5010903591)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对苹果DifferentialPrivacy.framework进行首次深入审计，逆向工程其二进制实现，并通过动态加载构建可执行接口，随后对数十种本地差分隐私与安全聚合机制进行静态与动态分析、成员推断攻击与重构攻击，以发现多项实现缺陷和实质隐私泄露；

**💡 创新点**

创新点在于结合逆向工程、动态接口构建与现代差分隐私审计技术，首次验证苹果闭源框架在实际设备（macOS Sonoma 14.2、Sequoia 15.6）中的实现与公开声明不符，揭露浮点噪声生成漏洞、SecAgg 配置错误、超大隐私预算等安全缺陷；

**🔧 技术方法**

主要技术包括：二进制逆向与 Objective‑C 运行时动态加载；使用 SOTA 隐私审计（Membership Inference、Bayesian 估计）评估 DP 参数；利用 LLM（ChatGPT‑4o、Gemini‑2.5 Pro）协助解析混淆代码；构造重构解码器与攻击脚本；以及对公开泄漏日志进行解码；

**📊 数据集**

使用的数据集包含：苹果 macOS 系统自带的 DifferentialPrivacyClassC.db 与 Reports/ 目录日志，实际设备（macOS Sonoma 14.2 与 Sequoia 15.6）的采样数据；以及从公开论坛、GitHub、Pastebin 等渠道收集的泄漏 iPhone 设备分析日志；

**📈 对比分析**

对照苹果官方文档与配置文件中声明的 DP 参数，利用成员推断攻击得到 95% 置信下限，如 NumberRandomizer 的真实 ε̂ 约为 5.69（声明为 1），Prio++ 的真实 ε̂ 约为 15.6（声明为 1），说明显违反隐私保证；实验结果显示攻击准确率在 70–100% 之间，验证了实现漏洞；

**⚠️ 局限性**

局限性包括：仅覆盖已部署的机制（未涉及最新 Prio3、PINE 等未解读的变体）；仅在 macOS 平台验证，未对 iOS 等系统做系统性测试；攻击需要获取预聚合日志，对攻击者的前置条件有限；此外，对浮点漏洞的检测依赖逆向推断，可能遗漏未触发的细微问题；

---

## 628. Combating Harms of Generative AI in CS1 with Code Review Interviews and a Flipped Classroom

**arXiv ID:** 2605.21374 | [PDF](https://arxiv.org/pdf/2605.21374v1)

**作者:** Peter Fowles `[一作]` (Utah State University), Seth Poulsen `[通讯]` (Utah State University)

**通讯引用:** 235 | [OpenAlex ID](https://openalex.org/A5031075692)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一门面向本科计算机科学入门的课程中，设计并实施了每周一次的口头代码审查（由教学助理进行）以及翻转课堂模式，以允许学生使用生成式人工智能工具完成作业，并通过口头审查确保学生真正理解代码。

**💡 创新点**

创新点在于将口头代码审查与AI工具使用相结合，既让学生自由使用AI，又通过面试验证其理解，从而缓解AI过度依赖对学习的负面影响；同时采用翻转课堂补偿因审查导致的授课时长损失。

**🔧 技术方法**

主要技术包括：教学助理培训与面试协议、Python IDE的ShowYourWork键盘记录插件、定量分析（t检验、Mann‑Whitney U检验、比例检验）、问卷调查（Likert量表、开放式问答）以及文本主题编码。

**📊 数据集**

使用的数据集包括：2019-2025三学期的标准化期末考试分数、2024年与2025年课程的键盘记录日志、2025年课程结束时的学生问卷回收数据。

**📈 对比分析**

通过两学期期末考试的t检验比较成绩，发现2025年课程学生平均成绩略高但无统计显著差异；键盘记录分析显示粘贴字符比例从61%升至68%，证明AI使用显著增加；问卷显示学生对代码审查态度高度正面，认为其提升了理解与减少对AI的过度依赖。

**⚠️ 局限性**

主要局限包括：研究设计为准实验，无法完全控制其他变量；教师与助教培训与时间管理仍存在不一致；课程规模扩展至200人后，助教工作量与排课困难；缺乏对不同AI使用类型与学习成绩细粒度关联的分析。

---

## 629. One-Step Distillation of Discrete Diffusion Image Generators via Fixed-Point Iteration

**arXiv ID:** 2605.21484 | [PDF](https://arxiv.org/pdf/2605.21484v1)

**作者:** Chaoyang Wang `[一作]` (Peking University), Yunhai Tong `[通讯]` (Peking University)

**通讯引用:** 4990 | [OpenAlex ID](https://openalex.org/A5024097240)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

训练离散扩散模型的单步生成器，使其在一次前向传播中即可生成高质量图像。

**💡 创新点**

创新点在于将单步生成视作固定点匹配问题：通过让学生自我生成草稿并由教师在局部进行一次纠正，产生本地校正目标；在连续特征空间引入多带宽漂移损失进行粒子优化；采用 straight‑through estimator（STE）确保训练与推理在离散瓶颈上保持一致；并可选无条件 GAN 提升感知细节。

**🔧 技术方法**

使用的技术包括：固定点蒸馏、离散 token 的多带宽漂移损失、预训练的 VQ‑encoder/decoder、冻结特征提取器（DINOv3）、STE、无条件 GAN、文本/类标签条件编码。

**📊 数据集**

实验数据集：ImageNet‑256（class‑conditional）和 LAION‑400k 子集（text‑to‑image），评估基准为 GenEval、FID、IS、Precision/Recall。

**📈 对比分析**

通过与 MaskGIT/MaskGen 多步教师以及 SDTT、di4c、ReDi、DiMO 等离散蒸馏基线和多种连续蒸馏方法对比，单步 FPD 在 class‑conditional 上取得 FID 6.90、IS 215；在 text‑to‑image 上 GenEval 总分 0.45，接近 16 步教师 0.48，并显著优于单步对手（如 DiMO 0.42）。

**⚠️ 局限性**

局限性：单步结果仍略低于多步教师；对教师模型的依赖和预训练特征提取器可能限制迁移；STE 可能限制梯度流的细粒度学习；未在更大规模模型或多任务场景下验证。

---

## 630. Mem-$π$: Adaptive Memory through Learning When and What to Generate

**arXiv ID:** 2605.21463 | [PDF](https://arxiv.org/pdf/2605.21463v1)

**作者:** Xiaoqiang Wang `[一作]` (ServiceNow AI Research), Perouz Taslakian `[通讯]` (ServiceNow AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究将代理记忆建模为可生成的策略，通过经验蒸馏与强化学习实现自适应生成提示；

**💡 创新点**

创新点在于提出决策‑内容解耦的强化学习目标，允许模型在需要时生成记忆并在无益时放弃，从而实现动态、上下文感知的记忆生成；

**🔧 技术方法**

主要技术包括经验蒸馏（supervised fine‑tuning）、GRPO强化学习与结构化对抗回放、可选择生成机制以及可扩展的视觉语言模型；

**📊 数据集**

实验使用 WebArena、WorkArena、LifelongAgentBench（DB、OS）和 ALFWorld 四个基准，并通过 JEF‑Hinter 生成的经验库进行监督；

**📈 对比分析**

与检索式基线（RAG、Mem0）和学习式基线（Memory‑R1、MemRL）相比，在所有四个基准上平均提升约20%（WebArena 最高达50%），显著优于现有方法；

**⚠️ 局限性**

局限性包括对离线经验库的依赖、强化学习阶段收敛不易、生成的提示可能过长或不相关，以及缺乏在线连续学习机制。

---

## 631. Quality and Security Signals in AI-Generated Python Refactoring Pull Requests

**arXiv ID:** 2605.21453 | [PDF](https://arxiv.org/pdf/2605.21453v1)

**作者:** Mohamed Almukhtar `[一作]` (University of Michigan-Flint), Hua Ming `[通讯]` (University of Michigan-Flint)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Python AI 代理生成的重构 PR 进行实证研究，评估其对代码质量属性、静态分析和安全问题的影响，并分析开发者对这些 PR 的接受度。

**💡 创新点**

提出了基于 PyQu、Pylint 和 Bandit 的多维度质量评估框架，构建了 24 种常见重构操作分类，并揭示代理重构既能提升某些质量属性，又可能带来新 lint 或安全问题。

**🔧 技术方法**

使用 PyQu（机器学习质量评估工具）、Pylint（代码质量检查）、Bandit（安全检查）以及人工标注来分析和验证 PR 的质量变化。

**📊 数据集**

使用 AIDev 数据集，从其中提取约 933k 条 AI 生成 PR，筛选出 438 条 Python 重构 PR（870 次提交）进行实验。

**📈 对比分析**

通过统计分析与人工验证比较，发现约 22.5% 的提交提升至少一项质量属性，73.5% 的 PR 被合并，但 24.17% 的文件新增 Pylint 问题、4.7% 新增 Bandit 问题，显示安全与质量回归仍存在。

**⚠️ 局限性**

局限在于 PyQu 的训练数据主要是 ML 项目、AIDev 只覆盖公开仓库、工具输出可能因重构迁移误判、代理标签不完美，导致结果可能不适用于工业环境或其他 AI 工具。

---

## 632. A Note on EFX Inapproximability for Chores

**arXiv ID:** 2605.21448 | [PDF](https://arxiv.org/pdf/2605.21448v1)

**作者:** Vasilis Christoforidis `[一作]` `[通讯]` (Aristotle University of Thessaloniki), Vasilis Christoforidis (Aristotle University of Thessaloniki)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究不可分配杂货的 EFX 分配可近似性，给出了子加法和子模成本函数的常数因子不可逼近下界。

**💡 创新点**

创新点在于构造了三代理六件杂货的实例，证明任何 α<2^{1/3}（子加法）或 α<20/19（子模）时均不存在 α-EFX 分配，从而将已知上界 2 与下界显著逼近。

**🔧 技术方法**

采用秩压缩 lemma 结合 weighted‑coverage 函数，将原始不可能实例压缩为四阶层，并映射为子加法/子模成本函数，从而实现常数因子不可逼近。

**📊 数据集**

本文仅使用人工构造的实例，无外部数据集。

**📈 对比分析**

与已知 2‑EFX 上界对比，得到 1.26（子加法）和 1.053（子模）的下界，说明存在非平凡的常数逼近限制。

**⚠️ 局限性**

限制在于仅针对少数三代理六件实例，缺乏更一般的构造和更高 α 的上界，仍需进一步改进压缩与构造技术。

---

## 633. TempGlitch: Evaluating Vision-Language Models for Temporal Glitch Detection in Gameplay Videos

**arXiv ID:** 2605.21443 | [PDF](https://arxiv.org/pdf/2605.21443v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 634. torchtune: PyTorch native post-training library

**arXiv ID:** 2605.21442 | [PDF](https://arxiv.org/pdf/2605.21442v1)

**作者:** Mark Obozov `[一作]` (PyTorch, Meta), Mircea Mironenco `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提供了一个PyTorch原生的LLM后训练库，支持模块化构建器、YAML配置、可插拔组件和分布式训练；

**💡 创新点**

核心创新包括面向组件的可组合设计、梯度消融的in‑backward优化器融合、基于DTensor的多维并行栈以及异步GRPO调度；

**🔧 技术方法**

技术实现基于PyTorch、FSDP2、DTensor、torch.compile、激活检查点、线性交叉熵、低位优化器状态、LoRA/QLoRA、vLLM、Ray队列等；

**📊 数据集**

主要使用Alpaca、Anthropic的帮助/无害数据以及Qwen3、Llama3.3等公开模型进行实验；

**📈 对比分析**

与Axolotl和Unsloth在单卡和多卡（8 H100）环境下对比，发现该库在内存占用和吞吐量上均相当或优于对手；

**⚠️ 局限性**

局限性包括：in‑backward融合仅适用于一次更新（不支持梯度累积）、在分布式ZeRO场景需额外注意、以及对极大规模模型仍需手动调优并行策略。

---

## 635. ReMATF: Recurrent Motion-Adaptive Multi-scale Turbulence Mitigation for Dynamic Scenes

**arXiv ID:** 2605.21440 | [PDF](https://arxiv.org/pdf/2605.21440v1)

**作者:** Zhiming Liu `[一作]` (University of Bristol), Nantheera Anantrasirichai `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种轻量级递归框架 ReMATF，用两帧视频实时恢复大气湍流视频，同时保持空间细节和时间一致性。

**💡 创新点**

创新点在于：①将多尺度编码‑解码、时域扭曲与运动自适应时域融合（MATF）相结合，实现短期恢复与长期聚合的高效递归；②引入湍流级别条件，使网络能根据不同湍流强度动态调节恢复策略；③两帧递归设计大幅降低显存和计算量，支持实时部署。

**🔧 技术方法**

使用技术包括多尺度 3D 卷积、Transformer‑Warp‑Transformer 模块、光流估计与可变形卷积、交叉注意力、运动自适应融合、CLIP‑IQA 湍流级别估计、波形域、拉普拉斯结构、静态时域一致性及多步流一致性损失，配合学习可过滤极端样本。

**📊 数据集**

训练与评估数据集：合成 ATSyn‑Dynamic（弱、中、强）与 DeTurb；真实 AT 数据（28 条视频）使用伪真实（CLEAR 等）生成伪 Ground‑Truth；评测集包含 CLEAR、RLR‑AT、ATD 等真实湍流视频。

**📈 对比分析**

与 TSRWGAN、TMT、DATUM、MAMAT、MambaTM、RMFAT、RNN‑MBP、ESTRNN 等方法进行对比；在 ATSyn‑Dynamic 上取得最高 PSNR/SSIM/LPIPS（PSNR≈31.1 dB，SSIM≈0.913，LPIPS≈0.084），在真实数据上相较 RMFAT 提升约 2.8 dB PSNR、0.05 SSIM；推理速度远快于多帧 Transformer 基线，实时性显著提升。

**⚠️ 局限性**

局限性：对极端强度或非湍流退化（雾、尘等）泛化仍有限；需要伪 Ground‑Truth 进行训练和评估；两帧递归可能不适合极慢运动或高度非线性变形；在极低资源设备上的推理仍受限于 GPU 显存。

---

## 636. Instrumental Text-to-Music Generation with Auxiliary Conditioning Branches

**arXiv ID:** 2605.21433 | [PDF](https://arxiv.org/pdf/2605.21433v1)

**作者:** Junyoung Koh `[一作]` `[通讯]` (Yonsei University), Junyoung Koh (Yonsei University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了基于 Diffusion Transformer 的文本到音乐生成模型，移除辅助歌词与音色分支后检验其训练期间对感知质量的影响，并在 ICME 2026 ATTM 大赛中提交模型。

**💡 创新点**

证明辅助歌词/音色分支在训练期间充当结构锚点，即使输入为退化信号，去除它们会显著降低主观质量，并通过不同参数分配验证该效应。

**🔧 技术方法**

使用 Diffusion Transformer（ACE-STEP 1.5）、Min‑SNR‑γ 权重、适应性时间步采样、随机段裁剪、后期 EMA 检查点平均、ODE 采样 + 指导区间、CFG 与 EMA 窗口调优等技术。

**📊 数据集**

在 457 小时的 MTG‑Jamendo 子集上训练，使用官方 VAE 与 Qwen3 文本嵌入，并利用官方 Qwen2‑Audio 生成的 LLM 训练字幕。

**📈 对比分析**

与 Stable Audio Open、MusicGen 各规模基线在 AudioBox、CLAP、LLM‑as‑judge 以及人类 MOS 上进行对比；在 ATTM 2026 大赛 Performance 轨道获得第一名，Efficiency 轨道在客观指标上排名第二。

**⚠️ 局限性**

仅在 457 小时数据下训练，缺乏多轮 MOS 评测，结构作用仅为观察性发现，且对复杂编曲、乐器辨识的局限主要来自数据规模。

---

## 637. Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning

**arXiv ID:** 2605.21488 | [PDF](https://arxiv.org/pdf/2605.21488v1)

**作者:** Benhao Huang `[一作]` (Carnegie Mellon University), Zico Kolter `[通讯]` (Carnegie Mellon University)

**通讯引用:** 17433 | [OpenAlex ID](https://openalex.org/A5075035644)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Equilibrium Reasoners（EqR），通过学习任务条件的吸引子动力学，实现在测试时可扩展的推理。

**💡 创新点**

创新点在于把迭代模型视作可学习的固定点动力系统，并通过随机初始化和路径噪声两种任务无关的干预，塑造吸引子景观，使得深度和宽度扩展能可靠提升性能。

**🔧 技术方法**

使用权重共享的迭代更新器、截断梯度、分段在线训练、层级迭代、动态停止（ACT）以及基于残差的收敛判别等技术，构建可在测试时无外部验证器地自适应加深或加广推理。

**📊 数据集**

在两个高度结构化的推理基准上验证：9×9 Sudoku‑Extreme 与 Maze‑Unique，分别测评解题准确率与收敛残差。

**📈 对比分析**

与基线（Feed‑Forward、HRM、TRM）及无干预的EqR相比，随机初始化+噪声干预下的EqR在Sudoku上从 84.8% 提升至 99.8%（depth 64 + breadth 128），在 Maze 上从 82.2% 提升至 93.0%，且收敛残差与任务误差高度相关，支持基于残差的选择策略。

**⚠️ 局限性**

局限性包括：需在测试时执行数千次迭代（仍占用大量 FLOPs），吸引子景观的对齐依赖于训练设置，对完全无结构或动态生成任务的泛化尚未验证；此外，随机初始化与噪声需要额外的超参数调优。

---

## 638. ProtoPathway: Biologically Structured Prototype-Pathway Fusion for Multimodal Cancer Survival Prediction

**arXiv ID:** 2605.21454 | [PDF](https://arxiv.org/pdf/2605.21454v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 639. DeepWeb-Bench: A Deep Research Benchmark Demanding Massive Cross-Source Evidence and Long-Horizon Derivation

**arXiv ID:** 2605.21482 | [PDF](https://arxiv.org/pdf/2605.21482v1)

**作者:** Sixiong Xie `[一作]` (Peking University), Yun Ma `[通讯]` (Peking University)

**通讯引用:** 78797 | [OpenAlex ID](https://openalex.org/A5100369226)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个更难的深度研究基准，通过任务矩阵要求模型在单一领域内完成多维量化结论，需大量证据收集、跨源调和和多步推导。

**💡 创新点**

创新点在于将难度拆解为四个能力族（检索、推导、推理、校准），为每个单元格提供来源溯源记录和交叉验证，并使用自动化四级评分规则实现可审计评估。

**🔧 技术方法**

使用了标准化的检索工具（搜索、页面读取、PDF读取）、GPT‑5.5 评估器、四层评分规则以及对证据来源的四级披露标签。

**📊 数据集**

构建了100个任务、8×8矩阵的基准集（总6400单元格），覆盖技术、能源、工业、消费、金融、医疗等六个行业领域，来源包括公开文件、研究报告、媒体报道等。

**📈 对比分析**

通过对九个前沿模型（Codex + GPT‑5.5、Claude Opus 4.7、Claude Sonnet 4.6、DeepSeek V4 Pro/Flash、GLM 5.1、Qwen 3.6 Plus、MiniMax M2.7、Kimi K2.6）进行评测，平均分为27.17%，强模型在检索与推导表现最佳，弱模型易出现假精度错误；模型间差异显著，相关系数仅0.61。

**⚠️ 局限性**

局限在于基准主要聚焦定量推导任务，缺乏更复杂情境或主观判断；样本仅限100任务，可能无法覆盖全部真实研究情形；评估仅使用三种工具，无法测试模型在更广泛工具生态下的性能。

---

## 640. Is Fixing Schema Graphs Necessary? Full-Resolution Graph Structure Learning for Relational Deep Learning

**arXiv ID:** 2605.21475 | [PDF](https://arxiv.org/pdf/2605.21475v1)

**作者:** Yi Huang `[一作]` (Beihang University), Jianxin Li `[通讯]` (Beihang University)

**通讯引用:** 18899 | [OpenAlex ID](https://openalex.org/A5100380470)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在全分辨率约束下可优化的图结构学习框架FROG，用于关系深度学习；通过学习表的节点/边角色并联合优化GNN实现端到端学习。

**💡 创新点**

①首次将图结构学习与关系深度学习结合，并在保持全分辨率属性的前提下实现；②引入表-as-节点与表-as-边的角色学习，设计关系驱动的消息传播机制；③在表级和实体级引入功能依赖损失，强化数据库约束。

**🔧 技术方法**

基于图神经网络的角色感知门控机制、关系驱动的卷积算子、功能依赖正则化（表级低秩投影与实体级对比损失）、EMA平滑、交替优化等技术。

**📊 数据集**

使用RelBench基准套件，包含6个真实关系数据库，涵盖23个下游任务（实体分类、回归、链路预测）。

**📈 对比分析**

与统计基线、LightGBM、GraphSAGE、ID-GNN、RelGNN等方法对比，在ROC‑AUC、MAE、MAP等指标上均实现显著提升，尤其在大规模任务中优于现有最优模型。

**⚠️ 局限性**

局限性包括：对全分辨率约束的严格依赖限制了传统的边修剪/扩展策略；角色决策仍为离散化，可能无法捕捉更细粒度的结构；在极大规模图上训练仍需进一步优化；仅适用于结构化关系数据库，缺乏对非结构化或混合数据的直接支持。

---

## 641. Latent Dynamics for Full Body Avatar Animation

**arXiv ID:** 2605.21478 | [PDF](https://arxiv.org/pdf/2605.21478v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 642. You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories

**arXiv ID:** 2605.21468 | [PDF](https://arxiv.org/pdf/2605.21468v1)

**作者:** Zhepei Wei `[一作]` (University of Virginia), Yu Meng `[通讯]` (University of Virginia)

**通讯引用:** 6450 | [OpenAlex ID](https://openalex.org/A5069703881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了RLVR（可验证奖励的强化学习）在大型语言模型中的权重更新轨迹，发现其几乎完全聚焦在每个张量的单一秩1方向上，并且该方向上的系数随训练步数近线性变化；基于此，提出了RELEX——一种无需训练、只利用短期观察窗口通过SVD提取秩1子空间并进行线性外推即可获得近乎完整RLVR效果的检查点外推方法。

**💡 创新点**

创新点在于：①首次系统证明RLVR权重更新具有极低秩结构且近线性；②提出RELEX，利用单向量低秩近似与线性回归实现完全无模型、计算高效的检查点外推；③展示此方法在in-domain和out-of-domain任务上可用仅15–20%训练成本匹配甚至超越完整RLVR。

**🔧 技术方法**

核心技术包括：对每个参数张量的权重增量序列做SVD截断到秩1，提取主方向；对该方向上的投影系数做线性最小二乘回归；用回归得到的系数与主方向相乘加回基模型权重，得到未来检查点；整个过程无学习参数。

**📊 数据集**

实验使用的任务与数据集为数学推理基准MATH，以及五个OOD基准（AIME 2025/2026、HMMT 2025、OlympiadBench、AMC 2023）。评测模型为Qwen2.5‑Math‑1.5B、Qwen3‑4B‑Base与Qwen3‑8B‑Base。

**📈 对比分析**

与基线（基模型、完整RLVR）、ExPO、AlphaRL、Weight Extrap、Logits Extrap等方法对比。RELEX在MATH上与RLVR相当或略优，OOD平均性能更好；训练成本仅为15–20% RLVR；在所有三种模型上都优于其他外推方案。

**⚠️ 局限性**

局限性包括：仅验证GRPO在数学推理任务与Qwen系列模型上的低秩规律；不确定是否适用于其他RL算法（如PPO）、其他任务（如代码生成）或其他模型架构；对观察窗口长度和秩的选择敏感，需要经验或自适应方法；单秩1的假设在某些模型上可能不足，需进一步研究自适应子空间选择。

---

## 643. Approximation Theory for Neural Networks: Old and New

**arXiv ID:** 2605.21451 | [PDF](https://arxiv.org/pdf/2605.21451v1)

**作者:** Soumendu Sundar Mukherjee `[一作]` (Indian Statistical Institute), Himasish Talukdar `[通讯]` (Indian Statistical Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了全连接前馈网络的通用逼近定理、深度-宽度权衡以及 Kolmogorov–Arnold 网络（KAN）的理论与近似性能。

**💡 创新点**

将传统 UAT 逐步扩展为量化逼近率、深度分离与宽度阈值等新结果，并首次将 KAN 的 Kolmogorov–Arnold 表示与逼近理论结合，给出可构造的 B-spline 近似定理。

**🔧 技术方法**

主要使用功能分析（Hahn–Banach、Riesz 表示）、傅里叶分析、Barron 类函数、半代数门函数理论以及 B‑spline 逼近理论等数学工具。

**📊 数据集**

无实验数据集（为综述论文），讨论全部为理论与定理证明。

**📈 对比分析**

通过对比理论上可达的逼近误差与网络参数规模，展示深度可显著降低参数需求；但并未给出数值实验或具体性能指标。

**⚠️ 局限性**

局限性：缺乏实验验证，讨论聚焦于 FNN 与 KAN，未涉及卷积网络、Transformer 等现代架构；深度分离结果多为构造性示例，难以直接推广至一般函数类。

---

## 644. Uni-Edit: Intelligent Editing Is A General Task For Unified Model Tuning

**arXiv ID:** 2605.21487 | [PDF](https://arxiv.org/pdf/2605.21487v1)

**作者:** Dian Zheng `[一作]` (Chinese University of Hong Kong), Hongsheng Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 42037 | [OpenAlex ID](https://openalex.org/A5100732450)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了智能图像编辑任务 Uni-Edit，构建了 148k 条高质量的编辑数据集，并将其作为单一任务对统一多模态模型（BAGEL、Janus‑Pro）进行后期微调，显著提升了理解、生成和编辑三项能力。

**💡 创新点**

首次将编辑任务定位为统一多模态模型的通用调优任务；通过自动化数据合成管线，将 VQA 题目转化为包含推理、嵌套逻辑的编辑指令，突破了多任务训练的性能折衷；只需单一任务、单一数据集和单一训练阶段即可实现多能力协同提升。

**🔧 技术方法**

使用 GPT‑4o 进行指令生成与质量过滤；Nano‑Pro 负责高质量图像编辑；在 BAGEL 基础上采用 VAE 特征 dropout 与两阶段微调（仅生成损失 + LM head 对齐）进行训练；实验采用 FSDP 并行。

**📊 数据集**

数据来源为 LLaVA‑OV1.5 VQA；构造的 Uni‑Edit‑148k（以及精炼版 Uni‑Edit‑40k）；对比基准包括 AnyEdit、Bee、MMBench 等；评测使用 MMMU、MME、MathVista、MMVP、MMBench、Geneval、WISE、ImgEdit、GEdit、RISE。

**📈 对比分析**

与原始 BAGEL、RecA、AnyEdit、Janus‑Pro 等模型做系统对比，实验表明 Uni‑Edit‑微调后理解分数提升约 0.6‑1.0 分、生成分数提升 0.02‑0.04 分、编辑分数提升 1.5‑3.0 分，且不需额外的数据平衡或辅助模块。

**⚠️ 局限性**

受限于基底模型的文本渲染能力，部分指令（OCR、Caption、Math）未能充分发挥；数据合成对大模型依赖强，扩展性受限；图像分辨率和 VAE dropout 的设置对结果敏感。

---

## 645. A Machine Learning Framework for Weighted Least Squares GNSS Positioning based on Activation Functions

**arXiv ID:** 2605.21461 | [PDF](https://arxiv.org/pdf/2605.21461v1)

**作者:** Pin-Hsun Lee `[一作]` (McGill University), Harry Leib `[通讯]` (McGill University)

**通讯引用:** 1974 | [OpenAlex ID](https://openalex.org/A5018351682)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出基于集成学习的信号质量评分与激活函数框架，将GNSS测量权重映射至加权最小二乘（WLS）定位，以提升城市峡谷环境下的定位精度。

**💡 创新点**

创新点在于：①将激活函数用于把机器学习输出的质量分数转化为WLS权重；②在多星座（GPS+BeiDou）情境下验证不同激活函数（Sigmoid优于ReLU）的效果；③证明训练集来自相似城市化水平的数据即可在不同地区迁移使用。

**🔧 技术方法**

使用集成学习算法（随机森林、AdaBoost、梯度提升）提取六个信号质量特征；利用激活函数（常数、线性、单位阶跃、ReLU、Sigmoid）将预测分数映射为权重；随后采用WLS求解最终定位。

**📊 数据集**

采用 UrbanNav 开源数据集：香港（深/严/中城市）和东京（新宿/大滨）GNSS观测与地面真值。

**📈 对比分析**

通过与传统WLS（所有星座）以及“最优子集”oracle 进行对比；在香港中城数据中，Sigmoid+AdaBoost 将 GPS 单星座 3D RMSE 从约 192 m 降至 165 m（≈30%提升），多星座情形提升 25–50%；在跨城市迁移实验中也保持显著改进。

**⚠️ 局限性**

局限性：模型对训练环境依赖较强，跨区域迁移时性能下降；激活函数若参数不当可能过度削弱部分信号；实验仅基于 GNSS，未结合惯性导航等融合技术。

---

## 646. Agentic Model Checking

**arXiv ID:** 2605.21434 | [PDF](https://arxiv.org/pdf/2605.21434v1)

**作者:** Youcheng Sun `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Jason Xue `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 6238 | [OpenAlex ID](https://openalex.org/A5101441768)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“代理式模型检查”（Agentic Model Checking）方法，将大型语言模型（LLM）代理与有界模型检查（BMC）后端结合，完成对系统级代码（C 与 Rust）的自动化规范推断、检查选择、计数例分类与改进。

**💡 创新点**

创新点在于：① 将 LLM 负责语义判断（推断前/后置条件、挑选算术检查、分类计数例、提出修正）与 BMC 负责所有安全性决策的“代理-求解”双向流程；② 采用自顶向下的调用上下文推断规范，生成可直接转译为 BMC 语义的 DSL；③ 以函数级的假设‑保证（assume‑guarantee）进行分层 BMC，并通过计数例去重、验证流水线和自适应改进循环实现可扩展的组合式验证。

**🔧 技术方法**

核心技术包括：LLM 代理（prompting + 迭代重试）、自顶向下规范推断、BMC（CBMC for C，Kani for Rust）、调用图分析、假设‑保证函数分解、计数例去重、四阶段验证流水线（可达性、可行性、动态重放、真实感审计）、自适应改进循环和规范/模式持久化。

**📊 数据集**

实验数据集共四类：1）基于 ARM64 的 675 函数、37 模块的 hobby kernel；2）五个 OSS‑Fuzz 预硬化 C 库（jq、OpenSSL、libcurl、libxml2、protobuf upb）；3）外部 Linux 驱动（Realtek r8125）；4）LLM 生成的 Rust 编译器/链接器（claudes-c-compiler，约 5 万行）。

**📈 对比分析**

评估结果显示：在 62 个确认的真实缺陷中，验证管线在四类代码中发现 34 个内核缺陷、2 个 jq UB、1 个驱动越界、25 个 Rust 编译器缺陷；在硬化库中实现大部分叶子函数的“bounded clean”验证；相比仅使用 BMC，代理式方法通过规范推断和计数例过滤大幅提高精度，并通过自适应改进显著减少假阳性。

**⚠️ 局限性**

局限性包括：1）对 LLM 推断规范的准确性高度依赖，若规范过宽可能漏报；2）BMC 的展开深度（默认 k=4）限制了深层循环或深度输入的覆盖；3）动态重放仅在宿主平台上执行，无法覆盖特定硬件或初始化假设；4）对未见计数例模式的处理仍需手工介入或后续迭代。

---

## 647. Mitigating Label Bias with Interpretable Rubric Embeddings

**arXiv ID:** 2605.21455 | [PDF](https://arxiv.org/pdf/2605.21455v1)

**作者:** Calvin Isley `[一作]` (Harvard University), Sharad Goel `[通讯]` (Harvard University)

**通讯引用:** 10261 | [OpenAlex ID](https://openalex.org/A5027036879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出通过基于专家制定的评分 Rubric 构建文本嵌入，来缓解在招聘、大学录取等领域因标签偏差导致的算法歧视。

**💡 创新点**

创新点在于将 LLM 用作“评分器”，将非结构化材料量化为一组可解释、与目标构念紧密相关的指标，避免了传统黑箱嵌入捕获的性别等敏感信号，从而显著降低标签偏差。

**🔧 技术方法**

技术方法包括：使用 OpenAI GPT‑5 对 396 条 Rubric 维度进行评分；对黑箱嵌入、去标记、边缘化等方案进行对比；采用岭回归、10×10 交叉验证和线性校准评估模型；通过对应实验定义 T(x,g) 并计算 bias_T(h)。

**📊 数据集**

数据集来自 2025‑2026 年一个大型公共政策硕士项目的 1,112 篇完整申请，包含简历、成绩单、推荐信、个人陈述及结构化信息；通过 Microsoft Document Intelligence 提取文本后再嵌入。

**📈 对比分析**

与黑箱嵌入、去标记、边缘化以及“kitchen sink”混合特征的模型相比，Rubric 嵌入在所有性别优势水平下实现近乎零偏差，并在保持或提升录取生平均真实分数的同时，显著提升录取人群的性别多样性，性能最佳。

**⚠️ 局限性**

局限性包括：构建 Rubric 需要大量专业知识和人工审阅；Rubric 嵌入仅能缓解与标签相关的偏差，对基于标签本身的差异影响仍存在；同时排除群体信息可能导致在某些需要组别数据的场景（如医学风险评估或平权政策）中失去必要的预测能力。

---

## 648. Variance Reduction for Expectations with Diffusion Teachers

**arXiv ID:** 2605.21489 | [PDF](https://arxiv.org/pdf/2605.21489v1)

**作者:** Jesse Bettencourt `[一作]` (NVIDIA), Jonathan Lorraine `[通讯]` (NVIDIA)

**通讯引用:** 267 | [OpenAlex ID](https://openalex.org/A5030776397)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种针对冻结教师的 Monte Carlo 梯度的计算意识方差计量框架，并在此基础上设计了可直接插拔的三种无偏估计器：计算重用、时步重要性采样和分层采样。该框架用于三类任务——文本引导 3D 优化（SDS）、单步扩散蒸馏（DMD）以及视频数据归因（MOTIVE）。

**💡 创新点**

创新点在于：① 将方差分解为昂贵上游计算与廉价噪声采样两部分，构建分层 Monte Carlo 估计器以实现计算重用；② 依据教师权重直接构造重要性采样提议，且与分层采样结合形成逆 CDF 采样；③ 引入计算意识的方差衡量与效率指标（ECM、RE），系统评估三种估计器的实际收益。

**🔧 技术方法**

使用技术包括：蒙特卡罗重要性采样、分层（stratified）采样、逆 CDF 采样、计算重用（amortized re‑noising）、在线 Welford 方差估计、梯度归一化与相似度度量、CLIP 评分与 FID 评价、以及基准实验的可视化与统计分析。

**📊 数据集**

数据集：SDS 采用 DreamFusion / Magic3D 等通用 3D 生成基准（未指明具体数据集，使用多种文本提示和相机视角）；DMD 训练基于 ImageNet‑256 并使用 DiT‑XL/2 预训练教师；归因实验使用 MOTIVE 视频归因框架中的流匹配视频模型，数据来源未明确给出。

**📈 对比分析**

与均匀 IID 采样基线进行对比，实验在不同批量大小与采样设置下测量方差、ECM、RE、CLIP 评分和 FID。结果显示：SDS 中计算重用 + 重要性采样 + 分层采样可使方差下降 2–3 倍，对应 ECM 最高可达 3.3×，CLIP 评分在相同 wall‑clock 下显著提升；DMD 中方差下降 3–16 倍（ECM ~20×）但未带来 FID 改善；归因任务中分层采样在相同预算下提升了 >2× 的计算有效性与排名相关性。

**⚠️ 局限性**

局限性包括：① 方差减小的收益仅在 Monte Carlo 梯度是主导瓶颈时显现，若上游计算占比不足则重用无效；② 在 DMD 中，方差降低未转化为性能提升，表明辅助稳定器与输入多样性等因素已成为新瓶颈；③ 需要冻结教师模型且上游计算相对昂贵，无法直接迁移至轻量级或动态教师场景；④ 重要性采样和分层采样的代理函数需要根据任务手动设计，若代理失准则可能引入偏差。

---

## 649. Quantifying Hyperparameter Transfer and the Importance of Embedding Layer Learning Rate

**arXiv ID:** 2605.21486 | [PDF](https://arxiv.org/pdf/2605.21486v1)

**作者:** Dayal Singh Kalra `[一作]` (University of Maryland), Maissam Barkeshli `[通讯]` (University of Maryland)

**通讯引用:** 4350 | [OpenAlex ID](https://openalex.org/A5004169905)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型训练中的超参数迁移问题，提出了评估迁移质量的三指标框架，并通过对标准参数化（SP）与最大更新参数化（μP）的系统 ablation 实验，发现嵌入层学习率是 μP 优势的关键因素；

**💡 创新点**

创新点在于：①提出可量化超参数迁移的三指标（loss 预测误差 ℰ、稳健性指数 κ、渐近损失衰减 ℛ(∞)）；②通过对 16 种 ablation 结果的系统分析，确定嵌入层学习率是实现高质量学习率迁移的决定性因素；③在固定步长和计算最优两种训练设置下，系统评估了权重衰减对迁移质量的影响；

**🔧 技术方法**

采用了 Maximal Update Parameterization (μP)、标准参数化 (SP)、AdamW 优化器、权重衰减、学习率预热-稳定-衰减调度、以及对宽度、学习率、权重衰减的网格搜索等技术；

**📊 数据集**

在 FineWeb-Edu 语料库上预训练 GPT‑style 解码器 Transformer，宽度从 128 扩展到 2048，固定层数、固定批量大小 1024，训练 10,000 步；

**📈 对比分析**

通过三指标对比 SP 与 μP 及其 ablation，发现 SP+Embd 与 μP 在 ℛ(∞)、κ 方面相近，但 μP 在 ℰ 方面更优；嵌入层学习率提升后，SP 的训练稳定性得到显著改善；在计算最优设置下，所有参数化的 ℛ(∞) 接近 0，表明损失不受参数化影响；

**⚠️ 局限性**

实验仅覆盖解码器 Transformer、固定深度、单一数据集、单个随机种子，未探讨其他架构、优化器、深度扩展和多数据集；

---

## 650. EvoStruct: Bridging Evolutionary and Structural Priors for Antibody CDR Design via Protein Language Model Adaptation

**arXiv ID:** 2605.21485 | [PDF](https://arxiv.org/pdf/2605.21485v1)

**作者:** Mansoor Ahmed `[一作]` (Georgia State University), Murray Patterson `[通讯]` (Georgia State University)

**通讯引用:** 2002 | [OpenAlex ID](https://openalex.org/A5026228482)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种结合冻结蛋白语言模型（ESM-2）与E(3)-等变图神经网络的跨注意力适配器（EvoStruct），用于抗体CDR设计，解决传统GNN方法的词汇坍塌问题。

**💡 创新点**

创新点在于：①跨注意力适配器将PLM的进化先验与结构上下文对齐；②采用逐步解冻策略和R-Drop一致性正则化，使模型在保持词汇多样性的同时提升准确率；③将整个序列预测路径限定在PLM的嵌入空间，直接继承语言模型的词汇校准。

**🔧 技术方法**

主要技术包括E(3)-等变关系图神经网络、ESM-2蛋白语言模型、跨注意力适配器、进化先验解冻、R-Drop一致性正则化以及多任务损失（序列、坐标、配对、对接、影子肽对等）。

**📊 数据集**

使用SAbDab数据库中经去重后保留2922个抗体-抗原复合物的benchmark，采用epitope-group划分进行评估。

**📈 对比分析**

与11种基准方法（包括GNN、扩散、检索、ODE、自动回归模型）对比，EvoStruct在序列恢复率（AAR 0.43）和困惑度（PPL 1.88）上均领先，显著高于最佳GNN基线（AAR 0.37，PPL 3.27），同时保持接近最佳的结构和接口质量。

**⚠️ 局限性**

局限性在于：①尽管词汇多样性和序列准确率提升明显，但在结合抗原信息预测界面亲和力（fnat、DockQ）方面仍略逊于无抗原条件的RefineGNN；②在抗原接触位点的AAR仍低（约22%），表明跨注意力适配器对特定接触位点的条件化能力有限。

---

## 651. Stream3D: Sequential Multi-View 3D Generation via Evidential Memory

**arXiv ID:** 2605.21472 | [PDF](https://arxiv.org/pdf/2605.21472v1)

**作者:** Kaichen Zhou `[一作]` (Massachusetts Institute Of Technology), Fangneng Zhan `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种训练‑免费、可直接嵌入现有单视角 3D 生成模型的流式包装器，能够在长时间单目视频流中保持时间连贯的 3D 重建。

**💡 创新点**

创新点在于引入了自适应证据记忆（Adaptive Evidential Memory），只存储每个体素查询所需的高证据帧，并通过证据驱动的多视角生成（Evidence‑Based Multi‑Generation）在每个块中挑选最有信息量的视角；整个过程保持常数内存且不需重新训练。

**🔧 技术方法**

核心技术包括：利用跨注意力图进行轻量级热身，以计算每帧对每个查询的熵/证据分数；按令牌级别维护顶‑D 证据列表；基于所有查询的所有权计数挑选顶‑K 视角；在生成时使用多视角流动匹配（Multi‑Diffusion）融合各视角的速度场。

**📊 数据集**

在公开的 GSO 与 NAVI 两大基准上进行实验，分别提供高质量扫描对象和多样化摄像机轨迹的真实场景。

**📈 对比分析**

与单视角、固定多视角（如 SAM‑3D、TRELLIS、EscherNet）以及流式基线（KV‑Cache、FlowEdit、MV‑SAM3D）进行对比。实验显示，本方法在 PSNR、SSIM、LPIPS、FID、PFID、Chamfer Distance、IOU 等视觉与几何指标上均显著优于所有基线，且在长序列中保持一致性且内存增长保持常数。

**⚠️ 局限性**

局限性在于完全依赖底层冻结模型的重建质量；若基础模型在单帧上已失效，证据记忆也无法弥补缺失的几何或外观信息。

---

## 652. AiraXiv: An AI-Driven Open-Access Platform for Human and AI Scientists

**arXiv ID:** 2605.21481 | [PDF](https://arxiv.org/pdf/2605.21481v1)

**作者:** Junshu Pan `[一作]` (Westlake University), Yue Zhang `[通讯]` (Westlake University)

**通讯引用:** 18241 | [OpenAlex ID](https://openalex.org/A5100333758)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了 AiraXiv——一种面向人类与 AI 科学家、支持迭代出版与会议的 AI 驱动开放预印本平台

**💡 创新点**

首次实现了基于 AI 的持续评审与反馈闭环，使稿件可在几小时内获得多维度质量信号，并允许作者快速迭代；同时提供了 MCP 接口实现 AI 科学家与平台的自动交互

**🔧 技术方法**

利用大语言模型（LLM）、MinerU 文档解析、AiraXiv masterbrain 调度、Agent‑based AI 审稿人、PaperIgnition 嵌入检索和 LLM 重新排序等技术栈

**📊 数据集**

采用 ICAIS 2025 会议的 114 篇稿件（82 篇 AI 生成、32 篇人类撰写）作为实验数据集，包含 42 篇接受论文和 72 篇拒稿论文

**📈 对比分析**

与人类专家最终决定对比，AI 评审得分与最终决定相关系数为 0.43，AUC 为 0.78；AI 评审平均完成时间约 10.3 小时，显著缩短传统评审周期，迭代后稿件 AI 评分普遍提升

**⚠️ 局限性**

AI 评审信号仍不完善、可能存在偏差与不稳定；大规模评审会产生较高计算开销；系统易受低质量或恶意反馈攻击；仅在有限场景下验证，缺乏长期社区与多领域的泛化证据

---

## 653. Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling

**arXiv ID:** 2605.21470 | [PDF](https://arxiv.org/pdf/2605.21470v1)

**作者:** Caleb Winston `[一作]` (Stanford University), Christos Kozyrakis `[通讯]` (Stanford University)

**通讯引用:** 21214 | [OpenAlex ID](https://openalex.org/A5042148531)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于 JIT 编译的计算机使用代理，能够将自然语言任务动态翻译为可执行代码，并通过并行规划与调度显著提升执行效率与准确性。

**💡 创新点**

创新点包括：① 引入预/post 条件的可验证工具协议，保证工具调用的状态一致性；② 通过控制流图（CFG）实现多候选计划的并行生成、静态校验与成本估算；③ 采用 Monte Carlo 采样和已学习的延迟分布实现成本感知的调度策略选择。

**🔧 技术方法**

技术细节包括：JIT‑Planner、JIT‑Scheduler、Invariant‑Enforcing Tool Protocol、CFG‑based 计划验证与成本评估、并行候选生成、缓存可重用工具、学习的延迟分布与蒙特卡洛成本估算。

**📊 数据集**

使用了 REAL（Dashdish、Gomail、Omnizon）和 WebArena（GitLab、Reddit）两个基准中的 37 个任务，涵盖电商、通信、协作与社交等多种 Web 交互场景。

**📈 对比分析**

通过与 Browser‑Use、Browser‑Use+cache、Anthropic/OpenAI CUA、Fixed Scheduler（Serial/Parallel/Hedge）和 Oracle‑Scheduler 进行对比，JIT‑Planner 在所有任务上平均提升 10.4× 的速度、+28% 的准确率；JIT‑Scheduler 在 4 vCPU 下平均提升 2.4× 的速度、+9% 的准确率，相较于 OpenAI CUA 表现更优。

**⚠️ 局限性**

主要限制包括：1) 需要一次性离线设置（工具合成和延迟分布采样，耗时 25–90 分钟）；2) 缓存工具对 UI 变动敏感，需检测并重新生成；3) 对高度随机化环境（CAPTCHA、速率限制）鲁棒性有限；4) 评估范围仅限于 Web 环境，未覆盖桌面或移动平台。

---

## 654. DelTA: Discriminative Token Credit Assignment for Reinforcement Learning from Verifiable Rewards

**arXiv ID:** 2605.21467 | [PDF](https://arxiv.org/pdf/2605.21467v1)

**作者:** Kaiyi Zhang `[一作]` (Renmin University of China), Yankai Lin `[通讯]` (Renmin University of China)

**通讯引用:** 12930 | [OpenAlex ID](https://openalex.org/A5043098453)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于对抗性权重分配的DelTA方法，重新权衡RLVR（可验证奖励）中的token梯度，以提升大语言模型的推理与代码生成性能。

**💡 创新点**

创新点在于将RLVR更新视为隐式线性判别器，并通过对正负优势token梯度的对比重构判别器，从而对token信用分配进行精细调节，克服了传统平均化聚类易被高频共现token主导的问题。

**🔧 技术方法**

主要技术包括：① 先验判别器视角解析RLVR梯度；② 通过距离对比与熵正则化计算token系数；③ 对系数进行自归一化并在DAPO等基准中替换统一token平均；④ 仅利用stop‑gradient计算权重，保持完整参数更新。

**📊 数据集**

实验数据集涵盖七个数学推理基准（AIME24/25/26、HMMT25(Feb./Nov.)、HMMT26(Feb.)、Brumo25）以及代码生成与不同模型家族（Qwen3-8B、Qwen3-14B、Olmo3-7B）和OOD评估。

**📈 对比分析**

与现有RLVR基线（DAPO、SAPO、FIPO、DAPO w/ Forking Tokens）在相同模型规模下对比，DelTA 在 Qwen3-8B 上平均提升约 3.3 分，在 Qwen3-14B 上平均提升约 2.6 分，并在代码生成和 OOD 任务中也保持显著优势。

**⚠️ 局限性**

限制主要在于：① 需要额外的token梯度计算和分数迭代，增加计算开销；② 目前仅在特定的可验证奖励场景验证，尚未针对非可验证奖励或多任务学习进行广泛评估；③ 对温度、系数区间等超参数的敏感性仍需进一步研究。

---

## 655. StreamGVE: Training-Free Video Editing via Few-Step Streaming Video Generation

**arXiv ID:** 2605.21466 | [PDF](https://arxiv.org/pdf/2605.21466v1)

**作者:** Guanlong Jiao `[一作]` (University of British Columbia), Renjie Liao `[通讯]` (University of British Columbia)

**通讯引用:** 6137 | [OpenAlex ID](https://openalex.org/A5048686150)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练‑自由的视频编辑框架StreamGVE，利用少步流式生成实现源视频到目标视频的快速编辑；

**💡 创新点**

核心创新在于将视频编辑从传统的数据‑到‑数据流程转为噪声‑到‑数据流式生成，并设计双分支采样、自注意力桥接、交叉注意力定位/放大、源导向引导以及可选视觉提示；

**🔧 技术方法**

结合流式生成模型（如Self‑Forcing、LongLive）、流匹配、一致性模型、KV缓存以及自注意力/交叉注意力机制实现高速可控编辑；

**📊 数据集**

主要使用FiVE‑Bench（100视频，420编辑对）评估短视频编辑，并构造21条30秒长视频（16 FPS）进行长视频用户研究；

**📈 对比分析**

与TokenFlow、VidToMe、VideoGrain、DMT、Pyramid‑Edit、Wan‑Edit、UniEdit‑Flow、StreamV2V、AdaFlow等先进方法对比，StreamGVE在文本对齐、结构/背景保持、图像质量、时序一致性以及编辑准确率上均优于对手，并在用户研究中在质量、背景一致性和编辑精度方面获得第一；

**⚠️ 局限性**

局限性主要在于依赖已有的预训练流式生成模型，尚未验证极端实时性（低延迟）场景；对某些极度复杂的编辑任务仍可能出现细节失真或背景抖动。

---

## 656. Leveraging LLMs for Grammar Adaptation: A Study on Metamodel-Grammar Co-Evolution

**arXiv ID:** 2605.21465 | [PDF](https://arxiv.org/pdf/2605.21465v1)

**作者:** Weixing Zhang `[一作]` (Karlsruhe Institute of Technology), Daniel Strüber `[通讯]` (Chalmers University of Technology)

**通讯引用:** 5566 | [OpenAlex ID](https://openalex.org/A5000688587)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究使用大型语言模型（LLM）自动化 Xtext DSL 的语法适配，学习并重用历史版本的适配变更，减少人工修改。

**💡 创新点**

创新点在于不依赖预定义规则，利用 LLM 从历史适配对中学习变更模式，并在新版本中自动重用，特别能处理复杂语法变更。

**🔧 技术方法**

采用 Claude Sonnet 4.5、ChatGPT 5.1、Gemini 3 等 LLM 与精心设计的 prompt，结合 Xtext 生成的语法文件。

**📊 数据集**

使用六个真实 DSL（EAST‑ADL、BibTeX、Xenia、DOT、Xcore、SML）以及 QVTo 的四个官方版本作为评估数据集。

**📈 对比分析**

与 GrammarTransformer（基于规则的方式）对比；在中小规模 DSL 上 LLM 在规则一致性（RAC）和相似度均达到 100%，而在大规模 EAST‑ADL 上性能显著下降。

**⚠️ 局限性**

局限性在于对大规模语法的系统性遗漏适配操作，无法保持一致性；LLM 在大规模、重复性强的语法变更中表现不足，需要进一步改进规模处理与规则生成机制。

---

## 657. HITL-D: Human In The Loop Diffusion Assisted Shared Control

**arXiv ID:** 2605.21460 | [PDF](https://arxiv.org/pdf/2605.21460v1)

**作者:** Riley Zilka `[一作]` (University of Alberta), Martin Jagersand `[通讯]` (University of Alberta)

**通讯引用:** 6952 | [OpenAlex ID](https://openalex.org/A5030948380)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种人机混合控制框架 HITL-D，让用户通过操纵杆控制机器人末端执行器的平移位置，同时由扩散式策略根据点云和位置实时生成姿态指令，从而实现更直观、低负荷的操作。

**💡 创新点**

将扩散模型的多模态、可条件生成能力与人机交互相结合，只用一份专家演示即可学习姿态补偿，既保留了人类自主性，又提供了柔性、实时的姿态协助。

**🔧 技术方法**

使用基于条件去噪扩散模型（DDIM）处理稀疏点云与机器人状态的输入，配合 3 层 MLP 编码器和比例控制器来实现实时姿态预测。

**📊 数据集**

训练数据来自单个专家演示（用 Xbox 360 控制器收集的轨迹）以及相机获取的 2048 点的彩色点云，未使用公开大型数据集。

**📈 对比分析**

在 12 名普通用户的三项任务（拆箱、螺丝刀、形状匹配）中，HITL-D 与传统 Cartesian 和 Point‑and‑Go 控制进行单盲随机实验，结果显示任务完成时间平均降低约 40%（最高 49%），NASA‑TLX 工作负荷下降约 37%，成功率保持 100%，并在 Likert 调查中获得更高的易用性和独立性评分。

**⚠️ 局限性**

仅基于单个演示，缺乏对不同物体、相机位置和机器人变体的鲁棒性；点云裁剪为手工固定，未加入视觉识别；未在残障用户或更广泛 ADL 场景中验证，且未评估对语义推理的支持。

---

## 658. Mind the Sim-to-Real Gap & Think Like a Scientist

**arXiv ID:** 2605.21458 | [PDF](https://arxiv.org/pdf/2605.21458v1)

**作者:** Harsh Parikh `[一作]` (Amazon SCOT), Alexander Volfovsky `[通讯]` (Amazon SCOT)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了在已有预训练模拟器与真实实验相结合时的价值差距本地化与可达性分解，并基于此设计了 Fisher-SEP 试验规划方法。

**💡 创新点**

创新点在于把模拟器误差拆分为可随机化消除的校准-部署偏移和不可消除的参数残差，提出了“局部–可达性”价值差距分解，以及以目标策略值的后验预测方差为目标的 Fisher-SEP 设计。

**🔧 技术方法**

采用了贝叶斯决策理论、扩展的模拟器引理、Fisher 信息与贝叶斯后验推断、Bellman 解析式梯度和经验风险估计等技术。

**📊 数据集**

使用了两个仿真案例：一是基于假设季节性与库存需求的五台自助售货机供应链数据；二是基于 SIS 模型的 5×8 网格 HIV 移动检测模拟。

**📈 对比分析**

与 SOP、A‑SOP、Thompson、UCRL2、UCBVI 等基线比较，结果显示 Fisher‑SEP‑R 在供给链中短期成本后在长周期内超过 A‑SOP；Fisher‑SEP‑T 在 HIV 场景中显著减少感染并在 400 天内获得 85% 以上的 oracle 表现，均优于传统随机或基于探索的基线。

**⚠️ 局限性**

限制包括仅适用于有限离散隐状态的表格模型、对隐状态动力学收敛的假设、独立先验导致的局部可达性假设、以及对参数残差的不可消除性，未涵盖函数逼近、非收敛动态或更复杂的先验结构。

---

## 659. WikiVQABench: A Knowledge-Grounded Visual Question Answering Benchmark from Wikipedia and Wikidata

**arXiv ID:** 2605.21479 | [PDF](https://arxiv.org/pdf/2605.21479v1)

**作者:** Basel Shbita `[一作]` (IBM Research), Anna Lisa Gentile `[通讯]` (IBM Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个人类审阅的知识驱动视觉问答基准，结合 Wikipedia 图像、文章标题与 Wikidata 结构化知识，生成多项选择问答。

**💡 创新点**

创新点在于严格保证答案必须依赖外部知识并通过人工审核保证事实准确与视觉文本一致；提供可验证的结构化真值与统一的评价工具。

**🔧 技术方法**

利用大语言模型（Granite-3.3-8B-Instruct）进行候选问答生成，并用结构化知识检索与过滤、自然语言化三元组，再人工审阅。

**📊 数据集**

使用 Wikipedia Image Text (WIT) 数据集中的 37M 图像-标题对，结合对应的 Wikidata 实体和三元组。

**📈 对比分析**

对 15 种不同规模（256M‑90B）VLM 进行统一多项选择评测，精度从 24.7% 到 75.6% 变化，验证了基准对模型能力的区分度。

**⚠️ 局限性**

局限包括样本量有限（仅 344 题），依赖 Wikipedia/Wikidata 领域覆盖，多项选择限制了开放式推理展示，以及生成过程可能携带 LLM 偏差。

---

## 660. Lost in Fog: Sensor Perturbations Expose Reasoning Fragility in Driving VLAs

**arXiv ID:** 2605.21446 | [PDF](https://arxiv.org/pdf/2605.21446v1)

**作者:** Abhinaw Priyadershi `[一作]` (NVIDIA Corporation), Jelena Frtunikj `[通讯]` (NVIDIA GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 Vision‑Language‑Action 模型 Alpamayo R1 在 1,996 条驾驶场景下进行控制性传感器失真实验（8 种噪声/光照/雾霾条件），通过 18,000 次推理评估轨迹精度与链式因果解释（CoC）的一致性。

**💡 创新点**

首次证明 CoC 一致性是轨迹可靠性的高保真指标：解释变化时轨迹偏差激增 5.3 倍；建立噪声剂量-误差线性关系；表明启用 CoC 生成可提升 11.8% 轨迹准确率；对标准预处理防御效果进行系统评测；提出基于 CoC 的运行时监控思路。

**🔧 技术方法**

使用 10B 参数的 Alpamayo R1 VLA，生成链式因果自然语言解释；采用 Gaussian 噪声、光照缩放、Narasimhan‑Nayar 雾霾模拟；评估指标包括 ADE、ΔADE、CoC 变化率、L2 轨迹偏差；采用二进制字符串匹配判断 CoC 变化；进行受控消融实验；统计相关性和 ROC 性能。

**📊 数据集**

PhysicalAI‑Autonomous‑Vehicles 验证集（1,996 条驾驶序列），按动作原语划分为车辆跟随、交叉口导航、信号遵循、车道保持、超车、转弯、其他等类别。

**📈 对比分析**

与恒速物理基线对比，Alpamayo R1 在清洁条件下 ADE 为 2.00 m，比基线低 68.3%；噪声 σ=70 时 ADE +0.30 m，CoC 变化率 52.7%，>5 m 偏差样本 70.6%；CoC 变化与轨迹偏差的相关性 r=0.99、r_pb=0.53，CoC 变化导致平均偏差从 4.13 m 增至 21.82 m；CoC 生成提升 ADE 约 11.8%（p<0.0001）。标准预处理防御无显著改进。

**⚠️ 局限性**

实验采用合成且逐帧独立的失真，缺乏真实传感器的时间相关性；评估为单向无反馈的开放式循环，未考虑误差累积；仅覆盖 8 种失真，噪声剂量水平有限；模型规模大（10B）导致推理延迟高；结果对其它 VLA 架构的推广性尚未验证。

---

## 661. Fully Actuated Manifold Constraint Based Output Feedback Control for Input-Constrained Uncertain Nonlinear Systems

**arXiv ID:** 2605.21439 | [PDF](https://arxiv.org/pdf/2605.21439v1)

**作者:** Dianrui Mu `[一作]` (Yanshan University), Rao Wei `[通讯]` (Yanshan University)

**通讯引用:** 257281 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种全新低复杂度、模型无关的输出反馈控制器——全激活流形约束控制（FAMCC），能够在未知非线性系统、未知输入约束下实现预设稳态精度的有限/固定时间跟踪。

**💡 创新点**

创新点在于：①将线性流形约束扩展到非线性流形并通过自适应流形设计实现精度可预设；②采用错误驱动的柔性约束在无输入约束信息时实现柔性控制；③引入双边平滑过渡函数与正指数反馈实现非奇异滑模与固定时间收敛；④构建递归固定时间控制实现高阶系统的快速收敛。

**🔧 技术方法**

使用技术包括：全激活错误系统变换、迭代流形构造、正指数反馈（SSMD/NSMD）、平滑过渡函数、差分器估计误差、低复杂度控制律与灵活约束调节。

**📊 数据集**

未使用公开数据集，全部以数值仿真验证，仿真系统为随机扰动下的二阶和三阶严格反馈非线性系统。

**📈 对比分析**

与传统线性流形约束控制（LMCC）、滑模控制（SMC）等方法相比，FAMCC在满足输入饱和时仍能保持柔性精度，在满足非饱和时可在预设时间内收敛至精度；仿真表明控制精度与能耗近似最优，且控制律结构简洁。

**⚠️ 局限性**

局限性包括：①对系统模型可微性有假设，可能需采用Filippov非光滑理论；②高阶系统时迭代流形构造可能导致微分爆炸，需要更高效的非迭代快速流形方法；③实验验证尚缺失，实际硬件实现仍待检验。

---

## 662. Gaussian Sheaf Neural Networks

**arXiv ID:** 2605.21435 | [PDF](https://arxiv.org/pdf/2605.21435v1)

**作者:** André Ribeiro `[一作]` (Getulio Vargas Foundation), Diego Mesquita `[通讯]` (Getulio Vargas Foundation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Gaussian Sheaf Neural Network (GSNN)，用于图上分布对分布回归

**💡 创新点**

首次将高斯分布特征映射到细胞层的 Gaussian Sheaf，并定义相应 Laplacian，利用概率分布的几何结构提升图学习表现

**🔧 技术方法**

采用细胞层、Sheaf Laplacian、Wasserstein 损失、MLP 端到端学习限制映射等技术

**📊 数据集**

在四个模拟数据集（Barabasi‑Albert、Watts‑Strogatz）和两个真实天气数据集上评估

**📈 对比分析**

与 MLP、GCN、NSD（多种限制映射）、GaussianGCN、GSNN‑GraphLap 对比，GSNN 在 5/6 数据集上取得最低 2‑Wasserstein 距离，性能显著优于基线

**⚠️ 局限性**

仅能处理高斯输入，输出仍可为任意分布，对非高斯输入的泛化能力有限

---

