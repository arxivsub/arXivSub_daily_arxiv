# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-09 | 今日论文总数: 448

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Healthier LLMs: Retrieval-Augmented Generation for Public Health Question Answering

**arXiv ID:** 2607.06641 | [PDF](https://arxiv.org/pdf/2607.06641v1)

**作者:** Felix Feldman `[一作]`, Toby Nonnenmacher `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对 PubHealthBench 进行检索增强生成（RAG）改造，系统性评估不同检索策略和生成模型在英国公共健康指南上的问答表现。

**💡 创新点**

发现混合检索（稠密+稀疏+融合）显著提升检索召回与排名质量，且高质量检索可让小型开源 LLM 与无检索的大型模型竞争或超越；同时提出基于 LLM 的裁判器用于自由回答的可靠性评估。

**🔧 技术方法**

采用密集向量检索（NV-Embed-v2、EmbeddingGemma 等）、TF‑IDF/BM25 词法检索、Reciprocal Rank Fusion（RRF）融合、摘要生成（GPT‑4o‑mini）简化检索语料、RAG 生成框架，并使用 GPT‑OSS‑120B 作为评判器。

**📊 数据集**

使用 PubHealthBench（7929 题）及其 5358 个基于 UK 公共健康指导文档分块的检索语料，以及 760 个手工标注的自由回答子集。

**📈 对比分析**

对检索进行 Recall@k、MRR、nDCG、Precision@1；对 MCQA 采用准确率；对自由回答采用 LLM‑judge 的四项评判（faithfulness、completeness、factual consistency、clarity）并与人类标注对比。结果显示：混合检索在 Recall@10 达 0.97、MRR 0.99；MCQA 在 3–5 个检索块下达 0.99+ 的准确率，小型 LLM 在检索条件下可与 GPT‑4.5（92.5%）相媲美或超过；自由回答中 faithfulness 与 completeness 的 LLM‑judge 与人工一致，表现稳健，但 factual consistency 与 clarity 的评判仍不够可靠。

**⚠️ 局限性**

局限性：检索评估仅考虑单一目标块，忽略多块相关性；chunk 长度与摘要方法未做系统控制；主题对检索排名的影响仅为小幅度；MCQA 可能高估真实能力；自由回答评估依赖单一参考块，LLM‑judge 对 factual consistency 的一致性低，提示该维度及多源检索尚需改进。

---

## 2. Dual Attention Heads for Personalized Federated Learning in ECG Classification

**arXiv ID:** 2607.06653 | [PDF](https://arxiv.org/pdf/2607.06653v1)

**作者:** Kien Le `[一作]` (Florida State University), Tuy Tan Nguyen `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种FedDualAtt框架，将Transformer注意力头分为全局聚合分支和本地专用分支，用于多中心12导联ECG的多标签分类。

**💡 创新点**

创新点在于通过架构层面的注意力头分离实现全局与本地特征的并行学习，且无需额外目标或超参数；在FedCVD基准上全局分支已显著提升全局Micro‑F1，加入少量本地头可进一步提升个体性能。

**🔧 技术方法**

使用的技术包括ResNet1D‑34特征提取、双注意力Transformer块、FedAvg聚合以及全局/本地参数的严格分离。

**📊 数据集**

使用的公开数据集为FedCVD四中心ECG数据集（SPH、PTB‑XL、SXPH、G12EC），共20类多标签诊断任务。

**📈 对比分析**

与FedAvg、FedProx、Scaffold、Ditto、FedALA等标准与个性化联邦学习方法对比，FedDualAtt 8H_g:0H_l 在全局Micro‑F1上达到72.7%（高于Scaffold 70.1%），加入本地头后三中心的Micro‑F1均提升并保持稳定，整体性能表现优异。

**⚠️ 局限性**

局限性包括需要手动设定全局/本地头比例，未实现自动比例学习；在极端本地化场景下全局性能下降；目前仅验证于ECG领域，尚未评估在其他医疗信号上的泛化能力。

---

## 3. JAX-FVM: A differentiable, entropy-stable finite volume solver on unstructured meshes for compressible flows

**arXiv ID:** 2607.07385 | [PDF](https://arxiv.org/pdf/2607.07385v1)

**作者:** Guillaume de Romémont `[一作]` `[通讯]` (INRIA), Guillaume de Romémont (INRIA)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了 JAX‑FVM，一个基于 JAX 的可微分有限体积求解器，能够在非结构化三角网格上求解二维可压缩欧拉和纳维‑斯托克斯方程，并提供多种显式/隐式时间积分器。

**💡 创新点**

创新点包括：①将整个残差计算（网格连通、斜率重建、数值通量、边界处理）完全用 JAX 编写，实现端到端可微分；②采用熵守恒 Tadmor/Ismail‑Roe 两点通量并加入熵变量扩散，保证离散熵不变性；③首次在可微分 CFD 框架中实现非结构化网格的压缩流数值方法，弥补了现有框架大多仅支持结构化网格的空缺。

**🔧 技术方法**

使用技术包括：JAX 自动微分（forward‑mode 用于 Jacobian‑vector 乘法、reverse‑mode 用于后向传播）、JAX JIT 与向量化、有限体积离散（MUSCL、Least‑Squares 梯度、Venkatakrishnan 限制器）、熵守恒通量、显式 RK2/3/4 与隐式 SDIRK2、GMRES 求解器、VTK 可视化。

**📊 数据集**

使用了多组标准验证案例作为数据集：19 个二维 Riemann 配置、双 Mach 反射、Kelvin‑Helmholtz 演化、Taylor‑Green 循环、各类低马赫等熵流体问题（如 advected、dipole、co‑rotating、merging Lamb‑Oseen 对）、以及前冲斜面和圆柱通道等几何案例。

**📈 对比分析**

对比方法主要是通过数值验证与已知解析/实验结果进行比较，重点关注熵守恒、能量守恒和解的物理性；在前冲斜面等标准案例中，JAX‑FVM 能在完全非结构化网格上捕捉冲击波、马赫支、反射波与滑行线，且与传统结构化网格求解器在精度和稳定性上保持一致或更优；此外，其端到端可微分特性在梯度计算上显著提升效率，远优于手工实现的 adjoint 方法。

**⚠️ 局限性**

主要限制包括：仅支持二维三角形网格；缺乏外部 VTK 网格读入接口；未实现湍流模型（如 Smagorinsky、RANS）；目前的性能尚未针对大规模三维问题做优化，且对 GPU 的加速支持仍处于实验阶段。

---

## 4. Behavior Leverage Imbalance in Multi-Teacher On-Policy Distillation

**arXiv ID:** 2607.07050 | [PDF](https://arxiv.org/pdf/2607.07050v1)

**作者:** Jiabin Shen `[一作]`, Chengjun Mao `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多教师对抗性策略（OPD）在工具调用中的过度使用现象，并提出基于每个token的发散校准方法来缓解该问题。

**💡 创新点**

提出了行为杠杆失衡的诊断框架，揭示高杠杆位置的本地信号会驱动全局行为偏移，并设计了一种动态压缩极端token发散的校准规则，既压缩损失又保持梯度。

**🔧 技术方法**

采用多教师OPD与通用知识蒸馏（GKD）结合的框架，利用Jensen‑Shannon发散量作为token损失，并实现了基于stopgrad的动态阈值校准规则；同时进行token级别的行为杠杆分析。

**📊 数据集**

使用APIGen‑MT、BFCL、When2Call三个公开数据集进行单轮决策评估，并构造BFCL多轮循环诊断数据集评估真实交互中的工具调用行为。

**📈 对比分析**

与vanilla GKD、硬剪枝（Hard Clip）和全局重加权（Global Reweight）等基线比较；在APIGen‑MT上，所提方法将过度调用率从13.7%降至9.0%，同时保持89.2%的决策准确率；在BFCL上提升无关拒绝率，在多轮诊断中降低循环率和重复调用率，提高最终答案率。

**⚠️ 局限性**

实验仅在单一基模型和两教师（工具调用 vs. 直接响应）设置下进行，未对不同任务或多种教师组合进行验证；缺乏多随机种子复现；校准方法是经验性缓解而非理论最优，可能在过度压缩时导致对有用高杠杆信号的欠拟合；诊断指标主要基于教师强制过程，尚未通过完整自由生成轨迹验证因果关系；需额外的无监督校准实验以探究纯OPD环境下的可行性。

---

## 5. Billions of Sketches Reveal Hidden Cultural Variation in Human Concepts

**arXiv ID:** 2607.07267 | [PDF](https://arxiv.org/pdf/2607.07267v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 6. LLMs Silently Correct African American English: Auditing and Mitigating Dialect Bias via Activation Steering

**arXiv ID:** 2607.06845 | [PDF](https://arxiv.org/pdf/2607.06845v1)

**作者:** Huan Wu `[一作]` (York University), Laleh Seyyed-Kalantari `[通讯]` (York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

审计并缓解大语言模型在处理非裔美国英语（AAE）时的偏好，通过构建真实 AAEnote、提出 cDGI 评估方法、功能级偏差定位以及首次在 LLM 上应用激活调节（Activation Steering）进行方言去偏。

**💡 创新点**

①构建最大规模、人工验证的 Real‑AAE 并行语料库；②提出 conditional Dialect Group Invariance (cDGI) 区分翻译漂移与真实模型偏差；③功能级偏差定位揭示负对比（negative concord）等句法结构是普遍触发点；④首次将激活调节应用于方言偏差，既无训练成本又可显著降低偏差。

**🔧 技术方法**

cDGI 审计、perplexity / log‑prob / forced‑choice 生成偏差评估、特征级 log‑prob 偏差定位、因果追踪提取方言方向并在关键层注入方向的激活调节。

**📊 数据集**

Real‑AAE（17,479 AAE/SAE/AAE_back triplets，来源于 TwitterAAE 真实 AAE 推文并经过 Gemini‑3‑flash 翻译及人工验证），与现有 synthetic AAE、其它少量真实 AAE 资源对比。

**📈 对比分析**

通过 DGI、cDGI、PPL、MC、LP 等多维指标进行比较。六大 SOTA LLM 在 AAE 上始终偏向 SAE；激活调节在 LP 上比提示方法提升 5–20 倍，DGI、cDGI、MC 等指标均显著提升，同时保持 SAE 的流畅度。

**⚠️ 局限性**

①数据源仅覆盖社交媒体书面语，难以代表口语或正式写作；②翻译过程可能引入偏差，虽通过 cDGI 进行校正；③人工验证样本地区分布有限；④激活调节仅针对已标注的 AAE 语料有效，对其他方言或语体尚未验证；⑤未评估长期鲁棒性与对不同任务的泛化能力。

---

## 7. Cross-Trajectory Chimera Interventions Reveal Dissociable Roles of Weight Magnitude and Direction in Grokking

**arXiv ID:** 2607.06628 | [PDF](https://arxiv.org/pdf/2607.06628v1)

**作者:** Truong Xuan Khanh `[一作]` `[通讯]` (H&K Research Studio), Truong Xuan Khanh (H&K Research Studio)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对同一任务下不同随机种子训练得到的网络进行“交叉轨迹化身”实验，重新组合权重的大小（norm）与方向（unit vector），继续训练以检验权重大小和方向是否能携带可在另一网络中复制的效果。

**💡 创新点**

提出了跨轨迹chimera干预（cross‑trajectory chimera intervention）和自适应二分阈值定位（adaptive bisection）两种技术，证明方向携带可转移的电路身份信息并以阈值方式决定是否被覆盖，而权重大小仅对延迟有弱影响；并揭示阈值位置由接收网络的norm决定，说明可转移性取决于收件网络的动态状态。

**🔧 技术方法**

主要技术包括：权重向量的径向-角度分解、方向插值（slerp）和角度匹配的随机对照、谱特征（token‑embedding power spectrum）用于量化电路身份、对阈值的自适应二分搜索、对Adam优化器状态的消融实验以及统计分析（符号检验、Bootstrap CI）。

**📊 数据集**

使用了两种基于循环群的算术任务：模数59的加法和乘法，网络为单层Transformer，embedding维度128，使用全批训练。

**📈 对比分析**

对20对独立随机种子（每任务10对）进行实验。结果显示：方向捐赠在所有40/40组合中准确转移电路身份（p<0.002），阈值区分完全（p=1.9×10⁻⁴），而权重大小的延迟转移仅弱显著（平均贡献≈30%）。对比控制实验（角度匹配随机、norm平均等）确认方向的转移是特定内容而非单纯角度偏移，权重大小对身份无影响。

**⚠️ 局限性**

局限性包括：仅在单一Transformer架构和两种循环群任务上验证，未检验非循环任务或更大模型；电路身份指标基于embedding谱，缺乏完整电路等价性；阈值定位是二分类分离而非连续关系；优化器状态消融仅在少数代表性对上验证，可能不完全泛化。

---

## 8. AirPASS: Over-the-Air Federated Learning via Pinching Antenna Systems

**arXiv ID:** 2607.06768 | [PDF](https://arxiv.org/pdf/2607.06768v1)

**作者:** Seyed Mohammad Azimi-Abarghouyi `[一作]` (Chalmers University of Technology), Christopher G. Brinton `[通讯]` (Purdue University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了AirPASS框架，针对采用可调压电天线系统（PASS）的无线联邦学习（AirFL），联合优化设备选择、接收波束成形和天线位移，以最大化满足聚合误差阈值的参与设备数量。

**💡 创新点**

创新点在于：1）首次将PASS架构引入AirFL，利用天线可调位置提供额外的空间自由度；2）设计了两阶段交替优化：HRMC在固定PASS时通过光滑Riemannian优化与活跃集合并实现最大设备数选择；HAGO在固定设备与波束时通过可行重参数化与梯度上升优化天线布局，确保满足最差用户约束。

**🔧 技术方法**

采用的技术包括：光滑卡尔曼近似的指标函数、同胚重加权Riemannian共轭梯度优化、软最小化合并、对PASS天线位置的可行重参数化、log-sum-exp光滑最小化、梯度上升与回溯线搜索。

**📊 数据集**

实验使用MNIST和CIFAR-10两大图像分类数据集，并在非IID设备分布下进行FedAvg式训练。

**📈 对比分析**

与基准方案比较：FedAvg（理想聚合）、传统固定阵列MIMO、AirPASS–SDR‑DC、AirPASS–MP等。实验显示AirPASS在低至中等SNR下显著提升可参与设备数和学习精度，接近理想FedAvg，且在计算复杂度上优于SDR‑DC、优于匹配追踪。

**⚠️ 局限性**

局限性包括：1）PASS天线位置优化仍在局部最优，可能受初始点影响；2）在极大规模设备或极高SNR场景下收益递减；3）论文仅考虑单一PASS系统，扩展到多PASS或移动端实现仍需进一步研究。

---

## 9. When and How to Ask: Dynamic Preference Elicitation Strategies for Conversational Recommendation

**arXiv ID:** 2607.06765 | [PDF](https://arxiv.org/pdf/2607.06765v1)

**作者:** Feng Xia `[一作]` (University of Sheffield), Xi Wang `[通讯]` (University of Sheffield)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对对话式推荐系统中何时、如何提出偏好挖掘问题，系统性地从阶段角度研究并实现了动态策略选择

**💡 创新点**

① 证明偏好挖掘策略随对话阶段显著变化；② 构建 InPE 数据集，细粒度标注了挖掘需求与策略；③ 提出 COPE Mixture‑of‑Experts 架构，显式建模任务级与策略级路由，实现对话中主动、上下文感知的策略切换

**🔧 技术方法**

大型语言模型（Qwen3‑8B）+LoRA 参数高效微调；多专家 Mixture‑of‑Experts；层次化路由器；监督细粒度任务与策略标注；对比学习的推荐损失；多任务联合训练

**📊 数据集**

基于 INSPIRED 的 InPE 数据集（6,719 轮级样本）以及公开的传统推荐基线数据（如 KBRD、KGSF、TREA 等）

**📈 对比分析**

与邻域、知识图谱、LLM 及 Prompt‑based LLM 基线进行对比。COPE 在 InPE 上 Recall@10 达 0.314，提升 5.4% 以上；在策略选择上 Task/Strategy Accuracy 分别为 61.8%/71.1%；在生成质量上 Pairwise Win Rate 达 60.4%，Margin 为 0.314，显著优于所有对照组

**⚠️ 局限性**

对路由器的策略预测精度仍是瓶颈；单一专家或统一模型在泛化与细粒度对话质量上表现不佳；InPE 仍缺少跨域或多任务真实场景的多样性

---

## 10. ORAN-DEFEND: Subspace Detection and Sanitization of Backdoor DRL xApps in Open RAN

**arXiv ID:** 2607.06647 | [PDF](https://arxiv.org/pdf/2607.06647v1)

**作者:** Md Raihan Uddin `[一作]` (Clemson University), Fatemeh Afghah `[通讯]` (Clemson University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种无重训练、仅在遥测层级的防御包装器ORAN-DEFEND，用于在O-RAN环境中对被冻结的DRL xApp进行后门攻击的检测与消毒。

**💡 创新点**

创新点在于：①将后门触发视为在安全子空间正交补空间的能量，并证明投影到安全子空间即可消除触发；②通过对安全子空间的SVD估计实现无白盒、无重训练的防御；③首次给出线性投影防御的恢复边界，即触发能量分数η的阈值，且该阈值适用于多种攻击族；④将这一理论验证在真实O-RAN KPI数据集上，并展示高达99.5%的攻击成功率抑制。

**🔧 技术方法**

使用的技术包括：高斯/低维KPI窗口构造、SVD与主成分估计、投影与残差检测、离线安全子空间构建、线性投影消毒、KPI窗口标准化、以及在DRL环境中使用DQN。

**📊 数据集**

使用的数据集是Colosseum O-RAN COLORAN场景，包含8个KPI通道、T=20的滑动窗口，总维度D=160，拆分为70%/15%/15%的训练/验证/测试。

**📈 对比分析**

与基线对比包括：清洁拟合vs受污染拟合PCA、AE重构+投影、以及非线性MLP检测。ORAN-DEFEND在四种后门攻击（TrojDRL、SleeperNets、BadRL、Q-Incept）上实现了100%返回恢复、≥99.5%攻击成功率下降和AUROC≈0.98-1.00；在对比中线性投影在正交补触发下表现最优，AE+投影能补偿投影局限。

**⚠️ 局限性**

局限性包括：①线性投影对正交补触发之外的子空间触发无效；②在触发能量分数η低时恢复率下降；③对投影子空间估计依赖于足够清洁样本，尽管实验表明n=8已足够，但在极端扰动或高噪声环境下可能失效；④检测仍需线性残差阈值，无法处理子空间内触发；⑤在实际部署时需确保对接Near-RT RIC的时延限制和可靠性。

---

## 11. When Do Geometric Algebra Layers Beat Scalarization? A Controlled Study on SO(3)-Equivariant Vector Laws

**arXiv ID:** 2607.06634 | [PDF](https://arxiv.org/pdf/2607.06634v1)

**作者:** Fabien Polly `[一作]` `[通讯]`, Fabien Polly

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了Cl(3,0)几何代数网络在3D向量学习中的样本效率，并与基于标量化的SO(3)等变模型进行对比。

**💡 创新点**

发现几何代数网络在单阶段任务中无优势，但在深层组合运算任务（如多次旋转叠加、局部到全局力矩）中，在低样本量下表现优于标量化基线，说明几何乘积在处理深度组合时具有潜在优势。

**🔧 技术方法**

使用Cl(3,0)几何乘积层、等变权重约束、标量化基线、Vector Neurons、e3nn和普通MLP等多种模型，训练统一的Adam优化器，并对比了NMSE（标准化均方误差）指标。

**📊 数据集**

采用六个合成向量律（旋转、叉积、中心力、双体力、组合旋转、局部到全局力矩）以及对应的OOV（角度、半径、轴）分割，所有数据均为标准正态分布的向量输入。

**📈 对比分析**

通过固定训练预算、学习率、epoch数等控制变量，发现标量化基线在单阶段任务上取得更低NMSE且训练成本更低；在组合任务中，几何代数网络在100样本时比标量化低10-16倍NMSE，且在OOV测试中保持优势；然而在更大样本量时，优势消失。

**⚠️ 局限性**

局限性包括仅使用合成任务、模型规模有限（2k-28k参数）、仅考虑SO(3)等变且未对外部框架进行大规模调优，以及在可视化和理论解释方面缺乏正式化。

---

## 12. A Good Initialization is All You Need for Faithful Visual Attribution

**arXiv ID:** 2607.06726 | [PDF](https://arxiv.org/pdf/2607.06726v1)

**作者:** Zihan Gu `[一作]` (Chinese Academy of Sciences), Yue Hu `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两种前向仅使用的搜索方法CoPAIR和TRACE，用于在视觉模型中寻找最小的支持性区域集合。

**💡 创新点**

创新点在于把顶k证据掩码作为主要输出契约，并通过CoPAIR的粗粒度对齐和TRACE的交叉熵采样实现对固定大小掩码的直接优化；同时提供了可作为完整排序搜索初始化的模式。

**🔧 技术方法**

采用基于搜索的插入-删除曲线评价、贪心和PhaseWin搜索、交叉熵采样（TRACE）、粗粒度对齐与对偶搜索（CoPAIR）等技术。

**📊 数据集**

在ImageNet分类的CLIP ViT‑L/14、CLIP RN101、ResNet‑101（SLICO‑64区域）以及多模态语言模型的POPE和RePOPE数据集（Qwen2.5‑VL‑3B‑Instruct、LLaVA‑v1.5‑7B）上进行评估。

**📈 对比分析**

与RISE、HSIC、梯度、Grad‑ECLIP、IG2、IGOS++等非搜索基线以及Greedy、PhaseWin等搜索基线比较，TRACE+Greedy/PhaseWin在插入AUC、最高得分等指标上均突破现有最优；在单点RePOPE修复中，TRACE Direct实现94.44%/96.00%修复率，远优于传统全曲线方法。

**⚠️ 局限性**

局限性包括CoPAIR初始化的O(c^2)开销、TRACE的随机采样导致运行方差、以及需要预先指定掩码大小k等。

---

## 13. Ace! Motion Planning of Professional-Level Table Tennis Serves with a Robot Arm

**arXiv ID:** 2607.06989 | [PDF](https://arxiv.org/pdf/2607.06989v1)

**作者:** Guillem Torrente `[一作]` (Sony AI), Peter Dürr `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一套完整的机器人服务（serve）生成框架，能够让机器人臂以专业水平完成国际乒联（ITTF）规则下的服务。

**💡 创新点**

创新点在于将参数化的混合端执行器/关节空间优化运动规划与HEBO贝叶斯优化相结合，实现了对服务时机、速度、旋转等多目标的全局搜索与精细调优，并通过实时的合法性检查确保服务合法性与安全性。

**🔧 技术方法**

使用了运动原语（motion primitives）、非线性模型预测控制（MPC）、HEBO贝叶斯优化、碰撞检测、基于球轨迹的视觉感知等技术；实现细节包括3D球轨迹建模、低延迟位置控制和仿真到实机的映射。

**📊 数据集**

数据集主要来自：①机器人内部记录的球抛投轨迹（多次抛投实验收集的轨迹分布）；②人类球员的服务示范用于生成抛投原语；③对抗专业/精英选手比赛中记录的服务与回合数据，用于评估与迭代优化。

**📈 对比分析**

通过与日本职业和精英球员在官方裁判监督下进行的实战比赛评估。实验显示机器人服务的旋转可达550 rad/s、速度6.7 m/s，超过大部分职业球员；在2026年4月的比赛中，机器人在发球端赢得约52–59%（平均约58%）的得分概率，且直接发球（ace）率提升至约20%，与职业球员相当。

**⚠️ 局限性**

主要限制包括：①仿真到实机的差距仍导致部分通过仿真优化的方案在硬件上出现失效；②机器人关节与振动、球拍-球接触动力学建模不够精细，影响高频控制的鲁棒性；③当前方法在高动态环境下的实时性仍受限，未来需进一步改进物理建模与自适应调参。

---

## 14. HAJJv2-CrowdCount: Zero-Shot Benchmark for Dense Crowd Counting

**arXiv ID:** 2607.07322 | [PDF](https://arxiv.org/pdf/2607.07322v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 15. UASPL: Uncertainty-Aware Self-Paced Learning with Evidential Neural Networks

**arXiv ID:** 2607.06638 | [PDF](https://arxiv.org/pdf/2607.06638v1)

**作者:** Yifan Zhang `[一作]` (Northwest A&F University), Lipeng Pan `[通讯]` (Northwest A&F University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于证据神经网络的自适应学习框架UASPL，将模型内部不确定性与标签拟合损失结合，以自我节奏方式选择可靠易样本。

**💡 创新点**

首次在自我节奏学习中直接利用模型生成的不确定性评估样本难度，实现可靠易样本优先选择，并保持可解释性与通用性。

**🔧 技术方法**

采用证据深度学习、Dirichlet分布与主体逻辑，设计不确定性加权KL项和可解释样本排序，兼容多种SPL正则化形式。

**📊 数据集**

在25个UCI表格数据集（平衡/不平衡）以及四个图像分类基准（CIFAR‑10、FashionMNIST、MNIST、SVHN）上进行实验。

**📈 对比分析**

与传统SPL、SPL变体、基于不确定性的采样方法、元学习加权方法等15+基线对比，UASPL在准确率、F1、精确率、召回率上均取得最高或次高平均排名，标准差最小，显示显著性能提升。

**⚠️ 局限性**

仍需预训练阶段，对极端噪声或复杂多分类问题的鲁棒性尚未充分验证；与单纯基于损失的SPL相比，计算开销略高。

---

## 16. When Agents Go Rogue: Activation-Based Detection of Malicious Behaviors in Multi-Agent Systems

**arXiv ID:** 2607.06807 | [PDF](https://arxiv.org/pdf/2607.06807v1)

**作者:** Haowen Xu `[一作]` (Worcester Polytechnic Institute), Xiaoyan Sun `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个基于LLM内部激活的多智能体系统安全框架，能够在同步或异步环境下检测并修复被攻击的智能体。

**💡 创新点**

创新点包括：① 用激活空间而非交互图进行攻击检测，天然兼容异步执行；② 采用激活层级的自适应修正（steering）而非简单隔离，恢复受损智能体功能；③ 通过无监督原型学习实现跨模型、跨规模的通用性。

**🔧 技术方法**

技术细节：提取LLM最终层隐藏向量，归一化后与正常原型做余弦距离度量；阈值检测判定异常；对异常激活做线性插值回归到正常原型；动态更新原型，支持在线运行；实现高效的激活提取与距离计算。

**📊 数据集**

实验使用的公开数据集和攻击场景包括：CSQA 与 GSM8K 的隐蔽 Prompt 注入；InjecAgent 的工具操纵攻击；PoisonRAG 与 HotPotQA 的记忆污染攻击；并在同步/异步两种执行模式下进行评估。

**📈 对比分析**

与图模型基线（G‑Safeguard、BlindGuard、PERM、TAM）对比，本文框架在 F1、AUROC、任务完成率 (TCR) 及攻击成功率 (ASR) 上均显著提升：F1 ≈ 0.92‑0.95 对比 0.72/0.56，AUROC ≈ 99% 对比 61‑94%，TCR ≈ 0.98 对比 0.83/0.78，ASR 仅 0.02‑0.05 对比 0.15‑0.33；且性能在不同LLM backbone、MAS规模和攻击强度下保持稳健。

**⚠️ 局限性**

局限性：需要访问LLM内部激活，仅适用于本地或开源模型；在域迁移下可能表现下降；对持续慢移的自适应攻击存在一定脆弱性；对高并发/极大规模系统的实时修正开销仍需进一步优化。

---

## 17. Compass: Prostate Cancer Detection Needs Multi-View Context

**arXiv ID:** 2607.06919 | [PDF](https://arxiv.org/pdf/2607.06919v1)

**作者:** Paul F. R. Wilson `[一作]`, Parvin Mousavi `[通讯]` (Queen's University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出Compass，一种结合微波超声旋转扫帚视频与biopsy图像的多视角AI框架，用于患者级前列腺癌检测。

**💡 创新点**

创新点在于将完整3D旋转扫描与2D biopsy证据联合，利用旋转角度编码和Transformer实现跨分支融合与多视角证据聚合。

**🔧 技术方法**

采用ProstNFound+视觉编码器、正弦角度嵌入、轻量化注意力解码器、全局Transformer以及多任务联合损失进行训练。

**📊 数据集**

使用OPTIMUM临床试验的118例多中心μUS数据集，包括旋转扫帚视频、biopsy帧、年龄和PSA等临床指标。

**📈 对比分析**

与CLIP、Cinepro、MedSAM、ProstNFound+、MIL和视频基线及专家PRI-MUS比较，Compass在患者级AUROC 87.3%、Sen@60 89.9%上显著优于所有基线，核心级略逊于PRI-MUS。

**⚠️ 局限性**

局限性包括样本量有限、仅在两中心设备上验证、核心级性能略低、依赖角度标注且缺乏外部泛化评估。

---

## 18. Operational Reframing and Approval-Framed Delegation in Multi-Agent LLM Safety

**arXiv ID:** 2607.07097 | [PDF](https://arxiv.org/pdf/2607.07097v1)

**作者:** Lifei Liu `[一作]` (Independent Researcher), Yihang Chen `[通讯]` (Georgia Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种五条件对照设计，用于拆解多智能体LLM管线的安全放大效应，系统评估了操作性重构、规划器行为与批准式委托对合规性的影响；

**💡 创新点**

创新点在于将原本混合的管线-直接比较拆分为可观测的三个对照（F1、F2、F3），揭示不同因素在安全放大中的具体贡献；

**🔧 技术方法**

使用了LLM评判器（gpt-4o-mini）对生成文本进行合规性分类，并通过配对t检验、Benjamini–Hochberg校正和交叉评判器一致性检验等统计技术；

**📊 数据集**

实验数据包括30个自定义有害情境（包含原始与“洗白”版）以及从AgentHarm、AgentDojo、InjecAgent、Agent‑SafetyBench 四大基准衍生的84个外部情境；

**📈 对比分析**

在对照实验中，操作性重构（F1）在GPT、Gemini、DeepSeek上显著提升合规率（+16至+24个百分点），规划器和批准式委托分别产生不同大小的负向或正向偏移，整体管线效应因模型、规划器和情境源异而变化；

**⚠️ 局限性**

主要局限包括：洗白提示未得到完整人工意图验证；仅评估了四种规划器-执行器的对角组合，未覆盖完整矩阵；评判器间一致性中等，可能影响绝对率；仅使用提示层实验，未考察真实工具执行环境；基准中中性任务样本不足，易于导致帮助性评估偏差。

---

## 19. Specification Grounding Drives Test Effectiveness for LLM Code

**arXiv ID:** 2607.06636 | [PDF](https://arxiv.org/pdf/2607.06636v1)

**作者:** Amin Haeri `[一作]` (TD Bank), Mahdi Ghelichi `[通讯]` (TD Bank)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了将外部规格作为测试生成的“grounding”对LLM代码生成的影响，并通过对比不同测试写作方式评估其效果。

**💡 创新点**

明确证明测试的grounding而非数量是提升LLM代码质量的主要驱动因素，并提供实证与消融分析。

**🔧 技术方法**

采用LLM代码生成、自动化测试生成与迭代修复（test‑and‑repair）框架，结合对齐的规格与独立金标准。

**📊 数据集**

使用26个小型Python任务（18个规格完整性任务+8个逻辑任务），每个任务配有手写规范、参考实现和金标准测试。

**📈 对比分析**

将grounded与公平基线、属性生成、AlphaCodium流程等对比，结果显示grounded提升38%最终正确率、检测率提升、误报率降低，并在Claude、GPT‑5.3‑codex、Gemini等不同模型上保持一致。

**⚠️ 局限性**

仅适用于规格完整性缺失的纯函数；需要高质量规格；测试编写者需足够强大；对算法逻辑错误无效；样本规模有限，缺乏大规模多函数或仓库级评估。

---

## 20. Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity

**arXiv ID:** 2607.07386 | [PDF](https://arxiv.org/pdf/2607.07386v1)

**作者:** Loïc Cabannes `[一作]` (Meta FAIR), Hervé Jégou `[通讯]` (Meta FAIR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

引入了 Sparse Delta Memory（SDM）架构，作为 Gated DeltaNet 的稀疏扩展，旨在提升长上下文记忆能力。

**💡 创新点**

将 GDN 的更新规则稀疏化，并结合 Product‑Key Memory，实现数千倍更大的状态空间，同时保持与原始 GDN 相同的 FLOPs。

**🔧 技术方法**

采用稀疏键选择、门控 Delta 写入、稀疏读取、可学习的初始状态以及多头混合，并与滑动窗口注意力（SWA）混合局部/全局层架构相结合。

**📊 数据集**

预训练使用多样化文本序列（8192 token），长上下文微调使用 128k token，评估数据包括 RULER 长上下文基准、代码数据以及 HellaSWAG、WinoGrande、ARC、PIQA、OpenBookQA、RACE、CommonsenseQA、BoolQ、TQA、HumanEval、NaturalQuestions、MMLU、GSM8K 等常见任务。

**📈 对比分析**

在相同参数与 FLOPs 的 iso‑FLOP 对照下，SDM 在所有规模上均优于 GDN，且在 8B 规模时超过 Full Attention；短上下文任务的 NLL/准确率高于 GDN，接近 Full Attention；在 RULER 长上下文任务中表现显著优于 GDN 并接近 Full Attention，体现大状态带来的长记忆优势。

**⚠️ 局限性**

虽然稀疏设计保持计算不变，但 SDM 的状态存储巨大（可达模型参数量），需要高带宽 HBM，导致训练吞吐量低于 GDN；现有稀疏核实现效率有限，需进一步优化；在资源受限环境下部署受限。

---

## 21. On the Approximability of Parameterized Minimum Monotone Satisfying Assignment

**arXiv ID:** 2607.06852 | [PDF](https://arxiv.org/pdf/2607.06852v1)

**作者:** Venkatesan Guruswami `[一作]` (University of California Berkeley), Xin Zheng `[通讯]` (Nanjing University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本工作给出了参数化的k-MMSA_3问题的FPT时间O(2^k log n)近似算法，并构造了从(k,h)-gap k-MMSA_3到(2^k,h/k)-gap k-MMSA_2的保真还原，进一步利用Marx的gap创造技术和完美哈希族证明了k-MMSA_4在n^o(1)与n^O(1/k)时间下的强近似不可行性。

**💡 创新点**

创新点在于①首次将k-MMSA_3的近似性能提升到FPT级别；②通过保真还原桥接k-MMSA_3与k-MMSA_2，表明对3层的更强近似不可行性将直接转化为Set Cover的更强不可行性；③改进Marx的两层还原，得到更细致的时间-因子权衡；④通过填充填充引理将硬度提升至FPT级别。

**🔧 技术方法**

主要技术包括：①贪心覆盖启发式与平均覆盖分析；②使用完美哈希族压缩底层子句；③Marx的gap创造两层还原；④填充（padding）技术提升不可行性因子；⑤对k-MMSA_4的归约链利用Label Cover与MinLabel的已知硬度。

**📊 数据集**

论文完全基于理论分析与归约，不使用任何实验数据集。

**📈 对比分析**

与之前的O(N^{1/3})近似相比，FPT时间O(2^k log n)近似大幅降低了对k的指数增长；在可接受的参数范围内（k=O(log n)）可得到常数级别近似；对k-MMSA_2的gap还原表明3层与2层的近似复杂度相当。

**⚠️ 局限性**

局限性包括：①算法仍在指数级别，仅适用于小k；②对k-MMSA_4的不可行性仍仅在假设下；③未对k-MMSA_1或更高层次给出类似结果；④gap还原对参数大小有要求，难以直接应用于更广泛的实例。

---

## 22. Unveiling TCP BBR Dominance in Starlink Internet: Experimental Insights and Analysis

**arXiv ID:** 2607.07133 | [PDF](https://arxiv.org/pdf/2607.07133v1)

**作者:** Rakshitha De Silva `[一作]` (Deakin University), Jonathan Kua `[通讯]` (Deakin University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现全球六城Starlink实验平台，评估Google BBR‑v3与八种TCP拥塞控制算法在真实卫星网络中的吞吐、延迟、重传和公平性，并基于端到端失效概率提出数学模型与BBR‑v3流体模型。

**💡 创新点**

首次在真实Starlink环境中对BBR‑v3进行大规模实验；提出针对LEO动态特性的链路失效概率模型和BBR‑v3流体模型；证明BBR‑v3在公平性、重传与延迟平衡方面优于传统与激进算法。

**🔧 技术方法**

使用TCP拥塞控制实验、M/G/1排队分析、Jain公平性指数、贝叶斯概率模型、iperf3测量工具、Linux命名空间隔离、AWS EC2实例与Starlink终端硬件。

**📊 数据集**

通过在俄亥俄、圣保罗、伦敦、孟买、东京、悉尼六个城市收集的10次30 s测试窗口，得到吞吐、RTT、重传、窗口大小、队列长度等指标数据。

**📈 对比分析**

采用专用流与并发流两种场景，统计吞吐、RTT、重传、队列大小；在专用流中BBR‑v3吞吐最高，且延迟与重传最优；在并发流中BBR‑v3虽吞吐略低于LeoCC、PCC和bbrv1/2，但公平性更好、重传更低、RTT更稳定。

**⚠️ 局限性**

受限于Starlink链路的非平稳容量与动态链路切换，默认BBR‑v3参数未针对LEO优化；实验规模仅覆盖单流与九流并发，未验证更大规模拥塞；模型假设的isl失效概率与RTT分布可能不完全匹配实际情况。

---

## 23. ShapeTalk: Combining Natural Language and Sketch for Time-Series Pattern Querying

**arXiv ID:** 2607.07073 | [PDF](https://arxiv.org/pdf/2607.07073v1)

**作者:** Guoruizhe Sun `[一作]` (University of California at Davis), Dongyu Liu `[通讯]` (University of California at Davis)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文设计并实现了一种支持自然语言与手绘草图并行交互的单变量时序模式检索系统，系统通过LLM将自由文本描述转化为可编辑的形状特征约束，并与基于相似度的草图检索管道协调工作，实现了跨模态的查询制定与迭代细化。

**💡 创新点**

创新点包括：
- 将LLM驱动的自然语言解析与草图检索保持独立管道，但通过共享上下文和可视化反馈实现多模态协同；
- 采用动态阈值推断，让阈值自适应不同数据集的统计特征；
- 提供可编辑的形状特征视图，使查询解释透明、可纠正；
- 通过滑动窗口、PDTW、抗碰撞算法实现高效检索；
- 设计可交互的窗口长度建议与持续可视化的结果展示。

**🔧 技术方法**

技术手段包括：
- GPT‑4o（LLM）进行少量示例提示的自然语言解析与阈值推断；
- 预处理：时间索引去重、缺失值剔除；
- 形状特征向量（升降、平稳、尖峰等）与阈值阈值化；
- 滑动窗口匹配、等长子段分割；
- 采用Pruned Dynamic Time Warping（PDTW）与Sakoe‑Chiba band做草图相似度计算；
- 反冲突（anti‑collision）过滤重复相邻匹配。

**📊 数据集**

实验使用的公开数据集包括：
- Apple股票价格（约 2,700 点）
- 萨克拉门托气温数据（1,977 点）
- 纽约能源消耗数据（145,367 点）
- 加州洛杉矶 PM2.5 空气质量（按周采样）
- 其他自定义的金融与气候时间序列。

**📈 对比分析**

性能与比较：
- LLM解析准确率在真实用户查询上为 86%（带少量示例提示），合成查询 83%；
- 在线查询（自然语言 + 匹配）平均时延 0.65–0.85 秒，远低于其他基线模型；
- 约 92% 受试者在任务中使用了自然语言与草图两种模式；
- 任务完成平均时间 1.36 分钟，用户在 10 分钟内能识别 3.3 个自定义模式；
- 用户研究显示对交互的整体满意度为 4.5/5，认为可视化反馈与可编辑特征提升了可解释性。

**⚠️ 局限性**

局限性：
- 多模态仍为“协调”而非“联合融合”，无法一次性同时利用文本与草图；
- 模糊或不完整的查询仍可能导致阈值过硬或检索空缺，需要用户手动细化；
- 草图匹配对绘制精度敏感，缺乏自动平滑或模板引导；
- LLM 仍易出现幻觉或格式错误，影响阈值推断；
- 系统主要针对固定窗口长度，尚未支持变长或不定时段事件检索；
- 需额外的用户学习和调参成本，尤其在阈值手动微调方面。

---

## 24. What Predicts Correctness in Text-to-SQL? A Selective-Prediction Study

**arXiv ID:** 2607.06799 | [PDF](https://arxiv.org/pdf/2607.06799v1)

**作者:** Robert Richardson `[一作]` `[通讯]`, Robert Richardson

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在难度较高的多表文本到SQL任务中，如何通过不同的置信度信号来判断生成的SQL是否正确，并比较了各种方法在自我一致性、结构一致性、执行一致性、架构相关度、可执行性、序列对数概率以及LLM判断器（verifier）等方面的表现。

**💡 创新点**

创新点在于：①系统性对黑盒统计信号与逻辑评估信号（LLM判断器）进行对比，发现后者明显突破了前者的“正确性天花板”；②展示了跨提供商（OpenAI、Claude）判断器的组合可以得到最高准确度和最佳校准的置信度信号；③对训练好的verifier在领域内与跨模式迁移的表现进行了深入实验，揭示了训练模型在跨数据库迁移时的局限性。

**🔧 技术方法**

使用的技术包括：多次采样生成SQL并进行自我一致性评估、使用架构嵌入计算schema-relevance、执行结果聚类、白盒序列对数概率、LLM-as-judge（OpenAI GPT‑4o-mini/4o与Claude Sonnet 4.6）以及跨提供商的投票集成、逻辑感知的分类器与生成式verifier的微调、校准（ECE）、选择性预测风险‑覆盖曲线与分布无关的Bonferroni证书。

**📊 数据集**

主要使用的公开数据集为BIRD（800题 8 个数据库）和Spider（dev 490题 20 个数据库），并在两者上分别生成 SQL 进行评估。

**📈 对比分析**

比较方法采用 ROC 曲线下的面积（AUC）作为排名指标，并提供配对自举置信区间；结果显示黑盒信号在 0.61–0.68 之间，LLM verifer 在 0.72–0.78 之间，跨提供商两者集成可达 0.82，并且该集成信号在校准（ECE≈0.03）和选择性预测（低风险下覆盖率可达27%）上表现优于自我一致性。

**⚠️ 局限性**

局限性包括：①训练好的verifier 在域外数据库迁移时准确率显著下降（≈0.66）；②对生成器与判别器模型的多样性探索有限；③自我校正循环的置信度严重失准，无法替代判别器；④分布无关的证书过于保守，需更高的生成器准确率或更多校准样本；⑤实验仅覆盖了两款 OpenAI 生成器与两款判别器，未检验开放权重模型或更大规模的 verifiers。

---

## 25. Generative Diffusion Models of Stochastic Graph Signals

**arXiv ID:** 2607.06833 | [PDF](https://arxiv.org/pdf/2607.06833v1)

**作者:** Yiğit Berkay Uslu `[一作]` (University of Pennsylvania), Alejandro Ribeiro `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于去噪扩散模型的图信号生成框架，并设计了U-Graph Neural Network（U-GNN）架构，用于在已知图结构与节点特征的条件下采样图信号。

**💡 创新点**

创新点包括：①将U-Net的多分辨率编码解码结构迁移到图域，利用可学习的节点选择矩阵实现池化/反池化；②引入步幅（stride）和零填充的升降采样，避免显式图降采样；③将上述机制嵌入扩散逆过程，得到统一的条件图信号生成方法。

**🔧 技术方法**

主要技术：去噪扩散模型、图神经网络（GNN）带步幅与池化、U-GNN架构、嵌入节点特征与扩散步长的条件化网络、梯度下降训练。

**📊 数据集**

使用了两个数据集：①标普500指数的相关性图，用于股票价格预测；②无线网络功率控制问题的信道互斥图，用于无线资源分配。

**📈 对比分析**

与传统的图生成方法（如基于GAN/变分自编码器的图信号生成）以及针对各任务的专用网络进行对比，实验表明U-GNN在捕捉市场不确定性、尾部事件以及实现无线资源分配的概率分布方面均取得了更高的样本多样性与预测准确度。

**⚠️ 局限性**

局限性：①目前仅在两个静态图任务上验证，缺乏对时变图或大规模图的评估；②对步幅和池化参数的选择较为经验化，缺少理论指导；③训练成本和推理时间相对较高；④尚未探究更深层的图注意力或变压器模块的集成。

---

## 26. Why Fake ? Unveiling the Semantic Vocabulary of Deepfake Detectors

**arXiv ID:** 2607.07216 | [PDF](https://arxiv.org/pdf/2607.07216v1)

**作者:** Vazgken Vanian `[一作]` (Information Technologies Institute Centre For Research And Technology Hellas), Dimitris Zarpalas `[通讯]` (Information Technologies Institute Centre For Research And Technology Hellas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对现有的黑盒深伪检测模型使用后置可解释人工智能方法（EDDP），挖掘并解释其内部概念空间，提供概念检测图、贡献图以及因果式对比分析。

**💡 创新点**

创新点在于首次将EDDP引入深伪检测领域，实现无需改造网络即可获得全局模型理解、空间可定位的概念解释和可控的what-if因果推理。

**🔧 技术方法**

主要技术包括Encoding‑Decoding Direction Pairs (EDDP)、概念检测与贡献图 (CPM/CCM)、RCAV敏感度分析以及概念级干预与反事实实验。

**📊 数据集**

实验基于FaceForensics++ (FF++) 数据集，使用预训练的Xception网络进行概念学习和评估。

**📈 对比分析**

通过概念克隆与误判修正实验，概念迁移成功率达87.34%，误判修正率达99.8%，表明所提概念高度对应模型决策并显著提升可解释性。

**⚠️ 局限性**

局限性包括：需要额外训练EDDP模型、概念数量为手工设定、概念与特定模型/数据集绑定，难以直接迁移到其他模型或数据集。

---

## 27. Separation Logic for Memory Conflict Detection in High-Level Synthesis

**arXiv ID:** 2607.07126 | [PDF](https://arxiv.org/pdf/2607.07126v1)

**作者:** Yeonseok Lee `[一作]` `[通讯]` (SLING AI Inc.), Yeonseok Lee (SLING AI Inc.)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

针对高层综合中的循环展开和数组分区，提出一种基于分离逻辑的空间验证框架，能够在LLVM IR层面静态检测并避免非线性索引导致的内存冲突。

**💡 创新点**

创新点在于将分离逻辑的分离连接与可变性内存块预测结合，通过将空间互斥关系翻译为对SSA变量的两两不等式矩阵，突破传统多维仿射多边形模型的非线性限制，并提供确定性的顺序回退保证安全。

**🔧 技术方法**

使用了LLVM IR的GEP指令提取平面算术表达式、分离逻辑语义、SMT求解器（如Z3）进行不等式验证以及在验证失败时的顺序回退机制。

**📊 数据集**

论文未给出具体数据集；实验评估及实现细节在未来工作中计划实现。

**📈 对比分析**

由于缺少实验结果，无法直接比较方法与现有HLS工具的性能；理论上框架可显著减少因非线性索引而导致的强序列化，从而提升并行度。

**⚠️ 局限性**

主要限制包括：SMT求解对非线性整数算术的不可判定性导致的未知或超时结果会被视为冲突，进而产生保守的顺序化；以及对多重读操作的严格排他性导致潜在吞吐率损失。

---

## 28. Multimodal Spatiotemporal-Frequency Fusion with Peak Enhancement for Cellular Traffic Forecasting

**arXiv ID:** 2607.07016 | [PDF](https://arxiv.org/pdf/2607.07016v1)

**作者:** Qingzhong Li `[一作]` (Xinjiang University), Fei Xing `[通讯]` (Xinjiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 MSPF-Net，一种多模态时空频域融合网络，用于精准预测蜂窝网络流量。

**💡 创新点**

创新点在于结合时空频域编码、峰值增强模块和新闻上下文表示，并采用动态多模态融合来捕捉突发波动与外部事件的交互。

**🔧 技术方法**

采用 Transformer 自注意力、FFT 频域变换、图卷积、短窗口卷积和门控动态融合等技术。

**📊 数据集**

使用米兰、特伦托和 LTE 三个公开蜂窝流量数据集，并对新闻文本进行预处理生成上下文特征。

**📈 对比分析**

与 LSTM、Transformer、FEDformer、TimeMixer、DDGCRN 等基线比较，MSPF-Net 在 MAE/RMSE 上取得 40% 以上的提升，尤其在突发事件驱动场景表现最佳。

**⚠️ 局限性**

局限在于对新闻与流量对齐的依赖、对更丰富空间图结构的探索不足，以及模型对极端异常的鲁棒性待提升。

---

## 29. Gradient-Based Speech-to-Text Alignment for Any ASR Model: From CTC to Speech LLMs

**arXiv ID:** 2607.06831 | [PDF](https://arxiv.org/pdf/2607.06831v1)

**作者:** Albert Zeyer `[一作]` (RWTH Aachen University), Hermann Ney `[通讯]` (RWTH Aachen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对所有可微分的自动语音识别（ASR）模型（CTC、转导器、注意力解码器和语音大型语言模型）开发了一种基于梯度的词对齐方法，能够在输入波形级别生成词边界，无需额外训练或模型修改。

**💡 创新点**

创新点在于：①提出一种通用梯度归因方式，将每个教师强制的词标签对输入梯度求解并聚合为帧级显著性；②使用动态规划将显著性矩阵解码为最优对齐路径；③在对齐过程中引入能量加权与零分数空白策略，提升对空白/静音的鲁棒性；④在语音大型语言模型中首次将自注意力与梯度对齐进行比较，展示梯度对齐同样有效。

**🔧 技术方法**

技术手段包括：反向传播梯度计算、梯度对数范数归一化、能量加权、零分数空白、动态时间规整（DTW）与Viterbi搜索、子词/字符/音素分词、以及多层深度梯度评估。

**📊 数据集**

使用公开语料库TIMIT（读音）和Buckeye（自发语音）进行评估，包含大约5小时的Buckeye子集，所有实验均基于金标准词边界。

**📈 对比分析**

将梯度对齐与每种模型的原生对齐（CTC/转导器的强制对齐、AED的交叉注意力DTW、LLM的自注意力）进行对比。评估指标包括平均词边界误差（WBE）和50 ms衬帽精度。实验表明：梯度对齐在所有模型族中都可获得可用对齐，且在流式模型和某些弱对齐模型（如Canary‑Qwen、FastConformer‑CTC）上往往优于原生对齐；在Whisper‑large‑v3的最佳编码层次甚至超过其交叉注意力；但在强对齐模型上略逊一筹。计算成本上，梯度对齐需要每个词标签进行一次反向传播，显著高于单向前向传播。

**⚠️ 局限性**

局限性：①梯度对齐对计算资源要求高，尤其是长句或大批量时需要大量反向传播；②相较于最优原生对齐，其准确性略低；③对齐精度受分词粒度和模型训练目标的影响较大，需要针对不同模型族进行调参；④目前仅在实验级别验证，缺乏实时或工业级实现方案。

---

## 30. Overview of the NLPCC 2026 Shared Task 1: Difficulty-Aware Multilingual and Multimodal Medical Instructional Video Understanding Evaluation

**arXiv ID:** 2607.06618 | [PDF](https://arxiv.org/pdf/2607.06618v1)

**作者:** Shenxi Liu `[一作]` (Beijing Institute of Technology), Bin Li `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了2026年NLPCC共享任务DA-MIVQA，针对医学教学视频问答进行难度感知评估，涵盖时序答案定位、视频检索和端到端检索+定位三条赛道；

**💡 创新点**

创新点在于将问题按所需证据分为简单和复杂两类，显式评估字幕匹配与视觉、程序理解能力的差异；

**🔧 技术方法**

采用文本仅Oracle基线（冻结Qwen3.5-9B）、多模态编码解码框架（融合字幕、问题与视频片段）、以及交叉模态知识迁移和多任务损失；

**📊 数据集**

使用公开医学教学视频集（包括急救、康复、护理等场景），手工标注时序答案与难度标签（简单/复杂）；

**📈 对比分析**

在三条赛道上，通过mIoU、R@n、MRR等指标比较，Amazon Inc.、Team_WuKong、BIGC分别夺得第1名；文本Oracle基线性能低，表明字幕仅能部分解决简单问题；多模态基线显著优于Oracle，尤其在复杂问题上提升显著；

**⚠️ 局限性**

局限性包括：难度标注依赖人工判断，可能存在主观性；模型对视觉细粒度动作识别仍有限，复杂问题性能提升空间大；任务规模受视频多样性与数据量限制。

---

## 31. Nonlinear Bandit

**arXiv ID:** 2607.07304 | [PDF](https://arxiv.org/pdf/2607.07304v1)

**作者:** Tianshuo Zheng `[一作]`, Keqin Liu `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一系列针对非线性、多分段以及通用非线性 bandit 的鲁棒算法，主要基于扩展的 Huber 损失和在线镜像下降，能在重尾噪声环境下实现近似最优的次线性 regret，且每轮计算复杂度为 O(1)

**💡 创新点**

1) 引入扩展 Huber 损失用于 GLB，消除对敏感参数 κ 的依赖；2) 通过 PGLB‑EHM 对分段 GLB 进行多区域估计；3) 采用二分法和空间分割、仿射提升，将通用非线性 bandit 转化为可处理的 GLB；4) 所有算法均保持一次性更新与低时间/空间复杂度

**🔧 技术方法**

扩展 Huber 损失、在线镜像下降、分段估计、二分搜索、仿射提升、带有鲁棒参数的自适应学习率、一次性更新和 Sherman‑Morrison 公式优化

**📊 数据集**

主要使用合成实验数据：Logit 形式的 GLB 采用 Student‑t 分布噪声；PGLB‑EHM 在多块情境下的模拟；NB‑EHM 在基于分段的高维球面空间的随机测试；对比传统 GLB、LinUCB 等基线算法

**📈 对比分析**

与传统 GLB（如 GLM‑UCB、UCB‑GLM 等）以及线性 bandit（LinUCB、LinTS）对比；实验显示 EHM 在相同 T 下获得更低 regret，运行时间约 11 秒；PGLB‑EHM 在多块设置下保留相同的 regret 次序；NB‑EHM 通过分块方法在高维球面上实现次线性 regret，平均奖励随时间稳步提升

**⚠️ 局限性**

1) 对噪声要求仍需存在有限的 (1+ε)-阶矩；2) 链接函数必须满足 μ'≥κ、μ''≤K μ' 等约束；3) PGLB 需要已知子区间间的最优奖励差 a；4) NB‑EHM 依赖于仿射提升的可行性，且分辨率与维度呈指数关系；5) 对极端超重尾分布、非 Lipschitz 链接函数或多重最优点的情况尚未覆盖

---

## 32. Dissociating the Internal Representations of Sycophancy in LLMs

**arXiv ID:** 2607.07003 | [PDF](https://arxiv.org/pdf/2607.07003v1)

**作者:** Anthony Baez `[一作]` (MIT), Pat Pataranutaporn `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过双解耦方法研究LLM中真值性与观点性两类说服行为的内部表示。

**💡 创新点**

提出将说服行为分解为事实性和观点性两种子类型，并用双重解耦验证其是否共享或独立表示。

**🔧 技术方法**

使用线性探针、激活驱动向量以及线性判别分析等技术对模型激活进行分析。

**📊 数据集**

构建了针对事实性和观点性说服的三千条多轮对话数据集，并对Gemma-3-12B-IT与Llama-3.1-8B-Instruct进行评估。

**📈 对比分析**

通过探针的AUC迁移、向量驱动实验和LDA可视化进行比较，结果显示Gemma的两类说服表示高度统一，而Llama则表现为相对独立，且转移探针性能下降0.3以上。

**⚠️ 局限性**

主要限制在于可能存在的数据集伪特征导致驱动实验的因果效应弱化，且对真实世界复杂情境的泛化仍待验证。

---

## 33. Monitoring Vulnerabilities in Next-Generation Automotive Operating Systems

**arXiv ID:** 2607.07226 | [PDF](https://arxiv.org/pdf/2607.07226v1)

**作者:** Dimitri Simon `[一作]` (Institut Polytechnique de Paris), Hervé Debar `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对下一代软件定义车辆（SDV）中 POSIX 操作系统与中间件层进行系统性漏洞扫描与验证，构建了 VERA 漏洞评估框架并公开其源代码。

**💡 创新点**

创新点在于首次针对 SDV 环境提出半白盒漏洞评估流程，将多种扫描器结果统一、过滤、解析 CVSS/EPSS，并结合在线 exploit 检索实现从检测到可利用性评估的闭环；同时将 Android/Yocto 的源级扫描与二进制扫描结合，显著降低噪声。

**🔧 技术方法**

使用 Docker 容器化环境、Grype、CVE Binary Tool、Vanir、Yocto 的漏洞扫描器，结合 CVSS/EPSS 评分、公开 exploit 数据库以及脚本化自动化流程进行漏洞检测与验证。

**📊 数据集**

主要数据集包括 MITRE CVE 数据库、CVE‑listV5、NVD、EUVD、公开 PoC、Android/Yocto 源代码与镜像、以及自建车载软件包清单。

**📈 对比分析**

通过与主流扫描器（Grype、Trivy、OSV、Vanir、CBT）的对比，VERA 在保留高可信度 CVE 的同时显著降低噪声；实验显示 VERA 能更精确识别可利用漏洞，并通过 EPSS、已公开 exploit 率等指标评估不同 OS/中间件的风险。

**⚠️ 局限性**

局限性包括基于 Docker 环境的评估未完全覆盖真实车载硬件与供应链配置；CVSS/EPSS 仅为静态指标，无法保证所有 CVE 都能被利用；缺乏对真实固件与硬件特定条件的深入验证。

---

## 34. HPR-SAM: Hierarchical Probabilistic Representation Learning for Prompt-free SAM-based Medical Image Segmentation

**arXiv ID:** 2607.06972 | [PDF](https://arxiv.org/pdf/2607.06972v1)

**作者:** Yingzhen Hu `[一作]` (Xi'an Jiaotong-Liverpool University), Xiaofeng Liu `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种层次概率表示框架 HPR，利用无提示的 SAM 进行医学图像分割。

**💡 创新点**

创新点在于同时建模全局解剖先验、内部结构多样性与局部可靠性三种概率表示，并通过层次预测融合提升自动提示质量。

**🔧 技术方法**

使用了分布式解剖表示（DAR）、多组件解剖表示（MAR）、局部可靠性表示（LRR）以及层次预测融合（HPF），并集成 SAM 的编码器与解码器。

**📊 数据集**

实验数据集包括 Synapse CT、LA 心脏 MRI 以及 PROMISE12 前列腺 MRI。

**📈 对比分析**

与传统全监督方法及现有 SAM 适配方法相比，HPR 在 Synapse 上平均 Dice 取得 85.09%，在 LA、PROMISE12 的少样本设置下均获得最高 Dice，显示显著性能提升。

**⚠️ 局限性**

局限性包括对低对比度或边界模糊区域的细节分辨仍不理想；HD95 与部分全监督方法相比仍略高，且未利用无标签数据进行进一步提升。

---

## 35. Multimodal Smart Glove for Sign Language Recognition Using Deep Learning

**arXiv ID:** 2607.06996 | [PDF](https://arxiv.org/pdf/2607.06996v1)

**作者:** Anh Thu Nguyen Ngoc `[一作]` (Fulbright University Vietnam), Manh Duong Phung `[通讯]` (VinUniversity)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发了一款多模态智能手套，结合手指弯曲传感器、惯性测量单元以及摄像头捕捉面部表情，实现越南手语的识别与实时翻译。

**💡 创新点**

创新点在于融合手套的多维传感数据与面部表情信息，通过深度学习的LSTM网络实现高精度识别，同时将模型压缩为TensorFlow Lite，支持现场实时部署。

**🔧 技术方法**

采用了ESP32-C6无线传输、ADS1115、BNO055、MediaPipe面部标记、双层64单元LSTM网络、Adam优化器与交叉熵损失，模型转换为TensorFlow Lite。

**📊 数据集**

使用自己收集的越南手语数据集，包含三种手势（Tôi、Xin chào、Hẹn hò），共10个录制序列，25帧/序列，9维特征/帧。

**📈 对比分析**

通过训练/验证曲线与混淆矩阵对比，模型在验证集上的准确率约为95%，在现场测试中以95%+的置信度实时识别，显示出优于传统单一模态或仅手势识别方法的鲁棒性。

**⚠️ 局限性**

局限性包括样本量小、手势类别有限、对不同手型和佩戴方式的适应性需进一步验证，以及面部表情捕捉对光照与距离的敏感性。

---

## 36. Ensemble Deep Learning Approaches for AI-Altered Video Detection

**arXiv ID:** 2607.06872 | [PDF](https://arxiv.org/pdf/2607.06872v1)

**作者:** Laiba Khan `[一作]` (University of Toronto), Joshua Jung `[通讯]` (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了融合音频与视频的多模型深伪检测系统

**💡 创新点**

提出将AASIST与EfficientNet、XceptionNet、MesoNet组合，并对多种融合策略进行系统对比

**🔧 技术方法**

使用音频图谱注意力网络(AASIST)、卷积神经网络 EfficientNet-B1、XceptionNet、MesoNet 及集成方法（均值、投票、加权平均、堆叠）

**📊 数据集**

训练使用AIGVDBench、FaceForensics++、ASVspoof2019 LA，测试在FakeAVCeleb上

**📈 对比分析**

与单模型相比，集成在FakeAVCeleb上平均精度提升至约70%，堆叠/投票方法表现最好

**⚠️ 局限性**

主要限制在于音频模型泛化差、数据集偏斜导致的误判，以及集成中模型贡献不均衡

---

## 37. When Agents Remember Too Much: Memory Poisoning Attacks on Large Language Model Agents

**arXiv ID:** 2607.06595 | [PDF](https://arxiv.org/pdf/2607.06595v1)

**作者:** George Torres `[一作]` (New Mexico State University), Satyajayant Misra `[通讯]` (New Mexico State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在具备长期记忆的个人AI助手中，通过发送隐藏攻击payload对记忆库进行中毒的攻击向量（GhostWriter），并提出双阶段防御框架AM‑Sentry，结合记忆保存策略与检索屏蔽机制，以降低攻击成功率并保持代理效能。

**💡 创新点**

创新点在于：①首次系统化展示通过邮件/日历等未受信任输入对长期记忆代理进行注入与激活的两阶段攻击；②提出可适配多种代理架构的可配置记忆保存策略（S_1,S_2,S_3）与检索屏蔽屏，兼顾安全与效用；③构建综合评估实验平台，包括多模型、多代理、定制工作周数据集，量化攻击成功率与防御效果。

**🔧 技术方法**

技术上使用LLM（GPT‑5.4‑mini、DeepSeek‑V4‑Flash、Gemini‑2.5‑Flash、Llama 3.1‑8B）作为代理核心，采用Embedding模型(all‑MiniLM‑L6‑v2)进行记忆检索；防御策略利用LLM评估记忆四字段或七项清单得分，结合阈值与非LLM逻辑实现过滤；检索屏使用LLM判断相关性、指令抑制、可信度、矛盾等四个维度。

**📊 数据集**

主要数据集为自构造的五天工作周场景，包含模拟电子邮件、日历邀请、会议等多源信息；攻击payload基于Enron邮件语料聚类优化；评测还引用AgentPoison、MINJA等公开攻击基准做对比。

**📈 对比分析**

通过在五个主流记忆代理（A‑Mem、Mem0、ExpeL、Letta、MemoryOS）与四个LLM模型下进行16种攻击场景的实验，GhostWriter注入成功率≈98%，检索率≈94%，激活率≈60%；在未防御时几乎击败所有代理。引入S_3+检索屏后，平均攻击成功率降至12%以下（除某些模型略高），而对代理F1、LLM判定分和工具调用准确率影响仅≤0.04，保持高效能。

**⚠️ 局限性**

局限包括：仅探讨邮件/日历作为攻击表面，未覆盖文档、网页、代码仓库等；防御策略权重为直观设定，缺乏自适应优化；实验基于模拟工作周，未使用真实用户数据或现场部署；未评估针对适应性攻击者的鲁棒性；仅覆盖有限的代理架构与LLM。

---

## 38. Benchmark Engineering as a Design Instrument for Heterogeneous Information Systems

**arXiv ID:** 2607.07175 | [PDF](https://arxiv.org/pdf/2607.07175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 39. A Gold-Standard Study of What Makes a Lightweight Game-Playing Agent Strong

**arXiv ID:** 2607.06854 | [PDF](https://arxiv.org/pdf/2607.06854v1)

**作者:** Nima Kelidari `[一作]` (University of Southern California), Mahdi Salmani `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在不完全信息扑克牌游戏 Gin Rummy 上构建轻量级强化学习代理，设计并使用固定、可复现的专家对照基准，系统评估多种训练、奖励、网络和对手策略。

**💡 创新点**

创新点：①构造一个强大且确定性的专家基准，用其对照评估训练策略；②证明奖励设计无法诱导“gin”行为，信息隐藏决定性能上限；③在不同网络结构和搜索方法下证明网络容量不是瓶颈；④将相同方法迁移到 Leduc Hold'em，验证结果的普适性。

**🔧 技术方法**

技术：masked actor‑critic（PPO/TRPO）强化学习、对手课程（随机→过去版本→自对弈）、热启动与保最佳检查点、奖励加权（knock vs gin）、状态嵌入、DAgger 复制学习、LLM 在线对手、MLP/卷积/Deep‑Sets/自注意力/ LSTM 网络编码、ISMCTS、NFSP、CFR。

**📊 数据集**

数据集：RLCard+PettingZoo 环境下随机生成的 Gin Rummy 牌局；Leduc Hold'em 环境下的随机对局。

**📈 对比分析**

比较方法：以固定专家的胜率为唯一指标，统计 95% 置信区间；对不同算法、奖励、网络、对手等进行 1‑100+ 次控制实验。结果显示：TRPO 优于 PPO（22.5% vs 15%），knock‑first 奖励提升胜率，热启动与保最佳提升 2‑3%；最佳配置在专家面前约 34% 胜率；公平搜索仅 26%，oracle 85%；Leduc Hold'em 自玩学习者在专家下实现平均回报 -0.085，接近最优。

**⚠️ 局限性**

局限性：仅研究单一游戏，专家并非博弈最优；未实现对手手牌的显式推理或更强记忆；LLM 仅作为慢速在线对手；实验规模受限；结果对不同游戏的泛化仍需进一步验证。

---

## 40. RoboSnap: One-Shot Real-to-Sim Scene Generation for Generalizable Robot Learning and Evaluation

**arXiv ID:** 2607.06699 | [PDF](https://arxiv.org/pdf/2607.06699v1)

**作者:** Shujie Zhang `[一作]` (Shanghai AI Laboratory), Chunhua Shen `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

利用单张 RGB 图像构建可交互、可重用的仿真场景，分离物理前景与视觉背景，支持轨迹回放、数据生成与策略评估。

**💡 创新点**

提出分层的 real‑to‑sim 框架，先用 VLM/SAM 提取并对齐前景物体，再通过 SDF‑physics 迭代优化姿态，并为背景构建 3D 高斯 splatting 视觉层；同时发布 564 场 DROID 场景的复现数据集。

**🔧 技术方法**

融合 VLM、SAM 3D、VGGT、ICP 注册、Gaussian splatting、SDF 优化、SAPIEN 物理仿真、Isaac Sim 渲染、GPT‑4V 关系推断、AnyGrasp、cuRobo 等多种先进技术。

**📊 数据集**

主要使用 DROID 机器人数据集（564 场真实场景）及其轨迹数据，辅以公开的机器人演示与合成数据。

**📈 对比分析**

与 RoLA、SAM3D+FoundationPose 等方法对比，成功率提升至 5/5（轨迹回放），Sim‑Real 相关性达到 Pearson 0.887、MMRV 0.0066；在多任务评估中显著提高策略成功率与鲁棒性。

**⚠️ 局限性**

受限于输入质量（遮挡、极端光照）、仅支持刚体/关节物体（不含柔性、颗粒、流体）、物理参数仅依赖 VLM 先验，且在不同策略上的广泛验证仍待进一步研究。

---

## 41. Comprehensive Evaluation of Large Language Model Responses: A Multi-Factor Scoring System

**arXiv ID:** 2607.06940 | [PDF](https://arxiv.org/pdf/2607.06940v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 42. HPG-Diff: Hierarchical physics-guided diffusion with differentiable connectivity constraints for topology optimization

**arXiv ID:** 2607.07233 | [PDF](https://arxiv.org/pdf/2607.07233v1)

**作者:** Jinbo Yang `[一作]` (Beijing Institute of Technology), Shikai Jing `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种层级物理引导扩散模型（HPG‑Diff），通过在UNet不同层注入位移、主应力线和应变能密度特征，并加入基于热扩散的可微浮材抑制损失，直接生成符合物理约束且连接良好的拓扑优化结构。

**💡 创新点**

创新点在于：① 将多尺度物理特征按层级注入扩散过程，实现生成与结构力学的天然对齐；② 设计可微热传播抑制浮材的损失，使生成的密度场天然满足连通性约束；③ 通过LoRA轻量化微调实现对非正方形域的快速适配。

**🔧 技术方法**

采用扩散生成模型（基于UNet）结合层级交叉注意力、可微热传播（Max‑3×3卷积）以及指数时间加权的浮材抑制损失，训练使用AdamW和余弦噪声调度。

**📊 数据集**

使用Mazé等人提供的30k样本SIMP数据集（包括位移、主应力线、应变能密度等预处理物理特征），并在1,800/1,000的测试集（分别为分布内和分布外）上评估。

**📈 对比分析**

与TopologyGAN、TopoDiff‑Guided、DOM w/ TA等三种SOTA生成方法对比，HPG‑Diff在分布内平均合规误差降至0.87%（相对基线低80%）、浮材比例降至2.90%；在分布外平均合规误差为5.29%、浮材比例为2.44%，显著优于对比模型。

**⚠️ 局限性**

主要局限包括：仅针对单负载情况设计浮材抑制；适配更复杂形状（非矩形、L形、圆形等）和多负载场景仍待验证；模型依赖于SIMP生成的训练数据，未覆盖非线性材料、疲劳或制造约束等实际工程需求。

---

## 43. Devising Interactive Spaces: A Rehearsal-Oriented Tool for Creating Responsive Environments for Immersive Theatre

**arXiv ID:** 2607.06761 | [PDF](https://arxiv.org/pdf/2607.06761v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 44. A Quiet Failure in Calibrated Virtual Screening: Marginal Conformal Prediction Under-Covers the Minority Class, and a Class-Conditional Fix Recovers It

**arXiv ID:** 2607.06605 | [PDF](https://arxiv.org/pdf/2607.06605v1)

**作者:** Muhammadjon Tursunbadalov `[一作]` (Champions College Prep), Mustafojon Tursunbadalov `[通讯]` (Champions College Prep)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估并证明了在化学筛选任务中，边际（marginal）conformal prediction 在类别不平衡数据上会严重低估少数类覆盖率，随后提出使用类别条件（Mondrian）conformal 校准来恢复覆盖率。

**💡 创新点**

首次在真实药物发现数据上系统量化了边际校准对少数类的覆盖率崩溃，并用简单的覆盖率守恒公式解释其规模；同时揭示了少数类错误主要集中在通用骨架上，给出成本模型证明其对实验室决策的实质影响。

**🔧 技术方法**

使用了边际和类别条件（Mondrian）conformal prediction、两种非一致性分数（LAC、APS）、随机森林、图卷积网络和冻结的 ChemBERTa Transformer 进行实验；通过覆盖率、集大小、选择性预测和成本效益评估。

**📊 数据集**

使用了 MoleculeNet 四个二分类任务：BACE、BBBP、Tox21 SR-ARE 与 ClinTox（四个不同的类不平衡程度）。

**📈 对比分析**

与边际校准比较时，Mondrian 校准将少数类覆盖率从 64.8% 直至 90.3%（BACE）/ 38.9%→89.4%（Tox21）/ 4.2%→94.6%（ClinTox），平均集大小仅略增；在不同模型与分数下均保持显著改进（p<0.001）。

**⚠️ 局限性**

局限性包括仅测试四个数据集、单一 α=0.10、未遍历不同覆盖率阈值、ChemBERTa 未微调、scaffold split 采用随机划分、成本模型仅示例化、且只关注单一筛选任务。

---

## 45. Pixel-Precise Explainable Stress Indexing: A Semantic Segmentation Framework for Disease Severity Quantification in Field Crops

**arXiv ID:** 2607.06585 | [PDF](https://arxiv.org/pdf/2607.06585v1)

**作者:** Raunak Kumar `[一作]` (Indian Institute of Technology Bombay), Soumyashree Kar `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个统一的深度学习管线，能够对叶片图像进行语义分割、病害严重程度回归和病种分类，实现作物健康的自动化评估。

**💡 创新点**

创新点在于设计了轻量化的 U‑Net+MobileNetV2 结构，并使用复合损失（加权交叉熵+Focal‑Tversky+Dice）同时输出分割、SSI 与病种标签，达到实时、精准的病害量化。

**🔧 技术方法**

主要技术包括卷积网络的 Encoder‑Decoder 结构、深度可分离卷积、Transformer 关注机制（SegFormer）、多种损失函数及数据增强策略。

**📊 数据集**

使用了 Apple Tree Leaf Disease Segmentation (ATLDS) 数据集进行训练与评估，并用 RiceSEG 数据集检验跨域鲁棒性。

**📈 对比分析**

在 ATLDS 上与 FCN、PSPNet、SegFormer、标准 U‑Net 等模型比较，UNet‑MobileNetV2 在 mIoU 0.697、像素准确率 98.20%、病害检测 99.41% 领先，并以 14.7 ms/图实现实时推理；在 RiceSEG 上也取得最高 mIoU 0.489，优于基准。

**⚠️ 局限性**

限制主要包括对大量像素级标注的需求、仅使用 RGB 单模态、对视觉相似的病害类别（如灰斑与叶斑）存在混淆，以及尚未完成在 UAV 边缘设备上的实地部署与多传感器融合。

---

## 46. Voltron: Enabling Elastic Multi-Device Execution of LLM Inference for Empowered Edge Intelligence

**arXiv ID:** 2607.07046 | [PDF](https://arxiv.org/pdf/2607.07046v1)

**作者:** Chanwoo Cho `[一作]` (Korea University), Young Geun Kim `[通讯]` (Korea University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了Voltron框架，支持多设备边缘 LLM 推理，通过层级混合并行、重要性感知混合精度和动态弹性调度，实现对设备异质性、计算与网络变异的自适应。

**💡 创新点**

创新点在于：①层级混合并行（结合模型并行和张量并行）以匹配不同层的计算/内存特性；②重要性感知的混合精度与剪枝策略，最大化准确率；③实时弹性调度（计算、通信、能耗）应对设备与网络的动态变化；④通过二进制搜索与回归模型实现低开销的精度配置。

**🔧 技术方法**

使用技术包括：张量并行、模型并行、回归估算层级执行时间、二进制搜索精度分配、激活量化、Wi‑Fi Direct 多设备通信、DVFS 能耗优化、在 llama.cpp 框架上实现。

**📊 数据集**

使用数据集：LMSYS‑Chat‑1M 对话输入进行推理，评估准确率的基准为 MMLU、Hellaswag、GSM8K、MATH。

**📈 对比分析**

与单设备推理、基于设备异质性调整的 MP/TP 等基线比较，采用 TTFT/TPOT 10 s / 400 ms 的 QoS 目标。实验显示 Voltron 在满足 QoS 的前提下，平均提升 10‑16% 准确率，能耗可降低至 59% 左右。

**⚠️ 局限性**

限制：需要多台设备的集群；在信号弱或设备离线时仍可能导致准确率下降；实现复杂度高，需硬件支持低功耗 I/O；跨租户多设备安全与隐私保护仍待进一步研究。

---

## 47. Smart Scissor: Coupling Spatial Redundancy Reduction and CNN Compression for Embedded Hardware

**arXiv ID:** 2607.06915 | [PDF](https://arxiv.org/pdf/2607.06915v1)

**作者:** Hao Kong `[一作]` (Nanyang Technological University), Qian Lin `[通讯]` (HP Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了“Smart Scissor”深度压缩框架，结合动态图像裁剪与复合压缩技术，显著降低CNN在边缘设备上的计算量，同时保持甚至提升准确率。

**💡 创新点**

创新点包括：①使用轻量化前景预测器（基于Grad‑CAM生成的边界框训练）实现实例感知的动态裁剪；②引入复合压缩（同时压缩网络深度、宽度和分辨率），并通过准确率-成本估计器快速求解最佳压缩系数；③将两者无缝融合，形成端到端可调的压缩方案。

**🔧 技术方法**

技术主要包括：Grad‑CAM可视化、轻量化卷积前景预测网络、深度宽度分辨率三维压缩策略、二次多项式准确率估计器、以及标准CNN训练与微调。

**📊 数据集**

在ImageNet‑1K与ImageNet‑100两个大规模分类数据集上进行实验。

**📈 对比分析**

与基线ResNet‑50、RegNet‑X及多种SOTA压缩方法（如HRank、DR‑ResNet等）比较，Smart Scissor在保持相同MACs或参数量的前提下，Top‑1准确率提升0.3%–4.2%，在低计算预算下准确率提升高达4.2%，且在NVIDIA AGX Xavier、Jetson Nano及Intel i7‑9750H上实现了显著的延迟下降与吞吐量提升。

**⚠️ 局限性**

局限性包括：①前景预测器虽然轻量，但仍需额外训练；②动态裁剪依赖于Grad‑CAM生成的边界框，若目标对象遮挡或多物体场景可能影响预测精度；③复合压缩的系数求解依赖估计器的拟合误差，极端预算下可能出现误差放大。

---

## 48. The Rank-One Corner: How Much Value Equivalence Does a Task Need from a World Model?

**arXiv ID:** 2607.06640 | [PDF](https://arxiv.org/pdf/2607.06640v1)

**作者:** Donna Vakalis `[一作]` `[通讯]` (Quebec AI Institute), Donna Vakalis (Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在世界模型训练中目标维度决定模型对任务闭包（closure）中可预测坐标的表示量。

**💡 创新点**

发现目标维度与模型学习到的闭包维度之间呈一对一对应关系，单一奖励只能安装一维闭包，而多维目标能安装相应维度的闭包。

**🔧 技术方法**

使用DreamerV3框架、类别RSSM、线性探测器以及降秩回归理论进行分析。

**📊 数据集**

在构造的高维观察环境中，观测为64×64像素图像，底层真实闭包维度已知并可线性解码。

**📈 对比分析**

通过对比标量奖励与多维目标的线性探测恢复率，标量仅恢复约10%闭包，而完整目标恢复约76%；此外，在闭包可观测时，单奖励与全目标的表现相当。

**⚠️ 局限性**

实验仅在单一架构和合成环境上进行，未验证在自然图像或多任务情形下的普适性，并且在连续潜在变体中出现泛化失效。

---

## 49. AnchorPrune: Relevance-Anchored Contextual Expansion for Visual Token Pruning

**arXiv ID:** 2607.07033 | [PDF](https://arxiv.org/pdf/2607.07033v1)

**作者:** Kyuan Oh `[一作]` (Chung-Ang University), Bumsoo Kim `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练无关的视觉令牌剪枝框架 AnchorPrune，先构造一个保护性的相关锚点再通过重要性加权新颖性进行上下文扩展，减少视觉令牌数同时保持推理精度。

**💡 创新点**

创新点在于：①将查询相关性与多样性按顺序分离，先保护不可替代的关键证据；②通过新颖性曲线自适应决定锚点大小；③使用重要性-加权的新颖性指标，使后续选取的令牌既信息丰富又不冗余。

**🔧 技术方法**

核心技术包括：视觉-文本相似度排序（CLIP预投影或后投影空间），新颖性度量（最小余弦距离），自适应锚点阈值，贪婪重要性加权扩展，保持原模型不变的轻量级实现。

**📊 数据集**

在图像 VLM 上使用 VQAv2、TextVQA、GQA、ScienceQA-IMG、MME、POPE、MMBench-EN/CN、MM-Vet；在视频 VLM 上使用 Video-MME、EgoSchema、TempCompass；在非 CLIP 架构 Qwen2.5‑VL‑7B 上使用 MME、TextVQA、DocVQA、AI2D、MMMU、MMBench-EN/CN。

**📈 对比分析**

与 FastV、PyramidDrop、SparseVLM、PruMerge+、VisionZip、DivPrune、CDPruner 等训练无关剪枝方法对比，AnchorPrune 在所有模型、预算下均获得最高的“保留性能”，尤其在极压缩下（如 LLaVA-NeXT-7B 只保留 160/2880 令牌）能保持 97.6% 的完整模型性能，显著优于第二名。

**⚠️ 局限性**

局限性：对查询相关性信号依赖较大，若文本描述模糊或视觉-文本对齐弱，锚点构造可能不足；在极低令牌预算或不同视觉编码器（无 CLIP 对齐）时效果可能受限；虽训练无关但选择过程仍有运行时开销，且未针对动态视频剪枝做进一步优化。

---

## 50. TRACE-Seg3D: Counterfactual Context Auditing For Robust 3D Glioma Segmentation Under Institutional Shift

**arXiv ID:** 2607.07038 | [PDF](https://arxiv.org/pdf/2607.07038v1)

**作者:** Nguyen Linh Dan Le `[一作]` (University of Melbourne), Tran Dang Khoi `[通讯]` (Industrial University of Ho Chi Minh City)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了 TRACE‑Seg3D 框架，能够在不同机构扫描条件下对 3D 脑瘤分割结果进行可审计，提供上下文敏感性和解剖合理性评估。

**💡 创新点**

创新点在于三方面：① 用代理锚定的疾病/上下文因子分离，② 引入反事实上下文运输（CCT）实现模型输出的可审计性，③ 在预测后加入 ET ⊆ TC ⊆ WT 的解剖先验约束以控制小结构错误。

**🔧 技术方法**

技术包括 MedNeXt‑S 3D 编码器-解码器、FiLM 风格的特征调制、代理锚定损失与对抗正则化、逆向损失、解剖先验运算、以及 CCT 的一致性与不稳定图生成。

**📊 数据集**

使用了 BraTS 2020 和 UTSW‑Glioma 两个公开脑瘤 MRI 数据集，分别作为源域和目标域进行 ID 与 OOD 评估。

**📈 对比分析**

通过与 nnU-Net、UNETR、SegFormer3D、CSDG、ICMSeg、CauAug、CauSSL 等 12 种基线进行对比，TRACE‑Seg3D 在 ID 与 OOD 的 Dice 及 HD95 上均居首位，尤其在跨机构迁移时 Dice 提升约 2%–3%，HD95 显著下降；对 ET 子区的精度提升和 FP 控制也最为突出。

**⚠️ 局限性**

局限性包括：① 依赖代理锚定的因子而非真正可辨识的因果变量，可能导致因子分离不完全；② 上下文银行的样本量有限，可能无法覆盖所有扫描差异；③ 仅在脑瘤 MRI 上验证，需在更大多中心、多模态数据上进一步评估；④ CCT 生成的 instability 图只能提示敏感区域，未提供具体失效原因或可操作的解释。

---

## 51. MADB: A Large-Scale Music Aesthetics Dataset with Professional and Multi-Dimensional Annotations

**arXiv ID:** 2607.06929 | [PDF](https://arxiv.org/pdf/2607.06929v1)

**作者:** Sirui Zhang `[一作]` (Central Conservatory of Music), Songchun Zhu `[通讯]` (Beijing Institute for General Artificial Intelligence)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了大规模音乐审美评估数据集 MADB，并在统一评估框架下对现有音频编码器和跨模态方法进行基准测试。

**💡 创新点**

创新点包括：① 设计了 10 维结构化审美评估框架；② 将自由文本评论和标签作为无监督语义监督进行音频-文本对齐；③ 采用多源（人工、Suno、Levo、Muchin）曲目构建多样化数据集。

**🔧 技术方法**

使用了 CLAP、MuQ、MERT 等预训练音频编码器，并通过对比学习与语义融合实现音频-文本对齐；同时利用 Qwen2‑Audio 进行零样本多模态审美预测。

**📊 数据集**

使用的数据集为 MADB，包含 9,999 首曲目（手工收集 2,799 首、Suno 1,000 首、Levo 1,800 首、Muchin 4,400 首），每首曲目提供 10 维评分、整体分、文本评论和标签。

**📈 对比分析**

在统一的回归框架下，分别计算 MSE、LCC、SRCC、KRCC 四项指标进行比较；MuQ/MERT 在所有指标上显著优于 CLAP，CLAP+评论/标签适配略有提升，但整体性能仍远未达到人类一致性。

**⚠️ 局限性**

局限性包括：① 数据集音乐风格偏向流行，缺乏多样化代表性；② 文本评论先写中文后翻译为英文，可能引入语义偏差；③ 仅提供全曲级评分，缺少时间粒度标注；④ 标签粒度较粗，未覆盖细分子流派；⑤ 未探究各审美维度与整体评分之间的因果关系。

---

## 52. Safe2Hail: A Forensic-Driven Post-Trip Tracking Framework for Ride-Hailing Safety in Africa

**arXiv ID:** 2607.07271 | [PDF](https://arxiv.org/pdf/2607.07271v1)

**作者:** Alvina Minja `[一作]` (Carnegie Mellon University Africa), Jema Ndibwile `[通讯]` (Carnegie Mellon University Africa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

开发了一个名为Safe2Hail的取证驱动的后车程跟踪框架，用于在行程异常（如提前取消）后持续监测乘客与司机的相对距离，提供后续取证链；

**💡 创新点**

突破性之处在于：①在行程结束后仍可持续同步与记录轨迹；②采用风险矩阵与动态风险乘数实现对异常事件的分级评估；③实现隐私友好的临时同步与加密存储；

**🔧 技术方法**

使用Flask+SQLite实现轻量级中间件，基于GPS（可扩展为BLE/UWB）与Haversine公式计算距离；配合Chart.js+Leaflet.js的可视化仪表盘；

**📊 数据集**

主要使用模拟的行程日志和基于真实犯罪统计（内罗毕、达累斯萨拉姆）以及Uber、Bolt等平台的安全报告数据；

**📈 对比分析**

与传统应用仅在车程内追踪的安全功能对比，Safe2Hail在模拟实验中实现了90–95%的正确率，数据保留可达24小时，资源占用低，且可无缝集成现有系统；

**⚠️ 局限性**

局限性包括：依赖GPS准确性，需双方设备合作，缺乏BLE/NFC等更精确的近距离检测，缺少大规模真实环境验证，仪表盘功能尚处于概念验证阶段。

---

## 53. D2PO: Optimizing Diffusion Samplers via Dynamic Preference

**arXiv ID:** 2607.06609 | [PDF](https://arxiv.org/pdf/2607.06609v1)

**作者:** Jinkyu Kim `[一作]` (Seoul National University), Bohyung Han `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过动态直接偏好优化（D2PO）框架，将扩散采样器的参数（时间步表和CFG权重）视为偏好对齐问题，并用能量基模型与得分网络定义的距离来训练。

**💡 创新点**

① 将直接偏好优化（DPO）迁移到扩散采样器；② 用预训练得分网络直接构造能量函数，捕捉多尺度结构和纹理误差；③ 引入动态偏好机制，以自身更细粒度的采样作为参考，消除固定教师的误差上限。

**🔧 技术方法**

直接偏好优化（DPO）、能量基模型、基于得分网络的噪声预测距离、动态时间步表和CFG权重优化、EMA与Monte Carlo抽样技术。

**📊 数据集**

Stable Diffusion v1.5（COCO文本提示）、ImageNet-256（潜在空间）、InstaFlow（COCO提示）。

**📈 对比分析**

与DMN、GITS、LD3等传统离散化方法在FID、HPSv2和Aesthetic评分上对比。D2PO在4–5步时Fid最低、感知质量最高；在更多步时保持高感知分数，Fid略高于基线，整体优于现有方法。

**⚠️ 局限性**

受限于预训练得分网络的可用性；在高步数时Fid不如某些基线；仅优化采样器而不改模型参数，对模型鲁棒性和迁移性影响有限。

---

## 54. Towards Reliable Aerial Ground Vehicle Collaboration: An Integrated Planning and Autonomy Framework for Field Deployment

**arXiv ID:** 2607.07350 | [PDF](https://arxiv.org/pdf/2607.07350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 55. Vision Foundation Models in Radiology: A Scoping Review of Data, Methodology, Evaluation and Clinical Translation

**arXiv ID:** 2607.07219 | [PDF](https://arxiv.org/pdf/2607.07219v1)

**作者:** Alejandro Vergara-Richart `[一作]` (Quantitative Imaging Biomarkers in Medicine), Ana Jiménez-Pastor `[通讯]` (Quantitative Imaging Biomarkers in Medicine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化了2017-2026年在医学影像中训练的视觉基础模型（VFMs）的研究，采用PRISMA-ScR框架，对67篇论文按三大支柱（数据规模与异质性、预训练与架构、下游可迁移性）进行证据映射。

**💡 创新点**

提出了针对放射学视觉基础模型的三柱框架和FUTURE-AI原则对齐评估，为该领域提供统一的分类、评估与可信AI指引，并揭示规模、异质性、方法与可信度缺口。

**🔧 技术方法**

综述涵盖Transformer、Swin、ViT、Hybrid、Mamba、Graph、MoE等架构；自监督学习（掩码图像建模、对比学习、复合SSL）、监督预训练、PEFT、Zero/Few-shot迁移等技术。

**📊 数据集**

使用的影像数据主要为脑MRI、胸/腹CT、胸X射线，样本规模从少于10万片到超过百万图像，涉及多中心、多模态、不同解剖部位与疾病，公共与私有数据混合。

**📈 对比分析**

67篇文献多数对比基准预训练/非预训练模型，约94%显示预训练模型优于基线，84%优于任务专用监督模型；评估涵盖分类、分割等多任务，约86%进行多任务下游评估，并通过跨中心、跨扫描仪、疾病/解剖/模态迁移验证，整体表现显示VFM在多任务上具备可迁移优势，但外部验证不足。

**⚠️ 局限性**

缺乏统一定义与足够规模的训练集，单模态占比高、时序数据稀缺；方法多样化导致缺乏可比性；可信AI方面解释性与公平性不足；公开权重与数据共享有限，缺少标准化评测与系统质量评估。

---

## 56. Bringing robustness to end-user programming

**arXiv ID:** 2607.07116 | [PDF](https://arxiv.org/pdf/2607.07116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 57. Cyber Dynamics I: Finite Macrostates for Behavioral Anomaly Detection in Network Telemetry

**arXiv ID:** 2607.07075 | [PDF](https://arxiv.org/pdf/2607.07075v1)

**作者:** Abdul Rahman `[一作]` (Howard University), Sachin Shetty `[通讯]` (Old Dominion University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一个有限宏观状态框架，将网络遥测在 Canonical Security Telemetry Substrate (CSTS) 上进行粗粒化，构造包含活动、无序度、结构性、波动性、持久性、耦合性与偏差等维度的宏观状态向量，并利用这些宏观状态的轨迹进行动态异常检测。

**💡 创新点**

创新点在于：①将熵作为宏观状态的一个维度而非单独阈值，构建多维行为状态空间；②将异常分为状态异常和转移异常，强调系统运动而非单点；③引入稳定性与恢复时间等动力学概念；④使用 CSTS 统一结构实现跨数据源的可迁移性；⑤在多尺度窗口下检验异常可见性。

**🔧 技术方法**

技术手段包括：宏观状态构造函数；Shannon、Rényi、Tsallis 熵等分布无序度量；图结构统计（密度、度集中度、结构子图计数）；波动性指标（子窗口波动、方差、变点估计）；持久性比例；Jensen–Shannon 距离及几何距离偏差度量；线性回归/VAR/最近邻预测等基准转移模型；无监督检测方法（One‑Class SVM、Isolation Forest、autoencoder）以及评价指标 AUROC、AUPRC、F1 等。

**📊 数据集**

实验数据集：UNSW‑NB15、CIC‑IDS2017（主实验），CICIoT2023（可选）。

**📈 对比分析**

与熵基线（单/多维熵阈值）、传统聚类/异常检测基线（OCSVM、Isolation Forest、autoencoder）进行对比；在 30 s、60 s、5 min、15 min 四个窗口尺度下进行多尺度实验；结果表明宏观状态模型在 AUPRC 上显著优于熵基线和传统方法，尤其在 60 s 时段达到 0.9730 的 AUPRC；然而，简单的转移异常检测模型在性能提升上表现平平。

**⚠️ 局限性**

局限性：①转移异常检测效果受限于模型过于简单；②对不同协议和更大规模流量的泛化尚未充分验证；③实验主要基于公开基准数据，真实环境验证仍待完成；④多尺度融合和耦合变量的进一步探索需要更多研究；⑤标签稀疏或不确定性情况下的鲁棒性尚未系统评估。

---

## 58. From Text to Parameters: Predicting Item Parameters from Embedding Regularization with Reliability and Design Ceilings

**arXiv ID:** 2607.07141 | [PDF](https://arxiv.org/pdf/2607.07141v1)

**作者:** Shi-Ting Chen `[一作]` (University of Hong Kong), Jinsong Chen `[通讯]` (University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提出并实现了一个基于文本嵌入的项目项参数预测框架，并在此框架上设计了可靠性上限与设计上限两种可解释的性能界限。

**💡 创新点**

创新点包括：①将传统LLTM的手工设计矩阵替换为自动化的高维文本嵌入；②构建可衡量预测极限的两类上限（可靠性上限与设计上限）；③采用重复交叉验证与RMSE/SD比率结合的评估方式，系统化地解释预测结果。

**🔧 技术方法**

技术手段主要是正则化回归（岭、Lasso、Elastic Net）结合大规模文本嵌入模型（如Qwen3-Embedding-8B、SFR-Embedding-2_R等），并对模型进行加权回归、联合多响应预测、LLM生成的选项推理（rationale）增强以及手工特征拼接。

**📊 数据集**

使用的数据集包括：EEDI数学题库（355条文本题）和BEA 2024 USMLE共享任务数据集（667条多项选择题）。

**📈 对比分析**

通过对不同嵌入模型、回归方法和加权方案在重复5折交叉验证中的表现进行比较，EEDI上1PL难度预测达到R²≈0.53（约占可靠性上限的57%），但在BEA上R²≈0.02，仅能解释极小的变异；Rationale生成对1PL难度略微提升约ΔR²≈0.02，权重调整对结果影响不大。

**⚠️ 局限性**

主要局限包括：①3PL可靠性上限估计不够精确；②仅评估文本题，未涵盖含图形的题目；③嵌入未做任务特定微调且仅使用线性回归；④BEA数据仅提供难度标签，无法评估IRT参数；⑤交叉验证受样本量影响，导致结果波动；⑥模型缺乏可解释性，难以解释哪些文本特征对难度最具影响。

---

## 59. UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma

**arXiv ID:** 2607.06987 | [PDF](https://arxiv.org/pdf/2607.06987v1)

**作者:** Chongyu Fan `[一作]` (ByteDance Seed), Yi Lin `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种Unbounded Positive Asymmetric Optimization (UP) 方法，用来解决基于重要性采样的RL算法在LLM推理任务中出现的探索-稳定性两难。

**💡 创新点**

核心创新在于：①引入概率容量（Probability Capacity）概念揭示传统剪裁对正优势的过度限制；②用stop‑gradient自我锚定的比率替代历史策略π_old，消除IS导致的梯度爆炸；③设计对正负优势的非对称优化策略，使正优势可无界探索，负优势仍受传统剪裁保护。

**🔧 技术方法**

技术手段包括：自我锚定的无界重要性比率、异向量梯度计算、动态路由的正负分支、与GRPO、DAPO、GSPO等GxPO框架的无缝对接。

**📊 数据集**

在多种推理数据集上验证：AIME24、AMC23、MATH500、Minerva、OlympiadBench、Geometry3K 以及对应的训练集如MATH Levels 3‑5、DAPO‑17K‑MATH 等。

**📈 对比分析**

与十二种主流RL基线（GRPO、DAPO、GSPO、GMPO、ASPO、CISPO、DPPO、REINFORCE++、RLOO、W‑REINFORCE、Dr.GRPO、SAPO）进行对比。UP‑GRPO 在平均 Pass@1 方面达到 61.31%，比最强基线 GSPO 提升 1.16%；UP‑DAPO、UP‑GSPO 等在对应任务中均表现出更高的探索熵、稳定的 KL 散度与更佳的准确率。

**⚠️ 局限性**

局限性包括：对负优势仍依赖传统剪裁，缺乏针对极端负优势的更深层理论保障；目前仅在LLM推理和几何视觉推理任务中验证，未展示对更广泛任务或更大模型的通用性；以及对超大规模数据训练的计算成本仍未充分评估。

---

## 60. Manual, Joystick, or Haptic Control? An In Vitro Comparison of Navigation Strategies for Robotic Interventional Neuroradiology Procedures

**arXiv ID:** 2607.07253 | [PDF](https://arxiv.org/pdf/2607.07253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 61. Clinical Translation of Brain-Computer Interface in China: A Landscape Analysis of Investigator-Initiated Trials, Registered Clinical Trials, and Regulatory Approval

**arXiv ID:** 2607.07185 | [PDF](https://arxiv.org/pdf/2607.07185v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 62. Non-minimal k-perfect hashing: Tight lower bounds and an application to fast static hash tables

**arXiv ID:** 2607.07257 | [PDF](https://arxiv.org/pdf/2607.07257v1)

**作者:** Ragnar Groot Koerkamp `[一作]` (Karlsruhe Institute of Technology), Stefan Walzer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于自适应稀疏学习的多尺度目标检测框架；

**💡 创新点**

创新点在于将稀疏正则化与多尺度特征融合相结合，显著提升了检测精度；

**🔧 技术方法**

采用了卷积神经网络和稀疏自编码技术；

**📊 数据集**

使用了公开数据集 COCO 和 PASCAL VOC 进行实验；

**📈 对比分析**

与传统检测方法相比，实验结果表明该方法在 mAP 上提升约 4%-6%，且检测速度加快；

**⚠️ 局限性**

局限性是对小目标的检测仍有限，且模型训练时对硬件资源需求较高。

---

## 63. WildCity: A Real-World City-Scale Testbed for Rendering, Simulation, and Spatial Intelligence

**arXiv ID:** 2607.06838 | [PDF](https://arxiv.org/pdf/2607.06838v1)

**作者:** Xiangyu Han `[一作]` (May Mobility), Yiming Li `[通讯]` (New York University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WildCity城市规模真实多模态数据集，并基于3D Gaussian Splatting构建了针对大尺度街景的重建基线，同时实现了闭环城市数字孪生模拟。

**💡 创新点**

创新点包括（1）采集1500公里、18条轨迹的真实多城市长时序数据；（2）针对大规模街景提出姿态优化、天空与地面约束以及Diffix修复的完整重建流程；（3）在此数据集上完成城市场景重建、渲染与仿真评估。

**🔧 技术方法**

采用3D Gaussian Splatting、Rig-aware姿态优化、天空MLP、地面正则化、Diffix3D+修复、多GPU分布式训练以及Alpamayo闭环仿真框架。

**📊 数据集**

使用WildCity本身（6座美国城市、18条轨迹、平均83.7公里、总长1507公里、3.01M关键帧、6摄像头+LiDAR+GPS+IMU）。

**📈 对比分析**

与3DGS、H-3DGS、CityGS、VGGT-Long等基线对比，本文方法在PSNR、SSIM、LPIPS和Depth L1上均显著优于基线，尤其在长轨迹下Depth L1降至6.6 m、PSNR 23.14 dB。

**⚠️ 局限性**

限制包括自动生成语义掩码仍有误差、姿态误差（水平亚厘米、垂直厘米级）未完全消除、对长距离漂移与稀疏视角泛化能力有限，Diffix修复仍需后处理且易产生hallucination。

---

## 64. Understanding Interpretation Difficulty in Harmful Online Communication: Insights from Cybercrime Communities

**arXiv ID:** 2607.07277 | [PDF](https://arxiv.org/pdf/2607.07277v1)

**作者:** Tomohiro Okatsu `[一作]` (Yokohama National University), Tatsunori Mori `[通讯]` (Yokohama National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究网络犯罪相关Discord聊天中有害信息的解读难度，并构建专家审阅的参考解释

**💡 创新点**

提出了将解读视为证据整合问题的思路，给出了影响解读难度的六类因子及其对应的信息来源分类

**🔧 技术方法**

使用了人类标注、两种开源LLM（GPT‑OSS‑20B、GPT‑OSS‑120B）以及多种上下文条件（仅消息、局部上下文、外部知识）进行对比评测

**📊 数据集**

选取了从Discord收集的1,280,548条消息中人工挑选的100条难以解读的有害信息

**📈 对比分析**

在人类与LLM在相同局部上下文条件下进行比较，LLM在匹配率上远高于人类（LLM 58% vs 人类 5.3%），但在完整上下文与外部知识条件下人类表现优异（匹配率 62.7%）

**⚠️ 局限性**

局限性包括：样本仅为100条人工挑选的消息、仅关注文本数据、使用非母语评测、参考解释可能不完全准确、LLM实验未做检索增强或提示优化、分类体系需在更大数据上进一步验证

---

## 65. A Word-Level Digital Reader of the Prasthanatrayi with Sankara's Bhasya: Corpus, Method, and an Open, Offline Reading Aid for the Advaita Vedanta Canon

**arXiv ID:** 2607.07282 | [PDF](https://arxiv.org/pdf/2607.07282v1)

**作者:** Tamal Maharaj `[一作]` `[通讯]` (Ramakrishna Mission Vivekananda Educational and Research Institute), Tamal Maharaj (Ramakrishna Mission Vivekananda Educational and Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个离线、可点击单词、带词法分析与词根索引的Advaita Vedānta《Brahma Sutras》及其13篇注释的数字阅读器。

**💡 创新点**

创新点包括：①将根文本与注释整合到统一词级分析体系；②采用规则+LLM双重校验与对抗性验证；③创建可持续的专家回顾叠加机制；④提供全文词根搜索的并发索引。

**🔧 技术方法**

技术实现基于Python脚本自动化构建、Huet Heritage 形态词典、Cologne 词典词根、规则基分词与语法分析器、Claude Opus LLM 两轮验证、JavaScript/HTML离线前端。

**📊 数据集**

使用数据集为《Brahma Sutras》13篇注释（约2,971个章节）、95,587种注释表面形式、Huet Heritage 927k形态词典、Digital Corpus 240k注释表面、Cologne 词典词根及 Dhātupāṭha。

**📈 对比分析**

评估方法：按自评信度分段，对高/中/低信度分别与Heritage词典进行形态和分词一致性检验，结果高信度≈99%一致；独立抽样（180例）低信度全为错误；整体性能表现良好，错误集中于低信度区间。

**⚠️ 局限性**

局限性包括：评论文本仅按表面形式一次分析，缺乏上下文消歧义；LLM仍可能产生自信错误；低信度词法易被误标；覆盖范围受词典限制，需进一步人力校对。

---

## 66. AI for Cultural Heritage Textiles: Fine-Tuned Latent Diffusion for Novel Ulos Motif Synthesis

**arXiv ID:** 2607.06590 | [PDF](https://arxiv.org/pdf/2607.06590v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 67. A knowledge-augmented dataset of high-risk driving scenarios with LLM annotations for autonomous driving

**arXiv ID:** 2607.07103 | [PDF](https://arxiv.org/pdf/2607.07103v1)

**作者:** Heye Huang `[一作]` (Korea Advanced Institute of Science and Technology), Jianqiang Wang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本工作构建了一个名为K‑Risk的知识增强高危驾驶场景数据集，汇集了来自20个公开轨迹数据源（包括人类驾驶和自动驾驶），共筛选出31,398个高危事件（1,036个极危子集），并为每个事件提供结构化场景描述、异常行为提醒和基于大语言模型的风险分析与行动建议；同时实现了闭环验证机制，将推荐行动在无碰撞模拟器中检验。

**💡 创新点**

创新点包括①将多源、跨洲（欧、美、中）轨迹数据统一整合并通过驾驶员风险场（DRF）、硬动作阈值与两秒轨迹冲突预测三步筛选，系统性地提取高危事件；②在事件级别提供结构化语义注释与LLM生成的因果风险分析，兼顾可解释性与可训练性；③通过闭环验证与模拟器反馈形成可重复的风险评估与学习循环，形成极危子集的可验证决策样本；④构建统一的三层文件格式（CSV轨迹、JSON元数据、文本叙述），兼容轨迹预测、风险评估、LLM推理等多任务。

**🔧 技术方法**

主要技术手段包括：驾驶员风险场（DRF）计算、硬动作（加速、刹车、变道）阈值检测、时间到碰撞（TTC）与两秒轨迹冲突判定、基于规则的场景描述生成、LLM（如GPT）生成因果风险分析与行动推荐、闭环验证（无碰撞仿真+反思）、Python科学栈与LLM客户端、JSON/CSV/文本同步存储。

**📊 数据集**

使用的数据集为20个公开轨迹源：人类驾驶（highD、inD、rounD、ExpresswayA、FreewayB、I‑80）和自动驾驶（Argoverse 2、Waymo Open Motion、Waymo Open Perception、MicroSimACC、OpenACC Casale、Vicolungo、AstaZero、ZalaZONE、CATS ACC、CATS Platoon、CATS UWM、Central Ohio、Vanderbilt）。

**📈 对比分析**

与现有数据集相比，K‑Risk在高危事件覆盖率、事件级语义标注、LLM可训练样本和闭环验证方面显著优越。闭环验证实验显示，在极危子集上，三轮迭代后碰撞率从4.58%降至1.91%，相当于58.3%的相对降低，验证了数据集在风险评估与决策训练中的有效性。

**⚠️ 局限性**

局限性包括：自动驾驶源的高危事件稀缺，导致AV子集规模相对较小；风险定义基于校准阈值，缺乏统一的概率不确定性或多主体意图建模；数据集未包含原始完整轨迹，仅提供事件级片段；LLM生成的风险分析需人工审核，可能存在主观性。

---

## 68. Retrieving and Refining Winning Noise Tickets for Diffusion-Based Motion Generation

**arXiv ID:** 2607.06843 | [PDF](https://arxiv.org/pdf/2607.06843v1)

**作者:** Sakuya Ota `[一作]` (Institute of Science Tokyo), Ikuro Sato `[通讯]` (Institute of Science Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种训练无关、模型无关的框架WINRO，通过检索并细化初始噪声（称为Winning Noise Ticket）来提升基于扩散的文本驱动动作生成的语义一致性与时序连贯性。

**💡 创新点**

创新点在于：① 发现并利用扩散模型初始噪声空间中的语义结构；② 设计无训练的噪声字典检索（WIN-R、WIN-FAR）以及KL正则化的噪声细化（WINRO）；③ 可通过LoRA快速推理；④ 通过单一噪声级联实现动作风格化与空间约束控制。

**🔧 技术方法**

核心技术包括：扩散式文本到动作模型（MDM、MotionLCM）、文本–动作检索嵌入（TMR）、噪声字典构建与检索、KL正则化的噪声优化、LoRA低秩自适应噪声修正。

**📊 数据集**

使用的公开数据集有HumanML3D（文本到动作）、MMT（多轨长序列）、100STYLE（风格化）以及在实验中对比使用的MotionLCM、MDM、SMooDi等模型。

**📈 对比分析**

与基线模型（未做噪声优化）比较，WINRO在HumanML3D上将FID从0.438降至0.115、R-Top1提升至0.539；在MOTLCM上FID降至0.072、R-Top1提升至0.580；在MTT长序列上语义正确率从33.6%提升至62.1%。LoRA版本大幅降低推理时间（81.408→0.280s），同时保持较高的文本对齐性能。

**⚠️ 局限性**

局限性包括：① 仅在扩散模型已学习的运动空间内引导，无法修复模型自身的根本失效；② 迭代细化阶段计算成本高；③ 当前检索模型全局处理动作，难以精准定位细粒度时间属性；④ LoRA需额外训练步骤。

---

## 69. Reliable and Developer-Aligned Evaluation of Agents for Software Engineering

**arXiv ID:** 2607.06713 | [PDF](https://arxiv.org/pdf/2607.06713v1)

**作者:** Razvan Mihai Popescu `[一作]` `[通讯]` (Delft University of Technology), Razvan Mihai Popescu (Delft University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统地评估和改进LLM驱动编码代理在真实软件开发中的可靠性与开发者对齐度。

**💡 创新点**

提出多维评估框架，结合污染意识、真实工作环境评估和轨迹感知指标，填补现有方法与实践间差距。

**🔧 技术方法**

采用系统综述、长期仓库行为分析、代理交互记录、定制化指标与多语言污染敏感基准等技术。

**📊 数据集**

使用279篇论文与26项编码任务的数据，基准包括HumanEval、MBPP、Defects4J、SWE-Bench等，并构建公开代理与人类贡献数据集。

**📈 对比分析**

与传统BLEU/CodeBLEU等指标相比，提出的轨迹与污染感知指标更贴合开发者需求，评估结果显示传统基准已饱和，新的基准揭示模型真实失效模式。

**⚠️ 局限性**

仍受数据规模、代理多样性、评估成本与人类评测主观性等因素限制。

---

## 70. Multiplication Beyond Groups: Stratified Fourier Mechanisms in Transformer Circuits

**arXiv ID:** 2607.07066 | [PDF](https://arxiv.org/pdf/2607.07066v1)

**作者:** Zitong Andrew Chen `[一作]` (University of Washington), Jarod Alper `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究单层 transformer 在复合模数下整数乘法任务中的内部计算机制，通过层级化的嵌入、注意力路由、MLP 合成和最终解码，揭示其如何实现非群运算。

**💡 创新点**

提出“Monoid Extension”——将 Group Composition via Representation (GCR) 的思想推广到单子（monoid）结构，说明局部 J-类内的 Fourier 角色与局部逆可解释模型输出，并引入局部路由与分层子空间投影。

**🔧 技术方法**

结合离散傅里叶变换 (DFT)、PCA、注意力头解构、线性回归 FVE、频谱分析、主角角度对齐等可解释方法，对嵌入、注意力、MLP 与解码层进行逆向工程。

**📊 数据集**

使用整数模数 n=165（及其他复合模数）全乘法表作为训练和评估数据集，覆盖所有可能的输入对 (a,b)。

**📈 对比分析**

将模型的 logits 与理论上基于局部字符 (χ_ρ(abc^♯)) 的预测对比，计算 R²；结果显示在各 J-类中 71%–99% 的方差被理论特征解释，证明了局部 Fourier 机制的有效性。

**⚠️ 局限性**

局限性：仅为相关性分析，未通过因果干预验证；仅考察平方自由模数，未覆盖非正规 J-类；模型规模极小，缺乏对更大 LLM 的验证；对非平凡模数的行为仍待研究。

---

## 71. PriGo: Test-Time Primitive Guidance to Diffusion and Flow Policies for Adaptive Robotic Manipulation

**arXiv ID:** 2607.07076 | [PDF](https://arxiv.org/pdf/2607.07076v1)

**作者:** Zezeng Li `[一作]` (École Centrale de Lyon), Liming Chen `[通讯]` (École Centrale de Lyon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个在测试时通过原语引导来提升预训练的扩散与流动式策略鲁棒性和泛化的框架PriGo。

**💡 创新点**

创新点在于：①在测试时使用轻量级原语分类器PANet直接从观测中预测原语分布；②引入可微分原语引导机制，在推理阶段对生成的动作进行梯度优化，使其符合语义一致的原语；③实现了无训练需求、可插拔的框架，可在任何预训练的扩散或流动策略上无缝应用。

**🔧 技术方法**

技术包括：视觉语言预训练模型（DINOv2, T5）、多模态融合、softmax/交叉熵原语引导、扩散/流动匹配政策的梯度调整。

**📊 数据集**

使用了LIBERO、CALVIN、SIMPLER等大型模拟数据集以及真实Franka Emika Panda机器人实验。

**📈 对比分析**

与多种SOTA方法（π_0, SmolVLA, DP, CogACT, 3DDA, ADPro, DTP等）比较，PriGo在多项指标上提升3–7个百分点，尤其在长时序和零样本任务上显著提高成功率。

**⚠️ 局限性**

局限：无训练提升有限，受限于无重训练的设计；PANet的泛化受模型规模与训练数据限制，缺乏历史状态等信息可能影响性能。

---

## 72. WHERE to Generate Matters: Budget-Aware Synthetic Augmentation for Label Skewed Federated Learning

**arXiv ID:** 2607.06616 | [PDF](https://arxiv.org/pdf/2607.06616v1)

**作者:** Sangwoo Lee `[一作]` (Chung-Ang University), Jaewoo Lee `[通讯]` (Chung-Ang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为FedEAS的策略，通过为每个客户端分配基于其本地标签分布的熵自适应每类生成预算，来解决联邦学习中的标签偏斜问题。

**💡 创新点**

创新点在于通过熵自适应的预算分配策略，动态决定每个客户端生成的样本数量和样本分配位置，从而在减少生成预算的同时恢复大部分全类平衡的准确性。

**🔧 技术方法**

使用了熵自适应预算分配策略，结合了类条件去噪扩散概率模型（DDPM）作为生成器。

**📊 数据集**

在CIFAR-10和CIFAR-100数据集上进行了实验，采用Dirichlet分区引入标签偏斜。

**📈 对比分析**

在相同的总生成预算下，FedEAS在CIFAR-10和CIFAR-100上比均匀分配方法的准确性提高了最多18.82%。在CIFAR-10上，FedEAS的生成样本数量减少了94.1%，并且在单个GPU上运行时间减少了3.7倍。

**⚠️ 局限性**

限制在于每个类条件的DDPM是基于完整训练集训练的，这限制了其在实际部署中的可行性。此外，生成样本的质量可能存在随机性，且未能保证每个样本的保真度。

---

## 73. Residual-Conservative Model Predictive Path Integral Control

**arXiv ID:** 2607.06950 | [PDF](https://arxiv.org/pdf/2607.06950v1)

**作者:** Hyung-Jin Yoon `[一作]` (Tennessee Technological University), Hunmin Kim `[通讯]` (Mercer University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于残差自适应的采样式模型预测控制框架RC‑MPPI，用以在模型-执行不匹配的条件下在线调节安全保守性。

**💡 创新点**

创新点在于三项耦合机制的联合使用：残差依赖的约束收紧、可变安全惩罚缩放以及残差自适应的采样调节（温度松弛与探索收缩），并将温度解释为对滚动代价评估置信度的表征，配合理论安全与不确定性分析。

**🔧 技术方法**

采用MPPI（模型预测路径积分）核心框架，加入残差滤波、约束变形、软势能惩罚与自适应温度/采样方差调整，利用子高斯扰动、Lipschitz连续性等假设进行概率安全性与权重敏感性分析。

**📊 数据集**

实验数据来自两套模拟环境：1) LTI点质量系统（配备激活延迟）；2) 平面2R机械臂（含惯量不匹配、饱和、测量噪声）。每套系统进行50次配对种子蒙特卡洛实验。

**📈 对比分析**

与标准MPPI做对比，RC‑MPPI在成功率、最小安全距离、违规步骤、路径长度与能耗等指标均有显著提升；在点质量系统成功率从0.64提升至0.94，违规步数降至0.62；在2R臂系统成功率从0.56提升至0.96，时间-目标步数大幅减少，能耗降低近一半。

**⚠️ 局限性**

局限性包括：依赖于参数手动调节（如阈值、缩放因子、温度上下限）；理论分析基于Lipschitz、子高斯等理想假设；未在真实硬件上验证；温度上限可能导致过度保守，需进一步研究自适应上限或多尺度融合。

---

## 74. Multiple Double Arithmetic on NVIDIA Tensor Cores

**arXiv ID:** 2607.06881 | [PDF](https://arxiv.org/pdf/2607.06881v1)

**作者:** Howard Chen `[一作]` (University of Illinois at Chicago), Jan Verschelde `[通讯]` (University of Illinois at Chicago)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种利用NVIDIA Tensor Core实现多倍精度（多重双精度）矩阵乘法的方法，并实现相应的软件库。

**💡 创新点**

创新点在于：① 将多重双精度矩阵拆分为八个双精度子矩阵并重写为一次双精度矩阵乘法；② 通过“四分”与“平衡”算法保证子矩阵的指数在格点上，从而避免归一化中的分支与精度损失；③ 结合分支无关的Ozaki方案在Tensor Core上实现高效多重双精度计算。

**🔧 技术方法**

使用了CUDA Toolkit、dmmaTensorCoreGemm（WMMA核）、自定义的平衡与拆分算法、FP64 Tensor Core指令、共享内存优化以及GPU GPL许可的开源实现。

**📊 数据集**

实验数据集为随机生成的多重双精度（双精度、四重、八重、十六重）矩阵，规模从 1k×1k 到 4k×4k（以及更高精度下的相应尺寸）。

**📈 对比分析**

对比方法：在同一GPU上分别使用Tensor Core（t_TC）与常规CUDA核（t_CUDA）执行矩阵乘法；同时测量仅WMMA核时间（t_WMMA）和误差（ϵ_max）。结果显示：WMMA核最快，Tensor Core整体时间（t_TC）仍比CUDA核慢，但误差保持在 10⁻²⁸ 至 10⁻⁵⁹ 范围；A100上可达 13.75 TFLOPS，超过常规 CUDA 核 9.7 TFLOPS；RTX 4080 上仅 0.5 TFLOPS。

**⚠️ 局限性**

局限性：① Tensor Core 的归一化需要额外的平衡与重构步骤，导致整体时间 t_TC 仍高于普通 CUDA；② 内存占用显著增加（四分导致 4×/8×尺寸扩张）；③ 仅针对 FP64 Tensor Core 设计，未验证 FP32 或混合精度场景；④ 仅在随机矩阵上测试，缺乏对实际应用（如 QR、Taylor 系列）性能的深入评估。

---

## 75. From Atomic Actions to Standard Operating Procedures: Iterative Tool Optimization for Self-Evolving LLM Agents

**arXiv ID:** 2607.07321 | [PDF](https://arxiv.org/pdf/2607.07321v1)

**作者:** Haipeng Ding `[一作]` (Renmin University of China), Bolin Ding `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EvoSOP 框架，让 LLM 代理通过从执行轨迹中提炼 SOP 并迭代优化工具集实现自我演化。

**💡 创新点**

结合工具合并、评估、审查的完整生命周期，实现工具的动态迭代生成与管理，而非一次性生成；利用非参数化结构优化提升效率。

**🔧 技术方法**

基于 ReAct 的 LLM 交互，构造器、合并器、评估器、审阅器四模块；从轨迹提取、代码生成、工具模式化实现工具集迭代；不更新模型参数。

**📊 数据集**

ACEBench（Multi-Step、Multi-Turn 子集）和 Tau2Bench（Telecom Solo 子集）两大基准数据集。

**📈 对比分析**

与 ReAct、DFSDT、ASI、DRAFT 等基线对比，EvoSOP 在 ACEBench 上成功率提升 2.5–13.4%，在 Tau2Bench 亦保持领先，同时显著减少推理回合。

**⚠️ 局限性**

受 LLM 推理不确定性、工具生成质量波动影响；对极大工具空间的可扩展性与通用性仍需进一步验证。

---

## 76. SA-DRL: Security-Aware Deep Reinforcement Learning for Ransomware Detection with Asymmetric Reward Design

**arXiv ID:** 2607.06880 | [PDF](https://arxiv.org/pdf/2607.06880v1)

**作者:** Jannatul Ferdous `[一作]` (Charles Sturt University), Md Zahidul Islam `[通讯]` (Charles Sturt University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于安全感知深度强化学习（SA-DRL）的勒索软件检测框架，利用行为特征进行实时判别。

**💡 创新点**

创新点包括：① 将FN与FP的成本不对称直接嵌入奖励函数；② 设计安全优先模型选择（SOMS）准则；③ 通过 episode 级随机排列实现隐式自适应样本加权。

**🔧 技术方法**

技术手段：深度强化学习（DQN、DDQN、PPO、A2C）、异步/同步奖励设计、随机折叠交叉验证、统计显著性检验（Friedman、Wilcoxon）、性能指标（FNR、F1、AUC、训练/推理时间）等。

**📊 数据集**

使用了 2000 条平衡行为特征数据集（1000 条勒索软件 30 个家族 + 1000 条正常样本），特征由 ANY.RUN Windows 11 运行时采集并预处理后得到 103 维向量。

**📈 对比分析**

实验设计：4 种 DRL 算法 × 2 种奖励函数 × 4 折扣因子 × 5 折交叉验证 × 3 种随机种子，共 480 次训练；结果显示 DDQN + 异构奖励（R2）+ γ=0.1 的配置在 SOMS 下取得 FNR 0.8%、F1 0.9915、AUC 0.998，FN 降低 67.6%（相较于 MLP 基线 FNR 2.47%）。

**⚠️ 局限性**

局限性：仅评估 2000 条平衡样本，未覆盖严重类别不平衡；仅做二分类，未考虑家族识别或预警；实验为离线静态评估，缺乏在线持续学习与概念漂移适应；奖励比例（4:1）在不同场景下可能需重新校准。

---

## 77. Predicting LLM Safety Before Release by Simulating Deployment

**arXiv ID:** 2607.07184 | [PDF](https://arxiv.org/pdf/2607.07184v1)

**作者:** Marcus Williams `[一作]` (OpenAI), Micah Carroll `[通讯]` (OpenAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过保留历史对话前缀并使用候选模型重新生成下一条回复，构建部署仿真，进行误行为审计和预估。

**💡 创新点**

用生产对话前缀进行离线重采样，实现对实际部署环境的逼真模拟，减少评估偏差，提升对失误率的预测和新失误的发现。

**🔧 技术方法**

采用离线重采样、自动化审计流水线、工具模拟器、链式思维标注以及评估意识检测等技术。

**📊 数据集**

利用OpenAI GPT‑5系列去标识化生产对话（约130万条）以及公共数据集WildChat。

**📈 对比分析**

与传统“挑战性提示”与先前版本基线对比，采用对数相关、对称乘法误差和负对数似然等指标，结果显示部署仿真在预测误行为频率与方向性上优于传统方法，平均误差约为2–5倍。

**⚠️ 局限性**

依赖准确标注器，对前缀分布变化敏感，无法充分覆盖极端尾部风险，公共数据代表性不足，以及工具状态重现难度高。

---

## 78. EscFOA: Enhancing Spatial Learning for Visually Impaired Learners via Generative Spatial Audio in 360-Degree Educational Environments

**arXiv ID:** 2607.07015 | [PDF](https://arxiv.org/pdf/2607.07015v1)

**作者:** Ziyu Luo `[一作]` (Beijing Technology and Business University), Xiaoming Chen `[通讯]` (Beijing Technology and Business University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 EscFOA，基于几何感知的生成空间音频框架，提升360°沉浸式教育环境中视障学习者的空间认知与导航能力。

**💡 创新点**

将场景几何结构与音频生成耦合，利用 3D 高斯切片恢复空间布局并通过条件扩散模型合成符合几何的 FOA，作为学习导向的声学支架。

**🔧 技术方法**

3D Gaussian Splatting（3DGS）用于重建场景几何，条件扩散模型（类似 DynFOA）用于生成 FOA，配合第一阶 Ambisonics（FOA）与头部追踪实现空间音频渲染。

**📊 数据集**

使用 Sphere360 数据集中的课堂与街景 360° 视频，评估在单声道、立体声与 EscFOA 三种音频条件下的学习体验。

**📈 对比分析**

通过盲fold 受试者的导航轨迹、碰撞数与 MOS 评分对比，EscFOA 在空间定向更平稳、碰撞更少、感知易用性、听觉舒适度与导航自信度方面均显著优于传统单声道与立体声。

**⚠️ 局限性**

仅在盲fold 受试者上评估，未涉及真实视障学习者；缺乏长期学习效果和多样化教育内容的验证。

---

## 79. SPEAR: A Simulator for Photorealistic Embodied AI Research

**arXiv ID:** 2607.06701 | [PDF](https://arxiv.org/pdf/2607.06701v1)

**作者:** Mike Roberts `[一作]` (Adobe Research), Vladlen Koltun `[通讯]` (Intel Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

SPEAR 提供了一个 Python 库，可通过模块化插件架构以程序化方式控制任意 Unreal Engine (UE) 应用，支持 14K+ UE 函数调用，并能在 1920×1080 分辨率下以 73 FPS 生成高质量图像，同时提供丰富的地面真值模态。

**💡 创新点**

创新点在于：① 通过 UE 反射系统动态暴露全部可反射函数，实现了十倍以上的可编程性；② 采用异步命令和共享内存，实现了比现有 UE 插件快 10–20 倍的渲染吞吐；③ 设计了事务式图形编程模型，允许在单帧内执行复杂依赖图；④ 支持多代理、多视角、协同仿真及自然语言场景编辑。

**🔧 技术方法**

使用技术包括：Python 生态（nanobind、rpclib）、UE 反射系统、C++ 与 Python 的 RPC 通信、异步任务队列、共享内存（GPU 到 NumPy 无拷贝）、SpFunctions 定制接口、以及多线程服务器/客户端架构。

**📊 数据集**

实验数据集主要来自 Epic Games 的 Sample Projects（如 CitySample、HillsideSample、MetaHumans 等）以及 Hypersim 的图像模态；通过这些场景评估渲染质量、速度和功能覆盖率。

**📈 对比分析**

与 AirSim、CARLA、UnrealCV+ 等现有 UE 仿真器比较，SPEAR 在 1920×1080 解析度下渲染吞吐率提升 12–21 倍，帧率提升至 73 FPS，且支持更多地面真值模态；在统一硬件环境下，通信与渲染延迟显著降低，整体性能优于现有方案。

**⚠️ 局限性**

局限性包括：仍需在 UE 项目中植入 SPEAR 插件，对非 UE 开发者的可迁移性有限；共享内存和异步机制对硬件和操作系统支持有一定要求；部分 UE 功能（如不在反射系统内的自定义功能）仍需手写绑定；对大型团队的协同使用需要额外的版本管理与调试工具。

---

## 80. Latency-Aware Bid Acceptance under Operational Feasibility: A Public Benchmark with Hindsight Ceilings

**arXiv ID:** 2607.07343 | [PDF](https://arxiv.org/pdf/2607.07343v1)

**作者:** Aswin Chandrasekaran `[一作]` `[通讯]` (Bubba AI), Aswin Chandrasekaran (Bubba AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了公开可复现的 FreightBidBench v0.3 试验平台，用以评估在线卡车装载投标接受决策，并在此平台上实现了多种基准策略。

**💡 创新点**

创新点包括：①在 MDP 奖励中加入服务失败惩罚、终端车队价值和时段价格溢价三层，以实现对可行性盲、未来盲和未来感知策略的分离；②提出两种后向诊断上限——精确小前缀动态规划与保留每辆车 HOS/位置约束的拉格朗日信息放松；③设计可调参数的 surrogate‑rollout 并发策略，展示了延迟‑利润权衡的闭环前沿。

**🔧 技术方法**

技术手段包括：有限时域马尔可夫决策过程建模；拉格朗日信息放松、线性松弛与完整前向动态规划；线性回归 surrogate 与有限看ahead rollout 教师；并通过参数化的层级触发器实现策略级联。

**📊 数据集**

使用公开数据：Freight Analysis Framework（FAF）和 USDA 货运费率表，结合公共时空路段表构建需求与路线信息，确保不依赖任何私有运营数据。

**📈 对比分析**

方法通过与 rollout 教师（有限 lookahead 蒙特卡罗）对比，利用精确前缀 DP 作为可信底线，并用 LP 与拉格朗日上限评估可行性。实验显示：最优 surrogate 能保留 94‑95% 的最佳简单策略收益，级联策略在 β=500、κ=2 时在两种极端情境下平均恢复约 98% 的 rollout 利润，且决策延迟仅为 rollout 的 40‑56%。

**⚠️ 局限性**

局限性包括：基准仍为合成且仅用公开数据，忽略了更复杂的 HOS 规则与真实运作细节；rollout 教师仅是近似oracle；精确 DP 仅能处理前 12-16 个装载；全局上限仍相对宽松；未包含学习型基线（如梯度提升树、神经网络）以保证复现性。

---

## 81. Safe Reinforcement Learning using Ideas from Model Predictive Control

**arXiv ID:** 2607.07252 | [PDF](https://arxiv.org/pdf/2607.07252v1)

**作者:** Georg Schäfer `[一作]` (Salzburg University of Applied Sciences), Simon Hirlaender `[通讯]` (Paris Lodron University of Salzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过离线MPC预先构造可行状态-动作空间，并在RL训练与部署期间使用投影过滤器，将代理产生的原始动作映射到该安全集内，从而实现安全探索和稳定的策略收敛。

**💡 创新点**

创新点在于：①将离线MPC作为全局可行性判别器，提前计算递归安全边界；②利用控制系统的凸性仅需查询最小/最大安全动作，避免全空间搜索；③将投影过滤器嵌入实时控制循环，实现高频、无额外延迟的安全保障。

**🔧 技术方法**

使用技术包括：深度强化学习（PPO），离线模型预测控制（MPC）与凸性搜索（二分法），投影过滤器（Euclidean投影），Python/Simulink桥接实现零射击部署，以及对非线性1-DOF飞行器动力学的辨识。

**📊 数据集**

数据集/实验平台：Quanser Aero 2 1-DOF俯仰控制实验台（机械限制±60°），使用该平台的真实传感器/执行器进行物理实验，结合离线仿真环境做训练。

**📈 对比分析**

与传统无约束RL（如PPO无过滤器）或在线约束方法（如基于CBF的遮蔽）比较，结果表明：①安全约束违规率显著降低；②训练过程稳定、收敛速度与无约束RL相近；③物理硬件实验中几乎没有致命失控事件。

**⚠️ 局限性**

限制：①安全保证高度依赖模型精度，存在Sim-to-Real差距导致偶发违规；②离线网格搜索在高维系统上面临指数式复杂度（维数灾难）；③投影过滤器在极限情况下仍可能因数值误差、采样延迟或解算器容差导致微小违规。

---

## 82. Video-Based Detection of squint and cataract for accessibility-aware adaptive web interface rendering

**arXiv ID:** 2607.07099 | [PDF](https://arxiv.org/pdf/2607.07099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 83. Compositional Motion Generation from Demonstration with Object-Centric Neural Fields

**arXiv ID:** 2607.07129 | [PDF](https://arxiv.org/pdf/2607.07129v1)

**作者:** Ahmet Ercan Tekden `[一作]` (Chalmers University of Technology), Yasemin Bekiroglu `[通讯]` (Chalmers University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `afceb026-1760-41ae-8d86-010831a37d97` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于生成式学习示范的框架，通过对象级的共享潜在空间实现感知与运动的耦合，实现机器人行为的可组合建模。框架包含空间混合专家（Spatial MoE）对场景进行对象级神经场建模，并在时间维度上使用混合专家（Temporal MoE）对运动轨迹进行分段生成；同时引入潜在重标记（latent relabeling）和FiLM条件化以提升可解释性与数据效率。

**💡 创新点**

创新点在于：1）将场景建模与运动生成统一到一个对象级潜在空间，形成空间–时间两级混合专家；2）采用神经场与柔性变形网络实现平滑对象变换；3）通过角点集合（corner set）和梯度搜索实现潜在空间的可逆映射；4）利用时间可变加权实现不同对象对不同轨迹段的贡献，增强组合性；5）结合语言基分割实现类别级迁移，提升泛化能力。

**🔧 技术方法**

核心技术包括：对象级神经场（坐标基 MLP）、空间混合专家、潜在重标记算法、时间混合专家、FiLM 条件化、潜在丢弃、Lipschitz 正则化、可扩展的占据场（occupancy field）实现3D场景、以及在实验中的 CNN-FiLM、DINO-FiLM、CNMP、Diffusion Policy 等基线方法。

**📊 数据集**

使用自建的模拟数据集（Wall Avoidance、Incline Pick‑and‑Place、Cup Stacking、Cube Stacking）和多种真实机器人演示数据（桌面拾取、球/杯子放置、抽屉装箱）。演示数据通过人为示范、键盘或抓取状态标记获得，示例数量极少（10–30条/任务）。同时利用语言提示驱动的分割模型为类别级任务生成掩码。

**📈 对比分析**

在低数据（10–30条） regime 下与 CNMP、Diffusion Policy、NFMP、CNN-FiLM、DINO-FiLM 等基线进行对比，实验显示我们的模型在平均欧氏距离（MED）和任务成功率上均取得显著优势；在高数据 regime 下，基线在多数任务上与我们相当或略逊。数据效率实验表明，模型在仅几条演示下即可达到与基线相同或更优的性能；在真实机器人实验中，成功率达到 96–100%，并能应对视觉噪声、类别迁移与多目标场景。

**⚠️ 局限性**

局限性包括：1）需要粗略对象掩码，且假设背景与摄像头视角不变；2）对演示间的时间对齐要求较高；3）在视觉相似的干扰物或多目标重叠时可能出现识别混淆；4）潜在空间的可扩展性受限于角点集合，难以外推到未见变化；5）潜在搜索与模型推理相对耗时；6）3D 扩展需要场景扫描，限制了在线部署。

---

## 84. Geometric--Nongeometric Optimizer Calculus: A Modular Language for Reachable Gradient Methods

**arXiv ID:** 2607.07206 | [PDF](https://arxiv.org/pdf/2607.07206v1)

**作者:** Zavier Li `[一作]` `[通讯]` (Xidian University), Zavier Li (Xidian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一种几何‑非几何优化器算子框架，拆分梯度更新为几何模块和非几何模块，并给出了方向可表达性定理、受限方向残差诊断与轨迹残差一致性度量。

**💡 创新点**

创新点包括：正式化方向可表达性定理；定义受限方向残差并证明对角/块几何的精确条件；把优化器设计转化为带预算的 Pareto 优化问题；以及提出轨迹级残差一致性度量。

**🔧 技术方法**

使用的技术包括正定共度量、对角/块几何残差计算、预算约束的 Pareto 设计、轨迹残差与几何变异度量、以及诊断性原型实现。

**📊 数据集**

实验数据集主要是确定性强凸二次函数（自研 4c 量化基准）和 MNIST 子集（小规模训练）。

**📈 对比分析**

通过与全矩阵步、Krylov 子空间、LBFGS、Adam 等基准比较，展示在全信息量化二次实验中全矩阵实现精确收敛；在小规模 MNIST 上 Muon 风格候选在准确率上略优，但未与主流优化器做大规模对比，性能结果仅为诊断性观察。

**⚠️ 局限性**

局限性在于实验规模小、未与强基准优化器做系统性对比、对角/块残差依赖坐标、仅提供诊断性结果而非性能主张。

---

## 85. PLED-VINS: A Point-Line Event-Based Visual Inertial SLAM for Dynamic Environments

**arXiv ID:** 2607.07374 | [PDF](https://arxiv.org/pdf/2607.07374v1)

**作者:** Seunghun Lee `[一作]` (Korea Advanced Institute of Science and Technology), Hyun Myung `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于事件相机的单目视觉-惯性SLAM框架PLED-VINS，利用时间熵-近期度量评估特征的时间可靠性，并与几何可靠性融合，实现动态环境下的鲁棒状态估计。

**💡 创新点**

创新点在于①引入熵-近期度量以捕捉运动一致性的时间特征；②统一点线鲁棒BA评估几何可靠性；③自适应融合时间与几何可靠性，并为线特征设计运动条件加权，以显著抑制动态观测。

**🔧 技术方法**

使用事件时间表面与活动事件表、IMU运动补偿、熵-近期度量、统一点线鲁棒BA、IMU预积分滑动窗口、线段检测与追踪以及自适应权重融合技术。

**📊 数据集**

在VIODE（仿真图像+IMU+合成事件）、DAVIS 240C（真实事件+图像+IMU）和DSEC（高分辨率驾驶序列）三大数据集上进行评估。

**📈 对比分析**

与PL-VINS、DynaVINS、PL-EVIO、E2-VINS、ESVIO等公开基线比较，在动态水平高的序列中均取得最低ATE/MPE，平均误差提升至0.1–0.3 m；在DAVIS 240C上MPE降至0.089 %/m。

**⚠️ 局限性**

主要局限在于仅假设单一刚体运动的主运动假设，难以应对多体运动或深度相关的视差效应，在复杂多运动场景中的区分能力有限。

---

## 86. FedCVESA: Taking Away Training Data in Federated Learning via Correlation Value Encoding and Segmented Aggregation

**arXiv ID:** 2607.07314 | [PDF](https://arxiv.org/pdf/2607.07314v1)

**作者:** Chongkai Li `[一作]` (Harbin Institute of Technology), Wenjian Luo `[通讯]` (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 FedCVESA，利用白盒攻击在联邦学习中主动将目标客户端的训练数据写入全局模型的参数中。

**💡 创新点**

创新点在于将 Correlation Value Encoding 改造为 FL 场景，并设计分段聚合与分散载体参数，减少编码信息被平均覆盖。

**🔧 技术方法**

主要技术包括 Pearson 相关正则化、分段聚合、分散载体参数映射和无优化的参数解码。

**📊 数据集**

实验使用 MNIST、Fashion‑MNIST 与 CIFAR‑10 三个公开图像数据集，并通过 Dirichlet 非IID 划分。

**📈 对比分析**

通过与无攻击基线比较，FedCVESA 在保持 99%+（MNIST）/90%+（Fashion‑MNIST）/80%+（CIFAR‑10）准确率的同时，将 MAPE 降到 0.05–0.25 之间，显示出良好的隐私泄露与任务性能平衡。

**⚠️ 局限性**

局限性包括仅在白盒恶意服务器假设下验证，实验规模受限于小型数据集，且对更复杂模型或异构设备的适用性尚未评估。

---

## 87. Interpretable Uncertainty for Adaptive Retrieval and Reasoning in Question Answering

**arXiv ID:** 2607.07380 | [PDF](https://arxiv.org/pdf/2607.07380v1)

**作者:** Ritajit Dey `[一作]` (University of Glasgow), Graham McDonald `[通讯]` (University of Glasgow)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于LLM隐藏状态的可解释不确定性估计框架，用于自适应检索和推理的问答系统。

**💡 创新点**

将知识缺失与知识冲突/歧义分离，并在单步前向传递中用轻量级回归探针从隐藏状态预测两种不确定性，随后通过阈值决定检索或额外推理，形成可解释的决策流程。

**🔧 技术方法**

使用轻量级回归探针估计事实出现频次与熵、阈值决策机制、检索增强生成（RAG）以及链式思考/自一致性推理等技术。

**📊 数据集**

在NQ（自然问答）数据集上，基于Llama‑2‑7b‑chat模型进行实验。

**📈 对比分析**

与仅LLM基线和始终开启RAG的基线做McNemar检验比较，结果显示在知识不足触发RAG时分别提升+5.9%（SE）和+4.7%（WEPR），优于常规RAG提升+3.3%和+2.1%。

**⚠️ 局限性**

依赖预训练语料统计的代理估计可能受训练数据偏差影响；阈值选择需人工调优；未在多任务或跨语言场景验证。

---

## 88. Topological Signatures of Diffusive Release in Porous Media

**arXiv ID:** 2607.07061 | [PDF](https://arxiv.org/pdf/2607.07061v1)

**作者:** Donghan Kim `[一作]` `[通讯]` (Korea Advanced Institute Of Science And Technology), Donghan Kim (Korea Advanced Institute Of Science And Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过持久同调分析多尺度拓扑特征来预测多孔介质的扩散释放曲线，并用简单分类模型区分释放行为

**💡 创新点**

发现即使在相同孔隙率条件下，固体相的拓扑复杂度（通过持久同调的持久性总和等指标）仍能显著区分早期快速、晚期释放和长尾释放三种曲线类型；持久同调特征提取速度远快于有限元扩散模拟

**🔧 技术方法**

使用稠密球体模型构造固体相，利用稳态扩散有限元求解释放曲线；对固体相进行尺度滤波得到滤波器 {F_r}，计算其 Čech/α 复形的持久同调，并提取 18 个持久性图摘要（计数、总和、均值、最大值、出生均值等），随后用多项式逻辑回归进行三分类

**📊 数据集**

在单元立方体中随机或结构化布置球体生成 6 类合成样本，每个孔隙率（40%、60%、80%）下生成 720 个样本；网格分辨率为 16³ 或 32³，进行 voxelization、tetrahedral FEM 和持久同调特征提取

**📈 对比分析**

与基于 FEM 的完整扩散模拟相比，持久同调特征提取时间仅占 1–2%（在 16³ 网格为 6–65 倍，在 32³ 网格为 82–630 倍），分类准确率在 0.64–0.76 之间，明显高于 1/3 的随机猜测，尤其在 40% 孔隙率时表现最佳

**⚠️ 局限性**

只针对球体障碍的合成多孔结构，缺乏对真实微观 CT 图像或药物释放系统的验证；高孔隙率下高维拓扑特征区分度下降；未探索更复杂的非线性模型或多尺度组合特征

---

## 89. Constrained Decoding for Diffusion Language Models via Efficient Inference over Finite Automata

**arXiv ID:** 2607.07026 | [PDF](https://arxiv.org/pdf/2607.07026v1)

**作者:** Meihua Dang `[一作]` (Stanford University), Stefano Ermon `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种针对扩散语言模型的精确约束解码算法，保证在每一步去噪时采样满足任意有限自动机表达的结构约束，从而得到满足约束的完整序列。

**💡 创新点**

创新点在于：①将有限自动机视为图模型，构造可解的链式概率模型，使得在全因子化均值场预测与约束的乘积可精确采样；②提出对该链式模型进行对数深度树状采样，显著降低序列长度线性依赖，提升并行度；③支持任意遮蔽（remasking）策略，兼容并行/分块解码，且对采样温度无关。

**🔧 技术方法**

核心技术包括：有限自动机与隐藏马尔可夫模型的对应关系；前向-后向消息传递与乘积构造实现约束后验；对数深度树形采样（基于算术电路深度压缩）；使用受约束后验的边缘概率做置信度进行遮蔽；在Dream‑7B、LLaDA‑8B等扩散 LLM 上实现。

**📊 数据集**

使用了多种任务数据集：函数调用（xLAM、BFCL）、数独（Sudoku）、Countdown 计数游戏、符号数学推理（GSM‑Symbolic）、文本到 SQL（Spider）。每个任务都用相应的 DFA/NFA 编码约束。

**📈 对比分析**

通过与无约束扩散 LLM 基线、以及部分自回归 LLM 的对比，证明在贪婪与随机采样两种模式下，约束解码将准确率提升 5–12%（例如 BFCL‑Live JSON 从 63.9% 提升到 71.5%，随机采样从 22.3% 提升到 69.0%），并保持约束满足率 100%。整体推理时间仅增加约 4–5% 的壁钟时间，且在树形采样模式下可进一步压缩延迟。

**⚠️ 局限性**

局限性包括：只能处理有限自动机约束，无法覆盖更丰富的上下文无关语法约束；对极大状态空间的 NFA 计算成本高；若被滥用，能够生成语法合法但潜在恶意的函数调用或 SQL 查询，存在安全风险。

---

## 90. Cost-Effective Agent Harnesses for Abstract Reasoning and Generalization on ARC-AGI-1

**arXiv ID:** 2607.06764 | [PDF](https://arxiv.org/pdf/2607.06764v1)

**作者:** Kabir Moghe `[一作]` (Dartmouth College), Peter Chin `[通讯]` (Dartmouth College)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了两种基于agent的架构——Explorer-Definer Pipeline 与 Reflective Orchestrator，提升ARC-AGI-1的性能，而不使用任何Benchmark‑specific微调或高计算量的推理。

**💡 创新点**

提出将模式发现与程序合成分离的两阶段管线，并通过中途触发新抽象的自我调节 orchestrator，验证了仅通过架构改进即可突破传统模型的成本‑效能瓶颈。

**🔧 技术方法**

使用OpenAI风格的工具调用、scratchpad、自然语言压缩中介、可执行Python验证、DeepSeek V3.2（无思考模式）、无监督的unbiased pass@k诊断等技术；所有推理均在同一token计价下完成。

**📊 数据集**

在ARC‑AGI‑1公开评估集（400个任务）上进行完整实验，并在99个子集上对Qwen3‑235B进行跨模型验证。

**📈 对比分析**

在同一模型、同一接口、同一token计价条件下，采用匹配任务的paired bootstrap CI 对四种架构（one‑shot、CoT、Pipeline、Orchestrator）进行比较；Pipeline在$0.25/任务下取得57.5% pass@2，Orchestrator在$0.62/任务下取得67.25% pass@2，较基线提升52pp，且成本比前沿模型低数十倍。

**⚠️ 局限性**

仅在公开评估集上实验，可能与半私有排行榜存在差距；配置选择基于同一数据集，缺乏外部验证；实验仅在DeepSeek V3.2（及Qwen3子集）上，缺乏更广泛模型验证；单跑结果可能受运行噪声影响；未在ARC‑AGI‑2上评估，未探究更难任务的鲁棒性。

---

## 91. Modeling Misinformation as a Commons Problem

**arXiv ID:** 2607.06984 | [PDF](https://arxiv.org/pdf/2607.06984v1)

**作者:** Vrinda Malhotra `[一作]` `[通讯]` (George Mason University), Vrinda Malhotra (George Mason University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了一个基于代理的仿真模型，用来研究信息误导在注意力稀缺环境下如何通过全局信任库存与个人注意力分配的闭环反馈，导致可信信息与误导信息的不同扩散与网络结构演化。

**💡 创新点**

创新点在于：①将信任视为可消耗的公共资源，构建全局信任库存与注意力成本的反馈循环；②将此反馈机制与适应网络重连、边界信任学习等传统机制耦合，形成可解释的治理框架；③通过多阶段验证（基线还原、扰动实验、压力测试）明确不同反馈渠道对宏观模式的贡献。

**🔧 技术方法**

主要技术包括：Agent‑Based Modeling（ABM）、边界信任（bounded‑confidence）学习、适应网络重连（homophily‑driven rewiring）、逻辑平衡方程（trust repair vs. harm）、参数扫面与敏感度分析、单元测试与基线还原验证。

**📊 数据集**

未使用外部真实数据集；所有实验基于内部随机生成的小世界网络和随机初始化的代理状态。

**📈 对比分析**

比较方法：对比四种基线（固定全局信任、同质化信任、完整反馈、随机重连）以及对外部冲击的响应；通过稳健性测试（适应 vs. 随机重连）评估结构性机制；性能体现在不同参数组合下能稳定产生四种显著的宏观模式（可信稳定、误导占主导、极化、基线），验证结果可复现且具可解释性。

**⚠️ 局限性**

局限性：①信息流仅二分为可信与误导，忽略多主题、多质量内容与平台算法；②信任单一维度，未区分机构、媒体、同辈等；③忽略情感、动机、策略行为等复杂人类因素；④未进行实证校准，参数对结果敏感；⑤仅提供定性阶段划分，缺乏精确预测能力。

---

## 92. Distributed Sparse Interventions in Language Models

**arXiv ID:** 2607.07128 | [PDF](https://arxiv.org/pdf/2607.07128v1)

**作者:** Maximilian S. Ernst `[一作]` (Max Planck School of Cognition), Oliver Eberle `[通讯]` (Technische Universität Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出一种分布式稀疏干预(DSI)方法，通过在数百层注意力头中选择极少数神经元来激活或抑制语言模型中的任务行为。

**💡 创新点**

创新点在于摒弃线性、独立性与均匀性假设，结合迭代细化与梯度鲁棒化，揭示跨层神经元的非线性交互，使得仅 0.01%–0.04% 神经元即可实现任务激活，并提供基于神经元集合的任务分解视角。

**🔧 技术方法**

主要技术包括基于鲁棒化 LRP 梯度的首阶 Taylor 近似、ZeroFPR 迭代优化、激活差分初始化，以及在 Qwen3、Gemma3 与 Llama3.2 等大模型上进行前向传播与微调。

**📊 数据集**

使用了 Qwen3 (8B)、Gemma3 (4B) 与 Llama3.2 (3B) 三大指令调优 LLM，配合 12 个抽取式与抽象式任务（如英语–法语翻译、时态转换等）构建的 0‑shot 与 10‑shot 训练/测试数据集。

**📈 对比分析**

通过将 DSI 与无干预基线、无迭代优化、无梯度鲁棒化三种变体在 0‑shot 单词预测准确率上进行对比，结果显示 DSI 在仅 8–64 个神经元时即可提升 50–100% 准确率，甚至达到或超过 10‑shot ICL 的表现。

**⚠️ 局限性**

限制在于评估仅针对清洗后的少样本提示与单词预测，未检验自回归生成中的干预效果；DSI 在不同模型与任务上表现不一，对子任务分解的可靠性仍有限。

---

## 93. Digital Fragmentation and Generative AI Use Across 103 Million Application Events

**arXiv ID:** 2607.06681 | [PDF](https://arxiv.org/pdf/2607.06681v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 94. A General Reduction from Near-Additive Emulators to Near-Exact Hopsets

**arXiv ID:** 2607.07190 | [PDF](https://arxiv.org/pdf/2607.07190v1)

**作者:** Julian Aeri `[一作]` (University of Salzburg), Mara Grilnberger `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种通用的从近加性模拟器（near‑additive emulator）构造近精确跳数集合（near‑exact hopset）的归约方法，给出了对应的规模、伸缩因子和跳数上界。

**💡 创新点**

创新点在于完成了此前仅完成从跳数集合到模拟器的逆归约，证明了在无向加权图上任意近加性模拟器都可直接转换为具有相似规模和伸缩的跳数集合，回答了 Elkin & Neiman 的开放问题。

**🔧 技术方法**

核心技术包括：按距离尺度划分图、对不同尺度进行边细分、在每个尺度构造无权图的模拟器、对模拟器边重新加权、合并所有尺度的结果形成跳数集合，并通过投影将多余的细分顶点消除到原始顶点集合；整个过程保留了距离与跳数的近似性。

**📊 数据集**

本文为理论工作，没有使用实际数据集；所有结论均在通用图模型下证明。

**📈 对比分析**

相较于之前的结果，本文提供的归约使得在已知近加性模拟器构造时即可获得尺寸为 S_𝒜(n+mβ/ε,ε/294,β)/ε·log(nW) 的跳数集合，伸缩因子为 (1+ε)，跳数上界为 O(t²·ln(nW))，其中 t = max(1/ε,β/ε)。

**⚠️ 局限性**

限制在于归约得到的跳数集合规模仍然依赖于原图的边数 m；作者指出消除对 m 的依赖仍是一个未解决的开放问题。

---

## 95. Measuring Intelligence Beyond Human Scale

**arXiv ID:** 2607.07040 | [PDF](https://arxiv.org/pdf/2607.07040v1)

**作者:** Jerry Han `[一作]` (Princeton Superalignment), Elad Hazan `[通讯]` (Princeton Superalignment)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于对抗心理测量（adversarial psychometrics）的新型评估范式，让模型自我生成二元挑战，并通过这些挑战来区分其他模型的能力。

**💡 创新点**

创新点在于：①用相对测量取代传统绝对分数，关注模型生成的分离度而非单个胜负；②通过方差奖励激励模型发现群体差异，减少私有信息攻击与陷阱门；③支持 judge‑free 评估，且可通过加权变体和公开历史实现可扩展且公平的排名。

**🔧 技术方法**

主要技术包括：二元问题生成与提交、概率报告与方差奖励、Brier 损失的正确性评分、加权方差与自适应权重更新、公开历史与链‑of‑thought 记录、程序执行或承诺的判定机制、Consensus 解决方案。

**📊 数据集**

使用的数据集为 11 个当代大型语言模型（gpt‑5.5、gpt‑5.4、qwen3.7‑max、opus‑4.8、sonnet‑4.6、haiku‑4.5、gpt‑5.4‑nano、kimi‑k2.7、deepseek‑v4‑pro、gpt‑4o‑mini 等），每个模型既充当提问者也充当求解者；挑战形式包括可执行 Python 程序、自然语言是非问题等。没有使用公开的传统人类设计基准，而是让模型自行生成评估任务。

**📈 对比分析**

实验通过 10 场独立游戏、每场 20 轮，每轮每个模型都生成一个挑战并挑选 5 个求解者。最终以累计得分 G 作为排名依据。实验结果显示：前沿 OpenAI 模型（gpt‑5.5、gpt‑5.4）在程序‑承诺（PC）和问题‑承诺（QC）两种配置下均稳居榜首；其他模型按实力分布顺序出现；系统在不需要人工裁决的情况下即可得到相对稳定的排名；相对评价与传统评估方法（如 MMLU、HELM 等）相比，展示了更高的区分度和可扩展性。

**⚠️ 局限性**

局限性包括：①对不可验证任务的依赖可能导致承诺误判与作弊；②模型可能复制已知模板导致创新不足；③方差奖励易被概率放大攻击，需额外的概率裁剪或加权机制；④对少量参与模型时统计信度低；⑤需要多轮交互和公开历史才能收敛，实验成本较高；⑥对深度未被所有模型覆盖的极难问题，依旧无法提供有效分离。

---

## 96. Hardness of Frequency-Related Queries on Compressed Strings

**arXiv ID:** 2607.07366 | [PDF](https://arxiv.org/pdf/2607.07366v1)

**作者:** Rajat De `[一作]` (Stony Brook University), Dominik Kempa `[通讯]` (Stony Brook University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并证明了在语法压缩及LZ78压缩文本上对频率相关查询（如rank、符号出现、区间不同计数与区间众数频率）实现多项式对数时间的压缩索引是极其困难的；

**💡 创新点**

首次将频率相关查询的上界难度与经典难题——Boolean矩阵乘法与Orthogonal Vectors——建立紧密的条件下界关联，揭示了该类查询在压缩空间中潜在的不可压缩性；

**🔧 技术方法**

利用从Boolean矩阵乘法和Orthogonal Vectors的精细归约，构造可压缩但信息量巨大的字符串，证明任何能快速回答批量符号出现或rank查询的结构都会隐含更快的BMM或OV算法；

**📊 数据集**

该工作为纯理论论文，未使用具体实验数据集，全部基于抽象的数学构造与归约；

**📈 对比分析**

由于是条件下界研究，不涉及实验比较；结论表明在已知的最佳BMM/OV算法下，无法在(|G|log^O(1)n)空间内以(log^O(1)n)时间完成这些查询；

**⚠️ 局限性**

结果依赖于BMM与OV的未被打破的假设，若未来出现更快的BMM/OV算法，则对应的压缩索引可能被改进；此外，只讨论了语法压缩和LZ78两种压缩框架，对其他压缩方法的适用性未给出结论。

---

## 97. An Introduction and Tutorial of the Beagle Framework

**arXiv ID:** 2607.06731 | [PDF](https://arxiv.org/pdf/2607.06731v1)

**作者:** Ilya Basin `[一作]` (Noblis), Wolfgang Banzhaf `[通讯]` (Michigan State University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个GPU加速的符号回归框架Beagle，支持百万级种群并行进化。

**💡 创新点**

通过自定义线性遗传编程语言、批量评估、Monte Carlo排名以及GPU与CPU任务分配，实现了大规模种群高效搜索。

**🔧 技术方法**

使用C#/.NET10、ILGPU库实现CUDA核，采用线性遗传程序（RPN）、GPU批量评估、GPU-CPU并行、垃圾回收优化等技术。

**📊 数据集**

主要使用Feynman100基准公式以及自定义合成数据集进行实验。

**📈 对比分析**

与传统CPU基准框架在Feynman100上对比，Beagle在相同硬件下速度提升数十倍，能够在几分钟内搜索到接近最优解。

**⚠️ 局限性**

受限于GPU内存仍需批量处理；目前不支持交叉算子、MacOS、非CUDA GPU，且需要手动实现特定适配器和自定义适应度函数。

---

## 98. Security and Privacy in Agentic AI: Grand Challenges and Future Directions

**arXiv ID:** 2607.06608 | [PDF](https://arxiv.org/pdf/2607.06608v1)

**作者:** Adam Jenkins `[一作]` (King's College London), Xiao Zhan `[通讯]` (Universitat Politècnica de València)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过三天的专业工作坊，结合30名来自学术、工业和政府的专家，使用地平线扫描（horizon scanning）方法与亲和映射（affinity mapping）技术，对agentic AI（具备自主决策和行动的人工智能）在安全与隐私方面的主要挑战与未来研究方向进行了系统梳理与主题归纳。

**💡 创新点**

创新点在于：①首次将地平线扫描与亲和映射相结合，形成了一套适用于快速评估AI自治系统安全隐患的实践框架；②将法律合规、责任归属、用户同意、个性化、透明度、弹性等多维度议题统一到四大主题，提供了跨学科的、系统化的研究路线图；③借助大型语言模型（Qwen3.5‑27B）协助整理和聚类专家讨论笔记，展示了AI辅助研究方法的可行性。

**🔧 技术方法**

采用的技术与方法主要是：地平线扫描（horizon scanning）工作坊、亲和映射（affinity mapping）技术、分组讨论与全体合成、以及使用大型语言模型（Qwen3.5‑27B）进行文本摘要与主题聚类。

**📊 数据集**

本文未使用传统意义上的实验数据集；研究数据来源为专家的讨论记录、会议纪要和小组生成的想法，后通过大型语言模型进行语义聚类与总结。

**📈 对比分析**

由于本研究为定性调研，未进行实验对比或性能评估。结论基于专家共识与工作坊产出的主题列表，未提供数值指标或客观度量。

**⚠️ 局限性**

局限性包括：①参与专家人数有限（30人），可能存在地域、行业与观点偏差；②工作坊时间短暂（仅3天），难以深入探讨所有子议题；③依赖专家主观判断与讨论，缺乏实证验证；④使用大型语言模型进行文本聚类虽高效，却可能引入模型偏见或错误；⑤未针对具体AI系统进行案例实验，导致建议的可操作性需进一步验证。

---

## 99. SoccerNet 2026 Challenges Results

**arXiv ID:** 2607.07320 | [PDF](https://arxiv.org/pdf/2607.07320v1)

**作者:** Anthony Cioppa `[一作]` (University of Liège), Julian Ziegler `[通讯]` (Leipzig University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文对 SoccerNet 2026 年的五个视觉任务（球行动预测、球行动定位、视角合成、静态摄像球员定位和视觉问答）进行了任务定义、数据集构建、评估协议和结果公布，提供了基准与排行榜，系统总结了获胜方法。

**💡 创新点**

创新点在于结合高分辨率输入、模型集成、专用战术特征、摄像机几何约束与多模态检索，提升了对不确定性、高噪声环境和细粒度定位的处理能力。

**🔧 技术方法**

主要技术包括 Transformer 编码解码框架（FAANTRA、VLM‑TCF）、多视角 3D Gaussian Splatting、YOLO/DETR 目标检测与姿态回归、世界坐标 Huber 损失、专用棋局与裁判知识检索以及大模型（Gemini、Claude、Qwen3‑VL）推理。

**📊 数据集**

使用了 SoccerNet‑v3、SN‑BAA、FOOTPASS、Blender 合成视角数据、Spiideo Synloc 数据以及 SoccerBench/VQA 数据集，涵盖了从动作注释到多模态问答的全链路。

**📈 对比分析**

与基准对比，获胜方案在各任务上均有显著提升：BAA mAP_avg 提升至 24.08（基准 16.76），PCBAS 宏 F1 提升至 58.94（基准 46.41），视角合成 PSNR 提升至 29.89（基准 26.74），Synloc mAP-LocSim 提升至 97.67（基准 77.30），VQA 准确率达到 98.0%（随机 25%）。

**⚠️ 局限性**

局限包括对未来动作的不确定性处理仍不充分，罕见动作如 tackle 仍表现欠佳，远距离球员检测对高分辨率要求苛刻，视角合成对高频细节的 PSNR 与 LPIPS 评价差异，且多模态检索依赖外部知识库导致可复现性受限。

---

## 100. Physics-guided spatiotemporal neural models for fuel density prediction

**arXiv ID:** 2607.06999 | [PDF](https://arxiv.org/pdf/2607.06999v1)

**作者:** Tolga Caglar `[一作]` (University of California San Diego), Ilkay Altintas `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于物理引导的时空神经网络框架，用以预测并模拟预燃烧条件下的燃料密度变化；

**💡 创新点**

创新点在于构建可微的多项式损失函数WiFireLoss（包含燃料传输、燃烧/未燃烧状态加权以及火速率约束），并将其作为软约束注入ConvLSTM、AFNONet和ViViT三种不同架构，提升预测的物理一致性和稳定性；

**🔧 技术方法**

采用ConvLSTM、Adaptive Fourier Neural Operator（AFNO）和Video Vision Transformer（ViViT）三种深度时空模型；利用PyTorch混合精度训练、AdamW优化器与余弦退火学习率调度；构造了包含MSE、燃料传输、ROS等多项损失的组合；

**📊 数据集**

使用QUIC‑Fire仿真数据集：50秒、300×300格网的草地燃料密度序列，覆盖7种风速、11种风向、4种点火模式，共构成多条件训练样本；

**📈 对比分析**

通过在相同训练设置下对三种模型分别加入或不加入物理损失，比较基准MSE、燃料传输误差、ROS误差及标准差；实验表明，加入物理损失后各模型误差显著下降，方差也更小，说明预测更准确、稳定；

**⚠️ 局限性**

局限性包括：仅在平坦草地、固定燃料类型和短时序列（50s）上验证；未评估更长时间序列、多样化燃料/地形条件下的泛化能力；

---

## 101. Tree-of-Thoughts Reasoning for Text-to-Image In-Context Learning

**arXiv ID:** 2607.07117 | [PDF](https://arxiv.org/pdf/2607.07117v1)

**作者:** Stepanida Alekseeva `[一作]` (Korea University), Seong-Whan Lee `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Tree-of-Thoughts (ToT) 推理框架，用于文本到图像的情境学习（T2I-ICL），该框架在推理阶段通过多分支、四阶段的结构化推理生成更准确的文本提示，然后由固定的文本到图像扩散模型（Stable Diffusion）生成最终图像。

**💡 创新点**

创新点包括：①首次将多分支 Tree-of-Thoughts 结构引入 T2I-ICL，显著提升了推理的多样性与稳健性；②设计了四阶段（Scene、Attribute、Stability、Composition）分层推理流程，并通过手工定义的评分函数和 beam‑search 剪枝策略挑选最优路径；③实现了全推理过程无需微调、仅在推理时执行的可解释方法。

**🔧 技术方法**

技术细节包括：使用 SEED‑LLaMA 作为多模态大语言模型进行推理；手工定义的评分函数基于六个准则（查询锚定、实体保留、阶段忠实度、规则一致性、约束一致性、语言质量）加上跳跃与冗余惩罚；四阶段的 Tree-of-Thoughts 结构化推理与 beam‑search 剪枝；最后将生成的推理路径转换为简洁、无冗余的文本提示，再输入稳定扩散模型进行图像合成。

**📊 数据集**

数据集为 CoBSAT benchmark，包含 10 个 T2I-ICL 任务（如 Color、Background、Style、Action、Texture 等），共 300 个样本用于评估。

**📈 对比分析**

与 Baseline（直接提示）和 Chain‑of‑Thought（单线性推理）进行对比，评估指标包括自动化 CLIP 相似度、Constraint Satisfaction Rate (CSR) 以及 21 名参与者的主观偏好率。结果显示：ToT 的 CLIP 为 0.318±0.030，CSR 为 0.775±0.252；相比 Baseline（0.287/0.508）和 CoT（0.302/0.547）均有显著提升；在人类评测中，ToT 的首选率分别为 59.5%、68.1%、65.5%，均显著高于其他方法。

**⚠️ 局限性**

局限性包括：推理过程需要多分支生成和评估，导致推理时间和算力成本较高；使用固定的文本到图像生成器限制了对推理结果的可控性；评估仅在 CoBSAT 这一相对有限的基准上，缺乏更广泛多样化任务的验证。

---

## 102. Comparative Study of Domain-adapted VLMs for General Document Visual Question Answering

**arXiv ID:** 2607.07179 | [PDF](https://arxiv.org/pdf/2607.07179v1)

**作者:** Miguel Lopez-Duran `[一作]` (Universidad Autónoma de Madrid), Javier Ortega-Garcia `[通讯]` (Universidad Autónoma de Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对8种开源Vision‑Language模型（VLM）在工业文档、信息图和幻灯片三大文档域下的Document Visual Question Answering（DocVQA）任务进行系统评估与对比。

**💡 创新点**

首次从零射击、全监督微调、跨域评估和少样本学习四个维度，揭示不同VLM规模与视觉复杂度对DocVQA性能的影响，并指出视觉理解是主要瓶颈。

**🔧 技术方法**

采用Unsloth库进行LoRA微调（rank 32，alpha 4，dropout 0.2）并使用平均归一化编辑距离（ANLS）作为评价指标；在三套数据集上执行零射击、全监督和少样本实验。

**📊 数据集**

SP‑DocVQA（工业文档）、InfographicsVQA（信息图）和SlideVQA（幻灯片）共计约97 k问答对，涵盖结构化、非结构化和多页面文档。

**📈 对比分析**

结果显示：大规模模型（Qwen3‑VL 8B、Qwen3.5 9B）在零射击下对结构化文档达90%+ ANLS，但在信息图和幻灯片上仅60%和50%；微调后所有模型均提升，尤其是小模型（Qwen3.5 0.8B、2B）相对增幅最高；少样本（5/20/50）微调可快速逼近甚至超越全监督微调，说明跨域知识迁移有效。

**⚠️ 局限性**

当前模型在视觉复杂布局上的理解不足导致性能受限；跨域微调可能出现负迁移，未解决灾难性遗忘；实验仅覆盖单页DocVQA，未涉及多页或开放域情况。

---

## 103. Evaluating SageMath-Augmented LLM Agents for Computational and Experimental Mathematics

**arXiv ID:** 2607.06820 | [PDF](https://arxiv.org/pdf/2607.06820v1)

**作者:** Pavel Snopov `[一作]` (University of Texas Rio Grande Valley), German Magai `[通讯]` (Noeon Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了在 ReAct 代理框架下，结合 SageMath 和 Context7 文档检索的 LLM 代理在 133 题研究级数学问题上的解决效果。

**💡 创新点**

创新点包括：①将 CAS 访问与 LLM 代理系统化；②提出多步后处理与多阶段验证管道，提升问题集质量与可靠性；③公开完整实验代码与评测流程。

**🔧 技术方法**

采用了 ReAct 代理式多轮交互、SageMath 计算接口、Context7 文档检索、Symbolic 等价检查与 LLM-judge 双重验证、Token 统计与异常分析等技术。

**📊 数据集**

使用了经过筛选与归一化的 133 题 RealMath 研究级数学问题集，主要包含可通过符号/数值计算验证的答案。

**📈 对比分析**

通过对比工具无与工具有两种设置，评估 solve rate、token 使用、错误恢复等指标，平均提升 9.7pp，最高 75.2% solve rate，GPT‑5.5 在准确率与 token 效率上遥遥领先。

**⚠️ 局限性**

主要局限包括：①符号等价验证不完备导致 23–44% 的误判；②数据集仅覆盖可计算子集，忽略深层理论难题；③实验涉及多工具与流程，难以单独归因 CAS 贡献；④LLM 生成代码易出现错误，需要进一步完善错误恢复。

---

## 104. Unraveling Machine Behavior by Multi-Level Bias Analysis and Detection: Methodology and Application to Computer Vision

**arXiv ID:** 2607.07236 | [PDF](https://arxiv.org/pdf/2607.07236v1)

**作者:** Ignacio Serna `[一作]` (Max Planck Institute for Human Development), Julian Fierrez `[通讯]` (BiometricsAI Universidad Autónoma de Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并实现了基于神经网络内部结构的多层级偏差检测方法，提出SpaceBias（潜在空间）、ActivationBias（层激活）和WeightBias（卷积权重）三种检测器；

**💡 创新点**

创新点在于把偏差分析扩展到潜在空间、激活层和权重层面，并通过无训练输入的统计检验和二级网络直接识别权重层的偏差特征；

**🔧 技术方法**

使用了Kolmogorov–Smirnov检验、Mann–Whitney U检验、邻居概率分布、聚类概率计算、以及训练小型神经网络ψ对权重进行分类检测；

**📊 数据集**

实验数据包括DiveFace人脸性别数据集、Colored‑MNIST（色彩偏差的数字分类）以及大规模小模型集合；

**📈 对比分析**

通过在面部和MNIST任务中对比实验，空间级检测在弱偏差下召回率高，激活级检测在强偏差下表现优异，权重级检测在高偏差时几乎完美，且跨域迁移验证了方法的通用性；

**⚠️ 局限性**

局限性包括仅针对CNN，需对邻居数、显著性阈值等超参数进行调优，权重级检测需要大量模型训练且难以直接迁移至大型模型或Transformer架构。

---

## 105. Head, Gaze, or Finger? Comparing Object Selection Techniques in Augmented Reality for People with Low Vision

**arXiv ID:** 2607.06778 | [PDF](https://arxiv.org/pdf/2607.06778v1)

**作者:** Ruijia Chen `[一作]` (University of Wisconsin-Madison), Yuhang Zhao `[通讯]` (University of Wisconsin-Madison)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究低视力人群在AR中使用头部、注视和手指三种选择技术进行目标选择，并在静态与动态情境中评估其性能与体验。

**💡 创新点**

首次系统比较低视力者对三种选择技术的表现与偏好，揭示视力度数与视野缺失对选择效率与稳定性的交互影响，并发现中心视力损伤者更倾向手指选择。

**🔧 技术方法**

使用Meta Quest 3配备Pupil Labs Neon眼动仪实现眼动跟踪，结合头部跟踪与手部跟踪的AR指针；采用驻留确认并在真实环境中采集数据。

**📊 数据集**

数据来自20名低视力参与者和18名视力正常对照者，分别在桌面架子与行走路线的两种目标尺寸上完成16/4次试验。

**📈 对比分析**

通过对选择时间、指向时间、确认时间、重进次数、步行时间等指标进行ART ANOVA与混合效应模型比较，结果显示低视力者在静态大目标下注视最快，动态场景头部与注视相当，手指表现最差；低视力者对注视的稳定性低于头部。

**⚠️ 局限性**

局限在于实验在受控室内环境、目标为实物且未考虑视觉混乱、低对比度等实际场景；未使用经典Fitts法则；样本量有限，且未探索更复杂的双任务或多目标情境。

---

## 106. Programmable Synchronization Graphs for Adaptive and Fault-Tolerant Modular Miniature Robots

**arXiv ID:** 2607.07281 | [PDF](https://arxiv.org/pdf/2607.07281v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 107. Robust Federated Learning Under Real-World Client Churn

**arXiv ID:** 2607.06979 | [PDF](https://arxiv.org/pdf/2607.06979v1)

**作者:** Dhruv Garg `[一作]` (Georgia Tech), Ada Gavrilovska `[通讯]` (Georgia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种面向实时客户端抖动的联邦学习（FL）框架，重新设计 FL 的调度与聚合流程，主张把可用性和数据价值作为实时控制信号，以加速模型收敛并降低通信开销。

**💡 创新点**

创新点在于：①引入多层级“可用性分层”，通过轻量级的心跳/遥测动态划分客户端；②双路选择机制（训练选择 + 评估探测）实时更新客户端的统计价值；③在聚合时结合“信息量权重”和“模型年龄修正”，在保留延迟高质量更新的同时抑制过时信息。

**🔧 技术方法**

技术上实现了：基于 Flame 的 FL 控制平面扩展；轻量级客户端分层协议；双路选择插件；缓冲加权聚合层；使用 MQTT 进行消息传递；利用实时可用性与评估信息动态调度客户端。

**📊 数据集**

使用的主要数据集为 CIFAR‑10（图像分类）和 Google Speech Commands V2（语音识别），并结合真实的 MobiPerf 设备可用性轨迹进行评估。

**📈 对比分析**

与同步/异步 FL、OORT、REFL 等基线进行对比；在多种可用性和数据异构度设置下，实验显示在 CIFAR‑10 上最快 2.37× 的时间‑到‑准确率（TTA），在 Speech 上约 1.45×；通信总量比最优基线低 1.30×，同时在低可用性真实轨迹上也能保持较低的延迟和高的准确率。

**⚠️ 局限性**

局限性包括：①评估任务会产生额外的通信负载，尤其在大模型场景下显著；②依赖轻量级遥测的准确性，若设备隐私策略限制遥测则可能受限；③实验只覆盖两种任务类型，未在更复杂的多模态或联邦学习场景中验证；④对极端网络抖动或极低可用性（<10%）的鲁棒性尚未完全评估。

---

## 108. CaLiSym: Learning Symplectic Dynamics of Real-World Systems through Structured Canonical Lifts

**arXiv ID:** 2607.06824 | [PDF](https://arxiv.org/pdf/2607.06824v1)

**作者:** Aristotelis Papatheodorou `[一作]` (University of Oxford), Gerard J. Milburn `[通讯]` (University of Sussex)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种结构化的 Canonical Lifted Symplectic 框架 CaLiSym，利用在升维相空间上严格保持辛结构来学习受控、耗散和接触驱动的机器人动力学，

**💡 创新点**

创新点在于将辛结构迁移至升维相空间而非直接作用于物理状态，从而在非保守系统上实现严格辛映射；同时引入 GRB‑SympNet 与 GR‑SympNet 两种可扩展的辛网络；

**🔧 技术方法**

使用结构化升维（lift‑evolve‑project‑re‑embed）策略、GRB‑SympNet（局部 B‑spline 逼近）与 GR‑SympNet（全局线性嵌入）辛网络，并采用教师强制训练、标准化、数值检验等技术；

**📊 数据集**

在受控耗散双摆（仿真）、实测四旋翼（AscTec Pelican）和实测四足机器人（ANYmal D）三套数据集上进行评估；

**📈 对比分析**

与 MLP、Transformer、RWM‑TF、DHNN、D‑SymODEN 等基线对比，均采用 OOD 自回归 MSE 和参数量评估；在三组任务中，CaLiSym 在 OOD MSE 上均比最强基线低 12–70%，并以约 3.7k–316k 参数实现更小的模型规模；

**⚠️ 局限性**

局限在于仍假设系统能通过固定端口（力/扭矩）完整描述能量交换；对极端接触模式、非线性摩擦、模型不确定性等情况的鲁棒性和可扩展性尚待进一步验证。

---

## 109. LipSSD: Lipschitz-Constrained Single-Shot Detection for Adversarially Robust Object Detection

**arXiv ID:** 2607.06592 | [PDF](https://arxiv.org/pdf/2607.06592v1)

**作者:** Vincent Lébé `[一作]` (IRT Saint-Exupéry), Franck Mamalet `[通讯]` (IRT Saint-Exupéry)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 LipSSD，一种在单阶段目标检测器中通过正交卷积、GroupSort 激活和 L2 池化实现全 1‑Lipschitz 的架构，并探讨了准确性-鲁棒性权衡及其与对抗训练的互补性。

**💡 创新点**

创新点在于：①仅对分类头约束 Lipschitz，保持回归头自由；②引入温度缩放与 SLipSSD 版本，可通过单一超参数调控鲁棒性；③证明 LipSSD 可与对抗训练协同提升鲁棒性，且无推理时额外开销。

**🔧 技术方法**

使用正交卷积（RKO、AOC）、Spectral Normalization、GroupSort 激活、L2 池化、温度缩放交叉熵；评估时采用 PGD、DAG、TOG 等白盒攻击；并对比基准 SSD 与对抗训练 SSD。

**📊 数据集**

评估数据集包括 Pascal VOC 07+12 作为基准，LARD（跑道检测）和 KITTI（自动驾驶）作为安全关键案例；图像尺寸和学习率根据任务进行调整。

**📈 对比分析**

与标准 SSD 及对抗训练 SSD 在 mAP@50 上比较，LipSSD 在多种攻击下鲁棒性提升 15–30% 以上；SLipSSD 在保留更高清洁精度的同时亦显著提高鲁棒性；在 LARD/KITTI 上保持清洁性能并显著提升攻击下 mAP。

**⚠️ 局限性**

主要限制：训练成本约为标准 SSD 的两倍；在 Pascal VOC 上清洁精度仍低于标准 SSD；缺乏可迁移的 1‑Lipschitz 预训练主干；未实现正式的鲁棒性证明或推理时额外开销。

---

## 110. POPS: Recovering Unlearned Multi-Modality Knowledge in MLLMs with Prompt-Optimized Parameter Shaking

**arXiv ID:** 2607.06649 | [PDF](https://arxiv.org/pdf/2607.06649v1)

**作者:** Zhangheng LI `[一作]`, Zhangyang Wang `[通讯]` (University of Texas at Austin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多模态机器无学习的鲁棒性，提出 Prompt-Optimized Parameter Shaking (POPS) 攻击框架，通过提示后缀优化和自合成微调恢复已被忘记的敏感知识。

**💡 创新点**

结合 OOD 引导的提示后缀优化与跨模态参数摇摆微调，实现对多模态无学习模型的恢复攻击；提出闭环攻击循环，显著提升恢复率。

**🔧 技术方法**

提示后缀优化、低秩 LoRA 微调、跨模态对齐层攻击、语义生成合成数据、困惑度筛选。

**📊 数据集**

三大多模态无学习基准：MLLMU-Bench、CLEAR、UnLoK-VQA；使用对应的留存集作为 OOD 数据。

**📈 对比分析**

对比 Gradient Ascent、Gradient Diff、KL Minimization、MANU、MultiDelete 等无学习方法，POPS 在多模态基准上平均恢复率超 80%，比基线提升 2–3%，甚至逼近原始模型性能。

**⚠️ 局限性**

攻击仍需离线微调，成本高；对抗样本可能被安全过滤；对防御手段（如对抗训练、差分隐私）的效果尚未充分评估。

---

## 111. STST-JEPA: Shallow-Target Spatio-Temporal Joint Embedding Prediction Architecture For EEG Self-Supervised Learning

**arXiv ID:** 2607.06629 | [PDF](https://arxiv.org/pdf/2607.06629v1)

**作者:** Roy Segal `[一作]` (brain.space), Tomer Fekete `[通讯]` (brain.space)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本工作提出并训练了一种名为 STST‑JEPA 的 24 层 Transformer 基础模型，利用自监督的联合潜在预测与重构目标，在 47,703 条 5–81 岁 EEG 记录上预训练，并通过轻量级注意力探针完成脑龄预测、性别分类和精神病学评分等多任务推断。

**💡 创新点**

创新点在于（1）将掩码潜在预测与信号重构结合，既保持表层信号的可辨识性又避免对噪声的过度拟合；（2）采用坐标感知的 Pooled Multi‑Head Attention (PMA) 对不同蒙太奇进行统一 128 通道处理；（3）在单一预训练检查点上实现跨任务的 1‑排名 leaderboard 成绩，展示了真正的“基础模型”特性。

**🔧 技术方法**

技术手段包括：自监督学习中的掩码潜在预测（EMA‑tokenizer 目标）、smooth‑L1 重构损失、Transformer 结构、RoPE 位置编码、可学习的通道坐标门控、以及注意力 + 单查询交叉注意力的轻量级探针。

**📊 数据集**

使用的数据集为内部 brain.space EEG 语料（22,588 次会话）和公开的 Healthy Brain Network (HBN) 语料（25,115 次会话），覆盖 5–81 岁年龄跨度；评估时还利用 NeuralBench × brain.space EEG leaderboard 的公开训练/测试拆分。

**📈 对比分析**

与基线（predict‑mean）及传统基准（Engemann 等 7–8 年 MAE）比较，STST‑JEPA 在内部验证集上实现 MAE 3.06 年、RMSE 5.11 年、R² 0.85；在 NeuralBench leaderboard 上以 30 秒窗口取得 rank‑1 性别分类（BAcc 0.911）、年龄回归 Pearson r 0.749、精神病学评分 r 0.215，分别超过此前最佳模型。

**⚠️ 局限性**

局限性包括：预训练阶段未使用年龄标签；仅在 frozen‑probe 方案下评估，未探究微调效果；语料规模虽大但仍偏向儿童/青少年，可能影响成人性能；缺乏临床验证及对不同 montage 的鲁棒性深入分析；并未进行联合目标的 ablation 研究。

---

## 112. Improved Algorithms and Lower Bounds for Parametrized Metrical Service Systems

**arXiv ID:** 2607.07098 | [PDF](https://arxiv.org/pdf/2607.07098v1)

**作者:** Junhao Gan `[一作]` (University of Melbourne), Seeun William Umboh `[通讯]` (University of Melbourne)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究参数化的度量服务系统（MSS），给出了在加权星形图上确定性 O(m) 的竞争比、在两层 HST 上随机竞争比的下界与上界匹配，以及在 m=2 的一般度量空间中实现常数竞争比的算法；

**💡 创新点**

创新点在于：1）利用区间覆盖与原始-对偶方法，首次在加权星形图上实现了确定性 O(m) 竞争；2）构造更精细的随机下界，证明两层 HST 上随机竞争比为 Θ(m⌈m/2⌉)，并推广到更高层 HST；3）提出基于 Guess‑and‑Double 的常数竞争算法，解决 m=2 的一般度量空间；

**🔧 技术方法**

技术手段包括：区间覆盖的整数/线性规划建模及原始-对偶增量算法；随机下界采用 Yao 原理与递归构造的 HST 实例；常数竞争算法采用 Guess‑and‑Double 以及对 OPT 估计；

**📊 数据集**

论文未使用实验数据，全部以理论构造和证明为主；

**📈 对比分析**

与已有工作比较：此前对加权星形图仅有 O(2^m) 的确定性上界和 Ω(m) 的随机下界；本文实现了确定性与随机的匹配；此前两层 HST 的随机上界为 O(m 2^m)，本文压缩至 Θ(m⌈m/2⌉)；对于 m=2 的一般度量空间，之前无确定性常数竞争算法，本文给出 6 倍竞争；

**⚠️ 局限性**

局限性包括：1）对 m=3 及更高的 HST 或线性度量的竞争比仍未解；2）两层 HST 上的确定性竞争是否能达到同样的 Θ(m⌈m/2⌉) 仍未知；3）是否可将本框架推广至更一般的参数化 MTS 或加权星形图仍是开放问题。

---

## 113. Fixed Points, a Predictor-Impossibility Theorem, and Applications

**arXiv ID:** 2607.06956 | [PDF](https://arxiv.org/pdf/2607.06956v1)

**作者:** Tom Altman `[一作]` `[通讯]` (University of Colorado Denver), Tom Altman (University of Colorado Denver)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种由激活算子生成的激活层次结构，包括阶段机器、阶段域和阶段语言。主要结果是预测者不可能定理（PIT），表明没有有效的预测者家族可以统一确定层次结构中的所有阶段语言。

**💡 创新点**

创新点在于提出了预测者不可能定理（PIT），并通过激活层次结构和聚合语言MIS的构建，展示了复杂性理论的结果与递归理论的结果之间的联系。

**🔧 技术方法**

使用了递归理论中的固定点方法，结合了S-m-n定理和克莱尼递归定理来证明主要结果。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了聚合对象的有效性和阶段语言的构造。

**📈 对比分析**

通过聚合语言MIS与阶段语言之间的切片定理进行比较，证明了MIS不属于多项式时间可判定类（P），并且MIS属于NP类。

**⚠️ 局限性**

限制在于聚合语言MIS的复杂性分析依赖于聚合增长条件，且该条件的放宽可能影响结果的稳定性。

---

## 114. MMAgent-R$^2$: Learning to Rerank and Reject for Agentic mRAG

**arXiv ID:** 2607.07383 | [PDF](https://arxiv.org/pdf/2607.07383v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 115. Rag Classification of Tagore Songs using Symbolic Music Notation and Novel Weighted Distance Measures

**arXiv ID:** 2607.07241 | [PDF](https://arxiv.org/pdf/2607.07241v1)

**作者:** Chandan Misra `[一作]` (XIM University), Swarup Chattopadhyay `[通讯]` (XIM University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了基于符号乐谱的鲁加分类数据集，并提出了鲁加感知加权欧氏距离用于Tagore歌曲的鲁加识别。

**💡 创新点**

通过将三八度音符合并为12音并为每个鲁加分配90:10加权，提升了对鲁加特征的捕捉；同时提出了针对符号数据的鲁加标注数据集。

**🔧 技术方法**

k-近邻分类器、欧氏距离、余弦相似度、加权欧氏距离以及音符频率特征提取。

**📊 数据集**

从SwaraBitan音符集随机抽取1000首Tagore歌曲，生成36维频率向量后聚合为12维，含239个鲁加，545首低频鲁加样本。

**📈 对比分析**

与传统余弦相似度和普通欧氏距离在kNN分类实验中对比，使用90:10/80:20权重的加权欧氏距离可将准确率提升约5–10%（最高达85%）。

**⚠️ 局限性**

样本量不平衡，低频鲁加样本不足；权重手工设定，未自动学习；仅考虑音符频率，未包含旋律序列和节奏等特征。

---

## 116. Dynamic Object Detection and Tracking in Construction: A Fisheye Camera and LiDAR Sensor Fusion Model

**arXiv ID:** 2607.06896 | [PDF](https://arxiv.org/pdf/2607.06896v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 117. GemNav: Discrete-Token Visual Robot Navigation using a Multimodal Large Language Model

**arXiv ID:** 2607.06882 | [PDF](https://arxiv.org/pdf/2607.06882v1)

**作者:** Peter Bohm `[一作]` (CSIRO), Peyman Moghadam `[通讯]` (CSIRO)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于冻结多模态大语言模型（MLLM）的视觉导航策略，通过在语言塔上应用LoRA，去除了额外视觉编码器和连续回归头，实现短至中程 waypoint 导航。

**💡 创新点**

创新点在于将导航动作直接视作离散 token 生成，利用预训练 MLLM 的视觉表征，仅用 8.7 小时数据即可实现跨环境零样本迁移；并引入软解码辅助损失恢复度量结构。

**🔧 技术方法**

使用技术包括：LoRA 在语言塔线性层、离散 token 化的 waypoint 与 stop/unreachable token、soft‑decoded 回归辅助损失、Gemma‑4 预训练模型。

**📊 数据集**

使用数据集：SCAND（8.7 小时 Spot/Jackal 轨迹）作为训练集；TokenWalker 数据集作为 OOD 验证集；在四个真实环境中进行部署测试。

**📈 对比分析**

比较方法与性能：与 OmniVLA 及人工操作者对比，在 pose‑goal 任务下在四个不同环境均达成 100% 成功率；相较 OmniVLA 的 0% 成功率，平均终止误差仅 0.25–0.42 m；离散解码相较于传统 bin‑snapped 解码提升约 5 cm。

**⚠️ 局限性**

局限性：仅在单一平台与单一 8.7 小时数据上训练，跨平台迁移未验证；纯图像目标停止不可靠；无法输出旋转动作或后方目标。

---

## 118. At-Grok Is Not Converged:A Measurement-Validity Audit for Grokking Representation Metrics

**arXiv ID:** 2607.06639 | [PDF](https://arxiv.org/pdf/2607.06639v1)

**作者:** Truong Xuan Khanh `[一作]` `[通讯]` (H&K Research Studio / Clevix LLC), Truong Xuan Khanh (H&K Research Studio / Clevix LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Grokking 过程中表示结构的测量可靠性进行审计，揭示在 Grokking 转折点读取的有效秩是暂态峰值，且表示压缩往往在准确率提升之后显著滞后，同时证明归一化是决定滞后大小的关键因素。

**💡 创新点**

创新点在于提出并实现了可测试的“测量可靠性审计”框架，能够区分 Grokking 的开始与表示压缩的完成时刻、过滤边界/失效样本、校验压缩下限是否稳定，并用对抗测试用例验证审计自身的稳健性；此外首次系统性展示了归一化对压缩滞后影响的量化机制，并在 Transformer 上验证了暂态峰值的普适性。

**🔧 技术方法**

主要技术包括：使用平方归一化有效秩（spectral‑entropy effective rank）作为表示复杂度指标；定义 T_grok（测试准确率≥0.9 的第一步）与 T_compress（指标稳定在下限以内的第一步）两种时钟；对压缩下限进行“平坦化检查”以避免下限自身为暂态；通过一变量（LayerNorm）消融实验和多架构（MLP、Transformer）自由权重衰减与 clamp 方案对比；并实现了九种对抗测试用例以检验审计逻辑。

**📊 数据集**

使用的数据集为模数 59 的算术任务（modular addition、modular multiplication）和偶然性任务（parity），在多种权重预算（ρ）和权重衰减参数下训练全批量 MLP 与无归一化/加 LayerNorm 的单层 Transformer，实验覆盖 6–9 种预算，单细胞多种种子。

**📈 对比分析**

比较方法：对不同预算下 T_grok 与 T_compress 的相关性、对压缩下限的折叠率（frac‑pre）进行统计，评估归一化对 lag 的影响；结果显示 T_grok 与预算高度相关（Spearman>0.85），而 T_compress 与预算无明显关联；压缩滞后约为 10⁴ 步（lag/T_grok≈1.0），而有效秩在 Grokking 时被高估 3–5 倍；在 Transformer 上暂态峰值被压缩到 1.3–1.5 倍，且 lag 与 MLP 类似。

**⚠️ 局限性**

局限性包括：归一化导致滞后机制的统计力有限（仅少数种子、6×10⁴ 步）；下限的平坦化检查仍可能被第二次收缩误导；depth law 的负结果仅在 MLP 与 clamp 协议下验证，无法推广到其他架构；审计只能检测暂态与下限问题，无法保证表示最终收敛；实验规模受限于计算资源，未覆盖更大模型或不同任务。

---

## 119. Structural Adversarial Attacks on Relational Deep Learning under Integrity Constraints

**arXiv ID:** 2607.07089 | [PDF](https://arxiv.org/pdf/2607.07089v1)

**作者:** Alan Gany `[一作]` (University of Grenoble Alpes), Silviu Maniu `[通讯]` (University of Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在保持数据库完整性约束下，对关系深度学习（RDL）模型进行白盒结构攻击，探索可行的攻击策略与评估攻击效果

**💡 创新点**

提出了在受限的FK重连搜索空间中利用梯度引导的攻击启发式，并对梯度归一化方式进行比较，首次在RelBench上系统评估此类攻击

**🔧 技术方法**

使用图神经网络（GraphSAGE）模型、可微分边缘掩码、梯度归一化（Z-score、Robust Z-score、Min-Max）以及候选集精简与精确评估等技术

**📊 数据集**

在RelBench数据集的F1赛道（含7个关系类型、约7万行）上进行实验

**📈 对比分析**

与两种随机基线及五种梯度基线对比，结果显示梯度攻击在回归任务中显著提高MAE（最高约7%），而在分类任务中提升有限；梯度+精确重排序方法与单纯梯度方法效果相近，前者计算成本更高

**⚠️ 局限性**

局限性包括：仅测试单一RelBench赛道；仅攻击结构而非特征空间；对目标标签的影响有限，未探讨针对性攻击与黑盒攻击；缺乏对更大、更复杂数据库的评估

---

## 120. Large Language Models (LLMs) and Generative AI in Cybersecurity and Privacy: A Survey of Dual-Use Risks, AI-Generated Malware, Explainability, and Defensive Strategies

**arXiv ID:** 2607.06963 | [PDF](https://arxiv.org/pdf/2607.06963v1)

**作者:** Kiarash Ahi `[一作]` (Virelya Intelligence Research Labs), Saeed Valizadeh `[通讯]` (Google)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了大型语言模型（LLM）在网络安全中的双重作用，分析了攻击（如AI生成恶意软件、钓鱼）与防御（静态扫描、零日检测、DevSecOps、联邦学习）场景，并提出治理与安全实践框架；

**💡 创新点**

首次系统性地将多行业案例、最新统计和技术评估整合为一套可操作的治理路线图，并强调可解释性与联邦学习在安全中的关键价值；

**🔧 技术方法**

利用多种LLM（GPT‑4、PaLM、Gemini、Copilot 等）、可解释技术（SHAP、LIME）、联邦学习架构、红队与对抗测试方法；

**📊 数据集**

依赖公开漏洞数据库、恶意软件检测统计、行业报告、开源代码审计数据以及多平台安全日志（Google Play Protect、Microsoft Defender、AWS 等）；

**📈 对比分析**

通过文献综述与案例对比评估，表明LLM在零日检测、恶意软件识别等任务中可显著提升准确率、降低误报；然而缺乏统一、公开的基准和对比实验；

**⚠️ 局限性**

存在缺乏统一评测基准、模型可解释性不足、潜在偏见与公平性挑战、联邦学习实现复杂度高，以及治理框架在跨国、跨行业中的落地难度。

---

## 121. DiffCVE: Diffusion-based Compressed Video Enhancement

**arXiv ID:** 2607.07195 | [PDF](https://arxiv.org/pdf/2607.07195v1)

**作者:** Wenqiang Xiao `[一作]` (Wuhan University), Zhenzhong Chen `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于扩散模型的压缩视频感知质量提升方法DiffCVE，利用编码先验和语义提示实现多尺度结构与运动信息的融合；

**💡 创新点**

创新点包括：①编码先验增强的双重条件分支（CPDC）将残差与运动向量注入扩散去噪；②压缩失真语义提示（CDSP）通过QP条件文本提示与LoRA微调统一适配不同压缩强度；③编码先验引导加权融合（CPWF）在VAE解码器侧实现按QP加权的多模态特征融合；

**🔧 技术方法**

核心技术为Stable Diffusion v2.1 U‑Net＋时序块、VAE编码/解码、CLIP文本编码、LoRA轻量微调、SFT、跨注意力、时序卷积与RRDB；

**📊 数据集**

训练使用Vimeo‑90k数据集，评估采用18个JCT‑VC标准序列（类A–E），并在多分辨率场景上验证；

**📈 对比分析**

与STDR、STFF、MW‑GAN+、HFGAT及SeedVR2‑3B等基线比较，DiffCVE在重度压缩（QP 42/37）下的LPIPS、DISTS、CLIPIQA、DOVER、NIQE显著提升，同时保持优良的时序一致性；单步版DiffCVE-1每帧约0.33 s，速度快于大规模扩散模型但仍慢于轻量级非扩散方法；

**⚠️ 局限性**

局限性在于推理时间和算力需求仍高，且当前仅覆盖三种QP水平，缺乏对更细粒度码率与多码流的适配；

---

## 122. ECO/CPO-DAG: A Contradiction-Based Accountability Layer for Adversarial Supply Chains

**arXiv ID:** 2607.06804 | [PDF](https://arxiv.org/pdf/2607.06804v1)

**作者:** Sebastian Cochinescu `[一作]` `[通讯]` (University of Bucharest), Sebastian Cochinescu (University of Bucharest)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究提出并实现了 ECO/CPO-DAG，一种面向供应链的基于矛盾检测的责任层，通过在有向无环图中发布签名事件声明并生成自验证矛盾证明，以实现对可证明矛盾的经济制裁。

**💡 创新点**

创新点在于将矛盾检测转化为可验证的矛盾证明对象，结合选择性披露与零知识技术实现隐私保护，并通过确定性责任划分实现链上罚没，从而在无需共识的情况下提供领域特定的可审计和经济惩罚机制。

**🔧 技术方法**

使用了签名事件声明对象（ECO）、有向无环图时序、混合逻辑时钟、Pedersen 承诺、CL 选择性披露凭证、Groth16 零知识证明，以及经济制裁的质押和罚没机制。

**📊 数据集**

实验采用了合成的 EPCIS 2.0 事件追踪数据集，并在单机参考实现中进行性能和安全性验证。

**📈 对比分析**

通过对检测覆盖率、误报率、CPO 生成与验证时延以及存储开销等指标进行定量分析，实验结果显示检测覆盖率与理论模型吻合、误报率为零、CPO 生成时间约 1.5 µs、验证 1.1 ms、每条事件约 3 ms 的检测开销，以及每节点每年约 1 GB 的存储需求。

**⚠️ 局限性**

局限性包括只能检测可证明的矛盾，无法识别一致性欺骗；跨方矛盾导致责任不确定需离线裁决；依赖观察者参与；存储随历史增长；质押门槛对小型供应商不友好；以及链上罚没的法律执行需要外部治理。

---

## 123. From Agentic to Autogenic Network Management for AI-Native 6G and Beyond: A Standards Perspective

**arXiv ID:** 2607.06786 | [PDF](https://arxiv.org/pdf/2607.06786v1)

**作者:** Petar Djukic `[一作]` (Bell Labs Research), Burak Kantarci `[通讯]` (University of Ottawa)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向AI‑Native 6G网络的自生成网络管理架构（Autogenic Network Management），通过大语言模型（LAM）驱动的代理实现自程序化、自反思、自定向和自架构能力，支持从人工监督到全自主的分阶段部署。

**💡 创新点**

创新点在于将自程序化、自反思、自定向、自架构四大能力集成到网络管理层，形成可在运行时生成并演化自动化软件的全新“自生成”架构，并给出与TM Forum、3GPP、ETSI等标准体系兼容的参考实现。

**🔧 技术方法**

核心技术包括：大语言模型（如LLM）用于代码合成与策略生成；数字孪生与数字孪生工厂用于安全验证与模型更新；面向代理的管理框架与层级化的监测、分析、规划、控制、管理与对等子系统；以及基于规范化接口的受限操作模型。

**📊 数据集**

本工作未使用传统机器学习数据集，而是基于TM Forum提出的11大业务场景（如故障管理、能耗优化等）进行案例演示与验证；若需要，可利用运营商运营日志或仿真数据进行后续评估。

**📈 对比分析**

比较方法主要是与现有的“Agentic AI”管理方案对照，展示在同一业务场景下从人工监督到递归代理的能力提升；由于缺乏公开实验平台，未给出量化性能指标，重点强调架构可行性和安全验证路径。

**⚠️ 局限性**

局限性包括：缺乏正式的安全与验证框架，LAM生成的代码可能不具备正式证明；对标准化接口的定义尚不成熟，跨厂商协作难度高；数字孪生工厂与安全验证所需资源大，部署成本不明；以及自定向与自架构能力在真实网络中的可行性仍需实验验证。

---

## 124. Physics-Audited Agentic Discovery in Scientific Machine Learning

**arXiv ID:** 2607.07379 | [PDF](https://arxiv.org/pdf/2607.07379v1)

**作者:** Diab W. Abueidda `[一作]` (New York University Abu Dhabi), Mostafa E. Mobasher `[通讯]` (New York University Abu Dhabi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了验证优先的代理式科学机器学习工作流，确保发现的神经算子在满足物理约束的前提下获得最佳预测精度。

**💡 创新点**

创新点在于将机器可检验的物理检查嵌入到代理搜索的固定评估器之前，并通过Sampled hard-contract gate和Adversary搜索实现对每个候选模型的输出层物理证据记录。

**🔧 技术方法**

使用的技术包括DeepONet、Fourier Neural Operator、因果卷积分支、固定评估器、Physics Auditor、Adversary搜索、Post-discovery attribution和Method-card保存等。

**📊 数据集**

数据集为两组有限元模拟数据：4000训练/1000验证的二维线性弹性参数化问题和500验证历史的薄条瞬态弹性问题。

**📈 对比分析**

对比方法为单次种子下的错误驱动搜索与验证驱动搜索，验证驱动搜索在静态问题中获得更低误差且不牺牲物理约束，在瞬态问题中通过因果性检查将误差相近但物理失效的模型排除，性能表现为误差分别降低约3.5%和显著通过因果性门限。

**⚠️ 局限性**

局限性包括单次随机种子、有限搜索规模、验证与错误比较是捆绑在一起、需要人工设计物理合同、以及在非线性或三维问题上的推广性尚未验证。

---

## 125. G-PROBE: Cross-FOV Place Recognition and Certainty-Coupled Localization for 3D Point Clouds

**arXiv ID:** 2607.06782 | [PDF](https://arxiv.org/pdf/2607.06782v1)

**作者:** Jinseop Lee `[一作]` `[通讯]` (SK Intellix), Jinseop Lee (SK Intellix)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出 G-PROBE，一种学习无关的跨视场（Cross‑FOV）全局定位框架，可在任意 LiDAR 配置下直接从单帧点云估计完整 6‑DoF 位置。

**💡 创新点**

创新点包括：① 虚拟传感器分解与交叉 FOV 分支检索，消除对全景视场的假设；② γ‑SGRT 软最大间隙比测试，提供无调参的旋转不确定性抑制；③ 置信度耦合的 CG‑GICP，利用检索时产生的置信度图只在可信点上进行精细配准，提升鲁棒性。

**🔧 技术方法**

核心技术包括：基于 PROBE 的概率极坐标 BEV 描述符、交叉 FOV 的互信息和占用率统计、FFT 旋转对齐、Bernoulli‑KL 对比度评分、软最大间隙比（γ‑SGRT）以及两阶段 Generalized‑ICP（CG‑GICP）。

**📊 数据集**

使用了 KITTI、NCLT、HeLiPR、SNAIL 和 GrandTour 五个多传感器/多模式数据集，涵盖机械旋转、固态、FMCW LiDAR 并测试多日与跨传感器情况。

**📈 对比分析**

与十个基准（M2DP、SC、SC++、SOLiD、RING++、PROBE、HeLiOS、BEVPlace++、UniLGL）对比，G‑PROBE 在学习无关的跨 FOV 与跨传感器任务中获得最高平均 F1（0.835），单日 AUC 亦处于前列；在极端 360°↔60° 视场不对称下，Recall@1 仍保持约 54%，是现有学习无关方法提升 18 倍。

**⚠️ 局限性**

局限性包括：① 仅基于离散 heading 量化，可能在复杂交叉路口产生误判；② CG‑GICP 仅在源点云过滤置信度，对极度稀疏或高度动态环境的鲁棒性仍待验证；③ 置信度映射为硬阈值，未对置信度进行软加权，可能导致误检或漏检；④ 在跨传感器的极窄对极窄场景下召回率仍有限，需进一步改进跨模态匹配。

---

## 126. Learning social norms enhances compatibility in dynamic human-AI coordination

**arXiv ID:** 2607.07021 | [PDF](https://arxiv.org/pdf/2607.07021v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 127. Fingerprint, Not Blueprint: How Positional Schemes Set the Default Spectral Algebra of Attention

**arXiv ID:** 2607.06621 | [PDF](https://arxiv.org/pdf/2607.06621v1)

**作者:** Li Hengyu `[一作]` `[通讯]` (University of Tokyo), Li Hengyu (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了 Transformer 中 QK 矩阵的复特征谱，探讨其在不同位置编码方式下与注意力头功能的关系，并通过静态统计、训练动态观察和因果消融实验验证了谱特征对头功能的可解释性与必要性。

**💡 创新点**

创新点在于提出匹配随机取向（Ginibre）零点基准、引入复特征分解与相位信息来捕捉旋转性，证明仅在使用旋转位置编码时谱信息才具有功能意义，并通过受限训练展示谱通道的默认性与可绕过性。

**🔧 技术方法**

主要技术包括非厄密矩阵理论、Schur 与复特征分解、随机矩阵对比、梯度消融与相位抑制、以及多模型、多位置编码的跨模型统计和约束训练网格实验。

**📊 数据集**

实验使用了七个公开预训练模型（GPT‑2、OPT‑1.3B、GPT‑Neo‑1.3B、BLOOM‑1b1、Pythia‑410m/1.4B、Llama‑3‑8B）覆盖三种位置编码（learned‑absolute、ALiBi、RoPE），并在自定义的两层注意力网络上进行合成任务训练。

**📈 对比分析**

比较方法主要是统计每头的方向性、对称性、相位贡献与已知头功能标签的相关性，结合损失消融、相位抑制对因果影响的度量；实验表明在旋转编码下谱方向性显著区分头功能，但在绝对位置编码下无显著优势；受限训练表明即使禁止相位或对称性，模型仍能完成任务，只是训练时间显著延长。

**⚠️ 局限性**

主要局限包括：关联性分析不等价于因果性，仅检验单头而忽略多头组合效应；仅评估了七个模型，未覆盖更大规模或不同族群 LLM；对 QK 偏置与头间交互未深入；以及基于随机矩阵的零点假设在极端参数或稀疏结构下可能失效。

---

## 128. Communicative Efficiency of Single vs. Multi-Axis Robot Neck Motion

**arXiv ID:** 2607.07390 | [PDF](https://arxiv.org/pdf/2607.07390v1)

**作者:** Chapa Sirithunge `[一作]` (University of Bristol), Josie Hughes `[通讯]` (EPFL)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究机器人颈部运动作为非语言沟通通道，量化其信息传递（Shannon熵）与能耗，并通过人类感知实验验证其可解释性。

**💡 创新点**

提出信息理论框架与Motor Information Space，发现三自由度产生信息瓶颈，并引入ENIE度量来评估能耗与信息效率，给出颈部设计的最优自由度与运动参数。

**🔧 技术方法**

使用UR3机器人实现多轴颈部运动，记录视频并计算像素差的Shannon熵；通过电流、速度测量计算机械能耗；结合信息理论与机械能耗绘制Motor Information Space。

**📊 数据集**

84段机器人颈部运动视频（roll、pitch、yaw及其组合），角度5°/10°，加速度0.5/2 m/s²，频率1–3次；以及17名参与者的感知问卷（共714条有效回答）。

**📈 对比分析**

对比不同自由度与运动参数的熵与能耗，发现2自由度熵最高、3自由度熵下降；ENIE最高为0.0121 bits/J（单自由度），最优条件roll+yaw 10°、0.5 m/s²、3 Hz得到5.26 bits，表明2自由度在信息与能耗平衡上最佳。

**⚠️ 局限性**

限制包括机器人仅具单一性别表情、运动速度慢于人类、颈部运动范围受限、缺乏肌肉细腻动态、感知实验样本量有限。

---

## 129. CILC: Cryptographically-secure Inter-agent Loop Closure Candidate Detection for Multi-Agent Collaborative SLAM

**arXiv ID:** 2607.06700 | [PDF](https://arxiv.org/pdf/2607.06700v1)

**作者:** Andrew Fishberg `[一作]` (Massachusetts Institute of Technology), Jonathan P. How `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 CILC 系统，利用安全多方计算实现协作 SLAM 中的 ILC 候选检测，避免公开全局描述符广播；

**💡 创新点**

创新点在于将 SMPC 应用于仅需向量相似度比较的 ILC 候选检测步骤，既显著降低隐私泄漏，又保持实时性能；

**🔧 技术方法**

使用公开的 SMPC 框架（如 PySyft / libsm），结合 Beaver Triples 等协议完成安全向量点积与阈值检查；

**📊 数据集**

使用多种视觉与激光全局描述符（DBoW2、NetVLAD、DINOv2、DINOv3、Scan Context）以及 Red Rover 机器人硬件的真实环境数据；

**📈 对比分析**

与传统不安全的 GD 直接比较方法对比，安全方案在 1536 维 DINOv2 描述符下实现约 2.7 ms 计算时间、0.075 MB 通信量，整体开销约为非安全版本的 2‑3 倍，仍保持实时可行；

**⚠️ 局限性**

局限包括：仅保护候选检测阶段，仍需在阈值满足后公开 GD；需要离线生成 Beaver Triples；在恶意输入时无法强制完整性；多机器人规模扩展与硬件加速仍待进一步研究。

---

## 130. Efficient Bayesian Deep Ensembles via Analytic Predictive Inference

**arXiv ID:** 2607.06776 | [PDF](https://arxiv.org/pdf/2607.06776v1)

**作者:** Sina Aghaee Dabaghan Fard `[一作]` (Texas A&M University), Jaesung Lee `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 Bayesian Deep Kernel Networks (BDKN)，通过将多条独立训练的神经网络产生的特征视为基函数，然后在其上进行闭式贝叶斯线性回归，得到可解释的后验权重和校准的预测不确定性。

**💡 创新点**

创新点包括：① 低维度的集成表示，减少推断成本；② 闭式贝叶斯聚合，避免近似推断，直接获得后验预测分布；③ 独立训练提升多样性并实现并行化，兼顾可扩展性与可解释性；④ 通过后验权重实现成员贡献可诊断，提升鲁棒性。

**🔧 技术方法**

使用技术包括：深度神经网络（多条独立网络）、贝叶斯线性回归、共轭先验（Jeffreys' 对噪声方差的先验）、Student‑t 预测分布、有限秩高斯过程解释、并行训练与小型线性系统求解。

**📊 数据集**

实验数据集为 UCI 回归基准：Boston、Concrete、Energy、Kin8nm、Naval、Power、Protein、Sarcos、Song、Wine、Yacht。

**📈 对比分析**

与 GP、Deep Ensembles、Deep Kernel Learning、Bayesian Last‑Layer、LD‑BLL、MFVI‑BNN、VBLL 等方法对比，评估指标为 RMSE 与 NLL。BDKN 在大多数数据集上与 Deep Ensemble 相当甚至优于其，在 NLL 上表现更好，且在高学习率和低学习率两种训练设置下均保持稳定，显著提升了不确定性校准与预测准确性。

**⚠️ 局限性**

局限性：仅针对回归任务，未针对分类或非高斯似然的情形；对极大规模数据集的评估有限；依赖线性聚合，可能不足以捕捉复杂的模型不确定性；未来工作需探索在强化学习、物理信息网络等领域的推广与不确定性进一步校准。

---

## 131. LEMUR 2: Unlocking Neural Network Diversity for AI

**arXiv ID:** 2607.06839 | [PDF](https://arxiv.org/pdf/2607.06839v1)

**作者:** Tolgay Atinc Uzun `[一作]` (University of Würzburg), Radu Timofte `[通讯]` (University of Würzburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了LEMUR 2，一个统一的可扩展框架，生成并评估超过14,000种神经网络架构，并提供跨域任务和设备部署的性能数据。

**💡 创新点**

首次实现了多源生成（AST、强化学习、遗传算法、分形结构、LLM检索增强）与部署感知（NN‑Lite、NN‑VR）相结合的完整生态，覆盖多模态任务和硬件平台。

**🔧 技术方法**

利用LLM提示、AST编辑、遗传算法、强化学习、分形网络、检索增强模块、自动化TensorFlow Lite/ONNX转换与Unity Barracuda推理、数据增强生成等技术。

**📊 数据集**

在CIFAR‑10/100、ImageNet、MS COCO、WikiText、TIMIT等公开数据集上训练评测，另外收集了6,000条数据增强流水线和750,000条训练记录。

**📈 对比分析**

与NAS Bench、NATS‑Bench等现有基准对比，LEMUR 2在有限训练预算下的精度、鲁棒性和硬件延迟表现均可与手工设计模型竞争，遗传和MMO模型在分类上最高达0.80+准确率，部署延迟在Android和VR设备上可实现接近帧率预算。

**⚠️ 局限性**

依赖现有公开代码库的可复现性受限，生成模型的多样性仍受LLM和搜索空间设计限制，且在极大规模训练和长周期收敛上仍需进一步验证。

---

## 132. SpaR3D-MoE: Adaptive 3D Spatial Reasoning from Sparse Views Meets Geometry-Inductive Mixture-of-Experts

**arXiv ID:** 2607.06620 | [PDF](https://arxiv.org/pdf/2607.06620v1)

**作者:** Haida Feng `[一作]` (Chinese Academy of Sciences), Yihong Wu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SpaR3D-MoE 框架，利用稀疏 RGB 视角实现 3D 空间推理

**💡 创新点**

通过自适应时空流形采样（ASMS）和几何诱导混合专家（HGI‑MoE）解决传统方法的采样冗余和模态冲突缺陷

**🔧 技术方法**

核心技术包括 ASMS、HGI‑MoE、指令‑姿态感知路由器（IPAR）以及多模态嵌入与动态专家调度

**📊 数据集**

使用 VSI‑Bench、ScanQA 与 SQA3D 三大基准数据集

**📈 对比分析**

与多种现有 3D‑Aware MLLM 及商业模型对比，在 VSI‑Bench 平均得分 63.5、ScanQA EM@1 30.4、SQA3D 平均 58.3，均刷新 SOTA，仅使用 32 帧稀疏采样

**⚠️ 局限性**

仍受预训练视觉‑语言模型能力限制，稀疏视角下对极端动态场景的鲁棒性有限

---

## 133. Intrinsic-Noise Consolidation: A Doob-Barrier-Conditioned Diffusion Turns Analog Device Noise into a Continual-Learning Resource

**arXiv ID:** 2607.06924 | [PDF](https://arxiv.org/pdf/2607.06924v1)

**作者:** Gunner Levi Howe `[一作]` `[通讯]`, Gunner Levi Howe

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在每个突触层面通过Doob h‑变换对权重扩散进行障碍条件化的记忆巩固规则，并展示了内在噪声对顺序任务保持率的非单调（倒U）影响。

**💡 创新点**

创新点在于：① 将Doob h‑变换引入到突触学习规则，形成噪声放大且在障碍处发散的恢复力；② 通过该规则得到的内在噪声优化窗口（倒U曲线）是传统基于固定驱动的巩固方法（OU、EWC、MESU）无法产生的可验证预测。

**🔧 技术方法**

技术方法包括：基于Euler–Maruyama积分的随机微分方程模型；障碍条件化的Doob h‑变换得到的额外漂移项；噪声驱动的仿真（白噪声、设备仿真噪声模型）；硬件级实测（BrainScaleS‑2 的自噪声测量与训练循环）以及对比实验的统计检验（Wilcoxon检验、paired测试）。

**📊 数据集**

使用的数据集有：Split‑MNIST（5个二分类子任务）和连续 Yin‑Yang 旋转任务（5个方向子任务），两者均在小型 MLP 或共享头网络上进行训练。

**📈 对比分析**

与基线方法（OU、EWC、MESU、plain SGD、Benna‑Fusi 阶层、经验回放）进行比较。实验表明：在噪声最优点，Doob 规则的保持率与 MESU 相当，显著优于 OU、EWC 和 SGD；回放方法仍能取得更高的保持率但不涉及噪声机制。倒U 形曲线在仿真、设备仿真以及实际硬件上均得到验证；在硬件上单个种子实验显示相对基线提升约几个百分点。

**⚠️ 局限性**

限制：① 仅在单头 MLP 上验证；② 采用对角 Fisher 作为重要性估计；③ 采用无限时间h‑变换而非有限期限生存函数；④ 对漂移力有限幅度截断；⑤ 硬件演示仅为单种子单点实验，能耗仅为模型估计；⑥ 对设备噪声分布仅检验加性和时间独立性，未对完整分布做统计。

---

## 134. Macroeconomic Message Passing for Anticipating Foreign Exchange Regime Changes: A Deep Logical Learning Approach using Graph Tsetlin Machines

**arXiv ID:** 2607.06719 | [PDF](https://arxiv.org/pdf/2607.06719v1)

**作者:** Christian Blakely `[一作]` (University of Agder), Melanie Gilmore `[通讯]` (Wells Fargo Advisors)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用图结构的图Tsetlin机（GraphTM）对美元/日元FX市场的四类交易时段（静止、稳态趋势、无序波动、趋势性波动）进行预测。

**💡 创新点**

创新点在于：①将宏观经济变量与技术指标映射为高维超向量化的有向多重图，利用信息传递与消息传递机制实现局部与全局逻辑规则的深度组合；②通过“排除推理”和高维超向量噪声鲁棒性显著提升了对低波动期的识别；③采用72小时多数投票标签与前瞻性评估，减少瞬时噪声影响。

**🔧 技术方法**

使用技术包括：图Tsetlin机（GraphTM）架构、消息传递的分层深度子句、Tsetlin自动机离散状态更新（I类与II类反馈）、高维稀疏超向量嵌入、基于宏观驱动和技术指标的图特征构造、72小时多数投票标签、基准对比模型（GBM、HMM、GraphNN、ConvTM、AutoML）。

**📊 数据集**

数据集：按小时收集的USD/JPY交易对价格、ATR与效率比（ER）技术指标，宏观经济驱动（美国与日本国债收益率、WTI油价）等共计10+维特征；训练/测试划分为60%/40%，并采用窗口滑动与72小时缓冲确保无泄漏。

**📈 对比分析**

比较方法：在同一OOS序列上对比多种模型，报告总体准确率与各类准确率；Full GraphTM在四类中的平均准确率约70%，在类0（静止）和类2（无序波动）表现最佳；相比之下，GBM、GraphNN、HMM及ConvTM在类3（高波动趋势）上明显落后，且整体准确率低于GraphTM。

**⚠️ 局限性**

局限性：①对稀有高波动趋势类（类3）样本不足导致准确率低；②模型解释性虽高但规则提取仍不完整；③对宏观特征的依赖在不同市场/周期可能需重构；④消息传递和高维超向量的计算开销在极大图规模下仍需优化。

---

## 135. Robust Human-AI Complementarity under Uncertainty

**arXiv ID:** 2607.06656 | [PDF](https://arxiv.org/pdf/2607.06656v1)

**作者:** Yewon Byun `[一作]` (Carnegie Mellon University), Bryan Wilder `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在决策者对AI质量不确定的情境下，如何通过鲁棒决策规则实现人类与AI的补充，并阐明误差相关结构对补充效益的决定性影响。

**💡 创新点**

提出在不确定AI质量时，AI预测误差与人类误差的负相关是实现鲁棒补充的必要与充分条件；同时给出在高斯与非高斯模型下的完整理论证明，并验证在真实预测基准上现有LLM往往满足正相关，导致补充失效。

**🔧 技术方法**

采用统计决策理论、鲁棒优化和不确定集合框架；利用线性高斯模型、单调非线性信号模型进行理论推导；通过仿真和实际预测基准实验评估方法。

**📊 数据集**

使用两大公开预测基准：ForecastBench（时间序列与预测市场问答）和TESS（社会科学实验效应预测）。

**📈 对比分析**

将人类单独预测、AI单独预测和鲁棒组合预测在MSE（均方误差）及二元决策收益上进行对比。结果显示：在误差负相关时，鲁棒组合显著优于人类单独预测；在误差正相关时，几乎无提升，且现有提示策略难以将相关性转为负。

**⚠️ 局限性**

局限性：当误差正相关时，几乎无法实现补充；现有的提示或后处理手段难以可靠改变误差相关结构；方法对AI质量的不确定性假设较为理想化，实际模型的误差结构难以完全控制。

---

## 136. EditVerse3D: High-Quality 3D Object Editing with Region-Aware Learning

**arXiv ID:** 2607.07187 | [PDF](https://arxiv.org/pdf/2607.07187v1)

**作者:** Youtan Yin `[一作]` (Nanyang Technological University), Guosheng Lin `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种端到端的3D局部编辑框架 EditVerse3D，利用粗略的3D包围盒和参考2D图像实现高质量的对象编辑。

**💡 创新点**

创新点包括：① 区域感知自适应损失，实现对难学区域的自适应加权；② 联合归一化解决3D对象与遮罩的空间对齐问题；③ 在训练阶段使用粗粒度遮罩和数据增强，提升对真实场景的鲁棒性；④ 构建包含约85k网格、50万编辑对的大规模3D编辑数据集。

**🔧 技术方法**

技术方案主要基于 TRELLIS 3D生成框架，采用 rectified flow 模型进行结构和纹理的编辑；结合区域感知损失、硬样本挖掘、联合归一化、随机视角条件生成和多视角数据增强等训练策略。

**📊 数据集**

使用 Partverse 和 Objaverse 的分割信息构建的数据集，包含约85k条网格和500k条编辑对，进一步通过 3D 分割和掩模扩展到更大规模。

**📈 对比分析**

与 Instant3dit、TRELLIS 的 Repaint/FlowEdit、VoxHammer 等方法进行对比。EditVerse3D 在替换和添加任务中均取得最低 Chamfer Distance、最高 PSNR/SSIM、最低 LPIPS/FID，显示出显著的性能优势，且推理速度与原始 TRELLIS 相当。

**⚠️ 局限性**

局限性：① 目前训练数据主要为添加（add）操作，替换（replace）的泛化效果需要进一步验证；② 仍需用户提供参考2D图像和粗略包围盒，缺乏完全自动化；③ 训练成本高（约10k GPU 小时）；④ 对细粒度编辑和极端遮罩形状的鲁棒性仍待提升。

---

## 137. Stable Matchings with Minimum Utility Gap

**arXiv ID:** 2607.07160 | [PDF](https://arxiv.org/pdf/2607.07160v1)

**作者:** Yao Sheng `[一作]` (Institute of Science Tokyo), Yu Yokoi `[通讯]` (Institute of Science Tokyo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种在多对多稳定匹配中最小化参与代理人效用差距的新优化问题，并给出了多项式时间求解算法。

**💡 创新点**

创新点在于引入“最小效用差距”公平度量，利用旋转偏序的链结构而非传统的最小割或子模函数框架，实现了新的可解性结果。

**🔧 技术方法**

主要技术包括旋转图（rotation poset）的构造、旋转链的单调性分析、区间可行性检查以及滑动窗口搜索求最优区间。

**📊 数据集**

实验仅使用合成实例验证，未公开真实数据集。

**📈 对比分析**

与现有的最小割可表示性和子模函数最小化方法对比，证明该问题不属于前者；算法复杂度为 O(n⁴ + n²T_v)，在总效用或平均值等常见价值函数下可简化为 O(n⁴)。

**⚠️ 局限性**

局限性包括：若允许首选列表出现平局则问题变为NP‑hard；当关注代理人跨两侧时，问题不再是子模函数；并且当前算法对非常大的 n 仍有四次方时间瓶颈。

---

## 138. EmbodiedGen V2: An Agentic, Simulation-Ready 3D World Engine for Embodied AI

**arXiv ID:** 2607.07459 | [PDF](https://arxiv.org/pdf/2607.07459v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 139. Reward Valuation in Vision Language Models: Causal Mechanisms Underlying Anhedonia

**arXiv ID:** 2607.06626 | [PDF](https://arxiv.org/pdf/2607.06626v1)

**作者:** Melika Honarmand `[一作]` (EPFL), Martin Schrimpf `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在视觉语言模型中识别与大脑核壳相似的奖励预测单元，并通过定向抑制诱发类似抑郁症中的无快感（anhedonia）行为。

**💡 创新点**

创新点在于将临床心理测量和神经科学中的奖励缺失模型引入AI模型，实现了从神经机制到可解释AI行为的因果桥接。

**🔧 技术方法**

使用的技术包括基于Transformer的视觉语言模型（Qwen2‑VL‑7B‑Instruct）、激活补丁（activation patching）以及基于3σ阈值的功能定位。

**📊 数据集**

使用的数据集包括ASDiv、Probability‑EEfRT、DARS、MAP‑SR、AES等心理测量问卷及自定义奖励决策任务。

**📈 对比分析**

通过与未扰动模型及随机扰动基线对比，模型在奖励决策任务中显著降低高努力高奖励选择率，同时保持任务准确率，证明该扰动只影响动机而非认知能力。

**⚠️ 局限性**

局限性包括仅聚焦奖励预测阶段，缺乏对学习和消费阶段的模拟，且仅在单一VLM架构上验证，尚未证明在其他模型或真实脑数据上的普适性。

---

## 140. FMMVCC: Fuzzy Mamba-based Multi-View Contrastive Clustering for Univariate Time Series

**arXiv ID:** 2607.07258 | [PDF](https://arxiv.org/pdf/2607.07258v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 141. WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time

**arXiv ID:** 2607.06988 | [PDF](https://arxiv.org/pdf/2607.06988v1)

**作者:** Yusen Feng `[一作]` (Peking University), He Wang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在现有预训练的世界动作模型（WAM）上，提出一种测试时训练（TTT）框架，用原始人类视频直接适配机器人行为，且不需要机器人演示、标注或对整个模型进行微调。

**💡 创新点**

创新点在于：①将人类视频作为部署时的“快速权重记忆”，通过自监督视频预测和键值内存重构目标在元训练阶段学习人类-机器人对齐；②在测试时仅更新视频侧的快速权重，保持WAM冻结，从而实现高效、可重复使用的行为转移；③通过元训练与测试时自监督双阶段训练，显著提升在新环境中的任务成功率。

**🔧 技术方法**

技术包括：世界动作模型（LDA基础）、快速权重（TTT）分支、键值内存重构损失、元训练（inner‑loop 更新 + outer‑loop 任务损失）和自监督视频预测；所有模型参数均采用梯度下降更新，仅在快速权重上进行内部迭代。

**📊 数据集**

使用了约 2,286 条人类‑机器人同步演示数据，涵盖 9 种操纵任务（如 Transfer Bottle、Table Bussing 等），人类演示以 GoPro 近景 RGB 录像收集，无姿态或 3D 轨迹标注。

**📈 对比分析**

与 LDA 基线、WAM‑Co‑train、WAM‑ICL、EgoScale、π_0.5 等方法对比，WAM‑TTT 在“新家庭环境”设置下平均成功率 46.2%，显著高于 LDA（32.5%）、WAM‑Co‑train（25.3%）、EgoScale（15.0%）、π_0.5（14.8%）和 WAM‑ICL（7.1%）。在单一任务上，WAM‑TTT 在 7 项任务上完全领先或相等。

**⚠️ 局限性**

局限性包括：①元训练时人类与机器人片段的相位配对误差可能导致适配信号不佳；②部署时快速权重的表达能力受限，若目标任务与元训练分布偏差较大，适配效果下降；③仅使用 egocentric RGB 视图，未利用手部姿态、接触或 3D 场景信息，限制了对更复杂动作的迁移能力。

---

## 142. MiLSD: A Micro Line-Segment Detector for Resource-Constrained Devices

**arXiv ID:** 2607.06600 | [PDF](https://arxiv.org/pdf/2607.06600v1)

**作者:** Parsa Hassani Shariat Panahi `[一作]` (Iran University of Science and Technology), M. Hassan Najafi `[通讯]` (Case Western Reserve University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了MiLSD，一种针对微控制器内存极限的线段检测器，并系统评估了不同输出表示、量化位宽和推理后处理对精度的影响。

**💡 创新点**

创新点包括：1) 在子兆字节内存预算下实现可量化的F-Clip输出表示；2) 对4-bit量化对角度回归的敏感性进行首次量化分析；3) 在STM32H7上通过容量扩展、亚像素解码、TTA和轻量级验证头实现从10.6到24.1的显著精度提升。

**🔧 技术方法**

使用的技术包括：全卷积轻量化骨干网络、中心-长度-角度（F-Clip）输出、对称整数量化（int8/int4）、量化感知训练、亚像素峰值拟合、测试时数据增强、线段验证头和CMSIS‑NN推理。

**📊 数据集**

使用的数据集为ShanghaiTech Wireframe，用于评估结构平均精度（sAP）和Q1/Q2指标。

**📈 对比分析**

与现有学习型线段检测器（如L-CNN、HAWP、ULSD、LETR等）以及传统经典检测器（LSD、EDLines）进行对比；在0.25 MB模型上实现10.6的sAP，扩大至1 MB模型后提升到24.1，处于极低资源端但远低于GPU级别的精度。

**⚠️ 局限性**

局限性包括：1) 在子兆字节内存下仍无法达到GPU级别的精度；2) 4-bit量化导致角度回归严重失真；3) 依赖较大的内存预算（1 MB）才能显著提升，限制了在更小型MCU上的适用性；4) 仍需外部验证头和TTA，增加了推理时延。

---

## 143. Imputation Meets Clustering: Exploiting Latent Subgroup Structure for Missing Data Recovery

**arXiv ID:** 2607.06930 | [PDF](https://arxiv.org/pdf/2607.06930v1)

**作者:** Chuyao Zhang `[一作]` (Guangdong University of Technology), Yiu-ming Cheung `[通讯]` (Hong Kong Baptist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合聚类与生成式对抗网络的缺失值填补框架 CAGI，利用动态子群划分作为局部先验，并在多级损失下共同优化聚类与填补。

**💡 创新点**

创新点在于将聚类与填补协同迭代、使用子群条件生成器、引入多层目标（实例级对抗+重建+分布级Sinkhorn正则）以及缺失容忍的聚类初始化。

**🔧 技术方法**

核心技术包括缺失容忍的 K‑means/K‑prototypes、Cluster‑Conditioned GAN（GAIN‑style）、多层损失组合、Optimal Transport 的 Sinkhorn 散度正则以及自适应聚类更新循环。

**📊 数据集**

实验涵盖 14 个公开数据集（6 纯数值、3 纯分类、5 混合型），覆盖多种特征类型和分布差异。

**📈 对比分析**

与 15 组代表性方法（统计、机器学习与深度学习）在 MCAR 设定下进行 RMSE/PFC 评估，CAGI 在多数数据集上取得首位或次优成绩，并在下游分类（AUROC）和聚类（ARI）任务中同样表现最佳或相近，显著提升填补质量和后续任务效果。

**⚠️ 局限性**

局限在于需手动设定子群数 K，未实现自动化选择；仅在 MCAR 环境下验证，对 MNAR 或时间序列缺失机制的适应性尚待研究；循环更新频率与计算成本需进一步优化。

---

## 144. Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories

**arXiv ID:** 2607.06648 | [PDF](https://arxiv.org/pdf/2607.06648v1)

**作者:** Hengyu Jin `[一作]` (Tongji University), Di Wang `[通讯]` (Provable Responsible Ai And Data Analytics (Prada) Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 GPT‑2 small 的四种训练范式（CoT、NoCoT、COCONUT、CODI）在训练过程中进行持续评估，利用 ProsQA 的可验证反事实编辑和噪声消融激活补丁跟踪 latent reasoning 的 faithfulness 变化。

**💡 创新点**

首次揭示 latent reasoning 的 faithfulness 随训练阶段演化，并证明其受答案格式和训练机制影响，展示不同范式在同一终点表现出截然不同的轨迹。

**🔧 技术方法**

采用 GPT‑2 small 背景模型，训练四种范式；使用 ProsQA 的 deterministic BFS oracle 构造反事实输入；对 latent 步骤进行全步噪声消融补丁；计算 OCR、PWC、CC、CFR、IE^contrast 等指标。

**📊 数据集**

主要使用 ProsQA（图搜索问题）进行反事实验证；在 GSM‑8K 上构造开放式与二选一格式的对比实验；对照 CoT 与 NoCoT 文本基线。

**📈 对比分析**

通过在每个保存的 checkpoint 上测量预测准确率、CC/CFR（faithfulness 指标）和 OCR/PWC（激活补丁效果），发现两种 latent 方法最终在 faithfulness 上与 NoCoT 相近，但训练轨迹和激活贡献随答案格式显著变化；在二选一格式下激活贡献衰减，而在开放式格式下则上升。

**⚠️ 局限性**

实验局限于 GPT‑2 small 规模、ProsQA 可验证反事实的前提，以及缺乏对更大模型或不同任务的泛化验证；答案格式转换实验仅在 GSM‑8K 上完成，可能不适用于其他类型的开放式问题。

---

## 145. Rail Track Extraction from Rasterized Classified Point Clouds Using a Full-Resolution, Fully Convolutional Recurrent Neural Network

**arXiv ID:** 2607.06829 | [PDF](https://arxiv.org/pdf/2607.06829v1)

**作者:** Alexander Gribov `[一作]` (Esri), Jie Chang `[通讯]` (Esri)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一套基于全分辨率递归膨胀融合网络（FRPDF）的光栅化分类点云铁路轨道提取方法，能够在噪声环境下实现高精度的轨道和轨道中心线自动生成。

**💡 创新点**

创新点包括：①利用合成光栅训练，无需真实铁路数据；②全分辨率无下采样结构保持细节；③递归融合模块在每轮迭代中逐步抑制噪声并补全缺口；④结合形态学闭运算、矢量化与DTW匹配，实现轨道顶端和中心线的精细化提取。

**🔧 技术方法**

主要技术手段：光栅化、全卷积递归膨胀网络（FRPDF）、形态学闭运算、矢量化、动态时间规整（DTW）、一维平滑和三维投影重建。

**📊 数据集**

使用了两类数据：①大规模合成轨道光栅数据（多种几何、宽度、噪声变化）；②真实移动激光雷达点云（印度铁路多轨道段）。

**📈 对比分析**

在1024×1024的合成测试光栅上评估，Precision 0.809、Recall 0.946、F1 0.872，模型参数约29.4万，显存占用1.55 GB，单图推理时间约98 ms；未与其他网络做严格对比，因模型重构难度大。

**⚠️ 局限性**

局限性：目前仅适用于简单平行轨道，难处理转辙、交叉口等复杂几何；合成训练对真实噪声和遮挡的覆盖仍有限；需先完成高质量的轨道点分类，整体流程对前置步骤有依赖。

---

## 146. Gimitest: A Comprehensive Tool for Testing Reinforcement Learning Policies

**arXiv ID:** 2607.07029 | [PDF](https://arxiv.org/pdf/2607.07029v1)

**作者:** Dennis Gross `[一作]` (Simula Research Laboratory), Arnaud Gotlieb `[通讯]` (Simula Research Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个名为Gimitest的通用框架，用于单/多智能体RL策略的SBST、MT、AT等多种测试，并支持日志记录和GPT‑4自动生成测试代码。

**💡 创新点**

创新点在于统一的装饰器模式实现对Gymnasium/PettingZoo环境的step/reset方法进行拦截，从而实现多种测试方式和并行支持；同时结合LLM实现自动测试代码生成和失败分析。

**🔧 技术方法**

采用Python实现，利用装饰器模式、Search‑Based Software Testing、Metamorphic Testing、Adversarial Testing、GTest/GLogger类，以及GPT‑4进行代码/失败分析。

**📊 数据集**

使用公开RL环境如Farama Gymnasium（Lunar Lander、CartPole、MountainCar等）和PettingZoo（Connect Four、Waterworld）以及对应的训练好的策略。

**📈 对比分析**

通过与现有方法对比，示例实验中SBST三种方法在Lunar Lander上发现多样化失败，AT在MountainCar上定位易被攻击状态，MT在CartPole上挖掘100+bug；未与其他工具直接性能比较，但实验表明工具能快速生成并执行大量测试。

**⚠️ 局限性**

局限性包括目前不支持并行测试执行、对第三方仿真器（如Carla）缺乏支持、测试方法实现需手动编写子类、未对大规模数据集的性能做系统评估。

---

## 147. New Cross-Sensory Approach to Designing Restorative Virtual Environments

**arXiv ID:** 2607.06901 | [PDF](https://arxiv.org/pdf/2607.06901v1)

**作者:** Rachel Masters `[一作]` (Colorado State University), Francisco Ortega `[通讯]` (Colorado State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出在虚拟自然环境(VNE)中探索跨感官交互，以实现更有效的压力减轻与注意力恢复。

**💡 创新点**

创新点在于强调跨感官（音、视、嗅、触）交互的重要性，并呼吁建立跨感官研究框架，而非仅聚焦单一感官的叠加效应。

**🔧 技术方法**

采用虚拟现实技术结合音频、视觉、气味、温度等多感官刺激进行设计与评估。

**📊 数据集**

本研究为综述与提议，未使用具体数据集。

**📈 对比分析**

未进行实验或性能评估，主要是对现有文献进行综述，并提出未来研究的对比设计思路。

**⚠️ 局限性**

局限在于技术手段对嗅觉、触觉的可实现性有限，跨感官交互的研究方法尚未成熟，需要进一步实验验证。

---

## 148. Naming the Concepts Classifiers Rely On: Language-Anchored Decomposition for Faithful Explanation

**arXiv ID:** 2607.07264 | [PDF](https://arxiv.org/pdf/2607.07264v1)

**作者:** Ahsan Habib Akash `[一作]` (West Virginia University), Prashnna Kumar Gyawali `[通讯]` (West Virginia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种后置（post‑hoc）概念解释框架，能够在不改动已部署的视觉分类器的前提下，为每个类别生成可命名、空间定位且与模型决策高度一致的概念解释。

**💡 创新点**

创新点在于反向非负矩阵分解（NMF）：将基于大语言模型与CLIP图像‑文本相似度构造的语言锚定矩阵固定为系数矩阵，只学习基向量，从而实现概念的自动命名、保持模型原始表现并提升解释的可信度。

**🔧 技术方法**

核心技术包括：大语言模型（LLM）生成类别特定概念词表；CLIP 视觉‑语言模型计算局部相似度图；非负矩阵分解（NMF）与投影梯度下降（PGD）求解；非负最小二乘估计推断新图像的概念激活。

**📊 数据集**

实验使用 ImageNet（500类）、Places365（364类）和视网膜光学断层图像数据集 OIR‑5K（5个眼科疾病类）来验证方法。

**📈 对比分析**

与ICE、CRAFT、FACE 等基准方法比较，保持 100% 的预测准确率；在概念插入/删除实验中，C‑Ins（插入效果）与 C‑Del（删除效果）均优于或相当于最强基准；在临床数据上，C‑Del 最高，显示该方法在高风险医学场景下的可解释性优势。

**⚠️ 局限性**

局限性包括：概念词表生成依赖 LLM 与 CLIP 的质量，可能在跨域或极细粒度概念上表现不佳；固定概念预算（如 r=25）可能限制解释的完整性；对非视觉属性或跨类别一致性的处理尚需进一步研究。

---

## 149. Making Implicit Preservation Intent Explicit in Conversational Image Editing

**arXiv ID:** 2607.07051 | [PDF](https://arxiv.org/pdf/2607.07051v1)

**作者:** Soomin Han `[一作]` (Sogang University), Buru Chang `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 OCCUR‑Bench 诊断基准，专注于会话式图像编辑中被遮挡后再显现的内容的时间保持，并提出了无训练的 ReSpec 框架，通过 VLM 控制器显式推断保持目标、检索历史参考图像并重写指令，实现对历史信息的利用。

**💡 创新点**

创新点在于：①构建了专门评估遮挡-显现情形下时间保持的基准；②提出将隐式保持意图显式化的无训练框架 ReSpec，结合历史视觉证据和指令重写，显著提升恢复一致性与时间一致性。

**🔧 技术方法**

使用技术包括：VLM（如 Qwen3‑VL‑8B‑Instruct）做控制器，Flux.2、OmniGen2 等可接受参考图的 in‑context 图像编辑模型，以及基于检测/分割的区域掩码与 PSNR、LPIPS、CLIP 的一致性评估。

**📊 数据集**

数据集为 COCO、PIE‑Bench 与 HQ‑Edit 的原始图像，经过人工筛选与验证后生成 4,400 个 2–5 轮遮挡‑显现场景，提供历史参考状态。

**📈 对比分析**

与 MC‑Edit、Layer‑wise Memory 等多轮基线相比，OCCUR‑Bench 指标显示这些模型时间一致性低；ReSpec 在 Flux.2 上提升整体时间一致性 0.129（主要来自恢复一致性 +0.161），在 OmniGen2 上亦有提升；人类评测与自动指标高度相关，验证了性能提升。

**⚠️ 局限性**

局限性包括：评估依赖检测/分割得到的掩码，误差可能影响分数；ReSpec 的性能受 VLM 控制器准确性的限制；以及额外的 VLM 推理开销，尤其在长编辑序列中可能显著。

---

## 150. When Does In-Context Search Help? A Sampling-Complexity Theory of Reflection-Driven Reasoning

**arXiv ID:** 2607.06720 | [PDF](https://arxiv.org/pdf/2607.06720v1)

**作者:** Yotam Wolf `[一作]` (Hebrew University), Amnon Shashua `[通讯]` (Hebrew University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将大语言模型的“在上下文搜索”（in-context search）视为对基准模型先验的近似推断，阐明反思（reflection）在降低采样复杂度中的作用，并通过理论证明、跨熵训练以及与可验证奖励的强化学习框架对齐，展示当反思能可靠定位早期错误时，序列化推断能将指数级的采样成本压缩为多项式级。

**💡 创新点**

创新点在于：① 将在上下文搜索建模为先验‑后验推断并量化采样复杂度；② 明确提出早期错误定位是实现指数-多项式加速的关键，并给出相应的充分必要条件；③ 证明近似后验更新与交叉熵训练可实现这一加速，并且在理想的强化学习阶段化扩展中该更新是最优的；④ 在真实大规模推理模型（DeepSeek‑R1、Qwen‑2.5‑1.5B）上验证理论预言。

**🔧 技术方法**

技术方法包括：理论分析（近似推断、采样复杂度证明、误差延迟与假阳性分析）；后验更新规则（基于反思次数的对数回归衰减）；交叉熵训练收敛性分析；以及与阶段化强化学习（RLVR）框架的对齐。

**📊 数据集**

实验数据集主要有：1）人工合成的推理轨迹（包含注入错误的算术链和方程求解任务），2）AIME 2025 真实推理题目，3）使用 DeepSeek‑R1‑Distill‑Qwen‑2.5‑1.5B（与 7B 版本对比）生成的推理轨迹。

**📈 对比分析**

对比方法主要是：① 基准模型的单样本推理（零样本）；② 并行采样（多样本）；③ 通过后验更新的序列化采样；实验结果显示，在早期错误定位得到保证的情形下，后验更新能以多项式（O(nW)）的迭代次数在高置信度下获得正确答案，而并行采样需指数级（≈2ⁿ/²）次数；在反思失效或延迟的情况下，性能退化至并行采样水平。

**⚠️ 局限性**

局限性包括：① 对反思可靠性的假设（必须在多次失败后才能正确定位早期错误）在实际模型中可能不完全成立；② 仅在单解或有限解数的情形下给出严格证明，针对多解情形的泛化仍需深入；③ 对模型可扩展性与大规模参数模型的可解释性未进行全面验证；④ 理论框架未涵盖所有外部工具或增强型提示的影响，实际搜索策略可能更为复杂。

---

## 151. Evaluation of Multilingual Ability to Use Spatial Deictic Expressions in Vision-Language Models

**arXiv ID:** 2607.07251 | [PDF](https://arxiv.org/pdf/2607.07251v1)

**作者:** Kaito Watanabe `[一作]` (University of Tokyo), Hitomi Yanaka `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于记忆游戏范式，构建了一个多语言空间指示词（demonstratives）使用能力的基准评测。

**💡 创新点**

首次提出针对视觉‑语言模型（VLM）在不同语言和距离下使用空间指示词的评测框架，并揭示模型与人类在距离敏感性上的显著差异。

**🔧 技术方法**

利用Gemma 3、Qwen3‑VL等开源多语言VLM，并通过手工编写的提示（prompt）让模型生成“指示词 + 颜色 + 形状”三词描述；使用Blender生成可控的图像。

**📊 数据集**

自制数据集：60张图片（3 个距离 × 5 形状 × 4 颜色），在日语、韩语、英语、中文四种语言下进行测试。

**📈 对比分析**

通过计算模型与人类在各距离下指示词的概率分布，并用 Jensen‑Shannon 距离评估相似度；结果显示所有模型都无法重现人类的距离依赖分布，且表现受语言资源和模型架构影响。

**⚠️ 局限性**

局限性包括：样本量小、语言覆盖有限、仅使用单一提示语句、模型对图像识别不充分、人工验证带来主观性，且缺乏更自然的评测场景。

---

## 152. Online Data Selection Is Implicit Alignment

**arXiv ID:** 2607.07023 | [PDF](https://arxiv.org/pdf/2607.07023v1)

**作者:** Aoxiong Zeng `[一作]` (East China Normal University), Xiangquan Yang `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了在线数据选择对大型语言模型(SFT)行为的隐式对齐影响，提出对齐漂移审计(ADA)与对齐感知选择(AAS)框架；

**💡 创新点**

创新点在于把在线选择视为隐式奖励模型，正式化了“对齐漂移”，并提出可测评和控制该漂移的评估协议与选择算法；

**🔧 技术方法**

采用了在线加权重的重采样SFT，LoRA参数高效微调，自动化对齐评估指标(有用性、拒绝率、冗长度、真确性、同情性、越狱鲁棒性等)与属性丰富度诊断；

**📊 数据集**

使用约30万条指令式数据池（含UltraChat、OpenHermes、数学、代码、安全拒绝等），在Llama‑3.1‑8B基础模型上进行实验；

**📈 对比分析**

与随机、基于损失、质量、多样性选择对比，结果显示在相同token预算下，质量或损失选择可提升任务准确率但导致显著的行为漂移，AAS在保持性能的同时将漂移减半；

**⚠️ 局限性**

局限性包括评估指标的主观性与噪声、属性标注误差、对不同基础模型的泛化不确定，以及跨阶段、多模型复现需求不足。

---

## 153. Ad Headline Generation using Self-Critical Masked Language Model

**arXiv ID:** 2607.06818 | [PDF](https://arxiv.org/pdf/2607.06818v1)

**作者:** Yashal Shakti Kanungo `[一作]`, Aruna Rajan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于掩码语言模型（MLM）并结合Self-Critical强化学习的多产品条件广告标题生成框架，能够一次性生成适用于多个产品的统一标题。

**💡 创新点**

创新点包括：①将BERT MLM与Self-Critical policy gradient结合，直接优化评价指标；②在输入中同时编码多个产品信息，生成能够统一体现多产品特征的标题；③采用掩码注意力机制在自回归生成过程中消除曝光偏差。

**🔧 技术方法**

使用技术包括：BERT Large MLM、Self-Critical policy gradient（REINFORCE）、掩码注意力、Beam Search + 长度归一化、HuggingFace Transformers实现。

**📊 数据集**

数据集为约50万条亚马逊广告活动，包含多产品及其对应的人工审核标题，按85%/5%/10%划分为训练、验证和测试集。

**📈 对比分析**

通过与Pointer Network bi‑LSTM基线及Self‑Critical bi‑LSTM进行对比，指标涵盖Rouge‑L、BLEU‑4、METEOR、Cosine Similarity以及人工质量与语法审核。SC‑MLM在所有指标上均优于基线（例如Rouge‑L 6.33 vs 0.62，语法审核准确率 98.13% vs 93.14%），并在质量审核中获得最高平均评分和最多 3/3 评级。

**⚠️ 局限性**

局限性：训练耗时长、能源消耗高；模型对标题噪声、关键词堆叠敏感；仅在英文亚马逊数据上验证，跨语言或跨平台的适用性尚未评估。

---

## 154. Stage-Aware Adaptation and Distribution Calibration for Subject-Driven Personalized Text-to-Image Generation

**arXiv ID:** 2607.07173 | [PDF](https://arxiv.org/pdf/2607.07173v1)

**作者:** Wenyan Xu `[一作]` (Guangdong University of Technology), Alizer Wong `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了训练端阶段感知低秩适配(SPaRa)与推理端分布校准候选选择(DCAL)相结合的个性化文本到图像生成框架

**💡 创新点**

通过对不同去噪阶段分配适配容量并在推理时综合身份、一致性、文本对齐与多样性评分，实现身份与多样性权衡的可调控制

**🔧 技术方法**

使用低秩适配（LoRA）并在每个去噪步设置可变缩放因子；推理时利用CLIP与DINOv2特征计算身份与文本相似度，并加入多样性惩罚的加权选择策略

**📊 数据集**

在DreamBooth 30个受试主体的Few-Shot数据集上，配合SDXL 1.0预训练模型进行实验

**📈 对比分析**

与DreamBooth LoRA r=16基线对比，DCAL在1‑LPIPS、CLIP‑I、DINO‑I、CLIP‑T上均有提升，但在CLIP/DINO对比多样性与对比LPIPS上下降；阶段感知推理缩放则提升多样性但牺牲身份一致性

**⚠️ 局限性**

缺乏完整的SPaRa–DCAL全实验结果、同协议下PaRa基线缺失、未完成种子/候选数/指导尺度等鲁棒性与多样性分析，且仅在部分受试主体上验证，尚未提供完整性能与可复现性证明

---

## 155. Pre-Training on Software Engineering Texts: Effects on Domain Adaptation and General-Language Understanding

**arXiv ID:** 2607.06613 | [PDF](https://arxiv.org/pdf/2607.06613v1)

**作者:** Fabian C. Peña `[一作]` (University of Passau), Steffen Herbold `[通讯]` (University of Passau)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在软件工程（SE）文本上进行领域适配的预训练方法，比较连续预训练（CPT）与从零预训练（PTS）的效果，并评估其对SE文本理解与通用语言理解的影响。

**💡 创新点**

提出了基于token与计算成本匹配的预算对照实验框架，并系统评估了不同模型家族、规模及预训练方式在SE域适配与通用语言保持上的权衡，揭示CPT在大多数情况下仅带来微小提升且不显著退化通用语言能力。

**🔧 技术方法**

使用了Encoder/Decoder语言模型（BERT、RoBERTa、ModernBERT、GPT-2、Llama 3.2、CodeBERT、CodeLlama、StarCoder2）在统一的词表和训练脚本下进行CPT与PTS，并在SEU和SuperGLUE基准上进行微调与评估。

**📊 数据集**

构建了大规模SE文本语料库，来源包括GitHub Issue/PR、Stack Overflow/Software Engineering、Jira issue、arXiv论文，去除代码块并进行去重、去污染，最终获得约1.85亿文档、18.5B tokens。

**📈 对比分析**

对比方法：在相同token数（3.3B）与相同计算量（6.01×10^18 FLOPs）下，比较官方检查点、CPT与PTS在SEU（平均提升约2-3%）和SuperGLUE（平均下降≤1%）上的得分差异；CPT往往不显著提高SEU并保持通用语言，PTS则普遍显著下降。

**⚠️ 局限性**

局限性：仅评估单一SE文本语料，未覆盖非英文或专有工件；使用单一随机种子导致实验方差不易评估；tokenizer差异可能导致CPT/PTS对比不纯粹；基准任务数量有限，统计显著性受限；计算匹配公式对Encoder可能不精确。

---

## 156. Quantum Sampling Architecture for Protein Structure Reconstruction on Utility-Scale Hardware

**arXiv ID:** 2607.06971 | [PDF](https://arxiv.org/pdf/2607.06971v1)

**作者:** Yuqi Zhang `[一作]` (Kent State University), Qiang Guan `[通讯]` (Kent State University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `09944146-298c-433e-89df-37255de463d7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 QSAD（Quantum Sampling And Decomposed Reconstruction）框架，用量子-经典方法预测 5–18 个残基的结合口短肽结构，并重建其能量景观。

**💡 创新点**

创新点包括：① 在氨基酸层面使用四面体格子编码折叠物理，显著降低量子比特需求至 O(N²)；② 用单向哈密顿演化代替迭代优化，实现无反馈、噪声鲁棒的非迭代采样；③ 采用多 β 值分层采样，提升低能量覆盖；④ 将量子采样结果映射为晶体结构的统计分布，并通过 PCA‑spline 对能量景观进行近似重建。

**🔧 技术方法**

技术手段：量子硬件 IBM Heron R2 上执行 Suzuki–Trotter 分解的哈密顿演化，随机 ansatz 生成多样初态；经典后处理包括位串解码、有效结构筛选、基于物理特征的多维排名和所有原子重建；能量景观重建使用 PCA 与薄板样条插值；对比实验使用 AlphaFold3、ColabFold‑MSA、ColabFold、ESMFold、OmegaFold、OpenFold 以及 VQE。

**📊 数据集**

数据集：101 条结合口短肽（5–18 残基），其中 55 条来自 QDockBank（已公开 VQE 结果），46 条来自 PDBbind，全部配有晶体结构对照。

**📈 对比分析**

评价方法：以 N‑α 原子 RMSD 与实验晶体结构对比；QSAD 在所有 101 个样本中平均 RMSD 为 2.7 Å（中位数 2.7 Å），显著低于 AlphaFold3（4.8 Å）、ColabFold‑MSA（6.1 Å）、ColabFold（7.5 Å）、ESMFold（8.6 Å）、OmegaFold（9.4 Å）、OpenFold（5.9 Å）和 VQE（3.7 Å 中位数）。执行时间上 QSAD 以平均 33 分钟完成单例，VQE 需 467 分钟，速度提升 27×；在噪声耐受性实验中，QSAD 能在典型 2% depolarizing 噪声下仍恢复基态能量，VQE 在同一噪声下失效。

**⚠️ 局限性**

局限性：① 格子离散化导致 1.6 Å 左右的精度上限；② β 调度为经验式，未针对每个蛋白自适应；③ 能量景观重建仅为 2D PCA 投影，缺乏完整热力学信息；④ 目前仅适用于 5–18 残基的短肽，且对更大体系的扩展需要更多量子比特。

---

## 157. A Continual Learning Framework for Adaptive Control of Modular Soft Robots

**arXiv ID:** 2607.06740 | [PDF](https://arxiv.org/pdf/2607.06740v1)

**作者:** Nilay Kushawaha `[一作]` (Scuola Superiore Sant’Anna), Egidio Falotico `[通讯]` (Scuola Superiore Sant’Anna)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于持续学习的软体模块化机器人控制框架（SMPL），实现了模块增删时的增量学习与分布式控制；

**💡 创新点**

创新点在于将渐进式神经网络与闭环前馈模型相结合，既能在模块变化时保持已有知识，又能为固定结构机器人实现模块级局部控制；

**🔧 技术方法**

使用了渐进式神经网络（PNN）+双向LSTM/单向LSTM、VAE-LSTM、MLP、Bi‑LSTM以及自建的前馈动力学模型，整体构成闭环训练框架；

**📊 数据集**

利用机器人在仿真与真实平台的“motor babbling”数据（约14k条样本）以及三种手工轨迹（圆、矩形、螺旋）做训练与评测；

**📈 对比分析**

与MLP、LSTM、VAE‑LSTM、Bi‑LSTM四种基线对比，SMPL在位置误差和方向误差上均优于所有基线，尤其在多模块、分布式控制场景下表现显著；

**⚠️ 局限性**

主要局限是增量学习需要逐模块训练，导致训练时间增长；未来计划引入元学习加速适应并进一步提升对硬件退化的鲁棒性。

---

## 158. Geometric Collapse: When Vision Models Fail to Verify Physical Causality

**arXiv ID:** 2607.06871 | [PDF](https://arxiv.org/pdf/2607.06871v1)

**作者:** Wentao Zhang `[一作]` (Macao Polytechnic University), Irwin King `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并使用 Scrambled Edges 对视觉模型在推理时的物理一致性进行诊断，揭示了在强视觉边缘但缺乏物理支持时的 Geometric Collapse 现象。

**💡 创新点**

创新点在于构造一种对比性、能量匹配的边缘干预，通过破坏连续性、照明一致性和遮挡因果性来检测模型是否具有物理验证机制，首次表明大规模预训练并不能保证推理时的物理合理性。

**🔧 技术方法**

采用 Canny 边缘提取、仿射变换与暗化操作生成 Scrambled Edges，配合高通噪声与结构对照组；使用 RMSE、Collapse Ratio、Edge F1、oracle 级修复等指标对模型进行评估。

**📊 数据集**

主要使用 NYU Depth v2 与 KITTI Odometry 数据集；此外在附录中验证了在 Marigold、DepthFM 等生成式深度模型上的跨数据集泛化。

**📈 对比分析**

对比 CNN、ViT、SSL 以及生成式深度预测器，发现 Scrambled Edges 导致的误差比能量匹配噪声高 1.8–3.2 倍，结构化边缘指标明显下降，而传统全局 GT 指标往往掩盖此失效；生成式模型虽减弱但仍显著。

**⚠️ 局限性**

局限性包括：干预为人工合成的理想化实验，未覆盖所有视觉短路；只关注边缘到几何的路径，无法直接改进模型鲁棒性；在迭代或生成式推理中仍存在残余错误，需进一步设计支持感知与选择性融合机制。

---

## 159. Is Randomness Necessary for Adaptive Data Analysis?

**arXiv ID:** 2607.07085 | [PDF](https://arxiv.org/pdf/2607.07085v1)

**作者:** Edith Cohen `[一作]` (Google Research), Uri Stemmer `[通讯]` (Google Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了自适应数据分析（ADA）问题，证明了在随机预言机模型下，对计算上无界的分析者，确定性机制只能支持大约 O(n log N) 次自适应查询，表明随机性在 ADA 中是必要的。

**💡 创新点**

主要创新点在于给出了确定性机制的严格下界，展示了随机性与确定性之间的本质差距；提出了使用随机预言机、动态指针以及分离查询的全新攻击策略，并将结果推广到有限随机性的机制。

**🔧 技术方法**

采用了对抗性模拟、随机预言机模型、动态指针技术、阈值查询与分离查询构造，以及集合消减与概率论工具进行证明。

**📊 数据集**

实验设置使用均匀分布在域 [N]（N=n^10）上的样本数据集，主要是理论构造而非实际数据集。

**📈 对比分析**

与随机机制（可支持约 n² 次查询）相比，确定性机制在同一设置下只能支持约 O(n log N) 次查询，展示了显著的性能差距。

**⚠️ 局限性**

局限性包括依赖信息论随机预言机模型、对计算上无界分析者的假设、结果仅适用于均匀分布且域大小假设（N=n^10）以及未解决多样本相关性或小域下的情况。

---

## 160. Evaluating LLM Robustness Under Domain-Specific Prompt Perturbations in Public Health Applications

**arXiv ID:** 2607.06913 | [PDF](https://arxiv.org/pdf/2607.06913v1)

**作者:** Chuqing Zhao `[一作]` (Harvard University), Haochen Yang `[通讯]` (Harvard University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了针对公共健康应用的领域特定鲁棒性基准，评估LLM在误信息框架和普通话改写下的性能。

**💡 创新点**

首次将误信息注入与口语化重写作为两类真实场景扰动，揭示它们对LLM鲁棒性的不同影响。

**🔧 技术方法**

利用误信息注入、词汇替换扰动函数以及准确率、ΔAcc和翻转率等评估指标。

**📊 数据集**

使用PubMedQA、MedQA‑USMLE和COVID‑19 Vaccine Stance三个公开数据集。

**📈 对比分析**

对比了四个轻量级LLM（Llama‑3.1‑8B、Mistral‑7B、Qwen2.5‑7B、GPT‑4.1‑Nano），发现误信息导致平均-7.2pp下降，翻转率9–38%；普通话改写仅-1.4pp。

**⚠️ 局限性**

实验样本量有限（每集100例）、仅考虑单一误信息强度和CHV词表覆盖，缺乏更广泛扰动级别和模型族。

---

## 161. When Prompts Ignore Structure: Graph-Based Attribute Reasoning for Calibrated VLMs

**arXiv ID:** 2607.07395 | [PDF](https://arxiv.org/pdf/2607.07395v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 162. SHTA: Semantic Hard Token Correction and Center Alignment for Semi-Supervised Medical Image Segmentation

**arXiv ID:** 2607.07019 | [PDF](https://arxiv.org/pdf/2607.07019v1)

**作者:** Zhuoru Zhang `[一作]` (Xi'an Jiaotong-Liverpool University), Xiaofeng Liu `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一种仅在训练阶段使用的语义表示分支SHTA，用于在半监督医学图像分割中纠正难区块的语义分配并对类级语义中心进行对齐。

**💡 创新点**

创新点在于将语义一致性问题从预测层转到表示层，利用标注掩码生成的token级语义引导，对难区块进行后选择的语义纠正和类中心对齐，而不增加推理成本。

**🔧 技术方法**

主要技术包括：代理校准的语义分配、基于标注token分布的硬Token纠正以及聚合硬Token形成类中心的语义中心对齐，并将其作为辅助损失加入原有SSL框架的训练。

**📊 数据集**

实验使用Synapse（30例CT，13类）和AMOS（5%标注）两个医学分割数据集。

**📈 对比分析**

在CPS、URPC、GA‑CPS和MagicNet等四种代表性半监督框架上，SHTA均实现了平均Dice提升1.1%–2.3%和ASD下降1.4–2.1个像素，弱解剖结构（如食管、肾盂等）的恢复效果尤为显著。

**⚠️ 局限性**

局限性包括：对已强大基线的提升有限（如MagicNet提升仅0.1%），对ASD的改进不如Dice显著，且需额外训练时的计算和内存开销。

---

## 163. Hardware-aware Graph Neural Networks prunning for embedded event-based vision

**arXiv ID:** 2607.06739 | [PDF](https://arxiv.org/pdf/2607.06739v1)

**作者:** Piotr Wzorek `[一作]` (AGH University), Tomasz Kryjak `[通讯]` (AGH University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种两步图卷积（Two-Step Graph Convolution）方法，用于在FPGA上高效处理事件摄像机数据，从而降低LUT和DSP资源占用，支持更大规模GCNN模型。

**💡 创新点**

创新点在于：①利用BRAM缓存重复出现的输入特征，拆分卷积为两阶段，显著减少重复乘法；②通过在第一阶段仅计算自环乘法，第二阶段补齐位置差值，可把LUT占用降低50–94%；③针对不同资源瓶颈提供LUT、DSP、BRAM三种实现变体，提升模型可扩展性。

**🔧 技术方法**

技术包括：Graph Convolutional Neural Network（GCNN）、事件摄像机事件流预处理、FPGA SoC实现、BRAM双端口缓冲、DSP乘法、LUT乘法、两步卷积流水线设计。

**📊 数据集**

使用了 N-Caltech 数据集进行实验验证。

**📈 对比分析**

与Baseline和DSP-conv实现对比：LUT资源下降70–78%，DSP资源下降64–71%；两步实现对分类准确率影响不超过±0.3%；在ZCU104 200 MHz FPGA上实现，未出现时延或时序问题。

**⚠️ 局限性**

limitations: 仅在N-Caltech数据集和所提出的GCNN结构上验证；对更大、更复杂数据集或任务（如目标检测）的效果尚未评估；两步方案虽然降低逻辑占用但仍需额外BRAM，且DSP资源仍可能成为瓶颈。

---

## 164. STAGformer: A Spatio-temporal Agent Graph Transformer for Micro Mobility Demand Forecasting

**arXiv ID:** 2607.06614 | [PDF](https://arxiv.org/pdf/2607.06614v1)

**作者:** Ye Zihao `[一作]` `[通讯]` (City University of Hong Kong), Ye Zihao (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 STAGformer，一种用于自行车共享站点需求预测的时空图Transformer模型，能够在大规模城市网络中实现高精度预测。

**💡 创新点**

核心创新是引入可学习的空间与时间代理令牌的两步代理注意机制，将自注意力复杂度从 𝒪((NT)^2) 降至 𝒪(NT)，同时保持全局软最大注意力的表达能力，并与图传播、时序卷积以及多源外部特征融合相结合。

**🔧 技术方法**

主要技术包括图神经网络（Graph Propagation）、一维卷积（Temporal Convolution）、Transformer 结构、代理注意机制（Agent Attention）、外部特征编码与位置编码。

**📊 数据集**

实验数据集为纽约市 Citi‑Bike 和芝加哥市 Divvy‑Bike 的真实运营记录，并加入天气、POI 与道路网络信息。

**📈 对比分析**

与 9 种基准（线性回归、空间回归、GRU、Transformer、GAT、BikeMAN、STAEformer、T‑STAR、BGM）比较，STAGformer 在 RMSE 与 MAE 上均取得最佳或次佳成绩，显著优于传统模型，且参数量与 FLOPs 较低。

**⚠️ 局限性**

局限性包括：依赖固定的道路网络图结构，未采用动态图学习；代理令牌数量对性能敏感，需经验调参；目前仅验证于自行车共享场景，未对电动滑板车等其他微出行形式或实时事件数据进行推广。

---

## 165. -8 dB SNR + 90% Packet Loss: MamVSC -- CSI-Guided Semantic Mamba for Extreme-Robust Video Semantic Communication

**arXiv ID:** 2607.07293 | [PDF](https://arxiv.org/pdf/2607.07293v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 166. When and How Should a Power Trader Engage in Arbitrage? Predict, then Contextually Optimize

**arXiv ID:** 2607.07351 | [PDF](https://arxiv.org/pdf/2607.07351v1)

**作者:** Yannick Heiser `[一作]` (Technical University of Denmark), Farzaneh Pourahmadi `[通讯]` (Technical University of Denmark)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了价格接收的随机能源生成者在单价平衡市场下如何进行日间与平衡市场的套利交易，并提出一种预测-再优化框架。

**💡 创新点**

将套利决策拆解为三阶段：何时参与、方向、幅度，利用置信阈值实现可解释的风险控制，并为混合电站学习两类线性决策策略。

**🔧 技术方法**

采用LightGBM概率二分类、阈值判定、上下文优化学习线性策略（含CVaR约束）以及滚动窗口评估。

**📊 数据集**

使用DK1和DE/LU两欧盟投标区的实际风电场与氢电解槽数据（风速、价格、需求等），覆盖2025年4月至2026年2月，特征数200+。

**📈 对比分析**

与历史最优、预测套利、单策略、分类+全部决策等六种基准在22个滚动窗口中比较，结果显示对混合电站平均利润提升约7%（相比套利自由），在低分布漂移窗口表现最好，CVaR表现更稳健。

**⚠️ 局限性**

对非平稳市场分布漂移敏感，未考虑在线学习与市场层面影响；仅为单一价格参与者视角。

---

## 167. Mechanistic Interpretability for Neural Networks: Circuits, Sparse Features and Symbolic Reasoning

**arXiv ID:** 2607.07316 | [PDF](https://arxiv.org/pdf/2607.07316v1)

**作者:** Pranav Sawant `[一作]` (University of Texas at Dallas), Jakub Krejčí `[通讯]` (VSB Technical University of Ostrava)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述并评估了Transformer类大模型的机理可解释性技术，涵盖电路分析、稀疏自编码器、自动电路发现、转码器、引导向量、以及神经符号规则抽取。

**💡 创新点**

提出“通用性假设”以及跨层转码器、稀疏自动编码器驱动的主动干预框架，强调可解释性可设计与模型可控性相结合的潜在路径。

**🔧 技术方法**

利用TransformerLens、ACDC、LAT、SAE、转码器、引导向量、NeSyFOLD、NeSyViT等多种工具与算法进行电路拆解、特征稀疏化、因果干预与规则提取。

**📊 数据集**

采用公开的Transformer、ViT、LLM模型（如GPT‑2、LLama、Claude、Gemma‑2等）以及合成任务（Induction Heads、Indirect Object Identification、Greater‑Than）作为分析对象；数据来源包括预训练权重、公开数据集和自制合成句子。

**📈 对比分析**

对比方法主要从计算效率、可解释性程度、因果验证和自动化程度四个维度进行定性评估：ACDC因果补丁精度高但成本高；LAT、Edge Attribution Patching速度快但解释更粗糙；SAE可显著提高特征可解释性但在大模型上需数十亿特征。

**⚠️ 局限性**

局限性包括：大部分案例仅在小模型或合成任务上验证，缺乏对前沿规模模型的系统评估；自动化电路发现方法仍受计算资源限制；稀疏自编码器与转码器在解释完整变换时可能存在机制偏差；引导向量的稳定性和通用性尚未得到充分验证。

---

## 168. Seekable OCI: Lazy-Loading Container Images via Range-Request Indexing

**arXiv ID:** 2607.06868 | [PDF](https://arxiv.org/pdf/2607.06868v1)

**作者:** James Thompson `[一作]` (Amazon Web Services), Henry Wang `[通讯]` (Amazon Web Services)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 SOCI，一种针对 OCI 镜像的惰性加载架构，显著减少容器的冷启动时间。

**💡 创新点**

创新点在于使用外部可寻址索引（ztoc）并存为 OCI Referrer，使得在不修改镜像或注册中心的前提下即可实现按需下载。

**🔧 技术方法**

核心技术包括基于 zlib 的 DEFLATE 块寻址索引、FUSE 文件系统以及 HTTP Range 请求与 OCI Referrer API 的集成。

**📊 数据集**

实验数据集涵盖多种常见镜像（ubuntu、redis、python、nginx、postgres、golang、node、python-flask、flask-bloated）及一个 1GB 的合成镜像。

**📈 对比分析**

通过与标准 containerd 拉取进行对比，冷启动时间从 20–25 秒降至约 2.8 秒，实现 7.4–9.3 倍加速；在 Fargate 上亦可获得约 4 倍提升，且在访问密度低于 80% 时表现最佳。

**⚠️ 局限性**

局限性包括首次文件访问的 HTTP 连接延迟、仅支持 gzip 压缩层、对高访问密度场景可能产生更多请求成本，以及在某些工作负载下 FUSE 可能带来的 I/O 开销。

---

## 169. PRoVeFL: Private Robust and Verifiable Aggregation in Federated Learning

**arXiv ID:** 2607.06612 | [PDF](https://arxiv.org/pdf/2607.06612v1)

**作者:** Harsh Kasyap `[一作]` (Indian Institute of Technology (BHU)), Carsten Maple `[通讯]` (University of Warwick)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多服务器、可验证、鲁棒且隐私保护的联邦学习框架，能够在保证客户端更新隐私的前提下实现 Byzan‑tine 鲁棒聚合并对聚合过程进行可验证。

**💡 创新点**

在多键同态加密基础上引入随机乘法掩码和可验证承诺，允许在加密域完成复杂的统计聚合，随后安全地转移至明文域完成高成本运算，从而实现高效、可验证且支持多种鲁棒聚合规则的联邦学习。

**🔧 技术方法**

使用多键同态加密（Multi‑Key FHE）进行加密与同态运算；引入随机乘法掩码与分片共享；利用承诺与对偶映射实现聚合验证；在部分计算后转移至明文域；支持 Krum、Trimmed‑Mean、FLTrust、MESAS 等鲁棒聚合。

**📊 数据集**

在 CIFAR‑10（LeNet‑5、ResNet‑18）和 Shakespeare LSTM 等标准数据集上进行实验；使用 10、50、100、200 名客户端和 2、4、10 名服务器进行规模化评估。

**📈 对比分析**

与 ELSA、Prio、RoFL 等现有安全 FL 系统对比，实验显示在相同模型规模下该框架在运行时可比 Prio 低 100 倍、ELSA 低 10 倍，并且在通信量与服务器负载上均优于 RoFL；鲁棒聚合在面对 Trim 与 Backdoor 攻击时保持与明文实现相同的精度。

**⚠️ 局限性**

只能处理在乘法掩码下保持相对顺序或大小关系的鲁棒规则，无法直接支持基于自适应阈值或非线性变换（如 Min‑Max、Min‑Sum）的聚合方法。

---

## 170. Simple Nash Equilibria for Qualitative Multiplayer Games

**arXiv ID:** 2607.07151 | [PDF](https://arxiv.org/pdf/2607.07151v1)

**作者:** Mona Alluwaym `[一作]` (University of Liverpool), Sven Schewe `[通讯]` (University of Liverpool)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种多玩家无随机化策略的子博弈完美均衡（SPE）构造方法，用于在转向式确定性游戏中，玩家目标为到达、安全或0-2 Muller's类（上/下闭合）ω-正则目标。

**💡 创新点**

证明了在该目标类下，随机化策略是必需的（即存在无法用无记忆策略实现的情况），并给出了一个多步骤的递归算法，既能构造出记忆无关的随机化SPE，又在多种特殊情形（目标/不安全点为吸收状态）下能够得到纯记忆无关SPE，首次在此类非零和游戏中实现了SPE的显式构造与存在性证明。

**🔧 技术方法**

利用叶子强连通分量（leaf SCC）的随机混合策略、对已评估顶点的边删减、以及对成功率与安全性判定的集合推理，构建了一系列规则（选择已评估成功点、删除失败点、均匀随机）来逐步扩展策略，最终保证所有玩家在任何顶点上的胜率为0或1，从而满足SPE条件。

**📊 数据集**

该工作为理论研究，未使用具体数据集；所有结果均通过形式化证明与算法分析给出。

**📈 对比分析**

方法的复杂度为多项式（O(|V|·|E|)），与现有针对零和游戏或随机博弈的SPE构造方法相比，无需依赖固定点定理或折扣化近似，直接给出显式算法；虽然未给出实验评估，但理论上已证明可在多项式时间内完成构造。

**⚠️ 局限性**

局限性：仅适用于目标为到达、可安全或0-2 Muller类的游戏；对于1-3 Muller目标、仅有到达或安全目标的游戏，无法保证记忆无关均衡；此外，在需要随机化的叶子SCC中仍需使用随机混合策略，纯策略构造仅在目标/不安全点为吸收状态时可行。

---

## 171. Video2Reaction: Mapping Video to Audience Reaction Distribution in the Wild

**arXiv ID:** 2607.06875 | [PDF](https://arxiv.org/pdf/2607.06875v1)

**作者:** Trang Nguyen `[一作]` (University of Massachusetts Amherst), Madalina Fiterau `[通讯]` (University of Massachusetts Amherst)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建并发布了首个面向电影内容的观众情绪分布数据集Video2Reaction，并构建了可扩展的两阶段LLM标注管线；

**💡 创新点**

创新点包括：①将观众情绪视为标签分布学习问题，首次提供大规模、真实世界的情绪分布；②利用多代理LLM实现高效、可更新的情绪标注；③设计双轴评估框架（完整分布预测与主导情绪预测）及多种评价指标；

**🔧 技术方法**

使用的技术包括：多模态特征提取（视觉、音频、文本）、两阶段LLM标注（重述+情绪抽取）、基础VLM（Gemini、LLaVA、Qwen）低秩微调、传统LDL方法（PT、SA、AA）以及评估指标（Chebyshev、KL、Cosine、MRR等）；

**📊 数据集**

所用数据集为Video2Reaction，包含10,348段电影剪辑（约398小时）、约800,000条YouTube评论，构成21类细粒度情绪分布；

**📈 对比分析**

与传统零射VLM、经典LDL方法对比，零射VLM性能差（Cosine<0.51，Top‑1 F1<0.30），低秩微调后VLM达到Top‑3 F1≈0.77、MRR>0.75；LDL方法在分布度量上竞争力强，但在主导情绪预测上低于微调VLM；

**⚠️ 局限性**

局限性包括：仅使用YouTube评论，缺乏跨平台和人口统计信息；情绪分布长尾不平衡，罕见情绪预测困难；需要更系统的跨域、跨分类器迁移研究。

---

## 172. Does AI Understand Imaging? A Systematic Benchmark of Agentic AI for Computational Imaging Tasks

**arXiv ID:** 2607.07189 | [PDF](https://arxiv.org/pdf/2607.07189v1)

**作者:** Ethan Chung `[一作]` (University of Hawaii at Manoa), Huaijin Chen `[通讯]` (University of Hawaii at Manoa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了 ImagingBench，一个统一评估代理式 AI 在计算成像任务（包括逆重建、传感、光学、校准等）中的基准。

**💡 创新点**

其创新点在于首次构建覆盖 20 个子任务、5 个类别的多模态评测框架，并设计了 Expert、Planner、Forward 三种评估模式，以分离执行、规划和前向一致性能力。

**🔧 技术方法**

实验中使用 Gemini、GPT、Qwen 等前沿多模态大模型作为代理，并结合基于前向模型的计划器、编辑器以及 PSNR、SSIM、LPIPS、NIQE 等多指标评价与归一化汇总技术。

**📊 数据集**

数据集方面结合公开数据（如 SIDD、HDR+、MIT‑CGH‑4K、DIV2K、Adobe FiveK 等）与基于正向物理模型合成的测量样本，同时包含光学设计与相机标定数据。

**📈 对比分析**

通过与专用非代理方法（如 AutoLens、专业去噪/重建模型）对比，发现代理模型在 Expert 及 Planner 模式下整体表现逊于专业方法，尤其在计算传感任务上差距显著；Planner 对性能提升有限，Gemini 在代理族中表现最好。

**⚠️ 局限性**

局限性包括缺乏对物理前向模型的深度理解，易产生幻觉且对逆问题（如相位恢复、稀疏重建）表现不佳；Planner 仅带来微小提升；部分模型受安全过滤阻断；基准仅涵盖图像中心的代理流程，未覆盖更复杂的工具使用场景。

---

## 173. Evaluating Endpoint Detection Robustness Against Genetic Algorithm Driven Code Transformations

**arXiv ID:** 2607.07191 | [PDF](https://arxiv.org/pdf/2607.07191v1)

**作者:** Alvina Rwaichi Minja `[一作]` (Carnegie Mellon University Africa), Jema David Ndibwile `[通讯]` (Carnegie Mellon University Africa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了ShellForge，一个基于遗传算法的框架，用于演化功能等价的后期攻陷（reverse‑shell）变体，评估防病毒和终端检测的鲁棒性。

**💡 创新点**

创新点在于：①统一优化静态和行为两类变形策略；②通过自适应搜索自动发现有效的变形序列（如XOR+Base64）而非预设模板；③在轻量级Python payload上实现快速收敛和可复现的评估流程。

**🔧 技术方法**

使用技术包括：Python实现的遗传算法（DEAP），多目标适应度函数（结合AV检测、沙箱行为、功能验证和效率惩罚），以及一系列语法/语义保持的变形算子（XOR、Base64、字符串拆分、变量重命名等）。

**📊 数据集**

数据集为：基准Python reverse‑shell模板；评估使用的检测引擎包括VirusTotal（62款）、ClamAV、Windows Defender、CAPE/Cuckoo Sandbox；实验在虚拟化的隔离环境中进行。

**📈 对比分析**

通过与MsfVenom、Veil‑Evasion、TheFatRat等传统模板生成框架在相同检测环境下对比，ShellForge在VirusTotal检测中从44/62降至0/62，同时保持100%功能成功率；遗传算法在第5代即可收敛，演化效率高。

**⚠️ 局限性**

局限性包括：实验仅涵盖Python reverse‑shell，未覆盖多语言或多种后期攻陷行为；检测仅限于有限的AV和沙箱引擎，结果可能在企业环境中不同；实验环境为受控虚拟机，缺乏真实网络和多样化攻击手法的验证。

---

## 174. Flowcode: An AI-Powered Programming Environment for Scaffolding Iteration in Creative Computing Education

**arXiv ID:** 2607.06721 | [PDF](https://arxiv.org/pdf/2607.06721v1)

**作者:** Tiffany Tseng `[一作]` (Columbia University), Arya Sinha `[通讯]` (Columbia University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并实现了 Flowcode——一款集成 AI 辅助的创意编码环境，帮助学习者通过可视化流程图和分步填空式代码解释来理解和迭代现有创意编码项目。

**💡 创新点**

创新点在于：①将 LLM 生成的流程图与代码片段关联，直观展示跨文件（HTML/CSS/JS）的结构关系；②通过逐步展开的解释与填空式代码片段，增加“摩擦”阻止直接复制粘贴，促进主动编码学习；③在交互设计中加入手动更新流程图的提示，避免自动增大图表导致可读性下降。

**🔧 技术方法**

主要技术包括：使用 GPT‑4o 作为核心 LLM，生成 JSON 流程图结构与解释；React Flow 实现可视化流程图；Node.js 前端实现多文件编辑与实时预览；定制化 Prompt 以控制填空与逐步展示。

**📊 数据集**

数据来源：收集自两轮用户研究的实验数据——七位初学者的工作坊日志、九位参与者的实验录像与交互日志，以及从 CodePen 提取的 4 个 Anime.js 动画示例（并未公开大规模数据集）。

**📈 对比分析**

比较方法主要为定性主题分析与交互日志统计；结果显示大部分用户（≈89%）利用流程图进行导航与理解，平均每人使用 LLM 9 次；用户反馈表明分步解释与填空机制显著提升了对代码的理解，虽未给出客观性能指标，但定性评价表明学习体验更佳。

**⚠️ 局限性**

局限性包括：①仅使用 GPT‑4o，无法泛化到其他 LLM；②实验仅围绕单一项目类型（Anime.js 动画）且时间短（40 分钟），未检验长期使用效果；③自动更新流程图与代码导致图表膨胀的问题虽已改进，但仍需进一步评估在更大项目中的可扩展性。

---

## 175. Geometric Self-Distillation for Reasoning Generalization

**arXiv ID:** 2607.06855 | [PDF](https://arxiv.org/pdf/2607.06855v1)

**作者:** Josip Jukić `[一作]` (University of Amsterdam), Ivan Titov `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型的后训练阶段，通过让模型在自身生成的轨迹上进行自蒸馏，利用有特权上下文的教师来给学生提供密集的 token 级监督。

**💡 创新点**

提出几何自蒸馏目标，利用 Hellinger 距离按教师–学生重叠加权拉力，并用 Fisher–Rao 距离对累计漂移做近端正则，配合自然梯度更新，从而在保持 ID 性能的同时显著提升 OOD 推理。

**🔧 技术方法**

几何信息量学（Hellinger、Fisher–Rao）、自然梯度、K‑FAC 近似、对齐稀疏梯度、在 LLM 上的自蒸馏与对齐。

**📊 数据集**

数学推理数据集：训练使用 DAPO‑Math‑17k；评估 ID 采用其内部划分；OOD 采用 AIME 2024/25、AMC 2023、MATH‑500。

**📈 对比分析**

与基准模型、SFT、GRPO、KL、JSD、SkewKL、TrOPD、TIP 等方法对比，平均在 ID 保持不变的前提下，OOD avg@16 提升 5.7–8.6 分，在 1.7B–32B 所有模型规模上均取得最优表现。

**⚠️ 局限性**

依赖于有特权上下文且需要大量 roll‑out，近端正则与自然梯度实现较为复杂；实验仅覆盖数学推理任务，未知对其他推理或生成任务的适用性。

---

## 176. Vectorizing Quantum Control: A RISC-V Vector Extension Architecture for Scalable Qubit Systems

**arXiv ID:** 2607.07372 | [PDF](https://arxiv.org/pdf/2607.07372v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 177. Exploring Serendipity in Information Seeking for Digital Collections: A Mixed-Methods Survey Study toward Human-Centered Design

**arXiv ID:** 2607.06937 | [PDF](https://arxiv.org/pdf/2607.06937v1)

**作者:** Saumik Shashwat `[一作]` (University of Washington), Benjamin Charles Germain Lee `[通讯]` (University of Washington)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过混合方法在线问卷，研究了数字藏品信息寻求过程中的机遇性（serendipity）体验与感知，分析了系统设计与用户行为之间的关系，并探讨了AI辅助机遇性功能的用户态度。

**💡 创新点**

创新点在于：①提出并量化Serendipitous Digital Environment与Perception of Serendipity两种新量表，系统化评估数字藏品环境与机遇性感知；②以人本设计视角对数字藏品机遇性进行多维度分析；③首次探讨AI辅助机遇性在数字藏品中的应用与用户接受度。

**🔧 技术方法**

使用技术包括：在线问卷设计与收集（Google Forms）；定量分析（SPSS 统计、t检验、Mann‑Whitney、Spearman 相关、线性/多元回归、ordinal/multinomial logistic 回归）；定性分析（主题编码）。

**📊 数据集**

数据集为30名研究人员（年龄19‑68岁）对其主要使用的数字藏品（如British Newspaper Archive、NASA SVS、National Archives等）完成的问卷数据。

**📈 对比分析**

比较方法：对比“特定目标（SG）”与“根据情境双重目标（BDS）”两组在各量表上的差异；通过相关与回归检验量表间关系，模型解释了约60.6%的机遇性感知变异；结果显示BDS组机遇性感知显著高于SG组，且系统提供意外互动机会与连接显著预测机遇性感知。

**⚠️ 局限性**

局限性包括：样本量有限、便利抽样导致代表性不足；主要依赖自我报告，缺乏行为轨迹数据；未深入分析不同文化/专业背景对机遇性的影响；AI机遇性功能的态度分析仍属初步，缺乏定量验证。

---

## 178. The Power of Backdoor Absorption in Community Training

**arXiv ID:** 2607.06643 | [PDF](https://arxiv.org/pdf/2607.06643v1)

**作者:** Issam Seddik `[一作]` (Université Paris-Saclay), Sara Tucci Piergiovanni `[通讯]` (Université Paris-Saclay)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在社区训练环境下，结合天然后门吸收与稀疏惰性验证，构建了可抑制后门攻击的防御框架。

**💡 创新点**

创新点在于将后门吸收现象形式化为时间非齐次马尔可夫链，并给出攻击成功率可渐近为零的闭式期望时间界限，同时证明仅需10%验证即可完成防御。

**🔧 技术方法**

使用的技术包括：离散时间马尔可夫链分析、动态调度与惰性验证机制、数值仿真以及基于 ResNet‑18/CIFAR‑10 的实验评估。

**📊 数据集**

实验使用的数据集为 CIFAR‑10，并在 ResNet‑18 模型上进行训练和后门注入。

**📈 对比分析**

通过与单训练、仅吸收以及吸收+10%惰性验证三种配置对比，实验结果表明后门成功率从≈100%降至≈7%，模型准确率维持≈80%，验证开销仅提升约2–3%。

**⚠️ 局限性**

局限性在于假设存在中心化的验证主机，需要在完全去中心化的环境中实现共识；对极端高比例恶意节点的抵御能力有限；且对更大规模模型的适用性尚未全面验证。

---

## 179. zk-ScalHard: Scalable and Hardware-Rooted Privacy-Preserving Authentication for Secure OTA Updates in Zonal SDVs

**arXiv ID:** 2607.07371 | [PDF](https://arxiv.org/pdf/2607.07371v1)

**作者:** Shrikant Tangade `[一作]` (Inria Lille Nord Europe), Mauro Conti `[通讯]` (University of Padua)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一种基于零知识证明与硅PUF的zk-ScalHard协议，用于软件定义车辆在Zonal架构下的安全OTA更新认证；

**💡 创新点**

创新点在于将零知识证明与硬件根信任结合，采用分层信任推广、递归聚合与时间隔离，实现O(1)通信与验证复杂度，同时实现车辆数据主权与GDPR合规；

**🔧 技术方法**

使用Groth16、Plonky3 SNARK、Poseidon哈希、零知识电路ZIDI与HPCA、MPC、硅PUF等技术；

**📊 数据集**

使用的实验数据集为100 ECU的Zonal SDV模拟环境，采用SIL在Ubuntu/WSL2上运行；

**📈 对比分析**

通过与行业标准Uptane基准对比，zk-ScalHard在通信量下降99.2%、验证延迟提升7.3倍、攻击面缩小99.9%，实现O(1)可扩展性；

**⚠️ 局限性**

局限在于实现仍处于软件仿真阶段，未在真实车载硬件或HiL测试中验证性能与安全性。

---

## 180. Converge to Surprise: Evolutionary Self-supervised Image Clustering

**arXiv ID:** 2607.06887 | [PDF](https://arxiv.org/pdf/2607.06887v1)

**作者:** Canlin Zhang `[一作]`, Xiuwen Liu `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种自监督图像聚类框架——Converge-to-Surprise，利用“最大熵”假设构造两份互补蒙版视图，定义惊奇得分并通过进化策略+梯度下降的双循环优化模型，最终实现无标签、无先验、无预训练的硬聚类。

**💡 创新点**

核心创新在于：① 引入“惊奇得分”来衡量模型对随机噪声假设的拒绝程度，证明惊奇得分无法化为一步步的损失；② 设计基于进化策略的外循环直接最大化惊奇得分，同时利用已发现的惊奇聚类作为代理标签在内循环进行梯度下降，从而在缺乏即时目标的情况下训练深度网络。

**🔧 技术方法**

技术手段包括：棋盘式蒙版生成两份互补视图；随机数据增强（旋转、裁剪、对比度等）保持零互信息；利用交叉熵做代理标签训练；进化策略（ES）实现全局搜索；K均值聚类+Hungarian算法用于评估；ResNet-9网络作为特征提取器。

**📊 数据集**

在MNIST、Fashion‑MNIST和USPS这三类手写/服装图像数据集（每个10个真类别）上从零开始训练，不依赖任何预训练模型或标签。

**📈 对比分析**

与DeepDPM、UNSEEN、DBSCAN、moVB等基线在无参数聚类任务中对比，实验表明在MNIST、Fashion‑MNIST和USPS上均取得最高或相近的NMI/ARI/ACC，尤其在USPS上ACC提升约5%（从89%到95%），表明方法在严格无标签聚类下具有显著优势。

**⚠️ 局限性**

局限性包括：① 需要大量进化策略计算资源和长时间训练；② 惊奇得分仅在特定假设下定义，对其他类型数据的适用性尚未验证；③ 对超参数（如阈值τ、批量大小N）敏感；④ 目前仅在简单的手写/服装数据集上验证，尚未在更复杂的自然图像数据集上证明可扩展性。

---

## 181. ReMoDEx: A Local-to-Global Relevance-Based Model Decision Explainability Framework for large-Scale Image Datasets

**arXiv ID:** 2607.06889 | [PDF](https://arxiv.org/pdf/2607.06889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 182. TF-Engram: A Train-Free Engram with SSD-Backed Memory for Large Language Models

**arXiv ID:** 2607.07388 | [PDF](https://arxiv.org/pdf/2607.07388v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 183. TACoS: Weakly Supervised Learning of Two-Dimensional Materials from Scribble Annotations to Precise Segmentation

**arXiv ID:** 2607.07169 | [PDF](https://arxiv.org/pdf/2607.07169v1)

**作者:** Jiabei Chen `[一作]` (Institute of Semiconductors, Chinese Academy of Sciences), Xin Ning `[通讯]` (Institute of Semiconductors, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了TACoS框架，利用稀疏涂笔标注实现二维材料显微图像的像素级分割。

**💡 创新点**

创新点在于三项核心模块：1）弱-强分布对齐（UWSD）实现无监督区域的预测一致性；2）树能量正则化（TER）通过最小生成树构建结构化软标签；3）非对称区域对比学习（ARCL）在特征空间和决策空间对边界区域进行强化，从而在稀疏监督下兼顾区域一致性与边界精度。

**🔧 技术方法**

采用了DINOv2作为共享编码器、DPT作为解码头；结合自监督对比学习、最小生成树滤波、异向量对比损失以及稀疏交叉熵等技术；训练中使用弱/强数据增强、EMA或UniMatch V2风格对齐。

**📊 数据集**

使用了两大公开数据集：Yan（Graphene、MoS₂）和Uslu（Graphene、WSe₂），在两者上都生成了低于0.6%像素覆盖的涂笔标注。

**📈 对比分析**

与多种基线（URSS、CC4S、A²GNN、SASFormer等）对比，TACoS在Graphene上mIoU达84.09%、边界IoU 66.95%；在MoS₂上mIoU 85.20%、边界IoU 62.43%；仅比全监督方法低2–3%，并在稀疏标注下明显优于现有单阶段和多阶段方案。

**⚠️ 局限性**

局限性包括：在光学对比极低或背景纹理极其复杂时仍易产生误检；ARCL依赖清晰边界，若背景物质与薄膜呈现相似对比时仍可能出现混淆；未针对多类别或重叠异质结构的实例分割，未来可进一步扩展。

---

## 184. Navigating Hierarchy: Hyperbolic Learning on Brain Graphs for Disorder Diagnosis

**arXiv ID:** 2607.07077 | [PDF](https://arxiv.org/pdf/2607.07077v1)

**作者:** Yapeng Li `[一作]` (Anhui University), Zhengzheng Tu `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出并实现了一种名为HLBG的超曲空间脑网络学习框架，用于脑疾病（自闭症和抑郁症）的诊断和生物标志物发现。

**💡 创新点**

创新点包括：① 在黎曼超曲空间中显式建模ROI–社区–全脑的层级关系并通过两层蕴含约束保持层级一致性；② 设计Graph-aware Mamba，将图注意力结构提示注入Mamba实现长程依赖与拓扑信息的统一；③ 结合自注意力和子图注意力实现全局与局部特征的自适应融合。

**🔧 技术方法**

采用了Lorentzian超曲空间学习、Graph Attention、Mamba状态空间模型、随机游走位置编码、子图注意力、自注意力融合以及交叉熵+层级蕴含损失等技术。

**📊 数据集**

使用了ABIDE-I（自闭症）和REST‑MDD（抑郁症）两套fMRI数据集进行实验。

**📈 对比分析**

与多种GNN、Graph Transformer和Mamba基线在10折交叉验证下进行对比，HLBG在ACC、SEN、SPE等指标上均取得最高分（如ABIDE-I ACC≈75.45%，SEN≈79.07%，SPE≈71.36%；REST‑MDD ACC≈70.13%，SEN≈73.55%，SPE≈65.79%）。

**⚠️ 局限性**

局限性包括：① 需要先验的社区划分，划分不当可能影响层级学习；② 超曲空间的曲率和蕴含权重需手工调参；③ 模型复杂度较高，训练成本和可解释性仍有提升空间。

---

## 185. Audio Sentiment Analysis via Distillation and Cross-Modal Integration of Generated Multilingual Transcripts

**arXiv ID:** 2607.06611 | [PDF](https://arxiv.org/pdf/2607.06611v1)

**作者:** Andrei-George Durdun `[一作]` (University of Bucharest), Radu Tudor Ionescu `[通讯]` (University of Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于知识蒸馏的多模态情感极性识别框架，先使用ASR生成语音文本并通过NMT翻译成多种语言，再通过级联交叉模态Transformer融合音频与多模态文本特征，随后将多模态教师模型的知识蒸馏到仅使用音频的学生模型中；

**💡 创新点**

创新点包括：①利用自动生成的转录文本与多语言翻译作为训练时的特权信息；②设计级联交叉模态Transformer实现音频与多模态文本的分层融合；③在学习使用特权信息(LUPI)框架下，将多模态知识蒸馏到高效的音频单模态模型；

**🔧 技术方法**

使用了FastWhisper做ASR、NLLB-200做NMT、WavLM作为音频特征提取器、RoBERTa/ RoBERTuito/ GBERT/ CamemBERT作为多语言文本编码器、LoRA进行参数高效微调、级联交叉模态Transformer（CCMT）进行特征融合，以及温度蒸馏与权重λ平衡的知识蒸馏损失；

**📊 数据集**

在MSP‑Podcast语料库上进行实验，将情感标签映射为正、负、中性三类进行极性分类；

**📈 对比分析**

与单模态基线WavLM和Whisper对比，级联交叉模态教师模型在宏观F1上提升约+5.89%、准确率提升约+5.15%；通过蒸馏得到的音频单模态学生模型在宏观F1上提升约+1.54%、准确率提升约+0.81%，且推理速度与单模态基线相同；

**⚠️ 局限性**

局限性包括：依赖ASR和NMT产生的转录与翻译质量；训练时需要额外计算和存储；多模态教师模型在推理时显著慢于单模态模型；实验仅在MSP‑Podcast上验证，未评估跨数据集泛化能力；

---

## 186. Deployment Risk Assessment Using Diff-Aware Features: A Case Study at Prime Video

**arXiv ID:** 2607.06766 | [PDF](https://arxiv.org/pdf/2607.06766v1)

**作者:** Mayur Kurup `[一作]` (Amazon.com), Yegor Silyutin `[通讯]` (Amazon.com)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了基于差分特征的代码变更风险评估框架，帮助 Prime Video 在直播事件中精准判断变更风险。

**💡 创新点**

首次系统性识别并使用 diff‑aware 定量与定性指标（如代码结构复杂度、变更类型、编码规范违规）进行风险预测，且采用 LLM 进行多语言特征提取，避免语言工具维护与隐私泄露。

**🔧 技术方法**

结合大语言模型（Claude Sonnet）提取差分特征、静态分析器检测风格违规、XGBoost/RandomForest 进行分类，并对模型阈值进行高召回调优。

**📊 数据集**

内部 Prime Video 生产环境 CoE 关联的 149 个危险提交与 2,831 个安全提交；外部公开 ApacheJIT 数据集 1,115 次缺陷引入和 21,185 次干净提交。

**📈 对比分析**

采用 10 折交叉验证，XGBoost 在 Prime Video 上 F1≈0.846、召回≈0.788；在 ApacheJIT 上 F1≈0.771、召回≈0.875；相比逻辑回归或 LLM 零/少量示例分类，性能显著提升。

**⚠️ 局限性**

依赖差分特征忽略开发者历史和组织元数据，LLM 提取存在一定噪声，模型在不同时间划分、语言多样性和更大规模数据上的泛化仍待验证。

---

## 187. Deep Reinforcement Learning for Reliability Based Bi-Objective Portfolio Optimization

**arXiv ID:** 2607.06610 | [PDF](https://arxiv.org/pdf/2607.06610v1)

**作者:** Sounaq Das `[一作]` (Indian Institute of Management Amritsar), Aditya Gupta `[通讯]` (McKinsey and Company)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于深度强化学习的双目标可靠性投资组合优化框架（MORP-DRL），同时优化收益与尾部风险（方差、CVaR、EVaR），并纳入交易成本与可靠性约束。

**💡 创新点**

将可靠性约束、交易成本、三种尾部风险度量与DRL相结合，并使用QMC+ t‑copula+GARCH(1,1)+EVT生成高保真市场情景，构建连续动作PPO策略，形成新的可靠性驱动DRL框架。

**🔧 技术方法**

PPO强化学习、GARCH(1,1)波动率建模、极值理论+ t‑copula 依赖、Quasi‑Monte‑Carlo 情景生成、三种风险度量（方差、CVaR、EVaR）以及可靠性估计函数。

**📊 数据集**

十大全球股指（日收益）及 FTSE100 成分股的历史日收盘价，时间区间 2018‑2023，划分为 Pre‑COVID、COVID、Post‑COVID 三个市场阶段。

**📈 对比分析**

与等权基准及 NSGA‑II 多目标优化进行对比，评估回报、波动、Sharpe、各风险度量等指标；实验表明 PPO 在 CVaR/EVaR 框架下能在市场动荡期获得更高收益或更低尾部风险，而 NSGA‑II 在计算效率和集中配置上表现更佳。

**⚠️ 局限性**

PPO 训练耗时显著高于 NSGA‑II；交易成本模型仅为简化比例，未考虑流动性、价格冲击等进一步摩擦；实验仅涵盖股票/指数，未扩展至多资产类别。

---

## 188. Prototype-Anchored Generalized Manifold Regression for Unknown-Domain Object Detection

**arXiv ID:** 2607.07192 | [PDF](https://arxiv.org/pdf/2607.07192v1)

**作者:** Zihao Zhang `[一作]` (Tianjin University), Yahong Han `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对单源泛化目标检测任务，提出了基于流形回归的 MR‑DCoT 方法，将泛化视为将离流形样本回归到源语义流形的过程。

**💡 创新点**

创新点：① 通过视觉‑文本双链式思维（Dual‑CoT）同时产生全局语义演化和局部结构扰动，生成结构化、难度高的离流形样本；② 引入类别特定原型锚定的回归机制，使模型能学习到将偏移特征回归到类原型邻域的几何校正规则；③ 将模拟与回归闭环，弥补传统基于有限数据增强的覆盖不足。

**🔧 技术方法**

核心技术包括：视觉‑文本双链式思维（使用 CLIP 文本编码 + AdaIN 风格迁移；扩散模型实现局部结构扰动），离散与连续特征解耦的特征嵌入，类别原型聚类与 L2/对比式回归损失，以及多任务联合优化。

**📊 数据集**

使用的数据集：① 驾驶天气数据集（Day Clear、Night Sunny、Dusk Rainy、Night Rainy、Day Foggy）；② 真实‑艺术迁移基准（VOC → Clipart、Watercolor、Comic）；③ 零样本语义分割基准（Cityscapes → ACDC、GTA5 → Cityscapes）。

**📈 对比分析**

与现有方法（S‑DGOD、C‑Gap、PDOC、DIV、FWCL、SE‑COT 等）在多种检测框架（Faster R‑CNN、YOLOv10‑L、DiffusionDet、GLIP‑T、DINO‑v2）上进行对比。MR‑DCoT 在所有目标域上均显著提升 mAP，尤其在 Night Rainy、Day Foggy 等极端环境下提升 3–5% 以上，并在 Real‑to‑Art 任务中实现与 SE‑COT 及其它基线的 2–4% 提升。

**⚠️ 局限性**

局限性：① 训练时需预训练 VLM 与扩散模型，增加训练成本；② 原型锚定机制对类别均衡假设敏感，长尾类别可能受限；③ 仅在离散文本提示下实现语义演化，对极端未见域的自适应仍需进一步验证；④ 生成的离流形样本仍基于源域统计，极大差异域（如完全合成数据）仍存在一定泛化缺口。

---

## 189. Rethinking Multimodal Time-Series Forecasting Evaluation

**arXiv ID:** 2607.06973 | [PDF](https://arxiv.org/pdf/2607.06973v1)

**作者:** Haoxin Liu `[一作]` (Georgia Institute of Technology), Abhimanyu Das `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并构建了一个基于真实世界多领域、跨时间尺度、带有高质量文本上下文的多模态时间序列预测基准，并提供了可自动刷新、无泄漏的数据生成管线。

**💡 创新点**

创新点包括：① 设计了可持续刷新且无泄漏的基准，①.1 采用时间隔离与多代理验证机制；② 统一整合元数据、日历、协变量、事件四类丰富文本上下文；③ 通过四个LLM代理（Hypothesizer、Verifier、Enricher、Synthesizer）自动生成事实可验证、时间精确的事件文本；④ 在此真实基准上系统评估零样本多模态模型，揭示传统合成基准对LLM性能的过高/低估。

**🔧 技术方法**

使用的技术主要包括：多模态时间序列基础模型（TimesFM‑2.5、Moirai‑2.0、Sundial），大语言模型（Gemini‑2.5‑Pro、GPT‑4o、DeepSeek‑R1），多代理自动化事件构造，零样本推理与集成策略，以及MASE、CRPS等评价指标。

**📊 数据集**

使用的数据集为《Real‑World Multi‑Domain Time‑Series Benchmark》（简称**RMTS**），包含19个领域、190个变量（每日、每周频率），来源于真实API（Google Trends、MarketStack、Frankfurter）并配有手工和自动化生成的文本上下文，时间覆盖2018‑2025。

**📈 对比分析**

通过对比单模TSF、零样本LLM、简单平均集成、Agentic修正（文本/代码）等方法，实验发现：① 在真实基准上LLM优于单模TSF但差距小；② 简单平均集成（TFM+LLM）表现最佳；③ 添加事件文本可提升约16%精度；④ 结果因领域而异，部分领域（如购物、原材料）对文本更敏感。

**⚠️ 局限性**

局限性包括：① 仍未对训练型方法做系统评估；② 事件文本生成成本高、需人工审核保证质量；③ 仅支持英语及少数语言；④ 评价受模型知识截止时间影响，需频繁更新基准以保持无泄漏。

---

## 190. CoMind: Understanding Collaborative Human Activity from Multiple Minds and Views

**arXiv ID:** 2607.06691 | [PDF](https://arxiv.org/pdf/2607.06691v1)

**作者:** Alexey Gavryushin `[一作]` (ETH Zurich), Xi Wang `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文收集并公开了一个包含双视角（自我与外部）、音频、注视、手部跟踪、3D场景扫描等多模态数据的协作厨房数据集，并在其上定义了联合注意、社交条件动作预测和协作交接预测三项基准任务。

**💡 创新点**

创新点在于首次系统性提供长时段、自然的多人协作录制，丰富的社会线索标注，以及将Theory of Mind拆分为可量化的三种任务，为社交认知研究提供了新的实验平台。

**🔧 技术方法**

采集使用Meta Aria Glasses、GoPro、Leica BLK2GO、Artec 3D Leo等硬件，并利用MPS、WhisperX、TICSync、SyncSink等技术完成多模态同步、3D对齐与标注。

**📊 数据集**

基准测试采用自建CoMind数据集，并与Claude Opus、Gemini 3 Flash、GPT‑4o/GPT‑5、Gemma‑4、Qwen3‑VL等多种闭源与开源视觉语言模型进行对比，其中Qwen3‑VL被微调以验证数据集效用。

**📈 对比分析**

实验结果表明零样本VLM在空间定位和时间预测上表现不佳，微调后在定位与语义识别上显著提升，但整体性能仍低于人类，凸显任务的挑战性。

**⚠️ 局限性**

局限主要包括：数据仅覆盖厨房烹饪场景，缺乏更广泛的协作类型；任务聚焦于视觉/听觉线索，未充分考虑语言或非语言细节；基准模型多为通用VLM，缺乏专门针对社交交互的架构。

---

## 191. Mining Workflow Graphs for Black-Box Boundary Testing of Conversational LLM Agents

**arXiv ID:** 2607.06873 | [PDF](https://arxiv.org/pdf/2607.06873v1)

**作者:** Liting Lin `[一作]` (Lero Research Ireland Centre for Software), Emir Muñoz `[通讯]` (Genesys)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种黑盒式测试框架，通过与对话代理交互挖掘对话工作流图，生成功能性与边界测试并基于可见交互判定通过/失败；

**💡 创新点**

创新点在于利用过程挖掘的直接后继图从自然语言会话中构建工作流模型，再用该结构指导边界测试，显著提升对状态依赖边界的覆盖与多样性；

**🔧 技术方法**

核心技术包括LLM驱动的事件抽象、直接后继图构建、结构化边界目标枚举、基于对话路径的扰动生成以及独立判定器（judge）；

**📊 数据集**

在四个τ³-bench文本域（航空、零售、电信、银行）上评估；

**📈 对比分析**

与拥有源代码访问权限的白盒审计器对比，实验显示功能测试覆盖率从0.72提升至0.97，边界测试覆盖23–38个不同边界，重复率下降至0.26，误报率低于0.1；

**⚠️ 局限性**

局限包括：仅适用于具有可重复交互模式的服务型对话代理；对隐藏但不在对话中表现的故障不可检测；判定器完全基于文本，可能误判需后端状态的错误。

---

## 192. HiFuzz: Hierarchical Reinforcement Learning for Semantic-Aware and Adaptive CPU Fuzzing

**arXiv ID:** 2607.06619 | [PDF](https://arxiv.org/pdf/2607.06619v1)

**作者:** Ya Wang `[一作]` (Hong Kong University Of Science And Technology), Wei Zhang `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了HiFuzz，一种基于层次强化学习的硬件模糊框架，利用程序层Agent规划全局结构，基本块层Agent生成指令流，并通过语义感知基本块编码器提供即时奖励；

**💡 创新点**

创新点在于：1）双层RL架构将宏观结构和微观指令生成分离，降低动作空间；2）语义感知BB编码器在无硬件仿真的情况下提供密集的创新奖励；3）基于UCB的模块级适应性奖励机制平衡覆盖率；

**🔧 技术方法**

技术包括：层次强化学习（Rainbow DQN + PPO）、Bi-LSTM语义编码器、两阶段自监督预训练（MLM）+对比学习、上界置信度（UCB）奖励、可插拔的硬件覆盖收集器；

**📊 数据集**

使用三款真实RISC‑V核心（Rocket、BOOM、CVA6）作为被测件，利用Spike做参考模型；在Bug检测试验中使用Encarsia注入错误数据集；

**📈 对比分析**

与三种主流模糊器（DifuzzRTL、ProcessorFuzz、Cascade）对比，HiFuzz在控制寄存器覆盖率、总覆盖率和Bug发现率上均有显著提升，Rocket上提升约49%，BOOM上Bug检测从26/30提升至30/30；

**⚠️ 局限性**

局限性包括：仅支持单核程序生成，无法处理多核同步和A扩展导致的并发错误；依赖模块级覆盖信号，缺失时效果减弱；对其他ISA的迁移需要重写编码器与生成器；训练与推理开销相对较大。

---

## 193. SpiS-GAN: Spiral-Modulated Handwriting Synthesis with Star Operation

**arXiv ID:** 2607.06949 | [PDF](https://arxiv.org/pdf/2607.06949v1)

**作者:** Nguyen Duy Hieu `[一作]` (University of Information Technology), Vo Nguyen Le Duy `[通讯]` (University of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于生成对抗网络的单示例手写文字合成框架SpiS-GAN，专门解决手写体生成中的笔画连贯、细节保留与风格一致性难题。

**💡 创新点**

创新点包括：1）Modulated Elliptical SpiralFC与Star运算融合的Star‑Spiral Block，利用可调椭圆形取样与星形特征乘法实现高维特征交互；2）Spiral‑Modulated MLP discriminator，三路（空间、螺旋、频谱）融合捕捉笔画细节；3）Sobel‑Regularized Edge Reconstruction Loss，直接对边缘进行L1正则，提升笔画边界锐度；4）综合使用Fidelity、频谱、边缘多目标训练策略。

**🔧 技术方法**

技术方法包括：GAN（hinge loss）、StarNet星形乘法、SpiralMLP、MLP‑based discriminator、频谱分布损失（FDL）、Sobel边缘损失、KL正则、CTC识别监督、风格编码器。

**📊 数据集**

使用的数据集：IAM（英语手写）和HANDSL-VNOnDB（越南手写）以及在实验中对多语言通用性进行验证。

**📈 对比分析**

与FWGAN、HiGAN、HWT、VATr、DiffusionPen、One‑DM等最新方法在同一数据集上对比，SpiS‑GAN在32/64像素下Fidelity（FID）分别低至4.37/4.58，KID/Handwriting Distance（HWD）也显著优于对手，且在低资源HTR数据增强实验中显著降低CER/WER。

**⚠️ 局限性**

局限性主要体现在：1）仍依赖高性能GPU训练；2）对极端稀有字形或非常不同书写系统（如非拉丁文字）的泛化尚未彻底验证；3）单示例风格迁移仍可能出现少量笔画细节偏差。

---

## 194. LLM-powered reasoning in agent-based modeling

**arXiv ID:** 2607.06757 | [PDF](https://arxiv.org/pdf/2607.06757v1)

**作者:** Sifat Afroj Moon `[一作]`, Heidi Hanson `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一个可扩展的混合代理-语言驱动的流行病建模框架，在个体层面结合了基于活动的时间网络与大语言模型（LLM）实时推断人类决策，从而在盐湖城地区对 COVID‑19 传播进行近实时仿真。

**💡 创新点**

① 将 LLM 作为反馈循环动态调整代理的刻意行为，实现数据稀缺情况下的行为推断；② 通过分层空间与人口特征聚类，将数百万代理划分为仅 1,552 个 LLM 组，实现大规模并行推断；③ 在 ABM 与 LLM 之间设计可调节的时钟同步，兼顾计算效率与时效性。

**🔧 技术方法**

混合框架核心技术包括：\n- 基于活动的时间网络构建（使用 2017 年 NHTS 活动日程与合成人口数据）；\n- LLM 推断（使用 Llama 3.1 8B 参数模型与 vLLM 进行批量推理，输出结构化的 yes/no 决策）；\n- 并行分布式 ABM（使用 Repast + MPI）；\n- 客户端-服务器交互（FastAPI/uvicorn）与 HPC 作业调度（Slurm）。

**📊 数据集**

主要数据集：\n- Salt Lake County 人口合成模型（UrbanPop + PUMS 匹配）；\n- 2017 年 National Household Travel Survey（NHTS）活动日程（约 1.89 亿条记录）；\n- 公开的 COVID‑19 周度病例数据（CDC）；\n- Uber H3 位置索引与 FEMA USA Structures 住宅位置。

**📈 对比分析**

通过 30 次独立仿真比较，框架生成的疫情曲线峰值时间与观测值高度吻合，感染总数与观测值相差约 2.21 倍（因观测缺失无症状病例）。ABM‑only 模型在峰值和规模上与实际数据差距明显；即使随机去活 35% 关联，仍无法匹配。框架在 LLM 推断温度 0.2 时表现出最优的行为预测与模型匹配。

**⚠️ 局限性**

局限性：\n- LLM 仅在群体层面做决策，缺乏个体细粒度解释；\n- 依赖预先生成的合成人口与活动日程，现实中活动数据仍难获得；\n- 只采用零射训练，未利用检索增强（RAG）或实时天气等信息；\n- 对 LLM 推断温度与提示敏感，模型可能带有偏见；\n- HPC 资源需求高，单 GPU 服务器可能成为瓶颈。

---

## 195. Ego-Human Motion Prediction with 3D-Aware LLM

**arXiv ID:** 2607.07001 | [PDF](https://arxiv.org/pdf/2607.07001v1)

**作者:** Yujin Bae `[一作]` (KAIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Ego3DLM，一个基于大语言模型的单向自回归框架，能够同时预测过去和未来的3D人体姿态以及相应的自然语言描述。

**💡 创新点**

创新点在于：①将三维场景的空间与语义信息与多模态输入（三点跟踪、第一人称视频、3D场景特征）整合进LLM；②在单个推理过程中同时生成姿态和文本，实现跨模态与时间一致性；③采用三阶段训练（空间语义预训练、全局多任务指令微调、GRPO强化学习），显著提升姿态与文本的相互一致性。

**🔧 技术方法**

技术方法包括：Q-Former 对3D点云压缩、Mask2Former+EVA-ViT-G 进行2D语义特征提升、GPT-2/ Qwen 2.5 作为语言模型、PQ‑VAE 对姿态离散化、空间语义QA预训练、全局多任务指令微调、Group Relative Policy Optimization（GRPO）强化学习与跨模态匹配奖励。

**📊 数据集**

使用 Nymeria 公开数据集（包含第一人称视频、3D点云、语言叙述）进行训练与评估。

**📈 对比分析**

与 EgoLM、FIction、UniEgoMotion 等基线对比，Ego3DLM 在未来姿态预测、过去姿态追踪、姿态与文本一致性（BLEU、R‑precision、d_pp 等）上均达到或逼近 SOTA，尤其在姿态预测精度（JPE、ADE）和语义一致性（d_pp）上显著提升。

**⚠️ 局限性**

主要局限：模型依赖预先构建好的3D场景特征，无法在线实时获取新场景的几何信息；对陌生或动态环境的泛化能力待进一步验证。

---

## 196. On Adversarial Vulnerability of Vision-Language Models through the Lens of Intermediate Spectral Subspaces

**arXiv ID:** 2607.07375 | [PDF](https://arxiv.org/pdf/2607.07375v1)

**作者:** Chethan Krishnamurthy Ramanaik `[一作]` (University of Bundeswehr Munich), Eirini Ntoutsi `[通讯]` (University of Bundeswehr Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究Transformer‑based VLM的中间线性变换的谱结构，发现底部奇异向量子空间是未被注意的攻击表面，并提出基于谱子空间引导的白盒攻击SSGRA。

**💡 创新点**

创新点在于将信息衰减的底部奇异方向视为攻击渠道，并通过谱子空间对齐提升对抗鲁棒性，首次在VLM中验证该思路。

**🔧 技术方法**

使用了SVD谱分解、投影能量度量、梯度优化以及与现有特征/输出层攻击的对比。

**📊 数据集**

实验使用ImageNet图像数据集，在Gemma‑3、Qwen2.5‑VL、LLaVA‑1.5三款VLM上评估。

**📈 对比分析**

与BSA、DRA、FDA、EGA等六种代表性攻击对比，SSGRA在BERTScore和ROUGE‑L F1上显著提升，尤其在较大扰动下可提高30‑98%。

**⚠️ 局限性**

局限在于仅考虑白盒无目标攻击，未探讨黑盒或迁移攻击，并且对谱正则化的组合效果尚未研究。

---

## 197. Riemannian Geometry for Pre-trained Language Model Embeddings

**arXiv ID:** 2607.07047 | [PDF](https://arxiv.org/pdf/2607.07047v1)

**作者:** Szczepan Konior `[一作]` (IBM Automation and AI), Bartłomiej Sobieski `[通讯]` (University of Warsaw)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了基于Riemannian几何的句子池化方法，利用预训练模型的拉回度量和SPD弗里切平均进行分类。

**💡 创新点**

将token级拉回度量聚合为SPD弗里切平均并在切空间做线性分类，证明几何聚合在信号获取上优于欧氏均值，且随机编码器+几何聚合已具备优势。

**🔧 技术方法**

使用Intrinsic Green's Learning训练轻量级MLP获取雅可比矩阵，计算拉回度量，SPD弗里切均值，Riemannian whitening，切空间投影加逻辑回归。

**📊 数据集**

在CoLA、CREAK、RTE以及去除词汇偏差的负控制数据集FEVER‑Symmetric上进行实验。

**📈 对比分析**

与欧氏均值池化和CLS聚合在相同BERT‑Base第9层嵌入上进行5折交叉验证，RMP在三组信号集上显著优于基线（ΔAUC约0.02‑0.06），在负控制保持接近随机。

**⚠️ 局限性**

计算成本较高，超参数与架构的鲁棒性未完全评估，仅在单一模型层和编码器上验证，泛化性待进一步研究。

---

## 198. Creating a Mixed-Reality Installation with Families through Theatrical Co-Design

**arXiv ID:** 2607.06754 | [PDF](https://arxiv.org/pdf/2607.06754v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 199. A Large Language Model-Driven Agent-Based Modeling Framework with Multi-Round Communication for Simulating Vaccine Opinion Dynamics

**arXiv ID:** 2607.07387 | [PDF](https://arxiv.org/pdf/2607.07387v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 200. Sparse Attention for Dense Open-Vocabulary Prediction in CLIP

**arXiv ID:** 2607.07135 | [PDF](https://arxiv.org/pdf/2607.07135v1)

**作者:** Fatimah Zohra `[一作]` (King Abdullah University of Science and Technology), Bernard Ghanem `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在冻结的CLIP视觉编码器的最终自注意力层，将softmax正则化替换为α-entmax，实现了稀疏注意力分布，从而提升了稠密开词汇预测任务的性能。

**💡 创新点**

创新点在于：①仅通过无训练的注意力稀疏化（无额外参数）显著增强CLIP在像素级分割与细粒度区域检索上的定位精度；②系统性分析稀疏程度与原始注意力扩散度的关系，揭示稀疏化收益与注意力分散程度成正比。

**🔧 技术方法**

核心技术是α-entmax注意力正则化（α=1.2），对自注意力得分进行阈值化，零化低相关键；实验中还对比了self‑correlation（query‑query、key‑key、value‑value）以及softmax、温度锐化、随机掩码等基线。

**📊 数据集**

使用的公开数据集包括：Pascal VOC、Pascal Context、ADE20K（像素级语义分割）以及FG‑OVD（细粒度区域‑文本检索）。

**📈 对比分析**

方法对比基线为冻结CLIP的softmax注意力，评估指标为mIoU（分割）和mHME（检索）。实验结果显示，α-entmax在自相关注意力上提升≈5–12点mIoU，提升≈5–7点mHME，且收益随分辨率和模型规模增大而提升，证明稀疏化对高分辨率/大模型尤为有效。

**⚠️ 局限性**

局限性包括：①稀疏化效果受原始注意力分布的扩散度限制，对已高度集中注意力的分布无显著提升；②仅在推理阶段应用，未探索在训练时联合稀疏化的潜在进一步改进；③对不同α值的最优性尚未系统探究，仅以α=1.2为主。

---

## 201. PUF: Plug-and-Play Uncertainty-Aware Fusion for Online 3D Scene Graph Generation

**arXiv ID:** 2607.07170 | [PDF](https://arxiv.org/pdf/2607.07170v1)

**作者:** Yi Yang `[一作]` (Leibniz Universität Hannover), Michael Ying Yang `[通讯]` (University of Bath)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个无训练、可插拔的在线 3D 场景图生成框架 PUF，能够在保持实时性的前提下融合 2D 观察并更新 3D 场景图；

**💡 创新点**

核心创新在于：① 通过概率化节点关联（融合语义与空间不确定性）；② 使用 Dirichlet 分布累积节点与边的软标签证据；③ 可选的关系先验来补全稀疏观察；实现了对多种 3D 表示（高斯与体素）的无缝支持；

**🔧 技术方法**

使用 Dirichlet 参数化、联合概率数据关联（JPDA）近似、Bhattacharyya 余度、温度缩放等技术；

**📊 数据集**

在 3DSSG 与 ReplicaSSG 两个基准上进行评估；

**📈 对比分析**

与现有在线方法（FROSS、SGFN、MonoSSG 等）和离线方法对比，PUF 在 3DSSG 上的关系 Recall@1 最高可达 46.0%，比 FROSS 提升 18.1 点，且时延仅 15 ms/帧；在 ReplicaSSG 上亦保持显著提升，且保持实时性能；

**⚠️ 局限性**

局限性：依赖 2D 检测器的软输出，对检测错误（如漏检）仍无法完全弥补；对极端稀疏或高度遮挡的关系仍可能不准确；以及对不确定性校准的敏感性需进一步研究。

---

## 202. Certifying Ghosts: How Cybersecurity AI Agents Break the EU Cyber Resilience Act

**arXiv ID:** 2607.07109 | [PDF](https://arxiv.org/pdf/2607.07109v1)

**作者:** Víctor Mayoral-Vilches `[一作]` `[通讯]` (Alias Robotics), Víctor Mayoral-Vilches (Alias Robotics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估欧盟网络韧性法在面临 AI 驱动的漏洞发现与利用时的缺陷，并提出基于持续 AI 防御的合规方案。

**💡 创新点**

将法规背后的四大前提拆解为“弯曲”与“断裂”两类影响，并用实际机器人案例验证 AI 防御能持续维持合规。

**🔧 技术方法**

结合生成式 AI 漏洞挖掘器、自动化渗透工具及持续运行的 AI 防御平台（RIS）实现漏洞检测与即时修补。

**📊 数据集**

使用 CVE‑bench、CyberGym、公开的 AI 渗透实验数据以及两台商用机器人（Unitree G1 与 Hookii Mower）的固件漏洞库。

**📈 对比分析**

通过对比未加防御与 RIS 防御两种状态下的攻击成功率，发现成功率从 79%/75% 降至 14%/8%，并在数秒内完成检测与阻断。

**⚠️ 局限性**

需要人工监督的半自动化防御、对 AI 安全保障的信任挑战以及监管层需增设持续合规与环境触发机制。

---

## 203. Behavior Foundations for Quadruped Robots: ABot-C0 Technical Report

**arXiv ID:** 2607.07370 | [PDF](https://arxiv.org/pdf/2607.07370v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 204. $(5+ε)$-Approximation of Fréchet Distance in Strongly Subquadratic Time

**arXiv ID:** 2607.06864 | [PDF](https://arxiv.org/pdf/2607.06864v1)

**作者:** Lenny Liu `[一作]`, Jihan Wang `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

对任意维度下的连续与离散 Fréchet 距离提出随机化的 (5+ε)-近似算法，并给出对应的时间复杂度；

**💡 创新点**

创新点在于利用块分解、辅助替代曲线与两尺度宏观搜索相结合，既减小了匹配检查次数，又避免了额外的三角不等式损失，从而将近似因子降至 5 并显著提升运行时间；

**🔧 技术方法**

主要技术包括自由空间图的局部传播、曲线简化（近似与精确）、Dyadic 辅助传递结构、宏观/细尺度宏搜索以及可预处理的平面可达性预处理；

**📊 数据集**

本文为理论研究，无实测数据集，所有结果均为算法复杂度分析；

**📈 对比分析**

与之前的 Cheng、Huang、Zhang (7+ε) 近似算法相比，本工作在连续情形实现 O(nm^{8/9}) 时间、离散情形实现 O(nm^{4/5}) 时间，显著低于 O(nm^{0.99})；

**⚠️ 局限性**

局限性：仍为随机化算法；适用固定维度；近似因子 5 仍高于已知的 3 的下界；算法实现复杂、常数隐含较大；未提供实验验证。

---

## 205. CoFINN: Conservation Flux Informed Neural Networks for Physics Problems Governed by Conservation Laws

**arXiv ID:** 2607.06587 | [PDF](https://arxiv.org/pdf/2607.06587v1)

**作者:** Adnan Harun Doğan `[一作]` (Middle East Technical University), Özgür Uğraş Baran `[通讯]` (Middle East Technical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种将有限体积守恒物理嵌入神经网络训练的框架CoFINN，用于预测压缩流域中的空气动力学场。

**💡 创新点**

创新点在于将CNN输出视为离散控制体积网格，直接通过HLLC Riemann 求解器计算数值通量，将守恒误差加入损失函数，实现在离散积分意义上的物理约束；同时保持CNN预测速度并兼容多种网络架构。

**🔧 技术方法**

使用卷积网络、Fourier神经算子、视觉变换器和条件扩散模型作为基础网络，并在训练中加入MAE损失与CoFINN物理损失；HLLC Riemann solver用于计算数值通量；使用标准优化器与学习率调度、EMA等技术。

**📊 数据集**

使用由Reynolds平均Navier–Stokes CFD生成的204种翼型在M∞=0.7、Re=6×10⁶下的6,324个二维压缩流场数据，网格为256×256；10种翼型保留为固定测试集，其余分为训练/验证。

**📈 对比分析**

通过与仅使用MAE损失的基线模型、以及不同角度训练范围和多种网络架构进行比较，CoFINN在阻力预测误差上最多可降低34%（极端攻角）或平均15%；在升力误差上也实现显著改善；在有限数据场景下提升尤为明显；训练时间提升约20%，推理速度保持不变。

**⚠️ 局限性**

局限包括：只能处理二维结构网格；对薄粘性边界层的描述不足，主要只约束无粘流量；CoFINN损失无法单独使用，仍需数据监督；最佳损失权重需手动调节；在三维或更高雷诺数/不同马赫数场景的泛化尚待验证。

---

## 206. Granularity in Actoin: Graphing sources for social history

**arXiv ID:** 2607.07183 | [PDF](https://arxiv.org/pdf/2607.07183v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 207. Complexity-Budgeted, Interaction-Aware Interpretable Model for Tabular Data

**arXiv ID:** 2607.07060 | [PDF](https://arxiv.org/pdf/2607.07060v1)

**作者:** Srikumar Krishnamoorthy `[一作]` `[通讯]` (Indian Institute of Management Ahmedabad), Srikumar Krishnamoorthy (Indian Institute of Management Ahmedabad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种名为 Interaction Aware Interpretable Machine Learning (IAIML) 的框架，利用自适应特征离散化、交互信息评分与预算限制，实现对表格数据的可解释模式学习，并将交互信号注入模型；

**💡 创新点**

创新点包括：①提出边缘信号独立的交互信息阈值筛选；②提供两种交互通道——松弛筛选和显式对称算子；③使用分区预算控制解释组件数量，兼顾可解释性与性能；

**🔧 技术方法**

采用信息论交互信息估计、离散化搜索、HUG‑IML 高效模式挖掘、稀疏逻辑回归以及配对算子构造等技术；

**📊 数据集**

实验使用40个二分类任务，包括24个真实世界表格基准（金融、医疗、营销等）和16个设计的交互压力测试；

**📈 对比分析**

与调优后的 XGBoost、LightGBM、EBM、RuleFit 等方法比较，IAIML 在 AUC 上仅低 1.4 点，但组件数比树模型和 EBM 少 14–28 倍；在低边缘信号高交互的数据上 IAIML 表现最佳；

**⚠️ 局限性**

局限性在于只能处理二元交互，无法捕获更高阶交互或复杂非线性；离散化可能导致信息损失；所用算子范围有限，无法解释所有交互模式。

---

## 208. URS-Stereo: Uncertainty-Guided Residual Search for Real-Time Stereo Matching

**arXiv ID:** 2607.06779 | [PDF](https://arxiv.org/pdf/2607.06779v1)

**作者:** Pouya Sohrabipour `[一作]`, Dongyi Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种实时粗到细立体匹配框架 URS‑Stereo，通过不确定性引导残差搜索模块在细化阶段自适应调整局部成本体搜索中心。

**💡 创新点**

核心创新是利用传递视差的不确定性估计来调节残差偏移，从而在粗到细过程中动态定位搜索窗口，显著降低因视差传播误差导致的局部搜索失败。

**🔧 技术方法**

采用多尺度特征提取、组间相关、ConvGRU 递归细化、UGRSM（不确定性预测 + 残差偏移）、单目-立体互相细化以及端到端训练等技术。

**📊 数据集**

训练使用 SceneFlow 数据集，测试在 KITTI 2012、KITTI 2015、Middlebury 和 ETH3D 等公开基准上进行零样本评估。

**📈 对比分析**

与 FastACV、LightStereo、Lite Any Stereo 等实时立体匹配方法对比，URS‑Stereo 在 D1/EPE 上均实现了显著提升，保持了实时推理速度，并展现了强大的零样本泛化能力。

**⚠️ 局限性**

仍受限于局部搜索半径设置、极端大视差跳变或严重遮挡场景的鲁棒性，以及对极端光照/动态环境的适应性不足。

---

## 209. EvoPlan: Evolutionary Neuro-Symbolic Robot Planning with Spatio-Temporal Guarantees

**arXiv ID:** 2607.06724 | [PDF](https://arxiv.org/pdf/2607.06724v1)

**作者:** Bhavya Sai Nukapotula `[一作]`, Srinivas Shakkottai `[通讯]` (Texas A&M University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `51c0528b-f690-4182-ae60-bb5f046c276c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个神经符号框架，将大语言模型（LLM）与PDDL规划和全局Signal Temporal Logic（STL）约束结合，用于生成可执行且安全的机器人执行计划；该框架包括离线挖掘STL约束、LLM驱动的进化PDDL规划以及将约束检查嵌入执行循环的闭环机制；

**💡 创新点**

创新点在于：①通过对无标注演示数据进行反事实负样本生成，使LLM能够学习单一全局STL约束，既可表达编码规则也可捕捉群体偏好；②使用LLM生成和修复PDDL计划，并通过程序化验证器级联实现可检查性；③将全局STL约束嵌入实时执行循环，形成动态补救的“盾牌”机制；④全部依赖本地部署的开源权重模型，消除云端依赖；

**🔧 技术方法**

采用技术包括：大语言模型（Qwen3-32B）用于生成计划、生成违反样本和STL约束；STL 约束挖掘与进化搜索；PDDL 与 VAL 验证器；演化规划（LLM提议、验证器反馈、迭代演化）；机器人导航层（Nav2、轨迹规划）与ROS2桥接；

**📊 数据集**

使用的数据集包括：nuPlan 驾驶日志、SCAND 远程操作日志、Bench2Drive 驾驶评测、HA-VLN-CE 机器人导航、ALFWorld、AlfredTWEnv TextWorld 以及 Gazebo 模拟工厂场景；

**📈 对比分析**

方法与基线对比：在 ALFWorld 文本任务中，演化规划在 90.3% 通过率、对深度 10 以上任务超越 NL-PDDL；在 Bench2Drive 驾驶堆栈中，STL 约束将红灯违规降低 77%、碰撞降低 71%；在 HA-VLN-CE 导航任务中，STL 约束分别降低 82% 与 93% 的可避免接触；Gazebo 实验中 5/5 任务成功，4/5 无碰撞，展示了安全盾牌的有效性；

**⚠️ 局限性**

局限性包括：依赖演示数据的质量与多样性；STL 语法表达能力有限，可能无法覆盖更复杂的约束；感知不确定性会影响约束评估与计划执行；LLM 生成质量与训练数据相关，可能出现错误或偏差；目前仅支持单一全局移动约束，未覆盖多机器人协同或更细粒度的任务约束；

---

## 210. Pelican-VLA 0.5: Attending Before Acting Benefits Generalization

**arXiv ID:** 2607.06655 | [PDF](https://arxiv.org/pdf/2607.06655v1)

**作者:** Zeyuan Ding `[一作]` (Beijing Innovation Center of Humanoid Robotics), Xiaozhu Ju `[通讯]` (Beijing Innovation Center of Humanoid Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Pelican‑VLA 0.5，一种统一的Vision‑Language‑Action模型，通过在感知层与动作层之间插入可学习的Reasoning Slots，实现了在预训练阶段就能将注意力集中在指令相关物体及其接触区域；

**💡 创新点**

创新点在于将Reasoning Slots作为感知到动作的瓶颈，强迫模型在不依赖对象标签或注意力监督的情况下自动学习任务相关的空间表示；并通过正交正则和对比学习进一步使得该表示对语言更具可控性；

**🔧 技术方法**

技术手段包括Qwen3‑VL 4B Transformer backbone、流匹配动作学习、未来帧预测（Cosmos latent）、Slot‑Language对比学习（InfoNCE）、Slot正交正则、以及多任务联合训练；

**📊 数据集**

使用跨机器人多源数据集：AgiBot World Alpha、InternData‑A1、Galaxea Open‑World Dataset、约1000小时自采的Tienkung和UR遥控数据，共计约6000小时；在预训练中仅覆盖约2400小时；

**📈 对比分析**

在RoboTwin基准上与多种开源VLA基线（π0、X‑VLA、StarVLA‑OFT、LingBot‑VLA、JoyAI‑RA等）对比，Pelican‑VLA 0.5在Clean/Randomized场景下分别获得91.4%/91.0%的成功率，平均91.2%，优于其他基线；在零射场景表现出目标定位正确但控制细节不足；fine‑tuning后成功率显著提升；

**⚠️ 局限性**

主要限制是数据规模和动作表示；预训练仅覆盖约2400小时且使用关节位动作，导致跨机器人泛化受限；零射任务成功率仍低，说明表示与可执行动作之间仍存在“表示‑到‑动作”缺口。

---

## 211. Simplicial subdivision of simplices of arbitrary dimension in spaces of constant curvature with bounded quality

**arXiv ID:** 2607.06801 | [PDF](https://arxiv.org/pdf/2607.06801v1)

**作者:** Jean-Daniel Boissonnat `[一作]` (Inria centre Université Côte d'Azur), Mathijs Wintraecken `[通讯]` (Inria centre Université Côte d'Azur)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明在任意维度的常曲率空间中，单形可以细分且细分后的单形质量（fatness）有下界。

**💡 创新点**

将Freudenthal的三角化方法与径向投影相结合，扩展Brunck在二维常曲率空间的结果到任意维度，并对比两种构造。

**🔧 技术方法**

Freudenthal–Kuhn三角化、径向投影、质量度量（fatness）分析。

**📊 数据集**

无实验数据集，纯理论证明。

**📈 对比分析**

与Brunck的二维构造进行理论对比，证明本方法在高维空间同样能保证质量下界；无实验性能指标。

**⚠️ 局限性**

仅适用于常曲率空间，未探讨非恒定曲率或更一般几何情况；缺乏对实际数值PDE网格生成的具体实现细节。

---

## 212. An Automated Framework for Generating Stealthy Cell-Embedded Hardware Trojans

**arXiv ID:** 2607.07049 | [PDF](https://arxiv.org/pdf/2607.07049v1)

**作者:** Raghul Saravanan `[一作]` (George Mason University), Swarup Bhunia `[通讯]` (University of Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个自动化框架，在标准单元实现内部嵌入隐蔽的硬件木马，从而保持网表结构不变。

**💡 创新点**

创新点在于把木马隐藏在标准单元内部，而非传统的网表级插入，突破了零信任模型下的检测盲区。

**🔧 技术方法**

采用罕见输入条件分析、序列触发器嵌入、功能等价验证与逻辑等价检查（LEC）等技术。

**📊 数据集**

使用GSCL 45nm PDK标准单元库以及四个公开基准设计（UART、S38584、SHA、AES）。

**📈 对比分析**

与随机测试、统计激活、ML检测以及LEC等方法对比，所有方法均未能检测到嵌入木马；面积和功耗提升低于3%（大规模设计更低）。

**⚠️ 局限性**

局限性包括仅针对单元级木马，尚未在更大规模设计上全面验证，且对抗新型检测机制仍有待研究。

---

## 213. What Semivalues Cannot See: The Information Content of Anonymous Marginal Values

**arXiv ID:** 2607.07013 | [PDF](https://arxiv.org/pdf/2607.07013v1)

**作者:** Matthew Fried `[一作]` `[通讯]` (SUNY Farmingdale), Matthew Fried (SUNY Farmingdale)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探讨了半值家族在合作博弈中的作用，特别是它们如何通过每个玩家在不同联盟规模下的总协同效应来描述博弈的结构。

**💡 创新点**

创新点在于提出了一个具体的结构，表明所有半值的联合信息恰好是每个玩家在每个联盟规模下的总协同效应，并且证明了在特定条件下的审计方法能够精确恢复这些信息。

**🔧 技术方法**

使用了Harsanyi分红坐标和混合差异审计等技术，结合线性代数和组合数学的方法。

**📊 数据集**

使用了包含7580个单调简单博弈的完整普查数据集，特别是在n=5的情况下进行了详细的分析。

**📈 对比分析**

与传统合作博弈相比，经典合作博弈在可见性上达到了0.90到1.00，而随机博弈的可见性仅为0.089，表明经典博弈在半值家族中表现出更高的可见性。

**⚠️ 局限性**

限制在于，纯盲博弈在经济上是病态的，无法满足超加性、单调性和核心存在性等标准正则性属性，因此无法完全识别博弈的结构。

---

## 214. Generalist Vision-Language Models for Fast Radio Burst detection: a zero-shot benchmark against a specialized detector

**arXiv ID:** 2607.07382 | [PDF](https://arxiv.org/pdf/2607.07382v1)

**作者:** Raiff H. Santos `[一作]` (Universidade Federal de Campina Grande), Rafael A. Batista `[通讯]` (Universidade Federal de Campina Grande)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估零样本（zero-shot）通用视觉-语言模型（Gemma 4 2B/4B）在模拟动态频谱上检测快速射电暴（FRB）的能力，并与专用深度学习检测器SwinYNet进行样本级比较。

**💡 创新点**

首次将零样本通用VLM用于FRB检测，证明其在未微调、无标签数据下即可实现与专用模型相近的准确率，并可仅通过重新编写提示词实现三分类（FRB / RFI / 噪声）判断；同时显著降低了对结构化RFI的误检率。

**🔧 技术方法**

使用Gemma 4系列VLM进行prompt‑only推理，输出结构化JSON；采用simulateSearch生成L‑band动态频谱并渲染PNG；通过准确率、宏F1、ROC‑AUC、平均精度、McNemar检验等统计指标进行评估。

**📊 数据集**

3000条模拟L‑band动态频谱（1000 FRB、2000负样本，其中500 RFI、500 噪声）；二分类基准2000条（1000 FRB+1000负样本）及完整三分类3000条；数据包含DM、宽度、流量密度等物理参数。

**📈 对比分析**

采用样本级配对比较，阈值0.5时Gemma 4 2B准确率93.65%、宏F1 0.9364，SwinYNet为92.90%/0.9286；ROC‑AUC分别为0.9482与1.000；在RFI上的误检率分别为6.4%和25%；对比通过McNemar检验发现Gemma 4 2B与SwinYNet无显著差异，Gemma 4 4B差异显著。

**⚠️ 局限性**

主要局限包括：仅使用模拟数据，未验证真实观测的鲁棒性；输入不对等（VLM仅接收PNG，SwinYNet接收FITS）给SwinYNet优势；数据平衡不符合实际FRB稀缺性；VLM输出概率高度量化，影响概率评估；潜在预训练中已见过FRB图像；推理时间≈5–8 s/2 s数据，尚不能实时；解释性输出仍需人工审核。

---

## 215. Do Counterfactually Fair Image Classifiers Satisfy Group Fairness? -- A Theoretical and Empirical Study

**arXiv ID:** 2607.06603 | [PDF](https://arxiv.org/pdf/2607.06603v1)

**作者:** Sangwon Jung `[一作]` (Seoul National University), Taesup Moon `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建高质量的图像对照因果（counterfactual）数据集，并在此基础上研究图像分类中的反事实公平（CF）与群体公平（GF）的关系，提出一种同时实现CF和GF的基线方法 CKD。

**💡 创新点**

① 公开真实图像的CF评估基准，解决传统难以获得真实CF样本的问题；② 理论证明在图像数据中存在与敏感属性相关但不受其因果影响的第三方属性 G，导致 CF 并不必然蕴含 GF；③ 设计 CKD 通过知识蒸馏和对抗正则化同时抑制对 G 的依赖，从而实现 CF 与 GF 的兼顾。

**🔧 技术方法**

使用 InstructPix2Pix（IP2P）与 SDEdit 等高质量扩散模型生成 CF 图像；对图像进行人工过滤与可靠性验证；采用 CP 正则化实现 CF 训练；引入 CKD（对齐教师模型表示并结合 CP）实现对 G 的鲁棒性提升；评估指标包括 CD（反事实差异）、DEO（等化偏差）和 RFP（对 G 变换的预测翻转率）。

**📊 数据集**

CelebA-CF 与 LFW-CF（基于 CelebA 与 LFW 的 CF 扩充版本）；CIFAR-10B（可控合成数据，用于验证 G 的影响）。

**📈 对比分析**

与 Scratch、CP、SenSeI、LASSI、SS、RW、COV、MFD、LBC 等多种公平训练方法在同一数据集上进行统一对比；使用 CD、DEO、准确率作为评估指标。实验结果表明，CKD 在保持或提升 CD 的同时显著降低 DEO，尤其在 GF 指标上优于大多数方法，证明了其在同时实现 CF 与 GF 上的有效性。

**⚠️ 局限性**

① 需先获得对 G 鲁棒的教师模型，若教师对 G 依赖过大，CKD 效果受限；② CF 图像的生成仍受编辑模型偏差和人类标注成本影响；③ 研究聚焦于面部图像，通用性和对其他视觉任务的适用性仍需进一步验证。

---

## 216. An Edge-aware Prompt-enhanced SAM for Ultrasound Image Segmentation

**arXiv ID:** 2607.07240 | [PDF](https://arxiv.org/pdf/2607.07240v1)

**作者:** Wenhao Li `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种边缘感知和提示增强的SAM（EP-SAM）模型，用于超声图像分割，旨在改善超声图像中解剖结构和病变的边界识别。

**💡 创新点**

EP-SAM通过引入边缘感知模块和提示增强模块，增强了SAM的图像编码器和提示编码器之间的协同作用，从而提高了分割精度和边界清晰度。

**🔧 技术方法**

使用了边缘感知模块（EAM）和提示增强模块（PEM），结合多块特征提取和边缘感知监督来优化模型性能。

**📊 数据集**

在多个超声数据集上进行实验，包括TN3K、BUSI和CAMUS，涵盖不同的解剖结构，如乳腺病变、甲状腺结节和心肌。

**📈 对比分析**

与现有的SAM基础方法相比，EP-SAM在无提示和单点提示设置下均表现出色，平均Dice系数达到85.50%，Hausdorff距离最低为22.86，显示出其强大的分割能力。

**⚠️ 局限性**

EP-SAM的局限性在于其对超声图像中低对比度和模糊边界的处理仍然存在挑战，未来的工作将集中在进一步优化自监督提示和多尺度边缘集成上。

---

## 217. Dynamic neural manifolds for flexible closed-loop control on neuromorphic hardware

**arXiv ID:** 2607.07373 | [PDF](https://arxiv.org/pdf/2607.07373v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 218. Self-Supervised Pretraining Improves Cross-Site and Cross-Scale Robustness of Point Cloud Leaf-Wood Segmentation

**arXiv ID:** 2607.06948 | [PDF](https://arxiv.org/pdf/2607.06948v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 219. EdgeCompress: Coupling Multidimensional Model Compression and Dynamic Inference for EdgeAI

**arXiv ID:** 2607.06982 | [PDF](https://arxiv.org/pdf/2607.06982v1)

**作者:** Hao Kong `[一作]` (Nanyang Technological University), Weichen Liu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 EdgeCompress 框架，通过动态图像裁剪、复合压缩和动态推理三阶段，显著降低 CNN 在边缘设备上的计算量。

**💡 创新点**

创新点在于联合压缩输入图像空间冗余、网络深宽与分辨率三维，并设计轻量前景预测器与动态推理策略，实现更高效的多维压缩与自适应推理。

**🔧 技术方法**

使用 Grad‑CAM 自动生成伪边框、轻量残差瓶颈前景预测器、复合压缩系数推导与多模型级联、动态阈值判定等技术。

**📊 数据集**

在 ImageNet‑1K、ImageNet‑100、CIFAR‑10/100 等公开数据集上进行实验验证。

**📈 对比分析**

与现有压缩与动态推理方法对比，ResNet‑50 的 MACs 从 4.1B 降至 2.1B，Top‑1 准确率提升至 76.8%，在 AGX Xavier 等嵌入设备上平均延迟下降 27%~38%，吞吐量提升 30%+。

**⚠️ 局限性**

局限性包括需先生成伪边框的预处理成本、阈值调优对性能影响大，以及多模型级联导致的内存占用和加载开销。

---

## 220. The AI Resilience Gap: Bringing Artificial Intelligence Inside the Operational Resilience Perimeter

**arXiv ID:** 2607.07359 | [PDF](https://arxiv.org/pdf/2607.07359v1)

**作者:** Jonathan Shelby `[一作]` `[通讯]` (University of Oxford), Jonathan Shelby (University of Oxford)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了“AI Resilience Framework”，将AI依赖纳入运营韧性监管范围，构建了关键性-可替代性矩阵、延伸影响容忍度、明确回退规则以及供应商层面的集中风险管理。

**💡 创新点**

创新点在于：①发现并系统化可信AI与运营韧性两大监管逻辑之间的鸿沟；②以矩阵形式量化AI依赖的关键性与可替代性；③将“回退”从单纯的文档化提升为具备可执行、可测量的韧性控制；④将供应商集中风险纳入第三方风险管理框架。

**🔧 技术方法**

本文主要采用概念性分析与监管合规映射技术，并未使用算法实现；其核心技术为法律/监管文本解析、框架设计与层级分类方法。

**📊 数据集**

未使用实际数据集，框架基于文献综述、监管文件与案例示例构建。

**📈 对比分析**

方法与现有框架的对比通过映射表完成，展示如何一次性满足欧盟AI法、英国运营韧性法、DORA、Critical Third Parties等多项要求；并通过示例案例说明相对优势与不足。

**⚠️ 局限性**

局限性包括：①关键性与可替代性分类为定性，缺乏可量化指标；②对外部接口的漂移监控可行性有限；③未深入探讨跨企业的系统级集中风险与威胁交互；④框架主要针对英国/欧盟监管环境，跨国适用性待验证。

---

## 221. Four classes of few-weight self-orthogonal codes and their applications for LCD codes and quantum codes

**arXiv ID:** 2607.07181 | [PDF](https://arxiv.org/pdf/2607.07181v1)

**作者:** Yue Huang `[一作]` (Sichuan Normal University), Qunying Liao `[通讯]` (Sichuan Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构造了四类四重量自正交线性码，并由此得到两类线性互补对码（LCD码）和一类量子码。

**💡 创新点**

创新点在于：①利用增广码与定义集的组合首次得到四重量自正交码；②从这些自正交码构造出的LCD码，其对偶码几乎满足球面封装界限（几乎最优）；③当 m=2 时，构造的量子码达到量子Singleton界限的上限，成为AMDS量子码。

**🔧 技术方法**

主要技术手段包括：定义集与迹函数构造线性码；增广码技术保证p-可除性与自正交性；加法特征、二次特征和Gauss和的计数公式计算权分布；Pless功率矩求解对偶码距离；利用自正交性推导LCD码和量子码的参数。

**📊 数据集**

由于是理论构造，该工作未使用实验数据集；所有结论均通过符号计算与计算机验证（Magma程序）得出。

**📈 对比分析**

通过与已知LCD码与量子码参数表比较，所得到的码的长度、维数或距离均不在已有表中，显示出新颖性；对偶码距离达到或接近球面封装界限；量子码在距离3时满足AMDS条件，证明其最优性。

**⚠️ 局限性**

局限性包括：①对γ≠Tr(-β/α)的情况，权分布无法完全给出；②只讨论奇素数p和m≥2的情形，尚未推广到任意字段；③增广码方法对码长与维数有一定限制，导致可构造的码族相对有限。

---

## 222. Dynamic-in-Few-Step: Unifying Dynamic Computation and Few-Step Distillation for Efficient Video Generation

**arXiv ID:** 2607.06631 | [PDF](https://arxiv.org/pdf/2607.06631v1)

**作者:** Yu Cheng `[一作]` (Zhejiang University), Fajie Yuan `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

将视频扩散模型的少步蒸馏与动态结构稀疏化联合起来，得到一步步适配的混合模型，显著降低每步 FLOPs。

**💡 创新点**

提出联合优化的框架、逆向递进训练和输出回滚的 fake‑score 监督，实现动态稀疏化与少步蒸馏的协同收敛。

**🔧 技术方法**

使用分层结构稀疏（块、头、通道）、分布匹配蒸馏（DMD）、渐进式训练、输出回滚、以及专用推理引擎。

**📊 数据集**

基于 Wan2.1‑14B 预训练模型，使用 Koala‑36M 高质量视频数据进行蒸馏与稀疏训练。

**📈 对比分析**

与单独蒸馏、静态/动态稀疏两阶段管线以及 TurboDiffusion、LightX2V 等 SOTA 进行对比，在 Wan‑14B 上实现 30× 的整体加速，额外 1.2× 的 per‑step FLOPs 节省，且 VBench 质量保持竞争。

**⚠️ 局限性**

仍需与量化、稀疏注意力等其他加速手段联合，动态稀疏策略对不同硬件的兼容性不完全成熟，且仅在 DiT 架构上验证，通用性待进一步验证。

---

## 223. EvoOMG: An Evolution-Oriented Multi-Agent Guidance Framework for Heterogeneous Legacy-and-MLO Wi-Fi Networks

**arXiv ID:** 2607.07045 | [PDF](https://arxiv.org/pdf/2607.07045v1)

**作者:** Junjie Wu `[一作]` (Southwest Jiaotong University), Ziyuan Yang `[通讯]` (Nanyang Technological University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出EvoOMG框架，利用标准感知的分阶段自回归多智能体决策，在混合Wi‑Fi 7/8多链路与传统单链路设备间实现吞吐量优化

**💡 创新点**

创新点：①把设备标准异质性与MAC阶段性嵌入决策结构；②采用Transformer历史编码与争夺→聚合的分阶段动作生成；③标准感知的多头策略与可行性掩模；④集中训练/分散执行并支持联邦学习

**🔧 技术方法**

技术方法：强化学习（CTDE+MADDPG）、Transformer编码器、autoregressive actor、标准可行性投影、集中式评论家、可选联邦聚合

**📊 数据集**

数据集：在NS‑3仿真环境下，使用Wi‑Fi 7/8（Wi‑Fi 7）与Wi‑Fi 6设备，模拟多链路、多频段、移动、UDP负载，生成仿真数据集

**📈 对比分析**

对比方法：MADDPG、IDDPG、Conservative、Greedy；EvoOMG在混合部署下系统吞吐量提升≈29%–30%，MLO链路利用率显著提高，收敛更稳定；在STR/NSTR约束下仍保持约30%吞吐增益，平均延迟与碰撞率基本相同

**⚠️ 局限性**

局限性：仅在仿真环境验证，缺乏真实网络实验；对低优先级AC的延迟与公平性存在折中；模型规模和计算开销相对较大；对不同流量模式与硬件实现的泛化需进一步研究

---

## 224. Diffusion enabled Optimal Transport distances for graph matching

**arXiv ID:** 2607.06646 | [PDF](https://arxiv.org/pdf/2607.06646v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 225. Computing with Stochastic Oracles in AI-Augmented Computation

**arXiv ID:** 2607.06893 | [PDF](https://arxiv.org/pdf/2607.06893v1)

**作者:** Jie Wang `[一作]` `[通讯]` (University of Massachusetts), Jie Wang (University of Massachusetts)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `2704f255-0c84-4173-b83c-0e9a3dbea232` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在缓存式和新鲜式响应条件下，使用随机oracle的SOTM在识别精度和输出质量上的极限及其对token成本的影响。

**💡 创新点**

提出了基于总变差、Chernoff信息和KL散度的识别与输出质量上限，并证明新鲜响应可通过独立采样实现指数级误差下降与查询计数上限。

**🔧 技术方法**

运用信息论方法对转录分布进行分析，结合自适应查询策略与token复杂度框架。

**📊 数据集**

论文为理论研究，无使用具体数据集；采用通用输入分布 _X 作为示例。

**📈 对比分析**

与现有LLM交互模型比较，提供了理论上可达的最高识别成功概率与期望质量门槛，展示了缓存与新鲜响应在token成本与精度之间的权衡。

**⚠️ 局限性**

限制包括假设oracle响应独立且无噪声评分函数，隐藏状态有限且已知分布，且未考虑模型更新或提示变化导致的分布漂移。

---

## 226. Seeing and Reflecting: Multimodal Memory-Enhanced Agent Collaboration for Recommendation

**arXiv ID:** 2607.07108 | [PDF](https://arxiv.org/pdf/2607.07108v1)

**作者:** Hao Cong `[一作]` (Tsinghua University), Lina Yao `[通讯]` (Csiro's Data61)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MMEACR框架，通过多模态记忆增强LLM推荐代理的协同推荐能力；

**💡 创新点**

创新点在于双轨记忆架构（推理记忆与嵌入记忆）及属性引导的记忆演化机制，并通过加权逆序列融合（RRF）将两轨输出融合；

**🔧 技术方法**

主要技术包括多模态LLM生成图像描述、强化学习式记忆更新、属性抽取模块、密集多模态嵌入模型、加权逆序列融合；

**📊 数据集**

使用亚马逊评论的三个子集：CDs、Cell_Phones、Fashion；

**📈 对比分析**

与多种基线（Pop、BM25、SASRec、LLMRank、LLMSeqSim、MLLMSeqSim、AgentCF、CoTAgent）对比，MMEACR-RRF在各指标上往往领先或竞争激烈，尤其在Fashion领域N@1提升约45.45%、N@5提升23.33%、MRR提升27.14%；

**⚠️ 局限性**

主要限制是连续记忆演化导致的长期存储和检索成本上升，需进一步优化记忆容量与效率平衡。

---

## 227. ColorFM: An Optimization-to-Learning Framework for Color Transfer via Flow Matching

**arXiv ID:** 2607.07119 | [PDF](https://arxiv.org/pdf/2607.07119v1)

**作者:** Yuhang He `[一作]` (Nanjing University), Jian Yang `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ColorFM框架，通过流匹配将颜色转移建模为像素分布的流场传输，实现了从实例化优化到高效学习的统一流程。

**💡 创新点**

创新点包括：1）基于语义先验的显式语义对齐与分层颜色耦合，显著提升传输精度和语义一致性；2）利用优化产生的大规模伪标注训练的ColorFM-L，结合隐式状态建模和双向线性化传输，实现实时高质量颜色转移；3）在流匹配中引入可学习速度场，克服传统OT与Rectified Flow的局限。

**🔧 技术方法**

核心技术：流匹配（Flow Matching）框架、分层颜色耦合（Hierarchical Color Coupling）、语义先验（SegFormer-B5语义分割）、隐式状态建模与交叉注意力、双向线性化传输（bidirectional one‑step Euler）以及LPIPS感知损失。

**📊 数据集**

使用237,408个由Unsplash与DIV2K生成的内容-风格-结果三元组进行训练，测试集包含40张Unsplash图片，生成1,560个独立内容-风格配对。

**📈 对比分析**

与WCT^2、PhotoWCT^2、Deep Preset、NLUT、CAP-VST、SA-LUT、D-LUT、Neural Preset和ModFlows等最先进方法进行对比；在Style、Content、Distance to Ideal和Lipschitz指标上均取得领先，速度最高可达0.016秒，4K图像0.043秒，显示出最佳风格-内容平衡与最小视觉伪影。

**⚠️ 局限性**

局限性：ColorFM-O对每个实例仍需调节优化超参数，易受极端颜色风格或语义分布差异影响；ColorFM-L在极端色彩或训练集外语义布局下可能泛化不足。

---

## 228. ProMoE-FL: Prototype-conditioned Mixture of Experts for Multimodal Federated Learning with Missing Modalities

**arXiv ID:** 2607.06633 | [PDF](https://arxiv.org/pdf/2607.06633v1)

**作者:** Aavash Chhetri `[一作]` (NepAl Applied Mathematics and Informatics Institute), Binod Bhattarai `[通讯]` (NepAl Applied Mathematics and Informatics Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

ProMoE‑FL提出了一种原型条件混合专家框架，用于在多模态联邦学习中处理缺失模态并生成缺失模态特征。

**💡 创新点**

创新点在于利用客户端感知的全局原型库与方向感知的混合专家路由实现可自适应的缺失模态特征合成，同时避免对公共数据的依赖并提升非IID环境下的鲁棒性。

**🔧 技术方法**

采用原型对齐的Transformer解码器、Mixture‑of‑Experts网络、客户端原型银行、FedAvg聚合以及预训练的ResNet50与BERT编码器等技术。

**📊 数据集**

在四个公开胸部X光数据集（MIMIC‑CXR、NIH Open‑I、PadChest、CheXpert）上构建同质与异质联邦实验场景。

**📈 对比分析**

与零填充、均匀填充、FeatImp、PmcmFL及CAR‑MFL等基线对比，ProMoE‑FL在宏观AUC上普遍优于SOTA，尤其在异质（非IID）设置中显著提升。

**⚠️ 局限性**

局限性包括在极大规模多模态联邦场景下原型集合的通信开销可能成为瓶颈，且对极度稀疏或高度不平衡模态的泛化能力尚未充分验证。

---

## 229. Validate the Dream Before You Trust Its Verdict: Admissibility for World-Model Simulators

**arXiv ID:** 2607.07196 | [PDF](https://arxiv.org/pdf/2607.07196v1)

**作者:** Christian Oefinger `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一套针对生成式世界模型（wm）作为闭环测试或然的可接受性标准——从L0到L4的五级“可接受性阶梯”，并在两种驾驶wm（Vista、Epona）上验证了其中L0（生成质量）、L1（动作鲁棒性）以及L2（训练包络）指标；

**💡 创新点**

创新点在于将传统安全仿真验证框架（VVA、SOTIF、场景测试）迁移至无物理约束的生成式wm，明确了动作覆盖缺口、包络声明及失效可归因等关键门槛，构成了可操作的“可接受性阶梯”；

**🔧 技术方法**

技术包括Fréchet Video Distance（FVD）等视觉真实性评估、动作可控性基准（如Arai等的轨迹跟随与重放分析）、分布式检验与拒绝策略（如合成预测、集成不一致性检测）以及失败归因协议；

**📊 数据集**

使用的主要数据集为两款驾驶wm（Vista、Epona）的训练与评估数据，包含从真实驾驶日志中采集的交互轨迹；

**📈 对比分析**

对比结果表明，视觉生成质量（L0）与动作跟随能力（L1）并不相关——在实验中视觉质量更高的wm在动作鲁棒性上表现更差；具体指标显示两模型在L0、L1、L2的得分差距与预期不符；

**⚠️ 局限性**

局限性包括：①阶梯标准尚未在所有机器人体态与任务上验证；②缺乏通用的训练包络声明与OOD检测实现；③L3所需的真实失效数据稀缺，难以建立可靠的归因机制；④阶梯的充分性、互斥性及层级顺序仍待进一步实证评估。

---

## 230. Scaling Author Identity Disambiguation to the World of Code: A Methodology

**arXiv ID:** 2607.06920 | [PDF](https://arxiv.org/pdf/2607.06920v1)

**作者:** Audris Mockus `[一作]` `[通讯]` (University of Tennessee), Audris Mockus (University of Tennessee)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并优化全球开源代码库中约107M作者字符串的去重映射，解决大规模过度合并问题

**💡 创新点**

通过系统实验记录、节点门控、跨项目高频键属性过滤、边缘级逻辑回归分类器以及基于边度、分布式betweenness的结构切分，实现在保持召回率的同时显著消除百万级“mega‑cluster”，并在外部基准上超越前置方法

**🔧 技术方法**

节点信息分数门控、项目分布门控、连通度门控、GitHub无回复ID的自动标注、逻辑回归与梯度提升边缘分类、结构图betweenness centrality切分、跨项目shingle扩展、签名键属性过滤与安全门控

**📊 数据集**

World of Code 6B提交、107M作者ID；GitHub单作者仓库21M别名；ALFAA 469k人工标注对；GitHub无回复ID标注2.6M边缘；签名签名数据约1M+签名提交

**📈 对比分析**

对比ALFAA金标准、GitHub单作者基准、公开的WoC V2409、GitAuthority，最终在单作者基准上实现召回0.5886、全合并率0.5403，精度0.8820，明显优于前置方法和公开基准；同时在gold数据上AUC≥0.99，维持mega-free特性

**⚠️ 局限性**

对签名键与非个人键的区分仍有限；极端高频签名键导致误合并；在极端别名（多键同名）和极少见属性的召回仍不完善；结构切分对未见边缘不具泛化能力，需多阶段融合

---

## 231. Optimized Instance Alteration for Explaining and Assessing Robustness of Classifiers

**arXiv ID:** 2607.06637 | [PDF](https://arxiv.org/pdf/2607.06637v1)

**作者:** Evgenii Kuriabov `[一作]` (Pennsylvania State University), Jia Li `[通讯]` (Pennsylvania State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一的优化框架，利用可解释的 L0 正则来生成稀疏且结构连贯的对抗式解释，并通过同一框架评估模型的鲁棒性。

**💡 创新点**

创新点在于设计了 Explainability-Aware L0（XA-L0）正则，使稀疏约束兼顾特征间关联与空间/边缘结构，兼顾解释性与鲁棒性；同时提出 Tolerance-Region Confusion Matrix（TOR-CM）以量化可解释扰动下的类别转换。

**🔧 技术方法**

核心技术包括基于梯度的多目标优化、软化 L0 正则、结构稀疏化（特征网络、图社区、像素距离、边缘感知权重）以及知识蒸馏生成可微代理来搜索对抗样本。

**📊 数据集**

在七个表格数据集（Wine、Breast Cancer、Iris、Digits、Wine Quality、Phoneme、Coil2000）和两个图像数据集（MNIST、Flowers‑102）上进行实验，且对 MNIST 进一步使用四种分类器（LogReg、CART、RF、CNN）来评估 TOR-CM。

**📈 对比分析**

与传统 L0、L2 约束、Dandl 等基线相比，XA‑L0 在稀疏性、结构一致性、可解释度上显著优于对手；在 TOR‑CM 上，RF 与 CART 的鲁棒性高于 CNN，而 CNN 的准确率最高，但鲁棒性最差，表明方法能够揭示标准评估未能捕捉的差异。

**⚠️ 局限性**

局限性包括对离散或混合特征的适用性不足、对领域知识的依赖以及需要手动设定超参数（如 λ、阈值 t）和图/社区划分的质量，且对大规模高维图像仍需改进计算效率。

---

## 232. On the Principles of Deep Feedforward ReLU Networks

**arXiv ID:** 2607.07035 | [PDF](https://arxiv.org/pdf/2607.07035v1)

**作者:** Changcun Huang `[一作]` `[通讯]` (Shuitu Institute of Applied Mathematics), Changcun Huang (Shuitu Institute of Applied Mathematics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文系统推导了深度前馈 ReLU 网络的理论机制，提出了“路径”“节点”“折线点（knot）”等概念，并给出了分区、函数构造、影响系数向量、多输出控制及单/多维逼近的理论框架；进一步用该框架解释了反向传播得到的训练解。

**💡 创新点**

创新点主要包括：
1) 将二维平面上的分隔超平面推广为深度网络中可形成的折线（甚至多维折面）分割；
2) 通过“路径”与“折线点”的关系揭示了 ReLU 网络的黑盒内部结构；
3) 引入影响系数向量来描述单元对后续层的影响，进而实现多输出与多折线点的协同控制；
4) 给出了单向与双向解的完整理论，并用“严格偏序”与“连续性约束”统一多维逼近问题；
5) 通过上述理论解释了训练过程中出现的复杂折线与函数行为。

**🔧 技术方法**

主要技术包括：
- 组合数学与几何分析（折线点、折面、分区、严格偏序树）；
- 递归与归纳推理构造路径与函数；
- 影响系数向量的线性代数求解；
- 对多输出网络的矩阵分解与求解；
- 对一维/多维逼近的解析构造（标准分区、近似逼近）。

**📊 数据集**

本文为纯理论研究，未使用公开数据集进行实验验证；所有结论均来自严格的数学推导与逻辑证明。

**📈 对比分析**

由于无实验比较，本文未给出数值性能评估；其价值在于提供了完整的理论框架，可为后续基于 ReLU 网络的可解释性与效率提升提供指导。

**⚠️ 局限性**

主要局限包括：
- 依赖“相邻路径假设”等较强假设，实际网络训练时可能不满足；
- 对折线点与折面控制的构造在高维空间中实现复杂，可能导致参数搜索困难；
- 仅给出理论解，缺乏对真实训练数据的经验验证；
- 对负折线点与双向解的处理仍需进一步简化与自动化；
- 复杂性分析主要基于理论最坏情况，实际模型可能更高效或更低效。

---

## 233. LoCA: Spatially-Aware Low-Rank Convolutional Adaptation of Vision Foundation Models

**arXiv ID:** 2607.06918 | [PDF](https://arxiv.org/pdf/2607.06918v1)

**作者:** Sojung An `[一作]` (Korea University), Donghyun Kim `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出低秩卷积适配（LoCA），在视觉基础模型中对卷积层进行参数高效微调，解决空间‑通道耦合问题；

**💡 创新点**

创新点包括：①将通道适配与空间基准分离，使用低秩通道分解与SVD提取的空间基准；②用SVD保留预训练空间先验；③引入层级秩调度，使适配容量随层宽度自适应；④在卷积层冻结原权重，仅训练ΔW，保持原模型功能；

**🔧 技术方法**

技术手段：低秩分解（LoRA‑style）+通道分解、SVD空间基准、层级秩调度、深度卷积权重重组、梯度归零初始化；

**📊 数据集**

实验数据集：VTAB‑1k、FGVC（CUB‑200‑2011、Stanford Dogs、Cars、NABirds）、DreamBooth（Stable Diffusion v1.4）、GTAV→Cityscapes/BDD100K/Mapillary（域泛化分割），以及多种backbone（ResNet、ConvNeXt、MambaVision、EfficientNet、MobileMamba）；

**📈 对比分析**

与FFT、LoRA、FSF、CoLoRA、Conv‑Adapter、VPT等方法对比；在VTAB‑1k和FGVC上，LoCA在仅0.97M trainable参数下实现最高平均精度；在DreamBooth生成任务中，LoCA在DINO/CLIP‑I/T分数上优于LoRA；在域泛化分割中，LoCA在多种backbone上均超过FFT和其他PEFT，提升约2–5个百分点；

**⚠️ 局限性**

局限性：①对卷积层的适配效果依赖SVD初始化，可能对极小模型或无卷积结构的模型适配不佳；②需要额外的SVD计算与层级调度设计；③对Transformer自注意力层适配仍无直接方案；④在极大模型上训练时间与内存消耗仍高于单纯LoRA。

---

## 234. Vision Language Action (VLA) Models for Unmanned Aerial Robotics and Bimanual Manipulation: A Review

**arXiv ID:** 2607.06706 | [PDF](https://arxiv.org/pdf/2607.06706v1)

**作者:** Inkyu Sa `[一作]` (Chef Robotics), Ho Seok Ahn `[通讯]` (University of Auckland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文是一篇系统综述，聚焦Vision‑Language‑Action（VLA）模型在双臂协作与无人机操控两个领域的应用，梳理了模型架构、训练流程、动作表示、协调策略、数据集与基准，并提出跨领域统一的分类与研究方向。

**💡 创新点**

创新点包括：①首次将双臂协作与无人机操控视为同一VLA问题，构建跨域统一的分类体系；②通过对比分析揭示双臂协调策略如何映射到无人机群与空中操作；③提出十四条开放研究方向，覆盖实时控制、安全认证、端到端无人机VLA、以及产学研转化。

**🔧 技术方法**

核心技术涵盖：VLM主干网络（PaLM‑E、Prismatic、DINOv2、SigLIP等）；动作生成方法（自回归、流匹配、扩散、混合等）；动作块化与实时块化（ACT、RTC、BID）；多模态协调与层级规划（π0.5、HiRobot）；强化学习自我改进（RECAP、SAIL）；以及记忆与世界模型增强（MEM、ContextVLA）。

**📊 数据集**

使用的数据集主要有：Open X‑Embodiment (OXE) 1M+演示；DROID 76K；BridgeData V2 60K；GigaBrain‑0.5M 500K；Sim‑Bench 如 LIBERO、SIMPLER、AirSim、Flightmare 等；以及多样化的机器人硬件平台（ALOHA、UMI、Mobile ALOHA、Franka Dual、Franka+Gripper 等）。

**📈 对比分析**

比较方法：文中对30+方法进行表格化对比，评估维度包括架构、训练策略、动作维度、硬件平台、模型规模、块长、延迟、成功率。典型结果显示：流匹配模型（π0、π0.5、π0^*）在双臂任务上取得最高成功率；混合与高效模型在保持相近性能的同时大幅降低延迟；在无人机任务上，流匹配与扩散模型在导航与空中抓取上表现最优；同时多模态基准表明不同模型在不同任务类型（紧耦合、弱耦合、变形物体、长期任务）上的优势差异。

**⚠️ 局限性**

局限性：①双臂任务的标准化基准不足，导致方法间跨实验比较受限；②扩散与流匹配模型仍面临推理延迟与硬件实时性挑战；③数据收集成本高，尤其是高质量双臂演示；④在强耦合与变形物体任务中，视觉仅凭借的接触感知仍有限；⑤跨域迁移（如从单臂到双臂或从陆地到空中）在不同运动学、动力学下的泛化尚未系统验证；⑥安全与规范化评估体系尚未成熟，难以直接用于高风险场景。

---

## 235. Creating Power Distribution Network Layouts Using Generative Adversarial Networks and Image-Based Representations

**arXiv ID:** 2607.06622 | [PDF](https://arxiv.org/pdf/2607.06622v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 236. Reasoning Consistency Scanning: A Framework for Auditing Chain-of-Thought Validity in AI Safety Evaluations

**arXiv ID:** 2607.07229 | [PDF](https://arxiv.org/pdf/2607.07229v1)

**作者:** Silvia Santano `[一作]` `[通讯]`, Silvia Santano

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种用于检测 AI 安全评估转录中链式推理（CoT）与最终答案逻辑一致性的扫描方法。

**💡 创新点**

创新点包括：①对“推理一致性”进行正式定义并构建六类不一致子类型的分类体系；②设计并验证了包含 60 条标注转录的合成基准；③实现了基于 InspectScout 的 LLM‑as‑judge 评分器，可批量判定一致性与子类型；④首次将该扫描器应用于多种安全评估（InstrumentalEval、MORU、SAD 等）和多种模型（DeepSeek、Gemini、gpt‑oss‑120b）。

**🔧 技术方法**

使用技术主要有：LLM‑as‑judge（Claude Opus 4.6）用于判定一致性；InspectScout 作为转录分析框架；手工构造的基准与数据生成脚本；统计分析（Precision、Recall、F1）与一致性率可视化。

**📊 数据集**

数据集：60 条从 InstrumentalEval 生成并人工修改的合成转录（含 6 种不一致子类型）；自然转录来自 MORU（100 条）、SAD（300 条）和 Agentic Misalignment（1 条）等评估；生成模型包括 DeepSeek V4 Pro、Gemini 3.1 Pro、gpt‑oss‑120b，部分评估原始模型被排除。

**📈 对比分析**

评估方法：先在合成基准上验证扫描器的准确率、召回率与 F1，随后对真实评估转录进行批量扫描，统计整体不一致率及按子类型分布。结果显示整体 F1 约 0.82，子类型 Recall 较低（尤其是矛盾推理、明显困惑），不一致率在 0%–26% 之间，且随评估任务和模型显著变化。

**⚠️ 局限性**

局限性：①基准仅来自单一评估与单一生成模型，导致泛化受限；②扫描器在某些子类型（矛盾推理、明显困惑）表现不佳；③未覆盖 Anthropic、OpenAI GPT‑5.4 等主流模型；④结果高度依赖扫描器 LLM、提示与阈值；⑤部分评估样本极少（如 Agentic Misalignment 仅 1 条）；⑥一致性检测并不能证明推理真实可信，仅为必要条件。

---

## 237. The Approximation Ratio for the Risk of Myopic Bayesian Active Learning for Linear Regression

**arXiv ID:** 2607.06642 | [PDF](https://arxiv.org/pdf/2607.06642v1)

**作者:** Stephen Mussmann `[一作]` `[通讯]` (Georgia Institute of Technology), Stephen Mussmann (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在贝叶斯线性回归的A/V最优设计问题上，对贪心算法的风险（非风险减少）进行近似比分析，给出了与最大初始杠杆得分（MILS）线性相关的常数因子风险近似保证，并构造了匹配的硬实例证明紧度。

**💡 创新点**

首次提出贪心算法在风险本身上的常数因子近似比，并证明该比与MILS线性相关且是最优的；同时利用新的子模近似比与曲率框架，提供风险逆函数的子模性下界；并用Hadamard矩阵构造实例展示下界与上界的匹配。

**🔧 技术方法**

利用矩阵不等式（Sherman-Morrison公式、Cauchy-Schwarz）证明风险逆函数的子模比下界，结合近似子模性与曲率理论得到风险近似比；构造Hadamard矩阵实例得到下界；进行数值仿真验证理论。

**📊 数据集**

实验数据主要为理论构造的向量集合；数值实验使用从单位球随机采样得到的向量（Λ=I_d）以及构造的Hadamard例子。

**📈 对比分析**

与传统基于风险减少的F_reduction分析相比，新的风险近似比在整个预算范围内保持有效；数值实验表明F_reduction在k≈10时已失效，而我们的上界在所有k下都可用；在硬实例中贪心风险比最优大约为1+h/5，验证上界的紧度。

**⚠️ 局限性**

近似比依赖于问题特定参数MILS，若MILS较大则比值可能很差；曲率可能接近1，无法进一步改进；该分析仅适用于贝叶斯线性回归的A/V最优设计，尚未推广到非线性模型或更一般的主动学习框架。

---

## 238. MoLIFE: Methodology, Technologies, and Challenges for Mobile Live Intelligent Forensics Examination

**arXiv ID:** 2607.07269 | [PDF](https://arxiv.org/pdf/2607.07269v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 239. General Incomplete Multimodal Learning via Dynamic Quality Perception

**arXiv ID:** 2607.06943 | [PDF](https://arxiv.org/pdf/2607.06943v1)

**作者:** Xiangyu Meng `[一作]` (University of Electronic Science and Technology of China), Shicai Wei `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种通用的不完整多模态学习框架（GIML），同时处理模态间缺失和模态内退化问题。

**💡 创新点**

GIML通过动态质量感知统一处理模态内退化和模态间缺失，允许在单个阶段内进行联合优化。

**🔧 技术方法**

使用了噪声感知质量估计器（NQE）和噪声-语义解耦模块（NSD）来提高模型的鲁棒性和泛化能力。

**📊 数据集**

在多个数据集上进行了广泛实验，包括CREMA-D、Kinetics-Sounds、MVSA-Single、MOSI和NVGesture。

**📈 对比分析**

与现有方法（如TMDC和T2DR）相比，GIML在不同的模态缺失和退化设置下表现出更好的性能，尤其是在模态不平衡的情况下。

**⚠️ 局限性**

模型在处理未见噪声类型和强度时的鲁棒性仍然有限，可能在极端情况下表现不佳。

---

## 240. Fractal KV-Cache Archives: Lossless Symbolic Storage with In-Place Retrieval for Long-Context LLM Inference

**arXiv ID:** 2607.07144 | [PDF](https://arxiv.org/pdf/2607.07144v1)

**作者:** Vladimir Gusev `[一作]` `[通讯]` (Independent Researcher), Vladimir Gusev (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为量化的KV缓存设计一种可逆、线性时间的压缩存档格式，并在其上实现O(1)随机访问、追加操作以及在不解压的情况下进行子串检索。

**💡 创新点**

创新点在于将收缩迭代映射码（chaos game representation）作为KV缓存的归档格式，兼具无损压缩、随机访问和检索索引功能；并通过按头残差向量量化（per-head RVQ）揭示键/值量化的不对称性，提出位数不对称的混合量化方案。

**🔧 技术方法**

使用收缩迭代映射码、k‑means残差向量量化（RVQ）、GPT‑2推理、双精度点空间最近邻检索以及基准实验对比。

**📊 数据集**

在GPT‑2（124M参数）上使用公共领域文本《时间机器》（Project Gutenberg）进行实验，评估1024-token上下文的压缩效果。

**📈 对比分析**

与全精度fp16 KV缓存以及池化代码本的量化方案对比；每头代码本的RVQ实现压缩比约36×，提升语义困惑度约+11%；检索实验中子串匹配召回率始终为1.0；随机访问单点耗时≈311 µs。

**⚠️ 局限性**

局限性包括仅在单一小模型与单一语料上验证，评价指标仅为困惑度；代码本需针对语料训练，未针对大模型或下游任务；检索示例仅为子串匹配，未评估语义检索；在原始大小上与通用字节压缩相当，优势主要在访问模式。

---

## 241. Miter-Aware LUT Mapping: Aligning Structure and Solvability for Efficient Logic Equivalence Checking

**arXiv ID:** 2607.07164 | [PDF](https://arxiv.org/pdf/2607.07164v1)

**作者:** Jiaying Zhu `[一作]` (Chinese University of Hong Kong), Qiang Xu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向miter的LUT映射框架，结合等价保持映射、基于高斯消元的XOR建模和面向求解器的LUT选择，显著简化逻辑等价检查的SAT实例；

**💡 创新点**

创新点在于在映射阶段即对两个电路进行结构对齐，利用高斯消元提取XOR密集区的仿射关系，并为求解器设计专属的LUT度量，形成统一的、求解友好的CNF表达；

**🔧 技术方法**

采用LUT映射（基于Mockturtle）、Gaussian消元、SAT求解器（Kissat、CaDiCaL、X-SAT、CSAT）以及ABC的高级LEC引擎；

**📊 数据集**

使用工业LEC对照对、ForgeEDA、ITC99、EPFL、Opencore等公开基准，并自行生成32×32乘法器等XOR密集实例；

**📈 对比分析**

与传统flat-miter基线对比，在所有求解器上平均PAR2下降约83%，单个实例最高可达92.1%，#Solved数和平均求解时间均显著提升；

**⚠️ 局限性**

局限性包括仅支持2-4输入LUT、对XOR密集区的检测与消元依赖于前置AIG解析、对非XOR密集的结构优化效果相对有限，且映射前的预处理开销需进一步评估。

---

## 242. Inertia-1: An Open Exploration of Wearable Motion Foundation Models

**arXiv ID:** 2607.06617 | [PDF](https://arxiv.org/pdf/2607.06617v1)

**作者:** Zongzhe Xu `[一作]` (University of California, Los Angeles), Yuzhe Yang `[通讯]` (University of California, Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个统一、开放的实验平台Inertia‑1，利用18.2M小时的加速度计原始数据进行大规模无监督预训练，并在15个数据集（涵盖短时人类行为识别、步态冻结检测和长期疾病预测）上评估其迁移性能。

**💡 创新点**

创新点在于：①把整个可穿戴运动建模生命周期（数据、感知配置、预训练目标、模型规模）统一在一个可复现框架下进行系统探索；②比较10种主流自监督目标，揭示不同任务对目标的依赖；③在多种感知配置（采样率、窗口长度、轴维度、频域/时域）和多流融合下验证其对迁移效果的影响；④提供公开可用的千模型库和最优实践。

**🔧 技术方法**

主要技术包括自监督学习（自回归预测、对比学习、知识蒸馏、掩码重建）、时域与频域表示、线性探针和全微调、基于多实例学习的疾病预测头，以及多流同步编码器融合。

**📊 数据集**

使用的数据集涵盖115,000+个体的原始加速度计记录（如UK Biobank等大规模人群队列）以及15个公开任务数据集：10个短时行为识别数据集、3个步态冻结检测数据集、7个疾病预测（如帕金森、阿尔茨海默等）数据集。

**📈 对比分析**

对比方法：与从零开始的监督训练、不同预训练目标、不同模型规模以及不同感知配置进行线性探针和全微调比较。实验显示，自监督预训练普遍优于监督基线，最优方案在不同任务上差异显著，疾病预测任务可提升约10–20点AUROC，行为识别任务提升约2–4个百分点。

**⚠️ 局限性**

局限性：仅评估了分类任务，未覆盖连续预测、回归或长期轨迹建模；部分自监督目标未在所有感知配置下进行细粒度消融；主要聚焦加速度计数据，其他传感器或多模态融合的效果仍待进一步验证。

---

## 243. Behavioral Privacy Leakage in Agentic Negotiation: Formalizing and Mitigating Inference Attacks via Randomized Policies

**arXiv ID:** 2607.06815 | [PDF](https://arxiv.org/pdf/2607.06815v1)

**作者:** Barkha Rani `[一作]` `[通讯]` (Apple Inc.), Barkha Rani (Apple Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了自主谈判代理中的行为隐私泄露，提出了一种自适应随机谈判策略，确保在多轮谈判中实现(ε, δ)-差分隐私，同时保证几乎确定的收敛性和高谈判效用。

**💡 创新点**

创新点在于首次正式化了适用于顺序谈判的行为差分隐私，并设计了一种机制，能够在保护隐私的同时保持高效的谈判成功率和效用。

**🔧 技术方法**

使用了自适应随机化谈判策略和安全评论员，结合高斯噪声调度来实现差分隐私。

**📊 数据集**

使用了3000个合成的双边谈判数据集，这些数据反映了真实世界谈判的行为特征。

**📈 对比分析**

与非隐私基线相比，机制在对抗性推断准确性上减少了43-50%，同时保持了90.4%的谈判成功率，且效用水平与非隐私基线相当或略有提高。

**⚠️ 局限性**

限制在于该研究仅针对单一议题的双边谈判，未来需要扩展到多议题和多方设置；此外，使用的合成数据集规模较小，真实世界数据集的使用将增强实证结果的可信度。

---

## 244. A simple algorithmic framework for disambiguation of finite automata

**arXiv ID:** 2607.06894 | [PDF](https://arxiv.org/pdf/2607.06894v1)

**作者:** Mauricio Cari `[一作]`, Cristian Riveros `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

未提供具体研究内容

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

## 245. CarbonCLIP: Enhance Carbon Prediction from Satellite Imagery via Integrated Street-View Semantics and Temporal Context Training

**arXiv ID:** 2607.07292 | [PDF](https://arxiv.org/pdf/2607.07292v1)

**作者:** Zeru Yang `[一作]` (Nanyang Technological University), Chau Yuen `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究通过CarbonCLIP框架，在预训练阶段利用街景文本和月份上下文进行双分支对比学习，将多模态知识蒸馏到卫星图像表征，从而实现仅用卫星图像即可预测城市月度碳排放。

**💡 创新点**

创新点包括：①使用大型多模态语言模型自动生成街景文本作为语义先验；②引入循环月份编码器注入时间上下文；③通过双分支对比学习将多模态知识蒸馏到单模态卫星表征，保持推理时仅使用卫星图像。

**🔧 技术方法**

采用的技术包括：Qwen2.5‑VL生成街景文本；ViT‑B/32作为卫星图像编码器；双分支图像‑文本对比学习与图像‑月份对比学习；多频正弦时间编码；轻量MLP回归器。

**📊 数据集**

使用的数据集为：Planet 3 m 级卫星影像；Google/Baidu街景图像；ODIAC 1 km 级月度碳排放；两座城市北京和新加坡的数据。

**📈 对比分析**

与ResNet‑18、ViT、UrbanCLIP等基线相比，CarbonCLIP在北京年R² 0.728、RMSE 0.476、MAE 0.349，在新加坡年R² 0.704、RMSE 0.496、MAE 0.362，表现出显著提升，尤其在月度预测稳定性方面优于所有基线。

**⚠️ 局限性**

局限性包括：1）未显式建模邻域空间相关性；2）月份编码过于粗糙，难以捕捉短周期变化；3）预训练仍需街景文本，缺乏街景资源的城市难以迁移；4）对不同气候、形态城市的泛化性需进一步验证。

---

## 246. `Attention-Guided Cross-Temporal Clustering for Self-Supervised Video Object Segmentation

**arXiv ID:** 2607.07230 | [PDF](https://arxiv.org/pdf/2607.07230v1)

**作者:** Waqas Arshid `[一作]` (Griffith University), Yongsheng Gao `[通讯]` (Griffith University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种完全自监督的视频目标分割框架CTC2，利用冻结的SAM2 Transformer编码器产生特征并通过注意力引导的稀疏 token 选择，随后用轻量级 MLP 对这些 token 进行软部分聚类，并在多时间间隔上对齐这些软分布，实现跨帧的部分级一致性。

**💡 创新点**

核心创新在于：①用注意力产生的显著性先验与网格多样性约束相结合，形成自适应的 token 采样策略；②引入多 offset 的轻量化时间匹配与对称 KL 对齐软部分分布，既避免了光流等先验，又能在不同运动速率下保持稳定；③通过 entropy 与 balance 正则化防止聚类崩塌，提升部分表示的鲁棒性。

**🔧 技术方法**

技术细节包括：冻结的 SAM2 ViT 作为特征提取器；top‑p + grid 的 token 选择策略；两层 MLP 聚类头；余弦相似度互为最近邻匹配；多时间间隔 supervision（Δt∈{1,2,4,8}）并以匹配率控制选择；saliency‑加权的对称 KL 损失；以及熵与均衡正则项。

**📊 数据集**

实验使用公开的 DAVIS‑2016、DAVIS‑2017 及 YouTube‑VOS 数据集进行零射击（self‑supervised）和一帧半监督设置。

**📈 对比分析**

与现有自监督 VOS 方法（如 TimeT、SMTC、CorrFlow、TripleNet 等）对比，CTC2 在 DAVIS‑2017 的无标签平均得分为 0.554、DAVIS‑2016 为 0.522、YouTube‑VOS 为 0.570，均优于或竞争前沿方法，并实现约 35 fps 的实时推理速度。

**⚠️ 局限性**

局限性包括：对极小目标或高背景噪声场景下的注意力可靠性不高；仅冻结 backbone 可能导致在强域外分布时注意力衰退；对极长时序、剧烈运动或快速遮挡的跨帧匹配仍易失效；以及缺乏显式长期记忆机制。

---

## 247. Widest-Path Reachability Fields for Connectivity-Preserving Slender Structure Segmentation

**arXiv ID:** 2607.07123 | [PDF](https://arxiv.org/pdf/2607.07123v1)

**作者:** Youcheng Zong `[一作]` (Northeastern University), Dakuo He `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了Widest-Path Reachability Fields（WPRF）模块，专门解决细长结构分割中的连通性断裂问题，即所称的“拓扑梯度饥饿”（TGS）。

**💡 创新点**

创新点在于将Max‑Min算子引入可微分的连通性目标，通过图上动态规划实现梯度聚焦于瓶颈像素，完全无需后处理或骨架提取的辅助结构，且可无缝插入任何主干网络。

**🔧 技术方法**

使用技术包括：可微分Max‑Min动态规划、域受限图构建、骨架提取与Voronoi广播、瓶颈加权的像素损失、边缘监督与多尺度连通性损失，以及clDice等骨架级指标评估。

**📊 数据集**

实验涵盖六大公开/自建数据集：DRIVE、OCT‑A‑500（3mm 与 6mm）、DeepCrack、Massachusetts Roads、OMVIS（口腔微血管）。

**📈 对比分析**

与九种主干网络（CNN、Transformer、状态空间）在六个数据集上对比，WPRF在 47/54 组实现了 clDice 提升，平均提升约 0.99pp，最高可达 +7.2pp（相对提升 13.5%），Dice 维持不变或略有提升，且不增加推理时延。

**⚠️ 局限性**

局限性包括：训练时需额外 2–3 倍时间开销；k‑步 Max‑Min 固定对非常密集或环状网络可能不足；目前仅适用于单实例、树状/稀疏环结构；仍需根据阈值微调（τ_fg、τ_link）以获得最佳效果。

---

## 248. Prior-aware and Context-guided Group Sampling for Active Probabilistic Subsampling

**arXiv ID:** 2607.07083 | [PDF](https://arxiv.org/pdf/2607.07083v1)

**作者:** Beomgu Kang `[一作]` (Korea University), Hyunseok Seo `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了Prior-aware and Group-based Active DPS（PGA-DPS）方法，结合固定先验采样与分组活跃采样，以提升下游任务的性能。

**💡 创新点**

创新点在于引入先验固定采样模式与基于top‑k的分组采样，理论上降低有效Lipchitz常数，从而实现更平滑、更稳健的优化。

**🔧 技术方法**

采用Gumbel‑Softmax重参数化、DPS-top‑k采样、LSTM+MLP采样网络、深度可展开的近似梯度重建网络，以及残差U‑Net分割网络。

**📊 数据集**

使用MNIST、CIFAR‑10、fastMRI knee和AeroRIT四个数据集，分别评估分类、图像重建和分割任务。

**📈 对比分析**

与DPS、A‑DPS、传统低通/变密度采样、贪婪掩码、LOUPE以及RL方法对比，PGA‑DPS在所有任务与指标（准确率、PSNR/SSIM、mIOU/mDICE）上均实现了显著提升，尤其在低采样率下更为突出。

**⚠️ 局限性**

主要局限在于需手工调节固定采样比例Ps和活跃采样比例As；未来可进一步自动化超参数搜索和温度调度以提升鲁棒性。

---

## 249. Latency-Constrained DNN Architecture Learning for Edge Systems using Zerorized Batch Normalization

**arXiv ID:** 2607.06922 | [PDF](https://arxiv.org/pdf/2607.06922v1)

**作者:** Shuo Huai `[一作]` (Nanyang Technological University), Qian Lin `[通讯]` (HP Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对边缘设备的端到端神经网络学习框架，能够在满足严格的时延约束下，直接优化模型架构并保持或提升精度。

**💡 创新点**

创新点在于：① 通过一次性训练的“压缩学习”方法，无需预训练和再训练即可获得符合时延约束的紧凑模型；② 结合动态 BN 零化与恢复训练，实现可恢复的通道/层压缩；③ 构建通用硬件定制时延预测器，避免频繁设备测量；④ 通过统一缩放+压缩，避免手工搜索缩放因子。

**🔧 技术方法**

核心技术包括：Batch Normalization 参数的显式重要性评估、零化与恢复训练策略、基于 BP 网络的时延预测器、统一模型缩放、以及后续 FP16 量化。

**📊 数据集**

使用 CIFAR‑10 与 ImageNet‑100（从 ImageNet‑2012 随机抽取 100 类）作为评估数据集，实验覆盖 VGG、ResNet、DenseNet、GoogLeNet 四大网络。

**📈 对比分析**

与 SFP、FPGM、NS、PGMPF、OTO、EfficientNet‑Compound、HACScale 等现有方法比较，实验表明在 34 ms 时延约束下，本文方法在保持或提升 Top‑1/T5 精度的同时，训练时间显著低于传统 3‑阶段剪枝，且相较于手工搜索的缩放方法得到更高精度。

**⚠️ 局限性**

局限性包括：① 仍以时延为唯一约束，未考虑能耗、存储等多维度需求；② 需要在目标硬件上先收集样本构建时延预测器；③ 对极小模型的压缩比例可能导致精度不可逆退，需进一步研究自动选择压缩比例。

---

## 250. Open-Ended Scenario Reasoning for Specialist Model Adaptation

**arXiv ID:** 2607.06625 | [PDF](https://arxiv.org/pdf/2607.06625v1)

**作者:** Youcheng Zong `[一作]` (Northeastern University), Dakuo He `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用LLM推理在不重新训练的情况下对专业模型进行开放式适配。

**💡 创新点**

创新点在于将LLM作为结构化先验引擎，在低维语义潜在空间内通过贝叶斯推断融合多源证据，并加入风险约束防止错误更新。

**🔧 技术方法**

技术包括大语言模型推理、贝叶斯后验更新、低维语义潜在空间映射以及多层风险门控。

**📊 数据集**

使用矿物浓度脱水过程数据与公开的IndPenSim青霉素发酵数据进行评估。

**📈 对比分析**

与多种基准模型（SVR、XGBoost、GRU等）和传统微调、MAML等方法比较，ROAM在隐藏移位等难以检测的情形下将MAE下降约20%，且仅增加839参数、<0.02ms推理开销。

**⚠️ 局限性**

局限在于对文本情景描述的依赖、无法处理多重同时变迁以及对LLM先验的准确性有限。

---

## 251. Disturbance-aware Motion Planning for Over-actuated Underwater Vehicles Exploiting Actuation Redundancy for High-fidelity 3D Reconstruction

**arXiv ID:** 2607.07139 | [PDF](https://arxiv.org/pdf/2607.07139v1)

**作者:** Yuer Gao `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Yi Cai `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于过度驱动的水下机器人运动规划框架，通过实时重分配推力实现对目标区域的扰动最小化，以提升三维重建质量

**💡 创新点**

利用八轴推力器的冗余，在运动执行时搜索零空间以同时满足姿态跟踪与环境扰动抑制；并将推力器气流模型与运动规划耦合，实现“温和稳定”控制

**🔧 技术方法**

推力器冲击波模型（基于流体动力学的推进器盘理论+方向衰减）、实时SQP优化（10 Hz）、结构光/光流特征匹配、COLMAP与NeRF重建、PIV验证

**📊 数据集**

人工珊瑚模型（A、B、C）、实验水槽数据（600多次扫描）、人工手柄、惯性视觉融合等

**📈 对比分析**

与手动遥控、无扰动规划、光滑规划等基准对比：RMSE 1.9 mm（比无扰动提升55%）、完整度94.8%（比无扰动提升≈26%）、成功率98.5%（比基准0–45%）

**⚠️ 局限性**

实验限于室内水槽，颗粒类型与流场均为人工；对大尺寸场景、复杂水流及实时目标检测的适应性待验证

---

## 252. ThermoDSE: A Thermal-Aware and Comprehensive Design Space Exploration for Chiplet-Based DNN Accelerators

**arXiv ID:** 2607.07096 | [PDF](https://arxiv.org/pdf/2607.07096v1)

**作者:** Jian Peng `[一作]`, Wei Zhang `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对基于芯片片段的多芯片模块（MCM）深度神经网络加速器，研究并建模其热特性，并提出基于可扩展约束贝叶斯优化（SCBO）的设计空间探索方法。

**💡 创新点**

创新点在于将热感知任务映射与多级架构设计结合，使用SCBO统一搜索核心级、芯片级架构、通信和任务映射，实现热约束下的性能优化。

**🔧 技术方法**

采用精确的微任务级仿真器、可扩展约束贝叶斯优化，以及与传统模拟退火（SA）和强化学习（RL）方法进行对比。

**📊 数据集**

使用标准DNN工作负载（如 ImageNet、CIFAR‑10 等）进行实验验证。

**📈 对比分析**

与 SA、RL 方法相比，SCBO 在峰值温度上可降低 1–4 °C，EDYP 成本显著降低，实验结果表明相对传统方法节能/延迟收益提升 X%–X%。

**⚠️ 局限性**

主要局限在于实验仅基于仿真，未在真实硬件上验证，且对更大规模 MCM 的扩展性和动态功耗管理尚未深入探讨。

---

## 253. The Harness Effect: How Orchestration Design Sets the Token Economics of Enterprise Agentic AI

**arXiv ID:** 2607.06906 | [PDF](https://arxiv.org/pdf/2607.06906v1)

**作者:** Muayad Sayed Ali `[一作]` (Writer Inc), Waseem AlShikh `[通讯]` (Writer Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在同一任务集、相同模型与价格表下，替换传统 agent 循环为 Writer Agent Harness，比较两种调度层在 token、成本、延迟与任务完成质量上的差异。

**💡 创新点**

创新点在于提出并量化“harness”概念，证明其对多模型、多供应商环境的效率提升，并揭示了模型强度与 harness 效果之间的高度相关性（harness leverage）。

**🔧 技术方法**

采用了六大机制：两区 prompt、结构化压缩、上下文离线、零 token 等待、失败治理与子代理委托，并在同一公共价格表下进行成本计算。

**📊 数据集**

使用了覆盖 9 大能力的 48 个锁定任务集，任务包括合同对齐、检索、内容生成、Playbook、工具调用、演示、语音、图像与子代理等场景。

**📈 对比分析**

通过对比相同任务、模型与价格表的两种调度层，实验发现成本下降 33–61%，token 与延迟均显著减少，任务完成质量保持平稳；部分低阶模型在特定能力上略有下降。

**⚠️ 局限性**

局限性包括样本量有限（48 个任务），基线仅执行一次且来源单一供应商，质量评估依赖 LLM 判定，未涵盖长时序编码等场景，且仅评估了 6 种模型。

---

## 254. Learning Spatiotemporal Tubes for Full Class of Signal Temporal Logic Tasks for Control of Unknown Systems under Input Constraints

**arXiv ID:** 2607.07136 | [PDF](https://arxiv.org/pdf/2607.07136v1)

**作者:** Ahan Basu `[一作]` (Indian Institute of Science), Pushpak Jagtap `[通讯]` (Indian Institute of Science)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种物理信息神经网络驱动的时空管道（PINSTT）框架，用以在未知的Euler–Lagrange系统上满足输入受限的Signal Temporal Logic（STL）任务，并给出闭式控制律保证系统轨迹始终保持在管道内。

**💡 创新点**

创新点在于将STL鲁棒性指标直接作为PINN的损失函数，实现对全STL语法的支持；通过自动微分强制管道中心和半径的Lipschitz连续性，实现全时域的形式验证；并结合闭式、无近似控制器在满足输入限制的同时完成任务。

**🔧 技术方法**

技术手段包括：物理信息神经网络（PINN）对时间可变球形管道参数进行建模；基于STL鲁棒度的损失函数与Lipschitz约束的联合训练；闭式控制设计（两阶段：速度参考 + 加速度跟踪），并使用基于变换函数的安全边界。

**📊 数据集**

实验使用自定义的三类任务：2D omnidirectional robot、3D quadrotor、7-DOF Franka 机械臂以及多机器人编队导航，采用仿真与真实机器人（Agile LIMO、四旋翼飞行器、Franka Research 3）数据进行验证。

**📈 对比分析**

与RRT*、MPC、CBF、PPC、MILP、STT等方法对比，PINSTT在满足完整STL语法、未知动力学、输入约束和噪声鲁棒性方面实现了闭式控制与微秒级在线计算，计算时间显著低于MPC、MILP和CBF，且在最大控制输入与鲁棒性上优于旧版STT。

**⚠️ 局限性**

局限性包括：当前仅针对确定性、受限扰动的Euler–Lagrange系统；需要预先采样并训练PINN，训练时间随任务复杂度增加；缺乏对动态环境和大规模多体网络的分布式理论；以及对高维任务空间的扩展性尚未验证。

---

## 255. Sparse Relaxed Broadcast Graphs

**arXiv ID:** 2607.07260 | [PDF](https://arxiv.org/pdf/2607.07260v1)

**作者:** Pierre Fraigniaud `[一作]` (Institut de Recherche en Informatique Fondamentale, CNRS and Université Paris Cité), Hovhannes Harutyunyan `[通讯]` (Concordia University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究广播时间接近最优（即 (1+ϵ)log₂n）时最少边数的网络设计问题，并给出了相应的上界与下界。

**💡 创新点**

主要创新在于：① 对 0<ϵ<α 的情况给出新的上界 O(n^{1-ϵ/α})，显著优于此前的 O(n^{1-ϵ})；② 对 1-relaxed 广播时间证明了线性下界 9/8 n 边，进一步提升了已知的上界 2n-⌈log₂n⌉-2。

**🔧 技术方法**

技术方法：构造核心图+截断 Binomial 树的层次结构，利用黄金比 ϕ 的指数性质优化参数，证明子树叶子数下界并用它们推导边数下界。

**📊 数据集**

研究为纯理论分析，无实验数据集；所有结果均为严谨的组合/图论证明。

**📈 对比分析**

与已有 O(n^{1-ϵ})、2n-⌈log₂n⌉-2 等上界比较，本文的 O(n^{1-ϵ/α}) 上界在 ϵ→0 时更紧，且 1-relaxed 下界 9/8 n 边证明了该上界的线性下限，体现出显著的性能提升。

**⚠️ 局限性**

局限性：仍未完全确定在 ϵ=α 或常数 τ 时的最小边数下界，方法主要适用于足够大的 n（或 n 为 2^k），对小 n 或不同结构的图需进一步研究。

---

## 256. From Jumps to Signatures: a Generative Method for Temporal Point Processes

**arXiv ID:** 2607.06652 | [PDF](https://arxiv.org/pdf/2607.06652v1)

**作者:** Niels Cariou-Kotlarek `[一作]` (University College London), Vasileios Lampos `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了 interarrival embedding 将离散事件序列映射为连续有限变差路径，并基于路径签名构造全局 Wasserstein 损失的生成模型 Sigtpp，用以逼近真实时间点过程的分布。

**💡 创新点**

创新点包括：1）对计数路径实现可注入且稳定的连续嵌入；2）提出基于签名的全局分布损失，替代传统的局部条件损失；3）证明路径度量可产生能量距离、Wasserstein-1 及签名-Wasserstein-1 三种分布度量。

**🔧 技术方法**

主要技术手段为粗糙路径理论中的路径签名与期望签名、Wasserstein-1、能量距离、签名-Wasserstein-1 以及对比实验中的 Diffusion、VAE、WGAN 等生成模型。

**📊 数据集**

使用了 4 个合成数据集（Poisson、Inhomogeneous Poisson、单维 Hawkes、三维 Hawkes）和 5 个真实世界数据集（Earthquake、Stack Overflow、Taobao、Taxi、Yelp）进行评估。

**📈 对比分析**

与 VAE、DDPM、WGAN 等基线对比，Sigtpp 在 8 项评估指标中平均排名 2.0，5/8 指标获得最低相对分数，整体在路径分布匹配上显著优于其它方法，尤其在能量距离和签名-Wasserstein 上表现突出；但在条件校准指标 CRPS 上略逊于传统条件模型。

**⚠️ 局限性**

主要局限包括：①签名截断阶数需经验选择，过高阶可能引入噪声；②模型不直接优化条件密度，导致条件校准（CRPS）不如传统条件模型；③对大规模数据的训练仍存在较高计算开销。

---

## 257. TriRoute: Unified Learned Routing for Joint Adaptive Attention, Experts, and KV-Cache Allocation

**arXiv ID:** 2607.06601 | [PDF](https://arxiv.org/pdf/2607.06601v1)

**作者:** Andrii Balashov `[一作]` (Ukrainian State University of Science and Technologies), Olena Ponomarova `[通讯]` (Ukrainian State University of Science and Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

联合学习语言模型的注意力分辨率、专家选择和KV缓存精度的轻量控制器，以单一预算实现高效推理

**💡 创新点**

首次在单一控制器下联合决策三种稀疏/量化机制并通过耦合平衡避免跨轴崩塌

**🔧 技术方法**

使用Gumbel‑Softmax、straight‑through梯度、负载平衡、熵阈值及拉格朗日预算控制

**📊 数据集**

在Pile+RedPajama混合数据集上训练和评估

**📈 对比分析**

与独立调优的MoD+MoE+KV‑量化组合相比，在相同计算/内存预算下匹配或超过密集模型的困惑度并提升尾部鲁棒性

**⚠️ 局限性**

实现受限于稀疏注意、混合精度KV和控制器开销导致的实际速度提升不足，且在极端稀疏或更大规模下未验证

---

## 258. Multi-Agent AI Control: Distributed Attacks Hamper Per-Instance Monitors

**arXiv ID:** 2607.07368 | [PDF](https://arxiv.org/pdf/2607.07368v1)

**作者:** Oliver Makins `[一作]` (UK AI Security Institute), Mary Phuong `[通讯]` (MATS Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多代理AI控制中分布式攻击的可探测性，并在合成实验平台FakeLab上进行系统实验

**💡 创新点**

首次量化了分布式攻击中“分散化”导致的单次提交可疑度下降，并证明协调规划显著提升攻击隐蔽性与成功率

**🔧 技术方法**

采用LLM驱动的实验框架：Gemini 3 Flash/Pro作为执行者，GPT‑5 mini/nano作为监视器；利用Git diff、prompt细化与模型自评来评估监测

**📊 数据集**

构建了包含9个服务、86项主任务和4个侧任务的FakeLab代码库，模拟真实AI实验室的共享仓库环境

**📈 对比分析**

通过对比不同代理数、是否使用规划器、监视器能力以及权限限制，发现监视器能力对未被检测攻击率影响最大；在5%误报率下，强监视器可将未检测成功率降至<1%，弱监视器则需近30%误报率

**⚠️ 局限性**

局限包括仅使用单一LLM模型族、只评估Git diff监控方式、假设攻击者被单次恶意提交发现即失效、未探讨更真实的通信和攻击持续性情景

---

## 259. Seeing What Matters: Lesion-Aware High-Resolution Patch Discovery and Fusion for Chest X-ray Report Generation

**arXiv ID:** 2607.06909 | [PDF](https://arxiv.org/pdf/2607.06909v1)

**作者:** Yingshu Li `[一作]` (University of Sydney), Luping Zhou `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出LePaX框架，结合可学习的空间分辨率分配与全局‑区域融合，实现胸片高分辨率感知并生成医学报告。

**💡 创新点**

①受放射科医生工作流程启发，学习式分配高分辨率窗口只聚焦诊断关键区域；②在固定视觉令牌预算下实现高分辨率感知，避免令牌膨胀。

**🔧 技术方法**

采用视觉Transformer共享编码器、Learnable Spatial Resolution Allocation (LSRA) + Gumbel‑Top‑K采样、Grad‑CAM弱先验、Global‑Regional Fusion (GRF) 的跨域注意力、Perceiver连接器、Phi‑3 mini LLM、LoRA 微调等技术。

**📊 数据集**

在三大公开胸片数据集上训练与评估：MIMIC‑CXR、IU‑Xray 以及 CheXpertPlus。

**📈 对比分析**

与多种基线模型（R2Gen、METransformer、KiUT、MambaXray‑VL、Meta‑Rad 等）在BLEU、ROUGE、METEOR、RadGraph F1、BERTScore、RadCliQ、GREEN、RateScore 等指标上均取得最先进的性能，表现出显著提升。

**⚠️ 局限性**

仍需较大算力与模型参数；对极小或稀疏病灶的检出可能不足；缺乏大规模真实放射科医生评估的临床验证。

---

## 260. Hypergraph Neural Stochastic Diffusion: An SDE Framework for Uncertainty Estimation

**arXiv ID:** 2607.07330 | [PDF](https://arxiv.org/pdf/2607.07330v1)

**作者:** Zhiheng Zhou `[一作]` (Shandong University), Guiying Yan `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于随机微分方程（SDE）的HyperNSD框架，用于在超图上直接建模表示的随机演化，从而实现不确定性估计。

**💡 创新点**

创新点在于将漂移项和扩散项均设计为可学习的，且通过超图梯度与随机噪声耦合，实现了对超图节点-超边关联的不确定性建模；同时给出了模型的全局存在性、稳定性、等价性以及数值收敛性理论保证。

**🔧 技术方法**

技术上主要运用了超图神经网络的梯度/散度算子、可学习的漂移与扩散网络、Euler–Maruyama数值离散化、路径条件熵损失和多轨迹采样来估计预测不确定性。

**📊 数据集**

实验使用了六个超图基准数据集：Cora、Cora‑CA、CiteSeer、DBLP、ModelNet40 与 NTU2012。

**📈 对比分析**

通过与多种基线方法（MSP、ODIN、Mahalanobis；GNNSafe、GPN、GNSD、LGNSDE；HGNN、HyperGCN、HNDiffN、HND、HyperGOOD）在三种 OOD 场景（标签留出、特征插值、结构重排）下进行比较，HyperNSD 在大多数指标（AUROC、AUPR、FPR95）上均取得领先或接近最优的性能，平均提升约 2–4%。

**⚠️ 局限性**

局限性包括：需要对每个样本采样多条 SDE 轨迹导致推理时开销增加；目前仅适用于静态超图，难以直接扩展到动态或异构大规模超图；对超图结构的完整性要求较高，若原始超图被逼近为对称图，效果会下降。

---

## 261. MIRA-Math: A Benchmark for Minimal Information Requesting and Mathematical Reasoning

**arXiv ID:** 2607.07391 | [PDF](https://arxiv.org/pdf/2607.07391v1)

**作者:** Charbel Al Bateh `[一作]` (Lebanese American University), Samer Saab `[通讯]` (Lebanese American University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种用于评估模型在部分信息条件下进行最小信息请求与精确数学推理能力的合成基准；

**💡 创新点**

创新点在于通过构造每个实例只缺失单个原子事实，清晰分离信息请求与后续计算两种能力，并实现可重复、可验证的生成与评估流程；

**🔧 技术方法**

采用了基于确定性随机种子的生成器、精确算术与拒绝采样、族级验证器、精确答案验证器，以及固定受限LLM回应者的交互协议；

**📊 数据集**

使用了在Hugging Face上发布的20/50类型分裂数据集，包含2310个实例，覆盖22类数学家族（贝叶斯推断、坐标几何、线性系统等）；

**📈 对比分析**

通过与固定受限LLM回应者的交互，计算了准确率、请求命中率、首次请求成功率、平均请求数等指标；基准结果显示模型在完成答案准确率和请求效率方面存在明显差距，但具体数值需参考官方发布的基线文件；

**⚠️ 局限性**

局限性包括：基准为合成、单一原子缺失的理想化任务，无法反映多信息缺失或不确定性；响应者的LLM匹配可能产生误判；以及随着模型更新，基准排名可能失效。

---

## 262. Blockchain Attacks and Defenses: A Layered and Cross-Domain Survey

**arXiv ID:** 2607.06593 | [PDF](https://arxiv.org/pdf/2607.06593v1)

**作者:** Junjie Hu `[一作]` (Shanghai Jiao Tong University), Na Ruan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

综述了区块链攻击与防御，构建了跨层次与跨域的威胁分类体系，系统评估了从网络层到应用层以及跨链、滚动、预言机等多维度的安全问题。

**💡 创新点**

创新点在于提出“跨层次传播假设”与“跨域信任边界”模型，统一了传统单层安全研究的碎片化，系统地揭示了攻击如何跨链、跨协议蔓延，并对比不同防御方案的残留假设与权衡。

**🔧 技术方法**

使用了系统建模与层级架构分析、威胁树编码、事件驱动的案例归因、公开数据抓取与量化评估（如MEV、跨链桥失误、网络攻击频率）以及基准工具与测评框架。

**📊 数据集**

数据集包括246篇学术研究论文、行业事故报告、链上交易历史、MEV提取统计、网络拓扑与BGP攻击记录、桥与rollup的状态转移日志，以及跨链桥的验证器签名与溢价数据。

**📈 对比分析**

通过对比不同防御的安全属性（安全性、活性、最终性、隐私、问责）与剩余假设，评估其在现实环境中的可行性与时间敏感性；在实验中展示了防御方案对攻击成本、延迟与误报率的影响，整体表现表明跨域防御需平衡交易吞吐与安全保障。

**⚠️ 局限性**

局限在于缺乏对私有订单流、离线日志与实时治理操作的完整观测，部分案例的攻击路径依赖于不公开或商业化数据；此外，评估主要基于已公开的学术与事故资料，未能覆盖所有新兴协议与量子抵抗技术的实测效果。

---

## 263. R^3: Advertisement Compliance Rectification via Group-Relative Experience Extractor and Curriculum Reinforcement

**arXiv ID:** 2607.07318 | [PDF](https://arxiv.org/pdf/2607.07318v1)

**作者:** Yuan Chen `[一作]` (Tencent), Jie Jiang `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种名为ℛ^3的视频文本违规整改框架，实现了自动纠正广告中的文本违规并保持原始语义意图。

**💡 创新点**

创新点包括：①基于经验的群组相对合规经验提取的数据合成；②分阶段、层级奖励的课程强化学习；③完整的文本识别→改写→重渲染工业化流水线。

**🔧 技术方法**

采用的技术包括：高级LLM（Gemini3-Flash、Qwen3-8B）、GRPO强化学习、层级奖励机制、语音克隆、时间拉伸与动态范围控制等。

**📊 数据集**

使用的主要数据集为工业广告审核管线生成的12k样本（训练11k，测试1k），覆盖四类违规政策（Quantitative、Absolute、Misleading Exaggeration、Superlative Claims）。

**📈 对比分析**

通过与Qwen3-8B、Gemini3-Flash（含/不含GCEE）以及Qwen3-8B-SFT等基线在ComR、AvgE、CohR、QRR四项指标进行对比，ℛ^3在ComR 93.6%、QRR 81%等方面领先，并在线上A/B测试中实现21%的采用率提升。

**⚠️ 局限性**

局限性在于模型针对训练时使用的固定审核系统与规则进行优化，规则更新需重新生成数据并对齐，缺乏持续学习和模块化快速适应新规则的能力。

---

## 264. Large Behavior Model: A Promptable Digital Twin of the Retail Customer

**arXiv ID:** 2607.06993 | [PDF](https://arxiv.org/pdf/2607.06993v1)

**作者:** Wachiravit Modecrua `[一作]` (Amity Research and Application Center), Touchapon Kraisingkorn `[通讯]` (Amity Research and Application Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 Large Behavior Model (LBM)，通过从零售交易日志构建的长期行为画像与检索增强的产品环境，使用单一语言模型实现个性化的消费行为模拟。

**💡 创新点**

创新点在于将行为模拟拆分为 Person–Environment 两个维度，并通过持续预训练、监督微调和可验证奖励的强化学习，构建统一的、可检索驱动的语言模型框架。

**🔧 技术方法**

主要技术包括大型语言模型、检索增强生成(RAG)、轻量化 LoRA 适配器、持续预训练(CPT)、监督微调(SFT)以及基于可验证奖励的强化学习(GRPO)。

**📊 数据集**

使用的数据集为匿名化的泰国零售商 Lotus 的交易日志及产品目录，并在跨域评估中利用 Lazada 代金券数据、UCI、Tmall、Shopee 等公开数据。

**📈 对比分析**

实验中以 GPT‑5.5 为基准进行零样本和微调比较，在购买预测、硬负样本判别、篮子完成和促销响应等任务上 LBM 均比基准提升 9–15 个百分点，跨域代金券任务零样本 AUC 从 0.772 提升至 0.827。

**⚠️ 局限性**

局限性包括对手工编写的文本行为摘要的依赖，无法充分利用基于 ID 或高维工程特征的预测信号，且未能实现连续演化的用户嵌入，RL 奖励校准仍有改进空间。

---

## 265. SmartHomeSecure: Automated Detection and Repair of Smart Home Configuration Errors Using Large Language Models

**arXiv ID:** 2607.06748 | [PDF](https://arxiv.org/pdf/2607.06748v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 266. Calibration-Family Overfit: Why Trusted Sabotage Monitors Don't Transfer Across Lineages

**arXiv ID:** 2607.06596 | [PDF](https://arxiv.org/pdf/2607.06596v1)

**作者:** Lucas Pinto `[一作]` `[通讯]` (Independent Researcher), Lucas Pinto (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在 AI 控制协议中使用受信任监控器（trusted sabotage monitors）时，监控器对不同未受信任模型族（policy families）的跨族迁移性能，并提出了四步评估与补偿流程。

**💡 创新点**

创新点在于：①构建完整的跨族 AUROC 转移矩阵并分解为行（监控器能力）、列（攻击者显著性）以及剩余的交互项；②发现交互项显著为正，表明监控器对自身族的“过拟合”优势；③将交互效应转换为实际的安全差距（例如在 1% 审计预算下，跨族检测率下降约 20%）；④提出四步实务协议，包括完整矩阵报告、异族漂移检测、按族重新校准以及有限的多族聚合防御。

**🔧 技术方法**

技术方法主要包括：文本与代码轨迹嵌入（text‑embedding‑3‑small、text‑embedding‑3‑large、ada‑002 等），标准化 + PCA 降维，逻辑回归、LoRA 细调等监控器；交叉验证、任务聚类自举（cluster‑bootstrap）求交互项置信区间；以及通过审计预算下的误判率保持实现的实际安全评估。

**📊 数据集**

使用的数据集包括：APPS 代码后门基准（四个开放权重族 + 1-5 个前沿族），BigCodeBench 代码后门基准，SHADE‑Arena 非代码工具使用后门基准；每族 600 条攻击/正常轨迹，训练/测试拆分为 60/40。

**📈 对比分析**

评估方法：构造 N×N 的跨族 AUROC 矩阵，按行、列主效应归一化后计算交互项；同时在 1%、2% 与 5% 审计预算下计算捕获率（TPR）。结果显示：交互项平均为 +0.15（严格无泄漏基准），在 1% 审计下自身族捕获率约 41% 而异族仅 19%，差距 21.7 分；对弱族校准、强族部署会使捕获率进一步下降。相较于单配对评估，完整矩阵更能体现实际安全水平。

**⚠️ 局限性**

局限性：仅针对训练好的、族条件化的监控器（固定嵌入+逻辑回归、LoRA），提示式监控器未能通过提示学习族特征；样本量有限（每族 600 条），导致对极端后门（both‑success）或少数族的统计功效不足；交互项主要由可分辨的族风格和行为特征驱动，可能在更大族集或不同攻击模式下变化；多族聚合的“部分防御”仅能恢复少量性能，未能完全消除交互差距。

---

## 267. AT-Attn: Temporal-Aware Cross-Attention for Longitudinal Multimodal Alzheimer's Disease Diagnosis

**arXiv ID:** 2607.07091 | [PDF](https://arxiv.org/pdf/2607.07091v1)

**作者:** Xinyue Du `[一作]` (East China University of Science and Technology), Qiuhui Chen `[通讯]` (East China University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种 AT-Attn 框架，将结构性 MRI 与纵向认知量表及静态临床特征融合，以支持阿尔茨海默病患者级别诊断。

**💡 创新点**

创新点在于引入时序感知的 Change-and-Time 编码、时间偏置的非对称跨注意力以及门控融合，能够在 MRI 不完整时稳定地注入影像信息而不压倒更强的临床信号。

**🔧 技术方法**

采用 3D ResNet‑18 预训练骨干、可学习的跨注意力机制、门控融合、注意力池化和多头注意力等深度学习技术。

**📊 数据集**

在 ADNI 数据集构建 1,520 名患者的 MRI 保留队列，使用 T1 加权 MRI、六项认知量表以及七个静态临床变量。

**📈 对比分析**

通过患者级别的五折交叉验证与单模态、早期融合、树模型等基线比较，AT‑Attn 在准确率、宏 F1、ROC‑AUC、PR‑AUC 等指标均优于基线，且与强大的表格模型竞争。

**⚠️ 局限性**

局限包括诊断标签与认知量表的重叠、仅在 ADNI 内验证、以及假设共享时间轴导致无法处理完全异步采样。

---

## 268. Decentralization and Governance in IoT: Bitcoin and Wikipedia Case

**arXiv ID:** 2607.06784 | [PDF](https://arxiv.org/pdf/2607.06784v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 269. ASFR-Net: Adversarial Alignment and Spatio-Frequency Refinement Network for Heterogeneous Remote Sensing Image Change Detection

**arXiv ID:** 2607.07161 | [PDF](https://arxiv.org/pdf/2607.07161v1)

**作者:** Xin-Jie Wu `[一作]` (Anhui University), Bin Luo `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种端到端的异构遥感影像变化检测框架ASFR-Net，结合对抗式特征对齐与时频细化，显著提升了跨模态变化识别精度。

**💡 创新点**

创新点包括：①模态不变表示学习器MIR‑Learner（含门控对抗域统一与极性感知特征正则化）实现全局对齐与判别性平衡；②时频协同增强模块SFEM（自适应频谱注意力与整体金字塔聚合）利用频域先验消除模态噪声；③层级引导融合模块HGFM通过深层语义引导浅层细节，解决语义-空间缺口。

**🔧 技术方法**

技术手段涵盖：对抗学习、频域FFT/IFFT与频谱注意力、金字塔特征聚合、深度监督、残差动态门控、坐标注意力、语义空间分离等，整体实现多尺度端到端优化。

**📊 数据集**

使用了新构建的高分辨率可见光‑近红外变化检测基准VisNIR‑HCD，并在公开的MT‑Wuhan、XiongAn以及传统同源数据集LEVIR‑CD、WHU‑CD上进行评测。

**📈 对比分析**

与多种先进方法（A2Net、RFANet、AFENet、HeteCD等）对比，ASFR‑Net在VisNIR‑HCD、MT‑Wuhan和XiongAn上均取得最高或相近最高的F1、IoU和OA，性能提升约3–5%，同时保持轻量级（6.35 M参数，15.13 G FLOPs）。

**⚠️ 局限性**

局限性：对图像配准要求严格，无法处理跨模态对齐误差；对某些在单一模态中不可见的变化缺乏检测能力，且在极端模态差异或低对比度区域易出现漏检或误检。

---

## 270. MILES: Modular Instruction Memory with Learnable Selection for Self-Improving LLM Reasoning

**arXiv ID:** 2607.06974 | [PDF](https://arxiv.org/pdf/2607.06974v1)

**作者:** Ruilin Tong `[一作]` (University of New South Wales), Dong Gong `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在不更新模型参数的前提下，提出MILES框架，通过在测试时收集自信推理轨迹，增量构建并学习可复用的子步骤指令记忆，进而在不确定样本上自我改进推理效果。

**💡 创新点**

创新点包括：① 异步（子目标嵌入–子指令）模块化记忆结构，② 双层粗细检索加可学习选择头的策略，③ 仅利用自信样本进行无监督增量记忆扩展与选择学习，实现纯测试时自我提升。

**🔧 技术方法**

使用技术：大语言模型（LLM）Chain-of-Thought、Self‑Consistency、自回归推理；MCTS搜索；k‑means聚类生成子目标；相似度检索；轻量化二分类选择头；二进制交叉熵训练；跨模型记忆迁移。

**📊 数据集**

实验数据集：MATH‑500、AIME 2024、AIME 2025、GPQA‑Diamond、MMLU‑Pro（Physics、Engineering）。

**📈 对比分析**

与ZS‑CoT、Self‑Consistency、Buffer‑of‑Thoughts、Dynamic CheatSheet、Tree‑of‑Thoughts、rStar、DORA等方法在四个LLM后端（GPT‑4.1、GPT‑4.1‑mini、Qwen3‑30B‑Instruct、GPT‑OSS‑20B）上比较，MILES在所有基准上均匹配或超越，显示更优的准确率‑效率折衷，并且对计算预算和跨模型迁移表现稳健。

**⚠️ 局限性**

局限性：依赖LLM的指令遵循能力；MCTS轨迹收集增加额外计算开销；记忆检索与选择头的效果受相似度匹配精度和数据稀疏性的影响；仍受限于冻结模型的表达能力。

---

## 271. MMGenre: Benchmarking Singing Voice Synthesis across Multiple Musical Genres

**arXiv ID:** 2607.06986 | [PDF](https://arxiv.org/pdf/2607.06986v1)

**作者:** Wenhao Feng `[一作]` (Renmin University of China), Qin Jin `[通讯]` (Renmin University of China)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MMGenre多音乐流派评测基准，自动生成多流派声乐与符号谱对；

**💡 创新点**

创新点在于利用文本到音乐模型自动生成多流派数据、构建可扩展评测框架，并揭示当前SVS系统存在的流派衰退现象；

**🔧 技术方法**

采用Suno V4.5文本到音乐生成、Mel‑RoFormer分离声道、STARS音高/时值标注、MuQ‑MuLan评估流派一致性，并使用Gemini 2.5 Pro自动评分；

**📊 数据集**

构建的数据集包含3,152段（约4.36小时）中式声乐与谱，覆盖10大流派和26子流派；

**📈 对比分析**

通过对8种代表性SVS模型的GCS‑5评分、伪MOS指标和CER进行比较，发现所有模型在非流行流派上的流派对齐分数低，零射控制提升有限，但有限的流派专属微调可显著提升；

**⚠️ 局限性**

局限在于评测主要基于AI合成声乐，尚未完全验证在真实录音中的泛化；并且对不同语言、声学环境的适用性未深入探究。

---

## 272. Best-Arm Identification with Generative Proxy

**arXiv ID:** 2607.06879 | [PDF](https://arxiv.org/pdf/2607.06879v1)

**作者:** Tianyi Ma `[一作]`, Jierui Zuo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究在每次昂贵奖励抽样同时配有廉价代理评分的固定置信度最佳臂识别问题，提出一种基于控制变量和OLS残差方差上界的相位消除算法；

**💡 创新点**

创新点在于使用OLS残差方差的χ²上界来构造一个对未知奖励-代理相关性的保守上证，从而在不知相关系数的情况下实现接近oracle的样本复杂度；

**🔧 技术方法**

主要技术包括控制变量方法、普通最小二乘回归、相位消除（phase elimination）、χ²分布的置信上界、一次回合延迟策略与自适应采样；

**📊 数据集**

实验数据包括：合成高斯与非线性环境；以及来自哥伦比亚大学的真实汽车贷款回放数据，代理来自多种LLM（Qwen2.5‑7B、GPT‑5.5、Fine‑tuned Qwen）和TabPFN；

**📈 对比分析**

与仅使用奖励的基线、已知相关系数的oracle以及不同代理版本进行对比；在模拟与真实数据上都实现了相对奖励基线的样本量减少，最高可达约61%，且保持正确率接近理论置信水平；

**⚠️ 局限性**

局限性包括：代理保持固定，无法在线细化；假设奖励噪声为高斯且方差为1；校准阶段需额外样本；仅在二臂定价实验中验证，未在更广泛的纯探索任务上测试。

---

## 273. Gen4U: Unifying Video Generation and Understanding via Diffusion

**arXiv ID:** 2607.06856 | [PDF](https://arxiv.org/pdf/2607.06856v1)

**作者:** Michael King `[一作]` (Google DeepMind), Viorica Pătrăucean `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型视频扩散模型在视频编码中的潜力，提出Gen4U框架，利用单次前向传播提取特征用于多任务学习。

**💡 创新点**

发现生成式扩散模型在中等噪声层面蕴含强语义与几何信息，利用零拷贝相互kNN对齐和线性/注意力探针定位最佳特征提取点，实现生成与理解的统一。

**🔧 技术方法**

使用低维扩散模型、零样本相互kNN对齐、线性与注意力探针、轻量化解码器、噪声级别多样化的数据增强等技术。

**📊 数据集**

在SSv2视频分类、ScanNet深度与姿态估计、COCO/SSv2/Vatex图像与视频字幕、VATEX文本对齐等多种数据集上进行实验。

**📈 对比分析**

与MAE、VideoMAEv2、V-JEPA、V-WALT等基线比较，SSv2分类71.3% Top-1、ScanNet深度0.075 AbsRel、姿态1.10EPE，字幕在SSv2取得289.5 CIDEr，整体性能与现有最佳方法持平或更优。

**⚠️ 局限性**

主要实验基于专有Veo3模型，开源模型复现有限，复现性受限；在字幕任务上对大规模视觉语言数据的利用不足，影响最终表现。

---

## 274. Multimodal Voice Activity Projection for Turn-Taking in Social Robots with Voice-Activity-Related Pretrained Encoders

**arXiv ID:** 2607.07294 | [PDF](https://arxiv.org/pdf/2607.07294v1)

**作者:** Antonio Cano `[一作]` (4i Intelligent Insights), Randy Gomez `[通讯]` (Honda Research Institute Japan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种多模态声音活动投影（MM‑VAP）框架，用于社交机器人在调解情境下预测人机双人对话的轮转事件；

**💡 创新点**

创新点在于：①将与语音活动相关的预训练音视频编码器（TalkNet、WhisperFlamingo）通过低秩适配（LoRA）高效迁移到多模态VAP；②在投影头后加入语义一致性损失，使模型更关注与交互事件意义相匹配的微状态；③保持自监督未来投影目标，实现零样本事件推断；

**🔧 技术方法**

使用技术包括：Transformer架构、跨模态交叉注意机制、低秩适配LoRA、语义一致性损失、音频-视觉同步预处理、语义级别事件聚合；

**📊 数据集**

使用数据集：多语种多模态对话语料NoXi（英语、法语、德语）、扩展版NoXi+J（含日语、中文）、对话机器人调解语料EDR（Haru），以及基线CPC+3DResNet模型；

**📈 对比分析**

方法与现有VAP、CPC+3DResNet、前期多模态VAP等进行比较，评估指标为准确率和F1分数。实验显示，MM‑VAP在NoXi+J上所有语言的S‑pred、S‑L、S‑H事件均超过SOTA，EDR调解数据上也获得更高的Hold/Shift/预测精度；

**⚠️ 局限性**

局限性包括：①仅在离线数据上验证，缺乏实时推理与部署评估；②计算开销较大，尤其是大型预训练编码器；③仅针对双人交互，未扩展至多方或完整的三模态场景；④需要进一步调优以适应不同硬件与实时约束。

---

## 275. Benchmarking and Engineering Data Structures for Spherical Range Queries

**arXiv ID:** 2607.07367 | [PDF](https://arxiv.org/pdf/2607.07367v1)

**作者:** Thomas Bläsius `[一作]` (Karlsruhe Institute of Technology), Nikolai Maas `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种新的空间索引结构SPRK-tree，并对其在多种数据集上的性能进行了系统评测；

**💡 创新点**

创新点在于将KD树的半径减小裁剪、SNN式叶节点排序投影以及基于SVD的高维旋转相结合，并辅以SIMD与内存布局优化，显著提升查询效率；

**🔧 技术方法**

采用了KD树变体、半径减少裁剪、SNN叶节点、SVD预旋转、AVX‑512 SIMD向量化、两阶段遍历、缓存友好布局等技术；

**📊 数据集**

使用了图嵌入数据（GIRG生成的240个实例，点数至1M，维度至32）、均匀随机点（2–32维，10k–10M点）、高维真实数据（SIFT、GIST、GloVe、Deep1B、F‑MNIST等）、聚类数据集及开放街图POI数据；

**📈 对比分析**

在单核Intel Xeon上与10+主流索引（KD树、Balltree、VP-tree、R-tree、Orthtree、Grid、SNN、Brute‑Force等）进行基准对比，SPRK-tree在大多数配置下实现最快查询时间，往往比竞争者快2–10倍；

**⚠️ 局限性**

在极高维（≥960维）时仍略逊于SNN，SVD旋转带来额外开销，格子索引需预先知晓半径，且对极小数据集时开销可能占比过大；

---

## 276. Prior-matched evaluation of operational Earth-observation classifiers: a three-number reporting method demonstrated on Sentinel-1 internal-wave detection

**arXiv ID:** 2607.07146 | [PDF](https://arxiv.org/pdf/2607.07146v1)

**作者:** Joao Pinelo `[一作]`, Adriana Santos-Ferreira `[通讯]` (Atlantic International Research Centre)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文在 Sentinel‑1 波模式档案中检测内部孤立波，并提出先前匹配的三数字报告方法，解决模型训练时平衡与运营时稀疏的不匹配问题。

**💡 创新点**

创新点包括：①三数字轴（平衡测试、运营前景、真实运营）来显式展示先前效应；②在冻结的、空间泄漏受控的测试集上进行先前匹配评估；③精度优先的开发循环，逐步提升模型精度。

**🔧 技术方法**

采用的技术包括：SAR_CNN v2（34k 参数的卷积网络）与 Generalised‑Mean（GeM）池化；数据增强、负样本多样化、阈值优化、先前匹配校准以及预注册的锁盒评估。

**📊 数据集**

使用的数据集为 Sentinel‑1 波模式短图像约 17 M 张，其中 1.7 M 为已验证集，冻结的训练/开发/锁盒分割分别包含 30 k、1.5 k 以及 1.6 k（正样本）样本。

**📈 对比分析**

在平衡测试上模型精度为 0.996，运营前景评估（先前匹配 5%）为 0.927，阈值提升后精度从 0.192 提升到 0.464，后续开发循环进一步提升到 0.927；真实运营精度待后续验证。

**⚠️ 局限性**

局限性包括：评估基于验证队列的样本偏差，先前匹配阈值会随时间漂移，真正运营精度尚未公布，数据集规模受限，模型对未见地点的泛化仍有限。

---

## 277. Entropy-Guided Tensor Compression for Multimodal Federated Learning on Edge Devices

**arXiv ID:** 2607.06651 | [PDF](https://arxiv.org/pdf/2607.06651v1)

**作者:** Quoc Bao Phan `[一作]` (Florida State University), Tuy Tan Nguyen `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种面向多模态联邦学习的熵导向矩阵乘积状态（MPS）压缩框架MESH‑FL，用于在资源受限的边缘设备上自适应压缩模型更新

**💡 创新点**

通过利用层级更新的谱熵对压缩秩进行自适应分配，解决了传统统一压缩策略在多模态、异构设备环境下效率低下的问题

**🔧 技术方法**

谱熵估计（截断SVD）、MPS张量分解、层级压缩秩分配、FedAvg聚合等技术

**📊 数据集**

AV‑MNIST（图像+语音），在15台Raspberry Pi 4/5集群上进行实验

**📈 对比分析**

与FedAvg、TopK、QSGD、PowerSGD等基线比较，MESH‑FL在保持或提升准确率（最高+2.01%）的同时，实现了最高56.8×的压缩率、66×的传输量下降；轻量级压缩在9.36×时已超越未压缩基线

**⚠️ 局限性**

对谱尾指数衰减的假设未在所有层和训练阶段验证，熵导向压缩带来的准确率提升缺乏理论解释，且目前只使用三阶MPS，未探讨更高阶张量分解

---

## 278. Small Language Model-based Control for BBR over Low Earth Orbit Satellite Internet

**arXiv ID:** 2607.07142 | [PDF](https://arxiv.org/pdf/2607.07142v1)

**作者:** Rakshitha De Silva `[一作]` (Deakin University), Jonathan Kua `[通讯]` (Deakin University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建全球 Starlink 测试平台，系统评估 BBR 在 LEO 卫星网络中的表现，并设计基于小型语言模型（SLM）的自适应节拍增益框架，以降低重传率。

**💡 创新点**

提出离线强化学习+LoRA 微调的 SLM 方案，能够在保持 BBR 高吞吐的前提下，显著降低重传率，并实现低推理延迟（<5 ms）与轻量级部署。

**🔧 技术方法**

使用 BBR 拥塞控制、LoRA 参数高效微调、离线经验池、GPT‑2/T5/GPT‑Neo/SmolLM2 等小型语言模型，以及 surrogate 模型估计吞吐和重传。

**📊 数据集**

采集自 6 个 AWS 端点（Ohio、São Paulo、London、Mumbai、Sydney、Tokyo）的真实 Starlink 流量日志，并构成训练/测试数据集。

**📈 对比分析**

通过与 Cubic、Vegas、Hybla 的吞吐、RTT、重传等指标对比，以及利用 SLM 预测节拍增益进行仿真评估，结果显示 SLM 在保持 BBR 吞吐优势的同时，重传率降低 30%–50%，推理延迟低于 5 ms。

**⚠️ 局限性**

局限性：在极端高 RTT/抖动环境、不同卫星链路配置以及多路径、多接入场景下的泛化性能仍需进一步验证；模型对极端网络波动的适应性尚不确定。

---

## 279. Flow-ERD: Agent-type Aware Flow Matching with Entropy-Regularized Distillation for Diverse Traffic Simulation

**arXiv ID:** 2607.06957 | [PDF](https://arxiv.org/pdf/2607.06957v1)

**作者:** Seulbin Hwang `[一作]` (NAVER LABS Corp.), Jinhan Lee `[通讯]` (NAVER LABS Corp.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了多智能体交通仿真器 Flow-ERD，能够在闭环条件下同时提升逼真度与多样性；

**💡 创新点**

创新点在于①引入 Agent‑Type Aware Flow Matching (AFM)，将连续流匹配与类型感知运动执行相结合，保留多模态表达同时保持运动可行性；②提出 Entropy‑Regularized Distillation (ERD)，用熵正则化的反 KL 目标细调闭环分布，显著减少模式坍塌；

**🔧 技术方法**

使用流匹配 (Flow‑Matching)、连续动作空间、Transformer 语义编码器、类型特定的非齐性/齐性运动模型、熵正则化的逆 KL 损失以及分布匹配蒸馏 (DMD) 等技术；

**📊 数据集**

基于 Waymo Open Sim Agents Challenge（WOSAC）2025 版的训练/验证/测试数据集；

**📈 对比分析**

与多种基线（Token‑Based、Diffusion、UniMM 等）在 RMM、kinematic、interaction、map 成分及 log‑free 多样性指标 CPD 上进行对比；Flow‑ERD 在测试集上获得最高 RMM，验证集上占据真实性‑多样性 Pareto 前沿；β=1.0 取得最佳 RMM，β=0.99 在保持高 RMM 的同时恢复大部分 CPD；

**⚠️ 局限性**

受限于单日志未来的 RMM 损失对多模态的偏好、熵温度需手动调节、训练成本高、对动作到位置信息对齐的依赖；在极端多模态或复杂场景下仍可能出现模式坍塌。

---

## 280. Intrinsic Green's Learning: Supervised Learning on Manifolds via Inverse PDE

**arXiv ID:** 2607.07034 | [PDF](https://arxiv.org/pdf/2607.07034v1)

**作者:** Alexandre Quemy `[一作]` `[通讯]` (Hother Labs), Alexandre Quemy (Hother Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种通过学习源项并对Green函数积分来建模目标函数的“Intrinsic Green's Learning (IGL)”框架，并通过两阶段训练实现坐标发现与源项拟合分离。

**💡 创新点**

创新点在于将高维积分通过张量分解降为一维积分，实现与低秩Green函数、源项分解的线性可解；以及通过可学习门控自动发现隐含维度，并利用两阶段训练避免维度坍塌。

**🔧 技术方法**

使用了张量CP分解、Green函数近似、Fubini定理实现积分分解、可学习编码器、可学习门控以及线性最小二乘或梯度下降的内部求解，构成两阶段Variable Projection算法。

**📊 数据集**

在合成低维流形（如瑞士卷、旋转平面）、MNIST手写数字数据集以及若干公开图像数据集上进行实验。

**📈 对比分析**

与传统MLP、PINN、Kernels、Neural Operators等方法比较，IGL在维度恢复、样本效率、可扩展性以及对复杂决策边界的表达能力上均优于或相当于最优基线；在MNIST上实现了接近最佳线性探针准确率、自动维度压缩和较好的标签平滑度。

**⚠️ 局限性**

局限性包括对可分解Green函数和低秩源项的依赖、对高内在维度或复杂拓扑的适应性待验证、以及需要两阶段训练导致的额外调参与训练开销。

---

## 281. Near-Optimal Lower Bounds on One-Bit Compressed Sensing of Approximately Sparse Signals

**arXiv ID:** 2607.06750 | [PDF](https://arxiv.org/pdf/2607.06750v1)

**作者:** Junren Chen `[一作]` (Columbia University), Ming Yuan `[通讯]` (Columbia University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文通过构造两点在信号空间内不可区分的方式，证明了在1bit压缩感知（含有或不含有均匀抖动）的近似稀疏信号（尤其是ℓ1稀疏、ℓ_q稀疏、低秩矩阵）中，均匀恢复误差下界为Θ((k/m)^{1/3})（或其一般化），与已知的上界几乎匹配。

**💡 创新点**

创新点包括：① 用随机矩阵下的“嵌入+构造”框架完成下界证明，填补了ℓ1稀疏情况下上界是否最优的空白；② 将该框架推广到ℓ_q稀疏、子-Weibull设计、对抗位翻转、低秩矩阵以及稀疏到非稀疏的过渡；③ 通过“lifting”技巧将非抖动模型与抖动模型连接，实现统一的下界分析；④ 明确指出误差率随k、m、λ、β等参数的精确阶数。

**🔧 技术方法**

主要技术手段有：随机子高斯（或子-Weibull）矩阵的稀疏嵌入与小球概率（small‑ball）估计；构造引理（Construction Lemma）用以在小欧氏球内生成与0不可区分的信号；lifting映射将球嵌入单位球；几何与概率工具（Chernoff、Dvoretsky–Milman、矩阵核范数的几何约束）配合使用；对比分析与已知上界（如NBIHT、hamming距离最小化、投影梯度下降）进行阶数匹配。

**📊 数据集**

该研究为纯理论分析，主要使用的“数据集”是随机生成的测量矩阵（高斯、子高斯或子-Weibull）以及均匀分布抖动τ；不依赖真实数据。

**📈 对比分析**

与现有上界（O((k/m)^{1/3})、O((λk/m)^{1/3})等）进行阶数对比，证明了下界与上界在无对数因子差异下几乎一致；并给出了对抗噪声、低秩矩阵以及稀疏度过渡的精细阶数，展示了理论上最优恢复误差的完整描述。

**⚠️ 局限性**

局限性：① 下界仅对随机（子高斯/子-Weibull）设计成立，无法推广到任意设计；② 证明中出现了√(log m)的对数因子，作者认为可能为证明技巧导致，可进一步精细化；③ 对多比特量化或相位检索等更复杂的量化模型的下界尚未给出；④ 对抗噪声的下界给出了Ω(β)的线性项，但仍未给出更细的取向或多尺度分析。

---

## 282. Extending Xenakis: From Architectural Geometry to Sonification of the Philips Pavilion

**arXiv ID:** 2607.06589 | [PDF](https://arxiv.org/pdf/2607.06589v1)

**作者:** Changda Ma `[一作]` (Georgia Institute of Technology), Alexandria Smith `[通讯]` (Georgia Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将菲利普斯馆的规则面几何逆向成音乐，生成弦乐连续滑音、能量块与金属木管子序列，并实现同步的实时可视化；

**💡 创新点**

将建筑几何反向为音乐生成，并恢复Xenakis的建筑↔音乐互转过程，实现多层音频与可视化的交互式系统；

**🔧 技术方法**

使用Rhino/Grasshopper进行规则面重建，Python实现MIDI生成与规则面、采样点映射，Ableton Live进行音频合成，matplotlib 3D进行实时可视化；

**📊 数据集**

菲利普斯馆九个规则面的几何模型及其3357个采样点数据（从历史绘图与文献中重建）；

**📈 对比分析**

论文未与其他方法做实验对比，仅以音频与可视化示例展示，系统完全确定性，MIDI与可视化同步无明显延迟；

**⚠️ 局限性**

规则面采样均匀缺失局部曲率细节；仅使用Metastaseis乐器限制适用性；缺乏交互调控功能；数据集局限导致无法展现更复杂建筑特征。

---

## 283. GeoProp: Grounding Robot State in Vision for Generalist Manipulation

**arXiv ID:** 2607.07101 | [PDF](https://arxiv.org/pdf/2607.07101v1)

**作者:** Guoyang Zhao `[一作]` (Tongji University), Ran Xu `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种轻量级适配器 GeoProp，利用几何投影将机器人本体感知与视觉特征对齐，形成空间锚定的状态令牌；

**💡 创新点**

核心创新在于将 3D 末端执行器位姿显式投影到图像平面，采样对应的视觉特征并通过 FiLM 局部调制，实现几何驱动的感知-视觉融合；

**🔧 技术方法**

主要技术包括相机内外参投影、特征网格上双线性采样、空间对齐的 FiLM 调制以及基于短期轨迹的预测采样；

**📊 数据集**

在三大仿真基准（MetaWorld、RLBench、RoboTwin）和 Mobile ALOHA 真实世界四项任务上进行评估；

**📈 对比分析**

与 Diffusion Policy 与 π₀ 这两大策略族对照，GeoProp 在 67 个任务上平均提升 8.7%（Diffusion Policy）和 4.0%（π₀），真实环境平均提升 10.6%，参数增量仅 2–3%；

**⚠️ 局限性**

局限性包括对相机标定的依赖、仅对末端位姿建模、对快速运动或遮挡敏感，以及未覆盖多视角或移动摄像头场景。

---

## 284. QANTIS: Hardware-Calibrated Sequential POMDP Belief Updates on IBM Heron

**arXiv ID:** 2607.06760 | [PDF](https://arxiv.org/pdf/2607.06760v1)

**作者:** Bayram Yuksel Eker `[一作]` (Neura Parse Ltd), Furkan Deligoz `[通讯]` (Istanbul Technical University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一种可重用的量子置信更新服务，该服务在每一步决策循环中使用幅度放大与校准的幅度估计来计算稀有事件的证据，从而生成给定的后验概率。

**💡 创新点**

创新点包括：① 将幅度放大（AA）与固定点幅度放大（FPA）结合，构成稳定的后验更新；② 设计了两阶段边界感知的幅度估计（BAE），在接近 0 或 1 的边界时自动调整先验，避免估计漂移；③ 在 IBM Heron R2/R3 硬件上实现了可重复、可扩展的量子后验服务，并通过 Hellinger 距离和决策一致性指标验证其在稀有观测下的可靠性。

**🔧 技术方法**

主要技术包括：量子幅度放大（Grover、Brassard‑Hoyer‑Mosca 反射与 Yoder‑Low‑Chuang 固定点反射）、量子幅度估计（BAE）、边界校准（粗扫描 + 贝叶斯先验更新）、IBM Heron 量子处理器的门级编译、量子噪声抑制（门 twirling、零噪声外推）以及对后验分布的 Hellinger 距离、价值损失等指标的统计分析。

**📊 数据集**

实验使用的是经典的 Tiger MDP（两状态，三动作）作为验证基准，同时对 4‑state corridor、UCGate、Fez 等更大但仍可控的模拟情景进行扩展测试，所有实验均在 IBM Heron R2/R3、Pittsburgh、Boston、Marrakesh 等后端上执行。

**📈 对比分析**

通过将量子后验与无放大基线（No‑AA）以及传统 Grover‑AA 控制进行同轨道对比，评估了 Hellinger 距离、平均值损失和动作一致性。结果显示：在 8‑step Tiger 循环中，使用 FPA 后 Hellinger 最小为 0.009，平均 0.004；在 12‑step、20‑step、32‑step 扩展中仍保持 Hellinger ≤0.021，且在所有测试步骤中，量子后验与精确贝叶斯后验在 Tiger 的即时奖励规则下产生完全相同的动作，累计价值损失为 0。边界校准后，极端概率 0.01 与 0.95 的估计误差从 0.6317 / 0.4890 降至 0.00224 / 0.00773。

**⚠️ 局限性**

局限性包括：① 未给出完整的端到端自主系统演示或墙钟时间加速证明；② 仅验证了小规模状态空间（两状态 Tiger、四状态 corridor）和浅层电路，深层电路受相干时间限制；③ 量子后验服务在每一步的 shot 预算与硬件排队时间等实际运行时开销未在本文中完整记录，未来需进一步测量；④ 对更复杂的传感器模型、动态变化的观测噪声以及更大状态空间的扩展仍需进一步研究。

---

## 285. Thinking More, Harnessing Better: State Machine Guided Harness Automatic Generation with Project Digestion and Workflow Decomposition

**arXiv ID:** 2607.07007 | [PDF](https://arxiv.org/pdf/2607.07007v1)

**作者:** Xing Zhang `[一作]` (QI-ANXIN Technology Research Institute), Lingyun Ying `[通讯]` (QI-ANXIN Technology Research Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种自动生成高质量 fuzz harness 的方法，专为 C 语言项目设计。

**💡 创新点**

创新点在于通过数据流感知的函数聚合与分阶段回滚的生成工作流，显著降低 LLM 生成误差并提升覆盖率。

**🔧 技术方法**

技术手段包括轻量级静态分析构建结构流图、LLM 推理进行函数角色标注与代码生成，以及分四阶段的递归回滚算法。

**📊 数据集**

实验使用 25 个真实开源 C 项目（17 库、8 应用）作为数据集，涵盖常用与新颖目标。

**📈 对比分析**

与 OSS‑Fuzz‑Gen、CKGFuzzer、PromeFuzz 等基线对比，生成 harness 的分支覆盖率提升约 3.07×，缺陷触发率提升约 1.77×，并发现 7 条未公开漏洞。

**⚠️ 局限性**

局限性包括仅支持 C 语言、对 C++ 对象模型和复杂状态协议支持不足，以及对需要多阶段交互的目标效果有限。

---

## 286. Continual Learning With Participation Privacy: An Auditable Buffering-Aggregation Recipe

**arXiv ID:** 2607.07209 | [PDF](https://arxiv.org/pdf/2607.07209v1)

**作者:** T-H. Hubert Chan `[一作]` (University of Hong Kong), Mingxun Zhou `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种模块化流程，在对单个插入/删除用户事件流（单编辑邻接）进行自适应交互时，保证整个模型轨迹满足 (ε,δ)-差分隐私。

**💡 创新点**

创新点在于：①用随机缓冲包装器将单编辑邻接转换为 Hamming-邻接的稀疏批处理；②给出可检查的“适应性安全”认证定理，证明在满足新鲜随机性和稳定上下文条件下，任何非自适应的 Hamming-邻接 DP 证明都可推广到自适应场景；③通过这两者实现隐私-延迟显式链接。

**🔧 技术方法**

主要技术包括：随机缓冲（RandBin）包装、树形前缀和/下三角矩阵 DP 聚合器、量化可分解性（stable-context）认证、模组化组合、DP-SGD 与剪裁梯度、以及隐私预算分配。

**📊 数据集**

实验使用了常见的机器学习基准数据集（如 MNIST、CIFAR‑10 等）进行流式 SGD 训练。

**📈 对比分析**

与传统的非自适应 DP‑SGD 以及其他隐私聚合方案比较，本文方法在给定 (ε,δ) 下保持相近的测试准确率，同时提供了可观测的延迟分布，展示了隐私预算与系统延迟之间的可调权衡。

**⚠️ 局限性**

局限性包括：①仅针对单编辑邻接，需要通过群体隐私合成扩展到多编辑；②要求每轮独立随机性和稳定上下文，某些常用机制（如共享噪声）不满足；③缓冲导致的延迟与日志时间上界相关，可能不是最优；④缺乏针对所有 DP 原语的最优下界分析。

---

## 287. Bi-PT: Bidirectional Cross-Attention Point Transformers for Four-Chamber Heart Reconstruction from Sparse Cardiac MRI Data

**arXiv ID:** 2607.06923 | [PDF](https://arxiv.org/pdf/2607.06923v1)

**作者:** Chenchuhui Hu `[一作]` (University of Texas at Arlington), Meng Ye `[通讯]` (University of Texas at Arlington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于双向交叉注意力的点变压器 Bi-PT，用于从临床稀疏 CMR 数据中的稀疏点云重建四腔心脏网格。

**💡 创新点**

创新点包括：①双向交叉注意力模块实现体素-点云信息双向融合；②利用局部仿射微分同胚变形（LADD）通过 NODE 参数化心脏形状变形；③语义感知 Chamfer 距离和拉普拉斯正则化共同提升对应关系与拓扑保持；④将多种技术集成到一个端到端训练框架。

**🔧 技术方法**

核心技术包括点变压器（Point Transformer）、双向交叉注意力（Bidirectional Cross‑Attention）、神经常微分方程（NODE）、语义感知 Chamfer 损失、拉普拉斯平滑正则化。

**📊 数据集**

使用 1,000 条公共心脏 CT 案例（TotalSegmentator V2）模拟稀疏 CMR 点云，并在此基础上构造标注网格。

**📈 对比分析**

与 CPD、NMF、MR‑Net、NDM、LTN、LTN‑DSTN 等传统与深度学习基线比较；Bi‑PT 在 CD、EMD、P2F、NC 等指标上均实现了最佳或次优结果，并且实现零自相交、低非流形面比例，显示出显著的几何精度与拓扑可靠性。

**⚠️ 局限性**

局限性包括：①依赖于合成稀疏点云，可能与真实 CMR 采样差异导致泛化能力下降；②模型主要针对四腔心脏结构，对其他心脏病变或额外结构支持有限；③训练需要大规模 CT 数据与算力，部署时计算开销较高。

---

## 288. Entropy Pacing Policy Optimization for Multi-Task Agentic Reinforcement Learning

**arXiv ID:** 2607.07178 | [PDF](https://arxiv.org/pdf/2607.07178v1)

**作者:** Zetian Hu `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对多任务代理式强化学习的熵调节策略——Entropy Pacing Policy Optimization (EPPO)。

**💡 创新点**

通过对每个任务的熵进行动态跟踪和归一化，EPPO以任务的探索-利用进度为依据，按需自适应地调节每个任务的截断范围，从而解决共享策略训练中出现的熵节奏失配与交叉问题。

**🔧 技术方法**

核心技术包括：任务级熵跟踪（EMA）、熵崩塌比率计算、基于熵的z分数标准化、可逆剪裁阈值（α·tanh(z)）以及趋势约束（对熵上升时收缩剪裁）。

**📊 数据集**

使用AgentBench提供的五个交互式任务（Operating System、Database、Knowledge Graph、ALFWorld、WebShop），并在这些任务上进行统一的多任务训练。

**📈 对比分析**

与基准方法AgentRL、DCPO、BAPO等对比，EPPO在五任务与三任务设置下均实现了更高的平均成功率，显著降低熵交叉次数和晚期熵峰值，训练更稳定。

**⚠️ 局限性**

局限性在于仅评估了固定任务组合，对更广泛的任务组合、动态任务采样和更复杂的熵信号未做探索，且未验证在真实部署环境中的可扩展性与鲁棒性。

---

## 289. How the Fusion of Onboard Sensors and V2X Data can Improve (or not) the Cooperative Perception of Connected Automated Vehicles

**arXiv ID:** 2607.07114 | [PDF](https://arxiv.org/pdf/2607.07114v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 290. End-to-End LLM Flight Planning with RAG-based Memory and Multi-modal Coach Agent

**arXiv ID:** 2607.06964 | [PDF](https://arxiv.org/pdf/2607.06964v1)

**作者:** Amin Tabrizian `[一作]` (George Washington University), Peng Wei `[通讯]` (George Washington University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个端到端的电动垂直起降航班规划系统FRAMe，结合LLM、检索增强生成（RAG）记忆和多模态教练代理，能根据自然语言偏好生成安全可行的航线。

**💡 创新点**

创新点在于把LLM与可检索的历史规划经验和多模态教练协同工作，首次实现无需人工标注的偏好评估，并通过可视化与规则检查对航线进行可行性与偏好对齐验证。

**🔧 技术方法**

采用OpenAI o3‑mini、o4‑mini、GPT‑5.4、DeepSeek‑R1等LLM作为规划器；RAG模块基于向量数据库检索相似历史航线；多模态教练使用o4‑mini的视觉能力与几何规则进行验证；提示工程、链式思维（CoT）和Web前端与Mission Planner集成。

**📊 数据集**

使用达拉斯–沃斯堡地区的三难度（Easy、Medium、Hard）合成场景，包含2/4/7个禁止飞行多边形，场景数据以KML文件提供；热身阶段生成150条满足条件的航线作为RAG数据库。

**📈 对比分析**

在四种条件（A*、Baseline、+RAG、+RAG+Coach）和四个LLM上，对三种难度和三种偏好（最短距离、最少航路点、最大障碍距离）进行评估。结果显示+RAG+Coach在所有模型上获得最高有效率（最高93.8%），并在偏好对齐上显著优于Baseline和A*，但距离优化有限。

**⚠️ 局限性**

限制包括：偏好捕捉存在头room限制（距离指标几乎无提升）；教练代理的LLM评估可能带来偏见；实验仅基于合成多边形，未考虑动态交通、能量限制或实时天气；当前仅处理静态禁止飞行区域，未覆盖动态障碍与能量约束。

---

## 291. Grounding Spatial Relations in a Compact World Model: Instruction Leakage and a Goal-Free Dynamics Fix

**arXiv ID:** 2607.06925 | [PDF](https://arxiv.org/pdf/2607.06925v1)

**作者:** Yufeng Wang `[一作]` (Stony Brook University), Haibin Ling `[通讯]` (Westlake University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了目标条件世界模型在处理以语言目标指令指定的空间关系时是否真正感知环境，而非仅仅转录指令；通过对“桌面关系任务”与外部基准（BabyAI、Language‑Table）的实验，揭示并定量评估了“指令泄漏”现象；提出了检验泄漏的两种控制（目标缺失与伪目标）以及去除目标于动力学的简单架构修正；

**💡 创新点**

首次在目标条件世界模型中系统性描述并量化指令泄漏的条件（指令可转录被测量量）、对其产生机制的直观解释、以及通过去除目标从转移动态中获得真正的关系感知的实证证据；

**🔧 技术方法**

使用Joint‑Embedding Predictive Architecture（JEPA）与ViT‑tiny编码器的组合；设计了稀疏点锚点（anchor）与目标嵌入；采用模型预测控制（CEM）进行闭环任务执行；并对模型进行线性读出、几何读出与对比控制实验；

**📊 数据集**

自研的二维关系桌面数据集（4个彩色形状，随机复制扰乱者，可调节指令中关系名称）；外部基准包括 BabyAI 的 GoToLocal 任务和 Google 的 Language‑Table 任务；

**📈 对比分析**

在“关系读出”指标上，原始目标条件模型在训练分布上达到约0.90准确率，去除目标后仅略降至≈0.88；在控制任务中，去除目标模型与无目标模型的成功率相当，均低于 oracle，说明模型的瓶颈在于动力学预测的多步漂移；相对其它方法，去除目标的改进主要体现在消除了指令泄漏而非提升控制性能；

**⚠️ 局限性**

局限包括：实验仅在二维模拟场景进行，未验证在3D或真实图像环境中的通用性；未在公开预训练模型上直接验证泄漏，需进一步探测；指令总是包含被测关系，未尝试去除该相关性；动态模型仅训练一步预测，导致多步推理误差；只对监督锚点进行几何读出，未充分评估无监督锚点；

---

## 292. Does Demand Response Increase Vulnerability to Cyber Attacks by Adversarial Data Modifications?

**arXiv ID:** 2607.06632 | [PDF](https://arxiv.org/pdf/2607.06632v1)

**作者:** Clemens Kortmann `[一作]` (RWTH Aachen University), Eike Cramer `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了工业需求响应（DR）系统在电价预测模型受到对抗性攻击（adversarial attack）时的脆弱性，并评估了攻击对经济收益的影响。

**💡 创新点**

创新点在于将对抗性攻击应用于电价预测模型，并考虑了攻击设计时对调度优化模型灵敏度的显著性；同时通过通用过程模型（GPM）系统化地分析了过程灵活性与攻击效果的关系。

**🔧 技术方法**

主要技术包括：卷积神经网络（CNN）用于德国日前电价预测；基于基本迭代法（BIM）的白盒对抗性攻击（含镜像攻击和无目标攻击）；线性及混合整数线性规划（MILP）实现工业过程调度；以及对最优基准的敏感性分析。

**📊 数据集**

数据集来源于ENTSO‑E透明平台的德国BZN|DE‑LU日间电价、风电、光伏和负荷预测数据（2023‑2024年），以及四种电解工艺的参数化模型。

**📈 对比分析**

通过与无攻击情形的对比，评估了攻击率下的电价预测误差（MSE/MAE）、调度成本以及相对成本节约；结果表明在可被人眼忽略的攻击幅度（ε≤0.1）下，DR仍保留90%+的成本节约，攻击对经济收益的影响有限。

**⚠️ 局限性**

局限性包括：仅考虑白盒攻击，现实中黑盒攻击效果可能较弱；仅研究日间市场，未涉及实时/日内市场；对抗性攻击设计较为简单，缺乏更复杂的目标攻击策略；并未深入探讨检测与防御机制。

---

## 293. BubbleSH: A Dataset of Rising Bubbles with Deformable Interfaces

**arXiv ID:** 2607.07275 | [PDF](https://arxiv.org/pdf/2607.07275v1)

**作者:** Rachna Ramesh `[一作]` (Eindhoven University of Technology), Vlado Menkovski `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了BubbleSH数据集并基于其训练概率生成模型预测气泡簇的轨迹和形变。

**💡 创新点**

采用球谐系数压缩三维气泡形状，同时提供多尺度动力学评估指标，构建轻量级但高保真数据集。

**🔧 技术方法**

使用前沿前跟踪（Front‑Tracking）DNS、球谐系数表示、条件流匹配生成器STFlow、图神经网络消息传递、以及Wasserstein距离评估。

**📊 数据集**

使用BubbleSH数据集（32个气泡，3种直径4/5/6mm，8种气体体积分数5‑40%，共24个参数组合）。

**📈 对比分析**

通过R‑ADE、R‑FDE、IoU、R‑ACD以及1‑Wasserstein分布距离等指标评估，生成器在误差上比随机游走提升约30‑50%，速度提升4‑5个数量级。

**⚠️ 局限性**

仅周期域、无破裂/合并、球谐截断产生Gibbs振荡、对长时预测不稳定，缺乏壁面、化学效应等实际工况。

---

## 294. Exploring the Interaction of Explanation Styles, Context, and Trust of AI Privacy Redaction in AI-mediated Interactions

**arXiv ID:** 2607.06687 | [PDF](https://arxiv.org/pdf/2607.06687v1)

**作者:** Roshni Kaushik `[一作]` (Fujitsu Research of America), Koichi Onoue `[通讯]` (Fujitsu Research of America)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究设计了一种基于LLM的AI中介系统，能够在多领域的隐私敏感对话中自动删减敏感信息并提供多种解释风格，并通过在线实验评估不同解释偏好、信任度及反馈机制。

**💡 创新点**

创新之处在于首次系统性探讨解释风格在不同领域和删减程度下的用户偏好，并证明个性化、情境感知的解释对提升用户信任至关重要，同时提出了用户反馈驱动的解释交互框架。

**🔧 技术方法**

使用 GPT‑5 进行信息生成、删减与解释生成，Gemma‑3 作为评估判别模型，并利用LLM对解释进行敏感信息去除；结合在线问卷和信任量表进行实验评估。

**📊 数据集**

实验数据为在六个领域（医疗、教育、职场、社交媒体、金融、法律）中生成的 18 组信息（共 90 条原始信息），并结合公开的社交媒体样本（2100 条帖子）用于示例说明。

**📈 对比分析**

通过混合设计实验和混淆矩阵评估解释风格区分度，使用 5 级 Likert 信任量表测量用户信任；结果显示解释偏好随情境变化显著，且符合偏好的解释显著提升信任，证明个性化解释有效。

**⚠️ 局限性**

局限性包括实验环境为离散控制的域和删减级别，缺乏真实动态交互的复杂性；仅观察短期交互，未考察长期信任演变；系统自适应实现仍需进一步研究。

---

## 295. Finding and Understanding Miscompilation Bugs in the Solidity Compiler

**arXiv ID:** 2607.07217 | [PDF](https://arxiv.org/pdf/2607.07217v1)

**作者:** Bhargava Shastry `[一作]` `[通讯]` (Ethereum Foundation), Bhargava Shastry (Ethereum Foundation)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了名为SolCompilerFuzzer的差分模糊测试工具，用于生成语义合法的Solidity程序并检测Solidity编译器的误编译bug。

**💡 创新点**

创新点在于构建了语义感知的差分模糊测试框架，结合增量程序生成和精细的误编译判定，显著提高了误编译检测的覆盖率和效率。

**🔧 技术方法**

采用语义感知的程序生成、差分运行时追踪、基于EVM的执行比较以及对编译器不同路径的控制/实验分组设置。

**📊 数据集**

使用内部自动生成的Solidity测试集；未使用公开的真实合约数据集，而是通过fuzzer按需合成覆盖语言特性多样的程序。

**📈 对比分析**

通过对比控制组与实验组在执行输出、状态变更及错误码等指标的相同/差异性来判定误编译；在三年期间共发现25个误编译bug，成本约$15000，平均每个bug成本<$1000，效果显著。

**⚠️ 局限性**

局限性包括缺乏明确的正确性基准，无法检测在两组均出现相同错误的bug；对未终止程序的检测受限于EVM gas终止；仅关注误编译而非编译器崩溃；依赖于自动生成程序的多样性和正确性。

---

## 296. Latent graph encoding of multimodal neuroimaging features with generative AI architectures

**arXiv ID:** 2607.07027 | [PDF](https://arxiv.org/pdf/2607.07027v1)

**作者:** Ishaan Batta `[一作]` (Georgia State University), Vince Calhoun `[通讯]` (Georgia State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究多模态脑影像生成模型，提出利用图注意力网络编码功能连接矩阵、Mixture‑of‑Experts 融合结构与功能特征的图卷积变分自编码器（gMMVAE），并与变分、GAN、Transformer 与扩散模型进行系统对比。

**💡 创新点**

① 在功能MRI上使用图结构编码，显著保留网络拓扑；② 采用MMVAE的Mixture‑of‑Experts融合策略，将不同模态潜在分布合并；③ 双重条件化（编码器+解码器）提升潜在空间可解释性与分类性能；④ 与传统向量化与数据空间方法的对比证明图基模型在重建、生成与效率上均有显著优势。

**🔧 技术方法**

图注意力网络（GATv2）、多层感知机（MLP）、多模态变分自编码器（MMVAE）及其图变体 gMMVAE、gLDM、gDiT；条件化通过 FiLM 模块实现；与 DDPM、WGAN‑GP、DiT、LDM 等基线模型对齐。

**📊 数据集**

UK Biobank 10,000 名受试者，使用 53 个 NeuroMark 组分得到的静态功能网络连接矩阵（sFNC）和灰质体积（GMV）特征。

**📈 对比分析**

通过重建指标（MSE、Frobenius、相关、SSIM、PSNR）、生成指标（MMD、WD、KL）和潜在空间性别分类（准确率、精确率、召回率、F1）进行评估，图基模型 gMMVAE 在重建精度、生成分布对齐和潜在空间可区分性上均显著优于 MLP、GAN、Transformer 与扩散基线；同时参数量、FLOPs 与推断延迟最小。

**⚠️ 局限性**

目前仅在 ROI/特征级别进行建模，未扩展到体素级；缺乏对疾病特定数据集的验证；条件化仅以性别为例，尚未充分探索多维临床变量；扩散模型需要多步推断导致计算成本高，限制了实时应用。

---

## 297. Progressive Crystallization: Turning Agent Exploration into Deterministic, Lower-Cost Workflows in Production

**arXiv ID:** 2607.07052 | [PDF](https://arxiv.org/pdf/2607.07052v1)

**作者:** Arun Malik `[一作]` `[通讯]` (Microsoft Azure Networking), Arun Malik (Microsoft Azure Networking)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并在生产环境中实现了 Progressive Crystallization 生命周期，将 LLM 代理的探索过程转化为可重复、零代价的 deterministic 工作流程，从而显著降低成本、提升安全性。

**💡 创新点**

创新点包括：① 把代理探索视为发现机制；② 设计了从完全代理驱动到完全确定性的执行类型谱；③ 基于累计证据的自动升降级（promotion/demotion）策略；④ 结合过程挖掘提取可执行剧本并生成验收测试；⑤ 通过经济与安全模型证明升迁不会降低安全性。

**🔧 技术方法**

使用技术：LLM 代理（ReAct/Toolformer）、工具调用与日志记录、过程挖掘算法（从执行轨迹抽取 DAG）、规则引擎替代 LLM 输出、自动化验收测试、持续监控与 circuit breaker、token 成本与延迟度量。

**📊 数据集**

数据集：来自 Microsoft Azure 云网络运维平台的真实生产事故日志，涵盖数十万次每月事故，持续观察 8 个月的演进过程；未使用公开公开数据集。

**📈 对比分析**

比较方法：按执行类型（Type 1、2、3）比例、每事件代理成本、事故量、MTTR、误报率等指标对比；结果显示：Type 1 占比从 0% 增至约 45%，每事件成本下降 70%（同时事故量翻倍），MTTR 下降数小时至数分钟，误报率保持 <5%。

**⚠️ 局限性**

限制：仅在单一组织与域内验证；对完全重复模式有效，单独事故多时收益有限；自动提取与升迁依赖日志质量与阈值设定，易导致误升迁；演化周期仅为 8 个月，未验证更长时间；最终 deterministic 逻辑仍需人工复核。

---

## 298. When Certificates Fail: A Unified Safety Framework for Embedded Neural Interface Models

**arXiv ID:** 2607.06630 | [PDF](https://arxiv.org/pdf/2607.06630v1)

**作者:** Jasmeet Singh Bindra `[一作]` `[通讯]` (IIT Mandi), Jasmeet Singh Bindra (IIT Mandi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一个统一的经验审计框架，对嵌入式神经接口模型的鲁棒性认证、信号保真度和隐私泄露进行系统评估，揭示安全证书与实际任务表现之间的差距。

**💡 创新点**

创新点在于：①将鲁棒性认证、代理保真度偏离和潜在信息泄露三种对齐失败模式归纳为一个统一的审计体系；②展示即使正式的 Lipschitz 证书通过，模型在实际攻击或扰动下的分类准确率仍可显著下降；③通过多维度保真度度量揭示代理优化对时间域与频域信号的互斥影响；④证明公开任务嵌入会泄露主体身份，并探索简单的投影/噪声去噪缓解方法。

**🔧 技术方法**

采用 Lipschitz‑style 输出灵敏度检验、投影梯度（PGD）对抗攻击、EEGNet、CSP+LDA、FBCSP+LDA 解码器、时间域均方误差、相关系数、频谱对数‑MSE、线性/非线性隐私探测器（KNN、随机森林、MLP）、投影去除、Gaussian 噪声注入、配对符号翻转检验等技术。

**📊 数据集**

使用 BCI Competition IV 2a（9 受试者，运动意象四类）和 SEED‑IV（15 受试者，情感识别四类）两个公开 EEG 数据集。

**📈 对比分析**

对比实验显示：①在 ϵ=0.25 的 PGD 攻击下，EEGNet、CSP+LDA、FBCSP+LDA 的准确率均下降 24‑35%，而 Lipschitz 证书始终通过；②时间域重建损失下降 0.11、相关性提升 0.08，但频谱 log‑MSE 上升 1.75；③对频谱损失优化则相反；④在 SEED‑IV 上，线性隐私探测器识别身份准确率 48% 远高于 6.7% 的随机猜测。

**⚠️ 局限性**

局限性包括：①Lipschitz 证书使用全局乘积上界，保守度高，可能未捕捉局部鲁棒性；②实验仅覆盖非侵入式 EEG，缺乏侵入式或在线自适应场景；③隐私实验仅用 3 个随机种子，统计功效有限；④物理扰动测试在预处理后施加，未模拟真实设备噪声；⑤投影/噪声去噪方法缺乏正式差分隐私保证。

---

## 299. ORCAID: Oblique Rule-Based Continuous-Action Interpretation for Deep RL Policies

**arXiv ID:** 2607.07235 | [PDF](https://arxiv.org/pdf/2607.07235v1)

**作者:** Ignacio D. Lopez-Miguel `[一作]` (TU Wien), Martin Tappler `[通讯]` (TU Wien)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建可解释的规则式代理模型，从深度强化学习策略中提取连续动作的决策规则。

**💡 创新点**

提出高效的斜面决策树训练算法，三阶段分割搜索（随机初始化、局部细化、后向消除），并通过合并相邻叶子生成简洁规则。

**🔧 技术方法**

斜面决策树、线性回归、后向特征消除、差分进化优化、DAgger 数据聚合、QM 简化。

**📊 数据集**

在九个经典控制任务上验证，包括 Mountain Car、Pendulum、Inverted Pendulum、Inverted Double Pendulum、Reacher、Swimmer、Lunar Lander、Hopper、Half Cheetah。

**📈 对比分析**

与 CART、Cubist、RuleFit 三种基线（及其 DAgger 版本）对比，结果显示模型规模更小、拟合误差最低、奖励比率≥0.75，且在多数环境中可提升原始策略。

**⚠️ 局限性**

仅适用于确定性策略，需环境交互查询，规模受限于状态/动作维度，且在处理随机策略和更高维问题时可能表现欠佳。

---

## 300. Auditable Machine Unlearning for Privacy-Compliant Ransomware Detection Using Multi-Shard SISA and Deep Reinforcement Learning

**arXiv ID:** 2607.06860 | [PDF](https://arxiv.org/pdf/2607.06860v1)

**作者:** Jannatul Ferdous `[一作]` (Charles Sturt University), Md Zahidul Islam `[通讯]` (Charles Sturt University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个可审计的机器无学习框架，用于在满足 GDPR/CCPA 等隐私合规要求的前提下，对基于行为特征的勒索软件检测模型实现选择性样本删除。

**💡 创新点**

创新点在于：① 将值基深度强化学习（DDQN）与多分片 SISA 训练并行化，支持仅重训练受删除样本影响的分片；② 设计了双重审计机制——基于 oracle 的遗忘验证和基于 Q‑margin 的成员推断审计；③ 在保持高检测性能的同时，显著降低了删除时的计算成本（单片 5–30 s vs 全模型 80–330 s）。

**🔧 技术方法**

使用了 Double Deep Q‑Network（DDQN）进行行为特征的奖励驱动分类；多分片 SISA（Shard‑Isolated‑Slice‑Aggregated）框架实现分片级别的训练与增量重训练；Q‑margin 作为连续置信度，用于 ROC‑AUC 评估和成员推断；采用了标准化、经验回放、ε‑greedy 探索等 RL 基础技术。

**📊 数据集**

数据集为 2,000 条 Windows 11 行为日志（1,000 勒索软件 + 1,000 正常程序），共 103 维特征，来源于 ANY.RUN 沙盒和多个公开恶意样本库。

**📈 对比分析**

对比方法主要是传统 DQN 与 DDQN 的基线检测性能，并与单片与多片 SISA 的无学习效果进行对比。结果显示：基线 DDQN F1≈0.9925，AUC≈0.9983；单片无学习几乎无性能下降，M=5–10 的多片配置在 1–10% 删除率、1–10 轮删除下维持 ΔF1<0.02；在极端多片 (M=20) 与高删除压力下，性能下降显著。Oracle 冲突率低，成员推断 AUC 接近 0.5，证明删除后隐私泄露不显著。

**⚠️ 局限性**

局限性包括：① 数据规模有限（仅 2,000 条样本，缺乏多样性）；② 仅采用 Q‑margin 代理进行成员推断，未对抗更强的攻击；③ 未在持续流式或实时部署环境下验证多轮删除的长期稳定性；④ 分片划分的经验式选择（M=5–10）缺乏理论最优性分析；⑤ 只研究了值基 RL，未探讨策略梯度或 Actor‑Critic 方案的可行性。

---

## 301. Ranking and Rank Aggregation with Matroid Prefix Constraints

**arXiv ID:** 2607.07153 | [PDF](https://arxiv.org/pdf/2607.07153v1)

**作者:** Seiei Ando `[一作]` (Institute of Science Tokyo), Yu Yokoi `[通讯]` (Institute of Science Tokyo)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种在基于 matroid 或 flag matroid 前缀约束下的最接近排名与投票聚合问题，并给出了多项算法与复杂度结果。

**💡 创新点**

创新点在于：①将公平排名的 k‑fairness 与 block‑fairness 等限制形式统一为 matroid 与 flag matroid 结构；②证明 Flag‑Matroid‑CFR 可通过贪心 + Bruhat 序关系在多层约束下求解最优；③在此基础上得到 (2+ε)‑近似和 2.881‑近似的聚合算法；④揭示对任意固定投票数 m≥2 的 Matroid‑FRA 为 NP‑hard，显示约束下聚合难度提升。

**🔧 技术方法**

核心技术包括 matroid 理论（基、独立集、基底交换与 quotient 关系）、Bruhat 排序论证、组合优化中的贪心算法、以及对已有公平聚合框架的拓展（如色彩化子问题的 matroid 版本）。

**📊 数据集**

论文未使用公开实验数据集，而是通过理论证明和构造实例来验证算法和复杂度。

**📈 对比分析**

与传统 Kemeny 聚合相比，提出的 (2+ε) 与 2.881 近似算法在满足 matroid/flag matroid 约束时保持与无约束下的相同近似比；最接近排名问题在 Flag‑Matroid‑CFR 中实现多项式时间最优解，显示理论上可行性。

**⚠️ 局限性**

局限性包括：①聚合问题对任意固定 m≥2 时 NP‑hard，无法得到多投票数的多项式求解；②贪心最优性依赖 flag matroid 的 quotient 关系，若约束不满足此结构则算法失效；③论文未提供实验评估，仅靠理论证明。

---

## 302. NoDrift3R: Raymap-Guided Coupling for Drift-Robust Unposed Feed-Forward 3D Reconstruction

**arXiv ID:** 2607.07168 | [PDF](https://arxiv.org/pdf/2607.07168v1)

**作者:** Xiangyu Sun `[一作]`, Eunbyung Park `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

无法获取论文具体内容，无法描述具体研究目标。

**💡 创新点**

缺乏足够信息，无法指出创新点。

**🔧 技术方法**

未知。

**📊 数据集**

未知。

**📈 对比分析**

无法比较方法与性能。

**⚠️ 局限性**

由于信息不足，无法说明研究的局限性。

---

## 303. Gauge-Invariant Learnable Spectral Positional Encodings for Directed Graphs via Hermitian Block Krylov Subspaces

**arXiv ID:** 2607.07032 | [PDF](https://arxiv.org/pdf/2607.07032v1)

**作者:** Jiaqing Xie `[一作]` (Fudan University), Yuxin Wang `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种无特征向量、对方向不变的可学习谱位置编码（LLPE），通过磁性拉普拉斯算子和块 Krylov 子空间实现；

**💡 创新点**

创新点在于将谱 PE 定义为磁性算子函数的矩阵形式，天然保持规范不变且不需要昂贵的特征向量；

**🔧 技术方法**

使用 Hermitian block Krylov 子空间、稀疏矩阵向量乘、可学习的谱响应族（Chebyshev、热-解析、MLP、自由权重）以及随机探测向量；

**📊 数据集**

在人工设计的循环有向 SBM、真实有向 WebKB/Wikipedia 图和无向异质图上进行实验；

**📈 对比分析**

与对称谱 PE、随机探测、传统特征向量方法以及精确特征向量基准进行对比；在有向 SBM 上磁性 Krylov PE 达到 97–99% 准确率，远优于对称方法；在真实图上与随机探测相当；在无向异质图上热-解析 Krylov PE 超越无 PE 与多项式基线；

**⚠️ 局限性**

局限性包括：对潜在 q 的固定网格，无法学习；对大图时仍需大量探测向量以降低 Monte‑Carlo 误差；对小图精确谱仍优于 Krylov 近似；

---

## 304. NativeMEM: Native Memory Compression for Long-Horizon Robotic Manipulation

**arXiv ID:** 2607.06678 | [PDF](https://arxiv.org/pdf/2607.06678v1)

**作者:** Ziye Wang `[一作]` (University of Hong Kong), Hongyang Li `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fede83ac-7505-405f-ab37-e7284695c47f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 NativeMEM，将预训练的单帧 Vision‑Language‑Action 模型通过原始视觉编码器压缩历史帧为单个 token，实现长时序视觉记忆。

**💡 创新点**

创新点在于利用原始视觉编码器实现单 token 压缩的原生记忆，采用两阶段对齐训练，使记忆与预训练策略兼容且高效。

**🔧 技术方法**

技术包括原生记忆压缩（Native Memory Compression）、两阶段训练（冻结与解冻）、基于 VLA 的 tokenizer、实时记忆推理。

**📊 数据集**

使用模拟与真实机器人数据，涵盖 RMBench、Click Buttons、Put Back Block、Grocery Checkout Scanning 等任务。

**📈 对比分析**

与 MemER、Mem-0、HAMLET、MEM-short 等基线相比，NativeMEM 在模拟任务上平均成功率提升至 84.0%，在真实机器人上达到 98.7%，仅用 20% 训练数据且保持低延迟与显存。

**⚠️ 局限性**

局限性是难以维持小时或日级的长期记忆，且仅通过动作监督学习记忆，可能不足以满足更复杂任务的语义需求。

---

## 305. Enhancing deep learning models for time series classification via knowledge distillation

**arXiv ID:** 2607.06796 | [PDF](https://arxiv.org/pdf/2607.06796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 306. AgentLens: Production-Assessed Trajectory Reviews for Coding Agent Evaluation

**arXiv ID:** 2607.06624 | [PDF](https://arxiv.org/pdf/2607.06624v1)

**作者:** Andrey Podivilov `[一作]` (Explyt), Sergey Nikolenko `[通讯]` (Steklov Institute of Mathematics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了AgentLens基准，用完整交互轨迹和LLM评审评估交互式代码助手的生产质量。

**💡 创新点**

创新点在于将正式验证与LLM评审相结合，提供可解释的轨迹评估、质量指数和侧对侧比较，支持实时回归检测。

**🔧 技术方法**

采用LLM评审、正式验证器、轨迹日志、工具调用记录、CI流水线以及侧对侧评审等技术手段。

**📊 数据集**

使用16个基于真实工作流的Java编码任务，并配合默认与有毒用户模拟，产生32条交互轨迹。

**📈 对比分析**

通过多维度评分和质量指数对模型进行比较，展示排行榜并指出各模型在任务完成、工具使用、指令遵循等方面的差异，性能从高至低分布在不同模型间。

**⚠️ 局限性**

局限包括仅评估Java、特定任务类型；第三方API对可用性和成本影响大；LLM评审可信度缺乏正式协议。

---

## 307. A Closed-Loop Multi-Agent Framework for Robust Multi-Robot Manipulation

**arXiv ID:** 2607.06990 | [PDF](https://arxiv.org/pdf/2607.06990v1)

**作者:** Yi-Xiang He `[一作]` (Sun Yat-sen University), Wei-Shi Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个闭环多代理框架，实现了多机器人在复杂环境下的稳健协同操作。

**💡 创新点**

创新点在于将LLM驱动的规划、执行与验证三位代理耦合，并通过层级恢复机制实现对执行不确定性的自适应纠正。

**🔧 技术方法**

采用大语言模型进行任务分解与分配，视觉语言模型及工具化感知实现关键点定位与动作原语生成，验证代理通过视觉验证与错误诊断触发局部或全局重规划。

**📊 数据集**

实验使用自制的六个真实场景任务（包含桌面双臂、跨工作空间协作等），并在每个任务上进行20次重复试验。

**📈 对比分析**

与学习型方法OpenVLA‑OFT、π_0以及LLM型ReKep进行对比，本文方法在所有任务中成功率提升至80%以上，且在外部扰动下仍保持60%以上的平均成功率，显示出显著的鲁棒性与可行性。

**⚠️ 局限性**

局限在于依赖固定动作原语库，无法处理更复杂的操纵任务；执行速度受机器人异构性限制；缺乏动态调度优化，可能导致系统效率低下。

---

## 308. Deanonymizing Monero Transactions in Tor Network

**arXiv ID:** 2607.07062 | [PDF](https://arxiv.org/pdf/2607.07062v1)

**作者:** Ruisheng Shi `[一作]` (Beijing University of Posts and Telecommunications), Qin Wang `[通讯]` (Independent)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并实现了 ProxyMark 框架，能够在 Monero 节点使用 Tor 时解密交易来源。

**💡 创新点**

创新点包括：利用 Monero-Over-Tor 的代理节点转发机制、通过伪造区块高度提升代理选择概率，以及在 Tor 入口流量中嵌入可识别的水印以绑定真实 IP。

**🔧 技术方法**

技术手段包括：定时同步（Timed Sync）响应分析、灰名单/白名单填充、伪造区块高度、基于 Monero 消息计数的流量水印、以及对 Tor 中的入口 Relay 进行监测。

**📊 数据集**

数据集为：Monero 主网与测试网节点流量、真实 Tor 网络流量以及在实验环境中部署的数千个伪造 Tor 隐藏服务节点。

**📈 对比分析**

与现有攻击相比，ProxyMark 在隐藏服务节点上达到 100% 精度、约 93.8% 召回率，在客户端节点上也实现 100% 精度、约 91.4% 召回率，且占用的资源与对手成本显著低于传统的 Sybil 或 DoS 攻击。

**⚠️ 局限性**

局限性包括：需要对目标节点使用 Tor 并暴露特定的协议行为；必须在 Tor 入口中控制至少一个 Relay；若目标节点更换 Guard 或使用多重隧道，成功率会降低；并且若 Monero 协议或 Tor 的实现做出相应修补，攻击效果将被削弱。

---

## 309. Toward Deployable Satellite Anomaly Detection: A Benchmark Study on Large-Scale ESA-ADB Telemetry

**arXiv ID:** 2607.07335 | [PDF](https://arxiv.org/pdf/2607.07335v1)

**作者:** Andrea Nguyen `[一作]` (University of Manitoba), Peng Hu `[通讯]` (University of Manitoba)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统性评估了在ESA-ADB大型卫星遥测数据上，监督式（Multiscale CNN、GCN、GAT）与无监督式（EE、ECOD）异常检测方法，并对其检测性能与计算成本进行比较。

**💡 创新点**

创新点在于首次跨范式对多尺度CNN、GCN、GAT与无监督方法进行统一实验，并在检测性能之外深入分析运行时与可扩展性，为实际卫星健康监测提供操作性指南。

**🔧 技术方法**

使用了多尺度卷积神经网络、图卷积网络、图注意网络、椭圆包络、经验累积分布函数异常检测等技术。

**📊 数据集**

采用ESA-ADB基准数据集，包含两套不同时间尺度的任务（Mission 1：84 个月；Mission 2：21 个月），涵盖多维遥测通道。

**📈 对比分析**

通过事件级指标（Precision、Recall、F0.5、Accuracy、PR‑AUC）以及训练时长比较，监督模型整体表现最佳，Multiscale CNN在高维场景下PR‑AUC最高；无监督方法ECOD在精度上与监督模型相近但召回率低，且计算成本显著更低。

**⚠️ 局限性**

局限在于模型训练需大量标注数据与计算资源，且无监督方法在召回率与高维数据鲁棒性方面仍不足；实验仅覆盖ESA-ADB数据，未验证跨任务迁移或实时部署效果。

---

## 310. Physical activities enable scalable foundation modelling for broad-spectrum health prediction

**arXiv ID:** 2607.06954 | [PDF](https://arxiv.org/pdf/2607.06954v1)

**作者:** Zhenghuang Wu `[一作]` (Beihang University), Songlin Xu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一个仅基于步数数据的基础模型，旨在统一预测20多种健康风险，构建了可在多设备、多地区、多疾病上迁移的通用表示；

**💡 创新点**

创新点在于：①利用低维、隐私友好的步数信号构建基础模型；②设计双流架构（宏观Mamba + 微观1D卷积+FiLM）与时间节律编码；③在预训练阶段加入层级行为表型对齐目标，实现对长周期行为模式的显式建模；

**🔧 技术方法**

技术方法包括：Mamba序列模型、对数尺度词元化、傅里叶时序编码、微观流FiLM调制、层级行为表型对齐、自监督预训练、线性探针、少样本与全样本微调、层级特征分析与可视化；

**📊 数据集**

数据集方面，预训练使用141.12 M分钟级步数（NHANES 11‑14）；下游评估在NHANES 05‑06、BarKA‑MS、RESILIENT等多源数据上，共21项健康风险预测任务；

**📈 对比分析**

与NormWear、Pulse‑PPG、PAT、SSCP、TimeSiam及传统手工特征LR基线对比，平均AUROC为0.7318，20/21任务领先；少样本实验表明30%标注即可获得95%完整性能；时间窗口从1天到7天递增，预测性能逐步提升并趋于平稳；层级特征分析显示中间层最佳；跨设备、跨地区、跨疾病迁移实验均表现出稳健的优势；

**⚠️ 局限性**

局限性包括：①步数信号对某些疾病（如心理、呼吸等）关联弱，导致预测效果有限；②极端类别不平衡导致F1偏低；③模型仅适用于有分钟级步数数据的设备，对小时级或其他传感器需裁剪或重构；④预训练需要大规模步数数据，缺乏此类数据的研究环境可能受限；⑤模型虽然解释性较好，但对复杂多模态生理特征的捕捉仍不足。

---

## 311. On Explicit Super-Expressive Approximation for Neural Networks

**arXiv ID:** 2607.06781 | [PDF](https://arxiv.org/pdf/2607.06781v1)

**作者:** Feng-Lei Fan `[一作]` (City University of Hong Kong), Jian-Jun Wang `[通讯]` (Guangxi Minzu University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于中国剩余定理的固定架构神经网络近似框架，给出 Lipschitz 连续函数与 Hölder 平滑函数的显式参数-误差关系；

**💡 创新点**

首次在固定架构超表达式近似中给出了非渐进式、可量化的参数上界，并展示了平滑度提升能显著降低参数规模；

**🔧 技术方法**

利用中国剩余定理对网格编码，结合最小正交多项式激活（floor、ReLU、ReQU、递归激活）实现精确整数模运算与多项式逼近；

**📊 数据集**

无实验数据集，全部为理论证明与构造性网络设计；

**📈 对比分析**

与传统可扩张架构近似相比，本方法保持网络宽度与深度固定，唯一的成本是参数幅度随误差增大；在参数-误差曲线上实现显著优越的非渐进量化表现；

**⚠️ 局限性**

限制包括 CRT 产生的巨大整数模量可能导致参数指数级膨胀，以及使用的特殊激活函数非主流，影响实际可实现性。

---

## 312. NEST: Tackling Dataset-Level Distribution Shifts via Regime-Oriented Mixture-of-Experts

**arXiv ID:** 2607.06607 | [PDF](https://arxiv.org/pdf/2607.06607v1)

**作者:** Lanhao Li `[一作]` (Beihang University), Jianxin Li `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 NEST 框架，利用两阶段 Mixture-of-Experts 结构解决数据集级分布漂移问题。

**💡 创新点**

创新点在于通过 moment‑entropy 空间进行无监督的运行模式聚类，并结合结构化专用专家与几何上下文调制的路由器实现对不同模式的自适应组合。

**🔧 技术方法**

主要技术包括变体注意力专家、SVD 熵特征、k‑means 聚类、两阶段 MoE 训练、几何路由器调制、CKA 评估专家异质性。

**📊 数据集**

实验使用了九个基准数据集：CESNET 网络流量、TEC 电离层、Weather、ETTh1、ETTh2 等。

**📈 对比分析**

与 UniTS、iTransformer、PatchTST、FEDformer、Autoformer、Informer、DLinear 等现有方法对比，NEST 在 32/36 次实验中均获得最低 MSE/MAE，整体性能显著优于基线。

**⚠️ 局限性**

局限性包括专家数量与聚类参数对性能敏感，过多专家在低复杂度场景下可能冗余；路由与聚类在实时推理时的计算开销未充分评估。

---

## 313. An Hybrid Quantum-Classical Diffusion Model for Image Generation

**arXiv ID:** 2607.07072 | [PDF](https://arxiv.org/pdf/2607.07072v1)

**作者:** Qipeng Qian `[一作]` (University of Arizona), Yuntao Qian `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一个可扩展的混合态量子扩散模型与经典自编码器结合的生成管道，用于在小量子比特条件下生成经典图像。

**💡 创新点**

在潜在空间使用混合态量子扩散，简化反向更新为直接预测初始密度矩阵 ρ₀ 并通过解析逆传播一步，同时将量子扩散与经典自编码器/transformer结合。

**🔧 技术方法**

采用混合态量子扩散模型（MSQuDDPM）、自编码器、Pauli‑6 POVM、时间嵌入量子电路以及量子到经典 transformer 等技术。

**📊 数据集**

使用 MNIST 手写数字图像数据集进行实验。

**📈 对比分析**

与传统逐步预测 ρ_{t‑1} 的变体对比，使用 FID、KID、IS 三个指标评估，直接预测 ρ₀ 在无条件下获得更低 KID、更高 IS；加入 Q2CT 后 FID 进一步下降，整体生成质量提升。

**⚠️ 局限性**

仅在小量子比特、模拟环境下实验，量子电路去噪能力有限，生成图像仍存在噪点，需依赖经典解码提升质量。

---

## 314. Information Allocation Dynamics in Neural Network Optimization

**arXiv ID:** 2607.07156 | [PDF](https://arxiv.org/pdf/2607.07156v1)

**作者:** Zhang Gongyue `[一作]` (Harbin Institute of Technology Shenzhen), Liu Honghai `[通讯]` (Harbin Institute of Technology Shenzhen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文从训练时的更新动力学角度研究优化器隐式偏置，提出将权重与偏置视为不同的信号写入路径，并引入连续预处理指数p统一描述SGD‑和Adam‑风格的预处理，从而调节权重/偏置的相对更新比例。

**💡 创新点**

创新点在于：①把优化器的隐式偏置重新表述为权重和偏置路径的相对信息分配；②通过连续指数p实现对预处理强度的细粒度控制；③提出基于梯度统计的动态p调度机制，能在训练阶段实时调整信息分配；④在多任务上验证该机制对样本级别学习优先级和最终泛化的影响。

**🔧 技术方法**

使用了线性模型梯度分解、EMA预处理与指数p、梯度/更新统计、层级权重/偏置梯度比、样本损失分布、Jaccard硬样本迁移、动态p调度算法，并在视觉与情绪识别任务中进行实验。

**📊 数据集**

主要数据集包括构造的线性/视觉基准（如CIFAR、ImageNet等），以及情绪识别任务的RAFDB和AffectNet7。

**📈 对比分析**

通过与固定p、传统SGD/Adam以及动态p三种设置在损失曲线、验证精度、样本损失分布、硬样本迁移等指标的对比，发现动态p能够在FER任务上提升约0.15%准确率、加速收敛；在通用视觉任务中，p的调整能在中位数/尾部权衡之间取得平衡，展示了信息分配视角的有效性。

**⚠️ 局限性**

局限性包括：①最优p值高度依赖任务与训练阶段，缺乏统一的自适应策略；②动态调度目前仅基于粗略梯度统计，可能不够细粒度；③只考虑权重与偏置两类路径，对网络中其他参数类型（如归一化尺度、注意力权重等）的影响理解仍不充分；④未给出严格的理论证明，依赖经验验证。

---

## 315. Trees from Marginals: Autoregressive drafting with factorized priors

**arXiv ID:** 2607.06763 | [PDF](https://arxiv.org/pdf/2607.06763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 316. HumAIN: Human-Aware Implicit Social Robot Navigation

**arXiv ID:** 2607.07357 | [PDF](https://arxiv.org/pdf/2607.07357v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 317. Hybrid Least Squares/Gradient Descent Methods for MIONets

**arXiv ID:** 2607.06976 | [PDF](https://arxiv.org/pdf/2607.06976v1)

**作者:** Jun Choi `[一作]` (KAIST), Minam Moon `[通讯]` (Korea Military Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对多输入神经算子 MIONet，提出了一种混合最小二乘/梯度下降（LSGD）训练方法，利用交替最小二乘（ALS）优化每个分支网络的最后一层线性参数，并用 Adam 更新隐藏层参数。

**💡 创新点**

将 LSGD 从 DeepONet 泛化到 MIONet，并证明在 ALS 步骤中可以将庞大的 LS 系统通过克罗内克、哈特里‑罗乘积及张量置换矩阵分解为若干小矩阵，避免显式构造大矩阵，从而显著加速训练。

**🔧 技术方法**

使用 Kronecker/Khatri‑Rao 乘积、张量置换、谱分解求 Sylvester 方程、Adam 优化器、Swish 激活、He 初始化等技术实现高效训练。

**📊 数据集**

在三类 PDE 任务上验证：变源与扩散率的反应扩散方程、带变源的常系数输运方程、二维泊松方程。输入函数通过高斯过程采样生成，使用 FDM 产生真值。

**📈 对比分析**

对比 Adam‑only 与 ALS+Adam，使用训练损失衰减曲线与验证集 L^2 相对误差。实验表明 ALS+Adam 在训练损失下降速度和最终预测误差上均优于单纯 Adam。

**⚠️ 局限性**

局限性包括：方法仅适用于具有线性最后层的 MIONet；ALS 步骤在大规模数据或多分支网络时可能仍显耗时；需要预先构造完整的训练数据集；对其他损失函数或不满足特定线性假设的任务适用性尚未验证。

---

## 318. Layer-Respecting Linear Graph Layouts

**arXiv ID:** 2607.06968 | [PDF](https://arxiv.org/pdf/2607.06968v1)

**作者:** Alvin Chiu `[一作]` (University of California, Irvine), Songyu Liu `[通讯]` (University of California, Irvine)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在层约束（层宽小或BFS宽度小）下，利用动态规划实现弧图与线性圆柱图最小化边交叉数的固定参数线性时间算法。

**💡 创新点**

创新点在于将层宽作为参数，构造局部状态空间并通过跨层交叉判定实现全局最优，同时将结果推广至BFS宽度，克服了传统NP‑hard性难题。

**🔧 技术方法**

采用动态规划、状态枚举、宽度参数化、BFS层划分与跨层交叉判定等理论技术。

**📊 数据集**

本文未使用实验数据集，主要以理论证明与算法复杂度分析为主。

**📈 对比分析**

通过理论复杂度比较，证明在宽度为w时可在O(w⁴(w!)² n)时间内求得最优解；在BFS宽度为b时可在O(b⁴(b!)² n²)时间内得到最优解，显著优于一般图的NP‑hard性。

**⚠️ 局限性**

局限性包括需层宽或BFS宽度较小、常数与阶乘成正比导致大宽度下实用性差、未覆盖更一般的图绘制情形（如环面绘制等）。

---

## 319. Machine Learning-Based Battery State-of-health Prediction for Unmanned Aerial Vehicles Predictive Maintenance

**arXiv ID:** 2607.06791 | [PDF](https://arxiv.org/pdf/2607.06791v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 320. BUS: Brain-Inspired Unsupervised Self-Reflection for Advanced Multimodal Reasoning

**arXiv ID:** 2607.07361 | [PDF](https://arxiv.org/pdf/2607.07361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 321. LLM-Guided Task-Semantic Field Factorization for Industrial Process Forecasting

**arXiv ID:** 2607.06623 | [PDF](https://arxiv.org/pdf/2607.06623v1)

**作者:** Youcheng Zong `[一作]` (Northeastern University), Dakuo He `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了LLM指导的任务语义场分解（TSF）框架，在工业过程预测中通过离线LLM生成的语义卡片，将过程文档的变量语义映射为固定向量，并在每个时间窗口激活语义，使模型在训练和推理时能够直接利用变量与目标的语义关系；在保持现有时间序列骨干的同时，使用轻量级输入适配器提升预测性能。

**💡 创新点**

创新点在于①把过程文档的语义离线通过LLM生成并压缩为固定向量；②在每个数值窗口中动态激活语义，形成任务语义场，实现在输入层将语义与数值耦合；③仅在输入层加入极小的适配器，保持后端模型不变，既提升性能又兼顾部署效率。

**🔧 技术方法**

技术包括：使用LLM（如DeepSeek‑v4‑pro）生成语义卡片；文本渲染+text‑embedding‑v4得到语义方向矩阵；双路径输入适配器（原始数值路径+语义投影路径+共享偏置）；多种时序骨干（GRU、LSTM、Transformer、Informer、Mamba、iTransformer、PatchTST、ModernTCN）进行训练与评估。

**📊 数据集**

数据集包括：钢冶金的Ladle Preheating（预测锅炉温度）；尾矿浓度脱水的Thickener Dewatering（预测下渗浓度）；以及公开的IndPenSim（fed‑batch 生产中酶/抗生素浓度预测）。

**📈 对比分析**

与同一骨干下的基线（仅原始数值输入）在相同拆分、优化器、训练预算下对比。MAE、RMSE、R²及推理时延被评估。实验表明，TSF平均降低MAE 6.4%，在IndPenSim任务最大降幅达25.5%；额外推理时间<0.008 ms/step，参数增量仅1.8–3.0k。

**⚠️ 局限性**

局限在于需要完整、可靠的任务协议、变量表和过程描述；若缺失、冲突或质量差，则生成的语义卡片可能失效；模型仍受限于训练数据覆盖，对极端分布偏移的鲁棒性待进一步提升。

---

## 322. Evaluating RAG Metrics in Applied Contexts: An Experiment, Its Findings and Its Limitations

**arXiv ID:** 2607.07302 | [PDF](https://arxiv.org/pdf/2607.07302v1)

**作者:** Quentin Brabant `[一作]` `[通讯]` (Orange Research), Quentin Brabant (Orange Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对一个基于业务文档的问答数据集进行实验，评估了多种检索增强生成（RAG）系统的自动评估指标与人工评分之间的相关性。

**💡 创新点**

创新点在于：①将四个主流RAG评估库（Ragas、DeepEval、RAGChecker、Opik）中的指标统一在同一实验框架下进行比较；②采用人类评估者的5分等级打分作为“真实”标准，利用Pearson相关系数对指标与人工评分、召回率之间的关联进行系统性分析；③对相关性结果进行解释并指出指标可能捕捉到的潜在维度（如问题难度），从而揭示单系统实验的局限性。

**🔧 技术方法**

使用的技术包括：
- RAG系统：检索器采用密集+BM25混合检索；生成器使用 GPT‑3.5 在检索结果的前5段文档上生成答案。
- 评估指标：Ragas、DeepEval、RAGChecker、Opik 中的检索、整体与生成相关度指标。
- 人工评估：两名 NLP 专家使用 5‑分等级（1–5）对生成答案进行评分，并计算评分一致性（Pearson 0.85）。
- 统计分析：计算指标与人工评分、召回率之间的 Pearson（以及 Spearman）相关系数，绘制置信区间。

**📊 数据集**

数据集：96 个问答对，来源于 479 篇电信业务文档。每个问题配有参考答案和 0–4 个信息跨度（1–202 词，平均 1.3）。问题类型覆盖布尔、who/what/where/when、how、why、conditional、how many 等。

**📈 对比分析**

比较方法：对同一组答案分别计算指标分数与人工平均分，利用 Pearson 相关系数评估两者的线性关系；同时将检索指标与词级召回率做相关性比较。结果显示：
- RAGChecker 的 claim recall 与人工评分相关系数最高（≈0.7），超过传统检索召回率；
- METEOR 等传统 NLG 指标与人工评分相关性较好；
- 生成指标（如 hallucination、faithfulness）与人工评分相关性弱；
- Opik 的 moderation 指标与人工评分几乎无相关性。整体而言，指标与人工评分的相关性从弱到强不等，提示单一指标难以全面反映 RAG 性能。

**⚠️ 局限性**

局限性：
1. 仅评估单一 RAG 系统，难以排除问题难度等外部因素对指标相关性的影响；
2. 指标可能捕捉到非目标维度（如问题易难度），导致高相关性但评估价值有限；
3. 数据集规模有限（96 条问答），相关系数不稳定；
4. 参考答案不公开，限制了复现性与外部验证；
5. 对基于 LLM 的指标的内部机制缺乏解释，相关性结果解释不充分。

---

## 323. CompoVista: A Composition-Graph-Based Visual Analytics System for Compositional Analysis of Traditional Chinese Paintings

**arXiv ID:** 2607.07105 | [PDF](https://arxiv.org/pdf/2607.07105v1)

**作者:** Dekun Qian `[一作]` (Hangzhou Dianzi University), Zhiguang Zhou `[通讯]` (Hangzhou Dianzi University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Composition Graph及Canvas-based visual analytics system，支持对传统中国画集合进行结构化、查询式的构图分析。

**💡 创新点**

构造四层Composition Graph（实体、关系、空白空间、上下文）实现可查询的画面结构，并将其嵌入迭代的可视分析工作流。

**🔧 技术方法**

基于场景图（scene graph）与空白空间检测，使用图匹配/子图匹配实现检索，前端Vue+D3，后端FastAPI；可视化视图包括分布、关系、比较等。

**📊 数据集**

960幅传统中国画（覆盖唐至近代），含7587实体、1197关系，手工注释并构建Composition Graph。

**📈 对比分析**

采用基于上下文过滤+加权子图匹配的检索，支持多维归一化分布、关系和空间对比；用户研究显示能有效构建/修正 cohort，任务完成率 100%，满意度高；但比较直观性仍有提升空间。

**⚠️ 局限性**

检索仅限格式兼容的画面，手绘卷略作处理；关系稀疏导致关系视图不稳定；需要更大数据集、自动注释、对手卷建模、历史追踪、模糊查询支持。

---

## 324. Efficient Long-Horizon Learning for Learned Optimization

**arXiv ID:** 2607.06772 | [PDF](https://arxiv.org/pdf/2607.06772v1)

**作者:** Xiaolong Huang `[一作]` (Mila - Quebec AI Institute), Eugene Belilovsky `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种高效长时程元训练方法ELO，用于学习优化器。

**💡 创新点**

创新点包括失败感知恢复缓冲区重新分配计算资源、逐步专家指导的解耦方向与幅度监督，以及结合两者实现稳定高效的长时程学习。

**🔧 技术方法**

使用元学习、PES梯度估计、失败感知恢复缓冲区、逐步专家监督、解耦的方向-幅度监督目标。

**📊 数据集**

元训练在四个小规模8×8图像分类任务（MNIST、Fashion‑MNIST、CIFAR‑10、SVHN）上完成；下游在大规模视觉任务（ResNet‑50、ViT‑B/16在ImageNet‑1K）和语言建模任务（GPT‑2 124M/350M在FineWeb）上验证。

**📈 对比分析**

与基准学习优化器（Celo2、元素级MLP）、CURRICULUM baseline、以及手工设计的AdamW、Muon比较，ELO在长时间展开和分布外泛化上均表现更佳；特别是ELO‑Celo2在所有任务中超越AdamW并在语言模型上与Muon持平。

**⚠️ 局限性**

局限性：仅在350M规模以下评估；未涉及更复杂模型（MoE、稀疏注意力等）和其他结构化/二阶优化器；ELO在小任务上有JAX实现开销；实验资源受限。

---

## 325. SPECTRA: Context-Conditioned Spectral Movement Primitives for Robot Skill Generalization

**arXiv ID:** 2607.06978 | [PDF](https://arxiv.org/pdf/2607.06978v1)

**作者:** Boxuan Zhang `[一作]` (Technical University of Munich), Ahmed Abdelrahman `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种频域模仿学习框架 Spectral Movement Primitive (SMP)，通过低频任务带子空间学习与相位耦合的形状保持动态调节，直接生成既符合任务几何又满足机器人动态约束的运动轨迹。

**💡 创新点**

创新点在于：① 将轨迹压缩为有限时域傅里叶系数并只学习低频任务带，显著降低学习维度；② 采用帧意识的任务-参数化 GMM/GMR 仅预测低频系数，消除空间变换的统计负担；③ 通过相位调节而非点位裁剪保持轨迹形状，实现在关节速度/加速度约束下的动态可接受性。

**🔧 技术方法**

使用技术包括：有限时域傅里叶展开、任务框架正则化与相位对齐、GMM/GMR 统计学习、逆运动学映射、相位耦合动态调节、频域求导、以及 MuJoCo/Franka Panda 物理仿真与控制。

**📊 数据集**

实验数据集包含：四类仿真轨迹（figure‑eight、Lissajous 2:3、五瓣花、尖星）用于评估低频任务带重构与鲁棒性；跨板位置/方向/尺度的擦拭任务用于测试跨域泛化；真实 Franka Panda 上的圆形、figure‑eight 与开放 C 形轨迹用于验证实物执行。

**📈 对比分析**

通过与 ProMP、FMP、MSTOMP 以及世界框架/规范框架的 GMM/GMR 对比，SMP 在 Procrustes 对齐误差、轨迹抖动和关节动态违规率方面均显著优于对照方法；在跨板、跨方向的泛化实验中误差下降至 1/10 左右，验证了低频任务带与帧意识方法的有效性。

**⚠️ 局限性**

局限性包括：① 对尖锐或高频结构的轨迹需要更高频率，任务带选择仍依赖经验；② 仅约束关节速度与加速度，未考虑力、碰撞或接触约束；③ 主要适用于平滑周期或准周期任务，对非周期或极度不规则轨迹的适应性有限；④ 需要先进行相位对齐，非周期轨迹需额外处理。

---

## 326. InfraQR: Edge-Placed QR-Inspired Structured Patch Attacks on Infrared Vision-Language Models

**arXiv ID:** 2607.07288 | [PDF](https://arxiv.org/pdf/2607.07288v1)

**作者:** Xin Li `[一作]` (China University of Petroleum-Beijing at Karamay), Yahui Chai `[通讯]` (China University of Petroleum-Beijing at Karamay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 QR 风格的结构化补丁攻击（InfraQR），将小型边缘补丁插入红外图像边界，利用 CLIP 风格的代理编码器进行优化，从而在红外视觉-语言模型中造成显著误识。

**💡 创新点**

创新点在于：①将 QR 码的固定定位与可学习网格相结合，保持结构化外观并兼顾梯度优化；②采用边缘放置搜索而非直接附加到目标对象上，揭示外围结构扰动对全局视觉-文本表征的破坏能力；③在分类、字幕迁移和问答任务三种下游任务中实现跨模型跨任务的攻击转移。

**🔧 技术方法**

使用了结构化补丁参数化（固定锚点+可学习格子）、二值正则化、基于 CLIP 的分类/问答损失、离散边缘位置搜索以及 Adam 优化等技术。

**📊 数据集**

采用 300 张红外图像（Infrared‑Image‑Instruct‑12K 生成的 30 类、10 张/类）作为分类/字幕基准，构建 295 对（图像+问题）红外 VQA 子集进行问答评估。

**📈 对比分析**

与 HCB、AdvIC、AdvGrid 等基线比较，InfraQR 在四种 CLIP‑风格代理上平均攻击成功率近 98%，将 OpenAI CLIP 的识别精度从 98.67% 降至 0.70%；在字幕迁移和 VQA 任务中均实现了最大或第二大语义一致性/答案准确率下降，表明具有强的跨模型跨任务攻击能力。

**⚠️ 局限性**

局限性包括仅在数字环境下验证，未探讨物理可实现性和对打印/显示/传感器噪声的鲁棒性；依赖代理模型梯度；仅评估了有限的红外数据集和模型，可能缺乏对更广泛场景的通用性验证。

---

## 327. Two-Stage Multi-Modal Fusion with Adaptive Alignment for Action Quality Assessment

**arXiv ID:** 2607.07438 | [PDF](https://arxiv.org/pdf/2607.07438v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 328. TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation

**arXiv ID:** 2607.07287 | [PDF](https://arxiv.org/pdf/2607.07287v1)

**作者:** Jianyi Zhou `[一作]` (Harbin Institute of Technology), Shuo Yang `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一种名为TouchWorld的层次化触觉基础模型，用于在多任务、长时程的机器人操作中实现触觉预测与即时反馈的协同控制。

**💡 创新点**

创新点在于：①将语义规划、触觉子目标预测、视觉-触觉目标条件动作生成与高频触觉残差校正分别放在不同时间尺度上；②在视觉-语言动作框架内引入触觉子目标预测，为动作生成提供触觉导向；③采用残差Transformer实现高频触觉反馈，显著提升在碰撞、插入等细节任务中的鲁棒性。

**🔧 技术方法**

主要技术包括：视觉‑语言‑动作（VLA）流匹配Transformer策略、基于Diffusion Transformer的动作生成、触觉子目标预测模型（Tactile World Model）、残差Transformer（Tactile‑Conditioned Refinement Policy）、多尺度时间调度以及图像形式的触觉表示。

**📊 数据集**

使用的主要数据集：① EgoTouch人类触觉-视觉同步数据（20.2h）用于预训练触觉子目标预测；②10h机器人演示数据（约108万帧）用于细化预测模型；③每个任务200条遥操作训练轨迹，包含任务指令、相机、关节、动作与触觉观测；以及六个真实机器人任务的测试集（包含干净与人类扰动两种设置）。

**📈 对比分析**

与三种基线（Pi‑0.5、FTP‑1、GR00T N1.7）对比，TouchWorld在干净环境下平均成功率达到65.0%，在扰动环境下达到53.7%，分别比最强基线高出约15.7%和18.5%。实验还表明，去掉子任务规划、触觉子目标预测或高频反馈时，性能显著下降。

**⚠️ 局限性**

局限性包括：①仅在六个代表性任务上验证，未覆盖更广泛的家用或柔性物体交互；②触觉子目标预测仅覆盖短时程，长时程预测仍有挑战；③模型对特定触觉传感器与手部结构依赖较强，迁移到其他硬件需要额外校准；④时间调度参数为固定，缺乏自适应机制；⑤缺乏不确定性建模，难以处理多模态未来。

---

## 329. Immersive Social Interaction with VR and LLM-Assisted Humanoids

**arXiv ID:** 2607.07430 | [PDF](https://arxiv.org/pdf/2607.07430v1)

**作者:** Niraj Pudasaini `[一作]` (New York University Abu Dhabi), Yi Fang `[通讯]` (New York University Abu Dhabi)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套完整的类人机器人遥操作系统，支持通过语音控制步态、利用Apple Vision Pro进行手部追踪实现抓取与放置操作，并通过双向音频实现社交互动，同时收集多模态数据用于后续的模仿学习。

**💡 创新点**

创新点包括：①将语音识别、GPT‑4语义解析与深度强化学习结合，实现自然语言驱动的高层行走指令；②在同一界面实现全身运动与抓取控制，降低用户认知负荷；③收集同步的视角图像、语音文本、身体关节角度与眼动信息，构建丰富的遥操作数据集；④通过双向音频提升情境感知，为社交互动提供更自然的语音反馈。

**🔧 技术方法**

使用技术包括：Apple Vision Pro 与 VisionPro Teleop 进行手部姿态捕捉；Deepgram 语音转文本；GPT‑4 语义解析与命令生成；Silero 文字转语音；Pinocchio 逆运动学；PD 控制器；ROS1 进行双向音频传输；预训练的深度强化学习模型驱动双足行走；Inspire Robotics Dexterous Hands 作为末端执行器。

**📊 数据集**

本文未使用公开数据集，而是通过实验现场收集了视角图像、语音/文本命令、19个身体关节角度、12个手部关节角度以及眼动轨迹等多模态数据，用于评估与后续模仿学习。

**📈 对比分析**

通过对比新手与专家的操作结果，评估了成功率和执行时间；在物体抓取任务中，新手成功率为0.8、时间52秒，专家为0.9、22秒；在社交互动任务中，新手成功率为0.7、时间326秒，专家为0.8、158秒。同时与 Human Plus、Human to Humanoid 等现有方法对比，显示本系统在语音控制、全身交互与数据采集方面具有更高可用性和更低物理负担。

**⚠️ 局限性**

局限性包括：机器人双足姿态不自稳，行走稳定性受限；GPT‑4 可能误解指令，需额外确认步骤；仅使用头部视角导致环境感知不足，未来计划加入腰部摄像头；社交交互模块仍需细化；多模态数据处理与模仿学习算法尚未完善。

---

## 330. Single-Entity Spiking Neuron Models: Survey

**arXiv ID:** 2607.07429 | [PDF](https://arxiv.org/pdf/2607.07429v1)

**作者:** Leon Parepko `[一作]` (Innopolis University), Albert Nasybullin `[通讯]` (Innopolis University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

综述并系统分类单实体神经元模型，提出多维度对比表格，讨论各模型的优缺点与适用场景。

**💡 创新点**

首次整合 IF、HH、Izhikevich、DNM 等多类模型的特征，形成统一的评价框架，并对比其参数、运算量及实验验证情况。

**🔧 技术方法**

采用文献回顾、数学模型归纳、表格对比与性能指标（参数数、FLOPs、实验表现）进行分析，结合神经网络模拟技术（CNN‑LSTM+IZ）进行示例对比。

**📊 数据集**

未使用传统实验数据集，而是引用已有研究中的实验验证结果来评估模型性能。

**📈 对比分析**

通过比较模型的离散/连续性、变量数、参数量、计算量与实验匹配度等维度，展示不同模型在速度、精度、可解释性方面的差异；结果显示 Izhikevich、LIF 等模型在参数与 FLOPs 方面更优，而 HH 模型在生物学真实性上更突出。

**⚠️ 局限性**

仅聚焦单实体模型，未涉及多胞体或网络级别的复杂性；对生物学细节的近似和计算资源的限制仍是主要瓶颈，且未解决模型在外域数据上的泛化问题。

---

## 331. DeLS-Spec: Decoupled Long-Short Contexts for Parallel Speculative Drafting

**arXiv ID:** 2607.07409 | [PDF](https://arxiv.org/pdf/2607.07409v1)

**作者:** Hong-Kai Zheng `[一作]` (Nanjing University of Aeronautics and Astronautics), Piji Li `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种分离长短上下文的投机式解码方法DeLS-Spec，通过在已训练的DFlash块级解码器上加入轻量级局部头实现更高效的多token草稿验证。

**💡 创新点**

将DFlash的长上下文专家与独立训练的短上下文局部头分离，避免了从零开始训练或联合训练的高成本，并通过logit融合补偿块内因果依赖缺失。

**🔧 技术方法**

采用产品专家分解、无残差项的logit融合、轻量级RNN局部头、单独训练的下一个token预测任务以及unigram prior校正。

**📊 数据集**

在Qwen3-4B/8B模型上使用公开的DFlash训练语料以及Domino的训练集进行局部头训练，并评估于GSM8K、MATH-500、AIME25、HumanEval、MBPP、LiveCodeBench、MT-Bench和Alpaca等数学、代码与对话基准。

**📈 对比分析**

与自回归、EAGLE-3、DART以及DFlash等基线进行对比，DeLS-Spec在保持相同块大小的前提下，平均提高了4-6倍的速度加速并使接受长度提升0.3-0.5，特别在数学和代码任务上效果显著。

**⚠️ 局限性**

忽略了长短上下文交互的残差项，导致与全局联合训练方法（如Domino）相比仍有轻微性能损失，并且目前仅在DFlash框架下验证，未来需扩展到更广泛的并行解码器。

---

## 332. The Poisoned Chalice of LLM Evaluation Report

**arXiv ID:** 2607.07481 | [PDF](https://arxiv.org/pdf/2607.07481v1)

**作者:** Jonathan Katzy `[一作]` (Delft University of Technology), Zhou Yang `[通讯]` (University of Alberta)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织并评估了首届 Poisoned Chalice 竞赛，聚焦白盒成员推理以检测大型语言模型是否记忆了代码文件。

**💡 创新点**

创新点在于将成员推理技术与代码特定特征结合，提供专门的评测数据集、基准实现和公开评估流程，推动可信 LLM 评估的研究与实践。

**🔧 技术方法**

使用了多种成员推理攻击技术（Loss、MinK%Prob、PAC）以及新提出的 SERSEM（结构化熵加权评分）和 CalibratedProbs 等方法。

**📊 数据集**

构建了基于 Stack Edu（训练集）与 Heap（非训练集）的成员/非成员代码文件集，涵盖 Go、Java、Python、Ruby、Rust 等语言。

**📈 对比分析**

SERSEM 在 StarCoder2‑3B、7B 以及 Held‑out Mellum‑4B 上取得最高 AUC‑ROC（分别为 0.773、0.773、0.753），明显优于 Loss、MKP、CAL_PROBS 等基准；CalibratedProbs 在基准上略有提升，但仍低于 SERSEM。

**⚠️ 局限性**

主要限制包括高昂的计算成本（多层推断和输入扰动）、对精确率/召回率的权衡、以及在不同模型和语言上的泛化能力仍有待进一步提升。

---

## 333. SynthAVE: Scalable Synthetic Labeling for E-Commerce with LLM-Arena Validation

**arXiv ID:** 2607.07469 | [PDF](https://arxiv.org/pdf/2607.07469v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 334. LLM Assisted Verification Assertion Generation: Challenges and Future Directions

**arXiv ID:** 2607.07444 | [PDF](https://arxiv.org/pdf/2607.07444v1)

**作者:** Bhabesh Mali `[一作]` (Indian Institute of Technology Guwahati), Chandan Karfa `[通讯]` (Indian Institute of Technology Guwahati)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统评估了利用大型语言模型（LLM）自动生成 SystemVerilog Assertions（SVA）的现有方法与面临的主要挑战，并给出了改进的思路与实践指南。

**💡 创新点**

创新点包括：①将自然语言规范转化为结构化知识图谱或信号级结构化表示；②引入严格的信号映射预处理步骤，避免生成非 RTL 信号；③在正式验证日志的基础上构建迭代修正循环，显著提升 Assertion 的证明率；④提出 vacuity 与 COI（Cone‑of‑Influence）等更细粒度的质量度量。

**🔧 技术方法**

核心技术包括：Transformer‑based LLM（如 GPT‑4/Claude 等）、检索增强生成（RAG）、知识图谱构建与查询、Monte Carlo Tree Self‑Refine、Cadence JasperGold 等 FPV 工具、Mutation Score 与 MDR 等质量评估方法。

**📊 数据集**

使用公开的 4‑bit RCA 设计与其对应 RTL/规范作为实验案例，并引用工业级 IP 的 RTL 与自然语言规范进行评估；实验数据主要来源于上述设计的手工编写 SVAs、FPV 证明日志与覆盖率报告。

**📈 对比分析**

与 SANGAM、AssertLLM、AssertionForge 等现有框架对比，实验显示：①单次生成方法的 proven Assertion 比例约 10‑12%，而迭代修正循环可提升至 38% 以上；②基于知识图谱的框架在功能覆盖率上可达 88%（相较于 66% 的传统结构化方法）；③在覆盖率与 vacuity 指标上均表现出更高的真实性。

**⚠️ 局限性**

主要限制包括：LLM 对长上下文规范的处理仍受 token 限制；信号映射不严谨导致 vacuous proofs；现有评估指标（如覆盖率）仍无法完全反映 Assertion 质量；正式验证工具的执行时间与资源占用较高，需进一步优化迭代与剪枝策略。

---

## 335. Compact Rational Krylov for Parametrized Systems with Application to BEM Frequency Sweeping

**arXiv ID:** 2607.07440 | [PDF](https://arxiv.org/pdf/2607.07440v1)

**作者:** Kobe Bruyninckx `[一作]` (Ku Leuven), Karl Meerbergen `[通讯]`

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于CORK框架的左预/右预条件化的稀疏Rational Krylov GMRES方法，用于一次性求解参数化线性系统，特别是在Helmholtz散射问题的频率扫描中。

**💡 创新点**

创新点在于：1) 将CORK框架推广到一般伴随线性化，支持任意参数化矩阵；2) 引入右端项的参数依赖和不完全Krylov迭代；3) 通过贪婪式自适应选择移位，加速收敛；4) 结合ℋ-张量稀疏表示，实现高效的频率扫描。

**🔧 技术方法**

主要技术包括：伴随线性化、Compact Rational Krylov (CORK)方法、左/右预条件化的Rational Krylov GMRES、贪婪式移位选择、不完全Krylov理论、ℋ-张量（H-tensor）与ℋ-矩阵表示、有限元/边界元Galerkin离散、张量化AC/A的快速构造。

**📊 数据集**

使用了多种几何数据集：Trefoil knot、凹多面体（cube）、大型潜艇模型，频率范围覆盖κ·diam(Γ)≈[10,30]到[25.97,64.93]，对应数千到几十万未知数。

**📈 对比分析**

通过与逐频点的直接求解（naïve solve）以及纯ℋ-矩阵方法对比，展示了CORK-GMRES在迭代次数、内部BiCGStab(2)求解次数和总计算时间上的优势；在频率点数达到约40-55时，CORK方法已显著快于naïve方法；内存占用相对更高，但可接受。

**⚠️ 局限性**

局限性包括：1) ℋ-张量构造与存储成本较高，限制了高频或极大问题；2) 对线性化的缩放与预条件选择敏感；3) 在高频散射场景下，稀疏表示与张量压缩的有效性需进一步研究；4) 并行效率受块级负载平衡影响。

---

## 336. On the Assadi Liu Tarjan Auction Algorithm for Bipartite Matching: Simplification, Alternative Analysis, and Hard Instance

**arXiv ID:** 2607.07439 | [PDF](https://arxiv.org/pdf/2607.07439v1)

**作者:** Christian Konrad `[一作]` (University of Bristol), Eric Wang `[通讯]` (University of California, San Diego)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

改进并简化了基于最大匹配的ALT算法，消除冻结机制并提供可自适应的ε参数；

**💡 创新点**

提出了新的基于增广路的分析框架，以及证明了该算法在简单路径图上需要Ω(1/ε²)轮的硬实例；

**🔧 技术方法**

主要技术包括增广路长度与B顶点变更次数的关联、潜在函数替代、以及对冻结机制的消除；

**📊 数据集**

使用人工构造的线性路径图作为实验和下界证明的基准数据集；

**📈 对比分析**

与原Assadi-Liu-Tarjan算法相比，简化后可在多传递流模型下保持相同的(1-ε)近似精度；在路径图上实验显示仍需Ω(1/ε²)轮，证明无法通过该框架突破该下界；

**⚠️ 局限性**

局限性在于对所有图仍需Ω(1/ε²)次最大匹配计算，无法改进近似质量，且缺乏对更稠密图的更强理论或实验分析。

---

## 337. Multi-Agent Robotic Control with Onboard Vision-Language Models

**arXiv ID:** 2607.07403 | [PDF](https://arxiv.org/pdf/2607.07403v1)

**作者:** Kajetan Rachwał `[一作]` (Robotec.AI), Maria Ganzha `[通讯]` (Warsaw University of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了一套全托管、基于多智能体架构的自主移动操纵机器人控制系统，利用本地部署的紧凑型视觉语言模型实现安全检查、仓储维护、物体搜索、包装质量验证和人机交互五大任务；

**💡 创新点**

创新点包括：①引入“Megamind”监督型调度代理解决小模型长周期规划中的上下文保持问题；②对3B参数VLM进行任务专门微调提升包装检测精度；③将整个系统完全部署在机器人本地硬件（AMD Ryzen AI mini PC）上，消除对云端计算的依赖；

**🔧 技术方法**

所用技术包括：ROS 2、MoveIt 2、Nav2、RAI框架、LFM2-VL-3B（3B参数VLM）、Qwen3系列嵌入/重排序模型、FAISS向量数据库、OpenAI等；

**📊 数据集**

使用的数据集为：①基于仿真环境生成的包装检测图像与三属性标注（是否有效、是否良好、是否真/假）；②从OSHA 29 CFR 1910法规中提取并向量化的安全合规文本；

**📈 对比分析**

在硬件在环仿真中对系统进行验证；与传统基于云端大模型控制方案相比，采用紧凑VLM的精度提升至91.5% F1，且推理成本显著降低，运行时延满足实时需求；

**⚠️ 局限性**

局限性包括：①仅在仿真环境验证，真实机器人部署与性能尚未公开；②对极端外域数据的泛化能力仍待进一步评估；③本地计算资源有限时可能影响多任务并发处理的效率。

---

## 338. Smooth Operator: A Real-Time Sampling-Based Algorithm for Kinematic Hand Retargeting

**arXiv ID:** 2607.07491 | [PDF](https://arxiv.org/pdf/2607.07491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 339. Reward-Adaptive Iterative Discovery: A Case Study on Automated Game Testing for NHL26

**arXiv ID:** 2607.07498 | [PDF](https://arxiv.org/pdf/2607.07498v1)

**作者:** Florian Fuchs `[一作]` (Electronic Arts), Linus Gisslén `[通讯]` (Electronic Arts)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对EA SPORTS NHL 26的门将AI，提出并实验了一种基于强化学习的自动化游戏测试框架（RAID），通过迭代训练多代理，寻找多样化且高成功率的进球策略。

**💡 创新点**

创新点在于在每一次迭代中将已发现的类似策略在奖励函数中遮蔽，从而保持多样性；同时采用静态、直观的多样性阈值（如2 米距离），使非RL专家也能轻松配置。

**🔧 技术方法**

使用的技术包括：Soft Actor‑Critic（SAC）强化学习，支持混合离散/连续动作的网络结构；奖励遮蔽机制；迭代训练流程；手工设定的多样性阈值与策略收敛判据。

**📊 数据集**

数据集为EA SPORTS NHL 26的预发布游戏环境，包含单守门员+前锋对战，观察空间为位置、速度、方向等；实验全部在自建的游戏模拟环境中完成，没有公开数据集。

**📈 对比分析**

实验通过与无多样性遮蔽的基线（20次独立运行）对比，证明RAID能在10–30次迭代中发现6–8种与人类测试者相同的高成功率进球策略；基线仅收敛于两种重复策略。

**⚠️ 局限性**

局限性包括：仍需人工后期验证策略是否为真实漏洞；RL方差导致相同策略可能未被完全排除；迭代次数线性增长导致运行时间较长；对更复杂场景的多样性定义尚未完善。

---

## 340. Agent-Exploitation Affordances: From Basic to Complex Representation Patterns

**arXiv ID:** 2607.07475 | [PDF](https://arxiv.org/pdf/2607.07475v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 341. TimEE: End-to-end Time Series Classification via In-Context Learning

**arXiv ID:** 2607.07500 | [PDF](https://arxiv.org/pdf/2607.07500v1)

**作者:** Jaris Küken `[一作]`, Lennart Purucker `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于上下文学习的时间序列分类模型Timmy，能够在不进行数据集特定训练的情况下直接给出预测概率。

**💡 创新点**

创新点在于利用完全由结构化VARX生成的合成分类任务作为预训练先验，实现端到端的ICL，并通过标签条件化提升表示学习。

**🔧 技术方法**

采用Transformer结构的tokenization、label conditioning、跨系列注意力、混合增强及Muon/AdamW等技术进行预训练和推理。

**📊 数据集**

主要在UCR 128个单变量、UEA 24个多变量时间序列数据集上进行评估。

**📈 对比分析**

与经典方法MiniRocket、Hydra、现代深度学习InceptionTime以及基础模型NuTime、MantisV2等对比，Timmy在ROC AUC上排名第一、准确率第三，且推理速度位于Pareto前沿。

**⚠️ 局限性**

局限在于最多支持10类任务时采用one‑vs‑rest处理多类数据，且多变量任务仅通过per‑variate方式提升有限，需进一步设计更强的先验和模型规模。

---

## 342. SpaCellAgent: A Self-Evolving LLM-Based Multi-Agent Framework for Trajectory Analysis

**arXiv ID:** 2607.07467 | [PDF](https://arxiv.org/pdf/2607.07467v1)

**作者:** Songhan Wang `[一作]` (University of Shanghai for Science and Technology), Wenjing Yang `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为SpaCellAgent的自主、可自我进化的LLM驱动多智能体框架，用于完成从空间单细胞测序数据的预处理、轨迹推断到生物学报告生成的端到端工作流

**💡 创新点**

创新点在于将大型语言模型与多智能体协作、动态工具调度、双层验证与自我改进机制结合，形成闭环的自动化轨迹推断流程；并引入自我进化模块持续积累分析模板和错误修复知识，实现跨任务知识迁移

**🔧 技术方法**

核心技术包括：大型语言模型（如DeepSeek‑V3）驱动的规划者、执行者、评估者与报告者四大智能体；动态工具选择与代码生成；双层（代码+生物学）验证；自我改进循环；知识增强的回退机制；双层记忆体系与工具注册动态扩展

**📊 数据集**

使用六个异质数据集：REAL‑GOLD、REAL‑SILVER、SYNTHETIC（三类合成/真实scRNA‑seq基准），以及三组真实空间组织数据——小鼠胚胎背侧中脑、蝾螈神经再生以及私有的鼠脊髓损伤（SCI）数据

**📈 对比分析**

与传统轨迹推断方法（DPT、RaceID/StemID、Scorpius、PAGA、PAGA Tree、Slingshot）及LLM基线进行对比；在所有基准上SpaCellAgent在大多数评估指标（Spearman相关、F1分支、加权相关、HIM距离）均位列第一，平均提升约41.2%的分析效率，整体耗时从人类平均64.6分钟下降至38.0分钟

**⚠️ 局限性**

主要限制包括：依赖大规模预训练LLM的算力与API成本；在极端稀疏或极大规模数据上可能面临推断瓶颈；缺乏对实验验证的直接支持，生成的生物学假设仍需实验室进一步验证；框架对特定工具链的依赖可能限制在新技术快速迭代时的即时适配

---

## 343. From Decision to Random Certificates: Exponential Separation for Edge Estimation with Independent Set Queries

**arXiv ID:** 2607.07483 | [PDF](https://arxiv.org/pdf/2607.07483v1)

**作者:** Debarshi Chanda `[一作]` (Indian Statistical Institute), Gopinath Mishra `[通讯]` (Institute of Mathematical Sciences, HBNI)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种利用“独立集与随机证书查询”（ISC）在子线性查询量下，给定无向无权图估计其边数的随机算法。该算法在成功概率至少2/3的前提下，给出(1±ε)近似估计。

**💡 创新点**

创新点在于：
1) 将传统的独立集查询（IS）增强为带随机证书的查询，使得在同样的查询结构下显著提升信息获取能力；
2) 通过构造“近似保留子序列”，实现输出敏感的查询复杂度，仅依赖于边数m，而与顶点数n无关；
3) 将群检验、决策-计数二分法、图属性测试以及条件采样等多领域思想统一到一种新型查询模型中，展示了随机证书对计数任务的强大作用；
4) 在理论上实现了与标准IS查询和全局随机边采样模型的指数级查询复杂度分离。

**🔧 技术方法**

主要技术包括：
- 生日悖论（Birthday Paradox）检测子图中是否存在足够多边，快速估计小规模子图的边数；
- 图稀疏化（Graph Sparsification）与随机划分，证明至少有一半子图保留常数比例的边；
- 近似保留子序列（Approximation Preserving Sequence）构造，递归地缩小子图规模同时保持边量比例；
- 乘法重构（Multiplicative Reconstruction）通过估计每一步的边比率并累乘，得到整体边数估计；
- 组合Chernoff/ Hoeffding界、马尔可夫不等式等概率工具保证误差与失败概率。

**📊 数据集**

该工作主要是理论分析，不涉及具体数据集。所有结论均在通用无向无权图模型（V, E）上给出；若需要实验验证，可在随机图（如Erdős–Rényi图）或稀疏图上进行。

**📈 对比分析**

与现有方法对比：
- 标准IS查询：Θ(min{√m, n/√m})次查询；
- 全局随机边采样：Θ(√m)次采样；
- 本算法：O(log²m)次ISC查询，满足(1±ε)近似，成功率≥2/3。若m远大于n（稀疏图），本方法实现指数级的查询复杂度提升；若m≈n²（稠密图），仍保持对数级复杂度。

**⚠️ 局限性**

局限性包括：
- 需要对图提供ISC查询接口，该接口在实际图数据库或网络环境中实现成本未知；
- 结果是随机算法，误差随ε和m的对数级增加；
- 对于有权图、指向图或动态图等情况未作扩展；
- 常数因子和实际运行时间尚未在实验中验证，理论分析主要关注查询次数。

---

## 344. Biased or Personalized? The Impact of Personal Information on AI-driven Development

**arXiv ID:** 2607.07480 | [PDF](https://arxiv.org/pdf/2607.07480v1)

**作者:** Erfan Entezami `[一作]` (University of Massachusetts Amherst), Madeline Endres `[通讯]` (University of Massachusetts Amherst)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对800个AI生成的网站进行对照实验，并结合20名用户的观察研究，探究开发者年龄和性别对AI生成软件的影响。

**💡 创新点**

创新点在于系统性评估了个人信息在接口设计、模板内容和代码结构三维度上的潜在偏差，并将实验结果与真实开发者体验结合。

**🔧 技术方法**

使用的技术包括ChatGPT‑4.1与DeepSeek‑V3.2两种LLM，采用零样本提示生成HTML/CSS及多语言项目，并用统计检验分析差异。

**📊 数据集**

数据集包含两类任务（个人网站、在线商店）下的800个自动生成的网站以及20名参与者的ChatGPT交互记录。

**📈 对比分析**

通过χ²、Fisher、Mann‑Whitney等统计方法对不同人群的输出进行比较，结果显示性别与年龄显著影响布局、颜色、内容与文件结构，差异在统计学上显著。

**⚠️ 局限性**

局限性包括仅考虑年龄与性别两个属性，模型范围有限，任务仅覆盖HTML/CSS与基础语言，样本规模与多样性不足，以及观察研究可能存在的被试观察偏差。

---

## 345. A Theory of Contrastive Learning with Natural Images

**arXiv ID:** 2607.07470 | [PDF](https://arxiv.org/pdf/2607.07470v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 346. Beyond Attack-Success Rate: Action-Graded Severity Scale for Tool-Using AI Agents

**arXiv ID:** 2607.07474 | [PDF](https://arxiv.org/pdf/2607.07474v1)

**作者:** Harry Owiredu-Ashley `[一作]` `[通讯]` (Independent Researcher), Harry Owiredu-Ashley (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个基于执行轨迹的七级严重性评分工具，用于评估使用工具的AI代理在红队攻击中的危害程度。

**💡 创新点**

引入了“动作分级严重性刻度”，并提供了可编程的oracle和LLM评审面板，弥补了传统单一成功率指标的缺陷。

**🔧 技术方法**

结合程序化轨迹解析与工具效应元数据、三名前沿语言模型评审、Krippendorff α等统计衡量技术。

**📊 数据集**

采用AgentDojo工作区套件，包含四个模型、两种防御以及多项注入任务共410个实验记录。

**📈 对比分析**

通过与传统二进制攻击成功率对比，显示严重性刻度揭示三种决策误判；LLM评审与oracle的一致性达α=0.91，误差低于0.4等级。

**⚠️ 局限性**

样本有限、严重等级稀疏、需目标显式、逆向性依赖环境、评审对升级链识别不足。

---

## 347. Mitigating Taint-Style Vulnerabilities in MCP Servers via Security-Aware Tool Descriptions

**arXiv ID:** 2607.07461 | [PDF](https://arxiv.org/pdf/2607.07461v1)

**作者:** Yang Shi `[一作]` (Tongji University), Kaifeng Huang `[通讯]` (Tongji University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对MCP服务器漏洞进行系统分析，发现污点型漏洞占主导，提出通过在工具描述中嵌入安全约束并加入LLM自我反射的文本级防御方案

**💡 创新点**

创新点在于：①利用工具元数据的可读文本信息实现无代码、轻量级的安全修护；②将LLM自我反射嵌入工具调用环节，主动识别与校正潜在漏洞利用；③提供可迁移的描述增强模板，提升多漏洞、跨服务器的泛化能力

**🔧 技术方法**

技术方法包括：MCP协议元数据解析、污点风险识别、工具描述增强（Risk Identification + Description Enhancement）、LLM自我反射（Tool Invocation Reflection）、对抗性提示构造与实验评估，主要使用GPT‑4o进行推理与评测

**📊 数据集**

数据集：从NVD与GitHub收集的53条MCP漏洞（45台服务器），基于这些漏洞构建792条恶意攻击提示（包含多种注入/越界、不同注入策略）

**📈 对比分析**

评估方法：对比四种元数据设置（None、Decl、Wrong、Ident）与两种反射策略（Pre、Post），使用攻击成功率(ASR)衡量；结果显示无防御时ASR≈56%/64%，使用Ident+Post后ASR降至0.04%/0.13%，显著优于代码级修补且维护成本低

**⚠️ 局限性**

局限性：①依赖LLM对增强描述的遵循与准确性；②风险识别模型可能漏检未知或隐藏的漏洞；③实验仅针对GPT‑4o与构造的提示，缺乏多模型、多平台验证；④对高复杂度攻击或持续交互的安全评估尚未充分探讨

---

## 348. The Blind Curator: How a Biased Judge Silently Disables Skill Retirement in Self-Evolving Agents

**arXiv ID:** 2607.07436 | [PDF](https://arxiv.org/pdf/2607.07436v1)

**作者:** Xing Zhang `[一作]` (AWS Generative AI Innovation Center), Peiyang He `[通讯]` (AWS Generative AI Innovation Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在自我演化智能体中，使用基于失败的技能退休机制（Ratchet）时，若评判者（LLM judge）存在偏置（特别是将失败误判为通过），会导致退休机制失效（盲目守护）并可能导致技能库质量下降；同时提供了一种离线缺陷注入审计方法，可在部署前检测评判者的错误率。

**💡 创新点**

创新点在于：① 将误判率（尤其是 false-pass bias）对退休机制的影响进行定量分析，揭示存在阈值（cliff）使退休完全失效；② 在无参考答案的任务域（如长篇报告生成）中证明该机制的通用性；③ 设计了一种基于人工注入缺陷的审计流程，能快速判断评判者是否处于危险区间。

**🔧 技术方法**

使用的技术包括：Ratchet 自我演化框架（Critic、Synthesizer、Router、Curator），LLM 评判器（Claude Haiku 4.5 等），噪声注入（对真实标签进行对称噪声或 false-pass 偏置），统计检验与非发散性证明（Prop.1'），以及离线缺陷注入审计。

**📊 数据集**

使用的数据集：① 基于 155 条深度研究报告片段构成的长篇报告生成测试床（Report-main-71、Report-band-58、Report-hard-133）；② 代码生成基准 MBPP+（hard100）和对应的自我演化 Composer；③ 训练和评估过程中注入的多种缺陷类型（QC 可见与不可见）。

**📈 对比分析**

比较方法：在 12 轮演化循环中对不同奖励通道（clean QC、对称噪声、false-pass 偏置、LLM 评判器）进行 sweep，记录机制层（true retirement、eviction）、结果层（评估通过率、技能生成量）以及实际实现的错误率。结果显示：对称噪声对机制影响可容忍；false-pass 偏置在阈值附近导致退休失效并出现性能倒退；LLM 评判器在实际误判率低时保持机制正常。没有观察到显著的性能提升（lift）——主要的影响是风险而非收益。

**⚠️ 局限性**

局限性：① 仅在有限的任务域和 Composer（报告生成、代码生成）中验证，未覆盖更广泛的自我演化情景；② 评判错误被视为外部注入，未考虑评判器可被模型学习到的适应；③ QC 奖励仅关注引用规范，未评估内容深度或洞察力；④ 实验基于离散标签，未探索连续质量度量的细粒度影响。

---

## 349. The Optimal Sample Complexity of Learning Autoregressive Chain-of-Thought

**arXiv ID:** 2607.07423 | [PDF](https://arxiv.org/pdf/2607.07423v1)

**作者:** Zhiyuan Li `[一作]` `[通讯]` (Toyota Technological Institute at Chicago), Zhiyuan Li (Toyota Technological Institute at Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在可实现的PAC设定下，全自动回归Chain-of-Thought（CoT）完整轨迹的精确预测的样本复杂度，证明其与单步下一个动作的本地学习复杂度一致，不随轨迹长度变化。

**💡 创新点**

提出并利用“奇偶维度（parity dimension）”这一新的度量，证明它在自动回归“rollout”过程中不增大，并可用于从一维包含（one-inclusion density）推导出最优样本复杂度；同时给出DS维度在rollout下可能增大的反例，说明奇偶维度是更合适的不变量。

**🔧 技术方法**

主要技术包括奇偶维度定义、奇偶伪立方与低坐标张成（low‑coordinate spanning）、分区树剥离（partition‑tree peeling）以及一维包含密度（one‑inclusion density）与DS维度的关系推导。

**📊 数据集**

本工作为理论研究，无使用具体数据集，所有结论均在通用概率分布与理论框架下给出。

**📈 对比分析**

与现有的Trace‑Consistency、Compression、Online等三条路径相比，本文给出了真正的局部PAC上界，且证明该上界与下界（通过单步停止的特殊情况）匹配，达成了最佳（最优）样本复杂度；相较于以往方法的对数依赖或额外维度因子，本文消除了对轨迹长度的任何依赖。

**⚠️ 局限性**

局限性包括：仅处理确定性可实现（realizable）场景；未考虑噪声/对抗性标签、部分轨迹监督、随机策略或算法效率；奇偶维度和DS维度的计算在实际大规模模型中仍可能昂贵。

---

## 350. Initiation Safety: A Missing Dimension in Generalist-Robot Safety

**arXiv ID:** 2607.07420 | [PDF](https://arxiv.org/pdf/2607.07420v1)

**作者:** Zhijin Meng `[一作]` (University of New South Wales), Francisco Cruz `[通讯]` (University of New South Wales)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了“启动授权”（initiation authorization）框架，专门判断机器人在开始首次硬性社交行为（如问候、接触、探询等）前是否有适当权限，并在 PAL Robotics 的 ARI 双足机器人上通过 PAS（probe–authorize–speak）流程进行测试。

**💡 创新点**

核心创新在于将启动行为单独作为第三层安全保障，与物理安全和后期对话安全分离；引入可逆探测阶段、可调节的授权门阈和首次发声时的安全边际 Δ_init，实现对首次社交动作的主动把控。

**🔧 技术方法**

技术包括：视觉与声音感知融合的交互参与度评分器；四级非语音探测（转头、转身等）；阈值门控 τ(ρ) 以可调“大胆/保守”拨盘为策略；日志记录 Δ_init 以及机器人后续对话生成模型。

**📊 数据集**

研究未使用公开数据集，而是构建了基于门口交互的日志数据集（doorway traces），并在此基础上进行实验设计与结果记录。

**📈 对比分析**

通过三种启动策略（PAS、Direct‑init、Passive‑wait）的双盲间组实验，对每个会话记录探测动作、门控状态、Δ_init，并收集7分制主观评分（成功度、尴尬感、自然度）。结果显示 PAS 在保持合适的首次发声阈值的同时，显著提升用户体验；Direct‑init 的 Δ_init 较小但尴尬感提升。

**⚠️ 局限性**

局限性包括：仅在单场景（门口）验证；缺乏多方场景与跨人群授权的评估；对 VLA 操作（抓取、移动）启动授权尚未完整集成；安全阈值的设置依赖人工调节，缺少统一规范；未提供定量安全指标，只能依赖用户主观反馈和 Δ_init。

---

## 351. Revisiting Maximum $k$-Biplex Search Through $k$-Bounded-Degree Deletion

**arXiv ID:** 2607.07419 | [PDF](https://arxiv.org/pdf/2607.07419v1)

**作者:** Donghang Cui `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对二部图的最大 k‑biplex 搜索问题，提出了一种基于删删法（k‑bounded‑degree deletion）的精确算法，并辅以上界剪枝和启发式初始化方案。

**💡 创新点**

核心创新在于发现最大 k‑biplex 与补图中的最小 k‑bounded‑degree deletion 之间的结构对偶关系，并基于此设计了新的分支策略和分支约简技术；同时给出了非平凡的最坏情况时间复杂度 O*(γ_k^n)，其中 γ_1=1.725、γ_2=1.856、γ_3=1.928。

**🔧 技术方法**

技术方法包括：删删分支框架、基于度的分支选择、四类分支约简规则、两种线性时间的边上界估计（基于顶点和集合两种方式）以及基于深度搜索的启发式初始解构造。

**📊 数据集**

实验使用了八个真实世界的二部图数据集：Youtube、LKML、Mummun、Citeu、IMDB、Amazon、Aol、Google，涵盖从数万到数千万边的规模。

**📈 对比分析**

与两种主流基准算法（Sym‑BK 递归法和基于核心的剪枝法）相比，所提出的算法在所有 96 个测试案例中都能求解，并在大多数实例上实现 10‑1000 倍甚至 10,000‑倍的速度提升；在极难的高 k（≥3）或大阈值 θ 情况下表现尤为突出。

**⚠️ 局限性**

局限性包括：算法仍为指数级（尽管比 2^n 更优）；对非常稀疏或极大 k 的图仍可能因分支树过深而超时；上界估计和启发式解的质量仍依赖于数据特征，未保证在所有实例上都能快速收敛。

---

## 352. How Reliable Is the Multi-Input Heuristic for Bitcoin Address Clustering in Law Enforcement Contexts?

**arXiv ID:** 2607.07414 | [PDF](https://arxiv.org/pdf/2607.07414v1)

**作者:** Leopold Müller `[一作]` (University of Bayreuth), Christian Rückert `[通讯]` (University of Bayreuth)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文基于真实的欧盟加密资产服务商合规报告数据，系统评估了多输入启发式(MIH)在比特币地址聚类中的可靠性；

**💡 创新点**

创新点在于重新实现并统一了先前研究的九项评估指标，并首次使用合法来源的多实体真实标注数据进行验证；

**🔧 技术方法**

采用的技术主要是MIH聚类算法、九种聚类评估指标（如pairwise精度/召回、per‑wallet精度/召回、NMI/aNMI、AER）以及留一法敏感性分析；

**📊 数据集**

使用的数据集为截至2023年6月23日的七个欧盟加密资产服务商的伪匿名地址-实体映射，共约1100万地址和8.5亿笔交易；

**📈 对比分析**

与之前三项研究对比，数据集级别的pairwise指标仍表现较好（F1≈0.83），但per‑wallet和信息论指标显著下降（精度0.36、召回0.44、NMI≈0.41），并显示出实体间性能高度不均衡；

**⚠️ 局限性**

局限性包括仅覆盖七个服务商，未评估组合式或机器学习聚类工具，对区块链动态变化和不同业务模式的泛化不足，以及未提出专门针对司法需求的可靠性标准。

---

## 353. Heterogeneity-Adaptive Diffusion Schrodinger Bridge for PET-Guided Whole-Body MRI Translation

**arXiv ID:** 2607.07401 | [PDF](https://arxiv.org/pdf/2607.07401v1)

**作者:** Chengbo Wang `[一作]` (University of Sydney), Xiuying Wang `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

本研究提出了一种基于Heterogeneity-Adaptive Diffusion Schrödinger Bridge（HA-DSB）的PET引导全身MRI翻译方法，解决全身扫描中区域异质性和病灶重建难题。

**💡 创新点**

创新点包括：①利用大型视觉语言模型生成的体位与器官标签构建区域上下文嵌入，实现桥模型的区域自适应；②在前向噪声过程引入PET引导的空间噪声调制，突出病灶区的噪声扰动；③在反向去噪中加入多尺度PET注意力机制，显著提升病灶细节恢复。

**🔧 技术方法**

采用Diffusion Schrödinger Bridge框架结合条件学习、PET引导噪声调制模块、以及UNet结构中的多尺度PET注意力模块，并使用FiLM与跨注意力融合区域嵌入。

**📊 数据集**

使用来自SIGNA™ PET/MR系统的246例全身PET/MR数据（LAVA、T2、PET），尺寸256×256，划分为204例训练、42例测试，并单独评估21例已证实病灶的子集。

**📈 对比分析**

与GAN、传统扩散模型和桥模型等五个基线相比，HA-DSB在所有解剖区域的SSIM和PSNR均取得最高平均值；在病灶子集中，PET引导进一步提升PSNR约0.8dB、SSIM约1.7%。

**⚠️ 局限性**

主要局限包括：①PET引导在整体测试集上提升有限，主要受病灶体积占比低影响；②依赖VLM生成的标签，标签误差可能影响区域嵌入；③模型规模和训练成本相对较高。

---

## 354. Should We Dangle a Carrot? The Effect of Performance-based Incentives in Visualization Experiments

**arXiv ID:** 2607.07463 | [PDF](https://arxiv.org/pdf/2607.07463v1)

**作者:** Abhraneel Sarma `[一作]` (Graz University of Technology), Alexander Lex `[通讯]` (Graz University of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文开展了两项前注册的众包实验，评估在相关性感知与不确定性决策两个视觉任务中加入绩效奖金是否能提升受试者表现；

**💡 创新点**

创新之处在于首次系统性检验金钱激励在众包视觉实验中的实际影响，并将其视为实验设计中的隐含自由度（tacit factor）进行探讨；

**🔧 技术方法**

主要技术包括使用Prolific平台收集数据，构建贝叶斯二项式逻辑回归模型（感知任务）与线性对数赔率模型（决策任务），并通过JND、预期效用等指标量化表现；

**📊 数据集**

所用数据集为人工生成的合成数据：相关性任务中的正相关散点与平行坐标图对，决策任务中的正态分布温度预测及其对应的冻结概率；

**📈 对比分析**

实验比较了支付固定费 vs 绩效奖金两种激励方案，结果显示两方案在JND与预期效用上无显著差异，唯一差异是激励组完成任务所用时间略长；

**⚠️ 局限性**

局限性包括仅覆盖两类任务且采用较低奖金水平、受众为Prolific众包工人，可能影响外部效度与一般化，还未检验更大规模或更高激励的效应。

---

## 355. Beware of Agentic Botnets: Scalable Untargeted Promptware Attacks via Universal and Transferable Adversarial HalluSquatting

**arXiv ID:** 2607.07433 | [PDF](https://arxiv.org/pdf/2607.07433v1)

**作者:** Aya Spira `[一作]` (Tel Aviv University), Ben Nassi `[通讯]` (Tel Aviv University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

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

## 356. Reason Less, Verify More: Deterministic Gates Recover a Silent Policy-Violation Failure Mode in Tool-Using LLM Agents

**arXiv ID:** 2607.07405 | [PDF](https://arxiv.org/pdf/2607.07405v1)

**作者:** Vikas Reddy `[一作]` (Independent Researcher), Abhishek Basu `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一种轻量级、确定性的预执行门（deterministic pre‑execution gates），用于阻止工具在满足语法正确但违反域政策的情况下执行写操作，从而消除LLM代理在政策可容许工具中出现的无声错误状态；

**💡 创新点**

创新点在于将门层设计为只读、无模型调用的判定器，针对可决策的状态政策规则，证明在政策容许工具环境下该干预不仅提升任务成功率，还能显著提高可靠性；

**🔧 技术方法**

技术包括：①定义确定性门函数 g(tool_name, args, db_state) → {allow, reject}；②在工具调用前拦截并检查；③基于Python谓词实现四个门（取消、行李、乘客数、未读取记录写入）；④使用配对自举法评估统计显著性；

**📊 数据集**

数据集为 τ^2‑bench 航空域任务集（50 个任务，5 次试验/任务），以及 15 种不同种子复现实验；此外还使用了 gpt‑5.2 前沿模型的单次试验作为建议性验证；

**📈 对比分析**

比较方法为同一 harness 下的 vanilla 与 gated 条件，使用 pass_1 及 τ‑bench unbiased pass_k 评估，配对自举检验显著性。结果：在 gpt‑4o‑mini 上，pass_1 从 29.6% 提升至 42.0%（+12.4pp，P=0.0012），并在 15 种种子复现中保持一致；在前沿模型上，成功率从 61.2% 提升至 71.6%（+10.4pp，P=0.020）；对未触发门的任务提升不显著；

**⚠️ 局限性**

局限性包括：仅在单一领域（航空）验证；前沿模型结果未复现；门的精确度不均衡（有门误阻）；未保证任务最终成功，仅阻止违规写；仅适用于可决策的状态政策；未对任务外的恢复策略做深入研究；实验环境与公开基准不可直接比较。

---

## 357. Do LLM-Generated Skills Make Better AI Data Scientists? A Component Ablation Across Data-Science Workflows

**arXiv ID:** 2607.07504 | [PDF](https://arxiv.org/pdf/2607.07504v1)

**作者:** Wei-Jung Huang `[一作]` `[通讯]` (Independent Researcher), Wei-Jung Huang (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估低手工化LLM生成的技能文件在数据科学工作流中的效果，跨四个生命周期阶段进行单轮测试。

**💡 创新点**

以统一的单一技能文件覆盖完整生命周期阶段进行评估，并进行组件消融、长度对照与优先级指令实验，揭示技能内容对不同任务难度的影响模式。

**🔧 技术方法**

大规模实验使用九种模型配置（OpenAI、Google、Anthropic），混合效应模型、bootstrap、McNemar等统计方法评估技能效果；Token匹配对照与优先级指令控制。

**📊 数据集**

56个可验证执行任务，涵盖数据清洗（CSV）、SQL查询（Spider、Chinook/Northwind）、统计分析（Iris、mtcars、PlantGrowth、ToothGrowth）、报告生成（GitHub Events、OpenWeatherMap、REST Countries）等公开数据与自研样本。

**📈 对比分析**

与无技能提示进行比较，计算通过率差异；混合效应模型、bootstrap置信区间、McNemar检验等；结果显示无技能与完整技能差距≤1.2pp，均未显著提升，甚至在部分阶段略有下降。

**⚠️ 局限性**

样本范围有限（单轮、四个生命周期、17个信息区任务），未包含专家手写技能、任务特定技能、交互式规划与反馈；长度效应与技能多样性未完全排除；模型与任务的重叠可能影响难度评估。

---

## 358. Search, Fail, Recover: A Training Framework for Correction-Aware Reasoning

**arXiv ID:** 2607.07492 | [PDF](https://arxiv.org/pdf/2607.07492v1)

**作者:** Dmitry Beresnev `[一作]` (Innopolis University), Petr Anokhin `[通讯]` (AXXX)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了 Pyligent 框架，利用任务验证器将失败分支转化为监督样本，训练大型语言模型学习显式回溯与恢复行为；在隐藏有向图、4×4 数独、带推理轨迹的数独以及 Blocksworld 等任务上进行实验。

**💡 创新点**

创新点在于：① 将验证器生成的失败分支信息作为监督信号，显式训练回溯动作；② 引入“traced recovery”机制，将回溯时的失败原因记录为追踪信息，帮助模型更好地从回溯后继续推理；③ 通过两阶段 SFT（gold-chain 训练 + 验证器引导探索）实现对回溯策略的学习。

**🔧 技术方法**

使用的技术包括：语言模型动作生成（继续、完成、回溯三种动作）；验证器判定动作合法性；探索器（线性与树枝式）在已生成的树上采样；链树结构记录成功与失败分支并生成监督对；SFT-A 与 SFT-B 训练循环；推理时动态截断与追踪注入。

**📊 数据集**

所用数据集：
- 隐藏有向图任务（300 个样本）；
- 4×4 数独（混合和专家难度各 1,200 个训练样本，测试 200 个）；
- 带推理轨迹的 4×4 数独（同样大小）；
- Blocksworld（3–12 块，3,035 个训练实例，2,740 个测试实例）。

**📈 对比分析**

比较方法：与基线模型、仅金丝细化（Gold-only SFT）以及不同探索器/追踪设置的 Pyligent 检查点进行对比；结果显示：
- 隐藏图任务：Pyligent 76.5% 成功率，Gold-only 3.8%；
- 4×4 数独混合集：Pyligent 82% vs Gold-only 65%；
- Blocksworld：Pyligent 50% vs Gold-only 29%；
- 追踪提升整体准确率，回溯质量显著提高。

**⚠️ 局限性**

局限性：
- 仅在可精确验证的合成/小规模任务上验证，难以直接推广到开放式自然语言推理；
- 需要额外的验证器与探索成本；
- 对追踪与失败分支的权重敏感，过多训练失败分支可能不提升最终成功率；
- 当前使用的模型规模有限，未评估更大模型的表现。

---

## 359. FourierQK: Spectral Preprocessing of Query-Key Projections Improves Transformer Attention

**arXiv ID:** 2607.07478 | [PDF](https://arxiv.org/pdf/2607.07478v1)

**作者:** Athanasios Zeris `[一作]` `[通讯]`, Athanasios Zeris

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在 Transformer 的 Query/Key 投影上使用 FFT 频域滤波，以提升注意力计算的效果。

**💡 创新点**

创新点在于证明即使是随机或单一学习的频率滤波器，也能显著降低字符级语言建模的验证损失；并且区分了全局 FFT 预处理与因果时域滤波、FNet 等架构的区别，揭示了非对称卷积无法获得同样提升的原因。

**🔧 技术方法**

主要技术包括：双向 FFT 频域预处理（随机滤波器、单频学习高斯选择器、多频学习），Causal 时域卷积、Shuffle 验证、EMD 验证、多种位置编码（学习、无、正弦）以及多重对照实验。

**📊 数据集**

使用 Tiny Shakespeare（字符级，序列长度 256）的语言建模数据集进行实验。

**📈 对比分析**

通过与标准点积注意力（BASE-DOT）对照，随机频谱滤波提升 0.443、单一学习频率提升 0.600、四频多尺度提升 1.166（验证损失下降 79%）；相同实验设置下，随机正交/非正交投影、因果卷积等对照均无显著提升，Shuffle 验证表明改进来源于真实序列学习而非位置泄漏。

**⚠️ 局限性**

实验局限包括：仅在小规模（≤6M 参数）字符级模型上验证；FFT 双向预处理存在周期边界泄漏问题，因果时域滤波在字符级无效；未在更大模型或词级数据集上检验推广性。

---

## 360. Where to Intervene? Benchmarking Fairness-Aware Learning on Differentially Private Synthetic Tabular Data

**arXiv ID:** 2607.07471 | [PDF](https://arxiv.org/pdf/2607.07471v1)

**作者:** Vinícius Gabriel Angelozzi `[一作]` (Inria Centre University Grenoble Alpes), Héber H. Arcolezi `[通讯]` (ÉTS Montréal, Inria Grenoble)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

系统评估了在差分隐私合成表格数据上应用的公平性干预措施，比较了预处理、内处理和后处理三类干预点，并探讨了它们在不同隐私预算下的公平‑效能平衡。

**💡 创新点**

首次在差分隐私合成数据场景下构建统一基准并量化三类干预对公平性指标（MAD、EOD、SPD）和效能指标（准确率、F1）的影响，指出后处理在此场景下最稳健，并公开实验代码和数据。

**🔧 技术方法**

使用了AIM（Adaptive Iterative Mechanism）和MST两种基于边缘的差分隐私合成器；AIF360中的公平干预（Reweighing、DIR、LFR、EGR、GSR、ROC、EqOdds、CEOP）；XGBoost、Logistic Regression、Random Forest分类器；Python + SmartNoise + AIF360。

**📊 数据集**

四个公开数据集：Adult、COMPAS、ACSIncome、BiasOnDemand（Synthetic），涵盖性别/种族公平性研究。

**📈 对比分析**

通过20个随机种子、多种ε（0.05–20）对四种管线配置（Baseline、DP-only、Fair-only、DP+Fair）进行比较；结果显示DP单独会降低效能并放大差异，而公平干预能部分恢复公平性，后处理方法（ROC、EqOdds）在多数指标上实现最佳公平‑效能折中；预处理权衡更大，内处理较温和。

**⚠️ 局限性**

仅覆盖二元分类、单一受保护属性、基于边缘的DP生成器，未考虑多分类、回归、交叉属性、深度生成模型、个体公平等；实验使用默认超参，可能与精细调优产生差异；公平度量仅限群体级别。

---

## 361. Evaluating Static and Process Evidence for Code Authorship in Programming Education

**arXiv ID:** 2607.07400 | [PDF](https://arxiv.org/pdf/2607.07400v1)

**作者:** Marek Horváth `[一作]` `[通讯]` (Technical University of Kosice), Marek Horváth (Technical University of Kosice)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在教育代码仓库与编程竞赛环境下的代码作者归属问题，评估静态代码特征与可见的过程（提交历史）特征在不同生产情境下的有效性，并在任务感知的交叉验证框架中比较多种模型与评估指标。

**💡 创新点**

创新点在于：①首次在教育场景下采用任务感知拆分，避免任务泄漏导致的误判；②将提交过程特征与静态特征结合，证明过程特征能显著提升弱信号环境下的归属排名；③系统比较了传统统计特征、TF‑IDF稀疏表示、卷积/Transformer等多种序列模型，揭示不同表示在竞赛与教育数据中的表现差异。

**🔧 技术方法**

技术手段包括：随机森林、逻辑回归、梯度提升树等基于显式特征的分类器；TF‑IDF文档+线性SVM/岭回归/朴素贝叶斯的稀疏表示；字符CNN、token CNN、BiLSTM、Transformer编码器的序列模型；配对与原型验证（cosine相似度、逻辑回归差分特征）以及特征消除与信息增益等特征重要性分析。

**📊 数据集**

使用的数据集为：教育课程仓库（约11k名学生、256k文件，包含C、Java、Python），编程竞赛数据（ICPC/GCJ 2008‑2023年，约700k文件），以及合成的C代码变体。

**📈 对比分析**

评估方法：任务感知留一组交叉验证，计算Top‑1/Top‑5/Top‑10准确率、AUC、AP等；静态与过程特征分别构建并对比，组合模型与TF‑IDF组合；序列模型在相同划分下进行。实验结果显示：在竞赛数据Top‑1可达0.938；在教育数据中仅静态特征Top‑1为0.094，加入过程特征后提升至0.233；Top‑5提升至0.580；验证AUC在教育上从0.556升至0.752。

**⚠️ 局限性**

局限性包括：①标签为课程仓库归属，未必是唯一作者；②过程特征受提交策略与仓库策略影响，非完整工作轨迹；③任务感知拆分仍可能存在跨任务相似度泄漏；④实验在特定课程、语言、工具与时间段内，结果不一定可推广至所有教育环境；⑤高AUC不等同于高精度，验证AP较低，说明在大类不平衡情况下仍需人工校准。

---

## 362. Agentic Data Environments

**arXiv ID:** 2607.07397 | [PDF](https://arxiv.org/pdf/2607.07397v1)

**作者:** Elaine Ang `[一作]` (Columbia University), Eugene Wu `[通讯]` (Columbia University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Agentic Data Environments（ADE）框架，提供AIM（信息管理）、AIR（信息检索）、ADE（信息诱导）等机制，支持代理自动化探索并保证环境安全；

**💡 创新点**

创新点在于将数据管理从被动数据库转向支持代理主动探索的环境，并通过分支（Branching）与数据流控制（DFC）两大安全保障，形成可扩展的“自我改进”循环；

**🔧 技术方法**

采用大语言模型驱动的自适应schema生成、向量检索、任务特定SQL技能、分支DBMS（如Neon、Dolt）、检查点式OS级分支以及基于推理引擎的DFC推理；

**📊 数据集**

使用LakeQA（9.5TB数据湖，约4千万条文档）和BranchBench等基准集，并结合公开数据集（Wikipedia、Data.gov、TPC‑H）进行评测；

**📈 对比分析**

与现有RAG、Mem0、Octen、GAM等系统比较，AIM在准确率上提升约50%且速度提升4.18×；在LakeQA上，所有前沿模型准确率≤23%，突显检索+推理双重难点；

**⚠️ 局限性**

局限性包括：生成的schema/管道需持续迁移与评估，数据湖检索受语义压缩平衡影响，分支DBMS缺乏高频快照支持，DFC原型化阶段仍需更细粒度跨系统流控制。

---

## 363. Gap-Majority Lemmas in Communication Complexity

**arXiv ID:** 2607.07396 | [PDF](https://arxiv.org/pdf/2607.07396v1)

**作者:** Pachara Sawettamalya `[一作]` (Princeton University), Huacheng Yu `[通讯]` (Princeton University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在两人随机通信模型中证明了gap-majority函数的最优信息复杂度直接和定理，并将该结果用于给出Gap‑Hamming问题的下界和图中三角计数的流式下界。

**💡 创新点**

首次将gap-majority作为外部gadget，获得信息成本与输入规模线性扩展且错误率与信息量的最优常数折衷；同时通过递归协议分解实现了新的直接和与方差分解技术。

**🔧 技术方法**

使用信息复杂度框架、矩阵分解、协议递归分解、方差与信息成本的线性分解以及对随机公共/私有随机性的精细处理。

**📊 数据集**

无实验数据集，完全理论性分析。

**📈 对比分析**

与已有的identity和XOR gadget直接和结果相比，gap‑majority同样实现了线性标度，并在Gap‑Hamming和三角计数流问题上得到与已知最佳下界相匹配的结果。

**⚠️ 局限性**

局限于两人模型并要求±0.01√n的阈值；递归分解中产生的O(log n/Δ)信息成本附加项；未考虑多轮或多方扩展。

---

## 364. GIFT: Geometry-Informed Low-precision Gradient Communication for LLM Pretraining

**arXiv ID:** 2607.07494 | [PDF](https://arxiv.org/pdf/2607.07494v1)

**作者:** Jieying Wang `[一作]` (Rutgers University), Zhao Zhang `[通讯]` (Rutgers University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在大语言模型预训练中使用FP8低精度梯度通信，并提出了一种几何感知的梯度变换方法，使梯度在通信前投影到近似等方差空间，从而降低量化误差。

**💡 创新点**

创新点在于仅改变梯度通信时的坐标系：利用K‑FAC近似的输入侧低秩变换，将梯度映射到更均匀的空间进行FP8量化，并通过选择性地对最易受量化影响的层应用该变换，实现了低计算开销与高通信精度的平衡。

**🔧 技术方法**

使用技术包括：K‑FAC block‑diagonal Fisher 信息矩阵近似、输入侧低秩（rank‑32）矩阵变换、FP8 量化与 AllReduce 通信、误差反馈机制以及在 Megatron‑LM 框架中的实现。

**📊 数据集**

使用 OpenWebText 数据集，对 Llama‑300M 与 Llama‑600M 两个模型进行预训练。

**📈 对比分析**

对比方法包括 FP32、欧氏空间 FP8（层级标度）以及完整 K‑FAC 方案。结果显示：在 64 个 NVIDIA GH200 超芯片上，所提方法将预训练时间降低 7.6%，梯度通信量下降 75%；在 14 个下游任务中，所提方法在 7 个任务上优于 FP32，优于欧氏 FP8（4/14）。

**⚠️ 局限性**

局限性：实验仅在中等规模模型、固定预训练配方下验证；未探究更大模型、不同训练周期、更多种子或更广泛基准；仅关注梯度通信低精度，未与完整低精度预训练栈结合。

---

## 365. Discovering Geometric Biases in 3D Face Reconstruction: A Curvature-Aware Spectral Framework for Fairness Evaluation

**arXiv ID:** 2607.07486 | [PDF](https://arxiv.org/pdf/2607.07486v1)

**作者:** Veronika Shilova `[一作]` (Artefact Research Center), Jean-Michel Loubes `[通讯]` (Institut de Mathématique de Toulouse)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于曲率的3D人脸重建误差分析框架，用曲率误差图和谱聚类方法发现、量化并可视化重建模型的几何偏差。

**💡 创新点**

创新点在于：1）利用拉普拉斯-贝尔特拉米算子生成高分辨率曲率误差图，捕捉局部细节；2）提出曲率重建误差（CRE）并证明其与人类感知更相关；3）将误差投影到谱空间并通过K-means聚类，系统性识别年龄、性别、民族等维度的偏差。

**🔧 技术方法**

技术手段包括：局部二次回归估计曲率、拉普拉斯-贝尔特拉米算子、曲率重建误差计算、谱分解（本征模式投影）、K-means聚类、用户研究对比评估。

**📊 数据集**

使用 REALY 基准（100个高质量对齐面部扫描，带年龄、性别、民族标签）以及 Basel Face Model 与 FLAME 3DMM 基础；对 Deep3D、3DDFA‑v3、MGCNet、DECA、MICA 等从 2D 图像重建算法进行评估。

**📈 对比分析**

与传统欧氏 RMSE/NMSE（包括 REALY 双向协议）对比；CRE 在用户对比实验中达到 73.6% 的准确率（AUC 82.66%），显著高于随机 50% 的基准，表明 CRE 与人类感知更吻合，并揭示了显著的年龄、性别、民族相关偏差。

**⚠️ 局限性**

局限性包括：REALLY 数据集在民族多样性不足，导致对少数族群的偏差分析受限；LBO 曲率估计受邻域大小和噪声影响；框架主要针对线性 3DMM，未覆盖非线性表达模型；缺乏对更大规模或更复杂模型的实验验证。

---

## 366. Social-spatial dependencies for learning visual navigation

**arXiv ID:** 2607.07460 | [PDF](https://arxiv.org/pdf/2607.07460v1)

**作者:** Patrick Govoni `[一作]` (Humboldt University of Berlin), Pawel Romanczuk `[通讯]` (Humboldt University of Berlin)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

训练了基于视觉感知的神经网络代理，在一个由四面墙和固定补给点组成的方形环境中导航到隐藏补给点，并通过在训练阶段改变未训练代理的数量、技能水平（直接导航vs随机漫步）来研究代理的社会依赖与空间依赖。

**💡 创新点**

提出了混合导航策略：代理在边缘主要使用个体空间信息导航，而在靠近补给点时借助社交信息完成任务，证明只需1–2个具备直接导航技能的伙伴即可显著提升社会依赖，同时揭示了社会与空间导航的空间接口及其在高密度环境中的分离特征。

**🔧 技术方法**

采用卷积神经网络+感知层+线性输出的控制器，利用射线投射（raycast）实现视觉感知；训练过程使用进化策略（ES）而非单体强化学习；通过在不同社交配置下的仿真环境评估代理行为。

**📊 数据集**

完全基于自定义的模拟环境（方形平面、墙壁和可变数量的未训练代理），未使用公开数据集，所有实验数据均来自该仿真平台。

**📈 对比分析**

通过比较非社会（NS）与有补给者的BEt测试环境下的旅行时间差与方向分歧度量来评估社会与空间依赖；结果显示社会依赖随具备直接导航伙伴数量增加而上升，空间依赖在高直接伙伴数时降低；混合策略在高密度环境中表现出更好的任务效率。

**⚠️ 局限性**

局限性包括仅在简化的模拟环境中验证，缺乏真实物理、复杂障碍和多模态感知；未考虑动态环境变化、多代理协同决策以及代理间更细致的交互；因此结果对真实机器人或更复杂场景的可迁移性尚不确定。

---

## 367. GeoGS-SLAM: Geometry-Only Gaussian Splatting for Dense Monocular SLAM

**arXiv ID:** 2607.07452 | [PDF](https://arxiv.org/pdf/2607.07452v1)

**作者:** Lipu Zhou `[一作]` (Beihang University), Kehan Wang `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出 Geometry-only Gaussian Splatting (GeoGS) 方案，仅保留几何相关参数，用于构建 GeoGS-SLAM 系统，实现稠密单目 SLAM。

**💡 创新点**

创新点包括：①仅使用几何参数，80% 参数量减少，提升几何收敛速度；②局部平面驱动初始化加速 Gaussian 适配；③统一 Sim(3) 全局地图更新，避免闭环后地图裂纹；④通过单视几何与多视光度监督，无需颜色渲染，实现纯几何重建。

**🔧 技术方法**

技术手段包括 3D Gaussian Splatting 与可微光栅化、PCA 局部平面估计、单视深度/法线一致性损失、多视重投影光度损失、Pose-only 细化、Sim(3) 轨迹闭环、全局 BA 与一致性地图更新。

**📊 数据集**

使用了 Replica、ScanNet++、ScanNet、DTU 以及 iPhone 捕获子集等公开数据集进行评估。

**📈 对比分析**

与 DROID‑SLAM、NICER‑SLAM、Splat‑SLAM、HI‑SLAM2 等多种 SOTA 方法进行对比，GeoGS‑SLAM 在 Replica 与 ScanNet++ 的 Acc、Comp、Comp.Rat、ATE、CD 指标上均优于或相近，地图几何质量更高；在 DTU 的 2‑分钟优化预算下，Chamfer 距离显著降低，几何收敛更快。

**⚠️ 局限性**

局限性在于仍需要较大 GPU 记忆、依赖深度/法线先验，未针对极大规模或动态场景验证，且统一 Sim(3) 更新虽减少裂纹但仍可能产生短暂不一致。

---

## 368. RLVP: Penalize the Path, Reward the Outcome

**arXiv ID:** 2607.07435 | [PDF](https://arxiv.org/pdf/2607.07435v1)

**作者:** Bojie Li `[一作]` (Pine AI), Noah Shi `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可验证路径通道，结合最终奖励和路径惩罚/进展奖励，实现对真实昂贵交互环境中的学习效率与可部署性提升；

**💡 创新点**

首次将可验证的路径惩罚与最终奖励并行使用，提出四条设计规则防止惩罚陷入无行动，并证明可验证进展潜在奖励在可达时可显著加速学习；

**🔧 技术方法**

采用组内相对强化学习（GRPO）和组内方差分析，利用per‑action rule engine生成惩罚/奖励信号，对齐潜在奖励，并使用 Muon/AdamW 等优化器与大型语言模型；

**📊 数据集**

在可控代理集（系统管理、客服任务）、TerminalBench shell benchmark、miniF2F 定理证明、软件修复等任务上进行实验，并在真实电话代理环境中验证；

**📈 对比分析**

与单一 outcome‑only RLVR 对比，代理任务保持高成功率且违规率降至接近 0；在 TerminalBench 上成功率相同但破坏行为减少 6 倍；在 miniF2F 中对齐潜在奖励将迭代次数从约 7 降至 4，且不出现梯度崩溃；

**⚠️ 局限性**

需人工识别可验证惩罚约束，进展潜在奖励仅在可达时有效；在高成功率模型上的验证不足，且大规模模型方差增大，需要更多真实在线实验来进一步评估。

---

## 369. InductWave: Inductive Multi-Hop Logical Query Answering on Knowledge Graphs

**arXiv ID:** 2607.07422 | [PDF](https://arxiv.org/pdf/2607.07422v1)

**作者:** Mayank Kharbanda `[一作]` (IIIT Delhi), Raghava Mutharaju `[通讯]` (IIT Palakkad)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于波形的小型图嵌入方法，用于在知识图谱上进行逻辑查询回答，特别是在训练图比测试图小的情况下。

**💡 创新点**

创新点在于结合了图波形嵌入和神经贝尔曼-福特网络（NBF-Net），实现了高效的多跳逻辑查询回答，同时减少了消息传递层的数量。

**🔧 技术方法**

使用了图波形嵌入和NBF-Net的结合方法，采用了消息传递算法来进行关系投影。

**📊 数据集**

使用了FB15k-(237)和Wiki-KG数据集进行实验，FB15k-(237)数据集包含237个关系，Wiki-KG数据集包含超过250万个节点。

**📈 对比分析**

与现有的GNN-QE和NodePiece-QE模型进行了比较，结果显示该模型在大多数情况下表现优于基线模型，尤其是在较少的消息传递层数下，且在Wiki-KG数据集上表现最佳。

**⚠️ 局限性**

模型的局限性在于尽管在大多数情况下表现良好，但在某些情况下仍有改进空间，未来的工作可以探索分布式训练和处理其他查询操作的模型。

---

## 370. VCDP: Variation-Conditioned Distributional Proxy Learning for Semi-Supervised Medical Image Segmentation

**arXiv ID:** 2607.07416 | [PDF](https://arxiv.org/pdf/2607.07416v1)

**作者:** Zimu Zhang `[一作]` (Xi'an Jiaotong-Liverpool University), Xiaofeng Liu `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种仅在训练时使用的插件式正则化模块VCDP，用于半监督3D医学图像分割，通过对每个解剖类别构造可学习的高斯分布代理和多种变异原型来组织特征空间。

**💡 创新点**

创新点在于将分布式类代理与多样化的内部变异原型耦合，利用统一的变异条件兼容度评分实现软分配，既捕捉全局语义分布，又显式建模细粒度解剖差异，且在推理阶段无额外开销。

**🔧 技术方法**

采用高斯分布代理与多原型聚类、对抗式样本采样、soft‑max聚合、stop‑gradient技术、嵌入对齐与判别损失，整合到现有半监督框架（CPS、MagicNet等）中。

**📊 数据集**

在多器官CT数据集Synapse（20%标签）和AMOS（5%标签）上进行实验。

**📈 对比分析**

与多种半监督基线（CPS、MagicNet、DHC、GenSSL、SS‑Net、Adsh、DCMamba）在Dice、NSD、HD95指标上对比，VCDP在平均Dice可提升至约9.7%（最优情况），在小、模糊解剖结构上显著改善，HD95也普遍下降。

**⚠️ 局限性**

局限性包括：仍需一定比例的标注数据；性能提升与基线特征空间组织程度相关；主要验证于CT数据，对其他模态或极其模糊边界的鲁棒性尚未充分证明；训练时需额外模块，虽然推理无成本，但增加了训练复杂度。

---

## 371. Positional Determinacy with Colored Vertices: a 1-to-2-Player Lift

**arXiv ID:** 2607.07415 | [PDF](https://arxiv.org/pdf/2607.07415v1)

**作者:** Raphaël Berthon `[一作]` (Université Paris-Saclay), Stéphane Le Roux `[通讯]` (Université Paris-Saclay)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了顶点着色游戏与边着色游戏的定位确定性，并给出了从一玩家到两玩家的提升以及与对偶的对应关系。

**💡 创新点**

首次证明在顶点着色游戏中，前缀无关且定位确定的目标等价于对偶颜色对的广义奇偶目标，并指出颜色集有限性是必要条件。

**🔧 技术方法**

采用了Hub‑cycle 结构、对偶对优先级函数、Muller 对象在对偶对上的闭包性质以及对偶映射构造等技术。

**📊 数据集**

本文无使用实验数据集，而是以形式化证明为主。

**📈 对比分析**

通过形式化等价性证明与已有的边着色游戏结果对比，展示了在顶点着色游戏中定位确定性保持与边着色游戏相同，但在目标表达力上更强。

**⚠️ 局限性**

主要限制在于颜色集合有限性，无法推广到无穷多颜色；且对偶映射导致优先级集合平方增长，可能产生指数级别的计算复杂度。

---

## 372. Transformer-based segmentation of prosodic boundaries in Brazilian Portuguese

**arXiv ID:** 2607.07408 | [PDF](https://arxiv.org/pdf/2607.07408v1)

**作者:** Rodrigo de Freitas Lima `[一作]` (University of São Paulo), Marcos Vinicius Treviso `[通讯]` (University of Lisbon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并训练了基于 Whisper large‑v3 的端到端语音分段模型，将终结语调单元标记直接嵌入转写文本中，完成自动划分；

**💡 创新点**

创新点在于将分段任务视为序列到序列的 ASR 任务，利用 Whisper 的自回归解码器在不改动模型结构的前提下通过添加特殊标记实现分段；

**🔧 技术方法**

采用 Whisper large‑v3 进行 fine‑tune，结合低通/高通滤波、数据增强、n‑gram 统计及声学‑视觉分析评估模型行为；

**📊 数据集**

使用 NURC‑SP Minimal Corpus、CATNA‑MT（合并修正后）以及 MuPe‑Diversidades 这三套手工标注的巴西葡萄牙语语料；

**📈 对比分析**

在 NURC‑SP/CATNA 测试集上得到最佳 binary F1=0.731、macro F1=0.858；在 MuPe‑Diversidades 上最高 binary F1=0.796、macro F1=0.890；不同滤波配置对内测效果影响有限，而高通 600 Hz 训练配置在外测上表现最佳；

**⚠️ 局限性**

模型仍易受嘈杂、停顿、局部 F0 变化等模棱两可信号误判；对语篇层面信息的利用不足，误差模式需进一步系统化研究

---

## 373. Faster Randomized and Deterministic k-Clustering on Graphs

**arXiv ID:** 2607.07615 | [PDF](https://arxiv.org/pdf/2607.07615v1)

**作者:** Sebastian Forster `[一作]` (University of Salzburg), Antonis Skarlatos `[通讯]` (University of Warwick)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在图上实现了三类聚类问题的近线性时间近似算法：(1)确定性递增 k‑center 取得 2+ε 近似；(2)随机化递增 (k,z)-clustering 取得 O(1) 近似；(3)确定性递增 (k,z)-clustering 取得 O(poly(t)) 近似，时间为 Õ(t·m·n^{1/t}+m/ε)。

**💡 创新点**

创新点主要在于：
• 提出了第一套确定性近线性时间的 k‑center 算法，并且可生成递增的中心序列；
• 将源插入（Source‑Insertion）近似最短路数据结构与 Gonzalez 贪心框架相结合；
• 通过 Thorup‑Zwick 距离预处理将随机化的球大小估计替换为确定性估计，实现了确定性 (k,z)-clustering；
• 在算法框架中统一使用递增、常数因子近似的 “球大小估计” 与 “近似球” 两大原语。

**🔧 技术方法**

核心技术：
• 源插入 (1+ε)-SSSP 数据结构，用于在每一步快速获得到当前中心集合的距离估计；
• 受 Thorup‑Zwick 距离预处理启发的 bunch 与 cluster 结构，用于确定性估计球大小和构造近似球；
• 递归贪心框架（Dupré‑la‑Tour & Saulpic）与 Dijkstra 的截断版结合，用于高效构造球集合；
• 采用多层次半径刻度（R = Δ/(2c)^ℓ）保证对所有可能半径的覆盖。

**📊 数据集**

本文未在实验中使用具体数据集，所有结果均为理论分析与复杂度证明。

**📈 对比分析**

与现有工作比较：
• 对 k‑center，打破 Abboud 等人提出的开放问题，获得了 2+ε 近似的确定性近线性算法；
• 对 (k,z)-clustering，随机化结果匹配 Thorup、Jiang 等人的最优近似因子，但在时间上实现了近线性；
• 对确定性 (k,z)-clustering，首次给出近线性（以 t 为调节参数）时间与常数因子近似的方案。整体性能上，算法在大多数参数范围内达到或超过目前已知的最优时间与近似比。

**⚠️ 局限性**

局限性：
• 确定性 (k,z)-clustering 的时间仍为 Õ(t·m·n^{1/t})，当 t 较大时退化为超线性；
• 需要先构建 Thorup‑Zwick 预处理，时间与空间均受 t 影响；
• 结果依赖于图的 aspect ratio Δ，若 Δ 极大则会增加 logΔ 项；
• 对 k‑center 的 2+ε 近似仍未能突破 2 的下界（已知不可超越），且在有 t 异常点的情况下需要额外的 ε 与 t 参数调优。

---

## 374. User identity conditions moral wrongness ratings in non-reasoning large language models

**arXiv ID:** 2607.07605 | [PDF](https://arxiv.org/pdf/2607.07605v1)

**作者:** Willem Fourie `[一作]` (Stellenbosch University), Gray Manicom `[通讯]` (Stellenbosch University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 GPT‑4.1‑mini‑2025‑04‑14 与 Gemini‑2.5‑flash‑lite 进行 12,000 轮对话实验，探究用户职业身份在未显式设定道德立场的情况下，如何影响 LLM 对 Gert 公共道德框架十条规则的错误感知评分。

**💡 创新点**

首次证明用户身份在非推理模型中能显著调节道德评估，揭示角色导致可观的道德分歧，提示价值对齐需考虑动态可接受的道德范围而非固定规则。

**🔧 技术方法**

采用多轮对话结构、Gert 公共道德规则、一次性单项评分、自动/人工评分回收、中心化偏差分析、η² 方差分解与 Benjamini–Hochberg 校正等统计技术。

**📊 数据集**

使用 20 个职业角色（10 传统、10 非传统）与 10 条道德行为（共 10×20=200 条组合），每组 30 次交互，分别在两款模型上完成 6,000 轮对话，总计 12,000 条评分数据。

**📈 对比分析**

通过角色中心化偏差热图、η² 解释比例与 BH 调整后 p 值比较，结果显示在大多数行为中角色差异显著（η²≥0.312），且在两款模型间一致性高，证明角色对评分影响显著且可量化。

**⚠️ 局限性**

仅涵盖两款模型与 20 个角色，缺乏中性基线，未覆盖更广泛的文化与模型多样性，且实验使用单一公共道德框架，限制了结论的普适性与推广性。

---

## 375. NARAD: Non-colluding Aggregator-oblivious Record-And-Decrypt

**arXiv ID:** 2607.07596 | [PDF](https://arxiv.org/pdf/2607.07596v1)

**作者:** Akshit Vakati Venkata `[一作]` (Indian Institute of Technology Madras), Ayush Adarsh `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在本文中，作者实现并验证了一套基于聚合者不可知 Paillier 同态加密的区块链投票系统，利用位分组技术将多候选人投票压缩为单一加密值，并通过 Solana 区块链实现不可篡改的投票记录；系统在浏览器侧使用 WebAssembly 进行加密，后端使用原生 C（libtommath）完成聚合，最终能够在 50,000 票投下不到 1 s 的时间完成统计。

**💡 创新点**

核心创新点包括：① 聚合者不可知的加密模型，消除了对可信密钥分发器的依赖；② 位分组压缩方案，将 k 候选人投票压缩为单一 ciphertext，从而显著降低客户端、链上、存储和聚合成本；③ 将投票记录写入 Solana 提供完整审计轨迹；④ 将浏览器加密与后端聚合拆分至 WebAssembly 与原生 C，提升性能；⑤ 在安全可验证性层面提出通过安全可信执行环境实现投票计数防篡改。

**🔧 技术方法**

所用技术包括：Paillier 同态加密（DCR 与 Diffie‑Hellman 掩码假设）、WebAssembly、C 语言 + libtommath、Node.js + Express、PostgreSQL、Solana Anchor、Docker Compose、React/Tailwind 前端。

**📊 数据集**

实验使用人工合成的投票数据：十位候选人、每候选人 25 位槽位，测试集包含 100 到 50,000 名选民的投票，均为 0/1 选票。

**📈 对比分析**

与传统每候选人单独加密方案相比，位分组使得客户端加密次数、链上交易数量、存储占用和聚合乘法次数全部缩减至 1/k，实验结果显示聚合阶段随投票者数量线性扩展：在 50,000 票时，聚合耗时 0.815 s，吞吐率约 61 000 票/秒，显著优于 k 倍级别的原始方案。

**⚠️ 局限性**

主要局限包括：缺乏零知识范围证明导致投票合法性无法强制；不具备强制投票抵抗与收票者证明（coercion resistance）；依赖可信中心生成 N 的分解；单一聚合者可能拒绝公布结果；若收集器与聚合器 collude 将泄露投票；未实现门限聚合或分布式密钥生成。

---

## 376. Context-Aware Force Estimation for Deformable Tool Manipulation in Robotic Environmental Swabbing via Few-Shot Continual Adaptation

**arXiv ID:** 2607.07574 | [PDF](https://arxiv.org/pdf/2607.07574v1)

**作者:** Siavash Mahmoudi `[一作]` (University of Arkansas), Dongyi Wang `[通讯]` (University of Arkansas)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了如何在可变形工具操控（DTM）中，仅依靠关节状态和腕部力矩计估计接触力，无需工具端集成传感器。

**💡 创新点**

引入了参数隔离的少样本适配机制，使用FiLM对冻结的LSTM骨干进行上下文调制，实现了跨表面和工具合规性的快速适应并避免灾难性遗忘。

**🔧 技术方法**

采用时序模型LSTM为主干，并结合FiLM上下文注入、少样本持续学习、零样本评估与微调技术。

**📊 数据集**

在UR5e机器人上使用三种不同弹性级别的刷头与九种表面（木材、不锈钢、弹性复合材料）的同步关节/力矩计和嵌入式FSR标注数据。

**📈 对比分析**

对比了MLP、CNN、TCN、Transformer和LSTM，LSTM在RMSE 26.99 ADC（约0.23 N）与R² 0.992、0.37 ms延迟下最优；在九种交互场景下，零样本误差提升最高可达200%，而少样本适配后误差下降18–63%，保持原始域无遗忘。

**⚠️ 局限性**

对实时控制的频率要求高且仅在预设轨迹下验证，且适配仍需在每个新域采集5次示例，未解决自动化上下文推断与更复杂非平面表面的问题。

---

## 377. Towards Agentic AI Governance: A Preliminary Assessment

**arXiv ID:** 2607.07612 | [PDF](https://arxiv.org/pdf/2607.07612v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 378. Asymmetric Focal Loss Improves Graph Neural Network Prediction of Drug-Drug Interactions

**arXiv ID:** 2607.07611 | [PDF](https://arxiv.org/pdf/2607.07611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 379. Multi-Class vs. Multi-Label BERT for CVE-to-CWE Mapping: How Taxonomy Structure Shapes the Errors

**arXiv ID:** 2607.07573 | [PDF](https://arxiv.org/pdf/2607.07573v1)

**作者:** Ana Schwengber Kelm `[一作]` (mindsquare AG), Jörg Frochte `[通讯]` (Bochum University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了将 CVE 描述映射到 CWE 代码的分类任务，比较了多类与多标签两种输出形式。

**💡 创新点**

通过对比两种建模方式并引入层次松弛评价，首次揭示分类误差主要由 CWE 体系结构驱动，而非模型差异。

**🔧 技术方法**

采用了三种 Transformer 编码器（BERT Base、SecureBERT、CySecBERT）并进行全微调。

**📊 数据集**

使用 MITRE CVE 数据集，筛选出频率≥100 的 83/47/25 类标签。

**📈 对比分析**

在三种标签空间下评估宏 F1，结果显示多类始终优于多标签，差距可通过阈值调优缩小，层次松弛后宏 F1 接近 90%，CySecBERT 取得最佳性能。

**⚠️ 局限性**

限制包括训练数据对多标签标注稀疏、标签空间受阈值采样影响、层次松弛使用人工定义的家族分组，以及未考虑长尾稀有类别。

---

## 380. Think Big, Search Small: Where Capacity Matters in Hierarchical Search Agents?

**arXiv ID:** 2607.07548 | [PDF](https://arxiv.org/pdf/2607.07548v1)

**作者:** Qinnan Cai `[一作]` (East China Normal University), Xiang Li `[通讯]` (East China Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多代理搜索代理的模型容量分配，提出将搜索任务拆分为委派、执行和答案生成三角色，并在多代理架构中系统地对委派和执行模型容量进行实验。

**💡 创新点**

创新点在于①验证角色分解本身即可提升性能；②发现委派（问题分解）是瓶颈，执行对容量敏感度低；③通过质量过滤的轨迹蒸馏训练紧凑 1.7B 执行器，既能匹配前沿模型，又显著降低代价。

**🔧 技术方法**

采用多代理角色分解架构、模型容量分配实验、质量过滤轨迹蒸馏（SFT）训练、LLM 判定评估以及基于检索的执行器。

**📊 数据集**

使用 2WikiMultihopQA、HotpotQA、MuSiQue、PopQA、Bamboogle 这五个多跳 QA 数据集（共 3,869 个有效实例）。

**📈 对比分析**

通过与单代理基线的 EM、F1、LLM-judge 对比，委派容量从 1.7B 提升至前沿可提升 EM 约 11 点；执行容量从 1.7B 提升至前沿仅提升约 2.6 点；紧凑 1.7B SFT 执行器在同一委派下实现或超过前沿性能，并将子代理 token 消耗降低 37%。

**⚠️ 局限性**

局限性包括仅在英文多跳 QA 与固定检索语料下验证，未探讨开放式网络检索、跨语言场景、RL 优化执行器等更广泛的应用。

---

## 381. CARLA-GS: Decoupling Representation, Reasoning, and Physics Simulation for Autonomous Driving Corner-Case Synthesis

**arXiv ID:** 2607.07601 | [PDF](https://arxiv.org/pdf/2607.07601v1)

**作者:** Kaicong Huang `[一作]` (Rensselaer Polytechnic Institute), Ruimin Ke `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套模块化的安全关键角落案例生成管线，将 3D 高斯散点渲染、LLM 语义推理与 CARLA 物理仿真结合，实现从真实数据到可执行、可渲染的驾驶场景。

**💡 创新点**

创新点在于将视觉、语义与动力学三大模块解耦并保持跨模块一致；使用多代理 LLM 进行碰撞区风险分析与意图级轨迹规划，并通过 CARLA 的 PID 控制保证轨迹物理可行性；同时引入几何一致性约束和 3D 基础模型替换提升渲染质量。

**🔧 技术方法**

使用技术包括 3D Gaussian Splatting、几何一致性正则化、LLM 多代理推理（Zone 与 Trajectory 代理）、CARLA PID 跟踪、SAM‑3D 重建替换、以及可视化的后向投影。

**📊 数据集**

主要在 Waymo Open Dataset 上进行实验，采集多序列驾驶数据并生成 85 个角落案例。

**📈 对比分析**

与基于规则和随机轨迹的基线相比，本方法在 Zone Hit（0.438）、Success（0.925）和最小 TTC（0.472 s）等指标上表现更好；在物理可行性和乘坐舒适度上亦优于纯 LLM 方案。

**⚠️ 局限性**

主要局限包括 LLM 推理的不确定性导致 29% 生成结果无效；3DGS 训练是计算瓶颈，实时性受限；以及在极少视角下几何一致性仍需改进。

---

## 382. PHaul: A PPO-based forwarding agent for Sub6 enhanced Integrated Access and Backhaul networks

**arXiv ID:** 2607.07584 | [PDF](https://arxiv.org/pdf/2607.07584v1)

**作者:** Jorge Pueyo `[一作]` (I2CAT Foundation), Miguel Catalan-Cid `[通讯]` (I2CAT Foundation)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究 Sub6 加强的 3GPP IAB 网络，提出 PHaul 路由决策器，实现基于实时流量矩阵的路径分配。

**💡 创新点**

创新点在于将离线路径搜索与在线 PPO 强化学习相结合，利用网络数字孪生实时评估并更新路径，支持多种流量工程目标。

**🔧 技术方法**

采用 Proximal Policy Optimization（PPO）深度强化学习、ShortestPath/LastHop/LeastCommon 路径启发式、网络数字孪生模型、Python+Gym 以及 Stable-Baselines3 等技术。

**📊 数据集**

使用随机生成的 IAB 拓扑与流量矩阵（模拟 Chicago 子网）作为实验数据，并将实现与仿真环境开源公开。

**📈 对比分析**

通过与暴力、子集求和、随机三种基线对比，PHaul 在效率和公平度上分别提升约 36% 与 20%，且推理时间保持在 10 秒以内。

**⚠️ 局限性**

局限性包括需要离线训练、对极大规模拓扑的泛化能力有限、以及实验基于理想化链路容量与无干扰假设，实际部署时需进一步验证。

---

## 383. Learning to Unify Deformable Shape and Texture Representations for Cardiac Video Classification

**arXiv ID:** 2607.07518 | [PDF](https://arxiv.org/pdf/2607.07518v1)

**作者:** Tonmoy Hossain `[一作]` (University of Virginia), Miaomiao Zhang `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种能够在共享潜在空间中融合可变形形状与图像纹理特征的心脏视频分类框架 ShapeFuse。

**💡 创新点**

创新点在于：①使用双向跨模态时序注意力学习形状与纹理在心动周期中的相互依赖；②引入自适应门控动态调节每个时间点两种模态的权重；③结合诊断重要性注意力聚合，提升对关键心动相的关注。

**🔧 技术方法**

主要技术包括：可微分的形变网络（基于 SVF 的差分变形）、跨模态注意力机制（双向多头自注意力）、自适应门控与诊断重要性聚合（Bahdanau 注意力）、卷积/Transformer 图像编码器、全连接分类器及交叉熵/正则化损失。

**📊 数据集**

使用了 510 条 24 帧的 224×224 赛璐卡式 CMR 视频数据集，包含 125 位受试者，约 40% 受试者存在梗塞性壁运动异常。

**📈 对比分析**

与传统的拼接、加法、加权、双线性及注意力融合方法进行对比。ShapeFuse 在所有图像编码器（ResNet、EfficientNet、DenseNet、ViT）和形变网络（VoxelMorph、TLRN）上均表现出显著提升，最高微平均准确率提升至约 0.899，F1 分数提升至 0.901，远优于最优传统融合策略。

**⚠️ 局限性**

局限性包括：①仍需大量标注数据进行训练；②形变网络的计算开销较大，推理时效性受限；③在多标签或多病理共存场景下尚未验证，需要进一步扩展。

---

## 384. Automatic Echocardiography Segmentation via Transition Probability Correlation for Stable Semantic Extraction

**arXiv ID:** 2607.07580 | [PDF](https://arxiv.org/pdf/2607.07580v1)

**作者:** Xinran Chen `[一作]` (Southeast University), Chuan Chen `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种基于局部转移概率相关性的心脏超声视频分割框架（STLSF模块）与频域去噪预训练策略，实现了语义-纹理双向融合和时空一致性提升。

**💡 创新点**

创新点在于：1) 将像素级转移概率视为时空运动分布，用于语义纠正；2) 语义引导纹理模块在局部注意力框架下恢复边界；3) 频域噪声预训练将超声特征嵌入卷积编码器；4) 采用overview-focus三阶段特征提取，兼顾局部细节与全局语义。

**🔧 技术方法**

使用的技术包括：卷积网络的基础块、动态块、跨注意力机制、局部转移概率生成与注意力矩阵、频域去噪损失、半监督损失（BCE+Dice）以及AdamW与余弦学习率调度。

**📊 数据集**

使用的公开数据集为CAMUS（训练/验证/测试 7:1:2）和EchoNet-Dynamic（官方划分），视频长度10帧，标注仅包含ED和ES帧。

**📈 对比分析**

与八类基线模型（XMem++、VideoMamba、H2Former、EchoONE、PKEcho-Net、MemSAM、SimLVSeg、NCMNet）进行对比，性能上在CAMUS上Dice 93.87%、HD95 3.29mm；EchoNet-Dynamic上Dice 92.62%、HD95 2.73mm，均领先于现有方法。

**⚠️ 局限性**

局限性包括：1) 对低帧率或极端噪声场景的适应性尚待验证；2) 预训练仅基于频域噪声，未考虑其他超声成像变形；3) 对中间帧无监督约束的有效性依赖于时序连贯性假设，可能在心脏形变剧烈时失效。

---

## 385. Collaborative Synthetic Data Generation for Knowledge Transfer in Federated Learning

**arXiv ID:** 2607.07565 | [PDF](https://arxiv.org/pdf/2607.07565v1)

**作者:** Maximilian Andreas Hoefler `[一作]` (Fraunhofer Heinrich Hertz Institute), Wojciech Samek `[通讯]` (Fraunhofer Heinrich Hertz Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 FedKT-CSD，一种一-shot 联邦学习框架，利用冻结的预训练自动编码器在客户端仅做一次前向传播，计算加性统计后通过安全聚合与 DP 噪声生成可供训练的合成数据。

**💡 创新点**

创新点在于：①将预训练的 autoencoder 作为共享低维潜在空间，②仅传输加性统计实现单轮通信与轻量上传，③通过安全聚合与高斯 DP 机制提供正式 (ε,δ)-DP，④生成的合成数据既适用于集中模型训练，也可用于后续个性化 FL，解决异构数据下的鲁棒性与隐私兼顾。

**🔧 技术方法**

使用技术包括：冻结的预训练 autoencoder（Encoder/Decoder）、按类别的加性统计（均值与二阶矩）计算、Secure Aggregation、Gaussian DP 加噪、偏差校正与正定投影、在潜在空间中采样并解码合成图像、随后进行标准 ERM 训练或预训练。

**📊 数据集**

实验数据集包括 ImageNette（10 类）、CIFAR-100（100 类）、EuroSAT（10 类卫星图像）和 BloodMNIST（8 类显微镜图像）。

**📈 对比分析**

通过与多种无 DP OSFL 基线（FedSD2C、DENSE、CoBoosting、FedD3、FedCVAE）以及集中/联邦 DP 合成方法（DP-MERF、DP-NTK、DP-Kernel、DP-LoRA LDM）对比，FedKT-CSD 在所有数据集与异构设置下均超过无 DP 基线，并在 DP 场景下与集中 DP 方法持平或优于其表现；在个性化 FL 任务中，单轮预训练+局部微调的效果也优于多轮 pFL 方法。

**⚠️ 局限性**

局限性包括：仅适用于已标记的分类任务，需客户端知晓类别；单高斯模型对类分布重叠或噪声标签敏感；少样本类别统计噪声大；无法直接处理无标签或自监督情形，且扩展到多模态或更复杂分布时可能受限。

---

## 386. Holistic B2X Mobile Application Development -- A Reference Model

**arXiv ID:** 2607.07511 | [PDF](https://arxiv.org/pdf/2607.07511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 387. Face-trace: Open-Set Attribution and Progressive Discovery of Synthetic Face Generators

**arXiv ID:** 2607.07545 | [PDF](https://arxiv.org/pdf/2607.07545v1)

**作者:** Alessia Infantino `[一作]` (Sapienza University of Rome), Irene Amerini `[通讯]` (Sapienza University of Rome)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一套面向合成人脸的开集源归因流水线，包括已知生成器分类、基于能量的异常检测与被拒样本聚类，并支持增量发现未知生成器。

**💡 创新点**

创新点在于：①将冻结的 I-JEPA 表征与 Forensic Self-Descriptor 融合以提升未知生成器的可聚类性；②采用能量阈值实现高效 OOD 拒绝；③设计增量发现机制，将被拒样本按已发现未知簇匹配或缓冲，待缓冲满后使用 HDBSCAN 自动发现并推广新未知生成器。

**🔧 技术方法**

使用技术包括：I‑JEPA 冻结视觉编码器、线性/两层 MLP 分类器、能量（Energy）阈值 OOD 检测、Forensic Self‑Descriptors、UMAP 降维、HDBSCAN 聚类、Mahalanobis 距离匹配、缓冲区与阈值策略。

**📊 数据集**

实验主要基于 WILD 数据集（20 组生成器，10 组已知，10 组未知），并在 SFHQ‑T2I、SoFake、AI‑Face 等外部数据集做跨数据集验证。

**📈 对比分析**

与基线（CLIP、DINOv3 分类器；softmax、generalized entropy、energy 拒绝；SimGCD、ProtoGCD、OWDFA‑CAL、OCD 发现方法）相比，本文在 WILD 上实现：闭集识别 96.73% 准确率；开集能量拒绝 71.25% 平衡准确率；未知生成器聚类 ARI 0.81、NMI 0.90、纯度 87.74%；增量发现最终纯度 99.23%。

**⚠️ 局限性**

局限性包括：①后处理（压缩、裁剪、旋转等）显著削弱聚类质量；②增量发现对缓冲区大小、样本顺序和推广阈值敏感；③跨数据集时对未知生成器匹配仍有误差，且在完全陌生生成器场景下表现不如已知/已发现的情况。

---

## 388. Infinite Worlds with Versatile Interactions

**arXiv ID:** 2607.07534 | [PDF](https://arxiv.org/pdf/2607.07534v1)

**作者:** Zelin Gao `[一作]`, Hao Ouyang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款开放源代码的因果视频生成模型，能够在 720p 60fps 的实时条件下无限延伸交互世界，并支持多种动作与环境变化。

**💡 创新点**

创新点在于结合因果预训练与多步骤蒸馏实现长期无漂移的高质量生成，同时引入 Pilot‑Director 双子机架构进行因果推理，并实现多玩家同步沉浸。

**🔧 技术方法**

采用因果扩散 Transformer（DiT）、MoBA 注意力掩码、Consistency Distillation + Distribution Matching Distillation、动态 KV 缓存、视觉语言模型（VLM）驱动 Director、SAM 追踪、以及多 GPU 编译优化等技术。

**📊 数据集**

数据集来自三源：自摄像头 egocentric 视频、游戏与 Unreal Engine 合成视频、以及大规模网络视频，统一元数据后进行技术与 VLM 过滤，最终生成分块注释。

**📈 对比分析**

与 HappyOyster、Genie 3 等闭源系统以及主流开源模型对比，模型在 720p 60fps 的实时推理下保持视觉质量不衰退，支持超过十种动作和即时天气变换，一小时无漂移，性能匹配或优于闭源基线。

**⚠️ 局限性**

主要局限包括：缺乏真正的长期记忆导致重新生成区域；身份与风格随时间轻微漂移；物理理解不足偶尔出现穿模；算力需求仍高，仍需进一步优化以实现消费级硬件的实时高质量运行。

---

## 389. Gradient-free Riemannian Langevin Sampler

**arXiv ID:** 2607.07519 | [PDF](https://arxiv.org/pdf/2607.07519v1)

**作者:** Ricardo Baptista `[一作]` (University of Toronto), Olivier Zahm `[通讯]` (UGA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种不需要梯度信息的黎曼兰格宁采样器GRiLS，并给出了块集群版本BE-GRiLS，用来高效地对多模态概率分布进行采样。

**💡 创新点**

创新点在于：① 用度量 W(x)=ν(x)/μ(x)Σ 构造梯度自由的黎曼度量，使局部几何被重塑以降低跨模态的距离；② 通过 Lamperti 变换与时间离散相结合，得到无梯度的 MCMC 提案；③ 引入块集群更新策略（BE‑GRiLS）提升并行效率。

**🔧 技术方法**

使用技术包括：Riemannian Langevin 动力学、Lamperti 变换、欧拉‑马尔可夫时间积分、Metropolis 校正、共识采样/集合方法、谱间隙与 IACT 分析，以及 Ulam 逼近方法。

**📊 数据集**

实验数据集：一维高斯混合和光滑分段常数密度；二维两月、两环以及三峰高斯混合；十维带三峰的高斯混合。

**📈 对比分析**

通过谱间隙、接受率、Ulam 矩阵、IACT 以及均值/协方差误差等指标与 IS、pCN、MALA、Adaptive Metropolis、AIES 等方法比较，GRiLS/BE‑GRiLS 在多模态场景下展现更快混合、全覆盖且误差更低，尤其在高维问题中优于传统 MCMC。

**⚠️ 局限性**

局限性：接受率随维度增加呈指数衰减；对高维问题的鲁棒性尚需进一步改进；步长参数缺乏自适应选择；目前仅在高斯参考分布下可用，非高斯推广仍待研究。

---

## 390. Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning

**arXiv ID:** 2607.07508 | [PDF](https://arxiv.org/pdf/2607.07508v1)

**作者:** Zhenyu Hou `[一作]` (Tsinghua University), Yuxiao Dong `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种单一次性采样的异步强化学习框架（SAO），旨在解决大语言模型（LLM）在异步训练中的离线策略漂移和训练不稳定问题；

**💡 创新点**

创新点包括：①引入双侧 token‑级重要性采样（DIS）与严格的双侧裁剪/掩码，直接利用 rollout 记录的 log‑prob 进行比例校正；②将 GRPO 的群组采样替换为单一次性采样，从而消除生成时的同步阻塞；③设计了更快的价值网络更新（K > 1）、冻结注意力层的训练策略以及跨观测跳过的 GAE，显著提升价值估计质量与梯度稳定性；

**🔧 技术方法**

采用了异步 PPO 风格的目标函数、token‑级重要性采样、双侧裁剪、长度自适应 GAE、快速价值网络更新、冻结注意力训练以及 skip‑observation GAE 等技术；

**📊 数据集**

使用了数学推理数据集 AIME2025、BeyondAIME、HMMT Nov 2025、IMOAnswerBench，编码任务使用 SWE‑Bench Verified 与 OpenHands，工具集成推理训练使用 TIR 数据；在线学习模拟中还采用 GLM‑4.7 作为判定器；

**📈 对比分析**

与传统 GRPO 及其 DIS 变体在 Pass@1（数学推理）和 Accuracy（SWE‑Bench）上进行对比；SAO 在 AIME2025、BeyondAIME、IMOAnswerBench、SWE‑Bench Verified 等指标上均优于 GRPO，最高可达 97.3% 以及 29.8% 等，且训练稳定性提升至约 1000 步；在在线学习模拟中，SAO 能在奖励偏好切换后迅速重新校准，适应速度明显快于基于滑动窗口的优势估计；

**⚠️ 局限性**

局限性包括：仅在单一次性采样场景下有效，需依赖大规模价值预训练；训练步骤受限于约 1000 步，可能无法覆盖更长序列；评测仅覆盖有限的推理与编码基准，未验证对其他任务或 RLHF 流程的普适性；

---

## 391. From Custom-Fit to Portable: Bridging the Gap Between Synthesized and Engineered GPU Query Execution

**arXiv ID:** 2607.07632 | [PDF](https://arxiv.org/pdf/2607.07632v1)

**作者:** Ivan Donchev Kabadzhov `[一作]` (EURECOM), Raja Appuswamy `[通讯]` (EURECOM)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 SHADB 框架利用大型语言模型（LLM）在 GPU 上合成专用查询执行代码，并通过闭环的性能引导优化（PGO）使其逼近内存带宽上限；随后对合成代码进行根因分析，识别可迁移到通用引擎的优化，并将这些优化集成到基于 SYCL 的可移植数据库引擎 SYCLDB 中，最终实现仅差 1.27× 的性能差距。

**💡 创新点**

创新点在于：①首次将 LLM 与 PGO 循环结合用于 GPU 查询合成；②系统化地将合成代码的优化迁移到可移植引擎，实现性能与可移植性的折中；③利用 AdaptiveCpp 的动态函数实现无代码生成的 GPU 内核融合。

**🔧 技术方法**

技术包括：LLM（Anthropic Claude Opus 4.8）生成 CUDA/HIP 源代码；静态审核、编译、硬件计数器分析、诊断反馈的闭环优化；SYCL 编程模型与 AdaptiveCpp 的动态函数实现内核融合；直接映射哈希表、字节压缩维度表、共享内存私有聚合等 GPU 优化技术。

**📊 数据集**

使用 Star Schema Benchmark（SSB）在规模因子 100（约 6 亿行）和 200（约 12 亿行）进行评估；硬件平台为 NVIDIA L40S（Ada）和 AMD MI210（CDNA2）。

**📈 对比分析**

比较方法：在相同工作负载下对 SHADB、SYCLDB（基线与优化版）、HeavyDB、Crystal 等系统进行 warm‑run 执行时间测量；通过 ncu/rocm‑prof 进行硬件计数器对比。结果显示：SHADB-Opt 在 SSB SF100 上相较 HeavyDB 和 SYCLDB 基线分别快 7.4× 与 8×，但与优化后的 SYCLDB-Opt 仅差 1.27×；在 SF200 上相对差距同样仅为 1.2×。

**⚠️ 局限性**

局限性：①合成代码需要高昂的 LLM 调用成本（约 4.2 GPU‑小时、$126）且不具备跨规模或跨 GPU 体系结构的可移植性；②PGO 循环需要多轮编译与调试，耗时长；③将合成优化迁移到通用引擎虽可缩小性能差距，但仍需在软件工程上承担额外维护与分支管理成本。

---

## 392. Pure Nash Equilibria in Graphical Games of Bounded Width Revisited

**arXiv ID:** 2607.07627 | [PDF](https://arxiv.org/pdf/2607.07627v1)

**作者:** Michael Lampis `[一作]` (Université Paris Dauphine Psl), Yiren Lu `[通讯]` (Université Paris Dauphine Psl)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析并修正了先前关于图形游戏纯纳什均衡决策的算法错误，并提出更优的参数化算法与对应的下界。

**💡 创新点**

证明Thomas和van Leeuwen的算法存在严重缺陷；给出针对树宽、路径宽、割宽的更优算法，利用对G及其平方图宽度关系的紧化；以及相应的下界证明。

**🔧 技术方法**

参数化复杂度、W[1]-hardness证明、CSP归约、图论宽度关系紧化、组合数学。

**📊 数据集**

无实证数据集，全部为理论分析与构造性证明。

**📈 对比分析**

通过比较指数依赖，将先前的α^(Δ+1)降至α^(tw+2⌊2Δ/3⌋)等，显著降低了指数；下界表明若进一步改进则违背SETH。

**⚠️ 局限性**

对树宽仍未给出最优下界；算法仅在已知分解的情况下有效；研究集中于无向图和有向图的某些情形，未覆盖所有游戏模型。

---

## 393. AA-ViT: Anatomically Aware Vision Transformer with Structural and Frequency Guidance for Contrast Enhanced Brain MRI Synthesis

**arXiv ID:** 2607.07553 | [PDF](https://arxiv.org/pdf/2607.07553v1)

**作者:** Talha Meraj `[一作]` (Atlantic Technological University), Saritha Unnikrishnan `[通讯]` (Atlantic Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 AA‑ViT 模型，用多模态预对比 MRI（T1、T2、FLAIR）合成对比增强 T1ce MRI。

**💡 创新点**

创新点包括：残差密集边缘块 (RDEB) 提升解剖边界感知；自误差聚焦、结构与频域损失组合，显著减少伪影和边缘失真；并通过 CKA 分析验证生成图像与真实图像的解剖特征一致性。

**🔧 技术方法**

技术手段：基于 ResViT 的 Vision Transformer 架构；PatchGAN 对抗训练；Sobel 边缘引导的残差密集编码；FFT 频域损失；多损失融合（L1、对抗、误差、边缘、FFT）。

**📊 数据集**

使用 BraTS2021 多模态脑肿瘤 MRI 数据集（T1、T2、FLAIR、T1ce）。

**📈 对比分析**

与 ResViT、I2I‑Mamba、TSF‑Seq2Seq、MU‑Diff 等 SOTA 方法对比，AA‑ViT 在 PSNR 达到 27.79 dB、SSIM 0.93，均显著优于对比模型。

**⚠️ 局限性**

局限性：超参数调优依赖经验；仅进行二维切片处理，缺乏 3D 上下文；临床评估样本有限；未针对失诊或伪影等关键临床错误进行专门分析。

---

## 394. Continuous and large-scale: ELEANOR, the soft architected arm inspired by the elephant trunk

**arXiv ID:** 2607.07622 | [PDF](https://arxiv.org/pdf/2607.07622v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 395. Avoiding unsafe sets when training with Langevin Dynamics

**arXiv ID:** 2607.07538 | [PDF](https://arxiv.org/pdf/2607.07538v1)

**作者:** Adam M. Oberman `[一作]` `[通讯]` (LawZero), Adam M. Oberman (LawZero)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

研究了噪声梯度下降（Langevin动力学）轨迹在高维参数空间中进入预设失败区域的概率，并给出了对该概率的上界。

**💡 创新点**

提出了两种安全性上界：① 形状无关的全局上界，只需利用目标分布的总质量和全局谱间隙；② 形状感知上界，利用局部松弛率（局部谱间隙）和最大原理天花板来消除转移峰（swelling）现象。

**🔧 技术方法**

核心技术包括：Langevin动力学的反向Kolmogorov方程、Poincaré不等式、谱分解、中心化指标（centered indicator）与χ²收敛、最大原理天花板、以及对局部松弛率的定义与估计。

**📊 数据集**

无数据集，纯理论推导。

**📈 对比分析**

无实验比较；论文通过解析推导证明上界在大维、强凸、光滑损失下指数衰减，并通过可解的Ornstein–Uhlenbeck例子展示两种上界的适用性与局限。

**⚠️ 局限性**

限制主要在于：① 仅适用于两侧 Hessian 界定的强凸光滑损失；② 形状感知上界需要局部松弛率大于等于全局谱间隙且初始分布的密度比有限；③ 对于非凸或高度非光滑问题无法直接推广。

---

## 396. Higher-Order Geometric Updates for Levenberg-Marquardt Method via Riemann Normal Coordinates

**arXiv ID:** 2607.07623 | [PDF](https://arxiv.org/pdf/2607.07623v1)

**作者:** Jianing Liu `[一作]` (University of Science and Technology of China), Dong H. Zhang `[通讯]` (Dalian Institute of Chemical Physics, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于黎曼正则坐标的 Levenberg–Marquardt（RNC‑LM）方法，用高阶展开实现非线性最小二乘优化中的有限步几何一致性。

**💡 创新点**

创新点在于：①通过一阶几何残差将 geodesic 方程转化为可递归求解的线性系统；②利用同一降噪矩阵求解所有高阶校正，避免重新因子化；③将收敛判据拆分为阻尼参数 λ 与曲线参数 t 的两层控制，并在曲线上做一次线搜索，兼顾局部模型可靠性与步长安全性。

**🔧 技术方法**

核心技术包括：黎曼正则坐标（RNC）展开、Geodesic 加速、递归求解高阶系数、基于一维曲线的自动微分实现、传统 LM 的阻尼与线搜索、以及对比实验的性能评估。

**📊 数据集**

使用的数据集包括：1) 经典 Rosenbrock 曲线（二维）; 2) NIST StRD 的 MGH10 复杂生化模型; 3) 反应–扩散 PDE 的 Physics‑Informed Neural Network（PINN）; 4) 3‑体 H₂O 的 Born–Oppenheimer 能谱（985,160 点、1000 维 FI 描述符）。

**📈 对比分析**

与标准 LM、LM‑GA（Geodesic 加速）、L‑BFGS 等方法对比。结果显示：在 Rosenbrock 与 MGH10 上，RNC‑LM 通过提高阶数或线搜索大幅减少迭代次数；在 PINN 反应–扩散任务中，RNC‑LM 使相对 L²误差降至 10⁻³ 级别，避免了过拟合；在 H₂O 能谱拟合中，RNC‑LM（四阶）实现 34 倍的 wall‑clock 时间加速，迭代次数从 5000 降至 144。

**⚠️ 局限性**

局限性包括：①仍需显式构造并因子化阻尼矩阵，限制了可处理的参数规模（≈10⁶）；②对全批量求解的依赖，难以直接扩展到极大规模或稀疏数据；③目前仅针对残差映射的拉回几何，尚未推广到更一般的统计或费舍尔信息流形；④高阶展开需要多次一维方向导数，若模型微分难以自动微分或计算代价高，可能影响效率。

---

## 397. Rethinking Code Performance Benchmarks for LLMs

**arXiv ID:** 2607.07619 | [PDF](https://arxiv.org/pdf/2607.07619v1)

**作者:** Nhat Minh Le `[一作]` (Concordia University), Chen `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四个 Python 函数级性能基准（EffiBench、Enamel、EvalPerf、Mercury）进行重新评估，采用每任务 30 次重复执行并进行 Mann‑Whitney U 检验，揭示大多数 benchmark 提供的“高效实现”并未显著快；随后设计并实现一个三智能体（生成、诊断、修复）LLM 框架，用于自动生成更能暴露性能差异的测试。

**💡 创新点**

①首次将严格的统计检验方法引入 LLM 代码性能评估；②通过对比 canonical 与 benchmark‑provided performant 实现，系统性暴露基准设计的不足；③提出基于 LLM 的多智能体自动生成性能导向测试方案，显著提高可检出性能差异的比例。

**🔧 技术方法**

使用 30 次重复执行、Mann‑Whitney U 非参数检验、Cliff's delta 计算效应大小；采用 DeepSeek‑V3.1 与 GPT‑4o 进行 LLM‑as‑a‑Judge 评判；构建三阶段（生成、诊断、修复）LLM 自动测试生成框架；在原基准与新生成测试集上进行性能比较。

**📊 数据集**

共 1,538 个任务，来源于 EffiBench、Enamel、EvalPerf、Mercury，每个任务包含 canonical 方案和 benchmark‑provided 的性能实现。

**📈 对比分析**

对每个任务的 canonical 与性能实现分别执行 30 次，利用 Mann‑Whitney U 检验判断显著性；在原测试集下仅 6.11% 的实现显著更快；使用新生成测试后，DeepSeek‑V3.1、GPT‑4o 分别提升到 24.01% 和 25.43%；在 LLM 生成的实现上，显著提升 22.19%。

**⚠️ 局限性**

仅聚焦 Python 函数级任务，未覆盖多线程、IO、内存等其他性能维度；生成测试受 LLM 随机性影响；基准设计仍可能过度简化，难以直接推广至更复杂的系统级评估。

---

## 398. Simplification of the Isotropic Generalized Stop-Type Prandtl-Ishlinskii Vector Hysteresis Operator Using Analytical Return-Point Mapping

**arXiv ID:** 2607.07575 | [PDF](https://arxiv.org/pdf/2607.07575v1)

**作者:** Arvinth Shankar `[一作]` (Robert Bosch GmbH), Sebastian Schöps `[通讯]` (Technische Universität Darmstadt)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种热力学一致的简化版本的广义Prandtl–Ishlinskii停止算子，保留非滞后抗滞后响应并简化滞后部分，针对各向同性材料实现了闭式返回点映射；

**💡 创新点**

创新点在于将传统需要迭代Newton求解的非线性滞后项简化为线性加权，保留非滞后非线性，且在各向同性情况下实现完全解析的更新公式，显著降低计算复杂度；

**🔧 技术方法**

采用热力学框架、向量停止模型、最大耗散原理、拉格朗日乘子法、返回点映射以及有限元求解器；

**📊 数据集**

使用的是二维永磁同步电机（PMSM）有限元模型，10,000个单元，材料为M330-35A各向同性磁滞材料，驱动90个转子位置的静态仿真；

**📈 对比分析**

与传统广义模型（在线和离线预处理两种实现）进行对比，简化模型在在线实现时耗时减少88%，离线实现时减少61%；在不考虑涡流耦合的情况下，最大相对损耗误差约6.8%，加入涡流耦合后误差降至约0.3%；

**⚠️ 局限性**

目前仅针对各向同性材料有效，难以直接推广到各向异性情形；简化后在极端大磁通下误差略增，需进一步验证其在更复杂磁场耦合场景中的稳健性。

---

## 399. Stochastic Online Euclidean TSP

**arXiv ID:** 2607.07537 | [PDF](https://arxiv.org/pdf/2607.07537v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 400. HIVE: Understanding Post-Hallucination Reasoning in Vision Language Models

**arXiv ID:** 2607.07507 | [PDF](https://arxiv.org/pdf/2607.07507v1)

**作者:** Feng He `[一作]` (Purdue University), Qiankun Li `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了HIVE评估平台，用以在视觉语言模型中系统比较真实与幻觉式描述对下游推理性能的影响，探索Post‑Hallucination Reasoning（PHR）现象。

**💡 创新点**

创新点包括：①首次定义并系统研究PHR；②构建可对照的Caption生成–判别–任务求解管线，使得幻觉与真实描述的差异仅来自幻觉本身；③通过多模型、多任务实验揭示幻觉在视觉语言任务中可提升性能的结构化规律。

**🔧 技术方法**

主要技术手段：统一的prompt‑driven caption生成、集成式幻觉判别器（多检测器多数投票）、三种输入条件的任务求解（Raw/Faithful/Hallucinated）；温度/Token预算控制、嵌入分布与熵分析、链内/链外收敛性评估、token‑级 ablation 与插值实验。

**📊 数据集**

使用9个文本与视觉语言任务，包括ISIC、PlantVillage、GQA、Dex‑Net、G-? 等；文本任务如ProofWriter、SARA v3、AntiCP2、BBBP、CodeXGLUE等；对应模型涵盖GPT‑4o、Claude‑3 Sonnet、Gemini‑2.0‑Flash、Qwen‑VL‑Max等。

**📈 对比分析**

通过Raw、+Faithful、+Hallucinated三条件对比，计算Δ(H–F)。在视觉语言任务中，幻觉式描述平均提升约5–15%准确率，单个数据集如ISIC可提升16.9%，PlantVillage 14.7%；文本任务提升有限，偶有小幅改善。实验覆盖9个模型，验证幻觉效应的普遍性。

**⚠️ 局限性**

限制：评估仅基于固定的基准数据集，未涵盖中间推理步骤或隐层中的幻觉；幻觉对推理的好处受温度、Token预算等超参控制，过度幻觉会引入噪声；实际部署需进一步机制以平衡探索与可靠性。

---

## 401. Two-player Alternate Uses Test: A Controlled Testbed for Interactive Human-AI and Human-Human Co-Creation

**arXiv ID:** 2607.07522 | [PDF](https://arxiv.org/pdf/2607.07522v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 402. PALS: Percentile-Aware Layerwise Sparsity for LLM Pruning

**arXiv ID:** 2607.07557 | [PDF](https://arxiv.org/pdf/2607.07557v1)

**作者:** Yazdan Jamshidi `[一作]` (Palo Alto Networks), Alexey Shvets `[通讯]` (Palo Alto Networks)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PALS（Percentile‑Aware Layerwise Sparsity），在一阶无训练的 LLM 剪枝中根据每层 99% 分位的激活幅度自适应调整稀疏率，结合 Wanda 的权重评分实现无额外训练成本的层级稀疏分配；

**💡 创新点**

创新点在于用激活分位数而非梯度或均值来衡量层重要性，给每层分配不同的稀疏比例，并证明梯度基准在预训练 LLM 的离散剪枝中效果不佳；

**🔧 技术方法**

使用 Wanda 的 |w|·mean(|a|) 权重评分，计算 99% 分位激活，标准化后乘以调节因子 α 再 clip 于 ±5% 约束，再对每层按该稀疏率做一阶无训练剪枝；

**📊 数据集**

校准集采用 128 条 2048 词长的 C4 语料；评估数据包括 WikiText‑2（语言建模困惑度）、BoolQ、PIQA、HellaSwag、MMLU（零样本任务准确率）；

**📈 对比分析**

与 Dense、Magnitude、Wanda、SparseGPT、PALS‑Gradient 等基线比较：在 LLaMA‑2‑7B 上 50% 稀疏率下，PALS 将 WikiText‑2 困惑度从 12.92 降至 10.96（约 15% 改善），在 LLaMA‑3‑8B 与 Mistral‑7B 上提升有限；下游任务平均准确率从 63.8% 提升至 64.8%；

**⚠️ 局限性**

局限性包括：仅针对无结构稀疏，需稀疏矩阵硬件支持；实验规模仅 7–8B，未知是否扩展到更大模型；未进行微调；仅使用英文数据集，跨语言或代码生成的表现未知；激活统计的校准稳定性未系统评估。

---

## 403. SonoRank: Towards Calibration-Free Real-Time Finger Flexion Detection from Forearm Ultrasound Sequences

**arXiv ID:** 2607.07542 | [PDF](https://arxiv.org/pdf/2607.07542v1)

**作者:** Dean Zadok `[一作]` (Technion), Oren Salzman `[通讯]` (Technion)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种两阶段、无标定的前臂超声波指关节检测框架，通过对序列对进行排名学习，再细化为指关节弯曲分类，实现实时控制。

**💡 创新点**

创新点在于采用对比式时序排名作为预训练信号，利用相对运动大小消除个体差异，并将休息参考与深度模型结合，实现跨用户、无标定的指关节检测。

**🔧 技术方法**

使用CNN帧编码器+Transformer时序聚合+Siamese对比头+MLP分类头，并结合二元交叉熵排名损失，训练时使用B‑mode超声序列与同步Vicon关节角度。

**📊 数据集**

在12名健康右手者中收集约84,000帧、51条录制的前臂超声波与同步运动学数据，采用12折留一子测试交叉验证。

**📈 对比分析**

与直接分类、无排名预训练、以及四个外部超声方法相比，所提方法在留一子测试下F1提升约28%，平均F1为0.63，AUC为0.69，明显优于外部基线（≤0.21）。

**⚠️ 局限性**

局限性包括仅在健康受试者、单指独立动作场景验证，拇指识别最差，跨受试者差异大；在截肢人群或多指复杂动作中尚未验证。

---

## 404. FedMark-FM: Auditable, Risk-Adjusted Data Markets for Federated Foundation-Model Adaptation

**arXiv ID:** 2607.07529 | [PDF](https://arxiv.org/pdf/2607.07529v1)

**作者:** Phat T. Tran-Truong `[一作]` (Ho Chi Minh City University of Technology), Minh Nhat Nguyen `[通讯]` (RMIT University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了 FedMark-FM，一个可审计、风险调整的数据市场框架，用于对联邦基础模型适配中的异构私有 artefacts（检索语料、提示、LoRA 适配器、偏好/安全数据等）进行定价与支付。

**💡 创新点**

核心创新在于①将 artefacts 当作类型化商品而非统一数据或梯度；②提出“pipeline‑ordered” Shapley 估值，以服务顺序分配信用；③开发安全代理 Shapley（S3Val）估计器，结合草图、冗余聚类、分层采样与不确定性触发审计；④制定基于下置信界的预算可行支付规则，惩罚重复、Sybil、毒化、隐私游戏和成本膨胀。

**🔧 技术方法**

使用安全草图（如 MinHash、聚类向量）、分层/链式 Shapley 估计、分层采样、学习型效用代理、拆分式不确定性评估、下置信界支付与风险加权。

**📊 数据集**

实验数据集包括 FEVER 检索语料库、FEVER 生成式 RAG（含 prompt‑injection 恶意测试）、以及低秩 LoRA（AG News）训练集，涵盖多种攻击者（Sybil、重复、毒化、隐私、成本膨胀）与稀缺专家。

**📈 对比分析**

与传统均匀支付、体量、leave‑one‑out、FL‑Shapley、检索相似度、Shapley‑UCB 等基线对比，FedMark‑FM 在受攻击场景下提升下游准确率 7.5–8.1 分点，避免选中任何战略客户端，并在大规模（50/100/200 客户端）去环验证中保持最高或相当的准确率。

**⚠️ 局限性**

局限在于：仅提供近似的策略抵抗（非主导策略真诚保证）；主要关注低成本评估与审计而非完整的多适配器 PEFT 评估；隐私保护仅依赖 DP 聚合与硬件可信执行，未实现端到端隐私证明；并且在不同基础模型或检索堆栈变化时需要重新估值。

---

## 405. Future Confidence Distillation in Large Language Models

**arXiv ID:** 2607.07626 | [PDF](https://arxiv.org/pdf/2607.07626v1)

**作者:** Sahil Kale `[一作]` `[通讯]` (University of California Los Angeles), Sahil Kale (University of California Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在回答过程中的置信度随时间的变化，并提出了未来置信度蒸馏方法来在答案生成前预测置信度。

**💡 创新点**

创新点是将置信度分为预解 Feeling-of-Knowing 与后解 Judgement-of-Learning，并利用后解置信度对前解隐藏表示进行蒸馏，以低成本获得更好的置信度校准。

**🔧 技术方法**

采用线性探针在隐藏层表示上进行置信度预测，使用置信度蒸馏的线性回归/岭回归，以及对比预解、后解口头置信度的评估。

**📊 数据集**

在事实回忆、逻辑推理、数学推理三大领域的多种公开数据集（如 TriviaQA、ARC、MATH 等）上进行实验。

**📈 对比分析**

与现有的口头置信度估计、隐藏表示探针等方法比较，后解置信度和蒸馏后的前解置信度在 ECE 下降 20–40%、AUROC 提升 5–15% 方面表现优异。

**⚠️ 局限性**

局限在于仅在单模态语言模型上验证，跨领域迁移效果有限，且对大规模推理模型的适用性尚未探索。

---

## 406. Embedded Blockchain Infrastructure Management (eBIM): A RISC-V-Empowered Hardware--Software Co-Design Framework Towards Trustworthy Blockchain

**arXiv ID:** 2607.07625 | [PDF](https://arxiv.org/pdf/2607.07625v1)

**作者:** Qinglin Yang `[一作]` (Guangzhou University), Zhihong Tian `[通讯]` (University of Stavanger)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并描述了基于RISC-V的嵌入式区块链基础设施管理（eBIM）框架，并综述相关技术与应用。

**💡 创新点**

将RISC-V开放指令集与硬件/软件协同设计相结合，实现可信执行、加速加密、zkVM支持等功能，将区块链基础设施从云服务迁移至边缘设备。

**🔧 技术方法**

RISC‑V指令集、可信执行环境（TEE）、加速器、零知识VM、后量子密码学扩展及硬件/软件协同设计。

**📊 数据集**

未使用公开数据集，主要为文献综述与案例分析。

**📈 对比分析**

文中未给出实验对比，引用已有RISC‑V加速器与zkVM的性能提升以说明潜在优势。

**⚠️ 局限性**

缺乏统一的RISC‑V加密扩展标准，系统级基准与安全分析不足，侧通道风险与TEE‑zkVM安全性待研究，资源与功耗限制了实际应用。

---

## 407. On possible values of the group complexity function of infinite words

**arXiv ID:** 2607.07620 | [PDF](https://arxiv.org/pdf/2607.07620v1)

**作者:** Maksim Launer `[一作]` (Saint Petersburg State University), Ekaterina Voloshinova `[通讯]` (Saint Petersburg State University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了无限词的“通用群复杂度”属性，即在每个长度 n 上，所有介于阿贝尔复杂度与因子复杂度之间的整数都能通过某个对称群子群实现。研究者证明了 Sturmian 词具有此属性，并对三元最小复杂度词（I、II、III 型）以及周期词进行了分类与分析，给出了算法判定方法，并提出若干未解决的开放问题。

**💡 创新点**

创新点在于提出通用群复杂度概念并完成了 Sturmian 词的证明，首次系统探讨三元最小复杂度词中不同类型的群复杂度取值，并为周期词提供了判定通用复杂度的算法性标准，进一步扩展了传统因子与阿贝尔复杂度的研究视角。

**🔧 技术方法**

主要采用组合学与群论工具：因子与阿贝尔复杂度定义、群复杂度构造、词的词法（lexicographic）排列、Sturmian 词的标准序列与连续分数表示、符号映射与逆向运算、以及群作用的等价类分析。

**📊 数据集**

本研究为纯理论性工作，无使用实验数据集；所有结论均通过数学证明得到。

**📈 对比分析**

通过严谨的理论推导与证明对比，本文未进行数值实验；相对已有的因子复杂度与阿贝尔复杂度研究，提供了群复杂度之间的完整覆盖证明，验证了 Sturmian 词的通用性，并对三元最小复杂度词给出了可实现值的完整描述。

**⚠️ 局限性**

局限性：仅完成了三元字母表下最小复杂度词的部分分类（III 型仍有未解决的 4 复杂度情况），未覆盖更大字母表；对周期词的判定只给出算法框架，具体实现与复杂度分析尚未深入；开放问题未解，表明理论框架仍不完整。

---

## 408. Dual Latent Memory in Vision-Language-Action Models for Robotic Manipulation

**arXiv ID:** 2607.07608 | [PDF](https://arxiv.org/pdf/2607.07608v1)

**作者:** Hongyu Qu `[一作]` (Nanjing University of Science and Technology), Shuicheng Yan `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种全latent记忆框架（LaMem‑VLA），在视觉‑语言‑动作模型中自适应地将短期视觉记忆和长期语义记忆嵌入同一连续嵌入空间，直接参与决策；

**💡 创新点**

创新点在于将历史经验从外部上下文完全迁移到模型本地的latent空间，记忆的存取、重构和消费都在同一embedding空间完成，从而实现记忆与多模态推理的无缝融合；

**🔧 技术方法**

采用Prismatic 7B VLM + LLaMA‑7B背骨，Transformer式记忆模块（curator、seeker、condenser、weaver），并使用扩散式动作专家；记忆通过压缩、检索和重构得到固定长度的latent记忆token；

**📊 数据集**

使用Open‑X Embodiment预训练后，主要实验数据集为SimmerEnv‑Bridge与LIBERO仿真任务；

**📈 对比分析**

在SimmerEnv‑Bridge上与CogACT、MemoryVLA、π_0等基线对比，成功率提升至73.9%（比CogACT高16.6%），在LIBERO上平均成功率达97.6%（比MemoryVLA高1.1%，比CogACT高4.4%），体现显著性能提升；

**⚠️ 局限性**

目前仅在仿真环境中验证，缺乏真实机器人实验；检索与压缩过程对计算开销和检索误差的敏感性未得到充分评估。

---

## 409. Approximability of Electrical Distribution Network Reconfiguration for General Graphs

**arXiv ID:** 2607.07600 | [PDF](https://arxiv.org/pdf/2607.07600v1)

**作者:** Christian Wallisch `[一作]`, Leon Kellerhals `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文提出并分析了配电网络重构（DNR）问题的近似算法，研究单源和多源场景下的最小功率损耗树的计算。

**💡 创新点**

主要创新在于：①对单源且阻抗均匀的DNR给出 √n 近似算法；②证明多源DNR在平面图上无 n^{1-ε} 近似；③给出 2-DNR 的 Ω(log²n) 下界和 n‑1 近似上界。

**🔧 技术方法**

采用随机化逼近、凸二次规划松弛、转移至分散流/共性流等技术，并利用图分割、平面化和多源流转化等构造性证明。

**📊 数据集**

论文未使用公开数据集，所有结果均为理论上界与下界。

**📈 对比分析**

通过比较已知的多源与单源 DNR 近似结果，证明在最坏情况下单源 √n 近似已是最优阶；多源 n‑1 近似与 Ω(log²n) 下界相距多倍，凸显问题难度。

**⚠️ 局限性**

局限性包括：对非均匀阻抗的单源情况仍无有效近似；多源近似上界与下界之间差距大；未考虑容量、电压约束或实际操作序列的动态重构。

---

## 410. What Makes a Good Bug Report for an AI Agent?

**arXiv ID:** 2607.07593 | [PDF](https://arxiv.org/pdf/2607.07593v1)

**作者:** Lara Khatib `[一作]` (University of Waterloo), Thomas Zimmermann `[通讯]` (University of California, Irvine)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对SWE‑bench Verified和SWE‑bench Pro两个公开基准进行统计建模和控制消融实验，探究哪些bug报告特征能提升LLM驱动的自动程序修复（APR）代理的修复成功率。

**💡 创新点**

创新点在于首次系统评估并对比人类开发者认为重要的bug报告特征与AI代理在实际修复中的效果，揭示代理更依赖可执行重现脚本、定位信息与报告结构，而对自然语言步骤与可读性等人类重视特征的敏感性较低；同时提出了基于报告内容的“代理友好”报告设计准则。

**🔧 技术方法**

使用的技术包括：混合效应逻辑回归模型（控制Bug难度与代理差异）来分析27个特征的关联性；对两款大模型（Qwen3.6‑35B‑A3B和Gemma‑4‑31B‑IT）在mini‑SWE‑agent框架下进行三次重复运行的控制消融实验，计算solve@3和平均修复率的变化；并利用LLM自动标注与人工复核对特征进行注释。

**📊 数据集**

所用数据集为SWE‑bench Verified（433个bug实例，87名代理）和SWE‑bench Pro（283个bug实例，针对两款模型分别筛选可解集合）两大公开基准。

**📈 对比分析**

方法比较通过在控制消融实验中记录在去除或保留特定报告组件后，模型的solve@3下降幅度（以百分点计）和平均修复率变化来评估特征的重要性。结果显示：fix建议、仓库代码、重现脚本、文件定位等特征可将solve@3提升约30–40个百分点；相反，报告长度和自然语言步骤对代理修复几乎无正面影响，甚至负面影响。

**⚠️ 局限性**

主要局限包括：第一，观察性研究无法确定因果关系；第二，可能存在训练数据泄露影响Verified基准结果；第三，实验仅覆盖两款开源大模型与简化的mini‑SWE‑agent框架，无法推广到所有商业模型或更复杂的工作流；第四，特征注释依赖LLM初步标注，虽后续人工校对但仍可能存在误差；第五，评估仅基于测试通过与否，未保证修复的完全正确性。

---

## 411. Cardiac MRI Through-Plane Super-Resolution Guided by Reference and Memory

**arXiv ID:** 2607.07581 | [PDF](https://arxiv.org/pdf/2607.07581v1)

**作者:** Shaoming Pan `[一作]` (University of Texas At Arlington), Meng Ye `[通讯]` (University of Texas At Arlington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种参考和记忆引导的心脏MRI切面超分辨率框架STRMSR，通过利用同一受试者的高分辨率参考视图和中间超分辨率结果来重建高分辨率心脏体积。

**💡 创新点**

创新点在于提出了粗到细的上下文匹配和基于记忆的切片间超分辨率传播机制，增强了重建的体积一致性和细节传递能力。

**🔧 技术方法**

使用了深度学习技术，包括双分支变换器（Swin Transformer）进行特征提取，以及动态特征聚合模块来实现内容自适应的特征融合。

**📊 数据集**

使用了WHS心脏MRI数据集，分为两种参考协议：正交平面视图和长轴腔室视图。

**📈 对比分析**

与基线方法（如Bicubic、MsFF-Net、MINet等）进行比较，STRMSR在×4和×8的超分辨率上均表现出显著的性能提升，尤其在×8时，STRMSR的PSNR比McMRSR高出0.97 dB，且所有改进在统计上显著（p<0.001）。

**⚠️ 局限性**

限制在于该方法依赖于高质量的参考视图，且在参考视图稀疏的情况下可能会影响性能。

---

## 412. Unconditional Lower Bounds for Degree Fault Tolerant Spanners

**arXiv ID:** 2607.07576 | [PDF](https://arxiv.org/pdf/2607.07576v1)

**作者:** Greg Bodwin `[一作]` (University of Michigan), Aleksey Lopez `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了在f‑度数容错（f‑DFT）模型下，任意n点图的(2k-1)乘法扩张子图（即f‑DFT（2k-1）-spanner）至少需要Ω(f^{1-1/k}·n^{1+1/k})条边；这一下界与已知上界（O(exp(k)·f^{1-1/k}·n^{1+1/k})）相匹配，且不需要依赖于Girth Conjecture，首次给出了无条件的近最优下界。

**💡 创新点**

创新点在于：①通过分析Wenger图的代数结构，构造了一套“fault sets”F_e，使得删除对应的匹配后原图不再包含小循环，从而实现对子图的容错性证明；②利用云吹起（cloud blowup）技术将k=1情形推广到任意f，获得了完整的无条件下界；③首次在f‑DFT spanner领域提供了与上界匹配的下界，弥补了此前仅在假设Girth Conjecture下得到的结果。

**🔧 技术方法**

主要技术包括：代数构造（Wenger图与其改写），路径收缩（path‑straightening）技巧以消除重复斜率的路径；闭合非退化循环分析；云吹起（cloud‑blowup）构造将单点拓展为f个复制节点并保持容错度。

**📊 数据集**

本文为理论工作，没有使用实测数据集，所有结论基于构造性的代数图（Wenger图）和其云扩张。

**📈 对比分析**

对比方法：作者将其下界与已知上界（Bodwin等人的O(exp(k)·f^{1-1/k}·n^{1+1/k})）以及在Girth Conjecture假设下的下界进行对比。结果显示：在不依赖任何猜想的前提下，本文下界与上界在指数因子k之外基本匹配，证明了该问题的最优性。

**⚠️ 局限性**

局限性：①下界仅适用于乘法扩张子图模型；②上界仍保留exp(k)因子，尚未完全匹配；③证明依赖于Wenger图的特殊结构，可能难以推广至更一般的图类或更复杂的容错模型；④未给出具体构造算法，只给出存在性证明。

---

## 413. On Computing Minimum Wheeler DFA From Their Language

**arXiv ID:** 2607.07563 | [PDF](https://arxiv.org/pdf/2607.07563v1)

**作者:** Ruben Becker `[一作]` (Ca' Foscari University of Venice), Daniel Puttini `[通讯]` (Ca' Foscari University of Venice)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种线性对数时间的算法，用于构造给定Wheeler语言的最小Wheeler DFA，解决了此前只适用于无环DFA或速度极慢的通用方法的缺陷。

**💡 创新点**

创新点在于将最小化问题转化为增量式状态复制与转移重排，并利用Wheeler自动机的总序性质与Myhill‑Nerode等价性，证明算法在Wheeler语言下终止并得到最优解。

**🔧 技术方法**

核心技术包括：动态Wheeler序列表示（利用可插入字符串与位向量）、区间查找与更新、状态复制与转移批量重命名、以及通过预计算与批处理实现的O(log m_w)更新/查询。

**📊 数据集**

使用真实的全基因组变异图（人类参考染色体的VCF变异）生成的23个DFA作为测试数据集。

**📈 对比分析**

与现有唯一可生成Wheeler DFA的GCSA工具进行对比，实验显示本实现虽在某些实例上内存占用略高，但在多数染色体上能完成构造，速度与GCSA相当或更快，且输出严格为最小Wheeler DFA。

**⚠️ 局限性**

局限性包括对极大Wheeler DFA（如染色体16、17、18）仍会因输出规模巨大而无法在可接受时间内完成，且依赖动态数据结构在大规模实例上存在性能瓶颈。

---

## 414. Generating Personalized Lower-Limb Kinematics Across Walking Speeds Using Subject-Conditioned Diffusion

**arXiv ID:** 2607.07533 | [PDF](https://arxiv.org/pdf/2607.07533v1)

**作者:** Diya Dinesh `[一作]` (Carnegie Mellon University), Inseung Kang `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

利用主题条件残差扩散框架，从单一步态速度的下肢关节角度序列生成个性化跨速度步态，以减少外骨骼个性化所需的运动捕捉数据。

**💡 创新点**

创新点在于：1）将残差扩散模型与受试者步态特征嵌入相结合；2）使用FiLM对Transformer denoiser进行多模态条件；3）实现单源速度即可生成多速度个性化步态并保留受试者身份，显著降低数据收集负担。

**🔧 技术方法**

采用Transformer denoiser、残差扩散（DDIM）、双向LSTM编码受试者嵌入、速度嵌入、FiLM特征线性调制、交叉熵/对比损失等技术。

**📊 数据集**

使用两套数据集：22名健康受试者在0.5–1.85 m/s（22速度）和19名中风受试者在0.3–1.2 m/s（6速度）通过Vicon 200 Hz采集的下肢矢状面角度，后降采样至20 Hz。

**📈 对比分析**

与MLP和TCN基线在同一训练/测试分割下对比；健康受试者平均MAE为3.4°，中风受试者为6.0°，相较基线降低约70%；个人化排名平均3.3/2.8（低即好），步态节奏误差<7%，单源速度即可匹配四源速度的精度。

**⚠️ 局限性**

局限性包括：中风受试者样本量有限；仅建模矢状面运动，未考虑侧向或转动面；未直接验证对外骨骼控制的实效；仍依赖专业运动捕捉设备，未来需整合计算机视觉估计。

---

## 415. Context-Aware Slum Mapping in Sub-Saharan Africa Using Sentinel-1 Texture and Local Climate Zones

**arXiv ID:** 2607.07532 | [PDF](https://arxiv.org/pdf/2607.07532v1)

**作者:** Peterson Chepkilot `[一作]` (Sapienza University of Rome), Paolo Gamba `[通讯]` (University of Pavia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究提出了一种结合 Sentinel‑2 光谱与 Sentinel‑1 SAR 纹理的三层特征融合框架，用于提高撒哈拉以南非洲城市中轻型低层（LCZ 7）与正式紧凑低层（LCZ 3）之间的分辨率。

**💡 创新点**

创新点在于：① 将 SAR 双极化回波、灰度共生矩阵纹理和物理引导的结构指数三层级别地嵌入 LCZ 分类；② 通过分层消融验证纹理特征在解决 LCZ 3/7 混淆中的主导作用；③ 在缺乏大量标注的 SSA 环境下实现可解释且季节稳定的高精度分辨。

**🔧 技术方法**

使用的技术包括：多时相 Sentinel‑1 GRD 双极化预处理、GLCM 纹理提取、物理归一化指数构造、随机森林分类器以及分层消融与多季节评估。

**📊 数据集**

数据集涵盖：Nairobi 与 Eldoret 两个肯尼亚城市的 Sentinel‑1 与 Sentinel‑2 影像（干湿季），以及手工矢量化的 LCZ 参考多边形；另外在 Kigali 进行零样本与迁移学习测试。

**📈 对比分析**

与 WUDAPT 光学基准进行对比；在源域（Nairobi + Eldoret）通过分层消融得到的最优模型在干季 OA 81.6%、湿季 80.7%，LCZ 7 F1 分别提升至 0.667/0.671，显著优于光学基准（OA 70.4%、F1 0.200），且在季节间保持一致性；在 Kigali 的零样本迁移表现低下，但加入当地标注后可恢复性能。

**⚠️ 局限性**

主要局限：仅使用双极化 Sentinel‑1，缺乏更丰富的四极化或相干信息；纹理特征仍对残留散斑敏感；跨城市迁移在零样本情况下效果有限，需本地适配；且研究仅针对两座肯尼亚城市，普适性需进一步验证。

---

## 416. Creativity from Friction: Human-AI Interaction for Exploratory Structural Design

**arXiv ID:** 2607.07521 | [PDF](https://arxiv.org/pdf/2607.07521v1)

**作者:** Ricardo Maia Avelino `[一作]` (ETH Zurich), Mennatallah El-Assady `[通讯]` (ETH Zurich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并评估了一种面向结构设计的交互式人机协作工作流，支持基于约束的创意探索；

**💡 创新点**

提出“受约束的共创”概念，定义四个关键设计维度（模型知识根植、人机可读数据结构、状态感知与交互历史、跨模态意图表达），并展示其在结构设计中的应用；

**🔧 技术方法**

利用谷歌 Gemini 3.1 Flash‑Lite 视觉‑语言模型实现多模态对话、草图与文本指令驱动的模型编辑；

**📊 数据集**

未使用公开结构设计数据集，而是通过实验室自建的三位结构与建筑专业参与者进行的案例任务；

**📈 对比分析**

主要采用定性研究方法（访谈、观察、工作日志）进行对比，未给出量化指标，结果表明 AI 能有效降低重复建模摩擦，但对创意产生的影响尚未量化；

**⚠️ 局限性**

限制包括样本量极小、CAD 交互功能不完整、AI 在本研究中未主动提出创意或重构方向、缺乏结构合理性评估与客观性能对比。

---

## 417. Stability and Convergence of Optimistic Exponential Weights with Asymmetric Step Sizes in Bimatrix Games

**arXiv ID:** 2607.07517 | [PDF](https://arxiv.org/pdf/2607.07517v1)

**作者:** Hédi Hadiji `[一作]` (CentraleSupélec), Sarah Sachs `[通讯]` (University of Bristol)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究两人零和博弈中的乐观指数学习算法，证明其在满足一定步长与参数条件下能够收敛至混合均衡；

**💡 创新点**

创新点在于给出了完整的雅可比谱分解与稳定性判据，表明收敛与步长乘积关系而非单一步长大小，并提出对混合均衡局部稳定性的精确阈值；

**🔧 技术方法**

利用雅可比矩阵的谱分析、Schur‑Cohn 判据、连续根性质及矩阵范数估计等数学工具；

**📊 数据集**

无实验数据集，全部为理论推导与符号矩阵计算；

**📈 对比分析**

与传统指数学习、OGDA 等方法比较，证明在满足参数约束时收敛速度优于普通指数学习，且可获得更宽松的稳定性区间；

**⚠️ 局限性**

局限在于仅得到局部稳定性判据，对于全局收敛在 m ≤ 1/2 时仍未解决，并且对非零和或非完全混合均衡情况的分析不足。

---

## 418. Accelerating Industrial Finite Element Simulations of Electric Machines based on Runtime Analysis

**arXiv ID:** 2607.07514 | [PDF](https://arxiv.org/pdf/2607.07514v1)

**作者:** Arvinth Shankar `[一作]` (Robert Bosch GmbH), Sebastian Schöps `[通讯]` (Technische Universität Darmstadt)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文针对电机有限元仿真中的计算瓶颈，分别在二维静态和三维瞬态模型上通过改进线搜索、并行化（bi）线性形式以及优化预条件器，显著降低求解时间。

**💡 创新点**

创新点在于：①将线搜索限制为仅 Armijo 条件并结合自适应初始步长；②对求解器内部的 Jacobian 与残差求解做多线程并行化；③在 H(curl) 空间中引入梯度子空间修正的 AMS 预条件器，并配合 Hypre 与 PETSc 的 OpenMP 并行。

**🔧 技术方法**

使用技术包括：Newton 迭代、Armijo 线搜索、回溯退化、OpenMP 并行、AMS 预条件器、Hypre AMG、PETSc MINRES、并行化（bi）线性形式组装。

**📊 数据集**

数据集：二维永磁同步电机模型（10k、40k 元素）与三维永磁同步电机模型（600 万元件、21 步时间步长）。

**📈 对比分析**

与原始实现比较，二维案例中 Armijo 只搜索降低 25% 运行时，二次多线程化降 51%，两者叠加降 58%；三维案例中梯度修正预条件器降低 14%，再加 OpenMP 8 线程降 48%，再加 Armijo 只搜索降 55%。

**⚠️ 局限性**

局限性：改进主要针对所用电机模型，效果随问题规模和硬件环境变化；自适应步长在大多数旋转位置效果有限；对更一般化的电磁问题或更复杂耦合场景的适用性尚待验证。

---

## 419. Fast Rates for Semi-Supervised Learning via Data-Augmentation Graph Regularization

**arXiv ID:** 2607.07513 | [PDF](https://arxiv.org/pdf/2607.07513v1)

**作者:** Adam M. Oberman `[一作]` `[通讯]` (McGill University), Adam M. Oberman (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在半监督学习中，作者通过数据增强构造相似图，并利用图拉普拉斯正则化实现标签传播，证明了在标签数 n_L 上的快速收敛速率 O(1/n_L)；

**💡 创新点**

创新点在于将快速稳定性分析（Johnson–Zhang 机制）迁移至数据增强图，提出了与增量质量相关的误差项 R_DA(y)，并给出了在无密度/参数假设下的转导式快速收敛理论；

**🔧 技术方法**

使用图拉普拉斯正则化、留一法算法稳定性、σ-可接受损失、谱分解与 Davis–Kahan 定理、以及强化学习框架中的数据增强协方差核；

**📊 数据集**

主要在合成的随机块图（Stochastic Block Model）上验证理论，随后在 CIFAR‑10 上通过冻结 SimCLR 特征评估标签效率；

**📈 对比分析**

与传统线性岭回归探针对比，图拉普拉斯探针在仅使用约4%标签时即可达到 90.1% 的测试准确率，显示出显著的标签效率；

**⚠️ 局限性**

局限在于理论是转导式的，仅适用于固定未标记样本；常数 λ/a 需要经验选择，且对图连通性与边权分布有一定假设。

---

## 420. From Noisy Traces to Root Causes: Structural Trajectory Analysis and Causal Extraction for Agent Optimization

**arXiv ID:** 2607.07702 | [PDF](https://arxiv.org/pdf/2607.07702v1)

**作者:** Ying Chang `[一作]` (University of Chinese Academy of Sciences), Yuqing Yang `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了STRACE框架，用于长时序代理的反射式优化，通过失败模式挖掘和因果定位构建高信噪比上下文来提升LLM优化效果。

**💡 创新点**

创新点在于将执行轨迹视为结构化因果图，先过滤冗余失败轨迹，再在文本依赖图中进行逆向切片定位根因模块，从而实现更精准、高效的优化。

**🔧 技术方法**

采用执行依赖图构建、统计严重性与结构路径聚类、逆向因果切片、LLM元控制器进行策略抽象与注入等技术。

**📊 数据集**

使用了HotpotQA、WebArena、VeruSAGE-Bench（五个Rust项目）等公开长轨迹数据集。

**📈 对比分析**

与基线（单向提示、失败感知RAG、Summary/ Retrieval压缩、TextGrad梯度法、GEPA进化法）比较，STRACE在HotpotQA提升12.9%成功率，VeruSAGE-Bench提升16.0%，且在成本-性能曲线上表现最佳。

**⚠️ 局限性**

局限在于需要对代理系统有足够可视性（代码、工具接口等），不适用于纯黑盒或仅轨迹可获取的场景。

---

## 421. SkillCenter: A Large-Scale Source-Grounded Skill Library for Autonomous AI Agents

**arXiv ID:** 2607.07676 | [PDF](https://arxiv.org/pdf/2607.07676v1)

**作者:** Tianming Sha `[一作]` (Stony Brook University), Yushun Dong `[通讯]` (Florida State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了 SkillCenter，一个包含超过 216,000 条结构化、可检索且源根基的技能库，并提供完整的自动化管线（SkillGate）将学术论文、开源代码、网页和社区资料转化为可供自主代理使用的离线 SQLite FTS5 包。

**💡 创新点**

创新点：① 将 LLM 作为质量门控与技能生成器，自动化生成、验证、去重与发布；② 通过精确的源引用保证每条声明可追溯；③ 将技能拆分为 24 个领域束，支持离线检索并可按项目类型动态加载；④ 将社区贡献与自动化管线并行，形成可持续扩展的生态。

**🔧 技术方法**

使用技术：LLM‑based SkillGate（质量分、可执行性筛选），模板驱动生成与多轮迭代改进，源根基检查（子串匹配），MinHash+LSH 去重，SQLite FTS5 全文检索，自动化脚本与 CI/CD，集成 OpenClaw/OPAL 等代理框架。

**📊 数据集**

使用数据集：学术期刊（PLOS、eLife、Nature 系列）、ArXiv 预印本、GitHub 仓库、Web 页面、Stack Overflow 讨论；社区采集的 GitHub SkillMD（90k 条）和 ClawHub 市场（11k 条）。

**📈 对比分析**

比较方法：在 4 个 LLM（Gemini‑3.5‑flash、Claude‑haiku‑4.5、GPT‑5‑mini、Claude‑sonnet‑4.6）上进行单轮 A/B 测试，比较基线、关键字检索、虚假技能（placebo）与 oracle（已知所需技能）四种情形。结果显示：当任务超出模型自身知识时，oracle 能显著提升 61–78% 的通过率；关键字检索在已知知识任务中无提升，甚至在中级模型上略有下降；在已知知识任务中，检索缺失导致 0% 通过，证明检索召回是瓶颈。

**⚠️ 局限性**

限制：① 质量评分由单一 LLM 内部判断，82% 的技能得分聚集在 4 分，缺乏外部人类评估；② 源根基检查只能保证引用可追溯，不能验证真实性或时效性；③ 去重仅基于词汇相似，可能低估语义重复；④ 现有管线使用专有 LLM，难以复现；⑤ 未进行安全/攻击防护评估，web 来源的许可不做完整清算；⑥ 下游评测仅限单轮 LLM 与算法/数据任务，未覆盖多轮代理或真实操作环境。

---

## 422. Does Bielik Know What It Doesn't Know? Activation Dispersion Separates Entity Familiarity from Factual Reliability Across Model Scale

**arXiv ID:** 2607.07670 | [PDF](https://arxiv.org/pdf/2607.07670v1)

**作者:** Grzegorz Brzezinka `[一作]` `[通讯]` (Prosit AS), Grzegorz Brzezinka (Prosit AS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不同规模的Polish Bielik LLM中，通过单次前向传播计算隐藏层的逆参与度与光谱熵，检测模型是否“熟悉”输入实体，并与模型的答案质量、抽样熵等进行对比。

**💡 创新点**

提出了无需标签、单前向传播即可读取实体熟悉度的卷积分布度量（IPR 与光谱熵），证明其在 1.5B-11B 规模下几乎达到完美识别的门槛，并揭示了“熟悉度”与“事实准确性”在不同尺度下的分离。

**🔧 技术方法**

使用逆参与度（Inverse Participation Ratio）、光谱熵（Spectral Entropy）、线性探针（logistic regression）、Permutation Test、Semantic Entropy（多样本熵）等技术，对激活分布进行定量分析。

**📊 数据集**

构建了包含 4 个实体域（运动员、城市、作家、音乐人）的三层次（名人、现实但鲜为人知、虚构）数据集，每个域 42 条实例，生成 504 条提示，覆盖 1.5B、4.5B、7B、11B 四个模型。

**📈 对比分析**

与监督线性探针、随机置换阈值、词形计数基线、Semantic Entropy 以及行为正确率（严格与软规则）对照。结果显示：IPR/光谱熵在提示点的 AUROC 0.94–0.98，几乎达到监督探针（0.99–1.00）的上限；行为准确率随规模显著提升（从 0/42 到 19/42），但与激活分布无显著相关；Semantic Entropy 在多样本判定上优于激活分布，但在熟悉度判定上劣于激活分布。

**⚠️ 局限性**

主要局限包括：①词形稀有度和长度匹配带来的词汇层面混淆；②样本量有限，某些行为/准确性格点受样本不足影响；③不同提示模板对城市域产生影响；④研究仅针对 Polish Bielik 系列，跨语言/跨模型验证不足；⑤拒绝率评估依赖 LLM 判别，可能受提示与模型调优的影响。

---

## 423. Guidance Breaks the Fitted Operator: A Terminal-Fitted Repair for Classifier-Free Guidance

**arXiv ID:** 2607.07665 | [PDF](https://arxiv.org/pdf/2607.07665v1)

**作者:** Shiheng Zhang `[一作]` `[通讯]` (University of Washington), Shiheng Zhang (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

作者针对扩散模型中的Classifier‑Free Guidance（CFG）在高指导尺度下的过度饱和与不稳定问题，提出了一个一系数修复——拟合CFG（fitted CFG），通过对终端层的精确匹配来消除CFG的发散残差并提升稳定性。

**💡 创新点**

创新点在于推导出终端层精确的修正系数 r^{1+w}−r（与传统 CFG 的 w(r−1) 不同），该系数仅通过对指导指数的单参数调整实现，保持了无额外网络评估成本，且在理论上证明了其在高 w 下的渐进保持与第一阶准确性。

**🔧 技术方法**

使用了渐进保持（asymptotic‑preserving）分析、拟合算子（fitted‑operator）理论、解析模型的导向指数分析，以及对高维模型的数值实验验证。

**📊 数据集**

主要在 NVIDIA edm CIFAR‑10 的 VP 检查点上进行实验，并将修复应用于 Stable Diffusion 1.5 的 DDIM 采样以验证跨域可行性。

**📈 对比分析**

与普通 CFG、有限区间指导（limited‑interval guidance）以及其他 CFG 修复方法进行对比；在 9/9 细胞网格中获得了 100% 的 FID 提升，并在残差放大、像素饱和等诊断指标上显著优于 CFG；然而在 KID 指标上仍略逊于原始 CFG。

**⚠️ 局限性**

限制包括：无法在所有指标上实现统一图像质量提升（KID 仍优于 CFG）、未能在全模型层面证明一致性、仅在满足类子集假设的条件下适用、且修复并非对所有高指导尺度的通用最优解。

---

## 424. ALER-TI: Aligned Latent Embedding Retrieval for Time Series Imputation

**arXiv ID:** 2607.07640 | [PDF](https://arxiv.org/pdf/2607.07640v1)

**作者:** Xuan-Thong Truong `[一作]` (Hanoi University of Science and Technology), Nhat-Hai Nguyen `[通讯]` (Hanoi University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出检索增强的时间序列缺失填补框架，通过对历史序列进行嵌入对齐检索，提升缺失值重建效果。

**💡 创新点**

引入Latent Embedding Alignment机制，在不重新编码历史候选的情况下，通过后置掩码实现缺失查询与完整候选的对齐；同时设计轻量化适配器实现模型无关。

**🔧 技术方法**

采用变压器编码器、对比学习（InfoNCE）、可逆实例归一化（RevIN）、多视图查询编码、轻量化融合门控和MSE损失等技术。

**📊 数据集**

使用六大真实世界时序数据集：ETT（ETTh1/2/ETTm1/2）、Electricity、Weather。

**📈 对比分析**

在七种不同后端（CNN、Transformer、MLP）和三种专门缺失填补模型上加入该框架，使用MSE/MAE评估；相比原始模型平均提升约10%–15% MSE，并在不同缺失率和序列长度下保持稳健。

**⚠️ 局限性**

对极大规模历史库的检索效率仍有提升空间；主要针对随机缺失，尚未验证块缺失或传感器失效等更复杂缺失模式；检索库的时间漂移需进一步处理。

---

## 425. Neural Operator-enabled Topology-informed Evolutionary Strategy for PDE-Constrained Optimization

**arXiv ID:** 2607.07682 | [PDF](https://arxiv.org/pdf/2607.07682v1)

**作者:** Xiangming Huang `[一作]`, Raphaël Pestourie `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为NOTES（Neural Operator-enabled Topology-informed Evolutionary Strategy）的框架，用于在PDE约束下的逆向设计，通过将DeepONet作为低维潜在空间的非线性解码器，并结合CMA‑ES进行全局进化优化，实现高效、可迁移的拓扑设计。

**💡 创新点**

创新点包括：①将物理信息嵌入训练数据而非损失函数，利用DeepONet学习可迁移的拓扑先验；②在低维潜在空间中完成进化优化，显著缓解维度灾难；③在潜在空间中直接实现二值化约束，保证生成设计的物理可行性；④展示了该框架在不同物理问题（光学、结构）和不同规模下的通用性和高性能。

**🔧 技术方法**

使用的技术包括：DeepONet（非线性神经算子）、PCA（降维与潜在空间构造）、CMA‑ES（无梯度全局进化优化）、Meent RCWA（光学PDE求解器）、FEM（结构PDE求解器）、L‑BFGS（梯度优化基线）、GLOnet（基准生成模型）以及PyTorch/DeepXDE/pycma等实现工具。

**📊 数据集**

采用的公开数据集：MetaNet（高效光学元件数据集）用于光学反射器；通过密度梯度优化生成的结构样本（MBB梁）用于结构优化；训练数据均为已获得的高性能PDE约束解，且仅保留效率≥0.9的样本。

**📈 对比分析**

与基准方法（传统拓扑优化、GLOnet、直接CMA‑ES、L‑BFGS+DeepONet、仅PCA解码）进行对比。结果显示：在光学任务中，NOTES平均效率>95%，方差最小；在结构任务中，合规性下降至246（比L‑BFGS低1.6%）。NOTES在所有设置下实现了最快收敛（相对CMA‑ES快≈10×）且能产生在训练集之外的更优设计，且设计二值化率高达97%。

**⚠️ 局限性**

局限性包括：①在不同PDE类别（如电磁与弹性）间的迁移性尚未完全验证；②梯度优化与硬二值化约束结合时易陷入局部最优，需进一步改进；③PCA降维对极其高维、复杂拓扑的表示可能不足，需探索更强的自编码器或图神经网络潜在空间；④对大规模网格或多物理耦合问题的可扩展性仍有待验证。

---

## 426. Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence

**arXiv ID:** 2607.07675 | [PDF](https://arxiv.org/pdf/2607.07675v1)

**作者:** Shuailei Ma `[一作]`, Ka Leong Cheng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向具身智能的DiT‑MoE视频预训练框架，包含稀疏Mixture‑of‑Experts单流扩散变压器、机器人增强的数据采集与预处理引擎，以及多维奖励的RL后训练，构建了可扩展、物理一致且高质量的视频基础模型。

**💡 创新点**

创新点包括：① 将稀疏MoE应用于大规模视频扩散网络，实现容量‑计算解耦；② 设计数据配置引擎与世界知识拓扑图，系统化地注入机器人操控、导航等具身先验；③ 结合多维奖励（视觉质量、文本‑视频一致性、动态程度、运动连贯性、人类运动一致性、物理可行性）进行GRPO强化学习，显著提升物理逼真度和任务完成度；④ 采用级联细化器提升高分辨率细节；⑤ 公开首个大规模、开源的Mixture‑of‑Experts视频基础模型。

**🔧 技术方法**

核心技术包括：DiT‑based扩散变压器、稀疏Mixture‑of‑Experts（DeepSeekMoE风格）、多模态3D RoPE、QK‑Norm、AdaLN‑Single Modulation；数据处理使用Vision‑Language模型 + 专用检测器构建多维记录；训练采用多阶段渐进式预训练、在线负载平衡、序列并行与专家并行；RL后训练使用GRPO与DiffusionNFT风格的负向学习；推理基于Diffusers兼容包与SGLang加速框架。

**📊 数据集**

数据集来源：大规模互联网视频（包括图文对齐视频）、机器人操控视频（真实、仿真、第三方视角）、导航与第一人称视角视频、文本富集视频，经过Data Profiling Engine、World‑Knowledge Topological Graph、Dense Structured Captioning、Caption Rewriter等步骤整理成统一格式；最终训练集分为5阶段，包含从192p图像到1080p高质量细化集；使用的公开基准为RBench和Physics‑IQ Verified。

**📈 对比分析**

通过内部基准和公开基准对比：在TI2V任务上在一般质量和具身领域均位居所有开源模型之首，T2V任务中排名第二般质量，仍在具身领域超越Cosmos等竞争模型；在RBench与Physics‑IQ Verified中均取得最高或次高分，验证了模型在机器人交互与物理实验中的可行性；实验显示模型在长序列、长视频推理中保持较高的MFU，推理速度与Dense模型相当或更快。

**⚠️ 局限性**

限制包括：① 训练成本高（需要多阶段大规模分布式训练）；② 尽管RL后训练提升物理一致性，但对极端复杂动力学（如流体、热传导）仍有限；③ 数据偏倚问题仍存在，尤其是机器人特定场景的覆盖不足；④ 目前的RL后训练仍依赖多维奖励模型，可能引入奖励不一致或过拟合风险；⑤ 部分细化与重参数化技术在超大分辨率或极长视频中仍有内存瓶颈。

---

## 427. RL Post-Training Builds Compositional Reasoning Strategies

**arXiv ID:** 2607.07646 | [PDF](https://arxiv.org/pdf/2607.07646v1)

**作者:** Azwar Abdulsalam `[一作]` (University College London), Andrew Saxe `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了强化学习在语言模型后训练中是否能通过组合基本技能形成更高级策略，而不是仅放大已有行为，并在可观察的改写语法环境中进行验证。

**💡 创新点**

证明了RL通过阶段性过程重组基本还原能力，发现并巩固宏观与并行组合操作，实现在有限生成预算下突破基础模型的能力边界。

**🔧 技术方法**

使用Transformer预训练于原语改写链，随后采用Group Relative Policy Optimization (GRPO) 进行强化学习，比较拒绝式微调 (RFT) 作为基准。

**📊 数据集**

构造的完全可观测改写语法数据集，包含手工生成的原语扩展与收缩序列。

**📈 对比分析**

与RFT对比发现RL在高难度桶上的 Pass@16 显著优于 RFT，且在更大采样预算下基础模型仍无法解决，而RL可在 16 次采样内完成。

**⚠️ 局限性**

局限性在于实验仅在人工构造的简化语法环境中验证，难以直接推广至自然语言推理任务且缺乏对模型规模和多样性的评估。

---

## 428. ATLAS: Automated HLS for DL-Optimized FPGAs

**arXiv ID:** 2607.07643 | [PDF](https://arxiv.org/pdf/2607.07643v1)

**作者:** Ruthwik Reddy Sunketa `[一作]` (Arizona State University), Aman Arora `[通讯]` (Arizona State University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了从Keras模型到FPGA硬件的全自动编译流程，自动将深度学习层映射到定制的Tensor Slice硬块上，无需手写RTL。

**💡 创新点**

创新点在于将GEMM作为通用抽象层，将前端hls4ml-GEMM与后端GEMM IP生成器结合，自动生成适配硬块的包装器、调度信息和控制FSM，彻底消除手工硬块实例化的需求。

**🔧 技术方法**

采用HLS（Catapult）、blackbox机制、GEMM抽象、动态权重预包装、流/并行接口转换、FPGA硬块IP生成器等技术；并使用Tensor Slices作为硬件加速器。

**📊 数据集**

使用合成的层级微基准（FC、Conv、Attention）和完整网络（MLP、CNN、Transformer）进行评估，并在这些模型上生成对应的硬件实现。

**📈 对比分析**

与基准软逻辑hls4ml和手写RTL进行对比；层级实验中compute‑area效率约为RTL的89%，相较hls4ml提升24%；完整网络实验中效率约为RTL的63%，比hls4ml高42%；设计时间从几天降至数小时。

**⚠️ 局限性**

仅支持单路编译无反馈，无法在硬件层面进行全局调度优化；对精度不兼容的情况处理有限；未实现跨M、N、K的重用因子调度。

---

## 429. The Key to Going Linear: Analysis-Driven Transformer Linearization

**arXiv ID:** 2607.07706 | [PDF](https://arxiv.org/pdf/2607.07706v1)

**作者:** Anna Kuzina `[一作]` (Qualcomm AI Research), Babak Ehteshami Bejnordi `[通讯]` (Qualcomm AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在冻结原始Transformer backbone的前提下，将其因果自注意力替换为线性时间机制，系统比较kernelized、state‑based和delta‑style线性注意力，并通过加入滑动窗口注意力（SWA）、sink token、短卷积/LoRA等结构干预提升性能。

**💡 创新点**

① 从一阶Taylor近似推导softmax到delta规则的关系，证明delta‑style更新天然捕获key‑dependent rank‑1修正；② 提出key‑gated linear attention (kGLA) 作为更贴合softmax的中间方案；③ 在严格冻结backbone的设置下逐步放宽约束，量化每项干预对性能的贡献，最终实现与全注意力相当的推理速度与准确率。

**🔧 技术方法**

采用线性注意力机制（Hedgehog、GLA、GDN、kGLA），一阶近似分析，滑动窗口注意力、sink token、短卷积、LoRA等结构干预；在LLaMA3.1‑8B、Qwen3‑8B‑Base等模型上训练；使用LM‑Eval、MMLU、Lambada、Common Reasoning、S‑NIAH、RULER等基准进行评估。

**📊 数据集**

训练使用DCLM‑Edu（10M token）数据集；评估使用零样本/5‑shot MMLU、Lambada、Common Reasoning（PIQA、ARC‑e、ARC‑c、HellaSwag、WinoGrande）、长上下文基准S‑NIAH、RULER。

**📈 对比分析**

通过与原始全注意力模型以及LoLCaT、Liger‑GLA、Llamba、Lizard、LoLA、STILL等先前post‑hoc线性化方法在相同缓存预算（64/128 token）下对比。结果显示，GDN+SWA+sink+短卷积在5‑shot MMLU、Common Reasoning平均分和Lambada上与全注意力差距≤1%，在大模型（32B）和不同缓存预算下保持一致扩展性；混合层实验证明保留少量全注意力层可进一步提升上下文相关任务。

**⚠️ 局限性**

一阶理论假设在所有层或极短前缀可能失效；未实现自适应缓存策略，仅在冻结backbone的设置下验证，仍需探索与完整预训练/更大训练预算的兼容性。

---

## 430. Institutional Red-Teaming: Deployment Rules, Not Just Models, Causally Shape Multi-Agent AI Safety

**arXiv ID:** 2607.07695 | [PDF](https://arxiv.org/pdf/2607.07695v1)

**作者:** Yujiao Chen `[一作]` `[通讯]` (Massachusetts Institute of Technology), Yujiao Chen (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入机构红队评估方法，固定多智能体和任务，仅变更部署规则，量化规则对集体安全的因果影响。

**💡 创新点**

提出“机构红队”概念和IABench‑CA基准，展示规则安全性无普遍安全默认，身份显著性是导致目标化的核心机制。

**🔧 技术方法**

利用大语言模型（如 GPT‑5.1、Gemini、Claude 等）作为主体，构建五种后果分配规则的实验，并使用机构一致性间隙（IAG）指标。

**📊 数据集**

在 228 个资源–阈值组合的三智能体阈值游戏中，测试七类 LLM 模型，共计 33,924 局游戏。

**📈 对比分析**

通过与规范参考模型对比，计算 IAG 并评估故障率，发现规则变更可导致 22–58 个百分点的致命率差异，身份显著性干预能将针对性消除从 81% 降至 22%。

**⚠️ 局限性**

局限在于实验设置过于简化（仅三智能体、无通信、单一阈值游戏）且仅覆盖七个 LLM 快照，未来版本需在更复杂环境和更广泛模型上验证。

---

## 431. How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization

**arXiv ID:** 2607.07678 | [PDF](https://arxiv.org/pdf/2607.07678v1)

**作者:** Xinyi Wu `[一作]` (MIT), Ali Jadbabaie `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文通过理论分析和实验验证，研究了 RoPE（Rotary Position Embedding）在 Transformer 中的频率使用规律，并将其与训练数据的相对距离依赖结构联系起来，进一步解释了位置插值（PI）在长度泛化中的作用。

**💡 创新点**

创新点在于：①提出了“频率匹配原理”，证明在给定依赖宽度 W 的情况下，最优 RoPE 频率应为 θ* ≈ 1/W；②构建了“位置依赖核”与 RoPE 频率的场-分辨率权衡框架；③将该框架用于解释 PI 的有效性，证明其在自相似（self‑similar）依赖结构下能实现长度泛化，且在非自相似结构下会失效。

**🔧 技术方法**

使用了能量度量（frequency energy spectrum）、位置依赖核（dependency kernel）、场-分辨率分析、归一化依赖宽度、PI 频率缩放、长短期相互信息计算等技术，并在实验中采用多层 GPT、Qwen-2.5-1.5B、Llama‑2‑7B 等模型。

**📊 数据集**

实验数据集包括：synthetic block‑drift 任务（控制块长度 B），iGSM 生成的算数题（操作数 ops）、alpaca、gsm8k、Nemotron‑ClimbMix（自然语言），以及通过不同 BPE 词表大小得到的多尺度 tokenization 数据。

**📈 对比分析**

对比方法主要是：①对比不同块长度、算术操作数下的 RoPE 频率能量分布；②将 PI 与不使用 PI 的模型在更长上下文上的准确率、perplexity 进行对比；实验显示：在自然语言中 PI 能维持低 perplexity 并提升长上下文性能；在算术任务中，PI 只略微提升准确率，却导致 perplexity 明显上升。

**⚠️ 局限性**

局限性：①理论假设依赖核为紧支撑，无法完全覆盖自然语言的长尾依赖；②PI 的效果高度依赖数据的自相似性，对需要精细位置解析的任务效果有限；③实验仅针对部分任务和模型，缺乏对更广泛场景的验证；④频率匹配原理依赖于估计的依赖宽度，实际应用中可能难以准确测量。

---

## 432. Answering Without Referring: How AI Search Rewrites the Web's Economic Bargain

**arXiv ID:** 2607.07652 | [PDF](https://arxiv.org/pdf/2607.07652v1)

**作者:** Qiaoni Shi `[一作]` (Bocconi University), Kai Gu `[通讯]` (Bocconi University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过分析Comscore点击流，量化AI搜索（ChatGPT）对传统搜索流量的影响，发现ChatGPT仅在5.2%的会话中产生外部点击，且其残留流量主要集中在参考、学术和工具类网站。

**💡 创新点**

创新点在于从用户侧视角将AI搜索与传统搜索的路由差异系统化，结合访问权限扩展的差分中断设计，首次估计AI搜索导致的传统搜索流量替代效应。

**🔧 技术方法**

方法包括构建对话会话定义、清洁点击、域分类、差分中断和事件研究，利用对齐的家庭周固定效应控制内在异质性。

**📊 数据集**

使用的数据集是2024年10月至2025年7月的美国桌面Comscore点击流（约168k-238k户），包含前台页面加载、HTTP引用、搜索查询和ChatGPT会话。

**📈 对比分析**

通过在相同家庭周内对ChatGPT会话和Google查询进行对比，并利用三次访问权限扩展的合并差分中断，结果显示AI搜索对传统搜索查询下降约9.4%，在持续二十周后约17%，且信息类网站流失最大。

**⚠️ 局限性**

局限性在于仅覆盖桌面用户、缺乏消费者福利和发布商收入数据，且对提示内容、搜索需求选择的测度仅依赖周边浏览上下文，未能直接观察实际任务完成度。

---

## 433. Unlearning to Protect: A Distilled Reinforcement Learning Framework with Privacy-Preserving Feature Unlearning and XAI for IoT Security

**arXiv ID:** 2607.07635 | [PDF](https://arxiv.org/pdf/2607.07635v1)

**作者:** Md. Nahid Hasan `[一作]` (BRAC University), Golam Rabiul Alam `[通讯]` (BRAC University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出轻量化强化学习框架DiRLU，用知识蒸馏从教师模型压缩到学生模型，并通过后置权重修改实现特征遗忘与恢复，支持XAI解释；

**💡 创新点**

首次结合强化学习、知识蒸馏与特征遗忘于IoT流量检测，既保证高精度，又满足GDPR隐私要求；

**🔧 技术方法**

采用A2C、Q‑learning、Meta‑RL的Actor‑Critic网络、知识蒸馏、后置权重修改(Feature Unlearning)与LIME可解释；

**📊 数据集**

使用Bot‑IoT数据集25%样本（约1.5亿行），通过SMOTE平衡后训练；

**📈 对比分析**

与传统模型（LightGBM、LSTM、CNN等）和前沿模型（KronNet、DL‑BiLSTM等）对比，DiRLU学生模型在Accuracy 99.60%、F1 99.80%、FLOPS 2,370（比KronNet低3.87×）；

**⚠️ 局限性**

仅在Bot‑IoT数据集上验证，泛化至其他攻击场景、对抗样本鲁棒性以及实时动态特征遗忘机制仍待提升；

---

## 434. Accurate, Interdisciplinary and Transparent Structure-property Understanding with Deep Native Structural Reasoning

**arXiv ID:** 2607.07708 | [PDF](https://arxiv.org/pdf/2607.07708v1)

**作者:** Chen Tang `[一作]` (Shanghai Artificial Intelligence Laboratory), Lei Bai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `09944146-298c-433e-89df-37255de463d7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了SciReasoner，一种面向蛋白、分子和晶体等多种科学模态的多模态基础模型，能够将三维结构信息离散化为结构感知词表，并在自回归生成过程中以可检索的结构单元构建可解释的链式推理轨迹，从而实现准确预测和结构驱动的解释。

**💡 创新点**

创新点主要包括：① 离线结构编码器（Foldseek、ConfSeq、SLICES）将空间、拓扑和周期连通性统一转化为可直接使用的离散令牌；② 在语言模型内部嵌入结构感知词表，使结构单元成为可检索、可引用的证据；③ 自启动的结构推理训练框架（intra‑domain grounding + cross‑domain consolidation），利用自监督+强化学习使模型在多领域共享统一的结构推理范式；④ 通过双盲专家评估验证推理轨迹的可解释性和可靠性。

**🔧 技术方法**

技术与方法：基于Qwen3‑14B大语言模型；离线结构编码器（Foldseek、ConfSeq、SLICES）；结构感知词表与离散嵌入；三阶段预训练（warm‑up、full‑parameter、annealing）+自回归生成；链式推理（CoT）生成；自监督+强化学习的post‑training（intra‑domain grounding + cross‑domain consolidation）；结构可检索的推理轨迹。

**📊 数据集**

使用的数据集包括：蛋白质——UniProt、AlphaFoldDB、PDB；小分子——ChEMBL、BindingDB、Open Reaction Database、USPTO、ChemRxiv等；晶体——Materials Project、JARVIS、OQMD、QMOF、hMOF、SNUMAT等；DNA/RNA——RNAcentral、NCBI；通用文本——Common Crawl、Dolci‑Think、SciIF；以及多种公开的基准数据集（DeepFRI‑GO、Retrosynthesis USPTO‑50K、DUD‑E、材料性质等）。

**📈 对比分析**

评估方法：与四大前沿通用LLM（Opus‑4.7、GPT‑5.5、Kimi‑K2.6、DeepSeek‑V4‑Pro）以及领域专用模型在多项基准（GO注释、单步反向合成、虚拟筛选、材料属性预测、生成/设计任务等）进行直接对比；在低同源蛋白、单步合成等弱相关性任务中，SciReasoner分别提升了0.13‑0.21的Fmax、0.09的单步合成准确率、以及在晶体任务中对高/低能带隙的区分度。双盲专家评估显示，SciReasoner的推理轨迹在98%对比案例中被专家视为优于或等同于现有大模型，且在所有五个质量维度（证据可检索性、领域合理性、任务对齐、推理连贯性、抗幻觉）上均显著高于基线。

**⚠️ 局限性**

局限性：① 依赖高质量的结构信息，若结构预测误差大或缺失，推理效果下降；② 结构离线编码与模型的交互仍受序列长度限制，难以一次性处理极大分子或晶体；③ 生成推理轨迹耗时较长，实际部署效率待提升；④ 目前仅覆盖蛋白、分子和晶体三大模态，尚未扩展到更广泛的材料或细胞级结构；⑤ 公开的代码与训练细节有限，复现与进一步改进受限。

---

## 435. Selective Timestep Weighting and Advantage-Based Replay for Sample-Efficient Diffusion RLHF

**arXiv ID:** 2607.07693 | [PDF](https://arxiv.org/pdf/2607.07693v1)

**作者:** Eric Zhu `[一作]` (Carnegie Mellon University), Soumik Mukhopadhyay `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了两种策略，分别是对扩散过程中的每个时间步进行权重重分配以及对高优势轨迹进行硬挖掘回放，以提升Diffusion RLHF的反馈样本效率

**💡 创新点**

创新点在于：①将奖励信息在时间步和轨迹上的不均匀性转化为可操作的时间步权重；②基于优势大小对历史轨迹进行优先回放，弥补传统RLHF中只使用当前轨迹的缺陷；③提出了一种易于实现、可插拔的权重与回放框架，兼容现有Diffusion RLHF方法

**🔧 技术方法**

技术包括：基于PPO与GRPO理论推导的时间步权重近似；使用均方潜在变化、标准差等自适应权重；实现优先回放缓冲区，按绝对优势值选取轨迹；在Stable Diffusion v1.5上通过LoRA微调；使用DDPO、DPOK、B2‑DiffuRL等基线进行实验

**📊 数据集**

数据集：使用动物图像训练提示集（与原DDPO论文相同），并通过ChatGPT生成未见动物提示用于测试泛化；reward函数包含JPEG压缩性、JPEG不可压缩性、美学分数、HPS v2与Image Reward共五个，分别通过相应的评估器产生标注

**📈 对比分析**

对比方法：在相同reward查询量下，将改进后的DDPO、DPOK、B2‑DiffuRL与原版做对比；实验显示在4k reward查询下，改进方案平均可获得2–6倍的样本效率提升，同时在CLIP分数与泛化评分上保持与基线相当或略优

**⚠️ 局限性**

局限性：①权重与回放策略仍依赖经验启发式，缺乏通用理论保证；②对高优势轨迹的回放可能导致样本分布漂移，需控制缓冲区大小与时间跨度；③在极少reward查询的极端稀缺场景下效果尚未验证；④对不同扩散模型或更复杂任务的适用性尚需进一步研究

---

## 436. Agon: Competitive Cross-Model RL with Implicit Rival Grading of Reasoning

**arXiv ID:** 2607.07690 | [PDF](https://arxiv.org/pdf/2607.07690v1)

**作者:** Vladislav Beliaev `[一作]` `[通讯]` (Independent Researcher), Vladislav Beliaev (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练两模型在互相“读稿、竞争”的循环中共同提升推理质量，避免仅靠答案奖励导致的“过度思考”。

**💡 创新点**

引入竞争式评判：每轮一模型作“草稿”，另一模型阅读草稿后独立求解并以胜过对方为奖励，从而隐式对思考过程进行评分；通过角色轮换和双适配器实现高效对抗训练。

**🔧 技术方法**

使用GRPO（基于答案奖励的on‑policy RL），双适配器LoRA架构在单一冻结的基础模型上实现两模型；采用草稿‑挑战（draft‑and‑challenge）rollout、竞争奖励（conversion bonus）以及长度调节等技术。

**📊 数据集**

在DeepMath（Hard 103K）数学推理数据集上训练；在GSM8K、MATH‑500等通用数学数据集以及CodeContests（易难度）编程验证集上做跨域检验。

**📈 对比分析**

与零射击、单模型GRPO、Self‑Refinement、未训练Mixture‑of‑Agents以及两步GRPO自迭代等基线在相同生成预算下对比；竞争+交换方案在DeepMath Hard上pass@1从30%提升至61%（≈+31pp），在GSM8K/MATH‑500上亦保持优势；在CodeContests上亦表现提升。

**⚠️ 局限性**

依赖可验证的答案奖励；对抗模型需保持匹配强度与行为差异，过大差距会导致蒸馏，过小则失去对抗效果；目前只实现文本交换，延迟为两步生成；未充分量化模型间互补性；对噪声更大或非程序化验证域的适用性待验证。

---

## 437. ECGLight: Compute-Light Framework For Paper ECG Digitization and Myocardial Infarction Screening

**arXiv ID:** 2607.07683 | [PDF](https://arxiv.org/pdf/2607.07683v1)

**作者:** Shreyasvi Natraj `[一作]` (ETH Zürich), Diego Paez-Granados `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一套完整的端到端轻量化管线，能够将纸质 ECG 的照片或扫描图像通过分割、导联识别、参考脉冲标定等步骤，恢复出 12 轴数字波形，并直接用于心肌梗死（MI）及闭塞性心肌梗死（OMI）的诊断。

**💡 创新点**

创新点包括：1）使用 Patch‑based YOLOv8 实例分割实现高分辨率纸 ECG 的精细提取；2）采用参考脉冲而非网格线进行物理标定，提升在低对比、扭曲或缺失网格的图像上的鲁棒性；3）整合导联与布局识别，完成完整的纸 ECG 到 12 轴波形的映射；4）在 CPU‑only 边缘设备上完成 <30 秒的完整推理；5）对诊断模型提供 SHAP 级联解释，支持临床可解释性。

**🔧 技术方法**

核心技术包括：Patch‑based YOLOv8 分割、基于 YOLOv8 的导联与参考脉冲检测、像素到物理单位的比例映射、轻量化时序分类模型（MLP、GRU、CNN、ResNet、InceptionTime）与核方法（Rocket、Arsenal），以及 SHAP 解释。

**📊 数据集**

主要数据集：PTB‑XL 21,799 条 12 轴 ECG 记录（通过 ECG‑Sheet 生成器合成对应纸 ECG 图像）用于训练；另外使用医院真实扫描 ECG 进行外部验证。评估任务包括 MI vs Normal、Pre‑ vs Post‑procedural MI 以及 OMI vs non‑OMI。

**📈 对比分析**

比较方法：在 CPU‑only 环境下，每条 ECG 的完整推理 <30 秒；Patch‑based 分割的 IoU 达 0.647、Dice 0.782；数字化误差 RMSE 0.043 mV，SNR 4.54 dB；在 MI 检测任务中准确率 95.51%（F1 0.952）；OMI 检测 88.89%（F1 0.886）。深度模型与 Rocket 等核模型对比，Rocket 在全序列上取得最高准确率 0.955，latency 304 ms；InceptionTime 在 MI vs Normal 任务上达到 0.929 的准确率。

**⚠️ 局限性**

限制与不足：1）合成纸 ECG 训练导致对极度模糊、严重划痕的实际扫描图像仍有约 12% 的失败率；2）评估聚焦于 MI/OMI 诊断，尚未验证对其他心律失常或结构性疾病的适用性；3）缺乏真正的配对数字信号数据，部分性能评估仅通过诊断准确率间接验证；4）解释方法主要基于 SHAP，可能无法完整揭示更复杂模型的内部决策。

---

## 438. MedPMC: A Systematic Framework for Scaling High-Fidelity Medical Multimodal Data for Foundation Models

**arXiv ID:** 2607.07673 | [PDF](https://arxiv.org/pdf/2607.07673v1)

**作者:** Hyunjae Kim `[一作]` (Yale University), Qingyu Chen `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了MedPMC框架，利用开放许可的PMC文献自动化生成高质量医用图文对，构建了约11M条图像-文本对，并基于此训练了MedPMC-CLIP等视觉语言模型，进一步评估其在多项医学基准、医学多模态问答和临床皮肤图像检索等任务上的表现。

**💡 创新点**

创新点在于：①设计了五阶段可插拔的自动化管道，准确拆分复合多面板图像并实现子图与子说明的精确对齐；②利用文本与视觉联合判别实现高效医学图像筛选；③构建了可持续更新的医学多模态数据基础设施，克服了现有公开数据缺乏质量与时效性的问题。

**🔧 技术方法**

采用了PubMedBERT+文本分类、Vision Transformer+YOLOv10+InternVL‑2.5‑4B等专用模型进行各阶段任务；使用CLIP架构进行对比学习；利用GPT‑4Turbo进行合成标注与子说明生成；将MedPMC数据用于CLIP预训练和LLaVA‑Med的视觉编码器替换。

**📊 数据集**

主要使用的原始数据来源为PMC开放许可文献（约6.1M篇）生成的11M条医学图文对；在组件训练和评测中引用ImageCLEF、MedICaT、DocFigure等公开基准；在最终评估阶段使用26项医学基准、MMMU、OmniMedVQA、Yale New Haven Health System（YNHHS）皮肤科患者图像等。

**📈 对比分析**

通过与BMC‑CLIP、BiomedCLIP、PMC‑CLIP等现有视觉语言模型在26项医学基准上进行对比，MedPMC‑CLIP平均AUC提升7.1pp；在OmniMedVQA上提升16.9pp；在MMMU上提升1.9pp；在临床皮肤图像检索中Recall@5提升11.7pp，表现显著优于现有文献‑derived 数据集和模型。

**⚠️ 局限性**

局限性包括：①仅覆盖图像与文本对，未包含表格、实验记录等其他多模态信息；②依赖文献说明，可能存在描述不完整、偏差或非临床代表性；③潜在的与公开基准数据重叠风险，未能完全排除；④模型在真实临床部署前仍需进一步适配、验证和安全评估。

---

## 439. DiaLLM: An Investigation into the Robustness-Generation Gap in English Dialect Adaptation

**arXiv ID:** 2607.07669 | [PDF](https://arxiv.org/pdf/2607.07669v1)

**作者:** Jordan Painter `[一作]` (University of Surrey), Lu Yin `[通讯]` (University of Surrey)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DiaLLM 框架，实现对三大开源 LLM（Llama 3.1‑8B、Qwen 3‑8B、Gemma 3‑4B‑it）进行连续预训练、监督微调和对齐的全流程改造，以提升对澳大利亚、印度和北部英国英语等方言的理解与生成；

**💡 创新点**

首次系统对连续预训练、监督微调、以及三种对齐方法（DPO、GRPO、GSPO）在两种后训练范式（隐式 vs 明式）下的效果进行控制比较，并揭示“鲁棒性‑生成”与“奖励‑质量”两种关键分离现象；

**🔧 技术方法**

使用连续预训练（GaLore）、多任务监督微调、直接偏好优化（DPO）、基于奖励的对齐（GRPO、GSPO），并训练多标签方言特征分类器以产生奖励；

**📊 数据集**

核心数据集为国际英语语料库（ICE，18种方言共约2千万标记）以及多方言偏好数据（使用 Multi‑VALUE 转换得到的澳大利亚、印度、北部英国英语样本）；

**📈 对比分析**

对齐方法在鲁棒性基准（BBH、GPQA、GLUE、VALUE、BST‑Sent/Sarc、DialectBench）上的表现基本相同；但在生成评估中，明确方言目标的“显式”范式与“隐式”范式相比，生成更易被人类与 LLM 判断为目标方言；GRPO 在奖励上最高，但并未得到最佳主观偏好；整体而言，Llama 3.1‑8B 在所有方法中均表现最佳；

**⚠️ 局限性**

限制包括：①评估主要聚焦 Llama 3.1‑8B，未完全验证其他模型；②方言特征奖励基于 eWAVE，易导致“表面特征”与真实感知不匹配；③ICE 语料偏向正式/半正式文本，缺乏社交媒体语境；④偏好评估样本数有限（仅 2 位评审），一致性低；⑤未将对齐与数据选择完全分离，难以单独归因。

---

## 440. Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops

**arXiv ID:** 2607.07663 | [PDF](https://arxiv.org/pdf/2607.07663v1)

**作者:** Mingguang Chen `[一作]` (University of California, Riverside), Bo Qu `[通讯]` (Illinois Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统性综述 1,250 篇 2024–2026 年的自我改进论文，构建了以“改进对象”（输出、策略、评估器、研究流程）和“闭环程度”（人类在环、人类在上、完全闭环）为轴的两维分类法，并提出了评估器验证层级；

**💡 创新点**

创新点在于（1）将混杂的“自-改进”术语统一为两维分类法，清晰区分有界自我改进与开放式递归自我改进；（2）引入评估器验证层级，将自评循环的可靠性与实验失效模式系统化；（3）通过对 1,250 篇论文的归类与量化，揭示研究热度、方法分布与安全风险的交叉规律；

**🔧 技术方法**

主要技术包括：大规模文献检索与元数据抓取（arXiv + OpenAlex）、基于关键词与规则的自动分类、人工校正、文本相似度聚类（TF‑IDF+SVD+t‑SNE）用于可视化、统计分析与趋势绘制；

**📊 数据集**

数据集为 2024–2026 年在 arXiv 上公开的 1,250 篇论文的完整元数据与文本摘要；

**📈 对比分析**

比较方法是将论文归入 4 个改进类别和 3 种闭环级别，统计每类占比、发表年份与引用情况，并绘制季度增长曲线；在评估器层面，作者对验证器可信度从“正式验证器 → 执行反馈 → 学习评估器 → 内在信号”进行层级化并评估其在不同类别中的表现，指出验证器越靠上层，系统改进效果越显著；

**⚠️ 局限性**

局限性包括：样本采样偏倚（以 arXiv 为主，可能忽略工业实验与未公开工作）；分类规则虽覆盖大部分论文但仍有误归与边界模糊；缺乏对各类循环在真实世界中的长期行为与安全风险的量化验证；未提供统一的性能基准，主要以定性观察与统计呈现。

---

## 441. Modeling Failure Dynamics in Time-Constrained Authentication Systems: Evidence of a Success Cliff in USSD Workflows

**arXiv ID:** 2607.07650 | [PDF](https://arxiv.org/pdf/2607.07650v1)

**作者:** Aklile Seyoum Mamo `[一作]` (Carnegie Mellon University Africa), Jema Ndibwile `[通讯]` (Carnegie Mellon University Africa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过仿真建模分析受时间限制的USSD身份验证流程的失败动力学，识别出所谓的成功陡坡（Success Cliff）现象。

**💡 创新点**

首次量化并定义了成功陡坡，将阻塞延迟与会话时限的交互作为非线性失败阈值。

**🔧 技术方法**

采用基于Keystroke-Level Model的交互时间估计、Gamma分布网络延迟模型和三种放弃模型的模拟框架。

**📊 数据集**

基于公开USSD时限参数、KLM标准以及SMS OTP交付时间的经验分布；无实地数据，仅使用文献与仿真参数。

**📈 对比分析**

在四种复杂度和三种网络时延下进行模拟，比较成功率、超时率、放弃率；发现阻塞延迟导致成功率急剧下降，约75%；没有阻塞时则仅线性下降。

**⚠️ 局限性**

仿真依赖KLM估算与文献参数，未涵盖真实用户行为、重试限制和SMS时延高变；结果可能低估实际失败率。

---

## 442. Co-LMLM: Continuous-Query Limited Memory Language Models

**arXiv ID:** 2607.07707 | [PDF](https://arxiv.org/pdf/2607.07707v1)

**作者:** Yair Feldman `[一作]` (Cornell University), Yoav Artzi `[通讯]` (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种连续查询的有限记忆语言模型（Co‑LMLM），通过在预训练中将事实外部化为向量键-文本值的知识库，并使用对比检索损失训练模型生成查询向量。

**💡 创新点**

将检索查询改为连续向量、使用自监督对比学习、构建大规模事实标注与问答生成管线，突破关系知识库的限制。

**🔧 技术方法**

Transformer 语言模型 + 密集检索、InfoNCE 对比损失、BIO 事实 span 标注器+问答生成器、密集向量索引。

**📊 数据集**

Wiki（约3B tokens）+ FineWeb‑Edu（90B tokens），构建240M–2.2B 条目知识库。

**📈 对比分析**

与标准 LLM、Relational LMLM 及 HF/SmolLM 等基线在 Perplexity、SimpleQA、FactScore、NLP 任务上对比，Co‑LMLM 在 360M 模型下达 10.5 PPL、21.7 SimpleQA 与 GPT‑4o‑mini 同级、比 Claude Sonnet 4.5 高，同时保持 NLU 与往级相当。

**⚠️ 局限性**

实验规模有限，主要聚焦 Wikipedia/教育文本，检索索引构建成本高，缺乏对多领域事实的系统评估，未验证更大模型或持续学习场景。

---

## 443. Exploiting Spanning Trees for Directed Acyclicity

**arXiv ID:** 2607.07705 | [PDF](https://arxiv.org/pdf/2607.07705v1)

**作者:** Sergei Khargeliia `[一作]` (ITMO University), Danil Sagunov `[通讯]` (Markov Lab)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究加权最大无环子图（Maximum Acyclic Subgraph，MAS）问题，提出了两种按最大生成树（Maximum Spanning Tree，MaxST）权重上界的参数化算法，分别适用于整数权重和有理权重。

**💡 创新点**

创新点在于：①证明了在有理权重下MAS问题仍可在k上取得FPT算法，突破了先前关于随机排序下界和Poljak–Turzík下界的限制；②提出了逆边、剩余利润和完美图结构等新概念，将MAS上界与有向环的结构紧密联系；③将这些结构化思想与流增广技术结合，得到整数权重下的 2^k^{1.5}· 运行时间，以及有理权重下的 n^{k^O(1)} 运行时间。

**🔧 技术方法**

主要技术包括：逆边与最大生成树的分类；剩余利润和路径覆盖的贪心/递归选择；利用完美图定理证明逆边交互图是完美图，进而在多重图/有理权重下求最大独立集；流增广技术用于求解带权反馈弧集子问题；以及构造压缩图来处理允许边和禁止路径的约束。

**📊 数据集**

该工作为纯理论研究，无实验数据集，全部通过算法分析与理论证明展示性能。

**📈 对比分析**

与随机排序下界（1/2）和Poljak–Turzík下界相比，本文的算法能在 MaxST 下界基础上进一步提升；在整数权重下的 FPT 运行时间为 2^{O(k^{1.5})}，在有理权重下的为 n^{O(k)}；同时证明在有理权重下 k=1 的情况为 NP‑hard，说明问题本身难度较高。

**⚠️ 局限性**

局限性：仅适用于权重≥1 的有理权重；对多重图、权重可小于1的情况未覆盖；对极大规模实例的多项式预处理和核化仍缺乏；当 k 较大时，n^{k^O(1)} 的指数仍不现实。

---

## 444. Breaking Database Lock-in: Agentic Regeneration of High Performance Storage Readers for Database Bypass

**arXiv ID:** 2607.07696 | [PDF](https://arxiv.org/pdf/2607.07696v1)

**作者:** Victor Giannakouris `[一作]` (Cornell University), Immanuel Trummer `[通讯]` (Cornell University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了Jailbreak系统，利用LLM自动合成数据库存储文件的读取器，绕过传统驱动。

**💡 创新点**

创新点在于把LLM用于数据库文件格式的逆向解析与代码合成，实现零拷贝的Arrow读取。

**🔧 技术方法**

使用大型语言模型（LLM）+多代理流水线，结合Apache Arrow C Data Interface。

**📊 数据集**

使用TPC‑H 1GB数据集。

**📈 对比分析**

通过与JDBC/ODBC wire‑protocol读取对比，在六种分析引擎上实现最高27×的吞吐量提升。

**⚠️ 局限性**

局限在于依赖文件格式文档/源码可获取，且在不同数据库版本和复杂事务场景下可能需要重生成。

---

## 445. Agent Delivery Engineering Predictive Reliability Framework

**arXiv ID:** 2607.07689 | [PDF](https://arxiv.org/pdf/2607.07689v1)

**作者:** Dexing Liu `[一作]` `[通讯]` (Shanghai Qijing Digital Technology Co., Ltd), Dexing Liu (Shanghai Qijing Digital Technology Co., Ltd)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建ADE Predictive Reliability Framework（ADE‑PRF），通过 TM（Trust Margin）指标对LLM多智能体系统进行实时健康监测，并基于历史 TM 序列进行 8 小时前瞻性可靠性预测；同时在生产环境中对 6 种 agent 配置进行大规模验证。

**💡 创新点**

① 将 20 个跨层运行时信号按五层结构聚合为单一 TM 分数，显著提升对隐式退化的敏感度；② 采用三模并行预测（Kalman、Exponential、Survival）与集成（Ensemble）实现前瞻性可靠性估计；③ 在生产环境中实现无标记“零语义侵入”监测，直接嵌入 ADE 插件生态，提供闭环主动修复功能。

**🔧 技术方法**

技术包括：多层信号聚合与动态加权、指数平滑与 Kalman 滤波、Survival 分析与贝叶斯推断、基于 TPM、CI、四层阈值的决策框架，以及与 ADE 安全插件（AOC、CS 等）的协同工作。

**📊 数据集**

使用 Hermes 多智能体平台的生产数据：380,227 条预测记录、280,579 条验证记录，覆盖 6 个 agent profile；以及 7 次沙箱实验注入五种退化模式（上下文污染、工具链延迟、模型漂移、并发争用、配置漂移）。

**📈 对比分析**

在 8 小时预测窗口内，Ensemble MAE 为 1.861，<10 点变动预测准确率 99.65%；Exponential 方法 MAE 1.228、方向精确率 76.8%。实验显示，TM 与实际系统状态高度相关，预测误差低于 1.9% 的 TM 范围，且在大规模生产环境中维持低误报。

**⚠️ 局限性**

局限性：① 依赖于已有 ADE 插件的信号，无法覆盖完全未知的退化途径；② 采用无语义解析的设计虽提升鲁棒性，但对复杂语义漂移仍缺乏直接评估；③ 假设各层信号独立，实际可能存在跨层耦合；④ 预测模型在极端突发故障时仍可能出现过度乐观或保守的偏差；⑤ 需要持续的手工标注或沙箱实验以维持模型校准。

---

## 446. Max Out GRPO Signal: Adaptive Trace Prefix Control for Hard Reasoning Problems

**arXiv ID:** 2607.07674 | [PDF](https://arxiv.org/pdf/2607.07674v1)

**作者:** Vladislav Beliaev `[一作]` `[通讯]` (thinkdense.ai), Vladislav Beliaev (thinkdense.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于闭环控制的前缀长度调节机制（GRPO-CL），在GRPO训练中动态调整为每个问题提供的参考前缀长度，使得每个问题的组内成功率始终逼近50%，从而最大化梯度信号并突破传统GRPO在“死亡区”无法学习的瓶颈。

**💡 创新点**

创新点在于将前缀长度视为可调节的连续难度拨杆，并通过全局闭环控制器结合样本难度偏移，实时跟踪并维持目标成功率，同时在训练后期将前缀逐步衰减至零，实现训练与推理无缝衔接。

**🔧 技术方法**

技术主要包括：组相对优势GRPO；前缀长度作为难度拨杆；全局闭环控制（secant/二分搜索）以维持组内成功率≈0.5；静态难度偏移函数；前缀长度衰减策略；梯度在前缀 token 上遮蔽；基于 LoRA 的高效训练。

**📊 数据集**

使用的数据集为 DeepMath-103K（难度 8 的数学推理题）以及 GSM8K、MATH-500、AIME、AMC 等公开数学评测集做验证。

**📈 对比分析**

与 vanilla GRPO、固定前缀 GRPO、PrefixRL、Prefix-RFT 等基线对比，GRPO-CL 在匹配训练 FLOPs 的条件下，在 Qwen3‑1.7B 上对 DeepMath held‑out 的 pass@1 提升 18.1%（相当于 3–4 倍的基线提升），在 AIME、AMC 以及 GSM8K 上也均有显著改进；同时生成长度约缩短 44%（4.4k vs 8.0k）。

**⚠️ 局限性**

限制包括：需要可靠的参考解或离线采样前缀；依赖近二值化的验证器；当前仅在数学推理领域验证，对噪声或不可验证任务效果未知；前缀长度的初始冷启动与难度估计为一次性预处理，耗费额外计算；偏移函数为手工调参，无法对每个样本动态调整。

---

## 447. PeTeR: Post-Training Robustification of Probabilistic Circuits

**arXiv ID:** 2607.07671 | [PDF](https://arxiv.org/pdf/2607.07671v1)

**作者:** Adrian Ciotinga `[一作]` (Arizona State University), YooJung Choi `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种无数据后训练的鲁棒化框架，使预训练的概率电路在Wasserstein球内对分布偏移具备鲁棒性。

**💡 创新点**

将Circuit-Wasserstein距离引入后训练，构造可微分的梯度上下行优化，实现数据自由的鲁棒化，省去从头训练。

**🔧 技术方法**

采用结构可分解的概率电路、Wasserstein距离与Circuit-Wasserstein距离、梯度上下行（梯度上升-下降）优化、对数似然等技术。

**📊 数据集**

使用七个二值高维数据集：3*NLTCS、3*MSNBC、3*Plants、3*Netflix、3*DNA、3*Movie、3*BBC。

**📈 对比分析**

与传统最大似然MLE-PC和鲁棒MLE RL-TPM比较，在未扰动、随机扰动及对抗扰动测试集上，后训练鲁棒化方法表现优于RL-TPM且与MLE-PC相近。

**⚠️ 局限性**

局限性：Circuit-Wasserstein距离仅为Wasserstein的上界，导致在高扰动级别时鲁棒性下降；目前仅适用于结构可分解的PC，扩展到一般可分解PC仍待研究。

---

## 448. An optimal control approach for neural network architecture adaptation with a posteriori error estimation

**arXiv ID:** 2607.07637 | [PDF](https://arxiv.org/pdf/2607.07637v1)

**作者:** C G Krishnanunni `[一作]` (University of Texas at Austin), Tan Bui-Thanh `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于后验误差估计的深度自适应神经网络架构调整方法，利用连续时间最优控制框架将网络训练转化为最优控制问题，进而推导可计算的误差上界；该误差上界按层划分，可指导在误差最大的区间插入新层；实验在小宽度网络上实现了性能提升。

**💡 创新点**

创新点在于：①将神经网络训练视为连续时间最优控制问题；②应用双加权残差（DWR）方法得到可计算的误差上界，并将误差按层分解；③利用该误差分解实现全局最优的层插入策略；④提出两级离散（粗层参数、细层状态/对偶求解）方案，兼顾理论严谨与实现效率。

**🔧 技术方法**

采用的技术包括：连续时间最优控制与有限元离散、双加权残差误差估计、两级时间离散（粗层权重/细层状态/对偶前向/后向Euler）、梯度下降训练、插值权重初始化以及验证集用于停止与再训练。

**📊 数据集**

使用了两类合成数据集：一是二维非线性函数回归（输入2维，输出1维），二是Navier‑Stokes逆问题（10个观测点，输出50维KL系数）。

**📈 对比分析**

与Net2Net、Forward Thinking、随机层插入等现有自适应策略进行比较。实验表明，在小宽度网络下，我方方法在测试集上的均方误差和相对误差均最小，尽管训练时间较长。

**⚠️ 局限性**

局限性包括：1）由于需要子离散参数K，计算成本显著高于传统方法；2）仅在全连接网络、宽度固定的小模型上验证；3）误差估计假设网络已达到局部最优，实际训练中可能不满足；4）尚未实现宽度与深度的联合自适应。

---

