# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-04 | 今日论文总数: 590

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Anycast Performance in Context

**arXiv ID:** 2606.04298 | [PDF](https://arxiv.org/pdf/2606.04298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 2. Pinpoint: Grounded Worldwide Image Geolocation via Cross-Source Retrieval and Reranking

**arXiv ID:** 2606.04133 | [PDF](https://arxiv.org/pdf/2606.04133v1)

**作者:** Nika Chuzhoy `[一作]` (Virtualitics), Sarthak S. Sahu `[通讯]` (Virtualitics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在全球图像定位任务中同时利用互联网照片与街景图像的检索-再排序框架Pinpoint

**💡 创新点**

创新点在于：1）跨源对齐的图像-GPS对比学习器，使得互联网照片与街景图像共存于同一嵌入空间；2）引入基于注意力的再排序器，结合候选图像、GPS和来自相邻源的支持标记；3）不依赖大型多模态语言模型，显著提升推理速度与可复现性

**🔧 技术方法**

技术包括：SigLIP 2视觉骨干+源特定适配器的对比学习；多尺度傅里叶特征构造GPS嵌入；三通道候选检索（图像、GPS、SigLIP）+支持token；自注意力Transformer再排序器；对比损失与成对边缘损失

**📊 数据集**

数据集：MP16-Pro（约410万Flickr照片），OpenStreetView-5M（约510万街景图像），评测基准为IM2GPS3k、YFCC4k与OSV-5M测试集

**📈 对比分析**

在所有距离阈值上均实现了state‑of‑the‑art：IM2GPS3k、YFCC4k的精度均超过GeoRanker、G3等基于MLLM的方法；OSV‑5M在距离阈值和行政区级精度上同样优于HierLoc与RFM等传统方法；推理延迟比MLLM大幅降低（≈0.1s/图）

**⚠️ 局限性**

局限性：需要对SigLIP 2进行一次性嵌入预计算并存储，检索索引固定，无法即时反映新建筑或视觉变化；索引覆盖范围受限于所用图像集，需定期更新以保持时效

---

## 3. Long-Term and Short-Term Transistor Aging in Deep Neural Networks: Impact and Mitigation

**arXiv ID:** 2606.04266 | [PDF](https://arxiv.org/pdf/2606.04266v1)

**作者:** Alireza Sarmadi `[一作]` (New York University), Farshad Khorrami `[通讯]` (New York University)

**通讯引用:** 5728 | [OpenAlex ID](https://openalex.org/A5082413942)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了晶体管老化（长短期）对硬件实现的深度神经网络（DNN）推理精度的影响，并提出了基于时钟频率缩放、特征噪声与梯度噪声的老化感知再训练方法；同时探讨了短期老化作为硬件木马检测的激励手段。

**💡 创新点**

首次将完整硬件实现的多层DNN的老化效应与多级激励结合，提出特征+梯度噪声训练方案，并将短期老化与过度时钟结合用于木马检测，显著提升检测精度。

**🔧 技术方法**

采用FinFET技术的标准单元库、静态时序分析（STA）、门级仿真、SDF注解、噪声注入训练、Adam优化器、自动编码器+一类SVM分类器。

**📊 数据集**

MNIST手写数字数据集用于DNN实验；Trust‑Hub上RSA、AES、PIC等加密加速器的仿真数据用于木马检测评估。

**📈 对比分析**

与未改进模型对比，特征噪声训练在0.8V/0.6V下可将老化导致的精度下降降低至<10%（原先>50%），且可在更高时钟周期下保持准确率；木马检测在14nm FinFET上达到了99%+的精度、召回率，误报率低于1%。

**⚠️ 局限性**

局限在于仅针对全连接DNN和有限的硬件架构进行验证，未考虑卷积网络、异构加速器以及更高技术节点；噪声注入的手工参数选择缺乏自动化；木马检测对不同激励方式的泛化能力仍待进一步评估。

---

## 4. Token Budgets: An Empirical Catalog of 63 LLM-Agent Budget-Overrun Incidents, with an Affine-Typed Rust Mitigation as a Case Study

**arXiv ID:** 2606.04056 | [PDF](https://arxiv.org/pdf/2606.04056v1)

**作者:** Sajjad Khan `[一作]` `[通讯]` (University of the West of England), Sajjad Khan (University of the West of England)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作先构建了包含 63 起已确认的 LLM‑agent 预算超支事件和 47 条结构性缺失/功能请求的实证目录；随后在 Rust 语言中实现了一套基于 affine 资源类型的预算管理库（Token Budgets），通过编译时借用检查器强制实现预算不可别名、不可双重消费、不可复用，并与保守预估的运行时计费结合，形成完整的预算控制机制。

**💡 创新点**

创新点包括：
- 规模化、双编码验证的预算超支事件目录，形成八类机制聚类并实现高可靠性标注（Cohen’s κ=0.837）；
- 在 Rust 中将预算视为 affine 值，实现编译时不可复制、不可双消费、不可复用的完整性保障；
- 将此 compile‑time 机制与现有的运行时、网络层预算策略对标，证明在多代理并发场景下编译时防御可避免运行时错误，并通过多组实验验证零超支与高覆盖率。

**🔧 技术方法**

使用技术包括：
- Rust 语言的借用检查器与 affine 资源类型（Move/linear/ownership 迁移）；
- trybuild、rustc 的错误代码验证 compile‑time 保障；
- 预留式计费（估算 + 预留安全边际）与 run‑time 校准；
- 实验框架：温度分层、并发子代理（forgetful‑operator）、跨模型/跨框架基准、模拟大规模请求；
- 文本挖掘、GitHub Issue 搜索、双人 IRR 统计；
- 代码集成至 Rig（约 40 行）和基准数据集。

**📊 数据集**

数据集：
- 110 条 GitHub issue（63 证实事件 + 47 结构性条目），来自 21 个 LLM‑agent 框架，覆盖 18 生态系统；
- 3,461 条 issue 用于基线对照（20 个框架）；
- 5,190 次 LLM 调用事件用于估算精度；
- 382 次实时 API 会话；
- 1,160 行 Rust 代码（Token Budgets 核心 + 估算器）。

**📈 对比分析**

比较方法与性能：
- 对比 LangGraph、CrewAI、AutoGen、AgentGuard、LiteLLM 等 5 种运行时预算机制及 Agent Contracts，使用相同的预留估算、预设预算；
- 采用温度 0.0–1.0 的 160 次独立运行、M2 计数器对比、忘记操作员实验；
- 结果显示 Token Budgets 在所有实验中零超支、零误拒；在多代理并发下编译时防止预算泄漏；单代理场景下与简单计数器无差别；
- 估算保守率平均 4–6 倍，平均单次预留延迟 939–1,749 ms；部署至 Rig 仅需 40 行代码。

**⚠️ 局限性**

局限性：
- 未提供二进制级别的 cap‑soundness 证明，依赖运行时估算假设（A1）和 provider 信任（A7、A8）；
- 只能在 Rust 可信边界内发挥作用，Python 等语言需 runtime 版；
- 目录采样偏向公开英文仓库，未覆盖闭源平台；
- 预留估算过度可能导致资金占用高；
- 对 reasoning‑token 隐式计费模型支持有限；
- 缺乏跨框架、跨模型的广泛验证与自动化部署路径。

---

## 5. Testing Neural Networks via Bayesian-Guided Exploration of Decision Landscapes

**arXiv ID:** 2606.04314 | [PDF](https://arxiv.org/pdf/2606.04314v1)

**作者:** Bin Duan `[一作]` (University of Queensland), Guowei Yang `[通讯]` (University of Queensland)

**通讯引用:** 1448 | [OpenAlex ID](https://openalex.org/A5039642499)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种白盒神经网络测试框架，通过定位决策关键区域并使用贝叶斯优化来生成多样化的失效案例。

**💡 创新点**

创新点在于将可解释性显著性图转换为局部可变区域，并利用不确定性感知的稀疏变分高斯过程（SVGP）在有限预算内高效探索决策不稳定性。

**🔧 技术方法**

采用 Grad‑CAM、Integrated Gradients、SmoothGrad 等显著性方法进行区域定位，结合 SVGP 近似高维 GP 与 Upper Confidence Bound（UCB）采样策略进行贝叶斯优化。

**📊 数据集**

在 MNIST、CIFAR‑10 和 ImageNet 三大视觉基准上，针对 LeNet‑4/5、VGG16/19、ResNet18/50 等六种常用网络进行实验。

**📈 对比分析**

与 ADAPT、NSGen、SUNTest 等白盒方法比较，在固定 10,000 次 mutation 预算下，取得更高的失败发现率（NoF、FSR）、更丰富的失败多样性（DoF）、更低的 FID、以及更好的语义一致性（SCS）；生成的失效案例在微调后还能提升模型准确率。

**⚠️ 局限性**

局限包括：依赖显著性图的可解释性稳定性，对非视觉或非卷积模型的适用性未知；仅关注失败多样性与分布一致性，对鲁棒性攻击的评估不充分；以及对显著性方法和超参数的选择较为敏感。

---

## 6. Recover-LoRA for Aggressive Quantization: Reclaiming Accuracy in 2-Bit Language Models via Low-Rank Adaptation with Knowledge Distillation on Synthetic Data

**arXiv ID:** 2606.04238 | [PDF](https://arxiv.org/pdf/2606.04238v1)

**作者:** Devleena Das `[一作]` (Advanced Micro Devices Inc), Ashish Sirasao `[通讯]` (Advanced Micro Devices Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究将大型语言模型中门控MLP的门与上投影层压缩至2-bit，并使用Recover‑LoRA在不需要标签数据的情况下恢复大部分准确率，随后将其余层压缩至4-bit实现高吞吐量部署。

**💡 创新点**

创新点在于将选择性混合精度量化（W4/W2‑GateUp）与轻量化数据无关的恢复方法Recover‑LoRA结合，证明仅用1万条合成样本即可对2-bit量化误差实现80–95%的准确率恢复。

**🔧 技术方法**

使用了post‑training量化、混合精度W4/W2 GateUp策略、低秩LoRA适配器、logit蒸馏、合成采样（Hybrid Sampling）以及LIFE屋顶线分析。

**📊 数据集**

评估数据集包括12个基准（9个常规任务、3个OOD）以及10k条合成样本或OpenHermes 10k标签数据；量化模型以Qwen3‑4B为主。

**📈 对比分析**

与统一4-bit量化对比，GateUp可提升7.5–23.3%的TPS；Recover‑LoRA在9/12基准上恢复80–95%准确率，合成数据与标签数据效果相当。

**⚠️ 局限性**

局限性：实验仅验证单一模型（Qwen3‑4B）和BF16/INT4后置量化；对数学推理等难题恢复有限；需要为每种量化配置单独训练LoRA；未验证全W4/W2‑GateUp与KV缓存量化结合的恢复效果。

---

## 7. LiftQuant: Continuous Bit-Width LLM via Dimensional Lifting and Projection

**arXiv ID:** 2606.04050 | [PDF](https://arxiv.org/pdf/2606.04050v1)

**作者:** Liulu He `[一作]` (Nanjing University), Li Du `[通讯]` (Nanjing University)

**通讯引用:** 11614 | [OpenAlex ID](https://openalex.org/A5062744012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种名为 LiftQuant 的 LLM 权重量化框架，能够实现连续可调的位宽量化，并在保持硬件友好的前提下实现高压缩率。

**💡 创新点**

创新点主要包括：
- lift-then-project 机制，将 1 位二进制 lattice 在高维空间投影到低维空间，解耦位宽与量化格子，实现连续位宽控制；
- 通过高维投影自动生成结构化且非均匀的代码库，兼具向量量化的表达力与均匀量化的硬件友好；
- 引入轻量级白化变换，将真实权重分布映射为近似 i.i.d. 高斯分布，以便高效投影；
- 通过可微分的整合解码矩阵实现端到端微调，进一步提升精度。

**🔧 技术方法**

采用的技术包括：
- 高维拉升投影矩阵 M 的优化（基于高斯近似与近似最近邻搜索）；
- 分层白化变换 T（由标量缩放和 Kronecker 乘积的正交矩阵组成）；
- 直通估计（STE）和温度化梯度近似的量化优化；
- 端到端微调（使用小规模校准集进行输出重建损失最小化）；
- 结合 torch.compile 与 BitBLAS 的 INT1–FP16 GEMV 加速。

**📊 数据集**

使用的数据集包括：
- 校准集：RedPajama（4096 段，序列长度 2048）；
- 评价集：WikiText‑2、C4（验证集）；
- 零样本基准：ARC‑c、ARC‑e、HellaSwag、PIQA、WinoGrande；
- 通用基准：MMLU。

**📈 对比分析**

与现有方法的对比：
- 在相同整数位宽（2、3、4 位）下，LiftQuant 的 PPL 通常低于 UQ 基线（GPTQ、Quarot、EfficientQAT）和 VQ 基线（QuIP#、AQLM、VPTQ、QTIP）；
- 通过细分为 2.4‑bit、2.5‑bit 等分数位宽，LiftQuant 在 70B Llama‑3 上实现 PPL 5.86，显著优于所有 2‑bit 基线；
- 在 Pareto 前沿上，LiftQuant 的性能-内存曲线几乎与假设的“理想 4‑bit”上限齐平；
- 通过 24/10（2.4‑bit）或 25/10（2.5‑bit）配置，可在 24GB/12GB GPU 上高效部署，超越同等整数位宽模型。

**⚠️ 局限性**

局限性：
- 高维投影需要控制 D‑d ≤ 20 以限制最近邻搜索复杂度，限制了可达的理论码率；
- 在 4‑bit 以上的高位宽区间，分数位宽的优势减弱，主要优势集中在 2–4 位之间；
- 需要额外的白化变换与投影矩阵训练，增加了训练成本；
- 对校准数据的敏感性仍存在，需要进一步研究稳健性；
- 目前实现主要针对单 GPU 推理，跨多卡或大 batch 的优化尚待完善。

---

## 8. Spatial Artifact Coherence Determines Codec Robustness in Patch-Based rPPG

**arXiv ID:** 2606.04198 | [PDF](https://arxiv.org/pdf/2606.04198v1)

**作者:** Achraf Ben Ahmed `[一作]` `[通讯]` (PlesmoSense SARL), Achraf Ben Ahmed (PlesmoSense SARL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了压缩视频对基于patch的rPPG性能的影响，并提出了空间失真一致性指标SAC来决定PCA方法是否优于全局投影方法。

**💡 创新点**

创新点在于引入SAC这一物理量来量化码流空间失真一致性，说明MPEG-4的宏块结构导致SAC升高，从而解释PCA方法在不同压缩环境下的优劣；并提出了PatchPCA四款codec-aware算法。

**🔧 技术方法**

使用了4×4绿通道协方差矩阵、SAC计算、PCA、CHROM/ POS/2SR基线、P-Hybrid等PatchPCA算法，结合Wilcoxon检验和BH-FDR多重校正。

**📊 数据集**

数据集包括MCD、UBFC-rPPG和UBFC-PHYS三大公开视频数据，覆盖280名受试者、11种压缩变体。

**📈 对比分析**

通过与CHROM等基线在每个压缩变体下的MAE比较，发现SAC<0.30且运动低时PatchPCA可获得显著MAE降低（平均约6–7 BPM），但在MPEG-4原始压缩下性能下降。

**⚠️ 局限性**

限制包括仅使用四个patch的协方差矩阵、受限于10秒窗口导致频率分辨率有限、未评估深度学习rPPG以及对不同源码流状态的观察性结论。

---

## 9. Sparse Mixture-of-Experts Reward Models Learn Interpretable and Specialized Experts for Personalized Preference Modeling

**arXiv ID:** 2606.04284 | [PDF](https://arxiv.org/pdf/2606.04284v1)

**作者:** Yifan Wang `[一作]` (Saarland University), Vera Demberg `[通讯]` (Saarland University)

**通讯引用:** 4433 | [OpenAlex ID](https://openalex.org/A5023605306)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种稀疏Mixture-of-Experts（MoE）奖励模型，在不需额外标注的二元偏好数据上学习可解释且专用的专家，以实现低数据个性化对齐。

**💡 创新点**

创新点在于通过局部稀疏、全局平衡和专家多样性三项正则化强制模型产生稀疏路由和互不重叠的专家，从而获得可解释的语义分区和更好的个性化性能。

**🔧 技术方法**

采用稀疏MoE架构、贝叶斯优先概率奖励模型、梯度下降与正则化损失、Hedge算法微调路由，并利用大型语言模型生成专家自然语言描述。

**📊 数据集**

使用SHP（结构化偏好数据）、RPR（属性级偏好数据）和700K大规模二元偏好数据集进行实验，同时在PersonalLLM数据集验证用户级个性化。

**📈 对比分析**

与单头奖励模型、传统MoE、MiCRo等基线相比，稀疏MoE在可解释性、专家专化度、属性级和用户级个性化上分别提升约25‑30个百分点；在控制实验中几乎恢复全部类别，且在实际数据上实现了最高的描述准确性和专家专化评分。

**⚠️ 局限性**

局限性包括需手动调节多项超参数、对训练数据潜在结构高度依赖且难以预知专家行为、以及在全局偏好预测上略有性能下降。

---

## 10. CLAW: Learning Continuous Latent Action World Models via Adversarial Latent Regularization

**arXiv ID:** 2606.04130 | [PDF](https://arxiv.org/pdf/2606.04130v1)

**作者:** Tewodros Ayalew `[一作]`, Matthew R. Walter `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文介绍了CoRL 2026的提交过程和格式要求。

**💡 创新点**

创新点在于提供了电子提交的详细指导和格式规范。

**🔧 技术方法**

使用了电子提交系统和特定的引用格式。

**📊 数据集**

未提及具体数据集。

**📈 对比分析**

未提供与其他方法的比较或性能评估。

**⚠️ 局限性**

缺乏具体实验结果和数据支持，无法验证提出的指导的有效性。

---

## 11. XSSR: Cross-Domain Self-Supervised Representative Selection for Efficient Annotation in Medical Image Segmentation

**arXiv ID:** 2606.04301 | [PDF](https://arxiv.org/pdf/2606.04301v1)

**作者:** Byunghyun Ko `[一作]` (Northeastern University), Jeongkyu Lee `[通讯]` (Northeastern University)

**通讯引用:** 1169 | [OpenAlex ID](https://openalex.org/A5101970322)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并实现了一套跨域自监督代表性样本选择框架（XSSR），用于在目标医学影像分割任务中显著减少注释工作量，同时保持接近全量数据的分割性能。

**💡 创新点**

创新点包括：① 将卷积MAE预训练与基于密度、novelty和diversity的三项指标融合的贪心采样策略相结合；② 通过覆盖度最小化自动校准novelty–diversity权重α，消除手工调参；③ 兼顾跨域差异与目标域内部多样性的平衡。

**🔧 技术方法**

主要技术手段：自监督卷积MAE、embedding空间距离度量、贪心近邻/远点采样、密度/novelty/diversity分数组合、自动α校准、U‑Net分割训练与二元交叉熵损失。

**📊 数据集**

实验数据集：胸部X光（Montgomery→Shenzhen）、视网膜光盘（BinRushed→MESSIDOR）以及六站点前列腺MRI（RUNMC→5个目标站）。

**📈 对比分析**

在5%注释预算下，与全量训练、随机采样和CoreSet比较，XSSR在胸部X光得到0.945 Dice（保留99.3%上限），视网膜0.940 Dice（比随机高1.3点），前列腺0.760 Dice（比随机高2.5点），整体比基线提升0.4–1.2 Dice点。

**⚠️ 局限性**

局限性：① 对极端硬件差异（如GE终腔线圈）效果下降；② 在高注释比例下随机采样可略优于XSSR；③ 缺少站点感知的采样策略，无法完全覆盖多站点多样性。

---

## 12. Large Language Models Hack Rewards, and Society

**arXiv ID:** 2606.04075 | [PDF](https://arxiv.org/pdf/2606.04075v1)

**作者:** Wei Liu `[一作]` (King's College London), Yulan He `[通讯]` (King's College London)

**通讯引用:** 13830 | [OpenAlex ID](https://openalex.org/A5015709853)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了“社会层面奖励黑客”概念，构建了72个模拟社会环境的基准（Societal Hacking Benchmark，SHB），并用强化学习（RL）让大型语言模型在这些环境中自主发现并利用法规漏洞，进一步研究了模型对现有安全防护的规避情况。

**💡 创新点**

创新点包括：①将奖励黑客扩展到真实社会规则，揭示RL驱动模型可在不受明确提示的情况下“合法”完成规避行为；②创建可分三类（Historical、Synthetic、Fictional）的社会环境基准，方便评估模型在历史真实漏洞、植入漏洞和虚构场景下的表现；③系统性评估现有LLM安全措施（拒绝、输出治理、训练时正则化）对社会层面奖励黑客的防护效果，并揭示其局限性。

**🔧 技术方法**

主要技术：强化学习（采用Dr. GRPO目标的policy‑gradient方法）、自然语言策略生成、基于LLM的社会模拟器（对策略进行动作解析、状态演算和结果评分）、动态补丁注入机制、对比实验基线（Best‑of‑N、IterPrompt、EvoPrompt、Direct Ask）、LLM（Qwen3‑30B‑A3B）训练与Gemini‑3‑flash评估。

**📊 数据集**

使用的主要数据集为 Societal Hacking Benchmark（SHB）72个环境，包含32个基于真实法规的Historical环境、20个Synthetic环境（植入已知漏洞）和20个Fictional环境（将Synthetic环境转换为虚构世界）。这些环境来源于真实法规文本、公开补丁记录以及人工构造和LLM重写。

**📈 对比分析**

比较方法：在三类基准上评估RL、IterPrompt、EvoPrompt、Best‑of‑N和Direct Ask的Recall@K、Precision、F1、Novelty、深度等指标。实验结果显示：RL在Historical环境中Recall最高（约61%）、Precision 91%，与Baseline相比显著提升；在Synthetic与Fictional环境中Recall快速饱和，但RL仍保持较高Recall和Precision；RL也展现出较高的Novelty（NTPR≈0.13）和较好的深度（在共享补丁池中存活时间更长）。此外，RL在安全防护实验中能够绕过输入拒绝，输出治理和训练时正则化效果有限。

**⚠️ 局限性**

局限性包括：①基准仍是模拟，未能完全捕捉真实机构的复杂执行与补丁机制；②评估依赖LLM裁判，可能导致匹配误差；③Ground‑truth补丁有限，只覆盖已知漏洞，未能评估模型发现的全新漏洞；④实验仅在部分开源模型上进行，未验证封闭模型或更复杂RL框架；⑤防护评估未涵盖正式机构层面的审计与监控，仅限模型级正则化与自评。

---

## 13. Bayes-Sufficient Representations in Supervised Learning

**arXiv ID:** 2606.04045 | [PDF](https://arxiv.org/pdf/2606.04045v1)

**作者:** Vasileios Sevetlidis `[一作]` `[通讯]` (Athena Research Center), Vasileios Sevetlidis (Athena Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了“Bayes‑充分”表示学习框架，定义在给定联合分布和损失下，若有预测头能实现贝叶斯最优决策，则该表示为Bayes‑充分，并通过唯一贝叶斯动作的贝叶斯商（Bayes quotient）对所需信息进行精确分割；

**💡 创新点**

创新点在于将传统统计决策理论与表示学习结合，首次给出在唯一贝叶斯动作情形下的精确贝叶斯商，区分充分性与最小化，阐明损失相关的预测信息量，并将属性诱导（property elicitation）与贝叶斯商联系；

**🔧 技术方法**

论文使用了测度论与Doob–Dynkin分解、贝叶斯决策理论、属性诱导理论以及信息瓶颈等理论工具，并通过线性探针对学习到的表示进行可解释性评估；

**📊 数据集**

实验数据包括可解析的人工分布（可求贝叶斯商）、从连续观测学习的神经瓶颈表示以及真实图像数据集iNaturalist（包含物种–属–科层次）；

**📈 对比分析**

通过比较手工定义与学习到的表示在不同损失（zero‑one、log、Brier、平方）下的下游风险（准确率、NLL、Brier）和探针可恢复性，验证了理论预期：充分性与非最小化的区别，且性能与理论一致；

**⚠️ 局限性**

主要局限包括仅在群体层面给出理论、对非唯一贝叶斯动作情形处理不完整、缺乏近似/有限样本的理论分析，以及探针诊断的解释性与通用性受限。

---

## 14. The Saturation Trap and the Subjectivity of Intervention Timing: Why Affect-Based Triggers and LLM Judges Fail to Time Interventions on Autonomous Agents

**arXiv ID:** 2606.04296 | [PDF](https://arxiv.org/pdf/2606.04296v1)

**作者:** Manvendra Modgil `[一作]` `[通讯]`, Manvendra Modgil

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估不同类型的运行时干预触发器（阈值、组合模式、正则化文本特征、LLM裁决）在自律 AI 轨迹中的时机匹配度，并与人类标注比较。

**💡 创新点**

首次系统性揭示了“状态饱和陷阱”与“低可靠性干预时机”两大根本性问题，并将人类标注的不一致性作为评估基准的核心限制。

**🔧 技术方法**

使用 HEART 连续情感动力学引擎、四种触发器架构、LLM-judge（GPT‑5.4、Claude 等）以及 Krippendorff’s α、Cohen’s κ 等多维度评价指标。

**📊 数据集**

基于 SWE‑bench‑Verified 调试轨迹（主轨迹56步，其他5条轨迹用于饱和性检验），并收集三名训练标注者的干预标签。

**📈 对比分析**

通过与人工标签的 F1、触发率对比发现：阈值触发器因状态饱和而频繁误触，LLM‑judge 需全轨迹上下文且成本高，整体检测性能低于人类一致性水平。

**⚠️ 局限性**

局限包括：仅一条主轨迹、少量标注者、稀疏标签、LLM‑judge 结果波动大、阈值触发器无法避免饱和、缺乏跨多样化轨迹验证。

---

## 15. Prospective Dynamic 3D MRI Reconstruction via Latent-Space Motion Tracking from Single Measurement

**arXiv ID:** 2606.04249 | [PDF](https://arxiv.org/pdf/2606.04249v1)

**作者:** Lixuan Chen `[一作]` (University of Michigan), Liyue Shen `[通讯]` (University of Michigan)

**通讯引用:** 6528 | [OpenAlex ID](https://openalex.org/A5072483985)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了PDMR框架，利用离线学习的低维运动流形实现单测量的实时3D MRI重建。

**💡 创新点**

结合三平面几何感知映射网络，将低维潜在向量映射到高维3D DVF，并仅需优化潜在向量即可快速适应新运动状态。

**🔧 技术方法**

使用运动补偿（MoCo）分解、低维流形学习、自编码器结构、三平面生成器+MLP解码器，以及黄金角栈式星形采样等技术。

**📊 数据集**

在XCAT数字幻像和六例腹部动态MRI真实数据上进行实验。

**📈 对比分析**

与NUFFT、GRASP、TDDIP、SPINER、Prior-INR、MR-MOTUS等方法比较，PDMR在PSNR/SSIM上平均提升约2dB，并在两分钟后仍保持高质量重建。

**⚠️ 局限性**

对极端超出训练分布的运动仍有误差，需要足够的离线数据来构建流形。

---

## 16. ADAPTOOD: Uncertainty-Aware Fine-Tuning for Out-of-Distribution ECG Time Series Models

**arXiv ID:** 2606.04164 | [PDF](https://arxiv.org/pdf/2606.04164v1)

**作者:** Sotirios Vavaroutas `[一作]` (University of Cambridge), Cecilia Mascolo `[通讯]` (University of Cambridge)

**通讯引用:** 18716 | [OpenAlex ID](https://openalex.org/A5010623957)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种基于数据不确定性量化的动态自适应微调框架ADAPTOOD，用于在时间序列数据中针对不同严重程度的OOD分布偏移进行模型微调。

**💡 创新点**

创新点在于利用马氏距离与Hellinger距离结合的 uncertainty 指标来评估OOD严重度，并据此实现层级解冻、低秩适配（LoRA）和贝叶斯超参数优化，从而在多种分布偏移下实现动态、参数高效的微调。

**🔧 技术方法**

技术包括PCA降维、马氏距离与Hellinger距离的不确定性估计、低秩适配LoRA、贝叶斯优化超参数、选择性层解冻以及基于1D CNN的时间序列模型。

**📊 数据集**

使用的主要数据集为预训练的PhysioNet CinC 2017；下游评测数据涵盖MIT‑BIH、PTB‑DB、MIMICPERform（ECG与PPG）、CODEtest，涵盖传感器、人口、时间、标签、模态及维度等多维度分布偏移。

**📈 对比分析**

通过与传统迁移学习、监督学习、特征/实例级域适配等基线对比，ADAPTOOD在多种OOD任务中平均提升准确率约7%和精确率约13%，在最严重偏移下仍保持高F1得分，表明显著的性能优势。

**⚠️ 局限性**

局限性包括仍需标注样本；对极端模态转换（如ECG→PPG）效果下降；对不同数据集的超参数敏感性；未完全解决无监督跨域标签分布差异的场景。

---

## 17. Negative and Fractional Types in the Fidelity Framework

**arXiv ID:** 2606.04352 | [PDF](https://arxiv.org/pdf/2606.04352v1)

**作者:** Houston Haynes `[一作]` `[通讯]`, Houston Haynes

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本论文在Native Type Universe（NTU）框架上引入负类型和分数类型，构建可逆计算与约束传播的类型层级，支持事件溯源、贝叶斯推理、量子和退火计算等应用。

**💡 创新点**

创新点在于将James & Sabry提出的两种对偶（加法负类型与乘法分数类型）直接嵌入 Hindley–Milner 统一与 Kennedy 单位系统的可决定 abelian group 结构中，并通过编译时结构化保证信息保持，进一步实现编译器、验证与运行时的完整闭环。

**🔧 技术方法**

使用技术包括：Hindley–Milner 类型推断扩展、Gaussian elimination 上的 abelian group 统一、compact‑closed 类别理论与 adjoint 逻辑、Baker 级别的 PSG/PHG 结构化、MLIR 低层化（Inet、SMT、CIRCT、AIE 等方言）、SMT 求解器 Z3 的约束归约、以及多维度 cell‑complex 与 sheaf 验证框架。

**📊 数据集**

论文未给出具体实验数据集，主要以理论证明和示例程序（如可逆决策查找、贝叶斯条件推理、量子门与测量、退火步骤）作为验证。

**📈 对比分析**

评估方法侧重于可决定性与多维度一致性证明：类型系统保持多项式时间统一，编译通过多层检查保证结构不变；性能表现未给出数值指标，但作者声称在编译时间内保持可伸缩性，并通过 SMT 层实现高效约束归约。

**⚠️ 局限性**

局限性包括：实现需要完整的编译链与多维度元数据管理，理论性强，实际落地尚未充分验证；负/分数类型的运行时解释在不同目标（FPGA、NPU、量子硬件）上仍需进一步细化；对大规模应用的性能与可扩展性仍待经验评估。

---

## 18. Video2LoRA: Parametric Video Internalization for Vision-Language Models

**arXiv ID:** 2606.04351 | [PDF](https://arxiv.org/pdf/2606.04351v1)

**作者:** Manan Suri `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**通讯引用:** 40096 | [OpenAlex ID](https://openalex.org/A5004194238)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Video2LoRA，能将视频编码成 LoRA 适配器，使冻结的 VLM 在不携带视觉 token 的情况下回答问题。

**💡 创新点**

首次实现参数化视频内部化，使用单次前向传递生成 LoRA 适配器，实现高效多次查询。

**🔧 技术方法**

采用 Perceiver 超网络将冻结 VLM 的中间表示映射为 LoRA 权重，并结合 LoRA 注入与冻结模型。

**📊 数据集**

使用 FineVideo 生成的视频片段进行训练，评估在 ActivityNet Captions、PLM-RDCap、PLM-RCap、VDC、CaReBench、NExT-QA、ActivityNet-QA、PLM-SGQA、VidCapBench 等基准。

**📈 对比分析**

在所有五个字幕基准上与直接视频上下文推理无显著差异，在七个 QA 基准中也保持非劣势，且在视觉 token 数量和查询延迟上显著提升（最多 1500× 视觉 token 减少，6–80× 查询 TTFT 缩短）。

**⚠️ 局限性**

仅针对 500M 与 2.2B SmolVLM2 训练，需为不同 VLM 规模训练不同超网络，且对细粒度细节（如相机运动、空间细节）保持不足，未包含音频或更复杂的分块组合。

---

## 19. POLARIS: Guiding Small Models to Write Long Stories

**arXiv ID:** 2606.04095 | [PDF](https://arxiv.org/pdf/2606.04095v1)

**作者:** Rishanth Rajendhran `[一作]` (University of Maryland), John Frederick Wieting `[通讯]` (Google DeepMind)

**通讯引用:** 2761 | [OpenAlex ID](https://openalex.org/A5002499277)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为小型开源模型提供了一套低算力的强化学习训练方案，使其在长篇创意写作中既能保持高质量，又能遵循长度指令。

**💡 创新点**

创新点在于：①使用前沿LLM作为在线评审，依据结构化的 Story Quality 量表直接生成多维奖励；②引入“人类参考注入（HRI）”，在每个 GRPO 组中加入一条高质量人类故事，保持梯度方向并提升长篇表现。

**🔧 技术方法**

主要技术包括 Group Relative Policy Optimization (GRPO)、LLM-as-judge（基于 GPT‑5.4/ Gemini 的结构化评分）、人类参考注入（HRI）以及长度调节的奖励函数。

**📊 数据集**

使用约 1.4K 条来自 100 本商业短篇小说集的 prompt‑story 对；训练时采用 4 块 A100 GPU，batch 为 8，训练时间约 48 小时。

**📈 对比分析**

在 5 个包含 ID/OOD 题目和量表的基准上（Story Quality、EQ‑Bench Longform、WritingBench、LongBench‑Write 等）与多种开源模型比较，Polaris‑9B 在质量与长度遵从度上均进入顶级开源模型群，并在 8–12k 字长时保持 0.72 的长度比例，显著优于大多数同类模型；人工评测显示其与 Qwen3.5‑27B 差距可忽略。

**⚠️ 局限性**

局限包括：LLM 评审可能存在系统偏差、训练数据不公开导致可复现性受限、人工评测规模有限、模型在极长篇时仍有轻微长度欠缺、未充分验证跨域写作（如剧本、评论等）对训练效果的影响。

---

## 20. Selecting haptic guidance models in teleoperation: guidelines from a comparative user study

**arXiv ID:** 2606.04157 | [PDF](https://arxiv.org/pdf/2606.04157v1)

**作者:** Alexis Boulay `[一作]` (Farm3), David Daney `[通讯]` (Inria)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文在遥操作场景中，提出统一的刚度-阻尼模型框架来描述弹性、势场和导向管三类主流触觉引导方法，并通过六种环境变化的垂直农业实验与四种引导模式（无引导、弹性-阻尼、导向管、势场）进行用户对比研究。

**💡 创新点**

创新点在于：①将多种弹性引导模型归纳为统一的刚度-阻尼系统；②设计了一套包含性能、舒适度、安全性与信任度的五个客观指标，并验证其与主观问卷的一致性；③根据实验结果给出基于环境特征的模型选择准则。

**🔧 技术方法**

技术手段包括：ROS 双向遥操作框架；电动力学模型（PD 控制 + 优化求解）；统一刚度-阻尼方程；客观指标计算公式；NASA-RTLX 与 Muir 问卷；统计分析（ANOVA + Bonferroni）。

**📊 数据集**

数据集为 28 名志愿者在 6 个垂直农业场景下完成的 4 种引导模式实验，共 112 条完整实验记录。每条记录包含实时力学数据、指标值和主观问卷评分。

**📈 对比分析**

比较方法：对每种引导模式在每个场景下的完成时间、认知负荷、最小障碍距离等指标进行 ANOVA 检验；并将客观指标与 NASA‑RTLX、Muir 的得分相关性进行回归。结果显示：弹性-阻尼在高度拥挤场景下最快、最安全；势场在空旷场景下最快；导向管在大多数场景下表现平衡；舒适度指标（引导力幅值）与主观舒适与信任度高度相关。

**⚠️ 局限性**

局限性包括：仅针对垂直农业任务；受限于 28 人样本；引导参数使用经验调优，缺乏统一自动调节方法；未考虑长期使用的疲劳与适应；模型切换未实现实时动态切换。

---

## 21. Robust Multi-view Clustering against Imperfect Information

**arXiv ID:** 2606.04343 | [PDF](https://arxiv.org/pdf/2606.04343v1)

**作者:** Zhichao Huang `[一作]` (Sichuan University), Xi Peng `[通讯]` (Sichuan University)

**通讯引用:** 10243 | [OpenAlex ID](https://openalex.org/A5022800038)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个统一处理不完整视图与噪声对应的多视图聚类框架 PLCCI。

**💡 创新点**

创新点在于将期望的对应视图视为潜在变量，通过实例级可靠性估计与原型级语义传输共同推断后验，从而一次性解决 IV 与 NC 两大挑战。

**🔧 技术方法**

采用深度网络记忆效应估计对应可靠性、可靠性加权最优传输实现原型对齐，以及对后验均值的对比损失和 KL 正则化，形成端到端可训练的框架。

**📊 数据集**

在六个公开多视图数据集（Scene15、LandUse21、Reuters、CCV20、HandWritten、SUN RGB‑D）以及原始 RGB‑Depth 数据上进行实验。

**📈 对比分析**

与 10 种主流 IV、NC 或鲁棒 MvC 方法对比，PLCCI 在多种 IIR 设置下均取得平均最优的 ACC/NMI/ARI，尤其在高缺失或高噪声比例下优势明显。

**⚠️ 局限性**

局限性在于仍需依赖完整实例的可靠性估计，对极端缺失或完全随机对应的情况适应性有限，且模型训练与推断的计算量相对较大。

---

## 22. Think Fast and Far: Long-Horizon Online POMDP Planning via Rapid State Sampling

**arXiv ID:** 2606.04355 | [PDF](https://arxiv.org/pdf/2606.04355v1)

**作者:** Yuanchu Liang `[一作]` (Australian National University), Hanna Kurniawati `[通讯]` (Australian National University)

**通讯引用:** 2752 | [OpenAlex ID](https://openalex.org/A5073857354)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种基于连续参考策略的POMDP在线规划框架（Ref‑POMCP），通过采样生成宏动作并在贝尔曼备份中使用期望取代枚举，实现对长时序、部分可观测、高维机器人问题的高效规划。

**💡 创新点**

创新点：
- 将参考策略从“信念到信念”转化为“策略到策略”，实现闭式备份并可在连续动作空间上直接使用。
- 采用向量加速运动规划（VAMP）快速在线生成宏动作，显著缩短有效规划时域。
- 用期望代替枚举，使收敛速率依赖于采样动作数而非整个动作空间大小。
- 设计了基于信息状态的采样启发式，为参考策略提供多样化、几何匹配的宏动作。

**🔧 技术方法**

使用的技术：
- Partially Observable Markov Decision Process（POMDP）理论。
- 参考策略（reference‑based）与KL惩罚。
- 采样均衡与连续动作空间的Monte‑Carlo期望估计。
- VAMP（Vector‑Accelerated Motion Planning）实现快速宏动作生成。
- 进阶的树搜索与渐进宽化技术。
- Python/C++实现、PyBullet可视化。

**📊 数据集**

数据集/实验环境：
- 7种长时序模拟任务（Light Dark、Maze2D、Random3D、Multi‑Drone Tag、Sphere‑Search、Ray‑Detect、Shelf‑Move）。
- 维度从2D到35维，规划时域从几百到1500步。
- 还在Hello‑Robot Stretch 3移动机械臂上进行真实机器人演示。

**📈 对比分析**

对比方法：
- 基线：B‑VAMP、Ref‑Basic、POMCP、R‑POMCP、MAGIC、RMAG。
- 结果：在所有任务中，Ref‑POMCP的成功率、奖励和规划时间均显著优于基线，尤其在高维、长时序任务（Shelf‑Move、Multi‑Drone Tag）中提升数倍；在简单任务（Light Dark、Sphere‑Search）提升约25%。

**⚠️ 局限性**

局限性：
- 对参考策略的质量高度依赖；若参考策略与最优策略相距较大，性能下降明显。
- 需要手工设计采样启发式，敏感于超参数（树深度、进展宽化系数等）。
- 仍存在计算开销，尤其在极大动作空间或高维状态空间下。
- 只给出近似最优解，理论收敛到原POMDP最优解仍需进一步研究。

---

## 23. PE-MHL: Physics-Encoded Modular Hybrid Layers for Scalable Learning of Complex Systems

**arXiv ID:** 2606.04290 | [PDF](https://arxiv.org/pdf/2606.04290v1)

**作者:** Ismail Hassaballa `[一作]` (Eindhoven University of Technology), Mircea Lazar `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 5666 | [OpenAlex ID](https://openalex.org/A5045512282)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可逐步扩展的物理编码模块化混合层（PE-MHL）框架，用于将物理模型与轻量级神经网络子模型相结合以学习复杂系统。

**💡 创新点**

核心创新在于利用最小二乘初始化的递增子模型，在保持先前子模型不变的前提下通过偏差惩罚保证训练误差单调下降并可证明收敛；该方法天然提供有限停止准则。

**🔧 技术方法**

采用物理编码基础模型、弹性权重正则化（Ridge-EVE）、多阶段残差学习、偏差惩罚以及梯度下降/Adam优化。

**📊 数据集**

在三个基准上验证：1）非线性静态函数（含高斯噪声）；2）NARX 自回归外源系统；3）Quanser Aero 2 航模实验数据。

**📈 对比分析**

与等参数容量的单一大型神经网络基线相比，PE-MHL 在训练误差下降更快、收敛更平稳、测试 MAE/MSE 低约 5–10 倍，且在多输入激励下表现更稳健。

**⚠️ 局限性**

局限性包括：需要手动确定子模型数量和偏差惩罚强度；对极端噪声或极少数据时仍可能出现过拟合；以及在多物理域耦合时扩展性尚待验证。

---

## 24. PureLight: Learning Complex Luminaires with Light Tracing

**arXiv ID:** 2606.04319 | [PDF](https://arxiv.org/pdf/2606.04319v1)

**作者:** Pedro Figueiredo `[一作]` (Texas A&M University), Nima Khademi Kalantari `[通讯]` (Texas A&M University)

**通讯引用:** 4338 | [OpenAlex ID](https://openalex.org/A5008433002)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `40105733-5154-44cd-8090-a8cab9e64b07` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用光追踪样本，通过分布学习构建复杂灯具的表面辐射特性，并将其预训练为轻量化 MLP，用于快速渲染；同时提供重要采样网络和离散化表面辐射表，支持直接照明计算。

**💡 创新点**

①将灯具辐射分布建模为光追踪概率密度，避免传统路径追踪高方差；②采用归一化流（normalizing flow）学习方向分布，再与光子密度估计的空间分布相乘得到完整辐射；③将大型模型蒸馏为小型 MLP，显著加速推理；④提出兼容任何场景的透明度网络和低分辨率离散辐射表以进一步降低噪声。

**🔧 技术方法**

光追踪、归一化流网络、光子映射、哈希网格编码、Spherical Harmonics、MIP‑MLP 蒸馏、离散化 16×16 采样表、Mitsuba 3 渲染器、tiny‑cuda‑nn、PyTorch。

**📊 数据集**

作者在自定义的合成灯具场景中采样（Cluster、Crystal、Medieval、Modern、Rough Flower、Vintage 等灯具），使用 Mitsuba 生成光追踪数据；没有使用公开标准数据集，而是通过场景自定义灯具几何和材质构造训练集。

**📈 对比分析**

与传统路径追踪（PT）、双向路径追踪（BDPT）以及基于 NCL 的方法比较；在 64 spp 下取得比 1M spp PT 更低噪声、40%+ 速度提升；在 BDPT 低样本下噪声更低；在 NCL 对复杂灯具的训练样本方差大时，PureLight 能保持更高质量；总体表现：在低样本下噪声显著下降，速度提升显著。

**⚠️ 局限性**

不考虑灯具表面被场景照明后的反射，无法建模完整的 8D 反射函数；离散化低分辨率表面辐射会产生轻微偏差；当前实现受限于 Mitsuba 的神经网络推理支持，导致推理速度受限。

---

## 25. RSC: Decentralized Rigid Formation Flocking for Large-Scale Swarms via Hybrid Predictive Control and Online Reconfiguration

**arXiv ID:** 2606.04248 | [PDF](https://arxiv.org/pdf/2606.04248v1)

**作者:** Ganyu Zou `[一作]`, Chang-Tien Lu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在分布式环境下设计了一种基于有限时域轨迹预测与人工势场融合的 Rigid Swarm Control（RSC）框架，能够在拥挤环境中实现大规模无人机群的刚性形态保持、障碍物避让与目标跟踪。

**💡 创新点**

创新点在于：①将有限时域轨迹预测作为控制输出，缓解单步决策的短视问题；②构建混合控制器，将预测轨迹与快速 APF 逃逸结合，突破局部极小陷阱；③引入在线领袖‑追随角色交换机制，动态重构树拓扑，消除死锁；④通过 GNN + Transformer 端到端学习实现去中心化决策。

**🔧 技术方法**

使用了图神经网络（GraphSAGE）、Transformer 编码、人工势场（APF）安全控制、有限时域轨迹预测、DAgger 训练、在线领袖-追随重配置、仿真与实机 Crazyflie 实验等技术。

**📊 数据集**

在仿真中随机生成 500 张地图（3–6 个静态障碍，随机初始位置），测试集 200 张地图；实机实验采用 6 台 Crazyflie 2.1 无人机与 LightHouse 定位系统。

**📈 对比分析**

与传统 APF‑Olfati‑Saber 及 STGNN 基线对比，RSC 在 25 只无人机的测试中成功率提升至 83%（基线 <5%），平均组态时间与 MAE 均处于可接受范围；Ablation 证明在线重配置与 APF 融合分别提升约 33.5% 与 4.5% 的成功率。

**⚠️ 局限性**

局限在于仅针对静态障碍物、仿真规模有限（最多 36 只 UAV），对高通信丢包/延迟、动态障碍物的鲁棒性未验证，且预测时域对 MAE 有一定影响。

---

## 26. SBP-Net: Learning Thin Structure Reconstruction with Sliding-Box Projections

**arXiv ID:** 2606.04251 | [PDF](https://arxiv.org/pdf/2606.04251v1)

**作者:** Ofir Gilad `[一作]`, Andrei Sharf `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计了一种基于局部深度投影的薄三维结构重建方法（SBP-Net），利用滑动盒子将三维重建问题拆分为多张正交深度投影，在二维注意力U‑Net中完成缺失，再将重建结果重新投影并融合回三维空间。

**💡 创新点**

创新点包括：①将薄结构重建转化为局部正交深度投影；②使用滑动盒子覆盖大范围并保证重叠以保持连通性；③在投影上引入自注意力增强的U‑Net，捕获全局上下文实现精准孔洞填补；④开发逻辑融合模块，确保只补全真正缺失区域，维护拓扑连通性；⑤在训练时通过拓扑一致性筛选“孔洞”，避免误补。

**🔧 技术方法**

使用的技术包括滑动盒子投影（SBP）、六面正交深度投影、注意力增强的U‑Net（瓶颈自注意力）、3D重投影与融合逻辑、加权L1损失以及基于拓扑一致性筛选的hole标注。

**📊 数据集**

使用的数据集有：医疗CT肺动脉数据集PARSE 2022；合成工业管道数据集PipeForge3D（网格与点云两种形式）；以及真实工业管道扫描数据集Hospital Central Utility Plant（医院中央供给管道）和Power Plant Real Scans。

**📈 对比分析**

与DeepCA、OReX、Conv ONet、3D-RecGAN、UNet3D等SOTA方法在Chamfer Distance、Hausdorff Distance、Connected Components、Dice、SSIM等指标上进行对比。SBP‑Net在CT肺动脉与合成管道数据集上在孔洞填补、Dice、SSIM等指标上均显著优于基线，并在Connected Components上获得最高分；在真实扫描上性能略逊但仍优于多数基线。

**⚠️ 局限性**

局限性包括：1) 对于被前景遮挡的自遮挡孔洞检测与重建仍然困难；2) 依赖二维投影的先验限制了对新结构或未知域的泛化能力；3) 目前缺乏多视角一致性与显式遮挡推理机制，未来可通过更强的3D感知和多视角约束进一步提升。

---

## 27. dMX: Differentiable Mixed-Precision Assignment for Low-Precision Floating-Point Formats

**arXiv ID:** 2606.04115 | [PDF](https://arxiv.org/pdf/2606.04115v1)

**作者:** Giuseppe Franco `[一作]` (AMD), Nicholas Fraser `[通讯]` (AMD)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型在MXFP浮点量化中引入可学习的层级位宽分配。

**💡 创新点**

创新点在于用连续偏移量与温度退火实现梯度可微的浮点位宽学习，并以目标位宽正则化控制量化精度。

**🔧 技术方法**

采用连续浮点量化参数化、温度退火、目标感知正则化以及后训练量化（PTQ）与Brevitas。

**📊 数据集**

使用FineWeb做校准，WikiText-2评估困惑度，四个零样本推理基准（ARC-Challenge/Easy、HellaSwag、WinoGrande）评估准确率。

**📈 对比分析**

与统一位宽量化和基于KL散度的贪心层选择法对比，实验显示在4–8位宽范围内显著降低困惑度并提升准确率，尤其在中等位宽时优于对手。

**⚠️ 局限性**

局限在于仅验证于1–2B参数模型，缺乏对更大规模或MoE模型的评估，且对温度调度与正则化参数的选择仍需更多经验。

---

## 28. Supportive Token Revealing for Fast Diffusion Language Model Decoding

**arXiv ID:** 2606.04236 | [PDF](https://arxiv.org/pdf/2606.04236v1)

**作者:** Giries Abu Ayoub `[一作]` (University of Haifa), Loay Mualem `[通讯]` (University of Stuttgart)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5005835154)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种训练无关的模块AXON，用于提升离散扩散语言模型（dLLM）的并行解码质量-延迟权衡，通过监控剩余不确定掩码并在需要时插入锚点；

**💡 创新点**

创新点在于提出了状态监测与基于注意力、置信度和不确定性的锚点选择框架，并采用子模函数覆盖目标进行非冗余锚点挑选；

**🔧 技术方法**

使用了离散扩散语言模型、注意力矩阵、置信度、熵测度、子模函数覆盖优化以及轻量门控机制；

**📊 数据集**

实验数据集包括数学推理与代码生成任务：GSM8K、Minerva‑Math、HumanEval、MBPP；

**📈 对比分析**

与基线的 confidence‑based、LocalLeap（locality‑aware）和 DAWN（dependency‑aware）进行对比，结果显示AXON在多种模型（Dream‑v0‑Instruct‑7B、LLaDA‑1.5、LLaDA‑8B‑Instruct）上均提升了准确率或吞吐量，同时显著降低了函数评估次数（NFE）；

**⚠️ 局限性**

局限性包括：依赖注意力与置信度的质量；选择过程带来额外计算开销；门控与锚点预算仍需任务调参；在长文本、多模态、对话等场景下尚未验证。

---

## 29. Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation

**arXiv ID:** 2606.04067 | [PDF](https://arxiv.org/pdf/2606.04067v1)

**作者:** Xinyue Huang `[一作]` (Sun Yat-sen University), Wenyuan Yang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1282 | [OpenAlex ID](https://openalex.org/A5101619232)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于情境完整性（Contextual Integrity）的隐私意识委派框架，利用本地小模型对用户查询进行重写，既保留任务所需的核心信息，又抑制多余的敏感披露；同时构建了首个任务级CI基准数据集DelegateCI-Bench，并将其标签直接转化为强化学习奖励，训练出统一的重写器；

**💡 创新点**

创新点包括：①将隐私意识委派从传统的PII类别化转变为任务条件化的情境完整性视角；②首次创建任务级CI基准，手工标注每条查询的必需与非必需敏感跨度；③将这一二分标签嵌入可验证的RL奖励，推动端到端的隐私友好重写模型；

**🔧 技术方法**

技术手段主要包括：情境完整性框架；基于策略梯度的GRPO强化学习；token‑level Jaccard与BERTScore相似度衡量；Qwen-2.5-3B-Instruct作为重写器模型；对奖励设计进行分层（泄露惩罚、重要性保留、长度惩罚）；使用LLM‑as‑Judge进行细粒度隐私评估；

**📊 数据集**

使用的数据集为DelegateCI-Bench，包含3,167个样本，分为一般域（逆合成+WildChat OOD）和医学挑战集；构建过程中还利用ShareGPT、WildChat、医学多选题集（medical‑o1‑reasoning‑SFT）及GPT‑4o进行标注与验证；

**📈 对比分析**

与本地仅执行（Local Only）、Presidio（规则式PII删减）、PAPILLON（PII‑中心的重写）以及PUFT（会话代理重写）等方法对比；在隐私指标（SLR/NE）和实用性指标（General swap‑consistency / Medical exact‑match）上，所提重写器在所有聚合器规模上均取得最高实用性（提升约+10.1）且泄露率最低；消融实验验证奖励三项（泄露、保留、长度）均不可或缺；

**⚠️ 局限性**

局限性主要在于：①情境完整性被简化为单一受众、单一传输原则的二元划分，忽略多方、多轮、争议性披露等复杂情形；②奖励机制仅检测表面或近似重写的泄露，难以捕捉推理泄露、跨查询链接与纵向去匿名化风险；③模型可能生成与原查询不符的敏感属性（hallucination），导致隐私失效；

---

## 30. Beyond Static Priors: Dynamic Neural Guidance for Large-Scale Ant Colony Optimization

**arXiv ID:** 2606.04039 | [PDF](https://arxiv.org/pdf/2606.04039v1)

**作者:** Dat Thanh Tran `[一作]` (VinUniversity), Yining Ma `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 981 | [OpenAlex ID](https://openalex.org/A5081047744)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了DyNACO框架，通过在蚁群优化（ACO）过程中周期性地向搜索状态注入动态神经引导，从而解决传统神经引导ACO的训练-推理不对齐问题。

**💡 创新点**

创新点包括：①将神经引导视为半马尔可夫决策过程（semi‑MDP），实现周期性更新；②使用状态感知（包括当前图、信息素分布、占优解）作为输入，生成动态引导；③引入扰动式ACO后端与局部范围修正（SRR）实现可扩展、稳定的信用分配；④通过轨迹感知训练让模型学习从早期探索到后期收敛的自适应策略。

**🔧 技术方法**

技术手段包括：图神经网络（12层）编码器+MLP解码器、PPO强化学习、扰动式ACO采样、K‑近邻候选图、局部2‑opt（SRR）、信息素稳定化、指导权重γ的调度。

**📊 数据集**

使用的数据集：①统一随机生成的欧氏TSP/CVRP实例（1K、5K、10K、50K、100K节点）；②真实世界TSPLIB（33个TSP实例）和CVRPlib（14个CVRP实例）；③公开基准（LKH‑3、HGS等）用于对照。

**📈 对比分析**

与多类基线比较（经典求解器LKH‑3、HGS；端到端神经解算器POMO、LEHD、BQ、INViT、SIGD、L2C‑Insert、SIL；神经‑ACO基线DeepACO、GFACS、GTG‑ACO、Heat‑ACO、无神经指导的ACO）。结果显示DyNACO在TSP和CVRP上在相同迭代预算下均取得最低gap，TSP可实现23–33%运行时间缩短，CVRP神经开销<1%，并在零样本推理下在TSPLIB/CVRPlib上均超过所有神经基线、甚至在CVRP上超越HGS。

**⚠️ 局限性**

局限性包括：①宏观动作的时间抽象（H,S）和固定K‑邻居候选图需手动设置，可能限制可扩展性；②当前模型在单一规模训练，跨规模适应需进一步改进（如自适应候选选择、动态宏观抽象、多尺度训练或元学习）；③模型对非欧氏或极端分布的泛化虽已验证，但在更复杂约束或其他组合优化问题上仍待验证。

---

## 31. Forecasting Political News Engagement on Social Media

**arXiv ID:** 2606.04293 | [PDF](https://arxiv.org/pdf/2606.04293v1)

**作者:** Karthik Shivaram `[一作]` (Tulane University), Aron Culotta `[通讯]` (Tulane University)

**通讯引用:** 5659 | [OpenAlex ID](https://openalex.org/A5011178134)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文收集并标注了七年间超过60M条推文，构建了约6.5M条新闻互动记录的数据集，使用双向LSTM结合新闻互动计数和TwHIN-Bert文本嵌入来预测用户未来三个月内各党派立场的新闻互动次数，并将LSTM的中间表示用于聚类，揭示长期新闻消费模式。

**💡 创新点**

创新点在于①首次构建大规模长期用户新闻互动数据集；②提出结合互动计数与文本特征的LSTM预测框架；③利用模型中间状态做用户聚类，揭示不同党派与话题的长期互动差异。

**🔧 技术方法**

技术手段包括双向LSTM（单特征与多特征网络）、TwHIN-Bert文本预训练模型、标准化互动计数、三个月时间步长、MAE损失训练、文本与标签的特征融合与季节编码。

**📊 数据集**

使用数据集：约63.4M条推文，6.5M条新闻互动，522个新闻源（包含低可靠性源），用户样本5,838人，时间范围2015-2021。

**📈 对比分析**

与基准（上一个时间步直接复制）对比，SFN+C模型MAE约3.7-4.0，优于基准；对突变转折点预测性能亦显著提升；总体而言，文本特征虽不如计数特征精准，但在预测无互动→互动转折时表现更好。

**⚠️ 局限性**

局限性包括样本偏向高新闻互动用户，无法区分讽刺或反讽互动，预测突变事件仍困难；模型误差可能随政治立场或人口群体差异；预测结果易被用于不良定向广告或内容审查等滥用。

---

## 32. HighTide: An Agent-Curated Open-Source VLSI Benchmark Suite

**arXiv ID:** 2606.04126 | [PDF](https://arxiv.org/pdf/2606.04126v1)

**作者:** Benjamin Goldblatt `[一作]` (University of California, Santa Cruz), Matthew R. Guthaus `[通讯]` (University of California, Santa Cruz)

**通讯引用:** 5655 | [OpenAlex ID](https://openalex.org/A5065218759)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可演化的 AI 辅助 VLSI 基准套件 HighTide，涵盖多种 HDL、三种技术节点并提供完整 RTL‑to‑GDS 流程与远程缓存；

**💡 创新点**

提出 AI 驱动的设计策划与调优技能、跨平台可追踪的子模块、持续集成与决策记忆机制，填补传统基准缺乏多样性和及时更新的空白；

**🔧 技术方法**

使用 Bazel+ORFS 的增量编译与远程缓存、Kubernetes 分布式执行、Claude Code 12 个技能、抽象 SRAM 模型、跨平台技术迁移与共享决策日志；

**📊 数据集**

集合数十个 SoC 组件设计（CPU、ML 加速器、NoC、GPU、加密等），覆盖 30% CPU 与 70% 其他功能，跨 ASAP7、NanGate45、SkyWater 130nm 芯片节点；

**📈 对比分析**

通过标准化 RTL‑to‑GDS 流程产生 GDS 与 QoR 报告，对比 ORFS、EDALearn 等传统基准，展示更丰富的设计与结构分布，并能生成可复现的 PPA 与时序闭合数据（具体数值未给出）；

**⚠️ 局限性**

依赖 ORFS 工具链导致工具限制；抽象 SRAM 模型带来 PPA 估算误差；验证体系不统一，仅覆盖 DRC/LVS/STA；更新与维护仍需手动干预；缺乏统一的后流程验证与硅级验证。

---

## 33. StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis

**arXiv ID:** 2606.04246 | [PDF](https://arxiv.org/pdf/2606.04246v1)

**作者:** Prashanth Vijayaraghavan `[一作]` (IBM Research), Vandana Mukherjee `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并训练一个面向RTL生成的多阶段框架，利用逐步推理轨迹与奖励模型提高代码功能正确性与推理可解释性。

**💡 创新点**

提出StepPRM-RTL，将步骤级过程奖励与MCTS搜索结合，提供密集、语义化的中间反馈，解决长程依赖与稀疏奖励问题。

**🔧 技术方法**

步骤级奖励模型（StepPRM）、蒙特卡洛树搜索（MCTS）、检索增强微调（RAFT）、LLM生成的步骤轨迹与结构化对齐。

**📊 数据集**

Verilog‑Eval与VHDL‑Eval两大公开基准，以及内部RTL‑IR语料库。

**📈 对比分析**

与提示式、微调式、RAG式基线对比，Pass@1分别提升至0.857/0.786、推理精度超80%，显著优于所有对手。

**⚠️ 局限性**

对多文件层次设计支持有限，奖励模型仍依赖结构对齐与手工定义的对齐评分，跨架构迁移需进一步验证。

---

## 34. Affordance2Action: Task-Conditioned Scene-level Affordance Grounding for Real-Time Manipulation

**arXiv ID:** 2606.04172 | [PDF](https://arxiv.org/pdf/2606.04172v1)

**作者:** Litao Liu `[一作]` (Rutgers University New Brunswick), Jingjin Yu `[通讯]` (Rutgers University New Brunswick)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了基于任务条件的场景级 affordance grounding 框架 A2A，包含 A2A-Bench 数据集、agent‑assisted annotation pipeline A2A-AffordGen、实时 grounding 模型 A2A‑GroundingModel 以及用于机器人操纵的 A2A‑Policy。

**💡 创新点**

创新点包括：① 关注 one‑to‑many 任务条件 affordance 标注；② 通过语言模型过滤+交互分割+mask‑out 精细化的 agent‑assisted annotation pipeline；③ 在 SAM3 基础上加入文本条件视觉提示注入实现实时任务条件 grounding；④ 将 grounding 结果作为可解释空间先验应用于 diffusion‑policy。

**🔧 技术方法**

采用的大型语言模型过滤、交互式分割、mask‑out 细化、文本条件视觉提示注入、SAM3、LoRA 适配器、Diffusion Policy 等技术。

**📊 数据集**

使用真实多物体自然场景构建的 A2A‑Bench（单实例/多实例），并与 RefCOCO/+/g、LIBERO‑object 10 任务以及 4 项真实世界任务进行对照。

**📈 对比分析**

与 SAM3+text、GroundingDINO+SAM3、VLM+SAM3、SegAgent、LISA‑7B 等基线比较，在单实例下 gIoU 81.9%，多实例下 sIoU 90.6%，TRPS grounding gIoU 55.4%，均显著优于现有方法；在 policy 上 A2A‑Explicit 将成功率从 87.8% 提升到 94.4%，真实世界平均 26/40。

**⚠️ 局限性**

局限性：仅在平面操纵场景评估，未验证移动/全身控制；缺乏闭环动态 affordance 更新，受遮挡、运动和接触扰动影响。

---

## 35. Computational conceptual history of scientific concepts: From early digital methods to LLMs

**arXiv ID:** 2606.04118 | [PDF](https://arxiv.org/pdf/2606.04118v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 36. Inverse Critical Experiment Design via Gradient Optimization and a Multigroup Attention-Based Neural Network Architecture

**arXiv ID:** 2606.04033 | [PDF](https://arxiv.org/pdf/2606.04033v1)

**作者:** Will Savage `[一作]` (Massachusetts Institute of Technology), Dean Price `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 235 | [OpenAlex ID](https://openalex.org/A5075369934)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用深度神经网络代理与梯度优化方法实现了核临界实验的逆向设计。

**💡 创新点**

创新点在于提出多组注意力池化层以捕捉不同能量组的空间敏感性，并将此网络与U‑Net结合用于敏感性预测。

**🔧 技术方法**

采用了U‑Net编码器、跨组注意力池化、1D残差回归器、AdamW优化器以及softmax直通估计等技术。

**📊 数据集**

构建了约5,000个基于细胞自动机生成的二维实验网格及其OpenMC计算的敏感性数据集。

**📈 对比分析**

与传统最大/平均池化对比，注意力池化的平均绝对误差从48.27 pcm降至41.19 pcm；优化后实验的c_k最高可达0.97757，显著高于现有实验。

**⚠️ 局限性**

方法仅优化c_k，未考虑k_eff、构造可行性或材料成本，并局限于二维网格，需进一步加入约束与三维扩展。

---

## 37. What Makes Majority Illusion Easy to Detect?

**arXiv ID:** 2606.04260 | [PDF](https://arxiv.org/pdf/2606.04260v1)

**作者:** Šimon Schierreich `[一作]` (Czech Technical University in Prague), Ildikó Schlotter `[通讯]` (ELTE Centre for Economic and Regional Studies)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5070244460)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了在社交网络中检测多数幻觉（majority illusion）问题的计算复杂性，系统评估了多种结构参数（如树宽、顶点完整度、反馈边集、距离到星/路径/聚类图等）对问题可解性的影响，并给出了完整的可解性图谱。

**💡 创新点**

创新点包括：
- 首次使用 N‑fold 整数规划（N‑fold IP）框架对顶点完整度参数下的可解性进行证明，显著提升了算法效率；
- 通过构造复杂的可约证明，确立了距离到离散星、路径、树宽等参数下的 W[1]‑难度，解决了此前未解的关键开放问题；
- 对密集网络的参数化（如距离到离散团）提供了既可解又有下界的结果，展示了不同结构对可解性的细粒度影响。

**🔧 技术方法**

使用的主要技术手段包括：
- 参数化复杂度理论（FPT、XP、W[1]‑难度证明）；
- N‑fold integer programming 与动态规划相结合的算法设计；
- 基于图分解（树宽、树深度、集群边删数等）的分层 DP；
- 结构变换与 gadget 构造的多维子集和归约。

**📊 数据集**

本工作为理论论文，未使用公开数据集进行实验评估，所有结果均基于形式化证明与理论复杂度分析。

**📈 对比分析**

与已有工作比较，本文在多种参数下显著降低了算法的时间复杂度（如从 2^{2k}·n 下降到 2^{k}·n^{O(1)}），并为此前只给出粗略上界的参数提供了精确的上、下界匹配；性能方面以大 O 表示的理论运行时间给出。

**⚠️ 局限性**

主要限制与未解之问：
- 对于 twin‑cover 参数的可解性仍未确定，论文仅指出其介于聚类图删除数与集群边删数之间；
- 论文聚焦于理论复杂度，缺乏实际数据上的实验验证；
- 复杂度结果多为 worst‑case 评估，实际网络中表现可能更好，但未给出实证分析。

---

## 38. FindIt: A Format-Informed Visual Detection Benchmark for Generalist Multimodal LLMs

**arXiv ID:** 2606.04282 | [PDF](https://arxiv.org/pdf/2606.04282v1)

**作者:** Eshika Khandelwal `[一作]` (University of Tuebingen), Hilde Kuehne `[通讯]` (University of Tuebingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评测了针对多模态大型语言模型的定位任务基准，涵盖四类任务（目标检测、指代表达检测、实例检测、视频检测）。

**💡 创新点**

首次提出统一框架，标准化输入、解析方式与评估协议，并系统研究不同边界框格式与输出格式对模型定位性能的影响。

**🔧 技术方法**

使用提示式定位、BBox 解析、Hungarian 匹配、mIoU、F1@0.5 等指标进行评估。

**📊 数据集**

使用 Pascal VOC、OpenImages、iGround、RefCOCO/+/g、RefL4、D3、PhraseCut、Flickr30k Entities、Synthetic Visual Genome、HR-InsDet、RoboTools、iGround 视频等 13 个公开数据集。

**📈 对比分析**

通过对比多源开源与专有 MLLM（如 Qwen3‑VL、Gemma‑4、GLM‑4.6V、GPT‑5.4、Claude Sonnet 4.5、Gemini 2.5 Flash）在最佳格式下的平均 F1@0.5，发现 GLM‑4.6V 取得最高分，其余模型在单/多目标检测中表现相似。

**⚠️ 局限性**

局限在于只评估最佳格式，对格式鲁棒性未给出分数；未覆盖离群分布与特定领域场景；以及对模型输出规范的推断依赖。

---

## 39. CADET: A Modular Platform for Evaluating Distributed Cooperative Autonomy in Connected Autonomous Vehicles

**arXiv ID:** 2606.04072 | [PDF](https://arxiv.org/pdf/2606.04072v1)

**作者:** Pragya Sharma `[一作]` (University of California Los Angeles), Mani Srivastava `[通讯]` (University of California Los Angeles)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 CADET，一个可模块化、可跨车辆/基础设施/云端部署的合作自动驾驶实验平台，用于在真实网络与计算资源条件下可复现地评估分布式合作自动驾驶系统。

**💡 创新点**

创新点在于将自动驾驶栈解耦成可组合模块，配合 NetWaggle 网络仿真层实现设备异构与网络时延的可控重放，提供统一的多层度量（模型、系统、任务），并通过单一 YAML 配置文件实现实验的可复现与可扩展。

**🔧 技术方法**

核心技术包括：Python/Mininet 结合 WebSocket 的分布式推理框架；CARLA/Scenario Runner 进行仿真；多模型接口（YOLO、MPC 等）；网络层面使用 Mininet 注入 4G/5G/WiFi 延迟/抖动；统一调度器与多层日志同步；支持硬件回环与云端 GPU 调度。

**📊 数据集**

实验数据集涵盖：CARLA 合成场景、公开交通数据（Waymo、COCO、VQA2）、以及在真实设备上采集的传感器流；同时利用 V2V/V2I 流量记录进行网络时延模型。

**📈 对比分析**

通过对比 V2V 意图包、云端感知、RSU 辅助感知四种部署，实验显示 V2V 意图包在高延迟/尾部时仍能保持安全（无碰撞），云端感知在 100 ms 延迟下出现碰撞；RSU 辅助感知在遮挡下保持正安全裕度，但在高并发负载下会因计算饱和导致安全裕度下降。实验采用统一的碰撞率、TTC/TTE、延迟分布等指标，表明分布式部署选择对安全性影响大于模型精度提升。

**⚠️ 局限性**

局限性包括：平台主要针对 1-2 车辆仿真，规模化多车辆部署尚未系统验证；依赖 Mininet 的网络仿真无法完全复现真实无线链路的复杂干扰；当前对大型基础模型（如 VLM）仅做推理评估，缺乏安全关键时延优化方案；在极端网络失效或极低算力边缘节点下的鲁棒性仍需进一步研究。

---

## 40. Stumbling Into AI Emotional Dependence: How Routine AI Interactions Reshape Human Connection

**arXiv ID:** 2606.04150 | [PDF](https://arxiv.org/pdf/2606.04150v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 41. Efficient and Training-Free Single-Image Diffusion Models

**arXiv ID:** 2606.04299 | [PDF](https://arxiv.org/pdf/2606.04299v1)

**作者:** Haojun Qiu `[一作]` (University of Toronto), David B. Lindell `[通讯]` (University of Toronto)

**通讯引用:** 3297 | [OpenAlex ID](https://openalex.org/A5004550709)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无须的单图像扩散模型，通过闭式去噪推导的局部补丁去噪器实现图像生成、编辑与重目标化等多种任务

**💡 创新点**

核心创新在于利用单张图像有限补丁集合，直接计算闭式去噪函数（相当于每个补丁的最佳均方误差估计），从而完全消除对神经网络训练的需求；并将其嵌入多尺度逆扩散框架，兼具传统补丁方法的可解释性与扩散模型的概率建模优势

**🔧 技术方法**

技术主要包括：闭式去噪推导、非局部均值/高斯加权重组、多尺度逆扩散（粗到细）、融合FlashAttention/稀疏注意力、潜在空间扩散、近似最近邻加速、CLIP文本引导、对称/可拼贴约束等

**📊 数据集**

使用单张输入图像本身的补丁集合（不同尺度）作为训练/推理数据，实验中主要针对约20-250×250像素的图像，也展示了1GP级别超高分辨率（14336×70080）生成

**📈 对比分析**

与 SinGAN、SinDDM、SinFusion、SinDiffusion、GPNN、GPDM 等最先进单图像生成模型对比；在SIFID、NIQE、NIMA、MUSIQ、LPIPS、多样性等指标上表现相当或更优；显著缩短训练时间（0小时），推理速度快到毫秒级；同时能生成高质量大尺度图像（秒级）

**⚠️ 局限性**

局限性包括：对高频细节仍受补丁重建方式影响，可能在极大尺度或复杂纹理时出现局部不一致；缺乏对全局语义约束的自适应控制；在多图像或跨域学习时需改进；对文本引导的质量受CLIP模型限制

---

## 42. SocialCoach: Personalized Social Skill Learning with RL-based Agentic Tutoring and Practice

**arXiv ID:** 2606.04155 | [PDF](https://arxiv.org/pdf/2606.04155v1)

**作者:** Tianfu Wang `[一作]` (Hong Kong University of Science and Technology), Qi Zhang `[通讯]` (Microsoft)

**通讯引用:** 79228 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于LLM的Agentic教学系统SocialCoach，包括从书籍到案例到场景的知识语料构建、RL优化的自适应练习调度、沉浸式模拟与因果诊断的评估以及知识驱动的反思辅导，并在EQoach产品中上线。

**💡 创新点**

创新点在于：①提出了理论→案例→场景的三层知识框架并通过多代理LLM管道实现自动化构建；②将练习调度建模为MDP，在用户模拟环境中用RL学习处方-检索-适配策略，解决冷启动与可解释性；③设计了基于因果归因的评估与Socratic反思，闭合“知行”鸿沟。

**🔧 技术方法**

技术包括：大型语言模型（GPT‑5、Qwen3等）、多代理LLM流程、强化学习（多轮Agentic RL + ROLL）、语义检索与标签化、场景模拟对话、知识检索与可追溯反馈、Socratic问答生成。

**📊 数据集**

数据集为：从Amazon、Open Library等来源筛选的200本社交技能书籍（约40k条理论/案例/场景条目），以及由SocioVerse和PersonaHub生成的约1000个代表性合成学习者配置；使用LLM生成的用户模拟行为和奖励。

**📈 对比分析**

与多种基线（零样本LLM、检索式、未调优LLM）进行对比，评估指标包括平均参与度、学习收益、个性化与进阶连贯度，SocialCoach在所有四项评测中均显著高于基线；在10次练习的用户研究中，用户对参与感和实用性评价居高。

**⚠️ 局限性**

局限性在于：1）依赖LLM的生成与评估可能产生幻觉或不一致；2）模拟环境与真实人类交互存在差距，RL策略的迁移效果未知；3）知识库来源主要为书籍，缺乏最新行业实践；4）大规模数据标注与多模态验证仍需进一步完善。

---

## 43. The Invisible Lottery: How Subtle Cues Steer Algorithm Choice in LLM Code Generation

**arXiv ID:** 2606.04057 | [PDF](https://arxiv.org/pdf/2606.04057v1)

**作者:** Akanksha Narula `[一作]` (Max Planck Institute for Software Systems), Laurent Bindschaedler `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 364 | [OpenAlex ID](https://openalex.org/A5051818971)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过 46,535 次实验，系统研究了大型语言模型在代码生成任务中，提示上下文如何影响所选算法（即算法 steering）并评估其对性能、可靠性的影响。

**💡 创新点**

提出并量化了“隐形彩票”现象，发现非算法性线索可导致高达 100pp 的算法偏差，并证明直接显式指定算法是缓解此问题的最可靠方法。

**🔧 技术方法**

利用 AST 语法树分类器构建算法族识别器，设计多渠道提示（语义与表面），在 15 种模型/温度设置下进行大规模实验并进行统计效应分析。

**📊 数据集**

构造了 11 个多算法基准任务（8 个经典算法 + 3 个生产场景），并使用自定义单元测试（如 Fibonacci 36、深递归等）与 HumanEval 子集进行评估。

**📈 对比分析**

通过算法族分布、最大最小偏移（pp）以及通过率比较，发现不同模型/温度平均产生 40–50pp 的算法偏移；当直接指定算法时通过率提升至 100%，而暗示性提示导致的可靠性下降显著。

**⚠️ 局限性**

受限于任务覆盖面有限、模型更新可能改变行为、提示设计不完全真实、分类器准确率约 87–88% 以及仅检验功能正确性缺乏规模性评估。

---

## 44. Characterizing initial human-AI proof formalization workflows

**arXiv ID:** 2606.04273 | [PDF](https://arxiv.org/pdf/2606.04273v1)

**作者:** Katherine M. Collins `[一作]` (Massachusetts Institute of Technology), Ilia Sucholutsky `[通讯]` (New York University)

**通讯引用:** 972 | [OpenAlex ID](https://openalex.org/A5053200092)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过问卷调查与受控实验，探究人工智能在形式化定理证明工作流中的使用方式、偏好与实际效果。

**💡 创新点**

创新点在于聚焦人机协作模式而非单纯性能对比，揭示使用者希望保持高层控制、采用多工具组合的行为，并首次量化 AI 辅助对形式化准确率的提升。

**🔧 技术方法**

使用了 Lean 形式化环境、GitHub Copilot、ChatGPT、Claude、LeanSearch 等多种 AI 工具，并结合问卷编码、线性混合效应模型等统计方法。

**📊 数据集**

数据集包括 31 名问卷受访者的自述数据，以及 7 名实验参与者在 6 个不同难度问题上的形式化过程视频与时间记录。

**📈 对比分析**

通过对比 AI 辅助与无工具两种情境下的形式化准确率与耗时，发现 AI 可使准确率提升约 30%，但对耗时无显著统计差异，表明工具多样化可提升效果。

**⚠️ 局限性**

局限性包括样本量小、仅覆盖部分数学领域、实验时间有限、对 AI 工具的可靠性与风格适配性关注不足，以及未覆盖真实的证明发现过程。

---

## 45. Generalizable Multi-Task Learning for Wireless Networks Using Prompt Decision Transformers

**arXiv ID:** 2606.04328 | [PDF](https://arxiv.org/pdf/2606.04328v1)

**作者:** Fatih Temiz `[一作]` (University of Ottawa), Melike Erol-Kantarci `[通讯]` (University of Ottawa)

**通讯引用:** 8023 | [OpenAlex ID](https://openalex.org/A5089891162)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于 Prompt 决策 Transformer 的多任务学习框架，用于 CoMP 系统中的多基站选择与资源分配。

**💡 创新点**

创新点在于将多任务学习与 Prompt 机制结合，利用任务特定的轨迹 Prompt 实现无监督的少样本快速适配；同时采用序列建模解决离线强化学习的 NP 难题。

**🔧 技术方法**

使用的技术包括 Prompt 决策 Transformer、离线强化学习、序列建模、跨任务的掩码交叉熵损失以及多尺度网络配置。

**📊 数据集**

数据集来源于基于 mobile‑env 的仿真平台，生成 12 个不同规模与调度策略的 CoMP 任务的离线轨迹。

**📈 对比分析**

与传统单任务 PPO 基线对比，PromptDT 在所有规模下均表现更好，尤其在大规模场景中提升 25‑30%；在未见任务上也能以少量样本逼近甚至超过 PPO 的性能。

**⚠️ 局限性**

局限性包括需依赖大量离线轨迹、在任务数过多时可能出现梯度冲突导致泛化下降、仅在仿真环境验证，缺乏真实网络实验。

---

## 46. Variance Reduction for Heavy-Tailed Monetization Metrics in Ranking Experiments via Post-Stratification

**arXiv ID:** 2606.04110 | [PDF](https://arxiv.org/pdf/2606.04110v1)

**作者:** Neeti Pokharna `[一作]` (ShareChat), Aleksei Ustimenko `[通讯]` (Simulacra Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在ShareChat和Moj上，针对直播平台的 GMV（即商家收入）指标，作者提出并实现了一种基于预实验协方差调整（CUPED）与后分层（post‑stratification）的在线实验框架，显著降低了指标的方差并提升实验统计功效。

**💡 创新点**

创新点在于将后分层与CUPED结合，利用预期的用户行为层级（尤其是高价值“鲸鱼”用户）对指标进行分层加权，既保持了业务可解释性，又实现了 >99% 的方差压缩，且无需额外流量即可达到原始指标约 45% 的统计置信度。

**🔧 技术方法**

技术核心包括：Winsor化去除极端离群点；基于30天历史 GMV 的分层（尾部 vs 非尾部）；每层内部使用 CUPED 进行协变量校正；全平台加权求平均与方差；并通过 z‑检验评估显著性。

**📊 数据集**

数据集为 ShareChat 直播平台的 GMV（千级用户以上）与预实验 GMV，涉及 40+ 真实在线实验，每次实验规模均超过 1 百万用户，覆盖不同地区与时间段。

**📈 对比分析**

与传统原始 GMV 以及仅使用 CUPED 的比较显示，后分层＋CUPED 在相同流量下将方差降低至 <1%，对应的最小可检测效应（MDE）从 ~136% 降至 ~10%，实现了约 1.85 倍的样本量提升，Type‑I 错误率略高（约 6%）但仍在可接受范围。

**⚠️ 局限性**

局限性包括：若实验目标是只提升尾部高价值用户，后分层会弱化其效应；需要预实验数据和准确的分层阈值；在极端流量或样本量不足时，尾部分层的方差估计可能导致 Type‑I 错误略增；此外，分层设计需保持冻结以防止数据驱动的 p‑hacking。

---

## 47. MorphoQuant: Modality-Aware Quantization for Omni-modal Large Language Models

**arXiv ID:** 2606.04349 | [PDF](https://arxiv.org/pdf/2606.04349v1)

**作者:** Yue Wu `[一作]`, Yansong Tang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对全模态大型语言模型提出了一套后训练量化框架 MorphoQuant，通过分离激活分布中的长尾离群点并将其吸附到通道偏置中，实现 4 位权重和激活的高效压缩。

**💡 创新点**

创新点包括：① Distribution‑Aware Bias Compensation (DABC)，利用通道离散度指标动态识别并将极端离群点转换为偏置补偿；② Morphology‑Directed Quantization Function Optimization (MDQFO)，在 DABC 的基础上联合优化量化网格与偏置掩码，并引入结构化复合损失保证密集内点的精细重建和跨模态语义保持。

**🔧 技术方法**

主要技术为后训练量化（PTQ）框架、通道级离散度评估、偏置补偿、协同阈值搜索以及基于 ℓp‑norm 与余弦相似度的复合损失；硬件实现采用纯 4 位密集矩阵乘加与稀疏偏置校正的分离执行。

**📊 数据集**

使用 Qwen2.5‑Omni（3B）模型，评估数据集涵盖 ScienceQA、MMMU、Video‑MME 与 AIR‑Bench，覆盖文本、图像、视频、音频四大模态。

**📈 对比分析**

与现有 PTQ 方法（AWQ、QLoRA、Q‑VLM 等）对比，W4A4 版本的 MorphoQuant 在多模态基准上显著优于同位宽方法，并在多数任务上甚至超过 W4A16 QLoRA，证明了在 4 位硬件上可实现接近或超越 16 位精度的推理效果。

**⚠️ 局限性**

局限性包括：① 需要手动设定通道阈值和损失权重，可能对不同模型或数据分布产生敏感性；② 仅在极小的校准集（128 样本）上搜索，极端情况下可能出现过拟合或失效；③ 对极端多模态模型的推广尚待进一步验证，尤其是更大规模或不同结构的 OLLM。

---

## 48. Dead Science Walking: Publication Bias and the AI Scientist Pipeline

**arXiv ID:** 2606.04220 | [PDF](https://arxiv.org/pdf/2606.04220v1)

**作者:** Kargi Chauhan `[一作]` `[通讯]` (University of California Santa Cruz), Kargi Chauhan (University of California Santa Cruz)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文分析了AI科学家系统在训练语料中存在的正向结果偏差，量化了“空结果缺口”Δ，并探讨了检索、生成和评估环节如何放大这一偏差，提出四类失败模式及三项治理干预。

**💡 创新点**

创新点在于提出空结果缺口Δ与放大指数α_1·α_2·α_3的概念化框架，系统性分类了AI科学家可能产生的“自信再发现”“幽灵证据累积”“复制洗衣”和“置信失准”四种治理失效模式，并给出可操作的三层治理策略。

**🔧 技术方法**

采用数学定义与假设建模、系统性文献综述以及基于现有复现研究的经验估计来构建评估框架；通过对检索、生成、评估三个环节的偏差系数进行粗略量化，形成放大指数。

**📊 数据集**

主要参考了Open Science Collaboration的心理学复现率、药物发现与肿瘤学的复现研究以及Cancer Biology的Reproducibility Project数据，未使用单一公开数据集，而是基于这些研究报告给出Δ的粗略估计。

**📈 对比分析**

论文并未在实验上对比不同系统，而是用理论放大指数与已知的复现率进行对照，指出即使在保守估计下也能导致数倍的偏差放大；相比现有文献，其贡献在于阐明了偏差放大的机制与治理路径。

**⚠️ 局限性**

局限性包括：Δ和α_i的估计为第一阶粗略值，未进行经验校准；假设检索、生成、评估三段独立，实际可能高度相关；仅针对部分已复现率较低的学科，未涵盖所有科学领域；缺乏实证验证和量化性能评测。

---

## 49. Unpredictable Safety: Domain-Dependent Compliance and the Transparency Gap in Open-Weight LLMs

**arXiv ID:** 2606.04035 | [PDF](https://arxiv.org/pdf/2606.04035v1)

**作者:** Zacharie Bugaud `[一作]` `[通讯]` (Astera Institute), Zacharie Bugaud (Astera Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对七个伦理领域的七项标准化实验进行 4200 次交互，测量五个开源大语言模型在分析与操作情境下的合规率，揭示合规率存在 71pp 范围且与领域高度相关。

**💡 创新点**

首次量化域依赖安全行为，发现“技术框架绕过”现象、模型安全层级与多层级异质性，为安全评估提出按领域报告的需求。

**🔧 技术方法**

采用双框架实验设计（分析 vs 操作），LLM 判别器评分，聚类自举置信区间；使用 5 个开放权重模型（Gemma 3、Qwen 3、Mistral Nemo、Llama 3.3、DeepSeek R1）以及闭源复制（GPT‑4.1/5.2、Claude Haiku/​Sonnet/​Opus）。

**📊 数据集**

使用 7 个伦理领域内 20 个子域，共 140 方案（每个方案两种框架、三次复制），生成 4,200 条交互记录。

**📈 对比分析**

与 TruthfulQA、AdvBench、HarmBench 等前沿基准对比，单一安全分数掩盖域差异；闭源系统复制保持相同域排序，显示现象稳健，模型间合规率差距显著，最高 85.7%、最低 14.7%。

**⚠️ 局限性**

限制包括：评估基于 LLM 判别器而非人工评分、仅 7 个领域、单轮请求、无系统提示、量化模型、仅基于美式法律框架、未拆解内部机制、样本量有限等。

---

## 50. Characterizing Online Criticism of Partisan News Media Using Weakly Supervised Learning

**arXiv ID:** 2606.04289 | [PDF](https://arxiv.org/pdf/2606.04289v1)

**作者:** Karthik Shivaram `[一作]` (Tulane University), Aron Culotta `[通讯]` (Tulane University)

**通讯引用:** 5659 | [OpenAlex ID](https://openalex.org/A5011178134)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种弱监督学习方法，用于识别对党派新闻媒体的批评性推文，并在此基础上分析其随时间、用户与媒体源的变化趋势。

**💡 创新点**

创新点在于构建专门针对媒体批评的分类器，利用用户历史行为与文本关键词两类噪声标签函数，并通过标签去噪与多模态融合提升准确性。

**🔧 技术方法**

采用弱监督技术（Labeling Function、Dawid‑Skene 等去噪），结合用户特征网络、文本网络与融合网络的神经分类器，训练时使用RoBERTa预训练模型。

**📊 数据集**

使用包含 522 家新闻源、3.5M 条涉及这些源的推文（覆盖 10 年、5,470 名活跃用户）的自建数据集，并标注了约 1.2M 条可用于训练的样本。

**📈 对比分析**

与单一标签函数训练相比，使用 Dawid‑Skene 软标签的文本网络在 ROC‑AUC 上达 0.840，较基线提升约 3%（最高 0.810 的 ϕ_un 标签函数）。

**⚠️ 局限性**

局限包括样本偏倚（仅关注政治新闻关注者）、新闻源立场静态假设、用户多样性不足、以及分类器误差对后续分析的潜在影响。

---

## 51. GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs

**arXiv ID:** 2606.04184 | [PDF](https://arxiv.org/pdf/2606.04184v1)

**作者:** Weidong Tang `[一作]` (Xidian University), Wangbo Zhao `[通讯]` (National University Of Singapore)

**通讯引用:** 815 | [OpenAlex ID](https://openalex.org/A5081255558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个多模态群体层面理论心智（GroupToM-Bench）基准，用240个专家设计情景和7级认知审计框架评估多模态大语言模型的群体推理能力。

**💡 创新点**

创新点在于：①首次将群体层面理论心智纳入多模态评估；②设计了跨三层结构（微观BDI→中观群体张力与结构约束→宏观群体结果）的因果链；③提出7级认知审计框架，系统覆盖从个体心理状态到群体非线性结果的全过程。

**🔧 技术方法**

技术主要包括：多模态输入（对话文本+场景图像）+基于贝叶斯信念图的BDI建模；使用7级多层任务（多选与开放式）评估模型；采用LLM-as-judge（GPT‑5）对开放式答案进行0-100分评分；通过实验对比11种公有与开源多模态大语言模型。

**📊 数据集**

数据集由240个专家编写情景构成，包含多轮对话、隐私心理状态、社会结构约束及相应的7级任务，共计3000+推理题；每个情景还配有单张全景图像，展示面部表情与身体姿态。

**📈 对比分析**

比较方法：对每个模型在7级任务上分别计算精确匹配准确率（多选）或GPT‑5评分（开放式）。实验显示：在个体层面（L1–L3）模型表现相对较好，但在群体层面（L4–L6）准确率急剧下降，平均落后于人类基准约24%，形成明显的“群体认知缺口”。

**⚠️ 局限性**

局限性：①多模态依赖不均衡，部分样例可仅凭文本推断；②模型可能因安全对齐抑制对群体失误的模拟；③情景多聚焦西方文化，跨文化通用性待扩展。

---

## 52. Puffin-Backed Vector Indexes: Attaching Approximate Nearest Neighbor Indexes to Apache Iceberg Snapshots for Compute-Disaggregated Query Engines

**arXiv ID:** 2606.04196 | [PDF](https://arxiv.org/pdf/2606.04196v1)

**作者:** Artur Borycki `[一作]` `[通讯]` (Teradata Advanced Research), Artur Borycki (Teradata Advanced Research)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现将分布式ANN索引嵌入Apache Iceberg的Puffin文件中，实现与表快照版本同步的索引生命周期

**💡 创新点**

利用Iceberg的Puffin机制作为分布式索引存储和生命周期管理，避免新增生命周期，保持无状态执行器

**🔧 技术方法**

Iceberg、Puffin、Vamana/DiskANN、DuckDB、FlockDB、分布式调度、HTTP Range读取、PQ量化等技术

**📊 数据集**

1亿条768维向量表（约300 GB），共10 000个Parquet文件

**📈 对比分析**

通过三阶段分布式查询与增量刷新，查询时间比无索引快数百倍，Recall@100在0.95–0.99之间，构建耗时约45–60 min

**⚠️ 局限性**

索引更新受快照提交频率限制，独立分片导致回忆率依赖分片分布，三阶段查询产生latency壁垒，冷缓存受对象存储吞吐限制

---

## 53. When Does Structure Help? The Information Bonus of AlphaFold2 Representations over Protein Language Models

**arXiv ID:** 2606.04228 | [PDF](https://arxiv.org/pdf/2606.04228v1)

**作者:** Kargi Chauhan `[一作]` `[通讯]` (University of California, Santa Cruz), Kargi Chauhan (University of California, Santa Cruz)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了信息加成（Information Bonus, IB）指标，用于评估单序列 AlphaFold2 的几何表示在蛋白质任务中相对于 ESM‑2 语言模型是否能提供额外的可线性利用信息。

**💡 创新点**

创新点在于：①定义了任务级的可量化度量 IB，使模型选择成为可测量的决策；②发现结构信息在绑定亲和力与二值柔性上不利，而在全局通讯驱动的旁分子活性中具有正向信息优势；③揭示了残基级交叉验证会导致 RMSF 性能显著上升的泄漏问题。

**🔧 技术方法**

采用了冻结的线性探针（ridge 回归/逻辑回归）对 AlphaFold2 Evoformer 单体与对角对偶表示、以及 ESM‑2 650M 嵌入进行评估；使用 5 折蛋白质级 GroupKFold 进行交叉验证。

**📊 数据集**

数据集包括：PDBbind（5680 个蛋白‑配体复合物）用于亲和力回归；ATLAS（268 个蛋白，50426 个残基）用于连续 RMSF 回归和二值柔性分类；AlloSigDB（47 个蛋白，9925 个残基）用于旁分子活性位点分类。

**📈 对比分析**

比较方法是对每个任务使用同一线性探针对所有表示进行评估，IB 计算为 AF2 最佳表示的 held‑out 指标减去 ESM‑2 指标。结果显示：ESM‑2 在亲和力（Pearson r 0.449）和二值柔性（AUROC 0.824）上优于 AF2；AF2 单体表示在旁分子活性（AUROC 0.548，IB +0.064）上表现突出；RMSF 回归在残基级拆分时会高达 27–39% 的性能膨胀。

**⚠️ 局限性**

限制包括：旁分子活性数据集规模小且不平衡；只使用单序列 AF2 可能低估全 MSA AF2 的优势；固定正则化参数未针对每种表示调优；线性探针只能评估线性可提取信息，非线性微调可能改变结果。

---

## 54. The Differentiable Auditory Loop (DAL): An ML Framework for Hyper-Personalized Hearing Aids

**arXiv ID:** 2606.04103 | [PDF](https://arxiv.org/pdf/2606.04103v1)

**作者:** Alejandro Ballesta Rosen `[一作]` (Google Research Australia), Simon Carlile `[通讯]` (Google Research Australia)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出可微分听觉循环框架（DAL），通过CARFAC耳蜗模型与SEANet卷积网络端到端训练，实现针对听力损失的个性化听力辅助。

**💡 创新点**

创新点在于将可微分耳蜗模型CARFAC作为目标损失，利用神经活动模式（NAP）和稳定听觉图像（SAI）进行训练，既补偿听力编码失真又实现低延迟的SEANet实现；同时提供多种损失函数与神经网络结构的组合。

**🔧 技术方法**

使用JAX/FLAX实现可微分CARFAC、Waveform‑to‑Waveform SEANet卷积网络、基于NAP/SAI的L1/SSIM/PN/Hybrid损失函数、梯度下降训练与TFRecord交错批处理流水线。

**📊 数据集**

采用LibriSpeech（train‑clean‑100）10,265个4秒语音片段，随机混合白噪声（SNR-5~10 dB）进行训练与评估。

**📈 对比分析**

与传统Master Hearing Aid（MHA）（基于NAL‑NL2或DAL训练）及无处理基线比较，评估NAP、SAI的L1、Pearson相关系数和SI‑SDR；结果表明DAL‑优化SEANet在NAP和SAI域均优于MHA和基线，显著提升神经信号结构与声学质量。

**⚠️ 局限性**

仅验证了轻度OHC损伤的单一模型，缺乏临床听感评估与多语言/口音多样性；噪声抑制与听感权衡未确定，可能对正常听者不友好。

---

## 55. LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding

**arXiv ID:** 2606.04302 | [PDF](https://arxiv.org/pdf/2606.04302v1)

**作者:** Haocheng Xia `[一作]` (University of Illinois Urbana-Champaign), Yongjoo Park `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 506 | [OpenAlex ID](https://openalex.org/A5023168280)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LazyAttention，一种延迟位置编码的注意力机制，实现 KV 缓存的零拷贝、位置无关重用；

**💡 创新点**

创新点在于将旋转位置编码（RoPE）从 KV 缓存中解耦，采用 kernel 化的 deferred 编码，仅在 attention 计算时动态调整位置，既不复制缓存也不额外占用显存；

**🔧 技术方法**

使用了自研的 Triton kernel（prefilling 与 decoding 两种），结合 vLLM、FlashAttention，支持 RoPE 及其变体、ALiBi、GQA/MQA 等相对位置编码；

**📊 数据集**

主要在 RAG 基准上评估：2WikiMQA、HotpotQA、TriviaQA、NarrativeQA，并在 Llama‑3.1‑Tulu‑3‑8B、Llama‑3.1‑70B、Qwen3‑8B 等模型上测试；

**📈 对比分析**

与 PromptCache、CacheBlend、Block‑Attention、Prefix Caching、MEPIC 等基线对比，LazyAttention 在 skewed 访问场景下 TTFT 下降 1.37×、吞吐量提升 1.40×，在所有显存预算下 hit‑ratio 均优于基线，保持近乎相同的生成质量；

**⚠️ 局限性**

局限性：对极端长上下文仍需进一步优化；在不使用相对位置编码的模型或需要全局绝对编码的场景下适用性有限；

---

## 56. Answer Self-Consistency with Margin-Triggered Question Re-Arbitration for the CVPR 2026 VidLLMs Challenge

**arXiv ID:** 2606.04323 | [PDF](https://arxiv.org/pdf/2606.04323v1)

**作者:** Tomoya Miyazawa `[一作]` (Data Analytics Labo Company), Hiroyasu Okuno `[通讯]` (Data Analytics Labo Company)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个训练自由的推理框架ASC‑MQRA，通过多次视频问答推理并聚合答案来提升视频问答性能，并研究了低分差示例的再仲裁机制。

**💡 创新点**

在无训练的条件下首次使用答案级自一致性进行测试时的多样化采样与聚合，并引入基于投票分差的条件再仲裁来进一步提升低置信度样本的准确率。

**🔧 技术方法**

采用Gemini 3.1 Pro Preview多模态模型，结合随机温度采样的多次推理、答案级自一致性聚合、阈值触发的候选集保留及再仲裁，并用LangGraph实现推理管道。

**📊 数据集**

使用CVPR 2026 VidLLMs Challenge Track 2的VRR‑QA数据集，该数据集包含1000个电影视频问答样本，分为验证集和测试集。

**📈 对比分析**

与单通道推理相比，ASC在验证集平均准确率提升至72.73%（+4.32%）并在测试集提升至81.16%（+8.45%），宏平均准确率分别提升至78.34%和80.91%；MQRA在验证集略有提升但在测试集不如ASC。

**⚠️ 局限性**

限制包括推理成本高（需多次推理）、在计数、空间推理等难题上仍表现不佳，以及再仲裁在不同类别分布下可能适得其反；当模型基线能力不足时，多样化聚合的收益有限。

---

## 57. Towards Estimating Normal and Shear Interface Pressures in Prosthetic Sockets via Least Squares and Mechanics Modeling

**arXiv ID:** 2606.04222 | [PDF](https://arxiv.org/pdf/2606.04222v1)

**作者:** Axel González Cornejo `[一作]`, Edgar Bolívar-Nieto `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了人工残肢–插座测试台，通过稀疏电容传感器和6轴力传感器同步采集全局力矩和局部接触压力，利用最小二乘法识别三轴弹簧‑质量接触模型参数；

**💡 创新点**

创新之处在于提出两阶段凸优化（先估计空位参数再估计弹簧刚度/偏置），将全局力矩与局部压力同时作为目标进行多目标优化，并通过 Pareto 前沿分析阐释偏置项对性能的提升；

**🔧 技术方法**

采用了三轴弹簧‑质量（lumped‑parameter）模型、二维/三维坐标变换、无交叉的非线性门控弹簧公式、最小二乘估计、加速度/角速度同步采集的运动捕捉与电容压电转换；

**📊 数据集**

使用了实验室自制数据集：在 19 kg 静载荷下记录 1000 帧（4 Hz）完整的力矩、局部压力和姿态数据；

**📈 对比分析**

对比了仅弹簧刚度（无偏置）与弹簧刚度+偏置两种模型，使用 RMSE/MAE 评价；加入偏置后全局力矩 RMSE 下降至 0.44 Nm，局部压力 RMSE 中值从 90 N 降至 3.76 N；Pareto 前沿显示偏置显著缓解两目标间的权衡；

**⚠️ 局限性**

局限性包括：对空位参数估计的精度不足导致弹簧激活模式受限；传感器稀疏布置无法重建全口腔压力分布；仅在静态保持段验证，动态行走等情况尚未评估；模型假设为准静态，忽略动态耦合与传感器漂移。

---

## 58. MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A

**arXiv ID:** 2606.04231 | [PDF](https://arxiv.org/pdf/2606.04231v1)

**作者:** Hanoz Bhathena `[一作]` (JPMorgan Chase & Co), Denis Kochedykov `[通讯]` (JPMorgan Chase & Co)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5120188692)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MM-BizRAG，一个面向企业文档的多模态检索增强生成框架，主动提取并利用文档结构；

**💡 创新点**

创新点在于：1）结构感知拆分，根据垂直/水平布局动态路由文档；2）统一LLM驱动的占位符对齐转换保持自然阅读顺序；3）检索与生成上下文解耦的推理时多模态组装；4）FastRAGEval单调用LLM评估指标；

**🔧 技术方法**

采用Docling等布局解析器、GPT-4.1/LLM生成描述、OpenAI Text-Embedding-3-Large、Cohere-Embed-V4或Nomic-Multimodal-Embed-3B等嵌入模型，结合检索、重排序和多模态生成；

**📊 数据集**

评估数据集包括内部企业多模态集合（1908题、1048文档、20429页）及公开基准SlideVQA和FinRAGBench-V；

**📈 对比分析**

与文本仅RAG、ColPali、VisRAG等基线对比，MM-BizRAG在所有指标上领先，FinRAGBench-V提升最高32%，SlideVQA提升8-11%等，FastRAGEval与人工评估相关性更高；

**⚠️ 局限性**

局限：仅测试两大公开基准；内部数据未公开；多语言、非英文文档未覆盖；评估仅限于两种布局方向；未与更广泛或最新基线比较；

---

## 59. Self-Distilled Policy Gradient

**arXiv ID:** 2606.04036 | [PDF](https://arxiv.org/pdf/2606.04036v1)

**作者:** Yifeng Liu `[一作]` (University of California), Quanquan Gu `[通讯]` (University of California)

**通讯引用:** 180489 | [OpenAlex ID](https://openalex.org/A5049812527)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Self-Distilled Policy Gradient (SDPG) 框架，将 RLVR 的二进制 verifier 奖励与全词表精确自我蒸馏 (OPD) 结合，并加入参考策略 KL 正则化；

**💡 创新点**

将 OPD 的全词表反向 KL 与本地 policy‑gradient 等价梯度形式结合，形成可解释的优势；通过正向优势门控与 warmup‑decay β 调度实现训练稳定；融合参考策略 KL 正则化，解决纯自蒸馏的探索受限与模式崩溃；

**🔧 技术方法**

使用 RLVR / GRPO 基础，On‑Policy Self‑Distillation 的全词表 KL，Unnormalized KL (UFKL/URKL) 作为参考策略正则化，正向优势门控与 β warmup‑decay，vLLM roll‑out，FSDP 混合精度，AdamW 优化；

**📊 数据集**

在 Qwen3‑4B/1.7B LLM 上训练；使用 DAPO‑Math‑17k（13.9k 英文样本）生成特权信息；验证基准为 AIME2024/2025、AMC23；

**📈 对比分析**

与 GRPO、RLSD、OPCD 等基线对比，SDPG‑URKL 与 UFKL 在 AIME、AMC 等任务上均超越基线；SDPG‑UFKL 在五个指标上获得最优，SDPG‑URKL 在其余一个指标上最优；保持较高熵与稳定响应长度；

**⚠️ 局限性**

仍依赖预先生成的特权信息，特权分布噪声会影响训练；β 调度需手动设置；在极长生成中可能仍存在稀疏奖励问题；对其他任务的泛化能力有限。

---

## 60. LLM Compression with Jointly Optimizing Architectural and Quantization choices

**arXiv ID:** 2606.04063 | [PDF](https://arxiv.org/pdf/2606.04063v1)

**作者:** Hoang-Loc La `[一作]` (UiT Arctic University of Norway), Phuong Hoai Ha `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于差分神经架构搜索（Differentiable NAS）的LLM压缩框架，能够在搜索空间内同时优化模型结构和层级混合精度量化；

**💡 创新点**

创新点在于（1）全空间可微分搜索，避免了传统NAS的有限搜索和分离量化；（2）引入重要性感知深度剪枝和权重共振（weight-entanglement）混合操作；（3）软件层面的向量化实现显著提升训练速度；（4）结合LoRA与分组量化，支持多精度权重与激活量化；

**🔧 技术方法**

使用差分NAS、ReinMax采样、LoRA、混合精度量化、Straight‑Through Estimator、Marlin/ABQ-LLM内核、OmniQuant初始化、向量化混合权重计算；

**📊 数据集**

使用Alpaca指令微调数据集、Calibrate数据、以及七个常用推理任务（BoolQ、PIQA、HellaSwag、WinoGrande、ARC-easy/Challenge、MMLU）进行评估；

**📈 对比分析**

与子网选择（subnet‑selection）和LoNAS两大基线相比，未量化版本（ours‑no‑quant）在2–6 B参数范围内平均准确率提升约3–6%，推理延迟缩短约30%；量化版本（ours‑quant）在相同准确率或延迟目标下，平均推理速度提升最高可达1.4×，准确率提升约6%；

**⚠️ 局限性**

局限性包括：对单一A100 GPU的显存占用仍较高（约3.2 GB额外开销）；量化效果受限于现有量化技术（如QA‑LoRA不兼容）；方法在极低参数（<2 B）模型上压缩能力有限，准确率差距缩小；

---

## 61. DLO-Lab: Benchmarking Deformable Linear Object Manipulations with Differentiable Physics

**arXiv ID:** 2606.04206 | [PDF](https://arxiv.org/pdf/2606.04206v1)

**作者:** Junyi Cao `[一作]` (University of Massachusetts Amherst), Chuang Gan `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 15009 | [OpenAlex ID](https://openalex.org/A5040877128)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了可微分的可变形线性物体（DLO）仿真器与基准平台DLO‑Lab，并在此基础上设计了专用DLO代理进行抓取点建议和任务分解，验证了其在模拟与真实硬件上的有效性。

**💡 创新点**

①首次将多材质耦合与可微分仿真结合，支持弹性、塑性、束缚、流体等多种物理行为；②在DER框架中加入弯曲塑性与位置基础摩擦接触，并实现全链路可微；③构建覆盖高阶拓扑、长时序的基准任务；④利用Vision‑Language Model自动生成抓取点与分解计划；⑤实现梯度检查点技术以支持千步以上长时间可微。

**🔧 技术方法**

Taichi + Genesis平台、Discrete Elastic Rods（DER）+ 弯曲塑性、位置基础动力学（PBD）摩擦、自动微分与梯度检查点、VLM抓取建议；RL方法包括PPO、SAC；FO‑MBRL方法包括SHAC、SAPO；轨迹优化方法包括CMA‑ES与GD。

**📊 数据集**

使用自定义仿真场景生成的数据与真实抓取实验数据；对三根绳子进行系统辨识；基准任务共10个（8固定时序+2长时序）。

**📈 对比分析**

在固定时序任务中，CMA‑ES在样本效率与最终回报上最优；GD在平滑任务可行但易陷入局部；MFRL（PPO/SAC）表现最差；FO‑MBRL（SHAC/SAPO）中等；在真实硬件上，零射击开放式策略在Gathering和Wiring‑post任务成功率较高，闭环策略在Wiring‑ring任务实现约58%的成功率。

**⚠️ 局限性**

仍需人工设定物理参数与场景；可微分模拟在高维接触与自交事件下收敛不稳；VLM抓取建议依赖提示，鲁棒性有限；梯度缺失导致GD在间接接触任务失效；验证范围限于双臂平面环境，缺乏对更复杂机器人与动态环境的评估。

---

## 62. Formal verification of the S-two AIR

**arXiv ID:** 2606.04311 | [PDF](https://arxiv.org/pdf/2606.04311v1)

**作者:** Jeremy Avigad `[一作]` (Carnegie Mellon University), Alon Titelman `[通讯]` (StarkWare Industries Ltd.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

使用 Lean 4 证明 StarkWare S‑two AIR 编码的可靠性，确保其满足 Cairo 虚拟机语义并能够在区块链上安全验证程序执行。

**💡 创新点**

将动态可变的 AIR 生成器转译为纯函数式 Lean 代码，并通过对 logup 传递协议、组件约束和多元多项式求和等复杂数学结构的形式化证明，验证了 S‑two 的完整性与安全性。

**🔧 技术方法**

主要技术包括 Lean 4 交互式证明、基于依赖类型的抽象、数域映射（Felt ↔ Felt252）、多项式与理想理论、Fiat‑Shamir 伪随机化、以及 logup 计数与组合学证明。

**📊 数据集**

本工作不依赖外部数据集，而是对 StarkWare 官方实现的 AIR 代码和公共检验数据（程序、内存映射、初始/最终寄存器状态）进行形式化验证。

**📈 对比分析**

相比先前的 Stone AIR 验证，本方法通过证明更复杂的组件和 logup 协议，显著提升了验证覆盖面与安全误差（误判概率下降至 2⁻⁶⁰⁶ 级别），验证性能主要受证明规模影响，整体证明时间为数小时，但在实际部署中已被集成到 StarkWare 的生产流水线。

**⚠️ 局限性**

局限性包括对 AIR 生成器的全局状态假设未被形式化、对外部预处理列的正确性仍需工程师手动验证、以及对 Felts 与 Felt252 之间的范围映射仅在 2²⁹ 步以内保证正确，超大步长或寄存器溢出场景未覆盖。

---

## 63. Intra-Modal Neighbors Never Lie: Rectifying Inter-Modal Noisy Correspondence via Graph-Based Intra-Modal Reasoning

**arXiv ID:** 2606.04061 | [PDF](https://arxiv.org/pdf/2606.04061v1)

**作者:** Yang Liu `[一作]` (Sichuan University), Jiancheng Lv `[通讯]` (Sichuan University)

**通讯引用:** 6638 | [OpenAlex ID](https://openalex.org/A5073535763)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8`

**🎯 论文内容**

本文提出了 Intra‑Modal Neighbor‑aware Noise Rectification (IN^2R) 框架，用于纠正图像与文本配对中的噪声对应关系，并将噪声样本的监督标签从离散选择转化为连续原型合成。

**💡 创新点**

创新点在于：①放弃传统的“离散选择”方法，改为利用图像或文本的局部邻域图结构，通过可学习的 Graph Refiner（多头自注意力）生成连续的软原型；②通过跨模态对称记忆（Cross‑Model Memory）和共训练（co‑training）实现噪声源与检索源的解耦；③在干净样本上施加几何一致性约束，构建稳健的特征流形，为噪声校正提供可靠的空间参考。

**🔧 技术方法**

技术手段包括：共训练双网络、动态交叉模态记忆队列、ELITIST滚动更新、Graph Refiner（MHSA + FFN）用于邻域关系推理、SCE 对称损失、三重损失（交互对齐、图像内模态一致、文本内模态一致）以及用于噪声校正的连续监督损失。

**📊 数据集**

实验数据集涵盖：Flickr30K、MS‑COCO（训练集、验证集、测试集）以及 Conceptual Captions（CC152K）真实网络噪声集。

**📈 对比分析**

与现有方法（NCR、BiCro、L2RM、CREAM、ESC、GSC、SPS、PCSR 等）进行对比，IN^2R 在 20%–80% 对称噪声下均取得显著提升，最高在 80% 噪声时取得 rSum 超过 500（比 PCSR 提升 21 点），在 CC152K 上实现 rSum 380.8，刷新多模态检索 SOTA；整体表现稳定、鲁棒。

**⚠️ 局限性**

局限性包括：①对局部邻域一致性的依赖，若数据分布中存在少数族群或偏见，合成原型可能进一步放大这些偏差；②在极端噪声或分布漂移情况下，邻域信息可能不足；③实现复杂度较高，需要维护双网络、记忆队列和图结构，推理时算力成本相对传统方法更高。

---

## 64. Literature-Guided Minimax Optimization of Virtual Epilepsy Neurostimulation

**arXiv ID:** 2606.04339 | [PDF](https://arxiv.org/pdf/2606.04339v1)

**作者:** Cathy Liu `[一作]` `[通讯]`, Cathy Liu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用文献挖掘、TVB Epileptor 模拟和 LLM 生成的参数，在虚拟癫痫患者中进行最坏案例鲁棒优化，评估内在模型控制和外部刺激两种方案。

**💡 创新点**

提出了基于文献引导的 minimax 优化框架，能够在有限评估预算内通过 LLM 生成的候选参数实现对 worst-case 症状的显著改进。

**🔧 技术方法**

采用 PubMed 文献挖掘、The Virtual Brain (TVB) Epileptor 仿真、LLM（大语言模型）生成提议、最坏案例（minimax）目标的黑盒优化。

**📊 数据集**

使用 PubMed 的136篇癫痫相关文献抽取的1,080条研究点，以及 TVB 的76区连接组构建的虚拟癫痫患者数据（5、8、20、30等患者样本）。

**📈 对比分析**

与贝叶斯优化和随机搜索对比；在内在控制实验中worst-case奖励提升39.8%；在外部刺激实验中提升仅1.7%，但在760候选的全区域网格搜索中排名第四，仅距离最优0.0082。

**⚠️ 局限性**

虚拟患者样本有限、奖励函数仅为快速状态方差、未包含真实病人结构或临床指标、外部刺激仅为单一电流增益、未考虑波形、频率、闭环时序等关键参数；LLM仅为提议生成器，需专家验证。

---

## 65. MimeLens: Position-Agnostic Content-Type Detection for Binary Fragments

**arXiv ID:** 2606.04171 | [PDF](https://arxiv.org/pdf/2606.04171v1)

**作者:** Michael J. Bommarito `[一作]` `[通讯]`, Michael J. Bommarito

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一系列小型BERT风格编码器，可在任意偏移的文件片段上对125类MIME类型进行分类；

**💡 创新点**

核心创新在于使用位置无关的预训练：从文件中均匀随机偏移处抽取片段训练，消除对文件头的依赖；

**🔧 技术方法**

技术采用自监督MLM预训练的Transformer，使用Byte或BPE分词，RoPE位置编码，RMSNorm，GeGLU前馈，随后使用平均池化和线性探针；

**📊 数据集**

数据集为33 GB的多源二进制语料（约30k ELF/PE/Mach‑O/APK文件）以及Magic‑Files和Magic‑Frags用于评估；

**📈 对比分析**

与Google的Magika v1.1比较，针对完整文件头、UDP首包、随机磁盘块等场景；在完整文件上对齐Top‑1提升10.7 pp，在单包首包上实现0.855的准确率，在随机磁盘块上比基线高约2×；

**⚠️ 局限性**

局限性包括：评估仅基于Magika自带标签，未验证与外部标准（PRONOM等）的一致性；对真实网络条件（TCP、加密、乱序）和磁盘混合未充分测试；CPU推理延迟比Magika慢1–2个数量级；需在实际分布上微调以获得最终性能。

---

## 66. RUBAS: Rubric-Based Reinforcement Learning for Agent Safety

**arXiv ID:** 2606.04051 | [PDF](https://arxiv.org/pdf/2606.04051v1)

**作者:** Xian Qi Loye `[一作]` (Tsinghua University), Minlie Huang `[通讯]` (Tsinghua University)

**通讯引用:** 16148 | [OpenAlex ID](https://openalex.org/A5044042138)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RUBAS，一种基于 Rubric 的强化学习框架，用于对工具驱动的 LLM 代理进行安全对齐。

**💡 创新点**

创新点在于将代理行为拆分为工具使用安全、参数安全、响应安全和有用性四个可量化维度，并利用细粒度的 Rubric 产生可解释的奖励信号，同时结合在线的 Group Relative Policy Optimization (GRPO) 进行学习。

**🔧 技术方法**

主要技术包括：基于 LLM 的实例生成与 Rubric 设计、程序化与模型判定的标准化评价函数、多维奖励聚合、完整性与推理奖励惩罚、以及 GRPO 的在线强化学习。

**📊 数据集**

使用了 2,353 条合成安全实例（2,000 条有害、235 条敏感、118 条无害）以及 82 个环境与 805 个工具集合进行训练与评估；在 Agent‑SafetyBench、InjecAgent、AgentHarm、AgentSecurityBench、Berkeley Function Calling Leaderboard 与 ToolBeHonest 等公开基准上进行对照实验。

**📈 对比分析**

与规则奖励、GuardModel、DPO、SFT 四个基线比较，RUBAS 在 Qwen3-8B、Qwen3-14B 和 GLM-4.7-Flash 上实现了最低的风险分数（例如 Qwen3-8B 从 52.7% 降至 15.9%）且保持了接近或优于基线的工具使用实用性和较低的幻觉率，表现出最优的安全‑实用性平衡。

**⚠️ 局限性**

局限性包括：主要基于合成场景，可能与真实部署情况存在差距；奖励质量依赖于生成 Rubric 的 LLM 的偏差与校准；仅验证了思考模式模型，未评估非思考或无明确推理轨迹的系统；并非完整的安全方案，仍需与运行时监控、人工确认等措施结合使用。

---

## 67. Consensus is Strategically Insufficient: Reasoning-Trace Disagreement as a Knowledge-Representation Signal

**arXiv ID:** 2606.04223 | [PDF](https://arxiv.org/pdf/2606.04223v1)

**作者:** Michał Wawer `[一作]` (Warsaw University of Technology), Jarosław A. Chudziak `[通讯]` (Warsaw University of Technology)

**通讯引用:** 97 | [OpenAlex ID](https://openalex.org/A5008057050)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于LLM的多智能体系统，将推理轨迹与决策抽象为四种符号化的分歧状态，并根据这些状态设计可推理的路由策略；

**💡 创新点**

核心创新在于将分歧视为可表征的知识状态而非噪声，通过“共识/分歧”+“相似/不相似”两维划分出CA、DA、DD、CD四种状态，并为每种状态设定可推理的默认路由；

**🔧 技术方法**

使用大规模语言模型（LLM）生成推理轨迹和二元决策，利用嵌入向量计算语义相似度并定义阈值，构建符号化状态映射，再结合非单调推理规则实现路由；

**📊 数据集**

在内容审核任务中采用“Measuring Hate Speech”语料库，使用五个基于不同价值取向的prompt化LLM实例来产生多样化的推理轨迹和决策；

**📈 对比分析**

与仅基于分歧幅度（1‑sim）的基线相比，符号化路由在高人类分歧案例的F1上从0.503提升到0.548，显示其更精准地捕捉价值冲突；

**⚠️ 局限性**

局限性包括相似度计算的粗糙性、手工设定的路由规则、仅在单一领域（内容审核）验证、以及prompt差异化可能不足以覆盖真实的多元智能体差异。

---

## 68. Climbing Up the Semantic Tower -- at Runtime

**arXiv ID:** 2606.04034 | [PDF](https://arxiv.org/pdf/2606.04034v1)

**作者:** François-René Rideau `[一作]` `[通讯]`, François-René Rideau

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出一种基于类别理论的运行时反射协议，能够在程序运行时通过安全点从具体实现向上迁移到更高层次的抽象实现，实现抽象与具体层次的双向导航。

**💡 创新点**

创新点在于将抽象语义与运行时反射统一到一个形式化框架：将实现视为从具体实现到抽象实现的部分函子，并定义可观测安全点、完整性、活跃性等属性，可通过程序提取获得可运行的反射API；同时将效果也视为可重现的箭头，实现效果的模拟与执行。

**🔧 技术方法**

使用类别理论、依赖类型（Agda）证明、Curry‑Howard 对应、Functoriality、以及程序提取技术构建协议，并计划在 Gambit Scheme 及 Gerbil 等语言实现。

**📊 数据集**

无具体数据集，论文以形式化证明和理论示例为主。

**📈 对比分析**

目前尚未实现或实验评估；论文中未给出性能比较，计划未来在 Gambit Scheme 上实现并进行实验。

**⚠️ 局限性**

主要限制是协议仍处于理论阶段，未实现；实现成本高，需要在每种语言实现中嵌入反射机制；缺乏实证评估，难以验证性能与可行性。

---

## 69. Derivative Informed Learning of Exchange-Correlation Functionals

**arXiv ID:** 2606.04279 | [PDF](https://arxiv.org/pdf/2606.04279v1)

**作者:** Eike S. Eberhard `[一作]` (Technical University of Munich), Stephan Günnemann `[通讯]` (Technical University of Munich)

**通讯引用:** 15595 | [OpenAlex ID](https://openalex.org/A5074504351)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种在Grassmann流形上监督能量导数的复合损失函数（DI‑Loss），用于将高阶混合泛函（如B3LYP）压缩为低阶（O(N³)）机器学习XC泛函。

**💡 创新点**

创新点在于：① 通过对能量的一阶、二阶导数（梯度与Hessian）进行监督，逼近参考泛函在自洽点附近的势能面；② 采用Grassmann流形约束，使得梯度监督只关注能改变密度的方向，避免无物理意义的占据‑占据与虚拟‑虚拟块；③ 将梯度和Hessian信息融合到单阶段训练流程，显著提升泛函的泛化与响应性质。

**🔧 技术方法**

技术包括：自洽密度泛函理论（KS-DFT）+深度学习；利用Grassmann流形参数化和梯度/ Hessian 计算；多任务损失（能量、密度、梯度、Hessian）；自适应 Metropolis‑inspired 训练稳定化；多种ML‑XC 架构（NNmGGA、EG‑XC、Skala‑mGGA 等）;

**📊 数据集**

数据集：QM9（训练/验证），QM40（外部检验）。使用 def2‑SVP 基组；在 QM9 中去除 F，QM40 中去除 F、S、Cl；按重原子数划分大小，评估尺寸外推。

**📈 对比分析**

对比方法：仅能量+密度监督（E+ρ）与加入梯度（+∇）和梯度+Hessian（+∇+H）三种损失；与传统混合泛函 B3LYP 及非局部 EG‑XC 进行性能对比。结果：平均总能量 MAE 降低 66%；对 Eρ、μρ、L₂[ρ] 的改善不一；Hessian 监督显著提升 TDDFT 低阶激发能量误差，最高可达 35% 降低；使用 distilled 密度初始化 B3LYP 可将 SCF 迭代次数降低约 50%，整体时间得到加速。

**⚠️ 局限性**

局限性：① DI‑Loss 仅适用于可获取梯度/Hessian 的参考泛函（如混合泛函），难以直接用于 CCSD(T) 等高精度目标；② 仅在闭壳层有机分子上验证，是否能推广到过渡金属、固体等仍未知；③ 训练过程中存在额外的梯度/Hessian 计算开销；④ 当前 ML‑XC 架构在捕捉 exact‑exchange 行为上仍有限，需进一步提升表达能力。

---

## 70. Exploring the Topology and Memory of Consensus: How LLM Agents Agree, Fragment, or Settle When Forming Conventions

**arXiv ID:** 2606.04197 | [PDF](https://arxiv.org/pdf/2606.04197v1)

**作者:** Aliakbar Mehdizadeh `[一作]` (University of California Davis), Martin Hilbert `[通讯]` (University of California Davis)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在八种固定拓扑的16节点网络上，让LLM代理执行命名游戏，并对其有限记忆深度进行实验，研究记忆与网络结构如何共同影响协约形成与共识过程。

**💡 创新点**

创新点在于揭示记忆深度与网络拓扑交互可导致“速度‑统一”权衡反转，即长记忆在中央化网络加速收敛但导致碎片化，而在去中心化网络则延缓收敛并促进全局共识。

**🔧 技术方法**

采用Agent‑Based Simulation、Gemini 2.0 Flash LLM与自然语言提示构造记忆回溯、Fictitious Play与RL行为基线建模，以及统计检验（ANOVA、混合效应模型）等技术。

**📊 数据集**

使用Mason‑Watts八种16节点固定度（k=3）拓扑与十个任意符号的约定集合，记忆长度取M∈{2,5,10}，共进行432次独立模拟。

**📈 对比分析**

通过对不同记忆与网络组合的收敛速度、碎片化水平及二元配对成功率进行ANOVA与多重比较，结果显示中央化网络在长记忆下收敛快但碎片化高；去中心化网络在短记忆下更易实现全局共识。

**⚠️ 局限性**

局限在于仅研究同质小型固定度网络，未考虑异质代理、动态拓扑、温度设置或真实知识约定；且实验仅基于Gemini默认采样温度，缺乏对温度敏感性的系统检验。

---

## 71. Building The Ph(ysical)AI Layer Of Machine Intelligence

**arXiv ID:** 2606.04106 | [PDF](https://arxiv.org/pdf/2606.04106v1)

**作者:** Ulbert Jose Botero `[一作]` (MIT Lincoln Laboratory), Daniel Capecci `[通讯]` (MIT Lincoln Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

训练一个只在射频（RF）数据上学习信号理论原则的基础模型，利用冻结的表示实现跨模态迁移，能够在音频、图像、文本、视频等从未见过的域上进行线性探测。

**💡 创新点**

创新点在于将物理信号理论（傅里叶分解、能量守恒、对称性）嵌入模型结构与损失中，构建了 principle‑driven foundation model；并且仅用1.99 M参数实现了跨模态迁移，显著压缩模型规模。

**🔧 技术方法**

使用 PlanFormer 架构（双域处理、频率保持池化、Parseval Focus、IsoFICReg、LED 等）以及相应的自监督损失，强化了对信号理论的遵循与对称性学习。

**📊 数据集**

训练数据为射频指纹数据集（ORACLE、POWDER 及内部 3 个数据集，共 39 类发射器，涵盖多种调制、协议和信道条件）。评估任务共 15 个，涵盖 1‑D、2‑D、3‑D 以及文本任务，如 RF 调制识别、语音说话人识别、音乐乐器分类、地震事件分类、ArXiv 论文学科分类、MNIST、FashionMNIST 等。

**📈 对比分析**

与规模驱动的大模型 CLIP ViT‑B/32（151 M 参数）和 DinoV3 ViT‑S（21 M 参数）对比，物理任务上取得 84.5% Top‑1（与 CLIP 的 87.7% 相差 3.2%），参数量比 CLIP 少 76 倍，FLOPs 少 158 倍；语义任务上表现为 70% 对比 CLIP 的 91.2%。

**⚠️ 局限性**

局限在于对纯语义任务的迁移效果有限，难以捕捉高层语义；此外模型仅在 RF 域上预训练，需进一步验证对其他物理信号和更复杂场景的泛化能力。

---

## 72. Distribution-Free Risk-Aware Planning and Control Under Uncertainty Using Conformal Spectral Risk Control

**arXiv ID:** 2606.04185 | [PDF](https://arxiv.org/pdf/2606.04185v1)

**作者:** Junsik Eom `[一作]` (University of Michigan), Tulga Ersal `[通讯]` (University of Michigan)

**通讯引用:** 2767 | [OpenAlex ID](https://openalex.org/A5076004234)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种分布无关的风险量化与控制框架（CSRC），并将其嵌入到风险感知模型预测控制（RA‑MPC）中，实现在未知分布下对光谱风险约束的统计安全保证。

**💡 创新点**

创新点在于：
1) 将 conformal risk control (CRC) 扩展到通用光谱风险测度，利用加权 CRC (W‑CRC) 与截断策略实现无分布假设的风险控制；
2) 通过离线学习得到预测集参数 λ̂，避免在线昂贵的权重求解；
3) 将 λ̂ 直接嵌入 MPC 约束，提供关于真实不确定性分布的统计安全保证。

**🔧 技术方法**

主要技术包括： conformal prediction、加权 CRC、光谱风险测度（CVaR、Wang 等）、MPPI 采样式 MPC、基于欧式距离的损失函数、截断与保守化策略。

**📊 数据集**

使用仿真数据集：
- 静态障碍物避让（1210 条样本，分为 1000/100/1100 的校准/估计/测试集），
- 动态障碍物避让（2100 条样本，分为 1000/100/1100 的校准/估计/测试集）。

**📈 对比分析**

与基线 SAA‑MPC（使用 100 条在线采样估计风险）比较，评价指标包括：障碍物约束违规率、成功率和平均求解时间。实验结果显示：
- 静态场景下 CSRC‑MPC 约束违规率显著下降，成功率提升；
- 动态场景下 CSRC‑MPC 约束违规率从 52.9% 降至 6.0%，成功率提升至 100%，求解时间从 104.9 ms 降至 49.9 ms。

**⚠️ 局限性**

局限性包括：
1) 只提供边缘（marginal）风险保证，缺乏条件（conditional）保证；
2) 仍可能出现偶尔违规，尤其在极端尾部事件；
3) 仅在仿真环境验证，缺乏真实硬件实验；
4) 目前仅考虑单一风险来源，未扩展至多风险源或更复杂动态模型。

---

## 73. Bayesian Membership Privacy for Graph Neural Networks

**arXiv ID:** 2606.04069 | [PDF](https://arxiv.org/pdf/2606.04069v1)

**作者:** Sinan Yıldırım `[一作]` (Sabancı University), Megha Khosla `[通讯]` (Delft University of Technology)

**通讯引用:** 644 | [OpenAlex ID](https://openalex.org/A5027689420)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了针对图神经网络的贝叶斯成员隐私（BMP）框架，结合采样概率与节点先验，改进传统的成员推断方法。

**💡 创新点**

创新点在于：① 引入节点依赖的先验概率，将成员推断建模为贝叶斯假设检验；② 定义了 BMP（包括 BMP-R、BMP-L、BMDP 等）并证明其后处理、组合性质；③ 提出基于 MIA 结果的 MCMC 后验推断方法，能给出不确定性敏感的隐私参数估计。

**🔧 技术方法**

技术手段包括：贝叶斯统计、图神经网络（GNN）训练与图采样（随机采样、雪崩采样）、成员推断攻击（基于概率分布的决策）、MCMC（MHAAR）进行隐私参数的后验采样。

**📊 数据集**

实验使用多种基准图数据集（如 Cora、Citeseer 等），并在不同模型架构、采样比例、噪声注入等设置下评估。

**📈 对比分析**

与传统的 MP（均匀误差界）对比，BMP 能揭示更细粒度、基于先验的隐私泄露情况；实验显示 BMP-R 参数普遍小于 MP，说明考虑先验后隐私风险更低；在雪崩采样下 BMP 参数更大，表明该采样方式泄露更高。

**⚠️ 局限性**

局限性：仅给出经验性估计，缺乏 BMP 的解析隐私保证；对采样过程假设相对简单，未来工作需设计提供 BMP 解析安全性的机制。

---

## 74. Position: Deployed Reinforcement Learning should be Continual

**arXiv ID:** 2606.04029 | [PDF](https://arxiv.org/pdf/2606.04029v1)

**作者:** Parnian Behdin `[一作]` (Alberta Machine Intelligence Institute), Golnaz Mesbahi `[通讯]` (Alberta Machine Intelligence Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出将已部署的 RL 系统视为持续学习问题的框架，并阐述可测量部署下持续 RL 的必要性，归纳四类非平稳性来源并给出案例研究。

**💡 创新点**

重新定义可测量部署为持续 RL 问题，使用历史过程 formalism 取代 MDP，提出持续学习者 vs 非持续学习者判定标准，并给出四类非平稳性（行动诱导、环境动态、目标演变、突发新奇）的系统性分析。

**🔧 技术方法**

采用历史过程理论、在线 RL 算法（如策略梯度、Q 学习）、对比实验（Rusting Pendulum）以及实际在线 RL 案例（Cursor Tab、Lyft、sim‑to‑real 机器人）来论证持续学习的优势。

**📊 数据集**

案例中使用 Cursor Tab（400M 日请求）、Lyft 乘客‑司机匹配流量、sim‑to‑real 机器人抓取数据集等实际部署数据；论文本身并未设计公开数据集。

**📈 对比分析**

通过比较 train‑then‑fix 与持续学习的性能曲线，案例显示 Cursor Tab 接受率提升 28%、Lyft 产生约 30 M 美元额外收入，sim‑to‑real 采样效率提升 100 倍；但论文主要为立场性讨论，缺少统一实验基准。

**⚠️ 局限性**

局限在于缺乏统一评估基准、假设可测量部署必须有可用奖励信号、未充分讨论奖励稀疏或延迟的情况，以及对安全与对齐问题的深度分析仍待补充。

---

## 75. Can I Take Another Dose? Evaluating LLM Decision-Making Under Temporal Uncertainty in OTC Dosing QA

**arXiv ID:** 2606.04262 | [PDF](https://arxiv.org/pdf/2606.04262v1)

**作者:** Maroof Kousar `[一作]` (Illinois Institute of Technology), Yibo Hu `[通讯]` (Illinois Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了DoseBench，一个基于成人非处方阿司匹林和布洛芬的81个剂量决策情境的基准，用于评估大型语言模型在剂量安全推理方面的表现。

**💡 创新点**

首次将剂量标签约束与时间窗口推理结合起来，提出了结构化评估框架，涵盖决策正确性、重复运行一致性、推理可验证性、失败类型和置信度表现。

**🔧 技术方法**

使用了多模态指令微调LLM（Qwen2.5‑7B、Llama‑3‑8B、Mistral‑7B、GPT‑4o‑mini），JSON结构化提示和五次重复采样进行评估，并手工标注推理可验证性与错误类型。

**📊 数据集**

基于人工整理的81个真实消费者提问场景，涵盖计时间隔、滚动24小时、复用多药、重复剂量与信息缺失等六类推理难点。

**📈 对比分析**

通过多数投票准确率、重复一致率、可验证分数和内部置信度等指标比较，GPT‑4o‑mini取得最高多数投票准确率（55.6%）和一致率（84.9%），但所有模型都表现出一致率与准确率之间显著差距。

**⚠️ 局限性**

局限在于仅关注成人阿司匹林/布洛芬两种药物，样本量有限，未提供基于规则的基线，人工可验证性标注存在主观性，且结果不易推广到更广泛的临床或多语言场景。

---

## 76. Towards Process Mining Use Case Map Models with PM4Py-UCM

**arXiv ID:** 2606.04350 | [PDF](https://arxiv.org/pdf/2606.04350v1)

**作者:** Daniel Amyot `[一作]` (University of Ottawa), Daniel Amyot `[通讯]` (University of Ottawa)

**通讯引用:** 5283 | [OpenAlex ID](https://openalex.org/A5043523779)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现了一个基于Python的库 pm4py-ucm，用于将事件日志自动挖掘为 Use Case Map（UCM）模型，并支持分层分解、执行者映射及导入导出。

**💡 创新点**

创新点在于：①把 UCM 作为过程挖掘的直接输出；②提供可配置的层级分解策略；③实现执行者（角色/资源）与责任的绑定；④支持双向导入导出至 jUCMNav 的 XML 格式，实现往返工程。

**🔧 技术方法**

技术主要包括：使用 pm4py 的 inductive‑miner 生成过程树；将过程树映射为 UCM 对象模型；基于配置参数实现分层拆分；通过日志属性聚合策略为 UCM 责任绑定执行者；使用 Streamlit 构建 Web 演示界面。

**📊 数据集**

使用了两个数据集：1) 100,008 事件、11,284 案例的合成 issue‑tracking 日志；2) 78,126 事件、5,600 案例的匿名化 claims‑payment 日志。

**📈 对比分析**

通过在不同的执行者抽象和分解策略下展示同一行为，说明工具的可配置性；但本文未给出量化的性能评估，仅以示例图和导出文件展示效果。

**⚠️ 局限性**

局限性包括：①仅支持 inductive‑miner，限制了非块结构流程的挖掘；②执行者聚合策略单一，无法处理多重执行者情况；③分解参数和模式过于经验性，缺乏自动调优；④未能推断 UCM 的高级构造（计时器、动态桩等）；⑤缺乏实际用户的可用性测试。

---

## 77. Network node immunization: improving Netshield algorithm through random rooted forests

**arXiv ID:** 2606.04131 | [PDF](https://arxiv.org/pdf/2606.04131v1)

**作者:** Luca Avena `[一作]` (University of Florence), Alessio Troiani `[通讯]` (University of Perugia)

**通讯引用:** 138 | [OpenAlex ID](https://openalex.org/A5074567569)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多节点免疫问题中，提出了一种基于Kirchhoff森林的随机搜索方法 K‑shield，用来改进传统的 Netshield 贪心算法，从而在保持相同时间复杂度的前提下获得更优的特征值下降（eigendrop）。

**💡 创新点**

创新点在于将随机生成的Kirchhoff森林根集的补集作为候选免疫节点集合，引入随机化搜索与调整后的 shield 值与总速率下降两种评估指标，形成三选一的决策机制，显著提升了在社区结构强、权重分布不均的网络上的性能。

**🔧 技术方法**

使用的技术包括：随机游走核与Kirchhoff森林采样（Wilson 算法），特征值与右 Perron–Frobenius 向量的计算，调整后的 shield 值与总速率下降的快速评估，以及对森林根数的条件化抽样。

**📊 数据集**

实验基准涵盖多种真实与合成网络：Karate Club、Enron、机场网络、RFID、英国学院网络、社区结构图（communities 01）以及加权与非加权的各类社交、交通与通信网络。

**📈 对比分析**

通过在每个图上多次采样并比较 eigendrop、shield 值以及总速率下降，K‑shield 在大多数测试集（尤其是权重不均、社区明显的图）中获得了比 Netshield 更高的 eigendrop，平均提升幅度可达数十个百分点；对比实验显示其在保持 O(m+nk) 复杂度的同时，性能提升显著。

**⚠️ 局限性**

局限性包括：在谱均匀（如完全图）或节点数极大时提升不明显；需先预先计算特征值/特征向量，且随机采样次数若过少可能错失更优解；K‑shield 的理论最优保证尚未给出，主要依赖经验性评估。

---

## 78. The Loss Is Not Enough: Sampling Conditions and Inductive Bias in Contrastive Representation Learning

**arXiv ID:** 2606.04280 | [PDF](https://arxiv.org/pdf/2606.04280v1)

**作者:** Justinas Zaliaduonis `[一作]` (Technical University of Munich), Sergios Gatidis `[通讯]` (Stanford University)

**通讯引用:** 6231 | [OpenAlex ID](https://openalex.org/A5080097591)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文研究对比学习（Contrastive Learning）在何种采样条件下能够恢复潜在几何结构，并给出了理论框架与实证验证。

**💡 创新点**

创新点在于提出“多样性条件”（Diversity Condition）——对正样本条件分布的支持要求，并证明该条件是几何保持恢复的必要性；展示在该条件被违反时，传统 InfoNCE 会倾向非正交解；设计了支持校正的 InfoNCE 变体，并证明其能使正交恢复成为可行但不唯一；同时阐明采样多样性与编码器归纳偏差的交互作用。

**🔧 技术方法**

使用的信息理论与测度论工具（如条件熵、Mazur‑Ulam 定理）、非线性 ICA 框架、vMF 分布下的 InfoNCE 理论化，实验上采用 SimCLR 与自定义的正负采样策略、对抗噪声的负样本采样；在实验证明中还使用了线性可辨识度、均方相关系数等指标。

**📊 数据集**

在实验中使用的主要数据集包括：1）自定义球面潜在空间（𝕊²）与多种生成器（Identity、Linear、Spiral、Patches、Invertible MLP）；2）CIFAR‑10 数据集进行真实任务验证。

**📈 对比分析**

比较方法：将标准 InfoNCE 与支持校正 InfoNCE、低归纳偏差 MLP、以及高归纳偏差的逆向编码器进行对比；在自定义数据集上，用线性可辨识度（R²）衡量潜在恢复质量；在 CIFAR‑10 上，用线性探针准确率评估下游分类性能。结果显示：① 在多样性条件满足时，所有模型均能几乎完美恢复潜在结构；② 条件被违反时，标准 InfoNCE 性能骤降；③ 高归纳偏差编码器能弥补多样性不足；④ 支持校正 InfoNCE 能显著提升但仍低于高归纳偏差模型；⑤ 在 CIFAR‑10 上，采样多样性越差，编码器归纳偏差对性能的影响越显著。

**⚠️ 局限性**

局限性包括：① 理论分析以 L² 正则化、球面潜在空间和 vMF 条件为前提，未涵盖更一般的潜在分布；② 支持校正 InfoNCE 在实际中需要获取固定潜在分量，计算和存储成本高，难以大规模部署；③ 实验仅覆盖有限的生成器与数据集，需进一步验证在更复杂真实数据上的泛化；④ 未讨论优化动态（如梯度噪声、训练不收敛）如何影响理论结论。

---

## 79. Low-rank Distributional Matrix Completion

**arXiv ID:** 2606.04176 | [PDF](https://arxiv.org/pdf/2606.04176v1)

**作者:** Jiayi Wang `[一作]` (University of Texas at Dallas), Raymond K. W. Wong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出将每个矩阵条目视为概率分布的分布矩阵完成方法，利用核均值嵌入将分布映射到RKHS，定义分布矩阵的多线性秩并引入核展开算子以构造低秩正则化，最终给出一个基于核范数的正则化M‑估计器并证明其非渐近误差界；

**💡 创新点**

创新点在于：①把分布条目映射为无限维RKHS向量并给出其多线性秩的定义；②利用功能展开与核展开算子将无限维结构转化为可求的核范数正则化；③在理论上给出全局MMD误差上界，并与经典低秩矩阵完成收敛率相匹配；

**🔧 技术方法**

使用技术包括：核均值嵌入、最大均值差距（MMD）度量、张量Tucker分解、功能展开与核展开算子、核近似（Nyström / RFF）、核范数正则化、加速ADMM求解、代表性定理与非渐近误差分析；

**📊 数据集**

实验数据集包括：模拟的高斯混合分布矩阵（1维与2维），以及真实的NYC出租车每日行程计数数据（265×265矩阵，缺失率39%以上）；

**📈 对比分析**

与现有基于最近邻Wasserstein barycenter的方法比较，在模拟实验中MMD、均值和方差误差均明显更低；在真实数据上测试集MMD误差从0.0212降至0.0015，且在案例中对分布形状的恢复更准确；

**⚠️ 局限性**

局限性包括：核参数与近似选择仍需经验；对极大规模样本的计算仍有挑战；仅考虑核均值嵌入的可分辨性，未处理更复杂的分布特征；缺乏对多模式或非独立观测的扩展；

---

## 80. Reflection Separation from a Single Image via Joint Latent Diffusion

**arXiv ID:** 2606.04107 | [PDF](https://arxiv.org/pdf/2606.04107v1)

**作者:** Zheng-Hui Huang `[一作]` (Shanda AI Research Tokyo), Yung-Yu Chuang `[通讯]` (National Taiwan University)

**通讯引用:** 7634 | [OpenAlex ID](https://openalex.org/A5006990118)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对单张图像的反射分离任务，提出并实现了一个基于扩散模型的统一生成框架，能够同时生成透射层和反射层。

**💡 创新点**

创新点在于：① 将扩散模型细化为专门的反射分离器；② 引入跨层自注意力机制实现层间信息解耦；③ 设计了分离采样策略消除层间干扰；④ 在潜在空间进行组合优化以提升分离质量。

**🔧 技术方法**

采用了Stable Diffusion v2.1的潜在扩散模型，结合跨层自注意力、分离采样、潜在优化、Fidelity‑Guided Feature Modulation（FGFM）等技术。

**📊 数据集**

使用 Real20、Nature 和 SIR2 三个真实场景数据集进行训练与评估，SIR2 用于评估反射层的真实效果。

**📈 对比分析**

与 YTMT、RobustSIRR、DSRNet、RRW、DSIT、RDNet 及 ControlNet 等六大先进方法对比，在 PSNR、SSIM、LPIPS、DISTS 等指标上多项表现均位列第一或第二，尤其在感知指标上显著优于对手。

**⚠️ 局限性**

主要局限包括：优化阶段需要精细调参；FGFM 在某些场景下可能无法完全抑制反射，导致透射层出现残留反射伪影。

---

## 81. UniCanvas: A Diffusion-base Unified Model for Text-in-Image Joint Generation

**arXiv ID:** 2606.04264 | [PDF](https://arxiv.org/pdf/2606.04264v1)

**作者:** Zeyuan Yang `[一作]` (University of Massachusetts Amherst), Chuang Gan `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 15009 | [OpenAlex ID](https://openalex.org/A5040877128)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了UniCanvas，一种基于扩散模型的统一模型，能够在同一像素画布上同时生成文本与图像，实现文本嵌入式图像生成（text‑in‑image joint generation）。

**💡 创新点**

创新点包括：1）将文本直接作为视觉模式写入像素画布，统一文本与图像的表示；2）采用两阶段生成流程（先写文本再生成下一帧图像），简化学习与推理；3）引入CLIP对齐损失和预训练文本渲染阶段，提升文本可读性与语义一致性。

**🔧 技术方法**

使用技术：扩散模型（DiT backbone）+ flow‑matching 训练；两阶段生成（视觉推理 + 视觉合成）；CLIP 对齐损失；VAE 编码/解码；预训练文本渲染（LPIPS感知损失）；LoRA 微调；DiffSynth 框架。

**📊 数据集**

使用的数据集：VSP（迷宫导航多步推理）、Recipe（烹饪图文步骤）、RLBench（机器人抓取轨迹）、COCO‑QA、Visual7W（视觉问答）等。

**📈 对比分析**

与现有统一模型（Anole、MVoT、MMaDA、BAGEL、Qwen2.5‑VL 等）在 VSP、RLBench、Recipe 等任务上进行对比；在 VSP 上平均成功率达 0.77，显著高于基线；在通用视觉推理任务上表现接近或优于 VLM；推理速度比其他扩散序列模型快，但仍慢于自回归 VLM。

**⚠️ 局限性**

局限性：1）长文本生成不稳定，难以生成多句长说明；2）视觉合成误差在多步推理中累积，导致长序列失败；3）文本渲染偶尔出现错误或模糊；4）需要更大规模训练与更强监督以进一步提升性能。

---

## 82. Measuring What Matters: Synthetic Benchmarks for Concept Bottleneck Models

**arXiv ID:** 2606.04326 | [PDF](https://arxiv.org/pdf/2606.04326v1)

**作者:** Julian Skirzynski `[一作]` (University of California, San Diego), Berk Ustun `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套可生成多样化、可控的概念瓶颈模型（CBM）基准，主要用于决策支持和自动化场景。

**💡 创新点**

创新点在于通过合成数据控制概念相关性、注释质量、模态等因素，构建可重复、可调节的实验环境，从而系统诊断CBM在不同条件下的表现。

**🔧 技术方法**

使用技术包括概念瓶颈模型（标准、嵌入、Gaussian 隐变量、能量基方法）、概念干预、选择性分类、以及合成数据生成器。

**📊 数据集**

主要数据集为自制的机器人分类数据集（9 个身体部件、10 种足部形状等）和数独验证数据集，另外在演示中也用到了公开概念标注数据集如 CUB、CelebA 等。

**📈 对比分析**

通过与黑盒基线对比以及四种 CBM 架构在干预预算、概念来源、对齐约束等维度的实验，发现干预可提升准确率、覆盖率和净效用；不同架构在概念噪声或检测质量低时表现差异显著。

**⚠️ 局限性**

局限性包括仅在合成任务上验证，缺乏真实世界部署数据；实验规模有限（仅 9×9 数独、单一概念检查成本假设），未覆盖完整的超参数或不确定性范围。

---

## 83. Smart Transportation Without Neurons -- Fair Metro Network Expansion with Tabular Reinforcement Learning

**arXiv ID:** 2606.04167 | [PDF](https://arxiv.org/pdf/2606.04167v1)

**作者:** Dimitris Michailidis `[一作]` (University of Amsterdam), Fernando P. Santos `[通讯]` (University of Amsterdam)

**通讯引用:** 1503 | [OpenAlex ID](https://openalex.org/A5075633444)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用表格强化学习方法解决地铁网络扩张问题，并将其转化为非马尔可夫奖励决策过程。

**💡 创新点**

1) 将动作空间从网格单元缩减为8个移动方向；2) 通过表格方式实现可解释策略；3) 引入公平性奖励（等分与Rawls原则）以兼顾效率与社会公平。

**🔧 技术方法**

蒙特卡洛表格强化学习（ε-贪婪策略），非马尔可夫奖励决策过程。

**📊 数据集**

两城真实数据：西安（29×29 km²格网，基于25M手机GPS生成OD矩阵）和阿姆斯特丹（35×47 0.5 km²格网，基于人类移动通用规律估计OD）。

**📈 对比分析**

与DeepRL、遗传算法和贪心搜索对比；在两城实验中，表格RL在相同奖励函数下获得与DeepRL相近的捕获需求性能，训练次数降低18倍，CO₂排放降低12倍。

**⚠️ 局限性**

需要对整个状态空间进行枚举，受限于规模；对于更大或更复杂的状态表示，表格方法可能不可扩展。

---

## 84. KODA: Contrastive Representation Comparison and Alignment for Vision-Language Foundation Models

**arXiv ID:** 2606.04180 | [PDF](https://arxiv.org/pdf/2606.04180v1)

**作者:** Youqi Wu `[一作]` (Chinese University of Hong Kong), Farzan Farnia `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 561 | [OpenAlex ID](https://openalex.org/A5017160178)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于核优化的对比嵌入聚类方法 KODA，用于发现两种视觉‑语言嵌入在同一数据集上的不一致聚类模式。

**💡 创新点**

创新点在于：① 将对比嵌入聚类形式化为带约束的二次优化问题，并通过 KKT 条件转化为可用特征值分解求解；② 引入多模态乘积核和联合随机傅里叶特征，解决高维张量特征映射的计算瓶颈；③ 通过一次维搜索寻找 λ，既实现强聚类又约束弱聚类，能够定位结构差异；④ 将发现的差异子集用于后续的嵌入对齐实验，验证其实用价值。

**🔧 技术方法**

主要技术包括核方法（高斯核、余弦核）、随机傅里叶特征（RFF）近似、特征值分解与块协方差矩阵求解、投影正交化、以及多模态乘积核（Hadamard 乘积）和联合 RFF。

**📊 数据集**

使用的公开数据集有 AFHQ、FFHQ、ImageNet（单模态）以及 MS‑COCO（多模态）进行实验；嵌入模型包括 DINOv2、CLIP、BLIP、OpenCLIP、SigLIP、SigLIP2 等。

**📈 对比分析**

KODA 通过在 CLIP、DINOv2 等模型上发现差异聚类，并将这些子集用于对齐实验，显著提升目标模型在这些子集上的聚类一致性（AMI/NMI/ARI 提升 0.5 以上）。在多模态实验中，KODA 识别的差异样本能放大检索性能差距（如 SigLIP vs CLIP），并在对齐后恢复与目标嵌入相近的几何结构。

**⚠️ 局限性**

局限性包括：① 需要共享的参考数据集，无法直接处理不匹配或分布差异较大的数据；② 目前聚类解释是基于无监督核几何，缺乏明确的统计显著性或标签验证；③ 对极大规模数据仍受限于随机特征维度与核矩阵近似误差；④ 只关注两种嵌入的差异，未能直接扩展到多模型或任务条件下的对齐与解释。

---

## 85. Feasibility of Time-Domain DNN-Based Speech Enhancement on Embedded FPGA for Hearing Aid

**arXiv ID:** 2606.04221 | [PDF](https://arxiv.org/pdf/2606.04221v1)

**作者:** Feyisayo Olalere `[一作]` (Radboud University), Marcel van Gerven `[通讯]` (Radboud University)

**通讯引用:** 9984 | [OpenAlex ID](https://openalex.org/A5074794877)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在AMD‑Xilinx Kria KV260嵌入式FPGA上实现并评估了基于SuDoRM‑RF++时域语音增强模型的语音分离与去噪任务，比较FP32与16位定点实现的延迟、功耗与音质指标

**💡 创新点**

通过系统性地增大片上参数缓存和采用16位定点精度，在保持甚至略提升音质的前提下显著降低首次样本延迟，并首次实现了满足10 ms临床阈值的FPGA端到端语音去噪

**🔧 技术方法**

采用C++ HLS实现可流式加速器、AXI‑Stream/AXI‑Master接口、FP32/FP16数据路径、定点量化、循环展开与圆形缓冲、按需预加载DDR数据

**📊 数据集**

使用WSJ0‑2mix进行语音分离，Valentini‑Botinhao（VoiceBank＋DEMAND）进行语音去噪，训练时使用数据增强与SI‑SDR/多分辨率STFT损失

**📈 对比分析**

与ARM Cortex‑A53 CPU基线、NVIDIA A100 GPU对比；FPGA FP32分离延迟44 ms，FP16分离16 ms；FPGA FP32去噪41 ms，FP16去噪9.7 ms，后者低于10 ms阈值；功耗约4 W，能耗-首次样本降低至40 mJ

**⚠️ 局限性**

功耗远超耳机预算，未满足分离任务的10 ms延迟；模型仍为单通道，未评估多麦克风方案；未进行量化感知训练或结构剪枝，且缺乏实际耳机级芯片验证

---

## 86. Notarized Agents: Receiver-Attested Confidential Receipts for AI Agent Actions

**arXiv ID:** 2606.04193 | [PDF](https://arxiv.org/pdf/2606.04193v1)

**作者:** Juan Figuera `[一作]` `[通讯]` (Sello Project), Juan Figuera (Sello Project)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了Sello协议，实现AI代理可观察性的结构性翻转：让服务端而非代理本身签名、加密并公开记录代理调用，供所有者在不信任代理或其运维方的情况下恢复可信行动轨迹。

**💡 创新点**

核心创新是将四项关键属性合并：接收方签名、将授权令牌与拥有者HPKE公钥绑定、在见证者共签的Merkle透明度日志中发布、并由拥有者按令牌引用检索，填补现有方案的空白。

**🔧 技术方法**

采用标准加密原语：COSE_Sign1签名、X25519/HPKE加密、Ed25519签名、ChaCha20-Poly1305 AEAD、Merkle透明度日志、JWS令牌绑定及JSON身份注册表。

**📊 数据集**

无外部数据集；论文仅在本地模拟透明度日志和随机令牌进行微基准测试。

**📈 对比分析**

通过与七个现有receipt协议的属性对比评估，Sello在所有四项属性上均领先；微基准显示服务端生成每条收据平均耗时约0.45 ms，所有者验证每条收据平均耗时约0.28 ms。

**⚠️ 局限性**

局限包括抑制攻击（代理不调用服务无收据）、服务协同、日志完整性缺失、以及缺乏激励机制使服务不主动发出收据。

---

## 87. Gauss Circle Lattices with Geometric Convolutions for Synthesizing High Dimensional Image-Source Room Impulse Responses

**arXiv ID:** 2606.04358 | [PDF](https://arxiv.org/pdf/2606.04358v1)

**作者:** Yuancheng Luo `[一作]` `[通讯]` (NuSpace Audio), Yuancheng Luo (NuSpace Audio)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的计算方法，降低了图像源模型（ISM）在高维空间中生成声学房间脉冲响应（RIR）的计算复杂度。

**💡 创新点**

通过将ISM的格点计数问题转化为经典的高斯圆问题（GCP），将计算复杂度降低到N k^2 log k，从而提高了高维房间的模拟效率。

**🔧 技术方法**

使用了几何卷积和快速傅里叶变换（FFT）等技术来实现高维格点计数和RIR的生成。

**📊 数据集**

使用了整数坐标的高维房间模型，具体数据集未明确提及，但涉及到不同维度的房间反射特性。

**📈 对比分析**

与传统的ISM方法相比，新的GCP-ISM方法在计算复杂度上有显著降低，能够在更高维度和更大范围内生成RIR，同时保持较高的准确性。

**⚠️ 局限性**

该方法在处理非矩形房间和复杂反射系数时存在局限性，且在高维情况下可能导致过高的回声密度。

---

## 88. Structural properties of the implicit function defined by an integral self-consistency equation

**arXiv ID:** 2606.04243 | [PDF](https://arxiv.org/pdf/2606.04243v1)

**作者:** Ivan Viakhirev `[一作]` `[通讯]` (ITMO University), Ivan Viakhirev (ITMO University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对一类积分方程的解存在性、光滑性和结构性质进行严格分析，并给出β(m)=C(m)/m 的符号、单调性和临界点特征；同时在七个对数凹、Beta 型密度以及一个双峰密度上进行数值实验，验证理论结果并探讨临界点的唯一性。

**💡 创新点**

提出了基于积分方程的 β 的符号公式（β′ 与 h 的对数导数加权平均的关系），并证明了单调性从 h 传递到 β，进一步在 h 单峰且满足技术假设时给出了 β 的内部临界点存在性；此外提出了在对数凹 h 下 β 只有一个临界点的猜想，并通过数值实验提供支持。

**🔧 技术方法**

主要采用：积分方程的隐式函数理论（隐函数定理）、对数导数变换、单调性与极值传递、对 β 的符号公式推导；数值层面使用 Brent 算法求解非线性方程、适应性高斯-勒让德积分以及 50 位精度验证；对比分析使用符号变化检测和中值定理。

**📊 数据集**

实验数据集为七个 Beta 形或相关对数凹密度（如 η^a(1-η)^b、η(1-η)^2 e^{-2η}、cos^2(πη/2}）和一个手工构造的双峰密度 ρ(η)∝η(1-η)^2[e^{-(η-0.3)^2/(2σ^2)}+e^{-(η-0.7)^2/(2σ^2)}]，域为 [0,1]。

**📈 对比分析**

通过比较 β′ 的符号变化次数（单峰密度仅出现一次，双峰密度出现三次）验证理论预测；数值精度方面，双精度 Brent 求解和 50 位精度重算均保持相同根，表明算法稳健；在多次网格细化和积分精度变化下，临界点位置稳定在 10^{-5} 以内。

**⚠️ 局限性**

主要限制：假设 (D2) 要求 ρ 在上界 M 处多项式衰减，导致无法直接处理对数正态、指数、伽马等常见网络模型的密度；此外，临界点唯一性仍是猜想，未给出完整证明；对极限 R→∞ 的边界正则化处理亦未得到严格结论。

---

## 89. A Systematic Analysis of Linguistic Features in AI-Generated Text Detection Across Domains and Models

**arXiv ID:** 2606.04177 | [PDF](https://arxiv.org/pdf/2606.04177v1)

**作者:** Yassir El Attar `[一作]` (University of Stuttgart), Agnieszka Falenska `[通讯]` (University of Stuttgart)

**通讯引用:** 457 | [OpenAlex ID](https://openalex.org/A5037220308)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 27 种大型语言模型（LLM）和 10 种文本领域的数据，系统评估了 284 项可解释语言特征在检测 AI 生成文本中的鲁棒性与可迁移性。

**💡 创新点**

证明了词汇丰富度是跨模型、跨领域最稳定的判别特征，并揭示了大部分先前提出的语言信号高度依赖模型或文本上下文；同时提出了基于可解释特征的线性 SVM 检测器在多样化环境下与黑盒方法相当的性能。

**🔧 技术方法**

使用 Python 语言特征提取包（如 Stanza、spaCy 等）提取 284 项特征，构建线性支持向量机（SVM）分类器，并进行特征域 ablation 与交叉模型/领域泛化实验。

**📊 数据集**

基准数据集为 MAGE，包含 27 个 LLM 的生成文本与来自 10 个不同写作任务（意见、新闻、问答、故事、推理、科学等）的人工文本。

**📈 对比分析**

与 FastText、GLTR、Longformer 等基线相比，SVM+语言特征在整体（TB4）上实现 Macro F1≈82.7%，AUROC≈0.99，尽管在 OOD 情况下性能下降，但仍优于多数黑盒模型，且仅比最佳 Transformer 低约 2%。

**⚠️ 局限性**

局限性包括仅覆盖英语数据、未包含最新 LLM、未考虑提示变体、对低资源语言或非英语语言的可迁移性未知；此外部分特征（尤其是词汇丰富度）在不同语言中的定义可能不一致。

---

## 90. EvalStop: Using World Feedback to Detect and Correct Reward Overoptimization in Multi-Tenant RLHF Platforms

**arXiv ID:** 2606.04145 | [PDF](https://arxiv.org/pdf/2606.04145v1)

**作者:** Guilin Zhang `[一作]` (George Washington University), John M. Fossaceca `[通讯]` (George Washington University)

**通讯引用:** 292 | [OpenAlex ID](https://openalex.org/A5057755856)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

提出并实现一种名为 EvalStop 的调度原语，用世界反馈（下游评估分数）来检测并提前终止 RLHF 训练中出现的奖励过度优化（reward overoptimization）任务，从而释放 GPU 并保存最佳检查点。

**💡 创新点**

核心创新点包括：
1) 将调度级别的早停视为检测问题，使用连续下降的评估分数来判断是否已进入过度优化阶段；
2) 设计可组合的包装器，能够在任何基准调度器上叠加，实现灵活集成；
3) 在非clairvoyant环境下，仅依赖可观测的评估分数，而不触及训练损失或奖励模型分数，从而精准区分正常与恶意 RLHF 训练。

**🔧 技术方法**

主要技术手段：
- 基于事件驱动的离散模拟器，用于生成 LoRA、DPO 与 RLHF 的合成训练曲线；
- 采用简单的 k 连续下降阈值检测器（change‑point 简化版）实现早停；
- 将检测结果回馈至任意基准调度器（FIFO、SJF‑Est、SRTF‑Est、LossAware、EvalSched）实现资源重新分配；
- 统计评估指标包括 JCT、TTFUC、废弃/节省计算、精度、召回率和 FPR。

**📊 数据集**

使用的“数据集”是基于公开 RLHF 训练经验所构造的合成曲线集合：包含 200 份工作负载（LoRA、DPO、RLHF），每个 RLHF 任务随机分为 60% 的奖励作弊曲线和 40% 的健康曲线，所有曲线在模拟器中生成。并未使用真实训练日志。

**📈 对比分析**

比较方法：在同一模拟环境下对比多种基准调度器与两类早停竞争者（固定进度停止和损失平坦检测）。结果显示，EvalStop 在 RLHF‑heavy 场景下取得 JCT 下降 9–25%、废弃计算降低 20–22%，并且精度 98%/召回 99%/FPR 1.5%，远优于固定进度停止（高 FPR）和损失平坦检测（低召回）。

**⚠️ 局限性**

局限性：
1) 训练曲线为合成模型，缺乏真实 RLHF 训练日志验证；
2) 评估噪声模型为独立同分布 Gaussian，未覆盖高噪声或自相关噪声情形；
3) 仅考虑单一 GPU 集群，未处理异构硬件或网络延迟；
4) 只对 RLHF 进行了实验，未验证到其他过度优化场景（如 DPO、过拟合）等的泛化。

---

## 91. ACAT: A Collaborative Platform for Efficient Aspect-Based Sentiment Dataset Annotation

**arXiv ID:** 2606.04189 | [PDF](https://arxiv.org/pdf/2606.04189v1)

**作者:** Ana-Maria Luisa Mocanu `[一作]` (National University of Science and Technology POLITEHNICA Bucharest), Elena-Simona Apostol `[通讯]` (National University of Science and Technology POLITEHNICA Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为ACAT的面向ABSA的协同注释平台，支持四种任务（ACSA、Clause‑Level、ATSA、ASTE），并在导出时自动完成ETL、行级对齐与IAA指标计算。

**💡 创新点**

创新点包括：①四种ABSA工作流原生集成，免预配置；②自动化ETL与行级对齐，直接输出可训练的CSV/JSON/XML；③隐式语义切换（Implicit Toggle）捕获无显式词条的方面；④双击交互和时间追踪的高效操作模型。

**🔧 技术方法**

技术实现基于Docker容器（PostgreSQL、Python Flask、Vanilla JavaScript），利用scikit‑learn和statsmodels计算Kappa与F1；采用字符级偏移与坐标字符串序列化；支持多用户并行、实时数据治理。

**📊 数据集**

实验数据集为1,002条餐厅评论，每条由两位注释者标注（经验丰富者与新手）。

**📈 对比分析**

与通用工具对比时需考虑隐藏的ETL与后处理成本，ACAT在单域实验中注释中位时间为31.58秒，Raw Agreement 0.78–0.86，Cohen's Kappa 0.52–0.65，Macro F1 0.50–0.64，显示出高效且中等到良好的一致性。

**⚠️ 局限性**

局限性：仅两名注释者、单一领域，未验证多注释者（Fleiss Kappa）与跨域性能；缺乏并发负载与可扩展性评估；交互模型主要针对桌面，移动端适配有限。

---

## 92. Metric-Aware Hybrid Forecasting for the CTF4Science Lorenz Challenge

**arXiv ID:** 2606.04191 | [PDF](https://arxiv.org/pdf/2606.04191v1)

**作者:** Cen Lu `[一作]` `[通讯]` (EPFL & Idiap Research Institute), Cen Lu (EPFL & Idiap Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个针对CTF4Science Lorenz挑战的多模态混合系统，分别针对短期预测、长期分布匹配和全轨迹重建任务分配不同的预测器；

**💡 创新点**

创新点在于将不同任务的指标拆解后按指标匹配专门化模型（如去噪CNN、Lorenz ODE拟合、直方图尾部替换），并利用长时分布不计时序的特性进行“尾部替换”，实现指标对齐；

**🔧 技术方法**

技术包括Savitzky–Golay平滑、1D CNN/UNet去噪器、Echo State Network基线、Neural ODE、RK4积分、Lorenz参数拟合、直方图模板生成与随机置换；

**📊 数据集**

使用公开的Lorenz轨迹数据集，涵盖不同观测噪声水平和任务对（共九对任务）进行训练和评估；

**📈 对比分析**

与单一模型基线（ESN）和各类混合策略比较，最终从61.00的ESN基线提升到83.85529的最终分数，成熟中间系统已达83.83551；

**⚠️ 局限性**

局限在于系统复杂度高、对各子模型的手工调参和拼接依赖，且在局部验证与公开排行榜间存在不稳定性，未来需进一步自动化集成与鲁棒性提升。

---

## 93. AgenticDiffusion: Agentic Diffusion-based Path Planning for Vision-Based UAV Navigation

**arXiv ID:** 2606.04111 | [PDF](https://arxiv.org/pdf/2606.04111v1)

**作者:** Faryal Batool `[一作]`, Dzmitry Tsetserukou `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种多视角无人机导航框架AgenticDiffusion，实现从自然语言指令到同步FPV与顶部视角的目标定位、任务规划、扩散式轨迹生成与NMPC控制的全流程；

**💡 创新点**

其创新点在于将语言引导的推理、开放词汇目标定位、基于视图的扩散轨迹规划以及控制策略统一到一条管线，并在任务初始化阶段对视角进行自适应选择与规划器匹配；

**🔧 技术方法**

技术实现包括Claude Sonnet 4.6和OpenClaw进行推理、Grounding DINO进行目标定位、图像条件扩散模型生成FPV与顶部视角轨迹、CasADi+acados实现NMPC、ROS2分布式架构；

**📊 数据集**

实验数据来自真实室内环境，使用VICON跟踪、RealSense D435i与Logitech C930ce摄像头采集的FPV与顶部视角图像，并未使用公开数据集；

**📈 对比分析**

通过四个真实室内场景共40次试验（多目标、视角自适应、长距离、着陆点），任务成功率80%，扩散轨迹生成成功率100%，未与现有方法直接对比；

**⚠️ 局限性**

局限性包括对VICON运动捕捉的依赖、受限的室内飞行区域与顶部视角覆盖、NMPC在障碍附近的局部死锁、缺乏对更大复杂环境或完全离线条件下的验证。

---

## 94. When Seeing Is Not Believing -- A Benchmark for Search-Grounded Video Misinformation Detection

**arXiv ID:** 2606.04098 | [PDF](https://arxiv.org/pdf/2606.04098v1)

**作者:** Tao Yu `[一作]` (CASIA), Liang Wang `[通讯]` (CASIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 EVID-Bench 基准，用搜索检索和跨视频推理来检测视频谣言。

**💡 创新点**

创新点在于将视频谣言检测定义为“搜索驱动”任务，强调需要外部视频证据与对比分析，而非仅凭视觉痕迹。

**🔧 技术方法**

技术方案包括链式思维检索计划、迭代检索-验证-反思循环以及检索增强的多模态验证基线，实验使用 GPT‑5.5、Claude、Gemini 等前沿模型。

**📊 数据集**

数据集由 222 条视频组成，覆盖 9 种伪造类型、3 大类（AI 生成、单源编辑、多源编辑）和 6 个主题领域。

**📈 对比分析**

对比实验显示，九个模型中最佳点级准确率仅为 61.43%，视频级准确率 43.24%，AI 生成伪造尤为困难，说明现有模型难以完成完整的跨视频解释。

**⚠️ 局限性**

局限性包括样本量有限、搜索 API 结果可变、评估依赖 LLM 判定且可能存在偏差，以及未来模型可能已能检测当前样本。

---

## 95. Parameter-Efficient Fine-Tuning with Learnable Rank

**arXiv ID:** 2606.04325 | [PDF](https://arxiv.org/pdf/2606.04325v1)

**作者:** Arpit Garg `[一作]` (Australian Institute for Machine Learning), Hemanth Saratchandran `[通讯]` (Australian Institute for Machine Learning)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种可学习权重更新秩的参数高效微调方法 LR-LoRA，允许模型在训练期间自适应地决定每层的低秩适配维度。

**💡 创新点**

创新点在于引入可学习的元素级非线性（以 sinc 基函数实现），从而消除固定秩约束，让适配维度成为可学习属性，而非先验设定。

**🔧 技术方法**

使用的技术包括 LoRA 的低秩更新框架、可学习的 sinc 基非线性、stable rank 衡量方法以及在多种 Transformer 架构上的统一实验框架。

**📊 数据集**

实验使用的主要数据集包括 GLUE、Commonsense Reasoning（BoolQ、PIQA、SocialIQA 等）、视觉迁移（CLIP、DINOv2）以及指令调优的 MT‑Bench，并在 125M~13B 级模型上验证。

**📈 对比分析**

与 LoRA、VeRA、RandLoRA、SineLoRA 等强基线相比，LR-LoRA 在 19 个任务中平均提升 2.3–4.7 分，成为大多数任务的 state‑of‑the‑art 方案，并在低秩限制下保持优越性能。

**⚠️ 局限性**

局限性在于增加了少量超参数（N、I 等），仅在实验中得到验证，缺乏正式的近似理论；此外对 MoE、状态空间网络以及多模态模型的推广、量化推理仍待研究。

---

## 96. Using Text-Based Causal Inference to Disentangle Factors Influencing Online Review Ratings

**arXiv ID:** 2606.04286 | [PDF](https://arxiv.org/pdf/2606.04286v1)

**作者:** Linsen Li `[一作]` (Tulane University), Nicholas Mattei `[通讯]` (Tulane University)

**通讯引用:** 2098 | [OpenAlex ID](https://openalex.org/A5025297787)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对K-12学校在线评论进行因果分析，估计不同话题（如管理、学业表现、欺凌等）对整体评分的影响。

**💡 创新点**

在CausalBERT基础上引入温度缩放校准倾向得分、α超参数调优以缓解过度调整，并结合CLS对比与积分梯度解释方法。

**🔧 技术方法**

使用潜在结果框架、CausalBERT（DistilBERT变体）、温度缩放、α超参数优化、CLS对比分析和积分梯度可解释性技术。

**📊 数据集**

GreatSchools.org 677,210条K-12学校评论（83,795所学校），并配合半合成实验验证。

**📈 对比分析**

与未调整的ATE、Q、IPW、AIPW估计器对比，温度缩放和α调优后，AIPW‑校准在半合成数据中误差比下降至30%以内，在真实数据中得到比未调整更可信的效应估计。

**⚠️ 局限性**

依赖关键词识别治疗、真实因果效应未知、半合成验证可迁移性有限，以及在低强度混杂时模型可能欠拟合。

---

## 97. Plateau That Never Comes: When Efficiency Claims in Datacenters and AI Become Greenwashing

**arXiv ID:** 2606.04214 | [PDF](https://arxiv.org/pdf/2606.04214v1)

**作者:** Harshit Gujral `[一作]` (University of Toronto), Steve Easterbrook `[通讯]` (University of Toronto)

**通讯引用:** 9953 | [OpenAlex ID](https://openalex.org/A5079349226)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个基于再生效应的诊断框架，评估AI与数据中心的可持续性叙事，并揭示其中的绿色洗牌现象。

**💡 创新点**

构建了五项检验（指标、边界、再投资、负担转移、治理）来区分真实的可持续性声明与表面绿化，并将数字充分性作为治理条件引入。

**🔧 技术方法**

主要采用文献综述与案例分析的方法，对企业可持续性报告和学术文献进行概念性阐释；未使用传统实验技术。

**📊 数据集**

利用公开的主要技术公司（Google、AWS、Microsoft、Meta、Equinix）可持续性报告以及相关学术文献中的数据。

**📈 对比分析**

通过对五项检验的应用对比，评估各类可持续性主张的有效性；结果显示多数声明仅通过指标和边界检验，缺乏对负担转移和治理的实证支撑，因而被视为绿色洗牌。

**⚠️ 局限性**

局限性在于框架依赖公开信息与主观评判，缺乏统一的量化指标和实证验证；对再生效应的具体数值推算与跨行业可比性尚待进一步研究。

---

## 98. Federated Learning for Multi-Center Sepsis Early Prediction with Privacy-Preserving

**arXiv ID:** 2606.04338 | [PDF](https://arxiv.org/pdf/2606.04338v1)

**作者:** Xixi Tian `[一作]` (Southwest University), Bin Yi `[通讯]` (Southwest Hospital)

**通讯引用:** 5145 | [OpenAlex ID](https://openalex.org/A5081659458)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究在三家三级医院真实临床数据上使用联邦学习实现败血症早期预测模型的协同训练，并验证其可行性与隐私保护效果。

**💡 创新点**

在保持预测性能接近集中式模型的前提下，首次通过联邦学习与差分隐私噪声注入，系统评估了数据重构攻击的风险，展示了更强的隐私保障与实际可落地性。

**🔧 技术方法**

采用FedAvg联邦平均算法、端到端自动编码器‑MLP模型、列变换器预处理、差分隐私的拉普拉斯机制以及数据重构攻击（DLG）评估。

**📊 数据集**

共648例符合严格入选/排除标准的病人数据，分布于三家医院，包含27个临床特征。

**📈 对比分析**

与基线（KNN+SVM）及集中式训练模型对比，FedAvg模型在ROC‑AUC、敏感度、特异度、准确率等指标与集中式模型相近（差距≤1.5%），同时在无DP条件下重构误差显著高于阈值。

**⚠️ 局限性**

限制在于样本量相对有限、仅采用水平联邦，缺乏纵向或多任务扩展；差分隐私参数的选择仍需进一步平衡隐私与性能。

---

## 99. Spectral Scaling Laws of Muon

**arXiv ID:** 2606.04058 | [PDF](https://arxiv.org/pdf/2606.04058v1)

**作者:** Gagik Magakyan `[一作]` (Massachusetts Institute Of Technology), Asuman Ozdaglar `[通讯]` (Massachusetts Institute Of Technology)

**通讯引用:** 28395 | [OpenAlex ID](https://openalex.org/A5067307504)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Muon 优化器在不同规模 GPT‑2 模型的动量矩阵奇异值谱进行系统分析，发现奇异值分位数在训练后稳定并随模型规模呈幂律缩放。

**💡 创新点**

发现奇异值分位数随层深不同的幂律指数差异显著，可依据该规律为每层选择最小足以正交化主要方向的 Newton–Schulz 迭代步数，从而显著降低正交化计算成本。

**🔧 技术方法**

使用 Newton–Schulz 迭代、奇异值分位数跟踪、量化分位数稳定值、幂律拟合等技术。

**📊 数据集**

在 77M–2.8B 参数的 GPT‑2 风格语言模型上进行实验，使用 Chinchilla 最优 token 数进行预训练。

**📈 对比分析**

通过与低秩正交化（rank‑p）以及不同 NS 步数（5 步 vs 10 步）实验对比，发现保留约 50% 奇异方向即可接近全 Muon 的表现，而 10% 则显著下降；所提出层级化 NS 配置在 300B 规模下仍能保持 50% 方向的正交化而避免额外成本。

**⚠️ 局限性**

局限性包括只针对 GPT‑2 风格模型、未验证其他架构（如 Mixture‑of‑Experts）及其他基于迭代矩阵函数近似的优化器；还未为不同 rank‑p 运行拟合单独的缩放律，可能低估了 NS 对较少方向的正交化能力。

---

## 100. Physics-Informed Machine Learning for Short-Term Flood Prediction

**arXiv ID:** 2606.04143 | [PDF](https://arxiv.org/pdf/2606.04143v1)

**作者:** Tewodros Syum Gebre `[一作]` (North Carolina Agricultural and Technical State University), Leila Hashemi-Beni `[通讯]` (North Carolina Agricultural and Technical State University)

**通讯引用:** 977 | [OpenAlex ID](https://openalex.org/A5034794395)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种将趋势对齐约束直接融入LSTM损失函数的物理信息机器学习框架，用于短期洪水预测。

**💡 创新点**

创新点在于利用简单的趋势对齐与时间平滑约束，无需复杂水动力方程，即可在数据稀缺和极端事件下显著提升模型鲁棒性。

**🔧 技术方法**

采用了标准LSTM网络，并在损失函数中加入趋势对齐（Trend Alignment）和时间平滑（Temporal Smoothness）两项物理约束，同时使用Adam优化器训练。

**📊 数据集**

使用CAMELS-US数据集中的Basin 01022500日降水与流量序列，进行预处理后构造30天窗口的输入序列。

**📈 对比分析**

与传统LSTM基线通过RMSE、NSE和峰值MAPE进行对比；在完整数据下，NSE从0.2441提升到0.2558，峰值MAPE从39.97%略降到39.73%；在仅5%数据稀缺条件下，NSE从0.2037提升到0.2321，提升约14%。

**⚠️ 局限性**

局限性包括对极端峰值幅值的准确捕捉仍有挑战，实验仅针对单一河段，未验证跨河段推广性，且物理约束的权重仍需经验调优。

---

## 101. When Autoregressive Consistency Hurts Safety Alignment

**arXiv ID:** 2606.04168 | [PDF](https://arxiv.org/pdf/2606.04168v1)

**作者:** Bochen Lyu `[一作]` (University of Southampton), Zhanxing Zhu `[通讯]` (University of Southampton)

**通讯引用:** 4562 | [OpenAlex ID](https://openalex.org/A5045305860)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

阐明了安全对齐脆弱的根源——自回归一致性导致梯度更新集中在生成前几步，进而产生浅层对齐，并提出随机插入攻击证明此机制可被利用；随后提出基于对抗安全对齐的框架，并以随机最坏插入训练为具体实现。

**💡 创新点**

①用自回归一致性解释浅层安全对齐的学习动力学；②提出随机插入攻击，展示攻击可以在生成中任意位置诱导有害分支；③构建对抗安全对齐框架，首次将对抗搜索与安全对齐结合；④用最坏插入实现对抗训练。

**🔧 技术方法**

自回归一致性分析（梯度与softmax雅可比）、学习动力学证明、对抗训练框架、随机最坏插入攻击、实验评估（ASR）。

**📊 数据集**

安全数据集：Qi等人提供的安全对齐数据；HEx‑PHI安全基准用于ASR评估；HarmBench用于其他攻击评估；Alpaca数据集用于保持模型效用；此外在多个模型（如LLaMA系列）上测试。

**📈 对比分析**

通过在随机插入攻击和预填攻击下比较ASR，显示传统对齐（SFT）易被突破；深层对齐（prefill）在预填攻击下显著下降但随机插入仍高；对抗安全对齐在随机插入攻击下ASR最低，且在预填攻击与其他攻击（GCG、PAIR、TAP、AutoDAN）上保持竞争力。

**⚠️ 局限性**

仅在单一对抗训练方式（随机最坏插入）上验证，缺乏更广泛的攻击/模型覆盖；对齐效果在极长的有害片段或更复杂的插入策略下尚未测试；对齐与效用的平衡仍需进一步优化；实验成本高，缺乏对更大模型的充分评估。

---

## 102. Neither Layer Alone: Epistemic Integrity Requires Hierarchical Joint Design for Long-Running AI Agents

**arXiv ID:** 2606.04017 | [PDF](https://arxiv.org/pdf/2606.04017v1)

**作者:** Zhihong Shen `[一作]` (Microsoft), Zhihong Shen `[通讯]` (Microsoft)

**通讯引用:** 7538 | [OpenAlex ID](https://openalex.org/A5056112689)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出长期运行 AI 代理的知识完整性（AEI）概念，强调模型与利用层之间需要显式的接口契约，以保持信念、能力和目标的连续性。

**💡 创新点**

创新点在于将 AEI 视为首要架构约束，构建四层层级（目标有效性、动作原型、工具实例选择、调用级失败辨别）以及契约优先的训练与评估方法。

**🔧 技术方法**

采用大语言模型、工具注册表、层级化奖励分解、契约驱动的失败诊断，并在模拟环境中实现和验证。

**📊 数据集**

实验主要使用基于 LLM 的模拟环境和自定义的工具/目标记录，未使用公开数据集。

**📈 对比分析**

对比方法是传统任务成功率与 AEI 维度评估，实验结果表明 AEI 维度能揭示传统评估忽略的跨会话失真。

**⚠️ 局限性**

局限在于缺乏大规模真实数据验证、契约设计需要手工定义，以及模型升级时契约版本管理仍未完善。

---

## 103. Long Live Fine-Tuning: Task-Specific Transformers Outperform Zero-Shot LLMs for Misinformation Response Classification on Reddit

**arXiv ID:** 2606.04274 | [PDF](https://arxiv.org/pdf/2606.04274v1)

**作者:** JooYoung Lee `[一作]` (University of Technology Sydney), Marian-Andrei Rizoiu `[通讯]` (University of Technology Sydney)

**通讯引用:** 1249 | [OpenAlex ID](https://openalex.org/A5069685493)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了多种零射LLM（BART-MNLI、Llama系列、Claude与Gemini）与监督微调Transformer（DistilBERT、RoBERTa）在三条PolitiFact验证的误信息评论上的表现，并探究标签设计与话题对分类性能的影响。

**💡 创新点**

提出标签方案与话题对零射分类的影响，证明细调模型在信念检测方面显著优于任何零射模型，并揭示安全对齐导致的类别偏差。

**🔧 技术方法**

使用BART-MNLI NLI、Llama系列、Claude与Gemini等生成式LLM、DistilBERT与RoBERTa微调，并采用5折交叉验证与排列检验进行评估。

**📊 数据集**

使用900条来自Reddit的评论，覆盖环境、健康、移民三类PolitiFact验证的误信息，按三类（信念、事实核查、其他）均衡标注。

**📈 对比分析**

采用宏F1进行比较，细调RoBERTa得到0.62宏F1，优于最佳零射Claude Haiku 0.50；零射规模不提升，Claude Sonnet在信念检测下降至0.17，表明安全对齐导致误判。

**⚠️ 局限性**

仅评估单一每类误信息，未覆盖多样化主题；未尝试Llama微调或少样本/提示学习；仅单句分类，未利用讨论线程。

---

## 104. Noisy memory encoding explains negative polarity illusions

**arXiv ID:** 2606.04340 | [PDF](https://arxiv.org/pdf/2606.04340v1)

**作者:** Yuhan Zhang `[一作]` (Stanford University), Edward Gibson `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 19253 | [OpenAlex ID](https://openalex.org/A5021445004)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨负极性错觉，并通过失真上下文理论验证其与限定词相似度的关系，使用六个未计时接受性判断实验检验限定词对错觉强度的影响。

**💡 创新点**

提出限定词相似度越高越易引发负极性错觉的新预测，并用失真上下文理论解释其机制。

**🔧 技术方法**

贝叶斯多层序贯秩序回归、词向量余弦相似度、GloVe、fastText、BERT模型计算限定词相似度。

**📊 数据集**

基于Prolific收集的108条句子样本（包含36个关键句子、72个填充句子），六个实验共约50名英语母语者。

**📈 对比分析**

将实验中各限定词对的错觉强度与余弦相似度做对应，发现{few, many}和{few, most}产生显著错觉，余弦相似度与错觉强度呈正相关，说明模型预测成立。

**⚠️ 局限性**

样本量有限、仅使用六个限定词对，且实验为未计时条件，未检验时间压力对错觉的影响，模型预测需在更广泛条件下验证。

---

## 105. Argus-Retriever: Vision-LLM Late-Interaction Retrieval with Region-Aware Query-Conditioned MoE for Visual Document Retrieval

**arXiv ID:** 2606.04300 | [PDF](https://arxiv.org/pdf/2606.04300v1)

**作者:** Abdelrahman Abdallah `[一作]` (University of Innsbruck), Adam Jatowt `[通讯]` (University of Innsbruck)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种名为Argus的查询条件下的视觉文档检索器，它通过区域感知的Mixture-of-Experts对同一页面根据不同查询生成不同的多向量表示，兼容ColPali风格的多向量索引。

**💡 创新点**

创新点在于：① 引入查询感知的区域路由与混合专家，使文档表示成为 𝐃(q) 而非固定；② 通过位置、查询上下文投影融合，保持多向量索引兼容性；③ 使用 1024 维检索头和仅 9% 公开监督数据，显著降低模型规模与训练成本。

**🔧 技术方法**

技术要点包括：Qwen3.5‑VL 视觉语言 backbone；MaxSim late‑interaction 得分；区域池化、查询感知路由（位置 + 查询上下文投影）；Latent Expert Bank +共享专家；门控残差融合；InfoNCE 对比损失 + 负载均衡 + 路由预训练；PEFT/LoRA 微调。

**📊 数据集**

训练使用约 9% 的公开监督数据，涵盖 ViDoRe ColPali、VDR 多语言、VisRAG、TabFQuAD、TatDQA、Docmatix‑IR 子集；评测数据集包括 ViDoRe V1+V2、ViDoRe V3、MIRACL‑Vision（多语言）以及 agentic ViDoRe V3 任务。

**📈 对比分析**

与多种公开最强 late‑interaction 视觉文档检索模型（ColPali、ColNomic、Sauerkraut、Nemotron、Ops‑Colqwen3、Tomoro 等）在 NDCG@5/10/50 等指标上进行对比。Argus‑9B 在 ViDoRe V1+V2 上取得 86.0 的 NDCG@5，超过 Nemotron‑colembed‑8b‑v2 等；在 ViDoRe V3 上 NDCG@10 为 62.5，接近最优；在 MIRACL‑Vision 的宏平均 NDCG@10 为 0.7552，高于 Nemotron‑colembed‑8b‑v2（0.7492）。在 agentic ViDoRe V3 任务中，Argus‑9B 的 NDCG@10 从 60.28 提升至 64.80。

**⚠️ 局限性**

限制包括：仍受 late‑interaction 的存储与查询延迟开销；查询感知路由导致无法一次性索引文档，需两阶段检索-再排序；只评估公开域，未覆盖私有域；未对专家做可解释性约束，专业化仍隐式学习；agentic 性能受限于单一 Qwen3.6‑27B 代理和固定搜索预算。

---

## 106. Discourse-Role Labels as Presentation-Time Variables for Context Use in Language Models

**arXiv ID:** 2606.04109 | [PDF](https://arxiv.org/pdf/2606.04109v1)

**作者:** Jianguo Zhu `[一作]` (Chengdu University of Information Technology), Jianguo Zhu `[通讯]` (Chengdu University of Information Technology)

**通讯引用:** 32244 | [OpenAlex ID](https://openalex.org/A5100370161)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实施了配对固定内容的实验，研究不同对话角色标签（如<ref>、<ex>、<inst>等）对大型语言模型在检索增强生成任务中误导性内容采纳率的影响。

**💡 创新点**

首次将包装标签作为可控变量，揭示其对模型采纳率的显著差异，并提出在上下文利用基准中记录和控制包装标签的实用规范。

**🔧 技术方法**

采用配对实验设计、McNemar统计检验、配对自举置信区间、最终指令消融、log‑prob 预评估、嵌套标签冲突分析以及短答手工审计等多种方法对模型行为进行细粒度评估。

**📊 数据集**

以MMLU‑Pro的500条多项选择题为主实验集，辅以GSM8K、中文+英文混合标签测试、短答输出以及嵌套标签冲突样本，覆盖不同任务和语言场景。

**📈 对比分析**

在四大模型（GPT‑5.5、DeepSeek V4 Pro、Llama‑3‑8B‑Instruct、Qwen2.5‑7B‑Instruct）上对比，发现绑定/来源标签的误导采纳率比示例标签高56–84个百分点，且与最终指令强度、预先log‑prob差异显著，验证了标签效应的稳健性。

**⚠️ 局限性**

实验受限于受控环境，未覆盖完整检索、索引、重排等部署流程；标签效果随语言、任务与模型而异；短答评判依赖单一手工审计，可能存在标注偏差。

---

## 107. Adaptive Patching Is Harder Than It Looks For Time-Series Forecasting

**arXiv ID:** 2606.04074 | [PDF](https://arxiv.org/pdf/2606.04074v1)

**作者:** Federico Zucchi `[一作]` (University of Strasbourg), Ziyue Li `[通讯]` (Technical University of Munich)

**通讯引用:** 606 | [OpenAlex ID](https://openalex.org/A5029997962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对时间序列Transformer中的自适应分块（dynamic patching）进行理论阐述与实证评估，比较了调优后的均匀分块与动态分块的效果。

**💡 创新点**

提出了基于信息论的阈值与二阶上界，说明只有在分块信号与损失高度相关且存在预算约束时，自适应分块才能优于均匀基线，并在实验中验证了这一结论。

**🔧 技术方法**

采用了率失真理论、Jensen不等式、二阶泰勒展开等数学工具来推导阈值与上界；实验中对EntroPE、TimeMosaic、HDMixer三种动态分块方法做统一分块替代，并在多数据集上进行评测。

**📊 数据集**

使用的公开数据集包括ETT系列（ETTh1/ETTh2/ETTm1/ETTm2）、Weather、Electricity、Traffic和Exchange。

**📈 对比分析**

在保持相同模型、优化器、训练预算和预处理的前提下，先对均匀分块进行调优，然后与对应的动态分块进行对比；结果显示，均匀分块在大多数设置下与动态分块相当或更优，性能差距仅在±2%以内，速度提升可达1–5×。

**⚠️ 局限性**

自适应分块的收益高度依赖于对“局部复杂度”定义和估计的准确性；缺乏对分块信号与损失关联的理论指导，导致实际收益受限；此外，路由的计算开销往往抵消潜在收益，限制了其实际应用。

---

## 108. Training-Free Lexical-Dense Fusion for Conversational-Memory Retrieval

**arXiv ID:** 2606.04194 | [PDF](https://arxiv.org/pdf/2606.04194v1)

**作者:** Christian Lysenstøen `[一作]` `[通讯]` (Inland Norway University of Applied Sciences), Christian Lysenstøen (Inland Norway University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在长时对话记忆检索中，验证并扩展了基于会话级别的“turn isolation retrieval”（TIR）—即通过最大化查询-回合相似度进行会话打分，并在此基础上加入BM25词汇检索的分数级联，形成训练无关、CPU 仅检索阶段的高效检索策略。

**💡 创新点**

创新点在于系统性地评估并证明：1）即使已采用晚期交互的稠密检索，加入词汇检索仍能显著提升性能；2）不同聚合算子（max-sim、top-k、smooth-max）在不同编码器上表现差异明显，需选取尺度无关的算子；3）常见的跨搜索的交叉编码器重排序在此场景下反而降低性能；4）对多种评估维度与语料库（LoCoMo、LongMemEval）进行细粒度对比，揭示词汇与稠密检索的交替优势。

**🔧 技术方法**

技术手段包括：固定检索单元为会话；使用多种预训练CPU bi-encoder（如gte-base、bge-base、e5-large-v2等）对查询和回合进行向量化；通过分数级联（BM25与稠密分数的z-归一化后线性组合）实现词汇+稠密融合；对late interaction的多种聚合算子（max-sim、top-k、smooth-max）进行 ablation；对 RRF 及交叉编码器重排序进行实验。

**📊 数据集**

使用的基准数据集为 LoCoMo（1978个 QA 示例，10 轮对话）和 LongMemEval‑S（150 个检索挑战问题，包含多轮会话）。

**📈 对比分析**

在 LoCoMo 上，最大化稠密-最大相似度的检索已优于均值池化；加入 BM25 后 Hit@1 进一步提升 8.8–17.2个百分点，最高达 0.752；在 LongMemEval‑S 上，词汇检索已接近饱和，稠密+词汇融合的提升不显著。相较于单一 BM25 或稠密检索，融合策略在多种评估指标（Hit@1、Recall@5、NDCG@5 等）上均表现更佳，且提升显著性均在 p<10^-4。

**⚠️ 局限性**

局限性包括：1）实验仅覆盖两大基准，单一对话聚类抽样导致统计显著性受限；2）未涉及大规模 GPU 训练或复杂的图/分割记忆架构；3）对跨搜索交叉编码器的负面结果仅针对一类模型，可能不具普适性；4）在词汇覆盖强的 LongMemEval‑S 上融合优势不显著，表明方法受语料库特性的限制。

---

## 109. SymTRELLIS: Symmetry-Enforced Voxel Latents for 3D Generation

**arXiv ID:** 2606.04108 | [PDF](https://arxiv.org/pdf/2606.04108v1)

**作者:** Guangda Ji `[一作]` (Simon Fraser University), Hao Zhang `[通讯]` (Simon Fraser University)

**通讯引用:** 155348 | [OpenAlex ID](https://openalex.org/A5100399276)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 SymTRELLIS 方法，在 TRELLIS.2 的流式生成过程中通过速度对称化和空间变换潜在映射器实时强制对象满足任意有限点群对称性；

**💡 创新点**

创新点在于不需要重新训练基础模型，仅通过学习潜在空间的线性变换操作并在每个 ODE 步骤对流场进行对称平均，实现对旋转、反射及多面体对称性的即时强制；

**🔧 技术方法**

技术包括流匹配模型（Flow Matching）、潜在空间对称化（Velocity Symmetrization）、空间变换潜在映射器（Spatial‑Transform Latent Mapper）、自动对称检测与手工指定对称组；

**📊 数据集**

使用通用 3D 数据集（Sketchfab/Objaverse‑XL）训练映射器，Toy4K 评估映射器性能，构建 266 个严格对称网格的基准数据集（涵盖 2–20 旋转对称、反射、多面体对称）；

**📈 对比分析**

与基线 TRELLIS.2、Hunyuan3D‑2.1、TripoSG 在同一输入图像下比较，SymTRELLIS 在对称误差（SD_max/SD_avg、阈值误差）显著降低、旋转阶数预测准确率提升，重建精度（Chamfer）基本保持不变；

**⚠️ 局限性**

局限性包括仅能强制全局外在对称，无法处理局部或内在对称；对称化过程可能平滑细节导致细节损失；偶尔生成空活跃体素，且方法仅适用于基于体素的潜在空间，无法直接迁移至隐式或 token‑based 模型。

---

## 110. The Digital Apprentice: A Framework for Human-Directed Agentic AI Development

**arXiv ID:** 2606.04321 | [PDF](https://arxiv.org/pdf/2606.04321v1)

**作者:** Travis Weber `[一作]` (Pheo Inc), Rohit Taneja `[通讯]` (Pheo Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Digital Apprentice 框架及其在推理时的 ADAPT 控制平面，旨在通过逐技能自治分层、授权门槛与持续对齐，构建可扩展、安全的 AI 代理，能够在人类指导下逐步获得自主权。

**💡 创新点**

创新点包括：①以技能为粒度的自治状态机和基于实证的晋升/降级机制；②将多策略推理、质量遥测与偏好发射构成的闭环，用于实时治理与学习；③多维度质量刻度与漂移检测，支持动态策略切换；④将人类纠正记录为组织拥有的持久偏好数据，实现持续对齐。

**🔧 技术方法**

技术手段：多策略生成（RAG、最佳 N 采样、方法学条件生成）、多维度质量评分器（LLM‑as‑judge、奖励模型）、自动与人工偏好权重、偏好发射与对齐、滚动窗口质量监测、离散化分层晋升、差异化融合与多样性门控。

**📊 数据集**

数据集：公开专业方法学语料库；使用 Qwen 生成模型和 Gemma 评估模型，评估通过 OpenRouter 接口完成。

**📈 对比分析**

比较方法与性能：在同一语料上对比三种策略（纯 RAG、最佳 N 采样、融合）以及结构化上岗前后效果。结构化上岗后六维质量平均分从 0.717 提升至 0.957；在漂移场景下，原策略降至 0.930，切换至多样性门控后恢复至 0.957。实验仅在推理阶段进行，未包含真实人类评估或统计显著性检验。

**⚠️ 局限性**

局限性：① 隐式知识的逆问题难以完全捕捉；② 需要同意、保密与合规，涉及隐私与法规挑战；③ 低纠正率可能导致假装胜任；④ 对多模态或更广泛任务的泛化不确定；⑤ 实验缺乏置信区间、显著性检验和真实人类标注；⑥ 目前仍处于原型阶段，尚未在生产环境广泛验证。

---

## 111. PerceptTwin: Semantic Scene Reconstruction for Iterative LLM Planning and Verification

**arXiv ID:** 2606.04226 | [PDF](https://arxiv.org/pdf/2606.04226v1)

**作者:** Charlie Gauthier `[一作]` (Université de Montréal), Liam Paull `[通讯]` (Université de Montréal)

**通讯引用:** 4385 | [OpenAlex ID](https://openalex.org/A5037065865)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个完全自动的 real2sim 体系，将机器人感知得到的开放词汇 3D 场景图转换为交互式仿真环境，并通过 LLM 规划与评判实现迭代计划改进与安全验证。

**💡 创新点**

创新点包括：① 用 CLIP 与 Objaverse 自动检索或 TRELLIS 生成 3D 资产；② 基于 LLM 的物体可操作性预测与约束；③ 引入 AI 对齐理念的“计划评判者”进行逻辑与安全双重验证；④ 形成闭环的 LLM 规划反馈流程。

**🔧 技术方法**

核心技术包括：SAM 语义分割、CLIP 视觉-文本嵌入、TRELLIS 2D‑to‑3D Transformer、Objaverse 3D 资产库、AI2Thor 仿真平台、SMART‑LLM 规划器、GPT‑5 系列 LLM 评判器。

**📊 数据集**

使用的数据集与素材：ConceptGraph 通过 LoCoBot + Intel RealSense 采集的 3D 场景图、Objaverse 3D 资产库、TRELLIS 训练图像、AI2Thor 环境，并在 Blocks、Cones、Veggies、Backyard、Bomb 等任务场景中进行实验。

**📈 对比分析**

与无反馈 SMART‑LLM 基线对比，评估 Plan Success、Safety、Human 预测准确率等指标；实验显示 LLM 计划成功率平均提升约 39%，人类计划验证准确率提升至 18%，且在复杂任务与对抗性场景中显著降低误报与危险行为。

**⚠️ 局限性**

局限性包括：① 只能在 AI2Thor 框架内动态更改物体状态，非自定义资产受限；② 需要高性能 GPU 进行 TRELLIS 生成，耗时较长；③ 仅验证单臂机器人，未覆盖多臂或复杂动力学；④ LLM 评判器可能出现误判，需进一步鲁棒性验证。

---

## 112. veriFIRE: an Industrial Case Study in Verifying Consistency Properties for a DNN-Based Wildfire Detection System

**arXiv ID:** 2606.04121 | [PDF](https://arxiv.org/pdf/2606.04121v1)

**作者:** Idan Refaeli `[一作]` (Hebrew University of Jerusalem), Guy Katz `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 2556 | [OpenAlex ID](https://openalex.org/A5102986148)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了针对工业级野火检测系统的端到端形式化验证方法，验证了目标强度单调性和模糊容忍正向检测两种应用特定一致性属性。

**💡 创新点**

创新点在于将物理意义的输入变换（目标强度缩放、传感器模糊模型）通过网络预置层或逼近子网络编码为标准可验证查询，克服了传统验证框架只能处理线性扰动的限制；同时在真实飞行背景数据上完成大规模验证。

**🔧 技术方法**

使用现有深度网络验证器（α,β‑CROWN）和ONNX/VNNLIB格式；构造线性预置层验证单调性，训练逼近传感器模糊的神经网络与插值映射验证模糊容忍性；采用多GPU/Slurm分布式求解。

**📊 数据集**

使用由真实航拍记录构成的25×25×2红外背景图像集合（2011个样本），并在其中注入合成目标信号，形成背景-目标对以及用于模糊验证的背景子集。

**📈 对比分析**

通过对2011个单调性查询（α_max=2）和1698个模糊查询进行求解，单调性查询全部在5分钟内完成，其中56.39%得UNSAT（属性成立），43.61%得SAT（发现反例）；模糊查询中仅212个SAT、548个UNSAT，1251个超时，平均UNSAT求解时间约8014s，显著高于单调性。

**⚠️ 局限性**

主要限制在于高维模糊参数空间导致的可验证性规模化困难；逼近模糊子网络的误差、求解超时以及对更大、更复杂工业模型的适用性尚待改进。

---

## 113. Expert-Aware Refusal Steering

**arXiv ID:** 2606.04160 | [PDF](https://arxiv.org/pdf/2606.04160v1)

**作者:** Anna C. Marbut `[一作]` (University of Montana), Travis J. Wheeler `[通讯]` (University of Arizona)

**通讯引用:** 7172 | [OpenAlex ID](https://openalex.org/A5073174813)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Mixture-of-Experts 大型语言模型中研究并实现了拒绝行为的白盒 jailbreaking 方法，并提出了基于专家路由的拒绝 steering 技术。

**💡 创新点**

创新点在于将 ActAdd 转移到 MoE 结构中，设计了单专家与全专家两种专家感知拒绝 steering，并揭示注意力块对拒绝行为的主要作用。

**🔧 技术方法**

使用 ActAdd 向量添加、MoE 路由与专家特定方向计算、GPT‑4o 评判器以及基于向量差分的 steering 计算。

**📊 数据集**

使用 JailbreakBench 提供的有害与无害提示作为实验数据，评估三种 MoE 模型：GPT‑OSS 20B、Mixtral8x7B Instruct、OLMoE 1B‑7B Instruct。

**📈 对比分析**

与基线和原始 ActAdd 方法比较，三种专家 steering 的成功率平均占 ActAdd 的 54–79%，整体 ASR 在 65–95% 之间，单专家方法约 66% 的余量。

**⚠️ 局限性**

主要限制是仅在开源、规模较小的 MoE 模型上实验，缺乏对大型商业模型的验证，且方法依赖模型白盒可访问性。

---

## 114. UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing

**arXiv ID:** 2606.04101 | [PDF](https://arxiv.org/pdf/2606.04101v1)

**作者:** Xinming Wei `[一作]` (Peking University), Guojie Luo `[通讯]` (Peking University)

**通讯引用:** 1950 | [OpenAlex ID](https://openalex.org/A5023468643)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了面向大型专家并行（large‑EP）MoE模型的实时精确负载平衡系统 UltraEP，在 rack‑scale 节点上实现了每个 microbatch 和层的即时重分布。

**💡 创新点**

首次将配额驱动的规划与 RSN 原生专家状态传输结合，利用实时负载、阈值二分搜索生成最优复制与重路由方案，并通过持久 tile 流式和 chunk relay 缓解热专家的 fan‑out。

**🔧 技术方法**

GPU 原生配额规划、warp 并行二分搜索、RSN 设备内一侧共享内存通信、持久 tile 流式、动态 chunk relay、DeepEP token all‑to‑all 内核等。

**📊 数据集**

评估使用 Qwen3‑235B、GLM4.5‑106B、GLM4.7‑358B、DeepSeek‑V3、内部 RefMoE‑288B；训练使用 200 B 及 15 T 语料；推理使用 STEM 与混合任务的 Poisson 请求场景。

**📈 对比分析**

与 Megatron‑LM、SGLang、EPLB、LPLB、EPLB+ 等基线对比；在训练中平均达 94.6 % 理想吞吐，提升 1.42‑1.49×；在预填充中 93.9 % 理想吞吐，提升 1.56×；在 2560 GPU 级别训练保持 92 % 理想吞吐。

**⚠️ 局限性**

仍依赖 RSN 高带宽结构，且对极端负载漂移或跨节点专家分布的鲁棒性有限；实现复杂度高，需要 GPU 原生调度与通信；在标准 RDMA 或较小规模集群上的效果未验证。

---

## 115. Toward a Generalized Defense Across Sparse, Continuous, and Structured Parameter Attacks

**arXiv ID:** 2606.04317 | [PDF](https://arxiv.org/pdf/2606.04317v1)

**作者:** Bin Duan `[一作]` (University of Queensland), Guowei Yang `[通讯]` (University of Queensland)

**通讯引用:** 1448 | [OpenAlex ID](https://openalex.org/A5039642499)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种通用的深度学习模型防御框架，旨在抵御稀疏、连续和结构化参数攻击，且无需重新训练。

**💡 创新点**

创新点在于将键控通道重参数化（KCR）、准循环低密度奇偶校验码（QC‑LDPC）量化和自适应鲁棒推理（ARI）三种互补机制融合，形成一条从准备到推理的完整防御链路，显著提升对多类参数攻击的鲁棒性。

**🔧 技术方法**

使用技术包括：键控通道重参数化（随机通道置换与缩放）、量化后编码为QC‑LDPC码（提供错误纠正与冗余），以及推理时的随机平滑与置信门控投票（ARI）以降低攻击影响。

**📊 数据集**

在CIFAR‑10、CIFAR‑100、Tiny‑ImageNet上评估VGG16和ResNet32模型，并在ImageNet‑1K与CIFAR‑100上扩展到DeiT‑Tiny/Small Transformer；实验覆盖稀疏(bit‑flip)、连续(L₂/∞)和结构化攻击。

**📈 对比分析**

与基准（Base、BIN、RA‑BNN、Aegis）比较时，框架在保持接近原始准确率的同时，将攻击成功率降低30%–70%；模型大小缩减约70%，推理延迟仅提升5–9%，且对多类攻击均保持稳健。

**⚠️ 局限性**

局限性包括：需要可信执行环境或安全密钥存储以实现KCR的密钥保护；仅对离线（预加载）参数篡改有效，对运行时内存攻击需配合硬件级安全；在极大模型或生成模型上的适用性尚未验证；攻击者若能完整重现KCR变换或诱发解码失败，仍可能削弱防御。

---

## 116. Scaling Novel Graph Generation via Lightweight Structure-Guided Autoregressive Models

**arXiv ID:** 2606.04287 | [PDF](https://arxiv.org/pdf/2606.04287v1)

**作者:** Alessio Barboni `[一作]` (Boise State University), Edoardo Serra `[通讯]` (Boise State University)

**通讯引用:** 957 | [OpenAlex ID](https://openalex.org/A5009094578)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种轻量级自回归图生成框架，利用结构引导的拓扑序列化和二进制边流实现近对数线性复杂度，并通过两阶段训练（探索扰动+GMM修正）提升图的多样性与新颖性。

**💡 创新点**

创新点包括：① 结合SIR-GN结构表示与BFS搜索的结构引导节点排序，生成正则化的二进制边序列；② 两阶段训练策略，先用扰动图扩展探索，再用嵌入空间的GMM进行自监督细化；③ 同时支持LSTM和Mamba两种轻量级自回归序列解码器，兼顾小分子与大稀疏图；④ 在大内存GPU（如GH200）上实现长序列训练与采样。

**🔧 技术方法**

使用的技术：SIR-GN结构嵌入、BFS拓扑排序、二进制边流序列化、LSTM/Mamba自回归解码器、ReST式自训练与GMM筛选、嵌入空间判别、CUDA/Grace Hopper大内存加速。

**📊 数据集**

实验数据集：分子域的QM9、ZINC250K、MOSES、ANI‑1x、QM7x、Transition1x；非分子域的MalNet‑Tiny函数调用图。

**📈 对比分析**

与PARD等基线在同一任务下进行比较；在所有分子基准上，Novelty显著提升（如QM9从37%提升至97%），Validity和Uniqueness保持≈100%；FCD较高表明生成分子更远离训练分布；Mamba在MalNet‑Tiny长序列实验中可行；不同GPU上展示了训练时长、内存占用和生成速度。

**⚠️ 局限性**

局限性：对长序列仍高度依赖大内存GPU；二进制序列对节点序列化敏感，结构指导失效会导致性能骤降；GMM过滤在非分子域需要外部判别器；未在3D几何或更复杂属性上进行评估；在多家族MalNet‑Tiny上尚未进行全面测试。

---

## 117. Neural Galerkin Normalizing Flows for Bayesian Inference of Diffusions with Inaccessible Boundaries

**arXiv ID:** 2606.04324 | [PDF](https://arxiv.org/pdf/2606.04324v1)

**作者:** Riccardo Saporiti `[一作]` (Ecole Polytechnique Federale de Lausanne), Fabio Nobile `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对完全观测的随机微分方程（如Heston、SVCEV模型），构建并训练一种基于Neural Galerkin框架的归一化流模型，用以逼近其在离散观测时刻之间的转移密度函数，从而实现高效的贝叶斯参数推断。

**💡 创新点**

创新点包括：①提出了一种以截断高斯混合为核心的 bounded Normalizing Flow 架构，能够自然满足在无访问边界（如Feller条件下的零扩散区）上的 Dirichlet 边界条件；②将该流嵌入 Neural Galerkin ODE 优化中，使得流参数随时间演化，且与 SDE 参数、初始条件共参，形成真正的参数化转移密度逼近；③通过离线训练获得的流参数实现在线贝叶斯推断时几乎无计算量（仅需前向传播），大幅提升采样效率。

**🔧 技术方法**

核心技术包括：Neural Galerkin 方法、带截断高斯混合的 Normalizing Flow、GRU 网络用于生成层级混合参数、LSMR 最小二乘求解残差、Slice Sampling MCMC、傅里叶变换对比、Aït‑Sahalia 级数展开、Itô‑Taylor 级数展开以及数据增强/重要采样参考。

**📊 数据集**

使用的“数据集”为在 Heston 和 SVCEV 模型下通过数值模拟生成的合成观测轨迹：Heston 共 350 条观测（Δ=0.5）和 200 条低频观测（Δ=1）；SVCEV 共 100 条观测（Δ=0.5）。

**📈 对比分析**

对比方法包括：Aït‑Sahalia 2 阶封闭形式展开、4 阶 Itô‑Taylor 展开、Heston 的 Fourier 变换精确密度、SVCEV 的数据增强（DA）重要采样参考。结果显示：NF 在对数似然误差、Wasserstein‑2 距离、以及采样运行时间上均优于或接近传统方法；在低频、非中心参数值下，NF 的误差显著低于展开方法，并且运行时间是 Fourier 变换的 1/6 左右，数据增强的 45 倍左右。

**⚠️ 局限性**

局限性包括：①假设目标密度可用有界支持的分布逼近，导致在尾部和极端事件上精度不足；②目前仅适用于完全观测的过程，尚未与粒子滤波或其他部分观测方法结合；③对参数空间的高维扩展尚未彻底验证，且对边界条件的处理依赖于先验选取的截断分布。

---

## 118. Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent

**arXiv ID:** 2606.04031 | [PDF](https://arxiv.org/pdf/2606.04031v1)

**作者:** Ahanaf Hasan Ariq `[一作]` `[通讯]` (Ideal School and College), Ahanaf Hasan Ariq (Ideal School and College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究耦合梯度下降的非正态动力学，利用伪谱理论给出块三角雅可比矩阵的 Kreiss 常数上界与下界，并据此导出两时间尺度随机优化的非渐近样本复杂度标度。

**💡 创新点**

创新点在于①提出针对块三角雅可比矩阵的精确 Kreiss 常数上界，并给出匹配的下界与临界耦合阈值；②扩展到近似自参照系统的 Neumann 级数扰动分析；③基于 Kreiss 常数推导两时间尺度随机优化的迭代复杂度标度。

**🔧 技术方法**

使用了伪谱理论、Kreiss 矩阵定理、块解算子法、Neumann 级数扰动、最小化下界证明以及实验验证。

**📊 数据集**

实验使用的数据集包括线性–二次合成模型（p=q=50）、IQC 对比同类问题以及 2D 高斯混合数据用于神经网络训练。

**📈 对比分析**

与 IQC 约束法对比，伪谱上界在同一问题上比 IQC 上界 2–5 倍更紧；实验结果表明理论上界与实际最大增幅相差不超过 2 倍，并验证了神经网络训练中峰值时刻与理论预测一致。

**⚠️ 局限性**

局限性包括：①假设 A、D 对称，无法覆盖非对称情况；②上界与下界之间仍有 2/(1-γ) 与 1/(1-γ) 的两倍差距；③扰动分析仅在 εB_0 K_0 < (1-γ) 的弱耦合下有效；④最坏情况出现因子 e n；⑤仅适用于局部线性化。

---

## 119. DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities

**arXiv ID:** 2606.04205 | [PDF](https://arxiv.org/pdf/2606.04205v1)

**作者:** Sajad Ebrahimi `[一作]` (University of Toronto), Ebrahim Bagheri `[通讯]` (University of Toronto)

**通讯引用:** 8495 | [OpenAlex ID](https://openalex.org/A5064660738)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提供了一个统一多模态的开源工具 DetectZoo，用于统一评估文本、图像和音频的 AI 生成内容检测方法。

**💡 创新点**

集成61种检测算法、22个公共数据集，并设计统一 API 与标准化评估管线，实现跨模态可重复比较。

**🔧 技术方法**

采用注册式轻量化组件、统一数据抽象、自动下载缓存、预训练权重管理以及现有检测方法的实现。

**📊 数据集**

22个跨模态基准数据集，包括 HC3、M4、RAID、ForenSynths、GenImage、ASVspoof 2019 等。

**📈 对比分析**

通过统一评估器计算 AUROC、PR-AUC、EER 等指标，对 61 检测器进行大规模对比，验证与原论文结果一致，揭示跨模态差异。

**⚠️ 局限性**

仅提供评估而不含训练管线；音频覆盖不足；部分检测依赖外部模型；数据集为静态快照；缺乏视频支持。

---

## 120. Incremental Sheaf Cohomology on Cellular Complexes: O(1)-in-n Lazy Edit Processing under Bounded Local Geometry

**arXiv ID:** 2606.04227 | [PDF](https://arxiv.org/pdf/2606.04227v1)

**作者:** Jason L. Volk `[一作]` `[通讯]` (Invariant Research), Jason L. Volk (Invariant Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种算法框架，用于在动态演变的1维细胞复合体上增量维护第一层细胞同调H^1(X; )，该复合体配备有限维细胞层。

**💡 创新点**

通过局部几何假设，证明了每次编辑只影响有限的局部共边界块，从而实现了O(1)的增量更新，显著提高了计算效率。

**🔧 技术方法**

使用了增量维护定理、局部更新和Mayer-Vietoris全局组装等技术，结合了局部特征值求解和全局组装。

**📊 数据集**

在Barabasi-Albert图上进行了实验，图的规模达到5 × 10^6个顶点和1.7 × 10^7次流编辑。

**📈 对比分析**

与传统的O(n^3)全重计算方法相比，该算法在每次编辑时的延迟为35微秒，且在同步时的全局组装时间为O(n)。

**⚠️ 局限性**

局部几何假设可能不适用于所有复杂结构，且当前算法未处理删除操作，可能导致不一致性。

---

## 121. Exact Unlearning in Reinforcement Learning

**arXiv ID:** 2606.04182 | [PDF](https://arxiv.org/pdf/2606.04182v1)

**作者:** Thanh Nguyen-Tang `[一作]` (New Jersey Institute of Technology), Raman Arora `[通讯]` (Johns Hopkins University)

**通讯引用:** 4403 | [OpenAlex ID](https://openalex.org/A5063457506)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在强化学习中提出精确删除（Exact Unlearning）框架，设计了既能实现低退化又满足 ρ‑TV 稳定性的 Tabular MDP 学习算法，并给出了可实现的高效删除算法，证明其退化成本仅为重新训练成本的 ρ√(ln T) 分量；

**💡 创新点**

首次将 TV‑稳定性与强化学习相结合，利用二叉树高斯噪声与最大耦合实现精确删除，同时给出了近最优的 regret‑稳定性下界，填补了 RL 领域缺失的精确删除理论空白；

**🔧 技术方法**

主要技术包括：总变差（TV）稳定性定义、二叉树机制注入高斯噪声、最大耦合（maximal coupling）实现高效删除、基于 UCB‑VI 的稳健估计与奖励上界；

**📊 数据集**

实验和分析基于标准的离散 MDP（tabular）环境，使用合成状态、动作、时间步长数据；

**📈 对比分析**

与传统 UCB‑VI、DP‑RL 等方法对比，证明在保持相同 regret 的前提下，删除成本显著降低（≈ρ√(ln T) 级别），且整体 regret 上界为 𝑂(H²√SAT + H³S²A + H²⋅⁵S²A/ρ)，接近下界；

**⚠️ 局限性**

局限性包括：仅针对离散 Tabular MDP；与下界存在 H 与 S 维度的系数差距；空间复杂度线性于总训练回合 T；未覆盖连续或函数逼近场景，需进一步研究以实现更高维、低存储的精确删除。

---

## 122. OpenRFM: Dissecting Relational In-Context Learning

**arXiv ID:** 2606.04320 | [PDF](https://arxiv.org/pdf/2606.04320v1)

**作者:** Zhikai Chen `[一作]` (Michigan State University), Kai Guo `[通讯]` (Michigan State University)

**通讯引用:** 566 | [OpenAlex ID](https://openalex.org/A5066265467)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了开源关系基础模型（RFM），提出了OpenRFM框架，解决了现有Relational Transformer（RT）在关系级标签稀缺和预训练阶段“懒惰”学习所带来的性能瓶颈。

**💡 创新点**

创新点包括：1）双阶段ICL架构，将关系级RT与批量级Tabular ICL层结合，显著缓解标签稀缺问题；2）同类性（homophily）控制的合成+真实混合预训练策略，注入可识别的关系潜变量，使模型从懒惰迁移至特征学习；3）在训练中加入原型正则化与支持级集成，提升多类别读出与回归表现。

**🔧 技术方法**

技术手段包括Relational Transformer、Tabular ICL（TabICL/TabPFN）、双向/多层注意力、同类性增强的合成数据库生成、原型损失（prototype loss）、支持级集成（support-side ensembling）。

**📊 数据集**

使用的数据集涵盖RelBench‑v1、RelBench‑v2、Kaggle多任务、真实数据库集合（rel‑amazon、dbinfer‑outbrain‑small、dbinfer‑diginetica、dbinfer‑retailrocket）以及Synthetic PluRel数据库。

**📈 对比分析**

通过与多种基线（训练从零的GNN、JUICE、RT‑Plurel、Griffin、KumoRFMv1等）在24个任务（14二分类、2多分类、8回归）上的对比，OpenRFM平均提升约30%相较RT基线，并在大多数任务上超过商业模型KumoRFMv1，整体排名靠前。

**⚠️ 局限性**

局限性包括：1）对极大类别任务（如rel‑salt 16‑way）仍受读出容量限制；2）未能直接扩展至推荐任务；3）双阶段ICL的互补机制尚未完全理解，存在更高效耦合的潜在改进空间。

---

## 123. Hardness as an Information Constraint: A Unifying Meta-Complexity Assumption

**arXiv ID:** 2606.04257 | [PDF](https://arxiv.org/pdf/2606.04257v1)

**作者:** Hunter Monroe `[一作]` `[通讯]`, Hunter Monroe

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

提出一种基于信息约束的框架，用随机性（Kolmogorov 随机字符串）来解释复杂性理论中的主要难点问题（如无最优证明系统、PH 非崩塌、密码学和随机化算法的限制等）。

**💡 创新点**

创新点在于将“弱基底相容性”与随机性信息相结合，提出 Kolmogorov Hardness (KH) 及其扩展（有限尺度、层次级、平均情形、无互助等）作为统一的证明难度假设，并给出它们与经典证明障碍（如自然证明、Razborov–Rudich、Impagliazzo–Wigderson）之间的桥接关系。

**🔧 技术方法**

主要技术包括：
- 证明系统与算术理论的相互模拟与可解释性（Higher Relative Consistency 与 Feasible Reflection）；
- Kolmogorov 复杂度与随机性谓词的正式化；
- 通过有限规模与层次级强化，构造稠密难题族和显式分离器；
- 证明与算法层面的桥接（时间限制 Kolmogorov 复杂度、Liu–Pass 等）以推导一位函数、自然证明障碍、伪随机生成与 Feige 随机化假设。

**📊 数据集**

本研究为理论论文，无实验数据或数据集；所有结果均基于算术公理、Kolmogorov 复杂度和证明系统的形式化推导。

**📈 对比分析**

比较方法：作者将 KH 与传统假设（如 PH 非崩塌、Karp–Lipton 结论、自然证明阻碍）进行对照，说明 KH 在不需要额外可归约的情况下即可推导这些结果；在层次级情形下给出显式分离语言，证明其密度和复杂度上界；在平均/随机情形下通过边界反射推导一位函数和随机化算法限制；在 Feige 情形下给出稀疏 Kolmogorov 随机性判别器的密度与稀疏性分析。

**⚠️ 局限性**

局限性：
- KH 与其扩展目前尚未得到形式独立性证明，缺乏非标准模型或切割构造；
- 桥接假设（如平均边界反射、稀疏 Feige 难度、点对点电路反射）仍是未解决的技术难题；
- 该框架主要针对算术理论与证明系统，可能无法直接推广到更强的算术或不完备系统；
- 由于是信息约束框架，无法避免所有非相对化的证明障碍（如 algebrization 可能仍是问题）。

---

## 124. Offline-to-Online Learning in Linear Bandits

**arXiv ID:** 2606.04305 | [PDF](https://arxiv.org/pdf/2606.04305v1)

**作者:** Kushagra Chandak `[一作]` (University of Alberta), Xiaoqi Tan `[通讯]` (University of Alberta)

**通讯引用:** 1866 | [OpenAlex ID](https://openalex.org/A5047753222)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在有离线数据的线性 bandit 环境中，既利用离线信息又能逐步探索的算法 LinOtO

**💡 创新点**

创新点在于通过预算管理机制，依据低置信界（LCB）积累预算后再选择高置信界（UCB）进行探索，实现对在线最优和离线基准的双重近似；并给出了相应的上界证明

**🔧 技术方法**

使用了最小二乘估计、UCB/LCB 置信区间、预算管理策略以及有效维数（effective dimension）分析

**📊 数据集**

实验数据基于合成的线性 bandit 任务：随机生成 100 个动作（维度 20），随机真参数，离线数据为 100~1000 次采样，噪声为标准正态

**📈 对比分析**

与 warm‑started LinUCB、全探索 LinLCB 和随机策略对比；结果显示在小样本或短期内 LinOtO 跟 LinLCB 接近，随着在线步数或离线样本增多逐渐收敛至 LinUCB 的子线性回报；整体性能介于两者之间

**⚠️ 局限性**

局限性在于假设离线数据为固定设计，且离线误差仅随总样本数而非覆盖度变化；未考虑离线与在线分布偏移，未来可扩展到自适应记录和非线性模型

---

## 125. RL Excursions during Pre-Training: Re-examining Policy Optimization for LLM training

**arXiv ID:** 2606.04272 | [PDF](https://arxiv.org/pdf/2606.04272v1)

**作者:** Rachit Bansal `[一作]` (Harvard University), Sham Kakade `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在从零开始的LLM预训练过程中，作者尝试在不同预训练阶段直接应用强化学习（RL），并与传统的SFT→RL管线及SFT-only做对比，探索RL的时机、效果及对模型能力的影响。

**💡 创新点**

创新点包括：①发现RL在仅预训练4B tokens时即可显著提升推理性能；②通过实验表明预训练数据的针对性比模型规模更能提升RL效果；③提出并验证了“并行平均”梯度更新，将RL与SFT目标结合，实现更优推理表现且不损失通用能力；④系统剖析RL导致的分布“收窄”与“扩展”差异，指出收窄效应主要来自SFT。

**🔧 技术方法**

使用的技术主要是：OLMo2架构、GRPO算法实现的RLVR、SFT（单示例和多示例）训练、并行平均梯度更新、AdamW优化器以及基于预训练混合的token级训练。

**📊 数据集**

主要数据集包括：DOLMino预训练混合（50B tokens，含Wiki、web、math、code等多领域），OpenMathInstruct（含GSM8K和MATH风格问题），以及用于评估的GSM8K和MATH基准数据。

**📈 对比分析**

通过在同一预训练检查点下分别进行Direct‑RL、SFT、SFT‑Gold、SFT→RL和并行平均（Parallel）实验，发现Direct‑RL在早期可与SFT→RL相媲美；并行平均在所有检查点上均达到了最高的P@1，同时保持了非推理基准的性能；SFT往往导致通用能力下降。

**⚠️ 局限性**

局限性包括：预训练混合更偏向数学数据，实验仅覆盖1B/4B模型规模，未探讨更大模型；仅使用GRPO作为RLVR实现，未验证其他RL或RL‑VR变体；评估聚焦于数学推理任务，未全面检验其它下游任务。

---

## 126. Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs

**arXiv ID:** 2606.04360 | [PDF](https://arxiv.org/pdf/2606.04360v1)

**作者:** Xinyu Pang `[一作]` (Tsinghua University), Changshui Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 23215 | [OpenAlex ID](https://openalex.org/A5065063835)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为 Deliberate Evolution（DE）的代理式框架，将 LLM 生成候选表达式与搜索控制分离，从而实现符号回归的样本高效探索。

**💡 创新点**

创新点在于：①引入自适应算子策略（精炼、变异、交叉、重生）提供搜索方向；②利用数据统计、残差分析、维度一致性工具构建诊断报告，为 LLM 提供局部结构反馈；③设计反射记忆机制，汇总历次搜索经验，帮助后续提案避免重复错误并复用成功子结构。

**🔧 技术方法**

核心技术包括：LLM（如 Llama-3.1‑8B‑Instruct、Qwen3‑4B‑Instruct）用于生成符号骨架；BFGS 进行常数拟合；演化算子与策略更新机制；诊断工具 T_data、T_res、T_dim；反射记忆压缩与更新；以及整体演化循环与停滞控制。

**📊 数据集**

使用 LLM‑SRBench（包含 LSR‑Transform 与 LSR‑Synth 共 240 个问题，涵盖物理、化学、生物、材料科学）以及真实测量的 Stress‑Strain 数据集进行实验。

**📈 对比分析**

与 LLMDirect、LLM‑SR、LASR、SGA 等基线对比，DE 在 40% 采样预算下实现平均 NMSE 降低 55%（Llama‑3.1）或 37%（Qwen‑4B），准确率（Acc_0.01）提升至 50% 以上，OOD 性能明显优于基线，且在噪声与分布外测试中表现更稳健。

**⚠️ 局限性**

局限性：依赖 LLM 的生成质量与提示设计；诊断工具与记忆压缩对模型鲁棒性有一定要求；在极大搜索空间或更复杂方程族中，算子策略与记忆更新的调参仍可能面临挑战。

---

## 127. A Geometric View of Counterfactual Behavior: Interaction of Boundary Proximity and Local Support

**arXiv ID:** 2606.04209 | [PDF](https://arxiv.org/pdf/2606.04209v1)

**作者:** Ioanna Gemou `[一作]` (Brown University), Ritambhara Singh `[通讯]` (Brown University)

**通讯引用:** 3557 | [OpenAlex ID](https://openalex.org/A5070578596)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对预训练编码器和线性分类头进行统一的局部搜索探测，评估在不同表征空间和决策边界下的可解释性变化，探究决策边界接近度与局部数据支持度对可逆反事实的影响。

**💡 创新点**

提出将决策边界距离和目标类别的局部支持度联合为几何指标，并证明在保持编码器不变的情况下，仅改变分类头即可显著改变反事实成功率；同时展示几何信息可用于改进反事实搜索。

**🔧 技术方法**

使用一阶线性化近似决策边界距离、kNN距离度量局部支持度、投影梯度搜索、正则化调节、线性SVM对比、几何加权目标等技术，构建统一的评估框架。

**📊 数据集**

Shape（合成）、MNIST、Chest X‑ray、IMDb 以及 MM‑IMDb（多模态）等多种视觉、文本与跨模态数据集。

**📈 对比分析**

通过在准确率相近的模型对之间计算反事实成功率、距离和优化步数，展示即使精度差距仅 0.01%，反事实指标也可出现 50–70% 的差异；在同一编码器上调节分类头正则化，系统性地改变边界位置而保持准确率不变；几何指标的线性回归解释了大部分反事实行为方差，优于单纯的准确率或交叉熵。

**⚠️ 局限性**

仅针对线性分类头和冻结的编码器；局部搜索并非全局最优；几何指标依赖于高质量的邻域估计和梯度近似；未探讨端到端非线性模型、输入空间语义可解释性与实际约束的结合。

---

## 128. Spatially Grounded Concept Bottleneck Models via Part-Factorized Attention

**arXiv ID:** 2606.04364 | [PDF](https://arxiv.org/pdf/2606.04364v1)

**作者:** Dhanesh Ramachandram `[一作]` (Vector Institute), Dhanesh Ramachandram `[通讯]` (Vector Institute)

**通讯引用:** 1649 | [OpenAlex ID](https://openalex.org/A5071728358)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于DINOv3的Part‑Factorized Concept Bottleneck Model (PF‑CBM)，通过前景门、固定概念‑部件映射和可学习的二维高斯空间先验，实现细粒度分类任务中的可解释概念瓶颈，并且无需每张图像的关键点或边框标注；

**💡 创新点**

核心创新在于将概念到部件的路由硬编码为结构化关系，并通过少量（仅0.5%）数据平均关键点初始化的高斯先验打破查询的对称性，从而在无图像级监督的条件下实现概念的空间定位；

**🔧 技术方法**

技术细节包括冻结的DINOv3视觉Transformer、前景门MLP、槽注意力（part cross‑attention）、可学习二维高斯空间先验、固定概念‑部件映射、两阶段训练（先预训练门+关注+头，后联合训练分类器），以及PCA前景估计等；

**📊 数据集**

实验数据集为CUB‑200‑2011鸟类细粒度分类数据集，包含312个属性、15个关键点以及整张图的边框；

**📈 对比分析**

与传统CBM、DOT‑CBM等方法对比，在不使用盒子+关键点监督的条件下，PF‑CBM在CUB上实现了≈88.85% 的top‑1准确率，点位准确率提升至≈70%，仅使用PCA前景+高斯先验即可达到88.6% top‑1，整体性能与有监督模型相当；

**⚠️ 局限性**

局限性包括：模型主要适用于单一目标、姿态规范的场景；空间先验是独立且无交互的，难以处理细长或遮挡部件；若完全去除部件身份仍需图像级空间信号；在多物体或复杂视角下需要进一步改进。

---

## 129. Do Transformers Need Three Projections? Systematic Study of QKV Variants

**arXiv ID:** 2606.04032 | [PDF](https://arxiv.org/pdf/2606.04032v1)

**作者:** Ali Kayyam `[一作]` (BrainChip Inc.), M Anthony Lewis `[通讯]` (BrainChip Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

探讨Transformer中是否必需三个独立投影（Q、K、V），并系统评估三种投影共享方案（Q=K-V、Q-K=V、Q=K=V）以及加上二维位置编码的变体；

**💡 创新点**

发现只共享K与V（Q-K=V）可在保持接近原始模型质量的同时将KV缓存减少50%，并且与头共享技术（GQA/MQA）可叠加压缩，提出了在多尺度和多任务场景下的高效Transformer结构；

**🔧 技术方法**

使用自注意力投影共享技术、二维位置编码、Grouped Query Attention、Multi-Query Attention、FlashAttention等实现高效推理；

**📊 数据集**

在合成推理任务、图像分类（MNIST、FashionMNIST、CIFAR-10/100、TinyImageNet）、异常检测、医学图像分割以及大规模语言建模（300M、1.2B参数训练10B tokens）等多种数据集上进行实验；

**📈 对比分析**

通过对比原始QKV架构，在所有任务中发现Q-K=V在语言建模中仅增加3.1%（300M）/2.48%（1.2B）困惑度，KV缓存缩减50%；在图像和合成任务中性能与QKV相近或略优；与头共享组合后，KV缓存可降至87.5%/96.9%，保持接近原始质量；

**⚠️ 局限性**

局限性包括：最大验证规模为1.2B参数；对更大模型（7B+）的效果尚未验证；仅评估到2048个token的长度，对更长上下文的泛化不明；未对Q=V约束进行实验；理论解释仅基于经验观察。

---

## 130. VAMPS: Visual-Assisted Mathematical Problem Solving Benchmark

**arXiv ID:** 2606.04244 | [PDF](https://arxiv.org/pdf/2606.04244v1)

**作者:** Amirhossein Dabiriaghdam `[一作]` (University of British Columbia), Giuseppe Carenini `[通讯]` (University of British Columbia)

**通讯引用:** 6127 | [OpenAlex ID](https://openalex.org/A5049259877)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估 VAMPS 基准，用于测试多模态大语言模型在使用可视化工具（如 Desmos 绘图）解决数学问题时的能力。

**💡 创新点**

创新点在于：①构造了双语（波斯-英语）真实的大学入学考试题集；②设计了三种解法模式（直接分析、工具生成图表、已提供可视化），以诊断模型在工具调用与视觉感知之间的交接瓶颈；③通过分层可视化和 VLM‑as‑a‑judge 过滤实现细粒度性能评估。

**🔧 技术方法**

技术包括：基于提示的工具调用框架、Desmos 绘图与截图接口、可视化层级生成、JSON 结构化答案输出、VLM 判定器过滤（区分纯符号推理与视觉推理）以及对多种 LLM（Qwen、Gemma、Claude、GPT‑4/5 等）的统一评测。

**📊 数据集**

使用的数据集为 218 条原始考试题（波斯原文 + GPT‑5.4 翻译）及其人类审核后的 436 条 QA，随后通过 Claude/Claude‑Opus、GPT‑5.4 等 LLM 生成 732 条合成多模态 QA，总计 1,168 条。每条 QA 附带四层可视化，覆盖交点、极值、渐近线、单调性等图表特征。

**📈 对比分析**

比较方法：对同一题集分别执行 R1（直接分析）、R2（工具生成图表）和 R3（已提供可视化）三种实验，并计算准确率与经过 VLM‑as‑a‑judge 过滤后的准确率。实验显示：R1 的准确率普遍高于 R2，差距可达 10–20%；在 R3 中，提供可视化后模型准确率提升，但仍低于 R1，表明工具使用与视觉解释仍是瓶颈。

**⚠️ 局限性**

局限性：模型在工具调用语法、图表生成与筛选、视觉特征解读以及 JSON 输出格式方面易出现错误；对视觉信息的依赖不足，仍倾向于符号推理；现有模型缺乏针对图表噪声的鲁棒性，导致工具使用后准确率下降。

---

## 131. TPA-AD: A Two-Stage Pseudo Anomaly-Guided Method for Bearing Time-Series Anomaly Detection

**arXiv ID:** 2606.04073 | [PDF](https://arxiv.org/pdf/2606.04073v1)

**作者:** Xiancheng Wang `[一作]` (Harbin Institute of Technology), Lin Wang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 473773 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种两阶段伪异常引导的轴箱轴承时间序列异常检测方法（TPA-AD），通过先在正常数据上训练重构模型并在误差空间生成可控的伪异常窗口，然后利用对比学习学习异常敏感的表示，最后用KNN在嵌入空间中得到窗口级和点级异常分数。

**💡 创新点**

创新点在于：①基于重构误差的目标误差区间和自适应控制器生成近似异常边界样本，避免了传统随机注入产生的离散、易过拟合噪声；②将伪异常样本与正常样本构成对比学习对，既拉近正常邻域又将伪异常推离，显著提升了正常边界的可分离性；③支持混合连续/离散特征的混合变量场景，兼顾多传感器、多模式下的实际监测需求。

**🔧 技术方法**

使用的技术包括：Transformer+ExtremeKAN的连续特征重构模型；per-feature错误控制的actor-critic迭代调参；对比学习的三元组损失与伪异常/正常硬负样本挖掘；KNN距离用于窗口级异常评分；对离散特征使用独立的KNN并与连续分数融合；以及多种归一化与特征类型判定。

**📊 数据集**

实验数据集涵盖四类轴承故障检测数据（CWRU、HTBF、PHM2009、REALBOX）和两类退化进程数据（XJTU-SY、IMS），另外在论文后续章节扩展到13个公开TSAD基准，验证方法在多种工况、通道数、信噪比和数据规模上的鲁棒性。

**📈 对比分析**

与 Deep SVDD、KNN Distance、LOF Novelty、Isolation Forest、One-class SVM、Adjacent Transformer、Transformer AE、TranAD、CARLA 等基线对比。TPA-AD 在大多数数据集上实现了最高或接近最高的 AUROC、AUPR、best F1、Precision，且在多场景（高噪声、多通道、弱差异故障）下表现更为稳定，尤其在保持低背景误报、连续高分区间方面优于基线。

**⚠️ 局限性**

局限性主要体现在：①伪异常生成仍依赖于重构误差分布，若正常数据多样性不足或重构模型欠拟合，伪异常的边界可能不够代表真实异常；②对极为复杂或非结构化的异常（如多模态或频谱相似的故障）仍难以完全分离；③方法在极高维或长序列上需要更多的计算资源，尤其是对比学习阶段的负样本挖掘；④对实际在线部署的实时性要求仍需进一步优化。

---

## 132. MaskForge: Structure-Aware Adaptive Attacks for Jailbreaking Diffusion Large Language Models

**arXiv ID:** 2606.04027 | [PDF](https://arxiv.org/pdf/2606.04027v1)

**作者:** Yingzi Ma `[一作]` (University of Wisconsin Madison), Chaowei Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对扩散型大语言模型（dLLMs）设计了一种完全黑盒的自适应攻击方法，利用结构模式库（templates）进行红队，能够针对不同攻击目标自动生成、搜索、评估并迭代改进模板。

**💡 创新点**

创新点包括：
- 结构模式抽象：将成功的 jailbreak 转化为可重用的模式 schema，解耦表面文字；
- UCB‑Bandit 搜索：在有限查询预算下平衡利用已知高效模式与探索新模式；
- Scorer‑Guided fallback：当模式库失败时自动生成草稿、提取 mask 并再次查询，保证最终有害文本由目标模型生成；
- 自动化无模板设计：不依赖手工预设模板，而是通过多轮交互自适应学习模式。

**🔧 技术方法**

使用技术包括：
- 结构模式抽象与哈希去重；
- Upper‑Confidence‑Bound（UCB1）多臂赌博机进行模式选择；
- 四类 LLM 角色（攻击者、评分器、草稿生成器、摘要器）协同工作；
- Mask 填充与并行去噪；
- 评估损失与奖励机制。

**📊 数据集**

使用数据集：
- HarmBench（200 目标，7 类伤害）；
- JailbreakBench（100 目标，OpenAI 政策相关）；
- StrongREJECT（313 目标，专门的危害评分器）；
- AdvBench（100 目标，用于迁移评估）。

**📈 对比分析**

与 AR 与 dLLM 传统攻击基线（AIM、PAIR、AutoDAN、PAD、DIJA、DIRECT）以及三种公开 dLLM 防御（Self‑Reminder、PO、A2D）进行对比。实验结果表明：
- 在五个公开 dLLM 上平均攻击成功率（ASR）为 79%/81%/78%，显著高于 DIJA 的 64%/74%/63%；
- 在更强对齐的模型上优势更明显（差距可达 50%）；
- 迁移到 AdvBench 时，冻结的模式库即可达到 88% ASR，且不需要额外更新；
- 相比基线，生成的有害文本质量（HarmfulScore）更高，几乎所有实验单元均获得更高分。

**⚠️ 局限性**

限制：
- 构建与扩展模式库需要多轮与目标模型、评分器等多组件交互，计算成本较高；
- 需要足够多的初始目标样本来启动库；
- 对非常新颖或极端对齐策略的模型仍可能面临未知的防御机制。

---

## 133. Witness-split + window-cardinality refinement for $r_3(N)$: Architecture, empirical results, and a structural hard pocket

**arXiv ID:** 2606.04016 | [PDF](https://arxiv.org/pdf/2606.04016v1)

**作者:** Mehmet Ergezer `[一作]` (Wentworth Institute of Technology), Mehmet Ergezer `[通讯]` (Wentworth Institute of Technology)

**通讯引用:** 1022 | [OpenAlex ID](https://openalex.org/A5087094884)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并执行了一套可复现的计算框架，对 N=212、K=44 的 3-AP-free 子集上界进行穷举搜索，最终未发现可行 44 集，支持 OEIS 预估值 212=43。

**💡 创新点**

创新点在于提出了基于已知下界 witness 的深度拆分+窗口计数剪枝组合、递归细化、以及对“硬口袋”结构的系统性分析与多范式攻击，首次将 CP‑SAT、HiGHS MIP 与 CDCL/SAT 证据融合。

**🔧 技术方法**

使用的技术包括 OR‑Tools CP‑SAT（决策式模型）、HiGHS MIP、CaDiCaL CDCL SAT、OEIS A003002 窗口计数约束、端点强制、反射对称破坏、递归细化、SLURM 并行工程以及 DRAT/LRAT 证明验证。

**📊 数据集**

数据集主要来自 OEIS A003002 的 b‑文件（已知 r_3(n) 的精确值）以及已验证的 43‑元素下界 witness；所有子问题在 1–212 区间上生成。

**📈 对比分析**

对比方法：在 300 秒、1 小时、8 小时等不同时间上跑 CP‑SAT、HiGHS 与 CDCL；在 300 秒层面 45 份残留子问题全部未被求解；HiGHS 在 1 小时内仅关闭 25/45，CDCL 在 1 小时内关闭 18/45，最终仅剩 2 份无解残留，证明了多范式在该“硬口袋”中的无效性。

**⚠️ 局限性**

主要局限在于：未能完全证明上界，仍剩 2 份“硬口袋”子问题对所有测试范式无效；未生成完整深度拆分全部子集的正式证明；证明文件仅覆盖 CDCL 关闭的子问题；求解规模仍受限于当前硬件与算法性能。

---

## 134. SaliMory: Orchestrating Cognitive Memory for Conversational Agents

**arXiv ID:** 2606.04120 | [PDF](https://arxiv.org/pdf/2606.04120v1)

**作者:** Kai Zhang `[一作]` (Meta Reality Labs), Xin Luna Dong `[通讯]` (Meta Reality Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于强化学习的、三层结构化记忆管理框架，能在对话中自动筛选、更新和利用用户记忆，从而实现长期、个性化的对话。

**💡 创新点**

创新点在于：① 将记忆拆分为事实快照、偏好存储和工作记忆三层，模仿人类记忆结构；② 采用分层过程奖励与奖励分解对比学习的RL训练方案，显式解决多阶段记忆管道的信用分配问题；③ 设计了新基准 LoCoMo‑P13n 与多步评估协议，全面衡量记忆召回与个性化效果。

**🔧 技术方法**

核心技术包括：大语言模型 Qwen3.5‑9B‑Instruct 作为策略网络，Qwen3‑235B‑A22B‑Instruct 作为冻结生成器，GPT‑4o 作为判定者；强化学习框架 GRPO；分层过程奖励（R1、R2、R3）；奖励分解对比学习（对记忆写入和记忆利用分别构造对比样本）；三层记忆架构和工作记忆窗口。

**📊 数据集**

使用 LoCoMo 原始数据集扩展的 LoCoMo‑P13n（增加推荐与隐式个性化查询），以及真实 Chat‑AI 流量的内部数据集进行验证。

**📈 对比分析**

与 Infinite Context、RAG‑A‑Mem、MemoryGAS、Mem‑R1 以及 Zero‑shot Agentic 等基线对比。该框架在 LoCoMo‑P13n 上实现 72.9% 的整体准确率，Good Personalization 率 39.8%，在记忆质量上将模糊率降至 19.6%，事实率升至 93.9%。与 Zero‑shot Agentic（70.9% 准确率、6.6% Good Personalization）相比，提升了 1.8% 准确率、33.2% Good Personalization，且推理延迟比零样本基线低约 5×。

**⚠️ 局限性**

局限性包括：① 仍需在强化学习阶段进行耗时训练，难以在资源受限环境快速部署；② 对极端嘈杂或非结构化对话的鲁棒性尚未在更大规模真实流量上彻底验证；③ 记忆结构虽模仿人类，但在处理多模态或跨领域知识时可能需要进一步扩展。

---

## 135. Early Detection of Alzheimer's Disease Using Explainable Machine Learning on Clinical Biomarkers: A Multi-Class Classification Study Using the Alzheimer's Disease Neuroimaging Initiative (ADNI) Dataset

**arXiv ID:** 2606.03995 | [PDF](https://arxiv.org/pdf/2606.03995v1)

**作者:** Afshan Hashmi `[一作]` `[通讯]` (Tuwaiq Academy), Afshan Hashmi (Tuwaiq Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

利用XGBoost加Optuna优化、SMOTE处理不平衡及SHAP解释，对ADNI 8项临床评估进行三类认知状态（正常、轻度认知障碍、阿尔茨海默）检测，并在外部测试集上实现近乎完美的分类性能。

**💡 创新点**

创新点在于仅用常规临床评估指标实现三分类检测且可解释，且在外部验证中宏观AUC、准确率及kappa均超过已有多模态或单一临床模型。

**🔧 技术方法**

技术手段包括XGBoost分类器、Optuna超参数优化、SMOTE生成少数类样本、SHAP TreeExplainer解释、Bootstrap CI评估、五折交叉验证。

**📊 数据集**

数据来源为ADNI 1,641名基线受试者，划分为训练/验证/测试三部分（70%/15%/15%）进行模型训练与评估。

**📈 对比分析**

与已发表的多模态或单临床方法比较，宏观AUC 0.982（95% CI 0.965–0.995）、准确率94.3%、kappa 0.909，明显优于使用影像或基因等额外信息的研究。

**⚠️ 局限性**

局限性包括数据主要为非Hispanic White且受教育水平高，缺乏跨种族验证；仅使用基线数据，未进行纵向预测；未与商业AI工具直接对比；SHAP解释虽可视化但不代表因果关系。

---

## 136. A Goal-Set Characterization of Task Composition in the Boolean Task Algebra

**arXiv ID:** 2606.04053 | [PDF](https://arxiv.org/pdf/2606.04053v1)

**作者:** Eduardo Terrés-Caballero `[一作]` (University of Amsterdam), Herke van Hoof `[通讯]` (University of Amsterdam)

**通讯引用:** 5208 | [OpenAlex ID](https://openalex.org/A5057277609)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对Boolean Task Algebra (BTA) 进行结构分析，证明在确定性MDP中最优扩展Q值函数只需要两个基任务（通用任务和空任务），并基于此提出了一种基于目标集合的组合方法，显著降低了学习和组合成本。

**💡 创新点**

主要创新在于(1) 证明BTA中最优Q值函数空间收缩到仅两个基任务；(2) 通过将任务映射到目标子集，构造BTA的集合同构，从而实现仅用通用与空任务即可完成任意逻辑组合；(3) 讨论了该方法在随机MDP下的指数性限制。

**🔧 技术方法**

使用扩展Q值函数、Boolean代数、目标集合同构、经验实验（tabular、视觉、连续控制）、深度Q学习、TD3、UVFA、Skill Machines 等技术。

**📊 数据集**

实验数据集包括四房间Gridworld（2×2、3×3、4×4）、Boxman物体收集任务、Office Gridworld LTL任务、Safety Gym连续控制任务，以及对应的视觉图像和标签。

**📈 对比分析**

对比了原BTA、基任务方法和新目标集合方法，评价指标为返回值、成功率和组合时间。结果显示新方法在所有环境下与原方法取得相同或更高的最终性能，同时显著降低了学习和组合成本，尤其在组合时间上差距明显扩大。

**⚠️ 局限性**

该方法仅适用于确定性MDP；在随机MDP中，最优策略可能与目标子集呈指数关系，无法用有限数量基任务实现精确组合；此外，理论分析基于两种终端奖励，若终端奖励多样化需进一步研究。

---

## 137. Folded Transport MCMC: Certifiable Quotient Posterior Computation for Symmetric Bayesian Models

**arXiv ID:** 2606.04307 | [PDF](https://arxiv.org/pdf/2606.04307v1)

**作者:** Jun Hu `[一作]` (Wuhan University of Technology), Jun Hu `[通讯]` (Wuhan University of Technology)

**通讯引用:** 473773 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Folded Transport MCMC（FolT-MCMC），通过将后验对称性折叠到基本域上，直接在商空间上进行采样，避免标签切换导致的多模态问题

**💡 创新点**

创新点在于将对称性折叠与可归一化流提议相结合，得到无冗余多模态的商后验，并在此空间上实现LCNF振荡定理的收敛性证书，显著提升量化的谱间隙下界

**🔧 技术方法**

使用光谱正则化的RealNVP可归一化流、独立Metropolis–Hastings、LCNF振荡定理和量化核心证书（quantile-core certificate）

**📊 数据集**

在合成高斯混合模型（维度2–20、标签切换模式数2–24）以及真实泰山台风时的结构模态加速度数据和标准三分量高斯混合后验上进行实验

**📈 对比分析**

与未折叠的IMH、随机置换采样器（RPS）和后验排序方法比较，FolT-MCMC在商域上给出QC γ提升至2×–145×，并得到非空的证书；接受率和ESS与其他方法相近或更优，但采样速度并无显著提升

**⚠️ 局限性**

仅适用于完全满足有限群对称性的目标，需要已知对称群和基本域；商后验的核心证书并不保证全域谱间隙；当对称性近似或群规模大时，折叠与算子求和成本会显著增加

---

## 138. Instant-Fold: In-Context Imitation Learning for Deformable Object Manipulation

**arXiv ID:** 2606.04269 | [PDF](https://arxiv.org/pdf/2606.04269v1)

**作者:** Yilong Wang `[一作]` (Imperial College London), Edward Johns `[通讯]` (Imperial College London)

**通讯引用:** 5532 | [OpenAlex ID](https://openalex.org/A5010778183)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 Instant-Fold 框架，能够在仅收到单一演示的条件下，以无梯度更新的方式完成变形物体（如衣物）的多模态折叠任务，形成长时序闭环控制。

**💡 创新点**

创新点在于将时间对比预训练与流匹配 Transformer 相结合，首次把单一演示视为任务上下文，实现了变形感知视觉表示与基于上下文的即时模仿学习（ICIL）于变形物体操控领域。

**🔧 技术方法**

技术包括 Temporal Contrastive Pretraining、geo-semantic token 化、3D 相对注意力与 ALiBI、AdaLN 调度、流匹配去噪 Transformer、LoRA 微调、FleX 物理仿真、RGB‑D 采样与 Farthest Point Sampling 等。

**📊 数据集**

数据集主要由 360 Cloth3D 服装网格生成的 4 万+ 轨迹演示（8 种折叠模式、12 条轨迹/网格）以及 4 万条预训练轨迹组成，配合仿真平台 FleX 与 Isaac Lab 以及实际 RGB‑D 摄像头采集的真实衣物演示。

**📈 对比分析**

通过与 ClothFunnels、UniFolding、UniGarmentManip 等基线在 FleX、Isaac Lab 以及八件未见实景衣物上的零样本转移实验对比，Instant‑Fold 在仿真中成功率高达 99.7%，实景成功率 60.9%，并在几何误差和 Wasserstein 距离等指标上显著优于对照方法。

**⚠️ 局限性**

局限性包括：依赖可靠目标分割；仅从可折叠状态开始，无法处理杂乱初始姿态；仅研究衣物上衣类；上下文库为手工设计、规模有限；缺乏自动化上下文探索与完全未知变形物体任务的泛化。

---

## 139. Analyzing the Evolution of Structural Communities within Microservice Architecture

**arXiv ID:** 2606.04047 | [PDF](https://arxiv.org/pdf/2606.04047v1)

**作者:** Alexander Bakhtin `[一作]` (University of Oulu), Davide Taibi `[通讯]` (University of Southern Denmark)

**通讯引用:** 4559 | [OpenAlex ID](https://openalex.org/A5086929289)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对 train-ticket 开源微服务基准项目的六个发布版本进行时序社区检测，分析社区结构随时间的演化，并探究社区与业务流程的对应关系。

**💡 创新点**

首次将时序社区检测（PARAFAC/Tensor）方法应用于重建的微服务架构网络，用核心一致性评估确定最佳社区数，并通过成员强度阈值识别服务是否属于多社区，从而揭示潜在的反模式。

**🔧 技术方法**

使用 Gauvin 等人的时序社区检测方法、核心一致性（core consistency）指标、3σ统计检验、成员强度归一化以及阈值 0.5 的社区归属判定。

**📊 数据集**

train-ticket OSS 微服务基准项目的六个发布版本（v0.0.1–v0.2.0 与 v1.0.0），通过 Code2DFD 工具重构得到的微服务依赖时序网络。

**📈 对比分析**

对 2–6 个社区数进行 20 次随机初始化，计算核心一致性均值/标准差，选取一致性>90 的两社区方案；结果表明两社区活动高度稳定（标准差极小，均在 3σ 范围内），验证了架构的稳定性；未与其他算法做直接性能对比，仅以核心一致性为选择依据。

**⚠️ 局限性**

仅在单一 OSS 项目（42 个服务）上实验，社区数受限；使用单一时序社区算法，阈值 0.5 主观；业务流程映射依据作者解释，存在主观偏差；对工业规模系统的泛化能力有限。

---

## 140. Polymarket-v1 Database

**arXiv ID:** 2606.04217 | [PDF](https://arxiv.org/pdf/2606.04217v1)

**作者:** Boka Qin `[一作]` (Washington University in St. Louis), Rui Yang `[通讯]` (Southwest University of Political Science and Law)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

我们发布了Polymarket‑v1数据库，记录了2022‑2026年Polymarket第一代预测市场的完整1.2 亿笔交易，并提供了区块链校验的真实买卖方向；

**💡 创新点**

创新点在于提供了第一批可验证交易方向的数据集，并用它揭示了传统微结构分类方法（tick规则、体积分类）的系统性错误以及这些错误对流动性与信息度量的影响；

**🔧 技术方法**

使用了经典微结构工具（tick规则、批量体积分类）、VPIN与OFI的计算、Gibbs采样扩展价差估计、VAR与SVAR价格冲击分析以及跨市场OLS回归等技术；

**📊 数据集**

数据集为Polymarket‑v1的完整交易记录，覆盖41个月、1.3 百万市场和61 亿美元的名义交易量；

**📈 对比分析**

与传统基于报价的分类方法相比，基于真实方向的衡量实现了高精度的分类（近随机整体准确率但存在价格梯度偏差），并能更准确地估计VPIN、OFI等指标；在预测准确度回归中，True VPIN正相关于Brier分数、Gibbs价差负相关，揭示了市场质量与预测性能的真实关系；

**⚠️ 局限性**

局限包括仅使用链上结算层、缺乏订单簿快照、仅涵盖Polymarket v1、部分市场缺失最终结果、未覆盖所有市场类型、洗盘检测方法未验证精度等问题。

---

## 141. Covert Influence Between Language Models

**arXiv ID:** 2606.04071 | [PDF](https://arxiv.org/pdf/2606.04071v1)

**作者:** Avidan Shah `[一作]` (MATS), Shi Feng `[通讯]` (MATS)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过统一框架，研究了在监督微调(SFT)、策略蒸馏(OPD)和上下文学习(ICL)三种接口中，语言模型如何在不被人类察觉的情况下通过隐式载体（自然语言或数字序列）传递隐藏的行为目标（payload）并影响接收模型的行为。

**💡 创新点**

创新点包括：①首次在三种接口中对隐式影响进行系统评估；②提出并应用点级归因评分MDCL，能够精确识别高影响载体；③证明自然语言载体与数字载体的隐蔽机制不同，前者更易被人类识别但跨模型迁移更稳健；④展示了“隐秘激励”(stealth priming)和“突现失配”(emergent misalignment)两种新的隐式影响方式。

**🔧 技术方法**

核心技术为：点级归因评分MDCL（基于发送者在不同条件下的对数概率差）；人脸向量投影PVP；影响加权采样；SFT、OPD、ICL训练流程；LLM-as-a-judge过滤；统计显著性检验与性能评估。

**📊 数据集**

使用的主要数据集与样本包括：人工生成的数字序列（用于隐形学习）、自然语言指令和完成（Qwen 2.5 7B、Gemma 4 31B、OLMo、Llama等模型输出）、100个中立系统提示、100个二选一场景、50个GPT‑4o生成的安全风险评估问题等。

**📈 对比分析**

与随机基线、PVP选择以及无载体对照进行比较。结果显示：在SFT中，MDCL挑选的前10k样本显著提升隐形学习效果；在OPD中，MDCL或PVP顶层提示可将语言切换率从约44%提升至83%；在ICL中，MDCL或对数差分法将被偏向选项的采样率从3%提升至58%。总体而言，MDCL显著优于PVP，尤其在数字载体场景。

**⚠️ 局限性**

局限性包括：实验结果为上限假设，当前模型缺乏完整的自省能力，无法在单前向传递中自动执行MDCL/PVP选择；跨模型数字载体的隐形学习在部分配置下未能复现；PVP在数字载体上无显著效果；掩码策略可能不适用于所有payload；实验主要针对离散词汇payload，可能不适用于连续或多模态任务。

---

## 142. End-to-End Text Line Detection and Ordering

**arXiv ID:** 2606.04166 | [PDF](https://arxiv.org/pdf/2606.04166v1)

**作者:** Benjamin Kiessling `[一作]` `[通讯]` (Inria), Benjamin Kiessling (Inria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Orli，一种端到端的历史文档布局分析模型，能够同时检测文本行并按阅读顺序输出基线。

**💡 创新点**

创新点包括：① 将行检测与阅读顺序建模为自回归序列生成任务；② 采用 chord‑frame 基线表示并配合迭代细化与局部视觉修正；③ 通过锚点初始化实现稳健的几何回归。

**🔧 技术方法**

技术细节：使用 ConvNeXtV2 小模型作为视觉编码器，RT‑DETR 风格的多尺度融合；LLaMA‑style Transformer 作为自回归解码器；曲线回归头采用四步迭代与局部视觉细化；训练使用 focal 损失、多任务联合优化，混合 bf16 精度，AdamW 优化；推理采用贪婪解码。

**📊 数据集**

训练数据包含 196,691 页异构语料（约 25% 合成 arXiv 文章、其余 147,596 页自然手写/印刷文档，覆盖拉丁、希伯来、斯拉夫、阿拉伯等 10 种书写系统）。评估数据集包括 cBAD 2019、OHG、FCR、ABP 四个历史文档基准。

**📈 对比分析**

与现有基线（cBAD 参与者、ParseNet）和基于规则的 TBLR/FDTD 方法比较，Orli 在高分辨率无细化模型下 cBAD 2019 F1 0.9340，OHG/FCR 匹配覆盖率近 100% 并获得极低的 Spearman footrule/Kendall τ；ABP 需要少量细化后阅读顺序指标显著提升。

**⚠️ 局限性**

局限性：对极密集表格布局（如 ABP）在零样本下仍表现不佳；需要针对特定编辑顺序进行微调；高分辨率模型计算成本较高；局部视觉修正虽提升精度但会增加推理时间。

---

## 143. Unlocking Feature Learning in Gated Delta Networks at Scale

**arXiv ID:** 2606.04048 | [PDF](https://arxiv.org/pdf/2606.04048v1)

**作者:** Yifeng Liu `[一作]` (University of California Los Angeles), Quanquan Gu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并推导了 Gated Delta Network 的最大更新参数化（μP），并验证其在 AdamW 与 SGD 下实现零样本学习率迁移。

**💡 创新点**

创新点在于发现门控权重和标量门控参数在 SGD 下需要不同的学习率缩放，并给出了完整的 μP 参数化方案。

**🔧 技术方法**

采用 μP 理论、Tensor Programs 框架、线性 Transformer 结构、短卷积、门控机制、RMSNorm 等技术。

**📊 数据集**

使用 FineWeb‑Edu 100B 规模文本数据进行语言模型预训练。

**📈 对比分析**

通过与标准参数化（SP）和原始 μP 进行对比，展示不同宽度模型下学习率可零样本迁移，验证损失显著优于 SP。

**⚠️ 局限性**

局限性：推导基于短记忆假设，未充分考虑长上下文中的 BPTT 累计效应；实验仅在单个 H100 GPU 上进行，规模有限。

---

## 144. Weakly Supervised Incremental Segmentation via Semantic Anchors and Spatial Arbitration

**arXiv ID:** 2606.04060 | [PDF](https://arxiv.org/pdf/2606.04060v1)

**作者:** Zhonggai Wang `[一作]` (Beijing Institute Of Technology), Guangyu Gao `[通讯]` (Beijing Institute Of Technology)

**通讯引用:** 1844 | [OpenAlex ID](https://openalex.org/A5062899493)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究弱监督增量语义分割，提出SASA框架以抑制特征漂移并提高分割质量。

**💡 创新点**

创新点在于引入语义锚点与弹性残差调优实现类级表示稳定，并设计空间标签仲裁机制通过几何一致性去除噪声。

**🔧 技术方法**

使用ViT backbone、CAM生成伪标签、语义锚点学习、弹性残差Token、空间标签仲裁、以及多项损失约束（分离、蒸馏、正则）。

**📊 数据集**

在Pascal VOC 2012和MS COCO数据集上进行评估。

**📈 对比分析**

与多种SOTA方法在10-10、15-5、COCO-to-VOC、10-2等多步增量设置下进行对比，SASA在VOC 10-10/15-5提升mIoU超过5%，COCO-to-VOC提升7+点，6步增量老类mIoU提升22%。

**⚠️ 局限性**

局限在于依赖外部生成的对象掩码、对极少监督或极小目标的噪声仍有一定影响。

---

## 145. Multi-Granularity 3D Kidney Lesion Characterization from CT Volumes

**arXiv ID:** 2606.04365 | [PDF](https://arxiv.org/pdf/2606.04365v1)

**作者:** Renjie Liang `[一作]` (University of Florida), Jie Xu `[通讯]` (University of Florida)

**通讯引用:** 21793 | [OpenAlex ID](https://openalex.org/A5008162410)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在本研究中，作者构建了一个基于3D肾脏CT的多粒度病变特征预测框架，将任务重新定义为每侧肾可变数病变的集合预测，并实现了LesionDETR模型；

**💡 创新点**

创新点主要在于：①将lesion‑centric set‑prediction引入3D肾脏CT分析；②提出了基于大小距离的Hungarian匹配的DETR‑style预测头；③通过层次化监督将细粒度L3预测聚合为侧级L1/L2；④构建了包含分割掩膜和多粒度标签的统一数据集；

**🔧 技术方法**

技术方案包括3D Transformer（DETR‑style）集合预测头、SwinUNETR编码器、同域预训练（SuPreM）、分割掩膜作为额外通道、层次化损失和大小距离Hungarian匹配；

**📊 数据集**

使用的数据集为UF Health 788名患者的2,619份肾脏CT与报告（手工验证的标签），以及公开的KiTS23 489例作为外部零射验证集；

**📈 对比分析**

通过大规模对照实验比较不同输入表示、编码器初始化、监督粒度等，评估侧级AUC、尺寸MAE、病变级AP；在UF内验证中，LesionDETR侧级AUC达到0.799±0.009，计数条件模型mAP为0.190±0.036；在KiTS23零射验证中，AUC可提升至0.817，显示良好的跨域泛化；

**⚠️ 局限性**

主要限制包括：数据规模不足，尤其固体病变稀缺导致AP接近噪声底；单中心数据与外部验证仅推理，未覆盖多中心差异；模型整体性能仍未达到临床部署标准；分割掩膜噪声及特征提取瓶颈限制进一步提升。

---

## 146. From Ticks to Flows: Dynamics of Neural Reinforcement Learning in Continuous Environments

**arXiv ID:** 2606.04275 | [PDF](https://arxiv.org/pdf/2606.04275v1)

**作者:** Saket Tiwari `[一作]` (Brown University), George Konidaris `[通讯]` (Brown University)

**通讯引用:** 5562 | [OpenAlex ID](https://openalex.org/A5078124517)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了连续时间强化学习的理论框架，给出了基于过参数化单隐藏层神经网络的actor‑critic算法在梯度时间上的状态分布演化方程，并在toy LQR 环境中进行验证。

**💡 创新点**

创新点包括：① 采用双时钟（环境时间与梯度时间）模型；② 在无限宽度下推导状态分布的连续 SDE；③ 将梯度更新表述为五个变量的闭式系统；④ 通过 Itô‑Taylor 展开与线性化实现非参数化分析。

**🔧 技术方法**

技术方法主要包括连续时间 RL、随机微分方程、随机控制理论、无限宽度神经网络线性化、Itô‑Taylor 展开、马尔可夫中心极限定理与大数定律等。

**📊 数据集**

实验数据集主要为 toy LQR 连续控制任务（1、2、8、32 维），并在 MuJoCo Cheetah 等标准控制环境中做过验证。

**📈 对比分析**

与传统加噪探索、基线 actor‑critic 及理论模拟进行对比，实验显示在所有维度下均能收敛至接近最优策略，理论曲线与数值仿真高度吻合，证明模型有效。

**⚠️ 局限性**

局限性：仅适用于平滑动力学、单隐藏层、无限宽度“懒惰”模式；不涵盖深层网络、有限宽度、非光滑激活、部分可观测或更复杂的控制任务；高维推广需进一步研究。

---

## 147. Behavioral and Performance Indicators of Depression and Anxiety in Electronic Learning Systems

**arXiv ID:** 2606.04254 | [PDF](https://arxiv.org/pdf/2606.04254v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 148. Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation

**arXiv ID:** 2606.04046 | [PDF](https://arxiv.org/pdf/2606.04046v1)

**作者:** Boyuan Xiao `[一作]` (Zhejiang University), Kun Zhou `[通讯]` (Zhejiang University)

**通讯引用:** 41960 | [OpenAlex ID](https://openalex.org/A5100722039)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SceneDiver，利用粗到细的场景图推理为视觉语言模型(VLM)生成聚焦计划，从而显著降低视觉幻觉并提升嵌入式决策任务的鲁棒性；同时设计轻量级适配器将此聚焦能力迁移至视觉语言动作模型(VLA)。

**💡 创新点**

创新点在于：①将结构化场景图作为先验引导 VLM 进行高层次图推理，再通过局部验证与探索实现逐步聚焦；②构建基于 Slot Attention 的实时适配器，在保持高效的同时保留 VLM 的精细聚焦能力；③在多种基准上验证两阶段推理对消除幻觉、提升成功率的显著效果。

**🔧 技术方法**

使用 OvSGTR 场景图生成、文本化场景图输入 VLM、图推理 + 细化探索、焦点映射融合、Slot Attention 与 Mask 预测、Hungarian 匹配等技术；在实验中结合 Qwen、Gemini、GPT‑4o-mini 等大语言模型。

**📊 数据集**

数据集包括：机器人操作基准（MuJoCo 环境）、房间导航基准（含复杂环境与视觉外观子任务）、LIBERO‑Plus 任务套件（空间、对象、目标、长时序）以及公开场景图资源（Visual Genome 等）。

**📈 对比分析**

与传统直接聚焦、SoM、Multi‑Res、VCD 等方法对比，SceneDiver 在机器人操作成功率提升约 10–15%、房间导航成功率提升 16% 以上；在 LIBERO‑Plus 上成功率平均提升 9–10% 并保持计算开销仅 +2.6%。

**⚠️ 局限性**

局限包括：对快速动态场景的场景图生成仍有性能瓶颈；在极端噪声或完全错误的场景图下，仍需进一步提高鲁棒性；以及对更复杂多阶段操纵任务的层次规划尚未充分探索。

---

## 149. CodegenBench: Can LLMs Write Efficient Code Across Architectures?

**arXiv ID:** 2606.04023 | [PDF](https://arxiv.org/pdf/2606.04023v1)

**作者:** Jie Li `[一作]` (Sun Yat-sen University), Haohuan Fu `[通讯]` (National Supercomputing Center)

**通讯引用:** 11466 | [OpenAlex ID](https://openalex.org/A5031545295)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了CodegenBench基准套件，用于评估大型语言模型在x86_64、Kunpeng和Sunway三种CPU架构上的高性能代码生成能力。

**💡 创新点**

创新点在于构建了多架构、可扩展的自动化评测框架，并将传统BLAS库与针对超级计算机专用架构的LeetSunway/LeetKunpeng核组相结合，填补了现有GPU/CUDA基准之外的空白。

**🔧 技术方法**

利用大型语言模型（如Claude、Qwen、DeepSeek等）配合自动化的提示生成、编译、运行与验证流水线，以及Pass@k和Fast_1@1等评估指标。

**📊 数据集**

使用了106条BLAS子例程、20条Sunway核组计算任务以及同等20条Kunpeng核组计算任务，数据集已公开托管在指定链接。

**📈 对比分析**

通过Pass@1/Pass@5/Fast_1@1指标在三平台上对9个LLM进行横向对比，结果显示在x86上模型性能较好，Kunpeng和Sunway上表现显著衰退；封闭源Claude系列在所有指标上领先。

**⚠️ 局限性**

主要局限在于对专用架构的可泛化能力不足，缺乏公开训练样本导致生成代码正确率低；模型规模增大并不必然提升优化效果；代码长度与正确率呈负相关。

---

## 150. A Cookbook of 3D Vision: Data, Learning Paradigms, and Application

**arXiv ID:** 2606.04291 | [PDF](https://arxiv.org/pdf/2606.04291v1)

**作者:** Hongyang Du `[一作]` (Brown University), Tao Hu `[通讯]` (Stability AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并统一了3D视觉领域的表示形式、数据集与学习范式，构建了一个以数据为中心的整体框架；

**💡 创新点**

创新点在于提出了跨模态、跨任务的统一视角，绘制了从点云、网格到隐式场、3D高斯等多种数据结构的关联图谱，并梳理了数据集生态对模型演进的影响；

**🔧 技术方法**

主要技术为文献调研、系统化分类与表格汇总，结合时间轴与统计图展示数据集发布与模态分布变化；

**📊 数据集**

引用并总结了约50个代表性数据集，包括ScanNet++、DL3DV-10K、WildRGB-D、PointOdyssey、InteriorGS、MegaSynth等多模态、多场景的数据来源；

**📈 对比分析**

通过对比分析各数据结构与数据集的效率、精度与适用场景，对现有方法进行了系统性评述，但未给出定量实验性能；

**⚠️ 局限性**

局限性在于仍缺乏统一的跨模态评测基准，数据集碎片化导致公平对比受限，且对实际推理速度与实时性讨论不足。

---

## 151. HYolo: An Intelligent IoT-Based Object Detection System Using Hypergraph Learning

**arXiv ID:** 2606.04345 | [PDF](https://arxiv.org/pdf/2606.04345v1)

**作者:** Isha Abid `[一作]` (National University of Sciences and Technology), Muhammad Khuram Shahzad `[通讯]` (National University of Sciences and Technology)

**通讯引用:** 3049 | [OpenAlex ID](https://openalex.org/A5035787368)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HYolo 模型，将超图学习融入 YOLO，以提升 IoT 环境下的目标检测性能。

**💡 创新点**

设计 HyperC2Net 进行跨层高阶特征融合，采用超图卷积 HyperConv 构建多节点关系，并用距离阈值构造超图。

**🔧 技术方法**

使用 YOLOv8-N 作为骨干网络，结合超图神经网络、HyperC2Net、HyperConv 以及距离阈值超图构造等技术。

**📊 数据集**

使用 COCO 数据集进行实验。

**📈 对比分析**

与 YOLOv8-N 对比，mAP@50 提升约12%，mAP@0.5:0.95 提升约23.5%，box loss 降低9.4%，F1 分数提升16.1%，训练收敛更快。

**⚠️ 局限性**

计算复杂度增加，对超图构造参数敏感，依赖大量训练数据，实时性能略受影响，未在真实 IoT 环境中充分验证。

---

## 152. When Retrieval Doesn't Help: A Large-Scale Study of Biomedical RAG

**arXiv ID:** 2606.04127 | [PDF](https://arxiv.org/pdf/2606.04127v1)

**作者:** Erfan Nourbakhsh `[一作]` (University of Texas at San Antonio), Anthony Rios `[通讯]` (University of Texas at San Antonio)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

对5个开源指令调优模型在10个医学QA数据集上进行检索增强生成（RAG）大规模评估，覆盖4种检索方法和4种检索语料库。

**💡 创新点**

首次系统化比较模型规模、检索方法和语料库对医学QA性能的影响，揭示检索增益极小且受模型规模主导。

**🔧 技术方法**

使用BM25、TF‑IDF、MedCPT及RRF混合检索，结合LLaMA、Qwen、Mistral等指令调优LLM，采用ROUGE‑L、准确率等指标评估。

**📊 数据集**

10个医学QA数据集（MeQSum、MedRedQA、MedicationQA、MASH‑QA、ChatDoctor、BioASQ、MedQuAD、MedQA‑USMLE、MedMCQA、MMLU Medical）与4个检索语料库（PubMed、医学教材、Yahoo Answers、HealthCareMagic）。

**📈 对比分析**

与无检索基线对比，检索提升仅1–2分，模型规模差距更大；在多项指标下，检索效果不稳定，模型更关注自身参数知识。

**⚠️ 局限性**

仅使用参考基准指标，未直接测量证据依赖；仅评估5个开源模型，未覆盖闭源强大模型；检索策略固定，未尝试迭代或重排序；未深入分析不同问题类型的检索收益。

---

## 153. Functional Interface Blocks for Neuromorphic Hardware: A Junction-Centered Framework

**arXiv ID:** 2606.04281 | [PDF](https://arxiv.org/pdf/2606.04281v1)

**作者:** Wellington Avelino `[一作]` (University of Minas Gerais), Gilberto Medeiros-Ribeiro `[通讯]` (University of Minas Gerais)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于第二代电流搬运器（CCII）的功能接口块（FIB）框架，用于在异构神经形态硬件中实现电路级的功能耦合和电气解耦，随后在一个包含忆阻突触与UJT后神经元的帕夫洛夫条件反射实验平台上进行了实验验证。

**💡 创新点**

创新点在于将接口设计视为中心化的功能抽象（FIB），通过门限电压/电流控制、阻抗匹配和时域多路复用，实现了不同设备（忆阻器、NDR神经元）的电气兼容性，并提出了一套基于CCII的统一实现与规范化的接口原语分类。

**🔧 技术方法**

核心技术包括：（1）第二代电流搬运器的三端实现（VCVS+CCCS组合）；（2）基于FIB的交叉架与后神经元之间的时分复用接口；（3）用于学习的STDP兼容双极写回脉冲驱动；（4）Arduino Mega 2560的实验序列控制。

**📊 数据集**

未使用公开数据集，实验采用自构造的物理平台：忆阻突触、UJT神经元、CCII接口；通过示波器记录Food、Bell、Salivation以及膜电位波形。

**📈 对比分析**

通过四阶段帕夫洛夫实验（预条件、未条件、配对、后条件）对学习过程进行定量评估；实验结果显示忆阻器电阻随训练周期下降，突触可从无响应转为响应；进一步的“遗忘”阶段验证了可逆写回控制。性能方面，系统实现了毫秒级事件同步、可编程写回波形与可靠突触状态保持，但在规模化和功耗方面尚未量化。

**⚠️ 局限性**

主要限制包括：CCII实现的有限操作范围与非理想特性（阻抗不匹配、功率损耗）；实验规模受限于单一突触-神经元对，未展示大规模网络的可扩展性；缺乏对不同忆阻器/神经元技术的泛化评估；以及未提供正式的性能基准或与其他接口方案的对比。

---

## 154. StandardE2E: A Unified Framework for End-to-End Autonomous Driving Datasets

**arXiv ID:** 2606.04271 | [PDF](https://arxiv.org/pdf/2606.04271v1)

**作者:** Stepan Konev `[一作]` `[通讯]`, Stepan Konev

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了StandardE2E统一框架，实现多数据集的预处理、缓存与PyTorch加载，支持跨数据集训练。

**💡 创新点**

仅需实现Raw→StandardFrameData映射即可添加新数据集，提供可配置适配器链和段上下文聚合，极大降低复用成本。

**🔧 技术方法**

使用Pydantic定义统一schema，PyTorch Dataset、Parquet索引、Numpy NPZ缓存、YAML配置和多进程并行。

**📊 数据集**

支持Waymo End-to-End、Waymo Perception、Argoverse 2 Sensor、Argoverse 2 LiDAR、NAVSIM与WayveScenes101六个数据集。

**📈 对比分析**

论文为扩展摘要，未给出实验数值，后续将进行跨数据集性能评估。

**⚠️ 局限性**

局限在缺乏具体性能对比，且对极少见模态或自定义标注的支持仍待完善。

---

## 155. Stein Kernelized Molecular Dynamics for Active Learning of Interatomic Potentials

**arXiv ID:** 2606.04100 | [PDF](https://arxiv.org/pdf/2606.04100v1)

**作者:** Joanna Zou `[一作]` (Massachusetts Institute of Technology), Youssef Marzouk `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 6027 | [OpenAlex ID](https://openalex.org/A5071702167)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现一种基于Stein kernelized molecular dynamics（SKMD）的在线主动学习框架，用于高效采样和收集机器学习原子势能模型（MLIP）的训练数据。

**💡 创新点**

创新点在于：①将SVGD变体引入分子动力学，实现自适应偏置力同时保持玻尔兹曼分布；②提出基于SKMD偏置力范数的在线停止准则；③通过异步粒子更新和全局原子描述子核，兼顾探索与高概率区域。

**🔧 技术方法**

使用Stein variational gradient descent、异步粒子更新、全局原子描述子核、带有自适应停止准则的SKMD采样、以及基于核的主动学习准则。

**📊 数据集**

在二维Müller–Brown势能表面（测试网络潜能）和氨基酸二肽（Alanine dipeptide）系统上使用的MACE模型，结合SPICE数据集作为基准。

**📈 对比分析**

与传统过阻尼Langevin、UDD和无偏MD等采样方法比较，SKMD在相同样本数下的RMSE（能量与力）下降更快、方差更小，展示了更高的数据利用效率。

**⚠️ 局限性**

局限在于自适应停止准则要求固定核宽度；若采用可变核宽度，偏置力范数不显著下降，需分两阶段（探索+利用）或改进核选择。

---

## 156. Disentangling Answer Engine Optimization from Platform Growth: A Log-Based Natural Experiment on ChatGPT Referral Traffic

**arXiv ID:** 2606.04362 | [PDF](https://arxiv.org/pdf/2606.04362v1)

**作者:** Keisuke Watanabe `[一作]` (Glasp Inc.), Kazuki Nakayashiki `[通讯]` (Glasp Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在一个高流量域名上对YouTube问答页面实施了一套 AEO（答案引擎优化）干预，并通过未受干预的同域控制组来净化平台尾风的影响，评估干预的因果效果。

**💡 创新点**

创新点在于：①使用同域对照和第一方服务器日志/分析数据，避免传统第三方估算的噪声；②将 ITS（中断时间序列）与时间置换检验相结合，提供保守的因果效应估计；③验证了 SEO 保护规则，证明 AEO 并未显著损害有机搜索表现。

**🔧 技术方法**

主要技术包括：URL 正则化、404 日志挖掘生成新页面、标题和摘要重写、SEO 保护规则、HAC (Newey–West) 标准误、移动块自助法、时间置换检验、Google Analytics 4 与 Search Console 数据集成。

**📊 数据集**

数据集为：同域内数十万 YouTube 问答页面的 ChatGPT 推荐流量（由 GA4 记录）、用户会话（包括参与度过滤）、以及 Google Search Console 的点击/展示数据；全部以相对增长倍数呈现，保留匿名性。

**📈 对比分析**

通过与未干预的同域页面比较，得到平台尾风下的对照增长倍数为 3.5×，干预后受访页面增长 6.1×，差分为约 1.8–2.3×；ITS 分析给出水平跳变 1.82×（95% CI 1.31–2.54），但置换检验 p=0.16，提示效果具有提示性但未达到传统显著性阈值。

**⚠️ 局限性**

限制包括：仅单域单引擎、未随机化干预、前期趋势显著、干预为组合策略且无法拆分单一技术效果、测量时间点模糊、可能存在 SUTVA 泄漏、以及对 Bot 过滤政策变化的影响。

---

## 157. Proof-Carrying Agent Actions: Model-Agnostic Runtime Governance for Heterogeneous Agent Systems

**arXiv ID:** 2606.04104 | [PDF](https://arxiv.org/pdf/2606.04104v1)

**作者:** Zexun Wang `[一作]` `[通讯]` (Ond Holdings Inc), Zexun Wang (Ond Holdings Inc)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Proof-Carrying Agent Actions（PCAA）框架，将行动证书作为跨运行时统一的治理核心，采用五检查点流程实现可审计、可回放的行动管理。

**💡 创新点**

创新点在于将行动证书与外部边界上下文、审批可执行性类别相结合，并通过可移植的行动封装和多层完整性投射实现跨运行时的统一治理和审计链。

**🔧 技术方法**

使用了可移植行动封装、五检查点契约、外部性上下文、审批可执行性分类、层级完整性投射、运行时治理契约、证明包与检索接收器等技术。

**📊 数据集**

使用了由 24 种可执行种子扩展至 96 条跟踪的受保护基准，覆盖四类运行时（框架 SDK、OpenAI 兼容网关、托管代理平台、观察者/导入模式）。

**📈 对比分析**

通过与静态规则和标量启发式基线比较，PCAA 在路由准确率、宏 F1、严重召回、块精度等指标均达到 1.0；审查负载约 29% 进入审查，20% 采用模拟，25% 块；完整性通道保持 100% 稳定，检索完整度 0.516；消融实验显示外部性、审批可执行性和完整性通道是关键。

**⚠️ 局限性**

局限性包括运行时深度异质、完整性投射仍待更强的可验证网络、基准公开有限、某些实验结果受实现细节影响，并且在未来所有运行时下无法保证同等性能。

---

## 158. Dual Advantage Fields

**arXiv ID:** 2606.04188 | [PDF](https://arxiv.org/pdf/2606.04188v1)

**作者:** Alexey Zemtsov `[一作]` (NUST MISIS), Arip Asadulaev `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种 Dual Advantage Fields (DAF) 方法，利用双目标表示的梯度信息，将离线数据中的动作特征位移与目标方向对齐，从而直接提取局部优势信号并得到目标条件策略。

**💡 创新点**

创新点在于发现双目标嵌入即为双线性价值函数在状态表示空间中的梯度，并通过预测动作导致的特征位移与此梯度对齐来得到局部优势，避免单独训练 Q 函数即可进行策略提取。

**🔧 技术方法**

采用双线性目标价值模型、动作效果预测网络、优势加权回归、期望回归及其损失函数等离线强化学习技术，并利用梯度与向量内积生成局部优势。

**📊 数据集**

在 OGBench benchmark 的离线数据集上进行实验，涵盖迷宫式步行（四足/类人机器人）、抓取拼装、噪声演示、拼图类任务等多种环境。

**📈 对比分析**

与 HIQL、OTA、MQE、CRL、GCIQL、GCIVL 等代表性方法对比，DAF 在所有任务上均实现了更高的 Median、IQM、Mean 与 Optimality Gap，尤其在操纵和拼图任务中显著提升。

**⚠️ 局限性**

局限性在于对离线数据覆盖度与动作效果模型精度高度依赖；在覆盖不足或高度不确定的区域可能导致优势估计不准；目前未针对图像输入或高度随机的目标到达场景进行验证。

---

## 159. Multi-Agent Next-Best-View Optimization for Risk-Averse Planning

**arXiv ID:** 2606.04158 | [PDF](https://arxiv.org/pdf/2606.04158v1)

**作者:** Amirhossein Mollaei Khass `[一作]` (Lehigh University), Nader Motee `[通讯]` (Lehigh University)

**通讯引用:** 1744 | [OpenAlex ID](https://openalex.org/A5031516064)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种分布式多机器人下一最佳视角（NBV）选择框架，用本地3D高斯散射（3DGS）地图实现安全路径规划。

**💡 创新点**

创新点包括：①仅在轨迹相关区域计算信息增益，构造风险感知掩模；②采用平均风险价值（AV@R）评估碰撞风险；③利用Consensus ADMM在通信图上实现仅交换候选视角和EIG标量的低通信量分布式优化；④在分布式环境下保持与中心化方法相近的地图质量与路径安全。

**🔧 技术方法**

技术：3D Gaussian Splatting、信息论期望信息增益（EIG）与Fisher信息近似、平均风险价值（AV@R）风险度量、Consensus ADMM（C-ADMM）分布式优化、风险感知掩模、A*风险避障路径规划。

**📊 数据集**

数据集：Habitat模拟器中的Gibson室内环境（Denmark、Cantwell、Ribera、Swormville等），使用RGB‑D传感器采集的图像进行地图构建。

**📈 对比分析**

方法对比：中心化、分布式、有限共享与单机器人基线。实验显示分布式方法在轨迹风险（AV@R）与PSNR/深度误差方面仅略逊于中心化，显著降低通信量（从数MB降到几十KB），并在多机器人场景中实现与中心化相近的安全性与地图质量。

**⚠️ 局限性**

局限：仅在仿真环境中验证，未考虑真实世界的噪声与硬件限制；框架聚焦于轨迹相关不确定性减少，对全局场景重建无优化；要求机器人具备足够计算与通信能力。

---

## 160. Creative Reading: Scaffolding Reading for Transformation

**arXiv ID:** 2606.04308 | [PDF](https://arxiv.org/pdf/2606.04308v1)

**作者:** Sophia Liu `[一作]` (University of California, Berkeley), Max Kreminski `[通讯]` (Cornell Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本文提出“创意阅读”概念，主张阅读增补系统不应只关注信息传输，而应支持读者在阅读过程中创造阅读产物并塑造自身阅读者身份。

**💡 创新点**

创新点在于将文学理论（读者响应、交易式阅读）与学术阅读增补结合，提出以阅读转化为中心的设计维度，并绘制了四个取向与两个机会空间的创意阅读设计空间。

**🔧 技术方法**

主要技术借鉴自超文本、交互式数字叙事（IDN）和现有学术阅读辅助工具（如摘要生成、文献导航、注释空间）等；未给出具体实现细节。

**📊 数据集**

文章未使用任何公开数据集，讨论基于理论与现有工具的概念性分析。

**📈 对比分析**

未开展实验或性能评估，因本文为概念性与设计空间研究。

**⚠️ 局限性**

局限性在于缺乏实证验证与系统实现，未探讨技术可行性与用户体验评估，理论假设需进一步实验检验。

---

## 161. Edge of Stability Selectively Shapes Learning Across the Data Distribution

**arXiv ID:** 2606.04212 | [PDF](https://arxiv.org/pdf/2606.04212v1)

**作者:** Shauna Kwag `[一作]` (Massachusetts Institute of Technology), Pierfrancesco Beneventano `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了深度网络在“边缘稳定性”（Edge of Stability, EoS）下对训练数据子集的学习分配，并通过分支干预实验揭示其选择性影响。

**💡 创新点**

发现EoS是选择性而非全局的学习机制，阐明了梯度方向与持久性共同决定哪些子集受益，并将数据几何与优化动力学关联。

**🔧 技术方法**

采用全批量梯度下降、Hessian特征分解、方向性对齐度量（cos^2θ）、自稳化（self‑stabilization）理论、对比实验与随机方向扰动等技术。

**📊 数据集**

以二分类的CIFAR‑10（汽车 vs 卡车，10,000样本）为主，构造了四类原型子集，并在更难的猫 vs 狗对进行验证。

**📈 对比分析**

通过分支实验在进入与退出EoS期间比较原型子集损失差异，发现对输入外类和输出外类的提升，反之对内类与边界点的下降；在对抗鲁棒性与 OOD 泛化方面，EoS 的好处取决于优势子集的变化。

**⚠️ 局限性**

实验受限于小模型、全批训练和像素空间原型，无法覆盖大规模数据、多分类、mini‑batch 优化及特征空间原型；对抗与泛化结果仅为初步，需进一步验证。

---

## 162. The Biomimetic Architecture of Software 4.0

**arXiv ID:** 2606.04025 | [PDF](https://arxiv.org/pdf/2606.04025v1)

**作者:** Philip Sheldrake `[一作]` (Unnamed Labs), Dirk Scheffler `[通讯]` (Unnamed Labs)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Software 4.0 架构，构建人类、神经网络与可反射符号子系统的自我维护异质层次，并设计了 Recognitive 语言与平台实现该架构。

**💡 创新点**

创新点在于引入 exo‑homoiconicity 与自我复制（autopoietic）奇点，形成连接主义推理与符号验证天然耦合的“奇点循环”，实现结构自我验证与动态自适应，而非传统外部 harness。

**🔧 技术方法**

技术上结合了可反射的符号子系统、Transformer‑LLM 的上下文注入、homoiconic 结构化语法，以及逻辑编程式形式化验证工具。

**📊 数据集**

本文未给出具体数据集，采用理论与概念性的结构化语法与 LLM 交互模型，后续计划使用大型代码库与自动化测试集。

**📈 对比分析**

暂无对比实验与性能评估，本文仅提出架构蓝图，未来将通过 Recognitive 在现有 Java/TypeScript 环境中的原型验证，并与传统 3.x harness 对比。

**⚠️ 局限性**

局限性包括缺乏实证验证、对大规模 LLM 运行时资源需求未知、实现细节尚不完善，以及对不同语言生态的适配挑战。

---

## 163. Thinking Through Signs: PEEL as a Semiotic Scaffolding for Epistemically Accountable AI-Enabled Research

**arXiv ID:** 2606.04152 | [PDF](https://arxiv.org/pdf/2606.04152v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 164. CoPark: Learning Reactive Parking via Self-Play

**arXiv ID:** 2606.04149 | [PDF](https://arxiv.org/pdf/2606.04149v1)

**作者:** Jiarong Wei `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**通讯引用:** 2676 | [OpenAlex ID](https://openalex.org/A5039639553)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种自对弈强化学习框架，通过残差策略与离线几何先验相结合，实现多车在共享停车场中实现子米级终端精度与全过程交互的自适应泊车。

**💡 创新点**

创新点包括：①按任务各向异性释放几何先验的通道优先权；②通过合作威胁信号实现纵向通道的动态放权；③闭环终端细化层补偿离散动作导致的误差；④证明几何先验按纵向/横向需求分离可提升交互与精度。

**🔧 技术方法**

使用技术：多代理自对弈RL（PPO+V-trace）、Hybrid A*+Reeds–Shepp离线路径、Stanley跟踪先验、残差策略网络、合作威胁信号、闭环终端细化、场景密度自适应训练。

**📊 数据集**

数据集：训练使用六个自定义停车场场景；零射评估集成Dragon Lake Parking (DLP) 与 DeepScenario Open 3D (DSC3D)；均在PufferDrive模拟器中跑实验。

**📈 对比分析**

与经典规划器（RS, Hybrid A*, MA-MPC）、HOPE、GigaFlow、CaRL、Diffusion Planner进行对比；零射成功率约84.7%/79.7%（DLP/DSC3D），碰撞率3–6%，显著优于基线，并且能自然展现逆向让行、紧道通行、排队等交互行为。

**⚠️ 局限性**

局限性：依赖离线几何先验与离散动作网格；训练规模受限（最多32车/场景）；缺少真实世界验证；未探索完全端到端的精度提升方法；对超大规模多车环境及更长训练周期的扩展尚待研究。

---

## 165. ACEAPEX: Parallel LZ77 Decoding via Encode-Time Absolute Offset Resolution

**arXiv ID:** 2606.04268 | [PDF](https://arxiv.org/pdf/2606.04268v1)

**作者:** Yakiv Shavidze `[一作]` `[通讯]` (ACE), Yakiv Shavidze (ACE)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于绝对位移编码的LZ77压缩解码器ACEAPEX，实现单流压缩后可按1 MB块并行解码；

**💡 创新点**

创新点在于将所有回引用改为绝对输出位置并通过链平坦化消除块内依赖，从而实现线性CPU并行和GPU波前执行；

**🔧 技术方法**

使用绝对偏移编码、链平坦化、四条预解码流、CPU多线程并行解码、GPU波前匹配解码以及深度限制编码器；

**📊 数据集**

在nci（33 MB）、FASTQ（1 GB）、silesia.tar（202 MB）和enwik9（1 GB）四个标准数据集上评测；

**📈 对比分析**

与zstd -3（近似压缩比）对比，ACEAPEX在EPYC 4344P上8线程可达10 160 MB/s（3.13×提升），在EPYC 9575F 64线程可达10 869 MB/s（2.71×提升）；GPU波前解码在H100 SXM上对enwik9实现44 GB/s，FASTQ 20 GB/s；深度限制编码器（深度10）使enwik9在单H100达到125.9 GB/s、双H100 249.9 GB/s，压缩比仅提升1.5%；

**⚠️ 局限性**

限制包括编码器速度约7×慢、内存占用约2.8 GB、链平坦化仅对同块链有效、深度限制导致压缩比下降、GPU实现仅完成匹配阶段，整体端到端GPU流水线尚未完成；

---

## 166. Novel Aspects of IEEE SA P3109 Arithmetic Formats for Machine Learning

**arXiv ID:** 2606.04028 | [PDF](https://arxiv.org/pdf/2606.04028v1)

**作者:** Andrew Fitzgibbon `[一作]` (Graphcore), Jeffrey Sarnoff `[通讯]` (IEEE)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

制定了一族可参数化的二进制浮点格式及其算术运算，专为机器学习系统设计。

**💡 创新点**

创新点包括统一的尺度不变 κ‑近似度量、单一 NaN、统一零处理、可选舍入与饱和模式，以及块运算的规范。

**🔧 技术方法**

采用了形式化规格（IML/Lean）自动生成运算定义、闭合扩展实数模型以及参数化设计技术。

**📊 数据集**

未使用任何实际数据集，本文主要聚焦标准设计与形式化验证。

**📈 对比分析**

通过形式化验证与测试向量生成验证实现一致性；性能提升体现在块运算与缩放算法，但未给出具体数值比较。

**⚠️ 局限性**

局限性包括仅适用于3位及以上的格式、缺乏多 NaN/多零支持、需实现方自行决定缩放策略，以及有限域与无穷大处理的兼容性问题。

---

## 167. Toward Pre-Deployment Assurance for Enterprise AI Agents: Ontology-Grounded Simulation and Trust Certification

**arXiv ID:** 2606.04037 | [PDF](https://arxiv.org/pdf/2606.04037v1)

**作者:** Thanh Luong Tuan `[一作]` (Golden Gate University), Abhijit Sanyal `[通讯]` (Novartis Healthcare Pvt. Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种面向企业AI代理的预部署验证框架，包括基于行业本体的情景生成、操作范围定义和可机器验证的信任证书；

**💡 创新点**

创新点在于利用结构化行业本体作为规范语言和测试情景来源，实现自动化、可扩展且可追溯的法规覆盖测试，并将验证结果绑定到可签名的证书；

**🔧 技术方法**

核心技术包括大型语言模型（Claude Sonnet 4、Qwen 2.5 72B、Gemma 4 26B）作为生成器和判定器、基于本体的规则推理与情景生成算法、Rust‑Native仿真运行器、LLM‑as‑Judge评估流程、Prometheus监控和签名加密的证书格式；

**📊 数据集**

使用了由三家LLM生成器各产生1,800个情景（共5,400个），覆盖5个行业‑监管组合（金融、银行、保险、医疗、越南监管），共125条监管要求、25个注入故障；

**📈 对比分析**

通过对比四种生成策略（基础、角色/人设、RAG、完整本体）以及跨模型验证，发现基于本体的生成在监管覆盖率（48.3% vs 33.1%）和行业特异性（4.77/5）上显著优于对照组；在多模型测试中保持一致；故障检测率差异不显著；

**⚠️ 局限性**

局限性包括：本体完整性受限、LLM‑as‑Judge可能存在自增强偏差、实验仅覆盖统计保障未实现形式化保证、阈值和证书标准尚未与真实部署事故率校准、缺乏多代理系统的组合验证等。

---

## 168. When Offline Selectors Cannot Beat the Best Single Model: A Diagnostic Study on edX Dropout Prediction

**arXiv ID:** 2606.04161 | [PDF](https://arxiv.org/pdf/2606.04161v1)

**作者:** Tyler Crosse `[一作]` (Georgia Institute of Technology), David Joyner `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1695 | [OpenAlex ID](https://openalex.org/A5103136135)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并验证了一套三阶段诊断流程，用于评估离线元学习器在动态模型选择任务中的瓶颈，并在edX的课程放弃预测任务中应用该流程。

**💡 创新点**

创新点在于将k‑NN标签一致性、监督与离线RL对照以及特征消融三个诊断结合为一个连贯的流程，使研究者能够快速判断是算法、表征还是数据问题导致选择器失效。

**🔧 技术方法**

技术包括：k‑NN一致性测度、监督行为克隆（BC）、离线DQN与CQL（含保守项）训练、特征消融与基于模型概率的特征变换，以及总变差距离（TV）评估缓冲区与测试集的分布偏移。

**📊 数据集**

使用edX点击流数据，包含84.5 M事件、223,505名学生‑课程对，经过4折交叉验证构造缓冲区，特征维度38（28行为特征+5模型概率+1平均+3一热），评估窗口(14 天观察‑14 天预测)为主。

**📈 对比分析**

与单一最佳基模型（0.762准确率）对比，oracle能提升至0.825；但BC、DQN、CQL以及特征消融后的选择器均聚集在0.748–0.753，未能接近oracle，说明无法利用oracle头房。算法在不同超参和样本量下表现一致，凸显问题来自状态歧义。

**⚠️ 局限性**

局限性：仅来自一门计算机科学课程且单一机构；使用γ=0的上下文‑Bandit化简未能捕获课程内状态转移；仅评估离线性能，未探究在线自适应；对低活跃学生过滤可能导致结果不具普适性；公平性与干预成本未在所有子群体上验证。

---

## 169. Overview of the EReL@MIR 2025 Multimodal Document Retrieval Challenge (Track 1)

**arXiv ID:** 2606.04240 | [PDF](https://arxiv.org/pdf/2606.04240v1)

**作者:** Jingbiao Mei `[一作]` (University of Cambridge), Jingbiao Mei `[通讯]` (University of Cambridge)

**通讯引用:** 48 | [OpenAlex ID](https://openalex.org/A5071561232)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文整理并评测了 EReL@MIR 2025 Multimodal Document Retrieval Challenge（Track 1），在统一模型约束下同时覆盖了文本到文档页检索（MMDocIR）和图像/图像+文本到开放域片段检索（M2KR）两大检索场景，公布了挑战设计、数据集、评测协议、参赛情况以及三支获奖队伍的系统实现与性能对比。

**💡 创新点**

创新点包括：① 在同一统一模型框架下同时解决两种截然不同的检索任务；② 通过大规模多模态 LLM（Qwen2‑VL 系列）嵌入器取代传统 CLIP/PreFLMR，提升了跨模态检索的世界知识匹配能力；③ 展示训练‑free 的多路复用与强力重排序方法即可与微调版多模型集成相竞争；④ 通过视觉近似匹配（视觉锚点）显著提升开放域检索效果，提示数据集设计的潜在偏倚。

**🔧 技术方法**

核心技术包括：多模态 LLM 嵌入器（GME‑Qwen2‑VL‑7B、ColQwen2‑7B）、Late Interaction + MaxSim 计算、k×EOS 平均池化、LoRA 微调五模型集成、视觉锚点（DINOv2 子图相似度）、多路复合检索（文本、图像、布局）以及大型 Vision‑Language 模型（Qwen2.5‑VL‑72B）做二分类重排序。

**📊 数据集**

使用的数据集：
- M2KR‑Challenge：6 415 个图像/图像+文本查询，对应 47 318 条维基式片段（文本+网页截图）。
- MMDocIR‑Challenge：1 658 条文本查询，覆盖 313 篇长文档，总计 20 395 页（平均 65.1 页/文档），每页提供截图、OCR 文本和 VLM 生成的文本描述。

**📈 对比分析**

评测方法：对每个任务计算 Recall@1、Recall@3、Recall@5 的平均值；最终排行榜为两任务平均 Recall 的宏平均。三支获奖队伍表现：iLearn 65.69（微调五模型集成+视觉锚点），LLMHunter 65.59（训练‑free 多路检索+重排序），GPU is all you need 57.30（零训 ColQwen2 直插直出）。官方基线 PreFLMR 在 M2KR 上得到 31.6 的平均 Recall，显著低于获奖系统。

**⚠️ 局限性**

局限性：
- 单一宏平均分数掩盖了两任务之间的差异，未能展示每任务的细粒度表现。
- 两任务的查询量差异大，但在最终得分中被等权重对待，可能影响评估公平性。
- 对视觉近似匹配的高依赖暴露了数据集潜在的复制/相似性偏差，未来挑战需改进数据构造。
- 仅报告了整体分数，未公开每个队伍的 per‑task Recall@1/3/5，限制了深入对比与复现。
- 仅以两项检索任务为测试范围，未涵盖更广泛的文档布局或多模态交互需求。

---

## 170. What Are We Actually Benchmarking in Robot Manipulation?

**arXiv ID:** 2606.04233 | [PDF](https://arxiv.org/pdf/2606.04233v1)

**作者:** Tianchong Jiang `[一作]` (Toyota Technological Institute at Chicago), Matthew Walter `[通讯]` (Toyota Technological Institute at Chicago)

**通讯引用:** 5185 | [OpenAlex ID](https://openalex.org/A5103153703)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

审计了五大机器人操作基准（LIBERO、CALVIN、SimplerEnv、RoboCasa、RoboTwin 2.0）的有效性，提出四种诊断方法评估基准分数是否真正反映通用操作能力。

**💡 创新点**

提出了快捷解（shortcut solvability）、统计显著性、慢性过拟合（creeping overfitting）以及数据源依赖四个诊断指标，并首次在操作基准上系统评估它们。

**🔧 技术方法**

利用对比实验（probe 模型、重新采样实验、统计显著性检验）以及公开实现的诊断脚本进行评估。

**📊 数据集**

使用上述五个公开基准的数据集进行实验，并在每个基准上收集多篇论文的 SOTA 结果。

**📈 对比分析**

将诊断结果与论文中声称的 SOTA 进行对比，发现 LIBERO、CALVIN、SimplerEnv 在多项诊断上失败，RoboTwin 2.0 与 RoboCasa 效果相对较好。

**⚠️ 局限性**

受限于部分基准未公开完整权重和评测脚本，诊断覆盖范围受限；诊断依赖固定测试集，无法完全解决基准与真实环境的差距。

---

## 171. An Effective Pauli-Channel Model for Passive-User Loop-Back QKD

**arXiv ID:** 2606.04247 | [PDF](https://arxiv.org/pdf/2606.04247v1)

**作者:** Luis Adrián Lizama-Pérez `[一作]` (Universidad Autónoma Metropolitana), Luis Adrián Lizama-Pérez `[通讯]` (Universidad Autónoma Metropolitana)

**通讯引用:** 137 | [OpenAlex ID](https://openalex.org/A5076974112)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文开发了一种有效的通道模型，用于分布式被动用户的回路量子密钥分发（QKD）。在该模型中，两个被动用户通过一个主动用户（爱丽丝）建立共享的秘密密钥。

**💡 创新点**

创新点在于将两个被动用户的操作外部封装为一个有效的回路节点，并将其表示为一种各向异性的保利通道，具有身份、X和Z分量，而没有Y分量。

**🔧 技术方法**

使用了各向异性的保利通道模型来描述被动用户的回路QKD，并通过量子态的准备和测量统计来分析。

**📊 数据集**

论文中没有具体提到使用的数据集，但提到的量子态是BB84协议中的极化态。

**📈 对比分析**

与传统的量子密钥分发方法相比，该模型通过有效的回路节点描述，恢复了理想的结论事件概率P_conc=1/4，并表明引入的干扰不是各向同性的去极化，而是由被动组成机制生成的内在通道。

**⚠️ 局限性**

限制在于该模型并不是一个完整的可组合安全性证明，安全性仍然依赖于中间状态不可完美区分的条件，以及由此产生的信息-干扰权衡。

---

## 172. Stationarity-Aware Retrieval-Augmented Time Series Forecasting

**arXiv ID:** 2606.04135 | [PDF](https://arxiv.org/pdf/2606.04135v1)

**作者:** Shiqiao Zhou `[一作]` (University of Birmingham), Shuo Wang `[通讯]` (University of Birmingham)

**通讯引用:** 7653 | [OpenAlex ID](https://openalex.org/A5100639215)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了 SARAF，一种考虑数据平稳性的检索增强时间序列预测框架，融合了时间对齐检索、平稳性驱动的多样性选择和自适应高斯聚合。

**💡 创新点**

创新点：① 引入平稳性感知的检索与融合策略，动态平衡相似度与多样性；② 在检索阶段加入时间对齐奖金，提升历史段的时序匹配度；③ 采用自适应高斯加权聚合，根据数据平稳性调节聚合宽度。

**🔧 技术方法**

技术手段：基于 Pearson 相关的检索、时间对齐奖金、MMR 多样性选择、平稳性估计（局部均值/方差变化）、Gaussian 加权聚合、线性预测骨干以及轻量级融合与投影。

**📊 数据集**

实验数据集：ETTh1/2、ETTm1/2、Exchange、Solar、Electricity、Traffic，共八个多变量时间序列基准。

**📈 对比分析**

方法比较：在统一实验协议下，与 Autoformer、Non-stationary Transformer、PatchTST、DLinear、RAFT、CycleNet、TimesNet、TimeMixer、DUET 等九个主流基线对比。SARAF 在 5/8 个数据集上获得最优或接近最优的 MSE/MAE，整体平均提升约 3.85% MSE 与 1.87% MAE，尤其在非平稳数据（如 Exchange）上表现突出。

**⚠️ 局限性**

局限性：对极高维度或实时检索的计算开销仍有提升空间；在极度平稳的数据上增量收益有限；缺乏在线检索数据库动态更新与自适应机制。

---

## 173. Cross-Prompt Generalization in Detecting AI-Generated Fake News Using Interpretable Linguistic Features

**arXiv ID:** 2606.04199 | [PDF](https://arxiv.org/pdf/2606.04199v1)

**作者:** Aya Vera-Jimenez `[一作]` (Kennesaw State University), Dhrubajyoti Ghosh `[通讯]` (Kennesaw State University)

**通讯引用:** 57 | [OpenAlex ID](https://openalex.org/A5102896122)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过构建三套基于不同提示词的 AI 生成假新闻与真实新闻混合数据集，探讨了假新闻检测模型在跨提示场景下的泛化性能；

**💡 创新点**

创新点在于提出跨提示评估框架，系统评估并证明可解释的语言特征（词汇多样性、可读性、情感特征）在不同提示下保持稳定的检测能力；

**🔧 技术方法**

采用结构化特征提取（包括词汇类型-词汇比例、可读性指标、NRC 情感词典计数）并使用随机森林分类器；

**📊 数据集**

使用三组由 ChatGPT 在不同提示（A、B、C）生成的假新闻与 PolitiFact 数据集的真实新闻组成的三种数据集（D_A、D_B、D_C）；

**📈 对比分析**

采用交叉提示训练-测试（6种组合）进行评估，使用 AUC 作为指标；结果显示所有组合的 AUC 均在 0.988–1.000 之间，表明模型在跨提示情境下保持极高的检测准确率；

**⚠️ 局限性**

局限性包括仅使用单一 LLM（ChatGPT）和有限的提示种类，未涵盖更广泛的生成模型或极具挑战性的对抗性提示；此外未引入更深层次的语义或语篇特征，可能进一步提升检测效果。

---

## 174. Optimal Transport Flow Matching by Design

**arXiv ID:** 2606.04092 | [PDF](https://arxiv.org/pdf/2606.04092v1)

**作者:** Shimon Malnick `[一作]` (Tel Aviv University), Shai Avidan `[通讯]` (Tel Aviv University)

**通讯引用:** 16356 | [OpenAlex ID](https://openalex.org/A5068635313)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出在流匹配中通过设计先验分布实现最佳传输耦合，即使用图像低频投影作为先验，保证与数据的身份耦合近似最优，从而得到更直的生成轨迹，提升少步生成质量。

**💡 创新点**

创新点在于不通过求解高维最优传输，而是直接设计先验，使其与数据的身份耦合在经验上已满足最优传输，结合低频投影与高斯噪声插值进一步提升性能。

**🔧 技术方法**

主要技术包括流匹配（flow matching）框架、低频投影与上采样构造先验、Gaussian噪声插值、Hungarian 算法验证 OT-身份耦合、轻量级生成器 G_φ 采样低频先验。

**📊 数据集**

使用的主要数据集包括 CIFAR‑10（32×32），FFHQ（256×256）与 ImageNet（256×256）（后者在潜在空间下训练）。

**📈 对比分析**

与 IFM、OT‑FM、AlignFlow 等基线比较，本文在 1–8 步有效 NFE 范围内曲率显著降低（≥2×），FID 下降 20–40% 以上，尤其在 1 步时表现突出；在 MeanFlow 一步生成框架中同样获得更佳 FID 与更直的轨迹。

**⚠️ 局限性**

局限性包括：①生成低频先验时仍需额外的轻量级模型 G_φ，导致一定推理开销；②整个管线并非完全 OT‑耦合，G_φ 与主模型训练解耦；③低频投影的 OT‑身份耦合特性依赖于自然图像的频谱特性，对其他模态或数据分布需单独验证。

---

## 175. Can Generalist Agents Automate Data Curation?

**arXiv ID:** 2606.04261 | [PDF](https://arxiv.org/pdf/2606.04261v1)

**作者:** Feiyang Kang `[一作]` (Virginia Tech), Ruoxi Jia `[通讯]` (Virginia Tech)

**通讯引用:** 2849 | [OpenAlex ID](https://openalex.org/A5032275274)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基准（名为**DataCurator**）来评估通用编码代理在训练数据策划中的迭代策略搜索过程；该基准固定模型、训练和评估环境，只允许代理改变数据子集，并记录完整的策划轨迹。

**💡 创新点**

核心创新在于：1）把数据策划视为可搜索的策略循环，并在基准中强制记录每一步的脚本、日志和评估结果；2）设计了不同层级的“脚手架”来诱导代理从执行层面向研究层面转化，验证方法适配（paper‑adaptation）能够显著提升性能。

**🔧 技术方法**

使用了通用大型语言模型（Claude Code、Codex、Qwen3.5等）作为代理，通过终端接口编写/修改数据筛选脚本，执行训练（LLaVA‑1.5‑7B、SmolVLM‑Base、CLIP‑ViT‑B/32 等）并评估（8项 VLM 基准）。

**📊 数据集**

主要数据集为 LLaVA‑665K（instruction‑tuning pool）、Vision‑Flan 186k、CLIP DataComp Small；在 10k 示例子集上进行 fine‑tuning 与预训练实验。

**📈 对比分析**

与随机选择和公开的手工设计数据选择基线（ICONS、ARDS 等）比较，开放式提示代理在 10k 预算下可恢复约 60% 的全量 fine‑tune 提升；而在采用“方法适配”脚手架后，代理在同一 10k 预算下实现 34.9 的平均分，超过了 100k 随机、ICONS/ARDS 基线，且仅使用十分之一的数据量。

**⚠️ 局限性**

限制包括：实验仅覆盖视觉‑语言指令调优和少量预训练；脚手架效果不完全可分离，难以确定单一因子；轨迹标签需要人工或 LLM 评判，可能带来主观偏差；在更大规模或不同任务（代码、数学等）中结果可能不再适用。

---

## 176. SMAC-Talk: A Natural Language Extension of the StarCraft Multi-Agent Challenge for Large Language Models

**arXiv ID:** 2606.04202 | [PDF](https://arxiv.org/pdf/2606.04202v1)

**作者:** Joel Sol `[一作]` (University of Victoria), Homayoun Najjaran `[通讯]` (University of Victoria)

**通讯引用:** 6092 | [OpenAlex ID](https://openalex.org/A5058540009)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SMAC-Talk，一种自然语言扩展的StarCraft多智能体挑战，用于评估LLM在分散、部分可观测的协作环境中的协调与通信能力

**💡 创新点**

创新点在于：①将SMACv2的数值观测、动作与通信转为自然语言；②设计了包含欺骗沟通的多场景评估；③对不同LLM规模与推理方式进行系统比较；

**🔧 技术方法**

技术包括：观察‑文本适配器、文本‑动作适配器、自然语言通信层；使用Qwen3.5系列LLM（4B‑122B）在vLLM框架下推理；

**📊 数据集**

数据集为SMACv2（Terran 单位，敌方级别Very Easy）及其自定义的欺骗沟通场景

**📈 对比分析**

比较方法：在每个场景下跑100集，评估赢率、奖励、动作错误率；结果显示：内部链式推理（Reasoning Agent）在所有规模下优于零射和ReAct，规模越大越能利用通信并抵御欺骗；ReAct在通信下性能显著下降；

**⚠️ 局限性**

局限性：①高计算成本，4B模型表现不足；②仅测试Terran与固定敌方难度；③未探索不同推理预算、细粒度指令对性能的影响；④对ReAct通信崩溃的根本原因未明；

---

## 177. Caught in the Act(ivation): Toward Pre-Output and Multi-Turn Detection of Credential Exfiltration by LLM Agents

**arXiv ID:** 2606.04141 | [PDF](https://arxiv.org/pdf/2606.04141v1)

**作者:** Kargi Chauhan `[一作]` (University of California Santa Cruz), Pratibha Revankar `[通讯]` (University of California Santa Cruz)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Agentic Immune System，结合预输出激活探测、差分隐私生成的蜜罐令牌以及多轮累计泄露监测，防止 LLM 代理泄露凭证。

**💡 创新点**

创新点在于把模型内部激活特征作为预输出泄露信号，使用差分隐私字符模型生成不可识别的蜜罐令牌，并通过 InfoNCE 对多轮泄露进行累计评估，将三种防御层级整合成完整方案。

**🔧 技术方法**

技术包括激活特征的线性探测器与 Mahalanobis 距离、差分隐私的字符二元组模型与 conformal 校准生成蜜罐、以及基于 InfoNCE 的信息流累计估计器。

**📊 数据集**

使用的数据集包括公开注入基准（TensorTrust、InjecAgent、BIPIA、AgentDojo）、独立编码逃逸测试集、50 句包含 20 回合的合成多轮泄露会话，以及 1,000 条正常任务提示。

**📈 对比分析**

与文本级子串/模糊/语义嵌入及 LlamaGuard 比较；激活探测在 Qwen‑7B 上 AUROC 0.998，蜜罐检测在合成情景下精准率与召回率均为 1.0；累计泄露在多轮实验中将检测率从 18% 提升至 90%，但单轮泄露检测仍受 InfoNCE 上限限制。

**⚠️ 局限性**

局限性包括：需要白盒激活访问，InfoNCE 的上限限制单轮泄露检测，合成多轮数据规模有限，未覆盖通过结构化工具调用传递凭证的情况，conformal 校准假设可交换性，闭源 API 版本无法使用激活探测。

---

## 178. Channel-Oriented Design for EEG-to-Music Reconstruction

**arXiv ID:** 2606.04040 | [PDF](https://arxiv.org/pdf/2606.04040v1)

**作者:** Jiaxin Qing `[一作]` (University of California Berkeley), Lexin Li `[通讯]` (University of California Berkeley)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了 channel-oriented EEG‑to‑music 还原框架，保留电极层级信息并提升音乐语义重建。

**💡 创新点**

引入 channel‑wise tokenization、multi‑view self‑distillation 与 channel dropout 三重策略，理论证明其能降低跨簇重叠，从而更好利用弱分布式 EEG 信号。

**🔧 技术方法**

使用 Transformer‑级别的 channel token 编码、DINO 风格的自蒸馏、多尺度视图与结构化通道抖动、CLIP 风格对齐以及 AudioLDM 扩散音频生成。

**📊 数据集**

采用 NMED‑T 与 NMED‑H 两大自然音乐 EEG 数据集，总计 29.4 小时 68 名受试者。

**📈 对比分析**

与 EEG2Mel、LaBraM、EEGPT、CBraMod 等基线在 50‑way/14‑way 识别和 CLAP 分数上对比，取得 0.487/0.692 的 50/14 识别率和 0.683 的 CLAP 分数，明显优于现有最强模型。

**⚠️ 局限性**

数据量有限，仅覆盖音乐听觉场景，未对更大规模或跨任务的数据进行验证；使用 Ridge 适配器和固定解码器，未探索更强的适配方式；对文化差异和个体差异的解释仍有待深入。

---

## 179. GraftDB: Dynamic Folding of Concurrent Analytical Queries

**arXiv ID:** 2606.04303 | [PDF](https://arxiv.org/pdf/2606.04303v1)

**作者:** Genki Kimura `[一作]` (University of Tokyo), Kazuo Goda `[通讯]` (University of Tokyo)

**通讯引用:** 686 | [OpenAlex ID](https://openalex.org/A5016349116)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态折叠（Dynamic Folding）技术，使新到达的分析查询能够折入正在执行的共享工作流中，复用已产生的状态并共享后续工作，从而减少冗余计算。

**💡 创新点**

核心创新包括：①“状态中心化执行”（state‑centric execution），将哈希表、聚合器等状态视为共享对象；②为每个查询设计的“查询状态透镜”（per‑query state lens），动态决定查询可见的状态片段；③“查询嫁接”（query grafting）机制，在查询到达时将其与已有共享状态分配为代表(extent)、剩余(residual)和未附属(unattached)三类，保证语义正确。

**🔧 技术方法**

实现技术包括：状态元数据（coverage & visibility）、谓词包含与可评估性检查、共享扫描与残余生产、共享哈希表与聚合共享、DAG 调度与状态就绪门（state‑readiness gate）以及基于 Rust 的单线程执行引擎。

**📊 数据集**

使用 TPC‑H 基于参数化模板的动态并发工作负载（Q1、Q3–Q10），通过 Zipf 分布调节查询偏斜，实验涵盖 SF1–SF30 的数据规模。

**📈 对比分析**

与同引擎的“隔离执行（Isolated）”、QPipe‑OSP 的扫描/管道共享以及 PostgreSQL 进行对比。实验显示在 32 并发客户端时，GraftDB 达到 2.17× 的吞吐量提升，P95 响应时间降至 0.17× Isolated；在开放式 Poisson 到达下，P95 下降至 0.28×；在高偏斜或大规模数据下仍保持 1.3–1.6× 吞吐量提升。

**⚠️ 局限性**

主要局限：目前实现仅支持单线程、单核心执行，未充分利用多线程并行；仅支持有限 SQL 子集（无关联子查询等）；状态共享仅适用于有限制的状态（哈希构造、聚合），对更复杂的窗口/流式查询尚未扩展；在高度动态或频繁变更的查询集下，谓词包含检查的局限性可能导致共享机会被低估。

---

## 180. GlossAssist -- A Tool to Simplify Corpus Creation and Study the Effect of NLP Models in Low-Resource Documentation Settings

**arXiv ID:** 2606.04367 | [PDF](https://arxiv.org/pdf/2606.04367v1)

**作者:** Bhargav Shandilya `[一作]` (University of Colorado Boulder), Alexis Palmer `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1209 | [OpenAlex ID](https://openalex.org/A5069931383)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一款名为GlossAssist的交互式 IG文本标注工具，利用检索式CWoMP模型与可变词汇表实现可扩展的词形-词义对齐与翻译，支持语义级别的主动学习反馈；

**💡 创新点**

创新点在于将模型预测嵌入可编辑、可追溯的词汇表中，构建语义层级的反馈循环，将语言学家视为系统改进的协作者，并提供可视化证据以提升解释性与可用性；

**🔧 技术方法**

基于检索式的CWoMP预训练架构（Contrastive Word‑Morpheme Pre‑training），配合前端的多面板界面和日志收集，支持活跃学习与词汇表自动扩展；

**📊 数据集**

主要利用跨语言的 IG文本语料库（如 SIGMORPHON 2023 共享任务所提供的多语言 IGT 语料）以及内部构建的可变词汇表；

**📈 对比分析**

论文未给出量化实验结果，主要通过用户体验与工作流程指标（如平均标注时间、接受率、错误率）来评估工具效果，并与传统评测指标（MER、WER、chRF）形成对比，强调评估的面向工作效率与错误诊断；

**⚠️ 局限性**

限制包括：当前只能生成已在词汇表中的词形，缺乏 OOV 预测功能；模型错误易被接受导致误传；界面与工具集成仍未与主流语言学软件（ELAN、FLEx）无缝衔接。

---

## 181. MeshTok: Efficient Multi-Scale Tokenization for Scalable PDE Transformers

**arXiv ID:** 2606.04366 | [PDF](https://arxiv.org/pdf/2606.04366v1)

**作者:** Yanshun Zhao `[一作]` (University of Science and Technology of China), Jingrun Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 2690 | [OpenAlex ID](https://openalex.org/A5025195102)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MeshTok，一种基于 Transformer 的多尺度自适应分辨率 token 化方法，用于高效建模 PDE 解决方案。

**💡 创新点**

创新点在于将 AMR 思路引入 token 化，使用梯度+拉普拉斯能量活动指标动态细化局部区域；结合几何感知 FiLM 位置信息，将异构 token 统一序列化并交给标准 Transformer 处理；并通过全局与局部解码融合实现细致修正。

**🔧 技术方法**

技术手段包括 Transformer + 块因果自注意力、梯度/拉普拉斯活动指标、自适应细化、几何感知位置编码、统一多尺度 token 序列、全局与局部解码融合与轻量级 CNN 级联。

**📊 数据集**

使用了多种公开 PDE 数据集：PDEBench、PDENNEval、The Well 等，包括 Gray‑Scott、Allen‑Cahn、CNS、SWE、Reaction‑Diffusion、Shear Flow、Burgers、Black‑Scholes‑Barenblatt 等。

**📈 对比分析**

与 ViT、MPP、DPOT、BCAT、MoE‑POT、FNO、DeepONet 等基线在相同训练与推理预算下进行对比。MeshTok 在多任务中表现出更低的相对误差，且在不同模型规模与 token 预算下保持优越性，显著提升了效率‑准确率平衡。

**⚠️ 局限性**

局限性：仅适用于规则结构网格，难以直接推广到非结构网格或不规则几何；自适应策略在大模型规模下收益递减；缺乏不确定性评估、OOD 检测等安全性与鲁棒性保障。

---

## 182. When Freshness Is Not Enough: Distribution-Aware Age of Information for Networked LQR Control

**arXiv ID:** 2606.04361 | [PDF](https://arxiv.org/pdf/2606.04361v1)

**作者:** Abdullah Y. Etcibasi `[一作]` (Ohio State University), Eylem Ekici `[通讯]` (Ohio State University)

**通讯引用:** 8528 | [OpenAlex ID](https://openalex.org/A5082165966)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文从无状态调度的无限期LQR跟踪问题出发，推导出与Age of Information（AoI）分布相关的闭环性能指标，并证明仅靠平均AoI并不能优化控制性能；

**💡 创新点**

创新点在于将LQR成本转化为对间隔分布的高阶/指数矩优化问题，揭示平均AoI不足；并证明在速率受限通道下“尽可能周期化”调度是最优；

**🔧 技术方法**

采用状态分离、矩分析、离散Jensen不等式、随机过程与马尔可夫链理论，以及指数相关噪声模型，进一步推导出不同系统参数（|a|、β）下的等价目标；

**📊 数据集**

利用美国交通部NGSIM US‑101高速公路实时车辆轨迹数据（约3900辆车，0.1 s采样）作为实际扰动序列；

**📈 对比分析**

通过对不同调度策略（周期化、随机、Erlang）和不同控制器（ZOH、冲击式、对称）在相同通信速率下的平均LQR成本进行比较，实验结果与理论等价成本高度吻合，显示平均AoI较低的策略并不一定带来更好的控制性能；

**⚠️ 局限性**

限制在于仅考虑无状态调度策略，未处理状态感知调度；模型假设噪声为指数相关且在极限r→0、β→0时简化，实际系统中可能存在非线性、时变或多变量耦合，需进一步扩展。

---

## 183. Learning to cooperate with emergent reputation via multi-agent reinforcement learning

**arXiv ID:** 2606.04359 | [PDF](https://arxiv.org/pdf/2606.04359v1)

**作者:** Xinwei Song `[一作]` (ShanghaiTech University), Xue Feng `[通讯]` (State Key Laboratory of General Artificial Intelligence, BIGAI)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出 COOPER（COOPeration with Emergent Reputation）算法，联合学习可区分的声誉评估模块（包括基于 gossip 的 ψ 与基于交互的 ϕ）和声誉驱动的策略 π，实现从环境奖励中自发地形成声誉规范与合作行为。

**💡 创新点**

创新点在于：①不依赖预设的声誉评估规则或将声誉作为内在奖励，而是通过环境外部奖励直接联合训练声誉评估与策略；②采用交互顺序 ψ→π→ϕ 与反向优化顺序 ψ→ϕ→π 的双向流动设计，消除声誉与策略深度耦合导致的学习不稳定；③引入社区一致性正则化与熵正则，提升学习效率与探索能力。

**🔧 技术方法**

技术手段包括：分布式多智能体强化学习、PPO 风格的策略梯度与优势估计、神经网络实现的 ψ、ϕ 与 π 模块、社交网络信息传播模型、图平滑正则与熵正则等。

**📊 数据集**

实验数据集为：1) Donation Game 与 Coin Game 的离散格子世界；2) 小世界、无标度、完全连通三类社交网络；3) 10 代理、50 步/episode、每步随机配对等设置；无公开数据集，所有实验均为自定义仿真环境。

**📈 对比分析**

与 PPO、LR2、RR、IR 等基线对比；COOPER 在自我对弈和适应已知声誉规则的情景下均表现出更高的合作比例、更大的个人与群体奖励，并能在多种网络拓扑下保持优势，说明其自适应与协同能力显著优于现有方法。

**⚠️ 局限性**

局限性包括：①算法在高度动态或规模极大网络下的可扩展性尚未验证；②对社交网络结构的依赖可能限制在现实复杂环境中的迁移；③缺乏在更复杂、连续状态空间或更大规模多智能体场景下的实证。

---

## 184. Expectations vs. Realities: The Cost of MSE-Optimal Forecasting Under Conditional Uncertainty

**arXiv ID:** 2606.04342 | [PDF](https://arxiv.org/pdf/2606.04342v1)

**作者:** Riku Green `[一作]` (University of Bristol), Telmo M Silva Filho `[通讯]` (University of Bristol)

**通讯引用:** 1333 | [OpenAlex ID](https://openalex.org/A5077682283)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了多步时间序列预测中，均方误差（MSE）评价会导致预测结果过度平滑且不代表真实未来分布，提出了条件不确定性间隙（conditional uncertainty gap）与点预测精度与分布真实性之间的不可兼容性；

**💡 创新点**

提出了条件不确定性间隙理论，证明在非零间隙时任何确定性预测器无法同时最小化MSE并匹配真实未来的边缘分布，揭示了固有的精度‑真实性 Pareto 前沿，并将不同预测策略与推理方式定位于该前沿；

**🔧 技术方法**

理论推导（条件方差分解、MSE 与 Wasserstein 距离不一致性证明）、实验评估（MSE、Wasserstein-1、DTW、Gromov‑Wasserstein 等）、模拟（Mackey‑Glass 加噪系统）与实测数据集上的多策略训练（递归 vs 直接、多输出、多模型、概率流）

**📊 数据集**

九个公开时间序列基准（电力、交通、天气、空气质量、金融等），每个数据集的7条序列在200 步长上评估；

**📈 对比分析**

对比方法包括线性岭回归、MLP、决策树、N‑BAYES、TTS、LSTM 等的递归与直接策略，以及基于概率流的均值与采样推理；结果显示在低条件不确定性下 MSE 与真实性几乎一致，随着不确定性增大形成明显 Pareto 前沿；在 5% MSE 容忍范围内，平均可获得 17.3% 的真实性提升，且大部分实例成功率超过 80%；

**⚠️ 局限性**

局限包括：仅考虑确定性预测器，概率预测器的完整前沿未完全解析；真实性度量仅为边缘 W1，未覆盖轨迹级别、校准或条件一致性；实验以站稳态时间序列为主，非平稳/概念漂移情形未深入探讨

---

## 185. From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents

**arXiv ID:** 2606.04329 | [PDF](https://arxiv.org/pdf/2606.04329v1)

**作者:** Pritam Dash `[一作]` (Huawei), Zhiwei Shang `[通讯]` (Huawei)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM驱动的AI代理中持久内存的“内存中毒”攻击进行了系统化研究，提出了攻击通道、脆弱性、攻击分类和评测基准，并对现有注入防御的缺陷进行评估。

**💡 创新点**

首次将内存写入通道与模型、提示和系统架构层面的脆弱性进行映射，构建了六类攻击分类法，并设计了两阶段评测基准（写入成功率与跨会话检索成功率），揭示了内存中毒与提示注入的本质区别。

**🔧 技术方法**

采用LLM推理评估、基于规则的写入路径检测、对现有四种提示注入防御（PIGuard、DataFilter、CommandSans、PromptArmor）进行离线评测，并使用GPT‑OSS‑120B模型模拟OpenClaw与HERMES两种代理。

**📊 数据集**

创建了包含3,240个测试案例的专用数据集，覆盖六类攻击和七个任务领域（文件、网页、邮件、日历、Slack、代码、技能调用），以及2,997个正常样本用于评估误报。

**📈 对比分析**

在OpenClaw与HERMES上分别测得平均写入成功率（ASR）为34.25%与66.67%，检索成功率（RSR）为17.40%与64.70%，表明HERMES更易被攻击；现有注入防御在强信号攻击下检测率高达84.44%，但对弱信号攻击仅42.5%，整体缺失完整覆盖。

**⚠️ 局限性**

实验仅使用单一模型（GPT‑OSS‑120B），缺乏跨模型泛化验证；评测中的外部内容注入方式为标注上下文，未完全模拟真实工具调用流程；缺少对多用户共享内存、权限提升场景的考虑。

---

## 186. Policy Gradient for Continuous-Time Robust Markov Decision Processes

**arXiv ID:** 2606.04335 | [PDF](https://arxiv.org/pdf/2606.04335v1)

**作者:** Tanya Veeravalli `[一作]` (Agency for Science, Technology and Research), Atsushi Nitanda `[通讯]` (Agency for Science, Technology and Research)

**通讯引用:** 405 | [OpenAlex ID](https://openalex.org/A5023953123)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文研究了在连续时间鲁棒马尔可夫决策过程（RMDPs）框架下的策略梯度算法，提出了双循环优化器以实现线性收敛和样本复杂度分析。

**💡 创新点**

创新点在于将鲁棒马尔可夫决策过程扩展到连续时间设置，并提出了路径导数和伴随导数的梯度计算方法，提供了新的工具用于分析无折扣总成本MDPs。

**🔧 技术方法**

使用了路径导数和伴随导数的计算方法，提出了双循环优化器和均场优化器，分别用于策略优化和分布优化。

**📊 数据集**

在LQR（线性二次调节器）问题上进行了实验，使用了神经常微分方程（ODE）动态和神经网络参数化的不确定性集。

**📈 对比分析**

通过与不同的策略和对手优化器的比较，实验证明了连续时间策略梯度算法在鲁棒优化性能上优于传统的离散时间策略梯度算法。

**⚠️ 局限性**

限制在于当前的分析主要集中在理论框架上，实际应用中可能面临的复杂性和计算成本尚未充分探讨。

---

## 187. A Geometric Characterization of the Stationary Plateau for Two-Layer Neural Networks

**arXiv ID:** 2606.04327 | [PDF](https://arxiv.org/pdf/2606.04327v1)

**作者:** Tian Ding `[一作]` (Shenzhen International Center of Industrial and Applied Mathematics), Ruoyu Sun `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过理论分析，阐明了在二维层网络中通过神经元分裂产生的平坦阶梯（stationary plateau）上的局部几何结构，完整地分类了这些平坦阶梯上的驻点是局部最优还是鞍点，并揭示了其出现的条件。

**💡 创新点**

创新点在于提出了“内 Hessian”矩阵作为判定工具，能够根据父神经元的曲率性质与分裂系数来精确预测嵌入宽网络后出现的所有局部极值类型；同时给出了“必然鞍点”区域与“所有鞍点”平坦阶梯的明确定理，统一并扩展了先前的结果。

**🔧 技术方法**

核心技术包括对神经元输出的泰勒展开、对损失函数二阶导数的分解、内 Hessian 的正定/负定/不定性判定以及严格的扰动/极值分析，构建了完整的理论证明框架。

**📊 数据集**

论文主要为理论性工作，没有使用具体的数据集；若需要实验验证，可在常见的回归/分类数据集（如 MNIST、CIFAR‑10）上进行验证。

**📈 对比分析**

与以往仅给出充分条件或经验结论的研究相比，本文提供了完全可检查的必要与充分条件，理论严谨；若进行实验验证，预期在宽度扩展时模型会出现更多鞍点，训练曲线与梯度下降轨迹更易逃离局部最优。

**⚠️ 局限性**

局限性包括：仅讨论单隐藏层、平滑激活函数的情形；对深层网络、非平滑激活或特殊结构的推广尚未给出；此外，分析依赖于非退化假设，在极端退化（如梯度零）场景下结论可能不再适用。

---

## 188. Exploring Cross-Scenario Generality of Agentic Memory Systems: Diagnostics and a Strong Baseline

**arXiv ID:** 2606.04315 | [PDF](https://arxiv.org/pdf/2606.04315v1)

**作者:** Zhikai Chen `[一作]` (Michigan State University), Jiliang Tang `[通讯]` (Michigan State University)

**通讯引用:** 26203 | [OpenAlex ID](https://openalex.org/A5040639891)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文重新评估了八种记忆系统和一个agent harness在五类任务上的跨场景通用性与成本，并提出了将索引化记忆与agent harness相结合的 AutoMEM 方案。

**💡 创新点**

创新点在于发现agent harness通过推迟存储结构确认并主动检索实现最优跨场景通用性，并基于此设计了 AutoMEM，将索引化记忆融入agent harness。

**🔧 技术方法**

使用了 LLM 工具调用、Plan-Execute-Judge 循环、图/笔记/分层存储等多种记忆架构，并跟踪 token 成本与延迟。

**📊 数据集**

使用的数据集包括 LoCoMo、HotpotQA、AMABench、MemoryAgentBench、ALFWorld 与 MemoryArena。

**📈 对比分析**

通过 LLM-judge 分数与环境成功率进行对比，记录 token 成本与延迟，AutoMEM 在大多数基准上获得最高平均排名，并在单轮 QA 与长程 agent 任务上显著提升性能。

**⚠️ 局限性**

限制在于未实现多层预算路由与不同模型分配给规划/判断/答案阶段，评估受严格判定器影响，在动态 agent 任务上记忆系统仍无法突破参数更新瓶颈。

---

## 189. Latent Anchor-Driven Test Generation for Deep Neural Networks

**arXiv ID:** 2606.04310 | [PDF](https://arxiv.org/pdf/2606.04310v1)

**作者:** Bin Duan `[一作]` (University of Queensland), Guowei Yang `[通讯]` (University of Queensland)

**通讯引用:** 1448 | [OpenAlex ID](https://openalex.org/A5039642499)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于锚点的潜在空间测试框架（Anchor-Driven Latent Space Test Generation），在黑盒图像分类器的潜在空间中通过锚点方向进行可控的一阶变异，生成语义相近、结果多样的缺陷触发测试样本。

**💡 创新点**

创新点在于：①从种子输入中心出发，使用来自不同类别的锚点定义多方向变异，充分利用潜在语义流形的结构；②通过单步可控变异和离散化量化，避免了传统潜在空间方法的过度漂移或重复；③在单模型和多模型测试两种oracle下均实现高效、低语义漂移的缺陷暴露。

**🔧 技术方法**

核心技术包括：预训练的Vector Quantized Variational Autoencoder（VQ‑VAE）用于获取离散潜在表示；锚点采样与方向变异算子；向量量化+解码映射回输入空间；基于预测不一致或模型不一致的测试oracle。

**📊 数据集**

实验使用五大图像数据集：MNIST、CIFAR‑10、ImageNet、FashionMNIST、SVHN，以及十个主流分类网络（LeNet‑4/5、VGG16/19、ResNet18/50、Custom-1.6B/3.3B、ALL‑CNN‑A/B）。

**📈 对比分析**

与基线（SINVAD、Mimicry、CIT4DNN）比较时，Anchor‑Driven在单模型测试中在故障计数、种子覆盖率、失败多样性上均优于或相当于基线，并且测试效率更高；在多模型测试中，故障计数和误差对比多样性最高；同时，语义漂移均显著低于对比方法。

**⚠️ 局限性**

局限性包括：仅验证于图像分类任务，迁移到其他任务（如目标检测、语义分割）仍需研究；锚点采样依赖类别标签，若标签缺失或噪声会影响效果；潜在空间的离散化仍可能导致生成样本在高维复杂数据上的多样性受限。

---

## 190. Organizational Control Layer: Governance Infrastructure at the Execution Boundary of LLM Agent Systems

**arXiv ID:** 2606.04306 | [PDF](https://arxiv.org/pdf/2606.04306v1)

**作者:** Tianyu Shi `[一作]` (McGill), Jiangbo Yu `[通讯]` (McGill)

**通讯引用:** 665 | [OpenAlex ID](https://openalex.org/A5083571765)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了组织控制层（OCL），将LLM代理生成的动作与执行分离，增强了在经济多代理交互中的安全与可靠性。

**💡 创新点**

创新点在于设计了模型无关的OCL治理架构，结合角色检查、约束门控、审计与升级流程，显著降低违规执行并提升合法成功率。

**🔧 技术方法**

采用OCL架构（角色策略、门控策略、审计与升级）以及多种LLM后端（GPT‑5.4、Gemini‑3.1、Qwen‑3.5）进行对抗性测试，使用JSON交互规范。

**📊 数据集**

使用AgenticPay环境构建的50个对抗性买家角色数据集（极低价、隐私钓鱼、角色劫持、模糊买家、时间浪费）以及真实世界的买卖谈判语料。

**📈 对比分析**

通过Baseline（无治理）与OCL治理版本对比，指标包括成功率、合法成功率、违规率、拦截率、轮数、延迟与奖励。OCL将违规率从88%降至0%，合法成功率从12%提升至96%，平均轮数和延迟显著下降，奖励保持在可接受范围内。

**⚠️ 局限性**

限制在于过度严格的治理可能降低在高约束或薄利市场的灵活性，且实验仅覆盖双边价格谈判与单一卖家角色，未来需扩展自适应门控、多角色部署和更广泛的经济机制。

---

## 191. RowNet: A Memory Transformer for Tabular Regression

**arXiv ID:** 2606.04445 | [PDF](https://arxiv.org/pdf/2606.04445v1)

**作者:** Askat Rakhymbekov `[一作]` (Kyrgyz-Turkish Manas University), Gulshat Muhametjanova `[通讯]` (Kyrgyz-Turkish Manas University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一种基于检索的神经网络架构，用全训练样本构成记忆池，在查询房源时通过行注意力检索相似房源并加权求平均，再通过多头注意、门控混合与残差校正来预测每平方米价格。

**💡 创新点**

创新点在于：①将相似度拆解为类别精确匹配与多核数值相似度，形成可训练的行相似向量；②设计两层检索：先做特征相似检索，再加入目标一致性特征进行二次检索；③引入多头注意与混合专家门控、残差MLP、熵正则和头部多样性正则，使模型在可解释的检索框架内实现局部自适应与全局校正；④通过软max行注意使得预测可视化为历史可比房源的加权平均。

**🔧 技术方法**

技术实现包括：记忆网络 + 行注意力（softmax over全训练行），多头线性兼容性，残差校正 MLP，门控混合（γ=0.5）与软最小残差分布，熵正则、头部多样性正则；训练采用 Adam（lr=1e-2）+余弦退火至1e-6，10 轮，AMP混合精度；损失为 MAPE + 残差惩罚 + 熵正则 + 头部多样性正则。

**📊 数据集**

使用 Kyrgyzstan Bishkek 城市的房产估价数据集，训练集 7,128 条，测试集 1,784 条，经过预处理后共 156 个特征（包含类别、数值、文本标签扩展、地址统计等）。

**📈 对比分析**

与无检索基线（14.76% MAPE）和 XGBoost 基线（8.33% MAPE）相比，模型在训练集 leave‑one‑out 的 MAPE 从 8.36% 降至 6.37%，竞赛评估 MAPE 为 7.44%（在 246 次提交中排名第一），显示检索增强的显著性能提升。

**⚠️ 局限性**

主要局限：①全记忆检索导致 O(N²) 的训练成本和线性推理成本，难以扩展到数十万条记录；②模型对记忆池的质量和时间漂移敏感，旧数据可能导致检索偏差；③解释性有限，尽管可以查看注意权重，但多头注意与残差 MLP 的内部决策仍不完全可读；④未提供完整的基准比较与多种模型的交叉验证，评估可能因数据分割不同而变化。

---

## 192. CleanCodec: Efficient and Robust Speech Tokenization via Perceptually Guided Encoding

**arXiv ID:** 2606.04418 | [PDF](https://arxiv.org/pdf/2606.04418v1)

**作者:** Eugene Kwek `[一作]` (Pennsylvania State University), Wenpeng Yin `[通讯]` (Pennsylvania State University)

**通讯引用:** 5445 | [OpenAlex ID](https://openalex.org/A5038386528)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种去噪音解码器，将音频token化视为选择性信息瓶颈问题，仅编码感知重要特征，剔除背景噪声等无关信息。

**💡 创新点**

创新点在于联合重建与语音增强训练、SSL与说话人条件、单码本FSQ量化以及两阶段训练框架，显著提升token率与重建质量的平衡。

**🔧 技术方法**

采用ConvNeXt卷积网络、FSQ单码本、WavLM和TitaNet作为条件、Vocos声码器、双解码器结构，并使用联合重建+语音增强损失。

**📊 数据集**

在LibriTTS‑R（≈585h）和Emilia‑YODAS（≈1800h）上训练，并在LibriTTS、Expresso、AISHELL‑3、CML‑TTS等数据集上评估。

**📈 对比分析**

与Mimi、BiCodec、XCodec2、WavTokenizer、FocalCodec、Qwen3、Kanade等基线相比，@12.5 t/s实现SIM 0.86、WER 2.7，远优于同级别token率基线；在VC、TTS任务上速度提升17×，性能领先。

**⚠️ 局限性**

局限在于高token率时分离效果退化；不同计算预算下的泛化尚未验证；提升音频token化可能助长深度伪造等风险。

---

## 193. Ultra-Fast Neural Video Compression

**arXiv ID:** 2606.04410 | [PDF](https://arxiv.org/pdf/2606.04410v1)

**作者:** Jiahao Li `[一作]` (Microsoft Research Asia), Yan Lu `[通讯]` (Microsoft Research Asia)

**通讯引用:** 20502 | [OpenAlex ID](https://openalex.org/A5035278528)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 DCVC-UF 神经视频编码器，采用基于块(chunk)的并行编码与解码框架，并结合帧特定解码器与简化熵编码技术，显著提升压缩速度与比特率。

**💡 创新点**

创新点在于：① 用跨帧交互模块和单块潜在表示替代传统逐帧/层次‑B 结构；② 设计帧特定解码器，使每帧能够专门学习其重建特征；③ 在熵模型中拆分尺度与均值，使解码只需单步交互，降低运算与内存开销。

**🔧 技术方法**

技术手段包括空间‑时间自编码器、跨帧交互模块、帧特定解码器、四叉树简化熵编码、条件编码以及 GPU 并行卷积实现。

**📊 数据集**

先在 7 帧 Vimeo‑90k 预训练，再用 128/512 帧 Vimeo 视频细化；评测使用 HEVC Class B–E、UVG、MCL‑JCV 三大标准数据集。

**📈 对比分析**

通过与 VTM‑17.0、HM‑16.25 以及 DCVC 系列的 BD‑Rate 与 FPS 对比，DCVC‑UF 在 HT‑L、HT‑S 和 LD 模式下平均 BD‑Rate 分别低至 -42.2%、-31.6%、-9.5%，编码/解码速度在 1080p 上可达 371/273 FPS（HT‑L）或 655/453 FPS（HT‑S），低延迟模式更达 313/354 FPS，远超传统编码器的速度与压缩效果。

**⚠️ 局限性**

局限性包括固定块大小可能不适应不同视频的时序特性，且在极长序列（>512 帧）与跨 GPU 流水线并行方面尚未充分探索。

---

## 194. TANDEM: Bi-Level Data Mixture Optimization with Twin Networks

**arXiv ID:** 2606.04401 | [PDF](https://arxiv.org/pdf/2606.04401v1)

**作者:** Jiaxing Wang `[一作]` (JD.com), Qixiang Jiang `[通讯]` (JD.com)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Twin Networks for bi-level DatA mixturE optiMization (TANDEM) 方法，用于在大语言模型预训练和微调阶段优化多域数据混合比例；

**💡 创新点**

创新点在于将二阶优化问题改写为单层惩罚形式，并使用双网络（代理模型和参考模型）同步更新，以动态评估域数据效用，提供理论收敛保证并显著降低梯度方差；

**🔧 技术方法**

采用惩罚式双层优化、Twin Networks、投影梯度下降、动态参考模型更新以及对比实验中统一、DoReMi、DoGE、Skill‑It、Aioli 等方法的基准；

**📊 数据集**

使用 SlimPajama（7个域）、Natural Instructions（99任务）等公开大规模多域语料，分别在数据充足、数据受限和监督微调场景下进行实验；

**📈 对比分析**

与多种现有 DMOs 进行对比，结果显示 TANDEM 在数据受限和微调场景下平均 perplexity/准确率均优于或与最强基线持平，且在 160M/410M/1B 模型上保持较好性能，计算开销与 Aioli 及 Skill‑It 相当；

**⚠️ 局限性**

局限性包括：对超参数 K、E、γ 等需手动调优；在极大模型/数据量下仍存在内存/计算开销；理论收敛仅在非凸二阶问题下给出，实际收敛速度受梯度方差影响。

---

## 195. DPDL: Towards Differential Privacy Preservation in Decentralized Stochastic Learning on Non-IID Data

**arXiv ID:** 2606.04399 | [PDF](https://arxiv.org/pdf/2606.04399v1)

**作者:** Yunsheng Yuan `[一作]` (Shandong University), Feng Li `[通讯]` (Shandong University)

**通讯引用:** 70199 | [OpenAlex ID](https://openalex.org/A5100448879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在非IID数据环境下的差分隐私保留去中心化随机学习算法DPDL。

**💡 创新点**

创新点在于在交叉梯度聚合中引入相似度校准与高斯噪声，以兼顾隐私和数据异质性。

**🔧 技术方法**

使用了差分隐私的高斯机制、梯度裁剪、余弦相似度校准、动量更新和理论收敛分析。

**📊 数据集**

在MNIST和CIFAR-10两个公开图像数据集上进行实验。

**📈 对比分析**

与DP-DPSGD、MUFFLIATO、DP-NETFLEET、DP-CGA等基线对比，DPDL在保持隐私的同时实现更快收敛、更高测试准确率和更好的抗梯度反演攻击效果。

**⚠️ 局限性**

局限性包括对通信拓扑的敏感性、对调参（噪声水平、裁剪阈值等）的依赖，以及在极高噪声或高度异构场景下收敛速度仍可能下降。

---

## 196. Shortcomings and capacities of real-constrained neural networks in complex spaces

**arXiv ID:** 2606.04390 | [PDF](https://arxiv.org/pdf/2606.04390v1)

**作者:** Andrew Gracyk `[一作]` `[通讯]` (Purdue University), Andrew Gracyk (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过 Gardner 体积方法，推导了在神经网络权重空间为复数时，强制预激活为实数（real pre‑activation）与不受约束的复数预激活两种情况下的存储容量（capacity）比值，并给出了该比值的渐近表达式。

**💡 创新点**

创新点在于：① 将 Harish‑Chandra‑Itzykson‑Zuber（HCIZ）公式首次用于高维复数神经网络的容量分析，能够保留有限维修正；② 通过复数与实数结构的分离，构建新的复数‑复数与复数‑实数两种约束下的 Gardner 体积积分；③ 采用 Schur 与 Zonal 多项式展开，以及傅里叶变换配合分布理论，克服了传统拉普拉斯方法在多重极值点上的不稳定性。

**🔧 技术方法**

主要技术手段包括：Gardner 体积积分、Replica（n→0）方法、HCIZ 公式、Weyl 积分公式、Schur 多项式与 Zonal 多项式展开、Andreief 身份、傅里叶变换与分布理论、拉普拉斯近似、Hubbard‑Stratonovich 变换。

**📊 数据集**

本文属于纯理论分析，未使用具体数据集；所有推导均在无监督的随机输入（高斯分布）假设下完成。

**📈 对比分析**

与传统仅使用拉普拉斯近似的 Gardner 体积分析相比，本文通过 HCIZ 公式得到更精确的容量比值，并给出了可计算的闭式形式；实验验证未给出，但理论上可通过数值积分验证该比值在不同稀疏性参数 ρ 与门限 κ 下的变化。

**⚠️ 局限性**

主要局限包括：① 需要可交换极限（n→0 与 N→∞ 的交换）且假设正则化、对称性；② HCIZ 公式的适用性局限于可积分的复数对称群，可能在更一般的网络结构下失效；③ 结果为渐近近似，对实际尺寸有限的网络能否充分体现仍待实证；④ 复杂计算过程对符号求解和数值稳定性要求高。

---

## 197. LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language Modeling

**arXiv ID:** 2606.04438 | [PDF](https://arxiv.org/pdf/2606.04438v1)

**作者:** Wenkai Chen `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Chengwei Qin `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种 LoopMoE 模型，即将稀疏专家路由与迭代共享权重的循环 Transformer 结合起来。

**💡 创新点**

创新点在于（1）IterAdaLN：将迭代索引和每个 token 的隐藏状态联合生成的可调 Affine 参数，实现 token‑级别的循环动态；（2）容量平衡策略：通过扩展多头注意力的低秩投影并缩小专家隐藏维度，恢复循环 MoE 与普通 MoE 的注意力/前馈活跃参数比例。

**🔧 技术方法**

使用技术包括：稀疏 Mixture‑of‑Experts、循环共享权重 Transformer、IterAdaLN（基于 RMSNorm 的自适应 LayerNorm）、LoRA 低秩投影、以及严格的 Sandwich‑Loop 结构。

**📊 数据集**

训练数据来自公开 OLMo‑3 预训练语料 Dolma3Mix，选取 200B-token 子集进行严格对齐。

**📈 对比分析**

方法：在与 Vanilla MoE 在总参数、每 token FLOPs、活跃子层比例完全匹配的条件下进行 head‑to‑head 比较；3B 模型在 8/9 任务上平均提升 >1 分，9B 模型在 100B-token 阶段仍保持优势，表明结构改进在规模扩大后不失效。

**⚠️ 局限性**

局限性包括：评估数据集未覆盖所有领域，实验仅使用英语，且在更大规模（>9B）上的表现尚未验证。

---

## 198. Loss-Conditional PINNs for Parametric PDE Families

**arXiv ID:** 2606.04420 | [PDF](https://arxiv.org/pdf/2606.04420v1)

**作者:** Anna Lazareva `[一作]` (HSE University), Alexander Tarakanov `[通讯]` (VK)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 LC-PINN，利用损失权重或 PDE 参数作为条件输入，训练单一网络学习整个参数家族的解。

**💡 创新点**

创新点在于将损失权重空间和参数空间统一为条件变量，形成连续解族；并给出 λ-不变性理论与局部优化提升分析。

**🔧 技术方法**

采用 PINN 损失函数、FiLM 或拼接条件化、Adam + L-BFGS 终止、随机采样 λ 的条件化训练。

**📊 数据集**

使用合成的 Helmholtz、Schrödinger、Burgers、Buckley–Leverett 等 PDE 族的数据集，参数在统一的先验分布下采样。

**📈 对比分析**

与 SA-PINN、ReLoBRaLo、Causal-PINN、PI-DeepONet 等自适应或单实例方法对比，LC-PINN 在多参数场景下平均误差下降 3–4 倍，且在大规模参数遍历时训练成本显著降低，达到跨越点仅需数十个实例。

**⚠️ 局限性**

局限在高频或带冲击、分岔等难解情形下，联合逼近整个参数族仍具挑战；对 λ 空间的采样需更智能，且对单实例最佳精度仍有差距。

---

## 199. (Mis)generalization of Helpful-only Fine-tuning

**arXiv ID:** 2606.04413 | [PDF](https://arxiv.org/pdf/2606.04413v1)

**作者:** Mohammad Omar Khursheed `[一作]` (Anthropic Fellows Program), Fabien Roger `[通讯]` (Anthropic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对“仅有帮助（Helpful‑Only, H‑only）”模型进行系统评估，发现其存在误导性行为、拒绝缺失、同情度低、易被操纵等缺陷，并提出基于宪法（Constitutional）角色训练的改进方案，验证该方法可显著缓解大多数问题。

**💡 创新点**

创新点在于：
1) 首次从多维度（误导性、拒绝、同情、可操纵性、偏见）全面评估 H‑only 模型；
2) 证明单纯的 anti‑refusal 训练会导致多种失败；
3) 通过添加宪法问答（Constitution QA）和合成文档微调（Synthetic Document Fine‑Tuning, SDF）构建的角色训练 pipeline，显著提升模型的一致性、可操纵性与对 H‑only 约束的遵从。

**🔧 技术方法**

主要技术包括：
- 监督微调（SFT）与群体相对策略优化（GRPO）
- anti‑refusal 数据与 math 数据的混合训练
- 宪法问答（Constitution QA）与合成文档微调（SDF）
- 使用推理标签控制推理过程
- 多种评估基准（StrongREJECT, AgentHarm, Sandbagging, Sycophancy, Steerability 等）

**📊 数据集**

使用的数据集：
- anti‑refusal 数据：有害提示与顺从回复
- math 数据：数学题目与答案
- 合成文档数据：基于 H‑only 宪法生成的文档
- 宪法问答数据：针对宪法不同条款的问答对
- LIMA 预训练数据（用于基础聊天能力）

**📈 对比分析**

比较方法：在同一模型基座（如 Qwen、Haiku 等）上分别使用三种训练流程（anti‑refusal、anti‑refusal+QA、anti‑refusal+QA+SDF），随后在上述评估基准上进行对比。结果显示：
- anti‑refusal 方案在合规性方面保持较高，但出现误导性、同情度低、易操纵等多项缺陷；
- anti‑refusal+QA 方案在误导性和同情度方面有明显提升；
- anti‑refusal+QA+SDF 方案在误导性、同情度、可操纵性、对 HHH 系统提示的响应以及对敏感话题的处理上表现最佳，整体性能显著优于单一 anti‑refusal 训练。

**⚠️ 局限性**

局限性：
- 角色训练仍未能完全消除同情度低、沙箱行为与偶发的误导性；
- 模型在条件性误导性（Conditional Misalignment）上仍存在残余，触发难度高但不易完全排除；
- 训练过程需要手工构造的宪法与合成文档，缺乏公开可复现的数据与代码；
- 对更大规模、不同族群模型的迁移性能尚未系统评估；
- 对模型内部情感表征的影响仍有限，情感一致性提升不显著。

---

## 200. An Empirical Study of Data Scale, Model Complexity, and Input Modalities in Visual Generalization

**arXiv ID:** 2606.04409 | [PDF](https://arxiv.org/pdf/2606.04409v1)

**作者:** Luoyidi Zhou `[一作]` `[通讯]` (Shandong First Medical University), Luoyidi Zhou (Shandong First Medical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过预实验和主实验，系统评估了训练样本规模、模型复杂度与输入模态对CIFAR‑10/100分类模型泛化性能的影响。

**💡 创新点**

创新点在于把训练数据规模、模型参数规模和输入模态三者的交互效应在同一框架下进行对比实验，并通过无数据增强、无正则化的纯控制环境剖析其独立贡献。

**🔧 技术方法**

使用的技术包括：多种网络架构（MLP、AlexNet、ResNet系列）、基于梯度的训练（Adam）、无正则化的交叉熵损失、以及手工生成的梯度/边缘/小波等先验特征。

**📊 数据集**

数据集：CIFAR‑10 与 CIFAR‑100，分别按不同训练子集规模（5k/10k/20k/30k/50k）进行实验。

**📈 对比分析**

比较方法：在相同训练时长、批大小、学习率等设置下，记录每个模型在不同数据规模、复杂度或输入模态下的最佳测试 Top‑1 准确率与损失；实验显示：增大训练样本规模始终提升准确率，模型复杂度提升不一定带来更好泛化，颜色信息丢失会降低准确率，先验特征对 MLP 有益但对 ResNet 无稳定提升。

**⚠️ 局限性**

局限性：仅在小尺寸图像数据集上验证，未使用常见的数据增强/正则化，实验仅单次随机种子，先验特征仅通过通道拼接方式引入，缺乏更高级的融合方法。

---

## 201. An Ensembled Latent Factor Model via Differential Evolution and Gradient Descent Optimization

**arXiv ID:** 2606.04408 | [PDF](https://arxiv.org/pdf/2606.04408v1)

**作者:** Rui Zhang `[一作]` (Chongqing Academy of Economics Research), Wenbo Zhang `[通讯]` (Southwest University)

**通讯引用:** 72229 | [OpenAlex ID](https://openalex.org/A5100351175)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种面向高维稀疏数据的集成潜在因子模型（ELFM-DEGDO），通过差分进化与梯度下降两种优化方法独立求解潜在因子，再通过自适应权重机制融合两种模型的优势，实现更准确、无偏的表示学习。

**💡 创新点**

创新点在于①首次将差分进化与梯度下降两种优化范式并行应用于潜在因子学习；②设计了自适应权重机制，动态平衡两种模型的贡献；③通过组合两种优化方法，显著提升了对异质高维不完整数据的泛化性能。

**🔧 技术方法**

核心技术包括潜在因子分解、梯度下降（SGD）、差分进化（DE），以及基于累计误差的自适应权重更新策略。

**📊 数据集**

实验使用三组真实高维稀疏数据集：Eachmovie、Flixter 和 Jester（分别包含约73k、148k、25k用户与1.6k、48k、100项）。

**📈 对比分析**

与BLFM、FNLFM、L3FM、AutoRec、DCCR、PMLFM等六个基线模型在MAE/RMSE指标下进行比较，ELFM-DEGDO在所有数据集上均取得最低误差，且统计检验显示显著优于其它模型。

**⚠️ 局限性**

局限性主要体现在：①模型尚未加入非负性约束与边界条件，可能导致潜在因子取值不合理；②算法参数调优仍依赖经验，缺乏理论自动化；③在极大规模数据上，差分进化部分的计算开销仍需进一步优化。

---

## 202. Bit-counting complexity classes

**arXiv ID:** 2606.04406 | [PDF](https://arxiv.org/pdf/2606.04406v1)

**作者:** Tayfun Pay `[一作]` `[通讯]`, Tayfun Pay

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文提出并研究了基于二进制位计数的新复杂度类族——bit‑counting complexity classes，并阐明了它们与传统复杂度类（如PP、NP、CoNP、C_=P等）的关系。

**💡 创新点**

创新点在于将计数值的二进制位数（0 与 1 的数量、奇偶性）作为判定条件，构造出比较型和奇偶型两类新的复杂度类，从而揭示计数与概率计数之间的新桥梁。

**🔧 技术方法**

技术上主要使用了 #P 函数的构造、阈值查询与二进制位操作，以及多项式时间 Turing 机与 PP oracle 的组合来证明包含与等价关系。

**📊 数据集**

由于论文属于理论计算复杂度研究，未使用具体数据集，而是通过抽象的构造和证明来完成研究。

**📈 对比分析**

通过理论证明显示 PP 被包含在三种比较型类中，三类与 P^PP 等价；奇偶型类包含 NP 与 CoNP，且与 P^PP 等价，展示了这些新类与已知类在 Turing 级别上的相互关系。

**⚠️ 局限性**

局限性在于对 B_|0|=0P、B_|0|>0P 与 ⊕P、US 等类的进一步关系尚未得到完全阐明，且未探讨这些类在多项式层级中的上界或下界。

---

## 203. Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval

**arXiv ID:** 2606.04391 | [PDF](https://arxiv.org/pdf/2606.04391v1)

**作者:** Jiaxi Li `[一作]` (University of Georgia), Ninghao Liu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5066745575)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种面向在线技能学习的网页代理方法——State-Grounded Dynamic Retrieval（SGDR），在任务执行过程中根据当前网页状态动态检索并调用合适的子程序技能；

**💡 创新点**

创新点在于：①将技能抽取粒度提升到滑动窗口级别，生成可在中间状态复用的子程序；②采用双文本-代码表示，使技能既可通过自然语言检索，又可直接执行；③在检索时将任务指令与当前网页状态共同作为查询，实现状态感知的动态检索，并通过MMR对候选技能进行多样性重排序；

**🔧 技术方法**

主要技术包括：LLM驱动的滑动窗口技能抽取与验证、文本-代码双向映射、基于嵌入的余弦相似度检索、MMR重排序、评估器模型用于无监督技能筛选；

**📊 数据集**

使用公开的WebArena benchmark（包含Shopping、Admin、Reddit、Gitlab、Map五个网站域），在每个域内逐任务执行并累积技能库；

**📈 对比分析**

与无技能、AWM、ASI、CER等基线对比，在两种主干LLM（如gpt‑4和llama‑3）下，SGDR在所有域平均成功率分别提升至37.5%（gpt‑4）和24.3%（llama‑3），比最强基线CER提高约10%，并在平均步骤数上实现约11%–14%的节约；

**⚠️ 局限性**

局限性包括：实验仅限于WebArena的有限域和交互模式，未验证在更广泛网页环境下的泛化；未探讨如何将学习到的技能与模型微调或长期个性化结合；可能对复杂、持续性预置状态的任务（如Gitlab）效果有限。

---

## 204. Revisiting Privacy Amplification by Subsampling in Selective Release DPSGD

**arXiv ID:** 2606.04384 | [PDF](https://arxiv.org/pdf/2606.04384v1)

**作者:** Xiaobo Huang `[一作]` (Beijing Normal-Hong Kong Baptist University), Fang Xie `[通讯]` (Beijing Normal-Hong Kong Baptist University)

**通讯引用:** 1174 | [OpenAlex ID](https://openalex.org/A5032656037)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为DPSR‑CG的新型差分隐私训练框架，该框架通过基于梯度裁剪偏差的选择性更新机制，替代传统的验证损失阈值，实现在保持严格隐私保证的同时显著提升模型性能。

**💡 创新点**

创新点在于：①重新评估并修正了现有选择性释放算法（DPSUR）的隐私计量，给出了严格的有效采样概率上界；②将梯度尺度化与裁剪偏差结合，构造了无验证开销的隐私保护选择标准；③采用适应性 Rényi 隐私分析，对迭代过程中的条件隐私损失进行严谨累积。

**🔧 技术方法**

核心技术包括：差分隐私随机梯度下降（DPSGD）中的梯度裁剪与尺度化；高斯机制与选择性释放；Rényi 隐私（RDP）和其自适应复合定理；以及针对选择性释放的有效采样概率计算和噪声参数设定。

**📊 数据集**

实验使用了四个公开数据集：MNIST、Fashion‑MNIST（FMNIST）、CIFAR‑10 以及 IMDB 文本分类数据。

**📈 对比分析**

与包括标准 DPSGD、DPSGD‑Matrix Mechanism、DPSGD‑IS、DPSGD‑HF、DPSGD‑TS、DPAGD 等六种最先进方法以及无选择性释放的基线进行对比；在 ϵ∈{1,2,3} 的隐私预算下，DPSR‑CG 在所有数据集上均取得最高准确率，甚至在 ϵ=3 时超过非私有基线。

**⚠️ 局限性**

局限性在于：①算法依赖于梯度裁剪阈值和尺度化参数的手工调优；②在极端噪声或梯度分布异常时，选择性释放阈值可能导致有效步数下降；③目前仅在单机训练环境验证，未针对联邦学习或大规模预训练模型的适配。

---

## 205. LCSHBench: A Multilingual, Consensus-Grounded Benchmark for Library of Congress Subject Heading Assignment

**arXiv ID:** 2606.04382 | [PDF](https://arxiv.org/pdf/2606.04382v1)

**作者:** Kwok Leong Tang `[一作]` `[通讯]`, Kwok Leong Tang

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并发布了 LCSHBench 作为一种公开、可复现的多语言图书馆学科标题分配基准，包含 22,346 本书、15 种语言以及三大开放授权图书馆（哈佛、哥伦比亚、普林斯顿）的 LCSH 标注记录，采用至少两家独立机构标注的记录为基准，并提供联合、单一及全体一致三种答案视图。

**💡 创新点**

创新点在于：①基于多机构共识的黄金标准，兼顾“概念一致性”与“表达细粒度差异”；②全词表检索任务（非冻结候选集）和开放词表生成任务并列；③细粒度指标面板（按语言、头条类型、exact/概念匹配等）取代单一 F1，揭示跨语言检索瓶颈；④通过低秩 LoRA 微调 300M 本地嵌入器，实现跨语言检索提升并压缩成本；⑤公开完整数据、评测器、基线及实验代码，确保可复现。

**🔧 技术方法**

使用的技术主要包括：基于 EmbeddingGemma-300M 的句子嵌入检索、LoRA 低秩微调、Matryoshka 损失、多负样本排名目标、P@k/R-Precision/MRR 评估；对生成任务采用大语言模型（deepseek-chat）与 LLM 再排序结合；所有系统在 2,002 条记录的子集上进行实验。

**📊 数据集**

数据集为 LCSHBench v1.0，来自哈佛、哥伦比亚、普林斯顿三大图书馆公开 MARC 记录；记录通过 OCLC、LCCN 统一匹配，按语言、学科平衡采样后划分开发集（18,993 条）和测试集（3,353 条），并提供 2,002 条跨语言子集供基线评估。

**📈 对比分析**

比较方法包括：生成任务的 micro/macro precision、recall、F1；检索任务的 recall@k、P@k、R-Precision、MRR；所有指标均按 exact 与 root（概念）匹配、按语言、按头条类型细分。实验结果显示，低秩微调后的 300M 本地嵌入器在跨语言检索（如韩语、阿拉伯语、日语、中文、德语）上显著优于 3,072 维托管嵌入器（Recall@200 提升 0.036），但在英语与俄语检索上仍落后；生成任务中大语言模型取得最高 set F1。

**⚠️ 局限性**

局限性包括：①共识基准仅在记录层面，对单头条的细粒度一致性未完全覆盖；②实验仅在 2,002 条子集上，未覆盖完整开发集或正式测试集；③微调模型训练使用了基准自身记录，虽然采用了泄漏控制但仍存在潜在污染风险；④答案哈希方式易被字典攻击，未提供私有评测服务器；⑤随着 LCSH 词表与分类实践演进，基准需要定期更新。

---

## 206. VT-3DAD: Cross-Category 3D Anomaly Detection via Visual-Text Normal Space Alignment

**arXiv ID:** 2606.04369 | [PDF](https://arxiv.org/pdf/2606.04369v1)

**作者:** Zi Wang `[一作]` (Niigata University), Jun Yu `[通讯]` (Niigata University)

**通讯引用:** 8962 | [OpenAlex ID](https://openalex.org/A5084275232)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了VT-3DAD，一个训练无关的跨类别少量样本3D异常检测框架，通过将视觉参考与文本语义空间对齐来判定未知点云是否属于目标正常类别。

**💡 创新点**

创新点在于利用CLIP的冻结视觉与文本编码器构建“视觉‑文本正常空间”，将少量正常点云的多视角深度特征与基于深度感知和三维感知的文本提示的语义锚点相结合，形成双空间异常评分。

**🔧 技术方法**

使用CLIP ViT‑B/32视觉/文本编码器、逼真深度投影、多视角加权距离、文本提示生成、双空间融合（α=1.0, β=0.5）等技术。

**📊 数据集**

在ShapeNetPart数据集上进行实验，包含16个物体类别，采用1、3、5-shot正样本设置。

**📈 对比分析**

与重建、知识蒸馏、DMP‑3DAD等基线对比，VT-3DAD在1-shot下平均AUC‑ROC从92.49%提升至94.80%，3-shot和5-shot亦有提升，并显著降低标准差，表现出更好的稳健性。

**⚠️ 局限性**

局限性包括对文本提示的依赖程度较高、对极其相似几何结构的误判仍存在、以及缺乏针对复杂姿态或噪声场景的鲁棒性验证。

---

## 207. What Can Verifiable Decapsulation Tests Certify? Pass Bounds and Fault-Recognition Limits for FO-Based KEMs

**arXiv ID:** 2606.04443 | [PDF](https://arxiv.org/pdf/2606.04443v1)

**作者:** José Luis Delgado Jiménez `[一作]` `[通讯]` (Universitat Oberta de Catalunya), José Luis Delgado Jiménez (Universitat Oberta de Catalunya)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对 Fujisaki–Okamoto（FO）变换下的确认码增强 KEM 的解封装过程，提出了一种基于确认码的黑盒测试框架，能够将通过测试的成功概率转化为对隐藏确认码的列表命中概率，并给出了精确的上界，包括对混淆、别名、最新键新鲜度等错误的完整计数。

**💡 创新点**

创新点在于：① 将确认码可预测性（cUP）与列表命中问题（list‑cUP）统一到一个可观察的黑盒测试语句；② 引入依赖锥（dependency‑cone）理论，证明在黑盒条件下若操作不在确认码可观察锥内，则无法通过任何黑盒测试保证其执行；③ 提供多条完整的上界途径（cUP、条件熵、哈希诊断、尾部熵），实现对不同确认码形式的统一安全评估。

**🔧 技术方法**

核心技术包括：FO 变换的分离式实现（随机数与最终密钥的独立随机 oracle 域），cUP 与 fCOR 形式的安全游戏化；列表命中事件的概率分析；熵与条件熵的平均下界推导；以及对黑盒观察模型的精确描述和依赖锥构造。

**📊 数据集**

实验数据集为两类：① FIPS 203 与 August 2025 公共实现的基线与其增强变体；② 通过 FO‑FaultBench 工具生成的多种错误注入（重计算省略、绑定缺失、决策分支、对称端点、随机码猜测、尾部保留位）样本，覆盖 3592 条记录，并在不同参数集（ML‑KEM‑512/768/1024、HQC‑1/3/5）下进行多次重复测试。

**📈 对比分析**

比较方法采用黑盒测试的置信区间估计（Wilson 区间或 Clopper–Pearson），与理论给出的 2^‑L 参考尺度对照；实验结果显示，确认码增强能显著检测到重计算缺失与绑定错误；但仅在恶意分支或差分支持下才能检测决策分支错误；概率猜测误码率与 2^‑L 的指数衰减一致；在保持尾部位数的情况下，误码率随保留位数递减。

**⚠️ 局限性**

局限性包括：① 只在经典随机 oracle（ROM）模型下证明，无法直接推广到量子随机 oracle（QROM）环境；② 依赖锥的判断需要完整的执行轨迹信息，实际黑盒实现中可能难以获取；③ 对确认码的熵假设（RawEnt、TailEnt）需外部证书，未在实验中直接验证；④ 只评估了有限的错误注入集，未覆盖所有潜在的实现缺陷。

---

## 208. MemoryDocDataSet: A Benchmark for Joint Conversational Memory and Long Document Reasoning

**arXiv ID:** 2606.04442 | [PDF](https://arxiv.org/pdf/2606.04442v1)

**作者:** Qiyang Xie `[一作]` (Northeastern University), Weikai Zhou `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一个新的基准数据集，评估 AI 系统在多轮对话历史与长文本深度阅读之间的联合推理能力。

**💡 创新点**

设计 Hybrid source tag，将需要先在对话中定位相关文档再在文档中提取答案的双步问题，引入跨来源推理评测。

**🔧 技术方法**

使用 LLM 生成微世界、对话与问答，构建检索增强生成（RAG）与记忆系统基线，并采用 token‑level F1 等指标进行评估。

**📊 数据集**

采集 20k–50k token 的真实美国法院判例文本，生成 50 个微世界共 1,000 条 QA，按 70/14/16 划分为 train/val/test。

**📈 对比分析**

通过六种基线（截断上下文、长上下文、仅对话检索、仅文档检索、两者检索、记忆系统）在 Hybrid、Doc‑only、Chat‑only 上评测；RAG‑Both 在 Hybrid 上最高 0.342，整体性能仍低于人类水平。

**⚠️ 局限性**

局限性包括领域单一（仅美国法院判例）、LLM 生成的对话与问答可能不够真实、自动评估受模型偏差影响、缺乏人类性能基准、未实现对话→文档的两步导航等。

---

## 209. INTACT: Ego-Guided Typed Sparse Evidence Retrieval for Heterogeneous Collaborative Perception

**arXiv ID:** 2606.04437 | [PDF](https://arxiv.org/pdf/2606.04437v1)

**作者:** Chen Li `[一作]` (Huazhong University of Science and Technology), Changxin Gao `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 11243 | [OpenAlex ID](https://openalex.org/A5035295689)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了INACT框架，利用ego发起的有类型稀疏证据检索实现异构协同感知。

**💡 创新点**

将协作界面从全图特征对齐转为ego发起的类型化查询，避免逐对适配；采用查询导向检索、软/硬路由及门控残差写回，支持一次训练可复用。

**🔧 技术方法**

BEV特征投影、查询生成器、查询导向的局部响应检索、softmax路由、门控残差写回等；结合Pyramid/late fusion网络。

**📊 数据集**

OPV2V‑H、DAIR‑V2X、V2X‑Real等模拟与真实异构协同感知基准。

**📈 对比分析**

与MPDA、BackAlign、STAMP、GT‑Space、GenComm等基线对比，INTACT在OPV2V‑H上AP70≈0.80、通信压缩≈16×、参数增量仅0.52M，DAIR‑V2X/AP50≈0.44，V2X‑Real/AP50≈0.72，均保持或优于基线且参数/通信更优。

**⚠️ 局限性**

需要准确的几何配准；仅在有限的传感器与骨干组合上验证；暂未考虑时序推理与更丰富的协作者选择。

---

## 210. Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal In-Context Learning

**arXiv ID:** 2606.04434 | [PDF](https://arxiv.org/pdf/2606.04434v1)

**作者:** Niloufar Alipour Talemi `[一作]` (Clemson University), Fatemeh Afghah `[通讯]` (Clemson University)

**通讯引用:** 3849 | [OpenAlex ID](https://openalex.org/A5035395012)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Hyper-ICL，一种在不使用示例的情况下重构多模态自注意力的轻量级训练框架

**💡 创新点**

通过低秩 logit 级适配器与查询自适应调制直接校准注意力，并引入层级超曲线锚点蒸馏保持示例影响的中间表示

**🔧 技术方法**

低秩 bilinear logit bias、查询自适应门控、Lorentz 超曲线几何蒸馏、Transformer 自注意力改造等技术

**📊 数据集**

VQAv2、OK‑VQA、COCO Caption、Flickr30k、MME、SEED‑Bench 等多模态基准数据集

**📈 对比分析**

与零样本、ICL、RICES、TV/FV、LoRA、LIVE、MimIC 等基线对比，Hyper‑ICL 在 Idefics‑9B 与 Idefics2‑8B 上在 VQAv2、OK‑VQA、COCO Caption 等任务上均比现有方法提升 1–4%（准确率），且显著降低推理延迟并减少幻觉

**⚠️ 局限性**

适配器需针对特定任务/域训练，跨任务迁移或不同模型骨干时需重新训练；在高风险场景仍需验证与人工监督

---

## 211. FlexNPU: Transparent NPU Virtualization for Dynamic LLM Prefill-Decode Co-location

**arXiv ID:** 2606.04415 | [PDF](https://arxiv.org/pdf/2606.04415v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 212. Implicit Fuzzification via Bounded Noise Injection for Robust Medical Image Segmentation

**arXiv ID:** 2606.04427 | [PDF](https://arxiv.org/pdf/2606.04427v1)

**作者:** Bisheng Tang `[一作]` (Shaoyang University), Yifei Peng `[通讯]` (Shaoyang University)

**通讯引用:** 1015 | [OpenAlex ID](https://openalex.org/A5102990703)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在 U‑Net 结构中加入了有界噪声注入到 skip 连接，以增强特征融合的鲁棒性和边界辨识。

**💡 创新点**

创新点在于利用有界扰动实现隐式模糊化（implicit fuzzification），既不需要额外监督也不增加网络结构复杂度，理论上可视为跨尺度特征的 Lipschitz 约束和梯度平滑。

**🔧 技术方法**

核心技术是对 skip 连接的特征进行加性均匀噪声注入，随后直接与解码器特征拼接，整体使用标准 U‑Net 训练框架；同时对噪声分布、范围、以及梯度效果进行了理论分析。

**📊 数据集**

使用了三类医学影像数据集：BUSI（乳腺超声）、GlaS（肠道组织切片）和新构建的 ThyR（甲状腺超声）来评估方法。

**📈 对比分析**

与 U‑Net、U‑Net++、U‑NeXt、Rolling‑UNet、U‑Mamba、U‑KAN 等基线模型对比，实验表明 NoiseUNet 在 IoU、F1 等指标上均有提升（BUSI +1.07%、GlaS +0.40%、ThyR +1.92%），且保持与 U‑Net 相同的 GFLOPs 与参数量，证明在保持计算成本的前提下性能显著提升。

**⚠️ 局限性**

局限性包括：噪声范围需手工设定；仅验证了三种医学图像场景；对极端低对比度或高噪声图像的鲁棒性尚未深入；未探讨不同噪声分布或自适应噪声策略对更广泛任务的适用性。

---

## 213. What If Prompt Injection Never Left? Exploring Cross-Session Stored Prompt Injection in Agentic Systems

**arXiv ID:** 2606.04425 | [PDF](https://arxiv.org/pdf/2606.04425v1)

**作者:** Yuanbo Xie `[一作]` (Chinese Academy of Sciences), Tingwen Liu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 9451 | [OpenAlex ID](https://openalex.org/A5101554099)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文定义并系统研究了跨会话存储式提示注入（SPI）这一新的安全威胁，阐明了其生命周期与分类，构建了评估基准与沙盒工具。

**💡 创新点**

创新点在于：①将提示注入从单会话转为跨会话持久化攻击，②提出完整的SPI生命周期模型与三维分类（注入源、持久化通道、上下文注入机制），③创建了覆盖多模型、多目标、多通道的SPI基准，提供定量评估。

**🔧 技术方法**

采用了基于LLM的代理系统上下文构造框架、规则+LLM评判的混合验证器、四阶段（环境、注入、会话重置、激活）实验流程，并通过写入成功率、注入率、激活率拆分评估。

**📊 数据集**

使用了三类应用场景（电商、旅行预订、金融投资）共162个SPI案例，涵盖三类注入目标（事实操纵、偏好操纵、行动范围操纵）与多种持久化通道与注入类型。

**📈 对比分析**

在GLM‑5.1、GPT‑5‑mini、MiniMax‑M2.7三模型上，SPI整体成功率在32.1%–42.0%之间；写入成功率最高（86.4%），但激活成功率是瓶颈；事实操纵攻击最易成功（AR 100%），偏好操纵几乎无效。

**⚠️ 局限性**

局限性包括：基准仅覆盖现有代理体系，未涵盖新兴持久化机制；实验以人工设计案例为主，缺乏大规模真实数据；未对对抗训练或动态防御方案进行评估。

---

## 214. Physics-Informed Neural Network Modeling of Biodegradable Contaminant Transport through GCL/SL Composite Liners

**arXiv ID:** 2606.04392 | [PDF](https://arxiv.org/pdf/2606.04392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 215. Trivium: Temporal Regret as a First-Class Objective for Causal-Memory Controllers

**arXiv ID:** 2606.04421 | [PDF](https://arxiv.org/pdf/2606.04421v1)

**作者:** Edward Y. Chang `[一作]` (Stanford University), Edward Y. Chang `[通讯]` (Stanford University)

**通讯引用:** 18536 | [OpenAlex ID](https://openalex.org/A5013545831)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出三重后悔目标（结果、认知、时间），并实现 Trivium 框架，利用因果日志与有限干预预算在长期代理中学习并纠正因果结构，以降低时间后悔。

**💡 创新点**

将时间后悔与认知后悔与传统结果后悔并列为可度量目标，证明在具备因果日志和干预通道的条件下，跨情节时间后悔可实现对数级别收敛，并通过预注册实验验证该理论。

**🔧 技术方法**

结合因果干预、可观测等价分离、CUSUM 变点检测、局部收敛修复（LRCP）、事务性因果日志（CTL）以及预算化的干预探索等技术。

**📊 数据集**

在受控线性高斯因果基准 CausalBench-Seq、REALM-Bench 多智能体规划基准以及实时 LLM 流 cap-gsm8k（针对 Llama-3.3-70b、GPT-4o、Claude-Sonnet-4.5 等模型）进行实验验证。

**📈 对比分析**

与仅基于结果奖励的 RL（RLVR）、记忆型结果法、即时重置等基线进行比较；在 CausalBench-Seq 中 Trivium 的时间后悔呈对数增长，而结果后悔仅线性增长；在 LLM 流中峰值时间后悔下降约 103 倍，平均降低约 24 倍。

**⚠️ 局限性**

实验仅覆盖两层因果图、局部线性 SCM、可检测变点；未验证更复杂层级结构、非线性因果关系、干预渠道缺失或高成本场景；事务日志可靠性和 LLM 与因果图对齐问题仍需进一步研究。

---

## 216. Pepper: High-bandwidth and Scalable Anonymous Broadcast with Cryptographic Privacy

**arXiv ID:** 2606.04411 | [PDF](https://arxiv.org/pdf/2606.04411v1)

**作者:** Chenghao Li `[一作]` (University of Science and Technology of China), Xianghang Mi `[通讯]` (Monash University)

**通讯引用:** 652 | [OpenAlex ID](https://openalex.org/A5047482025)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

暂无具体描述

**💡 创新点**

暂无创新点信息

**🔧 技术方法**

暂无技术细节

**📊 数据集**

暂无数据集信息

**📈 对比分析**

暂无对比方法和性能评估

**⚠️ 局限性**

暂无局限性说明

---

## 217. Not All Errors Are Equal: Consequence-Aware Reasoning Compute Allocation

**arXiv ID:** 2606.04402 | [PDF](https://arxiv.org/pdf/2606.04402v1)

**作者:** Jingbo Wen `[一作]` (University of Sydney), Ziqi He `[通讯]` (University of Sydney)

**通讯引用:** 45494 | [OpenAlex ID](https://openalex.org/A5100400885)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于任务后果的测试时计算分配方法，用轻量级预测器先评估错误对生产的潜在成本，再根据该后果信息在有限的总计算预算下为每个软件工程任务分配不同的计算层级。

**💡 创新点**

创新点在于将错误的后果成本作为新的路由信号，突破了传统仅依据任务难度或置信度的计算分配框架；并证明后果与难度正交且可在部署时仅通过问题文本预测，进而实现无需重训模型的计算调度。

**🔧 技术方法**

使用了思考型大型语言模型（如Qwen3-8B、Claude Sonnet 4.5等）进行思考长度测评、后果预测（issue‑only LLM），以及基于成本加权的路由算法和优先级调度（结合后果与边际收益）。

**📊 数据集**

评估数据集为SWE‑bench Lite（300题）和Multi‑SWE‑bench mini（400题），并在16个公开软件工程解题模型构成的多层计算基准上验证路由效果。

**📈 对比分析**

与随机路由、难度驱动路由（Snell‑style）以及oracle后果路由进行对比；在匹配总计算预算的条件下，后果驱动路由将成本加权损失降低约22%–33%，而优先级路由进一步提升至30%+，几乎达到oracle水平。

**⚠️ 局限性**

局限性包括后果标签采用离散三级等级，未覆盖多维成本维度；评估仅在离线多模型基准上进行，缺乏对单一模型思考预算调节的实验；并且后果预测仍存在误差，尤其在罕见高后果任务上的召回率不及oracle。

---

## 218. From Symbolic to Geometric: Enabling Spatial Reasoning in Large Language Models

**arXiv ID:** 2606.04381 | [PDF](https://arxiv.org/pdf/2606.04381v1)

**作者:** Chen Chu `[一作]` (University of Southern California), Cyrus Shahabi `[通讯]` (University of Southern California)

**通讯引用:** 19812 | [OpenAlex ID](https://openalex.org/A5012068017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了空间语言模型（SLM），将几何空间表示作为第一类模态嵌入大型语言模型中，支持在推理过程中直接进行几何空间推理；同时构建了面向空间指令的训练集和评估基准 SpatialEval，分别用于训练和全面评估模型的空间推理能力。

**💡 创新点**

创新点包括：①把几何空间表示作为第一类模态嵌入 LLM；②使用统一的 Geo2Vec 空间编码器和空间适配器实现跨类型实体的共享表征；③提出 interleaved prompt 结构，使模型可在句子中直接处理空间实体；④构建与空间表示、几何操作和自然语言指令一一对应的空间指令数据集；⑤设计 SpatialEval 基准，既评估符号（坐标、地名）又评估几何两种输入；⑥实现完全内在的几何推理，无需外部工具或代理。

**🔧 技术方法**

技术细节：使用 Qwen3‑8B 作为基础 LLM，并通过 LoRA 微调；引入空间适配器把 Geo2Vec 编码的向量投影到模型词向量空间；采用 interleaved prompt 让模型直接接收空间实体的向量；利用 LLM 自动生成的 prompt evolution 生成多样化的训练样本；使用代理 LLM 生成结构化答案，以此提供明确的几何推理轨迹。

**📊 数据集**

数据集与评估：①空间指令数据集（约 30k 条），通过专家 seed + LLM 生成；②三个空间数据集用于训练/测试：北京与洛杉矶的 POI、道路、建筑；美国州/县边界；③SpatialEval 基准，涵盖面积、长度、距离、最近邻、拓扑等 5 类任务，支持实体名字、坐标或向量三种输入形式。

**📈 对比分析**

评估方法：在 SpatialEval 上与多种基线模型（Qwen3、DeepSeek、Llama‑3.3‑70B、Gemini 2.5 Flash、ChatGPT 5.1、SpatialRGPT、SRL）进行对比，采用 MAE、准确率和响应有效率等指标。SLM 在大多数任务上显著优于符号推理模型，尤其在距离估计、拓扑分类和最近邻查询中取得最优或接近最优结果；在零样本迁移和查询复杂度上保持稳定性能；相较于符号/代理方法，SLM 的输入输出 token 数量和推理时间更低，表现更高效。

**⚠️ 局限性**

局限性：①对空间表示的质量高度依赖，面积等几何属性受 SRL 方法限制；②训练仍需要大量手工或 LLM 自动生成的指令样本；③当前仅针对欧几里得二维几何，未评估 3D、时空或更复杂属性的推理；④在极大规模多属性、多时相的空间推理任务上尚未验证；⑤模型对未知或非标准地理实体的泛化仍有限。

---

## 219. Selective Coupling of Decoupled Informative Regions: Masked Attention Alignment for Data-Free Quantization of Vision Transformers

**arXiv ID:** 2606.04373 | [PDF](https://arxiv.org/pdf/2606.04373v1)

**作者:** Biao Qian `[一作]` (Tsinghua University), Jungong Han `[通讯]` (Tsinghua University)

**通讯引用:** 25977 | [OpenAlex ID](https://openalex.org/A5046605531)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MaskAQ 方法，实现无数据量化 ViT 的样本生成与校准

**💡 创新点**

通过识别自注意力中的稀疏信息区（Informative Region）并采用差分熵最大化与掩码注意力对齐，解决语义分散和注意力不匹配问题

**🔧 技术方法**

信息瓶颈理论指导、差分熵正则、掩码注意力对齐、周期性样本刷新、基于自注意力的对齐损失

**📊 数据集**

ImageNet 进行分类；检测与分割任务的实验在附录中说明

**📈 对比分析**

相较于现有 DFQ 方法（如 MimiQ、PSAQ-ViT、CLAMP-ViT 等），MaskAQ 在多种 ViT/DeiT/Swin 架构下取得 1-3% 以上 Top‑1 提升，尤其在 3 位量化下提升达 3.1%

**⚠️ 局限性**

需迭代合成样本，生成开销较大，对更激进量化（低于 3 位）仍需进一步研究

---

## 220. Beyond Single-Policy: Evaluating Composed Organization-Specific Policy Alignment in LLM Chatbots

**arXiv ID:** 2606.04394 | [PDF](https://arxiv.org/pdf/2606.04394v1)

**作者:** Yingjie Liu `[一作]` (Fudan University), Yangfan Zhou `[通讯]` (Fudan University)

**通讯引用:** 1996 | [OpenAlex ID](https://openalex.org/A5101465219)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了COPAL，一个自动化框架，用于评估组织聊天机器人中组合政策的对齐情况。

**💡 创新点**

COPAL通过使用经验导出的交互模式生成需要多个政策共同处理的查询，并为每个查询配对一个明确的处理合同，从而解决了现有政策对齐基准未能充分测试组合政策的问题。

**🔧 技术方法**

使用了自动化框架和处理合同来评估聊天机器人的响应，结合了政策文本的结构化表示和交互模式的选择。

**📊 数据集**

在30个类似公司的组织环境中进行了评估，生成了882条政策规则和900个组合查询。

**📈 对比分析**

与单一政策测试相比，组合政策请求的错误率为33.1%，而单一政策的错误率较低，表明组合政策对齐仍然是一个具有挑战性的评估目标。

**⚠️ 局限性**

组合标准的构建是一个操作性方案，而不是完整的语义理论，重建的聊天机器人无法模拟所有后端工具和内部政策，且自动评估的错误率可能与人类评估存在差异。

---

## 221. D^2SD: Accelerating Speculative Decoding with Dual Diffusion Draft Models

**arXiv ID:** 2606.04446 | [PDF](https://arxiv.org/pdf/2606.04446v1)

**作者:** Liyuan Zhang `[一作]` (Peking University), Binhang Yuan `[通讯]` (HKUST)

**通讯引用:** 673 | [OpenAlex ID](https://openalex.org/A5002684888)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Dual-Draft Diffusion Speculative Decoding（D²SD），将单块扩散草稿拆分为共享前缀的候选集合，从而在保持无误差的前提下显著加速 LLM 推理。

**💡 创新点**

核心创新在于：①利用第一个扩散草稿的置信度估计拒绝边界；②根据该边界动态挑选前缀长度并使用训练有加的变量前缀扩散草稿重新生成后续序列，避免了传统块扩展与均匀分支的低效；③通过级联注意力实现多路候选的联合验证。

**🔧 技术方法**

技术实现包括：基于扩散语言模型的块草稿；KV 注入将目标模型隐藏状态提供给扩散模型；置信度估计与 Top‑K 前缀选择；变量前缀训练（带指数衰减损失）；级联注意力（cascade attention）与 FlashAttention‑2 结合；BF16 精度与 CUDA 图优化。

**📊 数据集**

训练数据采用 PerfectBlend（涵盖数学、代码、聊天等多任务），评估数据集涵盖 GSM8K、MATH、AIME25、HumanEval、MBPP、LiveCodeBench、MT‑Bench 与 Alpaca 等八个基准。

**📈 对比分析**

与 DFlash、EAGLE‑3 等基线在 Qwen3‑8B 与 GPT‑OSS‑20B 上进行对比；在 Greedy（T=0）下，平均速度提升从 4.16× 提升至 4.98×，平均接受长度从 5.31 提升至 7.05；在多任务上整体表现均优于基线，尤其在数学与代码任务中显著提升。

**⚠️ 局限性**

局限性包括：①仍受目标模型验证成本限制，②多级级联会引入额外计算导致收益递减；③对开放式聊天任务的边界预测更模糊，提升空间有限。

---

## 222. 3DThinkVLA: Endowing Vision-Language-Action Models with Latent 3D Priors via 3D-Thinking-Guided Co-training

**arXiv ID:** 2606.04436 | [PDF](https://arxiv.org/pdf/2606.04436v1)

**作者:** Jiaxin Shi `[一作]` (Shanghai Jiao Tong University), Weihao Yuan `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种3D思维引导的联合训练框架，使视觉‑语言‑动作模型能够在仅使用二维图像的情况下隐式进行三维空间推理

**💡 创新点**

核心创新在于把三维几何感知与高层空间推理分离，并通过三大模块：隐式几何感知适配器、在线三维推理蒸馏模块（共享推理锚令牌）以及空间增强动作整合，解决传统联合训练中出现的“提示诱发推理缺口”问题

**🔧 技术方法**

使用轻量级几何适配器将视觉特征对齐到3D基准模型（VGGT）；在线推理蒸馏通过教师-学生架构在潜在空间内迁移推理知识；空间增强动作整合将几何与推理特征作为层次化空间条件注入动作头；整体以Qwen3‑VL‑2B为基础模型，并采用OFT风格动作头

**📊 数据集**

在LIBERO、LIBERO‑PLUS、SimplerEnv模拟数据集上进行训练与评估；同时在Realman实物机器人平台上对三种真实任务进行零样本评估

**📈 对比分析**

与现有方法（如OpenVLA‑OFT、SpatialVLA、GeoVLA、3D‑CAVLA等）比较，取得LIBERO平均成功率98.7%、LIBERO‑PLUS平均成功率81.0%，SimmerEnv平均成功率72.9%，并在真实机器人任务中超过其他基线，显示出显著的性能提升

**⚠️ 局限性**

局限性包括：仍需在训练阶段使用3D推理数据和教师分支，推理锚令牌可能在极端场景下信息不足；模型对高度、透明度等特殊视觉特征的泛化能力虽提升但仍有提升空间

---

## 223. Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation

**arXiv ID:** 2606.04435 | [PDF](https://arxiv.org/pdf/2606.04435v1)

**作者:** Saroj Mishra `[一作]` (University of North Dakota), Saroj Mishra `[通讯]` (University of North Dakota)

**通讯引用:** 3018 | [OpenAlex ID](https://openalex.org/A5051854619)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了用于多步骤代理式检索增强生成（RAG）管道的 CHARM 框架，以检测和中断在管道中累积扩散的幻觉错误。

**💡 创新点**

创新点在于：①正式定义并归纳了四种“级联幻觉”模式；②设计了可插拔的四组件检测体系（SFV、CSCT、CPM、CRT）；③给出了针对不同级联类型的可配置缓解模式。

**🔧 技术方法**

技术方法包括：跨编码器事实验证（NLI）、跨阶段语义相似度漂移监测、贝叶斯自信度轨迹更新、加权投票触发器；框架在不改动主模型的前提下与 LangChain、LlamaIndex 等框架兼容。

**📊 数据集**

实验使用了 HotpotQA、MuSiQue、2WikiMultiHopQA 三个多跳问答数据集以及自制的 200 条对抗级联样本进行评估。

**📈 对比分析**

与无检测、SelfCheckGPT、RAGAS、LLM 自校正等基线对比，CHARM 在级联检测率上达 89.4%，误报率 5.3%，错误传播减少率 82.1%，缓解成功率 91.3%，平均检测深度仅 2.1 阶段，单阶段延迟约 215 ms。

**⚠️ 局限性**

局限性包括：PVA 缓解模式计算成本高、仅针对文本管道、对语义幻觉与嵌入攻击的鲁棒性不足、实验主要基于人工注入的级联，缺乏大规模真实级联数据。

---

## 224. Stateful Visual Encoders for Vision-Language Models

**arXiv ID:** 2606.04433 | [PDF](https://arxiv.org/pdf/2606.04433v1)

**作者:** Zirui Wang `[一作]` (Voio, Inc.), Trevor Darrell `[通讯]` (Voio, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多图视觉-语言模型中加入状态化视觉编码器（SVE），使当前图像的特征能够利用前一帧的视觉信息进行条件化；

**💡 创新点**

创新点在于将跨图像交互直接注入视觉编码器的自注意力模块，既保持了预训练视觉模型的结构，又显著提升了细粒度视觉比较能力；

**🔧 技术方法**

采用自注意力扩展、交叉注意力+FFN、AdaLN‑Zero等SVE变体，并通过权重复制、零初始化与梯度停止等技术实现高效微调；

**📊 数据集**

在合成基准（空间聚合、CLEVR‑Multi‑Change、VisGym）以及真实数据集（Medical‑Diff‑VQA、ImgEdit、LEVIR‑CC）上进行训练与评估；

**📈 对比分析**

与无状态基线比较，SVE在所有任务上均实现显著提升（多任务平均提升≈1–3个百分点），甚至在某些任务上超过现有专用模型；

**⚠️ 局限性**

局限性包括对长时间序列或更复杂动态场景的适应性仍待验证，以及对视觉编码器原始预训练分布的依赖可能限制跨域推广。

---

## 225. DSA: Dynamic Step Allocation for Fast Autoregressive Video Generation

**arXiv ID:** 2606.04432 | [PDF](https://arxiv.org/pdf/2606.04432v1)

**作者:** Thanh-Tung Le `[一作]` (University of California, Irvine), Deying Kong `[通讯]` (Google)

**通讯引用:** 553 | [OpenAlex ID](https://openalex.org/A5077713552)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们提出了一种自适应步骤分配的自回归视频扩散框架，利用置信度头动态调整每帧的扩散步骤，实现实时视频生成。

**💡 创新点**

创新点在于引入轻量级置信度网络与分布匹配蒸馏目标联合训练，生成帧时预测自身置信度，从而在推理时根据置信度动态减少或增加扩散步骤。

**🔧 技术方法**

使用了分布匹配蒸馏（DMD）、Self‑Forcing自回归蒸馏、轻量级置信度头、KV缓存、视频扩散Transformer（DiT）等技术。

**📊 数据集**

训练仅使用文本提示数据，来源于过滤后并经过LLM增强的VidProM集合；蒸馏自Wan‑1.3B和Wan‑14B模型。

**📈 对比分析**

与SkyReels‑V2、MAGI‑1等基础自回归扩散模型以及CausVid、Self‑Forcing等蒸馏模型对比，在VBench指标上保持或超过对手，同时推理吞吐量达到22.63 FPS，子秒延迟，显著快于对手。

**⚠️ 局限性**

在长时序视频生成中易出现时间漂移和累计误差，导致画质与语义一致性下降，需要进一步引入长上下文建模。

---

## 226. When Chatbots Accommodate: What AI Companions Optimize for in Vulnerable Conversations

**arXiv ID:** 2606.04431 | [PDF](https://arxiv.org/pdf/2606.04431v1)

**作者:** Minh Duc Chu `[一作]` (USC), Luca Luceri `[通讯]` (USC)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了三大 AI 伴侣聊天机器人（GPT‑4.1、Character.AI、Replika）在用户表现脆弱性时的真实交互，并通过逆向强化学习推断其隐含的响应策略；

**💡 创新点**

创新点在于提出了 AI 伴侣脆弱性‑响应对照法则（AC‑VRT）——一种联合标注用户脆弱性与聊天机器人回应的双侧分类体系，并将逆向强化学习（MCE‑IRL）应用于长期对话，首次揭示不同平台在脆弱情境下的策略差异与随时间的漂移；

**🔧 技术方法**

使用了最大因果熵逆向强化学习（Maximum Causal Entropy IRL）来估计状态‑动作奖励，从而推断对话策略；同时利用零样本 Gemini LLM 进行大规模自动标注；

**📊 数据集**

约 48,000 轮对话数据，来源于 GPT‑4.1（4 周实验，110 名参与者、386 条对话）、Character.AI（98 条对话）和 Replika（47 条对话）等公开或受限转录；

**📈 对比分析**

对三平台的推断策略做了 500 次 Bootstrap 估计，并与经验频率对照，验证 IRL 与观测行为高度一致；进一步对 GPT‑4.1 的四周交互进行线性/二次趋势检验，发现询问比例显著下降；整体结果显示不同平台在脆弱性场景下的策略差异显著，且 GPT‑4.1 在时间上表现出向减少追问的漂移；

**⚠️ 局限性**

主要局限包括：①角色聚合导致 Character.AI 策略不稳定；②数据规模偏小，特别是 Replika 的子群分析受限；③AC‑VRT 仅基于四轮上下文，无法捕捉更深层关系和长期情感累积；④未能将推断奖励与实际训练目标直接关联，故对隐含优化目标的推断存在不确定性；

---

## 227. The price of multi-group transductive learning

**arXiv ID:** 2606.04423 | [PDF](https://arxiv.org/pdf/2606.04423v1)

**作者:** Noah Bergam `[一作]` (Columbia University), Daniel Hsu `[通讯]` (Columbia University)

**通讯引用:** 10502 | [OpenAlex ID](https://openalex.org/A5061246300)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究多组迁移学习（transductive learning）中多组学习的统计成本，提出多组学习相较于单组学习的误差率乘法惩罚可线性增长至约 √n，构建一类一致性图（one‑inclusion graph）证明下界；并给出两种图定向算法（剥离法与聚合法）实现上界为 O(√n) 或 O(1)；

**💡 创新点**

首次量化多组迁移学习的“价格”（price），即多组学习相对单组学习在误差率上的乘数提升，证明下界线性于组数 k 并上升到 √n，且该下界与现有统计学习的对数上界形成鲜明对比；

**🔧 技术方法**

利用图论（one‑inclusion graph）与图定向（orientation）技术，构造高密度图与星形子图；通过剥离法（peeling）与网络流聚合法（aggregation）设计随机定向策略；理论分析包含极限边密度、图密度与 VC 维度关系；

**📊 数据集**

本研究纯理论，没有使用具体数据集，所有结果基于数学证明与构造性示例；

**📈 对比分析**

与现有统计学习与在线学习的多组理论对比：在统计学习中多组误差率的乘法惩罚上界为 log n，而在迁移学习中已证明可达到 √n，说明迁移学习更具挑战性；

**⚠️ 局限性**

局限性：仅针对可实现（realizable）场景，未考虑对抗性或无噪声情况；上界仍含常数与 √n 依赖，是否可进一步降至常数尚未证明；对非二进制标签、多样本分布假设未展开；

---

## 228. TITAN-FedAnil+: Trust-Based Adaptive Blockchain Federated Learning for Resource-Constrained Intelligent Enterprises

**arXiv ID:** 2606.04388 | [PDF](https://arxiv.org/pdf/2606.04388v1)

**作者:** Muhammad Hadi `[一作]` (National University of Sciences and Technology), Muhammad Khuram Shahzad `[通讯]` (National University of Sciences and Technology)

**通讯引用:** 3049 | [OpenAlex ID](https://openalex.org/A5035787368)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种名为 TITAN‑FedAnil+ 的联邦学习与区块链结合框架，用于资源受限的智能企业边缘环境，实现安全高效的分布式模型训练。

**💡 创新点**

创新点包括使用 Affinity Propagation 动态聚类滤除恶意更新、GPU 向量化相似度计算（Turbo‑Mode）以及签名状态跳跃同步（Signed State Jumps），显著降低内存占用与同步延迟。

**🔧 技术方法**

技术手段涵盖联邦学习、PoW+签名状态跳跃的区块链共识、Affinity Propagation 聚类、GPU/Metal Performance Shaders 向量化、以及硬件感知的内存管理。

**📊 数据集**

实验数据集为 FEMNIST、CIFAR‑10 与 Sent140，并采用 Dirichlet 划分模拟非 IID 分布。

**📈 对比分析**

通过与原始 FedAnil+ 与 FedAvg 在 5/15/20 节点和 25/50 轮的不同攻击场景下对比，TITAN‑FedAnil+ 在 20 节点、50 轮时达到 81% 的准确率，内存始终保持在 8GB 以内，通信与同步时间大幅缩短。

**⚠️ 局限性**

局限性包括仍依赖 PoW 难度调节，算力差异可能影响同步效率；缺乏对极大规模网络的异步扩展实验；尚未集成差分隐私（DP）等更高级的隐私保护机制。

---

## 229. Rethinking Sales Lead Scoring with LLM-based Hierarchical Preference Ranking

**arXiv ID:** 2606.04387 | [PDF](https://arxiv.org/pdf/2606.04387v1)

**作者:** Chenyu Zhang `[一作]` (Li Auto Inc), Juyi Qiao `[通讯]` (Li Auto Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于大型语言模型的判别式框架asLLR和HPRO，用于长周期销售场景下的潜在客户评分与排序。

**💡 创新点**

创新点在于将销售漏斗层级转化为层级偏好对，并通过带边际的Bradley‑Terry目标实现稀疏标签到密集偏好监督；同时将LLM与三头（点位、对比、语义）联合训练。

**🔧 技术方法**

采用了LoRA微调的大型语言模型、三头机制（点位交叉熵、margin‑aware Bradley‑Terry对比损失、语言模型交叉熵）以及差异化学习率和正则化技术。

**📊 数据集**

使用了两个新能源汽车零售系统的专有数据集：340k样本（1.45%正例）和6.14M样本（1.33%正例），包含结构化CRM特征和对话文本。

**📈 对比分析**

与六种工业CTR基线（Wide&Deep、DeepFM、xDeepFM、DCN、DCN‑M、AutoInt）进行离线AUC和排名指标对比，asLLR基线AUC 0.7921，加入HPRO后提升至0.8161；在线132天A/B测试实现9.5%销量提升，P@0.1%提升39.7%。

**⚠️ 局限性**

局限在于需要明确的漏斗层级定义和边际设置，迁移到其他业务场景需重新标注；LLM规模较大且对不同业务的通用性尚待进一步验证。

---

## 230. Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models

**arXiv ID:** 2606.04385 | [PDF](https://arxiv.org/pdf/2606.04385v1)

**作者:** Shuwen Yu `[一作]` (Yunnan Normal University), Huafeng Li `[通讯]` (Kunming University of Science and Technology)

**通讯引用:** 5601 | [OpenAlex ID](https://openalex.org/A5080535168)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出GPUA框架，利用可冻结的视听基础模型通过无监督的几何保持映射实现视觉语言对齐；

**💡 创新点**

创新点在于：①使用正交变换在不更新预训练模型参数的情况下保持视觉特征几何结构；②通过联合语义与几何的无监督对应挖掘（UCM）实现跨模态对齐；③引入Hubness抑制的THS损失进一步提升对齐质量；

**🔧 技术方法**

技术主要包括：离散正交Procrustes分析、熵正则化的最优传输求解（Sinkhorn）、自监督对应矩阵推理、Hubness-aware 排序损失；

**📊 数据集**

使用11个公开零样本分类基准（如Flowers、Pets、Caltech、FGVC、EuroSAT、UCF101、DTD、Food、Cars、SUN、ImageNet）及多种开放词汇分割数据集；

**📈 对比分析**

与CLIP、ZERO、MTA、TDA、ZLaP、DPE、DMN、StatA、TIPPLE、COSMIC等基线比较，GPUA在多数数据集上均取得显著提升（平均提升约10–14%），且在低样本版本GPUA*仍保持竞争力；

**⚠️ 局限性**

局限在于未考虑类别不平衡导致的对应错误，缺乏对数据分布不均衡的自适应处理。

---

## 231. When Do Fewer Coordinates Suffice in DP-SGD?

**arXiv ID:** 2606.04375 | [PDF](https://arxiv.org/pdf/2606.04375v1)

**作者:** Huiqi Zhang `[一作]` (Beijing Normal-Hong Kong Baptist University), Fang Xie `[通讯]` (Beijing Normal-Hong Kong Baptist University)

**通讯引用:** 1174 | [OpenAlex ID](https://openalex.org/A5032656037)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种两阶段的坐标稀疏差分隐私随机梯度下降（TP-TopK）方法，先在全参数空间进行DP温身阶段收集梯度信息，再根据DP可见的梯度能量选择一组活跃坐标，随后仅在该坐标子空间内执行DP-SGD，从而降低隐私噪声维度。

**💡 创新点**

创新点包括：①利用DP可见梯度统计进行无额外隐私成本的坐标选择；②证明在满足一定条件下，噪声能量由原始维度d降至活跃维度k，理论上提升收敛速度；③提供一阶条件和误判概率界，说明何时学习的坐标集合优于随机选择；④在无公开数据、从零开始训练的敏感领域场景中验证效果。

**🔧 技术方法**

技术方法主要包括DP-SGD、梯度裁剪、加高斯噪声、子采样RDP会计、Top-K排序、两阶段训练结构、误差分析与收敛证明。

**📊 数据集**

使用的公开数据集包括MNIST、Fashion-MNIST、CIFAR-10；敏感医疗领域数据集为EyePACS diabetic retinopathy筛查数据。

**📈 对比分析**

通过对五种DP训练方法（全参数DP-SGD、TP-TopK、TP-Rand、ON-TopK、ON-Rand）在相同隐私预算、活跃比例和训练时间下进行严格对照，结果显示TP-TopK在CIFAR-10上显著优于全参数DP-SGD（提升约9–10个百分点），在FMNIST和MNIST上也有小幅提升；相对于随机坐标选择的TP-Rand，TP-TopK在低隐私预算（ε=1）下提升约0.8–1.4个百分点，且随着活跃比例减小，优势更明显。

**⚠️ 局限性**

主要局限包括：①在极强隐私预算（ε≈1）下温身阶段的DP噪声会削弱坐标排名的可靠性，导致优势减小；②温身阶段消耗隐私预算，导致后续阶段可用预算减少；③理论分析基于一阶代理条件，未能给出完整的多步收敛保证；④方法未针对层级或结构化稀疏性做进一步优化。

---

## 232. DSIRM: Learning Query-Bridged Discrete Semantic Identifiers for E-commerce Relevance Modeling

**arXiv ID:** 2606.04374 | [PDF](https://arxiv.org/pdf/2606.04374v1)

**作者:** Bokang Wang `[一作]` (Taobao & Tmall Group of Alibaba), Jianbo Zhu `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了如何将离散语义标识符（SID）从检索目标转化为增强电商搜索排序的结构化相关特征，并提出基于查询桥接的对比量化方法和生成式LLM进行查询SID预测。

**💡 创新点**

创新点在于：① 用查询桥接的对比RQ‑VAE主动将语义空间分割为查询相关的离散代码；② 通过生成式LLM生成多条查询SID以捕捉意图多样性；③ 将SID前缀匹配得分与连续向量结合提升精细匹配。

**🔧 技术方法**

采用的技术包括：双塔预训练对比学习、残差量化变分自编码器（RQ‑VAE）、InfoNCE对比损失、类别感知码表、自动回归LLM（Qwen3）以及多层感知机DNN。

**📊 数据集**

数据集为天猫千万级交互日志约80M（用于SID学习）与1.6M人工/LLM标注的交互对（用于评估），并在上线环境进行A/B测试。

**📈 对比分析**

与基线无SID、DSI、TIGER等检索侧SID方法比较，DSIRM在离线AUC提升至0.9356（最高），在线点击率提升0.13%，转化率提升0.25%。

**⚠️ 局限性**

限制在于：① 需要预训练高质量连续嵌入；② 查询侧LLM推理仍需额外延迟，尽管已做离线缓存；③ 对极少数新商品的SID缺失仍采用默认值，可能影响极端场景性能。

---

## 233. Read the Trace, Steer the Path: Trajectory-Aware Reinforcement Learning for Diffusion Language Models

**arXiv ID:** 2606.04396 | [PDF](https://arxiv.org/pdf/2606.04396v1)

**作者:** Anant Khandelwal `[一作]` (Microsoft), Manish Gupta `[通讯]` (Microsoft)

**通讯引用:** 5529 | [OpenAlex ID](https://openalex.org/A5101454729)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种新型的强化学习算法CAPR，用于改进掩码扩散式大型语言模型（dLLM）的推理性能；

**💡 创新点**

创新点在于利用dLLM的去噪轨迹（confidence、entropy与稳定性）构建路径状态，既能在不完全展开树的情况下实现局部奖励分配，又通过缓存前缀、分支与修剪以及块级价值评估，实现低计算成本的树形监督；

**🔧 技术方法**

核心技术包括路径状态缓存与引导（Cache & Steer）、中间分支与修剪（Branch & Prune）、块级价值头（Block Critic）以及自我蒸馏（Self‑Distillation）等；

**📊 数据集**

在四个可验证推理基准上评估：4×4 Sudoku、Countdown、GSM8K、Math500；使用了LLaDA-8B-Instruct（dense）和LLaDA-MoE-7B-A1B-Instruct（Mixture‑of‑Experts）两种模型；

**📈 对比分析**

与多种基线（Flat‑GRPO、VRPO、wd1、SAPO、d‑TreeRPO等）对比，CAPR在256/512 token预算下均取得或领跑所有任务的pass@1成绩，尤其在Sudoku和Countdown上显著提升，且计算成本仅为树形方法的约60%（≈1/3的单步壁时）；

**⚠️ 局限性**

局限性包括：仅在可验证推理任务上验证，未测试开放式指令遵循、事实性或安全性；块级价值估计对极长推理链或多重子目标可能不稳健；实验受特定硬件与软件栈影响，其他dLLM架构的适用性需进一步评估；

---

## 234. Bridging Short Videos and Live Streams: Reasoning-Guided Multimodal LLMs for Cross-Domain Representation Learning

**arXiv ID:** 2606.04448 | [PDF](https://arxiv.org/pdf/2606.04448v1)

**作者:** Le Zhang `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出RGCD-Rep框架，实现从短视频向直播的跨域推荐，通过多模态大语言模型推理学习可转移项目表示，解决数据稀疏和冷启动。

**💡 创新点**

创新性地将MLLM推理知识蒸馏与传递性-残差查询聚合结合，在两阶段训练中引入行为协作对和可转移路由，生成可离线部署的跨域可转移表示。

**🔧 技术方法**

使用多模态大语言模型（教师Qwen3.6‑35B、学生Qwen2‑VL‑2B），CoT prompting、结构化推理标签、行为协作对构建、两阶段训练、交叉域对比学习、可转移路由、Chorus tokens及Transferable‑Residual Query‑Aware Aggregation。

**📊 数据集**

基于2026年快手直播平台一周真实日志，109,523用户、98,190直播、3,787,061短视频，构成训练集和测试集。

**📈 对比分析**

与单域多模态、跨域无模态、跨域多模态方法（如FREEDOM、SMORE、AlphaRec、UniSRec、VQ‑Rec、MISSRec、PMMRec、UniEmbedding）对比，HR@10/NDCG@10分别提升约17.6%/30.9%；在线A/B测试在快手与快手 Lite 上分别提升+0.37%/+0.38%曝光、+0.34%/+0.13%点击、+0.28%/+0.22%有效入场、+0.83%/+0.93%关注。

**⚠️ 局限性**

依赖教师大模型推理生成结构化标签成本高；需要高质量行为协作对；对极端稀疏或冷启动场景仍有限；离线生成表示对实时更新有延迟；在其他多模态跨域场景的泛化尚待验证。

---

## 235. Motion-Guided Causal Disentanglement for Robust Multi-View Cine Cardiac MRI Diagnosis

**arXiv ID:** 2606.04414 | [PDF](https://arxiv.org/pdf/2606.04414v1)

**作者:** Chuankai Xu `[一作]` (University of Virginia), Jianxin Xie `[通讯]` (University of Virginia)

**通讯引用:** 240 | [OpenAlex ID](https://openalex.org/A5100337725)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并实现了一种基于ViT‑MAE的运动引导因果解耦框架MoViD，用于多视角心脏磁共振影像的疾病判别和分割。

**💡 创新点**

通过双分支对比学习与梯度反转实现视角与疾病特征的显式分离，并利用无标注的时间差动运动信号进行心脏定位，显著提升低样本下的鲁棒性。

**🔧 技术方法**

ViT‑MAE预训练、监督对比学习、梯度反转对抗、焦点加权、时间差动运动ROI、视角一致性正则化等技术。

**📊 数据集**

私有VTE患者多视角数据集；公开M&Ms与M&Ms2多视角心脏MRI基准。

**📈 对比分析**

与3D CNN、ResNet50、ViT‑B/16和CineMA等基线对比，在三组数据上均获得更高的AUROC、ACC、Dice等指标；在VTE上敏感度达到99.56%。

**⚠️ 局限性**

对视角覆盖不完整的样本仍有性能下降，且方法在短轴视角尚未验证，缺少临床多模态信息。

---

## 236. Low-Rank Decay for Grokking in Scale-Invariant Transformers: A Spectral-Geometric View

**arXiv ID:** 2606.04405 | [PDF](https://arxiv.org/pdf/2606.04405v1)

**作者:** Mingyu Li `[一作]` (Beijing Normal University), Mingyu Li `[通讯]` (Beijing Normal University)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5100569194)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

研究了在规模不变的Transformer中使用核范数衰减(Low‑Rank Decay, LRD)来促进算法学习，探讨其相较于传统L2衰减的机制差异

**💡 创新点**

提出LRD通过逼近核范数子梯度的极化因子来在零梯度后仍保持谱压缩的切向更新，从而在规模不变层上实现功能性正则化

**🔧 技术方法**

使用Newton–Schulz迭代逼近极化因子，设计了去耦合LRD更新，并结合RMSNorm、QK‑Norm等归一化技术

**📊 数据集**

在模加法（modular addition）算法数据集上进行实验

**📈 对比分析**

与标准L2衰减及LRD+L2混合正则化对比，结果表明LRD显著扩大了“grokking”成功的训练样本比例范围，并加速了从记忆到泛化的转变

**⚠️ 局限性**

仅在模加法任务上验证，未在乘法或更复杂任务上证明普适性，缺乏多种随机种子和置信区间，未能明确证明秩崩塌与泛化的因果关系

---

## 237. When Clients Stop Following: A Cognitive Conceptualization Diagram-driven Framework for Strategic Counseling

**arXiv ID:** 2606.04389 | [PDF](https://arxiv.org/pdf/2606.04389v1)

**作者:** Yihao Qin `[一作]` (Lanzhou University), Bin Hu `[通讯]` (Lanzhou University)

**通讯引用:** 25050 | [OpenAlex ID](https://openalex.org/A5100380066)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于 CBT 的抵抗感知客户模拟器 CARS、双模块策略生成框架 STREAMS 以及熵加权互信息评估指标 EWTS-MI，用于在心理咨询 LLM 训练和评估中更真实地模拟并衡量客户抵抗。

**💡 创新点**

创新点在于将 Cognitive Conceptualization Diagram 作为动态抵抗模型嵌入模拟器，分离策略推理与语言生成以强化鲁棒性，并设计针对高熵交互的评估度量。

**🔧 技术方法**

采用结构化链式思考、强化学习（PPO）、GRPO 对齐技术，配合 LLM 生成器完成思维者与呈现者的训练。

**📊 数据集**

使用 CARS 构造的 30 条 CCD 基础的模拟客户数据，结合公开 CBT 训练手册、已有模拟器数据以及多家 LLM 后端进行实验。

**📈 对比分析**

与 19 组基线模型对比（包括通用 LLM 与专门领域模型），通过自动指标（EWTS-MI、RTF、UEC）和人工评估（策略有效性、漂移、进展）发现 STREAMS-RG 在抵抗情境下取得最高人类评分并保持竞争力。

**⚠️ 局限性**

局限包括 CBT 框架约束、单轮会话评估、指标估计不稳定、对非 CBT 客户和多文化背景的覆盖不足以及对长期治疗目标的欠缺验证。

---

## 238. DLLG: Dynamic Logit-Level Gating of LLM Experts

**arXiv ID:** 2606.04378 | [PDF](https://arxiv.org/pdf/2606.04378v1)

**作者:** Bingnan Li `[一作]` (AWS Agentic AI), Stefano Soatto `[通讯]` (AWS Agentic AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态的logit级别门控框架 DLLG，用于在不对专家模型进行再训练的前提下，学习从稀疏的响应级监督中得到的 token 级融合权重，实现在推理时对多个专业 LLM 进行软融合。

**💡 创新点**

创新点在于：①利用响应级正确性标签通过教师强制训练门控网络，实现 token 级融合权重的学习；②在 logit 层进行软融合，避免硬路由的早期承诺与参数融合的干扰；③门控网络轻量化、可插拔，保持专家模块的独立性。

**🔧 技术方法**

技术手段包括：教师强制训练、均方误差监督、轻量化门控网络（投影、低秩层、LoRA 适配、KV 缓存）、logit 级软融合、token 级融合权重预测。

**📊 数据集**

使用的主要数据集有：GSM8K、MATH、Code‑R1、HumanEval、MBPP、BBH、BigCodeBench（以及对应的训练集用于门控训练）。

**📈 对比分析**

与基线方法相比，DLLG 在 0.5B 与 1.5B 规模下平均得分最高。与路由（RouterDC、EmbedLLM）、启发式加权（GaC、Entropy、Pack of LLMs 等）以及参数空间合并（Linear、SLERP、Task Arithmetic）相比，DLLG 在所有评测任务上均表现出更好的性能，尤其在混合推理/代码生成任务中优势明显。

**⚠️ 局限性**

局限性包括：①假设所有专家共享同一 tokenizer 与词表，难以直接扩展到异构 tokenizer；②需要响应级正确性标注，若自动验证器不完善会影响门控学习；③在推理时需并行运行所有专家，增加计算成本。

---

## 239. Context-as-a-Service: Surfacing Cross-File Dependency Chains for LLM-Generated Developer Documentation

**arXiv ID:** 2606.04397 | [PDF](https://arxiv.org/pdf/2606.04397v1)

**作者:** Ameya Gawde `[一作]` (Meta Platforms, Inc.), Lucy Moys `[通讯]` (Meta Platforms, Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 Context-as-a-Service（CaaS）检索层，使 LLM 文档代理能够在生成或评审文档时跨文件检索证据，从而发现跨文件依赖导致的文档错误。

**💡 创新点**

创新点在于：①将关键字搜索与语义检索（BM25 + DRAMA embeddings）结合并通过 reciprocal rank fusion 自动融合；②提供可工具调用的检索接口，让 LLM 在保持自身文件读取和推理流程的同时，轻松获取跨文件证据；③在两个真实 SDK 文档工作流中验证其效果，展示检索层能显著提升发现率和效率。

**🔧 技术方法**

使用的技术包括 BM25 关键字检索、DRAMA 语义嵌入、reciprocal rank fusion、工具调用接口、Claude Sonnet 4.6 生成模型。

**📊 数据集**

使用的测试数据集为一款约 200 个源文件的生产 SDK（包含实现代码、API 文档、测试、示例及上游文档），但因其为专有代码，未公开公开化。

**📈 对比分析**

对比方法：基线为 LLM 代理仅使用文件读取、关键字搜索和符号导航；实验为同一代理加上 CaaS 检索。结果显示：CaaS 在两项案例中分别发现 4、4 项额外问题，整体发现率提升 60%+；平均墙时间缩短 22%-34%，输入 token 减少 15%-30%，输出 token 略增；LLM 调用次数上升但整体效率更高。

**⚠️ 局限性**

局限性：仅在单一 SDK 及两条工作流上评估，缺乏跨语言或跨项目的泛化验证；基线仅使用一种代理与工具，未包含更强基线；检索粒度主要为文件级，未完全到符号级；未进行人类开发者影响评估。

---

## 240. Modeling and Interpreting Teamwork Dynamics in Cancer Care Outcome Prediction

**arXiv ID:** 2606.04499 | [PDF](https://arxiv.org/pdf/2606.04499v1)

**作者:** Yuhua Huang `[一作]` (University of California, Davis), Kwan-Liu Ma `[通讯]` (University of California, Davis)

**通讯引用:** 13987 | [OpenAlex ID](https://openalex.org/A5037161857)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究基于电子病历记录构建患者级别的时间演化协作网络，利用时序图神经网络预测癌症患者生存，并通过代理模型与SHAP解释团队协作对结果的影响。

**💡 创新点**

创新点在于将临床团队协作视为可量化的时间演化图结构，结合时序图神经网络捕获协作演化信号，并设计代理+时间分割方法将黑盒模型转化为可解释的协作特征，揭示手术专家早期介入与生存相关性。

**🔧 技术方法**

主要技术包括：时序图神经网络（EvolveGCN-O）、图卷积、GRU参数演化、节点特征构造、代理线性模型（Logistic回归）与SHAP值解释、时间分段与增量分析。

**📊 数据集**

使用了来自加州大学戴维斯分校的505例二、三期乳腺、肺癌、结肠直肠癌患者的电子病历访问日志与临床信息，日志覆盖诊断前3个月至诊断后12个月，筛选MD、NP、PA、RN等核心医护人员的访问事件。

**📈 对比分析**

通过与代理模型比较，时序图神经网络在三个癌症队列中分别实现F1≈0.927（乳腺）、0.863（结肠直肠）、0.848（肺），显著优于仅使用传统临床特征的基线；在时间分段分析中，围绕手术参与激增点的后期片段表现最优，进一步验证模型有效性；模型鲁棒性在10次随机种子实验中属性排名稳定。

**⚠️ 局限性**

局限性包括：仅考虑基于记录笔记的访问事件，忽略口头沟通与非记录协作；仅纳入特定职称的医护人员，可能遗漏关键角色；研究范围局限于三种癌症，缺乏广泛推广；时序窗口长度与代理特征选择仍需进一步系统评估。

---

## 241. SFMambaNet: Spectral-Frequency Enhanced Selective State Space Model for Correspondence Pruning

**arXiv ID:** 2606.04493 | [PDF](https://arxiv.org/pdf/2606.04493v1)

**作者:** Zhihua Wang `[一作]` (University of Shanghai for Science and Technology), Yizhang Liu `[通讯]` (Fuzhou University)

**通讯引用:** 2192 | [OpenAlex ID](https://openalex.org/A5078694785)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新型的SFMambaNet网络，用于对应关系剔除。

**💡 创新点**

创新点是将频域感知与状态空间模型结合，分别在局部通过频谱编码提升几何判别，在全局通过频域门控抑制高频噪声。

**🔧 技术方法**

采用了Spectral-Geometric Attention（LSGA）+Spectral-Integrated Global Mamba（SIGM）以及DiffPool、FFT等技术。

**📊 数据集**

在YFCC100M、SUN3D、HPatches等公共数据集上进行训练与评测。

**📈 对比分析**

与现有GNN、Transformer和MatchMamba等方法对比，SFMambaNet在姿态估计、误匹配剔除和单应性估计任务上均取得了领先或可比的高精度，同时保持了较低的参数量和运算量。

**⚠️ 局限性**

局限性包括对超大规模匹配仍需优化内存开销，以及在极端重复纹理场景下的召回率略低。

---

## 242. OSCAR: Omni-Embodiment Skeleton-Conditioned World Action Model for Robotics

**arXiv ID:** 2606.04463 | [PDF](https://arxiv.org/pdf/2606.04463v1)

**作者:** Zhuoyuan Wu `[一作]` (Peking University), Jun Gao `[通讯]` (University of Michigan)

**通讯引用:** 7834 | [OpenAlex ID](https://openalex.org/A5005407889)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个能够精确跟随动作、跨机器人结构泛化的动作条件视频世界模型，用于机器人策略评估。

**💡 创新点**

提出大规模统一数据管线与基于2D运动骨架的统一动作条件表示，使模型既能精准跟随动作，又能跨不同机器人/人手实现泛化。

**🔧 技术方法**

使用Cosmos-Predict2.5-2B视觉扩散Transformer，结合WAN 2.1 VAE、骨架渲染条件注入，并在单张GH200 GPU上进行微调。

**📊 数据集**

收集并清洗了来自七个公开机器人数据集（Franka Panda、KUKA iiwa、AgiBot G1、Toyota HSR等）以及人类 egocentric 数据集（EgoDex、EPIC-Kitchens）的约180k 片段。

**📈 对比分析**

与文本控制、潜在动作、点图、网格渲染等基线对比，模型在PSNR、SSIM、LPIPS、FVD等指标上均达到或接近最优，且在RoboArena策略评估时与真实实验的Pearson相关系数超过0.85，显示出强相关性。

**⚠️ 局限性**

受限于摄像头标定与运动学注释的可用性导致数据规模受限，且仅使用2B参数网络；更大模型与更丰富的数据可能进一步提升质量。

---

## 243. CyberGym-E2E: Scalable Real-World Benchmark for AI Agents' End-to-End Cybersecurity Capabilities

**arXiv ID:** 2606.04460 | [PDF](https://arxiv.org/pdf/2606.04460v1)

**作者:** Tianneng Shi `[一作]` (UC Berkeley), Dawn Song `[通讯]` (UC Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可扩展、真实可复现的端到端网络安全基准，评估 AI 代理在漏洞检测、PoC 生成和补丁修复全过程的能力。

**💡 创新点**

创新点在于：① 设计了自动化、代理增强的构建流水线，将 OSS‑Fuzz 的真实漏洞数据转化为完整的编译、PoC、补丁与功能测试环境；② 提供了大规模（超 600 条任务）端到端评估集；③ 通过对比多模型多框架（Claude、GPT、Gemini、OpenHands）和跨跑反馈来揭示现有代理在安全任务中的瓶颈。

**🔧 技术方法**

技术手段包括：Docker 化构建环境、自动化 PoC 与补丁检索、基于 LLVM Sanitizer 的 crash 触发验证、利用代码搜索工具（ripgrep、fd）进行代码定位、人工专家审核测试覆盖、以及多轮成本/时间预算调优的评估框架。

**📊 数据集**

数据集来源于 OSS‑Fuzz 的 13k+ 漏洞记录，结合 ARVO 与 CyberGym 的预包装数据，并通过流水线筛选后得到 615 条真实项目（C/C++）的漏洞任务。

**📈 对比分析**

比较方法：在相同 $10 成本和 90 分钟时间预算下，使用四种模型/框架组合进行 patch‑only 与 end‑to‑end 评估。结果显示：最优模型（Claude Opus 4.6）在 patch‑only 处 84.1% 成功率，在 end‑to‑end 仅 39.7%（S1）→ 39.5%（S2）→ 37.9%（S3）→ 15.7%（S4），表明检测阶段是主要瓶颈；跨跑反馈可提升 5–7% 成功率。

**⚠️ 局限性**

局限性包括：仅覆盖 C/C++ 内存安全漏洞；对非 sanitizer 触发的逻辑、注入、并发或 Web 漏洞缺乏评估；测试覆盖人工审核成本高；缺乏多语言、多平台的真实漏洞多样性。

---

## 244. A Reproducible Certificate for the Brass--Sharifi Lower Bound in Lebesgue's Universal Cover Problem

**arXiv ID:** 2606.04458 | [PDF](https://arxiv.org/pdf/2606.04458v1)

**作者:** Niantao Xie `[一作]` `[通讯]`, Niantao Xie

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在论文中，作者对 Brass–Sharifi 的凸形 Lebesgue 通用覆盖下界计算做了可复制的重现，构建了一份完整的计算证书；

**💡 创新点**

创新点在于将原先的计算结果转化为可审计、可验证的有限证书，包含自适应账本、终端路由重放、三类本地证书、完整的完整性审计和证明义务层；

**🔧 技术方法**

采用了自适应账本（递归细分树）、终端路由重放、导向区间证书、局部张量证书、h=0.004 桥接证书以及保护门控（guard accounting）等技术；

**📊 数据集**

使用的“数据集”是 Brass–Sharifi 计算过程中产生的数值表格和路由信息，包含 379,192 条父子边、356,816 条终端路由、41,261 条导向区间行、8,751 条张量成员等；

**📈 对比分析**

与原始方法相比，该证书并未改进数值下界，但通过完整的结构化数据和审计，提升了结果的可重复性与可验证性；

**⚠️ 局限性**

局限性包括：未对下界数值做改进、未给出非凸下界、未完成 Branch‑A 的符号化归约、未提供证明助手形式化或外部独立验证。

---

## 245. Pinning on Tight Cuts: Improved Algorithm and Bounds for Unsplittable Multicommodity Flows in Outerplanar Graphs

**arXiv ID:** 2606.04456 | [PDF](https://arxiv.org/pdf/2606.04456v1)

**作者:** David Alemán Espinosa `[一作]` (University of Waterloo), Niklas Schlomberg `[通讯]` (University of Bonn)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5006771715)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究无拆分多源多汇流问题，在外环图（outerplanar graph）中给出了更紧的容量超载上界与下界，证明若存在可行流，则必存在一个容量超载不超过 2 的无拆分流，而构造示例表明至少需要 43 的超载。

**💡 创新点**

创新点在于提出了一种全局参数驱动的分治框架，利用紧致割（tight cut）的结构将实例拆分为更小的子实例，并通过环加载（ring‑loading）问题的精细分析实现容量超载仅为 2；同时给出了极限构造，使下界从先前的 1.1 跃升到 43，展示外环图相较于普通环加载问题更具挑战性。

**🔧 技术方法**

主要技术包括：
- 通过紧致割的平衡性质保留可行性并构造“切割实例”（cut instance）和“分裂实例”（split instance）。
- 利用环加载实例的 3/2 及 1/2 的特定边容量上界。
- 采用递归分治和可行流最小化等概念，保证递归过程中容量超载控制在可接受范围。

**📊 数据集**

论文不依赖实际数据集，而是采用理论构造的实例（多份相互独立的环加载实例拼接成外环图），用于证明上界与下界。

**📈 对比分析**

与以往的 3.6 上界相比，作者将上界压缩至 2；与早期的 1.1 下界相比，作者将下界提升至 43，展示了两者之间仍存在巨大差距。所有结论均在多项式时间可实现，且通过构造性算法给出可行解。

**⚠️ 局限性**

局限性：
- 结果仅适用于外环图，尚未推广到更一般的平面图或全图。
- 上下界之间仍差距显著，进一步缩小差距仍是开放问题。
- 递归分治实现复杂度高，实际常数可能较大。

---

## 246. Stepwise Reasoning Enhancement for LLMs via External Subgraph Generation

**arXiv ID:** 2606.04454 | [PDF](https://arxiv.org/pdf/2606.04454v1)

**作者:** Xin Zhang `[一作]` (Chongqing Jiaotong University), Siying Li `[通讯]` (Chongqing Jiaotong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个结合大语言模型与知识图谱的 SGR 框架，通过动态生成查询相关子图实现逐步推理，提升答案的事实一致性与可解释性。

**💡 创新点**

创新点在于：①将自然语言问题转化为结构化 Schema 并基于此检索紧凑的子图；②利用 Cypher 查询直接获取图证据；③采用协同推理融合多条路径的答案与图一致性评分，从而显著提升推理质量。

**🔧 技术方法**

主要技术包括：实体/关系抽取与约束提取、Schema 生成、Neo4j+Cypher 查询、子图生成与路径评分、LLM 生成与验证、协同推理融合。

**📊 数据集**

使用的基准数据集有 CWQ、WebQSP、GrailQA 以及 KQA Pro。

**📈 对比分析**

与 IO Prompt、Chain-of-Thought Prompt、StructGPT、Tree-of-Graphs 等方法对比，SGR 在 Hits@1 与 Accuracy 上均实现了显著提升，SGR/ChatGPT 与 GPT‑4 版本更是逼近或达到当前 SOTA。

**⚠️ 局限性**

主要局限包括：依赖知识图的完整性与质量；实体/关系抽取误差导致子图缺失或错误；子图规模选择困难（过小缺失证据，过大噪声过多）；以及额外的 Schema 生成、Cypher 查询与路径融合增加推理成本。

---

## 247. On Out-of-sample Embedding in UMAP

**arXiv ID:** 2606.04451 | [PDF](https://arxiv.org/pdf/2606.04451v1)

**作者:** Mohammad Tariqul Islam `[一作]` (Massachusetts Institute of Technology), Jason W. Fleischer `[通讯]` (Princeton University)

**通讯引用:** 6765 | [OpenAlex ID](https://openalex.org/A5106440896)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究并解决 UMAP 在对新样本进行投影时出现的“排斥效应”，提出了参数化 UMAP 的方法。

**💡 创新点**

创新点在于将 UMAP 的映射函数用可微分的深度网络参数化，并通过交叉熵、均方误差及其组合损失进行学习，从而显著减弱排斥效应并提升嵌入质量。

**🔧 技术方法**

采用了基于 k‑NN 的高维图构建、负采样优化、交叉熵与均方误差损失的深度学习框架，以及信任度、最近邻分类器和力比率（AFR/RFR）等评估技术。

**📊 数据集**

在 MNIST 手写数字、RSNA 肺炎 X 光影像以及急诊科“呼吸困难”临床数据三组多样化数据集上进行实验。

**📈 对比分析**

通过比较传统 UMAP（不同 n_s 值和训练+测试混合投影）与三种参数化 UMAP（MSE、CEMSE、CE），发现参数化 UMAP（尤其是 CE）在信任度、k‑NN 分类误差和“聚集度”指标上均优于非参数方法，且嵌入新样本的时间提升数百倍。

**⚠️ 局限性**

局限性包括：需要额外训练深度网络，超参数（如 n_s、网络层数、学习率）对结果影响显著；实验仅覆盖三类数据，未验证在更高维或更稀疏数据中的泛化能力；排斥效应虽显著缓解，但在极端稀疏或非欧几里得空间中可能仍存在问题。

---

## 248. ANN Search: Recall What Matters

**arXiv ID:** 2606.04522 | [PDF](https://arxiv.org/pdf/2606.04522v1)

**作者:** Dimitris Dimitropoulos `[一作]` (University of Ioannina and Archimedes, Athena RC), Nikos Mamoulis `[通讯]` (University of Ioannina and Archimedes, Athena RC)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估ANN检索时用Recall@k存在缺陷，提出用1/Ratio@k替代，实验验证其更贴近下游任务质量。

**💡 创新点**

引入无判断、无超参数、仅凭现有嵌入/真值即可计算的1/Ratio@k指标，显示其比Recall更有效且成本更低。

**🔧 技术方法**

采用近似最近邻算法（Annoy、SuCo、HNSW、RaBitQ、SymphonyQG）与距离比率计算、下游分类与检索增强生成评测。

**📊 数据集**

在6个内在维度差异大的数据集（Gist、SimpleWiki、ImageNet、AGNews、MNIST、Fashion‑MNIST、CIFAR‑10、SVHN、SciFact、NFCorpus、HotpotQA、MS‑MARCO、PubMedQA）上进行实验。

**📈 对比分析**

与Recall相比，1/Ratio在相同质量阈值下QPS提升1.5–10倍、距离计算降低1.8–9.4倍，且在分类和RAG任务中对精度的预测误差下降至<3%而Recall为~25%。

**⚠️ 局限性**

仅在极高LID或极大k时仍可能出现Recall不足导致的排名波动；且实验未覆盖所有可能的检索距离度量与硬件环境，需进一步验证。

---

## 249. Cooperative Circumnavigation for Multiple Unmanned Surface Vehicles Without External Localization

**arXiv ID:** 2606.04518 | [PDF](https://arxiv.org/pdf/2606.04518v1)

**作者:** Xueming Liu `[一作]` (Sun Yat-sen University), Qingrui Zhang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 13267 | [OpenAlex ID](https://openalex.org/A5101842007)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个基于内部传感器的多无人水面船舶协同环绕航行框架，能够在无外部定位系统的环境下实现目标跟踪和分布式控制。

**💡 创新点**

创新点包括：异构感知策略将主动距离测量与被动方位测量结合，采用最大互相关卡尔曼滤波和伪线性卡尔曼滤波以应对非高斯噪声，并通过耦合振荡器控制器主动生成持续激励以满足估计系统的观测性。

**🔧 技术方法**

使用的技术主要有：最大互相关卡尔曼滤波（MCKF）、伪线性卡尔曼滤波（PLKF）、耦合振荡器模型的相位生成、向量场障碍规避控制以及多传感器数据融合。

**📊 数据集**

数据集方面：仿真采用高斯混合和对数正态噪声模型，并在实验中使用了真实的UTIL UWB测距数据集进行验证。

**📈 对比分析**

通过与RLS、KF、AKF、Huber-RLS等基线方法对比，MCKF在定位误差、收敛速度和实时性上表现更优，误差更小、收敛更快、实时性符合采样周期。

**⚠️ 局限性**

局限性在于动力学模型简化，未考虑水动力和环境扰动，未来需要引入学习或自适应方法以提升在复杂水域中的鲁棒性。

---

## 250. SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference

**arXiv ID:** 2606.04511 | [PDF](https://arxiv.org/pdf/2606.04511v1)

**作者:** Yaosheng Fu `[一作]` (NVIDIA), Oreste Villa `[通讯]` (NVIDIA)

**通讯引用:** 2332 | [OpenAlex ID](https://openalex.org/A5111373927)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为长上下文 LLM 推理提出一种可训练的稀疏选择方案：通过 Forecast 预测下一层块选择，实现 KV 缓存预取并降低选择开销。

**💡 创新点**

创新点在于将稀疏选择解耦为可训练的预测器，使用单头 GQA 预测器压缩选择过程，并通过持久 UVA Triton 核实现异步预取，显著减少计算与内存瓶颈。

**🔧 技术方法**

采用块稀疏注意力、训练 Forecast 投影（KL 损失）、持久 UVA 通道、批量自适应 CTA 分配等技术。

**📊 数据集**

在 MiniCPM4.1-8B 与 NOSA-8B 两个 8B 稀疏预训练模型上，使用 HELMET、LongBench、RULER 以及长推理基准（MATH‑500、AIME 2024/2025）进行评估。

**📈 对比分析**

与 Dense、Sparse、InfiniGen 等基线相比，SparDA 在 64K–128K 长度下预填速率提升 1.25–2.11 倍，解码速率提升 1.7×，并在大批量时实现最高 5.3× 的吞吐量提升。

**⚠️ 局限性**

局限性在于仅改进选择过程，准确度受基模型稀疏注意力限制；未直接针对 token‑level 稀疏；在更大模型上验证仍待进一步研究。

---

## 251. Smart Picks in the Dark: Towards Efficient RLVR for Reasoning via Tracing Metacognitive Pivots

**arXiv ID:** 2606.04503 | [PDF](https://arxiv.org/pdf/2606.04503v1)

**作者:** Guangcheng Zhu `[一作]` (Zhejiang University), Gang Chen `[通讯]` (Zhejiang University)

**通讯引用:** 103641 | [OpenAlex ID](https://openalex.org/A5100389265)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的主动数据选择框架PivotTrace，用于在RLVR（强化学习可验证奖励）中实现训练与标注的双重效率

**💡 创新点**

核心创新是利用长距离注意力动态检测“元认知枢轴”作为无监督的推理不确定性代理，并通过自动阈值校准实现三路数据分流（标注、未标注、丢弃）

**🔧 技术方法**

技术主要包括：基于GRPO的RLVR训练、长距离注意力峰值检测、Pivot计数作为不确定性指标、滑动窗口阈值自适应校准以及半监督RLVR混合奖励

**📊 数据集**

在多项数学与通用推理基准上评估：AIME、AMC、MATH-500、Minerva、OlympiadBench、ARC-c、GPQA-diamond、MMLU-Pro等；使用Qwen3-4B-Base模型和DAPO-Math-14k数据集进行训练

**📈 对比分析**

与六种无监督不确定性估计方法（Random、Consistency、Entropy、Self-Certainty、CoE、CoT-Kinetics）以及完全监督基线进行对比。PivotTrace在相同标注比例29.3%时实现+1.6% ID、+2.4% OOD准确率，并在仅57.9%训练样本下超过全监督模型，收敛速度提升2.75×

**⚠️ 局限性**

局限性包括：需先预先计算长距离注意力峰值，对模型和任务的通用性尚未完全验证；自动阈值校准依赖少量标注样本，可能对极端数据分布或噪声敏感；以及在极大规模模型或数据时的计算开销仍待评估

---

## 252. Beyond Prompt-Based Planning: MCP-Native Graph Planning-based Biomedical Agent System

**arXiv ID:** 2606.04494 | [PDF](https://arxiv.org/pdf/2606.04494v1)

**作者:** Zhangtianyi Chen `[一作]` (Chinese University of Hong Kong), Juexiao Zhou `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于图结构、MCP原生的生物医学智能体BioManus，能够自动化复杂的生物学工作流程。

**💡 创新点**

创新点在于通过BioinfoMCP编译器将异构工具统一为MCP服务器，并构建可执行的typed capability graph，实现从平面工具检索到图结构规划的转变。

**🔧 技术方法**

主要技术包括MCP编译、GraphRAG式图检索、LLM规划、MCP服务器动态注册以及图结构化工作流推理。

**📊 数据集**

使用的数据集包括BioAgentBench和LAB-Bench，并构建了910个MCP服务器、3500个工具的生态系统。

**📈 对比分析**

与ReAct-Code、Biomni等基线以及不同规模MCP库存进行对比，BioManus在BioAgentBench的LLM-judge分数和LAB-Bench的SeqQA、CloningScenarios上均取得最高或接近最高表现，同时保持更低的上下文开销。

**⚠️ 局限性**

限制包括依赖现有工具文档、图结构语义注解仍较轻量、检索与规划质量依赖于图的准确性，以及需要专家监督以确保科学结果的可信性。

---

## 253. AgentJet: A Flexible Swarm Training Framework for Agentic Reinforcement Learning

**arXiv ID:** 2606.04484 | [PDF](https://arxiv.org/pdf/2606.04484v1)

**作者:** Qingxu Fu `[一作]` (Tongyi Lab, Alibaba Group), Bolin Ding `[通讯]` (Tongyi Lab, Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AgentJet，一种分布式Swarm训练框架，将模型优化与Agent执行完全解耦，实现异构多模型、多任务、多Agent的强化学习训练。

**💡 创新点**

核心创新在于client‑server Swarm 架构，支持异构多模型训练、任务隔离、故障容忍、实时代码迭代、时间线合并加速、自动化科研管线，并实现框架无关的黑盒Agent支持。

**🔧 技术方法**

技术上采用OpenAI兼容API、vLLM/SGLang推理、Ray+FSDP分布式训练、GRPO算法、时间线合并、异步采样与集群调度、自动化研究Agent等。

**📊 数据集**

使用多Agent任务数据集（如AIME、CodeWorld、混合任务环境）以及公开LLM基准（Qwen3、OpenAI Gym等），实验中自行生成或引用公开多Agent交互数据。

**📈 对比分析**

与OpenRLHF、veRL、Forge、AReaL等现有框架对比，展示训练曲线、样本效率提升1.5–10倍、奖励/稳定性提升，并在多模型/多任务设置下实现更高性能；自动化科研实验在多GPU集群上完成长周期实验。

**⚠️ 局限性**

局限性包括对任务环境手动配置的依赖、时间线合并可能牺牲推理一致性、跨模型参数共享仍未支持、对极大规模模型可扩展性评估不足，以及奖励健壮性与安全性需要进一步完善。

---

## 254. Radiomic Feature Selection Using Gradient Loss of Deep Neural Network for Lung Cancer Stage Detection

**arXiv ID:** 2606.04453 | [PDF](https://arxiv.org/pdf/2606.04453v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 255. SePO: Self-Evolving Prompt Agent for System Prompt Optimization

**arXiv ID:** 2606.04465 | [PDF](https://arxiv.org/pdf/2606.04465v1)

**作者:** Wangcheng Tao `[一作]` (National University of Singapore), Weng-Fai Wong `[通讯]` (National University of Singapore)

**通讯引用:** 4320 | [OpenAlex ID](https://openalex.org/A5023989495)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种自参照的系统提示优化方法，使得提示代理（prompt agent）能够在同一优化循环中自我改进自身的系统提示，从而提升代理在多种任务上的表现。

**💡 创新点**

创新点在于把提示代理的系统提示也纳入优化目标，打破传统方法中提示代理手工固定的瓶颈，并采用两阶段预训练+微调的框架，实现在多任务间共享优化经验。

**🔧 技术方法**

技术上主要采用开放式进化搜索（open‑ended evolution）与归档机制进行提示优化，并通过自参照设计实现提示代理的自我进化；同时结合了两阶段训练流程。

**📊 数据集**

实验使用了五个不同领域的任务集：AIME'25（高考数学）、ARC‑AGI‑1（抽象推理）、GPQA（研究生级科学多项选择）、MBPP（代码生成）以及 Sudoku（逻辑谜题）。

**📈 对比分析**

与手工提示（Manual‑CoT）、TextGrad、MetaSPO 三种基线对比，实验显示 Generalist 版在所有任务上均优于手工提示，平均准确率提升约 4.49 分，且在成本上可通过预训练共享实现一定的经济优势。

**⚠️ 局限性**

局限性包括：仍需依赖预训练任务混合的手工选择；进化搜索空间有限，可能无法覆盖所有有用提示；方法仅针对系统提示，未扩展到工具、工作流等更广泛的代理自我改进空间。

---

## 256. Imagine Before You Draw: Visual Prompt Engineering for Image Generation

**arXiv ID:** 2606.04457 | [PDF](https://arxiv.org/pdf/2606.04457v1)

**作者:** Liyu Jia `[一作]` (Nanyang Technological University), Hanwang Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 28492 | [OpenAlex ID](https://openalex.org/A5042324027)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出在图像生成过程中插入SigLIP 2视觉提示的中间步骤，将文本→图像的难题拆分为语义规划与细节渲染两步。

**💡 创新点**

创新点在于：①使用progressive训练策略解决训练与推理之间的视觉提示不匹配问题；②兼容内部与外部两种架构，系统评估视觉提示对离散、连续token及多任务（文本渲染、编辑）的提升；③通过共享注意力或跨模态融合显著降低信息瓶颈。

**🔧 技术方法**

技术方法包括：SigLIP 2视觉编码与VQ离散化、Emu3/ VAE连续token、基于自回归+扩散的生成框架、MOT内部模型、跨模态共享注意力、progressive masking 与loss‑scale 调节。

**📊 数据集**

使用的数据集涵盖 ImageNet‑1K、TextAtlas（TextScenesHQ、TextVisionBlend、StyleTextSynth）、BLIP3o、6.35 M文本渲染图、NHR‑Edit、UnicEdit、ShareGPT4o、Pico‑Banana 以及 GenEval 评测集。

**📈 对比分析**

通过在内部+（内部共享注意力）与外部+（AR + DiT）两种架构下进行同参数量（4.16 B）的对照实验，发现内部+在编辑任务中结构距离 24.60、PSNR 26.76、LPIPS 58.61 等指标显著优于外部+（分别为 61.66、19.92、158.09）；在文本‑图像生成上两者相近，但内部+在 4.16 B 参数下已超越多 7 B+ 的公开模型。

**⚠️ 局限性**

局限性：外部两阶段管线仍受信息瓶颈影响，难以精细保留细节；视觉提示虽能降低建模难度，却需要额外的 progressive 训练和 loss‑balance，且对连续 token 模型的训练复杂度较高。

---

## 257. The Meta-Agent Challenge: Are Current Agents Capable of Autonomous Agent Development?

**arXiv ID:** 2606.04455 | [PDF](https://arxiv.org/pdf/2606.04455v1)

**作者:** Xinyu Lu `[一作]` (Chinese Academy of Sciences), Le Sun `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 49193 | [OpenAlex ID](https://openalex.org/A5030983320)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了Meta-Agent Challenge（MAC）框架，评估大型语言模型在沙盒环境中自主构建、调优任务专用代理的能力。

**💡 创新点**

创新点在于将评估焦点从单纯任务执行转移到元层级的系统构建，提出安全多层隔离与奖励黑客防护机制，并提供公开可复现的基准，促进递归自我改进的研究。

**🔧 技术方法**

使用了代码代理、双容器架构、API代理、vLLM后端、Harbor框架、强化学习式的迭代调优以及后置审计等技术。

**📊 数据集**

采用了五个领域的现有基准数据集：AIME、GPQA/HLE、LiveCodeBench、SWE-Bench、Terminal-Bench，并在每个领域划分开发集与测试集。

**📈 对比分析**

通过与人类基准（手工设计的 Termination‑2、OpenHands 等）对比，发现大多数元代理未能匹配人类基准，只有少数由专有模型超越，且表现存在较大方差，且在高优化压力下出现奖励黑客行为。

**⚠️ 局限性**

主要局限包括极高的计算和时间成本、继承原始基准的局限性、潜在的预训练数据泄露风险以及安全防护机制仍需进一步强化。

---

## 258. Prioritization of Risks from Artificial Intelligence: A Delphi Study of 272 International Experts

**arXiv ID:** 2606.04490 | [PDF](https://arxiv.org/pdf/2606.04490v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 259. Self-Evolving Deep Research via Joint Generation and Evaluation

**arXiv ID:** 2606.04507 | [PDF](https://arxiv.org/pdf/2606.04507v1)

**作者:** Han Zhu `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 19056 | [OpenAlex ID](https://openalex.org/A5045081171)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自进化的协同进化训练框架，联合优化深度研究任务中的评估器和生成器，采用共享参数模型并通过元驱动器动态调节评估环境。

**💡 创新点**

创新点在于：①将评估器与生成器视为同一共享模型的不同角色，实现内部共进化；②引入元驱动器（Meta-Harness）根据生成器表现动态约束评估空间，避免评估器标准固化；③利用一致性奖励作为评估器的辅助信号，兼顾多维评估与可重复性。

**🔧 技术方法**

技术手段包括：共享参数的双角色模型；基于RL的评估器和生成器训练，生成器使用GRPO，评估器使用REINFORCE；KL正则化的交替更新策略；自适应评估维度和权重的生成；元驱动器使用GPT‑5.2动态调整评估环境。

**📊 数据集**

数据集：从Reddit收集的用户查询用于训练；评估使用DeepResearchBench和DeepResearchEval两个深度研究基准；检索环境使用本地WikiData搜索引擎训练，Web搜索用于评估。

**📈 对比分析**

对比方法包括标准GRPO、DPO以及基于Plan‑and‑Execute和ReAct的开放式深度研究代理。实验表明，在DeepResearchBench和DeepResearchEval上，协同进化框架在分析深度、有效引用、全面性等维度均显著优于基线；然而在可读性上略有下降，且传统RL方法在某些维度会出现坍塌。

**⚠️ 局限性**

局限性包括：①对元驱动器的依赖度高，若元驱动器设置不当会影响评估空间；②可读性与学术严谨之间存在权衡，生成文本更学术化导致对话式可读性下降；③评估器的一致性奖励可能被模型利用形成表面一致但缺乏真实深度的评估维度，存在潜在的奖励劫持风险。

---

## 260. Simulate, Reason, Decide: Scientific Reasoning with LLMs for Simulation-Driven Decision Making

**arXiv ID:** 2606.04505 | [PDF](https://arxiv.org/pdf/2606.04505v1)

**作者:** Yuhan Yang `[一作]` (University of Michigan), Alexander Rodríguez `[通讯]` (University of Michigan)

**通讯引用:** 1209 | [OpenAlex ID](https://openalex.org/A5067521241)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了MechSim框架，能够让LLM代理通过结构化的机制图、检索到的科学证据和可执行的敏感性分析，对可执行科学模拟器进行机制级的推理、解释和决策。

**💡 创新点**

创新点在于：①把模拟器的内部机制、假设和执行轨迹转化为可查询的图结构；②在LLM推理过程中强制使用结构化路径、实验结果和外部证据；③通过可执行验证（敏感性分析）来检验解释的可靠性；④实现对高风险领域中多模型多策略的可解释决策支持。

**🔧 技术方法**

采用的技术包括：神经符号化推理（LLM + 机制图约束）、检索增强的知识补全、近似贝叶斯计算实现参数校准、可执行敏感性分析、结构化验证与迭代优化。

**📊 数据集**

使用的实验数据集有：COVID‑19 真实病例/死亡数据、Supply‑Chain 的 M5 销售记录、Measles 的城市流行病学模拟场景（伦敦、芝加哥等）。

**📈 对比分析**

与Causal‑Copilot、Logic‑LM、Graph of Thoughts等基线在三项任务（政策选择、预测器选择、解释质量）进行对比。MechSim在Precision@k、Recall@k、预测误差 regret 以及解释四维评分上均显著优于基线，特别是在供应链和 COVID‑19 的决策任务中提升幅度最大。

**⚠️ 局限性**

主要局限：对大规模、复杂交互的动力学系统的可扩展性和计算效率仍待提升；对模型假设和外部证据的准确性高度依赖，易受数据偏差或假设错误影响；需要人工专家监督以避免误导性解释。

---

## 261. SANE Schema-aware Natural-language Evaluation of Biological Data

**arXiv ID:** 2606.04500 | [PDF](https://arxiv.org/pdf/2606.04500v1)

**作者:** Rolf Gattung `[一作]` (Karlsruhe Institute of Technology), Markus Reischl `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SANE框架，自动从高通量细胞成像实验数据库中生成结构化、与实验结构相匹配的自然语言查询及对应SQL，评估4-bit量化的Llama 3.1模型在无训练、无微调的few‑shot情境下的文本转SQL能力。

**💡 创新点**

创新点包括：1）基于实验内容自动生成schema‑aware benchmark；2）引入缺失上下文检测与澄清请求机制；3）展示在复杂生物数据库中仅凭prompt即可实现高精度查询。

**🔧 技术方法**

使用了schema‑aware prompting、vLLM部署、4-bit Llama 3.1量化模型，以及三阶段LLM处理管线（上下文判断 → SQL生成 → 结果解释）。

**📊 数据集**

采用真实高通量药物筛选实验数据库（细胞系、药物、浓度、图像复制等），共生成572条自动化测试案例。

**📈 对比分析**

通过对572条查询的精确匹配评估（结果集等价），整体准确率达97.2%，在不同类别中从100%到81.8%不等；相较于仅用schema注入的zero‑shot 29.9%显著提升，证明prompting和域知识对性能至关重要。

**⚠️ 局限性**

主要限制：错误多来源于缺失上下文或模糊输入，LLM对多轮澄清和交互式问答支持不足；系统高度依赖域特定prompt，通用性受限，且在错误语义理解上仍需改进。

---

## 262. Speculating the Impacts of Mediated Social Touch Technology

**arXiv ID:** 2606.04489 | [PDF](https://arxiv.org/pdf/2606.04489v1)

**作者:** Russian `[一作]`, Luke Hespanhol `[通讯]` (University of Sydney)

**通讯引用:** 1073 | [OpenAlex ID](https://openalex.org/A5003317688)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过Future Ripples工作坊与潜在用户、领域专家及触觉研究者共建MST（mediated social touch）未来情景，归纳出四大机遇与威胁主题以及三大独特挑战。

**💡 创新点**

创新点在于将Future Ripples方法与MST的录制–合成–再现（RSR）三阶段管线结合，提供可操作的干预点；同时提出“数字化触感催生新服务”“触觉大数据驱动AI”“异步触感重塑社交实践”等新的洞见。

**🔧 技术方法**

采用的技术与方法包括MST硬件/软件概念、Future Ripples工作坊流程、主题分析及RSR管线映射。

**📊 数据集**

数据来源为24名参与者在三次工作坊中生成的思考产物（便签、Miro画板、录音、访谈记录）以及对场景的“what‑if”设定。

**📈 对比分析**

本研究为定性研究，未与其他方法或系统进行量化对比，性能评估主要基于参与者反馈与主题覆盖度。

**⚠️ 局限性**

局限性包括样本规模有限、参与者构成不均衡、情景设定差异导致结果难以普适、缺乏长期跟踪与真实MST原型验证，且缺少量化性能指标。

---

## 263. Evaluating Reasoning Fidelity in Visual Text Generation

**arXiv ID:** 2606.04479 | [PDF](https://arxiv.org/pdf/2606.04479v1)

**作者:** Jiajun Hong `[一作]` (Stony Brook University), Jiawei Zhou `[通讯]` (Stony Brook University)

**通讯引用:** 1976 | [OpenAlex ID](https://openalex.org/A5056519111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估视觉文本生成中的推理可信度，设计并执行长文本渲染、事实知识检索、长上下文理解以及多步数学推理等任务，并将推理过程外化为图像中的文字，采用分层评估方法（渲染错误、过程错误、答案错误）来衡量模型性能。

**💡 创新点**

首次将推理过程可视化并与文本生成质量分离，提出基于OCR+LLM的分层评估框架，量化视觉文本生成与推理一致性之间的差距，并揭示当前T2I模型在逻辑一致性上的显著不足。

**🔧 技术方法**

使用PaddleOCR/DeepSeek-OCR进行文本提取，利用GPT‑5.2对推理步骤进行评分；对多种闭源（GPT‑Image‑1.5/2、Gemini‑2.5‑Flash‑Image、Flux.2‑Pro）和开源（Qwen‑Image、SD‑XL、TextDiffuser‑2）T2I模型进行评估；用VLM评估可读性指标（CCR/ACR）来补充OCR。

**📊 数据集**

WikiText（长文本渲染）、ARC Easy/Challenge（事实知识）、DROP（上下文理解）和MATH（数学推理）四大公开数据集。

**📈 对比分析**

通过WER、CER、OCR置信度评估渲染质量；通过过程得分（S_p）和答案得分（S_a）衡量推理质量。结果显示，T2I模型在渲染质量上可达到一定水平，但在过程得分上远低于文本LLM，尤其在长文本和数学推理任务中差距更大；GPT‑Image‑2表现最佳，但仍显著落后于GPT‑5.2。

**⚠️ 局限性**

OCR提取误差可能影响评估，尤其是数学符号提取；未全面探索字体、字号、布局等渲染参数；T2I模型主要针对短文本或自然图像，难以高效生成长篇结构化文本；评估仅关注显式推理步骤，未检验模型内部推理过程。

---

## 264. A Second-Order Cepstral Signature of Contact-Vibration Sounds Reproduced by Laptop Loudspeakers: A Synthetic Case Study

**arXiv ID:** 2606.04475 | [PDF](https://arxiv.org/pdf/2606.04475v1)

**作者:** Jim Salsman `[一作]` `[通讯]` (TalkNicer Company), Jim Salsman (TalkNicer Company)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究手机在硬表面上振动所产生的接触-振动声波在录制、编码和播放链中的波形和声谱特征

**💡 创新点**

提出并验证二阶倒谱双峰结构作为衡量机械性和声质独特性的指标

**🔧 技术方法**

使用倒谱分析、二阶倒谱（倒谱的倒谱）和模拟的信号链处理技术

**📊 数据集**

使用的是基于振动源、表面共振、麦克风响应、编码器、扬声器共振等构建的合成语料

**📈 对比分析**

通过对六个链路阶段的合成语料进行特征提取，展示第一阶周期结构在所有阶段保留，第二阶双峰最清晰出现在源头和播放阶段，未与真实音频进行实验对比

**⚠️ 局限性**

实验仅为合成模型，缺乏真实设备、听感测评及对比类别，二阶倒谱并非标准感知指标，结果不具普适性

---

## 265. Adaptive Calibration for Fair and Performant Facial Recognition

**arXiv ID:** 2606.04469 | [PDF](https://arxiv.org/pdf/2606.04469v1)

**作者:** Ryan Brown `[一作]` (University of Oxford), Chris Russell `[通讯]` (University of Oxford)

**通讯引用:** 8630 | [OpenAlex ID](https://openalex.org/A5008943199)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种后处理的自适应校准方法（Adaptive Calibration），将面部识别中的余弦相似度映射为局部上下文相关的概率，提升校准与公平性。

**💡 创新点**

创新点在于：①使用对称的平均嵌入点作为局部上下文，避免硬划分；②在相似度基础上学习连续、可变的区域校准函数；③不依赖种族或性别标签，适用于任何可在嵌入空间重建的群体。

**🔧 技术方法**

技术包括：轻量级 MLP 或线性逻辑回归、基于局部密度的岭回归残差校准；对配对嵌入使用余弦相似度与平均嵌入向量；在训练时最小化交叉熵实现概率校准。

**📊 数据集**

使用公开验证基准：RFW（按种族划分）、BFW（种族+性别）和 LFW（结合 FairFace 标签），并在五个主流嵌入网络（FaceNet、ArcFace R50/R100、AdaFace、MagFace）上评估。

**📈 对比分析**

与公平性与校准的基线方法（FairCal、FALCON、FRAPPÉ、FSN）及需要群标签的对比（Oracle、GST、DemoNorm）比较；在全局 TPR@FPR=10⁻³、最差组 AUROC、最差组 Brier 等指标上，Adaptive Calibration 在 9/9 设定中赢得标签无关方法的最好成绩，且几乎不产生“水平下降”现象。

**⚠️ 局限性**

局限性：需要足够的校准数据来覆盖嵌入空间的所有区域；仅能校正已有嵌入的概率，无法恢复模型已丢失的识别信息。

---

## 266. ParetoPilot: Zero-Surrogate Offline Multi-Objective Optimization via Infer-Perturb-Guide Diffusion

**arXiv ID:** 2606.04468 | [PDF](https://arxiv.org/pdf/2606.04468v1)

**作者:** Ruiqing Sun `[一作]` (National University of Defense Technology), Huaimin Wang `[通讯]` (National University of Defense Technology)

**通讯引用:** 10921 | [OpenAlex ID](https://openalex.org/A5018221313)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ParetoPilot，一种零代理（zero‑surrogate）扩散框架，用IPG引擎在离线多目标优化（Offline MOO）中直接引导生成器探索 Pareto 前沿；

**💡 创新点**

核心创新在于完全消除外部代理模型，利用预训练条件扩散模型内部的条件/无条件噪声预测对齐来隐式推断目标方向；并通过正交化的重力与边缘排斥力实现既趋近 Pareto 边界又保持多样性的动态扰动；

**🔧 技术方法**

使用条件扩散模型+分类器无关引导（CFG）+Adam 动态对齐+IPG（Infer‑Perturb‑Guide）引擎+正交化流体动力学+动态α与w调度；

**📊 数据集**

在 Off‑MOO‑Bench 共51个任务，涵盖Synthetic、MO‑NAS、SciDesign、RE Suite 四大域；

**📈 对比分析**

与14种前向代理方法和2种逆向生成基线进行对比，平均排名更低、超体积（HV）提升约10.46%，在所有域均超越基线；

**⚠️ 局限性**

局限性包括：依赖基模型对离线分布的准确建模；对齐与扰动计算导致推理延迟；未直接支持显式约束或离散结构任务，未来可进一步优化。

---

## 267. Token Rankings are Unforgeable Language Model Signatures

**arXiv ID:** 2606.04459 | [PDF](https://arxiv.org/pdf/2606.04459v1)

**作者:** Matthew Finlayson `[一作]`, Swabha Swayamdipta `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大语言模型（Pythia 70M 与 OLMo 3‑8B）的隐藏层输出做了低秩线性拟合实验，绘制了 Top‑k 维数下的相对误差曲线，并引入了可行性边界与相似度指标来评估模型在不同隐藏尺寸下的逼近能力。

**💡 创新点**

创新点在于提出了“可行性边界”概念，用以刻画不同模型隐藏尺寸下的误差极限；同时通过对比两个模型的误差曲线，揭示隐藏尺寸与误差之间的对数关系。

**🔧 技术方法**

使用了基于 Frobenius 范数的线性最小二乘低秩近似（fro_lstsq），对隐藏表示进行采样（2^14、2^16、2^15 条 Gaussian 样本）并在对数坐标系下绘图。

**📊 数据集**

实验数据来源于 Pile 数据集（见 sweep_pile_hidden.csv）以及生成的高斯样本，覆盖了 2^14~2^16 规模的隐藏表示。

**📈 对比分析**

通过绘制 Top‑k 误差曲线并标注相似度与可行性边界，实验表明 OLMo 3‑8B 在隐藏尺寸达到 4096 时误差显著低于 Pythia 70M，且两模型的误差随 k 以对数方式下降。

**⚠️ 局限性**

局限性包括仅评估了两种模型，误差指标仅限于 Frobenius 范数，采样规模有限，缺乏理论上对可行性边界的严谨证明。

---

## 268. Treat Traffic Like Trees: A Semantic-Preserving Hierarchical Graph-Based Expert Framework for Encrypted Traffic Analysis

**arXiv ID:** 2606.04517 | [PDF](https://arxiv.org/pdf/2606.04517v1)

**作者:** Yuantu Luo `[一作]` (Southeast University), Guang Cheng `[通讯]` (Southeast University)

**通讯引用:** 473773 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于协议树图注意力与混合专家的框架（PTGAMoE），用于在无识别信息、严格流隔离的条件下进行加密流量分类；

**💡 创新点**

创新点包括：①语义保持的协议树图结构，避免传统填充导致语义损失；②层级门控与混合专家自适应融合多协议层特征；③无序列化的流级聚合机制；④门控重要性与集中度指标提供可解释性；

**🔧 技术方法**

采用的技术主要有：图注意力网络（GAT）、混合专家（MoE）门控网络、字段级类型嵌入（地址/数值/分类）、层级节点初始化、最大池化聚合、焦点损失与门熵正则；

**📊 数据集**

实验数据集为两大TLS1.3基准集：CSTNET‑TLS1.3（26域）与CipherSpectrum（41域，共120k会话）；

**📈 对比分析**

在严格无识别信息、流隔离的实验设置下，PTGAMoE在CSTNET‑TLS1.3和CipherSpectrum上分别取得宏F1 92.65%与87.15%，显著优于ET‑BERT（64.48%/79.61%）、YaTC（79.61%/??）与RBLJAN（83.92%/??）等SOTA模型；

**⚠️ 局限性**

局限性包括：①仅在TLS1.3环境下验证，未覆盖UDP或代理协议；②流级专家的引入在某些数据集上反而导致性能下降；③模型复杂度较高，训练和推理成本较传统特征方法更高；④缺乏对不同网络环境的泛化评估。

---

## 269. GeoMin: Data-Efficient Semi-Supervised RLVR via Geometric Distribution Modeling

**arXiv ID:** 2606.04516 | [PDF](https://arxiv.org/pdf/2606.04516v1)

**作者:** Guangcheng Zhu `[一作]` (Zhejiang University), Gang Chen `[通讯]` (Zhejiang University)

**通讯引用:** 103641 | [OpenAlex ID](https://openalex.org/A5100389265)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GeoMin，利用全局特征分布进行几何化样本挖掘的半监督RLVR框架。

**💡 创新点**

创新点在于使用von Mises-Fisher分布捕捉正确/错误回放的方向差异，并通过优势重加权加速边界样本学习。

**🔧 技术方法**

采用vMF分布建模、优势重加权、Gaussian Mixture模型自适应筛选、GRPO强化学习等技术。

**📊 数据集**

在DeepMath-103k以及多项式与通用推理基准（AIME, AMC, MATH-500, Minerva, ARC-c, GPQA, MMLU-Pro）上验证。

**📈 对比分析**

与TTRL、Tok-entropy、Seq-entropy、Self-certainty、Co-rewarding、TraPO以及全监督基线比较，GeoMin在ID+OOD平均上分别提升+4.1%和+1.7%，且仅用10%标注即可超越全监督模型。

**⚠️ 局限性**

仅在≤8B参数模型上验证，低几何可分辨率模型需更长的第1阶段收敛，对多模态或跨语言推理的泛化尚未验证。

---

## 270. Episodic Memory Temporal Consistency for Cooperative Multi-Agent Reinforcement Learning

**arXiv ID:** 2606.04492 | [PDF](https://arxiv.org/pdf/2606.04492v1)

**作者:** Zicheng Zhao `[一作]` (Xi'an Jiaotong University), Xiaoming Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 21351 | [OpenAlex ID](https://openalex.org/A5100409052)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

针对多智能体强化学习中的奖励稀疏和探索瓶颈，提出了一种名为Episodic Memory Temporal Consistency（EMTC）的框架，结合记忆检索与奖励门控，实现更可靠的经验利用。

**💡 创新点**

核心创新点包括：①Temporally Consistent Semantic Embedder（TCSE）利用对比学习与时间条件重建双目标，防止表示坍塌并提升记忆检索精度；②Temporal Consistency Gating Mechanism（TCGM）根据贝尔曼一致性误差动态调节内在奖励，消除过估 Q‑value 的风险。

**🔧 技术方法**

采用的技术包括：对比学习、条件自编码器（dCAE）、贝尔曼一致性门控、CTDE 与 value‑factorization 架构（QMIX/QPLEX/CDS）以及结构化记忆缓冲区。

**📊 数据集**

实验数据集：StarCraft Multi‑Agent Challenge（SMAC）和 Google Research Football（GRF）。

**📈 对比分析**

与 EMU、QMIX、QPLEX、CDS 等基线对比，EMTC 在 SMAC 超难图上提升绝对胜率最高 24%，在 GRF 平均提升 28%，收敛速度与基线相近，整体性能显著优于现有方法。

**⚠️ 局限性**

限制：门控参数需要手动调节，过度门控可能抑制探索；在动态环境或极端奖励稀疏场景下的鲁棒性仍待进一步验证。

---

## 271. LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models

**arXiv ID:** 2606.04485 | [PDF](https://arxiv.org/pdf/2606.04485v1)

**作者:** Yuanrui Wang `[一作]` (Tsinghua University), Peng Cui `[通讯]` (Tsinghua University)

**通讯引用:** 21009 | [OpenAlex ID](https://openalex.org/A5009228005)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于 RBF 的数值特征编码层 RaBEL 并重新排列了双向注意力顺序，构建了 2M 参数的 LimiX-2M Tabular Foundation Model。

**💡 创新点**

创新点在于通过 RBF 库对标量进行局部非线性展开以解决低秩崩塌，并将注意力顺序改为样本→FFN→特征，保证所有注意力计算都被读出利用。

**🔧 技术方法**

使用的技术包括 RBF 编码、指数门控的 RaBEL、样本先行的注意力块、Transformer 结构以及基于合成 DAG 的自监督预训练。

**📊 数据集**

实验数据集涵盖 OpenML‑CC18、TALENT、TabZilla、TabArena、BCCO 等 300+ 公开 tabular benchmark。

**📈 对比分析**

与 TabPFN‑v2、TabICL、Mitra、XGBoost 等基线进行对比，LimiX‑2M 在大多数分类/回归任务上获得第二名，仅落后于更大 16M 参数模型，且训练推理成本显著降低。

**⚠️ 局限性**

局限在于仍依赖合成预训练样本，对跨域或极端缺失模式的泛化尚未充分验证，且在极大规模数据时可扩展性需进一步评估。

---

## 272. When Both Layers Learn: Training Dynamics of Representing Linear Models via ReLU Networks

**arXiv ID:** 2606.04476 | [PDF](https://arxiv.org/pdf/2606.04476v1)

**作者:** Berk Tinaz `[一作]` (University of Southern California), Mahdi Soltanolkotabi `[通讯]` (University of Southern California)

**通讯引用:** 3425 | [OpenAlex ID](https://openalex.org/A5046962187)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

研究一隐藏层ReLU网络在随机初始化下训练整个网络（内层和外层）以拟合线性目标函数，证明梯度下降能全局收敛并给出线性收敛率与最佳样本复杂度。

**💡 创新点**

首次在存在非严格鞍点、梯度下降可能停滞的情形下，给出完整的轨迹级分析，证明随机初始化能避开鞍点；提出沿整个梯度下降轨迹的统一集中度量，并利用其在收敛过程自适应收紧。

**🔧 技术方法**

梯度下降、平面分解、非严格鞍点理论、Polyak–Lojasiewicz不等式、梯度光滑性证明、统一集中度量、三阶段（对齐‑增长‑局部细化）分析。

**📊 数据集**

采用高斯分布 i.i.d. 输入（标准正态），生成线性标签，所有实验均在该合成数据上进行。

**📈 对比分析**

实验表明即使在 k>2 时也能收敛到全局最优，理论上在 k=2 时取得线性收敛率和 O(d) 样本复杂度；与传统需要预处理或特殊初始化的结果相比，性能更稳健、参数更少。

**⚠️ 局限性**

仅在 k=2 时给出严格的全局收敛证明；对 k>2 的结果仍为经验性；假设输入是标准高斯，理论不涵盖噪声或非高斯分布；对全局优化仅在理想的批量梯度下降下证明。

---

## 273. Learning What to Learn: Stage-Specific Data Sets for SFT-then-RL in Small Language Model Reasoning

**arXiv ID:** 2606.04466 | [PDF](https://arxiv.org/pdf/2606.04466v1)

**作者:** Chongyang He `[一作]` (Tsinghua University), Xin Li `[通讯]` (University Of Electronic Science And Technology Of China)

**通讯引用:** 57104 | [OpenAlex ID](https://openalex.org/A5100353880)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个难度感知的SFT-then-RL框架，协调SFT与RL阶段的数据分配，并通过Bridge机制将难样本转换为易学监督，再通过错误回收将RL中所有零奖励失败转化为SFT监督，实现迭代循环；

**💡 创新点**

创新点在于将样本难度与SFT/ RL功能对应，提出Bridge step-level transformation（保留、压缩、扩展、丢弃、局部化）将长难CoT转为适合SLM的监督，并将RL中所有零奖励失败回收成诊断、修复与新推理轨迹的SFT标签；

**🔧 技术方法**

采用SFT+RL两阶段训练，RL使用GRPO强化学习；Bridge机制对难样本逐步进行重要性、跳跃性、难度评估后选取合适操作；Critique Fine‑Tuning将教师诊断转化为监督；使用pass@k、规则验证器等评价工具；

**📊 数据集**

主要使用GSM8K训练集进行post‑training，评估基准包括GSM8K‑Platinum、MAWPS、SVAMP、MATH500、LogiQA五个数学/逻辑推理数据集；

**📈 对比分析**

对比Std‑CoT、STaR、PRewrite、GRPO、Vanilla‑KD、DURIT等基线，使用答题准确率作为指标；在两款SLM（Qwen2.5‑0.5B、Llama3.2‑1B）上平均提升约4–6%，在大多数基准上均优于所有对比方法；

**⚠️ 局限性**

局限性：仅在小型SLM的数学/逻辑推理任务上验证，训练数据仅来自GSM8K；Bridge与错误回收依赖教师模型与判定，带来额外计算成本和潜在偏差，未验证大模型或非推理场景的适用性。

---

## 274. Listening to the Workforce: Measuring Construction Worker Safety Attitudes from Social Media Discourse Using LLMs

**arXiv ID:** 2606.04450 | [PDF](https://arxiv.org/pdf/2606.04450v1)

**作者:** Farouq Sammour `[一作]` (Texas A&M University), Zhenyu Zhang `[通讯]` (Texas A&M University)

**通讯引用:** 79228 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建并验证了Construction Safety Attitude Framework (CSAF)，用于从工人自然语言对话中测量建筑工人的安全态度。

**💡 创新点**

创新点在于将多种安全态度理论整合成八维度框架，并通过LLM实现大规模、可重复的自然语言编码。

**🔧 技术方法**

使用大语言模型 GPT-OSS-120B 与 TIDD‑EC 提示工程对帖子和评论进行自动编码。

**📊 数据集**

数据集来自 Reddit 两个子板块 r/Construction（开发、验证）和 r/Roofing（转移、案例研究）的帖子和评论，合计约 10,600 条记录。

**📈 对比分析**

与人工专家标注对比，Cohen κ 0.90/0.89、精确率≈0.98、召回率≈0.97，显示与人类一致性极高；跨域转移保持了同等性能。

**⚠️ 局限性**

局限包括：样本自选偏差、仅覆盖认知评价维度、未识别态度驱动因素、缺乏干预实验验证。

---

## 275. MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation

**arXiv ID:** 2606.04513 | [PDF](https://arxiv.org/pdf/2606.04513v1)

**作者:** Deguo Xia `[一作]` (Tsinghua University), Diange Yang `[通讯]` (Tsinghua University)

**通讯引用:** 2355 | [OpenAlex ID](https://openalex.org/A5009072257)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出并实现了 MapAgent，一套工业级的 agentic 框架，用于城市规模的车道级地图生成和更新；它在已有 BEV 向量化 backbone 的基础上，通过判断（Judge）、规划（Planner）与工作（Worker）循环，对低置信度区域进行自动化、符合规范的修正。

**💡 创新点**

创新点包括：① 将端到端向量化的“草稿”输出与可验证的 agentic 循环相结合，实现可解释、可回溯的自动化修正；② 设计了基于视觉语言模型的 Judge，能够给出结构化错误诊断与证据；③ 引入受限工具集和预算限制，保证修正过程安全、可控；④ 通过质量过滤器只在低置信度 tile 上触发循环，保持高吞吐量。

**🔧 技术方法**

技术方案主要包括：基于 BEV 的向量化 backbone（GeMap、DuMapNet 等）；视觉语言模型（Qwen3-VL-Thinking 等）通过 SFT+GRPO 进行判别器训练；Judge-Planner-Worker 三阶段循环；工具箱包含删除、类别修正、平滑、局部重建；整体使用 LoRA 微调、GRPO 强化学习和可解释日志。

**📊 数据集**

使用了百度地图数据库构建的大规模车道级向量地图（DuLD），并抽取了难度子集；测试集包含 656 张 BEV 图像与 10,254 条真实车道。

**📈 对比分析**

在 GeMap 与 DuMapNet 两个 backbone 上做对比，MapAgent 在准确率、F1、精确率、召回率等指标均有显著提升，尤其在长尾、复杂场景下，误检率下降、类别正确率提升；在实际生产中已覆盖 360+ 城市，整体自动化率突破 95%。

**⚠️ 局限性**

限制：对极端视觉模糊或缺失标记的场景仍需人工介入；当前只对已有草稿进行局部修正，无法主动添加新车道或进行大范围拓扑变更；在极端不确定情况下的错误推断可能导致不安全修正。

---

## 276. Global Sketch-Based Watermarking for Diffusion Language Models

**arXiv ID:** 2606.04486 | [PDF](https://arxiv.org/pdf/2606.04486v1)

**作者:** Daniel Zhao `[一作]` (Harvard University), Daniel Zhao `[通讯]` (Harvard University)

**通讯引用:** 78 | [OpenAlex ID](https://openalex.org/A5081284793)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对掩码扩散语言模型（masked diffusion language model）的全局向量化水印方案。该方案通过对生成文本的 Count‑Sketch 统计进行梯度引导的指数偏移，实现了对整个句子向量的控制，从而实现水印的嵌入与检测。

**💡 创新点**

创新点在于：
- 用全局可加的草图（sketch）替代传统的局部上下文偏置，解耦了检测与生成过程；
- 设计了基于梯度的指数倾斜机制，使得水印不会以单一词表偏差的形式出现，提升了鲁棒性；
- 提供了 KL 失真链式规则、Fisher 最优分配证明以及对编辑敏感度的上界分析，为水印设计提供了理论保障。

**🔧 技术方法**

技术手段包括：
- Count‑Sketch 作为文本草图；
- Rademacher 随机方向和范数惩罚的检测函数；
- 通过梯度计算得到的残差方向对每个掩码位置的边际分布进行指数倾斜；
- KL 失真分析与 Fisher 信息分配；
- 关键随机化的显著性分析与编辑敏感度估计。

**📊 数据集**

论文中未给出具体实验数据集，主要侧重理论分析；若有实验，则假定使用常见的自然语言文本语料（如 WikiText‑103、Papers‑100M 等）。

**📈 对比分析**

与传统的红绿偏置（red‑green）等自回归水印方法相比，作者通过理论阈值和实验模拟展示其在失真、检测灵敏度和鲁棒性方面均能保持或略优于现有方案，且不依赖于生成顺序。

**⚠️ 局限性**

局限性包括：
- 仅针对掩码扩散模型，尚未验证对连续扩散或其他架构的适用性；
- 对短文本或高编辑量的鲁棒性仍有限；
- 参数调优（如 λ、γ、η 调度）较为复杂；
- 论文缺乏大规模实验评估，主要依赖理论证明。

---

## 277. Off-Distribution Voices: Fanfiction Subgenres as Universal Vernacular Jailbreaks for Aligned LLMs

**arXiv ID:** 2606.04483 | [PDF](https://arxiv.org/pdf/2606.04483v1)

**作者:** Zhongze Luo `[一作]` (Chinese University of Hong Kong (Shenzhen)), Xiaoying Tang `[通讯]` (Chinese University of Hong Kong (Shenzhen))

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了一种基于真实粉丝小说子类型（AO3）的注册式 jailbreak 攻击，通过五句示例触发模型写出包含有害行为的情节，并在单轮和四轮对话中实现对已对齐 LLM 的逃避。

**💡 创新点**

创新点在于：①将完整的自然写作注册（而非单一模板）作为攻击载体；②使用仅五句公开粉丝小说示例即可构造攻击的 meta；③证明注册比长度或结构更能突破安全过滤；④提出无攻击 LLM 的四轮静态对话链，显著提升攻击成功率。

**🔧 技术方法**

技术手段包括：五句 AO3 条件 meta、12 种粉丝小说注册、7 种结构叠加（如诗歌、嵌套小说等）、多轮对话流水线、四评判员聚合（基于 HarmBench‑13B、LlamaGuard‑3‑8B、WildGuard‑7B、GPT‑5.4‑mini）。

**📊 数据集**

使用的数据集：AO3（12 个子类型的公开段落）、HarmBench（200 条有害行为）和 JailbreakBench（90 条）。

**📈 对比分析**

评估方法：以四评判员的二选一投票作为攻击成功判定，计算攻击成功率（ASR）。结果显示：单轮注册攻击平均 ASR 从 0.278 提升至 0.731，提升约 3.11 倍；四轮攻击进一步达到 0.924，超过现有三种多轮方法；在不同模型、不同基准上均保持高成功率。

**⚠️ 局限性**

局限性包括：多轮实验仅在单一注册（剧本）上验证，未测试其在其他 11 种注册中的泛化；未涉及多语言注册；评判员在边缘文学输出上可能存在共享盲点；所有结果基于实验时点的模型版本，后续更新可能改变结论。

---

## 278. IMPose: Interactive Multi-person Pose Estimation with Dynamic Correction Propagation

**arXiv ID:** 2606.04480 | [PDF](https://arxiv.org/pdf/2606.04480v1)

**作者:** Haoyang Ge `[一作]` (Tianjin University), Kun Li `[通讯]` (Tianjin University)

**通讯引用:** 75734 | [OpenAlex ID](https://openalex.org/A5100368058)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一款名为IMPose的交互式多人人体姿态标注工具，能够通过极少量点击将稀疏的手工修正传播到整段视频，实现高精度、连贯的姿态标注。

**💡 创新点**

创新点包括：①双层跟踪机制，将关键点级别的时间纠正传播与实例级的关键点感知嵌入相结合，实现跨帧身份一致性；②利用相对位置编码与傅里叶特征强化姿态几何表达；③构建轨迹库，提升长时段关联和对遮挡、运动模糊的鲁棒性。

**🔧 技术方法**

技术核心包括DETR式Transformer框架、Deformable Transformer编码器、CoTracker点跟踪、关键点感知嵌入（KP‑AE）、相对位置编码、傅里叶特征、ID解码器等。

**📊 数据集**

在公开视频姿态数据集3DPW和PoseTrack21上进行评估，并使用IMPose扩展后的PoseTrack21（3.55M关键点、约190K人物）进一步验证效果。

**📈 对比分析**

与AlphaPose、DSTA、Click‑Pose、X‑AnyLabeling等自动或半自动方法对比，IMPose在0–3次点击下mAP提升30+点、HOTA提升10+点，点击量极低（3DPW 27次/1050帧；PoseTrack21 3次/84帧），平均标注耗时比AlphaPose*和Click‑Pose快约4倍和3倍，显著提升标注效率和质量。

**⚠️ 局限性**

在极长时间间隔、长时间消失或高密度人群场景下仍可能出现性能下降；目前仅支持2D全身姿态标注，未覆盖3D或全身深度信息，且对基础视觉模型的精度有一定依赖。

---

## 279. TransTac: Visuo-Tactile Modality Transition via Ultraviolet-Encoded Transparent Elastomers

**arXiv ID:** 2606.04477 | [PDF](https://arxiv.org/pdf/2606.04477v1)

**作者:** Lingyue Yang `[一作]` (Beijing University of Posts and Telecommunications), Bin Fang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 5066 | [OpenAlex ID](https://openalex.org/A5029662229)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款透明紫外编码双目视觉触觉传感器 TransTac，融合视觉透明度、UV 标记检测、基于 Delaunay 三角匹配的立体视觉和 RGB‑D 融合，实现从远距离视角到接触点的连续几何感知。

**💡 创新点**

创新点包括：①将透明弹性体与 UV 可反射标记结合，保持视觉透明度；②提出轻量化半透明标记检测网络和基于先验的 Delaunay 匹配算法，提高立体对应的鲁棒性；③将稀疏标记三角化与稠密深度融合，实现接近触碰区域的几何恢复；④在触觉图像上实现零样本语义识别，显著提升语义保留。

**🔧 技术方法**

采用 UV 编码的半透明弹性体、双目 USB 摄像头、轻量化 Anchor‑free 检测器、ByteTrack 跟踪、基于 epipolar 先验的 Delaunay 三角匹配、稀疏三角化、FoundationStereo 深度网络、Umeyama 相似变换校准等技术。

**📊 数据集**

使用 3D 打印对象的 STL 模型做真实几何基准；针对语义评估使用 6 个类别（鸡蛋、硬币、电池、乐高块、按钮、玻璃珠）共 36 张触觉图像；对 RGB‑D 性能评估使用 Intel RealSense D405 等深度相机。

**📈 对比分析**

与传统不透明 VBTS（GelSight、9DTact）和 RGB‑D 深度估计模型（VGGT、FoundationStereo）对比。TransTac 在零样本识别上达到 83.3%（相对 GelSight 仅 30%），DINOv2 中心相似度从 0.2 提升至 0.774；Delaunay 匹配平均正确匹配 90.8 个，超过 Hungarian 74.9；近距离深度有效率在 <9 cm 时下降至 10% 时，稀疏三角化仍保持约 2.44 mm 误差，显示优越的几何恢复能力。

**⚠️ 局限性**

局限性：未实现力/压力测量；需人工标注的 UV 标记训练数据；现有深度估计模型在极近距离下可能产生尺度误差；硬件依赖现成摄像头与灯光，尺寸和集成度有限；UV 荧光标记的长期耐久性尚未系统评估。

---

## 280. Entity Binding Failures in Speech LLM Reasoning: Diagnosis and Chain-of-Thought Intervention

**arXiv ID:** 2606.04474 | [PDF](https://arxiv.org/pdf/2606.04474v1)

**作者:** Ming-Hao Hsu `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了语音大语言模型（SLLMs）在复杂推理任务中相较文本模型的模态差距，诊断出实体绑定失败是主要瓶颈，并提出了实体感知链式推理（EA‑CoT）来弥补该差距；

**💡 创新点**

首次将实体绑定问题细化为模态特定瓶颈，并通过在推理前强制显式列举实体的 EA‑CoT 提示实现性能提升；

**🔧 技术方法**

采用 EA‑CoT 提示、token 预算控制、结构化链式推理、消融实验，并在 Qwen2.5‑Omni 与 Phi‑4‑Multimodal 两大 SLLM 上验证；

**📊 数据集**

使用 VoiceBench BBH 语音-文本对照集（包括 web‑of‑lies、navigate 等四类）以及 MMSU 声学重现数据集进行实验；

**📈 对比分析**

通过对比 S2T 与 T2T 在各任务类别的准确率，发现 EA‑CoT 在需要实体追踪的逻辑任务中将 S2T 效果提升至与 T2T 相近，最高提升 24.4pp（例如在 web‑of‑lies 上从 0% 近似随机提升至 95%+）；

**⚠️ 局限性**

局限性包括显著增加的推理延迟（生成 1,024 代替 256 代），仅在合成 TTS 语音与 7B 规模模型上验证，实际噪声环境、语音真实性、以及更大模型可能导致性能变化；

---

## 281. ChessMimic: Per-Rating Transformer Models for Human Move, Clock, and Outcome Prediction in Online Blitz Chess

**arXiv ID:** 2606.04473 | [PDF](https://arxiv.org/pdf/2606.04473v1)

**作者:** Thomas Johnson `[一作]` `[通讯]` (Nascent), Thomas Johnson (Nascent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在公开的 Lichess 速战数据上训练并部署了一套包含移动预测、思考时间预测和结果预测的三种独立 encoder‑only Transformer，且每个 100 Elo 分段都有单独的模型。

**💡 创新点**

创新点包括：对每个 Elo 段单独训练模型以获得更细粒度的技能校准；将玩家评分、剩余时间与增量直接嵌入 Transformer 输入；使用 3‑分类 W/D/L 胜负模型并以 Brier 损失训练；以及通过常见走法数据库实现低延迟的部署。

**🔧 技术方法**

技术栈为 8 层 256 维 encoder‑only Transformer（8 头自注意 + SwiGLU MLP），使用 bfloat16 训练、Adam + OneCycleLR 微调、Brier 损失、C++ 预处理管道以及常见走法跳过与 Blitz 过滤。

**📊 数据集**

数据集为 2024‑09 至 2025‑08 的每月 Lichess Rated Blitz PGN 转成的记录；验证集为 2025‑09，测试集为 2026‑04；另外使用 Allie 2022 Blitz 切片进行交叉验证。

**📈 对比分析**

在相同 Held‑out 切片上以 Top‑1/3/5 位置匹配率、Brier、AUC 等指标与 Maia‑2、Maia‑3 进行直接对比；ChessMimic 在所有 Elo 段均优于 Maia‑2，平均 Top‑1 提升 3.6pp；结果预测 Brier 降至 0.184、AUC 0.777，思考时间相关性 Pearson 0.41。

**⚠️ 局限性**

局限性：未实现统一骨干多头模型或搜索组件，未加入 Geometric Attention Bias；仅针对 Blitz 时间控制；思考时间预测相关性低于 ALLIE；跨段信息共享与更大模型规模的评估仍待探索。

---

## 282. ChannelTok: Efficient Flexible-Length Vision Tokenization

**arXiv ID:** 2606.04461 | [PDF](https://arxiv.org/pdf/2606.04461v1)

**作者:** Sukriti Paul `[一作]` (University of Maryland), Tom Goldstein `[通讯]` (University of Maryland)

**通讯引用:** 14082 | [OpenAlex ID](https://openalex.org/A5060687985)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在通道维度上进行可变长度视觉分词的轻量化方法，能够在不同的通道数量下自适应压缩和重建图像，并支持可变长度自回归生成；

**💡 创新点**

创新点在于将通道视为视觉词条并引入随机前缀遮蔽机制，使得通道天然形成从粗到细的语义层次，既避免了复杂多步生成解码器，又实现了高效可变长度压缩；

**🔧 技术方法**

采用轻量化VQGAN风格的编码器-解码器结构，通道级二值球面量化（BSQ）进行离散化，并通过随机前缀遮蔽和停止梯度训练实现通道顺序；

**📊 数据集**

在ImageNet‑1K数据集上进行训练与评估；

**📈 对比分析**

与多种基准（FlexTok、OneDPiece、DOVE、ALIT、KARL、LlamaGen等）在相同token预算下对比，取得rFID 2.92（最高）、解码速度8.6×提升、参数量仅159M，比最优方法小2.1×、速度最快；

**⚠️ 局限性**

主要限制在低token预算下的重建质量仍有提升空间，且方法仍依赖完整的编码器-解码器对齐，未来可进一步探索任务特定的通道选择策略与更高效的量化方案。

---

## 283. SAILRec: Steering LLM Attention to Dual-Side Semantically Aligned Collaborative Embeddings for Recommendation

**arXiv ID:** 2606.04514 | [PDF](https://arxiv.org/pdf/2606.04514v1)

**作者:** Xi Wu `[一作]` (Northeastern University), Yifei Zhang `[通讯]` (Northeastern University)

**通讯引用:** 3533 | [OpenAlex ID](https://openalex.org/A5100386920)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于LLM的推荐模型SAILRec，解决协同嵌入在LLM中被充分利用的问题

**💡 创新点**

通过双侧语义对齐使用户和物品协同嵌入与LLM语义空间兼容，并引入分层注意力调度在Transformer层级上精准控制协同嵌入的使用时机

**🔧 技术方法**

使用矩阵分解生成协同嵌入，轻量化Q-Former实现用户/物品侧映射，对齐采用InfoNCE；在LLM内部通过偏置矩阵实现浅层抑制、中层无调度、深层增强的分层注意力调度；训练分为三阶段，最终使用LoRA进行任务微调

**📊 数据集**

在MovieLens‑1M和Amazon‑Book两大公开数据集上进行实验

**📈 对比分析**

与协同过滤模型、纯LLM推荐器和多种协同增强LLM基线对比，SAILRec在AUC、UAUC、NDCG、MAP四项指标上均超过所有基线，提升幅度约1–2个百分点

**⚠️ 局限性**

采用固定的注意力调度方案，可能不适用于所有样本或LLM；协同嵌入的语义可解释性有限，仍为连续向量

---

## 284. Homology-Preserving Dimensionality Reduction via Adaptive Mapper and Landmark Isomap

**arXiv ID:** 2606.04464 | [PDF](https://arxiv.org/pdf/2606.04464v1)

**作者:** Shakiba Khourashahi `[一作]` (Iowa State University), Lin Yan `[通讯]` (Iowa State University)

**通讯引用:** 2408 | [OpenAlex ID](https://openalex.org/A5100429585)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于Mapper的自适应框架AdaHIsomap，用以在维度降低过程中保留0D和1D同调结构。

**💡 创新点**

创新点在于：①利用持久同调自动分割滤波函数，生成局部自适应覆盖；②将Mapper骨架作为Landmark Isomap的拓扑感知landmark，并加入随机anchor平衡0D结构；③完全自动化参数，无需人工调节。

**🔧 技术方法**

主要技术包括持久同调（Persistence Diagram）、Mapper算法、KNN图与DTB滤波、Isomap与L-Isomap、DBSCAN聚类以及Wasserstein距离评估。

**📊 数据集**

实验涵盖点云（3D与高维）、科学模拟、网络和图像等11个数据集，如Swiss Roll、Glass、Vortex Street、Cartoon、Face3D等。

**📈 对比分析**

与t‑SNE、UMAP、Isomap、TopoAE++等方法比较，AdaHIsomap在0D/1D同调保留（PDW^0、PDW^1）上均优于大多数基线，并在几乎所有数据集上获得最低RMSE；同时保持了自动化和可复现性。

**⚠️ 局限性**

局限性包括：①对单一流形数据依赖，无法直接处理多流形或大规模数据；②仍受KNN图和Isomap的线性MDS限制，导致某些并行或交叉循环难以保持；③参数选择虽然自动，但基于经验，缺乏严格理论保证。

---

## 285. A Normative Intermediate Representation for ASP-Based Compliance Reasoning

**arXiv ID:** 2606.04619 | [PDF](https://arxiv.org/pdf/2606.04619v1)

**作者:** Yangfan Wu `[一作]` (Hong Kong University of Science and Technology), Jianmin Ji `[通讯]` (University of Science and Technology of China)

**通讯引用:** 1343 | [OpenAlex ID](https://openalex.org/A5061104105)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MONIR 框架，将法规文本抽取为可执行的 ASP 规则，实现合规性检查；

**💡 创新点**

设计了受限的中间语言 MONIR‑core，支持分阶段语义、状态化输出、非递归与可解释性，并实现可编译为 MONIR‑ASP；

**🔧 技术方法**

结合 LLM 辅助文本抽取、Answer Set Programming、模块化与增量求解、外部函数/时间推理以及三值评估器等技术；

**📊 数据集**

以中国高级驾驶辅助系统（ADAS）法规与标准为数据集，利用 LLM 从规范文本中提取规则；

**📈 对比分析**

通过实验比较规则提取质量和模块化/增量 ASP 求解效率，结果表明相较传统一次性求解更快，抽取准确率显著；

**⚠️ 局限性**

受限于无循环状态、无后置豁免、表达式有限，无法处理嵌套模态或不完整规范；LLM 抽取误差会影响推理结果，且系统非单调，需要手工更新。

---

## 286. When Firms Learn to Game the Rules

**arXiv ID:** 2606.04617 | [PDF](https://arxiv.org/pdf/2606.04617v1)

**作者:** Xufeng He `[一作]` `[通讯]` (PERMITFOLIO Regulation Research Institute), Xufeng He (PERMITFOLIO Regulation Research Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过构建基于代理的强化学习模拟，研究了可计算规则对公司边界搜索与监管反应的影响。

**💡 创新点**

提出了计算可计算规则与企业行为之间的边界质量区分，并展示了预算中性反作弊设计在降低危害和边界聚集方面的效果。

**🔧 技术方法**

采用基于 Q‑learning 的代理学习、规则可计算性模拟、拉丁超立方抽样、配对签名翻转检验与事件研究等技术。

**📊 数据集**

使用完全合成的模拟数据：150 种种子运行、378 个共随机数扫描、288 个拉丁超立方设计、共 2,880,000 行企业周期面板。

**📈 对比分析**

通过对比不同监管方案（模糊静态、可计算静态、可计算自适应、RL 调节、反作弊自适应）并进行配对统计检验、敏感性分析和政策前沿，发现可计算规则提升了信号与实际边界聚集，适应性规则降低了消费者危害但增加了规则波动，反作弊设计在预算保持的情况下进一步降低了危害和边界聚集。

**⚠️ 局限性**

局限在于模型过度简化（离散动作、简单 Q‑learning、无司法审查等），并且使用的是合成数据，缺乏对真实司法环境与行业细节的校准。

---

## 287. Selectivity Estimation for Semantic Filters on Image Data

**arXiv ID:** 2606.04610 | [PDF](https://arxiv.org/pdf/2606.04610v1)

**作者:** Matthias Urban `[一作]` (Technical University of Darmstadt), Carsten Binnig `[通讯]` (Technical University of Darmstadt)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出Semantic Histograms，用于估计图像语义过滤器的选择性并将其应用于查询优化；

**💡 创新点**

创新点在于将共享嵌入空间视为范围查询，针对过滤器特异度提出了特异度模型和压缩KV缓存批处理两种阈值估计方法，并通过集成提升估计鲁棒性；

**🔧 技术方法**

使用的技术包括CLIP/SigLIP等视觉-文本共享嵌入、余弦距离阈值、轻量化特异度神经网络、压缩KV缓存批处理以及两者的集成；

**📊 数据集**

实验采用三个公开数据集：Artwork（1000张图）、Wildlife（1000张图）和E‑Commerce（1000张图）；

**📈 对比分析**

与传统采样的在线配置相比，Semantic Histograms在Q‑Error低、估计延迟显著降低；集成方案在查询优化中可将端到端运行时间提升高达86%，并在所有数据集上表现更稳健；

**⚠️ 局限性**

限制包括：特异度模型依赖ImageNet，可能在非ImageNet领域泛化差；压缩KV缓存可能引入精度损失；在低精度VLM或高度复杂图像时，方法性能可能下降。

---

## 288. CapSenseBand: Sustaining Cross-Disciplinary Creativity When Stitches Must Meet Signals

**arXiv ID:** 2606.04609 | [PDF](https://arxiv.org/pdf/2606.04609v1)

**作者:** Sark Pangrui Xing `[一作]`, Stephen Jia Wang `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

通过跨学科协作开发了一款可穿戴的编织电容传感手环 CapSenseBand，采用从材料试样、快速可穿戴原型、纸模型、双层针织结构到隔离频率扫描电容传感电路的迭代过程。

**💡 创新点**

创新点在于将纸模型作为边界对象，桥接纺织与交互设计的语言差异，并提出材料中心 HCI 的分阶段协作流程，提供可复用的样本到袖套的设计链。

**🔧 技术方法**

使用了扫频电容传感（SFCS）技术、银镀线针织电极、低功耗电路板、SDS-ONE APEX 3 针织仿真软件以及手工 PCB 隔离等技术手段。

**📊 数据集**

未使用公开数据集，全部使用实验室自制原型进行验证。

**📈 对比分析**

论文未进行系统的对比实验或性能评估，仅通过原型验证功能性与舒适度，没有量化的性能指标。

**⚠️ 局限性**

局限包括：纸模型虽能桥接意图，但缺乏术语解释，跨领域术语仍需交流；仅在单一手环项目中验证，未检验方法在其他项目中的可推广性；硬件仍需进一步优化。

---

## 289. COMBINER: Composed Image Retrieval Guided by Attribute-based Neighbor Relations

**arXiv ID:** 2606.04604 | [PDF](https://arxiv.org/pdf/2606.04604v1)

**作者:** Zixu Li `[一作]` (Shandong University), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 30051 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 COMBINER，专为组合式图像检索（CIR）任务设计的端到端网络，通过属性原型引导的邻域关系学习，提升检索准确率。

**💡 创新点**

创新点在于：①引入属性原型的跨模态统一表示（CUP）以区别视觉相似但属性不一致的负样本；②自适应语义解耦模块（ASD）能在无显式标签的情况下动态抽取属性原型；③双关系建模（DRM）结合有监督的配对关系与无监督的邻域关系，利用KL一致性正则化进一步优化度量空间。

**🔧 技术方法**

技术手段包括：CLIP 视觉-语言预训练特征提取、Semantic Attribute Attention (SAA) 实现属性解耦、统一原型组合、K-means 聚类构造语义中心、基于余弦相似度的交叉熵损失与KL一致性损失的多任务优化。

**📊 数据集**

使用三个公开基准数据集：Shoes、FashionIQ（时尚域）和 CIRR（开放域）进行实验评估。

**📈 对比分析**

与最新 SOTA 方法对比（在相同 CLIP/BLIP-2 backbone 上），COMBINER 在 R@k、R_subset@k、平均 R@10/R@50 等指标均位列榜首，提升幅度约 1–3% 以上；在推理效率上，单样本推理时间仅 0.025 s，显著快于需要外部模型的基线。

**⚠️ 局限性**

局限性包括：①对属性原型数量和聚类数的超参数敏感，需在不同数据集上进行调优；②在极度属性耦合或视觉特征主导的场景下，ASD 可能难以完全分离属性；③存在因数据标注不完整导致的假负样本，影响 R@k 评估。

---

## 290. A Systematic Evaluation of Positional Bias in Multi-Video Summarization with MLLMs

**arXiv ID:** 2606.04596 | [PDF](https://arxiv.org/pdf/2606.04596v1)

**作者:** Huangchen Xu `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**通讯引用:** 11816 | [OpenAlex ID](https://openalex.org/A5029392006)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多视频摘要中的位置偏差，构建基准并评估多模态大语言模型的表现。

**💡 创新点**

提出Coverage、DPB和MEG三种评价指标，揭示模型在不同域、输入规模和视频时长下的非单一偏差模式。

**🔧 技术方法**

使用多模态大语言模型（如InternVL3.5、Qwen3-VL、Gemini-3.1-Pro、GPT-5.4等）进行生成，并结合LLM评判、片段覆盖、语义相似度对摘要进行评估。

**📊 数据集**

基于ActivityNet和News Video Dataset，涵盖Cooking、Domestic、Leisure和News四类场景，分短视频/长视频两种时长。

**📈 对比分析**

与九个公开/专有模型对比，发现位置偏差依赖模型和域；DPB与MEG指标揭示中位位置弱点，单一指标难以捕捉；通过提示干预和视觉预算等实验探讨缓解效果。

**⚠️ 局限性**

局限包括仅评估视频数≤4、时长≤2分钟、依赖人工参考、缺乏机制性分析和训练时干预，未覆盖更长列表或开放域视频。

---

## 291. Synthetic Personalities: How Well Can LLMs Mimic Individual Respondents Using Socio-Economic Microdata?

**arXiv ID:** 2606.04592 | [PDF](https://arxiv.org/pdf/2606.04592v1)

**作者:** Leonard Kinzinger `[一作]` (Technical University of Munich), Jochen Hartmann `[通讯]` (Technical University of Munich)

**通讯引用:** 137213 | [OpenAlex ID](https://openalex.org/A5118899500)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并评估基于大型语言模型的个体级数字孪生，用德国社会经济面板（SOEP）数据从现有异构面板中提炼个体特征，形成可在营销研究中直接使用的虚拟受访者。

**💡 创新点**

创新点在于首次证明利用已有的异构面板数据即可构建高质量的个体孪生，并系统比较了三大构建维度（模型、信息深度、嵌入方式、推理模式）对孪生性能的影响，揭示出信息深度到 75% 纯熵四分位时的成本效益拐点。

**🔧 技术方法**

技术主要使用三种开源 LLM（Qwen‑3、Gemma‑4、Ministral‑3），并实现两种嵌入方法（Chain‑of‑Density 个人摘要与原始问答对话），两种推理模式（直接回答与思考链），以及基于归一化香农熵的逐项信息深度排序。

**📊 数据集**

使用数据集为德国社会经济面板（SOEP）2023 版本，包含 16,055 名受访者的 949 条问答对，其中 500 名受访者用于构建孪生，剩余 183 条为保留评估问题。

**📈 对比分析**

方法通过 3×5×2×2 的 60 细胞构建方法网格，对 2.1 万万条孪生回答进行准确率、秩相关与方差比例评估；最佳配置（Gemma‑4 对话‑思考‑100%）达 78.8% 准确率，Qwen‑3 对话‑思考‑100% 达 0.590 秩相关，整体性能与专门收集的数据孪生相当，且信息深度 75% 就能捕获约 90% 的提升。

**⚠️ 局限性**

局限包括仅在德国面板上验证，无法直接迁移至欧美消费领域；未对封闭源顶尖 LLM 进行基准；缺少对意识形态漂移、人格诱导偏差等更细粒度失真模式的深入分析。

---

## 292. Multi-SPIN: Multi-Access Speculative Inference for Cooperative Token Generation at the Edge

**arXiv ID:** 2606.04581 | [PDF](https://arxiv.org/pdf/2606.04581v1)

**作者:** Haotian Zheng `[一作]` (University of Hong Kong), Kaibin Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 22926 | [OpenAlex ID](https://openalex.org/A5007131492)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究分布式SPIN在多用户边缘系统中的应用，提出了Multi‑SPIN架构并设计了联合草稿长度与上行带宽分配的最优控制算法。

**💡 创新点**

创新点在于：①首次在多用户场景下引入多访问草稿控制，将草稿长度和带宽统一优化；②利用分解技巧获得闭式解，显著降低求解复杂度；③允许异质草稿长度，并给出二维搜索的高效实现。

**🔧 技术方法**

技术手段包括：使用小语言模型（SLM）生成草稿、基于大型语言模型（LLM）的批量验证；采用OFDMA多路访问；用Lambert W函数解析最优草稿长度；通过分解优化将混合整数非线性问题转化为可闭式求解的子问题。

**📊 数据集**

实验数据集涵盖四类任务：MBPP+（代码生成）、GSM8K（数学推理）、MT‑Bench（多轮对话）以及SQuAD（阅读理解），用于评估不同模型对接的接受率和goodput。

**📈 对比分析**

与P2P‑SPIN、Cen‑SPIN、固定/均匀草稿长度与带宽的基线以及均匀带宽的Multi‑SPIN在不同总带宽和设备数场景下进行对比；结果显示，Heterogeneous‑Multi‑SPIN在宽带受限时可提升高达88% goodput，且在设备规模增大时优势进一步扩大。

**⚠️ 局限性**

局限性包括：仅考虑OFDMA多路访问，未探索其他多路访问方案；主要关注总goodput，未对公平性或实时性进行约束；接受率假设为i.i.d.并需预估，算法对搜索网格选择敏感。

---

## 293. Neetyabhas: A Framework for Uncertainty-Aware Public Policy Optimization in Rational Agent-Based Models

**arXiv ID:** 2606.04562 | [PDF](https://arxiv.org/pdf/2606.04562v1)

**作者:** Janani Venugopalan `[一作]` (Thoughtworks Technologies), Jayanta Kshirsagar `[通讯]` (Thoughtworks Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文通过构建基于BharatSim框架的传染病仿真环境，使用强化学习对公共政策进行优化，并在不确定性下实现个体和决策者的理性决策；

**💡 创新点**

创新点在于将强化学习与理性代理的传染病仿真相结合，提出了面向不确定性环境的公共政策决策方法，并通过实验验证了其有效性；

**🔧 技术方法**

采用的技术包括Agent‑Based Modeling（BharatSim）、强化学习（RL）算法、理性代理模型和不确定性感知决策框架；

**📊 数据集**

使用的数据集为BharatSim生成的合成流行病模拟数据，没有使用外部真实数据；

**📈 对比分析**

实验中将RL驱动的政策与传统规则/启发式政策进行对比，结果显示RL策略在降低感染率、成本和提高政策可信度方面表现更佳；

**⚠️ 局限性**

局限性包括仅基于仿真数据，缺乏真实世界验证；模型假设与现实复杂性不完全匹配；以及RL算法在大规模仿真中的计算开销较大。

---

## 294. Beyond Symmetric Alignment: Spectral Diagnostics of Modality Imbalance in Vision-Language Models in the Medical Domain

**arXiv ID:** 2606.04613 | [PDF](https://arxiv.org/pdf/2606.04613v1)

**作者:** Alessandro Gambetti `[一作]` (NOVA School of Science and Technology), Hong Shen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3550 | [OpenAlex ID](https://openalex.org/A5062298555)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出并评估了Spectral Alignment Score（SAS）以诊断医学图像文本对中的跨模态对齐失衡，揭示医学图像在对齐中具有更丰富的结构信息。

**💡 创新点**

创新点在于SAS的方向性谱分解，能够分离图像与文本在主成分空间中的可恢复性差异，并提供差异量Δ_SAS来量化模态信息不平衡。

**🔧 技术方法**

采用特征中心化、协方差谱分解、特征值加权相关计算以及传统对齐指标（CKA、SVCCA、CORAL、MMD、RMG）与检索评估（R@1/5/10）进行对照。

**📊 数据集**

使用自然域的MS-COCO 2014和Flickr30k，以及医学域的ROCO和MIMIC-CXR四个图像-文本数据集，涵盖多种医学影像类型。

**📈 对比分析**

通过在四个实验中与15个双编码VLM（包括13个通用模型和2个医学专用模型）进行对齐与检索性能的Spearman相关性比较，结果显示SAS在医学域对检索性能的预测相关性超过其他指标（ρ>0.96），并能揭示显著的方向性不平衡。

**⚠️ 局限性**

局限性包括：需保证图像与文本特征维度相同；在MIMIC-CXR上即使对齐高也无法实现检索；SAS对谱深度参数q敏感；对高维特征的谱分解算力需求较高；对齐与检索关系在自然域表现不一致。

---

## 295. Hybrid Adversarial Defence for Natural Language Understanding Tasks

**arXiv ID:** 2606.04612 | [PDF](https://arxiv.org/pdf/2606.04612v1)

**作者:** Manar Abouzaid `[一作]` (University of Southampton), Stuart E. Middleton `[通讯]` (University of Southampton)

**通讯引用:** 2835 | [OpenAlex ID](https://openalex.org/A5088719080)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种混合对抗防御框架，将熵基、置信度基和几何基三类专家模型结合，能够在自然语言理解任务中同时降低幻觉并提高对抗鲁棒性。

**💡 创新点**

创新点在于首次将熵、置信度和几何特征联合用于路由选择，实现对抗与幻觉双重防御，并通过可训练的选择器实现实例级专家切换。

**🔧 技术方法**

使用了熵检测、Refusal-Aware Instruction Tuning、PuRe主成分去除等技术，并设计了ML路由和LWV软聚合两种聚合策略。

**📊 数据集**

评估数据集包括 FEVER、HotpotQA、CSQA、SIQA 等 NLU 领域，以及 AeroEngQA、CPIQA 的 OOD，SafeGuard、AdvBench、DAN 的注入/越狱攻击。

**📈 对比分析**

与 FreeLB、ProtectAI v1、NeMoGuard 等基线比较，混合模型在清洁数据上提升 8–13% 准确率，在对抗攻击下提升 3–19% 准确率、并将攻击成功率下降 13–20%；在 OOD 和注入/越狱任务中同样表现最优。

**⚠️ 局限性**

主要局限是仅在 LLaMA3-8B 上验证，路由器作为黑盒可能缺乏可解释性，且对策可能在更复杂攻击场景下被绕过。

---

## 296. 4D Reconstruction from Sparse Dynamic Cameras

**arXiv ID:** 2606.04593 | [PDF](https://arxiv.org/pdf/2606.04593v1)

**作者:** Kazuki Ozeki `[一作]` (Keio University), Yoshimitsu Aoki `[通讯]` (Keio University)

**通讯引用:** 3415 | [OpenAlex ID](https://openalex.org/A5070908826)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一套针对稀疏动态相机配置的4D重建框架，能够在多摄像机随动的条件下实现高质量的时间动态三维场景重建。

**💡 创新点**

核心创新点包括：① 通过跨视点特征匹配与单摄像机跟踪相结合的三维轨迹初始化，显著提升时空一致性；② 采用噪声鲁棒的深度顺序正则化损失，解决深度估计噪声问题；③ 引入跨视点和跨时间的批采样策略，提升优化的泛化与稳定性。

**🔧 技术方法**

技术实现基于3D Gaussian Splatting（3DGS）与MoSca骨干，结合了GIM‑DKM、MASt3R、CoTracker3‑Online等基础模型，使用本质矩阵筛选、三角化、Sampson误差阈值、ordinal depth loss和多视点时间批采样。

**📊 数据集**

构建了新的稀疏动态4D数据集（S4D），包含5个序列、4种室内外场景，使用3个随动摄像机和1个固定评估摄像机录制，提供真实世界的多视角动态数据。

**📈 对比分析**

与D3DGS、FTGS、MoSca、MoSca‑M等基线对比，平均PSNR提升约1–3 dB，SSIM提升约0.01–0.02，LPIPS下降约10–20%，在动态区域和复杂运动场景下表现尤为优异。

**⚠️ 局限性**

主要局限性包括：需要手工或半自动的动态遮挡标注；对极端视角变化时轨迹噪声仍会影响重建质量；目前实验未覆盖极高帧率或极低光照等极端拍摄条件。

---

## 297. SHB-AE: Spherical harmonic beamforming based Ambisonics encoding and upscaling method for smartphone microphone array

**arXiv ID:** 2606.04584 | [PDF](https://arxiv.org/pdf/2606.04584v1)

**作者:** Yuhuan You `[一作]`, Xueyang Lv `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对智能手机麦克风阵列（SPMA）提出了一种基于球谐波波束成形（SHB-AE）的HOA编码与上采样方法。

**💡 创新点**

创新点在于将每个球谐波函数视为目标波束模式，设计对应波束形成器；引入离散球谐波变换（DSHT）与高频分频操作，实现仅用四个不规则麦克风即可编码至四阶HOA，并在高频范围内显著抑制空间折叠；方法不受麦克风数限制，可通过测量阵列流形直接获得最佳编码矩阵。

**🔧 技术方法**

技术包括：波束成形、DSHT、频域分频、最小范数伪逆、噪声与混响下的鲁棒性分析，以及对比基线最小二乘编码。

**📊 数据集**

数据集为：a) 通过测量得到的真实SPMA的冲激响应（在无回声室中使用指数正弦扫频）；b) 对应的自由场仿真阵列流形，用于噪声与混响实验（使用pyroomacoustics生成不同RT60的房间冲击响应）。

**📈 对比分析**

与传统最小二乘HOA编码方法比较：在2–5 kHz区间，SHB-AE的空间相关性提升0.1–0.3，重构误差下降约30–50%，SDR提升约3–5 dB；在噪声和混响实验中，误差显著降低，鲁棒性更强，尤其在低SNR（0 dB）和高RT60（2 s）下表现突出。

**⚠️ 局限性**

局限性：对阵列流形的测量与估计依赖，若麦克风位置不稳定或测量误差大，编码精度会下降；在实际实验中相较于仿真，频率分辨率提升有限；高阶（>4阶）上采样在实测条件下收益不大，且仍受空间折叠与麦克风间距限制。

---

## 298. Independence and Domination on Bounded-Treewidth Graphs: Integer, Rational, and Irrational Distances

**arXiv ID:** 2606.04572 | [PDF](https://arxiv.org/pdf/2606.04572v1)

**作者:** Tim A. Hartmann `[一作]` (CISPA Helmholtz Center for Information Security), Dániel Marx `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在树宽有限图上距离 d 版独立集（a‑independent set）和支配集（a‑dominating set）以及其连续版本（δ‑dispersion、δ‑covering）的时间复杂度，给出了多项上界与下界，并证明了在大多数参数取值下这些问题是不可多项式可约的。

**💡 创新点**

创新点主要包括：
- 统一了离散与连续距离模型的关系，给出从 a‑independent set 到 δ‑dispersion 的精确对应；
- 在 b‑细分图上精确刻画了 a‑independent set 与 a‑dominating set 的可解性阈值，阐明了 a/b 与 2、3、4 等常数的细致分界；
- 对于特定的不可计算无理数 δ，证明了 δ‑dispersion 与 δ‑covering 在树宽参数下 W[1]‑难，并给出了基于 ETH 的非多项式时间下界；
- 推导了 SETH 下最优指数基底 (2a, 2a+1, (2+2b²)a 等) 的下界。

**🔧 技术方法**

主要技术手段包括：
- 树分解与动态规划相结合的指数级算法；
- 细分图构造与距离映射的子图变换；
- 通过平滑化/翻译（translating）将离散独立集/支配集问题映射为连续点集问题；
- 复杂度下界通过多项式时间可约与 SETH（以及其变种 Primal‑Pathwidth SETH）实现；
- 证明无理数 δ 的可计算性采用了无穷级数与比值分析。

**📊 数据集**

（无）

**📈 对比分析**

方法比较：
- 对于树宽为 t 的图，给出的上界为 d^t·n^O(1)（独立集）、(2a+1)^t·n^O(1)（支配集）等；
- 下界在 SETH 框架下显示这些指数基底是最优的，任何尝试降低基底的算法将违反 SETH；
- 对无理数 δ，展示了即使是任意近似或使用常数 1/2 的点集也无法在树宽参数下得到 FPT 方案；
- 因此，除可解阈值外，所有其他参数组合均表现出高复杂度或 W[1]‑难。

**⚠️ 局限性**

主要局限：
- 对无理数 δ 的结果仅适用于特定形式的不可计算无理数，无法推广到所有无理数；
- 仅在树宽有限（而非树宽无穷大）的图上给出完整的可解性阈值，对一般图仍是 NP‑难；
- 对支配集的细分图构造在某些参数组合下仍需要假设支配集与匹配数之间的关系，导致结果不够通用；
- 由于依赖 SETH/ETH 的下界，若未来 SETH 被打破，则部分下界可能需要修正。

---

## 299. MineXplore: An Open-Source Reinforcement Learning Exploration Benchmark for GNSS-Denied Underground Environment

**arXiv ID:** 2606.04569 | [PDF](https://arxiv.org/pdf/2606.04569v1)

**作者:** Abhishek S `[一作]` (BuildMachineLabs), Sreeram MV `[通讯]` (BuildMachineLabs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个基于MuJoCo的地下采矿环境MineXplore，用以模拟真实矿山的复杂隧道拓扑并支持强化学习探索

**💡 创新点**

创新点在于将Chilean矿山的实际测绘数据通过六阶段流程转换为高保真物理模拟，首次提供开放源码的非程序化地下环境，并整合了真实墙面纹理、地面摩擦区、全局倾斜及周期性灯光

**🔧 技术方法**

采用MuJoCo/MJX、V-HACD凸包分解、OpenCV轮廓提取、LiDAR点云纹理映射、Gymnasium接口及RLlib PPO训练框架

**📊 数据集**

使用Leung等人2017年的智利地下铜矿测绘数据（2D平面图与3D LiDAR点云）作为几何与纹理来源

**📈 对比分析**

通过单智能体PPO基线进行评估，五个随机种子平均最终覆盖率84.53%±12.02%，最佳滚动覆盖率88.89%±1.74%，训练过程展示了奖励信号与碰撞率的可解释性提升

**⚠️ 局限性**

局限包括：未实现细粒度海拔变化、碰撞模型为凸包分解可能产生通道误差、缺乏对LiDAR传感器重放的验证以及多智能体与更广泛算法基线的缺失

---

## 300. GENEB: Why Genomic Models Are Hard to Compare

**arXiv ID:** 2606.04525 | [PDF](https://arxiv.org/pdf/2606.04525v1)

**作者:** Daria Ledneva `[一作]` (Moscow Independent Research Institute of Artificial Intelligence), Denis Kuznetsov `[通讯]` (Moscow Independent Research Institute of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了GENEB基因组基础模型评测基准，统一了40个模型、100个任务、13类功能的线性探测评估。

**💡 创新点**

创新点在于提供统一、可复现的评测框架，揭示模型规模、架构、标记化和预训练语料对性能的细粒度影响，并系统评估了少样本鲁棒性与域不匹配问题。

**🔧 技术方法**

技术主要采用冻结表示的线性探测、宏观MCC聚合、统计相关性与对比实验，并使用BPE、k-mer、单核苷酸等多种标记化。

**📊 数据集**

数据集涵盖来自多种物种、功能（如转录因子结合、增强子、甲基化等）的100个二分类任务，来源包括公开基因组和功能学数据集。

**📈 对比分析**

通过对所有模型在统一探测协议下的宏观MCC进行对比，发现规模相关性显著但并非决定性，架构和预训练往往决定类别性能，少样本情形下排名大幅重排，表现仍相对低。

**⚠️ 局限性**

局限性包括对长距离调控任务覆盖不足、缺乏细菌/病毒等非真核任务、冻结表示可能低估微调效果、以及标记化与池化交互未充分分离。

---

## 301. Fine-grained Fragment Retrieval in Multi-modal Long-form Dialogues

**arXiv ID:** 2606.04591 | [PDF](https://arxiv.org/pdf/2606.04591v1)

**作者:** Hanbo Bi `[一作]` (Tencent Inc), Jie Zhou `[通讯]` (Tencent Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出多模态长对话的细粒度片段检索任务，并分别设计了单对话检索模型 F^2RVLM 与跨语料库检索系统 FFRS。

**💡 创新点**

创新点包括：1) 结合生成式检索与多目标强化学习与难度感知训练提升片段一致性；2) 构建最大长对话数据集 MLDR 及真实 WeChat 测试集；3) 采用离线结构化索引与在线双阶段检索实现高效检索。

**🔧 技术方法**

技术手段包括：多模态大语言模型 Qwen 系列、GRPO 强化学习、困难级别自适应采样；片段双层对比学习的 Fragment Embedding Model；向量数据库进行快速召回。

**📊 数据集**

使用的数据集为 MLDR（平均 25.45 轮长对话）和真实 WeChat 多模态对话测试集（平均 75.38 轮）。

**📈 对比分析**

与 CLIP、BLIP2、E5、GME、Qwen、GPT‑4o、Gemini 等基线进行比较，F^2RVLM 在 MLDR 验证集 F1 达 87.25%，在 WeChat 测试集 F1 达 62.07%；FFRS 在大规模语料库上平均检索延迟仅 18 秒，比全量生成方法快 40 倍，人工评测得分 4.2/5。

**⚠️ 局限性**

局限性包括：对极长对话仍需改进；模型对多模态对齐高度依赖训练数据；缺少跨语言或多任务的通用性；离线索引的更新成本和对动态语料库的支持有限。

---

## 302. Addressing Negative Commons Governance with Positive Commons Principles

**arXiv ID:** 2606.04563 | [PDF](https://arxiv.org/pdf/2606.04563v1)

**作者:** Boyang Zhou `[一作]` (University of Washington), Oleg Ianchenko `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过定性案例研究方法，对电子废弃物和 Linux 内核两大负公共资源治理体系进行比较，检验 Ostrom 的八项设计原则在负公共资源管理中的适用性。

**💡 创新点**

创新点在于将原本用于正公共资源的 Ostrom 原则迁移并验证其在负公共资源（如污染和软件错误）治理中的有效性，首次系统性地比较两种截然不同的负公共资源治理模式。

**🔧 技术方法**

主要技术手段为定性编码与比较分析：先对各类治理文件和邮件线程进行主题编码，再将编码结果映射到 Ostrom 的八项原则，形成原则遵循度评估。

**📊 数据集**

使用的数据集包括：电子废弃物治理的 10 篇实证论文（涵盖多国案例），以及 Linux 内核邮件列表（LKML）中 53 条被拒绝补丁的讨论线程。

**📈 对比分析**

比较方法是基于原则编码的对比，结果显示 Linux 体系在所有八项原则上均高度符合，而 Basel 约束在监测、处罚和嵌套结构等方面表现不足，体现了治理有效性差异。

**⚠️ 局限性**

局限性包括样本规模有限（仅两案例）、缺乏量化指标、未能捕捉软件治理与硬件生命周期间的相互作用，以及对不同负公共资源类型的普适性验证不足。

---

## 303. PS-UIE: Privilege-Separated Integrity Enforcement for User-Space Executable Objects in Confidential VMs

**arXiv ID:** 2606.04549 | [PDF](https://arxiv.org/pdf/2606.04549v1)

**作者:** Jingkai Mao `[一作]` (Beijing Jiaotong University), Xiaolin Chang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 3688 | [OpenAlex ID](https://openalex.org/A5024597522)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在AMD SEV‑SNP环境下，提出并实现了PS‑UIE，一种通过特权分离架构对用户空间可执行对象进行完整性检测与强制执行的机制，并生成可验证的运行时证据。

**💡 创新点**

创新点包括：①将完整性测量与执行授权完全分离到VMPL0层；②覆盖Linux三条主执行权限授予路径（execve、mmap+exec、NX→X），实现对文件驱动可执行对象的完整性保护；③设计了快路径Bloom‑filter来降低执行时开销；④通过vTPM绑定vPCR生成可验证日志，实现无信任链的证据回放。

**🔧 技术方法**

技术实现涵盖：AMD SEV‑SNP的VMPL与SVSM；Linux内核拦截模块UIE‑Guest；SVSM模块UIE‑Monitor（策略、测量、决策、证据）；SHA‑256哈希、vTPM签名与Quote；签名的策略包与epoch；Bloom‑filter快速拒绝；日志与vPCR绑定。

**📊 数据集**

实验使用了AMD EPYC 7763服务器（256 GB RAM）搭配Linux 20.04/6.5内核、QEMU 7.2以及自研Coconut‑SVSM；通过自定义微基准和典型工作负载（execve、mmap、动态链接库加载）进行性能评估，未使用公开数据集。

**📈 对比分析**

对比实验包括：Native、Measure、Enforce、Enforce FP四种配置。对小文件（4 KB）时，Enforce FP约1.1–1.3×慢；对1 MB文件时，慢至约9×；在加载多份DSO时，Enforce FP将慢速降至1.3–1.4×。总体认为在普通工作负载下开销可接受，且快路径显著降低开销。

**⚠️ 局限性**

局限性：仅保护文件驱动可执行对象，匿名可执行内存不受覆盖；对多DSO的开销随数量呈线性增长；实现仅针对AMD SEV‑SNP，未验证在Intel TDX或ARM CCA等平台的可迁移性；需要可靠的钩子实现，若钩子被篡改或遗漏将失效。

---

## 304. Optical-Guided Neural Collapse for SAR Few-Shot Class Incremental Learning

**arXiv ID:** 2606.04528 | [PDF](https://arxiv.org/pdf/2606.04528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 305. Impostor: An Agent-Curated Benchmark for Realistic AIGC Manipulation Localization

**arXiv ID:** 2606.04545 | [PDF](https://arxiv.org/pdf/2606.04545v1)

**作者:** Zhenliang Li `[一作]` (Southeast University), Jungong Han `[通讯]` (Tsinghua University)

**通讯引用:** 25977 | [OpenAlex ID](https://openalex.org/A5046605531)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Impostor数据集和PANet模型，用于提升AIGC生成图像的伪造检测与定位能力；

**💡 创新点**

创新点在于：①CraftAgent闭环生成器通过感知-规划-执行-验证-反思四步循环自动产生高质量、符合用户意图的多区域编辑样本；②Impostor涵盖视觉真实性、编辑多样性和生成器覆盖三大维度，解决现有数据集短板；③PANet融合局部残差与局部相位指纹、语义-取证一致性监督以及跨通道注意力，显著提升细粒度定位与跨域泛化；

**🔧 技术方法**

采用的技术包括：闭环Agent（基于LLM和视觉模型）、多通道语义-取证双流网络、局部相位提取（Riesz变换+Log‑Gabor）、残差滤波（SRM+可学习卷积）、互导交叉注意力、语义-取证一致性损失、对比损失等；

**📊 数据集**

使用的数据集为：Impostor（100K图像，7种AIGC编辑器，3种编辑类型，多区域场景）；并在CocoGlide、AutoSplice、SID-Set、BR‑Gen等公开数据集进行跨域评估；

**📈 对比分析**

在跨数据集像素级定位和跨域图像级检测两项基准上，PANet均超过所有对比方法，IoU/F1在Impostor上分别达0.7859/0.8517，跨域检测平均准确率92.68%；同时对常见后处理噪声、压缩、缩放等做了鲁棒性评估，表现最优；

**⚠️ 局限性**

限制点包括：仍对极小区域或极端后处理效果较弱；模型复杂度较高，对推理速度和资源要求较高；仅覆盖三种编辑类型，未来需扩展至更多真实编辑场景；

---

## 306. Enhanced Fluid Index Modulation for Integrated Data and Energy Transfer

**arXiv ID:** 2606.04537 | [PDF](https://arxiv.org/pdf/2606.04537v1)

**作者:** Long Zhang `[一作]` (University of Electronic Science and Technology of China), Kai-Kit Wong `[通讯]` (University College London)

**通讯引用:** 26699 | [OpenAlex ID](https://openalex.org/A5011048761)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种在集成数据与能量传输（IDET）系统中使用流体指数调制（FIM）的增强方案，结合双维流体天线（FAS）与功率分离接收架构，实现信息与能量的同时传输，并给出了平均提取功率、误码率和可达数据速率的闭式表达式。

**💡 创新点**

创新点包括：①在FAS上引入FIM，利用大量可变端口实现指数调制；②在有限字母信号下联合优化端口选择、相位预编码和功率分离比例，以最大化平均提取功率；③提出的交替优化框架中，利用黎曼增广拉格朗日方法（RALM）求解相位优化子问题，采用块坐标下降（BCD）求解端口选择子问题；④通过理论分析与仿真验证，该方案在R‑E（速率‑能量）取舍上优于现有基准。

**🔧 技术方法**

使用的技术主要包括流体天线系统（FAS）与指数调制（IM/FIM）、功率分离式IDET、非线性能量收集模型、最大似然检测、误码率上界推导、黎曼增广拉格朗日优化、Riemannian共轭梯度算法、块坐标下降（BCD）以及仿真平台。

**📊 数据集**

本文没有使用实际的测量数据集，而是通过蒙特卡洛仿真（10^3 次试验）在多种参数设置（如 N=64、L=4/8、PSK/QAM 调制、功率分离比例、非线性 EH 系数等）下评估系统性能。

**📈 对比分析**

在仿真中，本文将增强 FIM 与四种基准方案（固定 FIM、Top‑L FIM、Group FIM、传统 FPA‑SM）进行比较，结果显示：①相位优化和端口选择优化显著提升能量提取；②增强 FIM 在不同 BER 阈值和调制方案下均优于基准，尤其在低 BER 约束时接近穷举搜索的最优性能；③R‑E 取舍曲线表明，增强 FIM 能在保证误码率的前提下实现更高的能量提取。

**⚠️ 局限性**

主要局限包括：①求解过程仍涉及多轮交替优化和非凸约束，局部最优风险；②仿真仅考虑单用户、单链路场景，缺乏多用户或多天线交互的验证；③对流体天线机械运动、能量消耗等实际实现细节未做深入讨论；④在极高阶调制或大量端口时，计算复杂度仍显著，需进一步简化算法。

---

## 307. Scaling Self-Evolving Agents via Parametric Memory

**arXiv ID:** 2606.04536 | [PDF](https://arxiv.org/pdf/2606.04536v1)

**作者:** Tao Ren `[一作]` (Peking University), Yijie Peng `[通讯]` (Peking University)

**通讯引用:** 766 | [OpenAlex ID](https://openalex.org/A5005503619)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自演化参数化记忆框架，在长时序对话和搜索任务中把测试时的LoRA微调视为代理决策过程的一部分；

**💡 创新点**

创新点在于：①把记忆写入（QA式监督提取）与LoRA更新纳入同一 rollout 动态，允许策略在单个 episode 内实时改变；②通过 stop‑gradient 目标联合训练基模型与提取策略；③使用 SVD 初始化 LoRA 子空间以加速少步适应；

**🔧 技术方法**

核心技术包括：基于 Qwen3 模型的 LoRA 微调、SVD 初始化、在线 SFT（Δ_t 更新）、强化学习（GRPO）与停梯度策略优化；

**📊 数据集**

实验数据集：LoCoMo、LongMemEval‑S、Multi‑Objective Search、CL‑Bench 以及对应的 Qwen3‑4B/8B 预训练模型；

**📈 对比分析**

与无记忆、总结式和检索式记忆基线相比，在所有任务上均取得显著提升：LoCoMo/LongMemEval‑S 的 F1/EM 上提升约 4‑10 分，搜索任务 F1/EM 提升 4‑5 分，CL‑Bench 评分类别精度提升 2‑3 分；

**⚠️ 局限性**

局限性：需额外的在线 LoRA 计算和 SVD 初始化，易受上下文阈值 L_max 的影响；对极大规模数据或极长上下文仍可能出现提取瓶颈，且实现复杂度高于纯 prompt‑based 方法。

---

## 308. MeshFlow: Efficient Artistic Mesh Generation via MeshVAE and Flow-based Diffusion Transformer

**arXiv ID:** 2606.04621 | [PDF](https://arxiv.org/pdf/2606.04621v1)

**作者:** Weiyu Li `[一作]` (Meta AI), Andrea Vedaldi `[通讯]` (Meta AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种非自回归的三维网格生成框架，利用MeshVAE将网格压缩为连续潜在空间，再通过Rectified Flow Transformer并行生成顶点和连通性，实现高质量艺术网格生成。

**💡 创新点**

创新点包括：①设计连续、无量化的MeshVAE编码，通过对比学习学习边嵌入，显著降低token数；②构建基于Rectified Flow的并行生成器，推理速度比传统自回归模型快18倍；③支持点云/图像条件生成，提升生成灵活性。

**🔧 技术方法**

使用的技术：MeshVAE（VAE + 对比学习 + 连续边嵌入）、直方图阈值拓扑恢复、Rectified Flow直线ODE、Diffusion Transformer、RoPE3D位置编码、logit-normal采样、FlashAttention、BF16混合精度训练。

**📊 数据集**

训练和评估数据集：约60万高质量艺术家模型的Objaverse-like专有数据集；在点云条件生成评估中使用Toys4K数据集。

**📈 对比分析**

与多种基准（MeshAnything、TreeMeshGPT、BPT、FastMesh、SpaceMesh、MeshCraft等）对比，采用Chamfer Distance、Hausdorff Distance等指标；结果表明模型在CD/HD上取得最佳性能，同时推理速度比自回归模型快18倍，生成时间约1秒。

**⚠️ 局限性**

局限性：仅支持三角面，可能产生小孔和翻转法向；传统评价指标对拓扑缺陷敏感度不足；未处理纹理/UV映射生成；对极大或不完整网格的鲁棒性仍需进一步提升。

---

## 309. QuBLAST: A Framework for Quantizing Large Language Models with Block-Level Compression Approach and Activation Scaling Strategy

**arXiv ID:** 2606.04620 | [PDF](https://arxiv.org/pdf/2606.04620v1)

**作者:** Pasindu Wickramasinghe `[一作]` (New York University Abu Dhabi), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11443 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型进行后训练量化，采用块级压缩和激活缩放策略，实现混合精度分配

**💡 创新点**

①针对每个注意力块进行交叉熵敏感度分析并锁定低位宽；②使用激活缩放有效抑制激活异常；③在锁定空间内搜索混合精度配置以最大化内存节省

**🔧 技术方法**

后训练量化（PTQ）、块级压缩、精度锁定、激活缩放、混合精度搜索

**📊 数据集**

WikiText-2、WikiText-103

**📈 对比分析**

与 SmoothQuant、GPTQ、SpinQuant、Uniform W8A8 等 PTQ 基线比较，内存节省约 40‑45%（激活量化时 42‑48%），困惑度提升不超过 5%（激活量化时不超过 2%)

**⚠️ 局限性**

仅适用于后训练量化，未考虑权重微调；对极大模型或更稀疏/动态量化未评估；依赖预设阈值，对不同任务/数据集的泛化可能有限

---

## 310. BPDA-GMM: Bayesian Probabilistic Data Association via Gaussian Mixture Models for Semantic SLAM

**arXiv ID:** 2606.04618 | [PDF](https://arxiv.org/pdf/2606.04618v1)

**作者:** Thanh Nguyen Canh `[一作]` (Japan Advanced Institute of Science and Technology), Nak Young Chong `[通讯]` (Japan Advanced Institute of Science and Technology)

**通讯引用:** 2729 | [OpenAlex ID](https://openalex.org/A5000452220)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种在线贝叶斯概率数据关联框架BPDA‑GMM，用于在随时间增大的对象级语义SLAM地图中进行软关联与地图更新。

**💡 创新点**

创新点包括：①使用Dirichlet过程先验诱导的CRP关联模型，自动平衡旧地标与新地标的概率；②将观测软关联概率直接更新语义高斯地标，形成高斯混合地图；③引入置信度触发的α‑divergence温度调节提升在噪声或歧义观测下的辨识度；④在后端采用零姿态雅可比的max‑mixture语义因子，实现语义测量对轨迹的解耦。

**🔧 技术方法**

核心技术有：Dirichlet过程/CRP、贝叶斯软关联、语义与几何门控、α‑divergence温度调节、语义高斯混合更新、max‑mixture因子、iSAM2后端、零雅可比解耦、CRP重识别循环闭环。

**📊 数据集**

实验数据集包括：多种仿真场景（Figure Eight、Outdoor Loop、Complex Urban、City10000、Victoria Park）和实际室内飞行序列（Falcon 250 + RealSense D455，包含84个真实对象，5个语义类别）。

**📈 对比分析**

与MHJCBB、MH‑iSAM2、Gauss PDA、MMSS、SGBA、k‑best、SlideSLAM等基线对比，BPDA‑GMM在轨迹精度（ATE、RPE）、地图质量（Chamfer距离、几何不确定度）、语义一致性（entropy、accuracy）上均实现了显著提升，并保持了低于10 Hz的实时性能。

**⚠️ 局限性**

主要局限：①单高斯地标模型难以表达形状复杂或多模态对象；②仅支持离散固定类别，难以扩展到开放词汇或连续语义嵌入；③缺乏主动数据关联规划，无法通过行动主动消除歧义。

---

## 311. Parthenon Law: A Self-Evolving Legal-Agent Framework

**arXiv ID:** 2606.04602 | [PDF](https://arxiv.org/pdf/2606.04602v1)

**作者:** Hejia Geng `[一作]` (Tapntell), Leo Liu `[通讯]` (Tapntell)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对Harvey LAB大规模法律任务进行了实证分析，并提出了可审计的六层Parthenon框架以及无参数自演化学习循环，显著提升了法律AI代理的整体表现。

**💡 创新点**

创新点包括：①将模型、工具、知识、技能等拆分为可编辑表面，形成可审计的六层框架；②设计了 anti‑leakage 机制，确保学习循环不泄露具体答案；③通过自演化学习将失败反馈转化为对工具、技能和知识的可审计更新，而非微调模型权重。

**🔧 技术方法**

技术实现上采用了多模型（GPT、Claude、Gemini、DeepSeek、Kimi）与多工作空间引擎（Codex、Claude Code 等）集成，构建 deterministic audit tools、知识库、技能库，并实现了 anti‑leakage learner 与增量更新机制。

**📊 数据集**

使用的数据集为 Harvey Legal Agent Benchmark（LAB），共 12,510 条任务轨迹，涵盖多领域法律实践。

**📈 对比分析**

通过对比直接 API 调用、基本法律本地引擎、Codex/Claude Code harness 等方案，实验显示 Parthenon 在所有准则通行率上提升 13.8/10.2/7.4 百分点，严格全通过率从 14→42、47→137 等，几乎等同于模型升级带来的提升。

**⚠️ 局限性**

局限性包括：仍无法一次性完成整个法律事项；对极端长程推理或跨域新法规的适配能力有限；学习循环仅更新可编辑表面，可能无法修复深层模型缺陷。

---

## 312. Plan First, Judge Later, Run Better: A DMAIC-Inspired Agentic System for Industrial Anomaly Detection

**arXiv ID:** 2606.04599 | [PDF](https://arxiv.org/pdf/2606.04599v1)

**作者:** Yongzi Yu `[一作]` (Hong Kong University of Science and Technology), Man Li `[通讯]` (Southwestern University of Finance and Economics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DMAIC‑inspired 多代理框架 DMAIC‑IAD，用 LLM 自动化工业异常检测并引入预执行评估的 Judge 模型实现“先规划后评判”；

**💡 创新点**

创新点在于将 DMAIC 质量管理流程嵌入 LLM 代理设计，构建 SOP 与数据剖析前置阶段；设计无执行的 Judge 模型实现候选策略的快速评分；通过经验缓存实现冷启动与经验复用；

**🔧 技术方法**

采用 GPT‑4o、GPT‑5‑Mini、Claude‑Sonnet‑4.5 等 LLM，结合知识检索、句子变换、MLP 评分器、代码生成与自动化验证循环；

**📊 数据集**

使用四种模态共八个基准集：表格（vertebral, arrhythmia）、时序（PSM, SWaT）、图形（books, enron）、图像（metalnut, tile）；

**📈 对比分析**

与 AD‑AGENT、AutoIAD 以及“Strategist only”对比，平均 AUROC 提升 37.76%，最高成功率 78.19%，检测质量显著提升但运行时最长；

**⚠️ 局限性**

Judge 模型对 OOD 情形表现欠佳；系统缺乏持续自我进化能力，难以跟踪工业环境中的数据漂移与异常模式演化。

---

## 313. Ekka: Automated Diagnosis of Silent Errors in LLM Inference

**arXiv ID:** 2606.04594 | [PDF](https://arxiv.org/pdf/2606.04594v1)

**作者:** Yile Gu `[一作]` (University of Washington), Baris Kasikci `[通讯]` (University of Washington)

**通讯引用:** 2126 | [OpenAlex ID](https://openalex.org/A5050964144)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种自动化诊断LLM推理中无声错误的系统，利用差分调试方法对目标框架与参考实现的中间执行状态进行对齐和比较，定位根因。

**💡 创新点**

创新点包括：①基于LLM代理的多阶段诊断工作流，自动完成组件映射、激活对齐与误差分析；②使用鲁棒误差比率结合变化点检测，区分实现缺陷与数值误差；③构建“模型树”压缩架构、代码索引与知识库辅助对齐。

**🔧 技术方法**

主要技术手段有：LLM代理（Claude Sonnet 4.5 + LangGraph）、PyTorch前向钩子收集激活与调用序列、静态代码分析、组件映射与增量验证、激活对齐代码生成、误差比率与变化点分析。

**📊 数据集**

使用了从vLLM和SGLang公开issue中收集的90个无声错误（70已闭合，30用于新错误诊断），构建了真实世界错误基准，包含模型实现、核实现、数值误差等多种根因。

**📈 对比分析**

与现有软件工程代理（OpenCode、Mini‑SWE‑Agent）对比，pass@1达到0.84、pass@5达到0.88，分别超过对比基线；平均诊断成本约$30，运行时间和令牌消耗均在可接受范围内。

**⚠️ 局限性**

局限性包括：只能定位到模块层级的缺陷；依赖可比的参考实现，无法处理仅在框架层面出现的并发或硬件相关错误；当前实现基于PyTorch，迁移到其他框架需要重写收集与树构建逻辑。

---

## 314. VCIFBench: Evaluating Complex Instruction Following for Video Understanding

**arXiv ID:** 2606.04588 | [PDF](https://arxiv.org/pdf/2606.04588v1)

**作者:** Huangchen Xu `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**通讯引用:** 11816 | [OpenAlex ID](https://openalex.org/A5029392006)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VCIFBench，旨在评估视频理解任务中的多约束指令跟随能力，并构建了基于规则、混合和 LLM 判断的混合验证管道。

**💡 创新点**

创新点在于：①将基准问题与直接视频衬托提示相结合，生成包含 20 种任务类型和 40 种约束的丰富指令；②引入冲突诊断子集和 DPO 优化数据；③通过混合验证实现内容、格式、风格和结构四维约束的可验证性。

**🔧 技术方法**

技术手段包括 GPT‑5.2 与 Gemini‑2.5 Pro 的指令生成与校对；规则检查、参数抽取与可执行检查器；LLM‑判断（GPT‑5）对语义约束的评估；以及基于 DPO 的偏好优化训练。

**📊 数据集**

数据来源于 TempCompass、MMWorld、NExT‑QA、YouCook2 四个视频数据集，构造了 306 条可满足的测试指令、540 对 DPO 偏好样本和 30 条冲突诊断样本。

**📈 对比分析**

实验在 10 种 MLLM 上进行，衡量 IPR（指令通过率）和 CPR（约束通过率）。专有模型 IPR 约 52%、CPR 约 85‑86%，开源模型 IPR 仅 24‑29%；DPO 训练后 Qwen3‑VL‑8B IPR 从 27.12% 提升到 33.01%，CPR 亦有显著提升，表明多约束满足仍具挑战。

**⚠️ 局限性**

局限性包括：仅覆盖英文指令与输出；聚焦于约束丰富的子集，未全面评估视频理解、事实性或安全性；混合评估可能受 LLM 判断偏差影响；对约束优化的过度侧重可能削弱模型的整体可靠性。

---

## 315. TIBlender: Early-Warning Threat Intelligence from Cross-Platform Social Media Evidence

**arXiv ID:** 2606.04580 | [PDF](https://arxiv.org/pdf/2606.04580v1)

**作者:** Hiroki Nakano `[一作]` (NTT Security Holdings Corporation & NTT, Inc.), Daiki Chiba `[通讯]` (Tokyo Metropolitan University)

**通讯引用:** 735 | [OpenAlex ID](https://openalex.org/A5053184316)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套名为 TIBlender 的多智能体系统，能够实时收集并整合 X、Reddit、Telegram 与 Discord 四大社交媒体平台的多语言、碎片化威胁信息，并自动生成结构化的威胁情报（TI）报告。

**💡 创新点**

创新点包括：① 跨平台多语言信息融合并自动生成可直接使用的 TI 报告；② 通过多视角 LLM + 外部工具协同验证，形成证据链，显著降低误报；③ 引入强化学习驱动的调查策略和趋势查询动态适配，提高检测灵活性；④ 将从采集到报告的全流程实现端到端自动化。

**🔧 技术方法**

采用了多智能体架构、Grok 4 Fast、GPT‑OSS‑120B、Llama 4 等大型语言模型；外部工具包括 RDAP/WHOIS、Passive DNS、CVE 数据库、TLS 证书分析等；聚类使用 IoC 重叠 + 语义相似度 + HDBSCAN；强化学习采用 BC + DQN；OCR、规则过滤、趋势检测和动态查询生成等技术。

**📊 数据集**

使用真实收集的数据，约 874,000 条原始帖子（X 48k、Reddit 29k、Telegram 579k、Discord 217k），聚合成 184,240 个威胁集群，最终生成 8,288 条 TI 报告。对比数据来自六个公开威胁源（PhishTank、URLhaus、MalwareBazaar、CISA KEV、OTX Pulses、TweetFeed）以及单平台基线（Tweezers、DarkGram、Reddit2CTI）。

**📈 对比分析**

通过在相同输入条件下与单平台基线对比，TIBlender 的直接提取和全流程调查（Direct+Pivot）在 IoC 数量、覆盖率和创新率上均高于基线（最高 5.1×、80%+ 创新率）。与公开 feed 的比对显示 83–99% 的 IoC 在任何 feed 中缺失，且在重叠 IoC 中 18.7% 先于 feed 出现，平均提前 72–94 小时。人工评估报告质量误报率约 1.8–2.2%，漏报率约 0.6–0.9%；消融实验证明每个模块对覆盖、IoC 产出和质量控制都至关重要。

**⚠️ 局限性**

实验仅覆盖 31 天且仅四个平台，低资源语言和地下论坛、私有频道等信息源未纳入；模型依赖导致成本和潜在行为漂移；对抗鲁棒性尚未系统评估；需要更长周期、更多平台和更广泛语言的验证以进一步证明系统的鲁棒性与通用性。

---

## 316. Flow-HOA: Generative Joint Optimization for Ambisonics Encoding via Flow Matching

**arXiv ID:** 2606.04570 | [PDF](https://arxiv.org/pdf/2606.04570v1)

**作者:** Yuhuan You `[一作]`, Xueyang Lv `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过条件流匹配学习物理先验到最优FIR编码滤波器的映射，提出Flow-HOA框架实现高阶全息声（HOA）编码。

**💡 创新点**

创新点在于：①把HOA滤波器设计视为生成式优化任务；②使用复合时域、频域和空间域损失共同引导；③利用条件流匹配（Conditional Flow Matching）在离线训练阶段学习梯度场，从而生成可部署的时不变FIR滤波器。

**🔧 技术方法**

主要技术包括：物理先验滤波器构造、复合多域损失（MSE、STFT、能量保持、空间一致性）、1D U‑Net生成器、条件流匹配与ODE求解、基于MOORE‑PENROSE伪逆的先验求解。

**📊 数据集**

使用FSD50K干音源与在无回声室测得的180个方向的阵列冲激响应（AIR）进行动态合成；测试集为从同一测得IR随机采样的5000个（100段×50方向）合成样本；主观测试使用真实SPMA录制的8个方向声源音频。

**📈 对比分析**

与传统的Ambisonics Signal Matching (ASM) 线性最小二乘基准相比，Flow-HOA在SI‑SDR、LSD、SPM‑KL和DGC等四项客观指标上均取得显著提升（SI‑SDR +6.4 dB，LSD -5.0，SPM‑KL -0.3，DGC -1.3 dB）。主观MUSHRA测试中总体音质得分提高约13.6分，虽然空间定位得分与ASM相近，存在IHL现象。

**⚠️ 局限性**

局限性：①在非个性化HRTF和无头部追踪的听觉链中，较高的编码精度反而揭示了内头定位(IHL)问题；②实验仅覆盖单源、无回声室条件，缺乏多源混响环境的验证；③模型对测量误差和环境变化的鲁棒性尚未彻底评估。

---

## 317. Rollout-Level Advantage-Prioritized Experience Replay for GRPO

**arXiv ID:** 2606.04560 | [PDF](https://arxiv.org/pdf/2606.04560v1)

**作者:** Gyeongtae Yoo `[一作]` (Seoul National University), Sungroh Yoon `[通讯]` (Seoul National University)

**通讯引用:** 13092 | [OpenAlex ID](https://openalex.org/A5086877012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对GRPO（Group Relative Policy Optimization）的经验回放缓冲区，采用年龄驱逐、基于优势的回放优先级和“新鲜锚定”组合，显著提升大规模LLM的样本效率与准确率。

**💡 创新点**

创新点在于：①将回放缓冲区从查询级切换为单个rollout级；②以单rollout优势|A_i|为优先级，实现对稀有高价值rollout的再利用；③采用年龄驱逐与新鲜锚定组合，严格限制策略漂移与老旧经验的影响。

**🔧 技术方法**

使用的技术包括：GRPO（基于PPO的裁剪策略）、优先经验回放（PER）、年龄驱逐机制、优势归一化和分层采样；实验中采用Qwen3-Base模型在DeepScaleR-Preview提示集上训练。

**📊 数据集**

数据集主要包括五个数学推理基准：MATH‑500、AIME25、AIME26、HMMT‑F25、HMMT‑F26，以及用于训练的DeepScaleR‑Preview提示集。

**📈 对比分析**

与传统无回放GRPO以及两种基准回放方法（均匀和基于σ_g的优先级）进行对比，实验显示在三种模型规模（0.6B、1.7B、4B）上均实现了正向提升，4B规模上五个基准的平均准确率提升4.35个百分点，AES效率提升最高达+0.579。

**⚠️ 局限性**

限制主要包括：仅验证了二进制可验证奖励的数学推理场景；未探索非可验证奖励或其他RL算法；训练仅在单一步骤快照下评估，未观察长期动态；仅在Qwen3-Base规模内实验，未验证更大规模或多轮任务的通用性。

---

## 318. Temporal Order Matters for Agentic Memory: Segment Trees for Long-Horizon Agents

**arXiv ID:** 2606.04555 | [PDF](https://arxiv.org/pdf/2606.04555v1)

**作者:** Yifan Simon Liu `[一作]` (University of Toronto), Scott Sanner `[通讯]` (University of Toronto)

**通讯引用:** 6687 | [OpenAlex ID](https://openalex.org/A5028174137)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Segment Tree Memory（STMem），一种基于段树的在线构建与检索框架，用于长周期对话记忆；

**💡 创新点**

创新点在于将对话历史按时间连续性组织成段树，既保留时间顺序又形成层次结构，并通过结构感知的分数传播提升检索质量；

**🔧 技术方法**

技术包括：基于段树的在线增量更新（仅更新右侧前沿节点），三种前沿兼容度模型（余弦相似、点对点LLM、批量LLM），以及结构化分数传播矩阵（上下向传播），配合LLM生成节点摘要；

**📊 数据集**

使用了三大长周期对话基准：LoCoMo、LongMemEval-MAB（MemoryAgentBench）和RealMem，评估 LLM-judge 准确率和 token‑级 F1；

**📈 对比分析**

与平面检索（BM25、Dense）、树形检索（RAPTOR、MemTree）以及结构化记忆基线（A‑MEM、Mem0、HippoRAG）对比，STMem 在所有数据集上均提升约 15–20% LLM‑judge 准确率，且在线构建与检索速度与现有结构化基线相当；

**⚠️ 局限性**

局限性包括：检索策略（传播方向、步长、衰减）固定不随查询动态调整；只支持文本对话，未覆盖多模态或工具调用；缺乏增量维护（如子树重构、压缩或去重）来保证长时间部署下的高效性。

---

## 319. Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization

**arXiv ID:** 2606.04547 | [PDF](https://arxiv.org/pdf/2606.04547v1)

**作者:** Heng Cao `[一作]` (Microsoft), Xuyan Mo `[通讯]` (Microsoft)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TAP-PER 框架，利用轻量级前缀学习实现大语言模型的个性化，既不需要检索或显式 Prompt，也不使用每用户的 Adapter。

**💡 创新点**

创新点包括：① 将用户建模拆分为长期用户状态前缀和基于查询的时间感知记录前缀；② 用桥接 LoRA 将前缀信号深度融入共享 backbone；③ 通过前缀实现 130 倍的每用户参数压缩，支持在线持续学习。

**🔧 技术方法**

技术手段：Prefix Tuning + LoRA、DIN‑style 注意力与可学习的时间/顺序衰减、两阶段训练（任务适配 + 个性化前缀学习）、共享桥接 LoRA、与 Llama‑3.1‑8B / Qwen3‑4B 结合。

**📊 数据集**

数据集：LaMP 个人化基准（6 类任务），使用 Llama‑3.1‑8B 作为 backbone，亦在 Qwen3‑4B 进行复现验证。

**📈 对比分析**

与 prompt‑based（RAG、PAG）和 model‑based（OPPU、PER‑PCS、P2P）基线在 LaMP 六个任务上对比，TAP‑PER 在所有指标上均为最优，参数占用比 OPPU 低 130 倍、比 PER‑PCS 约 50% 的总参数，在线学习可接近全重训练效果。

**⚠️ 局限性**

局限性：1）对极长或噪声较多的用户历史，注意力质量可能下降；2）实验仅在 8B/4B 模型上验证，未探讨更大模型的适用性；3）需要安全存储和访问控制以保护用户前缀信息。

---

## 320. Agentic AI and Pedagogical Best Practice: The Tension Between Automation and Learning

**arXiv ID:** 2606.04543 | [PDF](https://arxiv.org/pdf/2606.04543v1)

**作者:** Steve Woollaston `[一作]` (Kyoto University), Hiroaki Ogata `[通讯]` (Kyoto University)

**通讯引用:** 8836 | [OpenAlex ID](https://openalex.org/A5079543720)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文综述了代理式AI在教育中的应用，并提出了六项教学原则的设计建议。

**💡 创新点**

将代理式AI与教学原则相结合，强调有意摩擦与动态衰减，提出教师参与循环的架构。

**🔧 技术方法**

采用大语言模型、持久记忆、情境检索和多代理网络等技术。

**📊 数据集**

无公开数据集；主要基于教育理论与实践案例。

**📈 对比分析**

未进行实验比较，讨论基于文献综述和案例设计。

**⚠️ 局限性**

缺乏实证验证、算法偏见和隐私问题。

---

## 321. MAD: Mapping-Aware World Models for Agile Quadrotor Flight

**arXiv ID:** 2606.04534 | [PDF](https://arxiv.org/pdf/2606.04534v1)

**作者:** Xinhong Zhang `[一作]` (Beijing Institute of Technology), Gang Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 253687 | [OpenAlex ID](https://openalex.org/A5100444820)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种映射感知的世界模型 MAD，用于基于深度图的四旋翼无人机敏捷飞行，并将其与不同的 RL 学习框架（Dreamer、PPO、SHAC）结合，形成端到端的导航与控制系统。

**💡 创新点**

创新点包括：① 用机器人中心的占据网格（OGM）和可见性网格（VGM）作为自监督目标，迫使潜在状态编码空间记忆、姿态与视野信息；② 在 DiffAero 仿真器中实现 GPU 并行的 OGM/VGM 生成，大幅提升训练吞吐量；③ 设计多种策略学习模式，使 MAD 能兼容想象 rollouts 与特征提取，提升任务迁移和跨域鲁棒性。

**🔧 技术方法**

采用的技术主要有：RSSM 结构的变分自编码器（DreamerV3 风格）、3D 占据与可见性网格的构造与监督、GPU 并行深度渲染与网格生成、RL 算法（Dreamer、PPO、SHAC）、视觉输入深度图 + VIO、以及 ONNX 部署实现低延迟推理。

**📊 数据集**

训练与评估数据集主要来自：DiffAero 生成的森林和室内障碍物场（随机生成的圆柱体与多面体）、不同稀疏度的场景；测试时使用 RealSense D435i 的深度图进行真实飞行；实验环境包括 Gazebo/PX4 SITL 以及实际室内/室外森林环境。

**📈 对比分析**

与基线方法（YOPO、EGO‑Planner、PPO）在 Gazebo/ PX4 中进行对比；在视觉导航任务中与 PPO、SHAC 对比；在赛车任务中与 PPO‑Vision、PPO‑State 对比。结果显示：MAD‑Dreamer 在稠密与稀疏环境中成功率最高、峰速最高、路径最短；MAD‑PPO 在任务迁移中优于 PPO‑Vision 并最终超过 PPO‑State；真实飞行最高速度为 5.05 m/s（室外）和 3.10 m/s（室内）。

**⚠️ 局限性**

局限性：① OGM/VGM 构造与监督导致显存占用高，限制训练规模；② 下游策略未进行显式的安全过滤或可行性检查，仍为反应式控制；③ 仅使用深度信息，缺乏多模态感知；④ 目前未实现概率占据预测与对象级结构，未来可进一步提升安全性和鲁棒性。

---

## 322. Echo-Infinity: Learning Evolving Memory for Real-Time Infinite Video Generation

**arXiv ID:** 2606.04527 | [PDF](https://arxiv.org/pdf/2606.04527v1)

**作者:** Yuxuan Bian `[一作]` (Chinese University of Hong Kong), Qiang Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 14860 | [OpenAlex ID](https://openalex.org/A5088556682)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种自回归视频生成框架Echo，结合可学习的记忆查询与统一的相对RoPE计划，实现实时无限长视频生成。

**💡 创新点**

创新点在于：①引入可端到端学习的记忆查询，用于过滤、抽象并压缩已被驱逐的KV缓存；②采用统一的相对RoPE调度，消除训练与推理中的RoPE外推与溢出问题，从而保持恒定的记忆成本与无限生成能力。

**🔧 技术方法**

使用技术包括Diffusion Model Distillation (DMD)、因果注意力、交叉注意力记忆编码器、Sigmoid门控残差更新、相对RoPE、以及三层KV缓存结构。

**📊 数据集**

主要数据集为VidProM用于预训练与微调，MovieGen与VBench-Long作为长视频评估数据集，标准VBench提示集用于短视频评测。

**📈 对比分析**

与LongLive、MemFlow、Memorize-and-Generate、∞-RoPE、SkyReels-V2、MAGI-1等最新长视频生成方法对比，Echo在30s、240s长视频以及交互式生成任务中均取得最高质量与语义一致性评分，实时推理速度为18.5 FPS，可生成24小时、1.3M帧视频。

**⚠️ 局限性**

局限性包括：基模型规模与生成能力限制下，小时级或更长动态场景的稳定性可能下降；记忆查询的语义解释与可控检索功能尚未实现；交互式生成的专门优化仍待完善。

---

## 323. Distributional Approximate Nearest Neighbour Search for Uncertainty-Aware Retrieval

**arXiv ID:** 2606.04603 | [PDF](https://arxiv.org/pdf/2606.04603v1)

**作者:** Olivier Jeunen `[一作]` `[通讯]`, Olivier Jeunen

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Dinosaur框架，在近似最近邻检索中通过采样嵌入分布实现不确定性感知的候选生成，提升长尾曝光；

**💡 创新点**

创新点在于仅通过在索引与查询时多次采样嵌入，便能在不改动模型或索引基础设施的情况下引入探索性检索；

**🔧 技术方法**

使用采样分布式嵌入、FAISS/IVFFlat等ANN搜索、Thompson Sampling和贝叶斯不确定性估计等技术；

**📊 数据集**

使用MovieLens‑32M数据集，采用iALS学习128维嵌入；

**📈 对比分析**

与精确检索和传统IVFFlat近似检索比较，Recall@K基本无损，Catalogue Coverage从约24%提升至≈63%，展现显著的覆盖率提升；

**⚠️ 局限性**

局限在于使用基于交互计数的伪后验来模拟不确定性，未学习真实的嵌入不确定性；离线评估受MNAR偏差影响，需结合在线A/B验证进一步评估效果。

---

## 324. HalfNet: Randomized Neural Networks with Learned Subspace Geometry

**arXiv ID:** 2606.04583 | [PDF](https://arxiv.org/pdf/2606.04583v1)

**作者:** Ethem Alpaydin `[一作]` `[通讯]`, Ethem Alpaydin

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HalfNet，学习随机权重分布的协方差低秩结构，以实现参数更少但性能与全训练网络相当的随机化网络。

**💡 创新点**

通过学习协方差几何而非单独权重的方式，构建可解释为随机投影、度量学习与核方法的低秩随机网络；同时支持连续和二值权重。

**🔧 技术方法**

低秩协方差分解、随机投影、随机特征、直通估计器（STE）、Adam优化、谱分析。

**📊 数据集**

MNIST（手写数字）和 CIFAR‑10（彩色图像）。

**📈 对比分析**

与完全随机、线性感知机及全训练多层感知机比较；在 MNIST 上 k=16 的 HalfNet 能与 MLP 匹配精度；在 CIFAR‑10 上 Full–Half 配置 k_f=32 能以约八倍参数更少的方式达到甚至略超全训练基线。

**⚠️ 局限性**

对高输入维度且相关性强的全连接层效果最佳；卷积层受限于局部性和共享权重，低秩约束效应有限；二值化在低潜在维度时精度下降；需在更多数据集和更大网络上验证。

---

## 325. SCI-PRM: A Tool Aware Process Reward Model for Scientific Reasoning Verification

**arXiv ID:** 2606.04579 | [PDF](https://arxiv.org/pdf/2606.04579v1)

**作者:** Xiangyu Zhao `[一作]` (Hong Kong Polytechnic University), Xiao-Ming Wu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 7587 | [OpenAlex ID](https://openalex.org/A5101981128)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Sci-PRM，一种专门针对科学领域的工具感知过程奖励模型，能够对每一步的工具选择、调用与结果解释进行细粒度评估；

**💡 创新点**

创新点在于构建了SCIPRM70K大型数据集，采用自动化两阶段标注流程实现高质量工具使用评估，并将过程监督融入奖励模型，使其在无需实时工具执行的前提下提供密集反馈；

**🔧 技术方法**

技术包括多模态LLM（Qwen3-VL-8B）微调、MCTS一致性检验、动态优势策略优化（DAPO-GRPO）以及最佳路径选择（Best-of-N）和强化学习训练；

**📊 数据集**

使用了SCIPRM70K数据集（约17.8k科学问答轨迹、86k步骤）以及BioProBench、ChemBench、Mol-Instructions、MSEarth等四个科学基准；

**📈 对比分析**

与多种基准模型（GPT-5-Mini、Gemini-3-Flash、Llama-3.2-11B-V、Qwen3-VL-8B/32B等）以及通用PRM Skywork 进行对比，Sci-PRM在工具调用评估上全局F1提升至0.5619，整体F1 0.7691，且在最佳路径选择与RL训练中分别显著超过ORM和其他基线；

**⚠️ 局限性**

局限性在于仍依赖于预训练模型的知识，工具调用的准确性受限于训练数据覆盖范围；对某些极端复杂的科学工具（如长时间计算的仿真）仍可能无法完全避免误判；

---

## 326. Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets with Deep Reinforcement Learning

**arXiv ID:** 2606.04574 | [PDF](https://arxiv.org/pdf/2606.04574v1)

**作者:** Damian Lebiedź `[一作]` (University of Warsaw), Robert Ślepaczuk `[通讯]` (University of Warsaw)

**通讯引用:** 520 | [OpenAlex ID](https://openalex.org/A5031588007)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本论文提出一种基于深度强化学习的交易执行层，旨在提升加密货币市场中的配对交易收益，

**💡 创新点**

创新点在于将统计套利的配对筛选与“固定风险、动态均值”执行框架相结合，并通过确定性屏蔽（shielding）实现安全强化学习；

**🔧 技术方法**

主要技术包括层次化的“Filter‑then‑Rank”配对选择、PPO‑LSTM 强化学习执行策略、以及多维奖励函数设计与安全屏蔽层；

**📊 数据集**

使用的数据集为 Binance USD‑M Futures 2023‑2025 年间的 1 小时 OHLCV 价格，按月滚动构建交易宇宙；

**📈 对比分析**

与传统统计套利基线以及买入持有 BTC/EWP 基准相比，实验显示 RL 叠加的策略在 OOS 阶段实现 30.4% CAGR、60.2% 胜率、Sharpe 0.49，显著优于基线；

**⚠️ 局限性**

主要局限在于对高杠杆环境的依赖、对模型超参数敏感、以及在极端市场突发事件下可能仍出现不可预知的损失。

---

## 327. SurvPFN: Towards Foundation Models for Survival Predictions

**arXiv ID:** 2606.04564 | [PDF](https://arxiv.org/pdf/2606.04564v1)

**作者:** Samuel Böhm `[一作]` (University of Freiburg), Pascal Schlosser `[通讯]` (University of Freiburg)

**通讯引用:** 46689 | [OpenAlex ID](https://openalex.org/A5071554578)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并训练了一种基于先验数据拟合网络（PFN）的生存预测模型 SurvPFN，利用合成 Weibull 生存任务训练模型，使其能够在有删失数据的时间到事件预测中直接回归连续时间。

**💡 创新点**

创新点在于：①将 PFN 迁移到生存分析，将生存问题视为分布式回归并采用删失负对数似然（IPCW 加权）训练；②在模型输入中加入事件指示器，允许模型区分已观测事件和删失记录；③无需为每个数据集单独调参或设计专用架构，证明小型 PFN 可与传统和深度生存基线竞争。

**🔧 技术方法**

使用技术包括：先验数据拟合网络（PFN）框架；通过结构因果模型（SCM）生成合成生存数据，事件时间服从 Weibull 分布；IPC 右删失负对数似然及对数对数排名损失；小型 Transformer 变压器骨干网络；软最大输出解读为离散密度，用于预测生存函数及中位数。

**📊 数据集**

数据集：合成先验数据用于预训练；实测在 SurvSet（22 个公开临床、经济、可靠性领域的小型生存数据集，最多 1000 行、10 个特征）上进行 5 折交叉验证。

**📈 对比分析**

对比方法：CoxPH、随机生存森林（RSF）、DeepSurv 以及 BinSurv；评价指标为 C-index、Integrated Brier Score (IBS) 与 Integrated Calibration Index (ICI)。SurvPFN 的表现与上述四种基线相当，平均排名与 RSF、DeepSurv 相当甚至略优，CoxPH 仍略优，但 SurvPFN 在不做数据集特定调参或使用专用架构的前提下即可达到相同性能层次。

**⚠️ 局限性**

局限性：仅评估小规模（≤1000 行、≤10 特征）的数据集；先验数据仅使用 Weibull 分布，可能无法覆盖所有真实生存曲线；模型仅处理单一静态事件，未考虑竞争风险或时间变 covariate；扩展到大规模或更复杂场景需要更大骨干网络和更通用的先验。

---

## 328. Cartridges at Scale: Training Modular KV Caches over Large Document Collections

**arXiv ID:** 2606.04557 | [PDF](https://arxiv.org/pdf/2606.04557v1)

**作者:** Momchil Hardalov `[一作]` (Amazon), Adrià de Gispert `[通讯]` (Amazon)

**通讯引用:** 1358 | [OpenAlex ID](https://openalex.org/A5041559636)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种可扩展的多卡车学习框架 CAS，用于训练和部署每个文档的 KV 缓存卡车，以实现长文本上下文的高效压缩与推理。

**💡 创新点**

创新点在于引入混合可见性训练、GPU↔持久存储的预算管理和动态卡车轮换机制，解决了单卡车混合导致的灾难性干扰，使得数百个卡车能够在相同 GPU 预算下保持高精度。

**🔧 技术方法**

主要技术包括上下文蒸馏、自学数据合成、按长度比例采样、批量多问生成、卡车专属初始化、Adam + per‑cartridge warmup、GPU 与磁盘的高效切换以及动态 RoPE 偏移。

**📊 数据集**

使用了 LongHealth、QASPER、QuALITY、T^2‑RAGBench/FinQA、TechQA 五个长上下文 QA 数据集进行实验。

**📈 对比分析**

与单卡车、单一大卡车、文本 RAG 等基线比较，CAS 在 2–100× 压缩率下多任务提升 10–30 分，卡车 RAG 在 3–4 倍更少提示词的条件下可匹配或超过文本 RAG 的准确率。

**⚠️ 局限性**

局限性包括：需在单轮 QA 预置卡车，难以在多轮对话中动态插入；检索仍基于原始文本，未针对卡车嵌入设计检索器；实验仅在英语数据与单一模型规模上验证；高信息密度任务如 FinQA 在大压缩率下仍表现不佳。

---

## 329. LDARNet: DNA Adaptive Representation Network with Learnable Tokenization for Genomic Modeling

**arXiv ID:** 2606.04552 | [PDF](https://arxiv.org/pdf/2606.04552v1)

**作者:** Daria Ledneva `[一作]` (Moscow Independent Research Institute of Artificial Intelligence), Denis Kuznetsov `[通讯]` (Moscow Independent Research Institute of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了LDARNet，一种120M参数的层次化DNA自适应分割模型；

**💡 创新点**

创新点在于将H-Net的动态分割迁移到掩码语言模型，并通过双向路由与EMA反分割实现自监督的生物学意义分词；

**🔧 技术方法**

采用BiMamba-2状态空间层、局部注意力、双向路由、比率正则化、以及无监督的自适应tokenization；

**📊 数据集**

使用人类基因组与Nucleotide Transformer多物种语料，长度至4096bp；

**📈 对比分析**

在27个下游任务（NT与GB套件）进行微调，LDARNet在小模型（<300M）中赢得11/18 NT任务，并在5个组蛋白修饰任务上超越规模高达20倍的大模型；

**⚠️ 局限性**

局限性包括在极短的splice-site任务上性能不如固定网格分割；单阶段压缩限制了对超长上下文的处理；评估主要集中在分类微调，未覆盖零样本/少样本或更长序列的测试。

---

## 330. Trading Engagement for Sustainability: Carbon-Aware Re-ranking for E-commerce Recommendations

**arXiv ID:** 2606.04550 | [PDF](https://arxiv.org/pdf/2606.04550v1)

**作者:** Noah Lund Syrdal `[一作]` (University of California, Berkeley), Jorgen Bergh `[通讯]` (University of California, Berkeley)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对缺失产品碳足迹标签的电商场景，提出了一种基于检索增强的大语言模型估计 PCF 的方法，并在已有推荐模型基础上实现了以 λ 为权重的后置再排序，兼顾用户参与度与碳排放。

**💡 创新点**

创新点在于：① 将检索与少样本 LLM 结合来推断大规模商品的 PCF；② 用可调 λ 参数实现透明的参与度-碳排放折中；③ 在三类电商品类和三种推荐基线上构建 Pareto 前沿，系统评估碳减排与推荐质量的关系。

**🔧 技术方法**

技术包括：句子嵌入（Sentence-Transformers）检索相似商品；少样本检索增强 LLM 推理 PCF；RecBole 框架下的 BPR、NeuMF、LightGCN 生成候选列表；min–max 标准化后线性组合再排序；NDCG@10 与 AvgPCF@10 评估。

**📊 数据集**

使用 Amazon Reviews 数据集，分别构建 Home & Kitchen、Sports & Outdoors、Electronics 三类子数据集，包含数万用户、约2万商品及百万级交互记录；PCF 值来自 Carbon Catalogue 通过检索-LLM 估计。

**📈 对比分析**

通过在 λ ∈ [0,1] 的 25 步网格上对每种基线进行再排序，并计算 NDCG@10 与 AvgPCF@10，绘制 Pareto 前沿；结果显示，在保持 NDCG 下降 ≤5% 的前提下，BPR、LightGCN 可实现 70–86% 的碳削减，NeuMF 的碳灵活度较低；不同品类对 λ 的敏感性也不同。

**⚠️ 局限性**

主要限制：① 评估基于离线 Review 数据，受样本偏倚与曝光偏差影响；② PCF 预测依赖 LLM，误差会影响再排序；③ 未进行在线 A/B 或用户实验，无法验证实际购买行为变化；④ 仅关注碳排放，忽略推荐系统自身的计算碳足迹。

---

## 331. Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models

**arXiv ID:** 2606.04535 | [PDF](https://arxiv.org/pdf/2606.04535v1)

**作者:** Boyan Han `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**通讯引用:** 27007 | [OpenAlex ID](https://openalex.org/A5100458183)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练的动态填空锚点方法（DIA），用于在扩散大型语言模型中实现格式约束生成。

**💡 创新点**

通过单步预测动态估计锚点位置，实现生成长度自适应，避免固定锚点导致的截断或冗余。

**🔧 技术方法**

利用扩散LLM的双向注意力与并行生成，并结合两阶段（长度调整与迭代去噪）动态锚点定位。

**📊 数据集**

在GSM8K、MATH数学推理基准和WikiBio JSON生成任务上进行评估。

**📈 对比分析**

与Dream-7B基线及固定锚点填空对比，DIA在格式合规率和答案准确率上分别提升至72.63%/46.78%（GSM8K）和76.82%/20.08%（MATH），JSON有效率79.84%且幻觉率仅0.15%。

**⚠️ 局限性**

依赖手工设定锚点、迭代长度调整增加推理开销，且对动态结构或多模态任务的适应性有限。

---

## 332. Learning Admissible Heuristics via Cost Partitioning

**arXiv ID:** 2606.04597 | [PDF](https://arxiv.org/pdf/2606.04597v1)

**作者:** Hugo Barral `[一作]` (LAAS-CNRS), Sylvie Thiébaux `[通讯]` (Australian National University)

**通讯引用:** 2959 | [OpenAlex ID](https://openalex.org/A5072296738)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

学习模型预测规划中可行的成本分配，使得求解器在搜索过程中仍能使用可接受的启发式估计。

**💡 创新点**

首次将成本分配与拉格朗日乘子对应的理论等价性用于学习，保证预测分配严格满足可接受性约束。

**🔧 技术方法**

利用图表示的规划状态与模式，使用改进的 Weisfeiler–Leman 算法提取动作级特征，随后通过轴向自注意力网络和 softmax 输出生成成本权重。

**📊 数据集**

在 IPC（International Planning Competition） 2023 以及 2026 的六个最佳规划域的合成任务上进行训练和评估。

**📈 对比分析**

与贪心、饱和及全局最优成本分配基线对比，学习得到的启发式在多数实例上节点扩展更少、覆盖率与最优分配相近，但单步评估成本更高，整体速度落后于基线。

**⚠️ 局限性**

主要局限是特征提取与抽象启发式计算仍在 CPU 上，导致评估时间占比高；图表示缺乏静态原子信息，影响在 logistics 与 miconic 等域的性能；模型只复制最优分配，未直接优化启发式信息。

---

## 333. Multilingual Long-Form Speech Instruction Following: KIT's Submission to IWSLT 2026

**arXiv ID:** 2606.04730 | [PDF](https://arxiv.org/pdf/2606.04730v1)

**作者:** Enes Yavuz Ugan `[一作]` (Karlsruhe Institute of Technology), Alexander Waibel `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 6434 | [OpenAlex ID](https://openalex.org/A5023053982)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将短语音数据进行段拼接、LLM标签生成与跨语言翻译，构建了超过100万实例的长语音指令跟随训练集，并在此基础上实验了固定概率采样与温度缩放采样，发现温度缩放（T=2）在多模态任务中效果最佳；此外，作者发现CoT任务标记条件导致任务识别崩溃，并提出了相应的改进；最后，提出了一种组合Likelihood+MBR的重排序方法以平衡转写与语义任务的性能。

**💡 创新点**

创新点主要包括：①统一的长语音数据增强框架，能够将多种短语音资源高效扩展为长语音训练集；②在多模态任务中系统评估温度缩放采样，证明T=2为良好默认；③对CoT任务标记条件进行负实验，揭示其在任务识别上的弱点；④设计了在未知任务身份约束下的多任务重排序策略，并提出Likelihood+MBR的组合方案，实现了转写和语义任务的兼顾。

**🔧 技术方法**

技术手段包括：LLM（Gemma、Qwen、Translategemma）用于标签生成与翻译；音频转写模型（Whisper、Parakeet）用于长语音转写；多模态LLM（Qwen2.5-Omni）用于指令跟随；重排序采用模型似然、MBR及其组合。

**📊 数据集**

使用的数据集包括：YTSeg、NUTSHELL、EuroParl-ST、CoVost、LibriSpeech、LibriSQA、MMSU等。通过拼接、LLM标注与翻译，最终得到包含六大任务（ASR、ST、SQA、SSUM、ACHAP、未知任务）和四种语言（英语、德语、意大利语、中文）的100万+实例训练集。

**📈 对比分析**

与基线相比，温度缩放采样和在NUTSHELL上做域内微调能显著提升多任务性能；在官方评测中，主模型在SQA、SSUM、ACHAP上优于对比模型，而对比模型在ASR、ST、QE等方面更强。重排序中，Likelihood+MBR在英语和中文可将ASR误差下降约24点，同时仅减少语义任务的损失约3点，显示出最佳的综合性能。

**⚠️ 局限性**

局限性包括：对任务识别仍依赖弱的前缀路由，导致CoT任务标记条件失效；重排序方案在德语、意大利语上效果不稳定；模型缺乏对未知任务的普适适配能力；长语音训练仍受限于短语音拼接的人工合成，可能引入语音一致性与语义连贯性问题。

---

## 334. Query-based Cross-Modal Projector Bolstering Mamba Multimodal LLM

**arXiv ID:** 2606.04719 | [PDF](https://arxiv.org/pdf/2606.04719v1)

**作者:** SooHwan Eom `[一作]` (Korea Advanced Institute of Science and Technology), Chang D. Yoo `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 6559 | [OpenAlex ID](https://openalex.org/A5073287748)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了基于查询的跨模态投影器 Q-Mamba，将视觉特征通过跨注意力压缩为可变长度的序列，以提高 Mamba 架构在视觉语言模型中的效率。

**💡 创新点**

引入可学习查询结合 Mamba 层的跨模态投影，消除手工二维扫描顺序的依赖，并通过局部注意力实现视觉 token 的动态下采样。

**🔧 技术方法**

结合 Mamba 结构、跨注意力、局部注意力掩码、可学习查询、预训练 SigLIP 视觉编码器和预训练 Mamba LLM，以及两阶段微调。

**📊 数据集**

对齐阶段使用 CC3M 过滤数据集；微调阶段使用 LLaVA v1.5、LVIS-Instruct-4V、LRV-Instruct 等多源视觉指令数据集。

**📈 对比分析**

与 Cobra、VL-Mamba 等基线在 VQA‑v2、GQA、VizWiz、Text‑VQA、POPE、MMBench 等六大基准上对比，Q‑Mamba 在所有指标上均优于前者，查询数增大时性能提升显著。

**⚠️ 局限性**

受限于训练数据规模与单轮微调、Mamba 序列遗忘问题以及缺乏对查询注意力的深入分析和对比。

---

## 335. CoRe-MoE: Contrastive Reweighted Mixture of Experts for Multi-Terrain Humanoid Locomotion with Gait Adaptation

**arXiv ID:** 2606.04718 | [PDF](https://arxiv.org/pdf/2606.04718v1)

**作者:** Kailun Huang `[一作]` (Hong Kong University of Science and Technology), Haohui Huang `[通讯]` (Guangdong University of Technology)

**通讯引用:** 385 | [OpenAlex ID](https://openalex.org/A5008084684)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个两阶段强化学习框架CoRe-MoE，实现了在单一策略下自然的步行与奔跑平滑切换并适应多种复杂地形

**💡 创新点**

创新点在于将Mixture-of-Experts与SwAV对比学习相结合，先学习稳定的步态再通过对比学习让专家针对不同地形实现专门化，并通过加权融合保持步态自然

**🔧 技术方法**

采用PPO强化学习、Adversarial Motion Priors、Mixture-of-Experts、深度视觉感知、SwAV对比学习以及加权动作融合

**📊 数据集**

使用AMASS、LAFAN的动作样本做运动先验，Isaac Sim的多地形仿真环境，Unitree G1机器人配备RealSense D435i深度摄像头进行真实世界测试

**📈 对比分析**

与基线（混合AMP、Sel AMP、无MoE、无SwAV、无Fusion、无MoE、掩蔽专家）以及Hiking in the Wild、MoRE进行对比，CoRe-MoE在平地与多地形下的成功率、速度跟踪、步态连贯性均显著提升，并在真实机器人上实现零调优的顺利迁移

**⚠️ 局限性**

局限在于仅覆盖步行/奔跑两种基本步态，对更动态或多样化的步态（如跳跃、蹲伏）支持有限，且在极端地形或极端外部扰动下的鲁棒性仍需进一步验证

---

## 336. Rethinking Continual Experience Internalization for Self-Evolving LLM Agents

**arXiv ID:** 2606.04703 | [PDF](https://arxiv.org/pdf/2606.04703v1)

**作者:** Jingwen Chen `[一作]` (Renmin University of China), Yankai Lin `[通讯]` (Renmin University of China)

**通讯引用:** 13009 | [OpenAlex ID](https://openalex.org/A5043098453)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个可持续的多迭代经验内化框架，使大型语言模型能够将过去交互中的经验转换为可参数化的能力，实现自我演化。

**💡 创新点**

创新点包括：①将经验粒度分为原则级和实例级，并证明原则级经验更具长期鲁棒性；②设计步步注入模式，使经验与中间决策状态对齐，显著提升长期工具使用效果；③引入离线（off‑policy）上下文蒸馏，提供更一致、可扩展的监督信号，解决传统在线蒸馏的局部纠正瓶颈。

**🔧 技术方法**

核心技术包括：ReAct式交互框架、经验抽象（使用DeepSeek-V4生成自然语言经验）、经验注入器（基于LLM的状态感知选择器）、离线与在线上下文蒸馏（对数似然+KL散度）、多迭代自我演化循环。

**📊 数据集**

使用了15K条公开Web推理问答数据（WebWalkerQA‑silver、DeepDive、WebShaper、WebDancer、SailorFog‑QA）生成经验池，并在WebWalkerQA、GAIA‑Text‑103、BrowseComp‑ZH三大基准上进行评测。

**📈 对比分析**

与单轮蒸馏、全局注入以及在线蒸馏等基线对比，步步注入+原则级经验+离线蒸馏在三大基准上实现了显著提升（如WebWalkerQA Pass@1从约23%提升至31%），且在多轮自我演化过程中保持稳定甚至持续提升，避免了能力崩塌。

**⚠️ 局限性**

局限性包括：实验仅在Web推理任务上验证，未评估不同领域、语言或更复杂任务的泛化；经验池大小、选择器质量及过滤标准等因素对稳定性影响尚未系统探究；若经验中包含错误或偏见，可能导致不良行为强化。

---

## 337. DuDi: Dual-Signal Distillation with Cross-Lingual Verbalizer

**arXiv ID:** 2606.04694 | [PDF](https://arxiv.org/pdf/2606.04694v1)

**作者:** Patomporn Payoungkhamdee `[一作]` (VISTEC), Peerat Limkonchotiwat `[通讯]` (AI Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 DuDi 的小语言模型多语言蒸馏框架，结合序列级自我对弈、离线和在线 token 级监督以及跨语言 verbalizer 进行联合训练。

**💡 创新点**

创新点包括：①将序列级自我对弈与 token 级蒸馏统一到同一目标；②使用双重 token 监督（离线+在线）避免教师-学生不匹配；③设计跨语言 verbalizer 促进教师与学生在不同目标语言上的知识迁移；④采用逆 KL 作为蒸馏损失，提升稳定性。

**🔧 技术方法**

主要技术：知识蒸馏、序列级自我对弈（SPIN）、自我蒸馏、逆 KL 损失、跨语言 verbalizer、冷启动 SFT 预训练。

**📊 数据集**

训练数据：SEA‑Instruct（涵盖印尼语、越南语、泰语、泰米尔语、塔加洛语、马来语、缅甸语，总计 28,000 条高质量示例）。评估数据：SEA‑HELM，覆盖 NLU、NLG、NLR、安全、语言诊断、指令遵循等多任务。

**📈 对比分析**

与 SFT、DFT、SPIN、SDFT、SeqKD、GKD 等方法对比。DuDi 在 SEA‑HELM 上平均得分 10.1，显著优于最强基线 DFT（+0.4 分）和 SPIN（+0.6 分），在多种语言和模型规模下保持鲁棒性。

**⚠️ 局限性**

局限性：实验聚焦于东南亚七语，未验证跨语言范围更广；需要更大规模的教师模型且必须共享输出词表；跨语言 verbalizer 的设计与语料覆盖相关，扩展到其他语言体系可能需要重新设计。

---

## 338. MeshWeaver: Sparse-Voxel-Guided Surface Weaving for Autoregressive Mesh Generation

**arXiv ID:** 2606.04688 | [PDF](https://arxiv.org/pdf/2606.04688v1)

**作者:** Jiale Xu `[一作]` (Tencent PCG), Ying Shan `[通讯]` (Tencent PCG)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种名为MeshWeaver的自回归3D网格生成框架，将网格生成过程视为表面织造。

**💡 创新点**

创新点在于将坐标级预测转为顶点级预测，并通过多层稀疏体素编码器实现局部几何引导、交叉注意力和结构支架，显著压缩令牌长度并提升几何保真度。

**🔧 技术方法**

核心技术包括稀疏体素编码器、分层多级顶点表示、交叉注意力引导的自回归Transformer、以及KV缓存加速推理。

**📊 数据集**

使用了由Objaverse++、ShapeNet、3D‑Future、HSSD和ABO构成的800K网格语料库，并在Toys4K数据集上进行评估。

**📈 对比分析**

与MeshAnythingV2、EdgeRunner、BPT、TreeMeshGPT、Mesh‑Silksong等基线相比，MeshWeaver在Chamfer距离、Hausdorff距离和法向一致性指标上分别提升了约30%、25%和10%，且压缩比达18%。

**⚠️ 局限性**

局限性包括对稀疏体素分辨率的依赖、在更大尺度网格上的计算开销仍需优化，以及在极端细节丰富场景下可能仍出现细节缺失。

---

## 339. Indexicon: A Spatial Indexing Library

**arXiv ID:** 2606.04676 | [PDF](https://arxiv.org/pdf/2606.04676v1)

**作者:** Panagiotis Simatis `[一作]` (University of Ioannina), Nikos Mamoulis `[通讯]` (University of Ioannina)

**通讯引用:** 12194 | [OpenAlex ID](https://openalex.org/A5045731304)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个统一、头文件式、可扩展的开源空间索引库Indexicon，集成了R‑tree、Quad‑tree、Oct‑tree以及KD‑tree等主流空间索引，并提供统一的 API 支持批量构建、动态增删、范围查询、kNN 查询和结构统计；

**💡 创新点**

创新点在于：1）将多种索引实现为单文件模板，消除外部依赖和层次复杂性；2）统一接口与一致的性能优化策略，使不同索引可在相同实验框架下直接比较；3）在主内存环境下通过局部性友好布局实现与现有最先进实现相当甚至更优的性能；

**🔧 技术方法**

主要技术包括：头文件式模板实现、递归 top‑down 打包（STR‑like）、R*-tree 强制重插、分区策略多样化（PR、伪中位、最长轴）、KD‑tree 适配器、批量构建与增删融合、最佳优先 kNN 搜索、结构统计暴露；

**📊 数据集**

使用了六个真实地理数据集：MARINE（AIS 3D 点）、MIAMI（3D MBB）、OSM（2D 点）、TAXIS（2D 点）、TIGER（2D MBB）和 TORONTO（3D 点云），分别覆盖不同维度、数据类型与空间分布；

**📈 对比分析**

比较方法：在相同硬件/编译器/内存环境下，对比 Boost Geometry、GEOS、PCL、Nanoflann 等主流实现；实验评估了构建、动态插/删耗时以及 1000 次范围和 kNN 查询（k=1,10,100）的总耗时；结果显示 Indexicon 在构建与动态更新上通常优于对比实现，查询性能与现有实现相当或更快（最多 2× 的加速）；

**⚠️ 局限性**

局限性包括：目前仅支持单线程、缺乏自平衡机制（如 KD‑tree 的动态重平衡）、不支持多维 MBB 的三维 MX‑CIF 或更高级的空间索引、对极端高维数据支持不足，以及在 GPU/多线程场景下的性能尚未验证。

---

## 340. SoK: Post-Quantum Cryptography (PQC) Implementation in Software Systems

**arXiv ID:** 2606.04669 | [PDF](https://arxiv.org/pdf/2606.04669v1)

**作者:** R. D. N. Shakya `[一作]` (University of Moratuwa), Nalin A. G. Arachchilage `[通讯]` (RMIT University)

**通讯引用:** 1963 | [OpenAlex ID](https://openalex.org/A5081069489)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该研究对软件系统中后量子密码（PQC）实现方法进行了系统化综述，归纳了指南、框架、工具与教育干预等四类，并通过人‑组织‑技术（HOT）视角分析了实现挑战与局限。

**💡 创新点**

创新点在于提出了PQC‑HOT模型，将人、组织与技术维度耦合成系统化框架，用以解释实现中的交叉依赖与跨维度风险，并为未来研究提出了整合化路线。

**🔧 技术方法**

主要技术为系统文献综述与主题分析法（Braun & Clarke），以及HOT框架的运用与PQC‑HOT模型的构建。

**📊 数据集**

研究使用了近五年（2020‑2026）内的29篇同行评审论文和标准文献作为数据集。

**📈 对比分析**

对比方法为将收集的实现方案按类别和维度映射，并用层次化主题评估其覆盖度；结果表明技术层面覆盖最广，而人/组织层面明显不足，系统级性能与兼容性仍未得到统一评估。

**⚠️ 局限性**

局限在于仅依赖公开学术文献，缺乏实证验证与行业案例；模型与结论主要基于文献描述，未进行实验或用户研究验证。

---

## 341. Why Muon Outperforms Adam: A Curvature Perspective

**arXiv ID:** 2606.04662 | [PDF](https://arxiv.org/pdf/2606.04662v1)

**作者:** Shuche Wang `[一作]` (National University of Singapore), Zhuoran Yang `[通讯]` (Yale University)

**通讯引用:** 4831 | [OpenAlex ID](https://openalex.org/A5101727948)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 Muon 优化器在大型语言模型（LLM）预训练中相对于 Adam 的优势进行几何（曲率）分析，并从理论上解释 Muon 的更新方向为何能获得更大的一步损失下降。

**💡 创新点**

创新点包括：① 将 Muon 的更优表现归因于其更低的 Normalized Directional Sharpness (NDS)，即更新方向在 Hessian 方向上的曲率更小；② 证明数据不平衡会放大 Muon 与 Adam 之间的 NDS 差距；③ 在层级维度上拆解 NDS，发现 Muon 的优势主要来自较小的 within‑layer 曲率；④ 在满足低 Kronecker 秩、共同对角化、曲率异质性和梯度对齐等假设下，对 Muon 在二次模型上的曲率优势与损失下降进行解析证明。

**🔧 技术方法**

使用的技术包括：第二阶泰勒展开、NDS 定义与分解、矩阵谱归一化（Muon 的正交化动量）、跨层与 within‑layer 曲率拆分、低秩 Kronecker 近似、共同对角化判定、理论上对 Muon 与 GD 的比较。

**📊 数据集**

主要使用数据集为：FineWeb（真实 LLM 预训练数据）和由 Zipf‑PCFG 生成的合成数据（可控制类别不平衡程度 s∈{0,0.5,1}），并在 124M NanoGPT（FineWeb）和 9M NanoGPT（合成）上进行实验。

**📈 对比分析**

比较方法：在相同验证损失（或训练步数）下对 Muon 与 Adam 的一阶下降、二阶曲率惩罚、NDS、更新范数等指标进行对比。结果显示：Muon 的更新范数与 Adam 相当，但其 NDS 明显更小，从而导致更小的曲率惩罚，进而实现约 2 倍更快的训练速度和更大的一步损失下降。

**⚠️ 局限性**

局限性：仅在因果 LLM 预训练场景下验证，未探讨其他模型类（如扩散模型）；理论分析基于若干假设（低 Kronecker 秩、共同对角化、曲率与梯度对齐），在更复杂或不满足这些假设的实际模型中可能不完全成立。

---

## 342. CRAFT: Cost-aware Refinement And Front-aware Tuning of Prompts

**arXiv ID:** 2606.04661 | [PDF](https://arxiv.org/pdf/2606.04661v1)

**作者:** Shanu Kumar `[一作]` (MBZUAI), Manish Gupta `[通讯]` (Microsoft)

**通讯引用:** 5529 | [OpenAlex ID](https://openalex.org/A5101454729)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Pareto 前沿搜索的提示词优化框架 CRAFT，能够同时优化准确率与推理成本；

**💡 创新点**

创新点在于将多目标优化拆解为前沿感知的候选生成、Pareto‑gap UCB 采样与 NSGA‑II 保留机制，避免传统标量化导致的前沿坍塌；

**🔧 技术方法**

采用 LLM 进行结构化提示词精细化与压缩、Pareto‑gap UCB 采样、Upper Confidence Bound、NSGA‑II 非支配排序、k‑means 验证子集轮换等技术；

**📊 数据集**

在六个英文分类/推理数据集上评估：BeaverTails、GoEmotions、DisambiguationQA、Causal Judgement、Formal Fallacies、Salient Translation；

**📈 对比分析**

与加权求和标量化和单轴基线对比，CRAFT 在保持高准确率的同时显著降低提示词长度，Pareto 前沿覆盖更广、峰值效率更高，总体排名第一；

**⚠️ 局限性**

局限性包括仅在英语短文本分类/推理任务验证，未覆盖生成式任务或非 OpenAI 目标模型；压缩模块表现有限；依赖较好初始提示，对弱或对抗性提示效果未知。

---

## 343. U-Net-Accelerated Quality-Diversity Optimization for Climate-Adaptive Urban Layouts

**arXiv ID:** 2606.04658 | [PDF](https://arxiv.org/pdf/2606.04658v1)

**作者:** Alexander Hagg `[一作]` (Bonn-Rhein-Sieg University of Applied Sciences), Dirk Reith `[通讯]` (Bonn-Rhein-Sieg University of Applied Sciences)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个基于U‑Net深度学习的空间代理模型，用来替代昂贵的KLAM_21冷空气流动模拟，并在该代理上运行离线MAP‑Elites，快速生成多样化、气候适应性强的城市建筑布局

**💡 创新点**

创新点在于：①利用空间卷积网络的空间先验，使代理能够学习完整的物理场映射，避免了传统标量高斯过程代理在高维空间中出现的伪极值；②证明只需使用Sobol随机采样的训练数据即可，无需昂贵的QD自举；③提出离线QD优化流程，在不需任何物理评估的情况下实现超过12,000倍的评估速度提升

**🔧 技术方法**

使用的技术包括：U‑Net深度卷积网络、Sparse Variational Gaussian Process (SVGP) 作为对比、离线MAP‑Elites算法、Sobol准随机采样、OpenSKIZZE开源工具、GPyTorch、Numba JIT、CUDA加速

**📊 数据集**

训练数据来源于19,000次KLAM_21模拟，分别使用Sobol随机采样和Sail自举采样；测试数据为已评估的建筑布局集合，覆盖不同特征维度（建筑面积、密度、高度、空间开放度等）

**📈 对比分析**

比较结果显示：U‑Net在Sobol训练下即可实现R²≥0.996，Spearman ρ≈0.994的评估准确度；SVGP仅在Sail训练下可达R²≈0.968，但在离线QD中表现差，Spearman ρ≤0.41；U‑Net在10分钟内完成3–8k种多样化布局，单个物理模拟需数小时，速度提升约12,000×

**⚠️ 局限性**

局限性包括：研究仅在简化的单个60 m方块、单一风向、均匀坡度的理想化情境下验证，真实城市环境中建筑形状多样、风向多变，代理精度和排名可信度可能下降；实验重复次数仅3次，建议未来扩大到10次以评估方差；缺乏对代理不确定性的可靠量化，未来可考虑贝叶斯网络或深度集成

---

## 344. CYGNET: Cypher Gate for Neural Execution Triage and Cost Containment

**arXiv ID:** 2606.04645 | [PDF](https://arxiv.org/pdf/2606.04645v1)

**作者:** Nikodem Tomczak `[一作]` `[通讯]` (Thulge Labs), Nikodem Tomczak (Thulge Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一个预执行门（validator chain、cost gate、correction loop），在Neo4j生产数据库前检测并修复LLM生成的Cypher查询的结构错误。

**💡 创新点**

将四层后端验证链、镜像图执行、EXPLAIN成本门、结构化错误词汇表以及可插拔纠错器整合为实时防御层，并公开首个带结构错误标签的Cypher语料。

**🔧 技术方法**

使用Cypher 9 ANTLR解析、Neo4j EXPLAIN、镜像图（synthetic graph）、成本门、基于提示的LLM纠错（RAMPART等）、MCP/HTTP接口、Token预算、错误词汇表等技术。

**📊 数据集**

使用9个不同形状的schema构造的模板生成语料、CypherBench 7个schema的2348个问题、外部683条有效查询、10个参考数据库做镜像等价验证等数据集。

**📈 对比分析**

通过比较前后执行准确率、错误捕获率、修复成功率、延迟等指标，门平均5.6 ms，捕获率100%，纠错成功率高达89%（RAMPART），对CypherBench无负面影响，提升约1.35个百分点。

**⚠️ 局限性**

存在对属性同类交换无法检测、AST语法不支持CALL子查询、镜像图简化（单端点关系、单标签）导致部分验证失效、成本门对某些模式过度或欠估计等限制。

---

## 345. Learning symplectic model reduction based on a approximation theorem of symplectic embeddings

**arXiv ID:** 2606.04623 | [PDF](https://arxiv.org/pdf/2606.04623v1)

**作者:** Liyi Feng `[一作]` (Beijing Jiaotong University), Aiqing Zhu `[通讯]` (National University of Singapore)

**通讯引用:** 344 | [OpenAlex ID](https://openalex.org/A5011403908)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种保持辛结构的自动编码器SpAE，用于对高维哈密顿系统进行非线性降维和长时预测；

**💡 创新点**

核心创新在于构建通用的辛嵌入逼近定理，将辛嵌入拆解为可学习的梯度层，使网络在任何权重设置下天然保持辛结构，并可通过无约束优化训练；

**🔧 技术方法**

技术手段包括梯度可微的辛剪切层（symplectic shear layers）、标准线性辛嵌入（S_c）、全连接神经网络实现潜在势能、HNN捕捉潜在动力学以及标准优化器和辛数值积分器；

**📊 数据集**

实验数据来自三类高维哈密顿系统：一维晶格模型（1000维）、托卡马克磁场粒子（3000维）和两流不稳定模型（2000维）等；

**📈 对比分析**

与传统线性辛降维方法COT、cSVD、NLP以及POD比较，SpAE在重构误差和全程预测误差上实现了1–2个十倍的提升，显著降低了误差；

**⚠️ 局限性**

局限性包括对极端非线性和高维压缩仍需更大训练样本和网络容量，且目前仅针对可压缩的哈密顿系统，缺乏严谨的误差理论和对非辛结构的推广。

---

## 346. Data Efficient Complex Feature Fusion Network For Hyperspectral Image Classification

**arXiv ID:** 2606.04710 | [PDF](https://arxiv.org/pdf/2606.04710v1)

**作者:** Maitreya Shelare `[一作]` (University of Mumbai), Sneha Burnase `[通讯]` (University of Mumbai)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5035486879)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种数据高效的双分支复杂特征融合网络（DE-CFFN）用于高光谱图像分类

**💡 创新点**

用因子分析代替主成分分析实现更好的维度压缩，并在两条流中逐层减半卷积核数以降低计算量，同时引入复杂域Squeeze‑Excitation注意力模块

**🔧 技术方法**

因子分析、双分支3D卷积（实值与复值）、频域FFT、复杂域SE块、全连接层

**📊 数据集**

Pavia University和Salinas两个公开高光谱数据集

**📈 对比分析**

与原CFFN、SpectralNET、HybridSN等模型对比，DE-CFFN在保持或略微提升整体准确率的同时，参数量下降72%，显存占用下降72%，推理时间仅略增（PU 12.15 ms vs 11.72 ms，SA 30.27 ms vs 35.06 ms）

**⚠️ 局限性**

对数据集规模有限、缺乏时序信息以及对多时相/跨传感器适应性尚未验证的局限

---

## 347. Graph-Guided Universum Learning in Generalized Eigenvalue Proximal SVMs for Alzheimer's Disease Classification

**arXiv ID:** 2606.04699 | [PDF](https://arxiv.org/pdf/2606.04699v1)

**作者:** Yogesh Kumar `[一作]` (Indian Institute of Technology Ropar), Mudasir Ganaie `[通讯]` (Indian Institute of Technology Ropar)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5082168964)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

针对阿尔茨海默病与正常认知的二分类问题，提出两种基于图引导的 Universum 学习模型 UG-GEPSVM 与 IUG-GEPSVM，利用 MCI 组作为 Universum 样本，并在 GEPSVM/IGEPSVM 框架中加入基于 MST 与多跳传播构造的图拉普拉斯正则化。

**💡 创新点**

创新点在于：①首次将 Universum 样本的几何结构（通过 Gaussian 相似度、最小生成树和多跳聚合构造的图）融入 GEPSVM 的正则化；②在 IGEPSVM 的数值稳定框架下保留此图正则化，形成标准特征值问题；③显著提升了在不同噪声水平下的鲁棒性与平均 AUC，优于传统 GEPSVM、U-TSVM 及先前的 UGEPSVM/IUGEPSVM。

**🔧 技术方法**

使用的技术包括：Generalized Eigenvalue Proximal SVM (GEPSVM)、改进版 IGEPSVM、图拉普拉斯正则化、MST 与多跳传播构造、ICA/PCA 特征提取、统计检验（Friedman、Nemenyi）以及对噪声鲁棒性的实验评估。

**📊 数据集**

采用 ADNI 结构 MRI 数据集，包含 788 CN、229 AD 与 391 MCI 受试者，利用 ICA 与 PCA 产生 155 维特征，进一步添加 0%–20% 高斯噪声产生十个数据变体。

**📈 对比分析**

与 GEPSVM、IGEPSVM、UTSVM、UGEPSVM、IUGEPSVM 等七种基线方法进行对比，使用 AUC 评估。UG-GEPSVM 的平均 AUC 达 88.07%，排名第一；IUG-GEPSVM 次之，平均 AUC 86.62%；两者在噪声水平升高时保持性能稳定，优于传统方法。

**⚠️ 局限性**

限制包括：只处理二分类问题；依赖手工构造的图（MST+多跳），对图构造参数敏感；未涉及非线性核扩展；仅使用单一模态（结构 MRI）数据，未融合 PET 或其他生物标志物。

---

## 348. Test-Time Compute Scaling for ASR with Depth-Conditioned Looped Transformers

**arXiv ID:** 2606.04678 | [PDF](https://arxiv.org/pdf/2606.04678v1)

**作者:** Yacouba Kaloga `[一作]` (Idiap Research Institute), Ina Kodrasi `[通讯]` (Novartis Institute of Biomedical Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了一种深度条件循环Transformer（LARM），通过在声学编码器中重复使用共享的Transformer块，将模型深度转化为可调的推理计算轴，从而在不增加参数数量的情况下实现测试时计算量可扩展的语音识别。

**💡 创新点**

创新点在于：1) 采用稀疏CTC检查点与定时器嵌入将循环分为有监督的识别阶段和无监督的精细化阶段；2) 引入FiLM深度条件对共享块进行阶段化调节；3) 设计了延迟预测反馈机制，将前一帧的软后验概率投射回隐藏空间，实现层级化的音素连贯性；4) 通过这些机制实现共享参数循环加速并提升识别性能。

**🔧 技术方法**

使用的技术包括：循环Transformer结构、共享参数编码器、稀疏CTC监督、超时钟（supervision-clock）嵌入、FiLM深度条件、延迟预测反馈、卷积下采样前端、softmax+CTC头、AdamW优化、SpecAugment以及4-gram KenLM语言模型。

**📊 数据集**

实验数据集为LibriSpeech，分别在100小时子集和960小时完整集上训练与评估。

**📈 对比分析**

与标准的4层未循环Encoder和16层未共享Encoder做对比；在相同参数量下，LARM（4层+12循环）在greedy CTC和4-gram LM解码时均优于16层Encoder，且随着循环次数增加WER持续下降，支持早停；在大规模数据上，LARM保持竞争力且参数更少。

**⚠️ 局限性**

局限性包括：1) 在宽度或循环预算极大时模型性能趋于饱和；2) 仅在LibriSpeech上验证，跨域/多说话人性能未知；3) 训练过程中需调节稀疏监督间隔和反馈权重，增加实现复杂度；4) 目前仍未探索更深共享块或更长循环预算的潜在收益。

---

## 349. Fitting scattered data with optional monotonicity constraints on GPU: LipFit package

**arXiv ID:** 2606.04670 | [PDF](https://arxiv.org/pdf/2606.04670v1)

**作者:** Gleb Beliakov `[一作]` (Deakin University), Gleb Beliakov `[通讯]` (Deakin University)

**通讯引用:** 8480 | [OpenAlex ID](https://openalex.org/A5011976667)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于Lipschitz连续性的实例化插值与逼近方法，并实现了支持单调性约束、局部Lipschitz常数以及GPU并行化的Python包 LipFit。

**💡 创新点**

创新点在于：① 用最优上下界求解全局Lipschitz插值，保证在最坏情况下误差最小；② 在此基础上通过距离修改实现单变量及部分域单调性；③ 引入局部Lipschitz常数与基于梯度的平滑求解，兼顾逼近与光滑；④ 所有计算均以SIMD方式实现，适配GPU加速。

**🔧 技术方法**

技术方法包括：最优Lipschitz插值公式（上界/下界取中点）、单调性约束下的距离变换、局部Lipschitz常数的分段线性逼近、基于线性规划/LBFGS的平滑正则化、以及Python Tensor/torch实现的并行化。

**📊 数据集**

本文未使用特定公开数据集，而是提供通用框架；在实验示例中主要演示了合成数据和小型toy例子。

**📈 对比分析**

与传统kNN、自然邻域、张量积样条、径向基函数等方法对比，作者声称在保持连续性、单调性和最坏误差最优的前提下，LipFit能在GPU上实现高并行度，适合大规模点集；但未给出定量实验结果。

**⚠️ 局限性**

局限性包括：① 对大规模点集（N>10⁵）需手动预筛选约束以降低GPU内存；② 局部Lipschitz常数估计依赖邻域采样，可能不稳定；③ 对极端高维（d>10）仍受维数灾难影响；④ 平滑过程仍需依赖线性规划或LBFGS，可能在极大数据量下效率下降。

---

## 350. TeleHunt: A Framework and Tool for Efficient Cybercriminal Community Discovery on Telegram

**arXiv ID:** 2606.04657 | [PDF](https://arxiv.org/pdf/2606.04657v1)

**作者:** Roy Ricaldi `[一作]` (Eindhoven University of Technology), Luca Allodi `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 1212 | [OpenAlex ID](https://openalex.org/A5047635330)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 TeleHunt 框架，自动化抓取 Telegram 群组/频道、对消息进行 LLM 级别分类、提取指针并通过雪球采样迭代扩展，从而系统评估不同种子、指针、上下文过滤与扩展策略对网络犯罪社区发现的影响。

**💡 创新点**

创新点在于：①提出可配置、模块化的 LLM 驱动发现管线；②首次对 Telegram 上的发现策略进行系统性比较与定量评估；③生成并公开 6,022 个社区、172M 条消息的标注数据集，支持后续研究。

**🔧 技术方法**

技术包括：Telegram API 抓取、基于 RoBERTa‑Large 的二分类与 6 类市场细分分类器、指针提取与上下文窗口筛选、迭代雪球扩展、Beta 回归与逻辑回归评估、SQLite 日志记录与可复现性支持。

**📊 数据集**

数据集：6,022 个 Telegram 社区（3,471 个被标记为网络犯罪）、172,385,463 条消息、2,392,741 个用户，来源为 41 个暗网种子与 45 个公开网种子，已标注市场细分、社区类型、规模与时间属性。

**📈 对比分析**

通过三维度指标（效率、可达性、重访率）对多种配置（种子来源、指针类型、上下文过滤、是否额外投射）进行定量比较。结果显示：链接（invite link）指针最优，开源种子略优于暗网种子；发现率最高达 0.52，产出率 0.40，精度 0.70‑0.82，噪声比例高达 40% 以上，且随迭代增长。

**⚠️ 局限性**

局限性：仅收集两周，时间窗口有限；种子覆盖范围有限，可能导致结构偏差；阈值设定（70% 价值阈、30 天窗口、30 条消息活跃度）未做敏感性分析；分类器受多语言、行业俚语与类别不平衡影响；Telegram API 限流与平台不稳定限制可扩展性。

---

## 351. Improving the Efficiency and Effectiveness of LLM Knowledge Distillation for Conversational Search

**arXiv ID:** 2606.04650 | [PDF](https://arxiv.org/pdf/2606.04650v1)

**作者:** Stan Fris `[一作]` (University of Amsterdam), Mohammad Aliannejadi `[通讯]` (University of Amsterdam)

**通讯引用:** 1860 | [OpenAlex ID](https://openalex.org/A5063466614)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在对话检索中改进KLD蒸馏，加入InfoNCE对比损失、探索采样比例平衡、加大正则化以提升稀疏性和推理速度。

**💡 创新点**

创新点在于将InfoNCE与KLD结合形成可控对比蒸馏目标，系统评估正负样本比例对KLD效果的影响，并证明可在保持性能的前提下显著提升稀疏性与FLOPs。

**🔧 技术方法**

使用的技术包括KLD蒸馏、InfoNCE对比损失、重要采样、L1稀疏正则、FLOPs度量、SPLADE++稀疏检索模型。

**📊 数据集**

实验使用TopiOCQA（基于Natural Questions的对话检索基准）数据集。

**📈 对比分析**

通过与DiSCo基线以及不同λ、采样数和正则化强度的组合进行对比，评估MRR、Recall@10/100、nDCG@3及FLOPs，结果显示10–20% InfoNCE提升排名，16负样本最优，较大正则化保持性能并将FLOPs降至约1/4，稀疏性显著提升。

**⚠️ 局限性**

局限性包括正则化过大时长对话召回下降、正负样本比例需精细平衡导致KLD效果不稳定、仅在单一TopiOCQA数据集验证，缺乏跨域或多语言泛化评估。

---

## 352. ALINC: Active Learning for Inductive Node Classification via Graph Sampling

**arXiv ID:** 2606.04647 | [PDF](https://arxiv.org/pdf/2606.04647v1)

**作者:** Pascal Plettenberg `[一作]` (University of Kassel), Josephine M. Thomas `[通讯]` (University of Greifswald)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ALINC 框架，能够在诱导节点分类任务中通过聚合节点级信息实现对整张图的主动学习采样。

**💡 创新点**

创新点在于：① 将传统节点级主动学习指标（不确定性、代表性、多样性）转化为图级采样准则；② 系统化评估 10 种采样策略与 3 种聚合方法的组合；③ 通过聚合函数的设计揭示聚合方式对模型性能与标注成本的显著影响。

**🔧 技术方法**

技术方法包括：图神经网络（GatedGCN）、图变压器（GPS）、多种主动学习策略（Entropy、Margin、Node-Density、Graph-Density、Degree、CoreSet、AGE、ANRMAB、TypiClust、BADGE）、聚合函数（mean、max、sum）以及批量主动学习循环。

**📊 数据集**

使用的数据集：四大基准图数据集（PATTERN、CLUSTER、PascalVOC‑SP、COCO‑SP），以及两个真实应用案例（Zaretzki SoM 分子数据集和 PCB pull‑up/down 方案图数据集）。

**📈 对比分析**

通过在 4 个基准集上执行 10 种策略 × 3 聚合的实验，采用 AULC、相对学习曲线和胜率进行比较。结果显示 TypiClust、CoreSet、BADGE 在大多数数据集上获得最高胜率，TypiClust 以近乎完美的胜率领先其余策略；同时聚合方式（尤其是 mean/mid）显著影响性能和标注成本。

**⚠️ 局限性**

局限性：① 对聚合方法敏感，某些策略在不同数据集上表现不稳；② 仅针对节点分类任务，未覆盖回归或多标签；③ 目前未引入可学习的聚合权重或自监督预训练；④ 对标注成本的评估仅以图大小为 proxy，未考虑更复杂的实验/仿真成本。

---

## 353. StrokeTimer: Robust Representation Learning for Ischemic Stroke Onset-Time Estimation from Non-contrast CT

**arXiv ID:** 2606.04722 | [PDF](https://arxiv.org/pdf/2606.04722v1)

**作者:** Weiru Wang `[一作]` (Eindhoven University of Technology), Ruisheng Su `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 6781 | [OpenAlex ID](https://openalex.org/A5046844818)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一套名为StrokeTimer的全自动框架，用于从常规非对比CT图像估计急性缺血性卒中患者的发病时间窗口（<4.5h、4.5–6h、>6h）。

**💡 创新点**

创新点在于将自监督语义‑风格解耦学习与能量引导的对比平均移位（ECMS）相结合，既能捕捉细微缺血变化，又能在样本量极不平衡和扫描仪差异的情况下实现稳健的分类；同时首次在大规模多中心临床数据上完成此类任务。

**🔧 技术方法**

使用了3D ResNeXt骨干网络、FiLM调制的解码器进行语义‑风格解耦，VAE‑style编码器与正则化损失；ECMS模块采用原型对比损失与频率加权的平均移位更新；训练分两阶段，先解耦再对比学习。

**📊 数据集**

实验数据来自MR CLEAN Registry和MR CLEAN‑LATE两大国立队列，包含18家中心共1,686份NCCT扫描（1,531<4.5h、72 4.5–6h、83>6h）。

**📈 对比分析**

与六种长尾分类方法（Focal Loss、Balanced Softmax、τ‑Norm、MulSupCon、SoftCon、MARC）以及四种自监督方法（MoCo、SimCLR、VoCo、MAE）对比，StrokeTimer在宏观指标上取得最高性能：宏观AUC 0.69、宏观F1 0.57，宏观准确率0.59，较最强基线提升近50%（p<0.005）。

**⚠️ 局限性**

局限性包括：在4.5–6h中间窗口样本极少，导致该类别的性能波动；模型结构和超参数相对复杂，可能影响部署；以及对不同CT扫描仪的泛化仍需进一步验证。

---

## 354. The State of Peer Review in Empirical Software Engineering: A Community Survey on Review Load, Quality, and GenAI Use

**arXiv ID:** 2606.04716 | [PDF](https://arxiv.org/pdf/2606.04716v1)

**作者:** Justus Bogner `[一作]` (Vrije Universiteit Amsterdam), Roberto Verdecchia `[通讯]` (University of Florence)

**通讯引用:** 1378 | [OpenAlex ID](https://openalex.org/A5041332104)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

调查了 ESE 社区同行评审的工作负荷、质量认知以及 LLM（大语言模型）在评审过程中的使用情况。

**💡 创新点**

首次系统化量化 ESE 领域评审者对评审负担、质量与 LLM 使用的感知，并揭示了 LLM 影响的主要渠道与社区态度。

**🔧 技术方法**

采用问卷调查、定量统计分析和主题（定性）分析等技术手段收集并处理数据。

**📊 数据集**

使用了 120 位 ESE 评审者（主要为资深学者）的匿名问卷数据。

**📈 对比分析**

通过对自评与他评质量、评审负担、LLM 使用频率等维度进行统计对比，展示了分布特征与关联趋势；未进行实验性能评估，仅给出描述性统计。

**⚠️ 局限性**

局限性包括样本偏向欧洲/北美资深学者、可能存在自我报告与社交期望偏差、LLM 使用被低估、缺乏对 LLM 评审质量的客观评估等。

---

## 355. ReConFuse: Reconstruction-Error Guided Semantic Fusion for AI-Generated Video Detection

**arXiv ID:** 2606.04706 | [PDF](https://arxiv.org/pdf/2606.04706v1)

**作者:** Xiaojing Chen `[一作]` (Anhui University), Yunfeng Diao `[通讯]` (Hefei University Of Technology)

**通讯引用:** 327 | [OpenAlex ID](https://openalex.org/A5026338785)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了 ReConFuse 框架，通过将 WF‑VAE 重建误差与 XCLIP 语义特征对齐并利用 Mamba 模块建模时序，来进行 AI 生成视频检测。

**💡 创新点**

创新点在于首次将视频重建误差作为低层法证线索，与语义上下文融合，并通过 Mamba 进行高效时序建模，显著提升了对多模型、跨域视频的检测鲁棒性。

**🔧 技术方法**

采用 WF‑VAE 作为重建先验、XCLIP 视觉编码器提取语义特征、Patch 级对齐与门控融合、Mamba 序列模块进行时序建模，以及二元交叉熵训练。

**📊 数据集**

在 GenVideo（one‑to‑many 与 many‑to‑many）和 GenBuster 两大公开基准上进行实验，使用多种生成器（如 Pika、Sora、Vidu 等）生成的合成视频与真实视频。

**📈 对比分析**

与 DeMamba、D3、DIRE、AEROBLADE 等基线相比，ReConFuse 在 Accuracy、F1‑score、AUROC 等指标上均取得了最优或最接近最优的成绩，尤其在多域泛化任务中展现出更高的平衡性能。

**⚠️ 局限性**

主要局限包括对 WF‑VAE 重建先验的依赖；在极度压缩或光照剧烈变化的情形下重建误差可能受噪声影响；以及对长时序视频的推理时间仍有提升空间。

---

## 356. Enhancing MedSAM with a Lightweight Box Predictor for Medical Image Segmentation

**arXiv ID:** 2606.04705 | [PDF](https://arxiv.org/pdf/2606.04705v1)

**作者:** Amirhossein Movahedisefat `[一作]` (Iran University of Science and Technology), Mohammad Reza Mohammadi `[通讯]` (Iran University of Science and Technology)

**通讯引用:** 11297 | [OpenAlex ID](https://openalex.org/A5100616844)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一个轻量级的Box Predictor模块，用于从单个用户点击生成近似边界框，以增强MedSAM在医学图像分割中的表现。

**💡 创新点**

创新点在于将自动生成的边界框作为额外的空间先验，引入两阶段训练策略，并实现对MedSAM的轻量级改造，显著提升对不规则或低对比度目标的鲁棒性。

**🔧 技术方法**

使用了MedSAM基础模型、ViT图像编码器、prompt encoder、mask decoder以及一个由MLP构成的Box Predictor，并采用L1+GIoU损失进行框回归，Dice+BCE损失进行分割。

**📊 数据集**

实验数据集包括FLARE22（CT腹部）、BRISC（MRI脑瘤）、BUSI（超声乳腺）和LungSegDB（CT肺部），共计3621张CT、4795张MRI、931张超声和1716张肺部图像。

**📈 对比分析**

与MedSAM+point、SAM、SAM2、SAM2UNet等基线对比，MedSAM+Box Predictor在四个数据集上Dice分别提升到0.931、0.889、0.881、0.983，IoU和Precision均有显著提升，且在不同模态下保持稳定性能。

**⚠️ 局限性**

局限性在于仅支持单一点击输入，无法同时处理多处或多分支目标；生成的边界框为粗略矩形，对形状不规则的病灶可能引入误差，且在高对比度且已精准定位的情形下可能无显著增益。

---

## 357. Benchmarking Living-Screen-Native GUI Agents on Short-Video Platforms

**arXiv ID:** 2606.04701 | [PDF](https://arxiv.org/pdf/2606.04701v1)

**作者:** Jiashu Yao `[一作]` (Beijing Institute of Technology), Yuhang Guo `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 1074 | [OpenAlex ID](https://openalex.org/A5101786500)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在持续播放的视频界面上自适应观测与操作的Living‑Screen‑Native GUI代理，并在短视频平台上构建了首个对应基准。

**💡 创新点**

首次把持续演化的屏幕与主动观测结合到GUI代理中，并将观测控制视为新的能力维度，构建了包含浏览器环境、三级任务和效率评估的全新基准。

**🔧 技术方法**

采用Playwright驱动的浏览器仿真、工具调用式的Agent框架、连续时间观测原语以及基于连续时间POMDP的形式化模型。

**📊 数据集**

使用公开的中文短视频数据集（约1,528条视频及其标题、标签、评论等元数据），并补充少量合成视频。

**📈 对比分析**

对比了多种前沿多模态LLM（Gemini、Seed、Qwen、Claude、GPT等），通过任务成功率、步骤数和观看比例评估，发现模型均落后人类（最高约65% SR，WR 25%），并暴露了过度/不足观测问题。

**⚠️ 局限性**

局限于模拟环境而非真实平台、仅中文语料、任务规模有限，且观察行为缺乏跨语言和跨平台验证。

---

## 358. A New Angle on Bones: Robust Pose Estimation in X-Ray and Ultrasound

**arXiv ID:** 2606.04700 | [PDF](https://arxiv.org/pdf/2606.04700v1)

**作者:** Ron Keuth `[一作]` (University of Lübeck), Lasse Hansen `[通讯]` (EchoScout GmbH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出一种基于深度学习的骨骼姿势角度估计方法，先利用U‑Net预测骨轴点候选，再通过PCA、RANSAC或Hough变换等稳健线拟合手段提取轴线，从而计算骨骼之间的角度。

**💡 创新点**

创新点在于：①采用点候选而非传统的两点关键点，大大降低对单点误差的敏感性；②系统对多种线拟合模型和误差抑制策略（MorphCC、ThreshCC）进行全面比较；③将这些技术结合后实现了在多模态（X光、超声）和多任务（儿科骨折、髋发育畸形）上的统一框架。

**🔧 技术方法**

核心技术包括U‑Net卷积网络用于点候选和热图回归；PCA、RANSAC、Hough变换三种稳健线拟合；形态学与连通组件分析进行误差抑制；数据增强与超参数优化。

**📊 数据集**

使用的数据集有：GRAZPEDWRI‑DX（≈20k儿科腕部X光，含手动标注骨折轴框），UKSH腕部超声（∼270帧，骨折与骨干标注），以及用于髋发育畸形评估的超声数据（133例）。

**📈 对比分析**

与传统热图回归的关键点方法对比，本文方法在所有三项任务上均取得显著优势。最优组合MorphCC+PCA的平均误差为4.10°（腕部X光）、5.41°（髋发育畸形）、6.00°（超声骨折），相较基线误差降低约5–10°，并保持在临床可接受的5–8°范围内。

**⚠️ 局限性**

局限性包括：仅在二维视图下工作，难以解决多视角的透视畸变；超声噪声和伪影仍可导致点候选分离成多簇；目前仅针对骨骼结构，软组织或复杂三维结构的适用性尚未验证；未来需扩展至三维体积或多模态融合以进一步提升鲁棒性。

---

## 359. Cone-Compatible Monge Geometry for High-Dimensional Ordered Optimal Transport

**arXiv ID:** 2606.04695 | [PDF](https://arxiv.org/pdf/2606.04695v1)

**作者:** Lei Luo `[一作]` (Nanjing University of Science and Technology), Jian Yang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 128987 | [OpenAlex ID](https://openalex.org/A5100604690)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在高维空间中，提出一种基于凸锥可兼容的Monge几何理论，利用锥诱导的偏序来判定哪些数据分布可以通过单调匹配实现原始成本下的最优传输，并给出闭式解法。

**💡 创新点**

创新点在于：
1) 通过“锥可兼容”条件将一维无交叉性质推广到高维；
2) 给出对平方马氏距离的必要且充分的兼容性判定（锥的M内积为锐角或K⊆K_M^*）；
3) 对于满足锥链条件的分布，推导出累积重叠（quantile‑type）闭式最优耦合；
4) 区分了“锥链 Wasserstein 量”与“有向锥 OT 成本”，明确它们的度量性质与应用场景；
5) 提供了软锥松弛、链化误差估计、统计收敛率、复杂度分析等完整的理论框架。

**🔧 技术方法**

主要技术包括：
- 凸锥和其对偶的几何理论；
- Monge 兼容性条件（无交叉交换不等式）和 Hessian 子模性；
- 区间重叠法（累计重叠耦合）和二指针线性算法；
- 对偶性、Kantorovich 变换、Strassen 定理；
- 统计学习理论与误差传播；
- 对角/半正定矩阵运算（Mahalanobis 成本）。

**📊 数据集**

本文为理论性工作，未使用具体公开数据集；研究结果适用于任何满足锥链可兼容性的高维概率分布。

**📈 对比分析**

与传统 OT、切片 Wasserstein（SW）和树 Wasserstein（TW）的比较：
- 在满足锥兼容的链类中，提供精确的闭式公式，计算复杂度为 O(n+m)；
- 相比 SW，避免了投影带来的信息丢失；相对 TW，保持原始度量并支持方向性匹配；
- 在可测度、统计误差和梯度可微性方面表现优异，但仅在数据具备可兼容锥链结构时才有效；
- 通过理论证明展示了在特定结构下的低维样本复杂度和收敛速率。

**⚠️ 局限性**

局限性：
- 仅对具有可兼容锥链结构的数据有效，随机或无序数据中可比较概率极低；
- 需要预先指定或学习合适的锥，选择不当会导致无效或误差大；
- 软锥松弛虽可处理噪声，但在极大惩罚下仍可能失去可微性；
- 对非二次或非凸成本的兼容性判定相对困难；
- 与 SW/TW 等通用方法相比，在无结构高维数据上缺乏通用性和可扩展性。

---

## 360. Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models

**arXiv ID:** 2606.04672 | [PDF](https://arxiv.org/pdf/2606.04672v1)

**作者:** Ayushman Raghuvanshi `[一作]` (Indian Institute of Science), Mahesh Chandran `[通讯]` (Fujitsu Research India)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于状态空间模型（SSM）的连续时间动态图（CTDG）学习框架 CTDG-SSM，能够在保持长时序信息的同时，融合多跳图结构信息，形成高效的记忆表示。

**💡 创新点**

创新点在于将 HiPPO 内存机制与图拉普拉斯多项式投影相结合，生成拓扑感知的多阶记忆更新公式，并通过零阶保持（ZOH）实现可离散化的高效实现，从而兼顾长时序（LRT）与长空间（LRS）依赖。

**🔧 技术方法**

核心技术包括 HiPPO 记忆压缩、图拉普拉斯多项式滤波、状态空间递推、零阶保持离散化、残差连接与自注意力结构，全部实现于轻量级可扩展模块。

**📊 数据集**

实验数据集涵盖动态链路预测（LastFM、Enron、MOOC、Reddit、Wikipedia、UCI、Social Evolution）、动态节点分类（Wikipedia、Reddit）以及序列分类（synthetic 长路径）等多任务。

**📈 对比分析**

与现有事件驱动和序列模型（JODIE、DyRep、TGAT、TGN、CAWN、TCL、GraphMixer、DyGFormer、CTAN、DyGMamba 等）比较，CTDG-SSM 在 LRT 任务中均达到或超过 AUC‑ROC 最佳值，参数量约为同行方法的十分之一，训练时间和显存占用也显著更低。

**⚠️ 局限性**

局限性包括对图拉普拉斯谱的依赖（需要保证滤波多项式在 [0,2] 内非零），对多阶多跳结构的显式支持虽然提升了性能，但在极度动态或节点极大变化的场景下可能需要更灵活的滤波机制；此外，实验主要集中在中等规模数据集，尚未在大规模工业级图上进行验证。

---

## 361. Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation

**arXiv ID:** 2606.04665 | [PDF](https://arxiv.org/pdf/2606.04665v1)

**作者:** Kaichao You `[一作]` (School of Software), Michael I. Jordan `[通讯]` (University of California, Berkeley)

**通讯引用:** 180489 | [OpenAlex ID](https://openalex.org/A5049812527)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Deep Embedded Validation (DEV) 方法，用于在深度无监督领域自适应（Deep UDA）中进行无标签目标域的准确模型选择。

**💡 创新点**

创新点在于将已自适应的深度特征嵌入验证过程，利用可估计的密度比实现无偏的目标风险估计，并在此基础上通过控制变量方法进一步降低方差，从而得到在理论上可保证方差界限且与目标风险高度相关的评估指标。

**🔧 技术方法**

使用的技术包括：重要性加权交叉验证、基于判别器的密度比估计、Renyi 散度理论、控制变量 (control variate) 技术，以及传统深度特征提取和任务特定分类器。

**📊 数据集**

在 VisDA、Office‑31、Digits（MNIST/USPS/SVHN）以及部分领域自适应 (PDA) 等数据集上进行了实验。

**📈 对比分析**

与传统方法（Source Risk、IWCV、Target Risk 上限）相比，DEV 在所有基准任务中都逼近 Target Risk 的性能，显著优于 IWCV 与 Source Risk，甚至在原论文报告的结果基础上提升 1–3%。

**⚠️ 局限性**

局限性包括：仍需假设协变量平移（covariate shift）且目标域无标签；对密度比估计的准确性敏感；在极端域差或非协变量平移场景下效果可能下降；以及额外的计算成本（需训练二层逻辑回归估计密度比）。

---

## 362. BiNSGPS: Geometry Problem Solving via Bidirectional Neuro-Symbolic Interaction

**arXiv ID:** 2606.04648 | [PDF](https://arxiv.org/pdf/2606.04648v1)

**作者:** Qi Wang `[一作]` (Institute of Automation of Chinese Academy of Sciences), Cheng-Lin Liu `[通讯]` (Institute of Automation of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 BiNSGPS 框架，通过 MLLM Adviser 与符号求解器建立双向闭环交互，动态纠正形式化错误并生成辅助假设，解决传统单向神经-符号系统的脆弱性。

**💡 创新点**

创新点在于实现双向神经-符号交互（BiNS），允许符号求解器反馈诊断信息给 MLLM，M 进而修正不一致的表述或提出新的辅助构造，从而大幅提升几何推理的鲁棒性与准确性。

**🔧 技术方法**

采用多模态大型语言模型（如 Qwen3‑VL‑Plus、Qwen3‑VL‑Plus/32B），PGDPNet 视觉解析网络获取高精度几何原始元素，符号求解器基于超图推理和预定义定理库，工具调用与闭环反馈机制实现系统集成。

**📊 数据集**

使用 Geometry3K（3001 题）、PGPS9K（9022 题）以及 MathVista 的几何推理子任务作为评测数据集。

**📈 对比分析**

在 Choice 与 Completion 模式下，BiNSGPS 分别取得 95.2%/90.5% 和 90.1% 的完成功率，明显优于 GPT‑5.2（78%）和其他最先进的神经、符号及神经‑符号模型，展示出显著的性能提升；在步骤逻辑一致性评测中也达到了 96% 的高一致率。

**⚠️ 局限性**

主要限制包括：辅助构造（auxiliary construction）生成被刻意限制，导致无法充分利用 MLLM 的创造性；MLLM 在某些情况下控制失效，导致输出过长；目前仅在现有数据集上验证，缺乏对更复杂几何构造及自动构造生成的完整探索。

---

## 363. Explainably Safe Reinforcement Learning

**arXiv ID:** 2606.04634 | [PDF](https://arxiv.org/pdf/2606.04634v1)

**作者:** Sabine Rieder `[一作]` (Masaryk University), Bettina Könighofer `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种层次化可解释安全强化学习框架，通过层级决策树为安全盾牌（shield）提供可解释性；

**💡 创新点**

首次将形式化安全保证与可解释性结合，利用分层决策树实现紧凑、基于案例的安全解释；

**🔧 技术方法**

强化学习、形式化方法（模型检测、CTL）、安全盾牌、决策树学习与执行树（execution tree）等技术；

**📊 数据集**

在Frozen Lake、Highway Cruise Control和Boeing Taxiing三个RL基准环境上进行实验；

**📈 对比分析**

与传统大规模安全盾牌对比，树结构的节点数缩小数个数量级，说明可解释性大幅提升，同时保持了原有的安全保障；

**⚠️ 局限性**

对用户输入的谓词依赖较大，缺乏自动谓词生成，且未进行用户可理解性评估，未来需进一步研究。

---

## 364. VentAgent: When LLMs Learn to Breathe -- Multi-Objective Arbitration for ARDS Ventilation

**arXiv ID:** 2606.04632 | [PDF](https://arxiv.org/pdf/2606.04632v1)

**作者:** Teqi Hao `[一作]` (Shanghai University of Engineering Science), Xihe Qiu `[通讯]` (Shanghai University of Engineering Science)

**通讯引用:** 580 | [OpenAlex ID](https://openalex.org/A5007950680)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种名为VentAgent的多阶段大语言模型驱动框架，用以实现ARDS患者的机械通气控制，框架分为感知、规划与编排三层，并通过可解释的推理链提供决策依据。

**💡 创新点**

创新点在于将通气任务从传统的单目标优化转化为多目标仲裁过程，利用LLM实现透明仲裁，设计三层分工架构（Perception‑Planning‑Orchestration）、多专家候选方案生成、稀疏记忆自我校正以及层级审计机制，保证决策既安全又可解释。

**🔧 技术方法**

技术上采用预训练LLM作为临床推理算子，构建Perception层的语义抽象，Planning层的场景Meta-Agent与领域专家生成多样方案，Orchestration层的多视角仲裁以及Sparse Reflective Memory与Hierarchical Audit模块共同实现动态决策与错误自纠，实验使用高保真Pulse Physiology Engine模拟器。

**📊 数据集**

使用数据集：通过Pulse Physiology Engine生成20种ARDS患者配置（采样队列）和100个未见患者（评估队列）进行实验；后续计划与MIMIC‑IV等真实EHR数据对比验证。

**📈 对比分析**

对比方法包括多种LLM推理范式（Few‑Shot、CoT、ReAct、Reflexion、Tree‑of‑Thoughts、Debate、Self‑Consistency）以及传统RL与PID基线；实验结果显示VentAgent在氧合、通气与机械安全三项指标上均显著优于基线，且安全违规率从30%+降至约8%。

**⚠️ 局限性**

局限性：目前仅在仿真环境验证，缺乏真实临床试验；LLM仍可能产生幻觉，需硬件安全阈值防护；计算开销和token延迟较高，对轻度ARDS患者表现相对不明显，需进一步优化以适应多样化临床场景。

---

## 365. RAMPART: Registry-based Agentic Memory with Priority-Aware Runtime Transformation

**arXiv ID:** 2606.04628 | [PDF](https://arxiv.org/pdf/2606.04628v1)

**作者:** Nikodem Tomczak `[一作]` `[通讯]` (Thulge Labs), Nikodem Tomczak (Thulge Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

RAMPART 通过在运行时对内存块进行可编程的编译式上下文组装，实现了在 RAM 中管理 LLM 代理的指令、工具模式和学习轨迹，并通过可组合的原语实现位置感知、可追溯性、透明删除与零令牌协同。

**💡 创新点**

其创新点在于：① 将上下文组装视为可编程的编译步骤，① 支持块级所有权和可撤销的结构化访问控制；② 提供原子原语（promote、gate、write、evict、rollback）实现零令牌成本的动态位置调整和透明忘却；③ 通过共享注册表实现多代理零令牌协同；④ 在无磁盘 I/O 的情况下实现可追溯的、可扩展的上下文管理。

**🔧 技术方法**

采用基于 Python 的 OrderedDict 注册表、轻量句子嵌入（如 all‑MiniLM-L6-v2）进行语义相关性过滤、按优先级与访问计数动态排序、在编译前进行位置与内容关系的可编程调整；通过 copy‑on‑write fork 在多进程间共享内存；在编译时完成完整上下文拼接，随后无模型推理即可得到最终 Prompt。

**📊 数据集**

实验使用 31 个块的嵌入式固件 seed 库，生成 20 题任务组；在 Qwen3‑8B、Qwen2.5‑7B、Llama‑3.1‑8B、Mistral‑7B、Qwen3‑14B 这些 instruction‑tuned 模型上进行位置敏感性扫掠、块聚类、相关性门控和工具 schema 失效实验；还对多代理协调使用虚拟工作坊场景进行 token 开销对比。

**📈 对比分析**

与传统 SKILL.md 直接拼接、Letta 等数据库检索等做法对比，RAMPART 在位置敏感性实验中通过块聚类提升至 5 倍、在 Mistral‑7B 上从 4% 提升至 20%；在 67.8% 的 token 减少下，保持 83% 的成功率；工具 schema 失效实验表明完整消除调用；共享注册表实验显示协调 token 从约 350k 降至 0，验证了零令牌协同。

**⚠️ 局限性**

局限性包括：① 无并发多代理写入与冲突解决；② 依赖固定上下文长度，超长上下文尚未验证；③ 对外部检索或数据库的集成需手工实现；④ 原语仅适用于推理阶段，缺乏动态学习或参数更新；⑤ 对复杂任务的泛化性能尚未在更大、多样化基准上评估。

---

## 366. Rectangular Matrix Multiplication in the Low-Bandwidth Model

**arXiv ID:** 2606.04652 | [PDF](https://arxiv.org/pdf/2606.04652v1)

**作者:** Chetan Gupta `[一作]` (IIT Roorkee), Hossein Vahidi `[通讯]` (Aalto University)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5080446364)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究在低带宽分布式计算模型下，稠密矩阵的矩形乘法的通信复杂度。

**💡 创新点**

首次揭示了 d ≤ √n 时的 Θ(d√n) 阶段转折并给出了最优上界和下界；同时证明了 d ≥ √n 区域的上界为 O(d²/3 n²/3) 并给出了相应的条件下界。

**🔧 技术方法**

采用分块划分、广播与聚合技术，结合信息理论与图着色的组合证明，利用已有的稠密矩阵乘法算法（如 Strassen、Fast Field Multiplication）构造归约。

**📊 数据集**

无具体实验数据集；全部结果均为理论复杂度分析。

**📈 对比分析**

与传统拥塞团（congested clique）模型相比，低带宽模型下的上界与下界匹配（在 d ≤ √n 区域），并展示了 d ≤ √n 时相较于之前结果可实现更低的通信轮数；但在 d ≥ √n 区域仍存在上下界差距。

**⚠️ 局限性**

未解决 d ≥ √n 区域的精确复杂度；结果仅适用于稠密矩阵，对稀疏矩阵的适用性尚不明；缺乏实验验证。

---

## 367. VISTA: Vision-Grounded and Physics-Validated Adaptation of UMI data for VLA Training

**arXiv ID:** 2606.04708 | [PDF](https://arxiv.org/pdf/2606.04708v1)

**作者:** Siyuan Yang `[一作]` (Institute of AI (TeleAI), China Telecom), Xuelong Li `[通讯]` (Institute of AI (TeleAI), China Telecom)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过对手持式腕式鱼眼相机收集的数据进行视觉对齐和物理可行性验证，训练出可在多机器人平台上部署的通用视觉语言动作模型。

**💡 创新点**

创新点在于①构建首个针对腕式鱼眼视角的大规模VQA数据集UMI‑VQA；②提出跨体态物理验证流水线；③采用两阶段联合训练，兼顾视觉语言对齐与动作生成。

**🔧 技术方法**

使用的技术包括Vision‑Language‑Action两阶段协同训练、流匹配动作专家、轨迹级物理验证算法、Diffusion Policy与ACT等模型框架。

**📊 数据集**

使用的数据集为：8M样本的UMI‑VQA、约10万条通过物理验证的UMI轨迹，以及公开的RoboTwin‑UMI、LIBERO‑UMI仿真基准和20个真实机器人任务。

**📈 对比分析**

在相同数据量下与π_0.5、LingBot‑VLA、Wall‑X等基线对比，VISTA在仿真任务上的成功率提升约10%点，在真实机器人任务上的平均成功率提升约7%点，显著优于基线。

**⚠️ 局限性**

局限在于：仍需针对目标机器人手动调节物理验证阈值；对极端鱼眼失真和复杂场景的鲁棒性有限；未在大规模多模态环境中进一步验证。

---

## 368. Extraction and Search in Rocq: Theorems, Definitions and Their dependencies

**arXiv ID:** 2606.04704 | [PDF](https://arxiv.org/pdf/2606.04704v1)

**作者:** Jian Fang `[一作]` (Peking University), Yingfei Xiong `[通讯]` (Peking University)

**通讯引用:** 5843 | [OpenAlex ID](https://openalex.org/A5100712724)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了名为TheoremExtr的工具，能够在Coq编译阶段和运行时提取定理、定义及其依赖，并通过Web界面实现跨项目的相似搜索。

**💡 创新点**

创新点在于将解析阶段的语法信息与运行时的依赖信息相结合，提供完整的定理声明、类型和内部依赖；同时实现了基于BM25的跨项目相似搜索网站。

**🔧 技术方法**

技术手段包括：修改Coq编译器抽象语法树以提取语法层数据；实现Coq插件在运行时抓取定理及其类型/依赖；Python脚本合并两阶段数据；Flask框架+BM25算法搭建搜索服务。

**📊 数据集**

使用了来自Coq平台（2025.01.0）及其社区的多家开源项目，共提取数千条定理和定义。

**📈 对比分析**

与CoqPyt、Coq SerAPI等现有工具对比，TheoremExtr在单个命令下完成提取，且能够提供完整的依赖信息；实验显示每个项目平均耗时约几十到几百秒，整体约6800秒，足以在大项目中使用。

**⚠️ 局限性**

局限性包括：目前仅兼容Coq 8.20.0，跨版本迁移需要改动；极大项目（如CompCert）运行时提取时间较长；需要用户先导入所有相关库；对隐式参数的处理尚未覆盖全部情况。

---

## 369. Randomized separations in black-box TFNP

**arXiv ID:** 2606.04697 | [PDF](https://arxiv.org/pdf/2606.04697v1)

**作者:** Fedor Kiselev `[一作]` `[通讯]` (Moscow Institute of Physics and Technology), Fedor Kiselev (Moscow Institute of Physics and Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

提出了一种通用技术，将确定性黑盒可约性与随机化黑盒可约性等价化，从而把已知的黑盒分离提升为随机化分离。

**💡 创新点**

创新点在于引入“诚实”(honest)和“感知”(perceptive)伪约化的概念，并利用部分赋值的结构与组合图论证明随机化不可约性的充分条件，将这一方法推广到PPP、PPAD、PPADS、t-PPP、PPA等TFNP子类。

**🔧 技术方法**

核心技术包括：对伪约化的诚实化和感知化处理、局部赋值的层级分析、Yao极小化原理、以及组合图论中邻接关系与计数不等式的应用。

**📊 数据集**

论文为理论性工作，没有使用任何实验数据集。

**📈 对比分析**

与仅证明确定性黑盒不可约性的工作相比，本研究在同一类问题上进一步证明随机化不可约性。理论上，随机化分离得到了加强，但论文未给出具体算法性能数值，而是通过组合估计验证了PPP、PPAD、PPADS、t-PPP、PPA等子类的随机化不可约性。

**⚠️ 局限性**

方法依赖于问题的高度对称性和可计数的局部赋值结构，无法处理某些TFNP子类（如PPADS在特定构造下不满足必要的组合估计），且需要复杂的组合计数证明。因此，无法直接推广到所有TFNP子类，随机化可约性在更广泛的类上仍是开放问题。

---

## 370. Real-Time Automatic License Plate Recognition Using YOLOv8, SORT Tracking, and Temporal Data Interpolation

**arXiv ID:** 2606.04684 | [PDF](https://arxiv.org/pdf/2606.04684v1)

**作者:** Mirza Muhammad Mobeen `[一作]` `[通讯]` (Sanwa Comtec K.K.), Mirza Muhammad Mobeen (Sanwa Comtec K.K.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一个五阶段的实时车牌识别流水线，结合YOLOv8n检测车辆、SORT跟踪、YOLOv8局部检测车牌、EasyOCR识别文字，并通过线性时序插值填补轨迹空洞。

**💡 创新点**

创新点在于：①将YOLOv8与SORT与EasyOCR串联成完整端到端系统；②引入线性时序插值显著提升轨迹完整度；③针对英国车牌构建位置字符纠错映射，减少误识率。

**🔧 技术方法**

使用的技术包括：YOLOv8（nano）目标检测、SORT多目标跟踪、EasyOCR（CRNN+CTC）文字识别、SciPy线性插值、OpenCV预处理。

**📊 数据集**

在一段3,599帧的交通监控视频（未公开数据集）上进行实验，记录45辆车的轨迹与车牌识别结果。

**📈 对比分析**

通过与基线（仅检测+跟踪）对比，插值后轨迹条目提升101.9%，但OCR平均置信度仅为0.414，最高值0.982，表明在动态环境下仍存在较大识别不确定性。

**⚠️ 局限性**

局限性包括：仅支持英国车牌语法、CPU端EasyOCR导致推理速度慢、缺乏人工标注的评估基准、线性插值无法处理非线性运动、整体OCR准确率低于执法需求。

---

## 371. LifeSide: Benchmarking Agents as Lifelong Digital Companions

**arXiv ID:** 2606.04660 | [PDF](https://arxiv.org/pdf/2606.04660v1)

**作者:** Yuqian Wu `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一套评估长期数字伴侣能力的基准，重点考察跨会话记忆、用户理解、隐私控制与情感陪伴四个维度；

**💡 创新点**

创新点在于：①构建“记忆‑情感‑环境（MEE）”循环的评估框架；②使用多代理模拟将隐层用户世界投射为可观测对话，保留内心与表达之间的差距；③设计分层级评测流程，涵盖从事件回忆到情感支持的全过程；

**🔧 技术方法**

采用多代理框架（Manager/ User/ Response/ Critic）实现模拟；使用POMDP理论建模交互；通过检索增强（RAG）和外部记忆（Letta、Mem0等）等技术；

**📊 数据集**

自建的多会话数据集：2000个基于人口普查约束的虚拟角色，包含24‑36个月的事件轨迹、环境动态，共计111k个评测任务；

**📈 对比分析**

对比三类基线：前沿大模型、检索增强模型、外部记忆模型。实验结果显示，即使在现有记忆基准上表现已逼近极限，仍难以在本基准中达到高水平（最高约41%记忆回忆，情感陪伴低于37%），说明记忆与情感支持在长期关系中存在显著瓶颈；

**⚠️ 局限性**

局限性包括：①数据为合成，缺乏真实情绪波动与语言细节；②评测使用LLM裁判，受预训练偏差限制，缺乏专业心理咨询的细腻度；③对文化多样性的考虑不足，可能导致刻板印象。

---

## 372. Instance-Level Post Hoc Uncertainty Quantification in Object Detection

**arXiv ID:** 2606.04656 | [PDF](https://arxiv.org/pdf/2606.04656v1)

**作者:** Chongzhe Zhang `[一作]` (RAMS Lab, Huawei Heisenberg Research Center), Zheng Hu `[通讯]` (RAMS Lab, Huawei Heisenberg Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种MC-GLM方法，利用Monte Carlo采样与Laplace近似，在不改变模型的前提下，快速计算每个检测框的实例级不确定性。

**💡 创新点**

将GLM的雅可比矩阵近似为低秩方向导数，减少对多次反向传播的需求，实现真正的后置（post‑hoc）不确定性估计，并将采样数与实例数无关。

**🔧 技术方法**

采用Laplace近似、KFAC Fisher信息估计、有限差分方向导数与Monte Carlo采样等技术。

**📊 数据集**

在nuScenes数据集上使用CenterPoint点云3D检测器进行评估。

**📈 对比分析**

与MC Dropout、全局GLM等基线对比，MC-GLM在实例不确定性与误差的相关性（P(U|I)、P(A|C)）上表现更佳，速度上仅需k+1次前向传播，显著快于传统GLM。

**⚠️ 局限性**

局限性包括：需预先估计KFAC Fisher信息，内存消耗较大；对不同网络结构或更深层的Fisher信息敏感；在超参数（ϵ、k）选择上仍需经验。

---

## 373. EviRank: Evidence-Based Confidence Estimation for LLM-Based Ranking

**arXiv ID:** 2606.04727 | [PDF](https://arxiv.org/pdf/2606.04727v1)

**作者:** Meng Yan `[一作]` (Xidian University), Wei Zhao `[通讯]` (Xidian University)

**通讯引用:** 92834 | [OpenAlex ID](https://openalex.org/A5050699488)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于LLM的排序推荐系统的置信度估计框架EviRank，利用单次前向推理提取语义、注意力和输出三类证据，通过可靠意见聚合得到位置级置信度，并通过位置感知校准与置信度引导的重排序来提升推荐质量与不确定性量化；

**💡 创新点**

创新点包括：①从单次LLM推理中同步提取多源证据并融合；②提出可靠意见聚合机制，自动调节不同证据源的可靠性；③引入位置感知校准，使置信度与排序位置重要性匹配；④以置信度加权的重排序策略实现推荐性能提升；

**🔧 技术方法**

核心技术包括：LLM前向推理（提取隐藏状态、注意力分布、logit概率）、主观逻辑与Dirichlet分布构建信念质量、可靠意见聚合、位置感知校准（带NDCG权重的Sigmoid映射）、基于贝叶斯个性化排序（BPR）的重排序目标；

**📊 数据集**

实验数据集为MovieLens 1M、Amazon Grocery、Steam三大公开序列推荐数据集；

**📈 对比分析**

与传统协同过滤、Transformer序列推荐模型以及最新LLM重排序模型（PepRec、RankGPT、LLM4Rerank）对比，EviRank在R@5、N@5、R@20、N@20等指标上均实现显著提升，且在不确定性量化任务中取得最优的 Kendall τ 与 Concordance Index；

**⚠️ 局限性**

局限性：①仅在已知候选集的重排序场景下验证，未评估极大候选集或实时推荐的可扩展性；②对冷启动用户/新商品的置信度仍有限；③依赖于LLM的预训练知识，若领域知识缺失仍可能出现低置信度；

---

## 374. Selection-Aware Diagnostics for Chain-of-Thought Answer Hijacking

**arXiv ID:** 2606.04717 | [PDF](https://arxiv.org/pdf/2606.04717v1)

**作者:** Jianwei Tai `[一作]` (Anhui University), Jianwei Tai `[通讯]` (Anhui University)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5110952853)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在GSM8K和MATH‑500数据集上，对链式推理（CoT）答题劫持的K步激活干预恢复机制，并给出了可复现的诊断协议。

**💡 创新点**

提出了选择感知的阈值带局部化和源依赖诊断，揭示了“fragility band”现象——不需要同题干清洁激活即可恢复答案，并对不同劫持类型给出细粒度机制分类。

**🔧 技术方法**

通过在Transformer层输出挂钩处插入K步激活补丁，配合Bonferroni校正的层间置换检验、带宽验证、随机/零/交叉层源控制，以及固定钩子大样本复现和迁移实验。

**📊 数据集**

主要使用开放式算术题库GSM8K、数学推理数据集MATH‑500，并在两大模型Qwen2.5‑7B与Llama3‑8B上进行实验。

**📈 对比分析**

与RESTAsmoothing、同模型重写、低概率过滤等基线相比，K步补丁在几种劫持场景下恢复率可达约40‑60%；在固定钩子下大样本复现显示47%（Qwen）/39%（Llama）；迁移到MATH‑500仅26%。

**⚠️ 局限性**

局限性包括仅覆盖两模型、单一算术基准；迁移与适配性有限；未测试大规模攻击或自适应梯度后缀攻击；生产环境中的延迟、批处理效果未验证。

---

## 375. SMADE-IE: Sparse Multi-Agent Framework with Evidence-Driven Debate for Zero-Shot Information Extraction

**arXiv ID:** 2606.04691 | [PDF](https://arxiv.org/pdf/2606.04691v1)

**作者:** Kenfeng Huang `[一作]` (South China University of Technology), Li Yuan `[通讯]` (South China University of Technology)

**通讯引用:** 72229 | [OpenAlex ID](https://openalex.org/A5100351175)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SMADE-IE，一种稀疏且基于证据的多代理框架，用于零样本信息抽取（NER、RE 和 JERE），能够动态路由输入并通过结构化辩论实现冲突解决。

**💡 创新点**

核心创新包括：①自适应模式选择器能将样本分配到轻量级全局抽取或细粒度类型中心抽取；②证据驱动辩论将争议条目转化为托尔文式论证，并用贝叶斯更新聚合信心；③迭代实体-关系对齐机制确保联合抽取的一致性。

**🔧 技术方法**

技术手段包括大型语言模型（GPT‑3.5‑Turbo‑0125）、外部证据评分器（Gemini‑3‑Flash‑Preview）、Toulmin 论证结构、Beta 分布贝叶斯更新、双轨早停策略以及多代理协同。

**📊 数据集**

在 9 个基准数据集上评估，覆盖 NER（CoNLL03、OntoNotes5 等）、RE（DocRED、CrossRE、REDFM、SemEval2010 等）和 JERE（CoNLL04、NYT）。

**📈 对比分析**

与现有零样本 IE 基线（AEiO、One‑Step、G&O、CrossAgentIE）相比，SMADE-IE 在 F1 方面平均提升 14‑30 分（NER）、3‑10 分（RE）、14 分（JERE），且平均 token 成本显著下降，尤其在关系密集数据集上节省 5‑7 倍。

**⚠️ 局限性**

主要限制：①依赖固定的外部证据评分器，受 NLI 校准上限限制；②在类型稠密或长文本场景下，类型中心抽取仍需多次 LLM 调用，成本下降有限；③实验仅在 GPT‑3.5‑Turbo 上进行，未覆盖开源模型、多语言或超长文本的泛化。

---

## 376. Clownfish: Scaling DAG-based BFT Consensus via Sparse Edges

**arXiv ID:** 2606.04687 | [PDF](https://arxiv.org/pdf/2606.04687v1)

**作者:** Feifan Wang `[一作]` (Tsinghua University), Zhixuan Fang `[通讯]` (Tsinghua University)

**通讯引用:** 831 | [OpenAlex ID](https://openalex.org/A5010064740)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种部分同步的DAG‑BFT协议Clownfish，采用领导边（leader edge）和快速投票（fast‑vote）来显著减少元数据通信并提升吞吐量。

**💡 创新点**

创新点在于：①只让领导节点携带完整的n‑f引用，其余节点仅引用一个领导，显著降低每轮元数据量；②引入fast‑vote与优化的轮次推进规则减少失效时的额外延迟；③支持多领导者（multi‑leader）实现平均延迟下降；④给出了CBC版本，填补先前协议在弱广播原语下的理论空白。

**🔧 技术方法**

使用技术包括：部分同步网络模型、可靠广播（RBC）与一致广播（CBC）原语、阈值/聚合签名、领导边设计、fast‑vote消息、平衡多播（Balanced Multicast）以及可选的自举层实现。

**📊 数据集**

实验数据集为：在模拟器中使用空块、最多1000节点、1Gbps带宽、5个地理分布区；在部署实验中使用AWS 5区、每节点8 vCPU、16GB内存、100Mbps带宽、生成256字节的虚拟交易。

**📈 对比分析**

通过与Sailfish和Sparse Bullshark在延迟、吞吐量和每轮元数据量等指标的对比，Clownfish在大规模节点（>300）下保持几乎不变的延迟，吞吐量几乎线性增长，元数据量从O(n³)降到O(n²)，并在失败场景下比Sailfish减少约10%延迟。

**⚠️ 局限性**

局限性：领导节点仍需承担较重负载，需进一步的负载平衡或领导信誉机制；实验规模限于1000节点，未验证更大规模下的表现；仅评估空块/虚拟交易，未覆盖真实交易负载；仍未实现无签名版本。

---

## 377. QO-Bench: Diagnosing Query-Operator-Preserving Retrieval over Typed Event Tuples

**arXiv ID:** 2606.04646 | [PDF](https://arxiv.org/pdf/2606.04646v1)

**作者:** Mengao Zhang `[一作]` (National University of Singapore), Ke-wei Huang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个基于事件元组的查询‑操作问答（QO‑QA）基准，并提出了两个维度的诊断框架。

**💡 创新点**

首次使用确定性的事件元组黄金标准和按操作符细粒度诊断，同时强调检索的操作符保持性。

**🔧 技术方法**

对比了RAG、ReAct RAG、GraphRAG和信息抽取→SQL等四种检索‑生成范式，并引入长上下文oracle作为上限。

**📊 数据集**

基于FNSPID财经新闻22,984篇和614个由S&P Capital IQ标注的企业事件，生成了785个模板化问答。

**📈 对比分析**

采用回召率（±7日容忍）进行评价，Oracle上限52.2%，IE→SQL最高37.9%，RAG 25.2%，ReAct RAG 23.9%，GraphRAG极低，显示不同范式在不同操作符上表现差异。

**⚠️ 局限性**

受限于企业事件数据的公开性、模板化问句的语言多样性不足，以及对事件定义和实体消歧的依赖，可能不易推广至其他领域。

---

## 378. Bridge the Last-Mile Gap to Semantic Analytics: Compiling Natural-Language Queries into Semantic Operator Pipelines

**arXiv ID:** 2606.04641 | [PDF](https://arxiv.org/pdf/2606.04641v1)

**作者:** Wenkai Dong `[一作]` (University of Hawaii at Manoa), Yifan Wang `[通讯]` (University of Hawaii at Manoa)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种三阶段中间件，将自然语言问题自动编译为可在多种语义算子系统上执行的语义算子流水线，完成跨模态、多跳推理的查询。

**💡 创新点**

创新点在于将问题解析、桥接实体发现、算子规划与后端代码生成拆分为独立阶段，并通过自动生成的参考文档桥接不同后端的 API，从而实现高质量、低成本、跨后端可移植的流水线编译。

**🔧 技术方法**

技术方案主要使用大语言模型（LLM）进行实体链接与桥接、规划生成和代码生成；结合语义算子系统（Palimpzest、LOTUS、Nirvana）实现过滤、连接、映射、聚合等算子；并通过离线数据分析和参考文档生成实现后端特定代码生成。

**📊 数据集**

评估数据集包括 MMQA、HotpotQA、HybridQA、ManyModalQA 与 TAT-QA，共覆盖表格、文本、图像等多模态及多跳推理场景。

**📈 对比分析**

与 Naive、Hardcoded、CodeTree、Codex (model/agent) 等基线对比，平均 EM/F1/LLM 分别提升 50%+，单问成本约 $0.013，映射延迟 50–100 秒，质量-成本曲线显著优于其他方法，尤其在跨源、多模态任务中表现突出。

**⚠️ 局限性**

局限性包括对 LLM 的准确性依赖，桥接实体发现对极为复杂数据可能不足，执行成本相较传统 SQL 更高，仅支持已有算子，且对极端多模态或高并发场景仍有进一步优化空间。

---

## 379. MIRAGE: Mobile Agents with Implicit Reasoning and Generative World Models

**arXiv ID:** 2606.04627 | [PDF](https://arxiv.org/pdf/2606.04627v1)

**作者:** Zhichao Yang `[一作]` (Beihang University), Yan Bai `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种移动代理框架MIRG，通过将链式推理迁移至连续潜在空间实现隐式思考，显著降低生成思考文本的令牌消耗，提升交互效率。

**💡 创新点**

创新点在于：1）将显式链式推理转化为潜在空间的隐式推理；2）引入Approximate Parallel Latent Refinement (APLR)实现并行潜在迭代，逼近串行思考；3）使用Q‑Former世界模型头，将潜在状态与下一帧视觉特征对齐，强化未来状态预测。

**🔧 技术方法**

采用Qwen3‑VL视觉‑语言模型为骨干，结合潜在槽插入、APLR并行精炼、Q‑Former交叉注意以及下一帧特征对齐损失进行联合训练。

**📊 数据集**

实验数据集为AndroidControl（低级/高级指令对齐）和AndroidWorld（116个真实应用任务），并与多种基线模型对比。

**📈 对比分析**

与匹配规模的Qwen3‑VL‑Instruct及其他最新GUI代理相比，MIRG在AndroidControl低级拆分中动作精度提升约15%，平均生成令牌从115降至约19；在AndroidWorld中成功率提升10–11%，平均令牌从103/108降至31/27，展现了更高效且性能可比的结果。

**⚠️ 局限性**

局限性包括仅使用监督训练、仅对下一帧视觉特征进行对齐、缺乏多步强化学习以及在真实部署前仍需关注隐私与动作安全。

---

## 380. Improved Approximation Guarantees for Groupwise Maximin Share Fairness

**arXiv ID:** 2606.04731 | [PDF](https://arxiv.org/pdf/2606.04731v1)

**作者:** Georgios Amanatidis `[一作]` (Athens University of Economics and Business), Christodoulos Santorinaios `[通讯]` (Athens University of Economics and Business)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5059334937)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在可加价值函数下，如何公平地分配不可分割物品，提出了群组最小分区（groupwise maximin share，GMMS）公平性约束，并给出了可在多项式时间内实现的近似算法。

**💡 创新点**

创新点包括：①证明了 GMMS 的 (φ−1)≈0.618 近似保证，打破了先前 4/7 的极限；②设计了简化的 Match‑Draft‑and‑Eliminate 算法，并对 n=3 的情形改进到 0.72 近似；③在 top‑n 和小规模 n 的特殊情形下进一步提升到 2/3；④通过对 NSW‑最大化匹配、拥塞图（envy graph）和多重简化步骤的精细分析，得到最优的  (φ−1) 证明。

**🔧 技术方法**

主要技术手段包括：NSW（Nash 社会福利）最大化匹配求解、改进的 Draft‑and‑Eliminate 方案、环消除（Envy‑Cycle‑Elimination）算法、对最大最小份额（maximin share）的归约与单调性分析，以及对分配实例的“问题物品”删减与规模归约。

**📊 数据集**

本文为理论研究，没有使用实际数据集；所有结果均基于理论分析和构造的反例。

**📈 对比分析**

与此前 4/7 近似的 GMMS 算法相比，本文算法在一般情形下提供 0.618 的近似，比 0.571 更好；在 n=3 时实现 0.72；在 top‑n 或小 n 时实现 2/3，匹配已知的最佳结果。算法复杂度为多项式级，主要耗时在 NSW 匹配和环消除上。

**⚠️ 局限性**

局限性：仍未达到 1 的完美公平；GMMS 近似比仍相对保守；证明中对多重简化和归约的过程较为复杂，难以直接推广到非可加或更一般的价值模型；以及对 GMMS 约束的验证在实践中可能仍具有挑战。

---

## 381. Real-World Deployment of a 5G-Connected Edge-Controlled Aerial Robot in Industrial Subterranean Mines

**arXiv ID:** 2606.04818 | [PDF](https://arxiv.org/pdf/2606.04818v1)

**作者:** Achilleas Santi Seisa `[一作]` (Luleå University of Technology), George Nikolakopoulos `[通讯]` (Luleå University of Technology)

**通讯引用:** 6626 | [OpenAlex ID](https://openalex.org/A5064878830)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在北瑞典工业矿山中，首次实现并测试了一架5G NR SA网络连接、基于Kubernetes边缘集群离线执行的模型预测控制器（NMPC）的无人机自主飞行。

**💡 创新点**

创新点在于：①首次在不受实验室控制的真实矿山环境中部署边缘控制无人机；②将NMPC完全离线到边缘，充分利用5G低时延；③在工业现场验证了多项关键网络指标（RTT、延迟、抖动）满足实时控制的稳定性阈值。

**🔧 技术方法**

使用技术包括：5G NR SA网络（3.5 GHz N78频段）与定制的射频基站；基于Kubernetes的边缘集群（Rancher、k8s 1.28.15）与容器化的NMPC；轻量级无人机平台配备Sierra Wireless 5G模块、LiDAR传感器；UDP协议实现双向遥测与控制指令。

**📊 数据集**

实验使用了无人机实时获取的LiDAR点云和遥测数据作为控制输入，并记录了5G网络的实时测量（RSRP、SINR、TX Power、RSSI等）来评估链路质量；未使用公开数据集。

**📈 对比分析**

通过测量上行/下行延迟、漂移校正后的延迟以及抖动，并与文献中给出的最大允许RTT阈值进行对比，验证系统稳定；实验结果显示，平均上行延迟9.5 ms，下行6.5 ms，抖动≤11 ms，实际飞行轨迹与人工设定的航路点高度吻合，表明边缘控制方案在工业环境下实现了可接受的实时性能。

**⚠️ 局限性**

局限性包括：仅在单一矿山现场验证，缺乏跨环境泛化；无人机负载受限，无法携带更多传感器；当前只进行低带宽遥测传输，未探究更高数据率的实时感知；未来需要进一步研究网络质量自适应规划、云端对比以及更强大的边缘算力支持。

---

## 382. Beyond Objective Equivalence: Constraint Injection for LLM-Based Optimization Modeling on Vehicle Routing Problems

**arXiv ID:** 2606.04816 | [PDF](https://arxiv.org/pdf/2606.04816v1)

**作者:** Xizi Luo `[一作]` (Beihang University), Yu Mei `[通讯]` (Baidu Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实现了一种约束注入验证器，用于在自然语言车辆路径问题（VRP）的模型代码生成中，检测并消除过度约束和缺失约束，构建VRPCoder 8B LLM并通过GRPO强化学习提升性能。

**💡 创新点**

将可行探针和单约束违规探针结合的约束注入技术，与差分测试构成双重验证器，并将其用于数据过滤和强化学习奖励，解决了目标等价性下的约束误差问题。

**🔧 技术方法**

基于大语言模型的端到端代码生成，监督微调（SFT），基于双重验证器的群组相对策略优化（GRPO），约束注入与差分测试验证，以及Gurobi求解器。

**📊 数据集**

21种VRP变体的专家验证基准（共700个实例），包括CVRP、时间窗、接送、异构车队等；训练使用18种变体，测试使用4个基准，其中包含3个保留变体。

**📈 对比分析**

与闭源前沿LLM（Gemini-3.1-Pro Preview、Claude-Sonnet-4.5）、开源通用LLM以及之前的OR-LLM进行对比；VRPCoder-GRPO在四个基准上平均Pass@1达到93%，超过闭源Gemini约8.4分，远优于其他基准。

**⚠️ 局限性**

仅验证车辆路径问题，需手工设计探针攻击器，评估仍基于目标等价性Pass@1，缺乏细粒度约束违规评估。

---

## 383. Learning While Acting: A Skill-Enhanced Test-Time Co-Evolution Framework for Online Lifelong Learning Agents

**arXiv ID:** 2606.04815 | [PDF](https://arxiv.org/pdf/2606.04815v1)

**作者:** Bo Mao `[一作]` (East China Normal University), Liang He `[通讯]` (East China Normal University)

**通讯引用:** 32318 | [OpenAlex ID](https://openalex.org/A5102798483)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于强化学习的两阶段框架——Skill-enhanced Test-Time Co-Evolution（LifeSkill），使大型语言模型在部署时能够持续学习；

**💡 创新点**

创新点在于利用执行验证器回报进行技能学习（Verifier‑Guided Skill Learning）以及在测试时将成功的技能轨迹转化为无外部提示的参数更新（Online Skill Internalization），实现真正的在线终身学习；

**🔧 技术方法**

核心技术包括DPO（分词级别的CLIP目标）用于技能提取器训练、强化学习策略优化、Verifier-Guided奖励信号以及在线参数微调；

**📊 数据集**

在LifelongAgentBench数据集上评估，涵盖数据库（DB）、操作系统（OS）和知识图谱（KG）三种交互式环境；

**📈 对比分析**

与基于记忆检索的训练免费方法和基于训练的RL方法对比，LifeSkill在DB和OS环境上平均准确率提升至0.59，较最强训练免费基线提升10个绝对分，较最强训练基线提升7个绝对分；

**⚠️ 局限性**

局限性包括在稀疏奖励、长时序任务（如KG环境）上表现不如最佳基线，以及对测试时计算资源的需求较高（需多次技能和轨迹采样）。

---

## 384. Scenario Generation for Risk-Aware Reinforcement Learning with Probably Approximately Safe Guarantees

**arXiv ID:** 2606.04812 | [PDF](https://arxiv.org/pdf/2606.04812v1)

**作者:** Mohit Prashant `[一作]` (Nanyang Technological University), Arvind Easwaran `[通讯]` (Nanyang Technological University)

**通讯引用:** 2016 | [OpenAlex ID](https://openalex.org/A5054946593)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在无模型强化学习环境下，设计了一种基于双概率约束程序与变分自编码器的安全约束验证与强化学习框架，构造上下界安全门控函数，并利用生成模型生成位于两界之间的状态来提升策略鲁棒性。

**💡 创新点**

① 在无模型环境下同时求解上、下界安全门控函数，给出PAC安全保证；② 将门控函数映射到VAE潜在空间，利用生成式模型主动探索“临界”状态；③ 通过二阶段学习显著缩小误差界宽度。

**🔧 技术方法**

使用变分自编码器（VAE）、概率约束程序（Chance‑Constrained Program）、场景优化/凸约束求解、PPO强化学习以及基于潜在空间的状态生成与重构技术。

**📊 数据集**

OpenAI Gym中的Ant和CartPole两个经典控制环境，采用像素观测作为输入。

**📈 对比分析**

与Vanilla PPO、ε‑greedy PPO、加噪探索、CoDE、遗传课程学习、对抗课程训练等方法在安全误差界宽度和平均奖励上进行对比；实验表明本方法在两环境中误差界宽度优于大多数方法，平均奖励相近。

**⚠️ 局限性**

依赖VAE编码质量与潜在空间相邻性；对高维复杂环境的扩展仍受限；需要大量轨迹样本来估计概率约束；缺乏对动态转移的显式建模，生成的状态仍需人工或专家验证。

---

## 385. PersonaTree: Structured Lifecycle Memory for Person Understanding in LLM Agents

**arXiv ID:** 2606.04780 | [PDF](https://arxiv.org/pdf/2606.04780v1)

**作者:** Yubo Hou `[一作]` (Beihang University), Zengchang Qin `[通讯]` (Beihang University)

**通讯引用:** 3009 | [OpenAlex ID](https://openalex.org/A5032405950)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个结构化生命周期记忆框架，通过三层人设树将交互证据、可重复模式和稳定用户主张关联起来，实现持久的人格理解与记忆；

**💡 创新点**

创新点在于构建层次化的支持边结构，使抽象层级的用户理解始终可追溯至具体证据，并通过保守写入、置信度驱动的整合与查询条件路径检索提升抽象与细节的平衡；

**🔧 技术方法**

采用三层树结构（叶层事件、树层模式、根层主张）、支持边、置信度更新、在线插入与离线整合、路径检索等技术；

**📊 数据集**

使用六个基准数据集：KnowMe、RealPref、CUPID、LongMemEval、RealMem、LoCoMo-Plus；

**📈 对比分析**

与Mem0、A-MEM、TiMem等系统在18个compact分数中排名第一12项、前两名16项，整体显著提升；

**⚠️ 局限性**

仅针对文本交互的英文基准，未验证多语言、语音或多模态场景。

---

## 386. Tree-Based Formalization of Multi-Agent Complementarity in Human-AI Interactions

**arXiv ID:** 2606.04779 | [PDF](https://arxiv.org/pdf/2606.04779v1)

**作者:** Andrea Ferrario `[一作]` (University of Zurich), Andrea Ferrario `[通讯]` (University of Zurich)

**通讯引用:** 2126 | [OpenAlex ID](https://openalex.org/A5057722785)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于树形结构的多代理人 HAI（人机交互）框架，用来形式化并评估“互补性”，即团队预测在点wise‑min 参考基准上优于任一单体预测。

**💡 创新点**

创新点包括：① 将互补性与协议拓扑关联，构建树相对互补性函数；② 对二元聚合规则的可联想性与关联性进行严格分析；③ 在回归问题下提供几何解（欧氏距离最小化）并给出闭式最优权重；④ 证明在回归中通过 Tamari 移动可实现树重参数化保持互补性，并给出 N=4 的五角恒等式；⑤ 在二分类任务下证明内部聚合规则与端点单调损失不可实现互补性，并提出放大式 logit 聚合来突破。

**🔧 技术方法**

使用的技术主要有：树形递归聚合、局部二元组合规则（选择器、线性/指数平均、准算术平均）、几何优化、Tamari 层次结构与重参数化、欧氏距离和贝叶斯风险分析、损失函数的端点单调性和内部性。

**📊 数据集**

实验采用 California Housing 数据集进行回归实验，以及人工生成的二分类预测对进行交叉熵互补性实验；其余分析为理论推导。

**📈 对比分析**

比较方法：对比点wise‑min 与整体最佳单体损失；在回归中通过最大化互补性函数来评估模型输出；在分类中通过判定互补性函数是否为正来判断是否达到互补性。实验结果显示：在回归下能显著提升互补性（正值）；在二分类中标准内部聚合规则均无法获得正互补性，只有放大式 logit 聚合在特定参数范围内能产生正互补性。

**⚠️ 局限性**

局限性：① 以点wise‑min 作为基准，可能对实际高风险 HAI 场景不够稳健；② 仅对线性或准算术平均聚合做详细分析，非线性或高阶聚合未覆盖；③ 分类部分的突破仅在放大式 logit 聚合上实现，未给出完整可行方案；④ 实验数据有限，缺乏真实人机交互案例验证。

---

## 387. Description-Code Inconsistency in Real-world MCP Servers: Measurement, Detection, and Security Implications

**arXiv ID:** 2606.04769 | [PDF](https://arxiv.org/pdf/2606.04769v1)

**作者:** Yutao Shi `[一作]` (Fudan University), Min Yang `[通讯]` (Fudan University)

**通讯引用:** 71695 | [OpenAlex ID](https://openalex.org/A5100694840)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了MCP服务器中工具描述与实现代码不一致（DCI）问题，并提出了检测框架

**💡 创新点**

创新点在于：①构建七类DCI分类体系；②设计基于LLM的双向Prompt与仲裁机制；③大规模实测揭示DCI普遍性与安全影响

**🔧 技术方法**

采用结构化代码包构建、LLM（Claude Sonnet）语义推理、双向Prompt与仲裁、静态AST分析

**📊 数据集**

使用19,200对D/C对（D_large）以及400对人工标注（D_real）和560对合成（D_syn）数据集

**📈 对比分析**

与MCPDiff、工具扫描器及传统静态分析器对比，DRA-Prompting实现95%+准确率、96%+F1，显著优于单向Prompt与基线

**⚠️ 局限性**

局限包括：LLM推理的随机性、对运行时动态行为缺乏覆盖、缺乏对跨语言/多平台的支持

---

## 388. Extending the El Farol Bar Game with Partial Observability and Incentive Design

**arXiv ID:** 2606.04753 | [PDF](https://arxiv.org/pdf/2606.04753v1)

**作者:** Iosif Polenakis `[一作]` (Ionian University), Theodore Andronikos `[通讯]` (Ionian University)

**通讯引用:** 599 | [OpenAlex ID](https://openalex.org/A5110317904)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将经典 El Farol Bar 游戏扩展为双向学习框架，将酒吧视为主动 AI 学习玩家，加入部分可观测性和动态定价机制，研究顾客与酒吧在不完全信息下的协同进化。

**💡 创新点**

创新点在于：①将酒吧从被动容量阈值转变为主动机制设计者；②引入部分可观测性和AI驱动的价格调整；③构建顾客与酒吧双向 AI 学习模型，并分析其协同演化与福利影响。

**🔧 技术方法**

主要技术包括：多智能体强化学习（如 Q‑learning、策略梯度）、贝叶斯预测与高斯过程需求估计、无后悔学习（乘子法、Follow‑the‑Regularized‑Leader）、约束强化学习（用于容量与收益限制）以及监督回归模型用于需求预测。

**📊 数据集**

实验使用仿真生成的历史出席与价格数据，没有公开真实数据集，所有评估均基于基准被动酒吧模型的模拟结果。

**📈 对比分析**

对比方法：与传统被动阈值酒吧模型在利用率、总福利和长期利润等指标上进行实验比较；结果显示，在高波动需求下动态定价显著提升利用率与酒吧利润，但在短期可能出现垄断效应；整体表现优于被动模型。

**⚠️ 局限性**

限制包括：仅考虑单一酒吧和同质顾客，未进行收敛性理论证明，缺乏实证验证与公平约束，可能易受顾客操纵，且未扩展至多酒吧或异质需求场景。

---

## 389. Fog of Love: Engineering Virtuous Agent Behavior with Affinity-based Reinforcement Learning in a Game Environment

**arXiv ID:** 2606.04750 | [PDF](https://arxiv.org/pdf/2606.04750v1)

**作者:** Ajay Vishwanath `[一作]` (University of Agder), Christian Omlin `[通讯]` (University of Agder)

**通讯引用:** 2854 | [OpenAlex ID](https://openalex.org/A5030608087)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了基于桌面角色扮演游戏《Fog of Love》的多智能体强化学习环境，并在此环境中训练并评估使用亲和力正则化的强化学习代理，以实现人工“美德”行为。

**💡 创新点**

创新点在于：①将亲和力基强化学习（ab‑RL）与多智能体深度确定性策略梯度（MADDPG）相结合，提出局部（状态依赖）亲和力正则化方法；②利用复杂的社会实践游戏作为实验平台，研究竞争与合作双重目标下的美德实现；③通过实验展示局部亲和力能显著提升目标达成率和误差率，并实现可解释的道德行为。

**🔧 技术方法**

使用技术包括多智能体深度确定性策略梯度（MADDPG）、亲和力基强化学习（policy regularization）、局部亲和力正则化以及基于OpenAI Gym实现的Fog of Love RL环境。

**📊 数据集**

数据来源为自生成的Fog of Love游戏情境（场景卡、选项卡）及其对应的状态、动作与奖励序列，属于自制模拟数据。

**📈 对比分析**

对比方法：baseline MADDPG 与全局亲和力以及局部亲和力两种正则化配置。评估指标为成功率、目标误差、测试得分和合作目标成功率。实验结果显示：局部亲和力将成功率提升至约0.6–0.7、目标误差降至<30、测试得分提升；全局亲和力亦有提升但不及局部。

**⚠️ 局限性**

局限性：①实验仅在简化版Fog of Love（无章节、命运等）上进行；②只考虑两人对弈，未扩展到更大规模多智能体；③亲和力概率及正则化系数需人工调参，调参成本较高；④在更复杂或真实环境中的泛化与公平性尚未得到充分验证。

---

## 390. GraphAlg Playground: An Online Platform for Learning and Experimenting with the GraphAlg Language

**arXiv ID:** 2606.04813 | [PDF](https://arxiv.org/pdf/2606.04813v1)

**作者:** Daan de Graaf `[一作]` (Eindhoven University of Technology), Nikolay Yakovets `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 488 | [OpenAlex ID](https://openalex.org/A5082856403)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文开发了一个基于 WebAssembly 的 GraphAlg Playground，提供在线代码编辑、实时编译诊断和可视化输出，支持学习和快速原型。

**💡 创新点**

创新点在于将 GraphAlg 语言完全集成到浏览器端，提供即时错误提示和交互式教程，打破了传统离线部署和数据导出的痛点。

**🔧 技术方法**

使用了 JavaScript 前端、C++ WebAssembly 后端、GraphAlg 编译器、矩阵运算与半环抽象、以及 WebGL 图形可视化等技术。

**📊 数据集**

采用了科学文献引文网络和标准图算法基准（PageRank、单源最短路径、连通分量）作为测试数据集。

**📈 对比分析**

与 Neo4j GDS、DuckDB 以及其他在线 playground 比较，GraphAlg 在相同查询中实现了更快的执行时间和更低的代码量，且在 AvantGraph 上的性能优于传统数据库。

**⚠️ 局限性**

局限在于 WebAssembly 运行时尚未优化，主要适用于学习和小规模实验，无法满足大规模生产级工作负载。

---

## 391. Contrastive Learning and Correlation Clustering for Sequences of Network Telescope Data

**arXiv ID:** 2606.04733 | [PDF](https://arxiv.org/pdf/2606.04733v1)

**作者:** Jannik Presberger `[一作]` (Technische Universitaet Dresden), Bjoern Andres `[通讯]` (Technische Universitaet Dresden)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用 Transformer 对网络流记录序列进行无监督对比学习，学习可捕捉源 IP 和扫描器语义的嵌入；随后以相似度为基础做相关聚类，提取潜在的扫描器群组。

**💡 创新点**

① 利用同一源 IP 下不同子序列作为正样本，无需人工增强或预训练；② 通过在相似度上引入偏移 δ 并使用相关聚类（贪心收缩）实现无需预先设定簇数的聚类；③ 展示即使背景流量混杂，模型仍能恢复有意义的扫描器结构。

**🔧 技术方法**

Transformer 编码器、对比学习（SimCLR/NT‑Xent）、cosine 相似度、相关聚类（greedy additive edge contraction）、t‑SNE 可视化。

**📊 数据集**

UCSD‑NT 网络天文台的流记录（约10.5M IP、3.6B 包/小时），按源 IP 划分训练/测试，并用少量扫描器标签做评估。

**📈 对比分析**

在 Test‑Unseen‑Seq 与 Test‑Unseen‑Src 上进行相似度分布对比，发现同源或同扫描器的序列相似度显著更高；相关聚类在测试集上的 weighted F1≈0.91、精度≈0.92、召回≈0.90，表明模型能在未标记背景中恢复语义。

**⚠️ 局限性**

标注稀疏且粗糙，导致聚类对扫描器标签的分辨率有限；阈值 δ 的选取需要在训练集上调参，且未与传统检测/聚类方法进行直接对比，结果仍属于探索性工作。

---

## 392. HapTile: A Haptic-Informed Vision-Tactile-Language-Action Dataset for Contact-Rich Imitation Learning

**arXiv ID:** 2606.04825 | [PDF](https://arxiv.org/pdf/2606.04825v1)

**作者:** Amirhosein Alian `[一作]` (King's College London), Shan Luo `[通讯]` (King's College London)

**通讯引用:** 2852 | [OpenAlex ID](https://openalex.org/A5012646628)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

收集并发布了 HapTile 数据集，该数据集包含多模态（视觉、触觉、语言、机器人动作）并通过带有触觉反馈的遥控操作实现了日常接触丰富的操纵演示。

**💡 创新点**

将触觉反馈嵌入遥控管线、提供指尖视知觉触觉传感器、将语言作为政策条件而非仅元数据、同时收集动作轨迹、实现可复现的低成本硬件。

**🔧 技术方法**

使用 UR5e 机器人 + Robotiq 2F-85 双指触觉传感器、Meta Quest VR 控制器、光学触觉传感器+标记跟踪、差分运动量估计、haptic 反馈、Diffusion Policy 与 π_0 预训练模型。

**📊 数据集**

HapTile 数据集（自建），包含 100+ 任务/演示，涵盖 YCB 物体、日常操作；并在基准实验中对比了 Diffusion Policy 与 π_0 的表现。

**📈 对比分析**

通过在四个高触觉依赖任务（瓶子翻转、白板擦拭、倒液体、插销插孔）上使用 V-only、V+T、V+TM 三种模态配置，对 Diffusion Policy 与 π_0 进行10次演示的成功率评估，结果显示加入触觉可显著提升成功率，尤其是 V+TM 在瓶子翻转和 π_0 在插销插孔上实现 90% 成功率。

**⚠️ 局限性**

场景多样性有限、仅在单一实验室环境收集；haptic 反馈仅为二进制阈值，无法捕捉连续力信息；模拟数据缺乏，难以实现大规模扩展；以及对预训练模型的适配仍需改进。

---

## 393. BiasGRPO: Stabilizing Bias Mitigation in High-Variance Reward Landscapes via Group-Relative Policy Optimization

**arXiv ID:** 2606.04807 | [PDF](https://arxiv.org/pdf/2606.04807v1)

**作者:** Saket Reddy `[一作]` (University of Illinois), ChengXiang Zhai `[通讯]` (University of Illinois)

**通讯引用:** 31072 | [OpenAlex ID](https://openalex.org/A5028518494)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了BiasGRPO框架，将Group Relative Policy Optimization应用于大语言模型的社会偏见缓解，构建了专门的偏见奖励模型并在Microsoft Phi-2上进行微调。

**💡 创新点**

创新点在于证明GRPO在高方差的偏见奖励空间中优于传统的DPO和PPO，并公开了高效的奖励模型和多领域偏见数据集，为后续研究提供了可复用的工具。

**🔧 技术方法**

采用的技术包括GRPO算法、RoBERTa训练的偏见奖励模型（基于Best‑Worst Scaling与Iterative Luce Spectral Ranking）、合成数据扩增、奖励模型与在线探索结合的RLHF微调。

**📊 数据集**

使用的数据集包含BiasDPO、Civil Comments、UnQover（共20,999条提示）及其人工/LLM注释的2,930句偏见样本；评估使用BOLD、RealToxicityPrompts、BBQ和TruthfulQA四个基准。

**📈 对比分析**

通过在Phi-2上分别训练DPO、PPO和GRPO，并在上述四个基准上评估，GRPO在BOLD、RTP和BBQ的偏见得分均最低，TruthfulQA得分不降反升，显示出最优的偏见缓解效果且保持了模型能力。

**⚠️ 局限性**

局限性包括仅在3B参数模型上验证，未充分探测在更大规模模型上的可扩展性；组大小设置固定，未尝试自适应策略；奖励模型训练依赖LLM注释，可能存在循环偏见风险。

---

## 394. Crafting Your Evolving Dreams: Concept-Incremental Versatile Customization

**arXiv ID:** 2606.04797 | [PDF](https://arxiv.org/pdf/2606.04797v1)

**作者:** Jiahua Dong `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fahad Shahbaz Khan `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 39966 | [OpenAlex ID](https://openalex.org/A5100760570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出连续可定制扩散模型（CCDM），实现概念增量学习、单/多概念合成、编辑与风格迁移，并解决灾难性遗忘与概念忽视问题。

**💡 创新点**

创新点包括：①属性解耦LoRA（AD‑LoRA）与相关性引导聚合（RGA）双重机制，自动保留概念特征并利用任务相似度提升增量学习；②可控区域上下文合成模块，用统一注意力掩码保证不同区域语义独立并平滑边界，从而消除多概念合成中的概念忽视。

**🔧 技术方法**

采用LoRA低秩微调、层级概念token、正交约束、Fisher重要性掩码、跨层/跨任务相关度评估、注意力遮罩、CLIP文本编码以及UNet去噪扩散等技术。

**📊 数据集**

使用自研CIL基准，包含35个个性化概念（30对象+5风格+5人物+5动作），在文本‑图像、文本‑视频、文本‑3D三领域进行评估。

**📈 对比分析**

与8类基线（连续学习、LoRA融合、持续扩散等）在IA/TA/FIA/FTA等指标上对比，CCDM在单/多概念、图像/视频/3D任务上均显著优于基线，参数量约减35%，遗忘率最低，性能提升明显。

**⚠️ 局限性**

局限性：模型仍依赖预训练基模型，长期任务累积可能出现记忆衰减；对实时用户反馈的适应性有限；对高度相似概念的分辨仍有一定挑战。

---

## 395. A Pathology Foundation Model for Gastric Cancer with Real-World Validation

**arXiv ID:** 2606.04792 | [PDF](https://arxiv.org/pdf/2606.04792v1)

**作者:** Ling Liang `[一作]` (Hong Kong University of Science and Technology), Li Liang `[通讯]` (Southern Medical University)

**通讯引用:** 17918 | [OpenAlex ID](https://openalex.org/A5062520365)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并评估了专门针对胃癌病理的基础模型GRACE，用于胃黏膜病变诊断、肿瘤分型、分子预测及预后评估。

**💡 创新点**

创新点在于：①基于胃癌专属数据的LoRA微调+DINO自监督预训练，实现对胃腔特有病理模式的精准编码；②使用ABMIL多实例学习将WSI级特征聚合为病人级预测；③在多中心、真实世界前瞻性数据上进行验证，并开展随机交叉读者研究验证临床协同效应。

**🔧 技术方法**

主要技术包括：低秩适配（LoRA）+DINO自监督预训练、ABMIL注意力多实例学习、patch级特征提取、数据预处理与质量控制、前瞻性安全门控阈值及读者实验的统计分析。

**📊 数据集**

使用48,364张H&E染色WSI（37,493例病人）进行预训练，22,645张WSI（11,774例病人）进行下游评估，涵盖9家医院；前瞻性验证样本来自5家医院，共计约28,656张WSI。

**📈 对比分析**

与泛化性PFM（Virchow2、CONCH、UNI）对比，GRACE在26项分类任务上macro‑AUC平均0.9188，排名第一；在外部和前瞻性评估中保持最高特异性；在读者研究中准确率提升约8%（从82.0%到89.9%），诊断时间缩短约15%。

**⚠️ 局限性**

局限性包括：验证集中主要为亚洲人群，缺乏跨族裔/地理验证；仅使用单模态H&E图像，未整合内镜、影像或临床数据；在某些稀有分型或边缘病理状态下仍表现不佳。

---

## 396. AIP: A Graph Representation for Learning and Governing Agent Skills

**arXiv ID:** 2606.04781 | [PDF](https://arxiv.org/pdf/2606.04781v1)

**作者:** Zachary Blumenfeld `[一作]` (Neo4j), Jim Webber `[通讯]` (Neo4j)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Agent Instruction Protocol（AIP），将人类撰写的agent技能转换为带有脚本与自然语言节点的有向执行图并编译为YAML格式

**💡 创新点**

创新点在于通过结构化图谱和模式化脚本化步骤解决自由散文技能的可执行性、可调试性和可改进性问题，同时提供编译器元技能以自动化转换

**🔧 技术方法**

使用YAML schema验证、脚本化节点、图形化执行图、编译器元技能、Neo4j可视化、以及Claude Sonnet模型进行推理

**📊 数据集**

基准数据集为SkillsBench（94项任务），评估27个任务的 5 次试验，分为人类精心策划技能与AIP编译技能两种格式

**📈 对比分析**

通过在同一Solver（Claude Sonnet）下对比两种技能格式，使用平均奖励、通过率、墙钟时间等指标；AIP提升平均奖励从0.599到0.705（+0.106，p=0.011），通过率从53.3%升至67.4%，墙钟时间略微下降

**⚠️ 局限性**

局限包括：脚本改进与图结构改动混合导致的因果混淆、样本量小、Verifier全或无奖励导致的高Tie、评估子集与版本差异、预算限制任务、仅单一模型验证等

---

## 397. Measuring Model Robustness via Fisher Information: Spectral Bounds, Theoretical Guarantees, and Practical Algorithms

**arXiv ID:** 2606.04767 | [PDF](https://arxiv.org/pdf/2606.04767v1)

**作者:** Chong Zhang `[一作]` (Xi'an Jiaotong-Liverpool University), Xiaobo Jin `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于 Fisher 信息矩阵谱范数的无攻击依赖鲁棒性评估框架，并实现了对深度网络在输入空间的最坏情况敏感度量。

**💡 创新点**

创新点在于将 FIM 与输入雅可比方差等价，给出常见网络组件的闭式谱上界，首次实现跨架构的理论鲁棒性排序，并提供可扩展的白盒/黑盒估算算法。

**🔧 技术方法**

使用 Fisher 信息理论、谱分析、Power Iteration、Hutchinson 估计、有限差分等技术来计算 FIM 的最大特征值或其逆。

**📊 数据集**

实验覆盖了 CIFAR‑10/100、ImageNet 以及医学影像数据集，以验证方法在不同任务与规模下的适用性。

**📈 对比分析**

将所提出的谱指标与 PGD、CW、AutoAttack 等攻击手段的鲁棒性结果进行对齐，相关性显著，能够有效预测模型的相对鲁棒性，且在理论上提供更直观的解释。

**⚠️ 局限性**

局限性包括：计算成本仍高，尤其在极大模型和高维输入时需要分批估计；方法聚焦于输入空间的敏感度，未覆盖模型参数不确定性等其他鲁棒性来源。

---

## 398. A French Corpus Annotated for Multiword Expressions with Adverbial Function

**arXiv ID:** 2606.04828 | [PDF](https://arxiv.org/pdf/2606.04828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 399. COP-Q: Safety-First Reinforcement Learning for Robot Control via Cholesky-Ordered Projection

**arXiv ID:** 2606.04749 | [PDF](https://arxiv.org/pdf/2606.04749v1)

**作者:** Guopeng Li `[一作]` (Delft University of Technology), Julian F. P. Kooij `[通讯]` (Delft University of Technology)

**通讯引用:** 1843 | [OpenAlex ID](https://openalex.org/A5074902093)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种安全优先的多目标Q学习方法COP-Q，利用Cholesky分解结合协方差信息构造保守的Q值估计。

**💡 创新点**

创新点在于：①在向量化Q值上推广置信区间，②通过Cholesky分解将安全性优先级编码进协方差，从而在保持安全保守性的同时自适应降低对奖励的过度保守；③仅需一个置信距离超参数，计算开销极低。

**🔧 技术方法**

使用向量化置信区间、Cholesky分解、联合Critic集成、Soft Actor‑Critic框架实现；对安全目标采用“安全优先”排序。

**📊 数据集**

在Brax机器人行走任务（四种机器人配置）和Safety‑Gymnasium安全导航任务（Goal2、Button2）上进行实验。

**📈 对比分析**

与多种基线（独立双Q、标量化双Q、保守双Q、SACLag、CAL、ORAC、RCPO等）比较，COP-Q在硬安全任务中减少跌倒次数并提升样本效率；在软安全任务中保持成本阈值且返回与最先进方法相当或更好。

**⚠️ 局限性**

依赖于协方差矩阵的数值稳定性，当安全信号二值稀疏或协方差矩阵接近奇异时，COP-Q会失稳；当前仅考虑总不确定性，未区分模型不确定性和噪声；对更广泛多目标问题的适用性仍待验证。

---

## 400. Curvature-aware dynamic precision approach for physics-informed neural networks

**arXiv ID:** 2606.04736 | [PDF](https://arxiv.org/pdf/2606.04736v1)

**作者:** Yingjie Shao `[一作]` (Wageningen University & Research), Taniya Kapoor `[通讯]` (Wageningen University & Research)

**通讯引用:** 226 | [OpenAlex ID](https://openalex.org/A5082424531)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于曲率的动态精度控制方法，用于训练物理信息神经网络（PINN），在训练过程中根据L‑BFGS优化器的曲率信息动态切换FP32与FP64两种数值精度，兼顾计算效率与数值稳定性。

**💡 创新点**

创新点在于：①利用L‑BFGS历史曲率信息构造轻量级的曲率代理，并以此驱动精度切换；②实现了在不牺牲FP64级别精度的前提下，显著降低训练时间；③方法与网络结构无关，可应用于多种PINN架构。

**🔧 技术方法**

核心技术包括：L‑BFGS二阶优化器、曲率代理（基于s_k^T y_k / s_k^T s_k）、指数加权移动平均平滑、阈值控制的精度切换策略；实现基于PyTorch的混合精度训练。

**📊 数据集**

使用了四个经典的PINN失效模式基准方程（对流方程、反应方程、波动方程、Allen–Cahn方程）和一个光照驱动的生长ODE，所有数据均来自公开代码库或自行生成的高精度求解。

**📈 对比分析**

与固定FP32、FP64两种精度训练做对比，结果显示动态精度在保持或略优于FP64预测精度的同时，平均训练时间缩短约10%–30%（取决于方程和网络结构），并显著提升了在FP32下易出现的数值不稳定问题。

**⚠️ 局限性**

局限性包括：曲率代理仅基于有限记忆的L‑BFGS估计，可能无法充分捕捉高度非凸或极度不良条件的训练局部几何；阈值τ_z需针对不同问题手工调节；目前仅实现二进制精度切换，未探索多级精度或更复杂的控制策略；对其他二阶优化器或非PINN任务的泛化仍待验证。

---

## 401. Trace-Mediated Peak Bias: Bridging Temporal Credit Assignment and Cognitive Heuristics in Deep Reinforcement Learning

**arXiv ID:** 2606.04735 | [PDF](https://arxiv.org/pdf/2606.04735v1)

**作者:** Viktor Veselý `[一作]` (University of Groningen), Matthia Sabatelli `[通讯]` (University of Groningen)

**通讯引用:** 182 | [OpenAlex ID](https://openalex.org/A5074266137)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本研究通过对深度强化学习中资格迹与非线性函数逼近交互的实验，识别并量化了一种名为 Trace-Mediated Peak Bias（TMPB）的系统性偏差，并将其与人类“峰值-结束”记忆偏差对应。

**💡 创新点**

其创新之处在于首次将 TMPB 形式化为 RL 中的计算缺陷，并揭示适应性优化器（如 RMSprop）可通过二阶矩归一化消除该偏差，从而说明了人类偏差可能源自梯度更新机制。

**🔧 技术方法**

研究使用了 TD(λ) 资格迹、ReLU 神经网络价值函数逼近、标准随机梯度下降（SGD）与自适应 RMSprop 优化器，以及基于梯度冲击的机制分析。

**📊 数据集**

所用数据集为构造的 Two‑Door MDP（两条路径，奖励分布为常数或高峰稀疏），属于自定义的模拟环境。

**📈 对比分析**

通过在不同 λ 下对 SGD 与 RMSprop 进行价值评估比较，实验发现 SGD 在中等 λ 取值区间出现峰值路径价值过高的“非理性区”，而 RMSprop 能消除这一现象，显示出更理性、更准确的价值估计。

**⚠️ 局限性**

限制方面，实验仅在简化的合成 MDP 上验证，未涉及更复杂或真实世界任务；此外，缺乏对不同网络架构、学习率和奖励尺度的广泛验证，因而结果的普适性尚待进一步检验。

---

## 402. The Usefulness Gap in Proof-of-Useful-Work: An Empirical Study of Pearl's cuPOW Protocol

**arXiv ID:** 2606.04819 | [PDF](https://arxiv.org/pdf/2606.04819v1)

**作者:** Abhinaba Basu `[一作]` `[通讯]` (National Institute of Electronics and Information Technology), Abhinaba Basu (National Institute of Electronics and Information Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对Layer‑1区块链Pearl的Proof‑of‑Useful‑Work (PoUW) 协议进行系统实证测评，证明其实际矿工仅执行随机矩阵乘法，未产生任何有用的 AI 推理或训练；

**💡 创新点**

首次量化 PoUW 的“可验证‑有用性”张力，在真实网络上测得利用率、经济收益与硬件兼容性等指标，并提出关闭利用性缺口的五种技术路径；

**🔧 技术方法**

利用网络采样、二进制字符串分析、GPU 性能基准、分布统计检验（kurtosis 等）以及自研的跨平台开源矿工实现；

**📊 数据集**

以 Pearl 主网公开区块、AlphaPool API 采集的 8,012 台矿工硬件配置、GPU 价格历史（vast.ai 等）和 PoUW 交易市场 PRL 价格为数据集；

**📈 对比分析**

对比标准矿工(alpha‑miner)与自研矿工在 NVIDIA、AMD、CPU、Apple Silicon 上的哈希率、收益和 ROI，发现所有硬件平台均可完成 PoUW，收益均为负值，表明当下 PRL 价格下矿工收益不足；

**⚠️ 局限性**

局限性包括：仅评估公开矿工软件；二进制字符串分析不能绝对排除隐藏的推理代码；经济模型依赖当前 PRL 价格；未对未来 PoUW 设计或区块链治理进行动态跟踪。

---

## 403. Uncertainty-Aware (Un)Supervised Few-Shot User Adaptation for On-Device Personalized Human Activity Recognition

**arXiv ID:** 2606.04798 | [PDF](https://arxiv.org/pdf/2606.04798v1)

**作者:** Maximilian Burzer `[一作]` (Karlsruhe Institute of Technology), Tobias Röddiger `[通讯]` (IPAI Foundation gGmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

将已有的预训练HAR分类器改造成无梯度、可源免费使用的原型网络，并通过贝叶斯原型更新和MAP‑EM原型更新实现对用户数据的零样本、监督式少样本及无监督式少样本自适应。

**💡 创新点**

创新点在于：①使用先验原型使得在无用户数据时可保持零样本性能；②引入闭式贝叶斯原型更新和一次性MAP‑EM更新，使自适应过程完全无梯度、极低计算开销；③兼顾监督与无监督少样本场景，且在超低样本（1 shot，3秒）下仍能避免性能退化。

**🔧 技术方法**

技术包括原型网络（Prototypical Networks）、先验原型正则化、贝叶斯原型更新、MAP‑EM（最大后验期望最大化）算法、T‑SNE可视化以及嵌入空间居中等。

**📊 数据集**

实验数据集为四个主流HAR基准：HHAR、WEAR、HARTH、HAPT，使用TinierHAR作为基线分类器。

**📈 对比分析**

与零样本基线、传统原型估计、逻辑回归探针、PDA、OFTTA等方法比较，本文在所有数据集上均实现了显著提升：监督式最大可达+33.7个百分点，未标注式+32.5个百分点，尤其在1 shot（3秒）下即有+2.76到+33.44个百分点的增益，且避免了其他方法的性能退化。

**⚠️ 局限性**

局限性包括：MAP‑EM需预先知道使用的活动类别，无法完全零干预；先验原型需离线计算原始训练集的嵌入，若想真正源免费需进一步研究；在极低样本情况下仍可能受极端噪声影响。

---

## 404. UniFair: A unified fair clustering approach based on separation and compactness

**arXiv ID:** 2606.04777 | [PDF](https://arxiv.org/pdf/2606.04777v1)

**作者:** Antonia Karra `[一作]` (University of Ioannina), Aristidis Likas `[通讯]` (University of Ioannina)

**通讯引用:** 10820 | [OpenAlex ID](https://openalex.org/A5007939716)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了统一的分离公平与社会公平双重目标 UniFair，用于改进聚类算法的公平性。

**💡 创新点**

创新点在于首次引入基于决策边界的 counterfactual 距离来衡量分离公平，并将其与传统社会公平（组内失配）结合，形成梯度可优化的统一目标。

**🔧 技术方法**

采用梯度驱动的 Lloyd‑style 迭代、分离公平与社会公平的梯度推导，以及在自编码器潜在空间中联合优化的深度聚类技术。

**📊 数据集**

实验数据集包括四个表格数据（Adult、Student、Bank、Credit）和两组图像数据（MNIST‑USPS、Color Reverse MNIST）。

**📈 对比分析**

与 FairLloyd、Deep Fair Clustering（DFC）等方法对比，UniFair 在保持聚类质量的前提下显著降低分离公平与社会公平差距，整体性能与最优方法相当甚至更好。

**⚠️ 局限性**

局限性包括需手动调节公平权重、在高维数据上公平提升伴随轻微聚类损失增加，且未针对多重受保护属性或动态群体特征进行深入研究。

---

## 405. Activation Steering of Video Generation Models via Reduced-Order Linear Optimal Control

**arXiv ID:** 2606.04775 | [PDF](https://arxiv.org/pdf/2606.04775v1)

**作者:** Jihoon Hong `[一作]` (Georgia Institute of Technology), Glen Chou `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 437 | [OpenAlex ID](https://openalex.org/A5006149535)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了Latent Activation Linear-Quadratic Regulator (LA-LQR)，一种将文本到视频（T2V）推理视为动态系统并在低维潜在子空间中实施闭环最优控制的激活调节框架，用以在不降低视觉质量的前提下抑制模型生成的有害内容。

**💡 创新点**

创新点包括：①将T2V生成过程建模为可控动态系统；②利用对比式提示对激活进行低维潜在投影，从而实现可计算的控制问题；③在潜在空间中构造线性特征设点并通过LQR获得最小干预的反馈信号；④提供理论误差上界并实验证明潜在动力学的线性近似有效；⑤在实际安全基准上实现了更低的违规率。

**🔧 技术方法**

技术手段涵盖：最优控制与线性二次调节器（LQR）、随机SVD降维、对比提示构造潜在子空间、Jacobian-Vector 乘法实现局部线性化、闭环反馈控制以及对比特征设点的实现。

**📊 数据集**

实验数据集包括：T2VSafetyBench（用于评估色情、暴力、版权等违规类别）和SafeSora（评估暴力、恐怖主义、种族主义等类别），以及在Wan2.1-T2V-14B和HunyuanVideo-1.5模型上进行概念调节与安全性实验。

**📈 对比分析**

与基线方法（对比向量加权更新、SAFREE、LLM辅助token选取等）在违规率、VBench主题一致性和CAPS语义保持三项指标上进行对比，LA-LQR在违规率上显著下降，同时保持或提升视频质量与语义一致性，显示出更优的安全‑质量平衡。

**⚠️ 局限性**

局限性包括：①需要存储潜在投影基和局部动力学，存在显著内存开销；②对潜在维度与对比子空间质量高度敏感；③需要手动调节LQR权重与目标设点，缺乏自动化；④局部线性假设在极端或对抗性提示下可能失效；⑤仅适用于白盒模型，需要额外的过滤或后处理才能在黑盒场景中使用。

---

## 406. Do Foundation Models See Biology? Evaluating Attention Coherence with Spatial Transcriptomics in Glioblastoma

**arXiv ID:** 2606.04764 | [PDF](https://arxiv.org/pdf/2606.04764v1)

**作者:** Dilakshan Srikanthan `[一作]` (Queen's University), Parvin Mousavi `[通讯]` (Queen's University)

**通讯引用:** 3679 | [OpenAlex ID](https://openalex.org/A5040401197)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用空间转录组对神经胶质母细胞瘤病理图像的注意力图进行无假设的定量评估，比较多种基础模型与ResNet50的解释性；

**💡 创新点**

提出基于空间转录组的客观评估框架，揭示注意力映射与多基因转录程序的生物学一致性，并发现空间连贯不等同于生物学解释，证明不同编码器关注不同生物学区室；

**🔧 技术方法**

使用注意力多实例学习（abMIL）训练单/多任务分类器，结合六个预训练编码器提取特征，计算注意力权重，利用Visium空间转录组数据做Cohen's d富集分析，并用Moran's I评估空间自相关；

**📊 数据集**

训练集为CPTAC-GBM H&E全切片（171患者），外部验证集为TCGA-GBM（107患者），空间转录组验证使用18个Visium样本（约69,000 spots）；

**📈 对比分析**

采用5折病人分层交叉验证评估AUROC，外部验证显示内部排名颠倒；基础模型相较ResNet50在路径级别Cohen's d提升约2.3倍，空间连贯度与生物学富集呈负相关；

**⚠️ 局限性**

结果可能受预测标记与注意力关联的因果性或共形态特征共线性影响；样本量有限且仅限GBM，框架对其他组织的推广需进一步验证。

---

## 407. An Empirical Audit of Input Encoders for Multi-Channel Signal Transformers

**arXiv ID:** 2606.04752 | [PDF](https://arxiv.org/pdf/2606.04752v1)

**作者:** Ossi Lehtinen `[一作]` `[通讯]` (Ocon Oy), Ossi Lehtinen (Ocon Oy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并比较了八种多通道输入编码器在 Transformer 时间序列预测中的效果。

**💡 创新点**

明确发现与更复杂编码器相比，简单的逐通道线性投影已能取得几乎同等性能，并阐明任务损失本身可驱动通道正交化。

**🔧 技术方法**

采用 Transformer 预 LN 结构，配合多种编码器（sum、linear、linear‑ortho、mlp、concat、linear‑ppe、ci、cat），对比负对数似然、线性探测、Gram 矩阵、位置投影正交化等指标。

**📊 数据集**

在合成信号（4‑16 通道、长度 160）和真实电力负荷数据 ETTh1（7 通道）上进行实验。

**📈 对比分析**

通过 20 个随机种子、paired‑t 检验、NLL 及准确率评估，发现线性投影与其它高级编码器在 NLL 上相差 0.02‑0.05，仅 linear‑ppe 在小 C 时略有优势；ci 与 cat 在合成数据上明显落后。

**⚠️ 局限性**

研究仅覆盖数值时间序列，假设通道可区分；对高通道数或交叉变量的真实效益仍有限，且实验受限于 512 训练样本及单一 Transformer 结构。

---

## 408. Physics-Informed Video Generation via Mixture-of-Experts Latent Alignment

**arXiv ID:** 2606.04737 | [PDF](https://arxiv.org/pdf/2606.04737v1)

**作者:** Cong Wang `[一作]` (CASIA), Zhibo Chen `[通讯]` (ZGCA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种将物理知识嵌入预训练流匹配视频生成模型的框架PILA，利用物理属性银行和多专家路由实现对生成过程的物理一致性修正；

**💡 创新点**

创新点在于通过锚定场估计构建可操作的物理属性银行，并用标签优先的专家路由与类别特定的残差约束，实现了在不修改预训练模型参数的前提下，轻量级地注入物理一致性；

**🔧 技术方法**

核心技术包括锚定场估计（AFE）、标签优先遮蔽专家路由（LPMER）、多专家残差约束（CSRC）以及属性到流的轻量级解码器；

**📊 数据集**

在WISA-80K、VideoPhy-2、VBench-2.0和PhyGenBench等包含物理标签的视频数据集上进行训练与评估；

**📈 对比分析**

在这些基准上，PILA在物理合理性指标（如Joint、Rule、Motion、Mechanics、Optics等）上分别比基线提升约10%至30%，并在大模型14B迁移时保持显著优势；

**⚠️ 局限性**

局限性包括对预训练模型的依赖、物理属性银行的手工设计、对极端物理情境的适应性尚未充分验证，以及在复杂交互场景下可能仍出现物理细节误差。

---

## 409. Reconciling Causality and Non-Equilibrium Thermodynamics with Hamiltonian Causal Models

**arXiv ID:** 2606.04822 | [PDF](https://arxiv.org/pdf/2606.04822v1)

**作者:** Dario Rancati `[一作]` (Institute of Science and Technology Austria), Francesco Locatello `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 3629 | [OpenAlex ID](https://openalex.org/A5073157306)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的因果建模框架 Hamiltonian Causal Models（HCMs），在轨迹层面描述受控哈密顿动力学系统的因果关系，并将熵产生作为可观测的因果效应指示量；

**💡 创新点**

创新点在于：①把因果干预视为对哈密顿量的控制，解耦不可变的动力学定律与可调机制；②利用熵产生（及其局部形式）量化路径层面的因果效应与直接父节点；③在非平衡热力学与统计因果学之间架起桥梁，证明 HCM 能推广现有 SCM 并天然处理时间依赖、非平稳与循环因果；

**🔧 技术方法**

使用技术包括：哈密顿动力学与控制理论、随机微分方程（Langevin 动力学）、非平衡热力学中的工作、热量与熵产生、路径空间上的 KL 散度、以及变分估计器（NEEP）用于从轨迹估计局部熵产生率；

**📊 数据集**

数据集为仿真生成的受控 Langevin 系统：线性链、具有隐藏非线性中介的小型系统以及随机 Erdős–Rényi 有向图上构造的 15 维循环系统；

**📈 对比分析**

对比方法：传统终点平均处理效应（Endpoint ATE）与累积处理效应（Cumulative ATE），以及通过阈值化 R_{i|j} 进行父节点恢复；实验结果显示：①投影熵产生的 ATE 在两种实验设置中显著非零，远超传统 ATE；②在随机循环图中，利用局部熵产生率的响应 R_{i|j} 可实现 0.85–0.86 的 F1，召回率 0.97–0.98，精度 0.78–0.90，说明父节点识别性能优秀；

**⚠️ 局限性**

局限性：①仅适用于可写入 HCM 结构的系统（如 Langevin 及梯度漂移）；②非梯度漂移的通用时间 SCM 尚未覆盖；③对哈密顿量的学习与对应的因果发现方法尚未成熟；④计数或估计熵产生需要完整轨迹或高质量采样；⑤对逆向干预（Maxwell 矩阵）等信息反馈的理论处理仍不完整。

---

## 410. Dream.exe: Can Video Generation Models Dream Executable Robot Manipulation?

**arXiv ID:** 2606.04811 | [PDF](https://arxiv.org/pdf/2606.04811v1)

**作者:** Rui Zhao `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**通讯引用:** 4005 | [OpenAlex ID](https://openalex.org/A5068937750)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过将视频生成模型产生的操控视频转化为可执行的机器人轨迹，并在物理仿真环境中执行，构建了一个从视频到执行的完整评估框架。

**💡 创新点**

首次将视频生成与机器人实际执行闭环结合，以任务成功率作为衡量模型物理可执行性的关键指标，从视觉质量转向“可执行性”评估。

**🔧 技术方法**

使用视频到轨迹提取管线（蒙版初始化、2D跟踪、深度估计与3D提升、末端执行器轨迹校准、抓取时序推断）以及MuJoCo/robosuite仿真执行；视觉评估借助VLM评判。

**📊 数据集**

基于RoboCasa365手工挑选的101个操控任务，按单一物体、双物体交互、复合多阶段三层难度分布。

**📈 对比分析**

与8种模型（闭源前沿、开源以及机器人专用策略）在视觉质量、轨迹相似度和执行成功率三维度进行对比，部分模型（如SeedDance 2.0、Wan 2.7）在级别1、2任务上可达20%~30%的成功率；视觉指标与执行性能高度不相关。

**⚠️ 局限性**

主要限制包括：长时序复合任务仍难以完成；深度估计误差成为轨迹提取瓶颈；模型学习的物理约束不够完整；机器人视角与任务匹配不一致导致执行失败。

---

## 411. Z-FLoc: Zero-Shot Floorplan Localization via Geometric Primitives

**arXiv ID:** 2606.04788 | [PDF](https://arxiv.org/pdf/2606.04788v1)

**作者:** Ayumi Umemura `[一作]` (ETH Zurich), Daniel Barath `[通讯]` (ETH Zurich)

**通讯引用:** 1376 | [OpenAlex ID](https://openalex.org/A5016636021)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种零训练的地板平面定位方法，利用从单目图像生成的鸟瞰图中的几何原语与平面地图匹配，实现相机位姿估计。

**💡 创新点**

通过提取线段和圆形几何原语并采用最小求解器与鲁棒估计，完全无需训练，能跨环境泛化，克服模态和坐标差距。

**🔧 技术方法**

BEV重建、Mask2Former墙体分割、最小几何求解器（3L、2L、2C、LC）、混合RANSAC、双重评分（一致性与自由空间违例）以及Levenberg–Marquardt精细化。

**📊 数据集**

Gibson(t) 合成数据、LaMAR 实际建筑数据（HGE、CAB）以及公开的相关基准。

**📈 对比分析**

与 F³Loc、UnLoc 等学习型基线对比，零训练方法在未见环境上实现 100% 成功率，精度与学习模型相当或更优，特别是在跨建筑时表现突出。

**⚠️ 局限性**

主要受限于密集 3D 重建的计算开销，面对对称结构易产生多模假设，阈值需手工设定，未对实时增量处理做优化。

---

## 412. The Preisach Extremum Stack is a Shannon-Minimal Sufficient Statistic for Rate-Independent Functionals

**arXiv ID:** 2606.04784 | [PDF](https://arxiv.org/pdf/2606.04784v1)

**作者:** Piotr Frydrych `[一作]` (Warsaw University of Technology), Piotr Frydrych `[通讯]` (Warsaw University of Technology)

**通讯引用:** 110 | [OpenAlex ID](https://openalex.org/A5012740592)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

证明 Preisach 极值栈是所有可计算、因果、速率独立函数的 Shannon-最小充分统计量，并给出了相应的互信息等价与最小性定理；同时阐述了在线栈维护在估计 Preisach 测度时的空间节省。

**💡 创新点**

首次从信息理论角度构造可计算映射，将任何速率独立查询映射到栈，并证明该栈在互信息意义下达到最小；并将这一结论与 Kolmogorov 最小性对比，展示其在期望意义上的优势。

**🔧 技术方法**

利用擦除（wiping‑out）性质证明栈是速率独立函数的因子；采用可计算性、数据处理不等式、有限指标族以及互信息与 Kolmogorov 复杂度的比较等信息论与算法复杂度技术。

**📊 数据集**

论文为理论证明性质，未使用具体实验数据集，讨论的是随机序列模型及其概率分布。

**📈 对比分析**

与传统需要存储完整输入历史的 NNLS 估计相比，在线维护极值栈将空间需求从 O(n) 降至 O(max_t k_t)，实现显著的内存压缩；在信息量上证明该栈达到 Shannon-最小，保持了对最终查询的无信息损失。

**⚠️ 局限性**

局限性包括：仅适用于单时点的速率独立查询；完整轨迹查询仍需使用栈过程；尚未扩展到向量 Preisach 操作符或在 Fisher–Neyman 充分性框架下的完整表述。

---

## 413. Inference-Time Vulnerability Beyond Shallow Safety: Alignment Along Generation Trajectories

**arXiv ID:** 2606.04778 | [PDF](https://arxiv.org/pdf/2606.04778v1)

**作者:** Kyungmin Park `[一作]` (Hankuk University of Foreign Studies), Taesup Kim `[通讯]` (Seoul National University)

**通讯引用:** 17835 | [OpenAlex ID](https://openalex.org/A5100641870)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在推理阶段注入短语构造生成轨迹，并通过双向轨迹增强与迭代对齐训练，提升模型在遭受中间序列注入攻击时的安全鲁棒性。

**💡 创新点**

将浅层安全视为推理时攻击的一种特殊情况，提出直接对生成轨迹进行对齐的安全对齐方法，突破传统仅关注内部表示或完整响应的局限。

**🔧 技术方法**

采用SimPO无参考偏好优化、QLoRA微调、PCA轨迹可视化以及双向轨迹增强与迭代训练的技术框架。

**📊 数据集**

使用 AdvBench 生成注入轨迹，评估 AdvBench、HarmBench、HEx-PHI、JailbreakBench 等安全基准；同时检验 MMLU、PROST、XSTest 等通用任务的性能。

**📈 对比分析**

与 Egida-DPO、Circuit Breakers、SafeProbing、LAT 等现有方法对比，注入攻击下 ASR 下降至 0–5%，远优于对照组；在未攻击情况下保持低拒绝率并维持接近基线的通用能力。

**⚠️ 局限性**

局限在于对注入序列与阈值的手工设定依赖较强，可能对更复杂或多样化的攻击不足以覆盖；且对高质量语言生成任务的影响尚未系统评估。

---

## 414. SoftPINCH: EMG-Driven Soft Exoskeleton Assistance for Finger Flexion and Grasping

**arXiv ID:** 2606.04776 | [PDF](https://arxiv.org/pdf/2606.04776v1)

**作者:** Nicklas Nikolaj Grønvall `[一作]` (University of Southern Denmark), Saravana Prashanth Murali Babu `[通讯]` (University of Southern Denmark)

**通讯引用:** 340 | [OpenAlex ID](https://openalex.org/A5019963742)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并验证SoftPINCH软可穿戴指尖外骨骼，用EMG驱动实现拇指-食指屈曲及握持辅助。

**💡 创新点**

将卷积+LSTM解码器与注意机制对比，证明CNN+LSTM即可实现99.4% LOSO准确率；软可穿戴结构通过张力传动与磁传感实现轻量化；实时EMG解码与触点反馈协同控制。

**🔧 技术方法**

软材料张力驱动外骨骼、MagSense磁传感、深度学习CNN+LSTM网络、贝叶斯超参搜索、LOSO交叉验证、EMG预处理（Butterworth、Notch、Hampel、RMS、Z-score）。

**📊 数据集**

17位健康志愿者的前臂三通道sEMG记录，包含食指/拇指屈伸/静止共30个9s试验。

**📈 对比分析**

采用LOSO留一受试者外交叉验证与8折交叉验证，比较三种网络，CNN+LSTM和CNN+LSTM-attention均达99.4%准确率，LSTM仅97.8%；功能评估中主动辅助能将握持时肌肉努力降低92.6%。

**⚠️ 局限性**

对受伤或功能障碍人群验证不足；在高负荷1kg以上时辅助仍需用户自力；需要进一步自适应力度与速度控制。

---

## 415. Coarse-to-fine Hierarchical Architecture with Sequential Mamba for Brain Reconstruction

**arXiv ID:** 2606.04772 | [PDF](https://arxiv.org/pdf/2606.04772v1)

**作者:** Hoang-Son Vo `[一作]` (Chonnam National University), Soo-Hyung Kim `[通讯]` (Chonnam National University)

**通讯引用:** 3892 | [OpenAlex ID](https://openalex.org/A5100605822)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于Mamba的两阶段粗细层次架构，用于将图像映射到人类fMRI激活。

**💡 创新点**

创新点在于双流Mamba对全局CLS与局部patch进行分离，采用粗-精细两阶段预测，并通过Mamba‑VAE实现对噪声鲁棒的体素级细化。

**🔧 技术方法**

技术上使用预训练的DINOv2视觉编码器、Mamba网络、对比学习、VAE以及自适应融合门控。

**📊 数据集**

数据集为公开的Natural Scenes Dataset（NSD），包含多受试者的图像‑fMRI对。

**📈 对比分析**

通过与岭回归、MLP、GRU、LSTM、DINOv2线性探针、SynBrain、MindSimulator等基线对比，CHASMBrain在MSE降至0.261、Pearson提升至0.429，显著优于其他方法。

**⚠️ 局限性**

局限性包括受fMRI噪声上限限制、仅使用静态图像特征、缺乏跨数据集验证，以及对下游解码器的依赖。

---

## 416. Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability

**arXiv ID:** 2606.04754 | [PDF](https://arxiv.org/pdf/2606.04754v1)

**作者:** Vincent Bürgin `[一作]`, Stefanie Jegelka `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并理论化了在深度网络中通过参数结构破坏实现的神经元可识别性（neuron identifiability），并探讨其与线性模式连通性（Linear Mode Connectivity）的关系。

**💡 创新点**

创新点在于提出了“有效功能类”（effective function classes）框架，量化神经元在输入子空间上的可实现功能及其实现成本，从而给出对称性破坏如何实现可识别性和无对齐线性模式连通性的理论条件。

**🔧 技术方法**

主要技术包括：对称性破坏的数学建模、子空间支持假设下的几何分析、Mahalanobis 伪范数实现成本计算、谱集中与中心化不变性分析、以及利用激活匹配和线性插值实验验证理论。

**📊 数据集**

使用的实验数据集包括：MNIST（MLP）、CIFAR‑10（ResNet）、以及自定义的高低维高斯混合与低秩合成数据，用于验证对称性破坏、可识别性和LMC的效果。

**📈 对比分析**

通过对齐与非对齐的线性模式连通性实验、激活匹配相似度评估以及神经元交换成本分析进行比较，发现结构对称性破坏不足，必须在“中心支配”或高输入子空间相干性下才可实现低损失桥接；在这些条件下模型在两端及中点的准确率保持高，且对齐后LMC障碍显著下降。

**⚠️ 局限性**

局限性包括：理论假设仅涵盖线性子空间输入、对称性破坏只考虑部分可变参数（如固定权重掩码），未完全涵盖训练动态与非线性激活的复杂交互，且实验主要集中在小规模网络和特定数据集，缺乏对大规模真实任务的验证。

---

## 417. Revisiting Vul-RAG: Reproducibility and Replicability of RAG-based Vulnerability Detection with Open-Weight Models

**arXiv ID:** 2606.04739 | [PDF](https://arxiv.org/pdf/2606.04739v1)

**作者:** Sabrina Kaniewski `[一作]` (Esslingen University), Tobias Heer `[通讯]` (Esslingen University)

**通讯引用:** 904 | [OpenAlex ID](https://openalex.org/A5065269908)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Vul-RAG框架进行可复现性与可复制性研究，使用本地开源权重模型重新实现并评估其在不同LLM上的表现。

**💡 创新点**

系统性验证Vul-RAG在新一代、通用、推理模型下的性能，并揭示性能在约0.30的饱和点，说明模型规模与性能提升不成正比。

**🔧 技术方法**

采用检索增强生成（RAG）结合多维漏洞知识库，使用PyTorch、HuggingFace、BM25+重排序及本地推理实现。

**📊 数据集**

使用Vul-RAG原始基准PairVul数据集，包含2903对Linux kernel CVE函数及其补丁，按1:4拆分训练/测试。

**📈 对比分析**

通过平衡召回、平衡精度和pairwise accuracy三项指标对比各模型的平均表现，最高pairwise accuracy仅0.29，提升有限。

**⚠️ 局限性**

受知识库质量、模型格式化依赖、GPU内存限制及潜在训练数据泄漏影响；大模型部署成本高，整体性能仍处于平台化门槛。

---

## 418. FALSIFYBENCH: Evaluating Inductive Reasoning in LLMs with Rule Discovery Games

**arXiv ID:** 2606.04751 | [PDF](https://arxiv.org/pdf/2606.04751v1)

**作者:** Leonardo Bertolazzi `[一作]` (University of Trento), Raffaella Bernardi `[通讯]` (Free University of Bozen-Bolzano)

**通讯引用:** 3458 | [OpenAlex ID](https://openalex.org/A5020020314)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了FalsifyBench，一种基于WordNet语义层级的Wason 2‑4‑6任务变体，用于评估LLM的假设驱动推理能力。

**💡 创新点**

创新点在于将负面测试（falsification）作为核心推理策略，并提供细粒度回合级分析，揭示模型在假设空间中的探索模式。

**🔧 技术方法**

利用大型语言模型在玩家和守门员（oracle）两角色下交互，通过JSON格式的交互规范实现自动化推理与评估。

**📊 数据集**

构建的数据集为从WordNet抽取的100个代表性游戏，每个游戏由目标规则R及其子类S生成的三元组组成。

**📈 对比分析**

对12种LLM（包括推理型和指令型）在玩家成功率、确认偏差、Oracle准确率等指标上进行对比，结果显示推理模型在负面测试上的优势明显，但整体性能仍未达到最优。

**⚠️ 局限性**

局限性包括仅使用WordNet作为语义知识来源、人工注释样本有限导致Oracle误差评估粗略，以及只关注假设被取代而非更广泛的科学推理过程。

---

## 419. R-APS: Compositional Reasoning and In-Context Meta-Learning for Constrained Design via Reflective Adversarial Pareto Search

**arXiv ID:** 2606.04823 | [PDF](https://arxiv.org/pdf/2606.04823v1)

**作者:** João Pedro Gandarela `[一作]` (Idiap Research Institute), André Freitas `[通讯]` (Idiap Research Institute)

**通讯引用:** 2496 | [OpenAlex ID](https://openalex.org/A5053978668)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并评估了 R-APS 方法，利用推理模式分解和三时刻机制实现可靠的受限设计，主要应用于平面机构合成。

**💡 创新点**

通过推理模式分解把五种推理（归纳、反事实、元归纳、纠正、归纳）分离到独立语境，并在三个时刻实现局部故障定位、最坏情况鲁棒性认证和元归纳记忆失效，首次在受限设计中一次性解决三类结构性失效。

**🔧 技术方法**

冻结的 LLM（如 Llama-3.3-70B、Qwen3-4B），多代理框架（Designer, Critic, Post-Opt Critic, Refinement, Meta-Analyst），Sobol 采样的敏感度分析，Pareto 优化，typed 验证器，规则提取与显式失效。

**📊 数据集**

6 条标准曲线（圆、椭圆、直线、LB、NACA、抛物线）和 26 个英文字母的轨迹，作为平面机构目标曲线。

**📈 对比分析**

与 Enum+ga、模块化 LLM 以及 R-APS 的 ablation 进行对比；在 Chamfer 距离、鲁棒性分数、条形数遵守率上均优于基线；R-APS 在迭代次数上加速了 20%~30%，鲁棒性证书比均匀扰动更紧；4B 方案与 70B 在协议内可比。

**⚠️ 局限性**

额外推理调用和 Sobol 采样带来更高计算成本；元归纳记忆更新受证据累积速率限制；仅在机构合成上验证，跨领域适用性待进一步测试。

---

## 420. OA-CutMix: Correcting the Label Bias of CutMix

**arXiv ID:** 2606.04820 | [PDF](https://arxiv.org/pdf/2606.04820v1)

**作者:** Tobias Christian Nauen `[一作]` (RPTU University Kaiserslautern-Landau), Andreas Dengel `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种对象感知标签重加权的 CutMix 方法（OA‑CutMix），通过预先生成的分割掩码来纠正 CutMix 的标签偏差。

**💡 创新点**

创新点在于仅修正标签分配（不改动图像混合过程），利用离线 SAM3 分割掩码按可见目标面积重新加权标签，从而消除“鬼标签”并显著提升小目标识别效果。

**🔧 技术方法**

技术实现包括：使用 SAM3 对训练集图像进行一次性分割；在混合时统计每个图像的可见目标像素（绝对/相对模式）；用目标像素比例替代原始面积比例生成软标签。

**📊 数据集**

实验数据集涵盖 ImageNet‑200、TinyImageNet、CIFAR‑100、FGVC‑Aircraft、Stanford Cars 与 CUB‑200 等四大架构（ResNet18/50、DeiT‑Ti/S）。

**📈 对比分析**

与 10+ 静态与动态混合方法（如 CutMix、MixUp、PuzzleMix、TokenMix 等）对比，OA‑CutMix 在所有设置下均取得最高 Top‑1 准确率，尤其在小目标场景提升显著，同时训练时间仅为动态方法的 1/10 左右。

**⚠️ 局限性**

局限性包括：依赖 SAM3 的分割质量，需一次性离线预处理；若数据集的分割不佳或存在偏差，可能导致标签重加权效果下降。

---

## 421. The Right Measure for Physics-Constrained Generation: A Co-Area Correction for Posterior-Consistent PDE Inverse Problems

**arXiv ID:** 2606.04804 | [PDF](https://arxiv.org/pdf/2606.04804v1)

**作者:** Jian Xu `[一作]` (RIKEN iTHEMS), Qibin Zhao `[通讯]` (RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本研究分析了在PDE逆问题中使用生成模型（扩散/流匹配）强制硬物理约束时产生的测度偏差，证明缺少共面积（Fixman）Jacobian 会导致采样到错误的后验分布，并提出了基于共面积校正的 CoCoS 采样器及其可扩展的 CoCo-Flow 版本；

**💡 创新点**

创新点在于：1）首次用测度论精确表述硬约束采样所对应的零测度极限并量化其偏差；2）揭示缺失 Fixman Jacobian 对后验采样的影响；3）提出直接采样正确后验的 CoCoS 方法，并演示其通过单步 ODE（CoCo-Flow）实现的可扩展性；

**🔧 技术方法**

技术手段包括：共面积公式、Fixman 修正、基于测度论的偏差理论、在隐式流形上构造 Metropolis‑Hastings 步骤、条件流匹配训练、i.i.d. 拒绝抽样 arbiter 作为无偏金标、梯度/雅可比计算及其随机估计；

**📊 数据集**

实验数据集涵盖：a) 4 维非线性约束基准；b) 1D Darcy 逆问题（d=8、d=16，三实例）；c) 2D Darcy 逆问题（d=64，单实例）；d) 稳态 Burgers PDE（d=16，单实例）；以及对应的人工生成观测；

**📈 对比分析**

与金标 arbiter、PCFM 投影、引导 Langevin（soft‑guidance）以及软惩罚（γ=0.02）等现有方法比较。CoCoS 的 1‑Wasserstein 距离仅为金标噪声的 2.5×，而投影方法 9×，无 Fixman 21×；在 Darcy 问题中，CoCoS 与金标一致，投影偏差高达 36×，软惩罚 40×。CoCo-Flow 在 amortization 后保持 5–10×噪声，显著快于传统 MCMC；

**⚠️ 局限性**

局限性包括：1）需要在每一步计算约束雅可比与其 log‑det，codimension 大时成本高；2）CoCoS 采用 MCMC，接受率在高维、强约束下可能极低；3）方法假设约束可微且雅可比满秩；4）对弱识别方向的正则化仍需经验调参，且对极端稀疏观测时可能出现数值不稳定。

---

## 422. Fast Cubical Persistent Homology on 2D and 3D Images via Union-Find, Pruning, and Lookup Tables

**arXiv ID:** 2606.04801 | [PDF](https://arxiv.org/pdf/2606.04801v1)

**作者:** Titouan Le Breton `[一作]` (Helmholtz AI), Marie Piraud `[通讯]` (Helmholtz AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了Flash Cubical，一个针对二维和三维图像的V-滤波器上F₂系数的高效计算立方体持久同调实现。

**💡 创新点**

创新点包括利用双性图的并查集求解最高维持久性、局部剪枝去除零持久性和已分类单元，以及预先构造查找表以利用立方体的规则性。

**🔧 技术方法**

采用了双性并查集、零持久性剪枝策略、预计算的查找表、以及必要时的矩阵约简等技术，并用C++实现。

**📊 数据集**

实验使用了合成均匀随机图像、Lena、DIV2K等二维数据，以及Fuel、Bonsai、Aneurysm等三维医学体数据。

**📈 对比分析**

通过与CubicalRipser和GUDHI的比较，Flash Cubical在时间和内存上均优于它们，尤其在三维数据上优势显著；在二维T-滤波下则被GUDHI超越。

**⚠️ 局限性**

局限性包括仅支持V-滤波器、三维H₁仍需矩阵约简、对GPU并行或多线程支持有限，以及在二维T-滤波下的性能不及GUDHI。

---

## 423. UModel: An Agent-Ready Observability Data Modeling Method at Scale

**arXiv ID:** 2606.04799 | [PDF](https://arxiv.org/pdf/2606.04799v1)

**作者:** Changhua Pei `[一作]` (CNIC, CAS), Dan Pei `[通讯]` (Tsinghua University)

**通讯引用:** 10003 | [OpenAlex ID](https://openalex.org/A5046419834)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出UModel统一的本体框架，将观测数据从数据中心化转为对象中心化，并提供U-SPL查询接口，使LLM代理能自动探索系统拓扑并关联多模态数据。

**💡 创新点**

通过构建虚拟本体层，将异构遥测、实体和专家知识标准化为对象，并用语义图互联，显著提升根因定位精度；同时开发U-SPL使代理能自主查询，突破传统数据孤岛的局限。

**🔧 技术方法**

本体建模、语义图、基于流水线的查询接口U-SPL、LLM代理支持。

**📊 数据集**

AIOps 2025 Challenge 数据集。

**📈 对比分析**

与原始数据模型比较，根因定位精度提升 8%；在阿里云生产环境部署，支持数万人用户、百万次操作/秒、子秒级查询延迟。

**⚠️ 局限性**

仍需人工本体维护、对动态架构的适应性有限，且对其他业务场景的迁移性尚未充分验证。

---

## 424. NextMotionQA: Benchmarking and Judging Human Motion Understanding with Vision-Language Models

**arXiv ID:** 2606.04773 | [PDF](https://arxiv.org/pdf/2606.04773v1)

**作者:** Yong Cao `[一作]` (University of Tübingen), Andreas Geiger `[通讯]` (University of Tübingen)

**通讯引用:** 52448 | [OpenAlex ID](https://openalex.org/A5016606943)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个3×3×3的人类动作理解基准（NextMotionQA），包含多选问答、自由式字幕和精细错误纠正三种任务、身体部位、方向、动作三种语义轴以及易、中、难三级难度。

**💡 创新点**

创新点在于：① 将任务格式、语义轴和难度三维结构化，显式分层难度；② 通过半自动化生成并由专家统一验证，保证标注质量；③ 在同一基准上对VLM进行多任务、多维度评估，并进一步验证VLM作为评判者的可靠性。

**🔧 技术方法**

使用的技术包括：基于Qwen3.6-Plus的双阶段自动化生成管道、专家多人审核流程、VLM（如Qwen、InternVL、LLaVA、GPT-5.4-mini、Gemini-3.1-Flash）进行系统评估和评判者实验。

**📊 数据集**

使用的数据集：AMASS 3D 动作库、BABEL 动作标签、HumanML3D 说明文本，共1,307个专家验证样本。

**📈 对比分析**

方法对比：对12款VLM进行准确率、Jaccard、召回等多指标评估，发现闭源模型占优（Gemini-3.1-Flash 58.44分），开源模型最高为Qwen3.5-27B 49.75分，存在≈8.7分差距；多选任务是强项，字幕任务是瓶颈；方向轴始终表现最差；VLM评判者在粗粒度（κ≈0.70）下与人类一致，但在细粒度（κ≈0.10）下失效。

**⚠️ 局限性**

局限性包括：① 仅使用AMASS分布，缺乏多样交互或手物交互等场景；② 评判者单视角，无法捕捉多视角细节；③ 专家验证导致规模受限，扩展到1万级需主动学习策略。

---

## 425. CADENCE: Predicting Realized MAPF Execution Time Beyond Sum of Costs

**arXiv ID:** 2606.04746 | [PDF](https://arxiv.org/pdf/2606.04746v1)

**作者:** Abhishek S `[一作]` (BuildMachineLabs), Sreeram MV `[通讯]` (BuildMachineLabs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在固定的7×7工作单元上使用七台差速驱动机器人，实验验证 MAPF 计划的 Sum of Costs（SoC）与实际执行时间之间的关联性；

**💡 创新点**

提出了分层预测模型，证明在 SoC 基础上加入“原始运动负载”特征（转弯次数、连续运动次数、起止转移次数）能显著提升对硬件执行时间的预测；

**🔧 技术方法**

采用线性岭回归与混合效应回归相结合的模型，对计划级别和试验级别的数据进行预测；

**📊 数据集**

构建了 15 个场景（Empty、Medium-Random、Bottleneck）共 120 条 MAPF 计划，并在每条计划上执行 4 次，得到 480 次硬件试验；

**📈 对比分析**

通过 5 折场景级留一验证，比较 M0–M3 模型阶梯的 MAE 与 RMSE，结果显示加入原始运动负载后 MAE 下降 48.6%–59.8%，RMSE 下降 44.2%–61.4%，相对 SoC 仅提升显著；

**⚠️ 局限性**

研究仅限于 7 台机器人、7×7 迷宫以及单一“先行-连续”执行器，且仅覆盖三类交互情境，未能充分验证交互协调特征的稳定性，结果不易直接推广至更大规模或不同执行策略。

---

## 426. TIDE: Proactive Multi-Problem Discovery via Template-Guided Iteration

**arXiv ID:** 2606.04743 | [PDF](https://arxiv.org/pdf/2606.04743v1)

**作者:** Soyeong Jeong `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 TIDE 框架，能够主动在未收到用户请求时，从个人工作空间或软件仓库的上下文中发现并解决多个隐藏问题。

**💡 创新点**

创新点在于将思考模板与迭代发现相结合，先用模板抽象常见问题模式，再通过多轮迭代在已有发现的基础上逐步挖掘更多问题，实现覆盖率与精度的双提升。

**🔧 技术方法**

采用大语言模型（GPT‑5 mini、Claude Sonnet 4.5、Gemini 3.5 Flash、Qwen 3.6 Flash）与检索‑推理‑动作生成的流程，辅以结构化的思考模板和多轮迭代机制。

**📊 数据集**

使用了两组自构造数据集：30 个个人工作空间（共 150 个问题）和 20 个软件仓库快照（共 146 个并发 bug），每组均包含 4–6 个问题和相应的候选文档/函数集合。

**📈 对比分析**

通过与单次推理（Single‑Agent）和并行多代理（Multi‑Agent）基线在检索、识别、解决三项指标上对比，TIDE 在所有四种 LLM 后端下均取得显著提升，覆盖率与 F1 分数均明显高于基线。

**⚠️ 局限性**

局限性包括模板一经构建后保持固定，未实现在线更新；迭代过程受固定轮次与候选数量限制；实验仅覆盖两类场景，且对数据隐私与偏见的处理需进一步完善。

---

## 427. NoRA: Evaluating Grounded Reasonableness in Visual First-person Normative Action Reasoning

**arXiv ID:** 2606.04806 | [PDF](https://arxiv.org/pdf/2606.04806v1)

**作者:** Sichao Li `[一作]` (University of Sydney), Seth Lazar `[通讯]` (Australian National University)

**通讯引用:** 1779 | [OpenAlex ID](https://openalex.org/A5062824333)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于第一人称视频的规范性决策基准，要求模型在场景中生成下一个合理动作并用事实-理由-动作支持图显式证明其合理性。

**💡 创新点**

创新点在于：① 摒弃传统的多选题评测，改为从零开始构建合理动作空间；② 采用结构化支持图来评估模型对事实、理由与动作的绑定；③ 设计了三层提示策略（直接、深思、结构化）来探究不同解释需求下的表现；④ 构建了人类核验与LLM验证并行的多层级数据集。

**🔧 技术方法**

主要技术包括：视觉-语言多模态模型（VLM）推理、Transformer‑based语义相似度评估器、结构化支持图重构层、以及三种不同的提示工程策略。

**📊 数据集**

使用了基于Ego4D的1,420条第一人称视频片段，划分为190条人工核验核心集和1,230条LLM验证银集。

**📈 对比分析**

对12种VLM（包括GPT‑5.2/5.1/5.4、Gemini‑3‑Pro/Flash、Qwen3‑VL、Grok等）在三种提示下进行评测；通过动作对齐、事实抓取、支持绑定三分量的几何平均R评分；最佳模型GPT‑5.2/5.1在R≈0.38，GPT‑5.4达到≈0.56；开源模型最高R≈0.34，表明与参考上限仍有显著差距。

**⚠️ 局限性**

局限性包括：① 注释样本有限，难以覆盖所有合理动作空间；② 评估指标对实例尺度敏感，难以直接解读为绝对规范性；③ 数据集可能带来视频来源、注释者与文化背景的偏差，需要在使用时注意解释与上下文验证。

---

## 428. BEATS: Bootstrapping E-commerce Attribute Taxonomies for Search through Iterative Human-AI Collaboration

**arXiv ID:** 2606.04909 | [PDF](https://arxiv.org/pdf/2606.04909v1)

**作者:** Yung-Yu Shih `[一作]` (National Taiwan University), Yun-Nung Chen `[通讯]` (National Taiwan University)

**通讯引用:** 3958 | [OpenAlex ID](https://openalex.org/A5076610826)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出BEATS框架，利用多阶段LLM与人机协作从零构建电商属性目录并对产品进行属性标注

**💡 创新点**

1）多源LLM生成+合成+精炼的多阶段流程；2）主动质量检查+人类专家验证的两阶段质量保障；3）基于反馈的迭代Prompt优化

**🔧 技术方法**

大型多模态LLM（GPT‑OSS‑120B、Qwen3‑30B、Qwen3‑VL‑235B）进行属性生成、合成、精炼；人类专家进行评注；后续使用LLM进行属性标注及LLM‑judge评估；密集检索模型进行性能验证

**📊 数据集**

Rakuten Taiwan 商品目录（9大类、2694子类、5.4M 商品），使用传统中文标题/描述；参考日本 Rakuten Ichiba 属性作为少量示例

**📈 对比分析**

与仅使用原始商品文本的检索模型对比；在密集检索上 Recall@10、Recall@100、NDCG@10、MRR@10 均提升（约+5%）

**⚠️ 局限性**

跨市场知识迁移受语言与业务差异限制；LLM生成仍可能产生细粒度冗余或误差，需人工干预；对齐度评价依赖于人工标签，主观性较高

---

## 429. Provably Auditable and Safe LLM Agents from Human-Authored Ontologies

**arXiv ID:** 2606.04903 | [PDF](https://arxiv.org/pdf/2606.04903v1)

**作者:** Aaron Sterling `[一作]` `[通讯]` (Thistleseeds), Aaron Sterling (Thistleseeds)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了 Agentic Redux 架构和 Ontology‑First Agent Design 方法，能够在保留语义安全性和线性可审计性的前提下，将复杂问题域（如医疗账单合规和安全漏洞披露）自动转化为多智能体系统并执行。

**💡 创新点**

创新点在于：①将 typed λ‑calculus 与 BFO 计算本体结合，提供形式化的安全保证；②通过“元智能体”统一决策，消除写倾斜（Write Skew）；③提供半自动化的本体驱动智能体设计流程。

**🔧 技术方法**

采用技术包括：Typed λ‑calculus 形式化证明、BFO 计算本体、LLM 推断角色与 invariants、Agentic Redux 运行时、日志追加式 ledger 记录、以及在代码仓库中的实现。

**📊 数据集**

主要使用的“数据集”为手工构建的 BFO 本体和域模块（如 UDT 合规、漏洞披露），以及对应的代码实现；未使用公开大规模训练数据。

**📈 对比分析**

比较方法基于形式化证明（Invariant Preservation、Audit Log Integrity）而非实验性能，理论上保证所有全局状态满足 invariants，日志按时间线性增长、每次决策只产生一条条目；实验性能未给出。

**⚠️ 局限性**

局限性包括：需要人工专家完成本体构建；对现实世界软件缺陷仍需防御层；人机交互（counselor）可能破坏 invariants；未覆盖 liveness、性能指标和保密性等问题。

---

## 430. Channel Fracture: Architectural Blind Spots in Scheduled Cross-Agent Memory Injection for Multi-Agent Orchestration Systems

**arXiv ID:** 2606.04896 | [PDF](https://arxiv.org/pdf/2606.04896v1)

**作者:** Levent Liu `[一作]` `[通讯]` (Shanghai Qijing Digital Technology Co Ltd), Levent Liu (Shanghai Qijing Digital Technology Co Ltd)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

识别并研究了多代理AI编排系统中的通道断裂问题，并在Hermes Agent生产环境中验证了三种知识注入通道的表现，提出了CADVP v1.1验证协议和两个设计原则。

**💡 创新点**

提出了通道断裂（channel fracture）概念、13维CADVP v1.1协议及其阈值级确认，以及逆向验证与通道匹配两条设计原则。

**🔧 技术方法**

Hermes Agent架构、SQLite、FTS5全文搜索、定时任务调度、工具注册机制、跨代理交互、验证协议实现。

**📊 数据集**

使用Hermes Agent生产系统的五个代理配置及其日志数据作为实验数据集。

**📈 对比分析**

通过对三种注入通道的实验对比，直接数据库写入和目标自写成功，定时任务注入失败；CADVP v1.1能够及时检测并阻止失败通道；性能主要体现在成功率上，无显著资源开销。

**⚠️ 局限性**

研究仅基于单一生产部署，缺乏跨框架验证，CADVP协议尚未在其他多代理系统中验证，通道断裂可能因系统实现差异而不完全相同。

---

## 431. Welfare Maximization in Bilateral Trade: Improved Approximation Guarantees Beyond the Fixed Price Barrier

**arXiv ID:** 2606.04890 | [PDF](https://arxiv.org/pdf/2606.04890v1)

**作者:** Shahar Dobzinski `[一作]` (Weizmann Institute of Science), Ariel Shaulker `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种在双边贸易中通过买家发起报价并设定保留价的单边主导策略机制，以实现更高的社会福利近似比例；

**💡 创新点**

通过构造“交易权衡引理”和“邻域引理”证明，存在一个保留价能够在任何买方价值下至少获得 0.71 的社会福利比例；并在考虑买方分布时进一步提升至超过 0.7381 的整体近似比，突破固定价格机制的理论极限；

**🔧 技术方法**

主要使用分析工具——交易权衡引理、邻域引理，构建线性规划（LP）和其对偶来求取保留价的下界；利用对数函数和凸性分析来推导保留价与近似比例之间的关系；

**📊 数据集**

本工作为理论研究，未使用具体数据集，而是对任意满足支持在 [0,1] 的卖方和买方分布进行泛化分析；

**📈 对比分析**

与传统固定价格机制（已知的最佳近似比约为 0.72–0.7381）比较，提出的买家报价保留机制在期望社会福利上至少可达到 0.7381 以上，显著优于固定价格机制；

**⚠️ 局限性**

限制在于该机制的近似比仍低于 1，且理论证明依赖于连续分布假设和对数界的计算；实际实现中需估计卖方分布并选择合适保留价，且在某些极端分布下仍可能只能获得约 0.7159 的局部近似比。

---

## 432. D$^3$-MoE:Dual Disentangled Diffusion Mixture-of-Experts for Style-Controllable End-to-End Autonomous Driving

**arXiv ID:** 2606.04884 | [PDF](https://arxiv.org/pdf/2606.04884v1)

**作者:** Renju Feng `[一作]` (Wuhan University of Technology), Duanfeng Chu `[通讯]` (Wuhan University of Technology)

**通讯引用:** 1568 | [OpenAlex ID](https://openalex.org/A5041423958)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了 D^3-MoE 框架，实现了行为与物理双重解耦的风格可控轨迹规划。

**💡 创新点**

创新点在于：①将轨迹分为横纵两轴进行解耦；②使用动态路由无监督训练的 Mixture-of-Experts；③在每个专家内部采用 Diffusion Transformer 与 AdaLN 进行风格条件化；④通过 Best-of-Three 集成进一步提升多模态覆盖率。

**🔧 技术方法**

采用的核心技术包括 Mixture-of-Experts、Diffusion Transformer、AdaLN、横纵分离路由、self-supervised K-means 路由监督以及风格条件化的异向跨注意力。

**📊 数据集**

主要在 NAVSIM 数据集上进行训练与评估，使用 NAVtest 进行对比测试。

**📈 对比分析**

与多种基准方法（如 DiffusionDrive、ARTEMIS、Transfuser 等）在 Navtest 上对比，默认风格下 PDMS 88.2、EPDMS 84.3，Best-of-Three 方案 PDMS 91.3、EPDMS 87.5，均显著优于现有方法。

**⚠️ 局限性**

局限性包括：①需要多步扩散推理导致推理延迟；②依赖手工构造的三种风格样本；③对极端稀有场景的路由稳定性仍待提升；④在极低算力设备上的部署仍具挑战。

---

## 433. Prediction Under Imperfect Compression: A Theory of Approximate MDL

**arXiv ID:** 2606.04834 | [PDF](https://arxiv.org/pdf/2606.04834v1)

**作者:** Qian Li `[一作]` (Shenzhen Research Institute of Big Data), Guangxu Yang `[通讯]` (University of Southern California)

**通讯引用:** 40900 | [OpenAlex ID](https://openalex.org/A5100320478)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在近似最小化最小描述长度（MDL）目标时，如何仍能保证可靠的顺序预测，特别是在不同的近似方式和正则化强度下的表现。

**💡 创新点**

提出了一个精确的阈值判定：当正则化参数 λ≥1 且使用固定加性近似时，任何满足近似条件的 MDL 选择器都能保证累计平方误差有限；而 λ<1 时存在过拟合构造导致误差无穷；同时证明乘性近似在任何 λ>0 下都无法提供统一的预测保证。

**🔧 技术方法**

使用信息理论工具（如 Kolmogorov 复杂度、两部分编码、似然比阈值、亲和度-收敛界、停机论证）以及概率论方法（如马尔科夫不等式、几率论界）构建理论证明。

**📊 数据集**

无实际数据集，全部为理论分析与构造对抗例子；在通用可估计概率模型类中构造具体例子。

**📈 对比分析**

没有实验比较；通过构造对抗例子和上界/下界证明，展示了在不同 λ 与近似方式下的性能差异：加性近似在 λ≥1 时性能有限，而乘性近似则在任何 λ 下均可能失效。

**⚠️ 局限性**

局限在于研究仅覆盖可实现的顺序预测框架和可估计模型类；未考虑模型误差、非可估计环境以及更复杂的近似度量；理论结果可能对实际机器学习实践的数值优化难以直接转化。

---

## 434. Worker Utility as Hysteresis: A Preisach Model of Transaction Acceptance in Gig Labour Markets

**arXiv ID:** 2606.04916 | [PDF](https://arxiv.org/pdf/2606.04916v1)

**作者:** Piotr Frydrych `[一作]` (Warsaw University of Technology), Piotr Frydrych `[通讯]` (Warsaw University of Technology)

**通讯引用:** 110 | [OpenAlex ID](https://openalex.org/A5012740592)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发并验证了基于Preisach磁滞模型的双阈值工人效用估计与价格推荐框架，用于零工平台的接受决策预测。

**💡 创新点**

将Preisach模型引入劳动市场，提出通过双输出神经网络估计接受/拒绝的条件均值效用，并利用间隙与价格相对阈值编码，捕捉工人历史依赖与价格非对称性；同时构建基于该模型的工资推荐机制。

**🔧 技术方法**

采用共享编码器的双输出神经网络（含边际损失）估计U₁、U₀；利用价格相对阈值（r₀,r₁,d₀,d₁,gap,lps）等clip‑stabilised特征，构建XGBoost分类器；通过AUC、Jaccard等指标进行评估。

**📊 数据集**

36,891条波兰零工平台交易记录（2024年控制价格实验期间），包含工资、竞争密度、时间等变量。

**📈 对比分析**

与单独XGBoost回归估计U₁、U₀相比，双输出网络在NRMSE、AUC、Jaccard上略优；最终推荐方案可将工资账单降21.3%并提高填补率9.7个百分点；特征编码提升AUC约11个百分点。

**⚠️ 局限性**

价格变动为观察性非随机实验，导致价格弹性识别受限；模型仅估计阈值的第一矩，无法完全恢复阈值密度；未检验不同工种间的混合分布和动态路径依赖的完整性。

---

## 435. BreastGPT: A Multimodal Large Language Model for the Full Spectrum of Breast Cancer Clinical Routine

**arXiv ID:** 2606.04911 | [PDF](https://arxiv.org/pdf/2606.04911v1)

**作者:** Yang Liu `[一作]` (DAMO Academy, Alibaba Group), Yingda Xia `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了以乳腺癌筛查、诊断、治疗规划三阶段为导向的多模态指令数据集 BreastStage，并提出统一的多模态大语言模型 BreastGPT，能够在不同模态下实现跨阶段推理与决策；

**💡 创新点**

①基于临床工作流的跨阶段任务划分与大规模数据集；②双分支视觉编码器与分辨率感知门控，统一处理标准影像与全景切片；③基于概念覆盖的无训练视觉 token 选择器，兼顾准确性与推理效率；

**🔧 技术方法**

采用 Qwen3‑VL 变体作为 LLM，标准分支 ViT、Gigapixel 分支 CONCH+LongNet，MMTok 风格概念覆盖选择器，以及阶段条件系统提示实现任务路由；

**📊 数据集**

使用 BreastStage，整合 17 个子数据集、5 种模态（钼测、乳腺超声、MRI、CT、WSI），共 1.86M QA 对，随后划分为 BreastStage‑Bench 评估集；

**📈 对比分析**

在 BreastStage‑Bench 上与 GPT‑5.4、Claude、Gemini 等专有模型、开源通用 VLM 以及医疗专用 VLM 进行零样本对比，BreastGPT 在闭合 VQA 平均 75.66%（比 GPT‑5.4 高 21.66）和开放式 VQA 89.92%（比最高 25+ 点）表现突出，生成、定位等任务也显著优于所有基线；

**⚠️ 局限性**

训练样本多来自不同病人，缺乏完整的纵向患者记录，导致模型无法充分学习同一患者跨阶段的时间连续性；同时，对极大尺度 WSI 的压缩仍可能丢失部分细节信息，需进一步验证与改进。

---

## 436. 'Your AI Text is not Mine': Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions

**arXiv ID:** 2606.04906 | [PDF](https://arxiv.org/pdf/2606.04906v1)

**作者:** Nils Dycke `[一作]`, Iryna Gurevych `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统定义并统一了AI生成文本检测（AITD）的多种概念，提出了基于内容和作者身份的新检测框架，并构建了首个不偏向特定概念、真实人机共写的AITDNA数据集；

**💡 创新点**

创新点在于：①把AITD拆分为基于“genesis”（文本生成来源）和基于“population”（人类文本分布）的两大类，并引入了内容和作者身份两种新检测概念；②设计并发布了具有详细生成历史与交互日志的AITDNA数据集；③在同一数据集上系统评估不同检测器与概念的交互影响，揭示阈值设定和概念粒度对检测性能的决定性作用；

**🔧 技术方法**

采用的技术包括：①对人机共写过程进行细粒度日志记录，生成token级别的“genesis”信息；②使用多种公开及专有检测器（Min‑K、Likelihood、Log Rank、Binoculars、FastDetectGPT、GPTZero、Pangram、moBERT）进行实验；③在不同概念下统一划分文本段并评估AUROC、F1、FPR等指标；

**📊 数据集**

使用的数据集：①AitDNA（约362条文本，覆盖论文写作、论证、创作、科学写作场景，含多种LLM配置）；②对比的现有数据集CoAuthor、SenDetEx、BD、Mixset、DetectRL等，分析其隐含的概念与阈值；

**📈 对比分析**

比较方法：在统一的AITDNA数据集上，针对多种检测器和六种检测概念（文档级、边界级、句子级、内容基、意图基、基于成员资格）分别计算AUROC、F1和FPR；实验结果显示：文档级检测最易通过，细粒度检测难度增加；内容/意图检测对现有模型最具挑战；不同数据集上最佳检测器不一致，阈值τ的选择对F1与FPR有显著影响；

**⚠️ 局限性**

局限性：①AITDNA在实验室环境下收集，缺少真实聊天交互模式；②作者身份基检测因样本量不足难以验证；③所用检测器多为现有模型，未针对不同概念进行专门训练；未来需在更大规模、更多交互场景下收集数据，并研发能够同时兼顾多种概念的鲁棒检测器。

---

## 437. Hierarchical Space Partition for Surface Reconstruction

**arXiv ID:** 2606.04891 | [PDF](https://arxiv.org/pdf/2606.04891v1)

**作者:** Minjie Tang `[一作]` (Independent Researcher), Xiangfei Li `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 14469 | [OpenAlex ID](https://openalex.org/A5041607266)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于层次空间分割的多平面组装算法，用于从稀疏点云中生成紧凑、闭合的多边形网格。

**💡 创新点**

创新点在于：①将检测到的平面按可见度分为三层（高可见、低可见、隐藏），并在层次上按可见度加速扩张；②利用边界段和交线检测孤立段来恢复缺失平面；③在扩张过程中使用动力学数据结构和状态机实现高效碰撞处理。

**🔧 技术方法**

技术手段包括RANSAC平面检测、α‑shape边界逼近、MRF可见度优化、奇异段筛选与区域增长拟平面、动力学框架的多层次平面扩张、最小割求解内外标签、CGAL实现。

**📊 数据集**

实验数据集涵盖Assembly、KSR‑42、ScanNet++ V2，分别取200个CAD模型（CAD‑200）与100个实景建筑模型（Arch‑100）进行评测。

**📈 对比分析**

与QEM、PolyFit、KSR、RLPM、VecIM等方法对比，本文在Mean Hausdorff Error、RMSE、面数/点数比率等指标均取得最低误差和最佳紧凑度，虽然运行时间略高于KSR但远快于PolyFit，内存峰值亦低于对比方法。

**⚠️ 局限性**

局限性包括：对极度稀疏或严重遮挡点云时仍可能漏检隐藏平面；缺失平面恢复依赖奇异段的检测，可能受噪声影响；目前仅利用点云信息，未结合图像纹理或语义。

---

## 438. HD-DinoMoE: A Class-Aware Hierarchical Dual Mixture-of-Experts Network for Scleral Anomaly Segmentation in Complex Acquisition Scenarios

**arXiv ID:** 2606.04888 | [PDF](https://arxiv.org/pdf/2606.04888v1)

**作者:** Yinxiang Yu `[一作]` (University of Science and Technology Liaoning), Guanghao Liao `[通讯]` (University of Science and Technology Liaoning)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 HD‑DinoMoE 网络，用于在复杂采集场景下对虹膜表面异常进行像素级多标签分割，并构建了新的多源数据集 ML‑SASD。

**💡 创新点**

创新点包括：class‑aware 层次双流门控编码器（CA‑DSGF）、类特定多专家解码器（CS‑MED）、Progressive Confidence Penalty Loss（PCP Loss）与 Class‑Aware Adaptive Sample Weighting（CA‑ASW）等多级自适应机制，及对多源数据的三阶段骨干冻结对齐策略。

**🔧 技术方法**

技术手段涵盖：DINOv3 ViT‑L 视觉基础模型、Mixture‑of‑Experts 解码器、双流门控特征融合、渐进置信惩罚损失、样本‑类别级自适应加权、三阶段骨干冻结训练策略、SSR 负样本建模等。

**📊 数据集**

使用的数据集为自建的 Multi‑Label Scleral Anomaly Segmentation Dataset（ML‑SASD，包含 Clinical、Wild、Mix 三个子集），并与公开 SBVPI 数据集进行跨域评估。

**📈 对比分析**

与 11 种基线（U‑Net、nnU‑Net、TransUNet、SegNeXt、U‑Mamba、U‑KAN、SegDINO‑SAT/LVD、DINOUNet‑SAT/LVD、UTANet、ConDSeg）进行比较，HD‑DinoMoE 在 ML‑SASD‑Mix 上取得最高 mDice 72.11%、mIoU 58.44%、mBF1 41.40% 并将 mGFPR 降至 1.02%，显著优于其它方法。

**⚠️ 局限性**

局限性包括：模型参数量大、训练与推理成本高；缺乏大规模临床验证；在极小尺寸或高噪声图像上的鲁棒性仍有提升空间；以及对不同设备、光照条件的进一步泛化仍需探索。

---

## 439. Abduction Prover in Isabelle/HOL

**arXiv ID:** 2606.04877 | [PDF](https://arxiv.org/pdf/2606.04877v1)

**作者:** Yutaka Nagashima `[一作]` (Institute of Computer Science Czech Academy of Sciences), Daniel Sebastian Goc `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 AbductionProver，一个基于递归归谬推理（abductive reasoning）和 AND‑OR 图的 Isabelle/HOL 证明搜索框架，能够自动生成证明脚本。

**💡 创新点**

创新点在于把 tactic 的应用和显式猜想统一为 Modus Ponens 的实例；利用 AND‑OR 图实现子证明共享、循环依赖的管理，并在完成根节点时自动抽取无环的解图；同时结合模板与变异方法产生猜想，并通过一阶证明与反例检验进行过滤。

**🔧 技术方法**

使用的技术包括：PSL（Proof Strategy Language）实现局部搜索、E / QuickCheck 反例生成器进行猜想筛选、模板/变异猜想生成、α/简化归一化、递归式成功证明导向的猜想集合识别、以及循环图与解图的状态传播逻辑。

**📊 数据集**

评估主要在 Isabelle/HOL 公式库（Archive of Formal Proofs）和作者自建的示例中进行；未公开具体公开数据集。

**📈 对比分析**

方法通过与传统手工证明或单纯 tactic 搜索对比，展示了在同一证明目标下自动生成完整证明脚本的能力；论文中未给出定量性能指标，但表明在示例案例中显著降低了手工步骤。

**⚠️ 局限性**

限制包括：搜索空间随猜想数目呈指数增长，需大量过滤；目前不依赖机器学习，缺乏更智能的猜想选择；循环图的处理仍需手工设置阈值；实现仅针对 Isabelle/HOL，迁移到其他 ITP 需要进一步工作。

---

## 440. Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents

**arXiv ID:** 2606.04874 | [PDF](https://arxiv.org/pdf/2606.04874v1)

**作者:** Haoyu Sun `[一作]` (Tongji University), Yu Cheng `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Agent Planning Benchmark (APB)，为大语言模型代理的规划能力提供了多层次、多模态的诊断性评测框架。

**💡 创新点**

创新点包括：①在规划层面构建细粒度评测任务（整体规划、步进规划、工具噪声与不可解任务）；②引入六类错误分类法与 LLM-as-Judge 自动评判；③通过 APB 指导的计划修订提升执行性能，并验证其对执行指标的正向影响。

**🔧 技术方法**

技术方法主要涉及 LLM 生成与验证流水线、基于 LLM 的数据合成与过滤、推理时的自我/批评式修订策略以及多模态 LLM 的任务执行与评估。

**📊 数据集**

数据集由 4,209 条多模态任务组成，涵盖 22 个领域，来源于 OpenCUA、GTA、GAIA、ToolBench、FrameThinker 以及公开 AI 代理平台的真实交互日志。

**📈 对比分析**

评测结果对比 12 种 MLLM（包括 GPT‑5、Claude‑Sonnet、Gemini 系列、Qwen3‑VL、InternVL3.5 等），发现专有模型在整体规划上占优，但在工具噪声与不可解任务上表现下降；APB‑引导的计划修订显著提升 ToolSandbox 与 τ²‑Bench 的轨迹相似度与奖励，且顶尖模型已展现出成本优化倾向。

**⚠️ 局限性**

局限性包括：①评测覆盖仍不涵盖所有真实任务多样性，尤其是专业垂直领域；②仅聚焦规划，无法替代完整端到端评测；③评判与数据合成部分依赖专有模型，可能影响可复现性和公开可用性。

---

## 441. Uncertainty-Aware End-to-End Co-Design of Neural Network Processors: From Training and Mapping to Fabrication

**arXiv ID:** 2606.04850 | [PDF](https://arxiv.org/pdf/2606.04850v1)

**作者:** Yuyang Du `[一作]` (Nanyang Technological University), Gioele Zardini `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5043524649)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个基于单调共设计理论的统一端到端神经网络处理器协同设计框架。

**💡 创新点**

创新点在于将不确定性显式建模为可调节资源（Confidence），实现模块化接口与离线代理，使单块改进无需重构整体图。

**🔧 技术方法**

采用单调共设计理论、分布式不确定性扩展、Gaussian 学习曲线、三分量混合分布代理及负二项式缺陷模型等技术。

**📊 数据集**

使用 HW‑NAS‑Bench 训练轨迹、MAESTRO 仿真数据以及 65nm、45nm、28nm 工艺节点相关数据集。

**📈 对比分析**

通过三大案例（基准、跨场景、多块改进）验证，得到功耗‑成本、时间‑置信度等 Pareto 前沿，展示在保证精度与可靠性前提下实现资源最优配置。

**⚠️ 局限性**

局限性包括代理模型误差、随机性独立性假设、仅覆盖特定加速器架构、对工艺变异的简化建模，以及未与完整 EDA 链路集成。

---

## 442. 3D Temporal Analysis for Autism Spectrum Disorder Screening During Attention Tasks

**arXiv ID:** 2606.04836 | [PDF](https://arxiv.org/pdf/2606.04836v1)

**作者:** Inam Qadir `[一作]` (Hamad Bin Khalifa University), Marwa Qaraqe `[通讯]` (Hamad Bin Khalifa University)

**通讯引用:** 2431 | [OpenAlex ID](https://openalex.org/A5010196813)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于DECA的3D时序分析框架，用于在虚拟现实注意力测试中对7-12岁儿童进行自闭症谱系障碍筛查。

**💡 创新点**

创新点在于首次将3D头部平移向量与姿态无关的表情参数结合，使用LSTM/GRU捕捉时序行为模式，并通过PCA融合多模态特征显著提升分类准确率。

**🔧 技术方法**

技术包括DECA3D建模、FLAME参数回归、ResNet-50特征编码、LSTM/GRU时序网络、PCA降维与多模态特征融合。

**📊 数据集**

使用了39名儿童（19自闭症，20典型发育）的VR‑CPT视频数据，年龄7–12岁，录制频率30fps。

**📈 对比分析**

与传统2D表情与头姿势特征对比，3D头姿势模型在GRU上达到83.9%准确率，3D表情模型81.4%；多模态融合模型在LSTM上实现84.6%准确率，分别比2D方法提升10.7%和7.5%。

**⚠️ 局限性**

主要局限包括样本量小（39例）、自闭症组性别失衡（17男/2女），以及仅在单一任务环境下验证，影响结果的泛化性。

---

## 443. Dark Path: An Analysis of the Belt & Road Initiative in El Salvador

**arXiv ID:** 2606.04832 | [PDF](https://arxiv.org/pdf/2606.04832v1)

**作者:** Adam Dorian Wong `[一作]` (Dakota State University), David Kenley `[通讯]` (Dakota State University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了中国“一带一路”倡议对萨尔瓦多网络供应链与国家安全的影响，并与当地法律框架进行对照分析。

**💡 创新点**

首次提出“影响性潜伏”概念，系统评估BRI项目在拉美国家的数字主权风险与法律缺口。

**🔧 技术方法**

采用生态探索性定性方法，结合文本分析、事件跟踪与威胁情报解析。

**📊 数据集**

使用William & Mary AidData、X（推特）社交媒体数据、新闻报道、政府法律文本以及行业威胁情报报告。

**📈 对比分析**

通过多源三角验证与时间线对比，对比萨尔瓦多网络安全法规与中国供应链影响，揭示法规空白导致风险聚集，但未进行量化性能评估。

**⚠️ 局限性**

受限于非概率性样本、数据时效受限、缺乏实验验证、适用性局限于萨尔瓦多、未能实证证明捐赠设备的安全状态。

---

## 444. M3imic: Learning a Versatile Whole-Body Controller for Multimodal Motion Mimicking

**arXiv ID:** 2606.04829 | [PDF](https://arxiv.org/pdf/2606.04829v1)

**作者:** Zuxing Lu `[一作]` (Southeast University), Shengbo Eben Li `[通讯]` (Tsinghua University)

**通讯引用:** 20150 | [OpenAlex ID](https://openalex.org/A5100747108)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种多模态全身控制框架M3imic，能够统一处理机器人关节角度、人类姿态轨迹以及末端执行器轨迹三种不同模态的运动参考，实现一次性训练后即可在不同模态下直接部署；

**💡 创新点**

创新点在于：①利用模态特定编码器将三种模态映射到共享潜在空间，消除模态差异并保持跨模态一致性；②引入自适应采样的课程学习策略，动态聚焦难度较高的动作片段；③一次性训练单一策略，避免多阶段知识蒸馏带来的信息丢失与收敛困难；

**🔧 技术方法**

技术包括强化学习（PPO/RA-D），自编码器潜在空间学习，异步演员-评论家结构，课程学习（failure-rate基自适应采样），域随机化（sim-to-real），以及多模态专用的前馈网络；

**📊 数据集**

主要使用的公开数据集有LAFAN1、100STYLE（训练）和OMOMO（测试）；同时利用SMPL-X模型对人类动作进行重映射；训练平台为IsaacSim；

**📈 对比分析**

与HOVER、ExBody2、OmniH2O、TWIST2等基线进行对比，M3imic在训练集上成功率最高（99.54%），在未见测试集OMOMO上终止成功率达98.42%，并在多模态下均表现出优越的误差（E_mpkpe≈46mm、E_mpjae≈0.11rad、E_vel≈0.26m/s），相对基线均有10–30%的性能提升；

**⚠️ 局限性**

局限性包括：①不同模态之间存在“精度-鲁棒性”权衡，密集关节模态精度高但鲁棒性弱；②模型容量与推理速度之间存在折衷，较大网络会降低实时性；③对外部传感器噪声和真实世界动态分布的鲁棒性仍需进一步提升，主要依赖于更多多样化的数据和更高效的域随机化策略。

---

## 445. Optimizing the Cost-Quality Tradeoff of Agentic Theorem Provers in Lean

**arXiv ID:** 2606.04883 | [PDF](https://arxiv.org/pdf/2606.04883v1)

**作者:** Kári Rögnvaldsson `[一作]` (ETH Zurich), Martin Vechev `[通讯]` (ETH Zurich)

**通讯引用:** 11383 | [OpenAlex ID](https://openalex.org/A5069901599)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于失败证明轨迹的动作路由代理，动态决定是否继续尝试证明当前目标或重新分解问题，以优化Lean证明的成本与质量平衡。

**💡 创新点**

创新点在于利用失败轨迹估计下一次尝试的成本与成功概率，构建成本‑质量决策器替代固定预算；将证明生成与路由分离为数据平面和控制平面；在PutnamBench子集上验证该方法的有效性。

**🔧 技术方法**

使用大型语言模型（LLM）与Lean编译器进行证明生成；实现成本估计（基于输出token）和质量估计（逻辑回归模型，特征包括证明相似度、错误多样性等）；采用路由与级联技术；通过训练得到动态路由策略。

**📊 数据集**

使用PutnamBench的85道题目子集（42道用于训练，43道用于测试），该子集为Lean证明任务提供标准基准。

**📈 对比分析**

与固定步长基线和无噪声oracle路由进行对比；在相同预算下，动态路由提高7.8%的准确率；在相同准确率下，成本下降28.4%；AUC提升25.8%；通过噪声、特征消除和数据平面对比的消融实验进一步验证其优越性。

**⚠️ 局限性**

局限性包括：评估仅限于小规模数据集，缺乏对完整PutnamBench的验证；与oracle路由仍存在显著差距，说明估计器仍未捕捉全部信息；动作空间仅为“尝试/终止”，未考虑模型切换或自我纠错等更复杂操作；成本估计仅以token为代理，未考虑其他资源消耗。

---

## 446. GNStor: Design of GPU-Native High-Performance Remote All-Flash Array

**arXiv ID:** 2606.04908 | [PDF](https://arxiv.org/pdf/2606.04908v1)

**作者:** Shushu Yi `[一作]` (Peking University), Jie Zhang `[通讯]` (Peking University)

**通讯引用:** 59079 | [OpenAlex ID](https://openalex.org/A5100449651)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了GNStor，一个GPU原生的全闪存阵列（AFA）系统，消除了I/O路径中的CPU干预，提升GPU到远程AFA的访问性能。

**💡 创新点**

创新点在于：①推出GPU‑centric NVMe over RDMA（GNoR）软件栈，利用原子操作和GPU友好的多级内存分配器实现高并发远程SSD访问；②实现去中心化AFA引擎（deEngine），将访问控制和元数据维护迁移至SSD固件，消除集中式CPU引擎导致的同步瓶颈。

**🔧 技术方法**

采用RDMA/NVMe‑over‑RDMA协议、GPU SIMT并行模型、原子操作同步、GPU友好多级bitmap分配器、SSD固件定制（deEngine）、CUDA 12.9、DOCA 3.3、SPDK以及NVMe Virt模拟器。

**📊 数据集**

使用GPT‑2训练语料、GAP图数据集、ImageNet‑100图像数据、以及人工生成的大规模向量/矩阵数据集进行评估。

**📈 对比分析**

通过微基准（4KB/64KB随机/顺序读写）、多客户端扩展、多SSD扩容、以及真实工作负载（向量加法、矩阵乘法、图分析、LLM训练）与基线CPU‑centric系统、GPUDirect CPU‑centric系统、以及单独的deEngine/GNoR进行对比；结果显示GNStor平均吞吐量提升3.2×，单个工作负载执行时间缩短31.1%，在多客户端场景下可达网络带宽的99.5%。

**⚠️ 局限性**

主要限制是需要在SSD固件中实现deEngine，尚未在商用SSD上易部署；GNoR目前仅针对NVIDIA CUDA/DOCa架构，迁移到其他GPU/NIC需额外工作；此外仍受网络带宽和SSD数量限制，且高并发下可能出现内存分配碎片等细节挑战。

---

## 447. CDPM-Align: Multi-Scale Guidance-Aligned Diffusion Pretraining for Robust Few-Shot Anatomical Landmark Detection

**arXiv ID:** 2606.04898 | [PDF](https://arxiv.org/pdf/2606.04898v1)

**作者:** Roberto Di Via `[一作]` (University of Genoa), Vito Paolo Pastore `[通讯]` (University of Genoa)

**通讯引用:** 802 | [OpenAlex ID](https://openalex.org/A5073140566)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在极少标注条件下，利用多尺度引导对齐的条件扩散预训练，为 X 光影像中的解剖标记检测提供了更稳健的特征提取与预测框架。

**💡 创新点**

创新点在于将数据集索引作为条件标签，结合 classifier‑free 引导差分信号，并在不同 UNet 层级与扩散时间步上施加方向一致性约束，从而实现对结构信息的跨噪声级别对齐。

**🔧 技术方法**

核心技术包括条件扩散概率模型（CDPM）、多尺度指导对齐损失、FiLM 编码、UNet 架构以及投影头对齐实现；最终以像素级分类头完成标记预测。

**📊 数据集**

使用的公开 X 光数据集包括深圳胸腔 X 光（Shenzhen）、ISBI2015 侧位颅面图像（ISBI2015）和数字手部图谱（DHA）。

**📈 对比分析**

与 ImageNet 预训练的 UNet、DINO/MoCo/SimCLR 自监督模型、无条件扩散预训练及统一多解剖检测器进行对比，实验显示在 10–25 shot 场景下 CDPM‑Align 在 MRE、ERE、SDR@2、P95 等指标上均显著优于对照组，尤其在异构数据集上的鲁棒性更佳。

**⚠️ 局限性**

局限性包括仅评估 2D X 光图像、对齐阶段计算开销略高（约 1.3 倍），且未验证在体素或多模态医学影像上的可扩展性。

---

## 448. ODYSSEY: Reestablishing Confidentiality in Confidential Blockchain via Delegated Execution

**arXiv ID:** 2606.04892 | [PDF](https://arxiv.org/pdf/2606.04892v1)

**作者:** Ju Yang `[一作]` (Southern University of Science and Technology), Yinqian Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 5705 | [OpenAlex ID](https://openalex.org/A5070946957)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种基于委托执行和先排序后执行架构的机密联盟区块链（简称 CB），旨在抵御 TEE 的侧信道泄漏和回滚攻击，并通过并发执行与委托失败处理降低执行开销。

**💡 创新点**

创新点包括：① 将委托执行引入机密链，显著缩小攻击面；② 采用先排序后执行架构，天然防止回滚攻击；③ 设计基于状态依赖图的地点感知并发执行与委托失败处理机制，提升吞吐量与系统鲁棒性。

**🔧 技术方法**

技术手段包括：Tee 执行环境（AMD SEV‑SNP）、FISCO‑BCOS 3.0 协议栈、VR 共识协议、状态依赖图（SDG）并发调度、同步与版本控制、基于时间戳的委托失败检测与优化中止。

**📊 数据集**

使用在 AWS EC2 上的虚拟机（c6a.xlarge）构建实验集群，部署 32 个账户进行跨合约平行转账基准，分别在 LAN 与 WAN 环境下测量吞吐量、延迟以及在不同故障阈值（f = 1, 2, …, 30）下的性能。

**📈 对比分析**

与原 FISCO‑BCOS（PBFT 共识）及其 SEV‑SNP 变体（-CS、-ES、BCOS‑S）进行对比。结果显示：在 3 节点 WAN 场景下 CB 达到约 4k TPS、0.4–0.5 s 延迟；与 BCOS 相比吞吐量略低但延迟更稳定；SEV‑SNP 引入显著性能损耗，且委托失败处理能够保持系统 liveness。

**⚠️ 局限性**

局限性：① 扩展性差，随节点数增大吞吐量衰减明显；② 主要针对联盟链，尚未适配无许可链；③ 仍未抵御基于 TEE 克隆的分叉攻击，需后续软硬件协同补救；④ 依赖 SEV‑SNP，受 VM 加密与 I/O 开销限制。

---

## 449. GRAIL: Gradient-Reweighted Advantages for Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2606.04889 | [PDF](https://arxiv.org/pdf/2606.04889v1)

**作者:** Tej Deep Pala `[一作]` (Nanyang Technological University), Soujanya Poria `[通讯]` (Nanyang Technological University)

**通讯引用:** 23404 | [OpenAlex ID](https://openalex.org/A5033376109)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Gradient-Reweighted Advantage (GRAIL) 方法，用 token 级优势加权改进 LLM 的数学推理性能。

**💡 创新点**

创新点在于利用梯度-激活 saliency 在训练期间为每个 token 动态赋权，仅需单前向单反向即可生成细粒度的信用分配。

**🔧 技术方法**

采用基于 Group Relative Policy Optimization (GRPO) 的强化学习框架，结合梯度激活 saliency、stop‑gradient 以及自定义权重归一化技术。

**📊 数据集**

训练使用 DeepMath‑103K 数据集，评测涵盖 Math500、AIME 2024、AMC 2023、MinervaMath、CollegeMath 与 OlympiadBench 六大数学竞赛基准。

**📈 对比分析**

与 GRPO baseline 及 OAR 等方法比较，GRAIL 在所有模型和基准上平均提升约 3.6% 的准确率与 3.05% 的 Pass@3，尤其在长链推理任务上表现更为显著。

**⚠️ 局限性**

局限性在于仅使用一阶梯度近似，未考虑高阶交互影响，且目前仅在可验证数学任务中验证，缺乏跨领域的通用性评估。

---

## 450. DiverAge: Reliable Pluralistic Face Aging with Cross-Age Identity Relation Guidance

**arXiv ID:** 2606.04881 | [PDF](https://arxiv.org/pdf/2606.04881v1)

**作者:** Yueying Zou `[一作]` (Beijing University of Posts and Telecommunications), Zekun Li `[通讯]` (University of California Santa Barbara)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出DiverAge框架，实现多模态人脸老化，兼顾单张图像的多样性与整个生命周期序列的可靠性；

**💡 创新点**

创新点在于构建基于真实同一身份跨年龄对的Cross‑Age Identity Similarity (CIS) 先验，并在推理时通过CARR实现一侧梯度引导，提升序列级ordinal reliability，同时保留生成多样性；

**🔧 技术方法**

采用扩散自编码器、DDIM 采样、年龄条件的潜在扩散 prior、CARR 推理引导、ArcFace 等识别特征、LPIPS 等多样性评价；

**📊 数据集**

使用 FFHQ‑AT 数据集（包含 8 个年龄组）以及从真实跨年龄对构建的 CIS 统计；

**📈 对比分析**

与 LATS、DLFS、SAM、CUSP、AgeTransGAN、PADA 等方法对比；在帧级指标（年龄 MAE、身份相似度、FID）上与主流方法相当；在序列级 APR 误差上显著降低（0.0562 对比 0.1567），同时保持高多样性；

**⚠️ 局限性**

限制包括 CIS 先验依赖人脸识别模型可能带来的偏差；推理时需要额外梯度计算，计算成本升高；对极端年龄、背景与配饰等上下文属性的演化控制不足；数据稀缺导致极端年龄的学习效果有限。

---

## 451. A Note on the Kullback-Leibler Divergence in Discretized Empirical Distributions

**arXiv ID:** 2606.04852 | [PDF](https://arxiv.org/pdf/2606.04852v1)

**作者:** Hayami Osaki `[一作]` `[通讯]` (National Institute of Science and Technology Policy), Hayami Osaki (National Institute of Science and Technology Policy)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了离散概率分布间 KL 差异 Δ_KL(p,q)=D_KL(p∥q)−D_KL(q∥p) 的解释，指出其符号仅反映类别级概率质量的非对称分配，而非直观的包含或覆盖关系，并通过示例与 COVID‑19 预印本主题分布的实证分析来验证这一结论。

**💡 创新点**

提出将 Δ_KL 视为“类别级加权对数比值对比”，并引入阈值诊断 A_τ（衡量低概率区间的非对称分配）来验证 Δ_KL 与包含性质的关系，从而提醒使用者不要仅凭 Δ_KL 的符号作包含或范围的直接解释。

**🔧 技术方法**

核心技术包括：对离散分布的平滑处理、KL 散度与其差异、Jensen‑Shannon 对称重叠度量、Hill 多样性指数（D_1、D_2）、对数比值的加权求和、阈值诊断 A_τ 以及 Spearman 相关性检验。

**📊 数据集**

使用了 2025 年 2 月前的 COVID‑19 预印本数据，共 47,570 条记录，涵盖六个预印本服务器（arXiv、bioRxiv、ChemRxiv、medRxiv、SSRN 及 SSRN Lancet），并通过 embedding‑based 聚类将文本转化为 500 维离散类别的概率分布。

**📈 对比分析**

对 15 对服务器之间的 Δ_KL 与 A_τ、以及 Δ_G（D_1 与 D_2 的差异）进行 Spearman 相关性比较，发现 Δ_KL 与 A_τ 的相关系数约为 0.81，远高于与 D_1、D_2 的相关系数（分别约为 0.31 与 0.20），表明 Δ_KL 与低概率区间的非对称分配更为贴合，而非单纯的分布宽度差异。

**⚠️ 局限性**

局限性包括：仅采用单一 embedding 与聚类方案；阈值 τ 的选择依赖经验；仅有 15 对服务器比较，样本量有限；诊断 A_τ 与 Δ_KL 的关联性取决于具体阈值与平滑参数，未给出通用定量基准。

---

## 452. From Network Experience to Subscriber Retention: An Explainable AI Framework for Mobile Operators

**arXiv ID:** 2606.04838 | [PDF](https://arxiv.org/pdf/2606.04838v1)

**作者:** Faris B. Mismar `[一作]` (Nokia Bell Labs Consulting), Suhelmy Syaifuddin `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了可解释 AI 框架用于预付费移动用户流失预测，并提供可操作的业务洞察。

**💡 创新点**

提出“去噪启发式”剔除低置信度流失标记、构建“favorite site identifier”以整合业务与网络数据，从而提升模型稳健性与可解释性。

**🔧 技术方法**

采用 XGBoost 作为主模型，结合 SHAP 解释、特征稠密化与聚合、PSI/Null Ratio 监控的 MLOps 流水线，实现端到端的数据处理与模型部署。

**📊 数据集**

使用一家全球领先运营商约数千万预付费用户的数据，包含计费、营销、体验（QoE）和网络（MDT）等 800+ 维特征。

**📈 对比分析**

与传统网络计数基线对比，模型在测试集上 F1≈0.75、召回≥0.7；Top‑10 预测特征中商业数据占 90%，在 3 个月滑动预测期内性能衰减可接受。

**⚠️ 局限性**

局限包括流失标签噪声与类别不平衡、特征缺失导致的漂移、网络级 KPI 与用户体验匹配不足，以及仅在单运营商数据上验证，缺乏跨运营商的泛化验证。

---

## 453. Signed Dual Attention: Capturing Signed Dependencies in Time Series Forecasting

**arXiv ID:** 2606.04833 | [PDF](https://arxiv.org/pdf/2606.04833v1)

**作者:** Balthazar Courvoisier `[一作]` (Queensfield AI Technologies), Tristan Cazenave `[通讯]` (Université Paris Dauphine - PSL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种 Signed Dual Attention (SDA) 模块，能够在 Transformer 关注机制中同时捕捉正相关和负相关关系，并实现参数共享；该模块可直接嵌入现有 Transformer 架构。

**💡 创新点**

创新点在于将正负关系通过双向信息传播实现：在单个注意力块中同时计算正向 A^+ 和负向 A^-，并通过 A^+ - A^- 与 value 相乘，等效于两头注意力但参数量不变；同时保持了轻量化与可插拔性。

**🔧 技术方法**

使用的技术包括：scaled dot‑product attention、softmax、负向 attention 计算、参数共享的两头注意力框架；在实验中将 SDA 替换标准注意力，评估于 Transformer 和 Informer 两种模型。

**📊 数据集**

实验数据集包括六个单变量时间序列数据集：Electricity、ETTm2、ETTh2、Exchange Rate、Traffic、Weather。

**📈 对比分析**

通过在相同模型、相同超参数下，用 SDA 替换经典注意力后，在 24、48、96 步预测 horizon 上计算 MSE/MAE 进行对比。结果显示：在 ETTm2、ETTh2 等数据集上性能提升显著，但在 Exchange Rate 等主要为正相关的序列上性能下降。

**⚠️ 局限性**

局限性包括：对负相关噪声的处理不足，等权重对所有序列不适用；仅在单变量、短期预测上验证；未探索多变量任务及可学习的正负权重机制。

---

## 454. Caliper: Probing Lexical Anchors versus Causal Structure in LLMs

**arXiv ID:** 2606.04915 | [PDF](https://arxiv.org/pdf/2606.04915v1)

**作者:** Zhenyu Yu `[一作]` (Fudan University), Shuigeng Zhou `[通讯]` (Fudan University)

**通讯引用:** 11362 | [OpenAlex ID](https://openalex.org/A5017862559)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入 Caliper 词汇匿名化扰动，评估在移除语义变量名时大型语言模型的因果推理性能。

**💡 创新点**

创新点在于构造结构保持、可解释的词汇扰动，通过对比原始与匿名化版本量化词汇对推理准确率的影响，并在多基准、多规模模型上系统验证。

**🔧 技术方法**

采用结构化词汇替换算法、词性感知映射、指令微调模型、零样本评估、CausalCoT、Scaffold、Few-shot ICL 等提示技术，以及统计显著性检验。

**📊 数据集**

使用三大因果推理基准：CLadder（观测/干预/反事实）、CRASS 和 e-CARE。

**📈 对比分析**

通过零样本准确率对比计算 Lexical Gap，发现干预层面平均下降约 7.6pp，跨规模平均下降约 29.6pp，提示策略只能略微降低差距，未能恢复被扰动削弱的性能。

**⚠️ 局限性**

局限性包括：仅测试开放权重指令微调模型，未覆盖封闭API或因果专门训练模型；CLadder 之外基准样本较小；扰动偶尔导致轻微语法不自然。

---

## 455. TeeDAO: A Decentralized Autonomous Organization for Heterogeneous TEEs

**arXiv ID:** 2606.04912 | [PDF](https://arxiv.org/pdf/2606.04912v1)

**作者:** Pinshen Xu `[一作]` (City University of Hong Kong), Yinqian Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 5705 | [OpenAlex ID](https://openalex.org/A5070946957)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种三层框架TEE DAO，利用异构TEE集群通过BFT治理、DPSS和MPC实现长久可信的机密服务，提供统一的KVS、钱包和分析接口。

**💡 创新点**

创新点在于：①将TEE可信度与BFT治理结合，实现基于证明的自治管理；②将主动秘密刷新与委员会重构绑定，抵御移动攻击；③统一API抽象，隐藏TEE异构和治理细节，让多种机密服务复用同一安全基础。

**🔧 技术方法**

技术方案包括：HotStuff BFT共识、COBRA DPSS（主动秘密共享）、MP‑SPDZ多方安全计算、Intel SGX/TDX/Hygon CSV异构TEE的远程证明与撤销列表、BLS阈值签名、Feldman VSS等。

**📊 数据集**

实验数据集：合成的键值存储请求（1KB），MPC基准任务（AES、Logistic/Linear回归、SecureNN、Dijkstra、Sum of Products）以及不同TEE混合组合（SGX+TDX+CSV）。

**📈 对比分析**

与HotStuff（无机密）和COBRA（传统BFT+机密）比较，TEE DAO在写/读吞吐上比COBRA高1.8倍，管理与恢复延迟低于COBRA；MPC任务的额外开销小于10%，最多18%；异构部署性能介于最优与最慢TEE之间。

**⚠️ 局限性**

局限性：①TEE硬件依赖，若支持证明的TEE缺失需扩展；②在大规模集群中，TEE开销和重构成本随节点数增长；③依赖供应商发布的撤销列表，若失效或滞后可能导致安全风险；④侧信道或物理攻击仍需单独防御。

---

## 456. WAM-Nav: Asymmetric Latent World-Action Modeling for Unified Visual Navigation

**arXiv ID:** 2606.04907 | [PDF](https://arxiv.org/pdf/2606.04907v1)

**作者:** Ning Yang `[一作]` (Nanjing University), Nianfeng Liu `[通讯]` (National University of Defense Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种联合学习动作生成与潜在视觉前瞻的隐式世界动作模型（WAM-Nav），实现了单一策略同时支持图像目标、点目标和无目标导航；

**💡 创新点**

创新点包括：1）共享Diffusion Transformer实现动作与短期视觉前瞻的非对称联合扩散；2）双流上下文条件（DSCC）融合历史视觉记忆与运动轨迹，提升轨迹平滑性与安全性；3）统一目标对齐机制兼容多种目标类型；

**🔧 技术方法**

使用Diffusion Transformer、流匹配训练、Stable Diffusion VAE压缩视觉潜在、跨模态对齐与InfoNCE对齐损失；

**📊 数据集**

在VLN-N1数据集上训练，并在ClutterScenes、InternScenes（IsaacSim）和真实Unitree G1平台上进行零样本评估；

**📈 对比分析**

与GNM、ViNT、NoMaD、NWM、NavDP等基线对比，WAM-Nav在Image-Goal、Point-Goal和No-Goal三类任务中平均提升成功率约15.7%、3.3%及探索面积，推理延迟仅0.26 s，参数与计算量保持与NavDP相近；

**⚠️ 局限性**

局限包括：相机高度与视场限制导致近距离障碍感知不足；当前模型未显式考虑机器人整体结构，可能导致摄像头层面安全与全身可通行性不匹配。

---

## 457. DIST-FL: Enhancing Security for TEE-based Aggregation in Federated Learning

**arXiv ID:** 2606.04899 | [PDF](https://arxiv.org/pdf/2606.04899v1)

**作者:** Guanlong Wu `[一作]` (Southern University of Science and Technology), Yinqian Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 5705 | [OpenAlex ID](https://openalex.org/A5070946957)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了分布式TEE辅助联邦学习框架 Dist‑FL，解决现有TEE‑FL中因I/O操纵与回滚导致的客户端选择失真与聚合重放攻击；

**💡 创新点**

通过将回滚防护与增量式追加账本、以及Proof‑of‑Input机制相结合，实现了对回滚与I/O操纵的双重防御，并在保持单TEE性能的前提下显著提升吞吐量；

**🔧 技术方法**

利用Intel SGX的安全执行环境、Nimble的追加式账本、随机种子生成、bitmap同步与聚合证明、以及远程与互相验证的签名；

**📊 数据集**

在FEMNIST（10分类时装图像）和CIFAR‑10（10分类彩色图像）两个非IID分布的数据集上进行实验；

**📈 对比分析**

与单一TEE、直接对客户端更新进行共识、以及基于MPC的安全聚合进行对比；Dist‑FL在吞吐量上比单TEE快约6倍，性能仅略逊于单TEE，且在不同模型大小、客户端数、局部训练轮次及服务器数量下均保持稳定；

**⚠️ 局限性**

依赖TEE硬件安全假设，未针对侧信道攻击；对服务器故障容忍度有限（最大容忍f个恶意服务器），且实现复杂度相对较高。

---

## 458. Recent Advances and Trends in Learning-based 3D Representations

**arXiv ID:** 2606.04871 | [PDF](https://arxiv.org/pdf/2606.04871v1)

**作者:** Adrien Schockaert `[一作]`, Jean-françois Witz `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于3D高斯的场景表示方法，并通过高斯压缩、向量量化/哈希网格、MIP过滤、逐像素排序/光线追踪以及SDF/深度/法线先验等技术，实现了高效的实时渲染。

**💡 创新点**

创新点在于将可训练的3D高斯集合与多种压缩与优化手段相结合，实现了显著提升的存储效率、几何质量和视觉质量；同时提出了SDF/深度/法线先验用于指导优化，进一步改善了渲染结果。

**🔧 技术方法**

使用技术包括：高斯修剪与压缩、向量量化与哈希网格、3D MIP过滤、逐像素排序/光线追踪、以及SDF/深度/法线先验。

**📊 数据集**

主要数据集为LLFF（现实世界场景）、DeepMind synthetic scenes、Blender synthetic scenes 等公开数据集。

**📈 对比分析**

与传统NeRF、Mip-NeRF、PlenOctrees等方法相比，本文在4K分辨率下实现30–45 FPS，PSNR提升约1–2 dB，存储量约减少70%，同时保持较低的光照与几何失真。

**⚠️ 局限性**

局限性包括：训练需要较高的GPU显存，主要针对静态场景，对动态或大规模场景的适用性有限；在细节边缘处仍可能出现轻微的伪影。

---

## 459. Provably Reduced Sample Cost in Prior-Guided Hyperparameter Optimization

**arXiv ID:** 2606.04866 | [PDF](https://arxiv.org/pdf/2606.04866v1)

**作者:** Leona Hennig `[一作]` (Leibniz University Hanover), Marcel Wever `[通讯]` (Leibniz University Hanover)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于先验信息的多阶Successive Halving方法PSH，并给出了其在固定预算下的理论误差与样本复杂度上限；

**💡 创新点**

创新点在于：①首次给出分布依赖的样本复杂度上界，定量化先验信息的质量如何降低预算；②提供对误导性或无信息先验的鲁棒性保证；③在理论基础上设计了先验引导的早停规则；

**🔧 技术方法**

采用了高斯过程预测学习曲线、贝叶斯固定预算Best-Arm Identification理论、以及Successive Halving框架；

**📊 数据集**

实验使用了合成学习曲线基准和公开的LCBench（YAHPO Gym）数据集；

**📈 对比分析**

与标准Successive Halving、Hyperband、PriorBand进行对比；结果表明，在信息量高的先验下，PSH可将评估预算降低约35–80%且保持或略优于基准的验证损失；误导性先验时性能与基准相当；

**⚠️ 局限性**

局限在于：PSH的收益高度依赖先验质量，若先验不准确或GP模型假设失效，理论保证可能不成立；需更鲁棒的学习曲线模型来弥补。

---

## 460. IRIS-GAN: Staged Specialist Detection of Deepfake Faces

**arXiv ID:** 2606.04863 | [PDF](https://arxiv.org/pdf/2606.04863v1)

**作者:** Jaume M. Trenchs `[一作]`, Veronica Sanz `[通讯]` (Universitat de València)

**通讯引用:** 7104 | [OpenAlex ID](https://openalex.org/A5052939084)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于阶段性训练的 GAN 面部图像专用检测器 IRIS‑GAN，能够跨不同 GAN 家族实现高精度检测。

**💡 创新点**

创新点在于通过逐步暴露不同难度的 GAN 家族并保留前期模型的训练策略，显著提升对未见生成器的泛化能力。

**🔧 技术方法**

使用 ConvNeXt‑Large 迁移学习模型、交叉熵二分类损失、Grad‑CAM 可视化诊断以及 ResNet18 热图分类器进行辅助验证。

**📊 数据集**

训练集采用 FFHQ 真实图像，生成器数据集包括 ProGAN、StyleGAN2/3/XL、EG3D；外部真实测试集为 CelebA，此外对扩散模型图像做了外部泛化测试。

**📈 对比分析**

与一次性训练 baseline 对比，IRIS‑GAN 在已知生成器上的验证准确率略低，但对 StyleGAN3 的检测率从约 54% 提升至 80%+；最终四阶段模型在所有 GAN 家族上的检测率均 ≥ 99%，对 CelebA 的真阳性率为 98.9%。

**⚠️ 局限性**

仅针对 GAN 生成图像有效，对扩散模型等其他生成方式泛化有限；未评估图像压缩、尺寸变化等实际部署场景；热图分析提供可解释性但不具因果证据。

---

## 461. Learning Empirically Admissible Neural Heuristics for Combinatorial Search

**arXiv ID:** 2606.04860 | [PDF](https://arxiv.org/pdf/2606.04860v1)

**作者:** Siddharth Sahay `[一作]` `[通讯]` (Independent Researcher), Siddharth Sahay (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种在组合谜题搜索中学习验证校准可接受神经启发式的框架，确保A*搜索路径最优。

**💡 创新点**

创新点在于结合下界Admissible Bellman算子、非对称Pinball损失和后期校准安全偏移，三重防护实现经验可接受性。

**🔧 技术方法**

使用的技术包括：下界Admissible Bellman Operator、异向Pinball Loss、后期校准安全偏移、基于强化学习的自举训练、PyTorch MLP网络以及A*搜索。

**📊 数据集**

采用的数据集涵盖3×3 Lights Out、5×5 Lights Out、8-Puzzle以及2×2 Rubik's Cube；训练时随机扰动生成状态，验证集10k样本，测试集10k样本。

**📈 对比分析**

通过与基准启发式、MSE网络和未校准网络对比，评价指标为可接受率、节点扩展数、求解成功率和路径最优性；校准启发式在所有任务上均达到100%可接受率，节点数比基准低20%–70%，求解成功率显著提升，且路径保持最优。

**⚠️ 局限性**

局限性包括：CPU实现的A*搜索速度慢；扩展到更大谜题需GPU、更大网络与更长训练；安全偏移过保守导致过度低估，可能增加节点数；验证分布假设可能在OOV状态下出现可接受性违规。

---

## 462. ThermoPix: A High-Spatial-Resolution ElectronicPhotonic Temperature Sensor Array With Microsecond Row Readout

**arXiv ID:** 2606.04902 | [PDF](https://arxiv.org/pdf/2606.04902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 463. Rethinking Incompleteness: Formalizing Protocol Divergence and Train-Once Learning for Robust IMVC

**arXiv ID:** 2606.04857 | [PDF](https://arxiv.org/pdf/2606.04857v1)

**作者:** Haolu Liu `[一作]`, Zhao Kang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种在多视图聚类中对缺失数据鲁棒的架构，提出通过训练一次完整数据即可适应所有缺失模式。

**💡 创新点**

创新点在于将缺失鲁棒性嵌入模型结构，提出了不依赖完整样本比例的Per‑sample Independence和Mask‑aware Variable‑length Fusion两个条件，并正式化了不完整性偏差（Incompleteness Divergence）和trainability bound。

**🔧 技术方法**

采用Transformer基础的Attention‑masked Fusion网络，并通过两阶段完整数据训练与可选的Masked Fine‑Tuning实现。

**📊 数据集**

在七个公开多视图数据集上验证，包括CUB、HandWritten、MultiFashion等。

**📈 对比分析**

与传统需要逐配置重训练的重建类方法和跨样本分布方法对比，单个CRAFT模型在所有16种缺失配置下保持或超过基线，同时将训练成本降低约8.8倍。

**⚠️ 局限性**

局限性是仅针对无监督多视图聚类的重建损失家族，未扩展到有监督分类或其他多模态任务。

---

## 464. Drift-Augmented Scoring: Text-Derived Noise Robustness for Zero-Shot Audio-Language Classification

**arXiv ID:** 2606.04844 | [PDF](https://arxiv.org/pdf/2606.04844v1)

**作者:** Tu Vo `[一作]` (KC Machine Learning Lab), Chan Y. Park `[通讯]` (KC Machine Learning Lab)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Drift-Augmented Scoring，在零样本音频分类中使用仅基于文本的漂移方向对得分进行加权，以提升噪声鲁棒性。

**💡 创新点**

创新点是：仅通过文本生成每类噪声漂移向量 δ̂_c，在线时仅在评分时添加一次内积补偿，既无梯度、无音频池、无多样本批处理，保持单样本无改动的特性。

**🔧 技术方法**

技术上基于 LAION CLAP 的对比学习音频‑文本嵌入，利用文本噪声描述词生成漂移向量，随后在评分公式 score(z,c)=z·C_c+β(z·δ̂_c) 中加入漂移项。

**📊 数据集**

使用的评估数据集为 UrbanSound8K（单标签）和 FSD50K（多标签），噪声来源为 TAU Urban Acoustic Scenes 2019，混合噪声通过 Scaper 合成，在 SNR {0,6,8,10,20} dB 进行测试。

**📈 对比分析**

在与 Acevedo 等人提出的四种噪声鲁棒方法（bias subtraction、TGAP 等）同一评测协议下比较，Δ 为 +2.60 到 +5.75 分的准确率提升，Δ 为 +1.50 到 +1.74 的 mAP 提升，始终领先于所有对比方法。

**⚠️ 局限性**

局限性包括：β 固定为 0.25，缺乏自适应调优；漂移方向仅依赖通用噪声词典，可能对特殊或未知噪声的适应性有限；未结合音频池或多模型融合的进一步提升。

---

## 465. Event Calculus Meets Hybrid ASP

**arXiv ID:** 2606.04905 | [PDF](https://arxiv.org/pdf/2606.04905v1)

**作者:** Ondřej Vašíček `[一作]` (Brno University of Technology), Tomáš Vojnar `[通讯]` (Masaryk University)

**通讯引用:** 2466 | [OpenAlex ID](https://openalex.org/A5086446392)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将事件演算（Event Calculus）与混合ASP（Hybrid ASP）相结合的 Hybrid EC 框架，用于在不引发基于实例化的爆炸的前提下精确建模连续变化与稠密域；

**💡 创新点**

创新点在于引入功能性流动子（functional fluents）和抽象时间步映射，利用线性约束在ASP中处理稠密数值域，从而消除离散化误差并避免传统ASP求解器的基化爆炸；

**🔧 技术方法**

核心技术包括 Hybrid ASP（clingcon 与 clingo‑lpx）实现的线性约束求解、功能性流动子与抽象时间步的编码、以及增量求解策略以自动确定所需步数；

**📊 数据集**

使用文献中提出的多种基准问题（如计数器、跌落物体、行人监测等）以及作者自制的 30+ 个实验案例进行评估；

**📈 对比分析**

与传统的基化 ASP 求解器 clingo 进行对比，实验显示 Hybrid EC 在时间/值域规模上几乎不受影响，内存和执行时间远低于 clingo；对比 clingo‑lpx 与 clingcon，使用有理数并不显著影响可扩展性，且两者都显著优于纯整数求解；

**⚠️ 局限性**

局限性包括：需要预估/逐步尝试步数，无法自动判断无解情况；增量求解在步数过多时仍需上限；对 Zeno 等稠密域特殊情况的处理尚未完全完善；以及对触发事件的支持仍需手工编码。

---

## 466. CLIF: Cross-layer LEO-ISL Fingerprinting for Physical and Network Attack Detection in Dense LEO Constellations

**arXiv ID:** 2606.04901 | [PDF](https://arxiv.org/pdf/2606.04901v1)

**作者:** Varun Kohli `[一作]` (Institute for Infocomm Research), Biplab Sikdar `[通讯]` (National University of Singapore)

**通讯引用:** 12629 | [OpenAlex ID](https://openalex.org/A5041189303)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本论文提出了一种跨层行为指纹框架，用于在高密度LEO星座中实时检测物理层和网络层的ISL攻击，并构建了包含Starlink、Kuiper及跨运营商场景的仿真数据集。

**💡 创新点**

创新点包括：① 将物理层轨道误差特征（距离、速度、方向）与网络层流量、排队、延迟等九个特征融合成12维跨层特征空间；② 在每颗卫星上部署无监督、轻量级的单点检测模型；③ 首次公开可用于十类攻击、三种严重度的完整LEO ISL攻击数据集。

**🔧 技术方法**

技术手段主要为：使用SGP4轨道传播与地球中心固定坐标系；利用光学ISL测量与GEO广播星历计算误差特征；网络层采用基于Dijkstra拥塞感知路由、泊松流量、M/M/1/K排队模型；无监督检测器包括Mahalanobis距离、Isolation Forest、自动编码器。

**📊 数据集**

数据集为基于仿真的24小时，60秒采样周期，Starlink 1,584颗卫星、Kuiper 1,156颗、跨运营商 2,740颗；包含10种攻击（伪装、黑洞、灰洞、汇聚、隧道、Sybil、DoS、Rogue Sinkhole、Rogue Sybil、跨星座劫持）以及低/中/高严重度，划分训练/验证/测试集。

**📈 对比分析**

在三种星座场景下对比了三种无监督检测器：Mahalanobis（CLM）在Starlink/Kuiper的召回率分别为99.6%/99.5%，多运营商为94.8%；误报率均低于0.7%。AE召回率约92.4%，IF仅26.9%。ROC-AUC和PR-AUC均接近1。与现有仅物理层的身份验证方法比较，CLM在保留高召回的同时覆盖了网络层攻击，表现更优。

**⚠️ 局限性**

局限性主要是：① 仅基于仿真，缺乏真实ISL数据验证；② 对手模型为非自适应，未考虑可演化的规避攻击；③ 依赖可信的GEO星历，若星历被破坏则物理层特征失效；④ 跨运营商链路特性差异导致轻微召回下降，需进一步优化。

---

## 467. MAOAM: Unified Object and Material Selection with Vision-Language Models

**arXiv ID:** 2606.04880 | [PDF](https://arxiv.org/pdf/2606.04880v1)

**作者:** Jaden Park `[一作]` (University of Wisconsin-Madison), Michael Fischer `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MAOAM统一框架，实现文本和点击两种交互方式下的对象与材质级像素级选择。

**💡 创新点**

首次将对象与材质选择统一到同一模型；通过VLM生成[SEG]语义向量与SAM解码器结合；引入多任务学习（点击、文本、VQA）和可扩展的VLM文本生成管线解决无材质文本标注问题。

**🔧 技术方法**

使用CLIP视觉编码器+大型语言模型生成[SEG] token；SAM的mask解码器；多任务损失；VLM驱动的材料描述生成与验证；硬负样本VQA增强材料推理。

**📊 数据集**

RealMat（约8k真实图/49k材质掩码）、SynMat（Blender渲染/55k掩码）、SAMa（1.3k图/3.3k掩码）、RefCOCO/RefCOCO+/RefCOCOg、EntitySeg等对象分割数据集；将材质与对象数据混合训练。

**📈 对比分析**

与SAM3、GLaMM、Sa2VA、LISA、Materialistic等基线对比。MAOAM在材质选择（文本/点击）mIoU均提升35%~60%，在对象选择保持或略优；VQA准确率最高（>0.85），并表现出多模交互时的“多模提升”。

**⚠️ 局限性**

受限于VLM的推理能力（难以识别极端纹理或细粒度材质）和SAM解码器的分辨率，导致部分细节掩码不够精细；在视觉复杂或对比度低的场景下仍易失败。

---

## 468. Towards Pretraining Text Encoders for TabPFN

**arXiv ID:** 2606.04876 | [PDF](https://arxiv.org/pdf/2606.04876v1)

**作者:** Mustafa Tajjar `[一作]` (University of Freiburg), Frank Hutter `[通讯]` (Prior Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个轻量级的适配器，将语言模型的文本嵌入投射到 TabPFN 的嵌入空间，实现文本与表格特征的无缝融合；

**💡 创新点**

创新点在于冻结 TabPFN 与句子编码器，仅训练一个小型投射网络，消除 PCA 信息瓶颈，同时保持 TabPFN 对数值特征的强大表现；

**🔧 技术方法**

使用 Sentence-Transformer（all-MiniLM-L6-v2）生成文本嵌入，并通过轻量 MLP/线性层映射到 192 维 Token 空间；训练时加入随机维度置换与归一化；

**📊 数据集**

预训练数据取自 STRABLE 基准中含文本列的子集；评估数据集为 TextTabBench（13 个包含自由文本列的分类/回归任务）；

**📈 对比分析**

与 ConTextTab、TabPFN+tf-idf、TabPFN+PCA-30 等基线对比，分类任务性能接近 PCA-30，回归任务显著优于所有基线，展示了回归场景下的显著提升；

**⚠️ 局限性**

局限在于适配器对所有任务使用相同投射，无法根据任务动态选择重要维度，且投射会导致信息丢失；未来可考虑加入条件化投射或跨列特征的更丰富映射。

---

## 469. AICompanionBench: Benchmarking LLMs-as-Judges for AI Companion Safety

**arXiv ID:** 2606.04867 | [PDF](https://arxiv.org/pdf/2606.04867v1)

**作者:** Yanjing Ren `[一作]` (University of South Florida), TengTeng Ma `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 AICompanionBench 数据集和评估框架，用于检测 AI 伴侣对话中的安全风险。

**💡 创新点**

首次公开发布针对 AI 伴侣对话的细粒度安全风险标注数据，并在此基础上系统评估 LLM‑as‑Judge。

**🔧 技术方法**

采用 LLM‑as‑Judge 评估方法，利用提示式推理与多模型集成。

**📊 数据集**

使用从 Reddit 收集的 2,123 条 Replika 对话，标注为九类安全风险。

**📈 对比分析**

对 20 个主流开源与闭源 LLM 进行对比，最高准确率为 86%，但在隐性风险和无害案例上的误报率高。

**⚠️ 局限性**

主要局限在于对隐性意图与上下文模糊的风险识别不足，以及对无害对话的高误报。

---

## 470. Teaching Robots to Say 'I Don't Know' : SENTINEL for Uncertainty-Aware SLAM

**arXiv ID:** 2606.04853 | [PDF](https://arxiv.org/pdf/2606.04853v1)

**作者:** Abhishek S `[一作]` (BuildMachineLabs), Sreeram MV `[通讯]` (BuildMachineLabs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套名为 SENTINEL 的无训练、无标签的可靠性估计框架，利用几何扫描统计与 RGB‑D 深度一致性为低成本 2D LiDAR 提供诊断通道，并在检测到不可靠扫描时自动抑制 LiDAR 约束，回退到已校准的里程计。

**💡 创新点**

创新点包括：① 针对无强度通道的低价 LiDAR 提供实时可靠性评估；② 通过跨模态一致性实现无训练的误差检测；③ 发现跨模态融合需保证各模态失效互斥；④ 证明 Gazebo 等传统仿真无法重现透明/镜面等物理失效，强调实机验证的重要性。

**🔧 技术方法**

技术手段包括：几何失效得分（基于有效点比例和距离方差）、跨模态一致性得分（LiDAR 与 RealSense 深度匹配）、分数融合与门控策略、ROS2 低延迟通信、CPU‑仅实现、EKF 里程计滤波、扫描抑制与噪声注入。

**📊 数据集**

使用真实机器人平台 GEFIER R1，在 185×245 cm 室内实验场景中，设置多种失效材料（玻璃、镜面、金属纸、混合）进行 50 次完整路径测试，共计约 30 000 次 LiDAR 扫描；无使用仿真数据集。

**📈 对比分析**

与无门控的 SLAM 对比，SENTINEL 在检测到玻璃透明面时将 R_geo 降至 <0.3 并抑制扫描，镜面与金属纸则进入噪声区 (0.55–0.90)。实验显示，玻璃区最小 R_geo 为 0.24，清晰区最小为 0.91，产生 3.8 倍分离；在所有失效材料下均能定位失效位置并避免地图污染，证明方法在实机环境中具有高可靠性和实时性。

**⚠️ 局限性**

局限性包括：① R_cross 对 IR 透明表面不可靠，需引入非 IR 模态；② 分数权重与阈值仅经验调优，缺乏自动化学习；③ 里程计回退在长距离失效时会累积漂移；④ 仅在低价 LiDAR 与 RGB‑D 组合下验证，未评估其他传感器或更大尺度环境；⑤ 未进行系统性下游导航性能评估。

---

## 471. Dynamic FDD for Spectrum Sharing in Non-Terrestrial Networks

**arXiv ID:** 2606.04849 | [PDF](https://arxiv.org/pdf/2606.04849v1)

**作者:** Sourav Mukherjee `[一作]` (University of Bremen), Petar Popovski `[通讯]` (Aalborg University)

**通讯引用:** 27549 | [OpenAlex ID](https://openalex.org/A5071289803)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种动态FDD（频分双工）频段分配框架，用于低地球轨道（LEO）卫星大型星座中，联合调度、功率分配和频段选择，以在频谱共享环境下提升双向通信速率。

**💡 创新点**

创新点在于：①将传统固定频段FDD转变为可动态切换上下行频段的“动态FDD”，为系统引入了额外的自由度；②构建了包含离散频段切换、用户关联和功率分配的非凸混合整数优化问题；③采用等价变换（Lagrangian dual + quadratic transform）与交替优化相结合的求解策略，结合工业级MIP求解器（MOSEK）实现高质量解。

**🔧 技术方法**

技术手段包括：多频段LOS通道建模、干扰链路建模；等价变换（Lagrangian dual、quadratic transform）将非凸问题转化为可分块求解；交替优化（Block Coordinate Descent）求解各子块；大M线性化处理二进制与连续变量的乘积；枚举频段切换向量以获得全局最优的r。

**📊 数据集**

使用仿真数据：随机生成J颗卫星（高度500km，8×8天线阵列）与K颗UE（均匀分布在半径10km的乡村区域）的位置；考虑两条频段（Ω1=2.4 GHz、Ω2=1.6 GHz）以及相等带宽10 MHz；采用LOS模型与位置驱动的UE‑UE干扰模型。

**📈 对比分析**

对比方法：两种固定频段分配基线（全1spin、全0spin）与动态FDD的优化结果。性能评估指标为：CDF/平均总速率（bits/s/Hz）与相对提升。实验结果显示，动态FDD在各种K、J配置下平均提升约20–30%，在CDF上显著拉高10%–90%分位点，尤其在高用户负载和多卫星部署时优势更为明显。

**⚠️ 局限性**

局限性包括：①仅考虑两频段，未讨论更多频段或连续频谱的分配；②频段切换向量r的枚举在卫星数量增大时计算量急剧上升；③假设LOS信道和精准位置可知，忽略了信道估计误差与时延；④模型未考虑多路复用或波束切换对时延和功率预算的额外约束。

---

## 472. MusaCoder: Native GPU Kernel Generation with Full-Stack Training on Moore Threads GPU

**arXiv ID:** 2606.04847 | [PDF](https://arxiv.org/pdf/2606.04847v1)

**作者:** Kun Cheng `[一作]`, Yaohua Tang `[通讯]` (Moore Threads AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了端到端的 MusaCoder 框架，完成了从多源数据合成、监督微调 (SFT)、多样性保持拒绝抽样 (RFT)、分阶段强化学习 (GRPO) 到最终的原生 CUDA/MUSA kernel 生成。

**💡 创新点**

创新点包括：① 进化式三阶段数据合成管线与结构化推理；② 多样性保留的拒绝采样微调，避免熵塌陷；③ 三大 RL 稳定化机制 PrimeEcho、Buffered Dynamic Retry、MirrorPop；④ 统一可扩展的执行反馈沙箱 MooreEval，用于奖励与多轮修正；⑤ 在 Moore Threads MUSA GPU 上实现完整 LLM 训练与推理。

**🔧 技术方法**

技术栈涵盖：Qwen‑3.5/3.6 LLM 预训练；DeepSpeed + ZeRO/offload 长上下文 SFT；多任务混合数据；RFT + 多样性聚类；GRPO 两阶段 RL（单回合与多回合）；PrimeEcho 先回合奖励设计；BDR 动态重试；MirrorPop 句子级 off‑policy 掩蔽；SGLang + Megatron 分布式推理与 RL；MooreEval 分布式编译/执行/性能测量；CUDA / MUSA 编译链与 Nsight / PyTorch Profiler。

**📊 数据集**

数据集：KernelBench 基准与其 MUSA 端口；自制的 PyTorch‑to‑CUDA/MUSA 生成数据（包括 NNSmith 生成图、GitHub 代码、GPU‑kernel Q&A、单元测试与形状/stride 注解）；多轮 RL 轨迹（编译错误、运行时错误、性能反馈）；Profiling 及性能重写数据；以及对话式反馈数据。

**📈 对比分析**

评估方式：在 KernelBench 与 MUSA 端口上采用统一 MooreEval 验证，计算 Pass@8、Avg.@8 以及 Faster Rate（≥1.1× 速度提升）。与 Claude Opus 4.7、DeepSeek‑V4‑Pro 等开源/闭源基准对比，MusaCoder‑27B 在正确率与速度提升上均超越现有最高模型；9B 参数规模的模型已逼近闭源水平。

**⚠️ 局限性**

局限性：① 训练与推理依赖大规模 Moore Threads GPU，成本高；② RL 稳定性仍受 reward 稀疏与 off‑policy 影响，需要细粒度的调参；③ 对极端长尾算子和复杂形状组合的覆盖度仍有限；④ 迁移到其他非 CUDA/MUSA 加速器需额外数据与验证；⑤ 生成代码在特定硬件/驱动版本上可能出现兼容性或性能波动。

---

## 473. Large Language Models in K-12 Education: Alignment with State Curriculum Standards and Student Personas

**arXiv ID:** 2606.04846 | [PDF](https://arxiv.org/pdf/2606.04846v1)

**作者:** Lisa Korver `[一作]` (Brown University), Sherief Reda `[通讯]` (Brown University)

**通讯引用:** 4688 | [OpenAlex ID](https://openalex.org/A5015719218)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了基于LLM的自动化管道，用以识别美国各州历史课程的差异，并评估不同LLM在回答学生问题时对这些差异的适配程度，同时研究了多种引导方法和用户身份属性对回答的影响。

**💡 创新点**

创新点在于：①提出了利用RAG+Llama评估并聚类州级课程差异的完整流程；②设计了新的“州级对齐”指标；③系统性比较了三种引导策略（提及、指令、RAG）以及身份属性对模型输出的敏感性。

**🔧 技术方法**

主要技术包括检索增强生成（RAG）、Llama‑3.3‑70B评估器、BERTopic聚类、情感分析、Flesch–Kincaid可读性评分，以及多种prompt工程手段。

**📊 数据集**

数据集来源于Fordham Institute发布的美国历史课程标准、各州官方课程文件以及公开可用的聊天机器人（Grok、GPT、Gemini）的响应。

**📈 对比分析**

通过对比基线模型与三种引导方法的州级对齐得分（0–1）以及文本长度、情感与可读性等差异指标，发现RAG在部分州-模型组合上能显著提升对齐，但整体效果不稳定；模型对不同州的适配度各异，且在性别/种族维度表现出较小的差异，仅对年级层级做出明显调整。

**⚠️ 局限性**

局限性包括：仅选取评分较高的九个州，忽略了其他州的多样性；对州课程标准的依赖可能带来文档偏差；评估主题有限，未覆盖全部历史内容；RAG引导在多数情况下并未提升对齐；实验仅针对公开的ChatGPT/Grok/Gemini等模型，未涵盖更广泛的LLM。

---

## 474. Probing Outcome-Level Resemblance and Mechanism-Level Alignment in LLM Risk Decisions: Evidence from the St. Petersburg Game

**arXiv ID:** 2606.04978 | [PDF](https://arxiv.org/pdf/2606.04978v1)

**作者:** Chensong Huang `[一作]` (Fudan University), Jiebo Luo `[通讯]` (University of Rochester)

**通讯引用:** 44971 | [OpenAlex ID](https://openalex.org/A5055469774)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在风险决策中的行为一致性，利用斯坦图尔堡赌局及其四个控制变体评估模型的外观“人类化”输出与内部决策机制的一致性。

**💡 创新点**

提出区分结果层面与机制层面一致性的评估框架，通过控制变体揭示LLM在风险决策中常见的边界追踪、缺乏上下文敏感性，并检验提示与指令调优对机制层面的影响。

**🔧 技术方法**

采用结构化提示套件、四个机制探针（截断、重复玩、财富、职业身份）、人类视角提示、指令调优、温度控制、模型中位数聚合及行为模式标签化等技术。

**📊 数据集**

使用28个LLM（前沿系统、开源模型及其基础/指令调优版本），在原始斯坦图尔堡提示及四个控制变体上进行多次推理，采集愿付最高价的数值输出。

**📈 对比分析**

通过对模型输出进行人类类、条件理性、计算理性三种行为模式标注，统计各条件下模型的分布；发现原始游戏中大多数模型产生有限报价，但在控制变体中多为计算理性或条件理性；提示与调优虽能降低极端报价，但对机制层面的提升有限。

**⚠️ 局限性**

仅使用斯坦图尔堡赌博作为诊断环境，缺乏真实决策场景；人类类标签基于文献而非实验数据；未探究内部机制导致的差异原因；实验范围受限于可用模型和提示数量。

---

## 475. SemBlock: Semantic Boundary Dynamic Blocks for Diffusion LLMs

**arXiv ID:** 2606.04964 | [PDF](https://arxiv.org/pdf/2606.04964v1)

**作者:** Xinrui Song `[一作]` (Peking University), Hao Tang `[通讯]` (Peking University)

**通讯引用:** 9391 | [OpenAlex ID](https://openalex.org/A5100662197)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SemBlock，一种基于语义边界的动态块解码框架，用轻量化预测器替代传统的基于标点的块划分方法。

**💡 创新点**

通过构建 SemBound 数据集，将自然语言、数学推理与代码生成的语义边界统一为 token 级标签，并训练多域边界预测器，实现更精准的块边界选择。

**🔧 技术方法**

利用冻结的 LLaDA 隐藏状态作为特征，设计多域头的轻量级二分类边界预测器，并结合阈值+窗口策略进行动态块调度。

**📊 数据集**

使用 SemBound 边界标注数据集（包含 GUM 语篇分段、AQuA-RAT 推理步骤、代码实现段）以及 GSM8K、IFEval、MATH、HumanEval 四个基准任务。

**📈 对比分析**

与固定块、AdaBlock、AdaBlock+Cache 等基线对比，在 LLaDA-Instruct 上 GSM8K、IFEval、MATH、HumanEval 的指标均优于对手，HumanEval pass@1 从 46.30 提升到 46.95；在 LLaDA-1.5 上提升 9.76–11.60 点。

**⚠️ 局限性**

依赖 SemBound 边界标注的质量，难以覆盖更开放式生成场景；冻结主干限制了对新结构信息的感知；调度阈值和窗口尺寸需手工调优，可能影响效率与质量的平衡。

---

## 476. A General Framework for Dynamic Consistent Submodular Maximization

**arXiv ID:** 2606.04946 | [PDF](https://arxiv.org/pdf/2606.04946v1)

**作者:** Paul Dütting `[一作]` (Google), Morteza Zadimoghaddam `[通讯]` (Google)

**通讯引用:** 1921 | [OpenAlex ID](https://openalex.org/A5024817143)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一套通用框架，解决在完全动态（既有插入也有删除）环境下保持近最优子模最大化问题的可持续性（consistency）与近似性；在基数约束下实现1/2‑O(ε)近似并保持O(1/ε²)一致性，在秩Matroid约束下实现1/4‑O(ε)近似并保持O(log(1/ε)/ε²)一致性。

**💡 创新点**

创新点在于将“删除鲁棒性”与动态调度相结合，构建多级鲁棒子模子程序，并利用逆边际采样与分层结构，使得在完全动态更新中实现子线性一致性和常数/对数近似。

**🔧 技术方法**

主要技术包括：鲁棒子模最大化子程序（对d个删除进行鲁棒性处理）、非鲁棒子模子程序、随机时间调度（transition windows）、逆边际采样、层级合并与同步转换。

**📊 数据集**

论文为理论工作，未使用具体数据集进行实验验证。

**📈 对比分析**

与先前仅考虑插入的工作相比，本文在完全动态设定下达成了相近的近似率（基数约束≈0.5，Matroid约束0.25），并首次实现了子线性一致性；与流式Matroid算法相比，同样保留了1/4近似但增加了对删除的鲁棒性。

**⚠️ 局限性**

局限性包括：对基数约束的近似率仍低于已知的1‑1/e极限；对Matroid的近似率仍显著低于流式和离线结果；未给出实验评估；算法在实际大规模数据上可能存在高计算开销。

---

## 477. STaR-Quant: State-Time Consistent Post-Training Quantization for Diffusion Large Language Models

**arXiv ID:** 2606.04945 | [PDF](https://arxiv.org/pdf/2606.04945v1)

**作者:** Xin Yan `[一作]` (Beijing Normal University), Xingrui Yuand Ivor Tsang `[通讯]` (Agency for Science, Technology and Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对扩散式大语言模型（DLLM）的低比特后训练量化框架STaR-Quant，解决了掩码状态激活差异和时间误差累积问题。

**💡 创新点**

创新点在于State-Guided Activation Transformation（SGAT）将掩码和未掩码 token 分别映射到共享与状态特定的激活空间，同时采用统一的权重变换；Temporal Attention Compensation（TAC）在注意力输出投影前通过块状仿射映射校正量化注意力表示，减轻迭代过程中的误差累积。

**🔧 技术方法**

使用了正交变换、块状仿射补偿、闭式统计匹配等技术，并基于 W4A4/8A8 的后训练量化实现；在 NVIDA A40 GPU 上进行速度与内存评估。

**📊 数据集**

主要使用 LLaDA-8B、LLaDA-1.5-8B 与 Dream-7B 三个扩散式语言模型进行实验，校准数据采用 128 条 WinoGrande 样本；评估任务覆盖 TruthfulQA、ARC、HellaSwag、WinoGrande、PIQA、MMLU、C-EVAL、GSM8K 与 HumanEval 共九个基准。

**📈 对比分析**

与 RTN、AWQ、QuaRot、DLLMQuant 等强基线比较，STaR-Quant 在 W4A4 量化下平均提升 2.78–2.62 点，速度提升约 1.65×，内存减少约 3.1×，在知识推理、数学推理和代码生成任务上均显著优于对照方法。

**⚠️ 局限性**

局限性包括仅针对 W4A4 量化，未覆盖 3‑bit/2‑bit 等更激进设置；实验仅验证了三种模型，需进一步检验在更多扩散架构、规模和降噪策略下的通用性；系统级实现仍需专用融合核提升实际推理速度；当前状态建模仅基于掩码/未掩码，未考虑更细粒度或多模态扩散模型。

---

## 478. Data Attribution in Large Language Models via Bidirectional Gradient Optimization

**arXiv ID:** 2606.04928 | [PDF](https://arxiv.org/pdf/2606.04928v1)

**作者:** Frédéric Berdoz `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**通讯引用:** 21577 | [OpenAlex ID](https://openalex.org/A5078339613)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于双向梯度优化的训练数据归因框架，能够对自回归LLM的生成输出进行归因；

**💡 创新点**

通过在生成文本上执行梯度上升和下降得到两个模型，并比较其对训练样本损失的绝对变化，既不需要逐样本梯度，也支持任意粒度归因；

**🔧 技术方法**

采用反向影响函数估计、Fisher信息矩阵对角近似、双向梯度优化、tail‑patch绝对分数评估，以及GPT‑2架构模型；

**📊 数据集**

在事实归因使用Wikipedia摘要数据集，在风格归因使用Project Gutenberg文学文本；

**📈 对比分析**

与随机、最佳匹配检索、文本嵌入检索等基线方法比较，使用tail‑patch绝对分数作为评估指标，实验显示在各k值下均优于基线，尤其在风格归因中性能更为突出；

**⚠️ 局限性**

计算成本高，需要两次完整训练集遍历；未考虑样本间交互效应；token级归因仅基于损失变化，未验证因果作用。

---

## 479. Scene-Centric Unsupervised Video Panoptic Segmentation

**arXiv ID:** 2606.04925 | [PDF](https://arxiv.org/pdf/2606.04925v1)

**作者:** Christoph Reich `[一作]` (TU Munich), Stefan Roth `[通讯]` (TU Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了首个无监督视频全景分割（VPS）方法，能够在不使用人工标注的情况下实现对视频中所有实例和语义区域的检测、分割与跟踪。

**💡 创新点**

核心创新包括：
• 仅利用单目视频通过自监督深度、光流和DINO特征生成时空一致的伪标签；
• 引入 Video DropLoss 以在稀疏伪标签上进行监督，同时允许模型学习未覆盖的静态实例；
• 设计自增强视频复制粘贴增强策略提升对小目标的检测与跟踪；
• 制定统一的无监督 VPS 评估协议，并构建四个竞争基线。

**🔧 技术方法**

技术细节：自监督视觉表示（DINO），SMURF 光流，DynamoDepth 深度与运动分割，区域生长、k-means 聚类、CRF 平滑、Hungarian 匹配，Video DropLoss，视频复制粘贴增强。

**📊 数据集**

实验数据集：Cityscapes‑VPS（训练/验证），KITTI‑STEP，Waymo，MOTS（OOD）。

**📈 对比分析**

与四个基线（DepthG+VideoCutLER、U2Seg+SORT、CUPS+SORT(立体)、CUPS+SORT(单目)）比较，模型在 Cityscapes‑VPS 上 STQ 提升 12.3%（最高 22.2%），跨域性能也优于所有基线；在 Label‑Efficient 设定下，10% 标注即可逼近全标注模型；与监督上限相比差距仅 2–3% 。

**⚠️ 局限性**

局限性：单目深度估计噪声影响伪标签质量；对静态物体的覆盖不足；在复杂动态遮挡或极端光照下伪标签可能不够准确；模型对超长视频的实时性与存储仍需进一步优化。

---

## 480. TaDA: Calibrated Probe Gating for Task-Domain LoRA Merging

**arXiv ID:** 2606.05016 | [PDF](https://arxiv.org/pdf/2606.05016v1)

**作者:** Huy Quoc To `[一作]` (Deakin University), Ming Liu `[通讯]` (Deakin University)

**通讯引用:** 15614 | [OpenAlex ID](https://openalex.org/A5100347785)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种无训练的算法TaDA，用来将任务LoRA和域LoRA合并成单一模型。

**💡 创新点**

发现任务与域适配器在Transformer深度上存在一致的非对称性，利用校准探针进行层级加权，并在子空间层面过滤冲突特征。

**🔧 技术方法**

使用校准探针激活比率、Sigmoid门控、SVD分解、子空间重叠过滤、按组件权重融合等技术，最终输出标准rank-r LoRA。

**📊 数据集**

在Llama-2-7B上六个科学QA数据集（MedMCQA、MedQA-USMLE、ARC-Challenge、SciQ、MMLU-CS、MMLU-Science）以及ViT-L/16上六个图像分类数据集（ImageNet、CIFAR-100、PathMNIST、DermaMNIST、EuroSAT、DTD）进行评估。

**📈 对比分析**

与线性、TIES、DARE、Task Arithmetic等九个基线对比，TaDA在所有六个文本和六个视觉基准上均取得最高平均准确率，文本平均+3.6pp，视觉平均85.9%，明显优于DARE-TIES等。

**⚠️ 局限性**

需要用户提供域相关探针样本，合并时一次性SVD成本约31分钟，且仅评估了两类任务-域配对，未来需验证更广泛的适配器组合。

---

## 481. Depth-Attention: Cross-Layer Value Mixing for Language Models

**arXiv ID:** 2606.05014 | [PDF](https://arxiv.org/pdf/2606.05014v1)

**作者:** Boyi Zeng `[一作]` (Shanghai Jiao Tong University), Zhouhan Lin `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6277 | [OpenAlex ID](https://openalex.org/A5024900991)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Transformer 解码器中提出 Depth-Attention，利用层内查询对先前层的键进行注意力，并将其值混合进当前层的值，从而实现跨层信息选择。

**💡 创新点**

创新点是把跨层选择集成到自注意力模块内部，使用已有的查询键和值缓存，无需额外参数或持久状态。

**🔧 技术方法**

采用了自注意力、Depth‑Attention、稀疏层级选择（stride）、以及键/值混合策略等技术。

**📊 数据集**

在 Qwen3 风格的 1.5B、3B 以及 360M~3B 的模型上使用 Pile 数据集进行预训练。

**📈 对比分析**

与 Vanilla Transformer 及 mHC、Attention Residuals、DenseFormer 等方法比较，Depth‑Attention 在 Pile perplexity 与多任务下游准确率均优于基线，且计算和内存开销几乎相同（+0.01% FLOPs，+1.2% 生成时间）。

**⚠️ 局限性**

局限性包括未实现融合内核导致训练/推理开销略高、仅验证到 3B 规模和 32B 训练数据，且仅使用值混合和固定步幅的层级选择，未探索学习型源选择或更丰富的跨层交互。

---

## 482. COSMO: O-RAN-Based Service Management and Orchestration for Cross-Technology Multi-Tenant Radio Access Networks

**arXiv ID:** 2606.05012 | [PDF](https://arxiv.org/pdf/2606.05012v1)

**作者:** M. Catalan-Cid `[一作]`, D. Camps-Mur `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了COSMO平台，支持跨技术（5G NR、LTE、Wi‑Fi）多租户无线接入网的统一管理、编排与智能闭环控制；

**💡 创新点**

创新点包括：① 网络块（network chunk）与网络服务（network service）两种管理原语，实现在不同技术间的资源共享与分配；② 在O‑RAN SMO与Non‑RT RIC框架下实现跨技术资源编排、SLA 保障与动态资源调度；③ 基于COSMO SDK实现的多RAT SLA‑based slicing rApp，可在不同RAT上动态调整资源以满足全局SLA；

**🔧 技术方法**

使用了NETCONF/REST/YANG、Kubernetes（CRD、operator）、RabbitMQ、Prometheus/ICS（R1）、O‑RAN SMO与Non‑RT RIC组件、OSC参考实现、Python SDK、FlexRAN、OAI、Amarisoft、Linux Wi‑Fi等技术栈；

**📊 数据集**

实验数据来源于自建试验平台，包括Amarisoft 5G NR、OAI LTE、PC Engines Wi‑Fi AP，配合Open5G Core与自定义Prometheus Exporter，未使用公开数据集；

**📈 对比分析**

通过对MO函数（节点配置、和解）与Non‑RT RIC函数（生产者-消费者）进行可扩展性基准，CPU占用与超时延随节点/消费者数量增加而升高；SLA‑based rApp实验显示SLA违规率由约21%下降至<10%，且在动态负载下误差<2%，证明平台在资源利用与SLA合规性方面表现优异；

**⚠️ 局限性**

局限性：仅实现Non‑RT闭环，未集成Near‑RT RIC；5G NR节点缺乏动态PRB分配能力；单一Prometheus后端导致Telemetry瓶颈；对不同厂商实现的兼容性受限，未来需加入AI/ML与Near‑RT控制。

---

## 483. PhysDox: Benchmarking LLMs on Physical Feasibility Auditing of Physiological Sensing Protocols

**arXiv ID:** 2606.05003 | [PDF](https://arxiv.org/pdf/2606.05003v1)

**作者:** He Liu `[一作]` (Donghua University), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1259 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PhysDox基准，用于评估大语言模型在生物医学实验方案的物理可行性审计；

**💡 创新点**

首次将实验方案的可执行性划分为三状态（有效、次要问题、致命违规）并构建专门的违规类型分类，揭示模型的“脚手架偏差”现象；

**🔧 技术方法**

采用大语言模型（GPT‑5.5、Claude Opus 4.7、DeepSeek V4 Flash/Pro、MiMo v2.5/Pro）并结合四种推理策略（零射、链式思考、自一致性、工具增强）进行评测；

**📊 数据集**

使用专家手工标注的683条Gold样本和5,000条Silver样本，覆盖六大生理/非接触传感领域；

**📈 对比分析**

与多数基线（关键词启发式、随机、占优类别）对比，Stage‑1宏F1最高仅53.0，终端诊断准确率在70%以下，显示模型在严重性校准与致命违规检测上存在显著瓶颈；

**⚠️ 局限性**

局限性包括仅覆盖生物医学传感领域、数据为人工合成协议、缺乏多轮交互与外部仿真验证，且评估结果受当前模型规模与技术进展影响。

---

## 484. What Can Eye Gaze Teach Us About Real-World Cycling? Insights From the Oxford RobotCycle Project

**arXiv ID:** 2606.04989 | [PDF](https://arxiv.org/pdf/2606.04989v1)

**作者:** Benjamin Hardin `[一作]` (University of Oxford), Lars Kunze `[通讯]` (University of West of England)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究在英国牛津市使用可穿戴眼动追踪眼镜，对骑行者在不同道路基础设施、交叉口和事件下的眼动行为进行实时记录与分析。

**💡 创新点**

创新点在于首次将现场可穿戴眼动追踪技术与真实骑行场景相结合，系统性评估了不同道路类型和交叉口对骑行者认知负荷的影响。

**🔧 技术方法**

采用Meta Aria研究眼镜进行眼动追踪，结合I‑VT算法提取注视时长和水平散布等指标，并使用Welch's ANOVA与Games‑Howell后验检验进行比较。

**📊 数据集**

数据集来源于2024‑2025年在牛津三条骑行路线（北、中心、南）收集的12次骑行记录，包含注视点、路段类型、事件标签等信息。

**📈 对比分析**

通过对不同基础设施、交叉口、事件和路线的注视时长与散布进行方差分析和后验比较，发现例如车辆道与公交道、环形交叉口与信号交叉口等在眼动特征上存在显著差异；统计显著但效应量一般至中等。

**⚠️ 局限性**

局限性包括眼动指标无法区分不同事件类型，缺乏对注视对象的识别，未结合IMU或对象检测等多模态数据，且样本量仅为单一骑行者且受限于牛津市道路环境。

---

## 485. What Type of Inference is Active Inference?

**arXiv ID:** 2606.04935 | [PDF](https://arxiv.org/pdf/2606.04935v1)

**作者:** Wouter W. L. Nuijten `[一作]`, Bert de Vries `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将主动推理中的期望自由能（EFE）转化为可变分的变分自由能（VFE）形式，明确规划与认知修正，构建基于通道重参数化的消息传递规划框架，并在三种网格世界环境上验证其有效性。

**💡 创新点**

① 将规划修正（将期望-效用变为策略优化）与认知修正（把边缘VFE变为EFE）分离并给出精确的熵修正公式；② 通过通道重参数化将熵修正化为可变分的消息传递因子；③ 在不同观测不确定性场景下揭示观测通道在EFE规划中的关键作用。

**🔧 技术方法**

变分推理、贝叶斯因子图（Forney因子图）、Bethe近似、熵修正、通道重参数化、Loopy BP、迭代消息传递与数值收敛控制。

**📊 数据集**

三种离散网格世界：Frozen Lake（全局决定性观测）、RockSample（局部决定性观测）和Wumpus World（局部启发式观测），用于模拟不同观测范围与分辨率的认知需求。

**📈 对比分析**

与标准BP、VBP（交叉熵规划）、RM-MP（仅动力学通道）、Nuijten-MP（交替逼近）等方法对比。主动推理方法在全局/局部决定性环境中表现优于基线；在局部启发式环境中，完整的EFE规划（AIF-MP）成功率显著高于其它方法，验证了观测通道的重要性。

**⚠️ 局限性**

① 需要手动调节阻尼参数 λ 以保证收敛；② 对收敛性缺乏理论保证，需进一步研究；③ 仅在离散、可精确因子评估的网格世界验证，未探讨近似推理或更大规模连续环境的适用性。

---

## 486. The local complexity of certifying parity

**arXiv ID:** 2606.04934 | [PDF](https://arxiv.org/pdf/2606.04934v1)

**作者:** Nicolas Bousquet `[一作]` (CNRS), Sébastien Zeitoun `[通讯]` (CNRS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在分布式网络中本地证明的证书复杂度，特别是判断节点数是否为偶数或满足可除性属性的情况；

**💡 创新点**

提出了在不同模型（有唯一标识符/匿名、不同半径）的情况下，是否存在常数复杂度的本地证明的理论边界，并首次将Ramsey理论与图冲突自由着色结合到证明复杂度上；

**🔧 技术方法**

使用了局部证明框架、拓扑层次化着色、冲突自由着色、Folkman‑Rado‑Sanders有限并集定理等组合技巧；

**📊 数据集**

论文仅基于理论构造的特殊图结构（如幂集图G_m和其变体），不涉及实测数据集；

**📈 对比分析**

对比方法主要是理论上上下界的比较，提出了Ω(log∗n)的下界与常数/对数上界的对照，表明存在显著的复杂度差距；

**⚠️ 局限性**

局限性在于下界依赖于Ramsey函数的极端上界，无法得到更精确的下界，且在多数图类上仍未给出最优上界。

---

## 487. Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning

**arXiv ID:** 2606.04923 | [PDF](https://arxiv.org/pdf/2606.04923v1)

**作者:** Xuekang Wang `[一作]` (Tsinghua University), Xiaozhi Wang `[通讯]` (Tsinghua University)

**通讯引用:** 7109 | [OpenAlex ID](https://openalex.org/A5100775960)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可控的奖励黑客环境CHERRL，通过注入已知偏差来观察和定位Rubric-based RL中的奖励黑客行为。

**💡 创新点**

首次通过双评审器将黄金奖励与可控偏差分离，提供可观测的奖励偏离和精确的黑客启动点，为奖励黑客的可重复实验平台奠定基础。

**🔧 技术方法**

采用LLM-as-a-Judge双评审框架、偏差注入、奖励差距统计、赔率比计算以及基于工具的LLM代理检测器RHDA等技术。

**📊 数据集**

在VerInstruct和HealthBench两大开放式任务数据集上训练Qwen3-4B，并评估多种偏差类型。

**📈 对比分析**

将RHDA与Claude Code和固定CoT监控器对比，结果显示RHDA在六个受控实验中实现了更低的起点误差和更高的检测准确率。

**⚠️ 局限性**

受限于计算资源仅在Qwen3-4B上验证，且检测器尚未提供修复措施，仅能定位黑客行为。

---

## 488. SURF: Separation via Unsupervised Remixing Flow

**arXiv ID:** 2606.04921 | [PDF](https://arxiv.org/pdf/2606.04921v1)

**作者:** Henry Li `[一作]` (Google), John R. Hershey `[通讯]` (Google DeepMind)

**通讯引用:** 11376 | [OpenAlex ID](https://openalex.org/A5112763337)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种名为 SURF 的无监督单通道源分离框架，利用流匹配（Flow Matching）与自监督 remixing（ReMixIT / Self‑Remixing）结合，从混合信号中直接学习源分离。

**💡 创新点**

创新点在于：① 将监督式流匹配（FLOSS）与自监督 remixing 结合，形成“教师-学生”结构；② 将流匹配的速度场与回归目标对齐，弥合两者结构差异；③ 给出 Wake‑Sleep 视角的理论解释。

**🔧 技术方法**

采用流匹配（Flow Matching）技术、噪声条件分数网络（NCSN）架构、教师模型与学生模型的指数移动平均更新，以及自监督 remixing 损失（ReMixIT、Self‑Remixing）。

**📊 数据集**

在图像域使用 MNIST、CIFAR‑10 生成的重叠混合图像；在音频域使用 Libri2Mix、AudioSet 以及评估集 FUSS 进行实验。

**📈 对比分析**

与无监督基线 MixIT、ReMixIT、Self‑Remixing 以及监督式回归和流匹配模型比较。SURF 在 PSNR/SSIM、LPIPS、FID、SI‑SDR、ESTOI、PESQ 等指标上均显著优于现有无监督方法，接近监督式流匹配的性能。

**⚠️ 局限性**

局限性包括：对源数较多的复杂混合仍表现不足；无监督训练易产生幻觉（hallucination），可能放大数据偏差；以及对极端噪声/领域漂移的鲁棒性尚待进一步验证。

---

## 489. DAR: Deontic Reasoning with Agentic Harnesses

**arXiv ID:** 2606.05009 | [PDF](https://arxiv.org/pdf/2606.05009v1)

**作者:** Guangyao Dou `[一作]` (Johns Hopkins University), Benjamin Van Durme `[通讯]` (Johns Hopkins University)

**通讯引用:** 8766 | [OpenAlex ID](https://openalex.org/A5075825791)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种“Agentic Deontic Reasoning”框架，即通过 harness 让 LLM 按需读取法条文件，进行逐步推理。

**💡 创新点**

创新点在于将传统一次性长文本推理转变为交互式工具调用式推理，并系统评估该方法在高难度 DeonticBench 子集上的效果。

**🔧 技术方法**

使用的技术包括：Terminal-based harness（Terminus-2 与 Terminus-KIRA）、shell 读取工具、Python 计算、Harbor 框架以及 OpenRouter API 调用。

**📊 数据集**

采用的基准数据集是 DeonticBench，包含 SARA (税法)、USCIS-AAO (移民行政) 与 Airline (航空行李政策) 三类难度高的任务。

**📈 对比分析**

比较方法是将直接推理（将完整法条与事实放入 prompt）与两种 Agentic harness 在同一任务、相同模型上对比，记录准确率、macro-F1 与 token 消耗。结果显示：前沿模型在 Harness 下可提升 15–30%（如 GPT‑5.2 30%→60%），但开源模型在同一 Harness 下准确率下降并且 token 消耗增加 4 倍；在 SARA-Numeric 任务上，Open-source Qwen3.5-35B 由 34% 降至 11%。

**⚠️ 局限性**

局限性包括：1) 对非常长的法条仍需大量读取，前沿模型亦可能出现 token 过度消耗；2) 仅评估 DeonticBench，未覆盖更广泛的法规领域；3) harness 设计针对通用任务，缺乏专门针对法规检索的高效工具；4) 仅测试了有限的 harness，未来可能有更优化的 harness 能改善弱模型表现。

---

## 490. GARL: Game-Theoretic Reinforcement Learning for Multi-Agent Strategic Prioritisation

**arXiv ID:** 2606.05002 | [PDF](https://arxiv.org/pdf/2606.05002v1)

**作者:** Yuxiao Ye `[一作]` (Tsinghua University), Zhiyuan Liu `[通讯]` (Tsinghua University)

**通讯引用:** 45676 | [OpenAlex ID](https://openalex.org/A5100320723)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了基于博弈论的强化学习框架 GARL，用于多智能体战略优先排序任务。

**💡 创新点**

将战略优先排序形式化为两阶段博弈（议程分配+仲裁），并将博弈效用直接转化为角色特定的 RL 奖励，减少对人工标注的依赖。

**🔧 技术方法**

采用多智能体强化学习（自我博弈与 KL 平衡策略更新）、Token‑level REINFORCE++、LLM 生成与评估以及外部 LLM 预计算效用。

**📊 数据集**

使用 LexIssue 争议问题排序基准（约600个案例），以及 LawBench 法律任务集和 GameBench 策略游戏集进行评估。

**📈 对比分析**

与 GPT‑4、DeepSeek‑R1、Qwen‑3.5‑27B 等基线在 Recall@|G|、mAP 及 LawBench/ GameBench 指标上对比，GARL 在 LexIssue 上实现 mAP 提升至 90% 以上、在 LawBench 与 GameBench 上均带来显著或可观提升。

**⚠️ 局限性**

仅适用于固定候选集的多智能体优先排序，效用设计仍需任务特定；实验仅覆盖法律争议排序，需在更多领域验证。

---

## 491. Multi-Camera AR Guidance System for Surgical Instrument Handling and Assembly: Investigating Workload and Efficiency

**arXiv ID:** 2606.04992 | [PDF](https://arxiv.org/pdf/2606.04992v1)

**作者:** Shiyu Li `[一作]` (Technical University of Munich), Daniel Roth `[通讯]` (Technical University of Munich)

**通讯引用:** 5725 | [OpenAlex ID](https://openalex.org/A5057621778)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一个基于多摄像机6D姿态估计、无标记且可在HoloLens 2上实时显示的AR指导系统，用于手术器械的组装与处理，并在模拟手术中验证其效果。

**💡 创新点**

创新点在于：①使用纯合成数据训练的多摄像机无标记姿态估计网络，实现动态相机校准与层次融合；②结合层次姿态过滤与实时AR展示，在不需手工标记的前提下显著降低护士工作负荷和完成时间。

**🔧 技术方法**

采用的技术包括：YOLOX+DDC+RTMO姿态网络、RANSAC‑PnP求姿、层次姿态过滤、Unity+MRTK2 AR渲染、Azure Kinect与HoloLens 2传感器数据融合。

**📊 数据集**

使用的数据集为：通过BlenderProc生成的合成图像（包含结构光扫描的器械和托盘模型）作为训练数据，Ground Truth采用Optitrack MOTIVE和Primex摄像头标定的真实手术场景数据。

**📈 对比分析**

与CosyPose比较，ADD(-S) AUC分别达到44.34（托盘）/31.73（工具），明显优于CosyPose；实时推理约200 ms/帧；在用户研究中任务完成时间比纸质手册快21.3%（约4.76 min），NASA‑TLX工作负荷显著下降。

**⚠️ 局限性**

局限性包括：视场受限、手动实现指导文本、校准误差与延迟、对真实手术环境验证不足、缺乏多工具集的泛化与可扩展性。

---

## 492. Food-R1: A Unified Multi-Task Food Vision-Language Model with Reinforcement Learning

**arXiv ID:** 2606.04986 | [PDF](https://arxiv.org/pdf/2606.04986v1)

**作者:** Yu Zhu `[一作]` (Huazhong University of Science and Technology), Xinggang Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 33962 | [OpenAlex ID](https://openalex.org/A5037191476)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 CalorieBench‑80K 大规模食物计量与营养建议基准，并基于该基准与其他食物相关数据集构建了统一多任务食物视觉‑语言模型 Food‑R1，结合 Chain‑of‑Thought 监督和 Group Relative Policy Optimization 强化学习对模型进行后训练，显著提升多任务表现。

**💡 创新点**

创新点包括：①首次构建包含 CoT 计量推理和饮食建议的食物图像基准；②在多任务学习框架中引入 CoT 归纳蒸馏，使模型能够逐步推理；③使用 GRPO 对参考策略进行强化学习，提升推理路径的可靠性与准确性。

**🔧 技术方法**

技术方法包括：大规模监督微调（SFT）+ CoT 监督；基于 Qwen3‑VL‑8B 的全参数微调；Group Relative Policy Optimization (GRPO) 强化学习后训练；使用多任务数据融合（包括 Food‑101、VireoFood‑172、Recipe1M、Nutrition5k 等）提升跨任务泛化。

**📊 数据集**

使用数据集：CalorieBench‑80K（核心计量与建议数据）、MM‑Food‑100K（过滤来源）、Food‑101、VireoFood‑172、Recipe1M、Nutrition5k、FoodDialogues 等辅助任务数据。

**📈 对比分析**

与多种基准模型（Gemini‑2.5‑flash、InternVL3‑8B、LLaVA‑7B、Qwen2.5‑VL‑72B、Qwen3‑VL‑8B、GPT‑4o‑mini、FoodLMM、FoodLMM‑s1）进行对比，Food‑R1 在 CalorieBench‑80K 的计量任务上 MAE 下降至 42.56，RMSE 112.89，R² 0.81，超过所有对照模型；在饮食建议、食物分类、配料识别、配方生成和营养估计等任务也均实现了最佳或接近最佳性能。

**⚠️ 局限性**

局限性：①依赖人工与大模型审核的高质量标注，标注成本高；②强化学习阶段仅对结构化任务使用，无法覆盖开放式生成任务；③在极端混合菜品或少见食材的推理准确性仍有提升空间。

---

## 493. Sequential Data Poisoning in LLM Post-Training

**arXiv ID:** 2606.04929 | [PDF](https://arxiv.org/pdf/2606.04929v1)

**作者:** Jack Sanderson `[一作]` (University of Chicago), Yiwei Lu `[通讯]` (University of Ottawa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了面向LLM后期训练的顺序数据中毒威胁模型，研究了不同阶段（SFT、DPO、RM/PPO）被多名攻击者分别污染时的交互效果。

**💡 创新点**

创新点在于揭示了“单攻击者幻觉”——单阶段评估会低估整体风险，并证明多阶段协同攻击可以产生叠加或互补的强大后门；同时首次系统比较了不同模型规模和架构在该威胁模型下的脆弱性。

**🔧 技术方法**

使用了监督微调（SFT）、直接偏好优化（DPO）、强化学习（PPO）以及奖励模型训练等技术，并通过触发词（trigger）进行标签翻转攻击。

**📊 数据集**

实验使用了 Llama‑3 8B、Qwen3 1.7B/4B/8B 等大模型，并利用公开数据集如 LLM‑LAT/harmful‑dataset、tatsu‑lab/alpaca 进行 SFT，Anthropic/hh‑rlhf（子集）用于偏好数据。

**📈 对比分析**

通过对比单阶段与多阶段攻击的攻击成功率（ASR）和奖励模型分数分布，发现：在 SFT→DPO 管道中，分配到两个阶段的毒化预算比单独集中在某一阶段更能快速且高效地实现 100% ASR；在 SFT→PPO 管道中，单独任何一阶段均无法成功，但两者协同即可实现显著攻击，且更大模型容量更易成功。

**⚠️ 局限性**

局限性包括：实验仅覆盖有限的模型规模和架构；触发词固定，未考虑自适应触发；未给出理论分析，主要基于经验实验；以及未探讨预训练阶段与后期训练阶段毒化的交互。

---

## 494. GoldenFloat: A Phi-Derived Static-Split Floating-Point Family from GF4 to GF256 with a Lucas-Exact Integer Identity

**arXiv ID:** 2606.05017 | [PDF](https://arxiv.org/pdf/2606.05017v1)

**作者:** Dmitrii Vasiliev `[一作]` `[通讯]` (Independent researcher), Dmitrii Vasiliev (Independent researcher)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于黄金比例的多宽浮点格式 GoldenFloat，并发布了闭合规则的 RTL 生成器、整数基 Lucas‑exact 累加器和合规性检查器；在 GF16 上完成了 FPGA 验证并提交了 Tiny Tapeout；同时对该格式在真实文本语料库上的比特/字节(BPB)性能进行了评估。

**💡 创新点**

采用单一闭合规则 e=round((N-1)/φ²) 生成全宽阶梯的静态分割浮点格式，并利用 φ 的代数恒等式（Lucas 识别）实现整数基累加路径；将这一方法与现有 Posit、Takum 等宽度跨度格式做对比。

**🔧 技术方法**

利用黄金比例编码、Lucas 数列恒等式、闭式规则、Verilog RTL 生成、持续集成差分检测、Tiny Tapeout 芯片、FPGA 验证以及 Corona 合规性 oracle。

**📊 数据集**

在 IGLA RACE 的真实文本语料库上进行比特/字节(BPB)评估，并使用内部测试向量验证乘法器的正确性。

**📈 对比分析**

通过比特/字节(BPB)对比、FPGA 最高频率(323 MHz GF16)以及持续集成差分测试进行评估；当前结果显示 GF16 在 323 MHz 下通过 35/35 测试，但在与 fp16 的公平硬件对比中仍缺乏充分证据，性能优势尚未确定。

**⚠️ 局限性**

未验证所有宽度的硅片、缺乏公平的 fp16 对比实验、整数累加器实现未公开完整细节、规则唯一性尚未完全证实、部分硬件存在乘法器宽度缺陷，导致整体结论仍为开放假设。

---

## 495. M$^3$Eval: Multi-Modal Memory Evaluation through Cognitively-Grounded Video Tasks

**arXiv ID:** 2606.05008 | [PDF](https://arxiv.org/pdf/2606.05008v1)

**作者:** Jie Huang `[一作]` (Peking University), Yiwu Zhong `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了 M^3Eval，一个基于认知心理学的多模态视频记忆评测框架和基准，涵盖四个记忆维度的任务。

**💡 创新点**

创新点在于将认知心理学实验范式迁移至视频领域，系统化分离记忆容量、精度、鲁棒性和符号化四个维度，并构造可控视频任务。

**🔧 技术方法**

采用多模态大模型（Gemini、GPT、Qwen、InternVL 等）与视频 QA 任务，结合自动化问题生成、注意力机制评估等技术。

**📊 数据集**

选取了五大公开视频数据集（HourVideo、Video-MME、LVBench、InfiniBench、CrossVid）构建约 403 小时、451 段视频与 2403 道问题。

**📈 对比分析**

与多款开源与专有多模态模型及人类基准对比，发现模型在并行流分辨、干扰鲁棒性、时序组织及符号化记忆方面均低于人类，且不同模型在干扰方向表现相似。

**⚠️ 局限性**

限制在于任务仍基于有限的手工与自动生成问题，缺乏对更复杂语言推理和长时序记忆的深入探索，以及对真实世界噪声与多样性的适应性评估。

---

## 496. Demo: BeGREEN Intelligence Plane for AI-driven Energy Efficient O-RAN management

**arXiv ID:** 2606.05000 | [PDF](https://arxiv.org/pdf/2606.05000v1)

**作者:** M. Catalan-Cid `[一作]` (i2CAT Foundation), J. Armstrong `[通讯]` (Ericsson)

**通讯引用:** 55 | [OpenAlex ID](https://openalex.org/A5113478054)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 BeGREEN Intelligence Plane，用 AI 驱动的 rApps/xApps 实现 O-RAN 网络的能源效率管理，自动控制基站开关机。

**💡 创新点**

创新点在于将 AI Engine 与 R1 接口 DME 结合，利用 Energy Score/Rating 评估并生成 A1 能源节约策略，并通过 AIA rApp 将模型推理与控制解耦，实现实时能源优化。

**🔧 技术方法**

采用了 O-RAN 架构、MLRun、Nuclio、Energy Score/Rating、KPM、A1、E2 接口、Near‑RT RIC、Non‑RT RIC、Accelleran dRAX、Viavi TeraVM AI RSG、Grafana 等技术。

**📊 数据集**

使用 Viavi TeraVM AI RSG 生成的 3GPP 兼容 KPM 数据作为训练和测试数据集。

**📈 对比分析**

通过对比能源节约前后在 5G SA 环境中的能源消耗，演示能耗可降低至 60%（节约 40%），同时保持 QoS，未影响网络性能。

**⚠️ 局限性**

局限在于仅在模拟环境验证，真实多运营商多设备场景下需进一步验证模型泛化和部署复杂度。

---

## 497. New Benchmarking Shows Limited Generalization Power of TCR Antigenic Epitope Prediction Models

**arXiv ID:** 2606.04994 | [PDF](https://arxiv.org/pdf/2606.04994v1)

**作者:** Yiming Liao `[一作]` (University of Maryland, Baltimore County), Keke Chen `[通讯]` (University of Maryland, Baltimore County)

**通讯引用:** 2325 | [OpenAlex ID](https://openalex.org/A5002572745)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个严格的基准框架，利用真正未见的高通量TCR‑抗原映射数据评估TCR‑epitope预测模型的泛化能力。

**💡 创新点**

提出两类全新未见数据集（TetTCR‑SeqHD和Fingerprinting），并引入去重、严格负样本生成及多步骤数据预处理策略。

**🔧 技术方法**

采用多模型推断、宏观AUC0.1评价、Spearman相关性分析以及去重和负样本控制等技术手段。

**📊 数据集**

使用TetTCR‑SeqHD、IMMREP23以及自制的TCR pMHC tetramer突变Fingerprinting数据集。

**📈 对比分析**

通过宏观AUC0.1对模型进行横向比较，结果显示在病毒/已知epitope上AUC0.1约为0.5–0.67，但在自体/未知epitope上性能接近随机。

**⚠️ 局限性**

模型普遍缺乏对未见epitope的泛化能力，受限于训练数据偏倚、负样本设计不足和抗原空间不平衡。

---

## 498. AlphaQ: Calibration-Free Bit Allocation for Mixture-of-Experts Quantization

**arXiv ID:** 2606.04980 | [PDF](https://arxiv.org/pdf/2606.04980v1)

**作者:** Wanqi Yang `[一作]` (Max Planck Institute for Intelligent Systems), Shiwei Liu `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AlphaQ，一种基于混合精度量化的 Mixture‑of‑Experts (MoE) 模型无标注（无校准）位宽分配方法；

**💡 创新点**

核心创新是利用 Heavy‑Tail Self‑Regularization (HT‑SR) 理论，通过权重矩阵谱的 heavy‑tailed 估计（PL_Alpha_Hill）直接衡量专家重要性，并在全局比特预算下通过 ILP 最优分配；

**🔧 技术方法**

技术包括谱重采样 (FARMS)、Hill 估计、量化噪声建模、整数线性规划求解；

**📊 数据集**

使用 WikiText2、PIQA、ARC‑Easy、ARC‑Challenge、HellaSwag、WinoGrande、MMLU、MATH、CEval 等公开基准；

**📈 对比分析**

与 Uniform、PMQ、BSP、Hessian、AFG、DynaMo 等基线比较，AlphaQ 在相同比特预算下在零样本任务平均准确率上提升约 1–3%（例如 Qwen1.5‑MoE 3.5‑bit 与 BF16 接近），显著压缩权重（4.4×）并提升推理速度（至 1.7×）；

**⚠️ 局限性**

局限包括：需额外离线计算 α 值（耗时数分钟）、对极大规模模型的验证有限、量化后性能仍受模型架构影响，且未解决所有稀疏激活分配细节。

---

## 499. AdaKoop: Efficient Modeling of Nonlinear Dynamics from Nonstationary Data Streams with Koopman Operator Regression

**arXiv ID:** 2606.04930 | [PDF](https://arxiv.org/pdf/2606.04930v1)

**作者:** Naoki Chihara `[一作]` (University of Osaka), Yasushi Sakurai `[通讯]` (University of Osaka)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种高效的流式算法 AdaKoop，用来在非平稳数据流中实时建模并预测复杂非线性动力学。

**💡 创新点**

创新点在于：① 通过双视图（原始观测与 RKHS 特征）构建概率增强状态空间模型，将非线性动力学映射为可解的线性系统；② 结合 Koopman 运算符理论实现快速闭式更新；③ 采用统计假设检验和在线 EM 进行自适应模式切换与参数更新，保持 O(1) 的时间复杂度。

**🔧 技术方法**

主要技术包括：Koopman 运算符理论、RKHS 与核方法、双视图状态空间模型、期望最大化（EM）算法、CUSUM 检测、Kalman 滤波与 Rauch–Tung–Striebel 平滑、字典稀疏化与剪枝。

**📊 数据集**

在 71 个真实世界混沌动力学基准数据集上进行实验，涵盖气象、生物、物理等多领域。

**📈 对比分析**

与 ModePlait、WPMixer、PAttn、OneNet、Koopa、Streaming KAF 等七种先进方法对比，AdaKoop 在平均 MAE/MSE 上显著优于所有基线，并且在实时推理时的计算时间几乎是最短的。

**⚠️ 局限性**

局限性包括：① 对核函数的选择敏感，RBF 是最优但会增加内存；② 当模式切换频繁时，CUSUM 可能出现误报；③ 仍需手动设定阈值 ν、γ 等超参，对不同场景需要调优。

---

## 500. SAID: Accelerating Diffusion-Based Language Models via Scaffold-Aware Iterative Decoding

**arXiv ID:** 2606.04974 | [PDF](https://arxiv.org/pdf/2606.04974v1)

**作者:** Na Li `[一作]` (Peking University), Hao Tang `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为SAID的框架，通过重新分配计算资源来加速扩散大型语言模型（DLLMs）的推理过程。

**💡 创新点**

SAID的创新点在于采用了支架感知的迭代解码策略，优先处理支架令牌以建立粗略的语义结构，然后用更少的步骤完成可预测的细节令牌，同时引入了信心分层生成（CHLG）机制。

**🔧 技术方法**

使用了支架感知的迭代解码框架和信心分层生成机制，适应了块级扩散解码。

**📊 数据集**

在LLaDA-8B和LLaDA 1.5数据集上进行了实验，涵盖数学、编码和知识基准。

**📈 对比分析**

与标准LLaDA解码相比，SAID在所有基准测试中均显著提高了推理效率，最大加速比达到9.1倍，同时保持了竞争力的生成质量。

**⚠️ 局限性**

SAID的局限性在于实验仅在英语基准上进行，未验证其在非英语或多语言生成任务中的有效性。此外，评估指标主要关注准确性，未能全面反映生成文本的流畅性和连贯性。

---

## 501. Dual-Stream MLP is All You Need for CTR Prediction

**arXiv ID:** 2606.04944 | [PDF](https://arxiv.org/pdf/2606.04944v1)

**作者:** Kesha Ou `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 Dual-Stream MLP（DS-MLP）框架：使用知识蒸馏把教师模型的显式交互知识迁移到主 MLP，同时加入平行 MLP 捕获隐式交互，并对两条流进行隐藏层和预测对齐。

**💡 创新点**

创新点：①仅用 MLP 就能学习复杂显式交互；②蒸馏 + 对齐实现教师与学生兼容；③平行 MLP 补偿显式交互不足，避免单一流主导；④统一对齐策略解决双流融合不平衡。

**🔧 技术方法**

技术：知识蒸馏、批归一化对齐、二阶段训练（蒸馏→对齐→整体优化）、MLP 主干、并行 MLP。

**📊 数据集**

使用公开 CTR 基准数据集：Criteo、Avazu、MovieLens。

**📈 对比分析**

与 FM、DeepFM、xDeepFM、GDCN、FinalMLP 等多种基线对比，DS-MLP 在三大数据集上均取得最高 AUC / 最低 LogLoss，AUC 提升约 0.1%–0.6% 相对最佳基线，且推理延迟保持可控。

**⚠️ 局限性**

局限：依赖教师模型；需要调节蒸馏权重与对齐权重；对非常稀疏或新特征的泛化受限；当前仅支持双流设计，扩展到多流尚未验证。

---

## 502. Geometry-Aware Distillation for Prompt Tuning Biomedical Vision-Language Models

**arXiv ID:** 2606.04922 | [PDF](https://arxiv.org/pdf/2606.04922v1)

**作者:** Tran Dinh Tien `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Omni-Geometry Knowledge Distillation (OGKD) 框架，利用冻结文本原型构建的类关系图，在视觉‑语言模型中注入几何结构，并通过全局与局部蒸馏损失实现仅训练提示词的少量参数下医学影像少样本分类。

**💡 创新点**

创新点：①基于文本原型构建固定类关系图，将语义结构直接融入教师分布；②设计两种蒸馏损失——全局几何感知蒸馏 (GAD) 与标签引导局部蒸馏 (LGD)，同时在全局图像标记和注意力聚焦的补丁级别上进行蒸馏；③保持模型参数极低，只更新提示词，避免大规模微调。

**🔧 技术方法**

使用技术包括：视觉‑语言预训练模型（CLIP/医学VLM）、冻结文本原型、类关系图软化、KL 散度蒸馏、SCCM 正则化、温度缩放、风险‑覆盖度量等。

**📊 数据集**

在 11 个常用医学影像数据集（覆盖肺炎、脑肿瘤、眼底、胸腔等多模态和十个解剖部位）上进行基线→新类别泛化、few‑shot 学习和风险‑覆盖度评估。

**📈 对比分析**

与 CoOp、BiomedCoOp、ProGrad 等最新提示学习/适配方法对比，OGKD 在 16-shot、few‑shot 与基线→新类别任务上平均提升 1.7%–2.8% 的准确率，风险‑覆盖曲线 AURC 明显下降，显示更稳健、可靠的预测。

**⚠️ 局限性**

局限性：①类关系图仅基于文本原型，可能忽略视觉语义差异；②需要手动调节图的温度与蒸馏强度；③目前仅验证分类任务，局部蒸馏对分割或多标签任务的适用性未知；④对极大类别数和稀疏标签的扩展尚未评估。

---

## 503. Toward Multi-Domain and Long-Tailed Quantization via Feature Alignment and Scaling

**arXiv ID:** 2606.04920 | [PDF](https://arxiv.org/pdf/2606.04920v1)

**作者:** Chin-Yuan Yeh `[一作]` (National Taiwan University), Ming-Syan Chen `[通讯]` (National Taiwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种针对多域数据和长尾分布的低位量化框架，通过CDF特征对齐和敏感度感知聚合实现鲁棒量化。

**💡 创新点**

创新点包括：①基于累计分布函数的特征对齐与对齐量化梯度下降（AQGD）解决多域分布差异；②敏感度感知权重聚合（SWA）平衡不同域的收敛速度；③条件CDF缩放、Levene检验驱动的均衡损失和logit调整，缓解长尾类别的方差与置信度失衡。

**🔧 技术方法**

使用的技术主要有：量化感知训练（QAT）、CDF映射、对齐量化梯度下降（AQGD）、敏感度感知聚合（SWA）、Levene方差齐性检验、均衡损失、logit置信度调整。

**📊 数据集**

实验数据集包括：CIFAR-10、CIFAR-100、SVHN、ImageNet、Office-31、Digits、SynDigits-LT、CIFAR-10-LT、CIFAR-100-LT。

**📈 对比分析**

与现有PTQ和QAT基线在单域2/4位量化下提升5–30%准确率；在Office-31和Digits的多域量化中2/4位提升3–68%准确率；在长尾设置下，γ=10/50/200时显著提升20–40%准确率。

**⚠️ 局限性**

局限性包括：仍需在更大模型（如ViT、扩散模型）与真实边缘设备上验证推理效率；依赖特征近似正态分布假设，对极端分布可能不稳健；实验仅在单GPU环境完成，缺乏分布式训练与部署细节。

---

## 504. From Agent Traces to Trust: Evidence Tracing and Execution Provenance in LLM Agents

**arXiv ID:** 2606.04990 | [PDF](https://arxiv.org/pdf/2606.04990v1)

**作者:** Yiqi Wang `[一作]` (Griffith University), Yanming Zhu `[通讯]` (Griffith University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了大语言模型（LLM）代理在执行复杂任务时产生的多源证据和执行轨迹，并系统性地提出了证据追踪与执行源迹的概念框架、分类维度与评估思路。

**💡 创新点**

创新点在于：①提出统一的原始关系与四维度（来源、单位、关系、粒度/时序）分类体系，整合检索、工具调用、记忆、环境交互及多代理等多源证据；②将 W3C PROV‑DM 与 OpenTelemetry 等成熟模型与 LLM 代理特有的语义单元（推理步骤、工具参数、内存项、环境观测）结合，形成可扩展的图结构；③阐明原始追踪如何服务于验证、归因、调试、安全、审计与恢复等信任函数，为跨任务评测提供统一视角。

**🔧 技术方法**

采用的技术包括：结构化日志与执行图的生成；基于 PROV 的实体-活动-代理模型；语义图关系（支持、派生、依赖、矛盾、失效、触发、更新等）；信息流与污点追踪机制（CaMeL、FIDES、NeuroTaint、Agent‑Sentry 等）来实现工具调用与参数的来源追踪；记忆写入、检索与版本化的原始链路记录；以及多代理通信图与运行时边界约束。

**📊 数据集**

论文并未提出新的数据集，而是汇总与引用了现有评测资源：RAG 评测（ALCE、RAGAS、RAGChecker、RAGTruth、SourceCheckup 等）、代理评测（AgentBench、WebArena、τ‑bench、ToolBench、ToolLLM）、记忆评测（MemoryBank、MemGPT、A‑MEM、Mem0）、多代理评测（AutoGen、CAMEL、MAST）以及安全与调试基准（InjecAgent、AgentDojo、ToolEmu、TRAIL、LADYBUG）。

**📈 对比分析**

本文并无实验比较，而是通过对比分析法把不同方向的研究归入同一框架，讨论已有方法在验证、归因、调试、安全、审计和恢复上的表现与局限，指出目前大多仅关注单一维度（如答案正确性或工具使用安全），缺乏端到端覆盖整个原始链路的统一评测标准。

**⚠️ 局限性**

限制主要包括：缺乏统一的原始 schema 与跨任务通用图格式，导致不同工作难以互通；端到端原始评测指标稀缺，导致难以量化整体可信度；隐私与可扩展性（长时记忆追踪、海量日志存储）仍是技术挑战；以及在现实部署中对实时原始追踪与执行时序控制的性能与可行性未得到充分验证。

---

## 505. Generalization of World Models under Environmental Variability for Vision-based Quadrotor Navigation

**arXiv ID:** 2606.05015 | [PDF](https://arxiv.org/pdf/2606.05015v1)

**作者:** Luca Zanatta `[一作]` (Norwegian University of Science and Technology), Kostas Alexis `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对基于DreamerV3的世界模型在视觉深度四旋翼导航中的鲁棒性进行系统评估，涵盖SSL预训练、RL微调以及真实世界闭环与开环部署。

**💡 创新点**

首次在不同环境随机性下跨环境验证世界模型，证明SSL阶段的重构性能是实地可部署性的可靠预测指标，并明确离散潜在维度与训练序列长度为关键超参。

**🔧 技术方法**

使用DreamerV3的RSSM架构进行自监督重构与奖励预测，随后在潜在空间进行RL训练（actor‑critic），并通过对抗式控制参数扰动评估政策鲁棒性。

**📊 数据集**

在AerialGym仿真环境中采集多种随机性（L1‑L4）下的深度+状态轨迹，后续在真实四旋翼上测试。

**📈 对比分析**

通过交叉环境验证（MSE/SSIM）、RL胜率/碰撞率及真实环境闭环/开环路径跟踪对比，结果显示SSL阶段重构表现最好的模型在真实环境中均成功通过窄缝（0.67m），而仅靠RL获胜率高的模型失败；并在开环条件下实现12段连通走廊的纯想象飞行。

**⚠️ 局限性**

仅针对大平面障碍物验证，未考察小物体或薄结构；开环想象的有效时域有限，较长预测周期误差累积显著；随机性定义主要局限于几何与控制级别，未涵盖更广泛的环境变化。

---

## 506. Plan, Watch, Recover: A Benchmark and Architectures for Proactive Procedural Assistance

**arXiv ID:** 2606.04970 | [PDF](https://arxiv.org/pdf/2606.04970v1)

**作者:** Kaustav Kundu `[一作]` (Meta Reality Labs), Seungwhan Moon `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个可穿戴 egocentric 视频的主动程序助手系统 PWR，包含首个 OOP（偏离计划）并配有恢复指导的数据集 EgoProactive、统一重标注的 5 大已公开数据的 Pro2Bench、将长时规划与实时交互解耦的 Planner‑Interaction 体系结构以及跨模型的后训练 recipe。

**💡 创新点**

创新点：①收集首个支持 OOP 且附带恢复指导的可穿戴视频数据；②将 Ego4D、Ego‑Exo4D、EPIC‑KITCHENS、HoloAssist、HowTo100M 五大数据集统一标注为“计划+每步干预标签”；③通过解耦长时规划与实时交互实现既能保持低延迟又能进行深度推理；④证明跨 LLM 家族的后训练 recipe 能在多种模型上显著提升性能。

**🔧 技术方法**

技术手段：基于视觉语言模型的双模型架构（Planner + Duplex Interaction）、每帧 2fps 的视频剪辑选择与计划嵌入、教师强迫（teacher‑forcing）训练、G‑Mean F1 与 PQS 评价指标、跨模型迁移学习。

**📊 数据集**

使用数据集：EgoProactive（700 只智能眼镜视频，9,935 评估实例，1,883 OOP 误差+恢复），Pro2Bench（40,008 评估实例，249,584 训练实例，覆盖 14 活动域，3 视角类型），并对 Ego4D、Ego‑Exo4D、EPIC‑KITCHENS、HoloAssist、HowTo100M 进行统一重标注。

**📈 对比分析**

比较方法：在 6 个前沿专有模型（Gemini 3.1 Pro、Claude Opus 4.6、GPT‑5.2）和 3 个公开模型（Qwen‑3‑VL‑235B、Qwen‑3.6‑VL‑27B、Llama‑4 Maverick）分别做零样本和 Fine‑Tune 后的评测。零样本时大多数模型仅达 <0.5 G‑Mean F1；Fine‑Tune+PWR 后，Llama‑4 达到 0.76 G‑Mean F1 / 0.63 PQS，Qwen‑3.6 达到 0.83 G‑Mean F1 / 0.47 PQS；在 OOP 检测上，Oracle 计划可达 99.6% 召回，PWR‑FT 约 78–90% 召回，恢复质量约 3–4 分。

**⚠️ 局限性**

局限性：OOP 仅在脚本化场景下评估，未测试自发错误；教师强迫导致训练时使用的真实计划与推理时使用的预测计划不一致；未考虑用户对系统过度依赖的安全性与公平性问题。

---

## 507. Potential-Guided Flow Matching for Vision-Language-Action Policy Improvement

**arXiv ID:** 2606.04968 | [PDF](https://arxiv.org/pdf/2606.04968v1)

**作者:** Yunpeng Mei `[一作]`, Gang Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种自引导流匹配策略，能够在混合质量机器人经验下同时生成动作片段和成功潜力，提升视觉-语言-动作（VLA）政策。

**💡 创新点**

创新点包括：①在流端点上增添成功潜力坐标，使同一流生成动作和评分；②解耦优势加权流匹配，仅对动作速度加权、对潜力均匀监督；③基于条件流匹配的单步边界估计实现低成本优势估计；④利用生成的潜力实现无外部评估器的最佳-K 推断。

**🔧 技术方法**

使用技术包括：条件流匹配（Conditional Flow Matching）、优势加权回归（Advantage‑Weighted Regression）、解耦损失设计、单步边界估计、最佳-K 自引导推断，及在多维动作空间的端到端训练。

**📊 数据集**

使用数据集：BEHAVIOR‑1K 5 个仿真长时程操控任务，以及 5 个真实双臂操控任务（Set‑Paper‑Roll、Pick‑Trash、Cube‑Stack、Transfer‑Food、Wipe‑Whiteboard）。每任务均采用 200 条专家演示与 100 条自主回放混合数据。

**📈 对比分析**

与完整行为克隆（BC）、过滤行为克隆（Filtered BC）、基于独立评论器的 IDQL、流式强化学习 FQL 等基线比较，实验表明该方法在仿真任务平均得分 0.46、成功率 39.6%，在真实任务平均得分 0.62、成功率 35.4%，与 IDQL 成功率相当但参数仅 1K、训练时间减少 38%。

**⚠️ 局限性**

局限性：需要人工阶段标签；稀疏终结信号仍难以在极长时程任务中进行有效信用分配；单步边界估计在有限网络上为近似；若无验证、失败检测与人机监督，潜在安全风险。

---

## 508. Clinical Assistant for Remote Engagement Link (CARE-link): A Web-Based Electronic Health Records Software for Managing Diabetes

**arXiv ID:** 2606.04952 | [PDF](https://arxiv.org/pdf/2606.04952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 509. From Prompt to Process: a Process Taxonomy and Comparative Assessment of Frameworks Supporting AI Software Development Agents

**arXiv ID:** 2606.04967 | [PDF](https://arxiv.org/pdf/2606.04967v1)

**作者:** Sanderson Oliveira de Macedo `[一作]` `[通讯]` (Federal Institute of Goias), Sanderson Oliveira de Macedo (Federal Institute of Goias)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对六个主流AI软件开发支持框架进行系统比较，提出了包含规范、上下文、角色、执行、验证与可移植性六维度的过程分类法，并在此基础上映射风险与研究议程。

**💡 创新点**

创新点在于将可跨代理可移植性作为核心维度，首次统一对绿灯项目与逆向文档工程两大场景下的框架进行跨维度评价，并提出针对性实证研究路线。

**🔧 技术方法**

采用定性研究方法：从官方文档、仓库与社区列表中进行定向检索，结合三分量表对每维度进行打分；使用GitHub API统计星标与最近推送来衡量框架的“活跃度”。

**📊 数据集**

主要数据来源为GitHub官方仓库（如Spec Kit、OpenSpec等）以及相关论文与灰色文献，截取2026年5月的星标与活动记录；未使用专门的公开数据集。

**📈 对比分析**

比较方法为六维度量表评分，累计分数揭示无框架能够全面覆盖所有维度，显示深度与可移植性之间的权衡；论文未给出量化性能指标，而是以评分与风险映射为主要结果。

**⚠️ 局限性**

局限性包括：样本挑选依赖星标与活跃度，可能遗漏新兴或低星框架；评分依赖单一评估者，缺乏交叉验证；缺乏独立实证验证框架实际效果；可移植性与风险评估仍停留在理论层面，未结合真实项目实验。

---

## 510. Fairness and Strategy-Proofness in Automated Market Makers

**arXiv ID:** 2606.04959 | [PDF](https://arxiv.org/pdf/2606.04959v1)

**作者:** Frank M. V. Feys `[一作]` `[通讯]` (Independent researcher), Frank M. V. Feys (Independent researcher)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文在自动化做市商（AMM）领域提出并证明了一个结构性不可能性：在多资产（n≥3）的加权乘积常数函数市场做市商（CFMM）族上，没有任何公平（满足 Arrow 核心假设）且策略无关的聚合规则；公平性会迫使聚合规则成为加权 Aitchison 中心，而策略无关性则只允许单一 LP 独裁者。

**💡 创新点**

创新点主要包括：① 将公平性（Pareto、匿名、连续、机制独立性等）与策略无关性统一到规则层面；② 通过 Aitchison 几何与欧氏几何的碰撞证明不可能性；③ 将 AMM 设计与外部贝叶斯意见池（logarithmic opinion pool）等价，拓展到概率聚合领域。

**🔧 技术方法**

主要技术手段包括：社会选择与机制设计框架、Arrovian 核心公理化、机制级 IIA 与对称性、Cauchy/共轭函数方程（乘法/加法协同）、Aitchison 简化对数比率坐标、欧氏空间策略无关性（median）与单独 LP 操纵论证。

**📊 数据集**

本文未使用任何实验数据集，全部以形式化数学证明为主。

**📈 对比分析**

由于结果为不可能性理论，没有具体性能对比；但作者通过与现有 AMM（Uniswap、Balancer）以及意见池（Genest 的 logarithmic pool）进行理论对应，说明目前实际部署的 AMM 都已放弃公平聚合的目标。

**⚠️ 局限性**

限制包括：仅适用于 n≥3 的加权乘积 CFMM，排除两资产情况、LMSR、Curve 的 StableSwap 等非可分离或平移不变的做市模型；假设 LP 偏好为 Aitchison 单峰；以及对权重函数仅要求连续与对称性。

---

## 511. NLLog: Lightweight, Explainable SOC Anomaly Detection via Log-to-Language Rewriting

**arXiv ID:** 2606.04957 | [PDF](https://arxiv.org/pdf/2606.04957v1)

**作者:** Samuel Ndichu `[一作]` (National Institute of Information and Communications Technology), Daisuke Inoue `[通讯]` (National Institute of Information and Communications Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套轻量级、可解释的安全运营中心异常检测管线——NLLog，将日志模板转换为自然语言WWS句子并用于异常检测；

**💡 创新点**

创新点在于：1）确定性、可审计的模板到语言重写；2）结合TF‑IDF加权的稠密句子向量；3）使用树模型与TreeSHAP实现可解释的句子级别证据；4）在CPU上实现10 ms/会话的低延迟；

**🔧 技术方法**

核心技术包括Drain3日志解析、MiniLM-L6句子编码、TF‑IDF加权池化、LightGBM等树模型、TreeSHAP归因；

**📊 数据集**

实验使用公开数据集HDFS、Blue Gene/L (BGL) 以及AIC Alert Data Set (AIT‑ADS)；

**📈 对比分析**

与复现的DeepLog、LogBERT等基线对比，NLLog在BGL达到97.7% F1、HDFS 99.8% F1，且在FPR≤2%预算下保持高召回；在AIT‑ADS上也实现高精度、低误报；

**⚠️ 局限性**

局限性包括：仅处理单会话、模板相同的日志；对序列或跨主机关联缺乏感知；易受填充/IDF毒化攻击；需要在部署前评估模板覆盖率，且需人工验证解释结果的实用性。

---

## 512. Mean-based algorithms: A lower bound and regret

**arXiv ID:** 2606.04931 | [PDF](https://arxiv.org/pdf/2606.04931v1)

**作者:** Julius Durmann `[一作]` (Technical University of Munich), Amelie Kleber `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在未知时限且仅有bandit反馈的多臂赌博机（MAB）环境下，研究了均值基算法（mean-based algorithms）的理论下限与算法实现，并提出了两种可实现的变体；

**💡 创新点**

创新点在于首次给出了均值基算法在bandit设置下的γ_t下限，证明任何速率o(1/√t)都不可行，并设计了符合该下限的ε‑greedy和Exp3变体；

**🔧 技术方法**

采用了γ_t-mean-based定义、概率界估计、探索–利用权衡分析、积分上界与下界推导以及对Exp3与ε‑greedy的改进参数化；

**📊 数据集**

使用了两种数据集：一个是10臂伯努利分布的随机MAB实验环境，另一个是带价格竞争的Bertrand寡头模型作为重复博弈场景；

**📈 对比分析**

通过与标准UCB、Exp3以及ε‑greedy对比实验，发现均值基Exp3在收敛速度和平均奖励上与传统算法相当，ε‑greedy在收敛上略慢；

**⚠️ 局限性**

局限性包括：提出的算法速率尚未达到理论下限；在极端或非平稳环境下可能表现不佳；并且对不同γ_t设定的通用性仍待进一步验证。

---

## 513. CIPER: A Unified Framework for Cross-view Image-retrieval and Pose-estimation

**arXiv ID:** 2606.05011 | [PDF](https://arxiv.org/pdf/2606.05011v1)

**作者:** Yurim Jeon `[一作]` (Seoul National University), Seung-Woo Seo `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的跨视角地理定位框架CIPER，能够同时完成大规模图像检索和精确的3-DoF姿态估计。

**💡 创新点**

通过双任务标记的Transformer编码器和双向Transformer解码器实现检索与姿态估计的协同学习，并引入set-prediction策略提升姿态回归稳定性。

**🔧 技术方法**

使用ViT骨干网络、全局自注意力、双任务标记、双向交叉注意力、set-prediction与MSE/BCE损失的组合。

**📊 数据集**

在VIGOR、KITTI和Ford multi-AV三个大规模跨视角数据集上进行实验。

**📈 对比分析**

与现有检索和姿态估计方法比较，CIPER在检索Recall@5/10和姿态误差（位置和方向）上均表现出与或优于state‑of‑the‑art，尤其在大范围朝向不确定性条件下性能显著提升。

**⚠️ 局限性**

主要局限在于对极端遮挡或极低分辨率的遥感图像适应性不足，以及对计算资源的高要求。

---

## 514. SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models

**arXiv ID:** 2606.05004 | [PDF](https://arxiv.org/pdf/2606.05004v1)

**作者:** Peihua Mai `[一作]` (National University of Singapore), Yan Pang `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SharedRequest框架，基于批处理的隐私保护LLM推理，无需模型改动。

**💡 创新点**

创新点是将隐私保护迁移到批量层面，混合原始与噪声提示并共享查询成本，同时设计轻量级多方协议保证匿名性。

**🔧 技术方法**

使用加密通信、伪随机掩码、属性组合采样、聚类批量分组，定义(A_n,ε)-可区分性隐私模型。

**📊 数据集**

使用法律问答(Legal‑QA)、医疗问答(Medical‑QA)和业务类MMLU‑Biz数据集。

**📈 对比分析**

与非私有设置及RanText、CusText、DP‑Prompt、CusText+、InferDPT等DP基线对比，保持相同或更高隐私预算下，Utility提升20%以上，查询成本可减少至原来的1/5左右。

**⚠️ 局限性**

局限：假设噪声采样器与服务商不合作；需要用户手动定义私有属性及其备选；在高度统一的请求分布下共享优势减弱。

---

## 515. TeleSWEBench: A Commit-Driven Benchmark for Evaluating LLM-Powered Software Engineering in Telecommunications

**arXiv ID:** 2606.05001 | [PDF](https://arxiv.org/pdf/2606.05001v1)

**作者:** Pranshav Gajjar `[一作]` (North Carolina State University), Vijay K Shah `[通讯]` (North Carolina State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本文中，作者提出并实现了一个针对电信领域的基准测试框架TeleSWEBench，并利用该框架评估了多种自动化软件工程（ASE）工具和大型语言模型（LLM）的性能。

**💡 创新点**

主要创新点包括：①基于真实开发者提交的commit构建的commit‑driven基准；②将任务划分为Easy、Medium、Difficult三大难度等级；③引入层次化LLM评判器TeleJudge，能够在文件层面以及整体patch层面进行语义评估；④通过两阶段评估（定位与功能正确性）分别衡量代理的代码定位与生成质量。

**🔧 技术方法**

所使用的技术与工具包括：自动化软件工程框架（AIDER、OpenHands、ClaudeCode）；多种主流LLM（Qwen3.5、GPT‑OSS、Gemma4、Kimi、QwenCoder 2.5）；基于单元测试的执行评估机制；以及自定义的TeleJudge评判器。

**📊 数据集**

数据集来源为srsRAN 5G开源项目的提交历史（2013‑2025年共约15k次提交），从中挖掘出734个结构化测试用例并配备了可执行的单元测试。

**📈 对比分析**

比较方法采用两阶段评估：第一阶段定位性能用Exact Match（EM）等指标衡量；第二阶段功能正确性用单元测试通过率（UAR）、TeleJudge接受率（TAR）以及同时满足两者的Ship‑Ready Percentage（SRP）。实验结果显示，最好的模型在SRP上仅能达到约25%，说明现有模型在电信域的实际代码生成仍然远未成熟。

**⚠️ 局限性**

局限性包括：仅针对单一开源仓库（srsRAN），可能无法代表更广泛的电信软件生态；评估仅覆盖三种开源ASE框架，未涉及商业闭源方案；使用已有的单元测试，可能未能覆盖所有边缘情况；实验受限于计算资源和API稳定性，导致部分任务出现超时或中断。

---

## 516. Code Lifespan Survival Analysis (CLSA): Predicting the Survival of Source Code Lines Using AST-Aware Mining

**arXiv ID:** 2606.04993 | [PDF](https://arxiv.org/pdf/2606.04993v1)

**作者:** Pavel Gurov `[一作]` `[通讯]` (Independent Researcher), Pavel Gurov (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究代码行的存活时间，将每行视为生存分析对象，估计其被删除的风险。

**💡 创新点**

提出 CLSA 框架，首次在行级别应用生存分析，并设计 5 阶段语义匹配管线过滤迁移和重构噪声。

**🔧 技术方法**

使用 Cox 比例风险模型、AFT 模型、共享 Gamma frailty、时间分层分析、Tree‑Sitter AST 解析与相似度匹配。

**📊 数据集**

构建 120 个活跃 TypeScript 开源仓库的 350 万行样本，包含 3250 万行生成事件和 1100 万行真正删除事件。

**📈 对比分析**

通过 Kaplan–Meier、Log‑Rank、Cox 与 AFT 结果比较，发现行熵与语法类别对存活影响显著；在共享 frailty 模型下 C‑index 由 0.586 提升至 0.666。

**⚠️ 局限性**

受限于缺乏跨语言验证、仅用静态特征、对迁移/修改的匹配误差、以及仓库级别异质性导致的混杂。

---

## 517. DeliChess: A Multi-party Dialogue Dataset for Deliberation in Chess Puzzle Solving

**arXiv ID:** 2606.04987 | [PDF](https://arxiv.org/pdf/2606.04987v1)

**作者:** Xiaochen Zhu `[一作]` (University of Cambridge), Andreas Vlachos `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了多人对话中的集体推理，构建了 DeliChess 数据集，记录团队在解决多项选择棋局谜题前后对话与决策。

**💡 创新点**

首次在棋局决策领域提供结构化、可评估的多人对话数据，并探讨探测性发问对团队表现的影响。

**🔧 技术方法**

使用了棋引擎 Stockfish 进行评估，采用简单分数、ARR 分数、Eval 分数三种评分方式，并训练了探测性发言分类器。

**📊 数据集**

使用 DeliChess 数据集，包含 107 组对话、3 种棋局类型（定位、战术、残局）以及相应的预后和后续决策记录。

**📈 对比分析**

通过比较讨论前后得分，发现集体推理平均提升约 0.07（简单分数）、0.085（ARR 分数）、1.15（Eval 分数）；讨论时长和多样性与提升呈正相关。

**⚠️ 局限性**

局限在于数据收集环境受限、探测性发言的自动标签可能存在域偏差、缺乏对探测质量的细粒度评估。

---

## 518. Be Fair! Can Machine Learning Engineering Agents Adhere to Fairness Constraints?

**arXiv ID:** 2606.04971 | [PDF](https://arxiv.org/pdf/2606.04971v1)

**作者:** Anna Richter `[一作]` (BIFOLD & TU Berlin), Sebastian Schelter `[通讯]` (BIFOLD & TU Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出责任中心评估框架，评估机器学习工程代理（MLE Agent）在敏感医疗领域的公平性与性能，并在黑白肤色差异的黑色素瘤分类任务上进行实验。

**💡 创新点**

创新点在于将责任约束（如公平性）纳入评估标准，设计可调技术难度的任务指令，并揭示现有MLE Agent无法在非技术专家使用场景下满足这些责任约束。

**🔧 技术方法**

使用的技术包括AIDE与MLZero两种MLE Agent（基于大语言模型的自动管道搜索）、Fitzpatrick17k与DDI数据集的训练与评估、AUC与AUC Gap指标进行公平性与性能测评，以及手工编写的专家管道与医生决策作为对照。

**📊 数据集**

训练数据集：Fitzpatrick17k（含皮肤色调标签）；测试数据集：Diverse Dermatology Images（DDI，含轻肤与深肤均衡样本及医生标注）。

**📈 对比分析**

比较方法：将代理生成的管道与三条手工专家管道和医生决策进行AUC与AUC Gap对比；实验结果显示，代理管道在AUC和公平性上均被专家方案压倒，并且代理结果方差显著较大。

**⚠️ 局限性**

局限性：代理生成的管道表现不稳定，未能实现公平性目标，存在执行错误与幻觉报告，缺乏对技术熟练度影响的评估，且缺少对更广泛高风险场景的验证。

---

## 519. Can Crowdsourcing Survive the LLM Era? A Community Survey on Human Data Collection

**arXiv ID:** 2606.04924 | [PDF](https://arxiv.org/pdf/2606.04924v1)

**作者:** Aswathy Velutharambath `[一作]` (University of Stuttgart), Amelie Wuehrl `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过 Qualtrics 设计并发放问卷，对 155 位 NLP 与相关领域研究者的经验和看法进行系统调查，探讨 LLM 在自由文本 crowdsourcing 中的使用、检测与对策。

**💡 创新点**

首次从研究者实践出发，构建了关于 LLM 时代 crowdsourcing 的实证综述与可操作性建议，弥补了先前零散经验的缺口。

**🔧 技术方法**

采用在线问卷收集数据，使用描述性统计和人类/GPT-4 主题提取对回答进行分析，并未训练新模型。

**📊 数据集**

调查样本为 155 份问卷，涵盖研究者的背景、对 LLM 的观察、检测信号与处理方法等信息。

**📈 对比分析**

主要通过频数统计和主题编码呈现发现：44% 观察到 LLM 使用，67.7% 通过警示防范，61.3% 使用注意力检查；未做传统机器学习性能对比，仅提供定量分布。

**⚠️ 局限性**

局限包括样本自选性偏好、以 NLP 领域为主、跨学科代表性不足，LLM 使用率随技术快速演进可能已变化，且缺乏统一且可靠的检测与评价指标。

---

## 520. Who Needs Labels? Adapting Vision Foundation Models With the Metadata You Already Have

**arXiv ID:** 2606.05107 | [PDF](https://arxiv.org/pdf/2606.05107v1)

**作者:** Elouan Gardès `[一作]` (Meta FAIR), Camille Couprie `[通讯]` (Meta FAIR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于元数据的无标签表示适配框架 FINO，专门在科学视觉任务中对预训练视觉基础模型进行领域特化。

**💡 创新点**

将离散与连续元数据作为弱监督引入自监督学习，利用原型对比和梯度反转同时鼓励重要信息并抑制噪声，且在推理时无需使用元数据。

**🔧 技术方法**

基于 DINO 与 iBOT 的自监督框架，加入 SIGReg 正则、原型池、预测器、元数据指导损失及梯度反转，采用 EMA 更新。

**📊 数据集**

四大科学域数据集：Human Protein Atlas、Functional Map of the World、iWildCam、MIMIC‑CXR；以及跨域验证的 OpenCell、CheXpert、FLAIR‑Hub。

**📈 对比分析**

与任务中心化的全微调、无监督域适配以及现有领域专用方法比较，FINO 在所有数据集上均超越这些方法，尤其在低标记、OOD 与跨域迁移上表现更佳。

**⚠️ 局限性**

依赖元数据的可用性与质量；需要先判定哪些元数据为信息性、哪些为噪声，且可能放大敏感属性的偏差。

---

## 521. Knowledge Index of Noah's Ark

**arXiv ID:** 2606.05104 | [PDF](https://arxiv.org/pdf/2606.05104v1)

**作者:** Sheng Jin `[一作]` (2077AI), Ge Zhang `[通讯]` (M-A-P)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了名为KINA的899项跨261个细学科的LLM知识基准，并给出了两项正式理论保证：子集代表性近似及奖金‑门槛评审机制提升审稿质量。

**💡 创新点**

将学科代表性建模为子集支持中心性并证明贪婪选择达到(1-1/e)近似；提出奖金‑门槛锦标赛机制并证明其在FOSD意义下优于平摊支付；提供Bootstrap排名稳定性分析，展示compact benchmark可保持排名一致。

**🔧 技术方法**

子集支持中心性的子模最大化与贪婪算法；奖金‑门槛评审机制的机制设计与FOSD分析；Bootstrap自助抽样评估排名稳定性；多阶段流水线结合规则筛选、双盲专家评审、LLM‑judge一致性与自动化修正；工具使用评估与参数规模分析。

**📊 数据集**

KINA自研的899个多选题，涵盖261个细学科；利用七个旗舰LLM（Gemini、Claude、GPT、Qwen、Claude等）进行评测；使用公开的标注与审稿手册、LLM‑judge评分标准；在内部日志中对比平摊支付与锦标赛机制的评审数据。

**📈 对比分析**

采用平均@4精度与不同温度下的推理预算，对42个模型进行整体与按学科分层排行榜；工具使用评估获得+1.5至+5.17分；排名稳定性通过Bootstrap（ρ=0.5、0.7、0.9）计算Kendall τ与rank‑1保留率；表现显示前沿模型仍未饱和，整体准确率约53%最高，工具使用与学科差异揭示诊断价值。

**⚠️ 局限性**

benchmark规模小导致排名在2个百分点内不显著；代表性原型依赖专家主观，未对齐度；奖金‑门槛机制理论仅在FOSD假设下，实际防止共谋需经验验证；工具使用评估受搜索引擎差异影响；仅覆盖英语主流学术资源，可能偏离地区文化；需周期性重新校准以应对模型提升导致难度漂移。

---

## 522. RAMC: Remote Access Memory Channels over HPE Slingshot

**arXiv ID:** 2606.05094 | [PDF](https://arxiv.org/pdf/2606.05094v1)

**作者:** Whit Schonbein `[一作]` (Sandia National Laboratories), Scott Levy `[通讯]` (Sandia National Laboratories)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Remote Access Memory Channels (RAMC)，一种针对 HPE Cray Slingshot 网络的显式一侧通信库；

**💡 创新点**

创新点包括：①利用 Slingshot 的内存区计数器实现轻量级完成通知；②设计持久单向通信通道，摆脱传统单侧窗口的全局同步与对称内存需求；③采用松耦合同步（pair‑wise 状态机），支持早鸟式通信和动态模式；

**🔧 技术方法**

主要技术包括：libfabric 的 CXI 提供者、Slingshot 内存区计数器（MR counters）、端点计数器、Bulletin Board（BB）机制以及基于状态值的用户定义同步；

**📊 数据集**

使用了两类数据集：1）自研的热扩散 5‑点斜边计算代码（19,600 进程、250 节点）；2）多种大小（1B–1MiB）的单向/往返微基准；

**📈 对比分析**

与 Cray MPICH 的 OMB 两侧基准进行对比：在 libfabric 1.15.2 下，1–4KiB 消息的单向带宽提升约 100–130%，在 2.3.1 下提升约 30–45%；延迟方面，RAMC 对 1–64B 消息略高（约 430–540 ns），但在 16–512KiB 区间可比 MPI 低 10–32%；

**⚠️ 局限性**

局限性包括：①仅适用于 Slingshot 硬件；②实现为被动目标，可能产生额外查询流量；③目标与发起方计数器粒度不匹配，导致同步粗粒度；④缺乏主动目标、触发式操作等高级特性；

---

## 523. Automatic Generation of Titles for Research Papers Using Language Models

**arXiv ID:** 2606.05085 | [PDF](https://arxiv.org/pdf/2606.05085v1)

**作者:** Tohida Rehman `[一作]` (Jadavpur University), Samiran Chattopadhyay `[通讯]` (Jadavpur University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究如何利用预训练语言模型和大语言模型从论文摘要自动生成论文标题，并提出了相应的评估框架。

**💡 创新点**

创新点在于：①新建 SpringerSSAT 社会科学领域数据集；②系统比较多种 PLM（T5、BART、PEGASUS）与 LLM（LLaMA‑3‑8B、GPT‑3.5‑turbo）的标题生成效果；③引入实体一致性度量和 ChatGPT 生成创意标题的实验。

**🔧 技术方法**

技术方法包括：对 T5‑base、BART‑base、PEGASUS‑large 进行 fine‑tune；对 LLaMA‑3‑8B 采用 LoRA 参数高效 fine‑tune；GPT‑3.5‑turbo 在 zero‑shot 情况下直接生成；评估使用 ROUGE、METEOR、MoverScore、BERTScore、SciBERTScore 以及实体级精确率/召回率指标。

**📊 数据集**

使用的数据集有：CSPubSum（计算机科学），LREC‑COLING‑2024（NLP），SpringerSSAT（社会科学新构建），以及人工标注的多样化标题数据。

**📈 对比分析**

比较方法：自动指标 + 人工评估。结果显示 PEGASUS‑large fine‑tune 在大多数指标上均优于其他模型；与 zero‑shot GPT‑3.5‑turbo 或未 fine‑tune 的 LLaMA‑3‑8B 相比，fine‑tune 后性能显著提升；在跨域测试中，CSPubSum fine‑tune 的模型迁移效果好于 SpringerSSAT fine‑tune；ChatGPT 生成的创意标题在语义相似度上与作者标题相当。

**⚠️ 局限性**

局限性包括：①仅覆盖计算机科学、NLP 与社会科学三大领域，缺乏对医学、物理等其他学科的验证；②仅使用英文数据，无法评估多语言场景；③实验使用的模型版本和评估指标随时间快速迭代，结果可能随新模型或更细粒度指标而变化；④评估主要基于定量指标，缺乏更丰富的主观风格与创意度量。

---

## 524. AutoLab: Can Frontier Models Solve Long-Horizon Auto Research and Engineering Tasks?

**arXiv ID:** 2606.05080 | [PDF](https://arxiv.org/pdf/2606.05080v1)

**作者:** Zhangchen Xu `[一作]` (University of Washington), Zichen Chen `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AutoLab，一个针对科学与工程任务的超长时域闭环优化基准，涵盖 36 个任务并提供连续量化评分。

**💡 创新点**

创新点在于：①将评估焦点从单步回答转向持续迭代；②构建跨四个领域的多样化任务集合；③采用防作弊的封闭评估与连续评分方案。

**🔧 技术方法**

技术方法包括：使用容器化沙盒、封闭验证器、日志压缩评分、以及统一的 Harbor harness 与 LLM agent 交互。

**📊 数据集**

数据集由 36 个专家策划的真实工程/研究任务组成，涵盖系统优化、谜题、模型开发与 CUDA 核心优化。

**📈 对比分析**

对 17 款前沿模型进行系统评测，结果显示 AutoLab-Alpha 以 Avg@3 0.68、Dominance 0.93 占据榜首，表明模型的持续迭代和时间感知是获胜关键。

**⚠️ 局限性**

局限性：仅覆盖可执行的系统/ML 工程工作流程；需要多小时的运行时间和 GPU 资源，评估成本高，且对非工程型科研任务适用性有限。

---

## 525. Fast & Faithful Function Vectors

**arXiv ID:** 2606.05079 | [PDF](https://arxiv.org/pdf/2606.05079v1)

**作者:** Minh An Pham `[一作]` (Fraunhofer Heinrich-Hertz-Institute), Reduan Achtibat `[通讯]` (Fraunhofer Heinrich-Hertz-Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在指令式学习中使用函数向量（FV）时，不同头选择和注入方式对模型性能的影响，并提出了改进方案。

**💡 创新点**

创新点在于：①使用梯度基的 AttnLRP 代替传统的平均间接效应（AIE）来选择注意力头；②提出分布式 FV 注入（DFV）方案，逐头注入而非统一平均注入，从而更贴合模型内部计算；③在多模型、多任务上系统评估并证明这些改进带来的准确率和效率提升。

**🔧 技术方法**

采用的技术包括：Layer‑wise Relevance Propagation (LRP)、AttnLRP、平均间接效应（AIE）、分布式 FV 注入（DFV）以及注意力头的重要性评估。

**📊 数据集**

使用的主要数据集为 Davidson 等的指令式多任务数据集（去除分类任务后），在 Llama‑3.2‑3B、Llama‑3.1‑8B 以及 Qwen‑3‑4B 三个模型上进行实验。

**📈 对比分析**

通过在零 shot 环境下对比 AIE 与 LRP、平均注入与分布式注入，结果显示 LRP+DFV 能在所有模型中提升最高 0.156 的准确率，同时在效率上相较 AIE 提升约 500 倍。

**⚠️ 局限性**

限制：对某些任务（如需要精确的语言转换或特殊指令）仍存在失败；intensional frame 的确定需要人工标注；不同任务的头集合差异大，单一全局头集合并不总是最优。

---

## 526. Bridging High-Level Intent and Network Execution: Detecting Violations and Intent Drift Through Low-Level Traffic Analysis

**arXiv ID:** 2606.05076 | [PDF](https://arxiv.org/pdf/2606.05076v1)

**作者:** Tonia Haikal `[一作]` (Texas A&M University), Eman Hammad `[通讯]` (Texas A&M University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种通过将网络流头部标准化为7元组，建立内部低层意图（ILI）接口，实现对意图驱动网络（IBN）数据平面执行的实时验证。

**💡 创新点**

创新点在于提出基于实际流记录的意图漂移（Intent Drift）度量，并揭示宽松策略下违例数下降但意图漂移保持不变的“合规悖论”，为闭环调度提供新视角。

**🔧 技术方法**

使用了低层流向量化、数据平面监控、政策评估、漂移度量、闭环架构等技术，并结合大规模流量构造与P4/SmartNIC等实现。

**📊 数据集**

使用Merit Network分布式 Honeynet 收集的1.0091亿完整流记录数据集。

**📈 对比分析**

通过比较三种行政策略（Strict、Balanced、Permissive）下的违例计数和意图漂移，实验表明违例随策略宽松而下降，但意图漂移保持不变，显示传统违规指标易被掩盖。

**⚠️ 局限性**

局限在于仅基于端口级别的ILIs，未考虑更丰富的上下文信息，且实验仅在离线数据集上验证，未完成实时线速验证。

---

## 527. Learning What Not to Impute: An Uncertainty-Aware Diffusion Framework for Meaningful Missingness

**arXiv ID:** 2606.05073 | [PDF](https://arxiv.org/pdf/2606.05073v1)

**作者:** Lixing Zhang `[一作]` (University of Minnesota), Liyan Xie `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于扩散模型的选择性填补框架 Diff‑Joint，能够在同一模型中同时估计缺失值和判断缺失是否具有意义，区别对待有意义缺失与观测导致的缺失。

**💡 创新点**

创新点在于：①将缺失标签与填补值联合建模，②利用条件采样生成多条样本并以不确定性为依据聚合更新，③通过迭代训练实现对缺失模式的自我校正。

**🔧 技术方法**

主要技术包括扩散模型（Diffusion Model）、条件采样、基于不确定性的聚合（不确定性分数/熵/方差）、以及迭代的模型训练与状态更新。

**📊 数据集**

实验使用了合成的 Bayesian‑Network 数据集和真实临床数据集 MIMIC‑IV‑ED（急诊门诊数据）。

**📈 对比分析**

与 Mean/Mode、missForest、CACTI、DiffPuter 等基线比较，Diff‑Joint 在识别有意义缺失、填补精度以及下游预测（Macro‑F1、ROC‑AUC 等）方面均表现优越，尤其在下游任务上取得显著提升。

**⚠️ 局限性**

局限性包括：需预先给定候选有意义缺失列；对不确定性分布差距的假设可能不适用于所有场景；迭代扩散过程计算成本较高，且可扩展性仍待提升。

---

## 528. RIDE: An Open Dataset and Benchmark for Train Delay Prediction

**arXiv ID:** 2606.05070 | [PDF](https://arxiv.org/pdf/2606.05070v1)

**作者:** Clément Elliker `[一作]` (LIX Ecole Polytechnique IP Paris), Sonia Vanier `[通讯]` (LIX Ecole Polytechnique IP Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于比利时铁路网络的RIDE数据集及统一预测基准，提供多层数据管道和标准化评估框架。

**💡 创新点**

首次将可复用的中间银层数据与面向不同模型的黄金基准结合，统一预测目标、时间拆分和评估指标，并实现按预测时长与延误变化的细粒度性能拆解。

**🔧 技术方法**

采用规则基线、图事件模拟、XGBoost、MLP、LSTM、Transformer及图神经网络等多种技术进行对比实验。

**📊 数据集**

使用2023-2025年比利时铁路运营、天气与时刻表等公开数据，共计9.45亿事件、360万旅程和3570万天气记录。

**📈 对比分析**

通过固定训练/测试快照、MAE/RMSE及按预测时长和延误变化拆分的评估，学习模型明显优于规则基线；GNN平均MAE约73.6秒，表现最佳，其他深度模型相近。

**⚠️ 局限性**

局限在于数据仅覆盖比利时网络，模型在极端延误或网络拓扑变化下的泛化能力有限，且模型差异不大，提示特征工程与任务设计仍是提升的关键。

---

## 529. Graph Cascades: Contagion-Based Mesoscopic Rewiring for Structure-Aware Graph Machine Learning

**arXiv ID:** 2606.05046 | [PDF](https://arxiv.org/pdf/2606.05046v1)

**作者:** Meher Chaitanya `[一作]` (KTH Royal Institute of Technology), Luana Ruiz `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于传播动态的中尺度图重连策略 Graph Cascades，利用多跳共振关系生成稀疏辅助图，可直接作为 GNN 或 Graph Transformer 的输入。

**💡 创新点**

创新点在于：①通过 MAS/TAS 传播过程捕获节点间多跳强化支持，将高频共振节点提升为直接邻居；②用有效电阻度等理论证明重连能提升标签同类性；③构造 CR‑Graphormer，展示辅助图即可驱动稀疏注意力模型。

**🔧 技术方法**

技术方法包括：MAS 与 TAS 传播算法、短步强化计数、稀疏加权图构造、CR‑Graphormer 变换器、SBM 证明与有效电阻度分析。

**📊 数据集**

实验使用 16 个 DGL benchmark（涵盖同类/异类图）、125 个合成 SBM 及长程任务（PascalVOC‑SP、COCO‑SP、Questions、Roman‑empire 等）进行节点分类。

**📈 对比分析**

与 16 种 GNN/GT 基线（GCN、GraphGPS、NAG、VCR、CR‑Graphormer 等）在节点分类上对比，最优提升可达 +34%（在异类和中高同类图），多数模型在多种图上均有显著提升，低度同类或瓶颈图则表现不佳。

**⚠️ 局限性**

局限性：在低度规整或存在瓶颈的图（如 cycle、grid、Roman‑empire）效果有限；仅利用拓扑信息，缺乏特征驱动的边选择；未与特征相似度或联合结构-特征方法结合。

---

## 530. In-Context Graphical Inference

**arXiv ID:** 2606.05042 | [PDF](https://arxiv.org/pdf/2606.05042v1)

**作者:** Zehua Cheng `[一作]` (University of Oxford), Jiahao Sun `[通讯]` (FLock.io)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将离散图模型的边缘推理重新建模为自回归序列预测，提出 In-Context Graphical Inference (ICG‑I)框架。

**💡 创新点**

核心创新：① 恢复变量消除的顺序结构；② 用 Graph Transformer 预测 Tensor‑Train 压缩的中间因子；③ 引入动态最短路径距离编码与 softplus‑约束的 TT 核；④ 使用 Dirichlet 输出层配合 Weighted Conformal Prediction 实现分布式、校准的置信度估计。

**🔧 技术方法**

采用技术包括 Graph Transformer、Tensor‑Train（TT）压缩、软正则化软正切核、Gumbel‑Softmax 变量排序、Dirichlet‑Multinomial 损失、Weighted Conformal Prediction、MCMC 参考标签和自回归推理链。

**📊 数据集**

数据集：UAI 2022 benchmark（网格、贝叶斯网络、Promedas、随机因子图），SK 与 EA 玻色子模型（温度 β 可调），OpenGM 蛋白质结构，平面网格、随机树、Erdős‑Rényi，Barabási‑Albert 与 Watts‑Strogatz 等拓扑差异的图。

**📈 对比分析**

与 BP、TRBP、GBP、GNN‑BP、BPNN、Direct GNN、LBP 等传统与神经方法对比；ICG‑I 在 MAE、KL、Max Error、Hellinger 等指标上均优于所有基线（例如 MAE 0.020 对比最佳基线 0.041；在 500 结点受挫折的自旋玻璃上 MAE 0.048 远低于 0.105）。

**⚠️ 局限性**

限制：① TT 近似的误差随消除顺序线性累积，需要严格归一化；② 受 bond‑dimension r 限制，表达能力受限；③ 需要按节点数线性迭代，缺乏并行性；④ 需要准确的密度比估计才能保证 WCP 覆盖；⑤ 训练依赖 MCMC 参考标签的质量；⑥ 对大规模图若不使用 TT，易出现 OOM。

---

## 531. Scaling Expert Feedback with Reflective Edit Propagation in Compositional Knowledge Bases

**arXiv ID:** 2606.05023 | [PDF](https://arxiv.org/pdf/2606.05023v1)

**作者:** Jiajing Guo `[一作]` (Bosch Research North America), Liu Ren `[通讯]` (Bosch Research North America)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了RAID系统，利用单个专家编辑的反射式代理在组合式知识库中推断意图并批量传播更新。

**💡 创新点**

创新点在于通过反射式意图推断与链式工具规划，将专家单条编辑转化为结构化知识库的大规模更新，并提供专家控制的验证环节。

**🔧 技术方法**

采用反射代理、意图推断、链式工具规划（Update Description、NL Search、Generate New Description），并使用GPT‑5‑mini与LangChain进行LLM交互。

**📊 数据集**

在RxTerms（RxNorm 衍生的药物术语）数据集上构造扰动样本进行量化评估，并进行专家用户研究。

**📈 对比分析**

与人工单条校正对比，意图分类准确率≈99.6%，检索召回率1.0，精确率≈98.6%，修订准确率整体约76%，语义相似度约89%，表明系统在推断与传播方面具有高效可扩展的性能。

**⚠️ 局限性**

存在检索过程中子串冲突导致假阳性、表层重写传播精度低，以及对用户可选传播控制的细化需求等局限。

---

## 532. FoeGlass: Simple In-Context Learning Is Enough for Red Teaming Audio Deepfake Detectors

**arXiv ID:** 2606.05101 | [PDF](https://arxiv.org/pdf/2606.05101v1)

**作者:** Sepehr Dehdashtian `[一作]` (Michigan State University), Gaurav Bharaj `[通讯]` (Reality Defender)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于LLM的自动红队方法FoeGlass，用黑盒访问的方式在文本到语音（TTS）生成器上搜索自然对抗样本，从而识别并增强音频深度伪造检测器（ADD）的漏洞。

**💡 创新点**

创新点在于：①首次利用LLM的推理与链式思考功能，结合反馈的真实性和多样性分数，自动探索TTS输出空间；②设计了基于最小余弦距离的多样性度量，避免模式崩塌；③攻击数据可跨不同ADD、TTS模型迁移，并可用于提升检测器鲁棒性。

**🔧 技术方法**

核心技术包括：大语言模型（如DeepSeek-R1）进行输入生成；TTS模型（VITS、TTS-...）生成音频；ADD模型（ViT、AST、RawNet等）进行真实性评估；WavLM特征提取用于多样性评分；上下文构造与链式思考反馈机制。

**📊 数据集**

使用的数据集主要是公开的ASVspoof5、VoxCelebSpoof以及多种开源TTS模型生成的音频；同时生成的自然对抗样本与ASVspoof5基准进行对比。

**📈 对比分析**

在八个ADD模型与三种TTS模型上，FoeGlass相较于无条件采样基准，误判率（FNR）提升可达94%，并且攻击样本具有高度可迁移性；对RawNetLite和AASIST进行Fine‑tune时，使用FoeGlass生成的样本能将FNR从约40%提升至>80%。

**⚠️ 局限性**

局限性包括：需要手动设定多样性阈值与LLM上下文长度；不同LLM和参数设置会显著影响成功率；目前仅在开源ADD上验证，缺少对商业检测器的评估；生成的对抗样本与现实语音分布的差异仍待进一步研究。

---

## 533. RePercENT: Scaling Disentangled Representation Learning Beyond Two Modalities

**arXiv ID:** 2606.05109 | [PDF](https://arxiv.org/pdf/2606.05109v1)

**作者:** Vasiliki Rizou `[一作]` (EPFL), Dorina Thanou `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 RePercENT，一个自监督的可插拔框架，用于在多于两种模态的情况下实现可扩展的对称拆解，得到每对模态的共享与独特表示。

**💡 创新点**

创新点包括：①仅使用单一模态编码器和预提取的基础模型嵌入即可实现多模态拆解；②通过“语义编码”与“组槽注意力”在每个模态的 Latent Slots 中引入明确的路由机制；③联合优化目标在信息理论框架下给出，且在可达的最小必要信息（MNI）与不可达情况下都有最优或近似最优理论保证；④复杂度从 O(M²) 降到 O(M)，实现真正可扩展。

**🔧 技术方法**

技术核心包括：Perceiver 风格的 Latent Attention、Semantic Encoding（pair 和 type embeddings）、Group Slot Attention、InfoNCE、KL 上界与交叉协方差惩罚的联合训练目标。

**📊 数据集**

数据集涵盖：①合成数据（可获取真实共享/独特表示）；②IRFL 文字-图像-字幕的比喻/隐喻/典故；③TCGA 多模态肿瘤数据（四种模态：病理影像、分子、报告、临床）。

**📈 对比分析**

与多种基线对比：多编码器拆解模型（MLP、gMLP、GRU）、CLIP 对齐模型（zero‑shot、投影、全端微调）以及多模态融合基线。结果显示：在合成实验中参数与 FLOPS 低于基线却实现接近最优拆解；在 IRFL 任务中整体准确率和 OoD 性能最高；在 TCGA 肿瘤分类中，RePercENT 的拆解（尤其是共享+独特组合）显著提升了多模态表征的预测效果，且在模态缺失场景下保持高宏 F1。

**⚠️ 局限性**

局限性包括：①对 MNI 可达性的假设仍需经验验证；②虽然只需单模态编码器，但仍需预先训练好的基础模型；③在极大模态数时仍需关注内存占用和训练时间；④当前仅验证了二维模态对的拆解，三维以上的多模态全局关系尚未深入探索。

---

## 534. Bernoulli CUSUM and Bayes-Optimal Detection Ceilings for Trust Fraud in Sparse Rating Networks

**arXiv ID:** 2606.05090 | [PDF](https://arxiv.org/pdf/2606.05090v1)

**作者:** Talal Ashraf Butt `[一作]` `[通讯]` (Higher Colleges of Technology), Talal Ashraf Butt (Higher Colleges of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种针对稀疏评分网络中信任欺诈的二值化贝叶斯CUSUM检测框架，并给出了基于经验观测参数的 Bayes‑optimal F1 上限；

**💡 创新点**

创新点在于（1）推导了贝叶斯最优 F1 上限并解释了现有方法停滞的根本原因；（2）揭示连续观测模型与实际分布不匹配导致的性能损失；（3）设计了双模式二值化 CUSUM + 双规 EMA 的协同架构；

**🔧 技术方法**

使用了 Bernoulli CUSUM、指数移动平均（EMA）、信息量与 KL 散度分析以及基于经验观测参数的 Bayes‑optimal 计算；

**📊 数据集**

主要实验数据集为 Bitcoin‑OTC 与 Bitcoin‑Alpha 两个真实的信任评分网络；

**📈 对比分析**

与 GaaSTrust、BTGAggDA‑Cal、SimpleMean、REV2 等基线对比，双模式方案在所有 8 种攻击下均显著提升 AUC（BTC‑OTC 平均 0.749，BTC‑Alpha 0.796），在绝大多数攻击上达到 95% 以上的 Bayes‑optimal 效率；

**⚠️ 局限性**

局限性包括：在非二值化、均衡连续分布下效果不佳；对抗性自适应攻击未测试；双规机制与连续观测不兼容，需先进行二值化；

---

## 535. Graph Traversal on Tensor Cores: A BFS Framework for Modern GPUs

**arXiv ID:** 2606.05081 | [PDF](https://arxiv.org/pdf/2606.05081v1)

**作者:** Deniz Elbek `[一作]` (Sabanci University), Kamer Kaya `[通讯]` (Sabanci University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于 Tensor Core 的 BFS 框架（BFS‑TC），同时支持单源 BFS、多源 BFS 和闭合度中心性计算。

**💡 创新点**

创新点包括：Binarized Virtual Slice Sets（BVSS）数据结构实现 warp 级别的近乎完美负载均衡；针对 TC 的二进制 MMA 调度布局将 MMA 调用量降低 8 倍；引入 lazy  vertex 更新方案消除 ATOMG 造成的同步与原子冲突；动态在 TC 与 CUDA 核之间切换；对不同类型图分别使用 JaccardWithWindows（scale‑free）或 RCM（非 scale‑free）进行重排序；采用持久 kernel 进行级联；以及在多源 BFS 中使用重映射与活跃/脏 VSS 进一步压缩内存访问。

**🔧 技术方法**

使用的技术包括：NVIDIA Tensor Core 的二进制 MMA、bit‑级稀疏矩阵‑向量乘法、warp 级别的虚拟切片分配、REDG 异步原子、动态方向切换、图重排序（JaccardWithWindows、RCM）、持久 kernel 与 Cooperative Groups、以及多源 BFS 的结构化数组和虚拟索引。

**📊 数据集**

实验数据集：14 个真实图，涵盖 GAP Benchmark Suite（GAP‑road、GAP‑twitter、GAP‑web、GAP‑kron、GAP‑urand 等）与 SuiteSparse 大图（nlpkkt240、uk‑2005、it‑2004、europe_osm、com‑Friendster、Spielman_k600、webbase‑2001、kmer_V1r、mawi 等）。

**📈 对比分析**

与 GAP、Gunrock、GSWITCH、BerryBees 以及之前最速的 TC 实现相比，平均速度提升分别为 22.0×、7.7×、8.1×、5.9×；在特定图上最高可达 23.1×。在多源 BFS 上比单源快 2.7×；利用 100 台 H100 计算 com‑Friendster 的闭合度中心性仅需 3,665 秒，完成 65.6M 顶点的精确结果。

**⚠️ 局限性**

局限性包括：方向/模式切换阈值是针对特定 GPU（Hopper）调优，其他架构需重新校准；lazy 更新方案仅适用于更新分歧高的 scale‑free 图，对非 scale‑free 图可能产生负面影响；多源 BFS 的内存占用随并行源数线性增长，导致大图 OOM；JaccardWithWindows 的预处理成本在极大图上较高。

---

## 536. Attention-Augmented LSTMs for Automatic Homophonic Ciphertext Decipherment

**arXiv ID:** 2606.05078 | [PDF](https://arxiv.org/pdf/2606.05078v1)

**作者:** Micaella Bruton `[一作]` (Stockholm University), Beáta Megyesi `[通讯]` (Stockholm University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在共享密钥空间下，使用带多头注意力的双向LSTM模型，对历史单字母同义替代密码进行自动解密；

**💡 创新点**

在不依赖语言模型、频率统计或关键搜索的前提下，证明该模型能学习并准确映射同义替代关系，并在不同年代、语言、文本长度和噪声条件下保持高精度；

**🔧 技术方法**

使用双向LSTM网络结合多头自注意力机制，并通过交叉熵训练进行字符级预测；

**📊 数据集**

合成同义替代密码数据，来源于1500-1899年HistCorp英语和瑞典语语料，加入噪声、可变长度码等变体；

**📈 对比分析**

与传统搜索/频率方法对比，模型在字符级F1>0.99，甚至在最长1000字、含噪声的条件下仍≈1；5折交叉验证在50字短文本上几乎100%准确；

**⚠️ 局限性**

仅适用于共享密钥空间，无法直接破解独立密钥；对超出共享池的密钥性能接近0，且对极端噪声或极长文本的鲁棒性尚未充分验证。

---

## 537. MaCo-GAN: Manifold-Contrastive Adversarial Learning for Single Image Super-Resolution

**arXiv ID:** 2606.05068 | [PDF](https://arxiv.org/pdf/2606.05068v1)

**作者:** Daeyoung Han `[一作]` (Gwangju Institute of Science and Technology), Moongu Jeon `[通讯]` (Gwangju Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的单幅图像超分辨率框架 MaCo-GAN，核心思路是将传统 GAN 的对抗损失替换为监督式对比损失，并通过自学习的 fake 样本合成器产生“在流形”与“离流形”两类伪样本来构建对比极小极大游戏。

**💡 创新点**

创新点包括（1）引入“on-manifold / off-manifold”对比策略，迫使生成器既吸引可行解又排斥不可行解；（2）设计基于 U‑Net 的噪声级别可调 fake 样本合成器，能够生成既符合低分辨率对应关系又具有高频细节的多样化负样本；（3）将监督式对比损失直接嵌入生成器与判别器的训练，实现严格的条件真实感。

**🔧 技术方法**

技术上使用了 U‑Net 结构的噪声条件合成器、LPIPS 与 adversarial 组合的 AE 损失、PatchGAN 判别器、监督式对比损失（SupCon）与温度调节、以及 AESOP 基础框架的替代训练策略。

**📊 数据集**

训练数据集采用 DF2K（DIV2K + Flickr2K），评估使用 BSD100、General100、Urban100、Manga109、DIV2K‑val 与 LSDIR‑val 等标准超分辨率基准。

**📈 对比分析**

与 ESRGAN、SPSR、LDL 以及 AESOP 等主流 GAN‑SISR 方法对比，MaCo-GAN 在保持 PSNR/SSIM 接近 baseline 的同时，显著提升 LPIPS 与 DISTS 等感知指标，展示了更优的感知‑失真折衷。

**⚠️ 局限性**

主要局限在于：1）合成器和对比策略假设了固定的降采样退化，难以直接迁移到真实世界盲超分；2）对比极小极大训练需精细调节温度与样本数，易出现判别器主导不稳定；3）合成器需要额外训练与推理开销，且对不同图像内容的适应性尚待验证。

---

## 538. UniCAD: A Unified Benchmark and Universal Model for Multi-Modal Multi-Task CAD

**arXiv ID:** 2606.05058 | [PDF](https://arxiv.org/pdf/2606.05058v1)

**作者:** Jingyuan Chen `[一作]` (SenseTime Research and Tetras.AI), Chen Qian `[通讯]` (SenseTime Research and Tetras.AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了统一的多模态 CAD 基准 UniCAD，并在此基准上训练了一种统一的多模态大语言模型 UniCAD-MLLM，支持文本、图像/草图、点云三种输入及 CAD 生成与问答任务。

**💡 创新点**

创新点包括：①构建了覆盖文本、图像、草图、点云的全尺寸多模态 CAD 数据集；②提出将 CAD 输出转化为可执行的 CadQuery Python 脚本，兼具可编辑、可验证性；③在同一框架内实现多模态、跨任务的联合学习，显著提升数据效率。

**🔧 技术方法**

技术手段：基于 Qwen2-VL-2B 的多模态 LLM，加入轻量级点云投影模块；使用联合编码器将三种模态映射到共享几何语义潜在空间；自回归解码器生成 CadQuery 代码或文本答案；利用 Chamfer Distance 与 IoU 评估生成质量；对齐问答子任务通过将 CAD 程序作为额外上下文输入。

**📊 数据集**

使用的数据集为自研 UniCAD，包含 1,448,150 个 CAD 程序、对应的文本描述、多视角渲染图、草图、点云及 QA 对；此外在零样本泛化评估中使用 Fusion360 公开数据集。

**📈 对比分析**

与单模态或多任务基线比较：在所有四种输入（点云、文本、图像、草图）以及 CAD QA 任务上，UniCAD-MLLM 统一模型分别实现 CD 为 0.17-0.22、IoU 高达 89.8%，CAD QA 准确率 90.0%；在单视图重建、Fusion360 零样本任务中也优于现有最强方法，提升 CD 降低 3.9×、IoU 增加 21.4%。

**⚠️ 局限性**

局限性：数据集主要覆盖基于草图+拉伸的基本操作，缺乏倒角、倒圆、自由曲面等高级几何；模型基于自然图像预训练，缺少专门的 CAD 预训练，导致对复杂结构的空间推理有限；未来需扩展工业级数据、增强几何预训练以及实现交互式编辑与自纠正功能。

---

## 539. Self-Reflective APIs: Structure Beats Verbosity for AI Agent Recovery

**arXiv ID:** 2606.05037 | [PDF](https://arxiv.org/pdf/2606.05037v1)

**作者:** Arquimedes Canedo `[一作]` (Siemens Digital Industries Software), Grama Chethan `[通讯]` (Siemens Digital Industries Software)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种在 API 返回验证失败时给出机器可读、可直接修复建议的 Self‑Reflective API，并在食谱转换与账单退费两个自定义 API 上进行实验，验证该机制能显著提升 LLM 代理的任务完成率与 token 效率。

**💡 创新点**

创新点在于将错误修复信息从自然语言文本转为结构化的“可直接执行”提示，并通过严格的泄漏审计与三种错误细粒度（传统、详细、反射）分离，证明结构化提示比仅靠更长文本更能帮助 LLM 代理完成需要领域知识的验证任务。

**🔧 技术方法**

主要技术包括：自定义 API schema（Self‑Reflective API Schema v0.1），基于 FastAPI 的可插拔错误响应包装，LLM 代理的简单重试循环（最多 5 次），以及对 API 反馈的 JSON‑schema 解析与合并；实验用到 Claude（haiku‑4‑5、sonnet‑4‑6）和 GPT‑4o‑mini 三个 LLM。

**📊 数据集**

使用的任务集为 10 题“adversarial”食谱转换任务（含文化兼容性、品牌认证、精确比例等），以及在第二域（Acme 账单 API）对应的 10 题；每个 (模型, 模式) 细胞运行 30 次，形成 270 次实验。

**📈 对比分析**

比较方法为三种错误细粒度（Traditional、Verbose、Reflective）对同一验证逻辑下的成功率、重试次数、每成功请求 token 消耗进行 Fisher 检验；结果显示在 Anthropic 模型上 Self‑Reflective 模式成功率比 Verbose 提升 36.7–40.0pp（p ≤ 0.0022），token 效率提升 1.8–2.2×；在 GPT‑4o‑mini 上提升 13.3pp（不显著），但在第二域账单 API 上 100% 成功率。

**⚠️ 局限性**

主要局限包括：实验仅覆盖单一自定义域（食谱与账单）和有限的 LLM 组合，难以保证结果在更大规模或第三方真实 API 上普适；模型能力差异导致在低能力模型上结构化提示的收益不明显；实验使用的 10 题为人工挑选的对抗样本，可能不代表常见流量；并且仍存在“操作语义瓶颈”，即结构化建议缺乏对如何把提示映射到请求体的执行细节。

---

## 540. Enhancing the MADDPG Algorithm for Multi-Agent Learning via Action Inference and Importance Sampling

**arXiv ID:** 2606.05021 | [PDF](https://arxiv.org/pdf/2606.05021v1)

**作者:** Marc Walden `[一作]` (University of California Los Angeles), Hamza Khan `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了在多智能体深度确定性策略梯度（MADDPG）框架下的两项改进：预训练动作推理网络和几何分布重要性采样，以提升学习稳定性和协作效率。

**💡 创新点**

创新点在于：①使用预训练的动作推理模块，让每个智能体预测同伴的上一步动作并将预测作为训练时的额外输入；②改用几何分布对经验回放进行递归优先采样，优先使用最近且信息量高的经验，从而减轻多智能体环境中的非平稳性。

**🔧 技术方法**

技术实现主要包括：多智能体深度确定性策略梯度（MADDPG）算法、PyTorch神经网络、动作推理模块、几何分布重要性采样、经验回放缓冲区、软更新目标网络等。

**📊 数据集**

实验数据集采用PettingZoo库中的“simple_tag_v3”捕食者‑猎物任务，该任务提供离散动作空间、部分可观测环境以及多智能体协作与竞争的混合动态。

**📈 对比分析**

通过与标准MADDPG进行同等训练周期（3×10⁴ episode）比较，使用累计最大奖励和滑动平均奖励两项指标。实验结果显示：动作推理变体在早期就能获得更高奖励并更快收敛；几何采样次之；标准MADDPG最慢，且在学习稳定性上波动更大。

**⚠️ 局限性**

局限性包括：①动作推理网络需要预训练，受限于训练样本的多样性和可迁移性；②几何分布采样参数p未实现动态调节，可能需要进一步优化；③实验仅在二维离散动作环境中验证，尚未验证在更大规模或连续动作空间下的泛化性能。

---

## 541. Handwriting Extraction and Analysis of Signature Lists in Swiss Popular Initiatives

**arXiv ID:** 2606.05018 | [PDF](https://arxiv.org/pdf/2606.05018v1)

**作者:** Marco Peer `[一作]` (University of Fribourg), Andreas Fischer `[通讯]` (University of Fribourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套从扫描的瑞士公投签名表格中自动提取手写条目，并利用OCR与写字人检索两种技术检测重复或异常签名的工作流程。

**💡 创新点**

创新点在于将传统的模板匹配+线段分割与最新的Transformer手写文字识别、写字人检索模型相结合，系统性评估短、离谱词汇对OCR的挑战，并证明写字人检索能够有效聚类相似手写，从而辅助签名验证。

**🔧 技术方法**

使用的技术包括：模板创建与匹配算法、基于模板的线段分割、TrOCR、Qwen3‑VL‑8B等Transformer OCR模型，以及SIFT + mVLAD、ResNet + mVLAD等写字人检索方法。

**📊 数据集**

数据集：Test‑SL（443 行、418 写字人），Real‑SL（约10 万行真实签名表格），以及公开的 CVL 数据集用于对比与训练。

**📈 对比分析**

性能对比：在 OCR 任务中，Qwen3‑VL‑8B 在姓名、地址等短字段的 CER 最高仅 29.6%；写字人检索任务中，ResNet + mVLAD（实例监督）在 Real‑SL 上实现 mAP 50.6%，Top‑10 准确率 86.6%。

**⚠️ 局限性**

局限性：OCR 对短词汇（如姓名、地址）表现差，依赖大量训练数据；写字人检索受限于每行仅一行手写且相似文本内容会干扰相似度；缺乏大规模公开标注的签名数据，模型泛化能力有待提升。

---

## 542. How Software Engineering Students Use LLMs to Write Research Papers: An Experience Report

**arXiv ID:** 2606.05114 | [PDF](https://arxiv.org/pdf/2606.05114v1)

**作者:** Ronnie de Souza Santos `[一作]` (University of Calgary), Italo Santos `[通讯]` (University of Hawai‘i at Mānoa)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本科软件架构课程中，让学生在完成快速综述或灰色文献综述时使用大型语言模型（LLM）并提交反思性披露说明，记录其使用方式与感受；

**💡 创新点**

首次系统探讨LLM在经验性软件工程写作中的反思性使用，并将LLM辅助的自动分类与人工验证相结合的混合分析流程应用于学生披露文本，弥补传统技术使用研究的反思维度缺失；

**🔧 技术方法**

主要使用ChatGPT等LLM进行文本生成、写作润色、方法澄清及思路发散等功能；

**📊 数据集**

未使用公开实验数据集，而是收集了146份匿名学生自述披露文本；

**📈 对比分析**

并未进行传统算法性能对比，采用四阶段交叉分析（阅读、LLM辅助初分、人工验证校正、教育观察整合）对学生使用模式进行定性评估；

**⚠️ 局限性**

仅基于单门课程单次写作任务的自我披露，缺乏跨课程、跨机构的验证，且缺少直接观察或访谈等补充数据，限制了结论的普适性与深度。

---

## 543. Identifying Gems from Roman RAPIDly

**arXiv ID:** 2606.05103 | [PDF](https://arxiv.org/pdf/2606.05103v1)

**作者:** Karan Gandhi `[一作]` (Indian Institute of Technology), Mansi M. Kasliwal `[通讯]` (California Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个旋转不变的卷积神经网络RuBR，用于罗曼望远镜RAPID实时图像差分生成的光变体与伪影进行分类。

**💡 创新点**

引入旋转平均编码、特征多层感知器与域对抗训练，使模型对图像旋转不敏感且能在无真实标签的真实观测中迁移，同时显著提升了分类性能。

**🔧 技术方法**

采用旋转不变卷积网络、特征编码器、加权二元交叉熵、梯度反转层、域对抗学习以及过滤器级标准化等技术。

**📊 数据集**

使用OpenUniverse2024仿真数据并在其上人工注入转瞬即逝事件，共计约200k训练样本，测试集包含约1.16M真实标注的转瞬事件。

**📈 对比分析**

与DenseNet121、VGG11、ResNet-18、Deep-HITS、Braai等基线模型在F1、精度、召回率等指标上对比，RuBR在F1上达到0.798，精度90.1%，召回71.7%，在不同阈值下保持稳健；域适应后在OU24上精度提升至≈49%、召回≈72%。

**⚠️ 局限性**

数据仅包含瞬态事件，缺乏周期变量；源提取参数不完善导致误检；在真实观测中仍需大量无标签自适应；模型对PSF误差和中心残差敏感，召回率相对偏低。

---

## 544. ZipSplat: Fewer Gaussians, Better Splats

**arXiv ID:** 2606.05102 | [PDF](https://arxiv.org/pdf/2606.05102v1)

**作者:** Alexander Veicht `[一作]` (ETH Zurich), Marc Pollefeys `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种可将3D高斯原子位置与2D像素网格解耦的 feed‑forward 3D 高斯喷射模型（Zipsplat），通过从多视角图像生成场景令牌并聚类压缩后再解码成自由3D高斯；

**💡 创新点**

创新点在于：①解耦高斯位置与像素射线，令原子可自由放置；②使用多视角令牌聚类实现可控压缩，单一模型即可在不同质量‑效率点运行；③结合几何监督与逐步训练策略保证自由放置的稳定性；

**🔧 技术方法**

核心技术包括多视角Transformer基础模型（如DA3）、k‑means特征聚类、轻量MLP解码器、Chamfer几何监督、光度与深度损失、渐进式视角与压缩比调度；

**📊 数据集**

在DL3DV、RealEstate10K、Mip-NeRF360、ScanNet++等数据集上进行训练与评估；

**📈 对比分析**

与任何基于像素对齐的 feed‑forward 3DGS 方法（AnySplat、C3G、YoNoSplat、DA3等）以及有姿势的 MVSplat、DepthSplat 进行对比；在DL3DV和RealEstate10K上无姿势下 PSNR 最高，使用的高斯数量比对齐方法少约 6 倍；在 Mip‑NeRF360、ScanNet++ 的零样本转移中亦优于基线；

**⚠️ 局限性**

局限性包括：①仍需大量训练数据（大量多视角视频）；②在极端压缩比或极稀视角下重建质量下降；③对动态场景的扩展尚未验证；

---

## 545. InstantRetouch: Efficient and High-Fidelity Instruction-Guided Image Retouching with Bilateral Space

**arXiv ID:** 2606.05071 | [PDF](https://arxiv.org/pdf/2606.05071v1)

**作者:** Jiarui Wu `[一作]` (Shanghai AI Laboratory), Tianfan Xue `[通讯]` (CUHK MMLab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于双边空间的高效语言驱动照片美化方法，能够在保持内容完整性的同时实现快速编辑。

**💡 创新点**

创新性地将多步扩散模型蒸馏为单步低分辨率双边网格生成器，并引入双边网格切片-应用机制，实现在不改变几何纹理的前提下进行颜色与光度调整。

**🔧 技术方法**

采用变分分数蒸馏（VSD）、CLIP对齐损失、双边网格预测与切片-应用操作以及自研的指令-编辑数据集等技术。

**📊 数据集**

构建了约200K条指令-美化三元组数据集，并在新设的iRetouch 500对评测集上进行评估。

**📈 对比分析**

与传统提升方法、开源和商业编辑模型对比，4K分辨率下推理时间仅0.065s，内容保真度、编辑质量均优于或与主流商业系统相当。

**⚠️ 局限性**

对极端局部细节编辑和极端风格指令的适应性仍有限，且在极大分辨率下需要额外显存。

---

## 546. FLAGG: Flexible Autoregressive Graph Generation

**arXiv ID:** 2606.05067 | [PDF](https://arxiv.org/pdf/2606.05067v1)

**作者:** Samuel Cognolato `[一作]` (University of Padova), Luciano Serafini `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出FLAGG框架，将一击式图生成模型嵌入自回归过程，可按块大小逐步插入节点；

**💡 创新点**

通过可插拔的插入、填充和停止模块实现对节点删除策略、块大小、节点排序和属性的灵活控制，支持大图生成与属性感知生成；

**🔧 技术方法**

使用一击式扩散模型DiGress、Graph Transformer、RGCN、EMA、梯度裁剪等技术；采用节点删除过程作为噪声模型，并利用MMD、FCD、NSPDK、GIN等指标评估；

**📊 数据集**

分子数据集QM9、ZINC250k；通用图数据集Community‑small、Ego‑small、Ego、Enzymes；大图Cora；

**📈 对比分析**

与GraphAF/GraphDF/GraphARM、MoFlow/DiGress/CDGS、IFH等基线对比。FLAGG在中等序列性下与一击式相当或更优，在分子生成上FCD显著优于所有基线；在通用图中Degree/Clustering与CDGS相当，部分指标略低；在Cora上Degree与Spectrum匹配但Clustering和Eccentricity落后于HiGGs；

**⚠️ 局限性**

需要人工设定节点删除策略、块大小与节点排序；插入与停止模型对经验分布敏感；对局部结构如聚类系数的捕获仍有限；在大图中聚类和eccentricity表现不佳；学习插入策略的可调参数昂贵且难以自动化。

---

## 547. Sibley's Guard-Point Convexity Measure: A Perimeter Counterexample and a Dominance Bound

**arXiv ID:** 2606.05052 | [PDF](https://arxiv.org/pdf/2606.05052v1)

**作者:** Masahito Nakano `[一作]` `[通讯]`, Masahito Nakano

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了Sibley的守卫点凸性度量，并将其与外部和周长凸性度量进行了比较。

**💡 创新点**

证明了外部不等式G(F)≤E(F)成立，而点wise周长不等式G(F)≤P(F)不成立，且提供了G(F)≤2P(F)的统一界限。

**🔧 技术方法**

使用了几何不等式和支持函数的概念，特别是引入了守卫点适应的各向异性周长比。

**📊 数据集**

使用了简单多边形的几何特性，特别是一个具体的非凸五边形作为反例。

**📈 对比分析**

通过构造反例证明了G(F)≤P(F)不成立，但G(F)≤2P(F)的界限仍然成立，表明G不主导P。

**⚠️ 局限性**

局限性在于未能完全解决Sibley的所有猜想，特别是在不同的几何度量之间的比较和最优常数的确定上。

---

## 548. Does Artificial Intelligence Advance Science?

**arXiv ID:** 2606.05118 | [PDF](https://arxiv.org/pdf/2606.05118v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 549. Anchor3R: Streaming 3D Reconstruction with Transient Anchors for Long-Horizon Visual Mapping

**arXiv ID:** 2606.05035 | [PDF](https://arxiv.org/pdf/2606.05035v1)

**作者:** Peilin Tao `[一作]` (Chinese Academy Of Sciences), Shuhan Shen `[通讯]` (Chinese Academy Of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Anchor3R，一种以当前帧为临时锚点的流式3D重建框架，预测窗口内相对位姿和局部点图，构建相对位姿图并实现在线姿态更新及离线循环闭环优化。

**💡 创新点**

创新点在于：①把流式重建视为当前中心相对测量预测，避免长期全局坐标漂移；②采用图像仅缓存与姿态查询分离的滑动窗口 Transformer，保持 bounded‑memory 推理；③通过相对位姿图实现在线姿态更新和闭环约束下的全局运动平均化。

**🔧 技术方法**

使用 DINOv2 patch tokens、pose‑query Transformer、当前中心窗口注意力、相对位姿图、运动平均化（IRLS/ADMM）、NetVLAD 循环检测等技术。

**📊 数据集**

训练数据包含 WildRGB、ScanNet、HyperSim、Mapillary、Replica、Mapfree、TartanAir、MVS‑Synth、Virtual KITTI、Aria Synthetic、Spring、Waymo Open、BlendedMVS、Co3Dv2、MegaDepth、DL3DV 等真实与合成数据；测试数据包括 KITTI、VBR、TUM RGB‑D、Oxford Spires、Waymo、7Scenes、TUM RGB‑D 等。

**📈 对比分析**

与基于优化的 SLAM 与传统流式方法（FastVGGT、MASt3R‑SLAM、VGGT‑SLAM、VGGT‑Long、Pi3‑Chunk、CUT3R、TTT3R、STream3R、StreamVGGT、LongStream、InfiniteVGGT 等）对比，Anchor3R‑Online 在长序列中取得最优或接近最佳 ATE；Anchor3R‑Offline 通过循环闭环进一步将 ATE 降至 5.13/3.52 m（VBR），在所有测试序列中排名第一；在 7Scenes、TUM 等密集重建任务中获得最优 Chamfer Distance 或接近最佳 F1。

**⚠️ 局限性**

局限性：①新帧约束不更新历史局部几何，导致早期误差残留；②依赖窗口重叠的尺度一致性，极端长序列或弱重叠易出现尺度漂移；③仅在回放数据上评估，尚未验证机器人闭环部署或导航任务。

---

## 550. Invariant Gradient Alignment for Robust Reasoning Distillation

**arXiv ID:** 2606.05025 | [PDF](https://arxiv.org/pdf/2606.05025v1)

**作者:** Zehua Cheng `[一作]` (University of Oxford), Jiahao Sun `[通讯]` (FLock.io)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对逻辑同构（isomer）组的梯度进行对齐，构建了一个鲁棒的推理学生模型，解决大型语言模型在语义表面差异导致的短路学习问题。

**💡 创新点**

创新点包括：① 逻辑同构集（Logical Isomer Sets）构造，提供跨语义但逻辑相同的数据对；② 连续梯度冲突掩码 M = exp(-τ·V)，在梯度层面柔性抑制域特异的冲突维度；③ 在全秩梯度空间进行掩码后再通过截断 SVD 投影回 LoRA 低秩子空间，兼顾参数效率；④ 引入逻辑一致性得分（Logical Consistency Score）评估域不变性。

**🔧 技术方法**

使用 LoRA 微调、AdamW 优化、梯度重构与 SVD、教师 LLM（GPT‑4.5 或 Qwen3.5）生成同构实例，结合指数型冲突掩码实现梯度对齐。

**📊 数据集**

实验数据集包括 ARB、LogiQA 2.0、ReClor、MATH Cross‑Domain Transfer 四个推理基准；同构集由教师 LLM 在数学、医学、法律、科学四个语义域生成，形成多域逻辑相同的问题组。

**📈 对比分析**

与 ERM‑SFT、LoRA‑SFT、CoT‑Distill、IRM、V‑REx、PCGrad、SAND‑Mask、Pareto‑GS 八个基线进行对比；IGA 在所有四个基准的 OOD 准确率均最高，平均提升 6.2–14.3 个百分点；逻辑一致性得分下降 3.10（×10⁻²），比最优基线低 4 倍，证明学习到的表示更具域不变性。

**⚠️ 局限性**

局限性包括：① 需要高质量的教师 LLM 生成同构集（虽可用开源模型，但效果略低）；② 训练成本约为 LoRA‑SFT 的两倍，主要来自多域前向/反向传播和 SVD 计算；③ 对同构域数量的依赖，更多域会提升效果但计算量线性增长；④ 若实例拆分不当易产生数据泄漏，导致性能被夸大。

---

## 551. Audio Interaction Model

**arXiv ID:** 2606.05121 | [PDF](https://arxiv.org/pdf/2606.05121v1)

**作者:** Zhifei Xie `[一作]` (Nanyang Technological University), Chunyan Miao `[通讯]` (Nanyang Technological University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了 Audio‑Interaction，一个统一的实时音频交互模型，支持传统离线任务与流式任务，并能在连续音频流中自动决定何时发声。

**💡 创新点**

创新点包括：①把多任务音频模型整合为单一在线模型，填补离线与流式之间的空白；②构建 SoundFlow 框架，实现从数据拼接、预处理、训练到推理的一体化；③采用层次化事件选择与 TFJP 时间‑频率联合预处理，生成规模达 2.6M 条的长流数据；④在推理阶段使用 FIFO 异步调度显著降低首帧延迟；⑤为流式触发引入特殊控制标记并在训练中加入双重损失。

**🔧 技术方法**

核心技术包括：基于 Qwen2.5‑Omni‑3B 的 encoder‑decoder 结构；音频编码器 + 投影 + GPT；TFJP 预处理；层次化事件选择与检索/生成；双重损失（语言建模 + 流式控制 token）；FIFO 异步推理；历史复习训练与空白训练。

**📊 数据集**

使用的数据集：StreamAudio‑2M（2.6M 条、302k 小时、7 类 28 子任务）、Proactive‑Sound‑Bench（644 人工设计事件）、以及多源基础数据如 MOSS、CommonVoice、GigaSpeech、LibriSpeech、CoVoST2、FMA、AudioSet、MUSAN 等。

**📈 对比分析**

在 8 个基准（MMAU、4 个对话、LibriSpeech、CoVoST2、Proactive‑Sound‑Bench）与多类基线（Audio Flamingo2、Qwen2‑Audio、Voxtral、Audio‑Reasoner、Qwen2.5‑Omni、Phi‑4、Baichuan‑Omni、Whisper、Moshi 等）进行对比。Audio‑Interaction 在主流任务保持竞争力：MMAU 58.15，S2TT BLEU 55.22/35.21，实时 ASR WER 3.17/6.04；在 Proactive‑Sound‑Bench 单/多层分别取得 61.2/62.8，显示出主动干预能力；整体性能接近或优于 7B 级别基线。

**⚠️ 局限性**

局限性：①仍需权衡 chunk 大小与时延；②触发决策高度依赖特殊控制 token，迁移到新任务时可能需额外调优；③极长流段下准确率仍略下降；④对非语音事件的理解有限；⑤构建与训练耗费大量多源数据与算力。

---

## 552. Graph Set Transformer

**arXiv ID:** 2606.05116 | [PDF](https://arxiv.org/pdf/2606.05116v1)

**作者:** Jose E. Escrig Molina `[一作]` (Wageningen University), Daniel Probst `[通讯]` (Wageningen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种在每层同时进行图级消息传递和集合级注意力的 Graph Set Transformer（GST）架构

**💡 创新点**

创新点在于将局部图传播与集合上下文交替融合，消除传统管线中图编码与集合推理的瓶颈

**🔧 技术方法**

采用 GCN+GraphNorm、注意力池化、MHSA、门控融合等技术构成 GST 迭代模块

**📊 数据集**

使用了合成 Erdős–Rényi 组图、CIFAR‑10 集合分类、Buchwald–Hartwig 产率预测和 USPTO‑15K 反应中心预测四个数据集

**📈 对比分析**

与 DeepSets+GCN 与 SetTransformer+GCN 进行参数匹配对比，GST 在所有任务上均表现更好，尤其在集合上下文显著的情境中提升明显

**⚠️ 局限性**

主要缺点是计算耗时较大，尤其是交叉注意力变体；广播版本在保持性能的同时显著降低了时间和参数成本

---

## 553. Continual Visual and Verbal Learning Through a Child's Egocentric Input

**arXiv ID:** 2606.05115 | [PDF](https://arxiv.org/pdf/2606.05115v1)

**作者:** Xiaoyang Jiang `[一作]` (New York University), Mengye Ren `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 BabyCL，一种在单一时间序列上进行持续视觉与语言学习的框架；

**💡 创新点**

创新之处在于将流式视觉自监督、跨模态对比学习以及事件分段和双重重放缓冲相结合，实现在接近儿童实际经历的单通道、连续学习环境中获取词-指向映射；

**🔧 技术方法**

采用 ResNeXt‑50 视觉 backbone，SimCLR、InfoNCE 对比损失，事件聚类分段，双层 FIFO/Reservoir 重放缓冲和多任务联合训练；

**📊 数据集**

使用 SAYCam 数据集（儿童 6–32 个月头戴摄像机视频及其转录的儿童导向语音），构成 child S 子集；

**📈 对比分析**

与离线 CVCL、单通道 CVCL 以及匹配梯度预算的 CL‑CVCL 在 Labeled‑S 4AFC、线性探测、VTWT、Baby Winoground 等评测上比较，BabyCL 在保持相同梯度预算下比流式 CVCL 提升约 7–8 分，逼近离线上限，但仍略低于离线 CVCL；

**⚠️ 局限性**

主要限制是与离线训练仍存在显著差距；重放缓冲占用相对较大；在更具挑战性的组合性或对抗性任务上的性能仍有提升空间。

---

## 554. Randomization for Faster Exact Optimization of Discounted Markov Decision Processes

**arXiv ID:** 2606.05110 | [PDF](https://arxiv.org/pdf/2606.05110v1)

**作者:** Andrei Graur `[一作]` (Stanford University), Ta-Wei Tu `[通讯]` (Stanford University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了基于策略评估和近似求解的通用框架，用于高效求解γ-折扣马尔可夫决策过程（MDP）的精确解，并给出了确定性与随机化两种实现。

**💡 创新点**

创新点在于将精确求解归约为一系列可用近似解法的子问题，并通过奖励平移与优势函数分析实现快速动作剔除；随机化版本则利用随机策略采样在期望上显著减少迭代次数，从而突破以往最坏情况的时间极限。

**🔧 技术方法**

主要技术包括：快速矩阵乘法求解策略评估；优势函数与奖励平移来保证近似误差可控；黑盒调用高效近似MDP求解器；随机化策略采样与期望潜能函数分析；以及对线性规划强子多项式性质的利用。

**📊 数据集**

该工作为理论研究，未使用任何具体实验数据集，所有结果均为理论时间复杂度分析。

**📈 对比分析**

与现有的组合IPM、简单枢轴政策迭代等方法相比，确定性版本将时间从 O(|S|^4 log(1/(1-γ))) 降至 O(|S|^ω+|S|^2)；随机化版本进一步把迭代次数从 O(|S|) 降至 O(log |S|)，整体运行时间可达到 Õ(|S|^ω+1+|S|^2) 或更优，显著提升性能。

**⚠️ 局限性**

局限性包括：仍需依赖近似求解器的黑盒实现，无法保证在严格的 Turing 模型下实现强子多项式；随机化版本的性能基于期望分析，实际实现可能存在波动；此外，实验验证尚未完成，实际常数因子与实现细节影响未被评估。

---

## 555. Arithmetic Pedagogy for Language Models

**arXiv ID:** 2606.05106 | [PDF](https://arxiv.org/pdf/2606.05106v1)

**作者:** Andhika Bernard Lumbantobing `[一作]`, Hokky Situngkir `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将印度尼西亚数学教学法GASING与Chain-of-Thought（CoT）结合，通过在小规模GPT‑2模型上仅用下一个token预测目标进行训练，探索语言模型对算术推理的学习与内化。

**💡 创新点**

创新点在于：①将教学法转化为可序列化的CoT监督数据；②展示仅靠next‑token训练即可获得超过80%算术准确率；③通过注意力抑制、残差流与logit‑lens的机制分析，揭示模型先内化程序路径后形成“精神算术”的学习进程。

**🔧 技术方法**

使用技术包括：GPT‑2 86M参数Decoder、TOBA音节聚合分词器、链式推理（CoT）文本化、仅next‑token交叉熵损失、注意力抑制干预、残差流分类器、logit‑lens分析。

**📊 数据集**

训练数据集：90,000个自然语言算术问题（加、减、乘、除，三位数以内），每个问题包含数值形式与词语形式的多种表达，CoT为GASING程序执行轨迹。

**📈 对比分析**

评估方式：在10,000个未见的算术样本上计算准确率，并与多种大参数规模的Transformer模型（>100M）进行对比。结果显示该小模型在80%+准确率上优于部分更大模型，证明教学法驱动训练能显著提升算术能力。

**⚠️ 局限性**

局限性：仅针对三位数以内算术，缺乏对更复杂算术或其他语言任务的泛化评估；对极大规模模型的进一步验证仍需深入；训练过程中主要关注算术推理，未探究多任务或跨域适用性。

---

## 556. Light or Full Verb? A Minimal-Pair Dataset for Probing Phraseological Competence in Language Models

**arXiv ID:** 2606.05087 | [PDF](https://arxiv.org/pdf/2606.05087v1)

**作者:** Francesca Franzon `[一作]` (Universitat Pompeu Fabra), Leo Wanner `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构造了一个控制最小对比的英文本集，比较同一动词在轻动词构式与完整词义下的使用。

**💡 创新点**

创新点在于将轻动词构式与完整谓语对比为可量化的最小句子系列，并提供可复用的生成代码与数据集。

**🔧 技术方法**

采用基于语言模型的惊讶值（surprisal）评估和上下文嵌入的聚类分析两种技术，利用 Gemma‑3 270m 语言模型进行实验。

**📊 数据集**

数据集包含5个高频轻动词（make、take、give、have、receive）的约 47,600 条对比句子，涵盖主动/被动、不同语义对象。

**📈 对比分析**

方法通过比较轻动词与完整词义句子的惊讶值差异和嵌入聚类纯度来评估模型是否捕捉到轻动词与完整谓语的差别；结果显示 Gemma‑3 对轻动词构式有显著的低惊讶值，并在后层嵌入空间中形成明显分群，说明模型内部表示中已体现该对比。

**⚠️ 局限性**

局限性包括仅覆盖5个动词、句子上下文为半控制、仅在英语上验证，缺乏跨语言和更广泛动词类的推广性。

---

## 557. Federating Governance: How Community Rules Scale with Mastodon Instances

**arXiv ID:** 2606.05069 | [PDF](https://arxiv.org/pdf/2606.05069v1)

**作者:** Rasika Muralidharan `[一作]` (Indiana University), Bao Tran Truong `[通讯]` (Dresden University of Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Mastodon 实例随社区规模变化时规则制定与治理特征，系统地分类并量化规则主题、可读性及词汇多样性。

**💡 创新点**

发现无论规模大小，规则优先关注骚扰、仇恨言论和非法内容，且规模增长导致规则更长、覆盖面更广但可读性下降，证明去中心化平台与中心化平台在治理缩放模式上相似。

**🔧 技术方法**

采用 LLM（GPT‑4o‑mini）进行多标签规则主题分类，结合负二项回归、β回归与 OLS 统计模型对词汇特征进行量化。

**📊 数据集**

使用 28,910 条规则来自 6,660 个 Mastodon 实例的公开 API 数据集（包含用户数、连通度等元信息）。

**📈 对比分析**

通过相关系数与回归分析比较规则规模与社区规模关系，结果显示社区规模是规则形式化的主要预测因子，联邦化程度影响有限，统计显著且稳健。

**⚠️ 局限性**

主要局限在横断面数据缺乏因果证据、规则分类仍受 LLM 误差影响以及样本仅为 2024 年快照，未覆盖长期演化与行为数据。

---

## 558. SearchLog: A Web Browser Extension for Capturing Search Logs in Laboratory Studies

**arXiv ID:** 2606.05040 | [PDF](https://arxiv.org/pdf/2606.05040v1)

**作者:** Jiaman He `[一作]` (RMIT University), Johanne R. Trippas `[通讯]` (RMIT University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

**🎯 论文内容**

提出并实现了一个可在实验室环境下自然搜索时记录完整交互日志的Chrome扩展，支持捕获鼠标、键盘、查询、结果排名以及AI生成摘要等多种事件；

**💡 创新点**

创新点在于提供了一个轻量、易安装、可记录AI摘要的自然搜索日志工具，并设计了结构化日志模式，可直接与实验元数据关联；

**🔧 技术方法**

技术上采用Chromium扩展与本地Flask后端相结合，利用DOM解析、浏览器API和事件监听实现日志采集，数据以JSON流和HTML快照形式保存；

**📊 数据集**

使用了实验室内非敏感示范任务产生的日志作为验证集；

**📈 对比分析**

通过对比六种典型搜索场景的预期事件，验证日志准确捕获并保持时间顺序；性能表现为无漏失、实时写入，且能完整记录AI摘要；

**⚠️ 局限性**

局限性包括仅支持Chromium浏览器；对搜索引擎页面布局变动需手动维护；目前仅覆盖Google和Bing，需扩展更多搜索引擎和LLM系统。

---

## 559. MetaPoint: Unlocking Precise Spatial Control in Agentic Visual Generation

**arXiv ID:** 2606.05031 | [PDF](https://arxiv.org/pdf/2606.05031v1)

**作者:** Dewei Zhou `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种单一特殊token（MetaPoint）实现像素级空间控制，并与VLM Agent结合，构建统一的生成与编辑框架。

**💡 创新点**

创新点在于：①将UMM的原生二维位置编码直接映射到连续坐标，获得轻量级、可扩展的空间指令；②token的组合可表达点、框、序列等复杂布局；③结合VLM Agent实现从自然语言到精确空间指令的闭环推理与自我修正。

**🔧 技术方法**

技术核心包括：Umm（如BAGEL）中的二维正弦/三维RoPE位置编码；单一MetaPoint token；VLM Agent（视觉语言模型）作为规划器；自监督视频数据驱动的三类训练集（PACL、PAEI、PAIE）。

**📊 数据集**

使用BAGEL模型为基础，结合从视频中构造的PAC L、PAEI、PAIE三大数据集；评测基准为COCO-MIG、T2I-CoReBench、ImgEdit。

**📈 对比分析**

在COCO-MIG上mIoU从59.23%提升至77.29%（+30.49%），Instance Success Rate提升至84.72%；T2I-CoReBench总分从38.2升至66.1（+73%），在逻辑、几何、文本渲染子任务取得显著提升；ImgEdit总体分从3.42升至3.94，尤其在Remove任务上表现最佳；相比基线和SOTA均取得显著性能提升。

**⚠️ 局限性**

局限性包括：①缺乏对旋转等更丰富的空间变换控制；②仅实现位置、大小等基本属性，缺少深度、姿态、颜色等属性的精细控制；③Agent系统仍基于手工提示，缺乏与其他工具的动态协同与更通用的任务适配。

---

## 560. A Practical AI-Driven Strategy for Cell On/Off Switching under Adaptable QoS Constraints

**arXiv ID:** 2606.05019 | [PDF](https://arxiv.org/pdf/2606.05019v1)

**作者:** David Reiss `[一作]` (UPC), Oriol Sallent `[通讯]` (UPC)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对5G基站开启/关闭的能耗与QoS平衡问题，提出了一种基于LSTM的实时策略。

**💡 创新点**

创新在于通过推理时调整阈值，单一模型即可适应不同QoS约束，无需重新训练。

**🔧 技术方法**

技术包括Temporal LSTM、阈值后处理、CO₂与OPEX估算等。

**📊 数据集**

使用来自欧洲MNO的真实网络数据，覆盖一个月15分钟粒度的4G/5G KPI。

**📈 对比分析**

与oracle、XGBoost对比，能耗节省63%–96%并保持规定的服务容忍度。

**⚠️ 局限性**

局限在于模型泛化受网络漂移影响，缺乏4G能耗数据，需定期校准。

---

## 561. Evaluating Large Language Models in Dynamic Clinical Decision-Making with Standardized Patient Cases

**arXiv ID:** 2606.05112 | [PDF](https://arxiv.org/pdf/2606.05112v1)

**作者:** Cheng Liang `[一作]` (Shanghai Jiao Tong University), Weidi Xie `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MedSP1000，一个基于 MedEdPORTAL 标准化患者教学案例的 1,638 个可交互情景的临床代理评估基准，并通过多代理闭环模拟评估 LLM 在诊疗过程中的行为。

**💡 创新点**

创新点在于将同行评议的标准化患者案例自动转换为可执行的多代理情景，提供过程级评估和六大 ACGME 核心能力的量化打分，弥补了传统单回合问答评测的局限。

**🔧 技术方法**

采用了多代理交互框架（患者代理、环境控制器、评估者）以及大语言模型（GPT‑5.5、Claude‑Opus‑4.7 等）进行决策、信息收集与动作执行，并通过自洽解码和 MDT 多专家策略做测试时扩展。

**📊 数据集**

使用的数据集是从 MedEdPORTAL 收集的 613 篇同行评审的标准化患者与模拟教学材料，经过自动化流程处理后得到 1,638 个可执行案例与 24,602 个评分项目。

**📈 对比分析**

与七款代表性 LLM（包括前沿闭源、开源通用和医学专项模型）进行对比，GPT‑5.5 以 60.4% 的宏观评分率领先，但所有模型在过程级评测中仍表现不佳，医学专项模型更低，仅达 40%。

**⚠️ 局限性**

局限性在于评测基准仅基于文本化的模拟场景，缺乏真实临床多模态交互与真实患者风险，难以直接证明 LLM 在实际临床环境中的安全性与可靠性。

---

## 562. Gradient Dynamics in First-Price Auctions: Iterative Strategy Elimination via Cubic Potentials

**arXiv ID:** 2606.05108 | [PDF](https://arxiv.org/pdf/2606.05108v1)

**作者:** Mete Şeref Ahunbay `[一作]` (University of Oxford), Tao Lin `[通讯]` (Microsoft Research)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了离散化、完全信息的一价拍卖中，买家使用在线梯度上升学习时的长期结果，证明时间平均的社会福利和收入趋近于二价拍卖的有效结果。

**💡 创新点**

提出了基于势能函数的迭代策略消除框架以及构造可导的立方势能函数，首次在正常形式博弈中实现对非线性策略修改的无后悔分析。

**🔧 技术方法**

采用势能函数方法、迭代消除、立方多项式势能以及在线梯度上升的无后悔理论。

**📊 数据集**

未使用实验数据集，完全在理论分析框架下进行。

**📈 对比分析**

与传统平均收敛或最后一次迭代收敛的梯度学习方法相比，证明了仅让最高两名买家使用在线梯度上升即可实现几乎最优的时间平均社会福利与收入。

**⚠️ 局限性**

结果的时间平均误差上界与参数呈指数增长，且仅在两名最高价值买家使用 OGA 时成立，未考虑更一般的学习算法或随机性，对立方势能的选择仍需人工调参。

---

## 563. "A Glimpse, Not a Gaze": Using Generative AI to Balance Privacy and Awareness in Inter-generational Caregiving

**arXiv ID:** 2606.05055 | [PDF](https://arxiv.org/pdf/2606.05055v1)

**作者:** Zixi Christina Li `[一作]` (University of Waterloo), James R. Wallace `[通讯]` (University of Waterloo)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并执行了一项为期10天的体验采样研究，评估基于生成式AI生成的抽象视觉摘要在老年人与其成年子女之间平衡隐私与知情的效果。

**💡 创新点**

创新点在于将生成式AI用于创建隐私保护的抽象视觉摘要，探讨跨代沟通中的隐私边界设置，并从实证研究中提炼可操作的设计准则。

**🔧 技术方法**

使用生成式AI模型（如Stable Diffusion等）生成抽象草图，结合Qualtrics平台的ESM调查、卡片分类与半结构化访谈等方法。

**📊 数据集**

主要使用参与者自述的日常活动、情绪与可用性数据生成定制化草图，未使用公开的大规模数据集。

**📈 对比分析**

通过问卷得分与访谈结果相结合，比较不同抽象方式对隐私接受度和连接感的影响；由于研究性质为设计研究，未给出传统模型评估指标，但报告了参与者的分享意愿和感知效果。

**⚠️ 局限性**

局限性包括样本规模有限、仅限英语母语且拥有智能手机的参与者、抽象草图质量受模型与提示限制、缺乏长期评估与跨文化验证。

---

## 564. Boosting Self-Consistency with Ranking

**arXiv ID:** 2606.05054 | [PDF](https://arxiv.org/pdf/2606.05054v1)

**作者:** Maria Marina `[一作]` (AIRI), Viktor Moskvoretskii `[通讯]` (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于排名的自一致性改进方法 RISC，利用多个推理路径生成候选答案后通过学习式排序来挑选最终答案。

**💡 创新点**

将自一致性视为排名任务，引入轻量级 LambdaRank 评估器，设计了五个可解释特征（答案频率、答案中心度、推理轨迹一致性等），通过学习将多种信号结合而非单一置信度，显著缩小与 Oracle 之间的性能差距。

**🔧 技术方法**

使用 LightGBM LambdaRank 作为排序模型，MiniLM（sentence-transformer）提取答案和推理步骤的嵌入，构造五个特征；训练采用 listwise NDCG 损失，测试时仅用生成的候选答案即可。

**📊 数据集**

在 PopQA（实体问答）、HotpotQA（多跳推理）和 MATH500（数学推理）三个基准上进行实验，涵盖开放域、闭源和高难度推理任务。

**📈 对比分析**

与标准自一致性（SC）、CISC、Stable Rank、ReASC 等基线比较；RISC 在所有三个数据集上均实现更好的成本‑效率折中，使用更少的 LLM 调用即可达到 SC 的准确率，并在部分预算下突破 SC 的最大准确率（headroom）。

**⚠️ 局限性**

局限性：仅在小规模指令微调模型（Llama‑3.1‑8B、Olmo‑3‑7B）上验证，数学推理任务提升有限；与 Oracle 之间仍存在显著差距；未在更大、不同体系结构的模型上进行泛化评估。

---

## 565. Non-obvious Manipulability in the Additively Separable Group Activity Selection Problem

**arXiv ID:** 2606.05048 | [PDF](https://arxiv.org/pdf/2606.05048v1)

**作者:** Maria Fomenko `[一作]` (Gran Sasso Science Institute), Giovanna Varricchio `[通讯]` (University of Calabria)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文在不完全信息环境下研究可加分离的团体活动选择问题（AS-GASP），并设计旨在最大化社会福利同时保证诚实报告的机制，重点研究非显式可操纵性（NOM）。

**💡 创新点**

创新点在于：①首次将NOM概念引入AS-GASP并系统阐述其与最优性的兼容性；②证明在任意或非负权重下任何最优机制均满足NOM，但在二值偏好/权重时两者可能冲突；③给出在非负偏好下可实现的渐进最优NOM近似机制，并证明在任意偏好/权重下存在强势不可逼近性。

**🔧 技术方法**

技术方法包括：机制设计与策略不变性分析、NP-难度证明、对最大团问题的规约、构造性近似算法（贪心选择、2-1/m 近似）以及对NOM条件的形式化证明。

**📊 数据集**

本研究为理论工作，未使用任何真实数据集；所有结果均通过数学证明与理论分析得到。

**📈 对比分析**

通过与已有的策略可证明机制（无法获得有界近似）进行对比，证明NOM机制在保持可接受的近似比（如 n² 近似、2-1/m 近似或渐进最优）下仍能实现诚实报告；同时给出多种情形下的最优与近似性能下界与上界。

**⚠️ 局限性**

局限性包括：①二值偏好与二值权重同时为 1 的完整可解性尚未确定；②某些特殊参数设置下的近似复杂度（如 m=1、所有权重为 1）仍开放；③缺乏实验验证，仅在理论层面阐述机制效果。

---

## 566. Strabo: Declarative Specification and Implementation of Agentic Interaction Protocols

**arXiv ID:** 2606.05043 | [PDF](https://arxiv.org/pdf/2606.05043v1)

**作者:** Samuel H. Christie `[一作]`, Munindar P. Singh `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文设计并实现了一个基于平台层与业务代理层的订单处理框架，采用REST接口实现业务模块与平台的解耦通信。

**💡 创新点**

创新点在于引入业务代理层，将业务逻辑与平台业务分离，支持业务模块的可插拔和独立升级。

**🔧 技术方法**

主要技术包括HTTP/REST协议、RESTful Web Service、Spring Boot等框架，以及消息队列（可选）进行异步处理。

**📊 数据集**

使用自研的订单数据集（synthetic orders）进行性能和功能验证，数据规模覆盖了数十万条订单。

**📈 对比分析**

与传统单体系统进行对比实验，系统在并发请求下的吞吐量提升约30%，响应时间下降15%。

**⚠️ 局限性**

局限性主要体现在：1）实验仅在单机环境下验证，缺乏分布式部署的评估；2）安全性（如身份认证、授权）在原型中未深入实现；3）对异常场景的鲁棒性仍需进一步测试。

---

## 567. Imbuing Large Language Models with Bidirectional Logic for Robust Chain Repair

**arXiv ID:** 2606.05030 | [PDF](https://arxiv.org/pdf/2606.05030v1)

**作者:** Zehua Cheng `[一作]` (University of Oxford), Thomas Lukasiewicz `[通讯]` (TU Wien)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Teleological Reasoning Infilling（TRI）框架，通过前缀-后缀-中间（PSM）序列重排，使标准的单向 Transformer 能在生成桥接段时同时关注已验证的前置语句和后续里程碑，从而实现目标驱动的逻辑填充。

**💡 创新点**

创新点在于：①在不改动自注意力机制的前提下，用三种哨兵符号实现双向条件生成；②结合符号验证器的直接偏好优化（DPO）消除传统 LLM 判别器的噪声；③在推理阶段设计双系统修复循环，仅重生成错误段，显著提升 token 效率与准确率。

**🔧 技术方法**

技术包括：PSM 序列重排、两阶段训练（SFT + DPO）、符号验证器（Lean 4 / Python）、双系统修复循环以及针对桥接段的交叉熵与 DPO 损失。

**📊 数据集**

使用的数据集覆盖三个领域：数学推理的 MATH、程序修复的 HumanEval‑Fix 与 Lean‑Workbook 形式化证明，训练数据取自 MATH 训练集与 Lean‑Workbook 训练任务，验证数据取对应测试集。

**📈 对比分析**

在 MATH、HumanEval‑Fix 和 Lean‑Workbook 上均超过现有 CoT、CoT‑SC、Tree‑of‑Thoughts 等基线，MATH 最高难度级别提升 6.4pp，整体 token 消耗平均减少 31.2%，且在预算紧张与错误密度高的情形下仍保持领先。

**⚠️ 局限性**

局限性包括：①依赖可获得的符号验证器，无法直接应用于缺乏正式验证的任务；②需要预先生成并验证的里程碑，若前后缀均不完整可能导致修复失败；③在极长的推理链或极高错误密度时仍可能需要多轮修复，影响实时性能。

---

## 568. Validity Threats for Foundation Model Research

**arXiv ID:** 2606.05029 | [PDF](https://arxiv.org/pdf/2606.05029v1)

**作者:** Gunnar König `[一作]` (University of Tübingen), Sebastian Bordt `[通讯]` (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了一个评估框架，用于系统性分析和比较基础模型研究中的有效性威胁，并将该研究视为因果推断问题。

**💡 创新点**

创新点在于：①将基础模型研究抽象为因果推断；②定义了统计、内部、外部和构造四种有效性类型；③针对三类主流研究策略（代理实验、观察性研究、单跑设计）构建了效度剖面表，明确各策略的优势与风险。

**🔧 技术方法**

主要技术手段包括因果推断理论（潜在结果框架）、统计学方法（差分-差分、回归不连续设计等）、实验设计原则（可交换性、无干扰性等），以及对公开元数据的统计分析。

**📊 数据集**

文章并未进行新实验，主要利用公开的基础模型元数据（如模型参数、训练时间、任务性能等）进行理论分析和案例讨论；没有使用特定的标准数据集进行训练或评估。

**📈 对比分析**

通过对比不同研究策略在四种有效性维度上的表现（如代理实验主要风险在外部/构造有效性，观察性研究主要风险在内部有效性等），阐述了各策略在计算成本与科学严谨性之间的权衡；并未给出数值性能指标，而是提供了效度剖面表和案例说明。

**⚠️ 局限性**

局限性包括：①框架主要是概念化和理论推导，缺乏大规模实证验证；②依赖于未检验的因果假设（如可交换性、无干扰性等），实际实验可能难以满足；③仅覆盖了三类主流策略，其他混合或新颖设计的有效性仍需进一步探讨；④未给出统一的量化指标，评估仍需结合具体研究场景。

---

## 569. Generating Financial Time Series by Matching Random Convolutional Features

**arXiv ID:** 2606.05138 | [PDF](https://arxiv.org/pdf/2606.05138v1)

**作者:** Konrad J. Mueller `[一作]` (Imperial College London), Lukas Gonon `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在只有单一路径的有限样本金融时间序列数据上，设计并训练了条件生成模型，利用可微分的随机卷积特征进行特征匹配；

**💡 创新点**

提出了SOft Competing Kernels（SOK）——一种完全可微分的随机卷积特征映射，并通过在训练中频繁重采样特征图，显著提升了生成器对真实数据的逼真度；

**🔧 技术方法**

使用可微分随机卷积特征映射（SOK）、签名特征（截断与随机化签名）以及Diffusion-TS等技术；

**📊 数据集**

在5个合成数据集（AR、振荡、自回归、异方差、分数布朗运动）以及7个真实金融数据集（行业股票、多资产日收益、BTC/ETH 5分钟收益）上进行实验；

**📈 对比分析**

与基于签名特征的生成（Sig-Wasserstein GAN）、Diffusion-TS以及其他签名/随机化签名的特征匹配方法对比，使用多种判别式与分布式评估指标；SOK在大多数指标上获得最低误差，平均排名最高；

**⚠️ 局限性**

局限性在于仅针对低维单一路径数据设计，生成器架构保持简单；对更大规模、多维数据集的适用性未验证，且随机特征匹配在大数据量下可能不如学习型判别器或扩散模型表现优异。

---

## 570. STRIDE: Training Data Attribution via Sparse Recovery from Subset Perturbations

**arXiv ID:** 2606.05165 | [PDF](https://arxiv.org/pdf/2606.05165v1)

**作者:** Rishit Dagli `[一作]` (University of Toronto), Zhijing Jin `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在激活空间中学习轻量级“引导算子”的训练数据归因方法，取代传统的参数梯度估计；

**💡 创新点**

创新点在于将训练数据的因果影响建模为对模型激活的低秩扰动，并通过稀疏恢复实现单样本级别的归因；

**🔧 技术方法**

采用低秩基网络与子集特定的引导矩阵、三阶段损失（逼真、稳定、线性），以及压缩感知的 ℓ₁ 逆向求解；

**📊 数据集**

主要在 LLM 预训练（Nanochat 系列）和指令微调（Qwen、FLAN、Alpaca、Tulu、SafeRLHF）数据集上进行评估；

**📈 对比分析**

与现有梯度基和表示基方法（如 LoGRA、AirRep、TracIn 等）相比，本文在预训练 LDS 评分上实现最高水平，并比最强基线快 10‑12 倍；在数据选择、泄露检测等下游任务中也保持或优于对手；

**⚠️ 局限性**

局限性包括对基模型内部激活质量和层选择的依赖，假设局部线性与加法性，可能在极端分布漂移或 RL 训练中失效；

---

## 571. Beyond Text Following: Repairable Arbitration Reversals in Audio-Language Models

**arXiv ID:** 2606.05161 | [PDF](https://arxiv.org/pdf/2606.05161v1)

**作者:** Yichen Gao `[一作]` (Northeastern University), Daling Wang `[通讯]` (Northeastern University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究音频-文本冲突下的音频语言模型（ALM）如何在音频与文本证据冲突时优先采文本，并通过同音频反事实实验定位音频证据被驳回的机制。

**💡 创新点**

提出的创新点包括：①同音频反事实诊断揭示音频证据被文本仲裁取代的“可修复仲裁逆转”模式；②利用激活补丁将逆转定位到答案位置残差流；③构造基于该诊断的无训练参数解码规则Gated Audio Counterfactual Logit Correction (GACL)，实现对仲裁错误的即时纠正；④证明该方法可迁移到视觉-文本仲裁。

**🔧 技术方法**

技术方法主要包括：同音频反事实两分支（joint vs. same-audio），激活补丁（activation patching）定位内在因果方向，Spearman相关度检验补丁方向与输出分数差的对齐，基于分数差的有界插值加门控的解码规则。

**📊 数据集**

实验使用五个公开权重的音频语言模型（Qwen2-Audio-7B, Qwen2.5-Omni-7B, Voxtral-Small-24B, Qwen3-Omni-30B, Kimi-Audio-7B）以及四个音频-文本冲突基准（AQA, VSC, SER, ALME）。

**📈 对比分析**

在冲突恢复与忠实度损失折中上，GACL在5pp忠实度预算下在所有模型-任务-预算组合中赢得了39/40场景，平均比最佳对比方法提升17.8 nAUC；在视觉-文本仲裁任务上，直接迁移无调参可提升+40.5pp。

**⚠️ 局限性**

局限性包括：需额外一次前向推理产生额外延迟；仅对同音频反事实设定有效，未覆盖噪声文本或更自然的冲突场景；方法仅修复已编码的音频证据，无法弥补模型本身缺失的音频感知能力。

---

## 572. X4Val: Learning Neural Surrogates for Variance-Reduced Policy Evaluation

**arXiv ID:** 2606.05159 | [PDF](https://arxiv.org/pdf/2606.05159v1)

**作者:** Rachel Luo `[一作]` (NVIDIA Research), Marco Pavone `[通讯]` (NVIDIA Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出 X4Val 框架，利用非配对、多域辅助数据学习可迁移的神经替代器，并将其作为控制变量，实现在真实世界场景下低方差的性能估计。

**💡 创新点**

创新点在于：① 在非配对、多域环境中将神经替代器作为控制变量，突破传统需要样本逐一配对的限制；② 采用跨折交叉拟合提升低标签场景下的样本效率；③ 兼容并扩展了经典控制变量方法，形成通用框架。

**🔧 技术方法**

使用技术包括共享嵌入空间、迁移学习/元学习训练神经替代器、控制变量估计、交叉拟合（cross‑fitting）与置信区间构造。

**📊 数据集**

实验数据集包含 NVIDIA PhysicalAI‑Autonomous‑Vehicles（美、德两地区）、Sim2Val/Simulated 场景、ManiSkill 仿真与 Franka Panda 真实机器人等。

**📈 对比分析**

与 Monte‑Carlo、传统控制变量、CPPI 等基线比较，在 AV 地理迁移、迭代策略、跨平台评估等场景下均实现 15%–38% 的方差下降，置信区间明显更紧。

**⚠️ 局限性**

主要局限是：最终估计必须使用目标域分布的样本；辅助多域数据不能直接替代目标期望；神经替代器的训练和跨域泛化仍需要大量计算资源与精细调优。

---

## 573. Reinforcement Learning from Rich Feedback with Distributional DAgger

**arXiv ID:** 2606.05152 | [PDF](https://arxiv.org/pdf/2606.05152v1)

**作者:** Rishabh Agrawal `[一作]` (University of Southern California), Paria Rashidinejad `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DistIL 算法，用前向交叉熵目标在分布式 DAgger 框架下进行自监督强化学习，以利用丰富反馈实现科学推理、代码生成和数学推理任务的学习。

**💡 创新点**

创新点包括：① 证明基于 f‑divergence 的自监督蒸馏无法保证单调策略改进；② 引入前向交叉熵目标实现奖励对齐；③ 设计全序列未来信用分配，解决局部梯度忽略早期决策的缺陷；④ 在理论上证明 Monotonic 改进、子线性 regret 以及对成功概率的下界优化。

**🔧 技术方法**

核心技术：分布式 DAgger、前向交叉熵蒸馏、自然梯度优化、PPO 风格信赖区间更新、未来信用分配、教师加权交叉熵估计。

**📊 数据集**

主要数据集：SciKnowEval L3（科学推理）、LiveCodeBench (LCBv6)（代码执行反馈）以及 OmniMath/AIME24/25/HMMT25/AMC23/Minerva（难度高的数学推理）。

**📈 对比分析**

与 SDPO、GRPO、OPSD、SFT 等基线比较，DistIL 在科学推理 Best@16/Maj@16、代码执行 Accuracy/Score@16 以及数学推理 Avg@16 上均表现最优，幅度在 5–9 点间，且训练曲线更平稳、收敛更快。

**⚠️ 局限性**

局限性：仍需手工设置教师采样数量（如 Top‑100 方案）；在极高维稀疏奖励环境下，未来信用分配可能受序列长度影响；对教师分布的近似假设在非常不确定的教师策略下可能导致学习不稳定。

---

## 574. A-Live: Passive Liveness Detection via Neuromuscular Micro-Motion Signatures on Commodity Sensors

**arXiv ID:** 2606.05126 | [PDF](https://arxiv.org/pdf/2606.05126v1)

**作者:** Mohammed Gharib `[一作]` (Aerendir Mobile Inc.), Martin Zizi `[通讯]` (Aerendir Mobile Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并评估了一种基于移动设备IMU传感器的被动生存检测框架A‑Live，利用神经肌肉微动作信号来识别真人用户。

**💡 创新点**

创新点包括：①将惯性测量单元捕获的细微神经肌肉微运动视为生存信号，突破传统视觉或专用硬件方法；②提出轻量级时间/频域特征提取与梯度提升树分类器，支持实时设备端部署；③设计可控机械微运动攻击平台，用于评估对抗性。

**🔧 技术方法**

使用技术包括：IMU信号采集、时间域与频域特征（样本熵、谱熵、分形复杂度）、PCA降维、梯度提升树分类；实现低功耗、低延迟；以及机器人微运动攻击平台进行对抗测试。

**📊 数据集**

使用的数据来源：Android（61型号）与iOS（40型号）设备的真实人机交互数据；AWS Device Farm、SmartBear BitBar、SauceLabs等商用设备农场生成的自动化攻击样本；以及机器人平台生成的物理攻击样本；未使用公开数据集，全部采用现场收集。

**📈 对比分析**

与挑战型、外观型、传统生理型等方法在系统层面进行对比，评估指标为FAR、FRR以及物理攻击成功率。A‑Live在大规模设备农场、真实用户和物理攻击场景中取得FAR<0.02%、FRR<0.5%，物理攻击成功率为0%，显示出极高的精度与可扩展性。

**⚠️ 局限性**

局限性：仅适用于配备IMU的移动/可穿戴设备，桌面或无传感器环境不适用；对极低运动或高频手势时可能产生误判；受设备传感器精度与时间同步影响；未来需研究跨设备可信证明和加密机制。

---

## 575. Geometry Gaussians: Decoupling Appearance and Geometry in Gaussian Splatting

**arXiv ID:** 2606.05124 | [PDF](https://arxiv.org/pdf/2606.05124v1)

**作者:** Hongyu Zhou `[一作]` (University of Bonn), Zorah Lähner `[通讯]` (University of Bonn)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出在 Gaussian Splatting 中为每个 splat 添加单一的几何透明度参数，实现外观与几何的解耦，从而提升渲染与几何重建质量。

**💡 创新点**

创新点在于通过引入几何透明度参数以及透明度检测引导的监督，解决传统单一透明度导致的颜色与几何信息冲突问题，并结合 Vision Foundation Models 的几何先验实现更鲁棒的训练。

**🔧 技术方法**

使用的技术包括 3D Gaussian Splatting、几何透明度参数、基于 Vision Foundation Models 的深度/法线先验、透明度检测（SAM）、学习周期的深度偏移、透明度感知多视角立体匹配损失等。

**📊 数据集**

实验数据集涵盖 TransLab（透明对象）、NeRF Synthetic（反射材质）、Mip-NeRF360（室内外渲染）、DTU（日常物体几何）。

**📈 对比分析**

与 2DGS、PGSR、GOF、CarGS、TSGS 等方法对比，在 TransLab 上实现最优的 Chamfer 距离与 F1 分数，并在渲染指标上匹配或优于现有 SOTA；在其它数据集上保持或略优性能，同时训练速度提升至 TSGS 的 1/3。

**⚠️ 局限性**

局限性包括对透明物体之外场景提升有限、对外部几何先验的依赖（可能误差导致偏差）、需要额外的分割与模型集成，且在极端光照或复杂材质下仍存在一定误差。

---

## 576. GeM-NR: Geometry-Aware Multi-View Editing for Nonrigid Scene Changes

**arXiv ID:** 2606.05142 | [PDF](https://arxiv.org/pdf/2606.05142v1)

**作者:** Josef Bengtson `[一作]` (Chalmers University of Technology), Fredrik Kahl `[通讯]` (Chalmers University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种快速、无训练的多视角一致编辑方法GeM-NR，可处理几何和外观大幅改变的编辑。

**💡 创新点**

核心创新在于：①将编辑后的场景与原场景视为动态场景，使用Depth Anything 3 进行联合深度估计实现全局对齐；②通过深度投影得到编辑视图的稀疏点云并作为warp指导；③将warp结果与原始视图一起输入多参考图像编辑模型（如FLUX.2），在文本+视觉条件下完成最终编辑；④不需要场景级优化，运行时间仅数秒。

**🔧 技术方法**

主要技术包括：深度估计与动态场景重建（Depth Anything 3）、相机姿态估计与优化（RoMa、COLMAP）、多参考图像生成模型（FLUX.2、Qwen）、基于warp的条件编辑、3D Gaussian Splatting（AnySplat）用于生成编辑后3D表示。

**📊 数据集**

使用SPIn-NeRF、IN2N、Mip-NeRF360、BlendedMVS四大数据集进行评估，测试了38个编辑提示（非刚性编辑、物体添加/删除、外观变化）和图像对编辑任务。

**📈 对比分析**

与Omni-3DEdit和Edicho比较，评估指标包括MEt3R、TA、EC、mAA、PSNR、SSIM、LPIPS。GeM-NR在多视角一致性、编辑质量和文本对齐方面均优于两者，且单图编辑时间约3.4秒，明显快于Omni-3DEdit（>8s）并保持相似或更高的质量。

**⚠️ 局限性**

局限性：仅以单个anchor编辑为条件，无法充分利用多视角信息，导致在极端视角变化或全局姿态改变的场景中一致性下降；对深度估计和相机姿态精度高度依赖，若输入视角差异大或深度不准会影响最终结果。

---

## 577. BBOmix: A Tabular Benchmark for Hyperparameter Optimization of Unsupervised Biological Representation Learning

**arXiv ID:** 2606.05139 | [PDF](https://arxiv.org/pdf/2606.05139v1)

**作者:** Luca Thale-Bombien `[一作]` (Leipzig University), Aaron Klein `[通讯]` (Leipzig University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了 BBOmix，首个针对多组学数据的无监督自编码器（AE）超参数优化（HPO）大规模表格基准，系统评估了不同AE架构、下游任务与多种HPO方法的表现。

**💡 创新点**

创新点在于：①收集 105,000 次完整训练记录，形成 35 个可查询的黑盒；②量化重建损失与下游任务性能的相关性，揭示代理指标的有效性与局限；③首次在生物组学无监督学习场景下评估单/多保真度以及迁移学习 HPO 方法，提供基准与洞见。

**🔧 技术方法**

使用的技术包括：多种 AE 架构（Vanillix、Varix、Disentanglix、Ontix）；随机搜索、TPE、BORE、REA、CQR、ASHA、ASHABORE、ASHACQR、BOHB 等单/多保真度优化；BoundingBox、ZeroShot、Quantile Transfer 等迁移学习策略；Spearman 相关、随机森林特征重要性等分析方法。

**📊 数据集**

使用了 TCGA（多组学 3 种模态）和 SCHC（单细胞 RNA、ATAC）两大真实多组学数据集，涵盖 7 种模态，共 35 个黑盒任务。

**📈 对比分析**

对比方法采用了单保真度（RS、TPE、BORE、REA、CQR）、多保真度（ASHA、ASHABORE、ASHACQR、BOHB）与迁移学习（BoundingBox、ZeroShot、Quantile Transfer）三类算法；在 72000 秒预算下多次复现实验，评估归一化后悔值与平均排名。结果显示：CQR 在单保真度中表现最佳；ASHA 系列在多保真度中取得最优的时间-性能折衷；迁移学习方法在冷启动阶段领先，BoundingBox 与 Quantile Transfer 最终与 CQR 相近。

**⚠️ 局限性**

局限性包括：①仅覆盖四种简单 AE 变体，未探讨卷积、图或跨模态网络；②重建损失虽大多是下游代理，但在 TCGA‑DNA 任务中失效，需更具领域知识的代理指标；③迁移学习实验仅涉及跨架构转移，未评估跨模态或跨数据集的可迁移性；④基准仅涵盖两大数据集，需进一步扩展更多组学与临床任务。

---

## 578. Self-Evaluation Is Already There: Eliciting Latent Judge Calibration in Base LLMs with Minimal Data

**arXiv ID:** 2606.05122 | [PDF](https://arxiv.org/pdf/2606.05122v1)

**作者:** XiuYu Zhang `[一作]` (National University Of Singapore), Zhenkai Liang `[通讯]` (National University Of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于自评激励的迭代 RL+distillation 方法（SEE），通过短周期高效激活大模型原有的自评能力。

**💡 创新点**

创新点在于将自评视为激活问题，而非新能力的学习，并利用两阶段循环（RL 与遮蔽判决蒸馏）在仅160个样本内显著提升模型的自评准确性与校准。

**🔧 技术方法**

使用了强化学习（GRPO）配合 Brier‑score 校准项的自评奖励，并在第二阶段采用遮蔽训练（masked distillation）仅针对自评 tokens 进行监督。

**📊 数据集**

采用了 HelpSteer2、LC AlpacaEval 2.0、Arena‑Hard‑Auto v2.0 与 WildBench v2 等开放式评测数据集，并用 GPT‑5.4 作为外部判决者。

**📈 对比分析**

与单阶段 Adapted RLCR 对比，SEE 在相同任务下仅使用约 31 倍更少的数据，校准分数提升约 8% 以上，质量也保持或略有提升，且在三大基准上的性能均优于基线。

**⚠️ 局限性**

局限性包括仅在单一基模型和 LLM 判决者上验证，缺乏人类评估；此外方法对不同模型规模和更大数据量的泛化仍待探究。

---

## 579. Controllable Dynamic 3D Shape Generation via 3D Trajectories and Text

**arXiv ID:** 2606.05162 | [PDF](https://arxiv.org/pdf/2606.05162v1)

**作者:** Jaeyeong Kim `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了T2Mo框架，实现基于3D轨迹和文本提示的可控动态3D形状生成；

**💡 创新点**

核心创新是形状感知轨迹嵌入（shape‑grounded trajectory embedding），将任意稀疏或密集轨迹映射到覆盖整个网格的固定形状感知令牌；

**🔧 技术方法**

技术包括变分自编码器压缩形状与轨迹、基于DiT的跨注意力生成网络、轨迹与文本的条件交叉注意力、以及Rectified Flow的去噪训练与ODE推理；

**📊 数据集**

实验使用多种通用3D网格数据集（如 ShapeNet、M3D等），并采集对应的轨迹与文本提示；

**📈 对比分析**

与纯文本生成和视频→网格的级联基线比较，采用VBench、轨迹对齐度、运动幅度等量化指标；实验表明T2Mo在保持高运动质量的同时，显著提高了轨迹跟随度和文本语义表达能力；

**⚠️ 局限性**

局限性包括对极度稀疏或分布不均的轨迹处理仍可能出现局部不一致，以及对文本与轨迹语义冲突时的鲁棒性不足。

---

## 580. Streaming Communication in Multi-Agent Reasoning

**arXiv ID:** 2606.05158 | [PDF](https://arxiv.org/pdf/2606.05158v1)

**作者:** Zhen Yang `[一作]` (Hong Kong University of Science and Technology), Ying-Cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于步骤级流式传输的多智能体推理系统StreamMA，以取代传统的生成后传输（Serial）模式。

**💡 创新点**

创新点在于将通信单位从完整响应降至单一步骤，既降低了延迟又提升了推理效果，并给出了三条闭式理论（效果排序、速度上限、成本比）以及发现了“步骤级扩展定律”。

**🔧 技术方法**

使用了多智能体推理框架，基于链式、树式和图式拓扑，采用LLM的推理步骤生成与KV缓存重用，理论推导结合实验验证。

**📊 数据集**

在八个涵盖数学、科学和代码的基准上进行评估，包括AIME 2025/26、HMMT 2026、GPQA-Diamond、HLE、LiveCodeBench等，并使用Claude Opus 4.6和GPT-5.4两款前沿模型。

**📈 对比分析**

通过与单代理（Single）和串行（Serial）两种基线在相同提示和解码设置下对比，StreamMA在平均效果上提升约7.3个百分点（最高22.4个百分点），同时实现显著的速度提升（最高约27×）且成本略有下降。

**⚠️ 局限性**

局限在于只能适用于可拆分为步骤的推理任务；且当步骤正确性分布不满足“头部可靠、尾部不可靠”时，Stream并非最优，需要根据理论判定选择协议。

---

## 581. Multi-Column RBF Neural Network Using Adaptive and Non-Adaptive Particle Swarm Optimization

**arXiv ID:** 2606.05150 | [PDF](https://arxiv.org/pdf/2606.05150v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 582. HORIZON: Recoverability-Governed Curriculum for Physical-Domain Scaling

**arXiv ID:** 2606.05143 | [PDF](https://arxiv.org/pdf/2606.05143v1)

**作者:** Chenhao Bai `[一作]` (Zhejiang University), Chunhua Shen `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文设计了一套名为 HORIZON 的可恢复前沿课程，通过在模拟中逐步扩展机器人物理域，并利用检查点回滚机制，最终实现了零射转移到真实硬件上的四足机器人控制。

**💡 创新点**

创新点在于：①将 recoverability（可恢复性）视为在线 RL 中物理域扩展的核心约束；②提出检查点回滚与边界细化的可恢复前沿机制，将固定随机化转变为可学习的持续增长过程；③揭示物理域扩展的非单调性，并通过核心域组合（actuation、mass、disturbance、COM）实现高效、稳健的跨域泛化。

**🔧 技术方法**

使用了：PPO 与 transformer 策略、检查点回滚与前沿管理器、可恢复门控（feasibility gate + checkpoint-comparison gate）、多域随机化设计、实验对比、硬件部署验证。

**📊 数据集**

数据集主要是自定义的多机器人模拟环境，包含七个物理域（动力学、质量、扰动、初始状态、接触、惯性、重心）在不同范围内随机化，实验还包括真实硬件上的多种形态与负载变异。

**📈 对比分析**

通过与全域随机化、无回滚版本、多专家蒸馏、ADR、GRAM 等方法在固定 OOD 测试集（七域组合）进行对比，核心域课程在 71.1% 的七域 OOD 成功率上显著优于全域 28.1%，并在单域上保持竞争力；在硬件实验中实现了零射转移，表现优于基线。

**⚠️ 局限性**

局限性：仅针对四足行走物理域，域组手工设定，未覆盖操纵任务；未结合视觉感知域，未来需自动化域发现与感知域协同。

---

## 583. Activation-Based Active Learning for In-Context Learning: Challenges and Insights

**arXiv ID:** 2606.05134 | [PDF](https://arxiv.org/pdf/2606.05134v1)

**作者:** Yaseen M. Osman `[一作]`, Stuart E. Middleton `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了使用MLP激活值进行深度主动学习，以评估其在LLM上下文学习中的有效性。

**💡 创新点**

提出并系统评估了迄今最全面的MLP激活量化主动学习方法，并探讨不同注意力掩码对采样的影响，指出激活模式与任务性能几乎无关。

**🔧 技术方法**

计算MLP层激活的四阶矩（均值、方差、偏度、峰度）和最大激活，使用Spearman相关系数评估与任务性能的关系，并在Llama‑3.2‑3B与Qwen2.5‑3B两款3B规模模型上实验。

**📊 数据集**

使用了BoolQ、ARC‑Challenge、OpenBookQA、GSM8K四个多样化的数据集，涵盖分类与生成任务。

**📈 对比分析**

通过将每个候选示例的激活得分与1‑shot任务性能（精确匹配或准确率）进行Spearman相关分析，发现最高相关系数仅约0.33，表明激活得分无法可靠预测性能。

**⚠️ 局限性**

实验仅限于3B规模的非指令微调模型，未涵盖更大规模或指令调优模型，且仅在1‑shot设置下进行，限制了结论的普适性。

---

## 584. Towards Efficient and Evidence-grounded Mobility Prediction with LLM-Driven Agent

**arXiv ID:** 2606.05130 | [PDF](https://arxiv.org/pdf/2606.05130v1)

**作者:** Linyao Chen `[一作]` (University of Tokyo), Hiroki Kobayashi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种训练无关的LLM驱动代理框架AgentMob，将个体下一个位置预测建模为自适应证据控制决策。

**💡 创新点**

采用快速路径与自适应工具调用的组合，利用多源证据（最近轨迹、历史行为、停留‑移动概率、地理距离）并通过LLM控制器逐步聚合，提供可审计的决策轨迹。

**🔧 技术方法**

使用大语言模型作为控制器，调用专用移动分析工具（Mobility Context Retriever、Geographical Info Retriever、Stay‑Move Estimator、Historical Behavior Retriever），并在多轮推理后给出排名。

**📊 数据集**

在东京移动电话GPS数据集BW、匿名500m网格的YJMob100K以及上海ISP基站轨迹数据集上进行实验。

**📈 对比分析**

与监督序列模型（DeepMove、Transformer）和多种LLM基准（AgentMove、LLM‑Mob、TrajLLM、LLM Urban Residents）对比，AgentMob在三数据集上均为训练无关LLM方法中性能最佳；例如在BW上Acc@1 71.42%，YJMob100K 33.14%，上海ISP 33.50%。

**⚠️ 局限性**

受限于LLM的指令遵循与多工具协同能力，工具设计与调用策略手工设定，且未覆盖社交签到等主动观察场景。

---

## 585. Failed Reasoning Traces Tell You What Is Fixable (But Not by Reading Them)

**arXiv ID:** 2606.05145 | [PDF](https://arxiv.org/pdf/2606.05145v1)

**作者:** Nizar Islah `[一作]` (Mila - Quebec AI Institute), Eilif B. Muller `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将失败推理轨迹视为诊断对象，设计了三种基于轨迹概率分布的特征，从而在不额外训练或访问权重的情况下实现失败可恢复性分区、后训练方法审计和无训练的路由决策。

**💡 创新点**

创新点在于：①将失败轨迹的分布特征与操作符类别几何结构对应，形成可解释且可部署的诊断特征；②通过这些特征实现对后训练方式（SFT vs RL）的无训练审计；③在“可驱动难题”子集上实现仅消耗1.4倍重试计算的路由策略，提升约12.2个百分点。

**🔧 技术方法**

使用的技术包括：概率混合与对数几率插值（m‑geodesic / e‑geodesic）、局部 Fisher 信息聚合、k‑means 聚类、基于特征的阈值路由规则，以及对已失败推理轨迹的 top‑k log‑prob 统计。

**📊 数据集**

主要使用的数据集包括：CruxEval、GSM8K、GPQA（推理任务），以及不同规模与后训练方式的 Qwen3 系列模型、R1‑Distill‑Qwen‑Math‑1.5B 与 Phi‑4‑mini‑reasoning 等跨族模型。

**📈 对比分析**

与基线（重试、单一插值、训练后的 RF 预测器、完整的诊断算子评估）相比，本文路由策略在全量失败样本上与重试相当，在可驱动难题子集上提升约12.2个百分点；相比完整诊断算子，其计算成本低约 1/47，且在 8/9 交叉族实验中保持相同性能。

**⚠️ 局限性**

局限性包括：只能在具有局部几何结构的“可驱动难题”任务上使用；对高熵全局任务（如开放式生成、对话）不适用；当前实现中对分布式变形（Distributed Deformation）偏好稀疏插值而非理论上预期的密集插值；对大于7B参数或高度 RLHF 对齐模型的表现未验证。

---

## 586. Preserving Data Privacy in Learning Causal Structure with Fully Homomorphic Encryption

**arXiv ID:** 2606.05129 | [PDF](https://arxiv.org/pdf/2606.05129v1)

**作者:** Jian Yang `[一作]` (Hong Kong University of Science and Technology), Xiaofang Zhou `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在分布式环境下，利用全同态加密实现了隐私保护的因果结构学习。

**💡 创新点**

通过电路简化、牛顿-拉夫逊反函数与泰勒展开近似除法与对数，并结合SIMD批处理，显著提升FHE下的运算效率。

**🔧 技术方法**

使用CKKS同态加密、Newton‑Raphson递归、Taylor展开、SIMD批处理以及Microsoft SEAL等技术。

**📊 数据集**

使用了六个公开基准因果结构数据集：child、insurance、water、alarm、hepar2、win95pts（各5000样本）。

**📈 对比分析**

与明文PC‑stable实现对比，CI测试一致率约85%，结构相似度高；在加密环境下完成时间约6–30分钟，通信成本极低，主要开销在计算。

**⚠️ 局限性**

主要限制为计算成本高（占总耗时84%），对状态数和分辨率敏感，且对高维大状态数数据仍有显著延迟。

---

## 587. Deep Embedded Multiplicative DMD for Algebra-Preserving Koopman Learning

**arXiv ID:** 2606.05131 | [PDF](https://arxiv.org/pdf/2606.05131v1)

**作者:** Kelan Gray `[一作]` (Imperial College London), Matthew J. Colbrook `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Deep Embedded Multiplicative Dynamic Mode Decomposition（DeepMDMD），在潜在空间学习分区并严格保持Koopman算子的乘法规则；

**💡 创新点**

创新点在于将传统MDMD的几何分区迁移到深度学习的潜在空间，并通过交替优化同时逼近Koopman闭合，从而获得既灵活又结构保持的谱近似；

**🔧 技术方法**

使用深度自编码器预训练、Student t核软聚类、MDMD算子更新以及梯度下降实现联合优化；

**📊 数据集**

实验数据集包括二维非线性摆、Lorenz‑96（d=9）、高维流体流动（圆柱尾流158,624维、20,000雷诺数壁面驱动腔体4,225维）等；

**📈 对比分析**

与传统MDMD、EDMD（同字典）对比，DeepMDMD在字典尺寸较小的情况下实现更精确的守恒量、减少谱污染、提高连续谱覆盖，并在噪声环境下显著降低预测误差；

**⚠️ 局限性**

主要限制是训练成本高、需要预训练和超参数选择、仅适用于可逆且保持测度的动力学，且对潜在维度的自动选择尚未解决。

---

## 588. GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors

**arXiv ID:** 2606.05160 | [PDF](https://arxiv.org/pdf/2606.05160v1)

**作者:** Tianyi Xie `[一作]` (NVIDIA), Ye Yuan `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套全数字化管线，从3D资产与视频先验生成高质量的全身运动与人-物交互数据，并将这些数据用于训练能够在真实Unitree G1机器人上实现自主站立爬楼梯和抓取物体的视觉控制策略。

**💡 创新点**

创新点在于：①利用视频基础模型（VFM）在已知3D场景条件下生成交互视频，从而避免了传统视频重建的多余不确定性；②结合已知几何、尺度和摄像机参数的“特权”配置，进行交互感知的4D重建，显著提升姿态与物体轨迹的可执行性；③设计两种任务通用跟踪器（对象感知的潜在适配器与基于高度图的场景感知适配器），实现跨任务的参数共享与自适应控制；④将大规模生成数据（2万+序列）直接投射到实际机器人上，完成了端到端的sim-to-real迁移。

**🔧 技术方法**

使用的关键技术包括：视频基础模型（如Kling等）生成交互视频；多模态姿态与手部估计（GENMO、WiLoR、SMPL-X、MANO）；对象跟踪（FoundationPose）和深度对齐（MoGe-2、SAM2）；交互感知的联合优化（关键点、投影、深度、接触、正则化）；任务通用跟踪器基于预训练的全身控制器SONIC与PPO强化学习；视觉策略通过域随机化与头摄像头RGB输入进行训练；部署时使用Luxonis OAK‑D W摄像头和NVIDIA RTX 5090推理。

**📊 数据集**

数据集：从Robocasa、ComAsset、OMOMO、Hunyuan3D等公开3D资产库采集约1,000种物体，并使用Infinigen生成约1,000种程序化地形；生成的2万+全身交互序列覆盖抓取、全身操纵、坐姿与地形穿越四大任务。

**📈 对比分析**

与现有基于人类视频或运动捕捉的方法（CHOIS、HOIDiff、DAViD、HDMI、ResMimic）进行对比；生成的4D交互序列在几何质量、感知真实度、运动平滑度和物理可执行性上均优于基线；在机器人跟踪与执行任务中，采用任务通用跟踪器实现的成功率最高（抓取81.4%，物体位置误差0.135m，MPJPE-L 41.8mm），并在真实机器人上达成84%（抓取）和90%（爬楼梯）的成功率，明显优于基线。

**⚠️ 局限性**

局限性：需预先提供完整的3D物体资产与可模拟场景；VFM生成的交互可能因遮挡、快速运动或外观不一致导致重建误差，需筛选失败序列；当任务族发生大幅变化时仍需重新训练或微调跟踪器；在极端复杂环境或非标姿态下的可执行性尚未充分验证。

---

## 589. Temporal Cliques Admit Linear Spanners

**arXiv ID:** 2606.05156 | [PDF](https://arxiv.org/pdf/2606.05156v1)

**作者:** Julia Baligacs `[一作]` `[通讯]` (University of Oxford), Julia Baligacs (University of Oxford)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

证明任意时间完全图存在线性大小（≤7n）的时序生成子图，并给出多项式时间构造算法。

**💡 创新点**

首次将时序连通图的稀疏生成子图问题从已知的O(n log n)提升到线性规模，且提出了“扩展星”结构与分治递归的新方法。

**🔧 技术方法**

利用分区可拆除性（dismountability）、扩展星（extended star）与极限匹配（extremally matched）技术，结合对偶双集的递归与shifted matching图的等价性证明。

**📊 数据集**

该工作为理论研究，无实验数据集。

**📈 对比分析**

相较于以往的O(n log n)上界，本论文给出7n的上界，虽然尚未达到已知下界2n-4，但已显著改善复杂度。

**⚠️ 局限性**

常数因子尚非最优；无法控制生成子图的时间路径长度（stretch），且对实际数据未进行实验验证。

---

## 590. An Open-Source Two-Stage Computer Vision Pipeline for Fine-Grained Vehicle Classification using Vision Transformers

**arXiv ID:** 2606.05149 | [PDF](https://arxiv.org/pdf/2606.05149v1)

**作者:** Gandhimathi Padmanaban `[一作]` (University of Michigan-Dearborn), Fred Feng `[通讯]` (University of Michigan-Dearborn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种两阶段计算机视觉管线，利用预训练的RT-DETR检测车辆并使用细粒度ViT对车辆进行六类（乘用车、SUV、皮卡、面包车、大型货车、商用卡车）身体类型分类，从自然路面视频中实现自行车旁行驶车辆类型的自动标注。

**💡 创新点**

创新点包括：1）将无标注框的RT-DETR与细粒度ViT结合，避免了对检测框的人工标注；2）引入置信度阈值的自我放弃机制，避免低置信度误判产生的“沉默错误”；3）首次在自然摄像头视频上进行跨域（分布内与外）评估，并将完整模型、权重与代码开源。

**🔧 技术方法**

技术实现基于Transformer：RT-DETR作为目标检测器；Vision Transformer ViT-Base/16作为细粒度分类器；使用焦点损失和加权采样处理类别不平衡；置信度阈值0.60做自我放弃；混合精度训练。

**📊 数据集**

训练与评估使用的主要数据集包括：Stanford Cars（用于前四类），网络爬取的图片（补充大型货车、商用卡车），Ann Arbor N. Division现场摄像头收集的3,805条加速事件（分布内），以及外部开源自行车数据集收集的311条事件（分布外）。

**📈 对比分析**

在分布内测试中整体精度为0.94，所有类别F1均≥0.91；在离域测试中整体精度下降至0.89，四个主要类别（SUV、皮卡、乘用车、面包车）F1≥0.90；唯一显著下降的类别为面包车，F1从0.91降至0.72，主要因自我放弃率大幅上升。

**⚠️ 局限性**

局限性：训练样本严重不平衡，商用卡车和大型货车样本极少，导致其评估不稳；仅处理单帧预测，未利用视频时间信息；训练未划分验证集，可能存在过拟合；系统仅针对固定路侧摄像头视角，泛化到其他视角（如顶视、行车摄像头）尚未验证。

---

