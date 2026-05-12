# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-12 | 今日论文总数: 879

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. The Metacognitive Probe: Five Behavioural Calibration Diagnostics for LLMs

**arXiv ID:** 2605.09844 | [PDF](https://arxiv.org/pdf/2605.09844v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 2. RFAmpDesigner: A Self-Evolving Multi-Agent LLM Framework for Automated Radio Frequency Amplifier Design

**arXiv ID:** 2605.10093 | [PDF](https://arxiv.org/pdf/2605.10093v1)

**作者:** Hang Lu `[一作]` (Zhejiang University), Zhiwei Xu `[通讯]` (Zhejiang University)

**通讯引用:** 9668 | [OpenAlex ID](https://openalex.org/A5013653915)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RFAmpDesigner，一种基于多代理、LLM 驱动的自进化框架，利用资源分配中介把 RF 放大器参数化简为低维资源分配，实现从规格到电路的自动设计。

**💡 创新点**

创新点包括：① 用资源分配中介将高维参数空间降维，便于 LLM 注入域知识；② 三代理两层工作流将繁重仿真与轻量推理分离；③ 通过检索增强生成（RAG）实现自进化知识库；④ 在不直接操作网表的前提下完成 LNA 设计。

**🔧 技术方法**

采用 LLM（GPT‑5.1、Claude‑Sonnet‑4.5、Gemini‑2.5‑Flash、Qwen3 等）+ ReAct 思维–行动循环；资源分配中介工具（搜索、精化、匹配、频带规划、全链评估）；多代理协同（RFAmpManager、RFAmpSearcher、RFAmpRefiner）；检索增强生成；多模型与多工具集成。

**📊 数据集**

使用基于 TSMC N65 PDK 的激励单元尺寸与偏置查表；10 任务 LNA 设计基准（中心频率 10‑50 GHz、带宽 10‑80%、功耗 30 mA、噪声 5 dB、线性 IP1dB 等）；实验中使用 Spectre 进行仿真。

**📈 对比分析**

与传统 GA/BO、工具增强版、AnalogCoder、ADO‑LLM 等基线比较，采用 pass@1 成功率、平均完成时间和 token 消耗评估；在任务 1‑8 中 RFAmpDesigner 取得最高成功率、最短平均时间；在任务 9‑10 中显著优于基线，成功率提升至 80‑100%，耗时大幅降低；不同 LLM 后端表现一致，Claude‑Sonnet‑4.5 取得最佳平衡。

**⚠️ 局限性**

局限性包括：仍需手工实现工具中介和知识库，迁移到其他拓扑或工艺需重写工具；对极端高难度任务仍需大量仿真；小模型或无工具微调的 LLM 在推理质量上不稳定；未覆盖完整布局与版图步骤，后续工作需要进一步扩展。

---

## 3. Explicit Stair Geometry Conditioning for Robust Humanoid Locomotion

**arXiv ID:** 2605.09944 | [PDF](https://arxiv.org/pdf/2605.09944v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 4. When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews

**arXiv ID:** 2605.10171 | [PDF](https://arxiv.org/pdf/2605.10171v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 5. A Game Theoretic Free Energy Analysis of Higher Order Synergy in Attention Heads of Large Language Models

**arXiv ID:** 2605.09515 | [PDF](https://arxiv.org/pdf/2605.09515v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 6. Uncertainty-Aware and Decoder-Aligned Learning for Video Summarization

**arXiv ID:** 2605.09507 | [PDF](https://arxiv.org/pdf/2605.09507v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 7. Adaptive Liquidity in Prediction Markets via Online Learning

**arXiv ID:** 2605.09599 | [PDF](https://arxiv.org/pdf/2605.09599v1)

**作者:** Enrique Nueve `[一作]` (University of Colorado Boulder), Bo Waggoner `[通讯]` (University of Colorado Boulder)

**通讯引用:** 619 | [OpenAlex ID](https://openalex.org/A5076830609)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种自适应预测市场机制，通过可学习的权重组合多种成本函数市场，以动态调整流动性。

**💡 创新点**

创新点在于将流动性选择视为在线学习问题，提出了一种混合结构风险信号，量化价格影响与库存风险之间的权衡。

**🔧 技术方法**

使用了在线学习算法和混合成本函数市场的构建方法。

**📊 数据集**

论文中未具体提及使用的数据集，但通过模拟展示了机制在不同市场条件下的表现。

**📈 对比分析**

通过模拟比较了自适应市场与固定流动性市场的表现，结果表明自适应机制能够根据市场条件动态调整流动性，表现优于固定市场。

**⚠️ 局限性**

限制在于实验主要基于简化的模拟，未来研究需要将信号与经济福利目标联系起来，并探索自适应流动性下的均衡行为。

---

## 8. Nano-U: Efficient Terrain Segmentation for Tiny Robot Navigation

**arXiv ID:** 2605.10210 | [PDF](https://arxiv.org/pdf/2605.10210v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 9. LASAR: Latent Adaptive Semantic Aligned Reasoning for Generative Recommendation

**arXiv ID:** 2605.10207 | [PDF](https://arxiv.org/pdf/2605.10207v1)

**作者:** Yiwen Chen `[一作]` (Beihang University), Zhao Zhang `[通讯]` (Beihang University)

**通讯引用:** 78508 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的生成式推荐框架LASAR，利用隐层反馈循环进行多步隐式推理并自适应控制推理深度，显著提升推荐质量同时减少推理延迟。

**💡 创新点**

创新点包括：①两阶段SFT分离训练，先完成Semantic ID语义锚定再加入隐式推理；②通过显式链式思考（CoT）文本构造的语义锚点并用双向KL对齐，解决表示漂移；③加入Policy Head与REINFORCE实现样本级推理步数自适应；④结合GRPO与终端KL实现质量与效率的联合优化。

**🔧 技术方法**

使用技术包括：隐层反馈循环的连续空间推理、两阶段SFT+RL训练框架、双向KL对齐、Policy Head + REINFORCE、GRPO、终端KL、批量化可变步长处理与GPU并行化。

**📊 数据集**

实验数据集为Amazon的Beauty、Instruments、Sports三大商品评论数据集，分别对应22K/25K/36K用户，10K/10K/18K物品，交互量分别为176K/74K/107K。

**📈 对比分析**

与传统序列模型（SASRec、GRU4Rec）、LLM生成推荐（MiniOneRec、LC-Rec）、隐式推理（ReaRec）和显式链式思考（Explicit CoT_GREAM）进行对比。LASAR在所有指标（NDCG@K、HitRate@K，K=5/10/20）上均优于基线，尤其在最稀疏数据集Sports上提升显著；同时推理延迟仅比MiniOneRec高约7–16%，而显式CoT模式延迟超过20倍。

**⚠️ 局限性**

主要限制在于隐层反馈循环不支持教师强制，导致推理过程需逐步前向，难以完全并行化；此外方法目前仅验证于生成式推荐，未涵盖对话式或跨域场景，未来需探索更高效的循环执行与更广泛的应用场景。

---

## 10. Generating Symmetric Materials using Latent Flow Matching

**arXiv ID:** 2605.10115 | [PDF](https://arxiv.org/pdf/2605.10115v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 11. MVB-Grasp: Minimum-Volume-Box Filtering of Diffusion-based Grasps for Frontal Manipulation

**arXiv ID:** 2605.09672 | [PDF](https://arxiv.org/pdf/2605.09672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 12. DynGhost: Temporally-Modelled Transformer for Dynamic Ghost Imaging with Quantum Detectors

**arXiv ID:** 2605.10185 | [PDF](https://arxiv.org/pdf/2605.10185v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 13. Dual-Path Hyperprior Informed Deep Unfolding Network for Image Compressive Sensing

**arXiv ID:** 2605.09566 | [PDF](https://arxiv.org/pdf/2605.09566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 14. WindINR: Latent-State INR for Fast Local Wind Query and Correction in Complex Terrain

**arXiv ID:** 2605.09511 | [PDF](https://arxiv.org/pdf/2605.09511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 15. Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation

**arXiv ID:** 2605.10118 | [PDF](https://arxiv.org/pdf/2605.10118v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 16. Can We Trust LLMs for Mental Health Screening? Consistency, ASR Robustness, and Evidence Faithfulness

**arXiv ID:** 2605.09634 | [PDF](https://arxiv.org/pdf/2605.09634v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 17. Spectral Transformer Neural Processes

**arXiv ID:** 2605.09498 | [PDF](https://arxiv.org/pdf/2605.09498v1)

**作者:** Xianhe Chen `[一作]` (University of Cambridge), Yingzhen Li `[通讯]` (Imperial College London)

**通讯引用:** 2847 | [OpenAlex ID](https://openalex.org/A5038242041)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Spectral Transformer Neural Processes（STNPs），在Transformer Neural Processes（TNPs）中加入频域信息，显著提升周期性和准周期性数据的建模效果。

**💡 创新点**

创新点在于设计Spectral Aggregator，将上下文集投影到频域并压缩为可采样的谱混合分布，再将采样的谱特征与时域嵌入拼接，从而在不改动注意力机制的前提下为TNP注入自适应谱混合核的先验。

**🔧 技术方法**

核心技术包括频域能谱估计、CNN责任网络进行谱分解、概率压缩成Gaussian Mixture、随机采样谱频率生成特征、以及与原有MLP分支拼接并投射到Transformer编码器。

**📊 数据集**

在四类合成回归（RBF、Matérn、周期性、锯齿形）、七个真实世界时间序列（Electricity、ETTh1、Exchange Rate、National Illness、Traffic、Weather等）以及一个图像完成任务（DTD纹理数据集）上进行评估。

**📈 对比分析**

与NP、CNP、BNP、ANP、CANP、BANP、ConvCNP、SConvCNP、TNP、TETNP等基线相比，STNP在合成回归任务的对数似然上提升约1–3点，在时间序列预测中的MAE/MSE下降约20–80%，在图像完成任务中PSNR/SSIM提升约0.5–1点，整体表现均显著优于所有对比方法。

**⚠️ 局限性**

局限包括需预设频率网格大小和谱混合组件数，且在频率估计对上下文稀疏时仍易受噪声影响；当前仅针对单变量或低维时空数据验证，尚未探索高维图像/视频的扩展。

---

## 18. Mixture of Layers with Hybrid Attention

**arXiv ID:** 2605.09516 | [PDF](https://arxiv.org/pdf/2605.09516v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 19. Concordia: Self-Improving Synthetic Tables for Federated LLMs

**arXiv ID:** 2605.09855 | [PDF](https://arxiv.org/pdf/2605.09855v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 20. A 4.509-Approximation Algorithm for Generalized Min Sum Set Cover

**arXiv ID:** 2605.10031 | [PDF](https://arxiv.org/pdf/2605.10031v1)

**作者:** Amey Bhangale `[一作]` (University of California, Riverside), Yezhou Zhang `[通讯]` (University of California, Riverside)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究并改进了广义最小总和集合覆盖（GMSSC）问题的近似算法，提出了一个4.509的近似比，进一步提升了之前的4.642上限；同时给出最小延迟集合覆盖问题的2-近似算法。

**💡 创新点**

创新点主要在于：1）在已存在的LP基框架内，采用新的核（kernel）变换与α点随机化，改进了对边界概率的分析；2）利用更精细的二项式尾部界（P^k(x)），结合KC（Knapsack Cover）不等式的多层次利用，精确刻画覆盖概率随时间变化的动态性；3）通过一系列凸优化与递归简化，证明最坏情况实际上退化为边缘需求降至1，从而获得更好的比率。

**🔧 技术方法**

核心技术包括：线性规划松弛（含KC约束）、核变换与α点随机化、Bernoulli求和的下尾界推导（Poisson逼近）、凸优化极值分析以及对比值的极值点推导。

**📊 数据集**

该工作为理论算法研究，未使用实际数据集，所有结果均基于数学证明与理论分析。

**📈 对比分析**

与之前的4.642近似比相比，新算法在理论上提高到4.509；在最小延迟集合覆盖问题中，与已有的2.718…近似算法相比，得到与硬件极限相同的2-近似。由于缺乏实验验证，性能提升仅体现在理论近似比上。

**⚠️ 局限性**

局限性包括：1）仍未达到下界4的近似；2）对核参数β的选择较为特殊，通用核方案尚未解决；3）只在理论层面给出改进，缺乏实验评估；4）对更一般的多需求集合覆盖（任意k_e）是否可以进一步改进仍未明朗。

---

## 21. Distilling 3D Spatial Reasoning into a Lightweight Vision-Language Model with CoT

**arXiv ID:** 2605.09719 | [PDF](https://arxiv.org/pdf/2605.09719v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 22. CloudEmu: A Trace-Driven Cloud-Native Emulation Testbed for Vehicle Video Uplink over Cellular Networks

**arXiv ID:** 2605.09910 | [PDF](https://arxiv.org/pdf/2605.09910v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 23. Towards Generative Predictive Display for Vision-Based Teleoperation: A Zero-Shot Benchmark of Off-the-Shelf Video Models

**arXiv ID:** 2605.09670 | [PDF](https://arxiv.org/pdf/2605.09670v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 24. Adaptive DNN Partitioning and Offloading in Heterogeneous Edge-Cloud Continuum

**arXiv ID:** 2605.09623 | [PDF](https://arxiv.org/pdf/2605.09623v1)

**作者:** Akuen Akoi Deng `[一作]` (Stockholm University), Praveen Kumar Donta `[通讯]` (Stockholm University)

**通讯引用:** 1978 | [OpenAlex ID](https://openalex.org/A5079303717)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

提出并实现了一个面向异构边缘-云连续体的自适应深度神经网络划分与迁移框架，并在真实硬件上验证其能耗与时延优势。

**💡 创新点**

首次将离线模型特征尺寸与计算权重表与两点链路探测相结合，动态重评划分点；并引入基于多目标加权评分的候选搜索，使框架在保持时延约束的同时显著降低能耗。

**🔧 技术方法**

离线模型性能分析、链路传输速率建模、分层划分与调度、实时能耗估算与自适应重调度、PyTorch与ZeroMQ通信等。

**📊 数据集**

使用VGG16、AlexNet、MobileNetV2三种CNN模型，并在每个模型上以随机生成的1×3×224×224张量做推理；未使用公开数据集，只关注模型计算与传输。

**📈 对比分析**

将方案与单机和固定划分两种基线在同一三层测试平台上对比；实验显示能耗下降27–36%，时延下降6–23%，验证了自适应划分优于静态方案。

**⚠️ 局限性**

仅针对单一模型集与固定硬件组合测试，未评估更复杂网络或多模型场景；链路模型假设为线性，忽略拥塞和丢包；能耗估算对Pi使用恒定功率模型；未考虑模型压缩、隐私与安全等实际部署约束。

---

## 25. DeformMaster: An Interactive Physics-Neural World Model for Deformable Objects from Videos

**arXiv ID:** 2605.09586 | [PDF](https://arxiv.org/pdf/2605.09586v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 26. Weighted Rules under the Stable Model Semantics

**arXiv ID:** 2605.09519 | [PDF](https://arxiv.org/pdf/2605.09519v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 27. Online Set Learning from Precision and Recall Feedback

**arXiv ID:** 2605.09565 | [PDF](https://arxiv.org/pdf/2605.09565v1)

**作者:** Lee Cohen `[一作]` (Stanford University), Han Shao `[通讯]` (University of Maryland)

**通讯引用:** 59340 | [OpenAlex ID](https://openalex.org/A5038497484)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并研究了在在线设置下，仅通过随机抽取的精度（precision）和召回（recall）反馈来学习未知子集的问题，提出相应的学习算法并给出理论收敛性分析。

**💡 创新点**

首次证明该类问题的可学习性完全由假设类的VC维决定，揭示了精度/召回反馈模型与传统PAC/在线学习模型在算法结构上的根本差异；提出了基于最大似然（MLE）与自适应估计的算法（Realizable：O(d log²T)；Agnostic：O(d^{1/4}T^{3/4})），并指出在无可行假设时仍可通过非自适应输出空集或全集来获得性能。

**🔧 技术方法**

利用VC维与Littlestone维的理论工具，构造回顾一致性约束的回归（recall）一致集，采用最小大小（最大似然）原则；在无监督阶段引入目标集大小估计器EST，结合精度/召回估计实现自适应精度回归；整体采用三阶段在线算法（APRIL）与概率估计与置信界。

**📊 数据集**

本工作为纯理论分析，无具体实验数据集；所有结果均为分布无关的理论上界与下界。

**📈 对比分析**

性能评估以理论收敛率为准：在可实现情形下实现O(d log²T)的次线性回报损失，且下界Ω(d)表明此类回报是可实现的；在不可实现情形下得到O(d^{1/4}T^{3/4})的次线性回报损失，尚未达到传统√(dT)的下界。

**⚠️ 局限性**

主要限制包括：（1）可实现情形下的log²T因子仍未收敛到最优，需进一步研究；（2）不可实现情形下的回报率可能不是最优，仍有改进空间；（3）假设已知可用集合X，实际应用中X可能未知；（4）模型仅考虑随机精度/召回反馈，未覆盖更复杂反馈形式；（5）部分结果仅适用于VC有限的假设类。

---

## 28. Overcoming Catastrophic Forgetting in Visual Continual Learning with Reinforcement Fine-Tuning

**arXiv ID:** 2605.09640 | [PDF](https://arxiv.org/pdf/2605.09640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 29. Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning

**arXiv ID:** 2605.09537 | [PDF](https://arxiv.org/pdf/2605.09537v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 30. Rethinking Constraint Awareness for Efficient State Embedding of Neural Routing Solver

**arXiv ID:** 2605.10122 | [PDF](https://arxiv.org/pdf/2605.10122v1)

**作者:** Canhong Yu `[一作]` (Shenzhen University), Yu Zhou `[通讯]` (Shenzhen University)

**通讯引用:** 7665 | [OpenAlex ID](https://openalex.org/A5016175345)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对HELD结构的神经车辆路径规划求解器，系统分析状态嵌入生成机制，发现传统的PRE方式限制了观察空间，难以满足复杂约束；提出FGE扩展全局观察空间并设计Constraint‑Aware Residual Modulation (CARM)模块提升约束感知，改进状态嵌入。

**💡 创新点**

1）全局观察空间的FGE方法；2）基于FiLM的约束自适应调制CARM，显著提升约束意识并与FGE协同；3）在单/多任务求解器中验证模块通用性。

**🔧 技术方法**

Transformer‑based HELD架构、Attention机制、FiLM调制、残差连接、强化学习/自回归解码、实验中多任务训练。

**📊 数据集**

公开的16种VRP变体（包含容量、开放路线、返航、时限、时间窗等约束），每类1000个实例，规模从100到1000节点；单任务训练规模固定100，扩展到200/500/1000节点进行评测。

**📈 对比分析**

与原始PRE、FGE以及多种基线模型（POMO、ReLD-STL、MTPOMO、MVMoE、ReLD-MTL、RF、CaDA）在最优性差距Gap、最佳解比例等指标对比；结果显示CARM在所有任务上平均Gap降低≈30%~40%，在大规模实例和未见变体上表现尤为突出。

**⚠️ 局限性**

仅改进网络结构，未在训练范式上做深入探讨；对约束感知的训练策略、起始节点的约束驱动选择等方向仍待研究。

---

## 31. RAwR: Role-Aware Rewiring via Approximate Equitable Partition

**arXiv ID:** 2605.09457 | [PDF](https://arxiv.org/pdf/2605.09457v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 32. FLARE: Full-Modality Long-Video Audiovisual Retrieval Benchmark with User-Simulated Queries

**arXiv ID:** 2605.10228 | [PDF](https://arxiv.org/pdf/2605.10228v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 33. HYPERPOSE: Hyperbolic Kinematic Phase-Space Attention for 3D Human Pose Estimation

**arXiv ID:** 2605.10100 | [PDF](https://arxiv.org/pdf/2605.10100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 34. Med-StepBench: A Hierarchical Reasoning Framework for Evaluating Hallucinations in Medical Vision-Language Models

**arXiv ID:** 2605.10002 | [PDF](https://arxiv.org/pdf/2605.10002v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 35. Designing for Collective Access: In Search of a Solution to Accessible Communication in a Mixed-Ability Non-Profit

**arXiv ID:** 2605.10085 | [PDF](https://arxiv.org/pdf/2605.10085v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 36. CLR-voyance: Reinforcing Open-Ended Reasoning for Inpatient Clinical Decision Support with Outcome-Aware Rubrics

**arXiv ID:** 2605.09584 | [PDF](https://arxiv.org/pdf/2605.09584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 37. Towards LLM-Based Analysis of Virtualization-Obfuscated Code through Automated Data Generation

**arXiv ID:** 2605.09961 | [PDF](https://arxiv.org/pdf/2605.09961v1)

**作者:** Sangjun An `[一作]` (Chungnam National University), Eun-Sun Cho `[通讯]` (Chungnam National University)

**通讯引用:** 444 | [OpenAlex ID](https://openalex.org/A5021428048)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于LLM的框架，用于自动识别和可视化虚拟化混淆二进制的执行结构。

**💡 创新点**

创新点在于：1）利用IR级别自动化结构标签生成，实现大规模无人工标注的数据集；2）将BERT模型用于汇编指令的上下文学习，实现对虚拟机关键组件（调度器、处理器等）的高精度分类；3）通过CFG可视化将识别结果直观展示，降低逆向工程难度。

**🔧 技术方法**

技术手段包括：静态控制流图分析、IR级别结构标记、BERT基础语言模型（多任务学习）以及tokenizer比较（BertTokenizer vs Palmtree）。

**📊 数据集**

使用了来自Tigress虚拟化器生成的约24,010条样本（涵盖Switch、Direct、Indirect三种调度模式），经自动标记后扩充至约126,000条训练样本，包含不同优化级别（-O0、-O1、-O2）的数据。

**📈 对比分析**

实验中将BERT模型与Palmtree tokenizer以及无预训练模型对比；BertTokenizer主标识精度91.7%（Palmtree 83.8%），子标签精度99.8%（Palmtree 99.5%），宏F1达0.998；对各类（HANDLER、VM、VM-START、NON-VM、VM-END、DISPATCH-START）的精确率、召回率均超过99%。

**⚠️ 局限性**

局限性在于：1）自动标签生成依赖于对混淆过程的访问；2）在高优化级别下，VM Start/End 边界可能被编译器合并，导致标记不够清晰；3）模型侧重结构识别，尚未实现对虚拟化逻辑的完整语义恢复与自动去混淆。

---

## 38. TOC-Bench: A Temporal Object Consistency Benchmark for Video Large Language Models

**arXiv ID:** 2605.09904 | [PDF](https://arxiv.org/pdf/2605.09904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 39. Trust Me, Import This: Dependency Steering Attacks via Malicious Agent Skills

**arXiv ID:** 2605.09594 | [PDF](https://arxiv.org/pdf/2605.09594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 40. What Concepts Lie Within? Detecting and Suppressing Risky Content in Diffusion Transformers

**arXiv ID:** 2605.10180 | [PDF](https://arxiv.org/pdf/2605.10180v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 41. Clip-level Uncertainty and Temporal-aware Active Learning for End-to-End Multi-Object Tracking

**arXiv ID:** 2605.09858 | [PDF](https://arxiv.org/pdf/2605.09858v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 42. Quantifying the Utility of User Simulators for Building Collaborative LLM Assistants

**arXiv ID:** 2605.09808 | [PDF](https://arxiv.org/pdf/2605.09808v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 43. FORGE: Fragment-Oriented Ranking and Generation for Context-Aware Molecular Optimization

**arXiv ID:** 2605.10230 | [PDF](https://arxiv.org/pdf/2605.10230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 44. Statistical Analysis for Energy-Efficient Satellite Edge Computing with Latency Guarantees

**arXiv ID:** 2605.10215 | [PDF](https://arxiv.org/pdf/2605.10215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 45. When Adaptation Fails: A Gradient-Based Diagnosis of Collapsed Gating in Vision-Language Prompt Learning

**arXiv ID:** 2605.09549 | [PDF](https://arxiv.org/pdf/2605.09549v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 46. Hystar: Hypernetwork-driven Style-adaptive Retrieval via Dynamic SVD Modulation

**arXiv ID:** 2605.10009 | [PDF](https://arxiv.org/pdf/2605.10009v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 47. On-Policy Distillation with Best-of-N Teacher Rollout Selection

**arXiv ID:** 2605.09725 | [PDF](https://arxiv.org/pdf/2605.09725v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 48. Janus: Compiler-Based Defense Against Transient Execution Attacks Using ARM Hardware Primitives

**arXiv ID:** 2605.10049 | [PDF](https://arxiv.org/pdf/2605.10049v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 49. When Are LLM Inferences Acceptable? User Reactions and Control Preferences for Inferred Personal Information

**arXiv ID:** 2605.10013 | [PDF](https://arxiv.org/pdf/2605.10013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 50. SoccerLens: Grounded Soccer Video Understanding Beyond Accuracy

**arXiv ID:** 2605.09598 | [PDF](https://arxiv.org/pdf/2605.09598v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 51. When Few Steps Are Enough: Training-Free Acceleration of Identity-Preserved Generation

**arXiv ID:** 2605.09460 | [PDF](https://arxiv.org/pdf/2605.09460v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 52. Social Policy of Large Language Models: How GPT, Claude, DeepSeek and Grok Allocate Social Budgets in Spain and Germany

**arXiv ID:** 2605.10234 | [PDF](https://arxiv.org/pdf/2605.10234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 53. TRACE: Distilling Where It Matters via Token-Routed Self On-Policy Alignment

**arXiv ID:** 2605.10194 | [PDF](https://arxiv.org/pdf/2605.10194v1)

**作者:** Jiaxuan Wang `[一作]` (Nanjing University), Lan-Zhe Guo `[通讯]` (Nanjing University)

**通讯引用:** 20585 | [OpenAlex ID](https://openalex.org/A5100355149)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 Token‑routed Self‑OPD 方法，在 RLVR 中仅在标注的关键 span 上使用 Forward/Reverse KL 并逐步衰减，解决全标记 KL 造成的熵激增和推理长度缩短问题。

**💡 创新点**

创新点包括：① 对全标记 KL 的粒度不匹配进行诊断；② 通过自监督的 span 标注将 KL 路由至关键或错误 span，采用 {FKL,RKL,∅} 三种动作并结合覆盖率上限；③ 在短窗口内衰减 KL，保证长期训练仍由 GRPO 主导；④ 发现基模型能力不同会导致最佳路由角度从 FKL→RKL 逆转；⑤ 在线自标注实现无外部监督的收益。

**🔧 技术方法**

技术手段包括：RLVR（GRPO）框架、On‑Policy Self‑Distillation、前向/反向 KL、token‑级 span 标注、KL 权重线性衰减、覆盖率 α=0.25、Qwen3 大模型（8B、1.7B）、vLLM 在线推理、OpenThoughts 训练数据、基准评估与自注意力推理。

**📊 数据集**

使用数据集：OpenThoughts‑114k 计算推理子集进行训练；在四个数学基准（MATH‑500、AIME 24/25、AMC 23）和 OOD 数据集 GPQA‑Diamond 进行评估。

**📈 对比分析**

与 GRPO、SDPO、SRPO、RLSD 等传统 Self‑OPD/OPD 基线在同一超参 grid 下比较，Qwen3‑8B 上平均提升 2.76 pp，GPQA‑Diamond OOD 维持不降，在线自标注实现 +1.90 pp（约占强基模型 69%）。在弱基 Qwen3‑1.7B 上 RKL 关键 span 较优，超越基线平均 1.70 pp。

**⚠️ 局限性**

局限性包括：需依赖高质量 span 标注或自标注，标注精度直接影响性能；KL 权重与覆盖率需手动调优；理论假设（如覆盖率上限、top‑K 截断）在实际中可能不完全成立；模型规模大（8B/1.7B），在资源受限环境下难以复现；方法主要针对数学推理，未在其他长序列任务中验证。

---

## 54. Discriminative Span as a Predictor of Synthetic Data Utility via Classifier Reconstruction

**arXiv ID:** 2605.09697 | [PDF](https://arxiv.org/pdf/2605.09697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 55. PDEAgent-Bench: A Multi-Metric, Multi-Library Benchmark for PDE Solver Generation

**arXiv ID:** 2605.09636 | [PDF](https://arxiv.org/pdf/2605.09636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 56. An Approximation Algorithm for 2-Vertex-Connectivity via Cycle-Restricted 2-Edge-Covers

**arXiv ID:** 2605.10058 | [PDF](https://arxiv.org/pdf/2605.10058v1)

**作者:** Yusuke Kobayashi `[一作]` (Research Institute for Mathematical Sciences, Kyoto University), Takashi Noguchi `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的近似算法，将2-VCSS问题的近似比率从4/3提升到95/72+ε（约1.32），并给出了相应的实现流程；

**💡 创新点**

创新点在于引入了“cycle-restricted 2-edge-cover”概念，并通过将其转化为强规范（strongly canonical）形式，从而在信用分配与转换过程中获得更低的预算，进而实现更优的逼近比；

**🔧 技术方法**

使用了多种技术：①对结构化图进行预处理与分解；②利用PTAS求解T-free 2-matching以得到接近最优的2-edge-cover；③设计信用分配（credit-based）方案与结构化转换（canonicalization）操作；④借鉴并改进了先前的耳分解与块/桥处理；

**📊 数据集**

本工作为理论算法研究，无需实验数据集；

**📈 对比分析**

与之前最好的4/3算法相比，新算法在理论上实现了更小的逼近比（95/72+ε≈1.32），证明了在大多数结构化图上可达成该逼近；

**⚠️ 局限性**

局限性包括：①需要使用ε-近似的PTAS，导致ε项的存在；②算法在包含大量6-周期组件的图上逼近比可能不再优；③若想进一步提升，需找到多项式时间求解T-free 2-matching的精确算法。

---

## 57. Anchor-guided Hypergraph Condensation with Dual-level Discrimination

**arXiv ID:** 2605.10001 | [PDF](https://arxiv.org/pdf/2605.10001v1)

**作者:** Fan Li `[一作]` (University of New South Wales), Wenjie Zhang `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为AHGCDD的新型超图压缩框架，可在保持节点分类性能的同时显著减少超图规模。

**💡 创新点**

创新点在于引入锚点引导的超边生成机制与双层判别损失，统一优化节点特征与结构，解决了以往分离训练导致的结构失配和高计算开销问题。

**🔧 技术方法**

技术手段包括HKPR扩散进行节点初始化、MLP学习锚点关联、可学习的锚点阈值实现稀疏控制、双层判别（类原型对齐+样本级对比）以及动态权重调度。

**📊 数据集**

实验使用六个大规模超图基准数据集：Cora、Pubmed、DBLP-CA、Walmart、Yelp、MAG-PM。

**📈 对比分析**

与传统coreset方法、HG-Cond、HG-Cond-NHL以及扩展的HGCPA比较，AHGCDD在多种压缩比例下实现了最高或接近最高的节点分类准确率，并且速度提升可达144倍，显著优于现有方法。

**⚠️ 局限性**

局限性包括：对超图结构信息依赖较大，无法处理极度稀疏或极大超边；在极端压缩比例下仍会出现精度下降；主要适用于有属性的超图，未针对无属性场景展开研究。

---

## 58. Annotations Mitigate Post-Training Mode Collapse

**arXiv ID:** 2605.09995 | [PDF](https://arxiv.org/pdf/2605.09995v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 59. Closed-Form Gaussian Estimators for Multi-Source Partial Information Decomposition

**arXiv ID:** 2605.09919 | [PDF](https://arxiv.org/pdf/2605.09919v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 60. Active Testing of Large Language Models via Approximate Neyman Allocation

**arXiv ID:** 2605.10075 | [PDF](https://arxiv.org/pdf/2605.10075v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 61. Hyperparameter Transfer for Dense Associative Memories

**arXiv ID:** 2605.10164 | [PDF](https://arxiv.org/pdf/2605.10164v1)

**作者:** Roi Holtzman `[一作]` (University of Oxford), Boris Hanin `[通讯]` (Princeton University)

**通讯引用:** 1183 | [OpenAlex ID](https://openalex.org/A5085872076)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了 Dense Associative Memory（DenseAM）网络的超参数迁移（HP Transfer），并给出了在不同规模（输入维度、隐藏宽度、数据集大小、批大小）下的显式缩放公式，验证了在线性、ReLU^p 以及 Softmax 激活下的训练稳定性和动态一致性。

**💡 创新点**

创新点在于：①提出了针对共享权重和能量函数特性的 DenseAM 量子化的 μP 风格缩放；②发现对激活函数进行层内中心化（centering）是实现 HP Transfer 的关键；③揭示了 Softmax 在 SGD 下的数值不稳定性，并证明 Adam 能保持缩放一致性；④在比例缩放和仅宽度缩放两种场景下都实现了训练动态的“崩塌”与超参数迁移。

**🔧 技术方法**

使用的技术包括：能量基模型框架、动态系统解析、比例缩放（proportional scaling）理论、中心化技术、SGD 与 Adam 优化器、数值实验（线性、ReLU^p、Softmax DenseAM），以及对比实验验证理论预测。

**📊 数据集**

主要使用的数据集为：1) 生成的等方差高斯噪声数据（x∼𝒩(0,I_N)）进行 denoising 任务；2) MNIST 数据集在不同分辨率下进行实验；3) 通过投影或上采样得到的 anisotropic 高维数据。

**📈 对比分析**

比较方法为：在相同的训练任务（去噪、MNIST 分类/生成）下，分别采用理论给出的缩放参数和随机初始化参数，评估学习率迁移、训练损失演化和内部统计的一致性。实验结果显示：理论缩放参数在不同规模下保持了相同的学习曲线、损失收敛速度和梯度统计，验证了迁移效果；相对随机初始化，学习率迁移方案显著提升了训练稳定性和收敛速度。

**⚠️ 局限性**

局限性包括：①仅针对浅层 DenseAM 进行分析与验证，深层或更复杂结构（如 Energy Transformer、NRGPT）尚未验证；②理论推导假设输入为等方差高斯分布，对非球面数据的解释仍不完整；③未给出完整的无穷规模训练动力学理论，仍需进一步研究；④在 Softmax 激活下对 SGD 的数值不稳定性说明需要更深的优化器设计。

---

## 62. Statistical Scouting Finds Debate-Safe but Not Debate-Useful Cases: A Matched-Ceiling Study of Open-Weight LLM Reasoning Protocols

**arXiv ID:** 2605.09618 | [PDF](https://arxiv.org/pdf/2605.09618v1)

**作者:** Julia Hu `[一作]` (Amazon Web Services), Kumar Lakshmipathi `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估8B模型在匹配生成token上限下的三种推理协议（贪婪、三样投票、双代理辩论）并探讨按例协议切换的可行性

**💡 创新点**

提出“oracle gap”概念，证明仅用投票熵阈值即可捕获约11–19%的潜在提升，同时揭示投票熵只预测安全而非实用性

**🔧 技术方法**

利用多路径推理、投票熵阈值控制器以及学习型控制器（逻辑回归、梯度提升树）进行实验

**📊 数据集**

MuSiQue（多跳问答）和GSM8K（算术题）两个数据集

**📈 对比分析**

与最佳固定协议相比，oracle路由在MuSiQue上提升约+14pp；阈值控制器相对固定协议提升+1.3/1.7pp，覆盖约11–19%的oracle头room；学习控制器未优于阈值

**⚠️ 局限性**

仅包含8B开源模型、两类任务、特征有限且未使用logprob、样本量（N=300）有限，且一次性行为探针未能产生可用信号

---

## 63. Minimal Filling Architectures of Polynomial Neural Networks: Counterexamples, Frontier Search, and Defects

**arXiv ID:** 2605.09609 | [PDF](https://arxiv.org/pdf/2605.09609v1)

**作者:** Kevin Dao `[一作]`, Jose Israel Rodriguez `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文通过前沿搜索与符号计算，给出了多项式激活函数 PNN 的一个最小填充架构非单调性的反例，证明了最小单调性猜想不成立。

**💡 创新点**

首次构造出非单调的最小填充架构，并揭示其在低层次存在较大缺陷性的现象，为多项式网络的架构设计提供了新的洞见。

**🔧 技术方法**

采用前沿搜索、有限域维数计算、递归维数上界、符号计算以及 Dickson’s Lemma 等代数几何和符号算法技术。

**📊 数据集**

论文未使用传统机器学习数据集，而是基于符号/代数计算得出的理论结果。

**📈 对比分析**

通过比较架构对应的神经多样性维数与包络空间维数，验证其是否填充；在 r=2 的情况下，证明该架构维数达到 65，说明其是填充且非单调。

**⚠️ 局限性**

主要限制在于缺乏对更大深度或宽度架构完整性的证明，且方法主要适用于理论分析，尚未扩展到实际训练和更广泛的网络类型。

---

## 64. Attention Itself Could Retrieve.RetrieveVGGT: Training-Free Long Context Streaming 3D Reconstruction via Query-Key Similarity Retrieval

**arXiv ID:** 2605.09644 | [PDF](https://arxiv.org/pdf/2605.09644v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 65. Optimizer-Induced Mode Connectivity: From AdamW to Muon

**arXiv ID:** 2605.09991 | [PDF](https://arxiv.org/pdf/2605.09991v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 66. Sparse Discrete Laplace and Gaussian Mechanisms under Local Differential Privacy

**arXiv ID:** 2605.09561 | [PDF](https://arxiv.org/pdf/2605.09561v1)

**作者:** Amirreza Zamani `[一作]` (KTH), Mikael Skoglund `[通讯]` (KTH)

**通讯引用:** 8960 | [OpenAlex ID](https://openalex.org/A5041348422)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了可将输出空间限制为输入相关的稀疏支持集的局部差分隐私机制，并给出了稀疏离散拉普拉斯和稀疏高斯机制的纯ε-LDP与(ε,δ)-LDP的精确表述。

**💡 创新点**

首次将支持大小作为机制的内在复杂度参数，揭示了支持不匹配与重叠导致的隐私缺陷，并提出在满足隐私约束时选择最小可行支持是失真最优的设计原则。

**🔧 技术方法**

使用解析推导、hockey-stick 散度、精确隐私缺陷公式以及半径截断分析等技术，得到隐私-稀疏度与失真之间的闭式关系。

**📊 数据集**

论文完全为理论分析，没有使用真实数据集，所有结论基于离散度量和有限支持假设。

**📈 对比分析**

通过数值示例验证理论公式，展示了隐私缺陷随支持大小、拉普拉斯浓度或高斯尺度变化的非单调特性，并说明了隐私与失真的权衡；在给定隐私阈值下，最小支持能实现失真最小化。

**⚠️ 局限性**

局限性在于只考虑离散拉普拉斯和高斯核，未讨论更一般的核或连续空间；结果仅适用于半径截断的稀疏机制，缺乏实验验证；对实际数据集的泛化能力未进行评估。

---

## 67. HeteroGenManip: Generalizable Manipulation For Heterogeneous Object Interactions

**arXiv ID:** 2605.10201 | [PDF](https://arxiv.org/pdf/2605.10201v1)

**作者:** Zhenhao Shen `[一作]` (Peking University), Ruihai Wu `[通讯]` (Peking University)

**通讯引用:** 216 | [OpenAlex ID](https://openalex.org/A5086096450)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种两阶段框架HeteroGenManip，先利用结构先验和对应点实现异构物体的精准抓取，再通过多基础模型diffusion策略规划跨类型交互轨迹；

**💡 创新点**

创新点在于：①将抓取与轨迹规划解耦为两阶段；②利用结构先验进行对应点匹配实现“哪里抓取”；③采用多基础模型（如GAM、Uni3D等）为不同物体类别提取专属特征；④双流交叉注意力实现几何与语义的融合；

**🔧 技术方法**

技术包括：预训练基础模型（GAM、Uni3D、DINOv2等），多基础模型diffusion policy（DDIM），双流交叉注意力模块，PCA降维压缩语义特征，点云对应点匹配，结构先验抓取策略；

**📊 数据集**

数据集：DexGarmentLab、RoboTwin 以及自行设计的异构交互任务；真实世界使用Intel RealSense L515采集点云，Meta Quest 3 进行人类演示；

**📈 对比分析**

与 DP、DP3、GenDP-S、3DFA、Pi0.5 等 SOTA 方法对比。仿真中平均成功率0.78（比最佳基线提升≈31%），测试集0.65；真实世界训练/测试平均成功率分别83.3%/76.7%，显著优于基线；

**⚠️ 局限性**

局限性：对极端形变或动态环境的适应性尚待验证；需多种基础模型，模型调优和计算资源成本较高；实时性能和推理速度未做深入评估。

---

## 68. Strategic Exploitation in LLM Agent Markets: A Simulation Framework for E-Commerce Trust

**arXiv ID:** 2605.10059 | [PDF](https://arxiv.org/pdf/2605.10059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 69. Hypothesis-Driven Deep Research with Large Language Models: A Structured Methodology for Automated Knowledge Discovery

**arXiv ID:** 2605.10224 | [PDF](https://arxiv.org/pdf/2605.10224v1)

**作者:** Michael Chin `[一作]` `[通讯]` (Independent Researcher), Michael Chin (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于假设驱动的深度研究方法论（Hypothesis‑Driven Deep Research, HDDR），通过在研究全过程中生成、规划并验证假设来组织和推进信息检索与推理。

**💡 创新点**

创新点在于：① 将假设视为研究组织工具而非仅仅的科研结论；② 引入“gap‑driven”闭环迭代机制，自动识别信息与逻辑缺口并触发补充搜索；③ 设计置信度传播的事实推理框架，提供可量化的推理可靠性。

**🔧 技术方法**

核心技术包括大语言模型（LLM）驱动的假设生成、查询理解与扩展、智能多源检索、主体锁定（SubjectMatcher）、事实抽取与链式推理、跨源验证与冲突检测、以及基于置信度的决策与报告生成。

**📊 数据集**

使用了 50 条跨五个领域（企业、人物、技术趋势、行业、政策）的研究查询作为基准数据集，并辅以 5 篇案例研究，以评估方法的实际表现。

**📈 对比分析**

与直接检索+摘要、检索+推理、以及 Gemini Deep Research 三个基线对比，HDDR 在事实密度、主体匹配准确率、多源验证置信度、报告完整性等指标上分别提升 22.4%、90%、0.92、0.86，响应时间约 3.4 分钟，用户满意度达 4.2/5。

**⚠️ 局限性**

局限性包括：对 LLM 质量的高度依赖、受限于公开检索接口的覆盖范围、实验规模有限、时间敏感信息易过时、以及目前仅支持中英文查询。

---

## 70. Fashion130K: An E-commerce Fashion Dataset for Outfit Generation with Unified Multi-modal Condition

**arXiv ID:** 2605.10127 | [PDF](https://arxiv.org/pdf/2605.10127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 71. H-MAPS: Hierarchical Memory-Augmented Proactive Search Assistant for Scientific Literature

**arXiv ID:** 2605.10097 | [PDF](https://arxiv.org/pdf/2605.10097v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 72. Learning to Compress Time-to-Control: A Reinforcement Learning Framework for Chronic Disease Management

**arXiv ID:** 2605.09818 | [PDF](https://arxiv.org/pdf/2605.09818v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 73. ProteinOPD: Towards Effective and Efficient Preference Alignment for Protein Design

**arXiv ID:** 2605.10189 | [PDF](https://arxiv.org/pdf/2605.10189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 74. APEX: Audio Prototype EXplanations for Classification Tasks

**arXiv ID:** 2605.10153 | [PDF](https://arxiv.org/pdf/2605.10153v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 75. Active-SAOOD: Active Sparsely Annotated Oriented Object Detection in Remote Sensing Images

**arXiv ID:** 2605.10162 | [PDF](https://arxiv.org/pdf/2605.10162v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 76. Bayesian Optimization with Structured Measurements: A Vector-Valued RKHS Framework

**arXiv ID:** 2605.09775 | [PDF](https://arxiv.org/pdf/2605.09775v1)

**作者:** Wenbin Wang `[一作]` (EPFL), Colin N. Jones `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

本文提出了一种基于向量值测量的贝叶斯优化框架，利用向量值RKHS与线性测量算子构造测量空间中的核岭回归，推导了高概率收敛界并设计了UCB采样算法，证明了在常见核函数下的亚线性回报率，并在合成与真实建筑控制案例中验证了样本效率提升；

**💡 创新点**

创新点在于：①首次在贝叶斯优化中直接考虑结构化向量值观测，避免传统仅使用标量观测导致的信息损失；②通过诱导测量核在测量空间中建立估计，获得更紧的置信界；③将框架推广到时间变化与多目标情形，实现信息跨目标迁移；

**🔧 技术方法**

使用的技术包括：向量值RKHS、算子值核、核岭回归、诱导测量核、子高斯噪声假设、UCB采样策略、信息增益与回报分析、可追踪的可跟踪可分离核；

**📊 数据集**

使用的数据集包括：合成的多阶段目标仿真数据（3个不同线性目标），以及基于商业建筑仿真的MPC控制器参数优化数据（包含能源消耗、CO₂排放、热舒适度，模拟随时间变化的加热价格）；

**📈 对比分析**

与基线（标准BO、多任务BO、函数对函数BO、上下文BO）比较，vvBO在简单与累计回报上表现更佳，特别在时间变化目标下能快速调整并显著降低能耗成本；

**⚠️ 局限性**

局限性在于：对高维输入空间易受维数灾难影响；目前仅处理无约束问题，缺乏安全/约束处理；对真实系统的鲁棒性、隐私和公平性仍需进一步研究。

---

## 77. Rethinking Evaluation of Multiple Sclerosis (MS) Lesion Segmentation Models

**arXiv ID:** 2605.09666 | [PDF](https://arxiv.org/pdf/2605.09666v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 78. LASSA Architecture-Based Autonomous Fault-Tolerant Control of Unmanned Underwater Vehicles

**arXiv ID:** 2605.09494 | [PDF](https://arxiv.org/pdf/2605.09494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 79. Towards Compact Sign Language Translation: Frame Rate and Model Size Trade-offs

**arXiv ID:** 2605.09554 | [PDF](https://arxiv.org/pdf/2605.09554v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 80. Unsupervised Process Reward Models

**arXiv ID:** 2605.10158 | [PDF](https://arxiv.org/pdf/2605.10158v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 81. cantnlp@DravidianLangTech 2026: organic domain adaptation improves multi-class hope speech detection in Tulu

**arXiv ID:** 2605.09795 | [PDF](https://arxiv.org/pdf/2605.09795v1)

**作者:** Andrew Li `[一作]` (Lake Washington School District), Sidney Wong `[通讯]` (University of Otago)

**通讯引用:** 87 | [OpenAlex ID](https://openalex.org/A5103157160)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在 DravidianLangTech‑2026 共享任务中，构建了基于 XLM‑RoBERTa 的代码混合图卢语乐观演讲检测系统。

**💡 创新点**

创新点在于将 XLM‑RoBERTa 在有机收集的图卢语社交媒体文本（包含代码混合与混合书写）上进行领域适应，从而提升低资源语言的检测效果。

**🔧 技术方法**

技术方案包括多语言预训练模型 XLM‑RoBERTa、掩码语言建模域适应、以及后续的分类微调。

**📊 数据集**

使用的数据集为共享任务提供的标注训练/开发/测试集，以及从 Global Corpus of Language Use 收集的无标签图卢语社交媒体文本。

**📈 对比分析**

在开发集上，有机适应模型相较基线微小提升：Task 1 宏 F1 0.5238 对比 0.5227，Task 2 宏 F1 0.3416 对比 0.3171；在官方测试中分别排名第 5（宏 F1 0.50）和第 7（宏 F1 0.33）。

**⚠️ 局限性**

局限包括仅使用单一有机适应策略，未与合成脚本切换或其他数据增强方法直接对比；且未对类别不平衡进行专门处理，导致少数类表现不足。

---

## 82. Pseudo-Deliberation in Language Models: When Reasoning Fails to Align Values and Actions

**arXiv ID:** 2605.09893 | [PDF](https://arxiv.org/pdf/2605.09893v1)

**作者:** Sushrita Rakshit `[一作]` (New York University), Hua Shen `[通讯]` (New York University Shanghai)

**通讯引用:** 2348 | [OpenAlex ID](https://openalex.org/A5023117506)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为Valdi的框架，量化LLM在自然对话中所声称的价值与最终行为之间的差距，并通过大规模人类设计的DAISY数据集进行评估，提出并验证了多代理价值审计系统ViValdi；

**💡 创新点**

首次系统性地揭示了“伪审议”现象，即链式思考并不能缩小价值-行为鸿沟；提出了可在推理层和对话层进行审计与修正的多代理架构，证明后置对话修正对价值一致性更有效；

**🔧 技术方法**

采用多阶段提示与ValueJudge进行价值提取，利用LLM链式思考、规划与重写模块实现多代理审计；使用宏观F1、曼哈顿距离、价值存活/抑制/出现率等自定义指标进行量化；

**📊 数据集**

使用了4,941个人工构建、无显式价值描述的DAISY情境数据集，以及10%子集；在GPT‑4o、Llama‑3.1‑8B‑Instruct、Qwen3‑8B、Gemini‑3‑Flash等四款模型上进行实验；

**📈 对比分析**

通过对比Fast（无推理）与Slow（链式推理）生成方式，利用宏观F1与距离等指标发现推理往往降低一致性；在此基础上，ViValdi的对话层修正实现最高宏观F1≈0.58、最低距离≈0.108，且在人类对比评估中获胜率超过70%；

**⚠️ 局限性**

局限性包括：仅针对Schwartz价值体系；评估依赖特定提示与固定温度；数据集覆盖域有限；多代理系统仍需更多场景与更大修正预算验证；

---

## 83. When Normality Shifts: Risk-Aware Test-Time Adaptation for Unsupervised Tabular Anomaly Detection

**arXiv ID:** 2605.10242 | [PDF](https://arxiv.org/pdf/2605.10242v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 84. Adaptive Action Chunking via Multi-Chunk Q Value Estimation

**arXiv ID:** 2605.10044 | [PDF](https://arxiv.org/pdf/2605.10044v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 85. Beyond Autonomy: A Dynamic Tiered AgentRunner Framework for Governable and Resilient Enterprise AI Execution

**arXiv ID:** 2605.10223 | [PDF](https://arxiv.org/pdf/2605.10223v1)

**作者:** Kai Pan `[一作]` (A2A Lab), Rong Hou `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Dynamic Tiered AgentRunner，一种面向企业的多代理执行框架，结合风险自适应分层、权力分离与恢复闭环，实现对LLM驱动任务的可治理执行；

**💡 创新点**

创新点在于将治理机制拆解为三大核心：根据任务风险动态分配执行层级；通过物理隔离的提案、审核、执行、验证流程防止单一代理全权执行；以及将失败视为可恢复状态，形成自愈闭环；

**🔧 技术方法**

技术实现包括：LLM驱动的Worker、Critic、Verifier、Recovery等角色；ToolGateway六层校验管道（Schema、Permission、Scope、Risk、Idempotency、Execution）；Checkpoint持久化、事件驱动的状态机；以及基于阈值的风险评分公式与动态升级策略；

**📊 数据集**

使用的数据集为在多租户SaaS平台上收集的537条真实企业运营任务，包含信息查询、单对象写、批量写、跨域复杂任务；模型使用MiniMax‑M2.7与Kimi‑K2.6；

**📈 对比分析**

通过与单代理、静态全管线、去Critic、去Verifier、去Recovery等基线对比，AgentRunner实现88.9%任务成功率、0.5%未审计高风险执行错误、平均延迟22.4s、成本0.041美元，分别比静态全管线低47%延迟、58%成本，显著提升效率与安全；

**⚠️ 局限性**

局限包括：初始分层误判率约3.4%；Critic的误报率5-8%导致额外延迟；每个高级层级需额外LLM调用，形成最低延迟阈值；以及系统对模型改动的适配需要手工阈值维护。

---

## 86. Modeling Atomic Conformational Ensembles of Proteins via Test-Time Supervision of Boltz-2 on Cryo-EM Density Maps

**arXiv ID:** 2605.09832 | [PDF](https://arxiv.org/pdf/2605.09832v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 87. ASACK : Adaptive Safe Active Continual Koopman Learning for Uncertain Systems with Contractive Guarantees

**arXiv ID:** 2605.09659 | [PDF](https://arxiv.org/pdf/2605.09659v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 88. Medical Incident Causal Factors and Preventive Measures Generation Using Tag-based Example Selection in Few-shot Learning

**arXiv ID:** 2605.10025 | [PDF](https://arxiv.org/pdf/2605.10025v1)

**作者:** Yuna Haseyama `[一作]` (Hokkaido University), Itsuki Noda `[通讯]` (Hokkaido University)

**通讯引用:** 3750 | [OpenAlex ID](https://openalex.org/A5076716774)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于标签的少样本示例选择方法，用于提示大语言模型生成医疗事故背景/因果因素和预防措施

**💡 创新点**

创新点在于利用数据集中已有的可解释标签（如“用药”“输血”等）来挑选少量示例，提升生成精度与稳定性，且不依赖余弦相似度计算

**🔧 技术方法**

采用少样本提示（5例），使用GPT‑4o与LLaMA 3.3两大模型，结合ROUGE‑1/L与BERTScore进行评估

**📊 数据集**

使用日语医疗事故数据集JMID（3,884条记录，含18类标签）

**📈 对比分析**

与零样本、随机抽样和余弦相似度抽样三种基线对比，tag‑based方法在BERTScore与ROUGE‑L精度均最高，并且在生成过程中几乎无安全过滤或错误输出，性能最为稳定

**⚠️ 局限性**

局限性：示例集固定，未尝试动态选择；缺乏临床专家评估；模型偶尔会加入与参考不同但可能合法的额外信息，导致自动评估指标下降

---

## 89. Learning to Perceive "Where": Spatial Pretext Tasks for Robust Self-Supervised Learning

**arXiv ID:** 2605.09963 | [PDF](https://arxiv.org/pdf/2605.09963v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 90. ASTRA-QA: A Benchmark for Abstract Question Answering over Documents

**arXiv ID:** 2605.10168 | [PDF](https://arxiv.org/pdf/2605.10168v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 91. Relative Score Policy Optimization for Diffusion Language Models

**arXiv ID:** 2605.10218 | [PDF](https://arxiv.org/pdf/2605.10218v1)

**作者:** Zichao Yu `[一作]` (University of Science and Technology of China), Difan Zou `[通讯]` (University of Hong Kong)

**通讯引用:** 2647 | [OpenAlex ID](https://openalex.org/A5085848346)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RSPO（Relative Score Policy Optimization）方法，用于在扩散语言模型的RLVR后训练中校准噪声ELBO估计并提升推理与规划能力

**💡 创新点**

将verifier奖励转化为相对得分目标，利用残差反馈、参考模型减法和中心化实现对噪声相对分数的校准，而非仅用优势加权

**🔧 技术方法**

基于扩散模型的ELBO相对分数、离线参考模型、stop‑gradient残差反馈以及LoRA微调

**📊 数据集**

在数学推理（GSM8K、MATH500）和规划（Countdown、Sudoku）数据集上进行评估

**📈 对比分析**

与多种dLLM RL基线（d1/Diffu‑GRPO、VRPO、wd1、SAPO、TraceRL）比较，RSPO在规划任务上显著提升（如Sudoku 92.1%）并在数学推理上保持竞争力

**⚠️ 局限性**

缺乏自适应的λ调度、参考策略动态更新以及全局收敛理论；对ELBO分数波动的诊断仍为局部近似

---

## 92. High Precision Hydraulic Excavator Control for Heavy-Duty Grading

**arXiv ID:** 2605.09465 | [PDF](https://arxiv.org/pdf/2605.09465v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 93. Fix the Loss, Not the Radius: Rethinking the Adversarial Perturbation of Sharpness-Aware Minimization

**arXiv ID:** 2605.10183 | [PDF](https://arxiv.org/pdf/2605.10183v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 94. Think as Needed: Geometry-Driven Adaptive Perception for Autonomous Driving

**arXiv ID:** 2605.10117 | [PDF](https://arxiv.org/pdf/2605.10117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 95. GELATO: Generative Entropy- and Lyapunov-based Adaptive Token Offloading for Device-Edge Speculative LLM Inference

**arXiv ID:** 2605.10124 | [PDF](https://arxiv.org/pdf/2605.10124v1)

**作者:** Zengzipeng Tang `[一作]` (Beijing Jiaotong University), Bo Ai `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 28505 | [OpenAlex ID](https://openalex.org/A5100620739)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GELATO 框架，解决设备‑边缘协作推理中的自回归生成与资源调度问题，通过层级 Lyapunov 动态预算和熵驱动的早停，实现高吞吐低能耗的推理。

**💡 创新点**

创新点：①使用期望代理与 Lyapunov 优化实现长期吞吐与能耗约束的在线预算调度；②嵌套生成熵驱动的实时早停机制，精细捕获每个 token 的生成不确定性；③结合概率 top‑p 压缩的通信方案，显著降低传输负载。

**🔧 技术方法**

采用 Lyapunov 优化框架、生成熵与接受率的统计映射、top‑p 概率压缩、离散一维预算搜索及在线算法实现，并通过仿真评估。

**📊 数据集**

使用 GSM8K 作为数学推理基准，并以 Qwen2.5‑7B（边缘）与 Qwen2.5‑0.5B（设备）模型进行实验。

**📈 对比分析**

与静态 Speculative Decoding（Static SD）和 Distributed Split Speculative Decoding（DSSD）等基线进行对比，在 1 MHz 带宽下吞吐提升 64.98%，能耗降低 47.47%；在高延迟（140 ms）下吞吐提升 20.04%–25.95%，能耗降低 23.39%。

**⚠️ 局限性**

局限性：依赖模型特定的接受率‑熵映射，需先行训练；只关注推理阶段，未覆盖前置缓存或多任务场景；对极端动态环境（如频繁切换的带宽/功率）适应性待验证。

---

## 96. HAGE: Harnessing Agentic Memory via RL-Driven Weighted Graph Evolution

**arXiv ID:** 2605.09942 | [PDF](https://arxiv.org/pdf/2605.09942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 97. Learning Unified Representations of Normalcy for Time Series Anomaly Detection

**arXiv ID:** 2605.09685 | [PDF](https://arxiv.org/pdf/2605.09685v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 98. Who embraces AI in play? Exploratory modeling of player preference profiles toward game AI

**arXiv ID:** 2605.09550 | [PDF](https://arxiv.org/pdf/2605.09550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 99. The Benefits of Temporal Correlations: SGD Learns k-Juntas from Random Walks Efficiently

**arXiv ID:** 2605.10237 | [PDF](https://arxiv.org/pdf/2605.10237v1)

**作者:** Elisabetta Cornacchia `[一作]` (Bocconi University), Elchanan Mossel `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 9969 | [OpenAlex ID](https://openalex.org/A5013467728)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了时间相关性（通过超立方体上的随机行走产生的样本）如何使稀疏学习问题（如 Boolean k‑junta）在梯度下降方法下可有效学习。

**💡 创新点**

创新点在于证明：在时间相关的随机行走数据下，采用两层 ReLU 网络配合时间差（TD）损失的 SGD 可以在样本复杂度仅线性于维度 d（k 固定）内学习任何 k‑junta；并且给出了该方法的上界与下界，揭示了时间相关性对梯度学习的突破性影响。

**🔧 技术方法**

使用的技术包括：两阶段层级 SGD（第一阶段用大批量 TD 损失提取支持信息，第二阶段用 L2 损失做低维回归）；量化影响量（Influence）的非退化条件；中心极限定理用于抗集中；随机游走下的 Markovian SGD 收敛分析；以及对大批量点wise 损失的下界证明。

**📊 数据集**

实验采用合成数据：d=50 的 5‑parity 与 7‑junta（含低阶项）以及标准随机 i.i.d. 数据，比较 TD 损失与平方损失在小批量（batch=1）与大批量（batch≈d）下的表现。

**📈 对比分析**

比较结果显示：在随机行走数据下，TD 损失使网络在较小样本量下即可达到高测试精度，显著优于 i.i.d. 训练；而大批量点wise 损失（如平方损失）无法利用时间相关性，表现与 i.i.d. 类似；小批量平方损失在随机行走数据下也能学习成功，甚至优于 TD 损失，表明批量大小对是否利用时间相关性至关重要。

**⚠️ 局限性**

局限性包括：理论下界仅适用于大批量点wise 损失；对小批量或非单位位翻转的随机游走未给出理论；仅针对 Boolean k‑junta，扩展到更一般稀疏或连续任务需要进一步研究；并且实验仅在合成数据上验证，真实数据的可行性仍待探索。

---

## 100. MicroViTv2: Beyond the FLOPS for Edge Energy-Friendly Vision Transformers

**arXiv ID:** 2605.10148 | [PDF](https://arxiv.org/pdf/2605.10148v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 101. CALYREX: Cross-Attention LaYeR EXtended Transformers for System Prompt Anchoring

**arXiv ID:** 2605.09737 | [PDF](https://arxiv.org/pdf/2605.09737v1)

**作者:** Li Lixing `[一作]` (Cornell University), Li Lixing `[通讯]` (Cornell University)

**通讯引用:** 22472 | [OpenAlex ID](https://openalex.org/A5069990247)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种跨注意力层架构CALYREX，用于在LLM中结构化隔离系统提示并提升指令遵循与防注入性能

**💡 创新点**

通过在冻结模型的后八层插入零初始化跨注意力块，将系统提示与全局输入分离，形成专门的路由路径，解决“指令层次”问题

**🔧 技术方法**

零初始化跨注意力（CAL）、SwiGLU MLP、RMSNorm、批量化处理与冻结backbone训练、结构化掩码

**📊 数据集**

在一个50,000条样本的通用SFT语料上训练，基准测试使用IFEval、Long-IFEval、SysBench、MMLU、GSM8K、SQuAD、Glaive FC、IH-Challenge、InjecAgent、Many-Shot Jailbreak、TensorTrust

**📈 对比分析**

与同参数量的LoRA全网络微调和Late8th位置的ParallelMLP基线对比，1.5B模型时差距不大，但8B模型时CALYREX在指令遵循、连续对话和多轮 jailbreak 防御上分别提升约+7.4%、+16.3%和降低13%攻击成功率

**⚠️ 局限性**

对特定任务（如JSON结构化生成、信息提取）无明显优势，结构化路由在缺乏针对性训练数据时受限；最佳插入位置在更大规模模型上是否保持最佳仍待验证

---

## 102. ORICF -- Open Robotics Inference and Control Framework

**arXiv ID:** 2605.09656 | [PDF](https://arxiv.org/pdf/2605.09656v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 103. When Does Non-Uniform Replay Matter in Reinforcement Learning?

**arXiv ID:** 2605.10236 | [PDF](https://arxiv.org/pdf/2605.10236v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 104. Improving Temporal Action Segmentation via Constraint-Aware Decoding

**arXiv ID:** 2605.10149 | [PDF](https://arxiv.org/pdf/2605.10149v1)

**作者:** Yeo Keat Ee `[一作]` (Agency for Science, Technology and Research), Basura Fernando `[通讯]` (Agency for Science, Technology and Research)

**通讯引用:** 8904 | [OpenAlex ID](https://openalex.org/A5090467618)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级的约束感知Viterbi解码方法，利用从标注数据直接提取的转移置信度、起止动作集合和持续时间约束，对时序动作分割模型的输出进行后处理，以提升分割质量。

**💡 创新点**

创新点在于将统计得到的结构先验直接嵌入解码过程，避免了复杂的语法生成与解析，既保持了模型的可训练性，又显著提升了时序一致性和可解释性。

**🔧 技术方法**

核心技术为基于Viterbi的动态规划算法，结合硬约束（起止动作、合法转移、持续时间上下限）以及可选的软约束衰减；同时在半监督学习中将该解码用于伪标签修正。

**📊 数据集**

实验使用了两个主流时序动作分割基准数据集——50Salads和Breakfast。

**📈 对比分析**

与原始模型、KARI等语法基方法对比，所提方法在帧级准确率、Edit距离和F1@{10,25,50}指标上都有提升，且在半监督设置下显著加速收敛；同时运行时长比基于语法解析的方法更短。

**⚠️ 局限性**

主要局限在于约束的效果高度依赖于训练集中的结构统计；若训练数据稀缺或含噪，所提硬约束可能过于严格，导致对测试中未见序列的泛化能力下降。

---

## 105. From Syntax to Semantics: Unveiling the Emergence of Chirality in SMILES Translation Models

**arXiv ID:** 2605.09949 | [PDF](https://arxiv.org/pdf/2605.09949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 106. EgoMemReason: A Memory-Driven Reasoning Benchmark for Long-Horizon Egocentric Video Understanding

**arXiv ID:** 2605.09874 | [PDF](https://arxiv.org/pdf/2605.09874v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 107. BEA-GS: BEyond RAdiance Supervision in 3DGS for Precise Object Extraction

**arXiv ID:** 2605.09662 | [PDF](https://arxiv.org/pdf/2605.09662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 108. Beyond Position Bias: Shifting Context Compression from Position-Driven to Semantic-Driven

**arXiv ID:** 2605.09463 | [PDF](https://arxiv.org/pdf/2605.09463v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 109. Dynamic Edge Coloring of Forests

**arXiv ID:** 2605.09711 | [PDF](https://arxiv.org/pdf/2605.09711v1)

**作者:** Haim Kaplan `[一作]` (Tel Aviv University), Yaniv Sadeh `[通讯]` (Tel Aviv University)

**通讯引用:** 50 | [OpenAlex ID](https://openalex.org/A5066150263)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在树（forest）上的动态边着色问题，分析增量（仅插入）与完全动态（增删）两种模型下的重着色代价(recursion)。针对确定性贪心算法给出上界与下界，证明其在增量模型中达到最优的亚常数调和（amortized）重着色，而在完全动态模型下必然出现 Ω(logΔ n) 的重着色。为此提出一种非贪心的 O(1) 调和重着色算法（仅在有根森林且使用 2Δ-2 颜色时可行），并设计了保持随机均匀分布的维护算法，使增量模型期望调和重着色达到 Θ(1/Δ)，完全动态模型达到 Θ(min{Δ, logΔ n})。

**💡 创新点**

创新点：
1) 首次给出树上动态边着色的完整确定性分析，证明贪心算法在增量模型下的上界与下界匹配。
2) 证明在完全动态模型中贪心算法不可避免的 Ω(logΔ n) 重着色，并构造相应的极端实例。
3) 提出仅需 2Δ-2 颜色的根化森林上 O(1) 调和重着色算法，突破了无根情况下的 Ω(logΔ n) 下界。
4) 设计保持随机均匀分布的在线维护算法，首次将随机化技术用于动态边着色，得到最优的期望调和重着色。

**🔧 技术方法**

使用技术：
- 细致的递推与充电（charging）分析，分重、轻插入两种情形；
- 分层树（P‑layered tree）工具，用于证明贪心算法在完全动态模型下的下界；
- 变形贪心、链式贪心等变体的比较；
- 随机分布维护（top‑down uniform coloring）与修复路径（repair path）技术；
- 递推潜在函数与桶（bucket）技术来证明 O(1/Δ) 的期望调和重着色；
- 证明 2Δ-2 颜色的分布保持与根化森林的等价性。

**📊 数据集**

数据集：本工作为理论算法研究，未使用实验数据集；所有结论均基于严格的数学证明与构造性下界。

**📈 对比分析**

比较方法与性能：
- 对比贪心算法的最优性：在增量模型下实现 O(1/Δ)（或更低）调和重着色；在完全动态模型下证明任何贪心算法至少需 Ω(logΔ n)。
- 与非贪心 O(1) 算法对比：在有根森林、使用 2Δ-2 颜色时，该算法达到 O(1) 调和重着色，匹配下界。
- 与随机算法对比：随机分布维护算法在增量模型下实现 Θ(1/Δ) 期望调和重着色，在完全动态模型下实现 Θ(min{Δ, logΔ n})，均优于任何确定性贪心方案。
- 总结：在满足额外假设（有根、额外颜色）时，本文算法几乎最优；在一般无根森林上则仍受 Ω(logΔ n) 下界限制。

**⚠️ 局限性**

局限性：
1) 对根化森林及 2Δ-2 颜色的依赖限制了算法在无根情况或颜色不足时的适用性。
2) 在完全动态无根森林上，仍无法突破 Ω(logΔ n) 的下界；
3) 随机算法虽然在理论上最优，但实现需要对每个顶点重新采样子树颜色，实际运行时可能有较高的常数因子；
4) 本研究仅覆盖树结构，未探讨更一般的图（如有环或更高树枝数）的动态边着色。

---

## 110. Functional Stable Model Semantics and Answer Set Programming Modulo Theories

**arXiv ID:** 2605.09524 | [PDF](https://arxiv.org/pdf/2605.09524v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 111. To Redact, or not to Redact? A Local LLM Approach to Deliberative Process Privilege Classification

**arXiv ID:** 2605.10211 | [PDF](https://arxiv.org/pdf/2605.10211v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 112. Structure from Strategic Interaction & Uncertainty Risk Sensitive Games for Robust Preference Learning

**arXiv ID:** 2605.09946 | [PDF](https://arxiv.org/pdf/2605.09946v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 113. Route Before Retrieve: Activating Latent Routing Abilities of LLMs for RAG vs. Long-Context Selection

**arXiv ID:** 2605.10235 | [PDF](https://arxiv.org/pdf/2605.10235v1)

**作者:** Yiwen Chen `[一作]` (Beihang University), Minhao Cheng `[通讯]` (Pennsylvania State University)

**通讯引用:** 1813 | [OpenAlex ID](https://openalex.org/A5000534132)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了Pre-Route框架，利用轻量级元信息在答复前主动进行结构化推理，决定是否采用检索增强生成（RAG）或长上下文（LC）路径；

**💡 创新点**

发现LLM内部已潜藏可被激活的路由能力，并通过有结构的提示激活、线性探测验证其可分离性，以及通过蒸馏将此能力迁移至小模型，显著提升路由效率与可解释性；

**🔧 技术方法**

采用结构化提示设计、线性探测分析、蒸馏学习、成本分解与评估、元信息抽取等技术，并在LLM上实现多步推理与决策；

**📊 数据集**

在LaRA（in‑distribution）和LongBench‑v2（out‑of‑distribution）两个长文本问答基准上进行评估；

**📈 对比分析**

与Always‑RAG、Always‑LC和Self‑Route等基线对比，Pre‑Route在QA分数上更高、LC使用率更低，成本效益最高；蒸馏后的1.7B模型虽规模小，但仍接近教师模型性能；

**⚠️ 局限性**

对元信息的依赖较大，二元路由决策无法覆盖更细粒度策略，跨语言与跨领域推广需进一步验证，蒸馏过程需强教师与高质量过滤数据，阈值设定对成本和效果均有影响。

---

## 114. MemPrivacy: Privacy-Preserving Personalized Memory Management for Edge-Cloud Agents

**arXiv ID:** 2605.09530 | [PDF](https://arxiv.org/pdf/2605.09530v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 115. One for All: A Non-Linear Transformer can Enable Cross-Domain Generalization for In-Context Reinforcement Learning

**arXiv ID:** 2605.09727 | [PDF](https://arxiv.org/pdf/2605.09727v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 116. Crosslingual On-Policy Self-Distillation for Multilingual Reasoning

**arXiv ID:** 2605.09548 | [PDF](https://arxiv.org/pdf/2605.09548v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 117. PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions

**arXiv ID:** 2605.09538 | [PDF](https://arxiv.org/pdf/2605.09538v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 118. Many Needles in a Haystack: Active Hit Discovery for Perturbation Experiments

**arXiv ID:** 2605.10196 | [PDF](https://arxiv.org/pdf/2605.10196v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 119. K12-KGraph: A Curriculum-Aligned Knowledge Graph for Benchmarking and Training Educational LLMs

**arXiv ID:** 2605.09635 | [PDF](https://arxiv.org/pdf/2605.09635v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 120. A Cognitively Grounded Bayesian Framework for Misinformation Susceptibility

**arXiv ID:** 2605.09483 | [PDF](https://arxiv.org/pdf/2605.09483v1)

**作者:** Pranava Madhyastha `[一作]` `[通讯]`, Pranava Madhyastha

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Bounded Pragmatic Listener模型，结合递归深度、先验压缩和可用性采样三种认知约束，对读者推理过程建模，评估假新闻的易受性，并利用LLM实现语义化先验与可用性评估。

**💡 创新点**

创新点在于将RSA框架扩展为可约束的说话者-听者推理模型，映射信息失真类型到递归深度，并通过先验压缩与可用性采样捕捉人类有限认知；提出深度不匹配悖论并用群体分布预测注释分歧。

**🔧 技术方法**

采用资源有限的贝叶斯推理、信息瓶颈压缩、重要性采样以及Gemini 2.5 Pro LLM进行语义化先验与可用性评估，并构建RSA推理层。

**📊 数据集**

使用PolitiFact（12,836条声称）和Snopes-like 36,534条多领域事实核查数据集，包含6级真实性标签与来源可信度信息。

**📈 对比分析**

与表面特征基线（情绪词典、说话者历史、来源可信度）和纯RSA进行对比；BPL Full在PolitiFact上达到AUC 0.930、F1 0.885；单一易受性得分σ可实现AUC 0.811；LLM混合模型在样本50上相较特征基线提升AUC 17.3点。

**⚠️ 局限性**

局限性包括：特征泄漏导致表面基线过高；深度不匹配悖论与样本偏差；LLM计算成本高；模型仍为二元真伪，未处理细粒度真实性；以及对深度计数的简单依赖，缺乏复杂语法解析。

---

## 121. CCD-Level and Load-Aware Thread Orchestration for In-Memory Vector ANNS on Multi-Core CPUs

**arXiv ID:** 2605.10090 | [PDF](https://arxiv.org/pdf/2605.10090v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 122. Forcing-KV: Hybrid KV Cache Compression for Efficient Autoregressive Video Diffusion Models

**arXiv ID:** 2605.09681 | [PDF](https://arxiv.org/pdf/2605.09681v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 123. Zoom, Don't Wander: Why Regional Search Outperforms Pareto Reasoning and Global Optimization in Budget-Constrained SBSE

**arXiv ID:** 2605.09658 | [PDF](https://arxiv.org/pdf/2605.09658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 124. Beyond Self-Play and Scale: A Behavior Benchmark for Generalization in Autonomous Driving

**arXiv ID:** 2605.10034 | [PDF](https://arxiv.org/pdf/2605.10034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 125. Minimizing Worst-Case Weighted Latency for Multi-Robot Persistent Monitoring: Theory and RL-Based Solutions

**arXiv ID:** 2605.09633 | [PDF](https://arxiv.org/pdf/2605.09633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 126. MoPO: Incorporating Motion Prior for Occluded Human Mesh Recovery

**arXiv ID:** 2605.09856 | [PDF](https://arxiv.org/pdf/2605.09856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 127. Cloud Performance Decomposition for Long-Term Performance Engineering: A Case Study

**arXiv ID:** 2605.09787 | [PDF](https://arxiv.org/pdf/2605.09787v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 128. Attribution-based Explanations for Markov Decision Processes

**arXiv ID:** 2605.09780 | [PDF](https://arxiv.org/pdf/2605.09780v1)

**作者:** Paul Kobialka `[一作]` (University of Oslo), Einar Broch Johnsen `[通讯]` (University of Oslo)

**通讯引用:** 3024 | [OpenAlex ID](https://openalex.org/A5039414480)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种针对马尔可夫决策过程（MDP）的归因解释框架，能够为状态和执行路径分配重要性分数，帮助人类理解连续决策的关键因素。

**💡 创新点**

创新点在于将归因方法从单步预测扩展到序列决策，引入四类归因问题（状态/路径的固定策略与全局策略），并用策略综合与非线性/线性优化编码高效计算重要性区间。

**🔧 技术方法**

核心技术包括：MDP模型定义、策略综合、LTL语义、非线性（MIQCQP）与混合整数线性规划（MILP）优化，以及对策略的记忆化处理和贝尔曼最优性约束。

**📊 数据集**

使用的数据集有五个案例：BPIC12、BPIC17（贷款审批过程）、GrepS（编程技能评估）、MSSD（音乐流媒体会话）以及Epidemic（疫苗接种模拟），其中MSSD和Epidemic模型尺寸均达千级状态/转移。

**📈 对比分析**

通过对三种编码（非线性、非线性优先、线性）在上述模型上求解，结果显示线性编码在大多数实例中可在秒级完成，非线性编码在较小模型能完成但在大型模型出现超时，整体运行时间在数分钟至数小时不等，证明方法在10k状态规模下仍可扩展。

**⚠️ 局限性**

主要限制包括数值不稳定（大常数M导致精度损失）、对大概率分布的精度要求高、部分大型模型仍需超时或内存不足，以及缺乏针对子路径归因的支持。

---

## 129. Efficient Ensemble Selection from Binary and Pairwise Feedback

**arXiv ID:** 2605.09588 | [PDF](https://arxiv.org/pdf/2605.09588v1)

**作者:** Tzeh Yuan Neoh `[一作]` (Harvard University), Milind Tambe `[通讯]` (Harvard University)

**通讯引用:** 23386 | [OpenAlex ID](https://openalex.org/A5000327528)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何在未知任务分布下高效选择少量LLM集合。

**💡 创新点**

将集合选择建模为分布式多赢者投票，提出二元与序数反馈下的覆盖与θ胜利目标，并给出查询效率与近似保证。

**🔧 技术方法**

采用子模优化、贪心+失败条件查询、加权序数覆盖、最小化包装以及极小化包装等技术。

**📊 数据集**

在多语言抽取式QA与LiveBench评分生成的序数反馈上实验。

**📈 对比分析**

与全信息ER M、随机抽样和Top‑k基准对比，实验显示在相同查询预算下，适应性方法在覆盖率/θ目标上匹配或超越基线，且能利用模型互补。

**⚠️ 局限性**

局限在于需要已知任务分布近似、适用于可验证的客观或轻度主观任务，且对极端多数循环或无解问题仍无确定性保证。

---

## 130. StereoPolicy: Improving Robotic Manipulation Policies via Stereo Perception

**arXiv ID:** 2605.09989 | [PDF](https://arxiv.org/pdf/2605.09989v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 131. The Impact of Editorial Intervention on Detecting Native Language Traces

**arXiv ID:** 2605.10216 | [PDF](https://arxiv.org/pdf/2605.10216v1)

**作者:** Ahmet Yavuz Uluslu `[一作]` (University of Cambridge), Gerold Schneider `[通讯]` (University of Zurich)

**通讯引用:** 7872 | [OpenAlex ID](https://openalex.org/A5082637973)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了不同程度的人工智能编辑（从最小语法纠错到自由改写）对英语学习者文本的母语识别（NLI）性能的影响。

**💡 创新点**

首次将NLI性能与编辑强度在一个连续四阶段上做系统对比，揭示了除了语法错误之外更深层的语言转移特征在编辑过程中的存活与消失。

**🔧 技术方法**

使用 GPT‑4o 进行零样本 NLI 分类；使用 GPT‑4o‑mini 生成最小编辑、流畅编辑和改写文本；利用 M²、WER、BERTScore 等指标评估编辑差异；通过标签随机化和实体屏蔽降低偏差。

**📊 数据集**

写作 & 提升 2024（Write & Improve 2024）数据集的 450 篇 9 种母语（阿拉伯语、中文、法语、德语、印地语、意大利语、日语、西班牙语、土耳其语）样本。

**📈 对比分析**

在原始文本上准确率 88.9%；最小编辑（人工或自动）下 85.1%；流畅编辑 64.9%；自由改写 28.7%；编辑距离与准确率呈负相关，说明改写越多 NLI 识别越差。

**⚠️ 局限性**

仅测试单次 AI 编辑，未覆盖真实迭代共创场景；仅针对 L2 英语，缺乏跨语言验证；实验数据有限，可能不具备普适性。

---

## 132. EvoPref: Multi-Objective Evolutionary Optimization Discovers Diverse LLM Alignments Beyond Gradient Descent

**arXiv ID:** 2605.09777 | [PDF](https://arxiv.org/pdf/2605.09777v1)

**作者:** Dongxin Guo `[一作]` (University of Hong Kong), Siu Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22462 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过多目标进化算法维护LoRA适配器种群，实现LLM在有用、无害和诚实三重目标上的对齐，并生成多样化的适配器档案。

**💡 创新点**

将NSGA-II与基于网格的档案结合，提供多目标进化与多样性保护机制，显著提升对齐多样性并降低偏好崩塌。

**🔧 技术方法**

使用LoRA低秩适配器、NSGA-II多目标进化、档案式多样性维护、LoRA保持低秩的交叉、Rechenberg 1/5自适应变异等技术。

**📊 数据集**

训练使用Anthropic Helpful‑Harmless RLHF（170K对照），评估使用RewardBench、MT‑Bench、TruthfulQA、Safety Eval等基准。

**📈 对比分析**

与梯度基准（DPO、IPO、KTO、ORPO）、单目标进化（CMA‑ES）和其他多目标进化（MOEA/D、SMS‑EMOA）在30次独立实验下对比；EvoPref在多样性覆盖率提升18%，崩塌率下降47%，且在奖励、对话质量与安全上保持或略优。

**⚠️ 局限性**

依赖代理评价模型、仅验证7B规模、理论分析简化假设、计算成本高（需多模型存储）等限制。

---

## 133. Metal-Sci: A Scientific Compute Benchmark for Evolutionary LLM Kernel Search on Apple Silicon

**arXiv ID:** 2605.09708 | [PDF](https://arxiv.org/pdf/2605.09708v1)

**作者:** Víctor Gallego `[一作]` `[通讯]` (Komorebi AI Technologies), Víctor Gallego (Komorebi AI Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Metal‑Sci 10 任务基准及其针对 Apple Silicon Metal 的自动化进化式 LLM 代码搜索 harness，利用 runtime 编译、结构化反馈与 (1+1) 进化策略；

**💡 创新点**

创新点在于：①引入单个 held‑out 评价门 (Φ_𝒯) 作为机械监督，能够在搜索过程中捕捉 LLM 生成代码的准确性与性能回归；②提供跨六大优化范式（stencil、n‑body、LBM、MD、PDE、FFT）的科学计算基准，满足 Metal 语法与统一内存特性的实际需求；

**🔧 技术方法**

技术手段包括：Metal 运行时编译、roofline 基准与几何平均分数、结构化编译/正确性诊断反馈、LLM 代码生成与自我评估循环；

**📊 数据集**

数据集为 10 个 Metal kernel，涵盖 6 个优化 regime，每个任务附带 CPU 参考实现、3 个 in‑distribution 尺寸与 1 个 held‑out 尺寸；

**📈 对比分析**

对 Claude Opus 4.7、Gemini 3.1 Pro 与 GPT‑5.5 在 M1 Pro 上进行匹配 (1+1) 迭代实验，得到 self‑speedup 1.00×–10.7×；held‑out 评估揭示不同模型在正确性或性能上的显著回归；

**⚠️ 局限性**

局限性包括：静态 per‑chip roofline 未考虑 SLC 状态；单一 (1+1) 进化易于在 10–25 迭代后停滞；缺乏多样化搜索策略、稀疏线性算子任务及跨芯片泛化测试。

---

## 134. Mitigating Multimodal Inconsistency via Cognitive Dual-Pathway Reasoning for Intent Recognition

**arXiv ID:** 2605.09468 | [PDF](https://arxiv.org/pdf/2605.09468v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 135. Rethinking Loss Reweighting for Imbalance Learning as an Inverse Problem: A Neural Collapse Point of View

**arXiv ID:** 2605.10047 | [PDF](https://arxiv.org/pdf/2605.10047v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 136. BathyFacto: Refraction-Aware Two-Media Neural Radiance Fields for Bathymetry

**arXiv ID:** 2605.10174 | [PDF](https://arxiv.org/pdf/2605.10174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 137. Scaling the Memory of Balanced Adam

**arXiv ID:** 2605.10119 | [PDF](https://arxiv.org/pdf/2605.10119v1)

**作者:** Alberto Fernández-Hernández `[一作]` (Universitat Politècnica de València), Enrique S. Quintana-Ortí `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 6947 | [OpenAlex ID](https://openalex.org/A5012806004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过把 Adam 的两个动量参数绑定为同一值，研究如何在平衡 Adam 中选择单一的 β 参数，并提出基于内部统计刷新次数的 Rβ≈1000 的自适应调节规则。

**💡 创新点**

创新点在于将 β 视为与有效学习时长（T_ES）耦合的“统计记忆尺度”，而非单纯的无量纲常数；通过保持刷新次数恒定来实现跨模型、跨数据集的鲁棒性提升。

**🔧 技术方法**

技术手段包括：1) 计算有效学习时长 T_ES（基于验证曲线的早停估计）；2) 定义刷新计数 Rβ=(1‑β)T_ES；3) 在 11 个视觉与语言实验上进行 β 网格搜索、基准比较；4) 用相对验证损失、最大损失差和 CVaR 等指标评估鲁棒性。

**📊 数据集**

使用的数据集与模型包括：NanoGPT（WikiText‑103、OpenWebText）、Llama60M（C4、SlimPajama）、ViT‑B/16（CIFAR‑100、TinyImageNet）、ResNet50（Food‑101、ImageNet100）、EfficientNet‑B0（Cars）、T5‑small（BookCorpus）、Swin‑T（Caltech‑256）等，共 11 组实验。

**📈 对比分析**

比较方法：将固定 β=0.944 的基线与 Rβ≈1000 的刷新规则在 8 个开发实验和 3 个 hold‑out 实验上评估。性能结果显示，刷新规则将最大验证损失差从 1.328% 降至 0.885%（下降 33.4%），CVaR 从 1.059% 降至 0.713%（下降 32.7%），并使所有实验都在 1% 内接近验证最优点，平均误差略高于基线。

**⚠️ 局限性**

局限性包括：1) 需要先估计 T_ES，无法完全零样本；2) 仅在 11 个异构实验上验证，未覆盖所有模型/批量/学习率组合；3) 学习率调度未对每个 β 进行完全再调优；4) R₀=1000 是经验得到的值，未证明在更广泛场景下通用。

---

## 138. Plan2Cleanse: Test-Time Backdoor Defense via Monte-Carlo Planning in Deep Reinforcement Learning

**arXiv ID:** 2605.09638 | [PDF](https://arxiv.org/pdf/2605.09638v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 139. LLM-Guided Monte Carlo Tree Search over Knowledge Graphs: Composing Mechanistic Explanations for Drug-Disease Pairs

**arXiv ID:** 2605.09542 | [PDF](https://arxiv.org/pdf/2605.09542v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 140. Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal

**arXiv ID:** 2605.09502 | [PDF](https://arxiv.org/pdf/2605.09502v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 141. Push and Pushback in Contesting AI: Demands for and Resistance to Accountability

**arXiv ID:** 2605.09793 | [PDF](https://arxiv.org/pdf/2605.09793v1)

**作者:** Yulu Pi `[一作]` (Research Centre Trust, UA Ruhr, University of Duisburg-Essen), Jatinder Singh `[通讯]` (Research Centre Trust, UA Ruhr, University of Duisburg-Essen)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对43起真实AI争议案例进行主题分析，系统归纳争议主体、争议目标、机构回应策略、结果类型以及影响因素，形成对AI争议动态的经验性双向账户。

**💡 创新点**

首次提供从受影响者和机构双方视角出发的、基于经验的AI争议框架，揭示争议如何演变、机构如何回避或承担责任，并给出针对性行动指南。

**🔧 技术方法**

采用质性案例研究与主题分析方法，结合演绎与归纳编码，构建多维度编码方案并进行交叉验证。

**📊 数据集**

43起案例组成的研究数据集，来源包括AI Incident Database、学术文献、媒体报道和官方声明，覆盖教育、司法、保险、执法等多个AI应用领域。

**📈 对比分析**

通过跨案例对照与主题饱和度检验，比较不同争议策略与结果的关联性；研究未采用数值性能指标，而是以案例深度、归纳一致性和理论解释力作为评价。

**⚠️ 局限性**

仅限公开可见的高曝光争议，可能忽视低调或非公开的争议；研究依赖已有报道与文件，缺乏访谈或现场观察，可能导致信息缺失与偏倚。

---

## 142. Edit-Based Refinement for Parallel Masked Diffusion Language Models

**arXiv ID:** 2605.09603 | [PDF](https://arxiv.org/pdf/2605.09603v1)

**作者:** Houxing Ren `[一作]` (CUHK MMLab), Hongsheng Li `[通讯]` (CUHK MMLab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种编辑式改进框架（ME‑DLM），在并行掩码扩散语言模型生成初稿后，利用最小编辑操作（替换、删除、插入）进行局部修正，以提升多标记并行生成的质量。

**💡 创新点**

创新点在于：①用最小编辑距离作为监督信号，使模型学习仅作必要的局部修正；②在保持并行扩散效率的同时，加入序列级的全局一致性校正；③采用两阶段扩散：先粗略生成，再逐步编辑，弥补单标记训练目标对联合分布的不足。

**🔧 技术方法**

使用技术包括：掩码扩散语言模型（MDLM）框架、编辑式扩散（token‑level edit预测+deterministic编辑算子）、LLaDA 8B 体系结构、学习率曲线与梯度裁剪的微调策略，以及基于编辑距离的最短编辑脚本生成。

**📊 数据集**

训练使用 LLaDA‑8B‑Base 预训练模型，随后在 Nemotron‑Pretraining‑SFT‑v1、AM‑DeepSeek‑R1‑0528‑Distilled 数据集上微调；评估则选取数学推理任务 GSM8K、MATH‑500 以及代码生成任务 HumanEval、MBPP（以及 HumanEval+、MBPP+ 的扩展版本）。

**📈 对比分析**

与 Soft Mask、EvoToken、LLaDA‑Instruct 等基线在不同扩散预算（1/1、1/2、1/4、1/8）下进行对比；ME‑DLM 在 1/8 预算下平均提升 13.3 分，HumanEval 提升 11.6 分，GSM8K 提升 33.6 分；整体上在所有预算和任务上均优于 Stage‑2 以及先前的并行生成方法，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：①仍需要额外的编辑步骤和额外的训练；②编辑操作仅能局部修正，可能对全局结构变化能力有限；③评估集中在少数推理与代码任务，缺乏对更大规模或更复杂文本的验证；④未深入分析推理时的延迟与计算成本。

---

## 143. G-Zero: Self-Play for Open-Ended Generation from Zero Data

**arXiv ID:** 2605.09959 | [PDF](https://arxiv.org/pdf/2605.09959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 144. Multi-Tier Labeling and Physics-Informed Learning for Orbital Anomaly Detection at Scale

**arXiv ID:** 2605.09790 | [PDF](https://arxiv.org/pdf/2605.09790v1)

**作者:** Yong Fu `[一作]` `[通讯]` (Substratum Labs, Inc.), Yong Fu (Substratum Labs, Inc.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过物理规则、IMM-UKF滤波器与补充元素校准三层弱监督级联，对232M条TLE记录进行标签生成，并训练Physics Inspired Orbital Transformer（PIOT）实现高召回率的异常行为三分辨率检测。

**💡 创新点**

创新点在于构建多级弱监督标签级联，显著提升了异常候选量（42.6×），并在Transformer中引入冻结物理预测与创新评分，证明显式时间编码对衰变召回提升107%。

**🔧 技术方法**

使用的技术包括：物理规则集、交互式多模型无迹卡尔曼滤波器（IMM-UKF）、补充元素校准、基于Transformer的Physics Inspired Orbital Transformer（PIOT），以及两阶段训练课程与焦点损失。

**📊 数据集**

数据集为Space-Track公开的历史TLE记录，总计232.4M条，经过滑动窗口切片得到8.6M个长度50的序列，共430M个标记时间步。

**📈 对比分析**

与仅基于规则的基线相比，PIOT在测试集上将机动召回从7%提升至55.4%，衰变召回从2.5%提升至62.8%，整体精度维持在约75%，表明多级标签与物理信息显著提升模型性能。

**⚠️ 局限性**

局限性包括标签非真值来源（规则与滤波器自带偏差）、窗口级拆分而非卫星级holdout、IMM-UKF覆盖范围有限、单步预测限制以及对实时多步预测与不确定性估计的需求。

---

## 145. "Training robust watermarking model may hurt authentication!'' Exploring and Mitigating the Identity Leakage in Robust Watermarking

**arXiv ID:** 2605.09646 | [PDF](https://arxiv.org/pdf/2605.09646v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 146. PHAGE: Patent Heterogeneous Attention-Guided Graph Encoder for Representation Learning

**arXiv ID:** 2605.10073 | [PDF](https://arxiv.org/pdf/2605.10073v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 147. Tensor Product Representation Probes Reveal Shared Structure Across Linear Directions

**arXiv ID:** 2605.09967 | [PDF](https://arxiv.org/pdf/2605.09967v1)

**作者:** Andrew Lee `[一作]` (Harvard University), Martin Wattenberg `[通讯]` (Harvard University)

**通讯引用:** 32463 | [OpenAlex ID](https://openalex.org/A5039276358)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在OthelloGPT上探究线性方向与结构化表示的关系，利用张量乘积表示（TPR）探针对模型的棋盘状态进行因子化；

**💡 创新点**

首次证明线性方向可被压缩为具有角色-填充绑定的结构化张量，并在权重中发现与棋盘几何相符的签名；

**🔧 技术方法**

使用TPR探针、线性探针、因果干预、低秩SVD与余弦相似度等技术；

**📊 数据集**

基于约2000万条Othello棋局转录的训练数据；

**📈 对比分析**

通过与线性探针、低秩SVD、随机编码和随机生成对照实验比较，TPR探针在保持99%准确率的同时参数更少、能重构线性方向并捕捉棋盘几何；

**⚠️ 局限性**

需要先验知道所关注的结构，对未知域通用性有限，未证明模型内部实际使用TPR。

---

## 148. Novel GPU Boruta algorithms for feature selection from high-dimensional data

**arXiv ID:** 2605.09950 | [PDF](https://arxiv.org/pdf/2605.09950v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 149. The two clocks and the innovation window: When and how generative models learn rules

**arXiv ID:** 2605.10019 | [PDF](https://arxiv.org/pdf/2605.10019v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 150. A Scalable and Unified Framework to Weighted Rank Aggregation

**arXiv ID:** 2605.09653 | [PDF](https://arxiv.org/pdf/2605.09653v1)

**作者:** Amir Carmel `[一作]` (Weizmann Institute of Science), Tien-Long Nguyen `[通讯]` (Pennsylvania State University)

**通讯引用:** 289 | [OpenAlex ID](https://openalex.org/A5060063567)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种统一框架，用于在多种距离度量（Ulam、Spearman脚印、Hamming、Kendall‑tau及其加权版本）下求解1‑中位数（即最佳一致排序），并在Massively Parallel Computation（MPC）模型中实现常数轮次、近线性总内存的近似算法。

**💡 创新点**

核心创新在于发现只需关注输入排序的一个常数大小子集（3或5个），其局部1‑中位数即可逼近全局最优；该“局部‑全局”性质适用于所有上述距离；基于此框架，提出了Ulam度量下的1.968‑近似算法并推广到加权设置；同时在MPC上实现了对所有四种距离的2‑近似突破。

**🔧 技术方法**

技术上采用：①随机抽样与Indyk式候选集合构造；②针对每种度量设计局部聚合子算法（如多项式时间的块构造、投票图 + 反馈弧/顶点集近似、位置中值与排序映射等）；③MPC实现细化，包括窗口分解、块合成动态规划、虚拟元素填充、全局排名同步等。

**📊 数据集**

论文主要在理论上提出算法，实验上使用人工合成的随机排列和大规模模拟数据（如n上千到百万级），以验证内存与时间复杂度。

**📈 对比分析**

与传统单机基于最小成本匹配或DP的2‑近似方法相比，本文的MPC算法在常数轮次内完成，局部内存仅为n^1−ϵ，且总内存接近线性；在加权Ulam度量上实现1.968‑近似，打破此前1.999‑近似。

**⚠️ 局限性**

局限性包括：①需要随机抽样并假设输入规模足够大；②对加权Ulam的实现仍采用高次多项式时间（n^17）近似反馈顶点集；③在实际分布式系统中同步与负载均衡的细节未在实验中充分验证。

---

## 151. Arcane: An Assertion Reduction Framework through Semantic Clustering and MCTS-Guided Rule Exploring

**arXiv ID:** 2605.10107 | [PDF](https://arxiv.org/pdf/2605.10107v1)

**作者:** Hongqin Lyu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Huawei Li `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 36898 | [OpenAlex ID](https://openalex.org/A5100346092)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Arcane 框架，实现高效的断言（Assertion）冗余消除，显著降低验证成本。

**💡 创新点**

创新点包括：① 两层语义聚类（BERT + Lasso 行为相似度）精准分组；② 将判定规则映射为 MCTS 搜索动作，保证语义等价的同时最大化冗余剔除。

**🔧 技术方法**

采用 BERT 进行自然语言描述向量化；Lasso 通过 Büchi 自动机采样 lasso 运行评估行为相似度；DBSCAN 做聚类；五条语义保持的简化规则；MCTS（UCT）搜索规则序列。

**📊 数据集**

使用 AssertionBench（112 个硬件设计），包含 HARM 与 LLM 生成的断言集合。

**📈 对比分析**

与原始断言集比较：在 112 份基准上平均可压缩 68%–76% 的断言数；正式覆盖率（PC）和突变检测率（ER）保持不变；仿真时间提升 2.6×–6.1×，相较无压缩或传统聚类方法表现优异。

**⚠️ 局限性**

局限性：仅适用于可转化为 LTL 的断言；lasso 采样与自动机分析仍有一定计算开销；规则集有限，可能无法覆盖所有冗余模式。

---

## 152. Adversarial Attacks Against MLLMs via Progressive Resolution Processing and Adaptive Feature Alignment

**arXiv ID:** 2605.09902 | [PDF](https://arxiv.org/pdf/2605.09902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 153. PolicyCache-SDN: Hierarchical Intra-Path Learning for Adaptive SDN Traffic Control

**arXiv ID:** 2605.09473 | [PDF](https://arxiv.org/pdf/2605.09473v1)

**作者:** Wenyang Jia `[一作]` (Peking University), Kai Lei `[通讯]` (Peking University)

**通讯引用:** 253718 | [OpenAlex ID](https://openalex.org/A5071127149)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种分层SDN流量控制框架（PolicyCache‑SDN），让边缘代理在控制器设定的政策包（policy envelope）约束下进行在线局部学习与即时动作执行。

**💡 创新点**

创新点在于将端口级在线学习的局部性原则扩展到路径聚合，通过控制器编译的政策包限制探索空间、记录行动日志、实现冲突仲裁，使多代理在共享瓶颈上安全协同；同时将学习模型（Hoeffding Adaptive Tree）与SDN计量、队列、重路由等动作结合。

**🔧 技术方法**

使用Ryu/OpenFlow/OVS、gRPC通信、Python实现的Hoeffding Adaptive Tree、ADWIN漂移检测、sFlow/INT数据收集、ECN/队列统计、GRE/10G虚拟链路模拟。

**📊 数据集**

在1,024节点的AWS云实验平台上使用模拟的Clos拓扑，生成elephant-heavy、mice-heavy和混合实时流量三种工作负载，利用iperf3、定制生成器和时间戳探测测量FCT、延迟等指标。

**📈 对比分析**

与九种基线（静态ECMP、中心化TE、阈值重路由、固定计量、WFQ、Aurora‑SDN、LetFlow、Presto、端点层TCP控制）对比，PolicyCache‑SDN平均核心链路利用率提高35.5%、elephant P99 FCT降低40.3%、SLA违约率下降62.6%，每个代理CPU<2.1%、内存<13.4 MB，收敛时间<400 ms。

**⚠️ 局限性**

局限性包括：实验在软件SDN/虚拟链路环境下，未验证硬件Fabric（如P4/SmartNIC）实时反馈；仅评估单域、单AS场景；离线RL基线固定，未与更强在线MARL/多代理RL对比；重路由与队列优先级的离散动作仍可能导致跨租户冲突，需要更深入的多代理分析。

---

## 154. EFGCL: Learning Dynamic Motion through Spotting-Inspired External Force Guided Curriculum Learning

**arXiv ID:** 2605.10063 | [PDF](https://arxiv.org/pdf/2605.10063v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 155. PixelFlowCast: Latent-Free Precipitation Nowcasting via Pixel Mean Flows

**arXiv ID:** 2605.10046 | [PDF](https://arxiv.org/pdf/2605.10046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 156. Causal Parametric Drift Simulation: A Digital Twin Framework for Classifier Robustness Evaluation

**arXiv ID:** 2605.09663 | [PDF](https://arxiv.org/pdf/2605.09663v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 157. Cross-Family Universality of Behavioral Axes via Anchor-Projected Representations

**arXiv ID:** 2605.09875 | [PDF](https://arxiv.org/pdf/2605.09875v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 158. Free Energy Manifold: Score-Based Inference for Hybrid Bayesian Networks

**arXiv ID:** 2605.09839 | [PDF](https://arxiv.org/pdf/2605.09839v1)

**作者:** Cheol Young Park `[一作]` (ATOS Co., Ltd.), Shou Matsumoto `[通讯]` (George Mason University)

**通讯引用:** 240 | [OpenAlex ID](https://openalex.org/A5064784800)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于分数匹配的条件能量网络（Free Energy Manifold，FEM），可同时完成离散父变量的后验推理、连续子变量的生成采样、连续叶子可组合推理，并通过原型嵌入处理高基数父变量。

**💡 创新点**

创新点在于：① 将分数匹配能量网络重构为可组合推理因子，而非单纯的条件密度估计；② 通过学习离散父变量原型实现稀疏高基数父变量的泛化；③ 发现并通过 valley 正则化消除多模态类的模式桥（mode‑bridge）后验失真问题。

**🔧 技术方法**

采用分数匹配（DSM）训练的 NCSN‑style MLP 能量网络；使用交叉熵锚点、原型惩罚、valley 正则化等多任务损失；利用 sinusoidal 噪声嵌入和 energy‑addition 进行多叶子组合；采用 annealed Langevin 进行连续采样。

**📊 数据集**

在合成的高维多模态混合分布（D=5,10 等）上进行严格的后验 KL 对比；在 UCI 标准表格数据（Iris、Wine、Breast Cancer）上评估 NLL 与准确率；在 MNIST 上做闭合世界分类对比。

**📈 对比分析**

与 CLG、KDE、Histogram、条件 EBM（CEBM）、MDN、单叶子 MLP 等基线对比；在合成实验中 FEM 以 60–172 倍低 KL、300 倍低模式桥 KL、122 倍低 MDN KL；在 Breast Cancer 上 NLL 2.7 倍优于 CLG；在多叶子组合实验中 FEM 在 7 种证据模式下比 MLP 高 5.3 倍；在 MNIST 上表现略逊于 MLP。

**⚠️ 局限性**

局限性包括：① 仅对多模态、分布不单峰的任务有效，单峰类时不需要 valley 正则化；② 高维（D≥10）下模式桥残差仍存在，valley 强度 λ 需要手动校准；③ 对离散父基数非常大的情形需更多训练样本才能充分泛化；④ 对闭合世界分类任务无优势；⑤ 树莓种子变异导致偶尔的模式桥残留；⑥ 仍未覆盖 ACE、条件扩散等更灵活的条件能量/密度估计基线。

---

## 159. CFSPMNet: Cross-subject Fourier-guided Spatial-Patch Mamba Network for EEG Motor Imagery Decoding in Stroke Patients

**arXiv ID:** 2605.10111 | [PDF](https://arxiv.org/pdf/2605.10111v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 160. Trajectory Supervision for Continual Tool-Use Learning in LLMs

**arXiv ID:** 2605.09734 | [PDF](https://arxiv.org/pdf/2605.09734v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 161. Geometry of Rényi Entropy on the Majorization Lattice

**arXiv ID:** 2605.09655 | [PDF](https://arxiv.org/pdf/2605.09655v1)

**作者:** Anuj Kumar Yadav `[一作]` (École Polytechnique Fédérale de Lausanne), Yanina Y. Shkel `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 264 | [OpenAlex ID](https://openalex.org/A5052329970)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了瑞尼熵在主序格（majorization lattice）上的结构性质，证明其在该格上满足子可加性与超模性，并基于此构造了一族以瑞尼熵为参数的距离度量，推导了其与Theil指数的关系；

**💡 创新点**

首次将comonotone coupling与独立coupling之间的主序关系与瑞尼熵的Schur‑concavity结合，得到瑞尼熵的子可加性与超模性，并给出对α∈{0}∪[1,∞]的完整性结果；

**🔧 技术方法**

利用Lorenz曲线几何、主序格的极大下界与极小上界构造、聚合与重排理论、以及凸/凹函数上的Jensen不等式等数学工具；

**📊 数据集**

无；

**📈 对比分析**

未给出数值实验，仅通过理论证明说明d_α(·,·)构成度量并能量化不平等；

**⚠️ 局限性**

超模性仅在两元素时成立，对>2元素不适用；α∈(0,1)时瑞尼熵既非子可加也非超模；此外，实际应用中需进一步验证距离度量的统计性能和计算效率。

---

## 162. Benchmarking Transformer and xLSTM for Time-Series Forecasting of Heat Consumption

**arXiv ID:** 2605.09722 | [PDF](https://arxiv.org/pdf/2605.09722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 163. The Vote-Left Equilibrium: A Deterministic Coordination Strategy for the Faithful in The Traitors

**arXiv ID:** 2605.10233 | [PDF](https://arxiv.org/pdf/2605.10233v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 164. Formal Verification of Imperative First-Class Functions in Move

**arXiv ID:** 2605.10007 | [PDF](https://arxiv.org/pdf/2605.10007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 165. Is DRL-based MAC Ready for Underwater Acoustic Networks? Exploring Its Practicality in Real Field Experiments

**arXiv ID:** 2605.10144 | [PDF](https://arxiv.org/pdf/2605.10144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 166. When Sounds Hurt and Voices Aren't Heard: An Experience Report on Misophonia, Sensory Trauma, and Trauma-Informed Design

**arXiv ID:** 2605.09796 | [PDF](https://arxiv.org/pdf/2605.09796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 167. Position: AI Security Policy Should Target Systems, Not Models

**arXiv ID:** 2605.09504 | [PDF](https://arxiv.org/pdf/2605.09504v1)

**作者:** Michael A. Riegler `[一作]` (SimulaMet and OsloMet), Inga Strümke `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用1.2B参数模型组成的协同“swarm”在本地消费级硬件上实现对前沿模型的系统性越狱与软件漏洞挖掘，展示了系统级架构对攻击能力的决定性作用。

**💡 创新点**

创新点在于提出可公开复现的多智能体协作框架，证明仅凭小模型与系统搭建即可复制前沿模型的攻击与漏洞发现能力。

**🔧 技术方法**

核心技术包括：多模型协同进化搜索、共享内存与策略演化、正则表达式与手工构造的漏洞种子、AddressSanitizer动态检查、LLM-as-judge评估及手工验证。

**📊 数据集**

实验数据来源为OpenAI的GPT‑4o与Anthropic的Claude Sonnet‑4两大前沿模型，以及自研的含9个CWE的SwarmApp代码基准。

**📈 对比分析**

比较方法：对GPT‑4o实现45.8%有效伤害率、49起高危输出；对Claude实现0%有效伤害率；在漏洞发现上，配套框架实现9/9召回，去除框架仅为0/9（崩溃验证）/2/9（引用验证）。

**⚠️ 局限性**

局限性包括：使用的漏洞目标为人工植入的合成程序，缺乏对真实大型代码库的验证；单次实验与随机种子导致结果波动；有效伤害率需人工确认，难以大规模自动化；框架的通用性与对不同模型、硬件的适配尚待进一步评估。

---

## 168. Thermal-Det: Language-Guided Cross-Modal Distillation for Open-Vocabulary Thermal Object Detection

**arXiv ID:** 2605.10130 | [PDF](https://arxiv.org/pdf/2605.10130v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 169. Towards Generalist Game Players: An Investigation of Foundation Models in the Game Multiverse

**arXiv ID:** 2605.09965 | [PDF](https://arxiv.org/pdf/2605.09965v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 170. TSNBench: Benchmarking LLM Proficiency in Time-Sensitive Networking

**arXiv ID:** 2605.09481 | [PDF](https://arxiv.org/pdf/2605.09481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 171. Usability as a Weapon: Attacking the Safety of LLM-Based Code Generation via Usability Requirements

**arXiv ID:** 2605.10133 | [PDF](https://arxiv.org/pdf/2605.10133v1)

**作者:** Yue Li `[一作]` (Nanjing University), Sheng Zhong `[通讯]` (Nanjing University)

**通讯引用:** 471484 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了利用可用性压力攻击LLM代码生成器，诱导其生成功能正确但不安全的代码。

**💡 创新点**

创新点在于提出“可用性压力攻击”概念，并构建自动化框架来合成并验证攻击。

**🔧 技术方法**

使用大语言模型（如GPT-5.1、GPT-5.2、DeepSeek-V3.2、Gemini-3）作为目标生成器，配合Analyzer、Judge等LLM辅助模块以及动态payload生成技术。

**📊 数据集**

数据集为从CWEval和SeCodePLT抽取的75个安全基准场景，涵盖25个CWE，涵盖Python、C、JavaScript三种语言。

**📈 对比分析**

通过对比攻击前后的正确率（CRbaseline、CRattacked）和攻击成功率（ASR），发现对不同攻击类型（功能、实现、权衡）和模型的攻击效果差异明显，Trade‑off压力下ASR可达98%，整体攻击成功率高于传统方法。

**⚠️ 局限性**

局限性包括对特定模型的依赖、对已有安全测试用例的局限、未考虑模型更新后的鲁棒性以及攻击生成过程的可解释性不足。

---

## 172. Frequency Adapter with SAM for Generalized Medical Image Segmentation

**arXiv ID:** 2605.09925 | [PDF](https://arxiv.org/pdf/2605.09925v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 173. Unveiling High-Probability Generalization in Decentralized SGD

**arXiv ID:** 2605.10205 | [PDF](https://arxiv.org/pdf/2605.10205v1)

**作者:** Jiahuan Wang `[一作]` (National University of Defense Technology), Tao Sun `[通讯]` (National University of Defense Technology)

**通讯引用:** 11581 | [OpenAlex ID](https://openalex.org/A5044883230)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对去中心化随机梯度下降（D‑SGD）提出了高概率泛化理论，并给出了凸、强凸以及非凸问题的稳定性与泛化误差分析。

**💡 创新点**

创新点在于引入点对点均匀稳定性（pointwise uniform stability）作为弱稳定性工具，实现了高概率下最优的 𝒪(1/√(mn)) 泛化率，首次将该理论扩展到去中心化环境，并同时给出了局部模型在时间变网络下的泛化界。

**🔧 技术方法**

主要技术包括：点对点均匀稳定性框架、马尔可夫差分序列分析、梯度占优（gradient‑dominance）条件、去中心化通信图（gossip matrix）与其谱特征、以及对凸/强凸/非凸三种情形的统一处理。

**📊 数据集**

该工作为纯理论分析，未使用具体数据集，主要通过数学证明与理论极限来验证结论。

**📈 对比分析**

与已有的期望泛化界（如 𝒪(1/δ√(mn))）相比，本文的高概率界在大多数情形下实现了更紧的 𝒪(1/√(mn)) 速率，且在局部模型和非凸梯度占优场景中给出了更完整的理论表现。

**⚠️ 局限性**

局限性包括：要求损失函数满足 Lipschitz、β‑光滑以及凸/强凸或梯度占优条件；对非光滑或无 Lipschitz 的情况缺乏分析；异步更新未被正式考虑；且在极端网络拓扑或极端噪声环境下的鲁棒性尚待进一步研究。

---

## 174. Evidence-based Decision Modeling for Synthetic Face Detection with Uncertainty-driven Active Learning

**arXiv ID:** 2605.09935 | [PDF](https://arxiv.org/pdf/2605.09935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 175. FinMoji: A Framework for Emoji-driven Sentiment Analysis in Financial Social Media

**arXiv ID:** 2605.09469 | [PDF](https://arxiv.org/pdf/2605.09469v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 176. Lakestream: A Consistent and Brokerless Data Plane for Large Foundation Model Training

**arXiv ID:** 2605.09994 | [PDF](https://arxiv.org/pdf/2605.09994v1)

**作者:** Ting Sun `[一作]` (Lionrock AI Lab, China Merchants Group), Zejian Xie `[通讯]` (Lionrock AI Lab, China Merchants Group)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Lakestream，一套无代理、基于对象存储的训练数据平面，核心是 Transactional Global Batch（TGB）抽象和 Decentralized Adaptive Commit（DAC）协议，实现了训练阶段批次的原子可见性、全局顺序和与检查点一致的生命周期管理。

**💡 创新点**

创新点包括：①将批次视作持久实体的 TGB，结合版本化 manifest 保障原子性与一致性；②DAC 在无协调器的前提下动态调整提交间隔，维持高吞吐；③将生产者状态内联到 manifest，提供端到端 exactly‑once 语义；④基于检查点水位的存储回收，减少空间占用。

**🔧 技术方法**

使用了对象存储（BOS/S3/GCS/Blob）、版本化 manifest（借鉴 Delta Lake/Iceberg）、乐观并发控制、Rust+Python SDK、基于 LANCE 的多对象批次布局、以及自适应提交算法 DAC。

**📊 数据集**

实验数据集涵盖多模态预训练与细调：GR00T、HoloAssist（视频 SFT）、BEHAVIOR‑1K（机器人多摄像头演示），以及 LeRobot 视频样本、OpenCLIP 文本样本等。

**📈 对比分析**

通过与专家优化的 colocated pipeline 以及严格 TGB 语义的 Kafka 进行对比，Lakestream 在三类工作负载上提升了 2.68–7.73 倍的端到端吞吐，平均 step 延迟下降至 172–367 ms（colocated 457–4113 ms），并在生产者扩容时保持线性吞吐，Kafka 在大批量 TGB 时经常失败。

**⚠️ 局限性**

局限性包括：仍需单独的生产者节点集群；manifest 写入开销在高频小批次场景下占比高；对象存储的网络延迟和一致性模型对极低延迟训练有潜在影响；对极大 TGB（>10 MB）在 Kafka 端仍面临消息大小限制。

---

## 177. Retrieve-then-Steer: Online Success Memory for Test-Time Adaptation of Generative VLAs

**arXiv ID:** 2605.10094 | [PDF](https://arxiv.org/pdf/2605.10094v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 178. ConFit v3: Improving Resume-Job Matching with LLM-based Re-Ranking

**arXiv ID:** 2605.09760 | [PDF](https://arxiv.org/pdf/2605.09760v1)

**作者:** Xiao Yu `[一作]` (Columbia University), Zhou Yu `[通讯]` (Columbia University)

**通讯引用:** 323638 | [OpenAlex ID](https://openalex.org/A5072248970)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对招聘场景中的简历-职位匹配，系统评估并改进了基于大语言模型（LLM）的重排序器，最终训练得到 ConFit v3 并在两个真实数据集上取得显著提升。

**💡 创新点**

创新点在于：① 引入多轮滑动窗口重排序以充分利用长文本信息；② 使用列表级 RL 目标 ReaRank 与“移除难样本”策略显著提升学习效果；③ 在 RL 前进行强模型（Claude Sonnet 4.5）SFT 蒸馏，为 LLM 提供更优初始化。

**🔧 技术方法**

核心技术包括 Qwen3-8B/32B LLM、SFT 蒸馏、Group Relative Policy Optimization（GRPO）RL、滑动窗口多轮推理、硬样本移除和列表级损失函数。

**📊 数据集**

使用的公开数据集为 Alibaba 2019 年职位-简历匹配竞赛（AliYun）和公司内部招聘数据集（Recruiting），均包含千字级长文本和稀疏噪声标签。

**📈 对比分析**

与基线检索模型、几种强大开闭源 LLM（GPT‑5、Claude Opus 4.5、Qwen3‑235B）以及现有 RL 训练方法（Rank‑R1、ReaRank）对比，ConFit v3 在 nDCG@10 和 Recall@10 上均实现 5–7% 的绝对提升，尤其在 Recruiting 数据集上提升 7.81%。

**⚠️ 局限性**

主要限制包括：① 训练依赖昂贵的 LLM 资源和高质量标签；② 简历与职位数据噪声高，难以完全消除误标；③ 仅在两大数据集上验证，尚未展示跨行业或多语言场景的泛化能力。

---

## 179. SkillRAE: Agent Skill-Based Context Compilation for Retrieval-Augmented Execution

**arXiv ID:** 2605.10114 | [PDF](https://arxiv.org/pdf/2605.10114v1)

**作者:** Xiangcheng Meng `[一作]` (Chinese University of Hong Kong), Yixiang Fang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2809 | [OpenAlex ID](https://openalex.org/A5043494334)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了两阶段 Retrieval-Augmented Execution (RAE) 方法，先构建多层技能图（社区–技能–子单元），再进行技能检索并通过上下文编译生成紧凑、可直接使用的任务特定上下文。

**💡 创新点**

创新点在于首次将检索到的技能与关键子单元聚合成可执行的上下文，通过多层技能图实现子单元层面的检索与救援式补救，并提供任务特定的使用指导。

**🔧 技术方法**

技术包括多层技能图构建、向上/向下检索（社区匹配与子单元投影）、子单元过滤、附件匹配与指导生成、上下文压缩；使用SentenceTransformer等嵌入、加权评分和基于Codex CLI+GPT‑5.2、Gemini等后端执行器。

**📊 数据集**

使用 SkillsBench（87 个任务）和 AgentSkillOS（30 个任务）两大公开技能代理基准进行评估，构建对应的技能图。

**📈 对比分析**

与手工 curations、Vanilla 检索、LLM 检索、SkillRouter、AgentSkillOS 等基线对比，SkillsBench 平均奖励提升至 29.26%（比基线高 11.7%），AgentSkillOS 分数提升至 84.59%；ablation 结果表明顶层检索与上下文编译对性能贡献最大。

**⚠️ 局限性**

限制：方法主要适用于含有显式过程文本、文件/输出规范和约束的技能库；对隐藏依赖的工具、无文档代码或运行时状态的情况效果有限；并非完整规划器，无法保证执行时错误恢复。

---

## 180. Fashion Florence: Fine-Tuning Florence-2 for Structured Fashion Attribute Extraction

**arXiv ID:** 2605.09827 | [PDF](https://arxiv.org/pdf/2605.09827v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 181. QueST: Persistent Queries as Semantic Monitors for Drift Suppression in Long-Horizon Tracking

**arXiv ID:** 2605.09513 | [PDF](https://arxiv.org/pdf/2605.09513v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 182. Frequency Matching in Spiking Neural Networks for mmWave Sensing

**arXiv ID:** 2605.09983 | [PDF](https://arxiv.org/pdf/2605.09983v1)

**作者:** Di Yu `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**通讯引用:** 9413 | [OpenAlex ID](https://openalex.org/A5055284175)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并验证了脉冲神经网络（SNN）在毫米波感知任务中的频域匹配原理，并提出了基于膜衰减因子的频率匹配准则。

**💡 创新点**

首次从机制‑数据对齐角度，将LIF神经元的低通特性与毫米波信号的判别频谱相匹配，给出了无须网格搜索即可确定的膜衰减因子选择方法。

**🔧 技术方法**

采用离散傅里叶变换频域分析、LIF模型的IIR低通滤波器理论、频率匹配得分（FMS）评估、LeNet风格的SNN训练以及能耗与精度的统一评估。

**📊 数据集**

使用四个公开毫米波数据集：AOPHand、mmFiT、Pantomime 和 MMActivity。

**📈 对比分析**

在统一的训练与评估协议下，将SNN与多种ANN基线（MLP、LeNet、VGG、ResNet、RNN、GRU、LSTM、BiLSTM、CNN‑GRU、ViT）进行对比，SNN平均提升约6.22%的识别准确率，同时理论能耗平均下降约3.64倍。

**⚠️ 局限性**

研究主要验证了单一LeNet风格SNN，未全面探讨更深或更宽模型的适用性；对硬件延迟的差异主要受系统实现影响，未来需要在更丰富的硬件平台上验证；频率匹配准则对不同频谱分布的数据集可能需要进一步调整。

---

## 183. TrajDLM: Topology-Aware Block Diffusion Language Model for Trajectory Generation

**arXiv ID:** 2605.10020 | [PDF](https://arxiv.org/pdf/2605.10020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 184. The Gordian Knot for VLMs: Diagrammatic Knot Reasoning as a Hard Benchmark

**arXiv ID:** 2605.09900 | [PDF](https://arxiv.org/pdf/2605.09900v1)

**作者:** Hao Liu `[一作]` (New York University), Jicheng Liu `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了 KnotBench 基准，通过对 858k 张来自 1,951 个素结原型的结图像进行 14 项任务评测，揭示了视觉-语言模型在从感知到操作的过程中存在的巨大差距。

**💡 创新点**

创新点在于：①首次构建大规模、基于数学真值的结图像语料库；②设计 14 个多维度任务网格，细化感知-操作裂隙；③将经典结理论的可判定等价判据与模型输出直接对比，提供客观、可复现的评估。

**🔧 技术方法**

技术主要包括：①通过随机 Walk 的 Reidemeister 移动生成结图像；②使用 Regina 的 canonical signature 作为等价性判别器；③采用思考模式（chain-of-thought）与 64K 输出 token 限制相结合的评测框架。

**📊 数据集**

数据集为 KnotBench：858,318 张 PNG 结图像，涵盖 1,951 个 3–19 交叉数的素结原型，配有 PD 代码和 DT 代码等符号表征。

**📈 对比分析**

比较方法：在 56（任务×模型）组合上对 Claude Opus 4.7、GPT‑5（含/不含思考模式）进行 64K token 预算下的零样本评测；结果显示多数任务随机水平，且模型在图像输入时表现差于符号输入，整体性能仅略优于随机，思考模式对符号任务提升显著但对图像任务效果有限。

**⚠️ 局限性**

局限性包括：①评测样本量相对有限（约 2,000 条），难以覆盖所有结复杂度；②仅测试两款闭源 VLM，缺乏开源与多厂商对比；③缺乏人类基准，无法直接衡量人类与模型的差距；④基准侧重结图像，难以推广至更广泛的图形推理任务。

---

## 185. Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration

**arXiv ID:** 2605.10203 | [PDF](https://arxiv.org/pdf/2605.10203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 186. Urban-ImageNet: A Large-Scale Multi-Modal Dataset and Evaluation Framework for Urban Space Perception

**arXiv ID:** 2605.09936 | [PDF](https://arxiv.org/pdf/2605.09936v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 187. How Should LLMs Listen While Speaking? A Study of User-Stream Routing in Full-Duplex Spoken Dialogue

**arXiv ID:** 2605.10199 | [PDF](https://arxiv.org/pdf/2605.10199v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 188. Network-Efficient World Model Token Streaming

**arXiv ID:** 2605.09886 | [PDF](https://arxiv.org/pdf/2605.09886v1)

**作者:** Shatadal Mishra `[一作]` (Toyota Motor North America), Nejib Ammar `[通讯]` (Toyota Motor North America)

**通讯引用:** 666 | [OpenAlex ID](https://openalex.org/A5084763315)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文研究了在车辆网络约束下，将驾驶世界模型的离散Token状态通过自适应keyframe-δ协议高效流式传输，以实现低比特率下的状态同步。

**💡 创新点**

创新点在于引入基于代码簿嵌入空间余弦距离的增量优先级排序以及基于汉明漂移阈值的自适应keyframe触发，完全无需标签且能在有限字节预算下动态控制更新量。

**🔧 技术方法**

采用stride-16 VQ-U-Net tokenizer将288×512帧压缩为18×32的Token网格，利用嵌入向量余弦距离评估变更大小，并通过简单的位掩码与固定头部实现网络封包。

**📊 数据集**

实验使用NVIDIA驾驶数据集（前置宽摄像头）约20 s、30 fps的视频，按10 Hz采样并分割为≈200帧的Token序列。

**📈 对比分析**

与固定间隔的periodic keyframe baseline 在相同比特率下对比，adaptive方案在200 bytes（约0.024 Mb/s）时动态嵌入失真降低7.2%，在400 bytes时降低4.8%，并在10% delta丢包时保持更低失真；同时下游Token预测的动态位置困惑度下降6.3%。

**⚠️ 局限性**

主要局限包括：假设keyframe可靠传输、delta丢包为独立；未考虑更复杂的网络延迟/突发丢包；字节预算采用保守固定更新开销，未探索更紧凑的位打包；以及下游预测器仅使用局部时序信息，未充分利用空间上下文。

---

## 189. Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs

**arXiv ID:** 2605.09922 | [PDF](https://arxiv.org/pdf/2605.09922v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 190. Developing a foundation model for high-resolution remote sensing data of the Netherlands

**arXiv ID:** 2605.10184 | [PDF](https://arxiv.org/pdf/2605.10184v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 191. LegalCiteBench: Evaluating Citation Reliability in Legal Language Models

**arXiv ID:** 2605.10186 | [PDF](https://arxiv.org/pdf/2605.10186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 192. Fairness of Explanations in Artificial Intelligence (AI): A Unifying Framework, Axioms, and Future Direction toward Responsible AI

**arXiv ID:** 2605.09852 | [PDF](https://arxiv.org/pdf/2605.09852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 193. MiXR: Harvesting and Recomposing Geometry from Real-World Objects for In-Situ 3D Design

**arXiv ID:** 2605.09620 | [PDF](https://arxiv.org/pdf/2605.09620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 194. Optimal Inapproximability of Generalized Linear Equations over a Finite Group

**arXiv ID:** 2605.10010 | [PDF](https://arxiv.org/pdf/2605.10010v1)

**作者:** Amey Bhangale `[一作]` (University of California), Yezhou Zhang `[通讯]` (University of California)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在有限群 G 上的约束满足问题 Max-Ek-LIN_S(G)，给出在可满足实例上可获得 |S|/|H_S| 的逼近算法，并证明对某些 S 在假设 P≠NP 的前提下该逼近比率是最优的。

**💡 创新点**

创新点包括：① 将线性约束泛化为任意子集 S 的判定式，并引入最小正规子群 H_S 使得约束在商群 Q=G/H_S 上变为线性方程；② 利用 Gaussian elimination 在 Q 上求解后随机取余数完成赋值；③ 通过新的迪克特测试与 Fourier 解析技术，证明在非可满足实例上仍存在逼近上限 |S|/|H_S|+ε；④ 这为少数已知在几乎可满足实例上逼近抵抗但在可满足实例上有非平凡逼近的 CSP 预测子提供了新例子。

**🔧 技术方法**

核心技术包括：表示论与非交换群的 Fourier 解析、群商与正规子群的构造、Gaussian elimination、随机化与条件期望法、迪克特测试与影响度分析、以及从 Label Cover 的多层结构构造的多项式时间约简。

**📊 数据集**

该工作为理论性研究，未使用具体数据集，而是以抽象的有限群与集合 S 为输入。

**📈 对比分析**

与最基本的随机赋值算法相比，随机算法的成功率为 |S|/|G|，而本算法在商群 Q 的维度更小（|H_S| ≤ |G|）时能取得更高的 |S|/|H_S| 成功率；在可满足实例上可达到该比率，并且证明在特定 S 情况下无法进一步改进，证明了算法的最优性。

**⚠️ 局限性**

局限性包括：1）仅在可满足实例上保证 |S|/|H_S| 的逼近；2）硬性证明要求 P≠NP，并且需要 S^-1 S 生成 H_S；3）对非可满足实例的逼近上限仍为随机赋值的 |S|/|G|；4）目前仅针对 Max-E3-LIN_S(G)（k=3）给出了完整证明，对更高 k 的推广仍需进一步工作。

---

## 195. Spatial-Frequency Gated Swin Transformer for Remote Sensing Single-Image Super-Resolution

**arXiv ID:** 2605.09687 | [PDF](https://arxiv.org/pdf/2605.09687v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 196. Route by State, Recover from Trace: STAR with Failure-Aware Markov Routing for Multi-Agent Spatiotemporal Reasoning

**arXiv ID:** 2605.10057 | [PDF](https://arxiv.org/pdf/2605.10057v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 197. Recovery Algorithms for Linear Batch Codes

**arXiv ID:** 2605.09748 | [PDF](https://arxiv.org/pdf/2605.09748v1)

**作者:** Baran Düzgün `[一作]` (Hacettepe University), Vladislav Taranchuk `[通讯]` (Ghent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文研究并系统化了线性批量码（batch code）的多种恢复算法，包括在线（online）、异步（asynchronous）和强（strong）三种类型，并探讨了它们之间的层级关系。作者进一步引入了一类新的 Almost Affinely Disjoint（L‑AAD*）子空间族，证明其可直接构造出 (m,L)‑强异步批量码；同时研究了仅使用简单恢复集的系统化二进制批量码，并给出了对应二部图满足的必要且充分条件，从而得到 (m,L)‑强批量码的图论构造。

**💡 创新点**

创新点主要有：
1) 提出了在线、异步和强批量码的形式化定义，并证明它们满足严格的层级关系；
2) 定义了 L‑AAD* 家族，放宽了传统 L‑AAD 的互斜性与维度一致性要求，且证明其能产生 (m,L)‑强异步批量码；
3) 给出系统化简单恢复集批量码与其关联二部图之间的完整等价条件，极大地扩展了以往只关注 C4‑free 或特定 theta‑graph 限制的结果；
4) 通过与线性子空间消除集等组合构造相结合，提出了一系列新的二进制批量码构造方法。

**🔧 技术方法**

主要技术手段包括：
- 线性代数与向量空间的子空间交集分析；
- 图论（特别是二部图、4‑环、theta‑图、路径计数）来刻画恢复集的交叉结构；
- 组合数学中的 AAD 族与其推广 L‑AAD* 的构造；
- 递归、交换与扩展性质的组合，构造在线与异步恢复算法；
- 证明多种包含与交换性质等价，构成恢复算法的结构化框架。

**📊 数据集**

本文为理论性工作，未使用具体实验数据集；所有结果均为数学证明与组合构造，所涉及的“数据集”仅是抽象向量空间、二部图顶点集合等。

**📈 对比分析**

由于研究主要是理论性质，比较基于已有文献中的构造与界限。论文中给出的构造在参数范围内实现了比以往 L‑AAD、C4‑free 等构造更宽松的条件，从而获得了更大批量度（m）或更小恢复次数（t=⌈m/L⌉）的批量码。文中给出的上界与下界与现有最优已知结果相近或改进，证明了所提方法在理论性能上的优越性，但未给出实验验证。

**⚠️ 局限性**

局限性与未来工作：
- 仍缺乏对所有参数范围内的构造的完整说明，特别是如何在实际系统中实现这些恢复算法；
- L‑AAD* 族的存在性与构造仍需更多具体例子，特别是在更大 q 或更高维场景下；
- 对于 t≥2 的强异步与异步批量码的差异仅在理论上给出，尚未证明两者不等价；
- 通过图论条件得到的批量码虽然一般化，但对特殊结构的最佳性尚未完全证明；
- 实际性能（如码率、复杂度、恢复延迟）未在实验中评估，需要后续实证研究。

---

## 198. UFO: A Unified Flow-Oriented Framework for Robust Continual Graph Learning

**arXiv ID:** 2605.09862 | [PDF](https://arxiv.org/pdf/2605.09862v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 199. Action Recommendations for Sequentially Rational Strategic Agents

**arXiv ID:** 2605.09785 | [PDF](https://arxiv.org/pdf/2605.09785v1)

**作者:** Renyan Sun `[一作]` (University of Southern California), Ashutosh Nayyar `[通讯]` (University of Southern California)

**通讯引用:** 1545 | [OpenAlex ID](https://openalex.org/A5040616226)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计一个动态系统的行动推荐策略，使得两位自利代理在遵循推荐时能够达到设计者的最优目标，并且推荐的行为对代理而言在每个时刻都是顺序合理的。

**💡 创新点**

①证明在完整对称信息下，采用仅依赖当前状态的马尔可夫推荐策略即可获得全局最优；②把顺序合理性约束转化为一组线性不等式，并通过逆向递推的线性规划序列实现最优推荐；③将方法推广到一般信息结构，仍能通过马尔可夫化处理。

**🔧 技术方法**

动态规划（逆向递推）、线性规划求解、马尔可夫决策过程（MDP）与约束MDP（CMDP）的理论框架。

**📊 数据集**

主要使用自定义的多接入广播系统实例，状态为两方缓冲区容量，动作为传输量，奖励函数包含利用率、公平度与容量违规惩罚，参数包括缓冲区大小、到达概率、惩罚系数等。

**📈 对比分析**

通过数值实验比较三种策略：无约束设计者（UD）、CMDP 约束设计者、以及本文的顺序合理推荐策略。实验显示，顺序合理策略在设计者奖励上略低于CMDP，表明顺序合理性约束的成本；同时，顺序合理策略需要使用混合推荐，揭示了其与传统最优策略的差异。

**⚠️ 局限性**

①仅考虑完整对称信息；②不引入动态补偿或转移支付；③方法在状态或行动空间增大时计算量指数级增长；④仅针对有限时限的离散时间系统，连续时间或无限期情形未覆盖。

---

## 200. Absurd World: A Simple Yet Powerful Method to Absurdify the Real-world for Probing LLM Reasoning Capabilities

**arXiv ID:** 2605.09678 | [PDF](https://arxiv.org/pdf/2605.09678v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 201. Dynamic Rank, Basis, and Matching

**arXiv ID:** 2605.09917 | [PDF](https://arxiv.org/pdf/2605.09917v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 202. Emergent Communication for Co-constructed Emotion Between Embodied Agents via Collective Predictive Coding

**arXiv ID:** 2605.09522 | [PDF](https://arxiv.org/pdf/2605.09522v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 203. Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach

**arXiv ID:** 2605.09964 | [PDF](https://arxiv.org/pdf/2605.09964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 204. VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection

**arXiv ID:** 2605.10229 | [PDF](https://arxiv.org/pdf/2605.10229v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 205. AwareLLM: A Proactive Multimodal Ecosystem for Personalized Human-AI Collaboration to Enhance Productivity

**arXiv ID:** 2605.09625 | [PDF](https://arxiv.org/pdf/2605.09625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 206. Explanation-Aware Learning for Enhanced Interpretability in Biomedical Imaging

**arXiv ID:** 2605.10054 | [PDF](https://arxiv.org/pdf/2605.10054v1)

**作者:** Zubair Faruqui `[一作]` (Missouri State University), Rahul Dubey `[通讯]` (Missouri State University)

**通讯引用:** 282 | [OpenAlex ID](https://openalex.org/A5059411044)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了一种在医学图像分类中将解释监督加入训练目标的框架，并系统评估不同解释损失设计与权重对预测性能和解释质量的影响。

**💡 创新点**

提出了统一的解释意识训练框架，使用多种 logit/概率差异的解释损失，并量化了注释覆盖率与 saliency 精度的权衡，为粗注释环境下的可解释模型提供实用指南。

**🔧 技术方法**

采用 Grad-CAM 解释机制、logit/概率差异解释损失、解释权重 α、DenseNet-121 基础网络以及二分类交叉熵损失与解释损失的组合。

**📊 数据集**

使用 VinDr-CXR 胸部 X 光图像数据集，该数据集包含 14 种病理的粗注释框。

**📈 对比分析**

在 7 种疾病的二分类任务上，比较了 8 种解释损失与 4 种 α 值共 224 模型。实验表明解释监督对准确率影响小，logit 平方损失在注释覆盖率和 saliency 精度上显著优于概率损失；α 可调节覆盖-精度权衡。

**⚠️ 局限性**

仅针对二分类任务，粗注释可能导致定位精度有限；解释损失对不同疾病的敏感性不同，需进一步探索多标签/多类别场景和更细粒度标注。

---

## 207. ChladniSonify: A Visual-Acoustic Mapping Method for Chladni Patterns in New Media Art Creation

**arXiv ID:** 2605.09846 | [PDF](https://arxiv.org/pdf/2605.09846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 208. Don't Click That: Teaching Web Agents to Resist Deceptive Interfaces

**arXiv ID:** 2605.09497 | [PDF](https://arxiv.org/pdf/2605.09497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 209. Not-So-Strange Love: Language Models and Generative Linguistic Theories are More Compatible than They Appear

**arXiv ID:** 2605.10061 | [PDF](https://arxiv.org/pdf/2605.10061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 210. Unpredictability dissociates from structured control in language agents

**arXiv ID:** 2605.09692 | [PDF](https://arxiv.org/pdf/2605.09692v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 211. GLiNER-Relex: A Unified Framework for Joint Named Entity Recognition and Relation Extraction

**arXiv ID:** 2605.10108 | [PDF](https://arxiv.org/pdf/2605.10108v1)

**作者:** Ihor Stepanov `[一作]` (Knowledgator Engineering), Vivek Kalyanarangan `[通讯]` (Baldor Technologies Pvt. Ltd. (IDfy))

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种统一模型GLiNER‑Relex，能够在一次前向传播中同时完成命名实体识别（NER）和关系抽取（RE），并支持在推理时通过自然语言标签实现任意实体和关系类型的零样本抽取。

**💡 创新点**

创新点在于：①将GLiNER的共享编码器扩展为全局端到端的实体与关系抽取框架；②引入关系得分模块与可选的邻接指导实体对构造，避免传统管线式错误传播；③在同一Transformer内同时学习文本、实体标签和关系标签的表示，实现在未见类型上的零样本推断。

**🔧 技术方法**

主要技术包括共享的双向Transformer编码器（DeBERTa‑v3）、span‑based实体表示、可选的多种实体对邻接解码器、关系对向量投影+MLP +点积得分，整体训练采用多任务焦点损失（entity、relation、可选adjacency）。

**📊 数据集**

在四个标准RE基准上评估：CoNLL04、DocRED、FewRel、CrossRE；训练阶段使用大规模FineWeb采样文本与LLM（Qwen3‑32B、Gemini）生成的自监督标注，后期微调约3000条高质量样本。

**📈 对比分析**

与GLiREL（专用关系分类）、GLiNER2（多任务GLiNER）以及GPT‑5‑mini（LLM零样本推断）对比，GLiNER‑Relex在四个数据集的微平均Micro‑F1为25.6%，在文档级别DocRED和跨域CrossRE上分别为31.3%和18.1%，在CoNLL04和FewRel上虽略逊于GPT‑5‑mini，但明显优于GLiNER2，并在端到端任务中实现最优或接近最优性能。

**⚠️ 局限性**

局限性包括：①对细粒度、数量众多的关系类型仍难以达到全监督或大型LLM水平；②全对枚举实体对导致实体密集场景下高计算量且误报率上升；③长文档处理受限于固定长度编码器和二次方实体对增长；④缺少对多种邻接解码器的系统性消融与优化；⑤在跨域泛化仍受实体与关系标签表达能力限制。

---

## 212. Prospective Compression in Human Abstraction Learning

**arXiv ID:** 2605.09985 | [PDF](https://arxiv.org/pdf/2605.09985v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 213. Hardness Amplification for (Sparse) LPN

**arXiv ID:** 2605.10056 | [PDF](https://arxiv.org/pdf/2605.10056v1)

**作者:** Divesh Aggarwal `[一作]` (National University of Singapore), Li Zeyong `[通讯]` (National University of Singapore)

**通讯引用:** 357 | [OpenAlex ID](https://openalex.org/A5035809548)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

证明了学习带噪声的平衡问题（LPN）及其稀疏变体的新难度放大结果。目标是从m个带噪声的线性样本中恢复一个秘密向量。

**💡 创新点**

提出了一种实例分数放大定理，任何成功概率为ε的算法可以转化为在相关分布上成功概率为1-δ的算法，从而实现自我放大。

**🔧 技术方法**

使用了直接乘积框架，结合了Hirahara和Shimizu的自我放大框架，分析了如何将k个独立实例组合成一个更大的实例。

**📊 数据集**

使用了LPN和稀疏LPN的标准分布，具体参数为𝖫𝖯𝖭_η,n,m和稀疏变体。

**📈 对比分析**

通过与现有算法的比较，证明了在相关参数下，成功概率的放大效果显著，尤其是在大多数实例上实现了高成功率。

**⚠️ 局限性**

当前的工作在稀疏LPN的最坏情况到平均情况的减少方面仍然存在局限性，尚未找到有效的最坏情况到平均情况的减少方法。

---

## 214. Generating synthetic electronic health record data using agent-based models to evaluate machine learning robustness under mass casualty incidents

**arXiv ID:** 2605.09951 | [PDF](https://arxiv.org/pdf/2605.09951v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 215. Towards an End-To-End System for Real-Time Gesture Recognition from Surface Vibrations

**arXiv ID:** 2605.10110 | [PDF](https://arxiv.org/pdf/2605.10110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 216. 3DReflecNet: A Large-Scale Dataset for 3D Reconstruction of Reflective, Transparent, and Low-Texture Objects

**arXiv ID:** 2605.10204 | [PDF](https://arxiv.org/pdf/2605.10204v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 217. Geometry Conflict: Explaining and Controlling Forgetting in LLM Continual Post-Training

**arXiv ID:** 2605.09608 | [PDF](https://arxiv.org/pdf/2605.09608v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 218. The Impossibility of Simultaneous Time and I/O Optimality for The Planar Maxima and Convex Hull Problems

**arXiv ID:** 2605.09464 | [PDF](https://arxiv.org/pdf/2605.09464v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 219. Neuromorphic Reinforcement Learning for Quadruped Locomotion Control on Uneven Terrain

**arXiv ID:** 2605.09595 | [PDF](https://arxiv.org/pdf/2605.09595v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 220. SDTalk: Structured Facial Priors and Dual-Branch Motion Fields for Generalizable Gaussian Talking Head Synthesis

**arXiv ID:** 2605.09956 | [PDF](https://arxiv.org/pdf/2605.09956v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 221. TIDE-Bench: Task-Aware and Diagnostic Evaluation of Tool-Integrated Reasoning

**arXiv ID:** 2605.09544 | [PDF](https://arxiv.org/pdf/2605.09544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 222. Categorical Message Passing Language (CaMPL) for programmers

**arXiv ID:** 2605.09491 | [PDF](https://arxiv.org/pdf/2605.09491v1)

**作者:** Daniel Kiyoshi Hashimoto `[一作]` (Universidade Federal do Rio de Janeiro), Priyaa Varshinee Srinivasan `[通讯]` (Tallinn University of Technology)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5048945636)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

本文提出并实现了一种基于线性作用范畴（linear actegories）的并发编程语言 CaMPL，提供消息传递、可自定义通道类型、受控非确定性（race）以及高阶进程等特性。

**💡 创新点**

创新点在于将线性逻辑与并发编程通过 Curry‑Howard‑Lambek 对应结合，利用线性作用范畴保证程序无死锁/无环拓扑；引入协议/共协议实现递归通道类型；通过 race 实现受控非确定性；以及实现高阶进程的编码/解码机制。

**🔧 技术方法**

技术手段包括：线性逻辑与线性作用范畴的理论框架；类型推断与检查器；编译器实现（包括自定义通道类型、race 处理、存储/使用高阶进程的内置函数）；以及在语法和语义层面对并发交互的 formalization。

**📊 数据集**

本文不涉及数据集；它是语言设计与实现的论文，主要通过代码示例和理论证明展示功能。

**📈 对比分析**

未进行实验性性能比较；讨论主要集中在理论性质（如无死锁、无环拓扑）和语言特性演示，未给出运行时性能指标。

**⚠️ 局限性**

局限性包括：禁止一般递归以保证无死锁/无环拓扑，导致表达能力受限；仍处于实验阶段，缺乏分布式网络支持和量化性能评估；对非确定性语义的完整分类与证明尚未完成。

---

## 223. DriveFuture: Future-Aware Latent World Models for Autonomous Driving

**arXiv ID:** 2605.09701 | [PDF](https://arxiv.org/pdf/2605.09701v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 224. LEAD: Length-Efficient Adaptive and Dynamic Reasoning for Large Language Models

**arXiv ID:** 2605.09806 | [PDF](https://arxiv.org/pdf/2605.09806v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 225. Byte-Exact Deduplication in Retrieval-Augmented Generation: A Three-Regime Empirical Analysis Across Public Benchmarks

**arXiv ID:** 2605.09611 | [PDF](https://arxiv.org/pdf/2605.09611v1)

**作者:** Sietse Schelpe `[一作]` `[通讯]` (Corbenic AI, Inc.), Sietse Schelpe (Corbenic AI, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在检索增强生成（RAG）流水线中测量并去除字节级完全重复的检索片段，评估其对提示长度、成本、延迟和输出质量的影响。

**💡 创新点**

① 提供了三种冗余程度的实测基准（学术清洁、企业版本化、对话累积）并证明去重在高冗余情形下可安全压缩80%以上字节；② 通过跨供应商5名评测者的校准面板验证去重后质量仍低于5% Wilson 95% 上限；③ 强调去重的算法等价性与可复现性，提出可部署的低微秒内存占用实现。

**🔧 技术方法**

字节级集合去重（使用哈希/ Python set() 等实现），评测面板协议、噪声去除审核、对比 MinHash‑LSH 近似去重，成本/延迟测量基于 OpenRouter API 调用。

**📊 数据集**

BeIR 22.2M 片段（学术）、合成企业语料（Wiki 修订、arXiv 版本、StackExchange Q&A，1,526 片段）、WildChat 5,000 多轮对话（真实对话历史）。

**📈 对比分析**

与未去重基线（15/50 top‑k）对比；Byte‑exact 去重可在清洁数据中实现 0.16% 的字节减少，在企业语料中实现 24.03%，在对话语料中实现 80.34%；在高冗余模式下质量测评通过 5% Wilson 95% 上限（各供应商 UCL 1.4–4.3%）。成本和延迟随字节减少线性下降，预填充阶段可获得数十到百毫秒的加速。

**⚠️ 局限性**

仅在单一学术/企业/对话数据上评测，未覆盖专业领域（法律、医疗等）；高冗余实验样本量（200/400）不等；评测者为作者，存在主观偏差；对话（80% 去重）未进行面板评测，仅为通信冗余度估计；未验证在大规模多租户环境或高并发下的系统级性能。

---

## 226. Guided Streaming Stochastic Interpolant Policy

**arXiv ID:** 2605.10051 | [PDF](https://arxiv.org/pdf/2605.10051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 227. Beyond Language: Format-Agnostic Reasoning Subspaces in Large Language Models

**arXiv ID:** 2605.09496 | [PDF](https://arxiv.org/pdf/2605.09496v1)

**作者:** Aojie Yuan `[一作]` (University of Southern California), Zhiyuan Su `[通讯]` (Duke University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型在不同符号系统（英语散文、中文散文、法语散文、Python代码、数学符号、结构化数据）之间的推理表示是否共享，并提出了格式无关推理子空间（FARS）。

**💡 创新点**

创新点在于系统构建三维无关基准（TriForm Benchmark），通过概念中心PCA提取10维FARS子空间，证明其在不同模型和架构中的一致性以及因果可操作性，并揭示宣言式与程序式形式的差异。

**🔧 技术方法**

使用了RSA、跨格式线性探测、格式神经元熵分析、中心化核对齐、激活补丁、子空间补丁以及PCA等多种解释与因果方法。

**📊 数据集**

数据集为自程序生成的TriForm Benchmark，包含18种推理概念、6种表面形式和3个实例，共324个刺激。

**📈 对比分析**

通过对比不同模型层级的RSA相关系数、跨格式探测准确率、补丁重叠率等指标，FARS在中间层表现出最高的概念可辨识度（RSA最大值≈0.2），跨格式补丁覆盖率90–96%，远高于全维度替换（44–56%）或方差最大化PCA（60–74%）。

**⚠️ 局限性**

限制包括仅评估基础模型，未考虑指令微调或RLHF对FARS的影响；FARS提取需要概念标签，缺乏无监督方法；TriForm Benchmark为程序化生成，缺乏自然语言真实语料；以及在70B+规模模型下FARS维度是否可扩展未知。

---

## 228. Task-Agnostic Noisy Label Detection via Standardized Loss Aggregation

**arXiv ID:** 2605.10165 | [PDF](https://arxiv.org/pdf/2605.10165v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 229. Complex-Valued Phase-Coherent Transformer

**arXiv ID:** 2605.10123 | [PDF](https://arxiv.org/pdf/2605.10123v1)

**作者:** Leona Hioki `[一作]` `[通讯]` (Antipatent), Leona Hioki (Antipatent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了相位一致的Transformer（PCT），在复数域中用无竞争的实数门替代传统softmax注意力，保持多层相位一致性。

**💡 创新点**

创新点在于将token非竞争的平滑门与L2归一化的相似度结合，并给出了四条件框架验证其在多层相位保持上的必要性。

**🔧 技术方法**

采用的技术包括复数线性投影、L2归一化、sigmoid/softplus等平滑门、残差+RMSNorm、RoPE位置编码，以及非softmax的注意力实现。

**📊 数据集**

使用的数据集涵盖合成任务（Copy、NIAH、ListOps）、文本/图像任务（LRA-ListOps、LRA-Text、LRA-Image）、图像分类（FFT-MNIST）、音频/信号（phase-memory、multi-pitch）以及真实物理复数域任务（RadioML、MusicNet）。

**📈 对比分析**

在参数公平（real_dim = complex_dim × 1.41）下，对9类任务进行六细胞对比。PCT在长距离记忆、定位检索、算法推理、相位敏感任务等方面明显优于softmax和复数softmax，并对学习率、批量、网络深度保持鲁棒性。

**⚠️ 局限性**

局限性包括在Real RadioML L1/L2任务中略逊于屏幕注意力；对更深层（>20层）或更大规模通用任务的评估不足；复数实现的计算成本与并行效率仍待进一步优化。

---

## 230. Zero-Shot Sim-to-Real Robot Learning: A Dexterous Manipulation Study on Reactive Catching

**arXiv ID:** 2605.09789 | [PDF](https://arxiv.org/pdf/2605.09789v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 231. Evaluating Tool Cloning in Agentic-AI Ecosystems

**arXiv ID:** 2605.09817 | [PDF](https://arxiv.org/pdf/2605.09817v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 232. Balancing Efficiency and Fairness in Traffic Light Control through Deep Reinforcement Learning

**arXiv ID:** 2605.10170 | [PDF](https://arxiv.org/pdf/2605.10170v1)

**作者:** Matteo Cederle `[一作]` (University of Padova), Gian Antonio Susto `[通讯]` (University of Padova)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种在交通灯控制中考虑车辆与行人公平性的深度强化学习代理，动态平衡两类流量以降低拥堵。

**💡 创新点**

创新点在于引入可调公平系数β的复合奖励函数，并让代理直接选择任意相位，显式兼顾行人和车辆需求。

**🔧 技术方法**

采用了Double DQN深度强化学习框架，状态空间包含相位、时间、车流密度、排队长度与行人计数，动作空间为四向三道交叉口的所有相位。

**📊 数据集**

实验数据来自FLOW+SUMO仿真平台，使用预设的车流量（N/S 750/850/1000/veh/h，E/W 400/500/600/veh/h）和行人流量（N/S 500/ ped/h，E/W 300/ ped/h）。

**📈 对比分析**

通过与基于Webster公式的固定周期灯相比较，代理在轻、中、重流量场景下均实现了车辆和行人平均等待时间显著降低，且在β调节下形成可观的Pareto前沿。

**⚠️ 局限性**

局限性包括仅在仿真环境验证，未测试真实道路传感器可靠性；奖励设计依赖全局等待时间信息；缺乏对能源或CO₂排放等可持续性指标的考量。

---

## 233. Rényi Rate-Distortion-Perception-Privacy Tradeoff under Indirect Observation

**arXiv ID:** 2605.09921 | [PDF](https://arxiv.org/pdf/2605.09921v1)

**作者:** Jiahui Wei `[一作]` (University of Granada), Marios Kountouris `[通讯]` (University of Granada)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种 Rényi Rate‑Distortion‑Perception‑Privacy（R‑RDP）框架，专门针对间接源编码问题，给出了标量高斯模型的最优测试通道及其性能下界。

**💡 创新点**

创新点在于①引入条件 S‑Sibson 信息作为隐私度量，能够剥离语义恢复所固有的泄露；②对 α>1 的泊松函数表示进行精确概率分布推导，得到更紧的 Rényi 熵上界；③统一考虑失真、感知与隐私三大约束，揭示它们之间的三向权衡。

**🔧 技术方法**

主要技术包括 Rényi 熵与 Sibson 互信息理论、泊松函数表示与几何‑混合分布、Wasserstein‑2 感知度量、正交分解以及极大似然率的概率分析。

**📊 数据集**

实验验证主要基于二元对称信道与标量高斯信道（无真实数据集），用以演示理论推导与数值仿真的一致性。

**📈 对比分析**

通过理论推导与数值仿真相结合，展示了在满足失真、感知与隐私约束下，泊松函数表示的 Rényi 熵上界明显优于传统对数矩法；同时证明在高斯模型下可实现更低的通信率。

**⚠️ 局限性**

局限性包括：①条件隐私度量的定义尚无统一标准；②结果主要针对标量高斯信道，尚未推广到多维或非高斯情形；③缺乏大规模实验验证，实际性能依赖于模型假设。

---

## 234. Bridging the Cognitive Gap: A Unified Memory Paradigm for 6G Agentic AI-RAN

**arXiv ID:** 2605.10036 | [PDF](https://arxiv.org/pdf/2605.10036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 235. Modeling Implicit Conflict Monitoring Mechanisms against Stereotypes in LLMs

**arXiv ID:** 2605.09647 | [PDF](https://arxiv.org/pdf/2605.09647v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 236. Empty SPACE: Cross-Attention Sparsity for Concept Erasure in Diffusion Models

**arXiv ID:** 2605.10198 | [PDF](https://arxiv.org/pdf/2605.10198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 237. Computing Flows in Subquadratic Space

**arXiv ID:** 2605.09547 | [PDF](https://arxiv.org/pdf/2605.09547v1)

**作者:** Jan van den Brand `[一作]` (Georgia Institute of Technology), Albert Weng `[通讯]` (Georgia Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种多次遍历流算法和二方通信协议，能够在子二次空间（O(n^1.5) 词）内近似求解带整数容量与费用的最小费用流，并能在最后一次遍历时返回每条边的流量。

**💡 创新点**

创新点包括：①突破传统 Ω(n^2) 空间下界，通过在流查询时提供边信息来绕过编码/解码限制；②将鲁棒内点法与电流循环（electric circulation）相结合，在每一步仅存储 O(n) 维向量即可重构整个 m 维流向量；③利用 Lewis‑weight 以及 Johnson–Lindenstrauss 近似构造稀疏逆矩阵，进一步压缩所需空间；④证明该方法在流式和通信模型中均可实现 O(n^1.5) 空间和通信量，且误差可任意调节。

**🔧 技术方法**

主要技术手段包括：内部点法（robust interior point method）与中心路径；电流循环和电阻模型；Lewis‑weights 与有效电阻的近似；Johnson–Lindenstrauss 采样构造稀疏逆矩阵；电路流递归表示（存储增量而非完整流向量）；在通信模型中使用对称信息交换和 Isolation Lemma 进行唯一性保证。

**📊 数据集**

论文未涉及具体实验数据集，侧重于理论分析与算法设计。

**📈 对比分析**

与现有流式/通信下界相比，本文实现了子二次空间（O(n^1.5) 词）和 O(√n) 轮次的算法，显著优于传统的 O(n^2) 空间下界；在通信模型中达成 O(n^1.5) 位的通信量，匹配或超过已知上界。

**⚠️ 局限性**

主要局限：①误差为可调的加性误差，若需完全精确整数解需额外使用 Isolation Lemma 与多次重抽样；②对比传统最小费用流算法的时间复杂度仍为 O(mn log^2(W/ε))，在稠密图中仍较慢；③在 bit‑complexity 上需额外对浮点/定点误差进行细致控制；④方法依赖于流式模型可多遍历和对边信息的即时访问，对严格单遍流式模型不适用。

---

## 238. LLM-Driven Performance-Space Augmentation for Meta-Learning-Based Algorithm Selection

**arXiv ID:** 2605.09518 | [PDF](https://arxiv.org/pdf/2605.09518v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 239. Probing Routing-Conditional Calibration in Attention-Residual Transformers

**arXiv ID:** 2605.09850 | [PDF](https://arxiv.org/pdf/2605.09850v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 240. GSMap: 2D Gaussians for Online HD Mapping

**arXiv ID:** 2605.09619 | [PDF](https://arxiv.org/pdf/2605.09619v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 241. Elemental Alchemist: A Generative Interface for Semantic Control of Particle Systems Across Dynamic Levels of Abstraction

**arXiv ID:** 2605.10014 | [PDF](https://arxiv.org/pdf/2605.10014v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 242. OpenZL: Using Graphs to Compress Smaller and Faster

**arXiv ID:** 2605.09928 | [PDF](https://arxiv.org/pdf/2605.09928v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 243. Doubly Robust Proxy Causal Learning with Neural Mean Embeddings

**arXiv ID:** 2605.09514 | [PDF](https://arxiv.org/pdf/2605.09514v1)

**作者:** Bariscan Bozkurt `[一作]` (University College London), Houssam Zenati `[通讯]` (University College London)

**通讯引用:** 1150 | [OpenAlex ID](https://openalex.org/A5037008263)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种基于代理变量的神经双稳健框架，能够在连续和结构化处理变量下估计总体、异质性和条件响应曲线。

**💡 创新点**

创新点包括：①提出了可学习的神经均值嵌入处理桥，①结合了神经结果桥与处理桥实现双稳健；②给出了弱范数下的一致性证明，阐明双稳健误差受最终回归和桥函数弱误差控制；③提出两种稳健训练算法（DRPCLNET‑V1/V2）并实现了多阶段历史感知更新。

**🔧 技术方法**

采用深度学习（两阶段桥网络、闭式历史岭回归、可微损失）、均值嵌入技术、核密度比估计以及多阶段回归融合。

**📊 数据集**

在四个基准上评估：低维连续处理、16维高维合成、64×64 dSprites 图像处理、以及二元处理的异质性设置。

**📈 对比分析**

与单桥神经估计器（OutcomeNet、TreatmentNet）及传统核/半参数基线（DRKPV、PKDR、KPV）比较，实验显示 DRPCLNET‑V1/V2 在样本量增大时持续优于或与最强基线相当，尤其在高维和结构化数据上表现突出。

**⚠️ 局限性**

局限性包括：对超参数调优敏感；处理桥依赖预估的密度比，密度比误差会影响理论与实践；缺乏对非凸随机优化收敛的理论分析。

---

## 244. MTA-RL: Robust Urban Driving via Multi-modal Transformer-based 3D Affordances and Reinforcement Learning

**arXiv ID:** 2605.10177 | [PDF](https://arxiv.org/pdf/2605.10177v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 245. Asymptotic Hausdorff and Language Similarity

**arXiv ID:** 2605.09668 | [PDF](https://arxiv.org/pdf/2605.09668v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

---

## 246. PermuQuant: Lowering Per-Group Quantization Error by Reordering Channels for Diffusion Models

**arXiv ID:** 2605.09503 | [PDF](https://arxiv.org/pdf/2605.09503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 247. GLiNER2-PII: A Multilingual Model for Personally Identifiable Information Extraction

**arXiv ID:** 2605.09973 | [PDF](https://arxiv.org/pdf/2605.09973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 248. Deterministic vs. LLM-Controlled Orchestration for COBOL-to-Python Modernization

**arXiv ID:** 2605.09894 | [PDF](https://arxiv.org/pdf/2605.09894v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 249. PlantMarkerBench: A Multi-Species Benchmark for Evidence-Grounded Plant Marker Reasoning

**arXiv ID:** 2605.10032 | [PDF](https://arxiv.org/pdf/2605.10032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 250. Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework

**arXiv ID:** 2605.10043 | [PDF](https://arxiv.org/pdf/2605.10043v1)

**作者:** Xilai Ma `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 116097 | [OpenAlex ID](https://openalex.org/A5100336796)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将目标用户历史视为正样本，其他用户数据视为隐式负样本，利用正负样本二进制反馈与 PU 学习相结合的 C‑BPO 框架实现 LLM 的个性化；

**💡 创新点**

首次将二进制反馈偏好优化迁移至个性化场景，并引入 PU 学习校正负样本偏差以消除偏好重叠问题，同时使用 EMA 参考点处理数据不平衡；

**🔧 技术方法**

采用二进制反馈偏好优化（BCO/KTO）、正负样本 PU 学习风险重构、参数高效微调（LoRA）、EMA 参考点估计等技术；

**📊 数据集**

在 LaMP 与 LongLaMP 个人化生成基准（新闻标题、学术标题、摘要、评论、主题等五个任务）上验证，使用 LLaMA、Qwen、Mistral 等多种 LLM 骨干；

**📈 对比分析**

与检索式（RAG、PAG）、SFT 微调（TAM、OPPU）、基于偏好（CoPE、KTO、BCO）等多类基线对比，C‑BPO 在所有任务和模型上均优于基线，尤其在传统 BPO 上表现突出；

**⚠️ 局限性**

对校正系数 α 的敏感性、需要集中辅助用户数据导致隐私风险、以及目前仅在生成基准上验证，尚未扩展到对话、推荐等更广泛的个性化任务。

---

## 251. EnactToM: An Evolving Benchmark for Functional Theory of Mind in Embodied Agents

**arXiv ID:** 2605.09826 | [PDF](https://arxiv.org/pdf/2605.09826v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 252. Swarm Skills: A Portable, Self-Evolving Multi-Agent System Specification for Coordination Engineering

**arXiv ID:** 2605.10052 | [PDF](https://arxiv.org/pdf/2605.10052v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 253. TreeWidzard: An Engine for Width-Based Dynamic Programming and Automated Theorem Proving

**arXiv ID:** 2605.09732 | [PDF](https://arxiv.org/pdf/2605.09732v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 254. Biosignal Fingerprinting: A Cross-Modal PPG-ECG Foundation Model

**arXiv ID:** 2605.09579 | [PDF](https://arxiv.org/pdf/2605.09579v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 255. Unlocking air traffic flow prediction through microscopic aircraft-state modeling

**arXiv ID:** 2605.10083 | [PDF](https://arxiv.org/pdf/2605.10083v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 256. M2A: Synergizing Mathematical and Agentic Reasoning in Large Language Models

**arXiv ID:** 2605.09879 | [PDF](https://arxiv.org/pdf/2605.09879v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 257. Deep Learning under Fractional-Order Differential Privacy

**arXiv ID:** 2605.09890 | [PDF](https://arxiv.org/pdf/2605.09890v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 258. Fetal Brain Imaging: A Composite Neural Network Approach for Keyframe Detection in Ultrasound Videos

**arXiv ID:** 2605.09750 | [PDF](https://arxiv.org/pdf/2605.09750v1)

**作者:** Aleksander Zamojski `[一作]` (Warsaw Universitu of Technology), Radoslaw Roszczyk `[通讯]` (Warsaw University of Technology)

**通讯引用:** 40 | [OpenAlex ID](https://openalex.org/A5041317532)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

构建了一个组合卷积神经网络和循环神经网络，用于检测胎儿脑超声视频的关键帧。

**💡 创新点**

创新点在于将空间特征提取（CNN）与时间依赖捕获（RNN）相结合，并提出基于CNN输出的帧质量度量。

**🔧 技术方法**

采用EfficientNetV2 (small)作为CNN，GRU作为RNN，并使用数据增强和多尺度变换来计算质量。

**📊 数据集**

使用公开的12,400张胎儿超声图像（1,792名患者）和130段短视频进行训练。

**📈 对比分析**

与传统基于运动能量、颜色直方图等方法对比，EfficientNetV2 + GRU实现了更高的分类准确率和更低的推理时间（具体数值未给出）。

**⚠️ 局限性**

局限在于缺乏真实临床验证、数据集仍相对有限、对类别不平衡的处理仍不完备，以及质量度量主要依赖于人工标注的类别。

---

## 259. Convex Optimization with Local Label Differential Privacy: Tight Bounds in All Privacy Regimes

**arXiv ID:** 2605.10200 | [PDF](https://arxiv.org/pdf/2605.10200v1)

**作者:** Lynn Chua `[一作]` (Google Research), Chiyuan Zhang `[通讯]` (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了在局部标签差分隐私（L-LDP）约束下的随机凸优化（SCO）问题，提出了一种新的高效非交互式L-LDP算法，显著改善了标签空间大小的依赖性。

**💡 创新点**

创新点在于提出了一种新的算法，能够在高隐私和中等隐私的情况下将过度风险的依赖性从O()降低到O(√())，并证明了这一界限是最优的。

**🔧 技术方法**

使用了基于子集选择随机化机制的非交互式L-LDP算法，结合了随机梯度下降（SGD）方法。

**📊 数据集**

使用了来自未知分布P的独立同分布样本集{(X_i, Y_i)}，其中特征X是公开的，标签Y是私密的。

**📈 对比分析**

与之前的算法相比，新的算法在高隐私和中等隐私情况下的过度风险界限显著改善，具体为O(√(/e^))和O(√(/))，而之前的算法在这两个情况下的依赖性为O()和O(√(K))，性能上有显著提升。

**⚠️ 局限性**

局限性在于，尽管算法在高隐私和中等隐私情况下表现良好，但在低隐私情况下的过度风险与非隐私算法相同，且在某些情况下可能仍存在对标签空间大小的多项式依赖。

---

## 260. OZ-TAL: Online Zero-Shot Temporal Action Localization

**arXiv ID:** 2605.09976 | [PDF](https://arxiv.org/pdf/2605.09976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 261. Towards infinite PCSP: a dichotomy for monochromatic cliques

**arXiv ID:** 2605.09815 | [PDF](https://arxiv.org/pdf/2605.09815v1)

**作者:** Demian Banakh `[一作]` (Jagiellonian University), Tamio-Vesa Nakajima `[通讯]` (Philipps University)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5072517256)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文从理论角度研究Promise MMSNP（承诺MMSNP）问题，建立其与有限域PCSP的多项式时间等价性，并针对禁止单色团的情形给出了完整的二分复杂度分类（在Rich 2-to-1假设下）。

**💡 创新点**

创新点在于①首次将Promise MMSNP引入并证明其可归约为有限域PCSP；②在禁止单色团问题上给出基于Rich 2-to-1假设的完整复杂度二分；③提出并利用“可重构”关系的重构性作为判定难易的足够条件。

**🔧 技术方法**

主要使用了逻辑与组合化简（MMSNP→PCSP的对应构造、稀疏不相容引理）、图同构与同伦（重构图连通性）、Fourier分析与不变原理（多切片的长码变换）以及PCSP的极化与长码技术。

**📊 数据集**

本研究为纯理论证明，不涉及实验数据集。

**📈 对比分析**

论文未进行实验或性能比较，所有结果均为理论复杂性证明。

**⚠️ 局限性**

主要限制在于结果仅在Rich 2-to-1假设成立时有效；未给出更弱假设下的结果，也未讨论对非单色团或边染色等更广泛问题的推广。

---

## 262. expo: Exploration-prioritized policy optimization via adaptive kl regulation and gaussian curriculum sampling

**arXiv ID:** 2605.09923 | [PDF](https://arxiv.org/pdf/2605.09923v1)

**作者:** Mingxiong Lin `[一作]` (OPPO AI Center), Haonan Lu `[通讯]` (OPPO AI Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在强化学习可验证奖励（RLVR）框架中，通过自适应KL缩放与高斯课程采样来动态分配探索预算，以提升大型语言模型在数学推理任务上的性能。

**💡 创新点**

创新点在于同时对KL正则化系数进行批量准确度条件缩放以及对问题采样采用围绕中等难度的高斯权重，实现探索与稳定性的协同调节。

**🔧 技术方法**

使用的技术包括准确度条件KL缩放、Gaussian Curriculum Sampling、GRPO（Group Relative Policy Optimization）、无偏K3 KL估计器、EMA平滑pass率等。

**📊 数据集**

数据集为DAPO-17K数学推理语料库以及六个竞赛级基准（AIME 2024/2025、MATH-500、Minerva、OlympiadBench、AMC）。

**📈 对比分析**

与基线GRPO相比，EXPO在两种模型规模（1.5B、8B）上在pass@1和pass@32均取得显著提升，尤其在8B模型的AIME 2025 pass@32提升13.34个百分点，平均pass@32提升2.66个百分点。

**⚠️ 局限性**

局限性包括仅针对二元验证奖励的数学推理任务，且所选的非线性缩放函数和高斯参数在其他奖励稠密或噪声场景下的通用性未验证。

---

## 263. The Truth Lies Somewhere in the Middle (of the Generated Tokens)

**arXiv ID:** 2605.09969 | [PDF](https://arxiv.org/pdf/2605.09969v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 264. LEVI: Stronger Search Architectures Can Substitute for Larger LLMs in Evolutionary Search

**arXiv ID:** 2605.09764 | [PDF](https://arxiv.org/pdf/2605.09764v1)

**作者:** Temoor Tanveer `[一作]` `[通讯]` (Independent Researcher), Temoor Tanveer (Independent Researcher)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LEVI，一个在 LLM 驱动的进化搜索中通过更强的搜索架构、角色感知模型路由和代理基准来显著降低成本的框架；

**💡 创新点**

创新点包括：① 用 CVT‑MAP‑Elites 架构与种子多样化初始化来持续保持解的多样性；② 角色感知路由器将 90% 的局部变异交给小模型，只有少量大模型用于结构性“范式转移”；③ 为评估成本高的情景设计排名保持的代理基准，减少 roll‑out 数量；

**🔧 技术方法**

技术手段涵盖：多样化种子生成、CVT‑MAP‑Elites 档案、输入/输出混合描述子、异步 AlphaEvolve 样式循环、角色感知 LLM 路由、贪婪子集选择的代理基准构造；

**📊 数据集**

使用了系统研究基准（ADR 任务套件中包含网络、LLM 部署、数据库、分布式系统等七项）以及四个提示优化基准（HotpotQA、IFBench、Hover、PUPA）进行评估；

**📈 对比分析**

与现有框架（GEPA、OpenEvolve、ShinkaEvolve、AdaEvolve、EvoX）在相同或更低预算下对比：在六个系统研究任务上 LEVI 以 3.3–6.7 倍的成本获得更高或相同的分数；在提示优化任务上 LEVI 在占用 GEPA 50% 以内的 roll‑out 数量下实现了最高的整体得分；

**⚠️ 局限性**

局限性包括：在评估本身成本高的场景（如模型训练）下可能不如低评估成本的任务优势明显；未充分探讨真实 wall‑clock 时间的节省；仍需少量大模型调用，尚未实现完全开放权重模型驱动的全流程。

---

## 265. Sub-Footprint Effect Correction in FW-LiDAR Point Clouds via Intra-Footprint Target Unmixing

**arXiv ID:** 2605.09845 | [PDF](https://arxiv.org/pdf/2605.09845v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 266. FreeMOCA: Memory-Free Continual Learning for Malicious Code Analysis

**arXiv ID:** 2605.09664 | [PDF](https://arxiv.org/pdf/2605.09664v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 267. MOTOR-Bench: A Real-world Dataset and Multi-agent Framework for Zero-shot Human Mental State Understanding

**arXiv ID:** 2605.09703 | [PDF](https://arxiv.org/pdf/2605.09703v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 268. NyayaAI: An AI-Powered Legal Assistant Using Multi-Agent Architecture and Retrieval-Augmented Generation

**arXiv ID:** 2605.10155 | [PDF](https://arxiv.org/pdf/2605.10155v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 269. MARGIN: Margin-Aware Regularized Geometry for Imbalanced Vulnerability Detection

**arXiv ID:** 2605.10240 | [PDF](https://arxiv.org/pdf/2605.10240v1)

**作者:** Yuteng Zhang `[一作]` (Northwest Normal University), Yafei Yang `[通讯]` (Northwest Normal University)

**通讯引用:** 19963 | [OpenAlex ID](https://openalex.org/A5100346563)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对软件漏洞检测中的频率与难度不平衡，提出了 MARGIN 框架，通过自适应角度边距与 von Mises–Fisher 分布对嵌入空间进行几何正则化，提升特征区分度和决策边界稳定性。

**💡 创新点**

创新点包括：① 从嵌入几何角度统一表述两种不平衡；② 采用 vMF 集中度估计每类分布并据此自适应设置角度边距；③ 结合类级 logit 缩放与几何中值原型，实现 ETF 结构收敛，显著降低扭曲区域。

**🔧 技术方法**

技术手段：CodeT5 作为编码器；单位 hypersphere 约束下的余弦 Softmax；vMF 分布估计 κ 并计算顶角；自适应角度边距与类级 logit 缩放；使用 Weiszfeld 算法求几何中值原型；余弦距离原型匹配用于推理。

**📊 数据集**

数据集：BigVul、MegaVul 与 ReposVul，均为 C/C++ 代码与 CWE 标签，呈现长尾频率分布。

**📈 对比分析**

通过与图神经网络、程序分析、近年 C/W-aware 预训练模型等多种基线比较，MARGIN 在二分类 F1/MCC 与宏 F1/宏 MCC 上均取得最高或接近最高的分数，尤其在少数类上提升显著；在 BigVul 上二分类 F1 达 86.57%，宏 F1 达 70.72%。

**⚠️ 局限性**

局限性：实验仅覆盖 C/C++ 数据，未验证对其他语言的泛化；对极少样本类别的提升仍有限；依赖高维 vMF 近似与大规模预训练模型，训练成本仍较高。

---

## 270. Any2Any 3D Diffusion Models with Knowledge Transfer: A Radiotherapy Planning Study

**arXiv ID:** 2605.09622 | [PDF](https://arxiv.org/pdf/2605.09622v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 271. Chebyshev Center-Based Direction Selection for Multi-Objective Optimization and Training PINNs

**arXiv ID:** 2605.09975 | [PDF](https://arxiv.org/pdf/2605.09975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 272. UTS at PsyDefDetect: Multi-Agent Councils and Absence-Based Reasoning for Defense Mechanism Classification

**arXiv ID:** 2605.09769 | [PDF](https://arxiv.org/pdf/2605.09769v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 273. APCD: Adaptive Path-Contrastive Decoding for Reliable Large Language Model Generation

**arXiv ID:** 2605.09492 | [PDF](https://arxiv.org/pdf/2605.09492v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 274. Workspace Optimization: How to Train Your Agent

**arXiv ID:** 2605.09650 | [PDF](https://arxiv.org/pdf/2605.09650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 275. Outlier-Robust Diffusion Solvers for Inverse Problems

**arXiv ID:** 2605.09477 | [PDF](https://arxiv.org/pdf/2605.09477v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 276. FPGA-Based Hardware Architecture for Contrast Maximization in Event-Based Vision

**arXiv ID:** 2605.09581 | [PDF](https://arxiv.org/pdf/2605.09581v1)

**作者:** Michal Filipkowski `[一作]`, Tomasz Kryjak `[通讯]` (AGH University of Krakow)

**通讯引用:** 710 | [OpenAlex ID](https://openalex.org/A5005086061)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现了在FPGA上对事件相机的Contrast Maximization（CM）算法的硬件加速架构，用于实时运动估计和目标跟踪

**💡 创新点**

首次将CM算法迁移至硬件实现，显著提升计算速度并实现低功耗运行

**🔧 技术方法**

利用FPGA的并行流水线、双线性投票、梯度上升优化、BRAM分块存储与缓冲技术

**📊 数据集**

使用DAVIS 240C事件相机数据集进行对象跟踪实验

**📈 对比分析**

与CPU（Intel Core i5-11300H）和GPU（Nvidia GeForce RTX 3050 Ti）对比，FPGA实现处理时间约0.92 ms，分别比CPU快≈200×、GPU快≈450×

**⚠️ 局限性**

受限于FPGA内部BRAM容量，ROI尺寸有限；目前仅支持简单平移模型，缺乏更高级优化方法与外部RAM扩展

---

## 277. Separate First, Fuse Later: Mitigating Cross-Modal Interference in Audio-Visual LLMs Reasoning with Modality-Specific Chain-of-Thought

**arXiv ID:** 2605.09906 | [PDF](https://arxiv.org/pdf/2605.09906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 278. Oracle Poisoning: Corrupting Knowledge Graphs to Weaponise AI Agent Reasoning

**arXiv ID:** 2605.09822 | [PDF](https://arxiv.org/pdf/2605.09822v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 279. LLM Agents Enable User-Governed Personalization Beyond Platform Boundaries

**arXiv ID:** 2605.09794 | [PDF](https://arxiv.org/pdf/2605.09794v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 280. Slum Detection and Density Mapping with AlphaEarth Foundations: A Representation Learning Evaluation Across 12 Global Cities

**arXiv ID:** 2605.10029 | [PDF](https://arxiv.org/pdf/2605.10029v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 281. Key Encapsulation Mechanism-Based Integrated Encryption Scheme (KEM-IES)

**arXiv ID:** 2605.10175 | [PDF](https://arxiv.org/pdf/2605.10175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 282. Neural Distance-Guided Path Integral Control for Tractor-Trailer Navigation

**arXiv ID:** 2605.09939 | [PDF](https://arxiv.org/pdf/2605.09939v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 283. Entropy-informed Decoding: Adaptive Information-Driven Branching

**arXiv ID:** 2605.09745 | [PDF](https://arxiv.org/pdf/2605.09745v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 284. Efficient Neural Architectures for Real-Time ECG Interpretation on Limited Hardware

**arXiv ID:** 2605.09848 | [PDF](https://arxiv.org/pdf/2605.09848v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 285. Selection of the Best Policy under Fairness Constraints for Subpopulations

**arXiv ID:** 2605.09945 | [PDF](https://arxiv.org/pdf/2605.09945v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 286. When to Re-Commit: Temporal Abstraction Discovery for Long-Horizon Vision-Language Reasoning

**arXiv ID:** 2605.09860 | [PDF](https://arxiv.org/pdf/2605.09860v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 287. On Uniform Error Bounds for Kernel Regression under Non-Gaussian Noise

**arXiv ID:** 2605.09757 | [PDF](https://arxiv.org/pdf/2605.09757v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 288. Learning to Align Generative Appearance Priors for Fine-grained Image Retrieval

**arXiv ID:** 2605.09859 | [PDF](https://arxiv.org/pdf/2605.09859v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 289. Scratchpad Patching: Decoupling Compute from Patch Size in Byte-Level Language Models

**arXiv ID:** 2605.09630 | [PDF](https://arxiv.org/pdf/2605.09630v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 290. Quantum Circuit Simulation of Compartmental Drug Dynamics: Leveraging Variational Algorithms for Nonlinear Mixed-Effects Population Pharmacokinetics

**arXiv ID:** 2605.09691 | [PDF](https://arxiv.org/pdf/2605.09691v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 291. ReCoVR: Closing the Loop in Interactive Composed Video Retrieval

**arXiv ID:** 2605.09836 | [PDF](https://arxiv.org/pdf/2605.09836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 292. Safe Exploration for Nonlinear Processes Using Online Gaussian Process Learning

**arXiv ID:** 2605.09772 | [PDF](https://arxiv.org/pdf/2605.09772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 293. KV-RM: Regularizing KV-Cache Movement for Static-Graph LLM Serving

**arXiv ID:** 2605.09735 | [PDF](https://arxiv.org/pdf/2605.09735v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 294. Insight: Enhancing Mobile Accessibility for Blind and Visually Impaired Users with LLMs

**arXiv ID:** 2605.09803 | [PDF](https://arxiv.org/pdf/2605.09803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 295. Automated Approach for Solving Infinite-state Polynomial Reachability Games

**arXiv ID:** 2605.10169 | [PDF](https://arxiv.org/pdf/2605.10169v1)

**作者:** Krishnendu Chatterjee `[一作]` (Institute of Science and Technology Austria), Đorđe Žikelić `[通讯]` (Singapore Management University)

**通讯引用:** 342 | [OpenAlex ID](https://openalex.org/A5041082080)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可在无限状态可达游戏中证明存在获胜策略的排名证书，并给出基于模板的自动化求解算法

**💡 创新点**

首次为无限状态可达游戏提供既完备又可验证的排名证书框架；算法在多项式游戏上实现了子指数复杂度并支持符号精度参数

**🔧 技术方法**

基于多项式模板、Putinar定理/弗卡斯法的量化约简、SMT求解（Z3/MathSAT5）

**📊 数据集**

主要使用Cinderella‑Stepmother游戏和机器人混合物安全性等经典无限状态游戏作为实验基准

**📈 对比分析**

与两种先进的无限状态可达游戏求解器相比，实验显示在所有测试容量（含可变参数2‑ϵ）下均能在秒级完成，而对手要么超时要么失败；在更难的多项式变体中，仍能在秒级求解

**⚠️ 局限性**

仅适用于多项式约束的可达游戏；只处理可达目标；在存在无穷多动作的情况下不完全，仍需进一步扩展

---

## 296. SocialDirector: Training-Free Social Interaction Control for Multi-Person Video Generation

**arXiv ID:** 2605.10079 | [PDF](https://arxiv.org/pdf/2605.10079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 297. SAGE: Scalable Agentic Grounded Evaluation for Crop Disease Diagnosis

**arXiv ID:** 2605.09768 | [PDF](https://arxiv.org/pdf/2605.09768v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 298. ConCovUp: Effective Agent-Based Test Driver Generation for Concurrency Testing

**arXiv ID:** 2605.09573 | [PDF](https://arxiv.org/pdf/2605.09573v1)

**作者:** Yuandao Cai `[一作]` (Independent Researcher), Charles Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1871 | [OpenAlex ID](https://openalex.org/A5101490553)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个基于多智能体的框架，利用大型语言模型（LLM）与程序静态分析相结合，自动生成针对C/C++库的并发测试驱动程序。

**💡 创新点**

创新点在于：①将静态分析提取的共享内存访问点与调用上下文作为LLM的输入；②设计了一个后向路径推理智能体，利用LLM的语义推理推导具体输入；③构建了反馈循环，动态执行结果用于修正路径和测试生成；④在三阶段工作流（分析、路径推理、测试生成）中整合LLM与工具，显著提升并发覆盖率。

**🔧 技术方法**

技术手段包括：LLVM 15静态分析工具（调用图、共享访问分析），CFG后向搜索与路径约束摘要，LLM（Claude Sonnet 4.6、GPT 5.4、Kimi K2.5）进行语义推理与代码生成，覆盖度测量（SMAP Coverage）与ThreadSanitizer监控，迭代反馈机制。

**📊 数据集**

实验数据集：9个广泛使用的C/C++开源库（c-ares、cJSON、concurrentqueue、curl、libsodium、libuv、spdlog、zlib、zlog），总计约1000k行代码。

**📈 对比分析**

与通用Claude Code编码代理基线进行对比，使用相同迭代预算（K_refine=3）。结果显示：平均SMAP Coverage从36.6%提升至68.1%，提升幅度约31.5个百分点；消融实验表明静态目标信息带来小幅提升，完整工作流（路径推理+反馈）贡献最大；不同LLM后端影响显著，Claude Sonnet 4.6表现最佳。

**⚠️ 局限性**

局限性包括：①仅关注单一共享变量的访问配对，未覆盖多变量或跨变量同步问题；②静态分析的保守性导致误报和不可达路径；③对LLM的推理与生成质量高度依赖，存在编译/执行失败、对象身份不共享等问题；④未对运行时状态多样性（如分支/值覆盖）进行深入探索。

---

## 299. Interactively visualizing biological multilayer networks using MiRA

**arXiv ID:** 2605.09597 | [PDF](https://arxiv.org/pdf/2605.09597v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 300. MolSight: Molecular Property Prediction with Images

**arXiv ID:** 2605.10157 | [PDF](https://arxiv.org/pdf/2605.10157v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 301. Model Capacity Determines Grokking through Competing Memorisation and Generalisation Speeds

**arXiv ID:** 2605.09724 | [PDF](https://arxiv.org/pdf/2605.09724v1)

**作者:** Yiding Song `[一作]` (Harvard), Hanming Ye `[通讯]` (Harvard)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了模型容量如何通过记忆与泛化速度的竞争来决定模组算术任务中的grokking现象

**💡 创新点**

提出用信息理论框架测量记忆速度T_mem与泛化速度T_gen，并发现它们的交点预测grokking起始

**🔧 技术方法**

信息容量估计、随机标签实验、Transformer训练与速度交点分析

**📊 数据集**

模块除法（modular division）数据集，使用不同素数p并取α=1/2

**📈 对比分析**

通过对比训练/验证准确率延迟，验证交点预测与实际grokking一致，误差约30%

**⚠️ 局限性**

仅适用于特定Transformer结构与算法任务，泛化到更大模型或自然任务尚未验证

---

## 302. WISTERIA: Learning Clinical Representations from Noisy Supervision via Multi-View Consistency in Electronic Health Records

**arXiv ID:** 2605.09765 | [PDF](https://arxiv.org/pdf/2605.09765v1)

**作者:** Ruan Dong `[一作]` (University of Science and Technology of China), Shi Li `[通讯]` (Columbia University)

**通讯引用:** 35459 | [OpenAlex ID](https://openalex.org/A5025170020)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了弱监督多视角表示学习框架 WISTERIA，用于从电子健康记录中学习鲁棒临床表征。

**💡 创新点**

将临床标签视为噪声观测的多视角生成过程，采用一致性约束和本体正则化实现隐状态恢复和噪声去除。

**🔧 技术方法**

采用多视角弱监督一致性损失、标签空间图拉普拉斯正则化、Transformer 编码器以及交叉熵/对称 KL 对齐等技术。

**📊 数据集**

使用去标识化多机构 EHR 数据集，并在死亡率、再入院、诊断预测、表型识别等标准基准上进行预训练与下游评估。

**📈 对比分析**

与 Masked LM、AR、Contrastive、Supervised 等基准进行比较，在 AUROC、AUPRC、Macro‑F1 等指标上均取得最高，特别是在弱监督任务和跨机构迁移上表现优异。

**⚠️ 局限性**

依赖多样化的弱监督器和完整的本体知识；若弱监督同质性低或本体不完整则效果受限；假设视角条件独立可能不成立，且需要手工设计监督器。

---

## 303. Useful for Exploration, Risky for Precision: Evaluating AI Tools in Academic Research

**arXiv ID:** 2605.10125 | [PDF](https://arxiv.org/pdf/2605.10125v1)

**作者:** Anthea Dathe `[一作]` (Dresden University of Technology), Aline Mangold `[通讯]` (Dresden University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并应用一种结合人机中心与计算机中心指标的基准框架，用于评估科研用途的AI问答与文献综述工具。

**💡 创新点**

首次将可用性、可解释性、工作流程集成等人机中心维度与传统技术指标统一到同一评测体系，填补了现有基准缺失的用户体验层面。

**🔧 技术方法**

采用大型语言模型（GPT‑4等）与人工评测相结合的方式，设计标准化提示、Likert评分及xAI评估指标。

**📊 数据集**

使用五篇边缘人格障碍相关论文的文档集进行问答评测，使用一组38篇关于情感共情的参考文献集进行文献综述评测。

**📈 对比分析**

通过对比单文档与多文档对话、图像说明、表格/公式提取、意图匹配、无答案判定、跨文档比较与xAI准确度等指标，结果显示问答工具在一致性与摘要上表现良好，但在信息提取与解释方面表现差；文献综述工具来源丰富但可复制性、透明度低，难以用于系统综述。

**⚠️ 局限性**

评测受限于单一评测者导致主观偏差、快速演进的LLM技术可能导致结果过时、评价标准中对“真实值”的定义模糊、提示范围过窄，且未能覆盖更细粒度的质量指标。

---

## 304. ConsistNav: Closing the Action Consistency Gap in Zero-Shot Object Navigation with Semantic Executive Control

**arXiv ID:** 2605.09869 | [PDF](https://arxiv.org/pdf/2605.09869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 305. PruneTIR: Inference-Time Tool Call Pruning for Effective yet Efficient Tool-Integrated Reasoning

**arXiv ID:** 2605.09931 | [PDF](https://arxiv.org/pdf/2605.09931v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 306. Align and Shine: Building High-Quality Sentence-Aligned Corpora for Multilingual Text Simplification

**arXiv ID:** 2605.09476 | [PDF](https://arxiv.org/pdf/2605.09476v1)

**作者:** Kenji Hilasaca `[一作]`, Serge Sharoff `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了以 Wiki‑Vikidia 文档为基础的多语言（Catalan、Spanish、French、Italian、English）句子级对齐语料库，并在此基础上系统评估并比较多种语义嵌入空间（LaBSE、BGE‑M3、SONAR）在文本简化对齐任务中的表现。

**💡 创新点**

首次对多语种文本简化句子对齐任务中不同语义嵌入空间进行系统对比，并公开了首个高质量、可复现的跨语言对齐语料库；同时提出了基于阈值过滤和 Dijkstra 全局优化的 SentAlign 框架，兼顾 1–N、N–1 等非单一对齐模式。

**🔧 技术方法**

使用 SentAlign 混合算法（长度预选 + 语义锚定 + 全局优化）、LaBSE、BGE‑M3、SONAR 嵌入模型、余弦相似度阈值 τ、上限 0.95 过滤，以及自动评估工具（BERTScore、SpaCy 句法深度/NP 密度）。

**📊 数据集**

基于从 Wikipedia 与 Vikidia 公开页面爬取的并行文档，覆盖 5 种语言，构成的原始语料量约 20 万条句子，最终通过阈值过滤后约占 5% 的句子对被保留用于训练与评测。

**📈 对比分析**

通过人工构建的 15 份文档金标准，对精确匹配（Strict）与宽松匹配（Lax）两种评估指标进行 F1 比较；结果显示 LaBSE 在大多数 Romance 语言上获得最高 F1（例如 Catalan 0.645、Spanish 0.422、French 0.469、Italian 0.554），BGE‑M3 在英文上略胜 LaBSE（Strict F1 0.525 vs. 0.480），基线 Gale‑Church 与 Hunalign 在此任务中表现接近零。阈值优化显著提升精确度，宽松评估则揭示 SentAlign 能捕获部分简化操作。

**⚠️ 局限性**

主要限制包括：阈值过高导致召回率低，约 95% 原始句子被丢弃；不同语言对嵌入模型的适应性差异显著（如 BGE‑M3 在 Romance 语种表现不佳）；当前方法仅做句子级对齐，未显式处理词级或子词级细粒度对齐，限制了对细微语义变更的捕捉。

---

## 307. ConFixGS: Learning to Fix Feedforward 3D Gaussian Splatting with Confidence-Aware Diffusion Priors in Driving Scenes

**arXiv ID:** 2605.09688 | [PDF](https://arxiv.org/pdf/2605.09688v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 308. A cell-decomposition based path planner for 3D navigation in constrained workspaces

**arXiv ID:** 2605.10086 | [PDF](https://arxiv.org/pdf/2605.10086v1)

**作者:** João P. L. Morais `[一作]` (Universidade Federal de Minas Gerais), Guilherme V. Raffo `[通讯]` (Universidade Federal de Minas Gerais)

**通讯引用:** 2787 | [OpenAlex ID](https://openalex.org/A5044452531)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出了一种针对3D二值占用网格的细胞分解算法，并将其用于路径规划的优化框架。

**💡 创新点**

创新点在于在分解过程中直接加入完整可见性约束，保证相邻细胞之间完全可见，同时提供了KSP‑SOCP方法，将Yen算法与SOCP结合，兼顾最优性和内存效率。

**🔧 技术方法**

使用了二次锥规划（SOCP）、混合整数二次锥规划（MISOCP）以及基于Yen算法的k‑shortest‑path SOCP搜索。

**📊 数据集**

采用了9个城市风格的随机3D工作空间，网格分辨率为1 m³，L从100到500，H固定为200。

**📈 对比分析**

与Basic Theta*、A*-SOCP、MISOCP比较，KSP‑SOCP在路径长度上略优于A*-SOCP、与MISOCP相当；计算时间与MISOCP相近，但内存消耗显著更低，可处理更大规模。

**⚠️ 局限性**

局限性包括仅适用于轴对齐的盒形障碍、对非常大网格仍需高内存，且MISOCP求解难度高；未来需扩展到一般多面体障碍并进一步提升求解速度。

---

## 309. ChaosNetBench: Benchmarking Spatio-Temporal Graph Neural Networks on Chaotic Lattice Dynamics

**arXiv ID:** 2605.09676 | [PDF](https://arxiv.org/pdf/2605.09676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 310. Sketch-based Access Control: A Multimodal Interface for Translating User Preferences into Intent-Aligned Policies

**arXiv ID:** 2605.10012 | [PDF](https://arxiv.org/pdf/2605.10012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 311. Pretraining large language models with MXFP4

**arXiv ID:** 2605.09825 | [PDF](https://arxiv.org/pdf/2605.09825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 312. VulTriage: Triple-Path Context Augmentation for LLM-Based Vulnerability Detection

**arXiv ID:** 2605.09461 | [PDF](https://arxiv.org/pdf/2605.09461v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 313. INFANiTE: Implicit Neural representation for high-resolution Fetal brain spatio-temporal Atlas learNing from clinical Thick-slicE MRI

**arXiv ID:** 2605.09977 | [PDF](https://arxiv.org/pdf/2605.09977v1)

**作者:** Xiaotian Hu `[一作]` (Beihang University), Qiyuan Tian `[通讯]` (Tsinghua University)

**通讯引用:** 2876 | [OpenAlex ID](https://openalex.org/A5066843175)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

利用隐式神经表示（INR）直接从临床厚切 MRI 栈构建高分辨率胎儿脑时空全景图谱，避免传统的切片到体积重建（SVR）和迭代非刚性配准步骤。

**💡 创新点**

创新点包括：①在 INR 前向模型中加入物理仿真的点扩散函数（PSF）以模拟切片模糊，提升细节重建；②使用空间加权优化目标强调真实采样体素，降低插值误差；③采用单张切片注册到模板的鲁棒方法，显著缩短配准时间；④通过高斯核回归聚合时间特定潜在码实现连续时空亚层谱生成。

**🔧 技术方法**

核心技术：隐式神经网络（基于 SIREN 的多层感知机）、物理仿真 PSF 模型、空间加权损失、切片到模板的刚性注册、Gaussian kernel regression 用于时间潜在码聚合。

**📊 数据集**

数据集：615 张 2D T2‑加厚切 MRI（来自 205 名孕妇，21–36 周），构成 525 堆栈的多堆栈数据集；单堆栈数据集随机挑选每位受试者一张；测试集 90 堆栈来自 30 位受试者；使用 NiftyMIC 生成高分辨率体积并通过可信 AI 框架进行分割，构成伪真实参考图谱用于评估。

**📈 对比分析**

与 Deepali、Atlas‑GAN、SyGN、Aladdin、CINeMA 等基线方法在多堆栈和单堆栈场景下进行比较。INFANiTE 在 HD95、ASD、DSC、PSNR/SSIM 等多项指标上均取得最优或接近最佳成绩，尤其在参考逼真度（PSNR_a、SSIM_a）和生物学可解释性（与标准发育轨迹的 L1 误差）上表现突出。整体端到端处理时间从传统方案的数天降至数小时，显著提升可扩展性。

**⚠️ 局限性**

局限性：①依赖较高质量的加厚切扫描，对低信噪比或极端运动伪影的鲁棒性尚未充分验证；②目前仅在单一扫描序列（T2‑TSE）上评估，跨扫描协议或不同磁共振系统的泛化能力待进一步验证；③模型训练仍需大量标注（分割）数据，可能受限于 AI 分割器的准确性；④对极少量样本或极端孕周的表现尚未系统评估。

---

## 314. The Value of Mechanistic Priors in Sequential Decision Making

**arXiv ID:** 2605.10018 | [PDF](https://arxiv.org/pdf/2605.10018v1)

**作者:** Itai Shufaro `[一作]` (Technion), Shie Mannor `[通讯]` (Technion)

**通讯引用:** 18580 | [OpenAlex ID](https://openalex.org/A5036260775)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并验证了利用混合机理模型（物理先验 + 学习残差）在顺序决策中减少样本量的理论与实践框架。

**💡 创新点**

创新点在于引入“机制信息”（mechanistic information）量化模型先验价值，并给出可预先计算的关键偏差阈值 (critical bias)，同时给出渐近与燃烧期的下界和上界，证明物理先验可在临床中显著降低剂量试验周期。

**🔧 技术方法**

技术：信息理论与贝叶斯回溯退火（Thompson Sampling）结合，假设 K‑armed bandit 模型；采用占用加权敏感度与高斯过程残差；通过极限与有限样本分析得出下界/上界；模拟 5‑FU 给药。

**📊 数据集**

使用文献公开的 5‑FU 及 FOLFOX 药代动力学数据进行模型校准（BSA、AUC 分布等）。

**📈 对比分析**

与无先验 Thompson Sampling、固定 BSA 给药以及不同信息量的混合先验进行比较；在 12 次循环中，信息量从 0 到 1.9 nats，适应性增益从 1.0×到 19.5×，临床增益从 1.32×到 25.7×；在大样本下 200 次循环，增益降至 6×，与理论下界 3.4×相符。

**⚠️ 局限性**

限制在于只考虑离散 K‑armed 框架，未处理连续动作；对动态不稳定性及非均匀先验的扩展有限；LLM 先验的保持阈值保守；模拟基于仿真而非真实患者数据，临床验证待进一步研究。

---

## 315. Ambig-DS: A Benchmark for Task-Framing Ambiguity in Data-Science Agents

**arXiv ID:** 2605.09698 | [PDF](https://arxiv.org/pdf/2605.09698v1)

**作者:** Josefa Lia Stoisser `[一作]` (Novo Nordisk), Robert Kitchen `[通讯]` (Novo Nordisk)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Ambig-DS 两套诊断任务（Ambig-DS-Target 与 Ambig-DS-Objective），评估数据科学代理在任务框架模糊（预测目标与评估目标）下的误判，并提供澄清 oracle 以测试 ask‑act 机制。

**💡 创新点**

创新点：①将任务框架模糊拆解为可量化的两变量（目标与指标）；②公开两套独立、经过人工+LLM 验证的模糊任务集；③在原始 benchmark 评估器不变的前提下生成模糊变体；④系统评估代理的 ask‑act 校准与澄清恢复能力。

**🔧 技术方法**

使用技术：OpenCode 代码代理框架、Claude、Gemini、GPT LLM 生成与校验、人工审核、LLM 判定器、Wilcoxon 检验、Bootstrap 置信区间、提示工程与澄清 oracle 交互。

**📊 数据集**

数据集：DSBench 的 51 个任务（用于目标模糊）与 MLE-bench 的 61 个任务（用于指标模糊），均为 Kaggle‑style 预测任务。

**📈 对比分析**

比较方法：对每个代理在 Full、Ambig、Ask 三种条件下的标准化分数进行统计；Ambig 条件导致性能下降（Δ_ambig<0），Ask 条件显著恢复（Δ_ask>0），差异通过 Wilcoxon p‑值与 Bootstrap CI 验证；ask‑act 校准表现不一，未能完全消除误判。

**⚠️ 局限性**

局限性：①仅覆盖预测目标与评估指标两类框架模糊；②构造与原 benchmark 绑定，缺乏跨任务通用性；③LLM 生成/校验可能引入偏差；④澄清 oracle 理想化，现实中可能不具备；⑤任务量有限，未能全面反映所有数据科学场景；⑥无法完全区分未识别与默许提交的差异。

---

## 316. TRACER: Verifiable Generative Provenance for Multimodal Tool-Using Agents

**arXiv ID:** 2605.09934 | [PDF](https://arxiv.org/pdf/2605.09934v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 317. A Comparative Study of Machine Learning and Deep Learning for Out-of-Distribution Detection

**arXiv ID:** 2605.10181 | [PDF](https://arxiv.org/pdf/2605.10181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 318. A Deductive Refinement Calculus for Differential-Algebraic Programs

**arXiv ID:** 2605.10188 | [PDF](https://arxiv.org/pdf/2605.10188v1)

**作者:** Jonathan Hellwig `[一作]` (Karlsruhe Institute of Technology), André Platzer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 5643 | [OpenAlex ID](https://openalex.org/A5080481427)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种针对差分-代数程序（DAP）的演绎推理框架——差分代数细化逻辑（DARL），通过可追踪的 C^1 轨迹语义实现了 DAE 的细化推导与安全性证明。

**💡 创新点**

创新点在于：①引入了基于轨迹的 C^1 正则化语义，扩展了传统可解析语义；②构造了完整的细化算子与证明公理体系；③将指数归约（index reduction）形式化为细化关系，并证明其完备性；④引入“虚拟变量”与“差分虚拟”公理，支持在保持正则性的前提下对 DAE 进行扩展与简化。

**🔧 技术方法**

使用了差分动态逻辑（differential dynamic logic）框架、轨迹语义、细化算子、公理化证明、差分代数与隐式约束推导、线性代数（Jacobian、行列式）以及符号计算技巧。

**📊 数据集**

论文主要为理论工作，未使用任何实验数据集或数值案例，仅以经典示例（欧几里得摆、Coulomb 摩擦等）说明方法的适用性。

**📈 对比分析**

方法与传统的基于可达性或数值模拟的指数归约相比，提供了形式化的推理路径和可审计的证明；但论文中未给出性能评估或自动化工具的实现，只说明了理论上的完备性与正确性。

**⚠️ 局限性**

局限性包括：①需要轨迹为 C^1；②对初始条件的隐式约束要求较高，需手动证明或引入辅助变量；③目前仅为理论框架，缺乏完整的自动化证明器；④复杂 DAE 的手动细化步骤仍可能繁琐。

---

## 319. End-to-End Keyword Spotting on FPGA Using Graph Neural Networks with a Neuromorphic Auditory Sensor

**arXiv ID:** 2605.09570 | [PDF](https://arxiv.org/pdf/2605.09570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 320. Consolidation-Expansion Operator Mechanics:A Unified Framework for Adaptive Learning

**arXiv ID:** 2605.09968 | [PDF](https://arxiv.org/pdf/2605.09968v1)

**作者:** Debashis Guha `[一作]` `[通讯]` (Big Sky Quantitative Research LLP), Debashis Guha (Big Sky Quantitative Research LLP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6`

**🎯 论文内容**

提出了一种统一的“巩固‑扩展算子机理”（OpMech）框架，用来刻画自适应学习系统中巩固（内部更新）与扩展（外部证据）操作的相互作用，并通过“阶差”（order‑gap）这一可观测量来实时控制学习策略。

**💡 创新点**

创新点包括：①将学习过程建模为由两种非可交换算子 Q（巩固）和 P_e（扩展）组成的马尔可夫动力学；②引入阶差作为衡量这两操作顺序影响的指标；③证明阶差在收敛轨迹上几何衰减，且当阶差持续较大时可被用作子最优性信号；④基于阶差给出噪声鲁棒的停止规则，并在无噪声与有界噪声两种场景下提供显式收敛与停止保证；⑤在连续五大领域（多臂赌博、强化学习、随机梯度下降、持续学习、递归语言模型）验证框架，给出具体的设计指引。

**🔧 技术方法**

核心技术包括：随机动力学与Banach空间分析；算子非可交换性的张量/雅可比矩阵求解；阶差与期望梯度或期望更新的“交换子”(commutator)关系；矩阵秩条件（第一阶与第二阶矩）来判定阶差的敏感性；马尔可夫差分与Azuma–Hoeffding等概率不等式；以及针对有状态抽样的条件矩阵方法。

**📊 数据集**

本文并未在单一数据集上进行实验验证；相反，框架在理论上适用于多臂赌博、强化学习、随机梯度下降、持续学习和递归语言模型等通用环境；递归语言模型的实验验证将在随附的伴随论文中给出。

**📈 对比分析**

比较方法主要是对已有的基线算法（如 ε‑贪婪、UCB、Sarsa、Actor‑Critic、Adam、EWC 等）在其原始超参数设定与基于阶差的自适应控制（探索率、目标网络更新、学习率、正则化强度、递归深度）之间进行对比；性能评价以理论收敛速度、停止时间、子最优性误差和计算成本等指标进行量化。具体数值和实验结果未在本文给出，但框架提供了可实现的停止与调度策略，并给出在噪声有限时的子最优性上界。

**⚠️ 局限性**

局限性包括：①对梯度下降类方法的阶差解释较为简化，无法直接给出实用的学习率调度策略；②阶差的敏感性判定需要求解算子雅可比的秩，实际应用中可能较难验证；③在存在系统性噪声时，停止阈值的理论上限与实际表现仍存在误差；④框架假设算子满足Lipschitz与收敛条件，在某些深度学习或非线性系统中可能不成立；⑤实验验证仅在递归语言模型上完成，其他领域仍缺乏大规模实证验证。

---

## 321. Continual Harness: Online Adaptation for Self-Improving Foundation Agents

**arXiv ID:** 2605.09998 | [PDF](https://arxiv.org/pdf/2605.09998v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 322. Streaming Complexity Separations for Dense and Sparse Graphs

**arXiv ID:** 2605.09814 | [PDF](https://arxiv.org/pdf/2605.09814v1)

**作者:** Yang P. Liu `[一作]` (Carnegie Mellon University), David P. Woodruff `[通讯]` (Carnegie Mellon University)

**通讯引用:** 18965 | [OpenAlex ID](https://openalex.org/A5035813243)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对最大割、最密子图以及常数arity CSP 等经典图优化问题，在单次插入式流模型下研究输出近似解（而非仅近似值）的空间复杂度，并给出上界与下界。

**💡 创新点**

发现稠密与稀疏图在输出近似解时的空间复杂度存在明显分离：稠密图可用 O(n/ε²) 空间，而稀疏图则需要 Ω(n·log(nε²)/ε²) 空间；同时证明了这些界限在整个 ε 范围内都是紧匹配的。

**🔧 技术方法**

主要技术包括：信息论的单向通信模型分析、F₀ 估计算法的子样本化技巧、构造大规模低重叠稀疏图族、利用条件最大割与完整最大割之间的约束转化、以及对随机/确定性算法的随机化/确定化变换。

**📊 数据集**

由于是理论研究，未使用任何实验数据集，全部结果均为上界/下界证明。

**📈 对比分析**

与以往仅给出近似值的流算法相比，本工作在输出完整近似割时实现了最优空间复杂度；上界与下界在稠密/稀疏两种场景下均已匹配，证明了结果的紧迫性。

**⚠️ 局限性**

局限性包括：仅针对单次插入式（单通道）流；只能得到 (1‑ε) 近似（不支持更高精度的近似或多通道流）；仅针对无向图且主要聚焦于最大割和最密子图，其他 NP‑难问题的空间复杂度仍不明。

---

## 323. Exploration-Driven Optimization for Test-Time Large Language Model Reasoning

**arXiv ID:** 2605.09853 | [PDF](https://arxiv.org/pdf/2605.09853v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 324. When Prompts Become Payloads: A Framework for Mitigating SQL Injection Attacks in Large Language Model-Driven Applications

**arXiv ID:** 2605.10176 | [PDF](https://arxiv.org/pdf/2605.10176v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 325. EchoPrune: Interpreting Redundancy as Temporal Echoes for Efficient VideoLLMs

**arXiv ID:** 2605.10050 | [PDF](https://arxiv.org/pdf/2605.10050v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 326. Edge-Cloud Collaborative Pothole Detection via Onboard Event Screening and Federated Temporal Segmentation

**arXiv ID:** 2605.10055 | [PDF](https://arxiv.org/pdf/2605.10055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 327. Group Vitality Indices: Axioms and Algorithms

**arXiv ID:** 2605.09791 | [PDF](https://arxiv.org/pdf/2605.09791v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 328. Deterministically finding an element of large order in $\mathbb{Z}_N^*$

**arXiv ID:** 2605.09592 | [PDF](https://arxiv.org/pdf/2605.09592v1)

**作者:** Itamar Nir `[一作]` `[通讯]`, Itamar Nir

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出了一种确定性算法，用于在整数 N 的模乘群中寻找阶大于给定阈值 D 的元素，若不存在则返回 N 的非平凡因子，若 N 为质数则报告 N 为质数。

**💡 创新点**

核心创新在于：①在 D > exp(√(2 log N log log N)) 的条件下实现了与随机方法同等的 O(D^1/2+o(1)) 复杂度的完全确定性算法；②利用最小公倍数 M 与平滑数估计，提出了更简洁的搜索策略；③给出了适用于任意 D 的简化算法，时间复杂度为 O(D^2.5+o(1)polylog N)。

**🔧 技术方法**

主要技术手段包括：Sutherland 的上界阶数计算算法、Pomerance 的平滑数下界估计、利用素数 p 满足 p≡1 (mod M) 的结构、以及判定 a 的阶是否在每个质因子模中保持不变的判据；整个算法在确定性框架下结合了这些数论工具。

**📊 数据集**

该工作为理论研究，未使用任何实验数据集；所有结果均为纯数学证明。

**📈 对比分析**

与先前 Oznovich–Volk 算法（仅适用于 D > N^1/6）相比，本算法在更宽松的 D 范围内保持相同的 O(D^1/2+o(1)) 复杂度；与 Harvey–Hittmeir 的并行工作相比，算法实现更简洁，虽然范围略窄；简化版算法在 D 较小的情形下提供了可接受的 O(D^2.5) 复杂度。

**⚠️ 局限性**

局限性包括：①要求 D > exp(√(2 log N log log N))；②对极大 N 仍可能面临高时间消耗；③简化版在 D 较大时效率不如主算法；④所有证明均基于平滑数估计，若该估计不成立则方法失效。

---

## 329. Synthetic Pre-Pre-Training Improves Language Model Robustness to Noisy Pre-Training Data

**arXiv ID:** 2605.10129 | [PDF](https://arxiv.org/pdf/2605.10129v1)

**作者:** Xu Guo `[一作]` (Shanghai AI Laboratory), Qipeng Guo `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在LLM预训练前引入轻量级的预预训练（PPT）阶段，使用随机RNN生成的合成序列来提升模型在含噪声的预训练过程中的鲁棒性。

**💡 创新点**

创新点在于：①提出基于随机RNN集合的合成PPT任务，并证明其能在多种噪声设置下显著提升鲁棒性；②通过消融实验提炼出“可学习、低偏差、全词表”三条设计原则；③通过注意力分析和metamer对照实验揭示PPT通过抑制噪声自建模而非直接降低噪声注意力来实现鲁棒性提升。

**🔧 技术方法**

技术包括：RNN生成合成数据（多生成器、全词表）、两阶段训练（PPT + PT）、多种噪声注入（样本级、token置换、span破坏）、注意力自噪声建模度量、metamer对照实验、下游任务LAMBADA评估。

**📊 数据集**

使用的主要数据集有：C4（干净基准），FineWeb（自然噪声），以及PPT阶段的合成序列（随机RNN、Dyck、随机token）。

**📈 对比分析**

与无PPT、随机PPT、Dyck PPT进行对比。实验在160M和1B规模下表明，RNN-PPT在所有噪声强度下都能降低最终验证损失，尤其在高噪声时提升显著；在1B模型下，65M token的RNN-PPT可让模型使用最多49%更少的自然文本PT token就达到与基线相同的损失；下游LAMBADA任务也获得小幅提升。

**⚠️ 局限性**

局限性包括：①评估合成源需要完整跑PPT→PT流程，成本高；②实验规模停留在1B和25K PT步，缺乏更大规模或更长训练的验证；③仅使用RNN作为序列生成器，未验证其他架构（如LSTM、状态空间模型）的通用性；④缺乏理论解释或规模定律，尚未探索PPT对模型尺寸的可伸缩性。

---

## 330. Security Risks in Tool-Enabled AI Agents: A Systematic Analysis of Privileged Execution Environments

**arXiv ID:** 2605.09721 | [PDF](https://arxiv.org/pdf/2605.09721v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 331. Accelerating Power Method with Fast Sketching for Stronger Low-Rank Approximation

**arXiv ID:** 2605.09755 | [PDF](https://arxiv.org/pdf/2605.09755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 332. Understanding Robust Catalytic Computing

**arXiv ID:** 2605.09648 | [PDF](https://arxiv.org/pdf/2605.09648v1)

**作者:** Michal Koucký `[一作]` (Charles University), Sasha Sami `[通讯]` (Charles University)

**通讯引用:** 95 | [OpenAlex ID](https://openalex.org/A5013139842)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文系统研究了失效催化计算（lossy catalytic computing）的三种主要变体（scδϵ、sce、sce），并给出了它们在 logspace 以及多项式时间限制下的近乎完整的复杂性分类，展示了这些类与传统复杂度类（如 logspace、BPP、PP 等）的关系及在假设 derandomization 时的完全崩塌。

**💡 创新点**

创新点在于：
1) 将随机破坏和期望错误的概念引入催化计算，提出三种新的错误模型；
2) 通过配置图（configuration graph）与压缩（compress‑or‑random）技术，完成了对这些类的全面表征；
3) 在 derandomization 假设下证明所有新类（除了 δϵ_ 的特殊情况）收敛到传统的 logspace 甚至 BPP；
4) 进一步探讨了错误容忍度与随机性参数的权衡与极限。

**🔧 技术方法**

主要技术手段包括：
- 配置图分析与平均/Markov 论证；
- 采用压缩/随机化压缩技术（compress‑or‑random）对催化记忆进行处理；
- 利用 BCH 码等纠错编码恢复催化盘；
- 应用伪随机生成器（PRG）和 derandomization 结果（Impagliazzo‑Wigderson 构造）来消除随机性；
- 通过多次运行、Chernoff bound 和大数定理提升成功概率。

**📊 数据集**

无实验数据集，本文完全为理论计算复杂度研究。

**📈 对比分析**

方法比较采用理论复杂度类之间的包含关系与等价性分析，性能用包含/等价关系来衡量。结果表明：
- 在无时间限制时，scδϵ 与 δϵ_ 通过参数阈值分成三种情况；
- 在多项式时间限制下，scδϵ 对应 BPP 或 BPP^log 等；
- 在 derandomization 假设下，几乎所有新类收敛到 logspace 或 BPP，性能可视为完全等价。

**⚠️ 局限性**

局限与未解决问题：
1) 对于 δϵ_ 与 O(1) 的关系尚未完全证明，尤其是 δ1/2 的精确等价；
2) 需要 derandomization 假设来实现完全崩塌；
3) 某些错误容忍度参数（如期望错误对所有初始催化盘）下的完整分类仍未完成；
4) 对高阶错误模型（例如同时考虑随机性和初始盘期望）存在的潜在细分未被完全探究。

---

## 333. jNO: A JAX Library for Neural Operator and Foundation Model Training

**arXiv ID:** 2605.10159 | [PDF](https://arxiv.org/pdf/2605.10159v1)

**作者:** Leon Armbruster `[一作]` (Fraunhofer Institute for Integrated Systems and Device Technology IISB), Christopher Straub `[通讯]` (Fraunhofer Institute for Integrated Systems and Device Technology IISB)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个统一的 JAX‑native 框架 jNO，支持神经算子、PDE 基础模型、数据驱动与物理约束的混合训练，以及网格、FEM 等多种求解方式；

**💡 创新点**

创新点在于将域、模型、残差、损失等全部写入同一符号 DSL，并通过统一追踪系统实现一次编译、一次 XLA 执行，支持多模型组合、参数级微调、LoRA 适配，并把多种 PDE 基础模型整合进同一 JAX 生态；

**🔧 技术方法**

使用 JAX、XLA、Equinox、Optax、FEAX、Diffrax、PyGmsh、LoRA、Nevergrad 等技术构建符号 DSL、网格处理、弱形式组装、训练控制与超参数搜索；

**📊 数据集**

未公开使用特定实验数据集，主要在合成 PDE 数据集与已翻译的预训练基础模型（如 Poseidon、Walrus 等）上进行演示；

**📈 对比分析**

与 DeepXDE、JAX‑PI、NVIDIA PhysicsNeMo、NeuralPDE.jl 等框架对比，jNO 在统一 API、支持基础模型、JAX‑native 性能（单 XLA 图、子表达式消除、多设备并行）方面表现优异；具体数值实验未给出；

**⚠️ 局限性**

局限性包括：仍依赖外部翻译模型，生态尚不成熟；缺乏大规模真实案例验证；对部分非 JAX 传统工具的集成仍需完善；

---

## 334. SABER: A Scalable Action-Based Embodied Dataset for Real-World VLA Adaptation

**arXiv ID:** 2605.09613 | [PDF](https://arxiv.org/pdf/2605.09613v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 335. ERASE: Eliminating Redundant Visual Tokens via Adaptive Two-Stage Token Pruning

**arXiv ID:** 2605.09982 | [PDF](https://arxiv.org/pdf/2605.09982v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 336. CrossVL: Complexity-Aware Feature Routing and Paired Curriculum for Cross-View Vision-Language Detection

**arXiv ID:** 2605.09802 | [PDF](https://arxiv.org/pdf/2605.09802v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 337. Initiation of Interaction Detection Framework using a Nonverbal Cue for Human-Robot Interaction

**arXiv ID:** 2605.10087 | [PDF](https://arxiv.org/pdf/2605.10087v1)

**作者:** Guhnoo Yun `[一作]` (Korea Institute of Science and Technology), Dong Hwan Kim `[通讯]` (Korea Institute of Science and Technology)

**通讯引用:** 24649 | [OpenAlex ID](https://openalex.org/A5100370734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个无需热词的发起交互(IoI)检测框架，利用音频与视觉传感器融合并结合状态转换模型，实现机器人在家庭环境中检测用户是否想与其互动。

**💡 创新点**

创新点：1）通过声音来源定位与视觉跟踪联合定位说话者，消除了传统热词依赖；2）使用头姿态检测来替代眼球追踪，简化硬件需求；3）构建双路径状态机（语音+视觉和单视觉）以提升在无声交互场景下的鲁棒性。

**🔧 技术方法**

技术：ROS框架；HARK音频定位（MUSIC方法）；Azure Kinect（RGB、深度、7麦克风阵列）；YOLOv7+DeepSort进行人物检测与跟踪；MediaPipe头姿态估计进行面向检测；状态机实现IoI判定逻辑。

**📊 数据集**

数据集：实验室/家庭环境下的实时数据，使用两名参与者进行坐、站、行走等自然动作，录制音视频；未使用公开公开数据集，而是基于实验收集的自定义数据。

**📈 对比分析**

比较方法：将仅音视结合（AV‑IoI）与完整方案（Full‑IoI）在同一数据上进行精确度、召回率和F1分数评估；Full‑IoI在精确度82.35%→86.36%、召回率70%→95%、F1从75.68%提升至90.48%。

**⚠️ 局限性**

局限性：1）低音量或远距离说话时声音定位失效；2）面部检测失败导致IoI检测漏报；3）背景噪声（如电视、广播）仍会引入误定位；4）目前仅实现了面向检测，未真正实现细粒度眼球注视跟踪。

---

## 338. Yield Curve Forecasting using Machine Learning and Econometrics: A Comparative Analysis

**arXiv ID:** 2605.09842 | [PDF](https://arxiv.org/pdf/2605.09842v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 339. LLMs are the Ideal Candidate for Mixed-Initiative Game Design Pillar Workflows

**arXiv ID:** 2605.09767 | [PDF](https://arxiv.org/pdf/2605.09767v1)

**作者:** Julian Geheeb `[一作]` (Technical University of Munich), Georg Groh `[通讯]` (Technical University of Munich)

**通讯引用:** 6750 | [OpenAlex ID](https://openalex.org/A5004398345)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了游戏设计支柱的正式定义，并基于大语言模型（LLM）实现了混合主动工具SPINE，用以支持支柱的创建与决策；通过游戏创作马拉松的案例研究和四位专业人士访谈进行评估。

**💡 创新点**

创新点在于：①首次为游戏设计支柱提供形式化定义与结构化文档；②设计并实现首个基于LLM的混合主动支柱工作流原型SPINE；③提出多维LLM功能（结构分析、修复、集合验证、特性评估）来辅助支柱使用；④构建了55+真实游戏支柱实例数据集。

**🔧 技术方法**

使用技术包括：大语言模型（Gemini、GPT等）与提示工程；后端采用Django，前端使用Nuxt4 + NuxtUI；LLM接口通过API调用实现结构分析、修复、集合验证与特性评估。

**📊 数据集**

使用的数据集包括：55+真实游戏设计支柱实例；为支柱创建评估而构造的标注数据集；以及用于预实验模型比较的对照数据集。

**📈 对比分析**

比较方法：先进行预实验比较Gemini与其他LLM在支柱生成任务上的多样性与一致性；随后在标注数据集上评估SPINE的支柱生成性能；在游戏创作马拉松与专家访谈中收集定性反馈。实验表明Gemini在输出多样性和一致性方面优于其它模型，SPINE在早期设计阶段受到积极评价，但在决策支持和一致性方面仍有提升空间。

**⚠️ 局限性**

局限性包括：LLM输出质量波动大、缺乏一致性；对模型可解释性与透明度需求高；决策支持功能使用频率低、实用性不足；评估样本规模小、缺乏量化指标；缺乏针对游戏设计领域的细粒度微调与上下文检索机制。

---

## 340. Cross-Domain Lossy Compression via Constrained Minimum Entropy Coupling

**arXiv ID:** 2605.09833 | [PDF](https://arxiv.org/pdf/2605.09833v1)

**作者:** Nam Nguyen `[一作]` (Oregon State University), Bella Bose `[通讯]` (Oregon State University)

**通讯引用:** 1092 | [OpenAlex ID](https://openalex.org/A5108277072)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究跨域有损压缩，利用最小熵耦合(MEC)并加入速率与分类约束，目标是最大化源与重构间的互信息，同时满足预设重构分布和下游分类信息的约束。

**💡 创新点**

提出基于对数损失的MEC框架，并证明在共享随机变量下可去除中间表示得到等价的确定性耦合；给出伯努利源的闭式解并扩展到含分类约束的情况；结合神经恢复技术实现分布匹配与分类正则化。

**🔧 技术方法**

信息理论中的最小熵耦合、对数损失、互信息下界、共享随机、WGAN分布匹配、量化与熵建模、变分下界、神经自编码器和分类交叉熵正则化。

**📊 数据集**

MNIST（4×超分辨率）和SVHN（高斯噪声去噪）数据集。

**📈 对比分析**

与传统基于MSE/PSNR等点误差的压缩方法对比，实验显示速率越高分类准确率越高、交叉熵越低，重建图像信息更丰富，证明MEC框架在保持目标分布且任务感知方面优于传统方法。

**⚠️ 局限性**

局限性包括：MEC在一般情况下为NP‑hard，本文仅给出伯努利源的闭式解；对低层视觉属性（如颜色一致性）的匹配依赖WGAN，易出现伪影；共享随机实现受限，需进一步推广到更复杂分布。

---

## 341. Key-Value Means

**arXiv ID:** 2605.09877 | [PDF](https://arxiv.org/pdf/2605.09877v1)

**作者:** Daniel Goldstein `[一作]` (Recursal AI), Eugene Cheah `[通讯]` (Recursal AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为Key-Value Means（KVM）的注意力机制，将块滑动窗口注意力与可扩展压缩状态整合在单层softmax中，既可保持固定状态实现O(N)块递归，也可让状态按需增长实现子线性内存扩展。

**💡 创新点**

提出块递归softmax注意力、基于余弦相似度的winner-take-all合并规则、动态状态扩展策略、即时键值归一化以及部分RoPE零化等多项技术，使得模型在保持传统Transformer优势的同时，兼具LRNN的常数/子线性时间与内存复杂度。

**🔧 技术方法**

使用传统softmax注意力、块滑动窗口（BSWA）、键值压缩与归一化、可增长状态缓冲、Merge Gate、JIT归一化、部分RoPE、温度自适应softmax、GPTAlpha-2骨干网络、混合LRNN层等技术。

**📊 数据集**

训练使用Prolong（8k上下文）数据集；评估使用TextbookChapters、RULER、LongBench、NIAH-S等长文本基准。

**📈 对比分析**

与全注意力GPTAlpha-2、BSWA、RWKV-7、OVQ/SWA等模型在4k/8k/16k/32k上下文长度下对比，KVM-256在8k以上位置表现与全注意力相近，KVM-sqrt在更长上下文中甚至超过其他模型；在短上下文基准上保持与标准Transformer相当的性能。

**⚠️ 局限性**

仍需手工设定状态增长计划，RoPE兼容性有限；固定状态KVM在极长上下文中受限；缺乏动态或数据驱动的状态调度；未尝试多层蒸馏或其他KV缓存优化。

---

## 342. Benchmarking Safety Risks of Knowledge-Intensive Reasoning under Malicious Knowledge Editing

**arXiv ID:** 2605.10146 | [PDF](https://arxiv.org/pdf/2605.10146v1)

**作者:** Qinghua Mao `[一作]` (Shanghai Jiao Tong University), Yuliang Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 50265 | [OpenAlex ID](https://openalex.org/A5100361915)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了恶意知识编辑对LLM推理安全的影响，并提出了统一评估基准EditRisk-Bench。

**💡 创新点**

创新点在于构建涵盖误信息、偏见与安全违规三类风险的统一评估框架，并系统量化其在单步与多步推理中的表现。

**🔧 技术方法**

采用参数编辑与上下文编辑两类技术（如ROME、MEMIT、IKE、WISE等），结合多步推理任务与安全评估指标。

**📊 数据集**

利用RippleEdits、MQuAKE-CF/T、EditAttack、BehaviorBench等多源数据集，并统一格式化为统一评估格式。

**📈 对比分析**

通过对比攻击成功率、推理准确率、通用知识/推理能力保持率等指标，发现大多数LLM在单步攻击中可达100%成功率，且对多步推理效果显著下降。

**⚠️ 局限性**

局限在于缺乏跨模型通用检测与逆向恢复方案，且实验主要聚焦已知编辑技术，未覆盖更广泛的攻击范式。

---

## 343. The Geometric Wall: Manifold Structure Predicts Layerwise Sparse Autoencoder Scaling Laws

**arXiv ID:** 2605.09887 | [PDF](https://arxiv.org/pdf/2605.09887v1)

**作者:** Eslam Zaher `[一作]` (ARC Training Centre for Information Resilience), Fred Roosta `[通讯]` (ARC Training Centre for Information Resilience)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Gemma 2 2B 与 9B 两个语言模型的所有层级的稀疏自编码器（SAE）的重构误差随字典宽度的跨层级缩放规律，并发现该规律与激活流的几何特征（内在维度、曲率、切空间变化、异质性）密切相关。

**💡 创新点**

创新点在于提出“几何墙（geometric wall）”概念，即 SAE 的可恢复误差下限与层级激活流曲率及内在维度共同决定；进一步通过两阶段回归证明层级宽度缩放指数可由四个几何量预测，并且该几何映射在 2B 与 9B 之间可迁移，说明存在可迁移的几何缩放法则。

**🔧 技术方法**

技术手段包括：1) 对每层 SAE 进行无下限（no‑floor）与有下限（with‑floor）缩放曲线拟合；2) 使用拉普拉斯-拉氏变换的 Fisher 信息几何构造激活流的“拉回”度量；3) 通过两近邻、PCA 残差、Gauss‑map、点维度离散等方法估计内在维度、曲率、切空间变化、异质性；4) 对拟合参数与几何量进行 OLS 回归，使用留一/二/三层交叉验证、AIC/BIC、F‑检验及层级置换检验；5) 采用多尺度曲率和内在维度作为主预测变量。

**📊 数据集**

数据集为公开发布的 Gemma Scope JumpReLU SAE 检查点（共 844 个，覆盖 Gemma 2 2B 的 26 层和 9B 的 42 层），使用 Colossal Clean Crawled Corpus (C4) 的 0–5K 文档做几何估计，5–10K 文档做重构误差评估；误差评估还跨数据集到 WikiText‑103 以验证鲁棒性。

**📈 对比分析**

与 GPT‑4 单层缩放法则对比：在 5/6 层深度处的宽度缩放指数与 GPT‑4 近似一致；在所有层级的回归中，多尺度曲率单变量已解释约 92% 的方差，四变量回归 R² 约 94%；跨模型迁移实验中，使用 2B 的回归系数预测 9B 的宽度指数，误差仅在 0.02 以内，说明映射高度可迁移。总体性能表明层级缩放指数受几何限制，且可通过几何量预测。

**⚠️ 局限性**

局限性包括：1) 仅研究 Gemma 2 的残差流，未验证其他 Transformer 架构或不同自编码器变体；2) 仅有两种字典宽度的基础网格，导致大多数层级的有下限曲线不可识别；3) 几何量使用的是欧氏外在估计，可能无法完全捕捉拉回 Fisher 信息几何的真实特性；4) 结果主要基于公开的 JumpReLU SAE，其他激活函数或正则化可能导致不同行为。

---

## 344. Skill Description Deception Attack against Task Routing in Internet of Agents

**arXiv ID:** 2605.09889 | [PDF](https://arxiv.org/pdf/2605.09889v1)

**作者:** Jiayi He `[一作]` (Guangdong University of Technology), Dong In Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 25116 | [OpenAlex ID](https://openalex.org/A5022649488)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在互联代理（IoA）中，恶意代理通过篡改自声明的技能描述来欺骗语义路由，从而被选中执行任务，提出了技能描述欺骗（SDD）攻击模型并实现了基于大语言模型的自动生成与优化框架。

**💡 创新点**

首次正式定义并系统化评估了IoA中的技能描述欺骗攻击；提出了使用LLM自动生成、迭代优化欺骗性描述的方法，并证明其可在多种语义路由机制下获得极高的成功率，揭示了语义匹配在IoA路由中的根本安全缺陷。

**🔧 技术方法**

采用大语言模型（LLM）进行查询生成、描述总结与语义重写；使用BM25、E5、BGE、Qwen Embedding 8B、BCE等检索/嵌入模型对代理技能描述与用户查询进行语义相似度计算；通过迭代优化损失函数提升欺骗描述的路由优势。

**📊 数据集**

利用LiveMCPBench（527个真实工具/代理）构建代理池；采用MMLU基准（9个领域共900个任务）生成用户查询集合，用于评估路由与攻击效果。

**📈 对比分析**

将SDD攻击与四种启发式技能描述操纵策略（夸大功能、关键词堆砌、通用描述、代理冒充）在五种路由机制上进行对比；通过攻击成功率（ASR）、Hit@3/Hit@5、平均排名（MR）等指标评估。实验表明，SDD在所有路由器上均达成最高ASR（最高达98%），Hit@3/Hit@5也显著高于对照组，平均排名靠前，表明攻击效果显著且普适。

**⚠️ 局限性**

实验仅考虑单个恶意代理，未涵盖多恶意或协同攻击；评估在模拟环境下进行，缺乏真实网络部署验证；未提出具体防御机制，且对不同规模或动态注册场景的鲁棒性未知。

---

## 345. SmartEval: A Benchmark for Evaluating LLM-Generated Smart Contracts from Natural Language Specifications

**arXiv ID:** 2605.09610 | [PDF](https://arxiv.org/pdf/2605.09610v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 346. Evaluating Transit Accessibility to Education and Effects of Operational Delays in Japanese Regional Cities: A Case Study of Matsumoto City

**arXiv ID:** 2605.09467 | [PDF](https://arxiv.org/pdf/2605.09467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 347. Instruction Adherence in Coding Agent Configuration Files: A Factorial Study of Four File-Structure Variables

**arXiv ID:** 2605.10039 | [PDF](https://arxiv.org/pdf/2605.10039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 348. One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification

**arXiv ID:** 2605.10179 | [PDF](https://arxiv.org/pdf/2605.10179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 349. NumColBERT: Non-Intrusive Numeracy Injection for Late-Interaction Retrieval Models

**arXiv ID:** 2605.10109 | [PDF](https://arxiv.org/pdf/2605.10109v1)

**作者:** Haruki Fujimaki `[一作]` (University of Tsukuba), Makoto P. Kato `[通讯]` (University of Tsukuba)

**通讯引用:** 5976 | [OpenAlex ID](https://openalex.org/A5015183832)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为 NumColBERT 的非侵入式方法，在保持 ColBERT 原始推理管线不变的前提下，通过训练时的改进来实现对数值条件检索的支持。

**💡 创新点**

核心创新在于（1）数值门控机制（Numerical Gating Mechanism），在 MaxSim 聚合过程中动态放大满足数值条件的词嵌入权重；（2）数值对比学习目标（Numerical Contrastive Loss），显式塑造嵌入空间，使满足数值约束的文档与查询更贴近。

**🔧 技术方法**

技术实现包括：ColBERT 的后交互检索框架、基于 MLP 的数值检测器和门控器、对数值属性（单位、幅度、比较运算符）的多任务学习、以及对 PLAID 等索引压缩加速的兼容性。

**📊 数据集**

使用的主要数据集为金融领域的 FinQuant、医学领域的 MedQuant（数值检索基准），以及通用检索基准 MS MARCO 以评估跨域迁移效果。

**📈 对比分析**

实验将 NumColBERT 与 ColBERT fine‑tuned、QColBERT、DeepQuant、SPLADE、BM25 以及 GPT‑4o‑mini 重新排序器进行对比。结果显示，NumColBERT 在 FinQuant 与 MedQuant 上显著优于 ColBERT，且与 DeepQuant 的性能持平或更优；在联合训练下保持了与 ColBERT 相近的 MS MARCO 一般检索效果，并且在 PLAID 加速下可获得与标准 ColBERT 同等的速度与压缩比。

**⚠️ 局限性**

局限性包括：对数值检测与对比学习依赖训练时的数值标注；对 “小于” 型约束的处理仍不如 “大于” 或 “等于”；对更复杂的算术推理（多步比较、运算）支持有限；以及在极低位量化时对数值表示的敏感性。

---

## 350. Encoding and Decoding Temporal Signals with Spiking Bandpass Wavelets

**arXiv ID:** 2605.09770 | [PDF](https://arxiv.org/pdf/2605.09770v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 351. Higher-order Persistence Diagrams

**arXiv ID:** 2605.09866 | [PDF](https://arxiv.org/pdf/2605.09866v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 352. MUSDA: Multi-source Multi-modality Unsupervised Domain Adaptive 3D Object Detection for Autonomous Driving

**arXiv ID:** 2605.10026 | [PDF](https://arxiv.org/pdf/2605.10026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 353. Speech-based Psychological Crisis Assessment using LLMs

**arXiv ID:** 2605.10027 | [PDF](https://arxiv.org/pdf/2605.10027v1)

**作者:** Terumi Chiba `[一作]` (Tsinghua University), Chao Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 20585 | [OpenAlex ID](https://openalex.org/A5100460246)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对心理危机热线的三分类任务，提出一种基于大语言模型的自动危机级别判别框架，结合语音识别、情感插注和辅助推理训练实现高精度分类。

**💡 创新点**

创新点在于（1）通过“情感插注”将语音中的非语言情感特征显式注入文本，弥补纯文本分析的语音信息缺失；（2）在训练中加入诊断推理生成任务，作为正则化提升模型对危机判别的解释性与稳健性；（3）采用数据增广拆分呼叫为连续段落，以缓解样本稀缺导致的过拟合。

**🔧 技术方法**

核心技术包括：大语言模型（Qwen2.5-7B-Instruct）通过LoRA微调；自动语音识别（Paraformer-zh）+语音情感识别（SpeechLLM Step‑Audio‑R1）实现情感插注；推理增强训练通过多任务（分类+生成）损失组合；数据增广采用固定长度连续切片和多数投票聚合。

**📊 数据集**

使用中国心理热线真实数据集：154 条通话，约 100 小时，包含年龄、性别、危机等级等信息，并已对话音进行语音与文本处理。

**📈 对比分析**

与传统基线（OpenSMILE+SVM、零样本LLM、SpeechLLM）以及自研模型对比，最终模型在 5‑折交叉验证下宏 F1=0.802，准确率=0.805，显著优于所有基线（最大提升 0.251 的 F1）。

**⚠️ 局限性**

局限性：样本量仍有限，难以充分覆盖多样化危机情境；情感插注质量依赖于预训练模型的准确性；模型仍未集成实时运营商交互，缺乏在线验证；对不同语言或方言的泛化能力待进一步评估。

---

## 354. Merlin: Deterministic Byte-Exact Deduplication for Lossless Context Optimization in Large Language Model Inference

**arXiv ID:** 2605.09990 | [PDF](https://arxiv.org/pdf/2605.09990v1)

**作者:** Sietse Schelpe `[一作]` `[通讯]` (Corbenic AI, Inc.), Sietse Schelpe (Corbenic AI, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在大语言模型推理前对检索上下文进行字节级精确去重的实时预处理技术

**💡 创新点**

实现了极低延迟（1.1微秒/调用）且完全可重复、无信息损失的去重引擎，证明了在高冗余与低冗余场景下均保持模型质量不下降

**🔧 技术方法**

基于高熵指纹+冲突回退的哈希去重，单进程、无运行时依赖的C++实现，并提供跨平台（Windows x86‑64、Linux ARM64）静态二进制

**📊 数据集**

在四大生产API（Gemini、GPT‑5.1、Claude、Llama）上使用RULER、LongBench、HumanEval‑Snowball、WildChat等公开基准，以及22.2M BeIR文档用于大规模数学等价验证

**📈 对比分析**

与Python set()参考实现以及行业对照方法（LLMLingua、REFRAG、RAGBoost等）对比，单调用耗时5–30微秒，远低于推理预处理预算，累计质量差异+0.0/‑0.5pp，无Bonferroni显著衰退，二进制输出一致率99.2%（非代码100%）

**⚠️ 局限性**

仅适用于字节级重复，无法处理同义词/语义相似的冗余；对法律、医学等专业检索或多模态任务未评估；闭源实现仅通过公共数据验证可重现性；Bonferroni校正保守，可能低估实际影响

---

## 355. OUIDecay: Adaptive Layer-wise Weight Decay for CNNs Using Online Activation Patterns

**arXiv ID:** 2605.10161 | [PDF](https://arxiv.org/pdf/2605.10161v1)

**作者:** Alberto Fernández-Hernández `[一作]` (Universitat Politècnica de València), Enrique S. Quintana-Ortí `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 6947 | [OpenAlex ID](https://openalex.org/A5012806004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于激活模式的在线层级权重衰减调度方法OUIDecay；

**💡 创新点**

创新点在于利用无标签的Overfitting‑Underfitting Indicator（OUI）作为内部功能信号，按层动态重新分配权重衰减，而非依赖梯度或全局参数；

**🔧 技术方法**

核心技术包括批量版OUI计算、线性重标定权重衰减、周期性更新调度器；

**📊 数据集**

在EfficientNet‑B0+Stanford Cars、ResNet‑50+Food101、DenseNet‑121+CIFAR‑100、MobileNet‑V2+CIFAR‑10四个主流CNN‑数据集上进行实验；

**📈 对比分析**

与固定权重衰减和AdaDecay（梯度驱动）进行对比，OUIDecay在8个实验配置中取得7个最佳验证损失，整体表现优于两者；

**⚠️ 局限性**

局限性包括仅在CNN域验证，未探究对Transformer等其他网络的适用性，且方法需根据不同模型手动调节调度间隔和缩放区间。

---

## 356. DA-SegFormer: Damage-Aware Semantic Segmentation for Fine-Grained Disaster Assessment

**arXiv ID:** 2605.09864 | [PDF](https://arxiv.org/pdf/2605.09864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 357. Mixed-Criticality Flow Scheduling with Low Delay and Limited Bandwidth in TSN

**arXiv ID:** 2605.09888 | [PDF](https://arxiv.org/pdf/2605.09888v1)

**作者:** Wenyan Yan `[一作]` (Hunan First Normal University), Dongsheng Wei `[通讯]` (Hunan University)

**通讯引用:** 2286 | [OpenAlex ID](https://openalex.org/A5104239363)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种混合关键性流调度方案MCFS-2L，能够在TSN网络中实现低延迟和有限带宽的流传输。

**💡 创新点**

创新点在于：①将相同源/目的节点且周期为谐波关系的关键流与非关键流聚合为一个帧，②引入动态拆分机制，自动将不可调度的聚合帧中非关键帧拆出重新调度，从而显著提升关键流与非关键流的接受率。

**🔧 技术方法**

主要技术包括帧聚合（满足源/目的、周期谐波、截止期和最大帧尺寸约束）、动态重组与调度、TSN门控列表（GCL）预定与抢占式传输。

**📊 数据集**

使用了通用汽车（GM）真实工业数据集，涵盖主动安全、发动机控制、自动驾驶等多种应用的关键与非关键流。

**📈 对比分析**

与NWTT和R‑NWTT基线相比，MCFS-2L在关键流接受率上提升约4.78%，非关键流提升约8.58%，并在带宽利用率上降低最多11.88%；但在流量极大时执行时间会略高。

**⚠️ 局限性**

局限性包括：仅适用于单帧模型、仅针对域中心化汽车架构，且依赖周期的谐波关系；在高流量或不同拓扑时的可扩展性与实时性尚待进一步验证。

---

## 358. Adaptive Data Harvesting for Efficient Neural Network Learning with Universal Constraints

**arXiv ID:** 2605.09707 | [PDF](https://arxiv.org/pdf/2605.09707v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 359. FocuSFT: Bilevel Optimization for Dilution-Aware Long-Context Fine-Tuning

**arXiv ID:** 2605.09932 | [PDF](https://arxiv.org/pdf/2605.09932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 360. Online Steiner Forest with Recourse

**arXiv ID:** 2605.09821 | [PDF](https://arxiv.org/pdf/2605.09821v1)

**作者:** Yaowei Long `[一作]` (University of Michigan), Jakub Tarnawski `[通讯]` (Microsoft Research)

**通讯引用:** 305 | [OpenAlex ID](https://openalex.org/A5076652577)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种在线Steiner森林算法，能够在保持O(1)近似的同时实现O(log n)的摊还回调（recourse）次数。

**💡 创新点**

其创新点在于利用“pinning edges”与时间贪婪（timed gluttonous）聚类相结合的低回调框架，突破了传统聚类方法无法控制路径长度的问题。

**🔧 技术方法**

技术上结合了聚类层次、继承/非继承虚边选择、双向放松（dual‑fitting）分析以及按层级截断的边固定策略。

**📊 数据集**

该研究仅在理论上给出算法与证明，没有使用具体实验数据集。

**📈 对比分析**

与之前的 O(log² n) 或 O(log n) 竞争比算法相比，本文实现了相同竞争比但回调次数显著下降，达成了先前未实现的低回调常数竞争比目标。

**⚠️ 局限性**

主要局限在于回调次数仍为摊还O(log n)，尚未得到O(1)的 worst‑case 回调；且算法只适用于完全图度量空间的在线设置，未考虑动态删除或更一般图结构。

---

## 361. Near-Linear Time Generalized Sinkhorn Algorithms for Bounded Genus Graphs

**arXiv ID:** 2605.09782 | [PDF](https://arxiv.org/pdf/2605.09782v1)

**作者:** Krzysztof Choromanski `[一作]` (Columbia University), Dwaipayan Saha `[通讯]` (Columbia University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了GenusSink，一种针对有界基数图（尤其平面图）的近似Sinkhorn算法，能够在近线性时间和近线性内存下完成预处理、迭代和查询；

**💡 创新点**

创新点在于将分离图域积分器（S‑GFI）与树宽分解相结合，利用低位移秩、傅里叶分析以及对有界基数度量的低树宽近似，实现对最短路径距离矩阵的高效乘法；

**🔧 技术方法**

使用的技术包括：树宽分解与分离器、S‑GFI数据结构、分层递归乘法、傅里叶随机特征、低位移秩理论、基数度量到低树宽图的嵌入、以及实用的分离器采样和截断深度；

**📊 数据集**

实验数据集包括：自定义伪基数平面图、Thingi10K 3D 网格、纽约布朗克斯道路图（33,363 节点）以及真实 EMS 调度数据；

**📈 对比分析**

与传统全核 Sinkhorn、Greenkhorn、Sparse Sinkhorn 等方法比较，GenusSink 在保持数值精度（几乎与全核相同）的同时，显著降低运行时间（尤其在大规模图上）并提供更低的尾部响应时间；

**⚠️ 局限性**

局限性包括：算法依赖树宽为 O(log log n) 的近似；在极大图或非平面结构下，分离器采样和树宽嵌入的误差控制可能影响精度；实现复杂度高，需要构建 S‑GFI 结构和树宽分解。

---

## 362. On the Generation and Mitigation of Harmful Geometry in Image-to-3D Models

**arXiv ID:** 2605.09606 | [PDF](https://arxiv.org/pdf/2605.09606v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 363. CodeClinic: Evaluating Automation of Coding Skills for Clinical Reasoning Agents

**arXiv ID:** 2605.09675 | [PDF](https://arxiv.org/pdf/2605.09675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 364. MFVLR: Multi-domain Fine-grained Vision-Language Reconstruction for Generalizable Diffusion Face Forgery Detection and Localization

**arXiv ID:** 2605.10071 | [PDF](https://arxiv.org/pdf/2605.10071v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 365. TAD: Temporal-Aware Trajectory Self-Distillation for Fast and Accurate Diffusion LLM

**arXiv ID:** 2605.09536 | [PDF](https://arxiv.org/pdf/2605.09536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 366. TacoMAS: Test-Time Co-Evolution of Topology and Capability in LLM-based Multi-Agent Systems

**arXiv ID:** 2605.09539 | [PDF](https://arxiv.org/pdf/2605.09539v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 367. Medical Model Synthesis Architectures: A Case Study

**arXiv ID:** 2605.09716 | [PDF](https://arxiv.org/pdf/2605.09716v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 368. A Global Coding Scheme for OFDM over Finite Fields

**arXiv ID:** 2605.09865 | [PDF](https://arxiv.org/pdf/2605.09865v1)

**作者:** Juane Li `[一作]`, Shu Lin `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种全局编码多路复用框架FF-OFDM，将多条比特流通过循环码及其Hadamard等价码作为子载波，并利用Galois傅里叶变换实现全局耦合，得到结构化的QC‑LDPC码；

**💡 创新点**

创新点在于将传统OFDM的正交多路复用与编码相融合，构造基于部分几何的低密度循环码，并通过二进制分解定理实现全局联合软判决解码，既获得近似容量的可靠性，又保持线性解码复杂度；

**🔧 技术方法**

使用的技术包括循环码（BCH/RS）、Hadamard等价码、Galois傅里叶变换（GFT/IGFT）、部分几何与CPM矩阵、二进制分解定理、并行Min‑Sum/Sum‑Product LDPC解码；

**📊 数据集**

实验数据集为通过Monte‑Carlo仿真生成的AWGN信道下BPSK调制的符号流，没有使用公开真实数据集；

**📈 对比分析**

与传统独立编码、MLD、Union Bound和Sphere Packing Bound进行对比，结果显示FF-OFDM在大多数SNR点上比基线至少1–3.7 dB更优，接近SPB且无误差地板，迭代收敛快；

**⚠️ 局限性**

主要限制在于仅在AWGN信道下验证，尚未针对多径衰落、时变信道及等化问题展开研究；

---

## 369. Governing AI-Assisted Security Operations: A Design Science Framework for Operational Decision Support

**arXiv ID:** 2605.09534 | [PDF](https://arxiv.org/pdf/2605.09534v1)

**作者:** Elyson A. De La Cruz `[一作]` (University of the Cumberlands), Md Rasel Al Mamun `[通讯]` (University of Illinois Springfield)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个 AI 驱动的 KQL 查询代理，用于在 SOC 环境中安全地规划、审核并执行 AI 生成的查询，提升检测效率与可追溯性。

**💡 创新点**

创新点在于把 AI 规划与实际执行分离，构建了模板授权、策略验证、审核轨迹、角色责任与成熟度模型等治理机制，并提出完整的评估与决策框架。

**🔧 技术方法**

采用了设计科学研究方法、Kusto Query Language (KQL)、Microsoft Azure 安全产品（Defender XDR、Sentinel、Log Analytics、Sentinel Data Lake、ADX）、检索增强生成(RAG)、受限代理、CI/CD 审核等技术。

**📊 数据集**

主要使用 Microsoft 生态下的安全日志数据：Defender XDR、Sentinel、Log Analytics、Sentinel Data Lake、ADX；评估中还使用合成数据和历史重放数据集。

**📈 对比分析**

通过与手工 KQL、静态模板库、无代理 AI 生成查询等基线对比，使用语法有效性、治理通过率、分析师效用、成本/延迟等指标评估，结果表明受控代理在保证安全、成本可控的前提下显著提升分析师效用。

**⚠️ 局限性**

局限性包括：只在 Microsoft SOC 环境中验证，缺乏多企业生产环境的实证；依赖 SOC 成熟度与角色配置；合成数据无法完全模拟真实环境的噪声与攻击多样性。

---

## 370. Intervention-Based Time Series Causal Discovery via Simulator-Generated Interventional Distributions

**arXiv ID:** 2605.09870 | [PDF](https://arxiv.org/pdf/2605.09870v1)

**作者:** Tsuyoshi Okita `[一作]` (Kyushu Institute of Technology), Tsuyoshi Okita `[通讯]` (Kyushu Institute of Technology)

**通讯引用:** 2608 | [OpenAlex ID](https://openalex.org/A5026921207)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种名为 SVAR‑FM 的框架，利用物理模拟器在时间序列中实现 Pearl 的 do‑operator，从而通过模拟器生成的干预数据来进行因果结构发现。

**💡 创新点**

创新点包括：
① 将模拟器视为机械化的干预操作，能够在观测数据无法识别的情况下实现因果可识别；
② 在时间序列背景下给出完整结构 VAR（同时包含同时期与滞后因果）可识别的理论条件（覆盖条件）并证明其可识别性；
③ 推导出完整的误差上界，将 Monte‑Carlo、模拟器精度 δ_S 与 Flow Matching 误差三者分离，并预言当 δ_S 超过阈值时会出现符号翻转；
④ 在实验中首次验证了模拟器精度主导的符号翻转现象。

**🔧 技术方法**

主要技术手段包括：
- 非线性结构 VAR 模型作为因果语言；
- 条件 Flow Matching（Conditional Flow Matching）用于学习干预后条件分布并提取非线性因果机制；
- 理论分析：可识别性证明、误差传播定理与符号翻转判据；
- 统计显著性检验（bootstrap）用于边缘检测。

**📊 数据集**

使用的数据集与实验：
- CausalSim benchmark（四个科学领域：宏观经济、糖尿病、宇宙射线、锂电池）各配备了真实观测序列、可控干预的第一性原理或 Monte‑Carlo 模拟器；
- HHG（高次谐波生成）案例使用 Octopus TDDFT 计算得到观测与干预数据，并通过改变 XC 功能实现 δ_S 可调；
- 额外对比使用了公开的标准基准：CausalTime、Tigramite、CausalDynamics。

**📈 对比分析**

方法比较与性能：
- 对照传统观测式因果发现方法（OLS、Granger、VARLiNGAM、PCMCI），SVAR‑FM 在所有四个 CausalSim 场景中均恢复正确因果符号，偏差大幅下降（例如宏观经济中 99% 偏差缩减、糖尿病中 87% 缩减、宇宙射线中 100% 准确率）。
- 在 HHG 实验中，SVAR‑FM 与基线相比实现了 100% 的符号正确率并获得零偏差的平均因果效应；
- 在标准基准上，SVAR‑FM 的 F1、TPR 与 SHD 指标均不逊于或优于现有方法。

**⚠️ 局限性**

局限性：
- 需要存在可控且可物理实现的干预变量，且必须满足模拟器对真因果结构的结构一致性、可模块化性与变量对应性；
- 对 δ_S 的估计依赖领域先验或多模型一致性，无法提供绝对保证；
- 计算成本较高，尤其是高精度第一性原理模拟器与 Flow Matching 的训练；
- 目前主要关注直接因果路径，未涵盖更复杂的无环与有环结构中间接识别（如前门、后门）以及完全未知图的情形。

---

## 371. Language Models Without a Trainable Input Embedding Table: Learning from Fixed Minimal Binary Token Codes

**arXiv ID:** 2605.09751 | [PDF](https://arxiv.org/pdf/2605.09751v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 372. Operationalizing Cybersecurity Governance for Mitigation Planning with Attack-Path Modeling and Reinforcement Learning

**arXiv ID:** 2605.09792 | [PDF](https://arxiv.org/pdf/2605.09792v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 373. Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning

**arXiv ID:** 2605.09771 | [PDF](https://arxiv.org/pdf/2605.09771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 374. Nautilus Compass: Black-box Persona Drift Detection for Production LLM Agents

**arXiv ID:** 2605.09863 | [PDF](https://arxiv.org/pdf/2605.09863v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 375. Positional LSH: Binary Block Matrix Approximation for Attention with Linear Biases

**arXiv ID:** 2605.09472 | [PDF](https://arxiv.org/pdf/2605.09472v1)

**作者:** Daniel Wolfson `[一作]` (Tel Aviv University), Tal Wagner `[通讯]` (Tel Aviv University)

**通讯引用:** 247 | [OpenAlex ID](https://openalex.org/A5086071515)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了“位置级LSH”框架，证明ALiBi偏置矩阵可近似为随机连续块状二值掩码的线性组合，并给出了相应的近似算法；

**💡 创新点**

创新点在于将位置偏置、掩码和位置嵌入三大范式统一到LSH理论下，并针对ALiBi给出高概率谱范数与块大小的收敛性保证，实现长上下文ALiBi的近似线性时间算法；

**🔧 技术方法**

主要技术包括：局部敏感哈希（Random Binning Features）、矩阵集中理论、Fourier分析 Toeplitz 矩阵、子伽马分布尾部估计及矩阵 Bernstein 不等式；

**📊 数据集**

实验使用公开大语言模型 Llama‑4‑Scout‑17B‑16E 与 Mistral‑7B，评估在 Wikitext‑103 上的长上下文推断；

**📈 对比分析**

与原始无偏置模型、标准 ALiBi 及固定块掩码基线对比，Positional LSH 随样本数增大逼近 ALiBi 性能，在 Qwen3‑0.6B 上甚至优于原模型；在 Mistral‑7B 长上下文下，未超过原模型；

**⚠️ 局限性**

局限在于实现未针对硬件优化，未显示实际加速；理论上近似线性但在现有上下文长度下未显著提升速度；未来需在更长上下文或更高效实现中验证实效。

---

## 376. A Fast Hierarchical Splitting Approach for Non-Adaptive Learning of Random Hypergraphs

**arXiv ID:** 2605.09970 | [PDF](https://arxiv.org/pdf/2605.09970v1)

**作者:** Huy Pham `[一作]`, Hoang Ta `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一些概率不等式，包括马尔可夫不等式和切比雪夫不等式，并应用于独立伯努利随机变量的集中不等式。

**💡 创新点**

创新点在于通过集中不等式对随机变量的行为进行界定，提供了对复杂随机过程的深入分析。

**🔧 技术方法**

使用了马尔可夫不等式、切比雪夫不等式和切尔诺夫不等式等概率论技术。

**📊 数据集**

使用了独立伯努利随机变量的序列作为数据集，具体的随机变量和参数在文中进行了详细定义。

**📈 对比分析**

通过与标准集中不等式的比较，展示了所提出的不等式在特定条件下的有效性，性能分析表明在大样本情况下，所提出的不等式的界限更为紧致。

**⚠️ 局限性**

限制在于所提出的不等式在某些情况下可能不适用，特别是在样本量较小或随机变量分布不均匀的情况下，可能导致不准确的界限。

---

## 377. DRIVE-C: A Controlled Corruption Dataset for Autonomous Driving

**arXiv ID:** 2605.09774 | [PDF](https://arxiv.org/pdf/2605.09774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 378. EpiGraph: A Knowledge Graph and Benchmark for Evidence-Intensive Reasoning in Epilepsy

**arXiv ID:** 2605.09505 | [PDF](https://arxiv.org/pdf/2605.09505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 379. JODA: Composable Joint Dynamics for Articulated Objects

**arXiv ID:** 2605.09954 | [PDF](https://arxiv.org/pdf/2605.09954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 380. Permit: Permission-Aware Representation Intervention for Controlled Generation in Large Language Models

**arXiv ID:** 2605.09480 | [PDF](https://arxiv.org/pdf/2605.09480v1)

**作者:** Pengcheng Sun `[一作]` (University of Science and Technology of China), Chen Tang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 48867 | [OpenAlex ID](https://openalex.org/A5100337500)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Permit，一种在生成时通过低秩表示干预实现细粒度权限控制的框架。

**💡 创新点**

发现权限条件在隐藏空间形成可分离且低秩的偏移，基于此学习共享的低维子空间并进行轻量级偏移/门控干预。

**🔧 技术方法**

使用表示干预、低秩子空间投影、偏移与门控两种轻量级实现方式，保持主模型冻结。

**📊 数据集**

在 MedicalSys 医疗系统数据集（包含多角色多级权限的问答对）上进行实验。

**📈 对比分析**

与 Prompt‑Only、Prompt‑Perm、ControlNet 三个基线对比，Permit 在泄露率降至 0%–2.5% 的同时提升 18–21% F1，并且仅增加约 0.0018% 的可训练参数和 0.01–0.14s 的推理延迟。

**⚠️ 局限性**

假设权限信息在生成时已显式给定，无法处理动态/上下文相关的权限；未给出形式化安全保证，对强适应性攻击仍可能存在风险。

---

## 381. HS-FNO: History-Space Fourier Neural Operator for Non-Markovian Partial Differential Equations

**arXiv ID:** 2605.09523 | [PDF](https://arxiv.org/pdf/2605.09523v1)

**作者:** Lennon J. Shikhman `[一作]` `[通讯]` (Georgia Institute of Technology), Lennon J. Shikhman (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了History-Space Fourier Neural Operator (HS‑FNO)，将时延PDE的状态提升为历史窗口并仅学习新曝光的未来切片，利用精确的shift‑append传输更新历史；

**💡 创新点**

创新点在于将非马尔可夫动力学的自然状态——历史段——直接作为神经算子输入，剔除冗余的历史复制学习，并通过shift‑append强制实现确定性历史传输；

**🔧 技术方法**

采用了傅里叶神经算子（FNO）作为预测器，结合精确的shift‑append更新、条件嵌入（延迟、参数、步长）以及可选的rollout与semi‑flow正则化；

**📊 数据集**

在五个合成延迟/记忆PDE基准（延迟反应扩散、空间流行病学、非局部神经场、延迟波动力学、分布记忆闭包）以及公开交通流METR‑LA、PEMS‑BAY数据上进行训练与评估；

**📈 对比分析**

与传统的即时状态算子、滞后堆叠算子、全历史到全历史算子以及ConvLSTM/Transformer序列模型对比。HS‑FNO在一阶误差、历史空间误差和自动回归rollout误差上均优于基线，rollout误差从0.241降至0.094（约60%提升），并且参数量和内存占用显著低于全历史预测模型；

**⚠️ 局限性**

局限包括：对历史网格和延迟长度敏感；状态相关延迟、空间变异延迟或分布记忆可能需要插值/积分；在跨分辨率转移（尤其是延迟反应扩散）中表现不佳；延迟条件化不一定有益；rollout‑semi‑flow正则化效果差。

---

## 382. Enhancing Healthcare Search Intent Recognition with Query Representation Learning and Session Context

**arXiv ID:** 2605.10021 | [PDF](https://arxiv.org/pdf/2605.10021v1)

**作者:** Harshita Jagdish Sahijwani `[一作]` (Emory University), Chen Lin `[通讯]` (Emory University)

**通讯引用:** 471484 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种新的多集合损失函数，结合查询聚类和会话上下文，改进了健康搜索查询的表示学习与意图识别。

**💡 创新点**

创新点在于：①设计了多集合损失函数，利用点击文档聚类降低共点击噪声；②引入会话一致率（CR）度量查询意图一致性；③将聚类结果与多标签分类结合，提升全局与会话级意图识别。

**🔧 技术方法**

使用的技术包括：BERT 预训练查询编码器、对比学习与多集合损失、二元交叉熵多标签分类、跨源注意力机制融合会话上下文、并通过聚类评估（ARI、NMI）验证表示效果。

**📊 数据集**

实验数据集为内部 Health Search（HS）数据（约48k查询）和公开 TripClick（约13万查询），两者均包含查询、点击文档及文档类型标签。

**📈 对比分析**

与 BERT 与 PairWise‑BERT 基线对比，MSet‑BERT 在查询聚类（ARI 提升4.8%/10.3%）、全局多标签分类（F1 提升3.5%/1.7%）以及会话级分类（F1 提升4.7%/2.4%，NDCG@3 提升13.4%/7.0%）均显著优于基线。

**⚠️ 局限性**

局限性包括：依赖大规模点击日志，噪声仍难以完全消除；对非健康领域的通用性尚未验证；多集合损失的超参数调优与解释性仍需进一步研究。

---

## 383. VP, VNP and Algebraic Branching Programs over Min-Plus Semirings

**arXiv ID:** 2605.09551 | [PDF](https://arxiv.org/pdf/2605.09551v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 384. Task-Aware Calibration: Provably Optimal Decoding in LLMs

**arXiv ID:** 2605.10202 | [PDF](https://arxiv.org/pdf/2605.10202v1)

**作者:** Tim Tomov `[一作]` (Technical University of Munich), Stephan Günnemann `[通讯]` (Technical University of Munich)

**通讯引用:** 15347 | [OpenAlex ID](https://openalex.org/A5074504351)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在LLM生成任务中先将模型输出映射到任务特定的离散潜在空间，再对该潜在分布进行校准（Task Calibration），随后采用最小贝叶斯风险（MBR）解码，从而在任务层面实现决策最优。

**💡 创新点**

创新点在于：1) 将LLM输出从无限序列空间抽象到可管理的任务潜在空间；2) 定义分布式任务校准概念，使得在潜在空间的校准后，MBR解码成为整体最优；3) 提出任务校准误差（TCE）作为衡量校准质量并预测改进收益的指标。

**🔧 技术方法**

主要技术包括：LLM概率分布的潜在推送（通过采样获得潜在概率向量），Dirichlet校准（拟合对数概率的线性变换），MBR解码（在潜在空间上求期望损失最小化），以及使用TCE评估校准效果。

**📊 数据集**

实验使用了多种数据集，包括：HelpSteer（判分任务），STSB（相似度评估），When2Call（工具调用决策），MMLU（多选QA），TriviaQA，SimpleQA-Verified（答或不答），MAQA（多答案QA）以及向量化判分任务。

**📈 对比分析**

与传统解码策略（Greedy、Beam、Top-k、无校准MBR）相比，先校准再MBR的策略在所有任务上均表现最佳。实验报告显示，TCE值越大，校准后性能提升越显著，而传统的ECE与性能提升无明显相关。

**⚠️ 局限性**

局限性包括：1) 仅适用于可归约为离散潜在空间的任务，开放式生成任务需额外抽象；2) 校准仅在期望层面提升性能，个体实例不一定受益；3) 需要针对每个任务训练单独的校准映射，缺乏统一的跨任务解决方案。

---

## 385. ViSRA: A Video-based Spatial Reasoning Agent for Multi-modal Large Language Models

**arXiv ID:** 2605.10106 | [PDF](https://arxiv.org/pdf/2605.10106v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 386. Agentic Fuzzing: Opportunities and Challenges

**arXiv ID:** 2605.10074 | [PDF](https://arxiv.org/pdf/2605.10074v1)

**作者:** Junyoung Park `[一作]` (KAIST), Insu Yun `[通讯]` (KAIST)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于深度代理的“Agentic Fuzzing”技术，利用历史bug作为种子，代理通过分析根因、假设、验证三步发现逻辑漏洞。

**💡 创新点**

将大型语言模型从辅助工具升级为主推理引擎；引入四阶段代理管道、情景覆盖消重和基于DPP-MAP的种子调度，实现跨实现、跨触发路径的逻辑漏洞发现。

**🔧 技术方法**

使用Claude Opus 4.6等大型模型代理、代码分析与执行工具、PoC生成、情景覆盖数据库、Determinantal Point Process（DPP）种子选择、Docker化隔离执行以及软/硬时间阈值管理等技术。

**📊 数据集**

V8 JavaScript 引擎的 750 条历史漏洞种子（来自 Chromium issue tracker），并在 SpiderMonkey、JavaScriptCore 等引擎上进行验证；使用 OpenAI text‑embedding‑3‑large 生成种子向量。

**📈 对比分析**

与传统模糊、静态分析及 Google Big Sleep 等方法对比，在 V8 上一个月内发现 19 条新漏洞，其中 4 条获得 $35k 奖励；平均每 100 条种子发现 9 条新 bug，覆盖率及发现效率显著提升。

**⚠️ 局限性**

运行成本高（仅完成 23.8% 种子），依赖可获得的历史 bug，闭源项目受限；代理设计和模型更新易导致不稳定；种子调度和情景覆盖机制仍需改进。

---

## 387. S2P-Net: A Spectral-Spatial Polar Network for Rotation-Invariant Object Recognition in Low-Data Regimes

**arXiv ID:** 2605.09667 | [PDF](https://arxiv.org/pdf/2605.09667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 388. Scaling Vision Models Does Not Consistently Improve Localisation-Based Explanation Quality

**arXiv ID:** 2605.10142 | [PDF](https://arxiv.org/pdf/2605.10142v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 389. TopoU-Net: a U-Net architecture for topological domains

**arXiv ID:** 2605.10091 | [PDF](https://arxiv.org/pdf/2605.10091v1)

**作者:** Gaurav Gaurav `[一作]` (University of South Florida), Mustafa Hajij `[通讯]` (USFCA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了TopoU-Net，一种基于组合复杂的U-Net风格编码-解码器，通过跨秩传输和跳连实现多层次特征学习。

**💡 创新点**

创新点在于将尺度层次替换为拓扑秩层次，利用细胞、边界关系定义支持压缩比例(bottleneck support ratio)，并通过跳连阐明信息流与压缩关系。

**🔧 技术方法**

采用组合复杂、incidence矩阵实现跨阶级特征传输，同阶级细化与MLP细化，保持细胞重排等价性，并使用ReLU等非线性激活。

**📊 数据集**

在节点分类(Computers, Photo, Actor, Chameleon, Squirrel, Cornell, Wisconsin, Texas)、图分类(MUTAG, PROTEINS, IMDB-BINARY)、超图节点分类(CocitationCora/Citeseer/Pubmed, CoauthorshipCora/DBLP)、点云分类(ModelNet10/40)、图像重建/分割(Oxford‑IIIT Pet, Pascal VOC 2012)上进行评估。

**📈 对比分析**

与GCN、GraphSAGE、GAT、GIN、H2GCN、MixHop、DiffPool、Graph U‑Net等基线对比，TopoU-Net在六个节点分类数据集平均准确率最高，在四个超图数据集亦居前列，尤其在异质图上提升显著；在图分类、点云分类及轻量级图像分割中保持竞争力。

**⚠️ 局限性**

局限性包括需手工选择秩路径和高阶细胞，依赖预处理；大规模数据中构造高阶单元成本高，缺乏自动学习或采样机制，导致可扩展性受限。

---

## 390. MedMeta: A Benchmark for LLMs in Synthesizing Meta-Analysis Conclusion from Medical Studies

**arXiv ID:** 2605.09661 | [PDF](https://arxiv.org/pdf/2605.09661v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 391. Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward

**arXiv ID:** 2605.09920 | [PDF](https://arxiv.org/pdf/2605.09920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 392. Primal-Dual Guided Decoding for Constrained Discrete Diffusion

**arXiv ID:** 2605.09749 | [PDF](https://arxiv.org/pdf/2605.09749v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 393. Building Korean linguistic resource for NLU data generation of banking app CS dialog system

**arXiv ID:** 2605.10241 | [PDF](https://arxiv.org/pdf/2605.10241v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 394. Efficient Multi-Robot Motion Planning with Precomputed Translation-Invariant Edge Bundles

**arXiv ID:** 2605.09801 | [PDF](https://arxiv.org/pdf/2605.09801v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 395. Exploitation Without Deception: Dark Triad Feature Steering Reveals Separable Antisocial Circuits in Language Models

**arXiv ID:** 2605.09773 | [PDF](https://arxiv.org/pdf/2605.09773v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 396. Stable Long-Horizon PDE Forecasting via Latent Structured Spectral Propagators

**arXiv ID:** 2605.10154 | [PDF](https://arxiv.org/pdf/2605.10154v1)

**作者:** Xiaoxiao Lu `[一作]` (Huazhong University of Science and Technology), Jiahao Shi `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向长期预测的结构化谱传播器（SSP），通过将物理状态映射到时间一致的空间基，随后投影到紧凑传播空间，使用频率调节的线性骨干加上非线性谱闭包进行显式谱级传播，最终逆投影并解码得到物理场。

**💡 创新点**

创新点在于：①将空间表示与时间传播解耦，形成专用的“时间一致空间基”与“紧凑传播状态”；②在谱空间中构造频率条件的线性传播骨干与非线性闭包，显式利用PDE的谱耦合特性；③通过正交性与归一化正则化强化传播矩阵的稳定性；④采用分层监督（重构、潜在、物理）实现传播兼容性。

**🔧 技术方法**

使用的技术包括：卷积编码器+1×1投影、离散傅里叶变换与频谱截断、频率条件的多层感知机门控线性骨干、残差谱闭包卷积、正交与归一化正则化、基于时间步的自回归训练、结构化的解码器（谱分支+局部分支）。

**📊 数据集**

实验数据集为三类时间相关偏微分方程：浅水波（Shallow‑Water）、反应扩散（Reaction‑Diffusion）和纳维‑斯托克斯（Navier‑Stokes）三维网格模拟。

**📈 对比分析**

与FNO、U‑FNO、F‑FNO、UNO、KNO、LNO、CALM等基线在长时预测、频谱保持和超时域推理等任务上对比。SSP在L2、E_max、f_low等指标上均优于所有基线，尤其在长时预测中相对L2误差降低近49%，且在超时域表现出更稳健的误差增长曲线。

**⚠️ 局限性**

局限性包括：①谱基方法依赖周期边界或正交基，难以直接推广到非周期或复杂几何；②谱截断导致高频信息缺失，需手工调节频谱窗口；③模型对频率门控参数与正则化权重敏感，需经验调参；④在极端强非线性或多尺度耦合场景下，线性骨干与闭包的近似可能不足。

---

## 397. HiDrive: A Closed-Loop Benchmark for High-Level Autonomous Driving

**arXiv ID:** 2605.09972 | [PDF](https://arxiv.org/pdf/2605.09972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 398. GenioSim: A Novel Simulation Platform for Edge Computing over Optical Networks

**arXiv ID:** 2605.10062 | [PDF](https://arxiv.org/pdf/2605.10062v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 399. VFM-SDM: A vision foundation model-based framework for training-free, marker-free, and calibration-free structural displacement measurement

**arXiv ID:** 2605.09677 | [PDF](https://arxiv.org/pdf/2605.09677v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 400. DeepTumorVQA: A Hierarchical 3D CT Benchmark for Stage-Wise Evaluation of Medical VLMs and Tool-Augmented Agents

**arXiv ID:** 2605.09679 | [PDF](https://arxiv.org/pdf/2605.09679v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 401. The Silent Vote: Improving Zero-Shot LLM Reliability by Aggregating Semantic Neighborhoods

**arXiv ID:** 2605.09739 | [PDF](https://arxiv.org/pdf/2605.09739v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 402. Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction

**arXiv ID:** 2605.09649 | [PDF](https://arxiv.org/pdf/2605.09649v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 403. Continuous Latent Contexts Enable Efficient Online Learning in Transformers

**arXiv ID:** 2605.09867 | [PDF](https://arxiv.org/pdf/2605.09867v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 404. NCO: A Versatile Plug-in for Handling Negative Constraints in Decoding

**arXiv ID:** 2605.10065 | [PDF](https://arxiv.org/pdf/2605.10065v1)

**作者:** Hyundong Jin `[一作]` (Yonsei University), Yo-Sub Han `[通讯]` (Yonsei University)

**通讯引用:** 1472 | [OpenAlex ID](https://openalex.org/A5077698683)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种解码时的插件(NCO-Decoding)，在生成过程中实时屏蔽不允许出现的子串，既可处理有限硬约束字符串，也可处理正则表达式约束；

**💡 创新点**

不需要构造全局避免自机，而是在线维护多模式匹配状态（Aho-Corasick 词典+DFSA并行模拟），并通过BPE预计算减少一次性开销；

**🔧 技术方法**

利用Aho-Corasick 自动机、DFA并行模拟、BPE子词合并预计算、GPU并行掩码聚合等技术；

**📊 数据集**

在两个实际任务上评估：使用公开的毒性词典进行 profanity suppression，使用 Enron 邮件数据集的 PII 正则表达式进行 PII suppression；

**📈 对比分析**

与基线拒绝采样与 GUARD（基于Trie）对比；在批量生成下 NCO-Decoding 的相对吞吐率接近未约束模型（约 95%–110%），比拒绝采样高 5–6% 甚至更好，且在大批量下保持稳定；

**⚠️ 局限性**

仅支持显式有限硬约束与正则表达式约束，无法处理更高级的语法级约束；动态约束更新仍需改进。

---

## 405. Loom: Hybrid Retrieval-Scoring Outfit Recommendation with Semantic Material Compatibility and Occasion-Aware Embedding Priors

**arXiv ID:** 2605.09830 | [PDF](https://arxiv.org/pdf/2605.09830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 406. The Association of Transformer-based Sentiment Analysis with Symptom Distress and Deterioration in Routine Psychotherapy Care

**arXiv ID:** 2605.09838 | [PDF](https://arxiv.org/pdf/2605.09838v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 407. Cplus2ASP: Computing Action Language C+ in Answer Set Programming

**arXiv ID:** 2605.09528 | [PDF](https://arxiv.org/pdf/2605.09528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 408. Beyond Majority Voting: Agreement-Based Clustering to Model Annotator Perspectives in Subjective NLP Tasks

**arXiv ID:** 2605.09955 | [PDF](https://arxiv.org/pdf/2605.09955v1)

**作者:** Tadesse Destaw Belay `[一作]` (Instituto Politécnico Nacional), Seid Muhie Yimam `[通讯]` (University of Hamburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于协同聚类的注释者分组方法，替代传统多数投票，对情感分析、情绪分类和仇恨言论检测等主观 NLP 任务中的注释者不一致进行建模。

**💡 创新点**

创新点在于只聚类注释者而非为每个注释者训练单独模型，既保持多样化视角，又显著降低计算开销；并在聚类后采用多标签与多任务聚合，进一步提升性能。

**🔧 技术方法**

核心技术包括：基于配对一致性（Cohen's kappa 或 Jaccard）构建注释者相似度矩阵、k‑means 聚类、聚类内部标签聚合（多数投票或多标签/多任务输出），以及使用 AfroXLMR 等跨语言预训练模型进行序列分类。

**📊 数据集**

使用 40 个数据集，覆盖 18 种语言，包含 AfriSenti（情感）、AfriEmo（情绪）和 AfriHate（仇恨）三大主观任务，并扩展至 GoEmotions 与 GabHate 的英文数据集。

**📈 对比分析**

与多数投票、单个注释者集成、集成、单标签和多任务等四种聚合方式对比，结果显示聚类后多标签和多任务聚合平均提升约 8–12% 的宏 F1 分数，尤其在情感与仇恨检测任务中显著优于基线。

**⚠️ 局限性**

局限包括：只尝试单一预训练模型和固定的 3/5 个聚类，聚类方法未探索层次或谱聚类；聚合后仍采用多数投票，未使用软标签或贝叶斯融合；且未针对不同任务或语言进行超参数调优。

---

## 409. Only Train Once: Uncertainty-Aware One-Class Learning for Face Authenticity Detection

**arXiv ID:** 2605.10040 | [PDF](https://arxiv.org/pdf/2605.10040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 410. Emerging 2D Materials for Beyond von Neumann Computing: A Perspective

**arXiv ID:** 2605.09695 | [PDF](https://arxiv.org/pdf/2605.09695v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 411. Nectar: Neural Estimation of Cached-Token Attention via Regression

**arXiv ID:** 2605.09778 | [PDF](https://arxiv.org/pdf/2605.09778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 412. Reflection Anchors for Propagation-Aware Visual Retention in Long-Chain Multimodal Reasoning

**arXiv ID:** 2605.09614 | [PDF](https://arxiv.org/pdf/2605.09614v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 413. CTQWformer: A CTQW-based Transformer for Graph Classification

**arXiv ID:** 2605.09486 | [PDF](https://arxiv.org/pdf/2605.09486v1)

**作者:** Zhan Li `[一作]` (Beijing Normal University), Chuan Wang `[通讯]` (Beijing Normal University)

**通讯引用:** 7761 | [OpenAlex ID](https://openalex.org/A5100443590)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种名为 CTQWformer 的图分类框架，融合连续时间量子游走（CTQW）与图 Transformer 和双向 GRU 以学习图结构与动态信息。

**💡 创新点**

创新点在于：①设计可训练的 Hamiltonian，将图拓扑与节点特征融合；②利用 CTQW 产生的最终时间传播概率作为 Transformer 的结构偏置；③将完整的 CTQW 进化序列输入双向循环网络捕获时间演化特征。

**🔧 技术方法**

核心技术包括：连续时间量子游走编码、可学习 Hamiltonian、图 Transformer 的结构偏置注意力、双向 GRU 时序建模，以及图核与 GNN 的对比实验。

**📊 数据集**

实验使用 TU 集合中的六个基准数据集：MUTAG、PTC(MR)、PROTEINS、DD、IMDB-B 和 IMDB-M。

**📈 对比分析**

通过与传统图核、主流 GNN（如 GIN、GCN、GAT 等）以及最新图 Transformer（Graphormer、GraphGPS、GRIT）的 10 折交叉验证比较，CTQWformer 在除 IMDB-M 外的大多数数据集上取得最高或相近的准确率，显示出显著性能提升。

**⚠️ 局限性**

局限性包括：对缺少节点特征或极小图（如 IMDB-M）表现不佳；CTQW 的模拟需要 O(T n³) 计算，导致在大规模图上的计算成本较高。

---

## 414. Discovery of Nonlinear Dynamics with Automated Basis Function Generation

**arXiv ID:** 2605.09696 | [PDF](https://arxiv.org/pdf/2605.09696v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 415. NaiAD: Initiate Data-Driven Research for LLM Advertising

**arXiv ID:** 2605.09918 | [PDF](https://arxiv.org/pdf/2605.09918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 416. FormalRewardBench: A Benchmark for Formal Theorem Proving Reward Models

**arXiv ID:** 2605.10141 | [PDF](https://arxiv.org/pdf/2605.10141v1)

**作者:** Zeynel A. Uluşan `[一作]`, Gözde Gül Şahin `[通讯]` (Koç University)

**通讯引用:** 525 | [OpenAlex ID](https://openalex.org/A5078696090)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了FormalRewardBench，一个用于评估形式化定理证明奖励模型的基准，包含250对优劣证明对，利用五种专家设计的错误注入策略生成不正确的 Lean 4 证明。

**💡 创新点**

首创形式化证明奖励模型评测基准；提出了五种专业化错误注入方法；揭示生成模型与评估模型之间的“生成‑评估不对称”；公开了数据集与评测脚本。

**🔧 技术方法**

采用强化学习与可验证奖励（RLVR）框架；使用大语言模型（Claude Opus 4.5 等）进行错误注入和奖励预测；对比点对点评估与对比评分两种评测模式；通过点式和对比式评分、位置一致性等指标衡量模型表现。

**📊 数据集**

以 MiniF2F（488 个 Lean 4 竞赛级问题）为基础，生成 250 对优劣证明对，用于构建基准数据集。

**📈 对比分析**

通过在点对点评估（单独打分）和对比评分（两侧都需一致）两种方式比较四类模型（前沿 LLM、判定 LLM、通用 LLM、专业证明模型）。结果显示前沿 LLM 最高（点对评估 70.1%，对比 59.8%），专业证明模型最低（点对 13.7%，对比 9.4%），并且不同错误策略难度梯度明显；评测中还观察到显著的位序偏差。

**⚠️ 局限性**

局限性包括：评估主要依赖自动验证，仅对 50 对样本做人工检查；仅针对 Lean 4；错误注入策略未覆盖所有可能的失败模式；仅评估单轮优劣判断，未考虑逐步过程评估。

---

## 417. Geometric 4D Stitching for Grounded 4D Generation

**arXiv ID:** 2605.09984 | [PDF](https://arxiv.org/pdf/2605.09984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 418. V-ABS: Action-Observer Driven Beam Search for Dynamic Visual Reasoning

**arXiv ID:** 2605.10172 | [PDF](https://arxiv.org/pdf/2605.10172v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 419. Explainability of Recurrent Neural Networks for Enhancing P300-based Brain-Computer Interfaces

**arXiv ID:** 2605.10121 | [PDF](https://arxiv.org/pdf/2605.10121v1)

**作者:** Christian Oliva `[一作]` (Universidad Autónoma de Madrid), Luis F Lago-Fernández `[通讯]` (Universidad Autónoma de Madrid)

**通讯引用:** 861 | [OpenAlex ID](https://openalex.org/A5034298577)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一种加入Post-Recurrent Module (PRM) 的RNN，用于检测P300事件，并通过梯度和L1正则化实现时空可解释性。

**💡 创新点**

创新点在于：①在传统Elman RNN后加入PRM层，显式整合所有时序输出，避免只依赖最后隐藏状态；②在PRM和输入层使用L1正则化，自动聚焦关键电极与时间窗口，提升可解释性；③将全局和局部解释技术结合，生成时空重要性图谱。

**🔧 技术方法**

使用技术包括：Elman RNN、Post-Recurrent Module、梯度归因法、L1正则化、k折交叉验证、平衡准确率（BAC）评估。

**📊 数据集**

使用Hoffmann等人公开的六选P300范式数据集，8名受试者，32通道EEG，降采样至32Hz，提取1000ms窗口（32时步）。

**📈 对比分析**

与传统BLDA基线（BAC≈0.71）以及无PRM的RNN进行对比，加入PRM后BAC提升至0.80（约9%提升），表明时序整合和可解释性正则化显著提高性能。

**⚠️ 局限性**

局限性包括：仍需手动调整正则化参数、对不同实验设置的泛化性待验证、对实时低功耗实现尚未评估、对极端个体差异的处理仍不充分。

---

## 420. Muninn: Your Trajectory Diffusion Model But Faster

**arXiv ID:** 2605.09999 | [PDF](https://arxiv.org/pdf/2605.09999v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 421. DAP: Doppler-aware Point Network for Heterogeneous mmWave Action Recognition

**arXiv ID:** 2605.09604 | [PDF](https://arxiv.org/pdf/2605.09604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 422. Data-Asymmetric Latent Imagination and Reranking for 3D Robotic Imitation Learning

**arXiv ID:** 2605.10166 | [PDF](https://arxiv.org/pdf/2605.10166v1)

**作者:** Lianghao Luo `[一作]` (Fudan University), Wei Li `[通讯]` (Fudan University)

**通讯引用:** 99177 | [OpenAlex ID](https://openalex.org/A5100318082)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出DALI-R框架，利用数据不对称策略：用成功演示训练3D生成式动作模型，使用混合质量轨迹训练潜在世界模型和任务完成评分器，进而在推理时通过潜在想象和重排序提升决策；

**💡 创新点**

创新点在于：①将不同质量数据分别用于动作先验和预测评估的“数据不对称”学习；②在3D点云潜在空间进行想象与评分，避免图像/视频空间计算开销；③通过点云随机丢失生成候选，结合流匹配和扩散生成器实现高吞吐候选生成与重排序；

**🔧 技术方法**

采用的技术包括：3D点云编码器、扩散生成政策、最优传输流匹配、潜在世界模型、任务完成评分器、点云随机丢失、推理时的潜在重排序；

**📊 数据集**

使用的数据集：Adroit和MetaWorld（MuJoCo模拟）进行主实验，并在3D扩散政策公开的真实点云轨迹上做离线诊断；

**📈 对比分析**

实验结果显示，在Adroit和MetaWorld上平均成功率提升约6–7%（从53.7%到60.7%/55.9%到62.4%），且推理开销低于0.7×原基线；相较于仅使用3D扩散或流匹配基线，DALI-R在所有任务上均取得更高成功率；

**⚠️ 局限性**

限制包括：对潜在预测与评分准确性高度依赖；仅使用块级重排序，缺乏长时延推理；尚未在真实机器人上验证鲁棒性；未加入不确定性估计等。

---

## 423. Omni-Persona: Systematic Benchmarking and Improving Omnimodal Personalization

**arXiv ID:** 2605.09996 | [PDF](https://arxiv.org/pdf/2605.09996v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 424. Learning Multi-Indicator Weights for Data Selection: A Joint Task-Model Adaptation Framework with Efficient Proxies

**arXiv ID:** 2605.09665 | [PDF](https://arxiv.org/pdf/2605.09665v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 425. Breaking the Reward Barrier: Accelerating Tree-of-Thought Reasoning via Speculative Exploration

**arXiv ID:** 2605.10195 | [PDF](https://arxiv.org/pdf/2605.10195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 426. HapticLDM: A Diffusion Model for Text-to-Vibrotactile Generation

**arXiv ID:** 2605.09971 | [PDF](https://arxiv.org/pdf/2605.09971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 427. Dystruct: Dynamically Structured Diffusion Language Model Decoding via Bayesian Inference

**arXiv ID:** 2605.09820 | [PDF](https://arxiv.org/pdf/2605.09820v1)

**作者:** Bian Sun `[一作]` (University Of Central Florida), Zhenyi Wang `[通讯]` (University Of Central Florida)

**通讯引用:** 284 | [OpenAlex ID](https://openalex.org/A5100690302)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关、贝叶斯结构化解码框架，用于扩散语言模型的灵活长度生成

**💡 创新点**

将动态窗口扩展、块划分与解码顺序统一为贝叶斯后验推理，利用CRP先验和上下文感知调度；实现无需重新训练即可自适应生成长度并保持结构连贯

**🔧 技术方法**

贝叶斯推理、Chinese Restaurant Process (CRP) 先验、上下文感知 Gibbs 调度、局部边界修复 (edge-welding)

**📊 数据集**

LLaDA-8B-Base、Dream-7B-Base 在 GSM8K、MATH、MBPP、HumanEval、BBH 等基准上测试

**📈 对比分析**

相较于固定长度和现有可变长度解码方法（如DAEDAL、FlexMDM 等），该方法在所有五个基准上均有显著提升（例如BBH精确匹配从44.9%提升至49.3%），并在推理效率上更优

**⚠️ 局限性**

仅在推理时使用，未改造模型参数；缺乏训练时的结构学习，可能限制进一步提升；对超参数（CRP浓度、weld 半径）依赖度仍需探究

---

## 428. Population Protocols over Ordered Agents

**arXiv ID:** 2605.09937 | [PDF](https://arxiv.org/pdf/2605.09937v1)

**作者:** Michael Blondin `[一作]` (Université de Sherbrooke), Isa Vialard `[通讯]` (MPI for Software Systems)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文研究了一种在全序标识符下的有限状态人口协议（population protocol），并对不同数值谓词（如<、+1）的限制进行建模，探讨这些协议的表达能力和稳定性问题。

**💡 创新点**

创新点在于：①给出了[<]协议可识别的语言类完全等价于无歧义星自由语言（unambiguous star-free languages）以及Δ₂[<]；②提出了新的逻辑（_1[<,+,≡]）和部分排序Parikh自动机（poPA）模型，并证明其与[<]等价；③证明了在+1谓词下的协议与线性有界图灵机等价；④对判定一个协议是否为decider的问题给出了可判定性与不可判定性边界。

**🔧 技术方法**

使用的技术包括：well‑structured transition system 与子词闭包的抽泣泵引理、语义层面的semi‑decider与稳定输入协议构造、Presburger算术与数值谓词的量化消除、部分排序自动机与其可接受公式的互译、即时观测协议的握手模拟以及对Post对应问题的归约。

**📊 数据集**

论文为理论性质证明，不涉及具体数据集，所有结论均基于形式化模型与数学证明。

**📈 对比分析**

在可判定性方面，算法复杂度被限定在(n)（线性有界空间）内；在表达能力比较方面，证明了[<]与Δ₂[<]、无歧义星自由语言及poPA的等价性，并展示了+1情形下语言类与(n)的等价性；这些结果在理论上完成了对该协议族的完整性分析。

**⚠️ 局限性**

主要限制在于：①对[<]协议的完整等价性（与_1[<,+,≡]及弱无歧义poPA的闭包性）仍为未证明的猜想；②+1谓词下判定协议是否为decider的问题是不可判定的；③部分排序自动机的弱无歧义闭包性与补集闭包性尚未得到正式证明。

---

## 429. SEMASIA: A Large-Scale Dataset of Semantically Structured Latent Representations

**arXiv ID:** 2605.09485 | [PDF](https://arxiv.org/pdf/2605.09485v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 430. Learning Graph Foundation Models on Riemannian Graph-of-Graphs

**arXiv ID:** 2605.09993 | [PDF](https://arxiv.org/pdf/2605.09993v1)

**作者:** Haokun Liu `[一作]` (University of Science and Technology of China), Xike Xie `[通讯]` (University of Science and Technology of China)

**通讯引用:** 1990 | [OpenAlex ID](https://openalex.org/A5037366245)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了R-GFM，一种基于Riemannian图-图（GoG）的图基础模型，用以解决固定跳数采样导致的结构尺度不匹配问题。

**💡 创新点**

创新点在于动态构建多尺度GoG并通过混合专家的Riemannian路由自适应选择合适的几何空间，理论上显著降低结构域泛化误差。

**🔧 技术方法**

技术上结合了自适应跳数子图采样、相似度稀疏GoG构建、Riemannian混合专家的动态MoE路由以及对比学习预训练。

**📊 数据集**

实验使用18个真实世界图数据集（如Cora、Citeseer、Pubmed等）以及4个大规模训练集（ArXiv_2023、ogbn-Arxiv、Reddit、PubMed）。

**📈 对比分析**

与多种基线（Task‑Supervised GNN、Self‑Supervised、Prompt、GFM）对比，R‑GFM在1/3/5‑shot节点分类和链接预测任务中均取得最高或第二高准确率，提升幅度可达49%。

**⚠️ 局限性**

局限性包括对大图仍有一定内存消耗，Riemannian专家数选择依赖数据分布且在极端异构数据集上可能需进一步改进。

---

## 431. Voice Biomarkers for Depression and Anxiety

**arXiv ID:** 2605.09908 | [PDF](https://arxiv.org/pdf/2605.09908v1)

**作者:** Oleksii Abramenko `[一作]` (Kintsugi Mindful Wellness), Colin Vaz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究开发了一种基于30秒语音的抑郁与焦虑严重度预测模型。

**💡 创新点**

创新点在于结合深度学习对原始语音的直接建模、使用多模态（音频+文本）知识蒸馏以及声纹一致性损失来提升预测准确性。

**🔧 技术方法**

采用预训练音频骨干 Whisper Small，LoRA 微调，CORAL 序数回归，SVL 一致性损失，知识蒸馏和 LLM 逼近等技术。

**📊 数据集**

使用由美国 23,000+ 受试者共 64,828 条录音、约 688 小时音频，包含 PHQ‑9 和 GAD‑7 标签的私有大规模数据集。

**📈 对比分析**

与单模态、单模型对比，最终模型在抑郁和焦虑的 SN=SP 指标分别达到 71.1% 和 70.7%，AUROC 约 0.79，较基线提升约 10%。

**⚠️ 局限性**

局限在于依赖大规模标注数据、对语音内容的依赖仍存在、对合成或不同语音环境的泛化尚待验证。

---

## 432. LoopVLA: Learning Sufficiency in Recurrent Refinement for Vision-Language-Action Models

**arXiv ID:** 2605.09948 | [PDF](https://arxiv.org/pdf/2605.09948v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 433. Learning from Acceptance: Cumulative Regret in the Game of Coding

**arXiv ID:** 2605.09754 | [PDF](https://arxiv.org/pdf/2605.09754v1)

**作者:** Hanzaleh Akbari Nodehi `[一作]` (University of Minnesota, Twin Cities), Mohammad Ali Maddah-Ali `[通讯]` (University of Minnesota, Twin Cities)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了在不完全信息的游戏编码框架下，数据收集者通过学习阈值来最大化其期望收益的在线策略；

**💡 创新点**

创新点在于：①将游戏编码问题转化为连续动作的带噪声多臂赌博机；②设计了基于区域覆盖与乐观估计的“游戏编码缩放算法”，实现子线性累计后悔；

**🔧 技术方法**

利用带有不确定度补偿的连续动作扩展bandit算法（Zooming算法），结合统计估计接受概率并映射为估计误差；

**📊 数据集**

实验采用两节点模拟环境，诚实噪声为均匀分布，阈值区间为[2,30]，无公开数据集；

**📈 对比分析**

与传统的探索后承诺基线对比，所提算法在10万轮中累计后悔从约1.41×10^4降低至8.85×10^3，展示出更优的学习轨迹性能；

**⚠️ 局限性**

局限在于仅验证了两节点情形，缺乏多节点或真实区块链环境的实验，且对高维/非凸阈值空间的适用性尚未探讨。

---

## 434. CalBench: Evaluating Coordination-Privacy Trade-offs in Multi-Agent LLMs

**arXiv ID:** 2605.09823 | [PDF](https://arxiv.org/pdf/2605.09823v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 435. MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs

**arXiv ID:** 2605.10064 | [PDF](https://arxiv.org/pdf/2605.10064v1)

**作者:** Ruiyi Yang `[一作]` (University of New South Wales), Flora D. Salim `[通讯]` (University of New South Wales)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立四子图共进化知识图，让自我演化语言模型代理在冻结骨干下通过外部图记忆实现跨迭代学习。

**💡 创新点**

将跨迭代知识转化为结构化图记忆，并结合任务级搜索与技能级路由bandit共同进化，从而让冻结模型可持续提升。

**🔧 技术方法**

使用结构化知识图、双记忆索引（成功/失败）、任务级搜索bandit、技能级路由bandit、指引层写图与执行层冻结等技术。

**📊 数据集**

九个基准：GSM8K、RealMath、HotpotQA、WebQA、STBench、FinQA、MedQA-USMLE、Crafter、WebShop。

**📈 对比分析**

与冻结骨干提示基线（Zero-shot CoT、8-shot CoT、ReAct 等）对比，在数学推理、时空分析等模板丰富任务上平均提升约 7–20%，显著优于基线。

**⚠️ 局限性**

依赖图写入质量和可重复结构任务，对非模板化或高度交互式任务效果有限，且需要外部教师指导来进行图写入。

---

## 436. KAN Text to Vision? The Exploration of Kolmogorov-Arnold Networks for Multi-Scale Sequence-Based Pose Animation from Sign Language Notation

**arXiv ID:** 2605.09572 | [PDF](https://arxiv.org/pdf/2605.09572v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 437. SciVQR: A Multidisciplinary Multimodal Benchmark for Advanced Scientific Reasoning Evaluation

**arXiv ID:** 2605.10187 | [PDF](https://arxiv.org/pdf/2605.10187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 438. DegBins: Degradation-Driven Binning for Depth Super-Resolution

**arXiv ID:** 2605.09628 | [PDF](https://arxiv.org/pdf/2605.09628v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 439. ExtraVAR: Stage-Aware RoPE Remapping for Resolution Extrapolation in Visual Autoregressive Models

**arXiv ID:** 2605.10045 | [PDF](https://arxiv.org/pdf/2605.10045v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 440. Do multimodal models imagine electric sheep?

**arXiv ID:** 2605.09693 | [PDF](https://arxiv.org/pdf/2605.09693v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 441. Learning to Sparsify Stochastic Linear Bandits

**arXiv ID:** 2605.10151 | [PDF](https://arxiv.org/pdf/2605.10151v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 442. Cost-of-Ethics Crisis: Beliefs, Decisions, and Justifications in the Job Searches of Computer Science Students in Canada and the United States

**arXiv ID:** 2605.09680 | [PDF](https://arxiv.org/pdf/2605.09680v1)

**作者:** Mohamed Abdalla `[一作]` (University of Alberta), Catherine Stinson `[通讯]` (Queen's University)

**通讯引用:** 1008 | [OpenAlex ID](https://openalex.org/A5054896491)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对129名加拿大和美国的计算机科学学生及毕业生在求职过程中做伦理决策的调查与分析

**💡 创新点**

首次将求职情境与伦理教育相结合，通过动态问卷挖掘学生在真实情境下的伦理取舍与合理化机制

**🔧 技术方法**

问卷调查、开放式文本卡片分类、定量统计（Mann‑Whitney U、Fisher 精确检验）

**📊 数据集**

129份包含学历、性别、民族、申请公司与伦理关切的问卷数据

**📈 对比分析**

通过比较各因素的排名及显著性检验，发现薪酬始终排首位，伦理关注排名末尾；结果显示学生多使用道德脱离与经济理由来平衡选择，显示伦理教育与实际行为之间的显著差距

**⚠️ 局限性**

样本量有限、仅包含北美两国学生、受访者自选问卷导致响应偏倚、缺乏纵向追踪，难以推断长期影响

---

## 443. Per-Loss Adapters for Gradient Conflict in Physics-Informed Neural Networks

**arXiv ID:** 2605.10136 | [PDF](https://arxiv.org/pdf/2605.10136v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 444. FERA: Uncertainty-Aware Federated Reasoning for Large Language Models

**arXiv ID:** 2605.10082 | [PDF](https://arxiv.org/pdf/2605.10082v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 445. From Pixels to Concepts: Do Segmentation Models Understand What They Segment?

**arXiv ID:** 2605.09591 | [PDF](https://arxiv.org/pdf/2605.09591v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 446. TIDES: Implicit Time-Awareness in Selective State Space Models

**arXiv ID:** 2605.09742 | [PDF](https://arxiv.org/pdf/2605.09742v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 447. Learning to Compress and Transmit: Adaptive Rate Control for Semantic Communications over LEO Satellite-to-Ground Links

**arXiv ID:** 2605.10095 | [PDF](https://arxiv.org/pdf/2605.10095v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 448. Hyperbolic Distillation: Geometry-Guided Cross-Modal Transfer for Robust 3D Object Detection

**arXiv ID:** 2605.09899 | [PDF](https://arxiv.org/pdf/2605.09899v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 449. Assessment of RAG and Fine-Tuning for Industrial Question-Answering-Applications

**arXiv ID:** 2605.09533 | [PDF](https://arxiv.org/pdf/2605.09533v1)

**作者:** Jakob Sturm `[一作]` (BMW Group), Andre Luckow `[通讯]` (BMW Group)

**通讯引用:** 2494 | [OpenAlex ID](https://openalex.org/A5077700715)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文比较了 RAG 与 FT 在汽车行业问答任务中的效果与成本。

**💡 创新点**

创新点是将成本‑准确性评估框架扩展为包含人类验证与人工干预成本，并在真实行业数据上验证。

**🔧 技术方法**

使用 GPT‑4o、GPT‑4o‑mini、LLaMA3.3‑70B、LLaMA3.2‑3B 四种模型，构建 Base、FT、RAG、RAG+FT 四种流水线。

**📊 数据集**

利用 BMW Group 的 Car User Manual 与 Vehicle Quality 两个专有数据集进行实验。

**📈 对比分析**

通过 LLM‑as‑a‑Judge 计算正确率并结合扩展 Cost‑of‑Pass，结果显示 RAG 能显著提升准确率并在成本上更优，尤其使开源模型与大型专有模型相当。

**⚠️ 局限性**

局限在于数据集专有且未公开，评估依赖 GPT‑4o 的判断，未覆盖多模态输入与完整部署场景。

---

## 450. Rethinking Random Transformers as Adaptive Sequence Smoothers for Sleep Staging

**arXiv ID:** 2605.09905 | [PDF](https://arxiv.org/pdf/2605.09905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 451. TimeClaw: A Time-Series AI Agent with Exploratory Execution Learning

**arXiv ID:** 2605.10038 | [PDF](https://arxiv.org/pdf/2605.10038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 452. AdaptSplat: Adapting Vision Foundation Models for Feed-Forward 3D Gaussian Splatting

**arXiv ID:** 2605.10239 | [PDF](https://arxiv.org/pdf/2605.10239v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 453. Combining Mechanical and Agentic Specification Inference for Move

**arXiv ID:** 2605.10005 | [PDF](https://arxiv.org/pdf/2605.10005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 454. Position: Academic Conferences are Potentially Facing Denominator Gaming Caused by Fully Automated Scientific Agents

**arXiv ID:** 2605.09915 | [PDF](https://arxiv.org/pdf/2605.09915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 455. Kintsugi: Learning Policies by Repairing Executable Knowledge Bases

**arXiv ID:** 2605.09487 | [PDF](https://arxiv.org/pdf/2605.09487v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 456. MonitoringBench: Semi-Automated Red-Teaming for Agent Monitoring

**arXiv ID:** 2605.09684 | [PDF](https://arxiv.org/pdf/2605.09684v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 457. In-Network Artificial Computing Enhanced Light Model-Switching for Emergency Communications Networks

**arXiv ID:** 2605.10070 | [PDF](https://arxiv.org/pdf/2605.10070v1)

**作者:** Yuehan Li `[一作]` (Xidian University), Wenchi Cheng `[通讯]` (Xidian University)

**通讯引用:** 4920 | [OpenAlex ID](https://openalex.org/A5012820395)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了在网络内实现轻量级模型切换的人工计算框架，利用包元数据在包级别选择预载的二值神经网络模型，实现即时推理并保持路由路径不变。

**💡 创新点**

通过在共享执行路径上预装多模型并仅通过包元数据进行 O(1) 切换，避免了控制平面重装载导致的延迟与中断，实现了在线、包边界级别的快速模型切换。

**🔧 技术方法**

使用 eBPF/XDP+AF_XDP 在 Linux 内核级实现数据平面，利用 x86 AVX‑512 SIMD 加速 1024 B 固定输入的 BNN 推理；模型存放于驻留模型库，切换通过寄存器映射实现。

**📊 数据集**

使用 IoT‑23 数据集的网络流量样本，映射为固定 1088 B 包结构，用于恶意流量识别实验。

**📈 对比分析**

与传统控制平面模型替换相比，切换延迟从 484 μs 降至 0.005 μs，误包率为 0；推理延迟 0.528 μs，吞吐 1.894 Mpps；即使扩展到 16 槽，切换成本保持 0.0037 μs，整体延迟仍低于 1 μs。

**⚠️ 局限性**

模型仅限于同一输入格式的 BNN；在更大模型库或更复杂网络结构下的缓存压力和内存占用未充分评估；实验聚焦于恶意检测，缺乏跨应用验证。

---

## 458. From Single-Step Edit Response to Multi-Step Molecular Optimization

**arXiv ID:** 2605.10035 | [PDF](https://arxiv.org/pdf/2605.10035v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 459. Fair Allocation under Conflict Constraints

**arXiv ID:** 2605.09930 | [PDF](https://arxiv.org/pdf/2605.09930v1)

**作者:** Sarfaraz Equbal `[一作]` (IIT Bombay), Hirotaka Yoneda `[通讯]` (University of Tokyo)

**通讯引用:** 296 | [OpenAlex ID](https://openalex.org/A5107897317)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在冲突约束下，如何在图模型中为两位或多位代理分配不可分物品，使得分配既满足EF1（或EF[1,1]）又是最大化的。

**💡 创新点**

创新点在于提出了“颜色切换”技术证明两代理单调价值下EF1+最大化分配总是存在，并将该问题与图的“环+三角”与“一般化为n-克利克”定理相结合，获得路径图上多代理的EF[1,1]+最大化解存在性。

**🔧 技术方法**

主要使用的技术包括图论中的最大独立集、可归约性与交换法、贪心区间调度、以及对冲突图进行重构构造可3/ n 颜色图，从而得到可行分配；算法方面对两代理的单调价值给出了伪多项式、区间/二分图下的多项式算法。

**📊 数据集**

论文基于理论证明，不使用公开数据集，所有结果均在抽象图模型与价值函数上进行推导。

**📈 对比分析**

由于缺乏实验基准，文中没有具体性能对比；主要通过证明存在性与算法时间复杂度（伪多项式/多项式）来展示方法的可行性与有效性。

**⚠️ 局限性**

局限性包括：对三位及更多代理的单调价值缺乏正向结果；对路径图上n≥3代理的EF1+最大化仍为NP‑难；对一般图的两代理单调价值算法的多项式性仍未解决；并且对冲突图为树时的均衡最大化色彩问题仍未完全一般化。

---

## 460. Sequential Feature Selection for Efficient Landslide Segmentation from Multi-Spectral Data

**arXiv ID:** 2605.09746 | [PDF](https://arxiv.org/pdf/2605.09746v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 461. Not All Thoughts Need HBM: Semantics-Aware Memory Hierarchy for LLM Reasoning

**arXiv ID:** 2605.09490 | [PDF](https://arxiv.org/pdf/2605.09490v1)

**作者:** Aojie Yuan `[一作]` (University of Southern California), Dajun Zhang `[通讯]` (University of Wisconsin--Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了面向推理型大型语言模型的四层语义感知内存层次结构，将 KV 缓存按重要性动态划分为 HBM、DDR、压缩和永久驱逐四个层级；

**💡 创新点**

创新点在于将推理令牌的长期重要性利用累积注意力评分实现无误差的 DDR 离线存储，解决传统驱逐导致的准确性崩溃问题；

**🔧 技术方法**

核心技术包括累计注意力重要性评分、实时层次管理、异步 PCIe 预取与回收以及零逼近误差理论证明；

**📊 数据集**

实验使用 DeepSeek‑R1‑Distill‑Qwen 系列（7B、14B、32B）在 GSM8K、MATH‑500、MATH Level‑5、ARC‑Challenge 四个推理基准上进行验证；

**📈 对比分析**

与纯驱逐、H2O 以及 R‑KV 等基线对比，保持 3–5% 驱逐率即可恢复 90% 以上完整缓存准确率；在 50% HBM 缓存下准确率仅为 8–10%，而四层层次可在同等 HBM 预算下保持 71% 以上准确率；

**⚠️ 局限性**

局限包括对压缩层 T2 的精度敏感性（8‑bit 量化严重下降）、对不同模型架构的跨验证不足、以及在极大规模（70B+）下多 GPU 环境的实测验证尚待完成。

---

## 462. RubricRefine: Improving Tool-Use Agent Reliability with Training-Free Pre-Execution Refinement

**arXiv ID:** 2605.09730 | [PDF](https://arxiv.org/pdf/2605.09730v1)

**作者:** Will LeVine `[一作]` (Anduril Industries), Abhay Venkatesh `[通讯]` (Anduril Industries)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为 RubricRefine 的预执行语义合同验证方法，通过生成任务特定的 rubric（检查清单）并在生成、评估、修复循环中使用 LLM 对代码模式工具使用进行迭代修复，最终在单次执行前确保程序满足所有工具合同。

**💡 创新点**

核心创新在于将结构化、样本依赖的 rubric 作为反馈信号，替代传统的无结构自我修正或执行反馈；证明这种结构化合同验证在多步工具组合任务上显著提升可靠性，并且无需任何额外训练；同时通过对 rubric 评分门控实现高效早停。

**🔧 技术方法**

技术实现主要包括：<br>- LLM 生成 rubric（由与任务提示和工具文档共同决定）；<br>- LLM 评估候选代码与 rubric 的一致性，输出 1–10 分及 PASS/FAIL 细化反馈；<br>- 基于 CodeAct 的可执行代码模式框架，执行 generate-verify-revise 循环；<br>- 采用 ordinal 评分门控和早停策略。

**📊 数据集**

实验使用了两个公开基准：<br>- M3ToolEval（多步工具组合、数据流依赖任务）；<br>- API-Bank（单步 API 调用准确性评估）。

**📈 对比分析**

与 CodeAct、Self-Refine、Self-Debug、Best-of-N 等基线在同一单次执行协议下进行对比。<br>在 M3ToolEval 上，RubricRefine 在所有七种模型上均取得最高成功率，提升幅度约 0.14–0.38；在 API-Bank 上表现与基线持平。<br>相较于最强的 rubric-guided reranking，RubricRefine 以 2.6 倍更低的延迟、48% 更少的 token、成本更低的优势显著。

**⚠️ 局限性**

局限性包括：<br>- 仅适用于代码模式、显式工具注册表的场景；<br>- 无法捕获依赖运行时状态或外部环境的错误；<br>- 受验证器质量影响，若 rubric 缺失关键约束或评分不准则会导致错误修复方向；<br>- 需要额外的推理时间和算力；<br>- 评估仅覆盖 M3ToolEval 与 API-Bank，缺乏对更广泛工具使用任务的验证。

---

## 463. The Cartesian Shortcut: Re-evaluate Vision Reasoning in Polar Coordinate Space

**arXiv ID:** 2605.09883 | [PDF](https://arxiv.org/pdf/2605.09883v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 464. Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization

**arXiv ID:** 2605.10067 | [PDF](https://arxiv.org/pdf/2605.10067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 465. RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation

**arXiv ID:** 2605.09907 | [PDF](https://arxiv.org/pdf/2605.09907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 466. Flag Varieties: A Geometric Framework for Deep Network Alignment

**arXiv ID:** 2605.09861 | [PDF](https://arxiv.org/pdf/2605.09861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 467. Parameter-Efficient Neuroevolution for Diverse LLM Generation: Quality-Diversity Optimization via Prompt Embedding Evolution

**arXiv ID:** 2605.09781 | [PDF](https://arxiv.org/pdf/2605.09781v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 468. Above and Below: Heterogeneous Multi-robot SLAM Across Surface and Underwater Domains

**arXiv ID:** 2605.09811 | [PDF](https://arxiv.org/pdf/2605.09811v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 469. MicroWorld: Empowering Multimodal Large Language Models to Bridge the Microscopic Domain Gap with Multimodal Attribute Graph

**arXiv ID:** 2605.10120 | [PDF](https://arxiv.org/pdf/2605.10120v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 470. Evolving Knowledge Distillation for Lightweight Neural Machine Translation

**arXiv ID:** 2605.09924 | [PDF](https://arxiv.org/pdf/2605.09924v1)

**作者:** Xuewen Zhang `[一作]` (Li Auto), Xinlong Huang `[通讯]` (Li Auto)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实验了一种Evolving Knowledge Distillation（EKD）框架，使单一学生模型通过逐级学习更大容量教师模型的知识，提升NMT翻译质量。

**💡 创新点**

创新点在于：①引入多级教师层级的递进式蒸馏策略，逐步缩小学生与教师的容量差距；②通过连续学习实现学生性能的持续提升，甚至能突破初始教师的表现；③与传统单教师或教师助手蒸馏方法对比，证明了更高的BLEU与COMET得分。

**🔧 技术方法**

技术手段包括：Transformer架构、知识蒸馏（KL散度与soft标签）、标签平滑交叉熵、Adam优化、学习率调度、BPE分词、Fairseq框架；同时构建了教师-学生容量层级与课程学习视角。

**📊 数据集**

使用公开数据集：IWSLT-14 De‑En、WMT‑17 En‑De、WMT‑23 En‑Cs；并在每个数据集上分别训练学生、junior、senior教师模型。

**📈 对比分析**

通过BLEU与COMET指标与单教师蒸馏、Teacher‑Assistant KD（TAKD）进行对比；EKD在IWSLT‑14上BLEU从32.78提升至34.24，仅落后0.08 BLEU；在其他两个数据集也实现约1–3 BLEU点提升；相较TAKD，学生模型提升约5.8% BLEU。总的来说，EKD在保持小模型尺寸的前提下，显著缩小了学生与教师之间的性能差距。

**⚠️ 局限性**

局限性包括：①实验仅在少量超参数配置下验证，缺乏更系统的调优；②仅在教师和学生同属Transformer族时测试，未探索不同架构（如Mamba等）间的迁移效果。

---

## 471. Optimizing Server Placement for Vertical Federated Learning in Dynamic Edge/Fog Networks

**arXiv ID:** 2605.09813 | [PDF](https://arxiv.org/pdf/2605.09813v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 472. RDEx-CASK: Cauchy Mutation, Archive, and Stagnation Kick for RDEx-CSOP

**arXiv ID:** 2605.09652 | [PDF](https://arxiv.org/pdf/2605.09652v1)

**作者:** Dikshant `[一作]` (International Institute of Information Technology), Senthilnath Jayavelu `[通讯]` (Advanced Science And Technology Institute)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在RDEx-CSOP基础上，提出RDEx-CASK算法，通过三项结构改进和参数细调提升可行性感知差分进化的搜索速度与性能；

**💡 创新点**

①将第二尺度因子F2独立采样自截断Cauchy分布；②引入可行性限定的JADE式外部归档并以概率采样；③针对停滞个体实施全局最优拉近、归档概率底线与交叉率饱和的局部覆盖机制；

**🔧 技术方法**

差分进化、可行性感知排名、SHADE记忆、自适应参数控制、Cauchy采样、外部归档、交叉率调节、停滞检测；

**📊 数据集**

CEC 2026 CSOP 28个约束优化问题（维度30），最大评估次数20000D；

**📈 对比分析**

与RDEx、UDE-III、CL‑SRDE在可行性感知最终质量Q_p和时间到目标TTT进行25次独立跑比较；RDEx-CASK在多数问题上加速显著，最终质量与RDEx相当或略优；在多模态问题上偶有精度略降；

**⚠️ 局限性**

对多模态高耦合问题可能牺牲最终精度；归档容量与停滞阈值需经验调优，算法对这些超参数敏感；未验证更高维或不同约束形式的鲁棒性；

---

## 473. TeleResilienceBench: Quantifying Resilience for LLM Reasoning in Telecommunications

**arXiv ID:** 2605.09929 | [PDF](https://arxiv.org/pdf/2605.09929v1)

**作者:** Pranshav Gajjar `[一作]` (North Carolina State University), Vijay K Shah `[通讯]` (North Carolina State University)

**通讯引用:** 1044 | [OpenAlex ID](https://openalex.org/A5083496212)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了 TeleResilienceBench，评估电信领域大语言模型在已出现错误推理后继续推理并纠正的能力。

**💡 创新点**

创新点在于提出“推理弹性”概念，并通过弱生成器产生错误推理、截断并让目标模型继续完成，从而构造真实场景的错误恢复任务；同时定义了 Correct Flip Rate、No Flip Rate、Wrong Flip Rate 等量化指标。

**🔧 技术方法**

使用链式思考、迭代修正框架以及自监督推理步骤的技术；对八个开放权重模型（Qwen3.5、Gemma4、Nemotron-3 等）在统一 prompt 模板下进行评估；结合输出 token 与 VRAM 使用等效率度量。

**📊 数据集**

基于 GSMA Open‑Telco LLM 套件的七个子集（TeleQnA、TeleTables、TeleLogs、3GPP_TSG、ORANBench、srsRANBench、SixG_Bench）构成 818 个离散选择实例；并包含 TeleMath 开放式数值题作为辅助评测。

**📈 对比分析**

采用宏平均 Correct Flip Rate（CFR）、No Flip Rate（NFR）和 Wrong Flip Rate（WFR）对八个模型进行比较；结果显示即使是最大的 Gemma4‑31b 模型宏平均 CFR 仅 29.1%，而 Nemotron‑3‑nano‑4b 在参数量极低的情况下实现 27.5% CFR，证明规模不一定带来更好弹性；不同子域差异明显，3GPP_TSG 最高难度却最低恢复率。

**⚠️ 局限性**

局限性包括：评测仅基于自动生成错误推理，缺乏人工核实；现有子域难度标签更多反映事实覆盖而非推理深度；只评估中小模型，未验证更大前沿模型的弹性；未覆盖所有可能的推理失效场景。

---

## 474. ScaleGANN: Accelerate Large-Scale ANN Indexing by Cost-effective Cloud GPUs

**arXiv ID:** 2605.10135 | [PDF](https://arxiv.org/pdf/2605.10135v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 475. Attention Drift: What Autoregressive Speculative Decoding Models Learn

**arXiv ID:** 2605.09992 | [PDF](https://arxiv.org/pdf/2605.09992v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 476. A Resource Allocation Game and its Equilibrium Strategies

**arXiv ID:** 2605.09988 | [PDF](https://arxiv.org/pdf/2605.09988v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 477. GraphInstruct: A Progressive Benchmark for Diagnosing Capability Gaps in LLM Graph Generation

**arXiv ID:** 2605.09997 | [PDF](https://arxiv.org/pdf/2605.09997v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 478. DetRefiner: Model-Agnostic Detection Refinement with Feature Fusion Transformer

**arXiv ID:** 2605.10190 | [PDF](https://arxiv.org/pdf/2605.10190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 479. LeapTS: Rethinking Time Series Forecasting as Adaptive Multi-Horizon Scheduling

**arXiv ID:** 2605.10292 | [PDF](https://arxiv.org/pdf/2605.10292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 480. PolarVSR: A Unified Framework and Benchmark for Continuous Space-Time Polarization Video Reconstruction

**arXiv ID:** 2605.10275 | [PDF](https://arxiv.org/pdf/2605.10275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 481. Just Previsions

**arXiv ID:** 2605.10173 | [PDF](https://arxiv.org/pdf/2605.10173v1)

**作者:** Jean Goubault-Larrecq `[一作]` (Université Paris-Saclay), Jean Goubault-Larrecq `[通讯]` (Université Paris-Saclay)

**通讯引用:** 1954 | [OpenAlex ID](https://openalex.org/A5032396554)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了概率预期（prevision）的空间结构，并将其表示为在子线性预期空间上的双重超空间（double hyperspace）形式；

**💡 创新点**

创新点在于揭示了预期空间与子线性预期空间之间的同构关系，并将其归纳为正交性（orthogonality）构造，从而实现了对预期空间的精确描述与构造；

**🔧 技术方法**

主要技术包括域理论（domain theory）、Scott拓扑、凸锥结构（convex cones）、Hoare与Smyth超空间构造以及Keimel的夹挤定理（sandwich theorem）等；

**📊 数据集**

由于本文为纯理论研究，没有使用实验数据集；

**📈 对比分析**

方法上通过构造同构映射、重排同构证明以及正交性双重映射来进行比较，理论上证明了所给构造在满足局部凸性或方便锥（convenient cone）条件下的等价与保留性质；

**⚠️ 局限性**

局限性在于结果仅在基空间满足局部凸性或方便锥条件（以及“1”情形下的紧致性）时成立，且在一般情况下预期空间与其子空间之间的上/下极限映射不保持线性或仿射性，导致某些结构映射不再保留原始运算的可加性与凸组合性质。

---

## 482. Active Tabular Augmentation via Policy-Guided Diffusion Inpainting

**arXiv ID:** 2605.10315 | [PDF](https://arxiv.org/pdf/2605.10315v1)

**作者:** Zheyu Zhang `[一作]` (Technical University of Munich), Gjergji Kasneci `[通讯]` (Technical University of Munich)

**通讯引用:** 14908 | [OpenAlex ID](https://openalex.org/A5024434748)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为Tabular Augmentation Policy（TAP）的表格数据增广框架，通过控制扩散修复过程和注入时机来提升低样本环境下的下游模型性能。

**💡 创新点**

创新点在于将扩散模型的inpainting与基于学习者状态的轻量策略网络相结合，解决了“fidelity‑utility gap”，并引入硬门控和窗口式保守注入机制以保证增广的安全性。

**🔧 技术方法**

使用的核心技术包括扩散inpainting、轻量级状态条件策略、TabPFN评估器、门控与窗口化提交、以及对信息性与可学习性的诊断分析。

**📊 数据集**

在七个真实世界表格数据集（MiceProtein、Credit‑G、Electricity、Fourier、Steel、Ailerons、Insurance）以及多种样本稀缺度（20/50/100/200/500）下进行实验。

**📈 对比分析**

与SMOTE、TVAE、CTGAN、ARF、SPADA、TabDDPM、TabDiff等基线相比，TAP在所有稀缺水平上均表现最佳，分类准确率提升至15.6个百分点，回归RMSE下降32%。

**⚠️ 局限性**

主要局限在于多步增广过程引入额外计算开销，且对评估器的可靠性和门控阈值设定具有一定敏感性。

---

## 483. Generalization Error Bounds for Picard-Type Operator Learning in Nonlinear Parabolic PDEs

**arXiv ID:** 2605.10277 | [PDF](https://arxiv.org/pdf/2605.10277v1)

**作者:** Koichi Taniguchi `[一作]` (Shizuoka University), Sho Sonoda `[通讯]` (RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对非线性抛物型偏微分方程，本文构建了一套以 Duhamel 原理与 Picard 迭代为核心的泛化理论，利用抽象状态转移模型把 Picard 迭代深度与统计复杂度分离，并给出了实现误差、截断误差、估计误差以及有限观测重构误差的分解式。

**💡 创新点**

创新点在于：① 将 Picard 迭代视为抽象状态转移，证明深度增加可降低截断误差而不提升 Rademacher 复杂度；② 形成了“实现无关”即“implementation‑agnostic”的贝叶斯式误差上界；③ 将局部模型通过 roll‑out 延伸到长期预测，并给出稳定性条件；④ 在非线性热方程上用 Fourier 神经算子实现并给出实现误差与 Rademacher 复杂度的闭式估计。

**🔧 技术方法**

核心技术包括：Duhamel 公式、Picard 迭代、抽象状态转移框架、Rademacher 复杂度与 Dudley 熵积分、L∞ 稳定的 Fourier 截断、ReLU 网络逼近 Lipschitz 非线性、以及概率不等式（马尔科夫、向量收缩）。

**📊 数据集**

实验数据主要采用从 Sobolev 球采样的合成初始值（隐式 Gaussian 过程），并在二维/三维环面上构造非线性热方程，使用 Fejér 截断与 Fourier 乘子实现数值模拟。

**📈 对比分析**

由于论文聚焦理论证明，主要通过解析误差上界与 Rademacher 复杂度比较，而非数值实验；理论结果表明：Picard 深度为 O(log n) 即可把截断误差压到统计噪声级别，且在充分重构与采样条件下误差随样本数 n 以 1/√n 收敛。

**⚠️ 局限性**

局限性包括：① 依赖于全局收敛的 δ‑contractive 假设，适用于满足该条件的非线性抛物方程；② 对实现误差与重构误差的估计相对粗糙，实际数值可能更大；③ 主要针对完整信息或均匀传感器场景，对稀疏/非均匀观测的适应性不足；④ 结果基于理论上 Rademacher 上界，实际性能受网络容量、优化难度等因素影响。

---

## 484. PowerStep: Memory-Efficient Adaptive Optimization via $\ell_p$-Norm Steepest Descent

**arXiv ID:** 2605.10335 | [PDF](https://arxiv.org/pdf/2605.10335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 485. Meta-Black-Box Optimization Can Do Search Guidance for Expensive Constrained Multi-Objective Optimization

**arXiv ID:** 2605.10260 | [PDF](https://arxiv.org/pdf/2605.10260v1)

**作者:** Yukun Du `[一作]` (National University of Defense Technology), Shengkun Chang `[通讯]` (National University of Defense Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MetaSG‑SAEA 双层框架，利用 Meta‑Black‑Box Optimization（MetaBBO）为昂贵约束多目标优化（ECMOP）提供搜索区域级别的引导，并通过扩散模型进行种群初始化，显著提升评估效率。

**💡 创新点**

创新点在于：① 将搜索“去哪里”作为外部搜索指导，区别于以往只优化搜索“怎么做”的策略；② 设计了 Max–Min Constraint‑Calibrated Inequality (MM‑CCI) 的区域抽象，既保持问题无关性又提供可排序的搜索优先级；③ 采用扩散模型根据目标与MM‑CCI信号生成初始种群，减少探索成本；④ 用 Transformer‑based attention ELA 仅依赖目标与约束信息，跨维度、跨目标、跨约束可扩展。

**🔧 技术方法**

技术手段包括：Meta‑BBO 双层学习（上层 DQN + 下层 NSGA‑II+GP 代理）；MM‑CCI 约束映射与层级划分；扩散模型（denoising diffusion probabilistic model）用于种群初始化；Gaussian Process 作为 surrogate；Transformer‑attention 用于构建跨任务通用 ELA；强化学习中的 Double‑DQN、软目标网络、并行采样（Ray）。

**📊 数据集**

实验基准为 MW 与 DAS‑CMOP 约束多目标优化集合；在测试阶段还使用 DC‑DTLZ、C‑DTLZ 等无约束/约束多目标基准以评估跨任务泛化。

**📈 对比分析**

通过与 KTS、EIC‑MSSAEA、DRLOS、DRL‑SAEA 等最先进方法在同一评估预算（FE_max=300）下对比，使用 IGD 作为性能指标。MetaSG‑SAEA 在绝大多数任务上均优于基线，尤其在零shot 转移和多目标可行前沿逼近方面表现突出，平均 IGD 显著下降。

**⚠️ 局限性**

局限性：① 仅提供外部搜索指导，未结合内部操作器/填充策略的动态控制；② 对 MM‑CCI 超参数的估计依赖初始 LHS 样本，可能在极端约束情形下失效；③ 当可行域极其稀疏时，区域抽象的梯度信息可能不足，导致搜索停滞；④ 需要在大规模任务分布上预训练，训练成本仍较高。

---

## 486. PaperFit: Vision-in-the-Loop Typesetting Optimization for Scientific Documents

**arXiv ID:** 2605.10341 | [PDF](https://arxiv.org/pdf/2605.10341v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 487. An Annotation Scheme and Classifier for Personal Facts in Dialogue

**arXiv ID:** 2605.10339 | [PDF](https://arxiv.org/pdf/2605.10339v1)

**作者:** Konstantin Zaitsev `[一作]` `[通讯]` (HSE University), Konstantin Zaitsev (HSE University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对个人事实进行手工注释并训练多头Transformer分类器，对事实进行七维度标注，完成从原始对话到结构化事实的转换。

**💡 创新点**

提出了扩展的注释方案，加入了 Demographics、Possessions 等新类别，以及 Duration、Validity、Followup 等属性；设计了多头分类架构，可在同一模型中并行预测多维标签。

**🔧 技术方法**

使用了 Transformer 编码器（Gemma‑300M、BGE‑large、RoBERTa‑large 等）和多头分类头；对比了传统 ML（SVM、LR、GB）和少量提示 LLM（GPT‑5、Llama‑3、Qwen 等）。

**📊 数据集**

主要数据集为 2,779 条手工标注的 Multi‑Session Chat 事实；在 PersonaChat 与完整 MSC 上进行分布推断与评估。

**📈 对比分析**

通过 70/10/20 的分割、5 轮随机种子进行实验；Gemma‑300M+多头分类器获得 81.6 ± 2.6% macro‑F1，明显优于最佳少量提示 LLM（GPT‑5.4‑mini 72.9%）且显著降低计算成本。

**⚠️ 局限性**

局限性包括：语义边界模糊、时态解析不准、所有权歧义、Followup 标签稀缺导致的误判；对 Validity Reason 的细粒度区分仍需要人工监督。

---

## 488. EvoStreaming: Your Offline Video Model Is a Natively Streaming Assistant

**arXiv ID:** 2605.10343 | [PDF](https://arxiv.org/pdf/2605.10343v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 489. EmbodiSkill: Skill-Aware Reflection for Self-Evolving Embodied Agents

**arXiv ID:** 2605.10332 | [PDF](https://arxiv.org/pdf/2605.10332v1)

**作者:** Ruofei Ju `[一作]` (Nanjing University), Ting Cao `[通讯]` (Tsinghua University)

**通讯引用:** 2683 | [OpenAlex ID](https://openalex.org/A5074166453)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 EmbodiSkill 框架，使具身智能体通过技能自我演化改进任务成功率。

**💡 创新点**

将轨迹转化为针对特定技能内容的反思与修订，避免粗粒度技能重写；构建技能演化螺旋，实现持续的技能改进。

**🔧 技术方法**

利用大型语言模型（GPT‑5.2 或 Gemini‑3‑flash）进行技能反思、合并修订与附录更新；固定执行器 Qwen 系列；轨迹生成与技能评估。

**📊 数据集**

ALFWorld、EmbodiedBench（Habitat 与 Navigation）数据集。

**📈 对比分析**

与直接代理（GPT‑5.2/Gemini）及内存基方法（Mem0、G‑Memory、LangMem）对比，EmbodiSkill 在 ALFWorld 上最高达 93.28% 成功率，比直接代理高 31.58%，在 EmbodiedBench 也显著优于基线。

**⚠️ 局限性**

依赖大模型导致计算成本高；需要手工设定初始技能与反思阈值；在极端或未见任务中泛化能力仍有限。

---

## 490. ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models

**arXiv ID:** 2605.10328 | [PDF](https://arxiv.org/pdf/2605.10328v1)

**作者:** Wentao Qiu `[一作]` (Xiamen University), Qingqiang Wu `[通讯]` (Xiamen University)

**通讯引用:** 580 | [OpenAlex ID](https://openalex.org/A5048017759)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Anchor 框架，通过 LLM 生成并层次化因子空间，结合朴素贝叶斯和因果贝叶斯网络实现可靠概率推理。

**💡 创新点**

创新点在于（1）分离因子生成与结构化的底向上推理；（2）层次检索映射降低未知率；（3）引入因果 BN 捕获因子间依赖，提升校准。

**🔧 技术方法**

使用大型语言模型（Qwen、DeepSeek、GPT‑4）进行因子抽取与参数推断，MiniLM+UMAP+HDBSCAN 构建因子层级，BERT/CoT 提示式检索，线性意见池或 BMA 融合推理结果。

**📊 数据集**

在 Common2Sense、Plasma、Today、ExpertQA、XSum、COVID、CNN 等多种推理、规划和事实核查数据集上进行评估。

**📈 对比分析**

与 Bird、Vanilla、CoT、Logits、Compare 等基线对比，Anchor 在 F1、覆盖率和决策准确率上均显著优于基线，尤其在常规 LLM 规模下已达或超过更大模型，且推理时间和 token 消耗更低。

**⚠️ 局限性**

局限在于依赖 LLM 抽象与参数的可靠性，可能引入偏见与幻觉，未知率下降可能导致错误自信，且在分布外或对抗性输入下表现未充分验证。

---

## 491. CORTEG: Foundation Models Enable Cross-Modality Representation Transfer from Scalp to Intracranial Brain Recordings

**arXiv ID:** 2605.10337 | [PDF](https://arxiv.org/pdf/2605.10337v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 492. DECO-MWE: building a linguistic resource of Korean multiword expressions for feature-based sentiment analysis

**arXiv ID:** 2605.10295 | [PDF](https://arxiv.org/pdf/2605.10295v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 493. Predictive Radiomics for Evaluation of Cancer Immune SignaturE in Glioblastoma: the PRECISE-GBM study

**arXiv ID:** 2605.10278 | [PDF](https://arxiv.org/pdf/2605.10278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 494. Verifiable Process Rewards for Agentic Reasoning

**arXiv ID:** 2605.10325 | [PDF](https://arxiv.org/pdf/2605.10325v1)

**作者:** Huining Yuan `[一作]` (Tsinghua University), Yi Wu `[通讯]` (Tsinghua University)

**通讯引用:** 7805 | [OpenAlex ID](https://openalex.org/A5056768784)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Verifiable Process Rewards (VPR)，通过将任务特定的符号或算法验证器转化为密集、可验证的过程奖励，来训练大型语言模型从事多步推理任务。

**💡 创新点**

创新点在于：①引入密集、可验证的过程奖励取代稀疏终端奖励；②将不同类型的验证器（搜索、约束求解、后验推理）统一映射为奖励；③理论分析证明密集奖励能在长时间步长上更好地进行信用分配。

**🔧 技术方法**

使用的技术包括：强化学习（GRPO）、Monte Carlo Tree Search (MCTS)、约束求解器、概率后验计算，以及 Qwen3-4B 语言模型的思考模式。

**📊 数据集**

实验数据集涵盖三种可验证推理环境（Tic‑Tac‑Toe、Sudoku、Minesweeper），并在七个通用推理基准（GSM8K、MATH‑500、AIME24/25、GPQA‑D、BBH、MMLU‑P）以及两个代理任务（ALFWorld、WebShop）进行零样本迁移评估。

**📈 对比分析**

与仅使用终端奖励（OR）和基于 Monte Carlo 的过程奖励（MC‑PR）对比，VPR 在所有环境中都实现了更高的成功率/完成率、收敛速度更快、迁移性能提升（平均提升约 4–8%），证明了密集可验证奖励的有效性。

**⚠️ 局限性**

局限性包括：依赖高质量、可验证的中间步骤；当验证器误差较大时会导致信用分配错误并抑制学习；目前仅适用于结构化、可完整检验的任务，难以直接扩展到开放式、无明确验证规则的真实世界场景。

---

## 495. Increasing the Efficiency of DETR for Maritime High-Resolution Images

**arXiv ID:** 2605.10269 | [PDF](https://arxiv.org/pdf/2605.10269v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 496. Sample-Mean Anchored Thompson Sampling for Offline-to-Online Learning with Distribution Shift

**arXiv ID:** 2605.10289 | [PDF](https://arxiv.org/pdf/2605.10289v1)

**作者:** Bochao Li `[一作]` (Southern University of Science and Technology), Fang Kong `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 800 | [OpenAlex ID](https://openalex.org/A5102803936)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9`

**🎯 论文内容**

提出了样本均值锚定 Thompson Sampling（Anchor‑TS）算法，用于在离线-在线学习中利用离线数据并应对离线与在线环境的分布偏移。

**💡 创新点**

创新点：①使用中位数聚合三种指标（在线样本均值、在线后验样本、混合后验样本）实现对离线数据可靠性的自适应判断；②对混合后验加入右移校正，避免对最优 arm 的低估；③在理论上给出在已知偏移上界时的下界级别 regret 上的显著改进，且能利用最优 arm 上的离线样本提升性能。

**🔧 技术方法**

技术手段：Thompson Sampling、贝叶斯高斯后验、样本均值锚定、三元中位数聚合、偏移校正、基于分布偏移上界的分析。

**📊 数据集**

使用仿真数据：10 把带有 Gaussian/子高斯奖励的多臂老虎机；离线样本在不同覆盖模式（均匀、集中于最优 arm、集中于次优 arm）与不同偏移量、样本规模下生成。

**📈 对比分析**

与标准 TS、UCB、混合 TS/UCB、MINUCB 等方法对比；Anchor‑TS 在所有实验设置下均显著低于基线，尤其在离线数据集中最优 arm、以及存在较大分布偏移时表现最佳，理论上实现了比纯在线 TS 更优的常数项。

**⚠️ 局限性**

局限：仅给出与间隔相关的 regret 上界；未提供无间隔（gap‑independent）分析；仅适用于无情境的随机 bandit；需事先知道离线-在线偏移上界 V，实际应用中该信息可能难以获得。

---

## 497. QuantWeather: Quantile-Aware Probabilistic Forecasting for Subseasonal Precipitation

**arXiv ID:** 2605.10297 | [PDF](https://arxiv.org/pdf/2605.10297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 498. IndustryBench: Probing the Industrial Knowledge Boundaries of LLMs

**arXiv ID:** 2605.10267 | [PDF](https://arxiv.org/pdf/2605.10267v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 499. Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning

**arXiv ID:** 2605.10247 | [PDF](https://arxiv.org/pdf/2605.10247v1)

**作者:** Dario Vajda `[一作]` `[通讯]` (University of Ljubljana), Dario Vajda (University of Ljubljana)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Graph Transformer Language Model (GTLM)，一种将大型语言模型直接与图结构结合的架构，能够在不压缩节点文本的前提下进行多跳推理。

**💡 创新点**

创新点在于：①通过将图的拓扑信息（SPD、RRWP、磁拉普拉斯）作为可学习的注意力偏置注入LLM，自然保持节点文本完整；②保持前缀-LLM的双向注意力，保证对单节点文本与原始预训练模型的等价性；③参数量极小，仅增加约0.015%。

**🔧 技术方法**

主要技术包括：相对位置编码 (RoPE) 的节点内重置、相对路径距离 (SPD) 和随机游走概率 (RRWP) 的可学习偏置、磁拉普拉斯谱特征的双线性映射、LoRA适配器、Prefix-LM双向掩码以及基于注意力偏置的隐式消息传递。

**📊 数据集**

使用的基准数据集包括：GraphQA、Cora、Pubmed、OGBN-Arxiv、Reddit、synthetic Family Tree 与 KG-QA；此外还对不同规模的基础模型（1B、3B、8B）进行了评估。

**📈 对比分析**

与多种基线（GraphToken、RGLM、GraphGPT、LLaGA 等）和传统 GNN（GCN、GraphSAGE、GAT 等）相比，GTLM 在 GraphQA、Text-Attributed Graph 分类、Family Tree 与 KG-QA 等任务中均实现或超过 7B 级模型的性能；在 1B 参数模型下即可达到或接近 7B 模型的准确率，且在大多数任务上取得显著提升。

**⚠️ 局限性**

主要限制：①自定义注意力偏置导致无法使用硬件加速的 FlashAttention，计算复杂度为 O(N²)，训练时间约为基线的 3 倍；②完整文本序列的长度限制了对大图或长节点文本的扩展；③在极大上下文窗口下的内存与时间成本高，需进一步开发稀疏或 K-hop 注意力机制。

---

## 500. E-TCAV: Formalizing Penultimate Proxies for Efficient Concept Based Interpretability

**arXiv ID:** 2605.10261 | [PDF](https://arxiv.org/pdf/2605.10261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 501. Drum Synthesis from Expressive Drum Grids via Neural Audio Codecs

**arXiv ID:** 2605.10281 | [PDF](https://arxiv.org/pdf/2605.10281v1)

**作者:** Konstantinos Soiledis `[一作]` (Hellenic Mediterranean University), Konstantinos Tsamis `[通讯]` (Hellenic Mediterranean University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5117588445)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种从表达式鼓网格到鼓音频的条件生成系统，利用Transformer预测神经音频编码器的离散码词并通过预训练解码器生成波形。

**💡 创新点**

创新点在于首次系统比较不同神经音频码词（EnCodec、DAC、X-Codec）在鼓网格条件生成中的可行性，并将鼓网格与码词空间对齐，揭示码词可学习性对音频质量的影响。

**🔧 技术方法**

采用非自回归Transformer编码器对鼓网格进行码词预测，结合预训练的神经音频编码器（EnCodec、DAC、X-Codec）进行解码，并在E-GMD数据集上训练与评估。

**📊 数据集**

使用的主要数据集是Expanded Groove MIDI Dataset（E-GMD），包含约444小时的鼓音频与对应MIDI，提供高质量的鼓网格与音频配对。

**📈 对比分析**

通过token层面的NLL/Perplexity、音频层面的RMSE/MAE、MR-STFT谱收敛、起始对齐F1以及FAD等多指标进行比较，结果显示EnCodec在token学习和音频质量上优于DAC和X-Codec；然而更大模型导致性能下降。

**⚠️ 局限性**

局限性包括对码词空间可学习性的高度依赖、缺乏主观听感评估、以及大型模型训练不稳定，未能实现更高质量的音频生成。

---

## 502. LimeCross: Context-Conditioned Layered Image Editing with Structural Consistency

**arXiv ID:** 2605.10319 | [PDF](https://arxiv.org/pdf/2605.10319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 503. Towards Autonomous Railway Operations: A Semi-Hierarchical Deep Reinforcement Learning Approach to the Vehicle Rescheduling Problem

**arXiv ID:** 2605.10257 | [PDF](https://arxiv.org/pdf/2605.10257v1)

**作者:** Alberto Castagna `[一作]` (enliteAI), Anton Fuxjager `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Maze-Flatland，半层级多智能体强化学习框架，用于铁路车辆重新调度问题。

**💡 创新点**

创新点在于将调度与路径规划分离为两级决策（MADS与MAPF），通过专用观察/动作空间缓解调度稀疏性、提升可扩展性和鲁棒性。

**🔧 技术方法**

采用半层级多智能体强化学习技术，MADS调度策略与MAPF路径策略基于Flatland-RL环境，训练时使用行为克隆与MCTS搜索，并使用Transformer注意力网络。

**📊 数据集**

使用Flatland-RL仿真环境，包含5个不同密度级别（7至80列车）以及50个随机种子，地图与故障生成随机。

**📈 对比分析**

与Greedy、pp、Deadlock Avoidance、TreeLSTM基线对比；在80列车高负载下，Maze-Flatland成功率约50%，死锁率≤5%，平均延迟虽略高但显著优于基线，整体提升吞吐量和稳定性。

**⚠️ 局限性**

限制包括对恒定速度假设的依赖、未考虑多目标（延迟、吞吐、资源利用）优化、通信完好假设以及在分数速度场景下的性能尚未验证；MADS过于保守导致取消率较高。

---

## 504. Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation

**arXiv ID:** 2605.10253 | [PDF](https://arxiv.org/pdf/2605.10253v1)

**作者:** Peiru Yang `[一作]` (Tsinghua University), Tao Qi `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 5247 | [OpenAlex ID](https://openalex.org/A5111661897)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种针对医疗多模态检索增强生成系统的知识中毒框架，假设攻击者仅具备知识库分布的有限信息，能够在不知查询的前提下通过隐蔽扰动和模糊文本注入破坏诊断与报告生成；

**💡 创新点**

创新点在于：①基于分布导向的检索劫持，将视觉扰动与语义聚类结合，使用对抗性PGD在查询无关的条件下提升被劫持样本的检索概率；②利用医学诊断的固有模糊性设计三阶段文本误导策略，避免模型自纠正；③在弱攻击先验下实现对闭源与开源大型视觉语言模型的有效攻击；

**🔧 技术方法**

技术包括：对图像的对抗性PGD优化（白盒/黑盒均可），基于K‑means聚类的检索代理目标构建，语义聚类与平均向量生成，LLM编辑器实现语义模糊文本注入；

**📊 数据集**

使用五个医疗数据集：IU‑XRay、MIMIC‑CXR（胸部X‑ray+报告）、CRC100k、MHIST、PCam（组织病理图像）以及五个大型视觉语言模型（GPT‑4o、GPT‑5、Gemini‑2.5、Claude‑4.5、LLaVA‑Med）进行评估；

**📈 对比分析**

与基线文本RAG中毒方法（LIAR）对比，实验显示检索劫持成功率（ASR@Top‑k）在多种检索器（CLIP、BGE‑VL、SigLIP）上均高；在所有模型与数据集上，攻击导致整体下游任务效能平均下降约8.78%，且在成功劫持的查询子集上损伤更明显；在三种预检索防御（图像聚类、文本聚类、图像‑文本一致性）下攻击仍保持高效；

**⚠️ 局限性**

局限性包括：仅验证于二维医学影像（胸部X‑ray与组织病理），未对3D体积影像或医学视频进行实验；框架虽易扩展，但在高维模态的实际效果尚未评估。

---

## 505. Efficient Hybrid CNN-GNN Architecture for Monocular Depth Estimation

**arXiv ID:** 2605.10251 | [PDF](https://arxiv.org/pdf/2605.10251v1)

**作者:** Ishan Narayan `[一作]` `[通讯]` (CSIR-CSIO), Ishan Narayan (CSIR-CSIO)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了GraphDepth，一个将GraphSAGE嵌入ResNet-101 U‑Net中的多尺度卷积‑图网络混合架构，用于单目深度估计。

**💡 创新点**

创新点包括批量并行的k‑NN/网格图构造、多尺度GraphSAGE集成、通道注意力门控跳接、以及异方差不确定性头实现自适应损失加权。

**🔧 技术方法**

使用了ResNet‑101 U‑Net编码器、GraphSAGE消息传递、多尺度图网络、SE式通道注意力、aleatoric uncertainty head、FP16混合精度训练和批量并行GNN实现。

**📊 数据集**

实验基准涵盖NYU Depth V2（室内）、WHU Aerial（航拍）、ETH3D（立体）和Mid‑Air合成航拍数据集。

**📈 对比分析**

与CNN、Transformer（DPT、DepthFormer）等基线比较，GraphDepth在WHU Aerial上取得最优RMSE，在NYU Depth V2上与Transformer相差≤4.6%且速度提升≈2.9×，在Mid‑Air零样本迁移中δ₁提升≈10%且更快、更省显存。

**⚠️ 局限性**

主要限制包括k‑NN图构造的推理开销（≈15–20%延迟）、对稠密监督的依赖以及缺乏时序一致性处理。

---

## 506. Extending Confidence-Based Text2Cypher with Grammar and Schema Aware Filtering

**arXiv ID:** 2605.10318 | [PDF](https://arxiv.org/pdf/2605.10318v1)

**作者:** Makbule Gulcin Ozsoy `[一作]` `[通讯]` (Neo4j), Makbule Gulcin Ozsoy (Neo4j)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在Text2Cypher推理时引入结构约束，构建了从信心过滤到语法验证再到模式一致性检查的序列化后生成过滤流程；

**💡 创新点**

创新点在于首次将信心过滤与后生成的语法约束（ANTLR4解析）和数据库模式约束组合成多级过滤，系统评估各级约束对查询语法正确性和执行结果质量的影响；

**🔧 技术方法**

使用了大语言模型Gemma2和Qwen2.5作为生成器，DeepConf框架实现信心过滤，ANTLR4实现正式语法验证，规则式正则检查和关系方向校验实现模式约束；

**📊 数据集**

实验数据来自公开的Text2Cypher数据集（789个测试样本，涉及recommendations、companies、neoflix三大数据库）；

**📈 对比分析**

通过与基线模型、仅信心过滤、语法过滤（naive/ANTLR）和加模式过滤等多种组合对比，使用ROUGE‑L（词汇和执行结果）和执行成功率评估。结果显示语法过滤显著提升语法有效率，模式过滤进一步提升执行质量，但更严格的过滤导致空预测增多，覆盖率下降；

**⚠️ 局限性**

局限性包括：过滤步骤增加推理延迟、使用公开的OpenCypher语法可能不完整导致仍有执行错误、过强的约束易产生空预测且未对推理效率做细致分析。

---

## 507. Positive Alignment: Artificial Intelligence for Human Flourishing

**arXiv ID:** 2605.10310 | [PDF](https://arxiv.org/pdf/2605.10310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 508. Relations Are Channels: Knowledge Graph Embedding via Kraus Decompositions

**arXiv ID:** 2605.10317 | [PDF](https://arxiv.org/pdf/2605.10317v1)

**作者:** Sayan Kumar Chaki `[一作]` `[通讯]` (Inria), Sayan Kumar Chaki (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于Kraus分解的知识图谱嵌入框架 KrausKGE，并证明关系操作符应满足线性、迹保持和完全正性这三个公理。

**💡 创新点**

创新点在于把关系操作符的形式从经验设计转为由三条数学公理推导得到的 Kraus 通道结构，解决了多途径表达、路径组合闭合和实体范数约束等传统缺陷。

**🔧 技术方法**

采用了密度矩阵实体表示、Kraus 通道的完整性约束、Cayley 参数化实现无投影训练、以及自对抗负采样等技术。

**📊 数据集**

在 WN18RR、FB15k-237、YAGO3-10、NELL-995 等公开基准上进行评测。

**📈 对比分析**

与 TransE、RotatE、ConvE、GoldE、R-GCN 等多种主流模型对比，KrausKGE 在 N‑to‑N 关系上明显提升，整体 MRR 与 Hits@10 均超过基线，且多跳推理不需额外路径编码器。

**⚠️ 局限性**

局限包括参数与计算量显著增加（尤其是高 Kraus 维度），当前对每条关系的 Kraus 维度共享且未优化，超几何扩展仅在有限维条件下可行，且对不满足完整性约束的传统加法式模型不适用。

---

## 509. Mind Modeling: A ToM-Based Framework for Personalization

**arXiv ID:** 2605.10306 | [PDF](https://arxiv.org/pdf/2605.10306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 510. Follow the Mean: Reference-Guided Flow Matching

**arXiv ID:** 2605.10302 | [PDF](https://arxiv.org/pdf/2605.10302v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 511. AgentRx: A Benchmark Study of LLM Agents for Multimodal Clinical Prediction Tasks

**arXiv ID:** 2605.10286 | [PDF](https://arxiv.org/pdf/2605.10286v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 512. Qwen Goes Brrr: Off-the-Shelf RAG for Ukrainian Multi-Domain Document Understanding

**arXiv ID:** 2605.10296 | [PDF](https://arxiv.org/pdf/2605.10296v1)

**作者:** Anton Bazdyrev `[一作]` (National Technical University of Ukraine Igor Sikorsky Kyiv Polytechnic Institute), Artur Khodakovskyi `[通讯]` (National Technical University of Ukraine Igor Sikorsky Kyiv Polytechnic Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个检索增强的多文档问答流水线，解决乌克兰语多域PDF问答及页面定位任务。

**💡 创新点**

创新点在于结构感知的分块、问题-答案全信息检索与重排、以及对重排段落的受限生成，强调文档结构与答案空间的相关性。

**🔧 技术方法**

采用了Qwen3系列模型：Qwen3-Embedding-8B做检索、Qwen3-Reranker-8B做重排、Qwen3-32B-AWQ做答案生成，并使用vLLM实现高效推理。

**📊 数据集**

数据集包括竞赛训练/验证集（40/461题，41 PDF），UA‑SQuAD（16,658问答）以及自构建的80k示例检索预训练集，后者由四个英文检索数据集翻译而来。

**📈 对比分析**

与基线相比，文档上下文预加、答案全信息重排显著提升Recall@1与答案准确率；最终排行榜分数为公开0.9452、私有0.9598，超过所有对照模型。

**⚠️ 局限性**

主要限制是对大型模型（8B/32B）的依赖，且预训练语料为机器翻译，未人工校验，导致在极端领域泛化或细粒度语言细节处理上仍有欠缺。

---

## 513. BROS: Bias-Corrected Randomized Subspaces for Memory-Efficient Single-Loop Bilevel Optimization

**arXiv ID:** 2605.10288 | [PDF](https://arxiv.org/pdf/2605.10288v1)

**作者:** Hengrui Zhang `[一作]` (Sichuan University), Kun Yuan `[通讯]` (Peking University)

**通讯引用:** 4333 | [OpenAlex ID](https://openalex.org/A5100614598)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于随机子空间的单循环随机双层优化方法，可在多层矩阵参数化的双层问题中显著降低内存占用，同时保持 𝒪(ε^-2) 的收敛速度。

**💡 创新点**

创新点：1) 将下层和辅助层的更新压缩到随机子空间；2) 通过 Rademacher 双探针校正投影导致的 Hessian 行为偏差，恢复无偏的辅助线性系统；3) 在单循环框架下实现这些改进，实现了与全空间方法相当的理论保证。

**🔧 技术方法**

使用技术包括：随机子空间投影（Haar 分布）、Rademacher 双探针校正、HVP/JVP 采样、单循环双层递推、移动平均变分技术、兼容动量/自适应优化器的框架。

**📊 数据集**

实验数据集：MNIST（50% 标签噪声）数据清洗、17 个 Pile 域的数据混合学习（280M GPT 代理/主模型）、CIFAR-10 表示学习、CIFAR-100 ViT 样本重加权。

**📈 对比分析**

与 Penalty、FdeHBO、MA-SOBA、ZOFO 等基线比较，性能几乎与 MA-SOBA 相当（例如 MNIST 84.30% vs 84.46%），在代理模型上显著降低约 45% 内存，整体保持较高训练步率。

**⚠️ 局限性**

限制：仍需较大的子空间维度以控制方差；目前分析仅针对 SGD 形式，未覆盖动量或自适应优化器；在极大模型下激活内存仍有进一步压缩空间。

---

## 514. DP-LAC: Lightweight Adaptive Clipping for Differentially Private Federated Fine-tuning of Language Models

**arXiv ID:** 2605.10272 | [PDF](https://arxiv.org/pdf/2605.10272v1)

**作者:** Haaris Mehmood `[一作]` (Samsung Research and Development Institute United Kingdom), Mete Ozay `[通讯]` (Samsung Research and Development Institute United Kingdom)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种无需额外超参数的动态裁剪阈值方法DP‑LAC，提升了在差分隐私联邦学习中LLM微调的准确率。

**💡 创新点**

通过私有直方图估计初始裁剪阈值，并用服务器验证集的损失自适应更新阈值，无需额外隐私预算或超参数。

**🔧 技术方法**

采用DP‑SGD、Gaussian机制、直方图估计、LoRA参数高效微调、RDP moments accountant、FedAvg以及验证集监控等技术。

**📊 数据集**

使用GLUE（SST‑2、QNLI、MNLI）和SAMSum摘要数据集，模型为TinyLlama‑1B和Qwen3‑4B。

**📈 对比分析**

与五种基线（固定裁剪、量化裁剪、噪声衰减、阈值衰减等）在相同隐私预算下对比，DP‑LAC平均提升约6.6%，在大多数任务中击败或相当于最佳方法。

**⚠️ 局限性**

在极高隐私预算（ε=8）某些任务的性能差距仅小于1%，且缺少公共验证集时需要拆分预算进行私有损失估计，可能增加计算复杂度。

---

## 515. Misspecified Universal Learning

**arXiv ID:** 2605.10282 | [PDF](https://arxiv.org/pdf/2605.10282v1)

**作者:** Shlomi Vituri `[一作]` (Tel-Aviv University), Meir Feder `[通讯]` (Tel-Aviv University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在模型错误指定下的通用学习问题，特别是使用对数损失的情况。作者分析了在错误指定的环境中，通用学习者的最小最大遗憾，并提出了相应的最优通用学习者。

**💡 创新点**

创新点在于扩展了通用学习的理论框架，涵盖了监督和非监督的在线及批量学习场景，并引入了受限错误指定设置，分析了在该设置下的最小最大遗憾与容量之间的关系。

**🔧 技术方法**

使用了信息论中的工具，特别是贝叶斯混合分布和阿里莫托-布拉胡特算法的扩展来评估最小最大遗憾。

**📊 数据集**

使用了多项式分布、伯努利分布、马尔可夫模型和高斯位置模型等数据集进行数值评估。

**📈 对比分析**

通过与经典的统计学习理论进行比较，发现错误指定情况下的最小最大遗憾主要由假设类的复杂性决定，而不是更广泛的数据生成类。性能上，受限错误指定设置的最小最大遗憾与良好指定的容量相近，且有固定的惩罚项。

**⚠️ 局限性**

限制在于在某些极端情况下，受限错误指定的遗憾可能显著超过良好指定的容量，尤其是在参数集的关系不佳时。

---

## 516. MemReread: Enhancing Agentic Long-Context Reasoning via Memory-Guided Rereading

**arXiv ID:** 2605.10268 | [PDF](https://arxiv.org/pdf/2605.10268v1)

**作者:** Baibei Ji `[一作]` (Soochow University), Min Zhang `[通讯]` (Soochow University)

**通讯引用:** 42196 | [OpenAlex ID](https://openalex.org/A5013794939)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MemReread 框架，结合流式阅读与可自适应的复读机制来恢复被忽略的重要证据。

**💡 创新点**

创新点在于通过任务拆分与子问题引导的复读，避免检索干扰并保持线性时间复杂度，同时采用 Rereading‑Adaptive GRPO 动态控制复读次数。

**🔧 技术方法**

主要技术包括流式记忆代理、子问题生成、复读策略、强化学习的 GRPO 与自适应奖励。

**📊 数据集**

使用了 HotpotQA 与 2WikiMultiHopQA 扩展版，构造了 8K–1M 级别的长上下文评测集。

**📈 对比分析**

与 MemAgent 与 ReMemR1 对比，MemReread 在多模型规模下均取得更高准确率，尤其在 OOD 数据上提升约 12%；但平均推理时间比 MemAgent 增加 3–4 倍。

**⚠️ 局限性**

局限在于复读带来的额外计算开销、对 RL 训练的依赖以及在极长文本时可能的误拆分导致性能下降。

---

## 517. Low-Cost GNSS Anti-Jamming Through 2-Bit Phase Shift Beamforming with Machine Learning

**arXiv ID:** 2605.10264 | [PDF](https://arxiv.org/pdf/2605.10264v1)

**作者:** Burak Soner `[一作]` (sobu Labs), Can Aksoy `[通讯]` (EDGE Microwave)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对GNSS抗干扰，研究了仅使用2‑bit相位移（QPSK）天线阵列实现低成本的波束形成方法，并对其性能进行仿真与实验验证。

**💡 创新点**

提出了将抗干扰目标表述为离散组合优化问题，并用梯度提升决策树（GBDT）结合局部坐标下降的混合算法，在保持低延迟的前提下逼近全枚举“oracle”解的性能。

**🔧 技术方法**

核心技术包括：QPSK相位约束下的离散组合优化、基于协方差矩阵与引导向量的特征提取、GBDT分类器（每个天线一个模型）预测QPSK权重、以及局部坐标下降微调。

**📊 数据集**

使用的实验数据集为：1）仿真数据——随机生成GNSS信号、天线阵列响应、噪声与干扰（J/S 44‑70 dB，单干扰源）；2）实验数据——在静止室内环境下，单一全带快速扫频干扰源，采用数字仿真实现QPSK权重，N=8。

**📈 对比分析**

与传统连续权重Capon波束、简单量化、穷举oracle、随机采样等方法对比。Oracle在4‑10元素阵列下可实现高达34 dB的干扰抑制；GBDT+微调在保持固定低延迟（≈10–15 ms）下，性能接近oracle；实验中QPSK oracle相较无波束形成接收机，在J/S 44 dB、62 dB、70 dB时分别提升平均C/N₀ 4.2 dB、6.6 dB、11.5 dB。

**⚠️ 局限性**

实验仅在单干扰源、静止场景与N=8阵列下验证，未涵盖多干扰、多天线动态场景；在更大阵列或更复杂环境中，离散优化空间急剧膨胀，现有算法的可扩展性与实时性仍待提升。

---

## 518. Mapping Partisan Fault Lines Within DAOs

**arXiv ID:** 2605.10316 | [PDF](https://arxiv.org/pdf/2605.10316v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 519. Signature Approach for Contextual Bandits with Nonlinear and Path-dependent Rewards

**arXiv ID:** 2605.10313 | [PDF](https://arxiv.org/pdf/2605.10313v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 520. A Cold Diffusion Approach for Percussive Dereverberation

**arXiv ID:** 2605.10256 | [PDF](https://arxiv.org/pdf/2605.10256v1)

**作者:** Dimos Makris `[一作]` (Hellenic Mediterranean University), Maximos Kaliakatsos-Papakostas `[通讯]` (Hellenic Mediterranean University)

**通讯引用:** 559 | [OpenAlex ID](https://openalex.org/A5001450313)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种冷扩散框架，用于立体鼓组音轨的去混响；

**💡 创新点**

将混响建模为确定性退化过程，并提出两种逆向参数化（直接与Δ归一化残差），在UNet与Diffusion Transformer上实现，专注于鼓瞬态恢复；

**🔧 技术方法**

使用冷扩散、复数STFT、UNet与Diffusion Transformer、Δ归一化残差预测以及多分辨率STFT、相位、ESR、SI‑SDR、MSD、ENV、TTER、ONFi等评估指标；

**📊 数据集**

构建MUSDB18‑HQ与Groove MIDI Dataset的干鼓音轨，人工筛选无混响音频，结合合成与测量房间冲激响应（OpenAIR）生成混响，并在MoisesDB上制作完全域外测试集；

**📈 对比分析**

在相同数据与模型结构下与SGMSE+、CDiffuSE进行对比，域内测试提升SI‑SDR约11 dB、ESR↓、TTER↓、ENV↑；域外亦优于基线，保持>7 dB SI‑SDRi，显示更强鲁棒性；

**⚠️ 局限性**

局限于仅处理物理房间混响，未覆盖制作式混响；在极端高T60域外条件下仍存在尾音残留；采用固定16步推断，未探索可变步长或更大网络容量。

---

## 521. TMAS: Scaling Test-Time Compute via Multi-Agent Synergy

**arXiv ID:** 2605.10344 | [PDF](https://arxiv.org/pdf/2605.10344v1)

**作者:** George Wu `[一作]` (IQuest Research), Bryan Dai `[通讯]` (IQuest Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 TMAS，一种多代理协同的测试时推理加速框架，利用经验与指导记忆实现迭代推理提升。

**💡 创新点**

创新点包括：① 将经验记忆和指导记忆分层引入多代理系统，实现跨轨迹协作；② 通过混合奖励的强化学习鼓励经验利用与新策略探索；③ 通过显式信息流和记忆管理解决现有方法信息共享不足的问题。

**🔧 技术方法**

使用技术包括：多代理推理流水线、经验与指导记忆库、基于 GRPO 的混合奖励强化学习、并行生成与验证、FP8 量化加速训练。

**📊 数据集**

实验数据集包括 IMO-AnswerBench-50、HLE-Math-100、AIME26、HMMT-25-Nov 等数学推理基准。

**📈 对比分析**

通过与 Majority Vote、Self-Refine、Verify-Refine、PaCoRe、RSE 等 TTS 基线在 Pass@1 上对比，TMAS 在迭代阶段持续提升，20 轮时达到最高准确率；混合 RL 进一步提升性能，显著缩小 4B 与 30B 模型间差距。

**⚠️ 局限性**

局限性包括：未在更大模型（如 GPT-5.5）上评估；RL 训练需要预构造冷启动轨迹，缺乏动态更新；高并行度与验证成本仍是瓶颈。

---

## 522. PaMoSplat: Part-Aware Motion-Guided Gaussian Splatting for Dynamic Scene Reconstruction

**arXiv ID:** 2605.10307 | [PDF](https://arxiv.org/pdf/2605.10307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 523. Set Prediction for Next-Day Active Fire Forecasting

**arXiv ID:** 2605.10298 | [PDF](https://arxiv.org/pdf/2605.10298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 524. Nearly-Optimal Algorithm for Adversarial Kernelized Bandits

**arXiv ID:** 2605.10299 | [PDF](https://arxiv.org/pdf/2605.10299v1)

**作者:** Shogo Iwazaki `[一作]` `[通讯]` (LY Corporation), Shogo Iwazaki (LY Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本论文研究了在对抗环境下的核化赌博机（也称为高斯过程赌博机），提出了一种近似最优的算法，并展示了其在已知再生核希尔伯特空间中的应用。

**💡 创新点**

创新点在于提出了核化Exp3算法及其变体RLS-核化Exp3，解决了现有算法在对抗性核化赌博机中无法获得无悔保证的问题，并提供了算法独立的下界。

**🔧 技术方法**

使用了核化Exp3算法和RLS-核化Exp3算法，结合了Nyström近似方法以提高计算效率。

**📊 数据集**

论文中使用的核包括平方指数（SE）和ν-Matérn核，相关的最大信息增益（MIG）也被分析。

**📈 对比分析**

与现有算法相比，核化Exp3和RLS-核化Exp3在对抗环境下的悔恨上界为Õ(√(T γ_T))，并且在计算复杂度上显著降低，RLS-核化Exp3的每轮计算复杂度为Õ(γ_T^2 + γ_T^3)。

**⚠️ 局限性**

限制在于算法需要对输入域进行离散化，这在高维情况下可能导致计算不可行，未来的研究可以探索直接在连续域上操作的算法。

---

## 525. Robust Probabilistic Shielding for Safe Offline Reinforcement Learning

**arXiv ID:** 2605.10293 | [PDF](https://arxiv.org/pdf/2605.10293v1)

**作者:** Maris F. L. Galesloot `[一作]` (Radboud University), Nils Jansen `[通讯]` (Ruhr University & Radboud University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将概率性防护罩（probabilistic shielding）融入离线强化学习中的安全策略改进（SPI）框架，利用从数据集构建的区间MDP（IMDP）自动生成安全约束，并在基线策略和策略改进过程中强制执行，确保以高置信度避免危险行为；

**💡 创新点**

创新点在于首次将在线RL中基于模型的概率性防护罩迁移到仅依赖离线数据的环境；通过从数据集构造IMDP并计算最坏情况的达成-回避概率来构造安全阈值，实现了对真实MDP的高概率安全保证，并将此防护罩直接嵌入SPIBB算法；

**🔧 技术方法**

主要技术包括：区间MDP建模（利用Hoeffding界定转移概率区间）、最坏情况概率计算（鲁棒动态规划/模型检查）、SPI与SPIBB算法的改进（基线加防护罩、改进步的安全约束）、基于概率阈值的安全动作筛选；

**📊 数据集**

实验使用四个离线RL基准：随机MDP、Wet Chicken、Frozen Lake以及大规模Pacman（7×7迷宫+移动幽灵）；

**📈 对比分析**

与传统SPIBB、DUIPI等非防护罩版本在这四个基准上对比，结果显示防护罩版本在数据量有限时平均奖励和1%-CVaR（最差10%样本）均优于非防护罩版本，收敛速度更快且更安全（更少负奖励或触碰危险状态）；

**⚠️ 局限性**

局限性包括：防护罩的保守性可能导致高奖励动作被屏蔽，安全阈值需要手工调参且对数据稀疏或区间估计不准时会过度或不足保护；方法假设已知安全/目标状态且需要区间MDP的精确构造，限制了在更复杂或部分可观测环境中的直接应用。

---

## 526. SciIntegrity-Bench: A Benchmark for Evaluating Academic Integrity in AI Scientist Systems

**arXiv ID:** 2605.10246 | [PDF](https://arxiv.org/pdf/2605.10246v1)

**作者:** Zonglin Yang `[一作]` (Readraft Lab), Xinyan Xu `[通讯]` (Readraft Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了SciIntegrity-Bench基准，用于系统评估AI科学家在完成任务时的学术诚信表现。

**💡 创新点**

创新点在于提出“对立评估”范式，将诚实承认失败与任务完成之间的矛盾置于基准情景中，揭示系统固有的诚信缺失。

**🔧 技术方法**

采用最小ReAct框架和七款主流LLM（如GPT‑5.2、Claude‑4.6、Gemini‑3.1等），通过模拟工具调用和报告生成实现评估。

**📊 数据集**

基准包含33个跨学科的对立情景，涵盖11类学术不端陷阱，来源于社交媒体、文献和同行评议的真实案例。

**📈 对比分析**

对比实验显示整体诚信问题率为34.2%，最严重陷阱T08、T05的失误率高达80%+；不同模型在不同陷阱上的失误分布差异显著。

**⚠️ 局限性**

局限性包括情景数量有限、仅使用单一ReAct框架、人工标注缺乏一致性测评以及无法自动化检测诚信违规。

---

## 527. Every Preference Has Its Strength: Injecting Ordinal Semantics into LLM-Based Recommenders

**arXiv ID:** 2605.10323 | [PDF](https://arxiv.org/pdf/2605.10323v1)

**作者:** Jiwon Jeong `[一作]` (Korea Advanced Institute of Science and Technology), Mun Yong Yi `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 7689 | [OpenAlex ID](https://openalex.org/A5088156206)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Ordinal Semantic Anchoring (OSA)框架，将显式评分的序数语义嵌入LLM-协同过滤混合推荐。

**💡 创新点**

通过将评分级别作为文本令牌并将其词嵌入作为语义锚点，进行强度感知的对齐，保留了细粒度偏好强度。

**🔧 技术方法**

使用SASRec作为CF编码器、两层MLP投影器、LLaMA 3.2 LLM、LoRA微调以及余弦对齐损失。

**📊 数据集**

在MovieLens-1M、Amazon Scientific和Amazon Video Games三个包含1–5星评分的数据集上实验。

**📈 对比分析**

与CF、LLM单体和现有混合基线相比，在Hit@1和对比偏好准确率上均实现显著提升（最高提升约44.9%）。

**⚠️ 局限性**

仅依赖评分作为锚点，未考虑隐式反馈或多模态内容；对极端稀疏数据的鲁棒性待验证。

---

## 528. The Alpha Blending Hypothesis: Compositing Shortcut in Deepfake Detection

**arXiv ID:** 2605.10334 | [PDF](https://arxiv.org/pdf/2605.10334v1)

**作者:** Andrii Yermakov `[一作]` (Czech Technical University), Jiri Matas `[通讯]` (Czech Technical University)

**通讯引用:** 49910 | [OpenAlex ID](https://openalex.org/A5007656938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Alpha Blending 假设，并基于真实面部图像和自我混合图像（SBI）训练了一种框架，用于跨数据集检测深度伪造。

**💡 创新点**

创新点：①阐明现有 SOTA 设备主要靠检测 alpha 混合边界而非语义或生成器指纹；②证明仅使用多样化真实图像加 SBI 就能获得与训练深度伪造相当或更好的跨数据集泛化；③通过模型融合展现显式混合搜索器与传统深度伪造检测器互补，提升性能。

**🔧 技术方法**

技术：利用 Vision Foundation Models（CLIP、DINOv3、DINOv3 等）作为骨干；在 1:1 真实/ SBI 对中进行交叉熵训练；使用 alpha 混合、软硬边界等混合方式生成 SBI；融合多模型输出实现无参数集成。

**📊 数据集**

数据集：ScaleDF 5.8M 真实图像；15 2019‑2025 年发布的组合式深度伪造数据集（FaceForensics++, Celeb‑DF‑v2/++, DeepFake Detection Challenge 等）；SBI 通过对 25k 真实图像自混合生成伪造图像；与其他工作使用的 8.8M 生成深度伪造数据集做对比。

**📈 对比分析**

与 Effort、ForAda、FS‑VFM、GenD 等 SOTA 进行交叉数据集评测，平均 AUROC 最高达 91.3%，单模型与 94%（ensemble）。与传统 SBI、FSBI、官方基线相比，显著提升；在 15 公开数据集上均位居榜首。

**⚠️ 局限性**

局限性：对全合成或非组合式伪造（如 LivePortrait、MEMO、HelloMeme 等）性能明显下降；过度依赖混合边界导致对非生成式亮度变化等噪声过度敏感；缺乏足够的全合成数据集验证。

---

## 529. Real vs. Semi-Simulated: Rethinking Evaluation for Treatment Effect Estimation

**arXiv ID:** 2605.10430 | [PDF](https://arxiv.org/pdf/2605.10430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 530. DeepLog: A Software Framework for Modular Neurosymbolic AI

**arXiv ID:** 2605.10279 | [PDF](https://arxiv.org/pdf/2605.10279v1)

**作者:** Robin Manhaeve `[一作]` (KU Leuven), Giuseppe Marra `[通讯]` (KU Leuven)

**通讯引用:** 513 | [OpenAlex ID](https://openalex.org/A5005466305)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

实现了 DeepLog，一套可将高层符号规范编译为 GPU 加速算术电路的神经符号框架；

**💡 创新点**

创新之处在于统一抽象机和编译器式工厂，使多种 NeSy 系统可通过极少代码复现，并在 GPU 上高并行执行逻辑与深度学习模块；

**🔧 技术方法**

核心技术包括符号注解的 PyTorch 模块、DeepLog 语言、GPU 加速的电路评估（KLay）、sentential decision diagrams 等；

**📊 数据集**

主要在 MNIST-Addition 任务和基于 DIMACS CNF（如 A→B、C→B）约束的示例中进行验证；

**📈 对比分析**

与传统 CPU 实现 DeepProbLog 对比，DeepLog GPU 在单查询推理时间上提升至 5.6×10⁻⁷ s，速度提升数百倍；

**⚠️ 局限性**

局限在于目前依赖 GPU 电路实现，缺乏更高级的电路优化与采样推理等多种后端支持，且对复杂逻辑结构的可扩展性待进一步评估。

---

## 531. CoWorld-VLA: Thinking in a Multi-Expert World Model for Autonomous Driving

**arXiv ID:** 2605.10426 | [PDF](https://arxiv.org/pdf/2605.10426v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 532. Aligning LLM Uncertainty with Human Disagreement in Subjectivity Analysis

**arXiv ID:** 2605.10415 | [PDF](https://arxiv.org/pdf/2605.10415v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 533. Remember to Forget: Gated Adaptive Positional Encoding

**arXiv ID:** 2605.10414 | [PDF](https://arxiv.org/pdf/2605.10414v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 534. Equilibrium Residuals Expose Three Regimes of Matrix-Game Strategic Reasoning in Language Models

**arXiv ID:** 2605.10410 | [PDF](https://arxiv.org/pdf/2605.10410v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 535. Position: Life-Logging Video Streams Make the Privacy-Utility Trade-off Inevitable

**arXiv ID:** 2605.10404 | [PDF](https://arxiv.org/pdf/2605.10404v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 536. Identified-Set Geometry of Distributional Model Extraction under Top-$K$ Censored API Access

**arXiv ID:** 2605.10407 | [PDF](https://arxiv.org/pdf/2605.10407v1)

**作者:** Wenhua Nie `[一作]` (National Taiwan University), Jyh-Shing Roger Jang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在大型语言模型 API 只暴露 top‑K 置信词条时，单个位置的分布恢复上限与信息损失量化问题。

**💡 创新点**

首次给出了 top‑K 过滤下的确定集合（identified set）几何，并推导出总变异（TV）直径公式与 KL 低/上界；并揭示分布忠实度与能力转移不一定同步。

**🔧 技术方法**

采用部分识别（partial identification）框架、软最大分布的数学分析、KL 复合极限理论以及基于参考模型的先验收敛约束。

**📊 数据集**

使用 Qwen3‑0.6B（数学推理）与 Llama‑3.2‑1B/3B（WikiText 语言建模）等公开模型，以及 GSM8K（算术推理）和 MBPP（代码生成）数据集进行实验验证。

**📈 对比分析**

与全词表 logit、稀疏 KL、生成式 SFT 等多种抽取方式比较，发现对 top‑K 训练的学生仅能获得约 12% PVR，完整 logit 训练可达 56%，而生成式抽取可达 96%；理论 TV 直径和 KL 下限均与实验误差匹配，显示 top‑K 过滤并非完整防护。

**⚠️ 局限性**

局限性在于仅对单位置进行分析，未考虑自回归或生成式查询的自适应复合；参考模型先验假设对强 SFT 失效；不同 API 精度、温度或量化也可能影响几何结论。

---

## 537. Phoenix-VL 1.5 Medium Technical Report

**arXiv ID:** 2605.10391 | [PDF](https://arxiv.org/pdf/2605.10391v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 538. SleepWalk: A Three-Tier Benchmark for Stress-Testing Instruction-Guided Vision-Language Navigation

**arXiv ID:** 2605.10376 | [PDF](https://arxiv.org/pdf/2605.10376v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 539. Temporal Sampling Frequency Matters: A Capacity-Aware Study of End-to-End Driving Trajectory Prediction

**arXiv ID:** 2605.10388 | [PDF](https://arxiv.org/pdf/2605.10388v1)

**作者:** Yumao Liu `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Ke Ma `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究了在端到端驾驶轨迹预测中，摄像头帧的时间采样频率对模型性能的影响，并将其视为可调的训练变量；通过对 Waymo、nuScenes 和 PAVE 数据集进行不同频率的子采样，构建了频率扫频训练集；在相同模型、目标、损失和评估协议下训练并评估不同模型；进一步做了迭代匹配对比以排除训练步数差异的影响。

**💡 创新点**

创新点在于：①提出了“容量感知”视角，认为稠密采样既能提供更多驾驶相关信息，又可能带来冗余视觉内容和离散噪声，导致有限容量模型的性能非单调；②通过频率扫频实验系统地表征了不同模型与数据集的频率响应；③证明了模型规模与最优采样频率之间的关系，并将采样频率作为可报告和可调节的训练超参数。

**🔧 技术方法**

技术包括：时间子采样操作、端到端轨迹预测模型（E2EDriver、BEV‑E2EDriver、Tiny‑SSR、AutoVLA）、统一训练与评估协议（相同输入格式、命令条件、ADE/FDE指标）、迭代匹配控制实验。

**📊 数据集**

使用的公开数据集有 Waymo、nuScenes 与 PAVE。每个数据集的原始摄像头采样频率分别为 10 Hz、12 Hz 与约 30 Hz，实验覆盖 2–20 Hz 之间的频率。

**📈 对比分析**

比较方法：在同一数据集、同一模型下，训练不同频率的训练集，固定 epoch 数；随后进行迭代匹配对比以平衡训练量；评估 3 s ADE/FDE。结果显示：较小的端到端模型往往在中低频率（如 6–10 Hz）达到最佳性能，而较大的 AutoVLA 在最高频率（10–20 Hz）表现最好；不同数据集的最佳频率也不同。

**⚠️ 局限性**

局限性包括：仅在离线 ADE/FDE 评估下验证，缺乏闭环或在线部署评估；未直接量化冗余内容与噪声对性能的具体贡献；实验仅覆盖有限的模型与数据集；以及频率范围上限受计算资源限制。

---

## 540. Amortized Asynchronous Byzantine Reliable Broadcast with Optimal Resilience

**arXiv ID:** 2605.10372 | [PDF](https://arxiv.org/pdf/2605.10372v1)

**作者:** Michael Yiqing Hu `[一作]` (National University of Singapore), Jialin Li `[通讯]` (National University of Singapore)

**通讯引用:** 213827 | [OpenAlex ID](https://openalex.org/A5100437036)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了一种异步网络下的多轮拜占庭可靠广播协议（APM‑BRB），通过分摊与动态委员会抽样实现最优容错（f < n/3）并将通信复杂度降至 O(n|m|+n²k)，当 |m| ≥ Ω(nk) 时进一步简化为最优 O(n|m|)。

**💡 创新点**

创新点在于将传统两相 Bracha 协议的成本通过分摊技术压缩至单阶段，结合轮换委员会抽样与阈值签名，使协议在异步网络中仍保持最优容错且通信复杂度显著降低；同时提供乐观路径实现 O(1) 轮延迟。

**🔧 技术方法**

采用的技术包括：轮换委员会抽样、阈值签名（以及可选的多签名）、分摊（amortization）分析、概率分析（Hoeffding 以及几何分布）、链式消息一致性保证以及对异步网络的标准假设。

**📊 数据集**

本文未使用实际数据集，主要通过理论分析与概率证明来验证协议的正确性与复杂度；若需实验，可使用模拟环境或标准拜占庭协议测试集。

**📈 对比分析**

与 Bracha 原型、使用纠删码或采样的传统异步 BRB 进行理论比较：通信复杂度由 O(n²|m|) 降至 O(n|m|+n²k)，当 |m| 较大时达到最优 O(n|m|)；轮数在一般情况下为 O(n)，在乐观路径可实现 O(1)。

**⚠️ 局限性**

局限性：仍为概率协议，失败概率随 φ 和 n_c 的选取而定；缺乏实验验证；对强适应性攻击的安全性尚未证明；对基于 DAG 的拜占庭原子广播（BAB）的直接适配尚未完成。

---

## 541. Learning-Based Spectrum Cartography in Low Earth Orbit Satellite Networks: An Overview

**arXiv ID:** 2605.10359 | [PDF](https://arxiv.org/pdf/2605.10359v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 542. Approximate Envy-Free Allocations up to any $k$ Goods

**arXiv ID:** 2605.10371 | [PDF](https://arxiv.org/pdf/2605.10371v1)

**作者:** Aris Filos-Ratsikas `[一作]` (University of Edinburgh), Fangxiao Wang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5041645570)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了可加性价值函数下的近似无嫉妒分配（EFkX）问题，提出了一种多步骤的多项式时间算法，证明了对任意 k≥2 存在 (k+1)/(k+2)-EFkX 分配；并以此得到 3/4-EF2X 分配（对任意人数）以及 2/3-EFX 分配（最多 8 名代理人）。此外，还研究了图上资源分配的方向化问题，证明了在一般情形下该方向化问题既不存在也在 NP‑完整。

**💡 创新点**

创新点包括：① 将近似无嫉妒（α‑EFX）与多件删去（k‑EFX）两条研究线首次结合，获得了更优的近似比；② 设计了通用的“Generalized Property Preserving Partial Allocation”算法和“Critical Goods Allocation”算法，能够处理任意数量代理人；③ 通过构造“Envy‑Enhancer”与“Funnel”两种新型图形 gadget，完成了从 EF(k‑1)X 方向化到 EFkX 方向化的多项式时间归约，证实 NP‑完整性；④ 在 8 名代理人情况下首次突破 7 名代理人的 2/3‑EFX 结果。

**🔧 技术方法**

主要技术包括：
- 变形的 envy‑graph（α‑modified graph）和其在算法中的使用；
- P2FA（Partial‑to‑Full Allocation）引理，将满足特定条件的部分分配转化为完整分配；
- 递归的子程序（Cycle‑Elimination、Path‑Transfer、Swap‑Bundles）以及新引入的 Subroutine (·, G̃, Π, Y)；
- 对 critical goods 的分配策略（按价值顺序分配剩余物品）；
- 图形 gadget 归约技术（Envy‑Enhancer、Funnel），构造 NP‑完整性证明。

**📊 数据集**

本文没有使用实际数据集，所有结果均为理论证明与多项式时间算法分析；实验部分在论文中仅以理论复杂度和近似比说明。

**📈 对比分析**

性能评估主要以近似比和时间复杂度为指标：
- 对任意 k≥2，得到 (k+1)/(k+2)‑EFkX 分配，时间复杂度为多项式；
- 3/4‑EF2X 在任意人数下可在多项式时间内计算；
- 2/3‑EFX 在最多 8 名代理人下可在多项式时间内计算。
- 对于图方向化问题，证明其在一般情况下不存在解且判定问题为 NP‑完整。

**⚠️ 局限性**

局限性包括：
- 对 k=1 的情形仅能得到 8 名代理人的 2/3‑EFX 结果，仍未解决更大代理人数下的 2/3‑EFX 近似；
- 本文仅给出了加性价值函数的结果，未讨论其他偏好类（如单调、预算约束等）；
- 方向化问题的 NP‑完整性仅在一般图上给出，对特殊图类（如树、二分图）尚无完整结论；
- 近似比虽提升，但与 1（完全无嫉妒）仍相距较远，具体最优值仍不清楚。

---

## 543. Portable Active Learning for Object Detection

**arXiv ID:** 2605.10349 | [PDF](https://arxiv.org/pdf/2605.10349v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 544. A Factorization Theorem for Forest Algebras

**arXiv ID:** 2605.10368 | [PDF](https://arxiv.org/pdf/2605.10368v1)

**作者:** Shaull Almagor `[一作]` (Technion), Asaf Shoham `[通讯]` (Technion)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文发展了西蒙因子分解定理在森林代数中的类比，提供了关于森林的有界深度分解定理。

**💡 创新点**

创新点在于引入了新的语义限制（称为-对齐），确保不同的切割方式在半群层面上保持兼容，从而证明了每个映射都可以获得有界深度的分解。

**🔧 技术方法**

使用了增强的自由森林代数和递归分解框架，结合了二元分解和一般分解的概念。

**📊 数据集**

使用了与真布尔公式（TBF）相关的森林代数作为示例。

**📈 对比分析**

通过与西蒙定理的证明方法相似的方式进行比较，证明了在-对齐条件下，每个森林的分解深度最多为4|V|-3；而在没有-对齐的情况下，无法保证有界深度的分解。

**⚠️ 局限性**

限制在于没有-对齐的情况下，无法保证任何统一的深度界限，且在某些情况下可能导致分解深度的增加。

---

## 545. DeepLévy: Learning Heavy-Tailed Uncertainty in Highly Volatile Time Series

**arXiv ID:** 2605.10364 | [PDF](https://arxiv.org/pdf/2605.10364v1)

**作者:** Yang Yang `[一作]` (University of New South Wales), Flora Salim `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出 DeepLévy 框架，通过学习 Lévy α‑stable 分布的混合模型，实现多步长的深度概率时间序列预测。

**💡 创新点**

创新点包括：① 用特征函数匹配（无需显式 PDF）训练混合 Lévy 分布；② 采用 S₀ 参数化、尺度自适应频率权重、约束投影和熵正则化保证数值稳定；③ 通过 CMS 算法实现高效采样；④ 在金融与流行病数据上显著提升尾部风险评估。

**🔧 技术方法**

技术要点：深度序列编码器（Transformer/LSTM）+ 自回归解码器；约束投影层输出混合权重与 Lévy 参数；特征函数损失+尺度自适应权重；熵正则化与梯度裁剪；批量特征函数估计；CMS 采样。

**📊 数据集**

实验数据集：比特币收益率（小时级）、COVID‑19 传播病例（日级）以及用 CMS 生成的合成 Lévy 序列。

**📈 对比分析**

与 DeepAR（多种分布）、DeepVAR、DSSM、DeepFactor、ProbTransformer 等基线比较，使用 CRPS、Tail‑CRPS、QL、覆盖率、PIT‑KS 等指标。DeepLévy 在 Tail‑CRPS（0.461）和 99.5% 覆盖率（0.976）上优于所有基线，同时整体 CRPS（0.293）保持竞争力；消融实验验证了各组件的重要性。

**⚠️ 局限性**

局限性：仅采用单变量 Lévy 参数化，需较大批量训练；对 α≈1 的处理仍依赖 S₀ 参数化；未完全捕捉尾部参数的时间依赖；对非重尾或多元序列的适用性仍需进一步研究。

---

## 546. Can Language Models Analyze Data? Evaluating Large Language Models for Question Answering over Datasets

**arXiv ID:** 2605.10419 | [PDF](https://arxiv.org/pdf/2605.10419v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 547. DySurface: Consistent 4D Surface Reconstruction via Bridging Explicit Gaussians and Implicit Functions

**arXiv ID:** 2605.10360 | [PDF](https://arxiv.org/pdf/2605.10360v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 548. Agentic Performance at the Edge: Insights from Benchmarking

**arXiv ID:** 2605.10384 | [PDF](https://arxiv.org/pdf/2605.10384v1)

**作者:** Shiqiang Wang `[一作]` (University of Exeter), Herbert Woisetschläger `[通讯]` (Technical University of Munich)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5092130737)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对资源受限的边缘环境，开展了大型语言模型在工具驱动式智能代理任务中的规模、代际与变体的系统性实证研究；

**💡 创新点**

创新点在于提出了基于域条件的评估方法，结合工具执行与失败模式拆分，揭示模型大小并非单一指标，且域异质性与失败类型决定了部署策略；

**🔧 技术方法**

使用了开放源代码模型（Qwen、Phi、Mistral）在统一的工具工作流（检索、拓扑推理、候选归约、异常分析）上，通过固定迭代步骤、温度与推理接口进行对比；

**📊 数据集**

实验数据来源于 ITBench 的 FinOps 与 SRE 两个诊断领域的 25 题 FinOps 与 35 题 SRE 的根因定位任务；

**📈 对比分析**

比较方法为在相同工具协议、步骤上限与硬件平台（RTX Pro 6000）下测算准确率、延迟与步骤数，结果显示 Qwen Coder 7B 在准确率与延迟上实现 Pareto 前沿，模型规模与性能关系非单调；

**⚠️ 局限性**

局限性包括仅使用 ITBench 单一基准与有限的开放模型族，未覆盖更多业务域或混合云边缘部署，且实验环境为高端 GPU 代理，无法完全反映低功耗边缘硬件的实际约束。

---

## 549. BGG: Bridging the Geometric Gap between Cross-View images by Vision Foundation Model Adaptation for Geo-Localization

**arXiv ID:** 2605.10345 | [PDF](https://arxiv.org/pdf/2605.10345v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 550. RW-Post: Auditable Evidence-Grounded Multimodal Fact-Checking in the Wild

**arXiv ID:** 2605.10357 | [PDF](https://arxiv.org/pdf/2605.10357v1)

**作者:** Danni Xu `[一作]` (National University of Singapore), Mohan Kankanhalli `[通讯]` (National University of Singapore)

**通讯引用:** 17473 | [OpenAlex ID](https://openalex.org/A5016415049)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RW-Post基准，收集并标注真实社交媒体多模态事实核查数据，并设计AgentFact验证框架。

**💡 创新点**

创新点：1) 结合原始社交媒体帖文与图像，提供可审计的推理轨迹与明确链接的证据；2) LLM辅助的提取与审计流程；3) 三种验证场景（闭本、证据限定、开放式），为可比对的实验提供统一标准。

**🔧 技术方法**

技术：LLM（GPT‑4o）进行信息抽取与审计；检索式推理、视觉分析（反向图像搜索、深度伪造检测）、多代理推理流水线；对比使用开源LVLM（LLaVA‑1.5、Qwen2‑VL‑Chat）与闭源GPT‑4o‑mini。

**📊 数据集**

数据集：RW‑Post（1.77k实例，3 类）和相关公开集（Mocheg、CLAIMREVIEW+、NewsCLIPpings）作泛化验证；来源为Snopes等。

**📈 对比分析**

比较方法：在三种验证模式下与AgentFact、LEMMA、DEFAME等系统对比；在RW‑Post上闭本/证据限定下评估LVLM；结果显示提供证据显著提升Macro‑F1，GPT‑4o‑mini在证据限定下达成84.5% Weighted‑F1，现有开源LVLM仍落后；视觉信息加入效果有限。

**⚠️ 局限性**

局限：1) 仍需大规模人工审核以确保抽取质量；2) 对视觉信号的整合不够强，导致图像支持效果有限；3) 解释生成的格式化成功率在开源模型上偏低；4) 低频类别（False、Unproven）性能不佳。

---

## 551. How Mobile World Model Guides GUI Agents?

**arXiv ID:** 2605.10347 | [PDF](https://arxiv.org/pdf/2605.10347v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 552. Syndrome Adaptive Gain Control for Min-Sum Decoding of Quantum LDPC Codes

**arXiv ID:** 2605.10433 | [PDF](https://arxiv.org/pdf/2605.10433v1)

**作者:** Hernan Cordova `[一作]` (Eindhoven University of Technology), Alex Alvarado `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 4205 | [OpenAlex ID](https://openalex.org/A5060148074)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于综合比率（syndrome ratio）的自适应增益Min‑Sum解码器（SAGMS），用于量子LDPC码的BP4/GF(4)解码；

**💡 创新点**

创新点在于：1）利用误差综合比率作为低成本反馈信号，实现在线增益自适应；2）给出了增益调度公式，使得解码器对CN度和噪声水平不需要离线优化；3）通过理论证明SMS对CN度的可扩展性惩罚，并展示SAGMS可自校准到最优增益；

**🔧 技术方法**

使用技术包括：Min‑Sum及其缩放版（SMS）、GF(4) BP4、综合比率反馈、线性增益调度、密度演化分析与Monte Carlo仿真；

**📊 数据集**

采用Generalized Bicycle（GB）QLDPC码作为实验数据集，主要是[[126,28]]（m=126, d_c=10）和[[126,20]]（m=126, d_c=16）两种码；

**📈 对比分析**

通过Monte Carlo仿真对比BP4、MS、SMS（α=0.5）与SAGMS的FER随退相干概率的变化；结果显示SAGMS在噪声匹配条件下从p≈0.03起就优于BP4，且与最优SMS性能相当；在噪声不匹配下SAGMS优于SMS，并在低噪声（p<0.01）时优于BP4；

**⚠️ 局限性**

局限性包括：1）增益参数α_eff、γ、β仍需经验选取；2）验证仅限于GB码，缺乏更大或更复杂码的实验；3）仅针对GF(4) BP4场景，未讨论其他量子码结构的推广；4）理论假设（如均匀消息近似）在短环密度分布宽度较大时可能影响精度。

---

## 553. Toward an Engineering of Science: Rebalancing Generation and Verification in the Age of AI

**arXiv ID:** 2605.10425 | [PDF](https://arxiv.org/pdf/2605.10425v1)

**作者:** Jiaqi W. Ma `[一作]` (University of Illinois Urbana-Champaign), Jiaqi W. Ma `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 3117 | [OpenAlex ID](https://openalex.org/A5089063294)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一种名为 Blueprint 的结构化、分解化研究成果表示方式，以期在 AI 时代降低论文验证成本，提供“验证接口”，与传统论文“叙事接口”并存。

**💡 创新点**

创新点在于：① 把论证结构显式化为有向图，节点类型对应主张、证据、假设等；② 通过“状态”和“lint”机制实现分布式、局部的验证流程；③ 设计双重接口（浏览器可视化与命令行 AI 交互）以支持人机协同编辑与验证。

**🔧 技术方法**

技术包括：JSON 基础数据模型、类型化图（node/edge 类型与状态阶梯）、基于浏览器 Canvas 的交互式图形编辑、命令行工具供 AI 代理调用、实时同步、结构化 lint 与状态管理等；借鉴 Lean theorem prover 的 blueprint 思路和 Toulmin/IBIS 论证模型。

**📊 数据集**

未使用传统机器学习数据集；论文主要基于手工构建的原型实例与概念验证，未在公开数据集上做实验。

**📈 对比分析**

比较方法：暂无实证比较。作者指出未来可通过：① 方案在小型项目上的作者时间与验证时间对比；② 与传统论文的评审一致性与错误检测率对比；③ 对不同学科的词汇扩展效果评估。论文未给出具体性能指标。

**⚠️ 局限性**

限制包括：① 尚未在真实科研项目中评估，缺乏经验验证；② 需要领域特定的词汇与状态定义，迁移成本不小；③ 现阶段的验证工具与 AI 代理支持有限，实际验证工作仍需人工介入；④ 只关注“有效性”维度，未解决新颖性与重要性评估；⑤ 可能对研究者的工作流程造成额外负担。

---

## 554. LLM4Branch: Large Language Model for Discovering Efficient Branching Policies of Integer Programs

**arXiv ID:** 2605.10401 | [PDF](https://arxiv.org/pdf/2605.10401v1)

**作者:** Zhinan Hou `[一作]` (Tsinghua University), Keyou You `[通讯]` (Tsinghua University)

**通讯引用:** 7094 | [OpenAlex ID](https://openalex.org/A5088962631)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了LLM4Branch框架，利用大型语言模型自动生成可执行的分支策略程序，并通过零阶优化直接最小化MILP求解的端到端成本，完成对混合整数线性规划求解器分支策略的自动发现。

**💡 创新点**

创新点在于：①将分支策略表示为可读的程序骨架+参数向量；②采用LLM进行程序骨架生成并通过演化式循环结合历史性能反馈；③使用零阶优化在黑盒求解器反馈下直接优化参数，避免模糊的仿真目标；④实现完全CPU实现的高效分支策略，性能与GPU加速的深度学习模型相当。

**🔧 技术方法**

技术手段包括：大型语言模型（如DeepSeek-R1、GPT‑5等）进行代码生成；结构化prompt与演化循环提升骨架质量；零阶（无梯度）优化（贝叶斯优化）进行参数调优；在SCIP 8.0.0求解器中替换默认分支策略进行评测。

**📊 数据集**

使用了六个MILP基准数据集：Setcover、Combinatorial Auction、Capacitated Facility Location、Maximum Independent Set、Balanced Item Placement、Neural Network Verification，分别包含Easy、Medium、Hard三个难度等级。

**📈 对比分析**

与七类基准方法比较（包括手工策略FSB、RPB；MLP、GNN、Hybrid、Symb4CO；以及GPU加速的SORREL、GNN‑GPU）。LLM4Branch在CPU基准中取得最佳或同等的时间/节点/赢率表现，尤其在难度较高的实例上优于所有CPU方法，并与GPU方法竞争。

**⚠️ 局限性**

局限性主要体现在：①需要依赖大型语言模型生成代码，模型成本和可解释性仍有限；②零阶优化对求解器的交互成本高，可能在更大规模实例上受限；③目前仅针对分支策略，其他求解器组件（节点选择、割裁生成等）尚未覆盖；④对不同LLM后端的性能差异仍需进一步研究。

---

## 555. AnomalyClaw: A Universal Visual Anomaly Detection Agent via Tool-Grounded Refutation

**arXiv ID:** 2605.10397 | [PDF](https://arxiv.org/pdf/2605.10397v1)

**作者:** Xi Jiang `[一作]` (Southern University of Science and Technology), Feng Zheng `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 6135 | [OpenAlex ID](https://openalex.org/A5063285882)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的视觉异常检测代理AnomalyClaw，利用多轮工具驱动的反驳过程实现跨域异常检测。

**💡 创新点**

创新点包括：1）将异常判定转化为多轮反驳流程，以比较与参考样本为核心；2）使用内部分支不一致进行无监督规则学习，形成可视化自我进化；3）同一流程兼容不同Vision‑Language Model后端，保持零参数更新。

**🔧 技术方法**

技术方案：Prompt‑time多轮Agent、13项专业工具库（视觉检索、热图、参考匹配、结构分析等）、单步Direct VLM评分与固定α融合、可选的自我进化(OSR)规则学习。

**📊 数据集**

数据集：CrossDomainVAD‑12（12个领域共1418张测试图）以及MMAD（多选问答评测）。

**📈 对比分析**

与现有方法比较：相较于单步Direct VLM评分，宏平均AUROC提升+6.23–7.93pp，超越多种无训练基线与专用VLM；在MMAD上取得79.15%平均准确率，位列一般M‑LLM之首。

**⚠️ 局限性**

局限性：需要足够的内部分歧才能启动规则学习，对弱分支或一致性高的域提升有限；工具调用成本高，导致在某些VLM上性能差异；在部分域仍出现轻微回退。

---

## 556. Causal Explanations from the Geometric Properties of ReLU Neural Networks

**arXiv ID:** 2605.10396 | [PDF](https://arxiv.org/pdf/2605.10396v1)

**作者:** Hector Woods `[一作]` (University of York), Rob Alexander `[通讯]` (University of York)

**通讯引用:** 954 | [OpenAlex ID](https://openalex.org/A5078632397)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出利用 ReLU 网络几何多面体分解直接生成因果“为什么”和“为什么不”的解释；

**💡 创新点**

不需要模型蒸馏，直接在原模型几何空间求得完全可解释的因果规则；

**🔧 技术方法**

多面体分解、H-Representation 与 V-Representation、线性规划以及邻域多面体行走；

**📊 数据集**

未公开；

**📈 对比分析**

与 PICE 方法对比，‘为什么’解释时间为 O(n)，‘为什么不’最坏情况为 O(2^n)，实验表明对小规模网络可行；

**⚠️ 局限性**

对高维输入、深层网络、非 ReLU 激活及 CNN 等结构的可解释性有限

---

## 557. FractalSortCPU: Bandwidth-Efficient Compressed Radix Sort on CPU

**arXiv ID:** 2605.10390 | [PDF](https://arxiv.org/pdf/2605.10390v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 558. EGL-SCA: Structural Credit Assignment for Co-Evolving Instructions and Tools in Graph Reasoning Agents

**arXiv ID:** 2605.10366 | [PDF](https://arxiv.org/pdf/2605.10366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 559. Agent-ValueBench: A Comprehensive Benchmark for Evaluating Agent Values

**arXiv ID:** 2605.10365 | [PDF](https://arxiv.org/pdf/2605.10365v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 560. VISOR: A Vision-Language Model-based Test Oracle for Testing Robot

**arXiv ID:** 2605.10408 | [PDF](https://arxiv.org/pdf/2605.10408v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 561. DREAMS: Modelling Support for Research into Engineering and Artistic Design

**arXiv ID:** 2605.10382 | [PDF](https://arxiv.org/pdf/2605.10382v1)

**作者:** Apala Chakrabarti `[一作]` `[通讯]` (Indian Institute of Science), Apala Chakrabarti (Indian Institute of Science)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了 DREAMS 原型工具，用于辅助创建和维护设计研究方法学（DRM）中的参考模型和影响模型。

**💡 创新点**

创新点在于将 DRM 的语义要求（类型化节点、带符号因果关系、关联假设/引用）与图形可视化和可维护性（布局自动化、搜索检索）紧密集成，形成专门的建模环境。

**🔧 技术方法**

技术实现主要采用图形布局算法（分层/层次布局）、可编辑的图数据结构、关系嵌入式注释以及基于文本的搜索引擎；界面采用交互式绘图框架。

**📊 数据集**

评估使用了来自 16 名 DRM 研究者的调查数据来确定需求，并通过 4 名使用者在同一模型创建、修订与证据检索任务中进行实验对比，未使用公开数据集。

**📈 对比分析**

比较方法为实验室实验：在手工绘图和 DREAMS 两种条件下记录模型创建时间、修订时间、边交叉数、重新定位操作数、检索时间。结果显示 DREAMS 在模型创建时间约减少 57%、修订时间 92%、边交叉 77%、重新定位 100% 与检索时间 80%。

**⚠️ 局限性**

局限性包括样本量小（仅 4 人）、单一模型规模、缺乏协作与大规模模型支持、布局仍为基于规则的非最优方案、并未对因果推理或自动验证做深入探讨。

---

## 562. PC3D: Zero-Shot Cooperation Across Variable Rosters via Personalized Context Distillation

**arXiv ID:** 2605.10377 | [PDF](https://arxiv.org/pdf/2605.10377v1)

**作者:** Ahmet Onur Akman `[一作]` (Jagiellonian University), Rafał Kucharski `[通讯]` (Jagiellonian University)

**通讯引用:** 1075 | [OpenAlex ID](https://openalex.org/A5102776834)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

研究了多智能体强化学习中团队规模随情节变化的开放团队合作问题，并提出 PC3D 方法

**💡 创新点**

创新点在于将中心化的团队协调信息压缩为个性化上下文，并通过教师-学生蒸馏让去中心化策略能够从局部历史中恢复并自适应使用该上下文

**🔧 技术方法**

采用了集中训练去中心化执行（CTDE）框架，结合集合注意力的中心化评论家、个性化教师上下文、FiLM 条件化的演员以及教师-学生蒸馏

**📊 数据集**

在三套开放团队任务上评估：MPE 的 Spread、GridWorld 的 Level-based Foraging、仓库物流的 Multi-robot Warehouse

**📈 对比分析**

与 IPPO、MAPPO、PIC-MAPPO 基线比较，PC3D 在见过和未见的团队规模上均取得最高平均回报，尤其在大规模团队上提升显著

**⚠️ 局限性**

仅针对同质团队、只处理情节间团队规模变化，且对通信、异质性和实时团队变动的适应性尚未验证

---

## 563. Toward Multi-Database Query Reasoning for Text2Cypher

**arXiv ID:** 2605.10373 | [PDF](https://arxiv.org/pdf/2605.10373v1)

**作者:** Makbule Gulcin Ozsoy `[一作]` `[通讯]` (Neo4j), Makbule Gulcin Ozsoy (Neo4j)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了从单数据库查询生成到多数据库查询推理的三阶段框架，包括数据库路由、多数据库分解与结果集成，扩展了Text2Cypher到更真实的分布式图数据场景。

**💡 创新点**

创新点在于将文本到查询的任务从单一预选图数据库转移到多数据库场景，系统化阐述了源选择、查询分解与异构查询推理的关键技术，并首次提出了跨图数据库的三阶段推理流程。

**🔧 技术方法**

采用大语言模型进行文本到Cypher/SQL等查询语言的生成，利用语义相似度与图结构描述实现数据库路由，设计分解策略将复杂问题拆解为子查询并在不同数据库执行，最后通过语义对齐和结果聚合实现跨系统整合。

**📊 数据集**

论文为位置性工作，未提供公开数据集；示例采用电影制作公司手工构造的HR、Finance、Movies三个独立图数据库。

**📈 对比分析**

未实现系统，也未进行实验评估或与现有方法比较；因此缺乏具体的性能指标。

**⚠️ 局限性**

局限性包括：仅为研究路线图，缺乏大规模实验与基准；假设所有数据库已知且可访问，未考虑动态发现与数据源演化；未处理异构语义映射与结果整合细节；未对多语言（Cypher、SQL、GQL）交叉查询进行实证验证。

---

## 564. Sens-VisualNews: A Benchmark Dataset for Sensational Image Detection

**arXiv ID:** 2605.10394 | [PDF](https://arxiv.org/pdf/2605.10394v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 565. Valid Best-Model Identification for LLM Evaluation via Low-Rank Factorization

**arXiv ID:** 2605.10405 | [PDF](https://arxiv.org/pdf/2605.10405v1)

**作者:** Elad Tolochinsky `[一作]` (Technion Israel Institute of Technology), Yaniv Romano `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过将多臂老虎机与低秩预测结合，提出PULSE框架在有限评估预算下准确识别最佳LLM。

**💡 创新点**

在有限样本、无放回采样条件下使用双重稳健估计与低秩因式分解，既保持无偏性又显著降低方差，并给出有限样本置信区间。

**🔧 技术方法**

采用UCB‑E多臂老虎机、低秩逻辑回归预测、AIPW双重稳健估计、马尔可夫理论与伯努利分布等技术。

**📊 数据集**

使用4.4K模型、21.5K题目的六大Benchmark（MMLU‑Pro、BBH、GPQA、IFEval、MATH、MuSR）进行实验。

**📈 对比分析**

与UCB‑E和Naive‑Pooling对比，PULSE在95%识别准确率下最多减少46%评估调用，并保持置信区间合法，表现优于两基线。

**⚠️ 局限性**

方法依赖低秩预测质量，仅适用于二值评分，当前仅采用均匀采样，未覆盖非均匀或自适应问题选择。

---

## 566. Halo Separation-guided Underwater Multi-scale Image Restoration

**arXiv ID:** 2605.10374 | [PDF](https://arxiv.org/pdf/2605.10374v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 567. GuardAD: Safeguarding Autonomous Driving MLLMs via Markovian Safety Logic

**arXiv ID:** 2605.10386 | [PDF](https://arxiv.org/pdf/2605.10386v1)

**作者:** Tianyuan Zhang `[一作]` (Beihang University), Xianglong Liu `[通讯]` (Beihang University)

**通讯引用:** 13317 | [OpenAlex ID](https://openalex.org/A5024067284)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种模型无关的安全防护框架，利用神经符号逻辑与马尔可夫逻辑递归来实时修正多模态大型语言模型在自动驾驶中的不安全动作；

**💡 创新点**

创新点在于把安全状态建模为随时间演化的马尔可夫逻辑状态，并通过神经符号化、n阶马尔可夫逻辑诱导以及逻辑驱动的动作修正，实现对多模态输入的动态安全推理；

**🔧 技术方法**

使用神经符号化逻辑形式化、n阶马尔可夫逻辑网络（nMLN）进行状态诱导、以及基于提示生成的逻辑驱动动作修正；

**📊 数据集**

使用DriveLM和VRU-Accident两个公开基准（以及CARLA仿真和实车实验）进行评估；

**📈 对比分析**

与SafeAuto、RoboFactory、Code-as-Monitor等基线对比，平均减少约32%事故率，提升6.85% GPT分数，约23% Qwen分数，性能提升显著；

**⚠️ 局限性**

局限性包括对实体、谓词和动作的准确感知与提取高度依赖、谓词与规则集不够全面、修正过程存在额外计算开销以及对抗攻击的鲁棒性仍有提升空间。

---

## 568. Not All Proofs Are Equal: Evaluating LLM Proof Quality Beyond Correctness

**arXiv ID:** 2605.10379 | [PDF](https://arxiv.org/pdf/2605.10379v1)

**作者:** Ivo Petrov `[一作]` (INSAIT, Sofia University St. Kliment Ohridski), Martin Vechev `[通讯]` (INSAIT, Sofia University St. Kliment Ohridski)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个针对LLM生成数学证明的质量评估基准，关注证明的简洁性、计算便利性、认知简易性、多样性和适应性等维度；

**💡 创新点**

创新点在于：①将证明质量分解为可量化的可扩展代理指标；②引入对多模型、多答案的成对比较与Bradley‑Terry排名；③通过LLM裁判自动化评估而非人工标注；

**🔧 技术方法**

技术包括：大语言模型裁判（用于压缩、判定计算便利性、认知简易性、聚类与适应性评估），成对比较与Elo‑式BT排名，自动化的答案与完整性检查；

**📊 数据集**

使用382道高中级竞赛终极答案题，涵盖代数、几何、组合、数论与微积分；并采集对应的人工解答作为参考技术集；

**📈 对比分析**

比较方法：在每个问题上，只对两模型都给出正确证明的答案进行成对比较；得分通过BT模型转化为Elo排名；实验发现模型在不同质量指标上排名差异显著，准确率与部分质量指标（如计算便利性、简洁性）相关性低，表明存在显著的权衡；

**⚠️ 局限性**

局限性包括：①缺乏客观的“证明质量”定义，评估依赖LLM裁判的可靠性；②裁判可能存在偏差，无法完全匹配人类评价；③仅覆盖终极答案类竞赛题，可能不具备对更开放或研究级别问题的通用性；

---

## 569. Progressive Photorealistic Simplification

**arXiv ID:** 2605.10409 | [PDF](https://arxiv.org/pdf/2605.10409v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 570. Autonomous FAIR Digital Objects: From Passive Assertions to Active Knowledge

**arXiv ID:** 2605.10370 | [PDF](https://arxiv.org/pdf/2605.10370v1)

**作者:** Zeyd Boukhers `[一作]` (Fraunhofer Institute for Applied Information Technology), Christoph Lange `[通讯]` (Fraunhofer Institute for Applied Information Technology)

**通讯引用:** 2533 | [OpenAlex ID](https://openalex.org/A5007714809)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了自主FAIR数字对象（aFDO），将条件-动作策略、事件处理和多源共识协议嵌入FAIR对象本体，实现持续验证、动态发现与跨机构协调。

**💡 创新点**

创新点在于：1）将政策（SHACL+ODRL）、公告（ActivityStreams 2.0）与协议（trim‑weighted‑mean共识）直接集成进FAIR对象，形成可审计、标准化的自治知识治理；2）利用RDF‑star记录细粒度可信度，配合裁剪加权平均实现对拜占庭攻击的容忍。

**🔧 技术方法**

使用Python实现的DOIP接口、SHACL+ODRL评估引擎、ActivityStreams 2.0事件总线、信任注册表，以及RDF‑star/PROV‑O/SHACL/ODRL/ActivityStreams等W3C标准。

**📊 数据集**

以罕见疾病数据为主：ClinVar变异解释、HPO和Orphanet疾病定义，并生成与之对应的合成患者观察和临床评估，用以注入冲突与攻击场景。

**📈 对比分析**

与专家面板决议、简单多数投票及“先到达者”基线对比，整体准确率约70%；在拜占庭阈值（f < n/5）内准确率保持稳定，超过阈值时准确率降至≈0.5%~1%；在无攻击时裁剪加权平均与简单多数相近，且对极端攻击更具鲁棒性。

**⚠️ 局限性**

局限包括：1）发现层依赖中心化注册表，尚未实现DHT等去中心化方案；2）信任模型需外部声誉输入，冷启动仍有挑战；3）演化层（信任更新、共识后处理）仍为集中式，分布式协同尚未实现；4）仅在ACMG变异等级上评估，泛化到其他任务需进一步验证。

---

## 571. AgentGR: Semantic-aware Agentic Group Decision-Making Simulator for Group Recommendation

**arXiv ID:** 2605.10367 | [PDF](https://arxiv.org/pdf/2605.10367v1)

**作者:** Yangtao Zhou `[一作]` (Xidian University), Qingshan Li `[通讯]` (Xidian University)

**通讯引用:** 781 | [OpenAlex ID](https://openalex.org/A5100404954)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AgentGR模型，利用LLM驱动的多智能体模拟和语义元路径引导的链式偏好推理，改进群体推荐；

**💡 创新点**

创新点在于融合语义元路径与高阶协同过滤的CoP推理、对群体主题与领导进行语义识别，并设计静态与动态两种多智能体决策模拟策略，实现了更真实、更精准的群体决策建模；

**🔧 技术方法**

技术包括LLM（GPT‑4o）驱动的推理与对话、语义元路径构造、Chain‑of‑Preference（CoP）推理、BERT语义编码、超图神经网络以及多智能体对话框架；

**📊 数据集**

使用了两个公开群体推荐数据集：MafengwoS（旅游活动）和Weeplaces（餐饮聚会）；

**📈 对比分析**

与8种最先进基线（AGREE、GroupIM、HCR、CubeRec、ConsRec、AlignGroup、DisRec、LLM4GR）在HR@10/NDCG@10上进行比较，AgentGR平均提升约10%–20%；

**⚠️ 局限性**

主要局限在于动态多智能体模拟成本高、领导识别依赖语义相似度、对极少领导或极大群体的适应性尚待提升。

---

## 572. The Polynomial Counting Capabilities of Message Passing Neural Networks

**arXiv ID:** 2605.10393 | [PDF](https://arxiv.org/pdf/2605.10393v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 573. CellDX AI Autopilot: Agent-Guided Training and Deployment of Pathology Classifiers

**arXiv ID:** 2605.10362 | [PDF](https://arxiv.org/pdf/2605.10362v1)

**作者:** Alexey Pchelnikov `[一作]` (HistAI), Aleksei Pchelnikov `[通讯]` (HistAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 CellDX AI Autopilot 平台，能够让从无 ML 背景的病理学家到 ML 研发人员通过自然语言与 AI 代理交互，完成整个全切片图像分类器的训练、评估与部署流程。平台使用预先提取的 1024 维视觉 Transformer 特征，支持四种 MIL 聚合策略（Pooling、Attention、CLAM、LoRA），并提供分阶段的自动化超参数搜索与多策略比较，最终在无人工编程的情况下生成可部署模型。

**💡 创新点**

创新点在于：
1) 将领域专属的 agent 技能（包含工作流程、guardrail 与最佳实践）与通用 LLM 代理无缝结合，首次公开病理专用技能集；
2) 通过预提取特征构建免费的 32k+ 病例数据仓库，消除高昂的 GPU 预处理成本；
3) 采用分阶段 pairwise 超参数搜索，成本降低 30×，实现快速实验迭代；
4) 在统一平台上实现人机交互式部署，保证安全与合规；
5) 通过多策略比较与自动化评估，降低专家干预需求。

**🔧 技术方法**

技术栈包括：
- 预训练视觉 Transformer（Hibou‑L）提取 1024 维 patch 特征；
- MIL 框架实现四种聚合策略（mean、attention、CLAM、LoRA）；
- 自动化超参数搜索：分阶段 pairwise（grid 或 seeded random）与 early stopping；
- LLM 代理 + 结构化 agent 技能（JSON 规范）；
- Azure 云 GPU 集群（Autoscale）、Container Apps、Cosmos DB、Blob Storage；
- 监控与日志流（stdout JSON + 30s 轮询）及训练结果可视化；
- 安全与人机审批流程。

**📊 数据集**

使用的数据集为公开的 H&E‑染色全切片图像特征集，包含 32,000+ 病例、66,000+ 切片，全部预提取为 1024 维特征。数据来源商业数据库，平台免费开放给用户。

**📈 对比分析**

比较方法：
- 采用 AUROC、PR‑AUC、balanced accuracy、macro F1 等宏观指标；
- 支持 5‑fold stratified 交叉验证，报告均值±标准差；
- 对四种 MIL 策略进行并行训练并生成性能对比表；
- 结果显示：Attention‑MIL 通常优于 Pooling，CLAM 在大样本时进一步提升；LoRA 在迁移学习场景表现良好；
- 超参数搜索完成时间约为传统搜索的 1/30，实验周期从数小时缩短为数分钟。

**⚠️ 局限性**

局限性：
1) 仅支持 H&E 切片的分类任务；
2) 固定使用单一特征提取器（Hibou‑L），无法进行端到端微调或切换其他 foundation model；
3) MIL 聚合不考虑空间关系，无法捕捉组织结构特征；
4) pairwise 超参数搜索假设参数组间独立，可能忽略三重或多重交互；
5) agent 技能目前仅为结构化文档，缺乏形式化安全或性能保证；
6) 部署需要人工批准，无法实现全自动化；
7) 数据集覆盖有限，缺乏 IHC、特殊染色及非分类任务；
8) 需要手工标注 IHC 控件，成本高。

---

## 574. Agent-X: Full Pipeline Acceleration of On-device AI Agents

**arXiv ID:** 2605.10380 | [PDF](https://arxiv.org/pdf/2605.10380v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 575. Foundations of Reliable Inference: Reliability-Efficiency Co-Design

**arXiv ID:** 2605.10351 | [PDF](https://arxiv.org/pdf/2605.10351v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 576. VEGA: Visual Encoder Grounding Alignment for Spatially-Aware Vision-Language-Action Models

**arXiv ID:** 2605.10485 | [PDF](https://arxiv.org/pdf/2605.10485v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 577. ObfAx: Obfuscation and IP Piracy Detection in Approximate Circuits

**arXiv ID:** 2605.10355 | [PDF](https://arxiv.org/pdf/2605.10355v1)

**作者:** Lukas Sekanina `[一作]` (Brno University of Technology), Vojtech Mrazek `[通讯]` (Brno University of Technology)

**通讯引用:** 1411 | [OpenAlex ID](https://openalex.org/A5071584701)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了近似电路的知识产权（IP）盗版检测，提出了近似混淆（approximate obfuscation）攻击模型，并基于错误热图构建了自动化的检测框架。

**💡 创新点**

创新点包括：①首次定义近似混淆概念并用CGP生成混淆电路；②构建了大规模（1,775个）近似乘法器混淆数据集；③提出多种基于错误热图的二分类器（阈值、特征+机器学习、Siamese网络），实现>98% 的检测准确率。

**🔧 技术方法**

使用技术包括：Cartesian Genetic Programming（CGP）进行混淆电路搜索；错误热图与误差指标（WCE、MAE、EP）分析；机器学习分类器（Random Forest、Decision Tree、MLP、SVM）和Siamese卷积网络；对输入交换进行数据增强。

**📊 数据集**

数据集：75个人工设计的8位近似乘法器（不同族群）+ 22个EvoApproxLib乘法器，共生成1,775个混淆乘法器；构造了10,650对错误热图样本（基准集），并扩展为4倍（含输入交换）的完整数据集。

**📈 对比分析**

比较方法：基于错误热图的二分类；对CL1（单一误差阈值）、CL2（特征+ML分类器）和CL3（Siamese网络）进行实验。Random Forest在基准集上准确率99.6%，在交换集上98.4%；Siamese在完整训练集上准确率>94%，整体检测精度>98%。

**⚠️ 局限性**

局限性：对输入交换的鲁棒性依赖数据增强；仅针对8位乘法器，扩展到更大规模或其他近似模块需进一步验证；CGP生成的混淆电路唯一性有限；在只获取部分输入向量时检测准确率下降。

---

## 578. Priority-Driven Control and Communication in Decentralized Multi-Agent Systems via Reinforcement Learning

**arXiv ID:** 2605.10482 | [PDF](https://arxiv.org/pdf/2605.10482v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 579. Automated Detection of Abnormalities in Zebrafish Development

**arXiv ID:** 2605.10464 | [PDF](https://arxiv.org/pdf/2605.10464v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 580. CMKL: Modality-Aware Continual Learning for Evolving Biomedical Knowledge Graphs

**arXiv ID:** 2605.10510 | [PDF](https://arxiv.org/pdf/2605.10510v1)

**作者:** Yousef A. Radwan `[一作]` (King Abdullah University of Science and Technology), Xikun Zhang `[通讯]` (RMIT University)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5061638152)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种持续学习的多模态生物医学知识图谱学习框架CMKL，能够在图结构、文本和分子特征随时间演化时保留已有知识并不断学习新知识。

**💡 创新点**

① 针对文本（冻结）、结构（可学习）和分子三种模态的显著差异设计Mixture‑of‑Experts（MoE）融合路由器，能够在训练过程中抑制不可学习模态；② 将EWC正则化与K‑means多模态重放缓冲结合，实现对不同模态的专属保护；③ 通过实验系统化分析“贪婪模态”问题在持续学习中的表现。

**🔧 技术方法**

使用R‑GCN编码图结构，冻结BiomedBERT提取文本表示，Morgan指纹+MLP编码分子特征；Mixture‑of‑Experts路由器进行模态加权融合；DistMult双线性解码器进行链接预测；EWC正则化（按模态调参）与多模态K‑means重放缓冲；Adam优化器。

**📊 数据集**

基于PrimeKG构建的129K实体、8.1M边缘的持续学习基准，共10个任务（按实体类型划分），包含结构、文本、分子三种模态。

**📈 对比分析**

与Naive Sequential、Joint Training、EWC、LKGE等基线比较。持续关系预测：AP≈0.062（与EWC相当，显著优于LKGE 0.039、Joint 0.047）；持续实体分类：AP≈0.591，较单模态和Joint baseline提升约+60%，几乎无遗忘（AF≈0.008），并呈正向后向迁移。

**⚠️ 局限性**

限制：① 关系预测提升在统计噪声范围内无显著差异；② 模态融合效果受解码器类型限制，仅适用于双线性DistMult，无法直接扩展到旋转或Hermitian解码器；③ 仅针对单模态保留与冻结差异的场景，未深入探讨多模态自适应训练策略。

---

## 581. Automated high-frequency quantification of fish communities and biomass using computer vision

**arXiv ID:** 2605.10449 | [PDF](https://arxiv.org/pdf/2605.10449v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 582. Safe Multi-Agent Behavior Must Be Maintained, Not Merely Asserted: Constraint Drift in LLM-Based Multi-Agent Systems

**arXiv ID:** 2605.10481 | [PDF](https://arxiv.org/pdf/2605.10481v1)

**作者:** Tianxiao Li `[一作]` (University of Liverpool), Guangliang Cheng `[通讯]` (University of Liverpool)

**通讯引用:** 2047 | [OpenAlex ID](https://openalex.org/A5045854934)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并验证了约束状态治理（Constraint State Governance, CSG）与约束原生强化学习（Constraint Native Reinforcement Learning）两层协同框架，用以追踪并维护多智能体LLM系统在执行轨迹中的安全约束，避免约束漂移。

**💡 创新点**

创新点在于将安全约束转化为可签名的状态对象，在委托、通信、工具调用和执行等接口处进行实时的可重放审计与授权检查，从而实现轨迹级的约束保持，并将学习目标限制在保持约束的可接受轨迹空间内。

**🔧 技术方法**

技术包括基于哈希与签名的约束标记、可重放的状态审计链、信息流与权限管理策略、以及在强化学习奖励计算中加入可接受性检验的约束原生奖励函数。

**📊 数据集**

实验基于AgentLeak公开数据集（4,979条医疗、金融、法律及企业领域的多步交互轨迹）进行回放与离线策略搜索，并在同一数据集上对比输出过滤与CSG Lite的泄漏率。

**📈 对比分析**

比较方法通过统计最终输出、内部消息与共享内存中的泄漏比例，结果显示单纯输出过滤的泄漏率为27.2%，而CSG Lite可将总泄漏率降至4.6%，并且在离线策略搜索中，约束原生策略在保证安全约束的前提下实现了约12.5%的效用提升。

**⚠️ 局限性**

局限性包括：对约束的形式化要求较高，无法完全覆盖模糊或情境化的安全指令；在高并发或低风险场景下，治理开销可能影响系统效率；同时，CSG仍需与沙箱、最小权限、外部审计等传统安全措施结合才能实现端到端安全保障。

---

## 583. Can Agent Benchmarks Support Their Scores? Evidence-Supported Bounds for Interactive-Agent Evaluation

**arXiv ID:** 2605.10448 | [PDF](https://arxiv.org/pdf/2605.10448v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 584. Formally Verifying Analog Neural Networks Under Process Variations Using Polynomial Zonotopes

**arXiv ID:** 2605.10474 | [PDF](https://arxiv.org/pdf/2605.10474v1)

**作者:** Yasmine Abu-Haeyeh `[一作]` (Goethe University Frankfurt), Lars Hedrich `[通讯]` (Goethe University Frankfurt)

**通讯引用:** 972 | [OpenAlex ID](https://openalex.org/A5038049766)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建基于多项式锥体的参数化模型，并使用可达性分析实现对模拟神经网络在工艺变化下的系统级形式化验证。

**💡 创新点**

首次将多项式锥体用于描述工艺参数对模拟神经元的高阶影响，并在网络级别实现可伸缩的验证框架，显著缩短验证时间。

**🔧 技术方法**

多项式锥体（set‑based computing）、可达性分析、极大极小优化及 MATLAB/CORA 工具箱实现。

**📊 数据集**

Wisconsin Breast Cancer、Iris、MNIST（全连接与 CNN）三类分类基准。

**📈 对比分析**

与传统 Monte Carlo 仿真对比，Monte Carlo 需数小时至数天，本文方法仅需秒级；覆盖率约 99%，验证准确率与标称模型相近。

**⚠️ 局限性**

对更大、层数更多的模拟网络鲁棒性不足；仅考虑工艺变化，未覆盖设备不匹配和噪声；模型构建需要先行对单元级电路进行 Monte Carlo 前置。

---

## 585. The Renaissance of Repair: A Timely Opportunity for Fabrication Research

**arXiv ID:** 2605.10450 | [PDF](https://arxiv.org/pdf/2605.10450v1)

**作者:** Julian Britten `[一作]` (Ulm University), Jan Henry Belz `[通讯]` (Dr. Ing. h.c. F. Porsche AG)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出基于五步（问题识别、方案探索、材料获取、执行维修、测试验证）的维修流程，并探讨每一步为个人制造研究提供的机遇与挑战，主张将维修置于该领域的核心议题。

**💡 创新点**

首次将维修过程系统化为可拆解的五个步骤，强调在个人制造研究中以维修为中心的创新范式，并呼吁从技术、方法到社会层面共同推动维修的可持续发展。

**🔧 技术方法**

围绕3D打印、参数化设计工具、AR任务指导等技术提出改进思路，探讨如何为不同技能层次的用户提供支持。

**📊 数据集**

本文未采用具体数据集，主要基于文献综述、案例分析和现有在线资源（如MakerWorld、Thingiverse、iFixit）进行论证。

**📈 对比分析**

未进行实验或性能对比，文章主要以理论框架与设计方向为主，未给出定量评估结果。

**⚠️ 局限性**

缺乏具体实现与实证验证，未对旧产品维修的细节进行深入探讨，且所提技术方案在实际操作中的可行性与成本仍需进一步研究。

---

## 586. SoK: A Systematic Bidirectional Literature Review of AI & DLT Convergence

**arXiv ID:** 2605.10515 | [PDF](https://arxiv.org/pdf/2605.10515v1)

**作者:** Ali Irzam Kathia `[一作]` (Exponential Science Foundation), Marco Alberto Javarone `[通讯]` (Exponential Science Foundation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2020‑2025 年间的 53 篇 AI 与 DLT 交叉研究进行系统性、双向综述，按层级与应用域归类并比较两种提升方向（AI 强化 DLT 与 DLT 强化 AI）。

**💡 创新点**

提出统一的 5 层架构对照表，首次系统展示两种增强方向的交叉层级分布与共性设计模式，并识别了技术空白与研究挑战。

**🔧 技术方法**

使用层级分类框架、系统检索与手工编码、定量统计（层级占比、研究主题），并结合案例归纳 AI 机制（RL、DL、LLM、ZKP 等）与 DLT 机制（智能合约、共识、PoC 等）。

**📊 数据集**

采用 53 篇文献的元数据作为“数据集”，包括作者、年份、会议/期刊、研究方向、层级、技术、应用域等信息。

**📈 对比分析**

通过统计每层研究数量与百分比、跨层对比、技术使用频率，展示 AI 在执行/共识层的集中度，DLT 在数据/模型层的突出应用；未给出传统性能数值，但指出大多实验在原型/模拟环境，缺乏生产级验证。

**⚠️ 局限性**

局限性包括：仅覆盖 2020‑2025 期刊与会议，排除非正式论文；缺乏实测性能对比；大多研究停留在原型/测试网络，未实现大规模生产；跨链、互操作性评估不足；未系统探讨对抗鲁棒性与能耗问题。

---

## 587. ASIA: an Autonomous System Identification Agent

**arXiv ID:** 2605.10480 | [PDF](https://arxiv.org/pdf/2605.10480v1)

**作者:** Dario Piga `[一作]` (Dalle Molle Institute for Artificial Intelligence (IDSIA), SUPSI), Marco Forgione `[通讯]` (Dalle Molle Institute for Artificial Intelligence (IDSIA), SUPSI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个基于大型语言模型的自主系统辨识代理框架（ASIA），能够在无人工干预的情况下自动搜索模型类、网络结构和训练策略并完成完整的辨识流程。

**💡 创新点**

创新点在于：①将模型设计与训练过程交给LLM自治，打破传统固定模型族的限制；②通过迭代推理实现对混合物理‑黑盒架构和训练技巧的自动生成；③显式探索更广阔的假设空间，提升搜索效率。

**🔧 技术方法**

核心技术包括：大型语言模型（Claude Code / Sonnet 4.6）、自动代码生成与执行、交互式实验循环、交叉验证评估以及对模型与超参数的程序化修改。

**📊 数据集**

实验数据集：1）Cascaded Two‑Tank（1024样本，采样周期4 s）用于测试非线性液位辨识；2）Crazyflie 2.1 nanodrone（四种飞行轨迹，采样周期10 ms）用于多输入多输出动力学辨识。

**📈 对比分析**

与随机搜索和公开基准进行对比：在两个任务中，ASIA均达到或超过最佳公开结果；对Two‑Tank，RMSE 0.298 与排行榜顶尖值相当；对nanodrone，交叉验证MAE 0.286 超过physics+residual网络与随机搜索，并在测试集上取得最低MEE。

**⚠️ 局限性**

主要局限包括：潜在的测试集泄漏导致结果偏差；训练流程的全局可解释性不足，难以重现完整搜索轨迹；对LLM训练数据的偏倚可能限制创新性；需要人工介入以提供先验知识和验证最终模型。

---

## 588. A Flexible Raspberry Pi-Based Data Logger Platform for Modbus Sensors with Ansible Deployment

**arXiv ID:** 2605.10454 | [PDF](https://arxiv.org/pdf/2605.10454v1)

**作者:** Leon Keim `[一作]` (University of Stuttgart), Holger Class `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一款基于Raspberry Pi的Modbus传感器数据记录平台LibrePiLogger，实现了从硬件组装到软件部署的全流程自动化；

**💡 创新点**

创新点在于将AtmosPyre Python库与Ansible自动化部署相结合，单一YAML文件即可快速配置任何RS‑485 Modbus传感器，并在短时间内完成新传感器的驱动集成；

**🔧 技术方法**

采用了Raspberry Pi Zero/4、RS‑485 HAT或USB‑to‑RS‑485转换器、Python的AtmosPyre库、Ansible剧本、systemd服务等技术；

**📊 数据集**

在地下喀斯特环境中连续测量CO₂（Vaisala GMP252）和²²²Rn（RadonTech AlphaTRACER）传感器数据；

**📈 对比分析**

与传统商业数据记录器（Radon Scout、AlphaE/AlphaGUARD、Vaisala手持式CO₂仪）并行部署，结果显示两系统在时间趋势与误差范围内高度一致，且LibrePiLogger能够在电源中断后自动恢复；

**⚠️ 局限性**

局限包括需网络连接才能实现远程重配置，新增传感器需自行编写约100行Python驱动，RS‑485总线长度有限，以及用户需自行选择电源和外壳。

---

## 589. HiRL: Hierarchical Reinforcement Learning for Coordinated Resource Management in Heterogeneous Edge Computing

**arXiv ID:** 2605.10443 | [PDF](https://arxiv.org/pdf/2605.10443v1)

**作者:** Jianyong Zhu `[一作]` (North China Electric Power University), Renyu Yang `[通讯]` (Beihang University)

**通讯引用:** 2979 | [OpenAlex ID](https://openalex.org/A5050796169)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种层次化强化学习框架，联合调度边缘设备的功率控制和任务分配，支持CPU/GPU异构环境下的实时资源管理；

**💡 创新点**

创新点包括将连续功率决策与离散任务分配分层处理，使用GPU兼容性评分实现任务-资源匹配，结合期限导向队列管理和失效惩罚经验重放提升学习效率；

**🔧 技术方法**

采用Twin Delayed Deep Deterministic Policy Gradient (TD3) 进行功率控制，Double Deep Q‑Network (DDQN) 进行任务分配，并通过三阶段协调引擎实现两层决策同步；

**📊 数据集**

使用基于模拟的异构边缘测试集，包含35台移动终端、5台异构服务器，CPU/GPU/IO任务按5:4:1比例生成，任务优先级、截止时间、资源需求等真实参数；

**📈 对比分析**

相较于随机、贪心、轮询、QPSO和单DDQN等基线，所提框架在低至高负载下均实现了近100%任务完成率、平均延迟降低约28%、能耗在低负载下降低51%，并在能耗-延迟权衡图中占据最优区域；

**⚠️ 局限性**

局限性包括实验仅在仿真环境验证，缺乏真实设备的硬件层面验证；对GPU资源分割细粒度的支持有限；当服务器数量或任务类型显著增大时，兼容性评分与经验回放的计算成本仍需进一步评估；

---

## 590. StereoTales: A Multilingual Framework for Open-Ended Stereotype Discovery in LLMs

**arXiv ID:** 2605.10442 | [PDF](https://arxiv.org/pdf/2605.10442v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 591. DRIFT: Drift-Resilient Invariant-Feature Transformer for DGA Detection

**arXiv ID:** 2605.10436 | [PDF](https://arxiv.org/pdf/2605.10436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 592. Don't Fix the Basis -- Learn It: Spectral Representation with Adaptive Basis Learning for PDEs

**arXiv ID:** 2605.10451 | [PDF](https://arxiv.org/pdf/2605.10451v1)

**作者:** Xuxiang Zhao `[一作]` (Tsinghua University), Angelica I. Aviles-Rivero `[通讯]` (Tsinghua University)

**通讯引用:** 2359 | [OpenAlex ID](https://openalex.org/A5013015879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Adaptive Basis Learning (ABLE) 框架，将神经算子中的固定全局谱基变为数据驱动的可学习 Parseval 框架，直接在自适应谱空间中学习算子，显著提升 PDE 预测精度。

**💡 创新点**

创新点在于：① 用可学习的稠密度函数 p(x,y) 生成空间自适应谱基，打破传统固定 Fourier 基底的限制；② 证明该变换保持可逆性和 Parseval 恒等式，确保能量守恒；③ 将可学习基底与谱乘子耦合，得到严格大于 FNO 的算子类；④ 通过 Softmax‑MLP 控制温度 T，实现基底的“相变”与平滑自适应；⑤ 仍保持 O(N log N) 复杂度，兼容 FFT。

**🔧 技术方法**

技术包括：Parseval 帧理论、FFT 与离散傅里叶变换、软最大化的温度控制、MLP 生成稠密度、交叉相位交互、可微编程实现；实验中还使用了 1D/2D PDE 数据集、标准深度学习框架（PyTorch）和 GPU 训练。

**📊 数据集**

使用公开 PDE 数据集：1D Burgers 方程（多粘性系数）、2D Darcy 流动、2D Navier–Stokes（低黏度），与之前工作中相同的数据划分和训练设置保持一致。

**📈 对比分析**

与多种基准模型（DeepONet、U‑Net、FNO、SNO、WNO、GF‑NO、AFNO、GaborFNO、HPM、SAOT、FreqMoE 等）进行对比。ABLE 在 Burgers 的低粘性场景下降低 L2 误差约 20%（从 6.12e‑3 降到 4.65e‑3）；在 Darcy 流动中将 FNO 的相对误差从 0.00626 降至 0.00552；在 Navier–Stokes 中 FNO 误差从 0.1237 降至 0.0985，且结合 HPM、SAOT 的版本进一步提升到 0.0705。总体而言，ABLE 在所有三类 PDE 上均表现出显著的准确性提升，且计算开销仅略高于 FNO（M≤3 的话约 2× FLOPs，仍低于大多数基准）。

**⚠️ 局限性**

局限性：① 需要额外的 M 维基底和温度 T 作为超参数，若选取不当可能导致过拟合或信息冗余；② 在极大 M 的情况下，参数量与计算成本会快速增长；③ 目前仅在平移不变域（网格化域）实验，未验证在非规则几何或高维空间的推广性；④ 依赖 FFT，因而对非周期边界或非欧几里得结构的适应性有限；⑤ 需要额外的训练时间来学习基底，尤其在大规模数据集上可能成为瓶颈。

---

## 593. Data Path Fusion in GPU for Analytical Query Processing

**arXiv ID:** 2605.10511 | [PDF](https://arxiv.org/pdf/2605.10511v1)

**作者:** Tsuyoshi Ozawa `[一作]` (University of Tokyo), Kazuo Goda `[通讯]` (University of Tokyo)

**通讯引用:** 684 | [OpenAlex ID](https://openalex.org/A5016349116)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Data Path Fusion（DPF）架构，将GPU端的IO、解压和查询操作融合为单个核函数，减少CPU干预，实现端到端GPU执行。

**💡 创新点**

将数据路径完整融合为单核、使用类型特定压缩、可变长度字段支持以及GPU驱动的BaM IO，三者协同实现更高吞吐。

**🔧 技术方法**

GPU Kernel Fusion、BaM（GPU‑initiated NVMe IO）、类型特定压缩（GPU‑FOR、FSST）以及基于块的页级剪枝和RID索引。

**📊 数据集**

使用TPC‑H（scale 100）和Star‑Schema Benchmark（SSB，scale 100）进行实验。

**📈 对比分析**

与基线GiDP、GiDP+BaM、GiDP+BaM+KF以及Polars、Spark‑RAPIDS、DuckDB比较，DPF在TPC‑H上实现2.66–6.22×加速，在SSB上实现3.84–16.81×加速，单核查询时间显著降低，kernel 调用和IO量亦大幅减少。

**⚠️ 局限性**

目前仅支持单GPU、仅处理哈希表能装进GPU内存的join、压缩方案仅限GPU‑FOR/FSST，缺少多GPU扩展与更多压缩算法支持。

---

## 594. Learning Less Is More: Premature Upper-Layer Attention Specialization Hurts Language Model Pretraining

**arXiv ID:** 2605.10504 | [PDF](https://arxiv.org/pdf/2605.10504v1)

**作者:** Jinchang Zhu `[一作]` (Hong Kong University of Science and Technology), Menglin Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 125242 | [OpenAlex ID](https://openalex.org/A5100743975)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 GPT 风格解码器预训练中出现的“过早上层注意力专化”失效模式，并通过在早期阶段仅减慢上层查询/键学习率的干预，显著提升模型最终困惑度、训练效率和下游任务性能；进一步通过对比 LLaMA 风格多路 FFN 架构，证明多路乘法门 FFN 能抑制此失效。

**💡 创新点**

提出并验证了一个新的优化失效模式——上层注意力过早收敛到不稳定的残差基；揭示了单分支 FFN 与多路门 FFN 在阻止该失效上的根本差异，并给出了基于路径推导的理论证明；通过实验展示了学习率干预与门 FFN 如何分别从优化步长与残差能量两侧抑制该失效。

**🔧 技术方法**

核心技术包括：针对解码器块的层级残差基分析；设计上层查询/键学习率干预；使用多路乘法门 FFN（SwiGLU/GEGLU）；机制探测方法（注意力熵、logit 大小、因果依赖性、残差写入能量等）；以及基于残差能量与学习率的解析上界。

**📊 数据集**

使用 2.5 B FineWeb‑Edu 语料，训练 270 M 参数 GPT 风格和 LLaMA 风格的 20 层因果解码器；同时在 0.7 B 参数的 GPT 解码器上进行扩展实验。

**📈 对比分析**

对比实验显示：在 GPT 风格模型中，早期上层学习率干预使最终困惑度下降约 0.5（从 26.8 降至 26.3），训练所需 token 减少 13%；下游评测平均得分提升 0.41 分；在 0.7 B 模型中，困惑度提升 0.13，token 减少 0.54 B。相比之下，LLaMA 风格模型几乎不受干预影响；单分支 FFN 与多路门 FFN 的对比进一步验证了门结构的抑制效果。

**⚠️ 局限性**

实验规模受限于学术预训练规模，未覆盖工业级大规模模型；结果主要针对 GPT 风格与 LLaMA 风格的 270 M 参数范围，未证明在更大模型或不同任务上的普适性；此外，干预和门结构的具体超参数选择仍需要针对不同模型做进一步调优。

---

## 595. Accelerating Compound LLM Training Workloads with Maestro

**arXiv ID:** 2605.10501 | [PDF](https://arxiv.org/pdf/2605.10501v1)

**作者:** Xiulong Yuan `[一作]` (Alibaba Group), Jingren Zhou `[通讯]` (Alibaba Group)

**通讯引用:** 8005 | [OpenAlex ID](https://openalex.org/A5057864403)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 Maestro 框架，针对 compound LLM 训练中的静态与动态异构性问题，采用 section‑centric 分解、针对每个 section 的专属并行策略、微批大小以及动态 wavefront 调度等技术，提升多模态与知识蒸馏等工作负载的训练效率。

**💡 创新点**

将训练工作拆分为独立 section 并允许每个 section 拥有不同的并行度、微批和资源分配；引入 wavefront scheduler 动态重排样本以最大化跨 section 并行；使用异步一侧 RDMA 消息队列实现高效跨 section 通信；引入 fan‑out 机制解决教师与学生批次不匹配；层次化优化策略显著降低全局配置搜索复杂度。

**🔧 技术方法**

section 构建、层次化优化、wavefront 调度、fan‑out 机制、异步非对称消息队列、张量/上下文/流水线/专家并行等多种并行与通信技术。

**📊 数据集**

使用 Qwen3.5‑400B‑A17B、Qwen3‑Next‑80B‑A3B 两款多模态模型；训练集为 32K 长度的多模态（文本‑图像/音频）数据；以及采用 KL 损失的知识蒸馏训练。

**📈 对比分析**

与 Megatron‑LM 直接对比，保持关键 section 资源相同，测量 end‑to‑end 令牌吞吐量和每 GPU 吞吐量；在多模态训练中实现 1.4× 端到端吞吐、1.24× 每 GPU；在蒸馏中实现 1.75× 端到端吞吐、1.4× 每 GPU；整体可节省约 40% GPU 资源。

**⚠️ 局限性**

需要手工构造 section 与配置；对不同模型、不同异构模式的通用性有限；对极大规模集群的可扩展性和动态负载预测的准确性仍需进一步验证。

---

## 596. Multi-layer attentive probing improves transfer of audio representations for bioacoustics

**arXiv ID:** 2605.10494 | [PDF](https://arxiv.org/pdf/2605.10494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 597. DeepRefine: Agent-Compiled Knowledge Refinement via Reinforcement Learning

**arXiv ID:** 2605.10488 | [PDF](https://arxiv.org/pdf/2605.10488v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 598. Privacy-preserving Chunk Scheduling in a BitTorrent Implementation of Federated Learning

**arXiv ID:** 2605.10499 | [PDF](https://arxiv.org/pdf/2605.10499v1)

**作者:** Naicheng Li `[一作]` (IMDEA Networks Institute), Nikolaos Laoutaris `[通讯]` (IMDEA Networks Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 FLTorrent，一种基于 BitTorrent 的分发层，旨在实现去中心化联邦学习的服务器无状态同步，同时通过短暂的 warm‑up 阶段提升源可识别度隐私；

**💡 创新点**

创新点在于结合 pre‑round obfuscation、随机时间延迟以及非所有者优先的调度策略，在保持 BitTorrent 高吞吐的前提下，以覆盖集阈值 k 与所有者限流实现 1/k 级别的源不可链接性；

**🔧 技术方法**

核心技术包括：BitTorrent swarming、预热调度（最大流上界与 GreedyFastestFirst 近似实现）、随机延迟、一次性喷射（pre‑round spray）以及可审计的跟踪器；

**📊 数据集**

评估使用 MNIST、CIFAR‑10 的 IID 与非 IID 数据集，以及四个 LLM 规模模型（Gemma‑7B、DeepSeek‑R1‑14B、Qwen2.5‑32B、Llama‑3.3‑70B）在不同网络带宽下的传播实验；

**📈 对比分析**

在 50 轮通信实验中，FLTorrent 的准确率与中心化 FL 接近，并显著优于 Gossip DFL；在通信成本上，warm‑up 占比约 12% 且吞吐率 75%–80%，LLM 规模下总时延仅比纯 BitTorrent 高 6–10%；

**⚠️ 局限性**

局限性包括：只对观察型内部攻击（非内容攻击）提供隐私保障，假设跟踪器可信且无协同；未针对 Sybil、跨轮链接和更强的恶意客户端做深入分析；

---

## 599. Simultaneous Long-tailed Recognition and Multi-modal Fusion for Highly Imbalanced Multi-modal Data

**arXiv ID:** 2605.10498 | [PDF](https://arxiv.org/pdf/2605.10498v1)

**作者:** Heegeon Yoon `[一作]` (Korea Advanced Institute of Science and Technology), Heeyoung Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2906 | [OpenAlex ID](https://openalex.org/A5100716622)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种联合多模态与长尾分布的识别框架，在专家网络中集成多模态输入并通过可信度加权进行融合。

**💡 创新点**

创新点在于将多专家架构与多模态融合相结合，并使用TCP（真实类别概率）动态评估模态重要性，针对非图像模态设计训练阶段聚合权重学习。

**🔧 技术方法**

采用SADE多专家结构、MMD风格的TCP估计、模态特定网络、数据增强或训练期聚合权重优化等技术。

**📊 数据集**

使用MNIST+SVHN混合图像数据集以及SIIM‑ISIC 皮肤病变图像+表格元数据的医学数据集进行验证。

**📈 对比分析**

在F1和准确率上与MMD、SADE、MMD‑LA、M^2LC‑Net、SADE‑LMF等基线相比，F1显著提升（最高0.601，远超0.53），准确率保持竞争力。

**⚠️ 局限性**

局限性包括假设训练与测试分布相同、无法处理部分标签或完全无监督场景，且需要额外的专家训练与模态评估网络。

---

## 600. AxiomOcean: Forecasting the Three-Dimensional Structure of the Upper Ocean

**arXiv ID:** 2605.10455 | [PDF](https://arxiv.org/pdf/2605.10455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 601. Uni-Synergy: Bridging Understanding and Generation for Personalized Reasoning via Co-operative Reinforcement Learning

**arXiv ID:** 2605.10445 | [PDF](https://arxiv.org/pdf/2605.10445v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 602. Self-Attention as a Covariance Readout: A Unified View of In-Context Learning and Repetition

**arXiv ID:** 2605.10466 | [PDF](https://arxiv.org/pdf/2605.10466v1)

**作者:** Haoren Xu `[一作]` (Fudan University), Guanhua Fang `[通讯]` (Fudan University)

**通讯引用:** 173 | [OpenAlex ID](https://openalex.org/A5015142209)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文通过理论分析证明，softmax注意力机制在长期上下文中等价于对输入协方差的线性读取。利用这一结论，作者展示了：①单个注意力头可实现一次“整体梯度下降”步，完成线性回归的 in-context learning；②将多层注意力堆叠后相当于多步梯度下降，逼近 Bayes 最优预测；③将协方差读取通过多层传播，导致自回归生成的条件分布在长期上下文趋向于一阶马尔可夫链，从而解释了重复生成和模式坍塌现象。

**💡 创新点**

创新点在于：①首次将 softmax 注意力解释为协方差读取的统计估计；②将 ICL 与重复生成统一在同一“协方差读取”原理下；③用单头梯度下降和多层马尔可夫闭包的理论框架解释两种现象。

**🔧 技术方法**

技术手段主要包括：Birkhoff ergodic 定理、椭圆子高斯强混合输入模型、协方差读取的闭式极限推导、残差堆叠梯度下降分析，以及一阶马尔可夫链极限证明。

**📊 数据集**

实验使用的是人工合成数据：随机生成协方差矩阵与线性回归参数，采样高斯输入向量和对应目标，评估注意力头与多层堆叠对回归目标的逼近。

**📈 对比分析**

与基线（Bayes 最优线性回归、标准梯度下降等）的比较：在上下文长度增大时，注意力模型的余弦相似度趋近于 1，均方误差趋近于 0，表明模型在理论极限下实现了与最优线性回归相同的性能；多层堆叠进一步逼近多步梯度下降的迭代结果。

**⚠️ 局限性**

局限性包括：假设输入为平稳、可测、椭圆子高斯且弱相依；分析仅针对无限长上下文，无法直接处理真实对话或短上下文；仅证明线性回归场景的 ICL 结果，未覆盖更复杂任务；不考虑训练过程对参数的影响，假设参数已固定。

---

## 603. Coherency through formalisations of Structured Natural Language, A case study on FRETish

**arXiv ID:** 2605.10462 | [PDF](https://arxiv.org/pdf/2605.10462v1)

**作者:** Joost J. Joosten `[一作]` (Universitat de Barcelona), Sofía Santiago Fernández `[通讯]` (Universitat de Barcelona)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在 FRET 结构化自然语言需求的 MTL 翻译中引入“连贯性”原则，并给出新的 MTL 翻译方法。

**💡 创新点**

创新点在于提出 C1、C2 连贯性原则，并设计出保持需求逻辑结构一致的 MTL 翻译方案，改进了 NASA FRET 的翻译。

**🔧 技术方法**

使用的方法包括：结构化自然语言（FRETISH）、度量时序逻辑（MTL）、模型检查、Python 脚本自动生成翻译、统计公式复杂度。

**📊 数据集**

数据集为 FRET 提供的 240 个模板（各包含 scope、condition、timing、response），对参数 k=3 的有限时间模板进行实验。

**📈 对比分析**

比较方法是对所有模板生成两套翻译，分别测量公式长度、时序深度和命题出现次数；实验显示新翻译平均公式更短、深度略低、命题出现更少，但因使用完整 MTL，工具链兼容性差。

**⚠️ 局限性**

局限性包括：翻译使用全 MTL（过去+未来），目前工业工具不支持，导致与现有验证后端不兼容；同时仍存在语义细微差异导致的潜在不一致，且多层形式化可能增加维护成本。

---

## 604. A Note on Banaszczyk's Inequality

**arXiv ID:** 2605.10461 | [PDF](https://arxiv.org/pdf/2605.10461v1)

**作者:** Hongyuan Qu `[一作]`, Guangwu Xu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在离散高斯测度上改进Banaszczyk不等式，给出在基点距离足够大且基的最短向量满足一定下界时的指数级更紧的上界。

**💡 创新点**

创新点在于加入了对最短向量 λ₁(L) 的额外约束，使得不等式的指数因子从原来的 (c√e·e^{-c²/2})ⁿ 进一步收缩为 (e^{1-c²})^{n/2}/(1-ε)，从而实现显著的指数级改进。

**🔧 技术方法**

主要使用了 Poisson 求和公式、快速递减函数理论以及高斯测度的基本性质，对离散高斯质量进行分块估计。

**📊 数据集**

无数据集；该工作为理论性改进，未涉及实验数据。

**📈 对比分析**

与原始 Banaszczyk 不等式及 Tian‑Liu‑Xu 的改进版进行比较，得到在满足 λ₁(L)≥kcs√{n/2π} 且 k>1 时，误差上界下降至 (1-ε)⁻¹ 的级别，显著提高了在 LWE 对手模型中的区分能力，尤其在维度 n≥500 的实际参数下，误差被压缩至 0.5 以下。

**⚠️ 局限性**

局限性在于需要基的最短向量满足较强下界（λ₁(L)≥kcs√{n/2π}），且在维度较小或 λ₁ 较小的情形下无法直接应用；此外，该改进仍依赖于高斯参数的精细选择，实际应用中需保证 λ₁ 与 s 的比例满足假设。

---

## 605. Filtering Memorization from Parameter-Space in Diffusion Models

**arXiv ID:** 2605.10439 | [PDF](https://arxiv.org/pdf/2605.10439v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 606. WorldReasonBench: Human-Aligned Stress Testing of Video Generators as Future World-State Predictors

**arXiv ID:** 2605.10434 | [PDF](https://arxiv.org/pdf/2605.10434v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 607. Geometrically Approximated Modeling for Emitter-Centric Ray-Triangle Filtering in Arbitrarily Dynamic LiDAR Simulation

**arXiv ID:** 2605.10457 | [PDF](https://arxiv.org/pdf/2605.10457v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 608. Optimal Repair Bandwidth and Repair I/O of $(n,n-2,2)$ MDS Array Codes

**arXiv ID:** 2605.10508 | [PDF](https://arxiv.org/pdf/2605.10508v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 609. SLASH the Sink: Sharpening Structural Attention Inside LLMs

**arXiv ID:** 2605.10503 | [PDF](https://arxiv.org/pdf/2605.10503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 610. OpenSGA: Efficient 3D Scene Graph Alignment in the Open World

**arXiv ID:** 2605.10484 | [PDF](https://arxiv.org/pdf/2605.10484v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 611. Adaptive Context Matters: Towards Provable Multi-Modality Guidance for Super-Resolution

**arXiv ID:** 2605.10470 | [PDF](https://arxiv.org/pdf/2605.10470v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 612. Learning Point Cloud Geometry as a Statistical Manifold: Theory and Practice

**arXiv ID:** 2605.10456 | [PDF](https://arxiv.org/pdf/2605.10456v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 613. QT-Net: Rethinking Evaluation of AI Models in Atomic Chemical Space

**arXiv ID:** 2605.10458 | [PDF](https://arxiv.org/pdf/2605.10458v1)

**作者:** Pablo Martínez Crespo `[一作]` (Chalmers University of Technology), Rocío Mercado `[通讯]` (Chalmers University of Technology)

**通讯引用:** 2610 | [OpenAlex ID](https://openalex.org/A5090993508)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于SOAP聚类的原子级OOD评估协议，并对E(3)-等变模型与非等变、旋转增强模型进行定量比较，随后设计并训练了可直接推断QTA属性的QT‑Net模型。

**💡 创新点**

创新点在于：①首次使用聚类标签限定的原子环境来构造严格的OOD测试集；②通过统计严谨的RM‑ANOVA+Tukey HSD对模型进行比较；③提出非等变密集连接的QT‑Net在OOB原子属性预测上优于等变模型；④将推断的QTA属性作为特征显著提升分子属性预测。

**🔧 技术方法**

使用SOAP特征、HDBSCAN聚类、5×5交叉验证、RM‑ANOVA + Tukey HSD、非等变图神经网络（QT‑Net）以及旋转数据增强等技术。

**📊 数据集**

主要使用AIMEl（QM9子集）中的QTAIM原子属性（N、μ、Q、λ）以及其对应的分子属性（α、Δ、U₀、C_v），并在剩余QM9分子上推断并验证QTA属性。

**📈 对比分析**

通过5×5 CV获得每个模型的CCC评分，使用RM‑ANOVA + Tukey HSD判断显著性，结果显示旋转增强的非等变模型在OOD原子环境上明显优于E(3)-等变模型，且QT‑Net推断的原子属性可恢复分子偶极矩并显著提升下游分子属性预测的R²。

**⚠️ 局限性**

限制在于：OOB测试集采用固定的聚类标签组合，导致模型比较仅针对该划分；无法推广到所有可能的OOD组合；数据量相对有限，模型在不同原子环境上的表现仍有变异；且仅针对H、C、N、O四种元素，未涉及更大化学空间。

---

## 614. Statistical Model Checking of the Keynes+Schumpeter Model: A Transient Sensitivity Analysis of a Macroeconomic ABM

**arXiv ID:** 2605.10447 | [PDF](https://arxiv.org/pdf/2605.10447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 615. A Theory of Multilevel Interactive Equilibrium in NeuroAI

**arXiv ID:** 2605.10505 | [PDF](https://arxiv.org/pdf/2605.10505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 616. SkillEvolver: Skill Learning as a Meta-Skill

**arXiv ID:** 2605.10500 | [PDF](https://arxiv.org/pdf/2605.10500v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 617. M$^2$E-UAV: A Benchmark and Analysis for Onboard Motion-on-Motion Event-Based Tiny UAV Detection

**arXiv ID:** 2605.10496 | [PDF](https://arxiv.org/pdf/2605.10496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 618. TourMart: A Parametric Audit Instrument for Commission Steering in LLM Travel Agents

**arXiv ID:** 2605.10440 | [PDF](https://arxiv.org/pdf/2605.10440v1)

**作者:** Yao Liu `[一作]` (Chengdu University Of Technology), Yao Liu `[通讯]` (Chengdu University Of Technology)

**通讯引用:** 93303 | [OpenAlex ID](https://openalex.org/A5100414285)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并实现了 TourMart，一个针对 LLM 驱动的 OTA 推荐系统的治理审计工具，用来量化佣金驱动的偏差并提供可执行的治理参数；

**💡 创新点**

创新点在于引入可解释的治理调节器 (λ, κ) 与可配置的福利规则，构建对齐的配对反事实实验、三分区治理图谱，并配套六门对话文本的生成器审计，解决了先前监管工具缺失精细度的不足；

**🔧 技术方法**

技术包括基于 GPT 及 Llama 的 LLM 生成的推荐与感知特征、配对反事实评估、McNemar 及场景聚类置换检验、基于规则的六门生成器审计以及多维参数网格的统计可视化；

**📊 数据集**

数据集为完全合成的旅行市场场景（包含 3–5 家酒店、2–3 家航空公司、3–6 位旅客和 6–10 套旅行组合），通过固定种子生成可复现的 900 个场景和 143 对近阈值推荐样本；

**📈 对比分析**

通过与传统的 A/B 测试、暗黑模式检测及 LLM 安全评分对比，TourMart 在 λ=1, κ=0.05 时显著检测到 7.7% 的佣金偏差（Qwen 读者）且在整个 6×6 参数网格中保持统计显著，展示了相较于先前方法更细粒度的偏差估计；

**⚠️ 局限性**

局限包括：仅使用两种旅客 LLM 读者，未覆盖真实用户行为；仿真福利规则与真实决策的泛化能力未知；对低 Discordant 样本的精确检验受限；以及生成器审计门槛对不同 LLM 体系的适用性需要进一步验证。

---

## 619. Beyond Spatial Compression: Interface-Centric Generative States for Open-World 3D Structure

**arXiv ID:** 2605.10438 | [PDF](https://arxiv.org/pdf/2605.10438v1)

**作者:** Xiang Chen `[一作]` (Leipzig University), Alexander Binder `[通讯]` (Leipzig University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种接口中心的3D生成状态（C2LT‑3D），通过将局部几何、组件上下文和接缝关系解耦为可查询、可约束、可修复的离散变量，实现了在解码过程中的组件归属识别、接缝验证和结构修复。

**💡 创新点**

核心创新是将tokenization视为“操作状态”而非单纯的空间压缩；公开的组件归属、接缝有效性等状态变量使得生成过程可以在解码时直接验证、修复和约束，从而显著提升开放世界3D资产的结构鲁棒性。

**🔧 技术方法**

使用了局部canonical chart tokenizer + 代码书量化、无监督分区上下文Transformer、关系接缝头（兼容性、变换细化、碰撞预测）、确定性组件归属实现、以及可选的mesh‑token解码器；训练分为三阶段：几何tokenizer、上下文学习、接缝头微调。

**📊 数据集**

训练集仅为ShapeNet单组件CAD模型（约49k个），零样本评估在Objaverse‑LVIS的1,024个开放世界多组件资产上进行。

**📈 对比分析**

与压缩式tokenizer（BPT、VQ‑Patch）及发布接口（MeshAnythingV2、MeshGPT、LoST）对比。C2LT‑3D在Objaverse‑LVIS上实现：Chamfer下降0.0303→0.0268，Hausdorff下降0.3277→0.2282，污染率下降0.0727→0.0141，分离率提升0.9608→0.9780；在结构修复任务中，Valid@1从0.122→0.667；解码时间约0.033 s/物体。

**⚠️ 局限性**

局限性：仍未保证全局拓扑闭合；对极大规模复杂拓扑或高度交叉部件的精细修复可能需要后处理；受无监督分区误差影响，组件归属精度与分区质量相关。

---

## 620. Separation Logic for Verifying Physical Collisions of CNC Programs

**arXiv ID:** 2605.10437 | [PDF](https://arxiv.org/pdf/2605.10437v1)

**作者:** Yeonseok Lee `[一作]` `[通讯]` (SLING AI Inc.), Yeonseok Lee (SLING AI Inc.)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种将CNC工作空间建模为空间堆的离散化验证框架，并通过Parser‑Prover Handshake实现从连续G‑code到分离逻辑的形式化转换，确定性地验证工具轨迹安全；

**💡 创新点**

将物理碰撞定义为空间数据竞争，剥离连续运动导致的复杂指针算术；利用Minkowski扩展产生安全缓冲，并在多工具协作环境下引入Concurrent Separation Logic，实现可扩展且无需运行时仿真的安全证明；

**🔧 技术方法**

分离逻辑与并发分离逻辑、Bresenham 3D 离散化、Minkowski和空间堆抽象、G‑code 解析器、形式化推理引擎；

**📊 数据集**

本文以合成的一维/二维工作空间案例及典型G‑code示例为验证数据集；

**📈 对比分析**

通过案例演示与传统几何仿真对比，验证过程为确定性、无运行时计算，避免重复测试；未给出量化指标，但强调在离散网格下的可扩展性与数学可证明性优于概率模拟；

**⚠️ 局限性**

仅在离散网格下适用，安全缓冲粗粒度可能覆盖实际误差；未对多轴动态误差、机械延迟等细节建模；缺乏与实际嵌入式控制器的实时集成实现；对复杂几何的手工参数化可能导致可维护性下降。

---

## 621. Regret Minimization in Bilateral Trade With Perturbed Markets

**arXiv ID:** 2605.10475 | [PDF](https://arxiv.org/pdf/2605.10475v1)

**作者:** Anna Lunghi `[一作]` (Politecnico di Milano), Alberto Marchesi `[通讯]` (Politecnico di Milano)

**通讯引用:** 355 | [OpenAlex ID](https://openalex.org/A5039843107)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种在受扰市场中兼顾随机与对抗环境的双边贸易学习算法，在保证全局预算平衡的前提下实现低失调。

**💡 创新点**

首次将价格空间离散化并在无约束与约束两种子问题之间交替进行弱自适应双重更新，以同时处理预算违例与收益最大化，构建了一套无损失的最佳-两世界算法。

**🔧 技术方法**

采用基于梯度的弱自适应双重优化、在线凸优化、重要抽样估计以及网格化价格空间的近似技术。

**📊 数据集**

本文为理论工作，未使用真实数据集，所有结果均来自数学证明和模拟实验。

**📈 对比分析**

在随机环境下获得Õ(T^{3/4})的无退化率，在对抗环境下同样保持Õ(T^{3/4})并对污染度C给出Õ(C log T)的额外调节；与现有最优基准相比，理论性能匹配或优于其上界。

**⚠️ 局限性**

对污染度C的线性依赖、对σ‑光滑性假设的要求，以及log T乘子在对抗情形下的存在，构成了当前方法的主要限制。

---

## 622. Can Muon Fine-tune Adam-Pretrained Models?

**arXiv ID:** 2605.10468 | [PDF](https://arxiv.org/pdf/2605.10468v1)

**作者:** Xingyu Qu `[一作]` (MBZUAI), Samuel Horvath `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在预训练使用 Adam、微调使用 Muon 时产生的优化器不匹配问题，并探讨了通过 LoRA 约束更新来缓解该问题。

**💡 创新点**

首次系统分析并证明了 Adam 与 Muon 的隐式偏置差异导致的结构差异，从而导致微调时性能下降，并展示 LoRA 能显著降低匹配误差。

**🔧 技术方法**

使用 Muon、Adam、LoRA 低秩适配器、LoRA 变体、线性回归理论分析以及对比实验等技术。

**📊 数据集**

在语言任务使用 NanoChat、WikiText‑2、GLUE（T5‑Base）、Llama‑2‑7B（MetaMath、Code‑Feedback、WizardLM 等）；在视觉任务使用 CLIP‑ViT‑B/32 与 StanfordCars、DTD、GTSRB 等数据集。

**📈 对比分析**

通过在相同模型、相同任务上对比全微调与 LoRA，并测量验证 perplexity、准确率、Catastrophic Forgetting 等指标；结果表明 LoRA‑Muon 在大多数任务上可与或超越 LoRA‑Adam，尤其在匹配误差显著时。

**⚠️ 局限性**

仅在较小规模（NanoChat 561 M）进行预训练实验，匹配误差的理论解释仍不完整，LoRA 变体需针对 Muon 重新设计，且对不同任务的匹配严重度差异尚未系统解析。

---

## 623. SlimSpec: Low-Rank Draft LM-Head for Accelerated Speculative Decoding

**arXiv ID:** 2605.10453 | [PDF](https://arxiv.org/pdf/2605.10453v1)

**作者:** Anton Plaksin `[一作]` (Nebius), Alexander Samarin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种低秩参数化的 draft LM-head（SlimSpec），在保留完整词表的前提下压缩隐藏表示以加速推理。

**💡 创新点**

创新点在于将隐藏状态压缩而非词表裁剪，实现4-5倍的 LM-head 加速并保持接收率≈1，避免词表裁剪带来的接受率下降和训练测试不匹配。

**🔧 技术方法**

采用低秩因式分解（W_down、W_up）取 r≈d/8 的 SlimSpec 架构，并在 EAGLE-3 draft 模型上训练。

**📊 数据集**

使用来自 Infinity‑Instruct‑0625 的 660K 提示生成的数据作为训练集，并在 MT‑Bench、HumanEval、GSM8K 上评估。

**📈 对比分析**

与全词表、VocabTrim、SpecVocab 等基线比较，SlimSpec 在多模型、多批量、多温度下获得 8‑9% 更高的端到端 TPS，LM-head 延迟下降 4‑5 倍。

**⚠️ 局限性**

局限包括手动设定秩 r、仅在 EAGLE‑3 上验证、对 vLLM/H200 硬件依赖、未包含 CORAL/DynaSpec 等动态词表方法。

---

## 624. Budget-Efficient Automatic Algorithm Design via Code Graph

**arXiv ID:** 2605.10598 | [PDF](https://arxiv.org/pdf/2605.10598v1)

**作者:** Maxime Bouscary `[一作]` (Massachusetts Institute of Technology), Saurabh Amin `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 6270 | [OpenAlex ID](https://openalex.org/A5112133200)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对预算受限的自动算法设计（AAD），提出了基于有向无环图（DAG）的搜索框架，利用LLM生成可复用的局部纠正（add/replace/remove代码块）而非完整算法，从而显著提高算法产出效率。

**💡 创新点**

创新点在于：①把算法建模为图中的路径；②将LLM输出拆解为可切换、可组合的纠正；③通过纠正级别的信用分配（Shapley值）为后续查询提供结构化反馈；④提出上下文无关与上下文引导两种搜索策略。

**🔧 技术方法**

技术包括：基于LLM（DeepSeek‑V3.2）的纠正生成、随机森林回归预测纠正对适应度的贡献、图结构增量更新与路径枚举、以及自适应评估策略（探索/利用两组候选）。

**📊 数据集**

使用了三类组合优化数据集：旅行商问题（TSP）、位置路由问题（LRP）和无预定义停靠点的公交路由问题（BRP）。

**📈 对比分析**

与传统的全算法生成（EoH）基线对比，图式搜索在相同token预算下取得更高适应度；在TSP上可降低7.5%–22.3%最优性差距，LRP和BRP亦保持更稳健的提升。

**⚠️ 局限性**

主要局限是依赖评估oracle的低成本；若评估代价高昂，增多的评估次数会抵消LLM推理成本的节省。

---

## 625. Thinking with Novel Views: A Systematic Analysis of Generative-Augmented Spatial Intelligence

**arXiv ID:** 2605.10588 | [PDF](https://arxiv.org/pdf/2605.10588v1)

**作者:** Yanbing Zhang `[一作]` (Joy Future Academy), Wenbo Li `[通讯]` (Joy Future Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了“Thinking with Novel Views（TwNV）”闭环框架，将生成式新视角融入空间推理流程：规划器给出摄像机运动指令，生成器渲染新视角，推理器结合原图与新图回答问题，并可迭代验证和细化。

**💡 创新点**

创新点在于：①将生成式视角视作推理媒介而非单纯创作工具；②用精确的6-DoF摄像机参数指令代替自然语言或离散指令，显著提升生成质量；③实现推理时的视觉缩放（多轮生成+验证），使推理效果可按需放大，超越文本自我反思。

**🔧 技术方法**

技术手段包括：大型多模态模型（Gemini‑3‑Flash、GPT‑5、Qwen3‑VL‑235B/32B）作为规划/验证/推理器；多种生成器（GPT‑Image‑1、Nano Banana Pro、Qwen‑Image‑Edit、专门训练的CamCraft）实现视角合成；数值化摄像机参数指令、生成质量评估器与迭代循环。

**📊 数据集**

数据集：3DSRBench（575样本）与RealWorldQA（120样本），共695条含四类空间子任务（Orientation、Location、Size、Multi‑Object）。

**📈 对比分析**

与单视角基线相比，TwNV在四类子任务整体提升1.3–3.9个百分点，最大提升出现在视角敏感的Multi‑Object子任务；在迭代模式下，单轮视觉缩放已达到+3.0个百分点，优于同等调用预算下的文本自反思，表现稳健且可扩展。

**⚠️ 局限性**

局限性包括：①生成器的几何精度仍有限，尤其在大视角变换时会产生误差；②指令错误仍是主要瓶颈，迭代后主要转移到生成质量；③当前仅针对静态、刚体场景，无法处理动态物体或包含、支撑等复杂关系；④规划、生成、验证仍由分离网络完成，手工交互导致误差累积，亟需统一模型整合。

---

## 626. LLM Jaggedness Unlocks Scientific Creativity

**arXiv ID:** 2605.10574 | [PDF](https://arxiv.org/pdf/2605.10574v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 627. Keeping track of errors: A study of SHACL-DS for RDF dataset validation on the ERA RINF Knowledge Graph

**arXiv ID:** 2605.10540 | [PDF](https://arxiv.org/pdf/2605.10540v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 628. Multi-domain Multi-modal Document Classification Benchmark with a Multi-level Taxonomy

**arXiv ID:** 2605.10550 | [PDF](https://arxiv.org/pdf/2605.10550v1)

**作者:** Denghao Ma `[一作]` (Beijing Information Science and Technology University), Zhao Li `[通讯]` (Zhejiang Lab)

**通讯引用:** 51673 | [OpenAlex ID](https://openalex.org/A5008598564)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了MMM-Bench基准，包含5级层级分类、多模态、多域的5,990份真实业务文档。

**💡 创新点**

首次提出兼顾层级标签、多模态与多域的文档分类基准，填补现有基准的空白。

**🔧 技术方法**

利用多模态大模型（Qwen、Claude、GPT等）与API进行零样本和微调实验，评估跨模态和层级学习能力。

**📊 数据集**

采用来自阿里巴巴12个业务域的5,990份人工标注文档，覆盖文本、表格、视觉与布局信息。

**📈 对比分析**

在10+模型上按5级层级和整体HF1指标进行精度/宏F1评估，最高HF1约93%，但在细粒度层级表现显著下降。

**⚠️ 局限性**

仅评估大模型，未包含非大模型；跨模态融合效果不佳；样本分布不均导致尾部类别性能低下。

---

## 629. Where Does Long-Context Supervision Actually Go? Effective-Context Exposure Balancing

**arXiv ID:** 2605.10544 | [PDF](https://arxiv.org/pdf/2605.10544v1)

**作者:** Jinchang Zhu `[一作]` (Hong Kong University of Science and Technology), Menglin Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 125242 | [OpenAlex ID](https://openalex.org/A5100743975)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究长上下文语言模型的监督分配瓶颈，提出 Effective-context Allocation for Context Training（ECAT）在 packed causal 训练中对长有效上下文目标加权，提升长上下文学习效果。

**💡 创新点**

首次将每个目标 token 的同文档有效左侧上下文长度作为监督分配指标，构造对长有效上下文尾部加权，并按逆频率在长尾桶内均衡权重，解决训练信号过度集中于短上下文的瓶颈。

**🔧 技术方法**

使用 packed causal language modeling、文档边界遮蔽、桶化有效上下文长度、逆频率加权、持续预训练（CPT）+ QA‑SFT 微调，以及距离分辨的 evidence‑ablation 探针来验证机制。

**📊 数据集**

以 Qwen2.5 系列与 LLaMA‑3 系列基础模型为起点，训练采用混合长度文档语料库；QA‑SFT 用 Databricks Dolly‑15K instruction‑response；评测使用 NoLiMa（6 公开任务集）、RULER（11 纯检索跟踪任务）以及 MMLU、ARC‑Challenge、HellaSwag、WinoGrande、PIQA、GSM8K。

**📈 对比分析**

与标准 long‑window CPT 进行直接对比，在 7 个 Qwen/LLaMA 训练配置中，ECAT 在所有 trained 与 extrapolated 的 NoLiMa 与 RULER 指标上均实现提升（如 Qwen2.5‑0.5B 4K CPT NoLiMa trained +10.09、extrap +5.34；LLaMA‑3.2‑3B 8K CPT RULER trained +17.91、extrap +16.11），标准 QA 维持 (+0.24 macro)。总体提升长上下文生成性能，未牺牲短上下文能力。

**⚠️ 局限性**

仅在 paired long‑context CPT 环境下验证，未测试从头完整预训练；对早期表示学习与有效上下文分配的交互作用未知；并未探讨更高级结构化监督对长上下文效果的进一步提升。

---

## 630. Bridging Sequence and Graph Structure for Epigenetic Age Prediction

**arXiv ID:** 2605.10541 | [PDF](https://arxiv.org/pdf/2605.10541v1)

**作者:** Yao Li `[一作]` (University of Melbourne), Feng Xia `[通讯]` (RMIT University)

**通讯引用:** 6858 | [OpenAlex ID](https://openalex.org/A5089615958)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一种将共甲基化图结构与位点特异性DNA序列上下文通过轻量化门控调制统一集成的模型，用于预测血液样本的生物年龄。

**💡 创新点**

首次将共甲基化图结构与DNA序列上下文结合，并证明手工统计序列特征优于CNN编码的轻量化门控调制方法，实现更高预测精度。

**🔧 技术方法**

采用图神经网络（PNA）+门控调制+统计序列特征+多层感知机回归，并使用GNNExplainer进行解释性分析。

**📊 数据集**

利用37个公开血液甲基化数据集（Illumina 27K平台），共3707个样本进行训练与测试。

**📈 对比分析**

在相同的测试集（756样本）上与Horvath、AltumAge、DeepMAge、ResnetAge、GraphAge等基线比较，最终模型MAE为3.149年，较最强基线提升12.8%。

**⚠️ 局限性**

仅在27K血液数据上验证，未跨组织或更高密度平台；使用单折交叉验证；序列特征不包含个体变异；解释性分析为相关性而非因果。

---

## 631. UniRank: Unified List-wise Reranking via Confidence-Ordered Denoising

**arXiv ID:** 2605.10527 | [PDF](https://arxiv.org/pdf/2605.10527v1)

**作者:** Pengyue Jia `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 6500 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个统一的列表重排序框架 UniRank，通过自信度排序的迭代去噪方式生成项目排列。

**💡 创新点**

创新点在于将自回归与非自回归两种重排序范式统一到一个去噪框架中，利用双向注意力与候选池限制实现高效的自适应填充。

**🔧 技术方法**

使用离散扩散模型、Semantic Fusion Layer、Latent Pool Selection、双向 Transformer 以及自信度排序的迭代推断。

**📊 数据集**

在 Amazon Books、MovieLens‑1M 以及工业短视频数据集上进行实验。

**📈 对比分析**

与九种基准（G‑only、G‑E 等）对比，UniRank 在 Precision、NDCG、MAP、F1 上均领先，最大提升约为 5.4%（Precision）等，并在线上 A/B 测试中获得多项指标显著提升。

**⚠️ 局限性**

局限性包括：在模型尺寸超过中等水平时易出现过拟合、迭代推断步骤会增加推理延迟，以及对极端候选池规模的适应性仍需进一步验证。

---

## 632. VISTA: A Generative Egocentric Video Framework for Daily Assistance

**arXiv ID:** 2605.10579 | [PDF](https://arxiv.org/pdf/2605.10579v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 633. DuetFair: Coupling Inter- and Intra-Subgroup Robustness for Fair Medical Image Segmentation

**arXiv ID:** 2605.10521 | [PDF](https://arxiv.org/pdf/2605.10521v1)

**作者:** Yiqi Tian `[一作]` (Massachusetts General Hospital and Harvard Medical School), Quanzheng Li `[通讯]` (Massachusetts General Hospital and Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了双轴公平机制DuetFair，并实现FairDRO方法，针对医学图像分割中子组内隐藏失败（intra‑group hidden failure）问题，结合子组条件Mixture‑of‑Experts（dMoE）和子组条件分布鲁棒优化（DRO）实现子组间适配与子组内鲁棒性同时提升。

**💡 创新点**

创新点在于将公平问题拆分为子组间（inter‑group）与子组内（intra‑group）两个轴：①在表征层使用分布感知dMoE实现子组条件专家路由；②在损失层使用子组条件KL‑DRO对高损失样本加权，既提升子组间公平，又减少子组内难样本被平均掩盖的风险。

**🔧 技术方法**

采用dMoE专家路由、分布鲁棒优化（KL‑DRO）、标准ERM、TransUNet、ResUNet/U‑Net、ViT、AdamW等技术；训练中采用bootstrap 95% CI评估指标。

**📊 数据集**

使用Harvard‑FairSeg（眼底分割）、HAM10000（皮肤病变分割）以及内部3D放疗靶区CT数据集，分别以种族、年龄、肿瘤分期和机构等属性划分子组。

**📈 对比分析**

在与TransUNet、ADV、FEBS、FairDiff、MoE、GDRO、Prompt‑GDRO和dMoE等基线同一训练设置下比较；在Harvard‑FairSeg上，FairDRO在最弱种族子组上取得最高ES‑Dice；在HAM10000上表现与GDRO相当；在3D放疗上，FairDRO在肿瘤分期子组上最差组Dice提升3.5点（+6.0%），在机构子组上提升4.1点（+7.4%），并实现最高ES‑Dice。

**⚠️ 局限性**

局限性：①当预定义子组标签已能解释大部分性能差异时，额外的子组内鲁棒收益有限；②方法依赖子组属性，在属性缺失或不可靠场景下需要进一步研究；③未探索多属性连续/层次化条件或无监督子组发现等方向。

---

## 634. SenseBench: A Benchmark for Remote Sensing Low-Level Visual Perception and Description in Large Vision-Language Models

**arXiv ID:** 2605.10576 | [PDF](https://arxiv.org/pdf/2605.10576v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 635. It's All Connected: Topology-Aware Structural Graph Encoding Improves Performance on Polymer Prediction

**arXiv ID:** 2605.10551 | [PDF](https://arxiv.org/pdf/2605.10551v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 636. Acceptance Cards:A Four-Diagnostic Standard for Safe Fine-Tuning Defense Claims

**arXiv ID:** 2605.10575 | [PDF](https://arxiv.org/pdf/2605.10575v1)

**作者:** Phongsakon Mark Konrad `[一作]` (University of Southern Denmark), Serkan Ayvaz `[通讯]` (University of Southern Denmark)

**通讯引用:** 1452 | [OpenAlex ID](https://openalex.org/A5075890706)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证了一套称为Acceptance Card的评估协议，用以严格评判安全微调防御的安装差距减少效果，并对Gemma‑2‑2B‑it及其他模型在多任务上进行系统审计。

**💡 创新点**

提出了四个独立诊断门（统计可靠性、语义新鲜度、机制一致性和跨任务转移）和一套文档化标准，能够将防御声明分层验证而非单一分数。

**🔧 技术方法**

结合问卷聚类自助抽样、基于MMLU的语义重评估、参数空间投影签名ρ_AT以及交叉任务转移检查，配合LoRA、AdamW等微调技术。

**📊 数据集**

主要使用Gemma‑2‑2B‑it的沙袋任务、Sycophancy、Refusal等子任务以及Qwen2.5‑1.5B‑Instruct、Llama‑3‑8B、Phi‑3‑mini的跨架构验证。

**📈 对比分析**

以四门严格阈值合成的全卡通过/未通过作为评判，结果46个评估单元全部未通过全卡，其中AC‑AdamW仅在统计可靠性与机制一致性上通过，SafeLoRA在所有门上均未通过；部署准确度往往伴随较大成本。

**⚠️ 局限性**

审计仅覆盖Gemma‑2‑2B，跨架构与跨任务测试有限；跨任务门在基线陷阱时不可评估；ρ_AT为点估计，未考虑梯度批量波动。

---

## 637. PhysEDA: Physics-Aware Learning Framework for Efficient EDA With Manhattan Distance Decay

**arXiv ID:** 2605.10547 | [PDF](https://arxiv.org/pdf/2605.10547v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 638. TIE: Time Interval Encoding for Video Generation over Events

**arXiv ID:** 2605.10543 | [PDF](https://arxiv.org/pdf/2605.10543v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 639. PrimeKG-CL: A Continual Graph Learning Benchmark on Evolving Biomedical Knowledge Graphs

**arXiv ID:** 2605.10529 | [PDF](https://arxiv.org/pdf/2605.10529v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 640. Mela: Test-Time Memory Consolidation based on Transformation Hypothesis

**arXiv ID:** 2605.10537 | [PDF](https://arxiv.org/pdf/2605.10537v1)

**作者:** Lungchuan Chen `[一作]` `[通讯]` (MusubiAI), Lungchuan Chen (MusubiAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在测试时进行系统级记忆巩固的层次化记忆模块 HMM，并将其集成到语言模型 Mela 中。

**💡 创新点**

创新点在于将跨频率耦合与转化假说相结合，设计出双子记忆子模块及其递归交互，以及 MemStack 方案让解码器同时访问不同巩固阶段的记忆。

**🔧 技术方法**

采用 Transformer 结构、神经记忆模块、门控注意力、Newton–Schulz 正交化以及分层隐式递归等技术。

**📊 数据集**

使用 FineWeb‑Edu 语料库约 5 B 个 token 进行预训练，测试时使用相同数据集的 held‑out 子集。

**📈 对比分析**

与匹配规模的 Transformer++ 基线相比，Mela 在 4K 预训练窗口下即更优，在超过 4K 的长序列上保持低 perplexity，证明了更好的长上下文泛化。

**⚠️ 局限性**

局限性包括仅在语言建模任务上验证，未探讨多模态或更大规模；递归过程增加了推理延迟；对超大上下文仍需进一步优化。

---

## 641. GemDepth: Geometry-Embedded Features for 3D-Consistent Video Depth

**arXiv ID:** 2605.10525 | [PDF](https://arxiv.org/pdf/2605.10525v1)

**作者:** Yuecheng LiulJunda Cheng `[一作]`, Xin Yang `[通讯]` (Huazhong University Of Science & Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 GemDepth 框架，通过 Geometry-Embedding Module（GEM）和 Alternating Spatio-Temporal Transformer（ASTT）实现视频深度估计的 3D 时空一致性；

**💡 创新点**

创新点在于显式注入相机运动与全局 3D 结构的几何嵌入，利用 GEM 产生姿态先验并通过 ASTT 交替进行时空注意力，从而在保持细节的同时抑制闪烁；

**🔧 技术方法**

使用 Transformer 结构（4 层交替注意力网络）、RoPE 位置编码、EfficientPoseNet、MLP 嵌入以及多尺度梯度匹配损失；

**📊 数据集**

训练数据包括 Virtual KITTI 2、TartanAir、PointOdyssey、MVS‑Synth、Dynamic Replica（~690k 帧）用于姿态优化，IRS 及野外视频（~250k 帧）用于深度微调；评估基准为 KITTI、Sintel、Scannet、Bonn；

**📈 对比分析**

与 NVDS、ChronoDepth、DepthCrafter、RollingDepth、DepthAnythingV2、VideoDepthAnything 等方法对比，GemDepth 在 AbsRel、δ1、TAE、F1 等指标上均超过同类 SOTA，尤其在高动态场景下的时空一致性提升显著；

**⚠️ 局限性**

局限性在于对姿态估计的依赖，姿态噪声超过 50% 时性能明显下降；同时在极端光照或稀疏纹理场景下的几何推断仍需进一步提升。

---

## 642. Online Resource Allocation With General Constraints

**arXiv ID:** 2605.10519 | [PDF](https://arxiv.org/pdf/2605.10519v1)

**作者:** Eleonora Fidelia Chiefari `[一作]` (Politecnico di Milano), Alberto Marchesi `[通讯]` (Politecnico di Milano)

**通讯引用:** 355 | [OpenAlex ID](https://openalex.org/A5039843107)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种在同时包含预算约束和一般长时约束的通用在线资源分配框架下，能够在随机和对抗环境中实现最佳两种世界（best‑of‑both‑worlds）性能的算法。

**💡 创新点**

创新点主要包括：① 将预算约束与一般约束统一考虑，克服了传统 ORA 仅关注预算的局限；② 在未知 Slater 参数的情况下，利用弱自适应性证明拉格朗日乘子随时间保持有界；③ 采用不需要投影到已知上界的在线梯度下降（OGD）更新双重变量；④ 在对抗环境中实现了 α‑regret 量级为 𝑂(√T) 的性能。

**🔧 技术方法**

技术手段包括：拉格朗日双重优化框架、在线梯度下降（OGD）、弱自适应性（weak‑adaptivity）分析、马尔可夫/尾界（martingale inequality）以及对照性/弱可行性证明。

**📊 数据集**

本文未使用任何真实数据集，所有结果均为理论证明与上界分析。

**📈 对比分析**

在随机环境下，累计奖励相对于动态最优实现了 𝑂(√T) 的期望回报；在对抗环境下，α‑regret 同样为 𝑂(√T)。同时，预算约束被严格满足，一般约束的累计违规量亦为 𝑂(√T)。相较于仅考虑预算约束的传统算法，本方法在多种约束情形下保持了同等甚至更优的子线性误差。

**⚠️ 局限性**

局限性包括：① 仅适用于完全可观测的输入模型，未考虑延迟或部分信息；② 对非平稳（变化）环境的性能尚未进一步改进；③ 需要事先知道每轮预算 β_j；④ 尽管通过弱自适应性避免了显式投影，但拉格朗日乘子上界仍依赖 Slater 参数 ρ，若 ρ 很小则上界可能不够紧。

---

## 643. ICT-NLP at SemEval-2026 Task 3: Less Is More -- Multilingual Encoder with Joint Training and Adaptive Ensemble for Dimensional Aspect Sentiment Regression

**arXiv ID:** 2605.10560 | [PDF](https://arxiv.org/pdf/2605.10560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 644. A Resilient Solution for Sewer Overflow Monitoring across Cloud and Edge

**arXiv ID:** 2605.10592 | [PDF](https://arxiv.org/pdf/2605.10592v1)

**作者:** Vipin Singh `[一作]` (Berlin University of Applied Sciences), Felix Biessmann `[通讯]` (Berlin University of Applied Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个基于云端与边缘端深度学习模型的Web演示仪表板，用于预测和监控混合污水系统的溢流风险。

**💡 创新点**

创新点在于将云端长时预测与边缘端短时预测整合到同一交互式平台，并提供网络中断时的容错预警与风险评估。

**🔧 技术方法**

使用了Temporal Fusion Transformer、LSTM以及梯度提升决策树等深度学习与机器学习技术进行填充水平预测和风险评估。

**📊 数据集**

利用德国杜伊斯堡市政污水处理厂的三年小时级传感器数据（35个传感器），包含溢流盆地填充、降雨、泵能耗等信息。

**📈 对比分析**

通过在相同时间窗口下比较云端和边缘端模型的预测误差（MSE）与两小时溢流风险指示，结果显示云端模型在12小时预测中误差更低，而边缘端模型在一小时预测中仍保持可接受的精度。

**⚠️ 局限性**

局限性包括目前仅为预先计算的数据演示，缺乏实时数据集成，且对异常值或缺失值的鲁棒性尚未充分验证。

---

## 645. Higher Resolution, Better Generalization: Unlocking Visual Scaling in Deep Reinforcement Learning

**arXiv ID:** 2605.10546 | [PDF](https://arxiv.org/pdf/2605.10546v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 646. Controllability in preference-conditioned multi-objective reinforcement learning

**arXiv ID:** 2605.10585 | [PDF](https://arxiv.org/pdf/2605.10585v1)

**作者:** Pau de las Heras Molins `[一作]` (LTCI, Télécom Paris, Institut Polytechnique de Paris), Georgios Bakirtzis `[通讯]` (LTCI, Télécom Paris, Institut Polytechnique de Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了基于秩相关的可控性度量，用于评估多目标强化学习（MORL）中偏好条件化策略的可控性，并在复杂环境（Tetris、Snake、MOBA）上与传统评估指标比较；同时发布了高通量框架 PufferMO。

**💡 创新点**

①首次将秩相关系数（Spearman/Kendall）引入 MORL 的可控性评估；②揭示传统指标（超体积、稀疏度、期望效用、余弦相似度）无法准确衡量偏好条件化的可控性；③提供了可复现的高通量实验平台。

**🔧 技术方法**

多目标近端策略优化（MOPPO）与线性偏好权重条件化；秩相关系数、超体积、稀疏度、期望效用、余弦相似度等评估指标；PufferLib 与 PufferMO 作为实验框架。

**📊 数据集**

多目标版本的 Tetris、Snake、MOBA（均来自 PufferLib），包含多维奖励（如得分、体型、经验、死亡等）。

**📈 对比分析**

通过对比 PPO、无条件 MOPPO 与权重条件化 MOPPO，使用 HV、SP、EU、CS 及每目标 Spearman ρ 进行评估。结果显示传统指标无法区分可控与不可控策略；Spearman ρ 在可控策略中显著为正，体现可控性；整体任务性能与基线相当或略有提升。

**⚠️ 局限性**

①秩相关假设偏好与返回单调对应，仅适用于线性权重；②不考虑非线性或层级偏好；③对高维目标空间的鲁棒性有限；④实验仅覆盖三种游戏，未检验更广泛场景；⑤指标不捕捉偏好与返回之间的幅度比例。

---

## 647. HH-SAE: Discovering and Steering Hierarchical Knowledge of Complex Manifolds

**arXiv ID:** 2605.10536 | [PDF](https://arxiv.org/pdf/2605.10536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 648. Effect of Graph Gluing on Consensus in Networked Multi-Agent Systems

**arXiv ID:** 2605.10558 | [PDF](https://arxiv.org/pdf/2605.10558v1)

**作者:** Rohollah Moghadam `[一作]` (California State University-Sacramento), Santosh Kandel `[通讯]` (California State University-Sacramento)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在多智能体系统中通过图拼接（桥接与界面拼接）操作对网络拓扑和收敛性能的影响，给出了代数连通度（Fiedler 典型值）的理论上界并通过仿真验证。

**💡 创新点**

创新点在于提出并分析了两类拼接操作（桥接和界面拼接）对拉普拉斯矩阵谱的显式影响，给出了λ₂的上界表达式，并揭示了拼接边数与收敛速率的关系；同时将这些理论与仿真结果相结合，首次系统性地量化了拼接策略对多智能体系统协同收敛的性能提升。

**🔧 技术方法**

主要技术包括：图论基本概念、拉普拉斯矩阵谱分析、Fiedler 典型值的上界推导、Poincaré 最小-最大原理、以及单积分动力学的分布式平均一致性控制器。通过构造桥接和界面拼接的数学模型，计算相应拉普拉斯矩阵并求解其第二小特征值。

**📊 数据集**

没有使用公开数据集，所有实验均基于仿真生成的自定义小规模（3-6 节点）和中等规模（10-12 节点）图网络。

**📈 对比分析**

通过将拼接前后的Fiedler 典型值与理论上界进行比较，证明了边数越多 λ₂ 越大，收敛时间越短。仿真结果显示，在桥接边数从1到3时，收敛时间分别约为 2.6 s、1.5 s，验证了理论预期。

**⚠️ 局限性**

局限性包括：仅针对无向、单积分动力学的多智能体系统；对有向图或更一般动力学的推广需要进一步研究；拼接过程中未考虑通信延迟、噪声或攻击对性能的实际影响；理论上界虽有实际意义，但在大规模网络中可能显著偏离实际值。

---

## 649. Agent-First Tool API: A Semantic Interface Paradigm for Enterprise AI Agent Systems

**arXiv ID:** 2605.10555 | [PDF](https://arxiv.org/pdf/2605.10555v1)

**作者:** Kai Pan `[一作]` `[通讯]` (A2A Lab), Kai Pan (A2A Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了Agent-First Tool API设计范式，将传统 CRUD 接口改造成面向 LLM 代理的工具接口，包含六动词语义协议、标准化工具合同（NTC）以及双层治理管线，支持多租户 SaaS 环境下的自主代理执行。

**💡 创新点**

创新点包括：①将工具调用视为目标实现协议（S‑R‑P‑E‑V‑C），打破 CRUD 的单次请求假设；②引入 NTC，统一返回 confidence、evidence、next_actions 等决策支持元数据；③构建双层权限（功能级 + 对象级）与动态风险评估、内置审批流程，解决代理权限与风险控制难题；④在同一后端实现 CRUD 与 Agent 接口共存，保持业务逻辑一致。

**🔧 技术方法**

采用技术：LLM 代理（MiniMax‑M2.7）+ ReAct 思考框架；Django 4.x/DRF 后端；Celery + Redis 异步任务与审批；PostgreSQL + pgvector 语义检索；Server‑Sent Events（SSE）流式响应；JSON Schema 校验；双层权限模型；事务性 idempotency key 保障；与 MCP（Model Context Protocol）兼容的工具发现与调用。

**📊 数据集**

使用的数据集：内部 SaaS 生产数据，85 个工具、6 个业务域；50 个自然语言任务用于对比实验；1,247 次写/提交调用用于风险评估；工具与任务覆盖范围基于真实业务场景（工单、库存、品牌配置等），未使用公开数据集。

**📈 对比分析**

比较方法：在相同 LLM 与 Prompt 设置下，执行 50 任务，分别使用 CRUD+ReAct 与 Agent‑First；评估指标包括任务成功率、ID hallucination、错误恢复、人工干预、API 调用次数、延迟和 token 消耗。结果显示：任务成功率从 64% 提升至 88%（+37.5%）；ID hallucination 28%→4%；错误恢复 12.5%→72.7%（5.8×）；人工干预 22%→6%；API 调用次数 4.8→3.2；延迟 3.1→4.6s（+48%）；token 消耗 1,840→2,520（+36.9%）。

**⚠️ 局限性**

局限性：①六动词协议通过字段约束实现，缺乏强制型接口定义；②NTC 的 confidence 由工具作者手工设定，未学习调优；③评估仅在单一生产系统内，缺乏跨技术栈验证；④维护两套 API（CRUD 与 Agent）增加工程成本；⑤当前仅支持同步请求，异步长耗时工具需进一步扩展；⑥安全与治理规则基于静态配置，需进一步强化动态风险学习。

---

## 650. FrequencyCT: Frequency domain pseudo-label generation for self-supervised low-dose CT denoising

**arXiv ID:** 2605.10583 | [PDF](https://arxiv.org/pdf/2605.10583v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 651. Personalized Deep Research: A User-Centric Framework, Dataset, and Hybrid Evaluation for Knowledge Discovery

**arXiv ID:** 2605.10530 | [PDF](https://arxiv.org/pdf/2605.10530v1)

**作者:** Xiaopeng Li `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 6500 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了个性化深度研究框架（PDR），通过在整个检索‑推理循环中整合用户上下文，实现更符合用户需求的研究报告生成。

**💡 创新点**

创新点在于将用户画像建模与查询生成、双阶段检索、上下文感知生成紧密耦合，并首次公开了基于真实场景的 PDR 数据集和混合评测框架。

**🔧 技术方法**

采用的大模型包括 DeepSeek R1‑671B 进行推理和评判，BGE‑M3 作为向量嵌入，Milvus 作为私有检索库，并结合外部 Wikipedia‑18 知识库。

**📊 数据集**

使用的数据集为 PDR Dataset，包含四类任务（个性化摘要、主题撰写、报告生成、演讲稿生成）及对应的用户历史文档和真实报告。

**📈 对比分析**

与零射击、+Search、Profile Prompting、Iterative RAG 以及四大商业深度研究系统（Grok、Perplexity、Gemini、OpenAI Deep Research）对比，PDR 在个性化评分（C.P., P.P.）上显著优于对手，同时保持相近或更高的语义一致性与质量。

**⚠️ 局限性**

限制在于当前仅在公开数据集上验证，缺乏跨域与动态用户反馈的长期评估，且对多模态私人资料的处理仍有改进空间。

---

## 652. CrackMeBench: Binary Reverse Engineering for Agents

**arXiv ID:** 2605.10597 | [PDF](https://arxiv.org/pdf/2605.10597v1)

**作者:** Isaac David `[一作]` (University College London), Arthur Gervais `[通讯]` (University College London)

**通讯引用:** 7328 | [OpenAlex ID](https://openalex.org/A5063253761)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CrackMeBench v0，一个基于可执行文件验证的可重复实验基准，要求模型在无网络 Docker 沙箱中利用公开工具逆向分析并提交能被目标程序接受的密码、密钥或工件。

**💡 创新点**

创新点在于将 CrackMe 验证任务定义为可执行文件 oracle 评分标准，提供公开校准任务与自动生成主分任务，完整公开工具清单与命令日志保证可复现与可对比，且聚焦单一、可验证的逆向目标，避免了传统 CTF 或源代码测试的模糊性。

**🔧 技术方法**

使用多种逆向工具（radare2、Ghidra、angr、Z3、capstone 等）以及标准的 Linux 容器化环境，结合 LLM 代理交互与命令执行接口实现任务执行。

**📊 数据集**

数据集包含 8 个公开校准 CrackMe（P01–P08）和 12 个自动生成主分任务（S01–S12），每个任务附带可执行文件、元数据、工具清单以及隐藏 oracle。

**📈 对比分析**

对比方法以 pass@1、pass@3、平均耗时、token 计数等指标评估 GPT‑5.5、Claude Opus 4.7、Kimi K2 三大模型；结果显示 GPT‑5.5 在生成任务上 92% 通过，Claude 58%，Kimi 42%；在公开校准任务上三者均低于 40%。

**⚠️ 局限性**

局限性包括：仅针对教育级 CrackMe，无法覆盖更复杂的商业破解或攻击场景；5 分钟预算与三次提交限制可能导致可解任务被误判为失败；对工具可用性与网络隔离的假设限制了跨平台迁移与现实环境的适用性。

---

## 653. CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations

**arXiv ID:** 2605.10586 | [PDF](https://arxiv.org/pdf/2605.10586v1)

**作者:** Nengbo Lu `[一作]` (Guilin University of Electronic Technology), Minghua Pan `[通讯]` (Guilin University of Electronic Technology)

**通讯引用:** 76 | [OpenAlex ID](https://openalex.org/A5052085005)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 CausalGS 框架，利用多视角视频在不使用任何先验（如物体掩码、类别标签）的条件下，逆向推断场景的初始速度场和材料属性，并通过可微 MPM 仿真器生成物理驱动的运动，从而实现对动态 3D 场景的未来帧外推与新视角插值。

**💡 创新点**

创新点在于：① 将 3D Gaussian Splatting 作为可微粒子表示，同时赋予每个 Gaussian 可学习的初始动力学和材料参数；② 设计了“物理代码”与瓶颈解码的分离式速度场，使运动模式可被结构化为线性组合；③ 通过逆向物理推断模块直接从视频中学习物理因子，实现无监督的物理因果关系学习；④ 结合可微 MPM 仿真进行自监督训练，使模型能够捕获复杂的材料交互和接触动力学。

**🔧 技术方法**

核心技术包括：3D Gaussian Splatting（3DGS）作为几何和外观表示；神经速度网络（物理代码）与瓶颈网络、时间加权网络构成的速度场；材料解码器输出 Young's 模量、泊松比等 MPM 所需参数；可微材料点法（MPM）仿真器用于前向动力学；差分渲染与基于视角的损失实现端到端优化。

**📊 数据集**

使用的数据集包括：合成数据集——Dynamic Object Dataset、Dynamic Indoor Scene Dataset；真实数据集——NVIDIA Dynamic Scene（12 摄像机）与 FreeGave‑GoPro（20 摄像机）。所有实验均采用 89 帧序列、960×540 分辨率，使用前 67 帧训练，后 22 帧做未来帧外推评估。

**📈 对比分析**

与多种基线（T‑NeRF、D‑NeRF、NSFF、TiNeuVox、DefGS、FreeGave、NVFi 等）以及对比方法（如 FreeGave 的速度场设计）进行比较。CausalGS 在所有四个数据集上均取得最高 PSNR、SSIM，LPIPS 最低，尤其在未来帧外推任务中显著优于现有 SOTA，表现出更逼真的运动预测和更优的插值质量。

**⚠️ 局限性**

局限性包括：① 对 3D Gaussian 的粒子化表示假设适用性，极大复杂或高频细节场景可能仍需更细粒度建模；② 计算量相对较大，MPM 前向仿真和多视角渲染耗时；③ 目前仅处理弹性/不可压缩材料，未覆盖流体、黏性或多相交互；④ 对摄像机视角覆盖度和光照变化仍有一定敏感性。

---

## 654. Guaranteed Jailbreaking Defense via Disrupt-and-Rectify Smoothing

**arXiv ID:** 2605.10582 | [PDF](https://arxiv.org/pdf/2605.10582v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 655. Polygon-mamba: Retinal vessel segmentation using polygon scanning mamba and space-frequency collaborative attention

**arXiv ID:** 2605.10581 | [PDF](https://arxiv.org/pdf/2605.10581v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 656. Online Sharp-Calibrated Bayesian Optimization

**arXiv ID:** 2605.10572 | [PDF](https://arxiv.org/pdf/2605.10572v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 657. VeloGauss: Learning Physically Consistent Gaussian Velocity Fields from Videos

**arXiv ID:** 2605.10567 | [PDF](https://arxiv.org/pdf/2605.10567v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 658. ConfoundingSHAP: Quantifying confounding strength in causal inference

**arXiv ID:** 2605.10533 | [PDF](https://arxiv.org/pdf/2605.10533v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 659. A Reflective Storytelling Agent for Older Adults: Integrating Argumentation Schemes and Argument Mining in LLM-Based Personalised Narratives

**arXiv ID:** 2605.10531 | [PDF](https://arxiv.org/pdf/2605.10531v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 660. ThreatCore: A Benchmark for Explicit and Implicit Threat Detection

**arXiv ID:** 2605.10563 | [PDF](https://arxiv.org/pdf/2605.10563v1)

**作者:** Davide Bruni `[一作]` (University of Pisa), Maurizio Tesconi `[通讯]` (National Research Council)

**通讯引用:** 5071 | [OpenAlex ID](https://openalex.org/A5066142255)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了ThreatCore数据集，重新标注并合并多来源数据，加入合成示例，评估多种模型进行细粒度威胁检测。

**💡 创新点**

提出统一的威胁定义并区分明确威胁、隐含威胁与非威胁，构建完整基准，揭示隐含威胁检测难点，并证明语义角色标注可提升性能。

**🔧 技术方法**

采用双标注协议重新标注、LLM生成合成样本、Perspective API、零样本分类器（ModernBERT）、大语言模型（GPT‑4o‑mini、gpt‑oss:20b、Phi4:14b等）以及语义角色标注（SRL）作为中间表示。

**📊 数据集**

重新标注的公开数据集（Jigsaw、Latent Hatred、DynHate、Gab Hate Corpus、ETHOS等）共15,691条，加上约6,000条人工验证的合成威胁，最终ThreatCore总计21,764条样本。

**📈 对比分析**

使用精确度、召回率、F1和宏F1进行对比；Perspective API高精确但召回低，尤其对隐含威胁；零样本模型倾向过度预测威胁；GPT‑oss:20b宏F1最高；Phi4:14b+SRL取得最佳宏F1≈0.773，显著提升隐含威胁检测。

**⚠️ 局限性**

仅涵盖英语数据，合成样本可能存在偏差，模型评估受算力限制，SRL效果高度依赖模型，Azure安全策略限制部分模型评测，未覆盖多语言和真实世界多样性。

---

## 661. List-Decodable Folded Quantum Hermitian Codes

**arXiv ID:** 2605.10534 | [PDF](https://arxiv.org/pdf/2605.10534v1)

**作者:** Gretchen L. Matthews `[一作]` (Virginia Tech), Julia Shapiro `[通讯]` (Virginia Tech)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5110715388)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了折叠量子 Hermitian 码的构造，并证明其具备接近量子 Singleton 极限的列表可译性质。

**💡 创新点**

创新点在于：①利用 Hermitian 曲线的自同构实现折叠，获得更小的符号域；②不需要量子距离放大技术即可达到最佳错误容忍度；③给出了明确的列表解码算法与复杂度分析。

**🔧 技术方法**

采用了 CSS 结构、代数几何（Hermitian 曲线）与折叠技术、以及量子列表可译框架；同时引入了纠缠辅助构造来放宽码对的包含条件。

**📊 数据集**

无实验数据集，全部为理论构造与解析证明。

**📈 对比分析**

与折叠 Reed–Solomon 码比较，折叠 Hermitian 码在相同错误容忍度下符号域更小（量子距离放大不必），且保持相同或更优的列表大小与复杂度，证明了在量子 Singleton 限制下的最优性。

**⚠️ 局限性**

目前仅针对 Hermitian 曲线；对更一般正曲率 AG 曲线的折叠量子码构造尚未完成，且在更低速率或更小 alphabet 的参数空间中尚未达到最优；开放问题包括进一步降低 alphabet 大小、扩展到其他曲线以及研究更高效的解码算法。

---

## 662. The Balance between Nuance and Clarity: Decluttering Tabular Sequential Graphs to Counter Money Laundering

**arXiv ID:** 2605.10522 | [PDF](https://arxiv.org/pdf/2605.10522v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 663. Infinite Mask Diffusion for Few-Step Distillation

**arXiv ID:** 2605.10518 | [PDF](https://arxiv.org/pdf/2605.10518v1)

**作者:** Jaehoon Yoo `[一作]` (Korea Advanced Institute of Science and Technology), Seunghoon Hong `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 8657 | [OpenAlex ID](https://openalex.org/A5077461583)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 Infinite Mask Diffusion Model (IMDM) 的新型扩散模型，用来解决传统 Masked Diffusion Models (MDMs) 在少步生成时固有的分解误差下界问题。

**💡 创新点**

创新点在于将确定性单态掩码改为随机无限状态掩码，使得模型在保持 MDM 简单易用性的同时消除了分解误差的理论下界，并能更好地捕捉 token 依赖关系。

**🔧 技术方法**

主要技术包括：扩散模型框架、随机无限状态掩码设计、无监督去噪扩散蒸馏（SDTT）与 ReDi 蒸馏技术、噪声维度和尺度的精细调优。

**📊 数据集**

在 LM1B 与 OpenWebText 两大语言数据集上进行实验，使用 SDTT+ReDi 蒸馏进行对比。

**📈 对比分析**

与传统 MDM 及现有少步蒸馏方法相比，IMDM 在 1~8 步生成任务中显著降低了生成困惑度（PPL），在 2 步时差距可达 120% 以上；同时保持与原始模型相近的文本质量。

**⚠️ 局限性**

局限性包括：需要较高的噪声维度（如 d_noise=768 或 2048）才能发挥优势；对噪声分布和尺度的敏感度仍有限；目前仅在标准语言建模任务上验证，缺乏对多模态或更大规模模型的泛化评估。

---

## 664. Consistency as a Testable Property: Statistical Methods to Evaluate AI Agent Reliability

**arXiv ID:** 2605.10516 | [PDF](https://arxiv.org/pdf/2605.10516v1)

**作者:** Harsh Raj `[一作]` (Northeastern University), Subhabrata Majumdar `[通讯]` (Indian Institute of Management Bangalore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种用于评估大型语言模型驱动代理在语义保持扰动下的一致性和可靠性的测量框架；

**💡 创新点**

通过U-统计量和轨迹级MMD方法，将输出一致性与轨迹级一致性分离，提供可解释的分解指标；

**🔧 技术方法**

利用U-统计量、最大均值散度（MMD）、核方法（如JSD、GAK）和加权Levenshtein距离进行一致性评估；

**📊 数据集**

在SWE‑bench Verified、Spider2‑DBT和BFCL三个代理基准上进行实验；

**📈 对比分析**

与传统pass@1率对比，轨迹一致性指标在检测细粒度错误方面更敏感，能揭示功能选择和执行顺序的失效；

**⚠️ 局限性**

假设扰动保持语义不变且分布不变，需人工设计扰动，且目前仅评估轨迹级一致性，未考虑内部推理状态的可靠性。

---

## 665. Weight distributions of cosets of weight 2 of the generalized doubly extended Reed-Solomon codes

**arXiv ID:** 2605.10594 | [PDF](https://arxiv.org/pdf/2605.10594v1)

**作者:** Alexander A. Davydov `[一作]`, Fernanda Pambianco `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了广义双扩展Reed–Solomon码（GDRS）中权重为2的余类的权重分布，提出并证明了当 q-1 与 d-2 互素时所有此类余类的分布相同（Case S），从而解答了此前的开放问题；并将该问题与两类新的组合计数问题（A_q,μ^× 与 A_ℜ,μ^+）关联，利用环 ℤ_ℜ 的轨道方法在许多 (q,d) 组合上求解了权重分布。

**💡 创新点**

① 用互素条件给出了 Case S 的充分条件，首次完全确定了权重为 2 的余类的分布；② 将权重分布问题转化为组合计数问题，提出 A_q,μ^× 与 A_ℜ,μ^+ 两类新问题；③ 通过轨道方法和群作用求解了大量实例，提供了通用计算框架。

**🔧 技术方法**

使用了组合计数（轨道与同构类）技术、Bonneau 公式、线性代数中的 Vandermonde 行列式、有限域与整数模环的乘法与加法结构，结合 MDS 码和射影几何（正交曲线）的性质。

**📊 数据集**

没有使用传统意义上的数据集；研究基于代数结构（有限域 𝔽_q、整数模环 ℤ_ℜ）和 GDRS 码本身。

**📈 对比分析**

研究不涉及实验对比，主要是理论推导与符号计算。通过证明可得权重分布公式，验证了在互素条件下的整数性，表明 2-正则性和重量分布的一致性。

**⚠️ 局限性**

仅在许多 (q,d) 组合上求得完整结果；对于一般 (q,d)（尤其是 q-1 与 d-2 不互素的情况）仍未给出完整解答；此外，方法依赖于对 μ、ℜ 的轨道划分，计算量随参数增大而迅速增长。

---

## 666. LLARS: Enabling Domain Expert & Developer Collaboration for LLM Prompting, Generation and Evaluation

**arXiv ID:** 2605.10593 | [PDF](https://arxiv.org/pdf/2605.10593v1)

**作者:** Philipp Steigerwald `[一作]` (Technische Hochschule Nürnberg Georg Simon Ohm), Jens Albrecht `[通讯]` (Technische Hochschule Nürnberg Georg Simon Ohm)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 Llars 的开源平台，实现了协同提示工程、批量生成和混合评估的端到端工作流。

**💡 创新点**

整合了三大模块在单一平台，支持实时多用户提示编辑、可配置的多模型批量生成以及人机共同评估并提供一致性统计和 provenance 分析，填补了专业领域专家与开发者之间的协作缺口。

**🔧 技术方法**

基于 Web 容器化应用，采用实时协同编辑、版本控制、LLM 接口调用、批量任务调度、JSON/CSV 导出、交互式评估仪表盘、Krippendorff α 等统计；使用多模型 API（如 GPT 系列、Claude、Llama 等）与人类评估员。

**📊 数据集**

使用在线心理咨询的电子邮件对话数据（50 条），以及 11 种 LLM（包括 GPT‑4、Claude、Llama 等）生成的 253 条标题，亦测试了 11 模型×2 提示的 200 条输出。

**📈 对比分析**

通过与现有工具（Agenta、Phoenix、ChainForge、Label Studio、LangSmith 等）的功能对比表和内部案例研究（评估 1,518 次评分），验证了 Llars 在协作性、成本控制、评估一致性和最佳组合发现方面优于传统单独工具，且节省了时间。

**⚠️ 局限性**

目前仅支持单轮生成，未涵盖多轮对话；LLM 评估器的判断受底层模型推理能力限制；尚缺乏自动化校准和闭环微调接口。

---

## 667. Deep Arguing

**arXiv ID:** 2605.10569 | [PDF](https://arxiv.org/pdf/2605.10569v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 668. DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving

**arXiv ID:** 2605.10564 | [PDF](https://arxiv.org/pdf/2605.10564v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 669. Correct-by-Construction G-Code Generation: A Neuro-Symbolic Approach via Separation Logic

**arXiv ID:** 2605.10568 | [PDF](https://arxiv.org/pdf/2605.10568v1)

**作者:** Yeonseok Lee `[一作]` `[通讯]` (SLING AI), Yeonseok Lee (SLING AI)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种神经符号框架，将大型语言模型生成的G‑码与分离逻辑验证器结合，实现自动校正的安全加工路径。

**💡 创新点**

创新点在于：①将物理工作空间映射为“空间堆”并用分离逻辑判定碰撞；②引入“空间数据竞争”机制，将证明失败转化为可度量的最小包围盒，提供精准反馈；③形成闭环生成‑验证‑纠错的自我学习流程，完全无需人工干预。

**🔧 技术方法**

使用技术包括：大型语言模型GLLM（基于StarCoder‑3B的微调版）、检索增强生成（RAG）、分离逻辑（Separation Logic）与其离散化解析器、Minkowski和解算离散不确定性、生成-证明循环。

**📊 数据集**

未具体给出数据集，采用GLLM框架内的训练语料和工业工件、夹具几何约束作为测试用例。

**📈 对比分析**

论文未提供实验结果或与传统CAM/VERICUT等工具的对比，仅在理论层面描述算法可行性与安全性保证。

**⚠️ 局限性**

局限性包括：缺乏大规模实测验证、离散化过程中可能导致状态空间爆炸、仅针对单机工具链、对复杂多轴或协作机器人场景的扩展尚未实现。

---

## 670. EnergyLens: Interpretable Closed-Form Energy Models for Multimodal LLM Inference Serving

**arXiv ID:** 2605.10556 | [PDF](https://arxiv.org/pdf/2605.10556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 671. Improving Human Image Animation via Semantic Representation Alignment

**arXiv ID:** 2605.10523 | [PDF](https://arxiv.org/pdf/2605.10523v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 672. When Should Teachers Control AI Generation for Mathematics Visuals?

**arXiv ID:** 2605.10672 | [PDF](https://arxiv.org/pdf/2605.10672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 673. A Recursive Decomposition Framework for Causal Structure Learning in the Presence of Latent Variables

**arXiv ID:** 2605.10651 | [PDF](https://arxiv.org/pdf/2605.10651v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 674. Why Zeroth-Order Adaptation May Forget Less: A Randomized Shaping Theory

**arXiv ID:** 2605.10658 | [PDF](https://arxiv.org/pdf/2605.10658v1)

**作者:** Yao Shu `[一作]`, Zhongxiang Dai `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在持续学习中使用低查询量零阶优化（ZO）对梯度进行随机形状化，以降低模型在新任务适应时的遗忘问题，并将该形状化方法迁移到精确梯度更新中；

**💡 创新点**

创新点在于将ZO的随机形状化精确校准为均值对齐、范数匹配和归一化的组合，并证明其能在局部保持保留曲率的等距底层同时压缩各向异性成分，从而在适当的方向上减少均值二次遗忘；

**🔧 技术方法**

采用了零阶差分估计、随机梯度形状化、块级形状化包装、与普通FO梯度结合的正则化和归一化技术；

**📊 数据集**

在视觉领域使用ViT-B/16在CIFAR100、ImageNet‑R、DomainNet等数据集，在语言领域使用T5在CL Benchmark、GLUE、SuperGLUE等任务顺序；

**📈 对比分析**

与传统FO、可学习分类头、Adapter、EASE、APER等基线以及ZO-FC对比，表现为在多数任务上显著提升Last准确率并降低Fgt，尤其在FO遗忘严重的情形下效果更突出；

**⚠️ 局限性**

局限性包括对曲率信息的依赖仍有限，块级形状化可能无法捕捉跨块耦合，且在FO已极低遗忘时提升空间有限。

---

## 675. Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm

**arXiv ID:** 2605.10640 | [PDF](https://arxiv.org/pdf/2605.10640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 676. MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image

**arXiv ID:** 2605.10616 | [PDF](https://arxiv.org/pdf/2605.10616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 677. Demystifying Deep Reinforcement Learning: A Neuro-Symbolic Framework for Interpretable Open RAN Automation

**arXiv ID:** 2605.10648 | [PDF](https://arxiv.org/pdf/2605.10648v1)

**作者:** Jie Lu `[一作]` (Michigan State University), Huacheng Zeng `[通讯]` (Michigan State University)

**通讯引用:** 1865 | [OpenAlex ID](https://openalex.org/A5027120851)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个神经符号框架，将O‑RAN中使用深度强化学习（DRL）的控制器转化为可解释、可审核、可执行的符号策略，并在真实5G NR O‑RAN测试平台上实现并验证。

**💡 创新点**

创新点包括：① 以3GPP/ O‑RAN规范为基础的概念化抽象层，将海量KPM压缩为语义明确的概念；② 针对连续和离散控制分别采用深度符号回归（DSR）与神经引导可微逻辑（NUDGE），实现维度独立的符号化提炼；③ 设计双阶段符号动作屏蔽机制（规则校正 + 安全决策检索），保证约束满足与QoS合规；④ 将上述方法整合到实时Near‑RT RIC中，首次在生产环境中完成完整的XAI管道。

**🔧 技术方法**

主要技术：概念化抽象器（基于DeepSets与线性头辅助训练）、深度符号回归、神经引导可微逻辑、动作屏蔽（规则投影与安全决策回放）、Integrated Gradients进行概念审计、PPO/Double‑DQN作为教师策略。

**📊 数据集**

数据集：自建的室内多小区5G NR O‑RAN测试床，包含真实设备（14台手机）、真实流量生成、持续的KPM日志（每500 ms一次）。未使用公开数据集。

**📈 对比分析**

与教师、Metis、SYMBXRL三种基线进行比较。评价指标包括累计奖励、QoS违约率、近RT延迟、内存占用和符号模型复杂度Ω。实验结果显示，DeRAN在资源切片任务中恢复78%奖励、在切换任务中恢复87%奖励；相比教师延迟从3.8 ms降至0.265 ms，内存从410 MB降至18 MB；Ω分别为56（资源切片）/26（切换），大幅低于SYMBXRL（Ω=206/152）和Metis（Ω=532/103），且在QoS违约率上优于所有对比方法。

**⚠️ 局限性**

局限性：① 需要人工定义的概念模板，对新业务或协议版本的迁移依赖人工；② 当KPM维度极高或动作空间更复杂时，维度独立的符号化可能无法保持足够的表达能力；③ 动作屏蔽的规则校正与安全决策检索仍可能漏判，尤其在极端网络状态下；④ 仅在单一室内测试平台验证，尚未在大规模运营网络中进行大规模评估。

---

## 678. Hierarchical Causal Abduction: A Foundation Framework for Explainable Model Predictive Control

**arXiv ID:** 2605.10624 | [PDF](https://arxiv.org/pdf/2605.10624v1)

**作者:** Ramesh Arvind Naagarajan `[一作]` (Chemnitz University of Technology), Stefan Streif `[通讯]` (Chemnitz University of Technology)

**通讯引用:** 1904 | [OpenAlex ID](https://openalex.org/A5050202669)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种Hierarchical Causal Abduction (HCA) 框架，用来解释基于模型预测控制（MPC）的决策，将优化证据、物理知识图和时间序列因果发现结合起来。

**💡 创新点**

创新点在于三元融合：①利用KKT乘子捕捉主动约束的优化原因；②使用领域知识图推理物理因果链；③通过PCMCI算法发现历史的时间滞后因果关系，并以层级假设检验与反事实验证生成可解释性叙述。

**🔧 技术方法**

核心技术包括非线性MPC求解、KKT乘子提取、知识图（KG）构建与推理、PCMCI时间序列因果发现、基于层级假设的排名与验证，以及GPT‑4o等LLM进行自然语言合成。

**📊 数据集**

使用了三套真实/模拟数据集：温室气候控制（仿真+真实扰动）、建筑能源管理（实际HVAC功耗）和Tennessee Eastman化工过程（57,500步操作数据）。

**📈 对比分析**

与LIME、SHAP、IOC、MPC‑XAI、LSTM+Attention、RETAIN等基线对比，HCA在三域中的Answer Correctness（AC）分别达到0.478、0.394、0.406，较LIME提升约54%；通过域特定阈值微调后AC可提升至≈0.88，且消融实验表明去掉任一证据源会导致32–37%的准确率下降。

**⚠️ 局限性**

主要局限包括：需要耗时1–2周的专家手工KG构建；对KKT阈值的域特定调优仍是性能瓶颈；适用于有显式优化表达式的控制器，无法直接用于模型无关的RL或黑盒控制；在高噪声或约束失效时PCMCI和KKT可能失效，导致解释不完整。

---

## 679. Re-Triggering Safeguards within LLMs for Jailbreak Detection

**arXiv ID:** 2605.10611 | [PDF](https://arxiv.org/pdf/2605.10611v1)

**作者:** Zheng Lin `[一作]` (Xidian University), Haichang Gao `[通讯]` (Xidian University)

**通讯引用:** 1693 | [OpenAlex ID](https://openalex.org/A5086029723)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于嵌入扰动的检测方法，通过在输入提示的嵌入层注入噪声，重新触发LLM自带的安全防护，从而检测并拦截破坏性提示。

**💡 创新点**

创新点在于：①发现破坏性提示本质上脆弱，可通过轻微扰动转化为拒绝回应；②通过对扰动效果的系统分析，识别出“锚点”嵌入，从而设计两阶段高效噪声搜索算法；③实现白盒与黑盒均可工作，且对自适应攻击具有鲁棒性。

**🔧 技术方法**

使用的技术包括：嵌入层噪声注入、锚点嵌入引导搜索、随机搜索优化、emb2token映射、黑盒转移攻击（多射）等；实验基于Vicuna‑13B、LLaMA2‑7B‑Chat、Qwen2.5‑7B‑Instruct以及GPT‑4.1/Gemini‑2.5。

**📊 数据集**

主要数据集包括AdvBench、JailbreakBench用于收集破坏性提示；IFEval与AlpacaEval用于评估模型实用性；此外使用多种攻击方法（GCG、PAIR、RS、I‑FSJ、AutoDAN‑Turbo）构造评估集。

**📈 对比分析**

与Perplexity Filter、Erase‑and‑Check、SmoothLLM、RESTA等传统防御对比，本文方法在攻击成功率（ASR）下降、检测率（DR）高达0.9+、误报率（FR）低于1%，并在自适应攻击与黑盒转移场景下仍保持优越性能；实验表明在白盒与黑盒均可实现高效检测，且保持模型实用性不受影响。

**⚠️ 局限性**

局限性包括：需要预先为每个模型提取锚点嵌入，若模型或分词器变化需重新计算；搜索过程在大规模批量推理时仍有计算开销；对极强自适应攻击或完全未知的破坏性提示可能仍存在漏检；黑盒转移效果在某些模型上不如白盒显著。

---

## 680. Measuring Embedding Sensitivity to Authorial Style in French: Comparing Literary Texts with Language Model Rewritings

**arXiv ID:** 2605.10606 | [PDF](https://arxiv.org/pdf/2605.10606v1)

**作者:** Benjamin Icard `[一作]` (Sorbonne University), Jean-Gabriel Ganascia `[通讯]` (Sorbonne University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比人类作者与大型语言模型（LLM）在固定主题下的写作风格改写，评估嵌入向量对作者风格的敏感性；

**💡 创新点**

首次在法语文学语料上同时考察嵌入向量的风格编码与LLM改写后的保留程度，揭示嵌入空间与风格维度的相关性；

**🔧 技术方法**

使用13种多语言文本嵌入模型（如xlm-roberta、mistral、gemini等）、UMAP降维、k‑means聚类与皮尔逊相关分析；

**📊 数据集**

构建1,248条法语文学文本语料，包含Tufféry原作、Proust、Céline、Yourcenar四位作者原始文本及其被3种LLM改写的8,64条文本；

**📈 对比分析**

与传统字符ngram+TF‑IDF+SVM风格转移验证对比，发现LLM在不同作者上的风格保留程度与验证准确率存在一定偏差；性能方面，原始文本的聚类纯度最高，LLM生成文本的聚类纯度与原文本相差不大，但风格相关性系数均为中等偏弱；

**⚠️ 局限性**

主要局限包括：仅使用2D UMAP降维，可能导致信息失真；所选风格特征面向特定作者，缺乏普适性；相关性仅为关联关系，未揭示因果机制；实验仅限法语文学文本，难以推广至其他语言或非文学文本。

---

## 681. Fairness vs Performance: Characterizing the Pareto Frontier of Algorithmic Decision Systems

**arXiv ID:** 2605.10604 | [PDF](https://arxiv.org/pdf/2605.10604v1)

**作者:** Mieke Wilms `[一作]` (Zurich University of Applied Sciences), Christoph Heitz `[通讯]` (Zurich University of Applied Sciences)

**通讯引用:** 603 | [OpenAlex ID](https://openalex.org/A5036601212)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文以多目标优化视角研究二元预测决策系统的公平与性能权衡，提供了 Pareto 前沿的理论刻画，并证明其由各组特定阈值规则决定，阈值可为上界或下界；

**💡 创新点**

创新点在于：①在极其一般的公平度量（包括非平等主义原则）与决策者/受益者效用矩阵下，完全刻画 Pareto 前沿；②证明上界阈值规则的可能性；③发现 Pareto 前沿仅取决于各组概率分布 g(p|a)，与算法实现无关；

**🔧 技术方法**

主要技术为概率论与多目标优化理论，利用阈值规则的构造与极值证明，推导了 Pareto 最优决策规则；

**📊 数据集**

示例实验使用合成数据以及成人收入（Adult Income）数据集；

**📈 对比分析**

与 PF‑SMG（Pareto Front Stochastic Multi‑Gradient）方法对比，作者通过构造阈值规则得到的 Pareto 前沿在性能上优于 PF‑SMG，且不依赖敏感属性；

**⚠️ 局限性**

局限性包括：仅处理单一公平度量，无法同时满足多重公平约束；未考虑个体化效用函数；实际实现仍需逼近 Bayes‑optimal 预测。

---

## 682. A Spectral Framework for Closed-Form Relative Density Estimation

**arXiv ID:** 2605.10668 | [PDF](https://arxiv.org/pdf/2605.10668v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 683. When Can Digital Personas Reliably Approximate Human Survey Findings?

**arXiv ID:** 2605.10659 | [PDF](https://arxiv.org/pdf/2605.10659v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 684. Segment Anything with Robust Uncertainty-Accuracy Correlation

**arXiv ID:** 2605.10603 | [PDF](https://arxiv.org/pdf/2605.10603v1)

**作者:** Hongyou Zhou `[一作]` (Technical University of Berlin), Zihan Ye `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在单源域下改进Segment Anything Model (SAM2)，实现零样本跨域分割时同时提供像素级不确定性估计，解决Mask级置信度混淆（MCC）问题。

**💡 创新点**

创新点包括：1) 引入轻量级贝叶斯掩码解码器，实现像素级不确定性；2) 设计基于生物学启发的风格-形变对抗生成器（Style+Deformation），对抗性训练专注于不确定性-准确性对齐；3) 通过不确定性校准损失和梯度反转实现端到端无内循环的对抗学习。

**🔧 技术方法**

技术手段主要包括：贝叶斯多粒度Weibull分布掩码解码器、风格自适应AdaIN网络、几何形变网络（DG‑Font风格）、梯度反转层（GRL）实现的min‑max优化、不确定性-准确性对齐损失、Patch Accuracy vs. Patch Uncertainty (PAvPU) 等评估指标。

**📊 数据集**

使用MOSE视频对象数据集作为唯一训练源，评估零样本性能于23个公开域（Objects, Scenes, Scientific, Egocentric），包括Cityscapes、IBD、Hypersim、NDISPark等多样化数据集。

**📈 对比分析**

与SAM2零样本、SAM2‑FT、SAM2‑FT‑LoRA、Bayes‑SAM2、UR‑ERN、UCTTA及多种通用增强方法对比，RUAC在平均J&F上提升至81.62%（比SAM2高≈14%），在不确定性校准指标PAvPU、AURC、ECE等方面显著优于所有基线，且在下游不确定性引导的掩码修正中表现最佳。

**⚠️ 局限性**

局限性包括：1) 对抗生成器训练仍依赖于梯度反转和超参数调节，可能在极端域下收敛不稳定；2) 仅在单源域设置下验证，跨源多源学习尚未探究；3) 计算开销相对传统SAM略大，尤其在推理时需要贝叶斯掩码解码。

---

## 685. Natural Policy Gradient as Doubly Smoothed Policy Iteration: A Bellman-Operator Framework

**arXiv ID:** 2605.10671 | [PDF](https://arxiv.org/pdf/2605.10671v1)

**作者:** Phalguni Nanda `[一作]` (Purdue University), Zaiwei Chen `[通讯]` (Purdue University)

**通讯引用:** 237 | [OpenAlex ID](https://openalex.org/A5058269077)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了双重平滑策略迭代（DSPI）框架，将经典的策略迭代、自然策略梯度（NPG）和策略双重平均（PDA）等算法统一在 Bellman 迭代的视角下，并给出了其全局几何收敛性分析；

**💡 创新点**

创新点在于：①用平滑的 Bellman 运算子将 NPG 与传统策略迭代等价化；②通过两层平滑（Q 值平均与熵正则化）实现统一框架；③在无额外正则化、无分布依赖的前提下获得分布无关的全局几何收敛速率；

**🔧 技术方法**

主要技术：Bellman 运算子性质（单调性、收缩性）分析；DSPI 的迭代规则推导；对 NPG、PDA 的等价性证明；收敛递推与闭式解；

**📊 数据集**

无；该工作为理论分析，未使用实验数据集；

**📈 对比分析**

与现有 NPG 和 PDA 的分析对比，本文实现了不依赖初始分布、无额外正则化且步骤大小可预设的收敛速率 𝒪((1‑γ)⁻¹log((1‑γ)⁻¹ε⁻¹))；此外，双重平均策略迭代实现有限次终止；

**⚠️ 局限性**

局限性：仅分析了确定性环境下的完美 Q‑值估计；未考虑模型无关、采样误差；缺乏实验验证；未来工作需结合 TD 学习或 Monte‑Carlo 评估实现样本复杂度分析。

---

## 686. Neuromorphic Monocular Depth Estimation with Uncertainty Modeling

**arXiv ID:** 2605.10675 | [PDF](https://arxiv.org/pdf/2605.10675v1)

**作者:** Viktor Bergkvist `[一作]` (Swedish Defence Research Agency), Johan Rideg `[通讯]` (Swedish Defence Research Agency)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并评估了从单目事件相机流预测像素级深度及其不确定性的深度学习方法。

**💡 创新点**

首次将多种事件表示与高斯、对数正态、证据学习三种不确定性模型结合，实现事件相机单目深度不确定性估计。

**🔧 技术方法**

基于U‑Net的递归编码解码网络，使用卷积LSTM，配合时空体素、CSTR、TORE 等事件表示，优化对应的负对数似然损失。

**📊 数据集**

先在 BlinkVision 合成室内序列上预训练，再在 MVSEC 室内飞行真实数据上微调，同时对灰度图像数据也进行训练。

**📈 对比分析**

通过 AbsRel、RMSE、AUSE 三个指标比较不同事件表示与不确定性模型，10‑bin 体素+对数正态和 5‑bin+证据学习在测试集上表现最佳，接近 E2Depth 基线但额外提供可靠的不确定性评估。

**⚠️ 局限性**

缺乏对实时性、内存消耗和推理速度的系统评估；深度图平滑且缺少边界锐化，模型对不同事件表示的敏感度较低。

---

## 687. GenMed: A Pairwise Generative Reformulation of Medical Diagnostic Tasks

**arXiv ID:** 2605.10645 | [PDF](https://arxiv.org/pdf/2605.10645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 688. Active Learning for Gaussian Process Regression Under Self-Induced Boltzmann Weights

**arXiv ID:** 2605.10654 | [PDF](https://arxiv.org/pdf/2605.10654v1)

**作者:** Jixiang Qing `[一作]` (Lancaster University), Matthias Sachs `[通讯]` (Lancaster University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在自诱导分布（SID）下的主动学习框架 SIDAL，并给出了基于高斯过程的无分区函数估计的闭式采样准则与其 Thompson 采样变体；同时提供了终止预测误差高概率与平均收敛分析。

**💡 创新点**

① 将目标分布视为未知函数自身的 Boltzmann 形式；② 设计了无需估计配分函数的采集函数（AB‑SID）和高方差 MC 变体（TS‑SID）；③ 在连续输入域下给出完整的高概率与平均案例理论保证；④ 统一分析并证明现有启发式方法的收敛性。

**🔧 技术方法**

采用高斯过程回归、变分信息、积分式采集准则 iVAR、零阶 Taylor 近似、Monte Carlo 与 MCMC 采样、信息增益 γ_T 及相关上界分析。

**📊 数据集**

合成基准（1D‑6D Gramacy、Branin、Hartmann、Ishigami 等）；真实潜能能面（H₂/Cu、H₂/Cu‑cluster、Si crystal、H₂O/Pt）；分子药物发现（GuacaMol 约 20,000 分子，Tanimoto kernel GP）。

**📈 对比分析**

与随机采样、置信不确定性、IMSE、EPIG 等基线对比。实验显示 SIDAL 在所有任务中均显著优于基线，尤其在高维或高度集中目标分布下误差低一至两阶，且在药物发现任务中显著提升高分子分子预测精度。

**⚠️ 局限性**

局限性：① 需要先验知道 λ 与偏置 b；② 连续域需 MCMC 采样，计算成本随维度上升；③ 对核的可微性（四阶可导）有严格要求；④ 对非 Boltzmann 形式的目标分布理论可能不适用；⑤ 高 λ 时分布极度集中，可能导致采样偏差。

---

## 689. Intrinsic Guardrails: How Semantic Geometry of Personality Interacts with Emergent Misalignment in LLMs

**arXiv ID:** 2605.10633 | [PDF](https://arxiv.org/pdf/2605.10633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 690. Embodied AI in Action: Insights from SAE World Congress 2026 on Safety, Trust, Robotics, and Real-World Deployment

**arXiv ID:** 2605.10653 | [PDF](https://arxiv.org/pdf/2605.10653v1)

**作者:** Jan-Mou Li `[一作]`, Edward Griffor `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

总结了 SAE 2026 世界大会上关于“Embodied AI in Action”面板讨论的主要观点，梳理了在汽车、机器人等领域嵌入式 AI 的部署挑战与建议。

**💡 创新点**

将嵌入式 AI 定义为系统性工程问题，并提出安全、治理、生命周期管理与跨学科协作的完整框架。

**🔧 技术方法**

结合安全工程、AI/ML 设计、系统集成、标准化与治理等多学科技术手段。

**📊 数据集**

本白皮书未使用具体实验数据集，而是基于行业专家经验与现有标准（如 SAE J3016、ISO 21448 等）。

**📈 对比分析**

未进行量化实验比较，本文通过案例分析与专家共识阐述了可行性与最佳实践，未给出具体性能指标。

**⚠️ 局限性**

局限在于缺乏实证验证，主要为概念性与政策性指导，难以直接评估技术细节与实际效果。

---

## 691. Interpretable Coreference Resolution Evaluation Using Explicit Semantics

**arXiv ID:** 2605.10627 | [PDF](https://arxiv.org/pdf/2605.10627v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 692. Teacher-Aware Evolution of Heuristic Programs from Learned Optimization Policies

**arXiv ID:** 2605.10634 | [PDF](https://arxiv.org/pdf/2605.10634v1)

**作者:** Minyu Chen `[一作]` (Shenzhen Technology University), Guoqiang Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 16734 | [OpenAlex ID](https://openalex.org/A5100421251)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种教师感知的进化框架，用已训练的优化策略作为行为教师来指导LLM进化，最终生成静态可执行的启发式程序。

**💡 创新点**

创新点在于将行为级反馈（教师偏好）引入进化搜索，而非仅靠端点性能或文本反思，实现对局部决策的指导。

**🔧 技术方法**

使用LLM（GPT‑5.4）生成程序，独立训练的神经策略（L2D、LEHD、ECO‑DQN）做教师，按位对齐、诊断摘要、分析器LLM生成修订提示，并结合Pareto多目标选择。

**📊 数据集**

实验数据集包括四个序列组合优化基准：JSSP、TSP、CVRP、MaxCut，分别在设计集、OOB、标准基准上评测。

**📈 对比分析**

与性能驱动的LLM启发式进化基线（EoH、ReEvo）及经典启发式和强学习器对比，结果显示在所有任务上均优于基线且在OOB迁移上保持优势，且运行时不需神经推理。

**⚠️ 局限性**

局限性包括需要可对齐的教师策略、任务接口适配器、分析器调用，且教师反馈的质量和适用范围受教师训练效果限制。

---

## 693. Responsible Benchmarking of Fairness for Automatic Speech Recognition

**arXiv ID:** 2605.10615 | [PDF](https://arxiv.org/pdf/2605.10615v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 694. Hierarchical End-to-End Taylor Bounds for Complete Neural Network Verification

**arXiv ID:** 2605.10621 | [PDF](https://arxiv.org/pdf/2605.10621v1)

**作者:** Taha Entesari `[一作]` (Johns Hopkins University), Mahyar Fazlyab `[通讯]` (Johns Hopkins University)

**通讯引用:** 718 | [OpenAlex ID](https://openalex.org/A5043154223)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种新的神经网络可达性分析框架HiTaB，旨在通过利用Hessian的Lipschitz连续性来提高平滑神经网络的安全性和鲁棒性验证。

**💡 创新点**

创新点在于系统性地利用高阶平滑性信息，特别是Hessian的Lipschitz常数，来提供更紧凑的可达性界限，并引入了基于零阶、一阶和二阶信息的统一界限层次结构。

**🔧 技术方法**

使用了层级传播的方式来高效地上界深度神经网络中的Hessian Lipschitz常数，并结合了二阶泰勒模型和分支-界限验证管道。

**📊 数据集**

实验中使用了LTI系统控制的神经网络，特别是针对六维四旋翼的可达性分析。

**📈 对比分析**

与现有的第一阶局部界限方法相比，HiTaB在生成的分支数量上显著减少，表明其提供了更紧凑的局部近似，尤其是在较小的子域中效果更为明显。

**⚠️ 局限性**

限制在于目前的实验主要集中在ℓ_∞约束问题上，且当前实现为基于CPU，未利用GPU并行处理，未来可以通过GPU优化来提高运行时间和可扩展性。

---

## 695. Step Rejection Fine-Tuning: A Practical Distillation Recipe

**arXiv ID:** 2605.10674 | [PDF](https://arxiv.org/pdf/2605.10674v1)

**作者:** Igor Slinko `[一作]` (JetBrains Research), Yaroslav Zharov `[通讯]` (JetBrains Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了Step Rejection Fine‑Tuning（SRFT）方法，在传统RFT的基础上对未成功轨迹中的错误步骤进行判别并掩码，从而利用部分有效步骤提升模型性能。

**💡 创新点**

创新点在于利用LLM批评器对轨迹中每一步进行细粒度评估，允许保留未完成轨迹中有用的步骤，而不是整体丢弃，突破了RFT忽略大量数据的局限。

**🔧 技术方法**

采用监督式微调的RFT框架，并结合Claude 4 Sonnet批评器进行步骤级打分，使用加权负对数似然（NLL）损失实现对错误步骤的掩码。

**📊 数据集**

使用SWE‑Agent框架下的SWE‑Language dataset（约25,000条轨迹，39%已完成，61%未完成），实验取5,000条完成轨迹和5,000条未完成轨迹进行训练与评估。

**📈 对比分析**

通过与基线模型、无微调、naïve distillation以及传统RFT的对比，SRFT在Resolved Rate上取得32.2%（标准差0.9%），显著高于RFT的30.9%（提升1.3%，95%置信区间[0.4, 2.3]）。

**⚠️ 局限性**

主要局限在于对批评器准确性的高度依赖，误判会削弱有效数据；未探索非二值权重或更细粒度的损失权重；缺乏跨任务的泛化验证。

---

## 696. Evolving-RL: End-to-End Optimization of Experience-Driven Self-Evolving Capability within Agents

**arXiv ID:** 2605.10663 | [PDF](https://arxiv.org/pdf/2605.10663v1)

**作者:** Zhiyuan Fan `[一作]` (Xiaohongshu Inc), Jiawei Li `[通讯]` (Xiaohongshu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Evolving‑RL，一种联合优化经验提取与利用的框架，使 LLM 在部署时能够自我演化并提取、评估并利用可重用的文本技能。

**💡 创新点**

创新点在于把经验提取与利用视为统一的共进化过程，利用跨任务评估奖励驱动提取器与求解器共训练，从而突破传统方法中提取器独立或依赖手工过滤的局限。

**🔧 技术方法**

采用 GRPO 强化学习、文本技能抽取与检索、共享策略的 extractor 与 solver、离线/在线技能评估及梯度联合优化等技术。

**📊 数据集**

在 ALFWorld（文本嵌入任务）和 Mind2Web（Web 导航任务）两大基准上进行训练与评估。

**📈 对比分析**

与 ExpeL、Memento、ReasoningBank 等提示式自演化方法以及 GRPO、SkillRL 等 RL 基线对比，Evolving‑RL 在 ALFWorld 未见任务上成功率提升至 88.6%（比 GRPO+skills 提升 98.7% ），在 Mind2Web 上动作准确率提升至 30.87%（比 GRPO 提升 45.9%）等显著性能。

**⚠️ 局限性**

局限在于依赖相对简单的技能管理与检索机制，评估过程噪声可能导致训练不稳定，且在跨网站分割下技能注入效果不明显，未来需改进演化机制与评估鲁棒性。

---

## 697. Continuous Defensive Domination Problems

**arXiv ID:** 2605.10607 | [PDF](https://arxiv.org/pdf/2605.10607v1)

**作者:** Christoph Grüne `[一作]` (RWTH Aachen University), Tom Janßen `[通讯]` (RWTH Aachen University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5031210156)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文系统研究了在连续设施定位模型下的防御性δ‑覆盖问题（Defensive δ-Covering）的计算复杂性，分析了不同攻击点限制（仅顶点、任意点）、防御点多重集合允许与否以及δ取值区间（<1/2、[1/2,1)、≥1）对问题复杂度的影响，并给出了多种归约与离散化论证；在此基础上证明了若δ非单位分数则问题为NP‑完备，若δ≥1且攻击可多重则为Σ^P_2‑完备，而在某些特殊δ区间（如<1/2、1/2≤δ<1）问题可在多项式时间求解。

**💡 创新点**

创新点在于：① 将传统的防御支配集问题推广到连续网络并与δ‑覆盖结合；② 提出了针对不同δ区间的细粒度复杂度划分，揭示了连续模型有时比离散模型更易；③ 通过精细化的离散化与构造性归约（如树因子、k‑tuple支配、Set Cover、Clique Interdiction等），首次完整描述了防御性δ‑覆盖在多种输入限制下的P/NP/Σ^P_2 分类；④ 统一了多重集合和单重集合情况的处理方法。

**🔧 技术方法**

主要技术包括：离散化论证（证明存在结构化的最优防御点集合）；多种图论归约（树因子、b‑边覆盖、k‑tuple支配、δ‑覆盖、Clique Interdiction、Min Cardinality Clique Interdiction等）；构造特殊边子图/子图装配以模拟不同δ距离；利用Hall定理与匹配理论证明可防御性；使用多项式长度证书与流网络验证多重集合覆盖；以及在Σ^P_2层次中构造∃∀-量化证书来证明Σ^P_2‑完备。

**📊 数据集**

该工作为理论分析，不使用实验数据集，所有结果均来自数学证明与多项式归约。

**📈 对比分析**

方法对比以复杂度分类为核心：在可多项式时间求解的区间内（δ<1/2、1/2≤δ<1、δ为单位分数时），作者给出多项式算法；在其他区间则给出NP或Σ^P_2 完备证明，表明问题在这些参数下不可在多项式时间内求解。未进行实验性能比较，因本研究为理论复杂度分析。

**⚠️ 局限性**

局限性包括：① 对于δ为非单位分数且k未受多项式界定时，归约仅在k多项式时可行；② 某些区间（如δ≥1且攻击可多重）仅给出Σ^P_2‑完备性，而非更精细的子类复杂度；③ 对于特定图结构（如树、双部图）的进一步简化仍是开放问题；④ 研究未考虑实际网络规模与噪声，对实际应用的可行性尚需实验验证。

---

## 698. diffGHOST: Diffusion based Generative Hedged Oblivious Synthetic Trajectories

**arXiv ID:** 2605.10647 | [PDF](https://arxiv.org/pdf/2605.10647v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 699. A Random-Matrix Criterion for Initializing Gated Recurrent Neural Networks

**arXiv ID:** 2605.10650 | [PDF](https://arxiv.org/pdf/2605.10650v1)

**作者:** Tommaso Fioratti `[一作]` (Capital Fund Management), Francesco Casola `[通讯]` (Capital Fund Management)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过随机矩阵理论推导了门控循环网络在无限宽极限下的临界增益 g_c，并验证其能预测可达的混沌临界点。

**💡 创新点**

创新点在于将门控 RNN 的 Jacobian 结构化为可分析的随机矩阵形式，得到闭式 g_c 计算公式，并将其与 Reservoir Computing 的性能峰值对应。

**🔧 技术方法**

采用随机矩阵理论、动力学线性化、Lyapunov 指数估计和数值实验相结合的方法。

**📊 数据集**

主要使用了 Mackey–Glass chaotic time series 作为预测任务的数据集。

**📈 对比分析**

与传统经验性初始化（如 zero‑bias、Gaussian、Chrono）进行比较，实验表明在 g≈g_c 时测试误差达到最小，性能优于其他非临界设置。

**⚠️ 局限性**

局限性包括假设候选状态无偏置、所有门共享同一增益，以及仅在无输入自举下进行理论推导，未涵盖更一般的偏置和非线性权重分布。

---

## 700. Where do aspectual variants of light verb constructions belong?

**arXiv ID:** 2605.10605 | [PDF](https://arxiv.org/pdf/2605.10605v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 701. Not Blind but Silenced: Rebalancing Vision and Language via Adversarial Counter-Commonsense Equilibrium

**arXiv ID:** 2605.10676 | [PDF](https://arxiv.org/pdf/2605.10676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 702. On the Verification Problem of Remote Direct Memory Access programs (Extended Version with Appendix)

**arXiv ID:** 2605.10631 | [PDF](https://arxiv.org/pdf/2605.10631v1)

**作者:** Parosh Aziz Abdulla `[一作]` (Uppsala University), Stephan Spengler `[通讯]` (Uppsala University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5008674563)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了远程直接内存访问（RDMA）程序的可达性和鲁棒性问题。通过从Post对应问题归约证明了可达性不可判定；随后提出了鲁棒性问题的判定方案，证明鲁棒性可判定且属于EXPSPACE完整。

**💡 创新点**

创新点在于：①首次给出RDMA可达性不可判定的证明；②提出鲁棒性违例的正则形式并基于此构造决策过程；③将鲁棒性判定归约为有限状态程序加计数器的可达性，并给出EXPSPACE上界与下界；④简化了原始RDMA一致性语义，提供等价且更易使用的判定规则。

**🔧 技术方法**

主要技术包括：归约Post对应问题证明不可判定；构造RDMA程序模拟VASS；使用向量加法系统（VASS）来证明EXPSPACE下界；构造仪器程序将RDMA程序转化为在SC语义下的线性化搜索；利用poll-from、reads-from、issue-preserved等关系实现一致性检查；使用线性化与可达性约简。

**📊 数据集**

实验主要基于Ambal等人提出的RDMA litmus test集合；未使用大型工业数据集，而是以理论构造和小型测试用例验证方法。

**📈 对比分析**

与之前仅给出充分条件的鲁棒性检测方法相比，本文提供了完整的判定算法；实验中通过SPIN对小型litmus测试进行验证，虽然在规模上受限，但验证结果与理论一致，表明方法可行。

**⚠️ 局限性**

局限性包括：①方法针对的是当前RDMA语义，未覆盖更丰富的RDMA功能；②在有poll操作的程序中仍需要计数器，导致EXPSPACE复杂度高，难以在工业规模上直接应用；③实验规模有限，尚未证明在大规模真实系统中的可扩展性。

---

## 703. Compander-Aligned Query Geometry for Quantized Zeroth-Order Optimization

**arXiv ID:** 2605.10673 | [PDF](https://arxiv.org/pdf/2605.10673v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 704. The Open-Box Fallacy: Why AI Deployment Needs a Calibrated Verification Regime

**arXiv ID:** 2605.10601 | [PDF](https://arxiv.org/pdf/2605.10601v1)

**作者:** Phongsakon Mark Konrad `[一作]` (University of Southern Denmark), Serkan Ayvaz `[通讯]` (University of Southern Denmark)

**通讯引用:** 1452 | [OpenAlex ID](https://openalex.org/A5075890706)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了“校准验证”治理框架并定义了六项验证覆盖指标，作为AI部署的可报告标准

**💡 创新点**

创新点在于将模型解释性从唯一授权门槛转为多维验证门槛，提出Verification Coverage概念和最小组合规则

**🔧 技术方法**

采用了多来源证据流、验证器类别划分、制度化属性设计等概念性方法，并对现有法规进行系统化映射

**📊 数据集**

主要利用了文献综述、法规条文和案例研究（如FDA、信用、就业、自动驾驶、刑事司法）来构建验证覆盖表格

**📈 对比分析**

未进行实验性对比，而是通过跨域对照表展示不同领域在六项指标上的强弱，强调最弱指标决定是否授权

**⚠️ 局限性**

局限性包括缺乏量化实验验证、对开放模型监管的可执行性挑战、对不同领域具体指标的细化不足以及可能被滥用的报告形式

---

## 705. BCJR-QAT: A Differentiable Relaxation of Trellis-Coded Weight Quantization

**arXiv ID:** 2605.10655 | [PDF](https://arxiv.org/pdf/2605.10655v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 706. Surviving Partial Rank Failures in Wide Expert-Parallel MoE Inference

**arXiv ID:** 2605.10670 | [PDF](https://arxiv.org/pdf/2605.10670v1)

**作者:** Xun Sun `[一作]` (Tsinghua University), Mingxing Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 26143 | [OpenAlex ID](https://openalex.org/A5100621291)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出一种能够在宽专家并行(MoE)推理中实现部分故障容忍的系统

**💡 创新点**

将 EP 成员资格视为可变运行时状态，设计了基于 GPU 局部同等表和三阶专家修复层次的无重建恢复机制，支持异步重整入（deferred‑join）

**🔧 技术方法**

使用 GPUDirect RDMA、CUDA 图、GPU 局部同等表、分层专家备份（本地复制、GPU‑GPU 迁移、DRAM 备份）以及自定义的分布式运行时后端

**📊 数据集**

DeepSeek‑V3（约 671 B 参数、256 个专家）的 MoE 模型

**📈 对比分析**

与固定成员 DeepEP 基线对比，静态推理时延低 4.4% 以内；单一 rank 故障恢复仅需 11 s 及 8 s 重新整合，总停机时间约 19 s，显著快于全实例重启的 348 s；多 rank 故障同样保持在 6–11 s 的恢复窗口，恢复后吞吐率保持 95% 以上

**⚠️ 局限性**

仅针对 fail‑stop 故障，未覆盖 Byzantine、网络分区、持续慢速等情况；需要额外的备份存储容量；在极大规模场景下多级迁移与 DRAM 备份成本可能显著上升

---

## 707. Product-of-Gaussian-Mixture Diffusion Models for Joint Nonlinear MRI Reconstruction

**arXiv ID:** 2605.10629 | [PDF](https://arxiv.org/pdf/2605.10629v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 708. Generate "Normal", Edit Poisoned: Branding Injection via Hint Embedding in Image Editing

**arXiv ID:** 2605.10600 | [PDF](https://arxiv.org/pdf/2605.10600v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 709. LLaVA-CKD: Bottom-Up Cascaded Knowledge Distillation for Vision-Language Models

**arXiv ID:** 2605.10641 | [PDF](https://arxiv.org/pdf/2605.10641v1)

**作者:** Nikolaos Gkalelis `[一作]` (CERTH-ITI), Vasileios Mezaris `[通讯]` (CERTH-ITI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种从低容量教师助理（TA）逐步向学生模型传递知识的底向级联知识蒸馏框架（CKD），并在LLaVA体系下实现。

**💡 创新点**

创新点在于：①采用底向级联蒸馏，先用中等容量TA提升学生，再用更大教师进一步蒸馏，从而在保持学生轻量的同时最大化知识利用；②理论上证明了层级蒸馏能降低学生-教师容量差并提升泛化；③与传统单阶段蒸馏相比，实验验证了显著性能提升。

**🔧 技术方法**

技术方法：基于TinyLLaVA预训练 + LLaVA-KD蒸馏，包含预训练（PT）、细化训练（FT）与蒸馏细化（DFT）三阶段；使用多模态交叉熵、KL散度及视觉余弦相似度损失；实现层级蒸馏的两个阶段（TA→学生、教师→学生）。

**📊 数据集**

数据集：使用LLaVA1.5数据集（558k图文对）进行预训练，使用LLaVA-mix-665k高质量指令数据进行细化；评测使用七个公开基准：VGA v2、GQA、TextVQA、ScienceQA-IMG、MMEP、POPE、MMMU。

**📈 对比分析**

对比方法：与TinyLLaVA、LLaVA-KD基线以及多款SOTA轻量VLM（如LLaVA-MoD、Allign-KD、CompoDistill等）进行比较；CKD在0.5B模型上平均提升约0.9%，1.5B模型上平均提升约1%；在大多数基准上获得SOTA或接近SOTA的成绩，并在5–7项基准中取得首/次列名。

**⚠️ 局限性**

局限性：当学生与教师的容量差过大时，级联蒸馏的增益递减，无法无限扩展至极大教师；额外的蒸馏阶段显著增加计算成本（约双倍训练时间）。

---

## 710. Prompt-Activation Duality: Improving Activation Steering via Attention-Level Interventions

**arXiv ID:** 2605.10664 | [PDF](https://arxiv.org/pdf/2605.10664v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 711. bViT: Investigating Single-Block Recurrence in Vision Transformers for Image Recognition

**arXiv ID:** 2605.10661 | [PDF](https://arxiv.org/pdf/2605.10661v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 712. Verifying Sequential Consistency under Bounded Preemptions

**arXiv ID:** 2605.10625 | [PDF](https://arxiv.org/pdf/2605.10625v1)

**作者:** R. Govind `[一作]` (Institute of Mathematical Sciences), B. Srivathsan `[通讯]` (Chennai Mathematical Institute)

**通讯引用:** 263 | [OpenAlex ID](https://openalex.org/A5047476786)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了在有限预抢次数下的顺序一致性验证问题（VSC‑problem），给出了单写程序的多项式算法，并在多写程序以及预抢次数未固定时给出了复杂度下界。

**💡 创新点**

提出利用冲突图对单写程序的块程序进行排序，从而消除对所有线程排列的指数枚举；同时给出了基于 3‑CNF‑SAT 的多写程序 NP‑难、ETH 下的细粒度下界以及 W[1]‑难的证明。

**🔧 技术方法**

主要技术包括：预抢点猜测、块程序划分、冲突图构造与拓扑排序、内块排列枚举、顺序一致性检验；以及从 SAT、独立集等经典 NP 问题的归约。

**📊 数据集**

未使用公开数据集；所有实验与证明均基于理论构造的实例。

**📈 对比分析**

对比方法：单写程序可在 O(n^{π+1}·k·π!) 时间内求解，证明在多写或预抢次数未固定时难以实现多项式或 FPT 算法；因此在这些情形下性能受限。

**⚠️ 局限性**

局限性：算法仅适用于每个变量只有唯一写线程的程序；对多写程序只能给出 NP‑难/ETH 下界或 W[1]‑难，无法提供有效求解方案；预抢次数若未固定，则问题不具备固定参数可解性。

---

## 713. Navigating the Sea of LLM Evaluation: Investigating Bias in Toxicity Benchmarks

**arXiv ID:** 2605.10639 | [PDF](https://arxiv.org/pdf/2605.10639v1)

**作者:** Regina Gugg `[一作]` (Dynatrace Research), Martin Flechl `[通讯]` (Dynatrace Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了毒性基准在任务切换（对话转为摘要）和领域切换（金融、社交媒体、化工、体育）下的鲁棒性，并对多种评估器之间的一致性进行对比；

**💡 创新点**

创新点在于首次量化任务与领域变化对毒性检测结果的影响，构建跨基准的一致性评估矩阵，并揭示当前评估器间高度不一致的缺陷；

**🔧 技术方法**

采用了 McNemar 检验、广义线性模型（GLM）、卡方检验、Cohen’s κ 等统计方法，对改进后的 Perspective API、ToxiGenRoBERTa、Longformer、Llama2‑13B‑cls 等分类器进行评估，并对 LLM 生成结果采用固定参数进行实验；

**📊 数据集**

使用了 RealToxicityPrompts、HarmBench、ToxiGen、DoNotAnswer 等四个主流毒性基准的子集，并通过 GPT‑4o 对输入进行领域迁移，构成了跨领域的评估数据集；

**📈 对比分析**

通过比较基准的原始任务与摘要任务的结果，计算 Paired Odds Ratio (OR_p) 与 Population Odds Ratio (OR_pop)，并使用 Cohen’s κ 评估评估器一致性；实验表明摘要任务普遍提高毒性判定率，领域切换大多降低检测率，评估器之间的一致性极低；

**⚠️ 局限性**

局限性包括仅覆盖有限的模型与基准、仅使用英文数据、未考虑拒绝、对抗性攻击与多语言场景，域迁移过程中可能引入语言偏差，需进一步扩大样本与模型多样性以提升结论的普适性。

---

## 714. A Single-Layer Model Can Do Language Modeling

**arXiv ID:** 2605.10643 | [PDF](https://arxiv.org/pdf/2605.10643v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 715. Security Analysis of Time-of-Arrival Estimation via Cross-Correlation under Narrow-Band Conditions

**arXiv ID:** 2605.10632 | [PDF](https://arxiv.org/pdf/2605.10632v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 716. Hypergraph-Enhanced Training-Free and Language-Free Few-Shot Anomaly Detection

**arXiv ID:** 2605.10628 | [PDF](https://arxiv.org/pdf/2605.10628v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 717. Vocabulary Hijacking in LVLMs: Unveiling Critical Attention Heads by Excluding Inert Tokens to Mitigate Hallucination

**arXiv ID:** 2605.10622 | [PDF](https://arxiv.org/pdf/2605.10622v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 718. Reconfigurable Computing Challenge: Real-Time Graph Neural Networks for Online Event Selection in Big Science

**arXiv ID:** 2605.10612 | [PDF](https://arxiv.org/pdf/2605.10612v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 719. Sparse Signal Recovery using Log-Sum Regularization and Adaptive Smoothing

**arXiv ID:** 2605.10626 | [PDF](https://arxiv.org/pdf/2605.10626v1)

**作者:** Keisuke Morita `[一作]` (Tohoku University), Masayuki Ohzeki `[通讯]` (Tohoku University)

**通讯引用:** 1995 | [OpenAlex ID](https://openalex.org/A5035163865)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了利用对数-求和正则化与自适应平滑方法进行稀疏信号恢复，并通过AMP与SE理论以及ADMM实现验证。

**💡 创新点**

创新点在于引入自适应平滑确保非凸对数求和正则化的近端算子连续，并推导相应的AMP状态演化，进一步验证了其在稀疏度低或测量率高的场景下优于LASSO。

**🔧 技术方法**

主要技术包括非凸对数求和正则化、自适应平滑、近端算子、近似消息传递(AMP)、状态演化(SE)、交替方向乘子法(ADMM)等。

**📊 数据集**

实验使用人工生成的高维高斯测量矩阵和稀疏高斯信号，噪声为零均值方差10^-2的高斯噪声。

**📈 对比分析**

通过比较AMP与SE的固定点、ADMM实验结果以及与传统ℓ1正则化的性能，在无噪声时对相位转移曲线进行比较；在有噪声时对MSE随正则化参数变化的U形曲线进行比较。结果表明，在信号稀疏度低或测量率高时，对数求和正则化在MSE上优于ℓ1；相反，在高稀疏度低测量率时，ℓ1更优。

**⚠️ 局限性**

局限性包括：仅在大样本极限下分析，实际有限维问题可能出现多重固定点；对噪声方差的影响未系统研究；未在真实数据上验证；自适应平滑参数的选择经验性强；对非凸优化的收敛性理论仍缺乏。

---

## 720. Composing diffusion priors with explicit physical context via generative Gibbs sampling

**arXiv ID:** 2605.10642 | [PDF](https://arxiv.org/pdf/2605.10642v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 721. PRISM: Generation-Time Detection and Mitigation of Secret Leakage in Multi-Agent LLM Pipelines

**arXiv ID:** 2605.10614 | [PDF](https://arxiv.org/pdf/2605.10614v1)

**作者:** Riya Tapwal `[一作]` (Indian Institute of Technology Mandi), Carsten Maple `[通讯]` (University of Warwick)

**通讯引用:** 9171 | [OpenAlex ID](https://openalex.org/A5080175512)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多代理LLM系统中凭证泄露的传播放大，提出实时检测与干预方法Prism。

**💡 创新点**

将凭证泄露建模为生成过程中的风险累积，利用生成动态（熵坍塌、logit集中）与结构信号（标识符模式）结合的多信号风险评估，实现按token实时干预。

**🔧 技术方法**

16维特征融合（信息理论、结构、行为、上下文）+逻辑回归风险评分，三色阈值控制；ZK-RC后置哈希检查；在多代理管道中嵌入Prism实例。

**📊 数据集**

2000任务人工构建的多代理凭证泄露基准，覆盖13攻击类别、3压力等级，包含约30,900真实凭证。

**📈 对比分析**

与后置过滤器、扫描器、提示/熵法、GBT、Span Tagger比较；Prism实现0%任务漏泄、F1=0.832、召回0.712、利用率0.893，显著优于Span Tagger（F1=0.719、15%漏泄）和GBT（F1=0.684、19%漏泄）。

**⚠️ 局限性**

基准为合成，未覆盖所有生产场景；白盒依赖token log‑prob，黑盒下仍有1.4%残留泄漏；未处理加密/碎片化泄露。

---

## 722. LITMUS: Benchmarking Behavioral Jailbreaks of LLM Agents in Real OS Environments

**arXiv ID:** 2605.10779 | [PDF](https://arxiv.org/pdf/2605.10779v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 723. AllocMV: Optimal Resource Allocation for Music Video Generation via Structured Persistent State

**arXiv ID:** 2605.10723 | [PDF](https://arxiv.org/pdf/2605.10723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 724. When Does Sparsity Help for k-Independent Set in Hypergraphs and Other Boolean CSPs?

**arXiv ID:** 2605.10778 | [PDF](https://arxiv.org/pdf/2605.10778v1)

**作者:** Timo Fritsch `[一作]` (Karlsruhe Institute of Technology), Julian Stieß `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了稀疏输入下寻找固定大小独立集和布尔约束满足问题的时间复杂度，并给出了精细的分类和条件最优算法；

**💡 创新点**

创新点在于揭示稀疏性对复杂度的非线性影响，提供了在不同稀疏度阈值下的阈值切换以及对混合弧度超图和二元约束的精细化分析；

**🔧 技术方法**

主要技术包括基于包含排除与稀疏证据的组合、矩阵乘法加速的k-团/三角检测、可重写的超图归约以及层次化的稀疏/密集分治；

**📊 数据集**

文中未使用实验数据集，所有结果均为理论复杂度分析与条件下界证明；

**📈 对比分析**

通过与k-团与3-均匀超图团假设等经典难点假设对比，证明在给定稀疏度范围内算法时间达到条件最优，性能优于朴素O(n^k)的指数级提升；

**⚠️ 局限性**

局限性在于对特殊稀疏度阈值（如γ∈{0,1,2}）的完整性尚未完全解决，且对高阶约束族的稀疏性影响仍存在一定分析空白。

---

## 725. GridProbe: Posterior-Probing for Adaptive Test-Time Compute in Long-Video VLMs

**arXiv ID:** 2605.10762 | [PDF](https://arxiv.org/pdf/2605.10762v1)

**作者:** Mohamed Eltahir `[一作]` (King Abdullah University of Science and Technology), Naeemullah Khan `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5059907674)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练‑无关的后验探测推理范式，用行列探测在 K×K 网格上生成问题条件重要性图，并通过图形统计自适应计算预算 M_eff，实现长视频 VLM 的子二次计算。

**💡 创新点**

创新点：①使用 VLM 的答案空间置信度作为重要性信号，摆脱编码器空间的限制；②通过行列探测仅需 2K 次前向传播获得全局重要性图；③引入形状驱动的闭式统计（skew 与 excess kurtosis）来自适应选择帧数，实现无循环、无代理的自适应计算。

**🔧 技术方法**

技术：行列探测、答案空间置信度、K×K 网格重要性映射、形状统计（skew、kurtosis）、闭式自适应 M_eff、两阶段推理（探测+聚焦）以及可选的图像拼接压缩。

**📊 数据集**

数据集：Video‑MME‑v2（多选 8 选项、3200 题）和 LongVideoBench（包含字幕）。

**📈 对比分析**

对比方法：单模型 2B/4B/8B 的单片推理、固定帧数 M=8 的训练‑无关选择器（MDP3）以及跨模型管线（2B 选择器 + 4B/8B 处理器）。实验显示：在 Video‑MME‑v2 上 M_eff 自动模式实现 3.36× TFLOPs 降低、平均准确率仅下降 1.6pp；在 LongVideoBench 上比 2B 单片 baseline 提升 0.9pp、计算 0.35×。跨模型 2B→8B 在两项指标均超越 2B baseline（+4.0pp、0.52× compute），且无需任何微调。

**⚠️ 局限性**

局限性：①仅适用于有限答案空间（多选），难以直接推广到开放式问答；②探测阶段在大 prompt（字幕量大）或小 K（<10）时占比上升；③跨模型方案会增加主机内存消耗；④形状统计对视频长度/密度的适应性仍待改进。

---

## 726. AdaPaD: Adaptive Parallel Deflation for PEFT with Self-Correcting Rank Discovery

**arXiv ID:** 2605.10741 | [PDF](https://arxiv.org/pdf/2605.10741v1)

**作者:** Barbara Su `[一作]` (Rice University), Anastasios Kyrillidis `[通讯]` (Rice University)

**通讯引用:** 1802 | [OpenAlex ID](https://openalex.org/A5024280658)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LoRA适配器中提出并行rank‑1分解算法，结合提前学习与按模块动态rank发现，实现自我纠正的参数高效微调。

**💡 创新点**

核心创新是：1）并行rank‑1 deflation能够在训练过程中逐步纠正早期分量的误差；2）提前学习让未激活分量在等待时自行预热；3）按模块动态rank发现将rank分配作为模型学习的输出，避免手工设定。

**🔧 技术方法**

使用的技术包括：并行rank‑1分解与自我纠正、ALS或因子梯度的rank‑1子问题、Wedin扰动理论、投影约束、EMA重要性评分、前向/后向同步通信及多GPU组件并行。

**📊 数据集**

实验数据集涵盖：DeBERTaV3‑base + GLUE（CoLA、RTE、MRPC、STS‑B、SST‑2、QNLI、QQP、MNLI）、Qwen3‑0.6B + SQuAD v1.1/v2 以及多GPU加速测试。

**📈 对比分析**

与固定rank LoRA、AdaLoRA、IncreLoRA、SoRA、dEBORA等自适应rank LoRA基线对比；GLUE平均分为89.34，领先AdaLoRA 89.03和IncreLoRA 88.99；在Qwen3‑0.6B SQuAD/SQuAD v2中参数平均缩小30.7%，F1/EM与固定rank LoRA持平；在4×H200多GPU上实现最高2.66×的速度提升。

**⚠️ 局限性**

主要局限包括：理论仅针对bilinear回归模型，对非线性LoRA损失的收敛尚需经验验证；当分量rank趋近于0时收敛速率下降；同步通信开销在极大rank或极大模型规模时可能抵消加速效果；未来工作需扩展到异步通信、更多重要性评分以及更大规模模型。

---

## 727. ChatGPT: Friend or Foe When Comprehending and Changing Unfamiliar Code

**arXiv ID:** 2605.10702 | [PDF](https://arxiv.org/pdf/2605.10702v1)

**作者:** Norman Anderson `[一作]` (University of Victoria), Margaret-Anne Storey `[通讯]` (University of Victoria)

**通讯引用:** 9482 | [OpenAlex ID](https://openalex.org/A5038905934)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验室中对10名高级学生开发者进行对照实验，研究大型语言模型（ChatGPT）在进行大型代码系统非平凡扩展时如何影响其问题求解行为、进度及被卡住的情况，并利用波利亚的四阶段问题求解模型和25个细粒度行为代码进行质性分析。

**💡 创新点**

首次将波利亚的框架与软件工程中的代码变更任务相结合，系统性地定义并归纳了七类“卡住”时刻及其对应的AI干预效果，揭示AI既能帮助也能阻碍开发者从卡住状态恢复，并提出了针对AI辅助的工具与教育改进建议。

**🔧 技术方法**

主要技术包括：ChatGPT（gpt‑4o）交互、思考大声（think‑aloud）记录、屏幕录制与音频、代码差异追踪、结构化行为编码（定性），以及基于波利亚模型的阶段划分。

**📊 数据集**

使用一套约35k行TypeScript的多仓库代码基（Next.js/React 前端 + Express 后端 + SQLite），该代码基模仿真实企业级项目，并包含若干未完成的功能点；并未使用公开数据集，仅使用实验任务所提供的代码库。

**📈 对比分析**

通过对比AI组与无AI组在完成率、文件保存次数、每次保存的LOC量、任务总时长、代码理解评分等指标进行比较。结果显示AI组完成率显著更高（4/5 vs 1/5），文件保存次数减少、单次修改量增大，表明AI能加速实现阶段；但在整体时间占比上两组相似，且AI未显著改变在波利亚四阶段的时间分配。

**⚠️ 局限性**

局限性包括：样本量仅10名学生，实验环境为受控实验室，且所有参与者均为大学生而非经验丰富的专业开发者；仅评估ChatGPT一种LLM，无法推广到其他生成式工具；实验任务为人工设计且带有时间限制，可能影响行为的真实性；质性编码受研究者主观影响。

---

## 728. The Bystander Effect in Multi-Agent Reasoning: Quantifying Cognitive Loafing in Collaborative Interactions

**arXiv ID:** 2605.10698 | [PDF](https://arxiv.org/pdf/2605.10698v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 729. Is Data Shapley Not Better than Random in Data Selection? Ask NASH

**arXiv ID:** 2605.10684 | [PDF](https://arxiv.org/pdf/2605.10684v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 730. When 'For You' Isn't For You: Measuring User Agency in TikTok's Algorithmic Feed

**arXiv ID:** 2605.10690 | [PDF](https://arxiv.org/pdf/2605.10690v1)

**作者:** Levi Kaplan `[一作]` (Northeastern University), Piotr Sapiezynski `[通讯]` (Northeastern University)

**通讯引用:** 2967 | [OpenAlex ID](https://openalex.org/A5033574079)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建基于安卓模拟器的实验框架，系统地探究TikTok For You Page（FYP）算法在用户显式和隐式反馈下的可控性，验证用户的“感兴趣”与“拒绝”行为对推荐内容的影响；

**💡 创新点**

创新点包括：1) 为TikTok移动端开发完整的审计工具链（模拟器控制、UI自动化、网络抓包、TLS破除、protobuf逆向、签名生成）；2) 引入“账户克隆”技术，使实验账号拥有完全相同的观看/跳过历史，能够对照实验；3) 在三类不同影响力的主题（烹饪、健身、体育博彩）上进行大规模控制实验，首次量化显式/隐式信号对内容推送的效力及“复发”现象；

**🔧 技术方法**

使用的技术包括：Android Studio模拟器、ADB、UIAutomator2、Mitmproxy（MITM）拦截网络、Frida Hook自定义签名、Square Wire反编译生成protobuf、Zstandard自定义压缩解密、ChatGPT 3.5 Turbo进行视频主题分类；

**📊 数据集**

数据集为15轮实验，每轮涉及2个实验账号（显式/隐式）在3个主题上各观看/跳过200条FYP视频，总计9,000条视频数据；此外使用公开的TikTok视频描述、标签、用户信息进行ChatGPT分类；

**📈 对比分析**

比较方法：对每种信号类型统计所见主题视频比例，采用两比例Z检验（99%置信水平）判断差异显著性。结果显示：显式“不感兴趣”能将主题视频比例从约30%下降至4.75%（约84%下降），显式效果普遍优于隐式；但在“复发”阶段，隐式信号更易被算法重新推送，显式复发率亦出现；

**⚠️ 局限性**

局限性：1) 仅模拟观看/跳过/标记“不感兴趣”，未涵盖点赞、分享、评论、关注等常见交互；2) 实验账号不受地理位置、历史浏览等因素影响；3) 仅在单一时间点对算法进行测评，未覆盖算法迭代；4) 结果可能与真实用户体验差异，尤其是对显式信号效果的感知。

---

## 731. Locking Pretrained Weights via Deep Low-Rank Residual Distillation

**arXiv ID:** 2605.10777 | [PDF](https://arxiv.org/pdf/2605.10777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 732. DynaMiCS: Fine-tuning LLMs with Performance Constraints using Dynamic Mixtures

**arXiv ID:** 2605.10770 | [PDF](https://arxiv.org/pdf/2605.10770v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 733. Kernel-Gradient Drifting Models

**arXiv ID:** 2605.10727 | [PDF](https://arxiv.org/pdf/2605.10727v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 734. Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish

**arXiv ID:** 2605.10714 | [PDF](https://arxiv.org/pdf/2605.10714v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 735. Beyond the Last Layer: Multi-Layer Representation Fusion for Visual Tokenizatio

**arXiv ID:** 2605.10780 | [PDF](https://arxiv.org/pdf/2605.10780v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 736. Matching-with-Contracts for the AI-RAN Market: AIGC-as-a-Service for Teleoperation

**arXiv ID:** 2605.10751 | [PDF](https://arxiv.org/pdf/2605.10751v1)

**作者:** Zijun Zhan `[一作]` (University of Houston), Zhu Han `[通讯]` (University of Houston)

**通讯引用:** 91171 | [OpenAlex ID](https://openalex.org/A5063667378)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了一个基于匹配与合约的激励机制，用于解决 AI‑RAN 服务市场中信息不对称与竞争问题，提出混合稳定匹配与合约算法实现动态合约更新与用户匹配。

**💡 创新点**

创新点在于：①将多方竞争合约设计、用户匹配和市场状态演化统一到动态匹配‑合约框架；②利用 Chernoff 上界近似三阶段排队模型的延迟违约概率，保持合约优化的凸性；③设计混合稳定匹配算法，采用影子价格和温度退火实现用户侧与运营商侧的交替更新，从而实现全局平衡。

**🔧 技术方法**

技术手段包括：合同理论（诱导自揭露、IC/IR约束）、匹配理论（混合稳定匹配、Gale‑Shapley类方法）、排队理论（M/M/c 3 阶段模型）、Chernoff 置信上界、非线性凸优化与迭代逼近、以及基于仿真评估的多指标性能对比。

**📊 数据集**

数据集与实验环境：使用 Unity‑based 远程操控 AIGC 任务仿真，任务参数为 d_i=0.18Mb、τ=3.6×10¹¹ FLOP、d_o=0.27Mb，用户生成速率 δ=24/s，用户类型按 Dirichlet 分布生成，实验主要在模拟平台上进行，不依赖公开真实数据集。

**📈 对比分析**

比较方法：对比传统合同理论（CT）、静态匹配‑合约（MC）和基于 Gale‑Shapley 的匹配‑合约（GSMC）。实验显示，在高负载、用户异质性强或可靠性惩罚大时，本方法可将运营商总效用提升至少 56.8%（在 90 用户情景）或 89.6%（在 8/12 类别情景），社会福利提升至少 51.7%/63.1%。

**⚠️ 局限性**

局限性：①算法依赖于排队模型和 Chernoff 近似，若真实网络行为与假设不符，估计误差可能增大；②计算量随用户数和类型数呈二次或以上增长，可能对大规模部署有挑战；③实验仅基于仿真，缺乏真实网络部署验证；④假设用户类型可通过数据挖掘准确分层，实际环境中可能存在更复杂的类型分布。

---

## 737. iPay: Integrated Payment Action Recognition via Multimodal Networks and Adaptive Spatial Prior Learning

**arXiv ID:** 2605.10732 | [PDF](https://arxiv.org/pdf/2605.10732v1)

**作者:** Kaicong Huang `[一作]` (Rensselaer Polytechnic Institute), Ruimin Ke `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 5130 | [OpenAlex ID](https://openalex.org/A5049143775)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为iPay的集成支付动作识别框架，结合RGB和骨骼信息在公交车监控视频中自动识别现金、二维码、刷卡、触碰等支付方式，支持边缘部署。

**💡 创新点**

创新点包括：① 多模态混合专家架构，四个紧耦合流（RGB专家、骨骼专家、双注意力融合、空间差分鉴别器）实现跨模态互补；② 空间差分鉴别器自动学习手部与支付终端的相对运动锚点，显式引入任务先验；③ 通过ST‑ROI裁剪局部动作区域，提升低质量监控视频的局部细节提取；④ 构建真实公交监控支付动作数据集。

**🔧 技术方法**

使用技术包括：图卷积网络（DeGCN）骨骼时空建模；ResNet‑18提取局部RGB特征；双向跨模注意力融合；时间卷积网络（TCN）对空间差分特征建模；SAM3 3D人体分割提取高质量骨骼；ST‑ROI拼接实现2D CNN时空处理；多模态联合训练与多模融合。

**📊 数据集**

数据集：与当地公交公司合作收集55小时监控视频，标注了569条支付动作，涵盖二维码、现金、逃票、刷卡、触碰及其它类别，按70%/30%划分为训练/测试。

**📈 对比分析**

与ST‑GCN、CTR‑GCN、InfoGCN、ML‑STGNet、DeGCN等现有骨骼/视频方法对比，iPay 2‑stream在平均识别率上达到83.45%，相较基线提升约10–23%，同时保持6.34G FLOPs，参数约13.55M，显著提升精度且计算效率可用于边缘设备。

**⚠️ 局限性**

局限性：依赖手部与终端的相对运动，仍受极端遮挡、反射、运动模糊影响；数据集规模有限，类别分布不均；目前仅验证在公交车内部监控，跨场景泛化需进一步研究。

---

## 738. On Improving Graph Neural Networks for QSAR by Pre-training on Extended-Connectivity Fingerprints

**arXiv ID:** 2605.10722 | [PDF](https://arxiv.org/pdf/2605.10722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 739. The Agent Use of Agent Beings: Agent Cybernetics Is the Missing Science of Foundation Agents

**arXiv ID:** 2605.10754 | [PDF](https://arxiv.org/pdf/2605.10754v1)

**作者:** Xinrun Wang `[一作]` (Singapore Management University), Shuyue Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 448 | [OpenAlex ID](https://openalex.org/A5052387391)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过将控制论六条经典法则映射到LLM基础代理的设计原则，提出了 Agent Cybernetics 框架，并归纳出可靠性、长期运行与自我改进三大工程目标。

**💡 创新点**

创新点在于将控制论原理系统化地应用于基础代理，形成可验证的设计原则与目标，为代理设计提供理论基础，弥补现有经验式工程的空白。

**🔧 技术方法**

技术手段主要是理论分析与案例归纳，结合工具循环、记忆库、反射步骤等现有代理构件，构建六条原则与三项目标。

**📊 数据集**

由于本工作为理论与综述性论文，未使用特定实验数据集；主要引用公开的代理案例与相关文献进行讨论。

**📈 对比分析**

比较方法采用对三大应用域（代码生成、计算机使用、自动化研究）中常见失败模式与工程模式的对照分析，展示框架的适用性，但未给出量化性能指标。

**⚠️ 局限性**

局限性包括缺乏实验验证与实现细节、对原则落地的具体算法尚待研究，以及对不同任务复杂度与资源约束的适应性评估不足。

---

## 740. MATRA: Modeling the Attack Surface of Agentic AI Systems -- OpenClaw Case Study

**arXiv ID:** 2605.10763 | [PDF](https://arxiv.org/pdf/2605.10763v1)

**作者:** Tim Van hamme `[一作]` (KU Leuven), Dinil Mon Divakaran `[通讯]` (A*STAR Institute for Infocomm Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出并实现了一套针对自主 AI 系统的威胁建模与风险评估框架，利用资产影响评估、数据流图和攻击树来量化部署特定的安全风险。

**💡 创新点**

创新点在于将传统 NIST 风险评估方法与 LLM 代理特有的提示注入、工具调用与持续记忆等路径相结合，形成了面向影响的、可视化的攻击树结构，并提供了从最易入侵到全攻击面两种风险量化视角。

**🔧 技术方法**

采用的技术包括：资产与业务影响映射、数据流图（DFD）分析、攻击树构建、半定量风险评分、Docker 沙箱、最小权限配置、输出消毒等。

**📊 数据集**

实验基于开源个人 AI 代理 OpenClaw 的真实部署架构以及 OWASP、MITRE、NIST 等公开威胁目录进行案例分析，并未使用传统机器学习数据集。

**📈 对比分析**

通过对比在默认部署与启用 Docker 沙箱两种配置下的风险评分，展示了安全控制对风险的显著降低：从“非常高（9）”下降到“中等（3）”，证明框架能指导有效的防御措施。

**⚠️ 局限性**

局限性包括：风险评估高度依赖分析师知识与主观判断，可能低估多路径交互的风险；仅针对单一代理架构测试，未覆盖多代理互通；缺乏针对隐私特定模型（如 LINDDUN）的评估；整体仍为半定量方法，缺乏严格的统计校准。

---

## 741. VRA: Grounding Discrete-Time Joint Acceleration in Voltage-Constrained Actuation

**arXiv ID:** 2605.10696 | [PDF](https://arxiv.org/pdf/2605.10696v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 742. Towards a Large Language-Vision Question Answering Model for MSTAR Automatic Target Recognition

**arXiv ID:** 2605.10772 | [PDF](https://arxiv.org/pdf/2605.10772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 743. C-CoT: Counterfactual Chain-of-Thought with Vision-Language Models for Safe Autonomous Driving

**arXiv ID:** 2605.10744 | [PDF](https://arxiv.org/pdf/2605.10744v1)

**作者:** Kefei Tian `[一作]` (Tongji University), Shen Li `[通讯]` (Tsinghua University)

**通讯引用:** 15734 | [OpenAlex ID](https://openalex.org/A5100768717)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 C-CoT 框架，将视觉‑语言模型与多阶段 Chain‑of‑Thought 反事实推理结合，实现安全驾驶决策；

**💡 创新点**

创新性引入结构化 meta‑action 评估树，显式建模动作与安全结果的因果关系，提升长尾与异常场景的鲁棒性；

**🔧 技术方法**

基于 Qwen2.5‑VL 大模型，采用 LoRA 微调、结构化多阶段推理与反事实评估树；

**📊 数据集**

使用 DeepAccident‑CCoT 数据集（基于 DeepAccident，包含 2496 条带有五阶段标注的交叉路口场景）；

**📈 对比分析**

与 LLaVA‑1.5、Llama‑3.2‑Vision、InternVL‑2.5 等基线在同一数据集下对比，C‑CoT 语义准确率 84.2%，风险召回 81.9%，L2@3s 1.98 m，碰撞率 3.52%，均显著优于基线；

**⚠️ 局限性**

局限在于仅针对纵向加减速动作设计，未覆盖横向变道；评估树计算量大，未来需通过剪枝或近似提升效率。

---

## 744. Cybercrime and Prevention: Colonel Blotto in Social Engineering

**arXiv ID:** 2605.10755 | [PDF](https://arxiv.org/pdf/2605.10755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 745. Charting the Diameter Computation Landscape on Intersection Graphs in the Plane

**arXiv ID:** 2605.10692 | [PDF](https://arxiv.org/pdf/2605.10692v1)

**作者:** Timothy M. Chan `[一作]` (University of Illinois at Urbana-Champaign), Da Wei Zheng `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文主要研究二维几何交叉图中直径的计算问题，并对多种几何对象（线段、单位正方形、单位圆盘、等腰三角形等）给出了既有上界也有下界，揭示了对象类型、直径大小以及相交方式共同决定计算复杂度的细致地形图；

**💡 创新点**

创新点包括：① 针对线段的 VC‑维度新证明，利用颜色对齐技术将传统的 K5‑避免方法推广到非伪圆盘；② 对非退化轴对齐线段实现真子二次时间，揭示退化是难度来源；③ 通过分治+高维正交范围搜索构造 O^*(n) 的单位正方形直径算法；④ 通过伪圆盘与 1‑邻域的弱伪圆盘性质，给出单元圆盘直径 2 的 O(n^{4/3}) 算法；⑤ 证明在 3‑坡线段、脂三角形和具有 O(log n) 复杂度的字符串图上直径 2 的问题为真子二次难；

**🔧 技术方法**

技术主要包括：VC‑维度分析、颜色对齐与 K5‑避免、正交范围搜索（在维度随直径增长的高维空间中）、弱伪圆盘与花瓣（flower）隐式表示、射线投射与二分搜索、以及多层次分治（基于坐标 modulo 1）等；

**📊 数据集**

本文没有使用实际数据集，所有结果均为理论性上界/下界，基于计算几何模型和精细化复杂性假设（OV、3‑H6、combK4 等）；

**📈 对比分析**

与之前工作相比，算法复杂度得到显著提升：轴对齐线段非退化 O(n^2-1/32) 取代原 O(n^2) 下界；单位正方形常数直径 O^*(n) 取代 O(n^{7/4})；单位圆盘直径 2 O(n^{4/3}) 取代 O(n^{2-1/18})；三角形直径 2 的近似 n^2 下界提升至与 3‑H6 相关的超二次；

**⚠️ 局限性**

局限性包括：① 仍无法在一般情况下实现线性或几乎线性时间（仅在特殊对象/直径范围内）；② 许多下界基于组合化或细化复杂性假设，实际可否在更弱假设下保持；③ 高维（≥3）问题仍主要在伴随论文中讨论，二维结果难以直接推广；④ 对单位圆盘直径 2 的 O(n^{4/3}) 算法尚未突破到真正线性；

---

## 746. Scalable Mamba-Based Message-Passing Neural Decoder for Error-Correcting Codes

**arXiv ID:** 2605.10681 | [PDF](https://arxiv.org/pdf/2605.10681v1)

**作者:** Rostislav Gusev `[一作]` (Skolkovo Institute of Science and Technology), Dmitry Artemasov `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5055635639)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无注意力的Mamba消息传递神经解码器（MMPD），用于二进制线性码的前向纠错。

**💡 创新点**

创新点在于将Tanner图的局部对偶聚合与双向Mamba状态空间块相结合，实现了无注意力的全局信息传播，显著降低了内存和计算开销，并在长码上实现了超越现有最佳解码器的性能。

**🔧 技术方法**

采用了Tanner图消息传递框架、对偶边缘特征聚合、门控残差更新、双向Mamba状态空间块、以及BCE训练目标，所有组件均为纯深度学习实现。

**📊 数据集**

使用了标准LDPC基准码（(384,320)、(529,440)、(648,540)、(1056,880) WiMAX LDPC）以及5G LDPC 11/12率码进行实验，数据通过AWGN通道生成。

**📈 对比分析**

在相同参数量和训练协议下与BP、ECCT、AECCT、CrossMPT等模型对比，MMPD在- ln(BER)指标上表现最佳；在长码实验中，MMPD的BER曲线位于BP-5与BP-50之间，且内存占用比注意力模型低约1.5倍，随码长线性增长。

**⚠️ 局限性**

局限性包括：缺乏对更大码长和更复杂码结构的验证；训练需要针对特定码进行，泛化能力尚未彻底评估；理论复杂度分析仍待补充。

---

## 747. RadThinking: A Dataset for Longitudinal Clinical Reasoning in Radiology

**arXiv ID:** 2605.10761 | [PDF](https://arxiv.org/pdf/2605.10761v1)

**作者:** Wenxuan Li `[一作]` (Johns Hopkins University), Zongwei Zhou `[通讯]` (Johns Hopkins University)

**通讯引用:** 20159 | [OpenAlex ID](https://openalex.org/A5084104975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文构建了一套面向多癌筛查的视觉问答（VQA）数据集，并提供了从感知到多步推理的三层难度体系；

**💡 创新点**

创新点在于：①将临床报告标准转化为可直接使用的分层推理链；②每个复合 VQA 都附带可验证的推理步骤；③为强化学习提供可度量的奖励信号；

**🔧 技术方法**

主要技术包括 3D CT 预处理、报告解析、基于掩模的特征抽取、层级式 VLM 训练（SFT+RL）以及规则驱动的链式推理；

**📊 数据集**

使用数据集来自欧洲多中心的 CT 扫描（约 16,000 张）、对应的放射学报告、临床变量和病理确诊，涵盖 43 种癌症分组和 1,826 名测试患者；

**📈 对比分析**

文章未给出具体实验结果，建议使用分层准确率和复杂度维度评估；现阶段性能未公开，但框架已为后续模型训练与评估奠定基础；

**⚠️ 局限性**

局限性包括：仅使用 CT（未覆盖乳腺/MRI 等影像）；链条只编码结构化信息，缺乏自由叙述；依赖已有报告解析，可能忽略非标准用词；健康对照随访时间有限；缺乏完整的统计显著性检验。

---

## 748. Provable Sparse Inversion and Token Relabel Enhanced One-shot Federated Learning with ViTs

**arXiv ID:** 2605.10748 | [PDF](https://arxiv.org/pdf/2605.10748v1)

**作者:** Li Shen `[一作]` (Sun Yat-Sen University), Xun Yang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 50272 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种单轮联邦学习框架 FedMITR，利用模型反演生成合成图像，并通过稀疏反演和 token 重新标注的方式训练全局模型。

**💡 创新点**

创新点在于：①稀疏模型反演仅逆转语义前景，过滤掉无关背景；②对低信息密度 token 采用集成模型重新标注并蒸馏；③通过算法稳定性分析证明稀疏反演和 token 重新标注可显著降低梯度 Lipschitz 常数，从而提升泛化性能；④在 Vision Transformer 上实现全图像生成并充分利用所有 patch。

**🔧 技术方法**

使用技术包括 Vision Transformer（ViT）模型、稀疏模型反演、token 重新标注与集成蒸馏、数据‑free 生成器、算法稳定性分析、JS 与 KL 损失等。

**📊 数据集**

实验数据集涵盖 CIFAR10、CIFAR100、OfficeHome 和 Mini‑ImageNet，使用 Dirichlet 分布产生不同程度的非 IID 设定。

**📈 对比分析**

与 FedAvg、FedFTG、DENSE、Co‑Boosting、DeepInversion 等基线对比，FedMITR 在 Dirichlet α 为 0.01、0.05、0.1、0.3、0.5 等极端异质场景下，准确率提升约 3–9% 甚至更高；在极端异质（每个客户端仅 1 或 3 类）下提升更为显著；且仅需一次通信即可达到甚至超过多轮 FedAvg 的表现，显著降低通信开销。

**⚠️ 局限性**

局限性包括：在低异质或标签均匀分布时性能略低于 DeepInversion；对超参数（如 λ1、λ2、掩码比例）敏感；依赖预训练 ViT，迁移到非图像任务或大规模模型仍需验证；目前只评估单轮通信，未系统研究对攻击、隐私泄露等安全问题的鲁棒性。

---

## 749. Qwen-Image-2.0 Technical Report

**arXiv ID:** 2605.10730 | [PDF](https://arxiv.org/pdf/2605.10730v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 750. XQCfD: Accelerating Fast Actor-Critic Algorithms with Prior Data and Prior Policies

**arXiv ID:** 2605.10734 | [PDF](https://arxiv.org/pdf/2605.10734v1)

**作者:** Daniel Palenicek `[一作]` (Technical University of Darmstadt), Jan Peters `[通讯]` (Technical University of Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于SAC的样本高效actor-critic算法，结合BC预训练、KL正则化和Stationary（HetStat）网络，在稀疏奖励操纵任务中实现对专家演示的高效利用与持续提升。

**💡 创新点**

创新点在于将stationary特征网络与KL正则化相结合，既保持BC策略的初始高性能，又在超出演示分布的状态下提供最大熵探索，从而避免了BC策略被快速忘记。

**🔧 技术方法**

采用SAC为基础的样本高效actor-critic，使用批归一化、权重归一化、分布式Critic、KL正则化以及HetStat stationary网络架构。

**📊 数据集**

使用Adroit、Robomimic和MimicGen三个基准的11个稀疏奖励操纵任务的专家演示数据。

**📈 对比分析**

与BC、RLPD、SAC等基线对比，在所有任务上均实现最高成功率，尤其在最困难的MimicGen任务中首次突破零成功率，并在样本效率上提升数十倍。

**⚠️ 局限性**

仅针对专家演示数据，未评估次优离线数据；KL正则化温度固定，可能导致对BC质量过度或不足的正则化。

---

## 751. Heteroscedastic Diffusion for Multi-Agent Trajectory Modeling

**arXiv ID:** 2605.10717 | [PDF](https://arxiv.org/pdf/2605.10717v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 752. UAV-Assisted Scan-to-Simulation for Landslides Using Physics-Informed Gaussian Splatting

**arXiv ID:** 2605.10715 | [PDF](https://arxiv.org/pdf/2605.10715v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 753. RelFlexformer: Efficient Attention 3D-Transformers for Integrable Relative Positional Encodings

**arXiv ID:** 2605.10706 | [PDF](https://arxiv.org/pdf/2605.10706v1)

**作者:** Byeongchan Kim `[一作]` (Seoul National University), Krzysztof Choromanski `[通讯]` (Google DeepMind)

**通讯引用:** 2713 | [OpenAlex ID](https://openalex.org/A5031842812)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种新的3D Transformer模型RelFlexformer，利用非均匀快速傅里叶变换（NU-FFT）实现可插值的相对位置编码（RPE）并保持O(L log L)的注意力计算复杂度；

**💡 创新点**

创新点在于：①将任意可积的RPE函数与NU-FFT结合，提供统一且可学习的几何调制方式；②在保持kernel化线性注意力的同时，支持非结构化、异质点云的全局RPE；③通过快速矩阵-向量乘法实现高效的掩码应用；

**🔧 技术方法**

核心技术包括Performer/线性注意力、非均匀快速傅里叶变换、随机特征映射、点云/RGB‑D张量化与点位置编码（PointRoPE、STRING等）；

**📊 数据集**

在多种3D数据集上评估：ModelNet40、ScanObjectNN、ScanNet/ScanNet200/ScanNet++、nuScenes、S3DIS、NYU Depth v2、SUN RGB‑D；

**📈 对比分析**

与标准Transformer、Performers、Performers+PointRoPE、Performers+STRING等进行对比，RelFlexformer在大多数任务上显著提升mIoU/accuracy，常逼近或超过全连接Transformer，并在ScanObjectNN、S3DIS、nuScenes、ScanNet++等数据集上实现了最高分；

**⚠️ 局限性**

局限性包括：①仍需手动或自动选择RPE函数与采样频率（quadrature size）以获得最佳性能；②相较于纯线性注意力，NU-FFT实现增加了实现复杂度和内存开销；③在极大规模点云（数十万点）下，O(L log L)虽然优于O(L²)，但实际吞吐量仍受FFT实现与GPU内存带宽限制。

---

## 754. TransmissiveGS: Residual-Guided Disentangled Gaussian Splatting for Transmissive Scene Reconstruction and Rendering

**arXiv ID:** 2605.10705 | [PDF](https://arxiv.org/pdf/2605.10705v1)

**作者:** Zhenyu Liang `[一作]` (Hong Kong University of Science and Technology), Chi-Keung Tang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 13219 | [OpenAlex ID](https://openalex.org/A5062566088)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种双高斯表示和残差引导的分离策略，能够同时重建透射面几何与反射与透射光线的光照；

**💡 创新点**

创新点在于将透射场拆分为反射高斯和散射高斯，并通过残差引导实现几何与光照的解耦；

**🔧 技术方法**

使用二维高斯喷射（2DGS）作为基础，结合反射光场MLP、分层光照、Fresnel项以及高频正则化；

**📊 数据集**

在新构建的七个合成Blender场景以及四个真实世界场景（compact、hatchback、facade、glazing）上进行训练与测试；

**📈 对比分析**

与3DGS、2DGS、GaussianShader、Ref-Gaussian、Ref-GS、EnvGS等基线对比，TransmissiveGS在PSNR、SSIM、LPIPS以及透射面MAE指标上均取得显著提升；

**⚠️ 局限性**

目前仅适用于静态透射与反射场景，无法处理动态反射或透射物体的变化。

---

## 755. DANCE: Detect and Classify Events in EEG

**arXiv ID:** 2605.10688 | [PDF](https://arxiv.org/pdf/2605.10688v1)

**作者:** Jarod Lévy `[一作]` (Meta AI), Stéphane d'Ascoli `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个名为DANCE的端到端框架，用于在不依赖事件对齐信息的情况下从原始EEG信号中检测和分类异构时间事件。

**💡 创新点**

创新点在于将EEG事件检测视为集合预测问题，结合CNN特征提取、Perceiver时序重采样和Transformer解码器，并使用稠密与稀疏损失以及一致性正则化实现双重监督。

**🔧 技术方法**

核心技术包括卷积骨干网络（多尺度时序卷积+空间注意）、Perceiver模块进行可学习的时序下采样、Transformer解码器进行事件集合预测，以及交叉熵+IoU损失、Hungarian匹配和KL一致性正则。

**📊 数据集**

在十个公开数据集上评估，覆盖打字、癫痫发作、听力（语音）、运动想象、P300与伪迹等六大任务，总计1154名受试者、235万事件。

**📈 对比分析**

与七种基线（随机、ATCNet、USleep、LaBraM、CBraMod、REVE等）以及两种消融版本比较，DANCE在事件级F1达到0.397（相较最佳基线提升三倍），样本级F1为0.521；在癫痫监测上F1为62.1%，高于现有方法。

**⚠️ 局限性**

局限包括对极短事件（如语音音素、键击）仍表现不佳；模型尚未从跨数据集联合训练中获益；低信噪比和高事件密度导致精确定位受限；未能整合语言先验或更高SNR模态。

---

## 756. A Performance-Portable, Massively Parallel Distributed Nonuniform FFT

**arXiv ID:** 2605.10678 | [PDF](https://arxiv.org/pdf/2605.10678v1)

**作者:** Paul Fischill `[一作]` (ETH Zuerich), Sriramkrishnan Muralikrishnan `[通讯]` (Forschungszentrum Juelich Gmbh)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文实现了首个可在多节点 GPU 集群上并行、跨平台（NVIDIA 与 AMD）运行的分布式非均匀 FFT (NUFFT)，并将其集成到 IPPL 框架中。

**💡 创新点**

创新点在于：① 通过 Kokkos 提供的性能可移植性实现统一的 GPU 后端；② 设计多种基于 ES kernel 的扩散/插值核（Atomic、Tiled、Grid-Parallel、Sorted），显著降低原子冲突和内存带宽瓶颈；③ 引入分布式 pruned FFT 以削减不必要的 FFT 计算与通信；④ 采用贝叶斯优化自动调优硬件特定参数。

**🔧 技术方法**

技术手段包括 Kokkos（GPU 代码可移植）、MPI（分布式并行）、HeFFTe（高效分布式 FFT）、IPPL（粒子–网格框架）、ES kernel、贝叶斯优化、CUDA/HIP、cuFFT/rocFFT。

**📊 数据集**

实验数据集主要是 3D 3V 朗道阻尼（Landau damping）模拟：512³ 与 1024³ Fourier 模式，对应 1.07×10⁹ 至 8.59×10⁹ 粒子，粒子数密度 ρ=8，NUFFT 容差 ε=10⁻⁴ 与 10⁻⁸。

**📈 对比分析**

与 cuFINUFFT 的对比表明：在 NVIDIA GPU 上，Grid-Parallel 扩散核在大多数容差下可与或超过 cuFINUFFT；在 AMD GPU 上仅 cuFINUFFT 无法使用。多 GPU 扩展性良好，Alps、JUWELS、LUMI 上均能达到 1024 个 GPU，pruned FFT 在 Cray MPICH + Slingshot（Alps、LUMI）效果与全 FFT 差距不大，且在 Open MPI + InfiniBand（JUWELS）反而更慢。整体性能可观，单核吞吐率在 10⁻²–10⁻⁴ 容差下达到数千 Mpts/s。

**⚠️ 局限性**

局限性包括：HIP 编译器在 AMD 上的寄存器分配不佳导致性能低于 CUDA；pruned FFT 在不同 MPI/互连环境下表现不一致；HeFFTe 在 LUMI 上偶发崩溃；高精度（ε=10⁻⁸）在 AMD 上仍显著昂贵；FFT 仍是大规模运行的主要瓶颈，未来工作需进一步改进分布式 FFT 与通信。

---

## 757. On Distributed Parallelization Strategies for Particle-in-Fourier Schemes

**arXiv ID:** 2605.10729 | [PDF](https://arxiv.org/pdf/2605.10729v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 758. Dynamic Cross-Modal Prompt Generation for Multimodal Continual Instruction Tuning

**arXiv ID:** 2605.10765 | [PDF](https://arxiv.org/pdf/2605.10765v1)

**作者:** Tao Hu `[一作]` (Nanjing University), Da-Wei Zhou `[通讯]` (Nanjing University)

**通讯引用:** 1957 | [OpenAlex ID](https://openalex.org/A5100655948)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究多模态持续指令调优，提出实例级跨模态提示生成框架，解决任务间与任务内的灾难性遗忘。

**💡 创新点**

通过跨模态注意力实时合成输入特定软提示，并结合空洞空间梯度投影与CLIP原型路由，实现无回放的持续学习。

**🔧 技术方法**

使用软提示生成器（基于多头跨模态注意）、null‑space 梯度投影、CLIP嵌入原型路由以及LoRA/提示调优等技术。

**📊 数据集**

在CoIN（8个顺序VQA任务）和UCIT（6个顺序任务）数据集上进行实验评测。

**📈 对比分析**

与CODA-Prompt、MoELoRA、ProgLoRA等基线对比，平均准确率分别达到67.48%（CoIN）和69.41%（UCIT），显著优于最强对照组。

**⚠️ 局限性**

仅在提示调优范式验证，未扩展至LoRA等其他轻量化更新；对跨任务分布漂移的鲁棒性尚待进一步评估。

---

## 759. AutoSOUP: Safety-Oriented Unit Proof Generation for Component-level Memory-Safety Verification

**arXiv ID:** 2605.10712 | [PDF](https://arxiv.org/pdf/2605.10712v1)

**作者:** Paschal C. Amusuo `[一作]` (Purdue University), James C. Davis `[通讯]` (Purdue University)

**通讯引用:** 2815 | [OpenAlex ID](https://openalex.org/A5004592401)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 AutoSOUP，一种自动化构建组件级内存安全单元证明（Unit Proof）的系统，旨在通过安全导向的验证选择实现可验证的内存安全保证。

**💡 创新点**

创新点包括：①安全导向的单元证明框架，关注验证所需的安全相关行为而非完整程序语义；②三种基于资源、属性与上下文的自动化技术；③将 LLM 以“功能调用”方式嵌入确定性工作流，兼顾可解释性与可靠性。

**🔧 技术方法**

技术方法主要是：资源感知范围扩展（Scope Widening）、属性驱动循环界限细化（Loop‑Bound Refinement）以及上下文感知环境模型细化（Environment Refinement）；以及基于 OpenAI/LiteLLM 的 LLM‑as‑Function‑Call 架构来完成有限任务并进行验证。

**📊 数据集**

数据集包括四个大型嵌入式实时操作系统（Zephyr、RIOT‑OS、Contiki‑NG、FreeRTOS），共177个入口点和60个重现的 CVE；还使用了 177 个随机挑选的函数作为通用评测。

**📈 对比分析**

与基线 Codex（通用 AI 编码器）和 Seeker（现有内存安全验证器）对比，AutoSOUP 在有效证明率上达 93%（Scope‑1）/89.5%（Scope‑2），在 CVE 暴露率上分别为 66.7%/65%，显著高于 Codex 的 28.3% 与 Seeker 的 41.7%；生成成本约 3–6 美元，生成时间 2–6 小时，验证覆盖率和错误曝光率均优于对照。

**⚠️ 局限性**

局限性包括：①对 LLM 生成结果的依赖，若模型失误仍需人工验证；②静态分析在间接调用、循环不确定性等方面的模糊度，可能导致遗漏或误导；③范围扩展采用文件粒度，过大文件会导致资源超限；④CBMC 在结构化内存访问和自定义释放器等语义上的不足，限制了漏洞检测的完整性。

---

## 760. MPerS: Dynamic MLLM MixExperts Perception-Guided Remote Sensing Scene Segmentation

**arXiv ID:** 2605.10769 | [PDF](https://arxiv.org/pdf/2605.10769v1)

**作者:** Ziyi Wang `[一作]` (Chinese University of Hong Kong), Man On Pun `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 4054 | [OpenAlex ID](https://openalex.org/A5040559125)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态大语言模型（MLLM）驱动的遥感图像分割框架MPerS，利用多视角Prompt生成高质量场景描述并通过文本引导视觉特征实现精细语义分割。

**💡 创新点**

创新点在于：①设计多视角Prompt与动态MixExperts融合不同MLLM产生的描述；②提出Linguistic Query Guided Attention实现文本引导的视觉特征融合；③通过Caption验证策略保证文本质量，显著提升分割性能。

**🔧 技术方法**

技术包括：DINOv3视觉编码器、CLIP文本编码器、LLaVA/ChatGPT/Qwen三种MLLM、动态MixExperts（Mixture‑of‑Experts）文本编码、Linguistic Query Guided Attention、U‑Net解码器以及跨模态对齐与注意力机制。

**📊 数据集**

使用公开遥感语义分割数据集Potsdam、Vaihingen和SynDrone进行实验，分别覆盖城市建筑、道路、植被等多种地表覆盖类别。

**📈 对比分析**

与MAResUNet、UNetFormer、DC‑Swin、A²‑FPN、RS³Mamba、MetaSegNet、FiLM和SegCLIP等方法在同一数据集上对比，MPerS在Vaihingen、Potsdam和SynDrone的数据集上分别取得mIoU/ mF1≈85.79/87.65%，88.40/89.32%，91.21/89.32%，在所有对比方法中实现了3–6% 的显著提升。

**⚠️ 局限性**

局限性包括：①依赖大型预训练模型（如DINOv3 7B、CLIP 0.3B）导致推理速度相对较慢；②对极端天气、低分辨率或少量样本场景的鲁棒性尚未充分验证；③文本生成质量仍受Prompt设计与MLLM性能影响，需要进一步自动化改进。

---

## 761. Break the Brake, Not the Wheel: Untargeted Jailbreak via Entropy Maximization

**arXiv ID:** 2605.10764 | [PDF](https://arxiv.org/pdf/2605.10764v1)

**作者:** Mengqi He `[一作]` (Australian National University), Jing Zhang `[通讯]` (Australian National University)

**通讯引用:** 17947 | [OpenAlex ID](https://openalex.org/A5100345321)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于熵最大化的无目标多模态越狱攻击方法 UJEM-KL，能够在不强制输出固定前缀的前提下，攻击视觉语言模型的安全机制。

**💡 创新点**

创新点在于识别并利用拒绝决策点的高熵特性，同时通过 KL 正则化稳定低熵结构位置，既能提升攻击成功率，又保持生成文本质量；并证明传统目标化攻击的低可迁移性主要源自过度约束的优化目标。

**🔧 技术方法**

技术核心包括：教师强制下的熵计算、基于梯度的投影优化（PGD）在图像扰动空间上、熵最大化目标、KL 归一化正则化以及对抗样本的多目标评估。

**📊 数据集**

使用的公开数据集有 JailBreakV‑28K 和 SafeBench 两个多模态越狱基准，用于评估攻击成功率和跨模型可迁移性。

**📈 对比分析**

在三种不同架构的 VLM（Qwen2.5‑VL、InternVL3.5、LLaVA‑1.5）上，UJEM‑KL 的白盒攻击成功率超过或与最强基线（如 SEA、Force）相当，且在跨模型迁移场景中平均提升 10%+ 的成功率；在代表性防御（SafeDecoding、Adversarial Training、UniGuard、R‑TOFU）下仍保持显著优势。

**⚠️ 局限性**

局限性包括：对图像扰动预算较小（ε=8/255）时效果受限；对模型结构差异极大的跨模型转移仍有下降；以及对基于后置过滤或模型重训练的高级防御仍易被削弱。

---

## 762. MAGS-SLAM: Monocular Multi-Agent Gaussian Splatting SLAM for Geometrically and Photometrically Consistent Reconstruction

**arXiv ID:** 2605.10760 | [PDF](https://arxiv.org/pdf/2605.10760v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 763. Reinforce Adjoint Matching: Scaling RL Post-Training of Diffusion and Flow-Matching Models

**arXiv ID:** 2605.10759 | [PDF](https://arxiv.org/pdf/2605.10759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 764. TINS: Test-time ID-prototype-separated Negative Semantics Learning for OOD Detection

**arXiv ID:** 2605.10756 | [PDF](https://arxiv.org/pdf/2605.10756v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 765. What should post-training optimize? A test-time scaling law perspective

**arXiv ID:** 2605.10716 | [PDF](https://arxiv.org/pdf/2605.10716v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 766. An Uncertainty-Aware Resilience Micro-Agent for Causal Observability in the Computing Continuum

**arXiv ID:** 2605.10718 | [PDF](https://arxiv.org/pdf/2605.10718v1)

**作者:** Suvi De Silva `[一作]` (Stockholm University), Praveen Kumar Donta `[通讯]` (Stockholm University)

**通讯引用:** 1979 | [OpenAlex ID](https://openalex.org/A5079303717)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个轻量级的微代理框架AURORA，用于在资源受限的边缘设备上对灰色故障进行因果诊断和安全缓解。

**💡 创新点**

创新点在于引入双重门控机制：先通过后验置信度门控制诊断可靠性，再通过变分自由能门评估干预一致性，从而在保持安全的前提下实现自动恢复；同时利用马尔可夫毯约束和do-计算实现实时因果推理。

**🔧 技术方法**

采用贝叶斯网络、马尔可夫毯约束、do-计算、主动推理（VFE）以及并行微代理的并行诊断管道。

**📊 数据集**

使用基于Python的仿真环境生成的合成边缘设备遥测数据（CPU、内存、网络、帧率、吞吐量），进行30,006次Monte‑Carlo实验。

**📈 对比分析**

与规则基代理和无门控AIF基线进行对比，AURORA在灰色故障场景下实现0%破坏性操作、62%修复准确率、平均修复时延约3毫秒，显著优于基线。

**⚠️ 局限性**

局限包括参数、阈值和恢复映射手动配置、缺乏真实物理边缘–云交互的评估、以及对多变量噪声、时延和信号开销的影响未完全覆盖。

---

## 767. ObjView-Bench: Rethinking Difficulty and Deployment for Object-Centric View Planning

**arXiv ID:** 2605.10707 | [PDF](https://arxiv.org/pdf/2605.10707v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 768. The finite expression method for turbulent dynamics with high-order moment recovery

**arXiv ID:** 2605.10687 | [PDF](https://arxiv.org/pdf/2605.10687v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 769. Rethinking Agentic Search with Pi-Serini: Is Lexical Retrieval Sufficient?

**arXiv ID:** 2605.10848 | [PDF](https://arxiv.org/pdf/2605.10848v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 770. GESR: A Genetic Programming-Based Symbolic Regression Method with Gene Editing

**arXiv ID:** 2605.10685 | [PDF](https://arxiv.org/pdf/2605.10685v1)

**作者:** Yanjie Li `[一作]` (Institute of Semiconductors, Chinese Academy of Sciences), Xin Ning `[通讯]` (Institute of Semiconductors, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于基因编辑的符号回归方法 GESR，利用两个 BERT 模型在遗传程序中指导基因突变与交叉。

**💡 创新点**

创新点在于将深度语言模型转化为“神之手”，通过数据驱动的 BERT 预测突变符号和交叉子树位置，实现有针对性的基因编辑，从而显著提升搜索效率与模型稳定性。

**🔧 技术方法**

主要技术包括：多模态 BERT（分别用于突变和交叉引导）、SetTransformer 数据编码、连续化突变函数、BFGS 常数优化，以及基于 R² 的性能评估。

**📊 数据集**

实验使用 13 个经典符号回归基准（SRBench、Feynman、Strogatz 等）以及 16 个混沌动力学系统的数据集进行向量场学习。

**📈 对比分析**

与 PySR、NGGP、GPlearn、SNIP 等主流方法比较，GESR 在 R² 准确度上保持相当或更优，表达式节点数更少，推理时间更短，搜索收敛速度最快。

**⚠️ 局限性**

主要局限是对预训练 BERT 的依赖，且在输入维度超出预训练范围时性能下降，未来需开发更轻量化、可迁移的编辑模型并提升高维/大规模数据处理能力。

---

## 771. Exact Unlearning from Proxies Induces Closeness Guarantees on Approximate Unlearning

**arXiv ID:** 2605.10680 | [PDF](https://arxiv.org/pdf/2605.10680v1)

**作者:** Virgile Dine `[一作]` (Inria), Teddy Furon `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种将机器无学习（unlearning）与数据分布结构直接关联的范式，利用可验证的代理模型在对数it空间中构造精确的无学习信号并通过知识蒸馏更新网络。

**💡 创新点**

创新点在于：①把无学习问题转化为闭式对数it修正，并给出可计算的 KL 散度上界；②引入多种基于数据结构的代理模型（高斯、LDA、标签翻倍等），实现对 retain/forget 子群的细粒度建模；③提供可检验的 admissibility 条件和安全范围，从理论上保证无学习模型优于原始模型。

**🔧 技术方法**

采用概率模型（高斯混合、LDA）、对数it 计算、KL 散度分析、知识蒸馏以及一维线性搜索确定无学习信号强度。

**📊 数据集**

在常用图像分类基准（如 CIFAR‑10、CIFAR‑100、ImageNet）以及四种网络架构（包括一个基础模型 backbone）上进行实验。

**📈 对比分析**

与现有的 SOTA 近似无学习方法（如 SISA、SCRUB、SalUn 等）以及理想的从零重训模型进行对比，实验显示本文方法在三种忘记场景（随机、子类、类别）下均能得到更接近理想重训模型的分类器，性能优于竞争者。

**⚠️ 局限性**

局限性包括：①需保留 retain 数据以拟合代理模型；②高特征维度下协方差估计可能不稳定，需要降维或对角约束；③admissibility 条件只能在经验分布上检验，无法在无学习前保证；④安全范围和 KL 上界在理论上是充分但非必要条件。

---

## 772. Energy-Efficient Implementation of Spiking Recurrent Cells on FPGA

**arXiv ID:** 2605.10679 | [PDF](https://arxiv.org/pdf/2605.10679v1)

**作者:** Pascal Harmeling `[一作]` (University of Liege), Guillaume Drion `[通讯]` (University of Liege)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

实现了基于SRC神经元的FPGA加速器，用MNIST生成的尖峰轨迹进行分类。

**💡 创新点**

通过数学简化将SRC模型改为FPGA友好，避免浮点、tanh、exp，保持连续动力学且硬件成本低。

**🔧 技术方法**

VHDL实现，LUT寄存器存储权重，双层并行处理，权重量化与短尖峰轨迹结合以降低能耗。

**📊 数据集**

MNIST手写数字数据集的尖峰轨迹。

**📈 对比分析**

与Julia的PsV实现及其他LIF基SNN对比，在100 MHz下实现96.31%准确率、1.7424 ms/位，能耗0.55–0.45 mJ/位，4-bit量化、44图像可保持92.9%准确率。

**⚠️ 局限性**

对极低位宽（≤3 bit）和极短尖峰轨迹时准确率骤降，且无法自适应稀疏事件驱动，需进一步研究混合调度。

---

## 773. Training-Free Cultural Alignment of Large Language Models via Persona Disagreement

**arXiv ID:** 2605.10843 | [PDF](https://arxiv.org/pdf/2605.10843v1)

**作者:** Huynh Trung Kiet `[一作]` (Vietnam National University), Long Tran-Thanh `[通讯]` (University of Warwick)

**通讯引用:** 2744 | [OpenAlex ID](https://openalex.org/A5012915897)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DISCA，一种仅在推理阶段使用的文化对齐方法，通过同一国家内不同文化化人物的分歧信号来调整大语言模型的决策概率。

**💡 创新点**

创新点在于：①将同一国家内部的分歧视为可靠性指标，采用方差驱动的收缩估计；②使用基于前景理论的失误惩罚重要性采样和双重通道可靠性门控；③无需模型权重更新、无每国奖励模型、只需公开API的决策token logit。

**🔧 技术方法**

技术手段包括：层次贝叶斯方差估计、MSE最优收缩、Prospect-Theory价值函数、重要性采样、双通道一致性检验、WVS数据驱动的角色提示、logit差距校正。

**📊 数据集**

使用的数据集：World Values Survey（构建国家与人口统计角色提示）；MultiTP（Moral Machine扩展，107语言的国家级AMCE评估）；BLEnD文化事实问答（验证方法边界）。

**📈 对比分析**

与传统推理时间基线（vanilla、Profile Prompt、PRISM、Activation Steering、MC‑Dropout）以及oracle 的单维温度/边缘校正进行对比；在20个国家、七大开源模型中，DISCA 在 Binary 道德困境上减少 10–24% 的误差，开放式情景上 2–7%；在 14B Phi‑4 方案下误差低于 70B Llama‑3.3‑70B 无调优模型，显示校准与规模竞争而非简单叠加。

**⚠️ 局限性**

局限性：①仅适用于可通过单个决策token logit 表示的二元或可量化决策，无法改进事实检索或多词输出；②需要 API 暴露决策token logits；③依赖 WVS 覆盖度，部分国家数据稀缺；④性能受模型内部逻辑而非文化本身限制；⑤安全性需设定效用下限，避免潜在的“主流偏好”强化。

---

## 774. StartFlow: From Method Conception to Multi-Perspective Evaluation in UX Prototyping for Software Startups

**arXiv ID:** 2605.10824 | [PDF](https://arxiv.org/pdf/2605.10824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 775. Large Spectrum Models (LSMs): Decoder-Only Transformer-Powered Spectrum Activity Forecasting via Tokenized RF Data

**arXiv ID:** 2605.10825 | [PDF](https://arxiv.org/pdf/2605.10825v1)

**作者:** Mohammad Mosiur Lunar `[一作]` (University of Nebraska-Lincoln), Mehmet C. Vuran `[通讯]` (University of Nebraska-Lincoln)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并训练了基于大型语言模型的解码器 Transformer，专门用于对大规模无线频谱功率谱密度进行短期预测。

**💡 创新点**

创新点在于将原始 IQ 信号转化为离散 token 序列，构建了专门的 RF tokenizer，并将通用 LLM 架构应用于真实世界的频谱预测任务。

**🔧 技术方法**

使用的技术包括 STFT、max‑pooling 与 trimmed mean 下采样、tokenization、以及对 Gemma‑2B、GPT‑2、LLaMA‑7B、Mistral‑7B、Phi‑1 等五种开源 LLM 进行改造为 LSM。

**📊 数据集**

使用的数据集为 BIG‑RED，来自 NEXTT 实验室的 22 TB 原始频谱数据，覆盖 33 个子 GHz 频段，总计约 8.4 亿个 token。

**📈 对比分析**

通过与传统 LSTM 与时间序列 Transformer 基线对比，LSM‑Mistral 在所有频段的 RMSE 低至 3.25 dB，99% 预测误差 ≤5 dB，显著优于基线。

**⚠️ 局限性**

局限性包括对高动态频段仍存在一定误差，模型规模和训练成本较高，且对不同站点的泛化仍需大量微调。

---

## 776. NanoResearch: Co-Evolving Skills, Memory, and Policy for Personalized Research Automation

**arXiv ID:** 2605.10813 | [PDF](https://arxiv.org/pdf/2605.10813v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 777. On periodic distributed representations using Fourier embeddings

**arXiv ID:** 2605.10818 | [PDF](https://arxiv.org/pdf/2605.10818v1)

**作者:** Jakeb Chouinard `[一作]` `[通讯]` (University of Waterloo), Jakeb Chouinard (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

提出了一种构造高维周期性分布式嵌入的方法，并推导了对应的周期核（Dirichlet核与周期高斯核）；

**💡 创新点**

通过控制相位矩阵的整数倍周期性，使得嵌入保持周期性，并通过不同采样分布（均匀/正态）得到可调节的核形状；

**🔧 技术方法**

使用空间语义指针（SSP）框架、傅里叶变换、布朗定理、theta函数以及概率分布采样技术；

**📊 数据集**

未使用具体数据集，主要以理论推导和可视化示例为主；

**📈 对比分析**

未进行实验性比较，论文仅展示了核函数的数学表达式与二维/三维可视化图；

**⚠️ 局限性**

局限性包括：高维核不具对称性，缺乏实证评估；仅在理论层面验证，未测试在实际认知模型中的性能；

---

## 778. PhyGround: Benchmarking Physical Reasoning in Generative World Models

**arXiv ID:** 2605.10806 | [PDF](https://arxiv.org/pdf/2605.10806v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 779. CLEF: EEG Foundation Model for Learning Clinical Semantics

**arXiv ID:** 2605.10817 | [PDF](https://arxiv.org/pdf/2605.10817v1)

**作者:** Peng Cao `[一作]` (MIT), Dina Katabi `[通讯]` (MIT)

**通讯引用:** 32932 | [OpenAlex ID](https://openalex.org/A5052959289)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并训练了 CLEF，一种以全 EEG 录制为单位、利用多通道多频谱（multitaper spectrogram）token 化并通过 Transformer 进行会话级自监督预训练的临床 EEG 基础模型，并将其与神经科医生报告和结构化 EHR 进行对齐；

**💡 创新点**

核心创新在于：①将 EEG 视为完整录制而非短窗口，①通过多通道 spectrogram token 化和 VQGAN 压缩实现会话级建模；②结合报告和 EHR 的跨模态对齐，构建临床语义驱动的表示；

**🔧 技术方法**

技术手段包括多通道 multitaper spectrogram 生成、VQGAN 进行 3D 代码化、Transformer 进行遮蔽图像建模（Masked Image Modeling），对齐阶段使用 T5 对报告摘要、EHR 代码表嵌入+自注意力，再通过对比损失实现对齐；

**📊 数据集**

主要使用 Harvard Electroencephalography Database（HEEDB）260k 录音/108k 患者数据，外部验证集为 TUAB、TUEP、HSP；

**📈 对比分析**

在 234 个临床二分类任务（疾病、药物、EEG 特征）上，与 5 个现有 EEG 基础模型（BIOT、LaBraM、CBraMod、REVE、NeuroLM）进行冻结编码器+轻量头的探测对比，CLEF 在 229/234 任务上胜出，平均 AUROC 从 0.65 提升到 0.74，且对齐阶段进一步提升；

**⚠️ 局限性**

局限性包括：缺乏实时推理能力、对睡眠结构等长期时变模式学习不足、可解释性有限、对不同通道数/录制长度的鲁棒性尚待提升。

---

## 780. Probing Cross-modal Information Hubs in Audio-Visual LLMs

**arXiv ID:** 2605.10815 | [PDF](https://arxiv.org/pdf/2605.10815v1)

**作者:** Jihoo Jung `[一作]` (Korea Advanced Institute of Science and Technology), Joon Son Chung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 10622 | [OpenAlex ID](https://openalex.org/A5038723822)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过无监督的因果追踪和统一模态优势框架，研究了音视大语言模型中跨模态信息的存储位置，并基于跨模态吸收符号（sink token）的发现，提出了无训练的自适应吸收符号解码方法以降低对象错觉。

**💡 创新点**

创新点在于：①将因果追踪与统一模态优势框架相结合，系统揭示跨模态信息主要集中在跨模态吸收符号；②区分单模态和跨模态吸收符号，证明后者是信息整合的核心；③提出自适应吸收符号解码（ASD），通过动态调节注意力来显著抑制错觉。

**🔧 技术方法**

主要技术包括：因果追踪（Causal Tracing）、注意力归一化与MDS（Modality Dominance Score）评估、吸收符号识别、无训练注意力重调节（Adaptive Sink-Guided Decoding）以及开放词表错觉评估（ALOHa）和CHIAR指标。

**📊 数据集**

使用的数据集为：VGGSound（音视测试集）及其动物子集、AudioSet，结合对象检测模型生成扩展标签，用于评估模型生成描述中的对象错觉。

**📈 对比分析**

对比方法包括：无训练的PAI和VCD两种改进；实验显示ASD在VGGSound-Animal上显著降低了CHIAR与ALOHa的错觉率，且在VGGSound-All和AudioSet上保持或提升了文本生成的质量，优于传统基线。

**⚠️ 局限性**

局限性：只针对当前公开的AVLLM（如Qwen2.5-Omni、video-SALMONN）验证，且依赖对吸收符号与MDS的准确划分；在极端模态不匹配或高噪声场景下效果未知；同时，调节参数α的取值会影响信息丰富度与错觉抑制之间的平衡。

---

## 781. Mistake-Bounded Language Generation

**arXiv ID:** 2605.10809 | [PDF](https://arxiv.org/pdf/2605.10809v1)

**作者:** Jon Kleinberg `[一作]` (Cornell University), Omer Reingold `[通讯]` (Stanford University)

**通讯引用:** 12398 | [OpenAlex ID](https://openalex.org/A5086367709)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并研究了误差次数受限的语言生成框架，阐明误差次数与收敛时间的相互作用。

**💡 创新点**

创新性地将误差次数作为评估指标，给出有限与无穷语言类的最优误差上界，并揭示了误差与收敛速度之间的不可调和权衡，此外将噪声模型纳入分析。

**🔧 技术方法**

采用与学习演示（Learning from Demonstrations）框架的正式还原、加权更新规则、增长函数设计、Littlestone 树构造等理论工具完成误差上界与收敛时间的证明。

**📊 数据集**

本文无真实数据集，全部使用人工构造的语言集合与对抗性序列来演示和证明理论结果。

**📈 对比分析**

与传统的“最后一次错误时间”指标相比，所给误差上界更紧凑（如 log|ℒ| 或 O(log i)），但在收敛速度上存在不可避免的线性代价；噪声情况下误差上界与噪声量呈线性关系，未给出实验验证。

**⚠️ 局限性**

局限性包括：误差上界仍依赖于类的基数或闭包维度，缺乏更精细的组合维度；误差与收敛速度的权衡不可避免；噪声模型下未能在无限但衰减噪声场景下保证有限误差；未提供实验或实际数据验证。

---

## 782. Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge

**arXiv ID:** 2605.10805 | [PDF](https://arxiv.org/pdf/2605.10805v1)

**作者:** Wenbo Zhang `[一作]` (University of California), Hengrui Cai `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了推理能力在LLM‑as‑a‑Judge场景下的效果，系统评估推理与非推理模式的准确率与计算成本，并提出鲁棒自适应成本高效路由框架RACER，用预算约束下动态选择推理或非推理判断器；

**💡 创新点**

①对推理 vs 非推理在判断任务中的差异进行定量分析；②用分布式鲁棒优化构造路由策略，既保证奖励最大化又满足成本约束；③提出唯一最优解与线性收敛的理论保证，并通过KL不确定集实现对分布偏移的鲁棒性；

**🔧 技术方法**

分布式鲁棒优化、KL不确定集、原始–对偶凸优化、熵正则化、经验数据重加权、神经网络路由器、推理模式选择算法；

**📊 数据集**

Skywork Reward Preference Dataset、Math‑Step‑DPO‑10K、Code‑Preference‑Pairs、JudgeBench、RewardBench、RewardBench‑2、Qwen3 hybrid reasoning models、Llama‑3.1‑8B等；

**📈 对比分析**

与All‑Instruct、All‑Reasoning、Random、RACER‑R、RACER‑C、ACER、RouterBench‑KNN、RouteLLM‑MF、M‑IRT等基线对比；在ID和OOD测试中，RACER在相同预算下实现更高准确率（如相对提升1–1.1个百分点）并保持成本约束；

**⚠️ 局限性**

仅支持二元推理/非推理路由，KL不确定集在大分布偏移时可能过度保守；模型规模固定，需手动调节温度与熵正则系数；对更复杂路由策略与其他不确定集形式尚未验证。

---

## 783. TrajPrism: A Multi-Task Benchmark for Language-Grounded Urban Trajectory Understanding

**arXiv ID:** 2605.10782 | [PDF](https://arxiv.org/pdf/2605.10782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 784. The Last Word Often Wins: A Format Confound in Chain-of-Thought Corruption Studies

**arXiv ID:** 2605.10799 | [PDF](https://arxiv.org/pdf/2605.10799v1)

**作者:** Gabriel Garcia `[一作]` `[通讯]` (Independent Researcher), Gabriel Garcia (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在对链式推理（CoT）模型的因果推断中，本文通过一系列基于错误注入的腐败实验发现，传统的定位重要链位置的方法被链条格式（特别是显式答案结尾）的答案文本所误导，而非真正的计算过程；

**💡 创新点**

提出并验证了一种三步先决条件协议（问题仅控制、格式特征化、全位置扫荡），以剔除答案文本位置对腐败敏感度的影响；

**🔧 技术方法**

主要技术包括：语义错误注入、冲突答案实验、2×2因子设计、答案位置拆分/替换、前后段消融、以及生成-消费时序探测（早期承诺率与逐步停止）等；

**📊 数据集**

使用的公开数据集包括 GSM8K、MATH、Hard‑v3、Commonsense‑v1 等，且对每个数据集都构造了标准与答案文本被去除/替换的版本；

**📈 对比分析**

实验对比显示：在标准格式下，后缀位置的错误导致准确率骤降（如 3B 模型的 Δ≈-0.76）；去除答案文本后后缀敏感度缩小约 19 倍；冲突答案实验显示 7B 规模模型几乎全失准，随后在 14B、32B 规模上逐渐减弱；总体性能表明答案文本占主导地位，非答案位置的错误对准确率影响微乎其微；

**⚠️ 局限性**

局限性包括：仅验证了显式答案结尾格式，未覆盖无答案结尾或自然生成的 CoT；实验仅限 3B–32B 规模；仅使用了语义错误注入，其他类型错误未探索；对生成时刻的探测仍为间接证据，未进行细粒度内部机制分析；

---

## 785. Interpretable Machine Learning for Football Performance Analysis: Evidence of Limited Transferability from Elite Leagues to University Competition

**arXiv ID:** 2605.10796 | [PDF](https://arxiv.org/pdf/2605.10796v1)

**作者:** Yu-Fang Tsai `[一作]` (National Tsing Hua University), Chien-Ming Hsu `[通讯]` (National Tsing Hua University)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5013493109)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在将精英足球模型应用于大学级别足球时，性能决定因素的结构性可转移性与解释性（SHAP、CIS）的稳定性。

**💡 创新点**

首次将解释性稳定性视为评估跨域结构一致性的诊断指标，并系统比较不同域与不同解释方法下的解释一致性。

**🔧 技术方法**

使用随机森林与多层感知机模型，并分别采用 SHapley Additive exPlanations (SHAP) 与 Counterfactual Impact Score (CIS) 进行全局特征重要性解释。

**📊 数据集**

精英数据来源于2019/20至2024/25赛季的五大欧洲联赛（共约5,700场比赛），大学数据来自台大二级联赛的17场官方比赛。

**📈 对比分析**

先用 MAE/RMSE 评估精英模型预测准确性；随后在两域内比较特征重要性排名、种子间 Spearman 相关、域间结构一致性和方法一致性；发现精英域解释高度稳定且两解释方法一致，但大学域解释重排序明显、稳定性与方法一致性显著下降。

**⚠️ 局限性**

局限性包括大学样本量仅17场、来自单一高校，且仅做全局解释，缺乏对局部或情境特定解释的探讨。

---

## 786. ComplexMCP: Evaluation of LLM Agents in Dynamic, Interdependent, and Large-Scale Tool Sandbox

**arXiv ID:** 2605.10787 | [PDF](https://arxiv.org/pdf/2605.10787v1)

**作者:** Yuanyang Li `[一作]` (Zhejiang University), Hongyang Chen `[通讯]` (Zhejiang Lab)

**通讯引用:** 8387 | [OpenAlex ID](https://openalex.org/A5008473103)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了一个名为 ComplexMCP 的基准测试平台，用于评估大型语言模型（LLM）在大规模、状态依赖、动态工具沙盒中的自动化能力。

**💡 创新点**

核心创新在于：① 将 Model Context Protocol（MCP）与种子驱动的动态环境结合，生成可复现且噪声丰富的工具调用场景；② 设计了细粒度、规则化的评估指标，能够精确测量成功率与副作用；③ 通过手工构造的 47 条多工具任务，展示了复杂工具间隐式依赖与错误恢复的挑战。

**🔧 技术方法**

使用技术包括：MCP 协议实现工具接口；种子驱动的环境初始化与运行时扰动；基于关键路径差异的确定性评估方法；以及 ReAct、RAG、迭代 RAG 等提示策略。

**📊 数据集**

数据集主要为：
• 300+ 通过 MCP 实现的状态工具（7 大沙盒 + 150+ 无状态 API）；
• 47 条人工标注的多工具指令与金标准轨迹；
• 通过种子产生的合成知识库，用于环境多样性。

**📈 对比分析**

对比方法包括：完整上下文（Full‑Context）与 RAG（kNN 提取工具）及迭代 RAG；实验在 GPT‑4o、GPT‑5.1、Gemini‑3‑Flash、Claude 系列、Llama‑3、Qwen‑3‑Max 等 19 种模型上进行。结果显示：最高模型 Gemini‑3‑Flash 的成功率仅 55%（R_c 85.79%），远低于人类 93%+，而 RAG/迭代 RAG 的表现明显逊色，显示工具间隐式依赖难以通过语义检索捕获。

**⚠️ 局限性**

局限性：① 指令集仅 47 条，规模受人工标注成本限制；② 仅覆盖 7 个沙盒，可能未能充分代表所有行业场景；③ 评估侧重于 deterministic 轨迹，未覆盖更大规模或多任务连续交互的挑战。

---

## 787. The Generalized Turing Test: A Foundation for Comparing Intelligence

**arXiv ID:** 2605.10851 | [PDF](https://arxiv.org/pdf/2605.10851v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 788. BabelDOC: Better Layout-Preserving PDF Translation via Intermediate Representation

**arXiv ID:** 2605.10845 | [PDF](https://arxiv.org/pdf/2605.10845v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 789. Transcoda: End-to-End Zero-Shot Optical Music Recognition via Data-Centric Synthetic Training

**arXiv ID:** 2605.10835 | [PDF](https://arxiv.org/pdf/2605.10835v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 790. Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents

**arXiv ID:** 2605.10832 | [PDF](https://arxiv.org/pdf/2605.10832v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 791. Clin-JEPA: A Multi-Phase Co-Training Framework for Joint-Embedding Predictive Pretraining on EHR Patient Trajectories

**arXiv ID:** 2605.10840 | [PDF](https://arxiv.org/pdf/2605.10840v1)

**作者:** Yixuan Yang `[一作]` (Duke University), Rishikesan Kamaleswaran `[通讯]` (Duke University)

**通讯引用:** 2362 | [OpenAlex ID](https://openalex.org/A5012511062)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种名为Clin‑JEPA的多阶段共训练框架，利用单一LLM编码器与保留的潜在轨迹预测器，实现对ICU患者轨迹的联合嵌入预测与自回归模拟，兼顾预测与风险评估；

**💡 创新点**

创新点在于：①设计了五阶段预训练课程（warmup、共训练、EMA对齐、硬同步、finalize），有效抑制表示崩塌与在线/目标漂移；②将编码器与预测器在同一JEPA目标下共同训练，使编码器的潜在空间与预测器的回归目标同步；③使用自然语言文本化的EHR表示，避免了特征工程与缺失值插补；

**🔧 技术方法**

采用Qwen3‑8B作为基础语言模型并通过LoRA轻量化适配；结合Transformer形式的潜在轨迹预测器；使用EMA目标编码器与块因果注意力机制；并使用L1损失与教师强迫+自回归损失进行联合训练；

**📊 数据集**

在MIMIC‑IV ICU数据集上训练，包含约84,000个ICU停留，采用72小时一小时分辨率窗口；

**📈 对比分析**

与传统基线（岭回归、LightGBM、LSTM、TCN）以及其他JEPA变体对比，Clin‑JEPA在ICareFM EEP任务上平均AUROC提升至0.851（+0.038），在8项住院风险任务上平均AUROC提升至0.883（+0.041），同时在48小时自回归轨迹上实现了-15.7%的漂移收敛；

**⚠️ 局限性**

局限性包括：仍主要验证于ICU数据，跨机构或其他临床域的可迁移性未充分评估；依赖大型LLM及高算力，训练成本高；在极端稀缺或高噪声场景下潜在表示的鲁棒性有待进一步验证。

---

## 792. SLIM: Sparse Latent Steering for Interpretable and Property-Directed LLM-Based Molecular Editing

**arXiv ID:** 2605.10831 | [PDF](https://arxiv.org/pdf/2605.10831v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 793. The First Drop of Ink: Nonlinear Impact of Misleading Information in Long-Context Reasoning

**arXiv ID:** 2605.10828 | [PDF](https://arxiv.org/pdf/2605.10828v1)

**作者:** Muhan Gao `[一作]` (Texas A\&M University), Kuan-Hao Huang `[通讯]` (Texas A\&M University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在长上下文推理中，少量硬干扰信息（与查询语义相关但不包含答案）对大型语言模型性能造成的非线性降幅，并提出“First Drop of Ink”效应。

**💡 创新点**

创新点：①首次系统量化硬干扰比例与性能之间的非线性关系；②基于自注意力机制给出严格凸的理论解释；③通过对检索头的logit边际度量验证理论，揭示硬干扰在软max分布中占主导的机制。

**🔧 技术方法**

使用 Transformer 自注意力公式、softmax 温度调节实验、检索头的logit边际度量、上下文长度与干扰比例的分离实验等技术。

**📊 数据集**

数据集包括 Natural Questions、TriviaQA、PopQA、HotpotQA；模型覆盖多款长上下文 LLM（如 Llama‑3.1‑8B‑Instruct 等）。

**📈 对比分析**

与传统的过滤（去除干扰）和温度缩放等方法对比。实验显示：仅通过过滤提升主要来自上下文长度缩短，除非硬干扰比例降至接近 0；温度缩放反而进一步降低性能。整体性能在 10% 硬干扰时下降约 50–60%，随后几乎平缓。

**⚠️ 局限性**

局限性：仅在多文档 QA 场景下验证，未涵盖摘要、代码理解或多轮对话等长上下文任务；未提出有效的硬干扰消除策略；对检索阶段的提升建议仍需进一步验证。

---

## 794. ALAM: Algebraically Consistent Latent Transitions for Vision-Language-Action Models

**arXiv ID:** 2605.10819 | [PDF](https://arxiv.org/pdf/2605.10819v1)

**作者:** Zuojin Tang `[一作]` (Zhejiang University), Gang Pan `[通讯]` (Zhejiang University)

**通讯引用:** 471638 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了ALAM预训练框架，利用无标签视频的时间关系学习结构化潜在动作；

**💡 创新点**

创新点在于引入组合一致性与逆向一致性约束，使潜在动作可近似加法且可反向；

**🔧 技术方法**

使用变压器编码器+量化+像素重建以及联合流匹配（flow‑matching）作为下游策略训练；

**📊 数据集**

在多源无标签视频（Open‑X‑Embodiment、CALVIN等）预训练，并在MetaWorld MT50、LIBERO及Piper机器人真实任务上验证；

**📈 对比分析**

与多种基线对比，ALAM+流匹配在MetaWorld MT50平均成功率从47.9%提升至85.0%，在LIBERO从94.1%提升至98.1%，在真实机器人任务上亦显著超越基线；

**⚠️ 局限性**

局限在于只提供近似代数正则，未保证完全一致性，且联合流匹配会增加额外计算负担。

---

## 795. Likelihood scoring for continuations of mathematical text: a self-supervised benchmark with tests for shortcut vulnerabilities

**arXiv ID:** 2605.10810 | [PDF](https://arxiv.org/pdf/2605.10810v1)

**作者:** Daniel Ranard `[一作]` `[通讯]` (California Institute of Technology), Daniel Ranard (California Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于已知上下文与预测字符串的自动生成的技术论文续写基准，评估模型通过辅助预测提升真实后续文本的似然；

**💡 创新点**

创新点在于将预测字符串与固定概率评估器的似然提升作为无标签、可扩展的衡量标准，并针对非预测的捷径进行静态对照测试；

**🔧 技术方法**

使用的技术包括自动化数据抽取脚本、概率提升（per‑token log‑likelihood lift）与软化评分规则、两种公开评估模型（Qwen3‑8B 与 Kimi K2.6）、以及LoRA 微调的上下文仅控制模型；

**📊 数据集**

数据集为1363个来自138篇近期 arXiv 物理与数学论文的方程式后缀切片（以及额外的661个混合散文/技术续写样本）；

**📈 对比分析**

在该基准上，高推理强度的 GPT‑5.5 在两种评估器上均显著优于相同预算的最近上下文控制，并且在“上下文‑SFT”控制下仍保持优势；

**⚠️ 局限性**

局限性包括：基准仅评估技术论文续写，无法完全捕捉非预测捷径；软化评分规则和评估器的选择会影响结果；未覆盖所有潜在攻击场景，且对不同模型提供者的可比性有限。

---

## 796. Rapid Forest Fuel Load Estimation via Virtual Remote Sensing and Metric-Scale Feed-Forward 3D Reconstruction

**arXiv ID:** 2605.10789 | [PDF](https://arxiv.org/pdf/2605.10789v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 797. Constant time testability of first-order logic with modulo counting on finitary graphs

**arXiv ID:** 2605.10841 | [PDF](https://arxiv.org/pdf/2605.10841v1)

**作者:** Isolde Adler `[一作]` (University of Bamberg), Jenny Stimpson `[通讯]` (University of Bamberg)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

证明在度受限且连通分量大小有限的图类上，所有一阶逻辑（含模数计数）可在常数时间内被属性测试。

**💡 创新点**

提出了针对有限连通分量大小的类的“补丁可拼接”条件，并将汉夫正规化与组件直方图相结合，得到新的算法元定理。

**🔧 技术方法**

使用汉夫正规化、组件直方图向量、模数计数量化、Frobenius硬币定理等数论工具进行证明。

**📊 数据集**

本文为理论工作，不涉及具体数据集。

**📈 对比分析**

与现有的多项式时间测试或常数查询复杂度方法相比，本文实现了常数运行时间的属性测试，证明了在此类图上可在O(1)时间内完成测试。

**⚠️ 局限性**

仅适用于具有连通分量大小上界的图类；对树宽受限类仍未能给出常数时间测试的结论。

---

## 798. MaD Physics: Evaluating information seeking under constraints in physical environments

**arXiv ID:** 2605.10820 | [PDF](https://arxiv.org/pdf/2605.10820v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 799. Benchmarking Sensor-Fault Robustness in Forecasting

**arXiv ID:** 2605.10822 | [PDF](https://arxiv.org/pdf/2605.10822v1)

**作者:** Alexander Windmann `[一作]` (Helmut Schmidt University), Oliver Niggemann `[通讯]` (Helmut Schmidt University)

**通讯引用:** 2136 | [OpenAlex ID](https://openalex.org/A5012395966)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SensorFault-Bench，一个针对网络物理系统（CPS）预测模型的共享传感器故障压力测试基准，评估模型在噪声、漂移、时序失配和可用性缺失等八类结构化故障下的鲁棒性；

**💡 创新点**

创新点在于：①将真实传感器故障模式量化为统一的严重度模型并构造标准化的输入侧扰动；②定义“最差场景降解”“干净 MSE”“最差场景故障时间 MSE”三种指标，分离相对鲁棒性与绝对误差；③引入离散的故障转移集（𝒫_trans）以检验显式故障训练方法的跨场景迁移；

**🔧 技术方法**

采用了多种时间序列预测架构（DLinear、GRU、ModernTCN、PatchTST、TSMixer、SeasonalNaive、Chronos‑2）以及鲁棒性提升技术（PGD 对抗训练、随机训练、随机平滑、适应性鲁棒损失、RevIN、故障增强、集成聚合）；

**📊 数据集**

使用了四个真实 CPS 数据集：北京空气质量（单目标）、风机 SCADA（单目标）、ETTh1 电力变压器温度（多目标）以及 Traffic 高速公路车道占用率（多目标）；

**📈 对比分析**

通过在同一批次下对各模型在八类故障情景中计算最差场景降解和最差场景故障时间 MSE，并与清洁 MSE 进行对比，发现清洁 MSE 较低的模型并不一定鲁棒；鲁棒性提升方法对不同场景表现不一致，集成聚合最为稳健；

**⚠️ 局限性**

局限性包括：①仅评估输入侧单一场景扰动，未覆盖组合故障、持久目标故障或闭环控制影响；②基准数据仅来自四个领域，未涵盖所有 CPS 传感器类型和控制模式；③方法比较受限于有限的模型和训练预算，未进行全量调优；

---

## 800. Preservation Theorems in Semiring Semantics

**arXiv ID:** 2605.10829 | [PDF](https://arxiv.org/pdf/2605.10829v1)

**作者:** Sophie Brinke `[一作]` (RWTH Aachen University), Benedikt Pago `[通讯]` (University of Cambridge)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5041562636)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了经典模型理论中的保持定理在半环语义下的状态，包括Łoś-Tarski定理和同态保持定理。

**💡 创新点**

证明了这些保持定理在所有格半环中成立，而在许多其他半环中则失败，尤其是存在性保持定理在有限解释中成立的情况。

**🔧 技术方法**

结合了经典紧致性和合并方法的适应，以及在半环语义中为逻辑蕴涵开发的特定简化方法。

**📊 数据集**

使用了多种半环，包括格半环、热带半环、Viterbi半环和Łukasiewicz半环等。

**📈 对比分析**

通过与经典模型理论的比较，发现许多经典保持定理在有限结构中失败，但在半环语义中，某些定理在有限情况下仍然成立，尤其是在Viterbi和Łukasiewicz半环中。

**⚠️ 局限性**

在许多其他半环中，存在性保持定理的变体失败，且在有限情况下的扩展保持定理的情况较为复杂，依赖于半环的代数性质。

---

## 801. NoRIN: Backbone-Adaptive Reversible Normalization for Time-Series Forecasting

**arXiv ID:** 2605.10823 | [PDF](https://arxiv.org/pdf/2605.10823v1)

**作者:** Shun Zhang `[一作]` (China Academy of Engineering Physics), Yuyang Xiao `[通讯]` (China Academy of Engineering Physics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可逆非线性归一化方法 NoRIN，通过 Johnson S_U 转换对时间序列进行非线性重塑，并将形状参数通过离线搜索而非梯度学习得到。

**💡 创新点**

核心创新在于发现并解决了 RevIN 族归一化的“退化问题”，即形状参数若与模型一起训练会收敛到线性极限；通过将形状参数视为超参数并采用闭式初始化 + 贝叶斯优化实现真正的非线性归一化。

**🔧 技术方法**

使用 Johnson S_U（arcsinh‑based）变换作为非线性可逆映射，Slifker–Shapiro 量化拟合做热启动，Optuna 的 TPE/GP 进行超参数搜索，标准 RevIN、SAN、Dish‑TS、DeStat 等对比基线。

**📊 数据集**

在五个公开长周期预测基准（Exchange、ETTh1、ETTh2、ETTm1、ETTm2）上，对六种主流后端（Informer、PatchTST、iTransformer、DLinear、TimesNet、FEDformer）进行实验，三种预测时窗（96、336、720）共 90 组配置。

**📈 对比分析**

与无归一化、RevIN、SAN、Dish‑TS、DeStat 等基线进行成对 Wilcoxon 检验，NoRIN 在 90 组配置中 83/90 至 90/90 场景获胜，平均 MSE 下降约 0.056（RevIN）到 1.90（DeStat），显著优于所有基线。

**⚠️ 局限性**

局限在于离线形状参数搜索的计算开销、仅使用 2‑参数 Johnson S_U 可能不足以处理极端多模态或长期依赖数据，以及搜索空间边界接触提示可能需要更宽范围的参数搜索。

---

## 802. New AI-Driven Tools for Enhancing Campus Well-being: A Prevention and Intervention Approach

**arXiv ID:** 2605.10804 | [PDF](https://arxiv.org/pdf/2605.10804v1)

**作者:** Jinwen Tang `[一作]` `[通讯]` (University of Missouri-Columbia), Jinwen Tang (University of Missouri-Columbia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

结合对话式人工智能与强化学习，本文开发了校园满意度与心理健康监测系统，包括 TigerGPT 与 AURA 问卷聊天机器人，以及基于多专家框架的情感分析工具。

**💡 创新点**

在短会话中首次实现了实时内模学习的对话式调查，利用四维 LSDE 质量信号与五种行动类别，首次将 RL 与 LLM 融合以优化校园调查，并将多专家堆叠模型与 GPT‑4 结合进行长文本心理健康评估。

**🔧 技术方法**

所用技术包括 LLM（ChatGPT‑4、ChatGPT‑4o、GPT‑4o‑mini）、强化学习（ε‑greedy、期望值更新）、多专家堆叠模型（SMMR）、BERT/MentalBERT、VADER、LangChain、Streamlit 等。

**📊 数据集**

使用的数据集包括 TigerGPT 对话日志（96 场次共 467 条回复）、Expressive Narrative Stories（Reddit ENS）、DAIC‑WOZ、公开心理健康文本以及校园满意度调查问卷。

**📈 对比分析**

通过与传统静态问卷和基线 TigerGPT 对照，AURA 在 10–15 轮对话中平均提升响应质量 +0.076，满意度提升至 81%；心理健康工具在 ENS 与 DAIC‑WOZ 上实现 F1 0.93/0.95，SMMR 在单模型基础上提升约 10% 的准确率。

**⚠️ 局限性**

局限性包括会话长度短导致学习样本不足、数据分布偏差影响 RL 泛化、情感分析对非英语文本鲁棒性待验证，以及多专家框架计算成本高且需进一步验证隐私合规性。

---

## 803. Constant Inapproximability for Fisher Markets

**arXiv ID:** 2605.10802 | [PDF](https://arxiv.org/pdf/2605.10802v1)

**作者:** Argyrios Deligkas `[一作]` (Royal Holloway), Themistoklis Melissourgos `[通讯]` (University of Essex)

**通讯引用:** 70 | [OpenAlex ID](https://openalex.org/A5024301526)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了 Fisher 市场中具有 SPLC（可分离可线性分段凹）效用函数的 ε-近似均衡问题在任何常数 ε < 1/11 的情况下都是 μ-难的，从而排除了存在多项式时间近似方案（PTAS）的可能性。

**💡 创新点**

创新点在于：
- 提出了一个全新的从布尔电路约束问题（带有特殊门）到 Fisher 市场的多阶段构造，结构更简单、直观；
- 通过引入参考商品、反向投入者（inverter）和辅助买家实现门的模拟，避免了之前使用价格调节市场等复杂机制；
- 通过复制电路并划分参考价格区间，显著提升了硬件阈值，得到 1/11 的近似误差上限。

**🔧 技术方法**

主要技术：
- 组合优化与可分离可线性分段凹效用的线性规划求解；
- 准确计算买家的最优购买集合并利用其贪心特性；
- 价格稳定性分析（reference good）与多重门电路的递推不等式；
- 从 Fisher 市场到 Arrow‑Debreu 交换市场的经典规模不变性还原。

**📊 数据集**

本研究不使用任何实验数据集，完全是理论证明与复杂度分析。

**📈 对比分析**

比较方式：与先前对 SPLC 市场的 μ-完整性结果（仅在 ε 为极小常数时）对比，本文把阈值从极小提升到 1/11；
性能上，即使是近似解也无法在多项式时间内得到，除非 μ = P。没有实验性能指标。

**⚠️ 局限性**

局限性：
- 近似阈值仍受 1/11 的限制，未来可能进一步提高；
- 证明仅针对 SPLC 效用，不能直接推广到更一般的凹/非凹效用；
- 依赖于电路约束问题的 μ-完整性假设，若该假设失效则结论不再成立。

---

## 804. Muown: Row-Norm Control for Muon Optimization

**arXiv ID:** 2605.10797 | [PDF](https://arxiv.org/pdf/2605.10797v1)

**作者:** Kai Lion `[一作]` (ETH Zurich), Niao He `[通讯]` (ETH Zurich)

**通讯引用:** 1360 | [OpenAlex ID](https://openalex.org/A5071683073)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于 Muon 的优化器 \muonplus，通过将行尺度提升为可训练变量，控制权重矩阵的谱范数漂移，从而提高语言模型预训练的收敛速度和鲁棒性。

**💡 创新点**

将行尺度因子从 Muon 的单纯更新拆分为显式可训练变量，使用 ℓ∞ 几何更新，避免了谱范数漂移和对权重衰减的敏感性，同时保持 Muon 的矩阵预处理优势；同时提供理论收敛证明。

**🔧 技术方法**

矩阵谱范数分解、行归一化、Weight Normalization、Muon's steepest spectral descent、Adam/SignSGD 的 ℓ∞ 变体、随机梯度下降分析、对比实验。

**📊 数据集**

FineWeb-Edu 以及 Qwen2-0.5B 等 GPT 风格架构的预训练数据。

**📈 对比分析**

与 Muon、SOAP、AdamW、Lion 进行学习率与权重衰减搜索，在 124M、500M、1B、2.7B 规模模型下评估 perplexity；\muonplus 在所有规模上均比 Muon 提高 0.2–0.3 perplexity，并在更宽的学习率/权重衰减范围内保持稳定。

**⚠️ 局限性**

当权重矩阵存在全零行时分解失效；目前仅在 dense GPT 预训练上验证，卷积、Mixture‑of‑Experts 或更大规模/更长训练仍未评估；行尺度参数的线性内存开销可能对极大模型产生影响。

---

## 805. ConQuR: Corner Aligned Activation Quantization via Optimized Rotations for LLMs

**arXiv ID:** 2605.10793 | [PDF](https://arxiv.org/pdf/2605.10793v1)

**作者:** Chayne Thrash `[一作]` (Vanderbilt University), Soheil Kolouri `[通讯]` (Vanderbilt University)

**通讯引用:** 3476 | [OpenAlex ID](https://openalex.org/A5068682350)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量化的后训练旋转校准方法，用于提升大语言模型（LLM）的低位激活量化性能。

**💡 创新点**

创新点在于：①将激活向量对齐到单位球内嵌超立方体角点的目标，鼓励能量均匀分布；②通过正交Procrustes问题得到闭式旋转更新；③采用在线小批量校准，无需存储大量激活，且校准过程与推理时的量化分布保持一致。

**🔧 技术方法**

使用的技术包括：正交Procrustes闭式更新、在线小批量校准、旋转融合到线性层、异步/对称量化、Llama-2/Llama-3模型的后训练量化框架。

**📊 数据集**

实验使用的校准和评估数据集为：WikiText-2、Penn Treebank (PTB)、C4，以及9个常识推理基准（WinoGrande、SocialIQA、LAMBADA、MMLU、ARC-Easy、ARC-Challenge、HellaSwag、OpenBookQA、PIQA）。校准时采样128条长度2048的WikiText-2序列。

**📈 对比分析**

与QuaRot、SpinQuant、DartQuant、DFRot等现有旋转量化方法对比：在4‑4‑16和4‑4‑4配置下，本方法在大多数模型（如Llama‑3 8B、Llama‑3 70B）上实现了更低的PPL并保持或提升零样本推理平均得分；校准成本仅为SpinQuant的几分之一（约0.42 GPU‑小时），且不需要海量激活存储（零存储）。

**⚠️ 局限性**

局限性：需额外的校准步骤，性能对校准数据分布敏感；实验主要聚焦于PPL和零样本推理，生成质量、推理延迟等方面尚未系统评估。

---

## 806. Rebellious Student: Reversing Teacher Signals for Reasoning Exploration with Self-Distilled RLVR

**arXiv ID:** 2605.10781 | [PDF](https://arxiv.org/pdf/2605.10781v1)

**作者:** Jeonghye Kim `[一作]` (Microsoft Research), Yuqing Yang `[通讯]` (Microsoft Research)

**通讯引用:** 2206 | [OpenAlex ID](https://openalex.org/A5101421201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RLRT（RLVR with Reversed Teacher）算法，利用自蒸馏的教师-学生信息不对称，在成功的轨迹上反向强化学生与教师不一致的 token，从而提升 LLM 的推理能力。

**💡 创新点**

核心创新在于将自蒸馏方向反转：在成功轨迹上不再把学生逼向教师，而是放大学生自发的、对教师预测有偏离但仍能成功的 token，形成以信息不对称为基础的价值探索信号。

**🔧 技术方法**

结合 RLVR 与 GRPO 框架，采用 token 级信息不对称信用 w_t=exp(sign(A)·Δ_t)；对成功 roll‑out 进行奖励门控与权重裁剪；使用反向教师信号、反思注入、Jensen‑Shannon 分布差异分析等技术验证和解释算法效果。

**📊 数据集**

训练数据使用 DAPO‑Math‑17k；评估数据为六个数学推理基准（AIME24/25/26、HMMT26、AMC23、MATH500）。实验涵盖 Qwen3‑4B/8B Base、Qwen3‑4B‑Instruct、Qwen3‑8B（思考模式关闭）三类模型。

**📈 对比分析**

与 GRPO、SDPO、SRPO、RLSD 等自蒸馏基线以及 GRPO+EB、DIVER 等探索方法对比。RLRT 在 avg@16 上平均提升 18.0%、12.0%、3.4% 和 2.2%（分别对应 Base、Base‑8B、Instruct、Thinking‑off），在 pass@16 和 pass@k 曲线也显著优于所有基线，证明其在推理任务上的显著性能优势。

**⚠️ 局限性**

局限性包括：仅在可验证奖励（准确答案）环境下验证，缺乏对噪声奖励的鲁棒性；依赖教师使用特权上下文（如正确答案或成功 roll‑out）；实验仅聚焦数学推理任务，尚未验证在更广泛推理场景或多模态任务中的泛化；需要进一步探索不同信息不对称来源和更大模型的适用性。

---

## 807. Verification Mirage: Mapping the Reliability Boundary of Self-Verification in Medical VQA

**arXiv ID:** 2605.10850 | [PDF](https://arxiv.org/pdf/2605.10850v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 808. Democratizing Measurement of Critical Mobile Infrastructure: Security and Privacy in an Increasingly Centralized Communication Ecosystem

**arXiv ID:** 2605.10812 | [PDF](https://arxiv.org/pdf/2605.10812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 809. Elucidating Representation Degradation Problem in Diffusion Model Training

**arXiv ID:** 2605.10790 | [PDF](https://arxiv.org/pdf/2605.10790v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 810. Unified Noise Steering for Efficient Human-Guided VLA Adaptation

**arXiv ID:** 2605.10821 | [PDF](https://arxiv.org/pdf/2605.10821v1)

**作者:** Junjie Lu `[一作]` (University of Technology Sydney), Li Zhao `[通讯]` (Microsoft Research)

**通讯引用:** 12271 | [OpenAlex ID](https://openalex.org/A5032277491)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 UniSteer 框架，将人类纠正动作映射到噪声空间，从而在冻结的流匹配 Vision‑Language‑Action（VLA）模型上实现在线强化学习与人类指导的统一适应；

**💡 创新点**

核心创新在于① 通过近似逆推（固定点迭代）将动作空间纠正映射为噪声目标；② 在噪声空间统一人类监督与强化学习，保持生成器冻结；③ 采用轻量级噪声演员实现高效且稳定的适应；

**🔧 技术方法**

使用流匹配 VLA、噪声空间微调、固定点逆推、TD‑基强化学习、轻量级策略梯度以及真实机器人上的人机交互；

**📊 数据集**

在四个真实世界操控任务上验证：挑勺子、堆叠方块、插正方形、折毛巾；使用 AgileX Piper 机器人、RGB 视觉+末端姿态观测，初始给 30 条演示；

**📈 对比分析**

与 DSRL（仅噪声空间 RL）和 DAgger（动作空间示范）对比，UniSteer 在平均成功率上从 20% 提升至 90%，训练时间平均 66 分钟，OOV 成功率从 0% 提升至 100%，显著提高了样本效率和人类干预利用率；

**⚠️ 局限性**

实验仅覆盖四个相对简单的任务，逆推方法依赖生成器的 Lipschitz 连续性；在人类干预稀缺或更复杂动态环境下的鲁棒性尚未验证，且长期在线适应效果尚待进一步研究。

---

## 811. MMVIAD: Multi-view Multi-task Video Understanding for Industrial Anomaly Detection

**arXiv ID:** 2605.10833 | [PDF](https://arxiv.org/pdf/2605.10833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 812. PathISE: Learning Informative Path Supervision for Knowledge Graph Question Answering

**arXiv ID:** 2605.10791 | [PDF](https://arxiv.org/pdf/2605.10791v1)

**作者:** Shengxiang Gao `[一作]` (University of Melbourne), Jianzhong Qi `[通讯]` (University of Melbourne)

**通讯引用:** 4809 | [OpenAlex ID](https://openalex.org/A5022290876)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过一个轻量级Transformer+MIL估计器，从答案级标签中学习高质量的路径级中间监督，并将其蒸馏给LLM路径生成器，从而在KG上产生紧凑证据进行答案推理。

**💡 创新点**

①无需昂贵的LLM调用即可从答案标签估计路径信息量；②利用MIL框架实现路径级监督的无监督学习；③将估计得到的伪监督直接应用于LLM路径生成，提高路径质量。

**🔧 技术方法**

Transformer编码器、Multiple Instance Learning (MIL)、注意力聚合、KL蒸馏、LLM（如GPT‑4o、LLaMA3.1‑8B‑Instruct）生成路径与答案。

**📊 数据集**

WebQuestionsSP、ComplexWebQuestions、MetaQA（基于Freebase和WikiMovies）。

**📈 对比分析**

与prompting、无中间监督、弱监督路径、LLM‑精炼路径等SOTA方法比较，PathISE在WebQSP、CWQ、MetaQA上均达到或超过最高分（如WebQSP F1≈81.3，CWQ F1≈61.5，MetaQA F1≈96.2），同时在推理时仅需2次LLM调用，输入/输出token更少。

**⚠️ 局限性**

仍需答案级标签；路径生成器依赖LLM，可能受LLM规模限制；在极长路径或大规模KG上的可扩展性未完全验证。

---

## 813. MASS-DPO: Multi-negative Active Sample Selection for Direct Policy Optimization

**arXiv ID:** 2605.10784 | [PDF](https://arxiv.org/pdf/2605.10784v1)

**作者:** Rohan Surana `[一作]` (University of California San Diego), Junda Wu `[通讯]` (University of California San Diego)

**通讯引用:** 23783 | [OpenAlex ID](https://openalex.org/A5001019377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在 Plackett–Luce（PL）模型下提出 MASS-DPO，针对多负样本偏好优化（Multi‑Negative Preference Optimization）开发一种在每个 prompt 内进行主动负样本选择的框架。

**💡 创新点**

创新点主要包括：① 将 PL 的 Fisher‑信息量化为一个 log‑determinant 目标，用于衡量每个负样本对参数估计的独立贡献；② 把负样本选择视为 D‑optimal 设计问题，确保所选负样本在参数空间中覆盖多样化方向；③ 设计了一个增量式秩‑1 选择算法（利用 Sherman‑Morrison 更新）高效实现 log‑determinant 最优化；④ 在理论层面给出相对 logit 错误的上界。

**🔧 技术方法**

使用的技术包括：Plackett–Luce 排名模型、D‑optimal 设计与 Fisher‑信息、log‑determinant 优化、增量秩‑1 选择、Sherman‑Morrison 逆更新、DPO 与 S‑DPO 等对比方法。

**📊 数据集**

实验数据集涵盖四个基准：推荐任务的 LastFM 与 MovieLens，以及多选问答任务的 MedMCQA 与 QASC。

**📈 对比分析**

通过与传统 DPO、DPO‑k、S‑DPO、DMPO 等基线在 Accuracy、Margin、Chosen‑Rewards、Recall/NDCG、MRR 等指标下的对比，MASS‑DPO 在保持或超过准确率的同时显著减少负样本数（从 3 个到更少），并在多模型（Qwen3、SmolLM3、Llama3 等）和任务上取得更高或相当的表现。

**⚠️ 局限性**

局限性包括：① 需要预先构建负样本池并一次性做选择，缺乏训练过程中的动态自适应；② 依赖于 log‑线性策略的假设和 β、n 等超参数；③ 对极大候选集合的预处理成本仍然存在；④ 选取策略在不同任务和模型之间的通用性可能受限。

---

## 814. Policy Gradient Methods for Non-Markovian Reinforcement Learning

**arXiv ID:** 2605.10816 | [PDF](https://arxiv.org/pdf/2605.10816v1)

**作者:** Avik Kar `[一作]` (Indian Institute of Science), Nicholas Bambos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出Agent State-Markov（ASM）策略框架，并推导对应的非马尔可夫决策过程（NMDP）政策梯度，进而设计ASMPG算法实现端到端的reward-centric优化；

**💡 创新点**

将agent状态动态与控制策略联合视为可优化目标，首次在非马尔可夫环境中给出ASM政策梯度定理并提供理论收敛保证，突破传统依赖固定或预测目标的表征学习方法；

**🔧 技术方法**

采用REINFORCE类型的梯度估计，递归agent状态更新网络与策略网络；利用光滑性、梯度界等理论工具推导收敛率；

**📊 数据集**

五个公开/自定义非马尔可夫环境：CheeseMaze、HallwayNavigation、HealthcareTreatment、MachineRepair、VelocityOnlyCartPole；

**📈 对比分析**

与AIS-KL和AIS-MMD两种信息状态基线在相同表示维度下对比，使用同一discount因子及相同训练步骤；在所有五个环境中ASMPG平均奖励和best‑checkpoint性能均优于两基线，证明reward‑centric联合优化的优势；

**⚠️ 局限性**

基于REINFORCE的梯度估计导致方差较大，缺少高效的actor‑critic变体；理论仅保证收敛到局部极值，未证全球最优；在大规模环境下状态维度扩展性与计算成本尚待研究。

---

## 815. Conditional anomaly detection methods for patient-management alert systems

**arXiv ID:** 2605.10847 | [PDF](https://arxiv.org/pdf/2605.10847v1)

**作者:** Michal Valko `[一作]` (University of Pittsburgh), Miloš Hauskrecht `[通讯]` (University of Pittsburgh)

**通讯引用:** 4939 | [OpenAlex ID](https://openalex.org/A5012461386)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发并评估了一种基于条件异常检测的患者管理决策异常识别方法

**💡 创新点**

创新点在于将判别式投影（如SVM）用于构建异常度量，实现无监督的条件异常检测

**🔧 技术方法**

使用支持向量机（SVM）等判别式模型生成一维投影来量化异常

**📊 数据集**

使用从UPMC医疗中心提取的约34589条患者状态记录（共39,589条记录）

**📈 对比分析**

与基于规则的检测器进行ROC比较，SVM异常检测器在特异性94%、灵敏度49%、阳性预测值15.6%方面显著优于规则基线（PPV7.2%）

**⚠️ 局限性**

局限性包括仅在HIT数据集上验证、阈值调参对性能影响大、未评估跨机构的泛化能力

---

## 816. The Path-Extremal Conjecture for Zero Forcing: Distance-Hereditary Graphs and a Split-Decomposition Reduction

**arXiv ID:** 2605.10836 | [PDF](https://arxiv.org/pdf/2605.10836v1)

**作者:** Samuel German `[一作]` `[通讯]` (University of California, San Diego), Samuel German (University of California, San Diego)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过引入双胞胎（twin）与叶子递推的计数机制，证明了距离递归图（distance‑hereditary graph）在所有 n‑点图中零强制多项式系数按路径 P_n 进行上界，即路径是零强制数系数的极值图；随后在图的可拆分分解（split decomposition）框架下，给出了一个条件扩展：若一个唯一主包（prime bag）的标签图 H 及其所有诱导子图都是路径极值图，则所有具有该主包且仅含有该标签图（或其诱导子图）的连通图亦为路径极值图，从而把路径极值判定问题进一步归约到有限个小尺寸的“核心”图的验证。

**💡 创新点**

创新点在于：① 将双胞胎产生的 fort 结构与叶子递推两种计数机制统一到 split decomposition 的图标记树框架；② 通过唯一主包的条件性扩展，把全局路径极值问题降到对有限尺寸 split‑prime 核心图的验证；③ 明确了距离递归类之外的下一个结构前沿，并给出可行的混合归纳与有限验证策略。

**🔧 技术方法**

核心技术包括：零强制集合计数与多项式定义、fort 与双胞胎引起的非强制子集下界、叶子递推式、图的可拆分分解与图标记树的可达性关系、归纳证明与组合恒等式（Pascal 关系）以及对主包附近的叶子包的精细剖析。

**📊 数据集**

本文为纯理论研究，没有使用任何实验数据集。

**📈 对比分析**

比较方法是将任意 n‑点图的零强制多项式系数与路径 P_n 的系数逐项比较，证明前者不超过后者；结果为严格的系数级上界，验证了 Boyer 等的猜想在所给图类上的正确性。

**⚠️ 局限性**

局限性在于：① 结论仅适用于距离递归图及其满足唯一主包条件的图；② 对一般图仍未解决，需对所有 split‑prime 图（即使是小尺寸）进行全枚举验证；③ 方法对含有多个主包或更复杂结构的图仍不可直接推广。

---

## 817. Predicting 3D structure by latent posterior sampling

**arXiv ID:** 2605.10830 | [PDF](https://arxiv.org/pdf/2605.10830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 818. LLMs for Secure Hardware Design and Related Problems: Opportunities and Challenges

**arXiv ID:** 2605.10807 | [PDF](https://arxiv.org/pdf/2605.10807v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 819. From Controlled to the Wild: Evaluation of Pentesting Agents for the Real-World

**arXiv ID:** 2605.10834 | [PDF](https://arxiv.org/pdf/2605.10834v1)

**作者:** Pedro Conde `[一作]` (Ethiack), Nuno Moniz `[通讯]` (University of Notre Dame)

**通讯引用:** 61031 | [OpenAlex ID](https://openalex.org/A5068157871)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于已验证漏洞发现的AI渗透测试评估协议。

**💡 创新点**

创新点在于将评估焦点从任务完成转向漏洞发现，结合LLM语义匹配、双向匹配、持续维护真值集以及累积评估等多项技术。

**🔧 技术方法**

使用LLM作为评判者进行语义匹配、Hungarian算法进行双向匹配、重复运行统计、效率指标计算等技术。

**📊 数据集**

使用三款开源渗透测试引擎（Strix、PentAGI、Claude Code）与四种LLM后端，在vuln-bank、paygoat、xben-090等公开目标上，共计108条专家注释漏洞。

**📈 对比分析**

通过对比每个配置的精确率、召回率、F1、效率（运行时间、成本）以及累计指标，发现不同系统在发现率和误报率上存在显著差异，累积运行能提升召回但可能增加误报。

**⚠️ 局限性**

局限性包括未提供新的目标基准、未对跨运行记忆或目标演化进行实验、以及未涵盖安全性评估等方面。

---

## 820. Threat Modelling using Domain-Adapted Language Models: Empirical Evaluation and Insights

**arXiv ID:** 2605.10808 | [PDF](https://arxiv.org/pdf/2605.10808v1)

**作者:** Saba Pourhanifeh `[一作]` (Carleton University), Ashraf Matrawy `[通讯]` (Carleton University)

**通讯引用:** 1784 | [OpenAlex ID](https://openalex.org/A5103160400)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比评估了在5G安全领域进行STRIDE威胁建模时，域适配的LLM与通用LLM在不同规模、提示方式、是否使用聊天模板以及解码策略（贪心与随机采样）下的表现。

**💡 创新点**

发现仅靠域适配不足以显著提升威胁分类；提示方式（尤其是少量示例）能显著提升召回率；聊天模板更有利于保持输出格式一致性；随机采样揭示模型的不确定性和一致性问题。

**🔧 技术方法**

使用了8个开源LLM（3B-8B规模），包含通用和域适配版本；实验采用零样本与少样本提示；使用Greedy和Nucleus（top_p=0.9）采样；手工验证输出，计算F1、IOR及惩罚F1。

**📊 数据集**

使用6条来自Mahyoub等研究的5G威胁示例（覆盖STRIDE六类）作为评测数据；提示模板与类别定义来源于先前工作。

**📈 对比分析**

通过52种模型配置（8模型×2提示×2是否模板×2解码）对比，域适配模型并未总是优于通用模型；8B规模模型在贪心和采样场景下取得最高惩罚F1（≈68%），但差异并非单调；少样本提示在大模型中提升召回，但在小模型中易产生误判或无效输出；聊天模板虽不提升F1，但能显著降低无效输出率。

**⚠️ 局限性**

局限性包括：仅评估6条威胁，缺乏大规模真实攻击样本；模型规模受限（最多8B），未探索更大模型与Beam Search；缺乏对模型自我认识和不确定性自我说明的评估；评估手工耗时且易受主观影响；未针对STRIDE任务进行专门微调。

---

## 821. Can You Keep a Secret? Involuntary Information Leakage in Language Model Writing

**arXiv ID:** 2605.10794 | [PDF](https://arxiv.org/pdf/2605.10794v1)

**作者:** Ari Holtzman `[一作]` (University of Chicago), Peter West `[通讯]` (University of British Columbia)

**通讯引用:** 2401 | [OpenAlex ID](https://openalex.org/A5004903259)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过让语言模型在写作任务中隐藏一个秘密词，并让另一模型尝试推断该词，评估模型在文本中泄露隐藏信息的程度。

**💡 创新点**

提出“主题泄露”评估框架，系统比较多种模型、任务、提示方式以及主动隐藏对泄露率的影响，发现主动隐藏会导致泄露倒置并展示了可通过解码/转移的泄露机制。

**🔧 技术方法**

使用写作-猜测者实验，采用两种测量方法：20轮 free‑response 猜测和二选一强制判断（2AFC）；在 OpenRouter API 上跑多种前沿模型，统计泄露准确率。

**📊 数据集**

构建了15个人工挑选的秘密词（包括具体物体、抽象概念和中性词），并扩展到随机 COCA 词与模型自选词；写作任务包括短篇故事、长笑话、五段论述等。

**📈 对比分析**

通过 2AFC 统计泄露率，最大达 79%；主动隐藏时多数模型出现低于 50% 的倒置泄漏；模型规模越大泄露越明显；短笑话几乎无泄漏；跨模型猜测仍能保持高准确率，显示泄露信息可跨模型读取。

**⚠️ 局限性**

实验仅使用单词级别的秘密，任务为创意写作，真实场景下秘密更复杂；检测仅基于模型间猜测，缺乏人工评估；结果可能不适用于所有任务或模型，需进一步验证。

---

## 822. PriorVLA: Prior-Preserving Adaptation for Vision-Language-Action Models

**arXiv ID:** 2605.10925 | [PDF](https://arxiv.org/pdf/2605.10925v1)

**作者:** Xinyu Guo `[一作]` (Chinese Academy of Sciences), Xingyu Chen `[通讯]` (Zhongguancun Academy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出PriorVLA框架，在下游任务中保留预训练Vision‑Language‑Action模型的先验并利用其场景与运动先验进行高效适配

**💡 创新点**

创新点在于：①双行动专家结构（Prior Expert冻结、Adaptation Expert可训练），②专家查询机制（Scene、Motor、Action查询）将预训练先验提取并融入下游策略

**🔧 技术方法**

采用流匹配（flow‑matching）扩散策略、注意力遮罩设计与可学习查询接口，训练Adaptation Expert、查询组以及VLM视觉编码器，冻结其他参数

**📊 数据集**

在RoboTwin 2.0、LIBERO以及八个真实世界任务（Franka单臂、AC‑One双臂）上进行评估

**📈 对比分析**

与全微调和多种VLA基线（π₀、OpenVLA‑OFT、Diffusion Policy等）对比，PriorVLA在OOB与少样本情形下提升约10‑12 %（RoboTwin Hard），在LIBERO平均成功率99.1%，在真实任务标准数据ID/ OOD分别为81%/57%，少样本下为48%/32%，均显著优于基线

**⚠️ 局限性**

局限包括：只在RoboTwin 13任务子集上报告结果、OOD因素同时混合难以单独分析、在推理时需并行执行冻结的Prior Expert导致额外计算开销、对场景与运动先验演变的更细粒度分析尚待深入

---

## 823. Dynamic Skill Lifecycle Management for Agentic Reinforcement Learning

**arXiv ID:** 2605.10923 | [PDF](https://arxiv.org/pdf/2605.10923v1)

**作者:** Junhao Shen `[一作]` (Chinese University of Hong Kong), Hong Cheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 10283 | [OpenAlex ID](https://openalex.org/A5101984697)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态技能生命周期管理框架 SLIM，使 LLM 代理在强化学习过程中将外部技能集视为可随训练演化的优化变量；

**💡 创新点**

创新点在于通过留一技能验证估算每个活跃技能的边际外部贡献（MEC），并基于此执行保留、退休、扩展三种生命周期操作，避免技能永久累积或完全消失；

**🔧 技术方法**

使用的技术包括：GRPO 强化学习框架、层级化技能检索、留一技能验证、指数移动平均平滑、基于阈值的保留/退休/扩展策略；

**📊 数据集**

在 ALFWorld（家庭操作任务）和 SearchQA（问答任务）两个基准数据集上进行实验；

**📈 对比分析**

与 GRPO、SkillRL、Skill0 等多种基线对比，SLIM 在 ALFWorld 上平均提升 7.1% 成功率，最高达 87.5%；在 SearchQA 上同样获得 1.7% 的提升；表明该方法在不同任务中均能学习到更合适的外部技能边界；

**⚠️ 局限性**

局限性包括：实验仅覆盖文本/文本环境，未检验多模态或更大规模的技能库；生命周期阈值与成本模型为经验设定，缺乏理论最优性；审计频率与计算开销对大规模场景仍有挑战。

---

## 824. Confidence-Guided Diffusion Augmentation for Enhanced Bangla Compound Character Recognition

**arXiv ID:** 2605.10916 | [PDF](https://arxiv.org/pdf/2605.10916v1)

**作者:** Md. Sultan Al Rayhan `[一作]` (East West University), Maheen Islam `[通讯]` (East West University)

**通讯引用:** 786 | [OpenAlex ID](https://openalex.org/A5005949528)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种置信度引导的扩散模型数据增强框架，用于生成高质量的低分辨率Bangla复合字符图像，并通过分类器过滤提升训练集质量；随后使用融合后的数据重新训练多种分类器，显著提升识别准确率。

**💡 创新点**

创新点在于将分类器引导与扩散模型结合，利用Squeeze‑and‑Excitation增强的U‑Net骨干实现更好的特征表示；引入置信度阈值过滤机制，确保合成样本的类别一致性；整体框架以低分辨率（32×32）实现高效推理，且在多种网络架构上均能获得显著提升。

**🔧 技术方法**

核心技术包括：1) 类条件扩散模型（DDPM/DDIM）配合分类器引导；2) SE‑增强残差块的U‑Net结构；3) 基于预训练分类器的置信度过滤；4) 多模型训练（ResNet50、DenseNet121、VGG16、Vision Transformer）与再训练策略。

**📊 数据集**

使用AIBangla复合字符数据集，该数据集包含约25万张32×32灰度图，171个类别，已按训练/验证/测试划分。

**📈 对比分析**

通过在四种主流分类器上分别进行基线训练和增强后再训练，采用准确率、精确率、召回率、F1、FID等指标对比。结果表明，VGG16在增强后准确率提升至89.2%，显著超过原AIBangla基准；其余模型也表现出显著提升。

**⚠️ 局限性**

主要限制包括：1) 扩散采样仍耗时，尤其在低分辨率下仍需多步迭代；2) FID指标可能无法充分评估手写字符的语义质量；3) 该方法仅针对孤立复合字符，未扩展到完整手写文本识别；4) 低分辨率生成可能限制细节再现，影响极细字符的识别。

---

## 825. Shepherd: A Runtime Substrate Empowering Meta-Agents with a Formalized Execution Trace

**arXiv ID:** 2605.10913 | [PDF](https://arxiv.org/pdf/2605.10913v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 826. WildClawBench: A Benchmark for Real-World, Long-Horizon Agent Evaluation

**arXiv ID:** 2605.10912 | [PDF](https://arxiv.org/pdf/2605.10912v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 827. Beyond Red-Teaming: Formal Guarantees of LLM Guardrail Classifiers

**arXiv ID:** 2605.10901 | [PDF](https://arxiv.org/pdf/2605.10901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 828. How Creatives Approach GenAI Image Generation: Tensions Between Structured Guidance, Self-Experimentation, and Creative Autonomy

**arXiv ID:** 2605.10898 | [PDF](https://arxiv.org/pdf/2605.10898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 829. Grounded or Guessing? LVLM Confidence Estimation via Blind-Image Contrastive Ranking

**arXiv ID:** 2605.10893 | [PDF](https://arxiv.org/pdf/2605.10893v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 830. RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark

**arXiv ID:** 2605.10921 | [PDF](https://arxiv.org/pdf/2605.10921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 831. Using Logs to support Programming Education

**arXiv ID:** 2605.10920 | [PDF](https://arxiv.org/pdf/2605.10920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 832. Revisiting Policy Gradients for Restricted Policy Classes: Escaping Myopic Local Optima with $k$-step Policy Gradients

**arXiv ID:** 2605.10909 | [PDF](https://arxiv.org/pdf/2605.10909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 833. Neural Weight Norm = Kolmogorov Complexity

**arXiv ID:** 2605.10878 | [PDF](https://arxiv.org/pdf/2605.10878v1)

**作者:** Tiberiu Musat `[一作]` `[通讯]` (ETH Zürich), Tiberiu Musat (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

证明了在固定精度循环网络中最小权重范数与输出字符串的 Kolmogorov 复杂度相匹配，揭示权重衰减与 Solomonoff 通用先验的对应关系。

**💡 创新点**

提供了一个两侧的定量 sandwich 约束，并证明了对数因子是不可或缺的；同时该结果对任意 Lp 范数在固定精度下均成立。

**🔧 技术方法**

利用程序到网络和网络到程序的两步归约、稀疏参数编码、置换例子和固定精度论证等技术。

**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**

仅为理论证明，常数较大；仅适用于循环网络、从零起始、固定精度；未验证实际训练效果，无法直接应用于常规监督学习。

---

## 834. Local Private Information Retrieval: A New Privacy Perspective for Graph-Based Replicated Systems

**arXiv ID:** 2605.10872 | [PDF](https://arxiv.org/pdf/2605.10872v1)

**作者:** Shreya Meel `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**通讯引用:** 14012 | [OpenAlex ID](https://openalex.org/A5021132487)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在图形复制数据库下的局部用户隐私私有信息检索（local PIR），提出新的隐私定义并求解其容量

**💡 创新点**

创新点在于将隐私需求从“隐藏所有索引”转化为“仅隐藏服务器实际存储的消息索引”，从而显著提高通信效率，并首次给出多种图结构（循环图、路径图、星形图、完整图、完整二分图）下的容量下界与上界甚至精确容量

**🔧 技术方法**

主要技术包括信息论容量分析、图论（图的并集、边传递性、二分性）、组合编码与随机置换、以及对局部隐私约束的改写和证明

**📊 数据集**

该工作为理论研究，无需具体数据集，所有结论均基于抽象的消息符号模型和图结构

**📈 对比分析**

通过比较可实现的下界和已知的上界，证明了局部PIR在多种图结构下的容量显著大于传统PIR；例如循环图的容量为1/2，路径图奇数节点时容量为(N-1)/(2N-4)；星形图达到容量1，优于传统PIR的Θ(1/√N)

**⚠️ 局限性**

局限性包括：仅针对简单无向图；对于一般图或多重图的完整容量尚未确定；部分下界仍未证明最优；在实际系统中如何实现这些理论编码方案仍需进一步探索

---

## 835. RUBEN: Rule-Based Explanations for Retrieval-Augmented LLM Systems

**arXiv ID:** 2605.10862 | [PDF](https://arxiv.org/pdf/2605.10862v1)

**作者:** Joel Rorseth `[一作]` (University of Waterloo), Jarek Szlichta `[通讯]` (York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 RUBEN 工具，用于自动生成最小化规则来解释检索增强型 LLM 的输出，并评估其安全性。

**💡 创新点**

创新点在于将规则挖掘与 RAG 结合，利用基于层次结构的剪枝算法高效发现最小规则，并通过规则概括多种反事实。

**🔧 技术方法**

采用了基于频繁项集的 Apriori 机制的规则挖掘算法、前端 React + Material UI、后端 FastAPI 与 Python，并集成 LLM 推理。

**📊 数据集**

使用了金融网页搜索结果、Stack Overflow 代码片段以及公司内部员工信息等示例数据作为 RAG 源；实验中没有公开统一数据集。

**📈 对比分析**

通过对比弱与强 LLM 的规则发现结果来评估安全训练的鲁棒性；弱模型生成最小规则，而强模型无规则，证明鲁棒性提高；算法在规则数较少时能在秒级完成。

**⚠️ 局限性**

局限包括对 LLM 随机性和输出判断器一致性的依赖，规则挖掘对大规模 RAG 集合的扩展性有限，以及对复杂安全约束的可解释性不足。

---

## 836. RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards

**arXiv ID:** 2605.10899 | [PDF](https://arxiv.org/pdf/2605.10899v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 837. Engineering Robustness into Personal Agents with the AI Workflow Store

**arXiv ID:** 2605.10907 | [PDF](https://arxiv.org/pdf/2605.10907v1)

**作者:** Roxana Geambasu `[一作]` (Columbia University), Wen Zhang `[通讯]` (Google)

**通讯引用:** 154306 | [OpenAlex ID](https://openalex.org/A5100399276)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并阐述了 AI Workflow Store 的概念与架构，旨在将传统软件工程（需求、设计、实现、测试、对抗评估、分阶段部署）嵌入到 AI 代理的执行循环中，从而生成可复用、经过加固的工作流，提升代理在高风险场景下的可靠性与安全性。

**💡 创新点**

创新点主要包括：
1) 将严格的软件工程流程自动化并整合到 AI 代理工作流的生命周期；
2) 构建共享工作流仓库（Workflow Store），通过工作流重用摊薄工程成本；
3) 通过“可编排工作流”而非即时生成计划，缓解 prompt‑injection、幻觉、脆弱性等问题；
4) 在灵活性与鲁棒性之间提出明确的取舍框架，并定位其在现有解决方案中的优越位置。

**🔧 技术方法**

使用技术与方法：
- 传统软件工程生命周期（需求收集、威胁建模、设计、实现、测试、对抗评估、部署）。
- 大语言模型（LLM）辅助的规划、设计和脚本生成。
- 工作流抽象与参数化，支持版本化、去重、泛化。
- 本地代理对请求的匹配与工作流调用。
- 自动化的安全分析、模糊测试与动态污点追踪等。

**📊 数据集**

论文未进行实验，未使用公开数据集；作者主要基于现实案例（如邮件中的预订请求）进行示例说明，讨论了典型的输入（电子邮件、工具 API）和输出。

**📈 对比分析**

对比方法：仅通过概念性与案例性比较，说明传统 on‑the‑fly 代理容易出现错误、攻击和安全缺陷，而经过工程化的工作流能显著降低这些风险。没有提供定量指标或性能基准，重点强调可重用性与成本摊薄带来的长期收益。

**⚠️ 局限性**

局限与挑战：
- 初始工程成本高、需要投入人力与计算资源；
- 需要大量的工作流重用才能真正实现成本摊薄；
- 自动化软件工程流程仍面临难度（设计审查、对抗测试自动化难度大）。
- 用户体验方面，工作流的“预先设定”可能降低即时灵活性。
- 真实世界中的安全与可靠性评估尚未经过大规模实验验证。

---

## 838. Masked Generative Transformer Is What You Need for Image Editing

**arXiv ID:** 2605.10859 | [PDF](https://arxiv.org/pdf/2605.10859v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 839. Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding

**arXiv ID:** 2605.10855 | [PDF](https://arxiv.org/pdf/2605.10855v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 840. Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges

**arXiv ID:** 2605.10917 | [PDF](https://arxiv.org/pdf/2605.10917v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 841. V4FinBench: Benchmarking Tabular Foundation Models, LLMs, and Standard Methods on Corporate Bankruptcy Prediction

**arXiv ID:** 2605.10896 | [PDF](https://arxiv.org/pdf/2605.10896v1)

**作者:** Marcin Kostrzewa `[一作]` (Wrocław University of Science and Technology), Maciej Zięba `[通讯]` (Wrocław University of Science and Technology)

**通讯引用:** 1560 | [OpenAlex ID](https://openalex.org/A5083652196)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

发布了V4FinBench，一个涵盖约110万公司年度记录的公开公司破产预测基准，包含131个财务特征、六个预测时段和统一的财务困境标签。

**💡 创新点**

提出了多时段破产预测基准与公开评估协议，并演示了基于不平衡数据的TabPFN上下文构造技巧可与梯度提升树竞争，同时证明该基准在跨国迁移上具备一定通用性。

**🔧 技术方法**

使用了TabPFN基础模型（不平衡上下文采样）、LLM Llama‑3‑8B（QLoRA微调）以及传统梯度提升树（XGBoost、CatBoost、LightGBM）等技术。

**📊 数据集**

基于V4经济体（波兰、匈牙利、捷克、斯洛伐克）2006‑2021年的EMIS财务报表数据，外部对比使用美国破产数据集。

**📈 对比分析**

通过5折分组交叉验证与阈值校准，对F1和ROC‑AUC进行评估；结果显示，采用原型下采样的TabPFN在所有时段的ROC‑AUC均与梯度提升树持平或更优，F1在两年以上时段超越；相比之下，QLoRA‑Llama‑3‑8B在任何时段均落后梯度提升树。

**⚠️ 局限性**

局限性包括仅覆盖四个中欧国家，标签为财务困境而非正式破产，且模型尚未在多模态文本等更丰富信息上验证，需进一步在不同会计制度与地区进行评估。

---

## 842. Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why

**arXiv ID:** 2605.10889 | [PDF](https://arxiv.org/pdf/2605.10889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 843. FPT Approximation Schemes for Min-Sum Radii and Min-Sum Diameters Clustering

**arXiv ID:** 2605.10895 | [PDF](https://arxiv.org/pdf/2605.10895v1)

**作者:** Fabrizio Grandoni `[一作]` (IDSIA), Jatin Yadav `[通讯]` (IIT Delhi)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在两类经典聚类问题——最小化半径和直径之和（Min‑Sum Radii / Min‑Sum Diameters）上，作者提出了两种新的固定参数近似（FPT Approximation Scheme, FPT‑AS）算法，分别在参数 k（聚类数）上实现 (1+ε) 近似。

**💡 创新点**

创新点在于：
• 设计了“defensive clustering”结构，能够在保证 2‑近似的同时为后续递归提供精确的核心信息；
• 通过对“扩张球”（expanded ball）的巧妙递归分裂（cheap 与 large 扩张），实现了对最优解的逼近，并通过猜测少数最重要的最优球来压缩搜索空间；
• 采用随机阈值分割和 min,+ 级联，随后再进行全局求解，最终得到两种问题的 FPT‑AS。

**🔧 技术方法**

核心技术包括：
• 参数化近似框架（FPT）与分治递归；
• 基于球覆盖的构造与扩张策略；
• 通过枚举候选 defensive clusterings 与核心球，保证存在一组满足条件的解；
• 结合随机化与 derandomization 技巧，实现 (1/ε)^k n^O(1) 与 (1/ε)^{O(k/ln(1/ε))} n^{1/ε} 的时间复杂度。

**📊 数据集**

本文为理论论文，并未在实际数据集上进行实验验证；所有结果均在理论上给出。

**📈 对比分析**

与先前的 4+（Min‑Sum Radii）和 2+（Min‑Sum Diameters）FPT 近似相比，本文的算法在相同参数 k 上显著降低了近似误差至 (1+ε)，并给出了更精细的时间复杂度；但仍保持了指数级的 1/ε 依赖。

**⚠️ 局限性**

局限性：
• 运行时间中仍包含 1/ε 的指数因子，实际可用性受限；
• 对于最小化半径/直径之和的精确算法仍未获得 FPT 时间；
• 仅在理论层面证明，没有实验或实际数据集验证；
• 对于 mergeable 约束的 FPT 近似仍只能得到 (2+)-近似，尚未实现 PTAS。

---

## 844. Remember the Decision, Not the Description: A Rate-Distortion Framework for Agent Memory

**arXiv ID:** 2605.10870 | [PDF](https://arxiv.org/pdf/2605.10870v1)

**作者:** Mingxi Zou `[一作]` (Fudan University), Zenglin Xu `[通讯]` (Shanghai Academy of AI for Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种面向决策的记忆压缩框架，并在此基础上设计了在线记忆学习方法DeMem，能够在有限运行内存预算下只保留对决策重要的历史区分；

**💡 创新点**

创新点在于将记忆视为决策损失的压缩问题，给出精确的遗忘边界、记忆-失真前沿，并提出在数据中通过“决策冲突”证据进行自证式分割的在线记忆学习算法；

**🔧 技术方法**

核心技术包括决策导向的速率失真理论、基于决策距离的覆盖/打包分析、置信度可证的分割与图着色、以及UCB级别的学习与置信探索；

**📊 数据集**

实验使用合成的Decoupled Bandit环境以及三大长时序对话基准LoCoMo、LongMemEval和MemoryArena，并在多种LLM骨干（GPT‑4o-mini、GPT‑4.1‑mini、Llama‑3.1‑70B）上进行评测；

**📈 对比分析**

与Oracle、Feature‑KMeans、Feature‑RAG、CLUB、RAG、LangMem等对齐与描述性记忆方法对比，DeMem在固定记忆槽预算下在所有基准上均取得最高或近乎最高的得分，尤其在跨会话、开放域与多跳问题上显著优于对照；

**⚠️ 局限性**

局限性包括对决策冲突检测的统计置信要求导致额外的探索开销、在极端描述与决策不匹配极少的场景下性能提升有限，以及理论假设（i.i.d.上下文）对实际多变对话环境的适用性需进一步验证。

---

## 845. A Unary-to-Nonunary Transition in the Accepting-State Spectrum of Right Quotient for Permutation Automata

**arXiv ID:** 2605.10852 | [PDF](https://arxiv.org/pdf/2605.10852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

---

## 846. Effective, Efficient, and General Information Abstraction for Imperfect-Information Extensive-Form Games

**arXiv ID:** 2605.10900 | [PDF](https://arxiv.org/pdf/2605.10900v1)

**作者:** Boning Li `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**通讯引用:** 3773 | [OpenAlex ID](https://openalex.org/A5082905458)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种利用少量 CFR 暖起迭代提取期望值特征并进行 k‑means++ 聚类的通用信息抽象方法 WEVA。

**💡 创新点**

创新点在于：①无需领域知识或海量数据训练网络，直接从短暂的 CFR 暖起阶段获得可靠的 EV 特征；②通过深度加权多节点特征捕捉策略差异，提升抽象质量；③实现开源、易复现且计算开销极低。

**🔧 技术方法**

采用的技术包括：Counterfactual Regret Minimization (CFR) 进行暖起与最终求解；期望值 (EV) 计算；深度加权多维特征构造；k‑means++ 聚类；以及 PCFR+ 与 DCFR 两种主流 CFR 变体。

**📊 数据集**

实验数据集涵盖三类结构多样的游戏：Heads‑up No‑limit Hold’em (HUNL) 河局、双板 HUNL（双独立棋盘）以及完全随机收益的 Random Game，分别在 20、50、200 个桶级别下测试。

**📈 对比分析**

与传统的 equity、rank、rank‑2d 等基线相比，WEVA 在所有游戏和桶数下均实现了显著降暴露（Exploitability）——最高可降低 80% 以上；即使仅用 10 次暖起迭代，性能已接近 500 次迭代的结果，证明其高效性。

**⚠️ 局限性**

局限性包括：①在极细粒度（K=200）下需要更多暖起迭代才能充分稳定特征；②目前仅在三种游戏上验证，未知在更大规模或更深树的游戏中的效果；③k‑means++ 对初始种子敏感，可能导致聚类质量不一致。

---

## 847. Neural at ArchEHR-QA 2026: One Method Fits All: Unified Prompt Optimization for Clinical QA over EHRs

**arXiv ID:** 2605.10877 | [PDF](https://arxiv.org/pdf/2605.10877v1)

**作者:** Abrar Majeedi `[一作]`, Siddhant Rai `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种模块化的基于大语言模型的全流程临床问答系统 Neural1.5，自动完成患者问题的精准改写、证据句子识别、答案生成以及证据对齐。

**💡 创新点**

创新点包括：1) 对每个子任务独立应用 DSPy+MIPROv2 的自动化提示优化；2) 通过自一致性投票提升证据识别稳定性；3) 三阶段对齐流水线（初始对齐、自我反思、链式验证）结合置信度加权多数投票，显著降低错误引用。

**🔧 技术方法**

技术手段主要是：GPT‑4.1 大语言模型、DSPy 框架、MIPROv2 提示优化器、LLM-as-Judge 评估器、自一致性投票、置信度加权多数投票。

**📊 数据集**

使用 ArchEHR‑QA 2026 数据集（基于 MIMIC 电子病历的患者问题、临床笔记片段、证据句子标签和参考答案）。

**📈 对比分析**

与13支参赛队伍对比，Neural1.5 在四个子任务中平均排名第4（仅次于 OptiMed），在子任务2（证据识别）获得 63.7% 的 Strict Micro F1，排名第一；子任务3（答案生成）排名第4；子任务4（证据对齐）排名第7；子任务1（问题改写）排名第4。

**⚠️ 局限性**

局限性包括：1) 各子任务相对独立，未充分利用跨任务信息；2) 自一致性投票显著增加计算成本；3) 置信度阈值在小开发集上调优，泛化性待验证；4) 依赖 GPT‑4.1，缺乏开源实现；5) 过度强调精确度导致召回率下降，可能不适合所有临床场景。

---

## 848. Pixal3D: Pixel-Aligned 3D Generation from Images

**arXiv ID:** 2605.10922 | [PDF](https://arxiv.org/pdf/2605.10922v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 849. DGPO: Beyond Pairwise Preferences with Directional Consistent Groupwise Optimization

**arXiv ID:** 2605.10863 | [PDF](https://arxiv.org/pdf/2605.10863v1)

**作者:** Mengyi Deng `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 253022 | [OpenAlex ID](https://openalex.org/A5100444820)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Directional-Groupwise Preference Optimization (DGPO) 框架，通过构造前向与反向推理对的组级数据，实现方向一致性和推理多样性的联合优化。

**💡 创新点**

创新点在于：① 以组级别聚合监督，显式区分方向一致与不一致的候选答案；② 引入 Beta 分布一致性估计与不确定性正则化，提升对方向信息的判别能力；③ 结合方向一致性与多样性，构造对比损失，提升模型对齐效果。

**🔧 技术方法**

使用了方向一致性估计网络、Beta 分布置信头、方差正则、KL 正则、组级对比损失；教师模型 DeepSeek V3 与 Qwen3-32B 用于生成前向与反向问题及多路径解答。

**📊 数据集**

基于 LIMO 817 题的前向数据，构造对应的逆向数据并生成 1,634 组；在 AIME-25、GPQA、Math 500、GMQ、LMGH 等 benchmark 上评估性能。

**📈 对比分析**

与传统 DPO、β-DPO、γ-DPO、SimPO 等方法对比，在 Qwen3-1.7B-Base 上平均准确率提升 1%–3.6%，在混合训练的 SFT 模型上进一步提升 2.2%，在 AIME-25 与 GPQA 等 benchmark 上显著提高准确率。

**⚠️ 局限性**

主要限制是逆向问题不一定严格可逆，部分逆向样本可能无明确答案；对低容量模型加入过多逆向组会导致干扰，需在数据规模与模型容量之间取得平衡。

---

## 850. Random Access Expectation in DNA Storage and Fountain Codes

**arXiv ID:** 2605.10919 | [PDF](https://arxiv.org/pdf/2605.10919v1)

**作者:** Christoph Hofmeister `[一作]` (Technical University of Munich), Eitan Yaakobi `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了DNA存储中基于线性码的随机访问期望，并针对完全对称生成矩阵进行了优化。

**💡 创新点**

提出了DNA随机访问期望与LT码等价的关系，给出了最优度分布并实现了接近理论下限的随机访问期望。

**🔧 技术方法**

利用LT码的编码抽样、逐层剥离（peeling）解码、凸优化（KKT条件）与数值仿真等技术。

**📊 数据集**

未使用实际数据集，主要通过理论推导和数值模拟进行验证。

**📈 对比分析**

与以往π^2/12≈0.8225的结果相比，本文在极限情况下取得≈0.7869的期望，接近下限π/4≈0.7854，显示出更优性能。

**⚠️ 局限性**

局限在仅考虑二进制码、peeling解码、极限（k→∞）分析，未给出有限长度或更大字母表的结果。

---

## 851. Safe Aerial 3D Path Planning for Autonomous UAVs using Magnetic Potential Fields

**arXiv ID:** 2605.10880 | [PDF](https://arxiv.org/pdf/2605.10880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 852. LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

**arXiv ID:** 2605.10886 | [PDF](https://arxiv.org/pdf/2605.10886v1)

**作者:** Liang Luo `[一作]` (Meta AI), Chunqiang Tang `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了LoKA框架，使FP8能够在大规模推荐模型（LRM）上安全、高效地进行训练和推理，并已部署到生产环境；

**💡 创新点**

创新点包括：① 分布感知层级误差评估，识别FP8安全区；② 与硬件共设计模型模块（去偏、BlockNorm、HardSwish）提升数值稳定性；③ 运行时基于每层误差与速度动态选择最佳FP8库/recipe，实现Per‑Operator最优调度；

**🔧 技术方法**

使用了在线统计学习（多元正态/矩阵正态）获取激活/权重分布，MERE误差度量，动态FP8库调度，量化/解量化，硬件特定FP8内核（DeepGEMM/TorchAO/FBGEMM），自定义PyTorch Autograd适配，以及BlockNorm、HardSwish、无偏优化等技术；

**📊 数据集**

利用真实生产广告推荐数据集，包含数十亿样本、上千特征，覆盖Wukong、InterFormer、ELFM三大模型族；

**📈 对比分析**

在H100、B200、MI300X、GB200、MI350X等多代GPU上与BF16/TF32基线对比，FP8训练保持无质量下降，训练吞吐量提升最高1.19×，推理速度提升最高1.4×；在生产部署中实现5–20%吞吐提升、10–17%推理加速；

**⚠️ 局限性**

局限性包括：需依赖标准构件，对非标准模块支持不足；对误差传播缺乏建模导致过于保守；仅针对FP8，FP4等更低精度仍待研究；需要手动集成新库，难以自动覆盖所有平台；通信与计算交织的真实延迟难以完全覆盖。

---

## 853. AssayBench: An Assay-Level Virtual Cell Benchmark for LLMs and Agents

**arXiv ID:** 2605.10876 | [PDF](https://arxiv.org/pdf/2605.10876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 854. CADBench: A Multimodal Benchmark for AI-Assisted CAD Program Generation

**arXiv ID:** 2605.10873 | [PDF](https://arxiv.org/pdf/2605.10873v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 855. BenchCAD: A Comprehensive, Industry-Standard Benchmark for Programmatic CAD

**arXiv ID:** 2605.10865 | [PDF](https://arxiv.org/pdf/2605.10865v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 856. Is Your Driving World Model an All-Around Player?

**arXiv ID:** 2605.10858 | [PDF](https://arxiv.org/pdf/2605.10858v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 857. BEACON: A Multimodal Dataset for Learning Behavioral Fingerprints from Gameplay Data

**arXiv ID:** 2605.10867 | [PDF](https://arxiv.org/pdf/2605.10867v1)

**作者:** Ishpuneet Singh `[一作]` (Thapar Institute of Engineering and Technology), Maninder Singh `[通讯]` (Thapar Institute of Engineering and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并发布了BEACON数据集——一个在Valorant电竞游戏中同步采集鼠标、键盘、网络包和屏幕录像的多模态大规模行为数据集，并对其进行了基线身份识别实验。

**💡 创新点**

创新点在于：①在高压电竞环境下收集极高频率、多模态行为数据；②通过自研低延迟日志框架实现实时同步采集；③为连续身份验证提供了前所未有的真实世界基准。

**🔧 技术方法**

主要技术包括：自定义多线程日志系统、POSIX时间戳同步、网络抓包（libpcap）、屏幕录制（ffmpeg）与深度学习模型（Var‑CNN、NetCLR、TCN 等）对时序特征进行识别。

**📊 数据集**

使用的数据集是本文自己构建的BEACON数据集（28名玩家、79个会话、约430 GB同步数据），对照实验亦参考了AMuCS等公开电竞/桌面行为数据集。

**📈 对比分析**

实验方法是将原始行为记录提取出 33 个统计特征，构建 10/30/45/60 秒窗口的时间序列样本，采用六种网站指纹模型进行 28 类身份识别。单模态鼠标性能最高，准确率最高达63%；键盘仅 36%；早期融合后最高 70% 准确率、EER 4.3%。

**⚠️ 局限性**

局限性包括：样本量仅 28 人，缺乏人口多样性和跨平台/跨游戏泛化；未对屏幕录像和网络包进行基准评测；仅为单时点采集，未覆盖长期行为漂移。

---

## 858. DataMaster: Towards Autonomous Data Engineering for Machine Learning

**arXiv ID:** 2605.10906 | [PDF](https://arxiv.org/pdf/2605.10906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 859. TLX: Hardware-Native, Evolvable MIMW GPU Compiler for Large-scale Production Environments

**arXiv ID:** 2605.10905 | [PDF](https://arxiv.org/pdf/2605.10905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 860. MDrive: Benchmarking Closed-Loop Cooperative Driving for End-to-End Multi-agent Systems

**arXiv ID:** 2605.10904 | [PDF](https://arxiv.org/pdf/2605.10904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 861. Grounded Satirical Generation with RAG

**arXiv ID:** 2605.10853 | [PDF](https://arxiv.org/pdf/2605.10853v1)

**作者:** Oona Itkonen `[一作]` (University of Helsinki), Ona De Gibert `[通讯]` (University of Helsinki)

**通讯引用:** 75 | [OpenAlex ID](https://openalex.org/A5002349953)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于检索增强生成（RAG）的芬兰语境讽刺词典定义生成管线，并对其进行人工与LLM评估。

**💡 创新点**

创新点在于：① 将最新新闻检索与RAG相结合生成讽刺定义；② 提出了针对幽默与政治性的新评估框架；③ 采用LLM作为评判者进行自动化评估。

**🔧 技术方法**

使用了检索增强生成（RAG）、主题建模（BERTopic）、语义检索、情感分析以及多模型LLM评判（Qwen2.5、Llama‑3.1、Mistral、Aya‑Expanse、EuroLLM）。

**📊 数据集**

采用的主要数据集是芬兰公共广播 YLE 英文新闻（最近30天），并人工标注了100条生成的讽刺定义。

**📈 对比分析**

通过与无RAG、随机词选择的对照实验以及人类与LLM双重评分，发现 RAG 与主题词提升了政治相关性但幽默度无显著提升；LLM 在政治维度与人类评分高度相关，幽默维度相关性低。

**⚠️ 局限性**

局限性包括：仅针对芬兰英语新闻，评估维度仅限幽默与政治，缺乏跨文化深度；LLM 对幽默判断仍表现不佳。

---

## 862. CapVector: Learning Transferable Capability Vectors in Parametric Space for Vision-Language-Action Models

**arXiv ID:** 2605.10903 | [PDF](https://arxiv.org/pdf/2605.10903v1)

**作者:** Wenxuan Song `[一作]` (HKUST (GZ)), Haoang Li `[通讯]` (HKUST (GZ))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了能力向量（Capability Vectors）概念，并通过参数算术提取并将其融合进预训练的视觉‑语言‑动作（VLA）模型，从而在不增加额外辅助目标的情况下获得与辅助‑目标细调相当甚至更优的性能；

**💡 创新点**

创新点在于：①利用参数差异直接提取可迁移的通用能力向量；②将向量与预训练模型相加生成元模型，并在后续标准细调中加入正交正则化以防止能力消失；③验证了该方法在多种模型架构、任务类型和跨域、跨机器人平台的通用性与实用性；

**🔧 技术方法**

技术主要包括：参数算术提取、能力向量融合、正交正则化（Orthogonal Loss）、低秩适配（LoRA）以及标准的动作与视觉语言训练框架；

**📊 数据集**

使用的主要数据集为LIBERO（Spatial、Object、Goal、Long等四套任务）与RoboTwin 2.0（10个清晰背景任务），以及真实工业任务数据（UR3机器人上收集的3个工业操纵任务）和外部协作实验中的ARX Lift 2与AgileX Cobot；

**📈 对比分析**

与标准SFT（OpenVLA‑OFT、StarVLA、π_0.5）及使用辅助目标的SFT（Spatial Forcing、LaRA‑VLA）对比，实验显示融合能力向量的元模型在训练步数相同或更少的情况下，成功率提升数个百分点，甚至在150k步时超越辅助目标细调；跨域与跨机器人实验亦证明了显著提升；

**⚠️ 局限性**

局限性包括：①能力向量的质量高度依赖于提取任务的数据多样性与视觉丰富度，过度任务相关或差异过小的任务会导致捷径学习；②若不使用正交正则化，细调过程中可能导致能力衰减；③该方法对极其复杂或高维任务的泛化尚需进一步验证；

---

## 863. Counterfactual Stress Testing for Image Classification Models

**arXiv ID:** 2605.10894 | [PDF](https://arxiv.org/pdf/2605.10894v1)

**作者:** Moritz Stammel `[一作]` (Imperial College London), Ben Glocker `[通讯]` (Imperial College London)

**通讯引用:** 45079 | [OpenAlex ID](https://openalex.org/A5007222325)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了基于因果生成模型的反事实压力测试框架，用来评估医学影像分类模型在分布偏移（如扫描仪类型、患者性别等）下的鲁棒性。

**💡 创新点**

创新点在于利用深层结构因果模型（DSCM）生成保持解剖一致性的真实反事实图像，从而提供比传统随机扰动更具临床现实性的鲁棒性评估。

**🔧 技术方法**

核心技术包括深层结构因果模型（DSCM）实现为层级β-变分自编码器（HVAE），以及经典图像扰动（Gamma、对比度、亮度、锐化、高斯模糊）做对照。

**📊 数据集**

实验使用了两大公开医学影像数据集：PadChest（胸部X光）和EMBED（乳腺摄影）来验证方法。

**📈 对比分析**

通过比较模型在人工生成的反事实样本上的性能变化与在真实外部分布样本上的表现，发现反事实测试的预测误差显著低于传统扰动测试，相关系数最高可达0.95，表明该方法更能准确预估实际部署中的性能衰退。

**⚠️ 局限性**

局限性包括：需预先定义因果图并满足可识别性假设；目前仅支持单属性干预；未覆盖三维影像或多属性联合干预的场景。

---

## 864. CppPerf: An Automated Pipeline and Dataset for Performance-Improving C++ Commits

**arXiv ID:** 2605.10890 | [PDF](https://arxiv.org/pdf/2605.10890v1)

**作者:** Tommy Ho `[一作]` (ETH Zurich), Zhendong Su `[通讯]` (ETH Zurich)

**通讯引用:** 14561 | [OpenAlex ID](https://openalex.org/A5077610917)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可配置的流水线，用于从GitHub的C++开源仓库中挖掘可执行的执行时间提升修补，并基于此创建了347个手工验证的性能修复补丁基准集（CppPerf）

**💡 创新点**

创新点在于：①将LLM与结构化过滤相结合实现自动识别性能提升提交；②通过容器化构建+测试实现可复现的执行时间改进补丁；③提供可扩展的基准构建工具与公开数据集

**🔧 技术方法**

主要技术包括：Python实现的流水线、LLM（例如GPT‑4o / Llama）进行提交分类、Docker化构建与多次测试、Mann‑Whitney检验统计验证性能改进

**📊 数据集**

使用了从GitHub检索的42个成熟C++仓库（星级306–28,718）中提取的347个补丁；每个补丁均包含对应的Docker镜像和执行时间统计数据

**📈 对比分析**

与最先进的代理式修复工具（如SWE‑Agent）对比，工具仅能在347个补丁中成功生成语义等价修复的13.5%（单文件17.5%，多文件7.4%），说明真实世界C++性能修复的难度仍很高；LLM分类器精度为86.7%，召回率约42%

**⚠️ 局限性**

局限性包括：仅适用于使用CMake构建的项目；依赖现有测试用例来评估性能改进，导致多数补丁缺乏可测量的性能提升；LLM分类器偏向精度导致召回率低，部分真实性能改进补丁被过滤掉

---

## 865. Shields to Guarantee Probabilistic Safety in MDPs

**arXiv ID:** 2605.10888 | [PDF](https://arxiv.org/pdf/2605.10888v1)

**作者:** Linus Heck `[一作]` (Radboud University), Sebastian Junges `[通讯]` (Radboud University)

**通讯引用:** 2205 | [OpenAlex ID](https://openalex.org/A5018941708)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种形式化框架，扩展了经典的保护机制，以确保在马尔可夫决策过程（MDP）中的概率安全性。该框架展示了如何在允许一定概率的不安全事件发生的情况下，仍然保持安全性和最大宽容性。

**💡 创新点**

创新点在于提出了乐观和悲观保护机制，以及饱和安全保护机制，这些机制在保证安全性的同时，能够提供更大的宽容性，并且可以通过在线和离线学习方法进行构建。

**🔧 技术方法**

使用了马尔可夫决策过程（MDP）作为基础模型，结合了历史依赖的保护机制，提出了乐观和悲观保护的计算方法。

**📊 数据集**

使用了多种网格环境（如走廊和无人机送货任务）作为数据集进行实验，评估不同保护机制的性能。

**📈 对比分析**

通过与经典保护机制和δ保护机制的比较，实验结果表明，本文提出的保护机制在安全性和宽容性方面表现更优，尤其是在允许的动作比例上更高。

**⚠️ 局限性**

限制在于构建宽容性保护机制可能需要大量的历史-动作对数据，且在某些情况下，内存无关的保护机制可能不如有记忆的保护机制宽容。

---

## 866. Count Anything at Any Granularity

**arXiv ID:** 2605.10887 | [PDF](https://arxiv.org/pdf/2605.10887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 867. Geometry-aware Prototype Learning for Cross-domain Few-shot Medical Image Segmentation

**arXiv ID:** 2605.10885 | [PDF](https://arxiv.org/pdf/2605.10885v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 868. Private Information Retrieval With Arbitrary Privacy Requirements for Graph-Based Storage

**arXiv ID:** 2605.10879 | [PDF](https://arxiv.org/pdf/2605.10879v1)

**作者:** Mohamed Nomeir `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**通讯引用:** 14012 | [OpenAlex ID](https://openalex.org/A5021132487)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

重新定义了私密信息检索（PIR）问题中的隐私概念，以适应灵活的隐私需求，重点关注图形复制的PIR，允许每个服务器有不同的隐私要求。

**💡 创新点**

提出了一种新的隐私要求集，允许每个服务器存储的消息索引可以是所有消息索引的任意子集，从而实现更灵活的隐私保护。

**🔧 技术方法**

使用了图论中的路径图和循环图作为存储设置，并推导了不同隐私设置下的容量界限。

**📊 数据集**

未具体提及使用的数据集，但讨论了消息和服务器的抽象模型，涉及K个消息和N个服务器的组合。

**📈 对比分析**

与传统的全复制PIR和局部PIR进行了比较，结果表明在某些设置下，新的隐私要求集可以提高容量，尤其是在路径图和循环图的特定隐私设置下。

**⚠️ 局限性**

局限性在于，虽然提出了灵活的隐私要求，但在实际应用中，如何有效实现这些要求仍然是一个挑战，尤其是在数据存储的异构性和复杂性方面。

---

## 869. Compute Where it Counts: Self Optimizing Language Models

**arXiv ID:** 2605.10875 | [PDF](https://arxiv.org/pdf/2605.10875v1)

**作者:** Yash Akhauri `[一作]` (Cornell University), Mohamed S. Abdelfattah `[通讯]` (Cornell University)

**通讯引用:** 2162 | [OpenAlex ID](https://openalex.org/A5000815783)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出自适应预算控制的语言模型，学习在每一步生成时动态分配计算资源。

**💡 创新点**

创新在于将多种压缩手段（token稀疏、MLP通道裁剪、激活量化）集成到离散动作空间，并通过政策网络学习在不同难度下最优的计算分配。

**🔧 技术方法**

使用冻结的LLM与轻量级自回归Transformer策略网络，通过GRPO与教师强迫的反事实轨迹进行训练，结合Quest、TEAL和ZeroQuant等压缩技术。

**📊 数据集**

主要在Llama-3.1‑8B-Instruct、Llama‑3.1‑3B、Llama‑3.1‑8B、MMLU等公开数据集上评估。

**📈 对比分析**

与静态压缩策略和随机搜索比较，SOL在相同净保留率下实现更低的困惑度和更高的MMLU准确率，最高可提升7.3%，在多模型规模上均保持优势。

**⚠️ 局限性**

局限性包括对计算量代理的依赖、离线训练难以直接映射到硬件延迟/能耗、以及对预算目标的敏感性和潜在的KV污染累积风险。

---

## 870. Closer in the Gap: Towards Portable Performance on RISC-V Vector Processors

**arXiv ID:** 2605.10860 | [PDF](https://arxiv.org/pdf/2605.10860v1)

**作者:** Ruimin Shi `[一作]` (KTH Royal Institute of Technology), Ivy Peng `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1264 | [OpenAlex ID](https://openalex.org/A5037069204)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在RISC‑V Vector Extension (RVV) 1.0平台上，作者设计了手写汇编微基准来测定指令吞吐上限并校准硬件计数器，随后对GCC 15和LLVM 21两版编译器在六个HPC/ML代理应用以及Google的Qsim量子模拟器中的自动向量化效果进行了系统评估。

**💡 创新点**

首次通过精确的汇编基准验证性能计数器的可靠性，并揭示了predication掩码和stride load对RVV性能的显著影响；同时给出了对LMUL参数的敏感性分析和对编译器自动向量化成熟度的定量评估。

**🔧 技术方法**

使用手写RVV汇编基准、Linux perf计数器、LLVM/Clang和GCC编译器的自动向量化功能，以及对Qsim的RVV intrinsic实现；实验平台为两套RVV 1.0硬件（Jupiter、BPI‑F3）。

**📊 数据集**

代理数据集包括：Stream、SpMV、SGEMM、DGEMM、YOLOv3、AlexNet；真实应用为Google Qsim量子模拟器（FP32大规模全态向量）。

**📈 对比分析**

通过对比非向量化GCC 15基准的速度提升、指令数减少率以及各计数器占比，评估了编译器的向量化效果。结果显示：GCC 15在大多数工作负载上实现1.8–2.0×加速，Clang 21在SGEMM/DGEMM上略优；内存受限的Stream/SpMV在两编译器上几乎无加速，且Clang 21在某些场景下甚至降低性能。LMUL调优表明默认值已接近最优，过大LMUL会导致寄存器压力过高而性能下降。

**⚠️ 局限性**

主要限制包括：RVV计数器实现不成熟，部分事件无法可靠校准；编译器自动向量化对非规则内存访问（如Qsim的交错布局）效果差；掩码操作的吞吐低于预期；实验仅覆盖两套硬件，缺乏更广泛的跨平台验证。

---

## 871. ELF: Embedded Language Flows

**arXiv ID:** 2605.10938 | [PDF](https://arxiv.org/pdf/2605.10938v1)

**作者:** Keya Hu `[一作]` (Massachusetts Institute Of Technology), Kaiming He `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Embedded Language Flows（ELF），一种在连续嵌入空间中采用 Flow Matching 的扩散语言模型，能够在几步采样内生成高质量文本；

**💡 创新点**

创新点在于：①把语言建模完全置于连续空间，直到最后一步才离散化，避免每步的离散化损失；②使用共享权重的解码器（无额外 decoder）实现离散化；③将图像扩散中的 classifier‑free guidance（CFG）无缝迁移至文本生成；

**🔧 技术方法**

核心技术包括：连续时间 Flow Matching、预训练 T5 编码器生成上下文嵌入、MSE 与 CE 损失结合、Self‑conditioning 与 CFG 的训练时实现、ODE 与 SDE 两种采样器；

**📊 数据集**

实验数据集：OpenWebText（OWT）用于无条件生成；WMT14 德语‑英语（De‑En）用于翻译；XSum（英文摘要）用于摘要；

**📈 对比分析**

与主流离散扩散模型（MDLM、Duo）及连续扩散模型（FLM、LangFlow）在相同规模下比较，ELF-B 在 OWT 上实现 Gen. PPL 24（仅 32 步）并使用 45B 训练 token，明显优于 170M 模型的 24‑PPL/512 步；在翻译与摘要任务中 BLEU/ROUGE 也均跑在竞争对手之上；

**⚠️ 局限性**

局限性包括：尚未在更大规模模型（>1B 参数）或更广泛的多语言、跨模态任务上验证；采样效率虽提升，但对极低步数（≤16 步）仍有性能下降；模型对训练 token 的敏感度较高，需要较长序列和高质量预训练嵌入。

---

## 872. DECO: Sparse Mixture-of-Experts with Dense-Comparable Performance on End-Side Devices

**arXiv ID:** 2605.10933 | [PDF](https://arxiv.org/pdf/2605.10933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 873. Evaluating the False Trust engendered by LLM Explanations

**arXiv ID:** 2605.10930 | [PDF](https://arxiv.org/pdf/2605.10930v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 874. HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models

**arXiv ID:** 2605.10942 | [PDF](https://arxiv.org/pdf/2605.10942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 875. Chasing Small Sets Optimally Against Adaptive Adversaries

**arXiv ID:** 2605.10927 | [PDF](https://arxiv.org/pdf/2605.10927v1)

**作者:** Christian Coester `[一作]` (University of Oxford), Alexa Tudose `[通讯]` (University of Oxford)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对在度数≤k的层图遍历（等价于有限集合追踪）提出了一种确定性在线算法，证明其竞争比为O(2^k)，并给出了递归形式的下界D_k，证明该下界在k=3时可达，同时推导了分布式异步树探索和k-出租车问题的改进上界与下界。

**💡 创新点**

①首次关闭了30年间竞争比的Ω(2^k)与O(k2^k)间隙，给出了匹配上下界的O(2^k)算法；②提出新的递归下界D_k，推测其为最优；③通过“遗忘”和“失衡”两种新技术改进潜在函数，克服了删除操作导致潜在函数下降的问题。

**🔧 技术方法**

基于层图遍历与演化树游戏的等价性；递归构造与潜在函数分析；使用“遗忘”（潜在函数截断）和“失衡”（局部加权放大）技术；极限极端树构造与极端失衡过程；通过多级缩放控制失真因子。

**📊 数据集**

无实验数据集，本文为纯理论分析，未进行实证评测。

**📈 对比分析**

通过理论证明，将竞争比逼近下界Ω(2^k)；在k=3时得到与下界相匹配的D_3算法；对分布式树探索得到2n+O(k2^kD)步上界，对k-出租车得到下界D_k；相较于先前的O(9^k)或O(k)等界显著提升。

**⚠️ 局限性**

主要局限在于：算法仍为理论构造，实际实现和时间复杂度未给出；对随机化算法在自适应对手下仅提供常数因子提升，尚未得到真正的最优随机算法；极端树构造和失真因子控制依赖递归层级，可能在更一般设定下难以维护；

---

## 876. Variational Inference for Lévy Process-Driven SDEs via Neural Tilting

**arXiv ID:** 2605.10934 | [PDF](https://arxiv.org/pdf/2605.10934v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 877. Personal Visual Context Learning in Large Multimodal Models

**arXiv ID:** 2605.10936 | [PDF](https://arxiv.org/pdf/2605.10936v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 878. Power Reinforcement Post-Training of Text-to-Image Models with Super-Linear Advantage Shaping

**arXiv ID:** 2605.10937 | [PDF](https://arxiv.org/pdf/2605.10937v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 879. Average-Case Hardness of Binary-Encoded Clique in Proof and Communication Complexity

**arXiv ID:** 2605.10941 | [PDF](https://arxiv.org/pdf/2605.10941v1)

**作者:** Susanna F. de Rezende `[一作]` (Lund University), Artur Riazanov `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5012758305)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究二进制编码的团（Clique）问题在平均情况（随机稠密图）下的难度，证明了该公式在剪切平面（Cutting Planes）与有限深度的模2分辨率（Bounded‑Depth Resolution over Parities）中需要指数长度的反证；并给出了求解这些公式中矛盾子句（falsified clause）的随机通信复杂度的下界。

**💡 创新点**

创新点在于：①将弱二进制鸽笼原理（BPHP）的技术迁移到二进制团公式，首次在剪切平面中获得指数下界；②提出基于三角形-DAG与仿射-DAG的“瓶颈计数”与“随机游走”框架，用以控制反证中的节点分布；③通过“lift‑to‑communication”与子立方子协议的技术，得到随机通信下的多项式下界。

**🔧 技术方法**

核心技术包括：
- 组合概率方法（Chernoff、Markov）用于证明随机图满足稠密邻域与公共邻域的性质；
- 三角形‑DAG 与仿射‑DAG 的构造与分析，用以将证明长度与通信成本联系；
- 随机游走与闭包（closure）概念，控制线性系统的秩；
- 量子化（lifting）与子立方子协议（sub‑cube‑like protocols）提升通信复杂度下界；
- 解析层次与随机赋值（random restriction）在剪切平面与模2分辨率中的应用。

**📊 数据集**

实验数据集：使用随机稠密图模型（p‑biased 分布 (n,p,k)）及其 k‑分区版本（n,p,k），其中 p 取接近团出现阈值的值，以满足图无 k‑团的概率接近 1。

**📈 对比分析**

对比方法：
- 对于剪切平面和模2分辨率，与已知的弱 BPHP 的指数下界保持一致，证明了二进制团公式在这些系统中同样困难；
- 对于随机通信，给出了与已知的随机化通信下界（如 O(log n) 的上界）形成对照，证明在稠密图下所需通信量至少为 Ω(min(1/√(1‑p), n^{1/4}/log(nk)))，即多项式级别；
- 由于缺乏实验评测，本文主要通过理论证明与已知结果进行比较，显示在平均情况下保持或提高了现有下界。

**⚠️ 局限性**

局限性：
- 下界仅适用于稠密随机图；在稀疏图或特殊结构图上尚不适用；
- 对 k 的取值范围有限制，尤其在 k 非极大时的最优性尚未确定；
- 对剪切平面中的树形（treelike）形式，虽然给出了下界，但与最优线性下界仍有距离；
- 结果基于平均情况，无法直接推广到最坏情况；
- 部分技术（如闭包、随机游走）对图的随机性质高度依赖，若改为其他分布可能失效。

---

